"""
OCR Text Translator
A computer vision application that extracts text from images/videos and translates it.

Video pipeline:
  Video -> Extract frames -> Detect text + bounding boxes (OCR) -> Expand bounding boxes
  -> Create mask (expanded) -> Inpaint to remove original text -> Track text regions across frames
  -> Render translated text inside same bounding box -> Rebuild video (FFmpeg rawvideo decode/encode
     when ffmpeg is available; set VIDEO_USE_OPENCV_VIDEO_IO=1 to force OpenCV VideoCapture/Writer).

- All detected text regions are fully removed: OCR boxes are expanded with padding so the
  inpainting mask covers every pixel of the original text.
- A single proper mask is built from expanded polygons and cv2.inpaint is used (no simple fill/blur).
- The same text regions are applied across nearby frames: with tracking (OpenCV trackers), regions
  are followed so the mask and translation stay consistent even when OCR does not detect text in
  every frame. Without tracking, the nearest sampled frame's regions are used for every frame.
- Rendering: translated text is drawn inside the same (tracked or sampled) box with dynamic font
  size and multi-line wrap so it never overflows.
"""

import cv2
import easyocr
import importlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional, Any, Dict


def _pil_truetype(path: str, size: int) -> ImageFont.FreeTypeFont:
    """
    Load a TTF/OTF with the BASIC text layout engine.

    When libraqm is available, Pillow defaults to RAQM. Combined with
    pre-shaped Arabic (arabic_reshaper + bidi) and draw.text(anchor=...),
    some Pillow builds raise inside getmask2 (e.g. TypeError) and the
    translated-photo path silently skips drawing. BASIC matches our
    already-visual glyph order and avoids that failure mode.
    """
    return ImageFont.truetype(
        path, size, layout_engine=ImageFont.Layout.BASIC
    )
import os
import re
import copy
import json
import threading
import atexit
import sys
import shutil
import subprocess
import tempfile
import statistics
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

from ocr_dncnn_torch import dncnn_denoise_bgr_maybe

_OCR_PREPROCESS_PROFILE: Dict[str, Any] = {}
_OCR_PREPROCESS_PREVIEW: Dict[str, Any] = {}


def last_preprocess_preview() -> Dict[str, Any]:
    """BGR frames from the most recent ``enhance_bgr_for_ocr`` (for Streamlit before/after UI)."""
    p = _OCR_PREPROCESS_PREVIEW
    out: Dict[str, Any] = {}
    for k in ("before_bgr", "after_openai_bgr", "after_final_bgr"):
        v = p.get(k)
        if v is not None and getattr(v, "size", 0):
            out[k] = v
    out["openai_used"] = bool(p.get("openai_used"))
    return out


def _clear_preprocess_preview() -> None:
    global _OCR_PREPROCESS_PREVIEW
    _OCR_PREPROCESS_PREVIEW = {}


def _safe_cv2_resize(
    bgr: np.ndarray,
    width: int,
    height: int,
    *,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resize with guards against zero output size (OpenCV ``inv_scale_x > 0``)."""
    if bgr is None or bgr.size == 0:
        return bgr
    ih, iw = bgr.shape[:2]
    if iw <= 0 or ih <= 0:
        return bgr
    nw = max(1, int(width))
    nh = max(1, int(height))
    if nw == iw and nh == ih:
        return bgr
    return cv2.resize(bgr, (nw, nh), interpolation=interpolation)


def _resize_bgr_to_max_dim(bgr: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale so longest side <= ``max_dim``. ``max_dim <= 0`` = keep original size."""
    if max_dim <= 0 or bgr is None or bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    if max(h, w) <= max_dim:
        return bgr
    scale = max_dim / float(max(h, w))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return _safe_cv2_resize(bgr, nw, nh)


def _commit_preprocess_preview(
    before_bgr: Optional[np.ndarray],
    after_openai_bgr: Optional[np.ndarray],
    after_final_bgr: Optional[np.ndarray],
    *,
    openai_used: bool = False,
) -> None:
    global _OCR_PREPROCESS_PREVIEW
    _OCR_PREPROCESS_PREVIEW = {
        "before_bgr": before_bgr.copy() if before_bgr is not None and before_bgr.size else None,
        "after_openai_bgr": (
            after_openai_bgr.copy() if after_openai_bgr is not None and after_openai_bgr.size else None
        ),
        "after_final_bgr": (
            after_final_bgr.copy() if after_final_bgr is not None and after_final_bgr.size else None
        ),
        "openai_used": bool(openai_used),
    }


def _ocr_env_float(name: str, default: float) -> float:
    try:
        return float((os.environ.get(name) or "").strip() or default)
    except ValueError:
        return default


def _ocr_env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _denoise_bgr_colored(bgr: np.ndarray, *, gentle: bool) -> np.ndarray:
    """Colored non-local means denoise (CPU). ``gentle`` uses lighter smoothing for fine strokes."""
    h = 4 if gentle else 7
    hc = h
    tw, sw = (5, 15) if gentle else (7, 21)
    return cv2.fastNlMeansDenoisingColored(bgr, None, h, hc, tw, sw)


def _unsharp_bgr(bgr: np.ndarray, amount: float) -> np.ndarray:
    """Gaussian unsharp: ``amount`` in ~0.2–0.8 is usually safe after denoise."""
    if amount <= 1e-6:
        return bgr
    amt = max(0.0, min(1.25, float(amount)))
    sigma = max(0.5, min(2.0, _ocr_env_float("OCR_PREPROCESS_UNSHARP_SIGMA", 1.0)))
    blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(bgr, 1.0 + amt, blur, -amt, 0)


def _brighten_lab_l(bgr: np.ndarray, factor: float) -> np.ndarray:
    """Lift dark regions in LAB L (``factor`` > 1.0). Favors shadows via sqrt curve."""
    if factor <= 1.0 + 1e-6:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    lf = l_ch.astype(np.float32)
    # More lift where L is low (shadows / dark menus)
    t = 1.0 - lf / 255.0
    lift = np.power(float(factor), np.sqrt(t))
    l2 = np.clip(lf * lift, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l2, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def _bilateral_bgr(bgr: np.ndarray) -> np.ndarray:
    """Edge-preserving smooth — reduces speckle before median / CLAHE / unsharp."""
    if not _ocr_env_truthy("OCR_PREPROCESS_BILATERAL", True):
        return bgr
    try:
        d = int(os.environ.get("OCR_PREPROCESS_BILATERAL_D", "5"))
    except ValueError:
        d = 5
    d = max(3, min(9, d | 1))
    sc0 = max(10.0, min(90.0, _ocr_env_float("OCR_PREPROCESS_BILATERAL_SIGMA_COLOR", 24.0)))
    ss0 = max(10.0, min(90.0, _ocr_env_float("OCR_PREPROCESS_BILATERAL_SIGMA_SPACE", 24.0)))
    try:
        passes = int(os.environ.get("OCR_PREPROCESS_BILATERAL_PASSES", "1"))
    except ValueError:
        passes = 1
    passes = max(1, min(2, passes))
    img = bgr
    for p in range(passes):
        sc = sc0 * (0.88**p)
        ss = ss0 * (0.88**p)
        img = cv2.bilateralFilter(img, d, sc, ss)
    return img


def _median_l_channel_bgr(bgr: np.ndarray) -> np.ndarray:
    """3×3 median on LAB L only — cuts salt-and-pepper without shifting hue like RGB median."""
    if not _ocr_env_truthy("OCR_PREPROCESS_MEDIAN_L", False):
        return bgr
    try:
        k = int(os.environ.get("OCR_PREPROCESS_MEDIAN_L_K", "3"))
    except ValueError:
        k = 3
    if k % 2 == 0:
        k += 1
    k = max(3, min(5, k))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l2 = cv2.medianBlur(l_ch, k)
    return cv2.cvtColor(cv2.merge([l2, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def _luma_percentile_stretch_bgr(bgr: np.ndarray, prof: Dict[str, Any]) -> np.ndarray:
    """Mild LAB-L percentile stretch: clearer text/background separation without CLAHE grain."""
    if not _ocr_env_truthy("OCR_PREPROCESS_LUMA_STRETCH", False):
        prof["luma_stretch"] = False
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    lf = l_ch.astype(np.float32)
    lo_p = max(0.5, min(8.0, _ocr_env_float("OCR_PREPROCESS_LUMA_STRETCH_LOW_PCT", 2.0)))
    hi_p = max(92.0, min(99.7, _ocr_env_float("OCR_PREPROCESS_LUMA_STRETCH_HIGH_PCT", 98.5)))
    lo, hi = np.percentile(lf, [lo_p, hi_p])
    span = float(hi - lo)
    prof["luma_stretch_span"] = round(span, 3)
    if span < 18.0:
        prof["luma_stretch"] = False
        prof["luma_stretch_skip"] = "low_span"
        return bgr
    stretched = np.clip((lf - lo) * (255.0 / max(span, 1.0)), 0.0, 255.0)
    blend = max(0.0, min(0.85, _ocr_env_float("OCR_PREPROCESS_LUMA_STRETCH_BLEND", 0.42)))
    # Very busy textured shots need less global contrast, otherwise paper grain competes with text.
    std = _gray_noise_std(bgr)
    if std > 28.0:
        blend *= 0.55
    elif std > 20.0:
        blend *= 0.75
    if blend <= 1e-5:
        prof["luma_stretch"] = False
        return bgr
    l2 = np.clip(lf * (1.0 - blend) + stretched * blend, 0.0, 255.0).astype(np.uint8)
    prof["luma_stretch"] = True
    prof["luma_stretch_blend"] = round(blend, 3)
    return cv2.cvtColor(cv2.merge([l2, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def _clahe_lab_bgr(bgr: np.ndarray) -> np.ndarray:
    """CLAHE on L channel — can amplify grain on noisy / textured paper; often best left off."""
    if not _ocr_env_truthy("OCR_PREPROCESS_CLAHE", False):
        return bgr
    clip = max(1.0, min(4.0, _ocr_env_float("OCR_PREPROCESS_CLAHE_CLIP", 1.8)))
    tw = int(max(4, min(16, _ocr_env_float("OCR_PREPROCESS_CLAHE_TILE", 8.0))))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tw, tw))
    l2 = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l2, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def _gray_noise_std(bgr: np.ndarray) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.std(g))


def _unsharp_bgr_adaptive(bgr: np.ndarray, amount: float, prof: Dict[str, Any]) -> np.ndarray:
    """Reduce or disable unsharp when the frame is still grainy (CLAHE-like halos on texture)."""
    if amount <= 1e-6:
        return bgr
    if not _ocr_env_truthy("OCR_PREPROCESS_ADAPTIVE_UNSHARP", True):
        return _unsharp_bgr(bgr, amount)
    std = _gray_noise_std(bgr)
    prof["pre_unsharp_gray_std"] = std
    # Phone noise / parchment texture often std 14–35+. Keep a small edge-restoration floor
    # so denoise/resize does not erase tiny OCR strokes.
    if std < 7.0:
        scale = 1.0
    elif std < 9.5:
        scale = 0.5
    elif std < 12.5:
        scale = 0.28
    elif std < 16.0:
        scale = 0.12
    elif std < 20.0:
        scale = 0.04
    else:
        scale = 0.0
    if _ocr_env_truthy("OCR_PREPROCESS_EDGE_RESTORE", True):
        scale = max(scale, _ocr_env_float("OCR_PREPROCESS_EDGE_RESTORE_FLOOR", 0.16))
    prof["unsharp_adaptive_scale"] = scale
    if scale <= 0.0:
        return bgr
    return _unsharp_bgr(bgr, amount * scale)


def last_preprocess_profile() -> Dict[str, Any]:
    return dict(_OCR_PREPROCESS_PROFILE)


def _ocr_preprocess_engine() -> str:
    # Prefer CUDA tensor pipeline whenever CUDA is available (falls back to legacy if unavailable).
    return (os.environ.get("OCR_PREPROCESS_ENGINE") or "gpu").strip().lower()


def _enhance_bgr_for_ocr_legacy(
    bgr: np.ndarray,
    max_dim: int = 1200,
    *,
    thin_strokes: bool = False,
    force_cpu: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    CPU-oriented stack (OpenCV + optional DnCNN numpy path): resize, brighten, DnCNN / NL-means,
    bilateral, median-L, optional CLAHE, adaptive unsharp.

    DnCNN: ``OCR_PREPROCESS_DNN_DEVICE``, ``OCR_PREPROCESS_DNN_WEIGHTS``, ``OCR_PREPROCESS_DNN_REPO``, ``OCR_PREPROCESS_DNN_FILE``.
    """
    global _OCR_PREPROCESS_PROFILE
    t0 = time.perf_counter()
    prof: Dict[str, Any] = {
        "backend": "resize_cpu",
        "thin_strokes": thin_strokes,
        "force_cpu": force_cpu,
    }
    if bgr is None or bgr.size == 0:
        prof["error"] = "empty"
        _OCR_PREPROCESS_PROFILE = prof
        return bgr, prof
    img = np.ascontiguousarray(bgr)
    h, w = img.shape[:2]
    prof["shape_in"] = (h, w)
    lim = int(max_dim) if max_dim else 0
    if lim > 0 and max(h, w) > lim:
        scale = lim / float(max(h, w))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        img = _safe_cv2_resize(img, nw, nh)
    prof["shape_after_resize"] = img.shape[:2]

    bright = _ocr_env_float("OCR_PREPROCESS_BRIGHTEN", 1.0)
    prof["brighten"] = bright
    if bright > 1.0:
        img = _brighten_lab_l(img, bright)

    dnn_used = False
    img, dinfo = dncnn_denoise_bgr_maybe(img, force_cpu=force_cpu)
    for dk, dv in dinfo.items():
        prof[str(dk)] = dv
    dnn_used = bool(dinfo.get("dnn_used"))

    denoise_on = _ocr_env_truthy("OCR_PREPROCESS_DENOISE", False) and not dnn_used
    prof["denoise"] = denoise_on
    prof["dnn_skipped_classical_denoise"] = dnn_used

    if denoise_on:
        gentle = bool(thin_strokes) or _ocr_env_truthy("OCR_PREPROCESS_DENOISE_GENTLE", False)
        img = _denoise_bgr_colored(img, gentle=gentle)
        prof["denoise_gentle"] = gentle

    bilat_on = _ocr_env_truthy("OCR_PREPROCESS_BILATERAL", True)
    prof["bilateral"] = bilat_on
    if bilat_on:
        img = _bilateral_bgr(img)

    med_on = _ocr_env_truthy("OCR_PREPROCESS_MEDIAN_L", True)
    prof["median_l"] = med_on
    if med_on:
        img = _median_l_channel_bgr(img)

    img = _luma_percentile_stretch_bgr(img, prof)

    clahe_wanted = _ocr_env_truthy("OCR_PREPROCESS_CLAHE", False)
    clahe_auto = _ocr_env_truthy("OCR_PREPROCESS_CLAHE_AUTO", True)
    std_for_clahe = _gray_noise_std(img)
    prof["pre_clahe_gray_std"] = std_for_clahe
    clahe_max = _ocr_env_float("OCR_PREPROCESS_CLAHE_MAX_STD", 10.5)
    clahe_ran = False
    if clahe_wanted:
        if clahe_auto and std_for_clahe > clahe_max:
            prof["clahe"] = False
            prof["clahe_skipped_noisy"] = True
        else:
            img = _clahe_lab_bgr(img)
            clahe_ran = True
            prof["clahe"] = True
    else:
        prof["clahe"] = False

    unsharp_amt = _ocr_env_float("OCR_PREPROCESS_UNSHARP", 0.34)
    prof["unsharp"] = unsharp_amt
    if unsharp_amt > 0:
        img = _unsharp_bgr_adaptive(img, unsharp_amt, prof)

    tags = ["resize"]
    if bright > 1.0:
        tags.append("brighten")
    if dnn_used:
        tags.append("dncnn")
    if denoise_on:
        tags.append("nlmeans")
    if bilat_on:
        tags.append("bilateral")
    if med_on:
        tags.append("median_l")
    if prof.get("luma_stretch"):
        tags.append("luma_stretch")
    if clahe_ran:
        tags.append("clahe")
    if unsharp_amt > 0:
        tags.append("unsharp")
    prof["backend"] = "+".join(tags)

    prof["shape_out"] = img.shape[:2]
    prof["total_s"] = time.perf_counter() - t0
    _OCR_PREPROCESS_PROFILE = prof
    return img, prof


def enhance_bgr_for_ocr(
    bgr: np.ndarray,
    max_dim: int = 1200,
    *,
    thin_strokes: bool = False,
    force_cpu: bool = False,
    openai_preprocess: Optional[bool] = None,
    local_preprocess: Optional[bool] = None,
    openai_api_key: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Image preprocessing for OCR.

    **OpenAI** (``OCR_PREPROCESS_OPENAI=1`` or ``openai_preprocess=True``): optional
    ``images.edit`` pass via ``openai_ocr_preprocess`` before local processing. On success,
    local denoise is skipped when ``OCR_PREPROCESS_OPENAI_ONLY=1`` (default).

    **Engine** (``OCR_PREPROCESS_ENGINE``): default ``gpu`` runs the CUDA pipeline in
    ``ocr_preprocess_pipeline`` when CUDA is available; falls back to legacy on failure or if
    ``force_cpu`` is passed unless ``OCR_PREPROCESS_FORCE_GPU=1`` (then preprocess still uses CUDA).
    ``auto`` behaves like ``gpu``. ``cpu`` / ``legacy`` / ``opencv`` force the legacy path only.

    GPU path: single H2D, tensor resize, adaptive illumination / gamma / contrast, DnCNN on tensor
    (no extra numpy hop), optional ``torch.jit`` denoiser via ``OCR_PREPROCESS_RIDNET_JIT``,
    edge-preserving unsharp, one D2H; optional perspective (``OCR_PREPROCESS_PERSPECTIVE``) and
    post CLAHE/median on CPU when enabled.

    Legacy path: resize, LAB brighten, optional DnCNN (numpy round-trip), NL-means, bilateral,
    median-L, optional CLAHE, adaptive unsharp — see ``_enhance_bgr_for_ocr_legacy``.
    """
    preview_before = bgr.copy() if bgr is not None and bgr.size else None
    if preview_before is not None and max_dim > 0:
        preview_before = _resize_bgr_to_max_dim(preview_before, max_dim)
    preview_after_oai: Optional[np.ndarray] = None
    oai_key = (openai_api_key or os.environ.get("OPENAI_API_KEY") or "").strip() or None
    oai_sidecar: Dict[str, Any] = {}

    use_openai = (
        openai_preprocess
        if openai_preprocess is not None
        else _ocr_env_truthy("OCR_PREPROCESS_OPENAI", False)
    )
    if use_openai:
        try:
            from openai_ocr_preprocess import openai_preprocess_bgr_for_ocr

            oai_prof: Dict[str, Any] = {}
            bgr_oai, oai_prof = openai_preprocess_bgr_for_ocr(
                bgr,
                max_dim=max_dim,
                api_key=oai_key,
                perf_stats=oai_prof,
            )
            oai_sidecar.update(oai_prof)
            if oai_prof.get("openai_used"):
                preview_after_oai = bgr_oai.copy()
                if _ocr_env_truthy("OCR_PREPROCESS_OPENAI_ONLY", True):
                    oai_prof["backend"] = "openai+resize"
                    _commit_preprocess_preview(
                        preview_before, preview_after_oai, bgr_oai, openai_used=True
                    )
                    return bgr_oai, oai_prof
                bgr = bgr_oai
        except Exception as e:
            oai_sidecar["openai_skip"] = repr(e)

    do_local = local_preprocess if local_preprocess is not None else True
    if not do_local:
        out = _resize_bgr_to_max_dim(bgr, max_dim)
        prof = {
            "backend": "resize_only",
            "openai_only_local_off": True,
            "shape_out": out.shape[:2] if out is not None else None,
        }
        if preview_after_oai is not None:
            prof["openai_used"] = True
        prof.update(oai_sidecar)
        _commit_preprocess_preview(
            preview_before,
            preview_after_oai,
            out,
            openai_used=preview_after_oai is not None,
        )
        return out, prof

    eng = _ocr_preprocess_engine()
    gpu_prof: Dict[str, Any] = {}
    # Allow GPU preprocess even when OCRTranslator passed force_cpu (e.g. after OCR OOM).
    preprocess_gpu = _ocr_env_truthy("OCR_PREPROCESS_FORCE_GPU", False)
    preprocess_force_cpu = bool(force_cpu) and not preprocess_gpu
    if eng not in ("cpu", "legacy", "opencv"):
        try:
            from ocr_preprocess_pipeline import try_preprocess_gpu

            out, gpu_prof = try_preprocess_gpu(
                bgr,
                max_dim,
                thin_strokes=thin_strokes,
                force_cpu=preprocess_force_cpu,
            )
            if out is not None:
                if preprocess_gpu and force_cpu:
                    gpu_prof["preprocess_force_gpu"] = True
                _commit_preprocess_preview(
                    preview_before,
                    preview_after_oai,
                    out,
                    openai_used=preview_after_oai is not None,
                )
                gpu_prof.update(oai_sidecar)
                return out, gpu_prof
        except Exception as e:
            gpu_prof = {"gpu_error": repr(e), "stages": []}

    out, prof = _enhance_bgr_for_ocr_legacy(
        bgr, max_dim, thin_strokes=thin_strokes, force_cpu=force_cpu
    )
    if gpu_prof:
        prof["gpu_pipeline_attempt"] = {
            k: gpu_prof[k]
            for k in ("gpu_skip", "gpu_error", "backend", "stages")
            if k in gpu_prof and gpu_prof[k] not in (None, "", [], {})
        }
    prof.update(oai_sidecar)
    _commit_preprocess_preview(
        preview_before,
        preview_after_oai,
        out,
        openai_used=preview_after_oai is not None,
    )
    return out, prof


def enhance_bgr_video_frame(
    bgr: np.ndarray,
    *,
    thin_strokes: bool = False,
    force_cpu: bool = False,
) -> np.ndarray:
    """Video path: cap long side via ``OCR_VIDEO_PREPROCESS_MAX_SIDE`` (default 1920)."""
    try:
        cap = int(os.environ.get("OCR_VIDEO_PREPROCESS_MAX_SIDE", "1920"))
    except ValueError:
        cap = 1920
    cap = max(640, min(3840, cap))
    out, _ = enhance_bgr_for_ocr(
        bgr, max_dim=cap, thin_strokes=thin_strokes, force_cpu=force_cpu
    )
    return out


def _ffmpeg_executable() -> Optional[str]:
    """Path to ffmpeg: bundled via imageio-ffmpeg, or system PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg")


def _transcode_mp4_to_h264_for_browser(src_path: str, dst_path: str) -> bool:
    """
    OpenCV 'mp4v' files often do not play in HTML5 video; Windows OpenCV may break on OpenH264.
    Re-encode to H.264 yuv420p + faststart so Streamlit/browsers can play the file.
    """
    exe = _ffmpeg_executable()
    if not exe or not os.path.isfile(src_path):
        return False
    preset = (os.environ.get("VIDEO_EXPORT_PRESET", "veryfast") or "veryfast").strip().lower()
    allowed_presets = {
        "ultrafast", "superfast", "veryfast", "faster", "fast",
        "medium", "slow", "slower", "veryslow",
    }
    if preset not in allowed_presets:
        preset = "veryfast"
    try:
        crf = int(os.environ.get("VIDEO_EXPORT_CRF", "23"))
    except ValueError:
        crf = 23
    crf = max(14, min(35, crf))
    cmd = [
        exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        src_path,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        dst_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=7200)
        return r.returncode == 0 and os.path.isfile(dst_path) and os.path.getsize(dst_path) > 0
    except (subprocess.SubprocessError, OSError):
        return False


def _finalize_mp4_after_opencv(encode_path: str, final_path: str) -> None:
    """Turn OpenCV mp4v output into browser-playable H.264, or move intermediate file to final_path."""
    if not final_path.lower().endswith(".mp4"):
        return
    same = os.path.abspath(encode_path) == os.path.abspath(final_path)
    if same:
        out_dir = os.path.dirname(os.path.abspath(final_path))
        try:
            fd, tmp_h264 = tempfile.mkstemp(suffix=".h264.mp4", dir=out_dir if out_dir else None)
            os.close(fd)
        except OSError:
            return
        if _transcode_mp4_to_h264_for_browser(encode_path, tmp_h264):
            try:
                os.replace(tmp_h264, final_path)
            except OSError:
                shutil.move(tmp_h264, final_path)
        else:
            try:
                os.unlink(tmp_h264)
            except OSError:
                pass
        return
    if _transcode_mp4_to_h264_for_browser(encode_path, final_path):
        try:
            os.unlink(encode_path)
        except OSError:
            pass
        return
    try:
        if os.path.isfile(final_path):
            os.unlink(final_path)
    except OSError:
        pass
    try:
        os.replace(encode_path, final_path)
    except OSError:
        shutil.move(encode_path, final_path)


def _use_ffmpeg_video_io() -> bool:
    """Decode/encode translated video with FFmpeg (usually faster than OpenCV VideoCapture/VideoWriter)."""
    if (os.environ.get("VIDEO_USE_OPENCV_VIDEO_IO") or "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    return bool(_ffmpeg_executable())


def _ffprobe_executable() -> Optional[str]:
    ff = _ffmpeg_executable()
    if ff:
        d = os.path.dirname(ff)
        try:
            for name in sorted(os.listdir(d)):
                low = name.lower()
                if low.startswith("ffprobe"):
                    p = os.path.join(d, name)
                    if os.path.isfile(p):
                        return p
        except OSError:
            pass
    return shutil.which("ffprobe")


def _parse_fraction_fps(s: Optional[str]) -> Optional[float]:
    t = (s or "").strip()
    if not t or t == "0/0":
        return None
    if "/" in t:
        a, _, b = t.partition("/")
        try:
            aa, bb = float(a), float(b)
            return aa / bb if bb else None
        except ValueError:
            return None
    try:
        return float(t)
    except ValueError:
        return None


def _probe_video_meta_ffprobe(path: str) -> Optional[Tuple[int, int, float]]:
    exe = _ffprobe_executable()
    if not exe:
        return None
    cmd = [
        exe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout or "{}")
        streams = data.get("streams") or []
        if not streams:
            return None
        st = streams[0]
        w = int(st.get("width") or 0)
        h = int(st.get("height") or 0)
        if w <= 0 or h <= 0:
            return None
        fps = _parse_fraction_fps(st.get("avg_frame_rate")) or _parse_fraction_fps(st.get("r_frame_rate"))
        if not fps or fps <= 0:
            fps = 25.0
        return (w, h, float(fps))
    except Exception:
        return None


def _probe_video_meta_opencv(path: str) -> Optional[Tuple[int, int, float]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if w <= 0 or h <= 0:
            return None
        return (w, h, fps)
    finally:
        cap.release()


def _ffmpeg_h264_nvenc_available() -> bool:
    if (os.environ.get("VIDEO_RENDER_NVENC") or "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    exe = _ffmpeg_executable()
    if not exe:
        return False
    try:
        r = subprocess.run(
            [exe, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=25,
        )
        blob = (r.stderr or "") + (r.stdout or "")
        return "h264_nvenc" in blob
    except (subprocess.SubprocessError, OSError):
        return False


def _ffmpeg_decode_raw_bgr_start(video_path: str, out_w: int, out_h: int) -> subprocess.Popen:
    exe = _ffmpeg_executable()
    if not exe:
        raise ValueError("ffmpeg executable not found")
    ow = max(2, int(out_w) - (int(out_w) % 2))
    oh = max(2, int(out_h) - (int(out_h) % 2))
    vf = f"scale={ow}:{oh}:flags=bilinear"
    pre: List[str] = []
    if (os.environ.get("VIDEO_RENDER_CUDA_DECODE") or "").strip().lower() in ("1", "true", "yes", "on"):
        pre = ["-hwaccel", "cuda"]
    cmd = [
        exe,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        *pre,
        "-i",
        video_path,
        "-an",
        "-sn",
        "-dn",
        "-map",
        "0:v:0",
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        bufsize=0,
    )


def _ffmpeg_read_bgr_frame(proc: subprocess.Popen, w: int, h: int) -> Optional[np.ndarray]:
    """Read one BGR frame; returns None on EOF or truncated tail."""
    assert proc.stdout is not None
    n = w * h * 3
    buf = bytearray()
    while len(buf) < n:
        chunk = proc.stdout.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))


def _ffmpeg_encode_raw_bgr_start(
    output_path: str,
    w: int,
    h: int,
    fps: float,
    *,
    use_avi: bool,
) -> subprocess.Popen:
    exe = _ffmpeg_executable()
    if not exe:
        raise ValueError("ffmpeg executable not found")
    fps_f = float(fps) if fps and fps > 0 else 25.0
    fps_s = f"{fps_f:.6f}".rstrip("0").rstrip(".")
    if fps_s == "":
        fps_s = "25"
    cmd = [
        exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        fps_s,
        "-i",
        "pipe:0",
    ]
    if use_avi:
        cmd.extend(["-c:v", "mjpeg", "-q:v", "5", output_path])
    else:
        try:
            crf = int(os.environ.get("VIDEO_EXPORT_CRF", "23"))
        except ValueError:
            crf = 23
        crf = max(14, min(35, crf))
        use_nvenc = _ffmpeg_h264_nvenc_available()
        if use_nvenc:
            nv_preset = (os.environ.get("VIDEO_RENDER_NVENC_PRESET") or "p2").strip().lower()
            if nv_preset not in ("p1", "p2", "p3", "p4", "p5", "p6", "p7"):
                nv_preset = "p2"
            cq = max(16, min(40, int(os.environ.get("VIDEO_RENDER_NVENC_CQ", str(crf + 4)) or (crf + 4))))
            cmd.extend(
                [
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    nv_preset,
                    "-rc",
                    "vbr",
                    "-cq",
                    str(cq),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-an",
                    output_path,
                ]
            )
        else:
            preset = (os.environ.get("VIDEO_EXPORT_PRESET", "veryfast") or "veryfast").strip().lower()
            allowed_presets = {
                "ultrafast",
                "superfast",
                "veryfast",
                "faster",
                "fast",
                "medium",
                "slow",
                "slower",
                "veryslow",
            }
            if preset not in allowed_presets:
                preset = "veryfast"
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    preset,
                    "-crf",
                    str(crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-an",
                    output_path,
                ]
            )
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )


def _ffmpeg_upscale_video_to_height(src_path: str, dst_path: str, target_h: int) -> bool:
    """Re-encode at taller resolution (second pass). Optional speed/quality tradeoff."""
    exe = _ffmpeg_executable()
    if not exe or target_h <= 0 or not os.path.isfile(src_path):
        return False
    th = max(2, int(target_h) - (int(target_h) % 2))
    cmd = [
        exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        src_path,
        "-vf",
        f"scale=-2:{th}:flags=lanczos",
        "-c:v",
        "libx264",
        "-preset",
        (os.environ.get("VIDEO_RENDER_UPSCALE_PRESET") or "veryfast").strip(),
        "-crf",
        str(max(14, min(28, int(os.environ.get("VIDEO_RENDER_UPSCALE_CRF", "20") or 20)))),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        dst_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=7200)
        return r.returncode == 0 and os.path.isfile(dst_path) and os.path.getsize(dst_path) > 0
    except (subprocess.SubprocessError, OSError):
        return False


def _scale_video_frame_map_geometry(
    frame_map: Dict[int, Tuple[List[Tuple], List[str]]],
    sx: float,
    sy: float,
) -> Dict[int, Tuple[List[Tuple], List[str]]]:
    if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
        return frame_map
    out: Dict[int, Tuple[List[Tuple], List[str]]] = {}
    for fk, (rows_in, txts) in frame_map.items():
        new_rows: List[Tuple] = []
        for row in rows_in:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            bbox = row[1]
            if not isinstance(bbox, (list, tuple)):
                continue
            nb = []
            for p in bbox:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    nb.append([float(p[0]) * sx, float(p[1]) * sy])
            if len(nb) < 3:
                continue
            if len(row) >= 4:
                new_rows.append((row[0], nb, row[2], row[3]))
            elif len(row) == 3:
                new_rows.append((row[0], nb, row[2]))
            else:
                new_rows.append((row[0], nb))
        out[fk] = (new_rows, list(txts))
    return out


def _video_blur_under_polys(frame_bgr: np.ndarray, expanded_list: List[np.ndarray], blur_sigma: float) -> np.ndarray:
    """Replace polygon regions with a strong Gaussian blur (faster than Telea inpainting)."""
    img_h, img_w = frame_bgr.shape[:2]
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for pts in expanded_list or []:
        if pts is not None and len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    if not np.any(mask > 0):
        return frame_bgr.copy()
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return frame_bgr.copy()
    pad = max(8, int(blur_sigma * 2))
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(img_h, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(img_w, int(xs.max()) + 1 + pad)
    roi = frame_bgr[y0:y1, x0:x1].copy()
    roi_mask = mask[y0:y1, x0:x1]
    k = max(15, int(blur_sigma * 4))
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(roi, (k, k), float(blur_sigma))
    out = frame_bgr.copy()
    m = roi_mask > 0
    out[y0:y1, x0:x1][m] = blurred[m]
    return out


def _video_rect_alpha_under_polys(
    frame_bgr: np.ndarray,
    expanded_list: List[np.ndarray],
    alpha: float,
    fill_bgr: Tuple[int, int, int],
) -> np.ndarray:
    """Semi-transparent fill under OCR polygons (cheap alternative to inpainting)."""
    img_h, img_w = frame_bgr.shape[:2]
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    poly_canvas = np.zeros_like(frame_bgr)
    b_, g_, r_ = fill_bgr
    poly_canvas[:] = (b_, g_, r_)
    for pts in expanded_list or []:
        if pts is not None and len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    if not np.any(mask > 0):
        return frame_bgr.copy()
    a = max(0.15, min(0.95, float(alpha)))
    out = frame_bgr.astype(np.float32)
    tint = poly_canvas.astype(np.float32)
    m = mask > 0
    out[m] = out[m] * (1.0 - a) + tint[m] * a
    return out.astype(np.uint8)


# Padding to expand OCR boxes so inpainting covers all text pixels (generous to avoid original showing)
DEFAULT_MASK_PADDING_PX = 30
DEFAULT_MASK_PADDING_RATIO = 0.36

def _ocr_use_gpu() -> bool:
    """Use EasyOCR detection on GPU when CUDA is available.

    Default: prefer GPU if PyTorch sees CUDA (no env var needed).
    Opt out: ``OCR_USE_GPU=0`` / ``false`` / ``no`` (e.g. low VRAM or unstable drivers).
    """
    v = os.environ.get("OCR_USE_GPU", "").strip().lower()
    if v in ("0", "false", "no"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _paddle_ocr_use_gpu() -> bool:
    """
    Use PaddleOCR in the worker on GPU when CUDA is available.
    Does not require OCR_USE_GPU=1 (Paddle runs in a subprocess; GPU is the default when CUDA is seen).
    Opt out: OCR_PADDLE_USE_CPU=1 (or true/yes), or if PyTorch is CPU-only.
    """
    v = (os.environ.get("OCR_PADDLE_USE_CPU") or os.environ.get("OCR_PADDLE_FORCE_CPU") or "").strip().lower()
    if v in ("1", "true", "yes"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _looks_like_quad(points: Any) -> bool:
    """Return True when `points` is a 4-point polygon [[x,y], ...]."""
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        return False
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            return False
        try:
            float(p[0])
            float(p[1])
        except (TypeError, ValueError):
            return False
    return True


_PADDLE_OCR_CLASS = None
_PADDLE_OCR_IMPORT_ERROR: Optional[Exception] = None


def _get_paddle_ocr_class():
    """Lazy-import PaddleOCR to avoid loading native libs unless needed."""
    global _PADDLE_OCR_CLASS, _PADDLE_OCR_IMPORT_ERROR
    if _PADDLE_OCR_CLASS is not None:
        return _PADDLE_OCR_CLASS
    if _PADDLE_OCR_IMPORT_ERROR is not None:
        raise _PADDLE_OCR_IMPORT_ERROR
    try:
        mod = importlib.import_module("paddleocr")
        _Cls = getattr(mod, "PaddleOCR")
        _PADDLE_OCR_CLASS = _Cls
        return _PADDLE_OCR_CLASS
    except Exception as e:
        _PADDLE_OCR_IMPORT_ERROR = e
        raise


def _paddle_langs_from_codes(codes: List[str]) -> List[str]:
    """
    Map app OCR language codes to PaddleOCR ``lang`` values.

    Paddle instantiates **one recognition model per lang**; there is no single multilingual
    ``lang`` string. When the user selects several supported languages (e.g. English + Chinese),
    we run one worker pass per distinct Paddle lang and merge boxes (see ``_run_paddle_ocr_subprocess``).

    Pass order:
    - **ch / japan / korean** first — PP-OCR CJK detects Han reliably; English-only misses most Han.
    - **ar before en** when both are selected — Arabic PP-OCR establishes RTL line boxes first; the English
      pass then adds Latin-only regions so bilingual signage behaves closer to EasyOCR (fewer cross-script merges).

    Duplicates are dropped.
    """
    mapping = {
        "zh": "ch",
        "ja": "japan",
        "ko": "korean",
    }
    supported = {
        "en",
        "ch",
        "japan",
        "korean",
        "ar",
        "ru",
        "hi",
        "fr",
        "german",
        "it",
        "es",
        "pt",
    }
    seen: set = set()
    out: List[str] = []
    for c in codes or ["en"]:
        if not c:
            continue
        p = mapping.get(str(c).strip().lower(), str(c).strip().lower())
        if p not in supported or p in seen:
            continue
        seen.add(p)
        out.append(p)
    out = out if out else ["en"]
    if len(out) <= 1:
        return out

    def _paddle_pass_order(lang: str, orig_idx: int) -> Tuple[int, int]:
        cjk = {"ch": 0, "japan": 1, "korean": 2}.get(lang)
        if cjk is not None:
            return cjk, orig_idx
        if lang == "ar":
            return 30, orig_idx
        if lang == "en":
            return 31, orig_idx
        return 40, orig_idx

    return sorted(out, key=lambda lg: _paddle_pass_order(lg, out.index(lg)))


def _paddle_lang_from_codes(codes: List[str]) -> str:
    """First Paddle lang from ``codes`` (backward compat; daemon uses a single model)."""
    return _paddle_langs_from_codes(codes)[0]


def _paddle_multilang_max_workers(n_langs: int) -> int:
    """
    Worker cap for Paddle subprocesses (one process per language).

    Default is **sequential** (one language at a time) — concurrent multi-language
    Paddle runs trigger duplicate detector work and merge churn that often loses
    small Latin tokens on bilingual signage; serial runs are more stable and
    easier on VRAM.

    Overrides:
    - ``OCR_PADDLE_PARALLEL_WORKERS=N`` (N ≥ 2) opts into parallel and caps
      concurrency at ``min(n_langs, N)``.
    - ``OCR_PADDLE_PARALLEL_OFF=1`` is still honored for backward compatibility
      and forces sequential.
    """
    if n_langs <= 1:
        return 1
    if os.environ.get("OCR_PADDLE_PARALLEL_OFF", "").strip().lower() in ("1", "true", "yes"):
        return 1
    raw = (os.environ.get("OCR_PADDLE_PARALLEL_WORKERS") or "").strip()
    if raw:
        try:
            w = int(raw)
            return max(1, min(n_langs, w))
        except ValueError:
            pass
    return 1


def _paddle_detection_env_soft_defaults(
    env: Dict[str, str], *, small_text_boost: bool, high_recall: bool
) -> None:
    """
    When UI enables fine-print / high-recall OCR, inject softer Paddle detector defaults **only if**
    the user has not set ``OCR_PADDLE_*`` variables (helps thin English such as ``System &``, ``Access``).

    Does not affect an already-running Paddle daemon; ``_run_paddle_ocr_subprocess`` bypasses the daemon
    whenever these boosts are active so each worker child sees this env.
    """

    def _unset(key: str) -> bool:
        return not (env.get(key) or "").strip()

    if not (small_text_boost or high_recall):
        return
    if _unset("OCR_PADDLE_TEXT_DET_THRESH"):
        env["OCR_PADDLE_TEXT_DET_THRESH"] = "0.26"
    if _unset("OCR_PADDLE_TEXT_DET_BOX_THRESH"):
        env["OCR_PADDLE_TEXT_DET_BOX_THRESH"] = "0.38"
    if small_text_boost and _unset("OCR_PADDLE_DET_LIMIT_SIDE_LEN"):
        env["OCR_PADDLE_DET_LIMIT_SIDE_LEN"] = "2048"
    if high_recall and _unset("OCR_PADDLE_TEXT_REC_SCORE_THRESH"):
        env["OCR_PADDLE_TEXT_REC_SCORE_THRESH"] = "0.38"


def _paddle_subprocess_env(
    *,
    small_text_boost: bool = False,
    high_recall: bool = False,
) -> Dict[str, str]:
    """UTF-8 child env so Chinese in stdout JSON decodes correctly on Windows (cp1252 default breaks ``loads``)."""
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    _paddle_detection_env_soft_defaults(
        env, small_text_boost=small_text_boost, high_recall=high_recall
    )
    return env


# Persistent Paddle subprocess: avoids reloading PaddleOCR on every frame (dominant cost for video).
_paddle_daemon_proc: Optional[subprocess.Popen] = None
_paddle_daemon_key: Optional[Tuple[str, bool]] = None
_paddle_daemon_lock = threading.Lock()


def _paddle_daemon_env_enabled() -> bool:
    v = os.environ.get("OCR_PADDLE_DAEMON", "1").strip().lower()
    return v not in ("0", "false", "no")


def _parse_paddle_worker_json_list(payload_str: str) -> List[Tuple[Any, str, float]]:
    out: List[Tuple[Any, str, float]] = []
    try:
        payload = json.loads((payload_str or "").strip() or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    for row in payload:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        bbox, text, conf = row[0], str(row[1] or ""), row[2]
        if not text.strip() or not _looks_like_quad(bbox):
            continue
        try:
            c = float(conf)
        except (TypeError, ValueError):
            c = 0.0
        out.append((bbox, text, c))
    return out


def _paddle_shutdown_proc(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        if proc.stdin:
            proc.stdin.write("__EXIT__\n")
            proc.stdin.flush()
            proc.wait(timeout=8)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=3)
        except Exception:
            pass


def _paddle_invalidate_daemon() -> None:
    global _paddle_daemon_proc, _paddle_daemon_key
    with _paddle_daemon_lock:
        p = _paddle_daemon_proc
        _paddle_daemon_proc = None
        _paddle_daemon_key = None
    _paddle_shutdown_proc(p)


def _ensure_paddle_daemon(translator: Any) -> Optional[subprocess.Popen]:
    global _paddle_daemon_proc, _paddle_daemon_key
    if getattr(translator, "backend", "") != "paddleocr":
        return None
    # Daemon holds one PaddleOCR(lang=…) instance — multi-lang runs use fresh subprocesses per lang.
    _plangs = getattr(translator, "_paddle_langs", None)
    if _plangs is not None and len(_plangs) != 1:
        return None
    key = (str(translator._paddle_lang), bool(getattr(translator, "_paddle_use_gpu", True)))
    worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paddle_ocr_worker.py")
    if not os.path.isfile(worker_path):
        return None

    with _paddle_daemon_lock:
        proc = _paddle_daemon_proc
        if proc is not None and proc.poll() is None and _paddle_daemon_key == key:
            return proc

    old_to_kill: Optional[subprocess.Popen] = None
    with _paddle_daemon_lock:
        proc = _paddle_daemon_proc
        if proc is not None and proc.poll() is None:
            if _paddle_daemon_key == key:
                return proc
            old_to_kill = proc
            _paddle_daemon_proc = None
            _paddle_daemon_key = None

    if old_to_kill is not None:
        _paddle_shutdown_proc(old_to_kill)

    cmd = [
        sys.executable,
        worker_path,
        "--daemon",
        "--lang",
        key[0],
        "--use-gpu",
        "1" if key[1] else "0",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=_paddle_subprocess_env(),
        )
        line = proc.stdout.readline() if proc.stdout else ""
        if not line or line.strip() != "READY":
            try:
                proc.kill()
            except Exception:
                pass
            return None
    except Exception:
        return None

    with _paddle_daemon_lock:
        cur = _paddle_daemon_proc
        if cur is not None and cur.poll() is None and _paddle_daemon_key == key:
            try:
                proc.kill()
            except Exception:
                pass
            return cur
        _paddle_daemon_proc = proc
        _paddle_daemon_key = key
        return proc


def _paddle_daemon_request(proc: subprocess.Popen, tmp_path: str) -> List[Tuple[Any, str, float]]:
    with _paddle_daemon_lock:
        if proc.poll() is not None:
            raise OSError("paddle daemon exited")
        if not proc.stdin or not proc.stdout:
            raise OSError("paddle daemon missing pipes")
        proc.stdin.write(json.dumps({"path": tmp_path}) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
    if not line:
        _paddle_invalidate_daemon()
        raise OSError("paddle daemon EOF")
    return _parse_paddle_worker_json_list(line)


atexit.register(_paddle_invalidate_daemon)


def _ocr_min_confidence(high_recall: bool = False) -> float:
    """Minimum EasyOCR confidence to keep a box (0–1). Lower = more recall, more noise."""
    try:
        if high_recall:
            v = float(os.environ.get("OCR_HIGH_RECALL_MIN_CONFIDENCE", "0.22"))
        else:
            v = float(os.environ.get("OCR_MIN_CONFIDENCE", "0.30"))
        return max(0.12, min(0.95, v))
    except ValueError:
        return 0.22 if high_recall else 0.30


def _ocr_confidence_soft_floor(high_recall: bool = False) -> float:
    """
    Second-pass floor when no boxes pass min_conf (outlined / poster / stylized fonts score low).
    Set OCR_CONFIDENCE_SOFT_FLOOR=0.12 to tighten; OCR_SOFT_CONFIDENCE_OFF=1 to disable second pass.
    """
    if os.environ.get("OCR_SOFT_CONFIDENCE_OFF", "").strip().lower() in ("1", "true", "yes"):
        return 1.0  # disables soft pass (no score passes)
    try:
        default = "0.045" if high_recall else "0.055"
        key = "OCR_HIGH_RECALL_SOFT_FLOOR" if high_recall else "OCR_CONFIDENCE_SOFT_FLOOR"
        v = float(os.environ.get(key, default))
        return max(0.04, min(0.30, v))
    except ValueError:
        return 0.045 if high_recall else 0.055


def _photo_translated_font_scale() -> float:
    """
    Global scale for translated text size on static photo overlays.
    Increase to make translated text larger while still respecting per-box fit.
    """
    try:
        v = float(os.environ.get("PHOTO_TRANSLATED_FONT_SCALE", "1.6"))
    except ValueError:
        v = 1.6
    return max(0.7, min(2.5, v))


def _video_translated_font_scale() -> float:
    """
    Global scale for translated text size on video frame overlays.
    1.0 keeps the previous auto-fit size. Lower values (e.g. 0.7) shrink subtitles
    so they don't dominate the frame; higher values enlarge them. Always respects
    per-box fit (the renderer still shrinks further if the text would overflow).
    """
    try:
        v = float(os.environ.get("VIDEO_TRANSLATED_FONT_SCALE", "0.75"))
    except ValueError:
        v = 0.75
    return max(0.3, min(1.5, v))


def _ocr_row_unpack(row: Tuple) -> Tuple[str, List[List[int]], str, float]:
    """Normalize OCR segment to (text, bbox, lang, confidence). Older callers may omit confidence."""
    if len(row) >= 4:
        return str(row[0]), row[1], str(row[2]), float(row[3])
    return str(row[0]), row[1], str(row[2]), 0.55


def _ocr_effective_max_dim(
    requested: int, small_text_boost: bool, high_recall: bool = False
) -> int:
    """Use a higher pixel budget for small/fine print (slower, better box alignment)."""
    base = int(requested)
    if base <= 0:
        return 0
    if small_text_boost:
        try:
            floor = int(os.environ.get("OCR_SMALL_TEXT_MIN_DIM", "2000"))
        except ValueError:
            floor = 2000
        base = max(base, max(1600, min(3200, floor)))
    if high_recall:
        try:
            hr_floor = int(os.environ.get("OCR_HIGH_RECALL_MIN_DIM", "1800"))
        except ValueError:
            hr_floor = 1800
        base = max(base, max(1400, min(3200, hr_floor)))
        if small_text_boost:
            try:
                extra = int(os.environ.get("OCR_HIGH_RECALL_EXTRA_DIM", "200"))
            except ValueError:
                extra = 200
            base = min(3200, base + max(0, extra))
    return base


def _segment_ok_for_soft_confidence(t: str, high_recall: bool = False) -> bool:
    """Avoid keeping 1–2 character noise when using the soft confidence pass."""
    s = (t or "").strip()
    if len(s) < 2:
        return False
    arabic_like = 0
    for c in s:
        o = ord(c)
        if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF:
            arabic_like += 1
        elif 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF:
            arabic_like += 1
    if arabic_like >= 3:
        return True
    # Short Arabic words (e.g. بوابة) can be length 4–5; they already passed arabic_like>=3 above.
    # Two–three letter Arabic + diacritics sometimes scores arabic_like==2 — still real signage text.
    if arabic_like >= 2 and len(s) <= 72:
        latin_letters = sum(1 for c in s if ("A" <= c <= "Z") or ("a" <= c <= "z"))
        if arabic_like >= max(1, latin_letters):
            return True
    alnum = sum(1 for c in s if c.isalnum())
    long_cut = (95, 24 if high_recall else 28)
    if len(s) > long_cut[0] and alnum < long_cut[1]:
        return False  # long lines that are mostly non-letters (OCR garbage / bleed-through)
    if high_recall:
        return alnum >= 6 or len(s) >= 10
    return alnum >= 8 or len(s) >= 14


def _is_short_latin_token(t: str) -> bool:
    """
    Single-line Latin word (e.g. 'Gate', 'Exit') that is too short for _segment_ok_for_soft_confidence
    (which wants 8+ alnum) but is still real sign text. Used with a higher confidence floor so we do
    not let 3-letter noise through the soft-confidence path.
    """
    s = (t or "").strip()
    if len(s) < 2 or len(s) > 16:
        return False
    if re.search(r"[^\s'A-Za-z\-]", s):
        return False
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 2 or len(letters) > 12:
        return False
    return True


def _extract_triples_from_easyocr(
    results: List[Tuple], detected_lang: str, high_recall: bool = False
) -> List[Tuple[str, List[List[int]], str, float]]:
    """
    Keep high-confidence boxes, and also low-confidence boxes when the string looks like real text
    (poster fonts: white + black outline on gray often yield scores ~0.06–0.28 while still being correct).

    Short snippets need either score >= 0.82, the plausibility check, or (short Latin like 'Gate' with
    moderate conf). The soft pass still uses *only* plausibility (strict) to avoid 3-char junk at 0.06.
    """
    min_conf = _ocr_min_confidence(high_recall=high_recall)
    soft = _ocr_confidence_soft_floor(high_recall=high_recall)
    try:
        short_latin_min = float(os.environ.get("OCR_SHORT_LATIN_MIN_CONF", "0.35"))
    except ValueError:
        short_latin_min = 0.35
    short_latin_min = max(0.20, min(0.80, short_latin_min))
    out: List[Tuple[str, List[List[int]], str, float]] = []

    def key_of(t: str) -> str:
        k = _loose_text_key(t)
        return k if len(k) >= 2 else t.lower()

    def _find_duplicate_pass_same_spot(
        k: str,
        aabb: Tuple[float, float, float, float],
        conf: float,
    ) -> Tuple[int, bool]:
        """
        Find near-duplicate OCR boxes for the same text key.
        Returns (index_in_out, replace_existing_with_current).
        """
        cur_area = max(1.0, (aabb[2] - aabb[0]) * (aabb[3] - aabb[1]))
        for idx, (prev_t, prev_bbox, _, prev_conf) in enumerate(out):
            if key_of(prev_t) != k:
                continue
            prev_aabb = _poly_axis_aligned_bbox(prev_bbox)
            prev_area = max(1.0, (prev_aabb[2] - prev_aabb[0]) * (prev_aabb[3] - prev_aabb[1]))
            iou = _aabb_iou(aabb, prev_aabb)
            inter = _aabb_intersection_area(aabb, prev_aabb)
            iomin = inter / min(cur_area, prev_area) if min(cur_area, prev_area) > 0 else 0.0
            # IoU catches same-size duplicates; IoMin catches one-big/one-tight container duplicates.
            if iou < 0.22 and iomin < 0.72:
                continue
            replace = (
                iomin >= 0.72
                and cur_area < (prev_area * 0.78)
                and conf >= (float(prev_conf) - 0.18)
            )
            return idx, replace
        return -1, False

    for (bbox, text, confidence) in results:
        t = _postprocess_ocr_segment_text(text)
        if _is_junk_segment(t):
            continue
        plausible = _segment_ok_for_soft_confidence(t, high_recall=high_recall)
        stok = _is_short_latin_token(t)
        take = False
        if confidence >= min_conf and (
            confidence >= 0.82
            or plausible
            or (stok and confidence >= short_latin_min)
        ):
            take = True
        elif confidence >= soft and plausible:
            take = True
        if not take:
            continue
        k = key_of(t)
        aabb = _poly_axis_aligned_bbox(bbox)
        conf_f = float(confidence)
        dup_idx, replace = _find_duplicate_pass_same_spot(k, aabb, conf_f)
        if dup_idx >= 0:
            if replace:
                out[dup_idx] = (t, bbox, detected_lang, conf_f)
            continue
        out.append((t, bbox, detected_lang, conf_f))
    # If every line was filtered out (strict plausibility + confidence gates), the UI shows "no text"
    # even though EasyOCR returned boxes. Recover once with a low confidence floor and no plausibility gate.
    if not out and results:
        floor = max(0.08, min(0.45, _ocr_env_float("OCR_EMPTY_RESCUE_MIN_CONF", 0.12)))
        seen_k: set = set()
        scored: List[Tuple[float, str, Any]] = []
        for item in results:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            bbox, text, confidence = item[0], item[1], item[2]
            t = _postprocess_ocr_segment_text(text)
            if _is_junk_segment(t):
                continue
            try:
                conf_f = float(confidence)
            except Exception:
                conf_f = 0.0
            if conf_f < floor:
                continue
            if len(t.strip()) < 2:
                continue
            scored.append((conf_f, t, bbox))
        scored.sort(key=lambda x: -x[0])
        for conf_f, t, bbox in scored[:160]:
            k = key_of(t)
            if len(k) >= 2 and k in seen_k:
                continue
            if len(k) >= 2:
                seen_k.add(k)
            out.append((t, bbox, detected_lang, conf_f))
    return out


def _scale_ocr_triples_to_original_image(
    triples: List[Tuple],
    orig_h: int,
    orig_w: int,
    proc_h: int,
    proc_w: int,
) -> List[Tuple[str, List[List[int]], str, float]]:
    """
    EasyOCR returns boxes in the pixel space of the image passed to readtext (often downscaled for max_dim).
    Overlay and inpainting use the full-resolution file — map coordinates back so text lands on the real glyphs.
    """
    if not triples or proc_w <= 0 or proc_h <= 0:
        return triples
    if orig_w == proc_w and orig_h == proc_h:
        return triples
    sx = orig_w / float(proc_w)
    sy = orig_h / float(proc_h)
    out: List[Tuple[str, List[List[int]], str, float]] = []
    for row in triples:
        text, bbox, lang, conf = _ocr_row_unpack(row)
        nb = [[int(round(p[0] * sx)), int(round(p[1] * sy))] for p in bbox]
        out.append((text, nb, lang, conf))
    return out


def _poly_axis_aligned_bbox(poly: List[List[int]]) -> Tuple[float, float, float, float]:
    pts = np.array(poly, dtype=float)
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


def _aabb_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = aa + ba - inter
    return inter / denom if denom > 0 else 0.0


def _aabb_intersection_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    return iw * ih


def _ar_en_satellite_filter_enabled() -> bool:
    """Drop tiny Latin ghost boxes (e.g. ``Code``) sitting on Arabic lines. OCR_AR_EN_SATELLITE_FILTER=0 disables."""
    return os.environ.get("OCR_AR_EN_SATELLITE_FILTER", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _latin_satellite_bleed_token(t: str) -> bool:
    s = (t or "").strip()
    if not s or len(s) > 10:
        return False
    if _count_arabic_script_letters(s) > 0 or _text_has_cjk(s):
        return False
    core = re.sub(r"[^A-Za-z]+", "", s).lower()
    return core in ("code", "qr")


def _arabic_body_segment_for_satellite(text: str) -> bool:
    ac = _count_arabic_script_letters(text)
    lc = _count_basic_latin_letters(text)
    return ac >= 10 and ac >= max(6, 3 * max(1, lc))


def _vertical_overlap_frac(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    iy1, iy2 = max(a[1], b[1]), min(a[3], b[3])
    ih = max(0.0, iy2 - iy1)
    ha = max(1.0, a[3] - a[1])
    hb = max(1.0, b[3] - b[1])
    return ih / min(ha, hb)


def _filter_ar_en_satellite_boxes(triples: List[Tuple]) -> List[Tuple]:
    if not _ar_en_satellite_filter_enabled() or len(triples) < 2:
        return triples
    rows = [_ocr_row_unpack(r) for r in triples]
    aaabs = [_poly_axis_aligned_bbox(row[1]) for row in rows]
    drop_idx: set[int] = set()
    for i, (ti, _, _, _) in enumerate(rows):
        if not _latin_satellite_bleed_token(ti):
            continue
        ai = aaabs[i]
        la = max(1.0, (ai[2] - ai[0]) * (ai[3] - ai[1]))
        for j, (tj, _, _, _) in enumerate(rows):
            if i == j or j in drop_idx:
                continue
            if not _arabic_body_segment_for_satellite(tj):
                continue
            aj = aaabs[j]
            inter = _aabb_intersection_area(ai, aj)
            iomin_lat = inter / la if la > 0 else 0.0
            vy = _vertical_overlap_frac(ai, aj)
            if vy >= 0.38 and iomin_lat >= 0.10:
                drop_idx.add(i)
                break
    if not drop_idx:
        return triples
    return [triples[i] for i in range(len(triples)) if i not in drop_idx]


def _suppress_overlapping_photo_overlays(
    results: List[Tuple],
    translated_texts: List[str],
    img_w: int,
    img_h: int,
) -> Tuple[List[Tuple], List[str]]:
    """
    Drop duplicate overlapping OCR boxes for photo overlays.
    Keep non-duplicate overlaps (different OCR readings) so text recall does not regress.
    """
    if len(results) != len(translated_texts) or len(results) < 2:
        return results, translated_texts
    if os.environ.get("PHOTO_OVERLAY_SUPPRESS_OVERLAP", "").strip().lower() in ("0", "false", "no"):
        return results, translated_texts
    aaabs = [_poly_axis_aligned_bbox(r[1]) for r in results]
    areas = [max(1.0, (a[2] - a[0]) * (a[3] - a[1])) for a in aaabs]
    # Keep tighter boxes first; large container boxes are often low-quality duplicates.
    order = sorted(range(len(results)), key=lambda i: areas[i])
    kept: List[int] = []
    for i in order:
        ai = aaabs[i]
        hi = max(1.0, ai[3] - ai[1])
        cy_i = (ai[1] + ai[3]) / 2.0
        drop = False
        for j in kept:
            aj = aaabs[j]
            hj = max(1.0, aj[3] - aj[1])
            cy_j = (aj[1] + aj[3]) / 2.0
            iou = _aabb_iou(ai, aj)
            inter = _aabb_intersection_area(ai, aj)
            smaller = min(areas[i], areas[j])
            iomin = inter / smaller if smaller > 0 else 0.0
            if iou < 0.26 and iomin < 0.40:
                continue
            if abs(cy_i - cy_j) > 0.58 * max(hi, hj):
                continue
            txt_i = str(results[i][0]) if len(results[i]) > 0 and results[i][0] is not None else ""
            txt_j = str(results[j][0]) if len(results[j]) > 0 and results[j][0] is not None else ""
            # Only suppress when readings are clearly redundant, or one box almost fully contains another.
            # This avoids dropping useful alternatives that improve recall.
            redundant = _ocr_redundant_strings(txt_i, txt_j)
            strong_containment = iomin >= 0.82
            if redundant or strong_containment:
                drop = True
                break
        if not drop:
            kept.append(i)
    kept.sort(key=lambda idx: (aaabs[idx][1], aaabs[idx][0]))
    return [results[i] for i in kept], [translated_texts[i] for i in kept]


def _normalize_photo_overlay_text_color(
    estimated: Tuple[int, int, int],
    img_rgb: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> Tuple[int, int, int]:
    """On light walls/menus, avoid brown/gold bleed from wood — use neutral dark gray/black."""
    h, w = img_rgb.shape[:2]
    x_min = max(0, min(w - 1, x_min))
    y_min = max(0, min(h - 1, y_min))
    x_max = max(x_min + 1, min(w, x_max))
    y_max = max(y_min + 1, min(h, y_max))
    pad = max(8, min(28, (x_max - x_min) // 3))
    X0, Y0 = max(0, x_min - pad), max(0, y_min - pad)
    X1, Y1 = min(w, x_max + pad), min(h, y_max + pad)
    patch = img_rgb[Y0:Y1, X0:X1]
    if patch.size == 0:
        return estimated
    bg = float(np.median(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)))
    er, eg, eb = estimated
    lum_e = 0.299 * er + 0.587 * eg + 0.114 * eb
    mx, mn = max(er, eg, eb), min(er, eg, eb)
    chroma = mx - mn
    # Only neutralize warm mud/gold from wood sampling; keep intentional brand colors.
    if bg > 165.0 and lum_e < 195.0 and chroma < 42:
        return (26, 26, 28)
    return estimated


def _ocr_redundant_strings(a: str, b: str) -> bool:
    """Same reading or one is a clear substring fragment of the other (duplicate detection)."""
    if _loose_text_key(a) == _loose_text_key(b):
        return True
    sa, sb = "".join(a.split()), "".join(b.split())
    if len(sa) < 3 or len(sb) < 3:
        return sa == sb
    return sa in sb or sb in sa


def _text_has_cjk(s: str) -> bool:
    """Han / Hiragana–Katakana / Hangul — used to avoid Latin-only OCR merges dropping Chinese boxes."""
    for c in s or "":
        if "\u4e00" <= c <= "\u9fff":
            return True
        if "\u3040" <= c <= "\u30ff":
            return True
        if "\uac00" <= c <= "\ud7af":
            return True
    return False


def _count_arabic_script_letters(s: str) -> int:
    """Arabic / Arabic presentation blocks (aligned with ``_is_arabic_or_rtl``)."""
    n = 0
    for c in s or "":
        if "\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F" or "\u08A0" <= c <= "\u08FF":
            n += 1
        elif "\uFB50" <= c <= "\uFDFF" or "\uFE70" <= c <= "\uFEFF":
            n += 1
    return n


def _count_basic_latin_letters(s: str) -> int:
    return sum(1 for c in (s or "") if ("A" <= c <= "Z") or ("a" <= c <= "z"))


def _ocr_ar_lat_segment_kind(s: str) -> str:
    """
    Coarse script bucket for bilingual Arabic/Latin merge decisions (no CJK).

    ``mixed_ar_lat`` ≈ one Paddle crop spanning Arabic + embedded Latin (e.g. ``… QR Code``);
    must not collapse into a pure-Arabic or pure-Latin neighbor when merging Paddle ``ar`` + ``en`` passes.
    """
    ac = _count_arabic_script_letters(s)
    lc = _count_basic_latin_letters(s)
    # ``QR``, ``TM``, etc. can be only two Latin letters — still a bilingual crop vs pure Arabic lines.
    if ac >= 4 and lc >= 2:
        return "mixed_ar_lat"
    if ac >= 4:
        return "arabic"
    if lc >= 3 and ac <= 2:
        return "latin"
    return "weak"


def _ocr_ar_lat_family_merge_clash(a: str, b: str) -> bool:
    """
    True when two segments belong to different Arabic/Latin buckets — EasyOCR-like separation for signage.

    Weak buckets (digits, very short tokens) do not block merges so duplicate passes can still dedupe.
    """
    if _text_has_cjk(a) or _text_has_cjk(b):
        return False
    ka = _ocr_ar_lat_segment_kind(a)
    kb = _ocr_ar_lat_segment_kind(b)
    if ka == "weak" or kb == "weak":
        return False
    return ka != kb


def _ocr_merge_scripts_clash(a: str, b: str) -> bool:
    """
    True when one segment is CJK-heavy and the other is not (e.g. English vs Chinese on bilingual signs).

    High-IoU merge must not treat these as the same slot — otherwise the longer/higher-conf Latin reading
    replaces legitimate Chinese from another Paddle/EasyOCR pass.
    """
    a = (a or "").strip()
    b = (b or "").strip()
    ca, cb = _text_has_cjk(a), _text_has_cjk(b)
    if ca != cb:
        return True
    if _ocr_ar_lat_family_merge_clash(a, b):
        return True
    return False


def _union_horizontal_aabbs(
    boxes: List[List[float]],
) -> List[float]:
    """Merge [xmin, xmax, ymin, ymax] boxes into one axis-aligned box."""
    if not boxes:
        return [0, 0, 0, 0]
    return [
        min(b[0] for b in boxes),
        max(b[1] for b in boxes),
        min(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _group_easyocr_horizontal_into_lines(
    horizontal_list: List,
    y_frac: Optional[float] = None,
) -> List[List[List[float]]]:
    """
    Group EasyOCR horizontal boxes (same hand-written line) so TrOCR runs once per line.
    Per-word tiny boxes cause wrong / duplicated TrOCR output; the handwriting model expects line crops.
    """
    if not horizontal_list:
        return []
    if y_frac is None:
        try:
            y_frac = float(os.environ.get("OCR_TROCR_LINE_Y_FRAC", "0.42"))
        except ValueError:
            y_frac = 0.42
    y_frac = max(0.12, min(0.8, y_frac))

    boxes: List[List[float]] = [[float(x) for x in row] for row in horizontal_list]
    n = len(boxes)
    cys = [0.5 * (b[2] + b[3]) for b in boxes]
    hs = [max(1.0, b[3] - b[2]) for b in boxes]
    order = sorted(range(n), key=lambda i: cys[i])
    lines: List[List[int]] = []
    for idx in order:
        best_li = -1
        best_d = 1e18
        for li, line in enumerate(lines):
            mc = float(np.median([cys[i] for i in line]))
            mh = float(np.median([hs[i] for i in line]))
            tol = y_frac * max(hs[idx], mh, 1.0)
            d = abs(cys[idx] - mc)
            if d <= tol and d < best_d:
                best_d = d
                best_li = li
        if best_li >= 0:
            lines[best_li].append(idx)
        else:
            lines.append([idx])

    def _line_median_y(line: List[int]) -> float:
        return float(np.median([cys[i] for i in line]))

    lines.sort(key=_line_median_y)
    out: List[List[List[float]]] = []
    for line in lines:
        line.sort(key=lambda i: boxes[i][0])
        out.append([boxes[i] for i in line])
    return out


def _aabb_to_quad4(aabb: List[float]) -> List[List[int]]:
    xmin, xmax, ymin, ymax = aabb
    return [
        [int(round(xmin)), int(round(ymin))],
        [int(round(xmax)), int(round(ymin))],
        [int(round(xmax)), int(round(ymax))],
        [int(round(xmin)), int(round(ymax))],
    ]


def _join_merged_line_texts(parts: List[str]) -> str:
    """Join line fragments; drop adjacent near-duplicates (substring / same loose key)."""
    segs: List[str] = []
    for p in parts:
        p = " ".join(str(p).split())
        if not p:
            continue
        if not segs:
            segs.append(p)
            continue
        if _ocr_redundant_strings(segs[-1], p):
            if len(p) > len(segs[-1]):
                segs[-1] = p
            continue
        segs.append(p)
    return " ".join(segs)


def _merge_trocr_adjacent_same_line_boxes(
    results: List[Tuple[Any, str, float]],
    y_frac: Optional[float] = None,
) -> List[Tuple[Any, str, float]]:
    """
    Merge nearby boxes on the same text line and drop adjacent duplicate readings.
    Used after TrOCR hybrid so word-level / tilted leftovers become fewer, cleaner segments.
    """
    if os.environ.get("OCR_TROCR_MERGE_ADJACENT_OFF", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return results
    if len(results) <= 1:
        return results
    if y_frac is None:
        try:
            y_frac = float(os.environ.get("OCR_TROCR_MERGE_Y_FRAC", "0.38"))
        except ValueError:
            y_frac = 0.38
    y_frac = max(0.12, min(0.65, y_frac))

    items: List[Dict[str, Any]] = []
    for bbox, text, conf in results:
        t = _postprocess_ocr_segment_text(text)
        if _is_junk_segment(t):
            continue
        try:
            aabb = _poly_axis_aligned_bbox(bbox)
        except Exception:
            continue
        cy = 0.5 * (aabb[1] + aabb[3])
        h = max(1.0, aabb[3] - aabb[1])
        w = max(1.0, aabb[2] - aabb[0])
        items.append(
            {
                "bbox": bbox,
                "text": t,
                "conf": float(conf),
                "aabb": aabb,
                "cy": cy,
                "h": h,
                "w": w,
            }
        )
    if len(items) <= 1:
        return [(it["bbox"], it["text"], it["conf"]) for it in items]

    items.sort(key=lambda it: (it["cy"], it["aabb"][0]))

    lines: List[List[Dict[str, Any]]] = []
    for it in items:
        best_li = -1
        best_d = 1e18
        for li, line in enumerate(lines):
            mc = float(np.median([x["cy"] for x in line]))
            mh = float(np.median([x["h"] for x in line]))
            tol = y_frac * max(it["h"], mh, 1.0)
            d = abs(it["cy"] - mc)
            if d <= tol and d < best_d:
                best_d = d
                best_li = li
        if best_li >= 0:
            lines[best_li].append(it)
        else:
            lines.append([it])

    def _line_sort_key(line: List[Dict[str, Any]]) -> float:
        return float(np.median([x["cy"] for x in line]))

    lines.sort(key=_line_sort_key)

    out: List[Tuple[Any, str, float]] = []
    for line in lines:
        line.sort(key=lambda x: x["aabb"][0])
        median_w = float(np.median([x["w"] for x in line])) if line else 24.0
        try:
            max_gap = float(os.environ.get("OCR_TROCR_MERGE_MAX_GAP_PX", "0"))
        except ValueError:
            max_gap = 0.0
        if max_gap <= 0:
            max_gap = max(18.0, 0.78 * max(median_w, 12.0))

        group: List[Dict[str, Any]] = []

        def flush_group(g: List[Dict[str, Any]]) -> None:
            if not g:
                return
            x1 = min(x["aabb"][0] for x in g)
            y1 = min(x["aabb"][1] for x in g)
            x2 = max(x["aabb"][2] for x in g)
            y2 = max(x["aabb"][3] for x in g)
            joined = _join_merged_line_texts([x["text"] for x in g])
            if not joined.strip():
                return
            conf = max(x["conf"] for x in g)
            quad = _aabb_to_quad4([x1, x2, y1, y2])
            out.append((quad, joined, conf))

        for it in line:
            if not group:
                group.append(it)
                continue
            prev = group[-1]
            pa = prev["aabb"]
            ia = it["aabb"]
            gap = ia[0] - pa[2]
            same_band = abs(it["cy"] - prev["cy"]) <= y_frac * max(
                it["h"], prev["h"], 1.0
            )
            if same_band and gap <= max_gap:
                if _ocr_redundant_strings(prev["text"], it["text"]):
                    if len(it["text"]) > len(prev["text"]):
                        group[-1] = it
                    continue
                group.append(it)
            else:
                flush_group(group)
                group = [it]
        flush_group(group)

    return out if out else results


def _merge_overlapping_easyocr_results(
    results: List[Tuple[Any, str, float]],
) -> List[Tuple[Any, str, float]]:
    """
    Combine multiple readtext passes: drop duplicate boxes for the same line, keep longer / higher-conf text.
    """
    if not results:
        return []
    items: List[Tuple[Any, str, float, Tuple[float, float, float, float]]] = []
    for bbox, text, conf in results:
        t = _postprocess_ocr_segment_text(text)
        if _is_junk_segment(t):
            continue
        items.append((bbox, t, float(conf), _poly_axis_aligned_bbox(bbox)))
    items.sort(key=lambda x: -x[2])
    merged: List[Tuple[Any, str, float, Tuple[float, float, float, float]]] = []
    for bbox, t, conf, aabb in items:
        replaced = False
        for i, (mb, mt, mc, maabb) in enumerate(merged):
            if _aabb_iou(aabb, maabb) < 0.35:
                continue
            iou_hi = _aabb_iou(aabb, maabb) >= 0.56
            # Do not collapse unrelated scripts when boxes overlap (Paddle en + Paddle ch on same region).
            if iou_hi and _ocr_merge_scripts_clash(t, mt) and not _ocr_redundant_strings(t, mt):
                continue
            same_slot = _ocr_redundant_strings(t, mt) or iou_hi
            if not same_slot:
                continue
            if (len(t), conf) > (len(mt), mc):
                merged[i] = (bbox, t, max(conf, mc), aabb)
            else:
                merged[i] = (mb, mt, max(conf, mc), maabb)
            replaced = True
            break
        if not replaced:
            merged.append((bbox, t, conf, aabb))
    return [(b, tx, c) for b, tx, c, _ in merged]


def _ocr_multipass_disabled() -> bool:
    return os.environ.get("OCR_MERGE_PASSES_OFF", "").strip().lower() in ("1", "true", "yes")


def _ocr_use_multipass_extract(backend: str) -> bool:
    """Merge strict + relaxed (+ optional light) OCR passes in `_extract_text_core`.

    Each pass runs full `_readtext` — for **trocr_hybrid** / **trocr_only** that repeats EasyOCR
    **detection** plus TrOCR (~2× wall time). Default **single pass** for those backends.
    Set **OCR_TROCR_MULTIPASS=1** to restore multipass recall. **OCR_MERGE_PASSES_OFF=1** forces single pass for all backends.
    """
    if os.environ.get("OCR_MERGE_PASSES_OFF", "").strip().lower() in ("1", "true", "yes"):
        return False
    if backend in ("trocr_hybrid", "trocr_only"):
        return os.environ.get("OCR_TROCR_MULTIPASS", "").strip().lower() in ("1", "true", "yes")
    return True


def _ocr_pipeline_trace_enabled() -> bool:
    return os.environ.get("OCR_PIPELINE_TRACE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ) or os.environ.get("OCR_TROCR_PERF_LOG", "").strip().lower() in ("1", "true", "yes")


def _trocr_img_detect_fingerprint(img: np.ndarray) -> Tuple[Any, ...]:
    """Cheap fingerprint for detection caching (same image → reuse CRAFT boxes)."""
    import zlib

    if img is None or img.size == 0:
        return (0, 0, 0, "")
    h, w = int(img.shape[0]), int(img.shape[1])
    sy = max(1, h // 96)
    sx = max(1, w // 96)
    sample = np.ascontiguousarray(img[::sy, ::sx])
    crc = zlib.crc32(sample.tobytes()) & 0xFFFFFFFF
    return (h, w, crc, str(img.dtype))


def _trocr_detect_kw_cache_tuple(kw: dict) -> Tuple[Tuple[Any, Any], ...]:
    """Stable hashable view of EasyOCR ``detect`` kwargs."""
    return tuple(sorted((str(k), kw[k]) for k in sorted(kw.keys())))


def _ocr_skip_light_preprocess_pass() -> bool:
    return os.environ.get("OCR_SKIP_LIGHT_PREPROCESS_PASS", "").strip().lower() in ("1", "true", "yes")


def _ocr_video_skip_near_duplicate_frames() -> bool:
    """Skip OCR when consecutive sampled frames look identical (cheap MSE on a tiny grayscale thumb)."""
    v = os.environ.get("OCR_VIDEO_SKIP_NEAR_DUP_FRAMES", "1").strip().lower()
    return v not in ("0", "false", "no")


def _ocr_video_frame_mse_skip_threshold() -> float:
    """Mean squared error threshold (0–255² scale) on 96×54 grayscale thumbs; lower = stricter."""
    try:
        return float(os.environ.get("OCR_VIDEO_FRAME_MSE_SKIP_THRESH", "12"))
    except ValueError:
        return 12.0


def _video_frame_gray_small_fp(bgr: np.ndarray, size: Tuple[int, int] = (96, 54)) -> np.ndarray:
    sw, sh = max(1, int(size[0])), max(1, int(size[1]))
    if bgr is None or bgr.size == 0:
        return np.zeros((sh, sw), dtype=np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return _safe_cv2_resize(gray, sw, sh).astype(np.float32)


def _ocr_video_frame_single_pass() -> bool:
    """Video frames only: one `_readtext` call instead of merged strict+relaxed passes (faster).

    Applies to every backend: EasyOCR / PaddleOCR subprocess / trocr_hybrid — each extra pass is full work.
    Default is single-pass for speed. Set OCR_VIDEO_MERGE_PASSES_OFF=0 to use the same multipass merge
    as still images (when OCR_MERGE_PASSES_OFF is unset).
    """
    raw = os.environ.get("OCR_VIDEO_MERGE_PASSES_OFF")
    if raw is None or str(raw).strip() == "":
        return True
    v = str(raw).strip().lower()
    if v in ("0", "false", "no"):
        return _ocr_multipass_disabled()
    return True


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order corners as tl, tr, br, bl."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _warp_quad_to_frontal(img_bgr: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
    """Apply perspective transform; returns None if dimensions are unreasonable."""
    rect = _order_quad_points(pts.astype(np.float32))
    tl, tr, br, bl = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_w = max(int(width_a), int(width_b))
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_h = max(int(height_a), int(height_b))
    if max_w < 100 or max_h < 100:
        return None
    if max_w > 14000 or max_h > 14000:
        return None
    ar = max_w / float(max_h)
    if ar > 4.5 or ar < 0.22:
        return None
    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img_bgr, m, (max_w, max_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _find_largest_quad_points(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Find a document/screen quadrilateral in image coordinates (4x2 float).
    Runs edge detection on a downscaled copy; scales points back to full resolution.
    """
    h, w = img_bgr.shape[:2]
    scale = min(1.0, 920.0 / max(h, w))
    if scale < 1.0:
        small = _safe_cv2_resize(img_bgr, max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    else:
        small = img_bgr
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    area_img = float(sh * sw)
    for low, high in ((45, 140), (25, 110), (60, 180)):
        edged = cv2.Canny(blurred, low, high)
        edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=2)
        cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:25]
        for c in cnts:
            aa = cv2.contourArea(c)
            if aa < 0.10 * area_img:
                continue
            if aa > 0.96 * area_img:
                continue
            peri = cv2.arcLength(c, True)
            for eps in (0.02, 0.028, 0.035, 0.045, 0.055):
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) != 4:
                    continue
                pts = approx.reshape(4, 2).astype(np.float32)
                inv = 1.0 / scale
                pts[:, 0] *= inv
                pts[:, 1] *= inv
                return pts
    return None


def try_perspective_dewarp_image(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Rectify an angled photo of a screen or document. Returns warped BGR or None if no quad found.
    Set OCR_DEWARP_OFF=1 to disable.
    """
    if os.environ.get("OCR_DEWARP_OFF", "").strip().lower() in ("1", "true", "yes"):
        return None
    pts = _find_largest_quad_points(img_bgr)
    if pts is None:
        return None
    return _warp_quad_to_frontal(img_bgr, pts)


# Candidate fonts for matching original text style (Windows + generic)
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/timesbd.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/tahomabd.ttf",
    "C:/Windows/Fonts/verdana.ttf",
    "C:/Windows/Fonts/verdanab.ttf",
    "C:/Windows/Fonts/georgia.ttf",
    "C:/Windows/Fonts/georgiab.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "arial.ttf",
    "Arial.ttf",
    "DejaVuSans.ttf",
    "DejaVuSerif.ttf",
]

# TrueType search order for overlay text. If nothing loads, PIL uses load_default(), which
# cannot draw Arabic and shows '?' in translated photos.
def _truetype_path_candidates(prefer_arabic: bool) -> List[str]:
    seen = set()
    out: List[str] = []

    def _add(p: str) -> None:
        if not p:
            return
        ap = os.path.normpath(p)
        if ap in seen:
            return
        seen.add(ap)
        if os.path.isfile(ap):
            out.append(ap)

    if prefer_arabic:
        for _ek in ("OCR_ARABIC_FONT", "OCR_FONT", "OCR_UNICODE_FONT"):
            v = (os.environ.get(_ek) or "").strip()
            if v:
                _add(v)
    else:
        for _ek in ("OCR_FONT", "OCR_LATIN_FONT"):
            v = (os.environ.get(_ek) or "").strip()
            if v:
                _add(v)
    _d = (os.environ.get("OCR_TTF_DIR") or "").strip()
    if _d and os.path.isdir(_d):
        try:
            for _n in sorted(os.listdir(_d)):
                if _n.lower().endswith((".ttf", ".otf")):
                    _add(os.path.join(_d, _n))
        except OSError:
            pass

    if prefer_arabic:
        _core = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/seguiui.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/tahomabd.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/notonaskharabicui.ttf",
            "C:/Windows/Fonts/NotoNaskhArabicUI-Regular.ttf",
            "/mnt/c/Windows/Fonts/segoeui.ttf",
            "/mnt/c/Windows/Fonts/tahoma.ttf",
            "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
            "/usr/share/fonts/truetype/google-noto-vf/NotoNaskhArabic[wght].ttf",
            "/usr/local/share/fonts/NotoNaskhArabic-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Tahoma.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "arial.ttf",
            "Arial.ttf",
            "DejaVuSans.ttf",
        ]
    else:
        _core = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/seguiui.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/mnt/c/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "arial.ttf",
            "Arial.ttf",
            "DejaVuSans.ttf",
        ]
    for _p in _core:
        _add(_p)
    return out


def _annotate_font_paths_env_only() -> List[str]:
    """Explicit annotation fonts from environment."""
    seen: set = set()
    out: List[str] = []

    def _add(p: str) -> None:
        if not p:
            return
        ap = os.path.normpath(p)
        if ap in seen:
            return
        seen.add(ap)
        if os.path.isfile(ap):
            out.append(ap)

    for key in ("OCR_ANNOT_FONT", "OCR_UNICODE_FONT", "OCR_ARABIC_FONT", "OCR_CJK_FONT"):
        v = (os.environ.get(key) or "").strip()
        if v:
            _add(v)
    return out


def _annotate_font_paths_arabic_latin() -> List[str]:
    """Fonts that render Arabic + Latin (Arabic must come before CJK-only fonts in the global list)."""
    seen: set = set()
    out: List[str] = []

    def _add(p: str) -> None:
        if not p:
            return
        ap = os.path.normpath(p)
        if ap in seen:
            return
        seen.add(ap)
        if os.path.isfile(ap):
            out.append(ap)

    for p in (
        "C:/Windows/Fonts/seguiui.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/seguiem.ttf",
        "C:/Windows/Fonts/seguibl.ttf",
        "C:/Windows/Fonts/notonaskharabicui.ttf",
        "C:/Windows/Fonts/NotoNaskhArabicUI-Regular.ttf",
        "C:/Windows/Fonts/trado.ttf",
        "C:/Windows/Fonts/tradbdo.ttf",
        "C:/Windows/Fonts/arabtype.ttf",
        "C:/Windows/Fonts/arabtypes.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ):
        _add(p)
    return out


def _annotate_font_paths_cjk() -> List[str]:
    """Han / JP / KR biased fonts (often lack Arabic; use after Arabic-aware fonts for RTL labels)."""
    seen: set = set()
    out: List[str] = []

    def _add(p: str) -> None:
        if not p:
            return
        ap = os.path.normpath(p)
        if ap in seen:
            return
        seen.add(ap)
        if os.path.isfile(ap):
            out.append(ap)

    for p in (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/msjh.ttf",
        "C:/Windows/Fonts/msjhbd.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simsunb.ttf",
        "C:/Windows/Fonts/msyi.ttf",
        "C:/Windows/Fonts/YuGothR.ttc",
        "C:/Windows/Fonts/malgun.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ):
        _add(p)
    return out


def _annotate_overlay_font_candidates() -> List[str]:
    """Ordered font paths for OCR bounding-box labels (Arabic/Latin before CJK-only — avoids Arabic tofu)."""
    merged: List[str] = []
    merged.extend(_annotate_font_paths_env_only())
    merged.extend(_annotate_font_paths_arabic_latin())
    merged.extend(_annotate_font_paths_cjk())
    merged.extend(_truetype_path_candidates(False))
    seen: set = set()
    out: List[str] = []
    for p in merged:
        ap = os.path.normpath(p)
        if ap in seen:
            continue
        seen.add(ap)
        if os.path.isfile(ap):
            out.append(ap)
    return out


def _try_load_annotation_font(paths: List[str], size_px: int) -> Optional[ImageFont.ImageFont]:
    """Try paths in order; support ``.ttc`` face indices."""
    size_px = max(10, min(36, int(size_px)))
    last_err: Optional[Exception] = None
    for path in paths:
        trials: List[Optional[int]]
        if path.lower().endswith(".ttc"):
            trials = [0, 1, 2, 3]
        else:
            trials = [None]
        for idx in trials:
            try:
                if idx is None:
                    return _pil_truetype(path, size_px)
                return ImageFont.truetype(
                    path, size_px, index=int(idx), layout_engine=ImageFont.Layout.BASIC
                )
            except OSError:
                continue
    return None


def _annotate_font_for_bbox_label(label: str, size_px: int) -> ImageFont.ImageFont:
    """Pick a font that matches the script inside ``label`` (Arabic vs CJK vs Latin)."""
    chunks: List[str] = []
    chunks.extend(_annotate_font_paths_env_only())
    if _is_arabic_or_rtl(label):
        chunks.extend(_annotate_font_paths_arabic_latin())
        chunks.extend(_truetype_path_candidates(False))
        chunks.extend(_annotate_font_paths_cjk())
    elif _text_has_cjk(label):
        chunks.extend(_annotate_font_paths_cjk())
        chunks.extend(_annotate_font_paths_arabic_latin())
        chunks.extend(_truetype_path_candidates(False))
    else:
        chunks.extend(_truetype_path_candidates(False))
        chunks.extend(_annotate_font_paths_arabic_latin())
        chunks.extend(_annotate_font_paths_cjk())

    seen: set = set()
    deduped: List[str] = []
    for p in chunks:
        ap = os.path.normpath(p)
        if ap in seen:
            continue
        seen.add(ap)
        if os.path.isfile(ap):
            deduped.append(ap)

    font = _try_load_annotation_font(deduped, size_px)
    if font is not None:
        return font
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def _annotate_overlay_font(size_px: int) -> ImageFont.ImageFont:
    """Load first usable font using the full annotation candidate chain."""
    font = _try_load_annotation_font(_annotate_overlay_font_candidates(), size_px)
    if font is not None:
        return font
    try:
        return ImageFont.load_default()
    except Exception:
        raise RuntimeError("No overlay font available for annotations.") from None


# Optional: Arabic reshaping and RTL (logical → visual for PIL)
try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    _ARABIC_RESHAPE_CONFIG = {
        "delete_harakat": False,
        "delete_tatweel": True,
        "support_ligatures": True,
        "support_zwj": True,
        "support_zwnj": True,
    }
    try:
        _RESHAPER = arabic_reshaper.ArabicReshaper(configuration=_ARABIC_RESHAPE_CONFIG)
    except Exception:
        _RESHAPER = None
    _HAS_ARABIC_SUPPORT = True
except ImportError:
    arabic_reshaper = None  # type: ignore
    get_display = None  # type: ignore
    _RESHAPER = None
    _HAS_ARABIC_SUPPORT = False

# Contiguous Arabic-script runs (spaces / Latin break words — good for per-word Paddle fixes).
_ARABIC_SCRIPT_RUN_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+"
)


def _arabic_paddle_run_fix_enabled() -> bool:
    """Paddle sometimes emits reversed logical Arabic per word; fix before reshape+BiDi. Set OCR_ARABIC_PADDLE_RUN_FIX=0 to disable."""
    return os.environ.get("OCR_ARABIC_PADDLE_RUN_FIX", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _reshape_arabic_for_join_score(run: str) -> str:
    if not run:
        return run
    if _RESHAPER is not None:
        return _RESHAPER.reshape(run)
    if arabic_reshaper is not None:
        return arabic_reshaper.reshape(run)
    return run


def _arabic_join_quality_score(reshaped: str) -> int:
    """Higher → more presentation-form shaping (reversed Paddle runs reshape poorly)."""
    score = 0
    for c in reshaped:
        o = ord(c)
        if 0xFE70 <= o <= 0xFEFF or 0xFB50 <= o <= 0xFDFF:
            score += 4
        elif "\u0600" <= c <= "\u06FF":
            score += 1
    return score


def _correct_arabic_runs_for_logical_order(s: str) -> str:
    """
    Paddle / PP-OCR Arabic recognizer often outputs each word in reverse logical character order.
    Fix per contiguous Arabic script run before global reshape + BiDi (safe for mixed Arabic+Latin labels).
    """
    if not _arabic_paddle_run_fix_enabled() or arabic_reshaper is None:
        return s
    out_parts: List[str] = []
    pos = 0
    for m in _ARABIC_SCRIPT_RUN_RE.finditer(s):
        out_parts.append(s[pos : m.start()])
        run = m.group(0)
        pos = m.end()
        if len(run) < 2:
            out_parts.append(run)
            continue
        if any(c.isdigit() or ("\u0660" <= c <= "\u0669") for c in run):
            out_parts.append(run)
            continue
        try:
            fwd = _reshape_arabic_for_join_score(run)
            rev = _reshape_arabic_for_join_score(run[::-1])
            sf = _arabic_join_quality_score(fwd)
            sr = _arabic_join_quality_score(rev)
            if sr > sf:
                chosen = run[::-1]
            elif sf > sr:
                chosen = run
            else:
                # Scores often tie; Paddle frequently emits reversed logical order with Teh Marbuta first.
                chosen = run[::-1] if (run[0] == "\u0629" and len(run) > 1) else run
            out_parts.append(chosen)
        except Exception:
            out_parts.append(run)
    out_parts.append(s[pos:])
    return "".join(out_parts)


def _is_arabic_or_rtl(text: str) -> bool:
    """True if text contains Arabic or other RTL characters."""
    if not text or not text.strip():
        return False
    for c in text:
        if "\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F" or "\u08A0" <= c <= "\u08FF":
            return True
        if "\u0590" <= c <= "\u05FF":  # Hebrew
            return True
        if "\uFB50" <= c <= "\uFDFF" or "\uFE70" <= c <= "\uFEFF":  # Arabic presentation forms (NLLB/OCR often emit these)
            return True
        if c in "\u0590\u05D0\u05F0\u0600\u0700\uFB1D\uFB50\uFDF0\uFE70":
            return True
    return False


# Photo overlay: Latin loanwords match Arabic/Persian/Urdu line styling (font stack + outline).
_PHOTO_UNIFY_LATIN_TARGETS = frozenset({"ar", "fa", "ur", "ps"})
# Targets using a non-Latin primary script — do not apply Latin-only color unification (ar uses style_like_rtl).
_PHOTO_NON_LATIN_SCRIPT_TARGETS = frozenset(
    {"ar", "fa", "ur", "ps", "he", "zh", "ja", "ko", "ru", "hi", "th", "bn", "ta", "te", "ml", "my"}
)


def _photo_unify_latin_overlay_colors(target_lang: Optional[str]) -> bool:
    """German/English/… targets: one Latin text color so loanwords match translated lines (not OCR-white from the sign)."""
    t = (target_lang or "").strip().lower()
    if not t:
        return False
    if t in _PHOTO_NON_LATIN_SCRIPT_TARGETS:
        return False
    return True


def _is_translation_failure_overlay(raw: Optional[str]) -> bool:
    """True when translation failed — do not paint error blobs on the photo (area stays inpaint-only)."""
    t = (raw or "").strip()
    if not t:
        return False
    tl = t.lower()
    if tl.startswith("translation error:"):
        return True
    if tl.startswith("translation unavailable"):
        return True
    if "unsupported language" in tl:
        return True
    if "httpsconnectionpool" in tl or "max retries exceeded" in tl or "connection refused" in tl:
        return True
    if "defaultcredentialserror" in tl or "could not find the default credentials" in tl:
        return True
    if "winerror 10051" in tl or "unreachable network" in tl:
        return True
    return False


def _strip_ocr_uncertain_for_photo(s: str) -> str:
    """Remove app warning prefix from strings painted on the translated image (per-segment or whole paragraph)."""
    return re.sub(r"\(OCR uncertain\)\s*", "", (s or ""), flags=re.IGNORECASE)


def _photo_segment_display_text(trans_text: str) -> str:
    raw = _strip_ocr_uncertain_for_photo((trans_text or "").strip())
    if not raw or set(raw.replace(" ", "")) <= set("_-"):
        return "..."
    display_text = _sanitize_rendered_text(raw)
    if not (display_text or "").strip():
        return "..."
    return display_text


def _is_mostly_latin_script(s: str) -> bool:
    """
    True when the string is Latin-script only (no Arabic/Hebrew RTL in the segment).
    Used so loanwords left in English on Arabic-target photos use the same overlay font/outline as Arabic lines.
    """
    if not (s or "").strip():
        return False
    if _is_arabic_or_rtl(s):
        return False
    return any(("A" <= c <= "Z") or ("a" <= c <= "z") for c in s)


def _ocr_unstick_enabled() -> bool:
    """Split glued Latin subtitle words after OCR (wordninja). Set OCR_WORD_UNSTICK=0 to disable."""
    return os.environ.get("OCR_WORD_UNSTICK", "1").strip().lower() not in ("0", "false", "no")


def _ocr_latin_intraword_collapse_enabled() -> bool:
    """
    Collapse spurious spaces inside one OCR region (e.g. German ``A us gang`` → ``Ausgang``).
    Disable via OCR_LATIN_INTRAWORD_COLLAPSE=0.
    """
    return os.environ.get("OCR_LATIN_INTRAWORD_COLLAPSE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


# BMP Latin letters + Latin Extended for signage (includes äöüß etc.). Used only inside OCR collapse regex/heuristics.
_OCR_LATIN_CLASS_CC = r"a-zA-Z\xc0-\u024f\u1e00-\u1eff"


def _is_pure_unicode_letter_token(tok: str) -> bool:
    """True if ``tok`` is non-empty and every character has Unicode category Letter."""
    if not tok:
        return False
    for ch in tok:
        if unicodedata.category(ch)[0] != "L":
            return False
    return True


def _latin_intraword_merge_candidate_string(s: str) -> bool:
    """Skip exotic scripts so collapse logic stays Latin-centric."""
    if not (s or "").strip():
        return False
    if _is_arabic_or_rtl(s):
        return False
    for ch in s:
        o = ord(ch)
        if 0x0400 <= o <= 0x052F:  # Cyrillic blocks
            return False
        if 0x0530 <= o <= 0x058F:  # Armenian
            return False
        if 0x0590 <= o <= 0x05FF:  # Hebrew
            return False
        if 0x0600 <= o <= 0x06FF:  # Arabic
            return False
        if 0x0370 <= o <= 0x03FF:  # Greek
            return False
    return True


def _maybe_join_intraword_spaced_latin(parts: List[str]) -> Optional[str]:
    """
    If OCR split one word into letter-only tokens, join without spaces.

    Examples:
        A / us / gang → Ausgang
        Toilet / ten → Toiletten
        G / eschäft → Geschäft

    Keeps plausible two-word Latin such as ``Rest Rooms`` (capital boundary inside second token).
    """
    if len(parts) < 2:
        return None
    if not all(_is_pure_unicode_letter_token(p) for p in parts):
        return None

    if len(parts) >= 3 and any(len(p) == 1 for p in parts):
        return "".join(parts)

    if len(parts) == 2:
        a, b = parts[0], parts[1]
        if len(a) == 1 and len(b) == 1:
            return "".join(parts)
        if len(a) == 1:
            if a.lower() in ("i", "a") and len(b) <= 4:
                return None
            if b and b[0].islower():
                return "".join(parts)
            return None
        if len(b) == 1:
            if a[-1].islower() and b.lower() != "i":
                return "".join(parts)
            return None
        if a[-1].islower() and b and b[0].islower() and len(b) <= 4:
            return "".join(parts)

    return None


def _collapse_intraword_spaced_pure_latin_line(s: str) -> str:
    """Apply ``_maybe_join_intraword_spaced_latin`` to strings that are only letters + spaces."""
    raw = (s or "").strip()
    if " " not in raw or not _latin_intraword_merge_candidate_string(raw):
        return s
    merged = _maybe_join_intraword_spaced_latin(raw.split())
    if merged is None:
        return s
    lead = len(s) - len(s.lstrip())
    trail = len(s) - len(s.rstrip())
    pad_l = s[:lead] if lead else ""
    pad_r = s[len(s) - trail :] if trail else ""
    return pad_l + merged + pad_r


wordninja = None  # type: ignore[assignment]
_HAS_WORDNINJA = False
try:
    _wordninja_mod = importlib.import_module("wordninja")
    wordninja = _wordninja_mod
    _HAS_WORDNINJA = True
except Exception:
    _HAS_WORDNINJA = False


def _unstick_glued_latin_ocr_text(s: str) -> str:
    """
    Re-insert spaces in glued English tokens (e.g. Iwear, thatinecklace) using statistical splitting.
    Skips Arabic/RTL-only text; does not alter runs with camelCase glitches (loveiL) — those need vision fixes.
    """
    if not s or not _HAS_WORDNINJA or not _ocr_unstick_enabled():
        return s
    if _is_arabic_or_rtl(s) and not re.search(r"[A-Za-z]", s):
        return s

    def repl(m) -> str:
        w = m.group(0)
        if len(w) < 4:
            return w
        # Mixed camelCase inside token — don't guess (avoid splitting loveiL wrong)
        if re.search(r"[a-z][A-Z]", w):
            return w
        lower = w.lower()
        try:
            parts = wordninja.split(lower)
        except Exception:
            return w
        if len(parts) <= 1:
            return w
        # Wordninja is English-centric; keep plausible single signage tokens it splits (e.g. German compounds).
        if len(parts) == 2 and len(parts[0]) >= 3 and len(parts[1]) >= 3:
            if len(w) >= 2 and w[0].isupper() and w[1:].islower():
                return w
            if w.isupper():
                return w
        if w.isupper():
            return " ".join(p.upper() for p in parts)
        out = " ".join(parts)
        if w[0].isupper():
            return out[0].upper() + out[1:] if out else w
        return out

    try:
        return re.sub(r"[A-Za-z]{4,}", repl, s)
    except Exception:
        return s


def _collapse_fragmented_latin_ocr_text(s: str) -> str:
    """
    Collapse OCR-induced fake spacing inside one Latin word.
    Examples: "C H E E S E" -> "CHEESE", "CH EE SE" -> "CHEESE", "R O A S T" -> "ROAST",
    "G eschäft" -> "Geschäft", "Toilet ten" -> "Toiletten", "A us gang" -> "Ausgang".

    Extended Latin letters (German umlauts, accents) use Unicode categories / BMP Latin ranges so
    Paddle/EasyOCR splits are not missed by ASCII-only regex.
    """
    if not s:
        return s
    if not _ocr_latin_intraword_collapse_enabled():
        return s

    lc = _OCR_LATIN_CLASS_CC

    def _collapse_letters_only(m: re.Match) -> str:
        return m.group(0).replace(" ", "")

    def _collapse_short_chunks(m: re.Match) -> str:
        tok = m.group(0)
        parts = tok.split()
        short_piece = re.compile(rf"(?u)[{lc}]{{1,2}}")
        if len(parts) >= 3 and all(short_piece.fullmatch(p) for p in parts):
            if sum(len(p) for p in parts) >= 5:
                return "".join(parts)
        return tok

    def _detach_first_letter_if_suffix_lower(m: re.Match) -> str:
        a, b = m.group(1), m.group(2)
        if not b:
            return m.group(0)
        # Keep English splits like ``i Information`` or ``A street`` (article/proper noun).
        if b[0].isupper():
            return m.group(0)
        if a.lower() == "a" and len(b) >= 5:
            return m.group(0)
        return a + b

    def _one_round(out_in: str) -> str:
        out = out_in
        # 1) C H E E S E (Unicode Latin)
        out = re.sub(
            rf"(?u)\b(?:[{lc}]\s+){{3,}}[{lc}]\b",
            _collapse_letters_only,
            out,
        )
        # 2) CH EE SE / RO AS T
        out = re.sub(
            rf"(?u)\b(?:[{lc}]{{1,2}}\s+){{2,}}[{lc}]{{1,2}}\b",
            _collapse_short_chunks,
            out,
        )
        # 3) Detached first letter: C HEESE, G eschäft — only when the rest starts lowercase (not ``i Information``).
        out = re.sub(
            rf"(?u)\b([{lc}])\s+([{lc}]{{3,}})\b",
            _detach_first_letter_if_suffix_lower,
            out,
        )
        # 4) Heuristic syllable / OCR fragment joins (EasyOCR/Paddle German signage)
        out = _collapse_intraword_spaced_pure_latin_line(out)
        return out

    prev = None
    out = s
    for _ in range(8):
        prev = out
        out = _one_round(out)
        if out == prev:
            break
    return out


def _ar_en_bleed_scrub_enabled() -> bool:
    """Strip stray Latin tokens Paddle glued onto Arabic lines (e.g. QR row bleed). OCR_AR_EN_BLEED_SCRUB=0 disables."""
    return os.environ.get("OCR_AR_EN_BLEED_SCRUB", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _pipeline_latin_typo_fix_enabled() -> bool:
    """Apply ``_apply_overlay_english_ocr_typo_fixes`` during OCR postprocess (not only bbox overlay)."""
    return os.environ.get("OCR_PIPELINE_LATIN_TYPO_FIX", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _strip_qr_latin_prefix_before_arabic(text: str) -> str:
    """
    Drop ``QR Code`` / ``Code`` / ``QR`` and tiny Latin stubs (e.g. ``ge``) that the recognizer prepends to Arabic.

    Iterative so patterns like ``QR Code ge العربية`` collapse to the Arabic run.
    """
    if _count_arabic_script_letters(text) < 4:
        return text
    ar_mark = r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
    out = " ".join(text.strip().split())
    for _ in range(8):
        prev = out
        out = re.sub(
            r"^(?:QR\s*Code|Code|QR)\s+",
            "",
            out,
            flags=re.IGNORECASE,
        )
        out = " ".join(out.split())
        # Broken English fragments immediately before Arabic (Paddle/EasyOCR glue).
        out = re.sub(
            rf"^[A-Za-z]{{1,3}}\s+(?={ar_mark})",
            "",
            out,
        )
        out = " ".join(out.split())
        if out == prev:
            break
    return out


def _scrub_ar_latin_line_bleed(text: str) -> str:
    """
    Remove short Latin OCR bleed prefixes/suffixes on Arabic lines (spatial / detector confusion).

    Strips ``QR Code`` / ``Code`` / ``QR`` before Arabic; suffix junk after Arabic when Arabic clearly dominates.
    """
    if not _ar_en_bleed_scrub_enabled() or not (text or "").strip():
        return text
    out = " ".join(text.strip().split())
    out = _strip_qr_latin_prefix_before_arabic(out)
    cand = re.sub(
        r"^(?:Code|QR)\s+(?=[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF])",
        "",
        out,
        flags=re.IGNORECASE,
    )
    if cand != out and _count_arabic_script_letters(cand) >= 4:
        out = " ".join(cand.split())
    cand2 = re.sub(
        r"(?<=[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF])\s+(?:Code|QR)$",
        "",
        out,
        flags=re.IGNORECASE,
    )
    if cand2 != out and _count_arabic_script_letters(cand2) >= 4:
        out = " ".join(cand2.split())
    return out


def _postprocess_ocr_segment_text(text: str) -> str:
    """Normalize each OCR box string before translation or display."""
    raw = text or ""
    # Latin-space-collapsing heuristics can corrupt spacing around mixed scripts; skip for CJK runs.
    if _text_has_cjk(raw):
        return " ".join(raw.split())
    # Merge OCR intraword gaps first; ``wordninja`` unstick can split compounds like ``Ausgang``, then merge again.
    collapsed_once = _collapse_fragmented_latin_ocr_text(raw)
    chain = _unstick_glued_latin_ocr_text(collapsed_once)
    chain = _collapse_fragmented_latin_ocr_text(chain)
    chain = _scrub_ar_latin_line_bleed(chain)
    if _pipeline_latin_typo_fix_enabled():
        chain = _latin_ocr_typo_fix_inner(chain)
    return chain


def _overlay_english_typo_fix_enabled() -> bool:
    """PP/EasyOCR-style Latin fixes for green bbox labels only. Disable via OCR_OVERLAY_ENGLISH_TYPO_FIX=0."""
    return os.environ.get("OCR_OVERLAY_ENGLISH_TYPO_FIX", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


_OCR_OVERLAY_ENGLISH_FIX_PAIRS = tuple(
    (re.compile(pat), repl)
    for pat, repl in (
        # Paddle splits / drops glyphs inside English signage (street gates, QR notices).
        (r"(?i)\bReo\s+e\b", "Remote"),
        (r"(?i)\bRe\s+der\b", "Reader"),
        (r"(?i)\bReo\s+der\b", "Reader"),
        (r"(?i)\bR\s+code\b", "QR Code"),
        (r"(?i)\belectronic\s+ate\b", "ELECTRONIC GATE"),
        # Lost / thin Q stroke before "Code":
        (r"(?i)\bn\s+code\b", "QR Code"),
        (r"(?i)\bread\s+er\b", "Reader"),
        (r"(?i)\bacc\s+ess\b", "Access"),
    )
)


def _overlay_typo_fix_gate_case(sample: str) -> str:
    """Match casing style of an OCR token when swapping ate→gate."""
    if sample.isupper():
        return "GATE"
    if sample.islower():
        return "gate"
    if len(sample) > 1 and sample[0].isupper() and sample[1:].islower():
        return "Gate"
    return "GATE"


def _latin_ocr_typo_fix_inner(s: str) -> str:
    """
    Conservative Latin OCR cosmetic fixes (regex + isolated ``ATE`` → ``GATE``).
    Used from bbox overlay and optionally from ``_postprocess_ocr_segment_text``.
    """
    if not (s or "").strip():
        return s
    if not re.search(r"[A-Za-z]", s):
        return s
    out = s
    try:
        for rx, repl in _OCR_OVERLAY_ENGLISH_FIX_PAIRS:
            out = rx.sub(repl, out)
        core = out.strip()
        if re.fullmatch(r"(?i)ate", core):
            gate = _overlay_typo_fix_gate_case(core)
            lead = len(out) - len(out.lstrip())
            trail = len(out) - len(out.rstrip())
            return out[:lead] + gate + (out[len(out) - trail :] if trail else "")
    except Exception:
        return s
    return out


def _apply_overlay_english_ocr_typo_fixes(s: str) -> str:
    """Latin typo fixes for green bbox labels (see ``OCR_OVERLAY_ENGLISH_TYPO_FIX``)."""
    if not _overlay_english_typo_fix_enabled():
        return s
    return _latin_ocr_typo_fix_inner(s)


def _is_junk_segment(s: str) -> bool:
    """True if segment is placeholder/junk and should not be shown as real content on the card."""
    if not s or not s.strip():
        return True
    t = s.strip()
    if t in (".", "...", "-", "<", ">", "،", "،"):
        return True
    if set(t.replace(" ", "")) <= set("._-<>،"):
        return True
    return False


def _is_url_or_footer_ocr(s: str) -> bool:
    """Footer / URL lines (e.g. template 'really great site') — erase but do not paint translation."""
    t = (s or "").strip().lower()
    if len(t) < 4:
        return False
    if re.search(r"https?://|www\.|\.(com|net|org)\b", t):
        return True
    if "reallygreatsite" in t.replace(" ", "") or "greatsite" in t.replace(" ", ""):
        return True
    if re.match(r"^[\w.-]+\.(com|net|org)\s*$", t):
        return True
    return False


def _loose_text_key(s: str) -> str:
    """Normalize OCR/translation for duplicate detection (drops punctuation, extra spaces)."""
    t = " ".join(str(s).lower().split())
    t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)
    return " ".join(t.split())


def _ocr_frame_signature(results: List[Tuple]) -> Tuple[Any, ...]:
    """
    Fingerprint of a frame's OCR output for deduplicating consecutive video samples.
    Includes loose text keys and quantized box centers so static titles merge, but moved text does not.
    """
    if not results:
        return ("__empty__",)
    ordered = sorted(
        results,
        key=lambda item: (
            float(np.mean([p[1] for p in item[1]])),
            float(np.mean([p[0] for p in item[1]])),
        ),
    )
    parts: List[Any] = []
    for item in ordered:
        t, bbox, _, __ = _ocr_row_unpack(item)
        pts = np.array(bbox, dtype=float)
        yc = int(round(float(pts[:, 1].mean()) / 24.0))
        xc = int(round(float(pts[:, 0].mean()) / 48.0))
        parts.append((_loose_text_key(t), yc, xc))
    return tuple(parts)


def _dedupe_paragraph_parts(parts: List[str], loose: bool = False) -> List[str]:
    """Remove duplicate lines; loose=True merges OCR variants like 'thatinecklace' vs 'that necklace'."""
    seen: set = set()
    out: List[str] = []
    for p in parts:
        s = " ".join((p or "").split())
        if not s:
            continue
        k = _loose_text_key(s) if loose else s.lower()
        if len(k) < 2:
            k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _overlay_dedupe_text_key(orig: str, trans: str) -> str:
    """Normalized text key for overlay dedupe (subtitle / menu line identity)."""
    k = _loose_text_key(orig)
    if len(k) < 4:
        k = " ".join(trans.split()).lower()
    return k


def _overlay_dedupe_slot(orig: str, trans: str, bbox) -> Optional[Tuple[str, int, int]]:
    """
    Dedupe slot = text key + quantized box center.

    Same reading at the same place (twin OCR boxes on one subtitle line) collapses once;
    repeated labels on menus/forms at different positions stay visible.
    """
    k = _overlay_dedupe_text_key(orig, trans)
    if len(k) < 2:
        # Lone digits (e.g. order count "3") normalize to a 1-char loose key — keep them.
        fallback = (orig or trans or "").strip().lower()
        if fallback.isdigit() or (len(fallback) == 1 and fallback.isalnum()):
            k = fallback
        elif len(fallback) >= 2:
            k = fallback
        else:
            return None
    try:
        pts = np.array(bbox, dtype=float)
        yc = int(round(float(pts[:, 1].mean()) / 24.0))
        xc = int(round(float(pts[:, 0].mean()) / 48.0))
    except Exception:
        yc, xc = 0, 0
    return (k, yc, xc)


def _dedupe_overlay_segments_for_frame(
    results_f: List[Tuple],
    txts_f: List[str],
) -> Tuple[List[Tuple], List[str]]:
    """Drop duplicate OCR boxes on one frame (same text at the same spot / same translation slot)."""
    if len(results_f) != len(txts_f) or not results_f:
        return results_f, txts_f
    seen: set = set()
    out_r: List[Tuple] = []
    out_t: List[str] = []
    for r, t in zip(results_f, txts_f):
        orig, bbox, _, _ = _ocr_row_unpack(r)
        orig = (orig or "").strip()
        trans = (t or "").strip()
        slot = _overlay_dedupe_slot(orig, trans, bbox)
        if slot is None:
            continue
        if slot in seen:
            continue
        seen.add(slot)
        out_r.append(r)
        out_t.append(t)
    if not out_r:
        return results_f, txts_f
    return out_r, out_t


_OCR_ANNOT_BOX_RGB: Tuple[int, int, int] = (0, 210, 255)


def _ocr_annot_box_color_rgb(box_index: int = 0) -> Tuple[int, int, int]:
    """Cyan for OCR annotation boxes and in-box labels (readable on light and dark photos)."""
    return _OCR_ANNOT_BOX_RGB


def _ocr_annot_box_color_bgr(box_index: int = 0) -> Tuple[int, int, int]:
    r, g, b = _ocr_annot_box_color_rgb(box_index)
    return (b, g, r)


def _arabic_display_for_pil(logical: str) -> str:
    """Logical Arabic/RTL → visual string for PIL (ligatures + BiDi). Do not use on whole paragraphs before word-wrap."""
    if not (logical or "").strip():
        return logical
    if not _HAS_ARABIC_SUPPORT or get_display is None:
        return logical
    if not _is_arabic_or_rtl(logical):
        return logical
    try:
        logical_use = _correct_arabic_runs_for_logical_order(logical)
        has_arabic = any("\u0600" <= c <= "\u06FF" for c in logical_use)
        if has_arabic and arabic_reshaper is not None:
            if _RESHAPER is not None:
                reshaped = _RESHAPER.reshape(logical_use)
            else:
                reshaped = arabic_reshaper.reshape(logical_use)
        else:
            reshaped = logical_use
        return get_display(reshaped, base_dir="R")
    except Exception:
        return logical


def _prepare_text_for_draw(text: str) -> str:
    """Single-line logical → visual (for callers that already split lines)."""
    return _arabic_display_for_pil(text)


def _sanitize_rendered_text(s: str) -> str:
    """Drop replacement / control chars that draw as tofu with some fonts."""
    if not s:
        return s
    out = []
    for c in s:
        o = ord(c)
        if o == 0xFFFD:  # replacement character
            continue
        if o < 32 and c not in "\t\n\r":
            continue
        out.append(c)
    return "".join(out)


def _estimate_text_color_rgb(
    img_rgb: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> Tuple[int, int, int]:
    """
    Guess original text color from the region before erasure (RGB).
    Uses Otsu on a grayscale crop and picks the median RGB of the smaller class (usually text).
    """
    h, w = img_rgb.shape[:2]
    x_min = max(0, min(w - 1, x_min))
    y_min = max(0, min(h - 1, y_min))
    x_max = max(x_min + 1, min(w, x_max))
    y_max = max(y_min + 1, min(h, y_max))
    crop = img_rgb[y_min:y_max, x_min:x_max]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return (0, 0, 0)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    k = min(5, max(3, (min(gray.shape[0], gray.shape[1]) // 4) | 1))
    if k % 2 == 0:
        k += 1
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n_dark = int(np.count_nonzero(binary == 0))
    n_light = int(np.count_nonzero(binary == 255))
    if n_dark == 0 and n_light == 0:
        return (0, 0, 0)
    if n_dark <= n_light:
        mask = binary == 0
    else:
        mask = binary == 255
    if np.count_nonzero(mask) < 3:
        if float(np.mean(gray)) > 127.0:
            return (255, 255, 255)
        return (0, 0, 0)
    text_pixels = crop[mask]
    med = np.median(text_pixels.reshape(-1, 3), axis=0)
    return (int(round(med[0])), int(round(med[1])), int(round(med[2])))


def _approx_local_bg_bgr(img_bgr: np.ndarray, pts: np.ndarray, img_w: int, img_h: int) -> Tuple[int, int, int]:
    """
    Median BGR of a ring around the text polygon (excludes center) so fills match wood/paper.
    Inner exclusion is tight so orange/graphics inside large OCR boxes do not dominate the fill.
    """
    pts = np.asarray(pts, dtype=np.int32)
    x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
    x1, y1 = int(pts[:, 0].max()), int(pts[:, 1].max())
    m = 22
    X0, Y0 = max(0, x0 - m), max(0, y0 - m)
    X1, Y1 = min(img_w, x1 + m), min(img_h, y1 + m)
    patch = img_bgr[Y0:Y1, X0:X1]
    if patch.size == 0:
        return (128, 128, 128)
    ph, pw = patch.shape[:2]
    rel_x0, rel_y0 = x0 - X0, y0 - Y0
    rel_x1, rel_y1 = x1 - X0, y1 - Y0
    mask = np.ones((ph, pw), dtype=np.uint8)
    # Exclude only the OCR core (text); sample mostly from surrounding wood/texture
    cx0 = int(rel_x0 + (rel_x1 - rel_x0) * 0.04)
    cy0 = int(rel_y0 + (rel_y1 - rel_y0) * 0.04)
    cx1 = int(rel_x0 + (rel_x1 - rel_x0) * 0.96)
    cy1 = int(rel_y0 + (rel_y1 - rel_y0) * 0.96)
    mask[cy0:cy1, cx0:cx1] = 0
    if np.count_nonzero(mask) < 10:
        v = np.median(patch.reshape(-1, 3), axis=0)
        return (int(round(v[0])), int(round(v[1])), int(round(v[2])))
    vals = patch[mask == 1]
    v = np.median(vals.reshape(-1, 3), axis=0)
    return (int(round(v[0])), int(round(v[1])), int(round(v[2])))


def _erase_text_regions_solid_local(
    img_bgr: np.ndarray,
    results: List[Tuple[Any, Any, Any]],
    img_w: int,
    img_h: int,
    *,
    tight_photo: bool = False,
) -> None:
    """Fill expanded OCR polygons with local background color (in-place). tight_photo uses smaller padding so fills hug text."""
    # Default on: wider masks fix leftover English/ghosting on menus (set PHOTO_STRONG_TEXT_ERASE=0 to tighten).
    _se = os.environ.get("PHOTO_STRONG_TEXT_ERASE", "1").strip().lower()
    _strong = _se not in ("0", "false", "no")
    if tight_photo:
        d = min(img_w, img_h)
        inpaint_pad_px = max(8, min(24, int(8 + d * 0.007)))
        inpaint_pad_ratio = min(0.13, 0.085)
        # Wider mask: OCR boxes often miss anti-aliased fringes and neighbor glyphs (e.g. English left of Arabic).
        if _strong:
            inpaint_pad_px = int(max(12, min(52, int(inpaint_pad_px * 1.5))))
            inpaint_pad_ratio = min(0.24, inpaint_pad_ratio * 1.65)
    else:
        inpaint_pad_px = max(24, DEFAULT_MASK_PADDING_PX + 6)
        inpaint_pad_ratio = min(0.32, DEFAULT_MASK_PADDING_RATIO + 0.06)
        if _strong:
            inpaint_pad_px = int(inpaint_pad_px * 1.12)
            inpaint_pad_ratio = min(0.38, inpaint_pad_ratio * 1.08)
    dilate_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5) if _strong else (3, 3)
    )
    for row in results:
        bbox = row[1]
        pts = _expand_bbox(bbox, img_w, img_h, inpaint_pad_px, inpaint_pad_ratio)
        if len(pts) < 3:
            continue
        np_pts = np.array(pts, dtype=np.int32)
        fill_col = _approx_local_bg_bgr(img_bgr, np_pts, img_w, img_h)
        if tight_photo:
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [np_pts], 255)
            _dit = 4 if _strong else 2
            mask = cv2.dilate(mask, dilate_k, iterations=_dit)
            img_bgr[mask > 0] = np.array(fill_col, dtype=img_bgr.dtype)
            # Blend flat fill toward a Gaussian of the local ROI so edges pick up wall/wood color (less “flat card”).
            if os.environ.get("PHOTO_SOLID_SMOOTH_PATCH", "1").strip().lower() not in (
                "0",
                "false",
                "no",
            ):
                ys, xs = np.where(mask > 0)
                if len(ys) > 0:
                    pad = max(8, min(24, int(min(img_w, img_h) * 0.012)))
                    y0, y1 = max(0, int(ys.min()) - pad), min(img_h, int(ys.max()) + pad + 1)
                    x0, x1 = max(0, int(xs.min()) - pad), min(img_w, int(xs.max()) + pad + 1)
                    roi = img_bgr[y0:y1, x0:x1].copy()
                    mroi = mask[y0:y1, x0:x1]
                    k = max(5, min(21, pad * 2 + 1)) | 1
                    blur = cv2.GaussianBlur(roi, (k, k), 0)
                    a = (mroi > 0).astype(np.float32)[:, :, np.newaxis]
                    blended = roi.astype(np.float32) * (1.0 - a) + blur.astype(np.float32) * a
                    img_bgr[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            cv2.fillPoly(img_bgr, [np_pts], fill_col)


def _inpaint_remove_text_regions_bgr(
    img_bgr: np.ndarray,
    results: List[Tuple[Any, Any, Any]],
    img_w: int,
    img_h: int,
) -> None:
    """
    Remove text with Telea inpaint, **one OCR region at a time** (in-place).

    A single union mask + large dilate merged stacked menu lines into one blob and inpainted
    huge blurred bands across areas with no text. Per-region masks stay local; mild dilate only.
    """
    d = min(img_w, img_h)
    _se = os.environ.get("PHOTO_STRONG_TEXT_ERASE", "1").strip().lower()
    _strong = _se not in ("0", "false", "no")
    # Enough padding for anti-aliased strokes, but less than solid-fill so adjacent lines do not merge.
    inpaint_pad_px = max(8, min(22, int(6 + d * 0.006)))
    inpaint_pad_ratio = min(0.16, 0.092)
    if _strong:
        inpaint_pad_px = int(max(10, min(36, int(inpaint_pad_px * 1.35))))
        inpaint_pad_ratio = min(0.22, inpaint_pad_ratio * 1.4)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5) if _strong else (3, 3))

    def _y_center(item: Tuple[Any, Any, Any]) -> float:
        bbox = item[1]
        pts = np.array(bbox, dtype=float)
        return float(pts[:, 1].mean()) if len(pts) else 0.0

    ordered = sorted(results, key=_y_center)
    for row in ordered:
        bbox = row[1]
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        pts = _expand_bbox(bbox, img_w, img_h, inpaint_pad_px, inpaint_pad_ratio)
        if len(pts) < 3:
            continue
        cv2.fillPoly(mask, [pts], 255)
        if not np.any(mask > 0):
            continue
        mask = cv2.dilate(mask, k, iterations=2 if _strong else 1)
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            continue
        rh = int(ys.max() - ys.min()) + 1
        rw = int(xs.max() - xs.min()) + 1
        # Small radius limits smudge outside the line; was img//45 and blurred empty background.
        # Smaller radius reduces dark bleed from wood/contrast edges into light wall areas.
        rad = max(3, min(12, int(0.11 * max(rw, rh, 8))))
        rad = min(rad, max(3, min(img_h, img_w) // 85))
        if _strong:
            rad = min(max(rad + 2, 5), 16)
        # NS often preserves wall/wood structure better than Telea (less gray directional smear).
        _alg = (os.environ.get("PHOTO_INPAINT_ALGORITHM") or "ns").strip().lower()
        _flag = cv2.INPAINT_TELEA if _alg in ("telea", "t", "fast") else cv2.INPAINT_NS
        img_bgr[:] = cv2.inpaint(img_bgr, mask, rad, _flag)


def _contrast_text_rgb_for_rtl(
    estimated: Tuple[int, int, int],
    img_rgb: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> Tuple[int, int, int]:
    """Arabic overlay: avoid low-contrast cream/yellow on food backgrounds — force readable gray/white."""
    h, w = img_rgb.shape[:2]
    x_min = max(0, min(w - 1, x_min))
    y_min = max(0, min(h - 1, y_min))
    x_max = max(x_min + 1, min(w, x_max))
    y_max = max(y_min + 1, min(h, y_max))
    crop = img_rgb[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return (18, 18, 18)
    bg = float(np.median(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)))
    er, eg, eb = estimated
    lum_e = 0.299 * er + 0.587 * eg + 0.114 * eb
    if abs(lum_e - bg) < 52:
        return (252, 252, 252) if bg < 135 else (22, 22, 22)
    return estimated


def _edge_tilt_from_nearest_axis_deg(p0, p1) -> float:
    """0° = edge parallel to x or y axis; ~45° = diagonal. Used to detect slanted text lines."""
    d = np.subtract(p1, p0, dtype=np.float64)
    if float(np.hypot(d[0], d[1])) < 0.5:
        return 0.0
    deg = abs(float(np.degrees(np.arctan2(d[1], d[0]))) % 180.0)
    if deg > 90.0:
        deg = 180.0 - deg
    return min(deg, 90.0 - deg)


def _quad_tilted_for_perspective(pts) -> bool:
    """
    True when top/bottom edge of the OCR quadrilateral is clearly not horizontal, so
    we should use perspective warp. Complements is_rotated (area heuristic), which
    can miss a parallelogram that is still a thin, slanted line of text.
    """
    if pts is None or len(pts) < 2:
        return False
    p = np.array(pts, dtype=np.float64)
    t_top = _edge_tilt_from_nearest_axis_deg(p[0], p[1])
    t_bot = 0.0
    if len(p) >= 4:
        t_bot = _edge_tilt_from_nearest_axis_deg(p[3], p[2])
    try:
        th = float(os.environ.get("PHOTO_PERSPECTIVE_MIN_TILT_DEG", "0.7"))
    except ValueError:
        th = 0.7
    th = max(0.2, min(8.0, th))
    return max(t_top, t_bot) >= th


def _should_apply_perspective_warp(for_video: bool, explicit: Optional[bool]) -> bool:
    """
    Video: always allow quadrilateral warping. Static photo: on by default so translated
    text can follow the OCR quad; set PHOTO_OVERLAY_PERSPECTIVE_WARP=0/false to force
    axis-aligned (horizontal) text in the bounding rectangle only.
    """
    if for_video:
        return True
    if explicit is not None:
        return bool(explicit)
    e = (os.environ.get("PHOTO_OVERLAY_PERSPECTIVE_WARP") or "").strip().lower()
    if e in ("0", "false", "no", "off"):
        return False
    if e in ("1", "true", "yes", "on"):
        return True
    return True


def _bbox_geometry(bbox: List[List[int]], img_w: int, img_h: int) -> dict:
    """
    Compute geometry from OCR bounding box (exact coordinates preserved).
    Returns dict with: x_min, y_min, x_max, y_max, w, h, pts (4 int points),
    is_rotated (True if box is not axis-aligned).
    """
    pts = np.array(bbox, dtype=np.float32)
    if len(pts) < 4:
        # Fallback to axis-aligned
        x_min = int(np.min(pts[:, 0])) if len(pts) else 0
        y_min = int(np.min(pts[:, 1])) if len(pts) else 0
        x_max = int(np.max(pts[:, 0])) if len(pts) else 0
        y_max = int(np.max(pts[:, 1])) if len(pts) else 0
        pts = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
    x_min = int(round(np.min(pts[:, 0])))
    y_min = int(round(np.min(pts[:, 1])))
    x_max = int(round(np.max(pts[:, 0])))
    y_max = int(round(np.max(pts[:, 1])))
    x_min = max(0, min(img_w, x_min))
    y_min = max(0, min(img_h, y_min))
    x_max = max(0, min(img_w, x_max))
    y_max = max(0, min(img_h, y_max))
    w = max(1, x_max - x_min)
    h = max(1, y_max - y_min)
    # Clamp pts to image so warping/blending never goes off-screen (avoids cut-off text)
    pts_clipped = np.clip(pts, [0, 0], [img_w - 1, img_h - 1]).astype(np.int32)
    poly_area = cv2.contourArea(pts)
    rect_area = w * h
    is_rotated = rect_area > 0 and abs(poly_area - rect_area) / max(rect_area, 1) > 0.05
    tilt_persp = _quad_tilted_for_perspective(pts_clipped)
    return {
        "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max,
        "w": w, "h": h, "pts": pts_clipped,
        "is_rotated": is_rotated,
        "tilt_perspective": tilt_persp,
    }


def _order_indices_top_to_bottom(
    results: List[Tuple],
    img_w: int,
    img_h: int,
) -> List[int]:
    """Return indices so that results ordered by this list go from top to bottom (correct line order for display)."""
    if not results:
        return []
    centers_y = []
    for row in results:
        bbox = row[1]
        g = _bbox_geometry(bbox, img_w, img_h)
        centers_y.append((g["y_min"] + g["y_max"]) / 2)
    return sorted(range(len(results)), key=lambda i: centers_y[i])


def _merge_boxes_same_line(
    results: List[Tuple],
    translated_texts: List[str],
    img_w: int,
    img_h: int,
    line_threshold_frac: float = 0.035,
) -> Tuple[List[Tuple], List[str]]:
    """
    Merge OCR boxes that lie on the same line into one box with combined text,
    so "two boxes" become one and text is drawn in a single blended box.
    Threshold is tight so stacked menu rows are not merged into one blob.
    """
    if len(results) != len(translated_texts) or not results:
        return results, translated_texts
    geos = [_bbox_geometry(_ocr_row_unpack(r)[1], img_w, img_h) for r in results]
    centers_x = [(g["x_min"] + g["x_max"]) / 2 for g in geos]
    centers_y = [(g["y_min"] + g["y_max"]) / 2 for g in geos]
    # Estimate common line tilt from OCR quads (top/bottom edges). This makes grouping robust
    # when side-by-side words share a slight slope (camera angle / perspective).
    angle_samples: List[float] = []
    for g in geos:
        pts = np.array(g["pts"], dtype=np.float32)
        if len(pts) >= 4:
            top = pts[1] - pts[0]
            bot = pts[2] - pts[3]
            if abs(float(top[0])) >= 1e-3:
                angle_samples.append(float(np.degrees(np.arctan2(top[1], top[0]))))
            if abs(float(bot[0])) >= 1e-3:
                angle_samples.append(float(np.degrees(np.arctan2(bot[1], bot[0]))))
    theta_deg = float(np.median(angle_samples)) if angle_samples else 0.0
    theta_deg = max(-24.0, min(24.0, theta_deg))
    theta = float(np.deg2rad(theta_deg))
    n_x, n_y = -float(np.sin(theta)), float(np.cos(theta))
    # Same slanted text line => similar projection on the normal axis.
    line_coords = [n_x * centers_x[i] + n_y * centers_y[i] for i in range(len(geos))]
    avg_h = sum(g["h"] for g in geos) / len(geos)
    # Was img_h * 0.125 — far too large; merged many separate menu lines into one region.
    # Tighter vertical grouping so stacked menu rows rarely merge into one floating block.
    line_threshold = max(avg_h * 0.38, img_h * line_threshold_frac * 0.42)
    order = sorted(range(len(results)), key=lambda i: line_coords[i])
    groups = []
    for i in order:
        lc = line_coords[i]
        if not groups or abs(lc - groups[-1]["center"]) > line_threshold:
            groups.append({"indices": [i], "center": lc})
        else:
            groups[-1]["indices"].append(i)
            groups[-1]["center"] = (groups[-1]["center"] * (len(groups[-1]["indices"]) - 1) + lc) / len(groups[-1]["indices"])
    merged_results = []
    merged_texts = []
    for g in groups:
        idx = g["indices"]
        x_min = min(geos[i]["x_min"] for i in idx)
        y_min = min(geos[i]["y_min"] for i in idx)
        x_max = max(geos[i]["x_max"] for i in idx)
        y_max = max(geos[i]["y_max"] for i in idx)
        merged_bbox = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        orig_parts = [results[i][0] for i in idx]
        trans_parts = [translated_texts[i] for i in idx]
        merged_orig = " ".join((o or "").strip() for o in orig_parts).strip()
        merged_trans = " ".join((t or "").strip() for t in trans_parts).strip()
        lang = _ocr_row_unpack(results[idx[0]])[2]
        confs = [_ocr_row_unpack(results[i])[3] for i in idx]
        merged_conf = max(confs) if confs else 0.5
        merged_results.append((merged_orig, merged_bbox, lang, merged_conf))
        merged_texts.append(merged_trans)
    return merged_results, merged_texts


def _video_translation_merge_key(t: str) -> str:
    """Normalize translated subtitle text so duplicate OCR lines (ghost + clean) cluster together."""
    s = " ".join((t or "").split())
    if not s:
        return ""
    k = _loose_text_key(s)
    return k if k else s.lower()


def _merge_video_duplicate_translation_boxes(
    results: List[Tuple],
    translated_texts: List[str],
    img_w: int,
    img_h: int,
) -> Tuple[List[Tuple], List[str]]:
    """
    Merge multiple boxes in one frame that carry the same translation (duplicate OCR / ghost line).
    Keeps one union bounding box and one draw pass so subtitles do not stack or smear.
    """
    if len(results) != len(translated_texts) or len(results) < 2:
        return results, translated_texts

    n = len(results)
    geos = [_bbox_geometry(_ocr_row_unpack(r)[1], img_w, img_h) for r in results]
    keys = [_video_translation_merge_key(translated_texts[i]) for i in range(n)]

    def spatially_close(i: int, j: int) -> bool:
        a, b = geos[i], geos[j]
        gap_y = max(0, max(a["y_min"], b["y_min"]) - min(a["y_max"], b["y_max"]))
        min_h = max(1, min(a["h"], b["h"]))
        # Stacked duplicate lines (ghost above/below real subtitle)
        if gap_y > max(16, min_h * 0.62):
            return False
        ox = min(a["x_max"], b["x_max"]) - max(a["x_min"], b["x_min"])
        min_w = max(1, min(a["w"], b["w"]))
        if ox < -max(10, min_w * 0.12):
            return False
        overlap_frac = ox / float(min_w)
        return overlap_frac > 0.18 or ox > min_w * 0.12

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        if not keys[i]:
            continue
        for j in range(i + 1, n):
            if keys[i] != keys[j] or not keys[j]:
                continue
            if spatially_close(i, j):
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    merged_results: List[Tuple] = []
    merged_texts: List[str] = []
    for _root, idxs in clusters.items():
        idxs = sorted(idxs, key=lambda ii: geos[ii]["y_min"])
        if len(idxs) == 1:
            i0 = idxs[0]
            merged_results.append(results[i0])
            merged_texts.append(translated_texts[i0])
            continue
        x_min = min(geos[i]["x_min"] for i in idxs)
        y_min = min(geos[i]["y_min"] for i in idxs)
        x_max = max(geos[i]["x_max"] for i in idxs)
        y_max = max(geos[i]["y_max"] for i in idxs)
        merged_bbox = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        orig_parts = [(results[i][0] or "").strip() for i in idxs]
        merged_orig = max(orig_parts, key=len) if orig_parts else ""
        merged_trans = " ".join((translated_texts[idxs[0]] or "").split())
        lang = _ocr_row_unpack(results[idxs[0]])[2]
        confs = [_ocr_row_unpack(results[i])[3] for i in idxs]
        merged_conf = max(confs) if confs else 0.5
        merged_results.append((merged_orig, merged_bbox, lang, merged_conf))
        merged_texts.append(merged_trans)

    return merged_results, merged_texts


def _video_adjacent_box_merge_enabled() -> bool:
    return os.environ.get("VIDEO_MERGE_ADJACENT_BOXES", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _merge_video_adjacent_boxes(
    results: List[Tuple],
    translated_texts: List[str],
    img_w: int,
    img_h: int,
) -> Tuple[List[Tuple], List[str]]:
    """
    Merge OCR boxes that sit beside each other (same row, small horizontal gap) or directly
    above/below (small vertical gap with horizontal overlap) so masks and cards do not overlap.

    Intended for **video** overlays after ``_merge_boxes_same_line`` / duplicate merge; disabled
    when env ``VIDEO_MERGE_ADJACENT_BOXES=0``.
    """
    if not _video_adjacent_box_merge_enabled():
        return results, translated_texts
    if len(results) != len(translated_texts) or len(results) < 2:
        return results, translated_texts

    n = len(results)
    geos = [_bbox_geometry(_ocr_row_unpack(r)[1], img_w, img_h) for r in results]

    def adjacent(i: int, j: int) -> bool:
        a, b = geos[i], geos[j]
        ha = max(1, int(a["h"]))
        hb = max(1, int(b["h"]))
        wa = max(1, int(a["w"]))
        wb = max(1, int(b["w"]))
        min_h = min(ha, hb)
        min_w = min(wa, wb)
        v_gap = max(0.0, max(float(a["y_min"]), float(b["y_min"])) - min(float(a["y_max"]), float(b["y_max"])))
        h_gap = max(0.0, max(float(a["x_min"]), float(b["x_min"])) - min(float(a["x_max"]), float(b["x_max"])))
        iy1 = max(a["y_min"], b["y_min"])
        iy2 = min(a["y_max"], b["y_max"])
        ix1 = max(a["x_min"], b["x_min"])
        ix2 = min(a["x_max"], b["x_max"])
        overlap_y = max(0.0, float(iy2 - iy1))
        overlap_x = max(0.0, float(ix2 - ix1))
        v_overlap_frac = overlap_y / float(min_h)
        h_overlap_frac = overlap_x / float(min_w)
        # Same row / beside: strong vertical overlap, modest horizontal gap
        beside = v_overlap_frac >= 0.26 and h_gap <= max(10.0, 0.52 * float(min_h))
        # Stacked lines: horizontal overlap, small vertical gap (two-line block, subtitles)
        stacked = h_overlap_frac >= 0.16 and v_gap <= max(8.0, 0.48 * float(min_h))
        # Light touch: small gap on one axis if the other shows alignment
        touch = (
            h_gap <= max(8.0, 0.20 * float(min_w)) and v_overlap_frac >= 0.12
        ) or (v_gap <= max(8.0, 0.22 * float(min_h)) and h_overlap_frac >= 0.12)
        return bool(beside or stacked or touch)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if adjacent(i, j):
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    merged_results: List[Tuple] = []
    merged_texts: List[str] = []
    for _root, idxs in clusters.items():
        idxs = sorted(idxs, key=lambda ii: (geos[ii]["y_min"], geos[ii]["x_min"]))
        if len(idxs) == 1:
            i0 = idxs[0]
            merged_results.append(results[i0])
            merged_texts.append(translated_texts[i0])
            continue
        x_min = min(geos[i]["x_min"] for i in idxs)
        y_min = min(geos[i]["y_min"] for i in idxs)
        x_max = max(geos[i]["x_max"] for i in idxs)
        y_max = max(geos[i]["y_max"] for i in idxs)
        merged_bbox = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        orig_parts = [(results[i][0] or "").strip() for i in idxs]
        trans_parts = [(translated_texts[i] or "").strip() for i in idxs]
        merged_orig = " ".join(p for p in orig_parts if p).strip()
        merged_trans = " ".join(p for p in trans_parts if p).strip()
        lang = _ocr_row_unpack(results[idxs[0]])[2]
        confs = [_ocr_row_unpack(results[i])[3] for i in idxs]
        merged_conf = max(confs) if confs else 0.5
        merged_results.append((merged_orig, merged_bbox, lang, merged_conf))
        merged_texts.append(merged_trans)

    return merged_results, merged_texts


def _expand_bbox(
    bbox: List[List[int]],
    img_w: int,
    img_h: int,
    padding_px: int = DEFAULT_MASK_PADDING_PX,
    padding_ratio: float = DEFAULT_MASK_PADDING_RATIO,
) -> np.ndarray:
    """
    Expand bounding box by padding so the mask fully covers the original text.
    Returns 4-point polygon (int) clipped to image bounds.
    """
    pts = np.array(bbox, dtype=np.float32)
    if len(pts) < 4:
        x_min = int(np.min(pts[:, 0])) if len(pts) else 0
        y_min = int(np.min(pts[:, 1])) if len(pts) else 0
        x_max = int(np.max(pts[:, 0])) if len(pts) else 0
        y_max = int(np.max(pts[:, 1])) if len(pts) else 0
        pts = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
    x_min = float(np.min(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    x_max = float(np.max(pts[:, 0]))
    y_max = float(np.max(pts[:, 1]))
    w = max(1, x_max - x_min)
    h = max(1, y_max - y_min)
    pad_x = max(padding_px, int(w * padding_ratio))
    pad_y = max(padding_px, int(h * padding_ratio))
    expanded = np.array([
        [x_min - pad_x, y_min - pad_y],
        [x_max + pad_x, y_min - pad_y],
        [x_max + pad_x, y_max + pad_y],
        [x_min - pad_x, y_max + pad_y],
    ], dtype=np.int32)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, img_w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, img_h - 1)
    return expanded


def _rect_to_bbox_4pt(x: int, y: int, w: int, h: int) -> np.ndarray:
    """Convert (x, y, w, h) to 4-point polygon (top-left, top-right, bottom-right, bottom-left)."""
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)


def _create_tracker():
    """Create an OpenCV tracker for text region tracking across frames. Prefer KCF (fast, no contrib)."""
    for name in ("KCF", "CSRT", "MIL"):
        try:
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create") and name == "KCF":
                return cv2.legacy.TrackerKCF_create()
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create") and name == "CSRT":
                return cv2.legacy.TrackerCSRT_create()
            if hasattr(cv2, "TrackerKCF_create"):
                return cv2.TrackerKCF_create()
            if hasattr(cv2, "TrackerCSRT_create"):
                return cv2.TrackerCSRT_create()
        except Exception:
            continue
    return None


def _inpaint_region_bgr(img_bgr: np.ndarray, pts: np.ndarray, inpaint_radius: Optional[int] = None) -> np.ndarray:
    """Remove original text by inpainting the polygon region. Modifies a copy and returns it."""
    img = img_bgr.copy()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    radius = inpaint_radius if inpaint_radius is not None else max(3, min(img.shape[:2]) // 100)
    img = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    return img


def _estimate_font_size_from_original_text(
    orig_text: str,
    box_w: int,
    box_h: int,
    get_latin_font_fn,
) -> Optional[int]:
    """
    Largest font size (px) at which the source text fits in the OCR box using a Latin font —
    approximates the original printed size so translations can use the same point size.
    """
    t = (orig_text or "").strip()
    if not t:
        return None
    t = " ".join(t.split())
    if len(t) > 240:
        t = t[:240]
    usable_w = max(20, int(box_w * 0.90))
    usable_h = max(12, int(box_h * 0.86))
    hi = min(120, max(box_h + 8, 18))
    for fs in range(hi, 7, -1):
        try:
            font = get_latin_font_fn(fs)
            if font is None:
                continue
            if hasattr(font, "getbbox"):
                bbox = font.getbbox(t)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            else:
                tw, th = len(t) * fs * 0.45, fs
            if tw <= usable_w and th <= usable_h:
                return fs
        except Exception:
            continue
    # No Latin fit (unusual): use row-height heuristic so we still pass a target size downstream
    return max(8, min(120, int(box_h * 0.68)))


def _refine_target_font_size_for_translation(
    display_text: str,
    box_w: int,
    box_h: int,
    start_fs: int,
    prefer_arabic: bool,
    get_font_by_size,
) -> int:
    """Lower the English-derived target only if the translation does not fit at that size in its own font."""
    t = (display_text or "").strip()
    if not t:
        return max(8, min(120, int(start_fs)))
    usable_w = max(20, int(box_w * 0.90))
    usable_h = max(12, int(box_h * 0.86))
    cap = min(120, max(8, int(start_fs)))
    disp = _arabic_display_for_pil(t) if prefer_arabic and _is_arabic_or_rtl(t) else t
    for fs in range(cap, 7, -1):
        try:
            font = get_font_by_size(fs)
            if font is None:
                continue
            if hasattr(font, "getbbox"):
                bbox = font.getbbox(disp)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            else:
                tw, th = len(t) * fs * 0.45, fs
            if tw <= usable_w and th <= usable_h:
                return fs
        except Exception:
            continue
    return 8


def _font_bbox_lt(font, text: str) -> Tuple[int, int, int, int]:
    """Ink bbox consistent with ImageDraw.text(..., anchor='lt') in Pillow 10+."""
    try:
        return font.getbbox(text, anchor="lt")
    except TypeError:
        try:
            return font.getbbox(text)
        except Exception:
            return (0, 0, max(8, len(text) * 6), 12)


def _line_width_pil(font, logical_line: str, needs_bidi: bool) -> int:
    d = _arabic_display_for_pil(logical_line) if needs_bidi else logical_line
    bb = _font_bbox_lt(font, d)
    return max(0, int(bb[2] - bb[0]))


def _break_overflow_word(word: str, font, usable_w: int, needs_bidi: bool) -> List[str]:
    if not word:
        return []
    parts: List[str] = []
    chunk = ""
    for ch in word:
        trial = chunk + ch
        if _line_width_pil(font, trial, needs_bidi) <= usable_w or not chunk:
            chunk = trial
        else:
            parts.append(chunk)
            chunk = ch
    if chunk:
        parts.append(chunk)
    return parts if parts else [word]


def _wrap_logical_text_to_lines(logical_text: str, font, usable_w: int, needs_bidi: bool) -> List[str]:
    """Greedy wrap; measure each line after full-line Arabic shaping (avoids overlapping glyphs)."""
    logical_text = (logical_text or "").strip()
    if not logical_text:
        return []
    words = logical_text.split()
    if not words:
        return [logical_text]
    n = len(words)
    lines: List[str] = []
    i = 0
    while i < n:
        j = i + 1
        while j <= n:
            trial = " ".join(words[i:j])
            if _line_width_pil(font, trial, needs_bidi) <= usable_w:
                j += 1
            else:
                break
        if j == i + 1 and _line_width_pil(font, words[i], needs_bidi) > usable_w:
            for part in _break_overflow_word(words[i], font, usable_w, needs_bidi):
                lines.append(part)
            i += 1
            continue
        lines.append(" ".join(words[i : j - 1]))
        i = j - 1
    return lines if lines else [logical_text]


def _line_row_height(
    font,
    disp_line: str,
    font_size_px: int,
    *,
    outline: bool,
) -> int:
    """
    Vertical advance for one drawn row: ink bbox + outline slack + gap so wrapped lines never overlap.
    Raw getbbox height alone was too tight vs 8-way outline + Arabic vertical extent.
    """
    bb = _font_bbox_lt(font, disp_line)
    ink = max(1, int(bb[3] - bb[1]))
    outline_slack = 5 if outline else 2
    gap = max(6, int(font_size_px * 0.20))
    try:
        ascent, descent = font.getmetrics()
        em_h = int(ascent) + int(descent)
    except Exception:
        em_h = int(font_size_px * 1.08)
    return max(ink + outline_slack + gap, em_h + (gap * 2) // 3 + outline_slack, int(font_size_px * 1.14) + 6, 16)


def _wrapped_block_height(
    lines: List[str],
    font,
    needs_bidi: bool,
    font_size_hint: int,
    outline: bool,
) -> Tuple[List[int], int]:
    heights: List[int] = []
    for ln in lines:
        d = _arabic_display_for_pil(ln) if needs_bidi else ln
        heights.append(_line_row_height(font, d, font_size_hint, outline=outline))
    return heights, sum(heights)


def _render_text_into_box_pillow(
    text: str,
    box_w: int,
    box_h: int,
    get_font_fn,
    is_rtl: bool,
    text_color: Tuple[int, int, int] = (0, 0, 0),
    bg_color: Optional[Tuple[int, int, int]] = None,
    for_video: bool = False,
    target_font_size: Optional[int] = None,
    style_like_rtl: bool = False,
    *,
    force_outline: bool = False,
) -> Tuple[Image.Image, Optional[ImageFont.ImageFont]]:
    """
    Render text into a PIL Image of size (box_w, box_h). Prefers one line; wraps only when
    text is long. Text is vertically centered in the box to match original placement.
    If target_font_size is set (static overlay), start at that size and only shrink if needed.
    Returns (PIL Image of size box_w x box_h, font used).
    If ``force_outline`` is True, draw an 8-way stroke for readability (e.g. green OCR preview on photos).
    """
    logical_text = (text or "").strip()
    if not logical_text:
        out = Image.new("RGBA", (max(1, box_w), max(1, box_h)), (255, 255, 255, 0))
        return out, None
    needs_bidi = _is_arabic_or_rtl(logical_text)

    def _disp(s: str) -> str:
        return _arabic_display_for_pil(s) if needs_bidi else s
    # Video: minimum size for readability. Photos: small floor only — inflating to 24×24 vs a tight OCR
    # box made the layer larger than the erased region and triggered blend shifts / stray English.
    box_w = max(box_w, 40 if for_video else 8)
    box_h = max(box_h, 38 if for_video else 8)
    usable_w = max(20, int(box_w * 0.90))
    usable_h = max(12, int(box_h * 0.86))
    # Latin loanwords on Arabic-target photos: same stroke weight and scale as Arabic overlay lines.
    use_outline = (
        for_video
        or (needs_bidi and not for_video)
        or (style_like_rtl and not for_video)
        or force_outline
    )
    # Video: smaller type. Static Arabic: larger cap + scale so glyphs stay crisp (was often tiny/blurry).
    min_font = 8 if for_video else 12
    if for_video:
        max_scale, floor_cap = 0.28, 8
    elif needs_bidi or (style_like_rtl and not for_video):
        max_scale, floor_cap = 0.78, 22
    else:
        max_scale, floor_cap = 0.6, 14
    auto_max = min(120, max(floor_cap, int(min(box_w, box_h) * max_scale)))
    if for_video:
        # User-tunable shrink/grow for video subtitles (default 0.75 → ~25% smaller).
        translated_font_scale = _video_translated_font_scale()
    else:
        translated_font_scale = _photo_translated_font_scale()
    if target_font_size is not None and not for_video:
        # Match original English size; do not inflate above estimate
        max_font = max(
            min_font,
            min(120, int(round(float(target_font_size) * translated_font_scale))),
        )
    else:
        max_font = max(min_font, min(120, int(round(auto_max * translated_font_scale))))
    # When preserving size, use the same tight height rule as _estimate_font_size_from_original_text
    # (Phase 1 used th*1.2 which always forced Arabic smaller than the English estimate.)
    lock_target = target_font_size is not None and not for_video
    font_size = max_font
    font = None
    lines_final = [logical_text]
    single_line_fit = False

    if lock_target:
        fit_ok = False
        fs_try = max_font
        while fs_try >= min_font:
            font = get_font_fn(fs_try)
            if font is None:
                font = get_font_fn(min_font)
                if font is None:
                    break
            lines = _wrap_logical_text_to_lines(logical_text, font, usable_w, needs_bidi)
            _, total_h = _wrapped_block_height(lines, font, needs_bidi, fs_try, use_outline)
            if total_h <= usable_h:
                lines_final = lines
                single_line_fit = len(lines) == 1
                font_size = fs_try
                fit_ok = True
                break
            fs_try -= 1
        if not fit_ok:
            font_size = min_font
            font = get_font_fn(font_size)
            if font is None:
                font = get_font_fn(min_font)
            if font is not None:
                lines_final = _wrap_logical_text_to_lines(logical_text, font, usable_w, needs_bidi)
                single_line_fit = len(lines_final) == 1
    else:
        # Arabic/RTL: never use the single-line "fit" pass with raw getbbox — it often under-estimates
        # width vs anchor='lt' drawing and glyphs stack on one line. Always run the wrapper below.
        if needs_bidi:
            single_line_fit = False
        else:
            while font_size >= min_font:
                font = get_font_fn(font_size)
                if font is None:
                    font = get_font_fn(min_font)
                    if font is None:
                        break
                disp_full = _disp(logical_text)
                try:
                    bb = _font_bbox_lt(font, disp_full)
                    tw = bb[2] - bb[0]
                    th = bb[3] - bb[1]
                except Exception:
                    tw, th = len(logical_text) * font_size // 2, font_size
                line_height = int(th * 1.15) if th else font_size + 2
                if tw <= usable_w and line_height <= usable_h:
                    lines_final = [logical_text]
                    single_line_fit = True
                    break
                font_size -= 1
        if not single_line_fit:
            font_size = max_font
            while font_size >= min_font:
                font = get_font_fn(font_size)
                if font is None:
                    font = get_font_fn(min_font)
                    if font is None:
                        break
                lines = _wrap_logical_text_to_lines(logical_text, font, usable_w, needs_bidi)
                _, total_h = _wrapped_block_height(lines, font, needs_bidi, font_size, use_outline)
                if total_h <= usable_h:
                    lines_final = lines
                    break
                font_size -= 1
            if not single_line_fit and lines_final == [logical_text] and font is not None:
                lines_final = _wrap_logical_text_to_lines(logical_text, font, usable_w, needs_bidi)
    if not lines_final:
        lines_final = [logical_text]
    if font is None:
        font = get_font_fn(min_font)
    if font is None:
        out = Image.new("RGBA", (max(1, box_w), max(1, box_h)), (255, 255, 255, 0) if bg_color is None else (*bg_color, 255))
        return out, None
    max_block_h = max(8, box_h - 6)

    def _line_stack_height(fs: int) -> Tuple[List[int], int]:
        f = get_font_fn(fs)
        if f is None:
            return [], 99999
        hs = []
        for line in lines_final:
            disp_line = _disp(line)
            try:
                hs.append(_line_row_height(f, disp_line, fs, outline=use_outline))
            except Exception:
                hs.append(max(fs + 12, 18))
        return hs, sum(hs)

    # Shrink font until measured glyph stack fits. Skipped for single-line preserve mode.
    # Wrapped preserve mode: still shrink if the Arabic stack is taller than the box (looser bound).
    if not lock_target:
        _vf = 0
        while font_size >= min_font and _vf < 60:
            _vf += 1
            hs, total = _line_stack_height(font_size)
            if not hs:
                break
            if total <= max_block_h:
                break
            font_size -= 1
    elif not single_line_fit:
        max_block_h_wrap = max(8, box_h - 2)
        _vf = 0
        while font_size >= min_font and _vf < 60:
            _vf += 1
            hs, total = _line_stack_height(font_size)
            if not hs:
                break
            if total <= max_block_h_wrap:
                break
            font_size -= 1
    font = get_font_fn(font_size)
    if font is None:
        font = get_font_fn(min_font)
    lines_final = [x.strip() for x in lines_final if (x or "").strip()]
    if not lines_final:
        lines_final = [(logical_text or "").strip() or "..."]
    heights, _ = _line_stack_height(font_size)
    if not heights:
        heights = [max(10, font_size + 4)]

    # Build image: same box size, vertically centered text. Use opaque background so blend shows text clearly.
    bg_rgba = (255, 255, 255, 0) if bg_color is None else (*bg_color, 255)
    out = Image.new("RGBA", (box_w, box_h), bg_rgba)
    draw = ImageDraw.Draw(out)
    total_block_h = sum(heights)
    # Video: compact top for subtitles. Photos: optionally center the block in the OCR box (default on)
    # so text aligns with typical centered menu labels; set PHOTO_TEXT_CENTER_IN_BOX=0 for legacy top/edge.
    _center_photo = (
        not for_video
        and os.environ.get("PHOTO_TEXT_CENTER_IN_BOX", "1").strip().lower()
        not in ("0", "false", "no")
    )
    if for_video:
        if len(lines_final) == 1:
            y_pos = 2
            if y_pos + total_block_h > box_h - 2:
                y_pos = max(2, box_h - total_block_h - 2)
        else:
            y_pos = max(2, (box_h - total_block_h) // 2)
            if y_pos + total_block_h > box_h - 2:
                y_pos = max(2, box_h - total_block_h - 2)
            y_pos = max(2, y_pos)
    elif _center_photo:
        y_pos = max(2, (box_h - total_block_h) // 2)
        if y_pos + total_block_h > box_h - 2:
            y_pos = max(2, box_h - total_block_h - 2)
    else:
        if len(lines_final) == 1:
            y_pos = 2
            if y_pos + total_block_h > box_h - 2:
                y_pos = max(2, box_h - total_block_h - 2)
        else:
            y_pos = max(2, (box_h - total_block_h) // 2)
            if y_pos + total_block_h > box_h - 2:
                y_pos = max(2, box_h - total_block_h - 2)
            y_pos = max(2, y_pos)
    outline_rgba = (0, 0, 0, 255) if (text_color[0] > 200) else (255, 255, 255, 255)
    text_rgba = (*text_color, 255) if len(text_color) == 3 else text_color

    def _draw_text_lt(xy: Tuple[int, int], s: str, fill: Tuple[int, ...]) -> None:
        # anchor="lt" may fail on some Pillow/FreeType pairs; unanchored is slightly misaligned but visible.
        try:
            draw.text(xy, s, font=font, fill=fill, anchor="lt")
        except (TypeError, ValueError, OSError):
            draw.text(xy, s, font=font, fill=fill)

    for i, line in enumerate(lines_final):
        if i >= len(heights):
            break
        line_h = heights[i]
        # Rows are 0..box_h-1; a line of height line_h at y_pos uses y_pos..y_pos+line_h-1.
        # Must not break when y_pos+line_h==box_h (previous off-by-one skipped all draws in tight boxes).
        if y_pos + line_h > box_h:
            break
        disp_line = _disp(line)
        bb = _font_bbox_lt(font, disp_line)
        lw = max(1, int(bb[2] - bb[0]))
        if _center_photo:
            x_pos = max(0, (box_w - lw) // 2)
            x_pos = min(x_pos, max(0, box_w - lw - 1))
        else:
            x_pos = 4 if not is_rtl else max(4, box_w - lw - 4)
            x_pos = max(0, min(x_pos, box_w - lw - 1))
        if for_video or (needs_bidi and not for_video) or (style_like_rtl and not for_video):
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                _draw_text_lt((x_pos + dx, y_pos + dy), disp_line, outline_rgba)
        _draw_text_lt((x_pos, y_pos), disp_line, text_rgba)
        y_pos += line_h
    return out, font


def _blend_rgba_on_bgr(img_bgr: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    """Composite RGBA layer (PIL RGB order) onto BGR image at (x, y). Modifies img_bgr in place.
    Clips to image bounds at the OCR anchor so overlays are not slid sideways (which looked misaligned)."""
    h_l, w_l = rgba.shape[:2]
    h_i, w_i = img_bgr.shape[:2]
    if w_l <= 0 or h_l <= 0:
        return
    x0, y0 = x, y
    x1, y1 = x0 + w_l, y0 + h_l
    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(w_i, x1)
    iy1 = min(h_i, y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return
    sx0, sy0 = ix0 - x0, iy0 - y0
    sw, sh = ix1 - ix0, iy1 - iy0
    rgba = rgba[sy0 : sy0 + sh, sx0 : sx0 + sw]
    roi = img_bgr[iy0:iy1, ix0:ix1]
    if rgba.shape[2] == 4:
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        rgb = rgba[:, :, :3]
    else:
        alpha = np.ones((roi.shape[0], roi.shape[1], 1), dtype=np.float32)
        rgb = rgba[:, :, :3]
    if roi.shape[:2] != rgb.shape[:2]:
        return
    fg_bgr = rgb[:, :, ::-1].astype(np.float32)
    # Photo realism: feather alpha and slightly integrate text color with local background.
    # This reduces hard sticker-like edges while keeping readability.
    realistic = os.environ.get("PHOTO_REALISTIC_BLEND", "1").strip().lower() not in ("0", "false", "no")
    if realistic and rgba.shape[2] == 4:
        try:
            a2d = alpha[:, :, 0]
            a2d = cv2.GaussianBlur(a2d, (3, 3), 0)
            alpha = np.clip(a2d[:, :, np.newaxis] * 0.96, 0.0, 1.0)
            bg_mean = roi.reshape(-1, 3).mean(axis=0).astype(np.float32)
            fg_bgr = fg_bgr * 0.92 + bg_mean[np.newaxis, np.newaxis, :] * 0.08
        except Exception:
            pass
    roi[:] = (alpha * fg_bgr + (1 - alpha) * roi.astype(np.float32)).astype(np.uint8)


def _warp_text_layer_to_polygon(
    text_layer: np.ndarray,
    src_rect_wh: Tuple[int, int],
    dst_pts: np.ndarray,
) -> Tuple[np.ndarray, int, int, int, int]:
    """
    Warp a rectangular text layer to fit the 4-point polygon.
    src_rect_wh = (width, height) of text_layer.
    dst_pts = 4 points (x,y) in image coords. Order: top-left, top-right, bottom-right, bottom-left.
    Returns (warped BGR/RGBA image, x_min, y_min, x_max, y_max) for paste region.
    """
    w, h = src_rect_wh
    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    if len(dst_pts) >= 4:
        # OCR engines may return quad points in arbitrary order.
        # Normalize to tl,tr,br,bl so perspective mapping keeps the original angle/layout.
        dst_pts = _order_quad_points(dst_pts[:4])
    x_min, y_min = int(dst_pts[:, 0].min()), int(dst_pts[:, 1].min())
    x_max, y_max = int(dst_pts[:, 0].max()), int(dst_pts[:, 1].max())
    out_w = max(1, x_max - x_min)
    out_h = max(1, y_max - y_min)
    if len(dst_pts) != 4:
        if text_layer is None or text_layer.size == 0 or out_w <= 0 or out_h <= 0:
            return text_layer, x_min, y_min, x_max, y_max
        resized = _safe_cv2_resize(text_layer, out_w, out_h)
        return resized, x_min, y_min, x_max, y_max
    # Destination in local coords of the output crop
    dst_local = dst_pts - np.array([x_min, y_min], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_local)
    warped = cv2.warpPerspective(text_layer, M, (out_w, out_h), borderMode=cv2.BORDER_TRANSPARENT)
    return warped, x_min, y_min, x_max, y_max


class OCRTranslator:
    """Main class for OCR and translation operations."""
    
    def __init__(
        self,
        languages: List[str] = ['en'],
        force_cpu: bool = False,
        backend: str = "easyocr",
        trocr_model: Optional[str] = None,
    ):
        """
        Initialize the OCR reader.
        
        Args:
            languages: List of language codes for OCR (e.g., ['en', 'es', 'fr', 'de', 'zh'])
            force_cpu: If True, disable CUDA and initialize EasyOCR on CPU.
            backend: OCR backend: "easyocr", "paddleocr", "trocr_hybrid" (EasyOCR detect + TrOCR recognize),
                or "trocr_only" (TrOCR-only detection + recognition).
            trocr_model: Hugging Face model id for TrOCR (trocr_hybrid). Default: base handwritten or OCR_TROCR_MODEL env.
        """
        self.backend = (backend or "easyocr").strip().lower()
        use_gpu = _ocr_use_gpu() and not force_cpu
        # PaddleOCR: subprocess worker; GPU when CUDA (see _paddle_ocr_use_gpu), independent of OCR_USE_GPU.
        self._paddle_use_gpu = _paddle_ocr_use_gpu() and not force_cpu
        self._paddle_lang = "en"
        self._paddle_langs = ["en"]
        self._force_cpu_ocr = bool(force_cpu)
        self.trocr_model_name = (trocr_model or os.environ.get("OCR_TROCR_MODEL", "microsoft/trocr-base-handwritten") or "microsoft/trocr-base-handwritten").strip()
        self._trocr_processor = None
        self._trocr_model = None
        self._trocr_device = None
        self.last_ocr_error: Optional[str] = None
        self.last_ocr_preprocess_profile: Dict[str, Any] = {}
        self.openai_preprocess: bool = _ocr_env_truthy("OCR_PREPROCESS_OPENAI", False)
        self.local_preprocess: bool = True
        self.openai_api_key: Optional[str] = None
        self.last_video_render_profile: Dict[str, Any] = {}
        self._ocr_infer_generate_seconds_last = 0.0
        self._trocr_detect_cache_sig: Optional[Tuple[Any, ...]] = None
        self._trocr_detect_cache_lists: Optional[Tuple[List[Any], List[Any]]] = None
        _dev_tag = " (GPU)" if (use_gpu if self.backend != "paddleocr" else self._paddle_use_gpu) else " (CPU)"
        print(
            "Initializing OCR reader... This may take a moment on first run."
            + _dev_tag
        )
        if self.backend == "paddleocr":
            # Keep PaddleOCR isolated in a subprocess to avoid native DLL crashes
            # when Paddle and Torch coexist in one Windows process.
            self._paddle_langs = _paddle_langs_from_codes(languages)
            self._paddle_lang = self._paddle_langs[0]
            self.reader = None
            self.detected_language = languages[0] if languages else "en"
            if len(self._paddle_langs) > 1:
                parallel_n = _paddle_multilang_max_workers(len(self._paddle_langs))
                mode = "sequential" if parallel_n <= 1 else f"parallel x{parallel_n}"
                print(
                    f"PaddleOCR: OCR languages {self._paddle_langs} — "
                    f"one model pass each, merged ({mode}; daemon disabled for multi-lang). "
                    "Set OCR_PADDLE_PARALLEL_WORKERS=N (N>=2) to run concurrently."
                )
        else:
            if self.backend not in ("easyocr", "trocr_hybrid", "trocr_only"):
                self.backend = "easyocr"
            # trocr_only intentionally avoids EasyOCR initialization.
            self.reader = None if self.backend == "trocr_only" else easyocr.Reader(languages, gpu=use_gpu)
            self.detected_language = languages[0] if languages else 'en'
        if self.backend == "trocr_hybrid":
            print(
                f"TrOCR handwriting mode: detection=EasyOCR, recognition={self.trocr_model_name} (loaded on first use)."
            )
        elif self.backend == "trocr_only":
            print(
                f"TrOCR-only mode: detection=OpenCV line proposals, recognition={self.trocr_model_name} (loaded on first use)."
            )
        print(f"OCR reader initialized successfully! backend={self.backend}")

    def _ocr_wants_preprocess(self, local_preprocess_flag: bool) -> bool:
        """True when local and/or OpenAI preprocessing should run before OCR."""
        return bool(local_preprocess_flag) or bool(getattr(self, "openai_preprocess", False))

    def _normalize_paddle_output(self, output: Any) -> List[Tuple[Any, str, float]]:
        """Normalize PaddleOCR output to EasyOCR-like tuples: (bbox, text, conf)."""
        out: List[Tuple[Any, str, float]] = []

        def walk(node: Any) -> None:
            if not isinstance(node, (list, tuple)):
                return
            # Expected line format: [bbox4, (text, score)]
            if len(node) == 2 and _looks_like_quad(node[0]):
                text = ""
                score = 0.0
                rec = node[1]
                if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    text = str(rec[0] or "")
                    try:
                        score = float(rec[1] or 0.0)
                    except (TypeError, ValueError):
                        score = 0.0
                elif isinstance(rec, str):
                    text = rec
                if text.strip():
                    out.append((node[0], text, score))
                return
            for child in node:
                walk(child)

        walk(output)
        return out

    def _paddle_worker_oneshot(
        self,
        tmp_path: str,
        lang: str,
        env: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[Tuple[Any, str, float]], Optional[str]]:
        """Run ``paddle_ocr_worker.py`` once with ``lang``. Returns ``(rows, error_message)`` (daemon-safe / thread-safe)."""
        worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paddle_ocr_worker.py")
        cmd = [
            sys.executable,
            worker_path,
            "--image",
            tmp_path,
            "--lang",
            lang,
            "--use-gpu",
            "1" if getattr(self, "_paddle_use_gpu", True) else "0",
        ]
        try:
            timeout_sec = int(os.environ.get("OCR_PADDLE_TIMEOUT_SEC", "180"))
        except ValueError:
            timeout_sec = 180
        tmo = max(15, timeout_sec)
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=tmo,
                env=env if env is not None else _paddle_subprocess_env(),
            )
        except subprocess.TimeoutExpired:
            return [], (
                f"PaddleOCR worker timed out after {tmo}s lang={lang} "
                "(increase OCR_PADDLE_TIMEOUT_SEC or try CPU mode)."
            )
        if r.returncode != 0:
            err = (r.stderr or "").strip()
            return [], (
                f"PaddleOCR worker failed (exit code {r.returncode}) lang={lang}. "
                + (f"Details: {err}" if err else "No stderr output.")
            )
        return _parse_paddle_worker_json_list(r.stdout or ""), None

    def _run_paddle_ocr_subprocess(
        self,
        img: np.ndarray,
        *,
        small_text_boost: bool = False,
        high_recall: bool = False,
    ) -> List[Tuple[Any, str, float]]:
        """Run PaddleOCR in worker process(es); merge when several Paddle langs are selected."""
        self.last_ocr_error = None
        tmp_path = None
        try:
            timeout_sec = int(os.environ.get("OCR_PADDLE_TIMEOUT_SEC", "180"))
        except ValueError:
            timeout_sec = 180
        langs = getattr(self, "_paddle_langs", [self._paddle_lang])
        sub_env = _paddle_subprocess_env(
            small_text_boost=small_text_boost, high_recall=high_recall
        )
        paddle_boost_run = bool(small_text_boost or high_recall)
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            if not cv2.imwrite(tmp_path, img):
                self.last_ocr_error = "PaddleOCR worker: failed to write temp image."
                return []
            worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paddle_ocr_worker.py")
            if not os.path.isfile(worker_path):
                self.last_ocr_error = "PaddleOCR worker script not found: paddle_ocr_worker.py"
                return []

            if len(langs) == 1:
                if _paddle_daemon_env_enabled() and not paddle_boost_run:
                    proc = _ensure_paddle_daemon(self)
                    if proc is not None:
                        try:
                            return _paddle_daemon_request(proc, tmp_path)
                        except Exception as e:
                            self.last_ocr_error = f"PaddleOCR daemon request failed: {e}"
                            _paddle_invalidate_daemon()
                rows, err = self._paddle_worker_oneshot(tmp_path, langs[0], env=sub_env)
                if err:
                    self.last_ocr_error = err
                return rows

            combined: List[Tuple[Any, str, float]] = []
            errs: List[str] = []
            max_w = _paddle_multilang_max_workers(len(langs))
            if max_w <= 1:
                for lang in langs:
                    rows, err = self._paddle_worker_oneshot(tmp_path, lang, env=sub_env)
                    if err:
                        errs.append(err)
                    combined.extend(rows)
            else:
                with ThreadPoolExecutor(max_workers=max_w) as ex:
                    futures = {
                        ex.submit(self._paddle_worker_oneshot, tmp_path, lg, sub_env): lg
                        for lg in langs
                    }
                    for fut in as_completed(futures):
                        lg = futures[fut]
                        try:
                            rows, err = fut.result()
                        except Exception as e:
                            errs.append(f"{lg}: {e}")
                            continue
                        if err:
                            errs.append(err)
                        combined.extend(rows)
            merged = _merge_overlapping_easyocr_results(combined)
            if errs:
                self.last_ocr_error = None if merged else "; ".join(errs[:4])
            return merged
        except subprocess.TimeoutExpired:
            self.last_ocr_error = (
                f"PaddleOCR worker timed out after {max(15, timeout_sec)}s "
                "(increase OCR_PADDLE_TIMEOUT_SEC or try CPU mode)."
            )
            return []
        except Exception as e:
            self.last_ocr_error = f"PaddleOCR worker exception: {e}"
            return []
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _enhance_bgr_for_ocr(self, img_bgr: np.ndarray) -> np.ndarray:
        """Resize sampled video frames (caps long side via ``OCR_VIDEO_PREPROCESS_MAX_SIDE``)."""
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr
        out = enhance_bgr_video_frame(
            img_bgr,
            thin_strokes=False,
            force_cpu=self._force_cpu_ocr,
        )
        self.last_ocr_preprocess_profile = last_preprocess_profile()
        return out

    def preprocess_image_from_bgr(
        self,
        img_bgr: np.ndarray,
        max_dim: int = 1200,
        *,
        thin_strokes: bool = False,
    ) -> np.ndarray:
        """Resize an in-memory BGR image to the OCR working size (avoids duplicate disk reads)."""
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Empty image array for preprocessing.")
        oai_key = (getattr(self, "openai_api_key", None) or "").strip() or None
        out, prof = enhance_bgr_for_ocr(
            img_bgr,
            max_dim=max_dim,
            thin_strokes=thin_strokes,
            force_cpu=self._force_cpu_ocr,
            openai_preprocess=getattr(self, "openai_preprocess", False),
            local_preprocess=getattr(self, "local_preprocess", True),
            openai_api_key=oai_key,
        )
        self.last_ocr_preprocess_profile = prof if prof else last_preprocess_profile()
        return out

    def preprocess_image(self, image_path: str, max_dim: int = 1200) -> np.ndarray:
        """Load from disk and resize for EasyOCR (longest side ``max_dim``)."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        return self.preprocess_image_from_bgr(img, max_dim=max_dim, thin_strokes=False)

    def preprocess_image_skip_denoise(self, image_path: str, max_dim: int = 1200) -> np.ndarray:
        """Same resize path as ``preprocess_image``; ``thin_strokes=True`` kept for API compatibility."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        return self.preprocess_image_from_bgr(img, max_dim=max_dim, thin_strokes=True)

    def preprocess_preview_rgb_pair(
        self,
        image_path: str,
        preprocess: bool,
        max_dim: int,
        small_text_boost: bool = False,
        high_recall: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        RGB previews for Streamlit — same resize budget as ``_extract_text_core``.

        Returns ``(before_rgb, after_final_rgb, after_openai_rgb)``.
        After OCR, reads cached frames from ``enhance_bgr_for_ocr`` (no second OpenAI call).
        """
        img_orig = cv2.imread(image_path)
        if img_orig is None:
            raise ValueError(f"Could not read image from {image_path}")
        eff_dim = _ocr_effective_max_dim(max_dim, small_text_boost, high_recall=high_recall)
        before_bgr = _resize_bgr_to_max_dim(img_orig, eff_dim)
        before_rgb = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2RGB)
        use_openai = bool(getattr(self, "openai_preprocess", False))
        use_local = bool(preprocess)
        if not use_openai and not use_local:
            return before_rgb, None, None

        cache = last_preprocess_preview()
        c_before = cache.get("before_bgr")
        c_final = cache.get("after_final_bgr")
        c_oai = cache.get("after_openai_bgr")
        if (
            c_final is not None
            and c_before is not None
            and c_before.shape[:2] == before_bgr.shape[:2]
        ):
            before_rgb = cv2.cvtColor(c_before, cv2.COLOR_BGR2RGB)
            after_final_rgb = cv2.cvtColor(c_final, cv2.COLOR_BGR2RGB)
            after_openai_rgb = (
                cv2.cvtColor(c_oai, cv2.COLOR_BGR2RGB) if c_oai is not None and c_oai.size else None
            )
            return before_rgb, after_final_rgb, after_openai_rgb

        saved_oai = bool(getattr(self, "openai_preprocess", False))
        saved_local = bool(getattr(self, "local_preprocess", True))
        self.openai_preprocess = use_openai
        self.local_preprocess = use_local
        try:
            after_bgr = self.preprocess_image_from_bgr(
                before_bgr, max_dim=eff_dim, thin_strokes=False
            )
        finally:
            self.openai_preprocess = saved_oai
            self.local_preprocess = saved_local
        after_final_rgb = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB)
        cache2 = last_preprocess_preview()
        c_oai2 = cache2.get("after_openai_bgr")
        after_openai_rgb = (
            cv2.cvtColor(c_oai2, cv2.COLOR_BGR2RGB) if c_oai2 is not None and c_oai2.size else None
        )
        return before_rgb, after_final_rgb, after_openai_rgb

    def _ensure_trocr(self) -> None:
        """Lazy-load TrOCR (transformers) for trocr_hybrid backend."""
        if self._trocr_model is not None:
            return
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as e:
            raise ImportError(
                "TrOCR requires `transformers` and `torch` (already in requirements). "
                "Install or upgrade: pip install transformers torch"
            ) from e
        if self._force_cpu_ocr or not torch.cuda.is_available():
            device = torch.device("cpu")
            if self._force_cpu_ocr and torch.cuda.is_available():
                print(
                    "[trocr] CPU OCR is forced — TrOCR will not use CUDA (slow). "
                    "Unset force-cpu / OCR_FORCE_CPU if you want GPU inference.",
                    flush=True,
                )
            elif not self._force_cpu_ocr:
                print(
                    "[trocr] CUDA not available — TrOCR runs on CPU (slow). "
                    "Install CUDA-enabled PyTorch for GPU inference.",
                    flush=True,
                )
        else:
            device = torch.device("cuda")
        name = self.trocr_model_name
        allow_large = (os.environ.get("OCR_TROCR_ALLOW_LARGE", "") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if (
            not allow_large
            and isinstance(name, str)
            and "large" in name.lower()
            and "trocr" in name.lower()
        ):
            print(
                "[trocr] Large TrOCR checkpoints are much slower; using "
                "'microsoft/trocr-base-handwritten' for speed. "
                "Set OCR_TROCR_ALLOW_LARGE=1 or choose a base checkpoint to silence this.",
                flush=True,
            )
            name = "microsoft/trocr-base-handwritten"
            self.trocr_model_name = name
        print(f"Loading TrOCR model {name} on {device}…")
        self._trocr_processor = TrOCRProcessor.from_pretrained(name)
        self._trocr_model = VisionEncoderDecoderModel.from_pretrained(name).to(device)
        fp16_enabled = os.environ.get("OCR_TROCR_FP16", "1").strip().lower() in ("1", "true", "yes")
        if device.type == "cuda" and fp16_enabled:
            try:
                self._trocr_model = self._trocr_model.half()
            except Exception:
                pass
        self._trocr_model.eval()
        self._trocr_device = device

    def _trocr_batch_decode_crops(
        self,
        crops_bgr: List[np.ndarray],
        tr_max_new_tokens: int,
        _tr_beams: int,
        fp16_enabled: bool,
    ) -> List[str]:
        """
        Single batched TrOCR decode: **one** ``processor(...)`` and **one** ``model.generate(...)``
        for all valid crops (no inference micro-batching unless ``OCR_TROCR_CHUNKED_INFERENCE``).

        Fast greedy decode: ``num_beams=1``, ``max_new_tokens<=64``, AMP on CUDA.
        """
        self._last_decode_preprocess_s = 0.0
        self._last_decode_batching_s = 0.0
        self._last_decode_generate_s = 0.0
        if not crops_bgr:
            return []
        assert self._trocr_processor is not None and self._trocr_model is not None and self._trocr_device is not None
        import torch

        proc = self._trocr_processor
        model = self._trocr_model
        device = self._trocr_device
        cuda_amp = bool(fp16_enabled and device.type == "cuda")
        perf_enabled = os.environ.get("OCR_TROCR_PERF_LOG", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        validate_batch = os.environ.get("OCR_TROCR_VALIDATE_BATCH", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        chunked_infer = os.environ.get("OCR_TROCR_CHUNKED_INFERENCE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        try:
            chunk_bs = int(os.environ.get("OCR_TROCR_CHUNK_BATCH_SIZE", "64"))
        except ValueError:
            chunk_bs = 64
        chunk_bs = max(4, min(128, chunk_bs))
        try:
            max_side = int(os.environ.get("OCR_TROCR_CROP_MAX_SIDE", "256"))
        except ValueError:
            max_side = 256
        # Longest edge after resize (224–256 keeps encoder small; env can tune within band).
        max_side = max(224, min(256, max_side))

        # Greedy fast decode — cap sequence length (beam search disabled below).
        tr_max_new_tokens = max(8, min(64, int(tr_max_new_tokens)))

        print(f"Number of crops: {len(crops_bgr)}", flush=True)
        # STEP 7 — crop count visibility (batch validation / tuning).
        if perf_enabled or validate_batch:
            print(f"[trocr] len(crops)={len(crops_bgr)} device={device}", flush=True)
        if len(crops_bgr) > 15:
            print(
                "[trocr] NOTE: >15 crops — use line grouping (clear OCR_TROCR_WORD_MODE / "
                "raise OCR_TROCR_LINE_GROUP_WORD_THRESHOLD).",
                flush=True,
            )

        def _prep_one(crop: np.ndarray) -> Optional[Image.Image]:
            if crop is None or crop.size == 0:
                return None
            h, w = crop.shape[:2]
            if h < 6 or w < 6:
                return None
            if len(crop.shape) == 2:
                rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            if max(h, w) > max_side:
                scale = float(max_side) / float(max(h, w))
                nw = max(6, int(round(w * scale)))
                nh = max(6, int(round(h * scale)))
                rgb = _safe_cv2_resize(rgb, nw, nh)
            return Image.fromarray(rgb)

        t_pre0 = time.perf_counter()
        nc = len(crops_bgr)
        if nc <= 3:
            max_workers = 1
        else:
            max_workers = min(max(4, (os.cpu_count() or 4)), nc, 32)
        if max_workers <= 1:
            prepared = [_prep_one(c) for c in crops_bgr]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                prepared = list(ex.map(_prep_one, crops_bgr))
        preprocess_wall_s = time.perf_counter() - t_pre0
        self._last_decode_preprocess_s = float(preprocess_wall_s)

        out_texts: List[str] = [""] * len(crops_bgr)
        valid_indices = [i for i, im in enumerate(prepared) if im is not None]
        if not valid_indices:
            return out_texts

        n_valid = len(valid_indices)
        images = [prepared[i] for i in valid_indices]
        gen_kw = dict(
            max_new_tokens=tr_max_new_tokens,
            num_beams=1,
            do_sample=False,
            early_stopping=True,
            use_cache=True,
        )

        generate_wall_s = 0.0
        batching_wall_s = 0.0
        gpu_sync_wall_s = 0.0
        num_generate_calls = 0

        def _run_chunk(chunk_images: List[Image.Image], chunk_slot_indices: List[int]) -> None:
            nonlocal generate_wall_s, batching_wall_s, gpu_sync_wall_s, num_generate_calls
            tb0 = time.perf_counter()
            inputs = proc(images=chunk_images, padding=True, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(device, non_blocking=False)
            if cuda_amp:
                pixel_values = pixel_values.half()
            batching_wall_s += time.perf_counter() - tb0

            bs = int(pixel_values.shape[0])
            trace_here = _ocr_pipeline_trace_enabled() or perf_enabled or validate_batch
            if trace_here:
                print(
                    f"[trocr] pixel_values.shape={tuple(pixel_values.shape)} batch_dim={bs} "
                    f"(n_images={len(chunk_images)}) on_cuda={bool(getattr(pixel_values, 'is_cuda', False))}",
                    flush=True,
                )
            if bs != len(chunk_images):
                print(
                    "[trocr] WARN: batch dim mismatch vs PIL list — check processor outputs.",
                    flush=True,
                )

            if device.type == "cuda":
                ts0 = time.perf_counter()
                torch.cuda.synchronize()
                gpu_sync_wall_s += time.perf_counter() - ts0

            tg0 = time.perf_counter()
            with torch.no_grad():
                if cuda_amp:
                    with torch.cuda.amp.autocast():
                        generated_ids = model.generate(pixel_values, **gen_kw)
                else:
                    generated_ids = model.generate(pixel_values, **gen_kw)
            if device.type == "cuda":
                torch.cuda.synchronize()
            generate_wall_s += time.perf_counter() - tg0
            num_generate_calls += 1

            decoded = proc.batch_decode(generated_ids, skip_special_tokens=True)
            for slot_j, text in enumerate(decoded):
                out_texts[chunk_slot_indices[slot_j]] = (text or "").strip()

        if chunked_infer and n_valid > chunk_bs:
            print(
                f"[trocr] OCR_TROCR_CHUNKED_INFERENCE=1 — splitting {n_valid} crops "
                f"into chunks of {chunk_bs} (not recommended for latency).",
                flush=True,
            )
            for start in range(0, n_valid, chunk_bs):
                chunk_idx = valid_indices[start : start + chunk_bs]
                chunk_images = [prepared[i] for i in chunk_idx]
                _run_chunk(chunk_images, chunk_idx)
        else:
            _run_chunk(images, valid_indices)

        self._ocr_infer_generate_seconds_last = getattr(self, "_ocr_infer_generate_seconds_last", 0.0) + float(
            generate_wall_s
        )
        self._last_decode_batching_s = float(batching_wall_s)
        self._last_decode_generate_s = float(generate_wall_s)

        trace_here = _ocr_pipeline_trace_enabled() or perf_enabled
        if trace_here:
            print(
                "[trocr_timing_detail] "
                f"preprocess_resize_pil_s={preprocess_wall_s:.3f} tensor_pack_cuda_transfer_s={batching_wall_s:.3f} "
                f"gpu_idle_sync_before_generate_s={gpu_sync_wall_s:.3f} "
                f"model_generate_cuda_synced_s={generate_wall_s:.3f} generate_calls={num_generate_calls}",
                flush=True,
            )
            print(
                "[ocr_inference_only] trocr_model_generate_wall_s="
                f"{generate_wall_s:.3f} (CUDA synchronized around generate)",
                flush=True,
            )

        if generate_wall_s > 20.0:
            print(
                "[trocr] Architectural note: TrOCR uses an autoregressive decoder (VisionEncoderDecoder). "
                "Wall time scales roughly with (batch_size × sequence length × decoder layers); "
                "speed limits are often transformer decoding, not CPU batching. "
                "If GPU utilization looks low, decoding may still be memory-bandwidth bound.",
                flush=True,
            )

        if perf_enabled or validate_batch:
            print(
                f"[trocr] OCR_TROCR_CROP_MAX_SIDE={max_side} max_new_tokens={tr_max_new_tokens} "
                f"num_beams=1 chunked={chunked_infer and n_valid > chunk_bs}",
                flush=True,
            )
        return out_texts

    def _readtext_trocr_hybrid(
        self,
        img: np.ndarray,
        high_quality: bool = False,
        relaxed: bool = False,
        small_text_boost: bool = False,
        high_recall: bool = False,
    ) -> List[Tuple[Any, str, float]]:
        """
        EasyOCR text detection + TrOCR recognition. Best for Latin handwriting; slower than plain EasyOCR.
        Returns the same shape as EasyOCR readtext: list of (bbox, text, confidence).
        """
        from easyocr.utils import four_point_transform, reformat_input

        trace_h = _ocr_pipeline_trace_enabled()
        t_pipeline0 = time.perf_counter()

        self._ensure_trocr()
        assert self._trocr_processor is not None and self._trocr_model is not None and self._trocr_device is not None

        t_rf = time.perf_counter()
        img_in, _img_cv_grey = reformat_input(img)
        reformat_wall_s = time.perf_counter() - t_rf
        br = 0.24 if small_text_boost else 0.0
        hr = 0.06 if high_recall else 0.0
        mag_hr = 0.10 if high_recall else 0.0
        if relaxed:
            text_threshold = max(0.36, 0.46 - hr)
            low_text = max(0.14, 0.22 - hr * 0.75)
            link_threshold = max(0.14, 0.22 - hr * 0.75)
            mag_ratio = (2.12 if high_quality else 1.86) + br + mag_hr
        else:
            text_threshold = max(0.48, (0.6 if high_quality else 0.62) - hr)
            low_text = max(0.24, (0.36 if high_quality else 0.38) - hr * 0.85)
            link_threshold = max(0.24, (0.36 if high_quality else 0.38) - hr * 0.85)
            mag_ratio = (1.82 if high_quality else 1.62) + br + mag_hr

        fast_detect = os.environ.get("OCR_TROCR_FAST_DETECT", "1").strip().lower() not in ("0", "false", "no")
        try:
            canvas_override = int(os.environ.get("OCR_TROCR_DETECT_CANVAS_SIZE", "0"))
        except ValueError:
            canvas_override = 0
        if canvas_override > 0:
            canvas_sz = max(640, min(3840, canvas_override))
        elif fast_detect:
            try:
                canvas_sz = int(os.environ.get("OCR_TROCR_DETECT_CANVAS_FAST", "1600"))
            except ValueError:
                canvas_sz = 1600
            canvas_sz = max(960, min(2880, canvas_sz))
        else:
            canvas_sz = 2560
        mag_ratio_eff = mag_ratio * 0.88 if (fast_detect and canvas_override <= 0) else mag_ratio

        detect_kw = dict(
            min_size=20,
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
            canvas_size=int(canvas_sz),
            mag_ratio=mag_ratio_eff,
            slope_ths=0.1,
            ycenter_ths=0.5,
            height_ths=0.5,
            width_ths=0.5,
            add_margin=0.1,
            reformat=False,
            threshold=0.2,
            bbox_min_score=0.2,
            bbox_min_size=3,
            max_candidates=0,
        )
        perf_enabled_hybrid = os.environ.get("OCR_TROCR_PERF_LOG", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        detect_use_cache = os.environ.get("OCR_TROCR_DETECT_CACHE", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        img_fp = _trocr_img_detect_fingerprint(img_in)
        detect_kw_sig = _trocr_detect_kw_cache_tuple(detect_kw)
        cache_sig = (img_fp, detect_kw_sig)
        detect_cached = False
        t_det = time.perf_counter()
        if (
            detect_use_cache
            and self._trocr_detect_cache_sig == cache_sig
            and self._trocr_detect_cache_lists is not None
        ):
            _hlc, _flc = self._trocr_detect_cache_lists
            horizontal_list = copy.deepcopy(_hlc)
            free_list = copy.deepcopy(_flc)
            detect_wall_s = time.perf_counter() - t_det
            detect_cached = True
        else:
            horizontal_list, free_list = self.reader.detect(img_in, **detect_kw)
            detect_wall_s = time.perf_counter() - t_det
            horizontal_list, free_list = horizontal_list[0], free_list[0]
            if detect_use_cache:
                self._trocr_detect_cache_sig = cache_sig
                self._trocr_detect_cache_lists = (
                    copy.deepcopy(horizontal_list),
                    copy.deepcopy(free_list),
                )
        # TrOCR duplicate guard: when a free (rotated) box already covers a horizontal box,
        # keep only one stream so outer+inner boxes do not both produce the same text.
        strict_overlap = os.environ.get("OCR_TROCR_STRICT_OVERLAP", "1").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        t_ov = time.perf_counter()
        if strict_overlap and free_list and horizontal_list:
            try:
                ov_iomin = float(os.environ.get("OCR_TROCR_OVERLAP_IOMIN", "0.78"))
            except ValueError:
                ov_iomin = 0.78
            ov_iomin = max(0.45, min(0.98, ov_iomin))
            free_aabbs: List[Tuple[float, float, float, float]] = []
            for box in free_list:
                try:
                    free_aabbs.append(_poly_axis_aligned_bbox(box))
                except Exception:
                    continue
            filtered_horizontal = []
            for h in horizontal_list:
                try:
                    ha = (float(h[0]), float(h[2]), float(h[1]), float(h[3]))
                    h_area = max(1.0, (ha[2] - ha[0]) * (ha[3] - ha[1]))
                except Exception:
                    filtered_horizontal.append(h)
                    continue
                covered = False
                for fa in free_aabbs:
                    inter = _aabb_intersection_area(ha, fa)
                    if (inter / h_area) >= ov_iomin:
                        covered = True
                        break
                if not covered:
                    filtered_horizontal.append(h)
            horizontal_list = filtered_horizontal
        overlap_filter_wall_s = time.perf_counter() - t_ov

        out: List[Tuple[Any, str, float]] = []
        syn_conf = 0.92
        try:
            tr_max_new_tokens = int(os.environ.get("OCR_TROCR_MAX_NEW_TOKENS", "64"))
        except ValueError:
            tr_max_new_tokens = 64
        tr_max_new_tokens = max(8, min(512, tr_max_new_tokens))
        try:
            tr_beams = int(os.environ.get("OCR_TROCR_NUM_BEAMS", "1"))
        except ValueError:
            tr_beams = 1
        tr_beams = max(1, min(8, tr_beams))
        fp16_enabled = os.environ.get("OCR_TROCR_FP16", "1").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        word_mode = os.environ.get("OCR_TROCR_WORD_MODE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        try:
            wg_thr = int(os.environ.get("OCR_TROCR_LINE_GROUP_WORD_THRESHOLD", "15"))
        except ValueError:
            wg_thr = 15
        wg_thr = max(8, min(64, wg_thr))
        # Word-level horizontal boxes explode crop count — prefer line grouping past threshold.
        if word_mode and len(free_list) + len(horizontal_list) > wg_thr:
            print(
                "Number of crops (estimated word boxes):",
                len(free_list) + len(horizontal_list),
                f"— using line grouping (threshold OCR_TROCR_LINE_GROUP_WORD_THRESHOLD={wg_thr}).",
                flush=True,
            )
            word_mode = False

        maximum_y, maximum_x = img_in.shape[0], img_in.shape[1]
        crop_boxes: List[Any] = []
        crop_images: List[np.ndarray] = []

        t_jobs = time.perf_counter()
        # Reading order: free-form regions (tilted words/lines) then horizontal lines, each sorted by (y, x).
        free_jobs: List[Tuple[float, float, str, Any]] = []
        for box in free_list:
            try:
                pts = np.array(box, dtype=float)
                cy = float(pts[:, 1].mean())
                cx = float(pts[:, 0].min())
            except Exception:
                continue
            free_jobs.append((cy, cx, "f", box))
        free_jobs.sort(key=lambda t: (t[0], t[1]))

        horiz_jobs: List[Tuple[float, float, str, Any]] = []
        if word_mode:
            for box in horizontal_list:
                h = [float(x) for x in box]
                cy = 0.5 * (h[2] + h[3])
                cx = h[0]
                horiz_jobs.append((cy, cx, "w", h))
        else:
            for line_group in _group_easyocr_horizontal_into_lines(horizontal_list):
                u = _union_horizontal_aabbs(line_group)
                cy = 0.5 * (u[2] + u[3])
                cx = u[0]
                horiz_jobs.append((cy, cx, "u", u))
        horiz_jobs.sort(key=lambda t: (t[0], t[1]))

        all_jobs = free_jobs + horiz_jobs
        all_jobs.sort(key=lambda t: (t[0], t[1]))
        bbox_jobs_wall_s = time.perf_counter() - t_jobs
        t_crop = time.perf_counter()
        for _cy, _cx, kind, payload in all_jobs:
            if kind == "f":
                try:
                    rect = np.array(payload, dtype=np.float32)
                    warped = four_point_transform(img_in, rect)
                except Exception:
                    continue
                crop_images.append(warped)
                crop_boxes.append(payload)
            elif kind == "w":
                b = payload
                x_min, x_max, y_min, y_max = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                x_min = max(0, x_min)
                x_max = min(x_max, maximum_x)
                y_min = max(0, y_min)
                y_max = min(y_max, maximum_y)
                if x_max <= x_min or y_max <= y_min:
                    continue
                crop_images.append(img_in[y_min:y_max, x_min:x_max])
                crop_boxes.append([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
            else:
                u = payload
                x_min, x_max, y_min, y_max = int(u[0]), int(u[1]), int(u[2]), int(u[3])
                x_min = max(0, x_min)
                x_max = min(x_max, maximum_x)
                y_min = max(0, y_min)
                y_max = min(y_max, maximum_y)
                if x_max <= x_min or y_max <= y_min:
                    continue
                crop_images.append(img_in[y_min:y_max, x_min:x_max])
                crop_boxes.append([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
        crop_collect_wall_s = time.perf_counter() - t_crop
        print(f"len(crops): {len(crop_images)}", flush=True)
        if perf_enabled_hybrid or trace_h:
            print(
                f"[trocr_hybrid] detect_wall_s={detect_wall_s:.3f} crop_collect_wall_s={crop_collect_wall_s:.3f} "
                f"len(crops)={len(crop_images)} word_mode={word_mode} detect_cached={int(detect_cached)} canvas={canvas_sz}",
                flush=True,
            )

        self._last_decode_preprocess_s = 0.0
        self._last_decode_batching_s = 0.0
        self._last_decode_generate_s = 0.0
        merge_post_wall_s = 0.0

        batch_texts = self._trocr_batch_decode_crops(
            crop_images,
            tr_max_new_tokens=tr_max_new_tokens,
            _tr_beams=tr_beams,
            fp16_enabled=fp16_enabled,
        )
        for box, text in zip(crop_boxes, batch_texts):
            if not text:
                continue
            out.append((box, text, syn_conf))

        merged_result: List[Tuple[Any, str, float]] = []
        if out:
            t_mrg = time.perf_counter()
            merged = _merge_overlapping_easyocr_results(
                [(b, t, c) for b, t, c in out]
            )
            merged = _merge_trocr_adjacent_same_line_boxes(merged)
            merge_post_wall_s = time.perf_counter() - t_mrg
            merged_result = merged
            if trace_h:
                print(
                    "[trocr_hybrid_trace] "
                    f"reformat_input_s={reformat_wall_s:.3f} easyocr_detect_s={detect_wall_s:.3f} "
                    f"overlap_filter_s={overlap_filter_wall_s:.3f} bbox_sort_jobs_s={bbox_jobs_wall_s:.3f} "
                    f"crop_collect_s={crop_collect_wall_s:.3f} merge_post_s={merge_post_wall_s:.3f} "
                    f"relaxed={relaxed} high_quality={high_quality}",
                    flush=True,
                )

        total_pipeline_wall_s = time.perf_counter() - t_pipeline0
        bbox_overlap_jobs_s = overlap_filter_wall_s + bbox_jobs_wall_s
        prep_d = float(getattr(self, "_last_decode_preprocess_s", 0.0))
        bat_d = float(getattr(self, "_last_decode_batching_s", 0.0))
        gen_d = float(getattr(self, "_last_decode_generate_s", 0.0))
        silent_pb = os.environ.get("OCR_TROCR_SILENT_PIPELINE", "").strip().lower() in ("1", "true", "yes")
        if not silent_pb:
            print(
                "[trocr_pipeline_breakdown] "
                f"detection_time_s={detect_wall_s:.3f} bbox_overlap_jobs_time_s={bbox_overlap_jobs_s:.3f} "
                f"cropping_time_s={crop_collect_wall_s:.3f} preprocessing_time_s={prep_d:.3f} "
                f"processor_tensor_transfer_time_s={bat_d:.3f} gpu_inference_time_s={gen_d:.3f} "
                f"merge_post_time_s={merge_post_wall_s:.3f} hybrid_wall_total_s={total_pipeline_wall_s:.3f} "
                f"reformat_input_s={reformat_wall_s:.3f} crops={len(crop_images)} detect_cached={int(detect_cached)} "
                f"canvas={canvas_sz} fast_detect={int(fast_detect)}",
                flush=True,
            )

        return merged_result

    def _readtext_trocr_only(
        self,
        img: np.ndarray,
        high_quality: bool = False,
        relaxed: bool = False,
        small_text_boost: bool = False,
        high_recall: bool = False,
    ) -> List[Tuple[Any, str, float]]:
        """
        TrOCR-only OCR path (no EasyOCR):
        - Propose line boxes using OpenCV morphology/contours.
        - Run batched TrOCR recognition over proposed crops (`_trocr_batch_decode_crops`).
        """
        del high_quality, relaxed, small_text_boost, high_recall
        self._ensure_trocr()
        assert self._trocr_processor is not None and self._trocr_model is not None and self._trocr_device is not None

        syn_conf = 0.9

        try:
            tr_max_new_tokens = int(os.environ.get("OCR_TROCR_MAX_NEW_TOKENS", "64"))
        except ValueError:
            tr_max_new_tokens = 64
        tr_max_new_tokens = max(8, min(512, tr_max_new_tokens))
        try:
            tr_beams = int(os.environ.get("OCR_TROCR_NUM_BEAMS", "1"))
        except ValueError:
            tr_beams = 1
        tr_beams = max(1, min(8, tr_beams))
        fp16_enabled = os.environ.get("OCR_TROCR_FP16", "1").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        perf_only = os.environ.get("OCR_TROCR_PERF_LOG", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        img_in = img
        if img_in is None or img_in.size == 0:
            return []
        if len(img_in.shape) == 2:
            gray = img_in
            bgr = cv2.cvtColor(img_in, cv2.COLOR_GRAY2BGR)
        else:
            bgr = img_in
            gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

        t_prop = time.perf_counter()
        # Line proposals via connected strokes.
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h, w = bw.shape[:2]
        kx = max(15, w // 40)
        ky = max(2, h // 220)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        linked = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(linked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[Tuple[int, int, int, int]] = []
        min_w = max(24, w // 50)
        min_h = max(10, h // 120)
        min_area = max(180, (w * h) // 4500)
        for c in contours:
            x, y, ww, hh = cv2.boundingRect(c)
            if ww < min_w or hh < min_h:
                continue
            if ww * hh < min_area:
                continue
            boxes.append((x, y, x + ww, y + hh))

        boxes.sort(key=lambda b0: (b0[1], b0[0]))
        proposal_wall_s = time.perf_counter() - t_prop
        out: List[Tuple[Any, str, float]] = []
        crop_boxes: List[Any] = []
        crop_images: List[np.ndarray] = []
        pad_x = max(4, w // 300)
        pad_y = max(4, h // 300)
        t_crop_build = time.perf_counter()
        for x1, y1, x2, y2 in boxes:
            x1p = max(0, x1 - pad_x)
            y1p = max(0, y1 - pad_y)
            x2p = min(w, x2 + pad_x)
            y2p = min(h, y2 + pad_y)
            quad = [[x1p, y1p], [x2p, y1p], [x2p, y2p], [x1p, y2p]]
            crop_boxes.append(quad)
            crop_images.append(bgr[y1p:y2p, x1p:x2p])
        crop_collect_wall_s = time.perf_counter() - t_crop_build
        print(f"len(crops): {len(crop_images)}", flush=True)
        if perf_only:
            print(
                f"[trocr_only] proposal_wall_s={proposal_wall_s:.3f} crop_collect_wall_s={crop_collect_wall_s:.3f} "
                f"len(crops)={len(crop_images)}",
                flush=True,
            )

        if crop_images:
            batch_texts = self._trocr_batch_decode_crops(
                crop_images,
                tr_max_new_tokens=tr_max_new_tokens,
                _tr_beams=tr_beams,
                fp16_enabled=fp16_enabled,
            )
            for box, text in zip(crop_boxes, batch_texts):
                if not text:
                    continue
                out.append((box, text, syn_conf))

        # Fallback if contour proposals miss text: one full-image TrOCR pass.
        if not out:
            text = self._trocr_batch_decode_crops(
                [bgr],
                tr_max_new_tokens=tr_max_new_tokens,
                _tr_beams=tr_beams,
                fp16_enabled=fp16_enabled,
            )[0]
            if text:
                quad = [[0, 0], [w, 0], [w, h], [0, h]]
                out.append((quad, text, syn_conf))

        if out:
            merged = _merge_overlapping_easyocr_results(out)
            merged = _merge_trocr_adjacent_same_line_boxes(merged)
            return merged
        return []

    def _readtext(
        self,
        img: np.ndarray,
        high_quality: bool = False,
        relaxed: bool = False,
        small_text_boost: bool = False,
        high_recall: bool = False,
    ):
        """
        Run scene-text OCR on `img`. Backend-specific: EasyOCR `readtext` with tuned kwargs;
        PaddleOCR via `_run_paddle_ocr_subprocess` (relaxed/high_quality ignored); trocr_hybrid elsewhere.

        high_quality: beam search decoder + higher upscaling (slower, better on hard subtitles).
        relaxed: lower detection thresholds + stronger upscale — extra recall for posters / small text.
        small_text_boost: larger mag_ratio for textbook / fine print (use with higher max_dim).
        high_recall: lower detection thresholds + slightly stronger upscale (more boxes; more noise risk).
        """
        if self.backend == "paddleocr":
            return self._run_paddle_ocr_subprocess(
                img, small_text_boost=small_text_boost, high_recall=high_recall
            )
        if self.backend == "trocr_hybrid":
            return self._readtext_trocr_hybrid(
                img, high_quality, relaxed, small_text_boost, high_recall
            )
        if self.backend == "trocr_only":
            return self._readtext_trocr_only(
                img, high_quality, relaxed, small_text_boost, high_recall
            )

        br = 0.24 if small_text_boost else 0.0
        hr = 0.06 if high_recall else 0.0
        mag_hr = 0.10 if high_recall else 0.0
        if relaxed:
            kw = dict(
                paragraph=False,
                text_threshold=max(0.36, 0.46 - hr),
                low_text=max(0.14, 0.22 - hr * 0.75),
                link_threshold=max(0.14, 0.22 - hr * 0.75),
                mag_ratio=(2.12 if high_quality else 1.86) + br + mag_hr,
            )
            if high_quality:
                kw["decoder"] = "beamsearch"
                kw["beamWidth"] = 5
        else:
            kw = dict(
                paragraph=False,
                text_threshold=max(0.48, (0.6 if high_quality else 0.62) - hr),
                low_text=max(0.24, (0.36 if high_quality else 0.38) - hr * 0.85),
                link_threshold=max(0.24, (0.36 if high_quality else 0.38) - hr * 0.85),
                mag_ratio=(1.82 if high_quality else 1.62) + br + mag_hr,
            )
            if high_quality:
                kw["decoder"] = "beamsearch"
                kw["beamWidth"] = 5
        try:
            return self.reader.readtext(img, **kw)
        except TypeError:
            kw.pop("beamWidth", None)
            kw["decoder"] = "greedy"
            try:
                return self.reader.readtext(img, **kw)
            except TypeError:
                return self.reader.readtext(img)

    def _extract_text_core(
        self,
        working_path: str,
        preprocess: bool,
        max_dim: int,
        small_text_boost: bool = False,
        high_recall: bool = False,
    ) -> List[Tuple[str, List[List[int]], str, float]]:
        """OCR pipeline; `working_path` is the BGR file to read (original or rectified)."""
        _clear_preprocess_preview()
        trace = _ocr_pipeline_trace_enabled()
        self._ocr_infer_generate_seconds_last = 0.0
        t_extract0 = time.perf_counter()

        t_im = time.perf_counter()
        img_orig = cv2.imread(working_path)
        imread_wall_s = time.perf_counter() - t_im
        if img_orig is None:
            raise ValueError(f"Could not read image from {working_path}")
        orig_h, orig_w = img_orig.shape[:2]
        eff_dim = _ocr_effective_max_dim(max_dim, small_text_boost, high_recall=high_recall)

        detected_lang = self.detected_language
        stb = small_text_boost
        hr = high_recall
        use_multipass = _ocr_use_multipass_extract(self.backend)

        if not use_multipass:
            preprocess_wall_s = 0.0
            readtext_wall_s = 0.0
            readtext_calls = 0
            post_wall_s = 0.0
            if self._ocr_wants_preprocess(preprocess):
                t_pr = time.perf_counter()
                processed_img = self.preprocess_image_from_bgr(img_orig, max_dim=eff_dim, thin_strokes=False)
                preprocess_wall_s += time.perf_counter() - t_pr
                proc_h, proc_w = processed_img.shape[:2]
                t_rt = time.perf_counter()
                results = self._readtext(
                    processed_img,
                    high_quality=bool(preprocess),
                    small_text_boost=stb,
                    high_recall=hr,
                )
                readtext_wall_s += time.perf_counter() - t_rt
                readtext_calls += 1
            else:
                t_rs = time.perf_counter()
                img = _resize_bgr_to_max_dim(img_orig, eff_dim)
                preprocess_wall_s += time.perf_counter() - t_rs
                proc_h, proc_w = img.shape[:2]
                t_rt = time.perf_counter()
                results = self._readtext(
                    img, high_quality=False, small_text_boost=stb, high_recall=hr
                )
                readtext_wall_s += time.perf_counter() - t_rt
                readtext_calls += 1
            t_po = time.perf_counter()
            triples = _extract_triples_from_easyocr(results, detected_lang, high_recall=hr)
            scaled = _scale_ocr_triples_to_original_image(triples, orig_h, orig_w, proc_h, proc_w)
            scaled = _filter_ar_en_satellite_boxes(scaled)
            post_wall_s += time.perf_counter() - t_po
            extract_total_wall_s = time.perf_counter() - t_extract0
            infer_only = float(self._ocr_infer_generate_seconds_last)
            if trace:
                print(
                    "[ocr_pipeline_trace] single_pass "
                    f"imread_s={imread_wall_s:.3f} preprocess_resize_s={preprocess_wall_s:.3f} "
                    f"readtext_calls={readtext_calls} readtext_wall_incl_detect_s={readtext_wall_s:.3f} "
                    f"triples_scale_merge_s={post_wall_s:.3f} "
                    f"trocr_model_infer_generate_only_s={infer_only:.3f} "
                    f"extract_wall_total_s={extract_total_wall_s:.3f} backend={self.backend}",
                    flush=True,
                )
            if trace or self.backend in ("trocr_hybrid", "trocr_only"):
                print(
                    "[ocr_pipeline] "
                    f"trocr_infer_generate_only_s={infer_only:.3f} "
                    f"extract_wall_total_s={extract_total_wall_s:.3f} "
                    f"readtext_calls={readtext_calls}",
                    flush=True,
                )
            return scaled

        preprocess_wall_s = 0.0
        readtext_wall_s = 0.0
        readtext_calls = 0
        runs: List[List[Tuple[Any, str, float]]] = []
        if self._ocr_wants_preprocess(preprocess):
            t_pr = time.perf_counter()
            processed_img = self.preprocess_image_from_bgr(img_orig, max_dim=eff_dim, thin_strokes=False)
            preprocess_wall_s += time.perf_counter() - t_pr
            proc_h, proc_w = processed_img.shape[:2]
            hq = bool(preprocess)
            t_rt = time.perf_counter()
            runs.append(
                self._readtext(
                    processed_img, high_quality=hq, relaxed=False, small_text_boost=stb, high_recall=hr
                )
            )
            readtext_wall_s += time.perf_counter() - t_rt
            readtext_calls += 1
            t_rt = time.perf_counter()
            runs.append(
                self._readtext(
                    processed_img, high_quality=hq, relaxed=True, small_text_boost=stb, high_recall=hr
                )
            )
            readtext_wall_s += time.perf_counter() - t_rt
            readtext_calls += 1
            if not _ocr_skip_light_preprocess_pass() and getattr(self, "local_preprocess", True):
                try:
                    t_pr2 = time.perf_counter()
                    light_img = self.preprocess_image_from_bgr(img_orig, max_dim=eff_dim, thin_strokes=True)
                    preprocess_wall_s += time.perf_counter() - t_pr2
                    t_rt = time.perf_counter()
                    runs.append(
                        self._readtext(
                            light_img, high_quality=True, relaxed=True, small_text_boost=stb, high_recall=hr
                        )
                    )
                    readtext_wall_s += time.perf_counter() - t_rt
                    readtext_calls += 1
                except ValueError:
                    pass
        else:
            t_rs = time.perf_counter()
            img = _resize_bgr_to_max_dim(img_orig, eff_dim)
            preprocess_wall_s += time.perf_counter() - t_rs
            proc_h, proc_w = img.shape[:2]
            t_rt = time.perf_counter()
            runs.append(
                self._readtext(img, high_quality=False, relaxed=False, small_text_boost=stb, high_recall=hr)
            )
            readtext_wall_s += time.perf_counter() - t_rt
            readtext_calls += 1
            t_rt = time.perf_counter()
            runs.append(
                self._readtext(img, high_quality=False, relaxed=True, small_text_boost=stb, high_recall=hr)
            )
            readtext_wall_s += time.perf_counter() - t_rt
            readtext_calls += 1

        t_flat = time.perf_counter()
        flat: List[Tuple[Any, str, float]] = []
        for r in runs:
            flat.extend(r)
        merged = _merge_overlapping_easyocr_results(flat)
        merge_flat_wall_s = time.perf_counter() - t_flat
        t_po = time.perf_counter()
        triples = _extract_triples_from_easyocr(merged, detected_lang, high_recall=hr)
        scaled = _scale_ocr_triples_to_original_image(triples, orig_h, orig_w, proc_h, proc_w)
        scaled = _filter_ar_en_satellite_boxes(scaled)
        post_wall_s = time.perf_counter() - t_po
        extract_total_wall_s = time.perf_counter() - t_extract0
        infer_only = float(self._ocr_infer_generate_seconds_last)
        if trace:
            print(
                "[ocr_pipeline_trace] multipass "
                f"imread_s={imread_wall_s:.3f} preprocess_s={preprocess_wall_s:.3f} "
                f"readtext_calls={readtext_calls} readtext_wall_incl_detect_s={readtext_wall_s:.3f} "
                f"merge_flat_s={merge_flat_wall_s:.3f} triples_scale_s={post_wall_s:.3f} "
                f"trocr_model_infer_generate_only_s={infer_only:.3f} "
                f"extract_wall_total_s={extract_total_wall_s:.3f} backend={self.backend}",
                flush=True,
            )
        if trace or self.backend in ("trocr_hybrid", "trocr_only"):
            print(
                "[ocr_pipeline] "
                f"trocr_infer_generate_only_s={infer_only:.3f} "
                f"extract_wall_total_s={extract_total_wall_s:.3f} "
                f"readtext_calls={readtext_calls}",
                flush=True,
            )
        return scaled

    def extract_text_for_overlay(
        self,
        image_path: str,
        preprocess: bool = True,
        max_dim: int = 1200,
        dewarp_screen: bool = False,
        small_text_boost: bool = False,
        high_recall: bool = False,
    ) -> Tuple[List[Tuple], str]:
        """
        OCR like extract_text, but optionally applies perspective rectification first (angled screen photos).
        Returns (triples, image_path_for_render). Use the second path for draw_bounding_boxes / render_translated_image
        so boxes match pixels (rectified image when dewarp succeeded).
        """
        dewarp_tmp: Optional[str] = None
        working_path = image_path
        if dewarp_screen:
            img0 = cv2.imread(image_path)
            if img0 is not None:
                warped = try_perspective_dewarp_image(img0)
                if warped is not None and warped.size > 0:
                    fd, dewarp_tmp = tempfile.mkstemp(suffix=".png")
                    os.close(fd)
                    cv2.imwrite(dewarp_tmp, warped)
                    working_path = dewarp_tmp
        try:
            triples = self._extract_text_core(
                working_path,
                preprocess,
                max_dim,
                small_text_boost=small_text_boost,
                high_recall=high_recall,
            )
            return triples, working_path
        except Exception:
            if dewarp_tmp and os.path.isfile(dewarp_tmp):
                try:
                    os.unlink(dewarp_tmp)
                except OSError:
                    pass
            raise

    def extract_text(
        self,
        image_path: str,
        preprocess: bool = True,
        max_dim: int = 1200,
        small_text_boost: bool = False,
        high_recall: bool = False,
    ) -> List[Tuple]:
        """
        Extract text from an image using OCR.

        Args:
            image_path: Path to the image file
            preprocess: Whether to preprocess the image before OCR
            max_dim: Max width/height when downscaling (smaller = faster, e.g. 800)
            small_text_boost: Use higher resolution + stronger detector upscale for fine print.
            high_recall: Softer confidence filters + lower detection thresholds + higher min resolution.

        Returns:
            List of (text, bounding_box, lang, ocr_confidence). Bounding boxes match `image_path` pixel space.
        """
        return self._extract_text_core(
            image_path, preprocess, max_dim, small_text_boost=small_text_boost, high_recall=high_recall
        )

    def extract_text_from_video_frame(
        self,
        frame: np.ndarray,
        high_quality: bool = False,
        small_text_boost: bool = False,
        high_recall: bool = False,
    ) -> List[Tuple]:
        """
        Extract text from a video frame using the configured backend (EasyOCR, PaddleOCR worker, etc.).

        Video sampling loops may skip calling this entirely when frames look identical (see
        `_ocr_video_skip_near_duplicate_frames`) so Paddle/EasyOCR is not invoked twice for static scenes.

        Args:
            frame: Video frame as numpy array (BGR).
            high_quality: Use stronger decoder/upscale when True (e.g. matches "High-accuracy OCR").
            small_text_boost: Stronger detector upscale for fine print (use with higher ocr_max_dim on frames).
            high_recall: Lower detection thresholds and softer confidence filtering (more text, more noise).
        """
        detected_lang = self.detected_language
        stb = small_text_boost
        hr = high_recall
        if _ocr_video_frame_single_pass():
            results = self._readtext(
                frame, high_quality=high_quality, small_text_boost=stb, high_recall=hr
            )
            return _extract_triples_from_easyocr(results, detected_lang, high_recall=hr)
        r1 = self._readtext(
            frame, high_quality=high_quality, relaxed=False, small_text_boost=stb, high_recall=hr
        )
        r2 = self._readtext(
            frame, high_quality=high_quality, relaxed=True, small_text_boost=stb, high_recall=hr
        )
        merged = _merge_overlapping_easyocr_results(list(r1) + list(r2))
        return _extract_triples_from_easyocr(merged, detected_lang, high_recall=hr)

    def extract_text_from_video(
        self,
        video_path: str,
        sample_interval_sec: float = 2.0,
        preprocess: bool = False,
        small_text_boost: bool = False,
        high_recall: bool = False,
    ) -> List[Tuple]:
        """
        Extract text from video by sampling frames at regular intervals.

        Args:
            video_path: Path to the video file
            sample_interval_sec: Take one frame every N seconds
            preprocess: Whether to preprocess each frame before OCR

        Returns:
            List of (detected_text, bbox, detected_language); may contain duplicates across frames.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = max(1, int(fps * sample_interval_sec))
        all_results = []
        frame_idx = 0
        prev_sig: Optional[Tuple[Any, ...]] = None
        prev_gray_small: Optional[np.ndarray] = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            if preprocess:
                frame = self._enhance_bgr_for_ocr(frame)

            gray_small = _video_frame_gray_small_fp(frame)
            if (
                _ocr_video_skip_near_duplicate_frames()
                and prev_gray_small is not None
                and float(np.mean((gray_small - prev_gray_small) ** 2))
                < _ocr_video_frame_mse_skip_threshold()
            ):
                continue
            prev_gray_small = gray_small

            results = self.extract_text_from_video_frame(
                frame,
                high_quality=preprocess,
                small_text_boost=small_text_boost,
                high_recall=high_recall,
            )
            sig = _ocr_frame_signature(results)
            if prev_sig is not None and sig == prev_sig:
                continue
            prev_sig = sig
            all_results.extend(results)

        cap.release()
        return all_results

    def extract_text_from_video_with_frames(
        self,
        video_path: str,
        sample_interval_sec: float = 2.0,
        preprocess: bool = False,
        max_frames: int = 120,
        initial_dense_sec: float = 0,
        initial_dense_interval_sec: float = 0.25,
        ocr_max_dim: int = 0,
        small_text_boost: bool = False,
        high_recall: bool = False,
    ) -> tuple:
        """
        Extract text from video by sampling frames; return flat list and per-frame list for overlay.
        ocr_max_dim: if > 0, downscale frames to this max dimension for OCR (faster); bboxes are scaled back.

        Returns:
            (all_results_flat, per_frame_data) where per_frame_data is [(frame_idx, [(text, bbox, lang), ...]), ...].
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = max(1, int(fps * sample_interval_sec))
        initial_dense_frames = int(fps * initial_dense_sec)
        # Never sample the opening burst more often than the main interval (avoids redundant OCR)
        dense_interval = max(1, int(fps * initial_dense_interval_sec))
        dense_interval = min(dense_interval, frame_interval)
        all_results = []
        per_frame_data = []
        read_count = 0
        frames_with_text = 0
        prev_sig: Optional[Tuple[Any, ...]] = None
        prev_gray_small: Optional[np.ndarray] = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            in_initial = initial_dense_frames > 0 and read_count < initial_dense_frames
            if in_initial:
                should_sample = read_count % dense_interval == 0
            else:
                should_sample = (read_count - initial_dense_frames) % frame_interval == 0
            if should_sample:
                h, w = frame.shape[:2]
                ocr_frame = frame
                scale_x, scale_y = 1.0, 1.0
                if ocr_max_dim > 0 and max(h, w) > ocr_max_dim:
                    ocr_frame = _resize_bgr_to_max_dim(frame, ocr_max_dim)
                    new_h, new_w = ocr_frame.shape[:2]
                    scale_x, scale_y = w / max(1, new_w), h / max(1, new_h)
                if preprocess:
                    ocr_frame = self._enhance_bgr_for_ocr(ocr_frame)
                gray_small = _video_frame_gray_small_fp(ocr_frame)
                if (
                    _ocr_video_skip_near_duplicate_frames()
                    and prev_gray_small is not None
                    and float(np.mean((gray_small - prev_gray_small) ** 2))
                    < _ocr_video_frame_mse_skip_threshold()
                ):
                    read_count += 1
                    continue
                prev_gray_small = gray_small
                results = self.extract_text_from_video_frame(
                    ocr_frame,
                    high_quality=preprocess,
                    small_text_boost=small_text_boost,
                    high_recall=high_recall,
                )
                if scale_x != 1.0 or scale_y != 1.0:
                    results = [
                        (
                            text,
                            [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in bbox],
                            lang,
                            conf,
                        )
                        for (text, bbox, lang, conf) in (_ocr_row_unpack(x) for x in results)
                    ]
                sig = _ocr_frame_signature(results)
                # Skip consecutive identical scenes (common at video start: dense sampling sees the same title many times)
                if prev_sig is not None and sig == prev_sig:
                    read_count += 1
                    continue
                prev_sig = sig
                per_frame_data.append((read_count, results))
                if results:
                    all_results.extend(results)
                    frames_with_text += 1
                    if frames_with_text >= max_frames:
                        read_count += 1
                        break
            read_count += 1
        cap.release()
        return all_results, per_frame_data

    def _get_font(self, size: int, prefer_arabic: bool = False):
        """
        Load a font for overlay text. Segoe UI is tried first for broad Latin + Arabic + Cyrillic coverage
        on Windows (reduces hollow .notdef boxes vs. older Arial-only order). See _truetype_path_candidates
        for Linux/macOS/WSL and OCR_ARABIC_FONT / OCR_TTF_DIR.
        """
        for name in _truetype_path_candidates(prefer_arabic):
            try:
                return _pil_truetype(name, size)
            except (OSError, IOError, ValueError, TypeError):
                continue
        return ImageFont.load_default()

    def _draw_utf8_line_on_bgr(
        self,
        img_bgr: np.ndarray,
        x: int,
        y_ref: int,
        text: str,
        box_h: int,
    ) -> None:
        """Draw one line with PIL + Unicode fonts on a BGR image (in place). Used when OpenCV putText cannot render Arabic."""
        t = (text or "").strip()
        if not t:
            return
        rtl = _is_arabic_or_rtl(t)
        display = _prepare_text_for_draw(t) if rtl else t
        fs = max(10, min(32, int(max(12, box_h * 0.55))))
        font = self._get_font(fs, prefer_arabic=rtl)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil)
        try:
            bb = draw.textbbox((0, 0), display, font=font)
            th = max(1, int(bb[3] - bb[1]))
        except Exception:
            th = fs
        ty = y_ref - th
        if ty < 0:
            ty = 0
        tx = int(np.clip(x, 0, max(0, img_bgr.shape[1] - 2)))
        try:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                draw.text((tx + dx, ty + dy), display, font=font, fill=(0, 0, 0))
            draw.text((tx, ty), display, font=font, fill=(255, 255, 255))
        except TypeError:
            draw.text((tx, ty), display, font=font, fill=(255, 255, 255))
        img_bgr[:, :, :] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def _best_matching_font(
        self,
        img_rgb: np.ndarray,
        x_min: int, y_min: int, x_max: int, y_max: int,
        orig_text: str,
        font_size: int,
        is_rtl: bool,
    ) -> ImageFont.ImageFont:
        """Pick the font that best matches the original text in the image crop (render-original comparison)."""
        w = max(1, x_max - x_min)
        h = max(1, y_max - y_min)
        crop = img_rgb[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return self._get_font(font_size, prefer_arabic=is_rtl)
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        prep = _prepare_text_for_draw(orig_text) if is_rtl else orig_text
        if not prep.strip():
            return self._get_font(font_size, prefer_arabic=is_rtl)
        best_font = None
        best_score = -float("inf")
        for path in _FONT_CANDIDATES:
            try:
                font = _pil_truetype(path, font_size)
            except (OSError, IOError):
                continue
            try:
                tmp = Image.new("RGB", (max(w * 2, 400), max(h * 2, 80)), (255, 255, 255))
                d = ImageDraw.Draw(tmp)
                d.text((10, 10), prep, font=font, fill=(0, 0, 0))
                if getattr(d, "textbbox", None):
                    bx = d.textbbox((10, 10), prep, font=font)
                else:
                    tw, th = d.textsize(prep, font=font)
                    bx = (10, 10, 10 + tw, 10 + th)
                rendered = tmp.crop(bx)
                rendered = rendered.resize((max(1, w), max(1, h)))
                rend_gray = cv2.cvtColor(np.array(rendered), cv2.COLOR_RGB2GRAY)
                c_n = crop_gray.astype(np.float32) / 255.0
                rw, rh = max(1, int(w)), max(1, int(h))
                r_n = _safe_cv2_resize(rend_gray.astype(np.float32), rw, rh) / 255.0
                if c_n.shape != r_n.shape:
                    r_n = _safe_cv2_resize(r_n, max(1, c_n.shape[1]), max(1, c_n.shape[0]))
                mse = float(np.mean((c_n - r_n) ** 2))
                score = -mse
                if score > best_score:
                    best_score = score
                    best_font = font
            except Exception:
                continue
        return best_font if best_font is not None else self._get_font(font_size, prefer_arabic=is_rtl)

    def _render_translated_on_image(
        self,
        img_rgb: np.ndarray,
        results: List[Tuple],
        translated_texts: List[str],
        background: str = "white",
        text_color: Tuple[int, int, int] = (0, 0, 0),
        use_font_matching: bool = False,
        for_video: bool = False,
        use_inpainting: bool = True,
        match_original_text_style: bool = True,
        inpaint_style: str = "solid",
        merge_same_line_boxes: bool = False,
        preserve_original_font_size: bool = True,
        target_lang: Optional[str] = None,
        perspective_warp: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Draw translated text on an RGB image. Preserves exact bounding box from OCR.
        Pipeline: inpaint original text regions -> render each translation inside its box
        (font scaled to fit, top-left anchor, multi-line wrap; perspective warp to the OCR
        quadrilateral when the box is tilted — see perspective_warp and PHOTO_OVERLAY_PERSPECTIVE_WARP).
        When preserve_original_font_size is True (static images), font size is matched to the
        original line using the source string and Latin metrics, then only reduced if the translation overflows.
        By default (static photos), one **median** target size is shared across all boxes so a short
        English word in a large OCR box does not become giant Arabic next to normal lines. Set
        env PHOTO_UNIFORM_FONT_SIZE=0 to restore independent per-box sizing.
        When match_original_text_style=True (default), text is drawn on a transparent layer with
        colors estimated from the original (no white boxes). Set False for legacy black-on-white cards.
        For **video** with ``match_original_text_style=False``, all translations on a frame are drawn
        in **one** expanded axis-aligned white panel (reading order, newline-separated) so multiple
        OCR boxes do not stack into overlapping card shapes.
        inpaint_style: "solid" fills text regions with local background color (clearer on wood/photos).
        "telea" uses OpenCV inpaint (PHOTO_INPAINT_ALGORITHM=ns|telea; NS is default, Telea can smear on wood).
        Env PHOTO_OVERLAY_INPAINT=telea forces Telea when solid is off. PHOTO_TEXT_CENTER_IN_BOX=0 disables centering text in each OCR box.
        merge_same_line_boxes: If True, merge nearby horizontal boxes into one (subtitle-style). If False (default for
        static photos), each OCR segment keeps its own box so menus keep row-by-row layout.
        target_lang: optional UI language code (e.g. ar/fa/ur). For Arabic-script targets, Latin-only segments
        use the same overlay font, outline, and contrast rules as translated RTL lines instead of thin default Latin.
        perspective_warp: for static images, if True (default), map text onto the OCR quad when tilted. False = always
        draw a horizontal block in the axis-aligned box. If None, env PHOTO_OVERLAY_PERSPECTIVE_WARP=0 forces off; default on.
        Returns RGB image.
        """
        if len(translated_texts) != len(results):
            raise ValueError("translated_texts must have same length as results")
        img_h, img_w = img_rgb.shape[:2]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        bg_tuple = (255, 255, 255) if background == "white" else (240, 240, 240)
        _allow_perspective = _should_apply_perspective_warp(for_video, perspective_warp)

        if for_video:
            results_to_draw, translated_to_draw = _merge_boxes_same_line(results, translated_texts, img_w, img_h)
            results_to_draw, translated_to_draw = _merge_video_duplicate_translation_boxes(
                results_to_draw, translated_to_draw, img_w, img_h
            )
            results_to_draw, translated_to_draw = _merge_video_adjacent_boxes(
                results_to_draw, translated_to_draw, img_w, img_h
            )
            results_to_draw, translated_to_draw = _dedupe_overlay_segments_for_frame(
                results_to_draw, translated_to_draw
            )
        elif merge_same_line_boxes:
            results_to_draw, translated_to_draw = _merge_boxes_same_line(results, translated_texts, img_w, img_h)
        else:
            results_to_draw, translated_to_draw = list(results), list(translated_texts)

        if not for_video:
            results_to_draw, translated_to_draw = _suppress_overlapping_photo_overlays(
                results_to_draw, translated_to_draw, img_w, img_h
            )

        # 0–1) Remove original text: solid local fill vs Telea inpaint (see inpaint_style / PHOTO_OVERLAY_INPAINT)
        _style = (os.environ.get("PHOTO_OVERLAY_INPAINT") or inpaint_style or "telea").strip().lower()
        use_telea = _style == "telea" and not for_video

        if use_inpainting:
            if use_telea:
                _inpaint_remove_text_regions_bgr(img_bgr, results_to_draw, img_w, img_h)
            else:
                # tight_photo: smaller mask so fills hug each line (less giant flat patches on menus).
                _erase_text_regions_solid_local(
                    img_bgr, results_to_draw, img_w, img_h, tight_photo=not for_video
                )

        # One typographic scale for the whole menu (avoids huge Arabic in oversized OCR boxes vs tiny neighbors).
        _uniform_off = os.environ.get("PHOTO_UNIFORM_FONT_SIZE", "").strip().lower() in (
            "0",
            "false",
            "no",
        )
        photo_uniform_base: Optional[int] = None
        if (
            not for_video
            and preserve_original_font_size
            and not _uniform_off
        ):
            raw_fs: List[int] = []
            for row, trans_text in zip(results_to_draw, translated_to_draw):
                orig_text, bbox = row[0], row[1]
                geo = _bbox_geometry(bbox, img_w, img_h)
                w0, h0 = geo["w"], geo["h"]
                if w0 <= 0 or h0 <= 0:
                    continue
                if _is_url_or_footer_ocr(orig_text or ""):
                    continue
                o = (orig_text or "").strip()
                if not o:
                    continue
                est = _estimate_font_size_from_original_text(
                    o, w0, h0, lambda s: self._get_font(s, prefer_arabic=False)
                )
                if est is not None:
                    raw_fs.append(int(est))
            if raw_fs:
                photo_uniform_base = int(round(statistics.median(raw_fs)))
                photo_uniform_base = max(10, min(96, photo_uniform_base))

        # Latin-script targets (e.g. de→German): one text color for all Latin lines so English loanwords
        # match translated lines (per-box Otsu otherwise copies original sign white/bold color).
        latin_unified_color: Optional[Tuple[int, int, int]] = None
        if not for_video and _photo_unify_latin_overlay_colors(target_lang):
            long_rgb: List[Tuple[int, int, int]] = []
            all_rgb: List[Tuple[int, int, int]] = []
            for row, trans_text in zip(results_to_draw, translated_to_draw):
                orig_text, bbox = row[0], row[1]
                if _is_url_or_footer_ocr(orig_text or ""):
                    continue
                dt = _photo_segment_display_text(trans_text)
                if not _is_mostly_latin_script(dt):
                    continue
                geo = _bbox_geometry(bbox, img_w, img_h)
                if geo["w"] <= 0 or geo["h"] <= 0:
                    continue
                xm, ym, xM, yM = geo["x_min"], geo["y_min"], geo["x_max"], geo["y_max"]
                est0 = _estimate_text_color_rgb(img_rgb, xm, ym, xM, yM)
                est0 = _normalize_photo_overlay_text_color(est0, img_rgb, xm, ym, xM, yM)
                all_rgb.append(est0)
                if len(dt.strip()) >= 10:
                    long_rgb.append(est0)
            ref = long_rgb if len(long_rgb) >= 1 else all_rgb
            if ref and len(long_rgb) < 1 and len(ref) >= 2:
                arr = np.array(ref, dtype=np.float64)
                lum = 0.299 * arr[:, 0] + 0.587 * arr[:, 1] + 0.114 * arr[:, 2]
                if float(np.max(lum) - np.median(lum)) > 38:
                    i_drop = int(np.argmax(lum))
                    ref = [ref[i] for i in range(len(ref)) if i != i_drop]
            if ref:
                med = np.median(np.array(ref, dtype=np.float64), axis=0)
                latin_unified_color = tuple(int(round(c)) for c in med)

        # 2) Video + legacy black-on-white: one axis-aligned card for all segments on this frame.
        # Per-box white fillPolys stack with rounded PIL edges and look like multiple scalloped boxes.
        if for_video and not match_original_text_style:
            fill_pad_px_v = 6
            fill_pad_ratio_v = 0.08
            items: List[Tuple[float, float, float, float, str]] = []
            for row, trans_text in zip(results_to_draw, translated_to_draw):
                try:
                    orig_text, bbox = row[0], row[1]
                    if _is_url_or_footer_ocr(orig_text or ""):
                        continue
                    if _is_translation_failure_overlay(trans_text):
                        continue
                    dt = (_photo_segment_display_text(trans_text) or "").strip()
                    if not dt:
                        continue
                    geo = _bbox_geometry(bbox, img_w, img_h)
                    if geo["w"] <= 0 or geo["h"] <= 0:
                        continue
                    xm, ym, xM, yM = (
                        float(geo["x_min"]),
                        float(geo["y_min"]),
                        float(geo["x_max"]),
                        float(geo["y_max"]),
                    )
                    items.append((xm, ym, xM, yM, dt))
                except Exception:
                    continue
            if items:
                items.sort(key=lambda t: (t[1], t[0]))
                combined = "\n".join(t[4] for t in items)
                combined = _sanitize_rendered_text(combined)
                if len(combined) > 12000:
                    combined = combined[:11997] + "…"
                ux0 = min(t[0] for t in items)
                uy0 = min(t[1] for t in items)
                uX = max(t[2] for t in items)
                uY = max(t[3] for t in items)
                union_bbox = [[ux0, uy0], [uX, uy0], [uX, uY], [ux0, uY]]
                ex = _expand_bbox(union_bbox, img_w, img_h, fill_pad_px_v, fill_pad_ratio_v)
                xs = [float(p[0]) for p in ex]
                ys = [float(p[1]) for p in ex]
                cx0 = int(round(min(xs)))
                cy0 = int(round(min(ys)))
                cx1 = int(round(max(xs)))
                cy1 = int(round(max(ys)))
                cx0 = max(0, min(cx0, img_w - 2))
                cy0 = max(0, min(cy0, img_h - 2))
                cx1 = max(cx0 + 4, min(cx1, img_w - 1))
                cy1 = max(cy0 + 4, min(cy1, img_h - 1))
                cw, ch = cx1 - cx0, cy1 - cy0
                cv2.fillPoly(img_bgr, [np.array(ex, dtype=np.int32)], (255, 255, 255))
                is_rtl = any(_is_arabic_or_rtl(t[4]) for t in items)

                def get_font_fn(size: int):
                    return self._get_font(size, prefer_arabic=is_rtl)

                text_layer_pil, _ = _render_text_into_box_pillow(
                    combined,
                    cw,
                    ch,
                    get_font_fn,
                    is_rtl,
                    text_color=text_color,
                    bg_color=None,
                    for_video=True,
                    target_font_size=None,
                    style_like_rtl=False,
                )
                text_rgba = np.array(text_layer_pil)
                if text_rgba.size > 0:
                    _blend_rgba_on_bgr(img_bgr, text_rgba, cx0, cy0)
                return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2b) Draw translation per merged segment (optional white pad only in legacy mode)
        fill_pad_px = 6 if for_video else 10
        fill_pad_ratio = 0.08 if for_video else 0.15
        drawn_segments = 0
        for row, trans_text in zip(results_to_draw, translated_to_draw):
            try:
                orig_text, bbox = row[0], row[1]
                geo = _bbox_geometry(bbox, img_w, img_h)
                x_min, y_min = geo["x_min"], geo["y_min"]
                x_max, y_max = geo["x_max"], geo["y_max"]
                w, h = geo["w"], geo["h"]
                if w <= 0 or h <= 0:
                    continue
                # Template footers / URLs: remove via inpaint only; do not paint a translation box
                if _is_url_or_footer_ocr(orig_text or ""):
                    continue
                if _is_translation_failure_overlay(trans_text):
                    continue
                if not match_original_text_style:
                    expanded_fill = _expand_bbox(bbox, img_w, img_h, fill_pad_px, fill_pad_ratio)
                    cv2.fillPoly(img_bgr, [expanded_fill], (255, 255, 255))
                display_text = _photo_segment_display_text(trans_text)
                is_rtl = _is_arabic_or_rtl(display_text)
                _tgt = (target_lang or "").lower().strip()
                # English/Latin left untranslated on Arabic-script-target photos: draw like RTL lines (font stack, outline, contrast).
                style_like_rtl = (
                    not for_video
                    and _tgt in _PHOTO_UNIFY_LATIN_TARGETS
                    and not is_rtl
                    and _is_mostly_latin_script(display_text)
                )
                # Font matching compares OCR to the crop and picks a font that fits the *source* glyphs.
                # Translated text is often another script (Arabic, etc.) — that font can lack glyphs → tofu.
                # For video, never use crop matching: use Segoe/Tahoma chain only (see _get_font).
                use_shape_match = (
                    use_font_matching
                    and not for_video
                    and not _is_arabic_or_rtl(display_text)
                    and not style_like_rtl
                )
                if match_original_text_style:
                    est = _estimate_text_color_rgb(img_rgb, x_min, y_min, x_max, y_max)
                    est = _normalize_photo_overlay_text_color(est, img_rgb, x_min, y_min, x_max, y_max)
                    if latin_unified_color is not None and _is_mostly_latin_script(display_text) and not is_rtl:
                        box_text_color = latin_unified_color
                    elif is_rtl or style_like_rtl:
                        box_text_color = _contrast_text_rgb_for_rtl(est, img_rgb, x_min, y_min, x_max, y_max)
                    else:
                        box_text_color = est
                    if match_original_text_style and not for_video:
                        if not (
                            latin_unified_color is not None
                            and _is_mostly_latin_script(display_text)
                            and not is_rtl
                        ):
                            box_text_color = _normalize_photo_overlay_text_color(
                                box_text_color, img_rgb, x_min, y_min, x_max, y_max
                            )
                    pil_bg = None
                else:
                    box_text_color = text_color
                    pil_bg = bg_tuple
                if use_shape_match:
                    def get_font_fn(size: int, _x_min=x_min, _y_min=y_min, _x_max=x_max, _y_max=y_max, _orig=orig_text, _rtl=is_rtl):
                        return self._best_matching_font(img_rgb, _x_min, _y_min, _x_max, _y_max, _orig, size, _rtl)
                else:
                    def get_font_fn(size: int):
                        return self._get_font(size, prefer_arabic=(is_rtl or style_like_rtl))
                target_fs = None
                _rtl_fit = is_rtl or style_like_rtl
                if preserve_original_font_size and not for_video and (orig_text or "").strip():
                    if not _is_url_or_footer_ocr(orig_text or ""):
                        if photo_uniform_base is not None:
                            target_fs = _refine_target_font_size_for_translation(
                                display_text, w, h, photo_uniform_base, _rtl_fit, get_font_fn
                            )
                        else:
                            target_fs = _estimate_font_size_from_original_text(
                                orig_text, w, h, lambda s: self._get_font(s, prefer_arabic=False)
                            )
                            if target_fs is not None:
                                target_fs = _refine_target_font_size_for_translation(
                                    display_text, w, h, target_fs, _rtl_fit, get_font_fn
                                )
                text_layer_pil, _ = _render_text_into_box_pillow(
                    display_text,
                    w, h,
                    get_font_fn,
                    is_rtl,
                    text_color=box_text_color,
                    bg_color=pil_bg,
                    for_video=for_video,
                    target_font_size=target_fs,
                    style_like_rtl=style_like_rtl,
                )
                text_rgba = np.array(text_layer_pil)
                if text_rgba.size == 0:
                    continue
                # Legacy: force opaque text on white card. Transparent layer: do not turn clear pixels opaque.
                if for_video and text_rgba.shape[2] == 4 and pil_bg is not None:
                    not_white = (text_rgba[:, :, :3].astype(np.int32).sum(axis=2) < 255 * 3) | (text_rgba[:, :, 3] < 255)
                    text_rgba[not_white, 3] = 255
                # Map rectangular text layer onto the OCR quadrilateral so tilted text follows the same angle.
                # In local photo mode, always project onto the OCR quadrilateral when available.
                # This preserves layout/angle even when rotation heuristics are borderline.
                if _allow_perspective and len(geo["pts"]) >= 4:
                    warped, wx_min, wy_min, wx_max, wy_max = _warp_text_layer_to_polygon(
                        text_rgba, (w, h), geo["pts"]
                    )
                    wx_min = max(0, wx_min)
                    wy_min = max(0, wy_min)
                    _blend_rgba_on_bgr(img_bgr, warped, wx_min, wy_min)
                    drawn_segments += 1
                else:
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    _blend_rgba_on_bgr(img_bgr, text_rgba, x_min, y_min)
                    drawn_segments += 1
            except Exception:
                continue

        # Safety net: if erase happened but styled text drawing produced nothing,
        # draw a simple readable overlay so users still get translated text on-photo.
        if not for_video and drawn_segments == 0:
            for row, trans_text in zip(results_to_draw, translated_to_draw):
                try:
                    bbox = row[1]
                    geo = _bbox_geometry(bbox, img_w, img_h)
                    x_min, y_min = max(0, geo["x_min"]), max(0, geo["y_min"])
                    x_max, y_max = min(img_w - 1, geo["x_max"]), min(img_h - 1, geo["y_max"])
                    w, h = geo["w"], geo["h"]
                    if w <= 10 or h <= 10:
                        continue
                    text = _photo_segment_display_text(trans_text)
                    if not text or _is_translation_failure_overlay(text):
                        continue
                    text = text.replace("\n", " ").strip()
                    if not text:
                        continue
                    tx = x_min + 4
                    ty_ref = min(y_max - 4, y_min + max(16, int(h * 0.65)))
                    self._draw_utf8_line_on_bgr(img_bgr, tx, ty_ref, text, h)
                except Exception:
                    continue

        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _render_paragraph_card(
        self,
        img_bgr: np.ndarray,
        paragraph_text: str,
        card_margin_frac: float = 0.04,
        card_height_frac: float = 0.13,
        text_color: Tuple[int, int, int] = (0, 0, 0),
        prefer_arabic: bool = False,
    ) -> np.ndarray:
        """Draw a white card at the bottom of the image and render paragraph_text on it (word-wrapped)."""
        if not (paragraph_text or "").strip():
            return img_bgr
        h_i, w_i = img_bgr.shape[:2]
        margin_x = max(16, int(w_i * card_margin_frac))
        margin_y = max(10, int(h_i * card_margin_frac))
        card_h = max(48, int(h_i * card_height_frac))
        card_y = h_i - card_h - margin_y
        card_x = margin_x
        card_w = w_i - 2 * margin_x
        if card_y < 0 or card_w < 50:
            return img_bgr
        # White card
        cv2.rectangle(img_bgr, (card_x, card_y), (card_x + card_w, card_y + card_h), (255, 255, 255), -1)
        cv2.rectangle(img_bgr, (card_x, card_y), (card_x + card_w, card_y + card_h), (200, 200, 200), 2)
        # One flowing paragraph: don't force new lines on full stop or newline; break by width only
        logical = _sanitize_rendered_text(
            _strip_ocr_uncertain_for_photo(
                paragraph_text.replace("\n", " ").replace("\r\n", " ").strip()
            )
        )
        if not logical:
            return img_bgr
        needs_bidi = _is_arabic_or_rtl(logical)

        def _disp(s: str) -> str:
            return _arabic_display_for_pil(s) if needs_bidi else s

        if prefer_arabic or needs_bidi:
            font_size = max(14, min(28, int(card_h * 0.24)))
        else:
            font_size = max(9, min(18, card_h // 11))
        font = self._get_font(font_size, prefer_arabic=(prefer_arabic or needs_bidi))
        card_pil = Image.new("RGBA", (card_w, card_h), (255, 255, 255, 255))
        draw = ImageDraw.Draw(card_pil)
        # Leave margin so words don't touch the edge or each other (avoid overlap)
        max_line_w = max(40, card_w - 24)
        word_gap = 4  # extra pixels between words so they don't overlap

        def measure(s: str) -> int:
            d = _disp(s)
            try:
                return font.getbbox(d)[2] - font.getbbox(d)[0] if hasattr(font, "getbbox") else len(s) * font_size
            except Exception:
                return len(s) * font_size

        words = logical.split()
        lines = []
        line_cur = ""
        for word in words:
            cand = (line_cur + " " + word).strip() if line_cur else word
            cw = measure(cand) + (word_gap if line_cur else 0)
            if cw <= max_line_w and line_cur:
                line_cur = (line_cur + " " + word).strip()
            else:
                if line_cur:
                    lines.append(line_cur)
                # If single token is too wide, break by character so it wraps (no overlap)
                if measure(word) > max_line_w:
                    chunk = ""
                    for ch in word:
                        trial = (chunk + ch) if chunk else ch
                        if measure(trial) <= max_line_w - word_gap:
                            chunk = trial
                        else:
                            if chunk:
                                lines.append(chunk)
                            chunk = ch
                    line_cur = chunk
                else:
                    line_cur = word
        if line_cur:
            lines.append(line_cur)
        if not lines:
            lines = [logical[: max(1, card_w // (font_size // 2))]]

        y_pos = 8
        bottom_margin = 12  # same for all languages so no overlap at bottom
        min_line_h = font_size + 6  # minimum gap between lines so they never overlap
        for line in lines:
            disp_line = _disp(line)
            try:
                lb = font.getbbox(disp_line) if hasattr(font, "getbbox") else (0, 0, len(line) * font_size, font_size)
                line_h = max(lb[3] - lb[1] + 2, min_line_h)
            except Exception:
                line_h = min_line_h
            if y_pos + line_h > card_h - bottom_margin:
                break
            lw = measure(line)
            x_pos = 8 if not _is_arabic_or_rtl(line) else max(8, card_w - lw - 8)
            draw.text((x_pos, y_pos), disp_line, font=font, fill=(*text_color, 255))
            y_pos += line_h
        card_rgba = np.array(card_pil)
        _blend_rgba_on_bgr(img_bgr, card_rgba, card_x, card_y)
        return img_bgr

    def render_translated_image(
        self,
        image_path: str,
        results: List[Tuple],
        translated_texts: List[str],
        background: str = "white",
        text_color: Tuple[int, int, int] = (0, 0, 0),
        output_path: Optional[str] = None,
        use_font_matching: bool = False,
        use_paragraph_card: bool = False,
        inpaint_style: str = "telea",
        merge_same_line_boxes: bool = False,
        preserve_original_font_size: bool = True,
        target_lang: Optional[str] = None,
        perspective_warp: Optional[bool] = None,
    ) -> np.ndarray:
        """Overlay translated text. Paragraph card = one bottom bar; otherwise Arabic is drawn in each OCR box (in-place)."""
        # Defensive normalization: some callers may pass partially malformed OCR rows.
        safe_results: List[Tuple] = []
        safe_texts: List[str] = []
        n = min(len(results or []), len(translated_texts or []))
        for i in range(n):
            row = results[i]
            txt = translated_texts[i]
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            bbox = row[1]
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 3:
                continue
            ok = True
            norm_pts = []
            for p in bbox:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    ok = False
                    break
                try:
                    x = float(p[0])
                    y = float(p[1])
                except Exception:
                    ok = False
                    break
                norm_pts.append([x, y])
            if not ok:
                continue
            if len(row) >= 4:
                safe_results.append((str(row[0]), norm_pts, str(row[2]), float(row[3])))
            elif len(row) == 3:
                safe_results.append((str(row[0]), norm_pts, str(row[2])))
            else:
                safe_results.append((str(row[0]), norm_pts))
            safe_texts.append("" if txt is None else str(txt))
        if not safe_results or not safe_texts:
            raise ValueError("No valid OCR boxes available for translated image rendering.")
        results = safe_results
        translated_texts = safe_texts

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # OpenCV imread returns BGR
        img_bgr = img
        if use_paragraph_card:
            # Order segments top-to-bottom so first line on screen is first in text (fixes Arabic line order)
            img_h, img_w = img_bgr.shape[:2]
            img_work = img_bgr.copy()
            res_p, txt_p = _suppress_overlapping_photo_overlays(
                list(results), list(translated_texts), img_w, img_h
            )
            _pstyle = (os.environ.get("PHOTO_OVERLAY_INPAINT") or inpaint_style or "telea").strip().lower()
            if _pstyle == "telea":
                _inpaint_remove_text_regions_bgr(img_work, res_p, img_w, img_h)
            else:
                _erase_text_regions_solid_local(img_work, res_p, img_w, img_h, tight_photo=True)
            order = _order_indices_top_to_bottom(res_p, img_w, img_h) if res_p else list(range(len(txt_p)))
            translated_ordered = [txt_p[i] for i in order] if order else txt_p
            raw_parts = [(t or "").strip() or "..." for t in translated_ordered]
            parts = [p for p in raw_parts if not _is_junk_segment(p)]
            if not parts:
                parts = [p for p in raw_parts if p]
            parts = _dedupe_paragraph_parts(parts, loose=True)
            paragraph = " ".join(parts) if parts else ""
            paragraph = paragraph.rstrip(" .<>-\t")
            if _is_junk_segment(paragraph):
                paragraph = ""
            out_bgr = self._render_paragraph_card(
                img_work,
                paragraph,
                text_color=text_color,
                prefer_arabic=any(_is_arabic_or_rtl(t or "") for t in translated_texts),
            )
            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            out_rgb = self._render_translated_on_image(
                img_rgb,
                results,
                translated_texts,
                background,
                text_color,
                use_font_matching,
                inpaint_style=inpaint_style,
                merge_same_line_boxes=merge_same_line_boxes,
                preserve_original_font_size=preserve_original_font_size,
                target_lang=target_lang,
                perspective_warp=perspective_warp,
            )
        if output_path:
            Image.fromarray(out_rgb).save(output_path)
        return out_rgb

    def render_translated_frame(
        self,
        frame_bgr: np.ndarray,
        results: List[Tuple],
        translated_texts: List[str],
        background: str = "white",
        text_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        """Overlay translated text on a single video frame (BGR in and out). Uses video-tuned font scale and outline for readability."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out_rgb = self._render_translated_on_image(
            frame_rgb, results, translated_texts, background, text_color, use_font_matching=False, for_video=True
        )
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    def _inpaint_frame_with_mask(
        self,
        frame_bgr: np.ndarray,
        bboxes_expanded: List[np.ndarray],
        fill_color_bgr: Optional[Tuple[int, int, int]] = None,
        extra_bboxes_expanded: Optional[List[np.ndarray]] = None,
        *,
        dilate_scale: float = 1.0,
        inpaint_radius_scale: float = 1.0,
        solid_white: bool = False,
    ) -> np.ndarray:
        """Build mask from expanded bboxes (and optional extra), dilate, inpaint.
        If solid_white=True: skip Telea inpaint (no blurry smudge) and fill the mask with white.
        If fill_color_bgr is set (without solid_white), inpaint then replace mask with that solid color.
        Default None keeps Telea inpainting so the background matches the scene.
        dilate_scale / inpaint_radius_scale < 1 shrink the effective repaired area (for video subtitles)."""
        img_h, img_w = frame_bgr.shape[:2]
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for pts in bboxes_expanded:
            if len(pts) >= 3:
                cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
        if extra_bboxes_expanded:
            for pts in extra_bboxes_expanded:
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
        if not np.any(mask > 0):
            return frame_bgr.copy()
        if solid_white:
            # Wider dilate than tiny Telea kernel — must cover anti-aliased fringes and OCR undershoot
            # or original glyphs still show at the edges; scale with resolution
            r1 = max(8, min(30, int(min(img_w, img_h) / 40.0)))
            r1 = max(r1, int(6 * dilate_scale))
            k1 = r1 * 2 + 1
            mask = cv2.dilate(
                mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1)),
                iterations=1,
            )
            r2 = max(3, r1 // 2)
            k2 = r2 * 2 + 1
            mask = cv2.dilate(
                mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2)),
                iterations=1,
            )
            out = frame_bgr.copy()
            out[mask > 0] = (255, 255, 255)
            return out
        # Extra dilation eats anti-aliased / halo pixels outside tight OCR quads
        base_d = max(18, min(img_w, img_h) // 28)
        dilate_px = max(6, int(base_d * dilate_scale))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
        radius = int(max(14, min(img_h, img_w) // 32) * inpaint_radius_scale)
        radius = max(6, radius)
        out = cv2.inpaint(frame_bgr, mask, radius, cv2.INPAINT_TELEA)
        # Second light pass on a slightly grown mask removes stubborn letter edges
        k2 = 5 if dilate_scale >= 0.9 else 3
        mask2 = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2)), iterations=1)
        r2 = max(4, int((radius // 2) * inpaint_radius_scale))
        out = cv2.inpaint(out, mask2, r2, cv2.INPAINT_TELEA)
        if fill_color_bgr is not None:
            b, g, r = fill_color_bgr
            out[mask > 0, 0] = b
            out[mask > 0, 1] = g
            out[mask > 0, 2] = r
        return out

    def _video_apply_text_background(
        self,
        frame_bgr: np.ndarray,
        expanded_list: List[np.ndarray],
        mask_mode: str,
        *,
        vid_inpaint_kw: Dict[str, Any],
    ) -> np.ndarray:
        """
        Prepare pixels behind translated text. ``inpaint`` = Telea (slowest); ``solid`` = white fill;
        ``blur`` = Gaussian blur inside polygons; ``rect_alpha`` = tinted translucent overlay.
        """
        mode = (mask_mode or "solid").strip().lower()
        _kw = {k: v for k, v in vid_inpaint_kw.items() if k != "solid_white"}
        if mode == "inpaint":
            return self._inpaint_frame_with_mask(frame_bgr, expanded_list, **_kw, solid_white=False)
        if mode in ("telea", "cv2_inpaint"):
            return self._inpaint_frame_with_mask(frame_bgr, expanded_list, **_kw, solid_white=False)
        if mode in ("solid", "white", "fill"):
            return self._inpaint_frame_with_mask(frame_bgr, expanded_list, **_kw, solid_white=True)
        if mode == "blur":
            try:
                sigma = float(os.environ.get("VIDEO_RENDER_BLUR_SIGMA", "14") or 14)
            except ValueError:
                sigma = 14.0
            sigma = max(4.0, min(40.0, sigma))
            return _video_blur_under_polys(frame_bgr, expanded_list, sigma)
        if mode in ("rect_alpha", "alpha_rect", "rect"):
            try:
                alpha = float(os.environ.get("VIDEO_RENDER_RECT_ALPHA", "0.82") or 0.82)
            except ValueError:
                alpha = 0.82
            b = int(os.environ.get("VIDEO_RENDER_RECT_B", "245") or 245)
            g = int(os.environ.get("VIDEO_RENDER_RECT_G", "245") or 245)
            r = int(os.environ.get("VIDEO_RENDER_RECT_R", "245") or 245)
            return _video_rect_alpha_under_polys(frame_bgr, expanded_list, alpha, (b, g, r))
        return self._inpaint_frame_with_mask(frame_bgr, expanded_list, **_kw, solid_white=True)

    def render_translated_video(
        self,
        video_path: str,
        per_frame_data: List[Tuple[int, List[Tuple]]],
        translated_list: List[str],
        output_path: str,
        use_tracking: bool = True,
        mask_padding_px: int = DEFAULT_MASK_PADDING_PX,
        mask_padding_ratio: float = DEFAULT_MASK_PADDING_RATIO,
        use_paragraph_card: bool = False,
        fast_render: bool = False,
    ) -> None:
        """
        Write a new video with translated text overlaid.
        If use_paragraph_card=True: remove original text per frame (inpaint using nearest key bboxes),
        draw one white card at bottom with only that frame's sentence (time-synced).
        fast_render=True: paragraph card uses white-fill only (no inpaint), faster.

        Uses FFmpeg piped BGR24 decode + encode when ffmpeg is on PATH or bundled (imageio-ffmpeg);
        MP4 is written directly as H.264 (no OpenCV mp4v intermediate). Override with VIDEO_USE_OPENCV_VIDEO_IO=1.

        Performance env (summary): ``VIDEO_RENDER_WORK_HEIGHT`` — scale processing/effective export resolution;
        ``VIDEO_RENDER_MASK_MODE`` = ``solid`` | ``inpaint`` | ``blur`` | ``rect_alpha``; ``VIDEO_RENDER_NVENC`` (default on)
        selects ``h264_nvenc`` when available; ``VIDEO_RENDER_CUDA_DECODE=1`` tries CUDA decode (needs FFmpeg + NVIDIA).
        ``VIDEO_RENDER_PROFILE=1`` logs timing breakdown to stdout. Text drawing stays in Pillow for RTL/multi-line alignment
        (FFmpeg ``drawtext`` cannot reproduce that faithfully).
        """
        if not per_frame_data or sum(len(r) for _, r in per_frame_data) != len(translated_list):
            raise ValueError("per_frame_data and translated_list length mismatch")
        idx = 0
        frame_map: Dict[int, Tuple[List[Tuple], List[str]]] = {}
        for frame_idx, results_for_frame in per_frame_data:
            n = len(results_for_frame)
            frame_map[frame_idx] = (results_for_frame, translated_list[idx : idx + n])
            idx += n
        sampled_indices = sorted(frame_map.keys())
        if not sampled_indices:
            raise ValueError("No frames with text to overlay")

        # Keys that have at least one OCR result (so we never use empty overlay)
        keys_with_text = [k for k in sampled_indices if frame_map[k][0]]

        def overlay_key_for_frame(i: int) -> int:
            """Nearest preceding sampled frame (for use_tracking=False or before first sample)."""
            if i <= sampled_indices[0]:
                return sampled_indices[0]
            for k in reversed(sampled_indices):
                if k <= i:
                    return k
            return sampled_indices[0]

        def overlay_key_with_text(i: int) -> int:
            """Temporally closest key that has OCR text (so overlay matches this frame as much as possible)."""
            if not keys_with_text:
                return sampled_indices[0]
            # Use the key whose frame index is closest to i
            return min(keys_with_text, key=lambda k: abs(k - i))

        mask_mode_env = (os.environ.get("VIDEO_RENDER_MASK_MODE") or "").strip().lower()
        if mask_mode_env:
            mask_mode = mask_mode_env
        elif fast_render:
            mask_mode = "rect_alpha"
        else:
            mask_mode = "solid"

        prof: Dict[str, Any] = {"decode_s": 0.0, "mask_s": 0.0, "draw_s": 0.0, "encode_write_s": 0.0, "frames": 0}

        meta = _probe_video_meta_ffprobe(video_path) or _probe_video_meta_opencv(video_path)
        if not meta:
            raise ValueError(f"Could not open video: {video_path}")
        src_w, src_h, fps = meta
        if src_w <= 0 or src_h <= 0:
            raise ValueError("Invalid video dimensions")

        try:
            max_work_h = int(os.environ.get("VIDEO_RENDER_WORK_HEIGHT") or "0")
        except ValueError:
            max_work_h = 0
        w, h_vid = src_w, src_h
        if max_work_h > 0 and src_h > max_work_h:
            h_vid = max(2, max_work_h - (max_work_h % 2))
            w = max(2, int(round(src_w * h_vid / src_h)) - (int(round(src_w * h_vid / src_h)) % 2))
            sx = w / float(src_w)
            sy = h_vid / float(src_h)
            frame_map = _scale_video_frame_map_geometry(frame_map, sx, sy)
            keys_with_text = [k for k in sampled_indices if frame_map[k][0]]

        # Balance: large enough to erase all source pixels (incl. halos), not as wide as full Telea defaults
        vid_pad_px = max(22, int(mask_padding_px * 0.95 * (h_vid / float(src_h))))
        vid_pad_ratio = mask_padding_ratio * 0.92
        _vid_inpaint_kw = {"dilate_scale": 1.12, "inpaint_radius_scale": 0.95}
        w_out = max(2, w - (w % 2))
        h_out = max(2, h_vid - (h_vid % 2))

        def _frame_for_writer(f: np.ndarray) -> np.ndarray:
            if f is None or f.size == 0:
                return f
            if f.shape[1] == w_out and f.shape[0] == h_out:
                return f
            return _safe_cv2_resize(f, w_out, h_out)

        use_avi = output_path.lower().endswith(".avi")
        use_ff_io = _use_ffmpeg_video_io()

        cap: Optional[cv2.VideoCapture] = None
        dec_proc: Optional[subprocess.Popen] = None
        out: Any = None
        enc_proc: Optional[subprocess.Popen] = None
        encode_path = output_path
        mp4_intermediate: Optional[str] = None

        if use_ff_io:
            enc_proc = _ffmpeg_encode_raw_bgr_start(
                output_path, w_out, h_out, fps, use_avi=use_avi
            )
            dec_proc = _ffmpeg_decode_raw_bgr_start(video_path, w_out, h_out)
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            # Avoid avc1/H264 in OpenCV on Windows — triggers broken OpenH264 DLL. Write mp4v to temp, then FFmpeg -> H.264.
            if not use_avi and output_path.lower().endswith(".mp4"):
                out_dir = os.path.dirname(os.path.abspath(output_path))
                try:
                    fd, mp4_intermediate = tempfile.mkstemp(suffix=".cv2.mp4", dir=out_dir if out_dir else None)
                    os.close(fd)
                    encode_path = mp4_intermediate
                except OSError:
                    mp4_intermediate = None
                    encode_path = output_path

            if use_avi:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                out = cv2.VideoWriter(encode_path, fourcc, fps, (w_out, h_out))
            else:
                out = None
                for codec in ("mp4v", "XVID"):
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        out = cv2.VideoWriter(encode_path, fourcc, fps, (w_out, h_out))
                        if out is not None and out.isOpened():
                            break
                        if out is not None:
                            out.release()
                        out = None
                    except Exception:
                        out = None
            if out is None or not out.isOpened():
                if mp4_intermediate and os.path.isfile(mp4_intermediate):
                    try:
                        os.unlink(mp4_intermediate)
                    except OSError:
                        pass
                cap.release()
                raise ValueError(
                    "Video writer could not be opened for this format. "
                    "Try output filename .avi or install OpenCV built with FFmpeg support."
                )

        def _read_frame_bgr() -> Optional[np.ndarray]:
            t0 = time.perf_counter()
            try:
                if dec_proc is not None:
                    fr = _ffmpeg_read_bgr_frame(dec_proc, w, h_vid)
                    if fr is None:
                        return None
                    if fr.shape[:2] != (h_vid, w):
                        fr = _safe_cv2_resize(fr, w, h_vid)
                    return fr
                assert cap is not None
                ret, frame = cap.read()
                if not ret:
                    return None
                if frame.shape[:2] != (h_vid, w):
                    frame = _safe_cv2_resize(frame, w, h_vid)
                return frame
            finally:
                prof["decode_s"] += time.perf_counter() - t0

        def _write_frame_out(frame: np.ndarray) -> None:
            t0 = time.perf_counter()
            try:
                fw = _frame_for_writer(frame)
                if fw is None or fw.size == 0:
                    return
                contiguous = np.ascontiguousarray(fw, dtype=np.uint8)
                if enc_proc is not None:
                    assert enc_proc.stdin is not None
                    enc_proc.stdin.write(contiguous.tobytes())
                else:
                    assert out is not None
                    out.write(fw)
                prof["frames"] += 1
            finally:
                prof["encode_write_s"] += time.perf_counter() - t0

        def _cleanup_video_io() -> None:
            if dec_proc is not None:
                try:
                    if dec_proc.stdout:
                        dec_proc.stdout.close()
                except Exception:
                    pass
                try:
                    dec_proc.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    dec_proc.kill()
            if cap is not None:
                cap.release()
            if enc_proc is not None:
                try:
                    if enc_proc.stdin:
                        enc_proc.stdin.close()
                except Exception:
                    pass
                try:
                    enc_proc.wait(timeout=7200)
                except subprocess.TimeoutExpired:
                    enc_proc.kill()
            elif out is not None:
                out.release()
            if not use_ff_io and not use_avi and output_path.lower().endswith(".mp4"):
                _finalize_mp4_after_opencv(encode_path, output_path)

        def _paragraph_and_overlay() -> None:
            if use_paragraph_card:
                # Per frame: remove original text (inpaint with TIGHT padding so we don't cover the whole frame),
                # then show one sentence for this moment on a card at the bottom only.
                prefer_arabic = any(_is_arabic_or_rtl(t or "") for t in translated_list)
                # Enough pad that source text does not peek past the white fill before the bottom card
                card_inpaint_px = max(18, int(mask_padding_px * 0.72 * (h_vid / float(src_h))))
                card_inpaint_ratio = min(0.22, mask_padding_ratio * 0.55)
                frame_idx = 0
                while True:
                    frame = _read_frame_bgr()
                    if frame is None:
                        break
                    key = overlay_key_with_text(frame_idx)
                    results_f, txts_f = frame_map[key]
                    if results_f:
                        results_f, txts_f = _merge_boxes_same_line(results_f, txts_f, w, h_vid)
                        results_f, txts_f = _merge_video_duplicate_translation_boxes(results_f, txts_f, w, h_vid)
                        results_f, txts_f = _merge_video_adjacent_boxes(results_f, txts_f, w, h_vid)
                    # Remove original text: same solid-white erase for fast and normal (fast skips heavier work elsewhere only)
                    if results_f:
                        expanded_list = [
                            _expand_bbox(row[1], w, h_vid, card_inpaint_px, card_inpaint_ratio)
                            for row in results_f
                        ]
                        t_mask = time.perf_counter()
                        frame = self._video_apply_text_background(
                            frame, expanded_list, mask_mode, vid_inpaint_kw=_vid_inpaint_kw
                        )
                        prof["mask_s"] += time.perf_counter() - t_mask
                    # Order segments top-to-bottom so first line on screen is first in text (fixes Arabic line order)
                    order = _order_indices_top_to_bottom(results_f, w, h_vid) if results_f else list(range(len(txts_f)))
                    txts_ordered = [txts_f[i] for i in order] if order else txts_f
                    raw_parts = [(t or "").strip() or "..." for t in txts_ordered]
                    parts = [p for p in raw_parts if not _is_junk_segment(p)]
                    if not parts:
                        parts = [p for p in raw_parts if p]
                    parts = _dedupe_paragraph_parts(parts, loose=True)
                    sentence = " ".join(parts) if parts else ""
                    sentence = sentence.rstrip(" .<>-\t")
                    if _is_junk_segment(sentence):
                        sentence = ""
                    t_draw = time.perf_counter()
                    frame = self._render_paragraph_card(frame, sentence, text_color=(0, 0, 0), prefer_arabic=prefer_arabic)
                    prof["draw_s"] += time.perf_counter() - t_draw
                    _write_frame_out(frame)
                    frame_idx += 1
                return

            # Trackers: list of {tracker, trans_text, last_rect (x,y,w,h)}
            trackers: List[Dict[str, Any]] = []
            debug_overlay = os.environ.get("OCR_VIDEO_OVERLAY_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on")
            try:
                blend_alpha = float(os.environ.get("OCR_VIDEO_OVERLAY_ALPHA", "1.0"))
            except ValueError:
                blend_alpha = 1.0
            blend_alpha = max(0.0, min(1.0, blend_alpha))

            def _round_bbox_points(results_in: List[Tuple]) -> List[Tuple]:
                out_rows: List[Tuple] = []
                for row in results_in or []:
                    if not isinstance(row, (list, tuple)) or len(row) < 2:
                        continue
                    pts = np.array(row[1], dtype=float)
                    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
                        continue
                    pts_i = np.round(pts[:, :2]).astype(np.int32).tolist()
                    out_rows.append((row[0], pts_i, row[2], row[3] if len(row) > 3 else 0.5))
                return out_rows

            def _render_frame_overlay_once(
                base_frame_bgr: np.ndarray,
                results_in: List[Tuple],
                txts_in: List[str],
            ) -> Tuple[np.ndarray, int, int]:
                if base_frame_bgr is None or base_frame_bgr.size == 0:
                    return base_frame_bgr, 0, 0
                if not results_in or not txts_in:
                    return base_frame_bgr, 0, 0
                merged_r, merged_t = _merge_boxes_same_line(results_in, txts_in, w, h_vid)
                merged_r, merged_t = _merge_video_duplicate_translation_boxes(merged_r, merged_t, w, h_vid)
                merged_r, merged_t = _merge_video_adjacent_boxes(merged_r, merged_t, w, h_vid)
                merged_r, merged_t = _dedupe_overlay_segments_for_frame(merged_r, merged_t)
                clean_r = _round_bbox_points(merged_r)
                if len(clean_r) != len(merged_t):
                    n = min(len(clean_r), len(merged_t))
                    clean_r, merged_t = clean_r[:n], merged_t[:n]
                if not clean_r:
                    return base_frame_bgr, 0, 0
                t_draw = time.perf_counter()
                try:
                    overlay_frame = base_frame_bgr.copy()
                    overlay_rgb = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
                    overlay_rgb = self._render_translated_on_image(
                        overlay_rgb,
                        clean_r,
                        merged_t,
                        background="white",
                        text_color=(0, 0, 0),
                        use_inpainting=False,
                        for_video=True,
                        use_font_matching=False,
                        match_original_text_style=False,
                    )
                    overlay_frame = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                    # Single blend pass only (or direct overlay when alpha=1).
                    if blend_alpha >= 0.999:
                        return overlay_frame, len(clean_r), 1
                    final_frame = cv2.addWeighted(
                        base_frame_bgr, 1.0 - blend_alpha, overlay_frame, blend_alpha, 0
                    )
                    return final_frame, len(clean_r), 1
                finally:
                    prof["draw_s"] += time.perf_counter() - t_draw

            frame_idx = 0
            while True:
                frame = _read_frame_bgr()
                if frame is None:
                    break
                if frame.shape[:2] != (h_vid, w):
                    frame = _safe_cv2_resize(frame, w, h_vid)
                base_frame = frame.copy()
                final_frame = base_frame
                overlay_boxes = 0
                draw_calls = 0
                if frame_idx in frame_map:
                    results_f, txts_f = frame_map[frame_idx]
                    # If this sample has no text (e.g. frame 0), use nearest key that has text so we still inpaint/overlay
                    if not results_f and keys_with_text:
                        key = overlay_key_with_text(frame_idx)
                        results_f, txts_f = frame_map[key]
                    if results_f:
                        # Merge to one mask per line first so gaps between words are inpainted too.
                        results_f, txts_f = _merge_boxes_same_line(results_f, txts_f, w, h_vid)
                        results_f, txts_f = _merge_video_duplicate_translation_boxes(results_f, txts_f, w, h_vid)
                        results_f, txts_f = _merge_video_adjacent_boxes(results_f, txts_f, w, h_vid)
                        results_f, txts_f = _dedupe_overlay_segments_for_frame(results_f, txts_f)
                        results_f = _round_bbox_points(results_f)
                        expanded_list = [
                            _expand_bbox(row[1], w, h_vid, vid_pad_px, vid_pad_ratio)
                            for row in results_f
                        ]
                        t_mask = time.perf_counter()
                        base_frame = self._video_apply_text_background(
                            base_frame, expanded_list, mask_mode, vid_inpaint_kw=_vid_inpaint_kw
                        )
                        prof["mask_s"] += time.perf_counter() - t_mask
                        # (Re-)init trackers from merged regions (one tracker per line)
                        trackers = []
                        for row, trans_text in zip(results_f, txts_f):
                            orig_text, bbox, lang = row[0], row[1], row[2]
                            geo = _bbox_geometry(bbox, w, h_vid)
                            x0, y0, rw, rh = geo["x_min"], geo["y_min"], geo["w"], geo["h"]
                            if rw <= 0 or rh <= 0:
                                continue
                            tracker = _create_tracker()
                            if tracker is not None and use_tracking:
                                try:
                                    tracker.init(base_frame, (x0, y0, rw, rh))
                                except Exception:
                                    tracker = None
                            trackers.append({
                                "tracker": tracker,
                                "trans_text": trans_text,
                                "last_rect": (x0, y0, rw, rh),
                            })
                        final_frame, overlay_boxes, draw_calls = _render_frame_overlay_once(
                            base_frame, results_f, txts_f
                        )
                else:
                    # Non-sampled frame
                    if use_tracking and trackers:
                        # Update trackers; use extra padding on tracked boxes so small tracker drift doesn't leave original text visible
                        current_results = []
                        current_texts = []
                        expanded_list = []
                        track_pad_px = max(40, int((mask_padding_px * 1.05 + 14) * (h_vid / float(src_h))))
                        track_pad_ratio = min(0.44, mask_padding_ratio * 1.45)
                        for ent in trackers:
                            trans_text = ent["trans_text"]
                            x0, y0, rw, rh = ent["last_rect"]
                            if ent["tracker"] is not None:
                                try:
                                    ok, new_rect = ent["tracker"].update(frame)
                                    if ok:
                                        x0, y0, rw, rh = (int(round(x)) for x in new_rect)
                                        ent["last_rect"] = (x0, y0, rw, rh)
                                except Exception:
                                    pass
                            bbox_4pt = _rect_to_bbox_4pt(x0, y0, max(1, rw), max(1, rh))
                            expanded = _expand_bbox(bbox_4pt.tolist(), w, h_vid, track_pad_px, track_pad_ratio)
                            expanded_list.append(expanded)
                            current_results.append(("", bbox_4pt.tolist(), "", 0.5))
                            current_texts.append(trans_text)
                        if expanded_list:
                            t_mask = time.perf_counter()
                            base_frame = self._video_apply_text_background(
                                base_frame, expanded_list, mask_mode, vid_inpaint_kw=_vid_inpaint_kw
                            )
                            prof["mask_s"] += time.perf_counter() - t_mask
                        if current_results and current_texts:
                            final_frame, overlay_boxes, draw_calls = _render_frame_overlay_once(
                                base_frame, current_results, current_texts
                            )
                    else:
                        # No tracking or no trackers: use a key that has text so we always inpaint/overlay
                        key = overlay_key_with_text(frame_idx)
                        results_f, txts_f = frame_map[key]
                        if results_f:
                            results_f, txts_f = _merge_boxes_same_line(results_f, txts_f, w, h_vid)
                            results_f, txts_f = _merge_video_duplicate_translation_boxes(results_f, txts_f, w, h_vid)
                            results_f, txts_f = _merge_video_adjacent_boxes(results_f, txts_f, w, h_vid)
                            results_f, txts_f = _dedupe_overlay_segments_for_frame(results_f, txts_f)
                            results_f = _round_bbox_points(results_f)
                            extra_px = max(36, int(mask_padding_px * 1.62 * (h_vid / float(src_h))))
                            extra_ratio = min(0.42, mask_padding_ratio * 1.28)
                            expanded_list = [
                                _expand_bbox(row[1], w, h_vid, extra_px, extra_ratio)
                                for row in results_f
                            ]
                            t_mask = time.perf_counter()
                            base_frame = self._video_apply_text_background(
                                base_frame, expanded_list, mask_mode, vid_inpaint_kw=_vid_inpaint_kw
                            )
                            prof["mask_s"] += time.perf_counter() - t_mask
                            final_frame, overlay_boxes, draw_calls = _render_frame_overlay_once(
                                base_frame, results_f, txts_f
                            )
                if debug_overlay:
                    print(
                        f"[video-overlay] frame={frame_idx} boxes={overlay_boxes} "
                        f"draw_calls={draw_calls} processed_once=1"
                    )
                if final_frame is not None:
                    if final_frame.shape[:2] != (h_vid, w):
                        final_frame = _safe_cv2_resize(final_frame, w, h_vid)
                    _write_frame_out(final_frame)
                frame_idx += 1

        nvenc_available = bool(_ffmpeg_h264_nvenc_available() and not use_avi)

        try:
            _paragraph_and_overlay()
        finally:
            _cleanup_video_io()

        try:
            up_h = int(os.environ.get("VIDEO_RENDER_OUTPUT_UPSCALE_HEIGHT") or "0")
        except ValueError:
            up_h = 0
        if up_h > src_h and os.path.isfile(output_path) and output_path.lower().endswith(".mp4") and not use_avi:
            tmp_u = output_path + ".upscale.tmp.mp4"
            try:
                if _ffmpeg_upscale_video_to_height(output_path, tmp_u, up_h):
                    os.replace(tmp_u, output_path)
            except OSError:
                pass
            finally:
                if os.path.isfile(tmp_u):
                    try:
                        os.unlink(tmp_u)
                    except OSError:
                        pass

        prof["src_wh"] = (src_w, src_h)
        prof["work_wh"] = (w_out, h_out)
        prof["mask_mode"] = mask_mode
        prof["h264_nvenc_available"] = nvenc_available
        self.last_video_render_profile = dict(prof)
        if (os.environ.get("VIDEO_RENDER_PROFILE") or "").strip().lower() in ("1", "true", "yes", "on"):
            print(
                "[video-render-profile] "
                f"decode={prof['decode_s']:.3f}s mask={prof['mask_s']:.3f}s draw={prof['draw_s']:.3f}s "
                f"encode_write={prof['encode_write_s']:.3f}s frames={int(prof['frames'])} "
                f"mask_mode={mask_mode} work={w_out}x{h_out} src={src_w}x{src_h} "
                f"h264_nvenc_available={nvenc_available}"
            )

    def draw_bounding_boxes_on_frame(
        self,
        frame_bgr: np.ndarray,
        results: List[Tuple],
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes and OCR labels on a BGR frame (e.g. one video frame).

        Labels are rendered **inside** each OCR rectangle (word-wrapped, font scaled from the
        source line like the translated-photo overlay) and **warped to the OCR quadrilateral**
        when perspective is enabled, so tilted menu text follows the printed layout.

        Twin boxes for the same text at the same spot are dropped by default (quantized
        position + normalized text); repeated labels elsewhere on the page are kept.
        Set ``OCR_DRAW_BOXES_DEDUPE=0`` to draw every raw OCR box.
        Each box is drawn in cyan.

        Returns:
            RGB ``uint8`` ndarray (same convention as ``draw_bounding_boxes``).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Empty frame for bounding box overlay.")
        if len(frame_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = frame_bgr.copy()

        ih, iw = img_bgr.shape[:2]
        rows: List[Tuple] = list(results)
        dedupe_raw = (os.environ.get("OCR_DRAW_BOXES_DEDUPE", "1") or "1").strip().lower()
        if dedupe_raw not in ("0", "false", "no") and rows:
            texts = [(_ocr_row_unpack(r)[0] or "").strip() for r in rows]
            rows, _ = _dedupe_overlay_segments_for_frame(rows, texts)

        for idx, row in enumerate(rows):
            bbox = _ocr_row_unpack(row)[1]
            bbox_i = np.array(bbox, dtype=np.int32)
            cv2.polylines(img_bgr, [bbox_i], True, _ocr_annot_box_color_bgr(idx), 2)

        # Draw cyan polygons only; skip OCR text inside/on boxes (clean Annotated view).
        # Set OCR_DRAW_BOXES_HIDE_LABELS=0 to also render detected text in each box.
        _hide_labels_raw = (os.environ.get("OCR_DRAW_BOXES_HIDE_LABELS") or "").strip().lower()
        if _hide_labels_raw in ("0", "false", "no", "off"):
            _hide_labels = False
        else:
            _hide_labels = True
        if _hide_labels:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            out_rgb = np.asarray(img_rgb)
            if output_path:
                cv2.imwrite(output_path, cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
            return out_rgb

        _allow_perspective = _should_apply_perspective_warp(False, None)

        for idx, row in enumerate(rows):
            try:
                text_raw, bbox, _, _ = _ocr_row_unpack(row)
                label_rgb = _ocr_annot_box_color_rgb(idx)
                label_raw = (text_raw or "").strip()
                if not label_raw:
                    continue
                label_for_draw = _apply_overlay_english_ocr_typo_fixes(label_raw)
                label_for_draw = _sanitize_rendered_text(label_for_draw)
                if len(label_for_draw) > 2000:
                    label_for_draw = label_for_draw[:1997] + "…"

                geo = _bbox_geometry(bbox, iw, ih)
                w, h = int(geo["w"]), int(geo["h"])
                if w < 6 or h < 6:
                    continue
                x_min, y_min = int(geo["x_min"]), int(geo["y_min"])

                target_fs: Optional[int] = None
                try:
                    target_fs = _estimate_font_size_from_original_text(
                        label_raw,
                        w,
                        h,
                        lambda s: self._get_font(s, prefer_arabic=False),
                    )
                except Exception:
                    target_fs = None

                is_rtl = _is_arabic_or_rtl(label_for_draw)

                def get_font_fn(size: int, _rtl=is_rtl):
                    return self._get_font(size, prefer_arabic=_rtl)

                text_layer_pil, _ = _render_text_into_box_pillow(
                    label_for_draw,
                    w,
                    h,
                    get_font_fn,
                    is_rtl,
                    text_color=label_rgb,
                    bg_color=None,
                    for_video=False,
                    target_font_size=target_fs,
                    style_like_rtl=False,
                    force_outline=True,
                )
                text_rgba = np.array(text_layer_pil)
                if text_rgba.size == 0:
                    continue
                if _allow_perspective and len(geo["pts"]) >= 4:
                    warped, wx_min, wy_min, _, _ = _warp_text_layer_to_polygon(
                        text_rgba, (w, h), geo["pts"]
                    )
                    wx_min = max(0, int(wx_min))
                    wy_min = max(0, int(wy_min))
                    _blend_rgba_on_bgr(img_bgr, warped, wx_min, wy_min)
                else:
                    _blend_rgba_on_bgr(img_bgr, text_rgba, max(0, x_min), max(0, y_min))
            except Exception:
                continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out_rgb = np.asarray(img_rgb)
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))

        return out_rgb

    def draw_bounding_boxes(self, image_path: str, results: List[Tuple], 
                           output_path: Optional[str] = None) -> np.ndarray:
        """
        Draw bounding boxes around detected text.
        
        Args:
            image_path: Path to the original image
            results: OCR results with bounding boxes
            output_path: Optional path to save the annotated image
            
        Returns:
            RGB ``uint8`` ndarray with bounding boxes (matches Pillow / Streamlit / ``matplotlib``;
            ``output_path`` is still written as standard BGR by ``cv2.imwrite``).
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image for overlay: {image_path}")

        return self.draw_bounding_boxes_on_frame(img_bgr, results, output_path=output_path)


# Hot-reload / out-of-sync copies: app.py calls extract_text_for_overlay; bind if missing.
if not hasattr(OCRTranslator, "extract_text_for_overlay"):
    def _extract_text_for_overlay_fallback(
        self,
        image_path: str,
        preprocess: bool = True,
        max_dim: int = 1200,
        dewarp_screen: bool = False,
        small_text_boost: bool = False,
        high_recall: bool = False,
    ) -> Tuple[List[Tuple], str]:
        return (
            self.extract_text(
                image_path,
                preprocess=preprocess,
                max_dim=max_dim,
                small_text_boost=small_text_boost,
                high_recall=high_recall,
            ),
            image_path,
        )

    OCRTranslator.extract_text_for_overlay = _extract_text_for_overlay_fallback
