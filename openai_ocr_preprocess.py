"""
Optional OpenAI Images API pass to improve photo legibility before local OCR.

Uses ``images.edit`` (e.g. gpt-image-1.5) with a strict prompt: enhance contrast and reduce
noise/glare without adding, removing, or changing any text.

Enable: UI checkbox **OpenAI preprocess before OCR** or ``OCR_PREPROCESS_OPENAI=1``.

After a successful edit, local GPU/OpenCV denoise is skipped by default
(``OCR_PREPROCESS_OPENAI_ONLY=1``). When ``max_dim`` is 0, the OpenAI upload and
returned OCR image keep the original OCR canvas size.

Costs/latency: similar to translated-photo edits (~30–90s, billed per OpenAI image call).
Requires ``OPENAI_API_KEY``. Tune via ``OPENAI_IMAGE_ENHANCE_*`` or ``OCR_PREPROCESS_OPENAI_*``.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from openai_image_photo import (
    _compute_cache_key,
    _cache_enabled,
    _cache_lookup,
    _cache_store,
    _get_client,
    _normalize_image_path_for_edit,
    default_image_model,
    image_bytes_to_rgb_numpy,
)


_DEFAULT_PROMPT = (
    "Edit this photograph to maximize printed and handwritten text legibility for OCR. "
    "Improve lighting, contrast, and sharpness; reduce noise, blur, and glare. "
    "If the page or screen is noticeably tilted, straighten it while keeping all content in frame. "
    "CRITICAL: Do not add, remove, change, translate, or invent any text. "
    "Every character and word must remain exactly as in the original. "
    "Do not crop away text. Preserve layout, colors, and background."
)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _read_ocr_openai_options() -> Tuple[str, str, str, str]:
    quality = (os.environ.get("OCR_PREPROCESS_OPENAI_QUALITY") or "").strip()
    if not quality:
        quality = (os.environ.get("OPENAI_IMAGE_ENHANCE_QUALITY") or "").strip()
    if not quality:
        quality = "medium" if _env_truthy("OPENAI_IMAGE_ENHANCE_FAST", False) else "high"

    size = (os.environ.get("OCR_PREPROCESS_OPENAI_SIZE") or "").strip()
    if not size:
        size = (os.environ.get("OPENAI_IMAGE_ENHANCE_SIZE") or "").strip() or "auto"

    fidelity = (os.environ.get("OCR_PREPROCESS_OPENAI_INPUT_FIDELITY") or "").strip()
    if not fidelity:
        fidelity = (os.environ.get("OPENAI_IMAGE_ENHANCE_INPUT_FIDELITY") or "").strip() or "high"

    model = (os.environ.get("OCR_PREPROCESS_OPENAI_MODEL") or "").strip()
    if not model:
        model = (os.environ.get("OPENAI_IMAGE_ENHANCE_MODEL") or "").strip() or default_image_model()
    return quality, size, fidelity, model


def _resize_bgr_max_dim(bgr: np.ndarray, max_dim: int) -> np.ndarray:
    if max_dim <= 0 or bgr is None or bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    if max(h, w) <= max_dim:
        return bgr
    scale = max_dim / float(max(h, w))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def edit_photo_for_ocr(
    image_path: str,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    quality: Optional[str] = None,
    size: Optional[str] = None,
    input_fidelity: Optional[str] = None,
    perf_stats: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Call OpenAI ``images.edit`` for OCR-oriented legibility; return raw image bytes."""
    q, sz, fid, m = _read_ocr_openai_options()
    if quality:
        q = quality
    if size:
        sz = size
    if input_fidelity:
        fid = input_fidelity
    m = (model or m).strip()

    prompt = (os.environ.get("OCR_PREPROCESS_OPENAI_PROMPT") or "").strip() or _DEFAULT_PROMPT

    t0 = time.perf_counter()
    path, is_temp = _normalize_image_path_for_edit(image_path)
    if perf_stats is not None:
        perf_stats["openai_normalize_image_s"] = time.perf_counter() - t0

    try:
        try:
            with open(path, "rb") as nf:
                norm_bytes = nf.read()
        except OSError:
            norm_bytes = b""

        cache_key_hex = _compute_cache_key(
            norm_bytes,
            prompt,
            model=m,
            quality=q,
            size=sz,
            input_fidelity=fid,
            dedupe_retry=False,
        )
        if _cache_enabled() and norm_bytes:
            cached = _cache_lookup(cache_key_hex)
            if cached is not None:
                if perf_stats is not None:
                    perf_stats["openai_cache_hit"] = 1
                    perf_stats["openai_images_edit_s"] = 0.0
                return cached
        if perf_stats is not None:
            perf_stats["openai_cache_hit"] = 0

        client = _get_client(api_key)
        t_api = time.perf_counter()
        with open(path, "rb") as img_f:
            resp = client.images.edit(
                model=m,
                image=img_f,
                prompt=prompt,
                quality=q,  # type: ignore[arg-type]
                size=sz,  # type: ignore[arg-type]
                input_fidelity=fid,  # type: ignore[arg-type]
            )
        if not getattr(resp, "data", None):
            raise RuntimeError("OpenAI Images returned no data.")
        item = resp.data[0]
        b64 = getattr(item, "b64_json", None)
        if not b64 and hasattr(item, "model_dump"):
            b64 = item.model_dump().get("b64_json")
        if not b64:
            raise RuntimeError("OpenAI Images response missing image data (b64_json).")
        import base64

        out_b = base64.b64decode(b64)
        if perf_stats is not None:
            perf_stats["openai_images_edit_s"] = time.perf_counter() - t_api
            perf_stats["openai_model"] = m
            perf_stats["openai_quality"] = q

        if _cache_enabled() and norm_bytes:
            _cache_store(cache_key_hex, out_b, f"ocr_preprocess model={m} quality={q}")

        return out_b
    finally:
        if is_temp and path and os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass


def openai_preprocess_bgr_for_ocr(
    bgr: np.ndarray,
    *,
    api_key: Optional[str] = None,
    max_dim: int = 1200,
    perf_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run OpenAI image edit on a BGR uint8 array.

    When ``max_dim`` is 0, the OpenAI upload is not downscaled and the returned
    BGR is resized back to the original OCR canvas if the API changes dimensions.
    """
    info: Dict[str, Any] = {"openai_used": False}
    if bgr is None or bgr.size == 0:
        info["openai_skip"] = "empty"
        return bgr, info
    original_h, original_w = bgr.shape[:2]

    fd, tmp_in = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        if not cv2.imwrite(tmp_in, bgr):
            info["openai_skip"] = "imwrite_failed"
            return bgr, info
        prev_max_side = os.environ.get("OPENAI_IMAGE_EDIT_MAX_SIDE")
        if max_dim <= 0:
            os.environ["OPENAI_IMAGE_EDIT_MAX_SIDE"] = "0"
        try:
            raw = edit_photo_for_ocr(tmp_in, api_key=api_key, perf_stats=perf_stats)
        finally:
            if max_dim <= 0:
                if prev_max_side is None:
                    os.environ.pop("OPENAI_IMAGE_EDIT_MAX_SIDE", None)
                else:
                    os.environ["OPENAI_IMAGE_EDIT_MAX_SIDE"] = prev_max_side
        rgb = image_bytes_to_rgb_numpy(raw)
        out_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if max_dim <= 0 and out_bgr.shape[:2] != (original_h, original_w):
            out_bgr = cv2.resize(out_bgr, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
            info["openai_resized_back_to_original"] = True
        else:
            out_bgr = _resize_bgr_max_dim(out_bgr, max_dim)
        info["openai_used"] = True
        info["shape_out"] = out_bgr.shape[:2]
        return out_bgr, info
    except Exception as e:
        info["openai_skip"] = repr(e)
        return bgr, info
    finally:
        try:
            os.unlink(tmp_in)
        except OSError:
            pass
