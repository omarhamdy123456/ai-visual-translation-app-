"""
GPU-first OCR image preprocessing (CUDA when available).

Goals: strong readability for menus / documents / scene text while avoiding halos,
washed highlights, and plastic over-smoothing. Keeps ``torch.Tensor`` BCHW ``[0,1]`` on GPU
through denoise → corner-aware illumination → edge-weighted local contrast → edge-aware sharpen
→ luma micro-contrast (fine strokes) without strengthening DnCNN / RIDNet.

**RIDNet** (optional): ``OCR_PREPROCESS_RIDNET=1`` + ``OCR_PREPROCESS_RIDNET_WEIGHTS`` pointing
to official ``ridnet.pt`` (see saeed-anwar/RIDNet). **TorchScript** still supported via
``OCR_PREPROCESS_RIDNET_JIT``.

**Tuning (no extra denoise):** ``OCR_PREPROCESS_GPU_BORDER_LIFT``, ``OCR_PREPROCESS_GPU_ILLUM_CORNER_PULL``,
``OCR_PREPROCESS_GPU_ILLUM_SUBTLE`` (scales flatness correction), ``OCR_PREPROCESS_GPU_CLAHE_BLEND`` (CLAHE vs original),
``OCR_PREPROCESS_GPU_LOCAL_EDGE_WEIGHT`` / caps, ``OCR_PREPROCESS_GPU_MICRO_LUMA``, ``OCR_PREPROCESS_GPU_THIN_TEXT_UNSHARP``.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvf

from ocr_dncnn_torch import dncnn_apply_tensor_bchw
from ocr_ridnet_torch import ridnet_apply_tensor_bchw

_ridnet_jit: Any = None
_ridnet_jit_path: str = ""


def _truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float((os.environ.get(name) or "").strip() or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name) or "").strip() or default)
    except ValueError:
        return default


def _debug() -> bool:
    return _truthy("OCR_PREPROCESS_DEBUG", False)


def _log(msg: str) -> None:
    if _debug():
        print(f"[ocr_preprocess_gpu] {msg}", flush=True)


def _stage(prof: Dict[str, Any], name: str, ms: float, extra: Optional[Dict[str, Any]] = None) -> None:
    row: Dict[str, Any] = {"name": name, "ms": round(ms, 3)}
    if extra:
        row.update(extra)
    prof.setdefault("stages", []).append(row)
    _log(f"{name}: {ms:.2f} ms {extra or ''}")


def _effective_max_dim(h: int, w: int, max_dim: int) -> int:
    """Clamp longest-side target so resized area stays under megapixel budget."""
    if max_dim <= 0:
        return max_dim
    mp_cap = _env_float("OCR_PREPROCESS_GPU_MAX_MEGAPIXELS", 3.2)
    if mp_cap <= 0:
        return max_dim
    max_hw = max(h, w)
    area = float(h * w)
    max_area = mp_cap * 1_000_000.0
    if area <= max_area:
        return max_dim
    scale = (max_area / area) ** 0.5
    return max(400, min(max_dim, int(max_hw * scale)))


def _numpy_megapixel_cap(bgr: np.ndarray, prof: Dict[str, Any]) -> np.ndarray:
    """Optional CPU downscale before H2D to stabilize 4GB-class GPUs."""
    mp_cap = _env_float("OCR_PREPROCESS_GPU_MAX_MEGAPIXELS", 3.2)
    if mp_cap <= 0:
        return bgr
    h, w = bgr.shape[:2]
    a = float(h * w)
    if a <= mp_cap * 1_000_000.0:
        return bgr
    import cv2

    scale = (mp_cap * 1_000_000.0 / a) ** 0.5
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    t0 = time.perf_counter()
    out = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    _stage(prof, "numpy_mp_cap", (time.perf_counter() - t0) * 1000.0, {"shape": tuple(out.shape[:2])})
    prof["numpy_mp_cap_applied"] = True
    return out


def _perspective_maybe(bgr: np.ndarray, prof: Dict[str, Any]) -> np.ndarray:
    if not _truthy("OCR_PREPROCESS_PERSPECTIVE", False):
        prof["perspective"] = False
        return bgr
    t0 = time.perf_counter()
    try:
        from ocr_translator import try_perspective_dewarp_image

        warped = try_perspective_dewarp_image(bgr)
    except Exception as e:
        prof["perspective"] = False
        prof["perspective_err"] = str(e)
        _log(f"perspective import/run failed: {e}")
        return bgr
    dt = (time.perf_counter() - t0) * 1000.0
    if warped is not None and warped.size > 0:
        prof["perspective"] = True
        prof["shape_after_perspective"] = warped.shape[:2]
        _stage(prof, "perspective_cpu", dt)
        return warped
    prof["perspective"] = False
    _stage(prof, "perspective_cpu", dt, extra={"found": False})
    return bgr


def _bgr_numpy_to_tensor01(bgr: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    x = torch.from_numpy(np.ascontiguousarray(bgr)).to(device=device, dtype=torch.float32, non_blocking=True)
    x = x.permute(2, 0, 1).unsqueeze(0).mul_(1.0 / 255.0)
    return x, (time.perf_counter() - t0) * 1000.0


def _tensor01_to_bgr_u8(x: torch.Tensor) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    y = (x.squeeze(0).clamp(0.0, 1.0).permute(1, 2, 0) * 255.0 + 0.5).to(torch.uint8)
    arr = y.cpu().numpy()
    return arr, (time.perf_counter() - t0) * 1000.0


def _luminance(x: torch.Tensor) -> torch.Tensor:
    b, g, r = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.114 * b + 0.587 * g + 0.299 * r


def _resize_max_side(x: torch.Tensor, max_dim: int) -> torch.Tensor:
    if max_dim <= 0:
        return x
    _, _, h, w = x.shape
    m = max(h, w)
    if m <= max_dim:
        return x
    scale = max_dim / float(m)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    return F.interpolate(x, size=(nh, nw), mode="area")


def _laplacian_variance(gray: torch.Tensor) -> float:
    k = torch.tensor(
        [[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), k)
    return float(lap.float().var().item())


def _thumb_for_stats(x: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.shape
    side = max(h, w)
    if side > 320:
        scale = 320.0 / float(side)
        return F.interpolate(x, scale_factor=scale, mode="area")
    return x


def _compression_proxy(y: torch.Tensor) -> float:
    """Higher when 8×8 block means are uneven vs global texture (JPEG / heavy compression hint)."""
    ly = y
    mu = F.avg_pool2d(ly, 8, stride=8)
    if mu.numel() < 4:
        return 0.0
    rel = (mu - mu.mean()).abs().mean() / (ly.std() + 1e-5)
    return float(rel.clamp(0.0, 3.0).item())


def _hf_noise_proxy(y: torch.Tensor) -> float:
    sm = F.avg_pool2d(y, 3, stride=1, padding=1)
    hf = (y - sm).abs().mean()
    return float((hf / (y.std() + 1e-5)).clamp(0.0, 3.0).item())


def _thumb_stats(x: torch.Tensor) -> Dict[str, float]:
    th = _thumb_for_stats(x)
    y = _luminance(th)
    mean = float(y.mean().item())
    std = float(y.std().item())
    lap_v = _laplacian_variance(y)
    flat = y.flatten()
    q10 = float(torch.quantile(flat, 0.10).item())
    q90 = float(torch.quantile(flat, 0.90).item())
    span = max(0.0, q90 - q10)
    comp = _compression_proxy(y)
    hf = _hf_noise_proxy(y)
    return {
        "mean_l": mean,
        "std_l": std,
        "lap_var": lap_v,
        "luma_span": span,
        "compression": comp,
        "hf_ratio": hf,
    }


def _adapt_factors(stats: Dict[str, float], thin_strokes: bool) -> Dict[str, Any]:
    mean_l = stats["mean_l"]
    std_l = stats["std_l"]
    lap_v = stats["lap_var"]
    span = stats["luma_span"]
    comp = stats["compression"]
    hf_r = stats["hf_ratio"]

    dark = float(max(0.0, min(1.0, (0.44 - mean_l) / 0.44)))
    noisy = float(max(0.0, min(1.0, (std_l - 0.065) / 0.13)))
    noisy = max(noisy, 0.35 * min(1.0, hf_r / 1.1))
    flat = float(max(0.0, min(1.0, (0.058 - std_l) / 0.058)))
    blur = float(max(0.0, min(1.0, (260.0 - lap_v) / 260.0)))
    shadow_heavy = float(max(0.0, min(1.0, (span - 0.38) / 0.42)))
    compressed = float(max(0.0, min(1.0, (comp - 0.22) / 0.55)))

    base = 0.86 if thin_strokes else 1.0
    tags: List[str] = []
    if noisy > 0.55:
        tags.append("noisy")
    if blur > 0.45:
        tags.append("blurry")
    if dark > 0.45:
        tags.append("dark")
    if flat > 0.45:
        tags.append("low_contrast")
    if shadow_heavy > 0.45:
        tags.append("shadow_heavy")
    if compressed > 0.45:
        tags.append("compressed")

    return {
        "dark": dark,
        "noisy": noisy,
        "flat": flat,
        "blur": blur,
        "shadow_heavy": shadow_heavy,
        "compressed": compressed,
        "shadow_lift": base * (0.42 + 0.58 * dark) * (1.0 - 0.42 * noisy) * (1.0 - 0.12 * compressed),
        "illum_strength": base
        * (0.26 + 0.50 * dark + 0.26 * flat)
        * (1.0 - 0.40 * noisy)
        * (1.0 - 0.14 * compressed)
        * (0.94 + 0.06 * (1.0 - blur)),
        "gamma_pull": base * (0.18 * dark + 0.14 * flat - 0.07 * noisy - 0.05 * compressed),
        "contrast_pull": base * (0.42 * flat + 0.12 * dark - 0.28 * noisy - 0.12 * compressed),
        "local_contrast": base * (0.15 + 0.25 * flat - 0.36 * noisy - 0.09 * compressed) * (0.92 + 0.08 * blur),
        "unsharp_scale": base
        * (1.0 - 0.50 * noisy - 0.11 * dark + 0.19 * blur - 0.07 * compressed)
        * (1.0 + 0.11 * (1.0 if thin_strokes else 0.0))
        * (1.0 - 0.07 * noisy * (1.0 if thin_strokes else 0.0)),
        "detail_boost": base * (0.085 + 0.15 * blur - 0.20 * noisy + 0.045 * flat) * (0.97 + 0.06 * (1.0 if thin_strokes else 0.0)),
        "edge_unsharp_mix": float(
            max(
                0.41,
                min(
                    1.0,
                    (0.54 + 0.46 * blur - 0.34 * noisy) * (1.05 if thin_strokes else 1.0) + (0.038 if thin_strokes else 0.0) * (1.0 - noisy),
                ),
            )
        ),
        # Luma-only micro-contrast — slightly stronger on clean / soft-focus frames, still damped when noisy
        "micro_luma": base
        * (0.58 + 0.34 * (1.0 if thin_strokes else 0.0))
        * (0.66 + 0.34 * blur)
        * (1.0 - 0.42 * noisy)
        * (1.0 - 0.10 * compressed)
        * (0.94 + 0.06 * min(1.0, max(0.0, (lap_v - 95.0) / 320.0)) * (1.0 - 0.5 * noisy)),
        "scene_tags": tags,
    }


def _border_weight_map(x: torch.Tensor, inner: float = 0.34) -> torch.Tensor:
    """Soft map ~0 at image center → ~1 at edges (vignette / dark-border targeting)."""
    _, _, h, w = x.shape
    device, dtype = x.device, x.dtype
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w)
    d = torch.maximum(xs.abs(), ys.abs())
    m = ((d - inner).clamp(0.0, 1.0 - inner) / max(1e-6, 1.0 - inner)).pow(1.2)
    return m


def _shadow_lift_tensor(x: torch.Tensor, strength: float, border_lift: float = 0.0) -> torch.Tensor:
    if strength <= 1e-6 and border_lift <= 1e-6:
        return x
    y = _luminance(x)
    t = (1.0 - y).clamp(0.0, 1.0)
    lift = torch.ones_like(y)
    if strength > 1e-6:
        lift = 1.0 + strength * (torch.pow(t, 0.5) * 0.46)
        cap = torch.clamp((y - 0.78) / 0.22, 0.0, 1.0)
        lift = lift * (1.0 - 0.72 * cap) + 1.0 * (0.72 * cap)
        lift = torch.minimum(lift, 1.0 + strength * 0.28)
    if border_lift > 1e-6:
        bw = _border_weight_map(x)
        # Extra lift only in dark border zones — evens grainy vignettes without whitening center
        edge_dark = (t * bw).clamp(0.0, 1.0)
        lift = lift + border_lift * torch.pow(edge_dark, 0.62) * 0.34
        lift = torch.minimum(lift, 1.0 + max(strength, border_lift) * 0.34)
    return (x * lift).clamp(0.0, 1.0)


def _illumination_normalize(x: torch.Tensor, sigma: float, strength: float) -> torch.Tensor:
    if strength <= 1e-6:
        return x
    k = int(max(3, min(31, round(sigma * 6.0) | 1)))
    sig = float(max(0.8, min(sigma, 12.0)))
    blur = tvf.gaussian_blur(x, [k, k], [sig, sig])
    eps = 0.045
    illum = blur / (blur.mean(dim=(2, 3), keepdim=True) + 1e-5)
    y = (x / (illum + eps)).clamp(0.0, 1.0)
    return (x * (1.0 - strength) + y * strength).clamp(0.0, 1.0)


def _illumination_corner_aware(x: torch.Tensor, sigma: float, strength: float, corner_pull: float) -> torch.Tensor:
    """
    Blend a **wider** Gaussian estimate of illumination near frame edges (vignette / dark borders)
    with the standard estimate in the interior — flattens uneven border lighting without extra blur on x.
    """
    if strength <= 1e-6:
        return x
    cp = float(max(0.0, min(1.0, corner_pull)))
    k = int(max(3, min(31, round(sigma * 6.0) | 1)))
    sig = float(max(0.8, min(sigma, 12.0)))
    blur_f = tvf.gaussian_blur(x, [k, k], [sig, sig])
    sig_c = float(min(sig * 1.88, 14.0))
    kc = int(max(3, min(31, round(sig_c * 6.0) | 1)))
    blur_c = tvf.gaussian_blur(x, [kc, kc], [sig_c, sig_c])
    bmap = _border_weight_map(x) * cp
    blur = blur_f * (1.0 - bmap) + blur_c * bmap
    eps = 0.048
    illum = blur / (blur.mean(dim=(2, 3), keepdim=True) + 1e-5)
    y = (x / (illum + eps)).clamp(0.0, 1.0)
    return (x * (1.0 - strength) + y * strength).clamp(0.0, 1.0)


def _gamma_tensor(x: torch.Tensor, gamma: float) -> torch.Tensor:
    if abs(gamma - 1.0) < 1e-3:
        return x
    g = float(max(0.52, min(1.42, gamma)))
    return torch.pow(x.clamp(0.0, 1.0), 1.0 / g)


def _global_contrast_stretch(x: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha <= 1e-6:
        return x
    lo = x.amin(dim=(2, 3), keepdim=True)
    hi = x.amax(dim=(2, 3), keepdim=True)
    stretched = ((x - lo) / (hi - lo + 1e-4)).clamp(0.0, 1.0)
    a = float(max(0.0, min(0.38, alpha)))
    return (x * (1.0 - a) + stretched * a).clamp(0.0, 1.0)


def _local_contrast_highboost(x: torch.Tensor, amount: float, sigma: float) -> torch.Tensor:
    if amount <= 1e-6:
        return x
    k = int(max(3, min(15, round(sigma * 4.0) | 1)))
    sig = float(max(0.35, min(2.5, sigma)))
    blur = tvf.gaussian_blur(x, [k, k], [sig, sig])
    detail = x - blur
    return (x + amount * detail).clamp(0.0, 1.0)


def _local_contrast_edge_weighted(x: torch.Tensor, amount: float, sigma: float, edge_cap: float) -> torch.Tensor:
    """High-boost where edges are strong; damp on flat areas to avoid grainy halos in dark borders."""
    if amount <= 1e-6:
        return x
    k = int(max(3, min(15, round(sigma * 4.0) | 1)))
    sig = float(max(0.35, min(2.5, sigma)))
    blur = tvf.gaussian_blur(x, [k, k], [sig, sig])
    detail = x - blur
    ec = float(max(0.0, min(1.0, edge_cap)))
    if ec > 0.02:
        sm = _sobel_mag_norm(_luminance(x))
        w = 0.30 + 0.70 * torch.pow(sm, 0.88)
        w = (1.0 - ec) + ec * w
        w = w.expand(-1, 3, -1, -1)
        detail = detail * w
    return (x + amount * detail).clamp(0.0, 1.0)


def _luma_micro_contrast(x: torch.Tensor, amount: float, sigma: float) -> torch.Tensor:
    """Subtle luma high-frequency boost — crisps micro-text, preserves chroma (no color wash)."""
    if amount <= 1e-6:
        return x
    y = _luminance(x)
    k = int(max(3, min(7, round(sigma * 5.0) | 1)))
    sig = float(max(0.22, min(0.75, sigma)))
    blur_y = tvf.gaussian_blur(y, [k, k], [sig, sig])
    y2 = y + amount * (y - blur_y)
    ratio = (y2 + 1e-5) / (y + 1e-5)
    ratio = ratio.clamp(0.952, 1.058)
    return (x * ratio.expand(-1, 3, -1, -1)).clamp(0.0, 1.0)


def _sobel_mag_norm(gray: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    g = F.pad(gray, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(g, kx)
    gy = F.conv2d(g, ky)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    mag = mag / (mag.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return mag.clamp(0.0, 1.0)


def _unsharp_edge_weighted(x: torch.Tensor, amount: float, sigma: float, edge_mix: float) -> torch.Tensor:
    if amount <= 1e-6:
        return x
    k = int(max(3, min(13, round(sigma * 4.0) | 1)))
    sig = float(max(0.45, min(2.2, sigma)))
    blur = tvf.gaussian_blur(x, [k, k], [sig, sig])
    detail = x - blur
    em = float(max(0.0, min(1.0, edge_mix)))
    if em > 0.02:
        w = _sobel_mag_norm(_luminance(x))
        # Emphasize true edges over mid-gradients — reduces halos vs a linear edge map
        w = torch.pow(w.clamp(1e-6, 1.0), 1.06)
        w3 = w.expand(-1, 3, -1, -1)
        floor = 0.38 + 0.32 * (1.0 - em)
        w3 = floor + (1.0 - floor) * w3
        detail = detail * w3
    return (x + amount * detail).clamp(0.0, 1.0)


def _ridnet_jit_apply(x: torch.Tensor, prof: Dict[str, Any]) -> torch.Tensor:
    global _ridnet_jit, _ridnet_jit_path
    path = (os.environ.get("OCR_PREPROCESS_RIDNET_JIT") or "").strip()
    if not path:
        return x
    if not os.path.isfile(path):
        prof["ridnet_jit_skip"] = "missing_file"
        return x
    t0 = time.perf_counter()
    try:
        if _ridnet_jit is None or _ridnet_jit_path != path:
            _ridnet_jit = torch.jit.load(path, map_location=x.device)
            _ridnet_jit_path = path
        m = _ridnet_jit
        if hasattr(m, "eval"):
            m.eval()
        m.to(x.device)
        with torch.inference_mode():
            y = m(x)
        if y.shape != x.shape:
            prof["ridnet_jit_skip"] = f"bad_shape:{tuple(y.shape)}"
            return x
        y = y.clamp(0.0, 1.0)
        mix = float(max(0.0, min(1.0, _env_float("OCR_PREPROCESS_RIDNET_JIT_MIX", 0.62))))
        out = (x * (1.0 - mix) + y * mix).clamp(0.0, 1.0)
        _stage(prof, "ridnet_jit", (time.perf_counter() - t0) * 1000.0, {"mix": mix})
        prof["ridnet_jit_used"] = True
        return out
    except Exception as e:
        prof["ridnet_jit_skip"] = str(e)
        _log(f"RIDNet JIT failed: {e}")
        return x


def _adaptive_clahe_params(stats: Dict[str, float], af: Dict[str, Any], *, thin_strokes: bool = False) -> Tuple[float, int]:
    clip = _env_float("OCR_PREPROCESS_CLAHE_CLIP", 1.85)
    tile = int(max(4, min(16, _env_float("OCR_PREPROCESS_CLAHE_TILE", 8.0))))
    noisy = float(af.get("noisy", 0.0))
    blur = float(af.get("blur", 0.0))
    comp = float(stats.get("compression", 0.0))
    if noisy > 0.35:
        clip *= 0.74
        tile = min(16, tile + 2)
    if comp > 0.35:
        clip *= 0.88
        tile = min(16, tile + 2)
    if float(af.get("shadow_heavy", 0.0)) > 0.5:
        clip = min(clip * 1.05, 2.75)
    if float(af.get("flat", 0.0)) > 0.45:
        clip = min(clip * 1.04, 2.65)
    # Soft-focus / small text: modest clip bump when not grainy
    if blur > 0.38 and noisy < 0.42:
        clip = min(clip * (1.0 + 0.04 * min(1.0, blur)), 2.72)
    if thin_strokes and noisy < 0.40:
        tile = max(6, min(14, tile))
        clip *= 0.97
    clip = max(1.0, min(3.05, clip))
    return clip, tile


def _cpu_adaptive_clahe_maybe(
    bgr: np.ndarray,
    prof: Dict[str, Any],
    stats: Dict[str, float],
    af: Dict[str, Any],
    *,
    thin_strokes: bool = False,
) -> np.ndarray:
    use_adaptive = _truthy("OCR_PREPROCESS_GPU_ADAPTIVE_CLAHE", True)
    use_legacy = _truthy("OCR_PREPROCESS_GPU_CPU_CLAHE", False)
    if not (use_adaptive or use_legacy):
        return bgr
    if use_legacy and not use_adaptive:
        clip = max(1.0, min(3.5, _env_float("OCR_PREPROCESS_CLAHE_CLIP", 1.6)))
        tw = int(max(4, min(16, _env_float("OCR_PREPROCESS_CLAHE_TILE", 8.0))))
    else:
        clip, tw = _adaptive_clahe_params(stats, af, thin_strokes=thin_strokes)
    noisy = float(af.get("noisy", 0.0))
    if _truthy("OCR_PREPROCESS_CLAHE_AUTO", True) and noisy > 0.62 and stats.get("std_l", 0) > 0.11:
        prof["clahe_adaptive"] = False
        prof["clahe_skipped_noisy"] = True
        return bgr
    if _truthy("OCR_PREPROCESS_CLAHE_AUTO", True) and float(stats.get("compression", 0.0)) > 0.78:
        prof["clahe_adaptive"] = False
        prof["clahe_skipped_compressed"] = True
        return bgr
    import cv2

    pre = bgr
    t0 = time.perf_counter()
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tw, tw))
    l2 = clahe.apply(l_ch)
    out = cv2.cvtColor(cv2.merge([l2, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    blend = float(max(0.0, min(1.0, _env_float("OCR_PREPROCESS_GPU_CLAHE_BLEND", 0.76))))
    if blend < 1.0 - 1e-5:
        out = cv2.addWeighted(pre, 1.0 - blend, out, blend, 0)
        prof["clahe_blend"] = round(blend, 3)
    _stage(
        prof,
        "clahe_cpu_adaptive",
        (time.perf_counter() - t0) * 1000.0,
        {"clip": round(clip, 3), "tile": tw},
    )
    prof["clahe_adaptive"] = True
    return out


def _median_l_cpu_maybe(bgr: np.ndarray, prof: Dict[str, Any]) -> np.ndarray:
    if not _truthy("OCR_PREPROCESS_GPU_MEDIAN_L", False):
        return bgr
    import cv2

    t0 = time.perf_counter()
    k = max(3, min(5, _env_int("OCR_PREPROCESS_MEDIAN_L_K", 3)))
    if k % 2 == 0:
        k += 1
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l2 = cv2.medianBlur(l_ch, k)
    out = cv2.cvtColor(cv2.merge([l2, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    _stage(prof, "median_l_cpu_post", (time.perf_counter() - t0) * 1000.0)
    return out


def _cpu_luma_stretch_maybe(bgr: np.ndarray, prof: Dict[str, Any], af: Dict[str, Any]) -> np.ndarray:
    """Mild post-GPU LAB-L percentile stretch for clearer text without the grain of full CLAHE."""
    if not _truthy("OCR_PREPROCESS_LUMA_STRETCH", False):
        prof["luma_stretch"] = False
        return bgr
    import cv2

    t0 = time.perf_counter()
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    lf = l_ch.astype(np.float32)
    lo_p = max(0.5, min(8.0, _env_float("OCR_PREPROCESS_LUMA_STRETCH_LOW_PCT", 2.0)))
    hi_p = max(92.0, min(99.7, _env_float("OCR_PREPROCESS_LUMA_STRETCH_HIGH_PCT", 98.5)))
    lo, hi = np.percentile(lf, [lo_p, hi_p])
    span = float(hi - lo)
    prof["luma_stretch_span"] = round(span, 3)
    if span < 18.0:
        prof["luma_stretch"] = False
        prof["luma_stretch_skip"] = "low_span"
        return bgr
    blend = max(0.0, min(0.85, _env_float("OCR_PREPROCESS_LUMA_STRETCH_BLEND", 0.42)))
    noisy = float(af.get("noisy", 0.0))
    compressed = float(af.get("compressed", 0.0))
    blend *= 1.0 - 0.34 * noisy - 0.12 * compressed
    blend = max(0.0, min(0.85, blend))
    if blend <= 1e-5:
        prof["luma_stretch"] = False
        return bgr
    stretched = np.clip((lf - lo) * (255.0 / max(span, 1.0)), 0.0, 255.0)
    l2 = np.clip(lf * (1.0 - blend) + stretched * blend, 0.0, 255.0).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([l2, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    _stage(
        prof,
        "luma_stretch_cpu_post",
        (time.perf_counter() - t0) * 1000.0,
        {"blend": round(blend, 3), "low_pct": lo_p, "high_pct": hi_p},
    )
    prof["luma_stretch"] = True
    return out


def try_preprocess_gpu(
    bgr: np.ndarray,
    max_dim: int = 1200,
    *,
    thin_strokes: bool = False,
    force_cpu: bool = False,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    prof: Dict[str, Any] = {
        "backend": "gpu_pipeline",
        "thin_strokes": thin_strokes,
        "force_cpu": force_cpu,
        "stages": [],
    }
    if bgr is None or bgr.size == 0:
        prof["error"] = "empty"
        return None, prof
    if force_cpu or _truthy("OCR_PREPROCESS_FORCE_CPU", False):
        prof["gpu_skip"] = "force_cpu"
        return None, prof
    if not torch.cuda.is_available():
        prof["gpu_skip"] = "no_cuda"
        return None, prof

    t_all = time.perf_counter()
    img = np.ascontiguousarray(bgr)
    h0, w0 = img.shape[:2]
    prof["shape_in"] = (h0, w0)

    try:
        img = _perspective_maybe(img, prof)
        img = _numpy_megapixel_cap(img, prof)
        hi, wi = img.shape[:2]
        max_use = _effective_max_dim(hi, wi, int(max_dim))
        prof["max_dim_effective"] = max_use

        device = torch.device("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x, h2d_ms = _bgr_numpy_to_tensor01(img, device)
        torch.cuda.synchronize()
        _stage(prof, "h2d", h2d_ms)

        t0 = time.perf_counter()
        x = _resize_max_side(x, max_use)
        torch.cuda.synchronize()
        _stage(prof, "resize_gpu", (time.perf_counter() - t0) * 1000.0, {"shape": tuple(x.shape)})

        t0 = time.perf_counter()
        stats = _thumb_stats(x)
        torch.cuda.synchronize()
        _stage(prof, "stats_thumb", (time.perf_counter() - t0) * 1000.0, {k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()})
        prof.update({f"stat_{k}": v for k, v in stats.items()})

        af = _adapt_factors(stats, thin_strokes)
        prof["adapt"] = {k: v for k, v in af.items() if k != "scene_tags"}
        prof["scene_tags"] = af.get("scene_tags", [])

        bright_env = _env_float("OCR_PREPROCESS_BRIGHTEN", 1.0)
        prof["brighten_env"] = bright_env
        shadow_strength = 0.0
        if bright_env > 1.0:
            shadow_strength = (bright_env - 1.0) * 0.58 * float(af["shadow_lift"])
        border_lift = float(
            max(
                0.0,
                min(
                    0.36,
                    _env_float("OCR_PREPROCESS_GPU_BORDER_LIFT", 0.24)
                    * (0.44 * float(af["shadow_heavy"]) + 0.30 * float(af["dark"]))
                    * (1.0 - 0.44 * float(af["noisy"])),
                ),
            )
        )
        t0 = time.perf_counter()
        x = _shadow_lift_tensor(x, shadow_strength, border_lift)
        torch.cuda.synchronize()
        _stage(
            prof,
            "shadow_lift_gpu",
            (time.perf_counter() - t0) * 1000.0,
            {"strength": round(shadow_strength, 4), "border_lift": round(border_lift, 4)},
        )

        # Mild gamma before denoise so DnCNN sees more balanced range
        gamma_pre = float(max(0.88, min(1.12, 1.0 + float(af["gamma_pull"]) * 0.22)))
        t0 = time.perf_counter()
        x = _gamma_tensor(x, gamma_pre)
        torch.cuda.synchronize()
        _stage(prof, "gamma_pre_denoise", (time.perf_counter() - t0) * 1000.0, {"gamma": round(gamma_pre, 4)})

        t0 = time.perf_counter()
        x, dinfo = dncnn_apply_tensor_bchw(x, force_cpu=False)
        torch.cuda.synchronize()
        for dk, dv in dinfo.items():
            prof[str(dk)] = dv
        _stage(prof, "dncnn_tensor", (time.perf_counter() - t0) * 1000.0, {k: dinfo.get(k) for k in ("dnn_tensor_used", "dnn_skip", "dnn_fp16")})

        illum_sigma = float(max(2.5, min(11.0, _env_float("OCR_PREPROCESS_GPU_ILLUM_SIGMA", 6.2))))
        illum_s = float(max(0.0, min(0.88, _env_float("OCR_PREPROCESS_GPU_ILLUM_STRENGTH", 0.46) * float(af["illum_strength"]))))
        illum_s *= float(max(0.72, min(1.0, _env_float("OCR_PREPROCESS_GPU_ILLUM_SUBTLE", 0.90))))
        corner_pull = float(
            max(
                0.0,
                min(
                    1.0,
                    _env_float("OCR_PREPROCESS_GPU_ILLUM_CORNER_PULL", 0.62)
                    * (0.32 + 0.68 * float(af["shadow_heavy"]) + 0.22 * float(af["dark"]))
                    * (1.0 - 0.28 * float(af["noisy"])),
                ),
            )
        )
        t0 = time.perf_counter()
        x = _illumination_corner_aware(x, illum_sigma, illum_s, corner_pull)
        torch.cuda.synchronize()
        _stage(
            prof,
            "illumination_corner_gpu",
            (time.perf_counter() - t0) * 1000.0,
            {"sigma": illum_sigma, "strength": round(illum_s, 4), "corner_pull": round(corner_pull, 3)},
        )

        gamma0 = float(max(0.78, min(1.18, _env_float("OCR_PREPROCESS_GPU_GAMMA", 1.0) + float(af["gamma_pull"]) * 0.32)))
        t0 = time.perf_counter()
        x = _gamma_tensor(x, gamma0)
        torch.cuda.synchronize()
        _stage(prof, "gamma_gpu", (time.perf_counter() - t0) * 1000.0, {"gamma": round(gamma0, 4)})

        gcon = float(max(0.0, min(0.26, _env_float("OCR_PREPROCESS_GPU_GLOBAL_CONTRAST", 0.1)))) * (0.62 + float(af["contrast_pull"]))
        t0 = time.perf_counter()
        x = _global_contrast_stretch(x, gcon)
        torch.cuda.synchronize()
        _stage(prof, "global_contrast_gpu", (time.perf_counter() - t0) * 1000.0, {"alpha": round(gcon, 4)})

        loc_amp = float(max(0.0, min(0.52, _env_float("OCR_PREPROCESS_GPU_LOCAL_CONTRAST", 0.17)))) * (0.48 + float(af["local_contrast"]))
        loc_sig = float(max(0.55, min(1.65, _env_float("OCR_PREPROCESS_GPU_LOCAL_SIGMA", 1.0))))
        loc_ec = float(max(0.0, min(1.0, _env_float("OCR_PREPROCESS_GPU_LOCAL_EDGE_CAP", 0.78))))
        t0 = time.perf_counter()
        if _truthy("OCR_PREPROCESS_GPU_LOCAL_EDGE_WEIGHT", True):
            x = _local_contrast_edge_weighted(x, loc_amp, loc_sig, loc_ec)
        else:
            x = _local_contrast_highboost(x, loc_amp, loc_sig)
        torch.cuda.synchronize()
        _stage(prof, "local_contrast_gpu", (time.perf_counter() - t0) * 1000.0, {"amount": round(loc_amp, 4), "edge_weighted": _truthy("OCR_PREPROCESS_GPU_LOCAL_EDGE_WEIGHT", True)})

        t0 = time.perf_counter()
        x, rinfo = ridnet_apply_tensor_bchw(x, force_cpu=False)
        torch.cuda.synchronize()
        for rk, rv in rinfo.items():
            prof[str(rk)] = rv
        _stage(prof, "ridnet_native", (time.perf_counter() - t0) * 1000.0, {k: rinfo.get(k) for k in ("ridnet_used", "ridnet_skip", "ridnet_fp16")})

        x = _ridnet_jit_apply(x, prof)

        dboost = float(max(0.0, min(0.24, _env_float("OCR_PREPROCESS_GPU_DETAIL_BOOST", 0.08)))) * (0.65 + float(af["detail_boost"]))
        dec = float(max(0.0, min(1.0, _env_float("OCR_PREPROCESS_GPU_DETAIL_EDGE_CAP", 0.56))))
        t0 = time.perf_counter()
        if _truthy("OCR_PREPROCESS_GPU_LOCAL_EDGE_WEIGHT", True):
            x = _local_contrast_edge_weighted(x, dboost, 0.58, dec)
        else:
            x = _local_contrast_highboost(x, dboost, 0.58)
        torch.cuda.synchronize()
        _stage(prof, "detail_boost_gpu", (time.perf_counter() - t0) * 1000.0, {"amount": round(dboost, 4), "edge_cap": dec})

        mic = float(
            max(
                0.0,
                min(0.12, _env_float("OCR_PREPROCESS_GPU_MICRO_LUMA", 0.052) * float(af["micro_luma"])),
            )
        )
        mic_sig = float(max(0.18, min(0.72, _env_float("OCR_PREPROCESS_GPU_MICRO_LUMA_SIGMA", 0.36))))
        if mic > 1e-5:
            t0 = time.perf_counter()
            x = _luma_micro_contrast(x, mic, mic_sig)
            torch.cuda.synchronize()
            _stage(prof, "micro_luma_gpu", (time.perf_counter() - t0) * 1000.0, {"amount": round(mic, 5), "sigma": mic_sig})

        unsharp_base = float(max(0.0, min(1.05, _env_float("OCR_PREPROCESS_UNSHARP", 0.34))))
        unsharp_sig = float(max(0.5, min(2.1, _env_float("OCR_PREPROCESS_UNSHARP_SIGMA", 1.0))))
        scale = float(max(0.0, min(1.0, float(af["unsharp_scale"]))))
        if _truthy("OCR_PREPROCESS_ADAPTIVE_UNSHARP", True):
            std_l = stats["std_l"]
            if std_l < 0.068:
                adapt_scale = 1.0
            elif std_l < 0.092:
                adapt_scale = 0.62
            elif std_l < 0.12:
                adapt_scale = 0.36
            elif std_l < 0.152:
                adapt_scale = 0.16
            elif std_l < 0.19:
                adapt_scale = 0.055
            else:
                adapt_scale = 0.0
            if _truthy("OCR_PREPROCESS_EDGE_RESTORE", True):
                adapt_scale = max(adapt_scale, _env_float("OCR_PREPROCESS_EDGE_RESTORE_FLOOR", 0.16))
            prof["unsharp_adaptive_scale"] = adapt_scale
            scale *= adapt_scale
        thin_bonus = 0.0
        if thin_strokes:
            thin_bonus = float(max(0.0, min(0.28, _env_float("OCR_PREPROCESS_GPU_THIN_TEXT_UNSHARP", 0.16)))) * (1.0 - 0.4 * float(af["noisy"]))
        amt = float(min(0.52, unsharp_base * scale * (1.0 + thin_bonus)))
        sig_scale = float(max(0.82, min(1.0, _env_float("OCR_PREPROCESS_GPU_UNSHARP_THIN_SIGMA_SCALE", 0.94))))
        if thin_strokes and float(af["noisy"]) < 0.11:
            unsharp_sig = float(max(0.45, unsharp_sig * sig_scale))
        edge_mix = float(min(1.0, float(af["edge_unsharp_mix"]) * (1.05 if thin_strokes else 1.0)))
        t0 = time.perf_counter()
        x = _unsharp_edge_weighted(x, amt, unsharp_sig, edge_mix)
        torch.cuda.synchronize()
        _stage(prof, "unsharp_edge_gpu", (time.perf_counter() - t0) * 1000.0, {"amount": round(amt, 4), "edge_mix": round(edge_mix, 3)})

        micro_auto = float(af["blur"]) > 0.55 and float(af["noisy"]) < 0.42
        micro_thin = bool(thin_strokes) and float(af["noisy"]) < 0.38
        if _truthy("OCR_PREPROCESS_GPU_MICRO_UNSHARP", False) or micro_auto or micro_thin:
            micro = float(max(0.0, min(0.35, _env_float("OCR_PREPROCESS_GPU_MICRO_UNSHARP", 0.14)))) * (1.0 - 0.5 * float(af["noisy"]))
            if micro_thin and micro < 0.035:
                micro = 0.035 * (1.0 - 0.45 * float(af["noisy"]))
            if micro > 0.02:
                t0 = time.perf_counter()
                x = _unsharp_edge_weighted(x, micro, 0.72, edge_mix * 0.85)
                torch.cuda.synchronize()
                _stage(prof, "micro_unsharp_gpu", (time.perf_counter() - t0) * 1000.0, {"amount": round(micro, 4)})

        if _truthy("OCR_PREPROCESS_GPU_EMPTY_CACHE", False):
            torch.cuda.empty_cache()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_u8, d2h_ms = _tensor01_to_bgr_u8(x)
        del x
        torch.cuda.synchronize()
        _stage(prof, "d2h", d2h_ms)

        out_bgr = out_u8
        out_bgr = _cpu_luma_stretch_maybe(out_bgr, prof, af)
        out_bgr = _cpu_adaptive_clahe_maybe(out_bgr, prof, stats, af, thin_strokes=thin_strokes)
        out_bgr = _median_l_cpu_maybe(out_bgr, prof)

        prof["shape_out"] = out_bgr.shape[:2]
        prof["total_s"] = time.perf_counter() - t_all

        tags: List[str] = ["gpu", "resize", "shadow", "gamma_pre"]
        if prof.get("dnn_tensor_used"):
            tags.append("dncnn")
        tags.extend(["illum_corner", "gamma", "gcontrast", "local"])
        if prof.get("ridnet_used"):
            tags.append("ridnet")
        if prof.get("ridnet_jit_used"):
            tags.append("ridnet_jit")
        tags.append("detail")
        if any(s.get("name") == "micro_luma_gpu" for s in prof.get("stages", [])):
            tags.append("micro_luma")
        tags.append("unsharp")
        if any(s.get("name") == "micro_unsharp_gpu" for s in prof.get("stages", [])):
            tags.append("micro_unsharp")
        if prof.get("perspective"):
            tags.insert(0, "perspective")
        if prof.get("numpy_mp_cap_applied"):
            tags.insert(1 if prof.get("perspective") else 0, "precap")
        if any(s.get("name") == "clahe_cpu_adaptive" for s in prof.get("stages", [])):
            tags.append("clahe_adaptive")
        if any(s.get("name") == "median_l_cpu_post" for s in prof.get("stages", [])):
            tags.append("median_l")
        if any(s.get("name") == "luma_stretch_cpu_post" for s in prof.get("stages", [])):
            tags.append("luma_stretch")
        prof["backend"] = "+".join(tags)

        return out_bgr, prof
    except Exception as e:
        prof["gpu_error"] = repr(e)
        _log(f"gpu pipeline error: {e}")
        return None, prof


def gpu_pipeline_available() -> bool:
    try:
        return bool(torch.cuda.is_available()) and not _truthy("OCR_PREPROCESS_FORCE_CPU", False)
    except Exception:
        return False
