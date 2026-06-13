"""
Lightweight color **DnCNN** denoiser (PyTorch) for OCR preprocessing — small and fast vs large restorers.

Weights are downloaded once from Hugging Face ``deepinv/dncnn`` (``dncnn_sigma2_color.pth``, ~2.6 MB, BSD-3-Clause).
Architecture matches `deepinv.models.DnCNN` (depth=20, no batch norm) — residual form ``out_conv(features) + x``.

Not NAFNet; optional only via ``OCR_PREPROCESS_DNN=1``.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_model_lock = threading.Lock()
_model: Any = None
_model_device: str = ""
_failed_key: Optional[Tuple[str, str]] = None


def _truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


class _DnCNN(nn.Module):
    """DnCNN-3c depth=20, nf=64 (same layout as deepinv DnCNN, no BN)."""

    def __init__(self, depth: int = 20, nf: int = 64) -> None:
        super().__init__()
        if depth < 3:
            raise ValueError("depth must be >= 3")
        self.depth = depth
        self.in_conv = nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_list = nn.ModuleList(
            [nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True) for _ in range(depth - 2)]
        )
        self.out_conv = nn.Conv2d(nf, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.nl_list = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(depth - 1)])

    def forward(self, x):
        x1 = self.in_conv(x)
        x1 = self.nl_list[0](x1)
        for i in range(self.depth - 2):
            x_l = self.conv_list[i](x1)
            x1 = self.nl_list[i + 1](x_l)
        return self.out_conv(x1) + x


def _pick_device(force_cpu: bool) -> str:
    if force_cpu or _truthy("OCR_PREPROCESS_FORCE_CPU", False):
        return "cpu"
    raw = (os.environ.get("OCR_PREPROCESS_DNN_DEVICE") or "").strip().lower()
    if raw in ("cpu", "cuda"):
        if raw == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_weights_path() -> str:
    custom = (os.environ.get("OCR_PREPROCESS_DNN_WEIGHTS") or "").strip()
    if custom and os.path.isfile(custom):
        return custom
    repo = (os.environ.get("OCR_PREPROCESS_DNN_REPO") or "deepinv/dncnn").strip()
    fn = (os.environ.get("OCR_PREPROCESS_DNN_FILE") or "dncnn_sigma2_color.pth").strip()
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=fn)


def _model_cache_key(device_str: str) -> str:
    d = torch.device(device_str if device_str else "cpu")
    return "cuda" if d.type == "cuda" else "cpu"


def _get_model(device: str):
    global _model, _model_device, _failed_key
    with _model_lock:
        cache_key = _model_cache_key(device)
        if _model is not None and _model_device == cache_key:
            return _model, None
        try:
            path = _load_weights_path()
        except Exception as e:
            return None, str(e)
        key = (cache_key, path)
        if _failed_key == key:
            return None, "dnn_init_failed_cached"
        try:
            net = _DnCNN(depth=20, nf=64)
            state = torch.load(path, map_location="cpu", weights_only=True)
            net.load_state_dict(state, strict=True)
            net.eval()
            dev = torch.device("cuda" if cache_key == "cuda" else "cpu")
            net.to(dev)
            _model = net
            _model_device = cache_key
            _failed_key = None
            return _model, None
        except Exception as e:
            _failed_key = key
            _model = None
            return None, str(e)


def dncnn_apply_tensor_bchw(x: torch.Tensor, *, force_cpu: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Run DnCNN on ``x`` (B,3,H,W float in ``[0,1]``) on the **same device** as ``x`` (typically CUDA).

    Returns ``(y, info)`` with ``y`` same shape/device as ``x``. If disabled or init fails, returns ``(x, info)``.
    """
    info: Dict[str, Any] = {"dnn_tensor_used": False}
    if not _truthy("OCR_PREPROCESS_DNN", False):
        return x, info
    if x.dim() != 4 or x.shape[1] != 3:
        info["dnn_skip"] = "bad_shape"
        return x, info
    if force_cpu or _truthy("OCR_PREPROCESS_FORCE_CPU", False):
        info["dnn_skip"] = "force_cpu_tensor_path"
        return x, info

    cache_key = _model_cache_key(str(x.device))
    net, err = _get_model(cache_key)
    if net is None:
        info["dnn_skip"] = err or "init_failed"
        return x, info
    if next(net.parameters()).device != x.device:
        net.to(x.device)
    t0 = time.perf_counter()
    try:
        use_fp16 = _truthy("OCR_PREPROCESS_GPU_FP16", False) and x.device.type == "cuda"
        with torch.inference_mode():
            if use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    y = net(x)
            else:
                y = net(x)
        y = y.clamp(0.0, 1.0)
        info["dnn_tensor_used"] = True
        info["dnn_device"] = str(x.device)
        info["dnn_tensor_s"] = time.perf_counter() - t0
        info["dnn_fp16"] = use_fp16
        return y, info
    except Exception as e:
        info["dnn_skip"] = f"infer:{e}"
        return x, info


def dncnn_denoise_bgr_maybe(
    bgr: np.ndarray,
    *,
    force_cpu: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    If ``OCR_PREPROCESS_DNN=1``, run DnCNN color denoise and return BGR uint8.

    On failure (no torch, bad weights), returns the input image and ``dnn_skip`` in the info dict.
    """
    info: Dict[str, Any] = {"dnn_used": False}
    if not _truthy("OCR_PREPROCESS_DNN", False):
        return bgr, info

    if bgr is None or bgr.size == 0:
        info["dnn_skip"] = "empty"
        return bgr, info

    device_s = _pick_device(force_cpu)
    net, err = _get_model(device_s)
    if net is None:
        info["dnn_skip"] = err or "init_failed"
        return bgr, info

    t0 = time.perf_counter()
    h0, w0 = bgr.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(bgr)).to(device=torch.device(device_s), dtype=torch.float32)
    x = x.permute(2, 0, 1).unsqueeze(0) / 255.0
    try:
        with torch.inference_mode():
            y = net(x)
        y = (y.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()
        out = y.astype(np.uint8)
        info["dnn_used"] = True
        info["dnn_device"] = device_s
        info["dnn_s"] = time.perf_counter() - t0
        info["dnn_shape"] = (h0, w0)
        return out, info
    except Exception as e:
        info["dnn_skip"] = f"infer:{e}"
        return bgr, info


def dncnn_available() -> bool:
    if not _truthy("OCR_PREPROCESS_DNN", False):
        return False
    return True
