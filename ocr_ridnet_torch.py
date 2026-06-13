"""
RIDNet (ICCV 2019) real-image denoiser — optional GPU stage for OCR preprocessing.

Architecture matches `saeed-anwar/RIDNet` (``TestCode/code/model``). Weights are **not**
bundled: download ``ridnet.pt`` from the official repo (Google Drive / IceDrive links in
their README) and set ``OCR_PREPROCESS_RIDNET_WEIGHTS`` to the file path.

Input / output tensors are ``(N,3,H,W)`` float in ``[0,1]`` (internally scaled by ``rgb_range``).
"""

from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_model_lock = threading.Lock()
_model: Any = None
_model_device: str = ""
_failed_key: Optional[Tuple[str, str]] = None


def _truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _model_cache_key(device_str: str) -> str:
    d = torch.device(device_str if device_str else "cpu")
    return "cuda" if d.type == "cuda" else "cpu"


class MeanShift(nn.Conv2d):
    """Channel-wise normalize / denormalize (from RIDNet ``common.py``)."""

    def __init__(self, rgb_range: float, rgb_mean, rgb_std, sign: int = -1) -> None:
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = float(sign) * float(rgb_range) * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        for p in self.parameters():
            p.requires_grad = False


def _init_weights(_modules) -> None:
    pass


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ksize: int = 3, stride: int = 1, pad: int = 1) -> None:
        super(BasicBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True),
        )
        _init_weights(self.modules())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class BasicBlockSig(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ksize: int = 3, stride: int = 1, pad: int = 1) -> None:
        super(BasicBlockSig, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.Sigmoid(),
        )
        _init_weights(self.modules())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Merge_Run_dual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ksize: int = 3, stride: int = 1, pad: int = 1) -> None:
        super(Merge_Run_dual, self).__init__()
        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, ksize, stride, 2, 2),
            nn.ReLU(inplace=True),
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, 3, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, ksize, stride, 4, 4),
            nn.ReLU(inplace=True),
        )
        self.body3 = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True),
        )
        _init_weights(self.modules())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.body1(x)
        out2 = self.body2(x)
        c = torch.cat([out1, out2], dim=1)
        c_out = self.body3(c)
        return c_out + x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        _init_weights(self.modules())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        return F.relu(out + x)


class EResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, group: int = 1) -> None:
        super(EResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )
        _init_weights(self.modules())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        return F.relu(out + x)


class CALayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c1 = BasicBlock(channel, channel // reduction, 1, 1, 0)
        self.c2 = BasicBlockSig(channel // reduction, channel, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y1 = self.c1(y)
        y2 = self.c2(y1)
        return x * y2


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, group: int = 1) -> None:
        super(Block, self).__init__()
        self.r1 = Merge_Run_dual(in_channels, out_channels)
        self.r2 = ResidualBlock(in_channels, out_channels)
        self.r3 = EResidualBlock(in_channels, out_channels)
        self.ca = CALayer(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r1 = self.r1(x)
        r2 = self.r2(r1)
        r3 = self.r3(r2)
        return self.ca(r3)


class RIDNET(nn.Module):
    def __init__(self, n_feats: int = 64, reduction: int = 16, rgb_range: float = 255.0) -> None:
        super(RIDNET, self).__init__()
        kernel_size = 3
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std, sign=-1)
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, sign=1)
        self.head = BasicBlock(3, n_feats, kernel_size, 1, 1)
        self.b1 = Block(n_feats, n_feats)
        self.b2 = Block(n_feats, n_feats)
        self.b3 = Block(n_feats, n_feats)
        self.b4 = Block(n_feats, n_feats)
        self.tail = nn.Conv2d(n_feats, 3, kernel_size, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.sub_mean(x)
        h = self.head(s)
        b1 = self.b1(h)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b_out = self.b4(b3)
        res = self.tail(b_out)
        out = self.add_mean(res)
        return out + x


def _default_args():
    return SimpleNamespace(n_feats=64, reduction=16, rgb_range=255.0)


def _normalize_keys(sd: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        for key in ("model", "state_dict", "params"):
            if key in blob and isinstance(blob[key], dict):
                return _normalize_keys(blob[key])
        if any(isinstance(v, torch.Tensor) for v in blob.values()):
            return _normalize_keys(blob)
    raise ValueError("unrecognized checkpoint format for RIDNet")


def _get_model(device: str):
    global _model, _model_device, _failed_key
    with _model_lock:
        cache_key = _model_cache_key(device)
        if _model is not None and _model_device == cache_key:
            return _model, None
        wpath = (os.environ.get("OCR_PREPROCESS_RIDNET_WEIGHTS") or "").strip()
        if not wpath or not os.path.isfile(wpath):
            return None, "no_weights"
        key = (cache_key, wpath)
        if _failed_key == key:
            return None, "ridnet_init_failed_cached"
        try:
            net = RIDNET()
            state = _load_state_dict(wpath)
            try:
                net.load_state_dict(state, strict=True)
            except Exception:
                net.load_state_dict(state, strict=False)
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


def ridnet_apply_tensor_bchw(
    x01: torch.Tensor,
    *,
    force_cpu: bool,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Apply RIDNet to ``x01`` (B,3,H,W) in ``[0,1]``. Disabled unless ``OCR_PREPROCESS_RIDNET=1``
    and ``OCR_PREPROCESS_RIDNET_WEIGHTS`` points to a valid ``ridnet.pt``-style checkpoint.
    """
    info: Dict[str, Any] = {"ridnet_used": False}
    if not _truthy("OCR_PREPROCESS_RIDNET", False):
        info["ridnet_skip"] = "disabled"
        return x01, info
    if x01.dim() != 4 or x01.shape[1] != 3:
        info["ridnet_skip"] = "bad_shape"
        return x01, info
    if force_cpu or _truthy("OCR_PREPROCESS_FORCE_CPU", False):
        info["ridnet_skip"] = "force_cpu"
        return x01, info

    cache_key = _model_cache_key(str(x01.device))
    net, err = _get_model(cache_key)
    if net is None:
        info["ridnet_skip"] = err or "init_failed"
        return x01, info
    if next(net.parameters()).device != x01.device:
        net.to(x01.device)

    rgb_range = float(_default_args().rgb_range)
    t0 = time.perf_counter()
    try:
        x255 = (x01 * rgb_range).clamp(0.0, rgb_range)
        use_fp16 = _truthy("OCR_PREPROCESS_GPU_FP16", False) and x01.device.type == "cuda"
        with torch.inference_mode():
            if use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    y255 = net(x255)
            else:
                y255 = net(x255)
        y01 = (y255 / rgb_range).clamp(0.0, 1.0)
        mix = float(max(0.0, min(1.0, float((os.environ.get("OCR_PREPROCESS_RIDNET_MIX") or "0.72").strip() or 0.72))))
        out = (x01 * (1.0 - mix) + y01 * mix).clamp(0.0, 1.0)
        info["ridnet_used"] = True
        info["ridnet_device"] = str(x01.device)
        info["ridnet_s"] = time.perf_counter() - t0
        info["ridnet_fp16"] = use_fp16
        return out, info
    except Exception as e:
        info["ridnet_skip"] = f"infer:{e}"
        return x01, info


def ridnet_available() -> bool:
    if not _truthy("OCR_PREPROCESS_RIDNET", False):
        return False
    p = (os.environ.get("OCR_PREPROCESS_RIDNET_WEIGHTS") or "").strip()
    return bool(p and os.path.isfile(p))
