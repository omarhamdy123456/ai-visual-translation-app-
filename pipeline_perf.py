"""
Optional full-pipeline performance logging (wall-clock via time.perf_counter).

Enable:
  set PIPELINE_PERF_FULL=1

Logs appear on stderr/stdout as ``[pipeline_perf] stage=...``.

This module does not alter OCR or translation behavior — only observability.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional


def pipeline_perf_full_enabled() -> bool:
    return os.environ.get("PIPELINE_PERF_FULL", "").strip().lower() in ("1", "true", "yes")


def pipeline_perf_log(stage: str, seconds: float, **extra: Any) -> None:
    if not pipeline_perf_full_enabled():
        return
    bits = [f"stage={stage}", f"wall_s={seconds:.4f}"]
    for k, v in sorted(extra.items()):
        bits.append(f"{k}={v}")
    print("[pipeline_perf] " + " ".join(bits), flush=True)


class PerfSpan:
    """Simple span timer for nested or sequential stages."""

    __slots__ = ("name", "t0", "extra")

    def __init__(self, name: str, **extra: Any):
        self.name = name
        self.t0 = time.perf_counter()
        self.extra = extra

    def end(self) -> float:
        dt = time.perf_counter() - self.t0
        pipeline_perf_log(self.name, dt, **self.extra)
        return dt


class PerfSession:
    """Collect ordered spans; dump a compact summary line."""

    def __init__(self, label: str):
        self.label = label
        self.enabled = pipeline_perf_full_enabled()
        self.rows: List[str] = []
        self._last = time.perf_counter()

    def mark(self, stage: str, **extra: Any) -> None:
        if not self.enabled:
            self._last = time.perf_counter()
            return
        now = time.perf_counter()
        dt = now - self._last
        self.rows.append(f"{stage}:{dt:.4f}s")
        pipeline_perf_log(f"{self.label}.{stage}", dt, **extra)
        self._last = now

    def summary(self) -> str:
        if not self.enabled or not self.rows:
            return ""
        return f"[pipeline_perf] {self.label} :: " + " | ".join(self.rows)


def trocr_cuda_sync_before_generate() -> bool:
    """Extra synchronize before generate — hurts latency; use only for profiling."""
    return os.environ.get("OCR_TROCR_SYNC_BEFORE_GENERATE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def trocr_cleanup_after_batch() -> bool:
    return os.environ.get("OCR_TROCR_CUDA_CLEANUP_AFTER_BATCH", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def photo_render_max_side_default() -> int:
    """Max longest edge for local translated-photo workspace (0 = full resolution)."""
    raw = (os.environ.get("PHOTO_RENDER_WORK_MAX_SIDE") or "2048").strip()
    try:
        v = int(raw)
    except ValueError:
        v = 2048
    return max(0, min(8192, v))
