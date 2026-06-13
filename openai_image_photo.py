"""
Translated menu photo via OpenAI Images API (e.g. gpt-image-1.5) using images.edit.

Requires: pip install openai
Set OPENAI_API_KEY. Optional OPENAI_IMAGE_MODEL (default gpt-image-1.5).

Latency is dominated by the remote ``images.edit`` call (often 30–90s at high quality).

Payload optimization (defaults tuned for smaller uploads):
  OPENAI_IMAGE_EDIT_MAX_SIDE — default 1024; clamp 768–1024 unless
    OPENAI_IMAGE_EDIT_LARGE_INPUT=1 (then clamp 768–2048).
  OPENAI_IMAGE_EDIT_JPEG_QUALITY — default 78; clamp 70–95.
  OPENAI_IMAGE_EDIT_ALWAYS_JPEG — default 1: always RGB + JPEG temp for the API
    (drops alpha, strips heavy PNG metadata from uploads).

Caching (duplicate image + prompt + model settings):
  OPENAI_IMAGE_CACHE — default 1; set 0 to disable.
  OPENAI_IMAGE_CACHE_DIR — optional; else temp dir ``openai_image_edit_cache``.

HTTP: OPENAI_HTTP_TIMEOUT (default 180), OPENAI_HTTP_MAX_RETRIES (default 2).

✨ Enhance-only overrides (see app.py): OPENAI_IMAGE_ENHANCE_FAST=1 or
OPENAI_IMAGE_ENHANCE_QUALITY / _SIZE / _INPUT_FIDELITY / _MODEL.

Prompt: ``compact_prompt=True`` (✨ path) or OPENAI_IMAGE_PROMPT_MINIMAL=1 globally.
Images.edit does not stream partial image bytes; progressive UI is via app-side polling.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import tempfile
import time
from typing import Any, List, Optional, Tuple

from PIL import Image


def _get_client(api_key: Optional[str]):
    key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY or paste it in the app.")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Install the OpenAI SDK: pip install openai") from e
    try:
        timeout = float(os.environ.get("OPENAI_HTTP_TIMEOUT", "180"))
    except ValueError:
        timeout = 180.0
    try:
        max_retries = int(os.environ.get("OPENAI_HTTP_MAX_RETRIES", "2"))
    except ValueError:
        max_retries = 2
    max_retries = max(0, min(10, max_retries))
    return OpenAI(api_key=key, timeout=timeout, max_retries=max_retries)


def default_image_model() -> str:
    return (os.environ.get("OPENAI_IMAGE_MODEL") or "gpt-image-1.5").strip()


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _cache_enabled() -> bool:
    return _env_truthy("OPENAI_IMAGE_CACHE", True)


def _cache_dir() -> str:
    d = (os.environ.get("OPENAI_IMAGE_CACHE_DIR") or "").strip()
    if not d:
        d = os.path.join(tempfile.gettempdir(), "openai_image_edit_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_paths(key_hex: str) -> Tuple[str, str]:
    base = os.path.join(_cache_dir(), key_hex[:2])
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{key_hex}.bin"), os.path.join(base, f"{key_hex}.meta.txt")


def _cache_lookup(key_hex: str) -> Optional[bytes]:
    bin_path, _ = _cache_paths(key_hex)
    if os.path.isfile(bin_path):
        try:
            with open(bin_path, "rb") as f:
                return f.read()
        except OSError:
            return None
    return None


def _cache_store(key_hex: str, data: bytes, meta: str) -> None:
    bin_path, meta_path = _cache_paths(key_hex)
    try:
        with open(bin_path, "wb") as f:
            f.write(data)
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(meta[:4000])
    except OSError:
        pass


def _compute_cache_key(
    image_bytes: bytes,
    prompt: str,
    *,
    model: str,
    quality: str,
    size: str,
    input_fidelity: str,
    dedupe_retry: bool,
) -> str:
    h = hashlib.sha256()
    h.update(image_bytes)
    h.update(b"\0")
    h.update(prompt.encode("utf-8", errors="replace"))
    h.update(b"\0")
    h.update(
        f"{model}|{quality}|{size}|{input_fidelity}|dedupe={int(dedupe_retry)}".encode(
            "ascii", errors="replace"
        )
    )
    return h.hexdigest()


def _strip_ocr_uncertain(s: str) -> str:
    """Match ocr_translator: do not pass UI warning into image-edit prompts."""
    return re.sub(r"\(OCR uncertain\)\s*", "", (s or ""), flags=re.IGNORECASE)


def build_edit_prompt(
    pairs: List[Tuple[str, str]],
    target_lang_display: str,
    max_chars: int = 14000,
    *,
    compact: bool = False,
) -> str:
    pair_count = sum(1 for o, t in pairs if (o or "").strip() or (t or "").strip())
    if compact:
        header = (
            f"In-place edit: replace visible text using only the list ({pair_count} line(s)). "
            f"Keep scene and lighting; wording in {target_lang_display}. "
            "One translation per line; no duplicates or extra labels; erase originals then write translations; "
            "blend naturally (no solid gray boxes).\n\n"
            "Pairs:\n"
        )
    else:
        header = (
            "Edit this photograph in-place. Keep the same scene, lighting, colors, perspective, and background. "
            f"Replace visible text with natural {target_lang_display} wording in the same layout and reading order. "
            "Use ONLY the listed replacements below. Do NOT add new words, headings, labels, punctuation blocks, "
            "numbers, or decorative text that is not in the provided translations. "
            "Render each translated line exactly once (no duplicated lines, no echoed/shadow copies). "
            "Completely remove source text strokes in edited areas before writing the translated text. "
            f"The final rendered translated text count must equal {pair_count} line(s) from the list below. "
            "Never render the same translated line twice. "
            "Preserve notebook paper texture and ruled lines; do NOT paint opaque gray/white boxes or patches. "
            "Avoid large filled rectangles; blend edits naturally with surrounding paper. "
            "If a source line is unclear, leave that area blank rather than inventing text. "
            "Remove or cover the original text and render the translations clearly where the old text appeared.\n\n"
            "Text replacements (original → translation):\n"
        )
    lines: List[str] = [header]
    n = 0
    for orig, trans in pairs:
        o = (orig or "").strip().replace("\n", " ")
        t = _strip_ocr_uncertain((trans or "").strip().replace("\n", " "))
        if not o and not t:
            continue
        n += 1
        lines.append(f"{n}. «{o}» → «{t}»")
    body = "\n".join(lines)
    if len(body) > max_chars:
        body = body[: max_chars - 80] + "\n…(list truncated; apply the same style to any remaining text.)"
    return body


def _read_edit_max_side() -> int:
    try:
        max_side = int(os.environ.get("OPENAI_IMAGE_EDIT_MAX_SIDE", "1024"))
    except ValueError:
        max_side = 1024
    if max_side <= 0:
        return 0
    if _env_truthy("OPENAI_IMAGE_EDIT_LARGE_INPUT", False):
        return max(768, min(2048, max_side))
    return max(768, min(1024, max_side))


def _read_jpeg_quality() -> int:
    try:
        jpeg_quality = int(os.environ.get("OPENAI_IMAGE_EDIT_JPEG_QUALITY", "78"))
    except ValueError:
        jpeg_quality = 78
    return max(55, min(95, jpeg_quality))


def _normalize_image_path_for_edit(image_path: str) -> Tuple[str, bool]:
    """
    Return a path OpenAI can consume; second value is True if a temp file was created.

    By default always emits an optimized RGB JPEG (alpha stripped, no PNG metadata bloat)
    so upload/base64 payload stays small.
    """
    max_side = _read_edit_max_side()
    jpeg_quality = _read_jpeg_quality()
    always_jpeg = _env_truthy("OPENAI_IMAGE_EDIT_ALWAYS_JPEG", True)

    try:
        img = Image.open(image_path)
        img.load()
        img = img.convert("RGB")
        w, h = img.size
        largest = max(w, h)
        if max_side > 0 and largest > max_side:
            s = max_side / float(largest)
            nw = max(1, int(round(w * s)))
            nh = max(1, int(round(h * s)))
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    except Exception:
        return image_path, False

    if not always_jpeg:
        ext = os.path.splitext(image_path)[1].lower()
        try:
            needs_resize = max_side > 0 and max(img.size) > max_side
        except Exception:
            needs_resize = True
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif") and not needs_resize:
            try:
                with Image.open(image_path) as probe:
                    pw, ph = probe.size
                    if max_side <= 0 or max(pw, ph) <= max_side:
                        return image_path, False
            except Exception:
                pass

    try:
        fd, tmp = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        img.save(
            tmp,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
            progressive=True,
            subsampling=2,
        )
        return tmp, True
    except Exception:
        return image_path, False


def edit_photo_with_translations(
    image_path: str,
    pairs: List[Tuple[str, str]],
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    quality: str = "high",
    size: str = "auto",
    input_fidelity: str = "high",
    target_lang_display: str = "target language",
    dedupe_retry: bool = False,
    perf_stats: Optional[dict[str, Any]] = None,
    compact_prompt: bool = False,
) -> bytes:
    """
    Call OpenAI images.edit; return raw image bytes (PNG or as returned).

    If ``perf_stats`` is a dict, fills timing keys (seconds) for UI logs.
    """
    t_edit_phase = time.perf_counter()
    t_n = time.perf_counter()
    path, is_temp = _normalize_image_path_for_edit(image_path)
    if perf_stats is not None:
        perf_stats["normalize_image_s"] = time.perf_counter() - t_n

    try:
        use_compact = compact_prompt or _env_truthy("OPENAI_IMAGE_PROMPT_MINIMAL", False)

        t_p = time.perf_counter()
        prompt = build_edit_prompt(
            pairs,
            target_lang_display,
            compact=use_compact,
        )
        if perf_stats is not None:
            perf_stats["prompt_build_s"] = time.perf_counter() - t_p
            perf_stats["prompt_chars"] = len(prompt)

        try:
            with open(path, "rb") as nf:
                norm_bytes = nf.read()
        except OSError:
            norm_bytes = b""

        m = (model or default_image_model()).strip()
        cache_key_hex = _compute_cache_key(
            norm_bytes,
            prompt,
            model=m,
            quality=quality,
            size=size,
            input_fidelity=input_fidelity,
            dedupe_retry=dedupe_retry,
        )
        if _cache_enabled() and norm_bytes:
            cached = _cache_lookup(cache_key_hex)
            if cached is not None:
                if perf_stats is not None:
                    perf_stats["cache_hit"] = 1
                    perf_stats["openai_images_edit_s"] = 0.0
                return cached
        if perf_stats is not None:
            perf_stats["cache_hit"] = 0

        client = _get_client(api_key)

        def _edit_once(img_path: str, ptxt: str) -> bytes:
            t_api = time.perf_counter()
            with open(img_path, "rb") as img_f:
                resp = client.images.edit(
                    model=m,
                    image=img_f,
                    prompt=ptxt,
                    quality=quality,  # type: ignore[arg-type]
                    size=size,  # type: ignore[arg-type]
                    input_fidelity=input_fidelity,  # type: ignore[arg-type]
                )
            if not getattr(resp, "data", None):
                raise RuntimeError("OpenAI Images returned no data.")
            item = resp.data[0]
            b64 = getattr(item, "b64_json", None)
            if not b64 and hasattr(item, "model_dump"):
                b64 = item.model_dump().get("b64_json")
            if not b64:
                raise RuntimeError("OpenAI Images response missing image data (b64_json).")
            out_b = base64.b64decode(b64)
            if perf_stats is not None:
                perf_stats["openai_images_edit_s"] = perf_stats.get("openai_images_edit_s", 0.0) + (
                    time.perf_counter() - t_api
                )
            return out_b

        out = _edit_once(path, prompt)
        if dedupe_retry:
            retry_prompt = (
                build_edit_prompt(
                    pairs,
                    target_lang_display,
                    compact=use_compact,
                )
                + "\n\nSecond-pass: remove duplicate translated lines only; same wording.\n"
            )
            fd, tmp_retry = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            try:
                try:
                    ri = Image.open(io.BytesIO(out)).convert("RGB")
                    ms = _read_edit_max_side()
                    qjpeg = _read_jpeg_quality()
                    if max(ri.size) > ms:
                        rw, rh = ri.size
                        s = ms / float(max(rw, rh))
                        ri = ri.resize(
                            (max(1, int(round(rw * s))), max(1, int(round(rh * s)))),
                            Image.Resampling.LANCZOS,
                        )
                    ri.save(
                        tmp_retry,
                        format="JPEG",
                        quality=qjpeg,
                        optimize=True,
                        progressive=True,
                        subsampling=2,
                    )
                except Exception:
                    with open(tmp_retry, "wb") as f:
                        f.write(out)
                out = _edit_once(tmp_retry, retry_prompt)
            finally:
                try:
                    os.unlink(tmp_retry)
                except OSError:
                    pass

        if _cache_enabled() and norm_bytes:
            _cache_store(cache_key_hex, out, f"model={m} quality={quality}")

        return out
    finally:
        if perf_stats is not None:
            perf_stats["edit_photo_total_s"] = time.perf_counter() - t_edit_phase
        if is_temp and path and os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass


def image_bytes_to_rgb_numpy(image_bytes: bytes):
    """RGB uint8 array for Streamlit / download."""
    import numpy as np

    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(im)
