"""Optional OpenAI Vision / Chat helpers for handwriting workflow."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from typing import Optional, Tuple

_IMAGE_DATA_URL_CACHE: dict[str, str] = {}


def _get_client(api_key: Optional[str]):
    key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY or paste it in the app.")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Install the OpenAI SDK: pip install openai") from e
    try:
        timeout_sec = float(os.environ.get("OPENAI_VISION_TIMEOUT_SEC", "75"))
    except ValueError:
        timeout_sec = 75.0
    timeout_sec = max(20.0, min(300.0, timeout_sec))
    return OpenAI(api_key=key, timeout=timeout_sec)


def _image_file_to_data_url(path: str) -> str:
    """
    Encode image as a compact data URL for faster Vision requests.
    Defaults are tuned for latency while keeping handwriting legibility.
    """
    try:
        from io import BytesIO
        from PIL import Image
    except ImportError:
        # Fallback to raw bytes if Pillow is unavailable.
        ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
        if ext == "jpg":
            ext = "jpeg"
        with open(path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("ascii")
        return f"data:image/{ext};base64,{b64}"

    try:
        max_side = int(os.environ.get("OPENAI_VISION_IMAGE_MAX_SIDE", "1280"))
    except ValueError:
        max_side = 1600
    max_side = max(768, min(4096, max_side))
    try:
        jpeg_quality = int(os.environ.get("OPENAI_VISION_IMAGE_QUALITY", "65"))
    except ValueError:
        jpeg_quality = 72
    jpeg_quality = max(45, min(95, jpeg_quality))

    cache_key = None
    try:
        st = os.stat(path)
        cache_key = f"{path}|{st.st_mtime_ns}|{st.st_size}|{max_side}|{jpeg_quality}"
        ckey = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()
        hit = _IMAGE_DATA_URL_CACHE.get(ckey)
        if hit:
            return hit
    except OSError:
        ckey = None

    with Image.open(path) as im:
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        w, h = im.size
        largest = max(w, h)
        if largest > max_side:
            scale = max_side / float(largest)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        if im.mode == "L":
            im = im.convert("RGB")
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    out = f"data:image/jpeg;base64,{b64}"
    if ckey:
        if len(_IMAGE_DATA_URL_CACHE) > 24:
            _IMAGE_DATA_URL_CACHE.clear()
        _IMAGE_DATA_URL_CACHE[ckey] = out
    return out


def transcribe_handwriting_from_image(
    image_path: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Read handwriting from an image using OpenAI Vision only (no local OCR).
    Returns (transcript, iso639_1_source_guess) e.g. ("Hello\\nworld", "en").
    """
    mid = (model or os.environ.get("OPENAI_VISION_MODEL") or "gpt-4o-mini").strip()
    client = _get_client(api_key)
    url = _image_file_to_data_url(image_path)
    user = (
        "Read all handwritten text in this image.\n"
        "Reply in exactly this format (no markdown fences, no extra commentary):\n"
        "First line: two-letter ISO 639-1 code for the dominant language of the handwriting (e.g. en, ar, es).\n"
        "Second line: a single line containing only ---\n"
        "Remaining lines: the full transcript, preserving line breaks and reading order (top to bottom).\n"
        "If there is no handwriting, reply with:\\nen\\n---\\n\\n"
    )
    try:
        max_out = int(os.environ.get("OPENAI_VISION_MAX_OUTPUT_TOKENS", "900"))
    except ValueError:
        max_out = 1600
    max_out = max(256, min(4096, max_out))
    detail = (os.environ.get("OPENAI_VISION_IMAGE_DETAIL", "low") or "low").strip().lower()
    if detail not in ("low", "auto", "high"):
        detail = "auto"

    resp = client.chat.completions.create(
        model=mid,
        temperature=0.05,
        max_tokens=max_out,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": url, "detail": detail}},
                ],
            }
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    lines = raw.splitlines()
    if len(lines) >= 3 and lines[1].strip() == "---":
        code = (lines[0] or "en").strip().lower()[:8]
        body = "\n".join(lines[2:]).strip()
        return body, code
    # Fallback: whole body is transcript, guess English
    return raw, "en"


def transcribe_and_translate_handwriting_from_image(
    image_path: str,
    target_language: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    prefer_arabic_quality: bool = False,
) -> Tuple[str, str, str]:
    """
    Single Vision call that returns transcript + translation.
    Returns (transcript, translated_text, iso639_1_source_guess).
    """
    mid = (model or os.environ.get("OPENAI_VISION_MODEL") or "gpt-4o-mini").strip()
    client = _get_client(api_key)
    url = _image_file_to_data_url(image_path)
    tgt = (target_language or "en").strip()
    user = (
        "Read all handwritten text in this image and translate it.\n"
        "Return strict JSON with keys:\n"
        "- source_iso: two-letter source language code (e.g. en, ar, es)\n"
        "- transcript: full source transcript preserving line breaks and reading order\n"
        f"- translation: full translation in {tgt}, preserving paragraph structure\n"
        "Do not include any additional keys or commentary."
    )
    try:
        default_tokens = "1700" if prefer_arabic_quality else "1300"
        max_out = int(os.environ.get("OPENAI_VISION_MAX_OUTPUT_TOKENS", default_tokens))
    except ValueError:
        max_out = 1700 if prefer_arabic_quality else 1300
    max_out = max(384, min(4096, max_out))
    detail_default = "auto" if prefer_arabic_quality else "low"
    detail = (os.environ.get("OPENAI_VISION_IMAGE_DETAIL", detail_default) or detail_default).strip().lower()
    if detail not in ("low", "auto", "high"):
        detail = "auto"

    resp = client.chat.completions.create(
        model=mid,
        temperature=0.05,
        max_tokens=max_out,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": url, "detail": detail}},
                ],
            }
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    # Primary parse: strict JSON.
    try:
        obj = json.loads(raw) if raw else {}
        code = str(obj.get("source_iso") or "en").strip().lower()[:8] or "en"
        body = str(obj.get("transcript") or "").strip()
        tr = str(obj.get("translation") or "").strip()
        return body, tr, code
    except Exception:
        pass
    # Secondary parse for any legacy/plain responses.
    if "===TRANSLATION===" in raw and "---" in raw:
        try:
            top, tr = raw.split("===TRANSLATION===", 1)
            lines = top.splitlines()
            code = "en"
            body = top.strip()
            if len(lines) >= 2 and lines[1].strip() == "---":
                code = (lines[0] or "en").strip().lower()[:8]
                body = "\n".join(lines[2:]).strip()
            return body, tr.strip(), code
        except Exception:
            pass
    # Never do a second Vision API call here; return best effort from same response.
    return raw, "", "en"


def refine_transcript_with_vision(
    image_path: str,
    draft_text: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Use a vision model to correct / normalize handwritten transcript from OCR draft.
    Returns plain text only (no markdown fences).
    """
    mid = (model or os.environ.get("OPENAI_VISION_MODEL") or "gpt-4o-mini").strip()
    client = _get_client(api_key)
    url = _image_file_to_data_url(image_path)
    user = (
        "You see a photo of handwritten notes. Below is a noisy OCR draft.\n"
        "Task: output the **best possible transcript** of the handwriting in the image. "
        "Fix obvious OCR errors, preserve line breaks where helpful, and do not add commentary.\n\n"
        f"OCR draft:\n{draft_text or '(empty)'}"
    )
    detail = (os.environ.get("OPENAI_VISION_IMAGE_DETAIL", "auto") or "auto").strip().lower()
    if detail not in ("low", "auto", "high"):
        detail = "auto"
    resp = client.chat.completions.create(
        model=mid,
        temperature=0.1,
        max_tokens=min(4096, max(512, len(draft_text or "") * 2)),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": url, "detail": detail}},
                ],
            }
        ],
    )
    out = (resp.choices[0].message.content or "").strip()
    return out


def polish_bilingual_for_export(
    original_text: str,
    translated_text: str,
    target_lang_name: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Optional Chat-only pass: lightly clean spacing/punctuation for PDF export.
    Returns (original_polished, translated_polished).
    """
    mid = (model or os.environ.get("OPENAI_TRANSLATION_MODEL") or "gpt-4o-mini").strip()
    client = _get_client(api_key)
    system = (
        "You clean bilingual document text for export. "
        "Return TWO sections exactly in this format (no other text):\n"
        "===ORIGINAL===\n...\n===TRANSLATED===\n...\n"
        f"The translated side should be natural {target_lang_name}. "
        "Do not invent new facts; only fix obvious typos/spacing from OCR."
    )
    user = (
        f"===ORIGINAL===\n{original_text or ''}\n\n===TRANSLATED===\n{translated_text or ''}"
    )
    resp = client.chat.completions.create(
        model=mid,
        temperature=0.15,
        max_tokens=min(8192, max(1024, len(user))),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    if "===ORIGINAL===" in raw and "===TRANSLATED===" in raw:
        try:
            a, b = raw.split("===TRANSLATED===", 1)
            o = a.split("===ORIGINAL===", 1)[1].strip()
            t = b.strip()
            return o, t
        except Exception:
            pass
    return (original_text or "").strip(), (translated_text or "").strip()
