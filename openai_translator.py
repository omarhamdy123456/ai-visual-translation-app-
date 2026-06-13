"""
Translation via OpenAI Chat Completions API (GPT models).
Used by the app only when translating OCR text for **generated translated photos** (image mode).

Requires: pip install openai
Set OPENAI_API_KEY or pass api_key; optional OPENAI_TRANSLATION_MODEL (default gpt-4o-mini).
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from translator import (
    LANG_TO_NLLB,
    SUPPORTED_LANGUAGES,
    _normalize_nllb_source_text,
    _scrub_mixed_latin_noise_in_arabic_output,
)


def _lang_label(code: str) -> str:
    c = (code or "en").lower().strip()
    return SUPPORTED_LANGUAGES.get(c, c)


def _get_client(api_key: Optional[str]):
    key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY or paste it in the app.")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Install the OpenAI SDK: pip install openai") from e
    return OpenAI(api_key=key)


def _default_model() -> str:
    return (os.environ.get("OPENAI_TRANSLATION_MODEL") or "gpt-4o-mini").strip()


def _batch_system_prompt(src_label: str, tgt_label: str) -> str:
    return (
        "You are a professional translator.\n"
        f"Translate each item from {src_label} to {tgt_label}.\n"
        "Rules:\n"
        "- Return strict JSON object: {\"translations\": [\"...\", ...]}.\n"
        "- Keep array length exactly equal to input items length.\n"
        "- Preserve order by index.\n"
        "- Do not add explanations, markdown, or extra keys.\n"
    )


def translate_text_openai(
    text: str,
    target_language: str,
    source_language: Optional[str],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    fast: bool = False,
) -> dict:
    """Same shape as TextTranslator.translate_text."""
    if not source_language:
        source_language = "en"
    if source_language.lower() == target_language.lower():
        return {
            "original_text": text,
            "translated_text": text,
            "source_language": source_language,
            "target_language": target_language,
            "confidence": 1.0,
        }
    raw = _normalize_nllb_source_text((text or "").strip(), LANG_TO_NLLB.get(source_language))
    if not raw:
        return {
            "original_text": text,
            "translated_text": "",
            "source_language": source_language,
            "target_language": target_language,
            "confidence": None,
        }
    src_label = _lang_label(source_language)
    tgt_label = _lang_label(target_language)
    system = (
        f"You are a professional translator. Translate from {src_label} to {tgt_label}. "
        "Preserve meaning and tone. Output ONLY the translated text, with no quotes, labels, or explanation."
    )
    try:
        client = _get_client(api_key)
        mid = (model or _default_model()).strip()
        if fast:
            fast_model = (os.environ.get("OPENAI_TRANSLATION_FAST_MODEL") or "").strip()
            if fast_model:
                mid = fast_model
        max_out = min(4096, max(256, len(raw) * 3))
        if fast:
            try:
                fast_cap = int(os.environ.get("OPENAI_TRANSLATION_FAST_MAX_TOKENS", "900"))
            except ValueError:
                fast_cap = 900
            fast_cap = max(256, min(2048, fast_cap))
            max_out = min(max_out, fast_cap)
        resp = client.chat.completions.create(
            model=mid,
            temperature=0.2,
            max_tokens=max_out,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": raw},
            ],
        )
        out = (resp.choices[0].message.content or "").strip()
        tgt_nllb = LANG_TO_NLLB.get((target_language or "").lower())
        if tgt_nllb == "arb_Arab":
            out = _scrub_mixed_latin_noise_in_arabic_output(out)
        return {
            "original_text": text,
            "translated_text": out,
            "source_language": source_language,
            "target_language": target_language,
            "confidence": None,
        }
    except Exception as e:
        return {
            "original_text": text,
            "translated_text": f"Translation error: {str(e)}",
            "source_language": source_language,
            "target_language": target_language,
            "confidence": None,
        }


def translate_batch_openai(
    texts: List[str],
    target_language: str,
    source_language: Optional[str],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    fast: bool = False,
) -> List[dict]:
    """
    Single-request batch translation to avoid per-item sequential API latency.
    Falls back to single-item translation on parse/shape mismatch.
    """
    if not source_language:
        source_language = "en"
    normalized: List[str] = [
        _normalize_nllb_source_text((t or "").strip(), LANG_TO_NLLB.get(source_language))
        for t in (texts or [])
    ]
    if not normalized:
        return []
    if source_language.lower() == target_language.lower():
        return [
            {
                "original_text": texts[i],
                "translated_text": texts[i],
                "source_language": source_language,
                "target_language": target_language,
                "confidence": 1.0,
            }
            for i in range(len(texts))
        ]

    src_label = _lang_label(source_language)
    tgt_label = _lang_label(target_language)
    payload = {
        "items": [{"i": i, "text": normalized[i]} for i in range(len(normalized))]
    }
    try:
        client = _get_client(api_key)
        mid = (model or _default_model()).strip()
        if fast:
            fast_model = (os.environ.get("OPENAI_TRANSLATION_FAST_MODEL") or "").strip()
            if fast_model:
                mid = fast_model
        # Estimate output budget from source length, but keep a strict ceiling.
        est_tokens = max(512, min(8192, int(sum(max(1, len(t)) for t in normalized) * 2.4)))
        if fast:
            try:
                fast_cap = int(os.environ.get("OPENAI_TRANSLATION_FAST_MAX_TOKENS", "900"))
            except ValueError:
                fast_cap = 900
            fast_cap = max(256, min(4096, fast_cap))
            est_tokens = min(est_tokens, fast_cap)
        resp = client.chat.completions.create(
            model=mid,
            temperature=0.1,
            max_tokens=est_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _batch_system_prompt(src_label, tgt_label)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        obj = json.loads(raw) if raw else {}
        arr = obj.get("translations")
        if not isinstance(arr, list) or len(arr) != len(texts):
            raise ValueError("Batch response size mismatch")
        out_rows: List[dict] = []
        tgt_nllb = LANG_TO_NLLB.get((target_language or "").lower())
        for i, tr in enumerate(arr):
            txt = str(tr or "").strip()
            if tgt_nllb == "arb_Arab":
                txt = _scrub_mixed_latin_noise_in_arabic_output(txt)
            out_rows.append(
                {
                    "original_text": texts[i],
                    "translated_text": txt,
                    "source_language": source_language,
                    "target_language": target_language,
                    "confidence": None,
                }
            )
        return out_rows
    except Exception:
        # Robust fallback preserves functionality if strict JSON is not respected.
        return [
            translate_text_openai(
                t,
                target_language=target_language,
                source_language=source_language,
                api_key=api_key,
                model=model,
                fast=fast,
            )
            for t in texts
        ]
