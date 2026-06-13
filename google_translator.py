"""
Google Cloud Translate adapter.
Uses google-cloud-translate (v2 client) with ADC credentials.
"""

from __future__ import annotations

import html
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from typing import Any, Callable, Dict, List, Optional, Union

try:
    # Preferred for google-cloud-translate v2-style client.
    from google.cloud import translate_v2 as translate
except ImportError:
    # Fallback for environments exposing v2 client as `google.cloud.translate`.
    from google.cloud import translate as translate

from translator import (
    LANG_TO_NLLB,
    _normalize_nllb_source_text,
    _scrub_mixed_latin_noise_in_arabic_output,
)

try:
    from google.api_core.client_options import ClientOptions
except ImportError:
    ClientOptions = None  # type: ignore[misc, assignment]

# Cloud Translation v2 `translate`: max **128** strings per request (not 256 — larger batches fail).
# Character limits depend on Basic vs Advanced; oversized requests return 400 and trigger slow bisection.
_GOOGLE_TRANSLATION_CACHE: dict[tuple[str, str, str, str], dict] = {}
_GOOGLE_TRANSLATION_CACHE_MAX = 4000
_GOOGLE_CLIENT_LOCK = threading.Lock()
_GOOGLE_CLIENT_CACHE: dict[tuple[str, str], object] = {}
_PERF_LOCK = threading.Lock()


def _google_http_timeout_sec(override: Optional[float] = None) -> Optional[float]:
    """
    Per-request deadline for Google translate() HTTP calls (runs in a worker thread so we can enforce).
    Env: GOOGLE_TRANSLATE_HTTP_TIMEOUT_SEC — unset or <=0 disables timeout.
    Example: 5 for aggressive fallback to local models on slow networks.
    """
    if override is not None:
        return override if override > 0 else None
    raw = (os.environ.get("GOOGLE_TRANSLATE_HTTP_TIMEOUT_SEC") or "").strip()
    if not raw:
        return 45.0
    try:
        v = float(raw)
    except ValueError:
        return 45.0
    return None if v <= 0 else v


def _translate_http(
    client: Any,
    payload: Union[str, List[str]],
    kw: dict,
    timeout_sec: Optional[float],
) -> Any:
    """Blocking Cloud Translate v2 translate(); optional wall-clock deadline."""

    def _call() -> Any:
        return client.translate(payload, **kw)

    if timeout_sec is None or timeout_sec <= 0:
        return _call()
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call)
        try:
            return fut.result(timeout=timeout_sec)
        except FutureTimeoutError as e:
            raise TimeoutError("Google Cloud Translate request exceeded deadline") from e


def _perf_add(perf_stats: Optional[dict], key: str, delta: float) -> None:
    if perf_stats is None or delta <= 0:
        return
    with _PERF_LOCK:
        perf_stats[key] = perf_stats.get(key, 0.0) + delta


def _perf_inc(perf_stats: Optional[dict], key: str, n: int = 1) -> None:
    if perf_stats is None:
        return
    with _PERF_LOCK:
        perf_stats[key] = perf_stats.get(key, 0) + n


def _apply_google_translate_fast_env() -> None:
    """
    Preset for lowest latency: chunk packing + high worker counts.
    Does **not** force list-batch vs joined — use GOOGLE_TRANSLATE_STRATEGY=auto (default) so the
    app picks fewer HTTP round trips (often marker-joined for many short lines).
    Set GOOGLE_TRANSLATE_FAST=1 in .env. Does not override explicit env vars you already set.
    """
    if (os.environ.get("GOOGLE_TRANSLATE_FAST", "") or "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    os.environ.setdefault("GOOGLE_TRANSLATE_BATCH_MAX_CHARS", "99990")
    # Very high fan-out (e.g. 64) often increases tail latency from API throttling.
    # Keep "fast" aggressive but within a range that is usually stable.
    os.environ.setdefault("GOOGLE_TRANSLATE_BATCH_WORKERS", "16")
    os.environ.setdefault("GOOGLE_TRANSLATE_BATCH_WORKERS_MAX", "24")
    os.environ.setdefault("GOOGLE_JOINED_PARALLEL_WORKERS", "16")
    os.environ.setdefault("GOOGLE_JOINED_PARALLEL_WORKERS_MAX", "24")


_apply_google_translate_fast_env()


def _joined_segment_marker(segment_index: int) -> str:
    """Compact per-line marker (~10 chars vs ~17 legacy) so more text fits per Google request."""
    return f"[#T{segment_index:05d}#]"


def _google_batch_chunk_limits() -> tuple[int, int]:
    """
    Returns (max_strings, max_chars) for one synchronous batch call.
    Env:
      GOOGLE_TRANSLATE_BATCH_MAX_ITEMS — default 128, hard-capped at 128 (API limit).
      GOOGLE_TRANSLATE_BATCH_MAX_CHARS — default 98000 (~Basic 100KB text); set ~28000 if you use Translation Advanced only.
    """
    try:
        max_items = int(os.environ.get("GOOGLE_TRANSLATE_BATCH_MAX_ITEMS", "128"))
    except ValueError:
        max_items = 128
    max_items = max(1, min(128, max_items))
    try:
        max_chars = int(os.environ.get("GOOGLE_TRANSLATE_BATCH_MAX_CHARS", "99900"))
    except ValueError:
        max_chars = 99900
    max_chars = max(4096, max_chars)
    return max_items, max_chars


def _count_google_list_batch_chunks(texts: List[str], source_language: Optional[str]) -> int:
    """HTTP call count if using translate_batch_google (same packing as translate_batch_google)."""
    if not texts:
        return 0
    src = (source_language or "").strip().lower()
    normalized = [
        _normalize_nllb_source_text((t or "").strip(), LANG_TO_NLLB.get(src or "en"))
        for t in texts
    ]
    max_items, max_chars_req = _google_batch_chunk_limits()
    chunks = 0
    cur_n = 0
    cur_chars = 0
    for t in normalized:
        tlen = max(1, len(t))
        if cur_n > 0 and (
            cur_n >= max_items or (cur_chars + tlen) > max_chars_req
        ):
            chunks += 1
            cur_n = 0
            cur_chars = 0
        cur_n += 1
        cur_chars += tlen
    if cur_n:
        chunks += 1
    return chunks


def google_translation_should_use_joined(texts: List[str], source_language: Optional[str] = None) -> bool:
    """
    Choose marker-joined vs list-batch to minimize HTTP round trips.

    Many short OCR lines force list-batch into many chunks (max 128 strings each) — multiple
    waves of parallel requests. Marker-joined packs ~100 KB per request and often uses fewer calls.

    Env:
      GOOGLE_TRANSLATE_STRATEGY — auto (default), joined, or batch.
      GOOGLE_TRANSLATE_USE_JOINED — force off when 0 / false / no.
      GOOGLE_TRANSLATE_PREFER_BATCH — force list-batch when 1 (often slower for huge line counts).
      GOOGLE_TRANSLATE_JOINED_MIN_SEGMENTS / GOOGLE_TRANSLATE_JOINED_MAX_SEGMENTS
    """
    if not texts or len(texts) < 2:
        return False
    if (os.environ.get("GOOGLE_TRANSLATE_USE_JOINED", "1") or "").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return False
    if (os.environ.get("GOOGLE_TRANSLATE_PREFER_BATCH", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return False
    try:
        min_seg = int(os.environ.get("GOOGLE_TRANSLATE_JOINED_MIN_SEGMENTS", "4"))
    except ValueError:
        min_seg = 4
    try:
        max_seg = int(os.environ.get("GOOGLE_TRANSLATE_JOINED_MAX_SEGMENTS", "250000"))
    except ValueError:
        max_seg = 250000
    min_seg = max(2, min_seg)
    max_seg = max(min_seg, max_seg)
    n = len(texts)
    if n < min_seg or n > max_seg:
        return False

    strategy = (os.environ.get("GOOGLE_TRANSLATE_STRATEGY", "auto") or "").strip().lower()
    if strategy in ("joined", "marker", "markers"):
        return True
    if strategy in ("batch", "list", "list_batch"):
        return False

    batch_chunks = _count_google_list_batch_chunks(texts, source_language)
    joined_shards = len(_joined_parallel_ranges(texts))
    if joined_shards < batch_chunks:
        return True
    if joined_shards > batch_chunks:
        return False
    return False


def _batch_workers_config() -> int:
    """
    Effective parallel chunk workers for translate_batch_google.
    Env: GOOGLE_TRANSLATE_BATCH_WORKERS (default 16), GOOGLE_TRANSLATE_BATCH_WORKERS_MAX (default 24, max 64).
    """
    try:
        w = int(os.environ.get("GOOGLE_TRANSLATE_BATCH_WORKERS", "16"))
    except ValueError:
        w = 32
    try:
        cap = int(os.environ.get("GOOGLE_TRANSLATE_BATCH_WORKERS_MAX", "24"))
    except ValueError:
        cap = 56
    cap = max(1, min(64, cap))
    w = max(1, min(cap, w))
    return w


def _joined_parallel_workers(num_shards: int) -> int:
    """Workers for parallel marker-shard translates (often I/O bound)."""
    try:
        w = int(os.environ.get("GOOGLE_JOINED_PARALLEL_WORKERS", "16"))
    except ValueError:
        w = 32
    try:
        cap = int(os.environ.get("GOOGLE_JOINED_PARALLEL_WORKERS_MAX", "24"))
    except ValueError:
        cap = 56
    cap = max(1, min(64, cap))
    w = max(1, min(cap, w))
    return max(1, min(w, num_shards))


def _google_client_cache_key(project_id: Optional[str]) -> tuple[str, str]:
    ep = (os.environ.get("GOOGLE_CLOUD_TRANSLATE_ENDPOINT") or "").strip().lower()
    return (project_id or "", ep)


def _is_retryable_google_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    retry_tokens = (
        "429",
        "rate limit",
        "resource exhausted",
        "too many requests",
        "deadline exceeded",
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "service unavailable",
        "503",
    )
    return any(tok in msg for tok in retry_tokens)


def warm_google_translate_client(project_id: Optional[str] = None) -> None:
    """Resolve credentials and open HTTP pool before the first large translate (saves cold-start RTT)."""
    try:
        _google_client(project_id=project_id)
    except Exception:
        pass


def _google_client(project_id: Optional[str] = None):
    """Return a cached Client — avoids TLS + credential churn on every segment/shard."""
    ck = _google_client_cache_key(project_id)
    with _GOOGLE_CLIENT_LOCK:
        cached = _GOOGLE_CLIENT_CACHE.get(ck)
        if cached is not None:
            return cached
        opts = None
        ep = (os.environ.get("GOOGLE_CLOUD_TRANSLATE_ENDPOINT") or "").strip()
        if ep and ClientOptions is not None:
            opts = ClientOptions(api_endpoint=ep)
        kwargs: dict = {}
        if project_id:
            kwargs["project"] = project_id
        if opts is not None:
            kwargs["client_options"] = opts
        client = translate.Client(**kwargs) if kwargs else translate.Client()
        _GOOGLE_CLIENT_CACHE[ck] = client
        return client


def translate_text_google(
    text: str,
    target_language: str,
    source_language: Optional[str],
    project_id: Optional[str] = None,
    fast: bool = False,
    *,
    timeout_sec: Optional[float] = None,
    fallback_single_fn: Optional[Callable[[], dict]] = None,
    perf_stats: Optional[dict] = None,
) -> dict:
    """Same shape as TextTranslator.translate_text."""
    del fast
    src = (source_language or "").strip().lower()
    if src and src == target_language.lower():
        return {
            "original_text": text,
            "translated_text": text,
            "source_language": src,
            "target_language": target_language,
            "confidence": 1.0,
        }
    raw = _normalize_nllb_source_text((text or "").strip(), LANG_TO_NLLB.get(src or "en"))
    if not raw:
        return {
            "original_text": text,
            "translated_text": "",
            "source_language": source_language,
            "target_language": target_language,
            "confidence": None,
        }
    cache_key = (raw, target_language.lower(), src or "auto", project_id or "")
    hit = _GOOGLE_TRANSLATION_CACHE.get(cache_key)
    if hit is not None:
        return dict(hit)
    ts = _google_http_timeout_sec(timeout_sec)
    try:
        client = _google_client(project_id=project_id)
        req = {
            "target_language": target_language,
            "format_": "text",
        }
        if src:
            req["source_language"] = src
        t0 = time.perf_counter()
        try:
            result = _translate_http(client, raw, req, ts)
        except TimeoutError:
            if fallback_single_fn is not None:
                fb_t0 = time.perf_counter()
                out_fb = fallback_single_fn()
                _perf_add(perf_stats, "fallback_wall_seconds", time.perf_counter() - fb_t0)
                _perf_add(perf_stats, "google_wall_seconds", time.perf_counter() - t0)
                return out_fb
            return {
                "original_text": text,
                "translated_text": "Translation error: Google Cloud Translate request exceeded deadline",
                "source_language": src or "auto",
                "target_language": target_language,
                "confidence": None,
            }
        _perf_add(perf_stats, "google_wall_seconds", time.perf_counter() - t0)
        _perf_inc(perf_stats, "http_calls")
        translated_text = html.unescape((result or {}).get("translatedText", "")).strip()
        tgt_nllb = LANG_TO_NLLB.get((target_language or "").lower())
        if tgt_nllb == "arb_Arab":
            translated_text = _scrub_mixed_latin_noise_in_arabic_output(translated_text)
        out = {
            "original_text": text,
            "translated_text": translated_text,
            "source_language": src or "auto",
            "target_language": target_language,
            "confidence": None,
        }
        if len(_GOOGLE_TRANSLATION_CACHE) >= _GOOGLE_TRANSLATION_CACHE_MAX:
            _GOOGLE_TRANSLATION_CACHE.clear()
        _GOOGLE_TRANSLATION_CACHE[cache_key] = dict(out)
        return out
    except Exception as e:
        return {
            "original_text": text,
            "translated_text": f"Translation error: {str(e)}",
            "source_language": src or "auto",
            "target_language": target_language,
            "confidence": None,
        }


def translate_batch_google(
    texts: List[str],
    target_language: str,
    source_language: Optional[str],
    project_id: Optional[str] = None,
    fast: bool = False,
    *,
    fallback_batch_fn: Optional[Callable[[List[str], str, Optional[str]], List[dict]]] = None,
    timeout_sec: Optional[float] = None,
    perf_stats: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """
    Batch translation using the minimum number of Google list-translate HTTP calls (packed chunks).

    Optional ``fallback_batch_fn`` (e.g. local NLLB ``translate_batch``) runs when a chunk hits the
    deadline or persistent API failure — avoids unbounded bisection / per-string Google calls.
    """
    src = (source_language or "").strip().lower()
    ts = timeout_sec if timeout_sec is not None else _google_http_timeout_sec()
    normalized: List[str] = [
        _normalize_nllb_source_text((t or "").strip(), LANG_TO_NLLB.get(src or "en"))
        for t in (texts or [])
    ]
    if not normalized:
        return []
    if src and src == target_language.lower():
        return [
            {
                "original_text": texts[i],
                "translated_text": texts[i],
                "source_language": src,
                "target_language": target_language,
                "confidence": 1.0,
            }
            for i in range(len(texts))
        ]
    tgt_nllb = LANG_TO_NLLB.get((target_language or "").lower())
    out: List[Optional[dict]] = [None] * len(texts)

    max_items, max_chars_req = _google_batch_chunk_limits()
    chunks: List[List[int]] = []
    cur: List[int] = []
    cur_chars = 0
    for i, t in enumerate(normalized):
        tlen = max(1, len(t))
        if cur and (
            len(cur) >= max_items or (cur_chars + tlen) > max_chars_req
        ):
            chunks.append(cur)
            cur = []
            cur_chars = 0
        cur.append(i)
        cur_chars += tlen
    if cur:
        chunks.append(cur)

    _client = _google_client(project_id=project_id)
    _kw = (
        {
            "target_language": target_language,
            "source_language": src,
            "format_": "text",
        }
        if src
        else {
            "target_language": target_language,
            "format_": "text",
        }
    )

    def _fallback_bundle(chunk: List[int]) -> List[tuple[int, dict]]:
        if not fallback_batch_fn:
            raise RuntimeError("fallback_batch_fn required")
        subs = [texts[j] for j in chunk]
        fb_t0 = time.perf_counter()
        rows_fb = fallback_batch_fn(subs, target_language, source_language or src)
        _perf_add(perf_stats, "fallback_wall_seconds", time.perf_counter() - fb_t0)
        out_pairs: List[tuple[int, dict]] = []
        for k, j in enumerate(chunk):
            out_pairs.append((j, rows_fb[k]))
        return out_pairs

    def _translate_indices_recursive(chunk: List[int], depth: int = 0) -> List[tuple[int, dict]]:
        if not chunk:
            return []
        if len(chunk) == 1:
            i = chunk[0]

            def _single_fb() -> dict:
                return fallback_batch_fn(  # type: ignore[misc]
                    [texts[i]], target_language, source_language or src
                )[0]

            row = translate_text_google(
                texts[i],
                target_language=target_language,
                source_language=source_language,
                project_id=project_id,
                fast=fast,
                timeout_sec=ts,
                fallback_single_fn=_single_fb if fallback_batch_fn else None,
                perf_stats=perf_stats,
            )
            return [(i, row)]

        idx_texts = [normalized[i] for i in chunk]
        t_http = time.perf_counter()
        try:
            result = _translate_http(_client, idx_texts, _kw, ts)
        except TimeoutError:
            _perf_add(perf_stats, "google_wall_seconds", time.perf_counter() - t_http)
            if fallback_batch_fn:
                return _fallback_bundle(chunk)
            mid = len(chunk) // 2
            if mid == 0:
                return _translate_indices_recursive(chunk[:1], depth + 1)
            left, right = chunk[:mid], chunk[mid:]
            return _translate_indices_recursive(left, depth + 1) + _translate_indices_recursive(right, depth + 1)
        except Exception as e:
            _perf_add(perf_stats, "google_wall_seconds", time.perf_counter() - t_http)
            if _is_retryable_google_error(e):
                try:
                    retry_n = int(os.environ.get("GOOGLE_TRANSLATE_RETRY_COUNT", "2"))
                except ValueError:
                    retry_n = 2
                retry_n = max(0, min(4, retry_n))
                for attempt in range(retry_n):
                    try:
                        time.sleep(0.25 * (attempt + 1))
                        t2 = time.perf_counter()
                        result = _translate_http(_client, idx_texts, _kw, ts)
                        _perf_add(perf_stats, "google_wall_seconds", time.perf_counter() - t2)
                        _perf_inc(perf_stats, "http_calls")
                        rows = result if isinstance(result, list) else [result]
                        if len(rows) != len(chunk):
                            raise ValueError("Google batch response size mismatch")
                        mapped_r: List[tuple[int, dict]] = []
                        for local_i, row in enumerate(rows):
                            ii = chunk[local_i]
                            translated_text = html.unescape((row or {}).get("translatedText", "")).strip()
                            if tgt_nllb == "arb_Arab":
                                translated_text = _scrub_mixed_latin_noise_in_arabic_output(translated_text)
                            mapped_r.append(
                                (
                                    ii,
                                    {
                                        "original_text": texts[ii],
                                        "translated_text": translated_text,
                                        "source_language": src or "auto",
                                        "target_language": target_language,
                                        "confidence": None,
                                    },
                                )
                            )
                        return mapped_r
                    except Exception:
                        continue
            if fallback_batch_fn:
                return _fallback_bundle(chunk)
            mid = len(chunk) // 2
            left, right = chunk[:mid], chunk[mid:]
            try:
                pb_min = int(os.environ.get("GOOGLE_TRANSLATE_PARALLEL_BISECT_MIN", "64"))
            except ValueError:
                pb_min = 64
            pb_min = max(24, pb_min)
            parallel_bisect = (os.environ.get("GOOGLE_TRANSLATE_PARALLEL_BISECT", "1") or "").strip().lower() not in (
                "0",
                "false",
                "no",
            )
            try:
                pb_depth = int(os.environ.get("GOOGLE_TRANSLATE_PARALLEL_BISECT_MAX_DEPTH", "1"))
            except ValueError:
                pb_depth = 1
            pb_depth = max(0, min(2, pb_depth))
            if parallel_bisect and len(chunk) >= pb_min and depth <= pb_depth:
                with ThreadPoolExecutor(max_workers=2) as ex:
                    fl = ex.submit(_translate_indices_recursive, left, depth + 1)
                    fr = ex.submit(_translate_indices_recursive, right, depth + 1)
                    return fl.result() + fr.result()
            return _translate_indices_recursive(left, depth + 1) + _translate_indices_recursive(right, depth + 1)

        _perf_add(perf_stats, "google_wall_seconds", time.perf_counter() - t_http)
        _perf_inc(perf_stats, "http_calls")
        rows = result if isinstance(result, list) else [result]
        if len(rows) != len(chunk):
            if fallback_batch_fn:
                return _fallback_bundle(chunk)
            mid = len(chunk) // 2
            left, right = chunk[:mid], chunk[mid:]
            return _translate_indices_recursive(left, depth + 1) + _translate_indices_recursive(right, depth + 1)

        mapped: List[tuple[int, dict]] = []
        for local_i, row in enumerate(rows):
            i = chunk[local_i]
            translated_text = html.unescape((row or {}).get("translatedText", "")).strip()
            if tgt_nllb == "arb_Arab":
                translated_text = _scrub_mixed_latin_noise_in_arabic_output(translated_text)
            mapped.append(
                (
                    i,
                    {
                        "original_text": texts[i],
                        "translated_text": translated_text,
                        "source_language": src or "auto",
                        "target_language": target_language,
                        "confidence": None,
                    },
                )
            )
        return mapped

    def _translate_chunk(chunk: List[int]) -> List[tuple[int, dict]]:
        return _translate_indices_recursive(chunk)

    workers = _batch_workers_config()
    if workers == 1 or len(chunks) <= 1:
        for chunk in chunks:
            for i, row in _translate_chunk(chunk):
                out[i] = row
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(chunks))) as ex:
            fut_map = {ex.submit(_translate_chunk, chunk): chunk for chunk in chunks}
            for fut in as_completed(fut_map):
                for i, row in fut.result():
                    out[i] = row

    if (os.environ.get("GOOGLE_TRANSLATE_PERF_LOG", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        http_n = (perf_stats or {}).get("http_calls", "?")
        print(
            f"[google_translator] translate_batch_google chunks={len(chunks)} "
            f"texts={len(texts)} workers={workers} http_calls={http_n}",
            flush=True,
        )

    missing_idx = [i for i, r in enumerate(out) if r is None]
    if missing_idx:
        if len(missing_idx) <= 2:
            for ii in missing_idx:

                def _make_fb_single(j: int):
                    def _one_fb() -> dict:
                        return fallback_batch_fn(  # type: ignore[misc]
                            [texts[j]], target_language, source_language or src
                        )[0]

                    return _one_fb

                out[ii] = translate_text_google(
                    texts[ii],
                    target_language=target_language,
                    source_language=src or None,
                    project_id=project_id,
                    fast=fast,
                    timeout_sec=ts,
                    fallback_single_fn=_make_fb_single(ii) if fallback_batch_fn else None,
                    perf_stats=perf_stats,
                )
        else:
            fill_workers = min(_batch_workers_config(), len(missing_idx))

            def _fill_one(ii: int) -> tuple[int, dict]:
                def _fb_one() -> dict:
                    return fallback_batch_fn(  # type: ignore[misc]
                        [texts[ii]], target_language, source_language or src
                    )[0]

                return (
                    ii,
                    translate_text_google(
                        texts[ii],
                        target_language=target_language,
                        source_language=src or None,
                        project_id=project_id,
                        fast=fast,
                        timeout_sec=ts,
                        fallback_single_fn=_fb_one if fallback_batch_fn else None,
                        perf_stats=perf_stats,
                    ),
                )

            with ThreadPoolExecutor(max_workers=max(1, fill_workers)) as ex:
                futs = [ex.submit(_fill_one, i) for i in missing_idx]
                for fut in as_completed(futs):
                    i, row = fut.result()
                    out[i] = row
    for i, row in enumerate(out):
        if row is None:
            continue
        raw = _normalize_nllb_source_text((texts[i] or "").strip(), LANG_TO_NLLB.get(src or "en"))
        if not raw:
            continue
        ck = (raw, target_language.lower(), src or "auto", project_id or "")
        if len(_GOOGLE_TRANSLATION_CACHE) >= _GOOGLE_TRANSLATION_CACHE_MAX:
            _GOOGLE_TRANSLATION_CACHE.clear()
        _GOOGLE_TRANSLATION_CACHE[ck] = dict(row)
    return out  # type: ignore[return-value]


def _joined_parallel_ranges(texts: List[str]) -> List[tuple[int, int]]:
    """
    Split [0, len(texts)) into consecutive ranges for parallel marker-joined translates.

    Joined mode sends **one string** per HTTP call; the v2 limit of 128 strings applies to
    **list** batch translate only — do not shard every ~128 lines here (that inflated request count).

    Env:
      GOOGLE_JOINED_PARALLEL — default on; set 0 for one monolithic joined request.
      GOOGLE_JOINED_PARALLEL_MAX_CHARS — soft cap per shard (~Basic 100KB single request); default 99800.
      GOOGLE_TRANSLATE_JOINED_LEGACY_MARKERS=1 — use old [[[SEG_0001]]] markers if a model mangles [#T…#].
      GOOGLE_TRANSLATE_ADVANCED — set to 1 if you use Translation Advanced only (~30k code points/request);
          auto-lowers default joined shard chars when unset.
      GOOGLE_JOINED_PARALLEL_MAX_SEGMENTS — safety cap on lines per shard (default 8000); rarely hits.
    """
    n = len(texts)
    if n == 0:
        return []
    parallel = (os.environ.get("GOOGLE_JOINED_PARALLEL", "1") or "").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    if not parallel:
        return [(0, n)]
    advanced = (os.environ.get("GOOGLE_TRANSLATE_ADVANCED", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    try:
        max_c = int(
            os.environ.get(
                "GOOGLE_JOINED_PARALLEL_MAX_CHARS",
                "26000" if advanced else "99800",
            )
        )
    except ValueError:
        max_c = 26000 if advanced else 99800
    max_c = max(4096, min(100_000, max_c))
    try:
        max_seg = int(os.environ.get("GOOGLE_JOINED_PARALLEL_MAX_SEGMENTS", "8000"))
    except ValueError:
        max_seg = 8000
    max_seg = max(50, max_seg)

    ranges: List[tuple[int, int]] = []
    start = 0
    while start < n:
        seg_count = 0
        acc = 0
        i = start
        # Per line: compact marker [#T99999#] + newline (~12 chars overhead vs ~24 for legacy markers).
        while i < n:
            piece = len(texts[i] or "") + 12
            if seg_count > 0 and (seg_count >= max_seg or acc + piece > max_c):
                break
            seg_count += 1
            acc += piece
            i += 1
        ranges.append((start, i))
        start = i
    return ranges


def _translate_joined_google_shard(
    texts: List[str],
    target_language: str,
    source_language: Optional[str],
    project_id: Optional[str],
    fast: bool,
    fallback_batch_fn: Optional[Callable[[List[str], str, Optional[str]], List[dict]]] = None,
    timeout_sec: Optional[float] = None,
    perf_stats: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """
    One marker-joined Google request for a contiguous slice of segments.
    Falls back to translate_batch_google for this slice if markers do not round-trip.
    """
    if not texts:
        return []
    src_low = (source_language or "").strip().lower() or None
    legacy = (os.environ.get("GOOGLE_TRANSLATE_JOINED_LEGACY_MARKERS", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if legacy:
        markers = [f"[[[SEG_{i:04d}]]]" for i in range(len(texts))]
        split_rx = r"(\[\[\[SEG_\d{4}\]\]\])"
        match_rx = r"\[\[\[SEG_\d{4}\]\]\]"
    else:
        markers = [_joined_segment_marker(i) for i in range(len(texts))]
        split_rx = r"(\[#T\d{5}#\])"
        match_rx = r"\[#T\d{5}#\]"
    joined = "\n".join(f"{markers[i]} {texts[i] or ''}" for i in range(len(texts)))
    raw_payload = _normalize_nllb_source_text(joined.strip(), LANG_TO_NLLB.get(src_low or "en"))
    if not raw_payload:
        return [
            {
                "original_text": texts[i],
                "translated_text": "",
                "source_language": src_low or "auto",
                "target_language": target_language,
                "confidence": None,
            }
            for i in range(len(texts))
        ]

    ts = timeout_sec if timeout_sec is not None else _google_http_timeout_sec()
    client = _google_client(project_id=project_id)
    req: dict = {"target_language": target_language, "format_": "text"}
    if src_low:
        req["source_language"] = src_low
    t_http = time.perf_counter()
    try:
        result = _translate_http(client, raw_payload, req, ts)
    except TimeoutError:
        _perf_add(perf_stats, "google_wall_seconds", time.perf_counter() - t_http)
        if fallback_batch_fn:
            return fallback_batch_fn(texts, target_language, src_low)
        return translate_batch_google(
            texts,
            target_language,
            src_low,
            project_id=project_id,
            fast=fast,
            fallback_batch_fn=fallback_batch_fn,
            timeout_sec=timeout_sec,
            perf_stats=perf_stats,
        )
    except Exception:
        _perf_add(perf_stats, "google_wall_seconds", time.perf_counter() - t_http)
        if fallback_batch_fn:
            return fallback_batch_fn(texts, target_language, src_low)
        return translate_batch_google(
            texts,
            target_language,
            src_low,
            project_id=project_id,
            fast=fast,
            fallback_batch_fn=fallback_batch_fn,
            timeout_sec=timeout_sec,
            perf_stats=perf_stats,
        )

    _perf_add(perf_stats, "google_wall_seconds", time.perf_counter() - t_http)
    _perf_inc(perf_stats, "http_calls")
    out_joined = html.unescape((result or {}).get("translatedText", "")).strip()
    out_joined = str(out_joined)
    tgt_nllb = LANG_TO_NLLB.get((target_language or "").lower())
    if tgt_nllb == "arb_Arab":
        out_joined = _scrub_mixed_latin_noise_in_arabic_output(out_joined)

    parts = re.split(split_rx, out_joined)
    parsed: dict[str, str] = {}
    cur_marker: Optional[str] = None
    for p in parts:
        if not p:
            continue
        if re.fullmatch(match_rx, p):
            cur_marker = p
            parsed[cur_marker] = ""
        elif cur_marker is not None:
            parsed[cur_marker] = (parsed.get(cur_marker, "") + p).strip()
    if len(parsed) != len(texts):
        return translate_batch_google(
            texts,
            target_language=target_language,
            source_language=src_low,
            project_id=project_id,
            fast=fast,
            fallback_batch_fn=fallback_batch_fn,
            timeout_sec=timeout_sec,
            perf_stats=perf_stats,
        )
    out: List[dict] = []
    for i, m in enumerate(markers):
        txt = (parsed.get(m) or "").strip()
        if tgt_nllb == "arb_Arab":
            txt = _scrub_mixed_latin_noise_in_arabic_output(txt)
        out.append(
            {
                "original_text": texts[i],
                "translated_text": txt,
                "source_language": src_low or "auto",
                "target_language": target_language,
                "confidence": None,
            }
        )
    return out


def translate_joined_google(
    texts: List[str],
    target_language: str,
    source_language: Optional[str],
    project_id: Optional[str] = None,
    fast: bool = False,
    *,
    fallback_batch_fn: Optional[Callable[[List[str], str, Optional[str]], List[dict]]] = None,
    timeout_sec: Optional[float] = None,
    perf_stats: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """
    Translate many segments using stable segment markers. Large jobs are split into
    parallel shard requests so one huge document does not serialize on a single ~30s+ API call.
    Falls back to translate_batch_google per shard if marker parsing is imperfect.
    """
    if not texts:
        return []
    ranges = _joined_parallel_ranges(texts)
    if len(ranges) == 1:
        a, b = ranges[0]
        return _translate_joined_google_shard(
            texts[a:b],
            target_language,
            source_language,
            project_id,
            fast,
            fallback_batch_fn=fallback_batch_fn,
            timeout_sec=timeout_sec,
            perf_stats=perf_stats,
        )
    workers = _joined_parallel_workers(len(ranges))
    shard_rows: List[Optional[List[dict]]] = [None] * len(ranges)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        fut_to_ri = {
            ex.submit(
                _translate_joined_google_shard,
                texts[a:b],
                target_language,
                source_language,
                project_id,
                fast,
                fallback_batch_fn,
                timeout_sec,
                perf_stats,
            ): ri
            for ri, (a, b) in enumerate(ranges)
        }
        for fut in as_completed(fut_to_ri):
            ri = fut_to_ri[fut]
            shard_rows[ri] = fut.result()
    out: List[dict] = []
    for part in shard_rows:
        if part is not None:
            out.extend(part)
    return out


def translate_ocr_segments_google(
    texts: List[str],
    target_language: str,
    source_language: Optional[str],
    *,
    project_id: Optional[str] = None,
    fast: bool = False,
    prefer_joined: Optional[bool] = None,
    fallback_batch_fn: Optional[Callable[[List[str], str, Optional[str]], List[dict]]] = None,
    timeout_sec: Optional[float] = None,
    perf_stats: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """
    Single entry point: packs OCR segments into the fewest Google Cloud Translate HTTP calls
    (marker-joined document or list-batch chunks), with optional deadline and local fallback.

    ``prefer_joined``: when ``None``, uses ``google_translation_should_use_joined`` heuristics.
    """
    warm_google_translate_client(project_id)
    if perf_stats is not None:
        perf_stats.setdefault("segments_in", len(texts))
    use_joined = (
        prefer_joined
        if prefer_joined is not None
        else google_translation_should_use_joined(texts, source_language)
    )
    if perf_stats is not None:
        perf_stats["strategy"] = "marker_joined" if use_joined else "list_batch"
    if use_joined:
        return translate_joined_google(
            texts,
            target_language,
            source_language,
            project_id=project_id,
            fast=fast,
            fallback_batch_fn=fallback_batch_fn,
            timeout_sec=timeout_sec,
            perf_stats=perf_stats,
        )
    return translate_batch_google(
        texts,
        target_language,
        source_language,
        project_id=project_id,
        fast=fast,
        fallback_batch_fn=fallback_batch_fn,
        timeout_sec=timeout_sec,
        perf_stats=perf_stats,
    )
