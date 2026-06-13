"""
Persist OCR Studio history and saved translations.

When MONGODB_URI is set (and pymongo is installed), data is stored in MongoDB.
Otherwise falls back to JSON files next to the app.

Legacy JSON files are removed only after the first successful MongoDB connection.

Optional: OCR_STUDIO_RESET_STORAGE=1 clears MongoDB history + saved lists on startup (then loads empty).
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from typing import Any, List, Optional

_ROOT = os.path.dirname(os.path.abspath(__file__))
_HISTORY_FILE = os.path.join(_ROOT, ".ocr_studio_history.json")
_SAVED_TRANSLATIONS_FILE = os.path.join(_ROOT, ".ocr_studio_saved_translations.json")

_HISTORY_LIMIT = 50
_SAVED_LIMIT = 100

KV_COLLECTION = "studio_kv"
DOC_HISTORY = "history"
DOC_SAVED = "saved_translations"

_client = None
_kv_coll = None
_legacy_json_purged = False
_mongo_reset_done = False
_mongo_error: Optional[str] = None
_pymongo_import_cached: Optional[bool] = None


def _pymongo_import_ok() -> bool:
    """Whether pymongo can be imported in this interpreter (cached)."""
    global _pymongo_import_cached
    if _pymongo_import_cached is not None:
        return _pymongo_import_cached
    try:
        importlib.import_module("pymongo")
        _pymongo_import_cached = True
    except ImportError:
        _pymongo_import_cached = False
    return _pymongo_import_cached


def _set_mongo_error(msg: str) -> None:
    global _mongo_error
    _mongo_error = msg


def _clear_mongo_error() -> None:
    global _mongo_error
    _mongo_error = None


def mongodb_unavailable_message() -> Optional[str]:
    """
    User-facing message when MONGODB_URI is set but MongoDB cannot be used.
    History/saved items still fall back to local JSON when possible.
    """
    if not mongo_configured():
        return None
    if not _pymongo_import_ok():
        exe = sys.executable
        return (
            "MongoDB is configured (MONGODB_URI) but the pymongo package is not installed "
            f"for this Python interpreter ({exe}). "
            f'Install with: "{exe}" -m pip install pymongo — then restart the app. '
            "Until then, history and saved translations use local JSON files only."
        )
    if _mongo_error:
        return _mongo_error
    return None


def _mongo_uri() -> str:
    return (os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI") or "").strip()


def _db_name() -> str:
    return (os.environ.get("MONGODB_DB") or "ocr_studio").strip() or "ocr_studio"


def mongo_configured() -> bool:
    return bool(_mongo_uri())


def uses_mongodb() -> bool:
    if not mongo_configured():
        return False
    return _pymongo_import_ok()


def history_storage_caption() -> str:
    if mongo_configured():
        return (
            f"Recent runs target MongoDB (`{_db_name()}.{KV_COLLECTION}`, last {_HISTORY_LIMIT}). "
            f"If the database is unreachable, the app falls back to local `.ocr_studio_history.json`."
        )
    return f"Recent runs are saved locally (`.ocr_studio_history.json`, last {_HISTORY_LIMIT})."


def saved_storage_caption() -> str:
    if mongo_configured():
        return (
            f"Saved items target MongoDB (`{_db_name()}.{KV_COLLECTION}`, last {_SAVED_LIMIT}). "
            f"If the database is unreachable, the app falls back to local `.ocr_studio_saved_translations.json`."
        )
    return f"Saved items are stored locally (`.ocr_studio_saved_translations.json`, last {_SAVED_LIMIT})."


def _purge_legacy_json_files() -> None:
    global _legacy_json_purged
    if _legacy_json_purged:
        return
    _legacy_json_purged = True
    for path in (_HISTORY_FILE, _SAVED_TRANSLATIONS_FILE):
        try:
            if os.path.isfile(path):
                os.unlink(path)
        except OSError:
            pass


def _maybe_reset_mongo(coll: Any) -> None:
    global _mongo_reset_done
    if _mongo_reset_done:
        return
    flag = os.environ.get("OCR_STUDIO_RESET_STORAGE", "").strip().lower()
    if flag not in ("1", "true", "yes"):
        _mongo_reset_done = True
        return
    try:
        coll.replace_one(
            {"_id": DOC_HISTORY},
            {"_id": DOC_HISTORY, "items": []},
            upsert=True,
        )
        coll.replace_one(
            {"_id": DOC_SAVED},
            {"_id": DOC_SAVED, "items": []},
            upsert=True,
        )
        print(
            "[studio_storage] OCR_STUDIO_RESET_STORAGE: cleared MongoDB history and saved translations.",
            flush=True,
        )
    except Exception as exc:
        print(f"[studio_storage] OCR_STUDIO_RESET_STORAGE reset failed: {exc}", flush=True)
    _mongo_reset_done = True


def _kv_collection():
    """Return MongoDB collection; purge legacy JSON after first successful connect."""
    global _client, _kv_coll
    if _kv_coll is not None:
        return _kv_coll

    from pymongo import MongoClient  # type: ignore[import-untyped]

    uri = _mongo_uri()
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=10000)
        client.admin.command("ping")
        db = client[_db_name()]
        coll = db[KV_COLLECTION]
        _purge_legacy_json_files()
        _maybe_reset_mongo(coll)
        _client = client
        _kv_coll = coll
        _clear_mongo_error()
        return _kv_coll
    except Exception as exc:
        _set_mongo_error(
            "MongoDB is unreachable — history and saved translations are using local JSON until the connection works. "
            f"Details: {exc}"
        )
        raise


def _load_history_json() -> List[dict]:
    try:
        if os.path.isfile(_HISTORY_FILE):
            with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def _save_history_json(history: List[dict]) -> None:
    try:
        with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history[:_HISTORY_LIMIT], f, ensure_ascii=False, indent=0)
    except Exception:
        pass


def _load_saved_json() -> List[dict]:
    try:
        if os.path.isfile(_SAVED_TRANSLATIONS_FILE):
            with open(_SAVED_TRANSLATIONS_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)
                if isinstance(payload, list):
                    return payload
    except Exception:
        pass
    return []


def _save_saved_json(saved_items: List[dict]) -> None:
    try:
        with open(_SAVED_TRANSLATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(saved_items[:_SAVED_LIMIT], f, ensure_ascii=False, indent=0)
    except Exception:
        pass


def load_history() -> List[dict]:
    if uses_mongodb():
        try:
            coll = _kv_collection()
            doc = coll.find_one({"_id": DOC_HISTORY})
            items = doc.get("items") if doc else None
            if isinstance(items, list):
                out = [x for x in items if isinstance(x, dict)]
            else:
                out = []
            _clear_mongo_error()
            return out
        except Exception:
            return _load_history_json()
    return _load_history_json()


def save_history(history: List[dict]) -> None:
    items = history[:_HISTORY_LIMIT]
    if uses_mongodb():
        try:
            coll = _kv_collection()
            coll.replace_one(
                {"_id": DOC_HISTORY},
                {"_id": DOC_HISTORY, "items": items},
                upsert=True,
            )
            _clear_mongo_error()
            return
        except Exception as exc:
            _set_mongo_error(
                "MongoDB save failed — this run was saved to local JSON only. "
                f"Details: {exc}"
            )
    _save_history_json(items)


def load_saved_translations() -> List[dict]:
    if uses_mongodb():
        try:
            coll = _kv_collection()
            doc = coll.find_one({"_id": DOC_SAVED})
            items = doc.get("items") if doc else None
            if isinstance(items, list):
                out = [x for x in items if isinstance(x, dict)]
            else:
                out = []
            _clear_mongo_error()
            return out
        except Exception:
            return _load_saved_json()
    return _load_saved_json()


def save_saved_translations(saved_items: List[dict]) -> None:
    items = saved_items[:_SAVED_LIMIT]
    if uses_mongodb():
        try:
            coll = _kv_collection()
            coll.replace_one(
                {"_id": DOC_SAVED},
                {"_id": DOC_SAVED, "items": items},
                upsert=True,
            )
            _clear_mongo_error()
            return
        except Exception as exc:
            _set_mongo_error(
                "MongoDB save failed — saved items were written to local JSON only. "
                f"Details: {exc}"
            )
    _save_saved_json(items)
