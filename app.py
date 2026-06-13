"""
Streamlit web application for OCR Text Translation.
Provides a user-friendly interface for the translation pipeline.
"""

import os

# Avoid Streamlit's file watcher introspecting torch.classes (PyTorch uses a custom namespace;
# the watcher can trigger: Tried to instantiate class '__path__._path' ...).
# Use the canonical config env var and a legacy fallback.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("STREAMLIT_SERVER_ENABLE_FILE_WATCHER", "false")

# Translation VRAM: NLLB_USE_CPU / TRANSLATE_FORCE_CPU (see translator.py).
# OCR GPU: OCR_USE_GPU=0 and OCR_PADDLE_USE_CPU=1, or APP_USE_CPU=1 / UI **Compute** toggle.

import warnings

# Transformers lazy-import FutureWarning (e.g. ZoeDepth __path__). Unrelated to model downloads.
warnings.filterwarnings("ignore", message=r".*Accessing `__path__` from.*")
warnings.filterwarnings("ignore", message=r".*Returning `__path__` instead.*")
warnings.filterwarnings("ignore", message=r".*__path__.*zoedepth.*", category=FutureWarning)

# Pillow 10+ removed ANTIALIAS; EasyOCR still uses it.
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import io
import copy
import gc
import hashlib
import importlib
import inspect
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from ocr_translator import OCRTranslator
import ocr_translator as _ocr_module
from translator import TextTranslator, SUPPORTED_LANGUAGES, _heuristic_latin_source_when_src_equals_tgt
from google_translator import (
    google_translation_should_use_joined,
    translate_batch_google,
    translate_ocr_segments_google,
    translate_text_google,
    warm_google_translate_client,
)
from openai_translator import translate_batch_openai, translate_text_openai
from openai_image_photo import (
    default_image_model,
    edit_photo_with_translations,
    image_bytes_to_rgb_numpy,
)
from handwriting_io import cleanup_temp_paths, pdf_bytes_to_temp_png_paths
from handwriting_export import build_bilingual_pdf_bytes, build_bilingual_png_bytes
from openai_handwriting import (
    polish_bilingual_for_export,
    transcribe_and_translate_handwriting_from_image,
)
import tempfile
import shutil
import cv2
import json
import re
import unicodedata

try:
    
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False


# Single worker avoids overlapping OpenAI image jobs when using non-blocking ✨ enhance.
_ENHANCE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openai_enhance")

# Page configuration
st.set_page_config(
    page_title="Translate Studio",
    page_icon="✶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optional flag emoji for language grid (subset of SUPPORTED_LANGUAGES)
_LANG_FLAGS = {
    "en": "🇬🇧", "es": "🇪🇸", "fr": "🇫🇷", "de": "🇩🇪", "it": "🇮🇹", "pt": "🇵🇹",
    "ru": "🇷🇺", "ja": "🇯🇵", "ko": "🇰🇷", "zh": "🇨🇳", "ar": "🇸🇦", "hi": "🇮🇳",
    "nl": "🇳🇱", "pl": "🇵🇱", "tr": "🇹🇷", "vi": "🇻🇳", "th": "🇹🇭", "cs": "🇨🇿",
    "sv": "🇸🇪", "da": "🇩🇰", "no": "🇳🇴", "fi": "🇫🇮", "el": "🇬🇷", "he": "🇮🇱",
    "id": "🇮🇩", "ms": "🇲🇾", "uk": "🇺🇦", "ro": "🇷🇴", "hu": "🇭🇺", "sk": "🇸🇰",
    "bg": "🇧🇬", "hr": "🇭🇷", "sr": "🇷🇸", "sl": "🇸🇮", "et": "🇪🇪", "lv": "🇱🇻", "lt": "🇱🇹",
}

_POPULAR_LANG_CODES = ("hi", "de", "es", "fr", "zh", "en", "th", "it", "ja", "pt", "ar", "ru")


def apply_custom_styles():
    """Navy / purple glass-style UI, centered for desktop width."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;1,14..32,400&display=swap');

        :root {
            --space-bg: #060818;
            --space-bg-soft: #0d1330;
            --panel-bg: rgba(12, 20, 47, 0.85);
            --panel-border: rgba(150, 110, 255, 0.22);
            --text-main: #f8f8ff;
            --text-soft: #b6bdd7;
            --accent-start: #7b3cff;
            --accent-end: #d16bff;
            --accent-cyan: #56b8ff;
        }

        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', ui-sans-serif, system-ui, sans-serif !important;
        }
        .main .block-container {
            padding-top: 0.5rem; padding-bottom: 4rem; max-width: 1180px;
        }
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(1300px 620px at 20% -8%, rgba(120, 60, 255, 0.35) 0%, transparent 58%),
                        radial-gradient(900px 420px at 85% 12%, rgba(77, 173, 255, 0.18) 0%, transparent 60%),
                        radial-gradient(1000px 520px at 50% 100%, rgba(165, 92, 255, 0.14) 0%, transparent 65%),
                        var(--space-bg) !important;
        }
        [data-testid="stHeader"] { background: transparent !important; }
        [data-testid="stToolbar"] { background: rgba(10, 16, 38, 0.72); border-bottom: 1px solid var(--panel-border); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(11, 17, 40, 0.96) 0%, rgba(8, 13, 32, 0.96) 100%) !important;
            border-right: 1px solid var(--panel-border) !important;
        }
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label { color: var(--text-soft) !important; }
        [data-testid="stSidebar"] .stButton > button[kind="secondary"],
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            width: 100%;
            justify-content: flex-start;
            border-radius: 0.85rem !important;
            padding: 0.58rem 0.8rem !important;
            font-size: 0.9rem !important;
        }

        .app-topbar {
            display: flex; align-items: center; justify-content: space-between;
            padding: 0.4rem 0 0.9rem; margin-bottom: 0.15rem; max-width: 1180px; margin-left: auto; margin-right: auto;
        }
        .side-brand {
            display: flex;
            gap: 0.65rem;
            align-items: center;
            padding: 0.55rem 0.35rem 0.9rem;
        }
        .side-logo {
            width: 2.4rem;
            height: 2.4rem;
            border-radius: 0.65rem;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #6e3dff 0%, #3b82f6 100%);
            color: #fff;
            font-weight: 700;
            box-shadow: 0 0 22px rgba(110, 61, 255, 0.4);
        }
        .side-title { margin: 0; color: #f0f4ff; font-weight: 700; font-size: 1.02rem; }
        .side-sub { margin: 0; color: #9aa5c5; font-size: 0.74rem; }
        .side-divider { height: 1px; background: rgba(154, 165, 197, 0.24); margin: 0.9rem 0; }
        .premium-card {
            margin-top: 0.8rem;
            border: 1px solid rgba(143, 113, 255, 0.45);
            border-radius: 0.95rem;
            padding: 0.8rem;
            background: linear-gradient(180deg, rgba(47, 30, 108, 0.55) 0%, rgba(15, 24, 55, 0.8) 100%);
        }
        .premium-title { margin: 0; color: #f5d57b; font-weight: 700; }
        .premium-sub { margin: 0.2rem 0 0; color: #b3bdd9; font-size: 0.78rem; }
        .hello-line { margin: 0; color: #ccd4ee; font-size: 1.05rem; font-weight: 600; }
        .mode-card {
            border: 1px solid rgba(138, 168, 255, 0.28);
            border-radius: 1rem;
            padding: 0.85rem 0.95rem;
            min-height: 118px;
            background:
                radial-gradient(120% 100% at 15% 8%, rgba(96, 164, 255, 0.2) 0%, transparent 50%),
                radial-gradient(120% 100% at 88% 15%, rgba(187, 128, 255, 0.18) 0%, transparent 52%),
                rgba(16, 25, 58, 0.86);
            margin-bottom: 0.5rem;
        }
        .mode-ico {
            width: 2rem; height: 2rem; border-radius: 0.55rem;
            display: flex; align-items: center; justify-content: center;
            background: linear-gradient(135deg, #6f3dff 0%, #2484ff 100%);
            margin-bottom: 0.5rem;
        }
        .mode-title { margin: 0; color: #f3f6ff; font-size: 1.05rem; font-weight: 700; line-height: 1.2; }
        .mode-desc { margin: 0.25rem 0 0; color: #aeb8d4; font-size: 0.8rem; line-height: 1.35; }
        .tb-burger { font-size: 1.25rem; color: #94a3b8; }
        .tb-pro {
            display: inline-flex; align-items: center; gap: 0.3rem;
            background: linear-gradient(135deg, #6d28d9 0%, #4f46e5 100%);
            color: #f5f3ff; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em;
            padding: 0.2rem 0.6rem; border-radius: 999px; box-shadow: 0 0 20px rgba(109, 40, 217, 0.35);
        }
        .tb-av { width: 36px; height: 36px; border-radius: 50%;
            background: linear-gradient(135deg, #22d3ee 0%, #6366f1 100%); box-shadow: 0 0 0 2px #1e1b4b; }
        .hero-desk { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 1.25rem; margin-bottom: 1rem; }
        .hero-title { font-size: clamp(1.45rem, 2.2vw, 1.9rem); font-weight: 700; line-height: 1.2; margin: 0; letter-spacing: -0.03em; }
        .hero-title .g-grad { background: linear-gradient(90deg, #38bdf8, #a78bfa, #c084fc);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
        .hero-title .g-plain { color: #f8fafc; }
        .hero-sub { color: var(--text-soft); font-size: 0.92rem; margin: 0.5rem 0 0; max-width: 32rem; line-height: 1.5; }
        .hero-globe { position: relative; width: 100%; min-height: 120px; max-width: 320px; margin: 0 auto;
            background: radial-gradient(circle at 40% 40%, rgba(99, 102, 241, 0.35) 0%, transparent 50%),
                radial-gradient(circle at 60% 60%, rgba(6, 182, 212, 0.2) 0%, transparent 45%);
            border-radius: 1.5rem; border: 1px solid rgba(99, 102, 241, 0.25);
            box-shadow: inset 0 0 40px rgba(15, 23, 42, 0.6);
        }
        .hero-globe::before { content: ""; position: absolute; inset: 12% 20%; border-radius: 50%;
            background: linear-gradient(145deg, #1e1b4b, #312e81); box-shadow: 0 0 30px rgba(99, 102, 241, 0.4); }
        .bubble { position: absolute; font-size: 0.68rem; font-weight: 600; color: #e0e7ff; padding: 0.2rem 0.45rem; border-radius: 0.4rem; }
        .b1 { top: 8%; left: 8%; background: rgba(99, 102, 241, 0.5); }
        .b2 { top: 18%; right: 5%; background: rgba(6, 182, 212, 0.35); }
        .b3 { bottom: 20%; left: 12%; background: rgba(192, 132, 252, 0.4); }
        .b4 { bottom: 8%; right: 18%; background: rgba(99, 102, 241, 0.45); }

        .app-header { font-size: 1.35rem; font-weight: 700; color: #fafafa; margin: 0 0 0.35rem; letter-spacing: -0.03em; }
        .app-tagline { font-size: 0.875rem; color: #a0a0a0; margin-bottom: 1rem; line-height: 1.45; }
        .greeting-row { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.25rem; }
        .avatar-ring { width: 44px; height: 44px; border-radius: 50%;
            background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
            display: flex; align-items: center; justify-content: center; font-size: 1.25rem; flex-shrink: 0;
            box-shadow: 0 0 0 2px #1e1b4b; }
        .greeting-text { margin: 0; }
        .greeting-text strong { color: #fafafa; font-weight: 600; }
        .greeting-text span { color: #a0a0a0; font-size: 0.9rem; }
        .nav-shell-inner, .nav-shell { max-width: 1180px; }

        div[data-testid="stRadio"] > label { color: #a0a0a0 !important; font-size: 0.8rem !important; }
        div[data-testid="stRadio"] > div { gap: 0.5rem !important; flex-wrap: wrap !important; }
        div[data-testid="stRadio"] div[role="radiogroup"] {
            display: flex; flex-wrap: wrap; gap: 0.5rem; background: #12182a; padding: 0.4rem; border-radius: 999px;
            border: 1px solid rgba(99, 102, 241, 0.25);
        }
        div[data-testid="stRadio"] label {
            background: transparent !important; border: 1px solid rgba(99, 102, 241, 0.2) !important;
            border-radius: 999px !important; padding: 0.5rem 1.05rem !important; margin: 0 !important; color: #a0a0a0 !important;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"] span:last-child { font-size: 0.88rem !important; font-weight: 500 !important; }
        div[data-testid="stRadio"] label:has(input:checked) {
            background: linear-gradient(180deg, #6d28d9 0%, #5b21b6 100%) !important; color: #fafafa !important;
            border-color: transparent !important; box-shadow: 0 0 24px rgba(109, 40, 217, 0.35);
        }
        div[data-testid="stRadio"] label:has(input:checked) span { color: #fafafa !important; }
        [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
            display: flex !important;
            flex-direction: column !important;
            gap: 0.45rem !important;
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            padding: 0 !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] label {
            width: 100% !important;
            border-radius: 0.85rem !important;
            padding: 0.62rem 0.9rem !important;
            border: 1px solid rgba(99, 102, 241, 0.35) !important;
            background: rgba(15, 23, 42, 0.55) !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
            border-radius: 0.85rem !important;
            background: linear-gradient(180deg, #6d28d9 0%, #5b21b6 100%) !important;
            border-color: transparent !important;
            box-shadow: 0 0 18px rgba(109, 40, 217, 0.35) !important;
        }
        .st-key-text_tool_mic, .st-key-text_tool_scan, .st-key-text_tool_write {
            margin: 0 !important;
            position: static;
            z-index: 60;
            height: auto;
        }
        .st-key-text_tool_mic button, .st-key-text_tool_scan button, .st-key-text_tool_write button {
            width: 2.5rem !important;
            height: 2.5rem !important;
            min-height: 2.5rem !important;
            border-radius: 0.62rem !important;
            border: 1px solid rgba(238, 242, 255, 0.26) !important;
            background: rgba(8, 12, 22, 0.92) !important;
            color: #f8fbff !important;
            font-size: 1.05rem !important;
            padding: 0 !important;
            box-shadow: none !important;
        }
        .st-key-text_tool_mic button:hover, .st-key-text_tool_scan button:hover, .st-key-text_tool_write button:hover {
            border-color: rgba(248, 250, 255, 0.5) !important;
            background: rgba(16, 22, 38, 0.98) !important;
        }
        .st-key-switch_to_write_icon {
            display: flex;
            justify-content: flex-end;
            margin-top: 0.25rem;
        }
        .st-key-switch_to_write_icon button {
            width: 2.35rem !important;
            height: 2.35rem !important;
            min-height: 2.35rem !important;
            border-radius: 999px !important;
            padding: 0 !important;
        }

        .tx-card {
            background:
                radial-gradient(120% 90% at 12% 8%, rgba(126, 198, 255, 0.26) 0%, transparent 48%),
                radial-gradient(110% 90% at 86% 15%, rgba(186, 140, 255, 0.24) 0%, transparent 50%),
                linear-gradient(180deg, rgba(25, 42, 88, 0.88) 0%, rgba(13, 25, 58, 0.92) 100%);
            border: 1px solid rgba(142, 177, 255, 0.35); border-radius: 1.1rem;
            padding: 1rem 1.15rem; margin-bottom: 0.75rem; box-shadow: 0 10px 28px rgba(8, 14, 42, 0.46);
        }
        .tx-card-label {
            display: inline-block;
            font-size: 0.8rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #f1f5f9;
            margin-bottom: 0.55rem;
            padding: 0.28rem 0.55rem;
            border-radius: 0.4rem;
            background: rgba(2, 6, 23, 0.55);
            border: 1px solid rgba(226, 232, 240, 0.22);
            box-shadow: 0 1px 0 rgba(255, 255, 255, 0.06) inset;
        }
        .tx-card-body { color: #f1f5f9; font-size: 1rem; line-height: 1.55; white-space: pre-line; word-break: break-word; min-height: 4rem; }
        .wave-placeholder { height: 56px; display: flex; align-items: center; justify-content: center; gap: 3px; margin: 0.75rem 0; opacity: 0.85; }
        .wave-bar { width: 4px; background: #cbd5e1; border-radius: 2px; animation: wave 1.2s ease-in-out infinite; }
        .wave-bar:nth-child(odd) { animation-delay: 0.1s; }
        .wave-bar:nth-child(3n) { animation-delay: 0.25s; }
        @keyframes wave { 0%, 100% { height: 8px; opacity: 0.5; } 50% { height: 36px; opacity: 1; } }
        .translation-word-block {
            background:
                radial-gradient(130% 95% at 15% 12%, rgba(120, 193, 255, 0.23) 0%, transparent 52%),
                radial-gradient(110% 85% at 86% 14%, rgba(194, 145, 255, 0.2) 0%, transparent 55%),
                linear-gradient(180deg, rgba(24, 40, 82, 0.9) 0%, rgba(11, 23, 52, 0.95) 100%);
            color: #f4f4f5; padding: 1.25rem 1.5rem; border-radius: 1rem; min-height: 120px;
            border: 1px solid rgba(144, 179, 255, 0.34); white-space: pre-wrap; font-size: 1rem; line-height: 1.6; font-family: inherit;
        }
        .translation-word-block-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; margin-bottom: 0.5rem; }
        .stTabs { background: var(--panel-bg) !important; padding: 0.75rem; border-radius: 1rem; border: 1px solid var(--panel-border); }
        div[data-testid="stVerticalBlock"]:has(.stTabs) {
            background: var(--panel-bg); padding: 0.75rem; border-radius: 1rem; margin-top: 0.5rem; border: 1px solid var(--panel-border);
        }
        div[data-testid="stVerticalBlock"]:has(.stTabs) .stImage img { border-radius: 0.5rem; }
        div[data-testid="stVerticalBlock"]:has(.stTabs) [data-testid="stJson"] { background: #161b30; padding: 1rem; border-radius: 0.5rem; border: 1px solid rgba(99, 102, 241, 0.2); }
        div[data-testid="stVerticalBlock"]:has(.stTabs) .stMarkdown { color: #e4e4e7; }
        div[data-testid="stVerticalBlock"]:has(.stTabs) .stTextArea textarea { background: #161b30; color: #f4f4f5; border-color: #4338ca; }
        div[data-testid="stVerticalBlock"]:has(.stTabs) .stDownloadButton button { background: #1e1b4b; color: #f4f4f5; border-color: #4c1d95; }
        .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; background: rgba(17, 25, 56, 0.92); padding: 0.35rem; border-radius: 0.5rem; border: 1px solid var(--panel-border); }
        .stTabs [data-baseweb="tab"] { padding: 0.45rem 0.85rem; border-radius: 0.35rem; font-weight: 500; color: var(--text-soft); }
        .stTabs [aria-selected="true"] { background: linear-gradient(135deg, var(--accent-start) 0%, var(--accent-end) 100%) !important; color: #ffffff !important; box-shadow: 0 0 18px rgba(129, 73, 255, 0.45); }
        .stTextArea textarea {
            border-radius: 0.9rem !important; border-color: var(--panel-border) !important;
            background:
                radial-gradient(120% 95% at 14% 10%, rgba(130, 200, 255, 0.2) 0%, transparent 50%),
                radial-gradient(105% 90% at 88% 14%, rgba(188, 145, 255, 0.16) 0%, transparent 52%),
                rgba(14, 26, 61, 0.92) !important;
            color: var(--text-main) !important;
            font-family: 'Inter', ui-sans-serif, system-ui, sans-serif !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            line-height: 1.5 !important;
            -webkit-font-smoothing: antialiased !important;
            padding-right: 8.3rem !important;
            padding-bottom: 2.25rem !important;
        }
        /*
         * Translation output uses disabled=True (read-only). Base Web fades disabled fields via textarea
         * styles and often sets aria-disabled on the [data-baseweb="textarea"] wrapper — override both so
         * RTL/LTR match the editable source box (same color + weight as above).
         */
        [data-testid="stMain"] .stTextArea textarea[disabled],
        [data-testid="stMain"] .stTextArea textarea:disabled,
        .main .block-container .stTextArea textarea[disabled],
        .main .block-container .stTextArea textarea:disabled {
            opacity: 1 !important;
            color: var(--text-main) !important;
            -webkit-text-fill-color: var(--text-main) !important;
            font-family: 'Inter', ui-sans-serif, system-ui, sans-serif !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            line-height: 1.5 !important;
        }
        [data-testid="stMain"] [data-baseweb="textarea"][aria-disabled="true"],
        [data-testid="stMain"] [data-baseweb="textarea"][aria-disabled="true"] > div,
        .main .block-container [data-baseweb="textarea"][aria-disabled="true"],
        .main .block-container [data-baseweb="textarea"][aria-disabled="true"] > div {
            opacity: 1 !important;
        }
        /* Hide Streamlit default hint: "Press Ctrl+Enter to apply" */
        .stTextArea [data-testid="InputInstructions"] { display: none !important; }
        [data-testid="stSelectbox"] > div > div { background: rgba(17, 24, 56, 0.92) !important; border-color: var(--panel-border) !important; color: var(--text-main) !important; }
        .uploadedFile { border-radius: 0.9rem; border: 1px dashed rgba(165, 117, 255, 0.45); background: rgba(17, 24, 56, 0.9); }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--accent-start) 0%, var(--accent-end) 100%) !important; color: #ffffff !important;
            border: none !important; border-radius: 999px !important; padding: 0.55rem 1.35rem !important; font-weight: 600 !important;
            box-shadow: 0 0 22px rgba(129, 73, 255, 0.45);
        }
        .stButton > button[kind="primary"]:hover { background: linear-gradient(135deg, #8d50ff 0%, #e17dff 100%) !important; color: #fff !important; }
        .stButton > button[kind="secondary"] {
            background: rgba(16, 24, 55, 0.88) !important; color: #e6ebff !important; border: 1px solid var(--panel-border) !important; border-radius: 0.85rem !important;
        }
        .stButton > button[kind="secondary"]:hover { border-color: rgba(182, 126, 255, 0.7) !important; }
        .st-key-save_hist_item_unsaved button {
            color: #ffffff !important;
            font-size: 1.2rem !important;
            line-height: 1 !important;
        }
        .st-key-save_hist_item_saved button {
            color: #facc15 !important;
            font-size: 1.2rem !important;
            line-height: 1 !important;
        }
        .st-key-ai_enhance_photo_quick_btn button {
            width: 2.6rem !important;
            height: 2.6rem !important;
            min-height: 2.6rem !important;
            border-radius: 999px !important;
            border: 1px solid rgba(255, 255, 255, 0.32) !important;
            background:
                radial-gradient(70% 70% at 25% 20%, rgba(255, 255, 255, 0.44) 0%, transparent 45%),
                linear-gradient(135deg, #22d3ee 0%, #7c3aed 52%, #f472b6 100%) !important;
            color: #ffffff !important;
            font-size: 1.2rem !important;
            font-weight: 700 !important;
            box-shadow:
                0 0 0 2px rgba(168, 85, 247, 0.24),
                0 0 18px rgba(34, 211, 238, 0.38),
                0 0 30px rgba(244, 114, 182, 0.28) !important;
        }
        .st-key-ai_enhance_photo_quick_btn button:hover {
            transform: translateY(-1px) scale(1.04);
            border-color: rgba(255, 255, 255, 0.58) !important;
            box-shadow:
                0 0 0 2px rgba(56, 189, 248, 0.28),
                0 0 20px rgba(56, 189, 248, 0.48),
                0 0 36px rgba(232, 121, 249, 0.36) !important;
        }
        .st-key-ai_enhance_photo_quick_btn button:focus {
            outline: none !important;
            box-shadow:
                0 0 0 3px rgba(34, 211, 238, 0.3),
                0 0 22px rgba(167, 139, 250, 0.45) !important;
        }
        .streamlit-expanderHeader {
            background: rgba(16, 24, 55, 0.9) !important; border-radius: 0.9rem !important; border: 1px solid var(--panel-border) !important; color: #e9eeff !important;
        }
        .streamlit-expanderContent { background: rgba(11, 17, 39, 0.95) !important; border: 1px solid var(--panel-border); border-top: none; }
        [data-testid="stAlert"] { background: rgba(32, 24, 78, 0.82); border: 1px solid rgba(152, 104, 255, 0.42); }
        [data-testid="stMarkdownContainer"] p { color: inherit; }
        .lang-grid-title { font-size: 0.85rem; font-weight: 600; color: #a0a0a0; margin: 1rem 0 0.65rem; }
        .qa-wrap { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 0.5rem 0 1.15rem; }
        .qa-card { flex: 1; min-width: 140px; border-radius: 1rem; padding: 0.9rem 1rem; border: 1px solid rgba(255,255,255,0.08);
            background: #161b30; transition: transform 0.15s, box-shadow 0.15s; }
        .qa-card:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(0,0,0,0.25); }
        .qa-p { border-color: rgba(109, 40, 217, 0.4); } .qa-b { border-color: rgba(6, 182, 212, 0.35); } .qa-t { border-color: rgba(20, 184, 166, 0.35); }
        .qa-title { font-size: 0.92rem; font-weight: 700; color: #f8fafc; margin: 0 0 0.2rem; }
        .qa-desc { font-size: 0.75rem; color: #a0a0a0; margin: 0; line-height: 1.35; } .qa-ico { font-size: 1.25rem; margin-bottom: 0.35rem; }
        .lang-bar-caption { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; }
        .swap-hint { font-size: 0.7rem; color: #64748b; text-align: center; }

        /* Video previews */
        [data-testid="stMain"] [data-testid="stVideo"] {
          max-width: min(100%, 720px);
          margin-left: auto;
          margin-right: auto;
        }
        [data-testid="stMain"] [data-testid="stVideo"] video {
          max-height: min(48vh, 440px) !important;
          width: 100% !important;
          height: auto !important;
          object-fit: contain;
          border-radius: 0.5rem;
        }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _clipboard_button(label: str, text: str, key: str, tooltip: str = "Copy"):
    """Copy to clipboard via embedded HTML (HTTPS/localhost may be required for clipboard API)."""
    import streamlit.components.v1 as components
    safe = json.dumps(text)
    uid = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    icon_html = """
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true"
         xmlns="http://www.w3.org/2000/svg">
      <rect x="7" y="3" width="13" height="18" rx="3" stroke="currentColor" stroke-width="1.8"/>
      <rect x="3" y="7" width="13" height="14" rx="3" stroke="currentColor" stroke-width="1.8"/>
    </svg>
    """
    button_inner = icon_html if label in ("📋", "copy_icon") else label
    components.html(
        f"""
        <div style="font-family: system-ui,sans-serif;margin:0;padding:0;">
        <button id="cb_{uid}"
          title="{tooltip}"
          aria-label="{tooltip}"
          style="background:#161b30;color:#e2e8f0;border:1px solid rgba(99,102,241,0.35);border-radius:8px;padding:5px 9px;font-size:13px;cursor:pointer;display:inline-flex;align-items:center;justify-content:center;line-height:1;"
          type="button">{button_inner}</button>
        <script>
        document.getElementById("cb_{uid}").onclick = function() {{
          navigator.clipboard.writeText({safe}).then(function() {{
            var b = document.getElementById("cb_{uid}");
            var t = b.innerHTML;
            b.innerHTML = "✓";
            setTimeout(function() {{ b.innerHTML = t; }}, 1200);
          }});
        }};
        </script>
        </div>
        """,
        height=48,
    )


def _safe_show_image(
    img,
    *,
    caption: Optional[str] = None,
    use_column_width: bool = True,
    width: Optional[int] = None,
    original_bytes: Optional[bytes] = None,
):
    """Render images defensively to avoid PIL/Streamlit encoder edge-case crashes."""
    _kw: Dict[str, Any] = {"caption": caption, "output_format": "PNG"}
    if width is not None:
        _kw["width"] = int(width)
        _kw["use_column_width"] = False
    else:
        _kw["use_column_width"] = use_column_width
    try:
        st.image(img, **_kw)
        return
    except Exception:
        pass

    try:
        arr = _normalize_to_rgb_uint8_array(img)
        st.image(arr, **_kw)
        return
    except Exception as e:
        pass

    # Last-resort fallback: decode raw bytes via OpenCV and display RGB ndarray.
    if original_bytes:
        try:
            buf = np.frombuffer(original_bytes, dtype=np.uint8)
            dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if dec is not None and dec.size > 0:
                rgb = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
                st.image(rgb, **_kw)
                return
        except Exception:
            pass

    st.info("Preview unavailable for this image format.")


def _preprocess_rgb_to_jpeg_data_uri(rgb: np.ndarray, jpeg_quality: int = 86) -> str:
    """Encode an RGB uint8 preview as a compact JPEG data URI for embedding in HTML/JS."""
    import base64

    arr = _normalize_to_rgb_uint8_array(rgb)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    q = int(max(50, min(95, jpeg_quality)))
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok or buf is None:
        raise ValueError("JPEG encode failed for preprocessing preview.")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _bgr_to_preview_jpeg_bytes(bgr: np.ndarray, quality: int = 88) -> bytes:
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return b""
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else b""


def _preview_jpeg_bytes_to_rgb(jpeg_bytes: bytes) -> Optional[np.ndarray]:
    if not jpeg_bytes:
        return None
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _pack_preprocess_preview_into_payload(payload: dict) -> dict:
    """Attach JPEG previews from the last ``enhance_bgr_for_ocr`` run (for OCR cache)."""
    try:
        prev = _ocr_module.last_preprocess_preview()
    except Exception:
        return payload
    before = prev.get("before_bgr")
    after_oai = prev.get("after_openai_bgr")
    after_final = prev.get("after_final_bgr")
    if before is not None and before.size:
        payload["proc_before_jpeg"] = _bgr_to_preview_jpeg_bytes(before)
    if after_oai is not None and after_oai.size:
        payload["proc_after_openai_jpeg"] = _bgr_to_preview_jpeg_bytes(after_oai)
    if after_final is not None and after_final.size:
        payload["proc_after_jpeg"] = _bgr_to_preview_jpeg_bytes(after_final)
    payload["proc_openai_used"] = bool(prev.get("openai_used"))
    return payload


def _rgb_from_live_preprocess_bgr_cache() -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Read RGB previews from the BGR frames stored during the last OCR preprocess pass."""
    try:
        live = _ocr_module.last_preprocess_preview()
    except Exception:
        return None, None, None
    c_before = live.get("before_bgr")
    c_final = live.get("after_final_bgr")
    c_oai = live.get("after_openai_bgr")
    if c_before is None or not c_before.size or c_final is None or not c_final.size:
        return None, None, None
    before_rgb = cv2.cvtColor(c_before, cv2.COLOR_BGR2RGB)
    after_final_rgb = cv2.cvtColor(c_final, cv2.COLOR_BGR2RGB)
    after_openai_rgb = (
        cv2.cvtColor(c_oai, cv2.COLOR_BGR2RGB) if c_oai is not None and c_oai.size else None
    )
    if after_openai_rgb is None and live.get("openai_used"):
        after_openai_rgb = after_final_rgb
    return before_rgb, after_final_rgb, after_openai_rgb


def _sync_ocr_preprocess_settings(
    *,
    openai_preprocess: bool,
    local_preprocess: bool,
) -> None:
    _ocr = st.session_state.get("ocr")
    if _ocr is None:
        return
    _ocr.openai_preprocess = bool(openai_preprocess)
    _ocr.local_preprocess = bool(local_preprocess)
    key = (_resolve_openai_key() or "").strip()
    _ocr.openai_api_key = key or None


def _unpack_preprocess_preview_from_payload(
    payload: Optional[dict],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if not payload:
        return None, None, None
    before = _preview_jpeg_bytes_to_rgb(payload.get("proc_before_jpeg") or b"")
    after_final = _preview_jpeg_bytes_to_rgb(payload.get("proc_after_jpeg") or b"")
    after_oai = _preview_jpeg_bytes_to_rgb(payload.get("proc_after_openai_jpeg") or b"")
    return before, after_final, after_oai


def _load_preprocess_previews_for_image(
    overlay_path: str,
    *,
    local_preprocess: bool,
    openai_preprocess: bool,
    ocr_max_dim: int,
    ocr_small_text: bool,
    ocr_high_recall: bool,
    cached_payload: Optional[dict] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Before / final / OpenAI RGB previews for the Processing tab.

    Prefer JPEG blobs stored in the OCR cache; else frames from the last OCR preprocess pass;
    else rebuild via ``preprocess_preview_rgb_pair`` (may call OpenAI if cache is empty).
    """
    live_rgb = _rgb_from_live_preprocess_bgr_cache()
    if live_rgb[0] is not None and (live_rgb[1] is not None or live_rgb[2] is not None):
        return live_rgb

    pb, pa, poai = _unpack_preprocess_preview_from_payload(cached_payload)
    if pb is not None and (pa is not None or poai is not None):
        if poai is None and pa is not None and openai_preprocess and cached_payload and cached_payload.get(
            "proc_openai_used"
        ):
            poai = pa
        return pb, pa, poai

    _ocr_pv = st.session_state.get("ocr")
    if _ocr_pv is None or not hasattr(_ocr_pv, "preprocess_preview_rgb_pair"):
        return None, None, None
    if not overlay_path or not os.path.isfile(overlay_path):
        return None, None, None
    if not local_preprocess and not openai_preprocess:
        try:
            return _ocr_pv.preprocess_preview_rgb_pair(
                overlay_path,
                False,
                ocr_max_dim,
                ocr_small_text,
                ocr_high_recall,
            )
        except Exception:
            return None, None, None

    _sync_ocr_preprocess_settings(
        openai_preprocess=openai_preprocess,
        local_preprocess=local_preprocess,
    )
    try:
        return _ocr_pv.preprocess_preview_rgb_pair(
            overlay_path,
            bool(local_preprocess),
            ocr_max_dim,
            ocr_small_text,
            ocr_high_recall,
        )
    except Exception:
        return None, None, None


def _preprocess_arrays_visibly_differ(
    a: Optional[np.ndarray], b: Optional[np.ndarray], *, atol: int = 3
) -> bool:
    if a is None or b is None:
        return True
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.shape != bb.shape:
        return True
    try:
        return int(np.max(np.abs(aa.astype(np.int16) - bb.astype(np.int16)))) > atol
    except Exception:
        return True


def _render_preprocess_comparison_new_tab_button(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    *,
    component_key: str,
    page_title: str = "Preprocessing — large comparison",
    page_subtitle: str = "Same resolution as the OCR working image (before vs after env-driven steps).",
    before_heading: str = "Before preprocessing",
    after_heading: str = "After preprocessing",
) -> None:
    """Button that opens before/after preprocessing in a **new browser tab** (wide layout + per-pane ⤢ fullscreen)."""
    import streamlit.components.v1 as components

    try:
        b_uri = _preprocess_rgb_to_jpeg_data_uri(before_rgb)
        a_uri = _preprocess_rgb_to_jpeg_data_uri(after_rgb)
    except Exception:
        st.caption("Could not prepare images for opening in a new tab.")
        return
    uid = hashlib.md5(component_key.encode("utf-8")).hexdigest()[:14]
    idb, ida = f"wb_{uid}", f"wa_{uid}"
    css_esc = (
        "body{margin:0;background:#0b1020;color:#e2e8f0;font-family:system-ui,-apple-system,sans-serif;"
        "height:100vh;display:flex;flex-direction:column;}"
        "h1{margin:0;padding:14px 18px;font-size:1.05rem;font-weight:600;border-bottom:1px solid rgba(99,102,241,0.25);"
        "background:rgba(15,23,42,0.96);}"
        ".sub{padding:4px 18px 10px;font-size:0.78rem;color:#94a3b8;border-bottom:1px solid rgba(99,102,241,0.15);}"
        ".row{flex:1;display:flex;gap:14px;padding:14px;min-height:0;box-sizing:border-box;}"
        ".pane{flex:1;display:flex;flex-direction:column;min-width:0;background:#111827;border-radius:12px;"
        "border:1px solid rgba(99,102,241,0.35);overflow:hidden;position:relative;}"
        ".pane h2{font-size:0.8rem;margin:0;padding:10px 12px;background:rgba(17,24,39,0.98);"
        "font-weight:600;color:#cbd5e1;}"
        ".wrap{flex:1;display:flex;align-items:center;justify-content:center;min-height:0;padding:8px;background:#0f172a;}"
        ".wrap img{max-width:100%;max-height:calc(100vh - 150px);object-fit:contain;}"
        ".fs{position:absolute;top:10px;right:10px;width:38px;height:38px;border-radius:50%;"
        "border:1px solid rgba(255,255,255,0.28);background:rgba(0,0,0,0.58);color:#fff;font-size:17px;cursor:pointer;"
        "display:flex;align-items:center;justify-content:center;line-height:1;padding:0;}"
        ".fs:hover{background:rgba(30,27,75,0.92);}"
    )
    _sub_line = (page_subtitle or "").strip()
    _sub_html = ('<div class="sub">' + _sub_line + '</' + 'div' + '>') if _sub_line else ""
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{page_title}</title>
<style>{css_esc}</style>
</head>
<body>
<h1>{page_title}</h1>
{_sub_html}
<div class="row">
  <div class="pane">
    <h2>{before_heading}</h2>
    <div class="wrap" id="{idb}"><img src="{b_uri}" alt="Before"/></div>
    <button type="button" class="fs" title="Fullscreen this panel" aria-label="Fullscreen"
      onclick="var el=document.getElementById('{idb}');if(el){{if(el.requestFullscreen)el.requestFullscreen();else if(el.webkitRequestFullscreen)el.webkitRequestFullscreen();}}">⤢</button>
  </div>
  <div class="pane">
    <h2>{after_heading}</h2>
    <div class="wrap" id="{ida}"><img src="{a_uri}" alt="After"/></div>
    <button type="button" class="fs" title="Fullscreen this panel" aria-label="Fullscreen"
      onclick="var el=document.getElementById('{ida}');if(el){{if(el.requestFullscreen)el.requestFullscreen();else if(el.webkitRequestFullscreen)el.webkitRequestFullscreen();}}">⤢</button>
  </div>
</div>
</body>
</html>"""
    h_js = json.dumps(doc)
    html = (
        '<div style="font-family:system-ui,sans-serif;margin:6px 0 2px 0;">'
        f'<button type="button" id="ppt_{uid}" '
        'style="background:linear-gradient(135deg,#4c1d95 0%,#4338ca 100%);color:#fff;'
        'border:1px solid rgba(99,102,241,0.45);border-radius:10px;padding:8px 16px;font-size:13px;'
        'font-weight:600;cursor:pointer;">'
        "Open side-by-side in new tab"
        "</button></div>"
        "<script>"
        "(function(){"
        f'var btn=document.getElementById("ppt_{uid}");if(!btn)return;'
        f"var h={h_js};"
        'btn.addEventListener("click",function(){'
        'var w=window.open("","_blank");'
        "if(w){w.document.open();w.document.write(h);w.document.close();w.focus();}"
        "});"
        "})();"
        "</script>"
    )
    components.html(html, height=52)


def _render_preprocess_side_by_side_pair(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    *,
    before_caption: str,
    after_caption: str,
    zoom_large: bool,
    zoom_width: int = 920,
) -> None:
    if zoom_large:
        _c1, _c2 = st.columns(2)
        with _c1:
            _safe_show_image(before_rgb, caption=before_caption, width=zoom_width)
        with _c2:
            _safe_show_image(after_rgb, caption=after_caption, width=zoom_width)
    else:
        _c1, _c2 = st.columns(2)
        with _c1:
            _safe_show_image(before_rgb, caption=before_caption, use_column_width=True)
        with _c2:
            _safe_show_image(after_rgb, caption=after_caption, use_column_width=True)


def _render_processing_preprocess_views(
    proc_preview_before_rgb: Optional[np.ndarray],
    proc_preview_after_rgb: Optional[np.ndarray],
    proc_preview_after_openai_rgb: Optional[np.ndarray],
    *,
    preprocess: bool,
    ocr_openai_preprocess: bool,
    openai_preprocess_error: Optional[str] = None,
    zoom_state_key: str = "ocr_preprocess_zoom_large",
    zoom_toggle_key: str = "ocr_preprocess_zoom_toggle",
    new_tab_key_local: str = "ppt_preprocess_main",
    new_tab_key_openai: str = "ppt_preprocess_openai",
) -> None:
    """Processing tab: local and/or OpenAI before/after previews (same frames as OCR)."""
    if proc_preview_before_rgb is None:
        st.info("No preprocessing preview is available for this run.")
        return

    st.session_state.setdefault(zoom_state_key, False)
    zoom_large = bool(st.session_state.get(zoom_state_key))

    show_openai = (
        bool(ocr_openai_preprocess)
        and proc_preview_after_openai_rgb is not None
    )
    show_local = bool(preprocess) and proc_preview_after_rgb is not None
    any_preprocess_enabled = bool(preprocess) or bool(ocr_openai_preprocess)
    local_differs_from_openai = show_openai and show_local and _preprocess_arrays_visibly_differ(
        proc_preview_after_openai_rgb, proc_preview_after_rgb
    )

    if show_openai or show_local:
        st.button(
            "⊟ Smaller preview"
            if zoom_large
            else "🔍 Enlarge: before / after (side by side)",
            key=zoom_toggle_key,
            help="Toggle a large side-by-side view of preprocessing images used for OCR.",
            on_click=_ocr_preprocess_zoom_toggle_click,
        )

    if show_openai:
        st.markdown("#### OpenAI preprocessing (GPT-Image)")
        st.caption(
            "Before and after the OpenAI **images.edit** pass (contrast, glare, straighten — text must stay unchanged)."
        )
        _render_preprocess_comparison_new_tab_button(
            np.asarray(proc_preview_before_rgb).copy(),
            np.asarray(proc_preview_after_openai_rgb).copy(),
            component_key=new_tab_key_openai,
            page_title="OpenAI OCR preprocessing",
            page_subtitle="",
            before_heading="Before preprocessing",
            after_heading="After preprocessing",
        )
        _render_preprocess_side_by_side_pair(
            proc_preview_before_rgb,
            proc_preview_after_openai_rgb,
            before_caption="Before preprocessing (original OCR working image)",
            after_caption="After preprocessing (sent to OCR)",
            zoom_large=zoom_large,
        )

    if show_local and (not show_openai or local_differs_from_openai):
        if show_openai:
            st.markdown("#### Local preprocessing")
            st.caption("Additional OpenCV / GPU steps after OpenAI (`OCR_PREPROCESS_OPENAI_ONLY=0`).")
            local_before = proc_preview_after_openai_rgb
            local_before_cap = "Before local (after OpenAI)"
        else:
            st.markdown("#### Local preprocessing")
            local_before = proc_preview_before_rgb
            local_before_cap = "Before preprocessing (original OCR working image)"
        _render_preprocess_comparison_new_tab_button(
            np.asarray(local_before).copy(),
            np.asarray(proc_preview_after_rgb).copy(),
            component_key=new_tab_key_local,
            page_title="Local OCR preprocessing",
            page_subtitle="Before vs after local denoise / contrast / sharpen (env-driven).",
            before_heading="Before local",
            after_heading="After local (final for OCR)",
        )
        _render_preprocess_side_by_side_pair(
            local_before,
            proc_preview_after_rgb,
            before_caption=local_before_cap,
            after_caption="After preprocessing (final image for OCR)",
            zoom_large=zoom_large,
        )
    elif show_local and show_openai and not local_differs_from_openai:
        st.caption(
            "Local denoise was skipped (`OCR_PREPROCESS_OPENAI_ONLY=1`) — the OpenAI result above is the final OCR input."
        )

    if show_openai or show_local:
        st.caption(
            "Same image size as the OCR pass "
            "(includes perspective-straightened image when that option is on)."
        )
    elif not any_preprocess_enabled:
        _safe_show_image(
            proc_preview_before_rgb,
            caption="Working image for OCR (preprocessing disabled)",
            use_column_width=True,
        )
        st.info(
            "Turn on **OpenAI preprocess before OCR** and/or **Preprocess image before OCR** "
            "in Advanced options, then click **Translate** again to see before/after previews."
        )
    elif any_preprocess_enabled and not show_openai and not show_local:
        _safe_show_image(
            proc_preview_before_rgb,
            caption="Before preprocessing (OCR working size)",
            use_column_width=True,
        )
        st.warning(
            "Preprocessing was enabled for this run but no **after** preview is available. "
            "Click **Translate** again (avoid stale **cached OCR** — change an OCR option or re-upload). "
            "For OpenAI preprocess, confirm **OPENAI_API_KEY** and check the terminal for API errors."
        )
    elif ocr_openai_preprocess and proc_preview_after_openai_rgb is None:
        _safe_show_image(
            proc_preview_before_rgb,
            caption="Before OpenAI (OCR working size)",
            use_column_width=True,
        )
        _err = (openai_preprocess_error or "").strip()
        st.error(
            "OpenAI preprocess did not return an enhanced image — OCR used the original photo."
            + (f"\n\n**Detail:** `{_err}`" if _err else "")
            + "\n\nCheck **OPENAI_API_KEY** (sidebar or `.env`), billing, and that **gpt-image-1.5** is enabled for your account."
        )
    else:
        st.warning("Could not build the after-preprocessing preview for this file.")


def _video_read_frame_bgr_at(video_path: str, frame_idx: int):
    """Load one video frame by index for bbox overlay.

    Uses sequential ``grab``/``read`` from the start of the stream so the decoded frame matches
    ``extract_text_from_video_with_frames`` (which only uses ``cap.read`` in order). Many codecs
    report ``CAP_PROP_POS_FRAMES`` inaccurately, which misaligned boxes vs pixels.
    """
    if not video_path or not os.path.isfile(video_path):
        return None
    target = max(0, int(frame_idx))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        for _ in range(target):
            if not cap.grab():
                return None
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            return None
        return frame
    finally:
        cap.release()


def _video_overlay_frame_index_bgr(frame_bgr: np.ndarray, frame_index: int) -> np.ndarray:
    """Draw ``Frame <index>`` at the top-left (white text, dark outline) on a BGR image."""
    out = frame_bgr.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    h = int(out.shape[0])
    label = f"Frame {int(frame_index)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = float(max(0.55, min(2.2, h / 540.0)))
    thickness = max(2, int(round(h / 240.0)))
    (_tw, th), _baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad = max(6, h // 80)
    org = (pad, pad + th)
    outline = thickness + max(2, thickness // 2)
    cv2.putText(out, label, org, font, font_scale, (0, 0, 0), outline, cv2.LINE_AA)
    cv2.putText(out, label, org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def _video_overlay_frame_index_rgb(frame_rgb: np.ndarray, frame_index: int) -> np.ndarray:
    bgr = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(_video_overlay_frame_index_bgr(bgr, frame_index), cv2.COLOR_BGR2RGB)


def _video_annotate_slideshow_cache_sig(video_path: str, vp: list, fps: float) -> tuple:
    try:
        mt = os.path.getmtime(video_path)
    except OSError:
        mt = 0.0
    sig_frames = tuple((int(fi), len(r or [])) for fi, r in vp)
    # Bump when encoder output format changes (invalidates cached non-playable clips).
    _codec_v = 2
    return (os.path.abspath(video_path), float(mt), sig_frames, round(float(fps), 2), _codec_v)


def _video_annotate_slideshow_ffmpeg_pipe_to_mp4(
    frames_bgr: list,
    tw: int,
    th: int,
    *,
    fps: float,
) -> Optional[bytes]:
    """Encode BGR frames to browser-playable H.264 MP4 via bundled/system ffmpeg (rawvideo pipe)."""
    if not frames_bgr or not _ocr_module._ffmpeg_executable():
        return None
    out_path: Optional[str] = None
    proc = None
    try:
        fd, out_path = tempfile.mkstemp(suffix=".mp4", prefix="annotate_slideshow_ff_")
        os.close(fd)
        proc = _ocr_module._ffmpeg_encode_raw_bgr_start(
            out_path, int(tw), int(th), float(fps), use_avi=False
        )
        stdin = proc.stdin
        if stdin is None:
            return None
        try:
            for fr in frames_bgr:
                fr = np.ascontiguousarray(fr)
                stdin.write(fr.tobytes())
            stdin.close()
            stdin = None
        except BrokenPipeError:
            try:
                if stdin is not None:
                    stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=60)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            return None
        try:
            rc = proc.wait(timeout=600)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            return None
        if rc != 0 or not out_path or not os.path.isfile(out_path):
            return None
        try:
            sz = os.path.getsize(out_path)
        except OSError:
            sz = 0
        if sz <= 0:
            return None
        with open(out_path, "rb") as vf:
            return vf.read()
    except Exception:
        return None
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        if out_path:
            try:
                os.unlink(out_path)
            except OSError:
                pass


def _video_build_annotated_slideshow_mp4(
    video_path: str,
    vp: list,
    ocr: "OCRTranslator",
    *,
    fps: float,
) -> Optional[bytes]:
    """Encode sampled OCR frames (with boxes + frame index) into an MP4 clip for ``st.video``.

    OpenCV ``mp4v`` is transcoded to **H.264 yuv420p + faststart** so HTML5 players work; if that
    fails, frames are piped directly to ffmpeg (same path as translated-video export).
    """
    if not video_path or not os.path.isfile(video_path) or not vp:
        return None
    fps = float(max(1.0, min(30.0, fps)))
    frames_bgr: list = []
    target_hw: Optional[tuple] = None
    for fi, rows in vp:
        fbgr = _video_read_frame_bgr_at(video_path, fi)
        if fbgr is None or fbgr.size == 0:
            continue
        if target_hw is None:
            target_hw = (int(fbgr.shape[0]), int(fbgr.shape[1]))
        else:
            th, tw = target_hw
            if int(fbgr.shape[0]) != th or int(fbgr.shape[1]) != tw:
                if tw > 0 and th > 0:
                    fbgr = cv2.resize(fbgr, (tw, th), interpolation=cv2.INTER_AREA)
        rows = _merge_video_adjacent_ocr_rows(list(rows or []), int(fbgr.shape[1]), int(fbgr.shape[0]))
        try:
            if rows:
                rgb = ocr.draw_bounding_boxes_on_frame(fbgr, rows)
                bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
            else:
                bgr = fbgr.copy()
        except Exception:
            bgr = fbgr.copy()
        frames_bgr.append(_video_overlay_frame_index_bgr(bgr, int(fi)))
    if not frames_bgr:
        return None
    th, tw = frames_bgr[0].shape[:2]
    tw, th = int(tw), int(th)

    tmp_raw: Optional[str] = None
    tmp_h264: Optional[str] = None
    try:
        fd, tmp_raw = tempfile.mkstemp(suffix=".mp4", prefix="annotate_slideshow_raw_")
        os.close(fd)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_raw, fourcc, fps, (tw, th))
        wrote_raw = False
        if writer.isOpened():
            try:
                for fr in frames_bgr:
                    writer.write(fr)
            finally:
                writer.release()
            try:
                wrote_raw = os.path.isfile(tmp_raw) and os.path.getsize(tmp_raw) > 0
            except OSError:
                wrote_raw = False
        else:
            try:
                writer.release()
            except Exception:
                pass

        if wrote_raw:
            fd2, tmp_h264 = tempfile.mkstemp(suffix=".mp4", prefix="annotate_slideshow_h264_")
            os.close(fd2)
            if _ocr_module._transcode_mp4_to_h264_for_browser(tmp_raw, tmp_h264):
                with open(tmp_h264, "rb") as vf:
                    return vf.read()

        return _video_annotate_slideshow_ffmpeg_pipe_to_mp4(frames_bgr, tw, th, fps=fps)
    except Exception:
        return _video_annotate_slideshow_ffmpeg_pipe_to_mp4(frames_bgr, tw, th, fps=fps)
    finally:
        for p in (tmp_h264, tmp_raw):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


def _render_video_per_frame_annotate_ui() -> None:
    """Annotated-tab UI for **From video**: slideshow + per-sample still (session state)."""
    _vp = st.session_state.get("video_per_frame_preview")
    _vpath = st.session_state.get("_video_retained_for_preview")
    if _vp and _vpath and os.path.isfile(_vpath):
        st.caption(
            "Browse frames where OCR ran during video sampling—the green boxes match "
            "detections used for translation (and for generated video, if enabled)."
        )
        _fps = float(st.session_state.get("_video_annotate_slideshow_fps", 6.0))
        _fps = max(2.0, min(12.0, _fps))
        _sig = _video_annotate_slideshow_cache_sig(_vpath, _vp, _fps)
        _cache = st.session_state.get("_video_annotate_slideshow_cache")
        _mpv: Optional[bytes] = None
        if isinstance(_cache, dict) and _cache.get("sig") == _sig and isinstance(_cache.get("bytes"), (bytes, bytearray)):
            _mpv = bytes(_cache["bytes"])
        else:
            with st.spinner("Building detection preview video…"):
                _ocr = st.session_state.get("ocr")
                if _ocr is not None:
                    _mpv = _video_build_annotated_slideshow_mp4(_vpath, _vp, _ocr, fps=_fps)
            if _mpv:
                st.session_state["_video_annotate_slideshow_cache"] = {"sig": _sig, "bytes": _mpv}
        if _mpv:
            st.markdown("**OCR samples as video** (same order as scanning; **Frame** index at top-left).")
            try:
                st.video(_mpv, format="video/mp4")
            except Exception:
                st.video(_mpv)
            st.slider(
                "Preview speed (frames per second)",
                min_value=2.0,
                max_value=12.0,
                value=float(_fps),
                step=1.0,
                key="_video_annotate_slideshow_fps",
                help="Changing this rebuilds the short preview clip on the next run.",
            )
            st.caption("Pause or scrub in the player, or pick a single sample below.")
        else:
            st.warning(
                "Could not build an inline preview video on this system (OpenCV encoder). "
                "Use the frame picker below for still previews."
            )

        _nonempty = [i for i, (_, rows) in enumerate(_vp) if rows and len(rows) > 0]
        _frame_pick_options = _nonempty if _nonempty else list(range(len(_vp)))
        _n_pick = len(_frame_pick_options)
        if _nonempty:
            st.caption(
                f"**{_n_pick}** sampled frame(s) with at least one detection "
                f"(out of **{len(_vp)}** OCR samples). Use the dropdown or **Previous** / **Next**."
            )
        st.session_state["_video_annotate_n_pick"] = _n_pick
        _pick_key = "video_annotate_frame_idx"
        if _pick_key in st.session_state:
            try:
                _pv_cur = int(st.session_state[_pick_key])
                if _pv_cur < 0 or _pv_cur >= _n_pick:
                    st.session_state[_pick_key] = max(0, min(_n_pick - 1, _pv_cur))
            except (TypeError, ValueError):
                st.session_state[_pick_key] = 0
        _choice_ix = st.selectbox(
            "Sampled frame (pick which OCR pass to preview)",
            options=list(range(_n_pick)),
            format_func=lambda j: (
                f"#{j + 1}/{_n_pick} — video frame {_vp[_frame_pick_options[j]][0]} "
                f"— {len(_vp[_frame_pick_options[j]][1])} detection(s)"
            ),
            key=_pick_key,
            label_visibility="visible",
        )
        _choice_ix = int(_choice_ix)
        _choice_ix = max(0, min(_n_pick - 1, _choice_ix))
        _cur_pick = _choice_ix
        _nav_l, _nav_m = st.columns(2)
        with _nav_l:
            st.button(
                "◀ Previous",
                disabled=_n_pick <= 1 or _cur_pick <= 0,
                key="video_annotate_prev",
                on_click=_video_annotate_prev_click,
            )
        with _nav_m:
            st.button(
                "Next ▶",
                disabled=_n_pick <= 1 or _cur_pick >= _n_pick - 1,
                key="video_annotate_next",
                on_click=_video_annotate_next_click,
            )
        _sel = _frame_pick_options[_choice_ix]
        _fi, _rows = _vp[_sel]
        _fbgr = _video_read_frame_bgr_at(_vpath, _fi)
        if _fbgr is None:
            st.warning("Could not decode this frame from the processed video file.")
        else:
            _rows = _merge_video_adjacent_ocr_rows(list(_rows or []), int(_fbgr.shape[1]), int(_fbgr.shape[0]))
            if not _rows:
                _rgb = cv2.cvtColor(_fbgr, cv2.COLOR_BGR2RGB)
                _safe_show_image(
                    _video_overlay_frame_index_rgb(_rgb, _fi),
                    caption=(
                        f"Frame index {_fi}: OCR returned no boxes on this sample "
                        "(text may appear on other sampled frames)."
                    ),
                    use_column_width=True,
                )
            else:
                try:
                    _ann_v = st.session_state.ocr.draw_bounding_boxes_on_frame(_fbgr, _rows)
                    _ann_v = _video_overlay_frame_index_rgb(_ann_v, _fi)
                    _safe_show_image(
                        _ann_v,
                        caption=f"Frame index {_fi} · OCR bounding boxes",
                        use_column_width=True,
                    )
                except Exception as _av_e:
                    st.warning(f"Could not draw detection overlay: {_av_e}")
    elif _vp and not (_vpath and os.path.isfile(_vpath)):
        st.info(
            "Per-frame OCR data is available, but the source video file was removed. "
            "Run **Translate** again after uploading to refresh previews."
        )
    else:
        st.info(
            "No per-frame OCR list is available yet. Run **Translate** on this video "
            "(the app will re-scan if an old cache had text only). Then open this tab again. "
            "If the video file was deleted from disk, re-upload and translate."
        )


def _normalize_to_rgb_uint8_array(img) -> np.ndarray:
    """Convert PIL/array-like image data to safe RGB uint8 ndarray."""
    if isinstance(img, Image.Image):
        arr = np.asarray(img.convert("RGB"))
    else:
        arr = np.asarray(img)
    if arr is None or arr.size == 0:
        raise ValueError("Empty image data.")
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _image_mode_sync_src_and_rotation(raw: bytes) -> None:
    """Reset image-mode rotation when the user uploads a new file or a new camera frame."""
    h = hashlib.sha256(raw).hexdigest()
    if st.session_state.get("image_mode_src_hash") != h:
        st.session_state["image_mode_src_hash"] = h
        st.session_state["image_mode_rotation"] = 0
        st.session_state.pop("image_ui_snapshot_v2", None)
        st.session_state.pop("image_ui_persist_active", None)
        # New source image: drop prior OCR instance so stale GPU allocations are released
        # before the next run/model load.
        _release_ocr_resources(reset_loaded_sig=False)


def _apply_image_mode_rotation(pil_image: Image.Image) -> Image.Image:
    """Apply clockwise session rotation (0/90/180/270) for image mode preview + OCR."""
    deg = int(st.session_state.get("image_mode_rotation", 0)) % 360
    if deg == 0:
        return pil_image
    return pil_image.rotate(
        -deg,
        expand=True,
        resample=Image.Resampling.BICUBIC,
    )


def _teesseract_osd_clockwise_fix_deg(pil_image: Image.Image) -> int:
    """
    If Tesseract + pytesseract are available, return clockwise rotation to apply so
    the page reads upright (0, 90, 180, or 270). Otherwise 0. Best on text-heavy images.
    """
    e = (os.environ.get("AUTO_IMAGE_TESSERACT_OSD") or "1").strip().lower()
    if e in ("0", "false", "no", "off"):
        return 0
    try:
        pytesseract = importlib.import_module("pytesseract")
    except ImportError:
        return 0
    w, h = pil_image.size
    if min(w, h) < 32:
        return 0
    max_d = 1200
    if max(w, h) > max_d:
        s = max_d / float(max(w, h))
        im = pil_image.resize((int(w * s), int(h * s)), Image.Resampling.BICUBIC)
    else:
        im = pil_image
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    try:
        osd = pytesseract.image_to_osd(im, config="--psm 0")
    except Exception:
        return 0
    m = re.search(r"Rotate:\s*(\d+)", osd, re.IGNORECASE)
    if not m:
        return 0
    deg = int(m.group(1)) % 360
    if deg not in (0, 90, 180, 270):
        return 0
    return deg


def _image_mode_auto_orient_pil(pil_image: Image.Image) -> Image.Image:
    """
    1) Camera/phone EXIF (Orientation) → pixel-correct view.
    2) Optional Tesseract OSD: extra deskew for upside-down or 90/270 when EXIF is missing
       and enough text/structure is present. Set AUTO_IMAGE_TESSERACT_OSD=0 to disable.
    """
    out = pil_image
    try:
        out = ImageOps.exif_transpose(out)
    except Exception:
        pass
    cw = _teesseract_osd_clockwise_fix_deg(out)
    if cw:
        out = out.rotate(-cw, expand=True, resample=Image.Resampling.BICUBIC)
    return out


def _release_ocr_resources(reset_loaded_sig: bool = True) -> None:
    """
    Best-effort OCR teardown: release model refs, stop Paddle daemon, and free CUDA cache.
    """
    old_ocr = st.session_state.get("ocr")
    st.session_state.ocr = None
    if reset_loaded_sig:
        st.session_state._ocr_loaded_langs = None

    if old_ocr is not None:
        for method_name in ("close", "shutdown", "dispose"):
            fn = getattr(old_ocr, method_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        del old_ocr

    try:
        _paddle_kill = getattr(_ocr_module, "_paddle_invalidate_daemon", None)
        if callable(_paddle_kill):
            _paddle_kill()
    except Exception:
        pass

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _mount_source_icons_inside_box():
    """Mount source tool buttons inside source textarea (bottom-left)."""
    import streamlit.components.v1 as components

    components.html(
        """
        <script>
        (function() {
          const doc = (window.parent && window.parent.document) ? window.parent.document : document;

          function mount() {
            const srcRoot = doc.querySelector(".st-key-source_text_input_v2");
            const ta = srcRoot ? srcRoot.querySelector("textarea") : null;
            const taHost = ta ? (ta.closest("div[data-baseweb='textarea']") || ta.parentElement) : null;
            const mic = doc.querySelector(".st-key-text_tool_mic");
            const scan = doc.querySelector(".st-key-text_tool_scan");
            const kb = doc.querySelector(".st-key-text_tool_write");
            if (!ta || !taHost || !mic || !scan || !kb) return false;
            taHost.style.position = "relative";
            taHost.style.overflow = "hidden";

            // Keep text clear of overlayed icon row.
            ta.style.paddingBottom = "64px";
            ta.style.paddingLeft = "12px";

            let toolbar = taHost.querySelector(".source-inline-toolbar");
            if (!toolbar) {
              toolbar = doc.createElement("div");
              toolbar.className = "source-inline-toolbar";
              toolbar.style.position = "absolute";
              toolbar.style.left = "10px";
              toolbar.style.bottom = "10px";
              toolbar.style.display = "flex";
              toolbar.style.gap = "8px";
              toolbar.style.zIndex = "120";
              taHost.appendChild(toolbar);
            }

            [mic, scan, kb].forEach((n) => {
              n.style.position = "static";
              n.style.margin = "0";
              n.style.height = "auto";
              n.style.width = "auto";
              n.style.pointerEvents = "auto";
              n.style.transform = "none";
              toolbar.appendChild(n);
            });
            return true;
          }

          // Re-try briefly because Streamlit can reflow widgets asynchronously.
          let tries = 0;
          const timer = setInterval(() => {
            tries += 1;
            if (mount() || tries > 18) clearInterval(timer);
          }, 80);
          mount();
        })();
        </script>
        """,
        height=0,
    )


# History persistence file (optional)
_HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ocr_studio_history.json")
_SAVED_TRANSLATIONS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".ocr_studio_saved_translations.json",
)

def _load_history():
    try:
        if os.path.isfile(_HISTORY_FILE):
            with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def _save_history(history):
    try:
        with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history[:50], f, ensure_ascii=False, indent=0)
    except Exception:
        pass


def _load_saved_translations():
    try:
        if os.path.isfile(_SAVED_TRANSLATIONS_FILE):
            with open(_SAVED_TRANSLATIONS_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)
                if isinstance(payload, list):
                    return payload
    except Exception:
        pass
    return []


def _save_saved_translations(saved_items):
    try:
        with open(_SAVED_TRANSLATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(saved_items[:100], f, ensure_ascii=False, indent=0)
    except Exception:
        pass


def _history_item_already_saved(item: dict, saved_items: list) -> bool:
    src = (item.get("source_lang") or "Auto").strip()
    tgt = (item.get("target_lang") or "").strip()
    original = (item.get("original_text") or item.get("preview") or "").strip()
    translated = (item.get("translated_text") or "").strip()
    for saved in saved_items or []:
        if (
            (saved.get("source_lang") or "Auto").strip() == src
            and (saved.get("target_lang") or "").strip() == tgt
            and (saved.get("original_text") or "").strip() == original
            and (saved.get("translated_text") or "").strip() == translated
        ):
            return True
    return False


def _find_saved_item_index(item: dict, saved_items: list) -> int:
    src = (item.get("source_lang") or "Auto").strip()
    tgt = (item.get("target_lang") or "").strip()
    original = (item.get("original_text") or item.get("preview") or "").strip()
    translated = (item.get("translated_text") or "").strip()
    for i, saved in enumerate(saved_items or []):
        if (
            (saved.get("source_lang") or "Auto").strip() == src
            and (saved.get("target_lang") or "").strip() == tgt
            and (saved.get("original_text") or "").strip() == original
            and (saved.get("translated_text") or "").strip() == translated
        ):
            return i
    return -1


def _upsert_saved_translation(item: dict, saved_items: list) -> None:
    """Insert/update one saved translation at the top, then persist."""
    idx = _find_saved_item_index(item, saved_items)
    if idx >= 0:
        existing = dict((saved_items[idx] or {}))
        existing.update(item or {})
        saved_items.pop(idx)
        saved_items.insert(0, existing)
    else:
        saved_items.insert(0, item)
    _save_saved_translations(saved_items)


def _is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "cuda out of memory" in msg
        or ("out of memory" in msg and "cuda" in msg)
        or "cublas_status_alloc_failed" in msg
    )


def _sanitize_photo_overlay_payload(results: list, translated_texts: list) -> tuple[list, list]:
    """Keep only OCR rows with usable 4-point boxes; normalize text to str."""
    clean_results = []
    clean_texts = []
    n = min(len(results or []), len(translated_texts or []))
    for i in range(n):
        row = results[i]
        txt = translated_texts[i]
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        bbox = row[1]
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 3:
            continue
        ok = True
        norm_pts = []
        for p in bbox:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                ok = False
                break
            try:
                x = float(p[0])
                y = float(p[1])
            except Exception:
                ok = False
                break
            norm_pts.append([x, y])
        if not ok:
            continue
        if len(row) >= 4:
            clean_results.append((str(row[0]), norm_pts, str(row[2]), float(row[3])))
        elif len(row) == 3:
            clean_results.append((str(row[0]), norm_pts, str(row[2])))
        else:
            clean_results.append((str(row[0]), norm_pts))
        clean_texts.append("" if txt is None else str(txt))
    return clean_results, clean_texts


def _normalize_ocr_rows(rows: list) -> list:
    """Canonicalize OCR rows to (text, bbox4+, lang, conf) and drop malformed entries."""
    out = []
    for row in rows or []:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        text = str(row[0]) if row[0] is not None else ""
        bbox = row[1]
        lang = str(row[2]) if len(row) > 2 and row[2] is not None else "en"
        conf_raw = row[3] if len(row) > 3 else 0.55
        try:
            conf = float(conf_raw)
        except Exception:
            conf = 0.55
        b = np.asarray(bbox, dtype=np.float64)
        if b.ndim == 1 and b.size >= 6:
            b = b.reshape(-1, 2)
        if b.ndim != 2 or b.shape[0] < 3 or b.shape[1] < 2:
            continue
        pts = []
        ok = True
        for i in range(b.shape[0]):
            try:
                x = float(b[i, 0])
                y = float(b[i, 1])
            except Exception:
                ok = False
                break
            pts.append([x, y])
        if not ok:
            continue
        out.append((text, pts, lang, conf))
    return out


def _merge_video_adjacent_ocr_rows(rows: list, img_w: int, img_h: int) -> list:
    """Normalize OCR rows then merge side-by-side / stacked boxes (same rules as translated video)."""
    r = _normalize_ocr_rows(rows)
    if len(r) < 2:
        return r
    try:
        mr, _mt = _ocr_module._merge_video_adjacent_boxes(
            list(r),
            [row[0] for row in r],
            int(img_w),
            int(img_h),
        )
        return mr
    except Exception:
        return r


def _as_text(value) -> str:
    """Coerce any value (including int/float) to safe display text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _pick_other_language(current_lang: Optional[str]) -> Optional[str]:
    choices = sorted(SUPPORTED_LANGUAGES.values())
    if not choices:
        return None
    for name in choices:
        if name != current_lang:
            return name
    return current_lang


def _ensure_distinct_text_languages(changed: str):
    src = st.session_state.get("source_lang_main")
    tgt = st.session_state.get("target_lang_main")
    if src is None or tgt is None or src != tgt:
        return
    if changed == "source":
        st.session_state["target_lang_main"] = _pick_other_language(src)
    else:
        st.session_state["source_lang_main"] = _pick_other_language(tgt)


def _swap_main_languages():
    src = st.session_state.get("source_lang_main")
    tgt = st.session_state.get("target_lang_main")
    if src is not None and tgt is not None:
        st.session_state["source_lang_main"] = tgt
        st.session_state["target_lang_main"] = src
        _ensure_distinct_text_languages("source")


def _activate_voice_input():
    st.session_state["text_input_method"] = "voice"


def _activate_write_input():
    st.session_state["text_input_method"] = "write"


def _activate_image_mode():
    st.session_state["main_mode_tier"] = "Image"


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _apply_runtime_cpu_mode(use_cpu: bool) -> None:
    """Force OCR + local NLLB translation onto CPU (Paddle worker inherits env)."""
    st.session_state._ocr_force_cpu = bool(use_cpu)
    if use_cpu:
        os.environ["OCR_USE_GPU"] = "0"
        os.environ["OCR_PADDLE_USE_CPU"] = "1"
        os.environ["NLLB_USE_CPU"] = "1"
        os.environ["TRANSLATE_FORCE_CPU"] = "1"
    else:
        for key in ("OCR_USE_GPU", "OCR_PADDLE_USE_CPU", "NLLB_USE_CPU", "TRANSLATE_FORCE_CPU"):
            os.environ.pop(key, None)


def _on_cpu_only_toggle() -> None:
    _apply_runtime_cpu_mode(bool(st.session_state.get("use_cpu_only")))
    _release_ocr_resources(reset_loaded_sig=True)
    st.session_state.translator = None


def _ensure_local_translator():
    if st.session_state.get("translation_provider", "google_cloud") != "local_nllb":
        return
    if st.session_state.translator is None:
        st.session_state.translator = TextTranslator(
            nllb_variant=st.session_state.get("_nllb_variant_locked", "600m"),
        )

# Initialize session state
if 'translator' not in st.session_state:
    st.session_state.translator = None
if 'ocr' not in st.session_state:
    st.session_state.ocr = None
if 'history' not in st.session_state:
    st.session_state.history = _load_history()
if '_ocr_initializing' not in st.session_state:
    st.session_state._ocr_initializing = False
if '_ocr_loaded_langs' not in st.session_state:
    st.session_state._ocr_loaded_langs = None
if "_ocr_force_cpu" not in st.session_state:
    st.session_state._ocr_force_cpu = _env_truthy("APP_USE_CPU", False)
if "use_cpu_only" not in st.session_state:
    st.session_state.use_cpu_only = bool(st.session_state._ocr_force_cpu)
_apply_runtime_cpu_mode(bool(st.session_state.use_cpu_only))
if 'ocr_backend' not in st.session_state:
    st.session_state.ocr_backend = "easyocr"
if 'trocr_model_id' not in st.session_state:
    st.session_state.trocr_model_id = ""
if 'hw_png_paths' not in st.session_state:
    st.session_state.hw_png_paths = None
if 'translation_provider' not in st.session_state:
    st.session_state.translation_provider = "google_cloud"
if 'ui_nav' not in st.session_state:
    st.session_state.ui_nav = 'translate'
if 'saved_translations' not in st.session_state:
    st.session_state.saved_translations = _load_saved_translations()
if 'last_ai_enhance_ctx' not in st.session_state:
    st.session_state.last_ai_enhance_ctx = None


def _iso6391_to_app(code: str) -> str:
    """Map a model-supplied ISO 639-1 tag to an app/SUPPORTED_LANGUAGES code."""
    c = (code or "en").strip().lower()
    if len(c) > 2 and c[2] in "-_":
        c = c.split("-", 1)[0].split("_", 1)[0]
    c = c[:2] if len(c) >= 2 else "en"
    return c if c in SUPPORTED_LANGUAGES else "en"


# Internal perf keys omitted from the UI (duplicates or sub-timers rolled into text translation).
_PERF_STAGE_HIDE = frozenset(
    {
        "text_translation_google_total",
        "text_translation_google_or_fallback",
        "text_translation_google_cloud",
        "google_api_google_wall_seconds",
        "google_api_fallback_wall_seconds",
    }
)

# Merge legacy / provider-specific stage names into one caption bucket.
_PERF_STAGE_CANONICAL = {
    "text_translation_google_cloud": "text_translation",
    "text_translation_local": "text_translation",
    "text_translation_openai": "text_translation",
    "text_translation_nllb_fallback": "text_translation",
    "google_api_segment_count": "text_translation_segment_count",
    "google_api_http_calls": "text_translation_api_calls",
}

_PERF_STAGE_LABELS = {
    "image_ocr_model_init": "OCR model init",
    "image_ocr_pipeline": "OCR",
    "video_ocr_models_ready": "OCR model init",
    "video_ocr_extract": "OCR",
    "video_prepare_translation": "prepare translation",
    "text_translation": "text translation",
    "text_translation_segment_count": "translation segments",
    "text_translation_api_calls": "translation API calls",
    "video_assemble_outputs": "assemble outputs",
    "video_render_export": "render export",
    "handwriting_ocr_plus_prefill_translation": "OCR + prefill",
    "handwriting_photo_generate": "photo generate",
}

_PERF_STAGE_COUNT_KEYS = frozenset(
    {"text_translation_segment_count", "text_translation_api_calls"}
)


def _perf_stage_label(key: str) -> str:
    return _PERF_STAGE_LABELS.get(key, key.replace("_", " "))


def _format_run_perf_caption(perf_stages: dict, total_elapsed: float) -> str:
    """Human-readable run timing line (never exposes provider names like Google)."""
    merged: dict[str, float] = {}
    for key, val in (perf_stages or {}).items():
        if key in _PERF_STAGE_HIDE or not isinstance(val, (int, float)):
            continue
        canon = _PERF_STAGE_CANONICAL.get(key, key)
        if canon in _PERF_STAGE_HIDE:
            continue
        merged[canon] = merged.get(canon, 0.0) + float(val)
    if not merged:
        return ""
    parts: list[str] = []
    for key, val in merged.items():
        label = _perf_stage_label(key)
        if key in _PERF_STAGE_COUNT_KEYS:
            parts.append(f"{label}: {int(round(val))}")
        else:
            parts.append(f"{label}: {val:.2f}s")
    return "Performance — " + " | ".join(parts) + f" | total: {total_elapsed:.2f}s"


def _fmt_ai_enhance_perf(d: dict) -> str:
    """Format Enhance-with-AI timing dict for ``st.caption``."""
    order = (
        ("temp_png_write_s", "temp_png_write"),
        ("normalize_image_s", "normalize"),
        ("prompt_build_s", "prompt"),
        ("prompt_chars", "prompt_chars"),
        ("openai_images_edit_s", "openai_api"),
        ("decode_response_image_s", "decode_img"),
        ("edit_photo_total_s", "edit_phase"),
        ("ui_wall_total_s", "ui_total"),
    )
    parts: list[str] = []
    for key, label in order:
        v = d.get(key)
        if v is None:
            continue
        if key == "prompt_chars":
            parts.append(f"{label}: {int(v)}")
            continue
        if isinstance(v, (int, float)) and v >= 0:
            parts.append(f"{label}: {float(v):.2f}s")
    if d.get("cache_hit") == 1:
        parts.append("cache: hit")
    return (
        "Performance — AI enhance: " + " | ".join(parts)
        if parts
        else ""
    )


def _resolve_openai_key() -> str:
    t = os.environ.get("OPENAI_API_KEY")
    if t:
        return t.strip()
    try:
        sec = getattr(st, "secrets", None)
        if sec is not None and sec.get("OPENAI_API_KEY"):
            return str(sec["OPENAI_API_KEY"]).strip()
    except Exception:
        pass
    return (st.session_state.get("openai_api_key_input") or "").strip()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _openai_image_options_for_quick_enhance() -> tuple[str, str, str, str]:
    """
    Options for the ✨ Enhance-with-AI button only.

    Starts from sidebar/session values. OPENAI_IMAGE_ENHANCE_FAST=1 picks a quicker
    preset (medium / 1024x1024 / low fidelity). Individual overrides:
    OPENAI_IMAGE_ENHANCE_QUALITY, OPENAI_IMAGE_ENHANCE_SIZE,
    OPENAI_IMAGE_ENHANCE_INPUT_FIDELITY, OPENAI_IMAGE_ENHANCE_MODEL.
    """
    q = st.session_state.get("openai_image_quality", "high")
    sz = st.session_state.get("openai_image_size", "auto")
    fid = st.session_state.get("openai_image_input_fidelity", "high")
    model = (st.session_state.get("openai_image_model") or "").strip() or default_image_model()

    if _env_flag("OPENAI_IMAGE_ENHANCE_FAST"):
        q, sz, fid = "medium", "1024x1024", "low"

    eq = os.environ.get("OPENAI_IMAGE_ENHANCE_QUALITY", "").strip()
    if eq:
        q = eq
    es = os.environ.get("OPENAI_IMAGE_ENHANCE_SIZE", "").strip()
    if es:
        sz = es
    ef = os.environ.get("OPENAI_IMAGE_ENHANCE_INPUT_FIDELITY", "").strip()
    if ef:
        fid = ef
    em = os.environ.get("OPENAI_IMAGE_ENHANCE_MODEL", "").strip()
    if em:
        model = em.strip()
    return q, sz, fid, model


def _generate_openai_translated_photo(
    image_path: str,
    pairs: list[tuple[str, str]],
    target_lang: str,
    perf_stats: Optional[dict] = None,
    *,
    snapshot_oai_key: Optional[str] = None,
    snapshot_image_opts: Optional[tuple[str, str, str, str]] = None,
    snapshot_compact_prompt: Optional[bool] = None,
) -> np.ndarray:
    """Generate translated photo using OpenAI Images API and return RGB ndarray."""
    oai_key = (
        snapshot_oai_key.strip()
        if snapshot_oai_key is not None
        else (_resolve_openai_key() or "").strip()
    )
    if not oai_key:
        raise ValueError(
            "OpenAI Images needs an API key. Set OPENAI_API_KEY in .env or paste it under photo options."
        )
    if not image_path or not os.path.isfile(image_path):
        raise ValueError("No valid image path available for AI enhancement.")
    if snapshot_image_opts is not None:
        q, sz, fid, img_model = snapshot_image_opts
    else:
        q, sz, fid, img_model = _openai_image_options_for_quick_enhance()
    if snapshot_compact_prompt is not None:
        compact_prompt = snapshot_compact_prompt
    else:
        compact_prompt = not _env_flag("OPENAI_IMAGE_ENHANCE_FULL_PROMPT")
    raw = edit_photo_with_translations(
        image_path,
        pairs,
        api_key=oai_key,
        model=img_model,
        quality=q,
        size=sz,
        input_fidelity=fid,
        target_lang_display=target_lang,
        perf_stats=perf_stats,
        compact_prompt=compact_prompt,
    )
    t_dec = time.perf_counter()
    arr = image_bytes_to_rgb_numpy(raw)
    if perf_stats is not None:
        perf_stats["decode_response_image_s"] = time.perf_counter() - t_dec
    return arr


def _generate_openai_translated_photo_from_bytes(
    image_bytes: bytes,
    pairs: list[tuple[str, str]],
    target_lang: str,
    perf_stats: Optional[dict] = None,
    *,
    snapshot_oai_key: Optional[str] = None,
    snapshot_image_opts: Optional[tuple[str, str, str, str]] = None,
    snapshot_compact_prompt: Optional[bool] = None,
) -> np.ndarray:
    if not image_bytes:
        raise ValueError("No source image bytes available for AI enhancement.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        t_w = time.perf_counter()
        tmp_file.write(image_bytes)
        if perf_stats is not None:
            perf_stats["temp_png_write_s"] = time.perf_counter() - t_w
        tmp_path = tmp_file.name
    try:
        return _generate_openai_translated_photo(
            tmp_path,
            pairs,
            target_lang,
            perf_stats=perf_stats,
            snapshot_oai_key=snapshot_oai_key,
            snapshot_image_opts=snapshot_image_opts,
            snapshot_compact_prompt=snapshot_compact_prompt,
        )
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _enhance_background_job(
    args: tuple[
        bytes,
        list[tuple[str, str]],
        str,
        str,
        tuple[str, str, str, str],
        bool,
    ],
) -> tuple[np.ndarray, dict]:
    """Run ✨ OpenAI enhance off-thread; returns RGB array and perf stats dict."""
    image_bytes, pairs, target_lang, oai_key, img_opts, compact = args
    perf: dict = {}
    arr = _generate_openai_translated_photo_from_bytes(
        image_bytes,
        pairs,
        target_lang,
        perf_stats=perf,
        snapshot_oai_key=oai_key,
        snapshot_image_opts=img_opts,
        snapshot_compact_prompt=compact,
    )
    return arr, perf


# Max cached OCR payloads (image + video raw results); video entries can be large
_OCR_CACHE_MAX_ENTRIES = 4


def _sha256_file(path: str) -> str:
    """Stable hash of file contents for OCR cache keys."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ocr_preprocess_env_fingerprint() -> str:
    """Env knobs that change OCR preprocessing — part of OCR cache keys."""
    keys = (
        "OCR_VIDEO_PREPROCESS_MAX_SIDE",
        "OCR_PREPROCESS_ENGINE",
        "OCR_PREPROCESS_FORCE_GPU",
        "OCR_PREPROCESS_DEBUG",
        "OCR_PREPROCESS_FORCE_CPU",
        "OCR_PREPROCESS_PERSPECTIVE",
        "OCR_PREPROCESS_GPU_ILLUM_SIGMA",
        "OCR_PREPROCESS_GPU_ILLUM_STRENGTH",
        "OCR_PREPROCESS_GPU_ILLUM_CORNER_PULL",
        "OCR_PREPROCESS_GPU_ILLUM_SUBTLE",
        "OCR_PREPROCESS_GPU_BORDER_LIFT",
        "OCR_PREPROCESS_GPU_LOCAL_EDGE_WEIGHT",
        "OCR_PREPROCESS_GPU_LOCAL_EDGE_CAP",
        "OCR_PREPROCESS_GPU_DETAIL_EDGE_CAP",
        "OCR_PREPROCESS_GPU_MICRO_LUMA",
        "OCR_PREPROCESS_GPU_MICRO_LUMA_SIGMA",
        "OCR_PREPROCESS_GPU_THIN_TEXT_UNSHARP",
        "OCR_PREPROCESS_GPU_UNSHARP_THIN_SIGMA_SCALE",
        "OCR_PREPROCESS_GPU_GAMMA",
        "OCR_PREPROCESS_GPU_GLOBAL_CONTRAST",
        "OCR_PREPROCESS_GPU_LOCAL_CONTRAST",
        "OCR_PREPROCESS_GPU_LOCAL_SIGMA",
        "OCR_PREPROCESS_GPU_DETAIL_BOOST",
        "OCR_PREPROCESS_GPU_CPU_CLAHE",
        "OCR_PREPROCESS_GPU_MEDIAN_L",
        "OCR_PREPROCESS_RIDNET_JIT",
        "OCR_PREPROCESS_RIDNET_JIT_MIX",
        "OCR_PREPROCESS_RIDNET",
        "OCR_PREPROCESS_RIDNET_WEIGHTS",
        "OCR_PREPROCESS_RIDNET_MIX",
        "OCR_PREPROCESS_GPU_MAX_MEGAPIXELS",
        "OCR_PREPROCESS_GPU_ADAPTIVE_CLAHE",
        "OCR_PREPROCESS_GPU_CLAHE_BLEND",
        "OCR_PREPROCESS_GPU_MICRO_UNSHARP",
        "OCR_PREPROCESS_GPU_FP16",
        "OCR_PREPROCESS_GPU_EMPTY_CACHE",
        "OCR_PREPROCESS_BRIGHTEN",
        "OCR_PREPROCESS_DNN",
        "OCR_PREPROCESS_DNN_DEVICE",
        "OCR_PREPROCESS_DNN_WEIGHTS",
        "OCR_PREPROCESS_DNN_REPO",
        "OCR_PREPROCESS_DNN_FILE",
        "OCR_PREPROCESS_DENOISE",
        "OCR_PREPROCESS_DENOISE_GENTLE",
        "OCR_PREPROCESS_BILATERAL",
        "OCR_PREPROCESS_BILATERAL_D",
        "OCR_PREPROCESS_BILATERAL_SIGMA_COLOR",
        "OCR_PREPROCESS_BILATERAL_SIGMA_SPACE",
        "OCR_PREPROCESS_BILATERAL_PASSES",
        "OCR_PREPROCESS_MEDIAN_L",
        "OCR_PREPROCESS_MEDIAN_L_K",
        "OCR_PREPROCESS_CLAHE",
        "OCR_PREPROCESS_CLAHE_AUTO",
        "OCR_PREPROCESS_CLAHE_MAX_STD",
        "OCR_PREPROCESS_CLAHE_CLIP",
        "OCR_PREPROCESS_CLAHE_TILE",
        "OCR_PREPROCESS_LUMA_STRETCH",
        "OCR_PREPROCESS_LUMA_STRETCH_BLEND",
        "OCR_PREPROCESS_LUMA_STRETCH_LOW_PCT",
        "OCR_PREPROCESS_LUMA_STRETCH_HIGH_PCT",
        "OCR_PREPROCESS_EDGE_RESTORE",
        "OCR_PREPROCESS_EDGE_RESTORE_FLOOR",
        "OCR_PREPROCESS_ADAPTIVE_UNSHARP",
        "OCR_PREPROCESS_UNSHARP",
        "OCR_PREPROCESS_UNSHARP_SIGMA",
        "OCR_MIN_CONFIDENCE",
        "OCR_HIGH_RECALL_MIN_CONFIDENCE",
        "OCR_CONFIDENCE_SOFT_FLOOR",
        "OCR_HIGH_RECALL_SOFT_FLOOR",
        "OCR_EMPTY_RESCUE_MIN_CONF",
        "OCR_MERGE_PASSES_OFF",
        "OCR_PREPROCESS_OPENAI",
        "OCR_PREPROCESS_OPENAI_ONLY",
        "OCR_PREPROCESS_OPENAI_QUALITY",
        "OCR_PREPROCESS_OPENAI_SIZE",
        "OCR_PREPROCESS_OPENAI_MODEL",
    )
    default_version = "preprocess_defaults_v7_openai_ocr"
    return default_version + "|" + "|".join(f"{k}={(os.environ.get(k) or '').strip()}"[:160] for k in keys)


def _ensure_ocr_cache_store():
    if "_ocr_cache_store" not in st.session_state:
        st.session_state._ocr_cache_store = OrderedDict()


def _ocr_cache_get(key: tuple):
    """Return a deep copy of cached OCR payload or None."""
    _ensure_ocr_cache_store()
    od = st.session_state._ocr_cache_store
    if key not in od:
        return None
    od.move_to_end(key)
    return copy.deepcopy(od[key])


def _ocr_cache_set(key: tuple, payload: dict):
    """Store deep copy; evict oldest when over capacity."""
    _ensure_ocr_cache_store()
    od = st.session_state._ocr_cache_store
    od[key] = copy.deepcopy(payload)
    od.move_to_end(key)
    while len(od) > _OCR_CACHE_MAX_ENTRIES:
        od.popitem(last=False)


def _ocr_cache_pop(key: tuple) -> None:
    """Drop one OCR cache entry (e.g. legacy video aggregate without per-frame data)."""
    _ensure_ocr_cache_store()
    st.session_state._ocr_cache_store.pop(key, None)


def _html_escape(s: str) -> str:
    """Escape text for safe use inside HTML."""
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def _paragraph_join(parts: list) -> str:
    """Join OCR / translation segments with newlines (one segment per line)."""
    chunks = []
    for p in parts:
        if p is None:
            continue
        t = " ".join(str(p).split())
        if t:
            chunks.append(t)
    return "\n".join(chunks)


def _dedupe_key(t: str) -> str:
    """Normalize for matching duplicate OCR lines across video frames."""
    return " ".join(str(t).split())


def _loose_text_key(s: str) -> str:
    """Aggressive normalize (strip punctuation) so OCR variants merge before translate / display."""
    t = " ".join(str(s).lower().split())
    t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)
    return " ".join(t.split())


def _translation_dedupe_key(mode: str, t: str) -> str:
    """
    Key for merging duplicate lines before translation (unique_texts / translation map).

    **From video**: apply Unicode NFKC first so full-width punctuation and compatibility forms
    match plain ASCII across frames — fewer unique strings → faster Cloud Translation.
    """
    s = str(t or "").strip()
    if mode == "From video":
        s = unicodedata.normalize("NFKC", s)
    dk = _loose_text_key(s)
    if len(dk) < 2:
        dk = _dedupe_key(s).lower()
    return dk


# Script buckets for NLLB source inference (must match ocr_lang_options values)
_NON_LATIN_OCR_LANGS = frozenset({"ar", "zh", "ja", "ko", "ru", "hi"})
# Prefer real menu languages before English when several Latin OCR langs are enabled.
_LATIN_SOURCE_PRIORITY = ("de", "fr", "es", "it", "pt", "nl", "pl", "tr", "en")


def _infer_translation_source_lang(
    text: str,
    ocr_languages: list,
    fallback: str,
    target_lang: Optional[str] = None,
) -> str:
    """
    NLLB source language for one OCR segment. Uses EasyOCR's per-box `fallback` when it is in the
    enabled list (e.g. German). Previously, Latin text + English in OCR always used English, so
    German/French menus were translated en→ar and Arabic came out wrong.
    """
    ocr_set = set(ocr_languages or ["en"])
    fb = (fallback or "en").lower().strip()
    tgt = (target_lang or "").lower().strip()
    s = (text or "").strip()
    if not s:
        for c in _LATIN_SOURCE_PRIORITY:
            if c in ocr_set and c != tgt:
                return c
        return fb if fb in ocr_set else (list(ocr_set)[0] if ocr_set else "en")

    def is_arabic_char(c: str) -> bool:
        o = ord(c)
        if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF:
            return True
        # Presentation forms / compatibility (common in fonts and some OCR outputs)
        if 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF:
            return True
        return False

    arabic_chars = sum(1 for c in s if is_arabic_char(c))
    cyrillic_chars = sum(1 for c in s if "\u0400" <= c <= "\u04FF")
    latin_chars = sum(1 for c in s if ("A" <= c <= "Z") or ("a" <= c <= "z"))
    has_cjk = any(
        ("\u4e00" <= c <= "\u9fff") or ("\u3040" <= c <= "\u30ff") or ("\uac00" <= c <= "\ud7af")
        for c in s
    )

    # Latin-only line but EasyOCR tag is 'ar' (Arabic model listed first) — do not use ar as NLLB source
    if latin_chars > 0 and arabic_chars == 0 and fb == "ar":
        fb = next((c for c in _LATIN_SOURCE_PRIORITY if c in ocr_set), "en")

    if arabic_chars > 0 and "ar" in ocr_set:
        # Default rule: Arabic dominates by count or by share of the string.
        # Extra rule: bilingual signage often mixes Latin labels ("QR", "Code") inside an Arabic
        # sentence — if Arabic letters are not fewer than Latin letters, translate as Arabic
        # (ar→target). Otherwise Latin-heavy lines were sent as en/de→target and Arabic was left
        # untranslated or mangled.
        ratio = arabic_chars / max(len(s), 1)
        if (
            arabic_chars >= max(3, latin_chars + 1)
            or ratio > 0.38
            or (arabic_chars >= 4 and arabic_chars >= latin_chars)
        ):
            return "ar"
    if cyrillic_chars > 0 and "ru" in ocr_set:
        return "ru"
    if has_cjk and "zh" in ocr_set:
        return "zh"
    # Arabic line but OCR language list omitted "ar" (e.g. only en) — still use NLLB Arabic source
    # so text is not sent as en→de (model often copies Arabic through untranslated).
    if tgt and tgt not in ("ar", "fa", "ur", "ps", "ug", "sd", "ckb") and arabic_chars >= 2:
        return "ar"

    if latin_chars > 0:
        if fb in ocr_set and fb != tgt and fb not in _NON_LATIN_OCR_LANGS:
            return fb
        for code in _LATIN_SOURCE_PRIORITY:
            if code in ocr_set and code != tgt:
                return code
        for code in ocr_set:
            if code not in _NON_LATIN_OCR_LANGS and code != tgt:
                return code

    # Latin-only lines (e.g. English trade names on a German-only OCR list): if every Latin-script OCR
    # language equals the target, NLLB would use de→de / en→en and skip translation. Use English as
    # source so loanwords still translate (en→de, en→fr, …).
    if (
        latin_chars > 0
        and arabic_chars == 0
        and not cyrillic_chars
        and not has_cjk
        and tgt
    ):
        _latin_ocr = [c for c in ocr_set if c not in _NON_LATIN_OCR_LANGS]
        if _latin_ocr and all(c.lower() == tgt.lower() for c in _latin_ocr):
            guess = _heuristic_latin_source_when_src_equals_tgt(s, tgt, tgt)
            if guess:
                return guess
            return "en"

    if fb in ocr_set and fb != tgt:
        return fb
    return list(ocr_set)[0] if ocr_set else "en"


# When translating *into* these codes, we still run MT for Arabic (e.g. ar→ar identity, or Arabic→Farsi).
# For other targets (en, de, es, …), very short all-Arabic strings can be left as-is (names, labels).
_PRESERVE_ARABIC_DO_NOT_BYPASS: frozenset[str] = frozenset(
    {"ar", "fa", "ur", "ps", "ug", "he", "sd", "ckb", "ku", "dv"}
)


def _count_arabic_letters(s: str) -> int:
    n = 0
    for c in s:
        o = ord(c)
        if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF:
            n += 1
        elif 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF:
            n += 1
    return n


def _is_short_mixed_arabic_preserve_block(text: str) -> bool:
    """Heuristic: a short **mixed** Latin + Arabic label (e.g. brand + Arabic name), not plain Arabic copy."""
    t = (text or "").strip()
    if not t or len(t) > 68:
        return False
    if any(c.isdigit() for c in t):
        return False
    if "http" in t.lower() or "www." in t.lower() or "@" in t:
        return False
    if "\n" in t or "\r" in t:
        return False
    ar = _count_arabic_letters(t)
    if ar < 2:
        return False
    lat = sum(1 for c in t if "A" <= c <= "Z" or "a" <= c <= "z")
    # Pure Arabic lines are normal signage — they must go through MT (ar→de, etc.). Preserve only
    # when Latin is present (mixed token / logo line) so we do not skip real sentences/headings.
    if lat < 1:
        return False
    if lat > 0 and 2 * lat >= ar:
        return False
    alpha = sum(1 for c in t if c.isalpha())
    if alpha and (ar / max(alpha, 1)) < 0.70:
        return False
    words = t.split()
    if not words or len(words) > 2:
        return False
    for w in words:
        if _count_arabic_letters(w) < 1:
            return False
        if len(w) > 24:
            return False
    return True


def _should_preserve_arabic_without_translation(text: str, target_code: str) -> bool:
    if (target_code or "").lower().strip() in _PRESERVE_ARABIC_DO_NOT_BYPASS:
        return False
    return _is_short_mixed_arabic_preserve_block(text)


def _translation_bypass_preserve_dict(original: str, target_code: str) -> dict:
    return {
        "original_text": original,
        "translated_text": original,
        "source_language": "ar",
        "target_language": target_code,
        "confidence": 1.0,
    }


def _translation_try_preserve_arabic(
    text: str, target_code: str, user_enabled: bool
) -> Optional[dict]:
    if not user_enabled:
        return None
    if os.environ.get("TRANSLATION_PRESERVE_ARABIC_NAMES_OFF", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return None
    if not _should_preserve_arabic_without_translation(text, target_code):
        return None
    return _translation_bypass_preserve_dict(text, target_code)


# EasyOCR: if any Arabic-script model (ar/fa/ur/ug) is used, the Reader may only load those plus "en".
_EASYOCR_ARABIC_SCRIPT: frozenset[str] = frozenset({"ar", "fa", "ur", "ug"})
_EASYOCR_ARABIC_COMPAT: frozenset[str] = frozenset({"ar", "fa", "ur", "ug", "en"})
_EASYOCR_CHINESE_SCRIPT: frozenset[str] = frozenset({"ch_sim", "ch_tra"})
_EASYOCR_CHINESE_COMPAT: frozenset[str] = frozenset({"ch_sim", "ch_tra", "en"})
_EASYOCR_KOREAN_SCRIPT: frozenset[str] = frozenset({"ko"})
_EASYOCR_KOREAN_COMPAT: frozenset[str] = frozenset({"ko", "en"})


def _easyocr_reader_language_order(codes: list) -> list:
    """
    Order langs for easyocr.Reader. If ar/fa/ur/ug is selected, drop other codes (e.g. de, es) —
    EasyOCR does not allow ar+de in one reader (ValueError). Keep only ar/fa/ur/ug/en.
    """
    _easyocr_alias = {
        "zh": "ch_sim",  # EasyOCR expects Chinese simplified as ch_sim.
    }
    seq = [_easyocr_alias.get(c, c) for c in (codes or []) if c]
    if not seq:
        return ["en"]
    if set(seq) & _EASYOCR_ARABIC_SCRIPT:
        filtered = [c for c in seq if c in _EASYOCR_ARABIC_COMPAT]
        if not filtered:
            return ["en"]
        if "en" not in filtered:
            filtered = filtered + ["en"]
        if "ar" in filtered:
            return ["ar"] + [c for c in filtered if c != "ar"]
        return filtered
    if set(seq) & _EASYOCR_CHINESE_SCRIPT:
        filtered = [c for c in seq if c in _EASYOCR_CHINESE_COMPAT]
        if not filtered:
            return ["en"]
        if "en" not in filtered:
            filtered = filtered + ["en"]
        if "ch_sim" in filtered:
            return ["ch_sim"] + [c for c in filtered if c != "ch_sim"]
        return filtered
    if set(seq) & _EASYOCR_KOREAN_SCRIPT:
        filtered = [c for c in seq if c in _EASYOCR_KOREAN_COMPAT]
        if not filtered:
            return ["en"]
        if "en" not in filtered:
            filtered = filtered + ["en"]
        if "ko" in filtered:
            return ["ko"] + [c for c in filtered if c != "ko"]
        return filtered
    return seq


def _ocr_backend_reader_languages(codes: list, backend: str) -> list:
    """Return OCR languages transformed for the selected backend."""
    b = (backend or "easyocr").strip().lower()
    if b in ("easyocr", "trocr_hybrid"):
        return _easyocr_reader_language_order(codes)
    return [c for c in (codes or ["en"]) if c] or ["en"]


def _paragraph_join_deduped(parts: list) -> str:
    """Join segments with newlines; skip lines that are duplicates after loose normalize (OCR repeats + typos)."""
    seen: set = set()
    chunks = []
    for p in parts:
        if p is None:
            continue
        t = " ".join(str(p).split())
        if not t:
            continue
        k = _loose_text_key(t)
        if len(k) < 2:
            k = _dedupe_key(t).lower()
        if k in seen:
            continue
        seen.add(k)
        chunks.append(t)
    return "\n".join(chunks)


def _text_count_stats(text: str) -> dict:
    """Word, character, and sentence counts for JSON export (original or translated body)."""
    t = (text or "").strip()
    if not t:
        return {"word_count": 0, "character_count": 0, "sentence_count": 0}
    character_count = len(t)
    word_count = len(re.findall(r"\w+", t, flags=re.UNICODE))
    parts = re.split(r"[.!?؟۔]+", t)
    sentence_count = sum(1 for p in parts if p.strip())
    if sentence_count == 0:
        sentence_count = 1
    return {
        "word_count": word_count,
        "character_count": character_count,
        "sentence_count": sentence_count,
    }


def _sort_ocr_results_reading_order(ocr_results: list, rtl: bool = False) -> list:
    """Sort OCR results so boxes are in reading order: top-to-bottom, then left-to-right (or right-to-left for RTL)."""
    if not ocr_results:
        return ocr_results

    # Use typical box height so "same line" works for any resolution (image or video frame)
    heights = []
    for row in ocr_results:
        bbox = row[1]
        pts = np.array(bbox, dtype=float)
        h = float(pts[:, 1].max() - pts[:, 1].min())
        if h > 0:
            heights.append(h)
    line_tolerance = max(15, int(np.median(heights))) if heights else 25

    def sort_key(item):
        text, bbox, lang = item[0], item[1], item[2]
        pts = np.array(bbox, dtype=float)
        y_center = float(pts[:, 1].mean())
        x_center = float(pts[:, 0].mean())
        line_group = int(round(y_center / line_tolerance)) * line_tolerance
        if rtl:
            return (line_group, -x_center)  # right-to-left per line
        return (line_group, x_center)  # left-to-right per line

    return sorted(ocr_results, key=sort_key)


def _drop_container_ocr_boxes(ocr_results: list) -> list:
    """
    Remove oversized boxes that mostly contain multiple tighter OCR boxes.
    This targets occasional EasyOCR "container" detections around menu rows.
    """
    if not ocr_results or len(ocr_results) < 3:
        return ocr_results
    aabbs = []
    areas = []
    for row in ocr_results:
        pts = np.array(row[1], dtype=float)
        x1 = float(pts[:, 0].min())
        y1 = float(pts[:, 1].min())
        x2 = float(pts[:, 0].max())
        y2 = float(pts[:, 1].max())
        aabbs.append((x1, y1, x2, y2))
        areas.append(max(1.0, (x2 - x1) * (y2 - y1)))

    drop_idx = set()
    for i, ai in enumerate(aabbs):
        inside_count = 0
        for j, aj in enumerate(aabbs):
            if i == j:
                continue
            inter_w = max(0.0, min(ai[2], aj[2]) - max(ai[0], aj[0]))
            inter_h = max(0.0, min(ai[3], aj[3]) - max(ai[1], aj[1]))
            inter = inter_w * inter_h
            if inter <= 0.0:
                continue
            contained_ratio = inter / areas[j]
            if contained_ratio >= 0.84 and areas[i] >= areas[j] * 2.6:
                inside_count += 1
        if inside_count >= 2:
            drop_idx.add(i)
    if not drop_idx:
        return ocr_results
    return [row for idx, row in enumerate(ocr_results) if idx not in drop_idx]


def _reorder_segments_for_rtl_boxes(ocr_results: list, segments: list) -> list:
    """Reorder segments so the first segment goes in the rightmost box, etc. (for Arabic RTL)."""
    if not ocr_results or len(ocr_results) != len(segments):
        return segments
    # bbox is list of [x,y] points; use center x for horizontal position
    def center_x(item):
        bbox = item[1]
        pts = np.array(bbox, dtype=float)
        return float(pts[:, 0].mean())
    # Rightmost box first (largest x)
    indices_by_x = sorted(range(len(ocr_results)), key=lambda i: center_x(ocr_results[i]), reverse=True)
    rank_of = {idx: rank for rank, idx in enumerate(indices_by_x)}
    return [segments[rank_of[i]] for i in range(len(ocr_results))]


def _split_translation_to_lines(full_translation: str, original_lines: list, equal_split: bool = False) -> list:
    """Split one translated string back into N lines. equal_split=True gives equal words per line (better for RTL)."""
    if not original_lines or not full_translation.strip():
        return [full_translation] if original_lines else []
    if len(original_lines) == 1:
        return [full_translation.strip()]
    words = full_translation.split()
    if not words:
        return [full_translation.strip()] * len(original_lines)
    n = len(original_lines)
    parts = []
    if equal_split:
        # Equal word count per line (more reliable for RTL like Arabic)
        base, extra = divmod(len(words), n)
        w_start = 0
        for i in range(n):
            count = base + (1 if i < extra else 0)
            w_end = min(w_start + count, len(words))
            parts.append(" ".join(words[w_start:w_end]).strip())
            w_start = w_end
    else:
        total_orig = sum(max(1, len(line)) for line in original_lines)
        w_start = 0
        for i, line in enumerate(original_lines):
            if i == len(original_lines) - 1:
                parts.append(" ".join(words[w_start:]).strip())
                break
            frac = max(1, len(line)) / total_orig
            n_words = max(1, min(int(round(frac * len(words))), len(words) - w_start - 1))
            w_end = w_start + n_words
            parts.append(" ".join(words[w_start:w_end]).strip())
            w_start = w_end
    return parts


def initialize_models(ocr_languages, ocr_backend: str = "easyocr", trocr_model: Optional[str] = None):
    """Load OCR for the selected language list; reload when OCR languages change."""
    if not ocr_languages:
        ocr_languages = ["en"]
    backend_key = (ocr_backend or "easyocr").strip().lower()
    langs_tuple = tuple(sorted(ocr_languages))
    trocr_key = (trocr_model or "").strip() if backend_key == "trocr_hybrid" else ""
    loaded_sig = (backend_key, langs_tuple, trocr_key)
    if st.session_state.get("_ocr_loaded_langs") != loaded_sig:
        _release_ocr_resources(reset_loaded_sig=False)

    if st.session_state.ocr is None and not st.session_state._ocr_initializing:
        st.session_state._ocr_initializing = True
        try:
            reader_langs = _ocr_backend_reader_languages(ocr_languages, backend_key)
            lang_label = ", ".join(reader_langs)
            with st.spinner(f"Loading OCR reader ({backend_key}: {lang_label})…"):
                def _ocr_kw(force_cpu: bool) -> dict:
                    d = dict(
                        languages=reader_langs,
                        force_cpu=force_cpu,
                        backend=backend_key,
                    )
                    tm = (trocr_model or "").strip()
                    if backend_key == "trocr_hybrid" and tm:
                        d["trocr_model"] = tm
                    return d

                try:
                    st.session_state.ocr = OCRTranslator(**_ocr_kw(bool(st.session_state.get("_ocr_force_cpu"))))
                except Exception as e:
                    msg = str(e).lower()
                    if ("cuda out of memory" in msg or "out of memory" in msg) and not st.session_state.get("_ocr_force_cpu"):
                        st.session_state._ocr_force_cpu = True
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        st.warning("CUDA memory is full. Switching OCR to CPU mode for this session.")
                        st.session_state.ocr = OCRTranslator(**_ocr_kw(True))
                    elif backend_key == "paddleocr":
                        st.warning(
                            "PaddleOCR failed to initialize. Falling back to EasyOCR for stability. "
                            "Install Paddle with `pip install paddleocr paddlepaddle` if needed."
                        )
                        fallback_langs = _easyocr_reader_language_order(ocr_languages)
                        st.session_state.ocr_backend = "easyocr"
                        st.session_state.ocr = OCRTranslator(
                            languages=fallback_langs,
                            force_cpu=bool(st.session_state.get("_ocr_force_cpu")),
                            backend="easyocr",
                        )
                        backend_key = "easyocr"
                    else:
                        raise
                st.session_state._ocr_loaded_langs = (
                    backend_key,
                    langs_tuple,
                    trocr_key if backend_key == "trocr_hybrid" else "",
                )
        finally:
            st.session_state._ocr_initializing = False
    # Ensure translator object exists for local mode.
    _ensure_local_translator()


def _video_annotate_prev_click() -> None:
    """Used as st.button(on_click=…): runs before widget reconciliation so the frame index is not reset by the selectbox."""
    _pk = "video_annotate_frame_idx"
    cur = int(st.session_state.get(_pk, 0))
    st.session_state[_pk] = max(0, cur - 1)


def _video_annotate_next_click() -> None:
    _pk = "video_annotate_frame_idx"
    n = max(1, int(st.session_state.get("_video_annotate_n_pick", 1)))
    cur = int(st.session_state.get(_pk, 0))
    st.session_state[_pk] = min(n - 1, cur + 1)


def _ocr_preprocess_zoom_toggle_click() -> None:
    """Streamlit button on_click: toggles large preprocessing before/after view."""
    st.session_state["ocr_preprocess_zoom_large"] = not bool(st.session_state.get("ocr_preprocess_zoom_large"))


def _replay_video_translation_ui_video(ui: dict, *, text_output_placeholder) -> None:
    """Render post-OCR UI for **From video** only. Used after Translate and on widget reruns."""
    mode = "From video"
    detected_texts = ui["detected_texts"]
    detected_languages = ui["detected_languages"]
    translations = ui["translations"]
    results_json = ui["results_json"]
    target_lang = ui["target_lang"]
    target_lang_code = ui["target_lang_code"]
    input_lang = ui.get("input_lang") or "en"
    _run_t0 = float(ui.get("_run_t0", 0.0))
    _perf_stages = ui.get("_perf_stages") or {}
    generate_video = bool(ui.get("generate_video"))
    translated_video_path = ui.get("translated_video_path")
    translated_video_format = ui.get("translated_video_format") or "video/mp4"
    translated_video_ext = ui.get("translated_video_ext") or "mp4"

    st.success(f"Detected {len(detected_texts)} segment(s)")

    total_elapsed = time.perf_counter() - _run_t0
    _perf_caption = _format_run_perf_caption(_perf_stages, total_elapsed)
    if _perf_caption:
        st.caption(_perf_caption)

    if detected_languages:
        unique_langs = ", ".join(sorted(set(detected_languages)))
        st.caption(f"Source: {unique_langs}")

    translated_video_bytes = None
    if translated_video_path and os.path.isfile(translated_video_path):
        try:
            with open(translated_video_path, "rb") as vf:
                translated_video_bytes = vf.read()
        except OSError:
            translated_video_bytes = None

    if translated_video_path and os.path.isfile(translated_video_path):
        st.markdown("### Translated video")
        st.caption("Play below in the app, or download the file.")
        if translated_video_ext == "avi":
            st.caption(
                "Your system saved **AVI** (MP4 was unavailable). Many browsers cannot play AVI inline — "
                "use **Download translated video** and open it in **VLC** or the **Movies & TV** app if the player stays blank."
            )
        try:
            st.video(translated_video_path)
        except Exception:
            if translated_video_bytes:
                st.video(translated_video_bytes, format=translated_video_format)
            else:
                st.warning("Could not embed the video player; use download below.")
        if translated_video_bytes:
            st.download_button(
                label="Download translated video",
                data=translated_video_bytes,
                file_name=f"translated_video.{translated_video_ext}",
                mime=translated_video_format,
                key="dl_translated_video_main",
                use_container_width=True,
            )
        st.markdown("---")
    elif generate_video and not translated_video_path:
        st.info(
            "**Generate video with translated text** was on, but no video file was produced. "
            "Check that text was detected and try again. If it keeps failing, see the warning or error above."
        )

    _pj = _paragraph_join_deduped
    translated_block = _pj([t["translated_text"] for t in translations])

    tab_list = ["Text", "Annotated", "Processing", "Translated video", "JSON"]
    (
        translations_tab,
        annotated_tab,
        processing_tab,
        translated_output_tab,
        raw_tab,
    ) = st.tabs(tab_list)

    with translations_tab:
        original_block = _pj(detected_texts)
        ul = ", ".join(sorted(set(detected_languages))) if detected_languages else "—"
        src_title = f"Source · Auto ({ul})"
        pair_left, pair_right = st.columns(2)
        with pair_left:
            st.markdown(
                f'<div class="tx-card"><div class="tx-card-label">{_html_escape(src_title)}</div>'
                f'<div class="tx-card-body">{_html_escape(original_block) or " "}</div></div>',
                unsafe_allow_html=True,
            )
            _clipboard_button("📋", original_block, "tx_src", tooltip="Copy source")
            st.download_button(
                label="⬇️",
                data=original_block.encode("utf-8"),
                file_name=f"source_{target_lang_code}.txt",
                mime="text/plain",
                key="dl_src_txt_inline",
                help="Download source text (.txt)",
            )
        with pair_right:
            st.markdown(
                f'<div class="tx-card"><div class="tx-card-label">Target · {_html_escape(target_lang)}</div>'
                f'<div class="tx-card-body">{_html_escape(translated_block) or " "}</div></div>',
                unsafe_allow_html=True,
            )
            _clipboard_button("📋", translated_block, "tx_tgt", tooltip="Copy translation")
        st.download_button(
            label="⬇️",
            data=translated_block.encode("utf-8"),
            file_name=f"translation_{target_lang_code}.txt",
            mime="text/plain",
            key="dl_txt",
            help="Download translation (.txt)",
            use_container_width=True,
        )
        if st.button("Save this translation", key="save_translation_btn", type="primary", use_container_width=True):
            save_source_lang = ", ".join(sorted(set(detected_languages))) if detected_languages else "Auto"
            _upsert_saved_translation(
                {
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "source_lang": save_source_lang,
                    "target_lang": target_lang,
                    "original_text": original_block,
                    "translated_text": translated_block,
                },
                st.session_state.saved_translations,
            )
            st.success("Translation saved.")

    with annotated_tab:
        _render_video_per_frame_annotate_ui()

    with processing_tab:
        st.info(
            "Preprocessing before/after applies to **static image** OCR. "
            "Video uses the same steps on sampled frames during translation."
        )

    with translated_output_tab:
        if translated_video_path and os.path.isfile(translated_video_path):
            st.caption("Same preview as above; you can play or download here too.")
            try:
                st.video(translated_video_path)
            except Exception:
                if translated_video_bytes:
                    st.video(translated_video_bytes, format=translated_video_format)
            if translated_video_bytes:
                st.download_button(
                    label="Download translated video",
                    data=translated_video_bytes,
                    file_name=f"translated_video.{translated_video_ext}",
                    mime=translated_video_format,
                    key="dl_translated_video_tab",
                    use_container_width=True,
                )
        else:
            st.info(
                "Turn on **Generate video with translated text** in Advanced options, then click **Translate**. "
                "When processing finishes, the player and download appear above and in this tab."
            )

    with raw_tab:
        st.json(results_json)
        st.download_button(
            label="Download JSON",
            data=json.dumps(results_json, indent=2, ensure_ascii=False),
            file_name="translation_results.json",
            mime="application/json",
            use_container_width=True,
        )


def _replay_image_translation_ui_snap(snap: dict) -> None:
    """Re-render image translation tabs from session snapshot (widget reruns without **Translate**)."""
    detected_texts = snap["detected_texts"]
    detected_languages = snap["detected_languages"]
    translations = snap["translations"]
    results_json = snap["results_json"]
    target_lang = snap["target_lang"]
    target_lang_code = snap["target_lang_code"]
    input_lang = snap.get("input_lang") or "en"
    preprocess = bool(snap.get("preprocess"))
    ocr_openai_preprocess = bool(snap.get("ocr_openai_preprocess")) or bool(
        st.session_state.get("ocr_openai_preprocess")
    )
    proc_preview_before_rgb = snap.get("proc_before")
    proc_preview_after_rgb = snap.get("proc_after")
    proc_preview_after_openai_rgb = snap.get("proc_after_openai")
    annotated = snap.get("annotated")
    translated_image = snap.get("translated_image")

    st.success(f"Detected {len(detected_texts)} segment(s)")
    if detected_languages:
        unique_langs = ", ".join(sorted(set(detected_languages)))
        st.caption(f"Source: {unique_langs}")
    st.caption(
        "Showing your **last image translation**. Use **Processing** to enlarge the preprocessing preview — "
        "no need to click **Translate** again unless you change the image or options."
    )

    _pj = _paragraph_join
    translated_block = _pj([t["translated_text"] for t in translations])
    tab_list = ["Text", "Annotated", "Processing", "Translated photo", "JSON"]
    (
        translations_tab,
        annotated_tab,
        processing_tab,
        translated_output_tab,
        raw_tab,
    ) = st.tabs(tab_list)

    with translations_tab:
        original_block = _pj(detected_texts)
        ul = ", ".join(sorted(set(detected_languages))) if detected_languages else "—"
        src_title = f"Source · Auto ({ul})"
        pair_left, pair_right = st.columns(2)
        with pair_left:
            st.markdown(
                f'<div class="tx-card"><div class="tx-card-label">{_html_escape(src_title)}</div>'
                f'<div class="tx-card-body">{_html_escape(original_block) or " "}</div></div>',
                unsafe_allow_html=True,
            )
            _clipboard_button("📋", original_block, "tx_src", tooltip="Copy source")
            st.download_button(
                label="⬇️",
                data=original_block.encode("utf-8"),
                file_name=f"source_{target_lang_code}.txt",
                mime="text/plain",
                key="dl_src_txt_inline",
                help="Download source text (.txt)",
            )
        with pair_right:
            st.markdown(
                f'<div class="tx-card"><div class="tx-card-label">Target · {_html_escape(target_lang)}</div>'
                f'<div class="tx-card-body">{_html_escape(translated_block) or " "}</div></div>',
                unsafe_allow_html=True,
            )
            _clipboard_button("📋", translated_block, "tx_tgt", tooltip="Copy translation")
            st.download_button(
                label="⬇️",
                data=translated_block.encode("utf-8"),
                file_name=f"translation_{target_lang_code}.txt",
                mime="text/plain",
                key="dl_tgt_txt_inline_img",
                help="Download target text (.txt)",
            )
        if st.button("Save this translation", key="save_translation_btn", type="primary", use_container_width=True):
            save_source_lang = ", ".join(sorted(set(detected_languages))) if detected_languages else "Auto"
            _upsert_saved_translation(
                {
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "source_lang": save_source_lang,
                    "target_lang": target_lang,
                    "original_text": original_block,
                    "translated_text": translated_block,
                },
                st.session_state.saved_translations,
            )
            st.success("Translation saved.")

    with annotated_tab:
        if annotated is not None:
            _safe_show_image(annotated, caption="Detected text regions", use_column_width=True)
        else:
            st.info("Enable **Show bounding boxes** in Advanced options to see annotations.")

    with processing_tab:
        _render_processing_preprocess_views(
            proc_preview_before_rgb,
            proc_preview_after_rgb,
            proc_preview_after_openai_rgb,
            preprocess=preprocess,
            ocr_openai_preprocess=ocr_openai_preprocess,
            openai_preprocess_error=snap.get("openai_preprocess_error"),
            zoom_toggle_key="ocr_preprocess_zoom_toggle_replay",
            new_tab_key_local="ppt_preprocess_replay",
            new_tab_key_openai="ppt_preprocess_openai_replay",
        )

    with translated_output_tab:
        if translated_image is not None:
            _safe_show_image(translated_image, caption=f"Translated to {target_lang}", use_column_width=True)
            _arr_out = _normalize_to_rgb_uint8_array(translated_image)
            pil_out = Image.fromarray(_arr_out)
            buf = io.BytesIO()
            pil_out.save(buf, format="PNG")
            translated_png_bytes = buf.getvalue()
            st.download_button(
                label="Download image",
                data=translated_png_bytes,
                file_name="translated_image.png",
                mime="image/png",
                key="dl_translated_img",
            )
            detected_block = _pj(detected_texts)
            save_source_lang = ", ".join(sorted(set(detected_languages))) if detected_languages else "Auto"
            image_save_item = {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "source_lang": save_source_lang,
                "target_lang": target_lang,
                "original_text": detected_block,
                "translated_text": translated_block,
                "detected_text": detected_block,
                "translated_photo_png_b64": base64.b64encode(translated_png_bytes).decode("ascii"),
            }
            saved_idx = _find_saved_item_index(image_save_item, st.session_state.saved_translations)
            has_saved_image = saved_idx >= 0 and bool(
                (st.session_state.saved_translations[saved_idx] or {}).get("translated_photo_png_b64")
            )
            image_star = "★ Saved image" if has_saved_image else "☆ Save image result"
            if st.button(
                image_star,
                key="save_image_result_star",
                use_container_width=True,
                help=(
                    "Already saved with photo"
                    if has_saved_image
                    else "Save translated photo + detected text + translated text"
                ),
                disabled=has_saved_image,
            ):
                _upsert_saved_translation(image_save_item, st.session_state.saved_translations)
                st.success("Image translation saved.")
            st.caption("Use the ✦ Enhance with AI button below the result section.")
        else:
            st.info(
                "Enable **Generate photo with translated text** (image mode) in Advanced options, then run **Translate**."
            )

    with raw_tab:
        st.json(results_json)
        st.download_button(
            label="Download JSON",
            data=json.dumps(results_json, indent=2, ensure_ascii=False),
            file_name="translation_results.json",
            mime="application/json",
            use_container_width=True,
        )


def main():
    apply_custom_styles()

    lang_names = {v: k for k, v in SUPPORTED_LANGUAGES.items()}
    ocr_lang_options = {
        'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
        'Italian': 'it', 'Portuguese': 'pt', 'Chinese': 'zh', 'Japanese': 'ja',
        'Korean': 'ko', 'Arabic': 'ar', 'Russian': 'ru', 'Hindi': 'hi',
    }

    with st.sidebar:
        st.markdown(
            """
            <div class="side-brand">
              <div class="side-logo">A友</div>
              <div>
                <p class="side-title">SkyTranslate</p>
                <p class="side-sub">AI Translator</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        nav_items = [
            ("💬 Conversation", "translate"),
            ("🕘 History", "history"),
            ("🔖 Saved", "saved"),
        ]
        for i, (label, key) in enumerate(nav_items):
            if st.button(
                label,
                key=f"side_nav_{i}",
                use_container_width=True,
                type="primary" if st.session_state.ui_nav == key else "secondary",
            ):
                st.session_state.ui_nav = key
                st.rerun()

    if st.session_state.ui_nav in ("home", "search"):
        st.session_state.ui_nav = "translate"

    if st.session_state.ui_nav == "home":
        st.markdown(
            """
            <div class="greeting-row">
              <div class="avatar-ring">✶</div>
              <div class="greeting-text"><strong>Hi there,</strong><br/><span>Welcome back</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<p class="app-header">Translate Studio</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="app-tagline">Offline OCR and translation — text, camera scan, voice, and video.</p>',
            unsafe_allow_html=True,
        )
        if st.session_state.history:
            st.markdown("**Recent activity**")
            for item in st.session_state.history[:5]:
                prev = (item.get("preview") or "")[:56]
                st.markdown(
                    f'<div class="tx-card"><div class="tx-card-body" style="min-height:auto;font-size:0.9rem;">'
                    f'{item.get("num_segments", 0)} segment(s) → <strong>{item.get("target_lang", "")}</strong>'
                    f'{" · " + _html_escape(prev) + "…" if prev else ""}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Your recent translations will appear here. Open **Conversation** to get started.")
        if st.button("Open translator", type="primary", key="home_go_translate"):
            st.session_state.ui_nav = "translate"
            st.rerun()
        st.markdown("<br/>", unsafe_allow_html=True)
        hb = st.columns(4)
        with hb[0]:
            if st.button("Home", key="nav_home_b_home", use_container_width=True, type="primary"):
                st.session_state.ui_nav = "home"
                st.rerun()
        with hb[1]:
            if st.button("Conversation", key="nav_tr_b_home", use_container_width=True, type="secondary"):
                st.session_state.ui_nav = "translate"
                st.rerun()
        with hb[2]:
            if st.button("Saved", key="nav_saved_b_home", use_container_width=True, type="secondary"):
                st.session_state.ui_nav = "saved"
                st.rerun()
        with hb[3]:
            if st.button("Settings", key="nav_search_b_home", use_container_width=True, type="secondary"):
                st.session_state.ui_nav = "search"
                st.rerun()
        st.stop()

    if st.session_state.ui_nav == "search":
        st.markdown('<p class="app-header">Language &amp; region</p>', unsafe_allow_html=True)
        st.caption("Choose a default **target** language for translation (NLLB-200 · offline).")
        q = st.text_input("Search your language", placeholder="Type to filter…", key="lang_search_input", label_visibility="collapsed")
        query = (q or "").strip().lower()
        all_sorted = sorted(lang_names.keys())
        filtered = [n for n in all_sorted if not query or query in n.lower()]
        st.markdown('<p class="lang-grid-title">Popular choices</p>', unsafe_allow_html=True)
        popular_names = [SUPPORTED_LANGUAGES[c] for c in _POPULAR_LANG_CODES if c in SUPPORTED_LANGUAGES]
        for row_start in range(0, len(popular_names), 3):
            row_cols = st.columns(3)
            for j in range(3):
                idx = row_start + j
                if idx >= len(popular_names):
                    break
                name = popular_names[idx]
                code = lang_names[name]
                flag = _LANG_FLAGS.get(code, "🌐")
                with row_cols[j]:
                    if st.button(f"{flag} {name}", key=f"plang_{code}", use_container_width=True):
                        st.session_state.picked_target_lang = name
                        st.session_state.ui_nav = "translate"
                        st.rerun()
        st.markdown('<p class="lang-grid-title">All languages</p>', unsafe_allow_html=True)
        for row_start in range(0, min(len(filtered), 60), 3):
            row_cols = st.columns(3)
            for j in range(3):
                idx = row_start + j
                if idx >= min(len(filtered), 60):
                    break
                name = filtered[idx]
                code = lang_names[name]
                flag = _LANG_FLAGS.get(code, "🌐")
                with row_cols[j]:
                    if st.button(f"{flag} {name}", key=f"alang_{code}", use_container_width=True):
                        st.session_state.picked_target_lang = name
                        st.session_state.ui_nav = "translate"
                        st.rerun()
        if len(filtered) > 60:
            st.caption("Narrow your search to see more.")
        if st.button("Back to translator", key="search_back"):
            st.session_state.ui_nav = "translate"
            st.rerun()
        st.markdown("<br/>", unsafe_allow_html=True)
        sb = st.columns(4)
        with sb[0]:
            if st.button("Home", key="nav_home_b_search", use_container_width=True, type="secondary"):
                st.session_state.ui_nav = "home"
                st.rerun()
        with sb[1]:
            if st.button("Conversation", key="nav_tr_b_search", use_container_width=True, type="secondary"):
                st.session_state.ui_nav = "translate"
                st.rerun()
        with sb[2]:
            if st.button("Saved", key="nav_saved_b_search", use_container_width=True, type="secondary"):
                st.session_state.ui_nav = "saved"
                st.rerun()
        with sb[3]:
            if st.button("Settings", key="nav_search_b_search", use_container_width=True, type="primary"):
                st.session_state.ui_nav = "search"
                st.rerun()
        st.stop()

    if st.session_state.ui_nav == "history":
        st.markdown('<p class="app-header">History</p>', unsafe_allow_html=True)
        st.caption("All recent translation runs (auto-saved locally).")
        if st.session_state.history:
            for idx, item in enumerate(st.session_state.history):
                prev = (item.get("preview") or "")[:120]
                created_at = item.get("created_at", "Unknown time")
                source = item.get("source_lang", "Auto")
                target = item.get("target_lang", "")
                original_text = item.get("original_text") or item.get("preview", "")
                translated_text = item.get("translated_text", "")
                row_main, row_star = st.columns([12, 1])
                with row_main:
                    with st.expander(
                        f"{created_at} · {source} → {target} · {item.get('num_segments', 0)} segment(s)",
                        expanded=(idx == 0),
                    ):
                        st.markdown("**Text to translate**")
                        st.write(original_text or (prev if prev else "—"))
                        st.markdown("**Translation**")
                        st.write(translated_text or "—")
                with row_star:
                    is_saved = _history_item_already_saved(item, st.session_state.saved_translations)
                    star_label = "★" if is_saved else "☆"
                    star_key = "save_hist_item_saved" if is_saved else "save_hist_item_unsaved"
                    if st.button(
                        star_label,
                        key=f"{star_key}_{idx}",
                        help="Already saved" if is_saved else "Save this history item",
                        use_container_width=True,
                        disabled=is_saved,
                    ):
                        source_lang = item.get("source_lang", "Auto")
                        target_lang = item.get("target_lang", "")
                        original_text = item.get("original_text") or item.get("preview", "")
                        translated_text = item.get("translated_text", "")
                        _upsert_saved_translation(
                            {
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                "source_lang": source_lang,
                                "target_lang": target_lang,
                                "original_text": original_text,
                                "translated_text": translated_text,
                            },
                            st.session_state.saved_translations,
                        )
                        st.session_state.ui_nav = "saved"
                        st.success("Saved to Saved page.")
                        st.rerun()
        else:
            st.info("No history yet. Run a translation in **Conversation**.")
        if st.button("Clear history", key="clear_history_page", type="secondary"):
            st.session_state.history.clear()
            _save_history(st.session_state.history)
            st.rerun()
        st.stop()

    if st.session_state.ui_nav == "saved":
        st.markdown(
            """
            <div class="greeting-row">
              <div class="avatar-ring">✶</div>
              <div class="greeting-text"><strong>Saved</strong><br/><span>Your history</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<p class="app-header">Saved translations</p>', unsafe_allow_html=True)
        st.caption("Stored locally in `.ocr_studio_saved_translations.json` (last 100 items).")
        if st.session_state.saved_translations:
            for i, item in enumerate(st.session_state.saved_translations):
                created_at = item.get("created_at", "Unknown time")
                target = item.get("target_lang", "")
                source = item.get("source_lang", "")
                original = item.get("original_text", "")
                translated = item.get("translated_text", "")
                with st.expander(f"{created_at} · {source} → {target}", expanded=(i == 0)):
                    st.markdown("**Text to translate**")
                    st.write(original or "—")
                    st.markdown("**Translation**")
                    st.write(translated or "—")
                    photo_b64 = (item.get("translated_photo_png_b64") or "").strip()
                    if photo_b64:
                        st.markdown("**Saved translated photo**")
                        try:
                            photo_bytes = base64.b64decode(photo_b64)
                            if photo_bytes:
                                st.image(photo_bytes, caption="Saved photo result", use_column_width=True)
                                st.download_button(
                                    label="Download saved photo",
                                    data=photo_bytes,
                                    file_name="saved_translated_image.png",
                                    mime="image/png",
                                    key=f"dl_saved_photo_{i}",
                                    use_container_width=True,
                                )
                        except Exception:
                            st.caption("Saved image data is unavailable or corrupted.")
        else:
            st.info("Nothing saved yet. Translate text first, then press **Save this translation**.")
        if st.button("Clear all saved", key="clear_all_saved", type="secondary"):
            st.session_state.saved_translations.clear()
            _save_saved_translations(st.session_state.saved_translations)
            st.rerun()
        st.stop()

    # ---------- Translate workspace (main flow) ----------
    _ensure_local_translator()

    picked = st.session_state.get("picked_target_lang")
    default_names = sorted(lang_names.keys())
    default_idx = 0
    if picked and picked in lang_names:
        default_idx = default_names.index(picked)

    st.markdown(
        """
        <p class="hello-line">Hello! 👋</p>
        <h1 class="hero-title"><span class="g-plain">Translate</span> <span class="g-grad">anything, anywhere</span></h1>
        <p class="hero-sub">Break the language barrier with AI-powered translations.</p>
        """,
        unsafe_allow_html=True,
    )
    if "main_mode_tier" not in st.session_state:
        st.session_state.main_mode_tier = "Text"

    card1, card2, card3, card4 = st.columns(4)
    with card1:
        st.markdown(
            '<div class="mode-card"><div class="mode-ico">📝</div><p class="mode-title">Text Translate</p><p class="mode-desc">Translate typed text instantly.</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("Open text", key="mode_text_btn", use_container_width=True):
            st.session_state.main_mode_tier = "Text"
            st.session_state["text_input_method"] = "write"
            st.rerun()
    with card2:
        st.markdown(
            '<div class="mode-card"><div class="mode-ico">🖼️</div><p class="mode-title">Image Translate</p><p class="mode-desc">Extract and translate text from images.</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("Open image", key="mode_image_btn", use_container_width=True):
            st.session_state.main_mode_tier = "Image"
            st.rerun()
    with card3:
        st.markdown(
            '<div class="mode-card"><div class="mode-ico">🎞️</div><p class="mode-title">Video Translate</p><p class="mode-desc">Translate video frames and subtitles.</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("Open video", key="mode_video_btn", use_container_width=True):
            st.session_state.main_mode_tier = "Video"
            st.rerun()
    with card4:
        st.markdown(
            '<div class="mode-card"><div class="mode-ico">✍️</div><p class="mode-title">Handwriting</p><p class="mode-desc">Photo or PDF → OCR, translate, export PDF/PNG.</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("Open handwriting", key="mode_handwriting_btn", use_container_width=True):
            st.session_state.main_mode_tier = "Handwriting"
            st.rerun()

    mode_tier = st.session_state.main_mode_tier
    if mode_tier == "Text":
        if "text_input_method" not in st.session_state:
            st.session_state["text_input_method"] = "write"
        mode = "From voice" if st.session_state.get("text_input_method") == "voice" else "From text"
    elif mode_tier == "Image":
        mode = "From image"
    elif mode_tier == "Handwriting":
        mode = "From handwriting"
    else:
        mode = "From video"

    st.markdown("**Languages**")
    if mode in ("From text", "From voice"):
        english_name = SUPPORTED_LANGUAGES.get("en", "English")
        if "source_lang_main" not in st.session_state:
            st.session_state["source_lang_main"] = (
                english_name if english_name in default_names else default_names[0]
            )
        if (
            st.session_state.get("source_lang_main") is not None
            and st.session_state.get("target_lang_main") is not None
            and st.session_state["source_lang_main"] == st.session_state["target_lang_main"]
        ):
            st.session_state["target_lang_main"] = _pick_other_language(st.session_state["source_lang_main"])
        lr1, lr2, lr3 = st.columns([1, 0.2, 1])
        with lr1:
            input_lang = st.selectbox(
                "Source language",
                options=default_names,
                index=0,
                key="source_lang_main",
                on_change=_ensure_distinct_text_languages,
                args=("source",),
            )
        with lr2:
            st.caption(" ")
            st.button(
                "⇄",
                key="swap_lang_btn",
                use_container_width=True,
                help="Swap source and target",
                on_click=_swap_main_languages,
            )
        with lr3:
            target_lang = st.selectbox(
                "Translate to",
                options=default_names,
                index=default_idx,
                key="target_lang_main",
                on_change=_ensure_distinct_text_languages,
                args=("target",),
            )
        input_lang_code = lang_names[input_lang]
        ocr_languages = ["en"]
    elif mode == "From handwriting":
        st.caption("Translate to")
        target_lang = st.selectbox(
            "Translate to",
            options=default_names,
            index=default_idx,
            key="target_lang_main",
            label_visibility="collapsed",
        )
        input_lang_code = None
        input_lang = None
        ocr_languages = ["en"]
    else:
        ocr_col, target_col = st.columns([1, 1])
        with ocr_col:
            st.caption("OCR languages (select one or more)")
            _ocr_label_choices = list(ocr_lang_options.keys())
            _ocr_default_labels = st.session_state.get(
                "ocr_detect_lang_labels",
                ["English"],
            )
            _ocr_default_labels = [
                x for x in _ocr_default_labels if x in _ocr_label_choices
            ] or ["English"]
            selected_ocr_labels = st.multiselect(
                "OCR detect languages",
                options=_ocr_label_choices,
                default=_ocr_default_labels,
                key="ocr_detect_lang_labels",
                help="Choose one or more languages to detect from images/videos.",
            )
        with target_col:
            st.caption("Target language for translation")
            target_lang = st.selectbox("Translate to", options=default_names, index=default_idx, key="target_lang_main")
        input_lang_code = None
        input_lang = None
        if not selected_ocr_labels:
            selected_ocr_labels = ["English"]
        ocr_languages = [ocr_lang_options[name] for name in selected_ocr_labels]
        st.caption("Selected OCR languages: " + ", ".join(selected_ocr_labels))

    target_lang_code = lang_names[target_lang]

    st.markdown("**Compute**")
    st.checkbox(
        "Use CPU for OCR & translation",
        key="use_cpu_only",
        on_change=_on_cpu_only_toggle,
        help="Runs EasyOCR, TrOCR, PaddleOCR, and local NLLB on CPU. "
        "Google Cloud translation is always network-based. "
        "Set APP_USE_CPU=1 in .env to start with this on.",
    )
    if st.session_state.get("use_cpu_only"):
        st.caption("CPU mode: OCR and offline translation will not use the GPU.")

    google_project_id = None
    preprocess = False
    fast_ocr = False
    ocr_small_text = False
    ocr_high_recall = False
    ocr_openai_preprocess = False
    photo_dewarp = False
    fast_translation = False
    preserve_arabic_names = True
    draw_boxes = True
    generate_photo = False
    photo_inpaint_solid = True
    photo_merge_lines = False
    photo_realistic_blend = True
    photo_strong_erase = True
    photo_perspective_warp = True
    generate_video = False
    sample_interval_sec = 2.0
    max_video_frames = 60
    video_quality_enhance = False
    video_export_crf = 23
    video_export_preset = "veryfast"
    video_translated_font_scale = 0.75
    translation_provider = st.session_state.get("translation_provider", "google_cloud")
    use_google_cloud = translation_provider == "google_cloud"
    if mode == "From handwriting":
        translation_provider = "openai_chat"
        st.session_state.translation_provider = "openai_chat"
        use_google_cloud = False
        google_project_id = None
        draw_boxes = False
        preserve_arabic_names = False
        # Handwriting flow is OCR-heavy; prefer fast translation defaults.
        fast_translation = True
        ocr_small_text = False
        ocr_high_recall = False
        generate_photo = True
        generate_video = False

    elif mode != "From text":
        st.session_state.translation_provider = "google_cloud"
        translation_provider = "google_cloud"
        use_google_cloud = True
        st.session_state.translator = None
        google_project_id = st.text_input(
            "Google Cloud project ID (optional)",
            key="google_cloud_project_id",
            help="Optional. If empty, client uses default project from credentials.",
        ).strip() or None
        st.markdown("##### Advanced options")
        st.caption("OCR engine, preprocessing, and quality toggles for image and video.")
        fast_ocr = False
        st.selectbox(
            "OCR engine",
            options=["easyocr", "paddleocr", "trocr_hybrid", "trocr_only"],
            key="ocr_backend",
            help="easyocr: default scene text. paddleocr: isolated worker (Windows-friendly). "
            "trocr_hybrid: EasyOCR finds text boxes, TrOCR (handwriting) reads each line—best for neat Latin handwriting; "
            "trocr_only: no EasyOCR, uses TrOCR with OpenCV line proposals; "
            "first use downloads the TrOCR model. Override with sidebar field or OCR_TROCR_MODEL.",
        )
        if st.session_state.get("ocr_backend", "easyocr") in ("trocr_hybrid", "trocr_only"):
            st.text_input(
                "TrOCR model (Hugging Face id)",
                key="trocr_model_id",
                placeholder="e.g. microsoft/trocr-large-handwritten (empty = default / OCR_TROCR_MODEL)",
            )
        ocr_openai_preprocess = st.checkbox(
            "OpenAI preprocess before OCR (GPT-Image)",
            value=_env_flag("OCR_PREPROCESS_OPENAI"),
            key="ocr_openai_preprocess",
            help="Runs OpenAI **images.edit** to improve contrast and reduce glare before OCR (~30–90s, API cost). "
            "Works **without** turning on local preprocess — only this checkbox is required. "
            "Requires **OPENAI_API_KEY**. Optional: also enable **Preprocess image before OCR** for GPU/OpenCV steps after OpenAI. "
            "Faster/cheaper: OPENAI_IMAGE_ENHANCE_FAST=1.",
        )
        if mode in ("From image", "From video"):
            preprocess = st.checkbox(
                "Preprocess image before OCR",
                value=True,
                key="ocr_preprocess_before_ocr",
                help="**Engine:** ``OCR_PREPROCESS_ENGINE=gpu`` (default in code) runs the **CUDA** tensor pipeline when a GPU is available; ``auto`` is the same; ``legacy`` / ``cpu`` / ``opencv`` use OpenCV on CPU. Set ``OCR_PREPROCESS_FORCE_GPU=1`` to keep **preprocess on CUDA** even when OCR falls back to CPU after OOM. Scene analysis, DnCNN on tensor, illumination, edge-aware sharpen, etc.; ``OCR_PREPROCESS_DEBUG=1`` logs stage timings. "
                "Legacy path: resize, optional shadow lift, DnCNN or NL-means denoise, gentle bilateral smoothing, optional **L-channel median** speckle cleanup, optional **L-channel percentile stretch**, optional **CLAHE** (auto-skip when still noisy), then **adaptive edge restoration** so tiny OCR strokes are not blurred away. "
                "Optional: ``OCR_PREPROCESS_PERSPECTIVE=1`` (quad dewarp before GPU), ``OCR_PREPROCESS_RIDNET_JIT`` (TorchScript denoiser), ``OCR_PREPROCESS_GPU_EMPTY_CACHE=1`` after stages on tight VRAM. "
                "If results look **more noisy**, set ``OCR_PREPROCESS_GPU_ADAPTIVE_CLAHE=0``, lower ``OCR_PREPROCESS_UNSHARP``, ``OCR_PREPROCESS_BRIGHTEN`` closer to 1, and try ``OCR_PREPROCESS_BILATERAL_PASSES=2`` on legacy. "
                "Env reference: ``OCR_PREPROCESS_EDGE_RESTORE_FLOOR``, ``OCR_PREPROCESS_MEDIAN_L``, ``OCR_PREPROCESS_LUMA_STRETCH_BLEND``, ``OCR_PREPROCESS_CLAHE_AUTO``, ``OCR_PREPROCESS_CLAHE_MAX_STD``. Video: ``OCR_VIDEO_PREPROCESS_MAX_SIDE``.",
            )
        else:
            preprocess = False
        photo_dewarp = False
        if mode == "From image":
            photo_dewarp = st.checkbox(
                "Straighten angled screen / document (perspective)",
                value=False,
                key="photo_dewarp",
                help="Finds a quadrilateral (monitor or page) and warps to a frontal view before OCR. "
                "Use for photos taken at an angle — boxes and overlay align to the rectified image. "
                "OCR cache is skipped for this run when enabled.",
            )
        preserve_arabic_names = st.checkbox(
            "Keep short mixed Latin + Arabic labels untranslated (e.g. logos)",
            value=True,
            key="preserve_arabic_names",
            help="When translating into non–Arabic-script languages (English, German, …), leaves **only** "
            "very short lines that mix **Latin letters and Arabic** (typical logo / brand lines) unchanged. "
            "Pure Arabic headings and sentences are always translated. "
            "Turn off to never skip. Set env TRANSLATION_PRESERVE_ARABIC_NAMES_OFF=1 to disable globally.",
        )
        draw_boxes = st.checkbox(
            "Show bounding boxes",
            value=True,
            key="draw_boxes",
            help="Draws cyan boxes around each OCR region (no detected text drawn on the image). "
            "Duplicate lines are hidden by default (set OCR_DRAW_BOXES_DEDUPE=0 to show all). "
            "Set OCR_DRAW_BOXES_HIDE_LABELS=0 to also paint recognized text inside each box.",
        )
        generate_photo = (
            st.checkbox(
                "Generate photo with translated text",
                value=False,
                key="generate_photo",
                help="Uses OpenAI Images API to generate a translated photo while preserving scene/layout.",
            )
            if mode in ("From image", "From handwriting")
            else False
        )
        photo_inpaint_solid = True
        photo_merge_lines = False
        photo_realistic_blend = True
        photo_strong_erase = True
        if mode in ("From image", "From handwriting") and generate_photo:
            if mode == "From handwriting":
                st.session_state.photo_output_engine = "openai_gpt_image_1"
                st.caption("Handwriting translated photo uses OpenAI Images edit to preserve the page layout.")
            else:
                st.selectbox(
                    "Translated photo output",
                    options=["local_overlay", "openai_gpt_image_1"],
                    index=0,
                    key="photo_output_engine",
                    format_func=lambda x: (
                        "Local — erase and draw text in original OCR boxes (same layout)"
                        if x == "local_overlay"
                        else "OpenAI Images API — gpt-image-1.5 (edits your photo; billed by OpenAI)"
                    ),
                    help="Choose local in-place overlay or OpenAI Images edit.",
                )
            _engine = st.session_state.get("photo_output_engine", "local_overlay")
            if _engine == "local_overlay":
                photo_merge_lines = st.checkbox(
                    "Merge text boxes on the same horizontal line",
                    value=False,
                    key="photo_merge_lines",
                    help="Joins side-by-side OCR boxes into one. Leave off for menus so each row stays separate.",
                )
                photo_perspective_warp = st.checkbox(
                    "Match text angle (tilted / slanted text)",
                    value=True,
                    key="photo_perspective_warp",
                    help="Draws each translation in the same perspective as the detected OCR box. "
                    "Turn off for always-horizontal text. Env PHOTO_OVERLAY_PERSPECTIVE_WARP=0 also disables.",
                )
                photo_inpaint_solid = st.checkbox(
                    "Solid local fill (recommended for light walls/menus)",
                    value=True,
                    key="photo_inpaint_solid",
                    help="Fills each text region with the median color from around that line (tight mask). "
                    "Usually cleaner than inpainting for plain walls. If you use inpainting instead, the app defaults to "
                    "Navier–Stokes (set env PHOTO_INPAINT_ALGORITHM=telea for the older Telea look).",
                )
                photo_realistic_blend = st.checkbox(
                    "Realistic text blending (smoother edges)",
                    value=True,
                    key="photo_realistic_blend",
                    help="Feathers text edges and lightly adapts text color to local background so overlays look less pasted.",
                )
                photo_strong_erase = st.checkbox(
                    "Stronger erase (less leftover English / ghosting)",
                    value=True,
                    key="photo_strong_erase",
                    help="Widens the mask under each text line so more of the original pixels are removed. "
                    "Use for menus on wood/grain where faint English still shows.",
                )
            else:
                st.caption(
                    "Uses **OPENAI_API_KEY** (e.g. in `.env`). Model **gpt-image-1.5** may require **organization verification** on OpenAI."
                )
                st.selectbox(
                    "GPT-Image quality",
                    options=["auto", "low", "medium", "high"],
                    index=3,
                    key="openai_image_quality",
                )
                st.selectbox(
                    "Output size",
                    options=["auto", "1024x1024", "1024x1536", "1536x1024"],
                    index=0,
                    key="openai_image_size",
                )
                st.selectbox(
                    "Input fidelity (match your photo)",
                    options=["high", "low"],
                    index=0,
                    key="openai_image_input_fidelity",
                    help="Higher keeps the original scene closer to your upload when editing.",
                )
                st.text_input(
                    "OpenAI image model",
                    value=default_image_model(),
                    key="openai_image_model",
                    help="Default: gpt-image-1.5, or set OPENAI_IMAGE_MODEL in the environment.",
                )
            if mode == "From image":
                st.checkbox(
                    "Use OpenAI (GPT) for translation on this photo only",
                    value=False,
                    key="openai_photo_only",
                    help="Calls the OpenAI Chat API to translate OCR text before image generation. "
                    "Does not apply to text/voice/video. Set OPENAI_API_KEY or enter a key below.",
                )
            _need_openai_key_ui = bool(
                st.session_state.get("openai_photo_only")
            ) or st.session_state.get("photo_output_engine") == "openai_gpt_image_1"
            if _need_openai_key_ui:
                if os.environ.get("OPENAI_API_KEY"):
                    st.caption("Using **OPENAI_API_KEY** from the environment.")
                else:
                    try:
                        if getattr(st, "secrets", None) and st.secrets.get("OPENAI_API_KEY"):
                            st.caption("Using **OPENAI_API_KEY** from Streamlit secrets.")
                        else:
                            st.text_input(
                                "OpenAI API key (photo / Images)",
                                type="password",
                                key="openai_api_key_input",
                                help="https://platform.openai.com/api-keys — Chat translation and/or GPT-Image.",
                            )
                    except Exception:
                        st.text_input(
                            "OpenAI API key (photo / Images)",
                            type="password",
                            key="openai_api_key_input",
                            help="https://platform.openai.com/api-keys — Chat translation and/or GPT-Image.",
                        )
                if mode == "From image" and st.session_state.get("openai_photo_only"):
                    st.text_input(
                        "OpenAI model (this photo)",
                        value=os.environ.get("OPENAI_TRANSLATION_MODEL", "gpt-4o-mini"),
                        key="openai_model_photo",
                        help="e.g. gpt-4o-mini, gpt-4o — used only for **translation** when that option is on.",
                    )
            st.caption(
                "For clearer wording and placement: include **every language printed on the photo** in OCR languages, "
                "and prefer a larger NLLB model if possible. "
                "and make sure translated strings are accurate before generation."
            )
        generate_video = st.checkbox(
            "Generate video with translated text",
            value=False,
            key="generate_video",
            help="After Translate, the app shows the translated video with a player and a download button.",
        ) if mode == "From video" else False
        if mode == "From video":
            video_quality_enhance = st.checkbox(
                "Enhance generated video quality",
                value=False,
                key="video_quality_enhance",
                help="Improves output video rendering/export quality (visual quality), not translation wording.",
            )
            video_translated_font_scale = st.slider(
                "Translated subtitle size (video)",
                min_value=0.30,
                max_value=1.50,
                value=0.75,
                step=0.05,
                key="video_translated_font_scale",
                help="Scales the size of translated text drawn on video frames. "
                "Lower values (e.g. 0.5) shrink subtitles so they don't dominate the frame; "
                "1.0 = original auto-fit size. Per-box fit is still respected.",
            )
            video_export_crf = st.slider(
                "Video export quality (CRF, lower = better quality)",
                min_value=16,
                max_value=30,
                value=20 if video_quality_enhance else 23,
                step=1,
                key="video_export_crf",
                help="Used for final H.264 export. Lower values give better quality and larger files.",
            )
            video_export_preset = st.selectbox(
                "Video export speed/quality preset",
                options=["veryfast", "faster", "fast", "medium", "slow"],
                index=4 if video_quality_enhance else 0,
                key="video_export_preset",
                help="Slower presets improve compression quality but take longer.",
            )
            sample_interval_sec = st.slider(
                "Sample frame every (sec)",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5,
                key="sample_interval",
                help="OCR runs only on frames at this spacing (plus a short burst at the start). "
                "Subtitles that appear only between samples may be missed—use a smaller interval.",
            )
            max_video_frames = st.slider(
                "Max OCR passes (frames with text)",
                min_value=15,
                max_value=120,
                value=60,
                step=15,
                key="max_video_frames",
                help="Stops scanning the file after this many frames where OCR actually found text (not the same as video length). "
                "Raise this for long videos or if later scenes never get translated.",
            )
        else:
            sample_interval_sec = 2.0
            max_video_frames = 60

    temp_path = None
    input_text = ""
    audio_path = None
    video_path = None
    text_output_placeholder = None

    if mode == "From text":
        text_src_col, text_tgt_col = st.columns(2)
        with text_src_col:
            st.caption("Source language")
            input_text = st.text_area(
                "Paste or type text to translate",
                placeholder="Enter text here...",
                height=200,
                label_visibility="collapsed",
                key="source_text_input_v2",
            )
            src_tool_1, src_tool_2, src_tool_3, src_tool_spacer = st.columns([1, 1, 1, 10])
            with src_tool_1:
                st.button(
                    "🎤",
                    key="text_tool_mic",
                    help="Use voice input",
                    on_click=_activate_voice_input,
                )
            with src_tool_2:
                st.button(
                    "◻️",
                    key="text_tool_scan",
                    help="Switch to image mode",
                    on_click=_activate_image_mode,
                )
            with src_tool_3:
                st.button(
                    "⌨️",
                    key="text_tool_write",
                    help="Keep writing",
                    on_click=_activate_write_input,
                )
            _mount_source_icons_inside_box()
        with text_tgt_col:
            st.caption("Translation")
            text_output_placeholder = st.empty()
            text_output_placeholder.text_area(
                "Translation output",
                value=st.session_state.get("last_text_translation", ""),
                placeholder="Translation will appear here...",
                height=200,
                label_visibility="collapsed",
                disabled=True,
            )
        _n = len(input_text or "")
        st.caption(f"Characters **{_n}** / 5000 · Translation runs when there is text.")
    elif mode == "From image":
        img_src = st.radio(
            "Image source",
            ["Upload file", "Use camera"],
            horizontal=True,
            label_visibility="collapsed",
            key="image_source_mode",
        )
        if img_src == "Upload file":
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=["png", "jpg", "jpeg", "bmp", "tiff"],
                label_visibility="collapsed",
                key="image_file_uploader",
            )
            if uploaded_file is not None:
                try:
                    raw = uploaded_file.getvalue()
                    _image_mode_sync_src_and_rotation(raw)
                    image = Image.open(io.BytesIO(raw))
                    image = _image_mode_auto_orient_pil(image)
                    image = image.convert("RGB")
                    image_arr = _normalize_to_rgb_uint8_array(image)
                    image = Image.fromarray(image_arr)
                except Exception as e:
                    st.warning(f"Could not open image: {e}. Try a different file (PNG, JPG, BMP, TIFF).")
                    image = None
                if image is not None:
                    rot_c1, rot_c2, _ = st.columns([1, 1, 6])
                    with rot_c1:
                        if st.button(
                            "↶",
                            key="image_rot_ccw",
                            help="Rotate 90° counterclockwise",
                        ):
                            st.session_state["image_mode_rotation"] = (
                                st.session_state.get("image_mode_rotation", 0) - 90
                            ) % 360
                    with rot_c2:
                        if st.button(
                            "↷",
                            key="image_rot_cw",
                            help="Rotate 90° clockwise",
                        ):
                            st.session_state["image_mode_rotation"] = (
                                st.session_state.get("image_mode_rotation", 0) + 90
                            ) % 360
                    image = _apply_image_mode_rotation(image)
                    show_bytes = (
                        raw if int(st.session_state.get("image_mode_rotation", 0)) % 360 == 0 else None
                    )
                    st.caption(
                        "We auto-apply **camera EXIF** and, if Tesseract is installed, **page orientation** when possible. "
                        "Use **↶** / **↷** to override."
                    )
                    _safe_show_image(
                        image,
                        use_column_width=True,
                        original_bytes=show_bytes,
                    )
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                        image.save(tmp_file.name, format="PNG")
                        temp_path = tmp_file.name
        else:
            st.caption(
                "Allow camera access when prompted. Works on **localhost** or **HTTPS**; some browsers block camera on plain HTTP."
            )
            camera_shot = st.camera_input(
                "Take a photo",
                label_visibility="collapsed",
                key="image_camera_input",
                help="Point at the text, capture, then click **Translate**.",
            )
            if camera_shot is not None:
                try:
                    raw = camera_shot.getvalue()
                    _image_mode_sync_src_and_rotation(raw)
                    image = Image.open(io.BytesIO(raw))
                    image = _image_mode_auto_orient_pil(image)
                    image = image.convert("RGB")
                    image_arr = _normalize_to_rgb_uint8_array(image)
                    image = Image.fromarray(image_arr)
                except Exception as e:
                    st.warning(f"Could not read camera image: {e}. Try again or use **Upload file**.")
                    image = None
                if image is not None:
                    rot_c1, rot_c2, _ = st.columns([1, 1, 6])
                    with rot_c1:
                        if st.button(
                            "↶",
                            key="image_rot_ccw_cam",
                            help="Rotate 90° counterclockwise",
                        ):
                            st.session_state["image_mode_rotation"] = (
                                st.session_state.get("image_mode_rotation", 0) - 90
                            ) % 360
                    with rot_c2:
                        if st.button(
                            "↷",
                            key="image_rot_cw_cam",
                            help="Rotate 90° clockwise",
                        ):
                            st.session_state["image_mode_rotation"] = (
                                st.session_state.get("image_mode_rotation", 0) + 90
                            ) % 360
                    image = _apply_image_mode_rotation(image)
                    show_bytes = (
                        raw if int(st.session_state.get("image_mode_rotation", 0)) % 360 == 0 else None
                    )
                    st.caption(
                        "We auto-apply **camera EXIF** and, if Tesseract is installed, **page orientation** when possible. "
                        "Use **↶** / **↷** to override."
                    )
                    _safe_show_image(
                        image,
                        caption="Captured frame — click **Translate**.",
                        use_column_width=True,
                        original_bytes=show_bytes,
                    )
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                        image.save(tmp_file.name, format="PNG")
                        temp_path = tmp_file.name
    elif mode == "From handwriting":
        up = st.file_uploader(
            "Upload a photo or PDF of handwriting",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tiff", "pdf"],
            label_visibility="collapsed",
            key="handwriting_file_uploader",
        )
        if up is not None:
            raw = up.getvalue()
            tag = hashlib.sha256(raw).hexdigest()
            if st.session_state.get("hw_upload_tag") != tag:
                cleanup_temp_paths(st.session_state.get("hw_png_paths"))
                st.session_state.hw_upload_tag = tag
                low = (up.name or "").lower()
                try:
                    if low.endswith(".pdf"):
                        st.session_state.hw_png_paths = pdf_bytes_to_temp_png_paths(raw)
                    else:
                        image = Image.open(io.BytesIO(raw)).convert("RGB")
                        image_arr = _normalize_to_rgb_uint8_array(image)
                        image = Image.fromarray(image_arr)
                        fd, p = tempfile.mkstemp(suffix=".png")
                        os.close(fd)
                        image.save(p, format="PNG")
                        st.session_state.hw_png_paths = [p]
                except Exception as e:
                    st.warning(f"Could not read file: {e}")
                    st.session_state.hw_png_paths = None
            if st.session_state.hw_png_paths:
                n = len(st.session_state.hw_png_paths)
                st.caption(f"**{n}** page(s) ready. Click **Process** below.")
                try:
                    preview = Image.open(st.session_state.hw_png_paths[0]).convert("RGB")
                    _safe_show_image(preview, caption="First page preview", use_column_width=True)
                except Exception:
                    pass
    elif mode == "From voice":
        if not VOICE_AVAILABLE:
            st.warning("Install `openai-whisper` and ffmpeg for voice input.")
        else:
            audio_file = st.file_uploader("Upload audio", type=['wav', 'mp3', 'm4a', 'webm', 'ogg', 'flac'], label_visibility="collapsed")
            if audio_file is not None:
                ext = os.path.splitext(audio_file.name)[1] or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(audio_file.getvalue())
                    audio_path = tmp.name
                st.audio(audio_path)
                st.markdown(
                    """
                    <div class="wave-placeholder" aria-hidden="true">
                    <span class="wave-bar" style="height:22px"></span><span class="wave-bar" style="height:14px"></span>
                    <span class="wave-bar" style="height:32px"></span><span class="wave-bar" style="height:18px"></span>
                    <span class="wave-bar" style="height:28px"></span><span class="wave-bar" style="height:12px"></span>
                    <span class="wave-bar" style="height:36px"></span><span class="wave-bar" style="height:20px"></span>
                    </div>
                    <p style="color:#64748b;font-size:0.8rem;margin:0;">Ready to transcribe — tap <strong>Translate</strong> below.</p>
                    """,
                    unsafe_allow_html=True,
                )
        st.button(
            "⌨️",
            key="switch_to_write_icon",
            help="Back to writing",
            on_click=_activate_write_input,
        )
    else:
        video_file = st.file_uploader(
            "Upload video",
            type=["mp4", "webm", "avi", "mov", "mkv", "m4v"],
            label_visibility="collapsed",
        )
        if video_file is not None:
            ext = os.path.splitext(video_file.name)[1] or ".mp4"
            if ext.lower() not in (".mp4", ".webm", ".avi", ".mov", ".mkv", ".m4v"):
                ext = ".mp4"
            _upload_tag = None
            try:
                video_file.seek(0)
                _vh = hashlib.sha256()
                for chunk in iter(lambda: video_file.read(1024 * 1024), b""):
                    _vh.update(chunk)
                _upload_tag = _vh.hexdigest()
            except Exception as e:
                st.error(f"Could not read video: {e}. Try a smaller file or different format (MP4, WebM, AVI, MOV).")
                video_path = None
            else:
                if _upload_tag and _upload_tag == st.session_state.get("video_upload_tag"):
                    # Streamlit resends the same upload on every rerun — do not touch disk or clear translation snapshot.
                    _staged = st.session_state.get("_video_staged_path")
                    if _staged and isinstance(_staged, str) and os.path.isfile(_staged):
                        video_path = _staged
                    else:
                        _vkfb = st.session_state.get("_video_retained_for_preview")
                        if _vkfb and isinstance(_vkfb, str) and os.path.isfile(_vkfb):
                            video_path = _vkfb
                else:
                    try:
                        video_file.seek(0)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                            for chunk in iter(lambda: video_file.read(1024 * 1024), b""):
                                tmp.write(chunk)
                            _new_video_path = tmp.name
                    except Exception as e:
                        st.error(
                            f"Could not save video: {e}. Try a smaller file or different format (MP4, WebM, AVI, MOV)."
                        )
                        video_path = None
                    else:
                        _prev_staged = st.session_state.get("_video_staged_path")
                        if (
                            _prev_staged
                            and isinstance(_prev_staged, str)
                            and _prev_staged != _new_video_path
                            and os.path.isfile(_prev_staged)
                        ):
                            try:
                                os.unlink(_prev_staged)
                            except OSError:
                                pass
                        _prev_keep = st.session_state.get("_video_retained_for_preview")
                        if _prev_keep and _prev_keep != _new_video_path and os.path.isfile(_prev_keep):
                            try:
                                os.unlink(_prev_keep)
                            except OSError:
                                pass
                        st.session_state.pop("_video_retained_for_preview", None)
                        st.session_state.pop("video_per_frame_preview", None)
                        st.session_state.pop("_video_annotate_slideshow_cache", None)
                        st.session_state.pop("video_ui_persist_active", None)
                        st.session_state.pop("video_ui_snapshot_v2", None)
                        _ptr = st.session_state.pop("video_ui_translated_retain_path", None)
                        if _ptr and isinstance(_ptr, str) and os.path.isfile(_ptr):
                            try:
                                os.unlink(_ptr)
                            except OSError:
                                pass
                        st.session_state["video_upload_tag"] = _upload_tag
                        st.session_state["_video_staged_path"] = _new_video_path
                        video_path = _new_video_path
                        st.caption("Video ready. Preview it below, then click **Translate**.")
                        st.video(video_path)
        # Streamlit's uploader is often empty on reruns; keep using the temp file path saved after Translate
        # so widget clicks (e.g. Annotated **Next**) still enter the results branch instead of "Upload a video…".
        if video_path is None:
            _vk = st.session_state.get("_video_retained_for_preview")
            if _vk and isinstance(_vk, str) and os.path.isfile(_vk):
                video_path = _vk
        # Uploader is empty on most reruns; still show the retained source file so the player does not vanish after **Next**.
        if video_path and os.path.isfile(video_path) and video_file is None:
            st.caption("Original video (session file — re-upload if this preview stops working).")
            st.video(video_path)

    # Translate button in main area (for image/voice/video) so it's easy to find
    run_clicked_main = False
    if mode == "From image" and temp_path is not None:
        run_clicked_main = st.button("Translate", type="primary", key="translate_main", use_container_width=True)
    elif mode == "From voice" and audio_path:
        run_clicked_main = st.button("Translate", type="primary", key="translate_main", use_container_width=True)
    elif mode == "From video" and video_path is not None:
        run_clicked_main = st.button("Translate", type="primary", key="translate_main", use_container_width=True)
    elif mode == "From handwriting" and st.session_state.get("hw_png_paths"):
        run_clicked_main = st.button(
            "Process handwriting",
            type="primary",
            key="translate_main",
            use_container_width=True,
        )

    should_run = (mode == "From text" and bool(input_text.strip())) or (
        mode in ("From image", "From voice", "From video", "From handwriting") and run_clicked_main
    )
    st.markdown("---")

    with st.container():
        if mode == "From image" and temp_path is None:
            st.info("Upload an image or use the camera, then click **Translate**.")
        elif mode == "From voice" and not audio_path:
            st.info("Upload an audio file (e.g. WAV, MP3) to transcribe and translate.")
        elif mode == "From video" and not video_path and not (
            st.session_state.get("video_ui_persist_active")
            and isinstance(st.session_state.get("video_ui_snapshot_v2"), dict)
        ):
            st.info("Upload a video file to extract text from frames and translate.")
        elif mode == "From handwriting" and not st.session_state.get("hw_png_paths"):
            st.info("Upload a **photo** or **PDF** of handwriting, then click **Process handwriting**.")
        else:
            if should_run:
                with st.spinner("Processing..."):
                    try:
                        _run_t0 = time.perf_counter()
                        _stage_t0 = _run_t0
                        _perf_stages: dict[str, float] = {}

                        def _mark_stage(name: str) -> None:
                            nonlocal _stage_t0
                            now = time.perf_counter()
                            _perf_stages[name] = _perf_stages.get(name, 0.0) + (now - _stage_t0)
                            _stage_t0 = now

                        detected_texts = []
                        detected_languages = []
                        source_language = "en"
                        ocr_results = None
                        per_frame_data = None
                        proc_preview_before_rgb = None
                        proc_preview_after_rgb = None
                        proc_preview_after_openai_rgb = None
                        # Track OCR backend for translation strategy tuning later.
                        ocr_backend_key = "easyocr"
                        overlay_path = None
                        handwriting_prefill_translation = None
                        handwriting_pdf_bytes = None
                        handwriting_png_bytes = None

                        if mode == "From image":
                            initialize_models(
                                ocr_languages,
                                ocr_backend=st.session_state.get("ocr_backend", "easyocr"),
                                trocr_model=st.session_state.get("trocr_model_id", "").strip() or None,
                            )
                            if st.session_state.ocr is None:
                                st.error("OCR failed to load (e.g. first-time download or out of memory). Try refreshing the page or running again.")
                                if temp_path and os.path.isfile(temp_path):
                                    os.unlink(temp_path)
                                st.stop()
                            _mark_stage("image_ocr_model_init")
                            langs_key = tuple(sorted(ocr_languages))
                            ocr_backend_key = (
                                st.session_state.get("ocr_backend", "easyocr") or "easyocr"
                            ).strip().lower()
                            # Keep still-image OCR/preprocessing at the original resolution.
                            # Downscaling can crop or soften fine text in document photos.
                            ocr_max_dim = 0
                            try:
                                file_hash = _sha256_file(temp_path)
                            except OSError:
                                file_hash = None
                            # Perspective rectification produces a new image; do not reuse cached OCR from unwarped runs.
                            img_cache_key = (
                                (
                                    "img",
                                    file_hash,
                                    langs_key,
                                    ocr_backend_key,
                                    preprocess,
                                    ocr_max_dim,
                                    bool(photo_dewarp),
                                    bool(ocr_small_text),
                                    bool(ocr_high_recall),
                                    bool(ocr_openai_preprocess),
                                    _ocr_preprocess_env_fingerprint(),
                                )
                                if file_hash and not photo_dewarp
                                else None
                            )
                            cached_img = _ocr_cache_get(img_cache_key) if img_cache_key else None
                            overlay_path = temp_path
                            _sync_ocr_preprocess_settings(
                                openai_preprocess=bool(ocr_openai_preprocess),
                                local_preprocess=bool(preprocess),
                            )
                            if cached_img is not None:
                                cached_rows = _normalize_ocr_rows(
                                    cached_img.get("ocr_results_raw", [])
                                )
                                ocr_results = _sort_ocr_results_reading_order(
                                    cached_rows,
                                    rtl=(target_lang_code == "ar"),
                                )
                                ocr_results = _drop_container_ocr_boxes(ocr_results)
                                st.caption(
                                    "Using **cached OCR** (same file and OCR options). "
                                    "Change OCR options or re-upload the image to scan again."
                                )
                            else:
                                _ocr = st.session_state.ocr
                                if ocr_openai_preprocess and not (_resolve_openai_key() or "").strip():
                                    st.warning(
                                        "OpenAI preprocess is on but **OPENAI_API_KEY** is missing — "
                                        "falling back to local preprocessing only."
                                    )
                                _ocr_extract_kw = dict(
                                    preprocess=preprocess,
                                    max_dim=ocr_max_dim,
                                    small_text_boost=ocr_small_text,
                                    high_recall=ocr_high_recall,
                                )
                                _ocr_spinner = (
                                    "OpenAI image preprocess + OCR (this may take 1–2 minutes)…"
                                    if ocr_openai_preprocess
                                    else "Running OCR…"
                                )
                                if hasattr(_ocr, "extract_text_for_overlay"):
                                    with st.spinner(_ocr_spinner):
                                        ocr_results, overlay_path = _ocr.extract_text_for_overlay(
                                            temp_path,
                                            dewarp_screen=photo_dewarp,
                                            **_ocr_extract_kw,
                                        )
                                else:
                                    if photo_dewarp:
                                        st.warning(
                                            "Perspective straightening is unavailable until the app reloads the latest "
                                            "**ocr_translator.py** (restart Streamlit: stop the process and run `streamlit run app.py` again)."
                                        )
                                    with st.spinner(_ocr_spinner):
                                        ocr_results = _ocr.extract_text(
                                            temp_path,
                                            **_ocr_extract_kw,
                                        )
                                    overlay_path = temp_path
                                if photo_dewarp and overlay_path != temp_path:
                                    st.caption(
                                        "**Perspective straightening** applied — boxes and translated image use the rectified view."
                                    )
                                ocr_results = _normalize_ocr_rows(ocr_results)
                                if not ocr_results:
                                    paddle_diag = ""
                                    if ocr_backend_key == "paddleocr":
                                        paddle_diag = str(getattr(_ocr, "last_ocr_error", "") or "").strip()
                                    if paddle_diag:
                                        st.error(
                                            "PaddleOCR returned no text.\n\n"
                                            f"Reason: {paddle_diag}"
                                        )
                                    else:
                                        st.warning(
                                            "No text detected in the image after OCR. "
                                            "Include every script you need under **OCR languages** (e.g. Arabic + English), "
                                            "or turn off **Preprocess image before OCR** if it blurs the photo. "
                                            "Tune: ``OCR_MIN_CONFIDENCE``, ``OCR_EMPTY_RESCUE_MIN_CONF`` (default rescue floor 0.12)."
                                        )
                                    if temp_path and os.path.isfile(temp_path):
                                        os.unlink(temp_path)
                                    if overlay_path != temp_path and overlay_path and os.path.isfile(overlay_path):
                                        os.unlink(overlay_path)
                                    st.stop()
                                if img_cache_key:
                                    _ocr_cache_set(
                                        img_cache_key,
                                        _pack_preprocess_preview_into_payload(
                                            {"ocr_results_raw": ocr_results}
                                        ),
                                    )
                                ocr_results = _sort_ocr_results_reading_order(
                                    ocr_results, rtl=(target_lang_code == "ar")
                                )
                                ocr_results = _drop_container_ocr_boxes(ocr_results)
                            if not ocr_results:
                                st.warning(
                                    "No text detected in the image after OCR. "
                                    "Include every script you need under **OCR languages** (e.g. Arabic + English), "
                                    "or turn off **Preprocess image before OCR** if it blurs the photo. "
                                    "If you recently changed OCR settings, **re-upload** the image (cached runs may still be empty). "
                                    "Tune: ``OCR_MIN_CONFIDENCE``, ``OCR_EMPTY_RESCUE_MIN_CONF``."
                                )
                                if temp_path and os.path.isfile(temp_path):
                                    try:
                                        os.unlink(temp_path)
                                    except OSError:
                                        pass
                                if overlay_path != temp_path and overlay_path and os.path.isfile(overlay_path):
                                    try:
                                        os.unlink(overlay_path)
                                    except OSError:
                                        pass
                                st.stop()
                            _mark_stage("image_ocr_pipeline")
                            for row in ocr_results:
                                text, lang = _as_text(row[0]), _as_text(row[2])
                                detected_texts.append(text)
                                detected_languages.append(
                                    _infer_translation_source_lang(
                                        text, ocr_languages, lang, target_lang_code
                                    )
                                )
                            source_language = detected_languages[0] if detected_languages else "en"
                            if len(set(detected_languages)) == 1:
                                source_language = detected_languages[0]
                            if overlay_path and os.path.isfile(overlay_path):
                                try:
                                    (
                                        proc_preview_before_rgb,
                                        proc_preview_after_rgb,
                                        proc_preview_after_openai_rgb,
                                    ) = _load_preprocess_previews_for_image(
                                        overlay_path,
                                        local_preprocess=bool(preprocess),
                                        openai_preprocess=bool(ocr_openai_preprocess),
                                        ocr_max_dim=ocr_max_dim,
                                        ocr_small_text=ocr_small_text,
                                        ocr_high_recall=ocr_high_recall,
                                        cached_payload=cached_img,
                                    )
                                    st.session_state["last_openai_preprocess_error"] = ""
                                    _ocr_pf = getattr(st.session_state.get("ocr"), "last_ocr_preprocess_profile", None)
                                    if isinstance(_ocr_pf, dict) and _ocr_pf.get("openai_skip"):
                                        st.session_state["last_openai_preprocess_error"] = str(
                                            _ocr_pf.get("openai_skip")
                                        )
                                except Exception as _prev_exc:
                                    proc_preview_before_rgb = None
                                    proc_preview_after_rgb = None
                                    proc_preview_after_openai_rgb = None
                                    st.session_state["last_openai_preprocess_error"] = str(_prev_exc)

                        elif mode == "From handwriting":
                            paths = list(st.session_state.get("hw_png_paths") or [])
                            if not paths:
                                st.error("No pages to process. Upload a file again.")
                                st.stop()
                            oai_key = _resolve_openai_key()
                            if not oai_key:
                                st.error(
                                    "Handwriting mode needs **OPENAI_API_KEY** "
                                    "(e.g. in `.env` or Streamlit secrets)."
                                )
                                st.stop()
                            ocr_results = []
                            overlay_path = paths[0]
                            st.session_state.hw_last_preview_path = paths[0]
                            page_chunks: list[str] = []
                            page_trans_chunks: list[str] = []
                            first_iso = "en"
                            n_paths = len(paths)
                            any_transcript = False
                            first_non_empty_path = paths[0]
                            arabic_quality_hint = target_lang_code == "ar" or any(
                                (str(c).strip().lower() == "ar") for c in (ocr_languages or [])
                            )
                            if n_paths <= 1:
                                for page_idx, p in enumerate(paths):
                                    with st.spinner(
                                        f"Reading handwriting (OpenAI Vision) — page {page_idx + 1} / {n_paths}…"
                                    ):
                                        body, tr_body, iso_guess = transcribe_and_translate_handwriting_from_image(
                                            p,
                                            target_lang,
                                            api_key=oai_key,
                                            prefer_arabic_quality=arabic_quality_hint,
                                        )
                                    if page_idx == 0:
                                        first_iso = iso_guess or "en"
                                    b = (body or "").strip()
                                    tb = (tr_body or "").strip()
                                    if b:
                                        any_transcript = True
                                        if not first_non_empty_path:
                                            first_non_empty_path = p
                                    if n_paths > 1:
                                        page_chunks.append(f"## Page {page_idx + 1}\n\n{b}")
                                        page_trans_chunks.append(f"## Page {page_idx + 1}\n\n{tb}")
                                    else:
                                        page_chunks.append(b)
                                        page_trans_chunks.append(tb)
                            else:
                                # Parallel page OCR can substantially cut total latency for PDFs.
                                try:
                                    hw_workers = int(os.environ.get("OPENAI_HANDWRITING_MAX_WORKERS", "2"))
                                except ValueError:
                                    hw_workers = 2
                                hw_workers = max(1, min(4, hw_workers, n_paths))
                                ordered_pages: list[tuple[str, str, str] | None] = [None] * n_paths
                                with st.spinner(
                                    f"Reading handwriting (OpenAI Vision) — {n_paths} pages in parallel…"
                                ):
                                    with ThreadPoolExecutor(max_workers=hw_workers) as ex:
                                        fut_to_idx = {
                                            ex.submit(
                                                transcribe_and_translate_handwriting_from_image,
                                                p,
                                                target_lang,
                                                oai_key,
                                                None,
                                                arabic_quality_hint,
                                            ): i
                                            for i, p in enumerate(paths)
                                        }
                                        for fut in as_completed(fut_to_idx):
                                            i = fut_to_idx[fut]
                                            try:
                                                body, tr_body, iso_guess = fut.result()
                                            except Exception:
                                                body, tr_body, iso_guess = "", "", "en"
                                            ordered_pages[i] = (
                                                body or "",
                                                tr_body or "",
                                                iso_guess or "en",
                                            )
                                for page_idx, p in enumerate(paths):
                                    pair = ordered_pages[page_idx] or ("", "", "en")
                                    body, tr_body, iso_guess = pair
                                    if page_idx == 0:
                                        first_iso = iso_guess or "en"
                                    b = (body or "").strip()
                                    tb = (tr_body or "").strip()
                                    if b:
                                        any_transcript = True
                                        if not first_non_empty_path:
                                            first_non_empty_path = p
                                    if n_paths > 1:
                                        page_chunks.append(f"## Page {page_idx + 1}\n\n{b}")
                                        page_trans_chunks.append(f"## Page {page_idx + 1}\n\n{tb}")
                                    else:
                                        page_chunks.append(b)
                                        page_trans_chunks.append(tb)
                            full_doc = "\n\n".join(page_chunks).strip()
                            full_doc_trans = "\n\n".join(page_trans_chunks).strip()
                            if not any_transcript:
                                st.warning(
                                    "No handwriting was recognized on these pages. "
                                    "Try a clearer photo or higher resolution."
                                )
                                st.stop()
                            if full_doc_trans:
                                handwriting_prefill_translation = full_doc_trans
                            source_language = _iso6391_to_app(first_iso)
                            ocr_languages = list(dict.fromkeys([source_language, "en"]))
                            detected_texts = [full_doc]
                            detected_languages = [source_language]
                            st.session_state.hw_photo_source_path = first_non_empty_path
                            _mark_stage("handwriting_ocr_plus_prefill_translation")

                        elif mode == "From video":
                            initialize_models(
                                ocr_languages,
                                ocr_backend=st.session_state.get("ocr_backend", "easyocr"),
                                trocr_model=st.session_state.get("trocr_model_id", "").strip() or None,
                            )
                            if st.session_state.ocr is None:
                                st.error("OCR failed to load (e.g. first-time download or out of memory). Try refreshing the page or running again.")
                                if video_path and os.path.isfile(video_path):
                                    os.unlink(video_path)
                                st.stop()
                            _mark_stage("video_ocr_models_ready")
                            per_frame_data = None
                            ocr_results = []
                            langs_key = tuple(sorted(ocr_languages))
                            ocr_backend_key = (
                                st.session_state.get("ocr_backend", "easyocr") or "easyocr"
                            ).strip().lower()
                            try:
                                vid_hash = _sha256_file(video_path)
                            except OSError:
                                vid_hash = None
                            if generate_video:
                                if ocr_small_text:
                                    ocr_max_dim = int(os.environ.get("OCR_SMALL_TEXT_VIDEO_MAX_DIM", "1600"))
                                elif ocr_high_recall:
                                    ocr_max_dim = int(
                                        os.environ.get(
                                            "OCR_HIGH_RECALL_VIDEO_MAX_DIM",
                                            "1920" if not fast_ocr else "960",
                                        )
                                    )
                                elif fast_ocr:
                                    ocr_max_dim = 720
                                else:
                                    ocr_max_dim = 0
                                vf_cache_key = (
                                    "vframes",
                                    vid_hash,
                                    langs_key,
                                    ocr_backend_key,
                                    preprocess,
                                    float(sample_interval_sec),
                                    int(max_video_frames),
                                    bool(fast_ocr),
                                    bool(ocr_small_text),
                                    bool(ocr_high_recall),
                                    _ocr_preprocess_env_fingerprint(),
                                ) if vid_hash else None
                                cached_vf = _ocr_cache_get(vf_cache_key) if vf_cache_key else None
                                if cached_vf is not None:
                                    rtl = target_lang_code == "ar"
                                    per_frame_data = [
                                        (
                                            frame_idx,
                                            _sort_ocr_results_reading_order(
                                                _normalize_ocr_rows(results_for_frame),
                                                rtl=rtl,
                                            ),
                                        )
                                        for frame_idx, results_for_frame in cached_vf["per_frame_data_raw"]
                                    ]
                                    ocr_results = [r for _, results in per_frame_data for r in results]
                                    st.caption(
                                        "Using **cached OCR** (same video and sampling options). "
                                        "Change sampling, OCR languages, preprocess, or re-upload to scan again."
                                    )
                                else:
                                    ocr_results, per_frame_data = st.session_state.ocr.extract_text_from_video_with_frames(
                                        video_path,
                                        sample_interval_sec=sample_interval_sec,
                                        preprocess=preprocess,
                                        max_frames=max_video_frames,
                                        initial_dense_sec=0.5 if fast_ocr else 1.0,
                                        initial_dense_interval_sec=0.5,
                                        ocr_max_dim=ocr_max_dim,
                                        small_text_boost=ocr_small_text,
                                        high_recall=ocr_high_recall,
                                    )
                                    if per_frame_data and vf_cache_key:
                                        norm_pf = [
                                            (fi, _normalize_ocr_rows(rf))
                                            for fi, rf in per_frame_data
                                        ]
                                        _ocr_cache_set(
                                            vf_cache_key,
                                            {"per_frame_data_raw": copy.deepcopy(norm_pf)},
                                        )
                                        per_frame_data = norm_pf
                                    if per_frame_data:
                                        rtl = target_lang_code == "ar"
                                        per_frame_data = [
                                            (
                                                frame_idx,
                                                _sort_ocr_results_reading_order(
                                                    _normalize_ocr_rows(results_for_frame),
                                                    rtl=rtl,
                                                ),
                                            )
                                            for frame_idx, results_for_frame in per_frame_data
                                        ]
                                        ocr_results = [r for _, results in per_frame_data for r in results]
                            else:
                                va_cache_key = (
                                    "vagg",
                                    vid_hash,
                                    langs_key,
                                    ocr_backend_key,
                                    preprocess,
                                    float(sample_interval_sec),
                                    bool(ocr_small_text),
                                    bool(ocr_high_recall),
                                    int(max_video_frames),
                                    _ocr_preprocess_env_fingerprint(),
                                ) if vid_hash else None
                                cached_va = _ocr_cache_get(va_cache_key) if va_cache_key else None
                                raw_pf = cached_va.get("per_frame_data_raw") if cached_va else None
                                if cached_va is not None and raw_pf:
                                    rtl = target_lang_code == "ar"
                                    per_frame_data = [
                                        (
                                            frame_idx,
                                            _sort_ocr_results_reading_order(
                                                _normalize_ocr_rows(results_for_frame),
                                                rtl=rtl,
                                            ),
                                        )
                                        for frame_idx, results_for_frame in raw_pf
                                    ]
                                    ocr_results = [r for _, results in per_frame_data for r in results]
                                    st.caption(
                                        "Using **cached OCR** (same video and options). "
                                        "Change sample interval, OCR languages, preprocess, or re-upload to scan again."
                                    )
                                else:
                                    if cached_va is not None and va_cache_key:
                                        _ocr_cache_pop(va_cache_key)
                                        st.caption(
                                            "Re-scanning video once: cached OCR had no per-frame box list, "
                                            "which is required for the **Annotated** tab frame slider."
                                        )
                                    ocr_results, per_frame_data = (
                                        st.session_state.ocr.extract_text_from_video_with_frames(
                                            video_path,
                                            sample_interval_sec=sample_interval_sec,
                                            preprocess=preprocess,
                                            max_frames=max_video_frames,
                                            initial_dense_sec=0,
                                            initial_dense_interval_sec=0.25,
                                            ocr_max_dim=0,
                                            small_text_boost=ocr_small_text,
                                            high_recall=ocr_high_recall,
                                        )
                                    )
                                    ocr_results = _normalize_ocr_rows(ocr_results)
                                    if per_frame_data and va_cache_key:
                                        norm_pf = [
                                            (fi, _normalize_ocr_rows(rf)) for fi, rf in per_frame_data
                                        ]
                                        _ocr_cache_set(
                                            va_cache_key,
                                            {
                                                "ocr_results_raw": list(ocr_results),
                                                "per_frame_data_raw": copy.deepcopy(norm_pf),
                                            },
                                        )
                                        per_frame_data = norm_pf
                                    if per_frame_data:
                                        rtl = target_lang_code == "ar"
                                        per_frame_data = [
                                            (
                                                frame_idx,
                                                _sort_ocr_results_reading_order(
                                                    results_for_frame,
                                                    rtl=rtl,
                                                ),
                                            )
                                            for frame_idx, results_for_frame in per_frame_data
                                        ]
                                        ocr_results = [r for _, results in per_frame_data for r in results]
                            if not ocr_results:
                                st.warning("No text detected in the video. Try a different file or increase frame sampling.")
                                if video_path and os.path.isfile(video_path):
                                    os.unlink(video_path)
                                st.stop()
                            for row in ocr_results:
                                text, lang = _as_text(row[0]), _as_text(row[2])
                                detected_texts.append(text)
                                detected_languages.append(
                                    _infer_translation_source_lang(
                                        text, ocr_languages, lang, target_lang_code
                                    )
                                )
                            source_language = detected_languages[0] if detected_languages else "en"
                            if len(set(detected_languages)) == 1:
                                source_language = detected_languages[0]
                            _mark_stage("video_ocr_extract")
                            if per_frame_data:
                                st.session_state["video_per_frame_preview"] = copy.deepcopy(per_frame_data)
                            else:
                                st.session_state.pop("video_per_frame_preview", None)
                            if video_path and os.path.isfile(video_path):
                                _prev_keep = st.session_state.get("_video_retained_for_preview")
                                if _prev_keep and _prev_keep != video_path and os.path.isfile(_prev_keep):
                                    try:
                                        os.unlink(_prev_keep)
                                    except OSError:
                                        pass
                                st.session_state["_video_retained_for_preview"] = video_path

                        elif mode == "From voice":
                            transcribed = transcribe_audio(
                                audio_path,
                                language=input_lang_code,
                                model_size="base",
                            )
                            if not transcribed:
                                st.warning("No speech detected in the audio.")
                                if audio_path and os.path.isfile(audio_path):
                                    os.unlink(audio_path)
                                st.stop()
                            detected_texts = [transcribed]
                            detected_languages = [input_lang_code or "en"]
                            source_language = input_lang_code or "en"
                            if audio_path and os.path.isfile(audio_path):
                                os.unlink(audio_path)

                        else:
                            detected_texts = [input_text.strip()]
                            detected_languages = [input_lang_code or "en"]
                            source_language = input_lang_code or "en"

                        # Translate unique lines once, map back — video OCR repeats the same subtitle on every sampled frame
                        unique_texts: list = []
                        _seen_seg: set = set()
                        for t in detected_texts:
                            dk = _translation_dedupe_key(mode, t)
                            if dk not in _seen_seg:
                                _seen_seg.add(dk)
                                unique_texts.append(t)
                        # First occurrence of each line → EasyOCR raw lang code (de/en/…) for NLLB source
                        _ocr_fb_by_dk: dict = {}
                        if (
                            mode in ("From image", "From video", "From handwriting")
                            and ocr_results
                            and len(detected_texts) == len(ocr_results)
                        ):
                            for i, t in enumerate(detected_texts):
                                dk = _translation_dedupe_key(mode, t)
                                if dk not in _ocr_fb_by_dk:
                                    raw = ocr_results[i][2] if len(ocr_results[i]) > 2 else "en"
                                    _ocr_fb_by_dk[dk] = (raw or "en").lower() if isinstance(raw, str) else "en"

                        def _easyocr_fallback_for(u: str) -> str:
                            dk = _translation_dedupe_key(mode, u)
                            return _ocr_fb_by_dk.get(dk, source_language)

                        provider = st.session_state.get("translation_provider", "google_cloud")
                        if provider not in ("local_nllb", "google_cloud", "openai_chat"):
                            provider = "google_cloud"
                        if mode in ("From image", "From video", "From text", "From voice"):
                            provider = "google_cloud"
                        use_google_tr = provider == "google_cloud"
                        # OpenAI translation for image/photo mode only.
                        use_openai_tr = (
                            mode == "From image"
                            and generate_photo
                            and bool(st.session_state.get("openai_photo_only"))
                        ) or (mode == "From handwriting" and provider == "openai_chat")
                        if not use_google_tr and not use_openai_tr:
                            _ensure_local_translator()

                        _pr_ar = bool(preserve_arabic_names)

                        if mode == "From video":
                            _mark_stage("video_prepare_translation")

                        if use_openai_tr:
                            oai_key = _resolve_openai_key()
                            if not oai_key:
                                st.error(
                                    "OpenAI translation requires an API key. "
                                    "Set **OPENAI_API_KEY**, use Streamlit secrets, or paste the key under photo options."
                                )
                                if temp_path and os.path.isfile(temp_path):
                                    os.unlink(temp_path)
                                if video_path and os.path.isfile(video_path):
                                    os.unlink(video_path)
                                if audio_path and os.path.isfile(audio_path):
                                    os.unlink(audio_path)
                                st.stop()
                            oai_model = (
                                (st.session_state.get("openai_model_photo") or "").strip()
                                or os.environ.get("OPENAI_TRANSLATION_MODEL", "gpt-4o-mini")
                            )
                            unique_translations = [None] * len(unique_texts)
                            batched_indices: list[int] = []
                            batched_texts: list[str] = []
                            batched_sources: list[str] = []
                            if (
                                mode == "From handwriting"
                                and handwriting_prefill_translation
                                and unique_texts
                            ):
                                unique_translations[0] = {
                                    "original_text": unique_texts[0],
                                    "translated_text": handwriting_prefill_translation,
                                    "source_language": _infer_translation_source_lang(
                                        unique_texts[0],
                                        ocr_languages,
                                        _easyocr_fallback_for(unique_texts[0]),
                                        target_lang_code,
                                    ),
                                    "target_language": target_lang_code,
                                    "confidence": None,
                                }
                                start_i = 1
                            else:
                                start_i = 0
                            for i, u in enumerate(unique_texts[start_i:], start=start_i):
                                pb = _translation_try_preserve_arabic(
                                    u, target_lang_code, _pr_ar
                                )
                                if pb is not None:
                                    unique_translations[i] = pb
                                    continue
                                batched_indices.append(i)
                                batched_texts.append(u)
                                batched_sources.append(
                                    _infer_translation_source_lang(
                                        u,
                                        ocr_languages,
                                        _easyocr_fallback_for(u),
                                        target_lang_code,
                                    )
                                )
                            if batched_texts:
                                # High-impact speedup: one OpenAI call for all untranslated segments.
                                source_for_batch = (
                                    batched_sources[0]
                                    if len(set(batched_sources)) == 1
                                    else None
                                )
                                batched_rows = translate_batch_openai(
                                    batched_texts,
                                    target_lang_code,
                                    source_for_batch,
                                    api_key=oai_key,
                                    model=oai_model,
                                    fast=fast_translation,
                                )
                                for idx, row in zip(batched_indices, batched_rows):
                                    unique_translations[idx] = row
                            # Preserve behavior if batch fallback returns short output.
                            for i, row in enumerate(unique_translations):
                                if row is None:
                                    u = unique_texts[i]
                                    unique_translations[i] = translate_text_openai(
                                        u,
                                        target_lang_code,
                                        _infer_translation_source_lang(
                                            u,
                                            ocr_languages,
                                            _easyocr_fallback_for(u),
                                            target_lang_code,
                                        ),
                                        api_key=oai_key,
                                        model=oai_model,
                                        fast=fast_translation,
                                    )
                            _mark_stage("text_translation_openai")
                        elif use_google_tr:
                            unique_translations = [None] * len(unique_texts)
                            _google_fallback_warned = False
                            batched_indices: list[int] = []
                            batched_texts: list[str] = []
                            batched_sources: list[str] = []
                            for i, u in enumerate(unique_texts):
                                pb = _translation_try_preserve_arabic(
                                    u, target_lang_code, _pr_ar
                                )
                                if pb is not None:
                                    unique_translations[i] = pb
                                    continue
                                batched_indices.append(i)
                                batched_texts.append(u)
                                batched_sources.append(
                                    _infer_translation_source_lang(
                                        u,
                                        ocr_languages,
                                        _easyocr_fallback_for(u),
                                        target_lang_code,
                                    )
                                )
                            if batched_texts:
                                warm_google_translate_client(project_id=google_project_id)
                                # Prefer one dominant source language for speed (auto-detect per segment is slower).
                                _counts = {}
                                for _s in batched_sources:
                                    _k = (_s or "").strip().lower() or "en"
                                    _counts[_k] = _counts.get(_k, 0) + 1
                                source_for_batch = (
                                    max(_counts.items(), key=lambda kv: kv[1])[0]
                                    if _counts
                                    else (source_language or "en")
                                )
                                try:
                                    # TrOCR can output many long/noisy segments; one huge joined request may
                                    # become a slow tail (~tens of seconds). Prefer chunked list-batch there.
                                    prefer_batch_for_trocr = (
                                        mode in ("From image", "From video")
                                        and ocr_backend_key in ("trocr_hybrid", "trocr_only")
                                        and len(batched_texts) >= 8
                                    )
                                    use_marker_joined = (
                                        (not prefer_batch_for_trocr)
                                        and google_translation_should_use_joined(
                                            batched_texts, source_for_batch
                                        )
                                    )

                                    _google_seg_perf: dict = {}

                                    def _nllb_fallback_batch(texts_only, tl_code, sl_code):
                                        _ensure_local_translator()
                                        return st.session_state.translator.translate_batch(
                                            texts_only,
                                            tl_code,
                                            sl_code,
                                            fast=fast_translation,
                                        )

                                    batched_rows = translate_ocr_segments_google(
                                        batched_texts,
                                        target_lang_code,
                                        source_for_batch,
                                        project_id=google_project_id,
                                        fast=fast_translation,
                                        prefer_joined=use_marker_joined,
                                        fallback_batch_fn=_nllb_fallback_batch,
                                        perf_stats=_google_seg_perf,
                                    )
                                    if _google_seg_perf:
                                        for _gk, _gv in _google_seg_perf.items():
                                            if _gk == "strategy":
                                                continue
                                            if _gk == "segments_in" and isinstance(_gv, int):
                                                _perf_stages["text_translation_segment_count"] = float(_gv)
                                            elif _gk == "http_calls" and isinstance(_gv, int):
                                                _perf_stages["text_translation_api_calls"] = float(_gv)
                                    if len(batched_rows) != len(batched_texts):
                                        batched_rows = translate_batch_google(
                                            batched_texts,
                                            target_lang_code,
                                            source_for_batch,
                                            project_id=google_project_id,
                                            fast=fast_translation,
                                            fallback_batch_fn=_nllb_fallback_batch,
                                            perf_stats=_google_seg_perf,
                                        )
                                    for idx, row in zip(batched_indices, batched_rows):
                                        unique_translations[idx] = row
                                except Exception:
                                    # Keep text mode usable when Google credentials/network fail.
                                    _ensure_local_translator()
                                    if not _google_fallback_warned:
                                        st.warning(
                                            "Google Cloud translation is unavailable right now. "
                                            "Falling back to local NLLB for this run."
                                        )
                                        _google_fallback_warned = True
                            if any(row is None for row in unique_translations):
                                _ensure_local_translator()
                            for i, row in enumerate(unique_translations):
                                if row is None:
                                    u = unique_texts[i]
                                    unique_translations[i] = st.session_state.translator.translate_text(
                                        u,
                                        target_lang_code,
                                        _infer_translation_source_lang(
                                            u,
                                            ocr_languages,
                                            _easyocr_fallback_for(u),
                                            target_lang_code,
                                        ),
                                        fast=fast_translation,
                                    )
                            _mark_stage("text_translation")
                        elif mode in ("From image", "From video", "From handwriting"):
                            unique_translations = []
                            for u in unique_texts:
                                pb = _translation_try_preserve_arabic(
                                    u, target_lang_code, _pr_ar
                                )
                                if pb is not None:
                                    unique_translations.append(pb)
                                else:
                                    unique_translations.append(
                                        st.session_state.translator.translate_text(
                                            u,
                                            target_lang_code,
                                            _infer_translation_source_lang(
                                                u,
                                                ocr_languages,
                                                _easyocr_fallback_for(u),
                                                target_lang_code,
                                            ),
                                            fast=fast_translation,
                                        )
                                    )
                            _mark_stage("text_translation")
                        else:
                            unique_translations = []
                            for u in unique_texts:
                                pb = _translation_try_preserve_arabic(
                                    u, target_lang_code, _pr_ar
                                )
                                if pb is not None:
                                    unique_translations.append(pb)
                                else:
                                    unique_translations.append(
                                        st.session_state.translator.translate_text(
                                            u,
                                            target_lang_code,
                                            source_language,
                                            fast=fast_translation,
                                        )
                                    )
                            _mark_stage("text_translation")
                        _tr_by_key = {}
                        for u, tr in zip(unique_texts, unique_translations):
                            dk = _translation_dedupe_key(mode, u)
                            _tr_by_key[dk] = tr
                        _ocr_warn_off = os.environ.get("TRANSLATION_OCR_WARN_OFF", "").strip().lower() in (
                            "1",
                            "true",
                            "yes",
                        )
                        _tr_warn_thresh: Optional[float]
                        if _ocr_warn_off:
                            _tr_warn_thresh = None
                        else:
                            try:
                                # EasyOCR often scores 0.2–0.45 on valid night shots / Arabic — 0.38 was
                                # too aggressive and paired with low scores even when MT succeeded.
                                _tr_warn_thresh = float(
                                    (os.environ.get("TRANSLATION_OCR_WARN_CONF") or "").strip() or "0.24"
                                )
                            except ValueError:
                                _tr_warn_thresh = 0.24
                        translations = []
                        for i, t in enumerate(detected_texts):
                            dk = _translation_dedupe_key(mode, t)
                            base = {**_tr_by_key[dk]}
                            if (
                                _tr_warn_thresh is not None
                                and mode in ("From image", "From video", "From handwriting")
                                and ocr_results
                                and i < len(ocr_results)
                            ):
                                row = ocr_results[i]
                                try:
                                    conf = float(row[3]) if len(row) > 3 else 0.55
                                except (TypeError, ValueError, IndexError):
                                    conf = 0.55
                                if conf < _tr_warn_thresh:
                                    tt = (base.get("translated_text") or "").strip()
                                    orig_st = (t or "").strip()
                                    # Only flag when MT actually produced different text; do not prefix
                                    # when translation failed and target still equals source (misleading).
                                    if (
                                        tt
                                        and orig_st
                                        and tt != orig_st
                                        and _loose_text_key(tt) != _loose_text_key(orig_st)
                                        and not tt.startswith("(OCR uncertain)")
                                    ):
                                        base["translated_text"] = f"(OCR uncertain) {tt}"
                            translations.append({**base, "original_text": t})

                        # For translated photo: one translated string per box, in the same order
                        translated_texts_for_photo = None
                        if mode == "From image" and generate_photo and ocr_results:
                            if len(translations) == len(ocr_results):
                                translated_texts_for_photo = [
                                    _as_text(t.get("translated_text")) for t in translations
                                ]
                            else:
                                # TrOCR/easy preprocessing can change segment counts. Map each OCR row text to
                                # the best known translated segment key so local photo rendering still works.
                                mapped_txts = []
                                for row in (ocr_results or []):
                                    src_txt = _as_text(row[0]) if isinstance(row, (list, tuple)) and len(row) > 0 else ""
                                    dk = _loose_text_key(src_txt)
                                    if len(dk) < 2:
                                        dk = _dedupe_key(src_txt).lower()
                                    tr_obj = _tr_by_key.get(dk)
                                    mapped_txts.append(_as_text((tr_obj or {}).get("translated_text")))
                                if mapped_txts:
                                    translated_texts_for_photo = mapped_txts
                            # Segments already match box order (boxes were sorted in reading order, including RTL for Arabic)

                        _history_join = _paragraph_join_deduped if mode == "From video" else _paragraph_join
                        history_original_block = _history_join(detected_texts)
                        history_translated_block = _history_join([t["translated_text"] for t in translations])

                        results_json = {
                            "detected_texts": detected_texts,
                            "translations": [
                                {
                                    "original": t["original_text"],
                                    "translated": t["translated_text"],
                                    "source_lang": t["source_language"],
                                    "target_lang": t["target_language"],
                                }
                                for t in translations
                            ],
                            **_text_count_stats(history_original_block),
                        }
                        if mode in ("From text", "From voice"):
                            history_source_lang = input_lang
                        else:
                            history_source_lang = ", ".join(sorted(set(detected_languages))) if detected_languages else "Auto"

                        st.session_state.history.insert(
                            0,
                            {
                                "num_segments": len(detected_texts),
                                "target_lang": target_lang,
                                "preview": (detected_texts[0][:80] if detected_texts else ""),
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                "source_lang": history_source_lang,
                                "original_text": history_original_block,
                                "translated_text": history_translated_block,
                            },
                        )
                        _save_history(st.session_state.history)

                        if mode == "From video":
                            _mark_stage("video_assemble_outputs")

                        annotated = None
                        translated_image = None
                        translated_video_path = None
                        translated_video_format = "video/mp4"
                        translated_video_ext = "mp4"
                        if mode == "From handwriting":
                            orig_block = _paragraph_join(detected_texts)
                            trans_block = _paragraph_join(
                                [t["translated_text"] for t in translations]
                            )
                            _pk = _resolve_openai_key()
                            _polish_on = os.environ.get("OPENAI_HANDWRITING_POLISH", "").strip().lower() in (
                                "1",
                                "true",
                                "yes",
                            )
                            if _pk and _polish_on:
                                try:
                                    orig_block, trans_block = polish_bilingual_for_export(
                                        orig_block,
                                        trans_block,
                                        target_lang,
                                        api_key=_pk,
                                    )
                                except Exception as _pe:
                                    st.warning(f"OpenAI export polish skipped: {_pe}")
                            try:
                                handwriting_pdf_bytes = build_bilingual_pdf_bytes(
                                    "Handwriting translation",
                                    orig_block,
                                    trans_block,
                                    target_lang,
                                )
                            except Exception as _pdf_e:
                                st.warning(f"PDF export unavailable: {_pdf_e}")
                            try:
                                handwriting_png_bytes = build_bilingual_png_bytes(
                                    "Handwriting translation",
                                    orig_block,
                                    trans_block,
                                    target_lang,
                                )
                                translated_image = np.array(
                                    Image.open(io.BytesIO(handwriting_png_bytes)).convert(
                                        "RGB"
                                    )
                                )
                            except Exception as _png_e:
                                st.warning(f"PNG summary unavailable: {_png_e}")
                        if mode == "From video" and generate_video and per_frame_data and len(ocr_results) == len(detected_texts):
                            try:
                                # One translated string per detected segment, already in reading order
                                translated_list = [t["translated_text"] for t in translations]
                                # Boxes already in reading order (RTL for Arabic), segments match
                                if len(translated_list) == len(ocr_results):
                                    encode_err = None
                                    # MP4 plays in the in-app browser; AVI/MJPEG often does not — try MP4 first, then AVI.
                                    for ext, mime in (("mp4", "video/mp4"), ("avi", "video/x-msvideo")):
                                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                                            out_path = tmp.name
                                        try:
                                            _prev_crf = os.environ.get("VIDEO_EXPORT_CRF")
                                            _prev_preset = os.environ.get("VIDEO_EXPORT_PRESET")
                                            _prev_font_scale = os.environ.get("VIDEO_TRANSLATED_FONT_SCALE")
                                            os.environ["VIDEO_EXPORT_CRF"] = str(video_export_crf)
                                            os.environ["VIDEO_EXPORT_PRESET"] = str(video_export_preset)
                                            os.environ["VIDEO_TRANSLATED_FONT_SCALE"] = str(video_translated_font_scale)
                                            try:
                                                st.session_state.ocr.render_translated_video(
                                                    video_path, per_frame_data, translated_list, out_path,
                                                    fast_render=(fast_ocr and not video_quality_enhance),
                                                )
                                            finally:
                                                if _prev_crf is None:
                                                    os.environ.pop("VIDEO_EXPORT_CRF", None)
                                                else:
                                                    os.environ["VIDEO_EXPORT_CRF"] = _prev_crf
                                                if _prev_preset is None:
                                                    os.environ.pop("VIDEO_EXPORT_PRESET", None)
                                                else:
                                                    os.environ["VIDEO_EXPORT_PRESET"] = _prev_preset
                                                if _prev_font_scale is None:
                                                    os.environ.pop("VIDEO_TRANSLATED_FONT_SCALE", None)
                                                else:
                                                    os.environ["VIDEO_TRANSLATED_FONT_SCALE"] = _prev_font_scale
                                            if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                                                translated_video_path = out_path
                                                translated_video_format = mime
                                                translated_video_ext = ext
                                                break
                                            encode_err = "Output file was empty."
                                            if os.path.isfile(out_path):
                                                os.unlink(out_path)
                                        except ValueError as e:
                                            encode_err = str(e)
                                            if os.path.isfile(out_path):
                                                try:
                                                    os.unlink(out_path)
                                                except OSError:
                                                    pass
                                    if translated_video_path is None:
                                        st.warning(
                                            "Translated video file was not created. "
                                            f"{encode_err or 'Unknown error'} "
                                            "If MP4 failed, install a full OpenCV build (with FFmpeg) or use the downloaded AVI in VLC."
                                        )
                            except Exception as e:
                                st.error(f"Translated video failed: {e}")
                                with st.expander("Details"):
                                    st.exception(e)
                            _mark_stage("video_render_export")
                        if mode in ("From image", "From handwriting") and draw_boxes and ocr_results and overlay_path:
                            if mode == "From handwriting":
                                _fp = st.session_state.get("hw_ocr_first_page")
                                _rows_draw = _fp if _fp is not None else []
                            else:
                                _rows_draw = ocr_results
                            if _rows_draw:
                                annotated = st.session_state.ocr.draw_bounding_boxes(
                                    overlay_path, _rows_draw
                                )
                        if mode == "From image" and generate_photo and ocr_results and translated_texts_for_photo is not None and len(translated_texts_for_photo) == len(ocr_results):
                            try:
                                # Cap segments to avoid slow run / OOM (long menus need more rows)
                                max_boxes = int(os.environ.get("PHOTO_OVERLAY_MAX_SEGMENTS", "50"))
                                max_boxes = max(10, min(80, max_boxes))
                                res = ocr_results[:max_boxes]
                                txts = translated_texts_for_photo[:max_boxes]
                                res, txts = _sanitize_photo_overlay_payload(res, txts)
                                if not res or not txts:
                                    raise ValueError("No valid OCR boxes available for translated photo rendering.")
                                _photo_engine = st.session_state.get(
                                    "photo_output_engine", "local_overlay"
                                )
                                if _photo_engine == "openai_gpt_image_1":
                                    oai_key = _resolve_openai_key()
                                    if not oai_key:
                                        st.error(
                                            "OpenAI Images (**gpt-image-1.5**) needs an API key. "
                                            "Set **OPENAI_API_KEY** in `.env` or paste it under the photo options above."
                                        )
                                    elif not overlay_path or not os.path.isfile(overlay_path):
                                        st.warning("No image file path for OpenAI Images edit.")
                                    else:
                                        n = min(len(res), len(translations), len(txts))
                                        pairs = [
                                            (
                                                _as_text(translations[i].get("original_text")),
                                                _as_text(translations[i].get("translated_text")),
                                            )
                                            for i in range(n)
                                        ]
                                        q = st.session_state.get("openai_image_quality", "high")
                                        sz = st.session_state.get("openai_image_size", "auto")
                                        fid = st.session_state.get(
                                            "openai_image_input_fidelity", "high"
                                        )
                                        img_model = (
                                            st.session_state.get("openai_image_model") or ""
                                        ).strip() or default_image_model()
                                        raw = edit_photo_with_translations(
                                            overlay_path,
                                            pairs,
                                            api_key=oai_key,
                                            model=img_model,
                                            quality=q,
                                            size=sz,
                                            input_fidelity=fid,
                                            target_lang_display=target_lang,
                                        )
                                        translated_image = image_bytes_to_rgb_numpy(raw)
                                else:
                                    _rti = st.session_state.ocr.render_translated_image
                                    _rti_params = inspect.signature(_rti).parameters
                                    _photo_kw = {"use_font_matching": True}
                                    if "inpaint_style" in _rti_params:
                                        _photo_kw["inpaint_style"] = (
                                            "solid" if photo_inpaint_solid else "telea"
                                        )
                                    os.environ["PHOTO_REALISTIC_BLEND"] = (
                                        "1" if photo_realistic_blend else "0"
                                    )
                                    os.environ["PHOTO_STRONG_TEXT_ERASE"] = (
                                        "1" if photo_strong_erase else "0"
                                    )
                                    if "merge_same_line_boxes" in _rti_params:
                                        _photo_kw["merge_same_line_boxes"] = photo_merge_lines
                                    if "target_lang" in _rti_params:
                                        _photo_kw["target_lang"] = target_lang_code
                                    if "perspective_warp" in _rti_params:
                                        _photo_kw["perspective_warp"] = bool(
                                            st.session_state.get(
                                                "photo_perspective_warp", True
                                            )
                                        )
                                    try:
                                        translated_image = _rti(
                                            overlay_path, res, txts, **_photo_kw
                                        )
                                    except TypeError as te:
                                        if "not subscriptable" not in str(te).lower():
                                            raise
                                        retry_kw = dict(_photo_kw)
                                        if "merge_same_line_boxes" in _rti_params:
                                            retry_kw["merge_same_line_boxes"] = False
                                        retry_kw["use_font_matching"] = False
                                        safe_res, safe_txts = _sanitize_photo_overlay_payload(
                                            res, txts
                                        )
                                        if not safe_res:
                                            raise ValueError(
                                                "No valid OCR boxes available for translated photo rendering."
                                            )
                                        translated_image = _rti(
                                            overlay_path, safe_res, safe_txts, **retry_kw
                                        )
                            except Exception as e:
                                st.warning(
                                    "Could not render translated text on the photo for this input. "
                                    "Try re-uploading a clearer image, enabling OCR boost, or switching to OpenAI Images."
                                )
                                with st.expander("Photo render error details"):
                                    st.exception(e)
                        if mode == "From handwriting" and generate_photo:
                            try:
                                oai_key = _resolve_openai_key()
                                if not oai_key:
                                    st.error(
                                        "OpenAI Images (**gpt-image-1.5**) needs an API key. "
                                        "Set **OPENAI_API_KEY** in `.env` or Streamlit secrets."
                                    )
                                else:
                                    photo_src = st.session_state.get("hw_photo_source_path") or overlay_path
                                    if not photo_src or not os.path.isfile(photo_src):
                                        st.warning("No source page available for translated photo rendering.")
                                    elif not translations:
                                        st.warning("No translated handwriting text available for translated photo rendering.")
                                    else:
                                        if len(translations) > 1:
                                            st.caption(
                                                "Multiple pages detected — translated photo currently renders from the first page."
                                            )
                                        first_original = _as_text(detected_texts[0]) if detected_texts else ""
                                        first_translated = _as_text(translations[0].get("translated_text"))
                                        # Keep image-edit prompt clean: remove markdown page headers that can cause duplication.
                                        first_original = re.sub(
                                            r"(?im)^\s*##\s*page\s+\d+\s*$",
                                            "",
                                            first_original,
                                        ).strip()
                                        first_translated = re.sub(
                                            r"(?im)^\s*##\s*page\s+\d+\s*$",
                                            "",
                                            first_translated,
                                        ).strip()
                                        # Reduce hallucinated extra words: provide line-level pairs when alignment is feasible.
                                        orig_lines = [ln.strip() for ln in first_original.splitlines() if ln.strip()]
                                        trans_lines = [ln.strip() for ln in first_translated.splitlines() if ln.strip()]
                                        if (
                                            orig_lines
                                            and trans_lines
                                            and len(orig_lines) == len(trans_lines)
                                            and len(orig_lines) <= 24
                                        ):
                                            hw_pairs = list(zip(orig_lines, trans_lines))
                                        else:
                                            hw_pairs = [(first_original, first_translated)]
                                        # Guard against accidental duplicated replacements reaching image-edit prompt.
                                        _seen_hw_pairs = set()
                                        _uniq_hw_pairs = []
                                        for _o, _t in hw_pairs:
                                            _ok = re.sub(r"\s+", " ", (_o or "").strip()).lower()
                                            _tk = re.sub(r"\s+", " ", (_t or "").strip()).lower()
                                            _k = (_ok, _tk)
                                            if _k in _seen_hw_pairs:
                                                continue
                                            _seen_hw_pairs.add(_k)
                                            _uniq_hw_pairs.append((_o, _t))
                                        if _uniq_hw_pairs:
                                            hw_pairs = _uniq_hw_pairs
                                        # Faster OpenAI Images defaults for handwriting path.
                                        if target_lang_code == "ar":
                                            q = os.environ.get("OPENAI_IMAGE_HW_AR_QUALITY", "high")
                                        else:
                                            q = (
                                                st.session_state.get("openai_image_quality")
                                                or os.environ.get("OPENAI_IMAGE_HW_QUALITY", "low")
                                            )
                                        sz = (
                                            st.session_state.get("openai_image_size")
                                            or os.environ.get("OPENAI_IMAGE_HW_SIZE", "1024x1024")
                                        )
                                        if target_lang_code == "ar":
                                            # Prefer higher fidelity for Arabic handwriting pages to avoid destructive block artifacts.
                                            fid = os.environ.get("OPENAI_IMAGE_HW_AR_INPUT_FIDELITY", "high")
                                            sz = os.environ.get("OPENAI_IMAGE_HW_AR_SIZE", "auto")
                                        else:
                                            fid = (
                                                st.session_state.get("openai_image_input_fidelity")
                                                or os.environ.get("OPENAI_IMAGE_HW_INPUT_FIDELITY", "low")
                                            )
                                        img_model = (
                                            st.session_state.get("openai_image_model") or ""
                                        ).strip() or default_image_model()
                                        raw = edit_photo_with_translations(
                                            photo_src,
                                            hw_pairs,
                                            api_key=oai_key,
                                            model=img_model,
                                            quality=q,
                                            size=sz,
                                            input_fidelity=fid,
                                            target_lang_display=target_lang,
                                            dedupe_retry=(
                                                os.environ.get(
                                                    "OPENAI_IMAGE_HW_DEDUPE_RETRY", "0"
                                                ).strip().lower()
                                                in ("1", "true", "yes")
                                            ),
                                        )
                                        translated_image = image_bytes_to_rgb_numpy(raw)
                            except Exception as e:
                                st.warning(
                                    "Could not render translated text on the handwriting photo. "
                                    "Try a clearer page image and run again."
                                )
                                with st.expander("Photo render error details"):
                                    st.exception(e)
                            _mark_stage("handwriting_photo_generate")
                        if (
                            mode == "From image"
                            and overlay_path
                            and os.path.isfile(overlay_path)
                            and ocr_results
                            and translations
                        ):
                            n_ctx = min(len(translations), len(ocr_results))
                            ctx_pairs = [
                                (
                                    _as_text(translations[i].get("original_text")),
                                    _as_text(translations[i].get("translated_text")),
                                )
                                for i in range(n_ctx)
                            ]
                            with open(overlay_path, "rb") as _fimg:
                                st.session_state.last_ai_enhance_ctx = {
                                    "image_bytes": _fimg.read(),
                                    "pairs": ctx_pairs,
                                    "target_lang": target_lang,
                                }

                        if mode == "From image" and translations and detected_texts:
                            _snap_pb = (
                                np.asarray(proc_preview_before_rgb).copy()
                                if proc_preview_before_rgb is not None
                                else None
                            )
                            _snap_pa = (
                                np.asarray(proc_preview_after_rgb).copy()
                                if proc_preview_after_rgb is not None
                                else None
                            )
                            _snap_poai = (
                                np.asarray(proc_preview_after_openai_rgb).copy()
                                if proc_preview_after_openai_rgb is not None
                                else None
                            )
                            _snap_ann = np.asarray(annotated).copy() if annotated is not None else None
                            _snap_ti = None
                            if translated_image is not None:
                                try:
                                    _snap_ti = _normalize_to_rgb_uint8_array(translated_image).copy()
                                except Exception:
                                    _snap_ti = None
                            st.session_state["image_ui_snapshot_v2"] = {
                                "detected_texts": copy.deepcopy(detected_texts),
                                "detected_languages": copy.deepcopy(detected_languages),
                                "translations": copy.deepcopy(translations),
                                "results_json": copy.deepcopy(results_json),
                                "target_lang": target_lang,
                                "target_lang_code": target_lang_code,
                                "input_lang": input_lang,
                                "preprocess": bool(preprocess),
                                "ocr_openai_preprocess": bool(ocr_openai_preprocess),
                                "generate_photo": bool(generate_photo),
                                "proc_before": _snap_pb,
                                "proc_after": _snap_pa,
                                "proc_after_openai": _snap_poai,
                                "openai_preprocess_error": st.session_state.get(
                                    "last_openai_preprocess_error", ""
                                ),
                                "annotated": _snap_ann,
                                "translated_image": _snap_ti,
                            }
                            st.session_state["image_ui_persist_active"] = True

                        if mode == "From video":
                            # Keep a UI-only copy of the translated render so reruns (Annotated **Next**) are not
                            # left with a missing path if the encoder temp file is removed or locked.
                            _tv_snap_path = translated_video_path
                            _prev_tr = st.session_state.get("video_ui_translated_retain_path")
                            if _prev_tr and isinstance(_prev_tr, str) and os.path.isfile(_prev_tr):
                                try:
                                    os.unlink(_prev_tr)
                                except OSError:
                                    pass
                                st.session_state.pop("video_ui_translated_retain_path", None)
                            if translated_video_path and os.path.isfile(translated_video_path):
                                try:
                                    if os.path.getsize(translated_video_path) <= 200 * 1024 * 1024:
                                        _suf = "." + (translated_video_ext or "mp4").lstrip(".")
                                        fd, _tv_copy = tempfile.mkstemp(suffix=_suf)
                                        os.close(fd)
                                        shutil.copy2(translated_video_path, _tv_copy)
                                        _tv_snap_path = _tv_copy
                                        st.session_state["video_ui_translated_retain_path"] = _tv_copy
                                except OSError:
                                    pass
                            _vui_snap = {
                                "detected_texts": copy.deepcopy(detected_texts),
                                "detected_languages": copy.deepcopy(detected_languages),
                                "translations": copy.deepcopy(translations),
                                "results_json": copy.deepcopy(results_json),
                                "target_lang": target_lang,
                                "target_lang_code": target_lang_code,
                                "input_lang": input_lang,
                                "_run_t0": _run_t0,
                                "_perf_stages": dict(_perf_stages),
                                "generate_video": generate_video,
                                "translated_video_path": _tv_snap_path,
                                "translated_video_format": translated_video_format,
                                "translated_video_ext": translated_video_ext,
                                "video_per_frame_preview": copy.deepcopy(st.session_state.get("video_per_frame_preview")),
                                "_video_retained_for_preview": st.session_state.get("_video_retained_for_preview"),
                            }
                            st.session_state["video_ui_snapshot_v2"] = copy.deepcopy(_vui_snap)
                            st.session_state["video_ui_persist_active"] = True
                            _replay_video_translation_ui_video(
                                _vui_snap,
                                text_output_placeholder=text_output_placeholder,
                            )
                        else:
                            if mode != "From text":
                                st.success(f"Detected {len(detected_texts)} segment(s)")

                            # Performance visibility to pinpoint bottlenecks in one run.
                            total_elapsed = time.perf_counter() - _run_t0
                            _perf_caption = _format_run_perf_caption(_perf_stages, total_elapsed)
                            if _perf_caption:
                                st.caption(_perf_caption)

                            if detected_languages and mode in (
                                "From image",
                                "From video",
                                "From handwriting",
                            ):
                                unique_langs = ", ".join(sorted(set(detected_languages)))
                                st.caption(f"Source: {unique_langs}")

                            # Bytes for download; path for st.video (browser handles MP4 from file better than raw bytes in some setups)
                            translated_video_bytes = None
                            if translated_video_path and os.path.isfile(translated_video_path):
                                try:
                                    with open(translated_video_path, "rb") as vf:
                                        translated_video_bytes = vf.read()
                                except OSError:
                                    translated_video_bytes = None

                            if mode == "From video" and translated_video_path and os.path.isfile(translated_video_path):
                                st.markdown("### Translated video")
                                st.caption("Play below in the app, or download the file.")
                                if translated_video_ext == "avi":
                                    st.caption(
                                        "Your system saved **AVI** (MP4 was unavailable). Many browsers cannot play AVI inline — "
                                        "use **Download translated video** and open it in **VLC** or the **Movies & TV** app if the player stays blank."
                                    )
                                try:
                                    st.video(translated_video_path)
                                except Exception:
                                    if translated_video_bytes:
                                        st.video(translated_video_bytes, format=translated_video_format)
                                    else:
                                        st.warning("Could not embed the video player; use download below.")
                                if translated_video_bytes:
                                    st.download_button(
                                        label="Download translated video",
                                        data=translated_video_bytes,
                                        file_name=f"translated_video.{translated_video_ext}",
                                        mime=translated_video_format,
                                        key="dl_translated_video_main",
                                        use_container_width=True,
                                    )
                                st.markdown("---")
                            elif mode == "From video" and generate_video and not translated_video_path:
                                st.info(
                                    "**Generate video with translated text** was on, but no video file was produced. "
                                    "Check that text was detected and try again. If it keeps failing, see the warning or error above."
                                )

                            _pj = _paragraph_join_deduped if mode == "From video" else _paragraph_join
                            translated_block = _pj([t["translated_text"] for t in translations])
                            if mode in ("From text", "From voice"):
                                st.session_state["last_text_translation"] = translated_block
                                if mode == "From text" and text_output_placeholder is not None:
                                    text_output_placeholder.text_area(
                                        "Translation output",
                                        value=translated_block,
                                        placeholder="Translation will appear here...",
                                        height=200,
                                        label_visibility="collapsed",
                                        disabled=True,
                                    )

                            if mode != "From text":
                                if mode == "From video":
                                    output_tab_label = "Translated video"
                                elif mode == "From handwriting":
                                    output_tab_label = (
                                        "Translated photo + Exports (PDF / PNG)"
                                        if generate_photo
                                        else "Exports (PDF / PNG)"
                                    )
                                else:
                                    output_tab_label = "Translated photo"
                                tab_list = ["Text", "Annotated", "Processing", output_tab_label, "JSON"]
                                (
                                    translations_tab,
                                    annotated_tab,
                                    processing_tab,
                                    translated_output_tab,
                                    raw_tab,
                                ) = st.tabs(tab_list)

                                with translations_tab:
                                    original_block = _pj(detected_texts)
                                    if mode in ("From text", "From voice"):
                                        src_title = f"Source · {input_lang}"
                                    else:
                                        ul = ", ".join(sorted(set(detected_languages))) if detected_languages else "—"
                                        src_title = f"Source · Auto ({ul})"
                                    pair_left, pair_right = st.columns(2)
                                    with pair_left:
                                        st.markdown(
                                            f'<div class="tx-card"><div class="tx-card-label">{_html_escape(src_title)}</div>'
                                            f'<div class="tx-card-body">{_html_escape(original_block) or " "}</div></div>',
                                            unsafe_allow_html=True,
                                        )
                                        _clipboard_button("📋", original_block, "tx_src", tooltip="Copy source")
                                        st.download_button(
                                            label="⬇️",
                                            data=original_block.encode("utf-8"),
                                            file_name=f"source_{target_lang_code}.txt",
                                            mime="text/plain",
                                            key="dl_src_txt_inline",
                                            help="Download source text (.txt)",
                                        )
                                    with pair_right:
                                        st.markdown(
                                            f'<div class="tx-card"><div class="tx-card-label">Target · {_html_escape(target_lang)}</div>'
                                            f'<div class="tx-card-body">{_html_escape(translated_block) or " "}</div></div>',
                                            unsafe_allow_html=True,
                                        )
                                        _clipboard_button("📋", translated_block, "tx_tgt", tooltip="Copy translation")
                                        if mode in ("From image", "From handwriting"):
                                            st.download_button(
                                                label="⬇️",
                                                data=translated_block.encode("utf-8"),
                                                file_name=f"translation_{target_lang_code}.txt",
                                                mime="text/plain",
                                                key="dl_tgt_txt_inline_img",
                                                help="Download target text (.txt)",
                                            )
                                    if mode not in ("From image", "From handwriting"):
                                        st.download_button(
                                            label="⬇️",
                                            data=translated_block.encode("utf-8"),
                                            file_name=f"translation_{target_lang_code}.txt",
                                            mime="text/plain",
                                            key="dl_txt",
                                            help="Download translation (.txt)",
                                            use_container_width=True,
                                        )
                                    if mode not in ("From image", "From handwriting"):
                                        if st.button("Save this translation", key="save_translation_btn", type="primary", use_container_width=True):
                                            if mode in ("From text", "From voice"):
                                                save_source_lang = input_lang
                                            else:
                                                save_source_lang = ", ".join(sorted(set(detected_languages))) if detected_languages else "Auto"
                                            _upsert_saved_translation(
                                                {
                                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                "source_lang": save_source_lang,
                                                "target_lang": target_lang,
                                                "original_text": original_block,
                                                "translated_text": translated_block,
                                                },
                                                st.session_state.saved_translations,
                                            )
                                            st.success("Translation saved.")

                                with annotated_tab:
                                    if annotated is not None:
                                        _safe_show_image(annotated, caption="Detected text regions", use_column_width=True)
                                    elif mode == "From video":
                                        _render_video_per_frame_annotate_ui()
                                    elif mode == "From handwriting":
                                        _prev = st.session_state.get("hw_last_preview_path")
                                        if _prev and os.path.isfile(_prev):
                                            try:
                                                _pv = Image.open(_prev).convert("RGB")
                                                _safe_show_image(
                                                    _pv,
                                                    caption="Source page (first page — Vision has no box overlay)",
                                                    use_column_width=True,
                                                )
                                            except Exception:
                                                st.info(
                                                    "Handwriting uses **OpenAI Vision** on the full page; "
                                                    "there is no bounding-box overlay."
                                                )
                                        else:
                                            st.info(
                                                "Handwriting uses **OpenAI Vision** on the full page; "
                                                "there is no bounding-box overlay."
                                            )
                                    else:
                                        if mode == "From image":
                                            st.info("Enable **Show bounding boxes** in Advanced options to see annotations.")
                                        else:
                                            st.info("Annotated image is only available for image input.")

                                with processing_tab:
                                    if mode == "From image":
                                        _render_processing_preprocess_views(
                                            proc_preview_before_rgb,
                                            proc_preview_after_rgb,
                                            proc_preview_after_openai_rgb,
                                            preprocess=preprocess,
                                            ocr_openai_preprocess=ocr_openai_preprocess,
                                            openai_preprocess_error=st.session_state.get(
                                                "last_openai_preprocess_error"
                                            ),
                                        )
                                    elif mode == "From video":
                                        st.info(
                                            "Preprocessing before/after applies to **static image** OCR. "
                                            "Video uses the same steps on sampled frames during translation."
                                        )
                                    elif mode == "From handwriting":
                                        st.info(
                                            "Handwriting uses **OpenAI Vision** on the page; "
                                            "OpenCV OCR preprocessing preview is not used here."
                                        )

                                with translated_output_tab:
                                    if translated_image is not None:
                                        _safe_show_image(translated_image, caption=f"Translated to {target_lang}", use_column_width=True)
                                        if isinstance(translated_image, np.ndarray):
                                            _arr_out = translated_image
                                        elif isinstance(translated_image, Image.Image):
                                            _arr_out = np.array(translated_image.convert("RGB"))
                                        else:
                                            _arr_out = np.array(translated_image)
                                        pil_out = Image.fromarray(_arr_out)
                                        buf = io.BytesIO()
                                        pil_out.save(buf, format="PNG")
                                        translated_png_bytes = buf.getvalue()
                                        st.download_button(label="Download image", data=translated_png_bytes, file_name="translated_image.png", mime="image/png", key="dl_translated_img")
                                        if mode == "From handwriting":
                                            if handwriting_pdf_bytes:
                                                st.download_button(
                                                    label="Download PDF export",
                                                    data=handwriting_pdf_bytes,
                                                    file_name=f"handwriting_translation_{target_lang_code}.pdf",
                                                    mime="application/pdf",
                                                    key="dl_hw_pdf",
                                                    use_container_width=True,
                                                )
                                            if handwriting_png_bytes:
                                                st.download_button(
                                                    label="Download PNG summary",
                                                    data=handwriting_png_bytes,
                                                    file_name=f"handwriting_summary_{target_lang_code}.png",
                                                    mime="image/png",
                                                    key="dl_hw_png",
                                                    use_container_width=True,
                                                )
                                        if mode == "From image":
                                            detected_block = _pj(detected_texts)
                                            save_source_lang = ", ".join(sorted(set(detected_languages))) if detected_languages else "Auto"
                                            image_save_item = {
                                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                "source_lang": save_source_lang,
                                                "target_lang": target_lang,
                                                "original_text": detected_block,
                                                "translated_text": translated_block,
                                                "detected_text": detected_block,
                                                "translated_photo_png_b64": base64.b64encode(translated_png_bytes).decode("ascii"),
                                            }
                                            saved_idx = _find_saved_item_index(
                                                image_save_item, st.session_state.saved_translations
                                            )
                                            has_saved_image = (
                                                saved_idx >= 0
                                                and bool(
                                                    (st.session_state.saved_translations[saved_idx] or {}).get(
                                                        "translated_photo_png_b64"
                                                    )
                                                )
                                            )
                                            image_star = "★ Saved image" if has_saved_image else "☆ Save image result"
                                            if st.button(
                                                image_star,
                                                key="save_image_result_star",
                                                use_container_width=True,
                                                help=(
                                                    "Already saved with photo"
                                                    if has_saved_image
                                                    else "Save translated photo + detected text + translated text"
                                                ),
                                                disabled=has_saved_image,
                                            ):
                                                _upsert_saved_translation(
                                                    image_save_item,
                                                    st.session_state.saved_translations,
                                                )
                                                st.success("Image translation saved.")
                                        st.caption("Use the ✦ Enhance with AI button below the result section.")
                                    elif translated_video_path and os.path.isfile(translated_video_path):
                                        st.caption("Same preview as above; you can play or download here too.")
                                        try:
                                            st.video(translated_video_path)
                                        except Exception:
                                            if translated_video_bytes:
                                                st.video(translated_video_bytes, format=translated_video_format)
                                        if translated_video_bytes:
                                            st.download_button(
                                                label="Download translated video",
                                                data=translated_video_bytes,
                                                file_name=f"translated_video.{translated_video_ext}",
                                                mime=translated_video_format,
                                                key="dl_translated_video_tab",
                                                use_container_width=True,
                                            )
                                    else:
                                        if mode == "From video":
                                            st.info(
                                                "Turn on **Generate video with translated text** in Advanced options, then click **Translate**. "
                                                "When processing finishes, the player and download appear above and in this tab."
                                            )
                                        elif mode == "From handwriting":
                                            st.info(
                                                "Run **Process handwriting** after upload. "
                                                "Turn on **Generate photo with translated text** for a translated page image."
                                            )
                                        else:
                                            st.info("Enable **Generate photo with translated text** (image mode) in Advanced options, then run **Translate**.")

                                with raw_tab:
                                    st.json(results_json)
                                    st.download_button(label="Download JSON", data=json.dumps(results_json, indent=2, ensure_ascii=False), file_name="translation_results.json", mime="application/json", use_container_width=True)
                    except MemoryError:
                        st.error("Out of memory. Try a smaller image, close other apps, or disable «Generate photo with translated text».")
                    except Exception as e:
                        if _is_cuda_oom_error(e) and not st.session_state.get("_ocr_force_cpu"):
                            st.session_state._ocr_force_cpu = True
                            st.session_state.ocr = None
                            st.session_state._ocr_loaded_langs = None
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass
                            st.warning("CUDA ran out of memory. Switched OCR to CPU mode and retrying automatically.")
                            st.rerun()
                        st.error(f"Error: {str(e)}")
                        with st.expander("Details"):
                            st.exception(e)
                    finally:
                        if mode == "From image":
                            for _p in (temp_path, overlay_path):
                                if _p and os.path.isfile(_p):
                                    try:
                                        os.unlink(_p)
                                    except OSError:
                                        pass
                        if mode == "From video" and video_path and os.path.isfile(video_path):
                            _prev_keep = st.session_state.get("_video_retained_for_preview")
                            if _prev_keep and _prev_keep != video_path and os.path.isfile(_prev_keep):
                                try:
                                    os.unlink(_prev_keep)
                                except OSError:
                                    pass
                            st.session_state["_video_retained_for_preview"] = video_path
            elif (
                not should_run
                and mode == "From video"
                and st.session_state.get("video_ui_persist_active")
                and isinstance(st.session_state.get("video_ui_snapshot_v2"), dict)
            ):
                st.caption(
                    "Showing your **last video translation**. Use **Annotated** to change frames — "
                    "no need to click **Translate** again."
                )
                if st.session_state.get("ocr") is None:
                    initialize_models(
                        ocr_languages,
                        ocr_backend=st.session_state.get("ocr_backend", "easyocr"),
                        trocr_model=st.session_state.get("trocr_model_id", "").strip() or None,
                    )
                _sn = st.session_state["video_ui_snapshot_v2"]
                if _sn.get("video_per_frame_preview") is not None:
                    st.session_state["video_per_frame_preview"] = copy.deepcopy(_sn["video_per_frame_preview"])
                _vk = _sn.get("_video_retained_for_preview")
                if _vk:
                    st.session_state["_video_retained_for_preview"] = _vk
                _replay_video_translation_ui_video(_sn, text_output_placeholder=text_output_placeholder)

            elif (
                not should_run
                and mode == "From image"
                and st.session_state.get("image_ui_persist_active")
                and isinstance(st.session_state.get("image_ui_snapshot_v2"), dict)
            ):
                if st.session_state.get("ocr") is None:
                    initialize_models(
                        ocr_languages,
                        ocr_backend=st.session_state.get("ocr_backend", "easyocr"),
                        trocr_model=st.session_state.get("trocr_model_id", "").strip() or None,
                    )
                _replay_image_translation_ui_snap(st.session_state["image_ui_snapshot_v2"])

    if mode == "From image" and st.session_state.get("last_ai_enhance_ctx"):
        st.markdown("---")
        ai_ctx = st.session_state.get("last_ai_enhance_ctx") or {}
        ai_left, ai_right = st.columns([1, 7])
        with ai_left:
            ai_quick = st.button(
                "✨",
                key="ai_enhance_photo_quick_btn",
                type="secondary",
                help="Enhance with AI",
                use_container_width=True,
            )
        with ai_right:
            st.caption(
                "Enhance with AI — upload payload is resized/JPEG-optimized by default; duplicate runs can hit disk cache. "
                "Most wall time is still OpenAI image generation. "
                "**GPT-Image quality → medium/low**, `OPENAI_IMAGE_ENHANCE_FAST=1`, or "
                "`OPENAI_IMAGE_ENHANCE_NONBLOCKING=1` (polls with auto-rerun)."
            )
        _enhance_nb = _env_flag("OPENAI_IMAGE_ENHANCE_NONBLOCKING")
        _enhance_poll_s = 2.0
        try:
            _enhance_poll_s = float(os.environ.get("OPENAI_IMAGE_ENHANCE_POLL_S", "2"))
        except ValueError:
            pass
        _enhance_poll_s = max(0.5, min(30.0, _enhance_poll_s))

        _fut_existing = st.session_state.get("_enhance_future")
        if _enhance_nb and _fut_existing is not None and not _fut_existing.done():
            st.info("Enhancement in progress… (non-blocking mode)")
            time.sleep(_enhance_poll_s)
            st.rerun()

        if ai_quick:
            st.session_state.pop("_enhance_result", None)
            st.session_state.pop("_enhance_async_error", None)
            _eb = ai_ctx.get("image_bytes") or b""
            _epairs = ai_ctx.get("pairs") or []
            _etarget = ai_ctx.get("target_lang") or target_lang
            if _enhance_nb:
                _busy = st.session_state.get("_enhance_future")
                if _busy is not None and not _busy.done():
                    st.warning("An enhancement is already running — wait for it to finish.")
                else:
                    try:
                        _snap_key = _resolve_openai_key().strip()
                        if not _snap_key:
                            st.error(
                                "OpenAI Images needs an API key before running ✨ in the background."
                            )
                        else:
                            _snap_opts = _openai_image_options_for_quick_enhance()
                            _snap_compact = not _env_flag("OPENAI_IMAGE_ENHANCE_FULL_PROMPT")
                            st.session_state["_enhance_future"] = _ENHANCE_EXECUTOR.submit(
                                _enhance_background_job,
                                (
                                    _eb,
                                    _epairs,
                                    _etarget,
                                    _snap_key,
                                    _snap_opts,
                                    _snap_compact,
                                ),
                            )
                            st.success("Enhancement queued — refreshing until the API returns.")
                            time.sleep(_enhance_poll_s)
                            st.rerun()
                    except Exception as e:
                        st.error(f"AI enhancement failed to start: {e}")
            else:
                try:
                    _perf_ai: dict = {}
                    _t_ui_ai = time.perf_counter()
                    with st.spinner("Enhancing photo with AI..."):
                        ai_img = _generate_openai_translated_photo_from_bytes(
                            _eb,
                            _epairs,
                            _etarget,
                            perf_stats=_perf_ai,
                        )
                    _perf_ai["ui_wall_total_s"] = time.perf_counter() - _t_ui_ai
                    st.success("AI-enhanced photo generated.")
                    _cap_ai = _fmt_ai_enhance_perf(_perf_ai)
                    if _cap_ai:
                        st.caption(_cap_ai)
                    _safe_show_image(ai_img, caption="AI Enhanced", use_column_width=True)
                    ai_pil = Image.fromarray(np.array(ai_img))
                    ai_buf = io.BytesIO()
                    ai_pil.save(ai_buf, format="PNG")
                    st.download_button(
                        label="Download AI image",
                        data=ai_buf.getvalue(),
                        file_name="translated_image_ai.png",
                        mime="image/png",
                        key="dl_ai_quick_img",
                    )
                except Exception as e:
                    st.error(f"AI enhancement failed: {e}")

        if _enhance_nb:
            _fut_done = st.session_state.get("_enhance_future")
            if _fut_done is not None and _fut_done.done():
                st.session_state["_enhance_future"] = None
                try:
                    ai_img_nb, _perf_nb = _fut_done.result(timeout=5)
                    st.session_state["_enhance_result"] = (ai_img_nb, _perf_nb)
                except Exception as e:
                    st.session_state["_enhance_async_error"] = str(e)

        _async_err = st.session_state.pop("_enhance_async_error", None)
        if _async_err:
            st.error(f"AI enhancement failed: {_async_err}")

        _res_nb = st.session_state.get("_enhance_result")
        if _res_nb is not None and _enhance_nb:
            ai_img_r, _perf_r = _res_nb
            st.success("AI-enhanced photo generated.")
            _cap_r = _fmt_ai_enhance_perf(_perf_r)
            if _cap_r:
                st.caption(_cap_r)
            _safe_show_image(ai_img_r, caption="AI Enhanced", use_column_width=True)
            ai_pil_nb = Image.fromarray(np.array(ai_img_r))
            ai_buf_nb = io.BytesIO()
            ai_pil_nb.save(ai_buf_nb, format="PNG")
            st.download_button(
                label="Download AI image",
                data=ai_buf_nb.getvalue(),
                file_name="translated_image_ai.png",
                mime="image/png",
                key="dl_ai_quick_nb_img",
            )

    if st.session_state.history:
        st.markdown("---")
        with st.expander("Recent translations"):
            for item in st.session_state.history[:5]:
                prev = (item.get("preview") or "")[:50]
                st.caption(f"{item['num_segments']} segment(s) → **{item['target_lang']}**" + (f" · _{prev}..._" if prev else ""))
            if st.button("Clear history", key="clear_history"):
                st.session_state.history.clear()
                _save_history(st.session_state.history)
                st.rerun()


if __name__ == '__main__':
    main()
