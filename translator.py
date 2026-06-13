"""
Translation module for translating extracted text.
Uses NLLB-200 (No Language Left Behind) for high-quality offline translation.
"""

import warnings

warnings.filterwarnings("ignore", message=r".*Accessing `__path__` from.*")
warnings.filterwarnings("ignore", message=r".*Returning `__path__` instead.*")
warnings.filterwarnings("ignore", message=r".*__path__.*zoedepth.*", category=FutureWarning)
warnings.filterwarnings("ignore")

import re
import unicodedata
from typing import List, Optional, Dict
import torch

# NLLB-200 distilled 600M only (~1.2 GB). Same 200-language setup; Arabic = arb_Arab.
NLLB_600M = "facebook/nllb-200-distilled-600M"


def _nllb_model_id(force_600m: bool = False, variant: Optional[str] = None) -> str:
    """Hugging Face model id for pre-download scripts (always NLLB-200 600M distilled)."""
    del force_600m, variant
    return NLLB_600M


def _nllb_env_force_cpu() -> bool:
    """Set NLLB_USE_CPU=1 (or TRANSLATE_FORCE_CPU=1) to keep NLLB on CPU even if CUDA/MPS exists."""
    import os

    for key in ("NLLB_USE_CPU", "TRANSLATE_FORCE_CPU"):
        v = (os.environ.get(key) or "").strip().lower()
        if v in ("1", "true", "yes"):
            return True
    return False


# Map app language codes to NLLB/Flores-200 BCP-47 codes (language_Script)
LANG_TO_NLLB: Dict[str, str] = {
    'en': 'eng_Latn',
    'es': 'spa_Latn',
    'fr': 'fra_Latn',
    'de': 'deu_Latn',
    'it': 'ita_Latn',
    'pt': 'por_Latn',
    'ru': 'rus_Cyrl',
    'zh': 'zho_Hans',
    'ja': 'jpn_Jpan',
    'ko': 'kor_Hang',
    'ar': 'arb_Arab',
    'hi': 'hin_Deva',
    'nl': 'nld_Latn',
    'pl': 'pol_Latn',
    'tr': 'tur_Latn',
    'vi': 'vie_Latn',
    'th': 'tha_Thai',
    'cs': 'ces_Latn',
    'sv': 'swe_Latn',
    'da': 'dan_Latn',
    'no': 'nob_Latn',
    'fi': 'fin_Latn',
    'el': 'ell_Grek',
    'he': 'heb_Hebr',
    'id': 'ind_Latn',
    'ms': 'zsm_Latn',
    'uk': 'ukr_Cyrl',
    'ro': 'ron_Latn',
    'hu': 'hun_Latn',
    'sk': 'slk_Latn',
    'bg': 'bul_Cyrl',
    'hr': 'hrv_Latn',
    'sr': 'srp_Cyrl',
    'sl': 'slv_Latn',
    'et': 'est_Latn',
    'lv': 'lav_Latn',
    'lt': 'lit_Latn',
}

# For UI dropdown without loading the model
SUPPORTED_LANGUAGES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
    'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
    'nl': 'Dutch', 'pl': 'Polish', 'tr': 'Turkish', 'vi': 'Vietnamese',
    'th': 'Thai', 'cs': 'Czech', 'sv': 'Swedish', 'da': 'Danish',
    'no': 'Norwegian', 'fi': 'Finnish', 'el': 'Greek', 'he': 'Hebrew',
    'id': 'Indonesian', 'ms': 'Malay', 'uk': 'Ukrainian', 'ro': 'Romanian',
    'hu': 'Hungarian', 'sk': 'Slovak', 'bg': 'Bulgarian', 'hr': 'Croatian',
    'sr': 'Serbian', 'sl': 'Slovenian', 'et': 'Estonian', 'lv': 'Latvian',
    'lt': 'Lithuanian',
}


def _text_has_arabic_script(s: str) -> bool:
    for c in s or "":
        if "\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F" or "\u08A0" <= c <= "\u08FF":
            return True
    return False


def _token_has_arabic_script(tok: str) -> bool:
    for c in tok or "":
        if "\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F" or "\u08A0" <= c <= "\u08FF":
            return True
    return False


def _latin_letter_count(s: str) -> int:
    return sum(1 for c in s if "A" <= c <= "Z" or "a" <= c <= "z")


def _heuristic_latin_source_when_src_equals_tgt(text: str, src_lo: str, tgt_lo: str) -> Optional[str]:
    """
    Per-line OCR inference often yields ``en`` when only English models are enabled, even for French
    or German text. If that matches the **target** (e.g. en→en), NLLB would no-op and the UI looks
    like "translation does not work". Pick a plausible Latin source from diacritics / common words.
    """
    if src_lo != tgt_lo:
        return None
    t = (text or "").strip()
    if len(t) < 3 or _latin_letter_count(t) < 2:
        return None
    low = t.lower()
    low_sp = f" {low} "

    def _has_fr_diacritics() -> bool:
        return bool(re.search(r"[àâäéèêëïîôùûüÿçœæ]", t, re.I))

    def _has_fr_glue_words() -> bool:
        # Space-padded to avoid matching substrings inside English tokens.
        return any(
            w in low_sp
            for w in (
                " le ",
                " la ",
                " les ",
                " l'",
                " un ",
                " une ",
                " des ",
                " du ",
                " de ",
                " et ",
                " est ",
                " vous ",
                " nous ",
                " dans ",
                " pour ",
                " avec ",
            )
        )

    if tgt_lo != "fr" and "fr" in LANG_TO_NLLB:
        if _has_fr_diacritics() or (_has_fr_glue_words() and len(t) < 600):
            return "fr"

    if tgt_lo != "es" and "es" in LANG_TO_NLLB:
        if re.search(r"[ñáéíóúü¿¡]", t, re.I):
            return "es"

    if tgt_lo != "de" and "de" in LANG_TO_NLLB:
        if "ß" in t or re.search(r"[äöüÄÖÜ]", t):
            if not _has_fr_diacritics():
                return "de"

    return None


def _normalize_nllb_source_text(text: str, src_nllb: Optional[str]) -> str:
    """Unicode NFC + remove kashida for Arabic so NLLB sees standard spelling."""
    if not text or not src_nllb:
        return (text or "").strip()
    t = text.strip()
    if src_nllb == "arb_Arab":
        t = unicodedata.normalize("NFC", t)
        t = t.replace("\u0640", "")  # tatweel / kashida (common on signs)
    return t


def _split_text_for_nllb(raw: str, src_nllb: str) -> List[str]:
    """
    Chunk text before generate(). Latin .!? splitting is wrong for Arabic (different punctuation,
    and stray '.' from OCR breaks phrases). Arabic: split on newlines only (OCR lines).
    """
    raw = raw.strip()
    if not raw:
        return []
    if src_nllb == "arb_Arab":
        parts = [p.strip() for p in re.split(r"\n+", raw) if p.strip()]
        return parts if parts else [raw]
    parts = re.split(r"(?<=[.!?])\s+", raw)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [raw]


def _subsplit_long_segment(segment: str, src_nllb: str) -> List[str]:
    """Further split segments over ~400 chars. Arabic uses Arabic comma/semicolon, not ASCII only."""
    segment = segment.strip()
    if not segment:
        return []
    if len(segment) <= 400:
        return [segment]
    if src_nllb == "arb_Arab":
        sub = re.split(r"(?<=[،؛])\s+", segment)
        sub = [s.strip() for s in sub if s.strip()]
        if len(sub) > 1:
            return sub
        return [segment]
    subparts = re.split(r"(?<=[,;:])\s+", segment)
    subparts = [s.strip() for s in subparts if s.strip()]
    return subparts if subparts else [segment]


def _scrub_mixed_latin_noise_in_arabic_output(s: str) -> str:
    """
    When target is Arabic, NLLB often copies OCR/subtitle English into Arabic output. Remove that
    noise while keeping Arabic + digits.

    - CamelCase OCR glitches (loveiL, holelAir).
    - Whitespace tokens with no Arabic but 2+ Latin letters (handles enjoy:., @Oh;and, [fyouwant).
    - Embedded Latin words inside mixed lines (… Imnot devoid …).

    Set env ARABIC_KEEP_LATIN_TOKENS=1 to skip Latin removal (e.g. intentional English names).
    """
    if not s or not _text_has_arabic_script(s):
        return s
    import os

    # OCR/model glitch: lowercase letter immediately before an uppercase inside the same word
    for _ in range(3):
        s2 = re.sub(r"\b[A-Za-z]*[a-z][A-Z][A-Za-z]*\b", " ", s)
        if s2 == s:
            break
        s = s2

    if os.environ.get("ARABIC_KEEP_LATIN_TOKENS", "").strip().lower() in ("1", "true", "yes"):
        return re.sub(r"\s{2,}", " ", s).strip()

    # Drop whole tokens that have no Arabic but still contain English (any punctuation)
    kept: List[str] = []
    for tok in s.split():
        if _token_has_arabic_script(tok):
            kept.append(tok)
            continue
        if _latin_letter_count(tok) >= 2:
            continue
        kept.append(tok)
    s = " ".join(kept)

    # Remove embedded English substrings on mixed Arabic+Latin lines (e.g. … Imnot devoid …)
    if _text_has_arabic_script(s):
        s = re.sub(r"[A-Za-z]{2,}", " ", s)

    return re.sub(r"\s{2,}", " ", s).strip()


class TextTranslator:
    """Text translation using NLLB-200 (offline, high quality)."""

    def __init__(self, nllb_variant: Optional[str] = None):
        """Offline translation uses NLLB-200 distilled 600M only. ``nllb_variant`` is ignored (API compat)."""
        _ = nllb_variant
        self._nllb_variant = "600m"
        self.model = None
        self.tokenizer = None
        self.device = self._choose_device()
        self._load_error = None
        self._loaded_model_id: Optional[str] = None
        self._cache: Dict[tuple, str] = {}
        self._cache_max = 200
        self._cache_keys: List[tuple] = []
        self._model_uses_device_map = False
        print(f"Translation device: {self.device}; NLLB tier: {self._nllb_variant} (GPU for speed when available)")

    @staticmethod
    def _choose_device() -> str:
        """Use CUDA (or Apple MPS) when PyTorch reports it; else CPU. Override with NLLB_USE_CPU=1."""
        import os

        if _nllb_env_force_cpu():
            return "cpu"
        if torch.cuda.is_available():
            raw = (os.environ.get("NLLB_CUDA_DEVICE") or "0").strip()
            if raw.isdigit():
                return f"cuda:{int(raw)}"
            return "cuda" if raw in ("", "cuda") else raw
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _tensor_device(self):
        """Device for model inputs (handles device_map / multi-GPU splits)."""
        if self.model is None:
            return torch.device(self.device)
        return next(self.model.parameters()).device

    def _get_nllb_code(self, lang: str) -> Optional[str]:
        return LANG_TO_NLLB.get(lang.lower()) if lang else None

    def _load_nllb(self) -> bool:
        """Load NLLB-200 distilled 600M and tokenizer once (with retries)."""
        if self.model is not None and self.tokenizer is not None:
            return True
        if self._load_error is not None:
            return False
        import os
        import time
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

        if str(self.device).startswith("cuda"):
            try:
                n = torch.cuda.device_count()
                if n > 0:
                    dev_s = str(self.device)
                    di = 0
                    if dev_s.startswith("cuda:") and ":" in dev_s:
                        try:
                            di = int(dev_s.split(":")[1])
                        except ValueError:
                            di = 0
                    di = max(0, min(di, n - 1))
                    print(
                        f"PyTorch CUDA: {n} GPU(s) visible; using {self.device} "
                        f"({torch.cuda.get_device_name(di)})"
                    )
                else:
                    print("CUDA device requested but no visible GPU; using CPU fallback path.")
            except Exception as e:
                print(f"CUDA probe failed ({e}); continuing on CPU path.")
        else:
            print("Translation running on CPU mode.")

        mid = NLLB_600M
        last_error = None
        self._model_uses_device_map = False
        self.model = None
        self.tokenizer = None

        for attempt in range(3):
            try:
                print(f"Loading NLLB model: {mid} (device: {self.device})... attempt {attempt + 1}/3")
                tok = AutoTokenizer.from_pretrained(mid)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    mid, torch_dtype=torch.float32, low_cpu_mem_usage=True
                )
                model = model.to(self.device)
                model.eval()
                self.tokenizer = tok
                self.model = model
                self._loaded_model_id = mid
                print("NLLB model loaded successfully!")
                return True
            except Exception as e:
                last_error = e
                self.model = None
                self.tokenizer = None
                self._loaded_model_id = None
                err_str = str(e).lower()
                if "incompleteread" in err_str or "connection broken" in err_str or "connection error" in err_str:
                    print("Download interrupted. Retrying in 10s...")
                    time.sleep(10)
                else:
                    break

        self._load_error = str(last_error) if last_error else "Unknown error"
        print(f"NLLB load failed: {self._load_error}")
        return False

    def translate_text(
        self,
        text: str,
        target_language: str = 'en',
        source_language: Optional[str] = None,
        fast: bool = False,
    ) -> dict:
        if not source_language:
            source_language = 'en'

        src_lo = source_language.lower()
        tgt_lo = target_language.lower()
        # Mis-inferred Latin source with same code as target (e.g. de→de) leaves Arabic signage
        # unchanged because NLLB short-circuits identity. Force Arabic as MT source when the string
        # clearly contains Arabic script and the user is translating into a non–Arabic-script language.
        if src_lo == tgt_lo and _text_has_arabic_script(text) and tgt_lo not in (
            "ar",
            "fa",
            "ur",
            "ps",
            "ug",
            "sd",
            "ckb",
        ):
            source_language = "ar"
            src_lo = "ar"
        else:
            lat_fix = _heuristic_latin_source_when_src_equals_tgt(text, src_lo, tgt_lo)
            if lat_fix:
                source_language = lat_fix
                src_lo = lat_fix.lower()

        if src_lo == tgt_lo:
            return {
                'original_text': text,
                'translated_text': text,
                'source_language': source_language,
                'target_language': target_language,
                'confidence': 1.0,
            }

        src_nllb = self._get_nllb_code(source_language)
        tgt_nllb = self._get_nllb_code(target_language)
        if not src_nllb or not tgt_nllb:
            return {
                'original_text': text,
                'translated_text': f"Unsupported language pair: {source_language} -> {target_language}",
                'source_language': source_language,
                'target_language': target_language,
                'confidence': None,
            }

        raw = _normalize_nllb_source_text((text or "").strip(), src_nllb)
        cache_key = (hash(raw), target_language, source_language)
        if cache_key in self._cache:
            return {
                'original_text': text,
                'translated_text': self._cache[cache_key],
                'source_language': source_language,
                'target_language': target_language,
                'confidence': None,
            }

        if not self._load_nllb():
            return {
                'original_text': text,
                'translated_text': f"Translation unavailable. (Model failed to load: {self._load_error})",
                'source_language': source_language,
                'target_language': target_language,
                'confidence': None,
            }
        try:
            if not raw:
                return {
                    'original_text': text,
                    'translated_text': '',
                    'source_language': source_language,
                    'target_language': target_language,
                    'confidence': None,
                }

            num_beams = 2 if fast else 5
            if not fast and (src_nllb == "arb_Arab" or tgt_nllb == "arb_Arab"):
                num_beams = 10 if src_nllb == "arb_Arab" else 8
            parts = _split_text_for_nllb(raw, src_nllb)

            translated_parts = []
            tgt_token_id = self.tokenizer.convert_tokens_to_ids(tgt_nllb)
            self.tokenizer.src_lang = src_nllb

            for segment in parts:
                for sub in _subsplit_long_segment(segment, src_nllb):
                    if not sub.strip():
                        continue
                    inputs = self.tokenizer(
                        sub,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    )
                    inputs = {k: v.to(self._tensor_device()) for k, v in inputs.items()}
                    gen_kw = {
                        **inputs,
                        "forced_bos_token_id": tgt_token_id,
                        "max_length": 512,
                        "num_beams": num_beams,
                        "early_stopping": True,
                    }
                    if src_nllb == "arb_Arab" and not fast:
                        gen_kw["no_repeat_ngram_size"] = 4
                    with torch.no_grad():
                        out_ids = self.model.generate(**gen_kw)
                    out = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
                    translated_parts.append(out)

            translated_text = " ".join(translated_parts)
            if tgt_nllb == "arb_Arab":
                translated_text = _scrub_mixed_latin_noise_in_arabic_output(translated_text)
            if len(self._cache_keys) >= self._cache_max:
                old = self._cache_keys.pop(0)
                self._cache.pop(old, None)
            self._cache[cache_key] = translated_text
            self._cache_keys.append(cache_key)
            return {
                'original_text': text,
                'translated_text': translated_text,
                'source_language': source_language,
                'target_language': target_language,
                'confidence': None,
            }
        except Exception as e:
            return {
                'original_text': text,
                'translated_text': f"Translation error: {str(e)}",
                'source_language': source_language,
                'target_language': target_language,
                'confidence': None,
            }

    def translate_batch(
        self,
        texts: List[str],
        target_language: str = 'en',
        source_language: Optional[str] = None,
        fast: bool = False,
    ) -> List[dict]:
        if not source_language:
            source_language = 'en'
        results = []
        for text in texts:
            results.append(self.translate_text(text, target_language, source_language, fast=fast))
        return results

    def get_supported_languages(self) -> dict:
        return dict(SUPPORTED_LANGUAGES)
