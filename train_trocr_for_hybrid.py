"""
Train the **TrOCR recognizer** used by OCR backend ``trocr_hybrid``.

``trocr_hybrid`` = EasyOCR **detection** (boxes) + **TrOCR** **recognition** (text inside boxes).
You do **not** train a single combined checkpoint: CRAFT/EasyOCR stays pretrained unless you
train detection separately (EasyOCR trainer). This script is the same CLI as ``trocr_train.py``
with a short reminder banner.

**Data:** line crops — each row is one cropped line image + transcript (CSV columns
``image_path``, ``text``). Build manifests with ``standalone_trocr_arabic_prepare.py`` or your own CSV.

**Use the result in the app**

- Environment: ``OCR_TROCR_MODEL=R:\\project\\models\\trocr-hybrid-ft`` (directory saved by training).
- Streamlit: OCR backend **trocr_hybrid**, TrOCR model path = that folder.
- Or ``ImageTextTranslator(..., ocr_backend='trocr_hybrid', trocr_model='path/to/dir')``.

Arabic / mixed script: add ``--extend-tokenizer arabic`` or ``corpus`` (see ``trocr_train.py``).
"""

from __future__ import annotations


def _banner() -> None:
    print(
        "Training TrOCR for trocr_hybrid — EasyOCR detection stays default/pretrained; "
        "only the TrOCR weights you save here replace recognition.\n",
        flush=True,
    )


def main() -> None:
    _banner()
    from trocr_train import main as _train_main

    _train_main()


if __name__ == "__main__":
    main()
