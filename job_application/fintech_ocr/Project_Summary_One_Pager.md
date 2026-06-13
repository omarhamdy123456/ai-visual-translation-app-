# Translate Studio — OCR Portfolio Summary (One Pager)

**Purpose:** Supporting document for fintech OCR / backend developer applications  
**Candidate:** [Your Name]  
**Project type:** Document & image OCR pipeline (Python)

---

## Elevator pitch

Translate Studio is a **multimodal OCR system** I built for document-like content: photos, forms, video subtitles, and handwriting. It combines **multiple OCR engines**, **measurable accuracy benchmarks**, and a **worker-isolated inference layer** suitable for wrapping as a production REST API.

---

## Technical stack

| Layer | Technologies |
|-------|----------------|
| OCR engines | PaddleOCR 3.x, EasyOCR, TrOCR (Hugging Face) |
| CV / preprocessing | OpenCV, PIL, optional RIDNet denoising, LAB contrast, deskew |
| ML / translation (adjacent) | NLLB-200, Google Cloud Translate API |
| Runtime | Python 3.11+, PyTorch, subprocess OCR workers |
| Evaluation | jiwer (CER/WER), custom fair benchmarks |
| UI (demo) | Streamlit — **not** the production API target |

---

## OCR capabilities (job-relevant)

- **Multi-engine OCR** with language-aware routing (Arabic, English, Latin scripts, CJK)
- **Structured output:** text + quadrilateral bounding boxes + confidence scores
- **Full-page and line-level** processing paths
- **Image preprocessing** before OCR (denoise, sharpen, resolution control)
- **Isolated Paddle worker** (`paddle_ocr_worker.py`) — avoids DLL/GPU conflicts, supports daemon mode for batch/video
- **Benchmark discipline:** fair test sets only (line crops for Arabic; full-page FUNSD for English)

---

## Measured results (pretrained models)

| Test | Samples | CER | WER | Notes |
|------|---------|-----|-----|-------|
| Arabic line OCR (`merged_arabic_rec` val) | 222 | **0.19** | 0.60 | Fair line-level labels (≤10 chars) |
| English forms (FUNSD full-page test) | 50 | **0.43** | 0.58 | Full document, not word crops |

*Sources: `experimental_results/ocr_results.json`, `experimental_results/funsd_fullpage_test.json`*

---

## Architecture (API-ready separation)

```
Client / Streamlit UI
        │
        ▼
  ocr_translator.py  ── orchestration, preprocess, merge boxes
        │
        ├── subprocess ──► paddle_ocr_worker.py (PaddleOCR daemon)
        ├── EasyOCR / TrOCR (in-process or hybrid)
        └── eval_experimental_results.py (CER/WER benchmarks)
```

**For fintech API:** replace UI with **FastAPI** → same worker + preprocess → JSON response + encrypted storage.

---

## Mapping to fintech OCR product needs

| Client need | Current project evidence |
|-------------|-------------------------|
| Document OCR | Image + form OCR, full-page eval |
| ID / form fields | Bounding boxes + text per region (extend to field schema) |
| Engine flexibility | Paddle + EasyOCR + TrOCR; cloud translate already integrated |
| Quality metrics | CER/WER pipeline, reproducible eval scripts |
| Secure handling | Env-based credentials; CPU-only mode; no secrets in code |
| Similarity / comparison | Frame dedup, preprocess normalization — **extend to doc similarity API** |

---

## Honest gaps (and how I close them for your project)

| Gap in demo | Beta delivery plan |
|-------------|-------------------|
| Streamlit, not REST API | FastAPI + OpenAPI + auth |
| No AWS Textract / Vision yet | Adapter interface; integrate per your spec |
| No formal SOC 2 audit | Implement controls you specify (encryption, logging, retention) |
| No similarity endpoint | Add `/compare` with hash/SSIM/embeddings |

---

## Key files to reference in interviews

| File | Role |
|------|------|
| `paddle_ocr_worker.py` | OCR inference worker (daemon + one-shot) |
| `ocr_translator.py` | OCR orchestration, preprocess, video/image pipelines |
| `eval_experimental_results.py` | CER/WER evaluation |
| `benchmark_funsd_fullpage.py` | Full-page English OCR benchmark |
| `standalone_paddle_arabic_train.py` | OCR model training/export pipeline |
| `experimental_results/` | JSON/CSV benchmark outputs |

---

## Contact

[Your email] · [GitHub] · [Portfolio URL]
