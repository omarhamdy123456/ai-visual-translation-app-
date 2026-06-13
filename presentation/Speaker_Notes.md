# Speaker notes — Doctor presentation

## Slide 1 — Title
Introduce yourself and the project name **Translate Studio** (SkyTranslate in the UI). One sentence: multimodal OCR + translation for text, images, video, and handwriting.

## Slide 2 — Outline
Briefly walk through the flow of the talk so the supervisor knows what to expect.

## Slide 3 — Problem
Emphasize real-world pain: unreadable signs, menus, forms, subtitles. Existing apps are black boxes with weak OCR on Arabic and angled photos.

## Slide 4 — Objectives
Five bullet points map directly to thesis chapters: design, compare OCR, preprocess, evaluate, prototype.

## Slide 5 — System overview
End-to-end story: input → OCR if needed → translate → optional visual output. Mention Google Cloud as default translator.

## Slide 6 — Architecture
Point to separation of concerns: UI vs `ocr_translator.py` vs worker. Mention this is API-ready (FastAPI later).

## Slide 7 — Tech stack
Hit OCR engines and OpenCV preprocessing — this is the CV contribution. GPU/CPU toggle shows deployment awareness.

## Slide 8 — OCR methodology
Explain detection vs recognition. **Important:** Paddle runs **one model per language**, then merges — not one multilingual model.

## Slide 9 — Preprocessing
Why photos fail without it: glare, blur, skew. Mention dewarp for documents and optional OpenAI enhance for night shots.

## Slide 10 — Translation
Dedupe across video frames saves cost. Per-segment language inference fixes German/Arabic menu bugs.

## Slide 11 — Four modes
Quick demo roadmap: pick image + video for the live part; mention handwriting uses Vision API.

## Slide 12 — Evaluation design
Stress **fair benchmarks**: line crops for Arabic, full pages for English. Shows scientific rigor vs inflated word-crop scores.

## Slide 13 — Arabic results
CER 0.19 = moderate-good for line OCR. WER 0.60 = room to improve. Fine-tuning is the planned next step.

## Slide 14 — English FUNSD
Full-page is harder — honest result. Compare Paddle vs EasyOCR if asked.

## Slide 15 — Translation BLEU
Optional slide — NLLB offline benchmark. Production uses Google for quality.

## Slide 16 — Challenges fixed
Recent engineering wins: menu dedupe, digit “3”, bilingual merge, Windows worker. Shows iterative development.

## Slide 17 — Demo
Switch to live app or embedded screenshots. Prepare: menu image, bilingual sign, delivery app table.

## Slide 18 — Conclusions
Three takeaways: working system, measured OCR, modular architecture.

## Slide 19 — Future work
Fine-tune Arabic, REST API, form field extraction — aligns with thesis “future work” section.

## Slide 20 — Thank you
Invite questions on methodology, metrics, or demo.

---

### Likely supervisor questions

1. **Why multiple OCR engines?** — Different strengths: EasyOCR for general scene text, Paddle for Arabic/structured docs, TrOCR for handwriting.
2. **Why Google Translate vs NLLB?** — Quality and fluency for demo; NLLB kept for offline fallback and text benchmarks.
3. **Is CER 0.19 good enough?** — Acceptable for prototype; fine-tuning and preprocessing improve production use.
4. **What is your contribution vs libraries?** — Pipeline design, fair evaluation, multilingual merge logic, preprocessing, and integrated application.
