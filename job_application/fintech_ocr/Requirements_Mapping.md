# Job Requirements → Evidence Mapping

Use this table in proposals, interviews, or the "Additional details" field.

| Job requirement | Your evidence | Talking point |
|-----------------|---------------|---------------|
| **OCR tools (Tesseract, Textract, Vision, or equivalent)** | PaddleOCR 3.x, EasyOCR, TrOCR; Google Cloud Translate for text (adjacent) | "Equivalent" open-source + cloud; can add Textract/Vision via adapter in week 1–2 of beta |
| **Secure REST APIs in Python** | Modular Python OCR core; subprocess workers; eval CLI | Demo is Streamlit; core maps to FastAPI in M1–M2 |
| **Document / image processing** | Preprocess pipeline, full-page + line OCR, video frame OCR, handwriting path | End-to-end doc ingestion experience |
| **CV / ML image comparison** | Video near-duplicate skip, preprocess normalization, bbox merge | Propose `/compare` endpoint: SSIM, pHash, or CLIP embeddings on normalized crops |
| **SOC 2 aligned security** | `.env` credentials, CPU-only mode, worker isolation | Commit to: TLS, encrypted S3/blob storage, secrets manager, audit logs, PII redaction, retention TTL |
| **Hands-on OCR (not generic dev)** | CER/WER benchmarks, fair vs unfair test methodology, training script for Arabic rec | Show you measure OCR quality scientifically |
| **Fixed / milestone contracts** | See `Beta_Timeline_Milestones.md` | 5 milestones, 3–5 week beta estimate |

---

## Proof points to attach or link

1. Screenshot: Streamlit OCR result with bounding boxes (Annotated tab)
2. `experimental_results/ocr_results.json` — Arabic metrics
3. `experimental_results/funsd_fullpage_test.json` — English full-page metrics
4. Architecture snippet from `Project_Summary_One_Pager.md`
5. (Optional) 2-min screen recording: upload image → OCR → structured text

---

## Questions to ask the client (shows seniority)

1. Document types: IDs, bank statements, invoices, checks?
2. Languages and scripts required?
3. Cloud OCR mandatory (Textract/Vision) or on-prem allowed?
4. Output schema: plain text vs keyed fields (name, DOB, account #)?
5. Data retention and deletion SLA?
6. Expected volume (RPM) and max file size?
7. SOC 2 Type I vs II scope for beta vs GA?
