# Proposed Beta Timeline & Milestones (Fixed-Price Friendly)

**Scope assumption:** One primary document type (e.g. ID or bank statement page), REST API, one OCR backend + hook for cloud fallback, staging deployment, basic security.

**Total estimate:** **3–5 weeks** (1 engineer, spec provided upfront)  
**Extended estimate (PDF batch + similarity + security review):** **6–8 weeks**

---

## Milestone 1 — Spec & API skeleton (3–5 days)

**Deliverables**
- Review completed spec; confirm OCR engine strategy
- OpenAPI 3.0 contract (`POST /v1/documents/ocr`, `GET /health`)
- FastAPI project scaffold, Docker, staging environment
- API key or OAuth2 client-credentials auth (stub)

**Acceptance:** Authenticated health check returns 200 on staging URL

**Suggested fixed price:** [Your rate × 4 days]

---

## Milestone 2 — OCR extract endpoint (5–7 days)

**Deliverables**
- Image upload (JPEG/PNG) + single-page PDF
- Response JSON: `{ text, blocks[{text, bbox, confidence}], engine, processing_ms }`
- Integrate existing PaddleOCR worker pattern or Textract per spec
- Input validation, size limits, timeout handling

**Acceptance:** 10 sample docs from client spec OCR correctly in staging; p95 latency documented

**Suggested fixed price:** [Your rate × 6 days]

---

## Milestone 3 — Preprocessing & engine abstraction (5–7 days)

**Deliverables**
- Optional preprocess pipeline (deskew, denoise, contrast)
- `OcrEngine` interface: `PaddleEngine`, `TextractEngine`, `TesseractEngine`
- Config-driven engine selection
- Unit tests on golden files

**Acceptance:** Switch engines via env/config without API contract change

**Suggested fixed price:** [Your rate × 6 days]

---

## Milestone 4 — Security hardening (5–10 days)

**Deliverables**
- TLS termination (or document load balancer setup)
- Encrypted storage for uploads (SSE-S3 or equivalent)
- Secrets in vault/parameter store, not env files in prod
- Structured logs with request ID; no raw document text in logs
- Rate limiting, file type validation, virus scan hook (if required)

**Acceptance:** Security checklist signed off against client SOC 2 control list (beta subset)

**Suggested fixed price:** [Your rate × 7 days]

---

## Milestone 5 — Beta release & handoff (3–5 days)

**Deliverables**
- Docker images, deploy runbook, API documentation (Swagger/ReDoc)
- Basic load test report (e.g. 50 concurrent uploads)
- Error catalog and support runbook

**Acceptance:** Client can call beta endpoint from their sandbox with provided API key

**Suggested fixed price:** [Your rate × 4 days]

---

## Optional add-ons (separate milestones)

| Add-on | Effort | Deliverable |
|--------|--------|-------------|
| Document similarity API | 5–7 days | `POST /v1/documents/compare` → score + optional diff regions |
| Multi-page PDF async jobs | 7–10 days | Webhook/polling job status |
| Field extraction (KYC schema) | 10–15 days | Keyed JSON: name, DOB, ID number with validation |
| Fine-tuned OCR model | 2–4 weeks | Custom rec model + eval report (your `standalone_paddle_arabic_train.py` pattern) |

---

## Risk buffers

- **+1 week** if spec changes mid-sprint
- **+1 week** if compliance review blocks storage architecture
- **+3–5 days** if primary language needs custom training data

---

## Payment structure example

- 20% M1 · 25% M2 · 20% M3 · 25% M4 · 10% M5  
- Or: 30% kickoff · 40% beta endpoint live · 30% security sign-off
