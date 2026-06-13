# Doctor / Supervisor Presentation

PowerPoint deck for thesis or progress meeting with your supervisor.

## Files

| File | Description |
|------|-------------|
| `Doctor_Supervisor_Presentation.pptx` | Main slide deck (generated) |
| `generate_doctor_presentation.py` | Script to rebuild the deck after edits |
| `Speaker_Notes.md` | Talking points per slide |

## Before you present

1. Open `Doctor_Supervisor_Presentation.pptx` in PowerPoint or Google Slides.
2. Replace placeholders on slide 1 and the last slide:
   - `[Your Name]`
   - `[Supervisor Name]`
   - `[Department / University]`
   - `[Your email]`
3. Add **2–4 screenshots** from the app (Annotated tab, translated photo, video frame) to the **Live Demo** slide or as extra slides after it.
4. Optional: add your university logo on the title slide.

## Regenerate slides

```powershell
python presentation\generate_doctor_presentation.py
```

## Suggested timing (20–25 min)

| Section | Slides | Minutes |
|---------|--------|---------|
| Intro & objectives | 1–5 | 4 |
| Architecture & methodology | 6–12 | 8 |
| Modes & evaluation | 13–18 | 5 |
| Results | 19–21 | 4 |
| Challenges, demo, conclusion | 22–26 | 4 |
| Q&A | last | 5 |

## Key numbers to remember

- **Arabic line OCR:** CER 0.19, 43% exact match (222 val samples)
- **English FUNSD full-page:** CER ~0.43 (PaddleOCR)
- **Translation (NLLB demo):** BLEU ~41.8 en→ar (text mode benchmark)
