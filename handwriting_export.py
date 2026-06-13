"""Build downloadable PDF / PNG exports for handwriting workflow."""

from __future__ import annotations

import io
import os
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def build_bilingual_pdf_bytes(
    title: str,
    original_text: str,
    translated_text: str,
    target_lang_label: str,
) -> bytes:
    """Simple bilingual PDF using ReportLab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as e:
        raise ImportError("PDF export requires reportlab. Install: pip install reportlab") from e

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()
    story = []
    title_style = ParagraphStyle(
        name="HWTitle",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=12,
    )
    h2 = ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=12, spaceAfter=6, spaceBefore=10)
    body = ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=10, leading=14)

    def esc(s: str) -> str:
        return (
            (s or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )

    story.append(Paragraph(esc(title or "Handwriting export"), title_style))
    story.append(Paragraph(esc(f"Translation: {target_lang_label}"), body))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("<b>Original (OCR)</b>", h2))
    story.append(Paragraph(esc(original_text or "(empty)"), body))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(f"<b>Translated ({esc(target_lang_label)})</b>", h2))
    story.append(Paragraph(esc(translated_text or "(empty)"), body))
    doc.build(story)
    return buf.getvalue()


def build_bilingual_png_bytes(
    title: str,
    original_text: str,
    translated_text: str,
    target_lang_label: str,
    width: int = 1100,
) -> bytes:
    """Raster summary page (no OpenAI required): two columns Original | Translated."""
    w = max(640, min(1600, int(width)))
    margin = 36
    col_gap = 28
    inner_w = w - 2 * margin
    col_w = (inner_w - col_gap) // 2
    title_h = 56
    pad_y = 20

    def pick_font(size: int) -> ImageFont.ImageFont:
        for fp in (
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            if os.path.isfile(fp):
                try:
                    return ImageFont.truetype(fp, size)
                except OSError:
                    continue
        return ImageFont.load_default()

    font_title = pick_font(22)
    font_h = pick_font(15)
    font_b = pick_font(13)

    def wrap_lines(text: str, font: ImageFont.ImageFont, max_px: int) -> list[str]:
        lines: list[str] = []
        for para in (text or "").split("\n"):
            words = para.split()
            if not words:
                lines.append("")
                continue
            cur = words[0]
            for word in words[1:]:
                test = cur + " " + word
                try:
                    tw = font.getlength(test)
                except Exception:
                    tw = len(test) * 7
                if tw <= max_px:
                    cur = test
                else:
                    lines.append(cur)
                    cur = word
            lines.append(cur)
        return lines

    o_lines = wrap_lines(original_text, font_b, col_w - 8)
    t_lines = wrap_lines(translated_text, font_b, col_w - 8)
    line_h = 18
    body_h = pad_y + (max(len(o_lines), len(t_lines)) + 2) * line_h + pad_y
    h = title_h + body_h + margin

    img = Image.new("RGB", (w, h), (248, 250, 252))
    dr = ImageDraw.Draw(img)
    dr.rectangle((0, 0, w, title_h + 8), fill=(30, 41, 59))
    dr.text((margin, 16), title or "Handwriting export", fill=(248, 250, 252), font=font_title)
    dr.text((margin, title_h - 6), f"Target: {target_lang_label}", fill=(203, 213, 225), font=font_h)

    y0 = title_h + 12
    dr.text((margin, y0), "Original (OCR)", fill=(15, 23, 42), font=font_h)
    dr.text((margin + col_w + col_gap, y0), f"Translated", fill=(15, 23, 42), font=font_h)
    y = y0 + 28
    xo, xt = margin, margin + col_w + col_gap
    for i, ln in enumerate(o_lines):
        dr.text((xo, y + i * line_h), ln[:500], fill=(30, 41, 59), font=font_b)
    for i, ln in enumerate(t_lines):
        dr.text((xt, y + i * line_h), ln[:500], fill=(30, 64, 175), font=font_b)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
