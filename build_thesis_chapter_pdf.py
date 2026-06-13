from __future__ import annotations

import html
import re
from pathlib import Path
from typing import List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "Thesis_Chapter_Experimental_Results_and_Validation.md"
PDF_PATH = ROOT / "Thesis_Chapter_Experimental_Results_and_Validation.pdf"


def _register_fonts() -> tuple[str, str]:
    regular_candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\tahoma.ttf"),
    ]
    bold_candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf"),
        Path(r"C:\Windows\Fonts\tahomabd.ttf"),
    ]
    regular = next((p for p in regular_candidates if p.is_file()), None)
    bold = next((p for p in bold_candidates if p.is_file()), None)
    if regular:
        pdfmetrics.registerFont(TTFont("ThesisSans", str(regular)))
        regular_name = "ThesisSans"
    else:
        regular_name = "Helvetica"
    if bold:
        pdfmetrics.registerFont(TTFont("ThesisSans-Bold", str(bold)))
        bold_name = "ThesisSans-Bold"
    else:
        bold_name = "Helvetica-Bold"
    return regular_name, bold_name


FONT, FONT_BOLD = _register_fonts()


def inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", rf'<font name="{FONT}">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = text.replace("→", "&#8594;")
    text = text.replace("×", "x")
    text = text.replace("⁶", "6").replace("⁻", "-").replace("⁵", "5")
    text = text.replace(" ", " ")
    return text


def is_table_sep(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped.replace("|", "").replace("-", "").replace(":", "").strip()) == set()


def split_table_row(line: str) -> List[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def table_widths(n_cols: int, available: float) -> List[float]:
    if n_cols == 2:
        return [available * 0.28, available * 0.72]
    if n_cols == 3:
        return [available * 0.22, available * 0.25, available * 0.53]
    return [available / max(n_cols, 1)] * n_cols


def build_story(markdown: str):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontName=FONT_BOLD,
            fontSize=18,
            leading=22,
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading1Custom",
            parent=styles["Heading1"],
            fontName=FONT_BOLD,
            fontSize=15,
            leading=18,
            spaceBefore=12,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading2Custom",
            parent=styles["Heading2"],
            fontName=FONT_BOLD,
            fontSize=12,
            leading=15,
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCustom",
            parent=styles["BodyText"],
            fontName=FONT,
            fontSize=9.2,
            leading=12.5,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["BodyText"],
            fontName=FONT,
            fontSize=7.5,
            leading=9.4,
            wordWrap="CJK",
        )
    )

    story = []
    lines = markdown.splitlines()
    i = 0
    available_width = A4[0] - 4 * cm

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            story.append(Spacer(1, 0.12 * cm))
            i += 1
            continue

        if stripped == "---":
            story.append(Spacer(1, 0.2 * cm))
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and is_table_sep(lines[i + 1]):
            rows = [split_table_row(stripped)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_table_row(lines[i]))
                i += 1
            n_cols = max(len(r) for r in rows)
            data = []
            for row in rows:
                row = row + [""] * (n_cols - len(row))
                data.append([Paragraph(inline(cell), styles["TableCell"]) for cell in row])
            tbl = Table(data, colWidths=table_widths(n_cols, available_width), repeatRows=1)
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.white),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
                        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#888888")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(tbl)
            story.append(Spacer(1, 0.18 * cm))
            continue

        if stripped.startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(
                    ListItem(
                        Paragraph(inline(lines[i].strip()[2:]), styles["BodyCustom"]),
                        leftIndent=12,
                    )
                )
                i += 1
            story.append(ListFlowable(items, bulletType="bullet", start="bullet", leftIndent=16))
            story.append(Spacer(1, 0.08 * cm))
            continue

        numbered = re.match(r"^\d+\.\s+(.*)$", stripped)
        if numbered:
            items = []
            while i < len(lines):
                m = re.match(r"^\d+\.\s+(.*)$", lines[i].strip())
                if not m:
                    break
                items.append(
                    ListItem(Paragraph(inline(m.group(1)), styles["BodyCustom"]), leftIndent=14)
                )
                i += 1
            story.append(ListFlowable(items, bulletType="1", leftIndent=18))
            story.append(Spacer(1, 0.08 * cm))
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(inline(stripped[2:]), styles["TitleCenter"]))
            i += 1
            continue
        if stripped.startswith("## "):
            story.append(Paragraph(inline(stripped[3:]), styles["Heading1Custom"]))
            i += 1
            continue
        if stripped.startswith("### "):
            story.append(Paragraph(inline(stripped[4:]), styles["Heading2Custom"]))
            i += 1
            continue

        para = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (
                not nxt
                or nxt.startswith("#")
                or nxt.startswith("|")
                or nxt.startswith("- ")
                or re.match(r"^\d+\.\s+", nxt)
                or nxt == "---"
            ):
                break
            para.append(nxt)
            i += 1
        story.append(Paragraph(inline(" ".join(para)), styles["BodyCustom"]))

    return story


def main() -> None:
    markdown = MD_PATH.read_text(encoding="utf-8")
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.7 * cm,
        title="Experimental Results and Validation",
        author="OCR Translation Project",
    )
    doc.build(build_story(markdown))
    print(PDF_PATH)


if __name__ == "__main__":
    main()
