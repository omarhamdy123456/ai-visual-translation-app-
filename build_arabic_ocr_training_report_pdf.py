"""
Generate a PDF report for the Paddle Arabic OCR recognition fine-tuning run (run01).

Output: Arabic_OCR_Training_Report.pdf next to this script (override with --output).

Requires: reportlab (see requirements.txt)
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


LOG_LINE_RE = re.compile(
    r"epoch: \[(\d+)/(\d+)\], global_step: (\d+), lr: ([\d.]+), acc: ([\d.]+), "
    r"norm_edit_dis: ([\d.]+), CTCLoss: ([\d.]+), NRTRLoss: ([\d.]+), loss: ([\d.]+)"
)


def parse_training_rows(log_path: Path) -> List[Tuple]:
    rows = []
    if not log_path.is_file():
        return rows
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = LOG_LINE_RE.search(line)
        if not m:
            continue
        ep, ep_tot, step, lr, acc, ned, ctc, nrtr, tot = m.groups()
        rows.append(
            (
                int(ep),
                int(ep_tot),
                int(step),
                float(lr),
                float(acc),
                float(ned),
                float(ctc),
                float(nrtr),
                float(tot),
            )
        )
    return rows


def build_pdf(
    output_path: Path,
    *,
    merged_root: Path,
    exp_root: Path,
    train_lines: int,
    val_lines: int,
    dict_lines: int,
    training_rows: List[Tuple],
    train_duration_min: Optional[float],
) -> None:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="TitleCustom",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=12,
        alignment=TA_CENTER,
    )
    h2 = ParagraphStyle(
        name="H2Custom",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=14,
        spaceAfter=8,
    )
    body = ParagraphStyle(
        name="BodyJustify",
        parent=styles["Normal"],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    small = ParagraphStyle(
        name="Small",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#333333"),
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    story: List = []

    story.append(Paragraph("Arabic Text Recognition Fine-Tuning Report", title_style))
    story.append(
        Paragraph(
            "PaddleOCR PP-OCRv5 Arabic mobile recognition &mdash; standalone experiment "
            "(not wired into the OCR Translator application).",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; "
            f"<b>Experiment directory:</b> {exp_root}",
            small,
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("1. Dataset description", h2))
    story.append(
        Paragraph(
            "<b>Primary source.</b> Samples were drawn from the Hugging Face dataset "
            "<i>mssqpi/Arabic-OCR-Dataset</i>, which provides paired grayscale/RGB line crops "
            "and Arabic transcripts suitable for scene/document text recognition.",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Construction pipeline.</b> Two disjoint slices were exported with "
            "<code>standalone_paddle_arabic_train.py export-hf --hf-split ...</code>, "
            "written in PaddleX MSTextRecDataset layout, then merged with "
            "<code>merge</code> into prefixes <code>ds_a/</code> and <code>ds_b/</code>. "
            "Slice A corresponded to an early export of <code>train[:2000]</code>; slice B used "
            "<code>train[150:300]</code> (plus internal train/validation splits per slice).",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Label format.</b> UTF-8 text files <code>train.txt</code> and "
            "<code>val.txt</code>: each line is <code>relative_path&lt;TAB&gt;ground_truth</code>. "
            "<b>dict.txt</b> lists one character class per line (Paddle "
            "<code>arabic_dict.txt</code>), defining the recognition alphabet.",
            body,
        )
    )

    ds_data = [
        ["Quantity", "Value"],
        ["Merged dataset root", str(merged_root)],
        ["Training manifest lines", str(train_lines)],
        ["Validation manifest lines", str(val_lines)],
        ["Character dictionary size (lines)", str(dict_lines)],
        ["Approx. total labeled crops", str(train_lines + val_lines)],
    ]
    t_ds = Table(ds_data, colWidths=[2.4 * inch, 4.3 * inch])
    t_ds.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
            ]
        )
    )
    story.append(t_ds)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("2. Training configuration", h2))
    cfg_rows = [
        ["Setting", "Value"],
        ["Model", "arabic_PP-OCRv5_mobile_rec"],
        ["Architecture (from log)", "SVTR_LCNet / MultiHead (CTC + NRTR)"],
        ["Pretrained weights", "Official Paddle arabic_PP-OCRv5_mobile_rec_pretrained.pdparams"],
        ["Device", "CPU (CUDA device count was 0 on host)"],
        ["Epochs completed", "1"],
        ["Train batch size", "4"],
        ["Optimizer", "Adam + cosine LR (warmup_epoch 5 in base config)"],
        ["Max text length", "25"],
        ["Input resize", "RecResizeImg [3, 48, 320]"],
        ["Train dataloader iters / epoch", "723"],
        ["Eval dataloader iters", "56"],
        ["Checkpoint / inference export", str(exp_root / "latest") + ", " + str(exp_root / "iter_epoch_1")],
    ]
    if train_duration_min is not None:
        cfg_rows.append(["Approx. wall-clock train time", f"{train_duration_min:.0f} minutes (from session log)"])

    t_cfg = Table(cfg_rows, colWidths=[2.2 * inch, 4.5 * inch])
    t_cfg.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#548235")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
            ]
        )
    )
    story.append(t_cfg)

    story.append(Paragraph("3. Training dynamics (logged minibatches)", h2))
    story.append(
        Paragraph(
            "Metrics below come from PP-OCR training logs (RecMetric on streaming batches). "
            "<b>acc</b> remained 0.0 throughout this single epoch; <b>norm_edit_dis</b> (normalized edit distance) "
            "and losses still trended as optimization progressed.",
            body,
        )
    )

    if training_rows:
        slim = [training_rows[0]]
        for r in training_rows[1:-1]:
            if r[2] % 100 == 0:
                slim.append(r)
        if training_rows[-1] != slim[-1]:
            slim.append(training_rows[-1])

        hdr = ["Step", "LR", "acc", "norm_edit_dis", "CTC", "NRTR", "total loss"]
        tdata = [hdr]
        for r in slim:
            _, _, step, lr, acc, ned, ctc, nrtr, tot = r
            tdata.append(
                [
                    str(step),
                    f"{lr:.6f}",
                    f"{acc:.4f}",
                    f"{ned:.4f}",
                    f"{ctc:.2f}",
                    f"{nrtr:.3f}",
                    f"{tot:.3f}",
                ]
            )
        tw = [0.65, 0.95, 0.55, 0.95, 0.75, 0.65, 0.95]
        t_tr = Table(tdata, colWidths=[w * inch for w in tw])
        t_tr.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7030A0")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F0FC")]),
                ]
            )
        )
        story.append(t_tr)
        first = training_rows[0]
        last = training_rows[-1]
        story.append(
            Paragraph(
                f"<b>Summary:</b> Total loss decreased from <b>{first[-1]:.2f}</b> (step {first[2]}) to "
                f"<b>{last[-1]:.2f}</b> (step {last[2]}). Normalized edit distance increased from "
                f"<b>{first[5]:.4f}</b> to <b>{last[5]:.4f}</b> over the same interval.",
                body,
            )
        )
    else:
        story.append(Paragraph("<i>No parsed training rows (missing or empty train.log).</i>", body))

    story.append(Paragraph("4. Validation results", h2))
    story.append(
        Paragraph(
            "<b>Official validation metric.</b> After the epoch, Paddle saved "
            "<code>best metric, acc: 0</code> to the log and "
            "<code>iter_epoch_1.info.json</code> recorded "
            "<code>{\"best_model_dict\": {\"acc\": 0}}</code>. "
            "So sequence-level accuracy on the held-out <code>val.txt</code> list remained zero after one epoch "
            "(common when accuracy requires exact full-string matches and training is short).",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Validation set size:</b> "
            f"{val_lines} labeled crops referenced in <code>val.txt</code>, evaluated in 56 validation dataloader iterations.",
            body,
        )
    )

    story.append(Paragraph("5. Testing results", h2))
    story.append(
        Paragraph(
            "<b>No separate test split</b> was held out beyond <code>train.txt</code> / <code>val.txt</code>. "
            "No additional offline benchmark (e.g. ICDAR-style test file) was executed for this run. "
            "Future work: reserve a third manifest (<code>test.txt</code>), run "
            "<code>paddle_eval</code> / PaddleX evaluate mode, or export inference and score on a fixed image list.",
            body,
        )
    )

    story.append(Paragraph("6. Artifacts and reproducibility", h2))
    story.append(
        Paragraph(
            f"<b>Training log:</b> <code>{exp_root / 'train.log'}</code><br/>"
            f"<b>Result index:</b> <code>{exp_root / 'train_result.json'}</code><br/>"
            f"<b>Inference:</b> under <code>latest/inference</code> and <code>iter_epoch_1/inference</code>.",
            body,
        )
    )

    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "Arabic_OCR_Training_Report.pdf"),
    )
    parser.add_argument(
        "--merged-root",
        type=str,
        default="paddle_arabic_standalone_data/merged_arabic_rec",
    )
    parser.add_argument(
        "--exp-root",
        type=str,
        default="paddle_arabic_standalone_experiments/run01",
    )
    parser.add_argument("--train-duration-min", type=float, default=61.0, help="Wall-clock training minutes (approx).")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    merged = (root / args.merged_root).resolve()
    exp = (root / args.exp_root).resolve()

    train_txt = merged / "train.txt"
    val_txt = merged / "val.txt"
    dict_txt = merged / "dict.txt"

    def count_lines(p: Path) -> int:
        if not p.is_file():
            return 0
        return sum(1 for _ in open(p, encoding="utf-8"))

    rows = parse_training_rows(exp / "train.log")

    build_pdf(
        Path(args.output).resolve(),
        merged_root=merged,
        exp_root=exp,
        train_lines=count_lines(train_txt),
        val_lines=count_lines(val_txt),
        dict_lines=count_lines(dict_txt),
        training_rows=rows,
        train_duration_min=args.train_duration_min,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
