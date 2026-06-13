"""
Prepare **line-level** image + transcript CSVs for ``trocr_train.py`` (Arabic handwriting friendly).

Writes under ``trocr_arabic_standalone_data/`` by default. Does **not** modify ``ocr_translator.py``.

Line crops = one image file per transcript (paragraph / line), matching how ``trocr_hybrid`` feeds TrOCR.

Examples::

    # Tab-separated manifest (UTF-8): path<TAB>text per line
    python standalone_trocr_arabic_prepare.py tsv-to-csv \\
        --in manifest.tsv --out trocr_arabic_standalone_data/full.csv

    python standalone_trocr_arabic_prepare.py split \\
        --csv trocr_arabic_standalone_data/full.csv \\
        --train-out trocr_arabic_standalone_data/train.csv \\
        --val-out trocr_arabic_standalone_data/val.csv --val-ratio 0.1

    python trocr_train.py \\
        --train-manifest trocr_arabic_standalone_data/train.csv \\
        --val-manifest trocr_arabic_standalone_data/val.csv \\
        --model-name microsoft/trocr-base-handwritten \\
        --extend-tokenizer arabic --max-target-length 128 \\
        --output-dir models/trocr-ar-handwritten --fp16

Optional Hugging Face dataset export (needs ``datasets``)::

    python standalone_trocr_arabic_prepare.py export-hf \\
        --dataset mssqpi/Arabic-OCR-Dataset \\
        --split train \\
        --image-column image --text-column text \\
        --out-dir trocr_arabic_standalone_data/from_hf
"""

from __future__ import annotations

import argparse
import csv
import io
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_ROOT = Path("trocr_arabic_standalone_data")


def _read_tsv_manifest(path: Path) -> List[Tuple[Path, str]]:
    base = path.parent
    rows: List[Tuple[Path, str]] = []
    with open(path, encoding="utf-8-sig") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n\r")
            if not line.strip():
                continue
            if "\t" not in line:
                raise ValueError(f"{path}:{lineno}: expected path<TAB>text")
            p, text = line.split("\t", 1)
            img = Path(p.strip())
            if not img.is_absolute():
                img = (base / img).resolve()
            if not img.is_file():
                raise FileNotFoundError(f"{path}:{lineno}: missing {img}")
            rows.append((img, text))
    return rows


def cmd_tsv_to_csv(args: argparse.Namespace) -> None:
    inp = Path(args.input_path).resolve()
    outp = Path(args.out).resolve()
    rows = _read_tsv_manifest(inp)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "text"])
        w.writeheader()
        for img, text in rows:
            w.writerow({"image_path": str(img.resolve()), "text": text})
    print(f"Wrote {len(rows)} rows to {outp} (columns image_path, text).")


def cmd_split(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv).resolve()
    train_out = Path(args.train_out).resolve()
    val_out = Path(args.val_out).resolve()
    ratio = float(args.val_ratio)
    seed = int(args.seed)

    rows: List[Dict[str, str]] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "image_path" not in reader.fieldnames:
            raise SystemExit("CSV needs columns image_path,text")
        tc = "text" if "text" in reader.fieldnames else reader.fieldnames[-1]
        for rec in reader:
            ip = (rec.get("image_path") or "").strip()
            tx = rec.get(tc, "") or ""
            if ip:
                rows.append({"image_path": ip, "text": tx})

    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * ratio)) if len(rows) >= 2 else 0
    if len(rows) >= 2 and n_val >= len(rows):
        n_val = max(1, len(rows) // 10)
    val_rows = rows[:n_val] if n_val else []
    train_rows = rows[n_val:] if n_val else rows

    for path, subset in ((train_out, train_rows), (val_out, val_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image_path", "text"])
            w.writeheader()
            for r in subset:
                w.writerow(r)
    print(
        f"Split {len(rows)} → train {len(train_rows)} ({train_out}), "
        f"val {len(val_rows)} ({val_out}), seed={seed}"
    )


def _pil_save_image(obj: Any, dest: Path) -> None:
    from PIL import Image

    dest.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "save"):
        obj.convert("RGB").save(dest)
        return
    if isinstance(obj, dict):
        if "bytes" in obj and obj["bytes"]:
            Image.open(io.BytesIO(obj["bytes"])).convert("RGB").save(dest)
            return
        if "path" in obj and obj["path"]:
            import shutil

            shutil.copy2(obj["path"], dest)
            return
    raise TypeError(f"Unsupported image object: {type(obj)!r}")


def cmd_export_hf(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("pip install datasets") from e

    out_dir = Path(args.out_dir).resolve()
    img_dir = out_dir / "images"
    ds = load_dataset(
        args.dataset,
        args.config if args.config else None,
        split=args.split,
        trust_remote_code=bool(args.trust_remote_code),
    )
    col_img = args.image_column
    col_txt = args.text_column
    max_n = int(args.max_samples) if args.max_samples else None

    rows: List[Dict[str, str]] = []
    for i, ex in enumerate(ds):
        if max_n is not None and i >= max_n:
            break
        fn = f"{i:08d}.png"
        dest = img_dir / fn
        _pil_save_image(ex[col_img], dest)
        text = ex[col_txt]
        if text is None:
            text = ""
        rows.append({"image_path": str(dest.resolve()), "text": str(text)})

    csv_path = out_dir / "manifest.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "text"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} samples to {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="TrOCR Arabic / handwriting CSV preparation.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_tsv = sub.add_parser("tsv-to-csv", help="Convert path<TAB>text manifest to trocr_train CSV.")
    p_tsv.add_argument("--in", dest="input_path", required=True)
    p_tsv.add_argument("--out", required=True)
    p_tsv.set_defaults(func=cmd_tsv_to_csv)

    p_sp = sub.add_parser("split", help="Random train/val split of a CSV.")
    p_sp.add_argument("--csv", required=True)
    p_sp.add_argument("--train-out", required=True)
    p_sp.add_argument("--val-out", required=True)
    p_sp.add_argument("--val-ratio", type=float, default=0.1)
    p_sp.add_argument("--seed", type=int, default=42)
    p_sp.set_defaults(func=cmd_split)

    p_hf = sub.add_parser("export-hf", help="Export a Hugging Face dataset split to images + CSV.")
    p_hf.add_argument("--dataset", required=True)
    p_hf.add_argument("--config", default="", help="Dataset config name if needed.")
    p_hf.add_argument("--split", default="train")
    p_hf.add_argument("--image-column", default="image")
    p_hf.add_argument("--text-column", default="text")
    p_hf.add_argument("--out-dir", default=str(DEFAULT_ROOT / "from_hf"))
    p_hf.add_argument("--max-samples", type=int, default=0, help="Limit rows (0 = all).")
    p_hf.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to load_dataset.",
    )
    p_hf.set_defaults(func=cmd_export_hf)

    ns = ap.parse_args()
    ns.func(ns)


if __name__ == "__main__":
    main()
