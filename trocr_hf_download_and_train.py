"""
Download a Hugging Face line OCR dataset to disk and fine-tune TrOCR (``trocr_train.py``).

Default source: ``mssqpi/Arabic-OCR-Dataset`` (~2.16M samples). Export uses contiguous slices of the
``train`` split for train/validation CSVs (no separate HF val split).

Examples::

    # Default: **small** subset (2k train / 200 val) — fast disk + training
    python trocr_hf_download_and_train.py

    # Larger subset when you are ready
    python trocr_hf_download_and_train.py --train-samples 25000 --val-samples 2500 --epochs 1

    # Smoke run (tiny export + few steps)
    python trocr_hf_download_and_train.py --train-samples 400 --val-samples 100 --max-steps 30 --epochs 1

Output layout (default ``trocr_arabic_hf_pipeline/``)::

    train.csv  val.csv  train_images/  val_images/
    models/trocr_hf_finetuned/   # checkpoints + final model
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_DATASET = "mssqpi/Arabic-OCR-Dataset"
DEFAULT_DATA_ROOT = Path("trocr_arabic_hf_pipeline")
DEFAULT_MODEL_DIR = Path("models/trocr_hf_finetuned")


def _pil_save_image(obj: Any, dest: Path) -> None:
    from PIL import Image

    dest.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "save"):
        obj.convert("RGB").save(dest)
        return
    if isinstance(obj, dict):
        if obj.get("bytes"):
            Image.open(io.BytesIO(obj["bytes"])).convert("RGB").save(dest)
            return
        if obj.get("path"):
            import shutil

            shutil.copy2(obj["path"], dest)
            return
    raise TypeError(f"Unsupported image object: {type(obj)!r}")


def _try_tqdm(it, total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(it, total=total, desc=desc)
    except Exception:
        return it


def export_split_to_csv(
    dataset_id: str,
    split_expr: str,
    image_col: str,
    text_col: str,
    out_csv: Path,
    image_subdir: Path,
    label: str,
) -> int:
    from datasets import load_dataset

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset_id, split=split_expr)
    n = len(ds)
    rows: List[Dict[str, str]] = []
    iterator = _try_tqdm(range(n), total=n, desc=label)
    for i in iterator:
        ex = ds[i]
        fn = f"{i:08d}.png"
        dest = image_subdir / fn
        _pil_save_image(ex[image_col], dest)
        t = ex[text_col]
        rows.append(
            {"image_path": str(dest.resolve()), "text": "" if t is None else str(t)}
        )

    import csv

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "text"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="HF OCR dataset → CSV → TrOCR fine-tuning.")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--text-column", default="text")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Writes train.csv, val.csv, train_images/, val_images/",
    )
    ap.add_argument(
        "--train-samples",
        type=int,
        default=2000,
        help="Rows taken from HF train split starting at index 0 (default 2000).",
    )
    ap.add_argument(
        "--val-samples",
        type=int,
        default=200,
        help="Rows taken after train slice for validation (default 200).",
    )
    ap.add_argument("--skip-download", action="store_true", help="Reuse existing CSVs/images.")
    ap.add_argument("--skip-train", action="store_true", help="Only export CSVs/images.")

    ap.add_argument("--model-name", default="microsoft/trocr-base-handwritten")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=3e-5)
    ap.add_argument("--train-batch-size", type=int, default=8)
    ap.add_argument(
        "--eval-batch-size",
        type=int,
        default=2,
        help="Low VRAM: use 1. Validation runs generate() and is memory-heavy.",
    )
    ap.add_argument(
        "--generation-num-beams",
        type=int,
        default=1,
        help="Beams during eval (1 = lowest VRAM).",
    )
    ap.add_argument("--gradient-accumulation-steps", type=int, default=2)
    ap.add_argument("--max-target-length", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--logging-steps", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.set_defaults(fp16=True)
    ap.add_argument(
        "--no-fp16",
        action="store_false",
        dest="fp16",
        help="Disable FP16 (use if GPU/driver issues).",
    )
    ap.add_argument("--extend-tokenizer", choices=("none", "corpus", "arabic"), default="arabic")
    ap.add_argument("--unicode-normalize", choices=("none", "nfc", "nfkc"), default="nfc")

    args = ap.parse_args()

    root = args.data_root.resolve()
    train_csv = root / "train.csv"
    val_csv = root / "val.csv"
    train_img = root / "train_images"
    val_img = root / "val_images"

    ts, vs = int(args.train_samples), int(args.val_samples)
    if ts < 1 or vs < 1:
        raise SystemExit("--train-samples and --val-samples must be >= 1.")

    if not args.skip_download:
        print(
            f"Exporting {ts} train + {vs} val rows from {args.dataset} "
            f"(slice train[:{ts}] and train[{ts}:{ts + vs}])…",
            flush=True,
        )
        n_tr = export_split_to_csv(
            args.dataset,
            f"train[:{ts}]",
            args.image_column,
            args.text_column,
            train_csv,
            train_img,
            "train export",
        )
        n_va = export_split_to_csv(
            args.dataset,
            f"train[{ts}:{ts + vs}]",
            args.image_column,
            args.text_column,
            val_csv,
            val_img,
            "val export",
        )
        print(f"Wrote {n_tr} train + {n_va} val rows under {root}", flush=True)
    else:
        if not train_csv.is_file() or not val_csv.is_file():
            raise SystemExit("--skip-download requires existing train.csv and val.csv.")

    if args.skip_train:
        print("Skipping training (--skip-train).", flush=True)
        return

    trocr_train = Path(__file__).resolve().parent / "trocr_train.py"
    cmd: List[str] = [
        sys.executable,
        str(trocr_train),
        "--train-manifest",
        str(train_csv),
        "--val-manifest",
        str(val_csv),
        "--model-name",
        args.model_name,
        "--output-dir",
        str(args.output_dir.resolve()),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--max-target-length",
        str(args.max_target_length),
        "--num-workers",
        str(args.num_workers),
        "--logging-steps",
        str(args.logging_steps),
        "--save-steps",
        str(args.save_steps),
        "--eval-steps",
        str(args.eval_steps),
        "--generation-num-beams",
        str(args.generation_num_beams),
        "--extend-tokenizer",
        args.extend_tokenizer,
        "--unicode-normalize",
        args.unicode_normalize,
    ]
    if args.max_steps > 0:
        cmd.extend(["--max-steps", str(args.max_steps)])
    if args.fp16:
        cmd.append("--fp16")

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
