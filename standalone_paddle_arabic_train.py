"""
Standalone Paddle PP-OCR Arabic **recognition** fine-tuning (merge two datasets → PaddleX).

This stack writes only under ``paddle_arabic_standalone_data/`` and
``paddle_arabic_standalone_experiments/`` by default. It does **not** wire trained
weights into ``ocr_translator.py``, ``app.py``, ``main.py``, or ``paddle_ocr_worker.py``.

Requires (train/check): ``pip install paddlex paddlepaddle`` — use the CUDA build from
https://www.paddlepaddle.org.cn/install/quick if you want GPU.

Recognition layout (PaddleX MSTextRecDataset): ::

    dataset_root/
      dict.txt          # one character per line (use Paddle ``arabic_dict.txt`` for PP-OCR Arabic)
      train.txt         # relative_image_path<TAB>transcript (UTF-8)
      val.txt
      ... image files paths relative to dataset_root ...

Examples::

    python standalone_paddle_arabic_train.py merge \\
        --dataset-a path/to/paddle_ds1 --dataset-b path/to/paddle_ds2

    python standalone_paddle_arabic_train.py check --dataset-dir paddle_arabic_standalone_data/merged_arabic_rec

    python standalone_paddle_arabic_train.py train \\
        --dataset-dir paddle_arabic_standalone_data/merged_arabic_rec \\
        --output-dir paddle_arabic_standalone_experiments/run01 \\
        --device gpu:0 --epochs 20

Detection models are **not** trained here (different label format). This script targets
Arabic **text recognition** fine-tuning only.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

DEFAULT_MERGED_ROOT = Path("paddle_arabic_standalone_data") / "merged_arabic_rec"
DEFAULT_EXP_ROOT = Path("paddle_arabic_standalone_experiments") / "arabic_rec"
PADDLE_ARABIC_DICT_URL = (
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/"
    "ppocr/utils/dict/arabic_dict.txt"
)


def _ensure_dict(dest: Path, source_path: Optional[Path], url: str) -> None:
    if dest.is_file():
        return
    if source_path and source_path.is_file():
        shutil.copy2(source_path, dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _load_dict_chars(dict_path: Path) -> Set[str]:
    chars: Set[str] = set()
    with open(dict_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line:
                chars.add(line)
    return chars


def _read_rec_lines(txt_path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(txt_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Expected path<TAB>label in {txt_path}, got: {line[:120]!r}")
            rel, text = parts[0].strip(), parts[1]
            rows.append((rel.replace("\\", "/"), text))
    return rows


def _write_rec_lines(path: Path, rows: Sequence[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rel, text in rows:
            f.write(f"{rel}\t{text}\n")


def _filter_by_dict(
    rows: Sequence[Tuple[str, str]], allowed: Set[str], strict: bool
) -> Tuple[List[Tuple[str, str]], int]:
    kept: List[Tuple[str, str]] = []
    skipped = 0
    for rel, text in rows:
        bad = [c for c in text if c not in allowed]
        if bad:
            skipped += 1
            if strict:
                continue
        kept.append((rel, text))
    return kept, skipped


def cmd_merge(args: argparse.Namespace) -> None:
    root_a = Path(args.dataset_a).resolve()
    root_b = Path(args.dataset_b).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    dict_dest = out / "dict.txt"
    src_dict = Path(args.dict_path).resolve() if args.dict_path else None
    _ensure_dict(dict_dest, src_dict, args.dict_url)

    allowed = _load_dict_chars(dict_dest)
    train_a = _read_rec_lines(root_a / "train.txt")
    val_a = _read_rec_lines(root_a / "val.txt")
    train_b = _read_rec_lines(root_b / "train.txt")
    val_b = _read_rec_lines(root_b / "val.txt")

    def merge_side(
        prefix: str, root: Path, rows: Sequence[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        merged_rows: List[Tuple[str, str]] = []
        for rel, text in rows:
            src = (root / rel).resolve()
            if not src.is_file():
                raise FileNotFoundError(f"Missing image for label file: {src}")
            dst_rel = f"{prefix}/{rel}"
            dst = out / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            merged_rows.append((dst_rel, text))
        return merged_rows

    out_train = merge_side("ds_a", root_a, train_a) + merge_side("ds_b", root_b, train_b)
    out_val = merge_side("ds_a", root_a, val_a) + merge_side("ds_b", root_b, val_b)

    if args.strict_dict:
        out_train, s1 = _filter_by_dict(out_train, allowed, strict=True)
        out_val, s2 = _filter_by_dict(out_val, allowed, strict=True)
        if s1 or s2:
            print(f"strict-dict: skipped {s1} train and {s2} val lines (character not in dict.txt)")
    else:
        out_train, s1 = _filter_by_dict(out_train, allowed, strict=False)
        out_val, s2 = _filter_by_dict(out_val, allowed, strict=False)
        if s1 or s2:
            print(
                f"warning: {s1} train / {s2} val lines contain characters outside dict.txt "
                "(training may fail for those samples; use --strict-dict to drop them)"
            )

    _write_rec_lines(out / "train.txt", out_train)
    _write_rec_lines(out / "val.txt", out_val)
    print(f"Wrote merged dataset ({len(out_train)} train, {len(out_val)} val) → {out}")


def _pick_image_text_keys(
    column_names: Sequence[str],
    image_key: Optional[str],
    text_key: Optional[str],
) -> Tuple[str, str]:
    if image_key and text_key:
        return image_key, text_key
    img_cands = ("image", "img", "cropped_image", "line_image", "filepath", "file_name")
    txt_cands = ("text", "label", "transcription", "gt", "sentence", "ground_truth")
    ik = image_key
    tk = text_key
    if not ik:
        for c in img_cands:
            if c in column_names:
                ik = c
                break
    if not tk:
        for c in txt_cands:
            if c in column_names:
                tk = c
                break
    if not ik or not tk:
        raise ValueError(
            f"Could not infer image/text columns from {list(column_names)}; "
            "pass --image-key and --text-key explicitly."
        )
    return ik, tk


def _example_to_pil_image(ex: Dict, image_key: str):
    from PIL import Image

    v = ex.get(image_key)
    if v is None:
        raise KeyError(image_key)
    if hasattr(v, "convert"):
        return v.convert("RGB")
    import numpy as np

    if isinstance(v, dict) and "bytes" in v:
        import io

        return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
    arr = np.asarray(v)
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    if arr.ndim == 3:
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"Unsupported image type for key {image_key}: {type(v)}")


def cmd_export_hf(args: argparse.Namespace) -> None:
    from datasets import DatasetDict, load_dataset

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dict_dest = out / "dict.txt"
    _ensure_dict(dict_dest, Path(args.dict_path).resolve() if args.dict_path else None, args.dict_url)

    train_name = args.train_split
    val_name = args.val_split
    hf_split = getattr(args, "hf_split", None)

    if hf_split:
        # e.g. train[:5000] or train[5000:10000] — single Dataset; val comes from --val-ratio
        train_ds = load_dataset(
            args.hf_dataset,
            args.hf_config,
            split=hf_split,
        )
        val_ds = None
    else:
        ds = load_dataset(args.hf_dataset, args.hf_config)
        if isinstance(ds, DatasetDict):
            if train_name not in ds:
                raise ValueError(f"No split {train_name!r} in dataset keys {list(ds.keys())}")
            train_ds = ds[train_name]
            val_ds = ds[val_name] if val_name in ds else None
        else:
            train_ds = ds
            val_ds = None

    cols = train_ds.column_names
    image_key, text_key = _pick_image_text_keys(cols, args.image_key, args.text_key)

    def export_rows(sub_ds, split_label: str) -> List[Tuple[str, str]]:
        rows: List[Tuple[str, str]] = []
        for i, ex in enumerate(sub_ds):
            text = ex.get(text_key)
            if text is None:
                continue
            text = str(text).strip()
            if not text:
                continue
            pil = _example_to_pil_image(ex, image_key)
            fname = f"{split_label}_{i:06d}.png"
            rel = f"images/{split_label}/{fname}"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            pil.save(out / rel)
            rows.append((rel, text))
        return rows

    train_rows = export_rows(train_ds, "train")

    if val_ds is not None and len(val_ds) > 0:
        val_rows = export_rows(val_ds, "val")
    else:
        ratio = float(args.val_ratio)
        if not (0.0 < ratio < 1.0):
            raise ValueError("--val-ratio must be between 0 and 1 when no validation split exists")
        if len(train_rows) < 2:
            raise ValueError("Not enough train samples to create a validation split")
        rng = random.Random(args.seed)
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        n_val = max(1, int(len(train_rows) * ratio))
        val_rows = [train_rows[i] for i in order[:n_val]]
        train_rows = [train_rows[i] for i in order[n_val:]]
        print(
            f"No '{val_name}' split; held out {n_val} samples from train as val (seed={args.seed})."
        )

    _write_rec_lines(out / "train.txt", train_rows)
    _write_rec_lines(out / "val.txt", val_rows)
    print(f"Exported HF dataset {args.hf_dataset!r} → {out} ({len(train_rows)} train, {len(val_rows)} val)")


def _paddlex_config_path() -> Path:
    try:
        import paddlex  # type: ignore

        p = Path(paddlex.__file__).resolve().parent / "configs/modules/text_recognition/arabic_PP-OCRv5_mobile_rec.yaml"
        if p.is_file():
            return p
    except Exception:
        pass
    raise FileNotFoundError(
        "Could not find arabic_PP-OCRv5_mobile_rec.yaml inside paddlex. "
        "Install paddlex: pip install paddlex"
    )


def _run_paddlex_engine(overrides: List[str]) -> None:
    old = sys.argv[:]
    try:
        cfg = _paddlex_config_path()
        sys.argv = [old[0] if old else "standalone_paddle_arabic_train", "-c", str(cfg)]
        for o in overrides:
            sys.argv.extend(["-o", o])
        from paddlex.engine import Engine  # noqa: WPS433 — after argv patch

        Engine().run()
    finally:
        sys.argv = old


def cmd_check(args: argparse.Namespace) -> None:
    ds = Path(args.dataset_dir).resolve()
    overrides = [
        f"Global.mode=check_dataset",
        f"Global.dataset_dir={ds}",
        f"Global.output={Path(args.output).resolve()}",
        f"Global.device={args.device}",
    ]
    _run_paddlex_engine(overrides)


def cmd_train(args: argparse.Namespace) -> None:
    ds = Path(args.dataset_dir).resolve()
    od = Path(args.output_dir).resolve()
    od.mkdir(parents=True, exist_ok=True)
    overrides = [
        "Global.mode=train",
        f"Global.dataset_dir={ds}",
        f"Global.output={od}",
        f"Global.device={args.device}",
        f"Train.epochs_iters={args.epochs}",
        f"Train.batch_size={args.batch_size}",
        f"Train.learning_rate={args.learning_rate}",
        f"Train.log_interval={args.log_interval}",
        f"Train.eval_interval={args.eval_interval}",
        f"Train.save_interval={args.save_interval}",
    ]
    if args.pretrain_url:
        overrides.append(f"Train.pretrain_weight_path={args.pretrain_url}")
    _run_paddlex_engine(overrides)
    print(f"Training finished; artifacts under {od}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone Paddle Arabic rec training (not wired to the Streamlit app).")
    sub = p.add_subparsers(dest="command", required=True)

    pm = sub.add_parser("merge", help="Merge two Paddle-format recognition folders into one combined dataset.")
    pm.add_argument("--dataset-a", required=True, type=str, help="First dataset root (train.txt, val.txt, images).")
    pm.add_argument("--dataset-b", required=True, type=str, help="Second dataset root.")
    pm.add_argument("--output", type=str, default=str(DEFAULT_MERGED_ROOT))
    pm.add_argument("--dict-path", type=str, default=None, help="Optional dict.txt to copy (default: download Paddle arabic_dict).")
    pm.add_argument("--dict-url", type=str, default=PADDLE_ARABIC_DICT_URL)
    pm.add_argument(
        "--strict-dict",
        action="store_true",
        help="Drop lines whose labels contain characters not listed in dict.txt.",
    )
    pm.set_defaults(func=cmd_merge)

    pe = sub.add_parser(
        "export-hf",
        help="Export a Hugging Face image+text dataset to Paddle recognition format (then merge two exports if needed).",
    )
    pe.add_argument("--hf-dataset", required=True, type=str, help="HF dataset id, e.g. melsiddieg/qari-arabic-ocr-10k")
    pe.add_argument("--hf-config", type=str, default=None)
    pe.add_argument("--train-split", type=str, default="train")
    pe.add_argument("--val-split", type=str, default="validation")
    pe.add_argument(
        "--hf-split",
        type=str,
        default=None,
        help="HF datasets split DSL (overrides --train-split/--val-split), e.g. train[:4000] or train[4000:8000].",
    )
    pe.add_argument("--image-key", type=str, default=None)
    pe.add_argument("--text-key", type=str, default=None)
    pe.add_argument("--val-ratio", type=float, default=0.1, help="If no val split, split this fraction from train.")
    pe.add_argument("--seed", type=int, default=42)
    pe.add_argument("--output", type=str, required=True)
    pe.add_argument("--dict-path", type=str, default=None)
    pe.add_argument("--dict-url", type=str, default=PADDLE_ARABIC_DICT_URL)
    pe.set_defaults(func=cmd_export_hf)

    pc = sub.add_parser("check", help="PaddleX check_dataset on a prepared folder.")
    pc.add_argument("--dataset-dir", required=True, type=str)
    pc.add_argument("--output", type=str, default="paddle_arabic_standalone_check_output")
    pc.add_argument("--device", type=str, default="cpu")
    pc.set_defaults(func=cmd_check)

    pt = sub.add_parser("train", help="Fine-tune arabic_PP-OCRv5_mobile_rec via PaddleX (requires paddlex + paddlepaddle).")
    pt.add_argument("--dataset-dir", required=True, type=str)
    pt.add_argument("--output-dir", type=str, default=str(DEFAULT_EXP_ROOT))
    pt.add_argument("--device", type=str, default="gpu:0", help="e.g. gpu:0 or cpu")
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--batch-size", type=int, default=8)
    pt.add_argument("--learning-rate", type=float, default=0.001)
    pt.add_argument("--log-interval", type=int, default=20)
    pt.add_argument("--eval-interval", type=int, default=1)
    pt.add_argument("--save-interval", type=int, default=1)
    pt.add_argument(
        "--pretrain-url",
        type=str,
        default=None,
        help="Override pretrained weights URL (default: Arabic PP-OCRv5 mobile rec from PaddleX yaml).",
    )
    pt.set_defaults(func=cmd_train)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
