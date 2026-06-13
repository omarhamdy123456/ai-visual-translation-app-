"""
Standalone EasyOCR **recognition** dataset prep + trainer config helpers.

EasyOCR trains recognition with the upstream ``trainer/`` tree (modified
deep-text-recognition-benchmark). Line crops go in one folder per split with
``labels.csv`` (columns ``filename``, ``words``) beside the images — see
https://github.com/JaidedAI/EasyOCR/tree/master/trainer

This script writes only under ``easyocr_standalone_data/`` (default). It does **not**
wire custom weights into ``ocr_translator.py``.

Typical workflow::

    # 1) Manifest UTF-8: each line = image_path<TAB>label  (TAB separates path from rest)
    python standalone_easyocr_rec_train.py prepare \\
        --manifest crops_manifest.tsv --out easyocr_standalone_data/ar_en_lines

    python standalone_easyocr_rec_train.py check --dataset-dir easyocr_standalone_data/ar_en_lines

    python standalone_easyocr_rec_train.py split \\
        --dataset-dir easyocr_standalone_data/ar_en_lines \\
        --out-dir easyocr_standalone_data/ar_en_split --val-ratio 0.1

    python standalone_easyocr_rec_train.py characters \\
        --labels-dir easyocr_standalone_data/ar_en_split/train \\
        --preset arabic_g1 --merge-label-chars

    python standalone_easyocr_rec_train.py trainer-yaml \\
        --experiment-name ar_custom_rec \\
        --character "$(python standalone_easyocr_rec_train.py characters ...)" \\
        --train-select-folder ar_custom_train \\
        --valid-path all_data/ar_custom_valid \\
        --out easyocr_standalone_experiments/ar_custom_config.yaml

Training (after cloning EasyOCR and installing trainer deps)::

    cd EasyOCR/trainer
    # Copy prepared folders into trainer/all_data/, e.g.
    #   all_data/ar_custom_train/  all_data/ar_custom_valid/
    pip install torch torchvision lmdb natsort pillow pandas numpy opencv-python
    python train.py --yaml_path path/to/ar_custom_config.yaml

Inference bundle (after ``best_accuracy.pth`` or ``iter_*.pth``)::

    # Per custom_model.md you need three matching names in EasyOCR dirs:
    #   ~/.EasyOCR/user_network/<name>.yaml  <name>.py
    #   ~/.EasyOCR/model/<name>.pth
    # Use the official custom_example.zip from EasyOCR modelhub as a template for .py/.yaml.
    easyocr.Reader(["ar", "en"], recog_network="<name>",
                   user_network_directory="...", model_storage_directory="...")
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

DEFAULT_DATA_ROOT = Path("easyocr_standalone_data")
DEFAULT_EXP_ROOT = Path("easyocr_standalone_experiments")


def _preset_characters(name: str) -> str:
    """Character string bundled with EasyOCR pretrained recognition models."""
    try:
        from easyocr.config import recognition_models
    except ImportError as e:
        raise SystemExit(
            "easyocr is required for --preset (pip install easyocr)."
        ) from e
    key = (name or "").strip().lower().replace("-", "_")
    aliases = {
        "arabic": "arabic_g1",
        "ar": "arabic_g1",
        "english": "english_g2",
        "en": "english_g2",
        "latin": "latin_g2",
    }
    key = aliases.get(key, key)
    if key in recognition_models.get("gen1", {}):
        return str(recognition_models["gen1"][key]["characters"])
    if key in recognition_models.get("gen2", {}):
        return str(recognition_models["gen2"][key]["characters"])
    raise SystemExit(
        f"Unknown preset {name!r}. Try: arabic_g1, english_g2, latin_g2, ..."
    )


def _read_manifest(path: Path) -> List[Tuple[Path, str]]:
    rows: List[Tuple[Path, str]] = []
    base = path.parent
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n\r")
            if not line.strip():
                continue
            if "\t" not in line:
                raise ValueError(
                    f"{path}:{lineno}: expected path<TAB>label (no TAB found)."
                )
            p, label = line.split("\t", 1)
            p = p.strip()
            label = label.replace("\r", "")
            img_path = Path(p)
            if not img_path.is_absolute():
                img_path = (base / img_path).resolve()
            if not img_path.is_file():
                raise FileNotFoundError(f"{path}:{lineno}: missing image {img_path}")
            rows.append((img_path, label))
    return rows


def _read_labels_csv(folder: Path) -> List[Tuple[str, str]]:
    csv_path = folder / "labels.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}")
    rows: List[Tuple[str, str]] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "filename" not in reader.fieldnames:
            raise ValueError(f"{csv_path}: need CSV header with filename,words")
        words_col = "words" if "words" in reader.fieldnames else reader.fieldnames[-1]
        for rec in reader:
            fn = (rec.get("filename") or "").strip()
            w = rec.get(words_col, "")
            if fn:
                rows.append((fn, w))
    return rows


def _write_labels_csv(folder: Path, rows: Sequence[Tuple[str, str]]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "labels.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["filename", "words"])
        for fn, text in rows:
            w.writerow([fn, text])


def cmd_prepare(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest).resolve()
    out = Path(args.out).resolve()
    rows = _read_manifest(manifest)
    out.mkdir(parents=True, exist_ok=True)
    manifest_lines: List[Tuple[str, str]] = []
    skipped = 0
    for i, (src, label) in enumerate(rows):
        if not label.strip():
            skipped += 1
            continue
        ext = src.suffix.lower() or ".png"
        if ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"):
            ext = ".png"
        dest_name = f"{i + 1:06d}{ext}"
        dest = out / dest_name
        shutil.copy2(src, dest)
        manifest_lines.append((dest_name, label))
    _write_labels_csv(out, manifest_lines)
    print(f"Wrote {len(manifest_lines)} samples to {out} (skipped empty labels: {skipped}).")


def cmd_check(args: argparse.Namespace) -> None:
    folder = Path(args.dataset_dir).resolve()
    rows = _read_labels_csv(folder)
    missing = 0
    for fn, _ in rows:
        if not (folder / fn).is_file():
            missing += 1
            print(f"MISSING: {fn}")
    print(f"labels.csv entries: {len(rows)}, missing files: {missing}")


def cmd_split(args: argparse.Namespace) -> None:
    src = Path(args.dataset_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    rows = _read_labels_csv(src)
    rng = random.Random(int(args.seed))
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    if not rows:
        raise SystemExit("split: labels.csv is empty.")
    n_val = max(1, int(len(rows) * float(args.val_ratio))) if len(rows) >= 2 else 0
    if len(rows) >= 2 and n_val >= len(rows):
        n_val = max(1, len(rows) // 10)
    val_idx = set(indices[:n_val]) if n_val else set()
    train_rows: List[Tuple[str, str]] = []
    val_rows: List[Tuple[str, str]] = []
    dest_train = out_root / args.train_name
    dest_val = out_root / args.val_name
    for i, (fn, text) in enumerate(rows):
        sub = dest_val if i in val_idx else dest_train
        dest_folder = sub
        dest_folder.mkdir(parents=True, exist_ok=True)
        src_img = src / fn
        new_fn = fn
        dest_img = dest_folder / new_fn
        if dest_img.resolve() != src_img.resolve():
            shutil.copy2(src_img, dest_img)
        bucket = val_rows if i in val_idx else train_rows
        bucket.append((new_fn, text))
    _write_labels_csv(dest_train, train_rows)
    _write_labels_csv(dest_val, val_rows)
    print(
        f"Split {len(rows)} → train {len(train_rows)} ({dest_train}), "
        f"val {len(val_rows)} ({dest_val}), seed={args.seed}"
    )
    if not val_rows:
        print("Warning: validation set is empty (need more samples or lower --val-ratio).")


def _unique_chars(texts: Iterable[str]) -> str:
    s: Set[str] = set()
    for t in texts:
        for ch in t:
            s.add(ch)
    # Stable Unicode codepoint order (Arabic + Latin mixed OK).
    return "".join(sorted(s, key=lambda c: ord(c)))


def cmd_characters(args: argparse.Namespace) -> None:
    labels_dir = Path(args.labels_dir).resolve()
    rows = _read_labels_csv(labels_dir)
    texts = [w for _, w in rows]
    out_chars = ""
    presets = [p.strip() for p in (args.preset or "").split(",") if p.strip()]
    for p in presets:
        out_chars += _preset_characters(p)
    if args.merge_label_chars:
        out_chars += _unique_chars(texts)
    # Deduplicate while preserving order
    seen: Set[str] = set()
    merged: List[str] = []
    for ch in out_chars:
        if ch not in seen:
            seen.add(ch)
            merged.append(ch)
    result = "".join(merged)
    out_path = Path(args.out_file).resolve() if args.out_file else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result, encoding="utf-8")
        print(f"Wrote {len(result)} chars to {out_path}")
    else:
        sys.stdout.write(result + "\n")


def _yaml_escape(s: str) -> str:
    """Double-quoted YAML string with UTF-8 characters preserved."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def cmd_trainer_yaml(args: argparse.Namespace) -> None:
    char_source = (args.character or "").strip()
    if args.character_file:
        char_source = Path(args.character_file).read_text(encoding="utf-8").strip()
    if not char_source:
        raise SystemExit("Provide --character or --character-file.")
    exp = args.experiment_name
    lines = [
        f"experiment_name: {_yaml_escape(exp)}",
        "train_data: 'all_data'",
        f"valid_data: {_yaml_escape(args.valid_path)}",
        "manualSeed: 1111",
        f"workers: {int(args.workers)}",
        f"batch_size: {int(args.batch_size)}",
        f"num_iter: {int(args.num_iter)}",
        f"valInterval: {int(args.val_interval)}",
        "saved_model: ''",
        f"FT: {str(bool(args.ft)).lower()}",
        "optim: False",
        "lr: 1.",
        "beta1: 0.9",
        "rho: 0.95",
        "eps: 0.00000001",
        "grad_clip: 5",
        f"select_data: {_yaml_escape(args.train_select_folder)}",
        "batch_ratio: '1'",
        "total_data_usage_ratio: 1.0",
        f"batch_max_length: {int(args.batch_max_length)}",
        f"imgH: {int(args.img_h)}",
        f"imgW: {int(args.img_w)}",
        "rgb: False",
        "sensitive: True",
        "PAD: True",
        "contrast_adjust: 0.0",
        "data_filtering_off: False",
        "Transformation: 'None'",
        "FeatureExtraction: 'VGG'",
        "SequenceModeling: 'BiLSTM'",
        "Prediction: 'CTC'",
        "num_fiducial: 20",
        "input_channel: 1",
        "output_channel: 512",
        "hidden_size: 512",
        "decode: 'greedy'",
        "new_prediction: False",
        "freeze_FeatureFxtraction: False",
        "freeze_SequenceModeling: False",
        f"character: {_yaml_escape(char_source)}",
    ]
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(
        "Copy train/valid folders under EasyOCR/trainer/all_data/ to match "
        "train_select_folder and valid_path; then run train.py from trainer/."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="EasyOCR recognition dataset prep helpers.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Manifest TSV → folder with labels.csv + images.")
    p_prep.add_argument("--manifest", required=True, help="UTF-8 file: path<TAB>label per line.")
    p_prep.add_argument(
        "--out",
        default=str(DEFAULT_DATA_ROOT / "rec_lines"),
        help="Output directory (default: easyocr_standalone_data/rec_lines).",
    )
    p_prep.set_defaults(func=cmd_prepare)

    p_chk = sub.add_parser("check", help="Verify labels.csv vs image files.")
    p_chk.add_argument("--dataset-dir", required=True)
    p_chk.set_defaults(func=cmd_check)

    p_split = sub.add_parser("split", help="Train/val split (copies images).")
    p_split.add_argument("--dataset-dir", required=True)
    p_split.add_argument(
        "--out-dir",
        default=str(DEFAULT_DATA_ROOT / "rec_split"),
        help="Parent folder for train/val subfolders.",
    )
    p_split.add_argument("--train-name", default="train")
    p_split.add_argument("--val-name", default="valid")
    p_split.add_argument("--val-ratio", type=float, default=0.1)
    p_split.add_argument("--seed", type=int, default=42)
    p_split.set_defaults(func=cmd_split)

    p_char = sub.add_parser(
        "characters",
        help="Build merged charset string (preset + optional label chars). Prints to stdout unless --out-file.",
    )
    p_char.add_argument("--labels-dir", required=True, help="Folder with labels.csv.")
    p_char.add_argument(
        "--preset",
        default="arabic_g1",
        help="Comma-separated easyocr.config presets (e.g. arabic_g1,english_g2).",
    )
    p_char.add_argument(
        "--no-merge-label-chars",
        dest="merge_label_chars",
        action="store_false",
        help="Use only --preset charset (labels must not contain extra glyphs). "
        "Default: merge characters seen in labels.csv.",
    )
    p_char.set_defaults(merge_label_chars=True)
    p_char.add_argument("--out-file", default="", help="Write charset to this file.")
    p_char.set_defaults(func=cmd_characters)

    p_yaml = sub.add_parser(
        "trainer-yaml",
        help="Write a starter trainer YAML (generation1 / Arabic-sized CRNN).",
    )
    p_yaml.add_argument("--out", required=True)
    p_yaml.add_argument("--experiment-name", default="custom_rec")
    p_yaml.add_argument(
        "--character",
        default="",
        help="Full character string for CTC (same as training labels charset).",
    )
    p_yaml.add_argument("--character-file", default="", help="Read charset from file.")
    p_yaml.add_argument(
        "--train-select-folder",
        default="custom_train",
        help="Subfolder name under trainer/all_data/ for training crops.",
    )
    p_yaml.add_argument(
        "--valid-path",
        default="all_data/custom_valid",
        help="Path relative to trainer cwd for validation leaf folder.",
    )
    p_yaml.add_argument("--workers", type=int, default=4)
    p_yaml.add_argument("--batch-size", type=int, default=32)
    p_yaml.add_argument("--num-iter", type=int, default=300000)
    p_yaml.add_argument("--val-interval", type=int, default=5000)
    p_yaml.add_argument("--batch-max-length", type=int, default=48)
    p_yaml.add_argument("--img-h", type=int, default=64)
    p_yaml.add_argument("--img-w", type=int, default=600)
    p_yaml.add_argument("--ft", action="store_true", help="Fine-tune from saved_model.")
    p_yaml.set_defaults(func=cmd_trainer_yaml)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
