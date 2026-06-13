"""
Standalone Paddle **text detection** fine-tuning (PP-OCR detector — finds boxes only).

Improves **where** text is located (boxes / polygons). Recognition is a separate model.

Outputs default to ``paddle_det_standalone_experiments/`` — **not** wired into ``paddle_ocr_worker.py`` / the app
until you point PaddleOCR at custom ``det_model_dir`` yourself.

Requires: ``pip install paddlex paddlepaddle`` (same stack as recognition training).

Dataset layout (Paddle ``TextDetDataset``)::

    dataset_root/
      train.txt
      val.txt
      images/...   (paths in txt files are relative to dataset_root)

Each line in train.txt / val.txt::

    relative/path/to/page.png\\t[{"transcription": "###", "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}, ...]

Use ``###`` as transcription when you only care about boxes (standard for det-only).

Examples::

    python standalone_paddle_det_train.py prepare-from-coco --coco \\
        paddle_arabic_standalone_data/hf_ar_slice_a/../merged..  # see --help

    python standalone_paddle_det_train.py check --dataset-dir my_det_data

    python standalone_paddle_det_train.py train --dataset-dir my_det_data \\
        --output-dir paddle_det_standalone_experiments/run01 --device gpu:0 --epochs 100
"""

from __future__ import annotations

import argparse
import json
import locale
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_EXP = Path("paddle_det_standalone_experiments") / "det_run01"


def _ensure_paddle_det_api_configs() -> None:
    """Copy PaddleX text_detection YAMLs into PaddleOCR_api/configs (same pip-gap as recognition)."""
    try:
        import paddlex  # type: ignore

        root = Path(paddlex.__file__).resolve().parent
        api_cfg = root / "repo_apis" / "PaddleOCR_api" / "configs"
        src_dir = root / "configs" / "modules" / "text_detection"
        if not src_dir.is_dir():
            return
        api_cfg.mkdir(parents=True, exist_ok=True)
        for yml in src_dir.glob("*.yaml"):
            dst = api_cfg / yml.name
            if not dst.is_file():
                shutil.copy2(yml, dst)
        po_root = root / "repo_manager" / "repos" / "PaddleOCR"
        if po_root.is_dir():
            os.environ.setdefault("PADDLE_PDX_PADDLEOCR_PATH", str(po_root.resolve()))
    except Exception:
        pass


def _paddlex_det_config_path(model: str) -> Path:
    import paddlex  # type: ignore

    root = Path(paddlex.__file__).resolve().parent
    p = root / "configs" / "modules" / "text_detection" / f"{model}.yaml"
    if p.is_file():
        return p
    raise FileNotFoundError(f"Missing {p}; install paddlex")


def _run_paddlex_engine(config_path: Path, overrides: List[str]) -> None:
    old = sys.argv[:]
    _orig_pref_enc = locale.getpreferredencoding

    def _utf8_preferred(*_a: Any, **_k: Any) -> str:
        return "utf-8"

    try:
        os.environ.setdefault("PYTHONUTF8", "1")
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        locale.getpreferredencoding = _utf8_preferred  # type: ignore[assignment]
        _ensure_paddle_det_api_configs()
        sys.argv = [old[0] if old else "standalone_paddle_det_train", "-c", str(config_path)]
        for o in overrides:
            sys.argv.extend(["-o", o])
        from paddlex.engine import Engine  # noqa: WPS433

        Engine().run()
    finally:
        locale.getpreferredencoding = _orig_pref_enc  # type: ignore[assignment]
        sys.argv = old


def _bbox_to_quad(bbox: List[float]) -> List[List[float]]:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def cmd_prepare_from_coco(args: argparse.Namespace) -> None:
    """Build train.txt / val.txt from COCO JSON (e.g. FUNSD-style export)."""
    root = Path(args.data_root).resolve()
    out = Path(args.output_dir).resolve() if args.output_dir else root
    out.mkdir(parents=True, exist_ok=True)

    def process_coco(coco_path: Path, list_name: str) -> int:
        with open(coco_path, encoding="utf-8") as f:
            coco = json.load(f)
        images_by_id = {im["id"]: im for im in coco["images"]}
        anns_by_img: Dict[int, List[Dict]] = defaultdict(list)
        for a in coco.get("annotations", []):
            anns_by_img[a["image_id"]].append(a)
        lines_out: List[str] = []
        for img_id, im in images_by_id.items():
            anns = anns_by_img.get(img_id, [])
            if not anns:
                continue
            fn = str(im["file_name"]).replace("\\", "/")
            parts: List[Dict[str, Any]] = []
            for ann in anns:
                bb = ann.get("bbox")
                if not bb or len(bb) < 4:
                    continue
                parts.append({"transcription": "###", "points": _bbox_to_quad([float(t) for t in bb[:4]])})
            if not parts:
                continue
            label = json.dumps(parts, ensure_ascii=False)
            lines_out.append(f"{fn}\t{label}")
        target = out / list_name
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            for ln in lines_out:
                f.write(ln + "\n")
        return len(lines_out)

    train_n = process_coco(Path(args.train_coco).resolve(), "train.txt")
    val_n = process_coco(Path(args.val_coco).resolve(), "val.txt")
    print(f"Wrote {out}/train.txt ({train_n} images with boxes), val.txt ({val_n}).")


def cmd_check(args: argparse.Namespace) -> None:
    ds = Path(args.dataset_dir).resolve()
    cfg = _paddlex_det_config_path(args.model)
    overrides = [
        "Global.mode=check_dataset",
        f"Global.dataset_dir={ds}",
        f"Global.output={Path(args.output).resolve()}",
        f"Global.device={args.device}",
        f"Global.model={args.model}",
    ]
    _run_paddlex_engine(cfg, overrides)


def cmd_train(args: argparse.Namespace) -> None:
    ds = Path(args.dataset_dir).resolve()
    od = Path(args.output_dir).resolve()
    od.mkdir(parents=True, exist_ok=True)
    cfg = _paddlex_det_config_path(args.model)
    overrides = [
        "Global.mode=train",
        f"Global.model={args.model}",
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
    _run_paddlex_engine(cfg, overrides)
    print(f"Training finished; checkpoints under {od}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paddle PP-OCR text detection training (standalone).")
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser(
        "prepare-from-coco",
        help="Build Paddle det train.txt/val.txt from COCO detection JSON (bbox → quad, transcription ###).",
    )
    pp.add_argument("--data-root", required=True, help="Root that contains image paths referenced in COCO.")
    pp.add_argument("--train-coco", required=True, help="Path to train COCO JSON.")
    pp.add_argument("--val-coco", required=True, help="Path to val COCO JSON.")
    pp.add_argument(
        "--output-dir",
        default=None,
        help="Where to write train.txt and val.txt (default: same as --data-root).",
    )
    pp.set_defaults(func=cmd_prepare_from_coco)

    pc = sub.add_parser("check", help="PaddleX check_dataset for a TextDetDataset folder.")
    pc.add_argument("--dataset-dir", required=True)
    pc.add_argument("--output", default="paddle_det_standalone_check_output")
    pc.add_argument("--device", default="cpu")
    pc.add_argument("--model", default="PP-OCRv5_mobile_det")
    pc.set_defaults(func=cmd_check)

    pt = sub.add_parser("train", help="Fine-tune text detector (PP-OCRv5_mobile_det by default).")
    pt.add_argument("--dataset-dir", required=True)
    pt.add_argument("--output-dir", type=str, default=str(DEFAULT_EXP))
    pt.add_argument("--model", default="PP-OCRv5_mobile_det")
    pt.add_argument("--device", default="gpu:0")
    pt.add_argument("--epochs", type=int, default=100)
    pt.add_argument("--batch-size", type=int, default=4)
    pt.add_argument("--learning-rate", type=float, default=0.001)
    pt.add_argument("--log-interval", type=int, default=10)
    pt.add_argument("--eval-interval", type=int, default=1)
    pt.add_argument("--save-interval", type=int, default=1)
    pt.add_argument("--pretrain-url", type=str, default=None)
    pt.set_defaults(func=cmd_train)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
