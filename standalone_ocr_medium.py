"""
Standalone OCR experiments: prepare medium-sized data and train recognition + detection.

Outputs go under ocr_standalone_data/ and ocr_standalone_experiments/ only.
This project does NOT load these checkpoints in ocr_translator, app.py, or main.py.

Recognition: fine-tune TrOCR via trocr_train.py on word crops (FUNSD by default).
Detection: fine-tune Faster R-CNN on page images with word boxes as single-class "text".
This is not CRAFT/EasyOCR detector training; it is an isolated experiment stack whose weights are never loaded by the Streamlit app.

Requires: pip install datasets torchvision (torch already in requirements.txt).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

DEFAULT_DATA_ROOT = Path("ocr_standalone_data")
DEFAULT_EXP_ROOT = Path("ocr_standalone_experiments")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _funsd_boxes_words(example: Dict[str, Any]) -> Tuple[List[str], List[List[int]]]:
    words = example["words"]
    bboxes = example["bboxes"]
    return words, bboxes


def _clamp_box(box: List[float], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(round(x1))))
    x2 = max(0, min(w, int(round(x2))))
    y1 = max(0, min(h - 1, int(round(y1))))
    y2 = max(0, min(h, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def prepare_from_funsd(
    out_root: Path,
    hf_dataset: str = "nielsr/funsd",
    train_split: str = "train",
    val_split: str = "test",
) -> Tuple[Path, Path, Path, Path]:
    """Export word crops + manifests and detection COCO-style JSON."""
    from datasets import load_dataset

    ds_train = load_dataset(hf_dataset, split=train_split)
    ds_val = load_dataset(hf_dataset, split=val_split)

    rec_img_dir = out_root / "recognition" / "images"
    det_img_dir = out_root / "detection" / "images"
    _ensure_dir(rec_img_dir / train_split)
    _ensure_dir(rec_img_dir / val_split)
    _ensure_dir(det_img_dir / train_split)
    _ensure_dir(det_img_dir / val_split)

    train_csv = out_root / "recognition" / "train.csv"
    val_csv = out_root / "recognition" / "val.csv"

    def process_split(ds, split_name: str, csv_path: Path, coco_path: Path) -> None:
        coco_images: List[Dict[str, Any]] = []
        coco_anns: List[Dict[str, Any]] = []
        ann_id = 1
        rec_rows: List[Tuple[str, str]] = []

        for idx, ex in enumerate(ds):
            img: Image.Image = ex["image"].convert("RGB")
            w, h = img.size
            stem = f"{split_name}_{idx:05d}"
            page_path = det_img_dir / split_name / f"{stem}.png"
            img.save(page_path)

            coco_images.append(
                {
                    "id": idx + 1,
                    "file_name": str(page_path.relative_to(out_root)),
                    "width": w,
                    "height": h,
                }
            )

            words, boxes = _funsd_boxes_words(ex)
            for wi, (text, bbox) in enumerate(zip(words, boxes)):
                cb = _clamp_box([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])], w, h)
                if cb is None:
                    continue
                x1, y1, x2, y2 = cb
                tw, th = x2 - x1, y2 - y1
                if tw < 2 or th < 2:
                    continue
                label = (text or "").strip()
                if not label:
                    continue

                crop = img.crop((x1, y1, x2, y2))
                crop_name = f"{stem}_w{wi:04d}.png"
                crop_rel = Path("recognition") / "images" / split_name / crop_name
                crop_abs = rec_img_dir / split_name / crop_name
                crop.save(crop_abs)

                rec_rows.append((str((out_root / crop_rel).resolve()), label))

                coco_anns.append(
                    {
                        "id": ann_id,
                        "image_id": idx + 1,
                        "category_id": 1,
                        "bbox": [float(x1), float(y1), float(tw), float(th)],
                        "area": float(tw * th),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["image_path", "text"])
            for path_s, t in rec_rows:
                wcsv.writerow([path_s, t])

        coco = {
            "images": coco_images,
            "annotations": coco_anns,
            "categories": [{"id": 1, "name": "text"}],
        }
        with open(coco_path, "w", encoding="utf-8") as f:
            json.dump(coco, f)

    train_coco = out_root / "detection" / "train_coco.json"
    val_coco = out_root / "detection" / "val_coco.json"

    process_split(ds_train, train_split, train_csv, train_coco)
    process_split(ds_val, val_split, val_csv, val_coco)

    print(f"Wrote recognition train manifest: {train_csv}")
    print(f"Wrote recognition val manifest:   {val_csv}")
    print(f"Wrote detection COCO: {train_coco}, {val_coco}")
    return train_csv, val_csv, train_coco, val_coco


def cmd_prepare(args: argparse.Namespace) -> None:
    out = Path(args.data_root).resolve()
    _ensure_dir(out)
    prepare_from_funsd(out, hf_dataset=args.hf_dataset)


def cmd_train_recognition(args: argparse.Namespace) -> None:
    root = Path(args.data_root).resolve()
    train_csv = root / "recognition" / "train.csv"
    val_csv = root / "recognition" / "val.csv"
    out_dir = Path(args.output_dir).resolve()
    _ensure_dir(out_dir)

    trocr = _repo_root() / "trocr_train.py"
    if not trocr.is_file():
        raise FileNotFoundError(f"Missing {trocr}")

    cmd = [
        sys.executable,
        str(trocr),
        "--train-manifest",
        str(train_csv),
        "--val-manifest",
        str(val_csv),
        "--model-name",
        args.model_name,
        "--output-dir",
        str(out_dir),
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
    ]
    if args.fp16:
        cmd.append("--fp16")
    if args.bf16:
        cmd.append("--bf16")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


class CocoDetectionDataset(torch.utils.data.Dataset):
    """Minimal COCO detection loader (single category)."""

    def __init__(self, data_root: Path, coco_json: Path):
        self.root = data_root
        with open(coco_json, encoding="utf-8") as f:
            coco = json.load(f)
        self.images = {im["id"]: im for im in coco["images"]}
        anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for a in coco["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)
        with_ann = set(anns_by_img.keys())
        self.ids = sorted(i for i in self.images if i in with_ann)
        self.anns_by_img = anns_by_img

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        import torchvision.transforms.functional as F

        img_id = self.ids[index]
        meta = self.images[img_id]
        path = self.root / meta["file_name"]
        pil = Image.open(path).convert("RGB")
        tensor = F.to_tensor(pil)

        boxes = []
        labels = []
        for ann in self.anns_by_img.get(img_id, []):
            x, y, bw, bh = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            boxes.append([x1, y1, x2, y2])
            labels.append(1)

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(labels),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}
        return tensor, target


def _collate_detection(batch):
    return tuple(zip(*batch))


def _get_detection_model(num_classes: int = 2):
    import torchvision
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def cmd_train_detection(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).resolve()
    train_json = data_root / "detection" / "train_coco.json"
    val_json = data_root / "detection" / "val_coco.json"
    if not train_json.is_file():
        raise FileNotFoundError(f"Missing {train_json}; run prepare first.")

    want_cuda = str(args.device).startswith("cuda")
    if want_cuda and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
        if want_cuda:
            print("CUDA not available; using CPU.")

    ds_train = CocoDetectionDataset(data_root, train_json)
    ds_val = CocoDetectionDataset(data_root, val_json)

    loader_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate_detection,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    loader_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_detection,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = _get_detection_model(num_classes=2)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    out_dir = Path(args.output_dir).resolve()
    _ensure_dir(out_dir)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for images, targets in loader_train:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            losses.backward()
            optimizer.step()

            epoch_loss += float(losses.detach().cpu())
            n_batches += 1

        lr_scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} train loss: {avg:.4f}")

        # Faster R-CNN only returns loss dict in training mode.
        model.train()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for images, targets in loader_val:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += float(sum(loss_dict.values()).detach().cpu())
                vn += 1
        vavg = val_loss / max(vn, 1)
        print(f"          val loss: {vavg:.4f}")

        ckpt_path = out_dir / f"detection_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        if vavg < best_loss:
            best_loss = vavg
            torch.save(model.state_dict(), out_dir / "detection_best.pth")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_prepare(args)
    rec_args = argparse.Namespace(**vars(args))
    rec_args.output_dir = str(Path(args.rec_exp_dir).resolve())
    cmd_train_recognition(rec_args)
    det_args = argparse.Namespace(
        data_root=args.data_root,
        output_dir=args.det_exp_dir,
        epochs=args.det_epochs,
        batch_size=args.det_batch_size,
        lr=args.det_lr,
        lr_step=args.det_lr_step,
        num_workers=args.det_num_workers,
        device=args.det_device,
    )
    cmd_train_detection(det_args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone OCR prepare + train (not used by the app).")
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("prepare", help="Download FUNSD and export crops + COCO detection labels.")
    pp.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    pp.add_argument("--hf-dataset", default="nielsr/funsd")
    pp.set_defaults(func=cmd_prepare)

    pr = sub.add_parser("train-recognition", help="Run trocr_train.py on prepared CSVs.")
    pr.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    pr.add_argument("--output-dir", default=str(DEFAULT_EXP_ROOT / "trocr_medium"))
    pr.add_argument("--model-name", default="microsoft/trocr-base-printed")
    pr.add_argument("--epochs", type=float, default=5.0)
    pr.add_argument("--learning-rate", type=float, default=3e-5)
    pr.add_argument("--train-batch-size", type=int, default=8)
    pr.add_argument("--eval-batch-size", type=int, default=8)
    pr.add_argument("--gradient-accumulation-steps", type=int, default=2)
    pr.add_argument("--max-target-length", type=int, default=96)
    pr.add_argument("--num-workers", type=int, default=2)
    pr.add_argument("--logging-steps", type=int, default=50)
    pr.add_argument("--save-steps", type=int, default=500)
    pr.add_argument("--eval-steps", type=int, default=500)
    pr.add_argument("--fp16", action="store_true")
    pr.add_argument("--bf16", action="store_true")
    pr.set_defaults(func=cmd_train_recognition)

    pd = sub.add_parser("train-detection", help="Train Faster R-CNN on prepared COCO JSON.")
    pd.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    pd.add_argument("--output-dir", default=str(DEFAULT_EXP_ROOT / "detection_frcnn"))
    pd.add_argument("--epochs", type=int, default=6)
    pd.add_argument("--batch-size", type=int, default=2)
    pd.add_argument("--lr", type=float, default=0.005)
    pd.add_argument("--lr-step", type=int, default=4)
    pd.add_argument("--num-workers", type=int, default=2)
    pd.add_argument("--device", default="cuda:0")
    pd.set_defaults(func=cmd_train_detection)

    pa = sub.add_parser("all", help="prepare + train-recognition + train-detection")
    pa.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    pa.add_argument("--hf-dataset", default="nielsr/funsd")
    pa.add_argument("--rec-exp-dir", default=str(DEFAULT_EXP_ROOT / "trocr_medium"))
    pa.add_argument("--det-exp-dir", default=str(DEFAULT_EXP_ROOT / "detection_frcnn"))
    pa.add_argument("--model-name", default="microsoft/trocr-base-printed")
    pa.add_argument("--epochs", type=float, default=5.0)
    pa.add_argument("--learning-rate", type=float, default=3e-5)
    pa.add_argument("--train-batch-size", type=int, default=8)
    pa.add_argument("--eval-batch-size", type=int, default=8)
    pa.add_argument("--gradient-accumulation-steps", type=int, default=2)
    pa.add_argument("--max-target-length", type=int, default=96)
    pa.add_argument("--num-workers", type=int, default=2)
    pa.add_argument("--logging-steps", type=int, default=50)
    pa.add_argument("--save-steps", type=int, default=500)
    pa.add_argument("--eval-steps", type=int, default=500)
    pa.add_argument("--fp16", action="store_true")
    pa.add_argument("--bf16", action="store_true")
    pa.add_argument("--det-epochs", type=int, default=6)
    pa.add_argument("--det-batch-size", type=int, default=2)
    pa.add_argument("--det-lr", type=float, default=0.005)
    pa.add_argument("--det-lr-step", type=int, default=4)
    pa.add_argument("--det-num-workers", type=int, default=2)
    pa.add_argument("--det-device", default="cuda:0")
    pa.set_defaults(func=cmd_all)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
