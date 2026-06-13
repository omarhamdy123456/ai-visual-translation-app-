# TrOCR Large-Dataset Training Guide

This project now includes `trocr_train.py` for fine-tuning TrOCR on handwritten or printed line images.

## 1) Dataset format

Prepare line-level manifests (recommended for best accuracy):

- `train.csv`
- `val.csv`
- `test.csv` (optional)

Each file needs at least these columns:

- `image_path`: absolute or relative path to line image
- `text`: ground-truth transcript for that line

Example:

```csv
image_path,text
data/lines/000001.png,cost plus incentive fee
data/lines/000002.png,the cost is determined at the beginning
```

## 2) Install training dependencies

```bash
pip install -r requirements.txt
```

## 3) Start training (GPU)

```bash
python trocr_train.py ^
  --train-manifest data/train.csv ^
  --val-manifest data/val.csv ^
  --test-manifest data/test.csv ^
  --model-name microsoft/trocr-base-handwritten ^
  --output-dir models/trocr-handwriting-v1 ^
  --epochs 10 ^
  --learning-rate 3e-5 ^
  --train-batch-size 8 ^
  --eval-batch-size 8 ^
  --gradient-accumulation-steps 2 ^
  --max-target-length 96 ^
  --num-workers 4 ^
  --fp16
```

For CPU-only training, remove `--fp16` and reduce batch sizes (for example `--train-batch-size 2`).

## 4) Resume interrupted runs

```bash
python trocr_train.py ^
  --train-manifest data/train.csv ^
  --val-manifest data/val.csv ^
  --output-dir models/trocr-handwriting-v1 ^
  --resume-from-checkpoint models/trocr-handwriting-v1/checkpoint-2000 ^
  --fp16
```

## 5) Speed and accuracy tips

- Keep samples line-level (not full-page) for cleaner sequence targets.
- Clean transcript labels aggressively; bad labels cap final accuracy.
- Use `--freeze-encoder` for faster training if data is very similar to the pretrained domain.
- Start with `microsoft/trocr-small-handwritten` if speed matters more than peak accuracy.
- For faster inference later, use beam size 1 and smaller `max_length` in production.

