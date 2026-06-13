"""
Fine-tune TrOCR (``VisionEncoderDecoderModel``) on line-level image + transcript CSV/JSONL.

**Arabic handwriting:** start from ``microsoft/trocr-base-handwritten`` (or printed base),
use line crops + UTF-8 transcripts, and pass ``--extend-tokenizer arabic`` (or ``corpus``)
so Arabic letters become tokenizer tokens — the default RoBERTa tokenizer is Latin-centric.

Dataset prep: ``standalone_trocr_arabic_prepare.py`` (TSV → CSV, splits, optional HF export).

Use the trained folder with the app: ``OCR_TROCR_MODEL=/path/to/checkpoint`` or
``trocr_model`` pointing at your saved ``output_dir`` (same layout as HF).
"""

import argparse
import os
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
from datasets import DatasetDict, Image, load_dataset
from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune TrOCR on line-level OCR datasets (CSV/JSONL manifests)."
    )
    parser.add_argument("--train-manifest", required=True, help="CSV/JSON/JSONL file path.")
    parser.add_argument("--val-manifest", required=True, help="CSV/JSON/JSONL file path.")
    parser.add_argument(
        "--test-manifest",
        default=None,
        help="Optional CSV/JSON/JSONL file path for final evaluation.",
    )
    parser.add_argument(
        "--image-column",
        default="image_path",
        help="Manifest column that contains image file path.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Manifest column that contains ground-truth transcript.",
    )
    parser.add_argument(
        "--model-name",
        default="microsoft/trocr-base-handwritten",
        help="HF checkpoint to start from.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/trocr-finetuned",
        help="Directory where checkpoints and final model are written.",
    )
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2,
        help="Keep small (1–2) if validation hits CUDA OOM during generate().",
    )
    parser.add_argument(
        "--generation-num-beams",
        type=int,
        default=1,
        help="Beam count during eval metrics (1 = greedy, lowest VRAM; 2 is heavier).",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Stop after this many optimizer steps (-1 = use num_train_epochs only).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 mixed precision (recommended on modern NVIDIA GPUs).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable BF16 mixed precision (Ampere+ GPUs / supported hardware).",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze vision encoder for faster/cheaper training.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Lower memory usage at some speed cost.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint path to resume from.",
    )
    parser.add_argument(
        "--extend-tokenizer",
        choices=("none", "corpus", "arabic"),
        default="none",
        help="Add tokens for non-Latin script: "
        "'corpus' = any char that does not encode as a single id; "
        "'arabic' = distinct Arabic-script characters seen in labels (recommended for AR handwriting).",
    )
    parser.add_argument(
        "--unicode-normalize",
        choices=("none", "nfc", "nfkc"),
        default="nfc",
        help="Normalize transcripts before encoding (default nfc; use none to disable).",
    )
    return parser.parse_args()


def _dataset_format_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return "csv"
    if ext in {".json", ".jsonl"}:
        return "json"
    raise ValueError(f"Unsupported manifest extension for {path}. Use CSV/JSON/JSONL.")


def _load_split(path: str):
    return load_dataset(_dataset_format_from_path(path), data_files=path, split="train")


def load_splits(args: argparse.Namespace) -> DatasetDict:
    ds = DatasetDict()
    ds["train"] = _load_split(args.train_manifest)
    ds["validation"] = _load_split(args.val_manifest)
    if args.test_manifest:
        ds["test"] = _load_split(args.test_manifest)
    return ds


def _arabic_script_char(ch: str) -> bool:
    if not ch:
        return False
    o = ord(ch)
    return (
        0x0600 <= o <= 0x06FF  # Arabic
        or 0x0750 <= o <= 0x077F  # Arabic Supplement
        or 0x08A0 <= o <= 0x08FF  # Arabic Extended-A
        or 0xFB50 <= o <= 0xFDFF  # Presentation Forms-A
        or 0xFE70 <= o <= 0xFEFF  # Presentation Forms-B
    )


def _normalize_text(s: str, mode: str) -> str:
    if mode == "none" or not s:
        return s
    form = (mode or "nfc").upper()
    try:
        return unicodedata.normalize(form, s)
    except ValueError:
        # Lone surrogates / invalid UTF-8-ish strings from upstream data
        return s


def extend_tokenizer_for_labels(
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    texts: Sequence[str],
    mode: str,
) -> int:
    """Return number of new tokens added (decoder embeddings resized when > 0)."""
    if mode == "none":
        return 0
    tokenizer = processor.tokenizer
    to_add: List[str] = []
    seen: Set[str] = set()
    for t in texts:
        for ch in str(t):
            if ch in seen:
                continue
            seen.add(ch)
            if mode == "arabic":
                if _arabic_script_char(ch):
                    to_add.append(ch)
            else:
                ids = tokenizer.encode(ch, add_special_tokens=False)
                if len(ids) != 1:
                    to_add.append(ch)
    if not to_add:
        return 0
    added = tokenizer.add_tokens(to_add)
    if added:
        # VisionEncoderDecoderModel (TrOCR) must resize the **decoder** embeddings only.
        if hasattr(model, "decoder") and hasattr(model.decoder, "resize_token_embeddings"):
            model.decoder.resize_token_embeddings(len(tokenizer))
        else:
            model.resize_token_embeddings(len(tokenizer))
    return added


def prepare_dataset(ds: DatasetDict, image_column: str, text_column: str) -> DatasetDict:
    # Converts file paths in image_column to lazily decoded image objects.
    ds = ds.cast_column(image_column, Image(decode=True))
    keep_cols = {image_column, text_column}
    for split in list(ds.keys()):
        drop_cols = [c for c in ds[split].column_names if c not in keep_cols]
        if drop_cols:
            ds[split] = ds[split].remove_columns(drop_cols)
    return ds


def build_model_and_processor(args: argparse.Namespace):
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = args.max_target_length
    beams = max(1, int(args.generation_num_beams))
    model.config.num_beams = beams
    # Transformers rejects early_stopping=True when num_beams==1 (breaks save_pretrained).
    model.config.early_stopping = beams > 1
    model.config.no_repeat_ngram_size = 0
    model.config.length_penalty = 1.0

    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.num_beams = beams
        gen_cfg.early_stopping = beams > 1

    return model, processor


def build_preprocess_fn(
    processor: TrOCRProcessor,
    image_column: str,
    text_column: str,
    max_target_length: int,
    unicode_normalize: str,
):
    def _preprocess(batch: Dict[str, List]):
        pixel_values = processor(images=batch[image_column], return_tensors="pt").pixel_values
        texts = [
            _normalize_text("" if t is None else str(t), unicode_normalize)
            for t in batch[text_column]
        ]
        labels = processor.tokenizer(
            texts,
            padding="max_length",
            max_length=max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        labels = labels.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

    return _preprocess


@dataclass
class OCRDataCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        def _pv(x):
            t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
            return t.squeeze(0) if t.ndim == 4 and t.shape[0] == 1 else t

        def _lb(x):
            t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.long)
            return t.squeeze(0) if t.ndim == 2 and t.shape[0] == 1 else t

        pixel_values = torch.stack([_pv(f["pixel_values"]) for f in features])
        labels = torch.stack([_lb(f["labels"]) for f in features])
        return {"pixel_values": pixel_values, "labels": labels}


def build_metrics_fn(processor: TrOCRProcessor):
    def _compute_metrics(pred):
        pred_ids = pred.predictions
        labels_ids = pred.label_ids.copy()
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id

        # Generated IDs can be out of range early in training; tokenizer.decode overflows on bad ids.
        tok = processor.tokenizer
        vmax = max(len(tok) - 1, 0)
        pred_ids = np.asarray(pred_ids)
        if pred_ids.dtype.kind == "f":
            pred_ids = np.nan_to_num(pred_ids, nan=0.0).astype(np.int64)
        pred_ids = np.clip(pred_ids, 0, vmax)

        pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_text = processor.batch_decode(labels_ids, skip_special_tokens=True)

        pred_text = [t.strip() for t in pred_text]
        label_text = [t.strip() for t in label_text]

        cer_value = jiwer_cer(label_text, pred_text)
        wer_value = jiwer_wer(label_text, pred_text)
        return {"cer": cer_value, "wer": wer_value}

    return _compute_metrics


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    raw_ds = load_splits(args)
    raw_ds = prepare_dataset(raw_ds, args.image_column, args.text_column)

    model, processor = build_model_and_processor(args)

    # Extend vocab from **train** labels only (no leakage from val/test).
    label_texts = [str(t) for t in raw_ds["train"][args.text_column]]
    n_new = extend_tokenizer_for_labels(
        processor, model, label_texts, args.extend_tokenizer
    )
    if n_new:
        print(f"Extended tokenizer with {n_new} tokens ({args.extend_tokenizer}).")
    elif args.extend_tokenizer != "none":
        print(
            f"Tokenizer extension ({args.extend_tokenizer}) added 0 tokens — "
            "check that transcripts contain Arabic/script characters."
        )

    preprocess_fn = build_preprocess_fn(
        processor,
        args.image_column,
        args.text_column,
        args.max_target_length,
        args.unicode_normalize,
    )

    encoded_ds = raw_ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=raw_ds["train"].column_names,
        num_proc=args.num_workers,
        desc="Encoding OCR dataset",
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=max(1, args.generation_num_beams),
        fp16=args.fp16,
        bf16=args.bf16,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        dataloader_num_workers=args.num_workers,
        report_to="none",
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_ds["train"],
        eval_dataset=encoded_ds["validation"],
        data_collator=OCRDataCollator(pad_token_id=processor.tokenizer.pad_token_id),
        tokenizer=processor.tokenizer,
        compute_metrics=build_metrics_fn(processor),
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    eval_metrics = trainer.evaluate(eval_dataset=encoded_ds["validation"])
    print("\nValidation metrics:")
    for k, v in eval_metrics.items():
        if isinstance(v, (int, float, np.floating)):
            print(f"{k}: {float(v):.6f}")
        else:
            print(f"{k}: {v}")

    if "test" in encoded_ds:
        test_metrics = trainer.evaluate(eval_dataset=encoded_ds["test"], metric_key_prefix="test")
        print("\nTest metrics:")
        for k, v in test_metrics.items():
            if isinstance(v, (int, float, np.floating)):
                print(f"{k}: {float(v):.6f}")
            else:
                print(f"{k}: {v}")


if __name__ == "__main__":
    main()
