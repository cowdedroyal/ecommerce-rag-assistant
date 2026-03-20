#!/usr/bin/env python3
"""
Train the failure-first SFT adapter for Qwen2.5-3B.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_focus_models import build_focus_sft_config
from training.sft_train import train


DEFAULT_MODEL_PATH = "/data/wtw/Desktop/resume/models/Qwen2.5-3B"
DEFAULT_TRAIN_PATH = "outputs/failure_first_qwen25_3b_v1/sft_train.jsonl"
DEFAULT_VAL_PATH = "outputs/failure_first_qwen25_3b_v1/sft_val.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/failure_first_qwen25_3b_v1/sft_model_v1"


def build_cfg(
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    model_name: str,
    learning_rate: float = 8e-5,
    num_train_epochs: int = 3,
    save_steps: int = 20,
    eval_steps: int = 20,
    logging_steps: int = 5,
    assistant_only_loss: bool = True,
    bf16: bool = True,
    fp16: bool = False,
    optim: str = "paged_adamw_32bit",
    max_steps: int = -1,
) :
    cfg = build_focus_sft_config(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        output_dir=output_dir,
        model_name=model_name,
        profile_id="qwen25_3b_struct",
    )
    cfg.num_train_epochs = num_train_epochs
    cfg.per_device_train_batch_size = 4
    cfg.per_device_eval_batch_size = 2
    cfg.gradient_accumulation_steps = 4
    cfg.max_seq_length = 896
    cfg.learning_rate = learning_rate
    cfg.save_steps = save_steps
    cfg.eval_steps = eval_steps
    cfg.logging_steps = logging_steps
    cfg.save_total_limit = 2
    cfg.warmup_ratio = 0.05
    cfg.assistant_only_loss = assistant_only_loss
    cfg.bf16 = bf16
    cfg.fp16 = fp16
    cfg.optim = optim
    cfg.max_steps = max_steps
    cfg.output_dir = output_dir
    cfg.model_name = model_name
    cfg.train_data_path = train_data_path
    cfg.val_data_path = val_data_path
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train failure-first SFT for Qwen2.5-3B.")
    parser.add_argument("--train-data", default=DEFAULT_TRAIN_PATH, help="SFT train JSONL path")
    parser.add_argument("--val-data", default=DEFAULT_VAL_PATH, help="SFT val JSONL path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output model directory")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_PATH, help="Base model path")
    parser.add_argument("--learning-rate", type=float, default=8e-5, help="Learning rate")
    parser.add_argument("--num-train-epochs", type=int, default=3, help="Epoch count")
    parser.add_argument("--save-steps", type=int, default=20, help="Checkpoint save steps")
    parser.add_argument("--eval-steps", type=int, default=20, help="Evaluation steps")
    parser.add_argument("--logging-steps", type=int, default=5, help="Logging steps")
    parser.add_argument(
        "--assistant-only-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to enable assistant-only loss masking.",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="Enable bf16")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False, help="Enable fp16")
    parser.add_argument("--optim", default="paged_adamw_32bit", help="Optimizer name")
    parser.add_argument("--max-steps", type=int, default=-1, help="Override max training steps")
    args = parser.parse_args()

    cfg = build_cfg(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        assistant_only_loss=args.assistant_only_loss,
        bf16=args.bf16,
        fp16=args.fp16,
        optim=args.optim,
        max_steps=args.max_steps,
    )

    output_root = Path(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "train_config.json").write_text(
        json.dumps(cfg.__dict__, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train(cfg)


if __name__ == "__main__":
    main()
