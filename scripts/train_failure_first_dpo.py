#!/usr/bin/env python3
"""
Train the failure-first DPO adapter for Qwen2.5-3B.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config import DPOConfig
from training.dpo_train import train


DEFAULT_MODEL_PATH = "outputs/failure_first_qwen25_3b_v1/sft_model_v5_float32_gpu1_followupfix"
DEFAULT_TRAIN_PATH = "outputs/base_failure_dpo_qwen25_3b_v1/dpo_train.jsonl"
DEFAULT_VAL_PATH = "outputs/base_failure_dpo_qwen25_3b_v1/dpo_val.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/base_failure_dpo_qwen25_3b_v1/dpo_model_v1"


def build_cfg(
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    model_name: str,
    beta: float = 0.1,
    learning_rate: float = 3e-5,
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_length: int = 640,
    max_prompt_length: int = 320,
    save_steps: int = 20,
    eval_steps: int = 20,
    logging_steps: int = 5,
    use_4bit: bool = True,
    bf16: bool = False,
    fp16: bool = False,
    optim: str = "paged_adamw_32bit",
    max_steps: int = -1,
) -> DPOConfig:
    cfg = DPOConfig()
    cfg.model_name = model_name
    cfg.output_dir = output_dir
    cfg.train_data_path = train_data_path
    cfg.val_data_path = val_data_path
    cfg.beta = beta
    cfg.num_train_epochs = num_train_epochs
    cfg.per_device_train_batch_size = per_device_train_batch_size
    cfg.per_device_eval_batch_size = per_device_eval_batch_size
    cfg.gradient_accumulation_steps = gradient_accumulation_steps
    cfg.max_length = max_length
    cfg.max_prompt_length = max_prompt_length
    cfg.learning_rate = learning_rate
    cfg.save_steps = save_steps
    cfg.eval_steps = eval_steps
    cfg.logging_steps = logging_steps
    cfg.save_total_limit = 2
    cfg.warmup_ratio = 0.05
    cfg.use_4bit = use_4bit
    cfg.bf16 = bf16
    cfg.fp16 = fp16
    cfg.optim = optim
    cfg.max_steps = max_steps
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train failure-first DPO for Qwen2.5-3B.")
    parser.add_argument("--train-data", default=DEFAULT_TRAIN_PATH, help="DPO train JSONL path")
    parser.add_argument("--val-data", default=DEFAULT_VAL_PATH, help="DPO val JSONL path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output model directory")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_PATH, help="Starting model path")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--num-train-epochs", type=int, default=2, help="Epoch count")
    parser.add_argument("--per-device-train-batch-size", type=int, default=2, help="Train batch size")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1, help="Eval batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Grad accumulation")
    parser.add_argument("--max-length", type=int, default=640, help="Max response length")
    parser.add_argument("--max-prompt-length", type=int, default=320, help="Max prompt length")
    parser.add_argument("--save-steps", type=int, default=20, help="Checkpoint save steps")
    parser.add_argument("--eval-steps", type=int, default=20, help="Evaluation steps")
    parser.add_argument("--logging-steps", type=int, default=5, help="Logging steps")
    parser.add_argument(
        "--use-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable 4-bit loading for the base model",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False, help="Enable bf16")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False, help="Enable fp16")
    parser.add_argument("--optim", default="paged_adamw_32bit", help="Optimizer name")
    parser.add_argument("--max-steps", type=int, default=-1, help="Override max training steps")
    args = parser.parse_args()

    cfg = build_cfg(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        use_4bit=args.use_4bit,
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
