#!/usr/bin/env python3
"""
Train demo-ready SFT and DPO adapters on synthetic data.

This script is tuned for a single A100 40GB and prioritizes a stable,
reproducible run over maximum quality.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config import DPOConfig, SFTConfig
from training.dpo_train import train as train_dpo
from training.sft_train import train as train_sft

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LOCAL_QWEN_MODEL = "/data/wtw/Desktop/resume/models/Qwen2.5-7B-Instruct"


def build_sft_config() -> SFTConfig:
    cfg = SFTConfig()
    cfg.model_name = LOCAL_QWEN_MODEL
    cfg.output_dir = "outputs/sft"
    cfg.num_train_epochs = 1
    cfg.per_device_train_batch_size = 4
    cfg.per_device_eval_batch_size = 2
    cfg.gradient_accumulation_steps = 4
    cfg.max_seq_length = 1024
    cfg.learning_rate = 2e-4
    cfg.save_steps = 50
    cfg.eval_steps = 50
    cfg.logging_steps = 10
    cfg.save_total_limit = 2
    cfg.warmup_ratio = 0.03
    return cfg


def build_dpo_config() -> DPOConfig:
    cfg = DPOConfig()
    cfg.model_name = "outputs/sft"
    cfg.output_dir = "outputs/dpo"
    cfg.num_train_epochs = 1
    cfg.per_device_train_batch_size = 1
    cfg.per_device_eval_batch_size = 1
    cfg.gradient_accumulation_steps = 4
    cfg.max_length = 768
    cfg.max_prompt_length = 384
    cfg.learning_rate = 5e-5
    cfg.save_steps = 25
    cfg.eval_steps = 25
    cfg.logging_steps = 10
    cfg.save_total_limit = 2
    cfg.warmup_ratio = 0.03
    return cfg


def main():
    logger.info("Starting demo training pipeline")
    sft_cfg = build_sft_config()
    train_sft(sft_cfg)

    dpo_cfg = build_dpo_config()
    train_dpo(dpo_cfg)
    logger.info("Demo training pipeline completed")


if __name__ == "__main__":
    main()
