#!/usr/bin/env python3
"""
Merge LoRA adapter weights back into the base model for deployment.
Produces a standalone model that doesn't require PEFT at inference time.
"""

import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.config import MergeConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def merge(cfg: MergeConfig = None):
    """Merge LoRA adapter into the base model and save."""
    if cfg is None:
        cfg = MergeConfig()

    logger.info("=" * 60)
    logger.info("Merging LoRA Adapter")
    logger.info(f"Base model: {cfg.base_model}")
    logger.info(f"Adapter: {cfg.adapter_path}")
    logger.info(f"Output: {cfg.output_dir}")
    logger.info("=" * 60)

    # Load base model in full precision for merging
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model,
        trust_remote_code=True,
    )

    # Load LoRA adapter
    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, cfg.adapter_path)

    # Merge and unload
    logger.info("Merging weights...")
    model = model.merge_and_unload()

    # Save merged model
    output_path = Path(cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Optional: push to hub
    if cfg.push_to_hub and cfg.hub_repo_id:
        logger.info(f"Pushing to hub: {cfg.hub_repo_id}")
        model.push_to_hub(cfg.hub_repo_id)
        tokenizer.push_to_hub(cfg.hub_repo_id)

    logger.info("Merge completed!")
    return str(output_path)


if __name__ == "__main__":
    merge()
