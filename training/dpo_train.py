#!/usr/bin/env python3
"""
DPO training script. Uses the SFT model as the starting point
and applies Direct Preference Optimization with LoRA.
"""

import json
import inspect
import logging

import torch
from datasets import Dataset
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig as TRLDPOConfig, DPOTrainer

from training.config import DPOConfig
from training.sft_train import get_device_map, resolve_model_path, resolve_precision

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dpo_data(path: str) -> Dataset:
    """Load DPO preference pair data from JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} preference pairs from {path}")
    return Dataset.from_list(records)


def train(cfg: DPOConfig = None):
    """Run DPO training pipeline."""
    if cfg is None:
        cfg = DPOConfig()

    logger.info("=" * 60)
    logger.info("Starting DPO Training")
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"Output: {cfg.output_dir}")
    logger.info(f"Beta: {cfg.beta}")
    logger.info("=" * 60)

    model_path = resolve_model_path(cfg.model_name)
    logger.info(f"Resolved training start point: {model_path}")
    precision = resolve_precision(request_bf16=cfg.bf16, request_fp16=cfg.fp16)
    logger.info(
        "Precision resolved: bf16=%s, fp16=%s, torch_dtype=%s",
        precision["bf16"],
        precision["fp16"],
        precision["torch_dtype"],
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.use_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=precision["torch_dtype"],
    )

    # Load model and reference model. If model_path is a PEFT adapter from SFT,
    # continue training that adapter and freeze a mirrored reference copy.
    if (torch.cuda.is_available() and model_path) or model_path:
        try:
            peft_cfg = PeftConfig.from_pretrained(model_path)
            base_model_path = resolve_model_path(peft_cfg.base_model_name_or_path)
            logger.info(f"Detected PEFT adapter. Base model: {base_model_path}")

            model_base = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map=get_device_map(),
                trust_remote_code=True,
                torch_dtype=precision["torch_dtype"],
                local_files_only=True,
            )
            ref_base = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map=get_device_map(),
                trust_remote_code=True,
                torch_dtype=precision["torch_dtype"],
                local_files_only=True,
            )

            model = PeftModel.from_pretrained(
                model_base,
                model_path,
                is_trainable=True,
            )
            ref_model = PeftModel.from_pretrained(
                ref_base,
                model_path,
                is_trainable=False,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map=get_device_map(),
                trust_remote_code=True,
                torch_dtype=precision["torch_dtype"],
                local_files_only=True,
            )
            ref_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map=get_device_map(),
                trust_remote_code=True,
                torch_dtype=precision["torch_dtype"],
                local_files_only=True,
            )

    model.config.use_cache = False

    # Load data
    train_dataset = load_dpo_data(cfg.train_data_path)
    val_dataset = load_dpo_data(cfg.val_data_path)

    # Training arguments
    training_kwargs = {
        "output_dir": cfg.output_dir,
        "num_train_epochs": cfg.num_train_epochs,
        "max_steps": getattr(cfg, "max_steps", -1),
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "eval_steps": cfg.eval_steps,
        "eval_strategy": "steps",
        "save_total_limit": cfg.save_total_limit,
        "bf16": precision["bf16"],
        "fp16": precision["fp16"],
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "optim": cfg.optim,
        "seed": cfg.seed,
        "report_to": "none",
        "remove_unused_columns": False,
        "load_best_model_at_end": True,
        "beta": cfg.beta,
        "max_length": cfg.max_length,
        "loss_type": [cfg.loss_type] if isinstance(cfg.loss_type, str) else cfg.loss_type,
        "do_train": True,
        "do_eval": True,
    }
    if "max_prompt_length" in inspect.signature(TRLDPOConfig).parameters:
        training_kwargs["max_prompt_length"] = cfg.max_prompt_length

    training_args = TRLDPOConfig(**training_kwargs)

    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting DPO training...")
    train_result = trainer.train()

    # Save
    logger.info(f"Saving model to {cfg.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(cfg.output_dir)

    # Log metrics
    metrics = train_result.metrics
    logger.info(f"Training metrics: {metrics}")
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Evaluate
    eval_metrics = trainer.evaluate()
    logger.info(f"Eval metrics: {eval_metrics}")
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info("DPO training completed!")
    return cfg.output_dir


if __name__ == "__main__":
    train()
