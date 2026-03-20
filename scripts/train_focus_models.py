#!/usr/bin/env python3
"""
Recommended focused training configs for repair-only SFT/DPO experiments.
"""

from __future__ import annotations

from training.config import DPOConfig, SFTConfig
from training.focus_slices import DEFAULT_FOCUS_PROFILE_ID, get_focus_profile


def _resolve_profile(profile_id: str):
    return get_focus_profile(profile_id or DEFAULT_FOCUS_PROFILE_ID)


def build_focus_sft_config(
    train_data_path: str | None = None,
    val_data_path: str | None = None,
    output_dir: str | None = None,
    model_name: str | None = None,
    profile_id: str = DEFAULT_FOCUS_PROFILE_ID,
) -> SFTConfig:
    profile = _resolve_profile(profile_id)
    cfg = SFTConfig()
    cfg.model_name = model_name or profile["recommended_base_model"]
    cfg.output_dir = output_dir or profile["recommended_sft_output_dir"]
    cfg.train_data_path = train_data_path or f"{profile['recommended_output_dir']}/sft_train.jsonl"
    cfg.val_data_path = val_data_path or f"{profile['recommended_output_dir']}/sft_val.jsonl"
    cfg.num_train_epochs = 3
    cfg.per_device_train_batch_size = 2
    cfg.per_device_eval_batch_size = 1
    cfg.gradient_accumulation_steps = 8
    cfg.max_seq_length = 768
    cfg.learning_rate = 1e-4
    cfg.save_steps = 20
    cfg.eval_steps = 20
    cfg.logging_steps = 5
    cfg.save_total_limit = 2
    cfg.warmup_ratio = 0.05
    if profile_id == "qwen25_3b_struct":
        cfg.per_device_train_batch_size = 4
        cfg.gradient_accumulation_steps = 4
        cfg.max_seq_length = 896
        cfg.learning_rate = 8e-5
    return cfg


def build_focus_dpo_config(
    train_data_path: str | None = None,
    val_data_path: str | None = None,
    output_dir: str | None = None,
    model_name: str | None = None,
    profile_id: str = DEFAULT_FOCUS_PROFILE_ID,
) -> DPOConfig:
    profile = _resolve_profile(profile_id)
    cfg = DPOConfig()
    cfg.model_name = model_name or profile["recommended_sft_output_dir"]
    cfg.output_dir = output_dir or profile["recommended_dpo_output_dir"]
    cfg.train_data_path = train_data_path or f"{profile['recommended_output_dir']}/dpo_train.jsonl"
    cfg.val_data_path = val_data_path or f"{profile['recommended_output_dir']}/dpo_val.jsonl"
    cfg.num_train_epochs = 2
    cfg.per_device_train_batch_size = 1
    cfg.per_device_eval_batch_size = 1
    cfg.gradient_accumulation_steps = 4
    cfg.max_length = 512
    cfg.max_prompt_length = 256
    cfg.learning_rate = 5e-5
    cfg.save_steps = 20
    cfg.eval_steps = 20
    cfg.logging_steps = 5
    cfg.save_total_limit = 2
    cfg.warmup_ratio = 0.05
    if profile_id == "qwen25_3b_struct":
        cfg.per_device_train_batch_size = 2
        cfg.gradient_accumulation_steps = 4
        cfg.max_length = 640
        cfg.max_prompt_length = 320
        cfg.learning_rate = 3e-5
    return cfg
