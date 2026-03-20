"""
Training configuration for SFT and DPO.
Centralized hyperparameter management for reproducibility.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SFTConfig:
    """SFT training hyperparameters (optimized for A100 40GB)."""

    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "outputs/sft"

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Quantization (QLoRA)
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True

    # Training
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"

    # Logging & saving
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3

    # Precision
    fp16: bool = False
    bf16: bool = True

    # Data
    train_data_path: str = "data/training/sft_train.jsonl"
    val_data_path: str = "data/training/sft_val.jsonl"

    # Misc
    seed: int = 42
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"
    assistant_only_loss: bool = True


@dataclass
class DPOConfig:
    """DPO training hyperparameters."""

    # Model (start from SFT checkpoint)
    model_name: str = "outputs/sft"
    output_dir: str = "outputs/dpo"

    # DPO
    beta: float = 0.1
    loss_type: str = "sigmoid"

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # Training
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_length: int = 2048
    max_prompt_length: int = 1024
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"

    # Logging & saving
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50
    save_total_limit: int = 3

    # Precision
    fp16: bool = False
    bf16: bool = True

    # Data
    train_data_path: str = "data/training/dpo_train.jsonl"
    val_data_path: str = "data/training/dpo_val.jsonl"

    # Misc
    seed: int = 42
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"


@dataclass
class MergeConfig:
    """LoRA merge configuration."""

    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path: str = "outputs/dpo"
    output_dir: str = "outputs/merged"
    push_to_hub: bool = False
    hub_repo_id: Optional[str] = None
