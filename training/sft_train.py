#!/usr/bin/env python3
"""
QLoRA SFT training script for Qwen2.5-7B on e-commerce data.
Uses trl.SFTTrainer with 4-bit quantization and LoRA.
"""

import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig as TRLSFTConfig, SFTTrainer

from training.config import SFTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LOCAL_MODEL_CANDIDATES = {
    "Qwen/Qwen2.5-7B-Instruct": [
        Path("/data/wtw/Desktop/resume/models/Qwen2.5-7B-Instruct"),
        Path("/data/wtw/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"),
    ]
}

QWEN_CHAT_TEMPLATE_WITH_ASSISTANT_MASKS = """{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are a helpful assistant.' }}
    {%- endif %}
    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
    {%- elif message.role == "assistant" and not message.tool_calls %}
        {{- '<|im_start|>' + message.role + '\\n' }}
        {% generation %}
        {{- message.content + '<|im_end|>\\n' }}
        {% endgeneration %}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\\n<tool_call>\\n{\\"name\\": "' }}
            {{- tool_call.name }}
            {{- '", \\"arguments\\": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- message.content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}
"""


def resolve_model_path(model_name: str) -> str:
    """Resolve a model identifier to a local path when available."""
    direct_path = Path(model_name)
    if direct_path.exists():
        return str(direct_path)

    for candidate in LOCAL_MODEL_CANDIDATES.get(model_name, []):
        if candidate.exists():
            return str(candidate)

    return model_name


def get_device_map():
    """Use a single visible GPU for trainer compatibility."""
    if torch.cuda.is_available():
        return {"": 0}
    return "auto"


def resolve_precision(request_bf16: bool, request_fp16: bool = False) -> dict:
    """Resolve training precision based on the actual hardware capability."""
    if torch.cuda.is_available():
        bf16_supported = torch.cuda.is_bf16_supported()
        if request_bf16 and bf16_supported:
            return {"bf16": True, "fp16": False, "torch_dtype": torch.bfloat16}

        if request_bf16 and not bf16_supported:
            logger.warning("bf16 已请求，但当前 GPU 不支持，自动回退到 fp16。")

        if request_fp16 or request_bf16:
            return {"bf16": False, "fp16": True, "torch_dtype": torch.float16}

    if request_bf16 and not torch.cuda.is_available():
        logger.warning("bf16 已请求，但当前没有可用 GPU，自动回退到 float32。")

    return {"bf16": False, "fp16": False, "torch_dtype": torch.float32}


def load_sft_data(path: str) -> Dataset:
    """Load SFT training data from JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} samples from {path}")
    return Dataset.from_list(records)


def format_messages(example: dict) -> dict:
    """Format messages into a single text string for SFT."""
    messages = example["messages"]
    text_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    return {"text": "\n".join(text_parts)}


def maybe_enable_assistant_mask_chat_template(tokenizer) -> None:
    """Ensure the tokenizer chat template can emit assistant masks for assistant-only loss."""
    chat_template = getattr(tokenizer, "chat_template", None) or ""
    if "{% generation %}" in chat_template:
        return
    tokenizer.chat_template = QWEN_CHAT_TEMPLATE_WITH_ASSISTANT_MASKS
    logger.info("Replaced tokenizer chat template with generation-aware variant for assistant-only loss.")


def create_bnb_config(cfg: SFTConfig, compute_dtype: torch.dtype) -> BitsAndBytesConfig:
    """Create BitsAndBytes quantization config."""
    return BitsAndBytesConfig(
        load_in_4bit=cfg.use_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.use_double_quant,
    )


def create_lora_config(cfg: SFTConfig) -> LoraConfig:
    """Create LoRA configuration."""
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def train(cfg: SFTConfig = None):
    """Run SFT training pipeline."""
    if cfg is None:
        cfg = SFTConfig()

    logger.info("=" * 60)
    logger.info("Starting SFT Training")
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"Output: {cfg.output_dir}")
    logger.info(f"LoRA r={cfg.lora_r}, alpha={cfg.lora_alpha}")
    logger.info("=" * 60)

    model_path = resolve_model_path(cfg.model_name)
    logger.info(f"Resolved model path: {model_path}")
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
        padding_side="right",
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(cfg, "assistant_only_loss", False):
        maybe_enable_assistant_mask_chat_template(tokenizer)

    # Load model with quantization
    bnb_config = create_bnb_config(cfg, precision["torch_dtype"]) if cfg.use_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=get_device_map(),
        trust_remote_code=True,
        torch_dtype=precision["torch_dtype"],
        local_files_only=True,
    )
    model.config.use_cache = False

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = create_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    train_dataset = load_sft_data(cfg.train_data_path)
    val_dataset = load_sft_data(cfg.val_data_path)

    train_has_messages = "messages" in train_dataset.column_names
    val_has_messages = "messages" in val_dataset.column_names
    use_assistant_only_loss = bool(getattr(cfg, "assistant_only_loss", False) and train_has_messages and val_has_messages)
    dataset_text_field = "text"

    if train_has_messages and val_has_messages:
        logger.info("Using conversational dataset path with assistant_only_loss=%s", use_assistant_only_loss)
    else:
        logger.warning("Training data does not expose conversational messages. Falling back to plain text SFT.")
        if getattr(cfg, "assistant_only_loss", False):
            logger.warning("assistant_only_loss requested but disabled because dataset is not conversational.")
        train_dataset = train_dataset.map(format_messages)
        val_dataset = val_dataset.map(format_messages)
        use_assistant_only_loss = False

    # Training arguments
    training_args = TRLSFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=getattr(cfg, "max_steps", -1),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        eval_strategy="steps",
        save_total_limit=cfg.save_total_limit,
        fp16=precision["fp16"],
        bf16=precision["bf16"],
        gradient_checkpointing=cfg.gradient_checkpointing,
        optim=cfg.optim,
        seed=cfg.seed,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataset_text_field=dataset_text_field,
        max_length=cfg.max_seq_length,
        packing=False,
        completion_only_loss=False,
        assistant_only_loss=use_assistant_only_loss,
        do_train=True,
        do_eval=True,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting training...")
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

    logger.info("SFT training completed!")
    return cfg.output_dir


if __name__ == "__main__":
    train()
