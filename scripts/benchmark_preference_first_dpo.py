#!/usr/bin/env python3
"""
Run the preference-first benchmark suite for Phase 4c.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmark import Benchmark


DEFAULT_BASE_MODEL = "/data/wtw/Desktop/resume/models/Qwen2.5-3B"
DEFAULT_SFT_MODEL = "outputs/failure_first_qwen25_3b_v1/sft_model_v5_float32_gpu1_followupfix"
DEFAULT_DPO_MODEL = "outputs/preference_first_dpo_qwen25_3b_v1/dpo_model_v1"
DEFAULT_OUTPUT = "outputs/preference_first_dpo_qwen25_3b_v1/benchmark_preference_first.json"


def _parse_stages(raw: str) -> List[str]:
    stages = [item.strip() for item in raw.split(",") if item.strip()]
    return stages or ["base"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the preference-first DPO suite.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model path")
    parser.add_argument("--sft-model", default=DEFAULT_SFT_MODEL, help="SFT model path")
    parser.add_argument("--dpo-model", default=DEFAULT_DPO_MODEL, help="DPO model path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Benchmark JSON output path")
    parser.add_argument(
        "--stages",
        default="base",
        help="Comma-separated stage list, for example: base or base,dpo",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_BASE_MODEL,
        help="Judge model path; ignored when LLM judge is disabled",
    )
    parser.add_argument(
        "--use-llm-judge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to enable the shared LLM judge pass",
    )
    parser.add_argument(
        "--inference-use-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default 4-bit setting for all stages when no stage-specific override is given",
    )
    parser.add_argument(
        "--base-inference-use-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional 4-bit override for the base stage",
    )
    parser.add_argument(
        "--sft-inference-use-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional 4-bit override for the sft stage",
    )
    parser.add_argument(
        "--dpo-inference-use-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional 4-bit override for the dpo stage",
    )
    parser.add_argument(
        "--judge-use-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether the optional LLM judge should use 4-bit loading",
    )
    parser.add_argument("--judge-weight", type=float, default=0.55, help="LLM judge weight")
    parser.add_argument("--answer-max-new-tokens", type=int, default=256, help="Generation max tokens")
    parser.add_argument("--judge-max-new-tokens", type=int, default=320, help="Judge max tokens")
    args = parser.parse_args()

    stage_inference_use_4bit = {}
    if args.base_inference_use_4bit is not None:
        stage_inference_use_4bit["base"] = args.base_inference_use_4bit
    if args.sft_inference_use_4bit is not None:
        stage_inference_use_4bit["sft"] = args.sft_inference_use_4bit
    if args.dpo_inference_use_4bit is not None:
        stage_inference_use_4bit["dpo"] = args.dpo_inference_use_4bit

    benchmark = Benchmark(
        base_model=args.base_model,
        sft_model=args.sft_model,
        dpo_model=args.dpo_model,
        judge_model=args.judge_model,
        use_llm_judge=args.use_llm_judge,
        judge_weight=args.judge_weight,
        answer_max_new_tokens=args.answer_max_new_tokens,
        judge_max_new_tokens=args.judge_max_new_tokens,
        inference_use_4bit=args.inference_use_4bit,
        judge_use_4bit=args.judge_use_4bit,
        benchmark_profile="preference_first",
        stage_inference_use_4bit=stage_inference_use_4bit,
    )
    benchmark.run(stages=_parse_stages(args.stages), output_path=args.output)


if __name__ == "__main__":
    main()
