#!/usr/bin/env python3
"""
Repair benchmark outputs by reparsing stored judge/pairwise rationale fields
with the current parser implementation, then rebuilding merged scores/reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmark import Benchmark, LLMJudge, PairwiseLLMJudge


def _restore_heuristic_scores(case_result: dict) -> dict:
    scores = dict(case_result.get("scores", {}))
    if "heuristic_overall" in scores:
        scores["overall"] = scores["heuristic_overall"]
    scores.pop("judge_overall", None)
    return scores


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: repair_benchmark_judging.py <src_json> <benchmark_profile> <dst_prefix>")

    src_json = Path(sys.argv[1])
    benchmark_profile = sys.argv[2]
    dst_prefix = Path(sys.argv[3])
    dst_json = dst_prefix.with_suffix(".json")
    dst_md = dst_prefix.with_suffix(".md")

    results = json.loads(src_json.read_text(encoding="utf-8"))
    config = results.get("config", {})
    benchmark = Benchmark(
        base_model=config.get("requested_base_model") or config.get("base_model") or "/data/wtw/Desktop/resume/models/Qwen2.5-3B",
        sft_model=config.get("requested_sft_model") or config.get("sft_model") or "outputs/failure_first_qwen25_3b_v1/sft_model_v5_float32_gpu1_followupfix",
        dpo_model=config.get("requested_dpo_model") or config.get("dpo_model") or "outputs/preference_first_dpo_qwen25_3b_v2_realistic_service/dpo_model_v1",
        judge_model=config.get("requested_judge_model") or config.get("judge_model"),
        use_llm_judge=bool(config.get("llm_judge_enabled")),
        use_pairwise_judge=bool(config.get("pairwise_judge_enabled")),
        judge_weight=float(config.get("judge_weight", 0.55)),
        answer_max_new_tokens=int(config.get("answer_max_new_tokens", 256)),
        judge_max_new_tokens=int(config.get("judge_max_new_tokens", 320)),
        inference_use_4bit=bool(config.get("inference_use_4bit", True)),
        judge_use_4bit=bool(config.get("judge_use_4bit", False)),
        benchmark_profile=benchmark_profile,
        stage_inference_use_4bit=config.get("stage_inference_use_4bit") or {},
    )
    benchmark.cases = [benchmark.case_index[case["case_id"]] for case in results.get("cases", []) if case["case_id"] in benchmark.case_index]

    if results.get("config", {}).get("llm_judge_enabled"):
        judge = LLMJudge(results["config"].get("judge_model") or "", max_new_tokens=benchmark.judge_max_new_tokens)
        for stage_result in results.get("stages", {}).values():
            for case_result in stage_result.get("cases", []):
                repaired_judge = judge._parse_response(case_result.get("judge", {}).get("rationale", ""))
                case_result["judge"] = repaired_judge
                heuristic_scores = _restore_heuristic_scores(case_result)
                case = benchmark.case_index[case_result["case_id"]]
                case_result["scores"] = benchmark._merge_scores(case, heuristic_scores, repaired_judge)
            stage_result["avg_score"] = round(
                sum(item["scores"]["overall"] for item in stage_result.get("cases", [])) / max(1, len(stage_result.get("cases", []))),
                4,
            )

    if results.get("config", {}).get("pairwise_judge_enabled"):
        pairwise = PairwiseLLMJudge(results["config"].get("judge_model") or "", max_new_tokens=benchmark.judge_max_new_tokens)
        repaired = []
        for item in results.get("pairwise_judging", {}).get("comparisons", []):
            parsed = pairwise._parse_response(item.get("rationale", ""))
            winner = parsed.get("winner", "tie")
            winner_stage = "tie"
            if winner == "a":
                winner_stage = item["stage_a"]
            elif winner == "b":
                winner_stage = item["stage_b"]
            repaired.append(
                {
                    **item,
                    "winner": winner,
                    "winner_stage": winner_stage,
                    "confidence": parsed.get("confidence", "low"),
                    "rationale": parsed.get("rationale", ""),
                    "failure_tags": parsed.get("failure_tags", []),
                    "dimension_scores": parsed.get("dimension_scores", {}),
                }
            )
        results["pairwise_judging"] = benchmark._build_pairwise_summary(repaired)

    results["routing"] = benchmark._build_routing_recommendations(results)
    results["failure_analysis"] = benchmark._build_failure_analysis(results)
    results["summary"] = benchmark._build_summary(results)

    dst_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    dst_md.write_text(benchmark._render_markdown_report(results), encoding="utf-8")
    print(dst_json)
    print(dst_md)


if __name__ == "__main__":
    main()
