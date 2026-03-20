#!/usr/bin/env python3
"""
Build a closed-loop data plan from benchmark results.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scenario_taxonomy import build_taxonomy_index, export_taxonomy


TARGET_SCORE_BY_OWNER = {
    "sft": 0.82,
    "dpo": 0.78,
}

PRIORITY_WEIGHT = {
    "low": 0.8,
    "medium": 1.0,
    "high": 1.2,
}


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _priority_label(priority_score: float) -> str:
    if priority_score >= 0.4:
        return "P0"
    if priority_score >= 0.2:
        return "P1"
    return "P2"


def _suggest_sample_count(owner: str, base_target: int, current_score: float, risk_level: str) -> int:
    threshold = TARGET_SCORE_BY_OWNER.get(owner, 0.0)
    gap = max(0.0, threshold - current_score)
    if owner not in {"sft", "dpo"}:
        return 0
    if gap == 0:
        return max(64, int(base_target * 0.3))
    scaled = base_target * (0.55 + gap * 1.6) * PRIORITY_WEIGHT.get(risk_level, 1.0)
    return int(math.ceil(scaled / 10.0) * 10)


def _build_entry(case: Dict[str, Any], routing_map: Dict[str, Dict[str, Any]], taxonomy_map: Dict[str, Any]) -> Dict[str, Any]:
    scenario_type = case["scenario_type"]
    taxonomy = taxonomy_map[scenario_type]
    routing = routing_map.get(case["case_id"], {})
    scores = routing.get("scores", {})
    owner = taxonomy.failure_owner
    preferred_model = taxonomy.preferred_model
    recommended_stage = routing.get("recommended_stage", preferred_model)
    best_score = max(scores.values()) if scores else 0.0
    current_target_score = scores.get(preferred_model)
    if current_target_score is None and owner in scores:
        current_target_score = scores[owner]
    if current_target_score is None:
        current_target_score = scores.get("base", best_score)
    action_kind = owner
    if owner == "rag_or_prompt":
        action_kind = "rag_or_prompt"
    elif owner == "tooling":
        action_kind = "tooling"
    elif owner == "manual_review":
        action_kind = "manual_review"

    priority_score = max(0.0, TARGET_SCORE_BY_OWNER.get(owner, best_score) - current_target_score)
    priority_score *= PRIORITY_WEIGHT.get(case["risk_level"], 1.0)
    if action_kind == "manual_review" and case["risk_level"] == "high":
        priority_score = max(priority_score, 0.45)

    recipe = taxonomy.data_recipe
    suggested_count = _suggest_sample_count(
        owner=owner,
        base_target=recipe.sample_target if recipe else 0,
        current_score=current_target_score,
        risk_level=case["risk_level"],
    )

    entry = {
        "case_id": case["case_id"],
        "title": case["title"],
        "scenario_type": scenario_type,
        "family_id": taxonomy.family_id,
        "family_name": taxonomy.display_name,
        "risk_level": case["risk_level"],
        "online_route": taxonomy.online_route,
        "preferred_model": preferred_model,
        "recommended_stage_now": recommended_stage,
        "current_scores": scores,
        "current_target_score": round(current_target_score, 4),
        "priority_score": round(priority_score, 4),
        "priority_label": _priority_label(priority_score),
        "action_kind": action_kind,
        "diagnosis_checks": taxonomy.diagnosis_checks,
        "objective": recipe.objective if recipe else "",
        "sample_shape": recipe.sample_shape if recipe else "",
        "suggested_sample_count": suggested_count,
        "must_include": recipe.must_include if recipe else [],
        "hard_cases": recipe.hard_cases if recipe else [],
        "notes": recipe.notes if recipe else [],
    }
    return entry


def build_plan(benchmark_path: str | Path) -> Dict[str, Any]:
    benchmark = _load_json(benchmark_path)
    taxonomy_map = build_taxonomy_index()
    routing_map = {item["case_id"]: item for item in benchmark.get("routing", [])}
    cases = benchmark.get("cases", [])

    entries = [_build_entry(case, routing_map, taxonomy_map) for case in cases if case["scenario_type"] in taxonomy_map]
    entries.sort(key=lambda item: (item["priority_label"], -item["priority_score"], item["risk_level"]))

    summary = {
        "sft_cases": sum(1 for item in entries if item["action_kind"] == "sft"),
        "dpo_cases": sum(1 for item in entries if item["action_kind"] == "dpo"),
        "rag_cases": sum(1 for item in entries if item["action_kind"] == "rag_or_prompt"),
        "tooling_cases": sum(1 for item in entries if item["action_kind"] == "tooling"),
        "manual_review_cases": sum(1 for item in entries if item["action_kind"] == "manual_review"),
        "total_sft_samples": sum(item["suggested_sample_count"] for item in entries if item["action_kind"] == "sft"),
        "total_dpo_pairs": sum(item["suggested_sample_count"] for item in entries if item["action_kind"] == "dpo"),
    }

    return {
        "generated_at": datetime.now().isoformat(),
        "benchmark_path": str(benchmark_path),
        "taxonomy": export_taxonomy(),
        "summary": summary,
        "entries": entries,
    }


def render_markdown(plan: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Base 失败 -> 数据计划",
        "",
        f"- 生成时间：{plan['generated_at']}",
        f"- 基于 benchmark：{plan['benchmark_path']}",
        "",
        "## 汇总",
        "",
        "| 类别 | 数量 | 建议规模 |",
        "|---|---:|---:|",
        f"| SFT 场景 | {plan['summary']['sft_cases']} | {plan['summary']['total_sft_samples']} |",
        f"| DPO 场景 | {plan['summary']['dpo_cases']} | {plan['summary']['total_dpo_pairs']} |",
        f"| RAG/Prompt 优先 | {plan['summary']['rag_cases']} | 0 |",
        f"| Tooling 优先 | {plan['summary']['tooling_cases']} | 0 |",
        f"| 人工复核 | {plan['summary']['manual_review_cases']} | 0 |",
        "",
        "## Backlog",
        "",
        "| 优先级 | 场景 | 动作 | 当前建议路由 | 建议样本数 |",
        "|---|---|---|---|---:|",
    ]

    for entry in plan["entries"]:
        lines.append(
            f"| {entry['priority_label']} | {entry['title']} | {entry['action_kind']} | "
            f"{entry['online_route']} | {entry['suggested_sample_count']} |"
        )

    for entry in plan["entries"]:
        lines.extend(
            [
                "",
                f"## {entry['priority_label']} - {entry['title']}",
                "",
                f"- 场景家族：{entry['family_name']}",
                f"- 风险等级：{entry['risk_level']}",
                f"- 当前建议在线路由：{entry['online_route']}",
                f"- 当前最佳阶段：{entry['recommended_stage_now']}",
                f"- 目标模型：{entry['preferred_model']}",
                f"- 动作类型：{entry['action_kind']}",
                f"- 建议样本数：{entry['suggested_sample_count']}",
                f"- 目标：{entry['objective']}",
                "",
                "诊断检查：",
            ]
        )
        for item in entry["diagnosis_checks"]:
            lines.append(f"- {item}")

        if entry["must_include"]:
            lines.append("")
            lines.append("样本必须包含：")
            for item in entry["must_include"]:
                lines.append(f"- {item}")

        if entry["hard_cases"]:
            lines.append("")
            lines.append("Hard cases：")
            for item in entry["hard_cases"]:
                lines.append(f"- {item}")

        if entry["notes"]:
            lines.append("")
            lines.append("备注：")
            for item in entry["notes"]:
                lines.append(f"- {item}")

    return "\n".join(lines).strip() + "\n"


def save_plan(plan: Dict[str, Any], output_path: str | Path) -> Dict[str, Path]:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = output_file.with_suffix(".md")
    markdown_path.write_text(render_markdown(plan), encoding="utf-8")
    return {"json": output_file, "markdown": markdown_path}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build a data plan from benchmark results.")
    parser.add_argument(
        "--benchmark",
        default="outputs/model_showcase/benchmark_results.json",
        help="Path to benchmark JSON.",
    )
    parser.add_argument(
        "--output",
        default="outputs/model_showcase/data_plan.json",
        help="Path to write the generated plan.",
    )
    args = parser.parse_args()

    plan = build_plan(args.benchmark)
    paths = save_plan(plan, args.output)
    print(f"JSON: {paths['json']}")
    print(f"Markdown: {paths['markdown']}")


if __name__ == "__main__":
    main()
