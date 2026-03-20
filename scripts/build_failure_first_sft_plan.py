#!/usr/bin/env python3
"""
Build the Phase 2 failure-first SFT data plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scenario_taxonomy import build_taxonomy_index


PHASE_NAME = "phase_2_sft_plan_only"
ALLOWED_SCENARIOS = ("simple_recommendation", "followup_memory")
EXCLUDED_SCENARIOS = (
    "structured_recommendation",
    "structured_comparison",
    "tone_alignment",
    "honest_boundary",
    "after_sales_boundary",
    "general_advice",
)

FAILURE_FIRST_SFT_RECIPES: Dict[str, Dict[str, Any]] = {
    "simple_recommendation": {
        "bundle_id": "failure_first_simple_recommendation",
        "display_name": "Failure-First SFT - 简单推荐",
        "source_slice_id": "budget_grounded_recommendation",
        "source_task": "product_recommendation",
        "source_scenario": "budget_grounded_recommendation",
        "selection_mode": "category_exact_match",
        "objective": "用低风险 grounded recommendation 样本验证 3B 是否能在 failure-first 简单推荐场景里稳定命中预算、品类和推荐理由。",
        "owner_override": "sft",
        "override_reason": (
            "taxonomy 默认把 simple_recommendation 归到 rag_or_prompt，"
            "但这轮 failure-first 子集里已有 3 个 Qwen2.5-3B base < 0.5 的简单推荐 case，"
            "先用低风险 grounded recommendation 样本验证 SFT 是否能稳定救回。"
        ),
        "must_include": [
            "预算约束",
            "品类命中",
            "至少 2 个候选商品",
            "围绕用户场景给出简洁推荐理由",
        ],
        "hard_constraints": [
            "只允许蓝牙耳机 / 机械键盘 / 运动鞋三个 failure 品类",
            "不混入 structured_recommendation_repair",
            "不混入 structured_comparison",
            "不混入 tone_alignment / honest_boundary",
        ],
        "notes": [
            "这一轮只验证 failure-first SFT 能否改善 base 明显失分的简单推荐场景。",
            "后续导出时允许保留轻结构化回答，但不要再把它包装成旧的 structured repair 路线。",
        ],
    },
    "followup_memory": {
        "bundle_id": "failure_first_followup_memory",
        "display_name": "Failure-First SFT - 多轮追问记忆",
        "source_slice_id": "followup_direct_answer_repair",
        "source_task": "failure_repair_sft",
        "source_scenario": "followup_direct_answer_repair",
        "selection_mode": "category_exact_match",
        "objective": "增强多轮追问中的上下文延续、编号复用和直接结论能力，只验证 failure-first 的运动鞋 case。",
        "owner_override": None,
        "override_reason": "",
        "must_include": [
            "先给候选，再处理追问",
            "沿用前文编号",
            "直接给结论",
            "围绕新增约束补充原因",
        ],
        "hard_constraints": [
            "当前只保留运动鞋品类，和 failure case 一一对应",
            "不混入 clarify_then_recommend",
            "不混入单轮 structured recommendation repair",
        ],
        "notes": [
            "这一轮只覆盖 sft_followup_memory_sneaker，不扩展到其他 followup 品类。",
        ],
    },
}


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _priority_label(base_score: float) -> str:
    if base_score < 0.25:
        return "P0"
    if base_score < 0.4:
        return "P1"
    return "P2"


def _case_categories(case: Dict[str, Any]) -> List[str]:
    categories = []
    for doc in case.get("retrieved_docs", []):
        category = doc.get("category")
        if category and category not in categories:
            categories.append(category)
    return categories


def _match_bundle_record(
    record: Dict[str, Any],
    source_task: str,
    source_scenario: str,
    allowed_categories: Iterable[str],
) -> bool:
    metadata = record.get("metadata", {})
    if metadata.get("task") != source_task:
        return False
    if metadata.get("scenario") != source_scenario:
        return False
    categories = set(metadata.get("categories", []))
    return bool(categories & set(allowed_categories))


def _select_bundle_records(
    records: List[Dict[str, Any]],
    source_task: str,
    source_scenario: str,
    allowed_categories: Iterable[str],
) -> List[Dict[str, Any]]:
    allowed_category_set = set(allowed_categories)
    return [
        record
        for record in records
        if _match_bundle_record(record, source_task, source_scenario, allowed_category_set)
    ]


def _split_category_counts(records: List[Dict[str, Any]], allowed_categories: Iterable[str]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    allowed_category_set = set(allowed_categories)
    for record in records:
        categories = set(record.get("metadata", {}).get("categories", []))
        for category in categories & allowed_category_set:
            counter[category] += 1
    return dict(sorted(counter.items()))


def _preview_prompts(records: List[Dict[str, Any]], limit: int = 2) -> List[str]:
    prompts: List[str] = []
    for record in records:
        user_messages = [item.get("content", "") for item in record.get("messages", []) if item.get("role") == "user"]
        if not user_messages:
            continue
        prompts.append(user_messages[-1])
        if len(prompts) >= limit:
            break
    return prompts


def build_plan(
    failure_plan_path: str | Path,
    sft_train_path: str | Path,
    sft_val_path: str | Path,
) -> Dict[str, Any]:
    failure_plan = _load_json(failure_plan_path)
    benchmark_path = failure_plan["generated_from"]
    benchmark = _load_json(benchmark_path)
    benchmark_cases = {item["case_id"]: item for item in benchmark.get("cases", [])}
    taxonomy_map = build_taxonomy_index()

    sft_train = _load_jsonl(sft_train_path)
    sft_val = _load_jsonl(sft_val_path)

    included_records: List[Dict[str, Any]] = []
    excluded_records: List[Dict[str, Any]] = []
    records_by_scenario: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for record in failure_plan.get("records", []):
        scenario_type = record["scenario_type"]
        if record.get("proposed_action") == "sft" and scenario_type in ALLOWED_SCENARIOS:
            records_by_scenario[scenario_type].append(record)
            included_records.append(record)
        else:
            excluded_records.append(record)

    bundles: List[Dict[str, Any]] = []
    for scenario_type in ALLOWED_SCENARIOS:
        if scenario_type not in records_by_scenario:
            continue

        recipe = FAILURE_FIRST_SFT_RECIPES[scenario_type]
        taxonomy = taxonomy_map[scenario_type]
        linked_failures: List[Dict[str, Any]] = []
        category_order: List[str] = []

        for failure_record in sorted(records_by_scenario[scenario_type], key=lambda item: item["base_score"]):
            benchmark_case = benchmark_cases[failure_record["case_id"]]
            categories = _case_categories(benchmark_case)
            for category in categories:
                if category not in category_order:
                    category_order.append(category)

            linked_failures.append(
                {
                    "case_id": failure_record["case_id"],
                    "title": failure_record["title"],
                    "base_score": failure_record["base_score"],
                    "priority_label": _priority_label(failure_record["base_score"]),
                    "category_constraints": categories,
                    "user_turns": benchmark_case.get("user_turns", []),
                    "criteria": benchmark_case.get("criteria", {}),
                    "rationale": failure_record.get("rationale", ""),
                }
            )

        matched_train = _select_bundle_records(
            sft_train,
            source_task=recipe["source_task"],
            source_scenario=recipe["source_scenario"],
            allowed_categories=category_order,
        )
        matched_val = _select_bundle_records(
            sft_val,
            source_task=recipe["source_task"],
            source_scenario=recipe["source_scenario"],
            allowed_categories=category_order,
        )

        bundles.append(
            {
                "bundle_id": recipe["bundle_id"],
                "display_name": recipe["display_name"],
                "scenario_type": scenario_type,
                "family_id": taxonomy.family_id,
                "family_name": taxonomy.display_name,
                "target_stage": "sft",
                "target_case_count": len(linked_failures),
                "allowed_categories": category_order,
                "source_slice": {
                    "slice_id": recipe["source_slice_id"],
                    "task": recipe["source_task"],
                    "scenario": recipe["source_scenario"],
                    "selection_mode": recipe["selection_mode"],
                },
                "selection_rule": {
                    "task_equals": recipe["source_task"],
                    "scenario_equals": recipe["source_scenario"],
                    "categories_in": category_order,
                },
                "source_coverage": {
                    "train_count": len(matched_train),
                    "val_count": len(matched_val),
                    "total_count": len(matched_train) + len(matched_val),
                    "train_category_counts": _split_category_counts(matched_train, category_order),
                    "val_category_counts": _split_category_counts(matched_val, category_order),
                },
                "diagnosis_checks": taxonomy.diagnosis_checks,
                "objective": recipe.get("objective") or (taxonomy.data_recipe.objective if taxonomy.data_recipe else ""),
                "must_include": recipe["must_include"],
                "hard_constraints": recipe["hard_constraints"],
                "notes": recipe["notes"],
                "owner_override": {
                    "enabled": bool(recipe["owner_override"]),
                    "from_owner": taxonomy.failure_owner,
                    "to_owner": recipe["owner_override"] or taxonomy.failure_owner,
                    "reason": recipe["override_reason"],
                },
                "preview_prompts": _preview_prompts(matched_train),
                "linked_failures": linked_failures,
            }
        )

    bundles.sort(key=lambda item: min(case["base_score"] for case in item["linked_failures"]))

    summary = {
        "included_failure_case_count": len(included_records),
        "excluded_failure_case_count": len(excluded_records),
        "bundle_count": len(bundles),
        "scenario_counts": {
            scenario_type: len(records_by_scenario.get(scenario_type, [])) for scenario_type in ALLOWED_SCENARIOS
        },
        "source_train_count": sum(bundle["source_coverage"]["train_count"] for bundle in bundles),
        "source_val_count": sum(bundle["source_coverage"]["val_count"] for bundle in bundles),
        "source_total_count": sum(bundle["source_coverage"]["total_count"] for bundle in bundles),
    }

    excluded_failures = [
        {
            "case_id": item["case_id"],
            "title": item["title"],
            "scenario_type": item["scenario_type"],
            "proposed_action": item["proposed_action"],
            "base_score": item["base_score"],
            "reason": item["rationale"],
        }
        for item in sorted(excluded_records, key=lambda entry: entry["base_score"])
    ]

    output_root = Path(failure_plan_path).resolve().parent
    return {
        "generated_at": datetime.now().isoformat(),
        "phase": PHASE_NAME,
        "failure_plan_path": str(failure_plan_path),
        "benchmark_path": str(benchmark_path),
        "source_paths": {
            "sft_train_path": str(sft_train_path),
            "sft_val_path": str(sft_val_path),
        },
        "scope": {
            "include_actions": ["sft"],
            "include_scenarios": list(ALLOWED_SCENARIOS),
            "exclude_scenarios": list(EXCLUDED_SCENARIOS),
            "train_now": False,
            "generate_jsonl_now": False,
        },
        "summary": summary,
        "bundles": bundles,
        "excluded_failures": excluded_failures,
        "next_outputs": {
            "reserved_sft_train_path": str(output_root / "sft_train.jsonl"),
            "reserved_sft_val_path": str(output_root / "sft_val.jsonl"),
            "reserved_sft_model_dir": str(output_root / "sft_model_v1"),
        },
    }


def render_markdown(plan: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Failure-First SFT 数据计划",
        "",
        f"- 生成时间：{plan['generated_at']}",
        f"- 阶段：{plan['phase']}",
        f"- failure 输入：{plan['failure_plan_path']}",
        f"- benchmark 输入：{plan['benchmark_path']}",
        f"- 当前仅生成计划：是",
        f"- 当前启动训练：否",
        "",
        "## 汇总",
        "",
        "| 场景 | failure case 数 | source slice | train | val | total |",
        "|---|---:|---|---:|---:|---:|",
    ]

    for bundle in plan["bundles"]:
        coverage = bundle["source_coverage"]
        lines.append(
            f"| {bundle['scenario_type']} | {bundle['target_case_count']} | "
            f"{bundle['source_slice']['slice_id']} | {coverage['train_count']} | "
            f"{coverage['val_count']} | {coverage['total_count']} |"
        )

    lines.extend(
        [
            "",
            f"- 纳入 failure case：{plan['summary']['included_failure_case_count']}",
            f"- 排除 failure case：{plan['summary']['excluded_failure_case_count']}",
            f"- 计划可用 source 总量：{plan['summary']['source_train_count']} / "
            f"{plan['summary']['source_val_count']} / {plan['summary']['source_total_count']} "
            "(train / val / total)",
            "",
            "## 纳入范围",
            "",
            "| case_id | 场景 | base 分数 | 品类 | 对应 source slice |",
            "|---|---|---:|---|---|",
        ]
    )

    for bundle in plan["bundles"]:
        for case in bundle["linked_failures"]:
            lines.append(
                f"| {case['case_id']} | {bundle['scenario_type']} | {case['base_score']:.4f} | "
                f"{' / '.join(case['category_constraints'])} | {bundle['source_slice']['slice_id']} |"
            )

    if plan["excluded_failures"]:
        lines.extend(
            [
                "",
                "## 明确排除",
                "",
                "| case_id | 场景 | 动作 | base 分数 | 原因 |",
                "|---|---|---|---:|---|",
            ]
        )
        for item in plan["excluded_failures"]:
            lines.append(
                f"| {item['case_id']} | {item['scenario_type']} | {item['proposed_action']} | "
                f"{item['base_score']:.4f} | {item['reason']} |"
            )

    for bundle in plan["bundles"]:
        coverage = bundle["source_coverage"]
        lines.extend(
            [
                "",
                f"## {bundle['display_name']}",
                "",
                f"- 场景家族：{bundle['family_name']}",
                f"- failure case 数：{bundle['target_case_count']}",
                f"- 允许品类：{', '.join(bundle['allowed_categories'])}",
                f"- source slice：{bundle['source_slice']['slice_id']}",
                f"- source task/scenario：{bundle['source_slice']['task']} / {bundle['source_slice']['scenario']}",
                f"- 可用样本：{coverage['train_count']} / {coverage['val_count']} / {coverage['total_count']} "
                "(train / val / total)",
                f"- 目标：{bundle['objective']}",
            ]
        )

        owner_override = bundle["owner_override"]
        if owner_override["enabled"]:
            lines.append(
                f"- owner override：{owner_override['from_owner']} -> {owner_override['to_owner']}，"
                f"{owner_override['reason']}"
            )

        lines.extend(
            [
                "",
                "来源类别分布：",
            ]
        )
        for category, count in coverage["train_category_counts"].items():
            val_count = coverage["val_category_counts"].get(category, 0)
            lines.append(f"- {category}：{count} / {val_count} (train / val)")

        lines.extend(
            [
                "",
                "诊断检查：",
            ]
        )
        for item in bundle["diagnosis_checks"]:
            lines.append(f"- {item}")

        lines.extend(
            [
                "",
                "样本必须包含：",
            ]
        )
        for item in bundle["must_include"]:
            lines.append(f"- {item}")

        lines.extend(
            [
                "",
                "硬约束：",
            ]
        )
        for item in bundle["hard_constraints"]:
            lines.append(f"- {item}")

        if bundle["notes"]:
            lines.extend(
                [
                    "",
                    "备注：",
                ]
            )
            for item in bundle["notes"]:
                lines.append(f"- {item}")

        if bundle["preview_prompts"]:
            lines.extend(
                [
                    "",
                    "样本预览提示词：",
                ]
            )
            for item in bundle["preview_prompts"]:
                lines.append(f"- {item}")

        lines.extend(
            [
                "",
                "关联 failure case：",
            ]
        )
        for case in bundle["linked_failures"]:
            lines.append(
                f"- {case['priority_label']} {case['case_id']} ({case['base_score']:.4f}) "
                f"[{' / '.join(case['category_constraints'])}]：{case['rationale']}"
            )

    lines.extend(
        [
            "",
            "## 下一步边界",
            "",
            f"- 预留导出文件：{plan['next_outputs']['reserved_sft_train_path']}",
            f"- 预留导出文件：{plan['next_outputs']['reserved_sft_val_path']}",
            "- 下一步只允许按本计划导出 SFT 数据，不启动训练。",
        ]
    )

    return "\n".join(lines).strip() + "\n"


def save_plan(plan: Dict[str, Any], output_path: str | Path) -> Dict[str, Path]:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = output_file.with_suffix(".md")
    markdown_path.write_text(render_markdown(plan), encoding="utf-8")
    return {"json": output_file, "markdown": markdown_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the failure-first SFT data plan.")
    parser.add_argument(
        "--failure-plan",
        default="outputs/failure_first_qwen25_3b_v1/failure_cases_lt_0p5.json",
        help="Path to the failure-first case list.",
    )
    parser.add_argument(
        "--sft-train",
        default="data/training/sft_train.jsonl",
        help="Path to the base SFT train JSONL.",
    )
    parser.add_argument(
        "--sft-val",
        default="data/training/sft_val.jsonl",
        help="Path to the base SFT val JSONL.",
    )
    parser.add_argument(
        "--output",
        default="outputs/failure_first_qwen25_3b_v1/sft_data_plan.json",
        help="Path to write the SFT plan JSON.",
    )
    args = parser.parse_args()

    plan = build_plan(
        failure_plan_path=args.failure_plan,
        sft_train_path=args.sft_train,
        sft_val_path=args.sft_val,
    )
    paths = save_plan(plan, args.output)
    print(f"JSON: {paths['json']}")
    print(f"Markdown: {paths['markdown']}")


if __name__ == "__main__":
    main()
