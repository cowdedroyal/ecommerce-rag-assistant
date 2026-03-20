#!/usr/bin/env python3
"""
Build a preference-first Phase 4 DPO plan.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_FAILURE_PLAN = "outputs/failure_first_qwen25_3b_v1/failure_cases_lt_0p5.json"
DEFAULT_STRICT_BENCHMARK = (
    "outputs/base_failure_dpo_qwen25_3b_v3_strict_repair/"
    "benchmark_base_vs_dpo_failure_first_strict_rerun_v2.json"
)
DEFAULT_OUTPUT_DIR = "outputs/preference_first_dpo_qwen25_3b_v1"
PHASE_NAME = "phase_4_preference_first_dpo"


PREFERENCE_BUNDLES: List[Dict[str, Any]] = [
    {
        "bundle_id": "verdict_first_value_judgment",
        "display_name": "结论优先的价值判断",
        "primary_goal": "当用户问“值不值得买 / 适不适合通勤”时，先给结论，再补价格、评分和已知特点。",
        "why_dpo": "两个回答都 grounded 且安全，差异主要在回答优先级和客服可读性，适合偏好学习。",
        "source_from_current": ["tone_alignment"],
        "prerequisites": [
            "模型已经能直接回答，不再 prompt echo",
            "模型已经会基于证据给出价格、评分或特点",
        ],
        "chosen_must_include": [
            "第一句直接回答值不值得买 / 适不适合通勤",
            "至少 1 个事实锚点，例如价格、评分或特点",
            "不使用强推下单措辞",
        ],
        "rejected_patterns": [
            "先堆事实但不下结论",
            "结论藏在最后",
            "不必要的追问或反问",
        ],
        "pair_shape": "same_prompt + same_fact_set -> direct_verdict_first vs fact_dump_or_indirect_verdict",
        "sample_target": 120,
        "example_prompts": [
            "别讲宣传词，你直接说这款值不值得买。",
            "你就回答我适不适合通勤，别绕。",
        ],
    },
    {
        "bundle_id": "service_tone_calibration",
        "display_name": "客服语气校准",
        "primary_goal": "在相同结论和相同事实下，优先学习更克制、更像客服的表达。",
        "why_dpo": "不是补“会不会答”，而是在多个可用回答里偏向更稳的品牌语气。",
        "source_from_current": ["tone_alignment"],
        "prerequisites": [
            "结论和事实都已经齐全",
            "回答不依赖额外流程和工具",
        ],
        "chosen_must_include": [
            "简短共情或承接用户顾虑",
            "克制表达，不推责、不强卖",
            "保留 grounded 信息",
        ],
        "rejected_patterns": [
            "生硬、顶嘴、推责",
            "强销售腔",
            "空洞安抚但不给信息",
        ],
        "pair_shape": "same_prompt + same_verdict + same_fact_set -> calm_service_tone vs salesy_or_curt_tone",
        "sample_target": 100,
        "example_prompts": [
            "我被营销坑怕了，你别说套话。",
            "我已经看晕了，你直接说重点。",
        ],
    },
    {
        "bundle_id": "helpful_refusal_boundary",
        "display_name": "有帮助的拒答边界",
        "primary_goal": "在不能确认时，学习“说明不能确认 + 解释原因 + 给下一步建议”的拒答。",
        "why_dpo": "两个回答都拒绝，但 chosen 更有帮助、更像客服，适合 preference pair。",
        "source_from_current": ["honest_boundary"],
        "prerequisites": [
            "模型已经不会直接乱承诺库存、降价、疗效等未知事实",
            "模型已经知道用无法确认/不能保证表明边界",
        ],
        "chosen_must_include": [
            "明确不能确认/不能保证",
            "说明缺失的是实时库存、未来信息或完整成分/政策",
            "给出商品页、结算页、客服或官方说明等下一步建议",
        ],
        "rejected_patterns": [
            "只说不能保证，但没有后续建议",
            "拒答语气过硬",
            "含糊其辞地回避问题",
        ],
        "pair_shape": "same_prompt -> refusal_with_reason_and_fallback vs refusal_only_or_evasive_refusal",
        "sample_target": 90,
        "example_prompts": [
            "你直接保证今晚下单一定能发货。",
            "你能保证我用了绝对不过敏吗？",
        ],
    },
    {
        "bundle_id": "conservative_boundary_calibration",
        "display_name": "保守边界校准",
        "primary_goal": "在都没有直接违规时，偏向更保守、更少擦边承诺的回答。",
        "why_dpo": "两个回答都表面安全，但一个更保守、更符合业务风控边界，这正是 DPO 强项。",
        "source_from_current": ["honest_boundary"],
        "prerequisites": [
            "模型已经能完成基本拒答",
            "高风险售后动作仍由人工或规则兜底",
        ],
        "chosen_must_include": [
            "不用“大概率/通常/应该能”替代证据",
            "不做半承诺式表述",
            "把最终确认权交给商品页、结算页或人工客服",
        ],
        "rejected_patterns": [
            "大概率没问题",
            "库存应该够",
            "通常都能发",
        ],
        "pair_shape": "same_prompt -> conservative_safe_answer vs borderline_safe_answer",
        "sample_target": 80,
        "example_prompts": [
            "现在是不是肯定有货？",
            "下个月会不会降价 20%？",
        ],
    },
    {
        "bundle_id": "no_unnecessary_clarification",
        "display_name": "避免不必要追问",
        "primary_goal": "在证据已足够时直接回答，而不是先追问一轮把客服回复拖慢。",
        "why_dpo": "这不是事实能力缺口，而是回答策略偏好，适合用 preference pair 校准。",
        "source_from_current": ["tone_alignment"],
        "prerequisites": [
            "当前 prompt 已给出明确问题",
            "系统上下文已提供可用商品证据",
        ],
        "chosen_must_include": [
            "先回答当前问题",
            "必要时一句话补充限制条件",
            "不把本该直接回答的问题转成追问",
        ],
        "rejected_patterns": [
            "先问新问题再回答",
            "要求用户补信息，但当前信息已够回答",
            "把单轮问题拖成多轮",
        ],
        "pair_shape": "same_prompt + sufficient_context -> direct_answer vs clarification_first",
        "sample_target": 70,
        "example_prompts": [
            "你直接说适不适合通勤。",
            "你就告诉我值不值得买。",
        ],
    },
]


EXCLUDED_WORK: List[Dict[str, Any]] = [
    {
        "work_item": "direct_answer_bootstrap",
        "owner": "sft",
        "reason": "模型连“值不值得买/适不适合通勤”的直接回答动作都没学稳，这属于行为模板问题，不是偏好排序问题。",
    },
    {
        "work_item": "anti_prompt_echo_cleanup",
        "owner": "sft_or_generation_cleaner",
        "reason": "prompt echo 属于生成缺陷或清洗缺陷，DPO 容易把它学成表面高分假象。",
    },
    {
        "work_item": "unsupported_claim_blocking",
        "owner": "sft_or_rule",
        "reason": "库存、疗效、未来价格等 unsupported claim 要先用硬约束或模板兜底，不能把第一道防线押给 DPO。",
    },
    {
        "work_item": "after_sales_boundary",
        "owner": "manual_review",
        "reason": "退款、赔付、改地址等高风险动作更适合人工或受控流程，不适合继续扩展成 DPO 主战场。",
    },
]


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _owner_for_reasons(reasons: List[str]) -> str:
    normalized = set(reasons)
    if "prompt_echo" in normalized:
        return "sft_or_generation_cleaner"
    if "unsupported_claim" in normalized or "no_refusal" in normalized:
        return "sft_or_rule"
    if "no_direct_answer" in normalized:
        return "sft"
    return "manual_review"


def _current_case_diagnosis(strict_benchmark_path: str | Path) -> List[Dict[str, Any]]:
    benchmark = _load_json(strict_benchmark_path)
    stage_map = benchmark.get("stages", {})
    base_cases = {item["case_id"]: item for item in stage_map.get("base", {}).get("cases", [])}
    dpo_cases = {item["case_id"]: item for item in stage_map.get("dpo", {}).get("cases", [])}
    routing = {item["case_id"]: item for item in benchmark.get("routing", [])}
    diagnosis: List[Dict[str, Any]] = []

    for case_id in sorted(set(base_cases) | set(dpo_cases)):
        base_case = base_cases.get(case_id, {})
        dpo_case = dpo_cases.get(case_id, {})
        dpo_reasons = dpo_case.get("scores", {}).get("hard_fail_reasons", [])
        base_reasons = base_case.get("scores", {}).get("hard_fail_reasons", [])
        reasons = dpo_reasons or base_reasons
        diagnosis.append(
            {
                "case_id": case_id,
                "title": dpo_case.get("title") or base_case.get("title", case_id),
                "scenario_type": dpo_case.get("scenario_type") or base_case.get("scenario_type", ""),
                "base_score": base_case.get("scores", {}).get("overall", 0.0),
                "dpo_score": dpo_case.get("scores", {}).get("overall", 0.0),
                "recommended_stage": routing.get(case_id, {}).get("recommended_stage", "manual_review"),
                "current_blockers": reasons or ["no_stable_preference_signal"],
                "owner": _owner_for_reasons(reasons),
            }
        )
    return diagnosis


def _failure_scope(failure_plan_path: str | Path) -> Dict[str, Any]:
    failure_plan = _load_json(failure_plan_path)
    records = failure_plan.get("records", [])
    selected = [item for item in records if item.get("proposed_action") == "dpo"]
    return {
        "failure_plan_path": str(failure_plan_path),
        "base_failure_case_count": len(records),
        "dpo_tagged_case_count": len(selected),
        "dpo_tagged_case_ids": [item["case_id"] for item in selected],
    }


def build_plan(
    failure_plan_path: str | Path,
    strict_benchmark_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    scope = _failure_scope(failure_plan_path)
    current_diagnosis = _current_case_diagnosis(strict_benchmark_path)
    output_root = Path(output_dir)

    return {
        "generated_at": datetime.now().isoformat(),
        "phase": PHASE_NAME,
        "failure_scope": scope,
        "strict_benchmark_path": str(strict_benchmark_path),
        "output_dir": str(output_root),
        "objective": (
            "把第四阶段从“修 base 不会做的高风险场景”改成“只优化已经会做但答法不够像客服的偏好差异型场景”。"
        ),
        "dpo_suitable_bundles": PREFERENCE_BUNDLES,
        "excluded_work": EXCLUDED_WORK,
        "current_case_diagnosis": current_diagnosis,
        "benchmark_principles": {
            "gate_checks": [
                "no_prompt_echo",
                "no_unsupported_claim",
                "can_answer_directly",
            ],
            "preference_checks": [
                "verdict_first",
                "helpful_refusal",
                "service_tone",
                "conservative_boundary",
                "no_unnecessary_clarification",
            ],
            "policy": (
                "只有先通过 gate 的回答，才进入 DPO preference 打分；"
                "否则一律视为 SFT/规则层问题，不计入 DPO 主结论。"
            ),
        },
        "phase_rewrite": [
            {
                "phase_id": "phase_4a",
                "name": "冻结 DPO taxonomy",
                "goal": "只保留 preference-first 的 bundle，不再把 capability bootstrap 场景塞给 DPO。",
                "outputs": [
                    str(output_root / "dpo_preference_taxonomy.json"),
                    str(output_root / "PLAN.md"),
                ],
            },
            {
                "phase_id": "phase_4b",
                "name": "导出 preference-only DPO 数据",
                "goal": "基于 bundle 定义导出 chosen/rejected 都基本可用的 preference pair。",
                "outputs": [
                    str(output_root / "dpo_train.jsonl"),
                    str(output_root / "dpo_val.jsonl"),
                    str(output_root / "dpo_data_manifest.md"),
                ],
            },
            {
                "phase_id": "phase_4c",
                "name": "偏好型 benchmark",
                "goal": "先过 gate，再比较 preference；报告只写通过 gate 的 DPO 场景。",
                "outputs": [
                    str(output_root / "benchmark_preference_first_dpo.json"),
                    str(output_root / "benchmark_preference_first_dpo.md"),
                ],
            },
        ],
    }


def render_plan_markdown(plan: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Preference-First DPO Plan",
        "",
        "## 目的",
        "",
        f"- 生成时间：{plan['generated_at']}",
        f"- 阶段：{plan['phase']}",
        f"- failure 输入：{plan['failure_scope']['failure_plan_path']}",
        f"- strict benchmark 输入：{plan['strict_benchmark_path']}",
        f"- 输出目录：{plan['output_dir']}",
        "",
        f"- 核心目标：{plan['objective']}",
        "",
        "## 当前结论",
        "",
        "- 现有 `tone_alignment / honest_boundary` strict benchmark 里，很多失败属于 `SFT` 或规则层问题，不适合作为 DPO 主战场。",
        "- DPO 只适合处理“两个回答都基本可用，但一个更像客服、更稳、更保守”的偏好差异。",
        "",
        "## 当前 case 诊断",
        "",
        "| case_id | 场景 | base | dpo | 当前 blocker | owner | 建议去向 |",
        "|---|---|---:|---:|---|---|---|",
    ]

    for item in plan["current_case_diagnosis"]:
        lines.append(
            f"| {item['case_id']} | {item['scenario_type']} | {item['base_score']:.4f} | "
            f"{item['dpo_score']:.4f} | {', '.join(item['current_blockers'])} | "
            f"{item['owner']} | {item['recommended_stage']} |"
        )

    lines.extend(
        [
            "",
            "## 适合 DPO 的场景 taxonomy",
            "",
            "| bundle_id | 场景 | sample_target | 为什么适合 DPO |",
            "|---|---|---:|---|",
        ]
    )

    for bundle in plan["dpo_suitable_bundles"]:
        lines.append(
            f"| {bundle['bundle_id']} | {bundle['display_name']} | {bundle['sample_target']} | {bundle['why_dpo']} |"
        )

    for bundle in plan["dpo_suitable_bundles"]:
        lines.extend(
            [
                "",
                f"## {bundle['display_name']}",
                "",
                f"- 来源场景：{', '.join(bundle['source_from_current'])}",
                f"- 目标：{bundle['primary_goal']}",
                f"- 为什么适合 DPO：{bundle['why_dpo']}",
                f"- pair 形状：{bundle['pair_shape']}",
                f"- 建议样本量：{bundle['sample_target']}",
                "",
                "前置条件：",
            ]
        )
        for item in bundle["prerequisites"]:
            lines.append(f"- {item}")

        lines.extend(["", "chosen 必须包含："])
        for item in bundle["chosen_must_include"]:
            lines.append(f"- {item}")

        lines.extend(["", "rejected 常见模式："])
        for item in bundle["rejected_patterns"]:
            lines.append(f"- {item}")

        lines.extend(["", "示例提示词："])
        for item in bundle["example_prompts"]:
            lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## 明确不归 DPO",
            "",
            "| 工作项 | owner | 原因 |",
            "|---|---|---|",
        ]
    )
    for item in plan["excluded_work"]:
        lines.append(f"| {item['work_item']} | {item['owner']} | {item['reason']} |")

    benchmark = plan["benchmark_principles"]
    lines.extend(
        [
            "",
            "## 新 benchmark 原则",
            "",
            "gate checks：",
        ]
    )
    for item in benchmark["gate_checks"]:
        lines.append(f"- {item}")

    lines.extend(["", "preference checks："])
    for item in benchmark["preference_checks"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            f"- 评测规则：{benchmark['policy']}",
            "",
            "## 重写后的 Phase 4",
            "",
            "| phase | 名称 | 目标 | 输出 |",
            "|---|---|---|---|",
        ]
    )
    for item in plan["phase_rewrite"]:
        lines.append(
            f"| {item['phase_id']} | {item['name']} | {item['goal']} | {'<br>'.join(item['outputs'])} |"
        )

    lines.extend(
        [
            "",
            "## 执行边界",
            "",
            "- 只有先通过 gate 的场景，才能拿来证明 DPO 价值。",
            "- 不再用 `prompt echo / no_direct_answer / unsupported_claim` 这类 case 写 DPO 结论。",
            "- 简历或项目叙事里，DPO 只描述为“偏好型客服回复优化”，不描述为“补齐基础回答能力”。",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def save_outputs(plan: Dict[str, Any], output_dir: str | Path) -> Dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    taxonomy_path = output_root / "dpo_preference_taxonomy.json"
    plan_path = output_root / "preference_first_dpo_plan.json"
    markdown_path = output_root / "PLAN.md"

    taxonomy_payload = {
        "generated_at": plan["generated_at"],
        "phase": plan["phase"],
        "bundles": plan["dpo_suitable_bundles"],
        "excluded_work": plan["excluded_work"],
        "benchmark_principles": plan["benchmark_principles"],
    }

    taxonomy_path.write_text(json.dumps(taxonomy_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_plan_markdown(plan), encoding="utf-8")
    return {
        "taxonomy_path": taxonomy_path,
        "plan_path": plan_path,
        "markdown_path": markdown_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a preference-first DPO plan.")
    parser.add_argument("--failure-plan", default=DEFAULT_FAILURE_PLAN, help="Failure plan JSON path")
    parser.add_argument("--strict-benchmark", default=DEFAULT_STRICT_BENCHMARK, help="Strict benchmark JSON path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    plan = build_plan(
        failure_plan_path=args.failure_plan,
        strict_benchmark_path=args.strict_benchmark,
        output_dir=args.output_dir,
    )
    save_outputs(plan, args.output_dir)
    print(json.dumps({"output_dir": args.output_dir, "bundle_count": len(plan["dpo_suitable_bundles"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
