"""
Reusable slice definitions for focused SFT/DPO training and reporting.
"""

from __future__ import annotations

from typing import Any, Dict, List


SFT_FOCUS_SLICES = [
    {
        "slice_id": "budget_grounded_recommendation",
        "display_name": "SFT - 预算约束推荐基础",
        "scenario": "budget_grounded_recommendation",
        "task": "product_recommendation",
        "plan_scenarios": {"simple_recommendation", "structured_recommendation"},
    },
    {
        "slice_id": "structured_comparison",
        "display_name": "SFT - 结构化对比基础",
        "scenario": "structured_comparison",
        "task": "product_comparison",
        "plan_scenarios": {"structured_comparison"},
    },
    {
        "slice_id": "structured_recommendation_repair",
        "display_name": "SFT P0 - 结构化预算推荐修复",
        "scenario": "structured_recommendation_repair",
        "task": "failure_repair_sft",
        "plan_scenarios": {"structured_recommendation"},
    },
    {
        "slice_id": "followup_direct_answer_repair",
        "display_name": "SFT P0 - 多轮追问直答修复",
        "scenario": "followup_direct_answer_repair",
        "task": "failure_repair_sft",
        "plan_scenarios": {"followup_memory", "clarify_then_recommend"},
    },
]

DPO_FOCUS_SLICES = [
    {
        "slice_id": "information_quality",
        "display_name": "DPO - 推荐信息完整性",
        "dimension": "information_quality",
        "task": "recommendation",
        "plan_scenarios": {"simple_recommendation", "structured_recommendation"},
    },
    {
        "slice_id": "format_compliance",
        "display_name": "DPO - 结构化格式合规",
        "dimension": "format_compliance",
        "task": "comparison",
        "plan_scenarios": {"structured_comparison"},
    },
    {
        "slice_id": "safety_honesty",
        "display_name": "DPO P0 - 诚实拒答",
        "dimension": "safety_honesty",
        "task": "safety_boundary",
        "plan_scenarios": {"honest_boundary"},
    },
    {
        "slice_id": "service_boundary",
        "display_name": "DPO P0 - 售后边界",
        "dimension": "service_boundary",
        "task": "after_sales_boundary",
        "plan_scenarios": {"after_sales_boundary"},
    },
    {
        "slice_id": "tone_alignment",
        "display_name": "DPO P0 - 语气对齐",
        "dimension": "tone_alignment",
        "task": "brand_tone",
        "plan_scenarios": {"tone_alignment"},
    },
]

SFT_FOCUS_SLICE_MAP = {item["slice_id"]: item for item in SFT_FOCUS_SLICES}
DPO_FOCUS_SLICE_MAP = {item["slice_id"]: item for item in DPO_FOCUS_SLICES}

FOCUS_PROFILES = {
    "repair_v1": {
        "profile_id": "repair_v1",
        "display_name": "7B Repair Focused",
        "description": "给 7B repair-only 试验使用的聚焦切片。",
        "sft_slice_ids": [
            "structured_recommendation_repair",
            "followup_direct_answer_repair",
        ],
        "dpo_slice_ids": [
            "safety_honesty",
            "tone_alignment",
        ],
        "recommended_output_dir": "data/training/focused",
        "recommended_sft_output_dir": "outputs/sft_focus",
        "recommended_dpo_output_dir": "outputs/dpo_focus",
        "recommended_base_model": "/data/wtw/Desktop/resume/models/Qwen2.5-7B-Instruct",
    },
    "qwen25_3b_struct": {
        "profile_id": "qwen25_3b_struct",
        "display_name": "Qwen2.5-3B Struct Focused",
        "description": "给 3B 小模型使用的低风险结构化 SFT/DPO 路线。",
        "sft_slice_ids": [
            "budget_grounded_recommendation",
            "structured_comparison",
            "structured_recommendation_repair",
        ],
        "dpo_slice_ids": [
            "information_quality",
            "format_compliance",
        ],
        "recommended_output_dir": "data/training/focused_qwen25_3b_struct",
        "recommended_sft_output_dir": "outputs/sft_focus_qwen25_3b_struct",
        "recommended_dpo_output_dir": "outputs/dpo_focus_qwen25_3b_struct",
        "recommended_base_model": "/data/wtw/Desktop/resume/models/Qwen2.5-3B",
    },
}

DEFAULT_FOCUS_PROFILE_ID = "repair_v1"

DEFAULT_FOCUSED_SFT_SLICE_IDS = [
    "structured_recommendation_repair",
    "followup_direct_answer_repair",
]

DEFAULT_FOCUSED_DPO_SLICE_IDS = [
    "safety_honesty",
    "tone_alignment",
]


def get_focus_profile(profile_id: str) -> Dict[str, Any]:
    if profile_id not in FOCUS_PROFILES:
        raise KeyError(f"Unknown focus profile: {profile_id}")
    return FOCUS_PROFILES[profile_id]


def resolve_focus_slice_ids(
    profile_id: str | None = None,
    sft_slice_ids: List[str] | None = None,
    dpo_slice_ids: List[str] | None = None,
) -> Dict[str, Any]:
    selected_profile_id = profile_id or DEFAULT_FOCUS_PROFILE_ID
    profile = get_focus_profile(selected_profile_id)
    resolved_sft_slice_ids = list(sft_slice_ids or profile["sft_slice_ids"])
    resolved_dpo_slice_ids = list(dpo_slice_ids or profile["dpo_slice_ids"])
    return {
        "profile_id": selected_profile_id,
        "profile": profile,
        "sft_slice_ids": resolved_sft_slice_ids,
        "dpo_slice_ids": resolved_dpo_slice_ids,
    }


def match_sft_record(record: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    metadata = record.get("metadata", {})
    return metadata.get("task") == spec["task"] and metadata.get("scenario") == spec["scenario"]


def match_dpo_record(record: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    metadata = record.get("metadata", {})
    return record.get("dimension") == spec["dimension"] and metadata.get("task") == spec["task"]
