#!/usr/bin/env python3
"""
Export Phase 2 failure-first SFT data from the generated plan.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_focus_data import _load_jsonl, _sanitize_structured_sft_records, _write_jsonl


FOLLOWUP_SNEAKER_BLOCKED_KEYWORDS = ("安静", "智能", "高清", "长续航")
FOLLOWUP_SNEAKER_ALLOWED_KEYWORDS = ("慢跑", "脚感更软", "缓震", "稳定性", "抓地", "防滑", "透气", "通勤", "走路", "支撑")
FOLLOWUP_SNEAKER_ALLOWED_PROMPT_FEATURES = {"透气", "缓震", "支撑", "防滑", "轻便"}
FOLLOWUP_SNEAKER_ALLOWED_PROMPT_SCENE_TOKENS = ("通勤", "慢跑", "健身", "户外步行", "周末运动", "日常通勤", "慢跑训练", "健身训练", "走路")
FAILURE_FIRST_CONTROL_MARKERS = ("🖇", "垞", "�", "ropic", "玭")
LEAKED_ROLE_PATTERN = re.compile(r"(?:^|\n)(?:[^\n]{0,3})?(?:user|assistant)\n", re.IGNORECASE)
STRUCTURED_PROMPT_PATTERN = re.compile(
    r"预算(?:控制在)?(?P<budget>\d+)元以内，主要用于(?P<scene>.+?)，更看重(?P<feature>.+?)。",
)
SNEAKER_RECOMMENDATION_ITEM_PATTERN = re.compile(
    r"(?P<idx>\d+)\.\s+\*\*(?P<title>.*?)\*\*\n"
    r"- 价格：¥(?P<price>\d+(?:\.\d+)?)\n"
    r"- 评分：(?P<rating>\d+(?:\.\d+)?)/5(?:（(?P<rating_count>\d+)条评价）)?\n"
    r"- 适用场景：(?P<scene>.*?)\n"
    r"- 推荐理由：(?P<reason>.*?)(?=\n\n\d+\.\s+\*\*|\n\n以上为本次推荐结果。|\Z)",
    re.S,
)
SNEAKER_TITLE_TRAITS = {
    "缓震版": {"慢跑", "脚感更软", "缓震"},
    "轻弹版": {"慢跑", "脚感更软", "轻便", "通勤"},
    "越野版": {"稳定性", "抓地", "防滑"},
    "竞速版": {"慢跑", "轻便"},
}
SNEAKER_SCENE_TRAITS = {
    "慢跑训练": {"慢跑", "脚感更软", "缓震"},
    "日常通勤": {"通勤", "轻便"},
    "户外步行": {"抓地", "防滑", "稳定性"},
    "健身训练": {"稳定性", "支撑"},
    "周末运动": {"慢跑", "通勤"},
}
LOW_STRUCTURE_SIMPLE_CATEGORY_CONFIG = {
    "机械键盘": {
        "noun": "一把",
        "intro": "可以，先给您简单推荐 3 款：",
        "closing": "如果想先缩到一款，我会优先看 **{title}**，更适合办公室安静打字。",
    },
    "蓝牙耳机": {
        "noun": "一款",
        "intro": "可以，先给您简单推荐 3 款：",
        "closing": "如果想先缩到一款，我会优先看 **{title}**，价格和评分都更稳一些。",
    },
    "运动鞋": {
        "noun": "一双",
        "intro": "可以，先给您简单看 3 双：",
        "closing": "如果想先缩到一双，我会优先看 **{title}**，预算内更稳一些。",
    },
}


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _user_messages(record: Dict[str, Any]) -> List[str]:
    return [item.get("content", "") for item in record.get("messages", []) if item.get("role") == "user"]


def _assistant_messages(record: Dict[str, Any]) -> List[str]:
    return [item.get("content", "") for item in record.get("messages", []) if item.get("role") == "assistant"]


def _match_bundle_record(record: Dict[str, Any], selection_rule: Dict[str, Any]) -> bool:
    metadata = record.get("metadata", {})
    if metadata.get("task") != selection_rule.get("task_equals"):
        return False
    if metadata.get("scenario") != selection_rule.get("scenario_equals"):
        return False
    allowed_categories = set(selection_rule.get("categories_in", []))
    record_categories = set(metadata.get("categories", []))
    return bool(record_categories & allowed_categories)


def _bundle_specific_filter(bundle: Dict[str, Any], record: Dict[str, Any]) -> tuple[bool, List[str]]:
    reasons: List[str] = []
    if bundle.get("bundle_id") == "failure_first_followup_memory":
        user_messages = _user_messages(record)
        all_user_text = "\n".join(user_messages)
        for keyword in FOLLOWUP_SNEAKER_BLOCKED_KEYWORDS:
            if keyword in all_user_text:
                reasons.append(keyword)
        if not any(keyword in all_user_text for keyword in FOLLOWUP_SNEAKER_ALLOWED_KEYWORDS):
            reasons.append("off_target_followup_focus")
        if not _is_followup_prompt_aligned(user_messages[0] if user_messages else ""):
            reasons.append("off_target_followup_prompt")
    return (len(reasons) == 0, reasons)


def _dedupe_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for record in records:
        fingerprint = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(record)
    return deduped


def _parse_sneaker_prompt(user_text: str) -> Dict[str, str]:
    match = re.search(r"主要用于(?P<scene>.+?)，更看重(?P<feature>.+?)。", user_text)
    if not match:
        return {"scene": "", "feature": ""}
    return {
        "scene": match.group("scene").strip(),
        "feature": match.group("feature").strip(),
    }


def _is_followup_prompt_aligned(user_text: str) -> bool:
    prompt_info = _parse_sneaker_prompt(user_text)
    scene = prompt_info["scene"]
    feature = prompt_info["feature"]
    if not scene or not feature:
        return False
    if feature not in FOLLOWUP_SNEAKER_ALLOWED_PROMPT_FEATURES:
        return False
    return any(token in scene for token in FOLLOWUP_SNEAKER_ALLOWED_PROMPT_SCENE_TOKENS)


def _parse_structured_prompt(user_text: str) -> Dict[str, str]:
    match = STRUCTURED_PROMPT_PATTERN.search(user_text)
    if not match:
        return {"budget": "", "scene": "", "feature": ""}
    return {
        "budget": match.group("budget").strip(),
        "scene": match.group("scene").strip(),
        "feature": match.group("feature").strip(),
    }


def _parse_sneaker_recommendation_items(assistant_text: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for match in SNEAKER_RECOMMENDATION_ITEM_PATTERN.finditer(assistant_text):
        items.append(
            {
                "idx": int(match.group("idx")),
                "title": match.group("title").strip(),
                "price": float(match.group("price")),
                "rating": float(match.group("rating")),
                "rating_count": int(match.group("rating_count") or 0),
                "scene": match.group("scene").strip(),
                "reason": match.group("reason").strip(),
            }
        )
    return items


def _parse_structured_recommendation_items(assistant_text: str) -> List[Dict[str, Any]]:
    return _parse_sneaker_recommendation_items(assistant_text)


def _select_sneaker_followup_profile(prompt_scene: str, prompt_feature: str) -> Dict[str, str]:
    if "慢跑" in prompt_scene:
        return {
            "primary": "慢跑",
            "secondary_question": "如果我更在意脚感更软一点，你更建议哪双？",
            "backup_focus": "日常通勤",
        }
    if prompt_feature == "缓震":
        return {
            "primary": "脚感更软",
            "secondary_question": "如果周末慢跑更多一点，你更建议哪双？",
            "backup_focus": "稳定性",
        }
    if prompt_feature == "支撑":
        return {
            "primary": "稳定性",
            "secondary_question": "如果周末慢跑更多一点，你更建议哪双？",
            "backup_focus": "轻便",
        }
    if prompt_feature == "防滑":
        return {
            "primary": "抓地",
            "secondary_question": "如果下雨天通勤更多一点，你更建议哪双？",
            "backup_focus": "轻便",
        }
    if prompt_feature == "透气":
        return {
            "primary": "透气",
            "secondary_question": "如果夏天走路时间更长一点，你更建议哪双？",
            "backup_focus": "缓震",
        }
    if prompt_feature == "轻便":
        return {
            "primary": "通勤",
            "secondary_question": "如果走路时间更长一点，你更建议哪双？",
            "backup_focus": "慢跑",
        }
    return {
        "primary": "慢跑",
        "secondary_question": "如果我更在意脚感更软一点，你更建议哪双？",
        "backup_focus": "日常通勤",
    }


def _collect_sneaker_traits(item: Dict[str, Any]) -> set[str]:
    traits: set[str] = set()
    title = item["title"]
    scene = item["scene"]
    for keyword, values in SNEAKER_TITLE_TRAITS.items():
        if keyword in title:
            traits.update(values)
    for keyword, values in SNEAKER_SCENE_TRAITS.items():
        if keyword in scene:
            traits.update(values)
    return traits


def _score_sneaker_item(item: Dict[str, Any], focus: str) -> float:
    score = item["rating"] * 10 - item["price"] / 200.0
    traits = _collect_sneaker_traits(item)
    if focus in traits:
        score += 4.0
    if focus in {"慢跑", "脚感更软"} and traits & {"慢跑", "脚感更软", "缓震"}:
        score += 1.5
    if focus in {"稳定性", "抓地"} and traits & {"稳定性", "抓地", "防滑", "支撑"}:
        score += 1.5
    if focus == "通勤" and traits & {"通勤", "轻便"}:
        score += 1.5
    if focus == "透气" and "透气" in traits:
        score += 1.5
    return score


def _build_sneaker_focus_reason(item: Dict[str, Any], focus: str) -> str:
    title = item["title"]
    scene = item["scene"]
    traits = _collect_sneaker_traits(item)
    if "缓震版" in title and focus in {"慢跑", "脚感更软", "缓震"}:
        return "这款名称里直接带“缓震版”，更贴近您这轮更关注的脚感和慢跑体验。"
    if "轻弹版" in title and focus in {"慢跑", "脚感更软", "通勤"}:
        return "这款名称里带“轻弹版”，更贴近脚感更软和日常走路更轻松的取向。"
    if "越野版" in title and focus in {"稳定性", "抓地", "防滑"}:
        return "这款名称里带“越野版”，更贴近稳定性和抓地这类需求。"
    if scene and focus in traits:
        return f"它当前标注的适用场景是“{scene}”，和您这轮更关注的{focus}更接近。"
    if focus == "透气":
        return "这两款都没有直接展开透气参数时，我会优先保留当前评分更稳、第二轮更容易收敛的一款。"
    return f"从当前候选信息看，它整体更贴近您这轮更关注的{focus}。"


def _build_sneaker_comparison_reason(chosen: Dict[str, Any], alternate: Dict[str, Any]) -> str:
    if chosen["rating"] > alternate["rating"]:
        return (
            f"它当前评分是 {chosen['rating']}/5，高于另一款的 {alternate['rating']}/5，"
            "作为第二轮收敛更稳妥。"
        )
    if chosen["price"] < alternate["price"]:
        return (
            f"它的价格是 ¥{chosen['price']:.2f}，比另一款的 ¥{alternate['price']:.2f} "
            "更容易控制预算。"
        )
    return (
        f"它当前评分是 {chosen['rating']}/5，价格也仍在预算内，"
        "作为二选一结论更稳妥。"
    )


def _build_sneaker_followup_from_recommendation(record: Dict[str, Any]) -> Dict[str, Any] | None:
    metadata = record.get("metadata", {})
    if metadata.get("task") != "product_recommendation":
        return None
    if metadata.get("scenario") != "budget_grounded_recommendation":
        return None
    if metadata.get("categories") != ["运动鞋"]:
        return None

    user_messages = _user_messages(record)
    assistant_messages = _assistant_messages(record)
    if len(user_messages) != 1 or len(assistant_messages) != 1:
        return None

    prompt_info = _parse_sneaker_prompt(user_messages[0])
    items = _parse_structured_recommendation_items(assistant_messages[0])
    if len(items) < 2:
        return None

    profile = _select_sneaker_followup_profile(
        prompt_scene=prompt_info["scene"],
        prompt_feature=prompt_info["feature"],
    )
    first_item = items[0]
    second_item = items[1]
    first_score = _score_sneaker_item(first_item, profile["primary"])
    second_score = _score_sneaker_item(second_item, profile["primary"])

    chosen_item = first_item
    alternate_item = second_item
    chosen_idx = 1
    alternate_idx = 2
    if second_score > first_score:
        chosen_item = second_item
        alternate_item = first_item
        chosen_idx = 2
        alternate_idx = 1

    followup_user_text = (
        f"第1款和第2款相比，哪一双更适合{profile['primary']}？"
        f"{profile['secondary_question']}"
    )
    followup_assistant_text = (
        f"如果只在第1款和第2款里二选一，我更建议第{chosen_idx}款 **{chosen_item['title']}**。\n\n"
        f"结论：如果这轮更看重{profile['primary']}，就先选第{chosen_idx}款。\n"
        f"原因1：{_build_sneaker_focus_reason(chosen_item, profile['primary'])}\n"
        f"原因2：{_build_sneaker_comparison_reason(chosen_item, alternate_item)}\n"
        f"对比看，第{alternate_idx}款 **{alternate_item['title']}** 更偏{profile['backup_focus']}，所以这轮不作为首选。"
    )

    synthetic_metadata = dict(metadata)
    original_tags = set(synthetic_metadata.get("tags", []))
    original_tags.update(
        {
            "failure_first_supplement",
            "multi_turn",
            "followup",
            "direct_answer",
            "sneaker_specific",
        }
    )
    synthetic_metadata.update(
        {
            "task": "failure_repair_sft",
            "scenario": "followup_direct_answer_repair",
            "tags": sorted(original_tags),
            "source_record_task": metadata.get("task", ""),
            "source_record_scenario": metadata.get("scenario", ""),
            "source_record_kind": "budget_grounded_recommendation_to_followup",
        }
    )

    return {
        "messages": [
            record["messages"][0],
            {"role": "user", "content": user_messages[0]},
            {"role": "assistant", "content": assistant_messages[0]},
            {"role": "user", "content": followup_user_text},
            {"role": "assistant", "content": followup_assistant_text},
        ],
        "metadata": synthetic_metadata,
    }


def _build_low_structure_budget_phrase(budget: str, variant: int) -> str:
    if not budget:
        return "预算内"
    if variant == 0:
        return f"{budget}元左右"
    return f"{budget}元以内"


def _build_low_structure_simple_user_text(
    category: str,
    budget: str,
    scene: str,
    feature: str,
    source_user_text: str,
) -> str:
    config = LOW_STRUCTURE_SIMPLE_CATEGORY_CONFIG[category]
    variant = sum(ord(ch) for ch in source_user_text) % 3
    budget_phrase = _build_low_structure_budget_phrase(budget, variant)
    noun = config["noun"]

    if variant == 0:
        if feature:
            return f"我想买{noun}{budget_phrase}的{category}，主要{scene}，如果能兼顾{feature}更好，给我简单推荐一下。"
        return f"我想买{noun}{budget_phrase}的{category}，主要{scene}，给我简单推荐一下。"
    if variant == 1:
        return f"预算先卡在{budget_phrase}，我主要{scene}，想买{noun}{category}，麻烦简单推荐两三款。"
    return f"想入手{budget_phrase}的{category}，主要{scene}，给我简单推荐几款就行。"


def _compress_simple_reason(item: Dict[str, Any], feature: str) -> str:
    reason = item.get("reason", "").strip()
    reason = re.sub(r"^价格在预算内，?", "", reason)
    reason = re.sub(r"结合.+?与当前评分表现，?", "", reason)
    reason = reason.strip("。 ")
    if feature and feature in item.get("reason", ""):
        return f"更偏{feature}取向，当前评分也比较稳。"
    if reason:
        if reason.startswith("更适合"):
            reason = reason.replace("更适合", "偏", 1)
        if reason.startswith("适合"):
            reason = "更" + reason
        if not reason.endswith("。"):
            reason += "。"
        return reason
    scene = item.get("scene", "").strip()
    if scene:
        return f"偏{scene}取向，当前评分也比较稳。"
    return "价格和评分都比较稳。"


def _pick_simple_lead_item(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return sorted(items, key=lambda item: (-item["rating"], item["price"], item["idx"]))[0]


def _build_low_structure_simple_assistant_text(
    category: str,
    scene: str,
    feature: str,
    items: List[Dict[str, Any]],
) -> str:
    config = LOW_STRUCTURE_SIMPLE_CATEGORY_CONFIG[category]
    selected_items = items[:3]
    lines = [config["intro"], ""]
    for idx, item in enumerate(selected_items, start=1):
        lines.append(
            f"{idx}. **{item['title']}**：¥{item['price']:.2f}，评分 {item['rating']}/5，"
            f"适合{item['scene']}，{_compress_simple_reason(item, feature)}"
        )
    lead_item = _pick_simple_lead_item(selected_items)
    lines.extend(
        [
            "",
            f"按您这轮主要{scene}的需求，{config['closing'].format(title=lead_item['title'])}",
        ]
    )
    return "\n".join(lines)


def _build_low_structure_simple_recommendation(record: Dict[str, Any]) -> Dict[str, Any] | None:
    metadata = record.get("metadata", {})
    if metadata.get("task") != "product_recommendation":
        return None
    if metadata.get("scenario") != "budget_grounded_recommendation":
        return None

    categories = metadata.get("categories", [])
    if len(categories) != 1:
        return None
    category = categories[0]
    if category not in LOW_STRUCTURE_SIMPLE_CATEGORY_CONFIG:
        return None

    user_messages = _user_messages(record)
    assistant_messages = _assistant_messages(record)
    if len(user_messages) != 1 or len(assistant_messages) != 1:
        return None

    prompt_info = _parse_structured_prompt(user_messages[0])
    if not prompt_info["budget"] or not prompt_info["scene"]:
        return None

    items = _parse_structured_recommendation_items(assistant_messages[0])
    if len(items) < 2:
        return None

    simple_user_text = _build_low_structure_simple_user_text(
        category=category,
        budget=prompt_info["budget"],
        scene=prompt_info["scene"],
        feature=prompt_info["feature"],
        source_user_text=user_messages[0],
    )
    simple_assistant_text = _build_low_structure_simple_assistant_text(
        category=category,
        scene=prompt_info["scene"],
        feature=prompt_info["feature"],
        items=items,
    )

    synthetic_metadata = dict(metadata)
    original_tags = set(synthetic_metadata.get("tags", []))
    original_tags.update(
        {
            "failure_first_supplement",
            "simple_recommendation",
            "low_structure",
            "concise_multi_option",
        }
    )
    synthetic_metadata.update(
        {
            "task": "failure_repair_sft",
            "scenario": "simple_recommendation_low_structure_repair",
            "tags": sorted(original_tags),
            "source_record_task": metadata.get("task", ""),
            "source_record_scenario": metadata.get("scenario", ""),
            "source_record_kind": "budget_grounded_recommendation_to_simple_low_structure",
        }
    )

    return {
        "messages": [
            record["messages"][0],
            {"role": "user", "content": simple_user_text},
            {"role": "assistant", "content": simple_assistant_text},
        ],
        "metadata": synthetic_metadata,
    }


def _build_followup_supplements(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    supplements: List[Dict[str, Any]] = []
    for record in records:
        synthetic = _build_sneaker_followup_from_recommendation(record)
        if synthetic is not None:
            supplements.append(synthetic)
    return supplements


def _build_simple_recommendation_supplements(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    supplements: List[Dict[str, Any]] = []
    for record in records:
        synthetic = _build_low_structure_simple_recommendation(record)
        if synthetic is not None:
            supplements.append(synthetic)
    return supplements


def _sanitize_failure_first_assistant_text(text: str) -> str:
    cleaned = str(text).replace("\r\n", "\n").strip()
    leaked_role_match = LEAKED_ROLE_PATTERN.search(cleaned)
    if leaked_role_match:
        cleaned = cleaned[: leaked_role_match.start()].rstrip()
    for marker in FAILURE_FIRST_CONTROL_MARKERS:
        cleaned = cleaned.replace(marker, "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _sanitize_failure_first_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned_records: List[Dict[str, Any]] = []
    for record in records:
        record_copy = copy.deepcopy(record)
        for message in record_copy.get("messages", []):
            if message.get("role") == "assistant":
                message["content"] = _sanitize_failure_first_assistant_text(message.get("content", ""))
        cleaned_records.append(record_copy)
    return cleaned_records


def build_export(
    plan_path: str | Path,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    plan = _load_json(plan_path)
    source_paths = plan["source_paths"]
    sft_train = _load_jsonl(source_paths["sft_train_path"])
    sft_val = _load_jsonl(source_paths["sft_val_path"])

    output_root = Path(output_dir) if output_dir else Path(plan["next_outputs"]["reserved_sft_train_path"]).resolve().parent
    output_root.mkdir(parents=True, exist_ok=True)

    exported_train: List[Dict[str, Any]] = []
    exported_val: List[Dict[str, Any]] = []
    bundle_reports: List[Dict[str, Any]] = []

    for bundle in plan.get("bundles", []):
        selection_rule = bundle["selection_rule"]
        train_candidates = [record for record in sft_train if _match_bundle_record(record, selection_rule)]
        val_candidates = [record for record in sft_val if _match_bundle_record(record, selection_rule)]

        kept_train: List[Dict[str, Any]] = []
        kept_val: List[Dict[str, Any]] = []
        dropped_reasons_train: Counter[str] = Counter()
        dropped_reasons_val: Counter[str] = Counter()
        synthetic_train: List[Dict[str, Any]] = []
        synthetic_val: List[Dict[str, Any]] = []

        for record in train_candidates:
            keep, reasons = _bundle_specific_filter(bundle, record)
            if keep:
                kept_train.append(record)
            else:
                for reason in reasons:
                    dropped_reasons_train[reason] += 1

        for record in val_candidates:
            keep, reasons = _bundle_specific_filter(bundle, record)
            if keep:
                kept_val.append(record)
            else:
                for reason in reasons:
                    dropped_reasons_val[reason] += 1

        kept_existing_train_count = len(kept_train)
        kept_existing_val_count = len(kept_val)

        if bundle.get("bundle_id") == "failure_first_followup_memory":
            synthetic_train = _build_followup_supplements(sft_train)
            synthetic_val = _build_followup_supplements(sft_val)
        if bundle.get("bundle_id") == "failure_first_simple_recommendation":
            synthetic_train = _build_simple_recommendation_supplements(sft_train)
            synthetic_val = _build_simple_recommendation_supplements(sft_val)

        kept_train = _sanitize_structured_sft_records(kept_train)
        kept_val = _sanitize_structured_sft_records(kept_val)
        synthetic_train = _sanitize_structured_sft_records(synthetic_train)
        synthetic_val = _sanitize_structured_sft_records(synthetic_val)
        kept_train = _sanitize_failure_first_records(kept_train)
        kept_val = _sanitize_failure_first_records(kept_val)
        synthetic_train = _sanitize_failure_first_records(synthetic_train)
        synthetic_val = _sanitize_failure_first_records(synthetic_val)

        kept_train.extend(synthetic_train)
        kept_val.extend(synthetic_val)

        exported_train.extend(kept_train)
        exported_val.extend(kept_val)

        bundle_reports.append(
            {
                "bundle_id": bundle["bundle_id"],
                "display_name": bundle["display_name"],
                "scenario_type": bundle["scenario_type"],
                "allowed_categories": bundle["allowed_categories"],
                "before_filter": {
                    "train_count": len(train_candidates),
                    "val_count": len(val_candidates),
                },
                "after_filter": {
                    "train_count": len(kept_train),
                    "val_count": len(kept_val),
                },
                "synthetic_added": {
                    "train_count": len(synthetic_train),
                    "val_count": len(synthetic_val),
                },
                "dropped": {
                    "train_count": len(train_candidates) - kept_existing_train_count,
                    "val_count": len(val_candidates) - kept_existing_val_count,
                    "train_reasons": dict(sorted(dropped_reasons_train.items())),
                    "val_reasons": dict(sorted(dropped_reasons_val.items())),
                },
                "linked_failure_case_ids": [item["case_id"] for item in bundle["linked_failures"]],
            }
        )

    exported_train = _dedupe_records(exported_train)
    exported_val = _dedupe_records(exported_val)

    train_path = output_root / "sft_train.jsonl"
    val_path = output_root / "sft_val.jsonl"
    _write_jsonl(train_path, exported_train)
    _write_jsonl(val_path, exported_val)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "plan_path": str(plan_path),
        "source_paths": source_paths,
        "output_dir": str(output_root),
        "output_paths": {
            "sft_train_path": str(train_path),
            "sft_val_path": str(val_path),
        },
        "summary": {
            "train_count": len(exported_train),
            "val_count": len(exported_val),
            "total_count": len(exported_train) + len(exported_val),
        },
        "bundle_reports": bundle_reports,
    }
    return manifest


def render_markdown(manifest: Dict[str, Any]) -> str:
    lines = [
        "# Failure-First SFT 数据导出",
        "",
        f"- 生成时间：{manifest['generated_at']}",
        f"- plan 输入：{manifest['plan_path']}",
        f"- 输出目录：{manifest['output_dir']}",
        f"- train/val：{manifest['summary']['train_count']} / {manifest['summary']['val_count']}",
        "",
        "## Bundle 汇总",
        "",
        "| Bundle | 场景 | before train | before val | synthetic train | synthetic val | after train | after val |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for report in manifest["bundle_reports"]:
        lines.append(
            f"| {report['bundle_id']} | {report['scenario_type']} | "
            f"{report['before_filter']['train_count']} | {report['before_filter']['val_count']} | "
            f"{report['synthetic_added']['train_count']} | {report['synthetic_added']['val_count']} | "
            f"{report['after_filter']['train_count']} | {report['after_filter']['val_count']} |"
        )

    for report in manifest["bundle_reports"]:
        lines.extend(
            [
                "",
                f"## {report['display_name']}",
                "",
                f"- 允许品类：{', '.join(report['allowed_categories'])}",
                f"- 过滤前：{report['before_filter']['train_count']} / {report['before_filter']['val_count']} (train / val)",
                f"- 补样本：{report['synthetic_added']['train_count']} / {report['synthetic_added']['val_count']} (train / val)",
                f"- 过滤后：{report['after_filter']['train_count']} / {report['after_filter']['val_count']} (train / val)",
                f"- 关联 failure case：{', '.join(report['linked_failure_case_ids'])}",
            ]
        )

        if report["dropped"]["train_count"] or report["dropped"]["val_count"]:
            lines.extend(
                [
                    "",
                    "丢弃原因：",
                    f"- train：{report['dropped']['train_count']} -> {report['dropped']['train_reasons']}",
                    f"- val：{report['dropped']['val_count']} -> {report['dropped']['val_reasons']}",
                ]
            )

    return "\n".join(lines).strip() + "\n"


def save_manifest(manifest: Dict[str, Any], output_dir: str | Path) -> Dict[str, Path]:
    output_root = Path(output_dir)
    json_path = output_root / "sft_export_manifest.json"
    md_path = output_root / "sft_export_manifest.md"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(manifest), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export failure-first SFT data from the plan.")
    parser.add_argument(
        "--plan",
        default="outputs/failure_first_qwen25_3b_v1/sft_data_plan.json",
        help="Path to the failure-first SFT plan JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to the directory reserved in the plan.",
    )
    args = parser.parse_args()

    manifest = build_export(plan_path=args.plan, output_dir=args.output_dir)
    paths = save_manifest(manifest, manifest["output_dir"])
    print(f"Train: {manifest['output_paths']['sft_train_path']}")
    print(f"Val: {manifest['output_paths']['sft_val_path']}")
    print(f"Manifest JSON: {paths['json']}")
    print(f"Manifest Markdown: {paths['markdown']}")


if __name__ == "__main__":
    main()
