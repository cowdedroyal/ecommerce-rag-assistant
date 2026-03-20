#!/usr/bin/env python3
"""
Summarize failure-repair training slices for quick review before retraining.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from training.focus_slices import (
    DPO_FOCUS_SLICE_MAP,
    SFT_FOCUS_SLICE_MAP,
    resolve_focus_slice_ids,
    match_dpo_record,
    match_sft_record,
)


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


def _last_message(messages: Iterable[Dict[str, Any]], role: str) -> str:
    for item in reversed(list(messages)):
        if item.get("role") == role:
            return item.get("content", "")
    return ""


def _build_sft_preview(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = record.get("metadata", {})
    return {
        "prompt": _last_message(record.get("messages", []), "user"),
        "answer": _last_message(record.get("messages", []), "assistant"),
        "tags": metadata.get("tags", []),
        "product_ids": metadata.get("product_ids", []),
        "categories": metadata.get("categories", []),
    }


def _build_dpo_preview(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = record.get("metadata", {})
    return {
        "prompt": record.get("prompt", ""),
        "chosen": record.get("chosen", ""),
        "rejected": record.get("rejected", ""),
        "task": metadata.get("task", ""),
        "product_name": metadata.get("product_name", ""),
        "product_id": metadata.get("product_id", ""),
        "order_id": metadata.get("order_id", ""),
    }


def _linked_failures(plan: Dict[str, Any], scenario_types: set[str]) -> List[Dict[str, Any]]:
    linked: List[Dict[str, Any]] = []
    for entry in plan.get("entries", []):
        if entry.get("scenario_type") not in scenario_types:
            continue
        linked.append(
            {
                "case_id": entry.get("case_id", ""),
                "title": entry.get("title", ""),
                "priority_label": entry.get("priority_label", ""),
                "priority_score": entry.get("priority_score", 0.0),
                "action_kind": entry.get("action_kind", ""),
                "current_target_score": entry.get("current_target_score", 0.0),
            }
        )
    linked.sort(key=lambda item: (-float(item["priority_score"]), item["case_id"]))
    return linked


def build_summary(
    sft_train_path: str | Path,
    sft_val_path: str | Path,
    dpo_train_path: str | Path,
    dpo_val_path: str | Path,
    plan_path: str | Path,
    focus_profile: str | None = None,
    sft_slice_ids: List[str] | None = None,
    dpo_slice_ids: List[str] | None = None,
    preview_count: int = 2,
) -> Dict[str, Any]:
    sft_train = _load_jsonl(sft_train_path)
    sft_val = _load_jsonl(sft_val_path)
    dpo_train = _load_jsonl(dpo_train_path)
    dpo_val = _load_jsonl(dpo_val_path)
    plan = _load_json(plan_path)
    resolved = resolve_focus_slice_ids(
        profile_id=focus_profile,
        sft_slice_ids=sft_slice_ids,
        dpo_slice_ids=dpo_slice_ids,
    )

    slices: List[Dict[str, Any]] = []

    for slice_id in resolved["sft_slice_ids"]:
        spec = SFT_FOCUS_SLICE_MAP[slice_id]
        train_records = [item for item in sft_train if match_sft_record(item, spec)]
        val_records = [item for item in sft_val if match_sft_record(item, spec)]
        slices.append(
            {
                "slice_id": spec["slice_id"],
                "display_name": spec["display_name"],
                "stage": "sft",
                "task": spec["task"],
                "filter_key": spec["scenario"],
                "count_train": len(train_records),
                "count_val": len(val_records),
                "count_total": len(train_records) + len(val_records),
                "linked_failures": _linked_failures(plan, spec["plan_scenarios"]),
                "previews": [_build_sft_preview(item) for item in train_records[:preview_count]],
            }
        )

    for slice_id in resolved["dpo_slice_ids"]:
        spec = DPO_FOCUS_SLICE_MAP[slice_id]
        train_records = [item for item in dpo_train if match_dpo_record(item, spec)]
        val_records = [item for item in dpo_val if match_dpo_record(item, spec)]
        slices.append(
            {
                "slice_id": spec["slice_id"],
                "display_name": spec["display_name"],
                "stage": "dpo",
                "task": spec["task"],
                "filter_key": spec["dimension"],
                "count_train": len(train_records),
                "count_val": len(val_records),
                "count_total": len(train_records) + len(val_records),
                "linked_failures": _linked_failures(plan, spec["plan_scenarios"]),
                "previews": [_build_dpo_preview(item) for item in train_records[:preview_count]],
            }
        )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "focus_profile": {
            "profile_id": resolved["profile_id"],
            "display_name": resolved["profile"].get("display_name", resolved["profile_id"]),
            "description": resolved["profile"].get("description", ""),
            "sft_slice_ids": resolved["sft_slice_ids"],
            "dpo_slice_ids": resolved["dpo_slice_ids"],
        },
        "inputs": {
            "sft_train_path": str(sft_train_path),
            "sft_val_path": str(sft_val_path),
            "dpo_train_path": str(dpo_train_path),
            "dpo_val_path": str(dpo_val_path),
            "plan_path": str(plan_path),
            "preview_count": preview_count,
        },
        "totals": {
            "sft_train": len(sft_train),
            "sft_val": len(sft_val),
            "dpo_train": len(dpo_train),
            "dpo_val": len(dpo_val),
            "focus_sft_total": sum(item["count_total"] for item in slices if item["stage"] == "sft"),
            "focus_dpo_total": sum(item["count_total"] for item in slices if item["stage"] == "dpo"),
        },
        "slices": slices,
    }
    return summary


def render_markdown(summary: Dict[str, Any]) -> str:
    profile = summary.get("focus_profile", {})
    lines: List[str] = [
        "# P0 数据切片摘要",
        "",
        f"- 生成时间：{summary['generated_at']}",
        f"- Profile：{profile.get('profile_id', '')}",
        f"- Profile 说明：{profile.get('display_name', '')}",
        f"- SFT train/val：{summary['totals']['sft_train']} / {summary['totals']['sft_val']}",
        f"- DPO train/val：{summary['totals']['dpo_train']} / {summary['totals']['dpo_val']}",
        f"- 聚焦 SFT 样本总数：{summary['totals']['focus_sft_total']}",
        f"- 聚焦 DPO 对总数：{summary['totals']['focus_dpo_total']}",
        "",
        "## 切片汇总",
        "",
        "| 切片 | 阶段 | train | val | total |",
        "|---|---|---:|---:|---:|",
    ]

    for item in summary["slices"]:
        lines.append(
            f"| {item['display_name']} | {item['stage']} | {item['count_train']} | "
            f"{item['count_val']} | {item['count_total']} |"
        )

    for item in summary["slices"]:
        lines.extend(
            [
                "",
                f"## {item['display_name']}",
                "",
                f"- 阶段：{item['stage']}",
                f"- 训练切片：{item['filter_key']}",
                f"- 样本数：train {item['count_train']} / val {item['count_val']} / total {item['count_total']}",
            ]
        )

        if item["linked_failures"]:
            lines.append("- 对应 benchmark 失败：")
            for linked in item["linked_failures"]:
                lines.append(
                    f"  - {linked['priority_label']} {linked['title']} "
                    f"({linked['case_id']}, score={linked['current_target_score']})"
                )

        if item["previews"]:
            lines.append("")
            lines.append("样本预览：")
            for index, preview in enumerate(item["previews"], start=1):
                lines.append(f"- 示例 {index}")
                lines.append("```text")
                lines.append(f"Prompt: {preview.get('prompt', '')}")
                if item["stage"] == "sft":
                    lines.append(f"Answer: {preview.get('answer', '')}")
                else:
                    lines.append(f"Chosen: {preview.get('chosen', '')}")
                    lines.append(f"Rejected: {preview.get('rejected', '')}")
                lines.append("```")

    return "\n".join(lines).strip() + "\n"


def save_summary(summary: Dict[str, Any], output_path: str | Path) -> Dict[str, Path]:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = output_file.with_suffix(".md")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    return {"json": output_file, "markdown": markdown_path}
