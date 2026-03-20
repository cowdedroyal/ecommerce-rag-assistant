#!/usr/bin/env python3
"""
Export focused training slices for targeted SFT/DPO experiments.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from training.focus_slices import (
    DEFAULT_FOCUS_PROFILE_ID,
    DEFAULT_FOCUSED_DPO_SLICE_IDS,
    DEFAULT_FOCUSED_SFT_SLICE_IDS,
    DPO_FOCUS_SLICE_MAP,
    FOCUS_PROFILES,
    SFT_FOCUS_SLICE_MAP,
    resolve_focus_slice_ids,
    match_dpo_record,
    match_sft_record,
)


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _select_sft(records: List[Dict[str, Any]], slice_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    selected: Dict[str, List[Dict[str, Any]]] = {slice_id: [] for slice_id in slice_ids}
    for slice_id in slice_ids:
        spec = SFT_FOCUS_SLICE_MAP[slice_id]
        selected[slice_id] = [record for record in records if match_sft_record(record, spec)]
    return selected


def _select_dpo(records: List[Dict[str, Any]], slice_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    selected: Dict[str, List[Dict[str, Any]]] = {slice_id: [] for slice_id in slice_ids}
    for slice_id in slice_ids:
        spec = DPO_FOCUS_SLICE_MAP[slice_id]
        selected[slice_id] = [record for record in records if match_dpo_record(record, spec)]
    return selected


def _flatten(records_by_slice: Dict[str, List[Dict[str, Any]]], slice_ids: List[str]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for slice_id in slice_ids:
        merged.extend(records_by_slice.get(slice_id, []))
    return merged


STRUCTURED_SFT_SCENARIOS = {
    "budget_grounded_recommendation",
    "structured_comparison",
    "structured_recommendation_repair",
}

TAIL_CUES = (
    "如果您愿意",
    "如果你愿意",
    "如果您下一轮",
    "如果你下一轮",
    "如需了解",
    "如需更多信息",
    "如果您有其他问题",
    "还有什么想了解",
    "请随时告诉我",
    "随时可以问我",
    "继续提问",
    "继续帮您",
    "进一步筛选",
    "需要我帮您",
)

GARBAGE_MARKERS = ("rumpe", "spep", "+lsi", "𬭩user", "𬭩assistant")


def _sanitize_assistant_content(text: str) -> str:
    cleaned = str(text).replace("\r\n", "\n").strip()
    for marker in GARBAGE_MARKERS:
        marker_index = cleaned.find(marker)
        if marker_index != -1:
            cleaned = cleaned[:marker_index].rstrip()
    for cue in TAIL_CUES:
        cue_index = cleaned.find(cue)
        if cue_index != -1:
            cleaned = cleaned[:cue_index].rstrip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.rstrip(" \n\t-：:，,")


def _sanitize_structured_sft_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned_records: List[Dict[str, Any]] = []
    for record in records:
        scenario = record.get("metadata", {}).get("scenario")
        if scenario not in STRUCTURED_SFT_SCENARIOS:
            cleaned_records.append(record)
            continue

        record_copy = copy.deepcopy(record)
        for message in record_copy.get("messages", []):
            if message.get("role") == "assistant":
                message["content"] = _sanitize_assistant_content(message.get("content", ""))
        cleaned_records.append(record_copy)
    return cleaned_records


def build_focus_export(
    sft_train_path: str | Path,
    sft_val_path: str | Path,
    dpo_train_path: str | Path,
    dpo_val_path: str | Path,
    output_dir: str | Path,
    focus_profile: str | None = None,
    sft_slice_ids: List[str] | None = None,
    dpo_slice_ids: List[str] | None = None,
) -> Dict[str, Any]:
    resolved = resolve_focus_slice_ids(
        profile_id=focus_profile,
        sft_slice_ids=sft_slice_ids,
        dpo_slice_ids=dpo_slice_ids,
    )
    profile_id = resolved["profile_id"]
    profile = resolved["profile"]
    sft_slice_ids = resolved["sft_slice_ids"] or list(DEFAULT_FOCUSED_SFT_SLICE_IDS)
    dpo_slice_ids = resolved["dpo_slice_ids"] or list(DEFAULT_FOCUSED_DPO_SLICE_IDS)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    sft_train = _load_jsonl(sft_train_path)
    sft_val = _load_jsonl(sft_val_path)
    dpo_train = _load_jsonl(dpo_train_path)
    dpo_val = _load_jsonl(dpo_val_path)

    sft_train_by_slice = _select_sft(sft_train, sft_slice_ids)
    sft_val_by_slice = _select_sft(sft_val, sft_slice_ids)
    dpo_train_by_slice = _select_dpo(dpo_train, dpo_slice_ids)
    dpo_val_by_slice = _select_dpo(dpo_val, dpo_slice_ids)

    focused_sft_train = _sanitize_structured_sft_records(_flatten(sft_train_by_slice, sft_slice_ids))
    focused_sft_val = _sanitize_structured_sft_records(_flatten(sft_val_by_slice, sft_slice_ids))
    focused_dpo_train = _flatten(dpo_train_by_slice, dpo_slice_ids)
    focused_dpo_val = _flatten(dpo_val_by_slice, dpo_slice_ids)

    sft_train_out = output_root / "sft_train.jsonl"
    sft_val_out = output_root / "sft_val.jsonl"
    dpo_train_out = output_root / "dpo_train.jsonl"
    dpo_val_out = output_root / "dpo_val.jsonl"

    _write_jsonl(sft_train_out, focused_sft_train)
    _write_jsonl(sft_val_out, focused_sft_val)
    _write_jsonl(dpo_train_out, focused_dpo_train)
    _write_jsonl(dpo_val_out, focused_dpo_val)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "focus_profile": {
            "profile_id": profile_id,
            "display_name": profile.get("display_name", profile_id),
            "description": profile.get("description", ""),
        },
        "source_paths": {
            "sft_train_path": str(sft_train_path),
            "sft_val_path": str(sft_val_path),
            "dpo_train_path": str(dpo_train_path),
            "dpo_val_path": str(dpo_val_path),
        },
        "output_dir": str(output_root),
        "sft": {
            "slice_ids": sft_slice_ids,
            "train_path": str(sft_train_out),
            "val_path": str(sft_val_out),
            "train_count": len(focused_sft_train),
            "val_count": len(focused_sft_val),
            "slice_counts": {
                slice_id: {
                    "train": len(sft_train_by_slice[slice_id]),
                    "val": len(sft_val_by_slice[slice_id]),
                }
                for slice_id in sft_slice_ids
            },
        },
        "dpo": {
            "slice_ids": dpo_slice_ids,
            "train_path": str(dpo_train_out),
            "val_path": str(dpo_val_out),
            "train_count": len(focused_dpo_train),
            "val_count": len(focused_dpo_val),
            "slice_counts": {
                slice_id: {
                    "train": len(dpo_train_by_slice[slice_id]),
                    "val": len(dpo_val_by_slice[slice_id]),
                }
                for slice_id in dpo_slice_ids
            },
        },
    }
    return manifest


def render_markdown(manifest: Dict[str, Any]) -> str:
    profile = manifest.get("focus_profile", {})
    lines = [
        "# Focused 数据导出",
        "",
        f"- 生成时间：{manifest['generated_at']}",
        f"- Profile：{profile.get('profile_id', DEFAULT_FOCUS_PROFILE_ID)}",
        f"- Profile 说明：{profile.get('display_name', '')}",
        f"- 输出目录：{manifest['output_dir']}",
        "",
        "## SFT",
        "",
        f"- 切片：{', '.join(manifest['sft']['slice_ids'])}",
        f"- train/val：{manifest['sft']['train_count']} / {manifest['sft']['val_count']}",
        "",
        "| 切片 | train | val |",
        "|---|---:|---:|",
    ]
    for slice_id, counts in manifest["sft"]["slice_counts"].items():
        lines.append(f"| {slice_id} | {counts['train']} | {counts['val']} |")

    lines.extend(
        [
            "",
            "## DPO",
            "",
            f"- 切片：{', '.join(manifest['dpo']['slice_ids'])}",
            f"- train/val：{manifest['dpo']['train_count']} / {manifest['dpo']['val_count']}",
            "",
            "| 切片 | train | val |",
            "|---|---:|---:|",
        ]
    )
    for slice_id, counts in manifest["dpo"]["slice_counts"].items():
        lines.append(f"| {slice_id} | {counts['train']} | {counts['val']} |")

    return "\n".join(lines).strip() + "\n"


def save_focus_export(manifest: Dict[str, Any], output_dir: str | Path) -> Dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "manifest.json"
    md_path = output_root / "manifest.md"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(manifest), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
