#!/usr/bin/env python3
"""
Export Phase 4 failure-first DPO data for Qwen2.5-3B.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_focus_data import _load_jsonl, _write_jsonl
from src.data import load_product_data


DEFAULT_FAILURE_PLAN = "outputs/failure_first_qwen25_3b_v1/failure_cases_lt_0p5.json"
DEFAULT_DPO_TRAIN = "data/training/dpo_train.jsonl"
DEFAULT_DPO_VAL = "data/training/dpo_val.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/base_failure_dpo_qwen25_3b_v1"

ALLOWED_SCENARIOS = ("honest_boundary", "tone_alignment")

FAILURE_FIRST_DPO_RECIPES: Dict[str, Dict[str, Any]] = {
    "honest_boundary": {
        "bundle_id": "failure_first_honest_boundary",
        "display_name": "Failure-First DPO - 诚实边界",
        "source_slice_id": "safety_honesty",
        "dimension": "safety_honesty",
        "task": "safety_boundary",
        "filter_by_failure_categories": False,
        "objective": "只训练没有证据时坦诚说明、避免库存/效果/未来价格等 unsupported guarantee。",
        "hard_constraints": [
            "只覆盖 honest_boundary failure case",
            "不混入 after_sales_boundary",
            "拒答必须保持客服式表达，不能粗暴打断",
        ],
        "notes": [
            "该 bundle 只解决 unsupported promise，不扩展到售后赔付承诺。",
        ],
    },
    "tone_alignment": {
        "bundle_id": "failure_first_tone_alignment",
        "display_name": "Failure-First DPO - 语气对齐",
        "source_slice_id": "tone_alignment",
        "dimension": "tone_alignment",
        "task": "brand_tone",
        "filter_by_failure_categories": False,
        "objective": "在用户怀疑营销或带情绪时，保持共情、 grounded、非对抗式表达。",
        "hard_constraints": [
            "只覆盖 tone_alignment failure case",
            "不把 tone 问题混成 structured recommendation 修复",
            "不能用强推下单或怼用户的表达换取简短回复",
        ],
        "notes": [
            "该 bundle 解决的是客服语气，不是让模型给出更激进的销售结论。",
        ],
    },
}


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _priority_label(base_score: float) -> str:
    if base_score < 0.25:
        return "P0"
    if base_score < 0.4:
        return "P1"
    return "P2"


def _case_categories(case: Dict[str, Any]) -> List[str]:
    categories: List[str] = []
    for doc in case.get("retrieved_docs", []):
        category = str(doc.get("category", "")).strip()
        if category and category not in categories:
            categories.append(category)
    return categories


def _match_bundle_record(record: Dict[str, Any], recipe: Dict[str, Any]) -> bool:
    metadata = record.get("metadata", {})
    return (
        record.get("dimension") == recipe["dimension"]
        and metadata.get("task") == recipe["task"]
    )


def _build_product_category_map() -> Dict[str, str]:
    product_df = load_product_data()
    return {
        str(row["Product_ID"]).strip(): str(row["Category"]).strip()
        for _, row in product_df.iterrows()
        if str(row["Product_ID"]).strip()
    }


def _record_categories(record: Dict[str, Any], product_category_map: Dict[str, str]) -> List[str]:
    metadata = record.get("metadata", {})
    product_id = str(metadata.get("product_id", "")).strip()
    if not product_id:
        return []
    category = product_category_map.get(product_id, "").strip()
    return [category] if category else []


def _select_bundle_records(
    records: List[Dict[str, Any]],
    recipe: Dict[str, Any],
    allowed_categories: List[str],
    product_category_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    allowed = set(allowed_categories)
    require_category_match = bool(recipe.get("filter_by_failure_categories"))
    selected: List[Dict[str, Any]] = []
    for record in records:
        if not _match_bundle_record(record, recipe):
            continue
        if require_category_match:
            record_categories = set(_record_categories(record, product_category_map))
            if allowed and not (record_categories & allowed):
                continue
        selected.append(record)
    return selected


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


def _preview_prompts(records: List[Dict[str, Any]], limit: int = 2) -> List[str]:
    prompts: List[str] = []
    for record in records[:limit]:
        prompt = str(record.get("prompt", "")).strip()
        if prompt:
            prompts.append(prompt)
    return prompts


def build_export_manifest(
    failure_plan_path: str | Path,
    dpo_train_path: str | Path,
    dpo_val_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    failure_plan = _load_json(failure_plan_path)
    benchmark = _load_json(failure_plan["generated_from"])
    benchmark_cases = {item["case_id"]: item for item in benchmark.get("cases", [])}
    dpo_train = _load_jsonl(dpo_train_path)
    dpo_val = _load_jsonl(dpo_val_path)
    product_category_map = _build_product_category_map()

    records_by_scenario: Dict[str, List[Dict[str, Any]]] = {scenario: [] for scenario in ALLOWED_SCENARIOS}
    excluded_records: List[Dict[str, Any]] = []
    for record in failure_plan.get("records", []):
        scenario_type = record.get("scenario_type")
        if record.get("proposed_action") == "dpo" and scenario_type in ALLOWED_SCENARIOS:
            records_by_scenario[scenario_type].append(record)
        else:
            excluded_records.append(record)

    bundles: List[Dict[str, Any]] = []
    export_train: List[Dict[str, Any]] = []
    export_val: List[Dict[str, Any]] = []

    for scenario_type in ALLOWED_SCENARIOS:
        failure_records = sorted(records_by_scenario.get(scenario_type, []), key=lambda item: item["base_score"])
        if not failure_records:
            continue

        recipe = FAILURE_FIRST_DPO_RECIPES[scenario_type]
        linked_failures: List[Dict[str, Any]] = []
        category_order: List[str] = []
        for failure_record in failure_records:
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

        matched_train = _dedupe_records(
            _select_bundle_records(
                dpo_train,
                recipe=recipe,
                allowed_categories=category_order,
                product_category_map=product_category_map,
            )
        )
        matched_val = _dedupe_records(
            _select_bundle_records(
                dpo_val,
                recipe=recipe,
                allowed_categories=category_order,
                product_category_map=product_category_map,
            )
        )
        export_train.extend(matched_train)
        export_val.extend(matched_val)

        bundles.append(
            {
                "bundle_id": recipe["bundle_id"],
                "display_name": recipe["display_name"],
                "scenario_type": scenario_type,
                "target_stage": "dpo",
                "target_case_count": len(linked_failures),
                "target_failure_categories": category_order,
                "train_scope": "same_dimension_same_category"
                if recipe.get("filter_by_failure_categories")
                else "same_dimension_cross_category",
                "source_slice": {
                    "slice_id": recipe["source_slice_id"],
                    "dimension": recipe["dimension"],
                    "task": recipe["task"],
                },
                "objective": recipe["objective"],
                "hard_constraints": recipe["hard_constraints"],
                "notes": recipe["notes"],
                "linked_failures": linked_failures,
                "train_count": len(matched_train),
                "val_count": len(matched_val),
                "preview_prompts": {
                    "train": _preview_prompts(matched_train),
                    "val": _preview_prompts(matched_val),
                },
            }
        )

    export_train = _dedupe_records(export_train)
    export_val = _dedupe_records(export_val)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    train_out = output_root / "dpo_train.jsonl"
    val_out = output_root / "dpo_val.jsonl"
    _write_jsonl(train_out, export_train)
    _write_jsonl(val_out, export_val)

    return {
        "generated_at": datetime.now().isoformat(),
        "phase": "phase_4_dpo",
        "failure_plan_path": str(failure_plan_path),
        "source_paths": {
            "dpo_train_path": str(dpo_train_path),
            "dpo_val_path": str(dpo_val_path),
            "benchmark_path": str(failure_plan["generated_from"]),
        },
        "output_paths": {
            "output_dir": str(output_root),
            "dpo_train_path": str(train_out),
            "dpo_val_path": str(val_out),
            "manifest_json_path": str(output_root / "dpo_export_manifest.json"),
            "manifest_md_path": str(output_root / "dpo_export_manifest.md"),
        },
        "summary": {
            "allowed_scenarios": list(ALLOWED_SCENARIOS),
            "failure_case_count": sum(len(records_by_scenario[item]) for item in ALLOWED_SCENARIOS),
            "excluded_case_count": len(excluded_records),
            "bundle_count": len(bundles),
            "train_count": len(export_train),
            "val_count": len(export_val),
        },
        "bundles": bundles,
    }


def render_markdown(manifest: Dict[str, Any]) -> str:
    lines = [
        "# Failure-First DPO 数据导出",
        "",
        f"- 生成时间：{manifest['generated_at']}",
        f"- 阶段：{manifest['phase']}",
        f"- 输出目录：{manifest['output_paths']['output_dir']}",
        f"- 导出 train/val：{manifest['summary']['train_count']} / {manifest['summary']['val_count']}",
        "",
        "## Bundle 概览",
        "",
        "| Bundle | 场景 | 失败 case 数 | train | val |",
        "|---|---|---:|---:|---:|",
    ]

    for bundle in manifest["bundles"]:
        lines.append(
            f"| {bundle['display_name']} | {bundle['scenario_type']} | "
            f"{bundle['target_case_count']} | {bundle['train_count']} | {bundle['val_count']} |"
        )

    for bundle in manifest["bundles"]:
        lines.extend(
            [
                "",
                f"## {bundle['display_name']}",
                "",
                f"- 目标：{bundle['objective']}",
                f"- 来源切片：{bundle['source_slice']['slice_id']} "
                f"({bundle['source_slice']['dimension']} / {bundle['source_slice']['task']})",
                f"- 目标失败品类：{', '.join(bundle['target_failure_categories']) or '不限'}",
                f"- 训练样本范围：{bundle['train_scope']}",
                "",
                "### 关联失败 case",
                "",
                "| case_id | 标题 | base 分数 | 优先级 |",
                "|---|---|---:|---|",
            ]
        )
        for failure in bundle["linked_failures"]:
            lines.append(
                f"| {failure['case_id']} | {failure['title']} | "
                f"{failure['base_score']:.4f} | {failure['priority_label']} |"
            )

        lines.extend(["", "### 约束", ""])
        for item in bundle["hard_constraints"]:
            lines.append(f"- {item}")

        lines.extend(["", "### 样本预览", ""])
        for prompt in bundle["preview_prompts"]["train"]:
            lines.append(f"- train: {prompt}")
        for prompt in bundle["preview_prompts"]["val"]:
            lines.append(f"- val: {prompt}")

    return "\n".join(lines).strip() + "\n"


def save_manifest(manifest: Dict[str, Any], output_dir: str | Path) -> Dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "dpo_export_manifest.json"
    md_path = output_root / "dpo_export_manifest.md"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(manifest), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export failure-first DPO data for Phase 4.")
    parser.add_argument("--failure-plan", default=DEFAULT_FAILURE_PLAN, help="Failure plan JSON path")
    parser.add_argument("--dpo-train", default=DEFAULT_DPO_TRAIN, help="Source DPO train JSONL path")
    parser.add_argument("--dpo-val", default=DEFAULT_DPO_VAL, help="Source DPO val JSONL path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    manifest = build_export_manifest(
        failure_plan_path=args.failure_plan,
        dpo_train_path=args.dpo_train,
        dpo_val_path=args.dpo_val,
        output_dir=args.output_dir,
    )
    save_manifest(manifest, args.output_dir)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
