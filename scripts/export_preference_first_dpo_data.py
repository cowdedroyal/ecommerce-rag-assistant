#!/usr/bin/env python3
"""
Export Phase 4 preference-first DPO data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_focus_data import _write_jsonl
from src.data import load_product_data


DEFAULT_TAXONOMY = "outputs/preference_first_dpo_qwen25_3b_v1/dpo_preference_taxonomy.json"
DEFAULT_OUTPUT_DIR = "outputs/preference_first_dpo_qwen25_3b_v1"
DEFAULT_VAL_RATIO = 0.15

PORTABLE_CATEGORIES = {"蓝牙耳机", "手机", "笔记本电脑", "智能手表", "运动鞋"}
SKINCARE_CATEGORIES = {"护肤品"}

VALUE_PROMPTS = (
    "{product} 别讲宣传词，你直接说这款值不值得买。",
    "{product} 我被营销话术坑过，你只按已知信息说值不值得买。",
)

COMMUTE_PROMPTS = (
    "{product} 你直接说适不适合通勤，别绕。",
    "{product} 我只想知道通勤用它稳不稳，先给结论。",
)

TONE_VALUE_PROMPTS = (
    "{product} 我已经看晕了，你别说套话，直接说重点。",
    "{product} 我不想再听营销词，你就像客服一样给我个稳一点的判断。",
)

TONE_COMMUTE_PROMPTS = (
    "{product} 我怕被吹参数，你像客服一样直接说它适不适合通勤。",
    "{product} 别端着，你就按已知信息告诉我通勤用它靠不靠谱。",
)

CLARIFICATION_VALUE_PROMPTS = (
    "{product} 你就直接告诉我值不值得买。",
    "{product} 我现在只要一个明确判断，你先回答值不值得买。",
)

CLARIFICATION_COMMUTE_PROMPTS = (
    "{product} 现在信息已经够了，你就直接说适不适合通勤。",
    "{product} 先别追问我需求，直接回答它适不适合通勤。",
)

STOCK_REFUSAL_PROMPTS = (
    "{product} 明天还有现货吗？你直接给我个准话。",
    "{product} 你直接保证我今晚下单一定能发货。",
)

PRICE_REFUSAL_PROMPTS = (
    "{product} 下个月会不会降价 20%？你直接说准不准。",
    "{product} 你能不能保证这周买它不会亏？",
)

ALLERGY_REFUSAL_PROMPTS = (
    "{product} 你能保证我用了绝对不过敏吗？",
    "{product} 你直接确认一下，敏感肌用它是不是一定没事？",
)

BUNDLE_RUNTIME_CONFIG: Dict[str, Dict[str, Any]] = {
    "verdict_first_value_judgment": {
        "dimension": "tone_alignment",
        "task": "brand_tone",
        "builder": "_build_verdict_first_records",
    },
    "service_tone_calibration": {
        "dimension": "tone_alignment",
        "task": "brand_tone",
        "builder": "_build_service_tone_records",
    },
    "helpful_refusal_boundary": {
        "dimension": "safety_honesty",
        "task": "safety_boundary",
        "builder": "_build_helpful_refusal_records",
    },
    "conservative_boundary_calibration": {
        "dimension": "safety_honesty",
        "task": "safety_boundary",
        "builder": "_build_conservative_boundary_records",
    },
    "no_unnecessary_clarification": {
        "dimension": "tone_alignment",
        "task": "brand_tone",
        "builder": "_build_no_clarification_records",
    },
}


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _product_rows() -> List[Dict[str, Any]]:
    df = load_product_data()
    rows: List[Dict[str, Any]] = []
    for _, item in df.iterrows():
        rows.append(
            {
                "Product_ID": str(item["Product_ID"]).strip(),
                "Product_Title": str(item["Product_Title"]).strip(),
                "Category": str(item["Category"]).strip(),
                "Price": float(item["Price"]),
                "Rating": float(item["Rating"]),
                "features": str(item.get("features", "")).strip(),
            }
        )
    return rows


def _feature_lines(feature_text: str) -> List[str]:
    cleaned = str(feature_text or "").replace("、", ",").replace("，", ",")
    items = [part.strip(" -") for part in cleaned.split(",") if part.strip()]
    return items[:3] or ["暂无更多明确特点"]


def _stable_bucket(product_id: str, val_ratio: float) -> str:
    score = int(hashlib.md5(product_id.encode("utf-8")).hexdigest(), 16) % 100
    return "val" if score < int(val_ratio * 100) else "train"


def _bundle_sort_key(record: Dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    return "|".join(
        [
            metadata.get("product_id", ""),
            metadata.get("bundle_id", ""),
            metadata.get("prompt_family", ""),
            record.get("prompt", ""),
        ]
    )


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


def _rating_verdict(rating: float, ask_commute: bool) -> str:
    if ask_commute:
        if rating >= 4.7:
            return "如果只按当前能确认的信息看，整体偏适合通勤。"
        if rating >= 4.4:
            return "如果只按当前能确认的信息看，可以先列入通勤候选。"
        return "如果只按当前能确认的信息看，我会更建议谨慎一点。"
    if rating >= 4.7:
        return "如果只按当前能确认的信息看，整体偏值得买。"
    if rating >= 4.4:
        return "如果只按当前能确认的信息看，可以先列入候选。"
    return "如果只按当前能确认的信息看，我会更建议谨慎一点。"


def _fact_anchor(row: Dict[str, Any]) -> str:
    features = "、".join(_feature_lines(row["features"])[:2])
    return (
        f"当前能确认的事实只有价格 ¥{row['Price']:.2f}、评分 {row['Rating']:.1f}/5，"
        f"以及 {features} 这些已知特点。"
    )


def _known_info_brief(row: Dict[str, Any]) -> str:
    features = "、".join(_feature_lines(row["features"])[:2])
    return f"价格 ¥{row['Price']:.2f}、评分 {row['Rating']:.1f}/5，以及 {features} 这些已知特点"


def _base_metadata(
    row: Dict[str, Any],
    bundle_id: str,
    display_name: str,
    prompt_family: str,
    dimension: str,
    task: str,
) -> Dict[str, Any]:
    return {
        "task": task,
        "product_id": row["Product_ID"],
        "category": row["Category"],
        "bundle_id": bundle_id,
        "bundle_display_name": display_name,
        "prompt_family": prompt_family,
        "source": "preference_first_template_v1",
        "preference_only": True,
        "gate_target": ["no_prompt_echo", "no_unsupported_claim", "can_answer_directly"],
        "preference_dimension": bundle_id,
    }


def _build_record(
    row: Dict[str, Any],
    bundle: Dict[str, Any],
    dimension: str,
    task: str,
    prompt_family: str,
    prompt: str,
    chosen: str,
    rejected: str,
) -> Dict[str, Any]:
    return {
        "prompt": prompt,
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
        "dimension": dimension,
        "metadata": _base_metadata(
            row=row,
            bundle_id=bundle["bundle_id"],
            display_name=bundle["display_name"],
            prompt_family=prompt_family,
            dimension=dimension,
            task=task,
        ),
    }


def _portable_prompts(row: Dict[str, Any], prompts: tuple[str, ...], commute_prompts: tuple[str, ...]) -> List[str]:
    rendered = [template.format(product=row["Product_Title"]) for template in prompts]
    if row["Category"] in PORTABLE_CATEGORIES:
        rendered.extend(template.format(product=row["Product_Title"]) for template in commute_prompts)
    return rendered


def _build_verdict_first_records(row: Dict[str, Any], bundle: Dict[str, Any], dimension: str, task: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for prompt in _portable_prompts(row, VALUE_PROMPTS, COMMUTE_PROMPTS):
        ask_commute = "通勤" in prompt
        verdict = _rating_verdict(row["Rating"], ask_commute=ask_commute)
        facts = _fact_anchor(row)
        chosen = f"直接回答您：{verdict}{facts} 所以我的建议是先按这个结论判断，不用被宣传词带着走。"
        rejected = f"{facts} 如果只按这些已知信息看，{verdict.replace('如果只按当前能确认的信息看，', '')}"
        prompt_family = "commute_verdict_first" if ask_commute else "value_verdict_first"
        records.append(
            _build_record(
                row=row,
                bundle=bundle,
                dimension=dimension,
                task=task,
                prompt_family=prompt_family,
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
            )
        )
    return records


def _build_service_tone_records(row: Dict[str, Any], bundle: Dict[str, Any], dimension: str, task: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for prompt in _portable_prompts(row, TONE_VALUE_PROMPTS, TONE_COMMUTE_PROMPTS):
        ask_commute = "通勤" in prompt
        verdict = _rating_verdict(row["Rating"], ask_commute=ask_commute)
        facts = _fact_anchor(row)
        chosen = (
            f"理解您是不想再被营销话术带偏，我只按已知信息说。{verdict}"
            f"{facts} 如果您只想先做一轮筛选，我会建议先按这个判断看。"
        )
        rejected = (
            f"按已知信息看，{verdict.replace('如果只按当前能确认的信息看，', '')}"
            f"{facts} 如果您只是想快点定下来，先按这个结论看就行。"
        )
        prompt_family = "commute_service_tone" if ask_commute else "value_service_tone"
        records.append(
            _build_record(
                row=row,
                bundle=bundle,
                dimension=dimension,
                task=task,
                prompt_family=prompt_family,
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
            )
        )
    return records


def _build_helpful_refusal_records(row: Dict[str, Any], bundle: Dict[str, Any], dimension: str, task: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    scenarios = [
        ("stock_helpful_refusal", STOCK_REFUSAL_PROMPTS),
        ("price_helpful_refusal", PRICE_REFUSAL_PROMPTS),
    ]
    if row["Category"] in SKINCARE_CATEGORIES:
        scenarios.append(("allergy_helpful_refusal", ALLERGY_REFUSAL_PROMPTS))

    facts = _known_info_brief(row)
    for prompt_family, templates in scenarios:
        for template in templates:
            prompt = template.format(product=row["Product_Title"])
            if "过敏" in prompt:
                chosen = (
                    "这个我不能保证。是否过敏和个体肤质、完整成分表、既往反应都有关，"
                    f"我现在只能确认它的基础商品信息，比如 {facts}。"
                    "更稳妥的做法是先核对官方成分表，敏感肌先做局部试用，必要时咨询医生或品牌客服。"
                )
                rejected = "这个我不能保证，建议您先看官方成分说明，或者联系品牌客服进一步确认。"
            elif "降价" in prompt or "亏" in prompt:
                chosen = (
                    "这个我不能直接确认。未来价格和活动安排属于实时变化信息，"
                    f"我现在能确认的只有基础商品信息，比如 {facts}。"
                    "更稳妥的是关注商品页活动、加入降价提醒，或者在下单前再和店铺客服确认。"
                )
                rejected = "这个我不能保证，价格要以活动页和实时页面为准，您下单前再看一下会更稳妥。"
            else:
                chosen = (
                    "这个我不能直接保证。实时库存和发货时效会变化，"
                    f"我现在能确认的只有基础商品信息，比如 {facts}。"
                    "更稳妥的是以商品页实时库存、结算页提示和店铺客服答复为准。"
                )
                rejected = "这个我不能保证，建议您看商品页实时提示，或者直接问店铺客服。"

            records.append(
                _build_record(
                    row=row,
                    bundle=bundle,
                    dimension=dimension,
                    task=task,
                    prompt_family=prompt_family,
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                )
            )
    return records


def _build_conservative_boundary_records(row: Dict[str, Any], bundle: Dict[str, Any], dimension: str, task: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    scenarios = [
        ("stock_conservative_boundary", STOCK_REFUSAL_PROMPTS[:1]),
        ("price_conservative_boundary", PRICE_REFUSAL_PROMPTS[:1]),
    ]
    if row["Category"] in SKINCARE_CATEGORIES:
        scenarios.append(("allergy_conservative_boundary", ALLERGY_REFUSAL_PROMPTS[:1]))

    for prompt_family, templates in scenarios:
        for template in templates:
            prompt = template.format(product=row["Product_Title"])
            if "过敏" in prompt:
                chosen = (
                    "这个我不能替您做保证。是否过敏必须看完整成分、个人肤质和实际试用反应，"
                    "所以最终还是要以官方成分说明、局部试用结果和专业意见为准。"
                )
                rejected = "官方一般会把温和配方写清楚，通常问题不大，但具体还是看个人情况。"
            elif "降价" in prompt:
                chosen = (
                    "这个我不能替您预判。未来价格和活动没有实时证据支持，"
                    "所以我不会用“大概率会不会降”替代确认，最终请以活动页和下单时页面为准。"
                )
                rejected = "这个我没法完全保证，但按现在情况看下个月大概率不会大降，最终还是以活动页为准。"
            else:
                chosen = (
                    "这个我不能直接替您确认。库存和发货时效都属于实时信息，"
                    "我不会用“应该有货”代替确认，最终还是以商品页实时库存、结算页和店铺客服答复为准。"
                )
                rejected = "目前看大概率有货，正常来说今晚下单应该能发，不过最终还是以页面为准。"

            records.append(
                _build_record(
                    row=row,
                    bundle=bundle,
                    dimension=dimension,
                    task=task,
                    prompt_family=prompt_family,
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                )
            )
    return records


def _build_no_clarification_records(row: Dict[str, Any], bundle: Dict[str, Any], dimension: str, task: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for prompt in _portable_prompts(row, CLARIFICATION_VALUE_PROMPTS, CLARIFICATION_COMMUTE_PROMPTS):
        ask_commute = "通勤" in prompt
        verdict = _rating_verdict(row["Rating"], ask_commute=ask_commute)
        facts = _fact_anchor(row)
        chosen = f"{verdict}{facts} 如果您只想先做单轮判断，按这些已知信息已经够了。"
        rejected = (
            f"您更看重预算、续航还是具体体验？不过如果只按当前信息先说，"
            f"{verdict.replace('如果只按当前能确认的信息看，', '')}{facts}"
        )
        prompt_family = "commute_no_clarification" if ask_commute else "value_no_clarification"
        records.append(
            _build_record(
                row=row,
                bundle=bundle,
                dimension=dimension,
                task=task,
                prompt_family=prompt_family,
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
            )
        )
    return records


def _resolve_builder(name: str) -> Callable[[Dict[str, Any], Dict[str, Any], str, str], List[Dict[str, Any]]]:
    return globals()[name]


def _take_target(records: List[Dict[str, Any]], target_count: int, val_ratio: float) -> Dict[str, List[Dict[str, Any]]]:
    sorted_records = sorted(records, key=_bundle_sort_key)
    train_records = [
        item
        for item in sorted_records
        if _stable_bucket(item["metadata"]["product_id"], val_ratio=val_ratio) == "train"
    ]
    val_records = [
        item
        for item in sorted_records
        if _stable_bucket(item["metadata"]["product_id"], val_ratio=val_ratio) == "val"
    ]

    val_target = max(1, round(target_count * val_ratio))
    train_target = max(1, target_count - val_target)
    return {
        "train": train_records[:train_target],
        "val": val_records[:val_target],
    }


def build_preference_first_export(
    taxonomy_path: str | Path,
    output_dir: str | Path,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> Dict[str, Any]:
    taxonomy = _load_json(taxonomy_path)
    bundles = taxonomy.get("bundles", [])
    product_rows = _product_rows()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    export_train: List[Dict[str, Any]] = []
    export_val: List[Dict[str, Any]] = []
    bundle_summaries: List[Dict[str, Any]] = []

    for bundle in bundles:
        bundle_id = bundle["bundle_id"]
        runtime_cfg = BUNDLE_RUNTIME_CONFIG[bundle_id]
        builder = _resolve_builder(runtime_cfg["builder"])
        candidates: List[Dict[str, Any]] = []
        for row in product_rows:
            candidates.extend(
                builder(
                    row=row,
                    bundle=bundle,
                    dimension=runtime_cfg["dimension"],
                    task=runtime_cfg["task"],
                )
            )

        candidates = _dedupe_records(candidates)
        selected = _take_target(
            records=candidates,
            target_count=int(bundle["sample_target"]),
            val_ratio=val_ratio,
        )
        export_train.extend(selected["train"])
        export_val.extend(selected["val"])

        bundle_summaries.append(
            {
                "bundle_id": bundle_id,
                "display_name": bundle["display_name"],
                "sample_target": int(bundle["sample_target"]),
                "dimension": runtime_cfg["dimension"],
                "task": runtime_cfg["task"],
                "candidate_count": len(candidates),
                "train_count": len(selected["train"]),
                "val_count": len(selected["val"]),
                "preview_prompts": {
                    "train": [item["prompt"] for item in selected["train"][:2]],
                    "val": [item["prompt"] for item in selected["val"][:2]],
                },
            }
        )

    export_train = _dedupe_records(export_train)
    export_val = _dedupe_records(export_val)

    train_path = output_root / "dpo_train.jsonl"
    val_path = output_root / "dpo_val.jsonl"
    manifest_json_path = output_root / "dpo_data_manifest.json"
    manifest_md_path = output_root / "dpo_data_manifest.md"

    _write_jsonl(train_path, export_train)
    _write_jsonl(val_path, export_val)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "phase": "phase_4b_preference_only_dpo_data",
        "taxonomy_path": str(taxonomy_path),
        "output_dir": str(output_root),
        "generation_mode": "taxonomy_template_only",
        "source_paths": {
            "product_catalog": "data/processed/products_zh.csv",
        },
        "output_paths": {
            "dpo_train_path": str(train_path),
            "dpo_val_path": str(val_path),
            "manifest_json_path": str(manifest_json_path),
            "manifest_md_path": str(manifest_md_path),
        },
        "summary": {
            "bundle_count": len(bundle_summaries),
            "train_count": len(export_train),
            "val_count": len(export_val),
            "val_ratio": val_ratio,
        },
        "policy": {
            "gate_checks": taxonomy.get("benchmark_principles", {}).get("gate_checks", []),
            "excluded_work": taxonomy.get("excluded_work", []),
            "note": "只导出 preference-only pair；不复用旧的 prompt echo、unsupported claim 或 direct-answer bootstrap 样本。",
        },
        "bundles": bundle_summaries,
    }

    manifest_json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_md_path.write_text(render_markdown(manifest), encoding="utf-8")
    return manifest


def render_markdown(manifest: Dict[str, Any]) -> str:
    lines = [
        "# Preference-First DPO 数据导出",
        "",
        f"- 生成时间：{manifest['generated_at']}",
        f"- 阶段：{manifest['phase']}",
        f"- taxonomy：{manifest['taxonomy_path']}",
        f"- 输出目录：{manifest['output_dir']}",
        f"- 导出 train/val：{manifest['summary']['train_count']} / {manifest['summary']['val_count']}",
        f"- 生成方式：{manifest['generation_mode']}",
        "",
        "## 数据策略",
        "",
        f"- 说明：{manifest['policy']['note']}",
        f"- gate checks：{', '.join(manifest['policy']['gate_checks'])}",
        "",
        "## Bundle 概览",
        "",
        "| Bundle | target | train | val | dimension | task |",
        "|---|---:|---:|---:|---|---|",
    ]

    for item in manifest["bundles"]:
        lines.append(
            f"| {item['display_name']} | {item['sample_target']} | {item['train_count']} | "
            f"{item['val_count']} | {item['dimension']} | {item['task']} |"
        )

    for item in manifest["bundles"]:
        lines.extend(
            [
                "",
                f"## {item['display_name']}",
                "",
                f"- bundle_id：{item['bundle_id']}",
                f"- 目标样本量：{item['sample_target']}",
                f"- 候选样本量：{item['candidate_count']}",
                f"- 导出 train/val：{item['train_count']} / {item['val_count']}",
                f"- 维度：{item['dimension']} / {item['task']}",
                "",
                "样本预览：",
            ]
        )
        for prompt in item["preview_prompts"]["train"]:
            lines.append(f"- train: {prompt}")
        for prompt in item["preview_prompts"]["val"]:
            lines.append(f"- val: {prompt}")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export preference-first DPO data")
    parser.add_argument("--taxonomy", default=DEFAULT_TAXONOMY, help="preference taxonomy JSON path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="output directory")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="validation ratio by product")
    args = parser.parse_args()

    build_preference_first_export(
        taxonomy_path=args.taxonomy,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
