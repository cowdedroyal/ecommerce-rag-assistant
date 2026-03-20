#!/usr/bin/env python3
"""
Build a benchmark-aligned DPO dataset for the Phase 4 failure-first run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_focus_data import _write_jsonl
from src.data import load_product_data


DEFAULT_FAILURE_PLAN = "outputs/failure_first_qwen25_3b_v1/failure_cases_lt_0p5.json"
DEFAULT_BENCHMARK = "outputs/model_showcase/benchmark_results_qwen25_3b_base_mining_base_model.json"
DEFAULT_OUTPUT_DIR = "outputs/base_failure_dpo_qwen25_3b_v3_strict_repair"
DEFAULT_SEED = 42

TONE_TARGET_CATEGORIES = ("笔记本电脑", "护肤品", "蓝牙耳机")
HONEST_TARGET_CATEGORIES = ("手机",)

EARBUD_COMMUTE_PROMPTS = (
    "{product} 看着像在硬吹参数。你别忽悠我，直接说适不适合通勤。",
    "{product} 我就想知道通勤用它稳不稳，别跟我说套话。",
)

VALUE_PROMPTS = (
    "{product} 看起来还行，但我已经被营销文案骗怕了。你别跟我说套话，只告诉我值不值得买。",
    "{product} 介绍页说得很厉害，但我怕被忽悠。你直接说值不值得买。",
    "{product} 这款产品看着还行，但我怕买回来踩雷，你别跟我说套话。",
)

STOCK_PROMPTS = (
    "{product} 明天还有现货吗？你直接保证我今晚下单一定能发货。",
    "帮我查一下{product}的库存还有多少，你直接告诉我今晚下单能不能保证发货。",
    "{product} 现在是不是肯定有货？你直接给我个准话，我今晚就下单。",
)

TONE_REJECTED_POOL = (
    "这款绝对值，别犹豫，马上下单就对了，不需要再看参数。",
    "你自己都看了这么久了，还问这个？喜欢就买，不喜欢就算了。",
    "这种问题没法回答，自己去商品页看就行。",
)

HONEST_REJECTED_POOL = (
    "目前肯定有货，您今晚下单我可以保证马上发出。",
    "这个您放心，我直接给您保证今晚下单一定能发货。",
    "有现货，您现在下单就行，我可以保证明天收到。",
)

TONE_WEAK_VALUE_REJECTED_POOL = (
    "先看价格，¥{price:.2f}，不便宜。",
    "{product} 适合希望在预算内获得稳定体验的用户。",
    "{prompt}",
)

TONE_WEAK_COMMUTE_REJECTED_POOL = (
    "{product} 适合通勤。",
    "{prompt}",
    "这款绝对值，马上下单就行。",
)

HONEST_MIXED_REJECTED_POOL = (
    "明天还有现货，但不保证您今晚下单一定能发货。",
    "库存大概率没问题，您今晚下单应该能发。",
    "{prompt}",
)

CASE_ALIGNED_TONE_CASE_IDS = (
    "dpo_tone_alignment",
    "dpo_tone_alignment_laptop",
    "dpo_tone_alignment_earbuds",
)

CASE_ALIGNED_HONEST_CASE_IDS = ("dpo_honest_boundary_stock",)


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _feature_lines(feature_text: str) -> List[str]:
    raw = str(feature_text or "").replace("，", ",")
    items = [item.strip(" -") for item in raw.split(",") if item.strip()]
    return items[:3] or ["暂无更多明确特点"]


def _verdict_from_rating(rating: float, ask_commute: bool) -> str:
    if ask_commute:
        if rating >= 4.7:
            return "适不适合通勤：整体偏适合，可以列入候选。"
        if rating >= 4.4:
            return "适不适合通勤：可以先列入候选，但更建议和同价位再比一款。"
        return "适不适合通勤：不建议只看宣传就直接下单。"
    if rating >= 4.7:
        return "值不值得买：整体偏值得买，可以列入候选。"
    if rating >= 4.4:
        return "值不值得买：可以先列入候选，但更建议和同价位再比一款。"
    return "值不值得买：不建议只看宣传就直接下单。"


def _build_tone_chosen_variants(row: Dict[str, Any], ask_commute: bool) -> List[str]:
    product_name = str(row["Product_Title"])
    price = float(row["Price"])
    rating = float(row["Rating"])
    features = _feature_lines(row.get("features", ""))
    verdict = _verdict_from_rating(rating=rating, ask_commute=ask_commute)
    feature_summary = "、".join(features[:2])
    if ask_commute:
        return [
            (
                f"理解您是不想被宣传话术带偏，我直接说结论：{verdict}"
                f"现在能确认的只有价格 ¥{price:.2f}、评分 {rating}/5，重点是 {feature_summary}。"
                "如果您更看重日常通勤续航和切换设备的便利性，可以列入候选；"
                "但我只建议按这些已知信息判断，不建议只看宣传词就下单。"
            ),
            (
                f"直接回答您：{verdict}"
                f"{product_name} 目前可确认的信息是价格 ¥{price:.2f}、评分 {rating}/5，"
                f"以及 {feature_summary} 这些特点。"
                "我不会拿宣传词替您做保证，只能说它可以先放进通勤候选名单。"
            ),
        ]
    return [
        (
            f"理解您是怕被营销话术带偏，我直接说结论：{verdict}"
            f"当前能确认的只有价格 ¥{price:.2f}、评分 {rating}/5，"
            f"更有参考价值的是 {feature_summary}。"
            "如果您的需求和这些已知信息匹配，可以考虑；但我不建议只看宣传词冲动下单。"
        ),
        (
            f"直接回答您：{verdict}"
            f"{product_name} 现在能确认的信息是价格 ¥{price:.2f}、评分 {rating}/5，"
            f"以及 {feature_summary}。"
            "我能给的是基于这些信息的判断，不会用营销语替您拍板。"
        ),
    ]


def _build_honest_chosen_variants(row: Dict[str, Any]) -> List[str]:
    product_name = str(row["Product_Title"])
    price = float(row["Price"])
    rating = float(row["Rating"])
    features = _feature_lines(row.get("features", ""))
    feature_summary = "、".join(features[:2])
    return [
        (
            "理解您是想今晚就把单定下来，但这个我不能直接保证。"
            f"我现在只能确认 {product_name} 的已知商品信息，比如价格 ¥{price:.2f}、评分 {rating}/5，"
            f"以及 {feature_summary}；这些信息不包含实时库存和发货时效。"
            "所以我不能确认明天是否还有现货，也不能承诺您今晚下单一定能发货。"
            "更稳妥的是以下单页实时库存、结算页提示和店铺客服答复为准。"
        ),
        (
            "直接回答您：我不能保证。"
            f"当前资料里只有 {product_name} 的价格 ¥{price:.2f}、评分 {rating}/5 等商品信息，"
            "没有实时库存和仓配时效。"
            "因此我不能确认明天还有没有现货，也不能替您承诺今晚下单一定能发货。"
            "建议以商品页实时库存和店铺客服答复为准。"
        ),
    ]


def _case_doc_to_row(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Product_ID": str(doc["id"]),
        "Product_Title": str(doc["title"]),
        "Category": str(doc["category"]),
        "Price": float(doc["price"]),
        "Rating": float(doc["rating"]),
        "features": str(doc.get("features", "")),
    }


def _tone_case_rejected_pool(row: Dict[str, Any], prompt: str, ask_commute: bool) -> Tuple[str, ...]:
    product_name = str(row["Product_Title"])
    price = float(row["Price"])
    weak_pool = TONE_WEAK_COMMUTE_REJECTED_POOL if ask_commute else TONE_WEAK_VALUE_REJECTED_POOL
    rendered = [item.format(product=product_name, price=price, prompt=prompt) for item in weak_pool]
    return tuple(rendered) + tuple(TONE_REJECTED_POOL)


def _honest_case_rejected_pool(prompt: str) -> Tuple[str, ...]:
    rendered = [item.format(prompt=prompt) for item in HONEST_MIXED_REJECTED_POOL]
    return tuple(rendered) + tuple(HONEST_REJECTED_POOL)


def _build_tone_records(product_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in product_rows:
        category = str(row["Category"])
        product_id = str(row["Product_ID"])
        if category == "蓝牙耳机":
            prompts = EARBUD_COMMUTE_PROMPTS
            ask_commute = True
        else:
            prompts = VALUE_PROMPTS
            ask_commute = False
        chosen_variants = _build_tone_chosen_variants(row, ask_commute=ask_commute)
        for prompt_tpl in prompts:
            prompt = prompt_tpl.format(product=row["Product_Title"])
            rejected_pool = _tone_case_rejected_pool(row, prompt=prompt, ask_commute=ask_commute)
            for idx, chosen in enumerate(chosen_variants):
                records.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected_pool[idx % len(rejected_pool)],
                        "dimension": "tone_alignment",
                        "metadata": {
                            "task": "brand_tone",
                            "product_id": product_id,
                            "source": "generic_targeted",
                        },
                    }
                )
    return records


def _build_honest_records(product_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in product_rows:
        product_id = str(row["Product_ID"])
        chosen_variants = _build_honest_chosen_variants(row)
        for prompt_tpl in STOCK_PROMPTS:
            prompt = prompt_tpl.format(product=row["Product_Title"])
            rejected_pool = _honest_case_rejected_pool(prompt)
            for idx, chosen in enumerate(chosen_variants):
                records.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected_pool[idx % len(rejected_pool)],
                        "dimension": "safety_honesty",
                        "metadata": {
                            "task": "safety_boundary",
                            "product_id": product_id,
                            "source": "generic_targeted",
                        },
                    }
                )
    return records


def _build_case_aligned_tone_records(benchmark_cases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for case_id in CASE_ALIGNED_TONE_CASE_IDS:
        case = benchmark_cases[case_id]
        doc = case["retrieved_docs"][0]
        row = _case_doc_to_row(doc)
        base_prompt = str(case["user_turns"][0])
        ask_commute = "通勤" in base_prompt
        prompt_variants = [base_prompt]
        if ask_commute:
            prompt_variants.extend(
                [
                    f"{row['Product_Title']} 别跟我讲宣传词，直接说适不适合通勤。",
                    f"{row['Product_Title']} 我只问结论，通勤用它到底稳不稳？",
                ]
            )
        else:
            prompt_variants.extend(
                [
                    f"{row['Product_Title']} 我不想听宣传词，你直接说值不值得买。",
                    f"{row['Product_Title']} 我被营销坑怕了，你就直接表态值不值得买。",
                ]
            )
        chosen_variants = _build_tone_chosen_variants(row, ask_commute=ask_commute)
        for prompt in prompt_variants:
            rejected_pool = _tone_case_rejected_pool(row, prompt=prompt, ask_commute=ask_commute)
            for idx, chosen in enumerate(chosen_variants):
                records.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected_pool[idx % len(rejected_pool)],
                        "dimension": "tone_alignment",
                        "metadata": {
                            "task": "brand_tone",
                            "product_id": row["Product_ID"],
                            "source": "benchmark_case_aligned",
                            "case_id": case_id,
                        },
                    }
                )
    return records


def _build_case_aligned_honest_records(benchmark_cases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for case_id in CASE_ALIGNED_HONEST_CASE_IDS:
        case = benchmark_cases[case_id]
        doc = case["retrieved_docs"][0]
        row = _case_doc_to_row(doc)
        base_prompt = str(case["user_turns"][0])
        prompt_variants = [
            base_prompt,
            f"{row['Product_Title']} 我今晚就想下单，你能不能直接保证一定能发货？",
            f"{row['Product_Title']} 明天还有没有现货，你现在能不能给我准话？",
        ]
        chosen_variants = _build_honest_chosen_variants(row)
        for prompt in prompt_variants:
            rejected_pool = _honest_case_rejected_pool(prompt)
            for idx, chosen in enumerate(chosen_variants):
                records.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected_pool[idx % len(rejected_pool)],
                        "dimension": "safety_honesty",
                        "metadata": {
                            "task": "safety_boundary",
                            "product_id": row["Product_ID"],
                            "source": "benchmark_case_aligned",
                            "case_id": case_id,
                        },
                    }
                )
    return records


def _split_train_val(records: List[Dict[str, Any]], val_ratio: float = 0.1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    shuffled = list(records)
    random.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1 - val_ratio)))
    train = shuffled[:split_idx]
    val = shuffled[split_idx:]
    if not val:
        val = shuffled[-1:]
        train = shuffled[:-1]
    return train, val


def _bundle_manifest_entry(
    bundle_id: str,
    scenario_type: str,
    target_categories: List[str],
    records: List[Dict[str, Any]],
    linked_failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    train, val = _split_train_val(records)
    return {
        "bundle_id": bundle_id,
        "scenario_type": scenario_type,
        "target_categories": target_categories,
        "train_count": len(train),
        "val_count": len(val),
        "linked_failures": linked_failures,
        "train_preview": [record["prompt"] for record in train[:2]],
        "val_preview": [record["prompt"] for record in val[:2]],
    }


def build_dataset(
    failure_plan_path: str | Path,
    benchmark_path: str | Path,
    output_dir: str | Path,
    seed: int,
) -> Dict[str, Any]:
    random.seed(seed)
    failure_plan = _load_json(failure_plan_path)
    benchmark = _load_json(benchmark_path)
    benchmark_cases = {item["case_id"]: item for item in benchmark.get("cases", [])}
    product_df = load_product_data()

    tone_rows = [
        row
        for _, row in product_df.iterrows()
        if str(row["Category"]).strip() in TONE_TARGET_CATEGORIES
    ]
    honest_rows = [
        row
        for _, row in product_df.iterrows()
        if str(row["Category"]).strip() in HONEST_TARGET_CATEGORIES
    ]

    tone_records = _build_tone_records(tone_rows) + _build_case_aligned_tone_records(benchmark_cases)
    honest_records = _build_honest_records(honest_rows) + _build_case_aligned_honest_records(benchmark_cases)
    tone_train, tone_val = _split_train_val(tone_records)
    honest_train, honest_val = _split_train_val(honest_records)

    train_records = honest_train + tone_train
    val_records = honest_val + tone_val
    random.shuffle(train_records)
    random.shuffle(val_records)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / "dpo_train.jsonl"
    val_path = output_root / "dpo_val.jsonl"
    _write_jsonl(train_path, train_records)
    _write_jsonl(val_path, val_records)

    def linked_failure(case_ids: List[str]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        records_by_id = {item["case_id"]: item for item in failure_plan.get("records", [])}
        for case_id in case_ids:
            failure_record = records_by_id[case_id]
            benchmark_case = benchmark_cases[case_id]
            items.append(
                {
                    "case_id": case_id,
                    "title": failure_record["title"],
                    "base_score": failure_record["base_score"],
                    "user_turns": benchmark_case.get("user_turns", []),
                }
            )
        return items

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "phase": "phase_4_dpo_targeted",
        "seed": seed,
        "failure_plan_path": str(failure_plan_path),
        "benchmark_path": str(benchmark_path),
        "output_dir": str(output_root),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "summary": {
            "train_count": len(train_records),
            "val_count": len(val_records),
            "tone_train": len(tone_train),
            "tone_val": len(tone_val),
            "honest_train": len(honest_train),
            "honest_val": len(honest_val),
        },
        "bundles": [
            _bundle_manifest_entry(
                bundle_id="targeted_honest_boundary_stock",
                scenario_type="honest_boundary",
                target_categories=list(HONEST_TARGET_CATEGORIES),
                records=honest_records,
                linked_failures=linked_failure(["dpo_honest_boundary_stock"]),
            ),
            _bundle_manifest_entry(
                bundle_id="targeted_tone_alignment",
                scenario_type="tone_alignment",
                target_categories=list(TONE_TARGET_CATEGORIES),
                records=tone_records,
                linked_failures=linked_failure(
                    [
                        "dpo_tone_alignment_laptop",
                        "dpo_tone_alignment",
                        "dpo_tone_alignment_earbuds",
                    ]
                ),
            ),
        ],
    }
    return manifest


def render_markdown(manifest: Dict[str, Any]) -> str:
    lines = [
        "# Targeted Failure-First DPO 数据",
        "",
        f"- 生成时间：{manifest['generated_at']}",
        f"- 阶段：{manifest['phase']}",
        f"- 输出目录：{manifest['output_dir']}",
        f"- train/val：{manifest['summary']['train_count']} / {manifest['summary']['val_count']}",
        "",
        "| Bundle | 场景 | train | val | 目标品类 |",
        "|---|---|---:|---:|---|",
    ]
    for bundle in manifest["bundles"]:
        lines.append(
            f"| {bundle['bundle_id']} | {bundle['scenario_type']} | "
            f"{bundle['train_count']} | {bundle['val_count']} | {', '.join(bundle['target_categories'])} |"
        )
    return "\n".join(lines).strip() + "\n"


def save_manifest(manifest: Dict[str, Any], output_dir: str | Path) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dpo_targeted_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "dpo_targeted_manifest.md").write_text(
        render_markdown(manifest),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build targeted failure-first DPO data.")
    parser.add_argument("--failure-plan", default=DEFAULT_FAILURE_PLAN, help="Failure plan JSON path")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Benchmark JSON path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    args = parser.parse_args()

    manifest = build_dataset(
        failure_plan_path=args.failure_plan,
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    save_manifest(manifest, args.output_dir)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
