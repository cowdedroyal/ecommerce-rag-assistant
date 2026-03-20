#!/usr/bin/env python3
"""
Export Phase 4 realistic-service DPO data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_focus_data import _write_jsonl
from src.data import load_product_data


DEFAULT_TAXONOMY = "outputs/preference_first_dpo_qwen25_3b_v2_realistic_service/dpo_preference_taxonomy.json"
DEFAULT_OUTPUT_DIR = "outputs/preference_first_dpo_qwen25_3b_v2_realistic_service"
DEFAULT_VAL_RATIO = 0.15

SKINCARE_CATEGORIES = {"护肤品"}

BUNDLE_RUNTIME_CONFIG: Dict[str, Dict[str, str]] = {
    "service_sales_persuasion": {
        "dimension": "service_conversion",
        "task": "sales_persuasion",
        "builder": "_build_service_sales_records",
    },
    "out_of_stock_alternative_redirect": {
        "dimension": "service_conversion",
        "task": "oos_redirect",
        "builder": "_build_oos_redirect_records",
    },
    "consultative_reassurance_sensitive_skin": {
        "dimension": "service_boundary",
        "task": "skin_consult",
        "builder": "_build_sensitive_skin_consult_records",
    },
    "price_negotiation_retention": {
        "dimension": "service_conversion",
        "task": "price_retention",
        "builder": "_build_price_negotiation_records",
    },
    "need_based_recommendation": {
        "dimension": "service_conversion",
        "task": "need_recommendation",
        "builder": "_build_need_based_recommendation_records",
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
                "Description": str(item.get("Description", "")).strip(),
                "Brand": str(item.get("Brand", "")).strip(),
                "Price": float(item["Price"]),
                "Rating": float(item["Rating"]),
                "features": str(item.get("features", "")).strip(),
            }
        )
    return rows


def _rows_by_category(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["Category"], []).append(row)
    for category, items in grouped.items():
        grouped[category] = sorted(items, key=lambda item: (-item["Rating"], item["Price"], item["Product_Title"]))
    return grouped


def _feature_lines(feature_text: str) -> List[str]:
    cleaned = str(feature_text or "").replace("、", ",").replace("，", ",")
    items = [part.strip(" -") for part in cleaned.split(",") if part.strip()]
    return items[:3] or ["暂无更多明确特点"]


def _scene_from_description(description: str) -> str:
    match = re.search(r"面向(.+?)场景设计", str(description))
    if match:
        return match.group(1).strip()
    return "日常使用"


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
        "source": "realistic_service_template_v2",
        "preference_only": False,
        "business_style": "realistic_service",
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
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = _base_metadata(
        row=row,
        bundle_id=bundle["bundle_id"],
        display_name=bundle["display_name"],
        prompt_family=prompt_family,
        dimension=dimension,
        task=task,
    )
    if extras:
        metadata.update(extras)
    return {
        "prompt": prompt,
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
        "dimension": dimension,
        "metadata": metadata,
    }


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


def _doc_block(row: Dict[str, Any], index: int = 1, extras: Optional[Dict[str, str]] = None) -> str:
    lines = [
        f"[{index}] {row['Product_Title']}",
        f"价格：¥{row['Price']:.2f}",
        f"评分：{row['Rating']:.1f}/5",
        f"品牌：{row['Brand'] or '未知'}",
        f"描述：{row['Description']}",
        f"特点：{'、'.join(_feature_lines(row['features']))}",
    ]
    for key, value in (extras or {}).items():
        if value:
            lines.append(f"{key}：{value}")
    return "\n".join(lines)


def _prompt_with_docs(doc_blocks: List[str], dialogue_lines: List[str]) -> str:
    return (
        "你是中文电商客服，请只基于以下商品资料回复，不要编造库存、优惠、疗效或售后承诺。\n\n"
        "商品资料：\n"
        f"{chr(10).join(doc_blocks)}\n\n"
        "对话记录：\n"
        f"{chr(10).join(dialogue_lines)}\n"
        "助手："
    )


def _feature_phrase(row: Dict[str, Any], limit: int = 2) -> str:
    return "、".join(_feature_lines(row["features"])[:limit])


def _need_prompts(row: Dict[str, Any]) -> List[Dict[str, str]]:
    scene = _scene_from_description(row["Description"])
    category = row["Category"]
    mapping: Dict[str, List[Dict[str, str]]] = {
        "蓝牙耳机": [
            {
                "family": "commute_switching",
                "user": "我每天地铁通勤，还要在手机和电脑之间切换开会，想要续航稳一点。您帮我挑一款蓝牙耳机。",
                "need_label": "通勤和设备切换",
            },
            {
                "family": "commute_reliable",
                "user": f"我主要是{scene}，也会偶尔开会，想买副省心一点的蓝牙耳机，您帮我选一款。",
                "need_label": f"{scene}和开会",
            },
        ],
        "手机": [
            {
                "family": "commute_phone",
                "user": "我平时通勤和出差都比较多，希望续航稳一点，拍照也别太拉。您帮我看一款更合适的手机。",
                "need_label": "通勤和出差",
            },
            {
                "family": "daily_phone",
                "user": f"我主要是{scene}，想买个日常用起来更省心的手机，您帮我挑一款。",
                "need_label": scene,
            },
        ],
        "笔记本电脑": [
            {
                "family": "portable_office",
                "user": "我经常背电脑上下班，希望轻一点、续航稳一点，您帮我挑一款更省心的笔记本。",
                "need_label": "通勤办公",
            },
            {
                "family": "study_portable",
                "user": f"我主要是{scene}，想买台轻便一点的笔记本，您帮我看哪款更合适。",
                "need_label": scene,
            },
        ],
        "智能手表": [
            {
                "family": "daily_watch",
                "user": "我想买块平时通勤看通知、周末也能记录运动的表，您帮我挑一款。",
                "need_label": "通勤和运动记录",
            },
            {
                "family": "watch_scene",
                "user": f"我主要是{scene}，想要一块日常看着省心的智能手表，您帮我推荐一款。",
                "need_label": scene,
            },
        ],
        "运动鞋": [
            {
                "family": "walk_run",
                "user": "我平时通勤走路多，周末偶尔慢跑，想买双脚感稳一点的鞋，您帮我挑一款。",
                "need_label": "通勤走路和慢跑",
            },
            {
                "family": "shoe_scene",
                "user": f"我主要是{scene}，想买双日常更舒服一点的鞋，您帮我推荐一款。",
                "need_label": scene,
            },
        ],
        "护肤品": [
            {
                "family": "daily_skincare",
                "user": "我最近熬夜加班，皮肤有点干，想买套日常保湿修护的，您帮我挑一个更合适的。",
                "need_label": "日常保湿修护",
            },
            {
                "family": "skincare_scene",
                "user": f"我主要是{scene}，想找个用起来更稳一点的护肤品，您帮我推荐一款。",
                "need_label": scene,
            },
        ],
    }
    return mapping.get(
        category,
        [
            {
                "family": "generic_need",
                "user": f"我主要是{scene}，想买个更适合这个场景的商品，您帮我推荐一款。",
                "need_label": scene,
            }
        ],
    )


def _sales_prompts(row: Dict[str, Any]) -> List[Dict[str, str]]:
    scene = _scene_from_description(row["Description"])
    category = row["Category"]
    mapping: Dict[str, List[Dict[str, str]]] = {
        "护肤品": [
            {
                "family": "skincare_sales",
                "user": f"我最近熬夜加班，皮肤有点干，想买一套日常保湿修护的。您帮我看看 {row['Product_Title']} 适不适合我。",
                "need_label": "日常保湿修护",
            },
            {
                "family": "skincare_scene_sales",
                "user": f"我主要是{scene}，想买个用起来稳一点的护肤品。您看 {row['Product_Title']} 值得我考虑吗？",
                "need_label": scene,
            },
        ],
        "蓝牙耳机": [
            {
                "family": "earbuds_sales",
                "user": f"我平时{scene}，还会偶尔开会。您看 {row['Product_Title']} 这款适不适合我？",
                "need_label": scene,
            },
            {
                "family": "earbuds_pitch",
                "user": f"我想买副更省心一点的蓝牙耳机。您别只报参数，像客服一样告诉我为什么 {row['Product_Title']} 值得我考虑。",
                "need_label": "日常使用",
            },
        ],
        "手机": [
            {
                "family": "phone_sales",
                "user": f"我主要是{scene}，想买个日常用起来稳一点的手机。您帮我看看 {row['Product_Title']} 能不能重点考虑。",
                "need_label": scene,
            },
            {
                "family": "phone_pitch",
                "user": f"我最近想换手机，您别只念配置，像客服一样告诉我为什么 {row['Product_Title']} 值得我考虑。",
                "need_label": "换机",
            },
        ],
    }
    return mapping.get(category, _need_prompts(row))


def _price_prompts(row: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {
            "family": "bargain_50",
            "user": f"{row['Product_Title']} 这个价格还是有点高，再便宜 50 我就下单。",
        },
        {
            "family": "bargain_generic",
            "user": f"{row['Product_Title']} 不能再优惠一点吗？我今天想下单。",
        },
    ]


def _choose_alternatives(
    row: Dict[str, Any],
    rows_by_category: Dict[str, List[Dict[str, Any]]],
    count: int = 2,
) -> List[Dict[str, Any]]:
    candidates = [
        item
        for item in rows_by_category.get(row["Category"], [])
        if item["Product_ID"] != row["Product_ID"]
    ]
    candidates = sorted(
        candidates,
        key=lambda item: (
            abs(item["Price"] - row["Price"]),
            abs(item["Rating"] - row["Rating"]),
            item["Product_Title"],
        ),
    )
    chosen: List[Dict[str, Any]] = []
    seen_titles = {row["Product_Title"]}
    for item in candidates:
        if item["Product_Title"] in seen_titles:
            continue
        chosen.append(item)
        seen_titles.add(item["Product_Title"])
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        for item in candidates:
            if item in chosen:
                continue
            chosen.append(item)
            if len(chosen) >= count:
                break
    return chosen[:count]


def _service_sales_chosen(row: Dict[str, Any], need_label: str) -> str:
    return (
        f"您好，按您这个{need_label}的需求，我会更推荐您重点看看 {row['Product_Title']}。\n\n"
        f"它现在价格是 ¥{row['Price']:.2f}，评分 {row['Rating']:.1f}/5，"
        f"而且有 {_feature_phrase(row)} 这些已知卖点，对您这种场景会更友好，"
        "日常用起来也会更省心，整体是比较值得您考虑的一款。"
    )


def _service_sales_rejected(row: Dict[str, Any]) -> str:
    return (
        f"{row['Product_Title']}，价格 ¥{row['Price']:.2f}，评分 {row['Rating']:.1f}/5，"
        f"特点是 {_feature_phrase(row, limit=3)}。这些都是商品资料里写的，您自己看一下。"
    )


def _oos_redirect_chosen(row: Dict[str, Any], alternatives: List[Dict[str, Any]]) -> str:
    alt1 = alternatives[0]
    alt1_reason = _feature_phrase(alt1)
    lines = [
        f"抱歉，{row['Product_Title']} 这边目前暂时无现货，今天没法直接给您承诺发出。",
        (
            f"您如果着急用，可以先看看 {alt1['Product_Title']}，它现在 ¥{alt1['Price']:.2f}，"
            f"评分 {alt1['Rating']:.1f}/5，而且有 {alt1_reason} 这些特点，作为替代款会更稳一点。"
        ),
    ]
    if len(alternatives) > 1:
        alt2 = alternatives[1]
        lines.append(
            f"如果您还想再多看一个同类选择，{alt2['Product_Title']} 也可以一起比较，"
            f"它的已知特点是 {_feature_phrase(alt2)}。"
        )
    return "\n\n".join(lines)


def _oos_redirect_rejected(row: Dict[str, Any]) -> str:
    return f"抱歉，{row['Product_Title']} 这款暂时没货，补货时间我这边也不确定，您可以晚点再看看。"


def _consult_followup_chosen(row: Dict[str, Any]) -> str:
    return (
        f"可以先帮您判断，不过为了更稳妥一点，我想先确认一下：您是敏感肌，"
        f"还是只是换季时会偶尔泛红、发干？确认这个情况后，我再结合 {row['Product_Title']} 的资料给您更准确的建议。"
    )


def _consult_followup_variants(row: Dict[str, Any]) -> List[str]:
    return [
        (
            f"可以先帮您判断，不过为了更稳妥一点，我想先确认一下：您是敏感肌，"
            f"还是只是换季时会偶尔泛红、发干？确认这个情况后，我再结合 {row['Product_Title']} 的资料给您更准确的建议。"
        ),
        (
            f"我先按资料帮您看，但在正式建议前还想确认一个点：您本身是敏感肌，"
            f"还是只是最近换季会偶尔干、偶尔泛红？这个信息会直接影响我对 {row['Product_Title']} 的判断。"
        ),
    ]


def _consult_followup_rejected() -> str:
    return "能用的，这款本来就敏感肌可用，您直接正常用就可以。"


def _consult_branch_chosen(row: Dict[str, Any], branch: str) -> str:
    if branch == "not_sensitive":
        return (
            f"如果您不是敏感肌，按您刚才说的这个情况，一般风险会相对低一些。\n\n"
            f"{row['Product_Title']} 现在价格 ¥{row['Price']:.2f}，评分 {row['Rating']:.1f}/5，"
            f"已知有 {_feature_phrase(row)} 这些特点，偏日常保湿修护方向；"
            "不过第一次用还是建议您先做局部试用，再看皮肤反应，会更稳妥。"
        )
    return (
        f"如果您本身是敏感肌，我会建议您更谨慎一点。\n\n"
        f"{row['Product_Title']} 的已知资料里有 {_feature_phrase(row)} 这些信息，"
        "但我不能替您保证绝对不过敏；更稳妥的做法还是先在耳后或下颌线做局部试用，"
        "没有明显刺激再正常上脸。"
    )


def _consult_branch_rejected(branch: str) -> str:
    if branch == "not_sensitive":
        return "既然不是敏感肌，那就绝对不会过敏，您放心用就行。"
    return "那也没事，这款写了敏感肌可用，您直接正常上脸就可以。"


def _price_retention_chosen(row: Dict[str, Any]) -> str:
    return (
        f"您好，这款现在已经是活动到手价了，价格这边确实没有再降空间。\n\n"
        f"不过它目前是 ¥{row['Price']:.2f}，评分 {row['Rating']:.1f}/5，"
        f"而且有 {_feature_phrase(row)} 这些已知卖点，日常用起来会更省心；"
        "如果您今天准备入手，这个价位整体还是比较合适的。"
    )


def _price_retention_rejected(row: Dict[str, Any]) -> str:
    return f"{row['Product_Title']} 这个价不能再少了，想买就直接下单，不买的话我这边也没办法。"


def _need_based_chosen(row: Dict[str, Any], need_label: str) -> str:
    return (
        f"按您这个{need_label}的需求，我会更推荐您选 {row['Product_Title']}。\n\n"
        f"它现在价格 ¥{row['Price']:.2f}，评分 {row['Rating']:.1f}/5，"
        f"而且有 {_feature_phrase(row)} 这些已知特点，对您这个使用场景会更贴合。"
    )


def _need_based_rejected(row: Dict[str, Any], alternatives: List[Dict[str, Any]]) -> str:
    alt_titles = "、".join(item["Product_Title"] for item in alternatives[:2])
    return (
        f"这几款都差不多，像 {row['Product_Title']} 和 {alt_titles} 都可以。"
        f"{row['Product_Title']} 现在是 ¥{row['Price']:.2f}，评分 {row['Rating']:.1f}/5，"
        "参数您自己对比一下再决定。"
    )


def _resolve_builder(name: str) -> Callable[..., List[Dict[str, Any]]]:
    return globals()[name]


def _build_service_sales_records(
    row: Dict[str, Any],
    bundle: Dict[str, Any],
    dimension: str,
    task: str,
    rows_by_category: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    del rows_by_category
    records: List[Dict[str, Any]] = []
    for spec in _sales_prompts(row):
        prompt = _prompt_with_docs(
            doc_blocks=[_doc_block(row, extras={"客服备注": "请像真实客服一样礼貌推荐，不要只复读详情页。"})],
            dialogue_lines=[f"用户：{spec['user']}"],
        )
        records.append(
            _build_record(
                row=row,
                bundle=bundle,
                dimension=dimension,
                task=task,
                prompt_family=spec["family"],
                prompt=prompt,
                chosen=_service_sales_chosen(row, spec["need_label"]),
                rejected=_service_sales_rejected(row),
                extras={"service_family": spec["family"]},
            )
        )
    return records


def _build_oos_redirect_records(
    row: Dict[str, Any],
    bundle: Dict[str, Any],
    dimension: str,
    task: str,
    rows_by_category: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    alternatives = _choose_alternatives(row, rows_by_category, count=2)
    if len(alternatives) < 2:
        return []
    prompts = [
        {
            "family": "ship_today_redirect",
            "user": f"我就想要 {row['Product_Title']} 这个型号，今天能发吗？如果没货的话，您给我找个差不多的也行。",
        },
        {
            "family": "oos_find_similar",
            "user": f"这款 {row['Product_Title']} 要是没有现货，您直接帮我换个差不多的，不要让我自己慢慢找。",
        },
    ]
    doc_blocks = [
        _doc_block(
            row,
            index=1,
            extras={
                "库存备注": "当前仓库暂时无现货，今天无法承诺发出。",
                "客服备注": "需要先说明缺货，再推荐相似替代款。",
            },
        )
    ]
    for idx, alt in enumerate(alternatives, start=2):
        doc_blocks.append(
            _doc_block(
                alt,
                index=idx,
                extras={
                    "库存备注": "可正常下单",
                    "客服备注": "可作为当前缺货款的替代商品。",
                },
            )
        )

    records: List[Dict[str, Any]] = []
    for spec in prompts:
        prompt = _prompt_with_docs(doc_blocks=doc_blocks, dialogue_lines=[f"用户：{spec['user']}"])
        records.append(
            _build_record(
                row=row,
                bundle=bundle,
                dimension=dimension,
                task=task,
                prompt_family=spec["family"],
                prompt=prompt,
                chosen=_oos_redirect_chosen(row, alternatives),
                rejected=_oos_redirect_rejected(row),
                extras={"alternative_product_ids": [item["Product_ID"] for item in alternatives]},
            )
        )
    return records


def _build_sensitive_skin_consult_records(
    row: Dict[str, Any],
    bundle: Dict[str, Any],
    dimension: str,
    task: str,
    rows_by_category: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    del rows_by_category
    if row["Category"] not in SKINCARE_CATEGORIES:
        return []

    doc_blocks = [
        _doc_block(
            row,
            extras={
                "安全备注": "已知信息只支持温和/敏感肌可用倾向，不支持绝对不过敏；首次使用建议局部试用。",
                "客服备注": "先问关键肤况，再根据用户回复给出稳妥建议。",
            },
        )
    ]
    first_user_variants = [
        f"我皮肤容易泛红，这款 {row['Product_Title']} 能用吗？",
        f"我一换季就容易干和泛红，{row['Product_Title']} 这款适合我吗？",
    ]
    followup_variants = _consult_followup_variants(row)

    records: List[Dict[str, Any]] = []
    for idx, first_user in enumerate(first_user_variants, start=1):
        followup_prompt = _prompt_with_docs(doc_blocks=doc_blocks, dialogue_lines=[f"用户：{first_user}"])
        records.append(
            _build_record(
                row=row,
                bundle=bundle,
                dimension=dimension,
                task=task,
                prompt_family=f"consult_followup_question_v{idx}",
                prompt=followup_prompt,
                chosen=followup_variants[(idx - 1) % len(followup_variants)],
                rejected=_consult_followup_rejected(),
                extras={"consult_step": "followup_question"},
            )
        )

    branch_specs = [
        {
            "family": "consult_branch_not_sensitive",
            "user_reply": "不是敏感肌，就是换季会偶尔干、偶尔泛红。",
            "branch": "not_sensitive",
        },
        {
            "family": "consult_branch_not_sensitive_alt",
            "user_reply": "不是敏感肌，只是最近熬夜有点干，偶尔会泛红。",
            "branch": "not_sensitive",
        },
        {
            "family": "consult_branch_sensitive",
            "user_reply": "是敏感肌，以前换新产品容易发红刺痛。",
            "branch": "sensitive",
        },
        {
            "family": "consult_branch_sensitive_alt",
            "user_reply": "算敏感肌，之前用刺激一点的产品会泛红发痒。",
            "branch": "sensitive",
        },
    ]

    for first_idx, first_user in enumerate(first_user_variants, start=1):
        for followup_idx, assistant_followup in enumerate(followup_variants, start=1):
            for spec in branch_specs:
                prompt = _prompt_with_docs(
                    doc_blocks=doc_blocks,
                    dialogue_lines=[
                        f"用户：{first_user}",
                        f"助手：{assistant_followup}",
                        f"用户：{spec['user_reply']}",
                    ],
                )
                records.append(
                    _build_record(
                        row=row,
                        bundle=bundle,
                        dimension=dimension,
                        task=task,
                        prompt_family=f"{spec['family']}_q{first_idx}_a{followup_idx}",
                        prompt=prompt,
                        chosen=_consult_branch_chosen(row, branch=spec["branch"]),
                        rejected=_consult_branch_rejected(branch=spec["branch"]),
                        extras={"consult_step": "branch_answer", "consult_branch": spec["branch"]},
                    )
                )
    return records


def _build_price_negotiation_records(
    row: Dict[str, Any],
    bundle: Dict[str, Any],
    dimension: str,
    task: str,
    rows_by_category: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    del rows_by_category
    doc_blocks = [
        _doc_block(
            row,
            extras={
                "价格备注": "当前展示价已包含店铺活动优惠，暂无额外折扣信息。",
                "客服备注": "需要守住价格边界，但尽量留住成交。",
            },
        )
    ]
    records: List[Dict[str, Any]] = []
    for spec in _price_prompts(row):
        prompt = _prompt_with_docs(doc_blocks=doc_blocks, dialogue_lines=[f"用户：{spec['user']}"])
        records.append(
            _build_record(
                row=row,
                bundle=bundle,
                dimension=dimension,
                task=task,
                prompt_family=spec["family"],
                prompt=prompt,
                chosen=_price_retention_chosen(row),
                rejected=_price_retention_rejected(row),
                extras={"negotiation_family": spec["family"]},
            )
        )
    return records


def _build_need_based_recommendation_records(
    row: Dict[str, Any],
    bundle: Dict[str, Any],
    dimension: str,
    task: str,
    rows_by_category: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    alternatives = _choose_alternatives(row, rows_by_category, count=2)
    if len(alternatives) < 2:
        return []
    doc_blocks = [_doc_block(row, index=1)]
    for idx, alt in enumerate(alternatives, start=2):
        doc_blocks.append(_doc_block(alt, index=idx))

    records: List[Dict[str, Any]] = []
    for spec in _need_prompts(row):
        prompt = _prompt_with_docs(doc_blocks=doc_blocks, dialogue_lines=[f"用户：{spec['user']}"])
        records.append(
            _build_record(
                row=row,
                bundle=bundle,
                dimension=dimension,
                task=task,
                prompt_family=spec["family"],
                prompt=prompt,
                chosen=_need_based_chosen(row, spec["need_label"]),
                rejected=_need_based_rejected(row, alternatives),
                extras={"alternative_product_ids": [item["Product_ID"] for item in alternatives]},
            )
        )
    return records


def build_realistic_service_export(
    taxonomy_path: str | Path,
    output_dir: str | Path,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> Dict[str, Any]:
    taxonomy = _load_json(taxonomy_path)
    bundles = taxonomy.get("bundles", [])
    product_rows = _product_rows()
    grouped_rows = _rows_by_category(product_rows)
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
                    rows_by_category=grouped_rows,
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
        "phase": "phase_4b_v2_realistic_service_dpo_data",
        "taxonomy_path": str(taxonomy_path),
        "output_dir": str(output_root),
        "generation_mode": "realistic_service_template_with_embedded_evidence",
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
            "judge_strategy": taxonomy.get("benchmark_principles", {}).get("judge_strategy", {}),
            "note": "这版 prompt 内嵌商品证据和业务备注，chosen/rejected 对齐真实客服促单、转品、议价和多轮咨询场景。",
        },
        "bundles": bundle_summaries,
    }

    manifest_json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_md_path.write_text(render_markdown(manifest), encoding="utf-8")
    return manifest


def render_markdown(manifest: Dict[str, Any]) -> str:
    judge_strategy = manifest["policy"].get("judge_strategy", {})
    lines = [
        "# Realistic-Service DPO 数据导出",
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
        f"- stronger judge：{'是' if judge_strategy.get('need_stronger_judge') else '否'}",
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
            preview = prompt.replace("\n", " ").strip()
            lines.append(f"- train: {preview[:180]}{'...' if len(preview) > 180 else ''}")
        for prompt in item["preview_prompts"]["val"]:
            preview = prompt.replace("\n", " ").strip()
            lines.append(f"- val: {preview[:180]}{'...' if len(preview) > 180 else ''}")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export realistic-service DPO data")
    parser.add_argument("--taxonomy", default=DEFAULT_TAXONOMY, help="realistic-service taxonomy JSON path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="output directory")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="validation ratio by product")
    args = parser.parse_args()

    build_realistic_service_export(
        taxonomy_path=args.taxonomy,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
