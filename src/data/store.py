from __future__ import annotations

import json
import logging
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config import Settings

logger = logging.getLogger(__name__)

PRODUCT_ALIASES = {
    "title": "Product_Title",
    "average_rating": "Rating",
    "description": "Description",
    "price": "Price",
    "parent_asin": "Product_ID",
    "main_category": "Category",
    "rating_number": "Rating_Count",
    "brand": "Brand",
    "store": "Store",
    "image_url": "Image",
}

ORDER_ALIASES = {
    "payment_method": "Payment_Method",
    "Payment_method": "Payment_Method",
    "customer_id": "Customer_Id",
    "product_id": "Product_ID",
    "product_title": "Product",
}

DEFAULT_PRODUCT_COLUMNS = {
    "Product_ID": "",
    "Product_Title": "",
    "Category": "",
    "Description": "",
    "Brand": "",
    "Store": "",
    "Price": 0.0,
    "Rating": 0.0,
    "Rating_Count": 0,
    "features": "",
    "source": "",
    "Image": "",
}

DEFAULT_ORDER_COLUMNS = {
    "Order_ID": "",
    "Customer_Id": 0,
    "Product_ID": "",
    "Product": "",
    "Product_Category": "",
    "Sales": 0.0,
    "Quantity": 1,
    "Discount": 0.0,
    "Profit": 0.0,
    "Shipping_Cost": 0.0,
    "Order_Priority": "Medium",
    "Shipping_Status": "已签收",
    "Payment_Method": "Alipay",
    "Order_Date": "",
    "Time": "",
    "Order_DateTime": "",
    "source": "",
}

SYNTHETIC_BLUEPRINTS: List[Dict[str, Any]] = [
    {
        "category": "蓝牙耳机",
        "brands": ["EchoWave", "晨声", "SoundPeak", "Auralink"],
        "series": ["降噪版", "旗舰版", "通勤版", "运动版"],
        "features": ["主动降噪", "40小时续航", "双设备连接", "低延迟游戏模式", "IPX5 防汗", "空间音频"],
        "scenes": ["通勤地铁", "健身跑步", "视频会议", "宿舍追剧"],
        "price_range": (159, 699),
    },
    {
        "category": "手机",
        "brands": ["Nova", "星极", "讯曜", "Pixeline"],
        "series": ["Pro", "Air", "Max", "青春版"],
        "features": ["120Hz 高刷屏", "5000mAh 电池", "光学防抖主摄", "80W 快充", "轻薄机身", "AI 夜景"],
        "scenes": ["手游娱乐", "出差办公", "Vlog 拍摄", "日常通勤"],
        "price_range": (1899, 5999),
    },
    {
        "category": "笔记本电脑",
        "brands": ["NovaBook", "云岳", "ThinkAir", "Aster"],
        "series": ["14", "Pro 16", "Slim", "Creator"],
        "features": ["2.8K 高色域屏", "长续航 12 小时", "32GB 内存", "轻薄金属机身", "双风扇散热", "指纹解锁"],
        "scenes": ["学生上课", "远程办公", "内容创作", "差旅携带"],
        "price_range": (3999, 9999),
    },
    {
        "category": "护肤品",
        "brands": ["澄研", "LumiSkin", "植序", "润澈"],
        "series": ["修护系列", "焕亮系列", "清润系列", "舒缓系列"],
        "features": ["神经酰胺修护", "烟酰胺提亮", "清爽不黏腻", "敏感肌可用", "玻尿酸保湿", "无酒精配方"],
        "scenes": ["熬夜急救", "换季维稳", "送礼自用", "通勤淡妆前打底"],
        "price_range": (89, 499),
    },
    {
        "category": "运动鞋",
        "brands": ["StrideLab", "跃野", "RunArc", "AeroStep"],
        "series": ["轻弹版", "越野版", "竞速版", "缓震版"],
        "features": ["轻量缓震", "透气网布", "抓地大底", "足弓支撑", "回弹中底", "长距离友好"],
        "scenes": ["晨跑", "健身房训练", "通勤穿搭", "户外轻越野"],
        "price_range": (239, 899),
    },
    {
        "category": "机械键盘",
        "brands": ["KeyNova", "霓栈", "TypeCraft", "GridLab"],
        "series": ["87 键", "98 键", "客制化版", "办公静音版"],
        "features": ["热插拔", "PBT 键帽", "三模连接", "Gasket 结构", "RGB 灯效", "静音轴体"],
        "scenes": ["程序开发", "深夜宿舍", "游戏开黑", "桌搭升级"],
        "price_range": (199, 899),
    },
    {
        "category": "智能手表",
        "brands": ["PulseGo", "星环", "FitNex", "Chrono"],
        "series": ["Pro", "Mini", "Active", "Classic"],
        "features": ["血氧监测", "双频 GPS", "7 天续航", "50 米防水", "睡眠分析", "运动课程"],
        "scenes": ["跑步训练", "健康监测", "商务通勤", "周末骑行"],
        "price_range": (299, 1999),
    },
    {
        "category": "咖啡机",
        "brands": ["BrewMori", "豆研", "CremaOne", "Barista+"],
        "series": ["家用版", "Mini", "Pro", "全自动版"],
        "features": ["20bar 萃取", "一键奶泡", "可拆洗水箱", "预浸泡", "温度稳定", "桌面小巧"],
        "scenes": ["晨起提神", "办公室共享", "居家咖啡角", "朋友聚会"],
        "price_range": (399, 2599),
    },
]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_product_search_text(row: pd.Series) -> str:
    """构建商品检索文本。"""
    fields = [
        "Product_Title",
        "Description",
        "Category",
        "Brand",
        "Store",
        "features",
        "Selling_Points",
    ]
    parts = []
    for field in fields:
        value = row.get(field, "")
        if value:
            parts.append(str(value).strip())
    return " ".join(parts)


def normalize_product_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """统一商品字段结构，兼容真实/合成/历史数据。"""
    df = df.copy()
    df = df.rename(columns={key: value for key, value in PRODUCT_ALIASES.items() if key in df.columns})
    df = df.fillna("")

    for column, default in DEFAULT_PRODUCT_COLUMNS.items():
        if column not in df.columns:
            df[column] = default

    for column in ["Product_ID", "Product_Title", "Category", "Description", "Brand", "Store", "features", "source", "Image"]:
        df[column] = df[column].astype(str).fillna("").str.strip()

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(0.0)
    df["Rating_Count"] = pd.to_numeric(df["Rating_Count"], errors="coerce").fillna(0).astype(int)

    missing_id_mask = df["Product_ID"] == ""
    if missing_id_mask.any():
        df.loc[missing_id_mask, "Product_ID"] = [
            f"GEN-{idx:05d}" for idx in range(1, missing_id_mask.sum() + 1)
        ]

    missing_title_mask = df["Product_Title"] == ""
    if missing_title_mask.any():
        df.loc[missing_title_mask, "Product_Title"] = "未命名商品"

    missing_desc_mask = df["Description"] == ""
    if missing_desc_mask.any():
        df.loc[missing_desc_mask, "Description"] = (
            df.loc[missing_desc_mask, "Brand"].replace("", "精选")
            + " "
            + df.loc[missing_desc_mask, "Category"].replace("", "商品")
            + "，适合日常选购场景。"
        )

    missing_source_mask = df["source"] == ""
    if missing_source_mask.any():
        df.loc[missing_source_mask, "source"] = "synthetic-demo"

    df = df[df["Price"] > 0].copy()
    df["Rating"] = df["Rating"].clip(lower=0, upper=5)
    df["combined_text"] = df.apply(build_product_search_text, axis=1)

    return df.reset_index(drop=True)


def normalize_order_dataframe(df: pd.DataFrame, product_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """统一订单字段结构。"""
    df = df.copy()
    df = df.rename(columns={key: value for key, value in ORDER_ALIASES.items() if key in df.columns})
    df = df.fillna("")

    for column, default in DEFAULT_ORDER_COLUMNS.items():
        if column not in df.columns:
            df[column] = default

    for column in ["Order_ID", "Product_ID", "Product", "Product_Category", "Order_Priority", "Shipping_Status", "Payment_Method", "Order_Date", "Time", "Order_DateTime", "source"]:
        df[column] = df[column].astype(str).fillna("").str.strip()

    df["Customer_Id"] = pd.to_numeric(df["Customer_Id"], errors="coerce").fillna(0).astype(int)
    for column in ["Sales", "Quantity", "Discount", "Profit", "Shipping_Cost"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    if product_df is not None and not product_df.empty:
        product_map = product_df.set_index("Product_ID")
        missing_title_mask = df["Product"] == ""
        if missing_title_mask.any():
            df.loc[missing_title_mask, "Product"] = df.loc[missing_title_mask, "Product_ID"].map(product_map["Product_Title"]).fillna("")
        missing_category_mask = df["Product_Category"] == ""
        if missing_category_mask.any():
            df.loc[missing_category_mask, "Product_Category"] = df.loc[missing_category_mask, "Product_ID"].map(product_map["Category"]).fillna("")

    if "Order_DateTime" not in df.columns or (df["Order_DateTime"] == "").all():
        if "Order_Date" in df.columns and "Time" in df.columns:
            df["Order_DateTime"] = (df["Order_Date"].astype(str) + " " + df["Time"].astype(str)).str.strip()

    parsed_datetime = pd.to_datetime(df["Order_DateTime"], errors="coerce")
    missing_dt_mask = parsed_datetime.isna()
    if missing_dt_mask.any():
        fallback_now = datetime.now()
        fallback_values = [
            fallback_now - timedelta(hours=idx * 6) for idx in range(1, missing_dt_mask.sum() + 1)
        ]
        parsed_datetime.loc[missing_dt_mask] = fallback_values

    df["Order_DateTime"] = parsed_datetime.dt.strftime("%Y-%m-%d %H:%M:%S")
    blank_order_date = df["Order_Date"] == ""
    if blank_order_date.any():
        df.loc[blank_order_date, "Order_Date"] = parsed_datetime.dt.strftime("%Y-%m-%d")
    blank_time = df["Time"] == ""
    if blank_time.any():
        df.loc[blank_time, "Time"] = parsed_datetime.dt.strftime("%H:%M:%S")

    missing_source_mask = df["source"] == ""
    if missing_source_mask.any():
        df.loc[missing_source_mask, "source"] = "synthetic-demo"

    missing_order_id_mask = df["Order_ID"] == ""
    if missing_order_id_mask.any():
        df.loc[missing_order_id_mask, "Order_ID"] = [
            f"ORD-GEN-{idx:05d}" for idx in range(1, missing_order_id_mask.sum() + 1)
        ]

    return df.sort_values("Order_DateTime", ascending=False).reset_index(drop=True)


def _generate_synthetic_products(
    seed: int,
    products_per_category: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    products: List[Dict[str, Any]] = []

    for blueprint in SYNTHETIC_BLUEPRINTS:
        category = blueprint["category"]
        price_low, price_high = blueprint["price_range"]
        for index in range(1, products_per_category + 1):
            brand = rng.choice(blueprint["brands"])
            series = rng.choice(blueprint["series"])
            scene = rng.choice(blueprint["scenes"])
            feature_pool = rng.sample(blueprint["features"], k=3)
            price = round(rng.uniform(price_low, price_high), 2)
            rating = round(min(5.0, max(4.0, rng.uniform(4.1, 4.9))), 1)
            rating_count = rng.randint(180, 4200)
            product_id = f"SYN-{category[:2]}-{index:03d}"
            title = f"{brand} {category} {series}"
            description = (
                f"{title}，面向{scene}场景设计，主打{feature_pool[0]}、{feature_pool[1]}和{feature_pool[2]}。"
                f"适合希望在预算内获得稳定体验的用户。"
            )
            products.append(
                {
                    "Product_ID": product_id,
                    "Product_Title": title,
                    "Category": category,
                    "Description": description,
                    "Brand": brand,
                    "Store": f"{brand} 官方旗舰店",
                    "Price": price,
                    "Rating": rating,
                    "Rating_Count": rating_count,
                    "features": "、".join(feature_pool),
                    "source": "synthetic-demo",
                    "Image": "",
                }
            )

    return pd.DataFrame(products)


def _generate_synthetic_orders(
    product_df: pd.DataFrame,
    seed: int,
    order_count: int,
) -> pd.DataFrame:
    rng = random.Random(seed + 1)
    customer_ids = [10001 + idx for idx in range(64)]
    priorities = ["Low", "Medium", "High"]
    payment_methods = ["Alipay", "WeChat Pay", "Credit Card"]
    shipping_statuses = ["已签收", "运输中", "待发货", "退款中"]
    order_rows: List[Dict[str, Any]] = []
    base_time = datetime.now()

    products = product_df.to_dict("records")
    for index in range(1, order_count + 1):
        product = rng.choice(products)
        quantity = rng.randint(1, 3)
        discount = round(rng.choice([0, 0.05, 0.1, 0.15]), 2)
        gross_sales = float(product["Price"]) * quantity
        sales = round(gross_sales * (1 - discount), 2)
        shipping_cost = round(rng.uniform(0, 25), 2)
        profit = round(max(sales * rng.uniform(0.08, 0.26), 0), 2)
        order_time = base_time - timedelta(hours=index * rng.randint(2, 8))

        order_rows.append(
            {
                "Order_ID": f"ORD-SYN-{index:05d}",
                "Customer_Id": rng.choice(customer_ids),
                "Product_ID": product["Product_ID"],
                "Product": product["Product_Title"],
                "Product_Category": product["Category"],
                "Sales": sales,
                "Quantity": quantity,
                "Discount": discount,
                "Profit": profit,
                "Shipping_Cost": shipping_cost,
                "Order_Priority": rng.choices(priorities, weights=[2, 5, 3], k=1)[0],
                "Shipping_Status": rng.choices(shipping_statuses, weights=[6, 2, 1, 1], k=1)[0],
                "Payment_Method": rng.choice(payment_methods),
                "Order_Date": order_time.strftime("%Y-%m-%d"),
                "Time": order_time.strftime("%H:%M:%S"),
                "Order_DateTime": order_time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "synthetic-demo",
            }
        )

    return pd.DataFrame(order_rows)


def build_catalog_summary(product_df: pd.DataFrame, order_df: pd.DataFrame) -> Dict[str, Any]:
    """构建商品目录和订单概览，用于 API 与前端展示。"""
    top_categories = (
        product_df["Category"].value_counts().head(6).rename_axis("name").reset_index(name="count")
    )
    top_brands = (
        product_df["Brand"].replace("", "未标注").value_counts().head(6).rename_axis("name").reset_index(name="count")
    )
    sources = Counter(product_df["source"].tolist())
    dataset_mode = "synthetic-demo" if set(sources.keys()) == {"synthetic-demo"} else "mixed"

    return {
        "dataset_mode": dataset_mode,
        "product_count": int(len(product_df)),
        "order_count": int(len(order_df)),
        "category_count": int(product_df["Category"].nunique()),
        "brand_count": int(product_df["Brand"].replace("", pd.NA).dropna().nunique()),
        "avg_rating": round(float(product_df["Rating"].mean() or 0), 2),
        "avg_price": round(float(product_df["Price"].mean() or 0), 2),
        "price_range": {
            "min": round(float(product_df["Price"].min() or 0), 2),
            "max": round(float(product_df["Price"].max() or 0), 2),
        },
        "top_categories": top_categories.to_dict("records"),
        "top_brands": top_brands.to_dict("records"),
        "source_distribution": [{"name": name, "count": count} for name, count in sources.items()],
        "high_priority_orders": int(
            (order_df["Order_Priority"].astype(str).str.lower() == "high").sum()
        ),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def ensure_runtime_data(
    settings: Optional[Settings] = None,
    force: bool = False,
    products_per_category: Optional[int] = None,
    order_count: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """确保运行时商品与订单数据存在，不存在时自动生成高质量合成数据。"""
    settings = settings or Settings()
    product_path = Path(settings.PRODUCT_DATA_PATH)
    order_path = Path(settings.ORDER_DATA_PATH)
    summary_path = Path(settings.CATALOG_SUMMARY_PATH)

    if not force and product_path.exists() and order_path.exists():
        return {
            "product_path": product_path,
            "order_path": order_path,
            "summary_path": summary_path,
            "generated": False,
        }

    _ensure_parent(product_path)
    _ensure_parent(order_path)
    _ensure_parent(summary_path)

    seed = settings.SYNTHETIC_DATA_SEED if seed is None else seed
    products_per_category = (
        settings.SYNTHETIC_PRODUCTS_PER_CATEGORY
        if products_per_category is None
        else products_per_category
    )
    order_count = settings.SYNTHETIC_ORDER_COUNT if order_count is None else order_count

    product_df = normalize_product_dataframe(
        _generate_synthetic_products(seed=seed, products_per_category=products_per_category)
    )
    order_df = normalize_order_dataframe(
        _generate_synthetic_orders(product_df=product_df, seed=seed, order_count=order_count),
        product_df=product_df,
    )
    summary = build_catalog_summary(product_df, order_df)

    product_df.to_csv(product_path, index=False)
    order_df.to_csv(order_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "Synthetic demo data generated: %s products, %s orders",
        len(product_df),
        len(order_df),
    )
    return {
        "product_path": product_path,
        "order_path": order_path,
        "summary_path": summary_path,
        "generated": True,
        "summary": summary,
    }


def _resolve_paths(
    product_path: Optional[str | Path],
    order_path: Optional[str | Path],
    settings: Settings,
) -> Tuple[Path, Path]:
    default_product_path = Path(settings.PRODUCT_DATA_PATH)
    default_order_path = Path(settings.ORDER_DATA_PATH)

    resolved_product_path = Path(product_path) if product_path else default_product_path
    resolved_order_path = Path(order_path) if order_path else default_order_path

    if not resolved_product_path.exists() or not resolved_order_path.exists():
        if resolved_product_path == default_product_path or resolved_order_path == default_order_path:
            ensure_runtime_data(settings=settings)

    return resolved_product_path, resolved_order_path


def load_runtime_data(
    product_path: Optional[str | Path] = None,
    order_path: Optional[str | Path] = None,
    settings: Optional[Settings] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载运行时商品与订单数据。"""
    settings = settings or Settings()
    resolved_product_path, resolved_order_path = _resolve_paths(product_path, order_path, settings)

    product_df = normalize_product_dataframe(pd.read_csv(resolved_product_path))
    order_df = normalize_order_dataframe(pd.read_csv(resolved_order_path), product_df=product_df)
    return product_df, order_df


def load_product_data(
    product_path: Optional[str | Path] = None,
    settings: Optional[Settings] = None,
) -> pd.DataFrame:
    """仅加载商品数据。"""
    product_df, _ = load_runtime_data(product_path=product_path, settings=settings)
    return product_df


def load_order_data(
    order_path: Optional[str | Path] = None,
    settings: Optional[Settings] = None,
) -> pd.DataFrame:
    """仅加载订单数据。"""
    _, order_df = load_runtime_data(order_path=order_path, settings=settings)
    return order_df


def load_catalog_summary(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """加载目录概览，没有缓存文件时动态生成。"""
    settings = settings or Settings()
    ensure_runtime_data(settings=settings)
    summary_path = Path(settings.CATALOG_SUMMARY_PATH)
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    product_df, order_df = load_runtime_data(settings=settings)
    return build_catalog_summary(product_df, order_df)
