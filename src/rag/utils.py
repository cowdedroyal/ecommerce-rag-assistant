"""
Utility functions for Chinese and English text processing,
price formatting, date handling, and data manipulation.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Text Processing
# ============================================================

def preprocess_text(text: str, lang: str = "auto") -> str:
    """
    Preprocess text for matching/indexing.

    Args:
        text: Input text.
        lang: "zh" for Chinese, "en" for English, "auto" for detection.

    Returns:
        Cleaned text.
    """
    if not isinstance(text, str):
        return ""

    text = text.strip()
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    if lang == "en" or (lang == "auto" and not _contains_chinese(text)):
        text = text.lower()

    return text


def segment_chinese(text: str) -> List[str]:
    """
    Segment Chinese text using jieba.

    Args:
        text: Input text.

    Returns:
        List of segmented tokens.
    """
    import jieba

    tokens = jieba.cut(text)
    # Filter empty and single-char non-CJK tokens
    return [t for t in tokens if t.strip() and (len(t) > 1 or _is_cjk_char(t))]


def _contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _is_cjk_char(char: str) -> bool:
    """Check if a character is a CJK character."""
    if len(char) != 1:
        return False
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )


# ============================================================
# Similarity
# ============================================================

def calculate_semantic_similarity(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Calculate cosine similarity between query and documents.

    Args:
        query_embedding: Query embedding vector (1D).
        document_embeddings: Matrix of document embeddings (2D).

    Returns:
        Array of similarity scores.
    """
    # Normalize
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    doc_norms = document_embeddings / (
        np.linalg.norm(document_embeddings, axis=1, keepdims=True) + 1e-10
    )
    return np.dot(doc_norms, query_norm)


# ============================================================
# Price Formatting
# ============================================================

def format_price(price: float, currency: str = "auto") -> str:
    """
    Format price with proper currency symbol.

    Args:
        price: Price value.
        currency: "cny" for RMB, "usd" for USD, "auto" for auto-detect.

    Returns:
        Formatted price string.
    """
    if price is None or np.isnan(price):
        return "价格未知"

    if currency == "cny" or (currency == "auto" and price < 100000):
        return f"¥{price:,.2f}"
    elif currency == "usd":
        return f"${price:,.2f}"
    return f"¥{price:,.2f}"


def parse_price_range(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract price range from Chinese/English text.

    Examples:
        "500元以内" -> (None, 500)
        "1000-2000元" -> (1000, 2000)
        "2000元以上" -> (2000, None)

    Returns:
        (min_price, max_price) tuple.
    """
    # Chinese: X元以内 / X元以下
    m = re.search(r"(\d+\.?\d*)\s*元?\s*以[内下]", text)
    if m:
        return None, float(m.group(1))

    # Chinese: X元以上
    m = re.search(r"(\d+\.?\d*)\s*元?\s*以上", text)
    if m:
        return float(m.group(1)), None

    # Range: X-Y元
    m = re.search(r"(\d+\.?\d*)\s*[-~到]\s*(\d+\.?\d*)\s*元?", text)
    if m:
        return float(m.group(1)), float(m.group(2))

    # English: under $X
    m = re.search(r"under\s+\$?(\d+\.?\d*)", text, re.IGNORECASE)
    if m:
        return None, float(m.group(1))

    # English: above $X
    m = re.search(r"above\s+\$?(\d+\.?\d*)", text, re.IGNORECASE)
    if m:
        return float(m.group(1)), None

    return None, None


# ============================================================
# Date Handling
# ============================================================

def parse_date_range(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Parse date range strings into datetime objects.

    Args:
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).

    Returns:
        Tuple of parsed dates.
    """
    parsed_start = None
    parsed_end = None

    if start_date:
        try:
            parsed_start = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid start_date format. Use YYYY-MM-DD")

    if end_date:
        try:
            parsed_end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid end_date format. Use YYYY-MM-DD")

    if parsed_start and parsed_end and parsed_start > parsed_end:
        raise ValueError("start_date cannot be later than end_date")

    return parsed_start, parsed_end


def filter_dataframe_by_date(
    df: pd.DataFrame,
    date_column: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Filter DataFrame by date range."""
    filtered = df.copy()
    if start_date:
        filtered = filtered[pd.to_datetime(filtered[date_column]) >= start_date]
    if end_date:
        filtered = filtered[pd.to_datetime(filtered[date_column]) <= end_date]
    return filtered


# ============================================================
# Data Formatting
# ============================================================

def calculate_order_statistics(orders_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary statistics for orders."""
    return {
        "total_orders": len(orders_df),
        "total_sales": float(orders_df["Sales"].sum()),
        "average_order_value": float(orders_df["Sales"].mean()),
        "total_shipping_cost": float(orders_df["Shipping_Cost"].sum()),
        "orders_by_priority": orders_df["Order_Priority"].value_counts().to_dict(),
        "orders_by_category": orders_df["Product_Category"].value_counts().to_dict(),
    }


def format_product_response(product: Dict[str, Any], include_score: bool = False) -> str:
    """Format product information into a readable Chinese string."""
    title = product.get("Product_Title", product.get("title", "未知商品"))
    price = product.get("Price", product.get("price", 0))
    rating = product.get("Rating", product.get("rating", 0))

    lines = [f"- **{title}**"]
    lines.append(f"  价格：{format_price(float(price))}")
    lines.append(f"  评分：{float(rating):.1f}/5.0")

    desc = product.get("Description", product.get("description", ""))
    if desc:
        lines.append(f"  简介：{str(desc)[:100]}...")

    if include_score and "similarity_score" in product:
        lines.append(f"  相关度：{product['similarity_score']:.2f}")

    return "\n".join(lines)


def format_order_response(order: Dict[str, Any]) -> str:
    """Format order information into a readable Chinese string."""
    lines = ["订单详情："]
    lines.append(f"- 日期：{order.get('Order_Date', order.get('Order_DateTime', '未知'))}")
    lines.append(f"- 商品：{order.get('Product', order.get('Product_Category', '未知'))}")
    lines.append(f"- 金额：{format_price(float(order.get('Sales', 0)))}")
    lines.append(f"- 运费：{format_price(float(order.get('Shipping_Cost', 0)))}")
    lines.append(f"- 优先级：{order.get('Order_Priority', '未知')}")
    return "\n".join(lines)
