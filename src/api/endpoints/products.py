from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.config import Settings
from src.data import build_catalog_summary, load_order_data, load_product_data

router = APIRouter()
settings = Settings()


def _get_product_df():
    return load_product_data(settings=settings)


def _serialize_product(product: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(product.get("Product_ID", "")),
        "title": product.get("Product_Title", ""),
        "category": product.get("Category", ""),
        "brand": product.get("Brand", ""),
        "price": round(float(product.get("Price", 0) or 0), 2),
        "rating": round(float(product.get("Rating", 0) or 0), 1),
        "rating_count": int(product.get("Rating_Count", 0) or 0),
        "description": str(product.get("Description", ""))[:180],
        "features": str(product.get("features", "")),
        "source": product.get("source", ""),
        "image": product.get("Image", ""),
    }


def _serialize_products(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_serialize_product(record) for record in records]


@router.get("/search", response_model=List[Dict[str, Any]])
async def search_products(
    query: str = Query(..., min_length=1, description="搜索关键词"),
    category: Optional[str] = Query(None, description="商品分类"),
    min_rating: Optional[float] = Query(None, ge=0, le=5, description="最低评分"),
    max_price: Optional[float] = Query(None, description="最高价格"),
    limit: int = Query(default=10, ge=1, le=50, description="返回数量"),
):
    """搜索商品，支持关键词、分类、评分、价格过滤。"""
    filtered = _get_product_df().copy()

    search_mask = filtered["combined_text"].str.contains(query, case=False, na=False)
    filtered = filtered[search_mask]

    if category:
        filtered = filtered[
            filtered["Category"].str.contains(category, case=False, na=False)
        ]

    if min_rating is not None:
        filtered = filtered[filtered["Rating"] >= min_rating]

    if max_price is not None:
        filtered = filtered[filtered["Price"] <= max_price]

    if filtered.empty:
        raise HTTPException(status_code=404, detail="未找到匹配的商品")

    filtered = filtered.sort_values(["Rating", "Rating_Count"], ascending=[False, False]).head(limit)
    return _serialize_products(filtered.to_dict("records"))


@router.get("/category/{category}", response_model=List[Dict[str, Any]])
async def get_products_by_category(
    category: str,
    limit: int = Query(default=10, ge=1, le=50),
    min_rating: Optional[float] = None,
):
    """按分类获取商品。"""
    cat_products = _get_product_df()
    cat_products = cat_products[
        cat_products["Category"].str.contains(category, case=False, na=False)
    ].copy()

    if min_rating is not None:
        cat_products = cat_products[cat_products["Rating"] >= min_rating]

    if cat_products.empty:
        raise HTTPException(status_code=404, detail=f"分类 '{category}' 下没有商品")

    cat_products = cat_products.sort_values(["Rating", "Rating_Count"], ascending=[False, False]).head(limit)
    return _serialize_products(cat_products.to_dict("records"))


@router.get("/top-rated", response_model=List[Dict[str, Any]])
async def get_top_rated_products(
    min_rating: float = Query(4.0, ge=0, le=5, description="最低评分阈值"),
    category: Optional[str] = None,
    limit: int = Query(default=10, ge=1, le=50),
):
    """获取高评分商品。"""
    top = _get_product_df()
    top = top[top["Rating"] >= min_rating].copy()

    if category:
        top = top[top["Category"].str.contains(category, case=False, na=False)]

    if top.empty:
        raise HTTPException(status_code=404, detail="未找到符合条件的商品")

    top = top.sort_values(["Rating", "Rating_Count"], ascending=[False, False]).head(limit)
    return _serialize_products(top.to_dict("records"))


@router.get("/recommendations/{product_id}", response_model=List[Dict[str, Any]])
async def get_product_recommendations(
    product_id: str,
    limit: int = Query(default=5, ge=1, le=20),
):
    """基于同分类和相近价格区间的商品推荐。"""
    product_df = _get_product_df()
    target = product_df[product_df["Product_ID"].astype(str) == str(product_id)]
    if target.empty:
        raise HTTPException(status_code=404, detail=f"商品 {product_id} 不存在")

    target_row = target.iloc[0]
    similar = product_df[
        (product_df["Category"] == target_row["Category"])
        & (product_df["Product_ID"].astype(str) != str(product_id))
    ].copy()

    if similar.empty:
        raise HTTPException(status_code=404, detail="没有找到相似商品")

    target_price = float(target_row["Price"] or 0)
    similar["price_gap"] = (similar["Price"] - target_price).abs()
    similar = similar.sort_values(["Rating", "price_gap"], ascending=[False, True]).head(limit)
    return _serialize_products(similar.to_dict("records"))


@router.get("/catalog-summary", response_model=Dict[str, Any])
async def get_catalog_summary():
    """获取当前目录概览，用于演示模式和前端欢迎页展示。"""
    product_df = _get_product_df()
    order_df = load_order_data(settings=settings)
    summary = build_catalog_summary(product_df, order_df)
    summary["showcase_products"] = _serialize_products(
        product_df.sort_values(["Rating", "Rating_Count"], ascending=[False, False]).head(4).to_dict("records")
    )
    return summary
