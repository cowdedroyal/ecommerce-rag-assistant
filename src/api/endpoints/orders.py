from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from src.config import Settings
from src.data import load_order_data

router = APIRouter()
settings = Settings()


def _get_order_df():
    return load_order_data(settings=settings)


def _serialize_order(order: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "order_id": str(order.get("Order_ID", "")),
        "customer_id": int(order.get("Customer_Id", 0) or 0),
        "product_id": str(order.get("Product_ID", "")),
        "product": order.get("Product", ""),
        "category": order.get("Product_Category", ""),
        "sales": round(float(order.get("Sales", 0) or 0), 2),
        "quantity": int(order.get("Quantity", 0) or 0),
        "discount": round(float(order.get("Discount", 0) or 0), 2),
        "priority": order.get("Order_Priority", ""),
        "shipping_status": order.get("Shipping_Status", ""),
        "payment_method": order.get("Payment_Method", ""),
        "order_datetime": order.get("Order_DateTime", ""),
        "source": order.get("source", ""),
    }


@router.get("/customer/{customer_id}", response_model=List[Dict[str, Any]])
async def get_customer_orders(
    customer_id: int,
    limit: int = Query(default=10, ge=1, le=100, description="返回数量"),
):
    """查询指定客户的订单记录。"""
    order_df = _get_order_df()
    customer_orders = order_df[order_df["Customer_Id"] == customer_id].copy()

    if customer_orders.empty:
        raise HTTPException(
            status_code=404,
            detail=f"未找到客户 {customer_id} 的订单",
        )

    customer_orders = customer_orders.sort_values("Order_DateTime", ascending=False).head(limit)
    return [_serialize_order(order) for order in customer_orders.to_dict("records")]


@router.get("/priority/{priority}", response_model=List[Dict[str, Any]])
async def get_orders_by_priority(
    priority: str,
    limit: int = Query(default=10, ge=1, le=100, description="返回数量"),
):
    """按优先级查询订单。"""
    order_df = _get_order_df()
    priority_orders = order_df[
        order_df["Order_Priority"].str.lower() == priority.lower()
    ].copy()

    if priority_orders.empty:
        raise HTTPException(
            status_code=404,
            detail=f"没有找到优先级为 '{priority}' 的订单",
        )

    priority_orders = priority_orders.sort_values("Order_DateTime", ascending=False).head(limit)
    return [_serialize_order(order) for order in priority_orders.to_dict("records")]
