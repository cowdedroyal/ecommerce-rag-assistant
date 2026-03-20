from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .endpoints import orders, products, chat
from ..config import Settings
from ..data import ensure_runtime_data, load_catalog_summary


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = Settings()
    ensure_runtime_data(settings=settings)
    yield

# Initialize FastAPI app
app = FastAPI(
    title="电商智能助手 API",
    description="中文电商 RAG 智能助手 API，支持混合检索、重排序、多模型对比",
    version="2.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(orders.router, prefix="/orders", tags=["orders"])
app.include_router(products.router, prefix="/products", tags=["products"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    summary = load_catalog_summary(Settings())
    return {
        "status": "healthy",
        "version": "2.0.0",
        "dataset_mode": summary.get("dataset_mode", "unknown"),
        "product_count": summary.get("product_count", 0),
        "order_count": summary.get("order_count", 0),
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "电商智能助手 API",
        "version": "2.0.0",
        "endpoints": {
            "chat": "/chat/send (POST) - 发送消息",
            "chat_stream": "/chat/stream (POST) - 流式对话",
            "products": "/products/search - 商品搜索",
            "catalog_summary": "/products/catalog-summary - 目录概览",
            "orders": "/orders/customer/{id} - 订单查询",
            "docs": "/docs - API 文档",
        },
    }


if __name__ == "__main__":
    import uvicorn

    settings = Settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
