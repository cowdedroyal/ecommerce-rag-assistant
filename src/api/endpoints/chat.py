"""
Chat API endpoint with SSE streaming support and retrieval process visualization.
"""

import asyncio
import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy-loaded RAG instance
_rag_instances = {}


def _get_rag(model_variant: str = "base"):
    """Get or create RAG instance for the given model variant."""
    if model_variant not in _rag_instances:
        from src.config import Settings
        from src.rag.assistant import ECommerceRAG

        settings = Settings()

        # Map model variant to LLM path
        llm_map = {
            "base": settings.LLM_MODEL,
            "sft": settings.SFT_OUTPUT_DIR,
            "dpo": settings.DPO_OUTPUT_DIR,
        }
        llm_path = llm_map.get(model_variant, settings.LLM_MODEL)

        _rag_instances[model_variant] = ECommerceRAG(
            product_dataset_path=str(settings.PRODUCT_DATA_PATH),
            order_dataset_path=str(settings.ORDER_DATA_PATH),
            llm_model=llm_path,
        )
    return _rag_instances[model_variant]


class ChatRequest(BaseModel):
    """Chat request body."""
    message: str
    model: str = "base"  # base / sft / dpo
    retrieval_mode: str = "hybrid"  # hybrid / dense / bm25
    conversation_id: Optional[str] = None
    customer_id: Optional[int] = None
    use_llm: bool = True
    stream: bool = False


class ChatResponse(BaseModel):
    """Chat response body."""
    answer: str
    conversation_id: str
    retrieval_process: dict = {}
    recommended_products: list = []


@router.post("/send", response_model=ChatResponse)
async def send_message(req: ChatRequest):
    """
    Send a chat message and get a response.
    Returns the full response with retrieval process details.
    """
    rag = _get_rag(req.model)

    # Process query
    result = rag.process_query(
        query=req.message,
        customer_id=req.customer_id,
        use_llm=req.use_llm,
        retrieval_mode=req.retrieval_mode,
    )

    conversation_id = req.conversation_id or str(uuid.uuid4())

    return ChatResponse(
        answer=result["answer"],
        conversation_id=conversation_id,
        retrieval_process=result.get("retrieval_process", {}),
        recommended_products=result.get("recommended_products", []),
    )


@router.post("/stream")
async def stream_message(req: ChatRequest):
    """
    Send a chat message and get a streaming SSE response.
    Events:
      - retrieval: Retrieval process data
      - token: Generated token
      - products: Recommended products
      - done: End of stream
    """

    async def event_generator():
        rag = _get_rag(req.model)
        conversation_id = req.conversation_id or str(uuid.uuid4())

        # Step 1: Retrieval
        search_result = rag.search(
            req.message,
            mode=req.retrieval_mode,
        )

        # Send retrieval process
        yield _sse_event("retrieval", {
            "retrieval_process": search_result["retrieval_process"],
        })
        await asyncio.sleep(0)

        # Step 2: Generate response
        products = search_result["results"]
        answer = rag.generate_response(
            req.message,
            products,
            use_llm=req.use_llm,
        )

        # Simulate streaming by chunking the response
        chunk_size = 20
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i : i + chunk_size]
            yield _sse_event("token", {"text": chunk})
            await asyncio.sleep(0.03)

        # Step 3: Send recommended products
        recommended = [
            {
                "id": str(p.get("Product_ID", p.get("product_id", ""))),
                "title": p.get("Product_Title", p.get("title", "")),
                "price": float(p.get("Price", p.get("price", 0))),
                "rating": float(p.get("Rating", p.get("rating", 0))),
                "category": p.get("Category", p.get("category", "")),
                "source": p.get("source", ""),
                "brand": p.get("Brand", p.get("brand", "")),
                "description": str(p.get("Description", p.get("description", "")))[:140],
                "match_reason": rag._build_match_reason(req.message, p),
                "confidence": rag._estimate_confidence(p),
            }
            for p in products[:5]
        ]

        yield _sse_event("products", {"recommended_products": recommended})

        # Done
        yield _sse_event("done", {
            "conversation_id": conversation_id,
            "full_answer": answer,
        })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/clear")
async def clear_conversation(model: str = "base"):
    """Clear conversation history for a model variant."""
    rag = _get_rag(model)
    rag.clear_conversation()
    return {"status": "ok", "message": "对话历史已清除"}


def _sse_event(event: str, data: dict) -> str:
    """Format an SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
