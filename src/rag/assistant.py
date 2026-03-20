import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data import build_catalog_summary, build_product_search_text, load_runtime_data
from src.rag.generation_cleaner import sanitize_generated_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ECommerceRAG:
    """
    E-commerce RAG assistant with hybrid retrieval, cross-encoder reranking,
    and multi-turn conversation support.
    Supports both Chinese (primary) and English product data.
    """

    def __init__(
        self,
        product_dataset_path: Optional[str] = None,
        order_dataset_path: Optional[str] = None,
        model_name: str = None,
        use_reranker: bool = True,
        retrieval_mode: str = "hybrid",
        llm_model: str = None,
    ):
        """
        Initialize RAG system.

        Args:
            product_dataset_path: Path to product CSV.
            order_dataset_path: Path to order CSV.
            model_name: Embedding model name (defaults to config).
            use_reranker: Whether to use cross-encoder reranker.
            retrieval_mode: "hybrid", "dense", or "bm25".
            llm_model: LLM model path (for generation).
        """
        from src.config import Settings

        self.settings = Settings()
        self.product_df, self.order_df = load_runtime_data(
            product_path=product_dataset_path,
            order_path=order_dataset_path,
            settings=self.settings,
        )
        self.retrieval_mode = retrieval_mode
        self.use_reranker = use_reranker
        self.reranker = None
        self.catalog_summary = build_catalog_summary(self.product_df, self.order_df)

        self._preprocess_data()
        self._init_retriever(model_name)

        if use_reranker:
            self._init_reranker()

        # Conversation manager
        from src.rag.conversation import ConversationManager
        self.conversation = ConversationManager()

        # LLM (lazy load)
        self._llm = None
        self._llm_tokenizer = None
        self._llm_model_path = llm_model or self.settings.LLM_MODEL

    def _preprocess_data(self):
        """Build runtime-only fields used by retrieval and recommendation."""
        self.product_df["_search_text"] = self.product_df.apply(build_product_search_text, axis=1)
        self.order_df["Order_DateTime"] = pd.to_datetime(
            self.order_df["Order_DateTime"], errors="coerce"
        )
        self.order_df = self.order_df.sort_values("Order_DateTime", ascending=False).reset_index(drop=True)
        self.product_ids = self.product_df["Product_ID"].astype(str).tolist()
        self.product_index_map = {
            product_id: idx for idx, product_id in enumerate(self.product_ids)
        }

    def _build_search_text(self, row) -> str:
        """Build searchable text from product row."""
        return build_product_search_text(row)

    def _init_retriever(self, model_name: str = None):
        """Initialize retrieval components."""
        from src.rag.retriever import HybridRetriever

        embedding_model = model_name or self.settings.EMBEDDING_MODEL
        self.retriever = HybridRetriever(embedding_model=embedding_model)

        documents = [
            {"id": product_id, "text": search_text}
            for product_id, search_text in zip(self.product_ids, self.product_df["_search_text"].tolist())
        ]
        self.retriever.build_index(documents)
        logger.info("Retriever initialized with %d products", len(documents))

    def _init_reranker(self):
        """Initialize cross-encoder reranker."""
        from src.rag.reranker import CrossEncoderReranker

        self.reranker = CrossEncoderReranker(self.settings.RERANKER_MODEL)
        if self.reranker.available:
            logger.info("Reranker initialized")
        else:
            self.use_reranker = False

    def _load_llm(self):
        """Lazy-load the LLM for generation."""
        if self._llm is not None:
            return

        import json
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_path = Path(self._llm_model_path)
        adapter_config_path = model_path / "adapter_config.json"
        is_adapter_model = adapter_config_path.exists()

        if is_adapter_model:
            adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            base_model_path = str(adapter_config.get("base_model_name_or_path", "")).strip()
            if not base_model_path:
                raise ValueError(f"Adapter model is missing base_model_name_or_path: {adapter_config_path}")
            load_target = base_model_path
            tokenizer_target = str(model_path)
            logger.info("Loading adapter LLM: %s (base: %s)", self._llm_model_path, base_model_path)
        else:
            load_target = self._resolve_base_model_path(self._llm_model_path)
            tokenizer_target = load_target
            logger.info("Loading base LLM: %s", load_target)

        self._llm_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_target,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self._llm_tokenizer.pad_token is None:
            self._llm_tokenizer.pad_token = self._llm_tokenizer.eos_token

        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "local_files_only": True,
        }
        if self.settings.USE_4BIT:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        base_model = AutoModelForCausalLM.from_pretrained(load_target, **model_kwargs)
        if is_adapter_model:
            self._llm = PeftModel.from_pretrained(
                base_model,
                str(model_path),
                is_trainable=False,
                local_files_only=True,
            )
        else:
            self._llm = base_model
        logger.info("LLM loaded")

    def _resolve_base_model_path(self, model_path: str) -> str:
        """Resolve a usable local base model path for inference."""
        candidate = Path(model_path)
        if candidate.exists():
            return str(candidate)

        local_default = Path("/data/wtw/Desktop/resume/models/Qwen2.5-7B-Instruct")
        if local_default.exists():
            return str(local_default)

        return model_path

    def search(
        self,
        query: str,
        top_k: int = None,
        mode: str = None,
        min_rating: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Perform retrieval with optional reranking.

        Returns:
            {
                "results": [product_dict, ...],
                "retrieval_process": {
                    "bm25_results": [...],
                    "dense_results": [...],
                    "reranked_results": [...]
                }
            }
        """
        mode = mode or self.retrieval_mode
        top_k = top_k or self.settings.FINAL_TOP_K
        initial_k = self.settings.HYBRID_TOP_K

        # Stage 1: Retrieval
        detail = self.retriever.search_with_details(query, top_k=initial_k, mode=mode)

        # Convert results to product dicts with scores
        candidates = []
        for doc_id, score in detail["fused_results"]:
            idx = self._get_product_index(doc_id)
            if idx is not None:
                product = self.product_df.iloc[idx].to_dict()
                product["_retrieval_score"] = score
                candidates.append(product)

        # Stage 2: Reranking
        reranked_results = []
        if self.use_reranker and self.reranker is not None and candidates:
            rerank_candidates = [
                {
                    "id": str(c.get("Product_ID", c.get("product_id", i))),
                    "text": c.get("_search_text", ""),
                    "score": c.get("_retrieval_score", 0),
                }
                for i, c in enumerate(candidates)
            ]
            reranked = self.reranker.rerank(query, rerank_candidates, top_k=top_k)

            for r in reranked:
                idx = self._get_product_index(r["id"])
                if idx is not None:
                    product = self.product_df.iloc[idx].to_dict()
                    product["_rerank_score"] = r["rerank_score"]
                    reranked_results.append(product)

            # Update detail for visualization
            detail["reranked_results"] = [
                {"id": r["id"], "title": r.get("text", "")[:50], "score": r["rerank_score"]}
                for r in reranked
            ]
        else:
            reranked_results = candidates[:top_k]
            detail["reranked_results"] = [
                {
                    "id": str(c.get("Product_ID", "")),
                    "title": str(c.get("Product_Title", ""))[:50],
                    "score": c.get("_retrieval_score", 0),
                }
                for c in reranked_results
            ]

        # Apply rating filter
        if min_rating is not None:
            reranked_results = [
                p for p in reranked_results
                if float(p.get("Rating", 0)) >= min_rating
            ][:top_k]

        # Format retrieval process for frontend
        retrieval_process = {
            "bm25_results": [
                {"id": str(r[0]), "score": round(r[1], 4)}
                for r in detail.get("bm25_results", [])[:10]
            ],
            "dense_results": [
                {"id": str(r[0]), "score": round(r[1], 4)}
                for r in detail.get("dense_results", [])[:10]
            ],
            "reranked_results": detail.get("reranked_results", []),
            "summary": {
                "requested_mode": detail.get("requested_mode", mode),
                "selected_mode": detail.get("selected_mode", mode),
                "dense_available": detail.get("dense_available", True),
                "reranker_enabled": bool(self.use_reranker and self.reranker is not None),
                "dataset_mode": self.catalog_summary.get("dataset_mode", "unknown"),
            },
        }

        # Enrich with titles
        for key in ["bm25_results", "dense_results"]:
            for item in retrieval_process[key]:
                idx = self._get_product_index(item["id"])
                if idx is not None:
                    item["title"] = str(self.product_df.iloc[idx].get("Product_Title", ""))[:60]

        return {
            "results": reranked_results,
            "retrieval_process": retrieval_process,
        }

    def generate_response(
        self,
        query: str,
        retrieved_products: List[dict],
        use_llm: bool = False,
    ) -> str:
        """
        Generate a response based on retrieved products.

        Args:
            query: User query.
            retrieved_products: List of product dicts from search().
            use_llm: Whether to use LLM for generation (vs template-based).

        Returns:
            Response string.
        """
        if use_llm:
            return self._generate_with_llm(query, retrieved_products)
        return self._generate_template_response(query, retrieved_products)

    def _generate_with_llm(self, query: str, products: List[dict]) -> str:
        """Generate response using the LLM."""
        import torch

        self._load_llm()

        messages = self.conversation.build_rag_prompt(
            query=query,
            retrieved_docs=products,
            system_prompt=self.settings.SYSTEM_PROMPT,
        )

        text = self._llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_device = next(self._llm.parameters()).device
        inputs = self._llm_tokenizer(text, return_tensors="pt").to(input_device)

        with torch.no_grad():
            outputs = self._llm.generate(
                **inputs,
                max_new_tokens=self.settings.LLM_MAX_NEW_TOKENS,
                temperature=self.settings.LLM_TEMPERATURE,
                top_p=self.settings.LLM_TOP_P,
                do_sample=True,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._llm_tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = sanitize_generated_text(response)

        # Update conversation history
        self.conversation.add_message("user", query)
        self.conversation.add_message("assistant", response)

        return response

    def _generate_template_response(self, query: str, products: List[dict]) -> str:
        """Generate a template-based response (no LLM needed)."""
        if not products:
            return "抱歉，未找到与您需求匹配的商品。请尝试换个关键词搜索。"

        response = "根据您的需求，为您推荐以下商品：\n\n"
        for i, product in enumerate(products, 1):
            title = product.get("Product_Title", product.get("title", "未知商品"))
            price = product.get("Price", product.get("price", 0))
            rating = product.get("Rating", product.get("rating", 0))
            desc = product.get("Description", product.get("description", ""))

            response += f"**{i}. {title}**\n"
            response += f"   - 价格：¥{float(price):.2f}\n"
            response += f"   - 评分：{float(rating):.1f}/5.0\n"
            if desc:
                response += f"   - 简介：{str(desc)[:80]}...\n"
            response += "\n"

        response += "如需了解更多详情，请告诉我您感兴趣的商品编号。"

        # Update conversation history
        self.conversation.add_message("user", query)
        self.conversation.add_message("assistant", response)

        return response

    def process_query(
        self,
        query: str,
        customer_id: Optional[int] = None,
        use_llm: bool = False,
        retrieval_mode: str = None,
    ) -> Dict[str, Any]:
        """
        Process user query end-to-end.

        Returns:
            {
                "answer": str,
                "retrieval_process": {...},
                "recommended_products": [...]
            }
        """
        query_lower = query.lower().strip()

        # Handle greeting / small-talk queries before retrieval.
        if self._is_smalltalk_query(query):
            return self._handle_smalltalk_query(query)

        # Handle high priority orders
        if "高优先级" in query or "high priority" in query_lower:
            return self._handle_high_priority_query()

        # Handle order-related queries
        order_keywords_zh = ["订单", "物流", "快递", "发货", "退货", "退款"]
        order_keywords_en = ["order", "orders", "purchase", "bought", "shipping"]
        is_order_query = any(kw in query_lower for kw in order_keywords_zh + order_keywords_en)

        if is_order_query:
            return self._handle_order_query(query, customer_id)

        # Extract rating filter
        min_rating = self._extract_rating_filter(query)

        # Product search
        search_result = self.search(
            query,
            mode=retrieval_mode or self.retrieval_mode,
            min_rating=min_rating,
        )

        products = search_result["results"]
        answer = self.generate_response(query, products, use_llm=use_llm)

        # Format recommended products for frontend
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
                "match_reason": self._build_match_reason(query, p),
                "confidence": self._estimate_confidence(p),
            }
            for p in products[:5]
        ]

        return {
            "answer": answer,
            "retrieval_process": search_result["retrieval_process"],
            "recommended_products": recommended,
        }

    def _is_smalltalk_query(self, query: str) -> bool:
        """Detect greeting / identity / courtesy turns that should skip retrieval."""
        stripped_query = query.strip()
        normalized_query = stripped_query.lower()

        exact_queries = {
            "你好", "您好", "嗨", "哈喽", "hi", "hello", "hey",
            "在吗", "在不在", "谢谢", "感谢", "再见", "bye", "goodbye",
        }
        phrase_queries = [
            "你是谁",
            "你是干什么的",
            "你能做什么",
            "你可以做什么",
            "介绍一下你自己",
            "介绍下你自己",
            "自我介绍",
            "谢谢你",
            "谢了",
            "回头见",
        ]

        if stripped_query in exact_queries or normalized_query in exact_queries:
            return True

        return any(phrase in stripped_query or phrase in normalized_query for phrase in phrase_queries)

    def _handle_smalltalk_query(self, query: str) -> Dict[str, Any]:
        """Return a stable response for greeting / identity queries."""
        stripped_query = query.strip()

        if any(token in stripped_query for token in ["谢谢", "感谢", "谢了"]):
            answer = "不客气。我可以继续帮您做商品推荐、商品对比、参数解读或订单查询。"
        elif any(token in stripped_query for token in ["再见", "回头见"]) or stripped_query.lower() in {"bye", "goodbye"}:
            answer = "好的，随时需要再来找我。我可以继续帮您查商品、做对比或看订单。"
        elif any(token in stripped_query for token in ["你是谁", "自我介绍", "介绍一下你自己", "介绍下你自己"]):
            answer = (
                "我是电商购物助手，可以帮您做商品推荐、商品对比、参数解读和订单查询。"
                "您可以直接告诉我预算、品类、使用场景，或者给我一个商品名。"
            )
        else:
            answer = (
                "您好，我是电商购物助手。"
                "我可以帮您找商品、比参数、看评分价格，也可以查询订单信息。"
                "您可以直接说，比如“推荐一款5000元以内的笔记本”。"
            )

        self.conversation.add_message("user", query)
        self.conversation.add_message("assistant", answer)

        return {
            "answer": answer,
            "retrieval_process": {
                "bm25_results": [],
                "dense_results": [],
                "reranked_results": [],
                "summary": {
                    "requested_mode": "smalltalk",
                    "selected_mode": "smalltalk",
                    "dense_available": False,
                    "reranker_enabled": False,
                    "dataset_mode": self.catalog_summary.get("dataset_mode", "unknown"),
                },
            },
            "recommended_products": [],
        }

    def _handle_order_query(self, query: str, customer_id: Optional[int]) -> Dict[str, Any]:
        """Handle order-related queries."""
        if not customer_id:
            return {
                "answer": "请提供您的客户ID以查询订单信息。",
                "retrieval_process": {},
                "recommended_products": [],
            }

        orders = self.get_customer_orders(customer_id)
        if not orders:
            answer = f"未找到客户 {customer_id} 的订单记录。"
        else:
            answer = self._format_orders_zh(orders[:5])

        return {
            "answer": answer,
            "retrieval_process": {},
            "recommended_products": [],
        }

    def _handle_high_priority_query(self) -> Dict[str, Any]:
        """Handle high priority order queries."""
        orders = self.get_high_priority_orders()
        answer = self._format_high_priority_zh(orders)
        return {
            "answer": answer,
            "retrieval_process": {},
            "recommended_products": [],
        }

    def get_customer_orders(self, customer_id: int) -> List[Dict[str, Any]]:
        """Get orders for a specific customer."""
        customer_orders = self.order_df[self.order_df["Customer_Id"] == customer_id]
        return customer_orders.sort_values("Order_DateTime", ascending=False).to_dict("records")

    def get_high_priority_orders(self) -> List[Dict[str, Any]]:
        """Get high priority orders."""
        high_priority = self.order_df[
            self.order_df["Order_Priority"].str.lower() == "high"
        ]
        return high_priority.sort_values("Order_DateTime", ascending=False).head(5).to_dict("records")

    def _format_orders_zh(self, orders: List[dict]) -> str:
        """Format orders in Chinese."""
        if not orders:
            return "未找到订单记录。"

        response = "以下是您最近的订单：\n\n"
        for i, order in enumerate(orders, 1):
            dt = pd.Timestamp(order.get("Order_DateTime", "")).strftime("%Y-%m-%d %H:%M")
            product = order.get("Product", order.get("Product_Category", "未知商品"))
            sales = float(order.get("Sales", 0))
            priority = order.get("Order_Priority", "")
            response += (
                f"{i}. **{product}**\n"
                f"   - 下单时间：{dt}\n"
                f"   - 金额：¥{sales:.2f}\n"
                f"   - 优先级：{priority}\n\n"
            )
        return response

    def _format_high_priority_zh(self, orders: List[dict]) -> str:
        """Format high priority orders in Chinese."""
        if not orders:
            return "当前没有高优先级订单。"

        response = "以下是最近5条高优先级订单：\n\n"
        for i, order in enumerate(orders, 1):
            dt = pd.Timestamp(order.get("Order_DateTime", "")).strftime("%Y-%m-%d %H:%M")
            product = order.get("Product", "")
            sales = float(order.get("Sales", 0))
            customer = order.get("Customer_Id", "")
            response += (
                f"{i}. {product} — ¥{sales:.2f}\n"
                f"   客户ID: {customer}，时间: {dt}\n\n"
            )
        return response

    def _extract_rating_filter(self, query: str) -> Optional[float]:
        """Extract rating filter from query text."""
        import re

        # Chinese patterns
        zh_match = re.search(r"(\d\.?\d?)\s*[分星]以上", query)
        if zh_match:
            return float(zh_match.group(1))

        # English patterns
        en_match = re.search(r"above\s+(\d\.?\d?)", query.lower())
        if en_match:
            return float(en_match.group(1))

        return None

    def _build_match_reason(self, query: str, product: Dict[str, Any]) -> str:
        """为推荐商品生成简洁的解释理由。"""
        keywords = self._extract_query_keywords(query)
        searchable_text = " ".join(
            str(product.get(field, "")).lower()
            for field in ["Product_Title", "Description", "Category", "Brand", "features"]
        )

        matched_keywords = [keyword for keyword in keywords if keyword in searchable_text][:2]
        reasons = []
        if matched_keywords:
            reasons.append(f"命中关键词：{' / '.join(matched_keywords)}")

        rating = float(product.get("Rating", 0) or 0)
        if rating >= 4.7:
            reasons.append("口碑表现稳定")

        median_price = float(self.product_df["Price"].median() or 0)
        price = float(product.get("Price", 0) or 0)
        if price and median_price and price <= median_price:
            reasons.append("价格带更友好")

        return "；".join(reasons[:2]) or "综合语义匹配、评分和价格推荐"

    def _estimate_confidence(self, product: Dict[str, Any]) -> int:
        """估算推荐可信度，便于前端展示。"""
        raw_score = product.get("_rerank_score", product.get("_retrieval_score", 0.0))
        if raw_score is None:
            raw_score = 0.0

        if 0 <= float(raw_score) <= 1:
            score_component = float(raw_score)
        else:
            score_component = 1 / (1 + np.exp(-float(raw_score) / 5))

        rating_component = min(1.0, max(0.0, float(product.get("Rating", 0) or 0) / 5))
        confidence = round((score_component * 0.6 + rating_component * 0.4) * 100)
        return int(min(99, max(35, confidence)))

    def _extract_query_keywords(self, query: str) -> List[str]:
        """提取中文查询中的关键字。"""
        import re

        tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", query.lower())
        return [token for token in tokens if len(token) >= 2][:6]

    def _get_product_index(self, doc_id: str) -> Optional[int]:
        """Get DataFrame index for a product ID."""
        return self.product_index_map.get(str(doc_id))

    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation.clear()
