"""
Cross-encoder reranker module for refining retrieval results.

Takes coarse retrieval candidates (typically top-20 from hybrid retrieval)
and produces a fine-grained top-K ranking using a cross-encoder model
that jointly encodes the (query, document) pair.
"""

import logging
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from src.config import Settings

logger = logging.getLogger(__name__)

settings = Settings()


class CrossEncoderReranker:
    """Reranker using a cross-encoder model (e.g., BAAI/bge-reranker-base)."""

    def __init__(
        self,
        model_name: str = settings.RERANKER_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace model identifier for the reranker.
            device: Device to run inference on. Auto-detected if None.
            max_length: Maximum token length for the (query, document) pair.
        """
        self.max_length = max_length

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.available = True
        try:
            logger.info("Loading reranker model: %s on %s", model_name, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Reranker model loaded successfully.")
        except Exception as exc:
            logger.warning("Reranker unavailable, fallback to retrieval ranking: %s", exc)
            self.tokenizer = None
            self.model = None
            self.available = False

    def _compute_scores(
        self,
        query: str,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[float]:
        """
        Compute relevance scores for (query, text) pairs using the cross-encoder.

        Args:
            query: Query text.
            texts: List of candidate document texts.
            batch_size: Number of pairs to process at once.

        Returns:
            List of relevance scores (one per input text).
        """
        if not self.available or self.tokenizer is None or self.model is None:
            return [0.0 for _ in texts]

        all_scores: List[float] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            pairs = [[query, text] for text in batch_texts]

            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                logits = self.model(**inputs).logits.squeeze(-1)
                scores = logits.cpu().float().tolist()

                if isinstance(scores, float):
                    scores = [scores]

                all_scores.extend(scores)

        return all_scores

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = settings.FINAL_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieval candidates using the cross-encoder.

        Args:
            query: Query text.
            candidates: List of candidate dicts, each containing at least:
                        - "id": document identifier
                        - "text": document text
                        - "score": original retrieval score
            top_k: Number of top results to return after reranking.

        Returns:
            Top-K candidates sorted by cross-encoder score, each augmented with:
                - "rerank_score": the cross-encoder relevance score
                - "original_score": preserved from the input "score" field
        """
        if not candidates:
            return []

        if not self.available:
            fallback = []
            for candidate in candidates:
                entry = dict(candidate)
                entry["rerank_score"] = candidate.get("score", 0.0)
                entry["original_score"] = candidate.get("score", 0.0)
                fallback.append(entry)
            fallback.sort(key=lambda item: item["rerank_score"], reverse=True)
            return fallback[:top_k]

        texts = [c["text"] for c in candidates]
        scores = self._compute_scores(query, texts)

        # Augment candidates with reranker scores
        reranked = []
        for candidate, rerank_score in zip(candidates, scores):
            entry = dict(candidate)
            entry["rerank_score"] = rerank_score
            entry["original_score"] = candidate.get("score", 0.0)
            reranked.append(entry)

        # Sort by cross-encoder score descending
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]
