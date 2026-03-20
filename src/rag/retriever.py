"""
Hybrid retrieval module combining dense (embedding-based) and sparse (BM25) retrieval
with Reciprocal Rank Fusion for score aggregation.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from src.config import Settings

logger = logging.getLogger(__name__)

settings = Settings()


class DenseRetriever:
    """Dense retriever using SentenceTransformer embeddings for semantic search."""

    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL,
        device: Optional[str] = None,
    ):
        """
        Initialize the dense retriever.

        Args:
            model_name: HuggingFace model identifier for the embedding model.
            device: Device to run inference on ('cpu', 'cuda', etc.).
                    If None, SentenceTransformer will auto-detect.
        """
        self.is_available = True
        try:
            logger.info("Loading embedding model: %s", model_name)
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            logger.warning("Embedding model unavailable, fallback to BM25 only: %s", exc)
            self.model = None
            self.is_available = False
        self.doc_embeddings: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build the dense index by encoding all documents.

        Each document dict must contain:
            - "id": unique document identifier
            - "text": text content to encode

        Args:
            documents: List of document dicts with "id" and "text" fields.
        """
        if not documents:
            logger.warning("Empty document list provided; index will be empty.")
            self.doc_embeddings = np.array([])
            self.doc_ids = []
            return

        if not self.is_available or self.model is None:
            self.doc_embeddings = np.array([])
            self.doc_ids = [doc["id"] for doc in documents]
            return

        self.doc_ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]

        logger.info("Encoding %d documents for dense index...", len(texts))
        self.doc_embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        logger.info("Dense index built with %d documents.", len(self.doc_ids))

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Batch encode texts into embedding vectors.

        Args:
            texts: List of text strings to encode.

        Returns:
            Numpy array of shape (len(texts), embedding_dim).
        """
        if not self.is_available or self.model is None:
            return np.array([])

        return self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def search(
        self,
        query: str,
        top_k: int = settings.DENSE_TOP_K,
    ) -> List[Tuple[str, float]]:
        """
        Search for documents most similar to the query using cosine similarity.

        Args:
            query: Query text.
            top_k: Number of top results to return.

        Returns:
            List of (doc_id, score) tuples sorted by descending score.
        """
        if not self.is_available or self.model is None:
            logger.warning("Dense retriever unavailable, skipping dense search.")
            return []

        if self.doc_embeddings is None or len(self.doc_embeddings) == 0:
            logger.warning("Dense index is empty. Call build_index() first.")
            return []

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        # Cosine similarity via dot product (embeddings are L2-normalized)
        scores = np.dot(self.doc_embeddings, query_embedding)

        # Get top-k indices
        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = [
            (self.doc_ids[idx], float(scores[idx]))
            for idx in top_indices
        ]
        return results


class BM25Retriever:
    """Sparse retriever using BM25 with jieba Chinese word segmentation."""

    def __init__(self):
        """Initialize the BM25 retriever."""
        self.bm25: Optional[BM25Okapi] = None
        self.doc_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenize Chinese text using jieba.

        Args:
            text: Input text.

        Returns:
            List of tokens with stopwords and single-char tokens removed.
        """
        tokens = jieba.lcut(text)
        # Remove whitespace-only tokens and single characters that are not CJK
        return [
            t for t in tokens
            if t.strip() and (len(t) > 1 or "\u4e00" <= t <= "\u9fff")
        ]

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build the BM25 index from documents.

        Each document dict must contain:
            - "id": unique document identifier
            - "text": text content to index

        Args:
            documents: List of document dicts with "id" and "text" fields.
        """
        if not documents:
            logger.warning("Empty document list provided; BM25 index will be empty.")
            self.bm25 = None
            self.doc_ids = []
            self.tokenized_corpus = []
            return

        self.doc_ids = [doc["id"] for doc in documents]
        logger.info("Tokenizing %d documents for BM25 index...", len(documents))
        self.tokenized_corpus = [self._tokenize(doc["text"]) for doc in documents]

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 index built with %d documents.", len(self.doc_ids))

    def search(
        self,
        query: str,
        top_k: int = settings.BM25_TOP_K,
    ) -> List[Tuple[str, float]]:
        """
        Search for documents matching the query using BM25 scoring.

        Args:
            query: Query text.
            top_k: Number of top results to return.

        Returns:
            List of (doc_id, score) tuples sorted by descending score.
        """
        if self.bm25 is None:
            logger.warning("BM25 index is empty. Call build_index() first.")
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            logger.warning("Query produced no tokens after segmentation.")
            return []

        scores = self.bm25.get_scores(tokenized_query)

        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = [
            (self.doc_ids[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]
        return results


class HybridRetriever:
    """
    Hybrid retriever combining DenseRetriever and BM25Retriever
    with Reciprocal Rank Fusion (RRF) for score aggregation.
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        device: Optional[str] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        rrf_k: int = settings.RRF_K,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            dense_retriever: Pre-configured DenseRetriever instance.
                             A new one is created if None.
            bm25_retriever: Pre-configured BM25Retriever instance.
                            A new one is created if None.
            rrf_k: RRF smoothing constant (default 60).
        """
        self.dense_retriever = dense_retriever or DenseRetriever(
            model_name=embedding_model or settings.EMBEDDING_MODEL,
            device=device,
        )
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.rrf_k = rrf_k

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build indices for both dense and BM25 retrievers.

        Args:
            documents: List of document dicts with "id" and "text" fields.
        """
        self.dense_retriever.build_index(documents)
        self.bm25_retriever.build_index(documents)

    @staticmethod
    def _rrf_fusion(
        result_lists: List[List[Tuple[str, float]]],
        k: int,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Reciprocal Rank Fusion: score = sum(1 / (k + rank_i)) across lists.

        Args:
            result_lists: List of ranked result lists, each containing
                          (doc_id, score) tuples.
            k: RRF smoothing constant.
            top_k: Number of top fused results to return.

        Returns:
            Fused results as a list of (doc_id, rrf_score) tuples.
        """
        fused_scores: Dict[str, float] = {}

        for result_list in result_lists:
            for rank, (doc_id, _) in enumerate(result_list, start=1):
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def search(
        self,
        query: str,
        top_k: int = settings.HYBRID_TOP_K,
        mode: str = "hybrid",
    ) -> List[Tuple[str, float]]:
        """
        Perform hybrid search and return RRF-fused results.

        Args:
            query: Query text.
            top_k: Number of top results to return.

        Returns:
            List of (doc_id, rrf_score) tuples sorted by descending score.
        """
        detail = self.search_with_details(query=query, top_k=top_k, mode=mode)
        return detail["fused_results"]

    def search_with_details(
        self,
        query: str,
        top_k: int = settings.HYBRID_TOP_K,
        mode: str = "hybrid",
    ) -> Dict[str, Any]:
        """
        Perform hybrid search and return per-retriever results along with
        the fused ranking. Useful for front-end visualization and debugging.

        Args:
            query: Query text.
            top_k: Number of top results to return per retriever.

        Returns:
            Dict with keys:
                - "bm25_results": BM25 ranked results
                - "dense_results": Dense ranked results
                - "fused_results": RRF-fused results
        """
        bm25_results = self.bm25_retriever.search(query, top_k=top_k)
        dense_results = self.dense_retriever.search(query, top_k=top_k)

        if mode == "dense":
            fused_results = dense_results[:top_k] or bm25_results[:top_k]
            selected_mode = "dense" if dense_results else "bm25-fallback"
        elif mode == "bm25":
            fused_results = bm25_results[:top_k] or dense_results[:top_k]
            selected_mode = "bm25" if bm25_results else "dense-fallback"
        else:
            fused_results = self._rrf_fusion(
                result_lists=[bm25_results, dense_results],
                k=self.rrf_k,
                top_k=top_k,
            )
            if not fused_results:
                fused_results = bm25_results[:top_k] or dense_results[:top_k]
            selected_mode = "hybrid"

        return {
            "bm25_results": bm25_results,
            "dense_results": dense_results,
            "fused_results": fused_results,
            "requested_mode": mode,
            "selected_mode": selected_mode,
            "dense_available": self.dense_retriever.is_available,
        }
