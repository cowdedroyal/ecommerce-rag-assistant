#!/usr/bin/env python3
"""
Retrieval quality evaluation metrics: MRR, Recall@K, NDCG@K.
Measures how well the retrieval pipeline surfaces relevant documents.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def mean_reciprocal_rank(results: List[List[str]], relevant: List[List[str]]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        results: List of ranked result lists (each element is a doc_id).
        relevant: List of relevant doc_id lists for each query.

    Returns:
        MRR score in [0, 1].
    """
    rr_sum = 0.0
    for res, rel in zip(results, relevant):
        rel_set = set(rel)
        for rank, doc_id in enumerate(res, start=1):
            if doc_id in rel_set:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(results) if results else 0.0


def recall_at_k(
    results: List[List[str]],
    relevant: List[List[str]],
    k: int = 5,
) -> float:
    """
    Compute Recall@K averaged across queries.

    Args:
        results: List of ranked result lists.
        relevant: List of relevant doc_id lists for each query.
        k: Cutoff rank.

    Returns:
        Average Recall@K in [0, 1].
    """
    recall_sum = 0.0
    for res, rel in zip(results, relevant):
        rel_set = set(rel)
        if not rel_set:
            continue
        top_k = set(res[:k])
        recall_sum += len(top_k & rel_set) / len(rel_set)
    return recall_sum / len(results) if results else 0.0


def ndcg_at_k(
    results: List[List[str]],
    relevant: List[List[str]],
    k: int = 5,
) -> float:
    """
    Compute NDCG@K (binary relevance) averaged across queries.

    Args:
        results: List of ranked result lists.
        relevant: List of relevant doc_id lists for each query.
        k: Cutoff rank.

    Returns:
        Average NDCG@K in [0, 1].
    """
    ndcg_sum = 0.0
    for res, rel in zip(results, relevant):
        rel_set = set(rel)
        if not rel_set:
            continue

        # DCG
        dcg = 0.0
        for rank, doc_id in enumerate(res[:k], start=1):
            if doc_id in rel_set:
                dcg += 1.0 / math.log2(rank + 1)

        # Ideal DCG
        ideal_hits = min(len(rel_set), k)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))

        ndcg_sum += dcg / idcg if idcg > 0 else 0.0

    return ndcg_sum / len(results) if results else 0.0


def precision_at_k(
    results: List[List[str]],
    relevant: List[List[str]],
    k: int = 5,
) -> float:
    """
    Compute Precision@K averaged across queries.

    Args:
        results: List of ranked result lists.
        relevant: List of relevant doc_id lists for each query.
        k: Cutoff rank.

    Returns:
        Average Precision@K in [0, 1].
    """
    precision_sum = 0.0
    for res, rel in zip(results, relevant):
        rel_set = set(rel)
        top_k = set(res[:k])
        precision_sum += len(top_k & rel_set) / k
    return precision_sum / len(results) if results else 0.0


class RetrievalEvaluator:
    """End-to-end retrieval pipeline evaluator."""

    def __init__(self, retriever, test_queries: List[dict]):
        """
        Args:
            retriever: A retriever instance with `search(query, top_k)` method.
            test_queries: List of {"query": str, "relevant_ids": [str, ...]}.
        """
        self.retriever = retriever
        self.test_queries = test_queries

    def evaluate(self, top_k_values: List[int] = None) -> Dict[str, float]:
        """
        Run evaluation at multiple K cutoffs.

        Args:
            top_k_values: List of K values to evaluate.

        Returns:
            Dictionary of metric_name -> score.
        """
        if top_k_values is None:
            top_k_values = [1, 3, 5, 10, 20]

        max_k = max(top_k_values)

        # Retrieve results for all queries
        all_results = []
        all_relevant = []
        for item in self.test_queries:
            query = item["query"]
            relevant_ids = item["relevant_ids"]

            results = self.retriever.search(query, top_k=max_k)
            result_ids = [str(r[0]) for r in results]

            all_results.append(result_ids)
            all_relevant.append([str(rid) for rid in relevant_ids])

        # Compute metrics at each K
        metrics = {}
        metrics["MRR"] = mean_reciprocal_rank(all_results, all_relevant)

        for k in top_k_values:
            metrics[f"Recall@{k}"] = recall_at_k(all_results, all_relevant, k)
            metrics[f"Precision@{k}"] = precision_at_k(all_results, all_relevant, k)
            metrics[f"NDCG@{k}"] = ndcg_at_k(all_results, all_relevant, k)

        return metrics

    def print_report(self, metrics: Dict[str, float]):
        """Pretty-print evaluation results."""
        logger.info("=" * 50)
        logger.info("Retrieval Evaluation Report")
        logger.info("=" * 50)
        logger.info(f"  MRR:          {metrics['MRR']:.4f}")
        for key, value in sorted(metrics.items()):
            if key != "MRR":
                logger.info(f"  {key:15s} {value:.4f}")
        logger.info("=" * 50)


def load_test_queries(path: str) -> List[dict]:
    """Load test queries from a JSONL file."""
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries
