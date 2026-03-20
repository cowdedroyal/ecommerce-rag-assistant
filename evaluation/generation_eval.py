#!/usr/bin/env python3
"""
Generation quality evaluation metrics.
- ROUGE-L (Chinese-aware with jieba segmentation)
- Factual consistency scoring
- Multi-dimensional answer quality assessment
"""

import json
import logging
import re
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def rouge_l_chinese(prediction: str, reference: str) -> Dict[str, float]:
    """
    Compute ROUGE-L score for Chinese text using jieba segmentation.

    Args:
        prediction: Model generated text.
        reference: Ground truth reference text.

    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    import jieba

    pred_tokens = list(jieba.cut(prediction))
    ref_tokens = list(jieba.cut(reference))

    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Compute LCS length
    lcs_len = _lcs_length(pred_tokens, ref_tokens)

    precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(seq1: list, seq2: list) -> int:
    """Compute the length of the Longest Common Subsequence."""
    m, n = len(seq1), len(seq2)
    # Space-optimized: only keep two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def factual_consistency_score(
    answer: str,
    retrieved_docs: List[dict],
    key_fields: List[str] = None,
) -> Dict[str, float]:
    """
    Measure factual consistency between the answer and retrieved documents.
    Checks if key facts mentioned in the answer appear in source documents.

    Args:
        answer: Generated answer text.
        retrieved_docs: List of source documents (dicts with product info).
        key_fields: Fields to check for consistency.

    Returns:
        {"fact_coverage": float, "hallucination_ratio": float}
    """
    if key_fields is None:
        key_fields = ["title", "price", "rating", "brand"]

    # Extract factual claims from answer
    facts_in_docs = set()
    for doc in retrieved_docs:
        for field in key_fields:
            value = doc.get(field, "")
            if value:
                facts_in_docs.add(str(value).strip().lower())

    if not facts_in_docs:
        return {"fact_coverage": 0.0, "hallucination_ratio": 0.0}

    # Count how many doc facts appear in the answer
    answer_lower = answer.lower()
    matched = sum(1 for fact in facts_in_docs if fact in answer_lower)
    coverage = matched / len(facts_in_docs)

    # Extract numbers from the answer and check against doc numbers
    answer_numbers = set(re.findall(r"\d+\.?\d*", answer))
    doc_numbers = set()
    for doc in retrieved_docs:
        for field in ["price", "rating", "review_count"]:
            val = doc.get(field, "")
            if val:
                doc_numbers.update(re.findall(r"\d+\.?\d*", str(val)))

    # Numbers in answer not found in docs may indicate hallucination
    if answer_numbers:
        unsupported = answer_numbers - doc_numbers
        hallucination_ratio = len(unsupported) / len(answer_numbers)
    else:
        hallucination_ratio = 0.0

    return {
        "fact_coverage": coverage,
        "hallucination_ratio": hallucination_ratio,
    }


def answer_quality_score(answer: str) -> Dict[str, float]:
    """
    Multi-dimensional answer quality assessment.

    Dimensions:
        - structure: Does the answer use markdown formatting (lists, headers)?
        - completeness: Is the answer sufficiently detailed (length, paragraphs)?
        - specificity: Does the answer contain specific data (numbers, brands)?

    Args:
        answer: Generated answer text.

    Returns:
        Dictionary of dimension scores in [0, 1].
    """
    scores = {}

    # Structure: check for formatting elements
    has_list = bool(re.search(r"^[\s]*[-\d*●•]", answer, re.MULTILINE))
    has_header = bool(re.search(r"^#+\s|^\*\*.*\*\*", answer, re.MULTILINE))
    has_newlines = answer.count("\n") >= 2
    scores["structure"] = (
        (0.4 if has_list else 0.0)
        + (0.3 if has_header else 0.0)
        + (0.3 if has_newlines else 0.0)
    )

    # Completeness: length-based scoring
    char_count = len(answer)
    if char_count < 50:
        scores["completeness"] = 0.2
    elif char_count < 100:
        scores["completeness"] = 0.5
    elif char_count < 300:
        scores["completeness"] = 0.8
    else:
        scores["completeness"] = 1.0

    # Specificity: check for concrete data points
    has_price = bool(re.search(r"[¥￥$]\s?\d+|[\d,]+\s?元", answer))
    has_rating = bool(re.search(r"\d\.?\d?\s?[星分/]|评[分级]", answer))
    has_brand = bool(re.search(r"[\u4e00-\u9fff]{2,}牌|[A-Z][a-zA-Z]+", answer))
    has_numbers = len(re.findall(r"\d+", answer)) >= 3
    scores["specificity"] = (
        (0.3 if has_price else 0.0)
        + (0.25 if has_rating else 0.0)
        + (0.25 if has_brand else 0.0)
        + (0.2 if has_numbers else 0.0)
    )

    # Overall
    scores["overall"] = (
        scores["structure"] * 0.3
        + scores["completeness"] * 0.3
        + scores["specificity"] * 0.4
    )

    return scores


class GenerationEvaluator:
    """End-to-end generation quality evaluator."""

    def __init__(self, test_data: List[dict]):
        """
        Args:
            test_data: List of evaluation samples, each containing:
                - query: str
                - reference: str (ground truth answer)
                - retrieved_docs: List[dict] (source documents)
                - generated: str (model output, to be filled at eval time)
        """
        self.test_data = test_data

    def evaluate(self, predictions: List[str]) -> Dict[str, float]:
        """
        Run full evaluation on generated predictions.

        Args:
            predictions: List of model-generated answers.

        Returns:
            Aggregated metrics dictionary.
        """
        assert len(predictions) == len(self.test_data), (
            f"Prediction count ({len(predictions)}) != test data count ({len(self.test_data)})"
        )

        rouge_scores = []
        fact_scores = []
        quality_scores = []

        for pred, sample in zip(predictions, self.test_data):
            # ROUGE-L
            reference = sample.get("reference", "")
            if reference:
                rouge = rouge_l_chinese(pred, reference)
                rouge_scores.append(rouge["f1"])

            # Factual consistency
            docs = sample.get("retrieved_docs", [])
            if docs:
                fact = factual_consistency_score(pred, docs)
                fact_scores.append(fact)

            # Answer quality
            quality = answer_quality_score(pred)
            quality_scores.append(quality)

        # Aggregate
        metrics = {}
        if rouge_scores:
            metrics["rouge_l_f1"] = sum(rouge_scores) / len(rouge_scores)

        if fact_scores:
            metrics["fact_coverage"] = sum(s["fact_coverage"] for s in fact_scores) / len(fact_scores)
            metrics["hallucination_ratio"] = sum(
                s["hallucination_ratio"] for s in fact_scores
            ) / len(fact_scores)

        if quality_scores:
            for dim in ["structure", "completeness", "specificity", "overall"]:
                metrics[f"quality_{dim}"] = sum(
                    s[dim] for s in quality_scores
                ) / len(quality_scores)

        return metrics

    def print_report(self, metrics: Dict[str, float]):
        """Pretty-print evaluation results."""
        logger.info("=" * 50)
        logger.info("Generation Evaluation Report")
        logger.info("=" * 50)
        for key, value in metrics.items():
            logger.info(f"  {key:25s} {value:.4f}")
        logger.info("=" * 50)
