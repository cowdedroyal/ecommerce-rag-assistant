"""
Post-generation quality filter for SFT and DPO training data.

Applies the following rejection criteria:
    1. Too short   -- assistant reply < 50 characters
    2. Too long    -- assistant reply > 2000 characters
    3. No structure -- pure flat text without any list / heading / table markers
    4. No data     -- reply does not reference any concrete product data
                      (price, rating, product name, etc.)

Usage:
    qf = QualityFilter()
    clean_sft = qf.filter_sft(raw_sft_samples)
    clean_dpo = qf.filter_dpo(raw_dpo_samples)
    qf.print_report()
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Statistics container
# ------------------------------------------------------------------

@dataclass
class FilterStats:
    """Accumulates per-rule rejection counts."""
    total_before: int = 0
    total_after: int = 0
    rejected_too_short: int = 0
    rejected_too_long: int = 0
    rejected_no_structure: int = 0
    rejected_no_data: int = 0
    rejected_budget_mismatch: int = 0

    def rejection_summary(self) -> Dict[str, int]:
        """Return a dict summary of rejections."""
        return {
            "total_before": self.total_before,
            "total_after": self.total_after,
            "rejected_total": self.total_before - self.total_after,
            "rejected_too_short": self.rejected_too_short,
            "rejected_too_long": self.rejected_too_long,
            "rejected_no_structure": self.rejected_no_structure,
            "rejected_no_data": self.rejected_no_data,
            "rejected_budget_mismatch": self.rejected_budget_mismatch,
        }


# ------------------------------------------------------------------
# Quality filter
# ------------------------------------------------------------------

class QualityFilter:
    """Rule-based quality filter for generated training data."""

    # Tuneable thresholds
    MIN_REPLY_LEN: int = 50
    MAX_REPLY_LEN: int = 2000

    # Regex patterns that indicate structural formatting
    STRUCTURE_PATTERNS: List[re.Pattern[str]] = [
        re.compile(r"^\s*[-*]\s+", re.MULTILINE),      # bullet list
        re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE),   # numbered list
        re.compile(r"\*\*[^*]+\*\*"),                    # bold text
        re.compile(r"\|.*\|.*\|"),                       # table row
        re.compile(r"^#{1,4}\s+", re.MULTILINE),        # heading
        re.compile(r"\n\n"),                              # paragraph break
    ]

    # Patterns that indicate the answer references real data
    DATA_PATTERNS: List[re.Pattern[str]] = [
        re.compile(r"[\$\u00a5]\s*\d+"),                 # price ($xx or ¥xx)
        re.compile(r"\d+(\.\d+)?\s*元"),                  # price xx元
        re.compile(r"\d+(\.\d+)?\s*/\s*5"),              # rating x/5
        re.compile(r"\d+\s*条评[价价]"),                   # review count
        re.compile(r"评分[：:]\s*\d"),                     # "评分：4"
        re.compile(r"价格[：:]\s*[\$\u00a5]"),            # "价格：$"
        re.compile(r"价格[：:]\s*\d+(\.\d+)?\s*元"),      # "价格：399元"
        re.compile(r"订单编号"),                           # order ID mention
        re.compile(r"ORD\d+"),                            # synthetic order ID
    ]

    def __init__(
        self,
        min_len: Optional[int] = None,
        max_len: Optional[int] = None,
    ) -> None:
        if min_len is not None:
            self.MIN_REPLY_LEN = min_len
        if max_len is not None:
            self.MAX_REPLY_LEN = max_len

        self._sft_stats = FilterStats()
        self._dpo_stats = FilterStats()

    # ------------------------------------------------------------------
    # Core check helpers
    # ------------------------------------------------------------------

    def _is_too_short(self, text: str) -> bool:
        """Return True if *text* is below the minimum length threshold."""
        return len(text.strip()) < self.MIN_REPLY_LEN

    def _is_too_long(self, text: str) -> bool:
        """Return True if *text* exceeds the maximum length threshold."""
        return len(text.strip()) > self.MAX_REPLY_LEN

    def _has_structure(self, text: str) -> bool:
        """Return True if *text* contains any structural formatting."""
        return any(pat.search(text) for pat in self.STRUCTURE_PATTERNS)

    def _has_data_reference(self, text: str) -> bool:
        """Return True if *text* references concrete product / order data."""
        return any(pat.search(text) for pat in self.DATA_PATTERNS)

    @staticmethod
    def _extract_budget_limit(text: str) -> Optional[float]:
        """Extract a budget limit from the user prompt when the constraint is explicit."""
        patterns = [
            re.compile(r"预算\s*(\d+(?:\.\d+)?)\s*元"),
            re.compile(r"(\d+(?:\.\d+)?)\s*元以内"),
            re.compile(r"控制在\s*(\d+(?:\.\d+)?)\s*元"),
        ]
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return float(match.group(1))
        return None

    @staticmethod
    def _extract_prices(text: str) -> List[float]:
        """Extract prices from assistant text for coarse budget alignment checks."""
        values = re.findall(r"[\$\u00a5]\s*(\d+(?:\.\d+)?)", text)
        values.extend(re.findall(r"(\d+(?:\.\d+)?)\s*元", text))
        return [float(value) for value in values]

    def _respects_budget(self, prompt: str, text: str) -> bool:
        """Return True when answers with explicit budget constraints stay roughly in range."""
        budget_limit = self._extract_budget_limit(prompt)
        if budget_limit is None:
            return True

        prices = self._extract_prices(text)
        if not prices:
            return True

        return any(price <= budget_limit * 1.12 for price in prices)

    # ------------------------------------------------------------------
    # SFT filtering
    # ------------------------------------------------------------------

    def _extract_assistant_text(self, sample: Dict[str, Any]) -> str:
        """Extract the last assistant message from an SFT messages-format sample."""
        messages = sample.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""

    def check_sft_sample(self, sample: Dict[str, Any]) -> Optional[str]:
        """
        Validate a single SFT sample.

        Returns:
            None if the sample passes, or a rejection reason string.
        """
        text = self._extract_assistant_text(sample)
        prompt = ""
        for message in sample.get("messages", []):
            if message.get("role") == "user":
                prompt = message.get("content", "")
                break

        if self._is_too_short(text):
            return "too_short"
        if self._is_too_long(text):
            return "too_long"
        if not self._has_structure(text):
            return "no_structure"
        if not self._has_data_reference(text):
            return "no_data"
        if not self._respects_budget(prompt, text):
            return "budget_mismatch"
        return None

    def filter_sft(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter a list of SFT samples and return the ones that pass.

        Updates internal stats accessible via print_report().
        """
        self._sft_stats = FilterStats(total_before=len(samples))
        accepted: List[Dict[str, Any]] = []

        for sample in samples:
            reason = self.check_sft_sample(sample)
            if reason is None:
                accepted.append(sample)
            else:
                if reason == "too_short":
                    self._sft_stats.rejected_too_short += 1
                elif reason == "too_long":
                    self._sft_stats.rejected_too_long += 1
                elif reason == "no_structure":
                    self._sft_stats.rejected_no_structure += 1
                elif reason == "no_data":
                    self._sft_stats.rejected_no_data += 1
                elif reason == "budget_mismatch":
                    self._sft_stats.rejected_budget_mismatch += 1

        self._sft_stats.total_after = len(accepted)
        logger.info(
            "SFT filter: %d -> %d  (rejected %d)",
            self._sft_stats.total_before,
            self._sft_stats.total_after,
            self._sft_stats.total_before - self._sft_stats.total_after,
        )
        return accepted

    # ------------------------------------------------------------------
    # DPO filtering
    # ------------------------------------------------------------------

    def check_dpo_sample(self, sample: Dict[str, Any]) -> Optional[str]:
        """
        Validate a single DPO sample.

        The *chosen* answer must pass quality checks.  The *rejected* answer
        is intentionally low-quality, so we only verify it exists.

        For safety_honesty / service_boundary dimensions the chosen answer may intentionally omit
        concrete data (it says "I don't know"), so the no_data rule is relaxed.

        Returns:
            None if the sample passes, or a rejection reason string.
        """
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")
        dimension = sample.get("dimension", "")
        prompt = sample.get("prompt", "")

        # Reject if chosen is too short (it should be the high-quality one)
        if self._is_too_short(chosen):
            return "too_short"
        if self._is_too_long(chosen):
            return "too_long"

        # Chosen should be structured (rejected is intentionally not)
        if not self._has_structure(chosen):
            return "no_structure"

        # Chosen should have data references (relaxed for safety_honesty)
        relaxed_dimensions = {"safety_honesty", "service_boundary"}
        if dimension not in relaxed_dimensions and not self._has_data_reference(chosen):
            return "no_data"
        if dimension == "information_quality" and not self._respects_budget(prompt, chosen):
            return "budget_mismatch"

        # Rejected must not be empty
        if len(rejected.strip()) < 5:
            return "too_short"

        return None

    def filter_dpo(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter a list of DPO pairs and return those that pass.
        """
        self._dpo_stats = FilterStats(total_before=len(samples))
        accepted: List[Dict[str, Any]] = []

        for sample in samples:
            reason = self.check_dpo_sample(sample)
            if reason is None:
                accepted.append(sample)
            else:
                if reason == "too_short":
                    self._dpo_stats.rejected_too_short += 1
                elif reason == "too_long":
                    self._dpo_stats.rejected_too_long += 1
                elif reason == "no_structure":
                    self._dpo_stats.rejected_no_structure += 1
                elif reason == "no_data":
                    self._dpo_stats.rejected_no_data += 1
                elif reason == "budget_mismatch":
                    self._dpo_stats.rejected_budget_mismatch += 1

        self._dpo_stats.total_after = len(accepted)
        logger.info(
            "DPO filter: %d -> %d  (rejected %d)",
            self._dpo_stats.total_before,
            self._dpo_stats.total_after,
            self._dpo_stats.total_before - self._dpo_stats.total_after,
        )
        return accepted

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def sft_stats(self) -> FilterStats:
        """Access the latest SFT filtering statistics."""
        return self._sft_stats

    @property
    def dpo_stats(self) -> FilterStats:
        """Access the latest DPO filtering statistics."""
        return self._dpo_stats

    def print_report(self) -> None:
        """Log a human-readable filter report."""
        logger.info("=" * 60)
        logger.info("Quality Filter Report")
        logger.info("=" * 60)

        for label, stats in [("SFT", self._sft_stats), ("DPO", self._dpo_stats)]:
            summary = stats.rejection_summary()
            if summary["total_before"] == 0:
                continue
            logger.info("")
            logger.info("[%s Data]", label)
            logger.info("  Before filtering : %d", summary["total_before"])
            logger.info("  After filtering  : %d", summary["total_after"])
            logger.info("  Rejected (total) : %d", summary["rejected_total"])
            logger.info("    - too short    : %d", summary["rejected_too_short"])
            logger.info("    - too long     : %d", summary["rejected_too_long"])
            logger.info("    - no structure : %d", summary["rejected_no_structure"])
            logger.info("    - no data ref  : %d", summary["rejected_no_data"])
            logger.info("    - budget miss  : %d", summary["rejected_budget_mismatch"])
            if summary["total_before"] > 0:
                rate = summary["total_after"] / summary["total_before"] * 100
                logger.info("  Pass rate        : %.1f%%", rate)

        logger.info("=" * 60)
