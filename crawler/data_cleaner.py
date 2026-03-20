"""
Data cleaner: merges, deduplicates, normalises, and validates crawled product
data from multiple sources into a single CSV ready for the RAG pipeline.

Input:  data/crawled/*.json  (one file per source)
Output: data/processed/products_zh.csv
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Expected output columns (order matters for CSV)
OUTPUT_COLUMNS = [
    "product_id",
    "title",
    "brand",
    "category",
    "price",
    "rating",
    "review_count",
    "description",
    "specs",
    "features",
    "source",
]


class DataCleaner:
    """
    Merge, deduplicate, clean, and normalise product records from one or more
    crawled JSON files into a single pandas DataFrame / CSV.

    Usage::

        cleaner = DataCleaner(
            input_dir=Path("data/crawled"),
            output_path=Path("data/processed/products_zh.csv"),
        )
        df = cleaner.run()
    """

    def __init__(
        self,
        input_dir: Path,
        output_path: Path,
        similarity_threshold: float = 0.85,
    ):
        self.input_dir = Path(input_dir)
        self.output_path = Path(output_path)
        self.similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_json_files(self) -> List[Dict[str, Any]]:
        """Read all ``*.json`` files in the input directory."""
        all_records: List[Dict[str, Any]] = []
        json_files = sorted(self.input_dir.glob("*.json"))

        if not json_files:
            logger.warning("No JSON files found in %s", self.input_dir)
            return all_records

        for fp in json_files:
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    all_records.extend(data)
                    logger.info("Loaded %d records from %s", len(data), fp.name)
                else:
                    logger.warning("Unexpected JSON structure in %s (expected list)", fp.name)
            except (json.JSONDecodeError, IOError) as exc:
                logger.error("Failed to read %s: %s", fp, exc)

        logger.info("Total raw records loaded: %d", len(all_records))
        return all_records

    # ------------------------------------------------------------------
    # Field-level cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_price(value: Any) -> Optional[float]:
        """Normalise price to float; return None for invalid values."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return round(float(value), 2) if value > 0 else None
        text = str(value)
        text = re.sub(r"[^\d.]", "", text.split("-")[0])
        try:
            price = float(text)
            return round(price, 2) if price > 0 else None
        except ValueError:
            return None

    @staticmethod
    def _clean_rating(value: Any) -> Optional[float]:
        """Normalise rating to a 0-5 float scale."""
        if value is None:
            return None
        try:
            r = float(value)
        except (ValueError, TypeError):
            return None
        # If rating looks like a percentage (>5), scale to 0-5
        if r > 5:
            r = round(r / 100 * 5, 1) if r <= 100 else None
        elif r < 0:
            r = None
        else:
            r = round(r, 1)
        return r

    @staticmethod
    def _clean_review_count(value: Any) -> Optional[int]:
        """Normalise review/trade count to int."""
        if value is None:
            return None
        text = str(value).replace("+", "").replace("万", "0000").replace(",", "")
        digits = re.sub(r"[^\d]", "", text)
        try:
            return int(digits) if digits else None
        except ValueError:
            return None

    @staticmethod
    def _clean_text(value: Any) -> str:
        """Strip and collapse whitespace."""
        if not value:
            return ""
        text = str(value).strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _serialise_field(value: Any) -> str:
        """Convert list/dict fields to JSON strings for CSV storage."""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        if not value:
            return ""
        return str(value)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate products based on:
          1. Exact product_id match (same source).
          2. Title similarity above threshold (cross-source).
        """
        before = len(df)

        # Phase 1: exact product_id + source dedup
        df = df.drop_duplicates(subset=["product_id", "source"], keep="first")

        # Phase 2: title-based fuzzy dedup (character-level Jaccard)
        keep_mask = [True] * len(df)
        titles = df["title"].tolist()

        for i in range(len(titles)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(titles)):
                if not keep_mask[j]:
                    continue
                if self._title_similarity(titles[i], titles[j]) >= self.similarity_threshold:
                    keep_mask[j] = False

        df = df[keep_mask].reset_index(drop=True)
        after = len(df)

        logger.info("Deduplication: %d → %d records (%d removed)", before, after, before - after)
        return df

    @staticmethod
    def _title_similarity(a: str, b: str) -> float:
        """Character-level Jaccard similarity between two titles."""
        if not a or not b:
            return 0.0
        set_a = set(a)
        set_b = set(b)
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid(row: pd.Series) -> bool:
        """Return True if the record passes minimum quality checks."""
        # Must have a non-empty title
        if not row.get("title") or len(str(row["title"]).strip()) < 2:
            return False
        # Must have a positive price (allow None — some products legitimately lack it)
        if row.get("price") is not None and row["price"] <= 0:
            return False
        return True

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute the full cleaning pipeline:

        1. Load all JSON files from the crawled data directory.
        2. Normalise and clean each field.
        3. Deduplicate.
        4. Filter invalid records.
        5. Save to CSV.

        Returns the cleaned DataFrame.
        """
        records = self._load_json_files()
        if not records:
            logger.warning("No records to clean; producing empty output.")
            df = pd.DataFrame(columns=OUTPUT_COLUMNS)
            self._save(df)
            return df

        df = pd.DataFrame(records)
        logger.info("Raw DataFrame shape: %s", df.shape)

        # --- Ensure all expected columns exist ---
        for col in OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = None

        # --- Clean individual fields ---
        df["title"] = df["title"].apply(self._clean_text)
        df["brand"] = df["brand"].apply(self._clean_text)
        df["category"] = df["category"].apply(self._clean_text)
        df["description"] = df["description"].apply(self._clean_text)
        df["price"] = df["price"].apply(self._clean_price)
        df["rating"] = df["rating"].apply(self._clean_rating)
        df["review_count"] = df["review_count"].apply(self._clean_review_count)
        df["specs"] = df["specs"].apply(self._serialise_field)
        df["features"] = df["features"].apply(self._serialise_field)
        df["source"] = df["source"].apply(self._clean_text)

        # Generate product_id where missing
        df["product_id"] = df.apply(
            lambda r: r["product_id"] if r.get("product_id") else f"{r['source']}_{r.name}",
            axis=1,
        )

        # --- Deduplicate ---
        df = self._deduplicate(df)

        # --- Filter invalid rows ---
        valid_mask = df.apply(self._is_valid, axis=1)
        removed = (~valid_mask).sum()
        df = df[valid_mask].reset_index(drop=True)
        logger.info("Filtered %d invalid records; %d remain", removed, len(df))

        # --- Select and order columns ---
        df = df[OUTPUT_COLUMNS]

        # --- Save ---
        self._save(df)
        return df

    def _save(self, df: pd.DataFrame) -> None:
        """Write DataFrame to CSV."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False, encoding="utf-8-sig")
        logger.info(
            "Saved %d cleaned products to %s", len(df), self.output_path
        )
