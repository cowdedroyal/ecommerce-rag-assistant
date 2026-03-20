#!/usr/bin/env python3
"""
Data preprocessing script for Chinese E-commerce RAG system.
Supports both English raw data and Chinese crawled data.
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data import ensure_runtime_data, load_runtime_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_datasets(
    product_path: Path, order_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and perform initial cleaning of datasets."""
    logger.info("Loading datasets...")

    product_df = pd.read_csv(product_path)
    order_df = pd.read_csv(order_path)

    logger.info(f"Product columns: {product_df.columns.tolist()}")
    logger.info(f"Order columns: {order_df.columns.tolist()}")

    product_df.fillna("", inplace=True)
    order_df.fillna("", inplace=True)

    return product_df, order_df


def preprocess_product_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess product dataset (supports both raw English and crawled Chinese)."""
    logger.info("Preprocessing product data...")

    # Detect format and normalize columns
    if "title" in df.columns and "Product_Title" not in df.columns:
        # Raw English format
        df["combined_text"] = (
            df["title"].str.strip()
            + " "
            + df["description"].str.strip()
            + " "
            + df.get("main_category", pd.Series([""] * len(df))).str.strip()
        )
        df["Price"] = pd.to_numeric(df["price"], errors="coerce")
        df["Rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
        df["Rating_Count"] = pd.to_numeric(df["rating_number"], errors="coerce")

        df = df.rename(
            columns={
                "title": "Product_Title",
                "main_category": "Category",
                "description": "Description",
                "store": "Store",
                "parent_asin": "Product_ID",
            }
        )
    elif "product_id" in df.columns:
        # Chinese crawled format
        df["combined_text"] = (
            df["title"].str.strip()
            + " "
            + df.get("brand", pd.Series([""] * len(df))).str.strip()
            + " "
            + df["category"].str.strip()
            + " "
            + df.get("description", pd.Series([""] * len(df))).str.strip()
        )
        df["Price"] = pd.to_numeric(df["price"], errors="coerce")
        df["Rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df["Rating_Count"] = pd.to_numeric(df.get("review_count", 0), errors="coerce")

        df = df.rename(
            columns={
                "product_id": "Product_ID",
                "title": "Product_Title",
                "category": "Category",
                "description": "Description",
                "brand": "Brand",
            }
        )
    else:
        # Already in processed format
        if "combined_text" not in df.columns:
            df["combined_text"] = (
                df.get("Product_Title", pd.Series([""] * len(df))).str.strip()
                + " "
                + df.get("Description", pd.Series([""] * len(df))).str.strip()
            )

    # Clean and validate
    df = df[df["Price"].notna() & (df["Price"] > 0)]
    df = df[df["Rating"].notna() & df["Rating"].between(0, 5)]

    if "Product_ID" not in df.columns:
        df["Product_ID"] = range(1, len(df) + 1)

    logger.info(f"Products after cleaning: {len(df)}")
    return df


def preprocess_order_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess order dataset."""
    logger.info("Preprocessing order data...")

    if "Order_DateTime" not in df.columns:
        if "Order_Date" in df.columns and "Time" in df.columns:
            df["Order_DateTime"] = pd.to_datetime(
                df["Order_Date"] + " " + df["Time"], errors="coerce"
            )

    numeric_columns = ["Sales", "Quantity", "Discount", "Profit", "Shipping_Cost"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Sales"].notna() & (df["Sales"] > 0)]

    if "Order_ID" not in df.columns:
        df["Order_ID"] = range(1, len(df) + 1)

    # Standardize categorical fields
    for col in ["Order_Priority", "Payment_method", "Customer_Login_type", "Gender", "Device_Type"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()

    if "Sales" in df.columns and "Quantity" in df.columns:
        df["Total_Amount"] = df["Sales"] * df["Quantity"]
    if "Profit" in df.columns and "Shipping_Cost" in df.columns:
        df["Net_Profit"] = df["Profit"] - df["Shipping_Cost"]

    logger.info(f"Orders after cleaning: {len(df)}")
    return df


def create_embeddings(
    df: pd.DataFrame, model_name: str = "BAAI/bge-base-zh-v1.5"
) -> np.ndarray:
    """Create embeddings for product descriptions using the specified model."""
    logger.info(f"Creating embeddings using {model_name}...")

    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            df["combined_text"].tolist(),
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
        )
    except Exception as exc:
        logger.warning("Embedding model unavailable, skip embedding generation: %s", exc)
        embeddings = np.zeros((len(df), 0), dtype=np.float32)

    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def save_processed_data(
    product_df: pd.DataFrame,
    order_df: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: Path,
    zh_mode: bool = False,
):
    """Save processed datasets and embeddings."""
    logger.info("Saving processed data...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    product_filename = "products_zh.csv" if zh_mode else "processed_products.csv"
    product_df.to_csv(output_dir / product_filename, index=False)
    order_df.to_csv(output_dir / "processed_orders.csv", index=False)

    # Save embeddings
    embed_filename = "product_embeddings_zh.pkl" if zh_mode else "product_embeddings.pkl"
    with open(output_dir / embed_filename, "wb") as f:
        pickle.dump(embeddings, f)

    # Save metadata
    info = {
        "timestamp": datetime.now().isoformat(),
        "product_count": len(product_df),
        "order_count": len(order_df),
        "embedding_shape": str(embeddings.shape),
        "product_columns": product_df.columns.tolist(),
    }
    with open(output_dir / "preprocessing_info.txt", "w", encoding="utf-8") as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")

    logger.info(f"Saved {len(product_df)} products and {len(order_df)} orders to {output_dir}")


def main():
    """Main preprocessing pipeline."""
    from src.config import Settings

    settings = Settings()
    processed_dir = settings.PROCESSED_DATA_DIR
    ensure_runtime_data(settings=settings)

    try:
        product_df, order_df = load_runtime_data(settings=settings)
        zh_mode = True

        # Use appropriate embedding model
        model_name = settings.EMBEDDING_MODEL
        embeddings = create_embeddings(product_df, model_name)

        save_processed_data(product_df, order_df, embeddings, processed_dir, zh_mode)
        logger.info("Preprocessing completed successfully!")

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    main()
