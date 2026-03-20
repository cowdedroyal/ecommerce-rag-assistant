from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """应用全局配置"""

    # API 设置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_BASE_URL: str = "http://localhost:8000"

    # 数据路径
    DATA_DIR: Path = Path(__file__).parent.parent / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    CRAWLED_DATA_DIR: Path = DATA_DIR / "crawled"
    TRAINING_DATA_DIR: Path = DATA_DIR / "training"

    # 产品/订单数据路径
    PRODUCT_DATA_PATH: Path = PROCESSED_DATA_DIR / "products_zh.csv"
    ORDER_DATA_PATH: Path = PROCESSED_DATA_DIR / "processed_orders.csv"
    RAW_PRODUCT_DATA_PATH: Path = RAW_DATA_DIR / "Product_Information_Dataset.csv"
    RAW_ORDER_DATA_PATH: Path = RAW_DATA_DIR / "Order_Data_Dataset.csv"
    CATALOG_SUMMARY_PATH: Path = PROCESSED_DATA_DIR / "catalog_summary.json"

    # 合成数据配置
    SYNTHETIC_DATA_SEED: int = 42
    SYNTHETIC_PRODUCTS_PER_CATEGORY: int = 12
    SYNTHETIC_ORDER_COUNT: int = 240

    # 嵌入模型配置
    EMBEDDING_MODEL: str = "BAAI/bge-base-zh-v1.5"
    EMBEDDING_DIM: int = 768

    # 重排序模型配置
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_TOP_K: int = 5

    # LLM 配置
    LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    LLM_MAX_NEW_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 0.9
    USE_4BIT: bool = True

    # 检索配置
    DENSE_TOP_K: int = 20
    BM25_TOP_K: int = 20
    HYBRID_TOP_K: int = 20
    FINAL_TOP_K: int = 5
    RRF_K: int = 60  # RRF 融合参数

    # 对话配置
    MAX_HISTORY_TURNS: int = 5
    SYSTEM_PROMPT: str = (
        "你是一个专业的电商购物助手。根据用户的需求，基于检索到的商品信息，"
        "提供准确、有帮助的购物建议。回答要结构化、信息丰富，并基于实际数据。"
        "如果检索结果中没有相关信息，请坦诚告知用户。"
    )

    # 训练配置
    SFT_OUTPUT_DIR: str = "outputs/sft"
    DPO_OUTPUT_DIR: str = "outputs/dpo"
    MERGED_MODEL_DIR: str = "outputs/merged"

    # 爬虫配置
    CRAWL_DELAY_MIN: float = 2.0
    CRAWL_DELAY_MAX: float = 5.0
    CRAWL_MAX_RETRIES: int = 3
    CRAWL_CATEGORIES: list = [
        "手机", "笔记本电脑", "耳机", "智能手表", "护肤品",
        "运动鞋", "背包", "零食", "咖啡", "键盘",
        "鼠标", "显示器", "平板电脑", "相机", "音箱",
    ]

    # 开发配置
    DEBUG: bool = True
    RELOAD: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# 训练超参数（独立于 Settings，供 training/ 模块使用）
class SFTConfig:
    """SFT 训练超参数 (A100 40GB)"""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "outputs/sft"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_steps: int = 100
    fp16: bool = False
    bf16: bool = True


class DPOConfig:
    """DPO 训练超参数"""
    model_name: str = "outputs/sft"  # SFT 模型作为起点
    output_dir: str = "outputs/dpo"
    beta: float = 0.1
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    use_4bit: bool = True
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 2
    learning_rate: float = 5e-5
    max_length: int = 2048
    max_prompt_length: int = 1024
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 50
    bf16: bool = True
