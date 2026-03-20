# 电商 RAG Assistant

面向中文电商问答场景的 RAG 系统，集成 SFT / DPO 小模型对齐实验。核心方法论：通过 failure-first 选样、任务边界拆分和分层评测，在局部场景实现小模型对 base 的稳定超越。

## 核心成果速览

| 方法                    | Benchmark        |                      最佳结果 | 场景胜出        | 最佳证据                                                                      |
| ----------------------- | ---------------- | ----------------------------: | --------------- | ----------------------------------------------------------------------------- |
| SFT (failure-first)     | 低风险流程场景   | `0.7156` vs base `0.4531` | 4 场景赢 3 个   | `benchmark_sft_failure_first_no_judge_v5_followupfix_all4bit_cleaned_v2.md` |
| DPO (realistic-service) | 真实客服偏好场景 | `0.5595` vs base `0.4572` | 5 场景推荐 3 个 | `benchmark_realistic_service_judge_first_base_vs_dpo_rerun.md`              |

高风险场景（转品、议价）保留 `manual_review` 路由，未强行放开。

## 技术栈

| 类别        | 技术                                                            |
| ----------- | --------------------------------------------------------------- |
| LLM / 训练  | Qwen2.5-7B/3B-Instruct (4bit)、LoRA (r=64, alpha=128)、SFT、DPO |
| 嵌入 / 重排 | BAAI/bge-base-zh-v1.5 (768d)、BAAI/bge-reranker-base            |
| 检索        | BM25 + Dense Retrieval → RRF 融合 → Rerank → Top-K           |
| 后端        | FastAPI、Pydantic Settings                                      |
| 前端        | Vue 3、Pinia、Vite                                              |
| 数据        | 运行时合成演示数据 (seed=42)、爬虫采集                          |

## 系统架构

```text
┌─────────────┐     ┌──────────────────────────────────────────────┐
│  Vue 3 前端  │────▶│              FastAPI 后端                     │
│  Chat / 商品 │◀────│  /chat/send  /chat/stream  /products /orders │
└─────────────┘     └───────────────────┬──────────────────────────┘
                                        │
                    ┌───────────────────▼───────────────────┐
                    │            RAG Pipeline                │
                    │                                        │
                    │  Query ──▶ BM25 ──┐                    │
                    │            Dense ──┤── RRF ──▶ Rerank  │
                    │                   └──────────▶ Top-K   │
                    │                                        │
                    │  Context + History ──▶ LLM Generate    │
                    │  (Qwen2.5-7B / SFT / DPO)             │
                    └────────────────────────────────────────┘
                                        │
                    ┌───────────────────▼───────────────────┐
                    │         Training Pipeline              │
                    │                                        │
                    │  Data Gen ──▶ SFT ──▶ DPO              │
                    │  Failure-first ──▶ Targeted Benchmark  │
                    │  Hard Gate + Stronger Judge + Routing   │
                    └────────────────────────────────────────┘
```

## 快速开始

### 前置条件

- Python 3.10+
- NVIDIA GPU (推荐 A100 40GB；4bit 量化下 RTX 3090 24GB 可运行推理)
- Node.js 18+ / pnpm
- Qwen2.5-7B/3B-Instruct 模型权重 (自动从 HuggingFace 下载)

### 1. 安装依赖

```bash
pip install -r requirements.txt
cd frontend && pnpm install
```

### 2. 生成演示数据

```bash
python scripts/run_pipeline.py bootstrap-demo
```

### 3. 启动后端

```bash
python scripts/run_pipeline.py serve
```

### 4. 启动前端

```bash
cd frontend && pnpm dev
```

命令行对话模式：

```bash
python scripts/chat.py
```

## 训练与评测

### 基础工作流

```bash
# 数据生成
python scripts/run_pipeline.py generate-data --sft-count 3000 --dpo-count 1000

# 训练
python scripts/run_pipeline.py train-sft
python scripts/run_pipeline.py train-dpo

# 通用 benchmark
python scripts/run_pipeline.py evaluate --stages base,sft,dpo --output outputs/model_showcase/benchmark_results.json
```

### 研究脚本

| 功能                     | 脚本                                                                                                                                                                                    |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Benchmark**      | `benchmark_failure_first_sft.py`、`benchmark_failure_first_dpo.py`、`benchmark_preference_first_dpo.py`、`benchmark_realistic_service_dpo.py`                                   |
| **数据计划与构建** | `build_data_plan.py`、`build_failure_first_sft_plan.py`、`build_preference_first_dpo_plan.py`、`build_targeted_failure_first_dpo_data.py`                                       |
| **数据导出**       | `export_failure_first_sft_data.py`、`export_failure_first_dpo_data.py`、`export_preference_first_dpo_data.py`、`export_realistic_service_dpo_data.py`、`export_focus_data.py` |
| **训练**           | `train_failure_first_sft.py`、`train_failure_first_dpo.py`、`train_demo_models.py`、`train_focus_models.py`                                                                     |
| **评测修复**       | `repair_benchmark_judging.py`、`repair_pairwise_report.py`                                                                                                                          |
| **辅助**           | `summarize_focus_data.py`、`crawl_data.py`                                                                                                                                          |

这些脚本服务于定向实验和 failure-first 复盘，不是最小可运行 demo 的必需部分。

## 目录结构

```text
ecommerce-rag-assistant/
├── crawler/                    # 爬虫与清洗 (JD/淘宝商品数据采集)
├── src/
│   ├── api/                    # FastAPI 应用与 endpoints (chat/products/orders)
│   ├── data/                   # 运行时合成数据生成 (seed 可复现)
│   └── rag/                    # 检索、重排、对话管理、生成清理
│       ├── assistant.py        # RAG 主控：检索→重排→生成
│       ├── retriever.py        # BM25 + Dense 混合检索
│       ├── reranker.py         # BGE-Reranker 精排
│       ├── conversation.py     # 多轮对话历史管理
│       └── generation_cleaner.py  # 生成输出后处理
├── training/                   # SFT / DPO 训练入口与数据生成
│   └── data_gen/               # 训练数据合成模块
├── evaluation/                 # benchmark、检索评测、生成评测、场景 taxonomy
├── scripts/                    # CLI、训练、评测、数据计划脚本
├── frontend/                   # Vue 3 前端 (ChatView / ProductCard / RetrievalPanel)
├── data/                       # 本地数据目录 (raw/processed/crawled/training)
└── outputs/                    # 实验输出目录 (默认不入库)
```

## 检索链路

系统采用混合检索 + 精排的两阶段架构：

1. **BM25 召回** — 基于 jieba 分词的稀疏检索，Top-20 候选
2. **Dense 召回** — BGE-base-zh-v1.5 向量检索，Top-20 候选
3. **RRF 融合** — Reciprocal Rank Fusion (k=60) 合并两路召回结果，Top-20
4. **Rerank 精排** — BGE-Reranker-base 对融合结果做交叉编码精排
5. **Top-K 截断** — 取精排后 Top-5 作为 LLM 生成上下文

检索配置集中管理于 `src/config.py`，支持通过环境变量覆盖。

## 实验复盘与分析

详细的实验迭代记录和方法论总结在 `outputs/interview_retrospective_20260320/` 目录下：

| 文件                              | 内容                             |
| --------------------------------- | -------------------------------- |
| `01_overview_timeline.md`       | 总时间线与阶段划分               |
| `02_sft_retrospective.md`       | SFT 版本迭代、失败原因、成功证据 |
| `03_dpo_retrospective.md`       | DPO 版本迭代、失败原因、成功证据 |
| `04_common_failure_patterns.md` | 跨 SFT / DPO 的共性失败模式      |
| `05_interview_storylines.md`    | 核心叙事与常见追问               |
| `06_project_takeaways.md`       | 项目心得与回顾性改进建议         |
| `07_interview_2min_script.md`   | 2 分钟项目陈述稿                 |
| `08_rag_technical_review.md`  | RAG 检索与生成链路技术复盘        |

如果本仓库用于技术展示，这组文档比大体积 checkpoint 更值得保留和提交。
