# RAG 检索与生成链路技术复盘

## 一句话结论

本项目 RAG 链路的核心设计选择是"混合检索 + 交叉编码精排 + 生成后处理"三层架构。检索层通过 BM25 和 Dense 双路召回配合 RRF 融合兼顾精确匹配与语义泛化；精排层用交叉编码器对 query-doc 对做联合编码，弥补双塔模型的交互不足；生成后处理层则在小模型输出质量不稳定的前提下，通过规则清洗保证最终交付质量。

## 架构总览

```text
用户查询
  │
  ├──▶ BM25 (jieba 分词, Top-20)
  │
  ├──▶ Dense (bge-base-zh-v1.5, cosine, Top-20)
  │
  └──▶ RRF 融合 (k=60, Top-20)
         │
         ▼
  CrossEncoder 精排 (bge-reranker-base, Top-5)
         │
         ▼
  多轮对话上下文组装 (system + history + context + query)
         │
         ▼
  LLM 生成 (Qwen2.5-7B/3B, 4bit, LoRA adapter 可选)
         │
         ▼
  生成后处理 (generation_cleaner: 截断/去噪/去重/补句)
         │
         ▼
  最终响应
```

## 检索层设计

### 为什么选择双路混合检索

| 检索方式 | 强项 | 弱项 |
|---|---|---|
| BM25 (稀疏) | 精确关键词匹配、品牌名/型号/SKU 精准命中 | 无法处理同义改写和语义泛化 |
| Dense (稠密) | 语义相似性、"通勤耳机"→ 蓝牙降噪类商品 | 对精确术语和数字不敏感 |
| Hybrid + RRF | 互补两路召回 | 需要调 RRF k 参数 |

电商场景天然存在两类查询：

- **精确查询**："索尼 WH-1000XM5 价格" — BM25 强项
- **模糊需求**："通勤用的安静耳机 500 以内" — Dense 强项

单独依赖任一路都会在另一类查询上丢失召回率，因此混合检索是该场景下的必要选择。

### BM25 实现细节

```python
# 关键技术点：jieba 分词 + 停用词/单字过滤
tokens = jieba.lcut(text)
return [t for t in tokens if t.strip() and (len(t) > 1 or "\u4e00" <= t <= "\u9fff")]
```

分词策略直接影响 BM25 质量。项目采用 jieba 默认词典，过滤单字非 CJK token，保留有意义的中文单字（如品类字"鞋""机"）。

**已知限制**：未引入领域自定义词典。电商场景中"机械键盘""蓝牙耳机"等复合词依赖 jieba 默认切分，偶尔会被错误拆分为"机械"+"键盘"，导致召回偏移。

### Dense 实现细节

- 模型：`BAAI/bge-base-zh-v1.5`（768 维）
- 编码：L2 归一化后用内积近似余弦相似度（`normalize_embeddings=True`）
- 索引：全量 `np.dot` 暴力检索（文档规模 < 1K 时无需 ANN）

**设计权衡**：项目商品库规模在数百量级，暴力检索延迟可控（< 10ms），因此没有引入 FAISS 或 Milvus。如果商品规模扩展到 10K+，需要引入 ANN 索引（HNSW 或 IVF-PQ）。

### RRF 融合

```python
# score = sum(1 / (k + rank_i)) across all result lists
fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
```

- 参数 `k=60` 是标准默认值，控制排名靠后文档的得分衰减
- RRF 的优势在于不需要对两路分数做归一化（BM25 分数和余弦相似度量纲不同），直接基于排名融合

### 检索模式降级

项目实现了三种检索模式（`hybrid` / `dense` / `bm25`），并支持自动降级：

- Dense 模型加载失败时，自动回退到纯 BM25
- 请求指定 `dense` 模式但 Dense 不可用时，回退到 BM25-fallback
- 这种容错设计确保系统在 GPU 不可用的环境下仍能提供基础服务

## 精排层设计

### 为什么需要交叉编码精排

双塔模型（BM25 + Dense）的根本局限是 query 和 document 独立编码，无法捕捉细粒度的 query-doc 交互。交叉编码器将 query 和 document 拼接后做联合编码，能更准确地判断相关性。

| 阶段 | 模型 | 输入 | 候选量 | 延迟 |
|---|---|---|---:|---|
| 召回 | BM25 + Dense | query / doc 分别处理 | 全库 → Top-20 | < 20ms |
| 精排 | bge-reranker-base | [query, doc] 联合编码 | Top-20 → Top-5 | ~50ms |

### 实现要点

- 使用 `AutoModelForSequenceClassification` 加载，输出 logits 作为相关性分数
- 分批处理（batch_size=32），避免显存溢出
- 保留原始检索分数（`original_score`），精排分数单独存储（`rerank_score`），便于前端可视化对比
- 精排器不可用时，直接使用检索排名作为 fallback

### 精排对最终结果的影响

精排器解决的核心问题是：BM25 和 Dense 各自 Top-20 的交集往往不在最前面。RRF 融合后的排名可能把一个"BM25 第 15 名 + Dense 第 3 名"的文档排到前 5，但它不一定是 query 最相关的。交叉编码器能在 Top-20 候选内做更精确的重排序。

## 多轮对话管理

### 设计选择

- 使用 `deque(maxlen=max_turns * 2)` 管理历史，自动丢弃最早的对话轮次
- 默认保留最近 5 轮（10 条消息），平衡上下文长度和 token 预算
- 每轮对话的 RAG prompt 结构：`system (含检索上下文) → history → current query`

### 上下文组装策略

```text
System: 你是一个专业的电商购物助手...
        [检索到的商品信息]
        [1] 索尼 WH-1000XM5
            Price: 2499.00
            Rating: 4.8
            Description: ...

History: [之前的对话轮次]

User: [当前查询]
```

检索上下文嵌入 system message 而非 user message，这样做的好处是：
- 模型更倾向于将检索结果视为"参考资料"而非用户指令
- 减少 prompt injection 风险
- 与 Qwen2.5 的 chat template 对齐

## 生成后处理

### 为什么需要 generation_cleaner

小模型（尤其是 3B 量化后）在生成时会出现以下问题：

| 问题类型 | 表现 | 清洗策略 |
|---|---|---|
| 角色泄漏 | 输出中混入 `user` / `assistant` 标记 | 正则检测并截断 |
| 噪声 token | 出现 `rumpe`、`spep`、乱码字符 | 硬停止词 + 内联噪声过滤 |
| 重复生成 | 连续重复相同句子或行 | 逐行/逐句去重 |
| 截断不完整 | 句子在中间断开 | 补句号 |
| 结构化结束标记 | "以上为本次推荐结果。"后继续输出 | 在结束标记处截断 |
| Markdown 链接 | 生成虚假 URL | 剥离链接保留文本 |

### 清洗流水线（执行顺序）

```text
原始输出
  → trim_at_explicit_end_markers    # 在结构化结束标记处截断
  → trim_at_hard_stop_markers       # 在噪声硬停止词处截断
  → trim_leaked_roles               # 去除角色泄漏
  → strip_inline_noise              # 内联噪声 token 过滤
  → strip_markdown_links            # 剥离虚假链接
  → normalize_recommendation_prices # 价格格式统一
  → normalize_numbered_recommendation_lines  # 推荐列表格式统一
  → dedupe_consecutive_lines        # 连续重复行去重
  → dedupe_repeated_sentences       # 句内重复句子去重
  → ensure_service_intro            # 推荐列表补充开场白
  → expand_truncated_recommendation # 截断推荐补全
  → finish_sentence                 # 不完整句子补句号
```

这套清洗流水线的设计原则是**保守截断、不改语义**。只做格式修复和截断，不尝试改写模型输出的内容本身。

## 查询理解与路由

### 查询分类

系统在进入检索之前先做查询分类：

| 查询类型 | 检测方式 | 处理方式 |
|---|---|---|
| 闲聊/问候 | 关键词 + 短语匹配 | 跳过检索，直接返回模板响应 |
| 订单查询 | 关键词匹配（"订单""物流"等） | 走订单数据查询，不走商品检索 |
| 高优先级订单 | "高优先级"关键词 | 直接查询高优先级订单 |
| 商品查询 | 默认 | 走完整 RAG 链路 |

### 评分过滤

支持从查询中提取评分过滤条件：
- "4.5 分以上的耳机" → `min_rating=4.5`
- "above 4.0" → `min_rating=4.0`

这类结构化约束在检索后、返回前生效，避免在检索阶段引入偏差。

## 检索评测体系

项目实现了标准的检索质量评测指标集：

| 指标 | 含义 | 评测粒度 |
|---|---|---|
| MRR | 第一个相关文档的排名倒数的均值 | query 级 |
| Recall@K | 前 K 个结果中相关文档的召回比例 | query × K 级 |
| Precision@K | 前 K 个结果中相关文档的精确率 | query × K 级 |
| NDCG@K | 考虑排名位置的归一化折损累计增益 | query × K 级 |

评测在 K = {1, 3, 5, 10, 20} 五个截断点运行，能同时反映头部精度和长尾召回的表现。

## 前端可视化与 API 设计

### 检索过程透明化

API 返回完整的检索中间过程，前端可展示每个阶段的候选及分数：

```json
{
  "retrieval_process": {
    "bm25_results": [{"id": "P001", "score": 12.34, "title": "..."}],
    "dense_results": [{"id": "P002", "score": 0.89, "title": "..."}],
    "reranked_results": [{"id": "P002", "score": 6.12, "title": "..."}],
    "summary": {
      "requested_mode": "hybrid",
      "selected_mode": "hybrid",
      "dense_available": true,
      "reranker_enabled": true
    }
  }
}
```

这个设计服务于两个目的：
- **调试**：能直观看到每个阶段的排名变化，定位检索质量问题
- **展示**：前端 `RetrievalPanel` 组件可以可视化对比 BM25 / Dense / Reranked 的排序差异

### SSE 流式响应

`/chat/stream` 端点将响应拆分为 4 个事件阶段：
1. `retrieval` — 检索完成，推送中间过程
2. `token` — 逐块推送生成文本
3. `products` — 推送推荐商品卡片
4. `done` — 流结束

分阶段推送让前端可以在检索完成时立即展示检索面板，无需等待生成完成。

## 关键设计权衡总结

| 决策 | 选择 | 替代方案 | 选择理由 |
|---|---|---|---|
| 检索方式 | BM25 + Dense + RRF | 纯 Dense | 电商场景精确词和语义需求并存 |
| 向量索引 | 暴力内积 | FAISS HNSW | 商品库 < 1K，暴力检索延迟可控 |
| 精排模型 | bge-reranker-base | 无精排 | 双塔召回排名粗糙，交叉编码显著提升头部精度 |
| 分词 | jieba 默认 | 领域词典 | 开发速度优先；领域词典是已知改进方向 |
| 上下文注入位置 | system message | user message | 减少 prompt injection 风险，与 chat template 对齐 |
| 生成后处理 | 规则清洗 | 无 | 3B 量化模型输出质量不稳定，规则清洗成本低、效果确定 |
| 对话历史管理 | 固定窗口 deque | 摘要压缩 | 实现简单，5 轮窗口在客服场景下足够覆盖单次咨询 |

## 已知限制与改进方向

1. **jieba 分词精度** — 未引入电商领域自定义词典，复合品类词可能被错误拆分
2. **无 query rewriting** — 多轮对话中的代词消解和省略补全依赖 LLM 自身能力，未做显式 query 改写
3. **无缓存层** — 相同查询每次都走完整检索链路，高频查询场景下存在重复计算
4. **向量索引扩展性** — 商品规模超过 10K 后需引入 ANN 索引
5. **精排延迟** — 交叉编码器对 Top-20 候选的推理延迟约 50ms，实时性要求极高的场景可能需要蒸馏或量化
6. **检索评测覆盖** — 目前评测框架已实现，但缺少系统性的 ground truth 标注数据集
