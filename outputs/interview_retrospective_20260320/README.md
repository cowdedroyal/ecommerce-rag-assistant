# 实验复盘与分析 — 文档索引

这个目录的目标不是证明 "SFT / DPO 整体打赢了 base"，而是更准确地回答 3 个问题：

- 早期大多数实验为什么失败
- 为什么最后只有"部分场景"优于 base
- 这些失败和成功经验的核心结论与表述方式

当前具有可复现证据支撑的核心结论只有 2 条：

- SFT 的成功是 `failure-first` 子集里的局部成功，不是全量场景通杀。最佳证据是 `outputs/failure_first_qwen25_3b_v1/benchmark_sft_failure_first_no_judge_v5_followupfix_all4bit_cleaned_v2.md`，SFT 平均分 `0.7156`，base `0.4531`，4 个场景里赢了 3 个。
- DPO 的成功是"真实客服偏好场景"的局部成功，不是补齐基础能力。最佳证据是 `outputs/preference_first_dpo_qwen25_3b_v2_realistic_service/benchmark_realistic_service_judge_first_base_vs_dpo_rerun.md`，DPO 平均分 `0.5595`，base `0.4572`，5 个场景里推荐了 3 个。

同时也要明确说清：

- 高风险场景并没有被强行放开，而是继续走 `manual_review`
- 早期很多"模型失败"其实是数据分布、训练稳定性和评测链路一起失败
- 真正让结果变好的，不是多训几轮，而是把任务边界、数据边界、评测边界都收紧了

文件说明：

| 文件 | 内容 |
|---|---|
| `01_overview_timeline.md` | 整个项目从 2026-03-16 到 2026-03-20 的总时间线 |
| `02_sft_retrospective.md` | SFT 版本迭代、失败原因、成功证据 |
| `03_dpo_retrospective.md` | DPO 版本迭代、失败原因、成功证据 |
| `04_common_failure_patterns.md` | 跨 SFT / DPO 的共性失败模式 |
| `05_interview_storylines.md` | 核心叙事与常见追问 |
| `06_project_takeaways.md` | 项目心得、简历可写表述、回顾性改进建议 |
| `07_interview_2min_script.md` | 可直接用于表述的 2 分钟项目陈述稿和 90 秒压缩版 |
| `08_rag_technical_review.md` | RAG 检索与生成链路技术复盘 |
