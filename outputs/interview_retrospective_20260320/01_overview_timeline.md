# 总时间线

## 一句话总览

这条实验线可以分成 4 个阶段：

- 第一阶段是"数据分布与训练稳定性均未就绪"，base 明显更强
- 第二阶段是"结构化 SFT 出现窄场景胜利"，但还不是真正的 failure-first 成功
- 第三阶段是"failure-first SFT 仅针对 base 真实失败的案例做定向优化"，终于做出可信的局部优势
- 第四阶段是"DPO 从修能力改成修偏好"，再配合更真实的 benchmark 和更强 judge，才出现可信提升

## 时间线表

| 时间 | 阶段 | 证据文件 | 关键结果 | 应该怎么解读 |
|---|---|---|---|---|
| 2026-03-16 | 最早总评 | `outputs/model_showcase/benchmark_results.md` | base `0.6321` > sft `0.5476` > dpo `0.4904` | 早期对齐模型整体不如 base，尤其 SFT / DPO 都产生了明显偏离目标的输出。 |
| 2026-03-18 09:27 | regen / targeted 再训 | `outputs/model_showcase/benchmark_results_regen_v3_v4_targeted.md` | base `0.7475` > sft `0.5541` > dpo `0.3865` | 重生数据和 targeted benchmark 还不足以证明收益，base 仍是主力。 |
| 2026-03-18 14:36 | focus SFT partial | `outputs/model_showcase/benchmark_results_focus_sft_partial.md` | base `0.7543` > sft `0.4838` | 过早扩大 SFT 目标面，结果在高风险边界场景里反而更差。 |
| 2026-03-18 19:25 | struct targeted 首次局部翻盘 | `outputs/model_showcase/benchmark_results_qwen25_3b_sft_struct_targeted.md` | sft `0.7816` > base `0.7600` | 这是第一批"部分场景优于 base"的证据，但 benchmark 还偏结构化、偏窄。 |
| 2026-03-18 21:39 | clean + trim 后的结构化 SFT | `outputs/model_showcase/benchmark_results_qwen25_3b_sft_struct_targeted_clean_v1_trimmed.md` | sft `0.7712` > base `0.7600`，6 场景里赢 4 个 | 数据清洗和输出收敛让 SFT 变稳，但成功仍集中在结构化任务。 |
| 2026-03-18 23:57 | failure-first SFT 首版 | `outputs/failure_first_qwen25_3b_v1/benchmark_sft_failure_first.md` | base `0.5033` > sft `0.4685` | 方向对了，但训练和评测链路都没稳，第一次 failure-first 仍然失败。 |
| 2026-03-19 12:14 | failure-first SFT 稳定版起点 | `outputs/failure_first_qwen25_3b_v1/benchmark_sft_failure_first_no_judge_v4.md` | sft `0.6086` > base `0.5638` | 去掉不稳定 judge 之后，SFT 开始出现真实提升。 |
| 2026-03-19 19:18 | failure-first SFT 最佳版 | `outputs/failure_first_qwen25_3b_v1/benchmark_sft_failure_first_no_judge_v5_followupfix_all4bit_cleaned_v2.md` | sft `0.7156` > base `0.4531`，4 场景里赢 3 个 | 这是具有可复现证据支撑的 SFT 最佳版本。 |
| 2026-03-19 21:02 到 23:21 | targeted DPO 连续失败 | `outputs/base_failure_dpo_qwen25_3b_v1/...` 到 `outputs/base_failure_dpo_qwen25_3b_v3_strict_repair/...` | 多次 strict rerun 后 DPO 结果不稳，最终掉到 `0.1638` | 说明"拿 DPO 修基础能力"这条路本身就不对。 |
| 2026-03-20 00:35 | preference-first DPO v1 | `outputs/preference_first_dpo_qwen25_3b_v1/benchmark_preference_first_base_vs_dpo.md` | base `0.1454` = dpo `0.1454` | 把 DPO 改成抽象偏好学习还不够，场景设计不真实，收益几乎为零。 |
| 2026-03-20 16:00 | realistic-service DPO 最佳版 | `outputs/preference_first_dpo_qwen25_3b_v2_realistic_service/benchmark_realistic_service_judge_first_base_vs_dpo_rerun.md` | dpo `0.5595` > base `0.4572`，5 场景里推荐 3 个 | 这才是可信的 DPO 成功：真实客服偏好场景、强 judge、硬 gate、允许人工兜底。 |

## 贯穿全程的主线变化

### 1. 优化目标变了

- 早期是在追"整体平均分"
- 中期是在追"结构化输出是否更整齐"
- 后期才变成"只优化 base 明显失败的子集"

### 2. 方法边界变了

- 早期默认 SFT / DPO 都能补能力
- 后期明确拆成：
- `SFT` 负责低风险流程能力，如简单推荐、followup memory
- `DPO` 负责已有能力上的服务感、说服力、语气、挽留

### 3. 评测思路变了

- 早期更像看单次绝对分
- 后期开始看：
- hard gate
- stronger judge
- pairwise 对比
- case-level routing
- manual_review 兜底

## 核心结论

如果只用一句话概括整个项目：

> 项目最终不是靠"把小模型越训越强"产出结果，而是通过重新定义任务边界、数据边界和评测边界，让 SFT 在低风险流程场景、DPO 在真实客服偏好场景里，各自只解决自己真正擅长的问题，最终在部分场景上稳定优于 base。
