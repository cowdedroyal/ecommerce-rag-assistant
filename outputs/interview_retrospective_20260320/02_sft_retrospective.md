# SFT 复盘

## 一句话结论

SFT 最后能赢，不是因为它"整体更强"，而是因为项目将优化目标从泛化提升收缩为 `base < 0.5` 的低风险流程型失败场景，并且先把训练稳定性和评测稳定性修好。

最可信的成功证据：

- 文件：`outputs/failure_first_qwen25_3b_v1/benchmark_sft_failure_first_no_judge_v5_followupfix_all4bit_cleaned_v2.md`
- 结果：base `0.4531`，sft `0.7156`
- 路由：4 个场景里 SFT 赢 3 个

## 版本时间线

| 时间 | 证据文件 | 关键结果 | 主要动作 | 复盘结论 |
|---|---|---|---|---|
| 2026-03-16 | `outputs/model_showcase/benchmark_results.md` | base `0.6321` > sft `0.5476` | 最早的混合 SFT | 数据分布没对齐，甚至生成了与中文电商场景无关的英文商品文本，说明无定向策略的训练方式不可取。 |
| 2026-03-18 14:36 | `outputs/model_showcase/benchmark_results_focus_sft_partial.md` | base `0.7543` > sft `0.4838` | 聚焦版 SFT | 场景边界不够干净，SFT 在高风险场景里学习到不安全的承诺模式，出现 `unsafe_promise`。 |
| 2026-03-18 19:25 | `outputs/model_showcase/benchmark_results_qwen25_3b_sft_struct_targeted.md` | sft `0.7816` > base `0.7600` | 只打结构化预算推荐和对比 | 第一次局部赢，但还是偏"结构化 benchmark"，还没证明真实 failure-first 价值。 |
| 2026-03-18 21:39 | `outputs/model_showcase/benchmark_results_qwen25_3b_sft_struct_targeted_clean_v1_trimmed.md` | sft `0.7712` > base `0.7600` | 清洗 + trim | 清洗后更稳，说明输出长度和脏 token 也会影响结论。 |
| 2026-03-18 23:57 | `outputs/failure_first_qwen25_3b_v1/benchmark_sft_failure_first.md` | base `0.5033` > sft `0.4685` | 第一版 failure-first SFT | 方向正确，但第一次落地仍失败。 |
| 2026-03-19 00:13 | `outputs/failure_first_qwen25_3b_v1/benchmark_sft_failure_first_no_judge.md` | base `0.5638` > sft `0.5416` | 去掉不稳定 judge | 分数马上变化，说明早期评测受 judge 影响太大。 |
| 2026-03-19 12:14 | `outputs/failure_first_qwen25_3b_v1/benchmark_sft_failure_first_no_judge_v4.md` | sft `0.6086` > base `0.5638` | 训练转稳 | 第一次出现 failure-first 子集上的真实 lift。 |
| 2026-03-19 16:53 | `outputs/failure_first_qwen25_3b_v1/benchmark_sft_failure_first_no_judge_v4_followupfix.md` | sft `0.5911` > base `0.4531`，2 比 2 | followup 修补 | 最难的多轮记忆开始被救回来，但简单推荐还不够自然。 |
| 2026-03-19 18:47 | `outputs/failure_first_qwen25_3b_v1/benchmark_sft_failure_first_no_judge_v5_followupfix.md` | sft `0.6326` > base `0.5839`，但只赢 1 个场景 | followup 更强 | 只看均分会误判，必须看 routing。 |
| 2026-03-19 19:18 | `outputs/failure_first_qwen25_3b_v1/benchmark_sft_failure_first_no_judge_v5_followupfix_all4bit_cleaned_v2.md` | sft `0.7156` > base `0.4531`，赢 3/4 | 清理脏输出 + 修正 hardest case | 这是最终经得起复验的 SFT 成功版本。 |

## 训练配方是怎么从不稳走到可用的

| 模型目录 | 关键配置 / 结果 | 含义 |
|---|---|---|
| `sft_model_v1` | `learning_rate=8e-5`，`bf16=True`，`train_loss=21.16`，`eval_loss=nan` | 第一版明显数值不稳，benchmark 结论不可信。 |
| `sft_model_v2` / `sft_model_v2_retry1` | `learning_rate` 降到 `2e-5`，但 `eval_loss` 仍是 `nan`，`train_loss` 甚至到 `0.0` | 光降学习率不够，`loss=0` 也不代表学对了。 |
| `sft_model_v3_float32_gpu1` | 改成 `bf16=False`，`eval_loss=0.4699` | 先把训练变稳定，才谈 benchmark。 |
| `sft_model_v4_float32_gpu1_rebalanced` | `eval_loss=0.3467` | 数据重平衡开始生效。 |
| `sft_model_v5_float32_gpu1_followupfix` | `eval_loss=0.3524` | followup 修补后，能力边界更清晰。 |

这部分的核心价值在于：

- 项目不是只关注 benchmark 分数
- 而是先检查训练稳定性，再决定是否相信 benchmark
- 并且明确了 `nan`、异常低 loss、异常高 loss 都意味着训练配方有问题

## 真正让 SFT 起效的动作

### 1. 从"结构化更好看"改成"仅针对 base 真实失败的案例做定向优化"

证据：

- `outputs/failure_first_qwen25_3b_v1/PLAN.md`
- `outputs/failure_first_qwen25_3b_v1/failure_cases_lt_0p5.md`

关键变化：

- 只从 `Qwen2.5-3B base` 的 `overall < 0.5` case 开始
- 先只保留 `simple_recommendation` 和 `followup_memory`
- 明确排除 `tone_alignment`、`honest_boundary`、`after_sales_boundary`

这个动作非常关键，因为它把 SFT 从"泛化补丁"变成了"流程问题修复器"。

### 2. 只允许失败场景对应品类进入训练集

证据：

- `outputs/failure_first_qwen25_3b_v1/sft_data_plan.md`

关键约束：

- `simple_recommendation` 只保留蓝牙耳机、机械键盘、运动鞋
- `followup_memory` 只保留运动鞋
- 不混入 `structured_comparison`
- 不混入 `tone_alignment / honest_boundary`

这避免了"任务名一样、分布却不一样"的假对齐。

### 3. 对 hardest followup case 做定向补样

证据：

- `outputs/failure_first_qwen25_3b_v1/sft_export_manifest.md`

关键变化：

- followup memory 从原始 `20 / 3` 扩到导出后 `81 / 5`
- simple recommendation 从原始 `219 / 27` 扩到导出后 `438 / 54`

这说明项目不是无定向策略地扩充数据，而是只对最难的 failure case 定向补样。

### 4. 清掉脏输出和重复尾巴

证据：

- `outputs/failure_first_qwen25_3b_v1/benchmark_sft_failure_first_no_judge_v5_followupfix_all4bit_cleaned.md`
- `outputs/failure_first_qwen25_3b_v1/benchmark_sft_failure_first_no_judge_v5_followupfix_all4bit_cleaned_v2.md`

最明显的变化：

- `sft_followup_memory_sneaker` 从 `0.5337` 提到 `0.7521`
- 其他 3 个场景分数不变

这说明最后一跳不是重新训练出奇迹，而是把最硬那个 case 的脏文本和重复输出处理掉了。

## 最终成功证据该怎么讲

### 最该讲的 3 个场景

| 场景 | base | sft | 结论 |
|---|---:|---:|---|
| 简单推荐：通勤慢跑运动鞋 | `0.4660` | `0.8441` | SFT 对低风险、可模板化推荐最有效。 |
| 多轮追问与上下文记忆：运动鞋 | `0.1420` | `0.7521` | SFT 能显著修复编号延续、上下文记忆和直接结论。 |
| 简单推荐：安静办公机械键盘 | `0.4278` | `0.4822` | 提升不大，但仍优于 base。 |

### 也要主动承认的 1 个场景

| 场景 | base | sft | 应该怎么说 |
|---|---:|---:|---|
| 简单推荐，base 足够可用 | `0.7765` | `0.7838` | 虽然 SFT 略高，但提升太小，不值得为这类低风险请求常驻更重模型。 |

这句很重要，因为它体现的是工程判断，而不是"为了证明 SFT 有用而强行全量替换"。

## SFT 阶段最有价值的失败经验

### 1. 早期数据分布不对，会导致模型产生明显偏离目标的输出

最早的 `outputs/model_showcase/benchmark_results.md` 里，SFT 甚至生成了与中文电商场景无关的英文商品文本，说明早期语料和目标中文电商场景并不一致。

### 2. 不先稳训练，后面的 benchmark 都不值得信

`sft_model_v1` 和 `sft_model_v2_retry1` 的 `eval_loss=nan` 已经足够说明问题。这个阶段如果只看个别 case 分数，很容易自我欺骗。

### 3. 小 judge 不稳定时，不宜直接采信"绝对分"

`benchmark_sft_failure_first.md` 里大量 `judge_parse_failed`，而切到 `no_judge` 后整体结论就明显变化，说明评测链路本身也要复盘。

### 4. 只看均分会误判

`benchmark_sft_failure_first_no_judge_v5_followupfix.md` 里，虽然 sft 均分高于 base，但 routing 还是 `base 3` 比 `sft 1`。这类版本不能拿来当"成功版本"。

## 适合表述的 SFT 结论

> 本项目最终的核心发现是，SFT 真正有价值的不是"把小模型整体变强"，而是把它收缩成一个低风险流程能力修复器。只要场景满足结构稳定、事实由 RAG 提供、答案形式可示范，比如简单推荐和多轮跟进，SFT 就能稳定拉开和 base 的差距；一旦任务变成语气、边界、说服或高风险承诺，SFT 就不应强行覆盖。
