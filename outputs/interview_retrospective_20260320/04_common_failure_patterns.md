# 共性失败模式

## 1. 方法和问题类型不匹配

### 表现

- 用 SFT 去处理语气、边界、说服这类偏好问题
- 用 DPO 去修 `no_direct_answer`、`no_refusal`、`prompt_echo`

### 证据

- SFT 计划：`outputs/failure_first_qwen25_3b_v1/PLAN.md`
- DPO 计划：`outputs/preference_first_dpo_qwen25_3b_v1/PLAN.md`

### 真正原因

- SFT 更适合流程能力、模板能力、上下文延续
- DPO 更适合两个回答都基本可用时的偏好排序

### 关键表述

> 项目最终不是继续堆数据，而是先将"哪个问题该归 SFT，哪个问题该归 DPO"这件事厘清了。方法边界一旦定义错，后续再多实验也只是在噪声里找提升。

## 2. 训练数值不稳定时，任何 benchmark 结论都要打问号

### 表现

- `outputs/sft_regen/eval_results.json` 里 `eval_loss=nan`
- `outputs/dpo_regen/eval_results.json` 里 `eval_loss=nan`
- `outputs/failure_first_qwen25_3b_v1/sft_model_v1/eval_results.json` 里 `eval_loss=nan`
- `outputs/failure_first_qwen25_3b_v1/sft_model_v2_retry1/train_results.json` 里 `train_loss=0.0`

### 真正原因

- 混合精度和学习率组合没有先稳定下来
- 训练 loss 异常不等于模型学会了

### 关键表述

> 项目后期将"先稳训练，再看效果"作为阶段门槛。因为如果 `nan` 都没有处理掉，benchmark 分数实际上没有解释力。

## 3. 数据分布不对，模型会学到完全错误的任务形态

### 表现

- `outputs/model_showcase/benchmark_results.md` 里的早期 SFT / DPO 生成了与中文电商场景无关的英文商品文本
- `outputs/model_showcase/findings.md` 明确写了后来要改成基于当前中文商品目录重新生成数据

### 真正原因

- 训练数据分布和目标电商中文场景不一致
- 模型学到的是"会列商品"，不是"会做中文电商客服回复"

### 关键表述

> 本项目中最早的失败不是模型能力不足，而是数据分布错误。模型学到的输出风格与线上目标场景不一致，因此后续优先进行了数据边界收缩。

## 4. 评测链路本身也会失败

### 表现

- `outputs/failure_first_qwen25_3b_v1/benchmark_sft_failure_first.md` 里大量 `judge_parse_failed`
- `outputs/preference_first_dpo_qwen25_3b_v2_realistic_service/benchmark_realistic_service_judge_first_base_vs_dpo.md` 里 pairwise rationale 直接把原始 JSON 串打印出来，结果全成 `tie / low`

### 真正原因

- 小模型 judge 稳定性不够
- 解析逻辑不稳
- 没有把 hard gate 和 preference score 分层

### 关键表述

> 项目后期将"评测器本身是否可靠"作为独立问题排查。因为如果 judge 会 parse fail，最终复盘的就不是模型能力，而是评测链路缺陷。

## 5. 只看平均分，会把失败说成成功

### 表现

- `outputs/failure_first_qwen25_3b_v1/benchmark_sft_failure_first_no_judge_v5_followupfix.md` 里 sft 均分更高，但 routing 只有 1 个场景推荐 SFT
- DPO 线里也有 pairwise 全赢但仍需 `manual_review` 的场景

### 真正原因

- 平均分会掩盖 case-level 风险
- 业务上线看的是 routing 和风险，不是论文式均值

### 关键表述

> 项目最终不再仅关注平均分，而是逐案例判断最优路由。这样才能解释为什么"部分场景优于 base"是真实的，但"整体替换 base"仍然不成立。

## 6. 不愿承认人工兜底，会让项目叙事变假

### 表现

- `outputs/preference_first_dpo_qwen25_3b_v2_realistic_service/benchmark_realistic_service_judge_first_base_vs_dpo_rerun.md` 里有 2 个场景明确继续 `manual_review`
- `outputs/model_showcase/findings.md` 也明确把高风险客服场景保留给人工

### 真正原因

- 小模型即使在偏好上更像客服，也不代表已经满足业务正确性
- 真实系统必须有 route-out / human-in-the-loop

### 关键表述

> 项目没有为追求"模型全自动闭环"而放开高风险场景。相反，最终将 manual review 纳入路由策略，这恰恰使方案更贴近真实工程实践。

## 7. 真正有效的优化不是"更大更多"，而是"边界更收敛、覆盖更精准"

### 最终验证出来的有效做法

- 仅针对 `base < 0.5` 的 failure-first case
- 只保留和 failure case 同品类、同任务形态的数据
- 先修训练稳定性
- 先修评测链路
- SFT 和 DPO 各自只解决自己最擅长的问题

### 本项目最具方法论价值的经验

> 小模型对齐项目里，最难的不是"再训一版"，而是知道下一版为什么应该训、应该只训什么、不该训什么。
