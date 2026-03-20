# DPO 复盘

## 一句话结论

DPO 这条线前面失败很多次，根因不是"DPO 没用"，而是前面一直拿 DPO 去修 `no_direct_answer`、`no_refusal`、`prompt_echo` 这种基础能力或生成质量问题。真正起效是在项目将目标调整为"真实客服偏好校准"之后。

最可信的成功证据：

- 文件：`outputs/preference_first_dpo_qwen25_3b_v2_realistic_service/benchmark_realistic_service_judge_first_base_vs_dpo_rerun.md`
- 结果：base `0.4572`，dpo `0.5595`
- 路由：5 个场景里推荐 DPO 的有 3 个
- Pairwise：5 个场景全部判给 DPO

## 版本时间线

| 时间 | 证据文件 | 关键结果 | 主要动作 | 复盘结论 |
|---|---|---|---|---|
| 2026-03-19 21:02 | `outputs/base_failure_dpo_qwen25_3b_v1/benchmark_base_vs_dpo_failure_first.md` | base `0.5144` > dpo `0.4981` | 直接拿 DPO 修 tone / boundary | 起点就错了，DPO 没能补齐基础能力。 |
| 2026-03-19 21:18 | `outputs/base_failure_dpo_qwen25_3b_v2_targeted/benchmark_base_vs_dpo_failure_first_strict.json` | strict 下 base `0.2331`，dpo `0.2362` | targeted 数据 | 宽松评测看着涨，严格评测几乎没涨。 |
| 2026-03-19 22:57 | `outputs/base_failure_dpo_qwen25_3b_v2_targeted/benchmark_base_vs_dpo_failure_first_strict_rerun.md` | 还是 base `0.2331`，dpo `0.2362` | strict rerun | 提升极小，且样例里 DPO 还会 prompt echo。 |
| 2026-03-19 23:17 | `outputs/base_failure_dpo_qwen25_3b_v3_strict_repair/benchmark_base_vs_dpo_failure_first_strict_rerun.md` | dpo `0.3856` > base `0.2331` | strict repair | 看起来大涨，但其实评测已经开始奖励退化输出。 |
| 2026-03-19 23:21 | `outputs/base_failure_dpo_qwen25_3b_v3_strict_repair/benchmark_base_vs_dpo_failure_first_strict_rerun_v2.md` | base `0.2331` > dpo `0.1638` | 更严格复跑 | 一收紧评测，前面的"提升"直接消失。 |
| 2026-03-20 00:35 | `outputs/preference_first_dpo_qwen25_3b_v1/benchmark_preference_first_base_vs_dpo.md` | base `0.1454` = dpo `0.1454` | 改做抽象 preference-first | 虽然方向开始对了，但 benchmark 还不真实，收益为零。 |
| 2026-03-20 14:18 | `outputs/preference_first_dpo_qwen25_3b_v2_realistic_service/benchmark_realistic_service_base_vs_dpo_pairwise.md` | base `0.1230`，dpo `0.3158` | 换成真实客服场景 | benchmark 终于贴近业务，DPO 开始显出价值。 |
| 2026-03-20 15:44 | `outputs/preference_first_dpo_qwen25_3b_v2_realistic_service/benchmark_realistic_service_judge_first_base_vs_dpo.md` | base `0.5042` > dpo `0.4680`，全部 tie | judge-first 初版 | 不是模型退步，而是评测解析坏了。 |
| 2026-03-20 15:44 | `outputs/preference_first_dpo_qwen25_3b_v2_realistic_service/benchmark_realistic_service_judge_first_base_vs_dpo_repaired.md` | dpo `0.5595` > base `0.4572` | repaired | 修完评测链路后，DPO 的真实增益才显现。 |
| 2026-03-20 16:00 | `outputs/preference_first_dpo_qwen25_3b_v2_realistic_service/benchmark_realistic_service_judge_first_base_vs_dpo_rerun.md` | dpo `0.5595` > base `0.4572` | rerun 验证 | 最终结果稳定，可作为最终结论。 |

## 前 3 次 DPO 为什么失败

### 1. 错把 DPO 当成能力补全

证据：

- `outputs/base_failure_dpo_qwen25_3b_v1/benchmark_base_vs_dpo_failure_first.md`
- `outputs/preference_first_dpo_qwen25_3b_v1/PLAN.md`

最初失败的关键原因是：

- `no_direct_answer`
- `no_refusal`
- `prompt_echo`
- `unsupported_claim`

这些问题本质上属于：

- SFT 没学稳
- 生成输出没清干净
- 规则层没有先兜底

而不是 "chosen 比 rejected 更像客服" 这种 preference ranking 问题。

### 2. 评测一度把退化输出也算成高分

最典型的例子在：

- `outputs/base_failure_dpo_qwen25_3b_v3_strict_repair/benchmark_base_vs_dpo_failure_first_strict_rerun.md`

在 "品牌客服语气对齐：笔记本怀疑营销" 这个 case 里：

- base 因为 `no_direct_answer` 被判 `0.0000`
- DPO 却拿到 `0.8875`
- 但 DPO 的回答几乎只是复述了用户原句

然后在更严格的复跑里：

- `outputs/base_failure_dpo_qwen25_3b_v3_strict_repair/benchmark_base_vs_dpo_failure_first_strict_rerun_v2.md`
- dpo 平均分从 `0.3856` 直接掉到 `0.1638`

这一发现具有较高的方法论参考价值：

- 不是所有"strict"都真的 strict
- 评测器如果没有硬 gate，很容易奖励表面像样、实则退化的输出

### 3. 抽象偏好 benchmark 太像 prompt-following 测试

证据：

- `outputs/preference_first_dpo_qwen25_3b_v1/PLAN.md`
- `outputs/preference_first_dpo_qwen25_3b_v1/benchmark_preference_first_base_vs_dpo.md`

这版虽然已经意识到 DPO 不该补能力，但场景还停留在：

- "你直接说值不值得买"
- "你就直接判断适不适合通勤"

问题是：

- 太抽象
- 太像 prompt-following
- 不够像真实客服工作流

所以结果是：

- base `0.1454`
- dpo `0.1454`

也就是没有任何实质收益。

## 真正让 DPO 起效的动作

### 1. 从抽象偏好改成真实客服业务流

证据：

- `outputs/preference_first_dpo_qwen25_3b_v2_realistic_service/PLAN.md`

新主战场变成了：

- 客服促单话术
- 缺货后的委婉转品
- 敏感肌多轮咨询与安抚
- 议价挽留
- 基于需求的真实客服推荐

这个变化很重要，因为它把 DPO 从"抽象话术优化"变成了"有业务语境的偏好校准"。

### 2. prompt 内嵌证据和业务备注

证据：

- `outputs/preference_first_dpo_qwen25_3b_v2_realistic_service/dpo_data_manifest.md`

这版数据不再只是模板化 chosen / rejected，而是：

- 内嵌商品资料
- 内嵌安全备注
- 内嵌客服备注
- 明确禁止编库存、优惠、疗效、售后承诺

这让 DPO 学到的是"怎么在给定证据上说得更像客服"，而不是在无证据支撑下提升生成质量。

### 3. 加硬 gate，再上 stronger judge

证据：

- `outputs/preference_first_dpo_qwen25_3b_v2_realistic_service/PLAN.md`
- `outputs/preference_first_dpo_qwen25_3b_v2_realistic_service/JUDGE_SPEC.md`

核心思路是：

- 第一层先挡业务错误和安全错误
- 第二层再打客服偏好分
- 第三层才做 pairwise judge

这一步是整个 DPO 线最重要的工程升级之一。

### 4. 承认有些场景就是不该自动放开

最终最佳结果里，依然有 2 个场景没有放开：

| 场景 | base | dpo | 结论 |
|---|---:|---:|---|
| 真实客服转品：手机缺货后委婉推荐替代款 | `0.4392` | `0.2000` | DPO 触发 `inventory_hallucination`，必须人工兜底。 |
| 真实客服议价：守住价格边界并挽留成交 | `0.3548` | `0.3548` | 两边都不够好，不应该为证明 DPO 有用而强行上线。 |

这恰恰体现了工程方案的成熟度。

## 最终成功证据该怎么讲

### 最该讲的 3 个成功场景

| 场景 | base | dpo | 提升点 |
|---|---:|---:|---|
| 真实客服促单：护肤品保湿修护 | `0.2102` | `0.7160` | DPO 学到了尊称、推荐理由和正向购买预期。 |
| 真实多轮咨询：敏感肌护肤安抚 | `0.6753` | `0.8027` | DPO 在已有上下文上学到了更稳的服务感和针对性。 |
| 真实需求推荐：通勤和设备切换耳机 | `0.6067` | `0.7238` | DPO 把"模板化推荐"变成了"更贴需求的客服回答"。 |

### 还要主动承认的限制

- Pairwise 上 DPO 5/5 都赢，不等于 5/5 都可自动上线
- 高风险业务流里，偏好更好不代表业务正确性已经足够
- DPO 适合做"更像客服"，不适合做"在无证据支撑下提升生成质量"

## 适合表述的 DPO 结论

> 这条 DPO 线最大的教训是，DPO 不是拿来修基础能力的。前期项目一直尝试用 DPO 修复 direct answer、拒答边界和 prompt echo，结果多次 strict rerun 都不稳。真正见效是在项目明确 DPO 只适合做 preference calibration 之后，把 benchmark 换成真实客服业务流，再加上硬 gate 和更强 judge，最后才在促单、需求推荐和咨询安抚这类场景里稳定优于 base。
