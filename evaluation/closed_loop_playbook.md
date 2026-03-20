# 电商客服闭环模板

## 核心原则

先压测 `base`，再决定要不要生成 SFT / DPO 数据。

不要把所有失败都归给训练。先分清：

- `RAG / Prompt` 问题：检索没拿对、字段映射错、提示词没约束好
- `SFT` 问题：拿到证据后仍然不能稳定按流程或格式输出
- `DPO` 问题：语气、拒答、边界、优先级排序不对
- `Manual Review`：退款赔付承诺、改地址、投诉升级等高风险动作

## 建议流程

1. 固定一版基线
   - 固定 system prompt
   - 固定检索参数
   - 固定输出要求

2. 跑 base benchmark
   - `python scripts/run_pipeline.py evaluate`

3. 生成数据计划
   - `python scripts/run_pipeline.py build-data-plan`

4. 逐条处理 backlog
   - `rag_or_prompt`：先修工程
   - `tooling`：先修工具和字段映射
   - `sft`：生成流程化/结构化样本
   - `dpo`：生成偏好对
   - `manual_review`：线上直接转人工

5. 重训并复测
   - 只对已确认高价值失败的数据加量
   - 不看平均分，重点看原失败场景是否被消灭

## 场景 taxonomy

实现文件：

- [scenario_taxonomy.py](/data/wtw/Desktop/resume/ecommerce-rag-assistant/evaluation/scenario_taxonomy.py)

当前分桶：

- `base`
  - 开放式泛咨询
  - 简单推荐
- `sft`
  - 结构化推荐
  - 结构化对比
  - 澄清与多轮跟进
- `dpo`
  - 不知道就坦诚说明
  - 品牌客服语气
- `manual_review`
  - 售后赔付与承诺边界

## 数据计划脚本

实现文件：

- [build_data_plan.py](/data/wtw/Desktop/resume/ecommerce-rag-assistant/scripts/build_data_plan.py)

输入：

- `outputs/model_showcase/benchmark_results.json`

输出：

- `outputs/model_showcase/data_plan.json`
- `outputs/model_showcase/data_plan.md`

输出里会给出：

- 场景家族
- 风险等级
- 当前建议在线路由
- 当前最佳阶段
- 建议动作类型
- 建议样本规模
- 样本必须包含字段
- hard cases 清单

## 实操建议

### 什么时候优先做 SFT

- 用户已经说清需求，但模型仍然丢字段
- 结构化模板不稳定
- 多轮追问时上下文断裂
- 先澄清再推荐这类流程走不稳

### 什么时候优先做 DPO

- 明明不知道还要乱答
- 对用户态度太硬或太油
- 高风险请求里不会克制表达
- 不会把“解释流程”和“直接承诺结果”区分开

### 什么时候不要急着做训练

- 检索错品类
- 商品证据根本没进 prompt
- 订单字段映射错误
- 工具调用不存在或权限不足

## 当前项目最重要的现实结论

如果高风险场景里三种模型都不稳，不要硬选一个“看起来分最高”的模型上线。

先把它标成 `manual_review`，再针对该类失败单独做数据和训练。
