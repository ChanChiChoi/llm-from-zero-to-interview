# 评估体系与平台设计：深度精修研究记录

日期：2026-10-27  
正文：[第七册第十二章：评估体系综合设计与工程落地](../book-07-evaluation-experiments/chapters/12-评估体系综合设计与工程落地.md)

## 研究问题

本章已经完整覆盖任务契约、benchmark、人工评估、LLM judge、污染、能力专项评估、A/B、统计、错误分析、回归和平台构成。精修不再重复各专题章节的讲解，重点核验最后的 Eval Platform 是否把这些方法变成可复核运行：评估对象如何冻结，运行产物怎样关联，失败/重试怎样保留，分数怎样进入发布门禁。

## 来源及其实际贡献

| 来源 | 直接支持的事实或设计 | 它解决的缺口 | 证据边界 |
| --- | --- | --- | --- |
| [HELM 原论文](https://arxiv.org/abs/2211.09110) | 用 scenarios、adaptations 和 metrics 固定不同模型的比较条件；在 16 个核心场景尽量测多类指标，公开逐样本 prompts/completions。论文摘要报告：此前研究的模型平均只覆盖 17.9% 的核心场景，HELM 将其提高到 96%。 | 让模型结果更透明、场景交集更大，并允许外部检查输入输出。 | 96% 是 HELM 论文选择的核心场景覆盖，不代表业务覆盖率；场景选择受可行性、语言和数据可得性约束，公开样本也仍有污染和训练暴露风险。本文没有把它当成本地系统效果数据。 |
| [OpenAI Eval-Driven System Design 案例](https://developers.openai.com/cookbook/examples/partners/eval_driven_system_design/receipt_inspection) 与 [Evaluation Best Practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices) | 官方资料展示从失败样本定性分析、构建评估集、配置 grader 到版本间测量和迭代的应用工作流；评估可组合程序、人类和 LLM 评分。 | 把 prompt/app 的改进连接到可测的已知失败案例，减少无基线的盲目改 prompt。 | CookBook 是特定收据检查案例，不是跨任务对照研究；官方建议也不等于第三方验证。它提供工作流证据，不证明某一 grader 对目标任务有效。 |
| [MLflow Tracking](https://mlflow.org/docs/latest/tracking/) 和 [Evaluation Datasets](https://mlflow.org/docs/latest/genai/datasets) | Tracking 的 run 可记录参数、代码版本、指标、输出文件和输入 dataset 元信息；GenAI dataset 文档描述关联评估数据与 runs 的方式。 | 给实验运行、结果、数据输入和 artifact 建立可查询的记录模型。 | 记录字段完整不代表实验定义正确，也不能自动提供 holdout 权限、标注质量、评估器有效性或统计结论。此处以其作为实验账本实现实例，不称作通用评估平台标准。 |
| [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md) | GenAI evaluation 事件定义 evaluation name、score value/label、explanation 和 response ID 等字段；输入、输出和工具参数字段可能携带敏感内容。 | 让不同运行时和观察工具能用较一致的字段关联模型响应与评价事件。 | 语义规范只定义遥测字段，不定义正确性 rubric、人工 gold、分数校准或因果解释；规范还会演进，采集端仍需设置权限、脱敏和 retention。 |
| [OpenAI Working with evals](https://developers.openai.com/api/docs/guides/evals)、[Trace grading](https://developers.openai.com/api/docs/guides/trace-grading) 与[评估入门页](https://developers.openai.com/api/docs/guides/evaluation-getting-started) | 文档展示 dataset、prompt 版本、grader、run 以及 agent trace 评分等产品对象。页面于检索日注明，托管 Evals 将在 2026-10-31 转为只读并计划于 2026-11-30 关闭。 | 提供供应商平台如何支持应用迭代和 trace 诊断的实例，同时说明产品 API/功能是动态依赖。 | 关闭时间是 2026-10-27 页面记载的计划，应按发布方最新页面重核；它只涉及相应托管产品，不影响抽象的数据集/评分器/运行记录概念。 |

来源角色不同：HELM 是研究评测框架，OpenAI 材料是产品工作流和案例，MLflow 是实验追踪实现，OpenTelemetry 是遥测语义规范。它们不可按 benchmark 分数或功能数量做直接排名。精修将其放在同一比较表中，只比较各自问题、机制、收益和边界。

## 正文主要修订

1. 在平台系统边界中增加端到端对象流图：任务契约与数据快照 → run manifest → 调度与预算检查 → 模型 attempt → 原始输出/trace → 评分证据 → 聚合报告 → 人工复核和发布决策。明确每条关系应保留可追溯 ID 与版本，而非只保存 dashboard 指标。
2. 扩充执行层，区分 run/job 和 sample attempt；单独记录 attempt 顺序、错误、重试原因、请求 ID、成本与 trace。客户端超时不一定说明服务端未执行，故加入 unknown outcome 和不覆盖原始 attempt 的要求。
3. 扩充 manifest：加入数据 manifest digest、prompt 内容 digest、模型请求与解析版本、代码/镜像版本、指标定义/实现版本、采样方案和 retry policy。说明随机种子不保证托管模型完全确定；区别保存输出后的可重评分与重新调用系统的可重跑。
4. 将原来混在单一列表里的 `completed`、`model_error`、`evaluator_error`、`pending`、`invalid_data` 拆成执行、评分、结果成熟三个状态维度，给出模型成功但 judge 失败、线上观察未成熟两个组合例子。
5. 新增代表方案对照，解释 HELM、应用 eval flywheel、MLflow run tracking、OpenTelemetry GenAI conventions 分别解决什么，哪些证据支持它们、哪些问题仍需目标系统自己回答。
6. 在来源部分记录 OpenAI 托管 Evals 当前的计划退役日期，并标注其为有日期的产品状态，避免把一个即将关闭的供应商界面写成平台架构前提。

## Demo 静态审阅与手算

原 demo 的动作列表只为成本、安全和高严重度回归追加动作；最终 `decision` 又只判断 `missing` 和成本。因此即使 `weighted_score`、pairwise、safety、latency 或 regression signal 为假，仍可能输出 `continue_with_review`。本次把所有 false signal 映射到 `failed_signals` 和对应动作，任一缺项或门禁失败都会输出 `hold_release_and_resolve_blockers`；全部通过时也只输出 `eligible_for_release_review`，不自动批准发布。

手算原输入：计划覆盖 `23/27 = 0.85185...`，四舍五入 `0.852`；加权分数为 `(0.86×1 + 0.74×1.5 + 0.68×1.2 + 0.92×2) / 5.7 = 0.81158...`，显示 `0.812`；pairwise 胜率为 `(7 + 2/2) / 12 = 0.667`。缺少 `judge_calibration` 和 `multimodal_longctx`，且成本变化 `0.30 > 0.25`，所以失败 signal 为 `['cost']`，动作包含补齐两项和复核成本，决策为 hold。没有执行 Python；输出值是静态手算，不是运行结果。

## 证据范围和未验证项

- 没有连接目标模型 API、数据集、在线 trace、对象存储、调度器、权限系统或发布流程；新增图和对象模型是教学设计，不是生产架构验收。
- 没有运行 demo。手算只核对当前输入路径；没有验证空样本、零权重、非法 score、非有限数、judge 返回错误、重复 attempt 或并发调度等实现路径。
- 未做全章逐外链 HTTP 扫描、外部公式渲染或 PDF 构建。上述动态官方文档尤其是 OpenAI Evals 退役日期和 OpenTelemetry 语义约定需在再次采用时核对版本。
- 没有实测 benchmark 代表性、评估器校准、样本量/置信区间、平台吞吐/成本或上线效果；论文、产品文档和工程规范分别只支持其公开方法、接口和字段语义。

