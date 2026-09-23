# Gemini 3 Deep Think：榜单配置与官方证据边界

核验日期：2026-09-18。本笔记记录已经出现在 Artificial Analysis 的 `Gemini 3 Deep Think` 条目，并沿 Google 官方资料核验其身份和面试技术。当前公开证据不支持把它当成一个有独立 API model ID、独立 Model Card 或公开权重的基础模型。

## 1. 榜单锚点

- Artificial Analysis 条目：[Gemini 3 Deep Think](https://artificialanalysis.ai/models/gemini-3-deep-think)。页面结构化字段确认名称 `Gemini 3 Deep Think`、创建方 Google、`releaseDate: 2026-02-05`、`isReasoning: true`，并将其标为 proprietary。
- Artificial Analysis FAQ 还记录：参数规模未公开、权重不可用、输入为 text、输出为 text、上下文窗口约 `130K`。这些是第三方目录/页面整理字段，不能替代 Google 官方模型规格；尤其不能把 130K 推导成 Gemini 3.1 Pro API 的 1M 上下文边界。
- Artificial Analysis 快照 `/tmp/gdt-aa.html`：3,249,519 bytes，SHA-256 `b96e3a93bc9f7a882f71523cdc9cdf6023fa2b1536e72a93c579f942d31cd75f`。页面中的 Intelligence Index、速度、价格和 provider 可用性属于第三方配置级数据，本轮不将其与其他 Gemini 条目的分数合并。
- DataCurve DeepSWE v1.1 当前页面没有精确的 `mini_swe_agent_gemini_3_deep_think_*` 行。因此不把 Gemini 3.1 Pro、Gemini 3.7 Flash 或其他 Gemini 配置的 Pass@1、成本和 Agent steps 迁移给本条目。

## 2. 官方身份核验

### 2.1 没有独立官方 model endpoint

以下猜测入口本轮分别返回 404：

- [Google AI Developers Gemini 3 Deep Think 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3-deep-think)
- [Google DeepMind Gemini 3 Deep Think Model Card](https://deepmind.google/models/model-cards/gemini-3-deep-think/)
- Google Blog 的 Gemini 3 Deep Think 专属路径

可访问的官方对象是 [Gemini 3.1 Pro API 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview)、[Gemini 3.1 Pro Model Card](https://deepmind.google/models/model-cards/gemini-3-1-pro/) 以及 Gemini Thinking/工具/长上下文文档。3.1 Pro Model Card 明确将 Deep Think 作为安全评估中的推理设置，并把架构、训练数据、硬件和软件资料指向 Gemini 3 Pro；它没有把 `Gemini 3 Deep Think` 列成独立权重或公开 API 模型。

因此本项目采用以下归并：

```text
Artificial Analysis 条目：Gemini 3 Deep Think
官方基础模型证据：Gemini 3.1 Pro Preview / Gemini 3 Pro 系列
官方可观察技术：thinking_level、共同 output budget、tool/signature replay、1M context、caching
不能确认：独立 checkpoint、独立参数量、独立架构、独立训练 recipe
```

这不是把两个名称强行改成同一个模型，而是明确记录“排行榜名称”和“官方可核验对象”之间的证据缺口。

### 2.2 不把 3.1 Pro 资料升级成 Deep Think 独有事实

Gemini 3.1 Pro 官方资料能够支持以下面试知识，但这些是 3.1 Pro/API 或 Gemini 3 系列协议事实：

- `thinking_level` 是请求级推理控制信号；思考 token 与最终输出共同受 output budget 约束，不能只增加思考预算而忽略答案预算。
- Gemini 3 工具组合中的 thought、tool call 和 tool result 可能携带加密 `signature`。stateful 模式由服务端维护 interaction/id，stateless 模式需要完整回放相关 steps、id 和 signature；signature 不是可读 chain-of-thought。
- `gemini-3.1-pro-preview-customtools` 是面向 bash 与自定义工具选择的 endpoint variant，不应算作新的基础模型。
- 1M context 是 API 容量上限，不是有效召回率、永久记忆或无需 RAG 的证明；缓存、分块、索引、验证和成本账本仍然必要。

3.1 Pro Model Card 的安全页面还展示了一个重要评测原则：比较 Deep Think 与普通设置时要把 inference cost 纳入分析。test-time compute、能力、延迟、成本和风险必须一起报告，不能只引用最高分。

## 3. 论文、技术报告与代码

截至 2026-09-18，通过 Google DeepMind Research/Model Card、Google AI Developers、Google Blog 和 arXiv 定向检索，没有找到 Google 发布的 `Gemini 3 Deep Think` 独立技术报告、公开权重或独立代码仓库。搜索结果和第三方页面只能帮助定位候选链接，不能替代官方证据。

因此不把 Gemini 3 Pro/3.1 Pro 的前代 Model Card、外部评测论文或 Artificial Analysis FAQ 扩写成 Deep Think 的内部训练方法。现有 `gemini-3.1-pro-preview-source-notes.md` 已记录可复用的官方协议和评测边界，本笔记只补充该榜单配置的归并关系。

## 4. 闭环状态与面试映射

`Gemini 3 Deep Think` 当前为**资料级闭环（配置级锚点）**：

1. Artificial Analysis 有精确榜单条目；
2. DataCurve 没有精确 Agent 行，保留负证据；
3. Google 官方资料可以核验相邻的 Gemini 3.1 Pro/Deep Think 运行时与评测概念；
4. 独立官方模型身份、参数、架构和训练报告不存在公开确认，因此不新增专属正式章节。

适合面试的主线是：

- 如何区分排行榜配置、基础模型和 endpoint variant；
- 为什么 thinking budget 必须与可见输出、延迟、成本和成功率一起管理；
- 为什么工具 signature 是状态连续性材料，而不是明文思维链；
- 为什么官方 benchmark、Artificial Analysis 指数和 `mini-swe-agent` 结果不能拼成裸模型排名；
- 在没有独立技术报告时，如何把“已核验接口行为”和“未公开内部机制”分开表达。

仍待核验：Artificial Analysis 条目的 provider 后端映射、具体推理预算、Google 是否曾使用独立内部 checkpoint、独立 benchmark 复现，以及未来是否公开 Gemini 3 Deep Think 专属技术报告或模型卡。

## 5. 本轮快照

| 快照 | 内容 | 大小 | SHA-256 |
|---|---|---:|---|
| `/tmp/gdt-aa.html` | Artificial Analysis Gemini 3 Deep Think 详情页 | 3,249,519 | `b96e3a93bc9f7a882f71523cdc9cdf6023fa2b1536e72a93c579f942d31cd75f` |
| `/tmp/gdt-card31.html` | Google DeepMind Gemini 3.1 Pro Model Card | 161,703 | `7a18b367056f43ad9270b1e8d6917667e0c223d737171ecc5302bc4094c82ddc` |
| `/tmp/gdt-model31.html` | Google AI Developers Gemini 3.1 Pro 模型页 | 107,527 | `7962417c120571fbd2ec1374faf7bde6fd573191cd76b0e061b8453cfcf6c1db` |

