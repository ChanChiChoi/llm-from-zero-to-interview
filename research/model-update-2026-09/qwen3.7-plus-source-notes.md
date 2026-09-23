# Qwen3.7 Plus 官方资料摘记

核验日期：2026-09-22。本笔记只把 Artificial Analysis 与 DataCurve DeepSWE 作为模型发现入口；Alibaba Cloud Model Studio 的模型文档、推荐模型页和 OpenAI-compatible API 文档用于核验已经发现的 Qwen3.7 Plus 及其托管运行时合同。没有把官方云产品目录另发现的 embedding、Qwen3.8 或第三方模型新增为本锚点。

## 1. 榜单身份与证据边界

- Artificial Analysis 精确条目：[Qwen3.7 Plus](https://artificialanalysis.ai/models/qwen3-7-plus)。页面 canonical slug 为 `qwen3-7-plus`，页面字段显示 release 为 June 2026；当前详情快照为 `/tmp/q37-aa-20260922.out`，`3,859,009` bytes，SHA-256 `19e9b48bbc9c6d38fab3391bee6353cff5aaa2524bdf04589ae02bbad4a27040`。
- AA 当前第三方字段为 Intelligence Index `25.1622215821984`、median output speed `68.5428061089526 tokens/s`、Intelligence Index task cost `0.32527343198119174`、约 `1,000,000` context、输入 text/image/video、输出 text。页面价格字段为约 `$0.40/$1.60` 每百万 input/output token，cache hit 字段约 `$0.04`；这些属于 AA provider/configuration 测量，不是 Alibaba 的统一全球价格合同。
- AA 页面没有公开参数规模或权重。不能从 “Plus”、1M context、视觉/Agent 能力或第三方指数反推 dense/MoE、层数、训练数据、优化器、RL 配方或内部 GUI policy。
- DataCurve [DeepSWE](https://deepswe.datacurve.ai/) 当前快照为 `268,036` bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，页面生成时间为 `2026-09-22T06:27:15.860279+00:00`。当前没有精确 `mini_swe_agent_qwen3_7_plus_*` 行，因此不把其他 Qwen 的 Pass@1、成本、输出 token 或 Agent steps 迁移到 Qwen3.7 Plus。
- 当前状态是 **AA 单榜资料级闭环**：有榜单身份、官方模型/API 文档、研究笔记、书系配套和面试训练；没有 DataCurve 精确 Agent 结果，也没有独立 Qwen3.7 Plus 技术报告、公开权重或参数/架构披露。

## 2. 官方模型合同

主来源是 [Alibaba Cloud qwen3.7-plus 文档](https://www.alibabacloud.com/help/en/model-studio/qwen3-7-plus)，页面 Last Updated 为 Sep 20, 2026；本地快照 `/tmp/q37-q37-doc-hyphen-20260922.out`，`44,386` bytes，SHA-256 `662ebedd3a538e22490e00c4eb29def16c0c5e9ff2de1d3f8fe7d7d334b43656`。

官方短描述把 `qwen3.7-plus` 定位为 Alibaba Cloud Model Studio 托管的多模态交互式混合 Agent。已公开的产品能力包括：

| 维度 | 官方可确认内容 | 不应推出的结论 |
|---|---|---|
| 输入输出 | text、image、video 输入；text 输出 | 没有证明内部视觉 encoder、token merger 或早期融合结构 |
| 任务定位 | coding、tool use、productivity；感知现实场景、读屏、GUI 交互、依据视觉参考生成代码、移动端端到端导航 | GUI 动作成功率、policy 网络、执行器实现和真实设备兼容性未公开 |
| 工具合同 | 多数区域支持 Function Calling、Structured Outputs、Web Search、Prefix Completion、Context Caching | 支持接口不等于工具权限、搜索质量、JSON 业务语义或缓存一定命中 |
| 微调 | 官方 capability table 标记 Fine-tuning Unsupported | 不能推断底层训练完全不可调整，结论只限该托管 API 合同 |
| 快照关系 | 当前 alias 功能等价于 `qwen3.7-plus-2026-05-26`；该版本是 May 26, 2026 snapshot | alias、snapshot 与开放 checkpoint 不是同一身份；没有公开权重可下载 |

## 3. 区域 capability matrix

官方页面把 Model Studio region、scope 和 capability 分开列出。中国北京为默认能力较完整且支持 Batch Inference；新加坡为 International；德国法兰克福、日本东京、香港和美国 Virginia 的 Global scope 支持表大体一致。美国 Virginia 另有 US scope，该 scope 不支持 Structured Outputs 和 Web Search；不能把一个 region 的能力表写成全局模型能力。

| 区域/scope | Function Calling | Structured Outputs | Web Search | Prefix Completion | Context Caching | Batch | Fine-tuning |
|---|---:|---:|---:|---:|---:|---:|---:|
| China (Beijing) | yes | yes | yes | yes | yes | yes | no |
| Singapore / International | yes | yes | yes | yes | yes | no | no |
| Germany / Global | yes | yes | yes | yes | yes | no | no |
| US Virginia / Global | yes | yes | yes | yes | yes | no | no |
| US Virginia / US | yes | no | no | yes | yes | no | no |
| Japan / Global | yes | yes | yes | yes | yes | no | no |
| Hong Kong / Global | yes | yes | yes | yes | yes | no | no |

这个矩阵的面试价值在于把“模型能力”“provider region”“scope”“API feature flag”和“宿主权限”拆开。一个请求即使被模型生成了符合 schema 的工具参数，也仍然要经过应用侧授权、执行、超时、幂等和 verifier。

## 4. 上下文、思考和成本账本

官方 Context Limits 表给出：

- Context window：`1,000,000`
- Max input length：`991,808`
- Max output length：`131,072`
- Thinking mode max input：`983,616`
- Thinking mode max output：`131,072`
- Max chain-of-thought length：`262,144`

这些字段必须分开记录。`1,000,000` 不是可同时用于用户文本、视觉 token、工具 schema、工具结果、thinking 和输出的单一输入预算；thinking 模式还会减少可用输入空间。`max chain-of-thought length` 是官方合同字段，不等于可读的完整 CoT，也不能据此推断隐藏状态的实现。

官方价格按 region、scope 和 input length 分档。例如北京原价表在 `input <= 256K` 时列出 input `$0.276`、output `$1.101`、implicit cache input `$0.056`；`256K < input <= 1M` 时为 input `$0.826`、output `$3.301`、implicit cache input `$0.166`。新加坡和 Virginia 的 Global scope 在短输入档列出 `$0.4/$1.6`，长输入档列出 `$1.2/$4.8`，并另列 explicit cache creation/read。价格应和 region、scope、长度、cache 状态、batch 路径一起写入 manifest，不能用 AA 页面一组价格覆盖所有部署。

## 5. 从官方描述提取的面试主线

### 5.1 交互式混合 Agent 是一个跨模态执行闭环

“交互式”不应理解为模型输出一个坐标就完成了 GUI 操作。可审计的数据流是：

```text
image/video/screen -> model proposal -> action schema
-> region/provider capability -> host permission
-> GUI/mobile executor -> observation
-> retry/idempotency -> verifier/artifact
```

模型可以提出点击、滚动、输入、导航或基于截图生成代码的意图；真实动作由宿主执行器决定。截图 hash、窗口尺寸、设备像素比、页面或 App revision、坐标系、动作序号和执行回执应进入 trace。页面发生变化时，宿主应该重新观测或拒绝旧坐标，而不是把模型文本直接当成成功副作用。

### 5.2 视觉能力首先是预算问题

Qwen3.7 Plus 的官方合同接受图像和视频，但没有公开每种媒体的统一 token 化公式。工程上至少要记录媒体 hash、原始尺寸、resize/crop、视频采样率、frame index、timestamp、视觉 token 估算、工具历史、thinking 预算和 output reservation。这样才能区分：

1. API 接受媒体；
2. processor 保留了关键证据；
3. 模型在有限预算中使用了证据；
4. GUI/mobile executor 完成并验证了动作。

`context window`、AA 的约 1M 字段和“支持 video”都不能证明第 2--4 层。

### 5.3 Prefix Completion 和 Context Caching 是协议能力

Prefix Completion 允许在给定前缀上继续生成；Context Caching 则是 provider 侧对重复上下文的复用。两者都应在请求 manifest 里与 model alias、region/scope、cache key、创建/读取、媒体 revision、tool schema hash 和失效原因分开。它们不是 GPU KV cache，也不是应用 memory；compaction、工具追加、region 切换或媒体版本变化都可能让缓存失效。

### 5.4 工具和结构化输出仍需宿主验收

Structured Outputs 约束的是响应形状，Function Calling 约束的是模型提出工具调用的协议形状；两者都不保证业务语义、权限、幂等或工具结果正确。Web Search 支持也不等于所有 region、scope、查询或租户都可用。面试中应按以下责任链回答：

```text
model -> schema validation -> authorization -> executor
-> timeout/retry/idempotency -> observation -> independent verifier
```

## 6. 证据分层和负面结论

可以确认：榜单 identity、AA 第三方指标、官方托管 alias、输入输出模态、上下文/输出合同、区域 capability matrix、工具协议名称、缓存/前缀能力、官方价格/限流页面字段和 GUI/mobile/productivity 定位。

不能确认：参数规模、dense/MoE 结构、层数、attention 变体、视觉 encoder、训练数据规模、optimizer、RL 算法、奖励模型、GUI policy、手机 executor、完整 serving kernel、真实设备成功率、线上 acceptance、独立技术报告或独立 benchmark 复现。

本轮没有检出可归属于 Qwen3.7 Plus 的独立 arXiv 技术报告或公开权重仓库。Qwen3.5、Qwen3.8、Qwen3-VL、Qwen3-Omni 的技术不能因为名称相近就回写为 Qwen3.7 Plus 的内部架构；它们只能作为系列背景或对照，并必须保留版本边界。

## 7. 书系落点与状态

- 第十五册：补充视觉/视频输入、GUI screenshot、媒体 revision 和 1M context 的有效预算账本。
- 第十七册：补充交互式视觉 Agent 的 proposal、permission、executor、observation、idempotency 和 verifier 链。
- 第二十册：补充 region/scope capability manifest、模型 alias/snapshot、tool schema 和状态回放边界。
- 第二十四册：补充长输入分档、prefix/cache、视觉 prefill 与 tool-serving 的部署门禁。
- `INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`PAPERS.md`、`KNOWLEDGE_GRAPH.md`、`BOOK_SERIES.md`、`ROADMAP.md` 均同步本锚点。

当前结论是 **AA 单榜资料级闭环**。资料足以支持面试中的 API/runtime/Agent 责任边界回答，不足以支持 Qwen3.7 Plus 专属 Transformer 架构、参数量、训练 recipe 或生产性能结论。

## 8. 快照摘要

| 快照 | 内容 | 大小 | SHA-256 |
|---|---|---:|---|
| `/tmp/q37-aa-20260922.out` | Artificial Analysis Qwen3.7 Plus 详情页 | 3,859,009 | `19e9b48bbc9c6d38fab3391bee6353cff5aaa2524bdf04589ae02bbad4a27040` |
| `/tmp/q37-q37-doc-hyphen-20260922.out` | Alibaba Cloud Qwen3.7 Plus 官方模型文档 | 44,386 | `662ebedd3a538e22490e00c4eb29def16c0c5e9ff2de1d3f8fe7d7d334b43656` |
| `/tmp/q37-model-studio-20260922.out` | Alibaba Cloud 推荐模型页 | 27,344 | `9df37ab1723f032187895b9961865f2831cb575147985a281310d0616f8f4fe8` |
| `/tmp/q37-q37-api-20260922.out` | Model Studio OpenAI-compatible API 文档 | 60,448 | `ec3f4a6c3db9ea89fc4141e78cfcfaf389aafa67ff11efec664b3600aa40714` |
| `/tmp/deepswe-20260922` | DataCurve DeepSWE 当前页面 | 268,036 | `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1` |

