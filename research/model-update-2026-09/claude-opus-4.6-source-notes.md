# Claude Opus 4.6 官方资料摘记

核验日期：2026-09-18

## 1. 锚点身份与榜单证据

- 重点厂商：Anthropic。
- Artificial Analysis 发现：[Claude Opus 4.6 adaptive](https://artificialanalysis.ai/models/claude-opus-4-6-adaptive) 与 [Claude Opus 4.6](https://artificialanalysis.ai/models/claude-opus-4-6)。二者是同一基础模型的运行配置，不能按 `adaptive`、`high` 或榜单展示名称重复计数。
- DataCurve DeepSWE v1.1 当前没有精确的 `mini_swe_agent_claude_opus_4_6_*` 行。因此不能把 Opus 4.8、Opus 5 或 Sonnet 4.6 的 Pass@1、成本、输出 token 或 Agent steps 迁移给 Opus 4.6。
- 本轮已核验的 AA 详情快照：adaptive `3,497,470` bytes，SHA-256 `0bcf25542c95489db50a83925caf5e32bab672e4f7134126009fb94937b6b32d`；基础页 `3,485,093` bytes，SHA-256 `544de10658d16de8ac25da44e6c6792ab51b64093a5a410f3eb3be1173e6d219`。
- AA 页面字段：第三方 `releaseDate` 为 `2026-02-05`，Intelligence Index 约 `32`（estimated），context `1M`，输出速度约 `37.7 tokens/s`，TTFT 约 `19.75s`，输入/输出价格 `$5/$25` 每百万 token，proprietary，参数量未公开。以上都是 Artificial Analysis 的配置级字段，不是 Anthropic 的内部架构或训练证据。

## 2. 官方资料

- [Anthropic 发布页](https://www.anthropic.com/news/claude-opus-4-6)
- [Claude Opus 4.6 System Card](https://www.anthropic.com/claude-opus-4-6-system-card)
- [模型页 Markdown](https://platform.claude.com/docs/en/models/opus-4-6/overview.md)
- [Models Overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)
- [Adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking)
- [Effort](https://platform.claude.com/docs/en/build-with-claude/effort)
- [Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)
- [Computer use](https://platform.claude.com/docs/en/agents-and-tools/computer-use)
- [Tool search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
- [Advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use)
- [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Claude Code Agent Teams](https://code.claude.com/docs/en/agent-teams)

发布页与 System Card 是 Anthropic 的一手材料；API 文档说明可观察的接口和运行时协议；Engineering 文章与 Agent Teams 文档说明 harness 设计，不应被改写成 Opus 4.6 的内部网络结构。

## 3. 已确认的模型与 API 边界

### 3.1 上下文、输出与 effort

`claude-opus-4-6` 是 2026-02-05 发布的 legacy snapshot。官方模型资料列出 1M context、普通请求 128K 最大输出，Batch beta 可到 300K。默认 `effort` 为 `high`，可选 `max`、`high`、`medium`、`low`；官方 effort 表没有列出 `xhigh`，不能把其他 Claude 版本的档位直接迁移过来。

`effort` 是行为信号，而不是严格的思考 token 预算。它会影响模型在推理、工具和回答之间的行为倾向；`max_tokens` 才是请求的硬上限。容量账本要分别计算输入、thinking、可见输出、工具 schema、工具结果和重试，不能用 `1M + 128K` 推导实际并发。

### 3.2 Adaptive thinking 与状态回放

Opus 4.6 支持 adaptive thinking。默认 thinking 关闭，客户端需要显式发送：

```json
{"thinking":{"type":"adaptive"},"output_config":{"effort":"high"}}
```

旧的手动 extended-thinking 入口已被标记为 deprecated；不要因为模型支持 thinking 就假定 `budget_tokens` 仍是推荐控制面。thinking block 对客户端只返回摘要或空内容，并带 encrypted `signature`；多轮工具循环必须原样回传相关 thinking block/signature，不能把它压成普通文本或自行拼接摘要。

### 3.3 Server-side compaction

Opus 4.6 的长任务可以使用 server-side compaction，beta header 为 `compact-2026-01-12`。默认 trigger 约为 150K input tokens，最低可设为 50K；触发后 API 返回 `compaction` block。这个 block 是用于继续任务的协议状态，不是普通摘要，也不等同于 prompt cache 或永久记忆。

可靠的 serving trace 至少保留模型 snapshot、effort、原始 output items、thinking signature、tool call/result、compaction block、权限决定、执行回执、workspace artifact 和 verifier 结果。只保存最终可见文本会导致下一轮缺失 opaque state，产生重复工具副作用或错误提前结束。

### 3.4 Tool search

Opus 4.6 支持基于 `defer_loading` 的 tool search，官方文档描述了 regex 与 BM25 版本；搜索结果默认最多返回 5 个 `tool_reference`。Anthropic 的工程资料指出，工具规模超过约 30--50 个时，直接把全部定义放进上下文会降低选择质量；按需加载通常可减少超过 85% 的工具定义上下文。

这解决的是工具 schema 的上下文选择问题，不是权限控制。生产链路仍然是：

```text
tool search -> tool_reference -> schema/permission gate
-> executor -> receipt/result -> next model turn -> verifier
```

搜索到工具、模型提出调用、宿主接受调用、工具产生副作用和最终 artifact 完整，是五个不同状态。面试中应分别说明它们的审计字段和失败处理。

### 3.5 Computer use

Opus 4.6 的 computer use 仍使用旧版 `computer_20251124` beta，不是新的 `computer_toolset_20260801`。模型只提出屏幕动作；执行器、沙箱、域名 allowlist、人工确认、截图回灌、取消和 prompt-injection 防护由宿主系统负责。不能从“支持 computer use”推导模型获得浏览器权限，也不能把动作提案当作动作已经发生。

## 4. 发布方评测与外部论文边界

Anthropic 发布页的数字按“发布方自报”记录：1M 8-needle retrieval 为 76%，BigLaw Bench 为 90.2%（40% perfect，84% 超过 0.8）；盲测中 40 次有 38 次被选为最佳。发布页还描述同一 Agent harness 最多使用 9 个 subagents 和 100+ tool calls。它们不能与 Artificial Analysis Intelligence Index 或 DataCurve Pass@1 合并为一个总分。

本轮 arXiv 精确标题检索没有找到 Anthropic 官方 Opus 4.6 技术报告，只找到两篇把该模型作为研究对象的外部使用论文：

- [Geographic Blind Spots in AI Control Monitors: A Cross-National Audit of Claude Opus 4.6](https://arxiv.org/abs/2604.13069)
- [Poisoned Identifiers Survive LLM Deobfuscation: A Case Study on Claude Opus 4.6](https://arxiv.org/abs/2604.04289)

这两篇只可作为外部评测/使用证据，不能写成 Anthropic 发布的模型技术报告。

## 5. 面试主线与待核验项

面试时优先回答五个问题：

1. 为什么 `effort`、`max_tokens`、context window 和实际 reasoning token 不能混成一个预算？
2. 为什么 thinking signature、tool result 和 compaction block 必须原样进入 replay？
3. tool search 如何降低 schema 上下文，却为什么不能替代权限系统？
4. 为什么 computer use 的执行器、allowlist、人工确认和 prompt-injection 防护必须属于宿主？
5. 为什么发布方 benchmark、AA 指数和 DataCurve DeepSWE 不能直接横向拼接？

仍待核验：参数规模、稠密/MoE 结构、注意力变体、训练数据、完整 pre-training/post-training recipe、adaptive thinking 内部机制、compaction 内部编码、生产 kernel、目标硬件 profiling、线上 acceptance rate 和 Opus 4.6 专属独立 benchmark 复现。当前状态为“资料级闭环”，不新增独立 Transformer 架构章节。

