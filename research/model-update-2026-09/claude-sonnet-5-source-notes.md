# Claude Sonnet 5：排行榜、官方 API 与 Agent 运行时核验

核验日期：2026-09-15。Sonnet 5 的候选身份来自 Artificial Analysis 与 DataCurve DeepSWE；其余页面只用于核验已发现锚点的官方事实和周边技术。本文把基础模型、推理档位、fallback、provider、harness、工具和 verifier 分开记录。

## 证据入口与联网复核

- 排行榜锚点：[Artificial Analysis Claude Sonnet 5](https://artificialanalysis.ai/models/claude-sonnet-5)；[DataCurve DeepSWE](https://deepswe.datacurve.ai/)。
- 官方资料：[Anthropic Models Overview](https://platform.claude.com/docs/en/models/overview)、[Sonnet 5 模型页](https://platform.claude.com/docs/en/models/sonnet-5/overview)、[发布公告](https://www.anthropic.com/news/claude-sonnet-5)、[Sonnet 5 System Card](https://www.anthropic.com/claude-sonnet-5-system-card)。
- 官方 API/Agent 文档：[Adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)、[Effort](https://platform.claude.com/docs/en/build-with-claude/effort)、[Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)、[Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)、[Programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)、[Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)。
- 重新联网后，代理 `10.237.126.170:1234` 与 `10.24.27.134:8098` 对排行榜和 Anthropic 发布页均可用；`10.24.27.134:7890` 可访问部分页面，但 Artificial Analysis 大页面仍有 TLS/读取超时。因此单个代理失败不作为“页面不存在”的证据。

本轮临时快照及 SHA-256：

| 内容 | 文件与大小 | SHA-256 |
|---|---:|---|
| Artificial Analysis Sonnet 5 详情页 | `/tmp/recheck2-aa-sonnet5-20260915.html`，3,614,205 bytes | `3e8257d0efec85f2cc30bf78b31ed3e5fa10c0be4da6ec55413de7cae9b2d9bf` |
| DataCurve DeepSWE | `/tmp/recheck2-deepswe-sonnet5-20260915.html`，268,313 bytes | `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7` |
| Anthropic Sonnet 5 发布页 | `/tmp/recheck2-anthropic-sonnet5-release-20260915.html`，264,609 bytes | `5cc40094a584f1821874847040cff912056845dac56bc770826d4dbcd11cb725` |
| Sonnet 5 System Card PDF | `/tmp/recheck2-anthropic-sonnet5-card-20260915.pdf`，10,786,341 bytes | `33573adb9f1871b903f79b77d7755a44cc20bb913ef48ac9000ddb090e41f7ed` |
| Anthropic 开发者文档完整快照 | `/tmp/recheck-anthropic-llms-full-20260915.txt`，34,581,554 bytes | `7461c0cb181ea163b368e8f77e6260b5b79cca08ff93df73be1b2e2b6d85abe2` |
| Anthropic Research 页面 | `/tmp/anthropic-research-live.html`，315,200 bytes | `eec4cf8b7332fe074fc536f9cfc691d0cd5f0079448582450435e403c8ca7b0d` |
| arXiv 精确标题检索页 | `/tmp/arxiv-sonnet5-title-live.html`，16,449 bytes | `742bc93981fcab07bd8db46c10bd46f4a7814f3a27e93856c40e915007fd2cee` |

## Artificial Analysis：基础模型与配置级字段

页面 canonical slug 和模型 ID 都是 `claude-sonnet-5`，主详情页标题为 `Claude Sonnet 5 (Adaptive Reasoning, Max Effort)`。页面的 `releaseDate` 为 `2026-06-30`，当前未标记 deprecated；模型被标为 proprietary、非 open weights，`parameters` 为 null。

页面列出的 `max`、`xhigh`、`high`、`medium`、`low` 和 `Non-reasoning` 是同一基础模型的运行配置或页面变体，不是六个独立模型。主配置为 adaptive reasoning + `max` effort。页面字段还给出 1,000,000 context、约 80.0022 output tokens/s、约 202.6275 秒 median time to first chunk/页面 TTFT、输入 `$2` / 输出 `$10` 每百万 token，cache hit `$0.20/M`、cache write `$2.50/M`，以及约 `$5.0912`/Artificial Analysis Intelligence Index task。

Artificial Analysis 页面显示 Intelligence Index 为 `38.3576962882576`（页面主视觉约显示 38），评测整套累计约 `$6998.25`，产生约 `370M` output tokens；页面说明使用 Anthropic API 的 8 个 provider。该快照还说明 Intelligence Index v4.3 包含 10 个评测。上述数字都是 Artificial Analysis 的第三方配置级测量，不能与 Anthropic 发布方评测或 DeepSWE 结果拼成一个裸模型排名。

## DataCurve DeepSWE v1.1：配置 + harness 结果

当前 DataCurve 页面仍是 v1.1：113 tasks、91 repositories、5 languages，统一使用 `mini-swe-agent`，页面生成时间为 `2026-09-03T22:24:37.984682+00:00`。Sonnet 5 的五档配置如下：

| effort | Pass@1 | Pass@4 | 平均成本 | 平均输出 token | 平均 Agent steps |
|---|---:|---:|---:|---:|---:|
| `max` | 53.846% | 78.761% | `$26.40` | 214,118 | 268.45 |
| `xhigh` | 49.667% | 75.221% | `$11.89` | 120,699 | 185.53 |
| `high` | 48.230% | 79.646% | `$7.43` | 87,295 | 146.58 |
| `medium` | 39.778% | 64.602% | `$4.08` | 56,817 | 107.61 |
| `low` | 30.512% | 57.522% | `$2.19` | 35,595 | 76.89 |

这些不是裸模型分数，而是“Sonnet 5 配置 + `mini-swe-agent` + 工具调用 + 任务集 + 仓库环境 + timeout/预算 + verifier”的系统结果。不能因为 `max` 的 Pass@1 高于 `low`，就断言基础模型参数或训练能力发生了对应变化；面试中应同时报告 effort、harness、工具、环境和 verifier。

## Anthropic 官方模型与产品字段

官方目录和 Sonnet 5 发布公告共同确认：

- 模型 ID 为 `claude-sonnet-5`，发布日为 2026-06-30；模型目录字段为 active，退休时间不早于 2027-06-30。
- context 上限为 1,000,000 tokens，普通最大输出为 128,000 tokens，Batch 最大输出为 300,000 tokens；支持文本和图像输入、文本输出。
- 思考模式是 Adaptive，默认 `output_config.effort` 为 `high`；目录延迟字段为 `Fast`。这些是产品/API 字段，不是内部推理算法的公开定义。
- Claude API、Amazon Bedrock、Google Cloud、Microsoft Foundry 和 Claude Platform on AWS 均列在接入平台中；不同平台的 model ID、版本、限流和能力仍需按目标账户复核。
- 价格为输入 `$2/M`、输出 `$10/M`；目录还列出 5 分钟 cache write `$2.50/M`、1 小时 cache write `$4/M`、cache read `$0.20/M`。
- 可靠知识截止和训练数据截止字段为 `2026-01`，不代表实时联网、搜索或外部工具权限。

发布公告把 Sonnet 5 定位为“最 agentic 的 Sonnet”：可规划、使用浏览器和终端，并自主执行多步任务；Anthropic 声称其性能接近 Opus 4.8、价格更低，并在 reasoning、tool use、coding 和 knowledge work 上优于 Sonnet 4.6。这些是发布方定位和评测声明，不能替代独立复现。

## 面试重点：Adaptive thinking、effort 与预算

### Adaptive thinking 不是固定 thinking token budget

Sonnet 5 使用 adaptive thinking。官方文档说明，Claude 4.7 及之后的模型不接受手动 `thinking.type: "enabled"` + `budget_tokens`；在 Sonnet 5 上使用这种请求会返回 400。应使用 `thinking: {"type": "adaptive"}`（或对应 SDK 形式）并通过 `output_config.effort` 选择 `low`、`medium`、`high`、`xhigh` 或 `max`。

`effort` 是行为信号而不是严格 token 预算：它影响思考深度，也可能影响工具调用、工具参数和可见输出；同一 effort 仍可能因任务难度产生不同 token 数。`max_tokens` 则是思考、工具调用和最终文本共享的硬上限。面试回答应把“目标行为信号”“每请求硬上限”和“整个 Agent loop 的预算”分成三个层次，而不是把 effort 当成 `budget_tokens` 的别名。

默认 thinking display 为 `omitted` 时，响应仍可能出现 thinking block 和 signature。`content` 是异构 block 序列，客户端必须按 `content[].type` 解析；多轮工具调用应原样回传需要保留的 thinking blocks/signatures，不能把响应粗暴拼成一段纯文本。

### 新 tokenizer 与旧预算不可直接迁移

发布页说明 Sonnet 5 使用新 tokenizer；同一段文本通常会产生约 30% 更多 token，内容依赖范围约为旧计数的 `1.0–1.35×`。因此旧模型上按 token 预算、上下文切分、缓存命中或成本估算得到的阈值不能直接复制到 Sonnet 5，必须重新 count tokens 并重新测量 TTFT、输出长度和成本。

### 请求参数与上下文管理边界

- 非默认 `temperature`、`top_p`、`top_k` 会返回 400；assistant prefill 不支持，Priority Tier 也不支持。
- context awareness 会自动追踪剩余上下文预算；1M 是接口上限，不等于 1M tokens 都具有相同的检索质量，也不等于 KV cache 容量或无限有效记忆。
- server-side compaction 支持 Sonnet 5，相关 beta header 为 `compact-2026-01-12`。compaction 是长任务状态管理能力，不能被描述为模型自动获得了无限上下文。
- `computer_toolset_20260801` 可用于 Claude API 和 Google Cloud；browser use、web fetch 等能力仍受宿主权限、网络、沙箱、审批和审计控制。
- programmatic tool calling 通过 `code_execution_20260120` 或更新版本，让模型在代码执行容器内批量调用工具，减少模型往返和部分输入 token。文档中 BrowseComp/DeepSearchQA 的收益是通用文档结果，不能写成 Sonnet 5 专属 benchmark。
- Sonnet 5 不支持 mid-conversation system messages，应使用请求顶层 `system` 字段；工具协议和多轮历史必须保留服务端要求的 block 顺序与签名。

这些接口事实的面试价值在于：Agent 的能力由模型、协议、工具执行器、权限策略、上下文压缩和 verifier 共同决定。模型页支持某个工具，不代表宿主已经提供该工具，也不代表工具结果可信。

## System Card 与发布方评测

System Card/发布页记录的代表性数字如下；它们属于 Anthropic 发布方或其 system-card 评测，不是 Artificial Analysis 或 DataCurve 的独立复现：

| 评测 | 发布方结果 |
|---|---:|
| SWE-bench Verified | 85.2% |
| SWE-bench Pro | 63.2% |
| Multilingual SWE-bench | 78.3% |
| Terminal-Bench 2.1 | 80.4%（xhigh，`mini-SWE-agent`） |
| BrowseComp | 84.7% |
| OSWorld-Verified | 81.2% |
| GDPval-AA v2 | Elo 1618 |

安全材料还报告 Gray Swan bug-bounty unique attack success 约 0.19%；Shade coding 在无 safeguards 时 thinking/disabled 分别为 0.31%/0.29%，有 safeguards 时为 0.09%/0.13%。发布页称 Sonnet 5 相比 Sonnet 4.6 的 undesirable behavior 总体更低、agentic safety 更好，并称其执行网络安全任务的能力明显低于当前 Opus 模型。评测 awareness 约影响 6% rollouts，这类因素应作为安全评测限制一起报告，而不是从单个分数推断普遍安全性。

## 论文、Research 与公开性负检索

- 截至 2026-09-15，针对精确标题 `"Claude Sonnet 5"` 的 [arXiv 标题检索](https://arxiv.org/search/?query=%22Claude+Sonnet+5%22&searchtype=title) 返回 0 个结果。
- Anthropic [Research 页面](https://www.anthropic.com/research) 可访问，但本轮公开列表未检出 Sonnet 5 专属技术报告。
- 当前没有公开确认的 Sonnet 5 参数规模、稠密/MoE 架构、注意力变体、完整训练数据、优化器、后训练 recipe、内部 adaptive-thinking 机制或可独立复现的训练报告。

以上是截至日期、在上述公开入口中的负面检索结果，不是对未来发布论文或技术报告的绝对否定。官方模型目录、API 文档和排行榜配置字段不能替代这些缺失的内部证据。

## 面试主线与书系映射

建议围绕以下问题准备：

1. 为什么同一个 Sonnet 5 在 low/max 下的成功率、成本和输出长度不同，却不能称为两个模型？
2. 如何把 Artificial Analysis 的 Intelligence Index、DeepSWE 的 Pass@1 和 Anthropic 自报 SWE-bench 结果放在不同证据层，而不做错误横向排序？
3. 为什么 adaptive thinking 的 effort 不是 `budget_tokens`，`max_tokens` 又为什么仍然必要？
4. 工具调用、多轮 thinking signature、compaction 和 tokenizer 变化会怎样影响缓存、延迟、成本和可回放性？
5. 如何把一个 Agent 的“我完成了”转化为测试、lint、类型检查和领域 verifier 可检查的 artifact？

内容映射到第四册（模型/API 字段与上下文边界）、第六册和第二十四册（tokenizer、cache、TTFT/TPOT、并发与成本）、第七册（固定条件评测）、第十七册和第二十册（thinking block、工具循环、compaction、权限和状态恢复）。目前没有足够公开的 Sonnet 5 独立架构/训练证据，因此不新增 Sonnet 5 专属正式架构章节。

## 当前结论

Claude Sonnet 5 已完成“排行榜锚点 → 官方模型/API/Agent 资料 → System Card/发布评测 → arXiv/Research 负检索 → 研究笔记”的资料级闭环。闭环不表示内部架构和训练 recipe 已公开；下一锚点必须继续从两个排行榜的剩余重点厂商候选中选择，不能仅因 Anthropic 官方目录中的 Haiku 4.5 等名称直接新增候选。
