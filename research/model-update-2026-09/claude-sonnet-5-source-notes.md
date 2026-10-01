# Claude Sonnet 5：排行榜、官方 API 与 Agent 运行时核验

核验日期：初次核验 2026-09-15；排行榜/System Card 快照复核至 2026-09-28，官方 API/Agent 文档复核至 2026-09-28。Sonnet 5 的候选身份来自 Artificial Analysis 与 DataCurve DeepSWE；其余页面只用于核验已发现锚点的官方事实和周边技术。本文把基础模型、推理档位、fallback、provider、harness、工具和 verifier 分开记录。

## 证据入口与联网复核

- 排行榜锚点：[Artificial Analysis Claude Sonnet 5](https://artificialanalysis.ai/models/claude-sonnet-5)；[DataCurve DeepSWE](https://deepswe.datacurve.ai/)。
- 官方资料：[Anthropic Models Overview](https://platform.claude.com/docs/en/models/overview)、[Sonnet 5 模型页](https://platform.claude.com/docs/en/models/sonnet-5/overview)、[What's New](https://platform.claude.com/docs/en/models/sonnet-5/whats-new-sonnet-5)、[Migration guide](https://platform.claude.com/docs/en/models/sonnet-5/migration-guide)、[Prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5)、[发布公告](https://www.anthropic.com/news/claude-sonnet-5)、[Sonnet 5 System Card](https://www.anthropic.com/claude-sonnet-5-system-card)。
- 官方 API/Agent 文档：[Adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)、[Effort](https://platform.claude.com/docs/en/build-with-claude/effort)、[Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)、[Compaction overview](https://platform.claude.com/docs/en/build-with-claude/compaction)、[On-demand compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand)、[Threshold compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold)、[Compaction and preserved thinking](https://platform.claude.com/docs/en/build-with-claude/compaction-thinking-blocks)、[Background compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-background)、[Keep recent turns](https://platform.claude.com/docs/en/build-with-claude/compaction-keep-recent-turns)、[Preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking)、[Programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)、[Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)。
- 重新联网后，代理 `10.237.126.170:1234` 与 `10.24.27.134:8098` 对排行榜和 Anthropic 发布页均可用；`10.24.27.134:7890` 可访问部分页面，但 Artificial Analysis 大页面仍有 TLS/读取超时。因此单个代理失败不作为“页面不存在”的证据。
- 2026-09-28 本工作区显式使用 `curl --proxy http://10.24.27.134:7890` 得到百度 HTTP 200、2,381 bytes；Artificial Analysis Sonnet 5 详情、DataCurve DeepSWE 和下列 Anthropic 文档也均取得完整响应。代理连通性只对本次 URL/时点成立，不等于所有目标都可达。

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

2026-09-21 当前快照（用于与上表的 2026-09-15 历史值对照）：

| 内容 | 文件与大小 | SHA-256 |
|---|---:|---|
| Artificial Analysis Sonnet 5 详情页 | `/tmp/aa-sonnet5-20260921.html`，3,872,014 bytes | `2bd51075cf1ad426140a25dddb097e62af0ef4d07308c00fbc8c022b9158318e` |
| DataCurve DeepSWE | 当前快照 268,571 bytes | `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870` |
| Anthropic Sonnet 5 发布页 | `/tmp/sonnet5-release-20260921.html`，262,651 bytes | `61714676503e6aafa41689e7ad7fa27738983e244a31bbd9acbc95d8728925a0` |
| Sonnet 5 模型 overview | `/tmp/sonnet5-overview-20260921.html`，461,560 bytes | `57d20b24a8d7961bd2ea76d71080035677ec27deac07991bcc73cc3d305a03b5` |
| Sonnet 5 System Card | `/tmp/sonnet5-system-card-20260921.html`，10,786,341 bytes | `33573adb9f1871b903f79b77d7755a44cc20bb913ef48ac9000ddb090e41f7ed` |
| System Card 标准库文本提取 | `/tmp/sonnet5-system-card-extracted.txt`，约 237K 字符 | `3daed308eaedefa8ea61fe3bcfcc99a1206d03dcb02148dc63ba8ffdea9c47f6` |

2026-09-28 当前排行榜与 Anthropic 官方 Markdown 快照：

| 内容 | 文件与大小 | SHA-256 |
|---|---:|---|
| Artificial Analysis Sonnet 5 详情页 | `/tmp/aa-sonnet5-20260928.html`，3,849,279 bytes | `16557ff25caed39a04c1481ba199cd09eeb213db8a33534ff9c291b58fbce8a1` |
| DataCurve DeepSWE | `/tmp/datacurve-7890-20260928.html`，268,036 bytes | `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1` |
| What's New | `/tmp/sonnet5-whatsnew-20260928.md`，13,195 bytes | `02f0df53f8780f93ca88c9691f530f07032ebd5d41c88ec838fc0f9aa85007db` |
| Migration guide | `/tmp/sonnet5-migration-20260928.md`，38,656 bytes | `d53d15d2b5a4c30fbf75e4181954bc1748472dd3186a65f6dac517255857a2ca` |
| Prompting guide | `/tmp/sonnet5-prompting-20260928.md`，16,479 bytes | `07e7db846f96915a1d533fe6f62f7812790a966361b9982b6522557e74d08127` |
| Compaction overview | `/tmp/sonnet5-compaction-20260928.md`，11,974 bytes | `351fe6c8454b3762356aa0066575ad369506b289ecfcf3af1fad2c060a43c9cc` |
| On-demand compaction | `/tmp/sonnet5-on-demand-20260928.md`，46,724 bytes | `823190dbde68807fe73a9db88e83e37b05857395af7d0357b93588fdaa30e39d` |
| Threshold compaction | `/tmp/sonnet5-threshold-20260928.md`，113,841 bytes | `5877f49f14e6192d050829e64d5149e381f4ccbf8933dac9a2e0a34e1515d748` |
| Compaction/preserved-thinking rules | `/tmp/sonnet5-thinking-blocks-20260928.md`，30,956 bytes | `5a3c28c48516182033b5f83078e22ae729257d5b06a067bc1aad3a93a78f409e` |
| Background compaction | `/tmp/sonnet5-background-20260928.md`，23,242 bytes | `1a40328fd97ecebfbd6ca1e4092cc522f2afdb3f0052a950118cd9d9fa8b560f` |
| Keep recent turns | `/tmp/sonnet5-keep-recent-20260928.md`，21,987 bytes | `81682a12b3933d9dbff4a1cbfea739ad5479ada0c5c1e5bf4840488d42776745` |
| Preserved thinking | `/tmp/sonnet5-preserved-thinking-20260928.md`，113,656 bytes | `a93aaa211a4ca45a1dea9532f77f43652d3bf2b6e01ca14298277089e0f09457` |
| Models Overview | `/tmp/sonnet5-models-overview-20260928.md`，17,052 bytes | `404372e69b13defcb4598bb8a8cfbc359c0c31384d809daf53d75c3ca9316bc3` |

当前 AA 页面相对 9 月 15 日的 `38.3576962882576` Intelligence Index、约 80.00 tokens/s 和约 202.63s TTFT，变为 `38.1638712882576`、`86.1790204452512 tokens/s` 和 `137.660255319s`。这是第三方 provider/测量时点的字段漂移；没有 revision、权重或训练变更证据，不能把差异解释成模型升级。当前页面还保留 1M context、输入/输出 `$2/$10/M`、cache hit `$0.20/M`、约 `$5.0912`/task 和约 8 个 Anthropic API provider 的配置级语义。

## Artificial Analysis：基础模型与配置级字段

页面 canonical slug 和模型 ID 都是 `claude-sonnet-5`，主详情页标题为 `Claude Sonnet 5 (Adaptive Reasoning, Max Effort)`。页面的 `releaseDate` 为 `2026-06-30`，当前未标记 deprecated；模型被标为 proprietary、非 open weights，`parameters` 为 null。

页面列出的 `max`、`xhigh`、`high`、`medium`、`low` 和 `Non-reasoning` 是同一基础模型的运行配置或页面变体，不是六个独立模型。主配置为 adaptive reasoning + `max` effort。页面字段还给出 1,000,000 context、输入 `$2` / 输出 `$10` 每百万 token、cache hit `$0.20/M`、cache write `$2.50/M`，以及约 `$5.0912`/Artificial Analysis Intelligence Index task；速度和 TTFT 以采集日期绑定，见上面的历史/当前对照。

9 月 15 日历史快照中的 Intelligence Index 为 `38.3576962882576`、约 80.0022 output tokens/s、约 202.6275 秒 TTFT；9 月 21 日当前快照为上段所列值。页面还说明 Intelligence Index v4.3 包含 10 个评测、整套评测累计约 `$6998.25`、产生约 `370M` output tokens，并使用 Anthropic API 的 8 个 provider。上述数字都是 Artificial Analysis 的第三方配置级测量，不能与 Anthropic 发布方评测或 DeepSWE 结果拼成一个裸模型排名。

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
- Sonnet 5 支持两种不同的 server-side compaction beta：threshold 模式通过 `context_management.edits` 使用 `compact_20260112` / `compact-2026-01-12`；on-demand 模式使用顶层 `compaction: {type: "summarize"}` / `compact-2026-09-04`。两者不是同一接口，也不能在同一请求混用；其 block 放置、历史替换方式与支持平台不同。compaction 是长任务状态管理能力，不能被描述为模型自动获得了无限上下文。
- `computer_toolset_20260801` 可用于 Claude API 和 Google Cloud；browser use、web fetch 等能力仍受宿主权限、网络、沙箱、审批和审计控制。
- programmatic tool calling 通过 `code_execution_20260120` 或更新版本，让模型在代码执行容器内批量调用工具，减少模型往返和部分输入 token。文档中 BrowseComp/DeepSearchQA 的收益是通用文档结果，不能写成 Sonnet 5 专属 benchmark。
- Sonnet 5 不支持 mid-conversation system messages，应使用请求顶层 `system` 字段；工具协议和多轮历史必须保留服务端要求的 block 顺序与签名。

这些接口事实的面试价值在于：Agent 的能力由模型、协议、工具执行器、权限策略、上下文压缩和 verifier 共同决定。模型页支持某个工具，不代表宿主已经提供该工具，也不代表工具结果可信。

## System Card 深读：训练公开性、安全与评测边界（2026-09-21）

System Card 公开的是安全/能力评测所需的高层信息，不是完整训练报告。训练数据只被描述为专有的混合数据，包含公开互联网、公开和私有数据、合成数据，以及去重、分类、ClaudeBot 爬取、post-training/fine-tuning 等处理阶段；没有公开完整数据组成、优化器、训练超参数、RL 目标或可独立复现的训练 recipe。多数评测由 Anthropic 内部执行，通常使用最终 snapshot 和默认 safeguards；卡片会在个别能力或安全实验中明确关闭 safeguards。因而“有 System Card”不等于“训练方法和评测可以独立复现”。

RSP/能力风险部分把 Sonnet 5 放在不跨越 automated AI R&D threshold 的位置；Autonomy threat model 1 适用，但 stealth rate 接近零；CB-2 也未跨越。这些是发布方风险框架中的分级结论，不能换算成通用的“安全分数”。网络安全评测没有针对网络安全做专项训练，覆盖 ExploitBench、OSS-Fuzz、CyberGym 和 Firefox 147 等任务；在生产默认 mitigations 下，部分 cyber 评测结果为 0。阅读时要同时保留任务、safeguard 和执行环境，否则容易把“拒绝执行”与“不会完成能力任务”混为一谈。

Agentic safety 新增了两类拒答与攻击面证据：Claude Code 恶意请求拒答率为 `92.37%`，computer-use 恶意任务拒答率为 `84.68%`；Gray Swan IPI benchmark 覆盖 coding、computer use 和 tool use，共 28 个场景，去重后有 1,130 个攻击样本。它们测量的是给定 harness、工具和 safeguards 下的行为，不是模型内部安全模块的独立测量；宿主仍需通过权限、沙箱、allowlist、人工确认和 verifier 控制副作用。

### 发布方 benchmark 的配置标签

System Card 的常见能力配置为 adaptive thinking + `max` effort，通常运行 5 个 trials，context 上限随评测变化；BrowseComp 使用 10M token limit，并在 200K 附近触发 compaction。代表性发布方或其 system-card harness 结果如下：

| 评测 | 发布方结果 | 解读边界 |
|---|---:|---|
| SWE-bench Verified | 85.2% | 发布方 harness 结果 |
| SWE-bench Pro | 63.2% | 任务集和执行配置绑定 |
| Multilingual SWE-bench | 78.3% | 多语言任务配置绑定 |
| Terminal-Bench 2.1 | 80.4% | `xhigh`、`mini-SWE-agent` 配置 |
| BrowseComp | 84.7% | 10M token limit/compaction 语义 |
| OSWorld-Verified | 81.2% | computer-use harness 结果 |
| GDPval-AA v2 | Elo 1618 | 发布方比较评测 |
| Toolathlon Pass@1 | 54.3% | 工具、任务环境和 verifier 绑定 |
| AA-Briefcase | Elo 1393 | 发布方评测配置绑定 |

安全相关的发布方材料还报告 Gray Swan bug-bounty unique attack success 约 `0.19%`；Shade coding 在无 safeguards 时 thinking/disabled 分别为 `0.31%/0.29%`，有 safeguards 时为 `0.09%/0.13%`。评测 awareness 约影响 6% rollouts。以上数字必须与 Artificial Analysis 的 Intelligence Index、DataCurve 的 Pass@1/Pass@4 分开记账：三者的模型配置、harness、任务、verifier、统计和执行者不同，不能组成统一排名。

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

- 截至 2026-09-21，针对精确标题 `"Claude Sonnet 5"` 的 [arXiv 标题检索](https://arxiv.org/search/?query=%22Claude+Sonnet+5%22&searchtype=title) 仍返回 0 个结果。
- Anthropic [Research 页面](https://www.anthropic.com/research) 可访问，但本轮公开列表未检出 Sonnet 5 专属技术报告。
- 当前没有公开确认的 Sonnet 5 参数规模、稠密/MoE 架构、注意力变体、完整训练数据、优化器、后训练 recipe、内部 adaptive-thinking 机制或可独立复现的训练报告。

以上是截至 2026-09-21、在上述公开入口中的负面检索结果，不是对未来发布论文或技术报告的绝对否定。官方模型目录、API 文档和排行榜配置字段不能替代这些缺失的内部证据。

## 面试主线与书系映射

建议围绕以下问题准备：

1. 为什么同一个 Sonnet 5 在 low/max 下的成功率、成本和输出长度不同，却不能称为两个模型？
2. 如何把 Artificial Analysis 的 Intelligence Index、DeepSWE 的 Pass@1 和 Anthropic 自报 SWE-bench 结果放在不同证据层，而不做错误横向排序？
3. 为什么 adaptive thinking 的 effort 不是 `budget_tokens`，`max_tokens` 又为什么仍然必要？
4. 工具调用、多轮 thinking signature、compaction 和 tokenizer 变化会怎样影响缓存、延迟、成本和可回放性？
5. 如何把一个 Agent 的“我完成了”转化为测试、lint、类型检查和领域 verifier 可检查的 artifact？

内容映射到第四册（模型/API 字段与上下文边界）、第六册和第二十四册（tokenizer、cache、TTFT/TPOT、并发与成本）、第七册（固定条件评测）、第十七册和第二十册（thinking block、工具循环、compaction、权限和状态恢复）。目前没有足够公开的 Sonnet 5 独立架构/训练证据，因此不新增 Sonnet 5 专属正式架构章节。

## 2026-09-28：Compaction 双接口、回放约束与迁移提示

本轮用 `10.24.27.134:7890` 重新获取上述 11 份 Anthropic 官方 Markdown 文档，均返回 HTTP 200、`text/markdown`；9 月 28 日早前留下的 What's New/Migration/Prompting 70-byte 响应不是正文，本节只依据这次完整快照。没有调用 Anthropic Messages API，也没有把文档合同写成真实 endpoint 行为验收。

### 两条 server-side compaction 合同不可混为一谈

| 项目 | On-demand | Threshold / automatic |
|---|---|---|
| 版本 / 参数 | beta `compact-2026-09-04`；顶层 `compaction: {"type":"summarize"}` | beta `compact-2026-01-12`；`context_management.edits` 中的 `compact_20260112` |
| 触发与流转 | 应用主动发出独立 summarization 请求；返回单个 `compaction` block，`stop_reason: "compaction"`，不生成普通回复 | 普通 Messages 请求达到设定 input-token threshold 时在本次响应中压缩并继续；默认 trigger 为 150K、最小 50K，可用 `pause_after_compaction` 控制是否在 summary 后暂停 |
| 后续历史 | block 带可读 summary 与 signature；后续每次请求都要带 beta header，历史只留一个最新 block，将它放在 `messages` 最前并删除其已总结的前缀；block 内容/signature 原样回传 | response 按正常方式追加；block 在服务端响应中替代旧历史，后续 API 自动丢弃 block 之前的内容 |
| 关键边界 | 不能与 `context_management` 同请求；不能在上下文已超限后才请求压缩；不能留下未返回结果的 assistant tool call | 不能与携带 on-demand signed block 的 request 组合；threshold 路径不是单独的摘要请求 |

两页的当前支持列表都包含 `claude-sonnet-5`，但平台覆盖不完全相同：on-demand 文档列 Claude API、Claude Platform on AWS、Google Cloud、Microsoft Foundry beta，不列 Amazon Bedrock；threshold 文档还列 Amazon Bedrock beta。可用带相应 beta header 的 Models API 读取 `capabilities.compaction`，不要把某一平台能力外推到其他 provider。

On-demand 的失败也不能只看 HTTP status：summary request 可返回 HTTP 200 但 `content` 为空，此时应先检查 `stop_reason`；正常 compaction 才是 `stop_reason: "compaction"`。摘要调用记在 `usage.iterations`，顶层 input/output token 可能为 0。常见 400 包括 `compaction_block_misplaced`、`compaction_signature_invalid`、`compaction_content_mismatch`、重复 block，或最后一个 assistant tool call 尚无 tool result。图片、文档、`container_upload` 和被抓取 URL 等 summarized payload 不会随文本摘要保留，后续仍需要时必须重新附上或转成可引用 artifact。

长任务可以选择保留近期尾部，或让摘要在后台生成。Keep-tail 由应用选择切分点，tool call 与对应 result 不能被切开；后台模式在发起摘要时记录已提交前缀，等待期间只 append 新轮次，不能编辑旧历史或并行启动另一次 compaction，摘要到达后用 block 精确替换已提交前缀、保留新追加 tail。并发摘要会占用 rate limit 且同时打开两个请求，因此要为等待期间增长的上下文预留空间。

压缩之后的 cache 账本也会变化：summary 是新 cache content；官方建议在 system prompt 尾部设置 `cache_control` breakpoint，使系统提示仍可单独 cache，而只把摘要写成新 cache entry。`cache_control` 也可设在 compaction block 上，为摘要自身建立 breakpoint。

### Thinking 状态与证据边界

对确实保留 thinking 的 keep-tail/background 流程，官方给出的有效条件包括：参与 compaction 的模型具备相应 preserved-thinking 能力；保留消息紧邻 summary 覆盖边界并原样回放；`system` 与未延迟加载的 `tools` 保持一致。把“服务端接受 compaction block”与“每个保留 thinking block 仍通过会话绑定检查”分开。尤其要注意：当前 Preserved thinking 文档把 prefix-binding 检查描述为从 Claude Fable 5.1 开始；Claude Sonnet 5 属较早模型，不运行这项 prefix check。因此这些条件可用作安全的 harness 回放不变量，但不能声称 Sonnet 5 会因 prefix mismatch 必然 400 或 drop。

Prompting/Migration 文档补充了 Agent 面试点：Sonnet 5 默认开启 adaptive thinking，effort 是深度控制而非固定 token budget；`max_tokens` 同时覆盖 thinking 与最终文本，high/xhigh/max 时需给输出留空间。Anthropic 的 Sonnet 5 prompting 指南称 high/xhigh 在 coding/search 场景会更频繁触发工具和自验证，并建议按观测到的 thinking 长度而不是跨模型相同 effort 名称来校准；这是发布方操作建议，不是独立因果实验。迁移时 `thinking` block 可能先于 text block，客户端要按 `content[].type` 解析并在工具回合原样回传 block；默认 thinking display 为 `omitted`，并且非默认 `temperature/top_p/top_k` 会返回 400。网络安全请求若被拒，文档说明这是 HTTP 200 + `stop_reason: "refusal"`，需作为正常业务结果分支处理。

## 当前结论

Claude Sonnet 5 已完成“排行榜锚点 → 官方模型/API/Agent 资料 → System Card 深读/发布评测 → arXiv/Research 负检索 → 研究笔记”资料级闭环；2026-09-28 追加复核了 compaction 双接口、签名 block 回放、背景 tail、cache 和 migration/prompting 合同。闭环不表示内部架构和训练 recipe 已公开，也不代表已对真实 API 做集成验收；System Card 的安全数字不能替代独立复现。后续模型锚点仍只能从两个排行榜的重点厂商 canonical 条目中选择，不能仅因 Anthropic 官方目录中的其他名称直接新增候选。
