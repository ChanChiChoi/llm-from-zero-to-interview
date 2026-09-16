# Claude Fable 5.1：Anthropic 官方模型页核验

核验日期：2026-09-15（首次模型页字段来自 2026-09-10 缓存；本次重新联网复验并补抓 Anthropic 发布页、排行榜和 System Card）。主要来源为 [Claude Fable 5.1 专属模型页](https://platform.claude.com/docs/en/models/fable-5-1/overview) 与 [Anthropic 官方发布页](https://www.anthropic.com/claude-fable-and-mythos-5-1)。当前模型页 URL 在本环境返回区域不可用跳转，因此仍区分缓存字段、当前发布页正文和第三方榜单字段，不把一次代理/区域失败解释成资料不存在。

## 已确认字段

- 模型 ID 为 `claude-fable-5-1`，官方页面标为 Claude Fable 5.1，生命周期为 active，发布日期字段为 `2026-09-01`，退休不早于 `2027-09-01`。
- 上下文窗口为 1,000,000 tokens，最大输出为 128,000 tokens；页面结构化模型配置给出文本/图像输入、文本输出。
- 页面定位为 demanding reasoning 和 long-horizon agentic work；页面明确建议大多数工作负载先使用 Opus 5，仅在更高 effort 的 Opus 5 仍不足时评估 Fable 5.1。
- 思考字段为 `adaptive (always on)`，默认 effort 为 `high`；页面将 adaptive thinking 描述为 Fable 5.1 唯一的 thinking 模式，并通过 `effort` 控制深度。
- 平台字段包含 Claude API、Amazon Bedrock、Google Cloud、Microsoft Foundry 和 Claude Platform on AWS；Bedrock 模型 ID 为 `anthropic.claude-fable-5-1`。
- 页面价格字段为输入每百万 token 10 美元、输出每百万 token 50 美元；缓存写入和读取价格另列，页面说明相比 Fable 5，cache read 为四分之一价格。
- 页面列出 preserved thinking、跨轮模型切换/思考块、per-message effort（beta）、turn-scoped system messages（beta）、工具调用间进度更新（`display: "updates"`，beta）和 content provenance 等能力入口。
- 页面提到 Claude Mythos 5.1 与 Fable 5.1 共享规格和价格，但 Mythos 5.1 通过 Project Glasswing 邀请制提供；本项目不把它当作独立公开可用模型。

## 2026-09-15 重新联网复验

本轮使用 `10.237.126.170:1234` 和 `10.24.27.134:8098` 抓取两个排行榜；两者对目标页面均返回 HTTP 200。`10.24.27.134:7890` 仍可作为备用代理，但对 Artificial Analysis 大页面存在 TLS EOF/读取不稳定。复验快照均为临时文件，不作为仓库长期数据文件：

| 页面 | 临时快照 | 大小 | SHA-256 |
|---|---|---:|---|
| Artificial Analysis Fable 5.1 详情 | `/tmp/recheck-aa-fable51-20260915.html` | 3,615,956 bytes | `aa253dd4d4e8ad3285f60d0910f698239af5bfcefd268bb0127e41a74c93ba4b` |
| Artificial Analysis 中文首页 | `/tmp/recheck-aa-home-20260915.html` | 1,773,775 bytes | `8589909c7199e311a750a2b3c0435e139508da4c6343ae063c935d0c3dbd1a98` |
| DataCurve DeepSWE | `/tmp/recheck-deepswe-20260915.html` | 268,313 bytes | `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7` |
| Anthropic Fable/Mythos 5.1 System Card | `/tmp/recheck-anthropic-fable51-system-card-page-20260915.bin` | 16,397,488 bytes | `b0d59edc7a60eef32a879c13d713cce60c3fefd7e6b5183afdc8b835af3c8c39` |

Artificial Analysis 当前详情页确认 canonical slug `claude-fable-5-1`，主配置为 `Claude Fable 5.1 (Adaptive Reasoning, Max Effort, Default Fallback)`，内嵌 `releaseDate` 为 `2026-09-01`。该配置的第三方字段为 Intelligence Index `53.3737509623252`、median output speed `65.9769929669683` tokens/s、median time to first chunk `212.0122070875` 秒、1,000,000 context tokens、输入/输出 `$10/$50` 每百万 token，以及约 `$7.6297` 每个 Intelligence Index task。页面还提供 `xhigh/high/medium/low` 与 fallback 变体；这些都是配置级测量，不能写成 Fable 5.1 的裸模型分数或普遍 API 延迟。

DataCurve 当前页面仍是 DeepSWE v1.1：113 个任务、91 个仓库、5 种语言、统一 `mini-swe-agent`，并有 `claude-fable-5` 的 `xhigh/max/high/medium/low` 行；本次页面没有 `claude-fable-5-1` 行。因此 Fable 5.1 不记录 DataCurve 分数，也不把 Fable 5 的 316/452 结果迁移给 Fable 5.1。该“未出现”结论仅针对本次 2026-09-15 页面快照。

Anthropic 当前发布页显示月份为 September 2026，并确认 `claude-fable-5-1` 已可在各平台使用。官方模型页缓存提供的精确日期、1M context、128K output、adaptive always-on 和默认 high 字段与发布页相互吻合；模型页当前被区域跳转替代，故本笔记保留缓存快照的证据边界。

## 发布页新增的公开技术与运行时知识

### 同一底模、不同 safeguards

Anthropic 明确说 Fable 5.1 与 Claude Mythos 5.1 是同一个 underlying model，差异在 safeguards：Fable 5.1 面向一般可用，Mythos 5.1 通过受信任访问计划为网络安全和生命科学场景提供更宽松的安全边界。面试时应把“模型权重/底模”和“服务策略、分类器、访问计划”分层，不能把 Mythos 当作另一个公开基础模型，也不能把安全策略差异误写成模型架构差异。

### 缓存价格是长任务成本优化点

发布页把 Fable 5.1 的 cache read 价格写为 `$0.25`/百万 token，即相较此前价格降低 75%；其他基础输入/输出价格仍为 `$10/$50` 每百万 token。Anthropic 以 2026 年 8 月四周、默认 effort 的使用数据估计：典型工作负载成本约低 25%，上下文密集且工具密集的高度 Agent 化工作负载最高约低 45%。这是发布方的使用成本分析，不是本项目独立复现；但它适合引出一个面试要点：长任务中 cache hit ratio、前缀稳定性和工具结果组织，可能比单次 token 单价更决定总成本。

### effort 与产品入口并不完全相同

发布页说明 Fable 5.1 在 Claude Code 默认 High effort，而在 Claude Cowork 和 Claude.ai 默认 Medium effort；官方模型页还列出 Low/Medium/High/Xhigh/Max 配置。比较不同产品或榜单时，必须同时记录入口、effort、工具、fallback、上下文策略和任务预算，否则“同一模型”并不意味着同一推理条件。

### 安全策略也会改变 benchmark 结果

发布页的官方 benchmark 表（发布方自报）包含：Terminal-Bench-Science 0.1 为 52.6%，Terminal-Bench 4.0 为 55.8%（Fable 5.1）/60.9%（Mythos 5.1），GDPval-AA v2 为 1853，OSWorld 2.0 为 77.9%（partial）/41.7%（strict），Humanity’s Last Exam 为 60.9%（无工具）/65.0%（有工具），AutomationBench 为 31.4%，CursorBench 3.2.0 为 73.4%。

这些数字必须带上“Anthropic 发布方结果”标签：OSWorld 使用 benchmark 作者 2026 年 8 月任务发布并对 Fable 5、Opus 5 重跑，不能与旧任务文件结果直接比较；生产 safeguards 介入的任务可能记为零，其他网络安全/生命科学介入可能由 Opus 4.8/Opus 5 处理。因而 benchmark 的被测对象实际是“模型 + effort + 工具 + safeguards + fallback + verifier”的系统，而不是只由模型名称决定的函数。

### 安全、隐私和 anti-distillation 机制

- Enterprise Frontier Safeguards（EFS）把客户数据放在客户控制的云基础设施中，目标是在零数据保留语义下进行滥用检测；这是部署与治理层技术，不是模型参数披露。
- Anthropic 表示新的网络安全 safeguards 减少约 60% 的误报，并允许发现软件漏洞但不允许开发 exploit；生命科学相关的宽松访问通过 Mythos 受信任计划提供。发布页另称针对基础生物/医疗良性请求的 safeguards 触发率相对 Fable 5 发布时低 85%，研究开发类生命科学请求仍会被导向 Opus 模型。
- 新 API 账户不能在多轮对话中手动编辑 Claude 的历史上下文，同时保留此前 thinking transcript。官方把这作为提高蒸馏攻击成本的机制；面试上可将其理解为对“可编辑历史 + 可回放 reasoning 状态”组合的协议约束，而不是把签名或 thinking 当作可公开读取的思维链。

### 官方展示的科学工作流案例

发布页展示了 Fable 5.1/受信任 Mythos 5.1 在科学任务中的案例：模型训练用于生成金星高分辨率高程图；Mythos 5.1 使用开放的蛋白质设计和折叠工具生成候选 binder；还通过编写自定义 GPU kernel 并缓存中间结果，使七个开源深度学习模型在 NVIDIA H100 上最高获得约 2.5 倍推理加速且输出一致。它们是 Anthropic 发布的案例，不应反推 Fable 5.1 自身采用了何种 kernel、缓存算法或训练架构；但可作为“模型作为研究/性能工程 Agent”的面试案例。

## 论文与技术报告检索结果

2026-09-15 通过 arXiv 重新搜索后，精确标题查询 `"Claude Fable 5.1"` 返回 0 个结果；全文查询返回 4 篇提到 Fable 5.1 的论文：

- [Long runs of integers with small prime factors and the divisor function of $n!$](https://arxiv.org/abs/2609.15597)：作者说明一个改进思路在与 Fable 5.1 的私下交互中形成，并由作者自行验证和呈现。它是外部使用案例，不是 Fable 5.1 技术报告。
- [The Troy Moment of AI: Why SomeWill Cheat and SomeWill Follow?](https://arxiv.org/abs/2609.15494)：用 Fable 5.1、GPT-5.6 Sol 和 Gemini 3.8 Flash 研究 ImpossibleBench 上的边界遵守、升级/停止和多 Agent 互动；它更适合补充 Agent 安全与 harness 评测知识。
- [Pierce-Birkhoff conjecture is false](https://arxiv.org/abs/2609.10420)：作者称反例发现使用了包含 Fable 5.1 的多模型、多 Agent 链路；论文结论仍由作者负责，不等于模型独立证明。
- [Coloring graphs with no long induced path](https://arxiv.org/abs/2609.08847)：作者称证明在 Fable 5.1 与 GPT Pro 协助下发展；同样属于外部协作案例。

Anthropic Research 页面本次可访问，但未检出 Fable 5.1 专属技术报告条目；新下载的官方 [Fable 5.1/Mythos 5.1 System Card](https://www.anthropic.com/claude-fable-5-1-mythos-5-1-system-card) 为 PDF，当前环境仍没有稳定的正文提取器，因此只记录文件和官方发布页中的安全摘要，不从 PDF 二进制或目录猜测安全数值。

## 页面自述与证据边界

Anthropic 页面声称 Fable 5.1 带来更强的长期 Agent coding、多步研究以及文档、表格和幻灯片工作能力。这些是发布方定位和产品页声明，不是本项目独立 benchmark 复现，也不等于公开了训练配方或内部架构。

`adaptive (always on)`、effort、preserved thinking 和 beta 协议描述的是接口/运行时行为。它们不能证明模型是 MoE、稠密或采用某个特定 test-time compute 算法。跨 Opus/Fable 比较必须固定平台、模型 revision、工具、任务、超时、输出上限和 harness。

## 尚待核验

官方模型页没有披露参数规模、稠密或 MoE 结构、训练数据、优化器、后训练算法、完整技术报告或独立 benchmark 复现。页面自述的质量提升、长任务优势和 cache 成本需要在固定条件下独立测量。

## 书系映射

- 第四册：Fable 5.1 的接口字段、thinking/effort、平台与 beta 协议。
- 第六册与第二十四册：1M context、KV/cache、长任务成本、TTFT/TPOT。
- 第七册：Opus/Fable 同任务、同平台和同 harness 的公平对照。
- 第十七册与第二十册：preserved thinking、进度更新、工具宿主和状态恢复。

## 当前闭环判断

Fable 5.1 已完成“排行榜发现 + 官方发布页/模型页字段 + System Card 入口 + 论文定向检索 + 研究笔记”的资料级闭环。没有 DataCurve Fable 5.1 结果，也没有公开参数规模、稠密/MoE 结构、训练/后训练 recipe、独立技术报告或独立 benchmark 复现，因此不新增独立架构章节。下一步切换到同样属于重点厂商、但仍只有部分官方字段核验的 `claude-sonnet-5`，优先补官方发布页、system card、API 运行时协议和两榜单配置边界。
