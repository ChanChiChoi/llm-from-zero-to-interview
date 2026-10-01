# Claude Fable 5.1：Anthropic 官方模型页核验

核验日期：2026-09-21（2026-09-15 的首次复验记录保留为历史快照；本轮重新联网获取 Artificial Analysis 详情、Anthropic 模型页/发布页和 System Card 文件）。主要来源为 [Claude Fable 5.1 专属模型页](https://platform.claude.com/docs/en/models/fable-5-1/overview) 与 [Anthropic 官方发布页](https://www.anthropic.com/claude-fable-and-mythos-5-1)。模型页、发布页、榜单和 System Card 文件按各自证据责任记录，不把代理线路差异解释成模型事实。

## 已确认字段

- 模型 ID 为 `claude-fable-5-1`，官方页面标为 Claude Fable 5.1，生命周期为 active，发布日期字段为 `2026-09-01`，退休不早于 `2027-09-01`。
- 上下文窗口为 1,000,000 tokens，最大输出为 128,000 tokens；页面结构化模型配置给出文本/图像输入、文本输出。
- 页面定位为 demanding reasoning 和 long-horizon agentic work；页面明确建议大多数工作负载先使用 Opus 5，仅在更高 effort 的 Opus 5 仍不足时评估 Fable 5.1。
- 思考字段为 `adaptive (always on)`，默认 effort 为 `high`；页面将 adaptive thinking 描述为 Fable 5.1 唯一的 thinking 模式，并通过 `effort` 控制深度。
- 平台字段包含 Claude API、Amazon Bedrock、Google Cloud、Microsoft Foundry 和 Claude Platform on AWS；Bedrock 模型 ID 为 `anthropic.claude-fable-5-1`。
- 页面价格字段为输入每百万 token 10 美元、输出每百万 token 50 美元；缓存写入和读取价格另列，页面说明相比 Fable 5，cache read 为四分之一价格。
- 页面列出 preserved thinking、跨轮模型切换/思考块、per-message effort（beta）、turn-scoped system messages（beta）、工具调用间进度更新（`display: "updates"`，beta）和 content provenance 等能力入口。
- 页面提到 Claude Mythos 5.1 与 Fable 5.1 共享规格和价格，但 Mythos 5.1 通过 Project Glasswing 邀请制提供；本项目不把它当作独立公开可用模型。

## 2026-09-21 当前时点复验

本轮重新获取的 Artificial Analysis Fable 5.1 详情页为 `3,854,152` bytes，SHA-256 为 `bc83faa8117eebdd2ff28660800be4abf7016af7e10511cb4783f2ca12fbc02a`。页面的 canonical slug 为 `claude-fable-5-1`，主标题为 `Claude Fable 5.1 (Adaptive Reasoning, Max Effort, Default Fallback)`，`releaseDate` 为 `2026-09-01`，`deprecated=false`。当前主配置字段为 Intelligence Index `53.3549259623252`、median output speed `68.7301566560472` tokens/s、cost per Intelligence Index task `7.629706364004841`、context `1,000,000`；这些仍是 Artificial Analysis 的配置/provider 测量，不是裸模型能力或通用 API 延迟。

本轮官方资源快照如下，临时文件不作为模型权重或源码归档：

| 资源 | 大小 | SHA-256 |
|---|---:|---|
| Anthropic Fable 5.1 模型页资源 | `14,914` bytes | `f773571dce563d7cb8a501b9ad2eb6531d8164938c5868f1aee8dbda8b462f` |
| Anthropic Fable/Mythos 5.1 发布页资源 | `449,779` bytes | `70d5aaccdd890496070d810944693b9738e8b8c68a97c4e545ddd0e83f9fcd18` |
| Fable/Mythos 5.1 System Card PDF | `16,397,488` bytes | `b0d59edc7a60eef32a879c13d713cce60c3fefd7e6b5183afdc8b835af3c8c39` |

## 2026-09-23 当前时点复验与 System Card 正文证据

本轮重新抓取的 Artificial Analysis Fable 5.1 详情页为 `4,006,915` bytes，SHA-256 为 `d80b226b7a4756cac93d6a065f400f39c9f756b971b827ecf0efbd5a67bffaeb`。canonical slug 仍为 `claude-fable-5-1`，`releaseDate=2026-09-01`、`deprecated=false`；主配置的 Intelligence Index 为 `53.3549259623252`，median output speed 为 `65.4865856934115 tokens/s`，median TTFT 为 `298.446428812s`，context 为 `1,000,000`，cost per Intelligence Index task 为 `7.629706364004841`。这些仍是第三方 provider/configuration 测量，不能写成裸模型能力、固定 API 延迟或跨 provider 排名。

本轮官方快照如下；临时下载和正文抽取文件不作为模型权重或源码归档：

| 资源 | 大小 | SHA-256 |
|---|---:|---|
| Artificial Analysis 中文首页 | `1,783,605` bytes | `7a954aed8916ec9c5c88380cfc159274573dc9547182032ca142f556c0fc2916` |
| Anthropic Fable/Mythos 5.1 发布页 | `534,503` bytes | `610f2e5cf15100fdc85edf0c4bee750878cdbf2db3520606aed088f757bd33f2` |
| Fable/Mythos 5.1 System Card PDF | `16,397,488` bytes | `b0d59edc7a60eef32a879c13d713cce60c3fefd7e6b5183afdc8b835af3c8c39` |
| DataCurve DeepSWE | `268,036` bytes | `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1` |

使用新增的 [`pdf_text_extract.js`](code/pdf_text_extract.js) 对 System Card 做了只读解析，提取正文 `/tmp/fable51-system-card-extracted-20260923.txt`，大小 `376,181` bytes，SHA-256 为 `d61d0a99b770ae104150ceb2ee04e0ee02fc4a728d7521ee0319b3742ed091e5`。解析器只恢复 PDF 文本和 ToUnicode/CMap 映射，不改变原始 PDF，也不把临时文本当作官方独立发布物。

### 正文确认的模型、训练数据与风险边界

- System Card 明确写出 Fable 5.1 与 Mythos 5.1 共享相同模型权重，差异是配置与 safeguards。Fable 面向一般使用，Mythos 对部分高风险双用途生物/网络安全任务使用更宽松的 safeguards，并限制为受信任访问；不能把 Mythos 当成第二个公开 checkpoint。
- 训练数据被描述为互联网公开信息、公共和私有数据、其他模型生成的合成数据的组合；训练过程中使用 deduplication、classification 等清洗/过滤，并通过 ClaudeBot 抓取公共网站，遵守 robots.txt，不访问密码保护、登录或 CAPTCHA 页面。知识截止时间为 2026-06。公开资料仍没有参数规模、层数、dense/MoE 结构、优化器或完整训练/后训练 recipe。
- System Card 的 RSP 结论针对 Mythos 5.1：化学/生物风险保守判为 CB-1，未达到 CB-2；autonomy threat model 1 适用但整体风险评为 low，threat model 2 未达到阈值。Fable 的生产能力不能直接等同于这些更宽松 safeguards 下的 Mythos 评测结果。
- 网络安全章节主要在关闭或改变 safeguards 的 Mythos/helpful-only snapshot 上测量。Fable 5.1 允许源代码漏洞发现，但阻止编译二进制漏洞发现和 exploit 开发；相较 Fable 5 发布时 false positives 减少，但仍比 Opus 5 safeguards 更容易触发。因而 cyber capability、生产拒答和 fallback 必须分开记账。

### 正文能力评测与安全评测的条件账本

System Card 报告的代表性能力结果包括：Terminal-Bench 4.0 `55.8%`、Terminal-Bench-Science 0.1 `52.6%`、CursorBench 3.2.0 `73.4%`、OSWorld 2.0 partial/strict `77.9%/41.7%`、GDPval-AA v2 `1853`、AutomationBench `31.4%` 和 ProgramBench `87.6%`。这些数字必须绑定 Anthropic 的 snapshot、effort、工具、任务版本、fallback、safeguards、试验次数和 verifier；它们不是本项目独立复现，也不是 Fable 5.1 的统一综合分数。

安全章节还报告了 Gray Swan IPI 在 `k=1/10/15` 时 `0.1%/0.7%/1.0%` 的 attack success rate；较强 Shade coding attacker 在无 probes/启用 probes 时分别为 `56.87%/12.80%`，browser auto 为 `0/110`，无 safeguards 的 browser-use 对照为 `2.64%`。这些结果含有不同程度的 fallback，成功攻击主要来自 fallback 的结论也必须保留 `actual_model` 和 `fallback_reason`。System Card 同时披露了 rare permission-hook/classifier bypass、sandbox vulnerability、白盒内部分析、SHADE/LinuxArena covert behavior 与 chain-of-thought controllability；它们适合用于安全与 Agent 监控面试，不应被概括为“模型必然越权”或“生产攻击率”。

由此得到一个可迁移的面试规则：System Card 的能力、safeguard、fallback 和 monitorability 是四个不同维度。评测 manifest 至少应记录 `model_id`、snapshot、effort、tools、permissions、safeguard state、fallback target、task/environment、verifier、score、cost 和实际执行模型；缺失其中关键字段时，结论应降级为发布方条件结果或 `not_comparable`。

DataCurve 当前页面仍没有精确 `mini_swe_agent_claude_fable_5_1_*` 行，只有 Fable 5 的 effort 行。因此继续不记录 Fable 5.1 的 Pass@1、成本或 Agent steps，也不把 Fable 5 的 `316/452` 结果迁移给 Fable 5.1。

## 2026-09-15 重新联网复验

本轮使用 `10.237.126.170:1234` 和 `10.24.27.134:8098` 抓取两个排行榜；两者对目标页面均返回 HTTP 200。`10.24.27.134:7890` 仍可作为备用代理，但对 Artificial Analysis 大页面存在 TLS EOF/读取不稳定。复验快照均为临时文件，不作为仓库长期数据文件：

| 页面 | 临时快照 | 大小 | SHA-256 |
|---|---|---:|---|
| Artificial Analysis Fable 5.1 详情 | `/tmp/recheck-aa-fable51-20260915.html` | 3,615,956 bytes | `aa253dd4d4e8ad3285f60d0910f698239af5bfcefd268bb0127e41a74c93ba4b` |
| Artificial Analysis 中文首页 | `/tmp/recheck-aa-home-20260915.html` | 1,773,775 bytes | `8589909c7199e311a750a2b3c0435e139508da4c6343ae063c935d0c3dbd1a98` |
| DataCurve DeepSWE | `/tmp/recheck-deepswe-20260915.html` | 268,313 bytes | `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7` |
| Anthropic Fable/Mythos 5.1 System Card | `/tmp/recheck-anthropic-fable51-system-card-page-20260915.bin` | 16,397,488 bytes | `b0d59edc7a60eef32a879c13d713cce60c3fefd7e6b5183afdc8b835af3c8c39` |

当时的 Artificial Analysis 详情页确认 canonical slug `claude-fable-5-1`，主配置为 `Claude Fable 5.1 (Adaptive Reasoning, Max Effort, Default Fallback)`，内嵌 `releaseDate` 为 `2026-09-01`。该历史快照的第三方字段为 Intelligence Index `53.3737509623252`、median output speed `65.9769929669683` tokens/s、median time to first chunk `212.0122070875` 秒、1,000,000 context tokens、输入/输出 `$10/$50` 每百万 token，以及约 `$7.6297` 每个 Intelligence Index task。页面还提供 `xhigh/high/medium/low` 与 fallback 变体；这些都是配置级测量，不能写成 Fable 5.1 的裸模型分数或普遍 API 延迟。

DataCurve 当前页面仍是 DeepSWE v1.1：113 个任务、91 个仓库、5 种语言、统一 `mini-swe-agent`，并有 `claude-fable-5` 的 `xhigh/max/high/medium/low` 行；本次页面没有 `claude-fable-5-1` 行。因此 Fable 5.1 不记录 DataCurve 分数，也不把 Fable 5 的 316/452 结果迁移给 Fable 5.1。该“未出现”结论仅针对本次 2026-09-15 页面快照。

Anthropic 当前发布页显示月份为 September 2026，并确认 `claude-fable-5-1` 已可在各平台使用。官方模型页缓存提供的精确日期、1M context、128K output、adaptive always-on 和默认 high 字段与发布页相互吻合。2026-09-15 的历史记录曾遇到区域跳转，本轮已取得模型页资源；两次线路状态分别保留，不把历史访问失败写成模型不可用。

## 发布页新增的公开技术与运行时知识

### 迁移时必须处理的 breaking changes

模型页把以下行为列为 Fable 5.1 的迁移边界：强制工具调用在不兼容的 thinking 请求中返回错误；较早模型不能读取 Fable 5.1 产生的 thinking blocks；编辑较早历史 turn 会使相关 thinking blocks 失效。工程上应在 capability probe 中区分“请求被协议拒绝”“历史状态被标记失效”和“模型推理失败”，并保存原始错误、模型 ID、消息版本、thinking producer/consumer 与回滚路径。

这意味着 thinking block 不是可以任意剪切、编辑、跨版本复制的普通文本。多模型 fallback、历史重写、人工审阅和重试都必须先检查兼容矩阵；若不兼容，应重新生成允许的状态或走明确的无 thinking/非强制工具路径，而不是静默复用旧 block。

### 新增的运行时能力

本轮模型页新增或明确列出五类运行时变化：per-message effort（beta）允许在消息粒度调整投入；turn-scoped system messages（beta）允许只作用于当前 turn 的系统约束；工具调用之间可通过 `display: "updates"` 向用户发送可读进度；cache read 价格降低；content provenance 为生成内容附带来源/溯源协议入口。前四项改变请求、会话或成本账本，最后一项改变结果可信度和审计账本；都应记录 API 版本、schema、默认行为和降级路径。

`display: "updates"` 是用户可见的进度事件，不等于工具已经成功执行，也不替代结构化 tool result、权限检查、超时、幂等和最终 artifact verifier。content provenance 也不自动等于事实正确性，仍需记录来源、引用覆盖、工具回执和独立验证结果。

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

2026-09-21 复核 arXiv 后，精确标题查询 `"Claude Fable 5.1"` 仍返回 0 个结果；全文查询返回 4 篇提到 Fable 5.1 的论文：

- [Long runs of integers with small prime factors and the divisor function of $n!$](https://arxiv.org/abs/2609.15597)：作者说明一个改进思路在与 Fable 5.1 的私下交互中形成，并由作者自行验证和呈现。它是外部使用案例，不是 Fable 5.1 技术报告。
- [The Troy Moment of AI: Why SomeWill Cheat and SomeWill Follow?](https://arxiv.org/abs/2609.15494)：用 Fable 5.1、GPT-5.6 Sol 和 Gemini 3.8 Flash 研究 ImpossibleBench 上的边界遵守、升级/停止和多 Agent 互动；它更适合补充 Agent 安全与 harness 评测知识。
- [Pierce-Birkhoff conjecture is false](https://arxiv.org/abs/2609.10420)：作者称反例发现使用了包含 Fable 5.1 的多模型、多 Agent 链路；论文结论仍由作者负责，不等于模型独立证明。
- [Coloring graphs with no long induced path](https://arxiv.org/abs/2609.08847)：作者称证明在 Fable 5.1 与 GPT Pro 协助下发展；同样属于外部协作案例。

Anthropic Research 页面本次可访问，但未检出 Fable 5.1 专属技术报告条目；官方 [Fable 5.1/Mythos 5.1 System Card](https://www.anthropic.com/claude-fable-5-1-mythos-5-1-system-card) 的 PDF 正文已在 2026-09-23 用本地解析器提取并逐项核对。正文数字仍只作为 Anthropic 发布方、特定 snapshot/harness 和 safeguards 条件下的证据，不升级为独立复现或裸模型结论。

## 页面自述与证据边界

Anthropic 页面声称 Fable 5.1 带来更强的长期 Agent coding、多步研究以及文档、表格和幻灯片工作能力。这些是发布方定位和产品页声明，不是本项目独立 benchmark 复现，也不等于公开了训练配方或内部架构。

`adaptive (always on)`、effort、preserved thinking、thinking block 兼容性和 beta 协议描述的是接口/运行时行为。它们不能证明模型是 MoE、稠密或采用某个特定 test-time compute 算法。跨 Opus/Fable 比较必须固定平台、模型 revision、工具、任务、超时、输出上限、历史编辑策略和 harness。

## 尚待核验

官方资料和 System Card 正文没有披露参数规模、层数、稠密或 MoE 结构、优化器、完整训练/后训练 recipe、Fable 5.1 独立技术报告或本项目独立 benchmark 复现。System Card 的训练数据范围、风险判断和发布方评测已确认，但质量提升、长任务优势、cache 成本以及 Fable 与 Mythos 在具体业务中的差异仍需在固定条件下独立测量。

## 书系映射

- 第四册：Fable 5.1 的接口字段、thinking/effort、平台与 beta 协议。
- 第六册与第二十四册：1M context、KV/cache、长任务成本、TTFT/TPOT。
- 第七册：Opus/Fable 同任务、同平台和同 harness 的公平对照。
- 第十七册与第二十册：preserved thinking、进度更新、工具宿主和状态恢复。

## 当前闭环判断

Fable 5.1 已完成“排行榜发现 + 官方发布页/模型页字段 + System Card 正文 + 运行时 breaking/additive changes + 论文定向检索 + 研究笔记”的 **AA + System Card 正文证据闭环**。没有 DataCurve Fable 5.1 结果，也没有公开参数规模、稠密/MoE 结构、完整训练/后训练 recipe、独立技术报告或独立 benchmark 复现，因此不新增独立 Transformer 架构章节。下一步回到两张排行榜的八家重点厂商候选队列，不把官方文档中关联的其他版本自动升级为新锚点。

## 2026-09-23 `7890` 复验与状态协议 toy

本轮仍只沿两个排行榜已经确认的 `claude-fable-5-1` canonical 条目推进，没有从 Anthropic 文档、System Card 或论文另发现模型。两榜当前复抓结果为：Artificial Analysis 中文首页 `1,783,769` bytes / SHA-256 `e1acf794bd45f380ce3220a20c68de7a274f50ff7eb162a527a9faf4d58a166a`，Fable 详情 `4,008,017` bytes / `4ea24782d05bcaae7d31f3cf348e5a573851678998706a9df82fa34679292f96`，DataCurve `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。AA 当前 `max with fallback` 的 Intelligence Index 为 `53.3549259623252`，median output speed 为 `64.7060238772019 tokens/s`，cost per Intelligence Index task 为 `7.629706364004841`，context 为 `1M`；DataCurve 仍没有精确 `mini_swe_agent_claude_fable_5_1_*` 行。

7890 取得的 Anthropic 官方 Markdown 快照：

| 资源 | 大小 | SHA-256 |
|---|---:|---|
| [Fable 5.1 overview](https://platform.claude.com/docs/en/models/fable-5-1/overview.md) | `14,954` bytes | `13e8aeb6bbd207311ac032916f2ab37e7a454c9d752c027cf76967a5c958a076` |
| [What's new in Fable 5.1](https://platform.claude.com/docs/en/models/fable-5-1/whats-new-fable-5-1.md) | `37,545` bytes | `59d2a26f6e123d009a9e86aac9283705f7bfd15549b185dc07aa1858241baf19` |
| [Migration guide](https://platform.claude.com/docs/en/models/fable-5-1/migration-guide.md) | `94,368` bytes | `c1d7bd16475ce9ad03ad556cc363635d93e695ecafc293e6acb85023e9f869b6` |
| [Release page](https://www.anthropic.com/claude-fable-and-mythos-5-1) | `534,503` bytes | `f0038916a77f0ff73be47f238eeef9b2995fabc9720bb644efd70df42a0eb5f8` |
| [System Card](https://www.anthropic.com/claude-fable-5-1-mythos-5-1-system-card) | `16,397,488` bytes | `b0d59edc7a60eef32a879c13d713cce60c3fefd7e6b5183afdc8b835af3c8c39` |

本轮迁移正文补充了三个值得单独记录的部署门禁：assistant prefill 对 Fable 5.1 返回 400；模型只接受 adaptive always-on thinking，不能用 manual budget 替代；模型需要 30-day data retention，ZDR 组织需获得明确授权，且不支持 Priority Tier。prefix mismatch 可以在严格模式下拒绝，也可以通过 `thinking-binding-controls-2026-08-01` 的 `drop_block` 行为显式丢弃失效块，并从 `input_transformations` 记录原因。thinking block 的兼容性是有方向的：Fable 5.1 能读取早期 Claude 状态，早期模型不能读取 Fable 5.1 状态。

新增 [`code/claude_fable51_state_protocol_audit.py`](code/claude_fable51_state_protocol_audit.py)，仅使用 Python 标准库与合成 trace。它验证 adaptive/forced-tool/prefill capability gate、thinking producer/prefix binding、历史编辑的 error/drop 分支、进度更新与真实 tool result 分账、provenance 独立 verifier、fallback actual model 和相同幂等键的副作用去重。运行结果为 `evidence_level=local_protocol_toy`、首次副作用 1、重复回执 `true`、side-effect executions 1；没有外部网络请求。toy 不能证明 Anthropic 服务端 schema、隐藏 reasoning、模型质量、生产安全或 SLO。

## 2026-09-24：thinking block 的双重绑定与 Opus 5.5 单向兼容

本轮按两个唯一排行榜复核活动锚点：Artificial Analysis 中文首页经 `10.24.27.134:7890` 为 HTTP 200、1,782,611 bytes / SHA-256 `566b4adab724bd312436a302be3f0f4c5d9f7713188e9efb078cb636faddd02b`；Claude Fable 5.1 详情为 HTTP 200、3,979,333 bytes / `1268895cd715723c917adf4cb11e2cdd731919b321588478e64f32311a29c102`。Index `53.3549259623252` 与 cost/task `7.629706364004841` 保持此前观测。DataCurve 经 `10.24.27.134:8098` 为 HTTP 200、268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，与同日既有快照一致，仍无精确 `mini_swe_agent_claude_fable_5_1_*` 行；八家重点厂商没有新增 canonical 模型。

7890 取得 Anthropic 官方资料：Fable 5.1 overview 14,590 bytes / `2a98d9a06c7f3cf710fa11ad838e6b314c07de980044ebc3165a2bbcb47b6ec9`；What's New 37,545 / `59d2a26f6e123d009a9e86aac9283705f7bfd15549b185dc07aa1858241baf19`；migration guide 93,202 / `286fa675d0b48872f401c8bd749f2458374642cc3214366222cfc6960582a082`；通用 Thinking 文档 73,861 / `84dfb480c0895528e56205ab570c872c5769609cc47df2ee92f973a22e59f2a5`。What's New 与 9 月 23 日文件同 hash；overview 的高层 preserved-thinking 卡片不应被解读为所有模型、endpoint 和历史变更都兼容，pair-specific 结论以 Thinking 与 migration 正文为准。

这次补齐的重要兼容边是：Fable 5.1 可读取 Claude Opus 5.5 的 thinking block，但官方明确限定为 Claude API；Opus 5.5 不能反向读取 Fable 5.1 block。与此同时，Fable 5.1 可读早期 Claude 模型生成的 block。兼容方向、服务 surface 和 refusal fallback 是三件不同的事：Fable 文档列出的默认 fallback 仍是 Opus 4.8/Opus 5，不能因 Opus 5.5 单向可读就把它写成默认 fallback target。

应把可恢复状态拆成两条独立校验：

1. **model binding**：生产者与消费者 model pair 是否兼容。不可读 block 会在模型看到请求前被 API 丢弃，不计入 input token；启用 `thinking-binding-controls-2026-08-01` 时可用 `input_transformations` 记录 `model_binding_mismatch`，否则可能静默。
2. **prefix binding**：block 前面的 system、tools 与消息内容是否保持不变。修改旧 turn、重建 system/tools 或删除中间消息会使后续 block 失效。2026-08-31 00:00 UTC 起创建的新账户默认执行检查；更早账户只有显式配置 `thinking.block_binding.prefix_mismatch_behavior` 才按该策略处理。严格模式返回 400；`drop_block` 会丢弃失效 block 并记录 `prefix_binding_mismatch`。

官方允许用 append-only 的 mid-conversation system/tool updates、消息级 effort、server-side context editing/compaction 代替重写旧前缀。从最旧端连续移除 leading thinking blocks 可保留后续块；从中间移除会使之后的块失效。消息级 effort 更新可保持原 cache prefix；不能把这一点泛化成顶层 thinking/effort 配置改变也保持 cache hit。

新增知识已同步至第二十册第 21 章 21.29、面试题、练习、知识图谱和本轮计划/进度。toy 新增 Claude API 专属 Opus 5.5 可读边、反向不可读及其他 surface 不外推的断言；脚本仍只证明合成状态机。当前 Fable 5.1 状态为 **AA 单榜内容专题闭环 + System Card/API contract + local protocol toy**。Opus 5.5 → Fable 5.1 的方向性兼容、Claude API 专属范围、model/prefix binding 见[第二十册第 21 章 21.29](../../book-20-agent-harness-runtime/chapters/21-persistent-workspace与状态边界.md)。参数、内部架构、完整训练配方、精确 DataCurve Agent 行、真实 endpoint probe、目标硬件和生产 SLO 仍未核验。

当前状态为 **AA 单榜内容专题闭环 + System Card/API contract + local protocol toy**，正式面试专题见第二十册第 21 章 21.29。仍没有 Fable 5.1 专属参数/架构、完整训练/后训练 recipe、独立技术报告、精确 DataCurve Agent 结果、完整权重、目标硬件 profile、线上 tool acceptance 或生产 SLO；后续按两榜已有 canonical 集合选择下一项未收口锚点。
