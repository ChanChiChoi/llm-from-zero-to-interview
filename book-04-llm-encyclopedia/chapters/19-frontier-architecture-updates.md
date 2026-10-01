# Frontier Architecture 更新：AttnRes、KDA、CSA/HCA 与 mHC

> 更新时间：2026-09-22。条目依据 Kimi Linear、Attention Residuals、DeepSeek V4、K2 Horizon 官方论文/模型卡；模型具体版本和实验数字需回到来源核对。

## Attention Residuals / AttnRes

**一句话定义**：沿网络深度对历史层表示做 softmax 加权聚合的残差机制。

**核心直觉**：普通残差固定累加历史输出，AttnRes 让当前层选择更值得保留的深度表示；它沿深度轴工作，不是 token self-attention。

**关键形式**：`h_l = Σ_i α_{i→l} v_i`，权重由层专属伪查询和历史表示计算。Block AttnRes 在块内累加、块间选择块表示，以降低存储和通信。

**局限**：保存历史表示和跨 stage 通信有成本；块数、初始化和硬件决定收益。Kimi K3 文章提到 AttnRes，但没有公开 K3 的完整配置。

**相关章节**：第二十一册第 75 章。

## Kimi Delta Attention / KDA

**一句话定义**：带逐通道衰减、标量更新门和 delta correction 的线性注意力状态机制。

**核心直觉**：维护固定大小 key-value 关联状态；先让旧状态衰减，再沿当前 key 方向纠正并写入 value。

**关键形式**：`S_t=(I-β_t k_t k_t^T)Diag(α_t)S_{t-1}+β_t k_t v_t^T`，输出为 `S_t^T q_t`。

**局限**：固定状态会遗忘，远程精确检索可能弱于全注意力，因此 Kimi Linear 使用 KDA 与全局 MLA 混合。论文配置不能直接当作 Kimi K3 全部实现。

**相关章节**：第二十一册第 76 章。

## CSA / HCA

**一句话定义**：DeepSeek V4 的压缩注意力路径：CSA 压缩后做稀疏选择，HCA 更激进压缩后保持 dense attention。

**核心直觉**：远处历史先压缩，重要压缩块再选择；最近 token 用滑动窗口保留细节。

**工程要点**：不同层的压缩倍率、indexer、滑动窗口和未完成尾部导致异构 KV Cache；1M context 不是有效检索和并发能力的同义词。

**相关章节**：第二十一册第 77 章、第二十四册推理框架专题。

## mHC

**一句话定义**：将 Hyper-Connections 的残差映射约束到双随机矩阵集合的稳定化方法。

**核心直觉**：每行和列总量均为 1，限制多路残差流在深层矩阵连乘中的任意放大或衰减。

**关键概念**：Birkhoff polytope、Sinkhorn 交替归一化、流形约束；它约束残差映射，不等同于 AttnRes 的深度历史选择。

**局限**：引入投影、通信、激活和 kernel 开销；有限步 Sinkhorn 只是近似双随机。

**相关章节**：第二十一册第 78 章。

## GPT-6 Astra：接口能力、预算与运行时边界

**一句话定义**：GPT-6 Astra 是 OpenAI 官方模型页列出的 `gpt-6-astra` 模型；本项目当前只把模型页明确公开的接口、容量、工具和计费字段写成事实。

**已核验字段**：模型页列出文本/图像输入、文本输出，1,050,000 context window、922,000 maximum input tokens、128,000 maximum output tokens，以及 `low`、`medium`、`high`、`xhigh`、`max` 五档 reasoning effort。页面还列出 Chat Completions、Responses 和 Batch 支持，并在 Responses 下列出 web search、file search、image generation、code interpreter、hosted shell、apply patch、skills、computer use、MCP 和 tool search 等工具。

**边界解释**：context window、maximum input 和 maximum output 是三个不同约束，不能把它们简单相加后当作并发容量。工具列表表示接口可调用的能力，不表示模型天然拥有宿主机权限；文件、网络、终端和外部系统的授权仍由执行器、沙箱与策略层决定。`image generation` 表示可以调用图像生成工具，也不等于模型自身输出图像 token。

**成本与评测**：本次官方页快照标示输入每百万 token 10 美元、缓存输入 1 美元、缓存写入 12.5 美元、输出 50 美元；输入超过 272K 时，输入与缓存费率乘 2、输出费率乘 1.5。价格是带日期的文档快照，不能替代上线前的实时价格核对。比较不同 effort 时，应同时记录成功率、reasoning token、工具次数、TTFT、TPOT、p95 成本、超时和权限事件。

**新增运行时知识**：GPT-6 Astra 官方指南还公开了应用侧 function/custom tool 的 async calling，以及 Responses WebSocket 的 `response.steer`。前者允许模型在慢工具运行时处理独立部分，但应用仍负责执行、`call_id` 对齐、超时、幂等和结果回灌；后者允许用户在同一响应进行中追加约束，但不会撤销已发送文本、已启动工具或既有副作用。技能描述、`AGENTS.md` 和任务提示应按“短描述、渐进披露、按需读取、明确定义完成条件”治理；这些是 Agent/harness 工程建议，不是模型架构证据。安全监控是平台控制面，告警或阻断也不能替代宿主权限、人工审批和 verifier。

**尚未确认**：官方模型页未披露参数规模、稠密或 MoE 架构、训练数据、优化器、强化学习算法、首次发布日期或完整技术报告。本条目不从模型名、榜单排名或工具名称反推这些内部信息。

**相关章节**：第六册第 18 章 [`gpt-6-astra长上下文与工具预算.md`](../../book-06-llm-deployment/chapters/18-gpt-6-astra长上下文与工具预算.md)。

## GLM-5.3：长任务环境与验证器

**已核验事实**：Z.ai 官方文档称 GLM-5.3 沿用 GLM-5.2 基础模型，改进来自后训练；页面标称 1M 上下文、128K 最大输出，推理始终开启，并提供 `low`、`high`、`max` reasoning effort。文档还描述了可执行环境、judge agent、oracle/no-op/未完成状态检查和奖励捷径检查。

**知识边界**：文档提到 “SAO with compaction”，但未给出 SAO 全称、损失函数或压缩算法。本项目只把它记为待核验名称，不从缩写推断内部机制。长任务训练的通用方法、验证器假阳性/假阴性和协议迁移见第十六册 [`20-glm-5.3长任务环境与验证器.md`](../../book-16-reasoning-models/chapters/20-glm-5.3长任务环境与验证器.md)。

GLM-5.3 官方博客正文资源的增量证据进一步说明，前沿模型的能力记录必须把模型、后训练环境和评测 harness 分开。Z.ai Code Bench 的 50% 提升是发布方私有基准声明；CyberGym、ExploitBench 和 ExploitGym 分别处在漏洞发现/验证、更深利用推理和时间归一化任务完成阶段。269 个项目、2,436 个漏洞和 1,097 个中高危发现属于发布方合作统计，不能改写成模型单独的独立复现。

博客脚注还固定了 mini-swe-agent、Claude Code、容器隔离、Tool Search、域名白名单、turn/timeout、TPS 归一化和官方 verifier 等条件。它们适合进入模型能力的证据链，但不公开 GLM-5.3 独有网络结构或完整训练 recipe；相关长任务、验证器和评测正文见第十六册与第二十册。

## Kimi K3：发布证据与长任务 Harness

Kimi 官方发布文章提到 KDA、AttnRes、Stable LatentMoE、量化感知训练、原生视觉、百万 token 上下文和长任务 Agent 限制。当前应把它们视为发布方披露的研究信号；Kimi Linear 与 Attention Residuals 论文可分别解释 KDA 和 AttnRes 的一般机制，但不能直接证明 K3 的完整层配置、权重状态或每项训练细节。

评测时必须绑定模型 revision、harness、环境硬件、effort、任务版本、工具和 fallback。文章中不同 harness、硬件或比较对象混用时，分数不能拼成同条件排名。K3 对思考历史回传和跨模型切换的提示，应转化为状态 manifest、tool schema、权限、版本和恢复协议的工程检查，而不是简单归结为“上下文更长”。详见第十七册 [`15-kimi-k3发布证据与长任务harness.md`](../../book-17-agent-tool-use/chapters/15-kimi-k3发布证据与长任务harness.md)。

## GLM-5.3-Flash：混合注意力与视觉闭环

**一句话定义**：GLM-5.3-Flash 是 Z.ai 公开的原生多模态 GLM-5 系列模型，采用多数 `linear_attention` 加少数 `deepseek_sparse_attention` 的混合路径，并把视觉观察、渲染、交互和验证放进 coding loop。

**已核验字段**：固定 revision 模型卡/config 给出 320B total、18B activated、45 层、前 3 层 dense MLP、34 个 linear attention layer type、11 个 sparse attention layer type、288 routed experts、top-8、1 shared expert、1M position；IndexPool 相关字段为 4-key pooling、index top-k 2048，mHC 相关字段为 enabled、`hc_mult=4`、20 次 Sinkhorn。视觉配置给出 video/image/text/file 输入与 text 输出，以及 24 层视觉模块和 448 image size 等字段。

**核心直觉**：线性 attention 用固定形状的递归 state 汇总大部分历史，稀疏 attention 用 indexer 找回少量远程显式上下文；IndexPool 先把 4 个 indexer key vectors 加权压成一个候选表示，降低长上下文筛选开销。`index_topk=2048` 是配置字段，不能在没有完整 kernel/tracing 时直接解释成最终读取的 2048 个 token。

**视觉与 Serving**：Z.ai 官方资料把视觉能力描述为“生成—观察—验证—修改”的 native visual coding loop，并公开 self-visual judgment、test-time improvement、SGLang、ReplaySSM、W8A8、INT8/FP8/BF16 hybrid cache quantization、Layer Split 和 EPD（Encode–Prefill–Decode）worker pool。视觉判断不是完备 verifier；浏览器、文件、终端、网络、审批、超时和回滚仍由宿主系统负责。

**评测边界**：Artificial Analysis 的 Intelligence Index/速度/TTFT/价格是第三方 `max` 配置字段；DataCurve 的 284/448、Pass@1 63.392857% 等是 `mini-swe-agent` + 工具 + 环境 + verifier 的组合结果。官方 3.01×/4.44× attention/KV 和约 3× serving 也是发布方自报，不能当作普遍硬件保证。

**相关章节**：第二十一册第 84 章 [`GLM-5.3-Flash：混合注意力、mHC 与视觉闭环`](../../book-21-transformer-architecture-evolution/chapters/84-glm-5.3-flash混合注意力与视觉闭环.md)；研究证据见 [`glm-5.3-flash-source-notes.md`](../../research/model-update-2026-09/glm-5.3-flash-source-notes.md)。

## GLM-5.2：IndexShare 与长轨迹训练

Z.ai 正确博客路由为 [`z.ai/blog/glm-5.2`](https://z.ai/blog/glm-5.2)。新增公开技术线索包括四层共享 DSA indexer 的 IndexShare、用于 speculative decoding 的 MTP 改进、百万上下文的 KV/CPU cache 与调度优化、面向 compaction sub-trace 的 critic-based PPO，以及 coding-agent anti-hack 检测。它们解释了“长上下文可用性”如何同时落在注意力、解码、Serving、RL 和 verifier 上。

这些内容仍是官方博客披露，不能扩写为完整训练配方、生产 kernel 或独立复现；SAO/compaction 原始定义继续待核验。正式专题见第二十一册第 86 章 [`GLM-5.2：IndexShare、MTP 与长轨迹 RL`](../../book-21-transformer-architecture-evolution/chapters/86-glm-5.2-indexshare-mtp与长轨迹rl.md)。

## Mistral Small 4

**一句话定义**：Mistral Small 4 119B A6B 是 Mistral 模型卡公开的稀疏 MoE，使用请求级 `reasoning_effort` 在快速指令和较深推理之间切换。

**已核验字段**：模型卡给出 119B 总参数、每 token 约 6.5B 激活参数、128 个专家且每 token 激活 4 个、256K 上下文、文本/图像输入和文本输出，并标注 Apache 2.0。README 还提供 EAGLE 草稿头与 NVFP4 检查点。

**边界**：总参数决定权重与专家分片压力，激活参数只近似每 token 的主要计算量；EAGLE 接受率、NVFP4 误差和 `none/high` 的质量收益都要按后端、硬件和任务复测。模型名中的 `2603` 不单独证明发布日期，完整训练配方也未由模型卡披露。

**相关章节**：第二十一册第 79 章 [`Mistral Small 4：混合推理与 EAGLE/NVFP4 部署`](../../book-21-transformer-architecture-evolution/chapters/79-mistral-small-4混合推理与eagle量化.md)。

## Step 3.5 Flash

**一句话定义**：Step 3.5 Flash 是 StepFun 的稀疏 MoE 模型，把 MTP-3、滑动窗口/全注意力混合和约 11B 激活参数结合到长上下文服务路径中。

**已核验字段**：官方模型卡给出约 196.81B 总参数、45 层、256K 上下文、288 routed experts 加 1 个 shared expert、top-8 路由和 Apache 2.0；卡片还说明 MTP-3 与 3:1 sliding-window/full-attention 配置，并列出带 Context Manager 的 Agent 评测协议。

**边界**：局部注意力只降低部分层的 pair 数，周期性 full 层仍有长程成本；MTP-3 需要后端支持和草稿接受率，Context Manager 属于 harness 状态策略，不能写成模型永久记忆或固定三倍吞吐。

**相关章节**：第二十一册第 80 章 [`Step 3.5 Flash：MTP-3、滑动窗口和 11B 激活参数`](../../book-21-transformer-architecture-evolution/chapters/80-step-3-5-flash的mtp与滑动窗口moe.md)。

## Grok 4.6：接口证据与架构边界

**已核验字段**：xAI 官方模型页列出 `grok-4.6`、500K prompt/context、文本和图像输入、文本输出、function calling、structured outputs，以及 `low`、`medium`、`high`、`xhigh` 四档 reasoning effort，默认值为 `high`。页面摘要还显示输入/输出价格为每百万 token 2/6 美元，并给出 2026-02-01 知识截止日期。

**系统边界**：reasoning effort 是请求配置，Web Search/X Search 是实时数据接入，function calling 是协议输出；它们都不能证明模型的参数规模、训练方法或内部架构。工具执行、权限和回滚仍由宿主、沙箱和策略层负责。

**尚待核验**：官方模型页未披露参数量、MoE/稠密结构、训练数据、后训练算法、许可证、完整技术报告或独立 benchmark 复现。榜单日期只作为候选发现，不能当作首次发布日期。

## Claude Opus 5：长上下文与 adaptive thinking 的接口证据

**已核验字段**：Anthropic 官方模型目录列出 `claude-opus-5`，页面结构化字段给出 1M context、128K 最大输出和 300K batch 最大输出，并列出 Claude API、Bedrock、Vertex AI 与 Foundry 平台。思考字段为 `adaptive=yes`、`extended=no`，默认 effort 为 `high`；页面还给出输入/输出价格和 2026-05 知识/训练截止字段。

**正确解读**：1M 是接口上下文上限，不等于并发数、KV cache 容量或长任务成功率。adaptive thinking 与 effort 是请求级运行时配置；比较不同档位时应固定任务、模型快照、工具、平台、超时、输出上限和 harness，记录成功率、推理 token、TTFT、TPOT、p95 和单位成功成本。

**状态协议**：Opus 5 的 thinking 响应是异构 content block 序列；`thinking.display: "omitted"` 只改变可见展示，不意味着 thinking block/signature 可以丢弃。工具循环应按原序、原样保存并回放兼容的 opaque 状态，同时绑定模型 ID、revision、消息版本和编辑事件。`thinking disabled` 与 `xhigh/max` 存在能力门槛，客户端应在 capability probe 阶段分类请求契约错误，而不是静默改写后继续评测。

**回退与宿主边界**：`refusal` 是业务响应状态；server-side fallback 可能改变实际模型、工具权限、缓存命中、成本和安全策略，必须在评测账本中记录原始模型、回退目标、拒答类别和最终 artifact。Opus 5 不支持 web fetch 时，搜索/抓取、SSRF 防护、超时、溯源和 verifier 仍由宿主工具负责，不能把工具协议写成模型天然联网。

**宿主边界**：页面列出的平台和工具使用只说明接入路径。网络、文件、代码执行、审批、沙箱、权限、超时、回滚和审计由宿主系统负责，不能从模型目录推断模型天然拥有这些权限。

**尚待核验**：官方目录和关联页面没有在本轮核验中披露参数量、稠密/MoE 架构、训练数据、后训练算法、完整推理机制或独立 benchmark 复现。模型 ID、分类、价格和榜单行不能替代技术报告。

## Claude Fable 5.1：长任务定位与状态协议

**已核验字段**：Anthropic 专属模型页列出 `claude-fable-5-1`，页面发布日期字段为 2026-09-01，1M context、128K 最大输出、文本/图像输入、文本输出、adaptive thinking（always on）和默认 `high` effort。页面列出 Claude API、Bedrock、Google Cloud、Microsoft Foundry 与 Claude Platform on AWS，并给出输入每百万 token 10 美元、输出每百万 token 50 美元的快照字段。

**页面自述**：Fable 5.1 面向 demanding reasoning、long-horizon agentic work、多步研究和文档/表格/幻灯片任务；页面建议一般工作负载优先从 Opus 5 开始，仅在 Opus 5 更高 effort 仍不足时评估 Fable 5.1。这些属于官方产品定位，不能代替固定 harness 的独立评测。

**运行时边界**：preserved thinking、跨轮模型切换、per-message effort、turn-scoped system messages 和工具调用间进度更新等字段描述协议与状态管理能力。`adaptive always-on` 不是公开的内部推理算法，也不能据此推断稠密/MoE 结构；工具执行、权限、压缩、恢复和审计仍由宿主系统负责。

**版本兼容性**：本轮模型页还明确了三类迁移失败：forced tool use 在不兼容的 thinking 路径返回错误；较早模型不能读取 Fable 5.1 产生的 thinking blocks；编辑较早历史 turn 会使相关 thinking blocks 失效。因此 thinking block 应被视为带 producer/consumer、消息版本和编辑状态的 opaque runtime item，而不是可随意剪切的普通文本。多模型 fallback 或人工改写历史前必须做 capability probe，并在不兼容时重新生成状态或走明确降级路径。

**进度与溯源**：`display: "updates"` 是工具调用之间的用户可见进度事件，不是工具执行回执；必须与 tool-call ID、结构化 tool result、权限、幂等、超时和最终 artifact verifier 分开记账。content provenance 是来源/溯源协议入口，也不自动等于事实正确性。Fable 5.1 与 Mythos 5.1 共享 underlying model 但 safeguards 不同，拒答、工具权限和 benchmark 结果必须按服务策略分层，不能写成两个公开架构不同的基础模型。

**System Card 正文补证**：2026-09-23 解析 Anthropic System Card 后，可以进一步确认 Fable 5.1 与 Mythos 5.1 共享相同模型权重；差异主要在 safeguards 和访问计划。公开训练数据描述包括互联网公开信息、公共/私有数据和合成数据，并提到 deduplication、classification、ClaudeBot 以及 2026-06 knowledge cutoff。System Card 的 RSP 结论是 Mythos 5.1 具备 CB-1、未达 CB-2，autonomy threat model 1 适用但整体风险 low，threat model 2 未达阈值。它们是发布方风险判断，不是本项目对 Fable 生产能力的独立结论。

**评测条件账本**：System Card 报告 Terminal-Bench 4.0 `55.8%`、Terminal-Bench-Science 0.1 `52.6%`、CursorBench `73.4%`、OSWorld partial/strict `77.9%/41.7%`、GDPval-AA v2 `1853`、AutomationBench `31.4%` 和 ProgramBench `87.6%`。安全章节还报告 Gray Swan IPI、Shade、browser-use、permission-hook bypass 和 sandbox vulnerability 等结果。每个数字都必须绑定 snapshot、effort、tools、permissions、safeguards、fallback、任务环境、成功定义和 verifier；尤其 cyber 结果不能把 Mythos/helpful-only 或关闭 safeguards 的配置写成 Fable 生产结果。评测 manifest 应额外记录 `actual_model` 与 `fallback_reason`。

**状态协议审计**：本轮用 [`claude_fable51_state_protocol_audit.py`](../../research/model-update-2026-09/code/claude_fable51_state_protocol_audit.py) 做了零依赖教学审计。它把 Fable 5.1 的 capability error、thinking producer/prefix binding、历史修改后的 `drop_block`、`display: "updates"`、tool result、provenance verifier 和 fallback actual model 放在同一条可回放 trace 中；同一幂等键的重放只产生一次副作用。该结果是 `local_protocol_toy`，只验证宿主状态机，不是 Anthropic API、隐藏 reasoning 或模型质量的复现。

**尚待核验**：官方资料仍没有披露参数量、层数、稠密/MoE 结构、优化器、完整训练/后训练配方、完整推理机制、Fable 5.1 独立 benchmark 复现或本项目独立实验。System Card 的训练数据范围和发布方评测条件已确认，但不能越级替代这些缺失证据。Claude Mythos 5.1 虽被页面描述为共享权重的邀请制配置，但本项目不将其当作公开可用独立基础模型。

## DeepSeek V4.1-Flash：CED、CSA2 与重算型缓存

DeepSeek 官方模型卡将 V4.1-Flash 描述为 552B backbone、最多 1M context 的原生多模态 MoE。其语言主干采用 40 层 Causal Encoder-Decoder：20 层 causal encoder 加 20 层 decoder；模型卡给出 prefill 每 token 约 8B、decode 每 token 约 16B 激活参数。这里的激活参数不是 FLOPs，也不是显存大小，输入和输出应分别建立预算。

长上下文侧的主要机制是 CSA2、Hierarchical Sparse Indexer、FP4 main KV 和 SWA Bounded Replay。CSA2 为层分配 `Full`、`Reindex`、`Reuse` 三种静态模式，用于共享 main KV/indexer K 和复用 Top-K 索引；层次化 indexer 让后续层在第一层形成的候选池内继续选择。模型卡还给出 FP4 E2M1 main KV、每 16 个 channel 一个 E4M3 scale，以及 global KV 每 token 890 bytes、约为 V4-Flash 四分之一的口径。890 bytes 不是整个请求 HBM，SWA、indexer、尾部、workspace 和通信 buffer 需要另算。

SWA Bounded Replay 以最近 `n_win` 个 token 重放来重建缺失的滑动窗口状态，把持久化空间换成恢复计算；它要求版本、tokenizer、位置 offset、窗口边界和压缩块状态一致。模型卡另外列出 1 shared + 384 routed experts、每 token 6 个 routed experts、196B Engram conditional memory、Single-Pass mHC 和 DSpark speculative decoding。它们分别属于路由、条件记忆、残差流和草稿验证问题，不能合并为一个“长上下文优化”。

视觉路径使用 DeepSeek-ViT、2D-RoPE、3x3 pixel-unshuffle 和两层 MLP projector；模型卡称预训练包含 45T 多模态 token，在 64K 稀疏 attention 训练后以 34T token 扩展到 1M。后训练公开描述为 `SFT -> RL -> OPD`，较大的变化在 Agent 任务、环境和 rollout 数据管线；未公开的奖励权重、完整 kernel 和训练配方不能从名称推断。

官方模型卡给出的 MMLU-Pro 74.1、HumanEval 79.4、GSM8K 93.0、MMMU-Pro 56.5、DocVQA 95.6，以及 Terminal-Bench 2.1 90.6、DeepSWE v1.1 74.2 等数字都应标为发布方自报。Agent 数字绑定 `reasoning_effort=100`、harness、工具、环境、verifier 和上下文策略，不能直接当作基础模型能力。

模型名、API alias、effort 和工具权限也要分层记录。V4.1 编码说明使用独立 `encoding.py`，DSML 工具标签带前导空格，`reasoning_effort` 支持 1--100（`low=50`、`high=75`、`max=100`）。官方 API 页标明服务名为 `deepseek-flash`，旧 alias 的路由是带日期的服务端行为；协议编码不等于工具执行权限。

**相关章节**：第二十一册第 81 章 [`DeepSeek V4.1-Flash：CED、CSA2、缓存重算与原生多模态`](../../book-21-transformer-architecture-evolution/chapters/81-deepseek-v4.1-flash-causal-encoder-decoder.md)；第六册的 KV/cache 与成本账本；第七册、第十七册和第二十册的 Agent harness 评估。

## DeepSeek V4 Flash Vision：视觉 API 与 alias 路由账本

**锚点边界**：Artificial Analysis 有 `deepseek-v4-flash-vision` 的 `max` 配置，DataCurve 当前快照没有该条目。AA 页面给出 1M context、284/13 目录参数字段、第三方 Intelligence Index/速度/TTFT/价格；这些都是配置级或目录级字段，不能当作视觉变体的独立架构或裸模型能力。

**官方发布**：DeepSeek 2026-08-21 的公告把 `deepseek-v4-flash-vision-exp` 定义为实验性多模态 API，支持混合文本/图像、Chat Completions、Messages、Responses、base64、外部 URL 和 Files API，并描述多模态 Agent workflow。公告的“接近 Opus-4.8”和每图 384 image tokens 是发布方当时的产品表述，不是独立 benchmark 或视觉 encoder 结构。

**当前运行时**：2026-09-15 的 Quick Start/价格页使用 `deepseek-flash`，将其版本标为 DeepSeek-V4.1-Flash，并说明旧 Vision alias 由 V4.1-Flash 服务。应分开保存 requested model、Artificial Analysis catalog anchor 和 response 中的 served model；当前 Vision guide 的图片 detail、resize 和约 1024 tokens/image 规则不能回写成历史实验 alias 的永久属性。

**面试主线**：Files API 的 `file_id` 是带 key/过期/删除语义的资源句柄，不是模型永久记忆；Responses 的 `function_call -> function_call_output(input_image)` 把截图观察接回 Agent 状态，但模型 call 不是执行授权。最终评估要同时检查图片 token、协议事件、工具权限、行为测试、视觉 diff、artifact 和单位成功成本。

**相关章节**：第二十一册第 85 章 [`DeepSeek V4 Flash Vision：多模态 API、工具回灌与路由账本`](../../book-21-transformer-architecture-evolution/chapters/85-deepseek-v4-flash-vision多模态api与路由账本.md)；研究证据见 [`deepseek-v4-flash-vision-source-notes.md`](../../research/model-update-2026-09/deepseek-v4-flash-vision-source-notes.md)。

## Claude Sonnet 5：快速自适应思考的接口字段

**已核验字段**：Anthropic 官方模型目录列出 `claude-sonnet-5`，页面发布日期字段为 2026-06-30，生命周期为 active、退休不早于 2027-06-30，1M context、128K 最大输出、300K batch 最大输出、文本/图像输入、文本输出、`Adaptive` thinking、默认 `high` effort 和 `Fast` comparative latency。页面同时列出 Claude API、Bedrock、Google Cloud、Foundry 与 Claude Platform on AWS；平台 ID 为 Claude API `claude-sonnet-5`、Bedrock `anthropic.claude-sonnet-5`，Vertex AI/Foundry/AWS 为 `claude-sonnet-5`。输入/输出价格快照为每百万 token 2/10 美元，另列缓存写入/读取价格字段。

**页面自述**：模型总览把 Sonnet 5 描述为速度与智能的最佳组合。该句与 `Fast` 属于产品定位和比较字段，不能代替固定 revision、平台、任务集和 harness 下的独立评测。

**正确解读**：Sonnet 5 的 `Adaptive` 和多档 effort 是请求级配置；排行榜中的 low/medium/high/xhigh 行仍属于同一基础模型的评测条件。1M context 也不能直接换算并发或有效检索能力，跨平台对照必须绑定 revision、协议、工具、限流、硬件和 harness。

**尚待核验**：官方模型目录没有披露参数规模、稠密/MoE 架构、训练和后训练配方、完整推理机制或独立 benchmark 复现。页面的速度、价格和能力描述是平台快照，不替代线上复测。

**System Card 深读**：System Card 只把训练数据描述为公开互联网、公开/私有数据和合成数据构成的专有混合数据，并提到去重、分类、ClaudeBot 爬取和 post-training/fine-tuning；没有公开完整训练 recipe。因此面试中可以说“公开了安全与能力评测条件”，不能说“已经公开了 Sonnet 5 的训练算法”。RSP 还给出不跨 automated AI R&D threshold、Autonomy threat model 1 和未跨 CB-2 的分级结论；这些是发布方风险框架，不是通用安全分数。

**评测边界**：发布方报告的 SWE-bench Verified 85.2%、Terminal-Bench 2.1 80.4%、BrowseComp 84.7%、OSWorld-Verified 81.2% 等结果，绑定 adaptive/max、任务集、工具、环境、verifier 和 safeguards。DataCurve 的五档 Pass@1 与 Artificial Analysis 的 Intelligence Index 也各自绑定不同 harness/provider，不能拼成裸模型排名。尤其要检查是否使用了 5 trials、BrowseComp 的 10M token limit 和约 200K compaction 触发条件。

**安全与 Agent 运行时**：Claude Code 恶意请求拒答率 92.37%、computer-use 恶意任务拒答率 84.68%，Gray Swan IPI 覆盖 28 个场景、去重后 1,130 个攻击。这些结果说明 safeguards、工具和环境是测量对象的一部分；模型输出动作仍只是提案，权限、allowlist、沙箱、人工确认、幂等和 artifact verifier 仍属于宿主责任。

## Claude Haiku 4.5：低成本与 fastest 延迟字段

Anthropic 官方模型目录快照列出 `claude-haiku-4-5-20251001` 及别名 `claude-haiku-4-5`。页面结构化字段给出 2025-10-15 发布日期、200K context、64K 最大输出、文本/图像输入、文本输出、视觉和工具使用，并将 comparative latency 标为 `fastest`。thinking 字段为 extended thinking；目录没有给出 adaptive 或默认 effort 字段。

目录还列出 Claude API、Bedrock、Vertex AI、Foundry 和 AWS 平台，以及输入/输出每百万 token 1/5 美元的价格快照；可靠知识截止为 2025-02，训练数据截止为 2025-07。价格、平台和退休日期都需要按目标账户及实时文档复核。

`fastest` 是产品目录的比较字段，不能替代固定输入长度、输出预算、平台、硬件和并发下的 TTFT/TPOT 实测。200K context 与 64K output 也不能直接推出 KV cache 容量或长任务能力。参数规模、稠密/MoE 结构、训练配方、完整推理机制、system card 细节和独立 benchmark 仍待核验。相关页面仅作为发布和接口证据，不能从模型 ID 或延迟标签反推架构。

## DeepSeek-R1-0528：发布页与协议兼容性证据

DeepSeek API 官方发布页（页面导航标注 2025/05/28）将 `DeepSeek-R1-0528` 描述为一次更新，并列出 benchmark、前端能力和幻觉控制方向的改进。页面明确写出支持 JSON output 与 function calling，并声称 API 使用方式没有变化；这属于发布方协议兼容性承诺，不等于所有 SDK、账户、限流、流式和错误码行为永久不变。

页面同时提供开源权重入口和 benchmark 图片。权重链接证明存在公开交付入口，但不能单独证明当前 revision、许可证、参数量、架构或训练配方；图片中的数字也需要读取原始图表、任务集、采样条件、硬件和统计区间后才能进入正式结论。JSON/function calling 仍是模型输出协议，工具执行、权限、网络、沙箱、审批、超时、回滚和审计由宿主系统负责。

## K2 Horizon MoVA 36B/A4B：把稀疏路由推进到 value 路径

K2 Horizon MoVA 36B/A4B 是由 Artificial Analysis 榜单发现、再用 IFM 官方模型卡和固定 revision 核验的候选。配置与实现公开的结构是约 36B 总参数、约 4B/token active proxy、48 层；前 3 层是 dense attention + dense MLP，后 45 层同时使用 MoVA value routing 和稀疏 MoE FFN。后者有 100 个 routed experts、每 token top-8 和 1 个 shared expert；MoVA 有 64 个 value experts、每 token top-4。这里的 4B 是发布方的 active-parameter 口径，不是单卡显存、严格 FLOPs 或端到端速度保证。

MoVA 的关键变化不是复制 64 个完整 attention，而是让 value projection 由当前 token 选择专家：GQA 仍有 32 个 query heads 和 8 个 KV heads，value experts 输出 8 个 KV heads 乘 128 的表示，混合后再进入 `softmax(QK^T/sqrt(d))V`。实现中的 sigmoid score、只用于 top-k 的 router bias、归一化后乘 2.5，以及独立的 softplus attention gate 都是应固定的 checkpoint 语义。它们不能被简化成一个 top-12 FFN MoE 或普通 softmax router。

模型配置给出 524,288 context、BF16、`sliding_window` disabled；模型卡还披露从 8K 到 512K 的分阶段 midtraining/SFT。长上下文的服务账本必须同时列出 36B 权重、两套路由的 dispatch、GQA KV cache、workspace、padding、负载倾斜和通信，不能只拿 4B active 做显存估算。部署参考中的 TP=2、expert parallel 和 SGLang 的 EP/FlashAttention-3 参数属于后端配方，迁移到其他硬件仍需复测 TTFT、TPOT、峰值显存和输出一致性。

同系列 0.9B 模型卡披露的专家合并与 MOPD 只作为另一条训练流程记录，不能反推 36B 使用同一 recipe。`K2-Horizon-7B-Uno` 则是 K2 7B 的官方关联 adapter：冻结 AR base，使用 diffusion/LoRA path 生成 draft，再以 `Psi-Spec` 做 AR rejection verification。它是锚点周边的推测解码技术，不是排行榜新增模型，也不是 36B MoVA 架构变体。完整证据、版本和待核验项见 [`k2-horizon-source-notes.md`](../../research/model-update-2026-09/k2-horizon-source-notes.md)；专题章节见第二十一册第 82 章。

## Qwen3.8：QSA、Gated Residual 与 N-gram 容量轴

Qwen3.8 是本轮由 Artificial Analysis 发现并由 Qwen 官方资料核验的系列。榜单中有 Qwen3.8 Max、2.4T-A95B、27B 和 Flash-Next；DataCurve DeepSWE 当前快照只检出 Max。这里的 Max 是官方说明基于 A95B 的 hosted version，不能和 A95B 作为两个独立 open checkpoint 处理。

### 1. 模型边界

| 版本 | 公开规格 | 接口/模态边界 |
|---|---|---|
| Qwen3.8-27B | 27B dense，64 层；每 4 层 3 个 GDN + 1 个 Gated Attention | 原生 262K，可扩展约 1M；图像/视频；thinking 可关闭 |
| Qwen3.8-2.4T-A95B | 2.4T total、95B active，92 层；512 experts，10 routed + 1 shared | 原生 262K，可扩展约 1.01M；text-only；thinking 强制开启 |
| Qwen3.8-Flash-Next | 125B/6B active，约 51B N-gram、4B MTP；48 层；GDN + QSA | 原生 262K，可扩展约 1M；视觉；实验性架构预览 |
| Qwen3.8-Max | 基于 A95B 的托管版本 | Qwen Cloud 产品页增加视觉、非 thinking、默认 1M 和内置工具等能力 |

### 2. QSA 的检索预算

Flash-Next 用 GDN 处理固定状态记忆，用 Qwen Sparse Attention 在 micro-block 级别选择少量显式上下文。indexer 使用 4 个 query heads 和 1 个 shared key head；配置给出 block compression ratio `r=4`、token budget `K=2048`，所以最多选择 512 个完整 block，再保留当前未完成尾部 token。它不是简单的 top-k token，也不是把完整 attention 变成严格线性：indexer 仍有约 `O(n^2/r)` 的报告口径，核心稀疏 attention 在固定 K 下才可作 `O(nK)` 的教学近似。

QSA 的实现关键是先在 RoPE 之前平均池化 key，再给 block 起始位置做 partial RoPE；block-causal scoring 只允许 query 使用完整可见 block。训练分为 full-attention teacher 的 dense distillation 和启用稀疏核心后的 joint sparse training。官方报告自报 full/QSA 平均分 75.9/76.8、1M kernel prefill/decode 约 7.6x/4.9x；这些结果绑定 chunked prefill、batch、MTP 和 FlashInfer baseline。

### 3. GR、N-gram 和 Muon

Gated Residual 把 residual stream 扩为 4 个 branch：每个 branch 单独 RMSNorm，所有 branch 共同预测逐元素 read gate，每个 branch 用 scalar write gate，并删除 `Hres` branch mixing。它和 AttnRes、mHC 都讨论深层信息流，但 GR 把表达力集中在 channel-level read，mHC 的重点则是双随机 branch mixing。

N-gram Embedding 在 Layer 2 用局部 bigram/trigram 查表，公开配置列出 20M slots、约 51B 参数。它不是 RAG，也不是 KV cache；确定性地址使 table 可以 host-memory offload 并异步 prefetch，但成本会转向随机读、带宽和多租户容量。报告同时提醒，词表扩大可以让 loss 下降，不能保证所有下游 benchmark 同步提升。

Muon 主要处理 attention/GDN/MoE expert/N-gram projection 等二维线性映射；embedding、LM head、router 和 GR 低秩 projection 使用 AdamW，N-gram table 使用不带 weight decay 的 Adam。fused QKV、GDN input 和 SwiGLU FC1 要先按语义拆分，再独立做 Newton-Schulz；Canzona 和 CUDA graph 解决分布式重构、负载均衡与小 kernel launch 开销。优化器选择因此会改变训练并行和 kernel，而不只是改变一个超参数。

### 4. 关联阅读与边界

完整规格、来源哈希、模型家族边界和待核验项见 [`qwen3.8-source-notes.md`](../../research/model-update-2026-09/qwen3.8-source-notes.md)；架构推导、零依赖 demo 和面试追问见第二十一册第 83 章 [`Qwen3.8：QSA、Gated Residual、N-gram Embedding 与 Muon`](../../book-21-transformer-architecture-evolution/chapters/83-qwen3.8-qsa-gated-residual-n-gram-muon.md)。完整生产 kernel、线上 MTP acceptance rate、目标硬件 profiling、全系列训练/后训练配方和独立 benchmark 仍待核验。

## GPT-OSS：开放权重 MoE 的结构、协议与部署边界

OpenAI 的 `gpt-oss-120b` 与 `gpt-oss-20b` 是本轮从 Artificial Analysis 发现的同一开放权重模型族。官方 Model Card 披露两者分别为 116.8B/5.13B active 和 20.9B/3.61B active，采用 128/32 experts、每 token top-4 路由；active 参数只表示当前 token 的主要计算路径，不能代替总权重、KV cache、dispatch、通信和并发 workspace 的 serving 账本。

架构主线是交替 sliding-window/full attention、GQA、RoPE/YaRN 和 post-training MXFP4。局部层控制大多数 token 的访问范围，dense 层周期性完成远程信息交换；因此 130K context 不是每层都对 130K token 做 dense attention。MXFP4 又把量化、checkpoint 大小、MoE kernel 和单 GPU 部署目标联结起来，不能只把它当作上线后可随意替换的压缩格式。

模型还以 Harmony 作为训练/推理响应协议，规定角色层级、channel、recipient、工具调用和 structured outputs；`low/medium/high` 是同一权重的 variable-effort reasoning 配置。模型产生 tool call 不等于工具已经执行，宿主仍要做 schema、权限、沙箱、超时、结果回灌和 artifact 验证。参数、完整训练 recipe、router 负载均衡、MXFP4 kernel、目标硬件 profiling 和线上接受率仍待核验。研究证据见 [`gpt-oss-source-notes.md`](../../research/model-update-2026-09/gpt-oss-source-notes.md)，正式专题见第二十一册第 87 章。

## Gemini 3.5 Flash-Lite：低延迟推理与按需视频证据

Gemini 3.5 Flash-Lite 是本轮从 Artificial Analysis 发现的 Google 重点锚点。Google API 页面公开了 `gemini-3.5-flash-lite`、文本/图像/视频/音频/PDF 输入、文本输出、1M 输入和 65K 输出上限，以及 caching、code execution、File Search、function calling、Search/Maps grounding、structured output、thinking 和 Computer Use Preview。它适合高吞吐 subagent、文档解析和分类任务，但这些接口字段不等于内部 Transformer 结构。

它的一个可迁移知识点是“模型能力”和“读取策略”要分开：Video understanding 文档把 Lite 列入 agentic video。static 路径固定采样视频帧；agentic 路径根据 prompt 动态浏览时间轴，按需加载 transcript、帧或音频，并以 `processing_call`/`processing_result` 形成可审计事件。系统评估应同时记录证据召回、处理调用次数、总 token、TTFT、端到端成功率和单位成功成本，不能只报告一个质量百分比。

另一个边界是版本继承。DeepMind Model Card 明确 3.5 Flash-Lite based on Gemini 3.1 Flash-Lite，并将 architecture、training data、hardware 和 software 指向 3.1 Model Card。因此本百科只记录公开的 API、thinking 和视频处理契约，不把前代卡片内容包装成 Lite 独有发明，也不把 Gemini 3.5 Flash 页面的默认 `medium` 或 GA 叙述迁移给 Lite。完整证据见 [`gemini-3.5-flash-lite-source-notes.md`](../../research/model-update-2026-09/gemini-3.5-flash-lite-source-notes.md)。

## Grok 4.7：服务字段不能替代架构证据

Grok 4.7 是 Artificial Analysis 已发现的 `grok-4-7` canonical 条目，当前展示为 `xhigh`；DataCurve 没有精确的 `mini_swe_agent_grok_4_7_*` 行。AA 的指数、速度和成本属于第三方配置/provider 测量，不能与 xAI 发布方的 DeepSWE、CursorBench 或本地 Agent 结果合并成一个裸模型分数。

xAI 公开的是服务和训练方向：更大的 base model、更长的 RL run、困难长任务混合、自验证、长上下文管理和新的 safeguard stack。模型页公开的 500K context、`low/medium/high/xhigh`、价格、限流和 `algorithm` 配置字段回答的是 API 合同问题，不回答参数规模、MoE/稠密结构、层数或 optimizer。arXiv 精确标题检索没有 Grok 4.7 专属论文，因此不能用产品字段补齐内部架构。

读前沿模型资料时，应固定这条证据链：

```text
leaderboard identity -> provider/API contract -> official training disclosure
-> implementation artifact -> reproducible hardware/benchmark evidence
```

只有最后两层都出现，才可以讨论具体 kernel、参数账本或端到端性能。完整快照与未知项见 [`grok-4.7-source-notes.md`](../../research/model-update-2026-09/grok-4.7-source-notes.md)。

## K2 Horizon 3.7B：用 dense 对照读懂 MoVA 的额外成本

K2 Horizon 3.7B 是 Artificial Analysis 已发现、但 DataCurve 没有精确 Agent 行的 AA 单榜锚点。当前 IFM revision 使用 `K2HorizonForCausalLM`，36 层、hidden 2560、32Q/8KV GQA、head dim 128、524,288 position、default RoPE，`num_experts=0`、`mova_num_experts=0`，因此所有层都是 dense。它与 K2 Horizon MoVA 36B/A4B 共享 GQA 和 512K 目标，却不是 36B 的简单缩小版。

36B/A4B 在后 45 层增加 64 value experts/top-4 的 MoVA、100 routed FFN experts/top-8 和 1 个 shared expert；3.7B 则把变量减少到 dense GEMM、GQA KV、长 prefill 和 parser。这个对照能把“active parameter”之外的系统成本讲清楚：MoVA/FFN 还要支付 router、token dispatch、EP 通信、负载倾斜、workspace 和回收；3.7B 的 dense 速度不能按参数比例外推 36B serving。

3.7B 的模型卡还公开 22.9T/8K pretraining、32K/128K/512K 分阶段 midtraining、512K SFT Phase 1/2、Math/Code/STEM-Code RL 分支以及 self-attention ISO merge/其他权重 RAM。中间 checkpoint 是阶段性研究入口，但不等于完整训练 recipe。migration manifest 的 `k2_aurora -> k2_horizon`、copy、BF16、36 shards/327 tensors 是 artifact 迁移证据；vLLM/SGLang 的 H200 结果属于发布方 recipe。当前旧 APPENDIX 的 `XllmForCausalLM`/FP32 与新 config 的 `K2HorizonForCausalLM`/BF16 冲突，应按 revision 处理。完整底稿见 [`k2-horizon-3.7b-source-notes.md`](../../research/model-update-2026-09/k2-horizon-3.7b-source-notes.md)，正式对照见第二十一册第 82 章。

## Kimi K2.7 Code：长周期编码 Agent 的模型与服务边界

Kimi K2.7 Code 是 Artificial Analysis 与 DataCurve DeepSWE 中已出现的 coding-agent 锚点。固定 revision 模型卡公开约 1T total / 32B active 的 MoE、384 routed experts/top-8/1 shared、MLA、MoonViT 和 native INT4；这些字段可以支撑架构与部署预算讨论，但不能推出完整训练 recipe、真实驻留显存或生产吞吐。

官方 API 快速开始把服务合同讲得更具体：256K context 与默认 max_tokens 32K 分开计；thinking 必须启用，若干 sampling 字段固定；多模态视频输入由宿主工具检查/裁剪后回灌，reasoning_content 可用于多步状态连续性。文档称 Highspeed 与普通 ID 是同一模型，故仅作为服务路由变量，不另列模型。Kimi 发布方 benchmark、AA provider 指标、DataCurve mini-swe-agent 结果和本地 toy 必须分栏。

深入解释见[第二十一册第 94 章](../../book-21-transformer-architecture-evolution/chapters/94-kimi-k2.7-code-long-horizon-coding-agent.md)和[研究笔记](../../research/model-update-2026-09/kimi-k2.7-code-source-notes.md)。独立技术报告、完整训练配方、权重加载、目标硬件 profile 和线上 Agent 验收仍未确认。

## Qwen3-Omni：理解和语音生成的双节奏

Qwen3-Omni 是 Artificial Analysis 已发现的 AA 单榜锚点；DataCurve 当前没有精确 `mini_swe_agent_qwen3_omni_*` 行。官方技术报告把系统拆成 Thinker 与 Talker：Thinker 处理多模态理解、推理和工具/策略接口，Talker 利用音频/视觉特征生成流式多码本语音，再由 Code2Wav 生成 waveform。

它的面试主线是三本账：AuT 以约 12.5 Hz 形成约 80 ms 音频时间粒度，TM-RoPE 将 temporal/height/width 与真实音频/视频 timestamp 对齐，Talker 以首码本 AR + residual-codebook MTP 降低码本串行深度。报告的 `234/547 ms` 首包数字属于发布方口径，不能替代目标硬件和并发下的 p99。

模型边界也要保留：Instruct、Thinking、Captioner 是同家族 artifact；仓库可运行、Thinker serving、Instruct 音频输出和生产 streaming acceptance 是不同 gate。完整资料见 [`qwen3-omni-source-notes.md`](../../research/model-update-2026-09/qwen3-omni-source-notes.md)，专题见第二十一册第 91 章。

## Qwen3-VL：三轴位置与视觉层级

Qwen3-VL-235B-A22B 是 Artificial Analysis 已发现的 AA 单榜锚点；instruct/reasoning 页面归并为同一基础模型的配置，DataCurve 当前没有精确 `mini_swe_agent_qwen3_vl_*` 行。公开 config 和技术报告给出 SigLIP2 vision encoder、两层 MLP merger、Qwen3 MoE decoder、Interleaved-MRoPE、DeepStack `[8,16,24]` 与 Video Timestamp。

Interleaved-MRoPE 在 temporal/height/width 三轴之间交错分配低频和高频 rotary dimensions，目标是缓解长视频中的频谱偏置；它不是 1M context 证明。DeepStack 把多个视觉中间层经专用 merger 后残差注入语言模型早期层，不增加额外视觉 token 序列长度，但会增加投影和激活计算。Video Timestamp 把 seconds/HMS 形式的时间锚点写进视频 patch 上下文，仍需宿主保存采样、frame、chunk 和 replay 状态。

完整证据见 [`qwen3-vl-source-notes.md`](../../research/model-update-2026-09/qwen3-vl-source-notes.md)，正式专题见第二十一册第 92 章。AA 字段、论文训练阶段、上游 raw main 和本机多模态/GUI serving 实测必须分账。
