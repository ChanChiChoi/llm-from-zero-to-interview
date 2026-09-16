# Grok 4.6：榜单、训练发布与 Agent API 资料摘记

核验日期：2026-09-15。本笔记只研究已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 Grok 4.6；xAI 官方发布页、模型页和 API 文档用于核验候选事实与扩展面试知识，不作为新的模型发现入口。榜单分数保留配置、任务集和 harness 语境，不改写为裸模型能力。

## 榜单锚点

### Artificial Analysis

[Grok 4.6 详情页](https://artificialanalysis.ai/models/grok-4-6) 的 canonical 配置为 `Grok 4.6 (high)`，页面记录的第三方 `releaseDate` 为 2026-08-12。当前快照公开了以下字段：

- Artificial Analysis Intelligence Index：`44.4050073012592`；页面显示约 44.41。
- 输出速度中位数：`58.5035284934629` tokens/s；TTFT 中位数：`40.8595908855` s。
- context window：500,000 tokens；输入/输出价格：每百万 token `$2/$6`。
- `isReasoning: true`、`isOpenWeights: false`、`parameters: null`、`openSourceCategorization: proprietary`。
- 详情页同时列出 `low`、`medium`、`high`、`xhigh` 配置链接；这些是同一基础模型的推理配置，不计为四个模型。

本次直接详情页快照为 3,522,180 bytes，SHA-256 为 `8d6c96aa27f6db87537f3f1802ca07122d58385ee4b6d5e1a5ab84897f360383`；`1234` 与 `8098` 两个代理抓到的文件字节完全一致。其中的 `releaseDate`、参数空值、开放性和指数都是第三方目录字段；官方发布日期以 xAI 发布公告为准，参数/架构不能由页面字段补齐。

### DataCurve DeepSWE v1.1

[DataCurve DeepSWE](https://deepswe.datacurve.ai/) 页面标注更新时间为 2026-09-03，数据对象的 `generated_at` 为 `2026-09-03T22:24:37.984682+00:00`，任务集为 113 个任务、91 个仓库、5 种语言，统一 harness 为 `mini-swe-agent`。嵌入数据中 Grok 4.6 的四个配置为：

| effort | 配置 | Pass@1 | Pass@4 | 通过/尝试 | 平均成本 | 平均输出 token | 平均 Agent steps |
|---|---|---:|---:|---:|---:|---:|---:|
| low | `mini_swe_agent_grok_4_6_low` | 41.648% | 69.027% | 187/449 | `$1.0424` | 16,458 | 44.19 |
| medium | `mini_swe_agent_grok_4_6_medium` | 67.478% | 84.071% | 305/452 | `$3.4490` | 49,764 | 70.29 |
| high | `mini_swe_agent_grok_4_6_high` | 65.188% | 84.956% | 294/451 | `$4.3849` | 61,161 | 78.96 |
| xhigh | `mini_swe_agent_grok_4_6_xhigh` | 66.741% | 84.956% | 301/451 | `$5.4977` | 71,404 | 87.22 |

DataCurve 快照为 268,313 bytes，SHA-256 为 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。Pass@1/4、成本、输出 token 和 steps 绑定模型配置、4 次整套运行、`mini-swe-agent`、工具、任务集、环境和 verifier，不能与 Artificial Analysis 指数拼接成一个裸模型排名。页面主表的展示行与嵌入数据的多 effort 行也应保留原始配置字符串。

## xAI 官方发布资料

- [Introducing Grok 4.6](https://x.ai/news/grok-4-6) 的 JSON-LD `datePublished`/`dateModified` 均为 2026-08-12。发布页把模型定位为长运行 Agent、交互式项目和视觉工作，并称它相较 Grok 4.5 更能跨多步维持复杂任务。
- xAI 发布页的对比图给出其当时的四舍五入数值：AA Intelligence `61`、GDPVal-AA `1753`、DeepSWE 1.1 `65.9`、CursorBench 3.2 `69.9`、FrontierCode 1.1 `61.3`。这些是 xAI 发布页的展示口径，竞品数字来自各自公开 system card 或榜单；不能与当前 Artificial Analysis 详情页的 44.405 或 2026-09-03 DataCurve 原始行直接混成同一批测量。
- xAI 明确描述 Grok 4.6 进行了比 Grok 4.5 更长的 supplemental training run，使用经过筛选的模型生成数据（reasoning 与 advanced technical concepts）、高质量工程数据，以及改进的 optimizer 和 training recipe；xAI 只公开到这一层，没有给出优化器名称、超参数、数据规模或完整 recipe。
- xAI 还称使用 Grok 4.5 重新生成跨 reasoning effort、Agent harness、STEM、软件工程和知识工作的 SFT trajectories，并用 model-based checks 过滤问题轨迹。该描述支持“教师模型/轨迹筛选/SFT 质量控制”的面试讨论，但不等于公开了 Grok 4.6 的完整蒸馏算法。
- 后续 RL 覆盖 knowledge work、general coding，以及 kernel optimization、web development、computer-aided design 等领域环境。xAI 在长轨迹项目中观察到更多 self-testing and verification，并描述了从宽泛产品想法到可运行交互项目、再通过反馈迭代的工作流。
- 安全部分称 safeguards 与能力校准同步改进，使用更大范围的 pre-deployment capability/safeguard testing，以及 post-deployment 和 third-party testing。这是发布方的安全评估描述，不是公开的安全分类器结构或数值结果。
- 发布页还称 Grok 4.6 可通过 xAI API、Grok Build、Cursor 及 OpenRouter/Vercel/Cloudflare 等渠道使用；价格起点为每百万输入 token `$2`、输出 `$6`，另有价格为两倍的 fast variant。

上述内容是 xAI 的发布方描述，不应升级为参数量、MoE/稠密结构、层数、训练数据总量、RL 算法或独立复现事实。

## 官方模型页与 API 合同

主要来源为 [Grok 4.6 官方模型页](https://docs.x.ai/developers/grok-4-6) 与 [Markdown 版本](https://docs.x.ai/developers/grok-4-6.md)。页面元数据为 `datePublished/dateModified: 2026-09-02`，正文标注 Last updated: September 2, 2026。本次 HTML 快照为 408,905 bytes，SHA-256 为 `9669e2c5e96dafb74296f6e11af7c3b0fc74e8dec58a3d674edf18154c1caf27`；Markdown 快照为 1,890 bytes，SHA-256 为 `402b25df58dfd36b27a20ca4238354d38f47419026a4c43ecd39148354307567`。

官方模型页确认：

- model name 为 `grok-4.6`；context window 为 500,000 tokens；knowledge cutoff 为 2026-02-01。
- text/image input、text output；页面写明 no text output limit。
- input/output price 为 `$2/$6` 每百万 token。
- reasoning effort 支持 `low`、`medium`、`high`、`xhigh`，默认 `high`。
- APIs 为 Responses API 和 Chat Completions；工具包括 function calling、Web Search、X Search 和 Code Execution。结构化配置另列 structured outputs 与 reasoning。
- API 示例同时覆盖 xAI SDK、Vercel AI SDK、OpenAI-compatible SDK 和 `curl`，说明模型可作为 OpenAI-compatible Responses 端点接入不同 Agent harness。

官方页面还建议 Responses API 使用 `prompt_cache_key`，Chat Completions 使用 `x-grok-conv-id`，以便把同一会话路由到相同服务器、提高缓存命中；长 Agent loop 结合 [Context Compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction) 管理上下文。页面的服务端配置对象另外显示 200K 的 long-context threshold、150 RPS、50M TPM 和 500K 最大 prompt；不同区域/账户的实时价格和限流仍应以当前控制台为准。没有看到足够证据把 Grok 4.6 写成 Batch 不支持，因此本笔记不下这个结论。

## 可迁移的面试技术点

### 1. 数据生成、轨迹筛选与 agentic RL

一个可复述的训练链是：更长的补充训练 → 模型生成 reasoning/工程数据与质量筛选 → Grok 4.5 生成多 effort、多 harness、多领域 SFT trajectories → model-based checks 过滤问题轨迹 → SFT checkpoint → 面向知识工作、编码和领域环境的 agentic RL。面试时要明确哪些是 xAI 发布方披露，哪些仍未公开：没有足够证据说明具体 optimizer、RL objective、reward model、trajectory acceptance rate 或异步调度实现。

### 2. 长任务中的自测与验证

xAI 对 Grok 4.6 的公开描述从“会调用工具”推进到“长轨迹中自测试和验证”。工程上仍需把模型自称的完成拆成测试、lint、类型检查、运行结果和领域 verifier；模型产生的 verification 不能自动等价于可信验证器。该主线可映射到第七册评测、第十七册 Agent 工具循环和第二十册 harness/runtime。

### 3. reasoning state、缓存与 compaction

沿用 xAI reasoning 文档：reasoning model 的 reasoning 不能关闭，`reasoning_tokens` 可用于统计；`reasoning.encrypted_content` 是客户端不能解析的 opaque state，需要原样回传。Responses API 的状态化响应默认在服务端保存 30 天；更长任务需要应用自行保存历史与协议状态。`/v1/responses/compact` 返回单个 opaque compaction item，客户端不能解析、裁剪、重排或手工拼接。面试中应区分模型状态、服务端 response 状态、prompt cache 路由和应用持久化四个层次。

### 4. 工具责任与最小权限

server-side Web/X Search 与 Code Execution 由 xAI 服务端执行；custom function calling 只是模型提出调用，数据库、文件、网络和副作用仍由宿主执行器控制。通用 function calling 文档还提供 `auto`/`required`/`none`/指定函数、默认 parallel function calling 和最多 350 个工具定义等协议字段。Remote MCP 的 `allowed_tools` 应采用最小权限；工具定义本身也会占用 context 和影响成本。

### 5. effort 不是四个基础模型

AA 的四条 effort 链接和 DataCurve 的四条配置行应归并为 Grok 4.6 一个基础模型。对比 low/medium/high/xhigh 时，至少固定模型 revision、harness、工具集、任务集、超时、重试、上下文压缩和成本口径；这次 DataCurve 的 medium Pass@1 高于 high/xhigh 也不能解释成“medium 模型更强”，它只是该评测配置的一次观察。

## 论文、技术报告与代码检索边界

- [arXiv 精确标题检索](https://arxiv.org/search/?query=%22Grok+4.6%22&searchtype=title) 于本轮返回 `title: "Grok 4.6" produced no results`；快照 SHA-256 为 `2aa9e76dfac307b35c58488f760af873adc01e168aaaaadb9dd96201a22cc90b`。
- xAI 发布页和官方模型页没有链接 Grok 4.6 专属技术报告或公开权重仓库；本轮公开入口未检出专属论文/报告。这个结论是负面检索证据，不是对未来发布的绝对否定。
- 论文中把 Grok 4.6 当作黑盒被测模型的结果不能反推 xAI 的架构、训练配方或 reward 设计；因此不新增 Grok 4.6 专属架构章节。

## 书系映射与闭环状态

1. 第四册：模型接口、500K context、视觉输入、reasoning effort 与实时工具边界。
2. 第五册：模型生成数据、SFT trajectory 筛选、model-based checks、agentic RL 的公开证据边界。
3. 第六册与第二十四册：长上下文 KV/cache、prompt cache 路由、TTFT/TPOT、compaction 与单位成功成本。
4. 第七册：固定 harness 的 effort sweep、Pass@1/4、置信区间、成本、输出 token 和 steps。
5. 第十七册与第二十册：function calling、built-in tools、MCP、权限、重试、幂等、验证器和长任务状态账本。

当前状态：资料级闭环。两个排行榜的锚点、xAI 官方发布页、官方模型/API 文档、研究笔记和论文负检索均已具备；没有独立正式章节。参数规模、内部架构、完整训练/后训练 recipe、线上接受率、生产 kernel、独立 benchmark 复现和完整安全评测仍待核验。
