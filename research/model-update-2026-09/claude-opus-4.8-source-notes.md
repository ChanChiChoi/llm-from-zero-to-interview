# Claude Opus 4.8：动态工作流、effort 控制与长任务可靠性

核验日期：2026-09-15。本笔记只把 Artificial Analysis 与 DataCurve DeepSWE 的条目作为模型锚点，再沿 Anthropic 官方发布公告、system card 和 Claude Code 官方博客追踪面试相关技术。官方公开资料没有给出 Opus 4.8 的参数规模、网络结构或完整训练报告；产品行为、发布方评测和 API 字段不能反推出这些内部事实。

## 1. 榜单锚点与证据边界

- Artificial Analysis 条目为 [Claude Opus 4.8](https://artificialanalysis.ai/models/claude-opus-4-8)，配置名是 `Claude Opus 4.8 (Adaptive Reasoning, Max Effort)`，canonical slug 为 `claude-opus-4-8`，页面 `releaseDate` 字段为 `2026-05-28`。
- Artificial Analysis 当前详情页把它标记为 deprecated，并指向 `claude-opus-5`；页面仍显示约 41.99 的 Intelligence Index（展示为 42）、约 56.1 output tokens/s、1M context、约 `$4.08`/Intelligence Index task，以及 Anthropic API 的 `$5/$25` 每百万输入/输出 token 价格。它们是第三方在具体配置和任务上的测量，不是永久服务状态或裸模型能力。
- DataCurve [DeepSWE v1.1](https://deepswe.datacurve.ai/) 快照显示 113 个任务、91 个仓库、5 种语言和统一的 `mini-swe-agent` harness。Opus 4.8 的 `xhigh` 记录为 243/447，Pass@1 约 `54.36%`，Pass@4 约 `80.53%`，平均成本约 `$8.01`，平均约 94.6 个 Agent steps；`max` 记录为 253/429，Pass@1 约 `58.97%`，平均成本约 `$13.22`，约 120 steps。
- 两个榜单的配置不同：Artificial Analysis 是 `max`，DataCurve 是 `xhigh` 或 `max` 加 `mini-swe-agent`。不能把 42 与 DeepSWE 的 54%/59% 拼成一个排名，也不能把工具、harness、验证器和任务集的组合结果归因给基础模型。

DeepSWE 的观测对象应写成：

```text
result = F(base_model, model_revision, effort, harness, tools,
           task_set, verifier, timeout, retry, context_policy, provider)
```

本轮快照如下，哈希只用于页面复现线索：

| 页面 | 临时文件 | 大小 | SHA-256 |
|---|---|---:|---|
| Artificial Analysis Opus 4.8 详情 | `/tmp/aa-claude-opus-4-8-20260915.html` | 3,527,387 bytes | `8382b952d1b6412022bb4940db88f6155caddadc7af318bbb365011e279742cd` |
| DataCurve DeepSWE | `/tmp/deepswe-20260915.html` | 268,313 bytes | `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7` |

## 2. 官方发布与生命周期

Anthropic 的 [Introducing Claude Opus 4.8](https://www.anthropic.com/news/claude-opus-4-8) 将 Opus 4.8 定位为 Opus 4.7 的升级版本，发布日期为 2026-05-28，并称其改善 coding、agentic tasks 和 professional work 的一致性。公告称模型当时已全面可用，API model ID 为 `claude-opus-4-8`。

价格边界是：普通模式输入 `$5`、输出 `$25`/MTok；fast mode 输入 `$10`、输出 `$50`/MTok。公告称 fast mode 约为 2.5 倍速度，并且相对前代 fast mode 降价三倍。价格与可用性是发布时的服务声明，应按平台和当前模型目录复访，不把它当成内部模型规格。

当前 Artificial Analysis 页面显示 `deprecated=true`、`deprecatedTo=claude-opus-5`，说明榜单页面已经把它视为前代模型；本项目仍保留其 2026-05-28 的历史锚点和官方发布资料，不把历史条目写成当前推荐模型。

## 3. 面试技术主线

### 3.1 Dynamic workflows：从单 Agent 到外部编排图

Anthropic 的 [Introducing dynamic workflows in Claude Code](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code) 将其描述为 Claude 动态生成 orchestration scripts，在一个 session 中运行数十到数百个并行 subagents，并在结果进入用户答案前检查工作。它适合跨大型代码库的 bug hunt、优化审计、安全审计、框架迁移和需要独立复核的长任务。

核心流程可以抽象成：

```text
prompt -> dynamic plan -> parallel subagents -> independent checks
       -> adversarial/refutation passes -> convergence -> coordinated answer
```

与普通 tool loop 的区别不只是“并行调用更多工具”：编排发生在对话外，运行状态可持续保存，被中断的任务可以从保存的进度继续；子任务结果先验证再汇总，其他 agent 还可以尝试反驳发现。这样换来的代价是更高 token 使用量、并行结果合并、取消/超时、权限和最终 artifact 验证责任。

官方博客还说明 dynamic workflows 已在 Claude Code CLI、Desktop、VS Code extension、Claude API、Amazon Bedrock、Vertex AI 和 Microsoft Foundry 可用，但具体计划、租户设置和 API 能力要以当前平台文档为准。`ultracode` 是 Claude Code 专属设置，会把 effort 设为 `xhigh` 并让 Claude 自动决定何时启动 workflow；它是产品 harness 选择，不是模型架构名。

### 3.2 Effort 是可测的行为旋钮

Opus 4.8 默认 `high` effort；官方建议困难任务和长时间异步任务使用 `extra`（Claude Code 中为 `xhigh`）或 `max`。高 effort 让模型更频繁、更深入地思考并消耗更多 token，低 effort 更快且消耗更少 rate limit；这不是固定的模型版本，也不是简单等价于 `max_tokens`。

面试中应做 effort sweep：固定模型版本、任务集、工具、harness 和 verifier，只改变 effort，同时记录成功率、thinking/output token、工具轮次、总延迟、并行度、取消率和单位成功成本。否则不能判断“更高档位更强”还是“系统给了更多尝试机会”。

### 3.3 Mid-task system entries：不中断 prompt cache 的权限更新

Opus 4.8 发布公告记录了 Messages API 的一个接口变化：`messages` 数组可以在任务中接收 system entries。harness 可以在运行中更新权限、token budget 或环境上下文，而不必把指令伪装成 user turn；公告同时强调这不会破坏 prompt cache。

它解决的是运行时协议问题，不代表模型获得了新的记忆或安全能力。宿主仍要记录每次权限变更、限制可修改字段、处理缓存失效语义，并防止模型通过动态 system 内容扩大未授权能力。

### 3.4 Honesty 与结果验证

Anthropic 发布公告称，Opus 4.8 在其评测中比前代约少四倍出现“让代码缺陷未经提示就通过”的情况，并把这一改进描述为更愿意标记不确定性、少做无证据的进度声明。这个数字属于 Anthropic 的发布方评测，不能当作独立 benchmark；面试回答应把它转译成系统设计问题：

- 要求模型提供证据和验证状态，而不是只返回完成声明；
- 让测试、lint、类型检查或领域 verifier 成为最终质量门；
- 将“未验证”“失败”“跳过”和“已验证”分开记录；
- 对长任务中的中间进度和最终 artifact 都做独立检查。

## 4. 评测与系统边界

官方发布页引用了 Super-Agent、CursorBench、Legal Agent Benchmark、Online-Mind2Web 和长任务评测等结果，并说明 Terminal-Bench 2.1 使用 Terminus-2 public harness；相关客户引语和分数属于发布方或合作方条件。system card 提供更完整的安全和能力评估入口，本轮已下载 PDF，但当前环境没有稳定的 PDF 正文提取器，因此不从文件目录或 `strings` 输出扩写安全数值。

一个公平的 Opus 4.8 对照至少固定：model snapshot、effort、fast/regular mode、tool catalog、dynamic workflow 开关、并行 subagent 数量、harness、任务集、verifier、超时、重试、缓存策略、provider 和最终 artifact 规则。尤其不能把 dynamic workflows 的系统结果当作单次 Opus 4.8 的裸能力。

## 5. 未公开内容与闭环状态

本轮可获取的 Anthropic 官方发布公告、动态工作流博客和 system card 入口没有公开：

- 参数总量、激活参数、层数、专家数、稠密/MoE 结构或注意力变体；
- 预训练 token、数据配比、优化器、学习率、硬件和训练成本；
- SFT、RL/RLVR、verifier、蒸馏或安全后训练的完整配方；
- 可独立复现的 Opus 4.8 技术报告和完整外部 benchmark 数据。

因此 Claude Opus 4.8 当前状态为 **资料级闭环**：两个排行榜的可追溯锚点、Anthropic 官方发布公告、system card、动态工作流官方博客和研究笔记均已具备；但没有独立公开架构/训练报告，不新增 Opus 4.8 专属架构章节。榜单和发布方评测只作为带条件的证据保存。

## 6. 官方来源清单

- [Introducing Claude Opus 4.8](https://www.anthropic.com/news/claude-opus-4-8)
- [Claude Opus 4.8 System Card](https://www.anthropic.com/claude-opus-4-8-system-card)
- [Introducing dynamic workflows in Claude Code](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code)
- [Artificial Analysis Claude Opus 4.8](https://artificialanalysis.ai/models/claude-opus-4-8)
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/)
- [Claude Platform model documentation route](https://platform.claude.com/docs/en/models/opus-4-8/overview)（本轮环境被重定向到区域不可用页，未用其页面正文建立事实）

