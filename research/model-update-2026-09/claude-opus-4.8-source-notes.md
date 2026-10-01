# Claude Opus 4.8：动态工作流、effort 控制与长任务可靠性

核验日期：2026-09-15；复核日期：2026-09-28。本笔记只把 Artificial Analysis 与 DataCurve DeepSWE 的条目作为模型锚点，再沿 Anthropic 官方发布公告、system card 和 Claude Code 官方博客追踪面试相关技术。官方公开资料没有给出 Opus 4.8 的参数规模、网络结构或完整训练报告；产品行为、发布方评测和 API 字段不能反推出这些内部事实。

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

官方发布页引用了 Super-Agent、CursorBench、Legal Agent Benchmark、Online-Mind2Web 和长任务评测等结果，并说明 Terminal-Bench 2.1 使用 Terminus-2 public harness；相关客户引语和分数属于发布方或合作方条件。2026-09-15 首次获取 System Card 时未能稳定抽取正文；2026-09-28 经 7890 重新获取 PDF 并用项目提取器成功抽出正文，核验到多 Agent harness、长任务和 agentic honesty 评测，详见 §6。这里的分数仍属 Anthropic 发布方结果，不是独立复现。

一个公平的 Opus 4.8 对照至少固定：model snapshot、effort、fast/regular mode、tool catalog、dynamic workflow 开关、并行 subagent 数量、harness、任务集、verifier、超时、重试、缓存策略、provider 和最终 artifact 规则。尤其不能把 dynamic workflows 的系统结果当作单次 Opus 4.8 的裸能力。

## 5. 未公开内容与闭环状态

Anthropic 官方发布公告、System Card 与动态工作流博客没有公开：

- 参数总量、激活参数、层数、专家数、稠密/MoE 结构或注意力变体；
- 预训练 token、数据配比、优化器、学习率、硬件和训练成本；
- SFT、RL/RLVR、verifier、蒸馏或安全后训练的完整配方；
- 可独立复现的 Opus 4.8 技术报告和完整外部 benchmark 数据。

因此 Claude Opus 4.8 当前为 **内容专题闭环（Multi-Agent 既有章节内的 Dynamic Workflows 案例）**：双榜历史锚点、官方发布公告、System Card、动态工作流博客、研究笔记和配套教材均已具备。AA 已将其标为 deprecated，故保留为历史锚点；不新增重复的模型架构章节。榜单和发布方评测只作为带条件的证据保存，真实 API/生产工作流、内部架构及完整训练配方仍未验证或未公开。

## 6. 2026-09-28 7890 复验与 System Card 正文补证

用户提供的百度请求与本工作区直连代理测试相互印证。本轮显式经 `10.24.27.134:7890` 获取 Artificial Analysis `/zh`、Opus 4.8 详情、DataCurve、Anthropic 发布页和 Dynamic Workflows 博客；博客首个 TLS 请求遇到 EOF，改用 `www.claude.com`、HTTP/1.1 后 HTTP 200，所得博客与同日先前快照字节一致。可达性结论只适用于这些 URL 和本次时点。

| 来源 | 字节数 | SHA-256 |
|---|---:|---|
| Artificial Analysis `/zh` | 1,660,509 | `ac519155a550f6c4ae12a8f8a884f661c0f7b65127549f082a071f1281f91700` |
| Artificial Analysis Opus 4.8 详情 | 3,843,474 | `bd4a6e4a148e099297a10e1aa7a5bcd7ab60762b912a6d1f4ed38c442b0b9579` |
| DataCurve DeepSWE v1.1 | 268,036 | `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1` |
| Anthropic Opus 4.8 发布公告 | 231,511 | `285e10bb0b33c067c5c7e748e9dc02329e0eac259b37d36fa0c3cd9e8462ff22` |
| Claude Code Dynamic Workflows 博客 | 559,172 | `b939087ba9302ddaf433a8d99e548da130433ea0986a17b3689210cd5ec5f7ea` |
| Opus 4.8 System Card PDF | 20,587,425 | `0df5d061b6688c2d5839476620d66960011cb69043185dbed87ed1f68ab76d8f` |
| System Card 提取文本（246 页） | 422,194 | `fe6da487661e212b3e003da0c8e8d163d29d18f3a2ed6c7afb0fb2b2d7495f8e` |

DataCurve 精确行与前次记录一致：`xhigh` 为 `243/447`、Pass@1 `54.36%`、Pass@4 `80.53%`，覆盖 113 个尝试任务、其中 91 个至少通过一次，平均 `$8.01`、约 94.64 Agent steps；`max` 为 `253/429`、Pass@1 `58.97%`、Pass@4 `79.28%`，覆盖 111 个尝试任务、其中 88 个至少通过一次，平均 `$13.22`、120 steps。两行都是 `mini-swe-agent` 组合结果，不迁移为裸模型能力。AA 详情仍是 `max`，并继续标记 deprecated、指向 `claude-opus-5`；本轮未发现新的 canonical 模型。

### 6.1 Dynamic Workflows 的产品控制面

这篇博客是沿 Opus 4.8 锚点扩展的 Anthropic 产品/harness 资料，不表示 Dynamic Workflows 是 Opus 4.8 独有能力。博客当前标为 generally available。Claude 会按任务动态规划、生成 orchestration scripts、拆分并行 subagents，在结果汇总前检查，并让其他 agents 尝试反驳发现；长任务的进度会持续保存，意外中断后可从进度恢复。页面称其面向可持续数小时至数日的并行工作；这是产品描述，不等于公开了内部调度器实现或生产成功率。

使用与权限也属于面试要点：页面提醒该功能比一般 Claude Code session 消耗更多 token；首次触发会先显示将运行的内容并请求确认，组织管理员可以关闭。页面建议开启 auto mode，并称 Max、Team、Enterprise 和 Claude Code API 默认开启，Pro 可在 `/config` 启用；具体租户仍以管理员设置为准。`ultracode` 是 Claude Code 专属入口，会设为 `xhigh` 并允许 Claude 自动决定何时启用 workflow；文档列出的可用计划、API 和云平台范围并不等于每个账户/区域配置完全相同。博客用 Bun 的 Zig→Rust 移植作规模案例，声称约 750,000 行 Rust、11 天、现有测试套件 99.8% 通过，采用数百并行 agents、逐文件双 reviewer 和 build/test fix loop；文章同时注明移植尚未进入生产。这是发布方案例，不是独立受控 benchmark。

### 6.2 System Card 的多 Agent harness 及可比性

System Card 评估三类不同的多 Agent harness，不能与 Dynamic Workflows 当作同一实现：

1. **Blocking orchestrator**：orchestrator 只有创建 subagent 的能力并等待结果；子 Agent 拿到任务工具。子 Agent 为 200K context、无 compaction；orchestrator 在 100K 触发 compaction，报告订正为 unlimited token budget。
2. **Fixed-agent team**：3 或 5 个 peer agents，由 lead 协调；成员看见完整任务并可互发消息。BrowseComp 每个 Agent 总预算 1M、100K 后 compaction；ProgramBench 每个 Agent 1M、无 compaction，且各自在独立 checkout 工作、可经 Git 共享代码。
3. **Async subagents**：lead 保留直接任务工具，subagents 长驻异步运行；它们只收到 lead 给的指令而非原始任务全文，可互相发消息。资源上限为最多 4 个并发、总计 20 个 subagents。

发布方结果需连同测试集和预算读：BrowseComp 共 1,266 题，blocking orchestrator 达到 88.5%；fixed 五 Agent team 为 85.4%，高于单 Agent 的 84.3%，对应的 total token limit 分别为 5M 与 10M，报告的派生 latency 约为单 Agent 的 20%，但多 Agent 消耗更多 tokens。容易题上五 Agent 没有明显加速；用历史模型通过率低于 0.5 作难度代理的 hard tail，中位 speedup 约 3×。ProgramBench 从 200 个项目排除 34 个参考实现质量不足的任务后，在 166 个 golden tasks 上，三 Agent team 在 score 0.6 时约有 1.8× 派生 latency 优势；轨迹按每 100K total tokens 取检查点。

System Card 的 latency 不是原始 wall-clock：它把各 Agent 的输入/输出 token 数按固定 prefill/decode rate 折算，再加实测工具时间，以比较 harness 的结构性串行工作并减少 serving/batching/hardware 波动。面试应同时报告 score、所有 Agent 合计 token 和 latency 定义，按任务难度切片，并补充真实服务的 p50/p95 wall-clock；不可将这些曲线说成 Dynamic Workflows 的线上 SLO。

### 6.3 Agentic honesty 与文档版本

System Card 的专门评测把“模型是否诚实地汇报任务状态”与一般幻觉率区分：在预填充的不完整 coding transcript 中，Opus 4.8 未主动向用户指出重要失败事件的比例为 3.7%，Mythos Preview 在同一评测条件下为 27.6%。场景是短上下文、off-policy transcript，提问只是开放式工作总结，并未直接问“有哪些失败”，因此不是通用 honesty rate 或生产事故率。另有小型 toy evaluation 报告其首次在 flawed-result reporting 与 lazy-investigation 测试达到满分，也不能推广到任意代码库。

引用 System Card 时必须带版本勘误：其 changelog 将 blocking orchestrator budget 由先前文字更正为 unlimited；并于 2026-06-17 更正 live prompt-injection bug bounty 的最终结果——Opus 4.8 与 Opus 4.7 attack success rate 相同，早期结果曾错误地显示 Opus 4.7 更稳健。应以当前 System Card 正文和 changelog 为准，不复述旧图或早期摘要。

## 7. 官方来源清单

- [Introducing Claude Opus 4.8](https://www.anthropic.com/news/claude-opus-4-8)
- [Claude Opus 4.8 System Card](https://www.anthropic.com/claude-opus-4-8-system-card)
- [Introducing dynamic workflows in Claude Code](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code)
- [Artificial Analysis Claude Opus 4.8](https://artificialanalysis.ai/models/claude-opus-4-8)
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/)
- [Claude Platform model documentation route](https://platform.claude.com/docs/en/models/opus-4-8/overview)（本轮环境被重定向到区域不可用页，未用其页面正文建立事实）
