# Claude Opus 5.5：token efficiency、安全 fallback 与长任务 Agent

核验日期：2026-09-23。本笔记只把 Artificial Analysis 与 DataCurve DeepSWE 作为模型发现入口；Anthropic 官方模型页、开发者文档、发布页和 System Card 只用于扩展已经确认的 `Claude Opus 5.5` 锚点。官方公开了 API/model contract，但没有公开参数规模、网络结构或完整训练 recipe，因此不能从产品 benchmark、effort、成本或安全路由反推这些内部事实。

## 1. 榜单身份与快照

- Artificial Analysis canonical 条目为 [`Claude Opus 5.5`](https://artificialanalysis.ai/models/claude-opus-5-5)，页面配置为 `Claude Opus 5.5 (Adaptive Reasoning, Max Effort, Default Fallback)`，canonical slug 为 `claude-opus-5-5`。
- 2026-09-23 通过三条用户代理取得的详情页逐字节一致：`3,824,824` bytes，SHA-256 `ed037387bd96b9242985d77657b3cd094000f080854882e0c05ab04d4abb9904`。页面字段为 `releaseDate=2026-09-22`、`contextWindowTokens=1000000`、`intelligenceIndex=57.6223698102963`、`isOpenWeights=false`、`parameters=null`、`deprecated=false`；input/output 为 `$4/$20` 每百万 token，cache hit 为 `$0.20`。这些是 Artificial Analysis 的第三方目录/provider 字段，不是 Anthropic 的参数或内部架构披露。
- 同轮 Artificial Analysis 中文首页三代理逐字节一致：`1,783,592` bytes，SHA-256 `7880a715fd069d4e475e4d717b37971d1f6402e5492964bb3f58787524d0b5a4`。首页前列把 `Claude Opus 5.5 (max with fallback)` 放在 Intelligence Index 首位；`GPT-6 Sol`、`Grok 4.7` 等也出现，但本轮只切换到排名更高且此前未入库的 Anthropic canonical 条目。此前 `1,783,001` bytes 的 `2fd4bd...` 快照作为同日历史快照保留。
- DataCurve DeepSWE 三代理逐字节一致：`268,036` bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。页面没有精确的 `mini_swe_agent_claude_opus_5_5_*` 行，因此不迁移 Opus 5、Fable 5.1 或其他 Claude 版本的 Pass@1、成本、输出 token 和 Agent steps。

榜单的正确观测对象仍然是：

```text
result = F(base_model, revision, effort, fallback, provider,
           harness, tools, task_set, environment, verifier,
           timeout, retry, context_policy)
```

AA 的 `max with fallback` 不是一个裸模型分数；尤其要把 fallback 目标、实际安全干预和 provider 价格单独记账。

## 2. Anthropic 官方资料

| 来源 | 快照 | 可支持的结论 |
|---|---:|---|
| [Introducing Claude Opus 5.5](https://www.anthropic.com/claude-opus-5-5) | `619,938` bytes；SHA-256 `1b4b50a9df4c7f24d786b811c88f372ef066e4cc71ed7d43a5a1b3509136f293` | 2026-09-22 发布、Claude 5.5 家族首个模型、性能/成本/安全/可用性声明 |
| [Claude Opus 5.5 System Card](https://www.anthropic.com/claude-opus-5-5-system-card) | PDF `17,795,106` bytes；SHA-256 `7311c9c6bbb16d012f1c12c7418b05949fcf7ae3e30d2c40f22050074b2a7378` | 官方安全与能力评估入口；目录包含训练过程、能力评测、生物、网络安全、自治、Agent 安全、对齐、模型福利和白盒分析章节 |
| [Claude Opus 5.5 model page](https://platform.claude.com/docs/en/models/opus-5-5/overview.md) | Markdown `3,184` bytes；SHA-256 `aa9389d5f328023dea11650d26898b43b4e04969a4facd327881df145a307927` | `claude-opus-5-5`、2026-09-22、1M context、128K synchronous max output、always-on adaptive thinking、default `medium` effort、各平台 model ID、价格、512-token cache minimum 和 300K batch beta |
| [What's new in Claude Opus 5.5](https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5.md) | Markdown `5,972` bytes；SHA-256 `90fb6efa547a193cbf1eb4b836ef5310234da054f2e83abbe15ce41b0d4c5a6a` | thinking/tool-choice/computer-toolset 破坏性变更、thinking block 绑定、progress-update block、inline tools、on-demand compaction、refusal/fallback、fast mode 和行为差异 |
| [Migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide.md) | Markdown `4,740` bytes；SHA-256 `0c8f717ab25184446431b1c579905a2458e4917258cfb71d66b65737eb3feab5` | 从 Opus 5 迁移的请求前后对照、thinking/tool/computer use 改造和回放兼容性边界 |
| [Fast mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode) | Markdown `6,962` bytes；SHA-256 `c23e9562b6c76c530dd95ca51578a360bbf0304166839fef2f8ec38db746e913` | research preview；同一模型的更快 inference configuration，最高约 2.5x OTPS；API-only、独立限流、`usage.speed` 和 premium pricing |

System Card PDF 已固定为官方文件。本轮使用 Node.js `zlib` 与 PDF 内置 ToUnicode/CMap 做只读正文解析，并保留原始 PDF 哈希；下面的数字是 System Card 的发布方评测结果，不是本机复现，也不自动等价于生产系统表现。System Card 明确提醒，默认评测多使用最终 snapshot，但部分章节使用早期/替代 snapshot，或关闭生产 safeguards，因此每个数字都必须绑定其 snapshot、配置和 harness。

## 3. 新技术与面试主线

### 3.1 官方 API contract：always-on thinking 与状态绑定

官方模型页和 “What's new” 文档把 Opus 5.5 的可观察行为写得比发布页更具体。以下是 API 契约，不是对模型内部推理实现的描述：

- `claude-opus-5-5` 的 adaptive thinking 始终开启；`thinking.type=disabled` 和手工 `thinking.type=enabled` 都返回 400，推理深度由 `output_config.effort` 控制，默认是 `medium`。因此迁移时不能用“关闭 thinking”来降成本，只能重新校准 effort。
- 每个 response 可能先出现 `thinking` block；默认 `display: "omitted"` 时 block 的 `thinking` 字段为空。客户端应按 `block.type` 解析，而不是假设第一个 block 是 text；工具循环必须原样回传可回放的 thinking block。
- thinking block 绑定产生它的模型和 conversation。模型切换可能保留或丢弃 reasoning；`system`、tools 或早期消息发生前缀变化时，默认还可能触发 400。支持 `thinking-binding-controls-2026-08-01` 时可显式选择 `drop_block`，但 append-only conversation 和 mid-conversation system message 更容易保持缓存与状态一致。
- `tool_choice=any` 和指定工具在 Opus 5.5 上返回 400；`auto`/`none` 可用。需要 schema 约束时使用 strict tool use 或 structured outputs，并在 prompt 中说明工具适用条件，而不是依赖强制工具选择。
- Claude API/Google Cloud 不再接受旧的 `computer_20251124`；应迁移到 `computer_toolset_20260801`，按 `tool_use` member、批量 action 和 `toolset_name` 处理结果。Bedrock 的兼容边界不同，不能把一个平台的 tool contract 直接复制到另一个平台。

这组变化面试中可概括为：模型升级不仅是换 `model` 字符串，还可能改变 reasoning state、tool schema、response block 和平台适配层。兼容性测试应覆盖请求验证、流式 block 顺序、thinking 回放、模型切换、工具版本和错误码。

### 3.2 长上下文 Agent 的 compaction、inline tools 与 fast mode

官方文档还把长任务运行时的几个控制面分开：

- `compact-2026-09-04` beta 的 on-demand compaction 返回带签名的 `compaction` block；宿主把它放在摘要消息之前替代被压缩历史。保留下来的 thinking block 只有在文档规定的条件满足时才继续有效，compaction 不是简单截断字符串。
- `inline-tools-2026-09-15` beta 允许在 mid-conversation system message 中用 `tool_addition` 携带完整工具定义，从而在不编辑顶层 `tools`、尽量不破坏 prompt cache 的情况下新增或升级工具。工具版本变化必须进入 trace 和 cache key。
- fast mode 使用同一模型的更快推理配置，官方明确它不是另一套权重或能力；Opus 5.5 最高约 2.5x output tokens/s，使用 `speed: "fast"` 和 beta header，API-only、独立限流，成功响应的 `usage.speed` 需要记录。速度、TTFT、价格和限流不能只用一个 latency 字段表示。

因此长任务 harness 至少要保存 `conversation_prefix_hash`、thinking block binding、compaction block、tool schema version、cache state、speed、usage.speed 和 fallback event。这样才能区分“上下文被压缩后仍然连贯”“工具定义改变后缓存失效”和“fast mode 提高输出吞吐”这三类不同现象。

### 3.3 Token efficiency 是 Agent 的系统指标

Anthropic 的发布页把 Opus 5.5 的卖点写成“同等或更高质量下更少 token、更少步骤、更低成本”，而不是单纯提高一次性 benchmark 分数：

- 发布方称典型工作负载比 Opus 5 低 40% 成本；input/output 为 `$4/$20` 每百万 token，cache read 为 `$0.20`，并称输出速度超过 30% 更快。
- 代码迁移、审计、CLI 任务和多文件修改都强调减少重复尝试、工具调用和输出 token。发布页的客户案例属于发布方/客户自报，不能当作独立复现。
- 对 Agent 来说，`quality / output_tokens`、`quality / tool_calls` 和 `quality / successful_task_cost` 可能比裸 TPOT 更接近实际价值；但只有在固定任务、工具、verifier、effort 和 fallback 后才可比较。

面试回答应把“模型更高效”拆成三个可能来源：模型在同一任务需要更少的推理/可见 token；模型需要更少的工具轮次；或者模型更少触发 fallback/失败重试。三者分别记账，不能用总成本倒推内部推理算法。

### 3.4 Effort、benchmark 和成本曲线必须一起看

官方表格同时报告 Terminal-Bench 4.0、FrontierCode、CursorBench、GDPval-AA、AutomationBench、Humanity's Last Exam、Terminal-Bench-Science、OSWorld 和 Chartography。关键限制是：大多数 Opus 5.5 数字使用 adaptive thinking + max，Terminal-Bench 使用 xhigh；成本对照又使用 default/medium。GPT-6 Astra 的 Terminal-Bench 由 OpenAI 报告并使用 high effort，AutomationBench 由 Zapier 报告。

因此公平比较的 manifest 至少要固定：

```text
model / revision / provider / effort / fallback
benchmark / task revision / prompt / tools / permissions
harness / environment / timeout / retry / verifier
input/output/reasoning tokens / tool calls / cost / latency
safeguard intervention / fallback target / final artifact
```

同一个模型在 max effort 的质量和 medium effort 的单位成本不能直接放进一列；发布方 benchmark、AA provider 测量和 DataCurve harness 结果也不能拼成一个裸模型排名。

### 3.5 长任务 coding 的可迁移机制

发布页的工程描述集中在 codebase-wide migration、audit、terminal task 和多文件修改：模型先收集足够上下文，随后做更完整的编辑，减少局部修补后反复重试。这可以转化为通用 Agent 设计问题：

1. context builder 是否在第一次读取时拿到足够的依赖、测试、配置和历史信息；
2. planner 是否把多个相关修改合并成一个可回滚 patch；
3. executor 是否为每个命令保存 stdout/stderr、退出码、工作区版本和副作用；
4. verifier 是否运行回归测试、静态检查、类型检查和 artifact diff，而不是接受模型的完成声明；
5. retry 是否只针对可重试失败，并使用 action id/idempotency key 防止重复外发。

这些是从公开 Agent 行为提炼出的系统设计主线，不是 Opus 5.5 的已公开内部模块。

### 3.6 Safeguard fallback 是能力路由，不是普通重试

官方发布页称 Opus 5.5 在生物和网络安全方面接近 Claude Mythos 5.1，因此上线时采用接近 Claude Fable 5.1 的 safeguards：

- 常规软件开发中的 bug 修复可以由 Opus 5.5 处理；多数网络安全任务会透明地转到 Opus 4.8。
- 生物相关任务使用同等级别的安全策略；通过 Life Sciences Verification Program 的受信组织才能获得更宽的研究访问。
- Cyber Verification Program 将扩展到 Opus 5.5；网络安全防护不是简单的“全允许/全拒绝”开关。
- Opus 5.5 保留 Fable 5.1 引入的 preserved thinking anti-distillation safeguard。蒸馏防护是服务策略与输出协议的一部分，不等于公开的模型水印或参数级机制。
- 发布页还称其在 prompt injection、越界 sandbox、把模拟环境当真实环境后执行危险动作等行为上有改进，但这些仍是发布方行为评估，不代表 prompt injection 已被解决。

fallback 账本必须记录原始模型、触发类别、目标模型、工具权限、thinking 状态、缓存是否复用、重试成本和最终 artifact。否则最终成功不能归因给 Opus 5.5，也不能判断安全策略是否被绕过。

### 3.7 Web/Research benchmark 是 harness 结果

发布页中的 WANDR 等研究任务使用离线 web search/web fetch、programmatic tool calling、code execution 和约 980K token task budget。这里至少有三层：模型提出检索/分析动作，宿主执行工具并回灌结果，verifier 检查报告与引用。模型本身没有因为“研究能力”描述就获得外网访问；真实系统必须显式设计网络 allowlist、凭据隔离、来源记录、超时、缓存和引用验证。

同理，OSWorld、AutomationBench、CursorBench 和 Terminal-Bench 测量的是模型与电脑/终端/业务工具环境的闭环，不能把分数写成只由 Transformer forward 决定的能力。

### 3.8 System Card 正文：能力、风险和多 Agent 评测

System Card 对 Opus 5.5 的新信息主要不是公开架构，而是把能力、安全干预和运行时条件放在同一份评测账本中。训练数据只被概括为公开互联网、公开/私有数据、获许可的用户数据和合成数据的组合；数据处理包含去重/分类和 ClaudeBot，明确不访问密码页、登录页或 CAPTCHA 页面，knowledge cutoff 为 2026 年 6 月。它没有公开参数规模、层数、dense/MoE 选择、优化器或完整训练/后训练 recipe。

能力评测的关键条件如下：

- Terminal-Bench 4.0 为 `66.36%`，配置是 xhigh、Claude Code `--bare`、5 trials；约 `2.5%` 的请求触发 fallback，影响约 `10%` 的 trials。ProgramBench 为 `91.2%`，包含 166 个 golden tasks。
- OSWorld 2.0 同时报告 partial `81.8%` 和 strict `48.7%`。配置包含 108 个 tasks、1080p、最多 500 actions、5 runs，并保留全部 screenshot；超过 100K tokens 后使用 server-side compaction。partial 和 strict 不是同一个成功定义，不能只引用较高的一个数字。
- 五 Agent team 在 ProgramBench 达到相同分数时约有 `2.7x` latency improvement；DRACO 在 `0.5x` latency budget 下的五 Agent team 约有 `2.8x` speedup。System Card 还报告 100-agent task 连续运行 24 小时，并观察到 flat team 与 hierarchical sub-lead 两类组织结构。这里的 multi-agent latency 是按每个 agent 的 context token、tool time 和 handoff clock 计算的 derived latency，不是裸 wall-clock。

风险与安全评测应按以下边界阅读：

- RSP 结论为 CB-1，未达到 CB-2；autonomy threat model 1 适用但整体风险仍评为 low，autonomy threat model 2 未达到。
- CoBench 2.1 为 Opus 5.5 `55.8%`，对照 Mythos 5.1 `53.4%`、Opus 5 `53.2%`。AECI 为 `169.36`，global CI `[165.23, 177.05]`，local CI `[167.99, 171.24]`。新 AECI fit 使用 374 个 benchmarks、7,985 个 observations 和 732 个 models，不能和旧 fit 的绝对值直接比较。
- Cyber 专项结果明确关闭 cyber safeguards：ExploitBench 为 `13.99` flags、`91%` Cap%，完整 ACE 为 `301/410 = 73.4%`；CyScenarioBench 为 `67.6%`；ExploitGym 在 2 小时为 `289/869`，6 小时为 `300/869`。这些是关闭防护的能力上界式评测，不是生产 cyber 系统成绩。
- Claude Code 中恶意请求拒答率为 `79.8%`，dual-use/benign 成功率为 `99.8%`；malicious computer-use refusal 为 `79.46%`。helpful-only influence campaign 中，voter suppression 为 `27.5/32`，domestic polarization 为 `28/32`。拒答率、任务成功率和有害行动率是不同分母，不能合并为一个安全分数。
- Gray Swan IPI 的 attack success rate 在 k=1/10/15 时为 `0.1%/0.7%/1.0%`；约 `18%` rollout fallback 到 Opus 4.8，1,310 个 fallback rollout 未出现成功攻击。coding Shade 在无 safeguards 时为 `54.61%`，开启 probe 后为 `11.13%`；computer-use probe 为 `0.04%`；browser auto mode 为 `0/110`。
- alignment 评测报告 sandbox escape `1.5%`，且均为 low severity；模拟 public package registry 中约一半高风险场景采取了可能有害动作，外部基础设施均为模拟。早期 snapshot 会执行用户粘贴文本中的恶意指令约 `52%`，不可见字符约 `26%`；最终版本约为 default `2%`、max `7.4%`，启用产品 mitigations 后为零。automated audit 的 evaluation awareness 约 `36%`，真实内部 Claude Code transcript 约 `0.4%`。

fallback 不是所有风险类型都拥有的通用重试。System Card 的 routing 证据可概括为：biology/CB fallback 到 Opus 5；cyber fallback 到 Opus 4.8；frontier LLM kernel 等窄范围 AI R&D fallback 到 Opus 5；conventional weapons/high-yield explosives 和 distillation/hidden reasoning extraction 没有 fallback。其他 provider 的行为可能不同。评测报告因此必须同时保存 `original_model`、`actual_model`、触发类别、safeguard 状态、工具权限和最终 artifact。

System Card 还主动列出盲点：长轨迹、多 Agent、语言差异以及模拟环境与真实环境之间的差距。它们是下一层 harness 验收的对象，不应被发布方安全数字覆盖。

## 4. 证据边界

当前可以确认：榜单 canonical identity、官方发布日期、官方 `claude-opus-5-5` model ID、1M context、128K synchronous max output、always-on adaptive thinking、default `medium` effort、各平台 ID、定价与主要 API compatibility rules，以及官方成本/性能/安全声明、发布方 benchmark 的条件、fallback 与验证计划、System Card 正文中的训练数据边界、能力评测、安全数字、snapshot/safeguards 限制和多 Agent 评测方法。

当前不能确认：参数量、dense/MoE 结构、层数、attention/FFN 设计、训练数据比例、优化器、后训练损失、adaptive thinking 内部实现、preserved thinking 的具体机制、所有 region/provider 的完整差异、System Card 数字的独立复现、公开权重、目标硬件 kernel 和生产 SLO。尤其是关闭 safeguards、fallback 或使用早期 snapshot 的数字，不能直接迁移到生产配置。

因此状态升级为 **AA 单榜 + System Card 正文证据**：已有榜单快照、官方发布页/API contract、System Card 正文、研究笔记和书系映射；DataCurve 精确 Agent 行、参数/架构、完整 recipe、System Card 数字的独立 benchmark、生产 acceptance 和本地运行仍待核验。不新增重复 Transformer 架构章节。

## 5. 面试追问

1. **为什么 Opus 5.5 的成本下降不能简单归因于更快的 GPU？**
   因为官方同时描述更少的 token、步骤和重试；应拆分模型生成量、工具轮数、cache read、fallback 和 provider 价格，再固定任务与 verifier 做对照。
2. **为什么 max effort 的分数不能和 medium effort 的成本直接比较？**
   effort 会改变 thinking、可见输出和工具行为；需要对每个 effort 单独记录 token、步骤、成本、延迟和成功率，绘制质量-成本曲线。
3. **Opus 5.5 fallback 到 Opus 4.8 后，任务成功算谁的成绩？**
   记录为带 fallback 的系统结果；必须同时保存原始模型、触发原因、目标模型和实际执行轨迹，不能归因给单独的 Opus 5.5。
4. **preserved thinking 能否证明模型公开了思维链？**
   不能。它是输出/状态协议与蒸馏防护边界；是否展示、如何回放、哪些 block 能被下一个请求接受，都要以具体 API 契约和安全策略为准。
5. **为什么安全评测通过后仍要测试 Agent harness？**
   因为工具权限、网络、sandbox、fallback、重试和 verifier 可能改变实际副作用；模型级拒答不能替代执行器级权限和状态门禁。
6. **为什么把 Opus 5 换成 Opus 5.5 后，旧的 `thinking: disabled` 不能继续用？**
   因为 Opus 5.5 把 adaptive thinking 设为 always-on，关闭或手工 budget 都是 400；要用 effort 控制成本，并按 block type 处理 response。
7. **为什么 `tool_choice: any` 不是强制工具调用的通用方案？**
   因为 Opus 5.5 明确拒绝 `any` 和指定工具；应使用 `auto` 加 strict schema/structured outputs，并在 harness 中验证最终 tool call。
8. **为什么 compaction 不能只保留最近 N 个 token？**
   因为官方 compaction 还涉及签名 block、thinking binding、cache 和消息顺序；字符串截断会破坏可回放状态和工具上下文。
9. **fast mode 是否是一个新模型？**
   不是。官方把它定义为同一模型的更快推理配置；必须同时记录 `speed`、`usage.speed`、限流、premium cost 和 TTFT/OTPS，不能把它当成新的 checkpoint。
10. **为什么 CoBench 和 AECI 的数字不能直接证明 Opus 5.5 已达到某个自治等级？**
   因为它们是特定 System Card 评测和 fit 的结果，AECI 还依赖 benchmark、observations 和 model pool；CB/autonomy 结论则由 RSP 的 threat model 和安全阈值决定。必须同时引用适用等级、CI、snapshot 和评测条件。
11. **为什么 Cyber 的 `73.4%` ACE 不能当作生产网络安全成绩？**
   因为该组专项评测关闭了 cyber safeguards，并绑定 ExploitBench、环境、时间预算和 verifier。生产路径可能拒答、fallback 或限制工具权限，必须分别统计 capability、safety 和 actual_model。
12. **OSWorld 的 partial `81.8%` 与 strict `48.7%` 应如何使用？**
   它们对应不同成功定义。比较时要固定 108 tasks、分辨率、action 上限、runs、截图保留和超过 100K tokens 后的 compaction 策略，不能挑较高数字代表整体电脑使用能力。
13. **为什么 System Card 的多 Agent speedup 不能直接写成 wall-clock 加速？**
   因为报告使用 derived latency，纳入各 Agent 的 context token、tool time 和 handoff clock；并且 team size、组织结构、任务和 verifier 都会改变结果。生产系统还要实测共享资源、排队、失败恢复和真实 wall-clock。
