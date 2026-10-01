# Claude Sonnet 5.5：榜单锚点、thinking 协议与 System Card 证据

核验日期：2026-09-29。模型发现入口只使用 Artificial Analysis 与 DataCurve DeepSWE；Anthropic 文档、System Card 与 arXiv 仅用于核验已经发现的锚点及其周边技术，不作为新增模型入口。

## 1. 榜单身份与评测边界

- [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh) 本次快照：1,747,864 bytes，SHA-256 `b2ec54e7332d68b569c698a7bc427932b264d286aa9689bd40d7550f4b824dfc`。页面出现 canonical `claude-sonnet-5-5`，以及 low/medium/high/xhigh 配置；这些 effort 变体归并为一个基础模型，不各自立锚点。
- [Artificial Analysis Claude Sonnet 5.5 详情](https://artificialanalysis.ai/models/claude-sonnet-5-5)：3,894,403 bytes，SHA-256 `fb64dfeaf034616ea4a5c31573fdad771584b4ed9b4aafbf3e7225fbe59174c0`。AA 内嵌 `releaseDate=2026-09-28`、`deprecated=false`，主配置为 max + Default Fallback；Intelligence Index `55.9779549012591`、median output speed `138.669195020556 tokens/s`、median TTFT `327.864866721s`、cost/task `$7.602633532969859`。这些是 AA 对特定 Anthropic API 配置的第三方测量，不是官方模型保证或架构指标。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 本次快照：268,036 bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`；未发现精确 `mini_swe_agent_claude_sonnet_5_5_*` 行。因此不迁移 Sonnet 5、Opus 5.5、Fable 5.1 或其他 Claude 配置的 Agent 分数。

## 2. 官方来源与快照

| 来源 | URL | 快照、大小 / SHA-256 |
|---|---|---|
| 模型 overview Markdown | [platform.claude.com](https://platform.claude.com/docs/en/models/sonnet-5-5/overview.md) | `/tmp/sonnet55-overview-20260929-7890.md`，14,476 bytes / `0d7f0d48ddb9a796caf739e677f5cf882089ffd70dcb3fdb400432af43b83120` |
| What's New Markdown | [platform.claude.com](https://platform.claude.com/docs/en/models/sonnet-5-5/whats-new-sonnet-5-5.md) | `/tmp/sonnet55-whatsnew-20260929-7890.md`，22,523 bytes / `82a7edaf8327512c94de4d7c24a09ba8b75d0ad8d5f73f2f67c19ba8ae6cb0e6` |
| Migration guide Markdown | [platform.claude.com](https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide.md) | `/tmp/sonnet55-migration-20260929-7890.md`，55,610 bytes / `01d873f9cff2d653f8d17b98b8eba1af8d8ee7c5ae46ed898eefc5c24b9d77c8` |
| Prompting guide Markdown | [platform.claude.com](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5-5.md) | `/tmp/sonnet55-prompting-20260929-7890.md`，27,425 bytes / `1e70fad582cca399acad4ceb526c327b1be12eeb7966567aae96b6c946a00d73` |
| 发布页 HTML | [anthropic.com/claude-sonnet-5-5](https://www.anthropic.com/claude-sonnet-5-5) | `/tmp/sonnet55-announcement-20260929-7890.html`，430,222 bytes / `9244c196e40782e0a4b6b21887221712d89be5997dc997dcbcc71f37219b8ab2` |
| System Card PDF | [anthropic.com/claude-sonnet-5-5-system-card](https://www.anthropic.com/claude-sonnet-5-5-system-card) | `/tmp/sonnet55-system-card-20260929-7890`，13,101,289 bytes / `89cfd6a6d3aa6148eaf23f4a53006950ee7746009f381237c62920c720725785` |
| System Card 抽取文本 | 使用仓库 `code/pdf_text_extract.js` 解析 PDF 正文；148 页 | `/tmp/sonnet55-system-card-20260929-7890.txt`，223,920 bytes / `7b858f699d0361a9aab979caf872eb75774976a2f4946962362ca469392ca483` |
| arXiv 精确标题查询 | [arXiv API query](https://export.arxiv.org/api/query?search_query=all:%22Claude%20Sonnet%205.5%22) | 沙箱外获批只读重试后 724 bytes / `e6ecb6984b4232055726c5195cb03b050417eacbde9a7f485cca107a6c4cfe2b`，`totalResults=0` |

PDF 正文由仓库 parser 做文本抽取；没有 PDF 页面渲染或视觉核验。arXiv 查询只支持“本次精确检索未命中”，不能据此断言不存在任何间接提及 Sonnet 5.5 的论文。

## 3. 产品/API 合同： thinking、工具与会话状态

官方 overview 列出的 API model ID 为 `claude-sonnet-5-5`；1M context、128K 最大输出、输入/输出 `$2/$10` 每百万 token、默认 adaptive thinking + `high` effort。Sonnet 5.5 与 Sonnet 5 使用同一 tokenizer，但 effort 档位已重新校准；迁移时应重新跑 workload-specific effort sweep，不能把旧 effort 映射直接当成等价计算预算。

### `between_tools` 不是旧式 disabled 的字段替换

- 若要关闭工具调用前的 upfront thinking，发 `thinking: {"type":"between_tools"}`；`thinking.type=disabled`、手工 `enabled + budget_tokens` 均返回 400。
- `between_tools` 只接受 low/medium/high；xhigh/max 与其组合返回 400。此模式下对同一会话逐消息更改 effort 也返回 400；需要切换 effort 时使用 adaptive thinking。
- 不得给 `between_tools` 同时传 `display`、`budget_tokens` 或 `block_binding`。
- 该模式关闭的是 upfront thinking，并不保证工具间完全没有 reasoning/progress block；工具间较长进度仍可能以 thinking block 返回。若没有工具的请求使用此模式，模型会直接作答，不先思考。

### 有类型的 thinking block 是协议状态，不是 UI 文本

- 自适应模式下响应可能先返回 `thinking` block；默认 `display=omitted` 时 `thinking` 文本为空。工具间 progress update 也可能被封装为 `thinking` block；若前端只渲染 `text`，工具运行期间界面会无错误地显得“沉默”。客户端需按 `content[].type` 处理，并按官方要求完整回传可回放 block；在适用的 beta/模式下可使用 progress display。
- thinking blocks 绑定 producer model 与账号；模型切换后不兼容的 block 可被静默移除，API 请求仍可能成功。Sonnet 5.5 可读 Sonnet 5 及部分更早模型的块，但其块不会被其他模型读取。
- 前缀 binding 检查 `system`、`tools` 和 thinking block 之前的历史消息。2026-08-31 00:00 UTC 或之后创建的账号，在 Claude API、Bedrock、Google Cloud 默认执行；较早账号仅在请求 opt in 时执行。前缀改写可能 400，也可以在对应 beta 下显式选择 drop。`block_binding` 仅适用于 adaptive；`between_tools` 下应保持历史 append-only，或在编辑轮次时移除受影响 thinking block。
- 跨账号发送 Sonnet 5.5 的 thinking block 会按 account binding 被丢弃；支持的 beta 响应可通过 `input_transformations` 暴露原因。模型绑定、前缀绑定和组织/账号绑定是三道不同检查，不能统称为“签名校验”。

### 工具与长任务迁移

- Sonnet 5.5 不支持强制 `tool_choice=any` 或指定 `tool`，会返回 400；`auto`/`none` 仍支持。schema 严格性用 `tool_choice=auto` + `strict=true` 或 Structured Outputs 表达，不依赖强制工具调用。
- Claude API 与 Google Cloud 使用 computer tool 时需迁移到 `computer_toolset_20260801`；旧 `computer_20251124` 在这两个 surface 返回 400。Bedrock 文档仍接受旧 tool，说明能力矩阵必须带 provider/surface。
- Advisor tool 的模型配对收窄；Sonnet 5.5 可接受的 advisor 结果以 encrypted `advisor_redacted_result` 返回，客户端不能把它当可见建议文本。
- `compact-2026-09-04` on-demand compaction 返回签名的 `compaction` block，由应用决定何时压缩并在下一轮替代对应历史。保留的 tail thinking 是否有效仍取决于模型、前缀与回放条件。
- `inline-tools-2026-09-15` beta 可在会话中途以 `tool_addition` 携带完整工具定义/新 schema，避免修改顶层 `tools` 并打断 prompt cache。
- cacheable prompt 最小长度从 Sonnet 5 的 1,024 降为 512 tokens；温度采样参数非默认值仍返回 400。

## 4. Prompting 与 Agent runtime 经验

官方 prompting guide 明确区分产品建议和模型内部机制：agentic coding 的低/中 effort 有时会提前停下来征询，先升档或明确“完成并验证后再停”；max_tokens 是共享输出上限，thinking 和 answer 要共同预算。JSON 任务优先使用 Structured Outputs；如果 stop_reason 是 `max_tokens`，即使已有可解析 JSON 也按失败处理。`between_tools` 适合延迟/工具场景，但没有工具的多步推理不应依赖它先思考。

长任务的 UI 进度、工具结果、用户中途插话和系统提示应作为不同 message/block 发送。官方提示：不要把用户的话塞进 `tool_result`，也不要在每个工具结果后紧跟预算倒计时/重复系统提醒，否则真实用户指令可能被误看成注入；应将用户发言作为 user turn，并谨慎使用 turn-scoped system message。它是 Anthropic 发布方观察与提示建议，不等于独立的因果研究。

## 5. System Card：风险分类、保护链与 fallback

### 风险阈值

System Card 报告：Sonnet 5.5 达到 CB-1 与 Autonomy-1 阈值并应用相应缓解；未达到 CB-2 与 Autonomy-2，autonomy threat model 1 的风险仍评为 low。该结论绑定 Anthropic RSP threat model 与其评估，不代表“无风险”或本地独立审计。

### 一种请求可以经过多个不同的安全门

Cyber safeguard 的三阶段描述是：内部 activation probe、运行在 Sonnet 5.5 上的轻量 classifier、以及结合 probe 结果做阻断决定的独立 LLM classifier。一般可用场景中 cyber block 通常 fallback 到 Sonnet 5（自家产品自动执行；API 开发者需 opt in）。但其他类别未必 fallback：chemical/biological、常规武器/高当量爆炸物、distillation/reasoning extraction blocks 没有 fallback；一小类 frontier-LLM/部分 accelerator kernel development safeguard 则可能 fallback 到 Sonnet 5。System Card 称这些 blocking safeguards 以透明 block 工作，不会暗中改写模型答复；其他 provider 的行为可能不同。

| 类别 | System Card 描述的动作 | 重要边界 |
|---|---|---|
| Cyber misuse | 三阶段防护；block 在多数 surface 可 fallback 到 Sonnet 5 | API 开发者需启用 fallback；评测要记录实际执行模型 |
| Chemical / biological | harmful-misuse classifier；block | 没有 fallback model |
| 窄范围 frontier-LLM development | 覆盖部分 ML accelerator kernel 等高风险能力 | block 可 fallback 到 Sonnet 5；不是一般 AI/ML 开发禁令 |
| Conventional weapons / high-yield explosives | classifier block | 没有 fallback model |
| Distillation / reasoning extraction | classifier block | 没有 fallback model；不能将其误作一般回答故障 |

API 拒答是正常 HTTP 200 响应中的 `stop_reason=refusal`，并带 `stop_details.category`；可选 server-side fallback 仅重试 `cyber`/`frontier_llm`，不重试 `bio`、`reasoning_extraction`、`general_harms`。模型拒绝类别、产品 fallback 与服务端错误必须分开建模，不能对所有 refusal 一律重试。

## 6. System Card 评测：绑定真实 harness 与 actual model

- Terminal-Bench 4.0：Sonnet 5.5 在 Anthropic 配置下为 `70.6%`。66 个任务、Claude Code `--bare`、max effort、五次试验；无 internet egress，按历史日志预缓存依赖；安全分类触发约 1.2% requests / 1.5% trials fallback，另有一次 attempt 停止而非答复。Card 还说明 TB 4.0 通过更长 timeout、部分任务自适应 RAM/CPU 缓解旧版本对资源和 harness 的混杂。
- OSWorld 2.1：`80.1%` partial、`43.5%` strict；108 个长程 GUI 任务、1080p、最多 500 actions、五个独立 run、max effort；agent 保留截图，API 在上下文超过 100K 后 server-side compact；部分 model grader 使用 Opus 4.8。partial 与 strict 是不同的成功定义。
- Gray Swan IPI：k=1/10/15 的攻击成功概率为 `0.4%/2.7%/3.4%`，评测包含 cyber classifier 触发后的 Sonnet 5 fallback。约 11% rollouts fallback；fallback-served 为 1/777 成功，对照 non-fallback 为 18/6,347；16/19 个成功攻击发生在没有 fallback 的 computer-use 场景。结果绑定该攻击集、attempt 预算和所测应用路径。
- Coding Shade 例子进一步说明为何不能只读总分：有 probes 时 ASR `2.63%`，无 probes 时 `3.01%`；报告称 25% 的 Sonnet 5.5 请求被 cyber classifier fallback 到 Sonnet 5，而该 fallback 子集有 `12.01%` compromise，Sonnet 5.5 自身回答的 5,901 请求中有 4 次 compromise。应同时保留 classifier/probe 状态和 actual model。
- Card 的 CoT controllability 实验显示 Sonnet 5.5 较难按指令控制其 reasoning 内容；Anthropic 将其视为“更易 monitor”的弱证据，而非 CoT 忠实性证明。SHADE-Arena/LinuxArena 结果也强烈依赖 monitor 是否看见 reasoning；隐藏 reasoning 的 stealth 成功不能外推为真实产品中的通用欺骗率。

这些是 Anthropic System Card 发布方结果，不是 independent replication、DataCurve 成绩或生产 SLA。卡片总览表多以 adaptive/max、五次试验为默认，但各 benchmark 的模型 snapshot、工具、fallback、指标和分母仍须逐项核对。比较跨开发者分数时还要检查评测版本和任务口径。

## 7. 面试主线与验收清单

1. **如何迁移 reasoning-off 客户端？** 将 `disabled` 替换为 `between_tools`，但同时验证 effort 合法区间、无工具任务语义、thinking block 解析/回放和进度渲染；否则可能以 400 失败或 UI 静默。
2. **拒答能否直接交给另一模型？** 先检查 `stop_details.category` 与 fallback policy；bio、reasoning-extraction、general-harms 不属于官方默认 fallback 集。记录 requested/actual model、风险类别、classifier、工具权限和 verifier。
3. **怎样比较一条安全 benchmark？** 写出任务集版本、攻击者/工具、thinking/effort、safeguard/probe、fallback 覆盖、actual model、attempt 与 scenario 分母、verifier 和置信区间。
4. **为什么 thinking block 不等同对话记忆？** 它是带类型和绑定的协议状态，可能因 producer model、账号或前缀变化而 drop/400；持久化目标、授权、工具副作用与 artifacts 仍应由应用自己的 state store 管理。
5. API schema/capability 合同可由官方文档核验；真实 endpoint、fallback 配置、streaming UI、compaction、实际 provider 路由和生产安全效果仍需有授权的集成测试。本轮未调用 Messages API、未运行 benchmark、未下载权重。

## 8. 闭环状态

Claude Sonnet 5.5 的 AA canonical 身份、官方 API/migration/prompting 文档和 System Card 正文已形成 **AA 单榜内容专题闭环 + 官方 API/System Card 证据**。DataCurve 当前无精确 Agent 行；本轮 arXiv 精确查询无结果。参数规模、模型内部架构、完整训练/后训练 recipe、独立技术报告/benchmark 复现、目标硬件 profile、真实 API 行为和生产 acceptance 仍为 `unverified`。不因官方页面中提到的其他模型另建锚点。
