# Claude Sonnet 4.6 官方资料摘记

核验日期：2026-09-16

## 1. 两个排行榜锚点

Artificial Analysis 有 Claude Sonnet 4.6 的多个配置：[`claude-sonnet-4-6-adaptive`](https://artificialanalysis.ai/models/claude-sonnet-4-6-adaptive) 表示 Adaptive Reasoning/Max Effort，另有 [`claude-sonnet-4-6`](https://artificialanalysis.ai/models/claude-sonnet-4-6) 等非 adaptive/effort 条目。DataCurve DeepSWE v1.1 有 `mini_swe_agent_claude_sonnet_4_6_high`。

2026-09-16 快照：

- AA adaptive 页面 `/tmp/sonnet46-aa-adaptive-20260916.out`：`3,626,831` bytes，SHA-256 `5feda773b5ccefaaad7265b5f2d2401551d4d96e32cfa524a7fedd550db5b0de`。页面 FAQ 将 max 配置描述为 1M context、约 `49 tokens/s`，并给出 `$3/$15` 每百万 input/output token；这些是第三方配置/provider 测量字段。
- DataCurve 页面 `/tmp/overnight-recheck-deepswe-1234-20260916.out`（与 8098/7890 快照逐字节一致）：`268,313` bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。`mini_swe_agent_claude_sonnet_4_6_high` 为 135/451，Pass@1 `29.9335%`，Pass@4 `56.6372%`，平均成本约 `$5.5224`，平均输出 `76,160.31` token，平均 Agent steps `133.66`。

DeepSWE 的结果绑定 `mini-swe-agent`、4 runs、113 tasks、91 repositories、5 languages、工具、环境和 verifier；不能与 AA 指数或 Anthropic 发布方评测拼成基础模型能力。AA 的 adaptive、high 和其他行也应按推理/服务配置归并，不计为多个基础模型。

## 2. Anthropic 官方发布证据

Anthropic [Introducing Claude Sonnet 4.6](https://www.anthropic.com/news/claude-sonnet-4-6) 页面确认发布时间为 2026-02-17，API 模型 ID 为 `claude-sonnet-4-6`。官方定位覆盖 coding、computer use、long-context reasoning、agent planning、knowledge work 和 design；1M token context window 在发布时为 beta。页面还说明它是 Sonnet 4.5 的全面升级，Free/Pro Claude.ai 默认切换到该模型，API 价格保持 Sonnet 4.5 的价格层级。

官方发布页明确列出以下运行时能力：

- Claude Platform 支持 adaptive thinking 和 extended thinking；
- context compaction 处于 beta，会在接近上下文限制时总结较早内容，以延长有效任务长度；
- web search、web fetch、code execution、memory、programmatic tool calling、tool search 和 tool-use examples 可用于 Agent 工作流；
- computer use 通过屏幕观察和点击/输入等通用动作操作没有专用 API 的软件，但网页中的恶意指令会形成 prompt injection 风险；官方称 Sonnet 4.6 的相关安全评测较 Sonnet 4.5 改善，并接近 Opus 4.6。

官方 [System Card](https://www.anthropic.com/claude-sonnet-4-6-system-card) 页面本轮返回 HTTP 200，但正文以 Google Docs 嵌入形式提供；本笔记只记录发布页可直接核验的安全摘要，不从无法稳定抽取的嵌入文档扩写具体安全数字。

## 3. 面试技术主线

### 3.1 1M context 的有效容量

把整个代码库、合同或论文集合塞入窗口只解决容量问题，不保证检索和推理质量。面试中要区分 raw context window、可用 token 预算、思考 token、工具结果、压缩摘要和长期状态。Sonnet 4.6 的官方表述强调它能在大上下文上进行 long-horizon planning；这属于发布方行为描述，不等于每个位置均匀可检索或数学意义上的无损记忆。

### 3.2 Adaptive thinking 与 effort sweep

adaptive thinking 和 extended thinking 是 API/产品层的推理控制能力。实际系统应将 effort 当作可测的延迟—成本—成功率旋钮，在 low/high 或 adaptive 配置间做任务分层，而不是把每个 effort 当成新模型。比较时必须固定 `max_tokens`、工具、上下文、超时和 verifier；否则无法判断提升来自模型、推理预算还是 Agent harness。

### 3.3 Context compaction 的状态协议

Compaction 的工程目标是把早期对话压缩成后续仍可用的状态，同时维持任务目标、文件变更、约束、未完成事项和验证结果。风险包括摘要遗漏、错误状态固化、工具调用边界丢失和重复执行。面试回答应说明触发阈值、摘要 schema、原始 artifact 保留、恢复后的 sanity check 和幂等性设计。Sonnet 4.6 官方只确认 context compaction beta 能力，不公开内部摘要算法。

### 3.4 Computer use 与 prompt-injection 防御

电脑操作把模型输出变成真实环境动作：观察屏幕、选择坐标、点击/输入、读取结果，再继续规划。环境中的网页文本可能携带不可信指令，因此宿主应隔离权限、限制域名和动作、对外发消息/购买/删除等高风险操作要求人工确认，并把模型建议与执行器权限分开。Anthropic 发布页的安全表述支持这一风险模型，但不等于 Sonnet 4.6 已解决 prompt injection。

### 3.5 工具搜索和上下文预算

tool search、tool-use examples、web search、code execution、memory 和 programmatic tool calling 共同说明：Agent 系统的瓶颈不只是模型推理，还包括工具 schema 选择、工具结果过滤、上下文预算和执行权限。工具应按任务延迟加载或搜索，返回结构化且可验证的结果，避免把全部工具定义和噪声输出塞入每一轮上下文。

## 4. 评测证据边界

Anthropic 发布页公开了 computer use、coding、OfficeQA、SWE-bench、Terminal-Bench、HLE、BrowseComp 等比较和设置说明；部分任务使用工具、context compaction、max effort、adaptive thinking、不同 harness 或不同时间限制。它们属于发布方测量，应记录模型版本、effort、工具、harness、数据集版本、预算、重复次数和是否使用压缩。

DataCurve 的 135/451 与 AA 的页面字段属于不同评测协议；不能使用 Anthropic 发布方 benchmark 去填补 DataCurve 缺失字段，也不能把 DataCurve 的 `high` 分数写成 Sonnet 4.6 的通用 SWE-Bench 能力。

## 5. 负面证据与待核验

- 本轮没有发现 Claude Sonnet 4.6 的独立架构、参数规模、完整预训练/后训练 recipe 或可复现技术报告；System Card 页面可访问，但具体正文未稳定抽取。
- 官方平台文档的多个 URL 在当前环境跳转到区域不可用页；因此不把无法读取的 API 字段写成已确认事实，先以发布页和已有模型目录记录为准。
- 仍待核验 adaptive thinking 的内部预算策略、compaction 摘要格式、tool search 的实际选择质量、computer-use 线上接受率、目标硬件 profiling 和外部独立 benchmark。

## 7. 2026-09-16 夜间中断复验

- 三条代理重新获取 [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh) 和 [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 均返回 HTTP 200；Artificial Analysis 快照均为 `1,773,553` bytes、SHA-256 `60e491dc5eccaf2d032205a7380f30a24b2726583b5a9cfaccfe14c17d736476`，DataCurve 均为 `268,313` bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- 三条代理重新获取 [Sonnet 4.6 发布页](https://www.anthropic.com/news/claude-sonnet-4-6) 均返回 HTTP 200、`281,283` bytes；页面内容因动态响应哈希不同，但核心发布信息一致。`8098` 与 `1234` 访问模型目录被重定向到 `claude.com/app-unavailable-in-region`（`438,271` bytes），不能作为模型目录证据；`7890` 成功取得真实 [Anthropic Models Overview](https://platform.claude.com/docs/en/models/overview)，`728,089` bytes，SHA-256 `4c5ba69551075bbff00251943a1765c77be38f998d512fac67902067097cdfff`，页面包含 `Claude Sonnet 4.6`、`claude-sonnet-4-6`、1M context 和 128K output 字段。
- 结论：夜间保存的排行榜响应没有发现截断或代理间不一致；官方目录访问存在线路差异，已用可读取的 `7890` 线路补齐。三条线路的失败/区域重定向不解释为模型不存在。

## 6. 当前结论

Claude Sonnet 4.6 当前为**资料级闭环**：Artificial Analysis 与 DataCurve 的配置锚点、Anthropic 官方发布页、System Card 入口和面试技术主线均已具备；暂无独立正式架构章节。重点是 1M context 的有效容量、adaptive/extended thinking、compaction 状态协议、computer use 的执行器安全、tool search 与 Agent 上下文预算。内部架构、参数、训练 recipe 和独立复现仍待核验。
