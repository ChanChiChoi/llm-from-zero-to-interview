# Claude Opus 5：排行榜锚点、adaptive reasoning 与长任务运行时

核验日期：2026-09-15。本笔记只把 Artificial Analysis 与 DataCurve DeepSWE 的条目作为模型锚点，再沿 Anthropic 官方模型页、开发者文档、发布页和 system card 追踪面试相关技术。Anthropic 没有公开 Opus 5 的参数规模、网络结构或完整训练报告；运行时协议、产品定位和发布方评测不能反推出这些内部事实。

## 1. 榜单锚点与快照

- Artificial Analysis 的 canonical 条目为 [Claude Opus 5](https://artificialanalysis.ai/models/claude-opus-5)，配置名为 `Claude Opus 5 (Adaptive Reasoning, Max Effort)`，canonical slug 为 `claude-opus-5`，页面 `releaseDate` 字段为 `2026-07-24`。
- Artificial Analysis 当前详情页给出约 `50.7002` Intelligence Index、`50.0725` output tokens/s、`46.5045s` median time to first chunk、1M context 和约 `$5.8584`/Intelligence Index task。页面还把参数字段留为 null、开放性标为 proprietary；这些是第三方测量或目录字段，不是 Anthropic 的参数披露。
- DataCurve [DeepSWE v1.1](https://deepswe.datacurve.ai/) 页面标注 2026-09-03 更新、113 个任务、91 个仓库、5 种语言，统一使用 `mini-swe-agent`。Opus 5 的 max 配置是 `mini_swe_agent_claude_opus_5_max`：327/444 次尝试通过，Pass@1 为 `73.6486%`（页面约 `74% +/-4%`），Pass@4 为 `88.4956%`，平均成本 `$11.8376`，平均输出 117,566 tokens，平均 99.04 个 Agent steps，4 次完整 benchmark run。
- DataCurve 同时提供 low/medium/high/xhigh/max 档位；这些是同一模型在不同运行配置下的系统观测，不应计作五个基础模型。上面的 max 数字还绑定任务集、工具、环境、verifier、重试和 `mini-swe-agent` harness。

本轮复验临时快照如下。临时文件不作为长期数据源，哈希用于复现本轮页面版本和区分代理传输问题。

| 页面 | 临时文件 | 大小 | SHA-256 |
|---|---|---:|---|
| Artificial Analysis 中文首页 | `/tmp/recheck-aa-1234.html` | 1,773,814 bytes | `5f70a4b28d24ce6c560f83561ed0fe0f7150675b3e8a5a33ea649a238695e4ea3` |
| Artificial Analysis Opus 5 详情 | `/tmp/recheck-aa-opus5-1234.html` | 3,532,163 bytes | `c9626e404c0ff538a28bc58fff9f05bd276db64a1791b72bfa22ce7f4202d47f` |
| DataCurve DeepSWE | `/tmp/recheck-deepswe-1234.html` | 268,313 bytes | `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7` |

`10.237.126.170:1234` 和 `10.24.27.134:8098` 对三个目标页面均返回 HTTP 200；`10.24.27.134:7890` 对 DataCurve 返回 200，但 Artificial Analysis 大页面在超时前只收到部分内容。代理超时不解释为页面不存在。

## 2. 官方模型字段

Anthropic 官方 [模型总览](https://platform.claude.com/docs/en/models/overview)、[Opus 5 专属页](https://platform.claude.com/docs/en/models/opus-5/overview) 和官方完整 Markdown 文档确认：

- 模型 ID 为 `claude-opus-5`，生命周期为 active，页面发布日期为 `2026-07-24`；产品定位是复杂 agentic coding 和 enterprise work。
- 上下文窗口为 1,000,000 tokens，普通同步最大输出为 128,000 tokens；Batch 最大输出字段为 300,000 tokens。1M context 对 Opus 5 默认生效，不需要旧的 `context-1m-2025-08-07` beta header。
- 支持文本/图像输入、文本输出、视觉和工具使用；可用平台包括 Claude API、Amazon Bedrock、Google Cloud Vertex AI 和 Microsoft Foundry。
- 价格字段为每百万输入 token `$5`、每百万输出 token `$25`；缓存写入/读取、平台、区域和账户折扣必须按实时价格页重新核对。Priority Tier 不支持；Fast mode 是 API research preview，价格为 `$10/$50` 每百万输入/输出 token。
- 页面给出的可靠知识/训练数据截止为 `2026-05`。这两个字段不等于实时搜索能力，也不说明模型可以访问外部网页。
- 默认 thinking 为 adaptive，默认 effort 为 `high`；可选 `low`、`medium`、`high`、`xhigh`、`max`。`output_config.effort` 控制 thinking volume，不等于 visible response length。
- 最小可缓存 prompt 从 Opus 4.8 的 1,024 个 token 降为 512 个 token。该字段是缓存协议约束，不是模型上下文或参数规模。
- 文档注明 web fetch 不支持；需要联网的 Agent 必须由宿主提供搜索、抓取、代码或其他工具，并自行负责权限、超时、审计和结果验证。

## 3. API 与长任务面试主线

### 3.1 Adaptive thinking 是响应协议的一部分

Opus 5 相对 Opus 4.8 的重要接口变化是 thinking 默认开启。响应的 `content` 是异构 block 序列，thinking block 可能出现在 text block 之前，因此客户端必须按 `content[].type` 解析，不能假设 `content[0].text` 一定存在。文档还说明 `thinking.display` 默认是 `omitted`：响应可能返回空的 `thinking` 字段和签名，而不是可直接展示的原始思维链。

当 Agent 继续工具循环时，应用必须完整、原样回传 thinking blocks 及其签名；丢失、改写、重排或把它们当普通文本拼接，都可能导致 API 返回 400。面试中应把它解释成“模型状态/协议块的回放约束”，而不是把签名当作可读的思维链。

`thinking: {"type": "disabled"}` 只允许配合 `high` 或更低 effort；`xhigh`/`max` 与 disabled 组合会被拒绝。关闭 thinking 还可能使工具调用泄漏为普通文本或出现内部 XML 标签，因此生产客户端应验证 block 类型、tool schema 和 stop reason，而不是只检查最终字符串。

### 3.2 中途改工具和 effort

- `mid-conversation-tool-changes-2026-07-01` beta 支持在对话中途增加、删除或修改工具；工具目录因此成为会话状态的一部分，宿主仍要记录版本、权限和副作用。
- `mid-conversation-output-config-2026-07-01` beta 支持逐消息切换 effort，同时保持 prompt cache。它适合把规划阶段设为 `max`、执行或汇总阶段降为 `medium`，但必须在评测账本中记录每一轮的 effort、thinking token、工具轮次和缓存命中。
- effort 是全响应的行为旋钮，可能影响 thinking、可见答案和工具调用；`max_tokens` 仍是每个请求的硬输出上限。不能用 `max_tokens` 代替 effort，也不能把榜单的 effort 档位写成不同模型版本。

### 3.3 Refusal、fallback 与成本

Opus 5 的拒答是正常 API 响应中的业务状态，可能带 `stop_reason: "refusal"` 和 `stop_details.category`，不应只按 HTTP 错误处理。官方文档支持 `fallbacks: "default"` 与 `server-side-fallback-2026-07-01` beta header，让服务端根据 refusal category 选择默认 fallback；也提供 fallback credit，减少跨模型重试时重新写入 prompt cache 的成本。

fallback 会改变实际服务模型、缓存账本、工具能力和安全边界。评测必须记录原始模型、fallback 目标、拒答类别、是否真的重试、重试成本和最终 artifact，不能把 fallback 的成功率归因给 Opus 5 单体。

### 3.4 Prompt 与 Agent 行为

Anthropic 的 Opus 5 prompting 指引称该模型更倾向自我验证、迭代修正、产生较长的可见回答并主动启动 subagents。官方建议移除旧 prompt 中强制重复自检的指令，以免 over-verification；对 delegation 应显式限制深度、并发数和预算。这些是官方行为指导，不是公开的训练算法说明。

可迁移为面试设计题的控制面是：

1. 为规划、工具执行、验证和汇总分别设定 effort 与 token budget；
2. 对 subagent 设置最大深度、并发、总成本和取消条件；
3. 对每个工具调用保存 schema 版本、权限、输入、输出、错误和副作用状态；
4. 用测试、lint、类型检查、领域 verifier 或可下载 artifact 判断任务是否完成，而不是相信模型的完成声明；
5. 将 compaction、prompt cache、fallback 和 checkpoint 一起纳入长任务账本。

## 4. 发布方评测与安全边界

Anthropic 发布页声称 Opus 5 在 Frontier-Bench v0.1 超过其他模型且超过 Opus 4.8 两倍、在 CursorBench 3.2 的 max effort 下距离 Fable 5 峰值不到 0.5% 且成本约一半、ARC-AGI 3 约为次优模型三倍，并在 Zapier AutomationBench 与 OSWorld 2.0 上给出同成本或成本曲线优势。发布页还报告内部有机化学 benchmark 比 Opus 4.8 高 10.2 个百分点、蛋白质序列功能预测高 7.7 个百分点。

这些数字必须标成 Anthropic 发布方数据。Frontier-Bench 脚注明确其为内部运行、`mini-SWE-agent` harness、GKE backend、每任务 5 次尝试；Opus 5/Fable 5 的安全拒答会 fallback 到 Opus 4.8。它们不是独立复现，也不能和 Artificial Analysis 或 DeepSWE 的数字拼成一个裸模型排名。

安全边界方面，公告称 Opus 5 没有推进 risky dual-use capability frontier，且没有针对 cyber task 专门训练；它在漏洞发现方面接近 Mythos 5，但 exploit generation 明显落后。Claude.ai、Claude Code 和 Claude Cowork 的部分拒答默认 fallback 到 Opus 4.8，API 也可以开启 fallback；这进一步说明产品安全结果与基础模型结果必须分开记录。

官方 [Claude Opus 5 System Card](https://www.anthropic.com/claude-opus-5-system-card) 已通过代理下载：16,281,258 bytes，SHA-256 `0950dae1ba6b341e4f1a009e535e0f025625efeca21bd043a9a5ae148a3f2e6b`。当前环境没有稳定的 PDF 正文抽取器，不能仅凭目录、字体编码或二进制字符串扩写安全数字；本笔记只记录公告明确的结论。

## 5. 论文、研究入口与负面检索

- [Anthropic Research](https://www.anthropic.com/research) 本轮返回 200，快照 `/tmp/recheck-anthropic-research-1234.html` 为 315,200 bytes，SHA-256 `eec4cf8b7332fe074fc536f9cfc691d0cd5f0079448582450435e403c8ca7b0d`；页面的公开研究列表没有 Opus 5 专属技术报告条目。
- [arXiv 精确标题检索](https://arxiv.org/search/?query=%22Claude+Opus+5%22&searchtype=title) 本轮返回 2 个结果，快照 `/tmp/recheck-arxiv-opus5-1234.html` 为 25,204 bytes，SHA-256 `16950ff350d2f534fe918ecda302adc76fe695c53b6f278e6e277e670cf05c2f`。结果是 arXiv:2608.14992 和 arXiv:2608.07776：前者研究工具结果与文本权威性的合成任务，后者研究 SOC 2 合规代码；两篇把 Opus 5 当作被测模型，标题、作者和摘要都不显示它们是 Anthropic 发布的 Opus 5 技术报告。
- 因此截至 2026-09-15，公开入口没有检出 Opus 5 专属参数/架构报告、完整训练报告、官方论文或公开权重。这个结论是有范围和日期的负面检索证据，不是对未来发布的绝对否定。

官方发布页本轮通过 1234 返回 `/research/claude-opus-5`，快照为 355,501 bytes、SHA-256 `c247a02562f799d87e76546ec77a6c6750da7a966501fe101998ae03d6c1be1b`；Anthropic 开发者文档完整 Markdown `/tmp/anthropic-llms-full-1234` 为 34,524,040 bytes、SHA-256 `76330f83cc797addb31bb5a92dbbeef28b884e40b9de75a27c64d31b71d2c63f`。官方页面和文档是本轮 API 事实的优先证据。

## 6. 尚待核验与书系映射

尚待核验：参数规模、稠密或 MoE 结构、层数、注意力/FFN 设计、训练数据配方、优化器、后训练损失、adaptive thinking 的内部实现、完整 safety evaluation 数字、生产 API 的真实 block 行为和独立 benchmark 复现。

- 第四册：模型 ID、1M context、adaptive thinking、effort、fallback 和产品/模型证据边界。
- 第六册与第二十四册：长上下文 KV/cache、TTFT/TPOT、缓存失效、并发、成本和 fallback 账本。
- 第七册：固定模型快照、effort、provider、harness、verifier 和统计区间的公平评测。
- 第十六册：reasoning token、状态块回放、effort sweep、验证与过度思考控制。
- 第十七册与第二十册：Agent 工具宿主、工具目录版本、subagent budget、checkpoint、权限审计和 artifact 门禁。

当前状态：**资料级闭环**。已有两个排行榜锚点、官方模型/API/发布/system card 入口、研究笔记、负面论文检索和同步记录；没有独立参数/架构专题，因此不新增 Opus 5 专属正式章节。
