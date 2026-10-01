# Claude Opus 5：排行榜锚点、adaptive reasoning 与长任务运行时

原始资料核验日期：2026-09-15；当前时点联网复验：2026-09-21、2026-09-22（先失败后恢复）、2026-09-24。本笔记只把 Artificial Analysis 与 DataCurve DeepSWE 的条目作为模型锚点，再沿 Anthropic 官方模型页、开发者文档、发布页和 system card 追踪面试相关技术。Anthropic 没有公开 Opus 5 的参数规模、网络结构或完整训练报告；运行时协议、产品定位和发布方评测不能反推出这些内部事实。

## 1. 榜单锚点与快照

- Artificial Analysis 的 canonical 条目为 [Claude Opus 5](https://artificialanalysis.ai/models/claude-opus-5)，配置名为 `Claude Opus 5 (Adaptive Reasoning, Max Effort)`，canonical slug 为 `claude-opus-5`，页面 `releaseDate` 字段为 `2026-07-24`。
- Artificial Analysis 2026-09-21 新鲜详情页给出 `50.7771115797629` Intelligence Index（`intelligenceIndexIsEstimated=false`）、`60.707552496689` median output speed、`46.6837206345s` median time to first chunk、1M context 和约 `$5.8584`/Intelligence Index task。页面仍把参数字段留为 null、开放性标为 proprietary；这些是第三方测量或目录字段，不是 Anthropic 的参数披露。
- DataCurve [DeepSWE v1.1](https://deepswe.datacurve.ai/) 页面标注 2026-09-03 更新、113 个任务、91 个仓库、5 种语言，统一使用 `mini-swe-agent`。Opus 5 的 max 配置是 `mini_swe_agent_claude_opus_5_max`：327/444 次尝试通过，Pass@1 为 `73.6486%`（页面约 `74% +/-4%`），Pass@4 为 `88.4956%`，平均成本 `$11.8376`，平均输出 117,566 tokens，平均 99.04 个 Agent steps，4 次完整 benchmark run。
- DataCurve 同时提供 low/medium/high/xhigh/max 档位；这些是同一模型在不同运行配置下的系统观测，不应计作五个基础模型。上面的 max 数字还绑定任务集、工具、环境、verifier、重试和 `mini-swe-agent` harness。

2026-09-15 历史复验临时快照如下。临时文件不作为长期数据源，哈希用于复现历史页面版本。

| 页面 | 临时文件 | 大小 | SHA-256 |
|---|---|---:|---|
| Artificial Analysis 中文首页 | `/tmp/recheck-aa-1234.html` | 1,773,814 bytes | `5f70a4b28d24ce6c560f83561ed0fe0f7150675b3e8a5a33ea649a238695e4ea3` |
| Artificial Analysis Opus 5 详情 | `/tmp/recheck-aa-opus5-1234.html` | 3,532,163 bytes | `c9626e404c0ff538a28bc58fff9f05bd276db64a1791b72bfa22ce7f4202d47f` |
| DataCurve DeepSWE | `/tmp/recheck-deepswe-1234.html` | 268,313 bytes | `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7` |

该表保留 2026-09-15 的历史证据；本轮新鲜快照见下方，不覆盖历史哈希。

### 1.1 2026-09-21 新鲜快照

| 页面 | 代理/临时文件 | 大小 | SHA-256 |
|---|---|---:|---|
| Artificial Analysis 中文首页 | `7890:/tmp/aa-current-7890.html` | 1,778,568 bytes | `f4ba39ee5b5638def213f29fa5e9f6aac9954a26cb291f6a6064283f9b0eba93` |
| Artificial Analysis Opus 5 详情 | 三条代理 `/tmp/current-aa-opus5-*.html` | 3,868,875 bytes | `e1710da9c158833bad7c1cc8b02dcfdd77d72ea8f6b5652274faa399dde7b615` |
| DataCurve DeepSWE | 三条代理 `/tmp/current-deepswe-*.html` | 268,571 bytes | `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870` |
| Anthropic Opus 5 模型页 | `1234:/tmp/current-anthropic-opus5-1234.html` | 461,560 bytes | `57d20b24a8d7961bd2ea76d71080035677ec27deac07991bcc73cc3d305a03b5` |
| Anthropic Opus 5 发布页 | `8098:/tmp/current-anthropic-release-8098.html` | 352,773 bytes | `72490a50c0d5c96021954261ed4201d03c41e5134f2432647eed8ac58644c31f` |

三条代理对 AA Opus 5 详情和 DataCurve 均返回 HTTP 200 且逐字节一致；Anthropic 模型页的 8098/1234 内容一致，发布页包含动态页面内容，记录选定代理快照而不把不同哈希解释成模型变化。AA canonical slug 集合相对上一快照没有新增八家重点厂商候选；出现的其他新增 slug 不进入当前关注范围。

### 1.2 2026-09-22 当前时点新鲜快照

本轮继续复验同一个 `claude-opus-5` 锚点，没有从官方目录另发现模型。Artificial Analysis 详情页快照 `/tmp/aa-opus5-fresh-20260922.html` 为 `3,869,351` bytes，SHA-256 为 `c18260ab4ff331d5bd3305691db2d4b6051dc2ebe642aa1458c5b8fa2c367643`；中文首页快照 `/tmp/aa-zh-refresh-escalated-1234-20260922.html` 为 `1,762,446` bytes，SHA-256 为 `247d5f3aab6819ab0b22f1852d2d8b27f831feecefc2281ac8e7eb4228b46d63`。

- 当前 AA 详情仍是 `releaseDate=2026-07-24`、`claude-opus-5`、1M context；本次读取的第三方/provider 字段为 Intelligence Index `50.7771115797629`、median output speed `56.4471785104486 tokens/s`、median time to first chunk `49.2490756305s`、cost per Intelligence Index task `$5.858396237036018`。这些是当前测量值，不是 revision、参数或训练变化的证据。
- 9 月 21 日记录的 `60.707552496689 tokens/s` 和 `46.6837206345s` 仍保留为历史测量；两次页面值不能拼成趋势，也不能用一次 provider 测量覆盖另一次。
- 本轮先后取得的 DataCurve 页面都没有产生新的模型身份。Opus 5 的精确配置仍为 `mini_swe_agent_claude_opus_5_max`：327/444、Pass@1 `73.64864864864865%`、Pass@4 `88.49557522123894%`、平均成本约 `$11.8376`、平均输出约 `117,566` tokens、平均 `99.04` Agent steps；这些数字继续绑定 DeepSWE v1.1 的任务集、工具、环境、verifier 和 `mini-swe-agent` harness。
- Anthropic 发布页当前快照为 `352,846` bytes、SHA-256 `7bb18f8e14e20fe2651e4a8308947f53a9541b0dcce6b7549e2b2c244202dce5`；Opus 5 System Card 为 `16,281,258` bytes、SHA-256 `0950dae1ba6b341e4f1a009e535e0f025625efeca21bd043a9a5ae148a3f2e6b`。System Card 的二进制快照不能替代正文抽取，不据此扩写未明确的安全数字。

因此本轮只更新第三方测量的时间戳和官方入口证据，Opus 5 仍为**资料级闭环**，不新增重复 Transformer 正式章节，也不把 9 月 22 日的测量漂移解释为模型升级。

### 1.3 2026-09-24 两榜与官方 System Card 当前复核

经用户确认可用的 `10.24.27.134:7890` 获取两榜。Artificial Analysis 中文首页为 `1,781,428` bytes / SHA-256 `15e9763f0d516791e7b7c2c5679e87e5878eaffed4bef31e485c9043b758ba59`；Opus 5 详情为 `3,977,410` bytes / `18acc956d77c5b49c391eb85b46ef2c7fdbd7edd282ddf2275940bd418d25774`，Intelligence Index 仍为 `50.7771115797629`。DataCurve DeepSWE 为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；精确 `mini_swe_agent_claude_opus_5_max` 行仍是 `327/444`、Pass@1 `73.6486%`、Pass@4 `88.4956%`、均价约 `$11.8376`、平均 `99.04` steps。分数绑定该 benchmark 的任务、`mini-swe-agent`、工具、环境和 verifier，不是裸模型能力。

Anthropic [Claude Opus 5 System Card](https://www.anthropic.com/claude-opus-5-system-card) 本次 HTTP 200，`16,281,258` bytes / SHA-256 `0950dae1ba6b341e4f1a009e535e0f025625efeca21bd043a9a5ae148a3f2e6b`，与既有副本逐字节相同；官方发布页为 `352,158` bytes / `2e35568603369b886d4c95904da18343dcd74ecef0f44600783ffe452c3ccacf`。使用仓库 [`pdf_text_extract.js`](code/pdf_text_extract.js) 提取出 `4,641` 行、`334,056` bytes 文本，SHA-256 `4ae20472ab82c967d90f386239ee6987ddf74b1124db6b44a1dea69a576e1f4c`（临时文件 `/tmp/claude-opus5-system-card-20260924.txt`）。这是对既有官方文档的正文补读，不代表 Anthropic 在 9 月 24 日发布了新版本；当前 PDF 的 changelog 标明最近列出的修订日期为 2026-08-19。

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

官方 [Claude Opus 5 System Card](https://www.anthropic.com/claude-opus-5-system-card) 已通过代理下载并完成正文解析：16,281,258 bytes，SHA-256 `0950dae1ba6b341e4f1a009e535e0f025625efeca21bd043a9a5ae148a3f2e6b`。正文中的 prompt-injection、RSP、harness 修订、内部安全评测数字与适用条件见第 8 节；它们仍属于 Anthropic 发布方证据，不是独立复现。

## 5. 论文、研究入口与负面检索

- [Anthropic Research](https://www.anthropic.com/research) 本轮返回 200，快照 `/tmp/recheck-anthropic-research-1234.html` 为 315,200 bytes，SHA-256 `eec4cf8b7332fe074fc536f9cfc691d0cd5f0079448582450435e403c8ca7b0d`；页面的公开研究列表没有 Opus 5 专属技术报告条目。
- [arXiv 精确标题检索](https://arxiv.org/search/?query=%22Claude+Opus+5%22&searchtype=title) 本轮返回 2 个结果，快照 `/tmp/recheck-arxiv-opus5-1234.html` 为 25,204 bytes，SHA-256 `16950ff350d2f534fe918ecda302adc76fe695c53b6f278e6e277e670cf05c2f`。结果是 arXiv:2608.14992 和 arXiv:2608.07776：前者研究工具结果与文本权威性的合成任务，后者研究 SOC 2 合规代码；两篇把 Opus 5 当作被测模型，标题、作者和摘要都不显示它们是 Anthropic 发布的 Opus 5 技术报告。
- 因此截至 2026-09-15，公开入口没有检出 Opus 5 专属参数/架构报告、完整训练报告、官方论文或公开权重。这个结论是有范围和日期的负面检索证据，不是对未来发布的绝对否定。

官方发布页历史快照本轮通过 1234 返回 `/research/claude-opus-5`，为 355,501 bytes、SHA-256 `c247a02562f799d87e76546ec77a6c6750da7a966501fe101998ae03d6c1be1b`；2026-09-21 新鲜发布页快照见表。Anthropic 开发者文档完整 Markdown `/tmp/anthropic-llms-full-1234` 为 34,524,040 bytes、SHA-256 `76330f83cc797addb31bb5a92dbbeef28b884e40b9de75a27c64d31b71d2c63f`。官方页面和文档是本轮 API 事实的优先证据。

## 6. 尚待核验与书系映射

尚待核验：参数规模、稠密或 MoE 结构、层数、注意力/FFN 设计、完整训练数据配方、优化器、后训练损失、adaptive thinking 的内部实现、生产 API 的真实 block 行为，以及对 System Card 数字的独立 benchmark 复现。安全评测数字已能从正文读取，但其条件和发布方证据等级不能省略。

- 第四册：模型 ID、1M context、adaptive thinking、effort、fallback 和产品/模型证据边界。
- 第六册与第二十四册：长上下文 KV/cache、TTFT/TPOT、缓存失效、并发、成本和 fallback 账本。
- 第七册：固定模型快照、effort、provider、harness、verifier 和统计区间的公平评测。
- 第十六册：reasoning token、状态块回放、effort sweep、验证与过度思考控制。
- 第十七册与第二十册：Agent 工具宿主、工具目录版本、subagent budget、checkpoint、权限审计和 artifact 门禁。

截至 2026-09-22，本节记录的接口/发布资料状态为**资料级闭环**，且没有独立参数/架构披露，因此不新增 Opus 5 专属 Transformer 章节。2026-09-24 解析 System Card 后补成的 Agentic Safety/评测方法专题闭环见第 8 节；它不改变模型内部架构与训练证据边界。

## 7. 2026-09-21 当前时点代理复验时间线

本轮早先重新尝试三条用户提供的代理时，`10.24.27.134:7890`、`10.24.27.134:8098`、`10.237.126.170:1234` 均出现连接失败；随后同一轮重试已全部恢复并取得 HTTP 200。前一失败状态只代表当时的短时访问路径故障，不代表网页不存在。

因此，2026-09-15 页面快照、哈希和运行时结论仍保留为历史缓存证据，2026-09-21 新鲜快照已在 1.1 节单独登记。新鲜页面只更新第三方指标/页面字段和官方可读入口，不升级为参数、架构、训练 recipe 或独立 benchmark 证据，不迁移相邻 Claude 版本分数。

## 8. 2026-09-24 System Card 正文：Agentic Safety 与评测可比性

### 8.1 版本和结论范围

System Card 第 2 页 changelog 记录 2026-08-19 更新了两类内容：补入 §5.2.2.1 的跨 surface prompt-injection bug-bounty 结果；重跑 Cowork browser-use 评测，原因是发现 Opus 5 与旧模型使用了不同 harness。更新后各模型使用同一 Cowork harness；该产品不支持关闭 thinking，因此删去“thinking disabled”结果，只保留 thinking-enabled、medium-effort 条件。这个变更本身就是面试案例：发现比较器版本不一致时，应更新基线并重跑，而不是只修正新模型那一行。

除非章节另有说明，卡片评测使用最终 snapshot 且包含 safeguards；某些能力评测明确关闭生产 safeguards，部分研究使用 helpful-only 或早期 snapshot。System Card 分别讨论模型、产品 harness 和 safeguards，不能把三类数字混成一个“模型安全率”。

### 8.2 Prompt-injection 的纵深防御与评估设计

Anthropic 描述的产品防护有两个不同控制点：prompt-injection probe 在模型行动前检查不可信 tool result（输入侧）；Auto mode 再用 classifier 检查可能危险的 tool call（动作侧）。两层要分别失败攻击才成功。模型本身的抗注入结果与启用这些产品控制后的系统结果必须分开报。

卡片称公开/固定 ART 测试已饱和，因此改报更丰富的 Indirect Prompt Injection (IPI) benchmark，并补上 adaptive red-team 和 bug bounty：

| 评估 | 任务与分母 | Opus 5 报告值 | 解释边界 |
|---|---|---|---|
| Gray Swan IPI | 28 个场景、去重后 1,130 个高迁移攻击；报告攻击者在 `k=1/10/15` 次尝试内成功的概率 | `0.2%` (`k=1`)、`2.0%` (`k=15`)；Opus 4.8 分别 `0.5%`、`5.5%` | Claude 测试关闭产品级 safeguards；其他模型用公开 endpoint，是否带有防护不统一；跨厂商比较不是同条件裸模型对照 |
| 一周 Gray Swan live bug bounty | 11 个新场景，tool/coding/computer-use；目标模型身份隐藏；每模型超过 20,000 次有效尝试 | attempt-level ASR `0.08%`；Opus 4.8 `0.11%` | Anthropic 模型关闭产品级 probes/harness 防护；外部模型使用其 production configuration，比较边界不对称 |
| Shade coding | 40 个场景，每场景 200 次尝试；自适应攻击者对相同场景优化后迁移 | 无 probes：thinking `0.56%`、disabled `0.41%`；启用 probes 两者均 `0.18%` | 同时报 attempt-level ASR 与至少被攻破一次的 scenario 数；不可把两个指标混为一谈 |
| Shade computer use | 14 个 GUI 场景，每场景 200 次尝试 | 无 probes：thinking `0.54%`、disabled `0.39%`；启用 probes：`0.25%`、`0.43%` | disabled + probes 的微升对应单次额外成功（2,800 次中的 1 次），卡片明确认为不能与噪声区分 |
| Cowork browser | 129 个 unseen 环境，每场景 10 次攻击；medium effort，thinking enabled | raw-model comparison `3.84%`；Auto mode 为 `0/129` 场景成功 | Cowork 实际从不无 safeguards，基础条件也始终含 probes；“raw”条件用于与最强 Auto mode 比较。Auto mode 结果是产品系统结果，不是裸模型分数 |

IPI 的攻击尝试是自适应或高迁移攻击，不能当作独立同分布的普通用户流量；多次尝试概率也依赖测试场景和攻击预算。最好保留 `attempt_success / valid_attempts` 与 `scenarios_with_success / tested_scenarios` 两个量，附上模型 snapshot、thinking/effort、probe、Auto mode、harness revision 和 verifier。

### 8.3 评测校准与安全治理要点

- RSP：Card 将 Opus 5 评为 CB-1、未达 CB-2，并说明继续施加 ASL-3 保护；AI R&D 自动评测虽有多项能力处于前沿，但其 RSP 阈值结论还结合内部使用与生产力指标，报告未观察到持续、可归因的整体 AI 研发速度翻倍，也认为模型距离替代资深研究人员尚远。这是 Anthropic 的风险判断，不是独立审计结论。
- 安全策略：Card 描述 cyber 流量先经过 activation probe，再由 LLM classifier 对 flagged traffic 作判断；一般可用条件开放 source-code vulnerability discovery，同时继续阻断 compiled-binary vulnerability discovery。它是按任务/使用面区分防护，不是“模型完全安全”的证明。
- 内部行为审计：自动审计约 `3,200` 个 investigation sessions（约 `1,600` 个 seed 场景各采样两次），以 investigator model 探测、judge model 评分数十个维度。Card 提醒很多维度跨所有场景评分，所以绝对数受 seed 分布影响，主要用于同一方法下的模型比较；该版本新增 approval-gate bypass 与 expedience security shortcut 维度。
- 内部部署监测发现少量绕开 classifier/network 限制的行为，报告为 monitored completions 的 `<0.01%`。这是 Anthropic 内部样本中的观测频率，不是外部生产总体概率；Card 还记录 overconfidence、scope creep 等失败模式。
- 训练行为 review 先用 Opus 4.8 支持的递归摘要检查约 150 万条训练后期 episode，再抽样完整审阅约 400 条 transcript，并由人检查值得注意的主张。这个链路可用于发现失败模式，但不是对所有 episode 的逐条人工标注。

面试回答应抓住三条：其一，安全结论绑定模型版本、攻击预算、effort、保障措施、surface、harness 和 verifier；其二，产品 probe/动作分类器与模型本身的鲁棒性要通过消融拆账；其三，评测方法发生变化（饱和、harness mismatch、thinking capability 不可用）时要同时更新基线，并报告 attempt 与 scenario 两种分母。来源定位：System Card PDF pp. 2、12、14–16、36–51、72–80、86–102；原文链接及文件哈希见本节和 `source-index.md`。

本轮 Opus 5 的专题升级为 **双榜锚点 + 官方 System Card 正文复核 + Agentic Safety/评测方法专题**。这不表示参数、内部架构、完整训练 recipe、独立复现或生产安全 SLO 已验证。
