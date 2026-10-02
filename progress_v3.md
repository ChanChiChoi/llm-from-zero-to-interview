# 新模型与新知识点更新进度 v3

本文件保留以排行榜新进模型为锚点，基于官方模型资料、论文、技术文档和权威博客更新周边知识的完整执行记录。

## 当前状态

- 阶段：模型候选核验、专题写作与纵向同步（持续进行）
- 主线：新模型发现 → 官方资料核验 → 技术知识映射 → 书系同步 → 代码/链接/证据审计
- 规划文件：[`plan.md`](plan.md)
- 第一次计划进度：`progress.md`，仅作首轮全书编写历史记录
- 当前执行项：2026-10-01 排行榜重点模型资料级闭环已收口；后续只有在用户手动发起新一轮更新时，才从两个排行榜选择新锚点并继续核验官方资料、论文、技术文档和权威博客。当前没有独立运行中的模型更新任务。
- 最近完成的独立专题：Qwen3.6-35B-A3B 官方博客证据恢复。7890 首次请求失败，按权限流程的获批沙箱外重试返回 HTTP 200；文章 HTML 91,941 bytes / SHA-256 `706889d145ea17b8c8234c4cda35b00fdecc0b6bcb9e1f5f20d2ed3ff9e15ed1`。新增知识重点为公开分数的 evaluator/user model/toolchain 依赖，而非新的模型架构；`qwen3.6-flash` 记录为同一榜单锚点的 API alias。OpenClaw 128K/16K 是客户端预算，百炼 endpoint 未 probe。

## 已完成

- 2026-09-24 Kimi K2.7 Code 内容专题闭环：用户终端实测证明 7890 可访问百度，但本轮 agent 沙箱内连接失败，获准沙箱外重试后仍超时；通过备用 1234 刷新两榜和 Kimi 官方资源/API Markdown。AA K2.7 页面 4,047,429 bytes / 1a8818a6ee812da88ba7b9a95b691ae7481dac7b55d7126084fbd1d5e964e56b，Index 25.8121062401836；DataCurve 268,036 bytes / 14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1，精确行 113 tasks/4 runs、Pass@1 30.53%、Pass@4 61.06%。官方 Markdown 快照补充 max_tokens 默认 32,768、强制 thinking、fixed sampling、媒体 token estimate、视频抽帧/文件上传边界，并说明 Highspeed 与 K2.7 Code 是同一模型。固定 HF revision 代理超时，未伪称刷新成功。新增第二十一册第 94 章及 kimi_k27_contract_budget_toy.py，脚本通过，证据级别 local_protocol_toy；同步 source notes、inventory/interpretation、source index、书目、题库、练习、项目、知识图谱、plan 和 progress。专属训练报告、完整权重、真实 endpoint、GPU profile、独立 benchmark 和生产验收仍未确认，goal 保持 active。
- 2026-09-24 当前会话代理复测与 QA：用户 shell 的 7890 百度成功结果已收到；本 agent 会话经沙箱内及沙箱外对同一代理均在 10 秒连接超时，不能据此判定代理全局不可用。备用 1234 返回百度 HTTP 200，并取得 Artificial Analysis `/zh`（1,781,428 bytes，SHA-256 `15e9763f0d516791e7b7c2c5679e87e5878eaffed4bef31e485c9043b758ba59`）与 DataCurve（268,036 bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`）；与已有同日快照一致，未发现新 canonical 模型。全库 718 个 Markdown 文件围栏检查通过，代码/数学片段外 1,530 个本地链接缺失 0；16 个研究 Python 文件 AST 检查、`git diff --check` 均通过。

- 2026-09-24 Qwen3.8-Flash-Next serving/runtime 补证：通过备用 `10.237.126.170:1234` 复取 Artificial Analysis `/zh`（HTTP 200，1,781,428 bytes，SHA-256 `15e9763f0d516791e7b7c2c5679e87e5878eaffed4bef31e485c9043b758ba59`）和规范 DataCurve DeepSWE（HTTP 200，268,036 bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`）；均与同日已记录快照一致，未见重点厂商新 canonical 模型。7890 在用户终端可访问，但从当前 agent 网络路径（沙箱内及沙箱外重试）均连接超时；不能据此说代理全局不可用。1234 成功读取 GitHub API 和 Qwen 固定 README。已校正 QSA stream overlap、BF16/FP8 dtype 边界及 47.7 GiB/H100 信息的来源归属，并同步研究笔记、第 83 章、source index、inventory、PAPERS、题库、练习、项目、知识图谱、计划和进度。固定 vLLM/SGLang release 源码只做静态核对；未运行测试、安装 wheel、加载完整权重或做 GPU profiling，硬件/生产验收仍待核验。goal 保持 `active`。

- 2026-09-24 DeepSeek V4 Flash SGLang serving 补证：沿两榜已有锚点核验 SGLang `v0.5.20` 与 PR #34565/#30805/#29927/#39171；补充混合 Full/Compressed KV 与 SWA 分支点缓存、B200 CSA/HCA kernel、SM120 sparse-indexer/FP4 MoE 路径及 benchmark 归因边界。已同步研究笔记、source index、inventory/interpretation、第二十一册第 77 章、PAPERS、题库、练习、知识图谱、计划与进度；不新增重复章节。发布方测量、完整权重、目标硬件独立复现和生产 SLO 均未越级表述。

- 2026-09-24 GLM-5.3-Flash vLLM stable source-tree 补证：第一阶段确认 v0.30.0 目录项；随后固定 stable tag 与 main commit 做逐文件/hash/diff 审计，结论已由本进度末尾的最新记录细化。未把 main 独有的 Quark loading、sparse-indexer 和 stride-aware kernel 细节写成 stable；wheel/权重/硬件/SLO 均未声称通过。

- 2026-09-24 Claude Fable 5.1 thinking-state 专题：双榜与官方文档复验后，新增第二十册第 21 章 21.29；补入 Opus 5.5 单向/Claude API 专属兼容矩阵，区分 model-binding 和 prefix-binding，扩展本地协议 toy，并同步来源索引、盘点、题库、练习、知识图谱、计划与进度。内容专题闭环不代表真实 API、架构、训练、硬件或生产验收。

- 2026-09-24 7890 当前工作区复验：当前环境通过 `10.24.27.134:7890` 取得 Artificial Analysis `/zh`（HTTP 200，`1,782,611` bytes，SHA-256 `566b4adab724bd312436a302be3f0f4c5d9f7713188e9efb078cb636faddd02b`）和 DataCurve DeepSWE（HTTP 200，`268,036` bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`）；与已有同日快照一致，八家重点厂商没有新增 canonical 模型。该次质量门禁：`git diff --check`、研究代码 Python 编译、全库 Markdown 围栏检查通过；相对 Markdown 链接检查 1,495 个、缺失 0。随后已选择 DeepSeek V4 Flash serving 补证作为当前活动锚点，goal 保持 `active`。

- 2026-09-24 Kimi K3 runtime 更新：固定 vLLM v0.30.0 release commit、PyPI wheel metadata、registry 与 K3 NVIDIA/DSpark source；对照 v0.29.0 确认 stable entry 原已存在，新增记录只表述实现演进。已更新 `kimi-k3-source-notes.md`、第二十一册第 88 章、source index、model inventory、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`KNOWLEDGE_GRAPH.md`、`plan.md` 与本进度。wheel 未下载/安装，目标硬件与端到端 serving 未验收。
- 2026-09-24 后续联网与专题推进：沙箱内访问 7890 代理失败；经批准在沙箱外通过同一代理抓取榜单主机成功。Artificial Analysis `/zh` 为 HTTP 200、`1,784,793` bytes、SHA-256 `a374adfb81fea4fc68e7071ef191612d7f7d92c1c0e05412f3c338e7fe9edbc2`；规范主机 DataCurve `https://deepswe.datacurve.ai/` 为 HTTP 200、`268,036` bytes、SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`。当前 AA 重点厂商 slug 与 DataCurve 配置集未发现新 canonical 模型；AA 同日页面较早记录相差 12 bytes，按动态页面快照变化处理，不解释为模型 revision。
- 按计划选取已有锚点 Gemini 3.8 Flash，将 9 月 23 日已核验的 Google 官方 Interactions/Thinking/Tool combination/Thought signatures 资料落为[第二十册第 23 章](book-20-agent-harness-runtime/chapters/23-gemini-interactions与thought-state回放.md)。章节覆盖 stateful/stateless、interaction-scoped 配置、opaque signature 文档差异、SSE/工具幂等、保留删除、预算与 capability probe；研究笔记、模型盘点、榜单解释、来源索引、计划和本进度，以及目录、题库、练习、项目和知识图谱均已同步。状态升为“内容专题闭环”；内部架构/训练、真实 API probe、目标硬件和线上验收仍待核验。

- 2026-09-24 当前时点复验：通过 `10.24.27.134:7890` 获取 Artificial Analysis 中文首页（1,784,805 bytes / `5284847a4499b219872725221324a3461de68464956a4b8f44b5cdd96d34c2c8`）、DataCurve（268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`）、AA Claude Opus 5.5 与 GPT-6 Luna 详情，以及 Anthropic Opus 5.5 官方 model page。比较 9 月 23 日快照，八家厂商 AA canonical 集合和 DataCurve 配置集合无新增；Opus 5.5 Index/cost 与旧值相同，Luna Index/cost 相同、速度作为 provider 时点字段记录。Opus 5.5 官方服务合同补齐同步/Batch 输出限制、cache minimum 与缓存读写价格；研究笔记、来源索引、盘点、解释、计划和推理/Agent 正式章节已同步。DataCurve 无精确 Opus 5.5 Agent 行，不迁移其他 Claude 结果。
- 2026-09-24 Opus 5.5 治理证据审计：用户给出的 7890 代理请求可达；本轮对 Artificial Analysis 首页、Opus 详情、DataCurve 与 Anthropic 官方模型页均取得 HTTP 200。最新 AA Opus 详情 `3,809,722` bytes / `683005dd64dba034487fa6207167ce8239161e83cf198cd03d2a9445204caf23`，Index `57.6223698102963`、cost/task `$5.982012019521066` 未变；首页 `1,783,893` bytes / `3b895865547acf9b8d9d67d0c2a9a0a0369648f925014b2367817e1f5ac42f36`，DataCurve 和官方 model page 哈希与既有记录一致。System Card PDF 固定副本 `17,795,106` bytes / `7311c9c6bbb16d012f1c12c7418b05949fcf7ae3e30d2c40f22050074b2a7378` 校验通过。研究笔记和第二十册已包含上述评测条件；本轮仅在第八册第 11 章补充治理视角的证据账本，不重复 benchmark 数字。状态为 **AA 单榜内容专题闭环 + System Card/API contract + local protocol toy**；DataCurve 精确行、独立复现、参数/架构、完整 recipe、目标硬件和生产 acceptance 仍未核验。

- 上一活动锚点收口（2026-09-22）：已由 Artificial Analysis 发现的 `Qwen3-VL-235B-A22B` 完成专题写作和同步。三条代理取得 AA instruct/reasoning 详情页逐字节一致快照，DataCurve 没有精确 Agent 行；本轮完成 HF 固定 revision/config、Qwen3-VL Technical Report、三模块结构、Interleaved-MRoPE、DeepStack、Video Timestamp、S0-S3 curriculum、SAPO、Thinking with Images 和 tool-call reward 证据核验，并新增第二十一册第 92 章；K2 Horizon 3.7B、Qwen3-Omni、Grok 4.7 和 Claude Sonnet 5 作为前序历史记录保留。当前活动锚点见文件顶部和最后一条记录。

- 明确以新发布模型和公开排行榜作为资料更新入口。
- 建立模型记录模板、来源优先级和书系同步范围。
- 建立首批待核验模型候选清单，包含 GPT、Claude、Gemini、DeepSeek、Qwen、Kimi、GLM、Llama、Mistral、Grok 等系列。
- 已完成 GPT-6 Astra 与 GLM-5.3 两条可追溯专题闭环：官方文档摘记、正式章节、目录、百科、术语、题库、练习、论文、项目和知识图谱均已同步。
- 已核验新增章节的 Python 示例、AST、围栏配对、相对链接和 `git diff --check`；未公开架构、参数量、发布日期和 SAO 机制均保留为待核验。
- 已完成 K2 Horizon MoVA 36B/A4B 的榜单发现、官方模型卡/固定 revision 核验、第二十一册第 82 章和全局资料同步；K2-Horizon-7B-Uno 作为官方关联 adapter/论文技术记录，不作为排行榜新增模型。
- 已完成 Qwen3.8 的榜单归并、官方 27B/A95B/Flash-Next 模型卡、Flash-Next GitHub/技术报告和 Qwen Cloud 关系核验；新增研究笔记、第二十一册第 83 章及书系配套同步。Qwen3.8 当前状态为“内容专题闭环”。
- 已完成 Gemini 3.8 Flash 的两榜单归并、Google 官方模型页、DeepMind Model Card、发布博客、评测 PDF 及 Gemini API 周边文档核验；随后新增第二十册第 23 章并完成书系配套同步，当前状态为“内容专题闭环”；最新 SDK schema 补证见本进度末尾。
- 已完成 Gemini 3.8 Flash 2026-09-23 官方文档复验：通过 7890 获取模型页、Thinking、Interactions、Tool combination 和 Thought signatures；补齐 stable `1,048,576/65,536`、默认 medium、`max_output_tokens` 合计预算/`incomplete`、store/retention/delete、previous interaction 参数作用域、stateful/stateless replay、modality compatibility、tool call/result `id`、server/client-side tool 和 validated mode。Thinking 与 Tool combination 对标准 function call signature 的位置描述不完全一致，已保留为文档冲突并要求 capability probe。
- 已新增 [`gemini_interactions_replay_demo.py`](research/model-update-2026-09/code/gemini_interactions_replay_demo.py)，标准库合成验证 stateful/stateless、opaque signature、call/result id、SSE 事件顺序、store/delete/retention 和 modality gate；运行输出 `ok=true`、15 个有序事件、`network_called=false`。证据等级为 local protocol toy，不代表真实 Gemini API、模型质量或生产 SLA。
- 已完成 Gemini 3.7 Flash 的夜间断点联网复验：重新获取 Artificial Analysis、DataCurve、Google AI Developers 模型页、DeepMind Model Card、DeepMind Research/可用 publications 入口和 arXiv 精确/全文检索；新增研究笔记并完成来源索引、模型盘点和计划同步。Gemini 3.7 Flash 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Gemini 3.6 Flash 的夜间断点恢复与资料级闭环：重新获取 Artificial Analysis 详情/provider 页、DataCurve、Google AI Developers 模型页、DeepMind Model Card、Thinking/Interactions/工具组合/视频/缓存/Computer Use 文档和 arXiv 精确/全文检索；新增研究笔记并完成来源索引、模型盘点、榜单解释和计划同步。Gemini 3.6 Flash 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Gemini 3.5 Flash 的断点恢复与资料级闭环：重新获取 Artificial Analysis、DataCurve、Google AI Developers 模型页、What's New、DeepMind Model Card、Gemini 3 Flash 依赖 Model Card、Thinking/Interactions/工具组合/视频/缓存/Computer Use 文档和 arXiv 精确/全文检索；新增研究笔记并完成来源索引、模型盘点、榜单解释和计划同步。Gemini 3.5 Flash 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Gemini 3.5 Flash-Lite 的 AA 单榜资料级闭环：重新获取 Artificial Analysis、DataCurve、Google AI Developers 模型页、DeepMind Model Card/PDF、Thinking、Video understanding 和发布方评测；新增研究笔记并完成来源索引、模型盘点、榜单解释、计划同步及书系配套。已确认默认 `minimal` thinking、agentic video 时间轴按需读取和 `processing_call/result` 边界；Model Card 明确基于 Gemini 3.1 Flash-Lite，不新增独立架构章节。
- 已完成 GPT-5.6 Sol/Terra/Luna 的两榜单归并、三个 OpenAI 官方模型页、Reasoning/Agents/Tools/Prompt Caching/Compaction 文档和开发者博客核验；新增研究笔记并完成索引/清单/计划同步。GPT-5.6 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 GPT-5.5/GPT-5.5 Pro 的两榜单锚点归并、OpenAI 两个官方模型页、GPT-5.5 专属指南及 Reasoning/Tools/Tool search/Prompt Caching/Compaction/Images/Conversation state/Background 文档核验；新增研究笔记并完成索引/清单/计划同步。GPT-5.5 当前状态为“资料级闭环”，暂无独立正式章节；GPT-5.5 Instant May/June 已补做 AA 详情复核和 OpenAI 精确路径负检索，仍只作为榜单级关联配置。
- 已完成 Claude Fable 5 的 Artificial Analysis/DataCurve 锚点归并、Anthropic 官方模型页、发布/重新部署公告、API/Agent 文档和 system card 入口核验；新增研究笔记并完成来源索引、模型盘点、榜单解释、计划和进度同步。Claude Fable 5 当前状态为“资料级闭环”，暂无独立架构章节。
- 已完成 Claude Opus 4.8 的 Artificial Analysis/DataCurve 锚点归并、Anthropic 发布公告、System Card 入口和 Dynamic Workflows 官方博客核验；新增研究笔记并完成来源索引、模型盘点、榜单解释、计划和进度同步。Claude Opus 4.8 当前状态为“资料级闭环”，暂无独立架构章节。
- 已完成 Kimi K2.7 Code 的 Artificial Analysis/DataCurve 锚点归并、Moonshot/Kimi 官方资源/API 文档、固定 revision 模型卡、配置、许可证和部署指南核验；新增研究笔记并完成来源索引、模型盘点、榜单解释、计划和进度同步。Kimi K2.7 Code 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Grok 4.5 的 Artificial Analysis/DataCurve 锚点归并、xAI 发布公告、Grok 4.5 官方模型页、Reasoning/Responses/Compaction/Tools/Web Search/X Search/Remote MCP 文档核验；新增研究笔记并完成来源索引、模型盘点、榜单解释、计划和进度同步。Grok 4.5 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Grok 4.6 的 Artificial Analysis/DataCurve 四档锚点归并、xAI 发布公告、Grok 4.6 官方模型页、Reasoning/Compaction/Tools 文档和论文精确标题检索；新增研究笔记并完成来源索引、模型盘点、榜单解释、计划和进度同步。Grok 4.6 当前状态为“资料级闭环”，暂无独立正式章节。
- 历史阶段已完成 Grok 4.20 0309 v2 的 Artificial Analysis 精确条目核验、DataCurve 精确行缺失核验、xAI 模型页/模型注册表、Reasoning/Multi Agent/Compaction/Tools/Release Notes 和 arXiv/新闻入口负检索；新增研究笔记并同步来源索引、模型盘点、榜单解释、计划和本进度表。该阶段当时记为“AA 单榜资料级闭环”；2026-09-29 已补入第二十册第 19.30 节和 Multi-agent beta/API/成本合同，当前状态见本进度顶部，已升级为内容专题闭环。
- 已完成 Claude Opus 5 的 Artificial Analysis/DataCurve 锚点归并、Anthropic 官方模型目录/专属页、完整开发者文档、发布公告、System Card 下载和 Anthropic Research/arXiv 定向检索；研究笔记、来源索引、模型盘点、候选解释、计划和进度已同步。已确认 adaptive thinking、五档 effort、工具/effort 中途变更、512-token 缓存门槛、fallback、1M context 和配置级评测边界；Claude Opus 5 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Claude Fable 5.1 的断点恢复核验：重新抓取 Artificial Analysis、DataCurve、Anthropic 发布页和 System Card，补齐 Fable/Mythos safeguards 分层、cache read 定价、产品入口 effort、发布方 benchmark/安全摘要及 arXiv 外部使用检索；确认 DataCurve 当前没有 Fable 5.1 行。随后新增第二十册第 21 章 21.29，当前为“AA 单榜内容专题闭环”；参数/架构/训练和真实 endpoint 仍待核验。
- 已完成 Claude Sonnet 5 的断点恢复核验：重新获取 Artificial Analysis、DataCurve、Anthropic 发布页、System Card、完整开发者文档、Research 页面和 arXiv 精确标题检索；补齐 adaptive thinking、effort/`max_tokens`、thinking block/signature、new tokenizer、context awareness、server-side compaction、computer toolset、programmatic tool calling、System Card/发布方评测及负面证据。Claude Sonnet 5 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Grok 4.7 的基础锚点核验：三条代理取得一致的 Artificial Analysis 中文首页、Grok 4.7 详情页和 DataCurve 快照；已核验 xAI 模型页、发布页、Reasoning、Context Compaction、Function Calling、Structured Outputs、Remote MCP，并记录更大 base、更长 RL、困难长任务混合、自验证与 safeguard stack 的发布方披露。已同步 encrypted reasoning state、opaque compaction、工具最小权限和发布方 benchmark 的证据边界；DataCurve 没有精确行，不迁移 Grok 4.6 结果。Grok 4.7 当前状态为“AA 单榜内容专题闭环”，暂无独立架构章节。
- 已完成 Qwen3-Omni 30B A3B 的当前锚点核验：三条代理取得一致的 Artificial Analysis instruct 详情页，并取得 reasoning 页面快照；已核验 Qwen3-Omni 官方 GitHub、HF Instruct/Thinking 配置、Qwen 博客、阿里云文档、arXiv `2509.17765` 论文与源码。已同步 Thinker-Talker、AuT、TM-RoPE、视觉 encoder、多码本 Talker/MTP、Code2Wav、异步 chunked prefill、三阶段训练和 Thinker/Talker 后训练的证据边界；DataCurve 没有精确行，不迁移其他 Qwen 结果。新增第二十一册第 91 章，Qwen3-Omni 当前状态为“AA 单榜内容专题闭环”。
- 已完成 DeepSeek V3.2 候选身份归并：`deepseek-v3-2-reasoning-0925` 和 `deepseek-v3-2-0925` 是 deprecated 的 V3.2-Exp revision，`deepseek-v3-2-reasoning`/`deepseek-v3-2` 是 deprecated 的 V3.2 配置，`deepseek-v3-2-speciale` 是同家族专项 checkpoint；DataCurve 没有精确 V3.2 行，因此不新增模型或迁移 V4/V3.1 Agent 分数。
- 已完成 DeepSeek V4 Flash Vision 的断点恢复核验与资料级闭环：重新抓取 Artificial Analysis、DataCurve 和 DeepSeek Quick Start；确认两个可用代理对 AA 返回相同完整页面，三个代理对 DataCurve/Quick Start 返回相同完整页面，7890 对 AA 大页面仍 TLS EOF。已核验历史实验多模态 API、当前旧 alias → V4.1-Flash 路由、图像预算、Files `file_id`、Responses 图像工具回灌和证据边界；第 85 章 demo AST、正常执行及 9 组边界测试通过。
- 已完成 GPT-5.4 的断点恢复核验与资料级闭环：通过三个代理重新抓取两个排行榜，并通过 7890 实际抓取 GPT-5.4 详情页、官方模型页和专属指南；确认三条线路对排行榜返回相同完整页面，AA 详情与既有哈希一致，DataCurve 仍无 GPT-5.4 Pro 或 mini/nano 精确模型 ID 行（`mini_swe_agent_gpt_5_4_xhigh` 是 base + harness 行）。已提取 1M context、deferred `tool_search`、computer use、native compaction、custom tools/CFG、`allowed_tools`、`phase` 和 Responses 状态等面试主线；暂无独立正式章节。
- 已完成 GPT-5.4 mini/nano 的独立资料级闭环：Artificial Analysis 精确条目、OpenAI 两个 sibling 模型页与 GPT-5.4 运行时指南已核验；mini/nano 各自的 snapshot、400K context、272K maximum input、128K maximum output、effort、定位、价格和工具清单已分开记录。DataCurve 只有 `mini_swe_agent_gpt_5_4_xhigh` 的 GPT-5.4 base 行，没有 mini/nano 精确模型 ID，未迁移 base 的 Agent 成绩；详见 [`gpt-5.4-mini-nano-source-notes.md`](research/model-update-2026-09/gpt-5.4-mini-nano-source-notes.md)。
- 已完成 GLM-5.2 官方博客补证和正式专题落地：正确路由为 `https://z.ai/blog/glm-5.2`，正文资源中的 IndexShare、MTP speculative decoding、长上下文 serving、critic-based PPO 和 coding-agent anti-hack 已进入研究笔记与第二十一册第 86 章；SAO/compaction 原始定义、参数架构、完整训练 recipe、生产 kernel、硬件 profiling 和独立复现仍待核验。
- 已完成 GPT-6 Astra 运行时资料补证：通过 OpenAI 官方模型页、Reasoning、Prompt caching、Conversation state、Compaction 和 Tool search 文档，补齐动态 `configuration_update`、reasoning/`phase` item replay、deferred tool search、缓存前缀、canonical compaction context 和长任务状态账本；研究笔记与第六册第 18 章已同步。内部架构、参数、训练 recipe、system card 和专属技术报告仍待核验。
- 已完成 GPT-6 Astra 官方模型指南与 Agent 协议补证：通过 OpenAI 官方模型指南、Async tool calling、Mid-turn steering、Misalignment monitoring、Agents、Fast mode 文档和 `Rethinking skills and prompts for GPT-6 Astra` 博客，补齐应用侧异步 function/custom tool、`call_id`/job registry、WebSocket `response.steer`、steered continuation、异步安全告警/阻断、skills/`AGENTS.md` 渐进披露和完成定义等面试主线；已同步研究笔记、第六册第 18 章、第四册百科和全局配套。`phase` 当前专节以 GPT-5.5/GPT-5.4 为示例，已标明为通用 replay 规则，不写成 GPT-6 专属能力。
- 已完成 GPT-5.3 Codex 资料级闭环：Artificial Analysis 精确锚点、DataCurve 无精确行的负证据、OpenAI 官方模型页、Codex Prompting Guide、Compaction/Conversation state/Tools/Agents/Prompt caching 文档已核验；新增研究笔记、来源索引、模型盘点、榜单解释、计划/进度及书系配套。已确认 Responses-only、400K/272K/128K、reasoning effort、Codex harness、phase、完整 replay、compaction、工具契约和缓存前缀边界；参数、架构、训练 recipe、system card、专属报告、kernel 和线上 acceptance rate 仍待核验。
- 已完成 OpenAI gpt-oss 资料级闭环：从 Artificial Analysis 发现 `gpt-oss-120b` 与 `gpt-oss-20b`，确认 DataCurve 没有精确 `mini_swe_agent` 行；已核验 OpenAI Model Card/arXiv、官方 gpt-oss 仓库、Harmony、Hugging Face 模型卡/配置和 Cookbook。已新增第二十一册第 87 章并同步来源索引、模型盘点、榜单解释、计划、面试题、练习、术语、项目、论文、知识图谱及 Agent/Serving 相关章节；参数、完整训练 recipe、router 负载均衡、MXFP4 kernel、目标硬件 profiling、线上 acceptance rate 和独立 Agent 评测仍待核验。
- 已完成 Claude Opus 4.6 资料级闭环：从 Artificial Analysis 的 `adaptive` 与基础配置发现，确认 DataCurve 当前没有精确 `mini_swe_agent_claude_opus_4_6_*` 行；已核验 Anthropic 发布页、System Card、模型页、adaptive thinking、effort、thinking signature、server-side compaction、tool search、computer use、Advanced tool use 和 context engineering。已同步研究笔记、来源索引、模型盘点、计划及书系面试配套；参数、架构、完整训练/后训练 recipe、compaction 内部编码、生产 kernel、硬件 profiling、线上 acceptance rate 和独立复现仍待核验。
- 已完成 Claude Opus 4.7 内容专题闭环：从 Artificial Analysis 的 `claude-opus-4-7` 与 `claude-opus-4-7-non-reasoning` 配置发现，确认 DataCurve 当前没有精确 `mini_swe_agent_claude_opus_4_7_*` 行；已核验 Anthropic 发布页、System Card、模型页、`xhigh` effort、task budgets、更新 tokenizer、高分辨率视觉、compaction、tool search 和 cyber safeguards 控制面。已同步研究笔记、来源索引、模型盘点、计划、第二册 `7.28`、第二十四册 `32.36` 及书系配套；参数、架构、训练 recipe、生产 kernel、硬件 profiling、线上 acceptance rate 和独立复现仍待核验。
- 已完成 DeepSeek V4 Pro 0813 的双榜锚点核验：Artificial Analysis 精确条目为 `deepseek-v4-pro` 的 `Reasoning, Max Effort`，DataCurve 精确行为配置为 `mini_swe_agent_deepseek_v4_pro_max`；已补齐官方 V4 Pro GA、Quick Start、Responses/Thinking/Tool Calls、模型卡、配置与 `arXiv:2606.19348` 的证据链。已确认 `low/high/max`、Responses stateless、1.6T/49B、1M、CSA/HCA、mHC、Muon、FP4/FP8、SFT+GRPO+on-policy distillation 和配置字段边界；复用既有第二十一册章节，不新增重复 Transformer 专题。
- 已完成 DeepSeek V4 Pro 官方实现证据补强：固定 HF revision `b5968e9190ef611bbf34a7229255be88a0e937c1`，记录 64 个 safetensors 分片的 metadata 边界、`config.json`、DSML encoding、inference config/model/kernel 的文件级哈希；补齐 gated compressor/overlap state、causal top-k indexer、128-token local window、前三层 hash routing、FP4/FP8 kernel、MTP、Hyper-Connections/Sinkhorn 和 serving manifest。完整权重未下载，未运行 CUDA/TileLang inference，`MP=8` 仍只是官方转换示例。

## 进行中

- 只从 DataCurve DeepSWE 与 Artificial Analysis（含 `https://artificialanalysis.ai/zh`）两个排行榜记录新进入者；其他网站不再作为新的模型发现入口。
- 对已由两个排行榜发现的候选模型寻找官方发布、论文/技术报告、模型卡、代码仓库和 API 文档，用于事实核验与周边技术扩展。
- 区分模型页、API 文档、开发者博客和榜单快照的证据责任，避免把产品/运行时字段写成内部训练事实。
- 上一轮活动锚点依次为 `DeepSeek V3.2`、`DeepSeek V4 Pro 0813`、`DeepSeek V4.1-Flash`、`Qwen3.8 Max (0902)`、`Claude Opus 4.7`、`Claude Fable 5.1`、`Claude Opus 5`、`GPT-6 Astra`、`Kimi K3`、`Gemini 3.8 Flash`、`Grok 4.6`、`Grok 4.7`、`Qwen3-Omni 30B A3B`、`K2 Horizon 3.7B` 和 `Qwen3-VL-235B-A22B`；对应的 reference implementation、`deepseek-recipe` 协议、QwenCloud runtime、Anthropic/OpenAI runtime、K3 stable source、Gemini Model Card、Grok runtime、Thinker-Talker、dense-vs-MoVA 和视觉多模态资料已分别收口。最近完成 GPT-5.6 Luna、Qwen3.7 Plus、DeepSeek V4.1-Flash stable release、GLM-5.3 标准 DSA、Claude Opus 5.5 和 GPT-6 Sol 的阶段性复验，当前活动锚点以文件顶部为准。
- 当前 `GPT-6 Sol` 为 AA 单榜资料级闭环；本轮确认 AA 详情、OpenAI 模型页及 Reasoning/Agents/Tools/Compaction 文档，DataCurve 没有精确 Agent 行。下一步只补 GPT-6 Sol 的完整训练/架构来源、独立 benchmark、完整权重、目标硬件 profiling、tool acceptance 和生产 SLO 证据，不迁移其他 GPT 的 Agent 结果。
- 对 K2 3.7B 继续补齐完整训练 recipe、生产 kernel、目标硬件 profiling、线上 tool acceptance、独立 benchmark 和精确 DataCurve Agent 行；对 K2 36B/A4B 的训练报告、Uno 接受率和目标硬件 profiling 的补证仍属于既有锚点后续，不产生新的模型候选入口。

## 待完成

- 对仍处于“仅候选”“部分覆盖”或“资料级闭环”的重点锚点继续补齐官方来源、研究笔记和专题内容。
- 对已闭环锚点继续补证完整 kernel、目标硬件 profiling、线上接受率、完整训练/后训练配方和独立 benchmark，不把资料缺失误写成已确认事实。
- 完成全仓库链接、版本、许可证、时间语境和 Markdown/Python 代码质量检查。
- 继续按可用代理复访官方页面，重新确认价格、版本快照、模型卡、许可证和技术报告；单个代理的 TLS/DNS 失败不能被写成资料不存在。

## 记录规则

事实必须有来源和核验日期；未确认内容标为“待核验”，不进入正式结论。每完成一个模型或一个专题，立即在本文件追加条目。




## 按时间顺序的完整执行记录



## GPT-5.5 Instant June 本轮收口

- AA 的 June/May 条目仍保留为 revision 关联配置；官方 OpenAI 模型目录确认 `gpt-5.5` 与 snapshot `gpt-5.5-2026-04-23`，精确 `gpt-5.5-instant` 页面返回 HTTP 404。
- DataCurve 没有 Instant 精确行，因此不迁移 GPT-5.5 base 的 context、tool catalog 或 DeepSWE 结果。Instant June 当前状态为**榜单级关联配置 + 官方身份负证据**，不新增专属 Transformer 章节。

## Frontier Model Release Radar 维护框架（从 progress_v2.md 迁移）

584. `plan.md` 已补充 Frontier Model Release Radar 维护框架：明确用户提到 GPT-5.6、Fable 5 这类名字的意图是用 frontier model release 作为锚点，反向挖掘当下新知识，而不是把未核验型号写成正文事实；后续每次新模型、system card 或 technical report 发布，应从架构、预训练、后训练、test-time compute、推理 serving、Agent / tool use、多模态、评估、安全治理和产品工程十个层面抽取新知识点，并要求每个知识点说明解决的问题、相对上一代的变化、适用场景、优缺点、失败模式、安全边界和应同步的书籍/章节/题库/练习/术语入口。

585. `plan.md` 已补充前沿技术生命周期审计维度：在 Frontier Model Release Radar 中新增技术生命周期层，要求判断新技术处于 Emerging、Ecosystem Capture、Standardization、Absorbed by Model Capability 或 Obsolescent / Transient 哪个阶段；明确 MCP、A2A、plugin、skill、workflow、agent runtime 等生态技术不能只按热点追踪，还要分析它们是否被新的抽象抢占生态位，是否会随着模型智能、上下文长度、记忆能力、工具调用稳定性和产品内置能力提升而下沉为实现细节、被替代或成为过渡形态；面试表达要能说明“这个技术解决的约束是否仍存在”，而不是只说最近流行什么。

## 按日期整理的专题历史记录（从 plan.md 迁移）

以下内容原属于 `plan.md` 的按日期专题历史记录，现迁移到进度文件。日期、研究证据、来源快照、同步范围和未核验项均保留；后续专题执行记录继续按日期追加到本文件。
## Radar Sweep 2026-07-15：frontier release 初扫

本次按 Frontier Model Release Radar 做了一次轻量联网初扫，重点不是记录榜单分数，而是提炼新模型发布背后的技术趋势。主要参考 OpenAI GPT-5 / GPT-5.6 官方页面、Anthropic Claude 4 / Fable 5 / Mythos 5 / Skills / Connectors 页面、Google Gemini 2.5 官方发布页、Meta Llama 4 官方技术博客，以及 DeepSeek-R1 公开资料。

### 已确认的新信号

1. OpenAI GPT-5 系统化强调“fast model + thinking model + router”的统一系统形态；GPT-5.6 进一步把 Sol / Terra / Luna 分层、`max` / `ultra` reasoning、多 agent 并行、programmatic tool calling、computer use、端到端知识工作、cyber / science 专业评估和 safe-completions 放进同一个发布叙事。
2. Anthropic Claude 4 已强调 hybrid reasoning、extended thinking with tool use、parallel tool execution、memory files、Claude Code、MCP connector、code execution 和 prompt caching；Claude 4 的 SWE-bench 方法说明还显示，相比上一代已经不再需要某些 planning tool scaffold，这是“模型能力吸收外部 scaffold”的直接例子。
3. Anthropic Fable 5 / Mythos 5 页面确认第五代模型线已进入 long-running agentic work、days-long coding / knowledge work、risk-calibrated safeguards、fallback routing 和 trusted access 叙事。Fable 偏一般长周期专业工作，Mythos 偏 cyber / biology 等高风险能力并限制开放。
4. Claude Skills 与 Connectors 形成生态层：Skills 用 `SKILL.md`、reference files、scripts 等让模型学习组织流程；Connectors 通过 MCP 把外部工具、数据库和应用接入 Claude。它们当前处在 Ecosystem Capture / Standardization 之间，但随着模型记忆、上下文和内置 agent 能力增强，部分 skill / workflow 可能被吸收为产品内置能力或普通上下文能力。
5. Google Gemini 2.5 强调 thinking model、增强 base model + improved post-training、把 thinking capability 内建到更多模型，以及 GPQA、AIME、HLE、SWE-bench Verified 等评估口径。
6. Meta Llama 4 强调 open-weight native multimodal、MoE、early fusion、MetaP、FP8 训练、30T+ token、多语言、mid-training、10M context、iRoPE、轻量 SFT -> online RL -> 轻量 DPO、adaptive filtering、teacher / codistillation、异步 online RL infrastructure，以及 GOAT 这类自动化对抗评估。
7. DeepSeek-R1 仍是 RLVR / GRPO / reasoning distillation 路线的重要公开锚点；项目中已覆盖主线，但后续可继续关注其后续技术报告是否把 GRPO / DAPO / DrGRPO 等路线进一步稳定化。

### 与当前书稿的覆盖对照

当前项目已覆盖的主线：

1. RLVR、GRPO、DAPO、DrGRPO、DeepSeek-R1、SimPO、DPO / PPO / RLHF。
2. MoE、Mamba / SSM / hybrid architecture、MLA / MQA / GQA、长上下文、RoPE scaling、位置外推。
3. vLLM、SGLang、PagedAttention、RadixAttention、PD 分离、prefix cache、KV cache、continuous batching、speculative decoding。
4. Function Calling、MCP、A2A、Skill、Plugin、Tool Registry、Tool Router、trace / replay、permission gate。
5. WebArena、OSWorld、SWE-bench、GAIA、tau-bench、ToolBench、BrowseComp、DeepSWE、SimpleQA、HLE、RULER、FRAMES、MMMU、Video-MME 等评估入口。

本次初扫暴露的新增 P1 / 观察项：

1. Programmatic Tool Calling：需要作为 tool-use 生态位迁移案例补入第二十二册或第十七册。它说明工具调用不一定总是“模型看见所有工具结果再继续思考”，部分中间处理可以变成程序化 workflow，降低 token、round trip 和上下文压力。
2. Multi-agent / Ultra as Test-time Compute：需要在第十六册 test-time compute 和第十七册 multi-agent 中补充“并行 agent 不是产品噱头，而是一种 budget-aware parallel search / workflow execution”的表达，同时强调成本、协调、验证和失败合并。
3. Fable / Mythos 风险分层与 fallback routing：需要在第八册 safety / 第十八册产品化 / 第二十三册发布治理中补充“能力越强，越需要 risk-calibrated access、自动 fallback、trusted access 和 data retention / monitoring 约束”。
4. Skills / Connectors / MCP 生命周期：第二十二册已讲 Skill 和 MCP，但还需要增加“生态位迁移”段落：Skills 可能抢占 prompt / workflow / plugin 的位置，也可能被更强模型、更长上下文和记忆吸收为较薄的组织知识层。
5. MetaP、iRoPE、10M context 和 long-context mid-training：第二册/第二十一册已覆盖长上下文和位置编码主线，但 MetaP、iRoPE 作为 Llama 4 官方发布信号，可以作为观察项补一段“前沿模型如何把架构、训练 recipe 和长上下文一起发布”。
6. GOAT / 自动化对抗评估：第八册已有 red teaming 和危险能力评估，但 Meta 的 GOAT 说明自动化、多轮、动态对抗测试正在成为发布前评估基础设施，应补为 safety eval 观察项。
7. 专业 eval 新簇：GeneBench-Pro、LifeSciBench、MedChemBench、SEC-Bench Pro、ExploitBench、ExploitGym、Agents' Last Exam、Terminal-Bench 2.1、AutomationBench 等应进入第七册 benchmark 观察项，但不宜一次性写成“全量主流标准”，需要继续确认公开定义、规模和稳定性。

本次雷达修补已完成轻量落地：

1. Programmatic Tool Calling 与 Skills / MCP 生命周期已补入第二十二册工具协议生态未来演进章。
2. Multi-agent / ultra as test-time compute 已补入第十六册 Test-Time Compute Scaling 章。
3. Risk-calibrated access 与 fallback routing 已补入第八册 Policy / Governance / Model Card 章。
4. MetaP、iRoPE、10M context 和 long-context mid-training 已作为前沿发布观察项补入第二十一册位置编码章。
5. GOAT、专业 eval 新簇和自动化对抗评估已补入第七册评估总览。
6. `INTERVIEW_BANK.md`、`EXERCISES.md` 和 `GLOSSARY_EN_ZH.md` 已补充对应面试题、练习和术语入口。

### 面试可复述结论

```text
我会把最近 frontier model release 看成技术雷达，而不是榜单。趋势上，模型发布正在从单模型能力对比，转向系统能力对比：router 选择 fast / thinking model，多 agent 并行作为 test-time compute，programmatic tool calling 减少 token 和 round trip，long-running coding agent 需要 memory、trace、sandbox 和 verification，安全上用 risk-calibrated fallback 和 trusted access 控制高风险能力。与此同时，MCP、Skills、Connectors、workflow、agent runtime 这些生态技术不是线性替代关系，它们会随模型能力、上下文、记忆和产品内置能力变化而迁移：有的会标准化为基础设施，有的会被模型能力吸收，有的只是过渡 scaffold。
```


586. 已按 Frontier Model Release Radar 完成一次 2026-07-15 轻量初扫，并将结论写入 `plan.md` 的 `Radar Sweep 2026-07-15`：参考 OpenAI GPT-5 / GPT-5.6 官方页面、Anthropic Claude 4 / Fable 5 / Mythos 5 / Skills / Connectors 页面、Google Gemini 2.5 官方发布页、Meta Llama 4 官方技术博客和 DeepSeek-R1 公开资料，结论为当前书稿已覆盖 RLVR / GRPO / DAPO / DeepSeek-R1、MoE / 长上下文 / 位置编码、vLLM / SGLang / KV cache / PD 分离、Function Calling / MCP / A2A / Skills / trace / permission，以及主要 evaluation benchmark 主线；新增 P1/观察项包括 Programmatic Tool Calling、multi-agent / ultra as test-time compute、Fable / Mythos 风险分层与 fallback routing、Skills / Connectors / MCP 生命周期、MetaP / iRoPE / 10M context、GOAT 自动化对抗评估，以及 GeneBench-Pro、LifeSciBench、MedChemBench、SEC-Bench Pro、ExploitBench、ExploitGym、Agents' Last Exam、Terminal-Bench 2.1、AutomationBench 等专业 eval 新簇。

587. Frontier Model Release Radar P1/观察项轻量修补已完成：第二十二册 `50-工具协议生态未来演进与开放题.md` 补 Programmatic Tool Calling 与 Skills / MCP / Connector / Workflow 生命周期，强调生态位迁移和被模型能力吸收风险；第十六册 `07-test-time-compute-scaling.md` 补 multi-agent / ultra 模式作为 test-time compute 的系统化形态；第八册 `11-policy-governance与model-card.md` 补 risk-calibrated access、trusted access 和 fallback routing；第二十一册 `07-position-encoding总览-sinusoidal-learned-relative-rope-alibi.md` 补 MetaP、iRoPE、10M context 和 long-context mid-training 作为前沿发布观察项；第七册 `01-评估总览.md` 补专业 eval 新簇和 GOAT 类自动化对抗评估观察项；同步 `plan.md` 的落地状态、`INTERVIEW_BANK.md` 的 frontier release radar 面试题、`EXERCISES.md` 的 radar 复盘练习和 `GLOSSARY_EN_ZH.md` 的 Frontier Model Release Radar、Programmatic Tool Calling、Multi-Agent Test-Time Compute、Risk-Calibrated Access、Fallback Routing、Skill / MCP Lifecycle、MetaP、iRoPE、GOAT、Specialized Frontier Eval Cluster 等术语。
## 2026-08-05 Frontier Model 增量收口

这是当前进度的最新覆盖口径，优先于下面早期按“第几讲完成”的历史记录：

1. 上一轮登记的 42 个 frontier 专题扩写项已分散落在 36 个既有章节文件中；本轮复核没有把它们重新集中到 `plan.md` 或单一总览章。
2. 工作区现有 55 个新增独立章节文件，均已按知识点分别落盘；本轮进一步拆开了 Adaptive Thinking、Interleaved Thinking、Persistent Workspace、Fallback Routing、p-RoPE、Gated Attention、CSA/HCA 等相邻主题，避免用合并段落代替独立正文。
3. 2026-08-05 的 7,377 行、491,242 字节是上一轮统计；截至本轮扩写后，55 个新增章节共 16,648 行、1,127,615 字节正文，去掉代码/数学/标题后的正文字符约 5,908--14,490，已经补入连续论述、机制推导、worked example、trade-off、边界/失败模式、评测/练习和资料入口。
4. 这些章节已经同步到所属书册目录，并在 `plan.md` 建立“一知识点 -> 独立章节 -> 证据等级”的登记表；第六册新增部署专题已统一为第 15-17 章文件名。
5. 已联网复核的高可信入口包括 OpenAI 模型目录、Anthropic/Google 官方文档、DeepSeek/Qwen/Kimi/Gemma/Mistral/Step/Cohere 官方模型卡或一方页面，以及 AgentWorld、Shieldstral 和推测解码相关论文/引擎文档。
6. Qwen3.8-Max 仍标记为一方产品页信号；GLM-5.5 没有找到可核验的官方 model ID、模型卡或权重，继续保留为待核验观察项；专业评测新名称也按同样规则处理。
7. 本轮每个新增章节都要求有小白解释、专家机制、公式或数量关系、worked example、trade-off、失败模式、评测/练习和资料边界；最终审计还要检查链接、公式围栏、Markdown 围栏和异常文本。


588. 2026-08-05 frontier model anchor 更新已落地：第二十一册补 KDA/Gated DeltaNet、Gated MLA、NoPE、hybrid attention、total/active parameters 和显式 KV/latent/state cache；第五册和第十六册补领域专家培养、RLVR、on-policy distillation、reasoning effort、thinking levels、adaptive/preserve/interleaved thinking；第十七册和第二十册补 AgentWorld、Agent Swarm、长周期 workspace、context folding、harness-aware evaluation、Responses API 与 OpenAI-compatible API 边界；第六册和第二十四册补 MTP/EAGLE/NEXTN/DSpark、FP4/MXFP4/NVFP4、FP8 KV 和 native INT4；第十五册、第七册和第八册补 encoder-free multimodal、厂商自报证据等级、长上下文有效能力、安全路由和 Shieldstral；同步第四册百科、题库、练习、术语表和论文路线。GLM-5.5 保持待核验观察项，Qwen3.8-Max 明确标记为一方产品页信号。
## Radar Sweep 2026-08-05：新模型锚点落地

本轮以已核验的模型卡、官方模型目录、开发者文档和一方产品页为锚点，将新知识写入对应正文和纵向文件。落地重点不是把模型名堆进榜单，而是提炼它们暴露的架构、训练、推理、Agent、评测和治理变化。

### 已落地的知识主线

1. **架构与位置**：Kimi K3 发布文章披露的 KDA + Gated MLA，以及“无显式 position embedding”线索的正确证据边界（层比例与 NoPE 配置待核验）；Qwen3.5/3.6 的 Gated DeltaNet + Gated Attention；Gemma 4 的 local/global attention 和 p-RoPE；North Mini Code 的 local RoPE/global NoPE；DeepSeek-V4 的 CSA/HCA；total/active parameters 与显式 KV、latent cache、递归 state 的预算公式。
2. **后训练与 reasoning**：领域专家 SFT/RL、RLVR、on-policy distillation、reasoning effort、thinking levels、adaptive thinking、preserve thinking、interleaved thinking，以及不同厂商字段不能直接互换的协议边界。
3. **Agent 与 harness**：Qwen-AgentWorld 作为模型+环境体系，Kimi Agent Swarm 的并行协作成本，长周期任务的 persistent workspace、checkpoint、context folding、harness-aware evaluation，以及 Responses API/OpenAI-compatible API 的兼容边界。
4. **Serving**：MTP、EAGLE/EAGLE3、NEXTN、DSpark 的 draft 来源分层；acceptance length 和 speculative speedup 公式；FP4/MXFP4/NVFP4、FP8 KV、native INT4 与 custom encoding/chat template 门禁。
5. **多模态、评测与安全**：Gemma 4 12B Unified encoder-free 口径，原生多模态的 token budget；1M context 的有效能力门禁；厂商自报/产品页证据等级；Shieldstral 的 policy-adaptive classifier、trusted access 和 fallback routing。
6. **AI Infra 成本与容量**：第二十三册容量规划、Prefill/Decode/KV 资源画像和成本治理章节补充 total/active parameters、显式 KV/latent/state cache、reasoning/tool workload、FP8/FP4/native INT4 和 speculative decoding 的容量与单位成本边界。

### 本轮证据边界

1. GPT-5.5、GPT-5.6 Sol/Terra/Luna、Claude、Gemini、DeepSeek-V4、Qwen3.5/3.6、Kimi K2/K3、Gemma 4、Mistral、Step、Cohere 等均按官方目录、model card 或产品文档中能核对的字段写入。
2. Qwen3.8-Max 的结构和参数只标为一方产品页信号；缺少技术报告/公开权重的部分保持待核验。
3. GLM-5.5 本轮没有找到可核验的官方 model ID、model card 或权重，不写成已发布事实，仅保留 release radar 观察项。
4. 价格、吞吐、benchmark 排名和最大上下文都必须绑定版本、硬件、prompt、effort、harness 和评测条件，不从产品宣传语推导通用结论。

### 上一轮已落盘内容的联网复核

本轮没有只核对新增章节；上一轮已经写入正文的 42 个专题也按来源类型重新回看，并检查其是否仍然需要补充。上一轮内容分散在 36 个既有章节文件中，复核结论如下：

1. **仍可作为主干事实的部分**：DeepSeek-R1 技术报告和模型卡、OpenAI/Anthropic/Google 的公开模型与 API 文档、Meta Llama 4 官方发布资料、vLLM/SGLang/TensorRT-LLM 官方 serving 文档，以及 EAGLE、RULER、SWE-bench、OSWorld 等论文或项目主页。正文保留这些资料支持的接口、公开结构、评测定义和工程边界。
2. **需要保留条件的部分**：模型卡和官方产品页中的参数、context length、吞吐、价格、benchmark 数字和“frontier”定位，继续绑定 revision、硬件、prompt、effort、工具、harness 和日期；它们不能被改写成跨模型定律。已在相关既有章节补上版本、评测条件或资料边界的说明。
3. **产品页信号与待核验项**：Qwen3.8-Max 的结构/参数仍只按 QwenCloud 一方产品页记录；GLM-5.5 仍没有可核验的官方 model ID、模型卡或公开权重，正文不把它写成已发布事实。Qwen3.6、DeepSeek-V4 Flash-0731、North Mini Code 等则按对应版本的模型卡或产品资料区分正式版本与观察信号。
4. **前一轮确实需要补充的地方**：原来只有一句话或一段话的前沿信号已拆到本轮独立章节；既有章节中的补写则保留为横向背景、对比、来源等级和交叉引用，不再用综合章替代独立正文。对无法由一手资料确认的专业 benchmark、内部架构和训练配方，正文明确写成待核验或教学抽象。
5. **正文重写状态**：此前新增章节中的提纲式短段落已经作为中间版本处理；截至 2026-08-05，55 个独立章节已进一步扩写为连续书稿正文，补入问题场景、机制/公式、数量例子、工程取舍、失败诊断、评测和复习内容。下表仍然只承担索引作用，不能替代章节正文。
6. **联网失败的处理**：官方页面偶发 403、SSL reset 或超时时，不把网络可达性当作事实证据；以已保存的官方 URL、模型卡/论文/仓库 revision 和可复核文本为依据，并在章节中标注访问限制或证据等级。后续维护应优先重新访问一手来源，再升级观察项。

### 独立章节登记表：每个新增知识点都有正文落点

下面的表是索引和审计清单，不是正文替代品。每一行至少对应一个独立章节文件；章节正文应包含小白解释、专家机制、公式、worked example、trade-off、失败模式、评测、练习和资料边界。相邻概念若需要横向比较，另有综合章，但不能以综合章代替本表的独立正文。

#### 训练、推理和后训练

| 新增知识点 | 独立章节 | 主要证据层级 |
|---|---|---|
| 领域专家 SFT/RL | [第五册第15章](book-05-llm-training/chapters/15-领域专家sft与rl.md) | DeepSeek-V4 官方模型卡 |
| RLVR 与可验证奖励 | [第五册第16章](book-05-llm-training/chapters/16-rlvr与可验证奖励.md) | 论文、官方模型卡 |
| On-Policy Distillation | [第五册第17章](book-05-llm-training/chapters/17-on-policy-distillation.md) | DeepSeek-V4 模型卡、蒸馏论文 |
| Reasoning Effort | [第十六册第13章](book-16-reasoning-models/chapters/13-reasoning-effort与推理预算.md) | 官方模型卡/模型文档 |
| Thinking Levels | [第十六册第14章](book-16-reasoning-models/chapters/14-thinking-levels.md) | 官方模型卡 |
| Adaptive Thinking | [第十六册第18章](book-16-reasoning-models/chapters/18-adaptive-thinking与动态预算.md) | 产品接口 + 教学抽象 |
| Preserve Thinking | [第十六册第15章](book-16-reasoning-models/chapters/15-preserve-thinking与推理状态.md) | 官方工具/推理文档 |
| Interleaved Thinking | [第十六册第19章](book-16-reasoning-models/chapters/19-interleaved-thinking与工具协议.md) | 官方工具/Responses 文档 |
| Multi-Agent / Ultra Test-Time Compute | [第十六册第16章](book-16-reasoning-models/chapters/16-multi-agent与ultra-test-time-compute.md) | test-time compute 论文、产品资料 |
| DeepSeek-R1 的 RLVR 与蒸馏路线 | [第十六册第17章](book-16-reasoning-models/chapters/17-deepseek-r1的rlvr与蒸馏路线.md) | DeepSeek-R1 技术报告 |

#### Agent、Harness 和协议

| 新增知识点 | 独立章节 | 主要证据层级 |
|---|---|---|
| AgentWorld 模型与环境闭环 | [第十七册第13章](book-17-agent-tool-use/chapters/13-agentworld模型与环境闭环.md) | Qwen-AgentWorld 模型卡/论文 |
| Agent Swarm 并行协作 | [第十七册第14章](book-17-agent-tool-use/chapters/14-agent-swarm并行协作.md) | 一方产品资料 + Agent 评测方法 |
| Long-Running Agent | [第二十册第17章](book-20-agent-harness-runtime/chapters/17-long-running-agent与checkpoint.md) | Agent runtime 文档 |
| Persistent Workspace | [第二十册第21章](book-20-agent-harness-runtime/chapters/21-persistent-workspace与状态边界.md) | Agent tracing/状态文档 |
| Context Folding | [第二十册第18章](book-20-agent-harness-runtime/chapters/18-context-folding上下文折叠.md) | runtime 设计资料 + 教学抽象 |
| Harness-Aware Evaluation | [第二十册第19章](book-20-agent-harness-runtime/chapters/19-harness-aware-evaluation.md) | SWE-bench、OSWorld、Evals 文档 |
| Responses API | [第二十册第20章](book-20-agent-harness-runtime/chapters/20-responses-api.md) | OpenAI 官方 API 文档 |
| OpenAI-Compatible API 的协议边界 | [第二十册第22章](book-20-agent-harness-runtime/chapters/22-openai-compatible-api边界.md) | provider API 文档对照 |
| Programmatic Tool Calling | [第二十二册第51章](book-22-tool-protocol-ecosystem/chapters/51-programmatic-tool-calling.md) | Anthropic/OpenAI 工具文档 |
| Skills、MCP 与 Connectors 生命周期 | [第二十二册第52章](book-22-tool-protocol-ecosystem/chapters/52-skills-mcp与connector生命周期.md) | 官方协议/产品文档 |

#### 架构、位置和历史状态

| 新增知识点 | 独立章节 | 主要证据层级 |
|---|---|---|
| KDA / Kimi Delta Attention | [第二十一册第61章](book-21-transformer-architecture-evolution/chapters/61-kda递归注意力.md) | Kimi K3 官方发布文章；完整模型卡/技术报告待核验 |
| Gated DeltaNet | [第二十一册第62章](book-21-transformer-architecture-evolution/chapters/62-gated-deltanet门控线性注意力.md) | Qwen3.5/3.6 官方模型卡 |
| Gated MLA | [第二十一册第63章](book-21-transformer-architecture-evolution/chapters/63-gated-mla门控潜变量注意力.md) | Kimi K3 官方发布文章；完整模型卡待核验 |
| NoPE 与隐式顺序 | [第二十一册第64章](book-21-transformer-architecture-evolution/chapters/64-nope与隐式顺序.md) | Kimi/North Mini Code 模型卡 |
| Hybrid Attention | [第二十一册第65章](book-21-transformer-architecture-evolution/chapters/65-hybrid-attention局部全局与递归状态.md) | Gemma/Qwen/DeepSeek 模型卡 |
| Total Parameters 与 Active Parameters | [第二十一册第66章](book-21-transformer-architecture-evolution/chapters/66-total-parameters与active-parameters.md) | 官方模型卡 |
| iRoPE | [第二十一册第67章](book-21-transformer-architecture-evolution/chapters/67-irope与超长上下文位置机制.md) | Meta Llama 4 发布资料 |
| MetaP 与长上下文训练稳定性 | [第二十一册第68章](book-21-transformer-architecture-evolution/chapters/68-metap与长上下文训练稳定性.md) | Meta Llama 4 发布资料/待核验细节 |
| 显式 KV Cache | [第二十一册第69章](book-21-transformer-architecture-evolution/chapters/69-显式kv-cache在混合架构中的角色.md) | 模型卡 + serving 文档 |
| Latent Cache | [第二十一册第70章](book-21-transformer-architecture-evolution/chapters/70-latent-cache压缩历史表示.md) | MLA/模型卡资料 |
| Recursive State Cache | [第二十一册第71章](book-21-transformer-architecture-evolution/chapters/71-recursive-state-cache递归状态缓存.md) | 混合架构模型卡 + 教学抽象 |
| p-RoPE | [第二十一册第72章](book-21-transformer-architecture-evolution/chapters/72-prope与局部全局位置.md) | Gemma 4 官方模型卡 |
| Gated Attention | [第二十一册第73章](book-21-transformer-architecture-evolution/chapters/73-gated-attention门控显式注意力.md) | Qwen3.5 官方模型卡 |
| CSA / HCA | [第二十一册第74章](book-21-transformer-architecture-evolution/chapters/74-csa-hca压缩注意力.md) | DeepSeek-V4 官方模型卡/技术报告 |

#### Serving、精度和协议适配

| 新增知识点 | 独立章节 | 主要证据层级 |
|---|---|---|
| FP4、MXFP4、NVFP4 | [第六册第15章](book-06-llm-deployment/chapters/15-fp4-mxfp4-nvfp4.md) | NVIDIA 文档、Mistral 模型卡 |
| FP8 KV Cache | [第六册第16章](book-06-llm-deployment/chapters/16-fp8-kv-cache.md) | NVIDIA/vLLM serving 文档 |
| Native INT4 | [第六册第17章](book-06-llm-deployment/chapters/17-native-int4与原生量化部署.md) | 模型卡/引擎文档 |
| MTP / Multi-Token Prediction | [第二十四册第61章](book-24-llm-inference-engine/chapters/61-mtp与多token预测.md) | 模型卡/推理引擎文档 |
| EAGLE / EAGLE3 | [第二十四册第62章](book-24-llm-inference-engine/chapters/62-eagle与eagle3推测解码.md) | 论文、引擎文档 |
| NEXTN | [第二十四册第63章](book-24-llm-inference-engine/chapters/63-nextn多token候选路线.md) | 模型卡/引擎资料 |
| DSpark | [第二十四册第64章](book-24-llm-inference-engine/chapters/64-dspark附加推测解码模块.md) | DeepSeek 模型卡/引擎资料 |
| Acceptance Length | [第二十四册第65章](book-24-llm-inference-engine/chapters/65-acceptance-length与推测收益.md) | 推测解码论文/benchmark |
| Model Protocol Fit 与 custom encoding | [第二十四册第66章](book-24-llm-inference-engine/chapters/66-model-protocol-fit与custom-encoding.md) | 模型卡、tokenizer、serving 文档 |

#### 多模态、评测和安全

| 新增知识点 | 独立章节 | 主要证据层级 |
|---|---|---|
| Encoder-Free Unified Multimodal | [第十五册第13章](book-15-multimodal-generative-models/chapters/13-encoder-free-unified-multimodal.md) | Gemma 4 官方模型卡 |
| Native Multimodal Token Budget | [第十五册第14章](book-15-multimodal-generative-models/chapters/14-native-multimodal-token-budget.md) | 模型卡/processor 文档 |
| 1M Context 有效能力门禁 | [第七册第13章](book-07-evaluation-experiments/chapters/13-1m-context有效能力评测.md) | 模型卡 + RULER/长上下文评测 |
| Frontier Model Evidence Tier | [第七册第14章](book-07-evaluation-experiments/chapters/14-frontier模型证据等级.md) | 官方目录、模型卡、产品页分层 |
| Specialized Frontier Eval Cluster | [第七册第15章](book-07-evaluation-experiments/chapters/15-specialized-frontier-eval-cluster.md) | Terminal-Bench/SWE-bench/OSWorld；其余名称待核验 |
| GOAT 自动化对抗评估 | [第八册第13章](book-08-ai-safety-alignment/chapters/13-goat自动化对抗评估.md) | Meta 发布资料/安全评估方法 |
| Shieldstral 策略自适应多模态分类器 | [第八册第14章](book-08-ai-safety-alignment/chapters/14-shieldstral策略自适应多模态安全分类器.md) | Mistral 模型卡/论文 |
| Risk-Calibrated Access | [第八册第15章](book-08-ai-safety-alignment/chapters/15-risk-calibrated-access.md) | 官方模型文档、NIST AI RMF |
| Fallback Routing | [第八册第16章](book-08-ai-safety-alignment/chapters/16-fallback-routing与安全降级.md) | 官方模型文档、治理框架 |

#### AI Infra 成本

| 新增知识点 | 独立章节 | 主要证据层级 |
|---|---|---|
| Frontier Model Capacity Planning | [第二十三册第61章](book-23-ai-infra/chapters/61-frontier-model-capacity-planning.md) | 官方模型卡 + 容量模型 |
| KV/State Resource Model | [第二十三册第62章](book-23-ai-infra/chapters/62-kv-state-resource-model.md) | vLLM/SGLang 文档 + 架构模型卡 |
| Reasoning/Agent Unit Cost | [第二十三册第63章](book-23-ai-infra/chapters/63-reasoning-agent-unit-cost.md) | Evals/Agent tracing 文档 + 成本模型 |

本表的验收规则是：链接目标存在、章节不为空、章节内有机制解释和至少一个公式/数量关系，并在 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md` 中有对应入口。只有 `plan.md` 中的雷达条目而没有表中章节的知识点，不算已落地。

## 2026-09 Frontier Model 更新

- 第二十一册新增第 75-80 章：Attention Residuals、Kimi Delta Attention、CSA/HCA、mHC、Mistral Small 4、Step 3.5 Flash。
- 已同步论文路线、知识图谱、项目路线、术语、面试题和练习；相关来源与待核验边界记录在 `plan.md`、`progress_v2.md` 和 `research/model-update-2026-09/`。
- 后续继续核验 GPT-6 Astra、GLM-5.3、DeepSeek V4、Kimi K3 的新版本资料，并完善第四册百科交叉引用。

## 2026-09 新模型专题收口

- GPT-6 Astra、GLM-5.3 与 Kimi K3 已分别形成第六册、第十六册、第十七册正式章节，并同步百科、术语、题库、练习、论文、项目和知识图谱。
- GPT-6 Astra 章节代码、GLM-5.3 验证器代码均已运行通过；新增正文相对链接与 `git diff --check` 已通过。
- 官方网络本轮存在 DNS 不稳定，后续复访时仍需重新核对价格、版本快照、模型卡和技术报告；当前不把架构、参数量、SAO 机制或 K3 完整配置写成已确认事实。

## 2026-09 Mistral Small 4 与 Step 3.5 Flash

- 新增第二十一册第 79 章 `Mistral Small 4：混合推理与 EAGLE/NVFP4 部署`，以及第 80 章 `Step 3.5 Flash：MTP-3、滑动窗口和 11B 激活参数`；两章均从官方模型卡的结构字段出发，提供教学公式、代码/实验设计、工程取舍和面试追问。
- 已同步第四册百科、`GLOSSARY_EN_ZH.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`PROJECTS.md`、`PAPERS.md`、`KNOWLEDGE_GRAPH.md`、README、`BOOK_SERIES.md` 和 `ROADMAP.md`；来源和证据边界记录在 `research/model-update-2026-09/`。
- Mistral Small 4 的 119B/约 6.5B、128 experts/4 active、256K、`none/high`、EAGLE 和 NVFP4，以及 Step 3.5 的 196.81B/约 11B、288 routed + 1 shared、top-8、MTP-3、3:1 SWA/full attention 和 Context Manager，均限定为模型卡公开字段或评测协议；训练配方、发布日期含义、后端支持和独立复现仍待核验。
- 两章 Python 示例已做 AST/运行检查；本轮对 26 个改动/新增 Markdown 文件完成本地相对链接检查、数学/代码围栏配对检查和 `git diff --check`。当前环境未提供 `pandoc`、`xelatex`、`wkhtmltopdf` 或 `typst`，PDF 构建留待环境具备工具后复查；候选模型发现与官方核验目标仍保持进行中。

## 2026-09-09：Artificial Analysis 首次采集

- 来源：https://artificialanalysis.ai/
- 已成功获取首页 HTML；原始快照暂存 `/tmp/artificialanalysis.html`（临时文件，不作为长期归档）。
- 页面文本出现 GPT-6、GPT-5.6、Claude Opus 5、Claude Sonnet 5、Claude Fable 5.1、DeepSeek V4 Pro 0813 等候选名称。当前只确认名称出现在页面中，尚未确认对应榜单记录、发布日期、可用性或官方技术说明。
- 同时出现 Claude Code 与不同模型的组合名称，后续必须区分基础模型、推理配置及 Agent 系统，不能混成同一模型榜单。
- 下一步：解析具体条目及评测方法，再逐项交叉核验官方发布、模型卡、论文和技术文档；不从排名反推训练方法。

## 2026-09-09：结构化候选与 Kimi K3 官方核验

- 已从 Artificial Analysis 页面结构化数据提取 2026-09-09 历史表中的 273 个候选配置条目，见 `research/model-update-2026-09/model-inventory.md`；不是 273 个独立基础模型，另有 2026-09-14 实时增量单独记录。
- Kimi K3、GLM-5.3、GLM-5.3-Flash 均在 Artificial Analysis 与 DeepSWE 快照中发现。
- 已读取 Kimi K3 官方发布正文与博客索引，记录架构、训练、部署、评测限制及八条知识扩写入口，见 `research/model-update-2026-09/kimi-k3-source-notes.md`。
- Z.ai 博客索引本次返回内容不足，未将其记为官方核验完成。
- 尚未更新正式章节；下一步继续定位技术报告及模型卡，并核对其他模型官方发布。

## 2026-09-09：写作要求与 GLM 来源状态

- 已将用户强调的篇幅、逐知识点展开、生动例子、定义、解释、代码和分层深度要求写入 `plan.md`。
- Z.ai `/blog` 快照实际为 404 页面；本次访问 `/blog/glm-5.3` 未获得可读正文。两者都不能作为已核验官方发布的依据，后续转查官方模型文档、模型仓库及排行榜外链。
- 整体目标仍进行中，正式书籍章节扩写和配套同步尚未完成。

## 2026-09-09：GLM-5.3 官方文档已读取

- 成功读取官方模型文档，确认其披露沿用 GLM-5.2 基础模型、以环境扩展等后训练方法提升能力。
- 整理五个独立扩写方向及推理参数迁移差异，见 `research/model-update-2026-09/glm-5.3-source-notes.md`。
- SAO 算法定义、权重许可证、报告与评测复现仍待核验；未编造缺失机制。
- Kimi K3 发布文章的普通 HTML 锚点中未提取到论文、GitHub 或权重链接；仍需从官方研究与模型组织寻找。

## 2026-09-09：GPT-6 Astra 官方模型页已核验

- 已读取官方 Markdown 模型页，准确标识为 `gpt-6-astra`，整理输入输出模态、窗口预算、推理档位、工具支持及计费规则。
- 资料见 `research/model-update-2026-09/gpt-6-astra-source-notes.md`。
- 参数与训练机制未由该页披露，发布日期和技术报告待查；不将知识截止日期当成发布日期。

## 2026-09-09：Attention Residuals 原始资料

- 已读取 MoonshotAI 官方仓库 README 与 arXiv 摘要，核验论文首发日期 2026-03-16。
- 已记录深度注意力、可学习伪查询、块级聚合、张量维度和实验边界，见 `research/model-update-2026-09/attention-residuals-source-notes.md`。
- 已形成六步章节讲解设计；论文全文、教学实现与正式正文仍待完成。

## 2026-09-09：Attention Residuals 全文获取状态

- 已获取 arXiv HTML，但该版本生成失败，只有错误页和元信息；已确认 arXiv:2603.15031v1、提交日期 2026-03-16。
- 已下载论文 PDF（约 1.1 MB），当前环境没有 `pdftotext`，尚未完成正文提取；因此仍只依据官方仓库 README 和论文摘要记录机制，不扩写未核验的附录细节。
- 下一步可用 Python PDF 解析库或阅读器提取正文，再核对初始化、两阶段计算和实验设置。

- 已检查本地 PDF 解析能力：环境未提供 `pdftotext`、`pypdf`、`PyPDF2` 或 `fitz`；仅核验 PDF 元数据，未虚构正文提取结果。

## 2026-09-09：Attention Residuals LaTeX 源码核验完成

- 成功下载 arXiv e-print 源码并读取 `3-attnres.tex`、`4-infra.tex`，补齐 Full/Block AttnRes 定义、复杂度、跨 stage caching、两阶段推理和 online softmax 合并依据。
- 研究笔记已增加论文源码核验段落；现可据此设计教学公式和实现，但正式章节仍需独立写作、代码验证及与 Kimi K3 配置的边界说明。

## 2026-09-09：候选条目分层解释

- 已核对重点名称在候选表中的实际条目：GPT-6 Astra 多档 effort、Kimi K3 low/max、GLM-5.3 与 Flash、DeepSeek V4 Flash/Pro/Vision、多档 Claude Opus 5、Qwen3.8 多种规格及 Grok 4.6 多档 effort。
- 新增 `research/model-update-2026-09/inventory-interpretation.md`，正式采用基础模型、推理配置、Agent 系统三层记录规则，避免把排行榜行数当成独立模型数。

## 2026-09-09：Kimi Delta Attention 原始论文核验

- 从 arXiv 读取 `Kimi Linear` 论文摘要与 LaTeX 源码，确认 KDA 的递推、逐通道衰减、rank-1 delta 更新、DPLR 简化和 3:1 KDA/MLA 混合实验设计。
- 新增 `research/model-update-2026-09/kda-source-notes.md`；明确 Kimi Linear 的实验配置不能直接当作 Kimi K3 的完整架构事实。
- 后续将据论文机制设计独立教学章节和最小 recurrent demo，再决定同步到第二十一册、第二十四册与第四册百科。

## 2026-09-09：KDA 教学 demo 验证

- 新增零依赖教学代码 `research/model-update-2026-09/code/kda_recurrent_demo.py`，实现论文递推式的单头状态更新，不冒充 FlashKDA/chunkwise kernel。
- 当前环境未安装 PyTorch，因此先用纯 Python 验证输出与状态形状；运行通过。正式章节将同时提供 PyTorch 版本并解释教学实现与生产 kernel 的差距。

## 2026-09-09：GLM-5.2 前置资料核验

- 已读取 Z.ai 官方 GLM-5.2 文档，确认其长任务、1M 上下文、Agent 场景和产品工作流披露。
- 文档正文未给出 SAO 或 compaction 的算法定义；已新增 `research/model-update-2026-09/glm-5.2-source-notes.md`，明确不从使用示例反推内部训练机制。

## 2026-09-09：SAO/compaction 索引核验

- 已读取 Z.ai 官方 `llms.txt` 和 GLM-5.2 Markdown 文档版本；文档索引仅列出 GLM-5.2 页面，正文没有 SAO 或 compaction 算法定义。
- 当前证据只能支持“GLM-5.3 文档声称继承 GLM-5.2 的 SAO with compaction”，不能确认 SAO 全称、损失函数或压缩算法。已将该项保留为待核验，避免从缩写推断机制。

## 2026-09-09：DeepSeek V4 官方公告核验

- 已读取 DeepSeek 官方 V4 Pro GA、V4 Flash Vision Experimental 和 API Quick Start 页面。
- 已确认 V4 Pro 2026-08-13 公告、Flash Vision Experimental 2026-08-21 公告、API 别名与后端快照更新规则，以及 reasoning effort、Responses API、Codex 优化和多模态 Files API 等公开能力。
- 新增 `research/model-update-2026-09/deepseek-v4-source-notes.md`；参数规模、架构和训练方法仍待技术报告或模型卡核验。

## 2026-09-09：DeepSeek V4 模型卡与技术报告入口

- 读取官方 Hugging Face 模型卡，补齐 V4-Pro/Flash 参数量、激活参数、1M context、MoE、FP4/FP8、CSA/HCA、mHC、Muon、32T+预训练、SFT+GRPO+on-policy distillation 等披露。
- 已将这些信息与 V4 Preview 官方公告交叉记录；新增内容见 `deepseek-v4-source-notes.md`。
- 后续需通读 `arXiv:2606.19348`，核对公式、消融、压缩误差、mHC 与训练协议，暂不直接写入正式章节。

## 2026-09-09：DeepSeek V4 技术报告源码核验

- 成功下载并读取 `arXiv:2606.19348` LaTeX 源码，补齐 CSA/HCA 压缩与稀疏流程、mHC 双随机约束、GRPO+on-policy distillation 后训练、Muon/AdamW 分工、异构 KV cache 和工程优化依据。
- 已将指标基线、精度和压缩倍率的解释边界写入 `deepseek-v4-source-notes.md`；正式章节仍需单独写作、公式检查和教学代码验证。

## 2026-09-09：DeepSeek V4 书系映射

- 新增 `research/model-update-2026-09/deepseek-v4-book-mapping.md`，将 CSA、HCA、mHC、Muon、FP4/FP8 QAT、GRPO、on-policy distillation、异构 KV cache 与 1M context 评估分别映射到专题书和配套文件。
- 清单明确正式章节必须逐主题展开、提供例子/公式/代码/工程取舍和来源边界；当前尚未直接改动正式书稿，避免研究底稿未经完整核验进入正文。

## 2026-09-09：来源索引收口

- 新增 `research/model-update-2026-09/source-index.md`，统一记录 GPT-6 Astra、Kimi K3、GLM-5.3 和 DeepSeek V4 的排行榜发现来源、官方文档、模型卡、论文与待核验项。
- 当前四个重点系列均已有独立研究笔记；下一阶段应从“来源盘点”转入至少一个专题的正式章节扩写，并同步百科、题库、练习和术语。

## 2026-09-09：首篇正式专题落地

- 新增第二十一册正式章节 `book-21-transformer-architecture-evolution/chapters/75-attention-residuals深度方向注意力.md`，覆盖问题例子、标准残差、Full/Block AttnRes、公式、零依赖 demo、工程边界、面试追问和练习。
- 已将第 75 章加入第二十一册 `目录.md`。
- 已完成本次文件广告注入检查；后续需运行 Markdown 数学渲染检查、代码测试，并同步第四册百科、面试题库、练习、术语和知识图谱。

## 2026-09-09：KDA 正式专题落地

- 新增第二十一册正式章节 `book-21-transformer-architecture-evolution/chapters/76-kda从delta规则到线性注意力.md`，覆盖问题场景、状态记忆直觉、KDA 递推、逐通道衰减、Delta/DPLR 关系、KDA/MLA 混合、零依赖代码、训练/推理差异、Kimi K3 证据边界、面试追问和练习。
- 已将第 76 章加入第二十一册 `目录.md`。
- 提取并运行第 76 章中的一个 Python 代码块，通过；此前尝试用 `runpy` 执行 Markdown 的检查命令已识别为错误，不影响正文代码。

## 2026-09-09：CSA/HCA 正式专题落地

- 新增第二十一册正式章节 `book-21-transformer-architecture-evolution/chapters/77-csa-hca从压缩kv到百万上下文.md`，覆盖 KV 账本、CSA、HCA、滑动窗口、异构 cache、成本基线、教学代码、局限、面试题和练习。
- 已将第 77 章加入第二十一册目录；章节 Python 示例提取运行通过。

## 2026-09-09：mHC 正式专题落地

- 新增第二十一册正式章节 `book-21-transformer-architecture-evolution/chapters/78-mhc双随机残差连接.md`，覆盖 Hyper-Connections 背景、双随机矩阵定义、封闭性、Sinkhorn 投影、零依赖代码、与 AttnRes/KDA 区别、工程代价、面试追问和练习。
- 已将第 78 章加入目录；章节 Python 示例提取运行通过。

## 2026-09-09：纵向资料同步与代码复核

- 已将 AttnRes、KDA、CSA/HCA、mHC 的术语追加到 `GLOSSARY_EN_ZH.md`。
- 已将 7 道新面试题追加到 `INTERVIEW_BANK.md`，将 5 个新练习追加到 `EXERCISES.md`。
- 修复并复核第 75 章 RMSNorm 教学实现；修正 Block AttnRes 的退化表述；明确 mHC Sinkhorn 示例要求正矩阵输入。
- 提取并运行第二十一册第 75-78 章所有 Python fenced blocks，全部通过；检查未发现 `\\operatorname` 公式宏或广告注入文本。
- 仍待同步第四册百科、`KNOWLEDGE_GRAPH.md`、`PROJECTS.md`、`PAPERS.md` 和 `progress.md` 的正式索引，并完成 Markdown 数学渲染检查。

## 2026-09-09：论文、知识图谱与项目路线同步

- 已将 Kimi Linear、Attention Residuals、DeepSeek V4 技术报告加入 `PAPERS.md`。
- 已将 KDA、CSA/HCA、mHC、AttnRes、后训练合并和 Serving Cache 依赖关系加入 `KNOWLEDGE_GRAPH.md`。
- 已将 Million-Context Attention Budget Lab 与 Residual Flow Stability Lab 加入 `PROJECTS.md`。
- 本轮正式章节与纵向文件同步完成；仍需同步第四册百科条目、README/BOOK_SERIES/ROADMAP 的进度摘要，并执行全局链接和公式检查。

## 2026-09-09：同步检查结果

- PAPERS、KNOWLEDGE_GRAPH、PROJECTS、GLOSSARY、INTERVIEW_BANK、EXERCISES 的本轮新增内容均无 `\\operatorname`；注入防护关键词检查未发现异常文本。
- 第二十一册目录已包含第 75-78 章，来源索引和研究底稿均可追溯到官方页面/论文。
- PAPERS 中 DeepSeek V4 已有一条既存入口和一条专题入口，属于有意保留的不同上下文描述，后续可在总体验收时合并重复索引。
- 未同步 `book-04-llm-encyclopedia/`、README/BOOK_SERIES/ROADMAP/PROGRESS 等总控文件；这些是下一轮需要完成的收口项，不把当前轮标记为全目标完成。

## 2026-09-09：百科与总控文件收口

- 新增第四册百科条目 `book-04-llm-encyclopedia/chapters/19-frontier-architecture-updates.md`。
- 已同步 README、BOOK_SERIES、ROADMAP、PROGRESS 的新模型专题摘要和阅读路径。
- 本轮研究、正式章节、配套训练文件和总控入口已形成闭环；后续仍需检查链接可达性、目录索引和 PDF 构建结果。

## 2026-09-09：第四册目录索引修复

- 检查发现百科正文文件虽已创建，但未加入第四册 `目录.md`；现已补入第 19 项索引。
- 根目录摘要已包含新模型专题；总体验收继续检查链接、目录、公式与 PDF 构建。

## 2026-09-09：本轮链接检查

- 修正路径类型后重新检查新增章节、百科条目、plan 和 progress_v2；本地相对链接缺失数为 0。
- 外部链接已保留来源 URL；网络可达性与 PDF 构建将在环境具备相应工具时继续验证。

## 2026-09-09：Qwen3.8 候选核验

- Artificial Analysis 候选表中发现 `Qwen3.8 2.4T A95B`、`Qwen3.8 27B`、`Qwen3.8 Max` 和 `Qwen3.8-Flash-Next` 等配置。
- Qwen 官方旧博客首页已重定向到 `qwen.ai/research`；Qwen Studio 动态页面只暴露 `/blog?id=qwen3.8` 路由，未返回可读的发布正文、模型卡或技术报告。
- 已将 Qwen3.8 保留为“排行榜候选发现”，未把名称、日期或架构写成正式事实；状态与来源已加入 `inventory-interpretation.md` 和 `source-index.md`。

## 2026-09-09：DeepSeek V4 后训练章节落地

- 新增第五册正式章节 `book-05-llm-training/chapters/18-deepseek-v4领域专家与统一蒸馏.md`，逐项展开领域专家 SFT、GRPO 组内优势、verifier 与权限边界、on-policy distillation、reverse KL、教师路由、成本和失败模式。
- 已将第 18 章加入第五册目录；章节标注了 DeepSeek V4 报告披露范围，未虚构奖励权重、教师数量或完整 loss。
- 本章无 Python fenced block；已检查公式宏和注入防护关键词，未发现异常文本。
- 已补充零依赖 GRPO 组内优势代码示例，并提取运行通过；章节状态从“无代码示例”修正为包含一个教学实现。

## 2026-09-09：GPT-6 Astra 章节验证与配套同步

- 提取并运行第六册第 18 章唯一 Python fenced block，输出 `5.7`，与示例中的 300K 输入、100K 缓存命中、20K 输出和阈值倍率一致。
- 检查该章未发现 `\\operatorname`、异常广告/注入关键词或缺失本地相对链接；官方页本轮因 DNS 不可用无法复访，事实以已保存的 2026-09-09 官方模型页摘记为准。
- 已把 GPT-6 Astra 的已确认字段同步到第四册百科、`GLOSSARY_EN_ZH.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`PAPERS.md`、`KNOWLEDGE_GRAPH.md` 和 `PROJECTS.md`。
- 同步内容严格限制在 `gpt-6-astra` 模型 ID、文本/图像输入、文本输出、窗口/输入/输出预算、reasoning effort、工具/端点和价格阈值；参数量、架构、训练方法、发布日期和技术报告继续标为待核验。

## 2026-09-09：GLM-5.3 长任务章节与配套同步

- 新增第十六册第 20 章 `book-16-reasoning-models/chapters/20-glm-5.3长任务环境与验证器.md`，逐项展开可执行环境、任务契约、oracle/no-op/unsolved-state、奖励捷径、上下文压缩、协议迁移和证据边界；零依赖 verifier demo 已提取运行通过，输出与正文一致。
- 已将第 20 章加入第十六册目录，并同步第四册百科、`GLOSSARY_EN_ZH.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`PAPERS.md`、`KNOWLEDGE_GRAPH.md` 和 `PROJECTS.md`。
- 正文只使用 Z.ai 官方 GLM-5.3/5.2 文档已确认的版本关系、接口字段和高层训练流程；`SAO with compaction` 的全称、损失和压缩机制继续标记为待核验。
- 第 20 章检查通过：Python fenced block 运行输出为 `incomplete: throughput target not met | success | invalid: protected verifier changed`；未发现 `\\operatorname`、异常广告/注入关键词或缺失本地相对链接，`git diff --check` 通过。

## 2026-09-09：Kimi K3 发布证据与 Harness 章节

- 新增第十七册第 15 章 `book-17-agent-tool-use/chapters/15-kimi-k3发布证据与长任务harness.md`，展开发布文章与论文的证据分层、`R=F(M,H,E,B,D)`、思考状态清单、跨模型迁移、harness-aware evaluation 和 benchmark 读法。
- 已将第 15 章加入目录，并同步第四册百科、术语、题库、练习、论文、项目和知识图谱。
- 正文不把 Kimi K3 发布文章中的参数、权重承诺、Stable LatentMoE、量化或评测数字当作独立复现事实；KDA 与 AttnRes 仅用对应论文解释一般机制，K3 的完整配置和权重状态继续待核验。
- 已同步 README、`BOOK_SERIES.md`、`ROADMAP.md` 和 `progress.md` 的新模型阅读入口与完成状态；全目标仍保持 active，候选模型的后续官方核验未结束。
- Kimi K3 章节新增零依赖评测可比性 demo，输出 `0.1 0.5 False`，验证同 harness/硬件/effort/任务集才计算绝对与相对提升；Python AST、实际运行、链接检查和 `git diff --check` 均通过。

## 2026-09-09：Mistral Small 4 与 Step 3.5 Flash 专题落地

- 新增第二十一册第 79 章 `Mistral Small 4：混合推理与 EAGLE/NVFP4 部署`，覆盖模型卡公开的 119B/约 6.5B MoE、128 experts/4 active、256K、多模态输入、`reasoning_effort`、EAGLE、NVFP4、许可证和部署边界。
- 新增第二十一册第 80 章 `Step 3.5 Flash：MTP-3、滑动窗口和 11B 激活参数`，覆盖约 196.81B/约 11B active、45 层、288 routed + 1 shared、top-8、MTP-3、3:1 SWA/full attention、Context Manager 和评测协议边界。
- 已将两章加入目录，并同步第四册百科、术语、题库、练习、项目、论文、知识图谱、README、`BOOK_SERIES.md` 和 `ROADMAP.md`；来源索引新增对应模型卡与技术报告入口，重复的 Qwen3.8 状态已清理。
- 两章示例均已通过 Python AST 和实际运行检查；本轮对 26 个改动/新增 Markdown 文件完成本地相对链接检查，新增章节的数学/代码围栏配对检查和 `git diff --check` 通过。模型卡 benchmark、吞吐、后端 MTP 支持、完整训练配方与发布日期含义仍需绑定 revision、硬件和后端继续核验；当前环境未提供 `pandoc`、`xelatex`、`wkhtmltopdf` 或 `typst`，PDF 构建留待环境具备工具后复查。

## 2026-09-10：Claude Opus 5 官方模型目录核验

- Anthropic 官方模型目录缓存页已提供 `claude-opus-5` 的一手接口字段：1M context、128K 最大输出、300K batch 最大输出、adaptive thinking、默认 high effort、平台、价格和 2026-05 知识/训练截止字段；页面还给出 `2026-07-24` 发布日期字段。
- 已新增 `research/model-update-2026-09/claude-opus-5-source-notes.md`，并完成来源索引、候选解释、第四册百科、术语、题库、练习、论文、知识图谱、项目和总控文件同步。
- 仍不把参数量、MoE/稠密结构、训练数据、后训练算法、完整推理机制或独立 benchmark 复现写成事实；`adaptive`、effort 与榜单档位均按运行时配置处理。
- 网络复访当前受 DNS 影响；system card、announcement 和真实 API 行为留作下一轮逐页核验，整体新模型发现与官方核验目标保持 active。

## 2026-09-10：Claude Fable 5.1 官方模型页核验

- Anthropic Fable 5.1 专属模型页缓存已提供 `claude-fable-5-1` 的一手字段：2026-09-01 发布、1M context、128K 最大输出、adaptive always-on、默认 high effort、平台、价格和 2026-06 知识/训练截止字段。
- 页面自述的长任务、研究、文档处理优势，以及 preserved thinking、跨轮模型切换、per-message effort、turn-scoped system messages、工具间进度更新等 beta 能力已记录，但没有升级为独立 benchmark 或内部算法事实。
- 已新增 `research/model-update-2026-09/claude-fable-5.1-source-notes.md`，完成来源索引、候选解释、第四册百科、术语、题库、练习、论文、知识图谱、项目和总控文件同步。
- Fable 参数量、架构、训练/后训练细节和独立复现仍待核验；网络复访与关联 announcement/system card 逐页复核留待下一轮，整体目标保持 active。

## 2026-09-10：Claude Sonnet 5 官方模型目录核验

- 从 Anthropic 官方模型目录缓存页核验 `claude-sonnet-5`、2026-06-30 发布字段、1M context、128K 普通最大输出、300K batch 最大输出、Adaptive thinking、默认 high effort、Fast latency 字段、平台、价格和 2026-01 知识/训练截止字段。
- 已新增 `research/model-update-2026-09/claude-sonnet-5-source-notes.md`，并将模型字段同步到来源索引、候选解释、第四册百科、术语、面试题、练习、项目、论文、知识图谱以及 README、`BOOK_SERIES.md`、`ROADMAP.md`、`progress.md`、`progress_v2.md` 和 `plan.md`。
- 证据边界保持严格：模型目录只支持接口、平台和运行时字段，不支持参数量、稠密/MoE 架构、训练配方、后训练算法、完整推理机制或独立 benchmark 复现；价格和延迟字段仍需按目标平台实时复核。
- 网络复访仍受 DNS 影响；system card、announcement、跨平台差异和真实 API 行为留作后续逐页核验，整体新模型发现与官方核验目标保持进行中。

## 2026-09-10：官方页面复访（DNS 重试）

- 再次尝试访问 Anthropic Opus 5 announcement/system card、Opus/Fable 模型页，以及 Qwen3.8 官方博客/研究入口；`www.anthropic.com`、`platform.claude.com`、`qwen.ai` 和 `qwenlm.github.io` 均因 DNS 解析失败返回 curl 退出码 6。
- 本次没有新增一手事实；Opus/Fable 的关联 announcement/system card 与 Qwen3.8 的模型卡/技术报告继续保持“待核验”，不把网络失败当作资料不存在。

## 2026-09-10：新增专题代码离线验收

- 重新提取并运行 Mistral Small 4、Step 3.5 Flash、Attention Residuals、KDA、CSA/HCA、mHC、DeepSeek V4、GPT-6 Astra、GLM-5.3 和 Kimi K3 章节中的 10 个 Python fenced blocks；全部通过 AST 解析和独立运行，无失败样例。
- 新增章节与总控文件的 `git diff --check`、实际本地相对链接检查和 Markdown 围栏配对检查均通过；外部页面因 DNS 不可用未将访问失败写成资料不存在。

## 2026-09-10：DeepSWE v1.1 快照结构化

- 读取本地缓存 `/tmp/deepswe.html`，核验页面标注的 2026-09-03 更新时间、v1.1、113 个任务、91 个仓库、5 种语言、统一 `mini-swe-agent` harness，以及主表 21 个可见配置的 Pass@1、区间、平均成本、输出 token 和 Agent steps。
- 新增 `research/model-update-2026-09/deepswe-snapshot-notes.md`，并同步来源索引、候选解释和计划入口；结果明确归因于模型配置 + Agent harness + 工具/verifier，不把 DeepSWE 分数当作基础模型能力或实时榜单。
- 完整 model revision、供应商端点、系统提示、工具 schema、重试策略、硬件和独立复现仍待核验；网络复访失败时继续使用带日期的快照边界。

## 2026-09-10：排行榜与官方页面刷新尝试

- 尝试刷新 Artificial Analysis、DeepSWE、Anthropic Fable 页面和 Fable 发布入口，当前环境四个域名均因 DNS 解析失败返回 curl 退出码 6；因此候选盘点继续使用 2026-09-09 已保存的榜单快照，并明确其采集日期，不把它描述为 2026-09-10 实时排行榜。

## 2026-09-10 Claude Opus 5 官方模型目录核验

- 从 Anthropic 官方模型目录缓存页核验 `claude-opus-5`、页面发布日期字段 `2026-07-24`、1M context、128K 普通最大输出、300K batch 最大输出、Claude API/Bedrock/Vertex/Foundry 平台、adaptive thinking、默认 high effort、价格以及 2026-05 知识/训练截止字段。
- 已新增 `research/model-update-2026-09/claude-opus-5-source-notes.md`，并同步来源索引、候选解释、第四册百科、术语、面试题、练习、论文、知识图谱、项目、README、`BOOK_SERIES.md` 和 `ROADMAP.md`。
- 证据边界保持严格：模型目录支持接口与运行时字段，不支持参数量、稠密/MoE 架构、训练配方、后训练算法、完整推理机制或独立 benchmark 复现；当前网络无法复访官方页面时，以保存的页面快照和核验日期为准。
- 后续需对 Anthropic system card/announcement 做逐页复核，并在可复现平台上固定模型 revision、平台、effort、工具和 harness；长期候选模型核验目标仍保持进行中。

## 2026-09-10 排行榜与官方页面刷新尝试

- 尝试刷新 Artificial Analysis、DeepSWE、Anthropic Fable 页面和 Fable 发布入口，当前环境四个域名均因 DNS 解析失败返回 curl 退出码 6；候选盘点继续使用 2026-09-09 已保存的榜单快照，并保留其采集日期，不将其描述为 2026-09-10 实时排行榜。

## 2026-09-10 Claude Fable 5.1 官方模型页核验

- 从 Anthropic Fable 5.1 专属页面缓存核验 `claude-fable-5-1`、2026-09-01 发布字段、1M context、128K 输出、adaptive always-on、默认 high effort、平台、价格和 2026-06 知识/训练截止字段。
- 页面自述其面向 demanding reasoning、long-horizon agentic work、多步研究和文档/表格/幻灯片任务，并列出 preserved thinking、跨轮模型切换、per-message effort、turn-scoped system messages 和工具间进度更新等 beta 能力；这些不等于独立 benchmark 或内部架构证明。
- 已新增 `research/model-update-2026-09/claude-fable-5.1-source-notes.md`，同步来源索引、候选解释、第四册百科、术语、面试题、练习、论文、知识图谱、项目、README、`BOOK_SERIES.md` 和 `ROADMAP.md`。
- 参数规模、稠密/MoE 结构、训练配方、后训练算法、完整技术报告和独立 benchmark 复现继续标为待核验；Claude Mythos 5.1 的邀请制页面信号不作为公开独立模型纳入。

## 2026-09-10 Claude Sonnet 5 官方模型目录核验

- 从 Anthropic 官方模型目录缓存页核验 `claude-sonnet-5`、2026-06-30 发布字段、1M context、128K 普通最大输出、300K batch 最大输出、Adaptive thinking、默认 high effort、Fast latency 字段、平台、价格和 2026-01 知识/训练截止字段。
- 已新增 `research/model-update-2026-09/claude-sonnet-5-source-notes.md`，并将模型字段同步到来源索引、候选解释、第四册百科、术语、面试题、练习、项目、论文、知识图谱以及 README、`BOOK_SERIES.md`、`ROADMAP.md`、`progress_v2.md` 和 `plan.md`。
- 当前证据仅支持接口与运行时字段；参数规模、稠密/MoE 架构、训练配方、后训练算法、完整推理机制和独立 benchmark 复现仍待核验。`Adaptive`、`Fast`、effort 和排行榜配置不用于推断基础模型架构。
- 后续需在网络恢复后复访 system card、announcement、平台差异、价格/延迟快照和真实 API 行为；整体新模型发现与官方核验目标仍在进行中。

## 2026-09-10 官方页面复访（DNS 重试）

- 再次尝试访问 Anthropic Opus 5 announcement/system card、Opus/Fable 模型页，以及 Qwen3.8 官方博客/研究入口；`www.anthropic.com`、`platform.claude.com`、`qwen.ai` 和 `qwenlm.github.io` 均因 DNS 解析失败返回 curl 退出码 6。
- 本次没有新增一手事实；Opus/Fable 的关联 announcement/system card 与 Qwen3.8 的模型卡/技术报告继续保持“待核验”，不把网络失败当作资料不存在。

## 2026-09-10 新增专题代码离线验收

- 重新提取并运行 Mistral Small 4、Step 3.5 Flash、Attention Residuals、KDA、CSA/HCA、mHC、DeepSeek V4、GPT-6 Astra、GLM-5.3 和 Kimi K3 章节中的 10 个 Python fenced blocks；全部通过 AST 解析和独立运行，无失败样例。
- 新增章节与总控文件的 `git diff --check`、实际本地相对链接检查和 Markdown 围栏配对检查均通过；外部页面因 DNS 不可用未将访问失败写成资料不存在。

## 2026-09-10 DeepSWE v1.1 快照同步

- 从本地 `/tmp/deepswe.html` 快照整理页面标注的 v1.1、2026-09-03 更新时间、113 个任务、91 个仓库、5 种语言、统一 `mini-swe-agent` harness，以及主表 21 个可见配置的 Pass@1、区间、平均成本、输出 token 和 Agent steps。
- 新增 `research/model-update-2026-09/deepswe-snapshot-notes.md`，并将评测方法与证据边界同步到 README、`BOOK_SERIES.md`、`ROADMAP.md`、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`PROJECTS.md`、`GLOSSARY_EN_ZH.md` 和 `KNOWLEDGE_GRAPH.md`。
- 明确 DeepSWE 观测对象是模型 revision + effort + Agent harness + 工具 + verifier + timeout/retry/context policy/provider 的组合；不把榜单分数归因于基础模型，不与 Artificial Analysis、SWE-bench 或其他 harness 分数直接合并。
- 当前网络仍无法复访 DeepSWE 域名；快照按采集/页面更新时间保留，模型完整 revision、供应商、工具 schema、硬件、重试和独立复现继续标为待核验。

## 2026-09-14：Claude Haiku 4.5 官方目录快照补充

- 从本地 Anthropic Models Overview 快照结构化字段核验 Claude Haiku 4.5：`claude-haiku-4-5-20251001`、别名 `claude-haiku-4-5`、2025-10-15 发布字段、200K context、64K 最大输出、extended thinking、`fastest` 延迟字段、平台 ID、价格及 2025-02/2025-07 cutoff。
- 新增 `research/model-update-2026-09/claude-haiku-4.5-source-notes.md`，并同步来源索引、候选解释、模型百科、术语、面试题、练习、论文、知识图谱和项目审计器。未读取的 announcement/system card、参数规模、训练架构、完整推理机制和独立 benchmark 继续标为待核验。
- Haiku 4.5 未出现在 2026-09-09 Artificial Analysis 采集切片中，因此不补写榜单日期；明确区分“官方目录发现”和“排行榜发现”。
- 已将 Haiku 4.5 入口同步到 `plan.md`、`README.md`、`BOOK_SERIES.md` 和 `ROADMAP.md`，并完成受影响 Markdown 链接检查。

## 2026-09-14：Artificial Analysis 快照字段边界补强

- 从本地 `/tmp/artificialanalysis.html` 提取并复核 2026-09-09 快照中的配置级 Artificial Analysis Intelligence Index 示例：Fable 5.1 max with fallback 约 53.37、GPT-6 Astra max 约 52.81、Opus 5 max 约 50.70、GLM-5.3 max 约 44.86、Grok 4.6 high 约 44.41、Kimi K3 max 约 43.78、Gemini 3.8 Flash high 约 41.19。
- 已在 `inventory-interpretation.md` 和 `source-index.md` 标注：这些是第三方榜单快照配置，`releaseDate`/`parameters`/开放性字段不能替代官方发布日期、参数或许可证；指数不能与 DeepSWE 或其他 harness 分数直接合并。
- 已将该快照解释入口加入 `plan.md`，后续引用排行榜时统一保留采集日期、effort/fallback、模型 revision 和 harness 条件。
- 快照还显示 GLM-5.3/Kimi K3 的第三方 `parameters` 字段分别为 753/2800；已明确标注为目录估算或归档字段，不升级为官方参数规模或许可证事实。
- 复读本地 Z.AI `llms.txt` 官方文档索引，确认 GLM-5.3-Flash 专属文档、迁移指南和 coding-agent 接入页面入口；未将索引摘要升级为 Flash 的参数、模态、发布日期或 benchmark 事实。

## 2026-09-14：模型候选盘点表结构审计

- 对 `research/model-update-2026-09/model-inventory.md` 的 2026-09-09 历史 Markdown 表执行结构检查：273 条候选记录、273 个唯一名称、273 个唯一链接，日期范围为 2026-01-04 至 2026-09-07；2026-09-14 实时增量另列 8 个 canonical 条目，未覆盖历史表。
- 该统计只证明候选发现表的结构完整，不改变“榜单日期/条目需要回到官方资料核验”的证据边界。

## 2026-09-14：DeepSeek-R1-0528 官方发布页补充

- 从本地 DeepSeek API Docs 快照 `/tmp/deepseek-news.html` 核验 DeepSeek-R1-0528 发布页：页面标注 2025/05/28，明确写出 JSON output、function calling、API 使用方式不变，并提供开源权重入口。
- 新增 `research/model-update-2026-09/deepseek-r1-0528-source-notes.md`，同步来源索引、候选解释和模型盘点；benchmark 图片、参数、架构、训练配方、许可证和独立复现继续标为待核验。
- 该模型未出现在 2026 Artificial Analysis 候选表切片中，因此不伪造榜单日期或第三方分数。
- 已为 Artificial Analysis 快照记录文件大小、mtime 和 SHA-256，并为 Anthropic 模型目录快照记录文件大小与 SHA-256，便于网络恢复后的内容漂移比对。

## 2026-09-14：DeepSeek V4.1-Flash 官方核验与 Artificial Analysis 实时刷新

- 已使用可用网页代理 `10.24.27.134:7890` 复访 Artificial Analysis，独立保存 `/tmp/proxy-live-artificialanalysis.html`：1,771,961 bytes，SHA-256 `ebda1f3ff7fc1956dc629b82ab56400300683a8a134446882d12dfbb4d814423`。结构化数据约 702 个配置条目、673 个唯一名称，最新日期为 2026-09-11。
- 相对 2026-09-09 快照新增 13 个结构 slug/别名：DeepSeek V4.1 Flash、Agnes 3.0 Flash、Ling-3.0-flash-VL、K2 Horizon MoVA/7B/3.7B/0.9B 的成对配置、`mbzuai` 目录项和 DeepSeek V4 Flash Non-reasoning。新增独立文件 `artificial-analysis-2026-09-14-snapshot.md`，没有覆盖旧候选表或伪造榜单日期。
- DeepSeek 官方发布页 `news260910` 返回 HTTP 200，页面标注 2026/09/10；已核验 API 名 `deepseek-flash`、旧 alias 临时路由以及 2026-09-14 04:00 UTC 起 `deepseek-v4-pro` 路由说明。发布页快照 SHA-256 为 `420cbb7b5e8e97632fa45cb49cd2b5f22b57c8f9e125d1c34a22bd67bbc33705`。
- Hugging Face `deepseek-ai/DeepSeek-V4.1-Flash` 模型卡固定 revision 为 `dba1be0a40aa45a94ad051997016db3960a90277`；已核验 552B backbone、1M context、20 层 causal encoder + 20 层 decoder 的 CED、8B/16B prefill/decode active、SWA Bounded Replay、CSA2 `Full/Reindex/Reuse`、Hierarchical Sparse Indexer、FP4 main KV/890 bytes global KV token、384 routed + 1 shared/6 routed per token、196B Engram、DSpark、DeepSeek-ViT、45T token、64K 到 1M 扩展、`SFT -> RL -> OPD` 和数值 reasoning effort。模型卡 README SHA-256 为 `347c9db4e5506acb531cbc3b724407ab88e9af8781679152f0823d7bac16d251`。
- 已读取固定 `config.json`、encoding README、evaluation README；补记 hidden size 5120、40 层、64 attention heads、1 KV head、head dim 512、YaRN factor 16、`sliding_window=128`、indexer/candidate/Engram/DSpark 字段，以及 DSML 前导空格、`low/high/max=50/75/100`、中途 system message、`mini-swe-agent` 复现入口。配置、encoding、技术报告哈希已写入来源笔记。
- 已下载技术报告 PDF，SHA-256 `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`；环境无 `pdftotext`/`pypdf` 等工具，本轮只做下载与哈希核验，不声称已逐页读取报告正文。
- 新增第二十一册第 81 章，更新目录和第四册第 19 章；同步 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、README、`BOOK_SERIES.md`、`ROADMAP.md`、`plan.md`、来源索引、候选解释和历史模型盘点说明。
- 模型卡自报 base/Agent 数字均保留 benchmark、effort、temperature/top-p、harness、工具、环境和 verifier 条件；未把 DeepSWE 74.2 等组合结果归因给基础模型。章节 demo 和全局格式、链接、AST 检查仍需在本轮末执行，完整 kernel、API 价格图片、线上路由实测和独立 benchmark 继续待核验。

## 2026-09-14：DeepSeek V4.1-Flash 技术报告逐页核验

- 通过 `10.24.27.134:7890` 重新获取固定 revision 的技术报告，51 页逐页文本提取成功；哈希仍为 `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`，并记录 PDF 书签与页码范围。
- 报告补充并已写入研究笔记/第 81 章：CED 的 `O(NL/2+n_win*L/2)` 复杂度、CSA2 的 `m=2/m=1` 层分组、`2048*8=16,384` HSI 候选池、Single-Pass mHC traffic、Engram 两模块设置、DSpark 五位置草稿器、RoPE 后 FP4 QAT、EPD 解耦、SWA 的 10% host-DRAM 短 TTL 池、45T/100.6M-token 训练设置和异步 RL/OPD 机制。
- 报告的 Table 1/3、effort 曲线、Dsec 容器密度和 cache 生命周期均保留为发布方内部设置与自报结果；完整 kernel source、所有参数分片、线上接受率、目标硬件 profiling、API 价格/限流和独立 benchmark 仍待核验。
- 清理第 81 章中由区间读取造成的重复段落，更新 `source-index.md`、`PAPERS.md`、`BOOK_SERIES.md`、`KNOWLEDGE_GRAPH.md` 和 `plan.md` 的报告状态；全库结构检查随后执行。

## 2026-09-14：K2 Horizon MoVA 36B/A4B 与 Uno 周边技术

- K2 Horizon MoVA 36B/A4B 由 Artificial Analysis 榜单发现；DataCurve DeepSWE 本地 v1.1 快照未检出 K2，因此只把 Artificial Analysis 记为本轮 K2 的发现证据。
- 已读取 IFM 官方模型卡、固定 revision `de2d2efb32ed7639b7140bccbefe131a0063a982` 的配置与实现，以及 SGLang 部署参考；核验 36B/约 4B active proxy、48 层、前 3 层 dense、后 45 层 MoVA + MoE、32Q/8KV GQA、64 value experts/top-4、100 routed FFN experts/top-8 + 1 shared expert、524,288 context 和路由语义。
- 新增第二十一册第 82 章，并同步第四册第 19 章、目录、来源索引、模型盘点、候选解释、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、README、`BOOK_SERIES.md`、`ROADMAP.md` 和本计划/进度文件。
- `K2-Horizon-7B-Uno` 仅作为官方关联 adapter/论文技术记录：冻结 K2 7B AR base、LoRA diffusion draft 和 `Psi-Spec` AR rejection verification；它不是排行榜新增模型，也不是 36B 架构变体。0.9B 卡片的 MOPD 只保留为同系列训练流程边界，不能反推 36B recipe。
- 完整训练报告、MoVA/FFN 生产 kernel、Uno 接受率、目标硬件 profiling 和独立 benchmark 仍待核验；本轮新增资料不下载几十 GB 权重，正式评测需固定 revision、后端、硬件和 harness。

## 2026-09-14：两个排行榜代理复访与热门模型状态核对

- 通过用户提供的 `10.24.27.134:8098`、`10.24.27.134:7890` 和 `10.237.126.170:1234` 三个代理访问 [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh) 与 [DataCurve DeepSWE](https://deepswe.datacurve.ai/)，六次请求均返回 HTTP 200；这只证明当前入口可读取，不改变榜单字段的第三方证据等级。
- 本次保存 Artificial Analysis `/zh` 响应到 `/tmp/artificialanalysis-live-2026-09-14.html`：1,771,203 bytes，SHA-256 `510bb1916a029700fbd78084a4b26cbe1e2b2690ce9e25179fde2d4fe650a7d4`。与此前 `/tmp/proxy-live-artificialanalysis.html` 的规范化 release 记录比较，已跟踪模型没有新增或删除；页面仍包含 GPT-6 Astra（2026-09-03）、Claude Fable 5.1（2026-09-01）、GLM-5.3（2026-08-18）和 GLM-5.3-Flash（2026-08-26）。
- 本次保存 DataCurve 响应到 `/tmp/deepswe-live-2026-09-14.html`：268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`，与既有 `/tmp/deepswe.html` 字节一致；页面更新时间仍为 2026-09-03，数据对象生成时间仍为 `2026-09-03T22:24:37.984682+00:00`，任务/仓库规模仍为 113/91。当前可见模型行包括 GPT-6 Astra、GLM-5.3 和 Claude Fable 5，不包含 Claude Fable 5.1。
- 因此 GPT-6 Astra、Claude Fable 5.1 和 GLM-5.3 的“榜单发现 + 资料入库 + 书系配套同步”已完成；但三者的参数/内部架构/完整训练配方/独立复现并未全部公开核验。Fable 5.1 当前仍是官方接口字段与产品定位级核验，GPT-6 的训练架构和发布日期、GLM-5.3 的 SAO with compaction 原始定义与模型卡仍列为待补证；本轮没有把“已更新”写成“全部技术细节已确认”。
- 同步修正 `source-index.md`：Fable 5.1 保留为 Artificial Analysis 候选，但不再误记为由 DataCurve DeepSWE 发现；候选表和已完成的书系内容不受影响。

## 2026-09-14：主动关注厂商范围收敛

- 根据用户最新要求，后续主动关注的大模型厂商限定为 OpenAI、Anthropic、Z.ai（GLM）、Qwen、Kimi/Moonshot、DeepSeek、Google（Gemini）和 xAI（Grok）。
- 其他厂商及其新模型暂不主动检索、核验或扩写；现有历史记录不删除、不回退，仅保留作为已有资料，直到用户明确通知重新纳入关注范围。该范围约束已写入 `plan.md`。

## 2026-09-14：热点锚点首轮提取与逐项闭环检查

### 提取口径

- 本轮只从 [Artificial Analysis 2026-09-14 实时快照](research/model-update-2026-09/artificial-analysis-2026-09-14-snapshot.md) 和 [DataCurve DeepSWE v1.1 快照](research/model-update-2026-09/deepswe-snapshot-notes.md) 提取候选；厂商范围限定为 OpenAI、Anthropic、Z.ai（GLM）、Qwen、Kimi/Moonshot、DeepSeek、Google（Gemini）和 xAI（Grok）。
- `AA` 表示出现在 Artificial Analysis 快照，`DS` 表示出现在 DataCurve DeepSWE v1.1 的模型筛选或主表配置中。榜单中的 `effort`、fallback、provider 和 Agent harness 不作为独立模型重复计数。
- 这是一份面向面试技术追踪的首批热点锚点表，不是全量模型目录。GPT-4.x、Gemini 1/2.x、Qwen3 各小规格、DeepSeek V2/V3 历史版本、Claude 4.x 长尾版本等历史条目继续保留在旧盘点中，但不进入本轮主动研究队列。

### 锚点名称与状态

| 厂商 | 热点锚点（按基础模型/release 归并） | 榜单 | 当前闭环状态 | 已有证据与主要缺口 |
|---|---|---|---|---|
| OpenAI | GPT-6 Astra | AA/DS | 内容专题闭环 | 官方模型页、研究笔记、第六册第 18 章及配套同步已完成；参数、训练架构、发布日期和技术报告待补证。 |
| OpenAI | GPT-5.6 Sol、GPT-5.6 Luna、GPT-5.6 Terra | AA；Sol/Luna 亦见 DS | 资料级闭环 | 三个官方模型页、Reasoning/Agents/Tools/Prompt Caching/Compaction 文档、开发者博客和研究笔记已完成；参数、架构、训练配方、system card、技术报告和独立复现待核验。 |
| OpenAI | GPT-5.5、GPT-5.5 Pro、GPT-5.5 Instant | AA；基础 GPT-5.5 亦见 DS | GPT-5.5/Pro 资料级闭环；Instant 为 AA-only 关联配置 | GPT-5.5/Pro 的模型页、专属指南、推理/工具/缓存/状态/后台文档已完成；Instant May/June 的 AA 快照已复核，但官方 `gpt-5.5-instant` 路径 404、DataCurve 无精确行，不能并入 `gpt-5.5` 或迁移其 Agent 结果。 |
| OpenAI | GPT-5.4、GPT-5.4 Pro、GPT-5.4 mini/nano | AA；基础 GPT-5.4 亦见 DS | GPT-5.4/mini/nano 资料级闭环；Pro 为关联档位 | GPT-5.4 base 的两个排行榜配置、mini/nano 的精确 AA 条目、OpenAI sibling 模型页和专属指南已完成；mini/nano 的 400K/272K/128K、effort、任务定位、价格和精确工具矩阵已分开核验；未公开架构、训练 recipe、独立技术报告和 mini/nano 精确 Agent 复现。 |
| OpenAI | gpt-oss-120b、gpt-oss-20b | AA 单榜 | 内容专题闭环 | DataCurve 当前没有精确 `mini_swe_agent_gpt_oss_*` 行；OpenAI Model Card/arXiv、官方仓库、Harmony、Hugging Face 和 Cookbook 已核验；MoE、MXFP4、Harmony、variable-effort reasoning 已进入第二十一册第 87 章，完整 recipe、kernel、profiling 和独立 Agent 评测待补证。 |
| Anthropic | Claude Fable 5.1 | AA | 内容专题闭环（AA 单榜） | 已新增第二十册第 21 章 21.29，补齐 Opus 5.5 → Fable 5.1 单向 thinking-block 可读边、Claude API 专属范围及 model/prefix binding 区分；真实 API、内部架构/训练、独立复现和目标硬件仍待核验。 |
| Anthropic | Claude Fable 5 | AA/DS | 资料级闭环 | Artificial Analysis 的 `max + Opus 4.8 Fallback`、DeepSWE 的 `xhigh + mini-swe-agent`、Anthropic 模型页、发布/重新部署公告、API/Agent 文档、system card 入口和研究笔记已完成；参数、架构、训练配方和独立 benchmark 待核验。 |
| Anthropic | Claude Opus 5 | AA/DS | 资料级闭环 | 两榜单配置、官方模型目录/专属页、开发者文档、发布公告、System Card 入口、研究笔记和配套同步已完成；参数、架构、训练细节和独立 benchmark 待核验。 |
| Anthropic | Claude Opus 4.8 | AA/DS | 资料级闭环 | Artificial Analysis 的 `max`、DeepSWE 的 `xhigh`/`max` + `mini-swe-agent`、Anthropic 发布公告、System Card 入口、Dynamic Workflows 博客、研究笔记和配套同步已完成；参数、架构、训练配方和独立 benchmark 待核验。 |
| Anthropic | Claude Sonnet 5 | AA/DS | 资料级闭环 | 两榜单配置、Anthropic 发布页/模型目录、完整 API/Agent 文档、System Card、Research/arXiv 定向检索、研究笔记和配套同步已完成；参数、架构、完整训练/后训练 recipe 和独立 benchmark 仍待核验。 |
| Anthropic | Claude Sonnet 4.6 | AA/DS | 资料级闭环 | 两榜单配置、Anthropic 发布页、真实模型目录、System Card 入口和 Agent 运行时资料已完成；1M context beta、adaptive/extended thinking、compaction、tool search、computer use 已核验；参数、架构、训练 recipe 和独立 benchmark 待核验。 |
| Anthropic | Claude Opus 4.7 | AA；DataCurve 无精确行 | 内容专题闭环（复用章节） | AA 的 adaptive/max 与 non-reasoning/high 配置、Anthropic 发布页/System Card/模型页、三层预算、更新 tokenizer、高分辨率视觉、compaction、tool search 和 cyber safeguards 控制面已核验；参数、架构、训练 recipe、生产 kernel 和独立 benchmark 待核验。 |
| Z.ai（GLM） | GLM-5.3 | AA/DS | 内容专题闭环 + stable/main runtime source evidence | 已完成 Z.ai 文档/博客、SAO 关联论文、fixed config、标准 DSA 的 Transformers/vLLM/SGLang 入口、第二十一册第 89 章和配套同步；5.3 专属 compaction、完整 recipe、权重加载、目标硬件、独立 benchmark 与生产验收仍待核验。 |
| Z.ai（GLM） | GLM-5.3-Flash | AA/DS | 内容专题闭环 | 已核验 Z.ai 文档/博客、固定 revision 模型卡与配置；已新增第二十一册第 84 章并同步书系配套。完整训练 recipe、生产 kernel、硬件 profiling、线上接受率和独立评测仍待补证。 |
| Z.ai（GLM） | GLM-5 | AA 单榜 | 内容专题闭环（AA 单榜） | Artificial Analysis 有精确 `glm-5` 条目；DataCurve 当前没有精确 `mini_swe_agent_glm_5_*` 行。已核验 Z.ai 模型卡、GLM-5 专属技术报告、DSA、744B/40B MoE、`slime` 异步 RL 和 Agentic Engineering；已新增第二十一册第 89 章及配套同步。完整 DSA kernel、调度细节、训练 recipe、硬件 profiling 和独立复现待补证。 |
| Z.ai（GLM） | GLM-5.2 | AA/DS | 资料级闭环 | 已完成 AA/DataCurve high/max 复验、Z.ai 官方文档、正确博客正文、研究笔记和第二十一册第 86 章；IndexShare、MTP、长上下文 serving、critic-based PPO、anti-hack、1M/128K、MCP、缓存和工作流已核验；参数、架构、SAO/compaction 原始定义、训练 recipe 和独立复现待补证。 |
| Qwen | Qwen3.8 Max、Qwen3.8 Max (0902)、Qwen3.8 27B、Qwen3.8 2.4T A95B、Qwen3.8-Flash-Next | AA；Max/0902 亦见 DS 泛化条目 | 系列内容专题闭环；0902 revision 资料级闭环 | 三份官方模型卡、Flash-Next GitHub/技术报告、Qwen Cloud Max/0902 页面与运行时文档、研究笔记、第 83 章及配套同步已完成；0902 无精确 DataCurve 行；完整生产 kernel、线上接受率、目标硬件 profiling、全系列训练配方和独立 benchmark 待核验。 |
| Kimi/Moonshot | Kimi K3 | AA/DS | 内容专题闭环 + stable release/source implementation evidence | 官方发布资料、K3 仓库/技术报告/许可证、HF 固定 revision/config、FlashKDA README/commit、vLLM `0.29.0` stable release/source、recipe、研究笔记、第十七册第 15 章和第二十一册第 88 章已同步；完整权重未下载，未完成本地 wheel 安装、目标硬件加载、完整 recipe、profiling、独立复现、hybrid cache recovery 和线上 acceptance。 |
| Kimi/Moonshot | Kimi K2.7 Code | AA/DS | 双榜内容专题闭环 | 已新增第二十一册第 94 章和 local protocol toy；已核验 1T/32B active MoE、MLA、MoonViT、native INT4、32K 默认生成预算、256K context、thinking/tool-call 协议与多模态视频预算。Highspeed 按官方同模型服务变体记录，不新增候选；专属训练报告、完整后训练配方、权重加载、线上接受率、目标硬件和独立 benchmark 待核验。 |
| DeepSeek | DeepSeek V4.1 Flash | AA | 内容专题闭环 | 官方发布页、模型卡、技术报告、研究笔记和第二十一册第 81 章已完成；完整 kernel、线上接受率、目标硬件 profiling 和独立 benchmark 待补证。 |
| DeepSeek | DeepSeek V4 Pro、V4 Flash | AA/DS | V4 Pro 双榜内容专题闭环；V4 Flash 为系列资料闭环 | V4 Pro 已有 AA 精确 `deepseek-v4-pro`、DataCurve `mini_swe_agent_deepseek_v4_pro_max`、官方公告/API/模型卡/配置/技术报告和第 77/78 章映射；DataCurve 结果绑定 max + `mini-swe-agent`，生产 kernel、硬件 profiling、线上 acceptance 和独立复现仍待核验。V4 Flash/Flash Vision 的 alias 与视觉边界另行记录。 |
| DeepSeek | DeepSeek V4 Flash Vision | AA | 资料级闭环 | 已完成实验发布、Vision/Files/Responses/价格文档、研究笔记和第二十一册第 85 章；历史 alias、当前路由、图像预算和工具回灌已核验；视觉专属架构/训练报告、DataCurve 同名结果和独立评测仍待核验。 |
| DeepSeek | DeepSeek V3.2 | AA 单榜 | AA 单榜资料级闭环 | Artificial Analysis 有精确 `deepseek-v3-2` Non-reasoning 条目；DataCurve 当前无精确 `mini_swe_agent_deepseek_v3_2_*` 行。官方模型卡、技术报告和 V3.2-Exp 入口已核验；DSA 两阶段训练、2,048 KV top-k、scalable RL、GRPO 稳定化、Agent 合成/验证和 `thinking with tools` 已入库；公式/图表视觉复核、完整 kernel、线上 acceptance rate、硬件 profiling 和独立 benchmark 待补证。 |

## 2026-09-14：Qwen3.8 内容专题闭环

- Artificial Analysis 快照确认四类 Qwen3.8 条目：Max（2026-08-03）、2.4T-A95B（2026-08-12）、27B（2026-08-14）和 Flash-Next（2026-08-26）；DataCurve DeepSWE 当前快照只确认 Max。不同 `reasoning_effort`/Non-reasoning 行按配置归并，不按行数计基础模型。
- 官方核验完成：27B 和 A95B 模型卡、Flash-Next 模型卡与 GitHub、28 页技术报告、Qwen Cloud Max/27B/Flash 页面。研究笔记见 [`qwen3.8-source-notes.md`](research/model-update-2026-09/qwen3.8-source-notes.md)。
- 已整理面试主线：GDN + Gated Attention 的混合记忆、QSA 的 micro-block indexer/两阶段训练、四分支 Gated Residual、Layer 2 N-gram Embedding 的 host-memory prefetch、Muon/AdamW 分工与 batch-size warmup 取舍，以及 `enable_thinking`/`reasoning_effort`/`preserve_thinking` 协议。
- 新增第二十一册第 83 章，并同步第四册百科、目录、来源索引、模型盘点、候选解释、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、README、`BOOK_SERIES.md`、`ROADMAP.md`、`progress.md` 和本计划/进度文件。
- 报告自报边界：QSA 平均分 75.9 -> 76.8、512K--1M RULER 90.08 -> 93.00、1M MRCR 20.71 -> 26.44、kernel prefill/decode 约 7.6x/4.9x；这些结果绑定报告设置和 FlashInfer baseline，不是独立复现。
- 待核验保留：完整生产 kernel、线上 MTP/QSA acceptance rate、目标硬件 profiling、host-memory 预取端到端收益、全系列训练/后训练配方和独立 benchmark。Qwen3.8-Max 是基于 A95B 的 hosted version，不写成独立 open checkpoint。

## 2026-09-14：Qwen3.8 纵向同步与 demo 收口

- 已将 Qwen3.8 研究笔记和第二十一册第 83 章同步到论文路线、面试题库、练习、术语、项目、知识图谱、README、书系目录和路线图；核心主题是 GDN/Gated Attention、QSA、Gated Residual、N-gram Embedding、Muon/AdamW 与 thinking protocol。
- 已修复第 83 章 QSA demo 的预算截断问题：先预留当前 causal tail，再选择完整 block；`token_budget` 不足以容纳 tail 时显式拒绝，正常示例增加 tail 保留断言。
- 已检查 hosted/open 边界：Qwen3.8-Max 只记录为官方说明基于 A95B 的 hosted version，Flash hosted 不转写成独立 checkpoint，effort/Non-reasoning 行不计为新基础模型。
- 本轮不新增模型候选；Flash-Next 报告中的速度、loss、benchmark 和稳定性仍是发布方自报，完整生产 kernel、目标硬件 profiling、线上 acceptance rate、host-memory 端到端收益和独立 benchmark 保持待核验。

## 2026-09-14：Gemini 3.8 Flash 资料级闭环

- Artificial Analysis 快照确认 `Gemini 3.8 Flash` 的 high/medium/low 三个配置，榜单日期均为 2026-09-02；DataCurve DeepSWE v1.1 快照确认 high 配置。不同 effort 行按一个基础模型归并。
- DeepSWE 的 high 配置记录为 Pass@1 74%（±1%）、平均成本约 $2.36、输出 token 约 143K、Agent steps 166；这些数字属于模型配置 + `mini-swe-agent` + 工具/环境/verifier 的系统结果。
- Google 官方模型页和最新模型迁移页将 `gemini-3.8-flash` 标为 GA/stable，确认 1,048,576 输入 token、65,536 输出 token、文本/图像/视频/音频/PDF 输入、文本输出、low/medium/high thinking，以及 caching、Search/Maps grounding、function calling、structured outputs、URL Context、File Search、Code Execution 和 Computer Use（Preview）。
- 已从官方周边文档提取面试技术点：Interactions API 的 thought/tool/model-output steps 与 `previous_interaction_id`，thought summary/signature 的状态连续性，搜索/URL 深读/文件检索/代码沙箱/电脑操作的工具闭环，1M 长上下文与 implicit caching 的成本边界。
- 官方最新模型页称复杂任务会采用更小的 reasoning steps、迭代调用工具并验证结果；该内容按公开系统行为记录，不推断具体 RL、verifier、搜索树或 Transformer 架构。Model Card 将 3.8 架构、训练数据及软硬件信息指向 3.7 Model Card，当前没有独立 3.8 参数/架构/训练报告或论文证据。
- 新增研究笔记 [`gemini-3.8-flash-source-notes.md`](research/model-update-2026-09/gemini-3.8-flash-source-notes.md)，同步 `source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan.md` 和本进度表；暂无 Gemini 专属正式章节，先映射已有 Reasoning、Agent、工具协议、Computer Use、长上下文/Serving 和评测章节。
- 该项闭环属于“资料级闭环”，不等于内部技术已全部公开；后续补证重点为独立架构/训练资料、真实 API 行为、生产工具安全边界和外部 benchmark 复现。

## 2026-09-14：GPT-5.6 资料级闭环

- Artificial Analysis 2026-07-09 快照记录 Sol/Terra/Luna 的多档 effort；DataCurve DeepSWE v1.1 快照记录 `gpt-5.6-sol` max（73% ±3%、约 $6.46、60K 输出 token、61 steps）和 `gpt-5.6-luna` max（67% ±4%、约 $0.61、73K 输出 token、102 steps）。不同 effort 行归并为同一 GPT-5.6 家族；DeepSWE 数字归因于模型配置 + `mini-swe-agent` + 工具/环境/verifier。
- OpenAI 三个模型页确认 Sol/Terra/Luna 的层级、`gpt-5.6` 到 Sol 的 alias、1,050,000 context、922,000 maximum input、128,000 maximum output、text/image input、text output、reasoning tokens 以及 `none`--`max` effort；Chat Completions、Responses、Batch 和 Responses 工具支持也已记录。
- Reasoning 文档补充 GPT-5.6 的 Responses `standard/pro` mode、effort 解耦、`reasoning.context=all_turns` 默认语义、同家族 reasoning 复用、function call 后保留 reasoning items 和 output budget/incomplete 边界。
- Prompt Caching 文档补充 GPT-5.6 的 1,024 可见 token 最小缓存前缀、implicit/explicit 断点、最多四次写入、0.1x 读取、1.25x 写入、30m TTL、自动路由和 compaction 可能降低 cache hit 的边界；Compaction/Agents/Tools 文档补充长期任务、工具搜索、MCP、Skills、Shell 和 harness 分层。
- 官方开发者博客 `Codex as a platform` 给出 retained reasoning + context compaction 使特定 GPT-5.6 Sol 工程流程分数从 13.3% 到 38.3%、输出 token 减少六倍的系统级例子；该数字不写成裸模型 benchmark。`Shell + Skills + Compaction` 与 `One year of Responses` 用于解释长期 Agent 的 runtime 组合。
- 本次没有找到 GPT-5.6 专属参数、架构、训练/后训练 recipe、system card、独立技术报告或可复现 benchmark。`latest-model.md?model=gpt-5.6` 当前实际解析为 GPT-6 Astra，已排除出 GPT-5.6 证据链。研究笔记见 [`gpt-5.6-source-notes.md`](research/model-update-2026-09/gpt-5.6-source-notes.md)。
- GPT-5.6 当前为“资料级闭环”，不新增专属正式章节；面试映射到第六、七、五、八、十六、十七、二十和二十四册。下一锚点转向 GPT-5.5。

## 2026-09-14 Claude Haiku 4.5 官方目录快照补充

- 基于本地 Anthropic Models Overview 快照核验 `claude-haiku-4-5-20251001`、别名、2025-10-15 发布字段、200K context、64K 输出、extended thinking、`fastest` 延迟、平台、价格和 cutoff 字段。
- 新增研究笔记并同步百科、术语、题库、练习、论文、知识图谱和项目；参数、训练架构、system card 全文与独立 benchmark 仍待核验。
- 该模型未出现在 2026-09-09 Artificial Analysis 采集切片中，已在模型盘点中明确标记为官方目录发现，不伪造榜单日期。
- Haiku 4.5 入口已同步到 `plan.md`、`README.md`、`BOOK_SERIES.md` 和 `ROADMAP.md`，受影响链接检查通过。

## 2026-09-14 Artificial Analysis 快照字段边界补强

- 从本地 2026-09-09 Artificial Analysis 快照提取配置级指数示例，并同步来源索引与候选解释；明确 `releaseDate`、`parameters` 和开放性分类是第三方页面字段，不能替代官方资料。
- Fable/GPT-6/Opus/GLM/Grok/Kimi/Gemini 的指数均保留 effort/fallback 和快照日期边界，不与 DeepSWE 或其他 harness 分数直接拼接。
- 该第三方快照解释入口已加入 `plan.md`，后续引用统一绑定采集日期、effort/fallback、revision 与 harness。
- 记录了快照中 GLM-5.3/Kimi K3 的第三方 parameters 字段（753/2800）及其不确定性，未写入正式模型规格结论。
- 复读 Z.AI `llms.txt` 文档索引，补记 GLM-5.3-Flash 页面和 coding-agent 接入入口；Flash 的详细规格仍待专属页面或模型卡核验。

## 2026-09-14 模型候选盘点表结构审计

- `model-inventory.md` 的 2026-09-09 历史表有 273 条候选记录，名称与链接均唯一，日期范围 2026-01-04 至 2026-09-07，格式检查通过；2026-09-14 实时增量另列 8 个 canonical 条目。该结果仅说明盘点表结构完整，不替代官方逐项核验。

## 2026-09-14 DeepSeek-R1-0528 官方发布页补充

- 基于本地 DeepSeek API Docs 快照核验 2025/05/28 发布页、JSON/function calling、API 兼容性承诺和开源权重入口；新增研究笔记并同步来源索引与候选盘点。
- benchmark 图片数字、参数/架构、训练配方、许可证、revision 和独立复现仍待核验；该模型未出现在本轮 Artificial Analysis 切片，不补写榜单日期。
- 已记录 Artificial Analysis 与 Anthropic 页面快照的大小、时间和 SHA-256，后续复访可按哈希比较页面是否发生变化。

## 2026-09-14 DeepSeek V4.1-Flash 与实时榜单快照

- 通过网页代理 `10.24.27.134:7890` 成功复访 Artificial Analysis；独立保存 `/tmp/proxy-live-artificialanalysis.html` 的 1,771,961 bytes 快照，SHA-256 为 `ebda1f3ff7fc1956dc629b82ab56400300683a8a134446882d12dfbb4d814423`。页面解析到约 702 个配置条目、673 个唯一名称，最新日期字段为 2026-09-11；相对 2026-09-09 快照新增 13 个结构 slug/别名。
- 新增发现包括 Agnes 3.0 Flash、Ling-3.0-flash-VL、K2 Horizon 系列、`mbzuai` 目录项和 DeepSeek V4 Flash Non-reasoning 配置；在该快照记录时，只有 DeepSeek V4.1-Flash 已获得官方发布页、模型卡和固定 revision 支撑。之后 K2 Horizon MoVA 36B/A4B 已完成官方资料核验，其他名称继续标为待核验，不把榜单字段当作官方规格。
- 新增研究笔记 `research/model-update-2026-09/deepseek-v4.1-flash-source-notes.md` 与独立快照 `artificial-analysis-2026-09-14-snapshot.md`；记录官方发布页 SHA-256、模型卡 revision `dba1be0a40aa45a94ad051997016db3960a90277`、README/config/PDF 哈希和 PDF 尚未逐页提取的限制。
- 新增第二十一册第 81 章 `book-21-transformer-architecture-evolution/chapters/81-deepseek-v4.1-flash-causal-encoder-decoder.md`，覆盖 20+20 CED、8B/16B prefill/decode 激活口径、SWA Bounded Replay、CSA2 三种层模式、Hierarchical Sparse Indexer、FP4 global KV、MoE、Engram、mHC、DSpark、DeepSeek-ViT、协议迁移和 serving 账本；目录和第四册百科已同步。
- 已同步 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、`README.md`、`BOOK_SERIES.md`、`ROADMAP.md`、`plan.md`、`progress_v2.md`、来源索引、候选解释和模型盘点历史快照说明。
- 模型卡自报的 MMLU-Pro 74.1、HumanEval 79.4、GSM8K 93.0、MMMU-Pro 56.5、DocVQA 95.6、Terminal-Bench 2.1 90.6、DeepSWE v1.1 74.2、AutomationBench 54.8 和 Agent's Last Exam 31.8 均绑定官方评测协议；没有写成基础模型独立能力或本项目复现。
- 本轮新增章节教学 Python 围栏待最终统一验收；技术报告正文、完整 kernel、API 价格图片、实际 alias/限流行为和线上接受率仍待后续核验，整体新模型资料主线保持进行中。

## 2026-09-14 DeepSeek V4.1-Flash 技术报告逐页核验

- 通过可用网页代理 `10.24.27.134:7890` 重新获取固定 revision 的技术报告；51 页正文逐页提取成功，PDF SHA-256 仍为 `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`。
- 报告新增证据已补入研究笔记和第二十一册第 81 章：精确 CED/CSA2 层排布、HSI `2048*8=16,384` 候选池、Single-Pass mHC、Engram/DSpark 具体设置、FP4 QAT 位置、EPD 与 SWA cache 部署、45T/100.6M-token 训练、异步 RL/OPD 和评测边界。
- 报告 Table 1/3、effort 曲线、Dsec 容器密度和 cache 生命周期仍标为发布方自报；完整 kernel source、所有参数分片、线上接受率、目标硬件 profiling、API 价格/限流和独立 benchmark 仍待核验。
- 已同步 `source-index.md`、`PAPERS.md`、`BOOK_SERIES.md`、`KNOWLEDGE_GRAPH.md` 和 `plan.md` 的报告状态，随后执行全库围栏、Python AST、链接和 diff 检查。

## 2026-09-14 K2 Horizon MoVA 36B/A4B 与 Uno 周边技术

- K2 Horizon MoVA 36B/A4B 由 Artificial Analysis 榜单发现；DataCurve DeepSWE 本地 v1.1 快照未检出 K2，因此只把 Artificial Analysis 记为本轮 K2 的发现证据。
- 已读取 IFM 官方模型卡、固定 revision `de2d2efb32ed7639b7140bccbefe131a0063a982` 的配置与实现，以及 SGLang 部署参考；核验 36B/约 4B active proxy、48 层、前 3 层 dense、后 45 层 MoVA + MoE、32Q/8KV GQA、64 value experts/top-4、100 routed FFN experts/top-8 + 1 shared expert、524,288 context 和路由语义。
- 新增第二十一册第 82 章，并同步第四册第 19 章、目录、来源索引、模型盘点、候选解释、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、README、`BOOK_SERIES.md`、`ROADMAP.md` 和本计划/进度文件。
- `K2-Horizon-7B-Uno` 仅作为官方关联 adapter/论文技术记录：冻结 K2 7B AR base、LoRA diffusion draft 和 `Psi-Spec` AR rejection verification；它不是排行榜新增模型，也不是 36B 架构变体。0.9B 卡片的 MOPD 只保留为同系列训练流程边界，不能反推 36B recipe。
- 完整训练报告、MoVA/FFN 生产 kernel、Uno 接受率、目标硬件 profiling 和独立 benchmark 仍待核验；本轮新增资料不下载几十 GB 权重，正式评测需固定 revision、后端、硬件和 harness。

## 2026-09-14 Qwen3.8 纵向同步与 demo 收口

- 已完成 Qwen3.8 研究笔记、第二十一册第 83 章以及 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、`README.md`、`BOOK_SERIES.md` 和 `ROADMAP.md` 的纵向同步。
- 同步主线固定为 GDN/Gated Attention、QSA micro-block indexer/两阶段训练、四分支 Gated Residual、N-gram host-memory prefetch、Muon/AdamW 参数分工和 thinking protocol；Max 仅作为基于 A95B 的 hosted version 记录。
- 修复第 83 章 QSA 教学 demo 的 tail 截断 bug：先预留 causal tail 预算，再选择完整 block；预算不足时显式报错，并增加尾部保留断言。标准库 demo 已实际运行通过，输出保留尾部 `[8, 9]`，但不能替代生产 kernel。
- Flash-Next 报告的速度、loss、benchmark 和稳定性继续标为发布方自报；完整 kernel、线上 acceptance rate、目标硬件 profiling、host-memory 端到端收益、完整训练/后训练配方和独立 benchmark 仍待核验。

## 2026-09-14 Qwen3.8 配套同步收口

Qwen3.8 的专题内容已同步到 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、`README.md`、`BOOK_SERIES.md` 和 `ROADMAP.md`，并保留研究笔记、模型盘点、来源索引及第二十一册第 83 章的交叉入口。同步内容只围绕已发现锚点的 GDN/Gated Attention、QSA、Gated Residual、N-gram Embedding、Muon/AdamW 和 thinking protocol 展开。

第 83 章的 QSA 教学 demo 已修正为先为 causal tail 预留 token budget，再选择完整 micro-block；预算无法容纳 tail 时显式报错，并用断言锁定 tail 不被截断。该 demo 仍是标准库教学实现，不代表 Flash-Next 的生产 kernel；报告的 benchmark、loss、速度和稳定性数字继续按发布方自报处理。

下一步仍是补证队列：完整生产 kernel、目标硬件 profiling、线上 acceptance rate、N-gram host-memory 端到端收益、Muon 分布式实现、全系列训练/后训练配方和独立 benchmark。除非两个排行榜出现新条目，否则不新增模型候选。

## 2026-09-14 GPT-5.6 资料级闭环与后续顺序

GPT-5.6 的研究笔记已完成，闭环范围包括榜单发现、官方模型页、API 周边文档、开发者博客和证据边界。当前最适合面试的技术主线是：`standard/pro` 与 effort 解耦、跨轮 opaque reasoning state、function call 后的 reasoning item 回传、GPT-5.6 prompt-cache 显式断点/1,024 token 门槛/30m TTL、compaction 对缓存复用的影响，以及模型—协议—harness—执行器的责任分层。

本轮不把 GPT-5.6 或 GPT-5.5 的 DeepSWE 数字写成裸模型能力，也不把 API 字段推断为参数、MoE、训练或 RL 机制。下一锚点转向 Claude Fable 5；GPT-5.5/5.6 后续只补官方版本变更、真实 API 行为和独立评测复现。

2026-09-15 网络复访：`10.24.27.134:8098` 和 `10.237.126.170:1234` 均可访问 Artificial Analysis `/zh` 与 DataCurve 并返回 HTTP 200；`10.24.27.134:7890` 本次访问 DataCurve 返回 200，但访问 Artificial Analysis 出现 TLS EOF。Fable 5 页面快照已通过可用代理保存，页面大小和 SHA-256 见研究笔记。

## 2026-09-14 模型资料快照补记（从 progress_v2.md 迁移）

- 2026-09-14 官方 OpenAI 模型页复访因 DNS 解析失败（curl 退出码 6）；未新增一手事实，GPT-6、Kimi K3、GLM-5.3 等待核验边界保持不变。
- 同日复读本地保存的 Z.ai GLM-5.3 官方文档快照，补记其发布方自报的 Terminal-Bench 3.0（4.6→28.3）、DeepSWE v1.1（46.2→66.9）和 Agents' Last Exam（23.8→28.5）前后数值，以及私有 Z.ai Code Bench 的双指标描述；已明确标注为页面快照/发布方自报，未升级为独立复现或基础模型单独增益。
- 复读本地 Kimi K3 官方发布页缓存，补记 Availability 与评测脚注：Kimi Work 3.1.0+、Kimi Code `/model`、API `kimi-k3`、$0.30/$3/$15 每百万 token 价格、Mooncake 分离式推理与编码工作负载缓存命中率“超过 90%”的发布方自报，以及 `max`、temperature=1.0、top-p=1.0 和按基准切换 Kimi Code/Claude Code/Codex harness 的条件。均绑定发布文章快照，未升级为长期费率、普遍性能保证或独立复现。
- 复读本地 DeepSWE v1.1 HTML 快照的数据对象，补记精确生成时间 `2026-09-03T22:24:37.984682+00:00`、统一 `mini-swe-agent` 仓库链接、GPT-6 Astra 最近作业时间及 Kimi K3 的 309/451、Pass@4≈89.4%、4 次重复运行和约 ±4.5% 区间字段；强调表格展示值经过四舍五入，引用时应保留分子/分母、重复次数和区间。
- 官方页面复访仍因 DNS 解析失败，未新增一手事实；GPT-6、Kimi K3、GLM-5.3 等待核验项保持待核验。
- 复读本地 Kimi K3 官方发布页与 DeepSWE v1.1 快照：补记 Kimi API/客户端/价格与评测采样条件、Mooncake 和缓存命中率自报，以及 DeepSWE 精确生成时间、统一 `mini-swe-agent`、Kimi K3 309/451 与重复运行区间。所有字段均保留快照与发布方自报边界，不视为长期费率、普遍性能或独立复现。

## 2026-09-15：GPT-5.5 资料级闭环

- Artificial Analysis 历史快照确认 GPT-5.5 的 `xhigh/high/medium/low/Non-reasoning` 配置与 GPT-5.5 Pro `xhigh`，榜单日期为 2026-04-23；DataCurve DeepSWE v1.1 快照确认 `gpt-5.5` `xhigh`（67% ±6%、约 $7.23、46K 输出 token、82 steps）。这些数字归因于模型配置 + `mini-swe-agent` + 工具/环境/verifier，不写成裸模型能力。
- Artificial Analysis 另有 GPT-5.5 Instant 的 2026-05-05/2026-06-25 条目；当前没有把它们与 OpenAI API `gpt-5.5` 合并，仍标为仅榜单级关联发现。
- OpenAI 官方模型页确认 `gpt-5.5` 的默认 snapshot `gpt-5.5-2026-04-23`、1,050,000 context、128,000 max output、text/image input、text output、reasoning tokens 和 `none`--`xhigh` effort；Pro 页确认 `gpt-5.5-pro-2026-04-23`、Responses/Batch、`medium`--`xhigh`、默认 high、较高计算量和无 cached input discount。
- GPT-5.5 专属指南和 API 文档提取了面试主线：同 effort 下更少 reasoning tokens 的官方行为描述、outcome-first prompting、工具描述与 tool search、`text.verbosity`、图像 `detail` 的 token/精度权衡、Responses `phase` 的 assistant-item 回放、compaction、Pro background mode，以及与 GPT-5.6 不同的 prompt-cache 断点/key/TTL/统计/写入规则。
- 本轮未找到 GPT-5.5 专属参数规模、架构、训练/后训练 recipe、system card、独立技术报告或可复现 benchmark；不能从 `reasoning.effort`、`phase`、tool search、compaction、image_detail 或 1.05M context 反推内部机制。研究笔记见 [`gpt-5.5-source-notes.md`](research/model-update-2026-09/gpt-5.5-source-notes.md)。
- GPT-5.5/Pro 当前为“资料级闭环”，不新增 GPT-5.5 专属正式章节；内容映射到第六、七、十六、十七、二十和二十四册。下一锚点切换为 Claude Fable 5。

## 2026-09-15：Claude Fable 5 资料级闭环与联网复访

- Artificial Analysis 详情页确认 `claude-fable-5` 的 `max + Opus 4.8 Fallback` 配置，页面 `releaseDate` 字段为 2026-06-09；当前快照为 3,531,469 bytes，SHA-256 为 `a69a5ea342984d9b4054bec82756fab3e67863589e7aaf4b31ca38ce3f9ba4d0`。
- DataCurve DeepSWE v1.1 当前快照仍显示 113 个任务、91 个仓库、5 种语言和 `mini-swe-agent`；Fable 5 `xhigh` 为 316/452，Pass@1 约 70% ±3%、Pass@4 约 88.5%、平均成本约 `$13.41`。这些数字绑定模型配置、effort、harness、工具、任务集和 verifier，不是裸模型分数。
- Anthropic 官方资料确认 `claude-fable-5` 为 Active (legacy)，1M context、128K max output、adaptive always-on thinking、默认 `high`、拒答/fallback 协议；官方文档还覆盖 Fallback credit、memory、programmatic tool calling、compaction、context editing 和 task budgets。重新部署公告记录了 cyber safety classifier 更新及向 Opus 4.8 的安全 fallback。
- 面试技术主线已整理为：opaque thinking/signature 与原始思维链的区分、effort 与 `max_tokens` 的区别、HTTP 200 refusal 的业务处理、跨模型 fallback 的工具/权限/缓存边界、just-in-time memory、代码执行中的批量工具调用和三层长任务账本。
- 已下载并登记官方 Fable 5/Mythos 5 system card，文件约 26.96 MB，SHA-256 为 `f95d413845ad8624f384ba026963f2bad2158f10f2626575bb45e823e3c2e0ca`；当前环境未能稳定提取 PDF 正文，因此不从目录或文件字符串扩写安全数值。
- 本次三代理轻量复验结果：`10.24.27.134:8098` 与 `10.237.126.170:1234` 访问两个排行榜均返回 HTTP 200；`10.24.27.134:7890` 访问 DataCurve 返回 200，访问 Artificial Analysis 出现 TLS EOF。Artificial Analysis 快照为 1,772,838 bytes、SHA-256 `1aa28f64778508223642522caffb86da7b9c34395da1030437a5ca1827f369c3`；DataCurve 快照为 268,313 bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- Claude Fable 5 当前标记为“资料级闭环”，不新增独立架构章节；参数规模、内部架构、训练/后训练配方、完整技术报告和独立 benchmark 仍待核验。下一锚点切换为 `claude-opus-4.8`。

## 2026-09-15：Claude Opus 4.8 资料级闭环与联网复验

- Artificial Analysis 详情页确认 `claude-opus-4-8` 的 `max` 配置，页面 `releaseDate` 字段为 2026-05-28；当前页面标记 deprecated，并指向 `claude-opus-5`。DataCurve DeepSWE v1.1 当前快照仍显示 Opus 4.8 的 `xhigh` 和 `max` 配置，统一使用 `mini-swe-agent`。
- Opus 4.8 的 Artificial Analysis 条目约为 41.99 Intelligence Index、约 56.1 output tokens/s、1M context 和约 `$4.08`/task；DeepSWE 的 `xhigh` 记录为 243/447、Pass@1 约 54.36%、Pass@4 约 80.53%、平均成本约 `$8.01`，`max` 记录为 253/429、Pass@1 约 58.97%、平均成本约 `$13.22`。这些数字分别绑定榜单配置、effort、任务集、provider、harness、工具和 verifier，不是裸模型分数。
- Anthropic 官方公告确认 2026-05-28 发布、API ID `claude-opus-4-8`、默认 `high` effort、普通/fast mode 价格边界和更高 effort 使用建议；Messages API 支持任务中途插入 system entries，并强调不破坏 prompt cache。
- Dynamic Workflows 官方博客公开了动态生成 orchestration scripts、数十至数百并行 subagents、独立检查、反驳式复核和长任务断点恢复。面试主线整理为：外部编排图、effort sweep、权限/预算/环境更新、缓存保持、取消/超时治理，以及把“完成声明”转化为测试、lint、类型检查和领域 verifier 的 artifact 门禁。
- 已下载并登记 Opus 4.8 System Card；当前环境未能稳定提取 PDF 正文，因此不从文件目录或字符串扩写安全数值。Anthropic 公开资料没有参数规模、内部架构、训练/后训练 recipe、可独立复现技术报告或完整外部 benchmark。
- 新鲜联网复验：`10.237.126.170:1234` 对 Artificial Analysis `/zh`、Opus 4.8 详情页和 DataCurve 均返回 HTTP 200；`10.24.27.134:8098` 对两个排行榜均返回 HTTP 200。复验文件 SHA-256 为 AA `/zh` `264357ee84d21d14f383cc7b179c34843ab39ef45c5bafad397946c7fadc23fd`、DataCurve `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`、Opus 详情页 `369ef837ec32f3de7b59c8f6fdc7657b8608a22ccff0717c75c6cf3c0167c522`。
- Claude Opus 4.8 当前标记为“资料级闭环”，不新增独立架构章节；内容映射到第六、七、十六、十七、二十和二十四册。当前锚点为 `claude-opus-4.8`，下一锚点切换为 `kimi-k2.7-code`。

## 2026-09-15：Kimi K2.7 Code 资料级闭环与联网复验

- 重新使用用户提供的代理访问 [Artificial Analysis Kimi K2.7 Code](https://artificialanalysis.ai/models/kimi-k2-7-code)、[DataCurve DeepSWE](https://deepswe.datacurve.ai/)、[Kimi 官方资源页](https://www.kimi.ai/resources/kimi-k2-7-code) 和 [API 快速开始](https://platform.kimi.com/docs/guide/kimi-k2-7-code-quickstart)，均获得 HTTP 200。`10.24.27.134:7890` 对 Hugging Face 固定 revision 页面获得 HTTP 200；其对超大页面曾读取超时，未把超时写成资料不存在。
- Artificial Analysis 重抓快照为 3,605,694 bytes，SHA-256 `e37ce8e8f224233a95ca93771fe9f2130e544ef071703bae475295bd1ac527db`；DataCurve 重抓快照为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。两榜单条目均确认 canonical slug `kimi-k2-7-code`；DataCurve 配置为 `mini-swe-agent`、`reasoning_effort:null`，记录 138/452、Pass@1 约 30.53%、Pass@4 约 61.06%、平均成本约 `$2.8155`、平均 149.1 steps。
- 固定 revision `74797c9c62378b951a1f6fcf5c4631024e9b8bef` 的官方模型卡/config 确认 1T 总参数、32B active、61 层、1 dense layer、384 routed experts、top-8、1 shared expert、MLA、256K/262,144 context、MoonViT 400M 和 native INT4 配置；许可证为 Modified MIT，并含大规模商业产品显示名称的附加条款。
- Kimi API 快速开始确认只支持 thinking、强制 `preserve_thinking`、固定 temperature/top-p/n/penalty、`tool_choice` 仅 `auto`/`none`，以及多步 tool call 中回传完整 `reasoning_content` 的协议建议。官方部署指南覆盖 vLLM、SGLang 和 KTransformers；Kimi Code CLI 是推荐的 coding-agent harness。
- 模型卡自报 benchmark 已记录，但所有横向分数均绑定 CLI/Codex/Claude Code、effort、工具预算、上下文长度和重复次数；DeepSWE 结果还绑定 `mini-swe-agent`，不能拼成裸模型排名。动态工具、Context Caching、Partial Mode 和自动断线重连作为 Kimi API 周边技术记录，其中动态工具文档明确当前仅 K3 支持，不写成 K2.7 独有能力。
- 定向检索没有找到 Kimi K2.7 Code 专属 arXiv 论文或独立训练技术报告：官方资源页/模型卡没有提供专属报告链接，`site:arxiv.org "Kimi K2.7 Code"` 查询未返回专属结果，MoonshotAI GitHub repository API 同名查询为零结果。该项是本轮公开入口的负面证据，不是对未来发布的否定。
- Kimi K2.7 Code 当前为“资料级闭环”，暂无独立正式章节；研究笔记见 [`kimi-k2.7-code-source-notes.md`](research/model-update-2026-09/kimi-k2.7-code-source-notes.md)。当前锚点切换为 `kimi-k2.7-code`，下一锚点为 `grok-4.5`。

## 2026-09-15：Grok 4.5 资料级闭环与联网复验

- 本轮重新使用用户提供的代理复验网络状态：`10.237.126.170:1234` 访问 Artificial Analysis Grok 4.5、DataCurve 和 xAI 官方资料均返回 HTTP 200；`10.24.27.134:8098` 访问 Artificial Analysis 与 DataCurve 均返回 HTTP 200。`10.24.27.134:7890` 对 DataCurve 成功，但 Artificial Analysis 大页面出现 TLS EOF，因此不把单个代理失败写成资料不存在。
- Artificial Analysis Grok 4.5 页面配置为 `Grok 4.5 (high)`，第三方 `releaseDate` 为 2026-07-08，Intelligence Index 约 39.08、页面排序约 #38/200、输出速度约 60.98 tokens/s、context 500K；快照 `/tmp/aa-1234.html` 为 3,517,851 bytes，SHA-256 `839c29a20b9f5977fde997e6868f96c15d6c4d4f41c8fb53e5bac15bf7ce078`。
- DataCurve DeepSWE v1.1 页面更新时间为 2026-09-03，`generated_at` 为 `2026-09-03T22:24:37.984682+00:00`，任务/仓库/语言规模为 113/91/5，统一 harness 为 `mini-swe-agent`。Grok 4.5 行配置为 `mini_swe_agent_grok_4_5_high`、`reasoning_effort: high`，记录 243/452、Pass@1 53.761%、Pass@4 77.876%、4 runs、平均成本约 `$2.4157`、约 61.33 Agent steps、约 35,525 输出 token 和约 484.7s；快照 `/tmp/deepswe-1234.html` 为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。这些结果绑定 harness、工具、任务集、环境和 verifier，不是裸模型分数。
- xAI [Grok 4.5 发布公告](https://x.ai/news/grok-4-5) 标注 2026-07-16，公开定位为 coding、agentic tasks 和 knowledge work；发布方描述了 coding/science/engineering/math 数据、deduplication、quality scoring、domain-focused selection、数以万计 GB300 GPU、数十万任务 RL、automated/model-based grading 和 highly asynchronous agentic rollouts。DeepSWE 1.0/1.1、SWE Marathon、Terminal-Bench 2.1、SWE-Bench Pro 和 80 TPS 数字均按 xAI 自报及其各自配置记录，不反推完整训练配方。
- xAI 官方模型页确认 `grok-4.5`、`grok-4.5-latest`/`grok-build-latest` aliases、500K prompt、text/image input、text output、function calling、structured outputs、reasoning、Batch API 不支持、150 RPS、50M TPM、区域和 200K prompt 价格阈值。专属模型页列出 `low/medium/high/xhigh`，但通用 Reasoning 文档和 Release Notes 称 Grok 4.5 的可用档位为 low/medium/high、`xhigh` 按 high 处理；该官方文档不一致保留为待复验项。
- 已从 xAI 官方周边文档提取面试技术点：reasoning_tokens 与 opaque/encrypted reasoning state、Responses 30 天状态化会话、`/v1/responses/compact` 的单项 opaque compaction、server-side built-in tools 与 client-side function calling 的责任分层、默认 parallel function calling、Web/X Search、Remote MCP `allowed_tools` 最小权限。`grok-4.20-multi-agent` 的 4/16 agents 和 leader agent 属于独立模型，不能套到 Grok 4.5。
- [arXiv 精确标题检索](https://arxiv.org/search/?query=%22Grok+4.5%22&searchtype=title) 本轮返回 0 个结果；全文检索命中的论文只是将 Grok 4.5 作为被测或背景模型，当前公开入口未检出 xAI 发布的 Grok 4.5 专属论文或独立技术报告。该项是负面检索证据，不是对未来发布的绝对否定。
- 新增研究笔记 [`grok-4.5-source-notes.md`](research/model-update-2026-09/grok-4.5-source-notes.md)，同步来源索引、模型盘点、榜单解释、`plan.md` 和本进度表。Grok 4.5 当前为“资料级闭环”，没有独立架构/训练报告，因此不新增专属正式章节；当前锚点切换为 `grok-4.5`，下一锚点为 `grok-4.6`。

## 2026-09-15：Grok 4.6 资料级闭环与联网复验

- 本轮重新使用用户提供的代理获取并复验 Artificial Analysis Grok 4.6、DataCurve DeepSWE 和 xAI 官方页面：`10.237.126.170:1234` 与 `10.24.27.134:8098` 对两个排行榜均返回 HTTP 200，`10.24.27.134:7890` 对 xAI 官方模型页返回 HTTP 200。轻量复验中 `1234` 访问 xAI 曾返回代理 503，7890 访问 Artificial Analysis 偶发 TLS EOF；单代理失败未被解释为页面不存在。
- Artificial Analysis 详情页确认 `Grok 4.6 (high)`，并列出 low/medium/xhigh 配置；第三方 `releaseDate` 为 2026-08-12，Intelligence Index `44.4050073012592`，输出速度 `58.5035 tokens/s`，TTFT `40.8596s`，context 500K，价格 `$2/$6`，proprietary 且 parameters 为 null。两个代理抓到的快照均为 3,522,180 bytes，SHA-256 `8d6c96aa27f6db87537f3f1802ca07122d58385ee4b6d5e1a5ab84897f360383`。
- DataCurve DeepSWE v1.1 页面仍为 2026-09-03 更新，`generated_at` 为 `2026-09-03T22:24:37.984682+00:00`，任务/仓库/语言规模为 113/91/5，统一 harness 为 `mini-swe-agent`。Grok 4.6 四档 Pass@1 为 low 41.648%、medium 67.478%、high 65.188%、xhigh 66.741%，平均成本约 `$1.0424/$3.4490/$4.3849/$5.4977`；这些数字绑定 effort、harness、工具、任务集、环境和 verifier，不是裸模型分数。快照为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- xAI [Grok 4.6 发布公告](https://x.ai/news/grok-4-6) 的 JSON-LD 日期为 2026-08-12，公开披露更长 supplemental training、模型生成 reasoning/技术数据、高质量工程数据、改进 optimizer/training recipe、Grok 4.5 生成的多 effort/多 harness SFT trajectories、model-based checks、面向知识工作/编码/领域环境的 agentic RL，以及长轨迹 self-testing/verification。发布方未公开参数规模、网络结构、具体 optimizer、RL objective 或完整 recipe。
- xAI 官方模型页确认 `grok-4.6`、500K context、2026-02-01 knowledge cutoff、text/image input、text output、无文本输出上限、`low/medium/high/xhigh`（默认 high）、Responses/Chat Completions、function calling、structured outputs、Web/X Search、Code Execution、`prompt_cache_key`、`x-grok-conv-id` 和长循环 compaction 建议。实际 HTML 快照为 408,905 bytes，SHA-256 `9669e2c5e96dafb74296f6e11af7c3b0fc74e8dec58a3d674edf18154c1caf27`。
- [arXiv 精确标题检索](https://arxiv.org/search/?query=%22Grok+4.6%22&searchtype=title) 本轮返回 0 个结果；当前公开入口未检出 Grok 4.6 专属论文、独立技术报告或公开权重代码。该项是负面检索证据，不是对未来发布的绝对否定。
- Grok 4.6 当前为“资料级闭环”，不新增独立架构章节；研究笔记见 [`grok-4.6-source-notes.md`](research/model-update-2026-09/grok-4.6-source-notes.md)。当前锚点切换为 `grok-4.6`，下一锚点为 `claude-opus-5`。

## 2026-09-15：Claude Opus 5 资料级闭环与网络复验

- 重新发起联网搜索后，`10.237.126.170:1234` 与 `10.24.27.134:8098` 对 Artificial Analysis 中文首页、Artificial Analysis Opus 5 详情页、DataCurve DeepSWE 和 Anthropic 官方发布/研究页面均返回 HTTP 200；`10.24.27.134:7890` 对 DataCurve 返回 200，但 Artificial Analysis 大页面在超时前只收到部分内容。代理传输失败没有被解释成页面不存在。
- Artificial Analysis 详情页确认 `Claude Opus 5 (Adaptive Reasoning, Max Effort)`、canonical slug `claude-opus-5`、榜单 `releaseDate` `2026-07-24`，约 `50.7002` Intelligence Index、`50.0725` output tokens/s、`46.5045s` median TTFT、1M context 和约 `$5.8584`/task。详情页把 parameters 留为 null、开放性标为 proprietary；这些是第三方页面字段，不是 Anthropic 参数披露。
- DataCurve DeepSWE v1.1 仍标注 2026-09-03、113 个任务、91 个仓库、5 种语言和统一 `mini-swe-agent` harness。Opus 5 max 配置 `mini_swe_agent_claude_opus_5_max` 为 327/444、Pass@1 `73.6486%`（页面约 `74% ±4%`）、Pass@4 `88.4956%`、平均成本 `$11.8376`、平均输出 117,566 tokens、约 99.04 Agent steps；这是模型配置 + effort + harness + 工具/环境/verifier 的组合结果。
- 本轮快照：Artificial Analysis 中文首页 `/tmp/recheck-aa-1234.html` 为 1,773,814 bytes、SHA-256 `5f70a4b28d24ce6c560f83561ed0fe0f7150675b3e8a5a33ea649a238695e4ea3`；Opus 5 详情 `/tmp/recheck-aa-opus5-1234.html` 为 3,532,163 bytes、SHA-256 `c9626e404c0ff538a28bc58fff9f05bd276db64a1791b72bfa22ce7f4202d47f`；DataCurve `/tmp/recheck-deepswe-1234.html` 为 268,313 bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- Anthropic 官方模型总览、Opus 5 专属页和完整 Markdown 文档确认：`claude-opus-5`、2026-07-24、1M context、128K 普通/300K Batch 最大输出、文本/图像输入、adaptive thinking、默认 `high`、`low/medium/high/xhigh/max` 五档 effort、$5/$25 输入输出价格、512-token 最小缓存 prompt、Claude API/Bedrock/Vertex/Foundry 平台以及 2026-05 截止字段。文档还确认 thinking block 按 `content[].type` 解析、工具循环原样回传 thinking/signature、工具和 effort 中途变更、`fallbacks: "default"`、refusal 业务态和 web fetch 不支持。
- Anthropic 发布方声称 Frontier-Bench、CursorBench、ARC-AGI 3、Zapier AutomationBench、OSWorld 2.0 及内部生命科学任务有提升；这些保留为发布方数据。Frontier-Bench 脚注绑定内部运行、`mini-SWE-agent`、GKE、每任务 5 次尝试和安全拒答时向 Opus 4.8 fallback，不能与两个排行榜结果拼成裸模型能力。
- 官方 System Card 已下载登记：16,281,258 bytes，SHA-256 `0950dae1ba6b341e4f1a009e535e0f025625efeca21bd043a9a5ae148a3f2e6b`。当前环境无法稳定抽取 PDF 正文，因此不从目录或二进制字符串扩写安全数字。
- arXiv 精确标题检索返回 2 篇把 Opus 5 当被测模型的文章（arXiv:2608.14992、2608.07776），没有 Opus 5 专属技术报告；Anthropic Research 页面也没有 Opus 5 专属报告条目。该负面结果有日期和入口范围，不是否定未来发布。
- Claude Opus 5 当前为“资料级闭环”，不新增独立架构章节；研究笔记见 [`claude-opus-5-source-notes.md`](research/model-update-2026-09/claude-opus-5-source-notes.md)。当前锚点为 `claude-opus-5`，下一锚点为 `claude-fable-5.1`。

## 2026-09-15：Claude Fable 5.1 断点恢复、联网复验与闭环

- 针对“夜间 20:00—次日 09:00 可能发生代理中断”的风险，本轮重新发起联网搜索。`10.237.126.170:1234` 与 `10.24.27.134:8098` 对 Artificial Analysis 中文首页、Fable 5.1 详情页和 DataCurve 均返回 HTTP 200；7890 仍可访问部分页面，但对 Artificial Analysis 大页面有 TLS EOF/读取不稳定。没有把单代理失败解释成页面不存在。
- Artificial Analysis 新鲜详情页确认 `claude-fable-5-1` 的 `max + Default Fallback` 主配置、`releaseDate: 2026-09-01`、Intelligence Index `53.3737509623252`、输出速度 `65.9769929669683 tokens/s`、TTFT `212.0122070875s`、1M context、`$10/$50` 每百万 token和约 `$7.6297`/task；页面同时有 xhigh/high/medium/low fallback 变体。快照 `/tmp/recheck-aa-fable51-20260915.html` 为 3,615,956 bytes，SHA-256 `aa253dd4d4e8ad3285f60d0910f698239af5bfcefd268bb0127e41a74c93ba4b`；首页 `/tmp/recheck-aa-home-20260915.html` 为 1,773,775 bytes，SHA-256 `8589909c7199e311a750a2b3c0435e139508da4c6343ae063c935d0c3dbd1a98`。
- DataCurve DeepSWE v1.1 新鲜快照仍为 113 tasks、91 repositories、5 languages、`mini-swe-agent`，只有 `claude-fable-5` 的 effort 行，没有 `claude-fable-5-1`。因此不为 Fable 5.1 编造 DeepSWE 分数，也不把 Fable 5 的 316/452 结果迁移给它。快照 `/tmp/recheck-deepswe-20260915.html` 为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- Anthropic 发布页本次返回 HTTP 200，确认 Fable 5.1 与 Mythos 5.1 是同一 underlying model、不同 safeguards；Claude Code 默认 High，Claude Cowork/Claude.ai 默认 Medium；cache read 为 `$0.25/M`；典型工作负载成本约低 25%、高度 Agent 化工作负载最高约低 45%（均为发布方分析）；并公开 EFS 客户云数据边界、anti-distillation 历史编辑限制、自定义 GPU kernel/中间缓存科学案例和生产 safeguards 影响 benchmark 的说明。
- 发布方 benchmark 摘要已登记为自报结果：Terminal-Bench-Science 0.1 52.6%、Terminal-Bench 4.0 55.8%（Fable）/60.9%（Mythos）、GDPval-AA v2 1853、OSWorld 2.0 77.9% partial/41.7% strict、Humanity’s Last Exam 60.9% 无工具/65.0% 有工具、AutomationBench 31.4%、CursorBench 3.2.0 73.4%。这些数字绑定 safeguards、effort、工具和测试版本，不替代独立复现。
- System Card 重新下载成功：`/tmp/recheck-anthropic-fable51-system-card-page-20260915.bin`，16,397,488 bytes，SHA-256 `b0d59edc7a60eef32a879c13d713cce60c3fefd7e6b5183afdc8b835af3c8c39`。当前环境仍无稳定 PDF 正文提取器，未从二进制猜测安全数值。
- arXiv 标题精确查询 `"Claude Fable 5.1"` 返回 0 篇；全文查询返回 4 篇外部使用/评测论文（arXiv:2609.15597、2609.15494、2609.10420、2609.08847），没有 Fable 5.1 专属技术报告。Anthropic Research 页面可访问但未检出专属报告条目；详细标题和证据边界已写入研究笔记。
- 已同步 [`claude-fable-5.1-source-notes.md`](research/model-update-2026-09/claude-fable-5.1-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan.md`](plan.md) 和本文件。Fable 5.1 当前为“资料级闭环”，不新增独立架构章节；当前锚点切换为 `claude-fable-5.1`，下一锚点为 `claude-sonnet-5`。

## 2026-09-15：Claude Sonnet 5 资料级闭环与联网复核

- 考虑到用户工作时段可能造成夜间代理中断，本轮重新发起联网搜索并成功恢复 Sonnet 5 资料获取。`10.237.126.170:1234` 与 `10.24.27.134:8098` 对 Artificial Analysis、DataCurve 和 Anthropic 发布页均返回 HTTP 200；`10.24.27.134:7890` 可作为备用线路，但访问 Artificial Analysis 大页面仍有 TLS/读取超时。单个代理失败不被解释为页面不存在。
- Artificial Analysis 快照 `/tmp/recheck2-aa-sonnet5-20260915.html` 为 3,614,205 bytes，SHA-256 `3e8257d0efec85f2cc30bf78b31ed3e5fa10c0be4da6ec55413de7cae9b2d9bf`；DataCurve 快照 `/tmp/recheck2-deepswe-sonnet5-20260915.html` 为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；Anthropic 发布页快照 `/tmp/recheck2-anthropic-sonnet5-release-20260915.html` 为 264,609 bytes，SHA-256 `5cc40094a584f1821874847040cff912056845dac56bc770826d4dbcd11cb725`。System Card PDF、完整开发者文档、Research 和 arXiv 快照哈希已登记在研究笔记。
- Artificial Analysis 详情页确认 `claude-sonnet-5` 的 `max` 主配置及 `xhigh/high/medium/low/Non-reasoning` 变体；主配置 Intelligence Index 为 `38.3576962882576`、约 80.00 output tokens/s、约 202.63s TTFT、1M context、输入/输出 `$2/$10/M` 和约 `$5.0912`/task。DataCurve DeepSWE v1.1 仍为 113 tasks、91 repositories、5 languages、统一 `mini-swe-agent`；五档 Pass@1（max→low）为 53.846%/49.667%/48.230%/39.778%/30.512%，平均成本为 `$26.40/$11.89/$7.43/$4.08/$2.19`。这些数字绑定 effort、provider、harness、工具、环境、任务集和 verifier，不是裸模型排名。
- Anthropic 官方目录和发布页确认 `claude-sonnet-5`、2026-06-30、1M context、128K 普通/300K Batch 最大输出、文本/图像输入、Adaptive、默认 high、Fast、平台、价格和 2026-01 cutoff 字段。完整文档补齐的面试技术点包括：手动 `thinking.type: "enabled"` + `budget_tokens` 在 Sonnet 5 上返回 400；effort 是行为信号而非严格预算，`max_tokens` 是思考/工具/文本共享的硬上限；thinking blocks/signatures 要按异构 `content` 回放；新 tokenizer 约增加 30% token；context awareness、`compact-2026-01-12` server-side compaction、`computer_toolset_20260801` 和 programmatic tool calling 的长任务协议。
- System Card/发布页的 SWE-bench Verified 85.2%、SWE-bench Pro 63.2%、Multilingual SWE-bench 78.3%、Terminal-Bench 2.1 80.4%、BrowseComp 84.7%、OSWorld-Verified 81.2% 和 GDPval-AA v2 Elo 1618 单独作为 Anthropic 发布方结果记录。arXiv 精确标题 `"Claude Sonnet 5"` 截至本日返回 0 个结果，Anthropic Research 页面未检出 Sonnet 5 专属技术报告；参数、架构、完整训练/后训练 recipe、内部 adaptive-thinking 机制和独立 benchmark 复现仍待核验。
- Sonnet 5 当前已完成资料级闭环，不新增独立架构章节；研究笔记见 [`claude-sonnet-5-source-notes.md`](research/model-update-2026-09/claude-sonnet-5-source-notes.md)。下一模型候选必须继续从 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜的剩余重点厂商条目中选择。

## 2026-09-15：GLM-5.3-Flash 重新联网复验与内容专题闭环

- 针对用户提出的夜间工作时段可能造成网页访问中断，本轮重新发起联网复验。`10.237.126.170:1234` 与 `10.24.27.134:8098` 可获取 Artificial Analysis、DataCurve 和 Z.ai 官方页面；`10.24.27.134:7890` 作为备用线路，对 Artificial Analysis 大页面偶发 TLS EOF/读取超时。没有把单代理失败解释为页面不存在。
- Artificial Analysis 当前快照 `/tmp/glm53flash-aa-current-8098-20260915.html` 为 3,684,322 bytes，SHA-256 `7800ff202ced5e5cc170d7f1858d8070cf8d41c47b3ab6bace60b75c596b6319`；确认 `glm-5-3-flash`、`max`、页面 `releaseDate` `2026-08-26`、Intelligence Index `41.907366113455`、median output speed `114.22108687545 tokens/s`、median TTFT `2.45458272199994s` 和 1M context。输入/输出/cache hit 价格字段约为 `$0.15/$0.50/$0.026` 每百万 token，均是第三方配置字段。
- DataCurve 快照 `/tmp/glm53flash-deepswe-current-8098-20260915.html` 为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；`mini_swe_agent_glm_5_3_flash_max` 为 `max`、`n_runs=4`、284/448、Pass@1 `63.392857%`、Pass@4 `84.955752%`、平均成本 `$0.2409818562`、平均输出 `72829.77` token、平均 Agent steps `122.89`。这是 `mini-swe-agent` + 工具 + 任务集 + 环境 + verifier 的系统结果，不是裸模型分数。
- Z.ai 官方文档快照 SHA-256 `a127bf7eff2780aacebfc4ffdcadfac5820b75caeaafdb932da0c8942eee879f`，博客正文通过 JS 资源读取，SHA-256 `225196c63b5944629606d26982c0a43c4a8fbd6edb8a7a2e9bf8abfacd35fcbb`；固定 Hugging Face revision 为 `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`，README/config 哈希已写入研究笔记。
- 官方资料确认 320B total/18B activated、`Glm5NextForConditionalGeneration`、45 层、前 3 层 dense MLP、34 `linear_attention`/11 `deepseek_sparse_attention`、288 routed experts/top-8/1 shared、1M position、IndexPool（4-key weighted pooling）、mHC（`hc_mult=4`、20 次 Sinkhorn）和原生 video/image/text/file 输入。视觉 coding loop 以 observe—render/use—verify—refine 为主线；serving 线索包括 SGLang、ReplaySSM、W8A8、INT8/FP8/BF16 hybrid cache quantization、Layer Split 和 EPD。
- 已新增研究笔记 [`glm-5.3-flash-source-notes.md`](research/model-update-2026-09/glm-5.3-flash-source-notes.md) 和第二十一册第 84 章 [`glm-5.3-flash混合注意力与视觉闭环.md`](book-21-transformer-architecture-evolution/chapters/84-glm-5.3-flash混合注意力与视觉闭环.md)，并同步 `source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan.md`、第四册百科、第二十一册目录/时间线以及论文、题库、练习、术语、项目和知识图谱。
- 当前状态由“部分覆盖”升级为“内容专题闭环”。3.01×/4.44× attention/KV、约 3× serving、官方 benchmark 和视觉 workflow 数字均保留发布方自报边界；完整训练/后训练 recipe、生产 kernel、真实 state/KV/indexer/EPD bytes、硬件 profiling、线上 acceptance rate、API 错误/限流回归和独立 benchmark 仍待核验。下一锚点优先转向榜单已出现的 `deepseek-v4-flash-vision`，不从官方周边资料新增模型候选。

## 2026-09-15：DeepSeek V4 Flash Vision 断点恢复、专题验证与资料级闭环

- 针对夜间 20:00—次日 09:00 可能发生的代理中断，本轮重新发起当前时点复验。`10.237.126.170:1234` 与 `10.24.27.134:8098` 对 Artificial Analysis Vision 详情页均返回 HTTP 200、3,609,278 bytes，SHA-256 均为 `a3ab00557356d09aca3729c2fead3900459ab73b1c63d31b03e7c3267200ea68`；三个代理对 DataCurve 均返回 HTTP 200、268,313 bytes，SHA-256 均为 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；三个代理对 DeepSeek Quick Start 均返回 HTTP 200、46,116 bytes，SHA-256 均为 `6e2eb037db92ebef6a8f6408d87c12318c973388d6e27321606bb0e67dd67a6c`。`10.24.27.134:7890` 访问 Artificial Analysis 大页面仍 TLS EOF，但对另外两个入口成功；没有把代理失败解释成网页不存在。
- Artificial Analysis 当前锚点仍是 `deepseek-v4-flash-vision` 的 `max` 配置，`releaseDate` `2026-08-21`、1M context、284/13 目录参数、Intelligence Index `35.0122378035969`、约 `215.1792 tokens/s` 输出速度和 `1.2986s` median TTFT；DataCurve 当前没有 Vision 同名行，不迁移 V4 Flash/Pro 的 DeepSWE 分数。
- DeepSeek 官方实验公告、Vision/Files/Responses/价格文档已收口：历史 `deepseek-v4-flash-vision-exp` 的多模态 API、base64/URL/Files 输入和工具观察回灌，与当前 `deepseek-flash`/V4.1-Flash 路由分开记录；图像 detail/resize/token、`file_id` 生命周期、`function_call_output(input_image)`、兼容接口 capability probe 和端到端成本形成面试主线。历史公告的 384 image tokens 与当前 guide 的约 1024 tokens/image 保留日期、alias 和文档语境。
- 新增研究笔记 [`deepseek-v4-flash-vision-source-notes.md`](research/model-update-2026-09/deepseek-v4-flash-vision-source-notes.md) 与第二十一册第 85 章 [`deepseek-v4-flash-vision多模态api与路由账本.md`](book-21-transformer-architecture-evolution/chapters/85-deepseek-v4-flash-vision多模态api与路由账本.md)，并完成模型盘点、来源索引、榜单解释、第四册百科、第二十一册目录/时间线和书系配套同步。
- 第 85 章唯一 Python demo 已通过 AST、正常执行和 9 组边界测试；全库 679 个 Markdown 文件的 17,161 对围栏、1,704 个 Python 围栏 AST、932 个相对链接以及 `git diff --check` 均通过。全库链接审计使用平衡括号解析，避免把中文标点和代码内容误报为坏链接。
- DeepSeek V4 Flash Vision 当前由“部分覆盖”升级为“资料级闭环”。视觉专属参数/encoder、完整训练或后训练 recipe、独立技术报告、DataCurve Vision 结果、生产 kernel、目标硬件 profiling 和线上 acceptance rate 仍待核验；下一锚点从两个排行榜剩余重点条目中选择，不从 DeepSeek 官方目录额外发现模型。

## 2026-09-15：GPT-5.4 资料级闭环与夜间代理恢复复验

- 针对用户所说的夜间 20:00—次日 09:00 代理中断，本轮重新发起联网搜索。`10.24.27.134:8098`、`10.24.27.134:7890` 和 `10.237.126.170:1234` 对 Artificial Analysis 中文首页与 DataCurve DeepSWE 均返回 HTTP 200；三条线路取得的排行榜页面字节完全一致。AA `/zh` 快照为 `1,773,715` bytes、SHA-256 `4e089738feed2330ffce50ba7cd4d58141641e647ddbf0e50e1d8c4a3b975cf0`；DataCurve 快照为 `268,313` bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- GPT-5.4 详情页通过 `7890` 返回 HTTP 200，快照为 `3,488,457` bytes、SHA-256 `3438103fd097715e5a39d9ee7b3bc0710c23a251c013da3c1b9f8e05a35ada5e`；页面仍确认 `GPT-5.4 (xhigh)`、`releaseDate=2026-03-05`、Intelligence Index `38.9756`（estimated）、约 `143.44 tokens/s`、约 `93.69s` median TTFT、1.05M context，并标记 deprecated、指向 GPT-5.5。DataCurve 仍为 `mini_swe_agent_gpt_5_4_xhigh`：234/452、Pass@1 `51.7699%`、Pass@4 `77.8761%`、平均成本 `$5.6525`、平均输出约 `71,408.87` token、约 `70.47` steps；这些数字绑定 `mini-swe-agent`、工具、环境、任务集和 verifier，不是裸模型分数。
- OpenAI 官方模型页与专属指南本轮均实际抓取成功：模型页 `gpt-5.4-2026-03-05` 快照 SHA-256 为 `c30e86b38bc6ceacb3d6db269aa6ba4a09c3c5e1322bfaf90f924fddce4013a5`，指南快照 SHA-256 为 `61a8e21bea3bc4592dd4eb19383a577a292ee770ae10f2e982aa788c349c04be`。已核验 1,050,000 context、128,000 max output、`none`—`xhigh` effort、text/image → text、deferred `tool_search`、built-in computer use、native compaction、custom tools/CFG、`allowed_tools`、tool preambles、Responses `phase`、reasoning 状态回放、prompt caching 和 server-side compaction。
- GPT-5.4 当日记录为“资料级闭环”：GPT-5.4/Pro 的关联关系已记录，mini/nano 在 2026-09-15 当时仍是待独立核验的关联变体；该历史状态已由 2026-09-20 的 mini/nano 独立资料级闭环更新。没有公开参数规模、激活参数、层/专家结构、训练与后训练 recipe、system card、独立技术报告或完整 benchmark 复现，因此不新增 GPT-5.4 专属正式章节。研究笔记见 [`gpt-5.4-source-notes.md`](research/model-update-2026-09/gpt-5.4-source-notes.md)，当前后续仍从两个排行榜的剩余重点厂商候选中选择，不从 OpenAI 官方目录另发现模型。

## 2026-09-15：Gemini 3.7 Flash 断点恢复、联网复验与资料级闭环

- 针对用户所说的夜间 20:00—次日 09:00 可能造成代理中断，本轮重新发起联网搜索。沙箱内三条代理均无法连接；经受控联网重试后，`10.24.27.134:8098` 成功获取 Artificial Analysis 中文首页、DataCurve、DeepMind Model Card 和 DeepMind Research，`10.24.27.134:7890` 成功获取 Google AI Developers 模型页和 arXiv，`10.237.126.170:1234` 也成功获取 arXiv/DeepMind Research。7890 对 Artificial Analysis 大页面和部分 Google 页面存在超时，不能把单代理失败解释为网页不存在。
- Artificial Analysis high 详情页本轮复验为 3,605,373 bytes、SHA-256 `d0c5128baac580dbc983b403719f44730decf3bd6c3a24484d29633f13ffbe4c`；页面确认 `Gemini 3.7 Flash (high)`、`releaseDate: 2026-08-13`、Intelligence Index `39.4295316404896`、约 `292.3389 tokens/s`、约 `10.2161s` input/TTFT 字段、1M context、proprietary 且参数字段为空。输入/输出/cache hit 价格字段约为 `$0.75/$3.75/$0.075` 每百万 token；这些都是 Artificial Analysis 的第三方配置字段。
- DataCurve fresh 快照为 268,313 bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`，页面仍为 113 tasks、91 repositories、5 languages、统一 `mini-swe-agent`，包含 Gemini 3.7 Flash low/medium/high 四次运行配置。Pass@1 分别为 `53.7611%/65.4867%/65.2655%`，平均成本约 `$1.8323/$2.0251/$2.1763`；这些是配置 + harness + 工具 + 环境 + verifier 的系统结果，不是裸模型能力。
- Google AI Developers 模型页 fresh 快照 SHA-256 `53e8a497f27951f2d533efa4609a9a45e816e5bda697a4d7470489c88ed9243a`，确认模型 ID `gemini-3.7-flash`、输入 text/image/video/audio/PDF、text 输出、1,048,576 输入 token、65,536 输出 token、low/medium/high thinking、`minimal` 报错、caching/code execution/file search/function calling/Maps/Search grounding/structured outputs/URL context/Computer Use Preview，以及 Batch/Flex/Priority inference。
- DeepMind Model Card fresh 快照为 159,258 bytes、SHA-256 `b8051a19b7578abd1ab60e222f0afa829db38b415b66fbe91a24d6d89e3cdbc5`。官方只披露核心 reasoning foundation 的算法改进、agentic video understanding 和可调 thinking；架构、训练数据、硬件和软件信息指向 Gemini 3.6 Flash Model Card。没有把前代信息升级成 Gemini 3.7 独有事实。
- 已核验的面试技术线索包括：Interactions API 的 interaction/step、`previous_interaction_id`、store 保留/后台执行/implicit caching；tool context circulation、thought/tool signature 的 stateful/stateless 回放和 built-in/custom tool 执行边界；agentic video 的主动时间轴、transcript/帧/帧率/分辨率选择、`processing_call`/`processing_result` 以及 stateless processing step 回放。
- Google 官方评测 PDF 的发布方数字已单独记录：DeepSWE v1.1 65.3%、Terminal-Bench 2.1 85.8%、LVBench 85.4%、GDM-MRCR v2 128K 97.0%、OSWorld-2.0 47.9%。安全边界记录为 cybersecurity 达到 alert threshold 但未达到 CCL，CBRN 未达到 TCL/CCL；这些是 Google 自报安全/评测资料，不与 DataCurve 或 Artificial Analysis 拼接。
- arXiv 精确标题查询 [`title:"Gemini 3.7 Flash"`](https://arxiv.org/search/?query=%22Gemini+3.7+Flash%22&searchtype=title) 返回 0 个结果，快照 SHA-256 `a92b072d2026186568d09e3e42129649ee852886d6572d278cbb529f57b217a2`；全文查询找到 3 篇外部使用/评测论文：arXiv:2609.15983、2609.05232、2608.20563。它们分别研究多 Agent 研究 harness、执行资源契约和长时程安全 Agent 失败诊断，不是 Gemini 3.7 专属技术报告。
- 新增 [`gemini-3.7-flash-source-notes.md`](research/model-update-2026-09/gemini-3.7-flash-source-notes.md)，同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`plan.md`](plan.md) 和本进度表。Gemini 3.7 Flash 当前为“资料级闭环”，暂无独立正式章节；当前锚点切换为 `gemini-3.7-flash`，下一锚点继续从两个排行榜的剩余重点厂商条目中选择。

## 2026-09-15：Gemini 3.6 Flash 断点恢复、联网复验与资料级闭环

- 针对用户所说的夜间 20:00—次日 09:00 可能导致代理中断，本轮重新发起联网搜索。三条代理对 Artificial Analysis 中文首页均返回 HTTP 200；随后通过可用线路重新获取 Gemini 3.6 详情/provider 页、DataCurve、Google AI Developers 模型页、DeepMind Model Card、Thinking、Interactions、工具组合、视频理解、Context caching、Computer Use 和 arXiv 检索页。个别代理的瞬时连接失败没有被解释为页面不存在。
- Artificial Analysis 详情页快照 `/tmp/gemini36-recheck-aa-8098-20260915.html` 为 3,600,704 bytes，SHA-256 `362fd94a45955783974462edb3cdc99ed932625a56d8c0afe7b9bca4a7f2e721`；页面确认 `Gemini 3.6 Flash (high)`、`releaseDate: 2026-07-21`、Intelligence Index `34.3395678920714`、约 `192.7337 tokens/s`、TTFT `19.2239s`、1M context、约 `$0.15/$0.75/$3.75` cache-hit/input/output 每百万 token。provider 页当前只展示 `Google AI Studio` 一个 benchmark provider，不能与详情页 FAQ 的 provider 可用性数量混读。
- DataCurve 快照 `/tmp/gemini36-recheck-ds-8098-20260915.html` 为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；原始行是 `mini_swe_agent_gemini_3_6_flash_high`、`high`、4 runs、211/452、Pass@1 `46.6814%`、Pass@4 `75.2212%`、平均成本 `$2.2095066`、平均输出 `95,844.86` token、平均 Agent steps `116.73`、median peak context `151,398.5`。这是配置 + harness + 工具 + 环境 + verifier 的系统结果，不是裸模型分数。
- Google AI Developers 模型页快照 `/tmp/gemini36-recheck-ai-7890-20260915.html` 为 104,839 bytes，SHA-256 `e0d55ee75ca1c4e80c128284a907de9e2dd602d9a6a39e660599abea1bf43404`；确认 `gemini-3.6-flash`、text/image/video/audio/PDF 输入、text 输出、1,048,576 输入 token、65,536 输出 token、caching/code execution/File Search/function calling/Maps/Search grounding/structured outputs/thinking/URL context/Computer Use Preview，以及 Batch/Flex/Priority inference；稳定 alias 为 `gemini-3.6-flash`，latest update 为 July 2026。
- Thinking 文档明确 3.6 的默认档位为 `medium`，支持 `minimal/low/medium/high`；Interactions 文档模型表列出 3.6；视频文档将 3.6 列入 agentic video 模型，Context caching 文档给出 implicit caching 最低输入 `4,096` tokens。Gemini 3 工具组合文档记录 thought/tool signatures 与 tool context circulation；这些是 API/Agent 协议事实，不是内部 Transformer 机制。
- DeepMind Model Card 快照 `/tmp/gemini36-recheck-card-8098-20260915.html` 为 154,640 bytes，SHA-256 `c2e25fec9cc337856c4f612b729f8d1b3985a0627a7234e66091b53dddd9e1c3`。官方明确 3.6 based on Gemini 3.5 Flash，架构、训练数据、数据处理、硬件和软件资料指向 3.5 Model Card；公开 benchmark 包括 SWE-Bench Pro `58.7%`、DeepSWE v1.1 `49%`、Terminal-Bench 2.1 `78.0%`、GDPVal-AA v2 `1421`、OSWorld-Verified `83.0%`、GDM-MRCR v2 128K `91.8%`/1M `54.0%`。安全表和 Frontier Safety 结论按发布方/自动评测边界记录，未迁移成 3.6 独有训练事实。
- arXiv 精确标题检索快照 `/tmp/gemini36-recheck-arxiv-title-7890-20260915.html` 为 16,452 bytes、SHA-256 `03b6d5dabe226f9c165a6f725f670f2f846659fcc7308c037b9c0a4fe8db9127`，返回 0 篇；全文检索快照 `/tmp/gemini36-recheck-arxiv-all-7890-20260915.html` 为 63,191 bytes、SHA-256 `2e5c93fde5c302f8a1cd655425d9df2e885c31f4a904318c90f00db6ef372ffa`，返回 9 篇外部使用/评测论文，不是 Gemini 3.6 专属技术报告。9 篇的标题、摘要要点和证据边界已写入研究笔记。
- 新增 [`gemini-3.6-flash-source-notes.md`](research/model-update-2026-09/gemini-3.6-flash-source-notes.md)，并同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan.md`](plan.md) 和本进度表。Gemini 3.6 Flash 当前由“仅候选”升级为“资料级闭环”，暂无独立正式章节；当前锚点为 `gemini-3.6-flash`，下一锚点仍从两个排行榜剩余重点条目选择。

## 2026-09-15/16：Gemini 3.5 Flash 断点恢复、联网复验与资料级闭环

- 针对用户提出的 20:00—次日 09:00 代理中断规则，本轮重新发起联网收集。2026-09-15 工作时段重新抓取时，`10.237.126.170:1234` 与 `10.24.27.134:8098` 对 Gemini 3.5 Flash 的 Artificial Analysis 与 DataCurve 均返回 HTTP 200，两个代理取得的 AA 详情和 DataCurve 快照逐字节一致；7890 成功获取 Google AI Developers、DeepMind Model Card 和 Google API 周边文档。2026-09-16 轻量复探中，三个代理对 AA/DataCurve 均返回 HTTP 200，7890 对 Google 模型页返回 HTTP 200，8098/1234 对该大页面超时；代理差异仍不被解释成页面不存在。
- Artificial Analysis 快照为 3,607,081 bytes、SHA-256 `abdbed0cf8069ae81c272aea386724302a0659235fa5b950980a9a9a0d162ec7`；页面确认 `Gemini 3.5 Flash (high)`、第三方 `releaseDate: 2026-05-19`、Intelligence Index `32.9816033695905`、约 `221.8080 tokens/s`、约 `18.3631s` TTFT、1M context 和约 `$1.5625`/task。当前页面的 `deprecated: true`/`deprecatedTo: gemini-3-6-flash` 只作为 Artificial Analysis 目录状态记录，不能写成 Google 官方退役公告。
- DataCurve 快照为 268,313 bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`，仍为 113 tasks、91 repositories、5 languages、4 runs、统一 `mini-swe-agent`。`mini_swe_agent_gemini_3_5_flash_high` 原始行是 163/452、Pass@1 `36.0619%`、Pass@4 `63.7168%`、平均成本 `$3.4467`、平均输出 `75,730.19` token、平均 Agent steps `105.30`；这些是配置 + harness + 工具 + 环境 + verifier 的系统结果。
- Google 官方 API 页面确认 1,048,576 输入 token、65,536 输出 token、text/image/video/audio/PDF 输入、text 输出、caching/code execution/File Search/function calling/Maps/Search grounding/structured outputs/URL context、Computer Use Preview、Batch/Flex/Priority inference；What's New 页面补充 GA、默认 effort 从 high 改为 medium、`minimal/low/medium/high`、legacy `thinking_budget` 互斥、thought preservation 和 2025 年 1 月 knowledge cutoff。
- Interactions API 模型表明确支持 `gemini-3.5-flash`；工具组合文档的 `id`/`signature`、tool context circulation、stateful/stateless 回放和宿主执行责任已入库。Context caching 文档明确 implicit caching 最低输入 `4,096` tokens。Computer Use 文档明确 3.5 Flash 支持 opt-in screenshot prompt-injection detection，默认关闭。
- 视频证据保持保守：API 支持视频输入，但当前视频文档的 agentic processing 列表明确列出 3.8/3.7/3.6 和 3.5 Flash-Lite，没有明确列出 3.5 Flash；因此不把 3.6 的 `processing_call`/`processing_result` agentic video 结论迁移给 3.5 Flash。
- DeepMind Model Card 的发布方 benchmark、安全表和 Frontier Safety 结论已记录；Model Card 明确 architecture、training dataset、data processing、hardware 和 software 均指向 Gemini 3 Flash Model Card。arXiv 精确标题检索返回 0 篇，全文检索返回 30 篇外部使用/评测论文，没有检出 Gemini 3.5 Flash 专属技术报告。
- 新增 [`gemini-3.5-flash-source-notes.md`](research/model-update-2026-09/gemini-3.5-flash-source-notes.md)，同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan.md`](plan.md) 和本进度表。Gemini 3.5 Flash 当前由“仅候选”升级为“资料级闭环”，暂无独立正式章节；下一锚点优先选择 `gemini-3.1-pro-preview`，仍必须先由两个排行榜确认。

## 2026-09-15 Claude Opus 4.8 资料级闭环与网络复验

Opus 4.8 已完成从两个排行榜发现到官方资料、研究笔记和配套同步的资料级闭环。Artificial Analysis 详情页仍保留 `claude-opus-4-8` 历史条目，并标记为 deprecated、指向 Opus 5；DataCurve DeepSWE v1.1 同时保留 `xhigh` 和 `max` 配置。新鲜网络复验中，`10.237.126.170:1234` 对 Artificial Analysis `/zh`、Opus 4.8 详情页和 DataCurve 均返回 HTTP 200；`10.24.27.134:8098` 对两个排行榜也均返回 HTTP 200。复验快照哈希分别为 Artificial Analysis `/zh` `264357ee84d21d14f383cc7b179c34843ab39ef45c5bafad397946c7fadc23fd`、DataCurve `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7` 和 Opus 4.8 详情页 `369ef837ec32f3de7b59c8f6fdc7657b8608a22ccff0717c75c6cf3c0167c522`。

Opus 4.8 的公开面试主线是 Dynamic Workflows 的外部编排、并行 subagents、独立验证和断点恢复；`high`/`xhigh`/`max` effort 的成本—成功率—延迟对照；任务中途 system entries 对权限、预算、环境状态和缓存的影响；以及把模型的诚实性声明转成测试、lint、类型检查和领域 verifier 的 artifact 门禁。Anthropic 没有公开参数规模、网络结构、训练配方或独立可复现技术报告，因此这些空白保持为待核验，不能由产品行为反推。

当前锚点为 `kimi-k2.7-code`，已完成资料级闭环；下一锚点切换为 `grok-4.5`。后续仍只从两个排行榜核对 Grok 4.5，再沿 xAI 官方资料追踪论文、技术文档、模型页、API/Agent 协议和评测边界。

## 2026-09-15 Claude Fable 5.1 断点恢复与闭环

昨晚可能受工作时段代理中断影响，本轮重新发起联网搜索。`10.237.126.170:1234` 与 `10.24.27.134:8098` 对 Artificial Analysis Fable 5.1、Artificial Analysis 中文首页和 DataCurve 均返回 HTTP 200；7890 继续作为备用代理，但对 Artificial Analysis 大页面有 TLS EOF。新鲜抓取结果已经写入 [`claude-fable-5.1-source-notes.md`](research/model-update-2026-09/claude-fable-5.1-source-notes.md)。

本轮补证完成的重点是：Artificial Analysis 的 `max + Default Fallback` 配置与 53.3738 指数、65.98 tokens/s、212.01s TTFT 和 `$7.63`/task；DataCurve 没有 Fable 5.1 行；Anthropic 发布页对同一 underlying model/不同 safeguards、High/Medium 默认入口、`$0.25/M` cache read、EFS、anti-distillation 和发布方 benchmark 的说明；以及标题精确为 0、全文为 4 篇的 arXiv 定向检索。System Card 已重新下载并登记哈希，但 PDF 正文仍未稳定提取。

Fable 5.1 现在标为“资料级闭环”，不新增独立架构章节。官方没有公开参数规模、稠密/MoE 架构、完整训练/后训练配方、独立技术报告或独立 benchmark 复现；发布页 benchmark 和科学工作流只能作为发布方证据。下一锚点切换为 `claude-sonnet-5`，继续按“榜单配置 → 官方发布/模型页 → API/Agent 文档 → System Card/论文 → 书系映射”的顺序推进。

## 2026-09-15 Kimi K2.7 Code 资料级闭环与联网复验

本轮在用户提供的代理恢复后重新请求两个排行榜和 Kimi 官方资料。`10.237.126.170:1234` 对 Artificial Analysis K2.7 详情页、DataCurve 和 Kimi 资源/API 页面返回 HTTP 200；Hugging Face 固定 revision 页面通过 `10.24.27.134:7890` 返回 HTTP 200。三条代理对两个排行榜的轻量连通性均已复验成功，但 7890 对大页面曾出现读取超时，8098/1234 更适合完整抓取。

Artificial Analysis 详情页快照为 3,605,694 bytes，SHA-256 `e37ce8e8f224233a95ca93771fe9f2130e544ef071703bae475295bd1ac527db`；DataCurve 快照为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。DeepSWE 行为配置是 `mini-swe-agent`、`reasoning_effort:null`、138/452、Pass@1 约 30.53%、Pass@4 约 61.06%、平均成本约 `$2.8155` 和平均 149.1 steps；这些是系统组合结果。

官方 Hugging Face revision `74797c9c62378b951a1f6fcf5c4631024e9b8bef` 的模型卡和 `config.json` 确认 1T/32B active MoE、61 层、384 routed experts、top-8、1 shared expert、MLA、256K/262,144 context、MoonViT 400M 和 native INT4 路线。Kimi API 快速开始确认 always-on thinking、`preserve_thinking`、固定采样参数、`tool_choice` 限制、reasoning_content 回传和多步工具调用协议；部署指南覆盖 vLLM、SGLang 和 KTransformers。

本轮没有找到 Kimi K2.7 Code 专属 arXiv 论文或独立技术报告：官方资源页和固定 revision README 没有提供此类链接，定向 arXiv 检索未返回专属结果，GitHub 官方组织 API 对同名仓库查询为零结果。该负面证据只表示本轮公开入口未检出，不证明未来不会发布报告。Kimi Linear、Attention Residuals、Mooncake 和 K2 Thinking 继续作为关联路线单独记录，不能改写成 K2.7 Code 专属训练事实。

研究笔记：[`kimi-k2.7-code-source-notes.md`](research/model-update-2026-09/kimi-k2.7-code-source-notes.md)。当前不新增正式章节；后续锚点为 `grok-4.5`。

## 2026-09-15 Grok 4.6 资料级闭环与联网复验

Grok 4.6 已完成从两个排行榜发现到 xAI 官方资料、研究笔记和配套同步的资料级闭环。Artificial Analysis 详情页确认 `Grok 4.6 (high)` 及 low/medium/xhigh 配置；DataCurve DeepSWE v1.1 确认四档 `mini_swe_agent_grok_4_6_*` 配置。四档 DataCurve Pass@1 为 41.648%/67.478%/65.188%/66.741%，成本为 `$1.0424/$3.4490/$4.3849/$5.4977`，均绑定统一 `mini-swe-agent`、工具、任务集、环境和 verifier。

xAI 发布公告确认 2026-08-12 发布，并公开披露更长 supplemental training、模型生成 reasoning/技术数据、高质量工程数据、改进 optimizer/training recipe、Grok 4.5 生成的多 effort/多 harness SFT trajectories、model-based checks、面向知识工作/编码/领域环境的 agentic RL 和长轨迹 self-testing/verification。官方模型页确认 500K context、2026-02-01 knowledge cutoff、文本/图像输入、`low/medium/high/xhigh`、Responses/Chat Completions、function calling、structured outputs、Web/X Search、Code Execution、`prompt_cache_key`、`x-grok-conv-id` 和 compaction 建议。上述内容不支持参数、架构或完整训练 recipe 推断。

联网复验已重新发起并成功获取所需网页：`10.237.126.170:1234` 和 `10.24.27.134:8098` 对 Artificial Analysis Grok 4.6、DataCurve 均返回 HTTP 200；`10.24.27.134:7890` 对 xAI 官方模型页返回 HTTP 200。轻量复验中 `1234` 访问 xAI 曾返回代理 503，7890 访问 Artificial Analysis 偶发 TLS EOF，因此继续把代理失败与“页面不存在”区分开。官方模型页 HTML 快照为 408,905 bytes，SHA-256 `9669e2c5e96dafb74296f6e11af7c3b0fc74e8dec58a3d674edf18154c1caf27`。

arXiv 精确标题检索本轮未检出 Grok 4.6 专属论文或独立技术报告；这是当前公开入口的负面证据，不是对未来发布的绝对否定。研究笔记见 [`grok-4.6-source-notes.md`](research/model-update-2026-09/grok-4.6-source-notes.md)；下一步切换到 `claude-opus-5`，不新增 Grok 4.6 专属架构章节。

## 2026-09-15 Claude Sonnet 5 资料级闭环与后续顺序

考虑到用户工作时段可能造成夜间代理中断，本轮重新使用可用代理复核 Sonnet 5。`10.237.126.170:1234` 与 `10.24.27.134:8098` 对 Artificial Analysis、DataCurve 和 Anthropic 发布页均返回 HTTP 200；`10.24.27.134:7890` 仅作为备用线路，访问 Artificial Analysis 大页面仍出现 TLS/读取不稳定。当前不能把单个代理的失败解释为网页不存在。

本轮已完成 Sonnet 5 的资料级闭环：Artificial Analysis 主配置 `max` 的第三方指数、速度、TTFT、成本和 1M context；DataCurve v1.1 五档 effort 的 Pass@1、Pass@4、成本、输出 token 和 Agent steps；Anthropic 模型目录/发布页的模型与产品字段；完整开发者文档中的 adaptive thinking、effort、thinking block/signature、新 tokenizer、context awareness、server-side compaction、computer toolset、web/programmatic tool calling 和参数限制；System Card/发布方评测；以及 arXiv 精确标题和 Anthropic Research 负检索。

本轮明确保留三类边界：第一，Sonnet 5 的 effort/Non-reasoning 行是运行配置，不是不同基础模型；第二，Artificial Analysis、DeepSWE 和 Anthropic 自报 benchmark 不互相拼接为裸模型能力；第三，公开资料没有参数规模、稠密/MoE 结构、完整训练/后训练 recipe、内部 adaptive-thinking 机制或独立技术报告。故 Sonnet 5 暂不新增独立架构章节，内容映射到第四、六、七、十七、二十和二十四册。

下一锚点继续从两个排行榜中剩余的重点厂商候选选择；不因 Anthropic 官方目录中的 Haiku 4.5 或其他只在官方页面出现的名称新增候选。已闭环模型后续只补版本变更、真实 API 行为、完整 kernel/硬件 profiling、线上接受率和独立评测等证据缺口。

## 2026-09-15 DeepSeek V4 Flash Vision 资料级闭环与联网恢复复验

`deepseek-v4-flash-vision` 已完成从 Artificial Analysis 锚点到 DeepSeek 官方实验发布、当前 API 文档、研究笔记和第二十一册第 85 章的资料级闭环。历史实验 alias、当前服务路由和榜单配置被分开记录，避免把 API 产品字段误写成视觉模型内部架构。

本次针对夜间代理中断再次完整抓取：`10.237.126.170:1234` 与 `10.24.27.134:8098` 对 Artificial Analysis Vision 详情页均返回 HTTP 200、3,609,278 bytes，SHA-256 均为 `a3ab00557356d09aca3729c2fead3900459ab73b1c63d31b03e7c3267200ea68`；三个代理对 DataCurve 均返回 HTTP 200、268,313 bytes，SHA-256 均为 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；三个代理对 DeepSeek Quick Start 均返回 HTTP 200、46,116 bytes，SHA-256 均为 `6e2eb037db92ebef6a8f6408d87c12318c973388d6e27321606bb0e67dd67a6c`。`10.24.27.134:7890` 访问 Artificial Analysis 大页面仍发生 TLS EOF，不能据此判定页面不存在或内容不可用。

Artificial Analysis 当前仍确认 `deepseek-v4-flash-vision` 的 `max` 配置、`releaseDate` `2026-08-21`、1M context、284/13 目录参数、配置级 Intelligence Index `35.0122378035969`、约 `215.1792 tokens/s` 输出速度和 `1.2986s` median TTFT。DataCurve 当前没有 Vision 同名行，不能迁移 V4 Flash/Pro 的 DeepSWE 结果。

官方资料已收口为四条面试主线：一是 `requested_model`、排行榜 `catalog_anchor` 与响应 `served_model` 的三身份账本；二是图像 detail/resize、视觉 token、文本/工具/历史共同预算；三是 Files API `file_id` 的资源生命周期与 Responses `function_call_output(input_image)` 的观察回灌；四是 OpenAI/Anthropic 兼容接口的 capability probe、静默忽略字段、端到端成本和外部 verifier 门禁。实验公告的 384 image tokens 与当前 guide 的约 1024 tokens/image 保留各自日期、alias 和文档语境。

第 85 章 demo 已完成 AST 检查、正常执行和 9 组边界测试；教学估算不冒充生产 tokenizer、视觉 encoder 或计费公式。已同步模型盘点、来源索引、榜单解释、第四册百科、第二十一册目录/时间线和书系配套文件。

当前状态为“资料级闭环”，但视觉专属参数/encoder、完整训练或后训练 recipe、独立技术报告、DataCurve Vision 结果、生产 kernel、目标硬件 profiling 和线上 acceptance rate 仍待核验。下一步切换到两个排行榜中剩余的重点候选；不从 DeepSeek 官方目录或这次研究笔记额外发现新模型，已闭环锚点只补证据缺口。

## 2026-09-15 GPT-5.4 资料级闭环与夜间代理恢复复验

本轮针对用户所说的夜间 20:00—次日 09:00 代理中断重新发起联网搜索。`10.24.27.134:8098`、`10.24.27.134:7890` 和 `10.237.126.170:1234` 对 Artificial Analysis 中文首页与 DataCurve DeepSWE 均返回 HTTP 200；三条线路取得的排行榜页面字节完全一致。AA `/zh` 快照为 `1,773,715` bytes、SHA-256 `4e089738feed2330ffce50ba7cd4d58141641e647ddbf0e50e1d8c4a3b975cf0`；DataCurve 快照为 `268,313` bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。

GPT-5.4 详情页通过 `7890` 返回 HTTP 200，快照为 `3,488,457` bytes、SHA-256 `3438103fd097715e5a39d9ee7b3bc0710c23a251c013da3c1b9f8e05a35ada5e`；页面仍确认 `GPT-5.4 (xhigh)`、`releaseDate=2026-03-05`、Intelligence Index `38.9756`（estimated）、约 `143.44 tokens/s`、约 `93.69s` median TTFT、1.05M context，并标记 deprecated、指向 GPT-5.5。DataCurve 仍为 `mini_swe_agent_gpt_5_4_xhigh`：234/452、Pass@1 `51.7699%`、Pass@4 `77.8761%`、平均成本 `$5.6525`、平均输出约 `71,408.87` token、约 `70.47` steps；这些数字绑定 `mini-swe-agent`、工具、环境、任务集和 verifier，不是裸模型分数。

OpenAI 官方模型页与专属指南本轮均实际抓取成功：模型页 `gpt-5.4-2026-03-05` 快照 SHA-256 为 `c30e86b38bc6ceacb3d6db269aa6ba4a09c3c5e1322bfaf90f924fddce4013a5`，指南快照 SHA-256 为 `61a8e21bea3bc4592dd4eb19383a577a292ee770ae10f2e982aa788c349c04be`。已核验 1,050,000 context、128,000 max output、`none`—`xhigh` effort、text/image → text、deferred `tool_search`、built-in computer use、native compaction、custom tools/CFG、`allowed_tools`、tool preambles、Responses `phase`、reasoning 状态回放、prompt caching 和 server-side compaction。

GPT-5.4 当日记录为“资料级闭环”：GPT-5.4/Pro 的关联关系已记录，Pro 未单独建立完整资料档案，mini/nano 在 2026-09-15 当时仍为关联候选；该历史状态已由 2026-09-20 的 mini/nano 独立核验更新。没有公开参数规模、激活参数、层/专家结构、训练与后训练 recipe、system card、独立技术报告或完整 benchmark 复现，因此不新增 GPT-5.4 专属正式章节。研究笔记见 [`gpt-5.4-source-notes.md`](research/model-update-2026-09/gpt-5.4-source-notes.md)，后续仍从两个排行榜的剩余重点厂商候选中选择，不从 OpenAI 官方目录另发现模型。

## 2026-09-15 Gemini 3.7 Flash 资料级闭环与后续顺序

考虑到用户工作时段可能造成夜间代理中断，本轮重新复验 Gemini 3.7 Flash 的网页资料。沙箱内三条代理均无法直连代理端口；受控联网重试后，8098 成功获取 Artificial Analysis、DataCurve、DeepMind Model Card 和 Research，7890 成功获取 Google AI Developers 与 arXiv，1234 成功获取 arXiv/DeepMind Research。7890 对部分大页面仍有超时，单个代理失败不解释成资料不存在。

Gemini 3.7 Flash 已完成资料级闭环：两个排行榜的 high/medium/low 配置、Google AI Developers 模型页、DeepMind Model Card、官方评测/安全入口、Thinking/Interactions/工具/视频文档和 arXiv 精确/全文检索均已入库。面试主线是 1M 输入与 65K 输出预算、thinking budget、interaction/step/state/background、tool context circulation、加密 signature 的 stateful/stateless 回放、built-in/custom tool 责任边界、agentic video 主动时间轴浏览以及 `processing_call`/`processing_result`。

证据边界保持严格：Model Card 只披露核心 reasoning foundation 的算法改进、agentic video 和可调 thinking，架构/训练/软硬件资料指向 Gemini 3.6；没有 Gemini 3.7 独立参数、完整训练 recipe 或官方专属技术报告。arXiv 精确标题返回 0 个结果，全文检索的 3 篇论文只是外部使用/评测，不是 Google 模型报告。因此不新增 Gemini 3.7 专属架构章节，技术映射到 Reasoning、Agent/工具协议、视频多模态、长上下文/Serving、评测与安全章节。

当前锚点切换为 `gemini-3.7-flash`；下一锚点继续从 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜剩余的重点厂商条目中选择，不从 Gemini 3.6 前代资料、DeepMind 官方目录或外部论文另发现模型。

## 2026-09-15 Gemini 3.6 Flash 资料级闭环与联网恢复复验

昨晚可能受 20:00—次日 09:00 工作时段代理中断影响，本轮重新请求并确认三个用户提供的代理均可访问榜单入口；随后使用可用线路重新取得 Gemini 3.6 的 Artificial Analysis 详情/provider 页、DataCurve、Google AI Developers、DeepMind Model Card、Thinking/Interactions/工具组合/视频/缓存/Computer Use 文档和 arXiv 检索页。单一代理偶发连接失败仍不作为页面不存在的证据。

Gemini 3.6 Flash 已完成从两个排行榜锚点到官方资料的“资料级闭环”。Artificial Analysis high 详情页记录 `releaseDate` `2026-07-21`、Intelligence Index `34.3395678920714`、约 `192.7337 tokens/s`、TTFT `19.2239s` 和 1M context；DataCurve 的 `mini-swe-agent` high 行为 211/452、Pass@1 `46.6814%`、Pass@4 `75.2212%`、平均成本 `$2.2095`、平均 116.73 Agent steps。这些仍分别绑定第三方测量和 DataCurve harness/工具/环境/verifier。

Google 官方模型页确认 `gemini-3.6-flash`、1,048,576 输入 token、65,536 输出 token、text/image/video/audio/PDF 输入、text 输出、默认 medium 与 `minimal/low/medium/high` thinking、caching/code execution/File Search/function calling/grounding/structured outputs/URL context、Computer Use Preview 以及 Batch/Flex/Priority inference。官方文档还明确把 Gemini 3.6 列入 Interactions、agentic video 和 4,096-token implicit caching 支持表；Gemini 3 工具组合的 thought/tool signatures 和 tool context circulation 已作为协议层技术记录。

DeepMind Model Card 明确 3.6 based on Gemini 3.5 Flash，架构/训练数据/数据处理/硬件/软件资料指向 3.5 Model Card；已收录 Model Card benchmark、安全评估和 Frontier Safety 边界，但不把前代资料迁移为 3.6 独有架构。arXiv 精确标题检索返回 0 篇，全文检索 9 篇均为外部使用/评测论文，不是 Gemini 3.6 技术报告。

当前不新增 Gemini 3.6 专属正式章节；完整研究笔记、9 篇论文标题与摘要要点、快照哈希和待核验项见 [`gemini-3.6-flash-source-notes.md`](research/model-update-2026-09/gemini-3.6-flash-source-notes.md)。后续仍只能从 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜选择下一个重点锚点，不从 Gemini 3.5 Model Card、官方目录或外部论文另发现模型。

## 2026-09-15/16 Gemini 3.5 Flash 断点恢复与资料级闭环

昨晚 20:00—今早 09:00 的联网中断风险按“重新收集”处理：2026-09-15 工作时段通过 `10.237.126.170:1234` 与 `10.24.27.134:8098` 重新获取 Artificial Analysis Gemini 3.5 详情页和 DataCurve，两个代理返回的榜单快照逐字节一致；7890 获取 Google 官方页面。2026-09-16 轻量复探中，三个代理对 AA/DataCurve 均返回 HTTP 200，7890 对 Google API 模型页返回 HTTP 200，8098/1234 对该大页面超时；后续仍按站点选择代理，单代理失败不写成页面不存在。

Gemini 3.5 Flash 已完成资料级闭环。Artificial Analysis 的 `high` 详情页字段为第三方 `releaseDate` `2026-05-19`、Intelligence Index `32.9816033695905`、约 `221.8080 tokens/s`、TTFT `18.3631s`、1M context 和约 `$1.5625`/task；DataCurve `mini_swe_agent_gemini_3_5_flash_high` 为 163/452、Pass@1 `36.0619%`、Pass@4 `63.7168%`、平均成本 `$3.4467`、约 105.30 steps。两者均绑定配置、provider/harness、工具、任务环境和 verifier，不能写成裸模型能力；AA 当前 deprecated/指向 3.6 只保留为第三方目录字段。

Google API/What's New/Model Card 与周边文档确认 1M/65K、默认 medium 和四档 thinking、默认 thought preservation、Interactions、tool context circulation、signature/id 回放、4,096-token implicit caching、Computer Use Preview 及 screenshot prompt-injection detection。视频文档的 agentic processing 支持列表列出 3.5 Flash-Lite、3.6、3.7、3.8，但没有明确列出 3.5 Flash，因此不迁移 3.6 的 agentic video 结论。Model Card 把 architecture、training dataset、data processing、hardware 和 software 指向 Gemini 3 Flash Model Card；arXiv 标题精确检索为 0，全文检索为 30 篇外部使用/评测论文，没有发现独立 3.5 技术报告。

当前不新增 Gemini 3.5 专属正式章节，内容映射到 Reasoning、Agent/工具协议、Computer Use、长上下文/Serving、评测与安全章节。下一锚点优先为 `gemini-3.1-pro-preview`，但仍必须先在 Artificial Analysis 或 DataCurve 中确认其榜单锚点；不得仅因官方目录或 3.5/3.6 前代资料自动新增候选。

## 2026-09-16：Gemini 3.1 Pro Preview 断点恢复与资料级闭环

- 按用户设定的夜间 20:00—次日 09:00 中断规则，本轮重新收集 Artificial Analysis 中文首页、Gemini 3.1 Pro 详情页和 DataCurve DeepSWE。三条代理均返回 HTTP 200；AA 首页、详情页和 DataCurve 的同一页面快照逐字节一致。随后通过可用线路重新获取 Google API 模型页、Release Notes、DeepMind Model Card、评测方法、Thinking、thought signatures、工具组合、Long context、Caching 和 arXiv 检索页。
- Artificial Analysis 详情快照 `/tmp/overnight-aa-g31-1234-20260916.html` 为 `3,607,783` bytes，SHA-256 `ba842f82908f6237acd84c1ed0d1ab03d4652af503c8de6d00b56bb21b7731eb`；确认 `gemini-3-1-pro-preview`、第三方 `releaseDate` `2026-02-19`、Intelligence Index `30.3596656132261`、约 `108.4052 tokens/s`、约 `24.6088s` TTFT 和 1M context。
- DataCurve 快照 `/tmp/overnight-deepswe-1234-20260916.html` 为 `268,313` bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；确认 `mini_swe_agent_gemini_3_1_pro_preview_high`、`reasoning_effort: high`，53/452、Pass@1 `11.7257%`、Pass@4 `28.3186%`、平均成本约 `$2.1434`、平均输出 `28,368.88` token、平均 Agent steps `75.56`。这些结果绑定 `mini-swe-agent`、工具、任务集、环境和 verifier，不是裸模型能力。
- Google 官方资料确认 2026-02-19 发布、1,048,576 输入 token、65,536 输出 token、text/image/video/audio/PDF 输入、thinking、function calling、grounding、structured outputs 和 context caching；`gemini-3.1-pro-preview-customtools` 是 bash/custom tools 工具优先级优化的 endpoint variant，不是新基础模型。
- Model Card 明确 `Gemini 3.1 Pro is based on Gemini 3 Pro`，并将架构、训练数据、数据处理、硬件和软件资料指向 Gemini 3 Pro；因此本轮不把前代资料升级为 3.1 独有架构或训练事实。Model Card 的发布方 benchmark、安全和 Frontier Safety 结果单独保留，不与两个排行榜拼接。
- 面试主线已收口为：`thinking_level` 与共同 output budget；thought/tool `signature` 和 `id` 的 stateful/stateless 回放；tool context circulation；built-in/custom tool 的宿主责任；1M context 的缓存、召回和成本账本；评测设置与 CCL 证据边界。arXiv 标题精确检索为 0，全文 198 篇均为外部使用/评测结果，没有发现 Google 发布的 Gemini 3.1 Pro 专属技术报告。
- 已修正研究笔记中 Long context 快照哈希为 `cbed824426278d5f59e46f59b477eb268d5b8056970b29fe246406188b02cd78`，并同步研究笔记、来源索引、模型盘点和计划。Gemini 3.1 Pro Preview 当前为“资料级闭环”，暂无独立正式架构章节；下一主锚点仍只能从两个排行榜剩余重点条目选择。

## 2026-09-16：GLM-5 AA 单榜资料闭环

- 本轮重新使用三条用户提供的代理抓取 [Artificial Analysis GLM-5](https://artificialanalysis.ai/models/glm-5) 和 [DataCurve DeepSWE](https://deepswe.datacurve.ai/)。三条线路对两个页面均返回 HTTP 200；AA 快照为 `3,577,227` bytes、SHA-256 `57dbab2e4ef52e2c95c585bb1d0549f8044486903b77366fed1d5c534569c2a1`，DataCurve 快照为 `268,313` bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- AA 有精确 `GLM-5 (Reasoning)`/`glm-5` 条目，并列 `glm-5-non-reasoning`；页面第三方字段约为 744B total、40B active、200K context、72.4 tokens/s 和 1.34s TTFT。DataCurve 当前只检出 GLM-5.2、GLM-5.3 和 GLM-5.3 Flash 配置，没有精确 `GLM-5` 行，因此不记录 GLM-5 的 DataCurve 分数，也不迁移相邻版本结果。
- Z.ai 官方模型卡、[GLM-5 专属技术报告](https://arxiv.org/abs/2602.15763)、API 文档和 GitHub 已核验：744B total/40B active、28.5T 预训练 tokens、DSA、`GlmMoeDsaForCausalLM`、78 层、256 routed experts、top-8、1 shared expert、`index_topk=2048`、202752 position，以及面向长周期 Agent 的 `slime` 异步 RL 基础设施。
- 面试主线为：DSA indexer/top-k 召回与端到端成本；MoE 总参数、active compute、通信和显存的区分；rollout/trainer 解耦带来的 policy lag、样本新鲜度和 off-policy 风险；长轨迹 Agent RL 的 verifier 与 credit assignment；Agentic Engineering 中规划—工具—执行—测试—修复闭环；模型、API、harness、工具、环境和 verifier 的评测分层。
- 模型卡 README 的 HLE、SWE-bench、Terminal-Bench、BrowseComp、MCP-Atlas 等数字均是发布方在明确 prompt、最大生成长度、温度、harness、judge、超时和重复次数下的结果，不与 AA 或 DataCurve 拼成裸模型能力。
- 新增 [`glm-5-source-notes.md`](research/model-update-2026-09/glm-5-source-notes.md)，同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`plan.md`](plan.md) 和本进度表。GLM-5 当前为“AA 单榜资料闭环”，不新增独立正式章节；完整 DSA indexer 训练目标、生产 kernel、`slime` 调度、完整训练/后训练 recipe、硬件 profiling、线上接受率和独立复现仍待核验。

## 2026-09-16：GLM-5.2 双榜资料级闭环

- 按夜间 20:00—次日 09:00 可能中断的规则重新抓取 [Artificial Analysis GLM-5.2](https://artificialanalysis.ai/models/glm-5-2) 与 [DataCurve DeepSWE](https://deepswe.datacurve.ai/)，两个页面均返回 HTTP 200。AA 快照 `/tmp/glm52-aa-20260916.out` 为 `3,699,404` bytes，SHA-256 `25cf2438443a3ff1dac18964adb79ae98f24f41ed68d02ea4bd38e527a574244`；DataCurve 快照 `/tmp/glm52-ds-20260916.out` 为 `268,313` bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- 两个排行榜均有精确 GLM-5.2 配置：AA 为 `glm-5-2` 的 max/non-reasoning，DataCurve 为 `mini_swe_agent_glm_5_2_high` 和 `mini_swe_agent_glm_5_2_max`。DataCurve high/max 的 Pass@1 分别为 `36.2832%`/`43.7778%`，平均成本约 `$2.8355`/`$3.9199`，平均 Agent steps `121.88`/`129.13`；结果绑定 `mini-swe-agent`、4 runs、113 tasks、工具、环境和 verifier，不是裸模型能力。
- Z.ai 官方 [GLM-5.2 文档](https://docs.z.ai/guides/llm/glm-5.2) 快照为 `525,141` bytes，SHA-256 `d66a48a8a7654f1b033abca4fc9270a57d52961aace20d1335f36e9ab33ba6ba`，`dateModified` 为 `2026-09-03T14:49:41.429Z`。官方确认 1M context、128K 最大输出、thinking、function calling、context caching、structured output、MCP，以及项目级代码库、跨文件重构和分阶段验证的长周期工程工作流。
- “lossless context”、数月 Coding Agent 专项训练、开发者案例和长任务 benchmark 只按 Z.ai 发布方描述记录；不能解释成数学意义上的绝对无损，也不能把提示词中的 `/goal`、CLAUDE.md/Agent.md、ADB/logcat 或工作流案例写成内部算法。GLM-5.3 引用的 SAO with compaction 仍没有 GLM-5.2 原始定义。
- 已更新 [`glm-5.2-source-notes.md`](research/model-update-2026-09/glm-5.2-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan.md`](plan.md) 和本进度表。GLM-5.2 当前为“资料级闭环”，正式专题见第二十一册第 86 章；下一主锚点继续从两个排行榜的剩余重点候选中选择。

## 2026-09-16：Claude Sonnet 4.6 断点恢复、联网复验与资料级闭环

- 针对昨晚 20:00—今早 09:00 可能发生的代理中断，本轮重新使用三条代理抓取 Artificial Analysis 中文首页、DataCurve DeepSWE 和 Anthropic Sonnet 4.6 发布页，均返回 HTTP 200。三个代理取得的 Artificial Analysis 首页均为 `1,773,553` bytes、SHA-256 `60e491dc5eccaf2d032205a7380f30a24b2726583b5a9cfaccfe14c17d736476`；DataCurve 均为 `268,313` bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`，未发现截断或代理间差异。
- Sonnet 4.6 发布页三条线路均返回 HTTP 200、`281,283` bytes；动态响应哈希不同，但核心发布信息一致。`8098/1234` 访问 Anthropic 模型目录被重定向到区域不可用页，不能作为目录证据；`7890` 成功取得真实 `platform.claude.com` Models Overview，`728,089` bytes，SHA-256 `4c5ba69551075bbff00251943a1765c77be38f998d512fac67902067097cdfff`，确认 `Claude Sonnet 4.6`、`claude-sonnet-4-6`、1M context 和 128K output。
- Artificial Analysis 的 Sonnet 4.6 adaptive 快照为 `3,626,831` bytes，SHA-256 `5feda773b5ccefaaad7265b5f2d2401551d4d96e32cfa524a7fedd550db5b0de`；DataCurve 精确行 `mini_swe_agent_claude_sonnet_4_6_high` 为 135/451、Pass@1 `29.9335%`、Pass@4 `56.6372%`、平均成本约 `$5.5224`、平均输出 `76,160.31` token、平均 Agent steps `133.66`。这些属于榜单配置和 `mini-swe-agent` 系统结果，不能写成裸模型能力。
- Anthropic 官方资料确认 2026-02-17 发布、coding/computer use/long-context reasoning/agent planning/knowledge work/design 定位、1M context（发布时 beta）、adaptive/extended thinking、context compaction、tool search、web search/fetch、code execution、memory 和 programmatic tool calling。Computer-use prompt injection 风险虽称相较 Sonnet 4.5 改善，但不写成已解决。
- Claude Sonnet 4.6 当前为“资料级闭环”。没有公开参数规模、内部架构、完整训练/后训练 recipe 或独立技术报告；adaptive 预算、compaction 摘要格式、tool search 质量、线上 computer-use 接受率和独立复现仍待核验，不新增独立正式架构章节。研究笔记见 [`claude-sonnet-4.6-source-notes.md`](research/model-update-2026-09/claude-sonnet-4.6-source-notes.md)。

## 2026-09-16 Gemini 3.1 Pro Preview 断点恢复与资料级闭环

昨晚 20:00—今早 09:00 的联网中断风险按“重新收集”处理。2026-09-16 重新使用三条用户提供的代理抓取 Artificial Analysis 中文首页、Gemini 3.5/Gemini 3.1 Pro 详情页和 DataCurve；六份榜单请求均返回 HTTP 200，同一页面三条线路字节一致。随后通过 7890 重新取得 Google API 模型页、Release Notes、DeepMind Model Card、评测方法、Thinking、工具组合、Long context、Caching 和 arXiv 检索页。

Gemini 3.1 Pro Preview 已完成从两个唯一排行榜到权威资料的资料级闭环：Artificial Analysis 详情页记录第三方 Intelligence Index `30.3597`、约 `108.4052 tokens/s`、约 `24.6088s` TTFT、1M context；DataCurve `mini_swe_agent_gemini_3_1_pro_preview_high` 为 53/452、Pass@1 `11.7257%`、Pass@4 `28.3186%`、平均成本约 `$2.1434`、平均 75.56 steps。所有数字保留配置、provider/harness、工具、任务集、环境和 verifier 边界，不能拼成裸模型排名。

官方资料确认 2026-02-19 发布、1,048,576 输入/65,536 输出、多模态输入、thinking、function calling、grounding、structured outputs 和 context caching；`gemini-3.1-pro-preview-customtools` 是面向 bash/custom tools 的 endpoint variant，不新增模型。Model Card 明确 3.1 基于 Gemini 3 Pro，并把架构、训练数据、硬件和软件资料指向 Gemini 3 Pro；没有独立参数、层/专家结构、完整训练 recipe 或专属技术报告。

本轮提取的面试主线为 `thinking_level` 与共同 output budget、thought/tool `signature` 与 `id` 回放、tool context circulation、built-in/custom tool 宿主责任、1M context 的缓存/召回/成本工程，以及 Model Card/Frontier Safety 的评测设置与 CCL 证据边界。arXiv 标题精确检索为 0，全文检索 198 篇均为外部使用/评测结果。研究笔记见 [`gemini-3.1-pro-preview-source-notes.md`](research/model-update-2026-09/gemini-3.1-pro-preview-source-notes.md)。当前不新增独立正式架构章节；下一锚点继续从两个排行榜的剩余重点厂商候选中选择。

## 2026-09-16 Claude Sonnet 4.6 断点恢复与后续顺序

昨晚 20:00—今早 09:00 的联网中断风险已按计划重新复验。三条代理对 Artificial Analysis 中文首页、DataCurve DeepSWE 和 Sonnet 4.6 发布页均返回 HTTP 200；两个排行榜快照逐字节一致。`8098/1234` 的 Anthropic 模型目录被区域页重定向，`7890` 成功取得真实 `platform.claude.com` Models Overview，因此按线路差异记录，不把单代理失败解释为页面不存在。

Claude Sonnet 4.6 已完成“排行榜锚点 → 官方发布页/模型目录 → System Card 入口 → Agent 文档 → 研究笔记”的资料级闭环。后续面试重点是 1M context 的有效容量、adaptive/extended thinking 的成本—延迟—成功率分层、compaction 状态协议、computer-use 执行器安全，以及 tool search 与上下文预算；参数、架构、训练 recipe、compaction 内部格式和独立 benchmark 仍待核验，不新增独立架构章节。

该段记录的是 2026-09-16 的历史顺序。随后 `gpt-6-astra` 和 `gpt-5.3-codex` 已完成对应阶段的榜单复核与 OpenAI 官方资料补证；当前锚点以本文件最后一节的最新记录为准，继续按“榜单配置 → 官方资料 → 技术拆解 → 面试映射 → 证据边界”的顺序推进。

## 2026-09-17：联网恢复复核结果

- 为补查昨晚 20:00—今早 09:00 可能中断的联网任务，本轮重新测试三条用户提供的代理：`10.237.126.170:1234`、`10.24.27.134:8098`、`10.24.27.134:7890`，当前时点均无法连接代理服务器；Artificial Analysis 与 DataCurve 通过代理均返回 HTTP `000`，curl 退出码 `7`。
- 直连 `https://www.baidu.com` 返回 HTTP `200`、约 `2443` bytes，说明当前基础网络并非完全中断；Artificial Analysis、DataCurve、Z.ai 和 Google 直连均因 DNS 解析失败，curl 退出码 `6`。因此本轮将状态记录为“代理入口不可用/目标站点直连 DNS 失败”，不解释为目标网页不存在。
- 本轮没有把 IFM/K2-Horizon-7B-Uno 等非重点厂商条目提升为新锚点；网络恢复后仍只从 Artificial Analysis 与 DataCurve DeepSWE 重新确认 OpenAI、Anthropic、GLM、Qwen、Kimi、DeepSeek、Gemini、Grok 的新条目，再继续官方资料核验。

## 2026-09-17/18：排行榜恢复复核与 GPT-6 Astra 运行时资料补证

- 受控联网恢复后，三条用户提供的代理均可访问百度并返回 HTTP `200`；随后三条线路重新抓取 Artificial Analysis 中文首页和 DataCurve DeepSWE，均成功且逐字节一致。Artificial Analysis 快照为 `1,782,207` bytes、SHA-256 `d547f7bde6adc164aaea60026d06e9a59178fc86d89c8ce5333fe8d28ffb755a`；DataCurve 快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。
- 与已有快照比较，八家重点厂商没有新增基础模型候选；DataCurve 的模型集合也没有新增或删除。页面中出现的版本日期/目录条目变化不提升为新锚点，仍按基础模型、推理配置和 Agent harness 三层记录。
- GPT-6 Astra 仍同时出现在 Artificial Analysis 与 DataCurve：DataCurve 保留 `gpt-6-astra` 的 xhigh 配置；Artificial Analysis 主榜仍列 `GPT-6 Astra (max)`。本轮不把 AA 指数或 DeepSWE Pass@1 当作裸模型能力。
- 通过 `10.24.27.134:7890` 实际读取 OpenAI 官方 GPT-6 Astra 模型页、Reasoning、Prompt caching、Conversation state、Compaction 和 Tool search 文档。补齐的面试主线包括：动态 `configuration_update`、reasoning/`phase` 完整 item replay、`previous_response_id` 的状态与计费边界、deferred tool search 的缓存前缀/权限审计，以及 server-side/standalone compaction 的 canonical context。
- 已同步 [`gpt-6-astra-source-notes.md`](research/model-update-2026-09/gpt-6-astra-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan.md`](plan.md)、第六册第 18 章和面试配套文件。GPT-6 Astra 当前维持“内容专题闭环”，但没有公开参数、架构、训练 recipe、system card 或专属技术报告；下一锚点继续从两个排行榜中剩余的八家重点厂商条目选择。

## 2026-09-18：DeepSeek V3.2 AA 单榜资料级闭环

- 本轮重新抓取两个唯一排行榜：Artificial Analysis 当前详情页确认 `deepseek-v3-2` 的 `DeepSeek V3.2 (Non-reasoning)`，页面字段为 2025 年 12 月、128K context、Intelligence Index `16.043537719683`，第三方参数字段约 648B total/37B active；DataCurve 当前快照没有精确 `mini_swe_agent_deepseek_v3_2_*` 行，不迁移相邻版本结果。
- Artificial Analysis 详情快照为 3,391,821 bytes、SHA-256 `488c2c63fdb6d1746525f317222642d12cd57c7daf0f0abeba31653a6924ab7a`；DataCurve 快照为 268,571 bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。
- 官方模型卡核验三条技术主线：DeepSeek Sparse Attention（DSA）、scalable RL framework、large-scale agentic task synthesis pipeline；并核验 `thinking with tools`、新的工具调用模板、`developer` role 仅限 search-agent 且 API 不接受、V3.2-Speciale 不支持 tool calling、MIT license 和 V3.2-Exp 结构入口。
- 官方技术报告 PDF 已经通过 7890 下载，907,086 bytes、SHA-256 `f6fda5753db7b106baa5eeb1286877f17e6a111354762f8aa53c7e6556498df7`；已用标准库完成正文抽取，抽取文本 252,018 bytes、SHA-256 `f082910a550666ad32c914db94056daa59a13b75a7ffd2bbd548695aa5cd043c`。报告正文补充确认 DSA 的 dense warm-up/sparse training、2,048 KV top-k、约 2.1B/943.7B tokens、主 attention 复杂度边界、specialist distillation、GRPO 的 unbiased KL/off-policy masking/Keep Routing/Keep Sampling Mask，以及 code/search/general/interpreter Agent 数据规模与 verifier 闭环。公式/图表因 PDF 字体抽取失真仍待视觉复核；完整 kernel、层排布、训练超参、线上 acceptance rate、硬件 profiling 和独立 benchmark 继续待核验。
- 当前状态为“AA 单榜资料级闭环”，不新增重复正式章节；研究笔记见 [`deepseek-v3.2-source-notes.md`](research/model-update-2026-09/deepseek-v3.2-source-notes.md)。后续继续从两个排行榜剩余重点条目选择下一锚点。
- 配套闭环已完成：已同步 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`，并在第二十一册稀疏注意力、第四册后训练与第二十四册工具 serving 中补入 DSA、Agent 任务合成和 `thinking with tools` 的面试映射；未把 GLM/DeepSeek V4 的实现字段迁移为 V3.2 事实。
| Google（Gemini） | Gemini 3.8 Flash | AA/DS | 内容专题闭环 | Google 官方模型页、DeepMind Model Card、发布博客、评测 PDF、Interactions 专题和书系配套已入库；固定 SDK schema 补充 custom `function_call.id` → `function_result.call_id`、可选 signature 与 forward-compatible decode 证据；真实 endpoint 字段行为、3.8 独立架构/训练报告仍待核验。 |
| Google（Gemini） | Gemini 3.7 Flash | AA/DS | 资料级闭环 | 已完成 Google AI Developers 模型页、DeepMind Model Card、评测/安全入口、Interactions/工具/视频文档和 arXiv 定向检索；1M 输入、thinking、agentic video、step/state/signature 回放已核验；独立架构/训练报告待补证。 |
| Google（Gemini） | Gemini 3.6 Flash | AA/DS | 资料级闭环 | 已完成 Artificial Analysis/DataCurve 原始配置、Google AI Developers 模型页、DeepMind Model Card、Thinking/Interactions/工具组合/视频/缓存/Computer Use 文档和 arXiv 9 篇外部使用论文核验；默认 medium 与 `minimal/low/medium/high` thinking、agentic video、Model Card benchmark/safety 已记录，独立架构/训练报告待补证。 |
| Google（Gemini） | Gemini 3.5 Flash | AA/DS | 资料级闭环 | 已完成 Artificial Analysis/DataCurve 原始配置、Google AI Developers 模型页、What's New、DeepMind Model Card、Gemini 3 Flash 依赖 Model Card、Thinking/Interactions/工具组合/视频/缓存/Computer Use 文档和 arXiv 定向检索；默认 medium、thought preservation、工具上下文循环、Computer Use prompt-injection detection 和证据边界已记录；独立架构/训练报告待补证。 |
| Google（Gemini） | Gemini 3.5 Flash-Lite | AA 单榜 | 资料级闭环（AA 单榜） | Artificial Analysis 有精确 `gemini-3-5-flash-lite` 条目；DataCurve 当前无精确 `mini_swe_agent_gemini_3_5_flash_lite_*` 行。已核验 Google API 模型页、DeepMind Model Card/PDF、Thinking、Video understanding、发布方 benchmark 和安全边界；1M/65K、默认 minimal、agentic video、`processing_call/result` 已记录；独立架构/训练报告待补证。 |
| Google（Gemini） | Gemini 3.1 Pro Preview | AA/DS | 资料级闭环 | 已完成三代理榜单复验、Google API 模型页、Release Notes、DeepMind Model Card、评测方法、Thinking/工具组合/Long context/Caching 文档和 arXiv 检索；1M/65K、多模态、`thinking_level`、`customtools` endpoint、signature/id 回放和 tool context circulation 已核验；独立架构/训练报告待补证。 |
| xAI（Grok） | Grok 4.6 | AA/DS | 资料级闭环 | xAI 官方发布页、模型/API 文档、四档 DataCurve 配置、研究笔记和配套同步已完成；官方发布日期、500K context、模型生成数据/SFT 轨迹筛选、agentic RL、self-testing/verification 和工具协议已核验；参数、架构、完整训练配方、专属技术报告和独立 benchmark 待核验。 |
| xAI（Grok） | Grok 4.5 | AA/DS | 资料级闭环 | 两个排行榜的 `high` 配置、xAI 发布公告、官方模型/API 文档、研究笔记和同步记录已完成；参数、架构、完整训练配方、专属技术报告和独立 benchmark 待核验。 |

### 后续逐项顺序

（历史顺序记录）Grok 4.5、Grok 4.6、Claude Opus 5、Claude Fable 5.1、Claude Sonnet 5、Claude Sonnet 4.6、Gemini 3.7 Flash、Gemini 3.6 Flash、Gemini 3.5 Flash 和 Gemini 3.1 Pro Preview 均已完成“仅候选”到“资料级闭环”的升级，不新增专属架构章节。GLM-5 已完成 AA 单榜资料闭环，GLM-5.2 已完成双榜资料级闭环并新增第二十一册第 86 章；GPT-5.4、GPT-5.5、GPT-5.6、Gemini 3.8 Flash、Claude Fable 5、Claude Opus 4.8 和 Kimi K2.7 Code 继续只补独立架构/训练报告、真实 API 行为和外部评测；已达到“内容专题闭环”的 GPT-6 Astra、GLM-5.3、GLM-5.3-Flash、Kimi K3、DeepSeek V4/V4.1、K2 Horizon 36B/A4B 和 Qwen3.8 只补待核验项，不重复建立全量模型档案。该段之后已完成 GPT-6 Astra、Qwen3.8 Max (0902)、GPT-5.3 Codex、gpt-oss 与 Claude Opus 4.6 收口；本段为历史顺序，当前活动锚点以文件顶部状态和最后一条记录为准。

## 2026-09-18：Qwen3.8 Max (0902) revision 资料级闭环

- 已确认工作目录为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；本轮在该目录继续，保留已有用户修改。
- 重新核对两个唯一排行榜：Artificial Analysis canonical `qwen3-8-max` 当前页面标题为 `Qwen3.8 Max (0902)`，release slug 为 `qwen3-8-max-0902`，详情快照 `3,607,154` bytes、SHA-256 `e9152a5d81063cbeb45fb21ba621d7eb7da05073235b611f27a344f38c6288ae`，第三方 Intelligence Index `45.4354834980521`、context 约 `984K`；本轮 `/zh` 首页快照为 `1,769,534` bytes、SHA-256 `ce8fb0d2ca827cead0642152b02716022936060bfde1d74addbfa6657bc48f46`；DataCurve 当前只有泛化 `mini_swe_agent_qwen3_8_max_xhigh`。
- DataCurve 泛化行记录为 258/449、Pass@1 `57.4610%`、Pass@4 `83.1858%`、平均成本约 `$3.7291`、平均输出 `95,075` tokens、平均 Agent steps `111.34`、4 runs。由于没有 `qwen3_8_max_0902` 精确行，已明确禁止把该结果绑定成 0902 revision 的独立成绩。
- 通过 [Qwen3.8-Max-0902](https://www.qwencloud.com/models/qwen3.8-max-0902)、[Thinking](https://docs.qwencloud.com/developer-guides/text-generation/thinking)、[Function Calling](https://docs.qwencloud.com/developer-guides/tool-calling/function-calling) 和 [Context Cache](https://docs.qwencloud.com/developer-guides/run-and-scale/context-cache) 完成官方核验。产品页 alias 为 `qwen3.8-max-2026-09-02`，称其为 `qwen3.8-max` 的 upgraded snapshot，`last-modified` 为 `2026-09-18 10:45:04`。
- 已记录官方服务边界：1M context、991K 普通最大输入、983K thinking 最大输入、131K 最大输出；输入 `$2/M`、输出 `$6/M`、implicit cache `$0.25/M` 的价格字段按页面快照记录。官方产品定位包括 coding、工程规模项目、长周期 autonomous development、多工具 Agent 和视觉理解，但没有把这些描述升级成新架构事实。
- 已记录 `reasoning_effort` 的 `low/medium/xhigh`（默认 `xhigh`）与 `thinking_budget` 互斥；thinking 模式下 `tool_choice` 只能为 `auto`/`none`，强制工具选择需要关闭 thinking；多模态调用使用 `MultiModalConversation`。
- 已记录 explicit、implicit、session cache 的不同生命周期/命中/计费语义和最小 `1,024` tokens；没有把 provider cache 直接等同为永久 GPU KV cache。
- 新增 [`qwen3.8-max-0902-source-notes.md`](research/model-update-2026-09/qwen3.8-max-0902-source-notes.md)，同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan.md`](plan.md) 及书系配套文件，并扩展第二十一册第 83 章和第二十四册工具 serving 章节；不新增重复架构章节。
- 当前状态：Qwen3.8 Max (0902) 为“资料级闭环”。参数、层排布、专属技术报告、完整训练/后训练 recipe、生产 kernel、线上 acceptance rate、目标硬件 profiling 以及 hosted 0902 与 A95B 的精确服务差异仍待核验；下一轮仍从两个排行榜中的八家重点厂商条目选择锚点。

## 2026-09-18：GPT-5.3 Codex 锚点资料级闭环

- 已确认工作目录为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；本轮继续在该目录工作并保留既有修改。
- 重新核对两个唯一排行榜：Artificial Analysis `/zh` 快照为 `1,769,512` bytes、SHA-256 `d50456b4597b637829b46332b3991a8f6a004507341c3fc4d74bc390f42f3cff`；DataCurve 快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。GPT-5.3 Codex 出现在 Artificial Analysis 详情页；DataCurve 没有精确 `mini_swe_agent_gpt_5_3_codex_*` 行。
- Artificial Analysis 详情快照 `/tmp/gpt53codex-aa-detail-20260918.out` 为 `3,528,646` bytes、SHA-256 `565d91572b1a0bd9fb8f7f89f16c8beefbaadfdea79de5b229a9bd5a997b27ef`；页面标题 `GPT-5.3 Codex (xhigh)`、release date 字段 `2026-02-05`、Intelligence Index `32.5028174368983`（estimated）、context `400,000`、knowledge cutoff `2025-08-31`。这些是第三方配置字段。
- 通过 `10.24.27.134:7890` 获取 OpenAI 官方 [GPT-5.3-Codex 模型页](https://developers.openai.com/api/docs/models/gpt-5.3-codex.md) 和 [Codex Prompting Guide](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide.md)，并复核 Compaction、Conversation state、Tools、Agents、Prompt caching 文档。确认 `gpt-5.3-codex`、`low/medium/high/xhigh`、文本/图像输入、文本输出、400K context、272K maximum input、128K maximum output、Responses-only、function calling、web search、hosted shell、skills 和当前价格字段。
- 新增 Codex 运行时面试主线：较少 reasoning tokens、交互式任务优先 medium、困难任务使用 high/xhigh、长时自治、first-class compaction、工具 schema/并行调用/`apply_patch`/固定工作目录、assistant `phase`、完整 output item replay、opaque reasoning、canonical compaction context、KV prefix cache，以及模型输出/宿主授权/工具执行/artifact 的四段式责任边界。
- 明确 `tool_search` 不能从 GPT-5.4+ 文档迁移到 GPT-5.3 Codex；同时不把 GPT-5.5/5.6/5.4 或其他 Codex 配置的 DataCurve 结果迁移到本锚点。GPT-5.3 Codex 当前为“资料级闭环”，没有公开参数、架构、训练 recipe、system card、专属技术报告、生产 kernel、硬件 profiling 或线上 acceptance rate。

## 2026-09-18：OpenAI gpt-oss 内容专题闭环

- 确认工作目录为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。
- Artificial Analysis 当前存在 `gpt-oss-120b` 与 `gpt-oss-20b` 精确条目，详情快照分别为 `3,701,133` 和 `3,701,427` bytes；DataCurve 当前没有精确的 `mini_swe_agent_gpt_oss_120b_*`/`20b_*` 行，因此没有迁移其他 GPT/Codex 的 Agent 结果。
- OpenAI Model Card/arXiv、官方仓库、Harmony、Hugging Face 模型卡/配置与 Cookbook 已核验：120B/20B total/active 参数、128/32 experts、top-4、交替 sliding/full attention、GQA、RoPE/YaRN、MXFP4、`o200k_harmony`、CoT RL、Harmony channels/recipients 和 `low/medium/high` variable-effort reasoning。
- 已创建第二十一册第 87 章并加入目录；同步研究来源、模型盘点、榜单解释和 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`，并将工具权限/协议和 serving 账本映射到第六、五、十七、二十四册相关章节。
- 当前状态为“内容专题闭环”。完整训练/后训练 recipe、router 负载均衡、MXFP4 kernel/量化误差、目标硬件 profiling、线上接受率和独立 gpt-oss Agent 评测仍待核验；下一轮继续只从两个排行榜选择锚点。

## 2026-09-18：Claude Opus 4.6 资料级闭环

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留此前 dirty worktree 修改。Artificial Analysis 详情页确认 `claude-opus-4-6-adaptive` 与 `claude-opus-4-6`，DataCurve 当前没有精确 `mini_swe_agent_claude_opus_4_6_*` 行。
- AA adaptive/基础详情快照分别为 `3,497,470`/`3,485,093` bytes，SHA-256 分别为 `0bcf25542c95489db50a83925caf5e32bab672e4f7134126009fb94937b6b32d` 与 `544de10658d16de8ac25da44e6c6792ab51b64093a5a410f3eb3be1173e6d219`。第三方字段包括 `releaseDate=2026-02-05`、Intelligence Index 约 32（estimated）、1M context、约 37.7 output tokens/s、约 19.75s TTFT 和 `$5/$25` 每百万 token；不把这些字段写成 Anthropic 内部架构事实。
- Anthropic 官方资料已核验：发布页、System Card、模型页、Adaptive thinking、Effort、Compaction、Tool search、Computer use、Advanced tool use 和 context engineering。确认 1M context、128K 普通/300K Batch 最大输出、默认 high 与 `max/high/medium/low` effort、thinking signature 原样回传、`compact-2026-01-12` compaction、约 150K 默认/50K 最低 trigger、regex/BM25 tool search、最多 5 个 `tool_reference`，以及 `computer_20251124` 的宿主执行器边界。
- 发布方 1M retrieval、BigLaw Bench、盲测和 subagent 数字与 AA 指数分开保存；arXiv 精确检索未发现 Anthropic 的 Opus 4.6 专属技术报告，外部论文只作为使用/评测证据。完整研究笔记见 [`claude-opus-4.6-source-notes.md`](research/model-update-2026-09/claude-opus-4.6-source-notes.md)。
- 已同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`plan.md`](plan.md) 和书系配套面试资料。当前状态为“资料级闭环”，不新增独立 Transformer 章节；参数、架构、完整训练/后训练 recipe、adaptive 内部机制、compaction 编码、生产 kernel、硬件 profiling、线上 acceptance rate 和独立 benchmark 复现仍待核验。下一轮继续从两个唯一排行榜的八家重点厂商条目选择锚点。

## 2026-09-18：Claude Opus 4.7 正式章节收口

- 已在工作区恢复后继续上一轮遗留任务，确认 `/home/zzc/llm-from-zero-to-interview` 是 `/data/zzc/llm-from-zero-to-interview` 的同一项目目录。
- 已新增第二册 Agent 章节 `7.28 Claude Opus 4.7：把长任务预算和视觉输入纳入 Code Agent`，补充 `effort`、task budget、`max_tokens` 三层预算、tokenizer 迁移、高分辨率视觉、compaction 和 cyber safeguards 的面试/工程边界。
- 已新增第二十四册 serving 章节 `32.36 Claude Opus 4.7：三层预算、高分辨率视觉与安全控制面`，补充 serving manifest、预算账本、视觉输入账本、缓存/compaction 关系、安全策略和生产验收表。
- Opus 4.7 现在具备“排行榜发现 → 官方资料 → 研究笔记 → 正式章节 → 题库/练习/术语/项目/论文/知识图谱同步”的内容专题闭环；仍不把它升级为内部架构或独立 benchmark 已公开的结论。
- 下一轮仍只从 Artificial Analysis 与 DataCurve DeepSWE 的八家重点厂商条目选择锚点；已入库模型优先补齐版本变化、独立复现和 serving profiling，不从官方目录或论文搜索另发现模型。
## 2026-09-18：Kimi K3 官方技术报告与仓库核验

- 本轮在 `/data/zzc/llm-from-zero-to-interview` 继续，确认其与用户指定的 `/home/zzc/llm-from-zero-to-interview` 为同一项目；保留既有 dirty worktree 修改。
- 重新抓取两个唯一排行榜：Artificial Analysis K3、Artificial Analysis `/zh` 首页和 DataCurve DeepSWE。当前可用代理为 `10.237.126.170:1234`；`10.24.27.134:8098` 当前超时，`10.24.27.134:7890` 当前拒绝连接。代理失败只记录为线路状态，不解释为网页不存在。
- 新鲜快照：AA K3 `/tmp/aa-k3-1234b-20260918.out`，3,696,739 bytes，SHA-256 `258cafa80dba24b5528f0933a8c45c0e1c0b41d1064ababebef77680ecd8a310`；AA `/zh` `/tmp/aa-home-1234b-20260918.out`，1,769,161 bytes，SHA-256 `4c0304f1955031c61e41a078a3028df13bdff36fa1aacd384c09e19a513e9aa0`；DataCurve `/tmp/ds-1234-20260918.out`，268,571 bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。
- 已核验 MoonshotAI 官方仓库 `MoonshotAI/Kimi-K3`、README、`k3_tech_report.pdf` 和 Kimi K3 License。技术报告 PDF `/tmp/k3-tech-report-20260918.pdf` SHA-256 `86fb82a63ced501f0c3f4f404c0c6fa88a7a6cfac17aae81fd1a8f455998067c`；报告正文提取到 `/tmp/k3-tech-report-extracted-20260918.txt`。
- K3 核验配置：2.8T total、104B activated、93 layers、69 KDA + 24 Gated MLA、每 3 个 KDA 后接 1 个 Gated MLA、末尾 Gated MLA、8 个 Block AttnRes/每 block 12 层、896 routed experts/16 selected/2 shared、1,048,576 context、MoonViT-V2、MXFP4 weights + MXFP8 activations QAT、SiTU-GLU 和 Quantile Balancing。
- 报告新增面试主线：三条扩展轴；lower-bounded decay (`g_min=5`)；latent-width routed experts + full-width shared experts；冻结 bias + fixed Top-k；Per-Head Muon；multi-teacher on-policy distillation；KDA context parallelism、MoonEP、长轨迹 RL、外部 KV retention、可恢复 microVM sandbox、budget-aware effort scheduling；XTM 的 `[open]/[sep]/[close]/[end_of_msg]`、`think/response/tool` channel、动态 `tool-declare` 和 tool/index 配对。
- 许可证已核验，覆盖 weights、parameters、configuration、code 和 documentation，并含商业条件。README 声明完整权重已发布并给出 HF/ModelScope 入口，但具体权重仓库、文件和 revision 本轮未稳定确认，不能写成“已下载权重”。
- 已同步研究底稿、来源索引、榜单解释、计划和相关 Agent 章节；第二十一册新增第 88 章，KDA/AttnRes 旧章补充 K3 report 与前作论文的边界。
- 当前状态：Kimi K3 从“资料级、报告待核验”升级为“内容专题闭环”。仍待核验具体权重下载/revision、FlashKDA 代码细节、vLLM prefix-cache 合并状态、完整训练 recipe、硬件 profiling、独立 benchmark 复现和线上 acceptance rate。下一锚点仍只能从两个排行榜的八家重点厂商条目选择。

## 2026-09-18：Gemini 3 Deep Think 配置级锚点闭环

- 确认工作目录 `/data/zzc/llm-from-zero-to-interview` 与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留此前 dirty worktree 修改。
- Artificial Analysis 已确认精确 `Gemini 3 Deep Think` 条目；DataCurve 当前没有精确 `mini_swe_agent_gemini_3_deep_think_*` 行，因此不迁移 Gemini 3.1 Pro、Gemini 3.7 Flash 或其他 Gemini 配置的 Pass@1、成本和 Agent steps。
- Google 官方同名 API 模型页、DeepMind Model Card 和 Blog 路径本轮没有提供独立 Deep Think model ID、权重或技术报告；官方可核验对象是 Gemini 3.1 Pro / Gemini 3 系列的 Deep Think 运行配置与评测设置。研究笔记见 [`gemini-3-deep-think-source-notes.md`](research/model-update-2026-09/gemini-3-deep-think-source-notes.md)。
- 已同步 `plan.md`、`source-index.md`、`model-inventory.md` 和 `inventory-interpretation.md`。面试主线是 `thinking_level` 与共同 output budget、thought/tool signature 和 `id` 回放、1M context/caching，以及 test-time compute、能力、延迟、成本和安全评测的联合分析；不新增独立架构章节。
- 当前状态：资料级闭环（配置级锚点）。Artificial Analysis 的指数、约 130K context、速度、价格和 provider 字段仍仅作为第三方目录信息；独立 checkpoint、参数、架构、训练 recipe、独立 benchmark 和精确 DataCurve Agent 评测仍待核验。

## 2026-09-18：Qwen3.5-397B-A17B 内容专题闭环

- 已确认工作目录为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留已有 dirty worktree 修改。
- Artificial Analysis 精确确认 `Qwen3.5 397B A17B`，页面的 Reasoning/Non-reasoning 按同一基础模型归并；详情快照 `/tmp/qwen35-397-aa.html` 为 3,700,616 bytes，SHA-256 `04f09d032033c2705bc9a40c649ebc226f81eeef457adf11886f644490181719`。DataCurve 当前没有精确 `mini_swe_agent_qwen3_5_397b_a17b_*` 行，因此不迁移其他 Qwen 模型的 Agent 分数。
- 通过 Qwen 官方模型卡、`config.json`、QwenLM/Qwen3.8 官方仓库和 Qwen3.5 发布博客核验 397B total/17B active、60 层、hidden 4096、15 组 `3 x Gated DeltaNet + 1 x Gated Attention`、512 experts、10 routed + 1 shared、原生 vision encoder、262,144 native context、约 1,010,000 YaRN context 和 MTP serving 示例。
- 官方模型卡/博客还声明 early-fusion 多模态训练、trillions of multimodal tokens、million-agent RL environments、asynchronous RL framework 和 201 languages/dialects；这些内容按 Qwen 官方自报记录，不能升级为独立复现或完整训练 recipe。
- 新增 [`qwen3.5-397b-a17b-source-notes.md`](research/model-update-2026-09/qwen3.5-397b-a17b-source-notes.md)，同步来源索引、模型盘点、榜单解释、`plan.md`；第二十一册第 83 章新增 Qwen3.5 前置/对比小节并更新目录。
- 明确证据边界：Qwen3.8 的 QSA、Gated Residual、N-gram Embedding 和 Muon 不能反向迁移为 Qwen3.5 技术；GDN/GA 完整 kernel、state layout、MTP acceptance length、视觉独立复现、硬件 profiling 和 Qwen3.5-Plus hosted/open 精确差异仍待核验。
- 当前状态：Qwen3.5-397B-A17B 为“内容专题闭环”；下一轮仍只能从 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-18 Qwen3.8 Max (0902) revision 收口

本轮继续使用两个唯一排行榜作为模型入口。Artificial Analysis 当前 canonical `qwen3-8-max` 页面标题为 `Qwen3.8 Max (0902)`，release slug 为 `qwen3-8-max-0902`；DataCurve 只有 `mini_swe_agent_qwen3_8_max_xhigh` 泛化配置。Qwen Cloud 官方页面将 0902 定义为 `qwen3.8-max` 的 upgraded snapshot，alias 为 `qwen3.8-max-2026-09-02`，因此研究对象是 hosted service revision，不是新的开源基础模型。

本轮已完成：

- 新增 [`qwen3.8-max-0902-source-notes.md`](research/model-update-2026-09/qwen3.8-max-0902-source-notes.md)，记录榜单快照、官方服务页、thinking/function calling/context cache 证据、hosted/open 边界和待核验项。
- 更新 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md) 和 [`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)，明确 0902 的 revision 身份。
- 同步第二十一册既有 Qwen3.8 专题和第二十四册工具 serving 的 revision、请求预算、工具选择与缓存主线，并更新论文、题库、练习、术语、项目和知识图谱。
- 记录官方请求边界：1M context、991K 普通最大输入、983K thinking 最大输入、131K 最大输出；记录 `reasoning_effort=low/medium/xhigh`（默认 `xhigh`）与 `thinking_budget` 互斥，以及 thinking 模式下 `tool_choice` 只能是 `auto`/`none`。
- 记录 explicit、implicit、session cache 的不同生命周期、命中和计费语义；不把它们粗略等同为永久 GPU KV cache。

DataCurve 的 258/449、Pass@1 `57.4610%`、Pass@4 `83.1858%`、约 `$3.7291`/task、约 `95,075` output tokens 和约 `111.34` Agent steps 只绑定泛化 `qwen3_8_max_xhigh`、`mini-swe-agent`、工具、环境、verifier 和 4 runs；没有 0902 精确行，因此不迁移为 0902 的独立结果。0902 的参数、层排布、完整训练/后训练 recipe、专属技术报告、生产 kernel、线上 acceptance rate、目标硬件 profiling 和与 A95B 的精确服务差异仍待核验。

本轮状态：Qwen3.8 Max (0902) 为“资料级闭环”；Qwen3.8 架构主线继续归入第二十一册第 83 章，服务协议主线归入第二十四册工具 serving 章节。下一轮仍先从 Artificial Analysis 与 DataCurve DeepSWE 的八家重点厂商条目选择锚点，再沿该锚点的官方资料推进。

## 2026-09-18 GPT-5.3 Codex 锚点闭环

本轮重新确认两个唯一排行榜和 GPT-5.3 Codex 的候选身份。Artificial Analysis 详情页标题为 `GPT-5.3 Codex (xhigh)`，页面 release date 字段为 `2026-02-05`，第三方 Intelligence Index 为 `32.5028174368983`（estimated），context 为 `400,000`；DataCurve 当前没有精确的 `mini_swe_agent_gpt_5_3_codex_*` 行，因此不挂接其他 Codex/GPT 版本的 DeepSWE 结果。

本轮已完成：

- 新增 [`gpt-5.3-codex-source-notes.md`](research/model-update-2026-09/gpt-5.3-codex-source-notes.md)，固定 Artificial Analysis 详情快照和 OpenAI 官方资料快照哈希。
- 核验 [GPT-5.3-Codex 官方模型页](https://developers.openai.com/api/docs/models/gpt-5.3-codex.md)：`gpt-5.3-codex`、`low/medium/high/xhigh`、文本/图像输入、文本输出、400K context、272K maximum input、128K maximum output、Responses-only、function calling、web search、hosted shell 和 skills。
- 核验 [Codex Prompting Guide](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide.md)：较少 reasoning tokens、交互式任务优先 medium、困难任务使用 high/xhigh、长时自治、first-class compaction、工具 schema、并行调用、`apply_patch`、固定工作目录和 phase/replay。
- 沿官方运行时资料补齐完整 output item replay、加密 reasoning item、`previous_response_id`、server-side/standalone compaction、canonical context、prompt cache 前缀和工具/权限/artifact 账本；`tool_search` 的 GPT-5.4+ 支持边界单独保留，不迁移到 GPT-5.3 Codex。
- 同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、本计划、进度、面试题、练习、术语、项目、论文入口和知识图谱；扩展第二十四册 Agent serving 章节，并在第六册 GPT-6 Astra 章节加入对照，不新增重复 Transformer 架构章节。

GPT-5.3 Codex 当前为“资料级闭环”。官方没有公开参数规模、激活参数、层/专家结构、训练数据、完整 pre-training/post-training recipe、system card、专属技术报告、生产 kernel、目标硬件 profiling 或线上 acceptance rate。下一轮仍只能从 Artificial Analysis 与 DataCurve DeepSWE 的八家重点厂商条目选择新锚点。

## 2026-09-18 OpenAI gpt-oss 锚点推进

本轮从 Artificial Analysis 的精确条目选择 `gpt-oss-120b` 与 `gpt-oss-20b`，不从官方目录、论文列表或 Hugging Face Trending 另发现模型。DataCurve 当前没有两个模型的精确 `mini_swe_agent` 行，因此不迁移任何其他 OpenAI/Codex 模型的 Agent 结果。

已完成榜单快照、OpenAI Model Card/arXiv、官方 gpt-oss 仓库、Harmony、Hugging Face 模型卡/配置和 Cookbook 的资料核验，并新增第二十一册第 87 章。正式配套已映射到 MoE/稀疏注意力、MXFP4 量化、后训练 CoT RL、Harmony 输出协议、variable-effort reasoning、工具执行边界和 Agent serving。

本锚点的面试主线是：total 与 active parameters 的 serving 账本；交替 sliding/full attention 的长程交换；MXFP4 与量化/硬件联合设计；Harmony 与普通 chat template 的协议差异；`low/medium/high` 是同一权重的运行时配置；模型 tool call 与宿主执行器、权限、沙箱、verifier 的分层。完整训练 recipe、router 负载均衡、MXFP4 kernel/误差、目标硬件 profiling、线上接受率和独立 gpt-oss Agent 评测继续标为待核验。

当前状态：`gpt-oss-120b/20b` 为“内容专题闭环”；下一轮仍只能从 Artificial Analysis 与 DataCurve DeepSWE 的八家重点厂商条目选择新锚点，并继续区分模型、revision、effort、provider、fallback 与 harness。

## 2026-09-18 Claude Opus 4.7 锚点推进

本轮收口 `claude-opus-4.7`，因为它是 Artificial Analysis 中 Anthropic 的近期重点条目；`claude-opus-4-7` 与 `claude-opus-4-7-non-reasoning` 按同一模型的运行配置归并。DataCurve 当前没有精确 `mini_swe_agent_claude_opus_4_7_*` 行，因此不得迁移其他 Claude 版本的 DeepSWE 结果。

已完成 Anthropic 发布页、System Card、模型页、adaptive thinking、effort、Task budgets、Vision、Compaction、Tool search、Computer use、Advanced tool use 和 context engineering 资料核验，并新增研究笔记、来源索引、模型盘点、第二册 `7.28` 和第二十四册 `32.36`。面试映射集中在三层预算、tokenizer 迁移、高分辨率视觉、thinking signature replay、server-side compaction、按需加载工具 schema、宿主 computer-use 执行器和 cyber safeguards 控制面；没有公开参数、架构或完整训练报告，因此不新增独立 Transformer 章节。

下一轮继续只从 Artificial Analysis 与 DataCurve DeepSWE 的八家重点厂商条目选择锚点；已入库模型优先补充版本变化、完整 kernel/硬件 profiling、线上 acceptance rate 和独立复现，不从官方目录或论文搜索另发现模型。
## 2026-09-18 Kimi K3 技术报告与开放权重核验

本轮继续使用 Artificial Analysis 与 DataCurve DeepSWE 作为唯一模型发现入口。Kimi K3 已从“官方发布文章、完整配置待核验”升级为“内容专题闭环”：官方仓库、技术报告 PDF、README 和 Kimi K3 License 已核验，关键架构字段和 Agent 协议已可进入正式教材。

执行结果：

- 研究底稿已补齐 `k3_tech_report.pdf` 的架构、训练、基础设施和 XTM 协议摘记；`source-index.md`、`model-inventory.md` 和 `inventory-interpretation.md` 已同步。
- 第十七册第 15 章保留长任务 harness 与思考状态主线，并增加报告已核验后的证据边界；第二十一册新增第 88 章，统一收口 KDA、Gated MLA、Block AttnRes、Stable LatentMoE、量化和长轨迹 RL。
- 具体 Hugging Face/ModelScope 权重仓库、文件清单和 revision 仍待核验；README 的“完整权重已发布”不等于本地已下载权重。
- FlashKDA 实际代码版本、vLLM K3 prefix-cache 合并状态、完整训练数据与 optimizer recipe、目标硬件 profiling、独立 benchmark 复现和线上 acceptance rate 仍保持待核验。

后续顺序：先固定 K3 权重入口和 serving 实现证据，再从两个排行榜选择下一家重点厂商锚点；不从官方目录、论文检索或 Hugging Face Trending 另发现模型。

## 2026-09-18 Gemini 3 Deep Think 配置级锚点

`Gemini 3 Deep Think` 已在 Artificial Analysis 中确认精确条目；DataCurve DeepSWE 当前没有精确 `mini_swe_agent_gemini_3_deep_think_*` 行。Google 官方资料没有对应独立 API model ID、Model Card、公开权重或专属技术报告，官方可核验对象是 Gemini 3.1 Pro Preview / Gemini 3 系列中的 Deep Think 运行配置与评测设置。

本轮新增 [`gemini-3-deep-think-source-notes.md`](research/model-update-2026-09/gemini-3-deep-think-source-notes.md)，并同步来源索引、模型盘点、榜单解释和进度。记录的面试主线包括 `thinking_level` 与共同 output budget、thought/tool signature 和 `id` 回放、1M context/caching，以及 test-time compute、能力、延迟、成本和安全评测的联合分析。该对象标记为“资料级闭环（配置级锚点）”，不新增独立正式章节，也不把 3.1 Pro 的资料改写成 Deep Think 独有架构或训练事实。

## 2026-09-18 Qwen3.5-397B-A17B 锚点推进

本轮已完成 `Qwen3.5-397B-A17B` 的“榜单发现 → 官方资料 → 技术拆解 → 面试映射 → 书系同步”。Artificial Analysis 有精确条目，DataCurve 没有精确 Agent 行；Qwen 官方模型卡、配置、仓库和博客核验了 Gated DeltaNet/Gated Attention + sparse MoE、native multimodal、MTP、长上下文和 serving 边界。

新增 [`qwen3.5-397b-a17b-source-notes.md`](research/model-update-2026-09/qwen3.5-397b-a17b-source-notes.md)，并将 Qwen3.5 作为 Qwen3.8 的前置锚点补入第二十一册第 83 章。QSA、Gated Residual、N-gram 和 Muon 仍只归属于 Qwen3.8/Flash-Next 的已有证据，不反向迁移给 Qwen3.5。

下一轮仍只从 Artificial Analysis 与 DataCurve DeepSWE 的八家重点厂商条目选择新锚点；已入库模型优先补完整 kernel、独立 benchmark、硬件 profiling、线上 acceptance rate 和 revision 差异，不从官方目录、论文检索或 Hugging Face Trending 另发现模型。

## 2026-09-20：DeepSeek V4 Pro 0813 内容专题闭环（上一锚点）

- Artificial Analysis 精确页为 [`deepseek-v4-pro`](https://artificialanalysis.ai/models/deepseek-v4-pro)，标题 `DeepSeek V4 Pro 0813 (Reasoning, Max Effort)`；详情快照 3,934,926 bytes，SHA-256 `b14ce625eb337401f4faa63eed37f1b08fe109a693dcdf2a34e21924b6cddff1`。AA 目录字段为 release date `2026-08-13`、Intelligence Index `35.9967791278402`、1M context、约 1.6T/49B total/active。
- DataCurve 精确行是 `mini_swe_agent_deepseek_v4_pro_max`：Pass@1 `62.831858%`、Pass@4 `88.495575%`、`n_runs=4`、平均成本 `$1.6660232187`、平均输出 `105998.9` token、平均 `154.71` Agent steps。该结果绑定 max effort、mini-swe-agent、工具、任务集、环境和 verifier，不是裸模型分数。
- 官方闭环：V4 Pro GA 公告确认 2026-08-13、`low/high/max`、Responses API 和 Codex 优化；Responses 文档确认 stateless、function tools、`apply_patch`、并行调用和不支持参数可能静默忽略；模型卡/配置/技术报告支持 1.6T/49B、1M、CSA/HCA、mHC、Muon、FP4/FP8、32T+、SFT+GRPO+on-policy distillation 及 61 层/384 routed/6 selected/1 shared 等实现字段。
- 当前闭环状态：**内容专题闭环（双榜锚点）**。待核验完整独立 benchmark、生产 kernel、目标硬件 profiling、线上 tool acceptance、完整 recipe 和 API 快照差异；后续把这些作为补证，不再另建重复章节或迁移其他 DeepSeek 配置的评测。

## 2026-09-20：GLM-5 DSA、slime 与 Agentic Engineering 正式专题闭环

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。重新抓取两个唯一排行榜：Artificial Analysis 当前页面标题为 `GLM-5 (Reasoning)`，并提示已有更新模型 `GLM-5.1`；DataCurve 仍没有精确 `mini_swe_agent_glm_5_*` 行。
- 新鲜快照：AA `/tmp/glm-aa-20260920.out` 为 `3,811,809` bytes，SHA-256 `0b9c56ff97a87b1dd0a006d1c300057f5ef20c3f65a19bcddf0f6803d8a94f8d`；DataCurve `/tmp/aa-check-8098-ds-20260920.out` 为 `268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。不迁移 GLM-5.2/5.3/5.3 Flash 的 DeepSWE Pass@1、成本或 steps。
- Z.ai 博客资源 `/tmp/glm-blog-js-20260920.out` 为 `145,189` bytes，SHA-256 `99d27d6132c25e1b39fe26df0605ad1abd855e9b30b098996c64423093fb618f`，可复验 `GLM-5`、`DSA`、`slime`、`744B`、`40B`、`28.5T` 和 `Agentic` 等入口正文字符串。`https://docs.z.ai/guides/llm/glm-5.md` 及无扩展名文档页本轮由代理返回 503；这只记录为线路失败，不解释为官方页面不存在。
- 新增第二十一册第 89 章 [`GLM-5：DSA、MoE、slime 与 Agentic Engineering`](book-21-transformer-architecture-evolution/chapters/89-glm-5-dsa-slime与agentic-engineering.md)，并更新该册目录和架构创新时间线。章节将 DSA indexer/top-k 召回、744B/40B total/active 分账、MLA 低秩字段、`slime` rollout/trainer 解耦、policy lag/freshness、长轨迹 verifier 和最终 artifact gate 组织成面试主线。
- 已同步 [`glm-5-source-notes.md`](research/model-update-2026-09/glm-5-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan.md`](plan.md) 以及本进度文件；随后补齐论文、题库、练习、术语、项目和知识图谱。
- 当前状态：GLM-5 升级为“内容专题闭环（AA 单榜）”。DataCurve 精确评测、完整 DSA indexer loss/recall、生产 kernel、`slime` 调度与 freshness 控制、完整训练/后训练 recipe、硬件 profiling、线上 tool acceptance 和独立复现仍待核验；下一轮仍只从八家重点厂商在两个排行榜中的剩余条目选择锚点。

## 2026-09-20：GLM-5 之后选择 DeepSeek V3.2 下一锚点

- 重新抓取 [Artificial Analysis DeepSeek V3.2](https://artificialanalysis.ai/models/deepseek-v3-2)：`10.24.27.134:8098` 与 `10.237.126.170:1234` 均返回 HTTP 200；两个快照均为 `3,625,720` bytes，SHA-256 `7d4fd97ed7f2efbc9ab2f001072b353f00e2476cb884abe0d8185f16d7562052`，逐字节一致。页面仍为 `DeepSeek V3.2 (Non-reasoning)`，128K context、Intelligence Index `16.043537719683`、第三方参数约 648B total/37B active。
- DataCurve `/tmp/deepswe-20260920-8098.out` 为 `268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，仍没有精确 `mini_swe_agent_deepseek_v3_2_*` 行；不迁移 V3.1/V4 的 Pass@1、成本或 Agent steps。
- DeepSeek V3.2 已有研究笔记 [`deepseek-v3.2-source-notes.md`](research/model-update-2026-09/deepseek-v3.2-source-notes.md)，其中官方模型卡/技术报告/V3.2-Exp 已核验 DSA 两阶段训练、2,048 KV top-k、scalable RL、GRPO 稳定化、Agent 任务合成/验证和 `thinking with tools`；当前仍为“AA 单榜资料级闭环”，不新增重复正式架构章节。
- 当前顺序：GLM-5 已完成内容专题收口；`deepseek-v3-2` 作为下一锚点，后续只补 PDF 公式/图表视觉复核、完整 DSA kernel/层排布、训练细节、线上 tool acceptance、硬件 profiling 和独立 benchmark。

## 2026-09-20：DeepSeek V3.2 官方实现与 serving 证据补充

- 已确认工作目录为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；本轮只在该目录推进并保留既有 dirty worktree 修改。
- 研究笔记新增官方实现复核：V3.2-Exp inference demo 的 FP8 indexer、non-interleaved indexer RoPE、`fp8_index`、causal mask/top-k、prefill MHA、decode MQA、latent KV/position cache 和 FP8 KV cache；本地 `model.py`/`kernel.py` 快照哈希已登记。
- 明确配置边界：固定官方 V3.2 `config.json` 与 V3.2-Exp inference config 都核验为 `q_lora_rank=1536`；实验 config 同时记录 `dim=7168`、61 layers、3 dense layers、256 routed/8 active experts、`index_topk=2048` 和 FP8/`ue8m0`。两份 artifact 仍按 revision、实现路径和证据等级分账，不合并成完整生产配置。
- 核验 [TileLang DeepSeek V3.2 examples](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32)、[DeepGEMM PR #200](https://github.com/deepseek-ai/DeepGEMM/pull/200)、[FlashMLA PR #98](https://github.com/deepseek-ai/FlashMLA/pull/98) 和 [vLLM recipe](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html)：补齐 radix/histogram top-k、sparse MLA gather、pipelined double buffering、MoE/MQA logits、FP8/BF16 KV cache、`DP=8, EP=8, TP=1` 与 TP fallback 的面试边界。
- vLLM recipe 的 GSM8K 5-shot `0.9591`/20-shot `0.9538` 已标为 V3.2-Exp + vLLM + lm-eval 的 recipe/harness 结果，未写入 DataCurve 或模型能力字段；DataCurve 仍没有精确 V3.2 行。
- 已同步 [`deepseek-v3.2-source-notes.md`](research/model-update-2026-09/deepseek-v3.2-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan.md`](plan.md) 以及论文、题库、练习、术语、项目和知识图谱。
- 当前状态仍为“AA 单榜资料级闭环”：公开实验实现路径已补齐，但最终 production kernel 全覆盖、召回/误差曲线、各 GPU profiling、真实并发 p99、线上 tool-call acceptance rate 和独立 benchmark 仍待核验；不新增 V3.2 重复架构章节。

## 2026-09-20：Gemini 3.5 Flash-Lite 资料级闭环

- 重新确认工作目录为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；本轮继续保留既有 dirty worktree 修改。
- Artificial Analysis 精确条目为 [`gemini-3-5-flash-lite`](https://artificialanalysis.ai/models/gemini-3-5-flash-lite)，快照 `3,845,070` bytes、SHA-256 `7de22f529d3b4e2dd440ef8a8e9447480e18dde08437fc508ece06767571e8ef`；第三方字段为 `releaseDate: 2026-07-21`、`isReasoning=true`、Intelligence Index `22.1685424839812`。DataCurve 快照 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，无精确 Lite Agent 行，不迁移相邻 Gemini 的 Pass@1、成本或 steps。
- Google API 模型页确认 `gemini-3.5-flash-lite`、text/image/video/audio/PDF 输入、text 输出、1,048,576 输入 token、65,536 输出 token、caching/code execution/Computer Use Preview/File Search/function calling/Maps/Search grounding/structured output/thinking/URL context；Thinking 文档把默认设为 `On (minimal)`，支持 `minimal/low/medium/high`。
- Video understanding 文档把 Lite 列入 agentic video：static 路径约 1 FPS 固定取帧，agentic 路径按 prompt 动态读取时间轴、transcript、帧和音频，并用 `processing_call`/`processing_result` 暴露处理状态；88% token efficiency 与 7% quality 是文档发布方的总体表述，不是每个请求的保证。
- DeepMind Model Card/PDF 明确 Lite based on Gemini 3.1 Flash-Lite，并将 architecture、training dataset、data processing、hardware 和 software 指向 3.1 Model Card。SWE-Bench Pro `54.2%`、Terminal-Bench 2.1 `54.0%`、OSWorld-Verified `74.0%`、GDM-MRCR v2 128K/1M `72.2%`/`21.3%` 和 `$0.30/$2.50` 价格均按 Google 发布方评测/产品设置记录。
- 正确的 `whats-new-gemini-3.5` 页面是 Gemini 3.5 Flash 页面，不能把 Flash 的默认 `medium`、GA 或 thought preservation 回写到 Lite。当前状态为“资料级闭环（AA 单榜）”，不新增独立架构章节；完整证据和待核验项见 [`gemini-3.5-flash-lite-source-notes.md`](research/model-update-2026-09/gemini-3.5-flash-lite-source-notes.md)。

## 2026-09-20：Kimi K2.6 AA 单榜资料级闭环

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留已有 dirty worktree 修改。
- Artificial Analysis 精确条目为 [`kimi-k2-6`](https://artificialanalysis.ai/models/kimi-k2-6)，三个代理的详情快照逐字节一致：`3,938,140` bytes，SHA-256 `f24ad7d9cd3ddbb253470750dcf9f47aea3fc0d675c9f8f7eb8bdbcd50dcfe18`。页面的 deprecated/K3 迁移提示只作为第三方目录状态。
- DataCurve 快照为 `268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；当前没有精确 `mini_swe_agent_kimi_k2_6_*` 行，因此不迁移 K2.7 Code/K3 的 Pass@1、成本、输出 token 或 Agent steps。
- 已新增 [`kimi-k2.6-source-notes.md`](research/model-update-2026-09/kimi-k2.6-source-notes.md) 和第二十一册第 90 章 [`Kimi K2.6：Native Multimodal、Agent Swarm 与推理验收`](book-21-transformer-architecture-evolution/chapters/90-kimi-k2.6-native-multimodal-agent-swarm与推理验收.md)，同步目录、架构时间线、来源索引、模型盘点和榜单解释。
- 已核验官方 1T/32B MoE、MLA、MoonViT、256K、native INT4、K2.5 架构路线复用、vLLM/SGLang/KTransformers 部署、thinking/preserve_thinking/tool protocol、Agent Swarm 和 Kimi Vendor Verifier。300 sub-agents 属于 harness，不是内部 expert；KVV 属于部署/实现验收，不是模型分数。
- 当前状态：**AA 单榜资料级闭环**。完整训练/后训练 recipe、层排布、生产 kernel、INT4 误差与硬件 profiling、Agent Swarm coordinator、线上 tool acceptance、精确 DataCurve 行和独立复现仍待核验；下一轮仍只从两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20：GPT-5.4 mini/nano 资料级闭环

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作；Artificial Analysis 精确条目为 [`gpt-5-4-mini`](https://artificialanalysis.ai/models/gpt-5-4-mini) 与 [`gpt-5-4-nano`](https://artificialanalysis.ai/models/gpt-5-4-nano)。mini 详情快照为 `3,857,350` bytes、SHA-256 `a01a6da4a38077bc6309333d509bf303f8b1beb36729fd94f79fe7d3a8ffe0ee`；nano 为 `3,859,704` bytes、SHA-256 `2e6ddfefc9a3483437f0fe2648ce23ff42fe4f8614cec0c365dfc69777a166b4`。两页 release date 均为 `2026-03-17`，第三方 Intelligence Index 分别为 `24.0682169231964` 与 `20.7197352504557`，当前目录状态均提示迁移到 GPT-5.6 对应配置；这些仍是 AA 第三方字段。
- DataCurve 当前仅有 `mini_swe_agent_gpt_5_4_xhigh`，它绑定 GPT-5.4 base + `mini-swe-agent` + 工具/环境/verifier；没有 GPT-5.4 mini 或 nano 的精确模型 ID 行。因此不把 GPT-5.4 base 的 Pass@1、成本、输出 token 或 Agent steps 迁移给两个 sibling。
- OpenAI 官方模型页已确认 snapshot `gpt-5.4-mini-2026-03-17` 与 `gpt-5.4-nano-2026-03-17`，两者均为 text/image → text、400K context、272K maximum input、128K maximum output，并支持 `none/low/medium/high/xhigh` reasoning effort。mini 的官方定位偏高吞吐 coding、computer use 和 Agent workflow；nano 偏 classification、extraction、ranking 和窄任务 sub-agent。
- 能力矩阵按精确 model page 记录：mini 页面列出 `tool_search` 与 `computer_use`，nano 当前页面未列出这两项，不能从“GPT-5.4 家族”自动继承；两者的 function calling、web search、file search、code interpreter、hosted shell、apply patch、skills 和 MCP 仍需按 endpoint/宿主权限做 capability probe。
- 本轮新增 [`gpt-5.4-mini-nano-source-notes.md`](research/model-update-2026-09/gpt-5.4-mini-nano-source-notes.md)，并同步模型清单、榜单解释、来源索引、[`gpt-5.4-source-notes.md`](research/model-update-2026-09/gpt-5.4-source-notes.md)、[`plan.md`](plan.md) 和现有书系配套。mini/nano 复用 GPT-5.4 的 Agent serving、reasoning、routing 和评测章节，不新增重复 Transformer 章节。
- 当前状态：**资料级闭环（AA 单榜关联档位）**。参数、层/专家/注意力结构、训练与后训练 recipe、system card、公开权重、独立技术报告、mini/nano 精确 DataCurve Agent 评测、生产 kernel、目标硬件 profiling 和线上 acceptance rate 仍待核验；下一轮继续只从两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20：GPT-5.5 Instant (June 2026) 下一锚点与官方身份复核

- 本轮从 Artificial Analysis 的精确条目选择 `GPT-5.5 Instant (June 2026)`，同时复核 May 历史 revision；没有从 OpenAI 官方目录另发现模型。June 详情快照为 `3,847,333` bytes、SHA-256 `9c6fa94b9c42a89eb7c8f59bad08a30c168bf0860971fbb666c4be47e6a05599`，release date `2026-06-25`，AA Intelligence Index `26.0135173401368`，约 `130.9925 output tokens/s`，400K context，页面未标记 deprecated。May 详情为 `3,548,855` bytes、SHA-256 `ed410b6cecbd8eb1dcaf65715547771bcb7435e444f6c5ba7ce815e16614e51e`，release date `2026-05-05`，AA Index `22.6863693000789`，已 deprecated。
- 两个 AA 页面都显示 text/image → text、reasoning、知识截止 2025-08-31 和 `$5/M` input、`$30/M` output；这些是 Artificial Analysis 的第三方目录/测量字段或其 API 价格引用，不是 OpenAI 内部架构、训练机制或正式 model ID 证明。
- 通过 `7890` 获取的 OpenAI 官方页面确认的是 `gpt-5.5` 与 snapshot `gpt-5.5-2026-04-23`；访问精确 `https://developers.openai.com/api/docs/models/gpt-5.5-instant.md` 返回 HTTP 404（9 bytes）。因此不能把 Instant June 自动改写成 `gpt-5.5`，也不能把 GPT-5.5 base 的 1.05M context、tool catalog 或 DeepSWE 67% 结果迁移给 Instant。
- DataCurve 当前快照没有 `gpt_5_5_instant` 精确 `mini_swe_agent` 行；GPT-5.5 base 的 xhigh 结果仍绑定 base model、harness、工具、环境和 verifier。当前不新增 Instant 专属 Transformer 章节，研究底稿见 [`gpt-5.5-source-notes.md`](research/model-update-2026-09/gpt-5.5-source-notes.md) 第 9 节。
- 当前状态：**榜单级关联配置，官方身份待确认**。下一步优先查找 OpenAI 是否出现与 June revision 对应的官方发布、API alias、model card、system card 或技术说明；若仍无专属资料，则保留负面证据并切换到下一个重点厂商锚点。

## 2026-09-20：GLM-5.1 资料级闭环

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。按照两榜唯一发现规则，选择 Artificial Analysis 的 [`GLM-5.1`](https://artificialanalysis.ai/models/glm-5-1)，不从 Z.ai 官方目录或论文搜索另发现模型。
- 三条代理重新抓取 Artificial Analysis GLM-5.1 详情页，均返回 HTTP 200 且逐字节一致：`3,935,095` bytes，SHA-256 `f6b1ca673b777602013f13eb348a7c48773120e13684eadcbd7220b50b6c137d`。页面标题为 `GLM-5.1 (Reasoning)`，release 字段为 2026 年 4 月；第三方字段约为 744B total/40B active、200K context、Intelligence Index `26.0585912980095`、约 39.9 tokens/s。上述是 AA 配置/provider 字段，不是官方参数或裸模型能力。
- DataCurve 快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，当前没有精确 `mini_swe_agent_glm_5_1_*` 行；不迁移 GLM-5/5.2/5.3/5.3-Flash 的 Pass@1、成本、输出 token 或 Agent steps。
- 通过 7890 成功抓取 Z.ai GLM-5.1 Markdown 文档（`19,347` bytes，SHA-256 `69958d7d95452d3853903524612a0c6a83f368c79417a6889b9c531f45ab3d01`）、Hugging Face README（`10,751` bytes，SHA-256 `2cb148b732f596716f8c9366edf072e35746de6d06fe69c92301ddb492557077`）和 config（`1,379` bytes，SHA-256 `726e45e28d1be1636a5834048a4101a2cfe384819ad2ebe08d41f6fe04656526`）。核验 200K/128K、`glm-5.1` API ID、MIT、`GlmMoeDsaForCausalLM`、78 层、前三层 dense、256 routed/top-8/1 shared、`q_lora_rank=2048`、`kv_lora_rank=512`、`index_topk=2048` 和 202752 positions。
- 补齐官方 [Deep Thinking](https://docs.z.ai/guides/capabilities/thinking.md)、[Function Calling](https://docs.z.ai/guides/capabilities/function-calling.md)、[Context Caching](https://docs.z.ai/guides/capabilities/cache.md)、[Pricing](https://docs.z.ai/guides/overview/pricing.md) 和 [Release Notes](https://docs.z.ai/release-notes/new-released.md)。重点技术线是 long-horizon 约 8 小时 Agent、实验—分析—优化闭环、multi-turn SFT/RL/process-quality evaluation framework、thinking/tool/cache 协议；Z.ai 当前价格表为 GLM-5.1 input `$1.40`、cached input `$0.26`、output `$4.40`/每百万 token。
- Z.ai 自报 SWE-Bench Pro `58.4`、Linux desktop 655 次迭代/6.9× 向量数据库吞吐、KernelBench Level 3 `3.6×` 对比 `torch.compile` max-autotune `1.49×` 均保留为发布方结果，不能与 AA/DataCurve 拼接。`thinking.type=enabled/disabled` 是请求级模式；当前文档将 `reasoning_effort` 支持列为 GLM-5.2 及以上，不能迁移给 GLM-5.1。
- 证据边界已修正：2026-09-20 的一次访问曾使 `https://z.ai/blog/glm-5.1` 返回 HTTP 404；2026-09-21 已通过三条代理取得 HTTP 200 的官方博客壳和正文 JS 资源，因此旧 404 只代表当时的线路/页面状态，不能继续写成当前页面不可得。模型卡链接 GLM-5 技术报告；arXiv `all:"GLM-5.1"` 精确检索没有 GLM-5.1 专属技术报告。1234 对 HF 的 CONNECT 超时只记为线路失败，不解释为页面不存在。
- 新增 [`glm-5.1-source-notes.md`](research/model-update-2026-09/glm-5.1-source-notes.md)，同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md) 和 [`plan.md`](plan.md)。不新增重复 Transformer 正式章节；内容复用 GLM-5 DSA/MoE、Agentic Engineering、reasoning、工具协议和 serving 主线。
- 当前状态：**AA 单榜资料级闭环**。完整参数独立披露、DSA indexer loss/recall、完整训练/后训练 recipe、过程质量 verifier、生产 kernel、硬件 profiling、8 小时 harness、线上 tool acceptance、精确 DataCurve 行和独立复现仍待核验；下一锚点仍只从两个排行榜的八家重点厂商条目选择。

## 2026-09-20：Grok 4.20 0309 v2 AA 单榜资料级闭环

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。按照“两榜唯一发现”规则，从 Artificial Analysis 的精确 [`grok-4-20`](https://artificialanalysis.ai/models/grok-4-20) 条目选择 `Grok 4.20 0309 v2 (Reasoning)`，没有从 xAI 官方目录另发现模型。
- 通过 `10.24.27.134:7890`、`10.24.27.134:8098` 和 `10.237.126.170:1234` 重新抓取 Artificial Analysis，均 HTTP 200、逐字节一致：513,564 bytes，SHA-256 `2fc88248faf152f46f659310f300f2fe3e5b84e419babb9c8c7dc3752452cdd8`。AA 字段为 `releaseDate: 2026-04-07`、estimated Intelligence Index `25.6550155187053`、2M context、约 `97.0079` output tokens/s、TTFT `22.7098s`、input/output `$1.25/$2.50`；当前目录还标记 deprecated -> `grok-4-3`，只作为第三方状态记录。
- 三条代理重新获取 DataCurve，均 HTTP 200、逐字节一致；当前响应 `268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。页面有 Grok 4.5 high 与 Grok 4.6 low/medium/high/xhigh，但没有精确 `mini_swe_agent_grok_4_20_*` 行；因此不迁移相邻版本的 Pass@1、成本、输出 token 或 Agent steps。
- 通过 7890 获取 xAI [Grok 4.20 模型页](https://docs.x.ai/developers/models/grok-4.20)及 Markdown、模型注册表、[Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning.md)、[Multi Agent](https://docs.x.ai/developers/model-capabilities/text/multi-agent.md)、[Context Compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction.md)、Tools、Prompt Caching 和 [Release Notes](https://docs.x.ai/developers/release-notes.md)。官方主 ID 为 `grok-4.20-0309-reasoning`，并区分 non-reasoning 与 `grok-4.20-multi-agent-0309`；官方模型页写 1M maximum prompt/context、text/image -> text、function calling、structured output、reasoning、Batch、200K long-context price threshold、37 RPS/10M TPM。
- 官方新技术主线是 Realtime Multi-agent Research beta：多个专门 Agent 并行搜索、分析、交叉核验，由 leader agent 汇总；`agent_count=4/16` 或 Responses `reasoning.effort=low/medium` vs `high/xhigh` 控制协作规模，而不是普通模型的思考深度或 MoE expert 数量。内置 web/X/code/collections 工具由 xAI server 执行，子 Agent 中间状态可通过 `use_encrypted_content` 以 opaque encrypted state 保留。
- 已提取面试主线：Responses compaction 返回必须原样回放的单个 opaque item；自动 prompt caching 与 `x-grok-conv-id`/`prompt_cache_key` 是 provider 层复用；function calling 默认 parallel、streaming 时完整 call 在单 chunk 返回；Web/X Search 的 allowlist/filter/citation；Code Execution 的 Python sandbox；Remote MCP 的 Streaming HTTP/SSE、`allowed_tools` 最小权限和 tool schema/context 成本；混合 server/client tools 中 client call 会暂停请求并重置后续 request 的 `max_turns`。
- 证据差异已单独登记：AA 写 2M context，xAI 官方服务字段写 1M；普通 4.20 的完整 reasoning effort 档位未由专属页明确列出，不迁移 Grok 4.6 的 effort 表。xAI Release Notes 的 March 2026 条目确认 Grok 4.20/Multi-agent live，但 `x.ai/news/grok-4-20`、`grok-4.20`、`grok-4-20-multi-agent` 本轮均 HTTP 404。
- arXiv 标题精确检索 [`"Grok 4.20"`](https://arxiv.org/search/?query=%22Grok+4.20%22&searchtype=title)无结果；全文命中的 7 篇论文只是外部评测/使用。没有找到 Grok 4.20 专属技术报告、参数架构、公开权重、完整训练 recipe、生产 kernel、硬件 profiling、线上 acceptance 或独立 DataCurve Agent benchmark。
- 新增 [`grok-4.20-source-notes.md`](research/model-update-2026-09/grok-4.20-source-notes.md)，同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan.md`](plan.md) 和书系配套。当前状态：**AA 单榜资料级闭环**，不新增重复 Transformer 正式章节；下一锚点仍从两个排行榜的八家重点厂商条目选择。

## 2026-09-20：Grok 4.20 全局资料配套同步完成

- 在已有 Grok 4.20 研究笔记、来源索引、模型清单、榜单解释和书系章节基础上，补齐 [`INTERVIEW_BANK.md`](INTERVIEW_BANK.md)、[`EXERCISES.md`](EXERCISES.md)、[`PAPERS.md`](PAPERS.md)、[`KNOWLEDGE_GRAPH.md`](KNOWLEDGE_GRAPH.md)、[`GLOSSARY_EN_ZH.md`](GLOSSARY_EN_ZH.md) 和 [`PROJECTS.md`](PROJECTS.md)。
- 题库、练习和项目覆盖 4/16 agent-count 消融、leader/sub-agent trace、opaque encrypted state、compaction replay、prompt cache、server/client tools、跨请求 `max_turns`、MCP `allowed_tools`、2M/1M context discrepancy 和 DataCurve 精确行缺失；论文索引明确没有 Grok 4.20 专属 arXiv 技术报告。
- 全局资料继续保留证据边界：multi-agent 是 xAI runtime/编排能力，不是已公开的 MoE expert 数量或普通 reasoning 深度；AA 与官方 context 字段按 source/endpoint 分账；不迁移 Grok 4.5/4.6 的 DeepSWE 结果。
- 当前状态仍为 **AA 单榜资料级闭环**；暂无独立 Grok 4.20 架构/训练报告、公开权重、完整 recipe、生产 kernel、硬件 profiling、线上 acceptance 或精确 DataCurve Agent 行。下一轮只从两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20：Gemini 3.8 Flash 当前活动锚点与复验完成

- 本轮确认工作目录为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。模型发现规则仍只允许 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜，重点厂商仍为 OpenAI、Anthropic、GLM/Z.ai、Qwen、Kimi/Moonshot、DeepSeek、Gemini 和 Grok。
- 三条代理重新抓取 AA `/zh` 首页和 DataCurve，均 HTTP 200、逐字节一致：AA 首页 `1,777,695` bytes / SHA-256 `82b890a93772e857bfafd0aa0a96994d39afb4fcdd2a9fdf2d4e6ec01f9c7912`；DataCurve `268,571` bytes / SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。
- AA Gemini 3.8 high/medium/low 详情本轮分别为 `517,803` / `511,687` / `509,053` bytes，哈希为 `0618989e412ae947b3a9f499c1cdae16e4ea3f5ee65a25def360249ae33985ef` / `d96907dd254166f038084e7586e30ab869fc8a3225501c37c958650859aec6a8` / `ef35f6f567518bd1b9e9e1bc17e4826ea479b7d1e7868bc35f8daf7c464ee5b3`；当前字段仍为 2026-09-02 release、约 1M context、FAQ 指数约 41。三个 effort 行归并为一个基础模型。
- DataCurve 精确对象为 `gemini-3-8-flash` + `mini-swe-agent` + `high`：`n_runs=4`、`n_attempted=447`、Pass@1 `0.738255033557047`、Pass@4 `0.8584070796460177`、平均成本 `$2.362349413758389`、平均输出 `143242.6644295302` token、平均 `166.3131991051454` steps、median `161`。这些数字绑定 harness、工具、任务环境和 verifier，不外推到 low/medium 或其他 Gemini 版本。
- 通过 `7890` 重新核验 Google 模型页、Thinking、Interactions，响应分别为 `23,603` / `37,414` / `23,899` bytes，哈希已写入研究笔记与来源索引。官方字段确认 `low/medium/high`、`minimal` 不支持、thought summary/signature、Interactions steps、工具组合、Computer Use、1M 输入和 caching；本轮不把这些 API/runtime 能力写成内部架构或训练算法。
- 已补齐 [`gemini-3.8-flash-source-notes.md`](research/model-update-2026-09/gemini-3.8-flash-source-notes.md) 的 2026-09-20 快照复验，并同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`INTERVIEW_BANK.md`](INTERVIEW_BANK.md)、[`EXERCISES.md`](EXERCISES.md)、[`PAPERS.md`](PAPERS.md)、[`KNOWLEDGE_GRAPH.md`](KNOWLEDGE_GRAPH.md)、[`GLOSSARY_EN_ZH.md`](GLOSSARY_EN_ZH.md) 和 [`PROJECTS.md`](PROJECTS.md)。
- 书系已将 Gemini 3.8 的案例接入第十六册 `14.28`（thinking/signature）、第十七册 `7.30`（Interactions/Computer Use）、第二十册 `19.31`（harness-aware evaluation）和第二十四册 `32.41`（state/cache/tool serving）；由于没有独立 3.8 架构/训练报告，不新建重复的 Gemini 专属 Transformer 章节。
- 当前状态：**资料级闭环**。尚无独立 Gemini 3.8 架构/参数/训练报告、生产 kernel、目标硬件 profiling、线上 acceptance 或独立复现；下一步做真实 API capability probe、thinking 消融、Interactions replay、工具组合和长上下文/cache 实验，不新增重复正式章节。目标仍在持续推进，不标记为 complete。

## 2026-09-20：GLM-5.3 当前活动锚点与资料级闭环

- 本轮确认工作目录为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。模型发现仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商范围不变。
- Artificial Analysis 精确条目为 [`glm-5-3`](https://artificialanalysis.ai/models/glm-5-3) 的 max 配置；三代理详情快照逐字节一致，`3,928,329` bytes，SHA-256 `090279e870a3ebb72c7a69f24963f87fe10d63595642a4c61959b911f2a3c3a2`。第三方/provider 字段为 release `2026-08-18`、Intelligence Index `44.777392385614`、约 `72.1152 tokens/s`、TTFT 约 `2.99s`、1M context 和 `$1.40/$4.40`。
- DataCurve 当前快照三代理逐字节一致，`268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。精确 `mini_swe_agent_glm_5_3_max` 行为 451 次尝试、311 次通过、Pass@1 `68.9579%`、Pass@4 `87.6106%`、平均成本约 `$3.9934`、平均输出约 `80.4K` token、平均 124.47 steps；结果绑定 harness、工具、环境、预算和 verifier。
- 已复验 Z.ai GLM-5.3 文档、迁移指南、Thinking/Function Calling/Tool Streaming/Context Caching/Structured Output、release notes、官方博客资源、Hugging Face README/config、`slime` 和 IndexCache/SAO 关联论文。新增面试主线包括：同 GLM-5.2 基座的后训练归因、可执行长周期环境、judge 与无 reference verifier、oracle/no-op/unsolved-state、reward shortcut audit、train/rollout/data-buffer 单 dataflow、logprob 一致性、`thinking.type=enabled`、`low/high/max` effort 和 `tool_stream` delta 重组。
- 证据边界：SAO 直接论文对象为 GLM-5.2，不能写成 GLM-5.3 独有算法；IndexCache 是 DSA serving 关联证据；Z.ai benchmark、`1e-7` logprob 和 `>2.3x` throughput 是发布方结果/描述；HF config 字段不等于完整参数、训练 recipe 或生产 kernel。
- 已完成研究笔记、来源索引、模型清单、榜单解释、第十六册第 20 章和全局题库/练习/术语/项目/论文/知识图谱同步。当前状态：**双榜资料级闭环**；完整 recipe、5.3 专属 SAO/compaction 定义、production kernel、硬件 profiling、独立 benchmark 和线上 acceptance 仍待核验。下一轮仍只从两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20：DeepSeek V3.2 当前活动锚点与实现证据补强

- 本轮选择 Artificial Analysis 精确 [`deepseek-v3-2`](https://artificialanalysis.ai/models/deepseek-v3-2) 条目；DataCurve 当前没有精确 `mini_swe_agent_deepseek_v3_2_*` 行，不迁移 V3.1/V4 的 Pass@1、成本、输出 token 或 Agent steps。AA 详情快照为 `3,625,720` bytes、SHA-256 `7d4fd97ed7f2efbc9ab2f001072b353f00e2476cb884abe0d8185f16d7562052`；`Non-reasoning`、128K、约 648B/37B 和指数 `16.043537719683` 是第三方目录字段。
- 已复核并入库 V3.2 模型卡、官方技术报告、V3.2-Exp inference、TileLang 示例、DeepGEMM/FlashMLA PR 和 vLLM recipe。正式章节补充 DSA 的 FP8 indexer、non-interleaved indexer RoPE、`fp8_index`、causal/top-k、sparse MLA gather、prefill MHA/decode MQA、latent/positional cache、FP8 KV cache 与 dense fallback。
- 本轮明确最终 V3.2 与实验 artifact 分层：固定官方 `config.json` 与 V3.2-Exp inference 都是 `q_lora_rank=1536`；前版把最终字段读成不同数值的说法已纠正。实验代码、kernel 和 vLLM recipe 证明公开实现路径，不等于最终生产性能或裸模型能力。`DP=8, EP=8, TP=1`、TP fallback、GSM8K recipe 和 kernel 局部结果必须绑定版本、硬件和 harness。
- `thinking with tools` 的 reasoning/tool call/tool result/final 边界已与 V3.2-Speciale 的“不支持 tool calling”分开记录；宿主仍需 schema、权限、确认、超时、重试、回灌和 artifact verifier。研究笔记、全局配套和第 21 册第 19 章已同步。
- 当前状态：**AA 单榜资料级闭环**。完整层排布、最终 production kernel、召回曲线、完整训练/RL recipe、线上 tool acceptance、硬件 profiling、PDF 图表/公式视觉复核和独立 benchmark 仍待核验；下一轮仍只从两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20：Kimi K2.6 Kimi Code harness 补证与 K2.8 边界

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作；这是既有 Kimi K2.6 补证，不新增模型锚点，当前活动锚点仍是 `DeepSeek V3.2`。只把官方 Kimi Code 文档作为已发现 K2.6 的周边技术资料，不把官方产品目录当作新模型发现入口。
- [Kimi Code 模型配置](https://www.kimi.com/code/docs/kimi-code/models.html) 当前列出 K3、K2.8 Preview、K2.7 Code HighSpeed 三类、4 个 model ID：`k3`、`k3-256k`、`kimi-for-coding`、`kimi-for-coding-highspeed`。K2.8 Preview 虽在官方产品文档出现，但 2026-09-20 AA/DataCurve 快照没有精确 K2.8 条目，因此不进入候选清单、不建立 K2.8 研究笔记。
- [Kimi Code 内置工具/AgentSwarm](https://www.kimi.com/code/docs/kimi-code-cli/reference/tools.html) 与 [Agent/subagent](https://www.kimi.com/code/docs/kimi-code-cli/customization/agents.html) 补齐当前 harness 证据：最多 128 个 subagent、默认 2 小时超时、`resume_agent_ids`、模型池、聚合报告、AgentSwarm 必须是该响应中的唯一工具调用、默认从 5 个并发开始且每 700ms 增加 1 个、`KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY` 限制并发；工具和 subagent 列表在实际派发前再次校验，权限规则独立于可见工具列表。
- [Kimi Code 会话与上下文](https://www.kimi.com/code/docs/kimi-code-cli/guides/sessions.html) 补齐 `state.json`、`wire.jsonl` 事件流、`--continue`、`--session`、`/compact`、`/fork`、导出和工具 schema/MCP 清单回放。它支持“模型 + harness + 可恢复状态 + verifier”的面试拆解，但没有公开 K2.6 专属 coordinator 内部实现。
- 证据分账：K2.6 官方博客的 300 sub-agents/4,000 coordinated steps 是发布方案例；Kimi Code 当前 AgentSwarm 的 128 是 CLI 工具约束。二者不合并、不取最大值写成 K2.6 固有上限，也不把 subagent 数量写成 MoE expert 数量。完整细节与快照哈希已写入 [`kimi-k2.6-source-notes.md`](research/model-update-2026-09/kimi-k2.6-source-notes.md) 和第二十一册第 90 章。
- 当前状态仍为：`Kimi K2.6` **AA 单榜资料级闭环**；K2.6 专属训练/后训练 recipe、production kernel、INT4 profiling、Agent Swarm 内部 coordinator、线上 acceptance、精确 DataCurve 行和独立复现待核验。下一轮仍只从两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20：DeepSeek V3.2 官方 config 纠错与 DSML encoding 补证

- 本轮定位并修正 V3.2 专属证据错误：官方固定 revision `a7e62ac04ecb2c0a54d736dc46601c5606cf10a6` 的 `config.json`（1,552 bytes，SHA-256 `c7fa8b191e9936d8e6a57d864baab82b792fae16a116416cdd3a75ba76bc5af1`）确认最终 `q_lora_rank=1536`，与 V3.2-Exp inference config 同值；此前把最终字段读成不同数值的记录已同步纠正。
- 同步写入官方配置的 61 layers、前三层 dense、256 routed/top-8/1 shared、64 index heads、`index_head_dim=128`、`index_topk=2048`、163840 positions、YaRN factor 40、BF16 和 FP8 `e4m3`/`ue8m0`、128×128 block 字段。以上是固定 config 证据，不等于完整训练 recipe 或 production kernel。
- 固定 `encoding/encoding_dsv32.py` 快照为 14,317 bytes，SHA-256 `5e068c2ba2a6e5ebe37a49bb005650c507e7935d77a32f3f7c11ee071498b370`；新增 DSML function-call/result、role、`<think>`/`reasoning_content` 和 JSON 参数分支，并在第 19 章新增 19.33，明确 parser、schema、权限、执行器和 verifier 的责任边界。
- 已更新 [`deepseek-v3.2-source-notes.md`](research/model-update-2026-09/deepseek-v3.2-source-notes.md)、第 21 册第 19 章、[`plan.md`](plan.md)、模型清单、榜单解释、来源索引、[`PAPERS.md`](PAPERS.md)、[`PROJECTS.md`](PROJECTS.md)、[`KNOWLEDGE_GRAPH.md`](KNOWLEDGE_GRAPH.md) 和 [`INTERVIEW_BANK.md`](INTERVIEW_BANK.md)。当前目标仍 active；PDF 视觉复核、完整 production kernel、召回曲线、硬件 profiling、线上 tool acceptance 和完整训练/RL recipe 仍待核验。

## 2026-09-20：两榜当前时点联网复验

- 三条代理重新抓取 Artificial Analysis `/zh`：均 HTTP 200，1,777,695 bytes，SHA-256 `82b890a93772e857bfafd0aa0a96994d39afb4fcdd2a9fdf2d4e6ec01f9c7912`，逐字节一致。
- 三条代理重新抓取 DataCurve DeepSWE：均 HTTP 200，268,571 bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，逐字节一致。
- 当前 AA 首页前列的 Fable 5.1、GPT-6 Astra、Claude Opus 5、GLM-5.3、Grok 4.6、Kimi K3、Gemini 3.8 Flash、DeepSeek V4.1 Flash 等重点条目，以及 DataCurve 的重点精确行，均已在现有模型清单/研究笔记中有记录；本轮没有新增模型，也没有从排行榜之外引入候选。
- （历史复验记录）当时活动锚点仍为 DeepSeek V3.2；V3.2 的官方 config 纠错和 DSML encoding 补证已完成。此处只记录当时的研究入口；后续已依次切换到 DeepSeek V4 Pro 0813 和 DeepSeek V4.1-Flash，当前活动锚点以文件顶部和最新记录为准。

## 2026-09-20：Kimi K3 权重 revision、FlashKDA 与 vLLM serving 补证

- 本轮沿两个排行榜中已存在的 Kimi K3 条目推进，没有从 HF、GitHub 或 vLLM 页面另发现模型。工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。
- 官方 HF API 固定 revision 为 `f831ab66814297da540d832a5235f8e904f29d06`，metadata 响应 SHA-256 为 `24b2606b8f2604828ce078a93f76209b1ce97e366e4cfa0ea20176975d9ac07e`；固定 `config.json` 为 7,006 bytes，SHA-256 `9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213`。配置确认 `KimiK3ForConditionalGeneration`、93 layers、69 KDA + 24 full-attention/Gated MLA、`q_lora_rank=1536`、`kv_lora_rank=512`、896 experts/top-16/2 shared、1,048,576 context；完整 safetensors 未下载。
- FlashKDA README 为 4,431 bytes、SHA-256 `fc56ca9a3cd1786d0ff526a12be21500d79ee671e5f0eeb8352fad23d3ba8fb2`；master 最新可见 commit `7afb9f454f160a6c4bbc0999beca0a8c40a38934` 将 fp16 Neumann inverse 换为 `8x8 fp32 forward substitution + 16x16 bf16 merge`。README 要求 SM90+/CUDA 12.9+/PyTorch 2.4+，支持 `chunk_kda`、recurrent state 和变长 batch；H20/GB200 固定长度 benchmark 相对 baseline 为 `1.85x`/`2.31x`，属于仓库发布结果，不是本地或端到端复现。
- vLLM K3 recipe 页面 SHA-256 `98e380fbbc8b85d236b60b9a4570ace01359745451cb436532f077714899a0e6`，YAML 为 19,414 bytes、SHA-256 `5b2bcd21b8dfe210ea3f3d514c6a5448e395847d35fcb1f137d3567233e9d227`，更新时间 2026-09-10，要求 vLLM 0.29.0/K3-enabled nightly、CUDA 13/cu130 与 r580+ driver。hybrid KV manager 同时管理 MLA attention cache 与 KDA recurrent state；Blackwell 路径包含 FP8 KV、TOKENSPEED MLA、prefix caching 和 `--prefix-match-unit 128`，并按 DCP/TP/TEP/DEP/PP、RDMA/NVLink 记录部署拓扑。
- recipe 明确提示 K3 偶发生成 parser 不期望的 tool-call 格式；面试和教学实现已补充 schema validation、retry/idempotency、权限检查和 verifier 的分层。vLLM `0.29.0` stable release/source implementation entry 已由 PyPI metadata 与 v0.29.0 tag 固定；完整训练/optimizer recipe、目标硬件 profiling、独立 benchmark、完整权重加载和线上 tool-call acceptance 仍待核验。
- 已完成同步：`kimi-k3-source-notes.md`、`inventory-interpretation.md`、`source-index.md`、`model-inventory.md`、`plan.md`、本文件、`PAPERS.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、`INTERVIEW_BANK.md`、`EXERCISES.md` 和 `GLOSSARY_EN_ZH.md`。下一步是在验证本轮链接/Markdown/代码块后，再从两张排行榜选择下一个未闭环重点锚点；目标保持 active。

## 2026-09-20：DeepSeek V4 Pro reference implementation 与 serving manifest 补证

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径与 `/home/zzc/llm-from-zero-to-interview` 等同；没有新建模型候选，仍沿两个排行榜已经发现的 `deepseek-v4-pro` 推进。
- 固定官方 HF revision `b5968e9190ef611bbf34a7229255be88a0e937c1`，确认 64 个 safetensors 分片和 `1,598,839,674,782` bytes metadata storage；完整权重未下载。`README.md`、`config.json`、encoding、inference config/model/kernel 和 generation config 的文件级哈希已写入研究笔记。
- 从官方 `inference/model.py` 补齐 gated KV compressor、ratio-4 overlap state、learned causal/top-k indexer、MLA + compressed sparse attention + 128-token window、前三层 hash routing、后续 `sqrtsoftplus` routing、top-6 + shared expert、MTP 和 Hyper-Connections/Sinkhorn 证据；从 `kernel.py` 补齐 block FP8/FP4 quantization、GEMM、online softmax 和 HC Sinkhorn。
- 从官方 `encoding/encoding_dsv4.py` 补齐 DSML tool-call、tool role -> `<tool_result>`、string/JSON parameter 分支、`<think>`/reasoning 保留规则和严格 malformed-output 失败边界。已做 AST 与无 CUDA encode/parse round trip；未运行 TileLang/CUDA 或完整权重推理。
- 已更新 [`deepseek-v4-source-notes.md`](research/model-update-2026-09/deepseek-v4-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan.md`](plan.md)、第二十一册第 77 章、第二十四册第 32.42 节及全局论文/项目/知识图谱/题库/练习/术语。
- 该段记录当时的 `DeepSeek V4 Pro 0813` 阶段，状态为**双榜内容专题闭环 + reference implementation 证据补强**。当时待核验项包括完整权重加载、固定 kernel/convert commit、目标 GPU 的 FP4/FP8 profiling、compression/index/evidence recall、线上 tool acceptance、完整训练 recipe 和 API snapshot 差异；当前活动锚点已在后续记录切换为 `DeepSeek V4.1-Flash`，目标保持 active。

## 2026-09-20：DeepSeek V4.1-Flash 固定 revision 实现与 serving 边界补证

- 已确认当前工作目录 `/data/zzc/llm-from-zero-to-interview` 与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。本轮仍遵守“模型只从 Artificial Analysis 与 DataCurve DeepSWE 发现”的规则，选择 AA 已有精确 slug `deepseek-v4-1-flash`，没有从 HF 仓库另发现模型。
- 两榜当前时点复验快照：Artificial Analysis `/zh` 三代理均 HTTP 200、`1,777,695` bytes、SHA-256 `82b890a93772e857bfafd0aa0a96994d39afb4fcdd2a9fdf2d4e6ec01f9c7912`；DataCurve 三代理均 HTTP 200、`268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，结果逐字节一致。DataCurve 页面没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，不迁移 V4 Pro/V4 Flash 的 Pass@1、成本、输出 token 或 Agent steps。
- 通过 `10.24.27.134:7890` 读取固定 HF revision `dba1be0a40aa45a94ad051997016db3960a90277` 的 API 与源码；API response 为 `6,687` bytes，SHA-256 `fb3aefa7794da9101d0253ccc4e6e36fbace841778f6857b1370cb2880235d63`，列出 48 个 safetensors 分片。完整权重没有下载；HF inventory 的 dtype bucket 统计不替代模型卡 `552B backbone` 或本地参数加载。
- 新增实现证据：固定 revision 的 `inference/README.md`、`inference/config.json`、`model.py`、TileLang `kernel.py`、`convert.py`、Engram/vision、`encoding.py`、`evaluation/README.md` 和 `dsh-minimal.patch` 的文件大小/SHA-256 已写入研究笔记、来源索引和模型盘点。`model.py` 能看到 SWA ring、compressed KV overlap state、两级 candidate/index top-k、sparse attention、MoE、Engram、mHC/Sinkhorn 与 DSpark block；`kernel.py` 能看到 FP4/FP8 quantization/GEMM、sparse online softmax 和 Sinkhorn 路径。
- 关键边界已经写入第二十一册第 81 章与第二十四册第 32.43 节：官方 README 把 inference tree 定义为 readable reference implementation；`model.py` 虽有 `forward_spec`，但 `generate.py` 仍调用普通 `model.forward`，没有完整 draft/verify/rollback scheduler。因此当前不宣称 DSpark 已完成 speculative serving 或吞吐复现；EPD、host-DRAM SWA pool、global KV、alias/served-model 和 `not_applicable` DataCurve manifest 分开记录。
- 本地校验：下载的 Python 源码 `py_compile` 通过；encoding smoke test 覆盖 DSML 前导空格、`reasoning_effort=max`、中途 system message 和 thinking parse，结果通过。没有执行 CUDA、TileLang、完整权重、目标硬件 profiling、线上 API 或独立 benchmark。
- 已同步：[`deepseek-v4.1-flash-source-notes.md`](research/model-update-2026-09/deepseek-v4.1-flash-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan.md`](plan.md)、第二十一册第 81 章、第二十四册第 32.43 节、`PAPERS.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、`INTERVIEW_BANK.md`、`EXERCISES.md` 和 `GLOSSARY_EN_ZH.md`。
- 当前状态：**内容专题闭环 + reference implementation 证据补强（AA 单榜）**。仍待核验完整权重加载、生产 kernel 覆盖、候选/Top-K recall、FP4 误差、DSpark 接受长度与 rollback、EPD 真实调度、目标硬件 profiling、线上 tool acceptance 和独立 benchmark；goal 保持 active，下一轮仍只从两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20：两榜重点厂商前列审计

- 复核 2026-09-20 的 Artificial Analysis 首页快照（`1,777,695` bytes，SHA-256 `82b890a93772e857bfafd0aa0a96994d39afb4fcdd2a9fdf2d4e6ec01f9c7912`）和 DataCurve DeepSWE 快照（`268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`）。按当前重点厂商过滤，AA 前列依次覆盖 `Claude Fable 5.1`、`GPT-6 Astra`、`Claude Opus 5`、`GLM-5.3`、`Grok 4.6`、`Kimi K3`、`Gemini 3.8 Flash` 和 `DeepSeek V4.1 Flash`；Muse Spark 等非重点厂商不进入本项目更新队列。
- 上述八个重点条目均已有研究笔记、官方来源和书系/配套同步；本轮审计没有发现需要新建研究笔记的重点模型，也没有把 effort、fallback、provider 或关联 artifact 误计为新的基础模型。
- 因此当前活动锚点仍为 `DeepSeek V4.1-Flash`。下一步继续补齐实现级待核验项：完整权重加载、生产 kernel 覆盖、candidate/index Top-K recall、FP4 误差、DSpark draft/verify/rollback 调度、EPD 实际调度、目标硬件 profiling、线上 tool acceptance 和独立 benchmark。goal 保持 active。

## 2026-09-20：DeepSeek V4.1-Flash 召回与 FP4 教学实验

- 新增标准库脚本 [`deepseek_v41_cache_demo.py`](research/model-update-2026-09/code/deepseek_v41_cache_demo.py)，不下载权重、不使用 CUDA/TileLang，只用合成 indexer 分数和 toy KV 向量演示两个待核验指标：candidate-pool recall 与候选池内 conditional Top-K recall，以及分组 E2M1-like FP4 的 MSE/max error。
- 脚本大小 `6,317` bytes，SHA-256 `6c65d44f2f092369ac8b0cdf83a58fe4c2dad38e46698e5d7f1badc209c9517e`；运行和 `py_compile` 均通过，固定输出为 candidate recall `0.600`、conditional Top-K recall `0.667`、端到端 recall `0.400`、toy FP4-like MSE `0.025862`、最大绝对误差 `0.300000`。
- 已将脚本链接和证据边界同步到 DeepSeek V4.1 研究笔记、来源索引、模型盘点、榜单解释和第二十一册第 81 章。该实验只验证指标分解和教学代码，不升级为真实模型召回、FP4 质量、DSpark acceptance 或硬件性能证据；goal 保持 active。

## 2026-09-20 当前榜单审计与下一步

对 2026-09-20 两榜快照按重点厂商过滤后，Artificial Analysis 前列的重点条目为 `Claude Fable 5.1`、`GPT-6 Astra`、`Claude Opus 5`、`GLM-5.3`、`Grok 4.6`、`Kimi K3`、`Gemini 3.8 Flash` 和 `DeepSeek V4.1 Flash`；DataCurve 当前快照没有产生需要新增研究笔记的重点精确模型行。八个条目均已有研究笔记和闭环状态，因此不新增非重点厂商模型，也不为已有条目的 effort/provider 变体重复建档。

当前继续以 `DeepSeek V4.1-Flash` 为活动锚点；已新增 CPU 标准库教学实验，把 candidate-pool/Top-K recall 与 E2M1-like FP4 误差分开验证。下一步优先补它的实现级待核验项：完整权重加载、生产 kernel 覆盖、真实 candidate/index Top-K recall、FP4 误差、DSpark 的真实 draft/verify/rollback 调度、EPD 实际调度、目标硬件 profiling、线上 tool acceptance 和独立 benchmark。若后续两榜出现重点厂商的新 canonical 条目，再按“榜单快照 → 官方资料 → 书系同步”切换锚点。
- [GLM-5.3-Flash 官方资料摘记](research/model-update-2026-09/glm-5.3-flash-source-notes.md)：已复验两个排行榜、Z.ai 官方模型文档/博客、固定 revision 模型卡与配置；已核验 hybrid linear+sparse attention、IndexPool、稀疏 MoE、mHC、原生视觉 coding loop、EPD serving、API thinking/tool streaming 与评测边界；当前已进入“内容专题闭环”收口。
- [GLM-5 官方资料摘记](research/model-update-2026-09/glm-5-source-notes.md)：Artificial Analysis 有精确条目，但 DataCurve 当前没有精确 GLM-5 行；已核验 Z.ai 模型卡、专属技术报告、API/部署资料、DSA、744B/40B MoE、`slime` 异步 RL 和 Agentic Engineering，并新增第二十一册第 89 章及全套配套同步。当前为内容专题闭环（AA 单榜），不迁移 GLM-5.2/5.3 的 DeepSWE 结果。
- [GLM-5.2 官方资料摘记](research/model-update-2026-09/glm-5.2-source-notes.md)：已重新复验 Artificial Analysis 的 max/non-reasoning 与 DataCurve 的 high/max，补齐 Z.ai 官方文档、正确博客正文、IndexShare、MTP speculative decoding、长上下文 serving、critic-based PPO、anti-hack、1M/128K、MCP、缓存和工作流资料；当前为双榜资料级闭环，正式专题为第二十一册第 86 章，SAO 原始论文算法已补证，5.3 专属 compaction、参数/架构和完整训练 recipe 待核验。
- 2026-09-17 补齐 GLM-5.2 正确官方博客正文：IndexShare、MTP speculative decoding、长上下文 serving、critic-based PPO 和 coding-agent anti-hack 已进入研究笔记与第二十一册第 86 章；发布方 benchmark/训练系统描述仍按证据边界记录。
- 2026-09-17 联网复核：三条用户提供的代理当前均无法连接；百度直连 HTTP `200`，而 Artificial Analysis、DataCurve、Z.ai、Google 直连因 DNS 解析失败。该结果只记录为当前网络路径故障，不视为网页不存在；恢复后优先重新抓取两个排行榜，再继续选择八家重点厂商的下一锚点。
- 2026-09-17/18 网络恢复后重新抓取两个唯一排行榜：三条代理对 Artificial Analysis 中文首页均取得 `1,782,207` bytes、SHA-256 `d547f7bde6adc164aaea60026d06e9a59178fc86d89c8ce5333fe8d28ffb755a`；对 DataCurve 均取得 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。页面逐字节一致；八家重点厂商没有新增基础模型候选，DeepSWE 模型集合也没有新增或删除。
- 2026-09-18 GPT-6 Astra 补证：通过 `10.24.27.134:7890` 实际读取 OpenAI 官方模型页、Reasoning、Prompt caching、Conversation state、Compaction 和 Tool search 文档；新增动态 `configuration_update`、reasoning/`phase` 完整 item replay、deferred tool search、缓存前缀和 canonical compaction context 等面试主线。参数、架构、训练 recipe、system card 和技术报告仍待核验；正式内容继续落在第六册第 18 章，不新增重复架构章节。

GPT-5.6 已从“仅候选”升级为“资料级闭环”：Sol/Terra/Luna 的官方模型页、reasoning mode/effort、persisted reasoning、Responses 工具循环、prompt caching、compaction 和开发者博客工程证据已入库。由于没有独立架构或训练报告，暂不新增 GPT-5.6 专属正式章节；正式面试主线映射到第六、七、十六、十七、二十和二十四册。

GPT-5.5 与 GPT-5.5 Pro 已从“仅候选”升级为“资料级闭环”：两个排行榜的配置锚点、官方模型页、专属指南和 API 周边资料已入库。正式面试主线是高效 reasoning 的可测性、outcome-first prompting、tool search、图像 detail、`text.verbosity`、Responses `phase` 回放、compaction、Pro background mode，以及与 GPT-5.6 的 prompt-cache 协议差异；由于没有独立架构或训练报告，暂不新增 GPT-5.5 专属正式章节，映射到第六、七、十六、十七、二十和二十四册。GPT-5.5 Instant May/June 仍单独记为 AA-only 关联配置，官方精确 ID 尚未建立。

GPT-5.4 已从“仅候选”升级为“资料级闭环”：两个排行榜的可追溯配置、OpenAI 官方模型页与专属指南已入库，面试主线覆盖 1M context、reasoning/visible output 分账、deferred `tool_search`、computer use、custom tools/CFG、`allowed_tools`、tool preambles、Responses `phase`、opaque compaction 和模型—协议—harness—执行器分层。GPT-5.4 Pro 只作为关联服务档位记录；mini/nano 已通过各自精确 AA 条目和 OpenAI sibling 模型页独立核验，采用 400K/272K/128K 的专属字段，不继承 base 的 1.05M 或 Agent 评测结果。由于没有 GPT-5.4 家族专属参数、架构、完整训练报告或 mini/nano 精确复现，不新增重复正式章节。

Gemini 3.7 Flash 已从“仅候选”升级为“资料级闭环”：Artificial Analysis 与 DataCurve 的三档配置、Google AI Developers 模型页、DeepMind Model Card、官方评测/安全入口、Interactions/工具/视频文档、arXiv 精确标题负检索和外部使用论文均已入库。由于 Model Card 将架构/训练/软硬件资料指向 Gemini 3.6，暂无独立 Gemini 3.7 技术报告，因此不新增专属架构章节；当前锚点切换为 `gemini-3.7-flash`，下一轮仍从两个排行榜的剩余重点厂商条目中选择。

Claude Fable 5 已从“仅候选”升级为“资料级闭环”：两个排行榜的配置锚点、Anthropic 模型页、发布/重新部署公告、thinking/effort、拒答与 fallback、memory、程序化工具调用、compaction、context editing、task budgets 及 system card 入口已入库。面试主线是 opaque reasoning state、effort 与硬输出预算的区分、业务态拒答、跨模型路由、长任务 harness 和工具结果隔离；由于没有独立架构或训练报告，暂不新增 Fable 5 专属架构章节，映射到第六、七、十六、十七、二十和二十四册。

Claude Opus 4.8 已从“仅候选”升级为“资料级闭环”：两个排行榜的配置锚点、Anthropic 发布公告、system card、Dynamic Workflows 官方博客和研究笔记已入库。面试主线是动态编排图、并行 subagents 与独立验证、effort sweep、任务中途 system entries、prompt cache 保持和“完成声明”与 artifact 验证的分离；由于没有独立架构或训练报告，暂不新增 Opus 4.8 专属架构章节，映射到第六、七、十六、十七、二十和二十四册。当前锚点为 `claude-opus-4.8`，下一步转向 `kimi-k2.7-code`。

Kimi K2.7 Code 已从“仅候选”升级为“资料级闭环”：两个排行榜的条目、官方资源/API 文档、固定 revision 模型卡、MoE/MLA/MoonViT/Native INT4 配置、thinking/preserve-thinking/tool-call 协议、部署指南和研究笔记已入库。面试主线是长周期 coding agent、1T/32B active MoE、MLA 长上下文账本、INT4 部署、可回放 reasoning state 和 harness-aware evaluation；由于没有独立专属训练报告或外部复现，暂不新增 K2.7 Code 专属正式章节，映射到架构、推理、Agent、工具协议、Serving 和评测章节。

Grok 4.5 已从“仅候选”升级为“资料级闭环”：Artificial Analysis 的 `high`、DataCurve 的 `mini_swe_agent_grok_4_5_high`、xAI 发布方对训练/评测/异步 rollout 的公开描述、Grok 4.5 官方模型/API 文档和研究笔记均已入库。面试主线是 reasoning effort 与 opaque state、Responses 30 天状态、context compaction、server-side/client-side tool responsibility、MCP 最小权限、Web/X Search 和 harness-aware evaluation。由于没有独立架构或训练报告，不新增 Grok 4.5 专属正式章节；专属模型页与通用 Reasoning 文档的 `xhigh` 字段冲突保留为待复验项。

Grok 4.6 已从“仅候选”升级为“资料级闭环”：Artificial Analysis/DataCurve 的四档配置、xAI 2026-08-12 发布公告、模型生成数据与 SFT 轨迹筛选、agentic RL、长任务 self-testing/verification、Grok 4.6 API/工具/compaction 文档和论文负检索均已入库。由于没有独立架构或完整训练报告，不新增 Grok 4.6 专属正式章节；该锚点已完成，后续转入 Anthropic 系列。

新模型发现边界：Artificial Analysis 与 DataCurve DeepSWE 是唯一的候选入口。选定锚点后，可以沿官方资料追踪其同系列模型、adapter、论文、训练方法、推理内核和部署工具；这些是“锚点周边技术”，不能绕过两个排行榜变成新的独立候选发现。对排行榜未出现的周边模型（例如 Uno adapter）必须明确标为“官方资料中的关联技术”，不得写成榜单候选或独立模型排名。

Claude Opus 5 已从“仅榜单发现”升级为“资料级闭环”：两个排行榜的 `max`/`mini-swe-agent` 配置、Anthropic 官方模型/API 文档、发布公告、system card 入口、thinking/effort 中途变更、fallback、缓存边界、发布方评测和 arXiv/Research 负检索均已入库。没有公开参数、架构或完整训练报告，因此不新增 Opus 5 专属架构章节；内容映射到 Reasoning、长上下文/缓存、Agent 工具协议、安全 fallback 和公平评测章节。

Claude Sonnet 5 已从“仅榜单发现”升级为“资料级闭环”：两个排行榜的 `max/xhigh/high/medium/low/Non-reasoning` 配置、Anthropic 发布页/模型目录、完整 API/Agent 文档、System Card、Research/arXiv 定向检索和研究笔记均已入库。面试主线是 adaptive thinking 与 effort/`max_tokens` 的分层、thinking block/signature 回放、新 tokenizer 对预算的影响、context awareness/compaction、工具宿主责任以及 Artificial Analysis、DeepSWE 和发布方评测的证据隔离；由于没有独立架构或训练报告，不新增 Sonnet 5 专属正式章节。

Qwen3.8 已从“仅榜单发现”升级为“内容专题闭环”：27B、A95B 和 Flash-Next 的参数/层布局/模态/思考协议来自官方模型卡，QSA/GR/N-gram/Muon 来自 Flash-Next 官方技术报告与仓库；Qwen3.8-Max 只作为基于 A95B 的 hosted version 记录。不要把 Max、Flash hosted 或 `reasoning_effort` 行写成新的 open checkpoint。

GLM-5.3-Flash 已从“部分覆盖”升级为“内容专题闭环”：Artificial Analysis 与 DataCurve 只提供锚点和配置级 Agent 结果；Z.ai 文档/博客与固定 revision 模型卡支持 320B/18B、45 层、34 linear/11 sparse、288 routed/top-8/1 shared、IndexPool、mHC、视觉 self-verification、EPD serving 与 thinking/tool streaming 的专题写作。官方披露的 3.01×/4.44× attention/KV、约 3× serving、benchmark 和视觉工作流均保留为发布方自报；生产 kernel、真实 cache/state bytes、硬件 profiling、线上 acceptance rate、完整训练 recipe 和独立复现仍待核验。正式落点为第二十一册第 84 章。

2026-09-15 Claude Opus 5 资料级闭环：`10.237.126.170:1234` 和 `10.24.27.134:8098` 对 Artificial Analysis 中文首页、Opus 5 详情页、DataCurve 和 Anthropic 官方页面均返回 HTTP 200；`10.24.27.134:7890` 对 DataCurve 成功，但 Artificial Analysis 大页面传输超时。Opus 5 的 Artificial Analysis 详情快照为 3,532,163 bytes，SHA-256 `c9626e404c0ff538a28bc58fff9f05bd276db64a1791b72bfa22ce7f4202d47f`；DataCurve 快照为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。

Anthropic 官方资料确认 `claude-opus-5`、2026-07-24、1M context、128K/300K 输出、adaptive thinking、默认 `high`、五档 effort、512-token 缓存门槛、tool/effort 中途变更和 server-side fallback。DeepSWE max 的 327/444、约 73.65% Pass@1、约 `$11.84`/task 与 Artificial Analysis 约 50.70 指数均保留配置、harness、任务集、工具和 verifier 归因；不作为裸模型排名。发布方 Frontier-Bench/CursorBench/ARC-AGI 3/OSWorld 等结论也标为 Anthropic 自报，不能替代独立复现。

arXiv 精确标题检索得到两篇将 Opus 5 作为被测模型的文章，没有 Opus 5 专属技术报告；Anthropic Research 页也没有专属报告条目。System card 已登记哈希但当前环境无法稳定抽取正文，故不扩写未明确的安全数字。当前锚点切换为 `claude-opus-5`，下一锚点为 `claude-fable-5.1`；下一轮沿 Anthropic 官方资料补 Fable 5.1 的榜单配置、运行时协议和长任务证据。

2026-09-14 质量收口：第二十册第 13、17、18、21 章残留的 `max(1, …)` 指标分母已改为严格非空定义域，并补充 `not_applicable` 语义；第 13 章 demo 对空需求/空风险集合返回 `None`，聚合分数和门禁显式拒绝无样本。官方页面复访仍受 DNS 解析失败影响，未将网络失败写成资料不存在。

2026-09-14 榜单代理复访：Artificial Analysis 中文首页与 DataCurve DeepSWE 均可通过已提供的三个代理返回 HTTP 200。Artificial Analysis `/zh` 本次快照为 1,771,203 bytes、SHA-256 `510bb1916a029700fbd78084a4b26cbe1e2b2690ce9e25179fde2d4fe650a7d4`；与此前实时快照规范化比较，已跟踪的模型 release 记录无新增或删除。当前仍有 GPT-6 Astra（榜单 `releaseDate` 2026-09-03）、Claude Fable 5.1（2026-09-01）、GLM-5.3（2026-08-18）和 GLM-5.3-Flash（2026-08-26）。DataCurve 页面快照为 268,313 bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`，与既有 `/tmp/deepswe.html` 字节一致，页面更新时间仍为 2026-09-03；其中有 GPT-6 Astra、GLM-5.3 和 Claude Fable 5，但没有 Claude Fable 5.1。此次没有产生新的候选，后续仍按两榜单唯一入口推进。

## 2026-09-20 GLM-5 正式专题收口

本轮继续沿两个唯一排行榜推进 GLM-5。Artificial Analysis 当前详情页标题为 `GLM-5 (Reasoning)`，并提示已有更新模型 `GLM-5.1`；因此 GLM-5 保留为历史 AA 锚点。DataCurve 当前快照仍没有精确 `mini_swe_agent_glm_5_*` 行，不能迁移 GLM-5.2/5.3/5.3 Flash 的 DeepSWE 结果。

已新增第二十一册第 89 章 [`GLM-5：DSA、MoE、slime 与 Agentic Engineering`](book-21-transformer-architecture-evolution/chapters/89-glm-5-dsa-slime与agentic-engineering.md)，并更新目录、架构时间线、研究笔记、来源索引、模型盘点和榜单解释。章节把 DSA 的 indexer/top-k 召回、744B/40B 的 total/active 分账、MLA 低秩字段、`slime` 的 rollout/trainer 异步流水线、policy lag/freshness、长轨迹 verifier 和 Agentic Engineering 的 artifact gate 连接为面试主线。

本轮快照：AA `3,811,809` bytes / SHA-256 `0b9c56ff97a87b1dd0a006d1c300057f5ef20c3f65a19bcddf0f6803d8a94f8d`；DataCurve `268,571` bytes / SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；Z.ai 博客资源 `145,189` bytes / SHA-256 `99d27d6132c25e1b39fe26df0605ad1abd855e9b30b098996c64423093fb618f`。Z.ai 文档 Markdown 端点本轮由代理返回 503，只作为线路失败记录；已有官方快照继续有效。

GLM-5 当前升级为“内容专题闭环（AA 单榜）”：研究笔记、正式章节和配套同步已经具备，但 DataCurve 精确评测、完整 DSA indexer loss/recall、生产 kernel、`slime` 调度与 freshness 控制、完整训练 recipe、硬件 profiling、线上 tool acceptance 和独立复现仍待核验。下一轮仍只从八家重点厂商在两个排行榜中的剩余条目选择锚点。

GLM-5 之后的下一锚点确定为 `deepseek-v3-2`：2026-09-20 两代理取得逐字节一致的 Artificial Analysis 详情页，仍为 `DeepSeek V3.2 (Non-reasoning)`、128K、约 648B/37B，SHA-256 `7d4fd97ed7f2efbc9ab2f001072b353f00e2476cb884abe0d8185f16d7562052`；DataCurve 快照仍无精确 `mini_swe_agent_deepseek_v3_2_*` 行。V3.2 已补齐官方模型卡、技术报告、V3.2-Exp inference、TileLang、DeepGEMM/FlashMLA 和 vLLM serving 证据，下一步只补最终 artifact 与实验 config 的 commit/权重对应、FP8/BF16 端到端 profiling、线上 tool acceptance 和独立 benchmark，不新增重复架构章节。

## 2026-09-20 DeepSeek V3.2 实现证据补充

本轮执行重点从“论文主线已读”推进到“公开实现路径可追溯”：把最终 V3.2 模型字段、V3.2-Exp inference、研究可读 kernel、高性能 CUDA kernel 和 vLLM recipe 分栏。后续涉及 `q_lora_rank`、KV cache dtype、top-k、并行策略或 GSM8K 时，必须先标注所属 artifact、commit、硬件和 harness。

验收标准新增：面试答案能说明 non-interleaved indexer RoPE 与 MLA 布局差异、FP8 index score、prefill MHA/decode MQA、radix top-k、latent/positional cache、EP/DP/TP 约束和 recipe 结果的证据等级；研究报告必须把固定官方 `config.json` 与 V3.2-Exp inference 的 `q_lora_rank=1536` 记为同值字段，同时按 revision、实现路径和 serving recipe 分账，不能把它们合并成完整生产配置。

## 2026-09-20 Gemini 3.5 Flash-Lite AA 单榜资料级闭环

本轮从 Artificial Analysis 的精确 `gemini-3-5-flash-lite` 条目继续选择 Gemini 重点厂商锚点；DataCurve 当前没有精确 `mini_swe_agent_gemini_3_5_flash_lite_*` 行，因此不迁移 Gemini 3.5/3.6 Flash 或其他 Gemini 版本的 DeepSWE 结果。AA 快照为 `3,845,070` bytes、SHA-256 `7de22f529d3b4e2dd440ef8a8e9447480e18dde08437fc508ece06767571e8ef`，第三方字段为 `releaseDate: 2026-07-21`、`isReasoning=true` 和 Intelligence Index `22.1685424839812`。

已核验 Google API 模型页的 `gemini-3.5-flash-lite`、text/image/video/audio/PDF 输入、text 输出、1,048,576 输入 token、65,536 输出 token、caching、code execution、Computer Use Preview、File Search、function calling、Maps/Search grounding、structured output、thinking 和 URL context。Thinking 文档明确默认 `On (minimal)`，支持 `minimal/low/medium/high`；这是请求级 reasoning budget，不是四个模型权重。

已核验 Video understanding 文档把 Lite 列入 agentic video：static 路径固定约 1 FPS，agentic 路径按 prompt 动态浏览时间轴并按需加载 transcript、帧或音频，`processing_call`/`processing_result` 暴露处理状态。发布方声称长内容最多约 88% token efficiency 和约 7% quality 提升，必须保留为文档自报，不能当作每个任务的固定收益。

DeepMind Model Card/PDF 明确 3.5 Flash-Lite based on Gemini 3.1 Flash-Lite，并把 architecture、training dataset、data processing、hardware、software 指向 3.1 Model Card。SWE-Bench Pro `54.2%`、Terminal-Bench 2.1 `54.0%`、OSWorld-Verified `74.0%`、GDM-MRCR v2 128K/1M `72.2%`/`21.3%` 与 `$0.30/$2.50` 价格均是 Google 发布方设置，不能与榜单或其他 harness 拼接。正确的 `whats-new-gemini-3.5` 页面属于 3.5 Flash，不能把其默认 `medium`、GA 或 thought preservation 迁移给 Lite。

当前状态：资料级闭环（AA 单榜）。不新增独立架构章节，映射到已有 Reasoning、Agent/工具协议、多模态视频、长上下文/Serving、评测与安全章节。仍待核验 3.5 Lite 独有参数、层/专家/注意力结构、完整训练/后训练 recipe、独立技术报告、生产 kernel、硬件 profiling、线上 agent/tool acceptance rate、DataCurve 精确行和独立复现。下一轮仍只从两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20 Kimi K2.6 锚点推进

本轮继续只从 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜选择重点厂商锚点，确定 `Kimi K2.6`。Artificial Analysis 有精确 `kimi-k2-6`/Non-reasoning 条目；DataCurve 当前没有精确 `mini_swe_agent_kimi_k2_6_*` 行，因此不迁移 Kimi K2.7 Code 或 Kimi K3 的 DeepSWE 结果。

已完成：

- 新增 [`kimi-k2.6-source-notes.md`](research/model-update-2026-09/kimi-k2.6-source-notes.md)，固定 AA、DataCurve、Hugging Face README/config、部署指南、Kimi K2.6 博客、Thinking API 和 Vendor Verifier 的证据边界与快照哈希。
- 核验官方模型卡/配置的 1T total、32B active、61 层、384 routed experts、top-8、1 shared expert、MLA、`q_lora_rank=1536`、`kv_lora_rank=512`、YaRN 256K、MoonViT 约 400M 和 native INT4。顶层 `KimiK25ForConditionalGeneration`/`kimi_k25` 与文本子配置 `DeepseekV3ForCausalLM`/`kimi_k2` 只按实现兼容标识记录。
- 核验 K2.5 架构路线复用、vLLM/SGLang/KTransformers 部署、thinking/`preserve_thinking`/`reasoning_content`/interleaved tool call、Agent Swarm 和 Kimi Vendor Verifier。300 sub-agents 属于 Agent harness，不是 MoE expert；KVV 属于部署验收，不是模型能力 benchmark。
- 新增第二十一册第 90 章 [`Kimi K2.6：Native Multimodal、Agent Swarm 与推理验收`](book-21-transformer-architecture-evolution/chapters/90-kimi-k2.6-native-multimodal-agent-swarm与推理验收.md)，并同步该册目录、架构创新时间线、来源索引、模型清单和榜单解释。

当前状态：`Kimi K2.6` 为 **AA 单榜资料级闭环**。暂无 K2.6 专属完整技术报告或独立架构论文；完整层排布、训练/后训练 recipe、production kernel、INT4 误差与硬件 profiling、MoonViT 训练细节、Agent Swarm coordinator、线上 tool acceptance、精确 DataCurve 行和独立复现仍待核验。

后续顺序：先完成 K2.6 相关面试题、练习、术语、项目和知识图谱的最小同步；然后仍只从两个排行榜的八家重点厂商条目选择下一锚点，不从官方目录、论文检索或 Hugging Face Trending 另发现模型。继续区分基础模型、revision、effort、provider、fallback、harness 与 verifier。

## 2026-09-20 GPT-5.4 mini/nano 资料级闭环

本轮沿两个唯一排行榜继续核验 GPT-5.4 的两个低成本 sibling，而不是从 OpenAI 模型目录另发现模型。Artificial Analysis 有精确 `gpt-5-4-mini` 与 `gpt-5-4-nano` 条目；DataCurve 只有 `mini_swe_agent_gpt_5_4_xhigh` 的 GPT-5.4 base 系统结果，没有 mini/nano 精确模型 ID，因此不迁移 base 的 Pass@1、成本、输出 token 或 Agent steps。

已完成 OpenAI 官方模型页、GPT-5.4 指南和 Reasoning/Tools 资料的核验。mini/nano 均固定到 `2026-03-17` snapshot、400K context、272K maximum input、128K maximum output 和 `none/low/medium/high/xhigh` effort；mini 的定位是高吞吐 coding、computer use 和 Agent workflow，nano 的定位是 classification、extraction、ranking 和窄任务 sub-agent。官方价格也按精确 sibling 记录：mini `$0.75/$4.50`，nano `$0.20/$1.25`（input/output，单位为每百万 token；cached input 另计）。

本轮最重要的面试主线是：按 task shape 路由而不是按价格路由；为小模型补齐目标、依赖、工具顺序、失败恢复、schema、停止条件和 abstain 规则；`reasoning_effort` 不是硬 token 预算；mini/nano 必须按精确 model ID、snapshot、endpoint 做 capability probe。mini 页面列出 `tool_search`/`computer_use`，nano 当前页面未列出它们，不能按家族名自动继承。详细证据见 [`gpt-5.4-mini-nano-source-notes.md`](research/model-update-2026-09/gpt-5.4-mini-nano-source-notes.md)。

配套只复用第二十四册 Agent serving、十六册 reasoning、十七册工具协议和七册公平评测主线，不新增重复 Transformer 章节。当前状态为“资料级闭环（AA 单榜关联档位）”；参数、架构、训练/后训练 recipe、system card、独立技术报告、mini/nano 精确 DataCurve 结果、production kernel、硬件 profiling 和线上 acceptance rate 仍待核验。下一轮重新从 Artificial Analysis 与 DataCurve 的八家重点厂商条目选择锚点。

## 2026-09-20 GPT-5.5 Instant (June 2026) 关联配置核验

本轮从 Artificial Analysis 的精确 `gpt-5-5-instant-06-26` 条目选择下一锚点，并同时保留 May revision 的差异；不把榜单名称自动当作 OpenAI API model ID。June 详情为 2026-06-25、AA Index `26.0135173401368`、约 130.9925 output tokens/s、400K context；May 为 2026-05-05、AA Index `22.6863693000789` 且已 deprecated。两个页面的快照哈希与第三方字段见 [`gpt-5.5-source-notes.md`](research/model-update-2026-09/gpt-5.5-source-notes.md)。

官方核验只确认 `gpt-5.5`/`gpt-5.5-2026-04-23`；精确 `gpt-5.5-instant` 模型页通过 `7890` 返回 HTTP 404。DataCurve 当前没有 Instant 精确 `mini_swe_agent` 行，因此不能迁移 GPT-5.5 base 的 Agent 结果。本轮已完成“榜单级关联配置 + 官方身份负证据”核验，不把它合并为 `gpt-5.5`，也不再让它阻塞下一锚点。

当前状态：**榜单级关联配置 + 官方身份负证据**。面试上只讨论 AA 字段、revision/served-model 归因、API 身份映射和不迁移评测的证据纪律；GPT-5.5 base 的 context、tool catalog、effort、训练和 Agent 结果只作为对照，不写成 Instant June 的独有事实。

## 2026-09-20 DeepSeek V4 Pro 0813 当前活动锚点

本轮从 Artificial Analysis 的精确 [`deepseek-v4-pro`](https://artificialanalysis.ai/models/deepseek-v4-pro) 条目选择 `DeepSeek V4 Pro 0813 (Reasoning, Max Effort)`，并在 DataCurve 找到精确的 `mini_swe_agent_deepseek_v4_pro_max` 行。AA 详情快照为 3,934,926 bytes、SHA-256 `b14ce625eb337401f4faa63eed37f1b08fe109a693dcdf2a34e21924b6cddff1`；DataCurve 当前为 Pass@1 `62.831858%`、Pass@4 `88.495575%`、`n_runs=4`、平均成本 `$1.6660232187`、平均输出 `105998.9` token、平均 `154.71` steps。

已核验的官方链路包括 [V4 Pro GA 公告](https://api-docs.deepseek.com/news/news260813)、[Quick Start](https://api-docs.deepseek.com/quick_start)、[Responses API](https://api-docs.deepseek.com/guides/responses_api)、[Thinking](https://api-docs.deepseek.com/guides/thinking)、[Tool Calls](https://api-docs.deepseek.com/guides/tool_calls)、[DeepSeek-V4-Pro 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)、[配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json)和 [V4 技术报告](https://arxiv.org/abs/2606.19348)。面试主线为：`low/high/max` 是请求级 effort；Responses 的 stateless 与 function tools/`apply_patch`/并行工具边界；1.6T/49B、1M、CSA/HCA、mHC、Muon、FP4/FP8；SFT+GRPO 培养领域教师与 on-policy distillation 合并；以及 61 层、384 routed/6 selected/1 shared、`q_lora_rank=1536`、`o_lora_rank=1024`、`index_topk=1024` 的配置级证据。

本锚点状态为**内容专题闭环（双榜锚点）+ reference implementation 证据补强**。不新增重复 Transformer 正式章节，复用第二十一册第 77/78 章、DeepSeek V4/V4.1 章节和 Agent serving 主线；本轮固定 HF revision `b5968e9190ef611bbf34a7229255be88a0e937c1`，记录 encoding/inference 文件哈希、压缩/索引/路由/低精度实现字段，并补充第二十四册 serving manifest。后续只补独立 benchmark、目标硬件 profiling、线上 tool acceptance、完整训练 recipe 与 API 快照差异；reference code、完整权重 metadata 和 `MP=8` 示例均不等于生产验收。所有 DataCurve 数字继续绑定 `mini-swe-agent`、工具、环境和 verifier，不迁移给其他 DeepSeek 版本。

## 2026-09-20 GLM-5.1 资料级闭环

本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。按照“模型只从 Artificial Analysis 与 DataCurve DeepSWE 发现”的规则，选择 Artificial Analysis 的 `GLM-5.1`；DataCurve 当前没有精确 `mini_swe_agent_glm_5_1_*` 行。

已完成：

- 三条代理重新获取 Artificial Analysis GLM-5.1 详情页，快照逐字节一致，`3,935,095` bytes，SHA-256 `f6b1ca673b777602013f13eb348a7c48773120e13684eadcbd7220b50b6c137d`；记录 `GLM-5.1 (Reasoning)`、2026-04 release 字段、200K、约 744B/40B、第三方指数 `26.0585912980095` 和约 39.9 tokens/s。
- 通过 7890 获取 Z.ai GLM-5.1 官方 Markdown 文档、Hugging Face README/config；补齐 200K/128K、`glm-5.1` API ID、MIT、78 层、256 routed/top-8/1 shared、前三层 dense、`q_lora_rank=2048`、`kv_lora_rank=512`、`index_topk=2048`、202752 position 等配置字段。
- 获取 Z.ai 的 Deep Thinking、Function Calling、Context Caching、Pricing 和 Release Notes；提取 long-horizon 约 8 小时 Agent、实验—分析—优化闭环、multi-turn SFT/RL/process-quality evaluation framework、thinking/tool/cache 协议和价格字段。
- 补记证据边界：2026-09-20 的一次访问曾使 `https://z.ai/blog/glm-5.1` 返回 HTTP 404；2026-09-21 已通过三条代理取得 HTTP 200 的官方博客壳和正文 JS 资源，因此旧 404 只能作为当时的线路/页面状态，不能继续写成当前页面不可得。arXiv `all:"GLM-5.1"` 精确检索没有 GLM-5.1 专属技术报告；模型卡链接的是 GLM-5 报告。1234 本轮 HF CONNECT 超时只记录为线路失败，不能解释为页面不存在。
- 新增 [`glm-5.1-source-notes.md`](research/model-update-2026-09/glm-5.1-source-notes.md)，同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md) 和 [`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)。

证据边界：Z.ai 的 SWE-Bench Pro `58.4`、Linux desktop 655 次迭代/6.9× 向量数据库吞吐和 KernelBench Level 3 `3.6×` 对比 `torch.compile` max-autotune `1.49×` 都是发布方自报；不能与 AA 指数或其他 GLM 版本的 DataCurve 结果拼成裸模型能力。`thinking.type=enabled/disabled` 是 GLM-5.1 请求级模式；当前文档把 `reasoning_effort` 支持列为 GLM-5.2 及以上，不能迁移到 GLM-5.1。配置字段不等于完整参数账本、DSA indexer loss/recall 或生产 kernel。

当前状态：**AA 单榜资料级闭环**。不新增重复 Transformer 正式章节，复用第二十一册 GLM-5 DSA/MoE 与 Agentic Engineering、十六册 reasoning、十七册工具协议和第二十四册 serving 主线。待核验：独立参数/训练 recipe、过程质量 verifier、完整长周期 harness、生产 kernel、硬件 profiling、线上 tool acceptance、精确 DataCurve 行和独立复现。下一轮仍只从两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20 Grok 4.20 全局资料同步完成

Grok 4.20 已完成本轮“榜单锚点 → 官方 runtime 资料 → 面试知识 → 书系/全局资料”闭环。除研究笔记、来源索引、模型清单、榜单解释、书系章节外，已同步 [`INTERVIEW_BANK.md`](INTERVIEW_BANK.md)、[`EXERCISES.md`](EXERCISES.md)、[`PAPERS.md`](PAPERS.md)、[`KNOWLEDGE_GRAPH.md`](KNOWLEDGE_GRAPH.md)、[`GLOSSARY_EN_ZH.md`](GLOSSARY_EN_ZH.md) 和 [`PROJECTS.md`](PROJECTS.md)。

本轮全局资料固定以下面试主线：`grok-4.20-multi-agent` 的 4/16 agent-count 与普通 reasoning depth/MoE expert 数量的区分；leader/sub-agent 责任链和 opaque encrypted state；Artificial Analysis 2M 与 xAI 官方 1M context 的来源分账；compaction item 的完整有序 replay；prompt caching 与 reasoning state 的区别；server-side/client-side tools、跨请求 `max_turns` 和 Remote MCP `allowed_tools` 的最小权限；以及 DataCurve 没有精确 `mini_swe_agent_grok_4_20_*` 行时拒绝迁移 Grok 4.5/4.6 结果。

后续仍只从 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜选择八家重点厂商锚点；对 Grok 4.20 只补独立架构/训练报告、生产 kernel、目标硬件 profiling、线上 acceptance 和精确 Agent 评测等新证据，不重复建立同一 runtime 章节，也不从 xAI 官方目录或 arXiv 外部使用论文另发现模型。

## 2026-09-20 Gemini 3.8 Flash 当前活动锚点

Gemini 3.8 Flash 已完成本轮“榜单复验 → Google 官方 runtime 资料 → 面试知识 → 全局资料同步”。AA high/medium/low 三个配置归并为一个基础模型；DataCurve 只使用精确的 high 行，不把相邻 Gemini 或其他 effort 的结果迁移过来。当前为**资料级闭环**，不新增独立 Transformer 正式章节。

本轮计划聚焦以下面试主线：

- `low/medium/high` thinking 与共同 output budget 的消融，区分请求级预算和模型权重；
- thought summary/signature 与 Interactions stateful/stateless replay，区分 opaque 状态、可见摘要、应用会话和 KV cache；
- SSE step/event trace，以及 Search、URL Context、File Search、Code Execution、function calling 的上下文循环；
- Computer Use 的 screenshot/action intent/坐标/审批/宿主执行/副作用审计；
- 1M 输入、implicit caching、TTFT/TPOT、计费 token 和长上下文有效召回的分账；
- DataCurve `mini_swe_agent_gemini_3_8_flash_high` 的模型+harness+工具+环境+verifier 归因。

书系落点：第十六册 `14.28` 讲 thinking budget/signature，第十七册 `7.30` 讲 Interactions 与 Computer Use，第二十册 `19.31` 讲 harness-aware evaluation，第二十四册 `32.41` 讲 state/cache/tool serving；维持“不新增独立 Gemini 3.8 架构章”的证据边界。

下一步只补真实 API capability probe、Interactions replay、thinking 消融和工具组合实验；若没有新的独立架构/训练资料，不重复创建 Gemini 3.8 专属章节，也不把产品描述写成内部 RL、搜索树或 verifier 机制。

## 2026-09-20 GLM-5.3 当前活动锚点与资料级闭环

本轮从两个唯一排行榜中的精确 `GLM-5.3` 条目继续推进：Artificial Analysis 的 `glm-5-3` max 与 DataCurve 的 `mini_swe_agent_glm_5_3_max`。AA 详情快照为 `3,928,329` bytes、SHA-256 `090279e870a3ebb72c7a69f24963f87fe10d63595642a4c61959b911f2a3c3a2`；DataCurve 快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。DataCurve 精确行是 451 次尝试、311 次通过、Pass@1 `68.9579%`、Pass@4 `87.6106%`、平均成本约 `$3.9934`、平均输出约 `80.4K` token、平均 124.47 steps；这些数字绑定 `mini-swe-agent`、工具、环境、超时和 verifier，不能写成裸模型能力。

已核验的权威资料包括 [GLM-5.3 文档](https://docs.z.ai/guides/llm/glm-5.3)、[迁移指南](https://docs.z.ai/guides/overview/migrate-to-glm-new)、Thinking/Function Calling/Tool Streaming/Context Caching/Structured Output 文档、[Z.ai 发布资料](https://docs.z.ai/release-notes/new-released.md)、[模型卡与 config](https://huggingface.co/zai-org/GLM-5.3)、[slime](https://github.com/THUDM/slime)、[IndexCache](https://arxiv.org/abs/2603.12201) 和 [SAO](https://arxiv.org/abs/2607.07508)。

本轮面试主线是：GLM-5.3 沿用 GLM-5.2 基座且收益归因于后训练；环境由可执行长周期任务、judge agent、无 reference verifier 和 reward-shortcut audit 组成；oracle、no-op、unsolved-state 是训练前的质量门禁；`slime` 将 Megatron 训练、SGLang rollout 和 data buffer 放进同一 dataflow，并关注 train/rollout logprob 一致性；`thinking.type` 强制 `enabled`、`reasoning_effort` 为 `low/high/max`，`tool_stream` 需要按 delta 重组工具参数。

证据边界保持严格：SAO 论文直接绑定 GLM-5.2，不能改写成 GLM-5.3 独有算法；IndexCache 是 DSA serving 关联论文，不是 5.3 的架构变更声明；Z.ai 的 benchmark、`1e-7` logprob 差异和 `>2.3x` throughput 是发布方系统描述/测量；HF config 是实现字段，不等于完整参数或训练账本。GLM-5.3 当前为**双榜资料级闭环**，正式落点为第十六册第 20 章；完整 recipe、5.3 专属 SAO/compaction 定义、production kernel、硬件 profiling、独立 benchmark 和线上 acceptance 仍待核验。下一轮仍只从两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20 DeepSeek V3.2 当前活动锚点与实现证据补强

本轮选择 Artificial Analysis 的精确 `deepseek-v3-2` 条目；DataCurve 当前没有 `mini_swe_agent_deepseek_v3_2_*` 精确行，因此不迁移 V3.1/V4 的 Agent 结果。AA 详情快照为 `3,625,720` bytes、SHA-256 `7d4fd97ed7f2efbc9ab2f001072b353f00e2476cb884abe0d8185f16d7562052`；`Non-reasoning`、128K、约 648B/37B、指数 `16.043537719683` 是第三方目录字段。

已核验的权威链路包括 [V3.2 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)、[技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf)、[V3.2-Exp inference](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference)、[TileLang](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32)、[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/pull/200)、[FlashMLA](https://github.com/deepseek-ai/FlashMLA/pull/98) 和 [vLLM recipe](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html)。

面试落点已补到第二十一册第 19 章：DSA 必须分开讨论 FP8 indexer、causal/top-k candidate、sparse MLA gather、index/final evidence recall 和 dense fallback；固定官方 `config.json` 与 V3.2-Exp demo 的 `q_lora_rank` 都是 `1536`，但不同 revision/artifact 的字段不能合并为完整生产配置；prefill MHA、decode MQA、latent/positional cache、FP8 KV cache、DP/EP/TP recipe 也不能互相替代。V3.2 的 `thinking with tools` 与 V3.2-Speciale 的“不支持 tool calling”必须分开记录。

当前状态仍为**AA 单榜资料级闭环**：正式章节和全局配套已覆盖，但完整层排布、最终 production kernel、召回曲线、训练/RL 完整 recipe、线上 tool acceptance、硬件 profiling、PDF 图表/公式视觉复核和独立 benchmark 仍待核验。下一轮仍只从两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20 Kimi K2.6 harness 补证与 K2.8 候选边界

本轮是对既有 `Kimi K2.6` 的补证，不是新增模型锚点，也不改变当前活动锚点 `DeepSeek V3.2`。沿 K2.6 官方资料继续核验了当前 Kimi Code 的模型配置、Agent/subagent、AgentSwarm、工具和会话文档，并将其作为通用 coding-agent harness 资料写入 K2.6 研究笔记与第二十一册第 90 章。

- Kimi Code 当前文档列出 K3、K2.8 Preview、K2.7 Code HighSpeed 三类模型、4 个 model ID；`kimi-for-coding` 对应 K2.8 Preview。2026-09-20 AA 首页/详情和 DataCurve 快照没有精确 K2.8 条目，因此 K2.8 只记录为官方周边文档中的关联版本，不进入候选盘点、不建立独立研究笔记，也不迁移其 effort/context 字段到 K2.6。
- 当前 `AgentSwarm` 文档公开最多 128 个 subagent、默认 2 小时超时、`resume_agent_ids`、模型池、聚合报告、唯一工具调用约束、从 5 个任务开始且每 700ms 增加一个的并发爬坡，以及 `KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY` 限制。工具和 subagent 列表在派发前再次校验，权限规则与模型可见工具列表分层；这些都是当前 CLI harness 事实。
- Kimi Code 会话文档公开 `state.json`、`wire.jsonl` 事件流、`--continue`、`--session`、`/compact`、`/fork` 和导出机制。它补充了长周期 Agent 的 session replay、工具 schema/MCP 清单和上下文管理面试点，但没有公开 K2.6 专属 coordinator 内部实现。
- K2.6 博客中的 300 sub-agents/4,000 coordinated steps 与当前 CLI 的 128 上限按来源、产品形态和时间分账，不合并成单一的“K2.6 Agent 上限”。

当前状态保持：`Kimi K2.6` 为 **AA 单榜资料级闭环**；K2.6 专属训练 recipe、production kernel、INT4 profiling、Agent Swarm 内部 coordinator、线上 acceptance、精确 DataCurve 行和独立复现仍待核验。下一轮仍只从 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20 DeepSeek V3.2 官方配置与 encoding 证据纠正

- 复查官方 Hugging Face 固定 revision `a7e62ac04ecb2c0a54d736dc46601c5606cf10a6` 后，确认 `config.json`（1,552 bytes，SHA-256 `c7fa8b191e9936d8e6a57d864baab82b792fae16a116416cdd3a75ba76bc5af1`）中的最终 `q_lora_rank` 是 `1536`；此前把最终字段读成另一数值的记录已在研究笔记、章节、索引、题库和全局资料中纠正。
- 最终配置与 V3.2-Exp inference demo 的 `q_lora_rank=1536` 是同值字段，但仍须按 revision、权重/代码 artifact、kernel 和 serving recipe 分账，不能因字段相同就合并为一份生产配置。
- 新增固定 `encoding/encoding_dsv32.py` 证据（14,317 bytes，SHA-256 `5e068c2ba2a6e5ebe37a49bb005650c507e7935d77a32f3f7c11ee071498b370`）：DSML function-call/result、role、think/reasoning 和字符串/JSON 参数分支已写入第 21 册第 19 章，面试答案明确区分 parser、schema、权限、执行器和 verifier。
- 8098/1234 获取 HF 的 CONNECT timeout 仍只记为线路失败；已获得的 7890 固定 revision 证据足以完成本次纠正。PDF 图表/公式视觉复核、完整 production kernel、召回曲线、硬件 profiling、线上 tool acceptance 和完整训练/RL recipe 仍待核验。

## 2026-09-20 两榜当前时点复验

- 通过三条代理重新抓取 Artificial Analysis `/zh` 与 DataCurve DeepSWE：AA 均 HTTP 200、1,777,695 bytes、SHA-256 `82b890a93772e857bfafd0aa0a96994d39afb4fcdd2a9fdf2d4e6ec01f9c7912`；DataCurve 均 HTTP 200、268,571 bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，三代理各自结果逐字节一致。
- 当前 AA 首页前列的八家重点厂商条目与现有研究清单一致，DataCurve 精确 `mini_swe_agent` 集合也没有需要新增研究笔记的重点模型；不从首页之外的官方目录、论文或模型仓库另发现候选。
- 因此当时不新增模型；完成 K3 实现证据同步后，历史记录曾将研究入口记为 `Kimi K3`，随后进入 `DeepSeek V4 Pro 0813` 阶段。该段只记录历史顺序；当前活动锚点以文件顶部和本文件最新记录为准，已切换为 `DeepSeek V4.1-Flash`。

## 2026-09-20 Kimi K3 实现与 serving 补证

本轮继续只使用 Artificial Analysis 与 DataCurve DeepSWE 的 Kimi K3 条目作为锚点，不把 HF、GitHub 或 vLLM 页面中的关联版本另发现为模型。已将官方 HF 权重 metadata/config、FlashKDA 和 vLLM recipe 纳入证据链。

- HF API 固定 revision 为 `f831ab66814297da540d832a5235f8e904f29d06`；`config.json` 确认 `KimiK3ForConditionalGeneration`、93 layers、69 KDA + 24 full-attention/Gated MLA、`q_lora_rank=1536`、`kv_lora_rank=512`、896 experts/top-16/2 shared 和 1,048,576 context。完整 safetensors 没有下载，metadata 不等于本地权重加载成功。
- FlashKDA master commit 为 `7afb9f454f160a6c4bbc0999beca0a8c40a38934`；`8x8 fp32 forward substitution + 16x16 bf16 merge`、SM90+/CUDA 12.9+/PyTorch 2.4+、`chunk_kda` recurrent-state backend、变长 batch 和 H20/GB200 benchmark 已进入实现证据。benchmark 的 `1.85x`/`2.31x` 是仓库发布方相对 baseline，不能写成本地或端到端复现。
- vLLM K3 recipe 更新于 2026-09-10，最低声明版本为 0.29.0，但优化路径依赖 K3-enabled nightly/image、CUDA 13/cu130 和 r580+ driver。PyPI vLLM 0.29.0 stable release/source 已有 K3 registry/package/model implementation；hybrid KV manager 同时管理 MLA attention cache 与 KDA recurrent state；Blackwell 的 FP8 KV、TOKENSPEED MLA、prefix caching 和 `--prefix-match-unit 128` 要和硬件/并行拓扑一起记录。
- recipe 警告偶发 tool-call parser 格式不兼容；教学和生产 harness 必须加入 schema validation、retry/idempotency、权限检查和 verifier。目标硬件 profiling、独立 benchmark、完整权重加载、hybrid cache recovery 和线上 acceptance 仍待核验，不能把 stable source entry 写成生产验收。

该段是 Kimi K3 阶段的历史记录；随后完成了 `DeepSeek V4 Pro 0813` 的 reference implementation 补证，当前研究入口以本文件最新的 V4.1-Flash 记录为准。

## 2026-09-20 DeepSeek V4.1-Flash reference implementation 补证

本轮继续只从 Artificial Analysis 与 DataCurve DeepSWE 的重点条目推进，选择已在 Artificial Analysis 出现的 `deepseek-v4-1-flash`。两榜当前时点快照为 AA 首页 `1,777,695` bytes / SHA-256 `82b890a93772e857bfafd0aa0a96994d39afb4fcdd2a9fdf2d4e6ec01f9c7912`，DataCurve `268,571` bytes / SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；DataCurve 没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，不迁移相邻 DeepSeek 版本的 Agent 结果。

通过 `10.24.27.134:7890` 固定到 HF revision `dba1be0a40aa45a94ad051997016db3960a90277`，补齐 API inventory、`inference/config.json`、`model.py`、TileLang `kernel.py`、`convert.py`、Engram/vision、encoding 和 `dsh-minimal` patch 的源码哈希。实现账本确认 SWA ring、压缩 KV、两级 candidate/index top-k、sparse attention、MoE、Engram、mHC/Sinkhorn 和 DSpark forward path；README 明确这是 readable reference implementation，`generate.py` 仍为 plain autoregressive generation。

已做的验证只有 Python 静态编译和 encoding smoke test；没有下载完整权重，也没有 CUDA/TileLang、目标硬件 profiling、生产 DSpark draft/verify/rollback、线上 acceptance 或独立 benchmark。当前状态为**内容专题闭环 + reference implementation 证据补强（AA 单榜）**。下一轮仍先在两榜剩余八家重点厂商条目中选择锚点，并保持模型、revision、effort、provider、harness 和 verifier 分账。

## 2026-09-21 DeepSeek V4.1-Flash：deepseek-recipe 协议实现补证

本轮继续沿 Artificial Analysis 的 `deepseek-v4-1-flash` 锚点推进；DataCurve 没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，因此不迁移其他 DeepSeek 版本的 Agent 结果，也不把 GitHub 仓库中的关联 artifact 当作新模型。

已完成：

- 固定 DeepSeek 官方 [`deepseek-recipe`](https://github.com/deepseek-ai/deepseek-recipe/tree/8cadfede7063c896b944e7bae05daa3549ae97ea) 到 commit `8cadfede7063c896b944e7bae05daa3549ae97ea`；源码归档 3,996,119 bytes，SHA-256 `1116ca33e9dc62a913fb9214578c400f1704e6bca33487f4a4c31b32c67a21a6`。GitHub API 匿名 rate limit 返回 403，但官方 Atom、pinned raw README 和 pinned archive 均成功，commit 证据已固定。
- 核对 README、streaming/tokenizer 文档与关键 Rust 源码：`ConversationRequest` 的协议规范化、V4.1 encoding 的 effort/DSML/mid-system 规则、跨 chunk 的 reasoning/DSML/JSON/stop state machine、显式 tokenizer bridge、图像 URL/data URL/bytes quota 和 mock server 接线。
- 把面试主线写入 V4.1 研究笔记与第二十一册第 81 章 `81.10.4`：协议适配器、prompt/tokenizer、流式 parser、图像安全、inference backend、tool executor、verifier 和 transport 必须分账；parser 成功不等于工具已执行或业务结果已验证。
- 已同步 [`deepseek-v4.1-flash-source-notes.md`](research/model-update-2026-09/deepseek-v4.1-flash-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、第二十一册第 81 章、第二十四册第 32.44 节，以及 `INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`PAPERS.md`、`KNOWLEDGE_GRAPH.md`。

证据边界：`deepseek-recipe` 明确不负责模型 inference、HTTP transport、工具执行、权限或 verifier；README 列出的 `logprobs`、server-side `web_search`、JSON Schema/strict、`n>1`、Responses storage 和 encrypted thinking 未支持项属于协议库边界，不能改写成 V4.1 模型能力负面结论。默认图像限制为 600/32 MiB/64 MiB/8 concurrent，fetcher 不做 SSRF/private-address filtering；`server-py`/`server-rs` 是 mock inference 示例。

下一步仍保持：完整权重加载、production kernel 覆盖、真实 candidate/index recall、FP4 误差、DSpark draft/verify/rollback 调度、EPD 实际调度、目标硬件 profiling、线上 tool acceptance 和独立 benchmark；完成后再从两个排行榜选择下一未闭环重点锚点。

## 2026-09-21 两榜复验与 GLM-5.1 下一锚点

三条代理重新获取两张排行榜。Artificial Analysis `/zh` 当前快照为 `1,777,588` bytes、SHA-256 `10630c5152df60351ceff5819c90dfd104f1e9e502959eb36e470abae9ab2c9`；DataCurve 三份快照均为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，且逐字节一致。AA 的页面字节变化没有带来重点 canonical slug 新增/删除，DataCurve 的重点 `mini_swe_agent_*` 配置集合也没有变化。

本轮因此记录为“榜单复验、无新增模型”。`mini_swe_agent_kimi_k2_7_code_default` 已属于此前完成的 Kimi K2.7 Code 锚点；不因页面中再次出现而重复建模。下一活动锚点定为既有候选 `GLM-5.1`：已有研究笔记和 AA 单榜资料级闭环，继续补联网可用时的 Z.ai 官方博客/发布说明、模型卡/配置、API/代码/论文负面证据与长周期 Agent 评测边界；DataCurve 没有精确 GLM-5.1 行，不能迁移 GLM-5/5.2/5.3 的 Agent 结果。

## 2026-09-21 GLM-5.1 官方博客恢复与长周期评测补证

三条代理重新访问 `https://z.ai/blog/glm-5.1` 均返回 200 的官方前端壳；继续固定其 `glm-5.1-UPT9aJ4D.js` 正文资源，三条代理逐字节一致，221,954 bytes、SHA-256 `0e2a4ae9177f44509ee54e9105127d17ff65f6d3b5f95294a7b3b3bdb125c08b`。因此此前“博客 404”的记录只代表 9 月 20 日那次线路/页面状态，已由新鲜官方资源修正。

本轮新增的面试主线是三种反馈条件：VectorDBBench 用 Recall ≥ 95% 与 QPS 驱动 600+ iterations/6,000+ tool calls 的外层 edit/compile/test/profile/submit loop；KernelBench Level 3 用 50 problems、独立 Docker/H100、1,200-turn cap、数值正确性和 Claude Opus 4.6/GPT-5.4 反作弊审计形成 verifier 门禁；Linux desktop 则展示没有单一标量目标时的 8 小时 self-review harness。所有数字仍是 Z.ai 发布方实验设置，不升级为独立复现、训练 recipe 或模型内部推理深度。

已将上述证据同步到 `glm-5.1-source-notes.md`、来源索引、模型盘点、榜单解释、`PAPERS.md`、`PROJECTS.md`、`INTERVIEW_BANK.md`、`GLOSSARY_EN_ZH.md` 和 `KNOWLEDGE_GRAPH.md`；不新增重复 Transformer 章节。当前仍待核验生产 kernel/硬件 profiling、完整可复现 harness、过程质量 verifier 定义、精确 DataCurve 行和独立第三方复现。

## 2026-09-21 Qwen3.8 Max (0902) runtime recheck 与下一锚点

本轮重新核验两张排行榜和 Qwen3.8 Max 详情页。Artificial Analysis `/zh` 三条代理均 HTTP 200，当前首页为 `1,777,588` bytes、SHA-256 `3fa3fc0caa518a3617f5aaabf2618db26f6f68c5b9d28e50e245c1240530bdee`；DataCurve 三条代理均 HTTP 200，`268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，逐字节一致。按 canonical slug 和重点厂商集合归一化后，两榜没有新增或删除需要建立研究笔记的重点模型；因此本轮仍沿已有 `qwen3-8-max` 条目推进。

Artificial Analysis 当前 Qwen3.8 Max 详情快照为 `3,835,865` bytes、SHA-256 `e879b3b7dfe5beac76808114a200afa69ab7e078d61d1685cea4076c4d8e0541`，标题为 `Qwen3.8 Max (0902)`，第三方字段约为 Intelligence Index `45.4152084980521`、`37.275289705095 tokens/s`、`984K` context 和 `$5.408509428374016`/Intelligence Index task。这些仍是 Artificial Analysis 目录/测量字段。QwenCloud 页面 `last-modified` 更新为 `2026-09-21 11:01:05`，alias 为 `qwen3.8-max-2026-09-02`，并明确为 `qwen3.8-max` 的 upgraded snapshot；页面动态字段使三条代理哈希不同，但正文大小均为 `98,992` bytes。

本轮新增的服务层面试主线：Qwen 文档示例已从 `dashscope-intl.aliyuncs.com` 迁移到 `maas.qwencloudapi.com`，这是 provider adapter/transport 变化，不是模型升级；产品页当前同时给出 input/output/implicit cache `$2/$6/$0.25`，explicit cache creation/read `$2.50/$0.17` 每百万 token。Dynamic Rate Limiting 文档（`408,930` bytes，SHA-256 `ba010f2386fbb32c0b69bcbdde64efb520435cefc16475cb030b55253909949e`）确认 account+model 聚合、workspace override、按自然月调整的 soft TPM；`qwen3.8-max-0902` 保证 TPM 为 `1,500,000 / 1,500,000 / 1,500,000`。面试回答必须把 guaranteed TPM、observed TPM、429/`Retry-After`、queue latency、provider endpoint、account/workspace 和 model revision 作为 serving manifest 字段，不能把保证 TPM 当作模型吞吐或 GPU capacity。

DataCurve 仍只有泛化 `mini_swe_agent_qwen3_8_max_xhigh` 行，没有 `qwen3_8_max_0902` 精确 revision 行；继续不迁移其 `57.4610%` Pass@1、成本、输出 token 或 Agent steps。当前状态为**资料级闭环（runtime recheck）**：待核验 0902 专属技术报告/权重、完整训练配方、生产 kernel、硬件 profiling、线上 tool acceptance、真实限流/重试行为和独立 benchmark。已同步研究笔记、来源索引、模型清单、榜单解释、第二十四册 serving 章节以及论文、项目、题库、练习、术语和知识图谱；下一轮仍只从两张排行榜的八家重点厂商条目选择锚点。

## 2026-09-21 Claude Opus 4.7 runtime recheck 与活动锚点切换

本轮没有从排行榜之外引入模型。复核两张排行榜后，Artificial Analysis 仍有 `claude-opus-4-7` 与 `claude-opus-4-7-non-reasoning` 两个同一基础模型的配置条目；DataCurve 当前没有精确 `mini_swe_agent_claude_opus_4_7_*` 行，因此不迁移 Opus 4.6、4.8、5 或其他 Claude 版本的 Agent 分数。

- Artificial Analysis `/zh` 三条代理均 HTTP 200，快照为 `1,777,588` bytes、SHA-256 `9a3ba0c9ba3f23c190f1607073262738c010f255e4b2d4d37a638c73ba409f6f`；max 详情为 `3,798,671` bytes、SHA-256 `6f97fa78c44721bc9844e91eed41643dea73a736fd46f81a5743e6758f3bff38`。当前 max 字段为 Intelligence Index `40.6897928205908`（estimated）、约 `51.1354` output tokens/s、1M context、`$5/$25` input/output；这些是第三方目录/测量字段。
- DataCurve 三条代理均 HTTP 200、`268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，逐字节一致，仍无精确 Opus 4.7 行。
- Anthropic 官方模型页快照为 `11,882` bytes、SHA-256 `d7ca2e12d25c77f32dda9e59d2e6f1a201a71cc48fc93316c6d2c105188b759b`；确认 `claude-opus-4-7`、1M context、128K synchronous output、300K Batch beta、adaptive thinking、默认 `high`、更新 tokenizer，以及官方状态 `Active (legacy)` 和迁移 Opus 5 的建议。Task budgets、effort、vision、compaction 的本轮快照与哈希已登记在研究笔记。
- 面试主线继续保持三层预算：`effort` 控制 step，task budget 覆盖完整 Agent loop，`max_tokens` 约束单次请求；task budget 的 server-side countdown 不把重发历史重复计费，server-side compaction 也不重置当前 turn 已消耗预算。高分辨率视觉契约为最长边 `2576 px`、最多 `4784` visual tokens；更新 tokenizer 的 `1.0-1.35x` 只作为发布方经验范围。
- 当前状态：**内容专题闭环（runtime recheck）**。研究笔记、来源索引、模型清单、榜单解释、第二册 `7.28`、第二十四册 `32.36` 和 `PAPERS.md`/题库/练习/术语/项目/知识图谱均已同步；没有公开参数、架构、完整训练 recipe、生产 kernel、硬件 profiling、线上 acceptance 或精确 DataCurve Agent 评测。下一轮仍只从两张排行榜的八家重点厂商条目选择锚点或补充实现级证据。

## 2026-09-21 Claude Fable 5.1 runtime recheck 与活动锚点切换

本轮继续使用既有 Artificial Analysis 候选 `claude-fable-5-1`，没有从 Anthropic 官方目录或论文另发现模型。Artificial Analysis 详情快照为 `3,854,152` bytes、SHA-256 `bc83faa8117eebdd2ff28660800be4abf7016af7e10511cb4783f2ca12fbc02a`；canonical slug 为 `claude-fable-5-1`，标题为 `Claude Fable 5.1 (Adaptive Reasoning, Max Effort, Default Fallback)`，`releaseDate=2026-09-01`、`deprecated=false`，主配置 Intelligence Index `53.3549259623252`、median output speed `68.7301566560472 tokens/s`、context `1,000,000`、cost per Intelligence Index task `7.629706364004841`。这些是第三方配置/provider 字段。

本轮官方资源也已固定：模型页 `14,914` bytes / `f773571dce563d7cb8a501b9ad2eb6531d8164938c5868f1aee8dbda8b462f`，发布页 `449,779` bytes / `70d5aaccdd890496070d810944693b9738e8b8c68a97c4e545ddd0e83f9fcd18`，System Card PDF `16,397,488` bytes / `b0d59edc7a60eef32a879c13d713cce60c3fefd7e6b5183afdc8b835af3c8c39`。DataCurve 当前仍无精确 `mini_swe_agent_claude_fable_5_1_*` 行，继续不迁移 Fable 5 的 `316/452` 结果。

本轮把 Fable 5.1 的面试主线从“产品定位”推进到“版本化状态协议”：forced tool use 在不兼容路径返回错误；旧模型不能读取 Fable 5.1 thinking blocks；编辑历史 turn 会使 thinking blocks 失效。新增/明确的运行时能力包括 per-message effort（beta）、turn-scoped system messages（beta）、工具调用间 `display: "updates"` 进度事件、cache read 降价和 content provenance。后续题库、练习、术语、项目、论文索引和知识图谱均按“请求/状态/成本/溯源账本”同步，不新增 Fable 5.1 专属 Transformer 章节。

当前状态为**资料级闭环**：排行榜发现、官方模型页/发布页、System Card 文件入口、研究笔记、运行时 breaking/additive changes 和配套材料均已具备；参数规模、稠密/MoE 架构、完整训练/后训练 recipe、独立技术报告、生产 kernel、硬件 profiling、线上 acceptance 和精确 DataCurve Agent 结果仍待核验。下一步回到两榜单的八家重点厂商候选队列选择新锚点，goal 保持 active。

## 2026-09-21 Claude Opus 5 当前时点复验与活动锚点切换

本轮没有从官方目录、论文或博客另发现模型；活动锚点切换到既有两个排行榜条目 `claude-opus-5`。Artificial Analysis、DataCurve DeepSWE 和 Anthropic 官方页面均通过三条代理复验失败，错误均为无法连接代理服务器：`10.24.27.134:7890`、`10.24.27.134:8098`、`10.237.126.170:1234`。因此本轮不写入新的页面快照或指标，也不把研究笔记中 2026-09-15 的缓存证据宣称为本轮联网结果。

Opus 5 仍按已有证据保持“资料级闭环”：官方文档支持把 thinking block/signature 视为必须原样回放的状态协议，`thinking.display: "omitted"` 与 `thinking disabled` 的 capability 边界，工具/effort 中途变更、refusal/fallback、fallback credit、512-token prompt cache 门槛和 web fetch 不支持。后续面试重点是状态重放、fallback 归因、缓存/成本账本、宿主工具权限、subagent 深度/并发预算和 verifier 门禁；参数、架构、完整训练 recipe、独立技术报告、生产 kernel、硬件 profiling、线上 acceptance 和精确 DataCurve Agent 结果仍待核验。goal 保持 active。

## 2026-09-21 Claude Opus 5 联网恢复后的新鲜快照

本轮先出现短时代理连接失败，随后三条代理均恢复。Artificial Analysis 中文首页均返回 HTTP 200，7890 快照为 `1,778,568` bytes、SHA-256 `f4ba39ee5b5638def213f29fa5e9f6aac9954a26cb291f6a6064283f9b0eba93`；Opus 5 详情三条代理逐字节一致，为 `3,868,875` bytes、SHA-256 `e1710da9c158833bad7c1cc8b02dcfdd77d72ea8f6b5652274faa399dde7b615`。DataCurve 三条代理也逐字节一致，为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。

当前 AA `claude-opus-5` max 的 Intelligence Index 为 `50.7771115797629`（页面 `intelligenceIndexIsEstimated=false`）、median output speed `60.707552496689 tokens/s`、median time to first chunk `46.6837206345s`、1M context、约 `$5.8584`/task；这些是第三方配置/provider 字段，历史 9 月 15 日快照的 `50.7002` 不再作为当前值。按 canonical slug 归一化对比首页，新增项只有非重点厂商或既有 K2 关联条目，八家重点厂商没有新增模型候选。

Anthropic Opus 5 模型页通过 1234/8098 返回 HTTP 200，选定 1234 快照为 `461,560` bytes、SHA-256 `57d20b24a8d7961bd2ea76d71080035677ec27deac07991bcc73cc3d305a03b5`；官方发布页通过 8098 返回 `352,773` bytes、SHA-256 `72490a50c0d5c96021954261ed4201d03c41e5134f2432647eed8ac58644c31f`。页面仍支持 `claude-opus-5`、1M/128K/300K 输出边界、adaptive/high 与五档 effort、工具中途变更和 automatic fallback；发布方评测、fallback 和安全结论仍不能替代独立复现或内部架构证据。goal 保持 active。

## 2026-09-21 GPT-6 Astra 官方模型指南与 Agent 协议补证

本轮沿两个排行榜已有的 `gpt-6-astra` 锚点推进；Artificial Analysis/DataCurve 没有产生新的重点厂商 canonical 模型，因此不新增模型条目。通过 `10.24.27.134:7890` 成功抓取 OpenAI 官方模型页 HTML、专属模型指南、Agents、Async tool calling、Mid-turn steering、Misalignment monitoring 和 Fast mode 文档，以及开发者博客；`10.24.27.134:8098`、`10.237.126.170:1234` 对模型页返回 Vercel `403 Forbidden`，只记为线路/站点防护结果。

- 模型页 HTML：`430,363` bytes，SHA-256 `d2ce52cb3c514d70d124867ff18c1ab14105ae3fc372fead76d4f7532b621864`；正文 Markdown 仍为 3,812 bytes、`f45ae813c3f69708e2576328056de14cdf71fec174c875f4b81d236105545625`，与 9 月 18 日一致。
- 专属指南明确了 async function/custom tool calling：模型可在应用执行慢工具时继续处理独立部分，应用用原始 `call_id` 回传结果；需要额外审计任务句柄、超时、重试、幂等和响应 lineage，不迁移为 hosted built-in tool 的自动执行。
- 专属指南和 steering 文档明确了 GPT-6 Astra 的 Responses WebSocket `response.steer`：追加输入先排队，再创建 continuation；原响应可能以 `incomplete_details.reason=steered` 结束。它不会撤销已发送输出、已启动工具或外部副作用，必须保留补偿/回滚状态。
- 官方安全文档补充 misalignment monitoring 的边界：这是平台异步监控/告警/阻断控制面，不是内部对齐算法；被阻断时匹配 `misalignment_policy_violation`、停止自动重试、保留 request/response/tool 记录，并检查已发生的副作用。监控可能误报/漏报，不替代权限、审批、沙箱和 verifier。
- 官方开发者博客把技能描述、`AGENTS.md` 和任务提示视为上下文路由层，建议短描述、渐进披露、按需读取、减少冲突 recipe，并明确长任务的完成定义与持续范围。该知识已写入第六册第 18 章，配套题库/练习/术语/项目/论文/知识图谱同步。
- `phase` 证据边界已校准：Reasoning 文档现行专节以 GPT-5.5/GPT-5.4 为示例，只作为通用 assistant item 回放规则，不标成 GPT-6 专属新能力；GPT-6 专属直接证据集中在 `configuration_update`、async tool calling 和 WebSocket steering。

本轮官方快照哈希详见 [`gpt-6-astra-source-notes.md`](research/model-update-2026-09/gpt-6-astra-source-notes.md) 与 [`source-index.md`](research/model-update-2026-09/source-index.md)。新增 [`gpt6_agent_protocol_demo.py`](research/model-update-2026-09/code/gpt6_agent_protocol_demo.py)，已通过 `py_compile` 和主流程断言。当前状态：**内容专题闭环（官方运行时与 Agent 协议补证）**；参数规模、内部架构、完整训练/后训练 recipe、system card、专属技术报告、生产 kernel、目标硬件 profiling 和独立 benchmark 仍待核验。下一步做页面哈希复验、toy 的异常路径扩展，若两榜出现新的重点 canonical 模型再切换锚点；goal 保持 active。

## 2026-09-21 Kimi K3 vLLM upstream/runtime recheck

本轮继续只沿两个排行榜中已经发现的 Kimi K3 推进。新增证据分为四层：

1. vLLM stable supported-models 与 stable K3 API 已列出 K3 类名和 `KimiK3MTP`，证明文档/API 入口可见；
2. PyPI release metadata 确认 `vllm 0.29.0` stable wheel 于 2026-09-09 发布，公开 v0.29.0 tag 的 `registry.py`、K3 package init 和 NVIDIA K3 model source 已固定，证明 stable release/source 有 K3 实现入口；
3. K3 recipe 的 raw YAML 仍标记 `Pre-release`，要求 K3-enabled nightly、CUDA 13/cu130 和 r580+ driver，证明这是特定优化部署配方，不能据此否定 stable release，也不能据此证明所有目标硬件已验收；
4. FlashKDA Atom 仍停在 `7afb9f454f160a6c4bbc0999beca0a8c40a38934`，没有新的 kernel/upstream merge 证据。

当前决策：K3 状态从“固定 manifest/config/runtime source evidence”细化为“vLLM 0.29.0 stable release/source 已有实现入口，但目标硬件/权重/runtime/生产验收仍未证明”。后续门禁按 stable wheel 版本、完整权重加载日志、NVIDIA/ROCm 目标卡 profiling、MLA/KDA 双状态恢复、tool-call schema/retry/幂等/线上 acceptance 和独立 benchmark 分开验收；不把页面类名、recipe benchmark 或 stable source entry 写成生产结果。完成 K3 补证后再从两个排行榜的剩余重点 canonical 模型选择下一锚点。

## 2026-09-21 Kimi K3 stable artifact correction

通过 `10.24.27.134:7890` 获取的 PyPI metadata 固定 `vllm 0.29.0`：x86_64 wheel `315,961,042` bytes、SHA-256 `09d48617fc2be9c6cdcd5db480651ab0d84817b257204f2cc2e3ecbb70bbb635`，aarch64 wheel `310,033,787` bytes、SHA-256 `e6b0dfc2b6fd307315e9b34b73cd2bfe7b6b08958eda721828e61732bba426b`，均于 2026-09-09 上传。v0.29.0 tag 中的 registry、package init 和 NVIDIA K3 model source 也已固定；因此旧记录中“stable upstream merge 未证明”的措辞需要改为“stable release/source implementation entry 已证明”。

但本轮没有安装 wheel，也没有下载 K3 权重。recipe 仍要求 K3-enabled nightly/image、CUDA 13/cu130 和 r580+ driver，说明优化 deployment path 与 PyPI stable release 是两个维度。下一步不再重复证明类名，而是验证 stable wheel 的本地安装、目标卡上的完整权重加载、MLA/KDA 双状态、FlashKDA/vLLM backend、性能、故障恢复和工具调用 acceptance；当前环境没有 vLLM/PyTorch/Transformers 或可用 NVIDIA driver，故本轮只能完成静态 manifest/source 门禁，不能伪造运行结果。

## 2026-09-21 Gemini 3.8 Flash Model Card 与长周期 Agent 复验

本轮继续沿两个排行榜已发现的 `Gemini 3.8 Flash` 推进，没有从 Gemini 3.8 Flash Cyber 或 Google 官方目录另发现模型。Artificial Analysis `/zh` 新鲜快照为 `1,776,713` bytes、SHA-256 `0d3b7e635e7c785fb94c5ee30c28e0b61fd42ea398f20c4d122e20f1634266c8`；high 详情为 `3,862,728` bytes、SHA-256 `cf66e756c191ab44aad94ff3ab2867f33ce90225d7af185b4252d5f1c91d5785`。DataCurve 快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。按 canonical slug 归一化后，八家重点厂商没有新增模型。

Google DeepMind Model Card 本轮为 `156,591` bytes、SHA-256 `c09779a8eac8babcee393031fd1644cded00f7ee6f076224c87374271692f369`，明确 Gemini 3.8 基于 Gemini 3.7，架构、训练数据、数据处理、软硬件和评测方法均指向 3.7 Model Card。Google 发布博客本轮为 `407,603` bytes、SHA-256 `7a74091ed7600d91b00e170604633d6d98d7c20a0757591899f94f5af3049603`，公开描述 shared foundational intelligence、long-running agentic loops、递归评估/改进、额外 reasoning steps 和迭代工具调用；这些是发布方系统描述，不能升级为已确认的 RL、verifier 或网络结构。

DataCurve high 的 Pass@1 `73.8255%`、Pass@4 `85.8408%`、平均成本约 `$2.36` 和平均 `166.31` steps 继续绑定 `mini-swe-agent`、工具、任务环境和 verifier，不能迁移到 low/medium 或相邻 Gemini。Google AI Developers 本轮三条线路分别出现 503/超时，旧快照不冒充本轮新鲜响应。当前仍为**资料级闭环**，不新增 Gemini 3.8 专属 Transformer 章节；下一步只做真实 API capability probe、Interactions replay、thinking 消融和工具/长上下文成本实验。

## 2026-09-21 Grok 4.6 当前时点排行榜复验

本轮继续沿两个排行榜中已经发现的 `Grok 4.6` 推进，没有从 xAI 官方目录、论文或其他网站新增模型。Artificial Analysis 详情页当前快照 `/tmp/grok46-aa-20260921.out` 为 `3,859,075` bytes、SHA-256 `a23b19aceae3fee2b1a92421d21358eb5739043c86b6d24a81348d849f887e94`；`Grok 4.6 (high)` 的当前第三方字段为 Intelligence Index `44.3113073012592`、median output speed `66.6843264403358 tokens/s`、TTFT `46.00s`、500K context、`$2/$6`，`releaseDate` 为第三方字段 `2026-08-12`。

9 月 15 日旧详情页的 `44.4050073012592`、`58.5035284934629 tokens/s` 和 `40.8595908855s` 继续保留为历史测量。本计划将两组字段视为不同采集时点的榜单测量，不解释成模型 revision、训练或 API 合同变化。DataCurve 当前 `/tmp/ds-current-1234.html` 为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，四档精确行和原有 Pass@1/成本/steps 一致，仍绑定 `mini-swe-agent`、113 tasks、4 runs、工具、环境和 verifier。

本轮重新尝试 xAI 发布公告、官方模型页、Markdown、Reasoning、Compaction、Tools 和 Remote MCP；`1234` 为 EOF/超时，`7890` 与 `8098` 连接超时。已有 9 月 15 日 xAI 官方快照继续作为历史证据，不作为本轮新鲜响应。Grok 4.6 保持资料级闭环，不新增专属架构章节；下一步继续从两个排行榜的八家重点厂商条目选择新的锚点或补充当前锚点的待核验证据。

## 2026-09-21 GPT-5.6 Luna 当前时点复验与 OpenAI 访问边界

本轮沿两个唯一排行榜中的精确 `GPT-5.6 Luna` 条目继续推进。Artificial Analysis 详情页 `gpt-5-6-luna` 当前快照为 `3,861,087` bytes、SHA-256 `00c856c1ecc7bb7d79363f4d2b6814e9cd15a6a02cb8f0c99060862dfbd99cac`；`max` 的 Intelligence Index `37.3244239690841`、median output speed `164.509614645781 tokens/s`、1M context 和约 `$0.20/$1.20` 都标为第三方配置/provider 字段。DataCurve 快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；精确 `mini_swe_agent_gpt_5_6_luna_max` 为 301/448、Pass@1 `67.1875%`、Pass@4 `90.2655%`、平均成本约 `$0.6056`、平均输出约 `73.4K` token、平均 101.68 steps，绑定 `mini-swe-agent`、工具、环境和 verifier。

本轮对 `developers.openai.com` 的模型索引、Luna 模型页、Reasoning 和 Prompt Caching 页面：1234 返回 HTTP `403`，7890/8098 超时，直连 DNS 失败。既有官方快照继续作为历史资料，不能标成 9 月 21 日新鲜官方响应。GPT-5.6 当前为**双榜资料级闭环**；reasoning state、prompt caching、tool search、compaction 和 harness 主线已同步，参数、架构、完整训练配方、system card、独立报告和生产 profiling 仍待核验。本轮不新增 GPT-5.6 架构章节；官方页面恢复后再补行为/版本差异。

## 2026-09-21 DeepSeek V3.2 当前快照收口

本轮继续沿两个排行榜已发现的 `DeepSeek V3.2` 推进，没有从官方 README、kernel 仓库或 serving 文档另发现模型。AA 当前详情快照为 3,638,730 bytes / SHA-256 `4dbd4f6cb11f25092a221ef0490a910f91f43885330fe34cc38c212c4d7470c3`，结构化字段为 685B/37B、128K、Intelligence Index `16.043537719683`、0.28/0.42 美元；DataCurve 仍无精确 V3.2 行。9 月 20 日的 648B 仅保留为历史 AA 目录字段，不解释为模型 revision。

V3.2-Exp README 新鲜快照为 6,899 bytes / `dffcdf358a42599945d49293a4f210dbe589141085207b76c081c9ace1f8fd74`。本轮计划收口为：把 2025-11-17 indexer non-interleaved RoPE 与 MLA layout 差异、TileLang/DeepGEMM/FlashMLA 层次、SGLang dsv32 启动路径写入实现证据；把 vLLM recipe 当前 404 写成访问边界；保留历史 recipe 和固定 HF revision，不把 404 解释成无实现。

## 2026-09-21：DeepSeek V4.1-Flash `deepseek-recipe` 官方协议实现补证

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作（等同 `/home/zzc/llm-from-zero-to-interview`），没有新增模型候选；仍以 Artificial Analysis 已发现的 `deepseek-v4-1-flash` 为锚点。DataCurve 没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，不迁移其他 DeepSeek 版本的 Agent 结果。
- 通过现有代理获取 DeepSeek 官方 [`deepseek-recipe`](https://github.com/deepseek-ai/deepseek-recipe) 的官方 Atom、pinned raw README 和 commit archive；固定 commit `8cadfede7063c896b944e7bae05daa3549ae97ea`，归档 3,996,119 bytes，SHA-256 `1116ca33e9dc62a913fb9214578c400f1704e6bca33487f4a4c31b32c67a21a6`。GitHub API 因匿名 rate limit 返回 403，但不影响 pinned archive/Atom/raw 文件的版本证据；没有把 403 误判为仓库不存在。
- README 快照为 5,261 bytes、SHA-256 `0cccc69baa118d7689fc3ff2c2e47ab652e05410b777744c43a424f4db5fc0af`；streaming/tokenizer 文档和 Rust 源码进一步核验：`ConversationRequest` 协议规范化、V4.1 `reasoning_effort`/DSML/mid-system 渲染、跨 chunk 的 reasoning/DSML/JSON/stop 状态机、显式 tokenizer bridge、图像 URL/data URL/bytes quota、并发/重试/预处理和 mock server 接线。
- 新增面试知识：协议 adapter、prompt/tokenizer、stream parser、图像安全、inference backend、HTTP transport、tool executor、policy 和 verifier 必须分层；parser 成功不等于工具已经执行、JSON 业务语义有效或 verifier 通过。固定 commit README 明确不负责 inference/transport/tool execution/权限/verifier，并列出 `logprobs`、server-side web search、JSON Schema/strict、`n>1`、Responses storage 和 encrypted thinking 等协议库未支持项。
- 已同步 [`deepseek-v4.1-flash-source-notes.md`](research/model-update-2026-09/deepseek-v4.1-flash-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan.md`](plan.md)、第二十一册第 81 章 `81.10.4`/来源列表、第二十四册第 32.44 节，以及 `INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`PAPERS.md`、`KNOWLEDGE_GRAPH.md`。研究笔记记录了默认图像限制（600/32 MiB/64 MiB/8 concurrent）、5 redirects、10/60 秒超时和默认无 SSRF/private-address filtering 的边界。
- 本轮只完成固定源码归档、哈希和静态阅读；没有下载 V4.1 权重，没有运行真实 inference、CUDA/TileLang、目标硬件 profiling 或线上 API/tool acceptance。当前状态：**内容专题 + reference implementation + recipe protocol evidence（AA 单榜）**；goal 保持 active。下一步仍是完整权重、production kernel、真实 recall/FP4、DSpark draft/verify/rollback、EPD 调度、硬件 profiling、线上 acceptance 和独立 benchmark。

## 2026-09-21：两榜当前时点复验，无新增重点模型，切换 GLM-5.1

- 三条代理重新获取 Artificial Analysis `/zh`，快照为 `1,777,588` bytes，SHA-256 `10630c5152df60351ceff5819c90dfd104f1e9e502959eb36e470abae9ab2c9`。与 9 月 20 日快照相比页面内容/哈希有变化，但按 canonical `models/<slug>` 归一化后，八家重点厂商没有新增或删除模型条目。
- 三条代理重新获取 DataCurve DeepSWE，均为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，逐字节一致；重点 `mini_swe_agent_*` 配置集合与 9 月 20 日一致。`mini_swe_agent_kimi_k2_7_code_default` 是既有 Kimi K2.7 Code 条目，不是新增锚点。
- 本轮没有从排行榜之外引入模型，也没有把 effort/fallback/provider/harness/verifier 误计为基础模型。下一活动锚点切换为已在 Artificial Analysis 候选队列中的 `GLM-5.1`。
- GLM-5.1 当前仍是 **AA 单榜资料级闭环**：研究笔记已覆盖官方模型文档、HF 模型卡/config、Thinking/Function Calling/Cache、release note 和论文负检索；本轮继续补齐 Z.ai 官方博客/发布说明、长周期 Agent 评测 harness、过程质量 verifier、生产 kernel/硬件 profiling 等证据，不迁移相邻 GLM 版本的 DataCurve 结果。

## 2026-09-21：GLM-5.1 官方博客恢复与长周期评测补证

- 之前一次联网记录把 `https://z.ai/blog/glm-5.1` 记为 404；本轮三条代理均返回官方博客壳页面 200，并固定正文 JS 资源：221,954 bytes，SHA-256 `0e2a4ae9177f44509ee54e9105127d17ff65f6d3b5f95294a7b3b3bdb125c08b`。博客壳为 598 bytes，SHA-256 `6fa12ef1d6f8bd1e834b4d8074da0035e36112641ba609d7855d31f92e2727db`；旧负面结论已按时间和线路状态修正。
- 官方博客补出三类长周期反馈实验：VectorDBBench 的 Recall ≥ 95%/QPS 外循环（600+ iterations、6,000+ tool calls、21.5k QPS 发布方结果）；KernelBench Level 3 的 50 problems、独立 Docker/H100、1,200-turn cap、数值正确性和 Claude Opus 4.6/GPT-5.4 双审计器；以及没有单一标量目标的 8 小时 Linux desktop self-review harness。
- 已将新证据同步到研究笔记、来源索引、模型盘点、榜单解释、`PAPERS.md`、`PROJECTS.md`、`INTERVIEW_BANK.md`、`GLOSSARY_EN_ZH.md` 和 `KNOWLEDGE_GRAPH.md`。这些仍是 Z.ai 发布方实验设置，不等于 GLM-5.1 专属技术报告、训练 recipe、独立硬件 profiling 或线上 acceptance；DataCurve 无精确 GLM-5.1 行，继续不迁移相邻 GLM 版本结果。

## 2026-09-21：Qwen3.8 Max (0902) runtime recheck

- 本轮确认工作目录仍为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。模型发现规则仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，未从 QwenCloud 文档另发现模型。
- 三条代理重新获取两榜：Artificial Analysis `/zh` 均 HTTP 200，`1,777,588` bytes，SHA-256 `3fa3fc0caa518a3617f5aaabf2618db26f6f68c5b9d28e50e245c1240530bdee`；DataCurve 均 HTTP 200，`268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，逐字节一致。按重点厂商 canonical 条目归一化后没有新增模型。
- Artificial Analysis Qwen3.8 Max 详情当前为 `3,835,865` bytes、SHA-256 `e879b3b7dfe5beac76808114a200afa69ab7e078d61d1685cea4076c4d8e0541`；标题为 `Qwen3.8 Max (0902)`，第三方字段约 Intelligence Index `45.4152084980521`、`37.275289705095 tokens/s`、`984K` context 和 `$5.408509428374016`/task。QwenCloud 产品页 `last-modified` 为 `2026-09-21 11:01:05`，alias `qwen3.8-max-2026-09-02`，仍明确是 `qwen3.8-max` 的 upgraded snapshot；动态字段导致三条代理 hash 不同，但页面均为 `98,992` bytes。
- 本轮补齐 QwenCloud 服务契约：Thinking/Function Calling/Context Cache 示例已迁移到 `maas.qwencloudapi.com`；这属于 endpoint/provider adapter 变更，不是模型能力或架构升级。当前价格字段为 input/output/implicit cache `$2/$6/$0.25`，explicit cache creation/read `$2.50/$0.17` 每百万 token。
- 新增 [Dynamic Rate Limiting](https://docs.qwencloud.com/developer-guides/administration/dynamic-rate-limits) 证据：文档 `408,930` bytes、SHA-256 `ba010f2386fbb32c0b69bcbdde64efb520435cefc16475cb030b55253909949e`；限流按 account+model 聚合，workspace 可 override，TPM 按自然月调整，达到保证值后仍可能因余量继续服务；`qwen3.8-max-0902` 保证 TPM 为 `1,500,000 / 1,500,000 / 1,500,000`。这是 hosted quota，不是模型吞吐、GPU capacity 或榜单结果。
- DataCurve 仍没有 `qwen3_8_max_0902` 精确行，继续不迁移泛化 `mini_swe_agent_qwen3_8_max_xhigh` 的 `57.4610%` Pass@1、成本、输出 token 或 Agent steps。当前状态：**资料级闭环（runtime recheck）**；待核验 0902 专属技术报告/权重、完整训练 recipe、生产 kernel、硬件 profiling、线上 tool acceptance、真实 429/retry 行为和独立 benchmark。已同步研究笔记、来源索引、模型清单、榜单解释、`plan.md`、第二十四册 serving 章节和全局配套文件；goal 保持 active。

## 2026-09-21：Claude Opus 4.7 runtime recheck 与活动锚点切换

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作（与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同），保留既有 dirty worktree 修改；模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE。
- Artificial Analysis 仍确认 `claude-opus-4-7`（adaptive/max）与 `claude-opus-4-7-non-reasoning`（non-reasoning/high）属于同一基础模型的配置条目。`/zh` 三条代理均 HTTP 200，快照为 `1,777,588` bytes、SHA-256 `9a3ba0c9ba3f23c190f1607073262738c010f255e4b2d4d37a638c73ba409f6f`；max 详情为 `3,798,671` bytes、SHA-256 `6f97fa78c44721bc9844e91eed41643dea73a736fd46f81a5743e6758f3bff38`。当前 max Intelligence Index `40.6897928205908`（estimated）、约 `51.1354` output tokens/s、1M context、`$5/$25` input/output；这些是第三方字段。
- DataCurve 三条代理均 HTTP 200、`268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，逐字节一致；当前没有精确 `mini_swe_agent_claude_opus_4_7_*` 行，不迁移其他 Claude 版本的 Agent 结果。
- Anthropic 官方模型页快照为 `11,882` bytes、SHA-256 `d7ca2e12d25c77f32dda9e59d2e6f1a201a71cc48fc93316c6d2c105188b759b`，确认 API ID `claude-opus-4-7`、1M context、128K synchronous output、300K Batch beta、adaptive thinking、默认 `high`、更新 tokenizer；官方生命周期为 `Active (legacy)`，建议迁移 Opus 5。AA 详情页的 deprecated 只代表第三方目录状态，不能写成官方 API 已下线。
- 已复核的面试主线为 `effort`/task budget/`max_tokens` 三层预算、server-side countdown 与 compaction continuity、`1.0-1.35x` tokenizer 迁移成本、`2576 px/4784 visual tokens` 高分辨率视觉以及 cyber safeguards 的策略/授权/沙箱边界。相关研究笔记、来源索引、模型清单、榜单解释、第二册 `7.28`、第二十四册 `32.36` 和全局配套已同步。
- 当前状态：**内容专题闭环（runtime recheck）**；没有公开参数、架构、完整训练 recipe、生产 kernel、目标硬件 profiling、线上 tool acceptance 或精确 DataCurve Agent 评测。goal 保持 active，下一轮仍只从两张排行榜的八家重点厂商条目选择锚点或补充实现级证据。

## 2026-09-21：Claude Fable 5.1 runtime recheck 与活动锚点切换

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作（与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同），保留既有 dirty worktree 修改；模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，没有从官方目录、论文或博客另发现模型。
- Artificial Analysis Fable 5.1 详情快照为 `3,854,152` bytes、SHA-256 `bc83faa8117eebdd2ff28660800be4abf7016af7e10511cb4783f2ca12fbc02a`；canonical slug `claude-fable-5-1`，主配置标题为 `Claude Fable 5.1 (Adaptive Reasoning, Max Effort, Default Fallback)`，`releaseDate=2026-09-01`、`deprecated=false`，Intelligence Index `53.3549259623252`、median output speed `68.7301566560472 tokens/s`、context `1,000,000`、cost per Intelligence Index task `7.629706364004841`。这些是第三方配置/provider 字段。
- Anthropic 官方资源已重新固定：模型页 `14,914` bytes / `f773571dce563d7cb8a501b9ad2eb6531d8164938c5868f1aee8dbda8b462f`，发布页 `449,779` bytes / `70d5aaccdd890496070d810944693b9738e8b8c68a97c4e545ddd0e83f9fcd18`，System Card PDF `16,397,488` bytes / `b0d59edc7a60eef32a879c13d713cce60c3fefd7e6b5183afdc8b835af3c8c39`。DataCurve 仍无精确 `mini_swe_agent_claude_fable_5_1_*` 行，不迁移 Fable 5 的 `316/452` 结果。
- 新增面试知识：forced tool use 不兼容时返回错误；较早模型不能读取 Fable 5.1 thinking blocks；编辑较早历史 turn 会使 thinking blocks 失效。新增/明确能力为 per-message effort（beta）、turn-scoped system messages（beta）、工具调用间 `display: "updates"` 进度事件、cache read 降价和 content provenance。已将这些理解为带版本、状态、成本和溯源账本的运行时协议，不把 thinking block 当普通文本、不把进度事件当工具成功、不把 provenance 当事实正确性证明。
- 已同步 Fable 5.1 研究笔记、来源索引、模型清单、榜单解释、`plan.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`PAPERS.md`、`KNOWLEDGE_GRAPH.md` 和第四册百科既有章节；不新增 Fable 5.1 专属 Transformer 章节，复用 Agent、reasoning、tool protocol、serving 和安全章节。
- 当前状态：**资料级闭环**。参数规模、稠密/MoE 架构、完整训练/后训练 recipe、独立技术报告、生产 kernel、硬件 profiling、线上 acceptance 和精确 DataCurve Agent 结果仍待核验；goal 保持 active，下一轮回到两榜单的八家重点厂商候选队列选择新锚点。

## 2026-09-21：Claude Opus 5 当前时点复验与活动锚点切换

- 本轮没有从排行榜之外引入模型；活动锚点切换到已在 Artificial Analysis 与 DataCurve DeepSWE 出现的 `Claude Opus 5`。当前时点重新尝试 `10.24.27.134:7890`、`10.24.27.134:8098`、`10.237.126.170:1234`，三条代理均无法连接，未产生新的网页快照或指标。
- 研究笔记中的 Artificial Analysis/DataCurve/Anthropic 资料仍是 2026-09-15 的缓存证据，不能冒充本轮联网成功。Opus 5 继续保持资料级闭环，面试补证主线为 `display: "omitted"`、thinking block/signature 原样回放、`thinking disabled` 与 `xhigh/max` capability gate、refusal/fallback、fallback credit、工具/effort 中途变更、512-token cache、web fetch 宿主边界和 subagent/自验证预算。
- 不迁移其他 Claude 版本的 DataCurve 成绩；参数、架构、完整训练/后训练 recipe、独立技术报告、生产 kernel、硬件 profiling、线上 acceptance 和精确独立 Agent 评测仍待核验。goal 保持 active。

## 2026-09-21：Claude Opus 5 联网恢复后的新鲜快照

- 三条代理当前均可用：Artificial Analysis 中文首页 HTTP 200，7890 快照 `1,778,568` bytes / SHA-256 `f4ba39ee5b5638def213f29fa5e9f6aac9954a26cb291f6a6064283f9b0eba93`；Opus 5 详情三条代理逐字节一致，`3,868,875` bytes / SHA-256 `e1710da9c158833bad7c1cc8b02dcfdd77d72ea8f6b5652274faa399dde7b615`。DataCurve 三条代理逐字节一致，`268,571` bytes / SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。
- 当前 AA Opus 5 max Intelligence Index 为 `50.7771115797629`，median output speed `60.707552496689 tokens/s`，median TTFC `46.6837206345s`，1M context，约 `$5.8584`/task；这些是第三方字段。DataCurve max 仍为 327/444、Pass@1 `73.6486%`、Pass@4 `88.4956%`、平均成本 `$11.8376`，未迁移其他 Claude 版本结果。
- 两榜按 canonical slug 归一化后没有新增八家重点厂商的模型候选；AA 页面新增/移除的其他 slug 不进入本项目队列。Anthropic 模型页 1234/8098 内容一致，选定快照 `461,560` bytes / SHA-256 `57d20b24a8d7961bd2ea76d71080035677ec27deac07991bcc73cc3d305a03b5`；发布页 8098 快照 `352,773` bytes / SHA-256 `72490a50c0d5c96021954261ed4201d03c41e5134f2432647eed8ac58644c31f`。
- 当前状态仍为**资料级闭环**：新鲜快照只更新榜单/页面字段，不增加参数、架构、训练 recipe 或独立 benchmark 结论；goal 保持 active。

## 2026-09-21：GPT-6 Astra 官方模型指南与 Agent 协议补证

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作（等同 `/home/zzc/llm-from-zero-to-interview`），模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE；两榜没有新增八家重点厂商 canonical 模型，因此没有新增模型笔记或迁移其他模型的 Agent 结果。
- `10.24.27.134:7890` 成功返回 OpenAI 官方 GPT-6 Astra 模型页、专属模型指南、Agents、Async tool calling、Mid-turn steering、Misalignment monitoring、Fast mode 和开发者博客；模型页 HTML 为 `430,363` bytes、SHA-256 `d2ce52cb3c514d70d124867ff18c1ab14105ae3fc372fead76d4f7532b621864`。`10.24.27.134:8098` 与 `10.237.126.170:1234` 对模型页返回 Vercel `403 Forbidden`，只记录为线路/站点防护，不能解释为资料不存在。
- 专属模型指南新增 `async: true` 的 function/custom tool 调用：模型可在应用运行慢工具时处理独立部分，应用仍负责 job registry、原始 `call_id`、超时、重试、幂等、权限和结果回灌；新增 WebSocket `response.steer`：输入先排队，再生成 continuation，原响应可能以 `incomplete_details.reason=steered` 结束，已发送文本、已启动工具和外部副作用不会自动回滚。
- `misalignment monitoring` 被记录为平台安全控制面：它异步检查高后果请求的推理与动作，可能告警或阻断；阻断时匹配 `misalignment_policy_violation`、停止自动重试、保留 request/response/tool trace 并检查已有副作用。监控可能误报或漏报，不能替代权限、沙箱、审批和 verifier。
- 官方博客把 skills、`AGENTS.md` 和任务提示视为上下文路由层，补充短描述、渐进披露、按需读取、减少冲突 recipe、明确完成条件和持续范围等面试知识。已同步研究笔记、第六册第 18 章、第四册百科、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`PAPERS.md` 和 `KNOWLEDGE_GRAPH.md`。
- `phase` 证据边界已修正：Reasoning 文档现行专节以 GPT-5.5/GPT-5.4 为示例，只作为通用 assistant item replay 规则，不标成 GPT-6 专属新能力。GPT-6 Astra 当前仍没有公开参数规模、内部架构、完整训练/后训练 recipe、system card、专属技术报告、生产 kernel、目标硬件 profiling 或独立 benchmark。
- 当前状态：**内容专题闭环（官方运行时与 Agent 协议补证）**。新增 [`gpt6_agent_protocol_demo.py`](research/model-update-2026-09/code/gpt6_agent_protocol_demo.py)，6,621 bytes、SHA-256 `4156ef886bbcfc3a3ac8358d4735c80000da79fc4bfb875d0b65c229952b3afa`，已通过 `py_compile` 和主流程断言，覆盖 async tool 的 `call_id`/重复回传、steering lineage/不可回滚副作用和 skill progressive disclosure；下一步进行异常路径扩展、`git diff --check`、Markdown fence/链接检查和新鲜官方哈希复验；goal 保持 active。

## 2026-09-21：Kimi K3 固定 manifest、混合 cache 与源码边界审计

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作（等同用户指定的 `/home/zzc/llm-from-zero-to-interview`），没有从排行榜之外引入模型。Artificial Analysis 当前快照为 `1,778,568` bytes、SHA-256 `f4ba39ee5b5638def213f29fa5e9f6aac9954a26cb291f6a6064283f9b0eba93`；DataCurve DeepSWE 当前快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。按 canonical slug 归一化后，八家重点厂商没有新增模型候选；不把 effort、fallback、provider、harness 或 verifier 当成模型，也不迁移其他版本的 Agent 结果。
- 当前活动锚点切换为已在两榜发现的 `Kimi K3`。官方 Hugging Face 固定 revision 为 `f831ab66814297da540d832a5235f8e904f29d06`；配置快照为 `7,006` bytes、SHA-256 `9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213`。本轮重新获取 `model.safetensors.index.json`，文件为 `59,764,096` bytes、SHA-256 `a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd`。
- 新增 [`kimi_k3_manifest_audit.py`](research/model-update-2026-09/code/kimi_k3_manifest_audit.py)，只读取 JSON，不加载权重；`py_compile` 和主流程断言均通过。审计确认 `497,220` 个 tensor、连续的 `model-00001-of-000096.safetensors` 至 `model-00096-of-000096.safetensors`、93 层、69 个 KDA 层、24 个 full-attention 层、92 个 expert-bearing layer、每层 896 experts，以及 `247,296` 对 `weight_packed`/`weight_scale` 配对；量化字段为 MXFP4、4 bit。
- 证据口径已分开：index 的 `metadata.total_size=1,560,860,324,864` 是 MXFP4 packed 分片文件口径；HF API 的 `safetensors.total=2,779,931,837,184` 是 U8/BF16/F32 参数 dtype 统计，不能相加或互换为一个“模型大小”。完整 safetensors 权重没有下载，manifest 一致性也不等于本地模型加载成功。
- 固定的 `modeling_kimi_linear.py`（SHA-256 `9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a`）确认 `KimiDynamicCache` 分开保存 full-attention 的 `key_cache/value_cache` 与 KDA 的 `conv_states/recurrent_states`；单 token decode 使用 `fused_recurrent_kda`，prefill/chunk 使用 `chunk_kda` 和 `cu_seqlens`。因此 serving manifest 不能只记录一个 KV-cache 长度，cache 恢复必须同时验证 MLA attention state、KDA recurrent state 和短卷积 state。
- 已同步 `kimi-k3-source-notes.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan.md`、第二十一册第 88 章、十七册/二十四册对应章节以及题库、练习、术语、项目、论文和知识图谱。当前状态为 **内容专题闭环 + fixed manifest/runtime source + vLLM 0.29.0 stable release/source evidence**；仍待核验本地 wheel 安装、完整 serving recipe、完整权重加载、目标硬件 profiling、hybrid cache recovery、线上 tool-call acceptance、完整训练/optimizer recipe 和独立 benchmark。完成这些 K3 补证后，再从两个排行榜的剩余重点 canonical 模型选择下一锚点；goal 保持 active。

## 2026-09-21：Kimi K3 vLLM upstream/runtime recheck

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作（等同用户指定的 `/home/zzc/llm-from-zero-to-interview`），只沿 Artificial Analysis 与 DataCurve DeepSWE 中已经发现的 `Kimi K3` 推进，没有从 vLLM registry、API 文档或 FlashKDA 仓库另发现模型；goal 保持 active。
- 三条代理在本轮均可访问外网并已用于联网核验：`10.24.27.134:7890`、`10.24.27.134:8098`、`10.237.126.170:1234`。本节记录的是 2026-09-21 的 vLLM/FlashKDA 快照，不把前一时段失败线路当成网页不存在。
- vLLM stable supported-models 页面 `https://docs.vllm.ai/en/stable/models/supported_models/` 为 `738,169` bytes、SHA-256 `5420be70a431de9d6e25ba8322c27c1397239fc11643544b354591b8de80d069`，页面更新时间 2026-08-29；列出 `KimiK3ForConditionalGeneration`、`Kimi-K3` 和 `moonshotai/Kimi-K3`。stable K3 API `https://docs.vllm.ai/en/stable/api/vllm/models/kimi_k3/` 为 `799,430` bytes、SHA-256 `d17b052921ae5ac2216453edf2034854f45a7d1e46a37ef90eff13e25e5cab5b`，页面更新时间 2026-09-09，暴露 `KimiK3ForConditionalGeneration` 与 `KimiK3MTP`。latest supported-models 另为 `792,208` bytes、SHA-256 `4604b83e2dffa3d6cf154003bf4aa1a29d3b7d7503ed278b556c90f7c29d098f`，不与 stable 混写。
- vLLM main registry `https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/registry.py` 为 `64,391` bytes、SHA-256 `64c80d8c6659833a9abf836180f2b7549903db2a664e22bc38e38c258a6f572e`，包含 `KimiLinearForCausalLM`、`KimiK3ForConditionalGeneration`、`K3DSparkModel`、`KimiK3MTPModel`；K3 package `__init__.py` 为 `1,444` bytes、SHA-256 `a8ce9cbcd0d8f45f27fe58e70f7dbe080da30886a744f21e60a8ee13ea955a0b`，按 `current_platform` 分流 NVIDIA/ROCm，TPU 不主动加载 GPU 实现。本轮只把它们作为 main 源码/硬件隔离入口证据。
- vLLM K3 recipe raw YAML `https://raw.githubusercontent.com/vllm-project/recipes/main/models/moonshotai/Kimi-K3.yaml` 为 `19,414` bytes、SHA-256 `5b2bcd21b8dfe210ea3f3d514c6a5448e395847d35fcb1f137d3567233e9d227`，`date_updated=2026-09-10`、`min_vllm_version=0.29.0`，仍标记 `Pre-release`，并要求 K3-enabled nightly/image、CUDA 13/cu130、NVIDIA r580+ driver。recipe 的 hybrid KV manager、TP/TEP/DEP/PP、prefix-match unit 128、DCP 和 tool-call parser 警告已写入研究笔记，但不升级为 stable wheel/生产 SLO。
- FlashKDA README 当前为 `4,431` bytes、SHA-256 `fc56ca9a3cd1786d0ff526a12be21500d79ee671e5f0eeb8352fad23d3ba8fb2`；Atom 为 `8,454` bytes、SHA-256 `2155ff08883c6240b3fb53920a8ecdabfd79d2045f41db35d15e4b9d699f07c5`；最新可见 commit 仍为 `7afb9f454f160a6c4bbc0999beca0a8c40a38934`（2026-09-01）。没有新的 kernel 或 upstream merge 证据。
- 证据边界正式收口：stable docs/API 与 vLLM main registry/package 已有 K3 实现入口；recipe 仍是 pre-release/nightly 部署路径；完整权重加载、stable wheel 在目标硬件运行、NVIDIA/ROCm profiling、hybrid cache recovery、线上 tool-call acceptance 和独立 benchmark 仍未证明。不能把“文档能查到类名”说成“生产 serving 已完成”。
- 已同步 `kimi-k3-source-notes.md`、`source-index.md`、`model-inventory.md`、`plan.md`、第二十一册第 88 章、第二十四册 `32.46`，以及题库、练习、术语、项目、论文和知识图谱。manifest audit、`py_compile`、Markdown fence/链接检查和 `git diff --check` 均已通过；下一步只剩最终 diff 审阅和后续 stable wheel/目标硬件/线上 acceptance 验收。

## 2026-09-21：Kimi K3 vLLM 0.29.0 stable artifact correction

- 本轮沿两个排行榜已发现的 `Kimi K3` 继续核验，没有从 PyPI 或 vLLM 另发现模型。此前把 K3 写成“stable upstream merge 未证明/只有 nightly 入口”过于保守，现根据 PyPI release metadata 和 v0.29.0 tag source 修正证据边界。
- 通过 `10.24.27.134:7890` 成功取得 [PyPI vLLM metadata](https://pypi.org/pypi/vllm/json)：完整 JSON `253,652` bytes、SHA-256 `a232f3e31b111ebfb0d99cbe69b71da682a680ccb4cbf1c57f8cdab3801e5b94`；`info.version=0.29.0`。x86_64 wheel `315,961,042` bytes、SHA-256 `09d48617fc2be9c6cdcd5db480651ab0d84817b257204f2cc2e3ecbb70bbb635`，aarch64 wheel `310,033,787` bytes、SHA-256 `e6b0dfc2b6fd307315e9b34b73cd2bfe7b6b08958eda721828e61732bba426b`，均于 `2026-09-09` 上传。本轮没有下载或安装 wheel。
- v0.29.0 tag source：`registry.py` `63,102` bytes / SHA-256 `fef8293fe19cef01768a4c5bd8adc05e120202f1c1597eb264d090672894fbd7`；K3 package init `1,444` bytes / SHA-256 `a8ce9cbcd0d8f45f27fe58e70f7dbe080da30886a744f21e60a8ee13ea955a0b`；NVIDIA K3 `model.py` `87,313` bytes / SHA-256 `e74026a83c28cce7ee62889e9ed434d1fff6076dbd263d388aa379f4cf987a4d`。registry 含 K3/MTP/DSpark，package 含 NVIDIA/ROCm/TPU 分支，NVIDIA model 含 K3 KDA/MLA/MoE 实现入口。
- K3 当前正确状态：**vLLM 0.29.0 stable release/source implementation evidence 已证明；目标硬件加载、完整权重、FlashKDA/backend 性能、MLA/KDA hybrid cache recovery、线上 tool-call acceptance 和独立 benchmark 仍未证明**。K3 recipe 的 `Pre-release`/K3-enabled nightly、CUDA 13/cu130 和 r580+ driver 是特定优化部署路径，不是“stable release 不存在”的证据。
- 7890 成功获取 PyPI；直连 DNS 失败，8098/1234 对 PyPI metadata 分别超时并留下截断文件，均只记录为线路状态。一次 source-tar 下载也只得到空/截断文件，未引用其内容或哈希作为证据。
- 已同步研究笔记、来源索引、模型清单、计划和进度；下一步同步教材中“stable artifact vs optimized recipe”的措辞，并继续做目标 runtime/权重加载/双状态恢复门禁。goal 保持 active。

## 2026-09-21：Kimi K3 stable artifact 本机门禁复核

- 复核工作目录仍为 `/data/zzc/llm-from-zero-to-interview`（等同 `/home/zzc/llm-from-zero-to-interview`）。K3 manifest audit 重新运行通过：`497,220` tensors、96 shards、93 layers、69 KDA/24 full-attention、92 expert-bearing layers、896 experts/layer、`weight_packed`/`weight_scale` 各 `247,296` 且成对；`py_compile` 也通过。
- 本机当前没有 `vllm`、`torch` 或 `transformers` Python 包，`nvidia-smi -L` 报告无法与 NVIDIA driver 通信；因此不能在本环境声称完成 stable wheel 导入、K3 权重加载、CUDA kernel、双状态恢复或 profiling。
- 根据 PyPI metadata 解析到的 x86_64 wheel URL 通过 `10.24.27.134:7890` 尝试下载约 55 秒仍无字节返回，已中止且未留下 wheel 文件；不把该失败写成 artifact 不存在。现有 PyPI metadata 与 v0.29.0 tag source 仍足以证明 stable release/source implementation entry，但本机 installation gate 保持未通过。
- 证据口径已统一：K3 recipe 的 `Pre-release`/K3-enabled nightly 是优化部署路径；vLLM `0.29.0` stable release/source 有 K3 实现入口；两者都不能替代目标硬件和生产 serving 验收。下一步若获得可用下载线路、GPU 节点或预装运行环境，再执行 wheel import、最小 config/model construction、完整权重加载、MLA/KDA cache recovery 和 tool-call acceptance；否则继续从两榜剩余重点 canonical 模型选择下一锚点。

## 2026-09-21：Kimi K3 stable artifact 本机门禁复核

- 复核工作目录仍为 `/data/zzc/llm-from-zero-to-interview`（等同 `/home/zzc/llm-from-zero-to-interview`）。K3 manifest audit 重新运行通过：`497,220` tensors、96 shards、93 layers、69 KDA/24 full-attention、92 expert-bearing layers、896 experts/layer、`weight_packed`/`weight_scale` 各 `247,296` 且成对；`py_compile` 也通过。
- 本机当前没有 `vllm`、`torch` 或 `transformers` Python 包，`nvidia-smi -L` 报告无法与 NVIDIA driver 通信；因此不能在本环境声称完成 stable wheel 导入、K3 权重加载、CUDA kernel、双状态恢复或 profiling。
- 根据 PyPI metadata 解析到的 x86_64 wheel URL 通过 `10.24.27.134:7890` 尝试下载约 55 秒仍无字节返回，已中止且未留下 wheel 文件；不把该失败写成 artifact 不存在。现有 PyPI metadata 与 v0.29.0 tag source 仍足以证明 stable release/source implementation entry，但本机 installation gate 保持未通过。
- 证据口径已统一：K3 recipe 的 `Pre-release`/K3-enabled nightly 是优化部署路径；vLLM `0.29.0` stable release/source 有 K3 实现入口；两者都不能替代目标硬件和生产 serving 验收。下一步若获得可用下载线路、GPU 节点或预装运行环境，再执行 wheel import、最小 config/model construction、完整权重加载、MLA/KDA cache recovery 和 tool-call acceptance；否则继续从两榜剩余重点 canonical 模型选择下一锚点。

## 2026-09-21：Gemini 3.8 Flash Model Card 与长周期 Agent 复验

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作（等同 `/home/zzc/llm-from-zero-to-interview`），模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE；没有从 Gemini 3.8 Flash Cyber 或 Google 官方目录另发现模型。AA `/zh` 快照为 `1,776,713` bytes / SHA-256 `0d3b7e635e7c785fb94c5ee30c28e0b61fd42ea398f20c4d122e20f1634266c8`，Gemini 3.8 high 详情为 `3,862,728` bytes / SHA-256 `cf66e756c191ab44aad94ff3ab2867f33ce90225d7af185b4252d5f1c91d5785`；DataCurve 为 `268,571` bytes / SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。按 canonical slug 归一化后八家重点厂商没有新增模型。
- DeepMind Model Card 本轮为 `156,591` bytes / SHA-256 `c09779a8eac8babcee393031fd1644cded00f7ee6f076224c87374271692f369`，明确 3.8 基于 3.7，架构、训练数据、数据处理、软硬件和评测方法均指向 3.7 Model Card。官方博客本轮为 `407,603` bytes / SHA-256 `7a74091ed7600d91b00e170604633d6d98d7c20a0757591899f94f5af3049603`，公开描述 shared foundational intelligence、long-running agentic loops、递归评估/改进、额外 reasoning steps 和迭代工具调用；这些仍是发布方系统描述，不是已确认的 RL、verifier 或网络结构。
- DataCurve high 的 Pass@1 `73.8255%`、Pass@4 `85.8408%`、平均成本约 `$2.36` 和平均 `166.31` steps 绑定 `mini-swe-agent`、工具、环境和 verifier，不能迁移到 low/medium 或相邻 Gemini。Google AI Developers 本轮经 1234 返回 503，7890/8098 超时，旧快照不冒充新鲜响应。
- 当前状态保持**资料级闭环**：研究笔记、来源索引、模型清单、榜单解释、面试题库、术语、论文、练习、项目和知识图谱均已同步；不新增 3.8 专属 Transformer 章节。`git diff --check`、研究代码 `py_compile`、K3 manifest audit、GPT-6/DeepSeek/KDA 教学脚本运行和 Markdown 围栏检查均通过；下一步只补真实 API capability probe、Interactions replay、thinking 消融及工具/长上下文成本实验，goal 保持 active。

## 2026-09-21：Grok 4.6 当前时点排行榜复验

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE。本轮没有从 xAI 官方目录、论文或其他网站新增模型，活动锚点切换为已有两榜条目 `Grok 4.6`。
- Artificial Analysis 当前详情页 `/tmp/grok46-aa-20260921.out` HTTP 200，文件为 `3,859,075` bytes，SHA-256 `a23b19aceae3fee2b1a92421d21358eb5739043c86b6d24a81348d849f887e94`。`Grok 4.6 (high)` 当前第三方字段为 Intelligence Index `44.3113073012592`、输出速度 `66.6843264403358 tokens/s`、TTFT `46.00s`、500K context、`$2/$6`，`releaseDate` 为 `2026-08-12`。9 月 15 日缓存快照的 `44.4050073012592`、`58.5035284934629 tokens/s`、`40.8595908855s` 保留为历史测量，不解释为模型 revision 或训练变化。
- DataCurve 当前 `/tmp/ds-current-1234.html` 为 `268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；页面仍为 2026-09-03 更新、`generated_at=2026-09-03T22:24:37.984682+00:00`、113 tasks/91 repositories/5 languages、统一 `mini-swe-agent`。四档 `mini_swe_agent_grok_4_6_*` 的 Pass@1、Pass@4、平均成本、输出 token 和 Agent steps 与既有记录一致，并继续绑定 4 runs、工具、环境和 verifier。
- 本轮重新尝试 xAI 发布公告、模型页、Markdown、Reasoning、Compaction、Tools 和 Remote MCP：`10.237.126.170:1234` 为 EOF/超时，`10.24.27.134:7890` 与 `10.24.27.134:8098` 连接超时。已有 9 月 15 日官方快照不冒充当前响应；没有新增官方技术结论，不新增 Grok 4.6 专属正式章节。
- 当前状态：Grok 4.6 仍为**资料级闭环**；已同步研究笔记、来源索引、模型清单、榜单解释、计划和进度。本轮 `git diff --check`、研究代码 `py_compile`、K3 manifest audit 和 704 个 Markdown 文件的围栏配对检查均通过；本轮没有新增相对链接，既有全库相对链接检查保持通过。goal 保持 active，下一步仍从两榜八家重点厂商条目选择或复验锚点。

## 2026-09-21：GPT-5.6 Luna 当前时点复验与 OpenAI 访问边界

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；只沿两个排行榜已发现的 `GPT-5.6 Luna` 推进，没有从 OpenAI 官方目录另发现模型。AA 详情快照为 `3,861,087` bytes / SHA-256 `00c856c1ecc7bb7d79363f4d2b6814e9cd15a6a02cb8f0c99060862dfbd99cac`，max 的 Intelligence Index `37.3244239690841`、164.5096 tokens/s、1M context、约 `$0.20/$1.20`；DataCurve 快照为 `268,571` bytes / SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，精确 max 行 301/448、Pass@1 `67.1875%`、Pass@4 `90.2655%`、平均成本约 `$0.6056`、平均输出约 `73.4K` token、101.68 steps。
- 结果归因保持分账：AA 是第三方 max 配置测量；DataCurve 是 `mini-swe-agent`、工具、环境和 verifier 的组合系统，不能拼成裸模型能力。
- OpenAI 官方模型索引、Luna、Reasoning 和 Prompt Caching 页面本轮经 1234 返回 HTTP `403`，7890/8098 超时，直连 DNS 失败。既有官方页面快照继续作为历史证据，不冒充当前新鲜响应，也不因访问失败断言模型不存在或 API 已变化。
- 已同步 `gpt-5.6-source-notes.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan.md`、本文件、第四册 Agent harness 正文及论文/题库/练习/术语/项目/知识图谱配套。GPT-5.6 当前为**双榜资料级闭环**，不新增架构章节；参数、内部结构、完整训练/后训练 recipe、system card、独立报告、目标硬件 profiling 和线上 acceptance 仍待核验。goal 保持 active。

## 2026-09-21：GLM-5.3-Flash serving/runtime 复核完成

- 当前活动锚点回到已有双榜模型 `GLM-5.3-Flash`；本轮没有从官方页面新增模型。`GLM-5.3-FlashX` 仅作为 Z.ai 关联服务入口记录，不在两个排行榜中新增锚点。
- Artificial Analysis 当前详情快照为 `3,942,546` bytes、SHA-256 `42f880600d3637489a0c48ff27357fe7986e53510ad5ee016c7d6170bd201fae`；release `2026-08-26`、context `1,048,576`、Intelligence Index `41.807466113455`、median output speed `95.0131572798129 tokens/s`。与旧快照相比的指数/速度变化按第三方测量漂移记录，不解释成模型版本或训练变化。
- DataCurve 精确行仍为 `mini_swe_agent_glm_5_3_flash_max`：284/448、Pass@1 `63.392857%`、Pass@4 `84.955752%`、平均成本 `$0.2409818562`、平均输出 `72829.77` token、约 122.89 steps；结果仍绑定 `mini-swe-agent`、工具、任务集、环境和 verifier。
- Z.ai 官方文档补证了 1M context/128K max output、thinking enabled、`reasoning_effort`、`clear_thinking`、`tool_stream`、FlashX endpoint 与 Coding Plan 配额边界；API 字段没有被写成内部训练事实。
- SGLang cookbook 已核对：45 个文本层（MLA/DSA/KDA）、24 层视觉 encoder、288 routed/top-8、原生 MTP；paged KV pool 与 KDA state pool 双账本；低延迟 MTP `5/1/6`、高吞吐可关 speculative；Blackwell/Hopper 的 FP8/BF16 KV 与 TRT-LLM/TileLang DSA 必须配对；EPD/PD 的视频、encoder disaggregation、dummy-weight 和数值 correctness 门禁已写入研究笔记与第 84 章。
- vLLM recipe 已记录 v0.29.0+、native FP8/MTP、hybrid KDA+sparse MLA、Ascend/MI355X 等路线和 FlashInfer 页面内部版本门槛差异；Transformers GLM5-Next 明确不含 MTP layer。三者分别属于 serving recipe、部署实现和基础接入证据，不能互相升级。
- 已更新研究笔记、来源索引、模型盘点、榜单解释、`plan.md`、第 84 章、第 32 章及 `INTERVIEW_BANK.md`、`EXERCISES.md`、`PROJECTS.md`、`GLOSSARY_EN_ZH.md`、`PAPERS.md`、`KNOWLEDGE_GRAPH.md`；`git diff --check`、研究代码 `compileall`、全库 Markdown 围栏配对和 47 个改动 Markdown 文件的本地相对链接检查均通过。

## 2026-09-21：DeepSeek V3.2 当前 AA 快照与 Exp README 复核

- 工作目录仍为 /data/zzc/llm-from-zero-to-interview，等同 /home/zzc/llm-from-zero-to-interview；本轮只沿两个排行榜已经发现的 DeepSeek V3.2 推进，没有从官方仓库、kernel 或 serving 文档新增模型。
- Artificial Analysis 详情快照 /tmp/v32-aa-20260921.out 为 3,638,730 bytes、SHA-256 4dbd4f6cb11f25092a221ef0490a910f91f43885330fe34cc38c212c4d7470c3。当前结构化字段为 DeepSeek V3.2 (Non-reasoning)、release 2025-12-01、685B total/37B active、128K context、Intelligence Index 16.043537719683、输入/输出 0.28/0.42 美元；当前对象没有 output-speed/TTFT 字段。
- 9 月 20 日详情快照的 648B 与本轮 685B 属于第三方目录/provider 页面漂移，不构成模型 revision、训练或架构变化证据；旧快照仍保留为历史测量。当前 AA /zh 为 1,776,748 bytes / e73b156ffdc11dd391b48ac9c9f114f31bfab711ba9171a55f66c647cf22c73a；DataCurve 为 268,571 bytes / 67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870，仍没有精确 mini_swe_agent_deepseek_v3_2_* 行。
- V3.2-Exp README 为 6,899 bytes / dffcdf358a42599945d49293a4f210dbe589141085207b76c081c9ace1f8fd74；新增收口点为 V3.1-Terminus 对齐对照、indexer non-interleaved RoPE 与 MLA layout 修复、TileLang/DeepGEMM/FlashMLA 三层实现、SGLang dsv32 镜像和 tp=8, dp=8, enable-dp-attention 启动命令。README benchmark 仍是发布方对照，不是 DataCurve。
- vLLM recipe 当前 URL 通过 1234 返回 HTTP 404，已记录为 URL/线路负证据，不解释为 vLLM 没有实现；HF/PDF 503 也只代表当前访问失败，不覆盖固定 revision 历史证据。
- 已同步 V3.2 研究笔记、来源索引、模型清单、榜单解释、plan.md、第二十一册第 19 章及 INTERVIEW_BANK.md、EXERCISES.md、PROJECTS.md、GLOSSARY_EN_ZH.md、PAPERS.md、KNOWLEDGE_GRAPH.md。当前状态仍为 **AA 单榜资料级闭环**；完整 production kernel、召回/误差曲线、硬件 profiling、线上 tool-call acceptance、独立 benchmark 和完整 RL recipe 待核验。

## 2026-09-21：Claude Sonnet 5 System Card 深读与当前快照复核

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同用户指定的 `/home/zzc/llm-from-zero-to-interview`。本轮没有从官方目录新增模型，继续沿两个排行榜已经发现的 `Claude Sonnet 5` 锚点推进；goal 保持 active。
- 当前 Artificial Analysis Sonnet 5 详情快照为 `/tmp/aa-sonnet5-20260921.html`，3,872,014 bytes，SHA-256 `2bd51075cf1ad426140a25dddb097e62af0ef4d07308c00fbc8c022b9158318e`。当前 `max` 为 Intelligence Index `38.1638712882576`、median output speed `86.1790204452512 tokens/s`、median TTFC `137.660255319s`；9 月 15 日的 `38.357696...`、约 80 tokens/s、约 202.63s 只保留为历史 provider/测量值，不解释为模型 revision 或训练变化。
- DataCurve 当前快照为 268,571 bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；Sonnet 5 的 max/xhigh/high/medium/low 仍分别为 Pass@1 `53.846%/49.667%/48.230%/39.778%/30.512%`，绑定 `mini-swe-agent`、工具、任务集、仓库环境和 verifier，不能与 AA 或 Anthropic 结果拼成裸模型分数。
- System Card 快照为 `/tmp/sonnet5-system-card-20260921.html`，10,786,341 bytes，SHA-256 `33573adb9f1871b903f79b77d7755a44cc20bb913ef48ac9000ddb090e41f7ed`；标准库文本提取 SHA-256 为 `3daed308eaedefa8ea61fe3bcfcc99a1206d03dcb02148dc63ba8ffdea9c47f6`。深读确认：训练数据只公开为公开互联网、公开/私有和合成数据的专有混合数据，并提到去重、分类、ClaudeBot 爬取和 post-training/fine-tuning；没有完整训练 recipe、参数规模或内部 adaptive-thinking 机制。
- 安全和 Agentic 证据已补录：Sonnet 5 不跨 automated AI R&D threshold，Autonomy threat model 1 适用且 stealth rate 接近零，CB-2 未跨越；没有网络安全专项训练，覆盖 ExploitBench、OSS-Fuzz、CyberGym、Firefox 147，默认 mitigations 下部分结果为 0。Claude Code 恶意请求拒答率 `92.37%`，computer-use 恶意任务拒答率 `84.68%`；Gray Swan IPI 覆盖 28 个场景、去重后 1,130 个攻击。它们均是带 safeguards/harness 的发布方行为证据。
- 发布方 benchmark 另记录 SWE-bench Verified `85.2%`、SWE-bench Pro `63.2%`、Multilingual SWE-bench `78.3%`、Terminal-Bench 2.1 `80.4%`、BrowseComp `84.7%`、OSWorld-Verified `81.2%`、GDPval-AA v2 Elo `1618`、Toolathlon Pass@1 `54.3%`、AA-Briefcase Elo `1393`；配置通常为 adaptive + max、约 5 trials，BrowseComp 使用 10M token limit 并在约 200K 触发 compaction。所有数字保留发布方/system-card harness 标签。
- 已同步 `claude-sonnet-5-source-notes.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan.md`、本文件、第四册/第十六册/第十七册/第二十册/第二十四册相关章节，以及 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`PROJECTS.md`、`GLOSSARY_EN_ZH.md`、`KNOWLEDGE_GRAPH.md`。当前状态为**资料级闭环**；参数、内部架构、完整训练/后训练 recipe、独立 benchmark 复现、目标硬件 profiling 和线上 serving acceptance 仍待核验。下一步从两个排行榜剩余重点 canonical 模型中选择或复验锚点。

## 2026-09-22 Grok 4.7 执行记录与下一步

本轮通过三条代理重新刷新两个排行榜：Artificial Analysis 中文首页为 `1,785,528` bytes、SHA-256 `8d154f41216b952df387e689dbe9a5a254188b08fda9eba1791c4da9fabb1be8`，DataCurve 为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，三条线路内容一致。重点厂商新 canonical 条目为 `grok-4-7`，AA 详情为 `Grok 4.7 (xhigh)`；不从官方目录或论文另发现模型。

Grok 4.7 研究已覆盖 xAI 模型页/发布页、reasoning encrypted state、compaction opaque item、function calling、structured outputs、Remote MCP `allowed_tools`、发布方训练/安全披露和 arXiv 精确标题负检索。xAI 的 benchmark 数字和 AA/DataCurve 字段继续分账；没有精确 DataCurve 行，不迁移 Grok 4.6 结果。DeepSeek V3.2 的 0925、reasoning、non-reasoning、Speciale 条目已归并为历史 revision/configuration/checkpoint，不新增模型。

下一步：把 Grok 4.7 的 reasoning/compaction、工具权限和发布方 benchmark 证据补入书系与题库，完成全仓库链接/Markdown/Python 校验；之后继续只从 Artificial Analysis 与 DataCurve 的八家重点厂商 canonical 条目选择下一锚点。未公开参数、完整训练配方、生产 kernel、硬件 profiling、线上 acceptance 和独立复现继续列为待核验，不将 goal 标记为 complete。

当前仍是 **AA 单榜资料级闭环**。下一步不新增重复 Transformer 章节，优先补具体 revision/依赖/硬件下的 kernel 正确性、召回/误差、完整权重加载、EP/DP/TP serving、tool acceptance 和独立 benchmark；若这些门禁暂时无法取得，则从两榜八家重点厂商的既有 canonical 条目选择下一锚点。

## 2026-09-22 Grok 4.7 当前时点增量复验

本轮没有发现新的重点厂商 canonical 模型，两个排行榜仍是当前唯一模型入口。Artificial Analysis 中文首页刷新为 `1,762,420` bytes / SHA-256 `de45e583216f236e0eb6c0b69b65b251394eea04137368a056c43e072a97a6f6`，DataCurve 为 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；已存在的 `grok-4-7` 仍无精确 DataCurve 行。

经 `10.24.27.134:7890` 重新取得 xAI Reasoning、Context Compaction 和 Remote MCP 页面：Reasoning `522,822` bytes / `ce3cd1997308094472d2cce8017eee510cf3441c7be2711029870c9d18d1cc87`；Compaction `521,486` bytes / `2a58eb48a8ed56d5f747e20fdee8dc9a50cc5998507f3ac0f598afd3d0f48eb3`；MCP `475,038` bytes / `459b83b7522eaa9139677f9da9760593e56072d52fd63a1883ba6a370a13c705`。模型页/发布页刷新哈希分别为 `add926110deb683b8c90a126340a1f1fa4fc44a6aaa698ea5d3a7d68f537bdd5` 与 `88d0ce52f3c9edfe273d955a1b4bd2547202f5f6a226b14070729703d818f53b`；正文未观察到模型 revision 变化。

新增面试知识已同步：Grok 4.7 的 Responses 每次返回 encrypted reasoning，服务端工具输出也需保留；summary 流与 opaque encrypted state 分账；`store` 决定 `previous_response_id` 相关存储行为；MCP 的 `allowed_tools`/`headers` 与 xAI SDK 的 `allowed_tool_names`/`extra_headers` 不同，Streaming HTTP/SSE 是当前 transport，`require_approval`/`connector_id` 不可依赖。当前状态仍为 **AA 单榜内容专题闭环**，不新增 Transformer 正式章节；参数、内部架构、完整训练 recipe、完整权重、生产 kernel、目标硬件 profiling、线上 acceptance 和独立复现继续待核验。已完成本轮底稿/书系/题库/索引同步，goal 保持 active。

## 2026-09-22 Qwen3-Omni 执行记录与下一步

本轮没有从 Qwen 官方目录、Hugging Face 或论文列表另发现模型；`Qwen3-Omni 30B A3B` 只因已出现在 Artificial Analysis 才进入本轮锚点。instruct 详情三代理均 HTTP 200、`3,833,501` bytes、SHA-256 `e62d2c5dac6af3b1b08efa94b58e321df8ac16a813de05d95ad7fabe5eedb316`；reasoning 快照为 `3,845,331` bytes、SHA-256 `22f186bdd723dda3647aca48c7cbbe2d91af1832f08217620c3dad39bff5c748`。DataCurve 当前没有精确 `mini_swe_agent_qwen3_omni_*` 行，不迁移其他 Qwen 的 Agent 分数。

已完成官方 GitHub、HF Instruct/Thinking 配置、Qwen 博客、阿里云 Qwen-Omni 文档、arXiv `2509.17765` 论文与源码核验；研究笔记已记录 Thinker/Talker、AuT（约 2,000 万小时监督音频、12.5 Hz/约 80 ms）、Qwen3-VL/SigLIP2-So400m 视觉 encoder、TM-RoPE `24/20/20`、多码本 Talker、MTP、Code2Wav、三阶段预训练、Thinker/Talker 后训练和异步 chunked prefill。报告的 audio/video first packet `234/547 ms` 继续标注为报告口径，不是本机实测。

已新增第二十一册第 91 章 [`Qwen3-Omni：Thinker-Talker、AuT 与流式多模态`](book-21-transformer-architecture-evolution/chapters/91-qwen3-omni-thinker-talker-aut与流式多模态.md)，并同步 `model-inventory.md`、`source-index.md`、`inventory-interpretation.md`。下一步完成第四册百科、reasoning/Agent/serving 交叉段落、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、`BOOK_SERIES.md`、`ROADMAP.md` 和第二十一册目录/时间线同步，再做链接、Markdown 围栏和章节代码验证。

当前状态：**AA 单榜内容专题闭环**。完整 Thinker/Talker 参数账本、专家与跨模块接口、生产 kernel、目标硬件流式 profiling、音频首包/多码本 acceptance、完整训练 recipe、线上工具验收和独立 benchmark 仍待核验；不将 goal 标记为 complete。

## 2026-09-22 K2 Horizon 3.7B 当前活动锚点

本轮把活动锚点从 `Qwen3-Omni 30B A3B` 切换为已经出现在 Artificial Analysis 的 `K2 Horizon 3.7B`。三条代理取得逐字节一致的 AA 详情快照：`3,756,408` bytes、SHA-256 `72b4f94c55add582b0399333552e92b7b4aebd2f493c26830c00e09e58afc39d`；页面字段为 `releaseDate=2026-09-03`、reasoning、open weights、3.7B、524,288 context、Apache 2.0、AA Index `15.6103951330976`。DataCurve 快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，没有精确 `mini_swe_agent_k2_horizon_3_7b_*` 或 K2 Horizon 行，因此不迁移其他 K2/模型的 Agent 分数。

已核验 IFM 当前 main revision `6360f705b2e57d542959e6a2e67ebeb95dae0373`（`lastModified=2026-09-21T01:40:06Z`）及固定 artifact：`K2HorizonForCausalLM`、36 层、hidden 2560、32Q/8KV GQA、head dim 128、vocab 250,624、524,288 position、default RoPE、`num_experts=0`、`mova_num_experts=0`、BF16、36 shards/327 tensors/`10,116,510,720` bytes。3.7B 是 dense 对照，不继承 36B/A4B 的 MoVA value top-4、FFN top-8 或 shared expert；两者共享 GQA/512K 目标，但不能按参数比例推导 serving 成本。

训练资料补齐了 22.9T/8K pretraining、32K/128K/512K 分阶段 midtraining、512K SFT Phase 1/2、Math/Code/STEM-Code RL 分支和 ISO/RAM merge；中间 checkpoint 可用于阶段能力对照。`/tmp/k2-migration-fixed-20260922.out` 固定为 `k2_aurora -> k2_horizon`、`K2HorizonForCausalLM`、copy、`weights_reencoded=false`、BF16、36 shards/327 tensors，说明 artifact 迁移而非重新训练。

证据冲突已写明：当前 config/migration/index 支持 `K2HorizonForCausalLM` + BF16；旧 `APPENDIX.md` 的 `XllmForCausalLM`/FP32 和 `3.78B core / 5.06B including embeddings` 只能作为旧文档/参数口径冲突记录。vLLM recipe（2026-09-02）给出 5.06B dense、512K、H200、`k2_horizon` reasoning/tool parser；SGLang PR #37654 给出 H200 TP1/BF16/FlashAttention-3 和发布方 TTFT/TPOT/GSM8K 结果，但固定 revision `c177771836a4c460743c00002c22483f6f18d1eb` 当前 HF 404，不能写成模型不存在或本机复现。

已新增 [`k2-horizon-3.7b-source-notes.md`](research/model-update-2026-09/k2-horizon-3.7b-source-notes.md)，并把 3.7B dense 对照、长上下文训练、RL merge、migration、parser/revision 和字段冲突扩展到第二十一册第 82 章；模型清单、来源索引和榜单解释已同步。该段是历史锚点记录，当前活动锚点以文件顶部和最新记录为准。

## 2026-09-22 Qwen3-VL-235B-A22B 当前活动锚点

本轮把活动锚点从 `K2 Horizon 3.7B` 切换为已经出现在 Artificial Analysis 的 `Qwen3-VL-235B-A22B`。三条代理取得一致的 AA 中文首页、instruct 详情和 reasoning 详情：首页 `1,798,627` bytes / SHA-256 `4254f2dad0222fd7147ae50047e3bfe0b382ec1aed1c788c94ef9e8f683242fe`；instruct `3,832,139` bytes / `a58d3922d8e872c76684314e9de21eee78e431ee0d527d10c65cadc9f7755efa`；reasoning `3,849,165` bytes / `28f97633dfbeb145e696b9f87a01ba891755d80bfaaf7e7a1db4d8d7ecaa59b3`。DataCurve 当前没有精确 `mini_swe_agent_qwen3_vl_*` 行，因此不迁移其他 Qwen 的 Agent 结果。

已核验 Qwen3-VL 官方 GitHub、HF Instruct/Thinking 固定 revision/config、Qwen3-VL Technical Report `arXiv:2511.21631` 和 Transformers raw main。Instruct/Thinking/Reasoning 归并为同一基础模型的不同配置或 artifact；AA 的 `235B total / 22B active`、`262,144` context、速度/价格是第三方目录或 provider 字段。核心技术主线为 SigLIP2 vision encoder + 两层 MLP merger + Qwen3 MoE decoder、Interleaved-MRoPE 三轴频率交错、DeepStack `[8,16,24]` 中间视觉层残差、Video Timestamp、S0-S3 长上下文 curriculum、SAPO/General RL、Thinking with Images 和 tool-call reward。

已新增 [`qwen3-vl-source-notes.md`](research/model-update-2026-09/qwen3-vl-source-notes.md) 和第二十一册第 92 章 [`Qwen3-VL：Interleaved-MRoPE、DeepStack 与视频时间戳`](book-21-transformer-architecture-evolution/chapters/92-qwen3-vl-interleaved-mrope-deepstack与视频时间戳.md)，并同步模型清单、来源索引、榜单解释、题库、练习、术语、项目、论文、知识图谱、BOOK_SERIES、ROADMAP 和第二十一册目录/时间线。当前状态：**AA 单榜内容专题闭环**；完整权重加载、生产 kernel、目标硬件 profiling、端到端多模态 serving、GUI/tool acceptance、完整训练 recipe 和独立 benchmark 仍待核验。

## 2026-09-22 Opus 5 与 Gemini 3.5 Flash-Lite 当前时点复验

本轮重新处理此前可能受工作时段代理中断影响的联网任务。Opus 5 的 AA 详情、中文首页、DataCurve 和 Anthropic 发布页/System Card 已取得新鲜快照；9 月 22 日 AA 的速度/TTFC 与 9 月 21 日测量不同，但没有新增 canonical 模型或 revision 证据，因此历史值与当前值分开保存。Opus 5 仍是**资料级闭环**，不新增重复 Transformer 正式章节。

Gemini 3.5 Flash-Lite 的 AA 详情、DataCurve、Google API 模型页、Thinking 和 Video understanding 页面也已重新取得。AA 仍为单榜锚点，DataCurve 没有精确 `mini_swe_agent_gemini_3_5_flash_lite_*` 行；当前主线是 `minimal` thinking、请求级 effort、agentic video 的按需时间轴读取和 `processing_call/result` 事件审计。Model Card 明确基于 Gemini 3.1 Flash-Lite，不把前代架构/训练资料写成 Lite 的新发明；当前为**资料级闭环（AA 单榜）**，不新增独立 Transformer 正式章节。

下一步先完成模型清单、来源索引、榜单解释、进度和本计划的同步，然后执行 Markdown 围栏、相对链接、研究代码编译与证据边界检查；检查通过后，再从两个排行榜的八家重点厂商 canonical 条目选择下一个活动锚点。goal 保持 active。

## 2026-09-22 GLM-5.1 当前时点复验

本轮把活动锚点切换为已由 Artificial Analysis 确认的 `GLM-5.1`，没有从 Z.ai 文档、官方博客或 arXiv 另发现模型。三条代理取得一致的 AA 详情页；当前快照为 `3,971,543` bytes、SHA-256 `c806874d4eb235658b04ba6f527e1ca1db30b2c5e149e8690cd428fd5d2614c9`，中文首页为 `1,798,627` bytes、SHA-256 `4254f2dad0222fd7147ae50047e3bfe0b382ec1aed1c788c94ef9e8f683242fe`。页面仍标记 release `April 2026`、200K context；第三方当前测量为 Intelligence Index `26.0585912980095`、median output speed `37.1922485381327 tokens/s` 和 cost per Intelligence Index task `0.9217822401466147`。

DataCurve 当前快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，没有精确 `mini_swe_agent_glm_5_1_*` 行，因此不记录或迁移 GLM-5/5.2/5.3/5.3-Flash 的 Agent 结果。Z.ai 当前文档、HF config、博客正文资源和既有 API 资料支持 long-horizon Agent、multi-turn SFT/RL/process-quality evaluation、DSA/MoE 配置、thinking/tool/cache 协议；完整训练 recipe、indexer loss/recall、过程质量 verifier、生产 kernel、硬件 profiling、线上 acceptance、独立报告和复现仍待核验。

当前状态仍为 **AA 单榜资料级闭环**。本轮只更新榜单/provider 快照与证据边界，不新增第二十一册 Transformer 正式章节；下一步继续从两个排行榜八家重点厂商的 canonical 条目选择锚点，goal 保持 active。

## 2026-09-22 GPT-5.6 Luna 当前活动锚点与 Agent 运行时补证

本轮把活动锚点从 `GLM-5.1` 切换为已经同时出现在 Artificial Analysis 与 DataCurve DeepSWE 的 `GPT-5.6 Luna`。不从 OpenAI 官方模型目录另发现模型；官方资料只用于扩展已确认的榜单锚点。

- Artificial Analysis 当前详情为 `3,861,318` bytes / SHA-256 `1b425d8f2a93f8418702ac7acbb62b26850b98fe20680f82205bed5fce38fed0`；release `2026-07-09`、Intelligence Index `37.3244239690841`、`158.728370482714 tokens/s`、1M context、约 `$0.20/$1.20` 和 cost per Intelligence Index task `0.17829726152289094` 都是第三方 max/provider 字段。DataCurve 当前为 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，精确 `mini_swe_agent_gpt_5_6_luna_max` 行为 301/448、Pass@1 `67.1875%`、Pass@4 `90.26548672566371%`、约 `$0.6056`、约 73.4K 输出 token、约 101.68 steps；这些是 `mini-swe-agent` 系统结果。
- 7890 代理本轮取得 OpenAI 官方页面：Luna 3,744 bytes / `1f425d8f...fed0`，Reasoning 70,253 bytes / `91604d...6719a`，Agents 5,432 bytes / `df4f61...43a`，Tools 33,282 bytes / `4722fa...a341`，Tool Search 39,288 bytes / `9d6c38...1fd8`，Prompt Caching 47,097 bytes / `c70d85...b2d1`，Compaction 14,272 bytes / `73fd2f...0dd`；完整哈希与 URL 见 [`gpt-5.6-source-notes.md`](research/model-update-2026-09/gpt-5.6-source-notes.md)。
- 本轮把 Agent 运行时分为三层：Agents API 托管 Codex harness 和 session/turn/item 状态；Agents SDK 把 loop、工具、handoff、部署、存储、审批和 runtime 控制交给应用；Responses API 让应用直接管理 response item、历史和工具循环。Tool search 分为 hosted/client 两条协议，分别由服务端加载或由应用回传 `tool_search_call`/`tool_search_output`；加载工具追加到上下文末端以保护前缀缓存，但不改变权限边界。
- 本轮验收要求新增缓存与压缩联合账本：model、tools、parallel tool calls、structured output、reasoning effort、verbosity 和 context management 都可能影响缓存前缀；compaction 会替换较早上下文并造成首次 cache miss，tool search 会减小初始 schema，但会引入工具发现和版本状态。正式章节优先同步第二十册 Responses/Agent Runtime/MCP 与第十七册 Code Agent，题库增加 hosted/client tool search 和三层运行时所有权问题。

当前状态：**双榜资料级闭环**。参数、内部架构、完整训练/后训练 recipe、system card、生产 kernel、目标硬件 profiling 和独立 GPT-5.6 报告仍待核验；goal 保持 active。

## 2026-09-22 Qwen3.7 Plus 当前活动锚点收口

本轮已完成 `Qwen3.7 Plus` 的 AA 单榜资料级闭环：Artificial Analysis 精确条目、DataCurve 精确行缺失、Alibaba Cloud 官方模型文档、推荐模型页和 OpenAI-compatible API 文档均已留存快照与哈希；研究笔记、模型清单、来源索引、解释规则、进度、面试题、练习、术语、项目、论文、知识图谱、书系路线和第十五/十七/二十/二十四册已有章节均已同步。后续不新增 Qwen3.7 Plus Transformer 专属章节，也不从相邻 Qwen 版本迁移架构或 Agent 结果；下一轮继续从两个排行榜的八家重点厂商 canonical 条目选择活动锚点，goal 保持 active。

## 2026-09-22 DeepSeek V4.1-Flash vLLM runtime 证据分层规则

本轮继续沿两个排行榜中已经确认的 DeepSeek V4.1-Flash，不从 vLLM 仓库另发现模型。以后记录模型周边 serving 技术时，新增以下证据分层：

1. vLLM `main` registry/package 只能证明 upstream source entry 存在；必须记录 URL、文件路径、抓取线路、bytes 和 SHA-256。
2. stable registry/wheel 是 release surface 证据，不能用 `main` 类名替代。当前固定结论是：vLLM `main` 已登记 `DeepseekV41ForCausalLM`/`DSparkV41DraftModel`，而 v0.29.0 stable registry 的固定快照没有这两个 V4.1 专用类名。
3. 完整权重加载、GPU/ROCm kernel、真实 candidate/index recall、FP4 质量、DSpark acceptance/rollback、EPD 调度、tool acceptance、硬件 profiling 和生产 SLO 必须单独验收，不能由源码入口推导。
4. V4.1 vision wrapper 显式跳过 `mtp.*`，因此不把“原生视觉”和“DSpark/MTP”自动写成已合并能力；只有拿到视觉 token 的 draft/target verify、cache 一致性和目标硬件证据后，才可升级该组合结论。

本轮已将上述规则和证据同步到 DeepSeek 研究笔记、模型清单、来源索引、榜单解释、`progress_v2.md` 和第二十一册第 81 章。下一步继续补具体 release/hardware/harness 绑定的实测门禁，不新增重复 Transformer 章节；goal 保持 active。

## 2026-09-22 DeepSeek V4.1-Flash SGLang runtime 对照

沿两个排行榜中已经确认的 DeepSeek V4.1-Flash 继续补 serving 证据，不从 SGLang 仓库另发现模型。SGLang `main` 已公开 V4.1 vision、2D-RoPE ViT/Aligner、TP/EP/DP（不支持 CP/PP/MoE A2A）、V4/V4.1 FP8 block-size 分支、MXFP8/FP8 prefill autotune、FlashInfer、unified KV、DSV4 sparse indexer/cache，以及 DSpark 的 Markov/confidence head、`mtp.*` 权重映射和 draft stage `vision_n_layers=0`。这些内容用于面试中的“视觉 target 与 draft 是否同一条路径”“block size 是否随版本变化”“main implementation 与 release surface 如何区分”。

SGLang `v0.5.20` stable（2026-09-18 发布）对照源码没有 `deepseek_v41_vit.py`，通用 V4/DSpark 文件中 `deepseek_v41`/`dsv41` 计数为 0；因此计划中的证据分层更新为：HF reference -> vLLM main -> SGLang main -> stable release/wheel -> 完整权重和目标硬件验收。下一步仍只做可复现的 revision、硬件、harness 和 acceptance 门禁，不把 main 代码写成 stable 或生产结论，也不新增重复 Transformer 正式章节；goal 保持 active。

## 2026-09-22 Kimi K3 SGLang stable/main runtime 对照

本轮继续沿两个排行榜已经确认的 `Kimi K3` 推进，没有从 SGLang 源码目录另发现模型。Artificial Analysis 与 DataCurve 的 K3 锚点、HF revision/config/index、FlashKDA 和 vLLM stable/source evidence 继续沿用已有记录；新增的是 SGLang K3 文本/视觉源码的 stable/main 对照。

- SGLang `v0.5.20` tag commit 为 `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`，K3 text 文件为 168,114 bytes、blob `b0ede48c88264d518351a66abf623f1bcf8a730e`、SHA-256 `7a3ef867394a2fd52b3a71a979c053b51e9f2310c35cd4c12ef3aa8e7be172e5`；`main` text 文件为 171,101 bytes、blob `383a6f47812bccd1cb91b76814cd0730ff945dd7`、SHA-256 `54069412bf3766e974c168985b4e848ecfc29aa4f02dbe27f8e037345faefc0e`。
- 两版本 `kimi_k3_vl.py` 逐字节一致：32,790 bytes、blob `c423027994645e4a839ec47bd26bcc1b2a7cb012`、SHA-256 `2924c38f652a6ebff2ef79c49f2f336ba18723ea4b854d3ac14d95292988acdb`；本轮新增差异集中在文本 runtime，不把视觉文件写成 main 独有能力。
- `main` 代码证据包括 LatentMoE latent down/up projection、MegaMoE/DeepEP/Mooncake/Ascend-FuseEP/MoRI A2A、DP/SP token shard、shared-expert TP group 与 reduce-scatter、SBO/NPU dual-stream、ModelSlim fused QKVG/packed expert loader、KDA fused decode capability gate 和普通 fallback。`v0.5.20` 已有 K3 基础 text/vision、部分 EP/SBO/KDA 路径，但实现差异不能升级为 stable wheel 或生产 SLO。
- 当前 K3 状态更新为 **内容专题闭环 + HF/vLLM + SGLang stable/main source evidence**。完整权重加载、SGLang 依赖/目标硬件、MLA/KDA 双状态恢复、视觉数值正确性、目标 profile、tool/schema/retry/idempotency/verifier acceptance 和独立 benchmark 仍待核验。GitHub commit history 因本轮三条代理当前连接失败未固定，mutable `main` 不写成 release。

后续门禁按顺序执行：安装/导入目标 SGLang 版本、用 pinned K3 revision 做最小 full-weight load、分别测试 KDA/MLA state recovery 与 prefill/decode/verify、在目标 GPU/NPU/ROCm profile EP/A2A/SBO/视觉路径，再做 tool-call schema、权限、幂等和 verifier acceptance。SGLang commit history 已在网络恢复后固定；若环境仍不具备完整权重或硬件，则保持 source-evidence 状态，不伪造运行结果；goal 保持 active。

## 2026-09-22 两榜复验与 OpenAI gpt-oss provider 兼容性补证

本轮重新抓取两个唯一排行榜，三条代理逐字节一致：Artificial Analysis 中文首页 `1,762,420` bytes / SHA-256 `de45e583216f236e0eb6c0b69b65b251394eea04137368a056c43e072a97a6f6`；DataCurve DeepSWE `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。按八家重点厂商的 canonical slug 归一化后没有新的重点模型；当前 AA 中新增/保留的 K2 Horizon 0.9B、7B、375B 等其他厂商条目不进入本轮研究队列，也不把 K2-Horizon-7B-Uno 当作新的排行榜锚点。

下一锚点切换为已有 AA 锚点 `gpt-oss-120b`/`gpt-oss-20b` 的待核验周边。官方 gpt-oss README 当前 `24,453` bytes / SHA-256 `578ad0f82c1d823229f9bf2b52b3f1c55ce82772ac57f634f0ca4eb46a6370aa`，与 9 月 18 日一致；最新 commit history `44,611` bytes / `420dcfab2dad69aa386b5207ac3aa35f627151e292c63caced716d46c59e288c` 的最新核心仓库提交为 `7b583341fe16`（2026-07-24），是 You.com backend API key 修复，不构成模型/核心推理新版本。

OpenAI Cookbook 新鲜补证了 [实现兼容性验证](https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations)（`336,633` bytes / `4f293d2bd51666966a8d3657893eb1d1093b4b3230a405c9178cbc7302d664b8`）和 [raw CoT 处理](https://developers.openai.com/cookbook/articles/gpt-oss/handle-raw-cot)（`338,262` bytes / `7330273ff59d6e1a344abe3f435ae421a20ebd775bda9122d22ff606f0782072`）：

- Responses 用 `reasoning.content[].reasoning_text` 及 `response.reasoning_text.delta/done` 承载 raw CoT；下一轮需按 item/index/turn lineage 回放，raw CoT 默认不能展示给终端用户。
- Chat Completions provider 可采用 `reasoning` 字段和 delta 约定，但它是 gpt-oss 兼容层协议，不写成 OpenAI 托管模型的通用字段。
- 官方 `compatibility-test` 将 Harmony/API shape、tool call/result、streaming 和 invalid requests 作为 smoke test；AIME（每题 16 次）、GPQA（每题 8 次）、HealthBench（每题 1 次）再作为质量 eval，二者不互相替代。
- 0 invalid requests 且 `pass@k`/`pass^k` 均超过 90% 只是“很可能正确”的信号，仍不能证明 MXFP4 kernel、MoE dispatch、目标硬件 profile、线上接受率或生产 artifact verifier。

已同步 gpt-oss 研究笔记、来源索引、模型盘点、榜单解释、第二十一册第 87 章、第二十四册 serving 章节、题库、练习、项目、术语和知识图谱。当前状态更新为**内容专题闭环 + 官方 provider 兼容性/raw CoT protocol evidence**；完整训练/后训练 recipe、kernel/误差、目标硬件、独立 Agent 评测和线上 acceptance 仍待核验，goal 保持 active。

## 2026-09-22 Kimi K3 SGLang commit history 补证

本轮重新取得 SGLang kimi_k3.py 的最近 10 个提交。GitHub commit history API 经 10.237.126.170:1234 返回 51,251 bytes，SHA-256 为 2f7d8402aa104fe9a5b7839e85c9eb56060e98f7aeb4aecb787005f775351e04，最新记录时间为 2026-09-22T06:35:12Z。当前 main 文本源码复抓后仍为 171,101 bytes、blob 383a6f47812bccd1cb91b76814cd0730ff945dd7、SHA-256 为 54069412bf3766e974c168985b4e848ecfc29aa4f02dbe27f8e037345faefc0e，未形成新的源码版本。

已核验的 upstream 变化包括：8ac19cc 的 deferred KDA gate projection；c4d3770 的 CUDA graph stream fork/join 修复；c2c3629 的 O(1) expert weight lookup、ModelSlim fused QKVG 和 NPU packed mapping；72d5c5b 的 FP32 routing weight finalize；f4c2563 的 PP prefill + DCP decode + PD disaggregation + DSpark；2d0e94e 的 shared-expert process-group identity；8ac39c6 的 Ascend A5/NPU 路径；cb32dbc 的 ROCm KDA input projection fusion。

这些提交把面试重点推进到状态 ownership、capture/replay、checkpoint loader、process group 和硬件分支，但仍属于 mutable main source evidence。它们不能证明 stable wheel、完整权重、双状态恢复、目标硬件 profile、视觉数值、DSpark acceptance 或线上 tool/verifier 通过。三条代理本轮访问百度均 HTTP 200；GitHub commits API 的 1234 成功、7890 返回 403、8098 出现 TLS/代理协议错误，后两者只记录为线路状态。

## 2026-09-22：Grok 4.7 活动锚点闭环与 DeepSeek V3.2 归并

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree 修改，goal 保持 active。模型发现仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为八家范围。
- 三条代理取得一致的 Artificial Analysis 中文首页：`1,785,528` bytes，SHA-256 `8d154f41216b952df387e689dbe9a5a254188b08fda9eba1791c4da9fabb1be8`；DataCurve：`268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。新 canonical 条目为 `grok-4-7`，详情 `Grok 4.7 (xhigh)`，AA 指数 `46.4465506302286`、约 `38.7732 tokens/s`、500K、Intelligence task cost 约 `$3.7383`。
- DataCurve 没有精确 `mini_swe_agent_grok_4_7_*` 行，因此不迁移 Grok 4.6 结果。xAI 模型页快照 `376,912` bytes / `c3ddd6b44b97fb4b527096ca69e4d9eacdca99e0e4e44427c9da5b81a615181e`，发布页 `288,007` bytes / `af8eda968823b546480818ba27f6edf7f9787b5126c51a9f5ac4568e23da406f`；已补录 500K、effort、不可关闭 reasoning、Responses `encrypted_content`、compaction opaque item、function/structured/MCP 工具协议。
- xAI 发布方披露更大 base、更长 RL run、困难长任务混合、自验证、长上下文管理和 safeguard stack，并给出 CursorBench 4.0 `46.3%`、DeepSWE v1.1 `71.0%`、EEBench `64.0%`、AA Briefcase `1657`、Terminal-Bench `38.0%`、Harvey `19.6%`、HealthBench Professional `56.7%`、GDPval `1695` 等数字；均保留 benchmark/harness/effort 口径，不与 AA/DataCurve 合并。
- arXiv 精确标题检索无 Grok 4.7 结果，快照 SHA-256 `8f8c4bb0f6c8a0346e28a56864a4aa4d630bdde9c536ee1320be29f84e6e9da6`；不补写参数、架构、完整训练 recipe、生产 kernel 或独立 benchmark。Grok 4.7 当前状态为**AA 单榜内容专题闭环**。
- DeepSeek V3.2 的 `0925`、reasoning/non-reasoning、Speciale 对象已核验为 deprecated/redirect 的历史 revision、配置或同家族专项 checkpoint；DataCurve 没有精确 V3.2 行，不新增 DeepSeek 模型，不迁移 V3.1/V4 的 Agent 分数。
- 已同步 Grok 4.7 研究笔记、模型清单、来源索引、榜单解释、`plan.md`、本文件和书系/全局配套；下一步完成全仓库 Markdown、相对链接、Python 编译和证据审计，然后继续从两个排行榜的八家重点厂商 canonical 条目选择锚点。目标保持 active，不标记 complete。

## 2026-09-22：Qwen3-Omni 30B A3B 专题落地

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree 修改。模型发现仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为八家范围。
- 本轮沿 Artificial Analysis 已发现的 `Qwen3 Omni 30B A3B` 推进；instruct 详情三代理均 HTTP 200、`3,833,501` bytes、SHA-256 `e62d2c5dac6af3b1b08efa94b58e321df8ac16a813de05d95ad7fabe5eedb316`，reasoning 快照为 `3,845,331` bytes、SHA-256 `22f186bdd723dda3647aca48c7cbbe2d91af1832f08217620c3dad39bff5c748`。DataCurve 没有精确 `mini_swe_agent_qwen3_omni_*` 行，不迁移其他 Qwen 的 Agent 结果。
- 已核验 Qwen3-Omni 官方 GitHub、HF Instruct/Thinking 配置、Qwen 博客、阿里云 Qwen-Omni 文档、arXiv `2509.17765` 技术报告及源码；补齐 Thinker-Talker、AuT（约 2,000 万小时监督音频、8 倍 Conv2D、12.5 Hz）、Qwen3-VL/SigLIP2-So400m 视觉 encoder、TM-RoPE `24/20/20`、多码本 Talker/MTP、Code2Wav、三阶段预训练、Thinker/Talker 后训练和异步 chunked prefill。
- 已新增研究笔记 [`qwen3-omni-source-notes.md`](research/model-update-2026-09/qwen3-omni-source-notes.md) 和第二十一册第 91 章 [`Qwen3-Omni：Thinker-Talker、AuT 与流式多模态`](book-21-transformer-architecture-evolution/chapters/91-qwen3-omni-thinker-talker-aut与流式多模态.md)，并同步模型清单、来源索引、榜单解释、计划文件。Instruct、Thinking、Captioner 作为同家族 artifact 归并，不新增三个基础模型。
- 报告的 audio/video first packet `234/547 ms` 继续标注为论文/发布方口径，不是本机实测；官方 README 当前说明 vLLM 主要支持 Thinker，Instruct 音频输出仍在实现推进中。完整参数账本、生产 kernel、目标硬件流式 profiling、音频 acceptance、完整 recipe、线上工具验收和独立 benchmark 仍待核验。
- 当前状态：**AA 单榜内容专题闭环**；下一步同步第四册百科、reasoning/Agent/serving 交叉段落、全局论文/题库/练习/术语/项目/知识图谱、BOOK_SERIES/ROADMAP 和第二十一册目录/时间线，然后执行 Markdown、链接和章节代码验证。goal 保持 active。

## 2026-09-22：K2 Horizon 3.7B dense 对照专题收口

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree 修改。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为八家范围。
- 本轮继续处理已由 Artificial Analysis 发现的 `K2 Horizon 3.7B`，没有从 IFM 官方目录、HF 关联模型或 Uno 论文另发现模型。AA 三条代理详情页逐字节一致：`3,756,408` bytes、SHA-256 `72b4f94c55add582b0399333552e92b7b4aebd2f493c26830c00e09e58afc39d`；字段为 `releaseDate=2026-09-03`、reasoning、open weights、3.7B、524,288 context、Apache 2.0、AA Index `15.6103951330976`。
- DataCurve 当前快照 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870` 没有精确 `mini_swe_agent_k2_horizon_3_7b_*` 或 K2 Horizon 行；没有迁移 K2 36B、K2.7 Code、K3 或其他模型的 Agent 分数。
- 已固定 IFM main revision `6360f705b2e57d542959e6a2e67ebeb95dae0373`（`lastModified=2026-09-21T01:40:06Z`），当前 artifact 为 `K2HorizonForCausalLM`、36 层、2560 hidden、32Q/8KV GQA、head dim 128、vocab 250,624、524,288 position、default RoPE、`num_experts=0`、`mova_num_experts=0`、BF16；权重 index 为 36 shards/327 tensors/`10,116,510,720` bytes。3.7B 是 dense 对照，不能继承 36B MoVA value top-4、FFN top-8 或 shared expert 的 forward-time 路由。
- 训练证据已记录 22.9T/8K pretraining、32K/128K/512K 分阶段 midtraining、512K SFT Phase 1/2、Math/Code/STEM-Code RL 分支和 self-attention ISO merge/other weights RAM；中间 checkpoint 可用于阶段能力比较。`/tmp/k2-migration-fixed-20260922.out` 固定为 `k2_aurora -> k2_horizon`、copy、`weights_reencoded=false`、BF16、36 shards/327 tensors，归类为 artifact 迁移证据，不是重新训练证据。
- 已记录 vLLM recipe（5.06B dense、512K、H200、`k2_horizon` reasoning/tool parser）和 SGLang PR #37654（TP1/BF16/FlashAttention-3/H200、发布方 TTFT/TPOT/GSM8K）；SGLang 文档引用的 revision `c177771836a4c460743c00002c22483f6f18d1eb` 当前 HF raw/API 404，只记为旧部署 revision 不可解析。发布方 benchmark 不写成本机实测或统一裸模型分数。
- 当前文档冲突已分账：current config/migration/index 支持 `K2HorizonForCausalLM` + BF16；旧 `APPENDIX.md` 的 `XllmForCausalLM`/FP32 和 `3.78B core / 5.06B including embeddings` 标为旧 revision/残留字段。已新增 [`k2-horizon-3.7b-source-notes.md`](research/model-update-2026-09/k2-horizon-3.7b-source-notes.md)，扩展第二十一册第 82 章，并同步 model inventory、source index、inventory interpretation、`plan.md`。
- 当前状态：**AA 单榜资料级闭环**。仍待核验完整训练 recipe、生产 kernel、目标硬件 profiling、线上 tool acceptance、独立 benchmark 和精确 DataCurve Agent 行；下一步完成本轮书系配套文件同步和全仓库质量检查，再从两榜八家重点厂商的既有 canonical 条目选择下一锚点。该段是 K2 历史锚点记录，当前活动锚点以文件顶部和最新记录为准；goal 保持 active。

## 2026-09-22：Qwen3-VL-235B-A22B 专题落地

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree 修改。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为八家范围。
- 本轮沿 Artificial Analysis 已发现的 `Qwen3-VL-235B-A22B` 推进；instruct/reasoning 两个详情页三代理逐字节一致，AA 页面字段约为 `235B total / 22B active`、`262,144` context。DataCurve 当前没有精确 `mini_swe_agent_qwen3_vl_*` 行，不迁移其他 Qwen 的 Agent 分数。两个配置归并为同一基础模型，不新增两个模型。
- 已核验 Qwen3-VL 官方 GitHub、HF Instruct revision `710c13861be6c466e66de3f484069440b8f31389`、Thinking revision `6664affde68449468deb7527186455c7450c13c0`、config、README、Qwen3-VL Technical Report `arXiv:2511.21631` 和 Transformers raw main。已固定 `Qwen3VLMoeForConditionalGeneration`、94 层、64Q/4KV、128 experts/top-8、262K position、RoPE theta `5,000,000`、视觉 depth 27、patch 16/temporal patch 2、DeepStack `[8,16,24]`。
- 研究和章节已写入三模块结构、Interleaved-MRoPE、DeepStack、Video Timestamp、S0-S3（67B/1T/1T/100B、8K/8K/32K/262K）训练阶段、square-root normalized loss、SAPO/General RL、Thinking with Images、GUI/tool reward 和多模态 serving 证据边界；新增研究笔记 [`qwen3-vl-source-notes.md`](research/model-update-2026-09/qwen3-vl-source-notes.md) 与第二十一册第 92 章 [`Qwen3-VL：Interleaved-MRoPE、DeepStack 与视频时间戳`](book-21-transformer-architecture-evolution/chapters/92-qwen3-vl-interleaved-mrope-deepstack与视频时间戳.md)。
- 当前状态：**AA 单榜内容专题闭环**。完整权重加载、生产视觉/稀疏 kernel、跨视频 chunk replay、目标硬件 profiling、端到端 multimodal streaming、GUI/tool acceptance、完整训练/后训练 recipe 和独立 benchmark 仍待核验；不将 goal 标记为 complete。

## 2026-09-22：Opus 5 与 Gemini 3.5 Flash-Lite 断点复验

- 已确认工作目录为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；既有 dirty worktree 修改全部保留。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、GLM/Z.ai、Qwen、Kimi/Moonshot、DeepSeek、Google/Gemini 和 xAI/Grok。
- Claude Opus 5：当前 AA 详情快照 `3,869,351` bytes / SHA-256 `c18260ab4ff331d5bd3305691db2d4b6051dc2ebe642aa1458c5b8fa2c367643`；当前第三方字段为 Intelligence Index `50.7771115797629`、`56.4471785104486 tokens/s`、`49.2490756305s` TTFC、1M context 和约 `$5.8584`/task。9 月 21 日的速度/TTFC 保留为历史值；Opus 5 的 DataCurve max 仍为 327/444、Pass@1 `73.6486%`、Pass@4 `88.4956%`，不迁移相邻 Claude 版本结果。状态：**资料级闭环**。
- Gemini 3.5 Flash-Lite：AA 详情快照 `3,858,652` bytes / SHA-256 `bbe13cb52c85772080a105b1ec98c42bd71e7194dd67abbc26cb52f9ba1115d`；当前第三方字段为 release `2026-07-21`、Intelligence Index `22.1685424839812`、`386.373295824977 tokens/s`、`11.2053542005s` TTFC、1M context 和约 `$0.1235`/task。DataCurve 快照 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1` 没有精确 Lite Agent 行；不迁移相邻 `gemini_3_5_flash_high` 结果。状态：**AA 单榜资料级闭环**。
- 已将两份研究笔记、`model-inventory.md`、`source-index.md`、`inventory-interpretation.md`、`plan.md` 和本进度表同步；两者均不新增重复 Transformer 正式章节。下一步执行全仓库 Markdown 围栏、相对链接、研究 Python `py_compile`/`compileall`、`git diff --check` 和证据边界审计；goal 保持 active。

## 2026-09-22：GLM-5.1 当前时点复验与活动锚点切换

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree 修改。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为八家范围。
- 三条代理取得一致的 Artificial Analysis GLM-5.1 详情页；当前快照 `3,971,543` bytes / SHA-256 `c806874d4eb235658b04ba6f527e1ca1db30b2c5e149e8690cd428fd5d2614c9`，中文首页 `1,798,627` bytes / SHA-256 `4254f2dad0222fd7147ae50047e3bfe0b382ec1aed1c788c94ef9e8f683242fe`。页面仍为 `GLM-5.1 (Reasoning)`、release `April 2026`、200K context；当前第三方测量为 Intelligence Index `26.0585912980095`、`37.1922485381327 tokens/s` 和 cost per Intelligence Index task `0.9217822401466147`。与 9 月 20/21 日的速度和价格差异按 provider/采集时点漂移记录。
- DataCurve 当前快照 `268,571` bytes / SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870` 没有精确 `mini_swe_agent_glm_5_1_*` 行；不记录或迁移 GLM-5/5.2/5.3/5.3-Flash 的 Pass@1、成本、输出 token 或 Agent steps。
- 当前 Z.ai 文档与博客证据支持 long-horizon Agent、multi-turn SFT/RL/process-quality evaluation、VectorDBBench 外循环、KernelBench Level 3 verifier/harness、Linux desktop 自评、DSA/MoE 配置和 thinking/tool/cache 协议。博客、AA 和 provider 字段继续分账；模型卡链接的是 GLM-5 报告，不能把 GLM-5 的训练/架构结论改写为 GLM-5.1 专属事实。
- 已更新 `glm-5.1-source-notes.md`、`model-inventory.md`、`source-index.md`、`inventory-interpretation.md`、`plan.md` 和本进度表。当前状态仍为 **AA 单榜资料级闭环**，不新增重复 Transformer 正式章节；完整 recipe、indexer loss/recall、过程质量 verifier、生产 kernel、硬件 profiling、线上 tool acceptance、独立技术报告和复现仍待核验。goal 保持 active。

## 2026-09-22：GPT-5.6 Luna 当前活动锚点与官方运行时补证

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree 修改。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商范围仍为八家。
- 本轮沿两个排行榜已经确认的 `GPT-5.6 Luna` 推进，没有从 OpenAI 官方目录另发现模型。Artificial Analysis 详情 HTTP 200，当前快照 `3,861,318` bytes / SHA-256 `1b425d8f2a93f8418702ac7acbb62b26850b98fe20680f82205bed5fce38fed0`；第三方字段为 release `2026-07-09`、Intelligence Index `37.3244239690841`、median output speed `158.728370482714 tokens/s`、cost per Intelligence Index task `0.17829726152289094`、1M context、页面 input/output `$0.20/$1.20`。
- DataCurve 当前快照 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；精确 `mini_swe_agent_gpt_5_6_luna_max` 行为 attempted `448`、passed `301`、Pass@1 `67.1875%`、Pass@4 `90.26548672566371%`、平均成本 `$0.6056233620535714`、平均输出 `73399.70758928571` tokens、平均 Agent steps `101.68080357142857`、median peak context `201647`。这些结果继续绑定 `mini-swe-agent + tools + task environment + verifier`，不写成裸模型分数。
- 7890 代理取得官方 Markdown 页面并记录哈希：Luna `3,744` bytes / `1f425d8f...fed0`，Reasoning `70,253` / `91604df9...6719`，Agents `5,432` / `df4f61b6...43a`，Tools `33,282` / `4722fa10...a341`，Tool Search `39,288` / `9d6c3855...1fd8`，Prompt Caching `47,097` / `c70d858e...b2d1`，Compaction `14,272` / `73fd2fd1...0dd`；完整 URL、大小和 SHA-256 见 [`gpt-5.6-source-notes.md`](research/model-update-2026-09/gpt-5.6-source-notes.md)。
- 新增技术知识已完成归纳：Agents API 托管 Codex harness 和 session/turn/item 状态，Agents SDK 由应用控制 loop、工具、handoff、部署、存储和审批，Responses API 由应用直接管理 response item 与工具循环；tool search 分 hosted/client 两种，分别由服务端加载或应用回传 `tool_search_call`/`tool_search_output`；工具加载追加到上下文末端以尽量保留 cache prefix，但不改变权限、沙箱、幂等和 verifier 责任。
- 已将 prompt cache 与 compaction 的交互写清：model、tools、parallel tool calls、structured output、reasoning effort、verbosity 和 `context_management` 可能影响前缀；compaction 会替换旧上下文并造成首次 cache miss，tool search 会减少初始 schema token 但增加工具发现/版本状态。当前状态为 **双榜资料级闭环**，暂无 GPT-5.6 专属架构章节；正式书章和题库同步后，继续做全库质量检查，goal 保持 active。

## 2026-09-22：Qwen3.7 Plus 当前活动锚点与托管 Agent 合同

- 工作目录仍为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree 修改。模型发现入口继续严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商继续为 OpenAI、Anthropic、Z.ai、Qwen、Kimi、DeepSeek、Gemini、Grok。
- 本轮从 Artificial Analysis 当前详情选择 `Qwen3.7 Plus` 作为下一活动锚点：`/tmp/q37-aa-20260922.out` 为 `3,859,009` bytes / SHA-256 `19e9b48bbc9c6d38fab3391bee6353cff5aaa2524bdf04589ae02bbad4a27040`；release June 2026、Intelligence Index `25.1622215821984`、`68.5428061089526 tokens/s`、cost per Intelligence Index task `0.32527343198119174`、约 1M context 和约 `$0.40/$1.60` 均是第三方/provider 字段。
- DataCurve 当前快照 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，生成时间 `2026-09-22T06:27:15.860279+00:00`；没有精确 `mini_swe_agent_qwen3_7_plus_*` 行，不迁移其他 Qwen 的 Agent 分数。
- 重新取得 Alibaba Cloud 官方资料：Qwen3.7 Plus 文档 `44,386` bytes / `662ebedd3a538e22490e00c4eb29def16c0c5e9ff2de1d3f8fe7d7d334b43656`，推荐模型页 `27,344` / `9df37ab1723f032187895b9961865f2831cb575147985a281310d0616f8f4fe8`，OpenAI-compatible API 文档 `60,448` / `ec3f4a6c3db9ea89fc4141e78cfcfaf389aafa67ff11efec664b3600aa40714`；完整 URL 和快照表见 [`qwen3.7-plus-source-notes.md`](research/model-update-2026-09/qwen3.7-plus-source-notes.md)。
- 已提取的面试主线：多模态交互式混合 Agent、读屏/GUI/移动端导航和视觉参考生成代码是官方产品定位；模型只提出 action proposal，region/scope、schema、权限、GUI/mobile executor、观察回灌、幂等、重试和 verifier 组成真实闭环。Function Calling、Structured Outputs、Web Search、Prefix Completion、Context Caching 按 region/scope 分层，不能写成无条件的模型本体能力。
- 已记录官方上下文合同：1M context、991,808 max input、131,072 max output；thinking max input `983,616`，max chain-of-thought length `262,144`。北京/Global scope capability 与 Virginia US scope 不同，价格按区域、scope、输入长度、缓存和 batch 分档；不能用 AA 的 `$0.40/$1.60` 代替所有 provider 价格。
- 已新建 [`qwen3.7-plus-source-notes.md`](research/model-update-2026-09/qwen3.7-plus-source-notes.md)，并更新 `model-inventory.md`、`source-index.md`、`inventory-interpretation.md`、`plan.md`。接下来完成第十五、十七、二十、二十四册已有章节和八个全局面试/索引文件的同步；当前状态为 **AA 单榜资料级闭环**，goal 保持 active。

## 2026-09-22：DeepSeek V4.1-Flash 当前活动锚点与联网恢复复验

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree 修改。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为八家范围。本轮没有从 DeepSeek 官方目录、HF 关联文件或 recipe 仓库另发现模型。
- 三条代理对 Artificial Analysis 中文首页和 DataCurve DeepSWE 均返回 HTTP 200：AA 首页 `1,762,420` bytes / SHA-256 `de45e583216f236e0eb6c0b69b65b251394eea04137368a056c43e072a97a6f6`；DataCurve `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。没有新的重点厂商 canonical 模型。
- 沿既有锚点复验 DeepSeek V4.1 发布页和 AA `deepseek-v4-1-flash` 详情页，三条代理均 HTTP 200。发布页 `27,838` bytes / `bea79d60a0712f1971554724c94e8ff2145a048d3c0e80326efa4bacd2bf8e11`；AA 详情 `3,948,908` bytes / `114cc90d1cb8125174d9464141cbfd0faeac76f52b132f78630c0375c9fcf8fe`。AA 当前 Index `39.456167472527`、约 1M context、约 `$0.30/$1.20` 是第三方/provider 字段，不是本项目实测。
- DataCurve 没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，因此不迁移 V4 Pro/V4 Flash 或其他 DeepSeek 版本的 Agent 成绩。HF API 经 7890 成功取得 `6,714` bytes / `df3cb8b368d3a77a4eb3b96c8a4f85abfb3a199245bf9006a68d374a4b425ca3`，revision 仍为 `dba1be0a40aa45a94ad051997016db3960a90277`、`lastModified=2026-09-10T08:18:10Z`、48 个 safetensors 分片；8098 的 503/连接失败与 1234 失败只记录为线路边界。`deepseek-recipe` 经 1234/7890 可获取，8098 对 GitHub raw 为 TLS EOF，内容未见变化。
- 已同步 `plan.md`、DeepSeek 研究笔记、`source-index.md` 和 `inventory-interpretation.md`。当前状态保持 **内容专题 + reference implementation + recipe protocol evidence（AA 单榜）**；没有新模型或新权重 revision。后续只补完整权重加载、production kernel、candidate/index Top-K recall、真实 FP4 误差、DSpark draft/verify/rollback、EPD 调度、目标硬件 profiling、tool acceptance 和独立 benchmark，不新增重复 Transformer 章节。

## 2026-09-22：DeepSeek V4.1-Flash vLLM upstream runtime 补证

- 本轮沿 Artificial Analysis 已确认的 `deepseek-v4-1-flash` 锚点继续推进，没有从 vLLM 仓库另发现模型。通过 1234 和 7890 代理取得的 vLLM `main` registry 逐字节一致；8098 对同一组 raw 文件超时。`main` registry 快照为 `64,391` bytes / SHA-256 `64c80d8c6659833a9abf836180f2b7549903db2a664e22bc38e38c258a6f572e`，包含 `DeepseekV41ForCausalLM -> vllm.models.deepseek_v41` 与 `DSparkV41DraftModel -> vllm.models.deepseek_v41`；正确实现路径是 `vllm/models/deepseek_v41/` 包。
- 已固定 `__init__.py`、`quant_config.py`、NVIDIA/ROCm `vl_model.py` 与 `dspark.py` 快照及 SHA-256。代码证据确认：`expert_dtype=fp4/fp8` 分支分别对应 MXFP4 + `ue8m0` scale、block-FP8 + float32 scale；vision wrapper 通过 `inputs_embeds` 注入 ViT/aligner embedding、保留 raw `input_ids` 供 `bias_vl` 路由、支持 encoder CUDA graph/ViT data parallel，并显式跳过 `mtp.*`，当前 vision variant 不支持 MTP/DSpark draft heads；DSpark 使用 3 个 draft 层、目标层 ids、`[max_num_batched_tokens, index_topk]` Top-K buffer、target checkpoint 的 `mtp.{0,1,2}.*` 权重、Markov/confidence head、逐位置 sigmoid confidence、共享 embedding/lm head、context KV/SWA cache 插入和 FP4/FP8 scale 分支。
- 对照 vLLM `0.29.0` stable registry：`63,102` bytes / SHA-256 `fef8293fe19cef01768a4c5bd8adc05e120202f1c1597eb264d090672894fbd7`，本快照有通用 V4/DSpark 入口，但没有两个 V4.1 专用类名。因此当前状态更新为 **内容专题 + HF reference implementation + vLLM upstream main runtime evidence + recipe protocol evidence（AA 单榜）**。这不等于 stable wheel 已支持、完整权重已加载、GPU/ROCm 已验收、speculative acceptance/吞吐已复现或生产 SLO 已证明。
- 已同步 `plan.md`、DeepSeek 研究笔记、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md` 和第二十一册第 81 章；第 81 章新增 vLLM runtime 小节、2 个面试追问和来源。后续仍补完整权重加载、stable release/目标硬件、candidate/index Top-K recall、真实 FP4 误差、DSpark verify/rollback/acceptance、EPD 调度、GPU profiling、tool acceptance 和独立 benchmark，不新增重复 Transformer 正式章节，goal 保持 active。

## 2026-09-22：DeepSeek V4.1-Flash SGLang main/stable runtime 对照补证

- 本轮沿 Artificial Analysis 已确认的 `deepseek-v4-1-flash` 锚点推进，没有从 SGLang 仓库另发现模型。通过 `10.237.126.170:1234` 的 GitHub Contents API 固定 SGLang `main` 的 `deepseek_v4.py`、`deepseek_v4_dspark.py`、`deepseek_v41_vit.py` 和 `deepseek_v4_nextn.py`，分别为 `241,421`、`47,640`、`5,126`、`9,459` bytes；完整 SHA-256、Git blob 和 URL 已写入 DeepSeek 研究笔记与来源索引。raw 端点超时只代表线路边界，不能解释成源码不存在。
- main 代码确认 V4.1 vision 的 TP/EP/DP 路径和 CP/PP/MoE A2A 限制、2D-RoPE ViT/Aligner、attention data parallel、V4/V4.1 FP8 block-size 差异、MXFP8/FP8 prefill autotune、FlashInfer、unified KV、DSV4 sparse indexer/cache，以及 DSpark 的 Markov/confidence head、`mtp.*` 映射和 draft stage `vision_n_layers=0`。这属于 **SGLang upstream main runtime implementation evidence**。
- 对照 SGLang `v0.5.20`（tag `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`，发布 `2026-09-18T22:41:33Z`）：stable 只有通用 V4/DSpark 文件，`deepseek_v41_vit.py` 404，相关源码中 `deepseek_v41`/`dsv41` 计数为 0。不能把 main 实现升级为 stable V4.1 serving 支持。
- 已同步 `deepseek-v4.1-flash-source-notes.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan.md` 和第二十一册第 81 章，新增 SGLang 小节与面试追问。当前状态为 **官方内容 + HF reference + vLLM main + SGLang main + recipe protocol evidence（AA 单榜）**；完整权重、目标硬件、视觉 draft/target verify、FP4/FP8 质量、DSpark acceptance/rollback、EPD 调度、profiling、tool acceptance 和生产 SLO 仍待核验，goal 保持 active。

## 2026-09-22：Kimi K3 SGLang stable/main runtime 对照

### 后续联网补证：SGLang main commit history

- 前一条代理失败记录属于当时的首次尝试；随后网络恢复，百度经三条代理均 HTTP 200，GitHub commits API 经 10.237.126.170:1234 成功，7890 返回 403，8098 出现 TLS/代理协议错误。
- K3 commit history 响应为 51,251 bytes、SHA-256 2f7d8402aa104fe9a5b7839e85c9eb56060e98f7aeb4aecb787005f775351e04，最新记录时间 2026-09-22T06:35:12Z。当前 main 文本源码仍为 171,101 bytes、blob 383a6f47812bccd1cb91b76814cd0730ff945dd7、SHA-256 54069412bf3766e974c168985b4e848ecfc29aa4f02dbe27f8e037345faefc0e。
- 已补记 8ac19cc deferred KDA gate、c4d3770 CUDA graph stream、c2c3629 O(1) expert lookup、72d5c5b FP32 routing finalize、f4c2563 PP/DCP/DSpark、2d0e94e shared-expert process group、8ac39c6 Ascend A5/NPU 和 cb32dbc ROCm KDA projection。完整解释已同步研究笔记、来源索引、模型清单、正式章节、推理引擎章节和题库。
- 这些提交仍是 mutable main source evidence，不等于 stable wheel、完整权重、双状态恢复、目标硬件 profile、视觉正确性、DSpark acceptance 或线上 tool/verifier 通过；goal 保持 active。

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，等同 `/home/zzc/llm-from-zero-to-interview`；不回滚既有 dirty worktree。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为八家范围。本轮没有从 SGLang 源码目录另发现模型。
- 已固定 SGLang `v0.5.20` K3 text 源码：168,114 bytes、blob `b0ede48c88264d518351a66abf623f1bcf8a730e`、SHA-256 `7a3ef867394a2fd52b3a71a979c053b51e9f2310c35cd4c12ef3aa8e7be172e5`；tag commit 为 `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`、发布时间 `2026-09-18T22:41:33Z`。已固定 SGLang `main` K3 text：171,101 bytes、blob `383a6f47812bccd1cb91b76814cd0730ff945dd7`、SHA-256 `54069412bf3766e974c168985b4e848ecfc29aa4f02dbe27f8e037345faefc0e`。
- 两版本 `kimi_k3_vl.py` 逐字节一致：32,790 bytes、blob `c423027994645e4a839ec47bd26bcc1b2a7cb012`、SHA-256 `2924c38f652a6ebff2ef79c49f2f336ba18723ea4b854d3ac14d95292988acdb`。因此本轮新增差异主要是文本 runtime：LatentMoE 的 latent down/up、MegaMoE/DeepEP/Mooncake/Ascend-FuseEP/MoRI A2A、DP/SP token shard、shared-expert TP/reduce-scatter、SBO/NPU dual-stream、ModelSlim fused QKVG/packed loader、KDA fused decode capability gate 和 fallback。
- 已同步 K3 研究笔记、来源索引、模型清单、榜单解释、`plan.md`、第二十一册第 88 章和第二十四册第 32.52 节；面试题下一步补入 stable/main/source/target acceptance 对照。当前状态为 **内容专题闭环 + HF/vLLM + SGLang stable/main source evidence**，不等于完整权重、SGLang 依赖安装、双状态恢复、视觉数值、目标硬件 profile、线上 tool acceptance 或生产 SLO 已通过。
- 本轮重新请求 GitHub commits API 时，`7890`、`1234`、`8098` 三条代理均连接失败；因此没有固定 mutable `main` 的 commit history，也没有把失败解释成源码不存在。网络恢复后优先补 history/commit message，再决定是否需要更新源码哈希和证据等级；goal 保持 active。

## 2026-09-22：两榜复验与 gpt-oss provider 兼容性/raw CoT 补证

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；不回滚既有 dirty worktree。三条代理重新获取两个唯一排行榜并逐字节一致：Artificial Analysis 中文首页 `1,762,420` bytes / SHA-256 `de45e583216f236e0eb6c0b69b65b251394eea04137368a056c43e072a97a6f6`；DataCurve DeepSWE `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。
- 按八家重点厂商 canonical slug 归一化后没有新的重点模型。当前 AA 页面中可见的 K2 Horizon 0.9B、7B、375B 等其他厂商条目按用户范围排除，不建立新的锚点，也不把 K2-Horizon-7B-Uno 作为排行榜发现。
- 选择已有 Artificial Analysis 锚点 `gpt-oss-120b`/`gpt-oss-20b` 继续补证。gpt-oss README 当前 `24,453` bytes / SHA-256 `578ad0f82c1d823229f9bf2b52b3f1c55ce82772ac57f634f0ca4eb46a6370aa`，与 9 月 18 日一致；GitHub commit history `44,611` bytes / `420dcfab2dad69aa386b5207ac3aa35f627151e292c63caced716d46c59e288c` 的最新提交 `7b583341fe16`（2026-07-24）是 You.com backend API key 修复，不是模型或核心推理更新。
- OpenAI Cookbook 当前专题新增/列出两个直接相关的官方入口：[实现兼容性验证](https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations)，`336,633` bytes / SHA-256 `4f293d2bd51666966a8d3657893eb1d1093b4b3230a405c9178cbc7302d664b8`；[raw CoT 处理](https://developers.openai.com/cookbook/articles/gpt-oss/handle-raw-cot)，`338,262` bytes / SHA-256 `7330273ff59d6e1a344abe3f435ae421a20ebd775bda9122d22ff606f0782072`。
- 新知识已写入：Responses `reasoning.content[].reasoning_text`、`response.reasoning_text.delta/done`、item/index/turn lineage replay；Chat Completions provider 的 `reasoning`/delta 兼容约定；raw CoT 不直接面向终端用户；Harmony/API shape 与 tool-call smoke test、AIME 16 次/题、GPQA 8 次/题、HealthBench 1 次/题的质量 eval 分层。
- 官方 compatibility-test 的 0 invalid requests 且 `pass@k`/`pass^k` 均超过 90% 只是强信号，不等于 MXFP4/MoE kernel、硬件 profiling、独立复现或生产 acceptance。当前 gpt-oss 状态为**内容专题闭环 + 官方 provider 兼容性/raw CoT protocol evidence**；完整 recipe、kernel/误差、目标硬件、精确 DataCurve Agent 行和线上接受率仍待核验。
- 已同步 `gpt-oss-source-notes.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan.md`、第二十一册第 87 章、第二十四册 serving 章节、`INTERVIEW_BANK.md`、`EXERCISES.md`、`PROJECTS.md`、`GLOSSARY_EN_ZH.md` 和 `KNOWLEDGE_GRAPH.md`；goal 保持 active。

## 2026-09-22 GLM-5.3-Flash upstream/runtime 对照补证

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree，不回滚其他模型专题。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，当前活动锚点为 `GLM-5.3-Flash`。
- 本轮通过 `10.237.126.170:1234` 取得 SGLang/vLLM GitHub Contents/tree 与提交历史；`10.24.27.134:7890` 对部分 GitHub API 返回 403/rate limit，`10.24.27.134:8098` 出现 TLS wrong-version/读取失败；绕过代理时 DNS 不可用。代理失败只作为访问路径边界记录，不解释为源码或页面不存在。
- SGLang `v0.5.20` tag（commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`）已固定 `glm5_next.py`：`61,466` bytes、SHA-256 `12c5157b07fb7c6d93f34e84c43a37866d2e382e703729e2205aed9f8961f9c2`；对应 config `12,026` bytes、SHA-256 `3b3c7aa3ae60e1cf59edf91e1c11a6aa49be7532340c8e2482f7f75ed859f3e0`。这证明固定 stable source entry，不代表本机权重加载或目标硬件 profile。
- SGLang `main` tree commit `9d58189c12e4e14a7eea20f24f9aa7b17e221778` 的 GLM5Next 模型文件为 `68,097` bytes、SHA-256 `1cd324533aa0827e7542e27fc39c5901c2330823c4b3a32b79bcb26521dbcfec`；补证 projection fusion、KDA projection/prefill metadata、mHC boundary fusion、AMD FP8/Quark MXFP4 和 B200/H200 测试门禁。相关提交 `c8eb54c41da1`、`2fa6b94e3440`、`b44e2486824e` 作为 mutable upstream history 记录。
- vLLM `main` tree commit `81d7293c2167e39f3ffddc9a82d633f94e8a1eaa` 已有 `vllm/models/glm5next/`：`attention.py`、`kda.py`、`model.py`、`mtp.py`、`common/sparse_indexer.py`。源码事实包括 pool 粒度 `Glm5NextIndexerCache`、保存未完成 pool raw BF16 K/gate score 的 `Glm5NextTailCache`、独立 KDA/Mamba/GDN state、`num_spec` conv 宽度和 MTP layer/weight mapping、top-k reuse、slot compact/local argmax。
- vLLM `v0.29.0` tree 快照为 `1,979,822` bytes、SHA-256 `7131879ae9592d90738776d24ae213f789577317b2c5427f350761a87ca05070`，路径搜索未发现 `vllm/models/glm5next/` 专属路径。截至本条 2026-09-22 记录，证据为 SGLang stable/main + vLLM main implementation；2026-09-24 后续确认 v0.30.0 stable source 并完成逐文件对照，见本进度末尾最新 GLM 条目。recipe 的 `v0.29.0+` 仍不能写成 v0.29.0 stable tag 已含 GLM5Next。
- 已同步研究笔记、来源索引、模型清单、榜单解释、`plan.md`、第二十一册第 84 章、第二十四册第 32.47.6 节、`INTERVIEW_BANK.md`、`EXERCISES.md`、`PROJECTS.md`、`PAPERS.md`、`GLOSSARY_EN_ZH.md` 和 `KNOWLEDGE_GRAPH.md`。完整权重、stable wheel 运行、目标硬件数值/性能、PD/EPD recovery、MTP acceptance、tool/verifier 和生产 SLO 仍待核验；goal 保持 active。

## 2026-09-22 GLM-5.3 标准 DSA runtime 闭环推进

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同。保留既有 dirty worktree；模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，当前活动锚点从 `GLM-5.3-Flash` 切换为标准 `GLM-5.3`。
- 三代理新鲜排行榜复验一致：Artificial Analysis 中文首页 `1,762,420` bytes / SHA-256 `de45e583216f236e0eb6c0b69b65b251394eea04137368a056c43e072a97a6f6`；DataCurve `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。八家重点厂商没有新的 canonical 模型，因此没有新增模型名称，也没有把 Flash/其他模型的 Agent 结果迁移到标准版。
- 标准版 HF revision 固定为 `aca966e4e02791568aa6a4ced368624b3d897f42`；config 为 `29,464` bytes / SHA-256 `3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69`。它确认 `glm_moe_dsa`、`GlmMoeDsaForCausalLM`、78 层、前三层 dense、256 routed/top-8/1 shared、`q_lora_rank=2048`、`kv_lora_rank=512`、`index_topk=2048`、`index_topk_freq=4`、`index_share_for_mtp_iteration=true` 和 1,048,576 positions；实现账本为 21 个 Full、57 个 Shared indexer 层。
- Transformers main ref 为 `0bc252863a4e5c0e709893664cc57369ebcd7353`，固定了 interleaved indexer RoPE、Full 层 top-k、Shared 层 `prev_topk_indices` 和 sparse mask。vLLM main ref `81d7293c2167e39f3ffddc9a82d633f94e8a1eaa` 与 v0.29.0 registry 都把模型映射到 `deepseek_v32`；SGLang main ref `861b11f087af2822cb545ea1895059a721831014` 与 v0.5.20 都包含 `is_glm_moe_dsa`、`DeepseekV32ForCausalLM` 和跨层 top-k 状态。完整 bytes/blob/SHA-256 已写入 `source-index.md` 与研究笔记。
- 已同步 `research/model-update-2026-09/source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、本文件、第二十一册第 89 章、第二十四册第 32.53 节，以及 `INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`PAPERS.md`、`KNOWLEDGE_GRAPH.md`。标准版与 Flash 的边界已明确：不迁移 Flash 的 `RadixLinearAttention`、KDA、视觉、双 state pool、EPD 或 `glm5_next.py`。
- 当前状态：**双榜资料级闭环 + stable/main runtime source evidence**。source entry 不等于完整权重加载；index/evidence recall、MLA/indexer cache recovery、MTP acceptance、目标硬件 profiling、tool/verifier acceptance 和生产 SLO 仍待实测。下一步按 gate 做本地导入/最小构造与可用硬件 profiling；若环境不具备权重或目标硬件，保持 `unverified`，goal 继续 active。
- 本轮本地环境门禁已检查：`torch`、`transformers`、`vllm`、`sglang`、`safetensors` 均未安装；`nvidia-smi` 无法连接 NVIDIA driver；仓库及 `/tmp` 未发现标准 GLM-5.3 权重分片。故没有伪造 full-weight load、数值正确性、index/evidence recall、MTP acceptance 或 GPU profile 结果；这些项继续标记为 `unverified`，待具备依赖、权重和目标硬件后再执行。

## 2026-09-22 Grok 4.7 当前时点 reasoning/MCP 增量补证

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree，goal 保持 active。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、Z.ai、Qwen、Kimi、DeepSeek、Gemini、Grok。
- 三条代理刷新排行榜：Artificial Analysis 中文首页 `1,762,420` bytes / SHA-256 `de45e583216f236e0eb6c0b69b65b251394eea04137368a056c43e072a97a6f6`；DataCurve `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。没有新的重点厂商 canonical 模型；Grok 4.7 仍没有精确 `mini_swe_agent_grok_4_7_*` 行。
- `10.24.27.134:7890` 的 xAI 刷新快照已固定：Reasoning `522,822` bytes / `ce3cd1997308094472d2cce8017eee510cf3441c7be2711029870c9d18d1cc87`，Compaction `521,486` / `2a58eb48a8ed56d5f747e20fdee8dc9a50cc5998507f3ac0f598afd3d0f48eb3`，Remote MCP `475,038` / `459b83b7522eaa9139677f9da9760593e56072d52fd63a1883ba6a370a13c705`；模型页 `376,913` / `add926110deb683b8c90a126340a1f1fa4fc44a6aaa698ea5d3a7d68f537bdd5`，发布页 `288,007` / `88d0ce52f3c9edfe273d955a1b4bd2547202f5f6a226b14070729703d818f53b`。
- 新增技术点：Grok 4.7 Responses 每次返回 `reasoning.encrypted_content`，服务端工具的加密输出也需原样保留；Reasoning summary 通过 `response.reasoning_text.delta`/`response.reasoning_summary_text.delta` 流出，但不是完整 CoT；服务端 thinking trace rehydration 与 `store` 决定的 `previous_response_id` 存储行为分开；`presencePenalty`/`frequencyPenalty`/`stop` 是 reasoning 请求兼容性禁用字段；`xhigh` 从 Grok 4.6 起可用。
- Remote MCP 新增协议边界：只支持 Streaming HTTP/SSE；Responses 的 `allowed_tools`/`headers` 与 xAI SDK 的 `allowed_tool_names`/`extra_headers` 不同；`server_description` 可选；OpenAI Responses API 当前不支持 `require_approval`/`connector_id`。这些字段仍不替代宿主授权、凭据隔离、网络策略、审计、幂等和 verifier。
- 已同步 Grok 4.7 研究笔记、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan.md`、本文件和必要书系/题库/全局面试配套；当前仍为 **AA 单榜内容专题闭环**，不新增重复 Transformer 章节。参数、内部架构、完整训练/后训练 recipe、完整权重加载、生产 kernel、硬件 profiling、线上 acceptance 和独立复现继续标记为待核验。

## 2026-09-23 DeepSeek V4.1-Flash API contract 补证计划

本轮已把官方 API 文档作为当前锚点的服务契约补齐，但模型发现规则不变：新模型只能从 Artificial Analysis 与 DataCurve DeepSWE 发现，官方文档、HF、vLLM、SGLang 和 recipe 只用于扩展已经确认的 canonical 模型。

已纳入的 contract 字段：

1. `deepseek-flash` -> V4.1-Flash 的 alias、1M context、384K max output 和并发上限；记录 `requested_model` 与响应 `served_model`，不把兼容 alias 当 checkpoint。
2. Vision URL/`file_id`/request size、超时、600 图上限和当前约 1024 image tokens/图；历史 384 image-token 公告按旧 alias/日期保留。
3. Files `purpose=user_data`、64 MiB 文件、生命周期、25 GiB/用户和 10,000 文件配额；文件存在、图像被送入模型和图像证据被正确使用分开验收。
4. Responses stateless semantic SSE、递增 `sequence_number`、终态事件、无 `[DONE]`、不支持字段和 `function_call_output`/`custom_tool_call_output` 回灌。
5. `/beta strict=true` 的 JSON Schema 约束与 `deepseek-recipe` adapter 的能力分账；schema 合法不等于权限、执行、幂等和业务 verifier 通过。

下一阶段门禁顺序：先做不产生费用的 API/parser contract toy 和错误/重试/幂等账本，再绑定已有 HF revision、vLLM `v0.30.0`、SGLang main 的 source manifest；只有取得完整权重和目标硬件后，才测试 FP4/FP8 误差、candidate/index recall、DSpark draft/target acceptance、rollback、EPD、视觉 cache 一致性和 p99。API 文档、release registry、wheel、recipe README 和排行榜分数均不能替代这些实测。goal 保持 active。

本轮已完成第一阶段 toy：[`deepseek_v41_api_contract_audit.py`](research/model-update-2026-09/code/deepseek_v41_api_contract_audit.py) 在无网络条件下通过 semantic SSE、strict schema、权限、有限 retry、幂等和独立 verifier 自测。下一步优先检查本机是否具备完整 V4.1 权重、目标 runtime 和可用硬件；若不具备，则保留 `local_protocol_toy` 证据等级，并从两个排行榜八家重点厂商的既有 canonical 条目选择下一个周边资料锚点，不从官方 API/代码仓库另发现模型。

## 2026-09-23 Kimi K3 当前时点复验

本轮重新取得 [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh)、[Artificial Analysis Kimi K3](https://artificialanalysis.ai/models/kimi-k3) 和 [DataCurve DeepSWE](https://deepswe.datacurve.ai/)；三条代理逐字节一致。AA 首页为 `1,783,605` bytes / `7a954aed8916ec9c5c88380cfc159274573dc9547182032ca142f556c0fc2916`，K3 详情为 `4,080,091` bytes / `340a36ac5922f354aa42553fe3da2ce4acf4231bbb6af97fd6a88b974dbea4ee`，DataCurve 为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。八家重点厂商按 canonical slug 去重后没有新模型。

当前 K3 `max` 的 AA 观察值为 Intelligence Index `43.5938229518782`、median output speed `36.9982439081499 tokens/s`、cost per Intelligence Index task `2.0001323004425493`、1M context。DataCurve 精确行 `mini_swe_agent_kimi_k3_max` 为 `309/451`、Pass@1 `0.6851441241685144`、Pass@4 `0.8938053097345132`、平均成本 `$4.654682129933482`、平均输出 `81,499.84` tokens、平均 Agent steps `97.5876`、4 runs/113 tasks；这些数字绑定 `mini-swe-agent + tools + environment + verifier`，不迁移为裸模型能力。

K3 README 与 HF metadata 复验后仍分别为 `45,004` bytes / `849a303d849486aac61d1f0c253e3a1148ca59e801be807151c2dc3cce57ccb2` 和 `9,436` bytes / `6dfa5b1680074f4dfdd79a36b3342a191aacc32e15a9bd4a3895b98570773939`，HF revision 仍为 `f831ab66814297da540d832a5235f8e904f29d06`。SGLang main 当前文本文件的 `171,130` bytes / `9af22b45f8d8f8a5931c3bd60310316090973fd5cf9626aabd2a618a9e4baa6d` 相较前一快照仅有 import/type annotation 变化。结论是“当前时点复验且 artifact 未漂移”，不是新版本或架构升级。

本轮把 K3 的面试边界补入研究笔记：约 `2.5x scaling efficiency` 是发布方声明；preserved thinking history 要求结构化原样回传；跨模型切换可能不稳定；excessive proactiveness 是官方限制；Kimi Code、Claude Code、Codex、H20/H100 和 compaction 混用时不能拼接 benchmark。已同步 `kimi-k3-source-notes.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、第 15/88 章、题库和练习。下一步仍按完整权重、目标硬件、MLA/KDA 双状态 recovery、tool/schema/idempotency/verifier acceptance 和生产 SLO 门禁推进；若环境不具备，维持 `unverified`，再切换两个排行榜中已有的下一周边锚点。

## 2026-09-23 Claude Fable 5.1：System Card 正文与安全条件账本

本轮继续沿 Artificial Analysis 已发现的 `claude-fable-5-1` 推进。最新 AA 详情为 `4,006,915` bytes / SHA-256 `d80b226b7a4756cac93d6a065f400f39c9f756b971b827ecf0efbd5a67bffaeb`；DataCurve 当前没有精确 `mini_swe_agent_claude_fable_5_1_*` 行，不迁移 Fable 5 或其他 Claude 版本的 Agent 分数。Anthropic 发布页、System Card 原始 PDF 与正文抽取哈希已同步到研究笔记和来源索引。

System Card 正文确认 Fable 5.1 与 Mythos 5.1 共享模型权重，服务差异主要来自 safeguards 与访问计划；训练数据边界、ClaudeBot、去重/分类和 2026-06 cutoff 已记录。RSP 结果针对 Mythos：CB-1、未达 CB-2；autonomy threat model 1 适用但评为 low，threat model 2 未达。能力数字（Terminal-Bench、ProgramBench、OSWorld、CursorBench 等）和安全数字（Gray Swan、Shade、browser-use、permission-hook/sandbox 观察）必须绑定 snapshot、effort、tools、permissions、safeguards、fallback、任务环境和 verifier。

本轮计划验收项：

1. 将 System Card 的发布方 benchmark 与 AA provider 指标、DataCurve harness 指标分为三本账，不做跨榜单迁移。
2. 对 cyber、biology、prompt injection 和浏览器安全结果记录 `actual_model`、`fallback_reason`、safeguard state、是否 helpful-only/关闭 safeguards 以及最终 artifact verifier。
3. 用 Fable 5.1 的 thinking block 兼容、forced tool error、历史编辑失效、`display: "updates"` 和 content provenance 设计协议回放与状态恢复 toy；进度事件不得替代工具回执，provenance 不得替代事实验证。
4. 继续保持参数规模、层数、dense/MoE、完整训练/后训练 recipe、真实 API capability probe、完整权重、目标硬件 profile、独立复现和生产 SLO 为 `unverified`；不新增 Fable 5.1 Transformer 架构章节。

完成上述同步后，Fable 5.1 状态为 **AA + System Card 正文证据闭环**。下一轮仍先复抓两个排行榜，再从八家重点厂商的已有 canonical 条目选择锚点；官方目录、代码仓库和论文不能绕过两榜发现规则。

## 2026-09-23 GLM-5.3/SAO 下一阶段计划

本轮活动锚点切换为两个排行榜中已经存在的标准 `GLM-5.3`；没有从 SAO 论文、Z.ai 文档、HF 或 runtime 仓库另发现模型。新模型发现入口继续只允许 Artificial Analysis 与 DataCurve DeepSWE，重点厂商继续限定为 OpenAI、Anthropic、Z.ai/GLM、Qwen、Kimi/Moonshot、DeepSeek、Google/Gemini 和 xAI/Grok。

已完成的证据闭环：

1. 复抓 AA 中文首页、GLM-5.3 详情和 DataCurve，固定当前快照哈希、AA provider/config 字段与 `mini_swe_agent_glm_5_3_max` 精确行；DataCurve 结果继续绑定 `mini-swe-agent + tools + environment + verifier`。
2. 取得 Z.ai GLM-5.3/5.2 官方 Markdown，核对“同一 GLM-5.2 基座、post-training 改进”和“继承 SAO with compaction”的版本关系。
3. 取得并核读 [SAO 论文](https://arxiv.org/abs/2607.07508)：single-rollout、DIS、双侧 token clipping/masking、critic `K=2`、frozen-attention value model 和 Skip-Observation GAE 已形成可面试知识点。
4. 更新 GLM-5.2/5.3 研究笔记、第 86 章、source index、model inventory、inventory interpretation、PAPERS、题库、练习、术语、项目和知识图谱。

后续门禁按以下顺序推进：

1. 已完成不联网、不收费、不下载权重的 [`sao_async_rl_toy.py`](research/model-update-2026-09/code/sao_async_rl_toy.py)：固定 rollout barrier、policy ratio mask、action/observation GAE 和 critic update proxy，证据等级为 `local_protocol_toy`。
2. 继续核验 GLM-5.3 专属 compaction 的状态表示、序列化、切分/压缩质量门禁和完整 post-training recipe；没有一手证据就保持 `unverified`。
3. 如本机具备条件，再检查 pinned revision 的完整权重、runtime、数值、目标硬件和 tool/verifier acceptance；不得用论文实验、榜单分数或 release/source entry 替代这些门禁。
4. SAO 论文中的 Qwen3-30B-A3B 实验、GLM-5.2 部署声明、GLM-5.3 DataCurve 行和 AA 指标必须继续分账。goal 保持 `active`。

## 2026-09-23 GLM-5.3 compaction 门禁推进

- 重新联网时先验证代理可达性；本轮三条代理访问 Z.ai 页面和百度均在连接阶段失败，恢复后重新抓取 Artificial Analysis 中文首页、GLM-5.3 详情、DataCurve 和 Z.ai 官方 Markdown，并记录逐字节哈希。
- 在没有官方 schema/实现之前，只讨论可观察的 compaction 合同：状态 schema、确定性序列化、合法切分边界、tool call/result lineage、permission、幂等键、待执行副作用、artifact digest、verifier 状态、预算和恢复后的重复动作率。
- 使用 [`glm53_compaction_contract_audit.py`](research/model-update-2026-09/code/glm53_compaction_contract_audit.py) 作为零依赖教学审计；它的证据等级固定为 `local_protocol_toy`，不得写成 GLM-5.3 实现或 SAO 复现。
- 本机无完整权重、Transformers/vLLM/SGLang 依赖且 NVIDIA 驱动不可用，暂不下载大权重或虚构硬件结果。网络恢复后再按 pinned revision -> runtime import -> CPU/meta shape check -> 目标 GPU -> 数值/acceptance 的顺序推进。
- GLM-5.3 完成当前资料级门禁后，从两个排行榜八家重点厂商的已有 canonical 条目选择下一活动锚点；官方仓库、论文和 API 文档只能扩展已有锚点，不能绕过两榜发现规则。

## 2026-09-23 `7890` 恢复后的两榜复验与 Grok 4.7 下一步

本轮确认网络问题来自当前沙箱到代理的连接路径：沙箱内三条线路均失败；使用用户提供的 `10.24.27.134:7890` 受控沙箱外请求后，百度、Artificial Analysis 和 DataCurve 均 HTTP 200。当前 AA 首页 `1,783,572` bytes / `999ead1b8d025a8abccc5d6e879e548544770335863dec1be06b21d99378f8e2`，DataCurve `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；八家重点厂商没有新 canonical 模型。今后仍只把两张排行榜作为模型发现入口。

本轮沿已有的 `grok-4-7` 补证。AA 详情 `3,992,680` bytes / `9b24aa029ee4023aa2918495910a282a96f8bd9bba42848921f9496684e3e6bb`，xAI 模型页/Reasoning/Compaction 分别为 `376,911`/`72e4d59db379c9be4e1d68c9f5c14eff2281d9c9ffa5d2f3bcdb0e701471319c`、`522,822`/`f69666e9a7a47e75c7a394875e280ec15e6e2003a8cd1a6ec6a0e64334fa357a`、`521,433`/`ff467261f7174c235987232139c6c839d0488858507f3ac0f598afd3d0f48eb3`；正文只发生动态包装变化，未出现新 revision/架构/训练信息。

已新增 [`grok47_state_replay_audit.py`](research/model-update-2026-09/code/grok47_state_replay_audit.py)，把当前面试主线变成零依赖门禁：encrypted reasoning/tool state 原样保存，summary 不能替代 opaque item，function call/output 必须保持 call-id/idempotency lineage，compaction 不能重复执行副作用，`store=false` 不能依赖 `previous_response_id`。它只能标为 `local_protocol_toy`，不能替代 endpoint capability probe、真实加密状态、完整权重、硬件 profiling 或生产 SLO。

下一步顺序：先运行并审计该 toy，做 Markdown/链接/Python 校验；网络可用时再从两榜当前 canonical 集合选择下一项未闭环锚点。Grok 4.7 的参数、内部架构、完整训练 recipe、生产 kernel、硬件 profile、线上 acceptance 和独立 benchmark 继续保持 `unverified`，goal 保持 `active`。

## 2026-09-23 GPT-6 Astra 当前时点复验与失败路径门禁

本轮沿两榜已经确认的 `gpt-6-astra` 推进，没有从 OpenAI 文档、博客、代码仓库或 API 目录另发现模型。使用 `10.24.27.134:7890` 重新取得 Artificial Analysis 中文首页、Astra 详情和 DataCurve，均 HTTP `200`；AA 首页为 `1,783,572` bytes / SHA-256 `999ead1b8d025a8abccc5d6e879e548544770335863dec1be06b21d99378f8e2`，Astra 详情为 `3,997,013` bytes / `c4c040e6708555c1965efaf4eecd3d61b4199ffe337d07ce53970150b79103ce`，DataCurve 为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。八家重点厂商没有新 canonical 模型。

AA 当前 Astra `max` 字段为 release `2026-09-03`、knowledge cutoff `2026-04-30`、1M context、Intelligence Index `52.673669395513`、median output speed `58.6808088606541 tokens/s`、cost per Intelligence Index task `3.2575003134834164`、输入/输出 `$10/$50`、cache hit/write `$1/$12.5`；这些是第三方/provider 字段。DataCurve 生成时间为 `2026-09-22T06:27:15.860279+00:00`，五个精确配置为 `mini_swe_agent_gpt_6_astra_{low,medium,high,xhigh,max}`，分别 `303/452`、`329/452`、`331/452`、`335/452`、`331/452`；所有结果继续绑定 `mini-swe-agent`、工具、环境和 verifier。

重新抓取的官方 model page、Async tool calling、Steering 和 Misalignment monitoring Markdown 与上一轮一致；latest-model guide 从 9 月 21 日的 Astra 专属指南漂移为 9 月 23 日的 GPT-6 family 迁移指南，新增 Astra/Sol/Luna 的 effort、Responses/Chat Completions function-calling 和 EU Standard processing 分界。它没有产生新的 Astra 独有协议字段，但文档版本差异本身已写入研究笔记和 source index。新增的面试归纳是先读取当前 family/model capability matrix，再处理失败路径：唯一任务句柄和原始 `call_id` 必须贯穿 job registry；工具完成不等于模型消费；重复 steering 要拒绝；断线后不得隐式恢复旧 steering；`misalignment_policy_violation` 后停止自动重试并审计既有副作用。

已扩展 [`gpt6_agent_protocol_demo.py`](research/model-update-2026-09/code/gpt6_agent_protocol_demo.py)，本地 `py_compile`、主流程和 `git diff --check` 均通过。证据等级为 `local_protocol_toy`，不代表真实 API、服务端事件顺序、隐藏 reasoning、质量或生产安全。GPT-6 Astra 当前保持**内容专题闭环 + 双榜当前时点复验 + local protocol toy**；参数、架构、训练/后训练 recipe、system card、专属技术报告、完整权重、生产 kernel、目标硬件、独立 benchmark 和线上 acceptance 继续为 `unverified`。

下一步：先把本轮 Astra 失败门禁同步到第六册、题库和练习，随后从两榜当前八家重点厂商的已有 canonical 集合选择下一项未收口锚点；不从官方目录、论文、HF 或 runtime 仓库绕过两榜新增模型。

## 2026-09-23 Claude Opus 5.5 协议门禁补证

本轮重新取得 Artificial Analysis `claude-opus-5-5` 详情（`3,824,658` bytes / `727da6095c069bf0750a36cea442c7e582bbfeefdf4fc4a330a656bd8bd45c31`）和 Anthropic 官方 model/What's new/migration Markdown（`14,613`/`21,525`/`16,296` bytes；哈希见 `source-index.md`）。AA 的 canonical identity、2026-09-22 release、1M context、`57.6223698102963` Intelligence Index 与 `$4/$20` provider 字段保持不变；DataCurve 当前无精确 `mini_swe_agent_claude_opus_5_5_*` 行。

官方文档新增的可面试协议边界是：adaptive thinking 永远开启，disabled/manual budget 为 400；`tool_choice=any/tool` 为 400；thinking block 受 producer model、conversation 和 prefix 绑定；Claude API/Google Cloud 迁移 `computer_toolset_20260801`，Bedrock 旧工具仍有差异；工具间进度默认可能是空 thinking block；compaction、inline tools、refusal category、fallback 与 fast mode `usage.speed` 必须进入 trace。这些是服务/API 合同，不是内部架构或训练证据。

新增 [`claude_opus55_protocol_audit.py`](research/model-update-2026-09/code/claude_opus55_protocol_audit.py)，用标准库验证请求 capability gate、block-type parser、thinking binding、签名 compaction、tool idempotency、fallback 实际模型和负例错误分类。主流程、重复回放和 8 组失败路径通过，证据等级为 `local_protocol_toy`；不调用 API、不解密 thinking、不执行外部副作用。

当前 Opus 5.5 状态为 **AA 单榜 + System Card/API contract + local protocol toy**。参数、架构、完整 recipe、精确 DataCurve Agent 行、独立 benchmark、目标硬件和生产 acceptance 仍待核验。下一轮先复抓两榜，再从八家重点厂商既有 canonical 集合选择未收口锚点。

## 2026-09-23 GPT-5.6 Luna 当前时点复验与下一步

本轮继续只沿 Artificial Analysis 与 DataCurve DeepSWE 已确认的 `GPT-5.6 Luna` 推进。7890 已恢复官方页面访问：AA 详情当前为 `3,996,959` bytes / `4246b96416b4aaf4f8bbcacf72f8c45787944e24f02c59c6e3db1fed4de93c41`，DataCurve 为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；八家重点厂商没有新的 canonical 模型。AA 速度相对上一轮变化只记为 provider 测量漂移。

本轮已补齐 GPT-5.6 Luna 的官方 runtime 复验和 [`gpt56_luna_state_replay_audit.py`](research/model-update-2026-09/code/gpt56_luna_state_replay_audit.py)。toy 已通过 `py_compile` 和主流程，覆盖 persisted reasoning 的 `current_turn/all_turns`、完整 output item/function lineage、compaction canonical replay、cache prefix miss、四个 explicit breakpoint、30m TTL、hosted/client tool search 和 idempotent verifier；证据等级固定为 `local_protocol_toy`，不代表真实 endpoint、隐藏 reasoning、模型质量、完整权重、目标硬件或生产 SLO。

GPT-5.6 Luna 当前状态为 **双榜当前时点复验 + 官方 runtime contract + local protocol toy**。下一步按以下顺序推进：

1. 保留 AA 与 DataCurve 的独立账本，不把当前速度漂移写成模型升级，也不迁移其他 GPT/Agent 行的成绩。
2. 优先做真实 API capability probe（如环境允许），验证 model ID、reasoning context、stateless replay、compaction、cache usage 和 tool-search schema；失败时记录 endpoint/代理/权限边界，不用 toy 代替。
3. 继续检查是否存在 GPT-5.6 专属技术报告、system card、完整权重和目标 runtime；没有一手证据就保持 `unverified`，不新增重复 Transformer 章节。
4. 已完成第十七、二十和二十四册已有状态/工具/缓存章节的 toy 入口同步；后续运行全库 Python、Markdown 围栏、相对链接和 `git diff --check` 检查。goal 保持 `active`。

## 2026-09-23 OpenAI gpt-oss 当前时点复验与下一步

本轮使用 `10.24.27.134:7890` 重新验证百度、Artificial Analysis 中文首页和 DataCurve，均 HTTP `200`。AA 中文首页为 `1,783,626` bytes / SHA-256 `0580fad58c167fb96e87289addbabccd40cbd60302c527437805c8eb6ee7530b`，DataCurve 为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。八家重点厂商没有新的 canonical 模型，因此活动锚点切换为两个排行榜已有的 `gpt-oss-120b` / `gpt-oss-20b`；不从 OpenAI Cookbook、GitHub、Hugging Face 或论文另发现模型。

AA 详情当前快照为：120B `4,078,295` bytes / SHA-256 `43e5f13a3e58976b077acac1810bce82ac60ce33a7d81476f0dc2b17175532de`，20B `4,083,068` bytes / SHA-256 `64d67047a5964c0109f103108f9b77c64edbaf1ee17f3fec3c82ffccd7cea2cc`。当前第三方/provider 字段为：

- `gpt-oss-120b (high)`：117B total、5.1B active、131,072 context、Intelligence Index `11.6028431512592`、约 `196.389235173389 tokens/s`、cost/task `0.10742452290394947`。
- `gpt-oss-20b (high)`：21B total、3.6B active、131,072 context、Intelligence Index `8.9675171856126`、约 `185.656167974395 tokens/s`、cost/task `0.012460640242179213`。

上述字段只属于 AA/provider 当前采集时点，不替代官方规格，也不证明模型 revision 或架构升级。DataCurve 当前仍没有精确 `mini_swe_agent_gpt_oss_120b_*` / `mini_swe_agent_gpt_oss_20b_*` 行，不迁移其他 OpenAI/Codex 模型的 Agent 分数；该缺失只作为当前快照负证据。

7890 当前取得的官方 Cookbook 复核资料为：实现兼容性验证 `337,423` bytes / SHA-256 `e6500d57bc6041133a8f63acc03101260182ea58d45d2f7248806ae609c85474`，raw CoT 处理 `339,052` bytes / SHA-256 `1c08e3fb06a129256596c1af9ba907b7b3c435be3b61cfb00a72da83b1071fe6`。动态页面哈希变化没有产生新的模型 ID、权重 revision 或架构声明；面试主线继续是 Harmony/API shape、tool call/result 状态、raw CoT item/index/turn lineage、compatibility smoke test 与 AIME/GPQA/HealthBench 分层，随后才是 kernel/precision、目标硬件和生产 acceptance。

研究笔记、来源索引、模型盘点和榜单解释已经同步；第二十一册第 87 章及既有题库、练习、项目入口已覆盖本专题，不重复建章。当前状态为 **当前时点 AA 复验 + 官方 provider compatibility/raw-CoT evidence 的内容专题闭环**。下一步按 pinned revision -> runtime import -> CPU/meta shape -> 目标硬件 -> 数值/协议/生产 acceptance 推进；本机缺少完整权重、Transformers/vLLM/SGLang 和 NVIDIA 驱动时，保持 `unverified`，不下载大权重或虚构 benchmark。

## 2026-09-23 DeepSeek V4.1-Flash Harness preview 补证

本轮继续沿两个排行榜已经确认的 `deepseek-v4-1-flash`，没有从 DeepSeek Harness 文档另发现模型。使用用户确认可用的 `10.24.27.134:7890` 获取 DeepSeek Harness 官方 Preview 文档；其内容属于 V4.1 周边 runtime 资料，不能反推 V4.1 的参数、训练配方或内部网络结构。

已完成：

1. 固定 quickstart、providers、Python SDK、architecture/reference、MCP memory 和 GitHub review 六份官方文档的大小与 SHA-256；详细来源见 [`deepseek-v4.1-flash-source-notes.md`](research/model-update-2026-09/deepseek-v4.1-flash-source-notes.md) 和 [`source-index.md`](research/model-update-2026-09/source-index.md)。
2. 提取 provider ID/credential 脱敏、session model 固化、插件依赖与 cleanup、MCP namespace/secret 过滤/重连/注销，以及 GitHub signed webhook、异步 `202`、重复 delivery 和出站权限分离等面试主线。
3. 运行 [`deepseek_harness_protocol_audit.py`](research/model-update-2026-09/code/deepseek_harness_protocol_audit.py)：`ok=true`、`duplicate_webhook_admissions=2`、重连预算耗尽后 MCP 工具为空、`network_called=false`。证据等级为 `local_protocol_toy`，不代表真实 Harness/API、V4.1 质量或生产 SLO。
4. 已同步第二十一册第 81 章、第二十册第 19 章、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`PROJECTS.md`、`PAPERS.md` 和 `KNOWLEDGE_GRAPH.md`。

当前证据状态为 **AA 单榜内容专题 + 官方 API/runtime + Harness Preview + local protocol toy**。下一步仍按 pinned revision -> runtime import -> 完整权重 -> 目标硬件 -> 数值/工具/verifier/SLO 推进；若本机无完整权重、目标 runtime 或 NVIDIA 驱动，保持 `unverified`。模型发现入口继续严格限制为 Artificial Analysis 与 DataCurve，重点厂商继续限制为 OpenAI、Anthropic、GLM、Qwen、Kimi、DeepSeek、Gemini、Grok。

## 2026-09-23 Qwen3.8 Max (0902) Context Cache 协议补证

本轮仍只沿两个排行榜已经确认的 `Qwen3.8 Max (0902)` 推进，没有从 QwenCloud 文档另发现模型。使用用户确认可用的 `10.24.27.134:7890` 重新获取 QwenCloud Thinking、Function Calling、Context Cache 和 Dynamic Rate Limiting 页面，HTTP 均为 `200`；Context Cache server-rendered response 为 `1,133,845` bytes / SHA-256 `42d5d39fe4cb29680a27fcdd8d1b9bb1c07cdea93f01cb2ae31f40b02ca2edb8`，Dynamic Rate Limiting 为 `411,250` bytes / `b517a331efcdd822bfd63e81894e735a686db01baf1e77702155a0d8905ce82f`。上一轮短响应只暴露导航壳，本轮以完整 `initialState` 正文为准。

新增官方运行时知识点：

1. Explicit cache 的 `cache_control.type` 只能是 `ephemeral`；单请求最多四个 marker，超过四个只有最后四个生效；可缓存前缀至少 `1,024` tokens。
2. Explicit cache 的 backward prefix matching 有 20 个 content-block lookback 边界；并行 tool call 若把每个 tool result 拆成独立消息，会更容易超过窗口并 miss。连续同角色结果合并可以改善命中机会，但不保证命中。
3. Explicit/session cache 通常 5 分钟有效且命中刷新；implicit cache 自动启用、命中不保证、没有固定 TTL而由 provider 清理长期未使用数据。文档典型相对计费是 explicit/session 创建 `125%`、命中 `10%`，implicit 创建 `100%`、命中 `20%`；需与产品页当前美元价格和 usage 分开记录。
4. Session cache 使用 Responses API、`x-dashscope-session-cache: enable` 和 `previous_response_id`；账户/模型隔离必须进入 lineage。命中量读取 `usage.input_tokens_details.cached_tokens`，总 `input_tokens` 可能包含 marker 后通常少于 10 个 provider token，不能简单用加减式推导 prefill、KV 字节或账单。

已新增并运行 [`qwen_max0902_cache_contract_audit.py`](research/model-update-2026-09/code/qwen_max0902_cache_contract_audit.py)：输出 `ok=true`、`effective_marker_count=4`、显式命中/TTL/lookback/隔离、implicit 非保证命中、session `cached_tokens=1024`、thinking forced-tool 与 reasoning budget 冲突负例均通过；`network_called=false`，证据等级为 `local_protocol_toy`。已同步 Qwen 研究笔记、第二十一册第 83 章、第二十四册第 32 章、来源索引、模型盘点、榜单解释、论文、题库、练习、项目和知识图谱。

当前状态仍为 **Qwen3.8 Max (0902) hosted revision 资料级闭环 + Context Cache 官方协议补证 + local protocol toy**。没有新增 open checkpoint、0902 专属架构/训练报告、完整权重、生产 kernel、真实缓存命中率、线上 tool acceptance 或硬件 profiling。下一步按 pinned revision -> runtime import -> 完整权重 -> 目标硬件 -> 数值/工具/verifier/SLO 门禁推进；模型发现入口仍严格限制为两个排行榜，goal 保持 `active`。

## 2026-09-23 Claude Fable 5.1 状态协议门禁

本轮使用 `10.24.27.134:7890` 复抓两张模型发现榜和 Fable 详情：Artificial Analysis 中文首页为 `1,783,769` bytes / SHA-256 `e1acf794bd45f380ce3220a20c68de7a274f50ff7eb162a527a9faf4d58a166a`，DataCurve 为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，Fable 详情为 `4,008,017` bytes / `4ea24782d05bcaae7d31f3cf348e5a573851678998706a9df82fa34679292f96`。八家重点厂商按 canonical slug 去重后没有新模型；本轮只沿 `claude-fable-5-1` 推进。

7890 当前取得 Anthropic 官方 Markdown：overview `14,954` bytes / `13e8aeb6bbd207311ac032916f2ab37e7a454c9d752c027cf76967a5c958a076`，What's new `37,545` / `59d2a26f6e123d009a9e86aac9283705f7bfd15549b185dc07aa1858241baf19`，migration guide `94,368` / `c1d7bd16475ce9ad03ad556cc363635d93e695ecafc293e6acb85023e9f869b6`。官方正文再次确认 1M/128K、adaptive always-on、manual/disabled thinking 与 prefill 的 400、forced tool 的 400、方向性的 thinking block 可读性、prefix mismatch 的 error/drop 行为、per-message effort、turn-scoped system、progress updates、content provenance，以及 Fable 5.1 的 30-day retention/ZDR 与非 Priority Tier 边界。System Card PDF 为 `16,397,488` bytes / `b0d59edc7a60eef32a879c13d713cce60c3fefd7e6b5183afdc8b835af3c8c39`，未漂移。

已新增并运行 [`claude_fable51_state_protocol_audit.py`](research/model-update-2026-09/code/claude_fable51_state_protocol_audit.py)：`py_compile` 和主流程通过，输出 `evidence_level=local_protocol_toy`、首次/重复回放副作用 `1/1`、重复回执为 true；负例覆盖 disabled/manual thinking、forced tool、prefill、旧模型读取 Fable thinking、prefix mismatch、进度无 tool result 和 provenance 不支持。脚本不联网、不调用 Anthropic、不解密 thinking，也不代表服务端实现或模型质量。

已同步 Fable 研究笔记、source index、model inventory、inventory interpretation、第二十一册第 19 章、题库、练习和项目。当前状态为 **AA + System Card/API contract + local protocol toy**；没有新增参数、架构、完整训练 recipe、独立 Fable 5.1 技术报告、精确 DataCurve Agent 行、目标硬件 profiling 或生产 SLO 证据。下一步按两榜当前 canonical 集合选择下一项未收口锚点，goal 保持 `active`。

## 2026-09-23 GPT-6 Sol：当前时点复验与合同审计 toy

本轮继续只沿两个排行榜已经确认的 `gpt-6-sol` 推进。通过 `10.24.27.134:7890` 重新取得 Artificial Analysis 中文首页（HTTP 200，`1,783,769` bytes / `e1acf794bd45f380ce3220a20c68de7a274f50ff7eb162a527a9faf4d58a166a`）、DataCurve（HTTP 200，`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`）和 GPT-6 Sol 详情（HTTP 200，`3,994,562` bytes / `3d08f91c71e2770a4ef84b593c2a197d711193bbcbd8c3a1f330393b65f0ea86`）。AA 当前仍显示 Intelligence Index `47.5276426437724`，median output speed `126.038858615917 tokens/s`；速度相对上一快照的差异只作为 provider/采集时点漂移。DataCurve 没有精确 `mini_swe_agent_gpt_6_sol_*` 行，不迁移其他 GPT 的 Agent 结果。

OpenAI 官方模型页、Reasoning、Agents、Tools、Compaction 和 Prompt caching 快照已更新到当前大小/哈希。新增 [`gpt6_sol_contract_audit.py`](research/model-update-2026-09/code/gpt6_sol_contract_audit.py)，无网络、无第三方依赖，主流程输出 `ok=true`、`evidence_level=local_protocol_toy`、`network_called=false`；它覆盖 `standard/pro` 与 effort 正交、合法/非法 `configuration_update`、预算 `incomplete`、272K whole-request 计价、permission/executor/verifier、idempotent replay、opaque compaction 和 cache prefix miss。

研究笔记、来源索引、模型清单、榜单解释、计划、进度、第二十四/二十/十七/十六/第六册相关章节和题库/练习/项目/论文/知识图谱已同步。当前状态为 **AA 单榜资料级闭环 + 当前时点复验 + local protocol toy**；参数、架构、完整训练/后训练 recipe、完整权重、目标硬件、生产 kernel、精确 DataCurve Agent 行、独立 benchmark 和线上 acceptance 仍为 `unverified`。下一步先跑全库 Python、Markdown 围栏、相对链接和 `git diff --check`，goal 保持 `active`。

## 2026-09-23：DeepSeek V4.1-Flash vLLM `v0.30.0` stable release

- 本轮继续沿 Artificial Analysis 已确认的 `deepseek-v4-1-flash` 锚点推进，没有从 vLLM/PyPI/SGLang 的仓库目录另发现模型。DataCurve 当前仍没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，因此不迁移 V4 Pro、V4 Flash 或其他 DeepSeek 版本的 Agent 结果。
- vLLM `v0.30.0` release 于 `2026-09-22T05:20:54Z` 发布，tag commit 为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。固定 [release API 快照](https://api.github.com/repos/vllm-project/vllm/releases/tags/v0.30.0) 为 `65,334` bytes / SHA-256 `bc5d0dee9296de133c54209afab0ae4eb9d2c7c3f331261f1dfdd4d0ab23a48`；[v0.30.0 registry](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/model_executor/models/registry.py) 为 `64,420` bytes / SHA-256 `a08a98aaae52ced32226aa647f58d682ac600b9572651a9b377b97846bc99212`，已包含 `DeepseekV41ForCausalLM` 与 `DSparkV41DraftModel`。
- v0.30.0 stable package 的固定源码证据为：`__init__.py` `625` bytes / `6b928f07c6f67f4fd1b599a143cd15a1309a177c877f7f08b29d5fa30db659fe`；`quant_config.py` `9,006` bytes / `bfc500c4989607809577cbd10512b96e9162a7359ad407f772b7f695eaa34cd9`；NVIDIA `vl_model.py` `13,728` bytes / `a5a3f477225990946093092d4781db181b59b52102aff2e0e345e631973817f1`；NVIDIA `dspark.py` `23,059` bytes / `4d9c2bfa4c123aa5b95b637f24dc3d04748227455857bdc1376b27cda8af5954`；ROCm `vl_model.py` `13,736` bytes / `6f3fcc8a5896432ef51f809348097e92c7782ab226adb5ecb32cdc599ce05af4`；ROCm `dspark.py` `22,731` bytes / `a109e581711a74a7c5597b3f5a07d81ed05aac0ed2619dfa3f859050efc0b9d9`。
- v0.30.0 release notes 的面试级新增点包括：DeepSeek V4.1-Flash 模型接入；SM100 FlashMLA V4.1 record 的 MXFP8 whole-KV；DeepGEMM Mega-mHC；mHC post block folded into delayed pre projection；Triton-fused input metadata preparation；CPU-offloaded Engram async prefetch 与 Engram DP sharding；DSpark draft states 在 sequence-parallel all-gather 前折叠；DSpark 未继承未初始化 EPLB state；V4.1 strict tool parameters 的 XGrammar 约束；Responses text parts；Vision-Exp 专用 image sentinel padding。上述是 release notes/runtime implementation evidence，不是 DeepSeek 独立 benchmark 或本机运行结果。
- [PyPI `vllm/0.30.0` metadata](https://pypi.org/pypi/vllm/0.30.0/json) 快照为 `13,218` bytes / SHA-256 `43020551808911e4cabfca5ea71951766c25101b3817a10d306d88fe42d860b8`；x86_64 wheel `314,883,777` bytes / `ef52ee58c410ead0b8afb190838fa4cbcb52075596f67862a03859d984966ac4`，aarch64 wheel `309,984,160` bytes / `eb3e11bab695d085098579a6eda2d602419adec3826ebfbcce9a3ffa543eb62e`，sdist `42,432,229` bytes / `5f8f4e890c042ffa1c3e103f81c35e2d96f60a0a175ac43a4adbae7006bef62b`。
- 当前状态更新为：**内容专题 + HF reference implementation + vLLM `v0.30.0` stable release/source evidence + SGLang main + recipe protocol evidence（AA 单榜）**。完整权重、GPU/ROCm/NPU 运行、真实 FP4 质量、candidate/index Top-K recall、DSpark acceptance/rollback、EPD、tool acceptance、目标硬件 profiling、独立 benchmark 和生产 SLO 继续保持 `unverified`；goal 保持 active。
- 已同步研究笔记、来源索引、模型清单、榜单解释、`plan.md`、第二十一册第 81 章，以及 `PAPERS.md`、`BOOK_SERIES.md`、`ROADMAP.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md` 和 `KNOWLEDGE_GRAPH.md`；没有新增排行榜之外的模型候选。

## 2026-09-23：排行榜复访失败与 vLLM release 离线审计

- 本轮重新访问两个唯一排行榜时，三条代理均无法建立连接：`10.237.126.170:1234`、`10.24.27.134:7890`、`10.24.27.134:8098` 对 Artificial Analysis 中文首页和 DataCurve DeepSWE 均返回 curl connection failure。没有把该结果解释为榜单不存在，也没有据此新增、删除或切换模型候选；恢复后仍需重新获取两个页面并记录快照哈希。
- 利用已保存的 vLLM `v0.30.0` release API 离线复核：`tag_name=v0.30.0`、`target_commitish=main`、`published_at=2026-09-22T05:20:54Z`，API 快照 `65,334` bytes / `bc5d0dee9296de133c54209afab0ae4eb9d2c7c3f331261f1dfdd4d0ab23a48`；release API 列出 9 个平台 artifact，包含 CUDA 12.9 x86_64 wheel `545,459,905` bytes / `e98cb69659bfcfc849cf11ce0781a7161d40b02b51a6c3636924a5909f2aabcc` 与 aarch64 wheel `519,981,036` bytes / `fdb57ab5fa1c3ac4c94a6eff52579df32cc9aab5da9863880d03e7167e088e77`。
- 离线源码审计确认 registry 中 `DeepseekV41ForCausalLM` 位于第 379--381 行，`DSparkV41DraftModel` 位于第 647--648 行；NVIDIA/ROCm vision wrapper 都声明 V4.1 target 类，DSpark 文件都包含 `mtp.*` 映射。该结果强化 stable source evidence，但仍不产生完整权重、数值正确性、acceptance 或生产 SLO 证据。
- release body 的周边面试点已同步到研究笔记和来源索引：Fast Start per-GPU weight-cache/CUDA IPC、HiSparse host-resident sparse-MLA tier、Model Runner V2 overlap，以及 online adaptive verification。它们归因于 vLLM runtime release，不作为 V4.1 独有模型技术或新排行榜模型。

## 2026-09-23 Claude Opus 5.5 新锚点与联网恢复

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、Z.ai/GLM、Qwen、Kimi、DeepSeek、Gemini、Grok。
- 本轮先复验三条代理：`10.237.126.170:1234`、`10.24.27.134:7890`、`10.24.27.134:8098` 访问百度均 HTTP 200。随后三条代理对 Artificial Analysis 中文首页和 DataCurve DeepSWE 均返回 HTTP 200 且逐字节一致。AA 首页为 `1,783,001` bytes / SHA-256 `2fd4bd27a526ee60d807b9b596f0f3a8a4faa60223d5a3a21fca1ea643ec2bbf`；DataCurve 为 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。此前短时连接失败只记录为代理路径故障，不再作为当前状态。
- AA 当前前列新增/展示了重点厂商 canonical 条目 `Claude Opus 5.5`，本轮选择 `claude-opus-5-5` 作为下一活动锚点；`Grok 4.7`、`GPT-6 Sol` 等也在页面中出现，但本轮不并行展开。没有把 effort/provider 变体重复计作基础模型。
- Artificial Analysis Opus 5.5 详情三代理逐字节一致：`3,824,824` bytes / SHA-256 `ed037387bd96b9242985d77657b3cd094000f080854882e0c05ab04d4abb9904`。字段为 `releaseDate=2026-09-22`、1M context、Intelligence Index `57.6223698102963`、`parameters=null`、proprietary、input/output `$4/$20`、cache hit `$0.20`；这些是 AA 第三方/provider 字段。
- DataCurve 当前没有精确 `mini_swe_agent_claude_opus_5_5_*` 行，因此不迁移 Opus 5、Fable 5.1 或其他 Claude 版本的 Pass@1、成本、输出 token、steps 和 Agent 结果。
- Anthropic 官方发布页 canonical URL 为 [`https://www.anthropic.com/claude-opus-5-5`](https://www.anthropic.com/claude-opus-5-5)，快照 `619,938` bytes / SHA-256 `1b4b50a9df4c7f24d786b811c88f372ef066e4cc71ed7d43a5a1b3509136f293`。官方正文确认：Opus 5.5 是 Claude 5.5 家族首个模型；发布方称典型成本比 Opus 5 低 40%，cache read 为 `$0.20/M`，强调更少 token/steps、agentic coding、computer use、knowledge work 和更快输出。
- 新知识已归纳为五条面试主线：`quality / successful_task_cost` 需要同时看 token、工具轮次、fallback 与 verifier；effort/max/medium 与 benchmark 成本必须分账；长任务 coding 的关键是 context gather、完整 patch、回归测试和 artifact verifier；Cyber/Life Sciences/Distillation 使用 capability routing 与 fallback；WANDR/OSWorld/Terminal/Automation 等是模型+工具+harness 结果，不是裸模型分数。
- 官方发布页还称 Opus 5.5 使用接近 Fable 5.1 的网络安全、生物和蒸馏 safeguards，部分 cyber 任务 fallback 到 Opus 4.8，biology/frontier LLM development 任务可能由 Opus 5 完成；preserved thinking 被称为 anti-distillation safeguard。必须记录实际执行模型、触发类别、权限、缓存、重试和最终 artifact，不能把 fallback 成功归因给 Opus 5.5。
- System Card PDF 已取得：`17,795,106` bytes / SHA-256 `7311c9c6bbb16d012f1c12c7418b05949fcf7ae3e30d2c40f22050074b2a7378`；PDF 标题确认是 Claude Opus 5.5 System Card，目录含训练过程、能力、生物/化学、网络安全、自治、Agent 安全、对齐、模型福利和白盒分析章节。当前环境没有 PDF 文本抽取器，不把目录或二进制字符串升级为具体安全数字。
- Platform model page 在 1234/8098 仍重定向到区域不可用页，旧快照为 `439,341` bytes / SHA-256 `e89f00b6cdf35d049751c086b15a98846e40ba4e6bce18a12777020d3395d577`；7890 成功取得真实 Markdown `3,184` bytes / SHA-256 `aa9389d5f328023dea11650d26898b43b4e04969a4facd327881df145a307927`。这是线路差异，不代表模型不存在。
- 7890 同时取得 What's new `5,972` bytes / `90fb6efa547a193cbf1eb4b836ef5310234da054f2e83abbe15ce41b0d4c5a6a`、Migration guide `4,740` bytes / `0c8f717ab25184446431b1c579905a2458e4917258cfb71d66b65737eb3feab5` 和 Fast mode `6,962` bytes / `c23e9562b6c76c530dd95ca51578a360bbf0304166839fef2f8ec38db746e913`。新核验点为 always-on thinking、forced tool choice 400、thinking block model/conversation binding、`computer_toolset_20260801`、progress-update blocks、on-demand compaction、inline tools、fast mode `usage.speed`/独立限流和 refusal/fallback 契约；这些是 API observable behavior，不升级为内部架构结论。
- 已更新 [`claude-opus-5.5-source-notes.md`](research/model-update-2026-09/claude-opus-5.5-source-notes.md)，并同步 `plan.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md` 及 reasoning/Agent/serving 正式章节。当前状态：**AA 单榜资料级闭环**；参数、架构、完整训练/后训练 recipe、精确 DataCurve Agent 行、生产 kernel、目标硬件 profiling、独立 benchmark 和线上 acceptance 待核验；goal 保持 active。

## 2026-09-23：GPT-6 Sol 活动锚点与 OpenAI runtime 补证

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、Z.ai/GLM、Qwen、Kimi、DeepSeek、Gemini、Grok。
- 三条代理取得 Artificial Analysis 中文首页和 DataCurve DeepSWE 的当前快照；首页为 `1,783,001` bytes / SHA-256 `2fd4bd27a526ee60d807b9b596f0f3a8a4faa60223d5a3a21fca1ea643ec2bbf`，DataCurve 为 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。AA 前列展示 `GPT-6 Sol`，本轮从 `Claude Opus 5.5` 切换到 `gpt-6-sol`，没有把 effort/provider 变体重复计作基础模型。
- Artificial Analysis [`gpt-6-sol`](https://artificialanalysis.ai/models/gpt-6-sol) 三代理逐字节一致：`547,729` bytes / SHA-256 `88841ae9837f189a6ab739d8fa2bad6d142163b5b6486cab9181ebbfbc75f116`。详情标题为 `GPT-6 Sol (max)`，max Intelligence Index `47.5276426437724`、median output speed `115.205383643174 tokens/s`、cost per Intelligence Index task `1.0564240894076389`；这些是 AA 第三方/provider 字段。
- DataCurve 当前没有精确 `mini_swe_agent_gpt_6_sol_*` 行，因此不迁移 GPT-6 Astra、GPT-5.6 或其他 GPT 版本的 Pass@1、成本、输出 token、steps 和 Agent 结果。
- OpenAI 官方模型页 [`gpt-6-sol`](https://developers.openai.com/api/docs/models/gpt-6-sol.md) 快照为 `1,664` bytes / SHA-256 `5566589803b6f0ac79c00125a7c1af9291e46d5724c1c414f932d36ab92a8ad0`。已确认 `gpt-6-sol`、complex coding/agentic workflows、text/image input、text output、1,050,000 context、922,000 maximum input、128,000 maximum output、2026-04-20 knowledge cutoff、`none`--`max` effort、Responses/Chat Completions/Batch 端点和 Responses 工具目录。
- OpenAI [Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md) 快照为 `70,315` bytes / `93327c5eb19df7ecf8c0bd9d581f58052dee9f090c38bb48906110b08d0ce251`；[Agents](https://developers.openai.com/api/docs/guides/agents.md) 为 `5,432` bytes / `df4f61b609550619a3f9e445c66d28318b9f681affe1a819ef0098cde32bf43a`；[Tools](https://developers.openai.com/api/docs/guides/tools.md) 为 `33,282` bytes / `4722fa102070178c1a1d603e718c69575c7c1768fa901218eab359bafdcba341`；[Compaction](https://developers.openai.com/api/docs/guides/compaction.md) 为 `14,272` bytes / `73fd2fd1afd44bd6f29ce7bd86fd0ae3413c98ec00ae5476879e98171b0b60dd`。
- 新知识已分为四条面试主线：`reasoning.mode` 与 `reasoning.effort` 是独立控制面；GPT-6 family 的 `configuration_update` 可以在会话中调整后续 effort；Responses/Agents API/SDK 的状态和执行所有权不同；server-side/standalone compaction 返回 opaque/encrypted 状态，不能当可读摘要或永久记忆。模型页工具列表只证明能力入口，不证明内部架构或训练算法。
- 已新增 [`gpt-6-sol-source-notes.md`](research/model-update-2026-09/gpt-6-sol-source-notes.md)，并同步 `plan.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、Reasoning/Coding Agent/Responses runtime/serving 正式章节、`BOOK_SERIES.md`、`ROADMAP.md`、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md` 和 `KNOWLEDGE_GRAPH.md`。当前状态：**AA 单榜资料级闭环**；参数、架构、完整训练 recipe、独立 benchmark、完整权重、目标硬件 profiling、tool acceptance 和生产 SLO 待核验，goal 保持 active。

## 2026-09-23：GPT-6 Luna 当前活动锚点与 OpenAI sibling/runtime 核验

- 三条代理重新获取 [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh)，逐字节一致为 `1,782,868` bytes / SHA-256 `24182a9f96da6b6b96567bc2d069eddb553109a6bb323d07d79d900bb107770c`；首页前列出现 `gpt-6-luna`。三条代理对 [Artificial Analysis GPT-6 Luna](https://artificialanalysis.ai/models/gpt-6-luna) 逐字节一致，为 `3,992,084` bytes / SHA-256 `cda7ec1a8b90bfb76c85624f31cab242a0779732f68d65641b0dee5929a5d0f1`。详情记录 release date `2026-09-22`、max Intelligence Index `37.2559686869738`、median output speed `153.87508473888 tokens/s`；这些是 AA 第三方/provider 字段。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 当前快照仍为 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，没有精确 `mini_swe_agent_gpt_6_luna_*` 行。因此不迁移 GPT-6 Sol、GPT-6 Astra、GPT-5.6 或其他 GPT 的 Pass@1、成本、输出 token 和 Agent steps。
- 7890 代理取得 [GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna.md)，`4,019` bytes / SHA-256 `561a86af72eced9a76a4e3a96c45ae7fff7782732514a8429b55ebddb4404945`；确认 `gpt-6-luna`、focused/high-volume 定位、text/image input、1,050,000 context、922,000 maximum input、128,000 maximum output、2026-05-18 knowledge cutoff、`none`--`max` effort、Responses/Chat Completions/Batch、工具目录和 `$0.10/$0.50` token 价格。1234/8098 对该官方页返回 403，作为线路访问边界记录。
- Reasoning、Agents、Tools、Compaction 官方快照与上一轮 GPT-6 Sol 完全一致：分别为 `70,315`/`5,432`/`33,282`/`14,272` bytes，哈希不变。由此复用 GPT-6 family 的 mode/effort 分离、`configuration_update`、runtime ownership、tool capability surface 和 opaque compaction replay；不把通用文档写成 Luna 专属训练或架构披露。
- 已新增 [`gpt-6-luna-source-notes.md`](research/model-update-2026-09/gpt-6-luna-source-notes.md)，并同步 `plan.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、第六/十六/十七/二十/二十四册相关章节、`BOOK_SERIES.md`、`ROADMAP.md`、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md` 和 `KNOWLEDGE_GRAPH.md`。当前状态：**AA 单榜资料级闭环**；参数、架构、完整训练 recipe、独立 benchmark、完整权重、目标硬件 profiling、tool acceptance 和生产 SLO 待核验，goal 保持 active。

- 本轮质量门禁已通过：`git diff --check` 通过；全库 Markdown fenced block 平衡；GPT-6 Luna 新增相对链接均存在；三个 AA 详情快照、OpenAI model page 和 DataCurve 快照哈希与研究笔记一致。下一轮仍先重新抓取 Artificial Analysis 与 DataCurve，再从八家重点厂商中选择未闭环或新出现的 canonical 条目；不会从 OpenAI 官方目录另发现模型，也不会迁移缺失的 Agent 分数。

## 2026-09-23：Claude Opus 5.5 System Card 正文解析与资料级闭环升级

- 工作目录仍为 `/data/zzc/llm-from-zero-to-interview`，等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、Z.ai/GLM、Qwen、Kimi、DeepSeek、Gemini、Grok。
- 本轮复核三条代理的最新续抓快照：Artificial Analysis 中文首页 `1,783,592` bytes / SHA-256 `7880a715fd069d4e475e4d717b37971d1f6402e5492964bb3f58787524d0b5a4`；此前同日 `1,783,001` bytes / `2fd4bd...` 作为历史快照保留。DataCurve DeepSWE `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，三代理逐字节一致。
- `Claude Opus 5.5` AA 详情仍为 `3,824,824` bytes / SHA-256 `ed037387bd96b9242985d77657b3cd094000f080854882e0c05ab04d4abb9904`，release `2026-09-22`、1M context、Intelligence Index `57.6223698102963`、约 `$4/$20` input/output；DataCurve 没有精确 `mini_swe_agent_claude_opus_5_5_*` 行，不迁移 Opus 5、Fable 5.1 或其他 Claude 版本的 Agent 结果。
- System Card PDF 固定为 `17,795,106` bytes / SHA-256 `7311c9c6bbb16d012f1c12c7418b05949fcf7ae3e30d2c40f22050074b2a7378`。此前“没有 PDF 文本抽取器、只保留目录”的记录已由本轮 Node.js `zlib` + ToUnicode/CMap 只读解析更正；原始 PDF 哈希不变。
- 正文新增训练数据边界：公开互联网、公开/私有数据、获许可用户数据和合成数据的组合；去重/分类与 ClaudeBot；不访问密码页、登录页或 CAPTCHA 页面；knowledge cutoff 为 2026 年 6 月。参数规模、层数、dense/MoE、优化器、完整训练/后训练 recipe 和 adaptive thinking 内部机制仍未公开。
- 正文新增能力证据：Terminal-Bench 4.0 `66.36%`（xhigh、Claude Code `--bare`、5 trials；约 2.5% requests fallback，影响约 10% trials）、ProgramBench `91.2%`（166 golden tasks）、OSWorld 2.0 partial/strict `81.8%/48.7%`（108 tasks、1080p、最多 500 actions、5 runs、完整截图、>100K tokens 后 server-side compaction）。五 Agent team 在 ProgramBench 同分比较约 `2.7x` derived-latency improvement；DRACO 在 `0.5x` latency budget 下五 Agent team 约 `2.8x` speedup；100-agent task 运行 24 小时并出现 flat/hierarchical sub-lead 组织。
- 正文新增 RSP/安全证据：CB-1、未达 CB-2；autonomy threat model 1 适用但整体 low，threat model 2 未达；CoBench 2.1 Opus 5.5 `55.8%`、Mythos 5.1 `53.4%`、Opus 5 `53.2%`；AECI `169.36`，global CI `[165.23, 177.05]`、local CI `[167.99, 171.24]`，新 fit 为 374 benchmarks、7,985 observations、732 models。
- Cyber 结果明确关闭 safeguards：ExploitBench `13.99` flags、`91%` Cap%、完整 ACE `301/410 = 73.4%`；CyScenarioBench `67.6%`；ExploitGym 2h/6h `289/869`、`300/869`。Agent safety 为 Claude Code malicious refusal `79.8%`、dual-use/benign success `99.8%`、malicious computer-use refusal `79.46%`；Gray Swan IPI k=1/10/15 `0.1%/0.7%/1.0%`，约 18% fallback 到 Opus 4.8 且 1,310 个 fallback rollout 无成功攻击；coding Shade 无 safeguards/probe 为 `54.61%/11.13%`，computer-use probe `0.04%`，browser auto `0/110`。
- fallback 规则按类别分账：biology/CB -> Opus 5，cyber -> Opus 4.8，frontier LLM kernel 等窄范围 AI R&D -> Opus 5；conventional weapons/high-yield explosives 与 distillation/hidden reasoning extraction 无 fallback，其他 provider 可能不同。alignment 评测还包含 sandbox escape `1.5%`（均 low severity）、早期/最终 snapshot 的粘贴恶意指令 `52%` -> `2%`、不可见字符 `26%` -> `7.4%`，产品 mitigations 后为零；这些数字必须绑定 snapshot、safeguard、模拟环境和 verifier。
- 已同步 `claude-opus-5.5-source-notes.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`PAPERS.md`、第七/八/十六/十七/二十/二十四册相关章节和 `INTERVIEW_BANK.md`。当前状态升级为 **AA 单榜 + System Card 正文证据**；仍不新增 Transformer 架构章节，DataCurve 精确 Agent 行、参数/架构、完整 recipe、独立复现、目标硬件 profiling、生产 kernel 和线上 acceptance 继续保持待核验，goal 保持 active。

## 2026-09-23：下一活动锚点切换为 DeepSeek V4.1-Flash

- 最新 Artificial Analysis 中文首页续抓为 `1,783,592` bytes / SHA-256 `7880a715fd069d4e475e4d717b37971d1f6402e5492964bb3f58787524d0b5a4`，DataCurve DeepSWE 为 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；三条代理内容一致。AA 前列已有 canonical `DeepSeek V4.1 Flash (max)`，第三方 Intelligence Index 为 `39.456167472527`；本轮没有新的八家重点厂商模型名称。
- Opus 5.5 已完成本轮 System Card 正文同步，因此活动锚点切换到已有的 `deepseek-v4-1-flash`，不是从 vLLM、SGLang、HF 或官方目录另发现模型。当前研究笔记已经覆盖 HF revision、技术资料、reference inference、vLLM `main`/`v0.30.0` stable release surface、SGLang source、DSpark/视觉边界和 recipe 协议。
- 下一轮的闭环门禁为：完整权重加载、FP4/FP8 量化误差、candidate/index Top-K recall、DSpark draft/target acceptance 与 rollback、EPD/TP/EP/DP 调度、视觉 token/cache 一致性、目标硬件 profile、tool/verifier acceptance 和生产 SLO。release registry、wheel、README 或榜单分数不能越级替代这些结果。
- DataCurve 当前没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，不迁移 DeepSeek V4 Pro、V4 Flash、V3.2 或其他 DeepSeek 版本的 Agent 成绩。goal 保持 active，下一轮仍不新增重复 Transformer 架构章节。

## 2026-09-23：DeepSeek V4.1-Flash 官方 API contract 补证

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；本轮没有从 API 文档、HF、vLLM、SGLang 或 recipe 另发现模型，活动锚点仍是两个排行榜已确认的 `deepseek-v4-1-flash`。
- 通过 `10.237.126.170:1234` 取得并固定官方 Pricing、Rate Limit、Error Codes、Vision、Files、Responses API 和 Tool Calls 页面：大小/SHA-256 依次为 `23,149/2fecee48...a4506a`、`35,133/1190b37c...26413a`、`20,387/0df2698a...5094a`、`78,174/a805d8a...7dac7`、`61,828/1020efca...40345`、`56,250/1719ac1b...1e3ca2`、`70,636/5ee72ac...b4027`；完整哈希已写入研究笔记和来源索引。
- 当前可用于面试的 API 知识点已收口：`deepseek-flash` 对应 V4.1-Flash；1M context、384K max output、最高 2500 concurrency；Vision 的 URL 8192 字符/60 秒/32 MiB、`file_id` 64 MiB、请求体 48 MiB、600 图和约 1024 image tokens/图；Files 的 `purpose=user_data`、64 MiB、保存期、25 GiB/10,000 文件；Responses stateless semantic SSE、递增 `sequence_number`、completed/incomplete/failed 终态和不发送 `[DONE]`；图像 tool output 回灌；`/beta strict=true` schema 约束。
- 证据边界已明确：以上是 endpoint/provider contract，不是 V4.1 内部架构、完整权重、生产 kernel 或质量 benchmark；`function_call_output` 代表可回灌，不代表模型拥有权限；strict schema 代表结构校验，不代表业务 verifier 通过；`requested_model`、alias 和响应 `served_model` 必须分账。
- 已同步 [`deepseek-v4.1-flash-source-notes.md`](research/model-update-2026-09/deepseek-v4.1-flash-source-notes.md)、第二十一册第 81 章、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md) 和 [`plan.md`](plan.md)。下一步先做不收费的 parser/error/retry/idempotency toy，再按完整权重、目标硬件和 verifier 门禁推进；完整权重加载、FP4/FP8 误差、Top-K recall、DSpark acceptance/rollback、EPD、视觉 cache、tool acceptance、独立 benchmark 和生产 SLO仍为 `unverified`，goal 保持 active。

## 2026-09-23：DeepSeek V4.1-Flash API contract toy 完成

- 新增 [`deepseek_v41_api_contract_audit.py`](research/model-update-2026-09/code/deepseek_v41_api_contract_audit.py)。脚本只使用 Python 标准库和合成输入，不联网、不调用付费 API、不加载模型权重。
- 已验证 semantic SSE：把事件按 5 字符切块，仍得到 3 个有序事件、文本 `contract audit` 和 `response.completed`；序号乱序、终态后事件和非 JSON data 拒绝。
- 已验证工具四级账本：`schema_valid -> authorized -> executed -> verified`。一次预执行 transient failure 后第 2 次尝试成功；相同 idempotency key 的重复调用返回 duplicate receipt，side effect 计数仍为 1；额外字段触发 `SchemaError`，越权路径触发 `PermissionDenied`。
- 文本和 `input_image/file_id` 两种 `function_call_output` 结构均能生成 typed observation；`network_called=false`。这只升级为 **local protocol toy evidence**，不升级真实 endpoint、strict enforcement、模型质量、工具权限、完整权重、目标硬件或生产 SLO。
- 已同步研究笔记、第二十一册第 81 章 `81.13.9`、来源索引、模型清单、榜单解释和 `plan.md`。下一步检查本机 full-weight/runtime/hardware 条件；不可运行时维持 `unverified`，并按两榜既有 canonical 条目切换下一周边锚点，goal 保持 active。

## 2026-09-23：Gemini 3.8 Flash Interactions state/signature 与 thinking budget 复验

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、Z.ai/GLM、Qwen、Kimi/Moonshot、DeepSeek、Google/Gemini 和 xAI/Grok。本轮没有从 Google 官方目录另发现模型，活动锚点切换到已有排行榜 canonical `gemini-3-8-flash`。
- 通过 `10.24.27.134:7890` 获取 Google 官方五页快照：model page `103,745` bytes / `e609ede21f99462207a0552cde54c8a7f7462e23a28a7515fb774c84f8f16f2b`；Thinking `226,287` / `a443bebc66c224284ea8f18ab8f6576fc4aebeb9d3419113be722da4bf7feef8`；Interactions `105,650` / `ee250076b4cb66072a5e98951cc2c7d2003b0ae6fce3b5cce398821443b0a553`；Tool combination `124,824` / `038adc9426510e7f15075d966186d7689817a06fd3c3be1c52100a3f397991ff`；Thought signatures `87,723` / `6b269d05196fef00a5837fb9aa4e42d31034acd072f6bb6064fd91b6a6c89e8c`。1234/8098 对 Google 文档超时，作为线路边界记录；不把失败解释为页面不存在。
- Thinking 表精确确认 `gemini-3.8-flash` 默认 `On (medium)`，支持 `low/medium/high`，不支持 `minimal`；`max_output_tokens` 是 thought + visible output 的硬上限，思考阶段触顶返回 `incomplete` 或截断/空输出，已产生的 thought tokens 仍计费；`total_thought_tokens` 应与 visible output 分账。
- Interactions 页面精确确认：Interaction 是按时间排列的 execution steps；默认 `store=true`；付费层保留 55 天、免费层 1 天；付费项目可配置 7/14/28/55 天日志删除，也可以按 interaction ID delete；`store=false` 不能配 background execution 或后续 `previous_interaction_id`。`previous_interaction_id` 只保留 history，tools、system instruction、generation config 是 interaction-scoped；stateful/stateless 均支持 implicit caching；跨模型 continuation 必须检查输出模态兼容性。
- Thinking 与 Tool combination 两页对 signature 位置存在可见差异：前者把 Interactions signature 限定在 thought/built-in tool steps，后者将 Gemini 3+ tool call/result signatures 描述为 tool context circulation 的普遍字段。已在研究笔记中保留冲突，工程实现要求原样保存实际返回的 opaque fields，并以具体 endpoint/schema/SDK capability probe 为准；function call/result 的 `id` 对齐始终是硬门禁。
- Tool combination 还确认 built-in/custom tool 可组合，Search/Maps/URL/File Search 属于 server-side，Code Execution 有独立 server-side steps，Computer Use/custom function 属于 client-side；工具组合要求 `validated`，不支持 `auto`。宿主仍负责权限、执行、超时、幂等、沙箱、回灌和业务 verifier；工具字段不是模型内部架构证据。
- 新增 [`gemini_interactions_replay_demo.py`](research/model-update-2026-09/code/gemini_interactions_replay_demo.py)，无依赖、无网络、无真实 API/权重。运行通过：`ok=true`、15 个有序 toy events、stateful parent history、stateless signature preservation、tool id alignment、paid/free `55/1` retention、删除后拒绝 continuation、`network_called=false`。这是 **local protocol toy evidence**，不升级真实 endpoint、signature 验证、模型质量、billing、硬件或生产 SLO。
- 已同步 [`gemini-3.8-flash-source-notes.md`](research/model-update-2026-09/gemini-3.8-flash-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan.md`](plan.md) 和本进度表；书系/题库/练习/术语/项目/知识图谱的 Gemini 3.8 已补入 retention、parameter scope、signature conflict 和 toy replay 门禁。
- 当前状态仍为**资料级闭环**：没有独立 3.8 参数/架构/训练报告、真实 API capability probe、low/medium/high 独立消融、长上下文/cache billing、完整权重、目标硬件 profiling、独立复现或线上 acceptance；不新增重复 Gemini Transformer 正式章节。goal 保持 active。

## 2026-09-23：Kimi K3 当前时点复验与面试边界补证

- 工作目录继续为 `/data/zzc/llm-from-zero-to-interview`，等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、Z.ai/GLM、Qwen、Kimi/Moonshot、DeepSeek、Google/Gemini 和 xAI/Grok。本轮没有从 Kimi 官方仓库、HF、vLLM 或 SGLang 另发现模型，活动锚点切换到已有排行榜 canonical `kimi-k3`。
- 三条代理对 [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh)、[Kimi K3 详情](https://artificialanalysis.ai/models/kimi-k3) 和 [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 均成功且逐字节一致：首页 `1,783,605` bytes / SHA-256 `7a954aed8916ec9c5c88380cfc159274573dc9547182032ca142f556c0fc2916`；K3 详情 `4,080,091` bytes / `340a36ac5922f354aa42553fe3da2ce4acf4231bbb6af97fd6a88b974dbea4ee`；DataCurve `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。八家重点厂商按 canonical slug 去重后没有新增模型。
- AA K3 `max` 当前记录 Intelligence Index `43.5938229518782`、median output speed `36.9982439081499 tokens/s`、cost per Intelligence Index task `2.0001323004425493`、1M context。DataCurve 精确行 `mini_swe_agent_kimi_k3_max` 为 `309/451`、Pass@1 `0.6851441241685144`、Pass@4 `0.8938053097345132`、平均成本 `$4.654682129933482`、平均输出 `81,499.84` tokens、平均 Agent steps `97.5876`、4 runs/113 tasks；已明确绑定 `mini-swe-agent + tools + environment + verifier`，不迁移为裸模型分数。
- K3 README 复验为 `45,004` bytes / SHA-256 `849a303d849486aac61d1f0c253e3a1148ca59e801be807151c2dc3cce57ccb2`，HF metadata 为 `9,436` bytes / `6dfa5b1680074f4dfdd79a36b3342a191aacc32e15a9bd4a3895b98570773939`，revision 仍为 `f831ab66814297da540d832a5235f8e904f29d06`。固定 config、vLLM recipe、FlashKDA 证据未漂移；SGLang main 当前为 `171,130` bytes / `9af22b45f8d8f8a5931c3bd60310316090973fd5cf9626aabd2a618a9e4baa6d`，相较前一快照只有 import/type annotation 变化，没有实质 runtime 技术变化。
- 面试限制已补入研究笔记：约 `2.5x scaling efficiency` 是发布方声明而非独立复现；preserved thinking history 需要结构化原样回传；跨模型切换可能不稳定；excessive proactiveness 是官方限制；发布评测混用 Kimi Code、Claude Code、Codex、H20/H100 与 compaction 条件，不能拼接为统一排名。
- 已同步 [`kimi-k3-source-notes.md`](research/model-update-2026-09/kimi-k3-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、`plan.md`、第 15/88 章、题库和练习；K3 manifest audit、研究代码编译、协议 toy、Markdown 围栏、相对链接和 `git diff --check` 已通过。当前状态为**内容专题闭环 + 当前时点榜单/官方 revision 复验**；完整权重、目标硬件 profile、hybrid cache recovery、tool/verifier acceptance 和生产 SLO 仍为 `unverified`，goal 保持 active。

## 2026-09-23 Claude Fable 5.1 System Card 正文解析与证据闭环升级

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、Z.ai/GLM、Qwen、Kimi/Moonshot、DeepSeek、Google/Gemini 和 xAI/Grok。本轮没有从 Anthropic System Card、Research 页面或外部论文另发现模型。
- Fable 5.1 最新 Artificial Analysis 详情为 `4,006,915` bytes / SHA-256 `d80b226b7a4756cac93d6a065f400f39c9f756b971b827ecf0efbd5a67bffaeb`；canonical `claude-fable-5-1`、`releaseDate=2026-09-01`、`deprecated=false`，Intelligence Index `53.3549259623252`、median output speed `65.4865856934115 tokens/s`、median TTFT `298.446428812s`、1M context、cost per Intelligence Index task `7.629706364004841`。这些是第三方 provider/configuration 观察值，不是裸模型能力或统一 API 延迟。
- 本轮重新取得 AA 中文首页 `1,783,605` bytes / SHA-256 `7a954aed8916ec9c5c88380cfc159274573dc9547182032ca142f556c0fc2916`、Anthropic 发布页 `534,503` bytes / SHA-256 `610f2e5cf15100fdc85edf0c4bee750878cdbf2db3520606aed088f757bd33f2`、System Card PDF `16,397,488` bytes / SHA-256 `b0d59edc7a60eef32a879c13d713cce60c3fefd7e6b5183afdc8b835af3c8c39` 和 DataCurve `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。DataCurve 仍没有精确 `mini_swe_agent_claude_fable_5_1_*` 行，不迁移 Fable 5 的 `316/452` 或其他 Claude 的 Agent 结果。
- 新增 [`pdf_text_extract.js`](research/model-update-2026-09/code/pdf_text_extract.js)，使用 Node.js `zlib` 与 PDF ToUnicode/CMap 做只读文本抽取；正文 `/tmp/fable51-system-card-extracted-20260923.txt` 为 `376,181` bytes、SHA-256 `d61d0a99b770ae104150ceb2ee04e0ee02fc4a728d7521ee0319b3742ed091e5`。解析器检查通过，原始 PDF 未改写。
- System Card 正文确认 Fable 5.1/Mythos 5.1 共享相同模型权重，差异主要在 safeguards/访问计划；训练数据包括公开互联网、公共/私有数据和合成数据，使用 deduplication、classification 和 ClaudeBot，knowledge cutoff 为 2026-06。RSP 针对 Mythos 判为 CB-1、未达 CB-2；autonomy threat model 1 适用但整体 low，threat model 2 未达。参数规模、层数、dense/MoE、优化器和完整训练/后训练 recipe 仍未公开。
- 正文代表性能力数字已入研究笔记：Terminal-Bench 4.0 `55.8%`、Terminal-Bench-Science 0.1 `52.6%`、CursorBench `73.4%`、OSWorld partial/strict `77.9%/41.7%`、GDPval-AA v2 `1853`、AutomationBench `31.4%`、ProgramBench `87.6%`。这些是 Anthropic 发布方条件结果，必须绑定 snapshot、effort、tools、safeguards、fallback、任务环境和 verifier；不升级为本地复现或 Fable 裸模型综合分数。
- 安全证据已补入：Gray Swan IPI `k=1/10/15` 为 `0.1%/0.7%/1.0%`，较强 Shade coding attacker 无 probes/启用 probes 为 `56.87%/12.80%`，browser auto 为 `0/110`，无 safeguards browser-use 对照为 `2.64%`。System Card 还披露 rare permission-hook/classifier bypass、sandbox vulnerability、白盒内部分析、SHADE/LinuxArena covert behavior 与 CoT controllability；这些结果需保留 `actual_model`、`fallback_reason`、safeguard state 和 verifier，不能泛化成生产攻击率。
- 已同步 [`claude-fable-5.1-source-notes.md`](research/model-update-2026-09/claude-fable-5.1-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan.md`](plan.md)，并补充第四册百科、评测证据、安全 fallback、`INTERVIEW_BANK.md` 和 `EXERCISES.md` 的 System Card 条件。Fable 5.1 当前状态升级为 **AA + System Card 正文证据闭环**，不新增独立 Transformer 架构章节，goal 保持 active。
- 下一步门禁：把 System Card、AA、DataCurve 三类分数分账；设计 thinking block/forced tool/history edit/updates/provenance 的本地协议回放；继续保持真实 API capability probe、完整权重、目标硬件 profile、独立复现、tool/verifier acceptance 和生产 SLO 为 `unverified`。下一轮先复抓两榜，再从八家重点厂商已有 canonical 条目选择锚点。

## 2026-09-23 GLM-5.3：SAO 关联技术与长轨迹 RL 补证

- 工作目录确认：`/data/zzc/llm-from-zero-to-interview` 等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree。本轮没有从论文、官方目录、HF、vLLM 或 SGLang 另发现模型，活动锚点仍是两个排行榜已有的标准 `GLM-5.3`。
- 两榜当前快照已重新固定：Artificial Analysis 中文首页 `1,783,597` bytes / SHA-256 `ce87eece01c108877686e76d76f7e26d463416d0a6cd895e554e284549a286ad`；GLM-5.3 详情 `4,074,151` bytes / SHA-256 `805918eef4c32b3cdeaf482f316b3311016b0523aef4a24886700e02e906ca73`；DataCurve `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。AA 当前记录 Intelligence Index `44.777392385614`、median output speed `57.0553482500823 tokens/s`、cost per Intelligence Index task `2.0056375150449584`、1M context；这些是第三方/provider 配置字段。
- DataCurve 精确行是 `model=glm-5-3`、`harness=mini-swe-agent`、`reasoning_effort=max`、`config=mini_swe_agent_glm_5_3_max`：`311/451`、Pass@1 `0.6895787139689579`、Pass@4 `0.8761061946902655`、平均成本 `$3.9933584893126386`、平均输出 `80435.60975609756` token、平均 `124.47228381374723` Agent steps。继续绑定工具、任务集、环境和 verifier，不写成裸模型能力。
- 新增官方 Z.ai Markdown：GLM-5.3 `23,446` bytes / SHA-256 `9545c3d6fb1cabfa5951928bbe9a535e6958561d3bd6b5f4354cd5d3b0f6c929`；GLM-5.2 `25,293` bytes / SHA-256 `eeca971119068b0619137ce9c9d3c9362131baf16fdab0c067eefe5802919023`。确认 GLM-5.3 沿用 GLM-5.2 基础模型，改进来自 post-training，并继承 `SAO with compaction`；没有公开 5.3 专属 compaction 数学/序列化实现。
- 新增 SAO 论文证据：[arXiv:2607.07508](https://arxiv.org/abs/2607.07508)，HTML `190,454` bytes / SHA-256 `953b8968fa30d5f579a9650cc17d95521515ccac6cc7408b2a45c8126651822d`，PDF `664,828` bytes / SHA-256 `44c695be0428c666d06c914ba76c037e3ac77eeb5db0a81bbe239719c21bda48`。已核验 single-rollout immediate update、rollout logprob + DIS、双侧 token clipping/masking、critic `K=2`、frozen-attention value model 和 Skip-Observation GAE。
- 归因边界：论文主干实验使用 Qwen3-30B-A3B，摘要称 SAO 部署到 GLM-5.2（750B-A40B）Agent RL pipeline；不能把 Qwen3 实验、GLM-5.2 结果、论文算法或 DataCurve/AA 结果迁移为 GLM-5.3 独有能力。当前状态升级为 **GLM-5.3 双榜资料级闭环 + SAO 关联论文算法证据**。
- 已同步：[`glm-5.3-source-notes.md`](research/model-update-2026-09/glm-5.3-source-notes.md)、[`glm-5.2-source-notes.md`](research/model-update-2026-09/glm-5.2-source-notes.md)、第二十一册第 86 章、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md` 和 `KNOWLEDGE_GRAPH.md`。
- 待核验门禁：GLM-5.3 专属 compaction 状态/序列化、完整 post-training recipe、完整权重加载、FP8/量化误差、index/evidence recall、MTP acceptance、目标硬件 profile、独立 benchmark、tool/verifier acceptance 和生产 SLO。goal 保持 `active`。

## 2026-09-23 SAO local protocol toy 完成

- 新增 [`sao_async_rl_toy.py`](research/model-update-2026-09/code/sao_async_rl_toy.py)，仅使用 Python 标准库和合成数据；不联网、不调用付费 API、不加载 GLM-5.2/5.3 权重，也不声称复现 SAO 论文 benchmark。
- 运行通过：group barrier 的合成等待总量为 `11`，single-rollout wait 为 `0`；ratio `0.1`/`5.0` 被 mask，`1.0`/`1.1` 保留；observation token 从 `1` 增至 `3` 时 Skip-Observation GAE 的 `a0` 保持 `1.0603`，普通 token GAE 首段从约 `1.04276` 变为 `0.99734`；critic proxy 的 `K=2` 最终平方误差 `0.0625`，`K=1` 为 `0.25`。
- 这只升级为 **local protocol toy evidence**：验证公开机制的合成状态机内部一致性，不升级真实 rollout engine、policy lag calibration、GLM-5.2/5.3 训练效果、产品 compaction、完整权重、目标硬件、verifier 或生产 SLO。
- 已同步 [`glm-5.2-source-notes.md`](research/model-update-2026-09/glm-5.2-source-notes.md)、[`glm-5.3-source-notes.md`](research/model-update-2026-09/glm-5.3-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、`PROJECTS.md`、`EXERCISES.md`、`plan.md` 和本进度表。下一门禁仍是 5.3 专属 compaction/recipe 与 full-weight/runtime 条件，goal 保持 `active`。

## 2026-09-23 GLM-5.3 compaction 复核与本机门禁

- 本轮重新尝试三条代理：`10.237.126.170:1234`、`10.24.27.134:7890`、`10.24.27.134:8098`。它们访问 `https://docs.z.ai/guides/llm/glm-5.3`、`https://z.ai/blog/glm-5.3` 和 `https://www.baidu.com` 均在连接阶段失败，curl 返回 HTTP `000`。因此当前结果记录为代理不可达，不解释为网页不存在；网络恢复后仍需重新复抓两个排行榜和官方资料。
- 使用本机已保存的官方 Markdown `23,446` bytes / SHA-256 `9545c3d6fb1cabfa5951928bbe9a535e6958561d3bd6b5f4354cd5d3b0f6c929`、博客正文资源 `30,414` bytes / SHA-256 `f809bf586f74b40a01ecd8d674f9ec3de469a5abeae125e2a849c9805605d0f3` 做离线复核：资料仍只说 GLM-5.3 沿用 GLM-5.2 基座、改进来自 post-training，并继承 `SAO with compaction`；没有状态 schema、序列化、切分阈值、压缩质量门禁或完整 5.3 recipe。
- 本机不存在 GLM 权重文件；`torch`、`transformers`、`vllm`、`sglang`、`safetensors` 不可导入，`nvidia-smi` 无法连接驱动。因此未进行完整权重加载、kernel、数值、硬件 profiling 或线上 acceptance，相关项目继续标记 `unverified`。
- 新增并运行 [`glm53_compaction_contract_audit.py`](research/model-update-2026-09/code/glm53_compaction_contract_audit.py)：完整候选的 schema round-trip、goal/plan、工具回执链、幂等键、待执行副作用、artifact digest、verifier 状态、预算和 cut marker 检查全部通过；故意丢字段的候选被拒绝。输出证据等级为 `local_protocol_toy`，不代表 Z.ai 内部实现、真实 GLM-5.3 或 SAO 论文 benchmark。
- 当前状态保持：**GLM-5.3 双榜资料级闭环 + SAO 关联论文算法证据 + stable/main runtime source evidence**；5.3 专属 compaction、完整 recipe、完整权重、目标硬件和 tool/verifier acceptance 仍待核验，goal 保持 `active`。

## 2026-09-23：`7890` 代理恢复、两榜复验与 Grok 4.7 回放审计

- 当前工作目录确认是 `/data/zzc/llm-from-zero-to-interview`，与 `/home/zzc/llm-from-zero-to-interview` 为同一 inode。沙箱内访问代理失败，但使用已验证的 `10.24.27.134:7890` 沙箱外请求成功：百度 HTTP 200，Artificial Analysis 和 DataCurve 均 HTTP 200。
- 当前 Artificial Analysis 中文首页为 `1,783,572` bytes / SHA-256 `999ead1b8d025a8abccc5d6e879e548544770335863dec1be06b21d99378f8e2`；DataCurve 为 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。按八家重点厂商 canonical slug 去重后没有新增模型，未从官方目录、论文或 runtime 仓库扩展候选。
- 两榜当前仍含 `Grok 4.7`、`Claude Opus 5.5`、`GPT-6 Sol/Luna`、`GLM-5.3`、`Gemini 3.8 Flash`、`DeepSeek V4.1-Flash`、`Kimi K3`、`Qwen3.8 Max` 等既有条目；Grok 4.7 仍无精确 `mini_swe_agent_grok_4_7_*` DataCurve 行。
- Grok 4.7 详情当前 `3,992,680` bytes / `9b24aa029ee4023aa2918495910a282a96f8bd9bba42848921f9496684e3e6bb`，release `2026-09-21`、Intelligence Index `46.4465506302286`、500K context、proprietary、parameters null；动态页面 hash 变化不解释为 revision 更新。
- xAI 模型页、Reasoning、Context Compaction 当前分别为 `376,911`/`72e4d59db379c9be4e1d68c9f5c14eff2281d9c9ffa5d2f3bcdb0e701471319c`、`522,822`/`f69666e9a7a47e75c7a394875e280ec15e6e2003a8cd1a6ec6a0e64334fa357a`、`521,433`/`ff467261f7174c235987232139c6c839d0488858507f3ac0f598afd3d0f48eb3`；正文未观察到新的模型合同、架构或训练披露。
- 新增 [`grok47_state_replay_audit.py`](research/model-update-2026-09/code/grok47_state_replay_audit.py)：验证 opaque reasoning/tool state 原样回放、summary/state 分离、tool lineage、单一 compaction、`store=false`/`previous_response_id` 约束和幂等副作用。脚本不联网、不解密、不调用 xAI，证据等级为 `local_protocol_toy`，不代表 Grok 4.7 生产实现或质量。
- 当前 Grok 4.7 状态更新为 **AA 单榜内容专题闭环 + 当前时点复验 + local protocol toy**；参数、架构、完整 recipe、生产 kernel、硬件 profiling、线上 acceptance、独立 benchmark 和完整安全评测仍为 `unverified`。goal 保持 `active`。

## 2026-09-23 GPT-6 Astra 当前时点复验与失败路径审计

- 当前工作目录仍为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留 dirty worktree。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、Z.ai/GLM、Qwen、Kimi、DeepSeek、Gemini、Grok。
- 使用用户提供的 `10.24.27.134:7890` 重新抓取百度、Artificial Analysis 中文首页、Astra 详情和 DataCurve，全部 HTTP `200`。AA 中文首页为 `1,783,572` bytes / SHA-256 `999ead1b8d025a8abccc5d6e879e548544770335863dec1be06b21d99378f8e2`；Astra 详情为 `3,997,013` bytes / `c4c040e6708555c1965efaf4eecd3d61b4199ffe337d07ce53970150b79103ce`；DataCurve 为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。八家重点厂商没有新 canonical 模型。
- AA Astra `max` 当前字段为 `releaseDate=2026-09-03`、`knowledgeCutoffDate=2026-04-30`、1M context、Intelligence Index `52.673669395513`、median output speed `58.6808088606541 tokens/s`、cost per Intelligence Index task `3.2575003134834164`、输入/输出 `$10/$50`、cache hit/write `$1/$12.5`；全部按第三方/provider 字段记录。DataCurve 生成时间为 `2026-09-22T06:27:15.860279+00:00`，五个精确行是 `mini_swe_agent_gpt_6_astra_{low,medium,high,xhigh,max}`，对应 passed/attempts `303/452`、`329/452`、`331/452`、`335/452`、`331/452`；不把这些 Agent 结果写成裸模型能力。
- 7890 重新取得 Astra model page、latest-model guide、Async tool calling、Steering 和 Misalignment monitoring Markdown，分别为 `3,812`、`17,456`、`24,200`、`12,983`、`7,830` bytes。model page、Async/Steering/Misalignment 直接页面与既有快照一致；latest-model guide 从 9 月 21 日 Astra 专属版本 `16,592` bytes / `aac7e7e1b0bf90b346f5fc4ffd523902279841c60c3adf572452a529bd5b1e5e` 漂移为当前 GPT-6 family 版本 `17,456` bytes / `8982485767fcefe9b4c588777de67683b4f56abe9c9af1121c9431eb9c305557`。新增的是 family capability matrix 和迁移边界，没有新的 Astra 独有协议字段。
- 扩展 [`gpt6_agent_protocol_demo.py`](research/model-update-2026-09/code/gpt6_agent_protocol_demo.py)：新增重复 `task_handle`/`call_id`、错误 `call_id`、pending consume、重复结果内容漂移、重复 steering、断线后旧 steering 不重放、misalignment 阻断后禁止自动重试和已启动副作用保留等断言。当前脚本 `11,820` bytes / SHA-256 `f3ef1aea13de9dc432fb427f15e02a6f9b96e2505311a8cd219f54681f62f245`；`py_compile`、主流程、`git diff --check` 均通过。证据等级为 `local_protocol_toy`，不代表真实 API、服务端事件顺序、隐藏 reasoning、模型质量或生产安全。
- 当前 Astra 状态为 **内容专题闭环 + 双榜当前时点复验 + local protocol toy**。参数规模、内部架构、训练/后训练 recipe、system card、专属技术报告、完整权重、生产 kernel、目标硬件 profiling、独立 benchmark、tool/verifier acceptance 和生产 SLO 继续标记为 `unverified`。下一步先完成第六册、题库/练习的失败路径同步，再从两榜当前八家重点厂商已有 canonical 集合选择下一项未收口锚点。

## 2026-09-23：Claude Opus 5.5 协议门禁补证

- 本轮继续只沿两个排行榜已有的 `claude-opus-5-5` canonical 条目推进。使用 `10.24.27.134:7890` 取得 AA 详情 `3,824,658` bytes / SHA-256 `727da6095c069bf0750a36cea442c7e582bbfeefdf4fc4a330a656bd8bd45c31`；release `2026-09-22`、1M context、Intelligence Index `57.6223698102963`、proprietary、`parameters=null`、`$4/$20` input/output 保持不变。DataCurve `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1` 仍没有精确 `mini_swe_agent_claude_opus_5_5_*` 行，不迁移 Opus 5/Fable 5.1 的 Agent 分数。
- Anthropic 官方 raw Markdown 当前为 model page `14,613` bytes、What's new `21,525` bytes、Migration guide `16,296` bytes；哈希与来源索引一致。文档明确 adaptive thinking 永远开启，disabled/manual budget 和 `tool_choice=any/tool` 返回 400；thinking block 绑定 producer/conversation/prefix；Claude API/Google Cloud 需迁移 `computer_toolset_20260801`；工具间进度可能以空 thinking block 出现；compaction、inline tools、refusal/fallback 和 fast mode `usage.speed` 必须进入 trace。
- 新增 [`claude_opus55_protocol_audit.py`](research/model-update-2026-09/code/claude_opus55_protocol_audit.py)，标准库验证 request capability gate、按 `block.type` 解析、thinking binding、签名 compaction、tool idempotency、fallback 实际模型和 8 组负例。主流程、重复回放和失败断言通过；证据等级固定为 `local_protocol_toy`，不调用 Anthropic、不解密 thinking、不执行真实工具。
- 已同步 Opus 5.5 研究笔记、来源索引、模型盘点、计划、题库、练习、项目以及第八/十六/二十册相关章节。当前状态为 **AA 单榜 + System Card/API contract + local protocol toy**；参数、架构、完整 recipe、精确 DataCurve Agent 行、独立 benchmark、目标硬件、生产 kernel 和线上 acceptance 仍为 `unverified`。goal 保持 `active`。

## 2026-09-23：7890 恢复后的 OpenAI gpt-oss 当前时点复验

- 本轮使用用户确认可用的 `10.24.27.134:7890` 重新验证外网：百度、Artificial Analysis 中文首页和 DataCurve DeepSWE 均 HTTP `200`。AA 中文首页为 `1,783,626` bytes / SHA-256 `0580fad58c167fb96e87289addbabccd40cbd60302c527437805c8eb6ee7530b`；DataCurve 为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。八家重点厂商的 canonical 集合没有新增模型。
- 本轮继续只沿两个排行榜已存在的 `gpt-oss-120b` / `gpt-oss-20b` 推进，没有从 OpenAI Cookbook、GitHub 或 Hugging Face 另发现模型。AA 详情快照分别为 120B `4,078,295` bytes / `43e5f13a3e58976b077acac1810bce82ac60ce33a7d81476f0dc2b17175532de`、20B `4,083,068` bytes / `64d67047a5964c0109f103108f9b77c64edbaf1ee17f3fec3c82ffccd7cea2cc`。
- AA 当前第三方/provider 字段：120B/high 为 117B total、5.1B active、131,072 context、Index `11.6028431512592`、约 `196.389235173389 tokens/s`、cost/task `0.10742452290394947`；20B/high 为 21B total、3.6B active、131,072 context、Index `8.9675171856126`、约 `185.656167974395 tokens/s`、cost/task `0.012460640242179213`。这些是采集时点观察值，不是新的权重、架构或 revision 证据。
- DataCurve 当前仍没有精确 `mini_swe_agent_gpt_oss_120b_*` / `mini_swe_agent_gpt_oss_20b_*` 行，因此不迁移 GPT-5.x、Codex 或其他 OpenAI 模型的 Agent 分数；缺失只作为当前快照负证据记录。
- 7890 当前取得的官方 Cookbook 页面：实现兼容性验证 `337,423` bytes / `e6500d57bc6041133a8f63acc03101260182ea58d45d2f7248806ae609c85474`，raw CoT 处理 `339,052` bytes / `1c08e3fb06a129256596c1af9ba907b7b3c435be3b61cfb00a72da83b1071fe6`。动态正文哈希变化没有带来新的模型 ID、权重 revision 或架构声明；继续保留 Harmony/API shape、tool loop、raw CoT lineage、质量 eval 与 kernel/硬件/生产验收分层。
- 研究笔记、来源索引、模型盘点和榜单解释已完成本轮同步；第二十一册第 87 章及既有题库、练习、项目入口已存在，无需重复建章。当前 gpt-oss 状态为 **当前时点 AA 复验 + 官方 provider compatibility/raw-CoT evidence 的内容专题闭环**；完整 recipe、MoE 负载均衡、MXFP4 kernel/误差、目标硬件、独立 Agent 评测、tool/verifier acceptance 和生产 SLO 仍为 `unverified`。goal 保持 `active`。

## 2026-09-23 DeepSeek V4.1-Flash Harness Preview 补证

- 本轮确认工作目录 `/data/zzc/llm-from-zero-to-interview` 与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同，保留既有 dirty worktree。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、Z.ai/GLM、Qwen、Kimi/Moonshot、DeepSeek、Google/Gemini 和 xAI/Grok。
- 使用用户确认可用的 `10.24.27.134:7890` 继续联网；前序两榜复验没有新的八家重点厂商 canonical 模型。本轮不从 Harness 文档、GitHub、论文、HF 或 API 目录另发现模型，活动锚点继续是已有 `deepseek-v4-1-flash`。
- 固定官方 Harness Preview 文档：quickstart `83,023` bytes / `1e816c2bbead769f57b2c344334c02d6136eeb5831d46a9a3d5cc46f9f135966`；providers `113,025` / `1761dd552158cfe6e01245744414ef5c81273f2ba09026d4a5b6e94a3065182d`；Python SDK `110,733` / `3459ca57567d8a3ff982674ac57a390ecd8215336a5e426d40e4b4ead6c51edf`；architecture/reference `130,135` / `3595d42386804b571bb5976592e07f10a1342764d64485c08e37ea1604b74837`；MCP memory `96,150` / `43751eb6e0f5a363397d26844c2068a6d820adf6d65cbf8381fe993d3764fa33`；GitHub review `94,772` / `3dfae298c9af189c1965a1ff19e78ccbab3f575fd98b0f1ced45925497b0f31f`。
- 提取的权威知识点是：provider ID 持久化且 credential 只脱敏；session log 固化实际使用模型；插件依赖和 listener/resource 需要生命周期清理；MCP 使用 `mcp__<server>__<tool>` namespace、过滤 credential/`DSH_*` 环境变量、断线重连后重新发现并在预算耗尽后注销旧工具；GitHub review 的签名 webhook 和 `202` 只表示异步 admission，重复 delivery 与出站权限必须单独治理。
- 新增并运行 [`deepseek_harness_protocol_audit.py`](research/model-update-2026-09/code/deepseek_harness_protocol_audit.py)：输出 `ok=true`、provider credential `<redacted>`、`session_model=deepseek-flash`、`duplicate_webhook_admissions=2`、预算耗尽后 MCP 工具为空、`network_called=false`。证据等级固定为 `local_protocol_toy`，不代表 DeepSeek Harness 生产实现、真实 API、V4.1 模型质量或 SLO。
- 已同步第二十一册第 81 章、第二十册第 19 章、来源索引、模型盘点、榜单解释、计划、题库、练习、项目、论文和知识图谱。当前状态为 **AA 单榜内容专题 + 官方 API/runtime + Harness Preview + local protocol toy**；DataCurve 无精确 V4.1 行，不能迁移其他 DeepSeek 的 Agent 分数。下一步仍是 pinned revision -> runtime import -> 完整权重 -> 目标硬件 -> 数值/工具/verifier/SLO 门禁，goal 保持 `active`。

## 2026-09-23 Qwen3.8 Max (0902) Context Cache 协议补证

- 本轮继续只沿 Artificial Analysis 与 DataCurve 已确认的 `Qwen3.8 Max (0902)` 推进，没有从 QwenCloud 文档另发现模型。使用 `10.24.27.134:7890` 重新抓取 QwenCloud Thinking、Function Calling、Context Cache 和 Dynamic Rate Limiting 页面，HTTP 均为 `200`；Context Cache response 为 `1,133,845` bytes / SHA-256 `42d5d39fe4cb29680a27fcdd8d1b9bb1c07cdea93f01cb2ae31f40b02ca2edb8`，Dynamic Rate Limiting response 为 `411,250` bytes / `b517a331efcdd822bfd63e81894e735a686db01baf1e77702155a0d8905ce82f`。
- Context Cache 正文补充确认：`cache_control.type` 只能为 `ephemeral`；单请求最多 4 个 marker，超过 4 个只有最后 4 个生效；最小缓存前缀 `1,024` tokens；marker 到命中内容超过 20 个 content block 时 backward lookback 可能 miss。
- Explicit/session cache 的典型有效期为 5 分钟且命中刷新；implicit cache 自动启用、命中不保证、无固定 TTL而由 provider 清理长期未使用数据。文档典型相对计费为 explicit/session 创建 `125%`、命中 `10%`，implicit 创建 `100%`、命中 `20%`；与产品页当前 `$2/$6/$0.25/$2.50/$0.17` 价格字段分开记录。
- Session cache 的可观察合同为 Responses API + `x-dashscope-session-cache: enable` + `previous_response_id`；cache 按 account/model 隔离，命中从 `usage.input_tokens_details.cached_tokens` 读取。后端 marker 后追加通常少于 10 个 token，导致总 `input_tokens` 不一定等于 cache creation + cached hit tokens。
- 新增并运行 [`qwen_max0902_cache_contract_audit.py`](research/model-update-2026-09/code/qwen_max0902_cache_contract_audit.py)：`py_compile` 通过，主流程输出 `ok=true`、4-marker 截断、显式命中/过期、20-block miss、account/model 隔离、implicit 非保证命中、session `cached_tokens=1024`、thinking forced-tool 和 reasoning budget 冲突负例均通过；`network_called=false`，证据等级 `local_protocol_toy`。
- 已同步 [`qwen3.8-max-0902-source-notes.md`](research/model-update-2026-09/qwen3.8-max-0902-source-notes.md)、第 83 章、第二十四册第 32 章、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`PROJECTS.md` 和 `KNOWLEDGE_GRAPH.md`。Qwen3.8 Max 0902 仍是 **hosted revision 资料级闭环 + Context Cache 官方协议补证 + local protocol toy**，没有新增 open checkpoint 或 0902 专属架构证据；goal 保持 `active`。

## 2026-09-23 Claude Fable 5.1 状态协议门禁

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，使用用户确认可用的 `10.24.27.134:7890`。百度、Artificial Analysis 中文首页、Fable 详情和 DataCurve 均 HTTP `200`；AA 首页为 `1,783,769` bytes / SHA-256 `e1acf794bd45f380ce3220a20c68de7a274f50ff7eb162a527a9faf4d58a166a`，Fable 详情为 `4,008,017` bytes / `4ea24782d05bcaae7d31f3cf348e5a573851678998706a9df82fa34679292f96`，DataCurve 为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。八家重点厂商按 canonical slug 去重后没有新增模型。
- Artificial Analysis 当前 `claude-fable-5-1` 仍为 `max with fallback`，Intelligence Index `53.3549259623252`、median output speed `64.7060238772019 tokens/s`、cost per Intelligence Index task `7.629706364004841`、1M context；这些是第三方 provider/configuration 字段。DataCurve 没有精确 `mini_swe_agent_claude_fable_5_1_*` 行，继续保持负证据，不迁移 Fable 5 的 `316/452` 或其他 Claude 的 Agent 结果。
- 7890 取得 Anthropic 官方 Markdown：overview `14,954` bytes / SHA-256 `13e8aeb6bbd207311ac032916f2ab37e7a454c9d752c027cf76967a5c958a076`；What's new `37,545` / `59d2a26f6e123d009a9e86aac9283705f7bfd15549b185dc07aa1858241baf19`；migration guide `94,368` / `c1d7bd16475ce9ad03ad556cc363635d93e695ecafc293e6acb85023e9f869b6`。System Card 原始 PDF 仍为 `16,397,488` bytes / `b0d59edc7a60eef32a879c13d713cce60c3fefd7e6b5183afdc8b835af3c8c39`。
- 官方迁移正文新增可面试边界：prefill 也返回 400；模型只接受 adaptive always-on thinking；thinking block 可读性是单向的；prefix mismatch 可拒绝或显式 `drop_block` 并记录 `input_transformations`；30-day retention/ZDR 和 Priority Tier 是部署约束。`display: "updates"` 仍只是用户可见进度，不替代 tool result、权限、幂等和 artifact verifier；provenance 也不替代事实验证。
- 新增并运行 [`claude_fable51_state_protocol_audit.py`](research/model-update-2026-09/code/claude_fable51_state_protocol_audit.py)：`py_compile` 与主流程通过，输出 `local_protocol_toy`、首次副作用 1、重复回放 duplicate receipt true、side-effect executions 1；故意构造的 thinking 配置、forced tool、prefill、旧模型读取、prefix mismatch、缺失 tool result 和 unsupported provenance 均被拒绝。该脚本不联网、不调用 Anthropic、不解密 thinking。
- 已同步 Fable 研究笔记、来源索引、模型清单、榜单解释、第二十一册第 19 章、`INTERVIEW_BANK.md`、`EXERCISES.md` 和 `PROJECTS.md`。当前状态为 **AA + System Card/API contract + local protocol toy**；参数、架构、完整训练/后训练 recipe、独立 Fable 5.1 技术报告、精确 DataCurve Agent 行、目标硬件 profile、tool/verifier acceptance 和生产 SLO 仍为 `unverified`，goal 保持 `active`。

## 2026-09-23 GPT-5.6 Luna：7890 复验与状态回放审计

- 本轮继续只沿两个排行榜已有的 `GPT-5.6 Luna` 推进，工作目录仍为 `/data/zzc/llm-from-zero-to-interview`，等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree。使用 `10.24.27.134:7890` 重新取得 AA 详情和 OpenAI 官方 Markdown，没有从官方目录另发现模型。
- Artificial Analysis 详情 HTTP 200，当前快照 `3,996,959` bytes / SHA-256 `4246b96416b4aaf4f8bbcacf72f8c45787944e24f02c59c6e3db1fed4de93c41`；`max` 的 release `2026-07-09`、Intelligence Index `37.3244239690841`、median output speed `142.271568203031 tokens/s`、cost per Intelligence Index task `0.17829726152289094`、1M context 和约 `$0.20/$1.20` 均按第三方/provider 字段记录。相对 9 月 22 日速度 `158.728370482714` 的变化只作为采集时点漂移，不解释为模型 revision 或训练升级。
- DataCurve 当前快照仍为 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；精确 `mini_swe_agent_gpt_5_6_luna_max` 仍是 attempted `448`、passed `301`、Pass@1 `67.1875%`、Pass@4 `90.26548672566371%`、平均成本 `$0.6056233620535714`、平均输出 `73399.70758928571` tokens、平均 Agent steps `101.68080357142857`、median peak context `201647`。结果继续绑定 `mini-swe-agent + tools + task environment + verifier`，不写成裸模型能力。
- 7890 当前官方 Markdown 快照：Luna `3,744` bytes / `1f425d8f2a93f8418702ac7acbb62b26850b98fe20680f82205bed5fce38fed0`；Reasoning `70,315` / `93327c5eb19df7ecf8c0bd9d581f58052dee9f090c38bb48906110b08d0ce251`；Prompt caching `47,098` / `69680fbec38e31a7abc8b0e33a6582b0aae29687e8de98404789337a55b55e2e`；Compaction `14,272` / `73fd2fd1afd44bd6f29ce7bd86fd0ae3413c98ec00ae5476879e98171b0b60dd`；Tools `33,282` / `4722fa102070178c1a1d603e718c69575c7c1768fa901218eab359bafdcba341`。页面恢复可读，但没有产生公开参数、架构、训练 recipe 或 system card 证据。
- 新增并运行 [`gpt56_luna_state_replay_audit.py`](research/model-update-2026-09/code/gpt56_luna_state_replay_audit.py)：`py_compile` 和主流程通过，输出 `ok=true`、`evidence_level=local_protocol_toy`、`network_called=false`。覆盖 `current_turn/all_turns/auto`、同家族 reasoning 兼容、opaque reasoning/assistant phase/function call-output 原样回放、`call_id` 血缘、idempotency/verifier、compaction canonical window、compaction 后 cache miss、1,024 token 门槛、四个 explicit breakpoint、30m toy TTL、hosted/client tool search 所有权。
- 当前状态更新为 **双榜当前时点复验 + 官方 runtime contract + local protocol toy**。toy 不代表真实 endpoint schema enforcement、服务端事件顺序、隐藏 reasoning、模型质量、完整权重、目标硬件 profiling 或生产 SLO；不新增 GPT-5.6 专属 Transformer 正式章节。已同步研究笔记、source index、model inventory、inventory interpretation、`plan.md`、第十七/二十/二十四册相关章节及题库/练习/项目/论文/知识图谱，goal 保持 `active`。
- 下一步门禁：若环境允许，做真实 API capability probe；否则保留 `unverified`，继续按 pinned revision -> runtime import -> 完整权重 -> 目标硬件 -> 数值/工具/verifier/SLO 顺序推进，不从官方目录、论文、HF 或 runtime 仓库绕过两榜新增模型。

## 2026-09-23 GPT-6 Sol：7890 当前时点复验与合同审计 toy

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、Z.ai/GLM、Qwen、Kimi/Moonshot、DeepSeek、Google/Gemini 和 xAI/Grok。
- 使用已验证的 `10.24.27.134:7890` 重新抓取两个指定排行榜：Artificial Analysis 中文首页 HTTP 200，`1,783,769` bytes / SHA-256 `e1acf794bd45f380ce3220a20c68de7a274f50ff7eb162a527a9faf4d58a166a`；DataCurve HTTP 200，`268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。当前 AA 首页仍出现 `GPT-6 Sol`，本轮没有从官方目录、论文、HF、GitHub 或 runtime 仓库另发现模型。
- AA [`gpt-6-sol`](https://artificialanalysis.ai/models/gpt-6-sol) 详情 HTTP 200，当前快照为 `3,994,562` bytes / SHA-256 `3d08f91c71e2770a4ef84b593c2a197d711193bbcbd8c3a1f330393b65f0ea86`；标题为 `GPT-6 Sol (max)`，release 为 2026-09，Intelligence Index `47.5276426437724`、median output speed `126.038858615917 tokens/s`、cost per Intelligence Index task `1.0564240894076389`。上一轮约 `115.205383643174 tokens/s` 只保留为历史 provider 测量，速度变化不解释为模型 revision。
- DataCurve 当前没有精确 `mini_swe_agent_gpt_6_sol_*` 行，因此不迁移 GPT-6 Astra、GPT-5.6 或其他 GPT 的 Pass@1、成本、输出 token、steps 和 Agent 结果；AA 的 max 配置也只作为第三方/provider configuration 字段记录。
- 7890 当前官方 Markdown 快照为：GPT-6 Sol model page `3,991` bytes / `5566589803b6f0ac79c00125a7c1af9291e46d5724c1c414f932d36ab92a8ad0`；Reasoning `70,315` / `93327c5eb19df7ecf8c0bd9d581f58052dee9f090c38bb48906110b08d0ce251`；Agents `5,432` / `df4f61b609550619a3f9e445c66d28318b9f681affe1a819ef0098cde32bf43a`；Tools `33,282` / `4722fa102070178c1a1d603e718c69575c7c1768fa901218eab359bafdcba341`；Compaction `14,272` / `73fd2fd1afd44bd6f29ce7bd86fd0ae3413c98ec00ae5476879e98171b0b60dd`；Prompt caching `47,098` / `69680fbec38e31a7abc8b0e33a6582b0aae29687e8de98404789337a55b55e2e`。
- 官方新增/复核面试主线已归纳为：`reasoning.mode=standard/pro` 与 effort 独立；标准单 Agent 中 `configuration_update` 只改后续 effort、相邻 update 拒绝，且不能与 automatic compaction/truncation 或 standalone compact 历史组合；`incomplete` 需要分账 reasoning/output/context 预算；超过 `272K` input 是整次请求 input/cache 2x、output 1.5x；模型工具目录不授予宿主权限，必须分开记录 permission、executor、artifact verifier 和幂等 replay；opaque compaction 可能造成新的 cache prefix miss。
- 新增并运行 [`gpt6_sol_contract_audit.py`](research/model-update-2026-09/code/gpt6_sol_contract_audit.py)：`py_compile` 与主流程均通过，输出 `ok=true`、`evidence_level=local_protocol_toy`、`network_called=false`。toy 覆盖 mode/effort、合法/非法 update、compaction 后 update、budget incomplete、272K 计价、permission/executor/verifier、function-call lineage、idempotent replay、opaque canonical compaction、1,024 token cache minimum、4 个 explicit breakpoint 和 30m toy TTL；不调用 OpenAI、不解密 reasoning、不代表真实 endpoint、模型质量或生产 SLO。
- 已同步 [`gpt-6-sol-source-notes.md`](research/model-update-2026-09/gpt-6-sol-source-notes.md)、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan.md`，并将下一步入口写入第六/十六/十七/二十/二十四册及 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`。当前状态为 **AA 单榜资料级闭环 + 当前时点复验 + local protocol toy**；参数、架构、完整训练/后训练 recipe、完整权重、目标硬件 profiling、生产 kernel、精确 DataCurve Agent 行、独立 benchmark 和线上 acceptance 仍为 `unverified`，goal 保持 `active`。

## 从 plan.md 迁移的近期专题与历史执行记录

以下内容原属于计划文件中的“近期专题记录”和“历史执行记录”，现迁移到进度文件，保留原始研究证据、时间语境和待核验边界。后续新增执行记录直接按日期追加到本文件；计划文件只保留当前专题队列和下一步。

### 近期专题记录（从计划迁移）

以下保留近期已完成/复验事项的历史快照；当前锚点以文件顶部“当前活动锚点”为准，最近推进记录见文件末尾。

前序阶段记录（2026-09-24，GPT-6 Sol）：Artificial Analysis `/zh` 经 `10.237.126.170:1234` 获取（`1,783,966` bytes / SHA-256 `0baa28c8ab69a104a078645983ca843c95c5db879b6c52789104aa3d1758efb8`）；Sol 详情 `3,976,802` bytes / `ff0aeaedb21ad7a672653b3d019c0574c6e468a74808db1f3973731d311d5ba5`，Index `47.5276426437724`、cost/task `$1.0564240894076389` 不变，速度 `109.551294011457 tokens/s` 仅作 provider/采集时点变化。DataCurve 无精确 `mini_swe_agent_gpt_6_sol_*` 行；OpenAI EU residency 范围与 `Standard processing` / reasoning `mode=standard` 的区分属于 API/部署合同，不是模型内部技术。

前序阶段记录（2026-09-24，Kimi K3）：vLLM v0.30.0 stable/source 实现演进已同步至研究笔记、第 88 章和配套资料；wheel 未下载/安装，完整权重、目标硬件、数值正确性、双状态恢复与生产 SLO 未验证。
最近完成的独立专题（2026-09-24）：两榜发现的 `Claude Fable 5.1`。Anthropic Thinking 文档补出 Fable 5.1 读取 Opus 5.5 thinking blocks 的单向边，且仅 Claude API 明确支持；model binding 与 prefix binding 分开处理，已落入第二十册第 21 章 21.29 和协议 toy。内部架构、训练配方、真实 endpoint probe 与生产验收仍未确认。

2026-09-24 较早两榜复验（GPT-6 Luna 同日后续快照见本计划末尾）：Artificial Analysis 中文首页为 HTTP 200、1,784,805 bytes、SHA-256 `5284847a4499b219872725221324a3461de68464956a4b8f44b5cdd96d34c2c8`；DataCurve 为 HTTP 200、268,036 bytes、SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。AA 当前发布列表的最新重点模型为 Claude Opus 5.5、GPT-6 Luna 与 GPT-6 Sol；均已有专题记录。DataCurve 精确配置集合与前次快照相比没有发现新 canonical 模型。AA 当次详情快照：Opus 5.5 3,810,653 bytes / `cc5a94faec8125d88e46dc6545a564f91f32855b5d05da4df7cd3bca08851e95`，Intelligence Index `57.6223698102963`、成本/任务 `$5.982012019521066`；GPT-6 Luna 3,975,044 bytes / `4aaaeefba90808253aee562a64d10b95a32c1e808c1b4aacaf44a2e2adc04516`，Index `37.2559686869738`、成本/任务 `$0.06809498628701058`、速度 `131.449006457181 tokens/s`。这些均为榜单 provider/评测测量字段，不能解释为模型 revision 或架构变化。Anthropic 官方 Opus 5.5 model page 当时为 14,244 bytes / `3325e10a852cc3f40cfaf01737d97a5cdf3a37b4cd7c0b68005e204ea32b1dd4`，支持 1M context、128K sync output、always-on adaptive thinking、默认 medium effort、$4/$20 每百万 input/output、512-token cacheable prompt minimum；详细变化并入研究笔记。

Grok 4.20 复访：AA 详情 HTTP 200，`3,857,337` bytes / SHA-256 `d56d4557d8a661ee9c3015c6a3661f5baae6c00f9b16c5b2acf5eabcd3e61300`；页面仍列 estimated Index `25.6550155187053`、2M context，并显示速度 `106.190642707844 tokens/s`、TTFT `21.53034693s`、端到端 `26.23885972594964s`。页内告示限定仅默认 10K input token workload 继续更新 benchmark，其他 workload 历史且不再更新；因此各字段的 workload/新鲜度不得由抓取时间推断。DataCurve HTTP 200，`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，无精确 `mini_swe_agent_grok_4_20_*` 行；xAI 官方 Markdown HTTP 200，`1,564` bytes / `99057608aeb1a4e6ace5d87bf6c4b27b3bff4af5248babad475500da31fdc2e3`，仍记官方 1M context。AA 的 `deprecatedTo: grok-4-3` 不作为 xAI API 下线或迁移证据。

GPT-6 Sol 前序阶段备注：已完成当前榜单与 OpenAI runtime 复验；其既有 AA 单榜资料级闭环、官方工具/compaction/caching 契约和 toy 记录保留，不把 GPT-6 Sol 的 provider 指标或 Agent 结果迁移到 Luna/Astra。

本轮已将 Opus 5.5 的同步 Messages API 128K、带 `output-300k-2026-03-24` beta header 的 Batch 300K、512-token cache minimum 和缓存读写单价同步至研究笔记、来源索引、盘点、解释、reasoning/Agent 章节及面试练习。下一步重新以两榜 canonical 集合决定锚点；若仍无新增重点模型，优先补此前标为 unverified 且面试价值高的已有锚点，不扩大厂商范围。

上一阶段活动锚点（2026-09-23）为 `GPT-6 Sol`。其 AA 单榜资料级闭环已完成；本轮保留它的长上下文、mode/effort、工具和 compaction 证据，不把 Sol 的 provider 指标或 Agent 结果迁移到 GPT-6 Luna。

上一轮活动锚点为 `GPT-6 Luna`。该轮只沿两个榜单已经出现的 `gpt-6-luna` canonical 条目推进，不从 OpenAI 官方模型目录另发现其他模型；Luna 的 focused/high-volume 定位、1.05M/922K/128K、effort、runtime ownership 和 opaque compaction state 已按服务合同入库。

上一轮活动锚点为 `Claude Opus 5.5`。本轮只沿两个榜单已经出现的 `claude-opus-5-5` canonical 条目推进，不从 Anthropic 官方目录另发现 Sonnet 5.5/Haiku 5.5 等模型；System Card 正文的评测条件、安全路由、prompt injection、OSWorld compaction 和多 Agent derived latency 已收口。

上一阶段活动锚点为已有排行榜条目 `DeepSeek V4.1-Flash`。本轮已完成官方 API/provider contract 和标准库协议 toy；完整权重、量化/召回、目标硬件、DSpark acceptance、工具验收和生产 SLO 仍未通过门禁。DataCurve 当前没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，不迁移 DeepSeek V4 Pro、V4 Flash 或其他 DeepSeek 版本的 Agent 成绩。

当前活动锚点切换为已有排行榜条目 `Gemini 3.8 Flash`。本轮只沿两个排行榜已经出现的 canonical 模型推进，不从 Google 官方模型目录另发现模型；重点是 2026-09-23 Thinking/Interactions/Tool combination 文档的 state、signature、预算、工具上下文和生命周期合同。`low/medium/high` 仍归并为一个基础模型，DataCurve 只引用精确 high 行，不迁移相邻 Gemini 或其他 effort 的 Agent 结果。

当前活动锚点切换为已有排行榜条目 `Kimi K3`。本轮只沿两个排行榜已经出现的 canonical 模型推进，不从 Kimi 官方仓库、HF、vLLM 或 SGLang 另发现模型；重点是榜单当前时点复验、官方 artifact revision 是否漂移、DataCurve harness 账本和 K3 发布限制的面试表述。`low/max` 仍归并为一个基础模型；DataCurve 只引用精确 `mini_swe_agent_kimi_k3_max` 行，不迁移其他 Kimi 或相邻 effort 的 Agent 结果。

Claude Opus 5.5、DeepSeek V4.1-Flash 和 GPT-6 Sol 仍作为已完成阶段的活动记录保留：本轮不因 OpenAI 文档目录或 runtime 文档另发现模型；GPT-6 Luna 的参数、架构、完整训练 recipe、完整权重、目标硬件 profiling、独立 benchmark 和 production acceptance 仍未通过门禁。

当前活动锚点切换为已有排行榜条目 `Claude Fable 5.1` 的 System Card 正文证据复核。本轮不从 Anthropic System Card、Research 页面或外部论文另发现模型；只把这些权威资料用于扩展 AA 已发现的 canonical 条目。Fable/Mythos 的 shared weights、safeguards、训练数据边界、RSP、安全评测和 fallback 条件均按证据责任分层，DataCurve 缺少精确 Fable 5.1 行时继续保持 `not_applicable`。

当前活动锚点切换为已有排行榜条目 `GPT-6 Astra`。本轮只沿 `gpt-6-astra` canonical 条目推进；AA 的 max/xhigh/high/medium/low/Non-reasoning 是同一基础模型的推理配置，DataCurve 的五个 `mini_swe_agent_gpt_6_astra_*` 行是 effort + harness 组合，不新增模型条目。重点是当前时点榜单复验、官方运行时协议失败路径和 Agent 状态账本；不把 API 字段、DataCurve 结果或 local toy 写成内部架构、训练 recipe 或裸模型能力。

当前活动锚点切换为已有排行榜条目 `Claude Opus 5.5`。本轮沿 `claude-opus-5-5` canonical 条目推进，不从 Anthropic 官方目录、System Card 或平台文档另发现 Sonnet 5.5/Haiku 5.5 等模型；重点是 7890 当前时点复验、always-on thinking/effort、thinking block binding、工具与 computer-use 迁移、compaction/inline tools、fallback 和副作用审计。DataCurve 当前没有精确行，不迁移 Opus 5/Fable 5.1 的 Agent 结果。

当前活动锚点切换为已有排行榜条目 `gpt-oss-120b` / `gpt-oss-20b`。本轮沿两个排行榜已存在的 OpenAI canonical 条目推进，不从 Cookbook、GitHub、Hugging Face 或论文另发现模型；重点是 7890 当前时点榜单复验、AA/provider 字段、DataCurve 精确行负证据，以及 Harmony/API shape、tool loop、raw CoT lineage 和部署验收分层。两个尺寸归并为一个模型族，完整 recipe、kernel/误差、目标硬件、独立 Agent 评测和生产 acceptance 继续保持 `unverified`。

当前活动锚点切换为已有排行榜条目 `DeepSeek V4.1-Flash` 的 Harness 周边补证。两榜当前快照没有新的八家重点厂商 canonical 模型；本轮通过 `10.24.27.134:7890` 取得 DeepSeek Harness Preview 文档，重点核验 provider/session identity、credential redaction、plugin lifecycle、MCP reconnect/unregister 和 GitHub webhook admission。Harness 是官方 runtime 合同，不是 V4.1 内部架构；DataCurve 当前没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，完整权重、生产 kernel、DSpark acceptance、目标硬件、tool/verifier acceptance 和 SLO 继续保持 `unverified`。

### 历史执行记录（从计划迁移）

#### 2026-09-24 09:30 UTC：7890 当前快照复验与 Gemini 3.8 Flash 专题落地

- 当前网络路径：沙箱内连接 `10.24.27.134:7890` 被隔离拒绝；经批准在沙箱外使用同一代理后，Artificial Analysis 与 DataCurve DeepSWE 的官方榜单主机均成功 HTTP 200。DataCurve 应使用规范主机 `https://deepswe.datacurve.ai/`；误用其他域名曾超时，不作为网页不可用证据。
- Artificial Analysis `/zh` 本次响应为 `1,784,793` bytes，SHA-256 `a374adfb81fea4fc68e7071ef191612d7f7d92c1c0e05412f3c338e7fe9edbc2`；DataCurve 响应为 `268,036` bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`。与同日早些时候记录的 AA 页面相差 12 bytes，按动态榜单页面快照变化记录，不解释为模型 revision。当前八家重点厂商 canonical 集合和 DataCurve 配置集合均无新增。
- 按计划从既有 canonical 集合选择 `Gemini 3.8 Flash`：此前 9 月 23 日抓取的 Google Thinking、Interactions、Tool combination、Thought signatures 快照和协议 toy 已有来源/哈希；本轮不把旧 Google 文档哈希冒充为 9 月 24 日新抓取。
- 新增正式章节 [第二十册第 23 章](book-20-agent-harness-runtime/chapters/23-gemini-interactions与thought-state回放.md)，展开 stateful/stateless history 所有权、interaction-scoped 配置、opaque signature 与官方字段范围冲突、SSE 重连下工具幂等、留存删除、thinking/output 预算及 capability probe。
- 已同步研究笔记、模型盘点、来源索引、榜单解释、书目、面试题、练习、项目与知识图谱。Gemini 3.8 Flash 升级为“内容专题闭环”，但真实 API probe 与内部架构/训练、硬件/生产验收证据仍未获得；下一轮重新按两榜重点 canonical 集合挑选尚未收口锚点，goal 保持 active。

最新活动锚点覆盖（2026-09-22）：本轮重新取得 Artificial Analysis GPT-5.6 Luna 详情、DataCurve 和 OpenAI 官方 Markdown 页面。AA 详情为 `3,861,318` bytes / `1b425d8f2a93f8418702ac7acbb62b26850b98fe20680f82205bed5fce38fed0`，DataCurve 为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；精确行仍为 `mini_swe_agent_gpt_5_6_luna_max`，301/448、Pass@1 `67.1875%`、Pass@4 `90.26548672566371%`。AA 当前速度为 `158.728370482714 tokens/s`、Intelligence task cost `0.17829726152289094`、1M context；与 9 月 21 日 provider 测量分开保存，不解释为模型升级。7890 取得 Luna、Reasoning、Agents、Tools、Tool Search、Prompt Caching 和 Compaction 官方页 HTTP 200，哈希详见研究笔记；GPT-5.6 保持资料级闭环，不新增重复 Transformer 正式章节。

当前执行补充（2026-09-22 历史快照）：GLM-5.3-Flash 的 SGLang `v0.5.20` tag 已确认 `glm5_next.py` 与配置入口，SGLang `main` 继续出现 projection/KDA/mHC/AMD 量化演进；vLLM `main` 已有 `vllm/models/glm5next/`，可见 indexer pool/tail cache、KDA state 和 MTP top-k/slot mapping。对照 vLLM `v0.29.0` tree 未发现该专属目录。2026-09-24 已进一步确认 v0.30.0 stable GLM5Next source 并完成 fixed-main 逐文件对照（见本计划末尾）；stable source、recipe、wheel、完整权重和目标硬件结果仍分栏，不能把 `v0.29.0+` recipe 门槛当作 v0.29.0 tag 已包含 GLM5Next。

当前执行补充（2026-09-22，标准版）：新鲜两榜快照没有产生八家重点厂商的新 canonical 模型；本轮转入已有双榜条目 `GLM-5.3`。HF revision `aca966e4e02791568aa6a4ced368624b3d897f42` 的 `glm_moe_dsa` 配置与 Transformers main 的 `GlmMoeDsaForCausalLM` 对齐；vLLM `v0.29.0`/main 和 SGLang `v0.5.20`/main 都沿 DeepSeek-V3.2 DSA 共用路径承载该入口。`sglang/srt/models/glm5_next.py` 属于 Flash/linear-attention 变体，不能迁移为标准 GLM-5.3 的实现证据。后续 gate 分别记录 DSA indexer/MLA 源码、完整权重加载、目标硬件 profiling、index/evidence recall、MTP acceptance、tool/verifier 和生产 SLO。

当前执行补充（2026-09-23，DeepSeek V4.1-Flash）：vLLM `v0.30.0` 于 `2026-09-22T05:20:54Z` 发布，tag commit 为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。固定 release API、registry、PyPI `vllm/0.30.0` metadata 和 NVIDIA/ROCm V4.1/DSpark package 文件，确认 stable release 已有专用 release surface；v0.29.0 保留为历史对照，SGLang stable 仍为 `v0.5.20` 且本轮未证明其 V4.1 专用 vision/runtime 已进入 stable。后续 gate 仍按完整权重、GPU/ROCm/NPU、真实 candidate/index recall、FP4 误差、DSpark acceptance/rollback、EPD、tool/verifier 和生产 SLO 分栏。

当前网络复访记录（2026-09-23）：三条代理先前短时失败后已恢复；`10.237.126.170:1234`、`10.24.27.134:7890`、`10.24.27.134:8098` 访问百度均 HTTP 200，访问 Artificial Analysis 中文首页和 DataCurve DeepSWE 也均 HTTP 200 且逐字节一致。最新续抓 AA 首页为 `1,783,592` bytes / SHA-256 `7880a715fd069d4e475e4d717b37971d1f6402e5492964bb3f58787524d0b5a4`，此前同日 `1,783,001` bytes / `2fd4bd...` 作为历史快照保留；DataCurve 为 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。本轮据此新增 `Claude Opus 5.5` 候选并切换活动锚点；仍按 canonical slug、effort/provider/harness 和快照哈希分账。

当前执行补充（2026-09-23，Claude Opus 5.5 官方 API contract）：7890 代理取得 `overview.md`、`whats-new-opus-5-5.md`、`migration-guide.md` 和 fast mode 官方 Markdown。已确认 `claude-opus-5-5`、1M/128K、always-on adaptive thinking、default `medium` effort、thinking block model/conversation binding、强制 tool choice 400、`computer_toolset_20260801` 迁移、progress-update blocks、on-demand compaction、inline tools 和 fast mode 的 `usage.speed`/独立限流。1234/8098 仍会把平台文档重定向到区域不可用页，因此按线路分别记录，不把区域差异写成模型差异。

当前执行补充（2026-09-23，Claude Opus 5.5 System Card 正文）：固定 PDF `17,795,106` bytes / SHA-256 `7311c9c6bbb16d012f1c12c7418b05949fcf7ae3e30d2c40f22050074b2a7378`，已用 Node.js `zlib` 与 ToUnicode/CMap 只读解析。训练数据边界为公开互联网、公开/私有、获许可用户和合成数据，knowledge cutoff 为 2026-06；评测多数使用最终 snapshot，但部分章节使用早期/替代 snapshot 或关闭生产 safeguards。新增并入库 Terminal-Bench 4.0 `66.36%`（xhigh、Claude Code `--bare`、5 trials）、ProgramBench `91.2%`（166 tasks）、OSWorld 2.0 partial/strict `81.8%/48.7%`（108 tasks、1080p、500 actions、5 runs、>100K tokens compaction）、CoBench 2.1 `55.8%`、AECI `169.36`、Cyber/Gray Swan/Alignment 数字，以及五 Agent team/DRACO 的 derived-latency 结果。Cyber 数字绑定关闭 safeguards，fallback、simulated environment、long trajectory 和 multi-agent caveat 均已写入证据边界。

当前执行补充（2026-09-23，下一锚点选择）：最新 Artificial Analysis 中文首页续抓为 `1,783,592` bytes / SHA-256 `7880a715fd069d4e475e4d717b37971d1f6402e5492964bb3f58787524d0b5a4`，DataCurve 为 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；首页前列的 `DeepSeek V4.1 Flash (max)` 已有 canonical 条目，AA 图表第三方 Intelligence Index 为 `39.456167472527`。八家重点厂商没有新的模型名称，故切换到已有 `deepseek-v4-1-flash`，不从 runtime 仓库另建模型。

当前执行补充（2026-09-23，GPT-6 Sol）：三条代理取得 AA `gpt-6-sol` 详情页逐字节一致，`547,729` bytes / SHA-256 `88841ae9837f189a6ab739d8fa2bad6d142163b5b6486cab9181ebbfbc75f116`；max Intelligence Index 为 `47.5276426437724`。DataCurve 当前没有精确 `mini_swe_agent_gpt_6_sol_*` 行。OpenAI 官方模型页及 Reasoning、Agents、Tools、Compaction 快照已固定，确认 `gpt-6-sol`、1.05M/922K/128K、`none`--`max` effort、Responses/Chat Completions function-calling 边界、工具目录、GPT-6 family 的 mode/effort 分离与 `configuration_update`、Agent runtime 所有权和 compaction replay 规则；这些是 API/runtime 证据，不是内部架构披露。

当前执行补充（2026-09-23，GPT-6 Luna）：三条代理取得 AA `gpt-6-luna` 详情页逐字节一致，`3,992,084` bytes / SHA-256 `cda7ec1a8b90bfb76c85624f31cab242a0779732f68d65641b0dee5929a5d0f1`；release date `2026-09-22`、max Intelligence Index `37.2559686869738`、median output speed `153.87508473888 tokens/s`。DataCurve 当前没有精确 `mini_swe_agent_gpt_6_luna_*` 行。7890 取得 OpenAI 官方模型页 `4,019` bytes / SHA-256 `561a86af72eced9a76a4e3a96c45ae7fff7782732514a8429b55ebddb4404945`，确认 Luna 的 focused/high-volume 定位、1.05M/922K/128K、2026-05-18 knowledge cutoff、`none`--`max` effort、Responses/Chat Completions/Batch、工具目录和 `$0.10/$0.50` token 价格合同；1234/8098 返回 403，仅记录为线路边界。Luna 与 Sol 共享的 reasoning/Agents/tools/compaction 快照仅作为 GPT-6 family runtime 证据复用，不迁移 Sol 的指标或 Agent 结果。

当前执行补充（2026-09-23，GPT-6 Sol contract audit）：7890 重新取得两张指定排行榜：Artificial Analysis 中文首页 HTTP 200，`1,783,769` bytes / `e1acf794bd45f380ce3220a20c68de7a274f50ff7eb162a527a9faf4d58a166a`；DataCurve HTTP 200，`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。AA `gpt-6-sol` 详情当前 HTTP 200，`3,994,562` bytes / `3d08f91c71e2770a4ef84b593c2a197d711193bbcbd8c3a1f330393b65f0ea86`；Intelligence Index 仍为 `47.5276426437724`，速度当前为 `126.038858615917 tokens/s`。速度变化按 provider/采集时点漂移记录，不解释为模型 revision；DataCurve 仍无精确 Sol 行。

本轮官方快照已重新记录：模型页 `3,991` bytes / `5566589803b6f0ac79c00125a7c1af9291e46d5724c1c414f932d36ab92a8ad0`，Reasoning `70,315` / `93327c5eb19df7ecf8c0bd9d581f58052dee9f090c38bb48906110b08d0ce251`，Agents `5,432` / `df4f61b609550619a3f9e445c66d28318b9f681affe1a819ef0098cde32bf43a`，Tools `33,282` / `4722fa102070178c1a1d603e718c69575c7c1768fa901218eab359bafdcba341`，Compaction `14,272` / `73fd2fd1afd44bd6f29ce7bd86fd0ae3413c98ec00ae5476879e98171b0b60dd`，Prompt caching `47,098` / `69680fbec38e31a7abc8b0e33a6582b0aae29687e8de98404789337a55b55e2e`。新增 [`gpt6_sol_contract_audit.py`](research/model-update-2026-09/code/gpt6_sol_contract_audit.py)，主流程输出 `ok=true`、`evidence_level=local_protocol_toy`、`network_called=false`，覆盖 mode/effort、update/compaction 冲突、预算、计价、工具权限、幂等和 verifier。

当前状态为 **AA 单榜资料级闭环 + 当前时点复验 + local protocol toy**。下一步先完成全库 Python、Markdown 围栏、相对链接和 `git diff --check` 质量门禁；然后继续只从两张排行榜的八家重点厂商 canonical 集合选择下一项未收口锚点。没有 DataCurve 精确行、参数/架构/完整训练 recipe、目标硬件 profile 或生产 acceptance 时，保持 `unverified`，goal 保持 `active`。

当前执行补充（2026-09-23，Gemini 3.8 Flash Interactions）：通过 `10.24.27.134:7890` 获取 Google 模型页、Thinking、Interactions、Tool combination 和 Thought signatures；五个快照大小/哈希已写入 `gemini-3.8-flash-source-notes.md` 与 `source-index.md`。当前表明确 stable `gemini-3.8-flash` 为 `1,048,576/65,536` 输入/输出、默认 medium、仅 low/medium/high；`max_output_tokens` 包含 thought + visible output，触顶可 `incomplete`。Interactions 默认 store=true，付费/免费保留 55/1 天，store=false 不可继续 previous interaction，可 delete；previous interaction 只保 history，tools/system/generation config 为 interaction-scoped。Tool combination 补齐 call/result id、server/client-side 工具、validated mode 与 context circulation。Thinking 与 Tool combination 对标准 function call signature 的位置描述不完全一致，已作为证据冲突保留。

当前执行补充（2026-09-23，Gemini replay toy）：新增 `research/model-update-2026-09/code/gemini_interactions_replay_demo.py`，标准库合成验证 stateful/stateless history、signature/id 保留、SSE step 顺序/终态、store/delete/retention、modality gate；输出 `ok=true`、15 个有序事件、`network_called=false`。这是 local protocol toy evidence，不升级真实 API、模型质量或生产 SLA。下一步仅在有授权/预算时做真实 capability probe、thinking 消融、工具组合、长上下文/cache billing 和独立复现；没有独立 3.8 架构/训练报告时不新增 Transformer 正式章节。

本轮环境门禁：当前环境没有 `torch`、`transformers`、`vllm`、`sglang`、`safetensors`，`nvidia-smi` 无法连接 NVIDIA driver，且未发现标准 GLM-5.3 权重分片。因此在依赖、权重和目标硬件到位前，完整权重加载、数值/召回、MTP、profiling 与生产验收只能保持 `unverified`，不得由 source entry、registry、recipe 或榜单分数越级替代。Opus 5.5 的 System Card 数字也没有被误写成本机复现或生产 SLO。

## 2026-09-24 GLM-5.3：7890 复验完成后的执行计划

本轮已用 `10.24.27.134:7890` 重新取得两榜、标准 GLM-5.3 详情、Z.ai Markdown、博客壳和正文 JS。八家重点厂商没有新增 canonical 模型，后续仍只从 Artificial Analysis 与 DataCurve 已有集合选锚点；不从官方目录、论文、HF 或 runtime 仓库另建模型。

GLM-5.3 当前执行状态更新为：**双榜内容专题闭环 + stable/main runtime source evidence**。升级依据是第 89 章、研究笔记、SAO/环境—verifier/slime 资料、题库/练习/项目/知识图谱和标准 DSA runtime source entry 已经形成面试闭环；7890 复验只刷新证据时点，不改变模型身份。

保留的 GLM-5.3 门禁按优先级执行：

1. 先运行 `research/model-update-2026-09/code/glm53_compaction_contract_audit.py`，确认本地 toy、代码围栏和链接没有回归。
2. 在具备条件时按 pinned config/revision 验证完整权重加载、stable wheel、index/evidence recall、MLA/indexer cache recovery、FP8/量化误差、MTP 和目标硬件 profile。
3. 继续寻找并记录 5.3 专属 compaction schema/阈值、完整 post-training recipe、独立技术报告/benchmark、tool/verifier acceptance；找不到就明确保持 `unverified`，不以 SAO 论文替代。
4. 完成本轮全库 Markdown 链接、代码围栏、Python 编译和 `git diff --check` 后，再从两榜当前 canonical 集合选下一个未收口锚点，优先 `GPT-6 Luna` 等仍为资料级闭环的既有条目。

博客正文 JS 哈希 `f809bf…5d0f3` 与既有快照一致；当前 AA 动态数值、DataCurve 系统分数和官方博客发布方 benchmark 都继续按来源/配置/环境分账，不能写成模型 revision 或裸模型能力。

## 2026-09-24 GPT-6 Luna：7890 当前时点复验

本轮 7890 代理对 AA 首页、Luna 详情、DataCurve 和 OpenAI 官方模型页均返回 HTTP 200。AA 首页 `1,783,893` bytes / `3b895865547acf9b8d9d67d0c2a9a0a0369648f925014b2367817e1f5ac42f36`；Luna 详情 `3,974,113` bytes / `8a4603328627740cab856ad4d9b0b37415697b66cc33c59220b0f16615327c81`；DataCurve `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；OpenAI 模型页 `4,019` bytes / `561a86af72eced9a76a4e3a96c45ae7fff7782732514a8429b55ebddb4404945`。AA Index `37.2559686869738`、cost/task `$0.06809498628701058` 稳定，速度 `131.449006457181 tokens/s` 只记 provider 时点；DataCurve 仍无精确 Luna 行。

Luna 状态继续为 **AA 单榜资料级闭环**，本轮未出现 Luna 专属架构、训练 recipe、技术报告或 API 文档变化。当前已覆盖的正式落点和面试训练继续复用第六/十六/十七/二十/二十四册与题库，不新建重复章节。下一步仍从两榜当前重点厂商 canonical 集合中选择有未闭环高价值证据的既有锚点；不以官方目录扩充模型清单。

## 2026-09-24 Claude Opus 5.5：7890 复验与 System Card 治理证据审计

用户提供的 `curl` 已证明 `10.24.27.134:7890` 可达。本轮从当前环境经 7890 取得 Artificial Analysis 首页 `1,783,893` bytes / SHA-256 `3b895865547acf9b8d9d67d0c2a9a0a0369648f925014b2367817e1f5ac42f36`、Opus 5.5 详情 `3,809,722` bytes / `683005dd64dba034487fa6207167ce8239161e83cf198cd03d2a9445204caf23`、DataCurve `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1` 和官方 model page `14,244` bytes / `3325e10a852cc3f40cfaf01737d97a5cdf3a37b4cd7c0b68005e204ea32b1dd4`，均 HTTP 200。AA 指数 `57.6223698102963`、cost/task `$5.982012019521066` 未变；DataCurve 仍无精确 Opus 行；System Card PDF 固定文件 SHA-256 校验通过。

研究笔记与第二十册已包含具体评测条件。本轮在第八册第 11 章加入治理层的证据账本，把 model snapshot、safeguard 状态、harness、成功定义/分母、fallback 实际模型与 metric clock 绑定，避免将 OSWorld partial/strict 或多 Agent derived latency误写为单一裸模型指标/生产 wall-clock。内容状态为 **AA 单榜内容专题闭环 + System Card/API contract + local protocol toy**；独立复现、内部架构、完整训练 recipe、精确 DataCurve Agent 行、目标硬件和生产 acceptance 仍为 `unverified`。

下一锚点仍按两榜当前 canonical 集合选择尚有高价值权威证据缺口的重点厂商模型；不从官方目录另发现模型，也不因动态页面哈希变化判定 revision。goal 保持 `active`。

## 2026-09-24 GPT-6 Luna：7890 连通后的同日快照与驻留边界

用户提供的百度响应确认 `10.24.27.134:7890` 当前可达。本轮复核的 AA Luna 同日后续详情为 `3,974,386` bytes / SHA-256 `9c6376c8ca63fe1ed56fcc6a85cb5042900ba3ec96df92512760d85002cf594f`；Index 与 cost/task 保持 `37.2559686869738`、`$0.06809498628701058`，median output speed 从较早快照的 `131.449006457181` 变为 `132.242651126596 tokens/s`。这是 provider/测量时点变化，不足以推断模型 revision。此前详情快照 `3,974,113` bytes / `8a4603328627740cab856ad4d9b0b37415697b66cc33c59220b0f16615327c81` 保留作历史记录。

DataCurve 仍无精确 `mini_swe_agent_gpt_6_luna_*` 配置，不迁移其他 GPT 的 Agent 结果。OpenAI 官方 data-residency 文档当前快照为 `79,672` bytes / SHA-256 `4ea5386847a06cbaae7eef04f691d395fed3abcd925b133c4a808e999b700648`：Luna 的 EU residency 只对 Standard processing 的 Responses 与 Chat Completions 明确可用；regional storage 与 regional processing 不等价，system data 和 Remote MCP 第三方数据不自动包含在保证内，非美国地区还需满足相应 retention/abuse-monitoring 资格，适用地域费用单独计入成本账。

本轮已将活动锚点和快照状态同步到进度、研究笔记、模型盘点、榜单解释与来源索引。Luna 仍为 **AA 单榜资料级闭环**；没有新增专属架构、训练 recipe、技术报告或精确 DataCurve 行。下一步先完成本轮跨文件一致性检查，再按两个排行榜现有重点厂商 canonical 条目选择尚有高价值缺口的锚点；goal 保持 `active`。

## 2026-09-24 GPT-6 Sol：榜单复验与 EU data-residency 面试点

本轮复抓 Artificial Analysis `/zh`（1234 代理：`1,783,966` bytes / `0baa28c8ab69a104a078645983ca843c95c5db879b6c52789104aa3d1758efb8`）与 DataCurve DeepSWE（7890 代理：`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`）。AA 八家重点厂商和 DeepSWE 配置集合均未出现新 canonical 锚点；因此继续沿排行榜已有的 `GPT-6 Sol` 补其高价值服务合同，而非从 OpenAI 官方目录发现模型。

Sol AA 详情当前为 `3,976,802` bytes / SHA-256 `ff0aeaedb21ad7a672653b3d019c0574c6e468a74808db1f3973731d311d5ba5`；max Index `47.5276426437724`、cost/task `$1.0564240894076389` 稳定，速度 `109.551294011457 tokens/s`，与旧观测 `126.038858615917` 的差异仅按 provider/采集时点处理。DataCurve 没有精确 `mini_swe_agent_gpt_6_sol_*` 行；不迁移 Astra、Luna 或其他 GPT 的 Agent 成绩。

OpenAI 官方 [GPT-6 Sol model page](https://developers.openai.com/api/docs/models/gpt-6-sol.md) 当前 `3,991` bytes / `5566589803b6f0ac79c00125a7c1af9291e46d5724c1c414f932d36ab92a8ad0`，注明 EU data residency only with Standard processing；[data residency guide](https://developers.openai.com/api/docs/guides/your-data.md) HTTP 200，`79,672` bytes / `4ea5386847a06cbaae7eef04f691d395fed3abcd925b133c4a808e999b700648`，明确 Sol/Luna EU residency 仅对 Standard processing 的 Responses/Chat Completions 可用。可考点是区分 `Standard processing` 与 GPT-6 `reasoning.mode=standard`，以及 regional storage/inference processing、system data、Remote MCP、retention controls 和 10% regional uplift；Batch/Flex/Fast 价格不能替代 eligibility 证明。

已同步研究笔记、模型盘点、榜单解释、source index、第二十四册 32.55.3、面试题、练习、知识图谱、PAPERS/PROJECTS/BOOK_SERIES 与本进度。状态仍为 **AA 单榜资料级闭环 + 当前时点复验 + local protocol toy**，不表示内部架构/训练 recipe、精确 DataCurve Agent 行、完整权重、目标硬件、独立 benchmark 或生产 acceptance 已确认。下一步继续从两个排行榜已有的八家重点厂商条目中选择尚有高价值证据缺口的锚点；goal 保持 `active`。

## 2026-09-24 DeepSeek V4 Flash：SGLang serving 补证与配套闭环

本轮继续沿两榜已有的 DeepSeek V4 Flash 锚点推进。用户提供的 `curl` 响应确认 `10.24.27.134:7890` 当前可访问百度；此前工作区经该代理取得的 Artificial Analysis `/zh`（HTTP 200，`1,782,611` bytes / SHA-256 `566b4adab724bd312436a302be3f0f4c5d9f7713188e9efb078cb636faddd02b`）和 DataCurve（HTTP 200，`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`）与既有同日快照相同；没有发现新的重点厂商 canonical 模型。

SGLang 官方 `releases/latest` 仍为 `v0.5.20`（2026-09-18 发布）。本专题记录三类 serving 知识：PR #34565 在混合注意力状态下缓存 SWA 分支点，避免共享前缀分叉时 Full KV 命中但 SWA state 缺失；指定 V4-Flash-0731 workload 的 token hit rate 为 `43.81% → 60.75%`、mean TTFT 为 `1,569.93 → 1,069.58 ms`，属于 PR 自报结果。PR #30805 的 B200/SM100/103 FP8、TP=1 单元 kernel 对照报告约 `1.2x` prefill、`1.45x` decode；PR #29927 在 4× RTX PRO 6000/SM120 上的 paged-MQA indexer、sparse prefill、FP4 MoE 路径报告相对慢速 torch fallback 最高 `3.4x` TPOT，并披露 HC prenorm 的独立贡献。不同硬件、baseline 和测量范围不可直接横比；这些结果不是端到端模型或独立复现。PR #39171 仅支持 v0.5.20 含 V4.1 相关 FlashMLA pin 更新，不证明完整 V4.1 vision/runtime 已进入 stable。

已同步 `deepseek-v4-source-notes.md`、`deepseek-v4.1-flash-source-notes.md`、来源索引、模型盘点与榜单解释、第二十一册第 77 章，以及 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md` 和 `KNOWLEDGE_GRAPH.md`。V4 Flash serving 补证并入既有 CSA/HCA 专题，不新增重复章节。仍待完整权重加载、相同目标硬件/固定 workload 的独立 benchmark、cache/state 恢复、生产流量端到端 SLO 与 acceptance 验收；goal 保持 `active`。

## 2026-09-24 GPT-6 Sol：7890 当前状态与动态 effort 合同复核

本轮用确认可用的 `10.24.27.134:7890` 重新验证两榜。沙箱内请求发生 TLS EOF，按授权在 sandbox 外同代理复试成功；Artificial Analysis `/zh` 为 `1,783,966` bytes / `0baa28c8ab69a104a078645983ca843c95c5db879b6c52789104aa3d1758efb8`，规范 DataCurve `https://deepswe.datacurve.ai/` 为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`。八家重点厂商无新 canonical 模型；`www.datacurve.ai/deepswe` 的 404 是错误地址，不代表榜单不可访问。

通过 OpenAI 官方 `platform.openai.com/docs/...` clean HTML 路径（最终跳转到 Developers 官方文档）获取 GPT-6 Sol model page、Reasoning 和 Compaction 页面；对应响应 `432,023/1,070,709/455,842` bytes，SHA-256 分别为 `e8a4f18b6e60883740510c3c521801867e90961113e6a4b2f41959a9d606c836`、`3bcb1b7771731d3269783464180b72c2b45283688118dda185ebd2d65e84616c`、`d3825ca5955cba651e7fbf368a8fe6905056fc726cfc6d47a708d4f1b37dfa8a`。当前正文说明 GPT-6 family 的 `configuration_update` 仅用于 standard single-agent、可置于 Responses 或 WebSocket `response.create`；effective effort 保持至覆盖，但 response 的 `reasoning.effort` 仍记录 request-level value；追加 update 而不改变 request config 可保持原 prompt prefix、利于 cache reuse，但并不保证 cache hit。保留相邻 update、automatic compaction/truncation、独立 `/responses/compact` 的拒绝边界，并在显式 compact 后补 fresh update。

本轮把 telemetry 陷阱同步到 GPT-6 Sol 研究笔记、第十六/二十册、题库/练习、知识图谱和本进度；扩展本地 toy 检验 session effort 覆盖与 response field 分离、stable prefix digest 不变，并明确不代表真实缓存命中。脚本主流程 `ok=true`、`network_called=false`，`git diff --check` 通过。下一步从两榜已有八家厂商 canonical 中挑选仍有高价值证据缺口的锚点；不扩大厂商范围、不从官方目录发现新模型，goal 保持 active。

## 2026-09-24 Gemini 3.8 Flash：signature 字段范围冲突复核

用户给出的 curl 已证明 `10.24.27.134:7890` 可达；当前环境对同一代理的百度请求为 HTTP 200。继续使用两个排行榜已确认的 Gemini 3.8 Flash 作为锚点，不从 Google 官方目录扩展模型清单。重新取得 Google 官方 Thinking、Tool combination、Function calling 和旧 Thought signatures Markdown；对应大小/哈希与字段判断见 [`gemini-3.8-flash-source-notes.md`](research/model-update-2026-09/gemini-3.8-flash-source-notes.md)。

Thinking 当前文档明确写出 GenerateContent signature 可附在任意 part（包括 functionCall），但在 Interactions 中 signature 仅出现在 thought/built-in tool steps；Tool combination 仍把 Interactions 的 Gemini 3+ 自定义 function_call/function_response 也列入带 signature 的 tool steps。两页对 Interactions 的字段位置仍然冲突，Function calling 页的 SDK 自动处理说明不能裁决 schema。已同步第二十册第 23 章、研究笔记、source index 和 model inventory；不合成真实响应、不改 toy 验证范围。无 Gemini API key/真实 endpoint 调用，本轮不宣称 capability probe 完成；资料专题仍闭环，signature 行为保持 unverified。下一步完成链接/Markdown/Python 与 diff 门禁，再从两个排行榜的八家重点厂商既有 canonical 集合挑下一项高价值缺口。

## 2026-09-24 Kimi K3：vLLM v0.30.0 stable/runtime 更新

沿两榜已经发现的 Kimi K3 锚点继续推进；用户给出的 `10.24.27.134:7890` 百度请求成功，当前环境也取得榜单与 vLLM 固定 tag 快照。vLLM `v0.30.0` release commit 为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，于 2026-09-22 发布。关键版本结论：`v0.29.0` 已有 K3 registry/NVIDIA model/DSpark source entry；`v0.30.0` 是 runtime 实现演进，不是首次支持。

已固定 PyPI metadata、x86_64 wheel artifact metadata、registry、K3 NVIDIA model、DSpark 与 package init 的大小/hash；wheel 没有下载或安装。源码增量包括 MegaMoE IPC transformed-weight zero-copy reuse、streamed parameter 完成后 post-load finalize、跨 PP auxiliary hidden states 和 AttnRes 边界拒绝、DSpark context-KV 的 dtype/scale/layout gate，以及 KDA state dtype 账本。它们属于 serving framework 实现证据，不是 K3 新训练算法，也不等于完整权重加载、目标 GPU/ROCm 正确性、cache recovery、DSpark acceptance 或生产 SLO。

已同步 [`kimi-k3-source-notes.md`](research/model-update-2026-09/kimi-k3-source-notes.md)、[第二十一册第 88 章](book-21-transformer-architecture-evolution/chapters/88-kimi-k3-kda-stable-latentmoe与百万token-agent.md)、来源索引、模型盘点、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md` 和 `KNOWLEDGE_GRAPH.md`。下一步执行 Markdown/链接/代码和 `git diff --check` 门禁；然后从两榜已有八家厂商 canonical 集合选仍有高价值缺口的锚点，不扩大候选范围；goal 保持 `active`。
## 2026-09-24 Claude Fable 5.1：下一锚点的专题闭环

本轮沿 AA 已发现的 `claude-fable-5-1` 推进；AA 经 7890、DataCurve 经 8098 均返回 HTTP 200，同日 DataCurve 快照未变。未发现八家重点厂商的新 canonical 模型。Anthropic 官方 Thinking 文档补齐 Opus 5.5 与 Fable 5.1 thinking-block 的 Claude API 单向兼容边，并明确 model-binding 与 prefix-binding 是两个独立门禁。知识已落在第二十册第 21 章 21.29、研究笔记和面试配套材料；协议 toy 仅作 synthetic/local 证据。

当前闭环：**AA 单榜内容专题闭环 + 官方 System Card/API contract + local protocol toy**。仍未确认 Fable 5.1 参数与内部架构、完整训练/后训练 recipe、精确 DataCurve Agent 行、独立 benchmark、真实 endpoint 行为、目标硬件 profile 和生产 SLO。下一轮仍只从 Artificial Analysis 与 DataCurve DeepSWE 的八家重点厂商 canonical 集合选锚点；不从 Anthropic 文档或外部论文另发现模型。

## 2026-09-24 7890 当前工作区复验与下一锚点选择

用户提供的百度响应证明其终端可通过 `10.24.27.134:7890` 访问外网；本工作区也经同一代理直接取得 Artificial Analysis `/zh`（HTTP 200，`1,782,611` bytes，SHA-256 `566b4adab724bd312436a302be3f0f4c5d9f7713188e9efb078cb636faddd02b`）与规范 DataCurve DeepSWE 页面（HTTP 200，`268,036` bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`）。两份内容与已有同日快照一致；八家重点厂商的 canonical 集合未见新增模型。榜单外的厂商条目不进入本项目跟踪范围。

完成 `git diff --check`、研究代码目录 Python 编译、全库 Markdown 围栏配对检查，以及剔除代码/数学片段后的相对 Markdown 链接检查（1,495 个链接，缺失 0）。该次复验没有改动模型研究内容；其后已从两榜既有重点 canonical 锚点选择 DeepSeek V4 Flash serving 补证作为阶段专题，不扩大候选范围。goal 保持 `active`。

## 2026-09-24 GLM-5.3-Flash：vLLM v0.30.0 stable/fixed-main 源码审计

下一锚点从两榜已确认且仍有高价值 serving 证据缺口的 GLM-5.3-Flash 选择；没有从 vLLM release 或 runtime registry 新增模型。官方 vLLM `v0.30.0` release 于 `2026-09-22T05:20:54Z` 发布，commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。GitHub Contents API 的 `vllm/models?ref=v0.30.0` 返回 `vllm/models/glm5next/` 目录，tree SHA `118165530271cb3aa749d77bf927499957db2bd4`；此前固定的 `v0.29.0` tree 未检出该目录。

证据已升级为 **v0.30.0 stable source feature paths confirmed + fixed-main deltas documented**。stable `nvidia/attention.py` 已含 IndexerCache/TailCache；KDA 与 multimodal 对应文件和 main blob 相同；MTP diff 仅 import path；attention/indexer 文件存在 sparse-indexer 路径、pool-length workspace 与 RoPE 映射差异；main `model.py` 增加 Quark scale/gate-up mapping；main KPool tail kernel 按真实 stride 寻址，stable 版本的 dense offset 需要在目标 cache layout 下验证，当前没有据静态 diff 宣称 stable bug。

逐文件清单、Git blob、差异与证据边界已同步至研究笔记、source index、model inventory、inventory interpretation、第二十一册第 84 章、PAPERS、INTERVIEW_BANK、EXERCISES、KNOWLEDGE_GRAPH 和 progress。未下载/安装 wheel，未加载完整权重，未实测 cache/state recovery、MTP acceptance、EPD、目标硬件正确性/性能或工具/verifier SLO；下一步需以固定 artifact、模型 revision、cache layout、硬件与 workload 做验收。Goal 保持 `active`。

## 2026-09-24 Gemini 3.8 Flash：Interactions API reference 与固定 SDK schema 补证

沿两个排行榜已确认的 Gemini 3.8 Flash 锚点继续；用户终端提供的 `10.24.27.134:7890` 百度请求可达，工作区此前也经同一代理取得 Google 官方 Interactions API reference（HTTP 200，`787,186` bytes / SHA-256 `a06a779e779608c470b178dce0361fb3548ae5121b0c7a17766f8dcce91ceb2e`）。本轮不从 SDK/Google 文档扩展模型候选。

固定 Google `python-genai` revision `4742c9a5c213a587add126a500a271824e2f0add` 的 Interactions types 确认：`ThoughtStep.signature` 可选；`FunctionCallStep.id` 必需但 typed schema 未声明 signature；`FunctionResultStep.call_id` 必需且也未声明 signature；Google Search call/result 显式声明可选 signature。基础 `BaseModel` 的 `extra="allow"` 会保留额外字段，step union 使用 lenient open-union 并以 `UnknownStep.raw` 保存未知 payload。因此 SDK schema 没有要求 custom-function signature，但这不是服务端拒绝 extra fields 或 endpoint 不会返回 signature 的证据；Thinking 与 Tool combination 的官方范围差异仍待真实 endpoint capability probe。Interactions wire correlation 已明确为 `function_call.id` → `function_result.call_id`，与 GenerateContent 的 `functionCall`/`functionResponse` 命名分开。

更新研究笔记、第二十册第 23 章、source index、model inventory、PAPERS、INTERVIEW_BANK、EXERCISES、PROJECTS、KNOWLEDGE_GRAPH、计划与进度；扩展 `gemini_interactions_replay_demo.py`，输出 SDK-schema 缺省签名与较严格文档 profile 的 toy 对照，`ok=true`、15 个有序 SSE 事件、retention `55/1`、`network_called=false`。这是 synthetic/local protocol evidence，不是真实 API probe。下一步跑相关章节代码、Markdown/链接与 diff 门禁，再依据两榜当前快照从既有八家厂商 canonical 锚点中选下一个高价值缺口；goal 保持 `active`。

## 2026-09-24 Claude Opus 5：System Card 安全评测方法专题

- 通过用户确认可用的 `10.24.27.134:7890` 获取两榜和 Anthropic 官方资料。Artificial Analysis `/zh` 为 `1,781,428` bytes / SHA-256 `15e9763f0d516791e7b7c2c5679e87e5878eaffed4bef31e485c9043b758ba59`；Opus 5 详情 `3,977,410` bytes / `18acc956d77c5b49c391eb85b46ef2c7fdbd7edd282ddf2275940bd418d25774`，Index `50.7771115797629`。DataCurve DeepSWE 为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，有精确 `mini_swe_agent_claude_opus_5_max` 行 `327/444`；任务分数绑定 `mini-swe-agent` harness、工具、环境与 verifier。
- System Card PDF 当前响应 `16,281,258` bytes / SHA-256 `0950dae1ba6b341e4f1a009e535e0f025625efeca21bd043a9a5ae148a3f2e6b`，与既有 PDF 完全相同；用仓库 parser 新抽取正文 4,641 行/334,056 bytes，文本哈希 `4ae20472ab82c967d90f386239ee6987ddf74b1124db6b44a1dea69a576e1f4c`。卡片自身 changelog 日期为 2026-08-19；本轮是正文补读，不宣称官方当天发布新版本。
- 新知识：ART 饱和后的 IPI/adaptive red-team、attempt 与 scenario 两种 ASR、tool-result probe（输入侧）+ action classifier（动作侧）的纵深防御、Cowork harness mismatch 后重跑所有 baselines，以及 CB-1/CB-2/ASL-3 风险判断的证据边界。
- 已同步 Opus 5 研究笔记、source index、model inventory、inventory interpretation、第八册第 11 章 6.5、第二十册第 19 章 19.38、PAPERS、INTERVIEW_BANK、EXERCISES、PROJECTS 和 KNOWLEDGE_GRAPH。
- 状态：**Claude Opus 5 System Card 的 Agentic Safety/评测方法专题闭环**；不代表参数、架构、训练配方、独立评测或生产风险/SLO 已验证。运行 `git diff --check`、研究 Python、链接/围栏检查后，goal 继续 `active`，再从两榜已有重点 canonical 集合选下一锚点。

## 2026-09-24 DeepSeek V4 Flash Vision：当前路由与多模态预算复验

下一锚点从 Artificial Analysis 已有的 `deepseek-v4-flash-vision` 选择；本轮没有从 DeepSeek 文档、V4.1 发布页或论文新增模型。7890 首次访问 V4.1 release 单页连接失败，重试 HTTP 200；AA 首页、Vision 详情、DataCurve 与其余 DeepSeek 官方页面均 HTTP 200。AA 首页 `1,781,428` bytes / `15e9763f0d516791e7b7c2c5679e87e5878eaffed4bef31e485c9043b758ba59`，DataCurve `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，与已有同日快照一致；重点厂商 canonical 集合无新增。

AA Vision 详情当前为 `3,967,107` bytes / `a5e5259ee84eac5aa88915dd6436ba155e265ede940ab663b52c4a681ae43ed7`；Index `34.8390628035969`、median output speed `217.762710468086 tokens/s`、TTFC `0.970116780000126s`、cost/task `$0.314372044499279`。9/15 数字作为历史 provider/榜单测量保存，不据此推断模型 revision。DataCurve 仍无精确 `mini_swe_agent_deepseek_v4_flash_vision_*` 行。

官方 9/10 发布页与 9/24 Quick Start/Pricing 再次确认：旧 Vision alias 仍可请求，但对应模型已 retired；请求暂由 `DeepSeek-V4.1-Flash` 服务并按 `deepseek-flash` 峰谷价计费。Vision guide 仍为 resize 后最多 1,024 tokens/图，并明确最多 600 张、单图/请求体/合计体积、8192/4096 px 尺寸边界。知识已同步到 Vision 研究笔记、模型盘点、来源索引和第二十一册第 85 章。当前状态为 **AA 单榜资料级闭环 + 路由/图像预算当前时点复验**；无真实 API probe、精确 DataCurve 行、独立视觉架构/训练报告或独立 benchmark。下一步仍只从两榜已有 canonical 锚点选择，goal 保持 `active`。

## 2026-09-24 Qwen3.8-Flash-Next：固定版本 serving/runtime 源码补证

本轮继续沿 Artificial Analysis 已发现的 `Qwen3.8-Flash-Next` 锚点，不从 Qwen、vLLM 或 SGLang 仓库新增模型。7890 在用户终端对百度返回成功，但当前 agent 网络路径连 `10.24.27.134:7890` 超时（沙箱内和沙箱外只读重试均如此）；备用 `10.237.126.170:1234` 可用。本轮经 1234 刷新 Artificial Analysis `/zh`：1,781,428 bytes，SHA-256 `15e9763f0d516791e7b7c2c5679e87e5878eaffed4bef31e485c9043b758ba59`；DataCurve DeepSWE：268,036 bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`。二者与同日已有快照一致，八家重点厂商的 canonical 集合没有新增。GitHub API 经 1234 返回 HTTP 200；Qwen Flash-Next README 固定 commit `69885871a64393807d988b27b1b5e380e8f28526` 返回 HTTP 200、9,735 bytes。

固定版本核验结果：vLLM `v0.30.0` 的 Qwen4Exp registry/实现包含 QSA、PLE/N-gram、Gated Residual 和 MTP 源码/测试；当前 NVIDIA QSA 路径的激活/QKV 和主 KV 为 BF16，压缩 indexer-key cache 另支持 BF16 或 FP8 E4M3，且不能据此断言权重必须 BF16。SGLang `v0.5.20` 的 QSA indexer 在特定 graph-capture/token 条件下，可与当前 stream 的 Q/K/V 准备工作并行，之后再启动 attention；不是与最终 attention kernel 重叠。约 47.7 GiB FP8 PLE 表是 SGLang 源码注释；H100 80GB/GPU 对 51B PLE/N-gram 表余量不足是 vLLM recipe 的独立部署提醒。两边均有 pinned-host/UVA/offload 实现约束，recipe 数字不能外推为普遍性能承诺。

已同步 `qwen3.8-source-notes.md`、第二十一册第 83 章、`source-index.md`、`model-inventory.md`、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、本计划与 `progress_v2.md`。核验级别是 **固定版本源码与 recipe 文档审阅**：未运行 pytest、未安装 wheel、未加载权重、未做 GPU profile 或线上生产验收。Qwen3.8 专题的 serving 知识补证已收口；下一锚点仍须来自两榜现有八家重点厂商 canonical 集合，整体 goal 保持 `active`。

## 2026-09-24 Qwen3.5-Omni Plus / Flash：ARIA 与多模态时间对齐

沿用 Artificial Analysis 已存在的 Plus/Flash 两个 Qwen3.5-Omni 配置作为同一家族锚点；本轮没有从 arXiv、官方服务文档或其他仓库新增模型。用户终端给出的百度响应证明其 shell 可经 `10.24.27.134:7890` 出网，但当前 agent 网络路径对 7890 在沙箱内及获准沙箱外各超时一次。备用 `10.237.126.170:1234` 测试返回百度 HTTP 200，并成功刷新以下当前来源：

- AA Plus：HTTP 200，`3,826,061` bytes / SHA-256 `b2127d33c029c55eac112636964f2a9cb9196b69ca5f792c90a2a9ae79f387d9`；页面第三方字段有 256K、Index `20.3839890713777`、输出 `84.6202668152731 tokens/s`、输入/输出 `$0.40/$4.80` 每百万 token。
- AA Flash：HTTP 200，`3,834,828` bytes / `17bd65b3597f29af8a655e00d038b774dc2ae89c821848262b7d0e4bfc8d560c`。
- DataCurve：HTTP 200，`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`；无精确 `mini_swe_agent_qwen3_5_omni_*` 行，不迁移其他 Qwen 的 Agent 结果。
- [Qwen3.5-Omni Technical Report v2](https://arxiv.org/abs/2604.15804v2)：摘要 HTTP 200，`42,723` bytes / `7c93a28bd4e5bd795492e20ee6fcbb991920d5834bc6f9f1d4f83dfc0472a34a`；源码包 `2,984,159` bytes / `fd53a97d5be7eaa8c981c853f1e6a4faa9b86c69c5fe6f3dbcfa33b66abd81ff`。
- [阿里云 Model Studio Qwen-Omni 文档](https://help.aliyun.com/zh/model-studio/user-guide/qwen-omni)：HTTP 200，`402,771` bytes / `9140a68a7b0a98def6a8cca23e759359895046bd3fc6123b9c361c7526bd7578`；示例调用 `qwen3.5-omni-plus`、`stream=True`，文档列出 Plus/Flash 的 custom-voice 与 snapshot 边界。

面试知识已收敛为：Hybrid MoE Thinker/Talker；AuT 由 4 个 Conv2D 块下采样 16 倍、输出 6.25 Hz（约 160 ms）；保留 TM-RoPE 并插入秒级视频/音视频时间戳、随机插入音频 timestamp；ARIA 将文本和 speech token 合成单流，对任一前缀施加样本级累计 speech:text ratio 上限；Talker 再用 RVQ/MTP + 因果流式 codec 生成 waveform。训练披露 S1 encoder alignment、S2 约 4T（分类项直接相加约 4.29T）、S3 从 32,768 到 262,144；Thinker 做 specialist/on-policy distillation 与 interaction-aligned RL，Talker 涉及长上下文 CPT、DPO、rule reward + GSPO 和 speaker fine-tuning。100M+ 小时总体音视频、AuT 40M 与 Talker 20M+ 小时口径不同，不简单相加。

已新增 [`qwen3.5-omni-source-notes.md`](research/model-update-2026-09/qwen3.5-omni-source-notes.md) 与[第二十一册第 93 章](book-21-transformer-architecture-evolution/chapters/93-qwen3.5-omni-aria-aut-timestamp与流式语音.md)，同步来源索引、模型盘点/解释、PAPERS、题库、练习、项目、知识图谱、BOOK_SERIES 和进度。状态为 **AA 单榜内容专题闭环 + 官方技术报告/API 文档核验**；未下载/加载完整权重、未进行目标硬件 profile、独立 benchmark 或真实 API probe，生产 SLO/端到端语音验收仍 `unverified`。完成质量门禁后继续从两榜已有 canonical 集合选锚点；goal 保持 `active`。

## 2026-09-24 Kimi K2.7 Code：两榜复验与长周期 coding Agent 专题

本轮锚点只取自现有两榜，不从 Kimi API 页面或 arXiv 发现新模型。用户终端所贴百度响应证明 `10.24.27.134:7890` 对用户 shell 可达；本 agent 会话在沙箱内及沙箱外复测均于 10 秒连接超时。备用 `10.237.126.170:1234` 返回百度 HTTP 200，并成功刷新榜单。代理结论按网络路径区分，不将 agent 失败说成 7890 全局不可用。

| 来源 | 本轮快照 | 当前发现 |
|---|---|---|
| Artificial Analysis Kimi K2.7 Code | HTTP 200，4,047,429 bytes，SHA-256 1a8818a6ee812da88ba7b9a95b691ae7481dac7b55d7126084fbd1d5e964e56b | canonical slug 仍为 kimi-k2-7-code；Index 25.8121062401836、256K context 均是第三方/provider 字段 |
| DataCurve DeepSWE | HTTP 200，268,036 bytes，SHA-256 14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1 | 精确 default 行仍为 113 tasks、4 runs、452 attempts/138 passed，Pass@1 30.53%、Pass@4 61.06%，结果绑定 mini-swe-agent harness/tool/environment/verifier |
| Kimi 官方资源页 | HTTP 200，270,859 bytes，SHA-256 f3696d9842d595fa8777bfedb7dcd2b9251186ac965fe4e5d0cc0c3ba9a38f51 | 对已发现锚点作官方身份核验 |
| Kimi 官方 API 快速开始 Markdown | HTTP 200，12,445 bytes，SHA-256 7c1ecbbfb1d2103772a00ca99b6e058c7a6f018cb00744ae241042f7f8071ec7；HTML 455,287 bytes / 2613331b1ace91d15f9babd118aa854285e523cec4a598fab78dddf94a7530c0 | 可读文档的当前合同细节 |
| arXiv 精确搜索 | HTTP 200，16,442 bytes，SHA-256 a305e3e8bb51f7e9e64e44266a1dc58a7df5c6aae879b818f0c59654186af4fa | 对精确题名 Kimi K2.7 Code 的本次查询返回 no results；不把负检索扩写成永久不存在 |

官方 API 增量：256K context 与默认 max_tokens 32,768 是不同预算；thinking 默认 enabled 且不能关闭；temperature 1.0、top_p 0.95、n 1、presence/frequency penalty 0 为固定值，tool_choice 只允许 auto/none。文档建议多步工具循环保留 assistant reasoning_content，遗漏通常不报错但可能影响连贯性。图片/视频 token 动态计算，视频按关键帧形成输入；文档提供 token estimate、100 MB request-body 边界、图片 URL 不支持/需 base64、大视频文件上传建议、4K 图片与 1080p 视频建议，并展示宿主用 ffprobe/ffmpeg 按时间片段处理视频后回灌多模态结果。

当前 API 快速开始列出 kimi-k2.7-code-highspeed，并声明它与 kimi-k2.7-code 是同一模型；“普通版 5–6 倍”“典型中位输入约 180 tokens/s”“短上下文约 260 tokens/s”是带 workload 条件的发布方说法，且文档注明资源有限/体验可能波动。它只作为已发现模型的服务 route 变体，不新增 canonical 行。Kimi Code Bench v2、Program-Bench、MLS Bench Lite 相对 K2.6 提升 21.8%/11%/31.5%、Agent 基准约提升 10%也按官方自报保存，不作独立复现。

固定 Hugging Face revision 的 README/config 本轮经 1234 代理重试超时；原 2026-09-15 已固定 revision 和 hash 保持有效，但不冒称新快照。架构账本仍以固定 artifact 的公开字段为准：约 1T/32B active、61 层、384 routed/top-8/1 shared、MLA、MoonViT 约 400M、native INT4/group size 32 与明确的量化忽略模块。不能把 active 参数当作 resident weight，也不能把 1T×4-bit 当作实际 checkpoint/GPU 峰值。

已新增 [第二十一册第 94 章](book-21-transformer-architecture-evolution/chapters/94-kimi-k2.7-code-long-horizon-coding-agent.md) 和零依赖教学审计器 research/model-update-2026-09/code/kimi_k27_contract_budget_toy.py；脚本通过，输出 local_protocol_toy、network_called=false、全量 1T INT4 原始假设 465.661 GiB、假设性 256K MLA cache shape 17.156 GiB。二者都不是模型内存 profile、API probe 或生产验收。

研究笔记、模型盘点与解释、来源索引、第二十一册目录、BOOK_SERIES、INTERVIEW_BANK、EXERCISES、PROJECTS、KNOWLEDGE_GRAPH、计划与进度均已同步。当前状态为 **双榜内容专题闭环**；完整训练 recipe、完整权重加载、真实 endpoint、目标硬件 profile、独立 benchmark、线上工具接受率与生产 SLO 未确认。本轮全库门禁已通过：718 个 Markdown 文件无未闭合围栏，1,530 个代码/数学片段外本地链接无缺失，16 个研究 Python 文件 AST 可解析，`git diff --check` 通过。最新两榜快照与同日记录一致、无新增重点 canonical；下一步继续从两榜已有八家厂商 canonical 集合选择尚未覆盖的高价值知识缺口，不重复扩写已收口专题；goal 保持 active。

## 2026-09-24 后续锚点：Qwen3.6-35B-A3B

- 候选身份只按两个指定排行榜记录：AA 的 `qwen3-6-35b-a3b` 与 `qwen3-6-35b-a3b-non-reasoning` 归并为同一基础模型；DataCurve 无精确 Qwen3.6 行时不迁移其他 Qwen 的 Agent 成绩。
- 主要知识缺口落在官方模型卡 + 固定 chat template：交替 Gated DeltaNet/Gated Attention + MoE；`preserve_thinking` 的历史 transcript 序列化、与 `enable_thinking` 的职责区分，以及它不等同于持久 memory；补充 YaRN 静态缩放及 MTP serving 证据边界。
- 交付同步：Qwen3.6 source notes、第二十一册第 83 章 83.15、第二十册第 21 章 21.30、榜单清单、来源索引、PAPERS、题库/练习/知识图谱与本计划/进度。
- 下一步：完成受影响文件链接/围栏与 diff 门禁，再根据两个榜单中八家重点厂商 canonical 集合选择未覆盖的高价值知识缺口；遇到代理路径差异，分别记录用户 shell 与 agent 网络结果。

## 2026-09-24 后续锚点：Qwen3.6-27B 与 GDN Tree-Scan

- 锚点只按 Artificial Analysis 中 qwen3-6-27b / Non-reasoning 两个配置归并；DataCurve 没有精确 Qwen3.6-27B Agent 行，不搬用其它 Qwen 的分数。
- 官方 ModelScope 固定卡/config/template 支持 27B dense、64 层、周期性 3×Gated DeltaNet + 1×Gated Attention；preserve_thinking 与 Qwen3.6-35B-A3B 共享模板哈希，不能算作 27B 独有技术。
- 新发现的技术点来自外部单作者 arXiv 预印本 GDN Tree-Scan：树验证除 attention ancestry 外还需沿 parent path 做 GDN recurrent-state scan/replay，并只提交 accepted-chain state。实验数字是 Qwen3.6-27B-FP8、B=1、temperature 0.6 下的作者自报 decode 数据，不等同 Qwen 官方性能或端到端 task-wall 加速。
- 交付：扩写 Qwen3.6-27B source notes；同步第二十一册第 83 章 83.16.4、第二十四册第 61 章 61.31、来源索引、模型盘点、PAPERS、题库、练习与知识图谱。
- 网络：用户 shell 的 7890→百度请求成功；当前 agent 对 7890 的沙箱内/外请求均超时。备用 1234 成功访问 arXiv 与 GitHub，Qwen 官方博客仍未取到正文。此为网络路径差异，不称 7890 全局不可用。
- 下一步：跑链接/围栏与 git diff --check 门禁；再只从 AA 与 DataCurve 两榜八家重点厂商 canonical 集合选择下一个未覆盖的高价值技术缺口。整体 goal 继续 active。

## 2026-09-24 GLM-5.3：7890 复验与内容专题闭环升级

- 使用用户确认可用的 `10.24.27.134:7890` 重新抓取两榜和标准 GLM-5.3 资料；Artificial Analysis `/zh` HTTP 200，`1,784,793` bytes / SHA-256 `a374adfb81fea4fc68e7071ef191612d7f7d92c1c0e05412f3c338e7fe9edbc2`；DataCurve DeepSWE HTTP 200，`268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。八家重点厂商 canonical 集合没有新增，仍严格沿两榜已有模型推进。
- Artificial Analysis GLM-5.3 详情 HTTP 200，`4,057,703` bytes / SHA-256 `4f60e893bcfee4586479c77c9d43f1af9f9dee3145e92a23661666d7a072a3fd`；当前页面可见约 1M context、约 45 Index、约 `$2.01/task`、约 61 tokens/s、`$1.40/$4.40` input/output 和 open weights。均按第三方/provider 当前观测记录，不解释为 revision。
- Z.ai GLM-5.3 Markdown `23,446` bytes / `9545c3d6fb1cabfa5951928bbe9a535e6958561d3bd6b5f4354cd5d3b0f6c929`；博客壳 `598` bytes / `240cedb6d23b13b8bdd177e51410dbe1c7783fbd0cfca98be1e0af26688878c0`；博客正文 JS `30,414` bytes / `f809bf586f74b40a01ecd8d674f9ec3de469a5abeae125e2a849c9805605d0f3`。官方资料与既有快照一致；没有公开 5.3 专属 compaction schema、阈值或完整 recipe。
- 标准版 `glm_moe_dsa` / `GlmMoeDsaForCausalLM` 的 fixed config、Transformers、vLLM 和 SGLang source entry 已与第 89 章、研究笔记及配套题库/练习/项目/知识图谱对齐；Flash 版 KDA、视觉、双 state pool、EPD 不迁移。
- 当前状态由“资料级闭环 + stable/main runtime source evidence”升级为 **内容专题闭环 + stable/main runtime source evidence**。这表示面试知识闭环已经形成，不表示完整权重、stable wheel、目标硬件或生产服务验收完成。
- 未完成门禁：完整权重加载、index/evidence recall、MLA/indexer cache recovery、FP8/量化误差、MTP、目标硬件 profiling、5.3 专属 compaction 实现、完整 post-training recipe、独立 benchmark、tool/verifier acceptance 和生产 SLO。`glm53_compaction_contract_audit.py` 仍仅为 `local_protocol_toy`。
- 下一步：运行 GLM-5.3 compaction toy 与全库一致性检查；通过后从两榜当前 canonical 集合选择下一项未收口锚点，优先既有的 `GPT-6 Luna` 等资料级条目；goal 继续保持 `active`。

## 2026-09-24 GPT-6 Luna：两榜/官方合同复验

- 按 GLM-5.3 记录中的后续顺序，继续处理两榜既有的 `gpt-6-luna`，未从 OpenAI 官方目录另发现模型。使用 `10.24.27.134:7890` 对百度、Artificial Analysis `/zh`、AA Luna 详情、DataCurve 和 OpenAI 官方模型页请求成功；百度 HTTP 200，榜单对象及官方页面也均 HTTP 200。
- AA `/zh` 当前 `1,783,893` bytes / SHA-256 `3b895865547acf9b8d9d67d0c2a9a0a0369648f925014b2367817e1f5ac42f36`；AA Luna 详情 `3,974,113` bytes / `8a4603328627740cab856ad4d9b0b37415697b66cc33c59220b0f16615327c81`。Luna `max` Intelligence Index `37.2559686869738`、cost/task `$0.06809498628701058` 与既有观测相同；median output speed `131.449006457181 tokens/s` 记为当前 provider/采集时点字段。
- DataCurve 仍为 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，没有精确 `mini_swe_agent_gpt_6_luna_*` 行；不迁移 Sol、Astra 或 GPT-5.6 的 Agent 结果。
- OpenAI 官方 [GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna.md) 当前 `4,019` bytes / SHA-256 `561a86af72eced9a76a4e3a96c45ae7fff7782732514a8429b55ebddb4404945`，与原快照一致；1.05M/922K/128K、effort、endpoint/tool support 和 pricing 合同无可见变化。官方站内搜索路径返回 404，故直接核对并读取了精确官方模型页；此路径错误不表示页面不存在。
- Luna 内容仍落在第六、十六、十七、二十、二十四册现有正式小节和题库/练习中；本轮无新专属架构/训练披露，不重复建章。当前状态保持 **AA 单榜资料级闭环**，goal 保持 `active`。

## 2026-09-24 Claude Opus 5.5：7890 复验与 System Card 治理证据审计

- 使用 `10.24.27.134:7890` 读取 Artificial Analysis 首页、Opus 5.5 详情、DataCurve DeepSWE 和 Anthropic 官方 model page，均 HTTP 200。首页为 `1,783,893` bytes / `3b895865547acf9b8d9d67d0c2a9a0a0369648f925014b2367817e1f5ac42f36`；Opus 详情为 `3,809,722` bytes / `683005dd64dba034487fa6207167ce8239161e83cf198cd03d2a9445204caf23`；DataCurve 为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`；官方 model page 为 `14,244` bytes / `3325e10a852cc3f40cfaf01737d97a5cdf3a37b4cd7c0b68005e204ea32b1dd4`。
- AA 当前 Opus 5.5 Index `57.6223698102963`、cost/task `$5.982012019521066` 未变；DataCurve 没有精确 `mini_swe_agent_claude_opus_5_5_*` 行。System Card 固定 PDF 本地副本为 `17,795,106` bytes / `7311c9c6bbb16d012f1c12c7418b05949fcf7ae3e30d2c40f22050074b2a7378`，SHA-256 校验通过。
- 对照研究笔记、第二十册第 19 章和题库后，确认评测口径均已覆盖。本轮在第八册第 11 章新增治理审计案例：评测 claim 绑定 snapshot、safeguards、harness、成功定义/分母、fallback 实际模型及计时定义；OSWorld partial/strict 与多 Agent derived latency 不应简化成一个裸分数或线上 wall-clock。
- 已同步研究笔记、来源索引、模型盘点、榜单解释、`plan.md`、本进度及第八册治理章节。内容状态为 **AA 单榜内容专题闭环 + System Card/API contract + local protocol toy**；这不表示参数/架构/完整 recipe、独立复现、精确 DataCurve Agent 行、目标硬件或生产 acceptance 已通过。goal 保持 `active`。

## 2026-09-24 Grok 4.20：7890 复验与 AA benchmark freshness 边界

- 用户提供的 `curl` 结果证明 `10.24.27.134:7890` 可访问百度；本轮沿该代理取得 Artificial Analysis Grok 4.20 详情、DataCurve DeepSWE 和 xAI 官方 Grok 4.20 Markdown，均 HTTP 200。模型仍是 AA 已有锚点，不从 xAI 官方文档另发现模型。
- AA 详情为 `3,857,337` bytes / SHA-256 `d56d4557d8a661ee9c3015c6a3661f5baae6c00f9b16c5b2acf5eabcd3e61300`。当前页面标记 `deprecated: true`、`deprecatedTo: grok-4-3`，并明确仅默认 10K input token workload 的性能基准继续更新，其他 workload 结果历史且不再更新。页面仍显示 Index `25.6550155187053`（estimated）、2M context、速度 `106.190642707844 tokens/s`、TTFT `21.53034693s`、端到端 `26.23885972594964s`；这些数字应保留来源、workload 和更新时间边界，不能因刚抓取就统称为当前 benchmark，也不能从指标波动推断模型 revision。
- DataCurve 当前 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，无精确 `mini_swe_agent_grok_4_20_*` 行；不迁移 Grok 4.5/4.6 Agent 结果。xAI 官方模型 Markdown `1,564` bytes / `99057608aeb1a4e6ace5d87bf6c4b27b3bff4af5248babad475500da31fdc2e3`，内容未见变化，仍写 1M context。AA 的 deprecated/deprecatedTo 不是 xAI API 下线或迁移公告。
- 新的面试知识点已加入第二十册第 19 章、`INTERVIEW_BANK.md` 和 `EXERCISES.md`：每个 benchmark 结果应记录 snapshot、输入 workload、provider 与最后更新时点；页面抓取日期不能替代指标采集日期。已同步研究笔记、source index、model inventory、inventory interpretation、`plan.md` 和本进度。活动锚点保持在 Grok 4.20 的补证阶段，整体 goal 继续 active。

## 2026-09-24 GPT-6 Luna：7890 连通确认后的同日快照续查

- 用户贴出的 `http://www.baidu.com` 成功响应确认 `10.24.27.134:7890` 代理可达；当前环境也经该代理成功获取 OpenAI 官方 data-residency 文档。此证据只说明代理当前可用，不回溯证明此前失败请求成功。
- AA Luna 同日后续详情快照为 `3,974,386` bytes / SHA-256 `9c6376c8ca63fe1ed56fcc6a85cb5042900ba3ec96df92512760d85002cf594f`；Index `37.2559686869738`、cost/task `$0.06809498628701058` 未变。median output speed 从较早的 `131.449006457181` 变为 `132.242651126596 tokens/s`，保留为 provider/采集时点字段，不推断模型 revision。
- DataCurve 仍为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，没有精确 `mini_swe_agent_gpt_6_luna_*` 行。OpenAI 官方 residency 文档确认 EU residency 对 Luna 仅适用于 Standard processing 的 Responses/Chat Completions；regional storage 不等同 regional processing，system data 与 Remote MCP 第三方数据不自动包含在保证范围内，且资格/保留控制与地域附加费应单独核对。
- 已同步 `plan.md`、模型盘点、榜单解释、来源索引和 Luna 研究笔记。当前活动锚点为 GPT-6 Luna，状态仍是 **AA 单榜资料级闭环**；没有新 Luna 专属架构、训练 recipe、技术报告或精确 Agent 行，goal 保持 `active`。

## 2026-09-24 GPT-6 Sol：两榜复验与 EU residency 面试边界

- 刷新两个唯一模型发现榜单：Artificial Analysis `/zh` 经 `10.237.126.170:1234` 获取，HTTP 成功，`1,783,966` bytes / SHA-256 `0baa28c8ab69a104a078645983ca843c95c5db879b6c52789104aa3d1758efb8`；DataCurve DeepSWE 经 `10.24.27.134:7890` 获取，`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，与前次相同。对照八家重点厂商 canonical 集合，无新增模型/配置。
- 选取仍在 AA 的既有 GPT-6 Sol 继续补证。AA 详情为 `3,976,802` bytes / `ff0aeaedb21ad7a672653b3d019c0574c6e468a74808db1f3973731d311d5ba5`；Index `47.5276426437724`、cost/task `$1.0564240894076389` 稳定，速度 `109.551294011457 tokens/s`，旧观测 `126.038858615917` 作为 provider/时点历史，不解释为 revision。DataCurve 无精确 `mini_swe_agent_gpt_6_sol_*` 行，不迁移其他 GPT 的 Agent 成绩。
- 通过 7890 重新读取 OpenAI 官方 GPT-6 Sol model page（`3,991` bytes / `5566589803b6f0ac79c00125a7c1af9291e46d5724c1c414f932d36ab92a8ad0`）及 [data residency guide](https://developers.openai.com/api/docs/guides/your-data.md)（HTTP 200，`79,672` bytes / `4ea5386847a06cbaae7eef04f691d395fed3abcd925b133c4a808e999b700648`）。指南明确 Sol/Luna 的 EU residency 仅对 Standard processing 的 Responses/Chat Completions 可用；regional storage 不等于 regional processing，system data 与 Remote MCP 第三方数据不自动覆盖，非美国地区有 retention/abuse-monitoring 条件，适用 regional processing 有 10% uplift。
- 新增可考面试边界：`Standard processing` 与 `reasoning.mode=standard` 是两个控制面；将 `model × endpoint × region × processing_mode` 作为独立 capability gate，Batch/Flex/Fast 的价格不能替代资格证明。已同步 Sol 笔记、第二十四册 32.55.3、模型盘点/榜单解释/来源索引、题库、练习、知识图谱、论文索引、项目和书系索引。Sol 仍为 **AA 单榜资料级闭环 + 当前时点复验 + local protocol toy**；没有新增模型专属架构/训练 recipe、DataCurve 行或生产验收证据，goal 保持 `active`。

## 2026-09-24 GPT-6 Sol：7890 复验与动态 effort telemetry 补证

- 用户贴出的百度响应证明 `10.24.27.134:7890` 在其环境可访问。本轮受限环境内对榜单地址出现 TLS EOF；经批准在限制外使用同一代理，Artificial Analysis `/zh` HTTP 200，`1,783,966` bytes / SHA-256 `0baa28c8ab69a104a078645983ca843c95c5db879b6c52789104aa3d1758efb8`；规范 DataCurve `https://deepswe.datacurve.ai/` HTTP 200，`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`。八家重点厂商无新 canonical 模型。错误 host `www.datacurve.ai/deepswe` 返回 404，随后已改用规范主机。
- OpenAI Developers 的直接 `.md` 路径对 GPT-6 Sol model/Reasoning/Compaction 页分别出现 TLS EOF（7890）或 403（1234、8098）；改从官方 `platform.openai.com/docs/...` clean HTML 路径获取，跳转至 Developers 后 HTTP 200。Model 页面 `432,023` bytes / `e8a4f18b6e60883740510c3c521801867e90961113e6a4b2f41959a9d606c836`；Reasoning `1,070,709` / `3bcb1b7771731d3269783464180b72c2b45283688118dda185ebd2d65e84616c`；Compaction `455,842` / `d3825ca5955cba651e7fbf368a8fe6905056fc726cfc6d47a708d4f1b37dfa8a`。Agents/Tools Markdown 快照哈希与原记录一致。
- 重点补强 `configuration_update`：只控制 GPT-6 family standard single-agent mode 的 reasoning effort，可用于 Responses 与 WebSocket `response.create`；effective effort 持续到下一次覆盖；响应里的 `reasoning.effort` 仍代表 request-level setting。固定原 request config、把更新作为历史 item 插在新用户消息前，有利保留 prompt prefix/缓存复用，但不是 cache-hit 保证。与自动 compaction/truncation 或独立 `/responses/compact` 的冲突和显式 compact 后 fresh update 的要求已记入。
- 已扩展 `gpt6_sol_contract_audit.py`，toy 验证 low request + high update 的有效档位、覆盖回 low、response 字段仍报告 low，以及 stable prefix digest 不变；脚本主流程输出 `ok=true`、`cache_hit_claimed=false`、`network_called=false`，以执行成功同时确认语法有效。`git diff --check` 通过。已同步 GPT-6 Sol 研究笔记、第二十册/第十六册、`INTERVIEW_BANK.md`、`EXERCISES.md`、`KNOWLEDGE_GRAPH.md`、`plan.md`、模型清单与来源索引；goal 仍 `active`。

## 2026-09-24 Gemini 3.8 Flash：官方 signature 文档冲突复验

- 用户提供的 `curl` 已证明 `10.24.27.134:7890` 可连接；当前环境使用同一代理请求 `www.baidu.com` 得到 HTTP 200（2,381 bytes）。本轮沿用已复核的 Artificial Analysis `/zh`（`1,783,966` bytes / `0baa28c8ab69a104a078645983ca843c95c5db879b6c52789104aa3d1758efb8`）与 DataCurve DeepSWE（`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`）快照；两榜八家重点厂商无新 canonical 锚点。
- 7890 获取 Google 官方 Markdown：Thinking `286,147` bytes / `fb492e15a1ffeed4d2244e90db56352194f87d2adea5d775d2f2ac8fd993ed16`；Tool combination `136,858` / `f2b104688cf487094ec37fc944843261b417d0c68eee97203e34e0fa30a83de0`；Function calling `983,336` / `f0e00d0dc8f9876ba15d9a1f4fb6997332a199de794a5d8a0a933e2746934ee0`；Thought signatures 旧入口 `88,620` / `799b3ed22f5afcb653f5ab589903ecf04e170750961653720d078924dbaace45`，现已迁移/重定向至 Thinking。Interactions Markdown 本轮超时；未调用真实 API。
- 关键区分：Thinking 说 GenerateContent 没有 dedicated thought step，signature 可属于任意 part（含 `functionCall`）；Interactions 的 thought 是一等 step，signature 按 Thinking 只属于 thought/built-in tools、不属于标准 function call。Tool combination 却在 Interactions 上下文中把 Gemini 3+ `function_call`/`function_response` 也列为可能带 signature。文档冲突经当前时点复验仍存在；Function calling 页“SDK 自动处理”没有给出对冲突字段位置的独立裁决。
- 已将 API-scope 区分与当前冲突边界同步到 Gemini 3.8 研究笔记、第二十册第 23 章、source index、model inventory、plan 和本进度。工程要求继续原样保留实际返回的 opaque fields、不得自行生成/剥除/改写，并用 `id` 配对 function call/result；真实字段行为须在有授权、API key 和 endpoint 时另做 capability probe。当前仍为 **内容专题闭环 + 官方文档冲突待实测**，不把 local protocol toy 升级成服务端证据。
- 下一步先跑 Markdown/链接/Python 与 `git diff --check` 门禁，然后按两榜现有八家厂商 canonical 集合选下一项高价值未收口锚点；不扩大厂商范围，goal 保持 `active`。
## 2026-09-24 Kimi K3：vLLM v0.30.0 stable/runtime 更新

- 沿两榜已发现的 Kimi K3 锚点复核 vLLM `v0.30.0` release commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`、PyPI metadata、registry、NVIDIA model 与 DSpark source。`v0.29.0` 已有 K3 stable source entry，因此只记录 runtime 实现演进，不称首次支持。
- 新增面试知识：IPC transformed-weight zero-copy reuse 与资源 ownership；streamed weights 完成后再 post-load finalize；PP auxiliary hidden state/AttnRes boundary；DSpark context-KV dtype/scale/layout gate；KDA `mamba_ssm_cache_dtype`。
- 已同步研究笔记、第 88 章、source index、model inventory、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`KNOWLEDGE_GRAPH.md` 和计划。wheel 未下载/安装；完整权重、目标硬件数值/profile、cache recovery、DSpark acceptance 与生产 SLO 未验证；goal 继续 `active`。
## 2026-09-24 Claude Fable 5.1：thinking-state 专题同步

本轮使用 `10.24.27.134:7890` 获取 AA 首页、Fable 详情及 Anthropic 官方资料；7890 对 AA/Anthropic 返回 HTTP 200。DataCurve 的 7890 请求 TLS 提前断开，改由用户提供的 `10.24.27.134:8098` 重试成功，返回 HTTP 200 且与同日快照字节一致。未把单一线路失败记成页面不存在。两榜复验未发现八家重点厂商的新 canonical 模型，DataCurve 无精确 Claude Fable 5.1 Agent 行。

已新增第二十册第 21 章 21.29，解释 thinking state 的 model binding 与 prefix binding、Opus 5.5 → Fable 5.1 仅 Claude API 的可读边、反向不可读、block drop 的计费/telemetry、账号创建日期影响的 prefix enforcement，以及 append-only 更新与安全裁剪。同步更新研究笔记、source index、model inventory、INTERVIEW_BANK、EXERCISES、KNOWLEDGE_GRAPH、计划和进度；协议 toy 增加方向性与 endpoint-scope 合成断言。

当前状态：**AA 单榜内容专题闭环 + 官方 System Card/API contract + local protocol toy**。没有真实 API key/probe；参数、内部架构、训练配方、独立 benchmark、目标硬件与生产验收仍待核验。goal 保持 active。

## 2026-09-24 DeepSeek V4 Flash：SGLang serving 补证

用户提供的百度 `curl` 响应确认其环境内 `10.24.27.134:7890` 可用；工作区此前也经该代理取得两份排行榜 HTTP 200 快照，Artificial Analysis `/zh` 为 `1,782,611` bytes / SHA-256 `566b4adab724bd312436a302be3f0f4c5d9f7713188e9efb078cb636faddd02b`，DataCurve 为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`；与已有同日快照一致，无新重点厂商 canonical 模型。

选取既有 DeepSeek V4 Flash 锚点补 SGLang serving 证据：PR #34565 的 SWA branch-point cache 处理共享前缀分叉后的混合状态复用；#30805 是 B200 FP8/TP=1 CSA/HCA unit-kernel 对照；#29927 是 RTX PRO 6000/SM120 paged-MQA indexer、sparse prefill、FP4 MoE 路径。PR 的 `43.81% → 60.75%` token hit rate、`1.2x/1.45x` kernel 和最高 `3.4x` TPOT 均保留各自 workload/hardware/baseline 条件，不写为同类端到端或独立 benchmark。#39171 仅记录 V4.1 FlashMLA dependency pin 边界。

已把新知识同步进第二十一册第 77 章，并更新 DeepSeek V4/V4.1 研究笔记、source index、model inventory、inventory interpretation、PAPERS、INTERVIEW_BANK、EXERCISES、KNOWLEDGE_GRAPH、plan 与本进度。未新增重复架构章节。完整权重加载、目标硬件独立复现、混合 cache/state 线上恢复、端到端 Agent acceptance 与生产 SLO 仍待核验；goal 保持 `active`。

## 2026-09-24 GLM-5.3-Flash：vLLM v0.30.0 stable/fixed-main 文件对照

本轮沿两榜既有 GLM-5.3-Flash 锚点推进，vLLM 仅作 serving 来源。vLLM `v0.30.0` release 于 `2026-09-22T05:20:54Z` 发布，tag commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`；stable GLM5Next directory tree SHA 为 `118165530271cb3aa749d77bf927499957db2bd4`。对照固定 main commit `81d7293c2167e39f3ffddc9a82d633f94e8a1eaa`，不是 mutable main。

网络记录：用户终端通过 `10.24.27.134:7890` 请求百度成功；本执行环境沙箱内连接该代理失败，获准在沙箱外后通过同一代理只读取得 GitHub fixed-tree API 和不可变 raw source。故本轮失败属于沙箱网络隔离，不是代理或 GitHub 源码缺失。

- stable `nvidia/attention.py` 含 `Glm5NextIndexerCache` 与 `Glm5NextTailCache`；stable `nvidia/kda.py` 和 `nvidia/multimodal.py` 分别与 main `common/` 文件同 blob。MTP diff 仅 fused norm import 路径，主体保持一致。
- attention 两版不同 blob：main 将 sparse indexer 移入 GLM5Next 专属实现、将 indexer workspace 改按 pool 数 sizing，并调整 RoPE scaling 映射。stable KPool tail seed 依据 dense layout 算偏移；main 读取 `tail.stride(0/1)` 并校验 tensor layout。需针对实际 paged/alias cache 做测试；没有运行测试前不把它定性为 stable bug。
- main `model.py` 另增加 gate/up packed mapping 与 Quark `.weight_scale` 接受逻辑，这些不属于 v0.30.0 stable。stable/main 文件大小、Git blob 与来源链接已记在 `glm-5.3-flash-source-notes.md` 和 `source-index.md`。

状态为 **v0.30.0 stable source feature paths confirmed + fixed-main deltas documented**，不是运行时/生产验收。wheel 未安装，完整权重未加载，cache/state recovery、MTP acceptance、EPD、目标硬件数值/性能和 tool/verifier SLO 未测。资料同步至 model inventory、interpretation、第二十一册第 84 章、PAPERS、INTERVIEW_BANK、EXERCISES、KNOWLEDGE_GRAPH、plan 与 progress；goal 保持 `active`。

## 2026-09-24 Gemini 3.8 Flash：Interactions API reference 与固定 SDK schema 补证

用户提供的 `10.24.27.134:7890` 请求已证明代理可用；本轮沿双榜既有 Gemini 3.8 Flash 锚点复核官方 Interactions API reference（HTTP 200，`787,186` bytes / SHA-256 `a06a779e779608c470b178dce0361fb3548ae5121b0c7a17766f8dcce91ceb2e`）及固定 Google Gen AI Python SDK revision `4742c9a5c213a587add126a500a271824e2f0add`。

SDK typed schema：`ThoughtStep.signature` optional；`FunctionCallStep` 要求 `id` 但未声明 signature；`FunctionResultStep` 要求 `call_id` 但未声明 signature；Google Search call/result 的 signature 为 optional。`BaseModel extra="allow"`、lenient open union 与 `UnknownStep.raw` 表明未知字段/step 可被兼容性路径保留，typed field 缺席不能当作线上禁止字段的证据。Interactions 使用 `function_call.id` 对 `function_result.call_id`，区别于 GenerateContent 的 `functionCall`/`functionResponse`。Thinking 与 Tool combination 对 custom-function signature 的官方描述差异仍在；没有 API key/授权 endpoint，未进行真实 probe。

已同步研究笔记、第 23 章、source index、model inventory、PAPERS、题库、练习、项目和知识图谱。协议 toy 增加 SDK-schema-compatible 可缺省签名与较严格文档 profile 对照，运行通过：`ok=true`、15 个有序 SSE 事件、`network_called=false`。状态仍是**内容专题闭环 + 官方 schema 补证 + local protocol toy**；真实 endpoint、3.8 内部架构/训练、目标硬件和生产验收不在已证范围内。下一步执行受影响文件检查，再从两榜已有重点厂商 canonical 集合选择下一锚点；goal 保持 `active`。

## 2026-09-24 Claude Opus 5：System Card 安全评测方法专题

本轮沿两榜已有的 Claude Opus 5 继续，没有发现新的八家重点厂商 canonical 锚点。经 `10.24.27.134:7890` 取得 Artificial Analysis 首页（`1,781,428` bytes / `15e9763f0d516791e7b7c2c5679e87e5878eaffed4bef31e485c9043b758ba59`）、Opus 5 详情（`3,977,410` bytes / `18acc956d77c5b49c391eb85b46ef2c7fdbd7edd282ddf2275940bd418d25774`）及规范 DataCurve DeepSWE（`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`），均 HTTP 200。AA Index 为 `50.7771115797629`；DataCurve 精确 max 行为 `327/444`，仍按 harness/环境/verifier 结果记录。

重新读取 Anthropic 官方 System Card PDF，当前 `16,281,258` bytes / `0950dae1ba6b341e4f1a009e535e0f025625efeca21bd043a9a5ae148a3f2e6b`，与既有副本相同；仓库 parser 产生 4,641 行、334,056 bytes 文本 / `4ae20472ab82c967d90f386239ee6987ddf74b1124db6b44a1dea69a576e1f4c`。读到卡片 2026-08-19 changelog：新增 prompt-injection bounty 数据，并因 Cowork harness 不一致重跑基线、删除不支持的 thinking-disabled 条件。

专题归纳：区分 ART 饱和后采用的 IPI 与 adaptive red-team；attempt-level / scenario-level ASR 分开报告；输入侧 tool-result probe 与动作侧 classifier 分开消融；Cowork Auto mode 成绩是产品系统结果，不等于裸模型安全率。RSP 的 CB-1/CB-2 与 ASL-3 属 Anthropic 发布方判断。详见研究笔记 §8。

已同步研究笔记、`source-index.md`、模型盘点与解释、第八册第 11 章 6.5、第二十册第 19 章 19.38、PAPERS、题库、练习、PROJECTS 和 KNOWLEDGE_GRAPH。当前状态为 **System Card Agentic Safety/评测方法专题闭环**；参数、架构、完整 recipe、独立复现、外部真实 endpoint 和生产 SLO 仍 `unverified`。下一步执行门禁，再从两榜已有八家重点厂商 canonical 集合选择后续锚点；goal 保持 `active`。

## 2026-09-24：DeepSeek V4 Flash Vision 路由与图像预算复验

- 使用 `10.24.27.134:7890` 重新抓取指定榜单；Artificial Analysis 中文首页 HTTP 200，`1,781,428` bytes / SHA-256 `15e9763f0d516791e7b7c2c5679e87e5878eaffed4bef31e485c9043b758ba59`；DataCurve DeepSWE HTTP 200，`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`。两份快照均与已有同日记录一致，八家重点厂商无新 canonical 模型。
- AA `deepseek-v4-flash-vision` 详情 HTTP 200，`3,967,107` bytes / `a5e5259ee84eac5aa88915dd6436ba155e265ede940ab663b52c4a681ae43ed7`；Index `34.8390628035969`、median output speed `217.762710468086 tokens/s`、median TTFC `0.970116780000126s`、cost/task `$0.314372044499279`。9/15 的 Index `35.0122378035969`、速度 `215.179167697513`、TTFT `1.29855545700002s` 作为历史 provider 测量保留，不解释为 revision。DataCurve 没有精确 `mini_swe_agent_deepseek_v4_flash_vision_*` 行。
- DeepSeek V4.1-Flash 9/10 发布页首次连接失败，重试后 HTTP 200（`24,438` bytes / `f18dc22d37393381b31c9069996138f45aa7b02b08442d43af7c6c57f587bdce`）。Quick Start（`48,088` / `7ce9db1b1cc7e2efafe7cbfd57b9d46d240c20399f7bd87672c7e3a5250ccdd0`）、Pricing（`23,982` / `210f102275ccf1a6542f08a3bc9e4b4c7c83278cb74b35217bffa112df6363b2`）、Vision（`80,467` / `5654a198302edd80aea443fb123d14b14d675ccc76730959dd55414d9baee1e2`）、Files（`62,188` / `1a825256f4dfae25b40751044bc069860897e30582d5b492e8edeea96b37297a`）、Responses（`57,105` / `3af115c64774731d42e29b8b6982b914351d501882aaa9440dfa55c83f549ce0`）和 Vision 实验公告（`22,032` / `56babe7f597f37b31c4440174272d4090f32056902c76e3984be3d54d5864de8`）均成功。
- 官方现行合同：旧 `deepseek-v4-flash-vision-exp` 名称仍接受，但对应模型已退役，请求暂由 V4.1-Flash 服务并按 `deepseek-flash` 峰谷价计费。Vision guide 当前为 resize 后每图最多 1,024 tokens；最多 600 张/请求，区分 URL/base64 与 `file_id` 单图大小、请求体/合计体积及 8192/4096 px 限制。AA 页面 provider 价与现行官方 Flash 价分开记录。
- 已同步 `deepseek-v4-flash-vision-source-notes.md`、`source-index.md`、`model-inventory.md` 和第二十一册第 85 章。当前为 **AA 单榜资料级闭环 + 当前路由与图片预算复验**；未进行真实 API probe，未获得独立 Vision 架构/训练报告、DataCurve 精确行或线上验收证据。后续仍只从两榜八家重点厂商的已有 canonical 集合选择；goal 保持 `active`。

## 2026-09-24 Qwen3.5-Omni Plus / Flash：ARIA、AuT 与时间戳

- 用户提供的终端输出确认 `10.24.27.134:7890` 在用户 shell 可经代理访问百度。当前 agent 网络路径对同一代理重试两次（沙箱内、沙箱外）均连接超时；备用 `10.237.126.170:1234` 可用，百度返回 HTTP 200，并成功刷新本轮两条 Artificial Analysis 页面、DataCurve、arXiv v2 abstract 和 Model Studio 文档。
- Artificial Analysis Plus 页面 HTTP 200，`3,826,061` bytes / SHA-256 `b2127d33c029c55eac112636964f2a9cb9196b69ca5f792c90a2a9ae79f387d9`；Flash 页面 HTTP 200，`3,834,828` bytes / `17bd65b3597f29af8a655e00d038b774dc2ae89c821848262b7d0e4bfc8d560c`。Plus 当时第三方/provider 字段：256K、Index `20.3839890713777`、输出速度 `84.6202668152731 tokens/s`、输入/输出 `$0.40/$4.80` 每百万 token。
- DataCurve HTTP 200，`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`；未检出精确 `mini_swe_agent_qwen3_5_omni_*` 行，不迁移其他 Qwen Agent 结果。Plus/Flash 按同一家族配置记录。
- Qwen Team 技术报告 [Qwen3.5-Omni Technical Report v2](https://arxiv.org/abs/2604.15804v2) 摘要 HTTP 200，`42,723` bytes / `7c93a28bd4e5bd795492e20ee6fcbb991920d5834bc6f9f1d4f83dfc0472a34a`；源码包 `2,984,159` bytes / `fd53a97d5be7eaa8c981c853f1e6a4faa9b86c69c5fe6f3dbcfa33b66abd81ff`。已核读 architecture、pretraining、posttraining、experiments 段落。
- 技术要点：Hybrid MoE Thinker-Talker；AuT 16× Conv2D 下采样、6.25 Hz / 约 160 ms；TM-RoPE + 秒级显式视频/音视频 timestamp 和随机音频 timestamp；ARIA 单交错流及 prefix-level speech:text ratio 上限；RVQ/MTP residual codebooks 与因果流式 codec。训练为 S1 对齐、S2 约 4T 多模态 tokens、S3 32,768→262,144；Thinker specialist/on-policy distillation + interaction-aligned RL，Talker 长上下文 CPT/DPO/GSPO/speaker fine-tuning。报告的 100M+ 总体音视频、AuT 40M 与 Talker 20M+ 小时是不同口径，不相加。
- Model Studio 文档 HTTP 200，`402,771` bytes / SHA-256 `9140a68a7b0a98def6a8cca23e759359895046bd3fc6123b9c361c7526bd7578`；托管调用示例为 `qwen3.5-omni-plus` 且须 `stream=True`，custom voice 只列 Plus/Flash、snapshot 不支持。未做真实 endpoint probe。
- 已新增 [`qwen3.5-omni-source-notes.md`](research/model-update-2026-09/qwen3.5-omni-source-notes.md) 和第二十一册第 93 章，同步来源索引、模型盘点/解释、PAPERS、INTERVIEW_BANK、EXERCISES、PROJECTS、KNOWLEDGE_GRAPH、BOOK_SERIES 和计划。本专题为 **AA 单榜内容闭环 + 官方报告/API 文档核验**，不是完整参数/训练配方、权重、独立 benchmark、GPU profile 或生产 acceptance 验收；goal 继续 `active`。

## 2026-09-24 Qwen3.6-35B-A3B：Thinking Preservation 与混合注意力

用户 shell 提供的 `curl` 结果确认 `10.24.27.134:7890` 能访问百度。当前 agent 环境复测同一 7890 代理对百度连接超时；限制外重试仍超时。因此只说明两个网络路径结果不同，不说该代理全局不可用。备用 `10.237.126.170:1234` 在 agent 环境返回百度 HTTP 200（2,381 bytes），并成功刷新两个指定榜单：

| 来源 | 当前快照 | 结果 |
|---|---|---|
| Artificial Analysis `/zh` | HTTP 200，1,781,428 bytes，SHA-256 `15e9763f0d516791e7b7c2c5679e87e5878eaffed4bef31e485c9043b758ba59` | 与同日已有快照一致；八家重点厂商 canonical 集合本轮的未覆盖项选为 Qwen3.6-35B-A3B |
| AA Qwen3.6-35B-A3B Reasoning | HTTP 200，4,056,617 bytes，SHA-256 `120de76d2630e0fe777fa10bb82ec7b484f4c33ef2a7487085c489f4158e6391` | canonical `qwen3-6-35b-a3b`，榜单日期 2026-04-16；Non-reasoning slug 为同一模型配置 |
| DataCurve DeepSWE | HTTP 200，268,036 bytes，SHA-256 `14436c31be1e50a0b62171e4ee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1` | 当前快照无精确 Qwen3.6 Agent 行，不迁移其他 Qwen 的成绩 |

固定官方资料：QwenLM/Qwen3.8 README commit `2ea10dc725823bf7c3e21ce8557cbe15245132ae`，14,334 bytes / `a71ec46607f81d6056336fb0a8431a26a1c7d8db6ac568a0c021c36f1ed3c92e`；Qwen ModelScope 模型卡 README revision `913c459c5c83fa016a0e54a52e5b95f6c894e0fe`，64,550 bytes / `c4ddaa065649ff6352648f64747a16eda31726f3e34add94ce04abb461c77b75`；config/chat template revision `1a5ae24e867f8d82388070d3f61590158a01d15c`，分别 3,686 bytes / `93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99` 与 7,764 bytes / `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`。通过 1234 直取固定 GitHub raw README 本轮 TLS timeout，继续使用此前取得并固定哈希的 README；官方博客入口本次只返回客户端渲染 shell。arXiv 精确检索快照未见 Qwen3.6 专属报告，只作为当前检索边界。

官方卡确认 35B total/3B active、40 层的 `3×Gated DeltaNet + 1×Gated Attention` 周期布局、256 experts（8 routed + 1 shared）、MTP multi-step training、262,144 native context / YaRN 可扩展约 1,010,000，并提醒 static YaRN 可能影响短文本。chat template 在 `preserve_thinking` 关闭时只保留最新 user 之后的 assistant reasoning，开启时保留历史 reasoning；`enable_thinking` 独立控制生成前缀。面试落点是“历史推理序列化 ≠ 跨会话持久记忆”；Model Studio 与 vLLM/SGLang 参数路径亦不同。Qwen 对减少重复推理/KV-cache 效果属于发布方声明，没有独立验证。


## 2026-09-24 Qwen3.6-27B：官方 dense 架构与 GDN Tree-Scan serving 预印本

- 用户终端给出的响应确认 10.24.27.134:7890 在用户 shell 可访问百度。当前 agent 对同一代理的百度/Qwen 博客请求在沙箱内失败，沙箱外重试百度也连接超时；该结论只描述当前 agent 网络路径，不代表该代理全局不可用。备用 10.237.126.170:1234 成功取得 arXiv 搜索/API、预印本 HTML 与 GitHub pinned commit。
- Artificial Analysis Qwen3.6-27B 页面：4,051,359 bytes / SHA-256 97375509bc5e428c1886206338a512e3ffb5bccd72b46bc92ed8cd5df8f17e44；Reasoning 与 Non-reasoning 两配置按同一基础模型归并。DataCurve DeepSWE：268,036 bytes / 14436c31be1e50a0b62171e4ee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1，无精确 Qwen3.6-27B Agent 行。
- 官方 ModelScope README revision cea40373b9214dd387123e68841890af30dcd469：62,593 bytes / bb936d6da51014f1edc9aa4cf9abf28d98695b7616ad56adfeeebfa752051d3d；config/template revision c53c4820996523bb6413f1002e24c5dfb0bad548，哈希分别 69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c 与 e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259。卡片确认 27B dense、64 层、每四层 3×Gated DeltaNet + 1×Gated Attention；与 35B-A3B 的 chat-template 内容哈希相同，preserve_thinking 是家族共享序列化接口。
- arXiv 当前 ti:Qwen3.6-27B 和 ti:Qwen3.6 title-field 查询分别为 0 条；all-field 查询有 50 条，不据此断言不存在相关论文。成功读取 Zhiyuan Ma 单作者预印本 [GDN Tree-Scan: Served Tree Verification for Recurrent-Hybrid Language Models](https://arxiv.org/abs/2609.23900v1)，日期 2026-09-20。HTML 为 275,937 bytes / 8b935c50f17284cb6ad85cf03454ce321ec50f6c80da6269f7fd48639034c0f7；API XML 为 2,757 bytes / 7a149feb604a9450bd06f03b59bdc7762b50fd2b6ab2cd6b9f0458ae00e89986。
- 技术要点：在 Qwen3.6-27B-FP8 上，用 FA2 tree-bias 处理 attention ancestry、branch-local GDN scan/replay 维护各树分支的 parent recurrent state、MTP spine/root sibling 形成候选树、device-side multidraft committer 做提交，并只发布 accepted-chain state。作者在 B=1、temperature 0.6、四个 SWE/Codex tasks 报告 committed tokens/event 4.11→4.82（+17.2%）；verify-forward 0.137/0.138 秒；token-weighted decode TPS 18.80→23.88（+27%），per-request-equal TPS 17.80→18.51（+4%）。这是 decode-only 指标，不是任务总墙钟加速；重复 11K–14K prefill 且未开 prefix cache，论文未给出通用 task-wall 提速。
- 正确性依据是 40-turn recurrent-oracle p-rescore 与 native numerical flip floor 比较，不能称为 full distribution-distance proof；request-cluster bootstrap、更多 seeds、B=4 与 Stage-D timing 仍 open。固定 artifact 指向 Lumo_FlyWheel commit 55f55854328b37f262e97d57b5863d8fadd7ff76；本项目未运行代码、下载权重或复现实验。该预印本是第三方 serving 研究，不是 Qwen 官方技术披露。
- 已同步 qwen3.6-27b-source-notes.md、第二十一册第 83 章 83.16.4、第二十四册第 61 章 61.31、source-index.md、model-inventory.md、PAPERS、INTERVIEW_BANK、EXERCISES、KNOWLEDGE_GRAPH、计划与进度。仍未读取到 Qwen 官方博客正文或 Qwen 自有专属技术报告；完整训练 recipe、权重、真实 API、独立复现、production SLO 与硬件 profile 未验证。
- 当前状态：**AA 单榜锚点核验 + 官方模型卡/config/template + 外部 serving 预印本专题**。整体 goal 保持 active。下一步先完成全局链接/围栏与 diff 门禁，再从两榜八家重点厂商的 canonical 集合选择未覆盖的高价值技术缺口。

## 2026-09-28 Qwen3.6-27B 官方博客正文恢复与排行榜复验

- 通过 `10.24.27.134:7890` 获取百度 HTTP 200（2,381 bytes），并取得 Qwen 官方文章 API HTTP 200（94,527 bytes；含动态 `request_id` 的响应哈希 `036a9cfc6a38b04fed0b72aaf9339296296356dd9441a9e63d4b720283ed90c0`）。嵌入文章 HTML 为 91,758 bytes / SHA-256 `7748e75a7c5a47943d6abe4e6415cb6b8d4749713eeff39323eb831f2d8ae367`，标题《Qwen3.6-27B：270亿参数稠密模型，旗舰级编程能力》，Qwen Team，日期 2026-04-22。直抓 `qwenlm.github.io` canonical 页面仍连接失败，不将此特定路径失败外推为代理不可用。
- 9 月 28 日榜单快照：Artificial Analysis `/zh` HTTP 200，1,674,187 bytes / SHA-256 `891ad0a15cd12b7ae704512351bdee86c0136b75e11704b8bd5f6705c0dab42f`；AA release 页面 HTTP 200，941,371 bytes / `20d83181c16942ddc1b7a14268635f80848b87b4d77e11f9b6e9febc74f41afa`；DataCurve DeepSWE HTTP 200，268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`。八家重点厂商没有新增 canonical 锚点；DataCurve 的 Muse Spark 不在当前关注厂商范围，DeepSeek V3.2 Exp 为已弃用历史 revision，不作为新模型。
- 博客报告 Qwen3.6-27B 的 SWE-bench Verified 77.2、SWE-bench Pro 53.5、Terminal-Bench 2.0 59.3、SkillsBench 48.2，均按发布方结果保留并绑定各 benchmark/harness 脚注；补充 `preserve_thinking`、OpenAI-compatible Chat Completions/Responses、Anthropic-compatible API 和 OpenClaw/Qwen Code/Claude Code 示例。OpenClaw 的 `contextWindow=131072`、`maxTokens=16384` 是客户端预算，区别于模型卡 262,144 native context；百炼 API 可用性表述不一致，本轮未做真实 endpoint probe。
- 已更新 `qwen3.6-27b-source-notes.md`、第二十一册第 83 章 83.16.5、第二十四册第 61 章 61.32、来源索引、模型盘点、PAPERS、INTERVIEW_BANK、EXERCISES、KNOWLEDGE_GRAPH、BOOK_SERIES、本计划与进度。官方博客未披露新的内部架构或完整训练 recipe；外部 GDN Tree-Scan 仍严格标作第三方预印本。
- QA：本轮相关 Markdown 围栏平衡；研究笔记、两册章节、索引/盘点及 plan/progress 等关键交叉引用目标存在；`git diff --check` 通过。下一步仅依据 Artificial Analysis/DataCurve 两榜八家重点厂商 canonical 集合，选择下一项高价值未闭环技术缺口。goal 继续 `active`。

## 2026-09-28 Qwen3.6-35B-A3B 官方博客：Agent evaluator 与 harness 复核

- 锚点仍是 Artificial Analysis 已发现的 `qwen3-6-35b-a3b` / Non-reasoning 两配置（归并为一个模型）；DataCurve 当前没有精确 Qwen3.6 Agent 行。本轮只沿此既有锚点补官方来源，不因博客中的 `qwen3.6-flash` 服务名另发现模型。
- Qwen 官方文章 API `https://qwen.ai/api/v2/article/?language=zh-CN&path=qwen3.6-35b-a3b&type=qwen_ai` 通过 `10.24.27.134:7890` 的沙箱外获批请求返回 HTTP 200、94,678 bytes，响应 SHA-256 `d287402f27a6ffa3226466aa57407310f670eb73a6d72bcfef03597a94e35d49`（含动态 request_id）；嵌入 HTML 91,941 bytes / SHA-256 `706889d145ea17b8c8234c4cda35b00fdecc0b6bcb9e1f5f20d2ed3ff9e15ed1`。文章标题《Qwen3.6-35B-A3B：智能体编程利器，现已开源》，Qwen Team；博客元数据为 2026-04-15 10:00 +08，而 AA/Qwen 仓库记录 2026-04-16，保留来源差异。
- 博客比较 Qwen3.5-35B-A3B→Qwen3.6-35B-A3B：SWE-bench Verified 70.0→73.4、SWE-bench Pro 44.6→49.5、Terminal-Bench 2.0 40.5→51.5、SkillsBench Avg5 4.4→28.7、QwenClawBench 47.7→52.6、NL2Repo 20.5→29.4。重点新增方法是评测系统依赖账本：SWE-Pro 修订部分任务并重跑基线；Terminal-Bench 绑定 Harbor/Terminus-2、3 小时、32 CPU/48 GB、256K、80K 输出上限与 5 runs；SkillsBench 是 OpenCode 的 78 个 self-contained/no-API 子集与 5 runs；NL2Repo 的对比系统使用 Claude Code、最多 900 turns。
- evaluator/user-simulator/toolchain 的版本也属于系统定义：TAU3-Bench 使用 GPT-5.2 low-reasoning user model + default BM25；VITA-Bench 因 Claude 3.7 Sonnet judge 不可用而改用 Claude 4 Sonnet；MCPMark 固定 GitHub MCP v0.30.3、Playwright 输出截断 32K；MCP-Atlas 是公开集 + Gemini 2.5 Pro judge。上述跨厂商模型只作为 Qwen 评测依赖，不扩张本项目模型发现范围。QwenClawBench 是发布方称的内部 real-user-distribution benchmark；QwenWebBench 的渲染 + multimodal judge + Bradley–Terry/Elo 不等于独立 pass rate。
- 博客将该开源 checkpoint 的百炼服务名写作 `qwen3.6-flash`，并示例 OpenAI/Anthropic-compatible API、OpenClaw/Qwen Code/Claude Code；OpenClaw 使用 131,072 `contextWindow` 与 16,384 `maxTokens`，属于客户端预算而非 native context/API 实测限制。百炼 endpoint 未 probe；博客并非新架构/训练报告。
- 已同步 Qwen3.6-35B-A3B source notes、第二十一册第 83 章 83.15.3–83.15.4、第二十册第 21 章 21.30、第二十四册第 61 章 61.32、source-index、model-inventory、PAPERS、INTERVIEW_BANK、EXERCISES、KNOWLEDGE_GRAPH、BOOK_SERIES、本计划和进度。QA 结果：相关 Markdown 围栏平衡；笔记/三册章节/索引/盘点/计划的关键交叉引用文件存在；`git diff --check` 通过。下一步只从两榜已有八家厂商 canonical 集合选下一高价值知识缺口。goal 继续 `active`。

## 2026-09-28 Qwen3.8 Max (0902)：7890 当前时点复验与缓存协议精化

- 当前 agent 工作区显式使用 `--proxy http://10.24.27.134:7890` 取得 Artificial Analysis 首页/release/详情、DataCurve、QwenCloud 产品/开发者文档及 Responses API reference；均 HTTP 200。用户 shell 的百度结果与当前工作区目标站点复现分别记账。
- 两榜快照：AA 首页 1,674,048 bytes / `56359658c7d187e4a214ae65857f0a7668d42ebcc1da0e2d4fab477642e2640c`；AA releases 938,807 / `913306564eb762ad0587c8845bece903f5004a0198f64709af7318fbcb960ff2`；Qwen Max detail 3,881,440 / `f9c920db61e73a4d71fcd9a10d057d7bf22ce01d979974dd32ec4ed1f911cccb`；DataCurve 268,036 / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。AA 同日首页相差 139 bytes，按动态页面变化处理；未发现重点厂商新 canonical。DataCurve 无 Qwen3.8 Max 0902 精确行。
- QwenCloud 产品页 98,992 bytes / `ce97b7c9098ed54c195aa2c6b1a5c0dfc6ad43e7290aea10bcce8484f42b9a6b`，last-modified `2026-09-28 09:55:09`；Context Cache 1,134,820 / `47a01521f4ce689acf6f017c36c66af7c7c98984029eb5ba5c5f58e170b69a74`；Dynamic Rate Limits 412,225 / `033d1381bfd1c759834bb3402b75e53acca8a7c617480f3424b53b0080ec6460`；Responses API reference 952,619 / `fd467c8d7ce159e2c98f7220ee599014ee28a1f3b9a6dd6309075313926a1940`。
- 面试增量：分开记录 marker 前置 `≤20 content blocks` 与 follow-up `other messages≤20`；`1,024` tokens 是缓存资格而非命中保证，同页 explicit-cache 请求示例另写 `>1,024`，边界需 endpoint probe；Chat Completions usage 为 `prompt_tokens_details.cached_tokens`，Responses 为 `input_tokens_details.cached_tokens`；Responses continuation 延续 input/output 但不继承 `instructions`，`previous_response_id` 不能与 `conversation` 同用，`store=false` 不可续接，response ID 文档有效 7 天。
- 已更新研究笔记、来源索引、盘点、第二十一/二十四册、面试题、练习和 local protocol toy。没有真实 API probe、权重下载或新增架构章节。最终 QA 结果随后补录；goal 保持 active。

## 2026-09-28 Qwen3.7 Max：environment scaling、跨 harness RL 与 reward-hacking monitor

本锚点来自 Artificial Analysis 已有 `qwen3-7-max`；DataCurve 没有精确 Agent 配置。本轮复用 2026-09-28 AA/DataCurve 快照，只沿榜单发现的 Qwen3.7 Max 查官方材料，没有从博文、Qwen API 或其他厂商比较项新增模型。

Qwen 官方博客正文由文章 API 的 `path=qwen3.7` 取得。新增面试主线是：训练 rollout 正交拆分为 `Task × Harness × Verifier`，组合不同框架/验证器对同源任务做 RL；环境质量/多样性扩展与 OOD transfer 是 Qwen 发布方提出的 Agent scaling 方法。文章未给足任务数、环境构成、曲线和统计区间，故标为 publisher-reported methodology，不称为已确立的通用 scaling law。

第二个方法案例是未知平头哥真武 M890 PPU 上的 SGLang Extend Attention kernel Agent：约 35 小时、432 次 kernel evaluation、1,158 次工具调用，发布方报告多个 workload 相对 Triton 的几何平均 `10.0x`。另一区别于 H100 KernelBench L3 的 `1.98x/96%`。奖励作弊监控案例报告 >80 小时轨迹、>10,000 次调用、新增 13 条规则、识别 1,618 个案例，但未披露标注分母/precision/recall；不把规则计数写成准确率或“无作弊”证明。

版本约束：Model Studio `qwen3.7-max` alias 等价于 `qwen3.7-max-2026-05-20`（纯文本）；image/video 输入属于 `2026-06-08` snapshot。官方博文页面日期为 May 16，API metadata 又列 May 20，AA release 是 May 19，保留为来源日期差异。后续已找到 arXiv v1 “Qwen Technical Report” VHD-Play；它训练 Qwen3.6-35B-A3B，Qwen3.7-Max 仅作比较，不据此补写 Max 的内部 recipe，见本计划末尾的归属校正记录。

本轮内容同步到 [`qwen3.7-max-source-notes.md`](research/model-update-2026-09/qwen3.7-max-source-notes.md)、第二十册第 19.40、第十七册 Code Agent 小节、source index、model inventory、PAPERS、BOOK_SERIES、INTERVIEW_BANK、EXERCISES 和 KNOWLEDGE_GRAPH。QA 已通过：本轮涉及的章节/研究笔记围栏配对正确，关键相对引用目标存在，`git diff --check` 无错误。下一步仍只从两个指定排行榜里八家重点厂商的现有 canonical 集合选下一项知识缺口。

## 2026-09-28 Qwen Technical Report：VHD-Play 与 Qwen3.6-35B-A3B 的 Agent RL 环境

- 7890 代理可达性按具体 URL 记录：本工作区通过 `--proxy http://10.24.27.134:7890` 请求 arXiv 摘要页 HTTPS 200、43,529 bytes / SHA-256 `34bfd8e7f8b4931a008b3f1b35771f748ac00da837681be3861abaedad7a6e2b`；之后经同代理请求完整 `/html/2609.27321v1` 返回 curl error 7。用户贴出的百度 HTTP 成功与摘要页 HTTPS 成功证明代理在具体 URL/时点可用，但不保证任意目标可达。完整论文 HTML 使用已有 1234 快照分析，461,978 bytes / `da56349817e450cde05eeed3be1cf2274217cec17a849f4111c697e270cee28a`；PDF 下载不完整，未使用。
- arXiv v1 日期 2026-09-23，评论 “Qwen Technical Report”。VHD-Play 先采样并求解机制、固定 optimum/default reference 与 normalized outcome reward，再生成隐藏状态和 stateful tools 环境。论文报告 3,300 environments（2,200 train / 300 in-domain held-out / 800 unseen-family eval）、11 个机制族，其中 3 族用于训练。
- 训练主体是 Qwen3.6-35B-A3B：GRPO 34 steps，64 prompts/step，每 prompt 18 attempts 留 16 rollouts；作者报告五族 agentic mean `0.204 → 0.815`，written-out `0.962 → 0.992`。外部 BFCL 十 cell 均值 `61.25 → 64.08`、TravelBench `0.700 → 0.794`、E-Commerce 五次 run 期末余额均值 `54,294 → 182,844`。全部是论文作者结果，非本项目复现；作者注明 one training run / one evaluation seed，reference replay coverage 仅 8/11 families。
- 归属边界：Qwen3.7-Max 仅作额外 setter / benchmark comparator；E-Commerce 比较中训练模型均值 182,844 对 Max 165,224，TravelBench 则训练模型 0.794 低于 Max 0.891。非配对 setter draw 和小样本限制不能支持普遍胜出。论文未声明 VHD-Play 等于此前博客的 `Task × Harness × Verifier` rollout，不能将 3.6 的 GRPO recipe/分数写回 3.7 Max。
- 已更正 Qwen3.6-35B-A3B、Qwen3.7-Max 研究笔记的论文状态与关联边界，并同步第二十册 19.41、source index、model inventory、PAPERS、BOOK_SERIES、INTERVIEW_BANK、EXERCISES、KNOWLEDGE_GRAPH。本轮 QA 通过：`git diff --check` 无错误，相关 Markdown 围栏配平，新加的研究笔记/章节相对链接与锚点均存在。
- 后续只按两张指定排行榜维护八家重点厂商的模型锚点；该论文是既有 Qwen3.6/Qwen3.7 锚点的周边资料，不从作者比较项扩展新模型。完成 QA 后，从现有 canonical 队列继续下一项。

## 2026-09-28 Gemini 3.8 Flash：7890 当前联网与 Interactions schema 复核

- 继续沿双榜已有 `Gemini 3.8 Flash` 锚点核验协议字段，没有从 Google 文档或 SDK 扩展候选模型。用户贴出的百度响应与当前工作区测试一致：`10.24.27.134:7890` 返回 HTTP 200；本轮通过该代理重新获取 Google AI Developers 文档。
- Interactions overview 当前指向 `/api/interactions-api`（HTTP 200，739,123 bytes）；旧 `/api/interactions` 本轮 HTTP 404，按官方新链接更新引用。Thinking、Tool combination、Function calling、旧 Thought signatures redirect、Interactions overview 与 API reference 的当前快照大小/SHA 记录在 [`gemini-3.8-flash-source-notes.md` §18](research/model-update-2026-09/gemini-3.8-flash-source-notes.md)。
- API reference 与截至 9 月 26 日可见的固定 SDK main commit `6d012889752f65c1a51d0ad6e5970fc97d19c4ca` schema 一致：`function_call.id` → `function_result.call_id`；custom FunctionCall/Result typed fields 不声明 signature，ThoughtStep signature 为 optional。Thinking prose 则说 thought signature 必有并排除 custom call；Tool combination prose 仍称 `function_call`/`function_response` 的调用和结果均有 signature。schema 更支持 Thinking 的窄口径，但 SDK `extra="allow"` 与 `UnknownStep.raw` 保留 extra/未知 payload，schema 缺字段不代表 endpoint 禁止或不会返回。
- 收尾时当前工作区经同一 7890 重取 Interactions API reference：HTTP 200，739,123 bytes，SHA-256 `c5a0220ed169d8b6b60ece04fd548b392530a0685b6bae0cd6d140272d1400a7`。与本节较早同尺寸快照 hash 不同；本次可见的 `FunctionCallStep`/`FunctionResultStep` 字段与 `ThoughtStep.signature` optional 标记仍如上，不把动态页面 hash 差异当作 schema 或服务端行为变化。
- 两榜收尾复验：Artificial Analysis `/zh` HTTP 200，1,660,509 bytes，SHA-256 `c53e90519dc3bf22b391bc7673fe59fabecc83d930b74d5023e42cf4790a9d0d`；同日先前快照字节数相同但 hash 不同，抽取的 `/zh/models/...` 路径集合相同。DataCurve HTTP 200，268,036 bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，与先前快照和配置 ID 集合一致；两榜未发现新的 canonical/config ID。本轮只跟踪八家范围，不扩展至首页出现的其他厂商。
- 已同步 Gemini 3.8 研究笔记、第二十册第 23 章、来源索引、模型盘点/解释、PAPERS、INTERVIEW_BANK、EXERCISES、KNOWLEDGE_GRAPH 与本计划/进度。无 Gemini API key 或授权 endpoint，本轮未做真实 capability probe；没有下载模型权重或新增模型。
- QA 已通过：相关 Markdown 围栏配对，`git diff --check` 无错误；`gemini_interactions_replay_demo.py` 输出 `ok=true`、`network_called=false`。下一步仍只从 Artificial Analysis/DataCurve 已确认的八家重点厂商 canonical 集合选高价值知识缺口；goal 保持 `active`。

## 2026-09-28 GPT-6 Astra：官方资料复验与闭环确认

按两榜既有 GPT-6 Astra 锚点，经 7890 复取官方模型页、GPT-6 family guide 和技能/提示词博客 Markdown。模型页及 family guide 与已存快照逐字节一致；博客重读后确认 progressive disclosure、任务相关 `AGENTS.md`、精简指令和完成条件均已落入第六册第 18 章，没有新增技术点。Astra 仍为内容专题闭环，不新增模型、不调用 API、不推测内部架构；后续活动锚点回到本计划顶部记录的 GPT-6 Luna。

## 2026-09-28 Claude Sonnet 5：Compaction/API contract 复验

7890 在当前工作区返回百度 HTTP 200，并成功获取 11 份 Anthropic 官方 Markdown 全文。复核两榜后仍使用既有 `claude-sonnet-5` canonical，不从官方文档新增模型。Research notes 记录各快照哈希和完整接口边界；本轮只做文档研究，没有调用 Messages API。

关键协议增量：on-demand `compact-2026-09-04` 使用独立 `compaction` 参数，回传 signed summary block 时要求唯一、最新、完整原样且排在 history 首部；threshold `compact-2026-01-12` 使用 `context_management.edits`，在普通响应中自动压缩并由服务端裁剪旧前缀。threshold 默认 trigger 为 150K input tokens、最低可设 50K。两种 beta 模式各有平台兼容范围，不可同请求组合。keep-tail/background 需正确切前缀并保留并发期间新消息；cache breakpoint 可分离系统提示缓存与新 summary。Preserved-thinking 回放条件与 prefix-binding enforcement 需分开：文档说明后者从 Fable 5.1 开始，Sonnet 5 不运行此 prefix check。以上已同步第二十册第 7 章、INTERVIEW_BANK 与 EXERCISES，Sonnet 5 仍为资料级闭环。QA：`git diff --check` 通过，相关章节 fenced-code 数为 58（偶数），关键研究笔记/章节文件存在；下一步从两榜的八家重点厂商集合继续选锚点。

## 2026-09-28 Claude Opus 5.5：Compaction 与 preserved-thinking（已收尾锚点）

本专题曾从已完成本轮复验的 Claude Sonnet 5 转到两榜中已有的 `Claude Opus 5.5` canonical，现已收尾；没有从 Anthropic 文档发现新模型，也没有把 Sonnet/Fable 的 Agent 行迁移给 Opus。7890 当前可用：用户提供的百度请求成功；当前工作区也已通过该代理读取 Anthropic 文档和两榜页面。

本次收尾即时复测 7890：Anthropic compaction Markdown HTTP 200、11,974 bytes；Artificial Analysis `/zh` HTTP 200、1,660,509 bytes。此结果只说明这两个 URL 在本次时点可达，不代表所有站点或 API 均可用。

本轮将 Anthropic 的 on-demand (`compact-2026-09-04`) 与 threshold (`compact-2026-01-12`) compaction 分别建模，补明 Opus 5.5 threshold 前序 thinking 不保留、on-demand keep-tail 的条件、各平台 beta 覆盖差异，以及 2026-08-31 00:00 UTC 起账号默认 prefix-check 与旧账号 opt-in。model binding、prefix binding 和 compaction mode 是不同门禁。研究笔记、第二十册第 7.25 节、source-index、model-inventory、INTERVIEW_BANK、EXERCISES、BOOK_SERIES、KNOWLEDGE_GRAPH 均已同步；没有新增参数、架构或训练 recipe 证据。

专题状态保持 **AA 单榜内容专题闭环 + 官方 System Card/API contract + local protocol toy**；DataCurve 无精确 Opus 5.5 Agent 行。本轮未调用 Anthropic Messages API。QA：协议 toy 主流程及负例通过；受影响文件 `git diff --check` 通过；第二十册第 7 章代码围栏数 58（偶数），研究笔记 4（偶数），相对引用目标存在。Sonnet 5 标为上一活动专题，后续仍仅从 Artificial Analysis/DataCurve 的八家重点厂商 canonical 队列选下一高价值知识缺口；总目标保持 `active`。

## 2026-09-28 Gemini 3.8 Flash：Interactions SDK namespace 与 replay-policy 边界（当前活动锚点）

本轮仍以 AA 已确认的 `gemini-3-8-flash` high 与 DataCurve 精确 `mini_swe_agent_gemini_3_8_flash_high` 为锚点；AA 首页 slug 集合与同日快照无差异，DataCurve 页面/hash 与先前一致，没有从 Google 文档或 SDK 扩展模型候选。当前 Agent 行为 `330/447`，Pass@1 `73.8255%`、Pass@4 `85.8407%`、4 runs；只作 high + mini-SWE-agent + tools/environment/verifier 系统结果。

新补证：当前 Google official prose 继续存在 signature 范围冲突。固定最新 Python SDK main commit `6d012889752f65c1a51d0ad6e5970fc97d19c4ca` 的 Interactions 专属 `_gaos` schema 显示 custom `FunctionCallStep`/`FunctionResultStep` 未声明 signature、`ThoughtStep.signature` optional、`_gaos` BaseModel `extra="allow"`，且 lenient open union 用 `UnknownStep.raw` 保存未知 variant。同仓库 `_common.BaseModel extra="forbid"` 是不同类型栈。由此再区分 schema optional 与本地 replay toy 更严格的 host policy；不推断真实 endpoint 接受/返回行为。

已向研究笔记 §19、第二十册第 23 章、replay demo、题库/练习、模型盘点、来源索引、BOOK_SERIES 与知识图谱同步。尚待 QA 完成后，从同两榜八家重点厂商现有 canonical 队列继续选择高价值缺口。真实 Interactions API probe 仍待授权，goal 保持 `active`。

## 2026-09-28 DeepSeek V3.2：Search Agent 上下文管理补证

本轮沿既有 AA `deepseek-v3-2` 条目复核，7890 代理在当前工作区访问 AA 详情、DataCurve DeepSWE 和 arXiv v1 HTML/PDF 均 HTTP 200。AA 当前快照 3,647,642 bytes / SHA-256 `239b8fef11d18d7b06e5e7c177ed76cbbfd29e07d795d83c5dc3e6ac8acef0b8`，字段仍为 Non-reasoning、685B/37B、128K、Index `16.043537719683`；历史 648B 不解释为模型变化。DataCurve 快照 268,036 bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，无精确 V3.2 行。

arXiv v1 HTML（295,170 bytes / `5da74d488b218a45a995838b94feb95aacc63c03b806f8462496bfd4bceb07ef`）§4.4 给出 Search Agent 超过 80% context 时的 Summary、Discard-75%、Discard-all，与 N 条并行轨迹选最少 steps 的 baseline；约 20%+ 案例超过 128K，BrowseComp Pass@1 无管理/有管理为 51.4/67.6*。Summary 的 364 steps 与“up to 60.2”按原文保留，未臆加百分号。已同步到第二十一册第 19 章 19.35、第二十册第 18 章 18.27、INTERVIEW_BANK、已有 BrowseComp exercise、PAPERS、KNOWLEDGE_GRAPH、模型清单和 source-index；不新增模型、不迁移邻近版本成绩。

QA 已完成：当时受影响文件 `git diff --check` 无错误，两条跨章相对链接可解析，新增章节各有唯一 heading，书稿代码围栏数量均为偶数。Figure 6 当时尚未视觉核验，随后已于 2026-09-28 单独完成（见进度记录）；完整权重、production kernel、硬件 profiling、线上 acceptance、独立 benchmark 和完整 RL recipe 仍不在当时证据范围。总 goal 继续 `active`。

## 2026-09-28 Gemini 3.8 Flash：Agentic Video 周边知识补齐

- 锚点仍是两个指定榜单已发现的 Gemini 3.8 Flash；AA/DataCurve 的旧配置只作已有目录证据，本轮没有新增模型。DataCurve 精确行仍绑定 `mini-swe-agent` + high + 工具/环境/verifier，不迁移给其他配置。
- Google Video Understanding 早先经 7890 成功取得的 2026-09-28 快照明确展示 `gemini-3.8-flash` 的 agentic 视频处理。已记录 static 1 FPS、agentic 的 transcript/frame/audio 按需读取、`processing_call`/`processing_result` 配对、static-only 自定义 FPS/interval、thought/tool-use token 分账和长/短视频的 token-quality-TTFT 权衡。没有把发布方文档数据当作独立复现或内部架构证据。
- 当前时点链路复测：`http://www.baidu.com` 经 7890 HTTP 200 / 2,381 bytes；AA Gemini 3.8 详情和 Google Video Understanding 的 HTTPS 重试均 curl `35` TLS EOF。已将两者区分记录；不把单一 HTTP 站点成功等同于 HTTPS 站点普遍可达，也不以失败请求代替成功页面快照。
- 本轮 QA：`git diff --check` 通过；研究笔记、第十七册第 7 章和第二十册第 23 章的代码围栏分别为 2/18/6（均为偶数）；关键跨文件链接的目标文件与章节标题均存在。真实 Gemini API capability probe 仍需 API key/授权，不下载权重。下一步回到两排行榜的八家重点厂商既有 canonical 队列，选取下一个有权威材料且未闭环的面试知识缺口；不从排行榜外新增模型。

## 2026-09-28 DeepSeek V4.1-Flash：7890 网络复验与 vLLM release 状态

用户 shell 给出的百度 HTML 已在当前工作区重现；7890 下百度、Artificial Analysis `/zh`、DataCurve DeepSWE、DeepSeek 官方公告和 GitHub release 页面均 HTTP 200。GitHub release 网页确认 vLLM 最新 stable 仍为 `v0.30.0`；Atom feed 最新项是 `v0.30.1rc0` 预发布（ROCm/MI355 CI kernel mirror），不将 RC 写成 stable，也没有观察到 V4.1 专属 release-note 增量。`api.github.com` latest-release endpoint 单独返回 403，已改用网页与 Atom feed 交叉核验。SGLang/PyPI 当日留存快照仍为 `v0.5.20`。本轮没有解析榜单模型集合，不声称有无新增锚点；详细快照及边界见 DeepSeek V4.1 研究笔记。下一步继续从两榜既有八家重点厂商 canonical 集合选公开证据仍有价值的缺口；goal 保持 `active`。

## 2026-09-28 GPT-5.6 Terra：DataCurve 原始快照漏项修正

复查 7890 刷新的 AA/DataCurve 后，发现当前 DataCurve HTML 内嵌数据包含已有 AA 锚点 `GPT-5.6 Terra` 的 `low/medium/high/xhigh/max` 五个 `mini_swe_agent` 配置，而 `model-inventory.md` 曾标作没有 Terra 行。DataCurve SHA-256 与本日此前快照相同，故定性为此前抽取遗漏，不宣称榜单刚新增数据。五档的 attempts、Pass@1/Pass@4、平均成本、输出 tokens 和 Agent steps 已更正至模型笔记、盘点和 source-index。结果只属于配置 + mini-SWE-agent + tools/environment/verifier，不改变 GPT-5.6 家族技术闭环或架构证据等级；不新增正式书稿章节。细节见 `gpt-5.6-source-notes.md` §15 和本进度最新记录。下一步仍从两榜既有 canonical 集合选择公开证据有价值的缺口，goal 保持 `active`。

## 2026-09-28 GPT-6 Astra：7890 官方定价页补验完成

沿既有两榜锚点 `gpt-6-astra` 补查官方定价页与固定版本信息。用户终端百度成功结果与本工作区沙箱内连接代理失败并不矛盾：按权限获批的沙箱外请求已通过 `10.24.27.134:7890` 取得 OpenAI 官方 model/pricing 页面 HTTP 200。model Markdown 快照与既有记录一致；当前 default snapshot 仍只有 `gpt-6-astra`，没有日期化不可变 ID。

定价页确认 Standard、Batch、Flex、Fast mode 的 short/long-context 费率；`>272K` 输入时整次请求按 input/cache 2x、output 1.5x 计价。Astra Fast mode 虽按 2x 适用价格收费，但无 latency SLA，EU data residency 不支持。详细费率、HTTP 状态与 SHA-256 已落在 Astra 研究笔记、来源索引和本进度，第六册第 18 章已更新。此次只补服务/计费合同，没有新增架构、训练方法或模型候选。

下一步仍先依两榜筛选既有八家重点厂商 canonical 集合内尚有面试价值的公开证据缺口；排行榜外不加模型，Astra 内部架构、训练 recipe、system card 和专属技术报告继续标为未公开/未核验，整体 goal 保持 `active`。

## 2026-09-28 Qwen3.6-27B 官方博客正文恢复与跨书系同步

- 7890 当前实测：通过 `10.24.27.134:7890` 请求 `www.baidu.com` 返回 HTTP 200、2,381 bytes；同一代理访问 Qwen 官方文章 API `https://qwen.ai/api/v2/article/?language=zh-CN&path=qwen3.6-27b&type=qwen_ai` 返回 HTTP 200、94,527 bytes。JSON 快照 SHA-256 `036a9cfc6a38b04fed0b72aaf9339296296356dd9441a9e63d4b720283ed90c0` 含动态 request_id；稳定文章 HTML 为 91,758 bytes / `7748e75a7c5a47943d6abe4e6415cb6b8d4749713eeff39323eb831f2d8ae367`。元数据：Qwen Team，《Qwen3.6-27B：270亿参数稠密模型，旗舰级编程能力》，2026-04-22。
- 直抓 `https://qwenlm.github.io/zh/blog/qwen3.6-27b/` 仍因当前连接失败未取得；此失败与 API 成功并存，不能推断代理或整站不可用。Qwen 官方博客 API 的正文已完整可读，不再把此前的客户端 shell 当作正文证据。
- 9 月 28 日榜单复验：AA `/zh` HTTP 200，1,674,187 bytes / `891ad0a15cd12b7ae704512351bdee86c0136b75e11704b8bd5f6705c0dab42f`；AA release 页面 941,371 bytes / `20d83181c16942ddc1b7a14268635f80848b87b4d77e11f9b6e9febc74f41afa`；DataCurve 268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`。八家厂商无新 canonical 锚点；Muse Spark 超出关注范围，DeepSeek V3.2 Exp 是弃用历史 revision，不升格为新模型。
- 正文披露发布方 SWE-bench Verified/Pro、Terminal-Bench 2.0、SkillsBench 分数及修订/评测口径，并示例 `preserve_thinking`、Chat Completions/Responses 与 Anthropic-compatible API、OpenClaw/Qwen Code/Claude Code。新增的教学强调：OpenClaw `131072` context / `16384` maxTokens 是客户端预算，与 262,144 native context 分层；百炼可用性文案有不一致，本轮未调用真实 endpoint。博客没有提供 Qwen3.6-27B 独有架构或完整训练报告。
- 交付同步到 Qwen3.6-27B 研究笔记、第二十一册第 83 章 83.16.5、第二十四册第 61 章 61.32、source-index、model-inventory、PAPERS、INTERVIEW_BANK、EXERCISES、KNOWLEDGE_GRAPH、BOOK_SERIES、plan 和本进度。目标 Markdown 围栏平衡，关键交叉引用目标存在，`git diff --check` 通过；goal 保持 `active`，下一步继续从两榜八家重点厂商 canonical 集合中选取下一项知识缺口。

## 2026-09-28 Qwen3.6-35B-A3B 官方博客正文与 Agent evaluator 专题

- 当前锚点只来自既有 Artificial Analysis `qwen3-6-35b-a3b` 与 Non-reasoning 配置；DataCurve 无精确 Qwen3.6 Agent 行。本次 API 获取仅扩展已确认锚点。
- 7890 请求第一次连接失败；通过权限审批后在沙箱外重试成功：Qwen 文章 API HTTP 200、94,678 bytes，JSON SHA-256 `d287402f27a6ffa3226466aa57407310f670eb73a6d72bcfef03597a94e35d49`（含动态 request_id）；稳定 HTML 正文 91,941 bytes / `706889d145ea17b8c8234c4cda35b00fdecc0b6bcb9e1f5f20d2ed3ff9e15ed1`。标题《Qwen3.6-35B-A3B：智能体编程利器，现已开源》，Qwen Team，文章时间 2026-04-15 10:00 +08；Artificial Analysis 和 QwenLM README 标 2026-04-16，未强行统一。
- 博客报告对 Qwen3.5-35B-A3B 的若干提升：SWE Verified 70.0→73.4、SWE-Pro 44.6→49.5、Terminal-Bench 40.5→51.5、SkillsBench Avg5 4.4→28.7、QwenClawBench 47.7→52.6、NL2Repo 20.5→29.4。成绩保留为 publisher-reported；记下 SWE-Pro 修订/重跑全部 baseline、Terminal-Bench 资源/时限/5 次均值、SkillsBench 78-task 子集/5 次均值，以及 NL2Repo 的 Claude Code/900-turn 条件。
- Agent evaluator 不是旁枝：TAU3 user simulator 为 GPT-5.2 low reasoning + BM25；VITA judge 替换为 Claude 4 Sonnet（原 Claude 3.7 Sonnet 不可用）；MCPMark 固定 GitHub MCP v0.30.3 与 32K Playwright 截断；MCP-Atlas 采用 Gemini 2.5 Pro judge。它们仅作该 Qwen 发布方评测设置，不从中另发现模型。`qwen3.6-flash` 是博客中同一 35B-A3B checkpoint 的百炼 API alias；OpenClaw 128K/16K 配置与 native 262,144 context/API 实际限额分开记录，未调用真实 endpoint。
- 已将知识同步到两本现有架构/Agent 章节及 inference budget 章节、研究笔记、榜单索引/盘点、PAPERS、面试题、练习、知识图谱、书系、计划和进度。专属架构与完整 training recipe 仍未从该博客证实；goal 保持 `active`。
- QA：Artificial Analysis 当前快照中确认 `qwen3-6-35b-a3b` 与 `qwen3-6-35b-a3b-non-reasoning` 两条配置；相关 Markdown 围栏平衡，关键交叉引用文件存在，`git diff --check` 通过。全程为文档研究/编辑，未下载权重或发起百炼 endpoint 请求；goal 保持 `active`，下一步继续从两榜现存八家厂商锚点选取知识缺口。

## 2026-09-28 Qwen3.8 Max (0902)：7890 复验与 cache / Responses 状态精化

- 用户贴出的百度 HTML 证明用户 shell 可经 `10.24.27.134:7890` 出网；本工作区本轮也经显式 `--proxy` 访问两榜和 QwenCloud 文档成功，均 HTTP 200。此前 agent 路径失败只描述先前时点，不等同于当前状态。
- Artificial Analysis 首页 1,674,048 bytes / `56359658c7d187e4a214ae65857f0a7668d42ebcc1da0e2d4fab477642e2640c`；AA releases 938,807 / `913306564eb762ad0587c8845bece903f5004a0198f64709af7318fbcb960ff2`；Qwen3.8 Max detail 3,881,440 / `f9c920db61e73a4d71fcd9a10d057d7bf22ce01d979974dd32ec4ed1f911cccb`；DataCurve 268,036 / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。AA 首页同日响应差 139 bytes，按动态页面变化处理；无新 canonical，DataCurve 无 0902 精确 Agent 行。
- 官方 QwenCloud 产品页 98,992 bytes / `ce97b7c9098ed54c195aa2c6b1a5c0dfc6ad43e7290aea10bcce8484f42b9a6b`，last-modified `2026-09-28 09:55:09`；Context Cache 1,134,820 / `47a01521f4ce689acf6f017c36c66af7c7c98984029eb5ba5c5f58e170b69a74`；Dynamic Rate Limits 412,225 / `033d1381bfd1c759834bb3402b75e53acca8a7c617480f3424b53b0080ec6460`；Responses reference 952,619 / `fd467c8d7ce159e2c98f7220ee599014ee28a1f3b9a6dd6309075313926a1940`。
- 关键补充：content block 与后续 message 的两种 20 窗口不能混为一谈；1,024 是 cache eligibility，非 hit guarantee，同页 explicit-cache 请求示例另写 `>1,024`，文档在边界上措辞不一；Chat Completions/Responses 分别用 `prompt_tokens_details.cached_tokens` 与 `input_tokens_details.cached_tokens`。Responses continuation 不继承 `instructions`，不能与 `conversation` 同用，`store=false` 不可续接，ID 文档有效 7 天。
- 已更新 Qwen3.8 Max 0902 source notes、source index、model inventory、第二十一册第 83 章、第二十四册第 32 章、INTERVIEW_BANK、EXERCISES 和 cache audit toy；将“input must exceed 1024 tokens”准确归到 explicit-cache 请求示例，而非 Session 示例。toy 输出 `ok=true`、`evidence=local_protocol_toy`、`network_called=false`；`git diff --check`、目标 Markdown 围栏平衡、关键引用目标存在、错误旧标签无残留均通过。未发起 QwenCloud API 请求、未下载权重或声称生产 cache/Agent 已验收；goal 保持 active。

## 2026-09-28 GPT-5.5 Instant：OpenAI 官方动态 Instant alias 补证

- 本条目在前序阶段已经收口为 **AA 榜单级关联配置 + 官方身份负证据**；本次是当前官方目录复核，不重新列为活动锚点。继续只沿 AA 已有的 May/June 条目核查，没有从 OpenAI 模型目录另发现候选。
- 用户终端和当前工作区均已证明可通过 `10.24.27.134:7890` 访问外网；本轮对 OpenAI official docs 目录/模型页取回 HTTP 200，精确 `/api/docs/models/gpt-5.5-instant.md` 返回 HTTP 404（9 bytes / SHA-256 `e3ebaa16dd9d9b9fc107c42183fb6cf9d22927e1af03dbbdfa0ccc38e4e4ac31`）。OpenAI Models Markdown 12,055 bytes / `ad6adde919a6b4f2e92b03acb891366d2ded0e078229f406a519722c71a406c3`；GPT-5.5 模型页 4,132 / `fdb0fc8fe9ea7f276716c2a5b49b902e7e676ff262bb8adfdf2d5b8911bad3ba`；Chat Latest 3,294 / `b6b9dc5e8a6c720641cfc8ada231e311d66ebbf2ba5ac14d7d480535679079cc`；GPT-5.3 Chat 3,206 / `df599cfb2ef17e5de61e912684cefad0db9101f6dccd91ff15d72e5528cda6d2`。
- 官方目录将 `chat-latest` 描述为 ChatGPT 当前 latest Instant model，模型页说明其 underlying snapshot 会定期更新，但未披露当前 snapshot；GPT-5.3 Chat 页面明确指向 GPT-5.3 Instant 且已 deprecated。GPT-5.5 API 页明确 `gpt-5.5` / `gpt-5.5-2026-04-23`。没有官方证据证明 AA `GPT-5.5 Instant` 等于 `chat-latest` 或 `gpt-5.5`；不迁移 API 能力、价格或 DataCurve 结果，也没有发起真实模型 API 请求。
- 已同步到 GPT-5.5 source notes、model inventory、source index、第二十册 Responses API 既有章节和面试题；不新建重复架构章节。完整证据与 SHA-256 见 [`gpt-5.5-source-notes.md`](research/model-update-2026-09/gpt-5.5-source-notes.md) §9.2。下一步仍从两榜既有重点 canonical 队列选锚点，goal 保持 `active`。

## 2026-09-28 GLM-5.3：7890 复验与 slime compaction-aware trajectory

- 当前工作区显式通过 `10.24.27.134:7890` 请求两榜与 Z.ai 官方资料成功：AA `/zh` HTTP 200，`1,674,048` bytes / SHA-256 `b5b25416290ba65caf3aa8ab19cb4d35ccb42a1e77c00b140d75b505807d4e70`；GLM-5.3 详情 HTTP 200，`3,966,481` bytes / `cbc848636ff90d05fad3b45c1228291e0f4dc226b0a76dbf1837948e0b9bfe87`；DataCurve HTTP 200，`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。仍是 AA `glm-5-3` 与 DataCurve `mini_swe_agent_glm_5_3_max` `311/451`，没有重点厂商新增 canonical。AA 同为 1,674,048 bytes 的两份同日首页哈希不同，按动态响应差异处理。
- Z.ai GLM-5.3 Markdown 与 GLM-5.3 博客正文 JS 哈希分别仍为 `9545c3d6fb1cabfa5951928bbe9a535e6958561d3bd6b5f4354cd5d3b0f6c929` 和 `f809bf586f74b40a01ecd8d674f9ec3de469a5abeae125e2a849c9805605d0f3`；文档 `dateModified=2026-09-18T04:25:55.193Z`。官方继续只披露 GLM-5.3 继承 `SAO with compaction`，未公布 5.3 专属 schema、触发条件或质量门禁。
- 沿 Z.ai 博客引用的官方训练框架检查固定 [THUDM/slime commit `8ee9c1e`](https://github.com/THUDM/slime/tree/8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b)，提交日期 2026-09-23。GitHub REST API 命中匿名 rate limit 403，但 Atom、固定 codeload archive 与源码均成功取得；因此不把 API 403 误报为代理/整站不通。
- 新的训练侧知识：按 session message equality 建树，再按精确 token-ID prefix 对齐；保留 rollout 的 `prompt_ids/output_ids/logprobs`，tool/environment context 用 `loss_mask=0`、可追溯模型输出用 `1`；历史改写可 realign 或 fork，shared assistant prefix 只在一个分支训练，siblings 共享 `rollout_id`。默认 `fork_threshold_tokens=1,024` 是短 re-render/token-drift gate，不是 GLM compaction trigger。
- 固定 commit 的 customization 文档与 coding-agent README 写 `reward/K`，但 `TrajectoryManager.get_trajectory()`、`generate()` 和单测断言每个输出 sample 拿 full reward；记录为源内 docs-code discrepancy，不推断模型训练策略。源文件大小、SHA、证据边界与说明见 [`glm-5.3-source-notes.md`](research/model-update-2026-09/glm-5.3-source-notes.md#2026-09-28-当前榜单复验与-slime-的-compaction-aware-trajectory)。本地未安装 pytest，单测未运行。
- 已同步研究底稿、来源索引、模型盘点、第二十册第 20.22、INTERVIEW_BANK、EXERCISES 和 KNOWLEDGE_GRAPH；GLM-5.3 内容专题维持闭环，增加“关联框架源码证据”，不升级成 GLM-5.3 内部实现。QA 尚待本轮全部编辑完成后执行；goal 保持 `active`，随后只从复验过的两榜重点 canonical 队列选择下一项。

## 2026-09-28 GPT-6 Luna：7890 复验与 reasoning-update / compaction 边界

- 用户提供的百度 HTML 已证明其 shell 经 `10.24.27.134:7890` 可访问外网；本工作区也显式使用同一代理，百度、Artificial Analysis、DataCurve 与 OpenAI 官方页面均可读取。不同网络位置/时点的结果分别记账，不再把此前连接失败说成当前代理全局不可用。
- 当前快照：AA `/zh` `1,660,578` bytes / SHA-256 `d95b93814772c81fd71e3b33bf2851b7b87edf7d62ce5a7ef1336ae0169bae2f`；AA GPT-6 Luna detail HTTP 200，`3,840,137` bytes / `0c62e5133280f5b671d1d9c0b2fd550672d7ed2f3a96feab9149448969224f1f`，仍见 canonical `gpt-6-luna`。DataCurve `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，无精确 `mini_swe_agent_gpt_6_luna_*` 行；未迁移其他 GPT 配置分数。首页/详情动态字节变化不作为模型 revision 证据。对照现有模型清单没有新增重点厂商 canonical；AA 历史候选 GLM-5/GLM-5V Turbo 等不属于本轮锚点。
- OpenAI `gpt-6-luna.md` 为 `4,019` bytes / `561a86af72eced9a76a4e3a96c45ae7fff7782732514a8429b55ebddb4404945`；Reasoning `70,315` / `93327c5eb19df7ecf8c0bd9d581f58052dee9f090c38bb48906110b08d0ce251`；Agents `5,432` / `df4f61b609550619a3f9e445c66d28318b9f681affe1a819ef0098cde32bf43a`；Tools `33,282` / `4722fa102070178c1a1d603e718c69575c7c1768fa901218eab359bafdcba341`；Compaction `14,272` / `73fd2fd1afd44bd6f29ce7bd86fd0ae3413c98ec00ae5476879e98171b0b60dd`。与已保存快照一致，未见文档漂移。
- 官方 GPT-6-family API 合同补记：`configuration_update` 不得与自动 compaction/truncation 组合，含 update 的历史不能提交给 standalone `/responses/compact`；显式压缩可在 `/responses` 放 `compaction_trigger`，之后在下一条 user message 前重新加入 update。相邻 update 仍无效。该知识不是 Luna 专属架构，未调用真实 API；第二十册既有 20.27.2 已有通用状态说明，本轮仅补 Luna 来源锚点及面试/练习区分。
- 已同步 `gpt-6-luna-source-notes.md`、`model-inventory.md`、`source-index.md`、第二十册第 20.28 官方来源引用、INTERVIEW_BANK、EXERCISES、KNOWLEDGE_GRAPH、本计划与本进度。当前仍为 **AA 单榜资料级闭环**；官方未披露参数、架构、完整训练 recipe，精确 DataCurve 行、独立 benchmark 和生产验收仍缺。
- QA：对本轮 Markdown 执行围栏配对、关键本地链接和 `git diff --check` 检查；无 API 请求、模型权重下载或外部副作用。下一步仅从两个指定排行榜的八家重点厂商现有 canonical 中选择下一项高价值未闭环知识缺口；goal 保持 `active`。

## 2026-09-28 Qwen3.7 Max：官方博客正文、跨框架 RL 与长程 kernel

- 锚点只取自已有榜单：AA canonical `qwen3-7-max` 详情快照 `3,834,482` bytes / SHA-256 `829a322408dc044f12eefaa769eab0001904ceecb33d891abc801061fced7522`；同日 DataCurve DeepSWE 快照 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，无精确 `mini_swe_agent_qwen3_7_max_*` 行。AA Index `29`、约 207 tokens/s、1M context、约 `$1.15` 平均任务成本及 `$2.50/$7.50` input/output 均按第三方/provider 字段记录。
- 代理分开记账：本工作区显式经 7890 请求 `http://www.baidu.com` 返回 HTTP 200、2,381 bytes；同一代理下 Google、Qwen、Alibaba HTTPS 在 TLS 握手时 EOF。用户贴出的 shell 结果证明用户端 `7890` 可访问百度，不代表所有 HTTPS 站点/此时点的 agent 网络路径都成功。通过备用 8098 访问 Qwen 官方文章 API 成功（HTTP 200）。本轮复用当日 AA/DataCurve 快照，不把旧快照冒称为刚刚经 7890 重新获取。
- Qwen 官方文章 `https://qwen.ai/blog?id=qwen3.7` / API `path=qwen3.7` 返回完整正文：`122,153` bytes / SHA-256 `41e1f58384b99f1d2495111c3f6f55d01850283daa7ce5d07d38a764eea9108c`（含动态 request ID）。官方材料的新方法包括 environment scaling / OOD claim、`Task × Harness × Verifier` 组合式 rollout 与跨 harness/verifier RL；35 小时 M890 PPU kernel 案例报告 432 次 evaluation、1,158 次 tool calls、相对 Triton 几何平均 `10.0x`；SWE RL reward-hacking monitor 报告 >80 小时、>10k calls、13 条规则、1,618 个案例。均为发布方主张，不是独立复现；规则没有 precision/recall/分母披露。
- 重要区别：M890 的 10.0x vs Triton，不等于 H100 KernelBench L3 的 `1.98x/96%`；后者是中位 per-problem eager speedup / 快于 `torch.compile` 的题目比例。model alias `qwen3.7-max` 对应 May-20 纯文本 snapshot，June-08 才加 image/video；博文日期 (May 16)、API 元数据日期 (May 20) 与 AA release (May 19) 不强行合并。
- 论文状态：Qwen 博客称 environment-scaling 细节将见后续技术报告；arXiv 搜索请求超时，本轮未取到独立报告，不据此声称报告不存在。未下载权重、未做真实 API probe 或硬件复现。
- 已新增 [`qwen3.7-max-source-notes.md`](research/model-update-2026-09/qwen3.7-max-source-notes.md)，同步第二十册 19.40、第十七册 Code Agent 小节、source index、model inventory、PAPERS、BOOK_SERIES、面试题、练习、知识图谱及 plan。QA：研究笔记和两处章节围栏数均为偶数，关键相对引用目标存在，相关文件无行尾空格，`git diff --check` 通过；未下载权重、未调用模型 API 或运行 GPU 实验。goal 保持 `active`，下一步仍从两指定排行榜的现有 canonical 队列选高价值知识缺口。

## 2026-09-28 VHD-Play：Qwen3.6-35B-A3B 的 Agent RL 环境技术报告

- 7890 代理按 URL / 时点记账：arXiv 摘要页 HTTPS 200，43,529 bytes / SHA-256 `34bfd8e7f8b4931a008b3f1b35771f748ac00da837681be3861abaedad7a6e2b`；稍后相同代理请求完整 `/html/2609.27321v1` 遇 curl error 7。用户贴出的百度 HTTP 与摘要 HTTPS 都成功，但不保证每个目标稳定可达。此前经 1234 获取的完整论文 HTML 为 461,978 bytes / `da56349817e450cde05eeed3be1cf2274217cec17a849f4111c697e270cee28a`，据此分析全文；不完整 PDF 未使用。
- [arXiv:2609.27321v1](https://arxiv.org/abs/2609.27321v1)《Verifiable Hidden Dynamics Play: Generating Agentic RL Environments from Solved Mechanisms》，2026-09-23，页面评论为 “Qwen Technical Report”。VHD-Play 先抽样并求解数学机制，冻结 solver optimum/default references 与归一化终局 reward，再生成具有隐藏状态和 stateful tools 的环境。3,300 admitted environments 分为 2,200 train、300 同族 held-out、800 八种 unseen-family eval；11 个机制族里 3 个参与训练。
- 受训 checkpoint 明确是 Qwen3.6-35B-A3B。GRPO 报告设置为 64 prompts/step、每 prompt 从 18 attempts 留 16 rollouts、34 optimizer steps；五族 agentic diagnostic mean `0.204 → 0.815`，written-out `0.962 → 0.992`。外部 BFCL 十 cell mean `61.25 → 64.08`、TravelBench `0.700 → 0.794`、365-day E-Commerce 五次 run 的 mean ending balance `54,294 → 182,844`。所有分数/成本均属论文作者报告，不是独立复现。
- 重要限制与归属：作者注明 one training run / one evaluation seed；同类 reference replay 检查只支持 8/11 机制族。Qwen3.7-Max 是额外 setter / benchmark comparator，不是受训模型；E-Commerce 小样本中训练模型高于 Max，而 TravelBench 仍低于 Max。论文未声明 VHD-Play 等同于 Max 博客的 `Task × Harness × Verifier` 组合或其内部 recipe，不把 3.6 的训练结果回写给 3.7。
- 已补充 Qwen3.6-35B-A3B 与 Qwen3.7-Max source notes、第二十册第 19.41 节、source-index、model-inventory、PAPERS、BOOK_SERIES、INTERVIEW_BANK、EXERCISES、KNOWLEDGE_GRAPH、plan 与本进度。QA：相关 Markdown 围栏配平，新加本地相对链接/章节锚点均存在，`git diff --check` 通过。未下载权重、未调用模型 API、未独立运行论文代码或硬件实验；goal 保持 `active`，下一步按两榜 canonical 队列继续。

## 2026-09-28 Gemini 3.8 Flash：7890 联网确认与 Interactions 字段范围复核

- 当前工作区用 `10.24.27.134:7890` 请求百度返回 HTTP 200（2,381 bytes），与用户贴出的 shell 结果相符；同代理读取 Google 官方 Thinking、Tool combination、Function calling、Interactions overview 与新 API reference 均成功。Google overview 当前链接 `/api/interactions-api`；旧 `/api/interactions` 返回 HTTP 404。
- 最新正式 reference 与 SDK `main` 2026-09-26 commit `6d012889752f65c1a51d0ad6e5970fc97d19c4ca` 的生成 schema 一致：custom function 调用/结果分别为 `function_call.id` 与 `function_result.call_id`，二者 typed fields 未声明 signature；`ThoughtStep.signature` 在 reference/SDK 为 optional。Thinking prose 把 thought signature 说成必需并排除 custom function；Tool combination prose 则用 `function_response.id` 并称 Gemini 3+ 自定义 tool call/result 也有 signature，文档冲突仍在。
- 固定 SDK `extra="allow"`、dump 保留 extra fields、lenient open Step union 的 `UnknownStep.raw` 仍提供兼容性路径；这些静态 schema 不能证明真实 endpoint 接受/返回何种 extra 字段。没有 API key/授权 endpoint，未做真实 Interactions 请求或 capability probe。
- 已同步研究笔记 §18、第二十册第 23 章、source-index、model-inventory、inventory-interpretation、PAPERS、INTERVIEW_BANK、EXERCISES、KNOWLEDGE_GRAPH、plan。本轮只沿既有 Gemini 3.8 Flash 锚点，没有扩展模型清单。QA 通过：`git diff --check` 无错误、相关 Markdown 围栏配对；replay demo 输出 `ok=true`、`network_called=false`。真实 endpoint probe 仍待授权/API key，goal 保持 `active`。

## 2026-09-28 GPT-6 Astra：7890 官方文档复验

- 两榜当日快照中仍可见既有 GPT-6 Astra canonical；DataCurve 有 `mini_swe_agent_gpt_6_astra_{low,medium,high,xhigh,max}` 行。Artificial Analysis `/zh` 快照为 1,660,562 bytes / SHA-256 `19d1fb6c3bd1a433e16438e5a4fb7a330ceeca20716799c561c8eafa84fda838`；DataCurve 为 268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`。本条只记录已选锚点，不据首页片段推断全榜模型集合没有变化。
- 7890 下 OpenAI 官方模型页 Markdown 返回 HTTP 200、3,812 bytes / SHA-256 `f45ae813c3f69708e2576328056de14cdf71fec174c875f4b81d236105545625`；GPT-6 family guide 返回 HTTP 200、17,456 bytes / `8982485767fcefe9b4c588777de67683b4f56abe9c9af1121c9431eb9c305557`，均与现有快照一致。Astra 技能/提示词博客 Markdown 返回 HTTP 200、6,517 bytes / `a1deee4c0b3ca16a20385c27f0692e22d5150fdec81710c04742dd5da29f0468`。
- 技能渐进披露、任务相关 `AGENTS.md` 路由、避免冗长 recipe、定义完成条件与自主范围，已存在于 Astra 研究笔记和第六册第 18 章；没有发现模型架构、训练 recipe 或新运行时协议披露。已同步研究笔记、来源索引、模型盘点、正式章节、计划与进度；后续按计划继续 GPT-6 Luna。未调用模型 API 或下载权重，goal 保持 `active`。

## 2026-09-28 GPT-6 Luna：7890 复验与 reasoning effort 观测边界

- 用户终端提供的 `curl` 响应确认可经 `10.24.27.134:7890` 访问百度；本工作区本轮使用显式 `curl --proxy` 重新取得 OpenAI 官方 Luna model page、GPT-6 family guide、Reasoning 与 Compaction Markdown。四份文档分别为 `4,019`、`17,456`、`70,315`、`14,272` bytes，SHA-256 均与研究底稿中的既有版本一致。代理或单一 URL 的短时失败不据此外推为全局不可用。
- Reasoning guide 明确响应中的 `reasoning.effort` 继续报告 request-level setting，而非 `configuration_update` 选定的 effective effort；因此状态重建依赖有序会话历史。`configuration_update` 与 automatic compaction/truncation 及独立 `/responses/compact` 不兼容；显式 `compaction_trigger` 后，须在下一条 user message 前重新插入 update。该项是 GPT-6-family API/runtime 协议，不是 Luna 独有架构，本轮未调用真实 API。
- 已补充 [`gpt-6-luna-source-notes.md`](research/model-update-2026-09/gpt-6-luna-source-notes.md)、来源索引、本计划与本进度。Luna 仍为 **AA 单榜资料级闭环**，DataCurve 无精确 Agent 行；未迁移其他 GPT 成绩。下一步继续按两个指定排行榜的重点厂商 canonical 集合选择高价值、未闭环知识缺口；goal 保持 active。

## 2026-09-28 Claude Sonnet 5：7890 复验与 compaction 双接口入库

- 用户贴出的 `http_proxy=http://10.24.27.134:7890` 百度成功输出已在当前工作区复验：显式 `curl --proxy` 请求返回 HTTP 200、2,381 bytes。代理结论按具体 URL 与时点记录，不推断其他目标同样可用。
- 两榜仍锚定既有 Claude Sonnet 5，没有从 Anthropic 文档另增候选。AA 详情 `/tmp/aa-sonnet5-20260928.html` 为 3,849,279 bytes / SHA-256 `16557ff25caed39a04c1481ba199cd09eeb213db8a33534ff9c291b58fbce8a1`；DataCurve DeepSWE 为 268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`。
- 7890 取回 11 份 Anthropic 官方 Markdown：What's New、Migration、Prompting、Models Overview、Compaction overview、on-demand、threshold、preserved-thinking、keep-tail、background 与 thinking-block conditions，均 HTTP 200 / `text/markdown`；快照大小和 SHA-256 逐项登记在 [`claude-sonnet-5-source-notes.md`](research/model-update-2026-09/claude-sonnet-5-source-notes.md)。早先 70-byte regional/error 文件不作为正文证据。
- 新增重点：`compact-2026-09-04` 是显式 on-demand `compaction` 请求，返回 summary/signature block，后续唯一最新 block 必须原样置于 messages 首部并持续带 beta header；`compact-2026-01-12` 是由 `context_management.edits` 配置的 threshold compaction，服务端在普通请求中触发并裁剪旧前缀。两种 beta/header、block flow 和平台兼容表不同，不能混用；两份支持列表都含 Sonnet 5。on-demand 还需检查空 content/`stop_reason`、完整 tool result、summary usage、signed block 错误码及图像/文档/URL payload 不随摘要保留。
- 补充 background keep-tail 的并发换前缀流程、compaction 后 prompt-cache breakpoint、Models API `capabilities.compaction` 查询方法，以及 preserved-thinking 边界：通用原样 replay 是安全不变量；当前官方 prefix-binding enforcement 从 Fable 5.1 开始，Sonnet 5 不运行该 prefix check，不能把通用不变量写成 Sonnet 5 必然 400/drop。
- 已同步研究笔记、source-index、model-inventory、第二十册第 7 章、INTERVIEW_BANK、EXERCISES、plan 与本文件。Migration/Prompting 的面试补充包括默认 adaptive thinking、`content[].type` 解析、`max_tokens` thinking+text 共用硬上限、effort 档位与工具/自验证使用建议、sampling 非默认值 400 及 HTTP 200 refusal 分支；发布方建议未写成独立因果实验。
- QA/证据边界：本轮只阅读官方文档，不调用 Anthropic Messages API，不宣称真实线上 compaction、平台集成、prefix-check 行为或模型质量已经实测。Sonnet 5 保持资料级闭环，不新增模型或重复架构章节。受影响文件 `git diff --check` 通过，第二十册第 7 章 fence 数为 58（偶数），关键研究笔记/章节文件存在；goal 继续 active。

## 2026-09-28：7890 当前链路复验与两榜快照比对

- 当前执行环境显式经 `10.24.27.134:7890` 请求 `http://www.baidu.com`、Artificial Analysis `/zh` 和 DataCurve DeepSWE 均 HTTP 200；百度返回 2,381 bytes。由此确认本轮 agent 网络路径可用，不把单一代理/站点的历史失败外推为全局不可用。
- 刚取回的 AA `/zh` 快照为 1,660,509 bytes / SHA-256 `1e13276cae1ec1d9e5a510946b80029c3eb1c15cb746df7e7f647ecd6f1640bc`，与本日 `/tmp/aa-zh-7890-20260928-next-turn.html` 逐字节一致；DataCurve 为 268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，与本日 next-turn 快照逐字节一致。对照两榜重点厂商配置，没有新的 canonical 锚点；本轮不从模型文档、论文或代码仓库另发现模型。
- 作为既有 GLM-5.3 compaction 专题的只读 QA，重跑 `glm53_compaction_contract_audit.py` 成功：安全 candidate 的 schema round-trip、goal/plan、tool lineage、side-effect identity、pending effect、artifact/verifier、budget、cut marker 与 token budget 检查均通过；unsafe candidate 因状态丢失/重复副作用线索而被拒。输出明确标为 `local_protocol_toy`；不证明 GLM-5.3 实际 schema、触发逻辑或 SAO 实现。
- 本轮没有新增模型技术结论或专题章节，因此不改写闭环状态。下一步仍从复验过的两个排行榜既有八家重点厂商 canonical 集合中，挑选公开证据能继续推进、面试价值高的未闭环知识缺口；goal 保持 `active`。

## 2026-09-28 Claude Opus 5.5：Compaction 双协议与 thinking-state 边界

- 本轮沿 Artificial Analysis 已确认的 `claude-opus-5-5` canonical 锚点推进；DataCurve DeepSWE 当前无精确 `mini_swe_agent_claude_opus_5_5_*` 行，不迁移 Opus 5、Sonnet 5 或 Fable 5.1 的 Agent 成绩。没有从 Anthropic 官方文档另发现模型。
- 用户提供的 `10.24.27.134:7890` 百度请求成功；当前工作区先前同代理获取了 Anthropic 官方 compaction、preserved-thinking、Models Overview 与 Opus 5.5 What's New 等 Markdown。七份页面的 bytes/SHA-256 与逐页结论登记在 [`claude-opus-5.5-source-notes.md` §9](research/model-update-2026-09/claude-opus-5.5-source-notes.md)。两榜快照复验未发现新的 canonical 条目。
- 收尾时当前工作区再用 7890 直测：Anthropic compaction Markdown HTTP 200、11,974 bytes；Artificial Analysis `/zh` HTTP 200、1,660,509 bytes。仅记录这两个 URL 在该时点的可达性，不外推到所有外网地址。
- 关键增量：on-demand `compact-2026-09-04` 用顶层 `compaction`、返回 signed block，平台列表不含 Bedrock；threshold `compact-2026-01-12` 用 `context_management.edits`、在普通请求内压缩，平台列表含 Bedrock beta。Opus 5.5 只有 on-demand 在 preserved-thinking 条件满足时才可能保留最近 tail 的 thinking；threshold compaction 前的 thinking/redacted-thinking 不保留。prefix-mismatch 默认 enforcement 从 2026-08-31 00:00 UTC 创建的账号启用，旧账号需显式 opt in；model binding 与 prefix binding 是独立门禁。以上是官方 API 文档合同。
- 已同步第二十册第 7.25 节、研究笔记、source-index、model-inventory、INTERVIEW_BANK、EXERCISES、BOOK_SERIES、KNOWLEDGE_GRAPH、本计划和本进度。第二十册新增面试问答和练习覆盖协议混用、Bedrock 覆盖、thinking tail 与账号边界。
- QA：运行 `claude_opus55_protocol_audit.py` 成功，输出标记 `local_protocol_toy`、未发起网络调用；新增 negative cases 均通过。受影响文件 `git diff --check` 通过；第七章围栏数 58、研究笔记 4（均为偶数），相对引用目标存在。未调用 Messages API、未下载权重、未补写参数/架构/训练 recipe；Opus 5.5 保持 **AA 单榜内容专题闭环 + 官方 System Card/API contract + local protocol toy**。Sonnet 5 作为上一活动锚点；下一步仍从两指定排行榜的八家重点厂商 canonical 队列挑选，不扩展排行榜外模型，goal 继续 `active`。

## 2026-09-28 Gemini 3.8 Flash：7890 收尾复验

- 用户贴出的百度 HTML 与本工作区实际请求互相印证：当前工作区通过 `10.24.27.134:7890` 获取 Google Interactions API reference、Artificial Analysis `/zh` 与 DataCurve 均成功（HTTP 200）。本次 reference 为 `739,123` bytes / SHA-256 `c5a0220ed169d8b6b60ece04fd548b392530a0685b6bae0cd6d140272d1400a7`；相关字段仍显示 custom `FunctionCallStep`/`FunctionResultStep` typed fields 不含 signature，`ThoughtStep.signature` 为 optional。Hash 与同尺寸的先前快照不同，不据此推断 schema 变更；Thinking 与 Tool combination prose 的范围冲突仍未裁决。
- 两榜最新快照：AA `/zh` `1,660,509` bytes / SHA-256 `c53e90519dc3bf22b391bc7673fe59fabecc83d930b74d5023e42cf4790a9d0d`；同日先前页面 hash 不同，但页面抽取的 `/zh/models/...` 路径集合一致。DataCurve `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，与先前快照及配置 ID 集合一致；没有新的榜单路径/配置 ID，本轮不追其他厂商。
- QA：Gemini replay demo 输出 `ok=true`、`network_called=false`；研究笔记与第二十册第 23 章 fenced-code 数为 2、6；本轮相关改动 `git diff --check` 通过。真实 Gemini endpoint/capability probe 仍未进行，无 API key/授权 endpoint；goal 保持 `active`，下一步只从两榜既有八家重点厂商 canonical 队列选高价值知识缺口。

## 2026-09-28 DeepSeek V3.2：7890 复验与 Search Agent context management

- 当前工作区通过 `10.24.27.134:7890` 访问 arXiv v1 HTML/PDF、Artificial Analysis DeepSeek V3.2 详情与 DataCurve DeepSWE 均 HTTP 200。用户提供的百度结果与本工作区访问 arXiv 成功相互印证；代理结论只针对本时点的具体 URL。
- AA 当前详情 3,647,642 bytes / SHA-256 `239b8fef11d18d7b06e5e7c177ed76cbbfd29e07d795d83c5dc3e6ac8acef0b8`，仍为 `DeepSeek V3.2 (Non-reasoning)`、685B total/37B active、128K、Index `16.043537719683`；9 月 20 日 648B 是历史第三方目录字段，不是模型升级。DataCurve 268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，仍无精确 `mini_swe_agent_deepseek_v3_2_*` 行。
- 固定 [arXiv v1 HTML](https://arxiv.org/html/2512.02556v1) 295,170 bytes / SHA-256 `5da74d488b218a45a995838b94feb95aacc63c03b806f8462496bfd4bceb07ef`。§4.4 的新整理为：上下文用量超过 80% 时可摘要重启、丢弃最早 75% 工具历史、或清空全部旧工具历史；并行基线采 N 条独立轨迹并选最少 steps。报告称约 20%+ Search Agent 案例超过 128K；BrowseComp Pass@1 为 51.4 无管理 / 67.6* 有管理。Summary 平均 364 steps，“up to 60.2”原文未给百分号；结果绑定商业搜索 API 与 Agent harness，不是 DataCurve 或裸模型成绩。
- 已把知识写入第二十一册第 19 章 19.35、第二十册第 18 章 18.27、INTERVIEW_BANK、既有 BrowseComp Context Manager exercise、PAPERS、KNOWLEDGE_GRAPH、model-inventory、source-index、DeepSeek 研究笔记和 `plan.md`。未新增模型或迁移相邻版本分数。arXiv v1 PDF 与 HF 模型卡 PDF 是不同字节 artifact；章节引用版本化 HTML 段落，避免把它们称作字节一致。
- 该阶段 QA 已完成：受影响文件 `git diff --check` 无错误；第二十一册第 19 章与第二十册第 18 章的相对链接均解析到现存文件；新增 heading 唯一，两个书稿章节 fenced-code 数分别为 8 和 20（均为偶数）。当时 Figure 6 尚未视觉核验，随后已在本文件下方的“7890 Figure 6 视觉复核”记录中补核；完整权重/production kernel/硬件 profiling/线上 acceptance/独立 benchmark/完整 RL recipe 不在该阶段完成声明内；总目标保持 `active`。

## 2026-09-28 Claude Opus 4.8：Dynamic Workflows、System Card 多 Agent 评测与 PDF 正文补证

- 按既定规则沿两榜已有 `claude-opus-4-8` 历史锚点推进；未从 Anthropic 官方目录/博客另发现模型。用户提供的 `10.24.27.134:7890` 百度页面与本轮请求相互验证；经 7890 获取 AA `/zh`（1,660,509 bytes，`ac519155a550f6c4ae12a8f8a884f661c0f7b65127549f082a071f1281f91700`）、AA 详情（3,843,474 bytes，`bd4a6e4a148e099297a10e1aa7a5bcd7ab60762b912a6d1f4ed38c442b0b9579`）、DataCurve（268,036 bytes，`14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`）、Anthropic 公告（231,511 bytes，`285e10bb0b33c067c5c7e748e9dc02329e0eac259b37d36fa0c3cd9e8462ff22`）和 Dynamic Workflows 博客（559,172 bytes，`b939087ba9302ddaf433a8d99e548da130433ea0986a17b3689210cd5ec5f7ea`）。Dynamic Workflows 的首个 TLS 连接 EOF；用 `www.claude.com` + HTTP/1.1 重试 HTTP 200，内容 hash 与先前同日快照一致。
- System Card PDF HTTP 200，20,587,425 bytes / SHA-256 `0df5d061b6688c2d5839476620d66960011cb69043185dbed87ed1f68ab76d8f`。项目 `pdf_text_extract.js` 成功输出 246 页、422,194 bytes，提取文本 hash `fe6da487661e212b3e003da0c8e8d163d29d18f3a2ed6c7afb0fb2b2d7495f8e`。Card §8.11 定义 blocking orchestrator、fixed team、async-subagent 三类独立 benchmark harness；BrowseComp blocking 为 88.5%，五 Agent fixed team 85.4% vs single 84.3%（total token limit 5M vs 10M、derived latency 约 20%）；hard tail 中位约 3×、易题无明显 speedup。ProgramBench 166 golden tasks 上，三 Agent 在 score 0.6 约 1.8× latency improvement。Latency 是按固定 prefill/decode rate 折算 token 并加工具时间的派生指标，非 raw wall-clock/生产 SLO。
- 新知识落点：Dynamic Workflows 在对话外生成编排脚本、平行派发、独立检查/反驳、保存 checkpoint 后恢复；官方同时披露较高 token 使用、首次触发确认和管理员关闭选项。System Card 的短上下文 off-policy code-summary 测试中重要失败漏报率 3.7%，仅适用于该特定 transcript 评测，不能写成通用 honesty rate。Changelog 纠正 blocking budget 为 unlimited，并更正 live prompt-injection bounty 最终结果为 Opus 4.8 与 Opus 4.7 attack success 相同；以后引用以当前 Card/changelog 为准。
- DataCurve exact rows 未变化：`xhigh` 243/447（113 个尝试任务、91 个至少通过一次、Pass@1 54.36%、Pass@4 80.53%、平均 `$8.01` / 94.64 steps）；`max` 253/429（111 个尝试任务、88 个至少通过一次、Pass@1 58.97%、Pass@4 79.28%、平均 `$13.22` / 120 steps）。AA 当前标为 deprecated 并指向 `claude-opus-5`；本轮不把历史榜单状态写成当前推荐，不迁移跨 harness 分数。
- 已同步研究笔记、第十七册第九章、EXERCISES、INTERVIEW_BANK、PAPERS、BOOK_SERIES、KNOWLEDGE_GRAPH、source-index、model-inventory、`plan.md` 与本进度。状态为 Opus 4.8 历史锚点的 Dynamic Workflows/Multi-Agent **内容专题闭环**；没有独立架构章节，真实 API/生产 workflow、内部架构和完整训练 recipe 仍未验证/未公开；总体 goal 继续 `active`。

## 2026-09-28 Gemini 3.8 Flash：Agentic Video processing steps 与 7890 复测

- 本轮沿用两榜已有 Gemini 3.8 Flash 锚点；没有从 Google 文档或榜单外另增模型。已检查 DataCurve 本地快照中的精确 `mini_swe_agent_gemini_3_8_flash_high` 行：330/447 attempts passed、113 tasks attempted、4 runs、Pass@1 `73.8255%`、Pass@4 `85.8408%`、平均成本 `$2.3623`、平均输出 `143,242.7` tokens、平均 `166.31` Agent steps。该成绩绑定 high + mini-SWE-agent + 工具/环境/verifier。
- AA 本地详情快照确认 Gemini 3.7 Flash 已标为 deprecated，目录指向 `gemini-3-8-flash`；`39.0594566404896` Intelligence Index 属于 3.7 详情页的第三方字段，不能当作 3.8 当前分数。该阶段 AA 3.8 详情在线重试未成功；后续 7890 HTTP/1.1 重试成功，见本进度后续记录。
- Google Video Understanding 先前成功响应（经 `10.24.27.134:7890`）为 `303,547` bytes / SHA-256 `b2ec14972a66fd969108167f338e19e00cc3c80d3fc0ee604cfd962545d825c5`；同日 Gemini 3.8 模型页为 `106,650` bytes / `6ac136b1e6161438f10cdf1b509272e89825173e0b646a770e9c0cfe958a479b`。文档示例明确使用 `gemini-3.8-flash`。
- 新增面试主线：static 默认按 1 FPS 抽帧；agentic 根据问题动态加载 transcript/frame/audio，以 `processing_call.id` 和 `processing_result.call_id` 记录可观察的媒体证据读取过程。自定义 FPS/clip intervals 仅支持 static；视频导航 thought 使用 `total_thought_tokens`，按需媒体使用 `total_tool_use_tokens`。发布方称长视频最多约 88% token 减少、约 7% 质量提升，但短于 5 分钟可能增加 TTFT；均非本项目独立复现，需按固定 workload 实验。
- 7890 即时复测当时：`http://www.baidu.com` HTTP 200 / 2,381 bytes；AA Gemini 3.8 详情与 Google Video Understanding 的 HTTPS 请求均 curl `35`（TLS unexpected EOF），没有取得新响应。用户 shell 的 HTTP 成功与 agent 当时对具体 HTTPS 目标失败并不矛盾；后续 HTTP/1.1 重试成功，见本进度末尾。
- 已同步研究笔记 §20、第十七册 Code Agent、第十七册/第二十册交叉引用、题库、练习、PAPERS、BOOK_SERIES、KNOWLEDGE_GRAPH、model-inventory 和 source-index。本地只读的 2026-09-20 video 文档快照为 262,504 bytes / SHA-256 `7f58e353ad4b3753b6db639430f1aad76372c4fc37ebe5c26f4ac286bf40ede6`。QA：`git diff --check` 通过；研究笔记/第十七册第 7 章/第二十册第 23 章 fenced-code 数分别为 2/18/6（均为偶数）；关键跨文件目标与章节标题存在。真实 Gemini API probe 未做（无 API key/授权 endpoint），未下载权重；goal 继续 `active`。

## 2026-09-28 DeepSeek V4.1-Flash：7890 网络复验与 vLLM stable/RC 状态

- 用户贴出的 `http_proxy=http://10.24.27.134:7890` 百度请求成功，当前工作区也通过该代理取得百度 HTTP 200 / 2,381 bytes、Artificial Analysis `/zh` HTTP 200 / 1,660,542 bytes、DataCurve DeepSWE HTTP 200 / 268,036 bytes。这里只验证具体 URL 可达性，本轮没有解析两榜 canonical 集合。
- DeepSeek V4.1 官方公告经 7890 跟随重定向后 HTTP 200，`24,438` bytes / SHA-256 `f18dc22d37393381b31c9069996138f45aa7b02b08442d43af7c6c57f587bdce`。
- GitHub `vllm-project/vllm/releases/latest` 经 7890 HTTP 200，最终 URL 为 `.../releases/tag/v0.30.0`，`615,462` bytes / `ff2ddd43e2e969a5ccd9b141c43c9aa0dc36af24402cca781eb2e9760f4f8aa2`。release Atom feed 为 `734,203` bytes / `30131b0cf21894d9e8bf4cbd4eb0f51d8e47b364e3849db0ee1c5f8724810de5`，最新项是 `v0.30.1rc0`（2026-09-23T08:06:15Z）；固定 RC 页面 `224,871` bytes / `8034c62c3dbc34f8fb7897e8c2c5ed763ca8538884cc0965714a71b5629fb015`，可见标题为 ROCm/MI355 CI 的 NVFP4/MoRI kernel mirror。Latest-release API 返回 403，但网页和 Atom feed 可用；当前 stable 证据仍为 v0.30.0，没有看到 V4.1 专属 RC 增量。
- 当日已留存的 SGLang GitHub release 与 PyPI metadata 仍分别指向 `v0.5.20`，时间与 SHA-256 见 [DeepSeek V4.1 来源笔记](research/model-update-2026-09/deepseek-v4.1-flash-source-notes.md#2026-09-28-7890-复验与-vllm-stable预发布边界)。没有下载/安装 wheel、加载权重或做硬件验收；完整模型、目标硬件和生产 SLO 仍 `unverified`。无新增技术章节；goal 保持 `active`。

## 2026-09-28 GPT-5.6 Terra：DataCurve 原始快照漏项修正

- 沿 AA 已有 `gpt-5-6-terra` canonical 刷新两榜：AA `/zh` `1,660,542` bytes / SHA-256 `bc147b7c89c75bfb81257de530184f7f2d25b93543ee0ac82c93b2304a6cc38f`；AA Terra 详情 `3,841,417` bytes / `ecb893eec11fc3528e2df327abb152a773bd5cc1eb2a2cb13b7973d75333c41d`，`releaseDate=2026-07-09`。DataCurve `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，与本日先前保存的 DataCurve 快照逐字节一致。
- 原始页面内嵌数据已有五条精确配置：low `108/449`，Pass@1/4 `24.05%/44.25%`，均价 `$0.342`、21.46 steps；medium `158/450`，`35.11%/60.18%`，`$0.467`、25.15 steps；high `243/452`，`53.76%/80.53%`，`$0.908`、33.51 steps；xhigh `272/452`，`60.18%/80.53%`，`$1.702`、43.07 steps；max `314/451`，`69.62%/88.50%`，`$3.957`、75.93 steps。各档均尝试 113 个任务、4 runs；attempts 有轻微差异。
- 这是修复此前“Terra 无 DataCurve 行”的抽取遗漏，不是新的发布或分数更新。数据绑定 `mini-swe-agent + effort + tools + task environment + verifier`；可用于讨论成本/通过率观测权衡，不作裸模型质量或 effort 因果结论。已更新 GPT-5.6 研究笔记 §15、model-inventory、source-index、plan 和本进度；未新增书稿章节，GPT-5.6 原技术专题闭环状态不变。

## 2026-09-28 GPT-6 Astra：7890 恢复后官方定价页补齐

- 用户贴出的百度页面确认其终端可通过 `10.24.27.134:7890` 访问外网；当前工作区沙箱内显式请求同一代理先报 `curl: (7)`，获批的沙箱外重试成功建立代理隧道，并取得 OpenAI 官方 Astra model page 与 pricing page HTTP 200。此差异只说明执行环境路径不同，不代表代理或站点全局不可用。
- Astra model Markdown 为 `3,812` bytes / SHA-256 `f45ae813c3f69708e2576328056de14cdf71fec174c875f4b81d236105545625`，与 9 月 18/23 日相同；pricing Markdown 为 `23,475` bytes / `0de899a93b1d9a8c7cf7b3a01ed3553337636c7ff374153821f218a203a203c8`。对应 HTML 动态页面分别为 `431,650` bytes / `63be1ce7336601a62056eb8436f68d5cee0780aa6f095267c8f811dffe5c6ab5`、`552,704` bytes / `ec4a7fd545d1fb71e03de5278d417bb8eae77859df046b0c6d6d36590fe50f1a`；HTML hash/大小变化不单独视为模型 revision。
- 官方 model page 仍只列默认 snapshot/alias `gpt-6-astra`，无日期化固定 snapshot ID。pricing page 补全每百万 token 的 input/cached-input/cache-write/output 费率：Standard short/long `$10/$1/$12.50/$50`、`$20/$2/$25/$75`；Batch 与 Flex 分别为 `$5/$0.50/$6.25/$25`、`$10/$1/$12.50/$37.50`；Fast mode 为 `$20/$2/$25/$100`、`$40/$4/$50/$150`。模型页的 `>272K` whole-request 规则为 input/cache 2x、output 1.5x；Astra Fast mode 没有 latency SLA，EU data residency 不支持 Fast。费率和服务限制不构成架构、训练或质量披露。
- 已同步 Astra 研究笔记、第六册第 18 章、source-index、model-inventory、plan 与本进度。本轮不新增模型、技术章节或架构主张；GPT-6 Astra 继续为内容专题闭环，整体 goal 保持 `active`。

## 2026-09-28 DeepSeek V3.2：7890 Figure 6 视觉复核

- 用户提供的 `10.24.27.134:7890` 代理请求已返回百度页面；当前工作区也经该代理从 `https://arxiv.org/html/2512.02556v1/search.svg` 取得 HTTP 200、75,689 bytes。该 SVG SHA-256 为 `e2fb1029cf88458b2881d49546c1b9db4dd426fb4d2b3337dd6897b9c6fdb7a3`，与先前经 1234 取得的 `/tmp/deepseek-v32-figure6-1234.svg` 逐字节一致。
- 用 `gdk-pixbuf-thumbnailer` 将 SVG 渲染后，视觉确认 Figure 6 横轴 `Real Steps`、纵轴 `Browsecomp`，图例为 Summary、Discard-75%、Discard-all、Parallel-fewest-step。它支持论文对串行 context management 与并行 test-time compute 效率/扩展性的定性比较。
- 结合纵轴，正文 Summary “up to 60.2”是图中 Browsecomp 指标值，不是 `+60.2%` 相对增幅；364 steps 与 60.2 以论文正文为精确报告依据。图中散点无数字标签，未从像素估算并记录其他点位，也不声称其他 PDF 图表/公式已完成视觉核验。
- 已同步 DeepSeek V3.2 研究笔记、第二十一册第 19 章 19.35、source-index、model-inventory、`plan.md` 与本进度；没有发现/新增模型，活动锚点和整体 goal 不变。QA 已完成：`git diff --check` 通过，章节 heading 唯一，引用目标文件存在。

## 2026-09-28 Gemini 3.8 Flash：7890 HTTPS 失败项重试成功

- 两榜先行刷新：Artificial Analysis `/zh` 为 `1,660,542` bytes / SHA-256 `f5d63499b482b9510d14f2162e8317231da77f9b3875fa36171f53e5891e5c65`；与此前 `next-turn` 快照比较，55 个唯一 `/models/<slug>` 路由无新增/删除。DataCurve 为 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，与同日先前快照逐字节一致；未发现新 canonical 锚点。
- 重试此前记录的 AA Gemini 3.8 high 详情与 Google Video Understanding 文档，均通过 `10.24.27.134:7890` + `--http1.1` HTTP 200。AA 页面 `3,842,574` bytes / SHA-256 `f88d6155277116adc1580030fa298d821edf244375c76526e9d68572762b12f5`；release `2026-09-02`、Index `40.9262321765904`、1M context、`$0.75/$3.75` 未变。本页 `medianOutputSpeed=311.225851164667`，此前同日另一页面为 `328.820180257944`；按 AA 第三方 provider measurement snapshot 分开保存，不推断模型 revision。
- Google Video Understanding HTTP 200，`303,547` bytes / SHA-256 `e65216368a92d03403d4e37485148f7048b1f7e75be6de37891fd0ab2cb01414`。同字节数的先前成功页 hash 为 `b2ec14972a66fd969108167f338e19e00cc3c80d3fc0ee604cfd962545d825c5`；核心 `processing_call`/`processing_result`、`total_thought_tokens`/`total_tool_use_tokens` 字段仍在，动态 HTML hash 不单独证明文档语义变化。
- 结果已同步 Gemini 3.8 研究笔记 §20.4、source-index、model-inventory 和本进度；当前联网失败项修复，但无 API key，真实 Gemini API probe 仍未进行，也没有新增锚点或模型。

## 2026-09-29 DeepSeek V3.2：两榜刷新与 Figures 1–7 视觉核验

按既定范围刷新 Artificial Analysis 与 DataCurve，没有从官方资料发现新模型。Artificial Analysis `/zh` 快照 1,694,614 bytes / SHA-256 `04f62f3f39c1f65c3c1d8dd564782f9d7d20fb506fa671ee75fcfaf4e9482f9f`，唯一路由由 55 增至 62、无移除；七条新增中 GLM-5.3-Flash、Kimi K3、Qwen3.8 2.4T A95B、Qwen3.8 27B 已是盘点内重点 canonical，另外三条在关注范围之外。DataCurve 快照 268,036 bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，70 个唯一 `mini_swe_agent_*` 配置 ID 与留存快照无变化。结论：没有新增重点 canonical 锚点，也没有精确 DeepSeek V3.2 Agent 行。

7890 当前工作区请求在百度与 arXiv Figure 2 一度返回 HTTP 200；紧接的单独下载发生代理连接失败，获批重试后取得 Figure 2 SVG（269,427 bytes / SHA-256 `1e6bc6c61ea26ae2b8528edb1eab14874832470fd38f115fe9b8feb615facb9a`），渲染为 1600×833 并完成视觉核验。图示确认 Lightning Indexer → Top-k candidate selection → Core Attention 的模块关系；结合图注、§2.1 与 Eq. (2)，index score/top-k 是检索与筛选信号，不是主注意力输出。已新增第二十一册第 19 章 19.36，并同步研究笔记、source-index、model-inventory 和本进度。其他图表/公式仍未全部视觉检查；完整 production kernel、召回/误差曲线、硬件 profiling、线上 acceptance、独立 benchmark、完整 RL recipe 仍待核验。QA 通过后，从两榜既有八家重点厂商 canonical 队列继续选择下一知识缺口，goal 保持 `active`。

Figure 3 的 prefill/decode cost panels 也经 7890 获取并视觉核验：prefill SVG 33,398 bytes / SHA-256 `b045c26eb19d92325de7e86aabec905a9ac8bdb6729db94e04c8701693b19705`，decode SVG 31,868 bytes / SHA-256 `5894e01515f7f9e9ef9045ae8e30a092f5cc1e4229936c493c66c4748127c61d`，均渲染为 1600×1200。曲线显示长 token position 下 V3.2 每百万 token 成本曲线更平缓，但极短位置有交叉；报告将成本绑定 H800 实际部署服务和 `$2/GPU-hour` 租赁估价，不能称作 API 价格或通用成本保证。章节扩展到 19.37，并补充 interview bank 问题；Figure 2、Figure 3 与 Figure 6 已视觉检查，其他图表/公式及 production kernel、召回/误差、硬件 profiling、线上 acceptance、独立 benchmark 和完整 RL recipe 仍待核验。QA 已通过后，下一步从两榜既有八家重点 canonical 队列选择知识缺口，goal 保持 `active`。

随后复核 Figure 4 JPEG（76,981 bytes / SHA-256 `58623875cc487b3cbbd60955c14801c07d5f526d2b2e875795e3bf5ace511733`，原图 1280×671），视觉确认同一轮工具消息追加时 reasoning 延续，而新 user message 到来时旧 reasoning 被清理、tool calls/results 与前轮 answer 保留。该图帮助把 thinking-retention 讲成消息 role/history replay 合同，不是永久记忆；已同步至第二十一册 19.38、PAPERS 和 INTERVIEW_BANK。

Figure 5 [合成 Agent 数据 RL 曲线](https://arxiv.org/html/2512.02556v1/figures/synthesis-rl-plot.png) 经 7890 获取，222,137 bytes / SHA-256 `2107344cc2002f51bc1922df0bdcabd7afa74bb0f6cb5d3aa996f33e3f0d7cc3`，1432×1024，已视觉核验。图中比较从 V3.2-SFT checkpoint 出发、使用合成 general-agent tasks 做 non-thinking RL 的训练曲线，与 SFT 和 search/code RL 的 V3.2-Exp 基线；各环境走势总体上升但存在波动。将其严格表述为发布方训练消融：不证明最终 V3.2 的独立 benchmark、真实环境泛化因果、训练数据无污染或完整 RL recipe 已公开。已新增第二十一册 19.39，并同步研究笔记、source-index、model-inventory、PAPERS 与 INTERVIEW_BANK。

在 Figure 1 复核前已视觉核验 Figures 2–7；其余论文公式仍未全部检查。用户贴出的百度成功结果之外，本工作区也经 `10.24.27.134:7890` 访问 Google 得到 HTTP 200、84,771 bytes；只证明本时点该请求可达。下一步继续从两个排行榜内八家重点厂商的既有 canonical 队列选择高价值知识缺口，不从官方资料另发现模型；总 goal 保持 `active`。

后续从 arXiv v1 HTML Appendix A 找到 Figure 7 的真实资源，而此前 `figure7.svg` 返回 404 只是文件名猜错。MHA/MQA 两张 SVG 经 7890 取得并视觉核验，分别为 244,227 bytes / `afd21a50bbaef58314036862cb6ce44dca81a9d42a414c0969074fbf954b32b4` 与 224,311 bytes / `c6aa4f3e2ea222d2a75ef000dca2f9e8c48a820116957c6a741cbd0f253d77c3`。图解释 MLA 的 per-head K/V 投影与 shared latent KV 两种执行形态；图注的 V3.1-Terminus 阶段策略与 V3.2-Exp demo 分开归因。已补第二十一册 19.40、研究笔记、source-index、model-inventory、PAPERS、INTERVIEW_BANK；其余图表/公式仍待核查。

截至公式复核前，Figures 1–7 已视觉核验，当前未发现新的榜单锚点；继续从两榜既有八家重点厂商 canonical 队列推进。总 goal 保持 `active`。

Figure 1 的正式核验：`v32_performance.svg` 经 7890 返回 HTTP 200，128,338 bytes / SHA-256 `14354740f4d54692b6af6323cc12d3a5f0e0f937bc2b7dd65021307bd7820973`。视觉核实图表分开 Reasoning/Agentic 任务组，Codeforces Rating 走右侧独立纵轴，其他准确率/Pass@1 走百分比轴。报告限定 HMMT February 2025、HLE text-only；V3.2-Thinking 的 HLE 统一模板 bar 为 25.1，正文另报官方 HLE 模板 23.9；tool-use 还绑定 thinking/function-call 与内部 MCP 环境。已同步第二十一册 19.41、研究笔记、source-index、model-inventory、PAPERS、INTERVIEW_BANK；不把作者评测当 AA/DataCurve 或独立复现。

DSA 公式文本审计：从固定 arXiv v1 HTML `annotation encoding="application/x-tex"` 核对 Eq. (1)–(4)，串起 weighted-ReLU ranking score、Top-k latent-KV retrieval、dense teacher distribution KL 与 sparse selected-set KL。Eq. (4) 明示 `p_{t,S_t}`，但该段没交代子集后是否再次 normalization；记录为待查训练实现的问题，不臆测。7890 取得的 arXiv v1 PDF 为 980,616 bytes / SHA-256 `2bec0671778769c159ec389412727d1f3d4889fe1c71564b61edaa24705bd17b`；当前工作区没有可用 PDF renderer，故仅声称 HTML TeX 文本核验，不声称 PDF 公式视觉核验。已新增第二十一册 19.42、题库第 18 题并同步研究笔记、source-index、inventory、PAPERS 与本进度。

最终状态：Figures 1–7 已视觉核验，Eq. (1)–(4) 的 HTML TeX 源文本已核对但未在 PDF 上视觉核验，其他公式/实现细节仍待查；goal 保持 `active`。

训练实现查证：经 7890 访问 V3.2-Exp 官方仓库页面和 raw README，分别 HTTP 200（295,427 / 6,899 bytes），README hash 与先前快照相同。当前可见根目录有 `inference/`、报告 PDF、README、license、cost image；README 指向 inference demo 和外部 kernels，没有 trainer/loss 入口。GitHub API tree endpoint 403，故这只是“当前 V3.2-Exp 仓库没有公开路径能回答 Eq. (4) normalization”的范围内结论，不推断其他仓库/私有训练实现。

当前推进（2026-09-29，DeepSeek V3.2）：继续沿两榜已发现的 AA 锚点核对 arXiv v1 技术报告 Eq. (5)–(9)，未从论文或官方仓库新增模型。已补充 GRPO 的 group/token averaging、advantage 中心化、unbiased-KL importance weighting、negative off-policy sequence mask、Keep Routing 与 Keep Sampling Mask，并同步第二十一册 19.43 和书系配套文件。已核对 HTML 中全部编号公式 (1)–(9) 的 TeX 源文本，但没有 PDF 页面视觉核验；未编号行内数学表达式未做穷尽审计。完整训练 recipe、独立复现与生产验收仍待查。下一步沿既有八家重点厂商 canonical 队列选公开知识缺口。

## 2026-09-29 DeepSeek V4.1-Flash：7890 复验与 SGLang v0.5.20 serving 补证

本锚点已由 Artificial Analysis canonical `deepseek-v4-1-flash` 确认；本轮没有从 runtime 仓库发现新模型。用户终端通过 `10.24.27.134:7890` 访问百度成功；当前工作区经同一代理取得 AA `/zh`（1,747,860 bytes / SHA-256 `0933d776618e8530dccb05f28c0f3ce2398a3184552c8c988df9a60e6da5d0c5`）、DeepSeek V4.1-Flash 详情（3,969,971 bytes / `630f1c9df1abd6ce900d6a7016daf1038d5482c349fd5f4f2a419973ef790a91`）和 DataCurve（268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`）。详情页首请求遇 TLS EOF，重试后 HTTP 200；AA 路由集合与同日快照一致，无新增重点厂商 canonical。DataCurve 有 `mini_swe_agent_deepseek_v4_flash_max`，但没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，不迁移分数。

7890 取得 vLLM release Atom（734,377 bytes / `9933c44ff0413dce0b7565f5df9f4046eebea4fb121b95efc19ca94fd245933e`）：最新项是 2026-09-29 `v0.31.0rc1` 的 CUDA 12 镜像 CI 变更；稳定版仍是 `v0.30.0`。SGLang release Atom（1,058,795 bytes / `9c9c5e9bcc5fc55fea6be1d8e9c77fd84d330522638083a80e74f2134ddd5eb2`）最新稳定版为 `v0.5.20`（2026-09-18）。本轮只沿 V4.1 锚点读取 v0.5.20 中 DSV4 相关 PR：#39116 的 HIP/ROCm DSpark `swa_loc` 捕获缓冲区原位更新与 verify metadata 图内构造；#38192 的 unified-KV 下 per-request SWA ring 与容量记账（release note 报告 full-attention KV token capacity +83.6%）；#37764 将 AMD FP4 indexer prefill schedule preamble 从约 27 个小 Torch ops 收敛为 fused prep + CTA-info 两次 dispatch（release note 报告 concurrency 4 下输出吞吐 +15.3%）。均为上游源码/PR 或发布方指定负载结果，不是本机运行或独立复现。

边界：#39116/#38192/#37764 是 DeepSeek-V4 家族的 ROCm/HIP serving 路径，不能外推为 V4.1 专属架构或全平台性能；SGLang v0.5.20 含这些实现也不等于本机安装、完整权重加载或生产 acceptance。release feed 中另有 DSA cooperative top-k PR #37591，但其 patch comments/tests 明确举例 DeepSeek V3.2 与 GLM-5.2，不作为 V4.1 证据。细节与源码哈希见 `deepseek-v4.1-flash-source-notes.md`、第二十一册第 81.20 节和 source-index。QA 已通过：`git diff --check` clean，相关围栏和关键本地引用检查通过；goal 继续 `active`。

## 2026-09-29 Grok 4.20 Multi-agent：Beta API 边界与全量成本账本

按两榜已留存的重点 canonical `grok-4-20` 继续推进，没有从 xAI 文档另发现模型。通过 `10.24.27.134:7890` 取得 Grok 4.20 模型页 Markdown（1,564 bytes / `99057608aeb1a4e6ace5d87bf6c4b27b3bff4af5248babad475500da31fdc2e3`，与既有快照一致）、Multi Agent 专页（19,603 bytes / `0463d9d2022fd9453ece718e3fc898103c9cab20dc8cd4a2601d7c92b0fab6d4`）、Context Compaction（12,281 bytes / `466906990c559aeb84c16f785764f47ba9c735d7d7c821ca6ebd00511f33a938`）及 Release Notes（18,182 bytes / `ff4c876fbf0707ee7439b06357af40f5acc1103b4e7a9b264be7909f8a8753c7`）。专页把多 Agent 标为 beta；支持 xAI SDK/Responses、内置工具、Remote MCP 和 `previous_response_id`，不支持 Chat Completions、client-side/custom function tools 或 `max_tokens`。leader 与子 Agent 的全部 input/output/reasoning token、所有 server-side tool calls 均计费，可从 `usage` 与 `server_side_tool_usage` 观测；子 Agent 状态默认不可读，需 opt-in encrypted continuation。

已同步 [Grok 4.20 研究笔记](research/model-update-2026-09/grok-4.20-source-notes.md)、第二十册第 19.30 节、source-index、model-inventory、INTERVIEW_BANK、EXERCISES 与本计划/进度。状态为 **AA 单榜内容专题闭环（runtime/API 与 harness）**；不声称 Grok 架构、训练 recipe、真实 API probe 或生产验收已完成。`git diff --check`、新增本地引用、标题唯一性与相关 Markdown 围栏检查通过后，此活动锚点收口；整体 goal 保持 `active`，后续仍从两榜现有八家重点厂商 canonical 队列选公开资料缺口。

## 2026-09-29 Kimi K3：adaptive DSpark 与 ROCm serving 补证

7890 代理由用户提供的百度请求结果确认可达；本轮继续沿 Artificial Analysis/DataCurve 已发现的 `Kimi K3` 锚点，没有从 vLLM、SGLang 或 InferenceX 新增模型候选。两榜复验快照显示 AA `/zh` 为 1,746,726 bytes、60 个模型路由，DataCurve 为 268,036 bytes、70 个配置 ID；K3 仍有对应路由与精确 `mini_swe_agent_kimi_k3_max` 行。

本轮补齐 vLLM K3 recipe 的固定 2026-09-26 commit、adaptive variable-length DSpark graph/offset 实现、9 月 28 日 context-pointer invalidation 修复、ROCm SiTUv2/AITER 版本与布局门禁，以及 InferenceX 真实 block verification 与 synthetic acceptance sweep 的证据区别。研究笔记、第二十一册第 88 章、第二十四册第 32 章、source-index、model-inventory、榜单解释、面试题和练习已同步。

状态为**内容专题闭环 + stable/main/recipe source evidence**，不是硬件验收闭环：没有下载/加载完整权重、安装 nightly、运行目标 GPU、测真实线上 acceptance 或独立复现 benchmark。完成文档 QA 后仍沿两个指定排行榜的八家重点厂商 canonical 队列挑选下一知识缺口；目标保持 `active`。

文档收尾检查已通过：新增 K3 标题唯一，交叉引用目标存在；相关研究笔记、索引、清单、题库、练习和两处教材代码围栏均成对；`git diff --check` 无错误。整体 goal 继续保持 `active`。

## 2026-09-29 Kimi K2.7 Code：KTransformers 文档兼容性与 SFT 精度边界

本轮继续使用两榜已经发现的 Kimi K2.7 Code，不从 serving 文档或 runtime 仓库新增模型。当前 Kimi 固定 revision 部署指南给出 KTransformers RAWINT4 CPU/GPU 异构 serving 命令，并称 K2.7 与 K2.5/K2.6 架构相同、部署方法可复用；固定 KTransformers commit `c40722bf04c494f2492b7eb9e86ef01a4ede45b3` 的通用 Native Precision 矩阵却只列 Kimi-K2-Thinking。将其标为文档覆盖不一致、K2.7 兼容性未实测，不作支持/不支持二元结论。

本轮已将 dual prefill 阈值、NUMA/threadpool/GPU expert 参数含义和 K2.7 命令未显式启用 dynamic expert update 分层整理。另复核发现 K2.5 旧 source-install SFT 教程的 BF16 转换步骤不适用于 `0.7.0.post4`；固定 release tutorial 使用原始 K2.5 权重，配套 YAML 同列 RAWINT4 权重后端与 `bf16:true`，并披露 attention/expert LoRA、可恢复 checkpoint 与转成 SGLang adapter 的步骤。这是 K2.5 的发行配方，不是 K2.7 训练合同；K2.7 SFT 吞吐仍只是发布方场景数据，不是本地复现。

待办仅在可获得对应硬件/权重时升级：固定 K2.7 与 runtime revision，做完整权重加载和数值正确性测试；以实际 NUMA/CPU/GPU 配置 sweep `kt-cpuinfer`、GPU expert placement、prefill threshold 和 chunk size；分别 profile hybrid/layerwise prefill 的峰值 VRAM、TTFT/吞吐，并核对 K2.7 LoRA SFT 的权重格式/转换与有效 token 口径。未做这些验收前保持当前“内容专题闭环 + 固定文档证据”，goal 继续 active。

## 2026-09-29 GLM-5.3：Straw 异步队列、联合 checkpoint 与 archive

7890 当前可访问百度、Z.ai GLM-5.3 官方文档及 GitHub Atom，均 HTTP 200。继续从既有 GLM-5.3 榜单锚点出发；Z.ai 博客引用 slime，但 9 月 28–29 日的源码提交只作为通用关联框架更新，不作为 GLM-5.3 原始训练实现证据。

已核对 slime `#2410` 的 distributed fully async + Straw、`#2427` 的 joint model/queue checkpoint marker、历史 step branch rollback、immutable payload sharing、indexed archive 与 online-GC ownership。研究笔记、第二十册第 20.23 节、面试题、练习、source-index 和 model inventory 已同步；固定代码/test/patch 哈希及证据边界见 source index。

本机没有执行上游 Qwen2.5-0.5B 四卡集成测试；没有声称原子文件系统事务、所有权重 shard 的完整 hash 校验、GLM-5.3 专属 compaction 或生产验收。下一步从复验后的 Artificial Analysis/DataCurve 重点厂商 canonical 队列选择下一个知识缺口；目标仍保持 `active`。

## 2026-09-29：7890 会话可达性差异与榜单复验

用户 shell 提供的 `10.24.27.134:7890` 百度成功结果有效；但本会话显式经该代理访问百度时连接超时，在获批沙箱外重试也相同。本会话改用备用 `10.237.126.170:1234`，成功读取百度、Artificial Analysis `/zh` 和规范 DataCurve DeepSWE。两榜与本日较早快照逐字节一致（AA 60 个 canonical 路由，DataCurve 70 个配置 ID），没有新增重点厂商模型，因此不从排行榜外扩展候选。

沿既有 Kimi K3 锚点复核 vLLM PR #53954：当前 GitHub API 仍显示 `open`、未合并，patch 与 K3 研究笔记中的 7,242-byte 快照哈希一致。补丁所述 AITER/vLLM 两个 SiTU A8W4 环境开关若不一致会令 Kimi-K3 ROCm MoE layout 错配并静默退化；修复仍是开放 PR，不计作 stable/runtime 验收，也不归因给 DeepSeek V4.1。细节保留在 Kimi K3 source notes，本轮仅同步代理/榜单状态到 `progress_v2.md`。后续继续从两榜既有八家重点厂商 canonical 队列选取未闭环的权威技术缺口，整体 goal 保持 `active`。

## 2026-09-29 DeepSeek V4.1 锚点：SGLang main mixed-precision MegaMoE

两榜仍是当日确认的 60 个 AA 模型路由和 70 个 DataCurve 配置 ID，没有新 canonical。继续沿现有 AA `deepseek-v4-1-flash` 核查 SGLang `deepseek_v4.py` history，发现 PR [#39313](https://github.com/sgl-project/sglang/pull/39313) 于 `2026-09-29T09:25:11Z` 合入 main，commit `0e586fd12d63f06306ec20beb637bfeb331f1088`。PR patch、API、history 和 checks 的 bytes/SHA-256 见 [`source-index.md`](research/model-update-2026-09/source-index.md) 与 DeepSeek V4.1 source notes。

技术增量：guarded SM100/MegaMoE 路径将 block-FP8 shared expert 与 MXFP4 routed experts 送入 DeepGEMM FP8×FP4 kernel；scale layout 在 pre-dispatch 构造；为避免朴素融合破坏原有 shared/routed stream overlap，最终实现于 graph capture 中 fork 整段 fused region。PR 发布方在 4×B300、DeepSeek-V4-Flash-0731 上报告 accuracy 与 batch-swept throughput，regular batch 64 overall throughput略低于 baseline。该证据属于 V4-family SGLang main，不是 V4.1 benchmark，也不等于 stable：当前 SGLang Atom 仍指向 `v0.5.20`，而 PR 于之后的 9 月 29 日合入。检查快照有失败/跳过项，没有本机 GPU 复现或完整 acceptance。

已同步第二十一册第 81.21 节、研究笔记、面试题、EXERCISES、source-index、model-inventory、BOOK_SERIES、plan 和 progress。状态仍是 **V4.1 内容专题闭环 + family-level main runtime source evidence**；完整权重、目标硬件正确性、V4.1-specific FP4 质量/recall、DSpark acceptance、生产 profiling 和 SLO 未验证。下一项回到刷新后的两榜八家重点厂商 canonical 队列选择，不由 runtime PR 另增模型，goal 保持 `active`。

## 2026-09-29 DeepSeek V4.1 锚点：SGLang main 混合精度 MegaMoE 增量

- 在此前两榜刷新后，继续沿 AA 已有 `deepseek-v4-1-flash` 锚点查 SGLang `deepseek_v4.py` history；30 条提交中最新 `0e586fd12d63`（2026-09-29T09:25:10Z）为 PR #39313。未从 runtime 仓库发现新模型。
- PR API（45,675 bytes / SHA-256 `fbe33b6da871321cf7445d948086d7ce9456a3e211865ef7e4ddfe953db00970`）显示 #39313 于 `2026-09-29T09:25:11Z` 合入 main，merge commit `0e586fd12d63f06306ec20beb637bfeb331f1088`；patch 35,120 bytes / `2ceee8fb5122d9e571a1608416e5699d824ee703b4ad641e9ab50f8fc3192e6b`。机制是为 block-FP8 shared expert + MXFP4 routed experts 增加 guarded DeepGEMM FP8×FP4 MegaMoE 路径；朴素融合会丢失 alternate-stream overlap，最终代码在 CUDA-graph capture 下 fork 整段 fused region。PR 历史与完整哈希、fallback/门禁见 [`deepseek-v4.1-flash-source-notes.md`](research/model-update-2026-09/deepseek-v4.1-flash-source-notes.md)。
- PR 自报 4×B300（SM103）、DeepSeek-V4-Flash-0731、sgl-deep-gemm 0.1.7 结果：固定 1,024 输入/256 输出 token 下，DSpark overall tokens/s baseline→fusion+fork 的 batch 1/8/32/64 为 3,052.04→3,125.89、19,255.64→19,624.04、53,236.74→54,022.42、76,347.80→81,739.39；regular 为 737.25→748.13、5,147.25→5,252.39、16,123.12→16,245.61、27,562.54→27,506.76。数据有 workload 依赖；regular batch 64 略降。Greedy `0.899→0.897`、sampled pass@1 `0.900→0.910`，并有 per-request answer flips；不作显著性/普遍质量结论，绝不迁移成 V4.1 成绩。
- Fresh SGLang release Atom 仍为 v0.5.20（1,058,795 bytes / `9c9c5e9bcc5fc55fea6be1d8e9c77fd84d330522638083a80e74f2134ddd5eb2`），不含 9 月 29 日 main merge。Checks API 总数 171、默认返回 30 项中 15 success/9 skipped/6 failure；PR 的 Base/Extra/AMD aggregate 标记未全绿，不声称 CI 全通过。没有本机测试或 GPU benchmark。
- 已同步研究笔记、第二十一册第 81.21 节、EXERCISES、INTERVIEW_BANK、source-index、model-inventory、BOOK_SERIES 与 `plan.md`；内容状态为既有 V4.1 专题的 family-level main runtime 补证，仍非 stable/硬件/生产验收闭环；goal 保持 `active`。

## 2026-09-29：本会话代理差异、两榜复验与 Kimi K3 PR 状态

- 用户提供的 shell 输出证明其执行环境经 `10.24.27.134:7890` 访问百度返回 HTML；本会话显式 `curl --proxy` 访问同一代理时连接超时（curl 28，10 秒），按权限在沙箱外重试仍超时。因此只记录为本会话当前网络路径不可达，不推断代理全局失效。
- 备用 `10.237.126.170:1234` 本会话访问百度 HTTP 200（2,381 bytes），Artificial Analysis `/zh` HTTP 200（1,747,602 bytes / SHA-256 `970aaacc0dedaf2a624b5ede1e07fa7f6659da78f8b8d2a0dd451b67da3b3120`），规范 DataCurve DeepSWE HTTP 200（268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`）。两榜与本日 20:19 留存快照逐字节一致；AA 仍为 60 个 `/models/<slug>` 路由，DataCurve 为 70 个 `mini_swe_agent_*` 配置 ID，没有新的重点厂商 canonical。
- 通过 `1234` 重新取得 vLLM PR #53954 patch（7,242 bytes / SHA-256 `2197d91d91a9bf4cf0292bd1b591d2c2269aac956882123dfac5be2ec44342d7`），与 [Kimi K3 研究笔记](research/model-update-2026-09/kimi-k3-source-notes.md) 已记录快照一致。新鲜 GitHub PR API 响应（25,527 bytes / `a206d5c20f727962b00898bd17d47d8f81142cf3ac19a73faf3bb8ba4afe3a61`）显示标题为 `[Bugfix][ROCm] Derive a8w4 SiTU MoE layout from AITER_SITUV2_A8W4 too`、状态 `open`、`merged_at=null`，最后更新 `2026-09-15T05:52:57Z`；补丁明确针对 Kimi-K3，修复 AITER 与 vLLM layout gate 不一致导致的静默退化输出，不属于 DeepSeek V4.1 证据。技术结论已在 K3 笔记中，不重复新增书稿内容。
- 整体 goal 保持 `active`；下一项仍从两个排行榜已确认的八家重点厂商 canonical 中选择有权威材料支撑的未闭环知识缺口。

## 2026-09-29：7890 当前工作区复验与 Gemini Interactions schema 更新检查

- 用户贴出的百度页面确认其 shell 经 `10.24.27.134:7890` 可用；本工作区也用该代理取得百度 HTTP 200（2,381 bytes）、Artificial Analysis `/zh` HTTP 200（1,747,602 bytes / SHA-256 `970aaacc0dedaf2a624b5ede1e07fa7f6659da78f8b8d2a0dd451b67da3b3120`）、规范 DataCurve HTTP 200（268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`）和 Google Interactions 文档 HTTP 200。AA 同日对照快照为 1,747,864 bytes / `b2ec54e7332d68b569c698a7bc427932b264d286aa9689bd40d7550f4b824dfc`；两份快照的完整 `/models/<slug>` 路由集合均为 60 项且无差异。DataCurve 与此前快照逐字节一致，70 个 `mini_swe_agent_*` 配置 ID 无变化。此前试探的 `datacurve.ai/benchmarks/deepswe` 仍是错误路径并返回 404，不代表规范站点不可达。
- 本轮新鲜 Google 页面：Interactions overview 154,408 bytes / `d96ea69f069954e120811027a5b43117cd748192dc7ea40f30c74fee603bc71f`；API reference 739,127 / `6f0aad30591ec004c4de404f0746762a6bf2047b91f1b5c22bbcdadfd73347a6`；Tool combination 136,835 / `84e1ae87bfa171972f30cbdcbd73f526f4af14b3e4fa085503cc14c3512c2d30`；Thinking 286,232 / `079e06dc8cf69c42ee3ab6746b9e819f7a6999e713c72e692812f0aed2f6e554`。文档冲突仍在：Thinking 对 Interactions 标准自定义 function 与 thought signature 的窄口径，与 Tool combination 的宽口径不一致；API reference 仍未在 custom function schema 声明 signature，并把 `ThoughtStep.signature` 标为 optional。
- Google Gen AI Python SDK main Atom feed 最新项为 2026-09-29T06:17:59Z commit `b6535e822a6a984bec43056551b2788cd61509f0`（GCS tuning-job metrics URI）。该提交下 Interactions `FunctionCallStep`、`FunctionResultStep`、`ThoughtStep`、Google Search call step、Step union 和 `_gaos` BaseModel 的原始文件均成功经 7890 取得；六个文件哈希与研究笔记 §18 已记录值完全相同，schema 没有变化。没有 API key/授权 endpoint，未做真实 Gemini API probe，不把文档或 SDK schema 推断成服务端行为。
- 两榜本轮没有新 canonical。Gemini 条目是既有锚点的协议文档复验；最近已收口的排行榜锚点仍是 Kimi K2.7 Code，下一项仍须从刷新后的两榜八家重点厂商 canonical 中选择。总体 goal 保持 `active`。

## 2026-09-29：7890 外网代理验证与 Grok 4.20 Multi-agent 合同补齐

- 本工作区沙箱内经 `10.24.27.134:7890` 请求百度及 SGLang feed 均连接失败（curl error 7）；按授权在沙箱外重试官方 SGLang release Atom 后 HTTP 200，1,058,795 bytes / SHA-256 `9c9c5e9bcc5fc55fea6be1d8e9c77fd84d330522638083a80e74f2134ddd5eb2`，与已存快照一致，最新 stable 仍为 `v0.5.20`。因此本轮失败是当前 sandbox 网络路径限制，不能据此判定用户代理全局不可用；用户提供的百度成功结果与获批沙箱外成功请求均保留。
- DeepSeek V4.1-Flash 的 release/runtime 阶段未出现新的 SGLang stable；权重、目标硬件、真实 recall/FP4、DSpark acceptance 等仍是需后续 artifact/hardware 才能验证的工程门禁。本轮转到两榜已存在的 Grok 4.20 canonical，只沿官方 API/runtime 文档扩展，不从文档新增模型。
- 获批经同一 7890 代理复取 xAI 官方 Grok 4.20 Markdown（1,564 bytes / `99057608aeb1a4e6ace5d87bf6c4b27b3bff4af5248babad475500da31fdc2e3`，与此前一致）、Multi Agent（19,603 bytes / `0463d9d2022fd9453ece718e3fc898103c9cab20dc8cd4a2601d7c92b0fab6d4`）、Context Compaction（12,281 bytes / `466906990c559aeb84c16f785764f47ba9c735d7d7c821ca6ebd00511f33a938`）和 Release Notes（18,182 bytes / `ff4c876fbf0707ee7439b06357af40f5acc1103b4e7a9b264be7909f8a8753c7`）。Multi Agent 专页明确 beta、Responses/xAI SDK 支持面、Chat Completions/client-side tools/`max_tokens` 限制、`previous_response_id`、opaque 子 Agent state，以及 leader+workers tokens 和 server-side tools 全量计费。
- 已补充 Grok 4.20 研究笔记、第二十册第 19.30 节、来源索引、模型盘点、题库、练习、`plan.md` 和本进度。新增的是 Agent API contract/成本与审计知识，不是模型架构或训练披露；未使用 API key、未调用付费 endpoint、未下载权重。Grok 4.20 状态更新为 **AA 单榜内容专题闭环（runtime/API 与 harness）**，goal 保持 `active`。

## 2026-09-29：7890 代理复验与 Interactions 超时文档补取

- 用户终端的百度结果已在工作区扩展复验：`10.24.27.134:7890` 访问 Artificial Analysis `/zh` HTTP 200（1,746,541 bytes / SHA-256 `e0b467739db28960b4736aaee3e0e27951eda5d4794843d7d53f1284b32ca2e9`）；规范 DataCurve `https://deepswe.datacurve.ai/` HTTP 200（268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`）。错误路径 `https://datacurve.ai/benchmarks/deepswe` 返回 404，是路径/主机不正确，不是代理网络不通。
- AA 与当日先前留存页面相比，60 个唯一 `/models/<slug>` 路由无新增/删除；DataCurve 内容哈希相同，70 个唯一 `mini_swe_agent_*` 配置无变化。没有新增八家重点厂商 canonical；DataCurve 仍无精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行。
- 成功重试先前记录超时的 Google 官方页面：Interactions overview Markdown `154,404` bytes / `73047c07e2547f6ccd1817098c59ded2c2dfc4d1e968ae24e490bc127a9960ea`；API reference Markdown `739,127` / `ba1c19a6868a16d29c66008a34b27f66fb5babcbdbeeba00f3b33ebb48a857b8`；Tool combination `136,839` / `f4f2645b78da6388ce20f2814107f5475c4dcf11373c7f33f98225afa0b8426e`；Thinking `286,232` / `68cde323e37c84ab63aa66d02c22cd3f5d2bb028d8c4185beef3102263b19a22`。旧失败是当时具体请求/线路状态，不代表网页不存在。
- 新鲜文档仍有签名范围冲突：Thinking prose 与 API schema 对 thought signature 必需性的说法不同；Tool combination prose 对自定义 function call/result 的宽泛签名声明与 Thinking 窄口径及 API reference schema 不一致。只记录正式文档/schema 边界，不推断真实 Gemini endpoint 行为；无 API key/授权，未发起真实请求。详见 [`gemini-3.8-flash-source-notes.md`](research/model-update-2026-09/gemini-3.8-flash-source-notes.md) §21。此为既有锚点补充，不改变当前 DeepSeek V4.1-Flash 活动项；goal 保持 `active`。
- 同一代理还成功补取 Kimi K2.7 Code 固定 HF revision `74797c9c62378b951a1f6fcf5c4631024e9b8bef`：README 15,061 bytes / `c78bbe5af19636b180eda956abf1f30a79027ec04d6642ebddc44edbc4470879`、config 5,420 bytes / `ffbb57bff844e024f6c112640b63da80af228ceb8fb3e2e21908a21febbcdffb`，均与旧快照逐字节相同；固定部署指南 HTTP 200，3,689 bytes / `b1bc4c5fb7c8b1da727663d4da83999afc902f6166ff862fbaef514b86de2c65`，也与工作区留存的 2026-09-28 文件逐字节相同。复核的指南记录 KTransformers+SGLang RAWINT4 CPU/GPU 专家分工、8×L20+2×Intel 6454S 48-way 发布方 serving 结果，以及 2×4090+Intel 8488C LoRA SFT 发布方结果。原文 `1.97T RAM` 单位未解释，且两类 tokens/s 不可直接比较。已同步第二十一册第 94 章、研究笔记、INTERVIEW_BANK、model-inventory、source-index；未加载权重或复现硬件吞吐。

## 2026-09-29 Claude Sonnet 5.5：7890 恢复后继续联网研究

- 用户终端通过 `10.24.27.134:7890` 访问百度成功；当前工作区也经该代理获取 Artificial Analysis 中文首页（1,747,864 bytes / `b2ec54e7332d68b569c698a7bc427932b264d286aa9689bd40d7550f4b824dfc`）、Sonnet 5.5 详情（3,894,403 bytes / `fb64dfeaf034616ea4a5c31573fdad771584b4ed9b4aafbf3e7225fbe59174c0`）和 DataCurve（268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`）。AA 出现 canonical `claude-sonnet-5-5`，releaseDate `2026-09-28`，max + Default Fallback 与 low/medium/high/xhigh 配置；DataCurve 无精确 `mini_swe_agent_claude_sonnet_5_5_*` 行，不迁移其他 Claude 结果。
- 经 7890 获取 Anthropic 官方 overview、What's New、migration guide、prompting guide 与发布页；System Card PDF 13,101,289 bytes / SHA-256 `89cfd6a6d3aa6148eaf23f4a53006950ee7746009f381237c62920c720725785`，使用仓库 `pdf_text_extract.js` 解析为 148 页、223,920 bytes / `7b858f699d0361a9aab979caf872eb75774976a2f4946962362ca469392ca483`。未视觉核验 PDF。获批沙箱外 arXiv 精确检索返回 `totalResults=0`。
- 新增 [Sonnet 5.5 研究笔记](research/model-update-2026-09/claude-sonnet-5.5-source-notes.md)、第二十册第 24 章与第八册第 16.22 节；同步 PAPERS、source-index、model-inventory、榜单解释、BOOK_SERIES、知识图谱、面试题、练习和术语表。核心新增点是 `between_tools` 协议约束、三类 thinking block binding、forced-tool/computer-use breaking changes、拒答类别与 fallback 路由，以及 benchmark 中实际执行模型归因。
- System Card 结果按 publisher/harness 证据记录：Terminal-Bench 4.0 `70.6%`、OSWorld 2.1 `80.1% partial / 43.5% strict`，都绑定任务集、effort、工具、资源、fallback 和评分定义；没有独立复现。未调用 Messages API、未下载权重，也不推断参数/架构/完整训练 recipe。QA 已完成：`git diff --check` 通过；新笔记和章节的代码围栏成对，关键相对链接目标存在。之后已从两榜已确认 canonical 队列选择 DeepSeek V4.1-Flash 继续推进，goal 保持 `active`。

## 2026-09-29 DeepSeek V4.1-Flash：7890 当前可用与 SGLang serving 增量

- 用户终端的 `10.24.27.134:7890` 百度成功结果已在本轮复现到目标站点：Artificial Analysis `/zh` HTTP 200、1,747,860 bytes / SHA-256 `0933d776618e8530dccb05f28c0f3ce2398a3184552c8c988df9a60e6da5d0c5`；DeepSeek V4.1-Flash 详情首次 TLS EOF，重试 HTTP 200、3,969,971 bytes / `630f1c9df1abd6ce900d6a7016daf1038d5482c349fd5f4f2a419973ef790a91`；DataCurve HTTP 200、268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`。同日快照间 AA 模型路由集合无变化。
- 锚点身份仍是 AA `deepseek-v4-1-flash`。DataCurve 只有 `mini_swe_agent_deepseek_v4_flash_max`，没有精确 V4.1 配置名；不把 V4 Flash/Max 行迁移成 V4.1 成绩。用户本轮任务是联网/serving 来源补证，不从 SGLang/vLLM 发现模型。
- vLLM Atom feed 最新 `v0.31.0rc1`（2026-09-29T03:18:39Z）标题只涉及 CUDA 12 镜像 CI；稳定版仍为 `v0.30.0`。SGLang 官方 release Atom 的最新稳定版是 `v0.5.20`（2026-09-18）。当前工作区经 7890 获取两份 feed、AA 详情和 SGLang PR patches；大小、SHA-256 及边界见 [`deepseek-v4.1-flash-source-notes.md`](research/model-update-2026-09/deepseek-v4.1-flash-source-notes.md)。
- SGLang v0.5.20 的 V4 家族 AMD/HIP 增量：PR #39116 处理 DSpark 图 replay 中 `swa_loc` stale pointer、精确 token count/D2H sync 与 verify metadata；#38192 在 gated unified-KV 路径加入 request-owned SWA ring，release note 报告 full-attention KV token capacity +83.6%；#37764 融合 FP4 indexer prefill schedule prep，release note 报告 concurrency 4 下 output throughput +15.3%。这些是上游 patch 和发布方负载结果，不是本机 benchmark、V4.1 专属算法或生产验收。
- 边界检查发现同一 feed 的 cooperative exact top-k PR #37591 patch 明确举例 DeepSeek V3.2/GLM-5.2，故未归到 V4.1。新增第二十一册第 81.20 节、研究笔记、source-index、model-inventory、PAPERS、BOOK_SERIES 与面试题；没有改写权重/目标硬件/线上验收为已通过。QA 已通过：`git diff --check` 无错误；相关 Markdown 围栏成对、关键相对链接目标存在、新增章节/题目唯一，相关文件无尾随空白；goal 保持 `active`。

## 2026-09-29 DeepSeek V3.2：两榜复验与 Figures 1–7 视觉核验

首轮 QA：`git diff --check` 通过、19.36–19.38 heading 唯一、相关本地目标存在，所有受影响 Markdown 文件代码围栏均成对；后续 Figure 5/7 同步及 19.39/19.40 检查见本节后续记录。

- 用户终端的百度响应与本工作区复测相符：经 `10.24.27.134:7890`，当前工作区对百度和 arXiv Figure 2 曾分别取得 HTTP 200（百度 2,381 bytes；Figure 2 SVG 269,427 bytes）。随后的另一次代理连接短暂失败；按审批流程重试后取回 Figure 2，SHA-256 `1e6bc6c61ea26ae2b8528edb1eab14874832470fd38f115fe9b8feb615facb9a`。这是时点性连通记录，不代表代理持续稳定。
- Figure 2 已渲染为 1600×833 并视觉复核。Lightning Indexer 产生索引分数，Top-k Selector 选择候选 KV entries，随后候选与主 query 进入 MLA Core Attention 计算输出；图注、§2.1 与 Eq. (2) 共同支持。没有把分数/候选选择写成 attention 输出或 production-kernel 验收。
- Figure 3 的 `cost_prefilling.svg`（33,398 bytes / `b045c26eb19d92325de7e86aabec905a9ac8bdb6729db94e04c8701693b19705`）与 `cost_decoding.svg`（31,868 bytes / `5894e01515f7f9e9ef9045ae8e30a092f5cc1e4229936c493c66c4748127c61d`）经代理获取，均渲染为 1600×1200 并视觉核验。Token Position 越长，V3.2 曲线成本增长更缓；极短位置存在交叉。报告把曲线限定为 H800 部署服务、按 `$2/GPU-hour` 估价的发布方 benchmark，而不是 API 定价或独立复现。
- Figure 4 原始 JPEG 为 76,981 bytes / SHA-256 `58623875cc487b3cbbd60955c14801c07d5f526d2b2e875795e3bf5ace511733`，1280×671，已视觉核验。图示与 §3.2.1 一致：同一轮追加 tool messages 后保留 thinking；下一条 user message 到来后清除旧 thinking，但保留工具调用/结果与前轮 answer。该行为取决于 message role/history replay，不是永久记忆。
- Artificial Analysis `/zh` 快照为 1,694,614 bytes / SHA-256 `04f62f3f39c1f65c3c1d8dd564782f9d7d20fb506fa671ee75fcfaf4e9482f9f`；唯一路由数 55→62、无移除。新增七条中四条（GLM-5.3-Flash、Kimi K3、Qwen3.8 2.4T A95B、Qwen3.8 27B）已是重点 canonical，另三条不在八家范围。DataCurve 为 268,036 bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，70 个唯一配置 ID 无变化；没有新增重点锚点，也没有精确 V3.2 Agent 行。
- 已同步研究笔记、第二十一册第 19 章 19.36–19.39、来源索引、模型清单、PAPERS、INTERVIEW_BANK、计划与进度。Figure 2、3、4、5、6 已视觉核验，其他论文图表/公式仍未全部核对；production kernel、indexer recall/误差曲线、目标硬件 profiling、线上 acceptance、独立 benchmark 和完整 RL recipe 仍待核验。下一步继续从两榜既有八家重点 canonical 队列选择知识缺口，goal 保持 `active`。

Figure 5 补充记录：用户终端提供的百度 HTML 显示 7890 可用；当前工作区用同一 `10.24.27.134:7890` 请求 Google 也得到 HTTP 200、84,771 bytes。随后经该代理取得 arXiv v1 `synthesis-rl-plot.png`，原图 1432×1024、222,137 bytes / SHA-256 `2107344cc2002f51bc1922df0bdcabd7afa74bb0f6cb5d3aa996f33e3f0d7cc3`。视觉核验其图例、分面和总体训练走势后，补入 19.39：Figure 5 是 synthetic general-agent data 上的发布方 RL 消融，不能当最终模型独立 benchmark，也不能证明真实环境泛化因果、数据无污染或完整 recipe 已公开。

同步与 QA：已补研究笔记、第二十一册 19.39、source-index、model-inventory、PAPERS、INTERVIEW_BANK、plan。Figure 7 的猜测 URL `/figures/figure7.svg` 返回 HTTP 404（7,901 bytes），仅表示文件路径猜错/不存在，不据此认定无法访问或论文没有该图。`git diff --check`、19.39 唯一性、目标链接和相关 Markdown 围栏检查随后执行；goal 保持 `active`。

## 2026-09-29 DeepSeek V3.2：Figure 7 MLA-MHA/MQA 视觉复核

- 通过 7890 访问 arXiv v1 HTML 后，从 Appendix A 对应对象定位正确资源：`MLA-MHA.svg` HTTP 200、244,227 bytes / SHA-256 `afd21a50bbaef58314036862cb6ce44dca81a9d42a414c0969074fbf954b32b4`；`MLA-MQA.svg` HTTP 200、224,311 bytes / SHA-256 `c6aa4f3e2ea222d2a75ef000dca2f9e8c48a820116957c6a741cbd0f253d77c3`。两张图均渲染并视觉复核。先前 `/figures/figure7.svg` 的 404 是资源名错误，不是网络中断。
- Appendix A 和正文说明，MHA mode 将共享 latent KV 投影成各 query head 的 K/V；MQA mode 让 query heads 共享 latent KV entry 与位置 key。图注把 V3.1-Terminus 的 training/prefill 使用 MHA、decode 使用 MQA 明确绑定于该版本；V3.2-Exp inference demo 对应路径是独立实现证据，不能泛化成所有 V3.2 backend 的强制合同。图不独立证明具体 cache bytes、吞吐、kernel 覆盖或质量等价。
- 已同步第二十一册第 19 章 19.40、研究笔记、source-index、model-inventory、PAPERS、INTERVIEW_BANK 与 plan。当前 Figures 2–7 已视觉核验，其余图表/公式仍有待检查。最终 QA：`git diff --check` 通过；19.39 与 19.40 标题各唯一；本轮相关本地链接目标存在；所有受影响 Markdown 代码围栏成对。goal 保持 `active`。

## 2026-09-29 DeepSeek V3.2：Figure 1 benchmark protocol 视觉核验

- arXiv v1 `v32_performance.svg` 通过 7890 返回 HTTP 200，128,338 bytes / SHA-256 `14354740f4d54692b6af6323cc12d3a5f0e0f937bc2b7dd65021307bd7820973`；渲染后确认 Reasoning/Agentic 分组、benchmark/metric 标签、双纵轴和图例。Codeforces Rating 用右轴，其他 accuracy/Pass@1 用左轴。
- Figure caption 与 §4.1 确认 HMMT February 2025、HLE text-only；通用数学提示下 V3.2-Thinking 的 HLE bar 为 25.1，另用 HLE official template 为 23.9。Tool-use 用 thinking + standard function-call，MCP-Universe/MCP-Mark 使用作者内部环境；这些是发布方条件化评测，非榜单分数或独立复现。
- 已同步第二十一册 19.41、DeepSeek 研究笔记、source-index、model-inventory、PAPERS、INTERVIEW_BANK、plan 与本进度。至此 Figures 1–7 均完成视觉核验；公式后续核对见下一节。中间 QA：`git diff --check` 通过，19.39–19.41 标题唯一，相关链接目标存在、Markdown 围栏成对。

## 2026-09-29 DeepSeek V3.2：DSA Eq. (1)–(4) 原文文本审计

- 通过 `10.24.27.134:7890` 取得 arXiv v1 PDF HTTP 200，980,616 bytes / SHA-256 `2bec0671778769c159ec389412727d1f3d4889fe1c71564b61edaa24705bd17b`。本工作区没有 PDF 渲染/抽取器，GDK thumbnailer 也无法识别 PDF；因此本轮没有声称 PDF 页面视觉核验。
- 使用 arXiv v1 HTML `application/x-tex` annotations 逐项核对 Eq. (1)–(4)：Eq. (1) 是多头加权 ReLU index score（无 softmax，不是概率）；Eq. (2) 根据 Top-k latent KV 候选由主 attention 计算 `u_t`；Eq. (3) 以跨主 attention heads 聚合、序列维 L1-normalize 的 dense distribution 监督 indexer；Eq. (4) 将 KL 对齐限定在 `S_t`。正文还明确 dense warm-up freeze 主模型、只训 indexer；sparse stage detach indexer input，indexer 只由 `L_I` 更新、主模型只用 LM loss。
- Eq. (4) 展示 `p_{t,S_t}`，但报告这段没有明确写出选中子集后是否重新归一化。V3.2-Exp 官方仓库网页/raw README 通过 7890 分别 HTTP 200（295,427 / 6,899 bytes，README hash 与既有快照相同）；公开根目录列出 inference、报告、README、license、cost image，没有 trainer/loss 入口。GitHub API tree 单独 403；不将其扩大成“没有任何官方训练实现”的结论。该 repo 未解决 `p_{t,S_t}` 归一化，记为公开资料边界。
- 已同步第二十一册 19.42、研究笔记、source-index、model-inventory、PAPERS、INTERVIEW_BANK、plan。本轮标题 19.39–19.42 唯一；相关 Markdown 数学/代码 fence 配对与本地引用已检查，最终 `git diff --check` 待收尾执行，goal 仍 `active`。

## 2026-09-29 DeepSeek V3.2：GRPO Eq. (5)–(9) 原文文本审计

- 当前工作区经 10.24.27.134:7890 获取 arXiv v1 HTML，HTTP 200，295,170 bytes / SHA-256 5da74d488b218a45a995838b94feb95aacc63c03b806f8462496bfd4bceb07ef，与既有快照一致。
- Eq. (5)–(9) 的 application/x-tex annotations 已逐式文本核对。整篇 arXiv v1 HTML 共有的编号公式只有 (1)–(9)；重点澄清 group mean 与 response token mean 的嵌套口径；Eq. (6) advantage 展示减组均值而未写标准差归一化；Eq. (7) 外层 current/old ratio 与 KL 内层 reference/current ratio 不同；Eq. (8) mask 只作用于 clipped policy 项；Eq. (9) 只屏蔽负 advantage 且 response 平均 divergence 超阈值的样本。
- 补充 Keep Routing（复用 rollout MoE 路由）与 Keep Sampling Mask（复用 top-p/top-k 截断支持集），并明确这仍不是完整奖励、rollout、优化器或更新预算 recipe。PDF 页面公式视觉检查未完成；未编号行内数学表达式未做穷尽审计。
- 已新增第二十一册第 19 章 19.43，并同步 DeepSeek 研究笔记、source-index、model-inventory、PAPERS、面试题、练习、术语表与 plan。QA：git diff --check 通过；受影响 Markdown 代码围栏均成对；章节 display-math fence 数为 12（成对）；19.43 标题唯一；PAPERS 本地章节链接目标存在。整体 goal 保持 active。

## 2026-09-29 Kimi K3：adaptive DSpark 与 ROCm serving 补证

- 7890 当前可用的证据来自用户终端对 `http://www.baidu.com` 的 HTTP 成功结果；本轮没有把单站点连通性等同于所有 HTTPS 站点可达。按已有两榜证据继续 Kimi K3，没有从 serving 仓库发现新锚点。Artificial Analysis `/zh` 快照 1,746,726 bytes / SHA-256 `7a2d98fadd16d7c5b620ebd6a0a3b953e469b7a2bd10f5c21f0d5364d0a31bba`，60 个模型路由且含 `/models/kimi-k3`；DataCurve 为 268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，70 个配置 ID，仍有 `mini_swe_agent_kimi_k3_max`。榜单指标仍只解释为 provider observation 与 mini-swe-agent/tools/environment/verifier 系统结果。
- vLLM recipe 固定 commit `d71af9ad11aae587822a9a73a9c470c409c29ecc`（2026-09-26），32,538 bytes / SHA-256 `40d96d65ed07e88133ffa465f39f94b02aabc0438a331df0c8e7c069114ec257`，对应 MI355X ROCm 10 nightly/pre-release 路径。vLLM main adaptive DSpark commit `88aa0d287dd3abac9386e90ea1a63e6ed5d50580`（2026-09-23）的 patch 为 28,487 bytes / `7fdec0d0c8a050ff12f77ce6973549a486c59fc1ac8c8407db4108b7a10a56c4`：confidence head 产生逐位置接受概率，`k+1` capture 上界配合 offsets/masks 表达批内不同 verify 长度；row 分类依据 request 状态，不沿用可能被 scheduler 改写的 token count 等式。
- vLLM PR #58814 于 2026-09-28 合入，patch 1,416 bytes / SHA-256 `619dce6a98855a0508edc1568dd795136507125a918dd5f2f84bfbfe0dc6d933`，使 DSpark context pointer cache 在 cache buffer `data_ptr()` 变化后失效并重建，且不允许在 CUDA graph capture 中更新。ROCm SiTUv2 a4w4 需匹配 vLLM/AITER nightly 与 tuned config；PR #53954 检查时仍 Open，记录 kernel/weight layout gate 不一致可能静默退化。vLLM stable `v0.30.0` 不含这条 SiTUv2 recipe 路径；stable/source、main PR 与 pre-release recipe 分开记账。
- 固定 InferenceX MI355X AgentX lane 区分真实 block rejection 与 synthetic throughput sweep；synthetic 路径注入 golden acceptance length 并跳过 target verification，不能用于准确性、真实 acceptance-rate 或线上服务结论。本轮未下载权重、安装 wheel/nightly、运行 GPU、执行 cache recovery 或做独立 benchmark；SGLang stable `v0.5.20`、vLLM stable `v0.30.0` 与 `v0.31.0rc1` 的关系和源码哈希见研究笔记。
- 已同步 [`kimi-k3-source-notes.md`](research/model-update-2026-09/kimi-k3-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[第二十一册第 88 章](book-21-transformer-architecture-evolution/chapters/88-kimi-k3-kda-stable-latentmoe与百万token-agent.md)、[第二十四册第 32 章](book-24-llm-inference-engine/chapters/32-multi-turn-tool-use和agent-serving支持.md)、INTERVIEW_BANK、EXERCISES 与 plan。文档 QA 待本轮收尾；目标保持 `active`，下一步从两榜既有八家重点 canonical 队列选下一高价值缺口。

- QA 收尾已通过：本轮涉及的 10 个 Markdown 文件代码围栏均为偶数；新增活动标题唯一；K3 研究笔记、source-index、清单、解释文件和两处教材链接目标存在；`git diff --check` 无错误。没有运行权重、GPU 或生产 serving 测试。整体 goal 保持 `active`。

## 2026-09-29 Kimi K2.7 Code：7890 复验与 KTransformers 文档冲突核对

- 用户终端提供的 `www.baidu.com` 响应表明其 shell 的 7890 代理可用；本工作区本轮也经 `10.24.27.134:7890` 访问 Google 首页（HTTP 302，372 bytes），跟随跳转的定向搜索返回 HTTP 200（91,749 bytes），GitHub commit 页面返回 HTTP 200（283,724 bytes）。GitHub API tree 返回 403 rate-limit；两个初始猜测路径返回 404，修正到仓库 README 的真实文档路径后，固定 raw 文档均 HTTP 200。将端点限流/路径错误与代理网络状态分开记录。
- Kimi K2.7 Code 仍只沿 AA/DataCurve 既有 canonical 锚点推进；本轮未从 KTransformers 新发现模型。Kimi 固定 HF revision 的 deploy guidance 为 3,689 bytes / SHA-256 `b1bc4c5fb7c8b1da727663d4da83999afc902f6166ff862fbaef514b86de2c65`，称 K2.7 与 K2.5/K2.6 架构相同、部署方法可复用，并给出 KTransformers RAWINT4 命令。
- KTransformers main Atom 最新项为 2026-09-23 commit `c40722bf04c494f2492b7eb9e86ef01a4ede45b3`（21,972 bytes / `04d2f2e3ddfb18a5e03ebb93100569656db843f038eb4cad9eb275d67c0fca78`）。该固定提交下：K2.5 serving guide 5,536 bytes / `07f8d0c56ab503235c6b040da1a52a207d1cfb8104845593a3c45f7c49d18204`；Native Precision tutorial 9,597 bytes / `aed37372f057ef24d726cfb4c45010721afeb719156bcfc83fc9ab90b64aa93e`；expert scheduling tutorial 8,002 bytes / `f7714175e6d3d7f3c64a25a45f66d9533f47252fcffa1f89609ff107da365ced`；K2.5 SFT tutorial 6,858 bytes / `e6d792a8a7383340b24bc3675c7e94c31c29b42588e43a610f532561b511e6e0`。固定文档与同日快照逐字节一致。
- 证据结论：K2.7 发布指南存在 RAWINT4 serving 命令，但 KTransformers 同提交通用 Native Precision 支持矩阵只列 Kimi-K2-Thinking，`kt-cli` 示例名单也未列 K2.7/K2.5。记录为文档覆盖不一致、K2.7 兼容性未实测；不能因此判定支持或不支持。命令的 `kt-cpuinfer=96`、threadpool=2、每 MoE layer 的 GPU experts=30、prefill threshold=400、TP=4 和发布方 8×L20/48-way 数字均保留其硬件/文档口径。
- pinned tutorial 将 `< threshold` 定义为 CPU-GPU hybrid，将 `>= threshold` 定义为 layerwise prefill，并指出 layerwise 会把 CPU 权重传到 GPU、增加 VRAM；expert scheduling 文档称 GPU expert count 按 MoE layer 计，线程建议约 90% physical cores、threadpool count 对应 NUMA 节点。K2.7 示例未显式启用 dynamic expert update。另取回 K2.5 `0.7.0.post4` release tutorial（19,251 bytes / `c3345dccdccf43c3df8ad708a0e0644500bf00368aaef88ca17b85eb0df75d02`）、英文版（20,724 bytes / `8c2f5a5c5c05bf3313a3da5caaee3af8fa0a14574b7c672979387089105b1ff1`）与训练 YAML（1,660 bytes / `a42b713c57790afc192660e7419533d81781400f3f0bfa9fe9bb6b0639e7529a`）。旧 source-install 教程本身注明 post4 应使用新 release tutorial；新 YAML 使用原始 K2.5 权重、`kt_backend: RAWINT4` / expert `rawint4`，同时配置 `bf16: true`。该 K2.5 NekoQA recipe 还记录 attention/fused-expert LoRA、optimizer 等可恢复 checkpoint 与 SGLang adapter 转换流程。上述区别不能推断 K2.7 专属 SFT 配方或权重兼容性。
- 已同步 K2.7 研究笔记 §9、第二十一册第 94.14–94.15 节、source-index、model-inventory、inventory-interpretation、INTERVIEW_BANK、EXERCISES 和 plan。仍未下载/加载 K2.7 完整权重、安装目标 runtime、运行 GPU、做真实 API probe 或独立 benchmark；本轮只达到固定公开文档/source evidence。goal 保持 `active`，之后继续从两个指定排行榜的八家重点厂商锚点选择知识缺口。

## 2026-09-29 GLM-5.3：slime Straw async rollout、checkpoint rollback 与 indexed archive

- 工作区通过 `10.24.27.134:7890` 实测百度 HTTP 200（2,381 bytes）、Z.ai GLM-5.3 文档 HTTP 200（515,625 bytes）、GitHub slime Atom HTTP 200（16,120 bytes）。Atom 当前 hash `58ab5ae1ebf4cc3f5fc641d92798b229c338ff940121149470b5ddd891904999`，最新 main `8088a4b4450ac374e1439cf2310ba3add370997d` / #2427（9 月 29 日）；#2410 `68572249359201b42bd2020a584c370bc8b403e1`（9 月 28 日）引入 distributed fully async rollout + Straw。
- Z.ai GLM-5.3 博客引用 slime，slime current README 也列出 GLM-5.3；因此沿用 GLM-5.3 锚点研究其相关训练基础设施，但不从框架 repo 发现模型。9 月底提交晚于 GLM-5.3 的榜单 release date，不能据此声称模型原训练流程采用相同实现。
- #2427 的 `checkpoint.py` 在 actor/critic 与 rollout manager 保存后写联合 commit marker，包含 model 文件 size、部分 metadata SHA-256 和 queue/builder state SHA-256；optimizer/RNG 缺失时不能用于训练 resume。历史 step 恢复创建新 branch/queue ID，保留旧 branch，并可共享 Straw immutable payload；无 queue snapshot 时显式 empty recovery，不伪造精确 replay。
- `RolloutArchive` 用 sample/task key 查询 chunk，读取时剥离旧 queue lease/receipt；`.straw.json` 索引与数据 ownership 分离，reader close 后还需在无活跃 reader 时 release，所有 queue/checkpoint/archive 引用释放后 online GC 才能回收。`.pt` export 是自包含副本。Fully async 的 staleness 会参与排序，但不自动丢弃 stale sample。
- fork test 是 Qwen2.5-0.5B、4 GPU 的上游集成测试源码，设计覆盖 explicit/automatic rollback、queue cursor 与 ready sample 恢复、empty queue 以及 Straw/PT archive replay；本机未安装/运行对应训练栈，未执行 GPU 测试。源码没有证明 GLM-5.3 私有 compaction 或生产性能。
- 固定证据哈希：Straw guide `ae8cda5701cd90fc79c275745b3d59e81d35fb08c08fe00363d48b9762cb9a54`；archive `9f114885da58f970e00c1ed7ed4b8d4b2c809a80870ea18e0dfe2af8e3f46742`；checkpoint `38470782211793fde1a1633be9deb77fbf60516c3d64b2949b4c99aeff0de02e`；fork test `c79327f0a89764c6bb1ccc8f2061572a38d5ae10de1a8efae38b02caaff2c1dd`；#2427 patch `e14f0235af13fddd21dba39535e64c650219dd59c14641f03d4784a44a11fb2e`；#2410 patch `dff6b3688272d23fa6d6271160bbcfba082efa142f22a10b3cd8a43a383f46f8`。完整来源和 marker 校验边界见 GLM-5.3 研究笔记及 source index。
- 已同步 GLM-5.3 研究笔记、第二十册第 20.23 节、INTERVIEW_BANK、EXERCISES、source-index、model-inventory 和 plan。整体 goal 继续 `active`；下一步按两榜重点厂商 canonical 队列选下一知识缺口。

## 2026-09-29 当前时点榜单刷新重试

- 通过 `10.24.27.134:7890` 获取百度、Z.ai 官方文档和 GitHub slime Atom 仍为 HTTP 200；但对 Artificial Analysis `/zh` 与规范 DataCurve 主机的重试在连接阶段超时，沙箱内和获批沙箱外重试均为 curl HTTP `000`。这只说明本轮该代理到两个榜单主机的请求失败，不说明榜单不可访问或内容未更新。
- 最近成功的同日榜单基线沿用已记录快照：AA `1,747,602` bytes / SHA-256 `970aaacc0dedaf2a624b5ede1e07fa7f6659da78f8b8d2a0dd451b67da3b3120`，当时 60 个唯一路由；DataCurve `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，当时 70 个配置 ID。不要把旧快照的“无新增”外推到当前时点。
- 下一次选择模型锚点前先成功刷新两榜并对照 canonical 集合；当前没有据此新增或删除任何候选，整体 goal 保持 `active`。
- 2026-10-01 两榜与 Google 官方入口再次刷新：AA `/zh` HTTP 200，`1,930,483` bytes / SHA-256 `3a962c79cec028baf1e642d8e14fa20928c6b8470d2f70b3568b5fff7e559ee6`，仍有 `gemini-4-argon`；DataCurve HTTP 200，`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，70 个配置无精确 Argon 行。Google 模型目录 `155,241` bytes / `a0491f5f0b9f9fc107c42183fb6cf9d22927e1af03dbbdfa0ccc38e4e4ac31` 与 sitemap `253` bytes / `8dadac65fc82806836873c16fc1f813597e99b3e5f838d61c0e4435b8cfc8f7b` 均无 Argon。没有新的八家重点 canonical；Argon 继续保持官方资料缺口，goal 保持 `active`。
- 2026-10-01 扩展 Google 官方入口复查：Blog 搜索、Google AI Blog、Vertex AI 模型目录和 Gemini API 模型目录均 HTTP 200，但正文没有 Argon/Gemini 4；API/Vertex 页面当前可见 Gemini 3.x/3.8 条目。新增负检索哈希已写入 Argon 研究笔记与来源索引。该证据仍不足以形成模型闭环，goal 保持 `active`。
- 2026-10-01 重点厂商 canonical 集合审计：AA 当前相关路由为 Claude Fable/Sonnet/Opus 5.x、DeepSeek V4、Gemini 3.8/Argon、GLM-5.3、GPT-6.x、Grok 4.7、Kimi K3、Qwen3.8；DataCurve 当前 70 个配置的重点厂商行均已在 `model-inventory.md` 或对应研究笔记中登记。`gemini`、`grok-stt`、`qwen3-asr` 是聚合/语音入口，不新增主流文本模型锚点；`deepseek-v4-pro-non-reasoning` 与 Qwen3.8 Max 亦已有既有条目。没有新遗漏候选，Argon 仍是唯一官方资料缺口，goal 保持 `active`。
- 2026-10-01 Argon 详情与公开搜索复验：AA 详情页仍为 `Gemini 4 Argon (high)`，Google/Bing exact phrase 搜索未形成 Google 官方一手页面；详情和搜索快照哈希已写入研究笔记与来源索引。Argon 仍未闭环，goal 保持 `active`。
- 2026-10-01 Google 限定域名检索：对 `ai.google.dev`、`cloud.google.com`、`deepmind.google`、`blog.google` 和 `google.com` 的 exact/site 查询均未提取出官方 Argon URL；五个搜索快照哈希已写入研究笔记与来源索引。该结果进一步确认当前只有 AA/provider 发现证据，Argon 仍未闭环，goal 保持 `active`。
- 2026-10-01 Argon 官方发布出现：Google The Keyword 官方文章 `Introducing Gemini 4 Argon`（`408,929` bytes / `1c09a8019b06eada8703e67c7da2d4269ff3dc9169ae66226784dbec2d3043df`）已核验。文章披露 1M 输出上限、Fairwind 受限 rollout、coding/enterprise/cyber 场景、Google 自报 DeepSWE/AutomationBench/LVBench/CWE-bench 结果及分阶段安全措施。研究对象升级为“AA 单榜内容专题闭环（Google 发布方证据）”；公开模型卡、技术报告、权重、完整 API、独立复现和精确 DataCurve 行仍待核验。
- 2026-10-01 当前排行榜重点模型闭环审计完成：AA/DataCurve 重点 canonical 已全部在 inventory/研究笔记中登记；Argon 已补齐 Google 官方发布资料、正式章节、题库、练习、术语、论文索引、项目、知识图谱、计划和进度同步。剩余项目均属于参数/权重/硬件/独立 benchmark/生产 SLO 等工程验证边界，不属于本 goal 的资料级闭环门槛。按用户定义，当前 goal 达成。



## 2026-09-30 Kimi K3：DFlash draft / block diffusion 补证

两榜刷新后仍沿既有 `kimi-k3` 锚点推进。新增权威链为 DFlash v2 论文、HF 固定 revision/config/model card 与 SGLang PR #40794。重点面试点是 target 多层 hidden feature 的逐层 K/V 注入、block diffusion 的并行 draft、anchor/block sparse training mask、早位置 loss decay，以及 K3 layer-output capture 的索引语义。

同步范围已扩展至 K3 研究笔记、第二十一册第 88.26–88.27 节、PAPERS、INTERVIEW_BANK、EXERCISES、source-index、model-inventory、BOOK_SERIES、progress。状态为 **K3 内容专题闭环 + DFlash paper/HF/main source evidence**；未下载 draft 权重，SGLang stable 仍为 `v0.5.20`，PR benchmark 使用未公开 production draft，尚无本机 GPU、真实 acceptance、目标硬件和生产 SLO 验收。下一步先复核两榜 canonical 集合，再选择八家重点厂商中尚有官方技术缺口的锚点，goal 保持 `active`。
2026-10-01：Argon 官方资料已出现。Google The Keyword 公告形成发布方内容闭环；下一步只继续核验 Google 模型卡/API/技术报告/代码等更强资料，不把发布方评测升级为独立验收。正式章节为第二十一册第 95 章，配套题库、练习、术语、论文、项目、图谱和 inventory 已同步。

## 2026-09-30 Kimi K3：DFlash 论文、公开 draft 与 SGLang capture hook

- 通过 7890 沙箱外成功刷新两榜：Artificial Analysis `/zh` `1,752,755` bytes / SHA-256 `35e3197c2ad3b4259369bf55f72d912e03c0ecd61ad7d181db1fdda86f0dd527`；规范 DataCurve DeepSWE `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`。DataCurve 与旧快照一致；AA 为动态页面，未据字节变化推断模型 revision；没有新重点 canonical。
- 取得公开 [modal-labs/Kimi-K3-DFlash](https://huggingface.co/modal-labs/Kimi-K3-DFlash) metadata revision `c192d15a43407bf758b5ae0880d5c72052fef1de`（2,000 bytes / `48a5d60f2b729d742edcd072699c969bcd605f995f991e1543e3ce6d722a4c00`）、固定 README（4,058 / `6924c345a25dc3360ca0a554469d59849378570eb6bfc8ce00189c239ad95235`）和 config（1,215 / `92e2928e57f417921cd1c031a18840834c55ed13ed0d722acfa8f41b01080717`）。模型卡明确它是约 2.6B BF16 的 K3 draft-only 配套模型，generic data mix 不含 tool calls/agentic traces；本轮未下载权重。
- 取得 DFlash v2 HTML（391,888 / `3613e0871e0fe0570296d2e819c57e173879341bad9166b6bdab79cc2858c4a7`）。面试知识点已整理为：target prefill 抽取多层 hidden feature 并投影；向 draft 每层 K/V 注入并跨 iteration 保留；block diffusion 并行生成 mask positions；随机 anchor、块内双向/块间隔离 sparse mask、早位置指数 loss decay 和冻结 shared embedding/LM head。论文结果不作本地复现。
- 7890 取得 [SGLang PR #40794](https://github.com/sgl-project/sglang/pull/40794) 网页（381,910 / `05971ce5ca09d01068297c09db275c61246943afe9ab0b0a56b7d3293ccf3128`），PR 于 2026-09-23 合入 `208f6f7501f748e70b3aaab9fc96ec659d39463b`。K3 DFLASH hook 复用 DSpark layer-output taps，不做 `+1`；cookbook 按 PP/DP attention/NPU/Hopper/AMD 条件 gate。PR 的 8xB300 GSM8K 95.7%/accept length 4.99 使用未公开 production draft、多个 PR build；focused unit test 曾加入后在最终 patch 删除，CI aggregate 有失败项，不能写成公开 draft 的复现或单 PR 贡献。
- SGLang release Atom 仍为 stable `v0.5.20`（1,058,795 / `9c9c5e9bcc5fc55fea6be1d8e9c77fd84d330522638083a80e74f2134ddd5eb2`），因此 DFLASH 是 main/source evidence。已同步研究笔记、第二十一册第 88.26–88.27 节、PAPERS、INTERVIEW_BANK、EXERCISES、source-index、model-inventory、BOOK_SERIES 和 plan；goal 保持 `active`。未做权重下载、GPU 测试、真实 API probe 或生产 acceptance。

## 2026-10-01 Gemini 4 Argon：排行榜新锚点但官方资料未闭环

- 使用用户提供的 Clash 代理 `http://10.24.27.134:7890` 显式请求，Artificial Analysis `/zh` HTTP 200，`1,930,493` bytes / `4fa2574617fc125da6e0cf6a8575d5b81084baf22a134fb69a5ad450504c8c8d`；DataCurve HTTP 200，`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`。AA 当前包含 `gemini-4-argon`，DataCurve 没有精确 Argon 行。
- Google AI Developers 模型目录 HTTP 200，`155,241` bytes / `032840f32f36ad8cfa30d0b2767f9ae17dbaca5a17ed3acfc8a1e6fa97b95c47`，没有 Argon；本轮没有获得可核验的 Google 官方模型卡、技术报告、博客或代码入口。
- Argon 当前标记为“AA 新锚点 + 官方资料缺口”，不新增技术章节，不把 AA/provider 字段写成 Google 官方规格，也不迁移其他 Gemini 的 Agent 结果。goal 保持 `active`。
- 同一 AA 快照还出现重点厂商 OpenAI 的新 canonical `gpt-6-1-sol`（含 effort 变体）；DataCurve 没有精确 `mini_swe_agent_gpt_6_1_sol_*`。本轮 OpenAI 专属模型页请求随后再次失败，尚未取得官方资料，因此单独标为“仅榜单新锚点，官方资料待核验”，不迁移 GPT-6 Sol/Astra/Luna 的指标或 Agent 结果。

## 2026-10-01 GPT-6.1 Sol：官方模型与 runtime 合同收口

- 修正 OpenAI URL 后取得官方模型页 HTTP 200，4,294 bytes / `e3ebaa16dd9b9f9fc107c42183fb6cf9d22927e1af03dbbdfa0ccc38e4e4ac31`；Reasoning、Agents、Compaction 文档均 HTTP 200。
- 已核验 `gpt-6.1-sol`、Responses tool calling、Chat Completions 无 tool calling、`low/medium/high/xhigh/max`（默认 medium，禁用 none/minimal）、standard/pro mode、1,050,000 context、922,000 input、128,000 output、text/image input、EU residency 与 Fast mode 限制。
- `reasoning.context=all_turns` 可复用兼容历史 opaque reasoning items；`compact_threshold` 触发 encrypted compaction item，stateless chaining 必须回放该 item。Agents API、Agents SDK、Responses conversation 和 sandbox 是不同资源。
- 已同步研究笔记、source-index、model-inventory 和计划；正式书籍与题库配套随后补齐。当前状态为 **AA 单榜内容专题闭环 + OpenAI API/runtime contract**；参数、架构、完整训练 recipe、完整权重、独立 benchmark、真实 endpoint、目标硬件和生产验收仍待核验。goal 保持 `active`。

## 2026-10-01 Gemini 4 Argon：官方入口负检索复查

- 通过用户提供的 Clash 代理刷新两榜：AA HTTP 200，`1,930,493` bytes / `a47cc50856c95c96d04679f1d52c6cf76dacb26a78248d8e278ccacdf5590efe`；DataCurve HTTP 200，`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`。AA 没有新增重点厂商 canonical，DataCurve 无 Argon 行。
- Google AI Developers sitemap HTTP 200，`15,361,787` bytes / `f74529bc6bc9685feab9c9dba0116a2dfcebe0ea16c2b7a6b764c3857da00e98`，全文无 `argon`；AI Developers 与 DeepMind 猜测页面均 HTTP 404。
- Argon 继续标记为“AA 新锚点 + 官方资料缺口”，不新增技术章节或模型规格；goal 保持 `active`，等待官方资料出现或排行榜移除该 canonical 后重新审计。
- AA Argon 详情页复查 HTTP 200，`3,813,855` bytes / `aaaf7cbe1b821904de6149a9dfd0810c54c9db878054e30b9014676d77db341`；页面标注 Google/proprietary、2026-09、1 个 API provider，并说明参数规模未公开。该页是第三方目录/provider 观测，不能替代 Google 官方来源，因此 Argon 仍未闭环。

## 2026-10-01 排行榜专题收口

当前 Artificial Analysis 与 DataCurve/DeepSWE 重点模型的资料级闭环已完成。最后的 `gemini-4-argon` 已从 AA 榜单锚点推进到 Google The Keyword 官方发布证据，并同步第二十一册第 95 章、研究笔记、模型盘点、来源索引、题库、练习、术语、论文索引、项目、知识图谱、`plan.md` 与 `progress_v3.md`。发布方结果与独立评测保持分栏；参数、完整训练配方、权重、目标硬件、独立 benchmark、真实 API 和生产 SLO 仍是未验证边界。

最新增量记录见本文件，模型证据盘点见 [`research/model-update-2026-09/model-inventory.md`](research/model-update-2026-09/model-inventory.md)。

## 2026-10-02：计划文件职责收口

- 重构 `plan.md`：仅保留项目目标、研究边界、当前工作队列、工作流、主题 backlog、交付物关系和完成定义。
- 已从计划文件移除按日期的执行日志、网络/哈希快照、模型专题事实和完整来源清单；这些内容分别由本文件的历史记录和 `research/model-update-2026-09/` 下的专题笔记承载。
- `WRITING_SPEC.md` 保持不变，继续作为长期写作规范的唯一入口；`plan.md` 是当前唯一计划入口，`progress.md` 只保存第一次计划的执行记录。
