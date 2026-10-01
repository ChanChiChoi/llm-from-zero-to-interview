# 大模型资料更新计划 v2

更新时间：2026-09-29

## 第二轮全系列改写与精修执行计划

本节承接原 `SECOND_PASS_REVIEW_PLAN.md` 中的执行内容。长期写作规则已拆入 `WRITING_SPEC.md`；本节只保留阶段顺序、具体任务、知识点清单、阶段产物和状态，不作为正式正文规范。

### 优先级

1. **公式全量修正**：逐册检查公式、变量、推导、行内/块级写法和 GitHub Markdown 兼容性；重点修复 `\\text{...}` 下划线、变量不一致、缺少变量解释、推导跳跃和缺少直觉说明。产物是修正后的章节，以及维护文件中的专家复核项。
2. **教学 demo 补充**：优先覆盖 tokenization、attention、softmax 稳定性、cross entropy、top-k/top-p、temperature、embedding 检索、chunk overlap、BM25/hybrid retrieval、rerank、RAG assembly、LoRA、KV cache、prefill/decode、continuous batching、prefix cache、TTFT/TPOT、LLM-as-a-judge、bad case、成本估算、工具 schema 和 prompt injection 防护。
3. **广度覆盖审计**：对照目录、主流论文/框架/官方文档和岗位面试要求，检查 Transformer、预训练、后训练、数据、Tokenizer、Scaling、MoE、长上下文、RAG、Agent、工具协议、多模态、Reasoning、评测、安全、Serving、AI Infra、分布式训练、压缩量化、产品化和系统设计等方向，形成缺失主题清单。
4. **知识点深度增强**：为薄弱主题补齐历史动机、机制、最小例子或 demo、工程边界、失败模式、面试追问和专家视角；重点关注优化器、归一化、位置编码、Attention 变体、LoRA/QLoRA、DPO/RLHF、RAG 评测、Agent 安全、KV cache、PagedAttention、batching、评测统计、安全和线上 SLO。
5. **联网资料校准**：贯穿以上阶段，最后整体复查过时内容、新兴方向、官方来源、版本差异和证据边界。
6. **配套同步**：正文修改后按需同步 `INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、第四册百科、`PROJECTS.md`、`PAPERS.md` 和 `KNOWLEDGE_GRAPH.md`。
7. **结构收口**：修复目录链接和交叉引用，压缩重复内容，并更新 `README.md`、`BOOK_SERIES.md`、`progress_v2.md` 等维护入口。

### 章节级执行顺序

公式全量修正 → 教学 demo 补充 → 广度覆盖审计 → 深度增强 → 联网资料整体复查 → 配套文件同步 → 目录、链接和重复内容收口。联网校验实际贯穿前四阶段，不应等到最后才开始。

### 第二轮当前状态

- 24 本主书第一版已完成，`book-llm-engineer` 补充篇第一版已完成。
- 第二轮已持续推进公式修正、demo 代码、联网资料校准、纵向配套同步和进度记录。
- 第二十四册当前落盘的第 1--60 章已完成第二轮精修，相关百科、题库、练习、术语、项目路线和知识图谱已同步。
- 下一步继续处理未收口章节；对已完成专题执行总体验收、目录/索引一致性检查和最终收口。

## 2026-10-01 新锚点：Gemini 4 Argon 官方发布资料

两榜经用户提供的 `10.24.27.134:7890` 显式代理刷新成功：AA 当前包含 `gemini-4-argon`，DataCurve 没有精确 Agent 配置。Google The Keyword 官方公告现已核验，Argon 状态升级为“AA 单榜内容专题闭环（Google 发布方证据）”；公开模型卡、技术报告、权重、完整 API、独立复现和精确 DataCurve 行仍待核验。详见 `research/model-update-2026-09/gemini-4-argon-source-notes.md`。

同一快照还发现 OpenAI `gpt-6-1-sol` canonical 及 effort 变体；DataCurve 无精确 Agent 行。本轮 OpenAI 专属资料请求失败，暂记为“仅榜单新锚点，官方资料待核验”，不迁移其他 GPT-6 版本结果。

OpenAI 正确模型路径已通过显式代理取得：`gpt-6.1-sol` 模型页、Reasoning、Agents、Compaction 均可核验。该锚点现升级为“AA 单榜内容专题闭环 + API/runtime contract”，已确认 effort/mode、all-turns opaque reasoning、compaction、Responses tools、长上下文和 residency 边界；参数、架构、训练 recipe、硬件和生产验收仍待核验。

## 目标

以 2025 年以来新发布或快速进入主流评测的新模型为锚点，更新本项目的模型谱系、技术知识、论文资料和面试训练内容。新文件是当前执行计划的唯一入口；旧版 `plan.md` 保留作历史参考。自 2026-09-14 起，新模型候选的发现入口严格收敛为 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜。

## 最近收口锚点：GLM-5.3（2026-09-29）

沿两榜已有标准 GLM-5.3 锚点，补齐 slime 当前 main 的 Straw 异步 rollout、模型/queue 联合 commit marker、历史 step branch rollback 与 indexed archive retention。Z.ai 博客引用 slime，但 2026-09-28/29 的通用上游提交不能倒推为 GLM-5.3 原训练实现；4-GPU Qwen2.5-0.5B 集成测试没有在本机执行。研究笔记、第二十册第 20.23 节、题库/练习、source index、model inventory、计划和进度已同步。

## 前一锚点：Kimi K2.7 Code（2026-09-29）

沿两榜已有 Kimi K2.7 Code canonical 核验固定 HF revision 的部署指南与 KTransformers 文档。发布指南提供 RAWINT4 部署命令，但同提交的通用精度支持矩阵未列 K2.7/K2.5；因此记录为文档覆盖不一致、兼容性未实测，不推断支持或不支持。CPU/GPU 推理和 LoRA SFT 吞吐仍是发布方硬件结果，不是本机验收。研究笔记、第二十一册第 94 章及题库/练习已同步；没有下载权重或运行 GPU。

2026-09-29 后续复验：当前工作区经 `10.24.27.134:7890` 访问百度、Artificial Analysis、规范 DataCurve 和 Google Interactions 页面均成功。AA 的 60 个唯一 `/models/<slug>` 路由与同日基线一致；DataCurve 的 70 个 `mini_swe_agent_*` 配置 ID 与快照哈希均一致，未发现新重点厂商 canonical。Gemini 3.8 Flash 的 Thinking、Tool combination、Interactions overview/API reference 仍对 signature 字段范围存在文档/schema 冲突；Python SDK main 的最新提交只改 tuning-job metrics，Interactions 类型文件与既有哈希相同。该复验只延续旧锚点资料核对，不是新模型发现，也没有调用 Gemini endpoint。完整快照与证据边界见 [`progress_v2.md`](progress_v2.md) 和 [Gemini 3.8 source notes](research/model-update-2026-09/gemini-3.8-flash-source-notes.md) §21。

后续经备用 `10.237.126.170:1234` 已完成刷新：2026-09-29 20:32 AA `/zh` HTTP 200，1,747,602 bytes / SHA-256 `970aaacc0dedaf2a624b5ede1e07fa7f6659da78f8b8d2a0dd451b67da3b3120`；规范 DataCurve HTTP 200，268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`。两者与 20:19 同日快照逐字节一致，AA 路由/DataCurve 配置数仍为 60/70，没有新增八家重点厂商 canonical。7890 在用户 shell 可用，但本会话沙箱内及获批沙箱外复试均连接超时；只作为执行环境差异记录，不推断代理全局状态。

此前收口的 Grok 4.20 Multi-agent 仍为 AA 单榜内容专题闭环（runtime/API 与 harness）：官方专页明确 beta/API/tool/计费合同，且不代表架构/训练或生产验收闭环；详见 `progress_v2.md`。

当前推进：沿两榜已确认的 DeepSeek V4.1-Flash 锚点核查 SGLang `main` 新合入的 PR #39313。它提供 V4-family 混合精度 MegaMoE shared-expert fusion 与 stream-overlap 证据；其性能/准确性数据绑定 DeepSeek-V4-Flash-0731，不得迁移为 V4.1 结果。研究笔记、第二十一册 81.21、source-index、model inventory、面试题、计划和进度已同步；下一项从刷新后的两榜现有 canonical 中选择有权威资料支撑的知识缺口，不从官方资料另增模型。

## 当前重点厂商范围

主动关注范围仅包括：OpenAI、Anthropic、Z.ai（GLM）、Qwen、Kimi/Moonshot、DeepSeek、Google（Gemini）和 xAI（Grok）。其他厂商及其新模型暂不主动检索、核验或扩写；已有历史记录保留，但不进入下一轮更新队列，除非用户明确通知需要关注。

## 锚点研究原则

后续以两个排行榜中近期出现、排名靠前或在 Agent/推理/长上下文/多模态方向具有代表性的模型作为热点锚点，不追求把八家厂商的所有历史版本和长尾模型列全。对每个锚点，先归并 `effort`、fallback、provider 和 Agent harness 配置，再沿模型的官方博客、技术文档、模型卡、技术报告、论文、代码仓库和 API 文档追踪新技术、方法和知识点；这些权威资料用于扩展锚点，不作为新的模型发现入口。

“闭环”按内容阶段判断：排行榜发现、权威来源、研究笔记、正式章节和配套同步均具备，标为“内容专题闭环”；已有权威来源、研究笔记和配套同步但尚无独立正式专题，标为“资料级闭环”；只有部分官方入口或被系列章节覆盖，标为“部分覆盖”；只有排行榜条目，标为“仅候选”。任何状态都不代表参数、训练配方、内部架构或独立 benchmark 已全部公开确认。

最近已推进锚点（2026-09-28，已完成）：Qwen3.6-35B-A3B 官方博客与 evaluator/harness、Qwen3.8 Max (0902) hosted revision 的 cache/rate-limit/Responses 合同、GLM-5.3 compaction-aware trajectory，以及 Qwen3.7 Max 的 environment scaling / Task-Harness-Verifier / 长程 kernel 优化与 reward-hacking monitor 专题均已收口。Qwen3.7 Max 的详细来源、证据边界与本轮代理结果见下方及 `progress_v2.md` 最新记录。

补充 QA 状态：explicit-cache `>1,024` 示例出处已更正；Qwen cache contract toy 输出 `ok=true` / `local_protocol_toy` / `network_called=false`，Markdown 围栏、关键引用目标、旧标签搜索和 `git diff --check` 均通过。该 toy 不代表真实 QwenCloud endpoint 验收。

锚点队列补证（2026-09-28）：GPT-5.5 Instant May/June 已在前序阶段以“AA 榜单级关联配置 + 官方身份负证据”收口，本次不重开为活动锚点。OpenAI 当前目录仅说明 `chat-latest` 是会动态更新、底层 snapshot 未披露的 ChatGPT Instant alias；没有证据将它映射到 AA 的 GPT-5.5 Instant 或 API `gpt-5.5`。该负证据继续保留，不作为新的模型发现入口。

GLM-5.3 前序阶段（2026-09-28）已完成受影响文件 QA：固定 `THUDM/slime@8ee9c1e` 显示 token provenance、loss mask、trajectory realign/fork 与共享 rollout ID；它是关联训练框架证据，不是 GLM-5.3 内部实现。该提交 reward/K 文档与 full-reward-per-sample 代码/测试不一致仍作为上游证据冲突；GLM-5.3 专属 compaction schema/触发/质量门禁未公开。本次 Straw 增补见本计划最新段落。

上一活动锚点（2026-09-28，GPT-6 Luna）官方资料复验已收尾：`configuration_update` 与 compaction 的互斥、重新插入 update 的时序，以及 response `reasoning.effort` 仍是 request-level 字段等边界均已记录；DataCurve 无精确 Luna Agent 行，未做真实 API probe。

上一活动锚点（2026-09-28，Claude Sonnet 5）已完成复验：沿两榜已有 canonical 阅读 Anthropic What's New、Migration、Prompting、Models API/Overview 与 compaction 官方资料，补入两套不能混用的 API/回放状态机、签名 block 顺序/错误处理、后台 tail/cache 与 prefix-binding enforcement 范围。资料已同步研究笔记、第二十册、题库、练习、source-index 和 model-inventory；没有调用真实 API，Sonnet 5 保持资料级闭环。

2026-09-28 7890 当前链路复验：当前执行环境显式经 `10.24.27.134:7890` 请求百度、Artificial Analysis `/zh` 与 DataCurve DeepSWE 均 HTTP 200。AA 快照 `1,660,509` bytes / SHA-256 `1e13276cae1ec1d9e5a510946b80029c3eb1c15cb746df7e7f647ecd6f1640bc`，DataCurve `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，分别与本日先前保存的 `next-turn` 快照逐字节一致；八家重点厂商 canonical 集合没有新增。重跑 GLM-5.3 compaction audit 仅验证通用教学状态机，证据级别仍为 `local_protocol_toy`，不改变模型专题状态或官方“未公开专属 compaction schema”的边界。后续按既有 canonical 队列继续选择面试价值高、仍有公开证据可推进的知识缺口；不把真实权重/硬件验收作为公开资料研究的替代目标。

同日继续经 7890 刷新 AA `/zh`（1,660,542 bytes / SHA-256 `f5d63499b482b9510d14f2162e8317231da77f9b3875fa36171f53e5891e5c65`）和 DataCurve（268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`）；AA 与此前 next-turn 快照比较，55 个唯一 `/models/<slug>` 路由无增删，DataCurve 与同日快照逐字节一致，未发现新 canonical。随后重试 Gemini 3.8 Flash 的 AA 详情与 Google Video Understanding 两个曾报 TLS EOF 的 URL，指定 HTTP/1.1 后均 HTTP 200；AA detail 新 snapshot 与速度测量差异、Google 文档 hash/字段复核见 `progress_v2.md` 与 [`gemini-3.8-flash-source-notes.md`](research/model-update-2026-09/gemini-3.8-flash-source-notes.md)。没有从官方文档另增模型。

上一活动锚点（2026-09-28，Gemini 3.8 Flash，已收尾）：只沿两榜已有 `gemini-3-8-flash` canonical 深挖 Interactions signature/schema 文档差异，不从 Google 官方目录或 SDK 另发现模型。复核 AA 当前 high 页面及 DataCurve 精确 `mini_swe_agent_gemini_3_8_flash_high` 行后，追查 Google Gen AI Python SDK `main` 最新仍为 2026-09-26 commit `6d012889...`。本轮区分了 Interactions 专属 `google/genai/_gaos` typed schema（`extra="allow"`）与同仓库 `_common.BaseModel`（`extra="forbid"`），也区分 `ThoughtStep.signature` 的 schema optional 与宿主 stricter replay policy。正式文档范围冲突仍存在；无 API key/授权 endpoint，真实服务端行为保持未验证。研究笔记、第二十册第 23 章、demo、题库/练习、模型盘点、source-index、书系、图谱和进度已同步并完成 QA。

上一活动锚点（2026-09-28，DeepSeek V3.2，已收尾）：沿已有双榜锚点复补官方技术报告 §4.4，完成 Search Agent 80% context-management trigger、Summary/Discard-75%/Discard-all、BrowseComp 51.4/67.6* 的 harness 归属和证据边界；随后经 7890 下载并视觉核验 Figure 6 SVG，与 1234 抓取版逐字节一致。图上确认四策略、Real Steps/Browsecomp 坐标轴；未把无标签散点目测值写成精确数据，并明确 Summary “up to 60.2”是图中指标值而不是 `+60.2%`。不把作者 harness 结果移植为榜单或裸模型分数。章节、研究笔记、来源索引、模型盘点与本进度已同步，QA 记录见 `progress_v2.md`。

最近活动锚点（2026-09-28，Claude Opus 4.8，内容专题已收尾）：仅沿两榜既有历史锚点补 Anthropic 官方 Dynamic Workflows 与 System Card，不从官方目录/博客另发现模型。7890 复验 AA、DataCurve 和官方页；System Card PDF 已抽取为 246 页正文。重点写入对话外动态编排、反驳式复核、checkpoint/resume、first-run consent/admin disable、Multi-Agent score/token/derived-latency trade-off，以及代码总结漏报评测的适用边界和 System Card changelog 勘误。已同步研究笔记、第十七册第九章、练习/面试题、PAPERS、BOOK_SERIES、source-index、model-inventory、知识图谱与本计划/进度。Opus 4.8 已被 AA deprecated 并指向 Opus 5，保留为历史锚点；不新增架构章节，不将 System Card benchmark 当作 Dynamic Workflows 线上 SLO。下一步仍从两榜既有八家重点厂商 canonical 集合选择未收口知识缺口，goal 保持 `active`。

## 前序阶段记录

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

## 工作流

### 2026-09-24 09:30 UTC：7890 当前快照复验与 Gemini 3.8 Flash 专题落地

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

1. **模型发现（唯一入口）**：定期查看 [Artificial Analysis](https://artificialanalysis.ai/zh) 与 [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 两个排行榜；只有在这两个榜单中出现的模型才进入新的候选盘点。记录榜单名称、具体页面、配置/effort、榜单日期、采集日期和快照哈希。
2. **官方核验**：对已由上述两个榜单发现的候选，优先收集其官方博客、技术报告、论文、模型卡、代码仓库和 API 文档；这些资料只用于核验候选事实和扩展周边技术，不作为新的模型发现入口。社区博客只用于补充工程经验。
3. **技术拆解**：每个模型记录发布时间、组织、开放/闭源状态、参数或上下文信息、训练/推理特色、已确认来源和待核验点。
4. **知识映射**：把新技术映射到现有书册：架构、训练、推理、Reasoning、Agent、工具协议、多模态、AI Infra、评估与安全。
5. **内容更新**：先更新第四册百科和相关专题章节，再同步 `INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`PAPERS.md`、`KNOWLEDGE_GRAPH.md`。
6. **质量检查**：检查时间语境、模型名称、许可证、版本差异、公式和代码；所有事实保留来源链接与核验日期。多模态流式专题额外检查媒体 timestamp、chunk 边界、codebook 状态、音频 packet、取消/恢复和理论延迟是否误写成本机 SLO。

## 首批模型观察清单

以下是历史上从两个排行榜发现、随后进入核验流程的候选，不代表项目已确认其全部事实：GPT、Claude、Gemini、DeepSeek、Qwen、Kimi、GLM、MiniMax、Mistral、Step、Grok 及其他推理、代码、多模态和 Agent 模型。历史清单中的其他厂商记录继续保留，但当前不主动追踪其新模型。以后不得仅因官方新闻、Hugging Face Trending、LMSYS、云厂商目录、论文列表或搜索结果出现名称，就把它新增为模型候选；必须先回溯到上述两个排行榜中的条目，并且属于当前重点厂商范围，或已有用户明确指示。

## 每个模型的记录模板

- 模型与版本：
- 首次公开日期：
- 发布组织与链接：
- 开放性、许可证与可用渠道：
- 规模、上下文和模态：
- 核心新技术：
- 训练/后训练/推理特点：
- 主要基准与局限：
- 官方来源：
- 论文/技术报告：
- 社区资料：
- 核验日期与可信度：
- 对本书系的更新位置：

## 交付顺序

先提交模型盘点表和来源索引，再更新百科与专题章节，最后同步题库、练习、术语、项目、论文和进度文件。每轮以可审阅的小批次提交，避免再次形成难以维护的大型总表。

## 正式章节的深度要求

正式正文以讲清讲透为验收条件，不设置篇幅上限。研究清单和来源摘记只用于组织证据，不能直接充当章节。每个重要知识点独立展开，不把多个未解释的术语堆在同一段。

先用具体情境或生动例子说明问题，再逐步引入定义、历史动机、直觉、机制和公式。公式解释符号及适用假设；代码说明依赖、输入输出、关键步骤和运行现象。适合代码辅助理解的主题提供可运行的小实验，并说明教学实现与真实系统之间的差距。

面向初学者解释前置概念、常见误解与中间推导；面向有经验读者展开复杂度、工程取舍、失败条件、实验对照和面试追问。各项按主题需要组织，不机械填模板；必要时拆成多节或独立章节，不能通过省略解释缩短篇幅。

## 当前研究入口

- [模型候选盘点](research/model-update-2026-09/model-inventory.md)：需区分独立模型和推理配置，并核验榜单日期。
- [GPT-6 Astra 官方资料摘记](research/model-update-2026-09/gpt-6-astra-source-notes.md)：已核验两个排行榜锚点、官方模型页、专属模型指南、Reasoning、Prompt caching、Conversation state、Compaction、Tool search、Async tool calling、Mid-turn steering、Misalignment monitoring、Agents、Fast mode 和开发者博客；当前为内容专题闭环（官方运行时与 Agent 协议补证）。
- [GPT-6 Sol 官方资料摘记](research/model-update-2026-09/gpt-6-sol-source-notes.md)：已由 Artificial Analysis 的 `gpt-6-sol` canonical 条目确认，DataCurve 当前没有精确 Agent 行；已核验 OpenAI 官方模型页、Reasoning、Agents、Tools 和 Compaction 文档，记录 1.05M/922K/128K 预算、`none`--`max` effort、standard/pro mode、动态 `configuration_update`、Responses/Chat Completions function-calling 边界、工具目录、runtime 所有权和 opaque compaction state；当前为 AA 单榜资料级闭环，参数、架构、完整训练 recipe、独立 benchmark 和生产 acceptance 待核验。
- [GPT-5.6 官方资料摘记](research/model-update-2026-09/gpt-5.6-source-notes.md)：已核验 Sol/Terra/Luna 模型页、Reasoning、Agents、Tools、Prompt Caching、Compaction 和开发者博客；当前为资料级闭环。
- [GPT-5.5 官方资料摘记](research/model-update-2026-09/gpt-5.5-source-notes.md)：已核验 GPT-5.5/Pro 模型页、专属指南、Reasoning、Tools、Tool search、Prompt Caching、Compaction、Images/Vision、Conversation state 和 Background；当前为资料级闭环。
- GPT-5.5 Instant（May/June）只作为 Artificial Analysis 榜单关联配置保留；已重新核验两个详情页，并检查官方 `gpt-5.5-instant` 精确路径的 404 负证据，不能并入 `gpt-5.5` 或迁移其 Agent 成绩。本轮不再以它作为活动锚点。
- [GPT-5.4 官方资料摘记](research/model-update-2026-09/gpt-5.4-source-notes.md) 与 [mini/nano 独立摘记](research/model-update-2026-09/gpt-5.4-mini-nano-source-notes.md)：已复验两个排行榜、GPT-5.4 家族指南和 mini/nano 精确模型页；已分开记录 1M vs 400K context、deferred `tool_search`、computer use、native compaction、custom tools/CFG、`allowed_tools`、`phase`、snapshot、价格与精确工具矩阵。GPT-5.4/mini/nano 当前均为资料级闭环，Pro 仍只作为关联服务档位记录。
- [OpenAI gpt-oss 官方资料摘记](research/model-update-2026-09/gpt-oss-source-notes.md)：已从 Artificial Analysis 发现 `gpt-oss-120b`/`gpt-oss-20b`，并核验 Model Card、arXiv、官方仓库、Harmony、Hugging Face 和 Cookbook；当前为内容专题闭环，DataCurve 没有精确 Agent 行。
- [Claude Opus 4.6 官方资料摘记](research/model-update-2026-09/claude-opus-4.6-source-notes.md)：已从 Artificial Analysis 的 adaptive/基础配置发现并核验 Anthropic 发布页、System Card、模型页、adaptive thinking、effort、compaction、tool search 和 computer use；DataCurve 当前没有精确 Opus 4.6 行，当前为资料级闭环。
- [Claude Opus 4.7 官方资料摘记](research/model-update-2026-09/claude-opus-4.7-source-notes.md)：已从 Artificial Analysis 的 adaptive/max 与 non-reasoning/high 配置发现并核验 Anthropic 发布页、System Card、模型页、effort、task budgets、vision、compaction 和 tool search；DataCurve 当前没有精确 Opus 4.7 行，当前为内容专题闭环（复用 Agent/tool 与 inference serving 章节）。
- [Kimi K3 官方资料摘记](research/model-update-2026-09/kimi-k3-source-notes.md)：已核验官方发布文章、技术报告、许可证、HF 固定 revision/config、safetensors manifest、`modeling_kimi_linear.py`、FlashKDA README/commit、vLLM stable docs/API、PyPI `0.29.0` stable release、v0.29.0 registry/model source、package entry 和 K3 recipe；新增零依赖 manifest 审计脚本并完成 497,220 个 tensor、96 个连续分片、93 层及混合 cache 状态边界核验。当前为内容专题闭环 + stable release implementation evidence：recipe 仍为 pre-release/nightly 优化路径；完整权重未下载，目标硬件加载、完整 runtime、硬件 profiling、独立复现、hybrid cache recovery 和线上 acceptance 仍待核验。
- [Kimi K2.7 Code 官方资料摘记](research/model-update-2026-09/kimi-k2.7-code-source-notes.md)：已核验两个排行榜、Moonshot/Kimi 官方资源/API 文档、固定 revision 模型卡、配置和部署指南；当前为资料级闭环。
- [Claude Opus 5 官方资料摘记](research/model-update-2026-09/claude-opus-5-source-notes.md)：已核验两个排行榜锚点、模型目录/专属页、adaptive thinking、工具与 effort 中途变更、fallback、缓存边界、发布方评测与 system card 入口；当前为资料级闭环。
- [Claude Opus 5.5 官方资料摘记](research/model-update-2026-09/claude-opus-5.5-source-notes.md)：已由 Artificial Analysis 新增 canonical 条目确认，DataCurve 当前无精确 Agent 行；已核验 Anthropic 2026-09-22 发布页、官方 model page/What's new/migration/fast mode Markdown、System Card PDF 正文、`claude-opus-5-5`、1M/128K、always-on adaptive thinking、effort、thinking block binding、tool/computer-use 兼容性、compaction/inline tools、token/缓存成本、长任务 coding、Cyber/Life Sciences Verification、fallback、preserved thinking anti-distillation，以及 RSP、CoBench/AECI、Cyber、Agent safety、prompt injection、OSWorld 和 multi-agent 评测条件；当前为 **AA 单榜 + System Card 正文证据**，参数、架构、训练 recipe、独立复现和生产 acceptance 待核验。
- [Claude Fable 5 官方资料摘记](research/model-update-2026-09/claude-fable-5-source-notes.md)：已核验两个排行榜锚点、模型页、发布/重新部署公告、thinking/effort、拒答/fallback、memory、程序化工具调用、compaction、context editing 和 task budgets；当前为资料级闭环。
- [Claude Opus 4.8 官方资料摘记](research/model-update-2026-09/claude-opus-4.8-source-notes.md)：已核验两个排行榜锚点、Anthropic 发布公告、system card、Dynamic Workflows 博客、effort 和 Messages API 的 system entry；当前为资料级闭环。
- [Claude Fable 5.1 官方资料摘记](research/model-update-2026-09/claude-fable-5.1-source-notes.md)：已复验 Artificial Analysis 榜单、确认 DataCurve 暂无 Fable 5.1 行，并补齐 Anthropic 发布页、System Card 正文、Fable/Mythos safeguards、cache read 定价、发布方 benchmark、安全摘要、arXiv 外部使用检索和第二十册第 21 章 21.29 thinking-state 专题；当前为 AA 单榜内容专题闭环，真实 API/内部架构/训练待核验。
- [Claude Sonnet 5 官方资料摘记](research/model-update-2026-09/claude-sonnet-5-source-notes.md)：已完成两个排行榜、Anthropic 发布页、模型目录、完整 API/Agent 文档、System Card、Research/arXiv 定向复核；已核验 Adaptive thinking、effort/预算、thinking block、tokenizer、context awareness、compaction、工具调用和配置级评测边界，当前为资料级闭环。
- [Claude Sonnet 4.6 官方资料摘记](research/model-update-2026-09/claude-sonnet-4.6-source-notes.md)：已完成两个排行榜、Anthropic 发布页、真实 Models Overview、System Card 入口和 Agent 运行时资料复核；已核验 1M context beta、adaptive/extended thinking、context compaction、tool search、computer use 与 prompt-injection 边界，当前为资料级闭环。
- [Claude Haiku 4.5 官方资料摘记](research/model-update-2026-09/claude-haiku-4.5-source-notes.md)：已核验模型目录快照、200K context、64K 输出、extended thinking、fastest 延迟、平台与成本边界。
- [DeepSWE v1.1 排行榜快照](research/model-update-2026-09/deepswe-snapshot-notes.md)：已核验页面更新时间、任务/仓库/语言规模、统一 `mini-swe-agent` harness 和配置级 Pass@1/成本字段。
- [Artificial Analysis 快照字段解释](research/model-update-2026-09/inventory-interpretation.md)：已补记 2026-09-09 配置级指数示例及 `releaseDate`、参数和开放性字段的第三方证据边界。
- [DeepSeek V4.1-Flash 官方资料摘记](research/model-update-2026-09/deepseek-v4.1-flash-source-notes.md)：已核验发布页、固定 Hugging Face revision、模型卡、配置、encoding/evaluation README 和 API alias 边界；技术报告 PDF 已逐页提取，补充层排布、缓存/部署、训练基础设施、后训练和评测边界；2026-09-20 又固定了 inference/reference 源码哈希。完整 production kernel、线上接受率和独立 profiling 仍待核验。
- [DeepSeek V4 Flash Vision 官方资料摘记](research/model-update-2026-09/deepseek-v4-flash-vision-source-notes.md)：锚点为 Artificial Analysis 的 `deepseek-v4-flash-vision`；已核验实验发布、Vision/Files/Responses/价格文档、图像预算、`file_id` 生命周期、工具图像回灌和旧 alias → V4.1-Flash 路由；DataCurve 当前无同名行，视觉专属架构/训练报告仍待核验，当前为资料级闭环。
- [DeepSeek V3.2 官方资料摘记](research/model-update-2026-09/deepseek-v3.2-source-notes.md)：Artificial Analysis 有精确 `deepseek-v3-2` 的 Non-reasoning 条目，DataCurve 当前无精确行；已核验官方模型卡、技术报告正文、V3.2-Exp inference、TileLang、DeepGEMM/FlashMLA PR 和 vLLM recipe，提取 DSA 两阶段训练、2,048 KV top-k、FP8 indexer/稀疏 MLA 执行链、prefill MHA/decode MQA、FP8 KV cache、EP/DP/TP serving 与 recipe 证据边界；当前为 AA 单榜资料级闭环。
- DeepSeek V3.2 的书系配套已同步到 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`，并映射到稀疏注意力、后训练与 Agent serving 章节；不新增重复的 V3.2 专属架构章节。固定官方 `config.json` 与 V3.2-Exp inference 都核验为 `q_lora_rank=1536`；最终 production kernel 覆盖、召回曲线、FP8/BF16 端到端 profiling、线上 acceptance rate 和独立 benchmark 仍是待补证项。
- [Artificial Analysis 2026-09-14 实时快照](research/model-update-2026-09/artificial-analysis-2026-09-14-snapshot.md)：独立保存 1.77 MB 页面快照、哈希、约 702 个配置条目和 13 个新增结构 slug/别名；DeepSeek V4.1-Flash、DeepSeek V4 Flash Vision 与 K2 36B/A4B 已进入官方核验专题，其余新增项仍停留在发现层。
- [K2-Horizon-MoVA-36B-A4B 官方资料摘记](research/model-update-2026-09/k2-horizon-source-notes.md)：锚点来自 Artificial Analysis 榜单；官方 Hugging Face 模型卡、固定 revision、配置、实现和部署资料只用于核验该候选及扩展 MoVA/MoE/长上下文/Serving 技术。
- 上一锚点为 `IFM/K2-Horizon-MoVA-36B-A4B`：已完成 MoVA、稀疏 MoE、长上下文和 serving 的专题同步；后续新模型候选仍只能从两个排行榜产生，再按证据等级推进核验。
- [K2 Horizon 3.7B 官方资料摘记](research/model-update-2026-09/k2-horizon-3.7b-source-notes.md)：已核验 AA 单榜身份、DataCurve 精确行缺失、`K2HorizonForCausalLM` 当前 revision、dense/GQA/512K 配置、core/embedding 参数冲突、分阶段训练、RL expert merge、migration manifest、vLLM/SGLang recipe 和发布方 benchmark；正式对照已并入第二十一册第 82 章，当前为 AA 单榜资料级闭环。
- [Qwen3-VL-235B-A22B 官方资料摘记](research/model-update-2026-09/qwen3-vl-source-notes.md)：已核验 AA instruct/reasoning 两个配置归并、DataCurve 精确行缺失、HF 固定 revision/config、Qwen3-VL Technical Report、Interleaved-MRoPE、DeepStack、Video Timestamp、S0-S3 curriculum、SAPO、Thinking with Images 和 tool-call reward；正式专题已落地第二十一册第 92 章，当前为 AA 单榜内容专题闭环。
- [Qwen3.8 官方资料摘记](research/model-update-2026-09/qwen3.8-source-notes.md)：已核验 27B、2.4T-A95B、Flash-Next 的官方模型卡、Flash-Next GitHub/技术报告及 Qwen Cloud 的 hosted/open 边界；核心技术为 GDN/Gated Attention、QSA、Gated Residual、N-gram Embedding、Muon/AdamW 和 thinking protocol。
- 当前 Qwen3.8 路线进入“内容专题闭环”：正式落点为第二十一册第 83 章。QSA 的完整 kernel、GR/N-gram host-memory 端到端收益、Muon 分布式实现、线上 acceptance rate、目标硬件 profiling 和独立 benchmark 继续保持待核验；Flash-Next 报告结果只写成发布方自报。
- [Qwen3.8 Max (0902) 服务 revision 摘记](research/model-update-2026-09/qwen3.8-max-0902-source-notes.md)：已由两个排行榜确认条目，并通过 Qwen Cloud 产品页、Thinking、Function Calling 和 Context Cache 文档核验 0902 upgraded snapshot、alias、1M/991K/983K/131K 请求边界、`reasoning_effort`/`thinking_budget` 互斥、thinking 下 `tool_choice` 限制和 explicit/implicit/session cache；当前为资料级闭环，不写成新的 open checkpoint。
- Qwen3.8 Max 0902 的 DataCurve 只有泛化 `qwen3_8_max_xhigh` 行，不能迁移为 0902 独立 Pass@1；本轮不新增重复架构章节，扩展既有 Qwen3.8、Agent serving 和书系配套内容。
- [Gemini 3.7 Flash 官方资料摘记](research/model-update-2026-09/gemini-3.7-flash-source-notes.md)：已重新复验 Artificial Analysis、DataCurve DeepSWE、Google AI Developers 模型页、DeepMind Model Card、官方评测/安全入口、Interactions API、工具组合、Thinking、Context caching、Computer Use 和 agentic video 文档；当前为资料级闭环。
- Gemini 3.7 Flash 的面试主线是 thinking budget、interaction/step/state/background、tool context circulation、加密 signature 的 stateful/stateless 回放、built-in/custom tool 责任边界，以及 agentic video 的主动时间轴浏览和 `processing_call`/`processing_result`；Model Card 没有公开独立架构、参数或训练 recipe，暂不新增专属正式章节。
- [Gemini 3.6 Flash 官方资料摘记](research/model-update-2026-09/gemini-3.6-flash-source-notes.md)：已复验 Artificial Analysis、DataCurve、Google AI Developers 模型页、DeepMind Model Card、Thinking、Interactions、工具组合、视频、缓存和 Computer Use 文档，并完成 arXiv 精确/全文检索；当前为资料级闭环。
- Gemini 3.6 Flash 的面试主线是默认 medium 与 `minimal/low/medium/high` thinking、Interactions/state replay、Gemini 3 tool signatures 与 tool context circulation、agentic video 的 `processing_call`/`processing_result`、4,096-token implicit caching 和 Computer Use 的宿主执行边界；Model Card 将架构/训练/硬件/软件资料指向 Gemini 3.5 Flash，暂不新增专属正式章节。
- [Gemini 3.5 Flash 官方资料摘记](research/model-update-2026-09/gemini-3.5-flash-source-notes.md)：已重新复验 Artificial Analysis、DataCurve、Google AI Developers 模型页、What's New、DeepMind Model Card、Gemini 3 Flash 依赖 Model Card、Thinking/Interactions/工具组合/视频/缓存/Computer Use 文档和 arXiv 精确/全文检索；当前为资料级闭环。
- Gemini 3.5 Flash 的面试主线是默认 thinking 从 high 改为 medium、`thinking_level`/`thinking_budget` 互斥、thought preservation、Interactions/state replay、tool context circulation、4,096-token implicit caching、Computer Use prompt-injection detection，以及静态视频与 agentic video 支持列表的证据边界；官方把架构/训练/软硬件资料指向 Gemini 3 Flash，暂不新增专属正式章节。
- [Gemini 3.5 Flash-Lite 官方资料摘记](research/model-update-2026-09/gemini-3.5-flash-lite-source-notes.md)：已由 Artificial Analysis 精确条目发现，并核验 Google API 模型页、DeepMind Model Card/PDF、Thinking、Video understanding 和发布方评测；当前为资料级闭环（AA 单榜）。
- Gemini 3.5 Flash-Lite 的面试主线是低延迟高吞吐 subagent/document parsing、默认 `minimal` thinking、`minimal/low/medium/high` 请求级预算、agentic video 的时间轴按需读取和 `processing_call/result` 审计；Model Card 明确基于 Gemini 3.1 Flash-Lite，不把前代架构/训练资料写成 Lite 独有发明。正确的 `What's New` 页面属于 Gemini 3.5 Flash，不能把 `medium` 默认值或 GA 叙述迁移给 Lite。
- [Gemini 3.1 Pro Preview 官方资料摘记](research/model-update-2026-09/gemini-3.1-pro-preview-source-notes.md)：已重新复验 Artificial Analysis、DataCurve、Google AI Developers 模型页、Release Notes、DeepMind Model Card、评测方法、Thinking、thought signatures、工具组合、long context、caching 和 arXiv 定向检索；当前为资料级闭环。
- Gemini 3.1 Pro Preview 的面试主线是 `thinking_level` 与共同 output budget、`customtools` endpoint 的工具选择偏置、thought/tool signature 与 `id` 回放、tool context circulation、1M context/caching、评测设置隔离和 Frontier Safety 的 CCL/成本边界。Model Card 将架构/训练/硬件/软件资料指向 Gemini 3 Pro，暂无独立 3.1 技术报告，因此不新增专属正式章节。
- GLM-5 已完成 Artificial Analysis 单榜锚点到正式专题的推进：DataCurve 当前快照没有 `mini_swe_agent_glm_5_*` 精确行。官方技术报告已确认，研究主线转向 DSA indexer/top-k 召回、MoE 总容量与 active compute、`slime` 的 rollout/trainer 解耦、长轨迹 Agent RL 和评测分层；第 89 章已落地。完整 DSA kernel、异步调度细节、训练 recipe、硬件 profiling 和独立复现待补证。
- [Gemini 3.8 Flash 官方资料摘记](research/model-update-2026-09/gemini-3.8-flash-source-notes.md)：已核验 Google 官方模型页、DeepMind Model Card、发布博客、评测 PDF，以及 Thinking、Thought signatures、Interactions API、工具组合、Computer Use、File Search、URL Context、Code Execution、Structured outputs 和 Long context 文档。
- Gemini 3.8 Flash 已完成榜单发现、官方资料入库和周边技术提取，并新增第二十册第 23 章《Gemini Interactions：跨轮状态、签名回放与工具责任链》，状态为“内容专题闭环”。独立公开的 3.8 架构/训练报告、真实 API probe、目标硬件和生产验收仍待核验。
- [Grok 4.5 官方资料摘记](research/model-update-2026-09/grok-4.5-source-notes.md)：已核验两个排行榜的 `high` 配置、xAI 发布公告、Grok 4.5 模型页、Reasoning/Responses/Compaction/Tools/Web Search/X Search/Remote MCP 文档；当前为资料级闭环。
- [Grok 4.6 官方资料摘记](research/model-update-2026-09/grok-4.6-source-notes.md)：已核验两个排行榜的四档 effort、xAI 发布公告、官方模型页、Reasoning/Compaction/Tools 文档和论文精确标题检索；2026-09-21 已追加当前 AA/DataCurve 快照复验和 xAI 三代理失败边界；当前为资料级闭环。
- [Grok 4.20 官方资料摘记](research/model-update-2026-09/grok-4.20-source-notes.md)：已核验 Artificial Analysis 精确条目、DataCurve 精确行缺失、xAI 模型页/模型注册表、Reasoning/Multi Agent/Compaction/Tools/Release Notes 和 arXiv/新闻入口负证据；第二十册第 19.30 节及配套面试/练习已同步，当前为 AA 单榜内容专题闭环（runtime/API 与 harness），内部架构、训练 recipe、完整权重、独立 benchmark、目标硬件和生产 acceptance 仍待核验。
- [Grok 4.7 官方资料摘记](research/model-update-2026-09/grok-4.7-source-notes.md)：已由 Artificial Analysis 发现并完成 xAI 发布页、模型/API 文档、Responses encrypted reasoning、Context Compaction、Function Calling、Structured Outputs、Remote MCP 和 arXiv 精确负检索；DataCurve 当前没有精确 `mini_swe_agent_grok_4_7_*` 行，当前为 AA 单榜内容专题闭环。参数、架构、完整训练 recipe、生产 kernel、硬件 profiling、线上 acceptance 和独立复现仍待核验。
- Grok 4.7 当前时点补证已追加：Reasoning 文档确认每次 Responses 返回 encrypted reasoning 及服务端工具加密输出，公开 summarized reasoning 流事件，并区分 `store` 与 `previous_response_id`；Remote MCP 的 Streaming HTTP/SSE、SDK 字段差异和未支持的 `require_approval`/`connector_id` 已写入证据边界。模型页、发布页、Reasoning、Compaction 和 MCP 的 `7890` 刷新哈希见来源索引；正文没有形成新的模型 revision 或架构证据。

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

## 2026-09-14 Qwen3.8 配套同步收口

Qwen3.8 的专题内容已同步到 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、`README.md`、`BOOK_SERIES.md` 和 `ROADMAP.md`，并保留研究笔记、模型盘点、来源索引及第二十一册第 83 章的交叉入口。同步内容只围绕已发现锚点的 GDN/Gated Attention、QSA、Gated Residual、N-gram Embedding、Muon/AdamW 和 thinking protocol 展开。

第 83 章的 QSA 教学 demo 已修正为先为 causal tail 预留 token budget，再选择完整 micro-block；预算无法容纳 tail 时显式报错，并用断言锁定 tail 不被截断。该 demo 仍是标准库教学实现，不代表 Flash-Next 的生产 kernel；报告的 benchmark、loss、速度和稳定性数字继续按发布方自报处理。

下一步仍是补证队列：完整生产 kernel、目标硬件 profiling、线上 acceptance rate、N-gram host-memory 端到端收益、Muon 分布式实现、全系列训练/后训练配方和独立 benchmark。除非两个排行榜出现新条目，否则不新增模型候选。

## 2026-09-14 GPT-5.6 资料级闭环与后续顺序

GPT-5.6 的研究笔记已完成，闭环范围包括榜单发现、官方模型页、API 周边文档、开发者博客和证据边界。当前最适合面试的技术主线是：`standard/pro` 与 effort 解耦、跨轮 opaque reasoning state、function call 后的 reasoning item 回传、GPT-5.6 prompt-cache 显式断点/1,024 token 门槛/30m TTL、compaction 对缓存复用的影响，以及模型—协议—harness—执行器的责任分层。

本轮不把 GPT-5.6 或 GPT-5.5 的 DeepSWE 数字写成裸模型能力，也不把 API 字段推断为参数、MoE、训练或 RL 机制。下一锚点转向 Claude Fable 5；GPT-5.5/5.6 后续只补官方版本变更、真实 API 行为和独立评测复现。

2026-09-15 网络复访：`10.24.27.134:8098` 和 `10.237.126.170:1234` 均可访问 Artificial Analysis `/zh` 与 DataCurve 并返回 HTTP 200；`10.24.27.134:7890` 本次访问 DataCurve 返回 200，但访问 Artificial Analysis 出现 TLS EOF。Fable 5 页面快照已通过可用代理保存，页面大小和 SHA-256 见研究笔记。

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

## 2026-09-16 Gemini 3.1 Pro Preview 断点恢复与资料级闭环

昨晚 20:00—今早 09:00 的联网中断风险按“重新收集”处理。2026-09-16 重新使用三条用户提供的代理抓取 Artificial Analysis 中文首页、Gemini 3.5/Gemini 3.1 Pro 详情页和 DataCurve；六份榜单请求均返回 HTTP 200，同一页面三条线路字节一致。随后通过 7890 重新取得 Google API 模型页、Release Notes、DeepMind Model Card、评测方法、Thinking、工具组合、Long context、Caching 和 arXiv 检索页。

Gemini 3.1 Pro Preview 已完成从两个唯一排行榜到权威资料的资料级闭环：Artificial Analysis 详情页记录第三方 Intelligence Index `30.3597`、约 `108.4052 tokens/s`、约 `24.6088s` TTFT、1M context；DataCurve `mini_swe_agent_gemini_3_1_pro_preview_high` 为 53/452、Pass@1 `11.7257%`、Pass@4 `28.3186%`、平均成本约 `$2.1434`、平均 75.56 steps。所有数字保留配置、provider/harness、工具、任务集、环境和 verifier 边界，不能拼成裸模型排名。

官方资料确认 2026-02-19 发布、1,048,576 输入/65,536 输出、多模态输入、thinking、function calling、grounding、structured outputs 和 context caching；`gemini-3.1-pro-preview-customtools` 是面向 bash/custom tools 的 endpoint variant，不新增模型。Model Card 明确 3.1 基于 Gemini 3 Pro，并把架构、训练数据、硬件和软件资料指向 Gemini 3 Pro；没有独立参数、层/专家结构、完整训练 recipe 或专属技术报告。

本轮提取的面试主线为 `thinking_level` 与共同 output budget、thought/tool `signature` 与 `id` 回放、tool context circulation、built-in/custom tool 宿主责任、1M context 的缓存/召回/成本工程，以及 Model Card/Frontier Safety 的评测设置与 CCL 证据边界。arXiv 标题精确检索为 0，全文检索 198 篇均为外部使用/评测结果。研究笔记见 [`gemini-3.1-pro-preview-source-notes.md`](research/model-update-2026-09/gemini-3.1-pro-preview-source-notes.md)。当前不新增独立正式架构章节；下一锚点继续从两个排行榜的剩余重点厂商候选中选择。

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

## 2026-09-16 Claude Sonnet 4.6 断点恢复与后续顺序

昨晚 20:00—今早 09:00 的联网中断风险已按计划重新复验。三条代理对 Artificial Analysis 中文首页、DataCurve DeepSWE 和 Sonnet 4.6 发布页均返回 HTTP 200；两个排行榜快照逐字节一致。`8098/1234` 的 Anthropic 模型目录被区域页重定向，`7890` 成功取得真实 `platform.claude.com` Models Overview，因此按线路差异记录，不把单代理失败解释为页面不存在。

Claude Sonnet 4.6 已完成“排行榜锚点 → 官方发布页/模型目录 → System Card 入口 → Agent 文档 → 研究笔记”的资料级闭环。后续面试重点是 1M context 的有效容量、adaptive/extended thinking 的成本—延迟—成功率分层、compaction 状态协议、computer-use 执行器安全，以及 tool search 与上下文预算；参数、架构、训练 recipe、compaction 内部格式和独立 benchmark 仍待核验，不新增独立架构章节。

该段记录的是 2026-09-16 的历史顺序。随后 `gpt-6-astra` 和 `gpt-5.3-codex` 已完成对应阶段的榜单复核与 OpenAI 官方资料补证；当前锚点以本文件最后一节的最新记录为准，继续按“榜单配置 → 官方资料 → 技术拆解 → 面试映射 → 证据边界”的顺序推进。

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

## 2026-09-30 Kimi K3：DFlash draft / block diffusion 补证

两榜刷新后仍沿既有 `kimi-k3` 锚点推进。新增权威链为 DFlash v2 论文、HF 固定 revision/config/model card 与 SGLang PR #40794。重点面试点是 target 多层 hidden feature 的逐层 K/V 注入、block diffusion 的并行 draft、anchor/block sparse training mask、早位置 loss decay，以及 K3 layer-output capture 的索引语义。

同步范围已扩展至 K3 研究笔记、第二十一册第 88.26–88.27 节、PAPERS、INTERVIEW_BANK、EXERCISES、source-index、model-inventory、BOOK_SERIES、progress。状态为 **K3 内容专题闭环 + DFlash paper/HF/main source evidence**；未下载 draft 权重，SGLang stable 仍为 `v0.5.20`，PR benchmark 使用未公开 production draft，尚无本机 GPU、真实 acceptance、目标硬件和生产 SLO 验收。下一步先复核两榜 canonical 集合，再选择八家重点厂商中尚有官方技术缺口的锚点，goal 保持 `active`。
2026-10-01：Argon 官方资料已出现。Google The Keyword 公告形成发布方内容闭环；下一步只继续核验 Google 模型卡/API/技术报告/代码等更强资料，不把发布方评测升级为独立验收。正式章节为第二十一册第 95 章，配套题库、练习、术语、论文、项目、图谱和 inventory 已同步。


## 第二轮全系列精修执行计划（归档自 SECOND_PASS_REVIEW_PLAN.md）

## 1. 当前最紧急任务

第二轮优先级如下。

### 1.1 全书数学公式重写与修正

这是第二轮第一优先级。

目标：从头到尾遍历每本书，把所有数学式子、数学公式、变量说明和推导表达改写正确，并保证 GitHub Markdown 阅读效果良好。

重点问题：

- `\text{...}` 中包含下划线，导致 GitHub 渲染或阅读效果差。
- 公式中变量命名前后不一致。
- 公式缺少变量解释。
- 公式前后文字没有说明公式在解决什么问题。
- 数学表达过于跳跃，小白看不懂。
- 行内公式和块级公式混用不合理。
- 复杂公式没有配简单数值例子。

改写原则：

- 优先使用 GitHub Markdown 兼容写法。
- 避免在 `\text{...}` 内写包含下划线的变量名。
- 变量名优先使用 `\mathrm{...}`、普通数学变量或正文代码变量。
- 每个关键公式都要解释变量含义。
- 小白向章节中，公式后尽量补一句直觉解释。
- 专家向章节中，公式后补适用条件、边界和常见误解。

验收标准：

- 全书公式可读、可解释、变量一致。
- 没有明显影响阅读的 `\text{...}` 公式格式问题。
- 核心公式不只是摆出来，而是能帮助理解知识点。

### 1.2 全书补充 demo 级 Python 代码

这是第二轮第二优先级。

目标：从头到尾阅读所有章节，在所有适合增加代码的位置补充 demo 级 Python 代码，用最小可运行示例帮助小白理解知识点。

代码定位：

- 不是生产代码。
- 不是复杂框架工程。
- 是能复制运行、帮助理解当前知识点细节的教学 demo。

允许依赖：

- Python 标准库。
- `numpy`。
- `pandas`。
- 必要时可用 `matplotlib` 做简单可视化。
- 可以使用 `torch`，但不能用一个框架函数直接遮蔽知识点内部机制。如果一个知识点用 PyTorch 一个函数就能调用完成，但读者看不到内部细节，应优先用 `numpy` 或纯 Python 写出最小机制 demo，再用 PyTorch API 作为工程对照。

代码选择原则：

- 不为了使用 `numpy` 而使用 `numpy`，也不为了展示框架而使用 `torch`。工具选择服务于知识点理解。
- 核心目标是用最短、最聚焦的代码揭示当前知识点的内在实现逻辑。
- 如果 PyTorch 一个函数会把关键机制完全隐藏，且这个机制正是本节要解释的重点，应使用 `numpy` 或纯 Python 写最小版本。
- 如果手写机制会让代码明显冗长，反而分散读者注意力，应缩短 demo，只保留核心计算，或改用伪代码加少量可运行代码。
- 如果知识点重点是工程接口、shape、调用顺序或配置项，可以直接使用 PyTorch 示例。
- 高频核心知识点可以采用“最小手写机制 + PyTorch API 对照”，但不要机械地每个知识点都写两套代码。
- 每段 demo 都必须回答一个明确问题，例如“softmax 为什么要减 max”、“causal mask 如何阻止看未来”、“weight decay 为什么要解耦”。
- 代码服务理解，不追求把框架源码完整重写。

优先补代码的知识点：

- tokenization 的基本过程。
- attention 权重计算。
- softmax 数值稳定性。
- cross entropy。
- top-k / top-p sampling。
- temperature 对分布的影响。
- embedding 相似度检索。
- chunk 切分和 overlap。
- BM25 / hybrid retrieval 的直觉模拟。
- rerank 前后排序变化。
- RAG context assembly。
- LoRA 低秩矩阵直觉。
- KV cache 的形状和增长。
- prefill / decode 的 token 数差异。
- continuous batching 的调度模拟。
- prefix cache 命中收益模拟。
- TTFT / TPOT 统计。
- LLM-as-a-judge 的打分数据结构。
- bad case 分类统计。
- 成本估算和 token 预算。
- Agent 工具调用 schema 校验。
- prompt injection 简单防护示例。

代码要求：

- 每段代码必须能独立复制执行，或明确说明依赖输入。
- 代码前说明它演示什么。
- 代码后解释输出代表什么。
- 不写过长代码，优先 20-80 行以内。
- 对小白友好，变量名清楚。
- 不引入真实 API key。
- 不调用外部收费模型。
- 不需要复杂环境。

验收标准：

- 每个适合代码辅助理解的核心知识点，都有至少一个小 demo 或伪代码。
- 代码能自圆其说，和当前章节知识点强相关。
- 小白复制运行后，能更深入理解当前知识点。

### 1.3 全书大模型领域广度覆盖审计

这是第二轮第三优先级。

目标：检查 24 本主书和补充篇是否完整覆盖大模型相关领域，避免重要方向缺失或明显薄弱。

广度审计维度：

- Transformer 基础。
- 预训练。
- SFT。
- RLHF / RLAIF。
- DPO / preference optimization。
- 数据工程。
- tokenizer。
- scaling law。
- MoE。
- 长上下文。
- RAG。
- Agent。
- tool use / function calling。
- MCP / A2A / 工具协议。
- 多模态。
- Reasoning model。
- 评测。
- 安全与 alignment。
- 推理优化。
- serving engine。
- AI Infra。
- 训练系统。
- 分布式训练。
- 模型压缩与量化。
- 产品化与商业化。
- 系统设计面试。
- 求职与职业成长。

审计方式：

- 从目录和章节标题检查覆盖范围。
- 联网检索近年主流论文、框架、技术报告和官方文档。
- 对照当前主流大模型岗位 JD 和面试题。
- 标记缺失、过时、薄弱和重复主题。

产物：

- 广度覆盖表。
- 缺失主题清单。
- 需要补章、补节或交叉引用的位置。

验收标准：

- 读者沿本项目学习，能覆盖大模型算法岗、工程岗、推理岗、Agent 岗、AI Infra 岗的主流知识面。
- 不追求每个细分方向无限展开，但重要方向不能缺席。

### 1.4 每个知识点深度增强

这是第二轮第四优先级。

目标：每个知识点既要让小白知道“是什么”，也要让专家能看到“为什么、边界、取舍、细节和追问点”。

每个重要知识点建议补齐以下层次：

1. 它是什么。
2. 为什么需要它。
3. 它解决了前人方法的什么问题。
4. 核心机制是什么。
5. 一个最小例子或代码 demo。
6. 它的优点是什么。
7. 它的限制和失败场景是什么。
8. 工程实现时注意什么。
9. 面试中会如何被追问。
10. 和相邻概念的区别是什么。

重点增强方向：

- AdamW、Muon 等优化器。
- normalization、position encoding、attention 变体。
- LoRA、QLoRA、DPO、RLHF。
- RAG 检索、rerank、citation、评测闭环。
- Agent 工具调用、安全、状态管理、任务评测。
- 推理框架、KV cache、PagedAttention、batching、PD 分离。
- 评测指标、LLM-as-a-judge、数据污染、统计显著性。
- 安全、alignment、prompt injection、权限边界。
- 成本、容量规划、SLO、线上事故。

验收标准：

- 小白读完能理解基本概念。
- 中级工程师能知道工程落地怎么做。
- 专家能看到边界、取舍和更深追问点。

### 1.5 全程联网辅助知识扩展

这是第二轮第五优先级，也是执行约束。

目标：第二轮不能只凭已有知识改写。每一册、每个重要主题的精修，都要先联网检索辅助资料，再进行改写。

优先资料来源：

- 官方论文。
- arXiv 论文。
- 技术报告。
- 官方文档。
- GitHub 仓库。
- 框架文档。
- 标准协议文档。
- 权威工程博客。
- 主流开源项目 README 和 docs。

重点资料来源示例：

- OpenAI、Anthropic、Google DeepMind、Meta AI、Microsoft、NVIDIA、Hugging Face。
- PyTorch、Transformers、vLLM、SGLang、TensorRT-LLM、DeepSpeed、Megatron-LM。
- LangChain、LlamaIndex、DSPy、OpenAI Agents SDK、MCP 官方资料。
- OWASP LLM Top 10、NIST AI RMF。

执行要求：

- 每次精修某一册或某一主题前，先检索对应权威资料。
- 正文不堆链接，但观点要经过资料校验。
- 如果资料显示当前内容过时，要修正。
- 如果多个资料观点不同，要说明版本、场景和边界。

验收标准：

- 第二轮内容不是闭门造车。
- 重要技术点能跟上当前主流实践。
- 新增内容有更强深度和广度。

## 2. 第二轮执行顺序

第二轮按以下顺序推进。

### 阶段一：公式全量修正

从第一册开始，到第二十四册和补充篇结束。

逐章检查：

- 数学公式。
- 行内公式。
- 变量说明。
- 推导步骤。
- GitHub Markdown 兼容性。

阶段产物：

- 所有明显公式问题修正。
- 记录仍需专家复核的复杂公式。

### 阶段二：代码 demo 补充

从第一册开始，逐章判断是否需要补 Python demo。

补充优先级：

1. 小白难理解但适合代码演示的知识点。
2. 高频面试知识点。
3. 工程实践中容易误解的知识点。
4. 可以用 numpy/pandas 简洁表达的概念。

阶段产物：

- 各册新增可运行教学 demo。
- 必要时同步 `EXERCISES.md`。

### 阶段三：广度覆盖审计

对全书目录和主题做覆盖检查。

阶段产物：

- 标记缺失主题。
- 决定是补章、补节、补引用，还是放入后续版本。

### 阶段四：深度增强

对薄弱章节补充：

- 历史背景。
- 动机。
- 机制。
- 代码 demo。
- 工程边界。
- 面试追问。
- 专家视角。

阶段产物：

- 重点章节完成第二版精修。

### 阶段五：联网资料校准与补充

这个阶段贯穿前四阶段，但最后要整体复查。

阶段产物：

- 过时内容修正。
- 新兴方向补充。
- 官方资料和主流实践对齐。

### 阶段六：索引、题库、练习、百科同步

在正文修改后同步：

- `INTERVIEW_BANK.md`
- `EXERCISES.md`
- `GLOSSARY_EN_ZH.md`
- 第四册百科。
- `PROJECTS.md`
- `KNOWLEDGE_GRAPH.md`

阶段产物：

- 正文、题库、练习、百科互相一致。

### 阶段七：目录链接、交叉引用和重复压缩

最后做结构性收口。

阶段产物：

- 修复目录链接。
- 增加跨书引用。
- 压缩重复内容。
- 更新 `README.md`、`BOOK_SERIES.md`、`PROGRESS.md`。

## 3. 每章精修检查模板

第二轮处理每一章时，按以下模板检查。

```text
章节：
是否联网检索：是 / 否
参考资料：
公式问题：
需要新增代码 demo 的位置：
已有代码是否可运行：
小白解释是否足够：
专家深度是否足够：
是否缺少工程边界：
是否缺少面试追问：
是否需要同步题库：
是否需要同步练习：
是否需要同步百科：
是否存在重复内容：
最终处理结果：
```

## 4. 每次修改后的检查

每次批量修改后必须检查：

- 新增公式是否可读。
- 新增代码是否语法正确。
- 是否引入真实 key、内部数据或敏感信息。
- 是否出现广告、推广、群号、无关链接、base64 注入。
- 是否需要同步题库、练习、百科。
## 第二轮全系列改写与精修计划（归档自 SECOND_PASS_REVIEW_PLAN.md）

本计划适用于 24 本主书和 `book-llm-engineer` 补充篇的第二轮修改。第一轮目标是把内容写完整；第二轮目标是把内容写正确、写深、写广，让小白能读懂、专家能看到深度。本轮不仅检查目录和格式，还要从头到尾重读全书，逐章进行公式修正、代码补充、联网校验、广度审计和深度增强。

本文件是 24 本主书和 `book-llm-engineer` 补充篇进入第二轮修改的总控计划。第一版目标是“把内容写完整”；第二轮目标是“把内容写正确、写深、写广、写得小白能读懂、专家也能看到深度”。本轮不只是做目录和格式检查，而是从头到尾重读全书，逐章进行公式修正、代码补充、联网校验、广度审计和深度增强。
