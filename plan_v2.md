# 大模型资料更新计划 v2

更新时间：2026-09-23

## 目标

以 2025 年以来新发布或快速进入主流评测的新模型为锚点，更新本项目的模型谱系、技术知识、论文资料和面试训练内容。新文件是当前执行计划的唯一入口；旧版 `plan.md` 保留作历史参考。自 2026-09-14 起，新模型候选的发现入口严格收敛为 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜。

## 当前重点厂商范围

主动关注范围仅包括：OpenAI、Anthropic、Z.ai（GLM）、Qwen、Kimi/Moonshot、DeepSeek、Google（Gemini）和 xAI（Grok）。其他厂商及其新模型暂不主动检索、核验或扩写；已有历史记录保留，但不进入下一轮更新队列，除非用户明确通知需要关注。

## 锚点研究原则

后续以两个排行榜中近期出现、排名靠前或在 Agent/推理/长上下文/多模态方向具有代表性的模型作为热点锚点，不追求把八家厂商的所有历史版本和长尾模型列全。对每个锚点，先归并 `effort`、fallback、provider 和 Agent harness 配置，再沿模型的官方博客、技术文档、模型卡、技术报告、论文、代码仓库和 API 文档追踪新技术、方法和知识点；这些权威资料用于扩展锚点，不作为新的模型发现入口。

“闭环”按内容阶段判断：排行榜发现、权威来源、研究笔记、正式章节和配套同步均具备，标为“内容专题闭环”；已有权威来源、研究笔记和配套同步但尚无独立正式专题，标为“资料级闭环”；只有部分官方入口或被系列章节覆盖，标为“部分覆盖”；只有排行榜条目，标为“仅候选”。任何状态都不代表参数、训练配方、内部架构或独立 benchmark 已全部公开确认。

上一阶段活动锚点（2026-09-23）为 `GPT-6 Sol`。其 AA 单榜资料级闭环已完成；本轮保留它的长上下文、mode/effort、工具和 compaction 证据，不把 Sol 的 provider 指标或 Agent 结果迁移到 GPT-6 Luna。

上一轮活动锚点为 `GPT-6 Luna`。该轮只沿两个榜单已经出现的 `gpt-6-luna` canonical 条目推进，不从 OpenAI 官方模型目录另发现其他模型；Luna 的 focused/high-volume 定位、1.05M/922K/128K、effort、runtime ownership 和 opaque compaction state 已按服务合同入库。

上一轮活动锚点为 `Claude Opus 5.5`。本轮只沿两个榜单已经出现的 `claude-opus-5-5` canonical 条目推进，不从 Anthropic 官方目录另发现 Sonnet 5.5/Haiku 5.5 等模型；System Card 正文的评测条件、安全路由、prompt injection、OSWorld compaction 和多 Agent derived latency 已收口。

上一阶段活动锚点为已有排行榜条目 `DeepSeek V4.1-Flash`。本轮已完成官方 API/provider contract 和标准库协议 toy；完整权重、量化/召回、目标硬件、DSpark acceptance、工具验收和生产 SLO 仍未通过门禁。DataCurve 当前没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，不迁移 DeepSeek V4 Pro、V4 Flash 或其他 DeepSeek 版本的 Agent 成绩。

当前活动锚点切换为已有排行榜条目 `Gemini 3.8 Flash`。本轮只沿两个排行榜已经出现的 canonical 模型推进，不从 Google 官方模型目录另发现模型；重点是 2026-09-23 Thinking/Interactions/Tool combination 文档的 state、signature、预算、工具上下文和生命周期合同。`low/medium/high` 仍归并为一个基础模型，DataCurve 只引用精确 high 行，不迁移相邻 Gemini 或其他 effort 的 Agent 结果。

当前活动锚点切换为已有排行榜条目 `Kimi K3`。本轮只沿两个排行榜已经出现的 canonical 模型推进，不从 Kimi 官方仓库、HF、vLLM 或 SGLang 另发现模型；重点是榜单当前时点复验、官方 artifact revision 是否漂移、DataCurve harness 账本和 K3 发布限制的面试表述。`low/max` 仍归并为一个基础模型；DataCurve 只引用精确 `mini_swe_agent_kimi_k3_max` 行，不迁移其他 Kimi 或相邻 effort 的 Agent 结果。

Claude Opus 5.5、DeepSeek V4.1-Flash 和 GPT-6 Sol 仍作为已完成阶段的活动记录保留：本轮不因 OpenAI 文档目录或 runtime 文档另发现模型；GPT-6 Luna 的参数、架构、完整训练 recipe、完整权重、目标硬件 profiling、独立 benchmark 和 production acceptance 仍未通过门禁。

当前活动锚点切换为已有排行榜条目 `Claude Fable 5.1` 的 System Card 正文证据复核。本轮不从 Anthropic System Card、Research 页面或外部论文另发现模型；只把这些权威资料用于扩展 AA 已发现的 canonical 条目。Fable/Mythos 的 shared weights、safeguards、训练数据边界、RSP、安全评测和 fallback 条件均按证据责任分层，DataCurve 缺少精确 Fable 5.1 行时继续保持 `not_applicable`。

## 工作流

最新活动锚点覆盖（2026-09-22）：本轮重新取得 Artificial Analysis GPT-5.6 Luna 详情、DataCurve 和 OpenAI 官方 Markdown 页面。AA 详情为 `3,861,318` bytes / `1b425d8f2a93f8418702ac7acbb62b26850b98fe20680f82205bed5fce38fed0`，DataCurve 为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；精确行仍为 `mini_swe_agent_gpt_5_6_luna_max`，301/448、Pass@1 `67.1875%`、Pass@4 `90.26548672566371%`。AA 当前速度为 `158.728370482714 tokens/s`、Intelligence task cost `0.17829726152289094`、1M context；与 9 月 21 日 provider 测量分开保存，不解释为模型升级。7890 取得 Luna、Reasoning、Agents、Tools、Tool Search、Prompt Caching 和 Compaction 官方页 HTTP 200，哈希详见研究笔记；GPT-5.6 保持资料级闭环，不新增重复 Transformer 正式章节。

当前执行补充（2026-09-22）：GLM-5.3-Flash 的 SGLang `v0.5.20` tag 已确认 `glm5_next.py` 与配置入口，SGLang `main` 继续出现 projection/KDA/mHC/AMD 量化演进；vLLM `main` 已有 `vllm/models/glm5next/`，可见 indexer pool/tail cache、KDA state 和 MTP top-k/slot mapping。对照 vLLM `v0.29.0` tree 未发现该专属目录，因此后续 gate 必须分别记录 stable source、mutable main、recipe、wheel、完整权重和目标硬件结果，不能把 `v0.29.0+` recipe 门槛当作 stable tag 已包含 GLM5Next。

当前执行补充（2026-09-22，标准版）：新鲜两榜快照没有产生八家重点厂商的新 canonical 模型；本轮转入已有双榜条目 `GLM-5.3`。HF revision `aca966e4e02791568aa6a4ced368624b3d897f42` 的 `glm_moe_dsa` 配置与 Transformers main 的 `GlmMoeDsaForCausalLM` 对齐；vLLM `v0.29.0`/main 和 SGLang `v0.5.20`/main 都沿 DeepSeek-V3.2 DSA 共用路径承载该入口。`sglang/srt/models/glm5_next.py` 属于 Flash/linear-attention 变体，不能迁移为标准 GLM-5.3 的实现证据。后续 gate 分别记录 DSA indexer/MLA 源码、完整权重加载、目标硬件 profiling、index/evidence recall、MTP acceptance、tool/verifier 和生产 SLO。

当前执行补充（2026-09-23，DeepSeek V4.1-Flash）：vLLM `v0.30.0` 于 `2026-09-22T05:20:54Z` 发布，tag commit 为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。固定 release API、registry、PyPI `vllm/0.30.0` metadata 和 NVIDIA/ROCm V4.1/DSpark package 文件，确认 stable release 已有专用 release surface；v0.29.0 保留为历史对照，SGLang stable 仍为 `v0.5.20` 且本轮未证明其 V4.1 专用 vision/runtime 已进入 stable。后续 gate 仍按完整权重、GPU/ROCm/NPU、真实 candidate/index recall、FP4 误差、DSpark acceptance/rollback、EPD、tool/verifier 和生产 SLO 分栏。

当前网络复访记录（2026-09-23）：三条代理先前短时失败后已恢复；`10.237.126.170:1234`、`10.24.27.134:7890`、`10.24.27.134:8098` 访问百度均 HTTP 200，访问 Artificial Analysis 中文首页和 DataCurve DeepSWE 也均 HTTP 200 且逐字节一致。最新续抓 AA 首页为 `1,783,592` bytes / SHA-256 `7880a715fd069d4e475e4d717b37971d1f6402e5492964bb3f58787524d0b5a4`，此前同日 `1,783,001` bytes / `2fd4bd...` 作为历史快照保留；DataCurve 为 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。本轮据此新增 `Claude Opus 5.5` 候选并切换活动锚点；仍按 canonical slug、effort/provider/harness 和快照哈希分账。

当前执行补充（2026-09-23，Claude Opus 5.5 官方 API contract）：7890 代理取得 `overview.md`、`whats-new-opus-5-5.md`、`migration-guide.md` 和 fast mode 官方 Markdown。已确认 `claude-opus-5-5`、1M/128K、always-on adaptive thinking、default `medium` effort、thinking block model/conversation binding、强制 tool choice 400、`computer_toolset_20260801` 迁移、progress-update blocks、on-demand compaction、inline tools 和 fast mode 的 `usage.speed`/独立限流。1234/8098 仍会把平台文档重定向到区域不可用页，因此按线路分别记录，不把区域差异写成模型差异。

当前执行补充（2026-09-23，Claude Opus 5.5 System Card 正文）：固定 PDF `17,795,106` bytes / SHA-256 `7311c9c6bbb16d012f1c12c7418b05949fcf7ae3e30d2c40f22050074b2a7378`，已用 Node.js `zlib` 与 ToUnicode/CMap 只读解析。训练数据边界为公开互联网、公开/私有、获许可用户和合成数据，knowledge cutoff 为 2026-06；评测多数使用最终 snapshot，但部分章节使用早期/替代 snapshot 或关闭生产 safeguards。新增并入库 Terminal-Bench 4.0 `66.36%`（xhigh、Claude Code `--bare`、5 trials）、ProgramBench `91.2%`（166 tasks）、OSWorld 2.0 partial/strict `81.8%/48.7%`（108 tasks、1080p、500 actions、5 runs、>100K tokens compaction）、CoBench 2.1 `55.8%`、AECI `169.36`、Cyber/Gray Swan/Alignment 数字，以及五 Agent team/DRACO 的 derived-latency 结果。Cyber 数字绑定关闭 safeguards，fallback、simulated environment、long trajectory 和 multi-agent caveat 均已写入证据边界。

当前执行补充（2026-09-23，下一锚点选择）：最新 Artificial Analysis 中文首页续抓为 `1,783,592` bytes / SHA-256 `7880a715fd069d4e475e4d717b37971d1f6402e5492964bb3f58787524d0b5a4`，DataCurve 为 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；首页前列的 `DeepSeek V4.1 Flash (max)` 已有 canonical 条目，AA 图表第三方 Intelligence Index 为 `39.456167472527`。八家重点厂商没有新的模型名称，故切换到已有 `deepseek-v4-1-flash`，不从 runtime 仓库另建模型。

当前执行补充（2026-09-23，GPT-6 Sol）：三条代理取得 AA `gpt-6-sol` 详情页逐字节一致，`547,729` bytes / SHA-256 `88841ae9837f189a6ab739d8fa2bad6d142163b5b6486cab9181ebbfbc75f116`；max Intelligence Index 为 `47.5276426437724`。DataCurve 当前没有精确 `mini_swe_agent_gpt_6_sol_*` 行。OpenAI 官方模型页及 Reasoning、Agents、Tools、Compaction 快照已固定，确认 `gpt-6-sol`、1.05M/922K/128K、`none`--`max` effort、Responses/Chat Completions function-calling 边界、工具目录、GPT-6 family 的 mode/effort 分离与 `configuration_update`、Agent runtime 所有权和 compaction replay 规则；这些是 API/runtime 证据，不是内部架构披露。

当前执行补充（2026-09-23，GPT-6 Luna）：三条代理取得 AA `gpt-6-luna` 详情页逐字节一致，`3,992,084` bytes / SHA-256 `cda7ec1a8b90bfb76c85624f31cab242a0779732f68d65641b0dee5929a5d0f1`；release date `2026-09-22`、max Intelligence Index `37.2559686869738`、median output speed `153.87508473888 tokens/s`。DataCurve 当前没有精确 `mini_swe_agent_gpt_6_luna_*` 行。7890 取得 OpenAI 官方模型页 `4,019` bytes / SHA-256 `561a86af72eced9a76a4e3a96c45ae7fff7782732514a8429b55ebddb4404945`，确认 Luna 的 focused/high-volume 定位、1.05M/922K/128K、2026-05-18 knowledge cutoff、`none`--`max` effort、Responses/Chat Completions/Batch、工具目录和 `$0.10/$0.50` token 价格合同；1234/8098 返回 403，仅记录为线路边界。Luna 与 Sol 共享的 reasoning/Agents/tools/compaction 快照仅作为 GPT-6 family runtime 证据复用，不迁移 Sol 的指标或 Agent 结果。

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
- [Claude Fable 5.1 官方资料摘记](research/model-update-2026-09/claude-fable-5.1-source-notes.md)：已复验 Artificial Analysis 榜单、确认 DataCurve 暂无 Fable 5.1 行，并补齐 Anthropic 发布页、System Card 入口、Fable/Mythos safeguards、cache read 定价、发布方 benchmark、安全摘要和 arXiv 外部使用检索；当前为资料级闭环。
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
- Gemini 3.8 Flash 已完成榜单发现、官方资料入库和周边技术提取，状态为“资料级闭环”。由于没有独立公开的 3.8 架构/训练报告，暂不新增专属正式章节，先映射到 Reasoning、Agent/工具协议、Computer Use、长上下文/Serving 与评测章节。
- [Grok 4.5 官方资料摘记](research/model-update-2026-09/grok-4.5-source-notes.md)：已核验两个排行榜的 `high` 配置、xAI 发布公告、Grok 4.5 模型页、Reasoning/Responses/Compaction/Tools/Web Search/X Search/Remote MCP 文档；当前为资料级闭环。
- [Grok 4.6 官方资料摘记](research/model-update-2026-09/grok-4.6-source-notes.md)：已核验两个排行榜的四档 effort、xAI 发布公告、官方模型页、Reasoning/Compaction/Tools 文档和论文精确标题检索；2026-09-21 已追加当前 AA/DataCurve 快照复验和 xAI 三代理失败边界；当前为资料级闭环。
- [Grok 4.20 官方资料摘记](research/model-update-2026-09/grok-4.20-source-notes.md)：已核验 Artificial Analysis 精确条目、DataCurve 精确行缺失、xAI 模型页/模型注册表、Reasoning/Multi Agent/Compaction/Tools/Release Notes 和 arXiv/新闻入口负证据；当前为 AA 单榜资料级闭环。
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
