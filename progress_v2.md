# 大模型资料更新进度 v2

更新时间：2026-09-16

## 当前状态

- 阶段：模型候选核验、专题写作与纵向同步（持续进行）
- 主线：新模型发现 → 官方资料核验 → 技术知识映射 → 书系同步 → 代码/链接/证据审计
- 规划文件：[`plan_v2.md`](plan_v2.md)
- 旧版进度：`PROGRESS.md`，仅作历史记录

## 已完成

- 明确以新发布模型和公开排行榜作为资料更新入口。
- 建立模型记录模板、来源优先级和书系同步范围。
- 建立首批待核验模型候选清单，包含 GPT、Claude、Gemini、DeepSeek、Qwen、Kimi、GLM、Llama、Mistral、Grok 等系列。
- 已完成 GPT-6 Astra 与 GLM-5.3 两条可追溯专题闭环：官方文档摘记、正式章节、目录、百科、术语、题库、练习、论文、项目和知识图谱均已同步。
- 已核验新增章节的 Python 示例、AST、围栏配对、相对链接和 `git diff --check`；未公开架构、参数量、发布日期和 SAO 机制均保留为待核验。
- 已完成 K2 Horizon MoVA 36B/A4B 的榜单发现、官方模型卡/固定 revision 核验、第二十一册第 82 章和全局资料同步；K2-Horizon-7B-Uno 作为官方关联 adapter/论文技术记录，不作为排行榜新增模型。
- 已完成 Qwen3.8 的榜单归并、官方 27B/A95B/Flash-Next 模型卡、Flash-Next GitHub/技术报告和 Qwen Cloud 关系核验；新增研究笔记、第二十一册第 83 章及书系配套同步。Qwen3.8 当前状态为“内容专题闭环”。
- 已完成 Gemini 3.8 Flash 的两榜单归并、Google 官方模型页、DeepMind Model Card、发布博客、评测 PDF 及 Gemini API 周边文档核验；新增研究笔记并完成索引/清单/计划同步。Gemini 3.8 Flash 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Gemini 3.7 Flash 的夜间断点联网复验：重新获取 Artificial Analysis、DataCurve、Google AI Developers 模型页、DeepMind Model Card、DeepMind Research/可用 publications 入口和 arXiv 精确/全文检索；新增研究笔记并完成来源索引、模型盘点和计划同步。Gemini 3.7 Flash 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Gemini 3.6 Flash 的夜间断点恢复与资料级闭环：重新获取 Artificial Analysis 详情/provider 页、DataCurve、Google AI Developers 模型页、DeepMind Model Card、Thinking/Interactions/工具组合/视频/缓存/Computer Use 文档和 arXiv 精确/全文检索；新增研究笔记并完成来源索引、模型盘点、榜单解释和计划同步。Gemini 3.6 Flash 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Gemini 3.5 Flash 的断点恢复与资料级闭环：重新获取 Artificial Analysis、DataCurve、Google AI Developers 模型页、What's New、DeepMind Model Card、Gemini 3 Flash 依赖 Model Card、Thinking/Interactions/工具组合/视频/缓存/Computer Use 文档和 arXiv 精确/全文检索；新增研究笔记并完成来源索引、模型盘点、榜单解释和计划同步。Gemini 3.5 Flash 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 GPT-5.6 Sol/Terra/Luna 的两榜单归并、三个 OpenAI 官方模型页、Reasoning/Agents/Tools/Prompt Caching/Compaction 文档和开发者博客核验；新增研究笔记并完成索引/清单/计划同步。GPT-5.6 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 GPT-5.5/GPT-5.5 Pro 的两榜单锚点归并、OpenAI 两个官方模型页、GPT-5.5 专属指南及 Reasoning/Tools/Tool search/Prompt Caching/Compaction/Images/Conversation state/Background 文档核验；新增研究笔记并完成索引/清单/计划同步。GPT-5.5 当前状态为“资料级闭环”，暂无独立正式章节；GPT-5.5 Instant 仍是仅榜单级关联条目。
- 已完成 Claude Fable 5 的 Artificial Analysis/DataCurve 锚点归并、Anthropic 官方模型页、发布/重新部署公告、API/Agent 文档和 system card 入口核验；新增研究笔记并完成来源索引、模型盘点、榜单解释、计划和进度同步。Claude Fable 5 当前状态为“资料级闭环”，暂无独立架构章节。
- 已完成 Claude Opus 4.8 的 Artificial Analysis/DataCurve 锚点归并、Anthropic 发布公告、System Card 入口和 Dynamic Workflows 官方博客核验；新增研究笔记并完成来源索引、模型盘点、榜单解释、计划和进度同步。Claude Opus 4.8 当前状态为“资料级闭环”，暂无独立架构章节。
- 已完成 Kimi K2.7 Code 的 Artificial Analysis/DataCurve 锚点归并、Moonshot/Kimi 官方资源/API 文档、固定 revision 模型卡、配置、许可证和部署指南核验；新增研究笔记并完成来源索引、模型盘点、榜单解释、计划和进度同步。Kimi K2.7 Code 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Grok 4.5 的 Artificial Analysis/DataCurve 锚点归并、xAI 发布公告、Grok 4.5 官方模型页、Reasoning/Responses/Compaction/Tools/Web Search/X Search/Remote MCP 文档核验；新增研究笔记并完成来源索引、模型盘点、榜单解释、计划和进度同步。Grok 4.5 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Grok 4.6 的 Artificial Analysis/DataCurve 四档锚点归并、xAI 发布公告、Grok 4.6 官方模型页、Reasoning/Compaction/Tools 文档和论文精确标题检索；新增研究笔记并完成来源索引、模型盘点、榜单解释、计划和进度同步。Grok 4.6 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Claude Opus 5 的 Artificial Analysis/DataCurve 锚点归并、Anthropic 官方模型目录/专属页、完整开发者文档、发布公告、System Card 下载和 Anthropic Research/arXiv 定向检索；研究笔记、来源索引、模型盘点、候选解释、计划和进度已同步。已确认 adaptive thinking、五档 effort、工具/effort 中途变更、512-token 缓存门槛、fallback、1M context 和配置级评测边界；Claude Opus 5 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Claude Fable 5.1 的断点恢复核验：重新抓取 Artificial Analysis、DataCurve、Anthropic 发布页和 System Card，补齐 Fable/Mythos safeguards 分层、cache read 定价、产品入口 effort、发布方 benchmark/安全摘要及 arXiv 外部使用检索；确认 DataCurve 当前没有 Fable 5.1 行。Claude Fable 5.1 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Claude Sonnet 5 的断点恢复核验：重新获取 Artificial Analysis、DataCurve、Anthropic 发布页、System Card、完整开发者文档、Research 页面和 arXiv 精确标题检索；补齐 adaptive thinking、effort/`max_tokens`、thinking block/signature、new tokenizer、context awareness、server-side compaction、computer toolset、programmatic tool calling、System Card/发布方评测及负面证据。Claude Sonnet 5 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 DeepSeek V4 Flash Vision 的断点恢复核验与资料级闭环：重新抓取 Artificial Analysis、DataCurve 和 DeepSeek Quick Start；确认两个可用代理对 AA 返回相同完整页面，三个代理对 DataCurve/Quick Start 返回相同完整页面，7890 对 AA 大页面仍 TLS EOF。已核验历史实验多模态 API、当前旧 alias → V4.1-Flash 路由、图像预算、Files `file_id`、Responses 图像工具回灌和证据边界；第 85 章 demo AST、正常执行及 9 组边界测试通过。
- 已完成 GPT-5.4 的断点恢复核验与资料级闭环：通过三个代理重新抓取两个排行榜，并通过 7890 实际抓取 GPT-5.4 详情页、官方模型页和专属指南；确认三条线路对排行榜返回相同完整页面，AA 详情与既有哈希一致，DataCurve 仍无 GPT-5.4 之外的对应 Pro/mini/nano 主表行。已提取 1M context、deferred `tool_search`、computer use、native compaction、custom tools/CFG、`allowed_tools`、`phase` 和 Responses 状态等面试主线；暂无独立正式章节。

## 进行中

- 只从 DataCurve DeepSWE 与 Artificial Analysis（含 `https://artificialanalysis.ai/zh`）两个排行榜记录新进入者；其他网站不再作为新的模型发现入口。
- 对已由两个排行榜发现的候选模型寻找官方发布、论文/技术报告、模型卡、代码仓库和 API 文档，用于事实核验与周边技术扩展。
- 区分模型页、API 文档、开发者博客和榜单快照的证据责任，避免把产品/运行时字段写成内部训练事实。
- 当前锚点 `claude-sonnet-4-6` 已完成资料级闭环；下一锚点继续从两个排行榜剩余重点条目选择。GPT-5.4 已完成闭环，Pro 作为关联服务档位保留，mini/nano 尚未独立闭环；已入库锚点只补版本变更、接口差异、完整 kernel/硬件 profiling、线上接受率和独立评测复现。
- 对 K2 36B/A4B 继续补齐完整训练报告、生产 kernel、Uno 接受率和目标硬件 profiling；这些是已发现锚点的后续核验，不产生新的模型候选入口。

## 待完成

- 对仍处于“仅候选”“部分覆盖”或“资料级闭环”的重点锚点继续补齐官方来源、研究笔记和专题内容。
- 对已闭环锚点继续补证完整 kernel、目标硬件 profiling、线上接受率、完整训练/后训练配方和独立 benchmark，不把资料缺失误写成已确认事实。
- 完成全仓库链接、版本、许可证、时间语境和 Markdown/Python 代码质量检查。
- 继续按可用代理复访官方页面，重新确认价格、版本快照、模型卡、许可证和技术报告；单个代理的 TLS/DNS 失败不能被写成资料不存在。

## 记录规则

事实必须有来源和核验日期；未确认内容标为“待核验”，不进入正式结论。每完成一个模型或一个专题，立即在本文件追加条目。

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

- 已将用户强调的篇幅、逐知识点展开、生动例子、定义、解释、代码和分层深度要求写入 `plan_v2.md`。
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
- 仍待同步第四册百科、`KNOWLEDGE_GRAPH.md`、`PROJECTS.md`、`PAPERS.md` 和 `PROGRESS.md` 的正式索引，并完成 Markdown 数学渲染检查。

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

- 修正路径类型后重新检查新增章节、百科条目、plan_v2 和 progress_v2；本地相对链接缺失数为 0。
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
- 已同步 README、`BOOK_SERIES.md`、`ROADMAP.md` 和 `PROGRESS.md` 的新模型阅读入口与完成状态；全目标仍保持 active，候选模型的后续官方核验未结束。
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
- 已新增 `research/model-update-2026-09/claude-sonnet-5-source-notes.md`，并将模型字段同步到来源索引、候选解释、第四册百科、术语、面试题、练习、项目、论文、知识图谱以及 README、`BOOK_SERIES.md`、`ROADMAP.md`、`PROGRESS.md`、`progress_v2.md` 和 `plan_v2.md`。
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

## 2026-09-10：第六册第 4 章解码策略收口

- 第六册第 4 章 `book-06-llm-deployment/chapters/04-解码策略与生成控制.md` 已完成中断处的第二轮收口，当前 828 行、47080 字节；补充 speculative decoding 残差分母为零、PassRate 无有效 run 的 `not_applicable` 语义。
- 零依赖 logits demo 拒绝空/非有限 logits、非法 temperature、top-k/top-p/min-p、负 penalty、未知历史 token、越界采样索引和非法 step；内置 6 个非法配置回归，额外通过 10 个边界输入测试，预期输出同步 `invalid_cases`。
- 本轮对第 4 章及此前 10 个专题章节的 11 个 Python 围栏完成 AST 与独立运行；GenerationConfig、vLLM sampling/structured outputs 与 speculative decoding 论文在线复访因 DNS 解析失败，继续按待核验处理。

## 2026-09-11：第二十册第 5 章文件编辑审计边界收口

- 继续会话 `01a0843d-2910-7993-840a-8587cea054bd`，在 `book-20-agent-harness-runtime/chapters/05-文件系统与代码编辑.md` 增加空集合/空分母回归：`safe_div(1, 0)`、`safe_mean([])`、`rounded(None)` 为 `None`，比较门禁对 `None` 一律返回失败，避免无样本伪造 100% 通过。
- 正文新增 `not_applicable` 语义说明，明确不使用 epsilon、历史分数或默认 `1.0` 掩盖零分母；需补样本或由上层策略显式跳过整组评估。
- 章节 1106 行、35957 字节；Python fenced block 已通过 AST、独立执行和边界断言，正常 demo 输出与既定失败门禁保持一致。下一步继续第二十册第 6 章“终端执行与命令安全”。

## 2026-09-11：第二十册第 6-8 章审计指标边界收口

- 第 6 章终端执行 demo 去除 `safe_div` 默认 `1.0`，补充 `safe_mean`、空分母回归和 `not_applicable_metrics`；第 6 章当前 1111 行、37667 字节，AST、独立执行和边界断言通过，危险命令放行/网络未控/超时未取消/trace 缺失坏例保持使门禁失败。
- 第 7 章上下文 demo 将 `stale_summary_rate` 等空分母指标改为 `None` 语义，补充空集合回归与显式门禁比较；当前 1094 行、35067 字节，AST、独立执行和边界断言通过，缺失关键上下文、过期摘要、信号丢失和注入边界坏例保持可见。
- 第 8 章权限沙箱 demo 将空集合默认 1.0 改为 `None`，新增 `not_applicable_metrics` 和阈值安全比较；当前 1116 行、33545 字节，AST、独立执行和边界断言通过，矩阵缺行、网络外发、沙箱缺失及 dry run 缺失坏例保持可见。
- 第 5-8 章入口已在第二十册目录确认；下一步继续第 9 章 trace、日志回放与可观测性，并在阶段收口重跑全库围栏、相对链接和 `git diff --check`。

## 2026-09-11：第二十册第 9 章 Trace/Replay 指标边界收口

- 第 9 章 trace/replay demo 增加 `mean_defined` 与空集合回归：无 artifact 需求或无有效事件时返回 `None/not_applicable`，不再用 `mean([])=1.0` 伪造完整覆盖率；门禁对 `None` 显式失败。
- `artifact_reference_missing` 根因判断改为仅针对有定义且低于 1 的指标；唯一 Python 围栏通过 AST、独立运行和边界断言，既有 span 树、时间线、版本、隐私、回放和最终状态坏例保持可见。
- 章节当前 1246 行、37444 字节；下一步继续第二十册第 10 章 Evaluation Harness，并在阶段收口重跑全库检查。

## 2026-09-11：第二十册第 10 章 Evaluation Harness 指标边界收口

- 第 10 章 demo 将 `mean([])` 改为 `None`，`weighted_mean` 对零总权重返回 `None`，新增空集合回归、`at_least`/`at_most` 门禁和 `not_applicable_metrics` 输出；无有效 run 不再伪造成功率或安全零率。
- 唯一 Python 围栏通过 AST、独立运行和边界断言；既有环境/验收器/公平比较/flaky/安全执行/报告缺失坏例保持可见，`evaluation_harness_gate_pass=False`。
- 章节当前 1211 行、33865 字节；下一步扫描第二十册剩余章节并继续修复同类指标定义域问题。
- 2026-09-11：第二十册第 11、12、14、15、16 章统一收口空分母指标：`ratio`、`rate`、`avg` 对空样本返回 `None`，第 14 章 `a2a_lifecycle` 空集合不再除零；五章加入 `at_least`、`not_applicable_metrics` 与空集合回归。五个 Python fenced blocks 均通过 AST、独立运行和边界断言；第 17-22 章未发现同类评估代码或定义域问题。

## 2026-09-11：第二十册指标公式定义域复核

- 复读第 11、12、14、15、16 章指标公式，补充统一非空定义域：空集合或零分母均为 `not_applicable`（代码 `None`），不得用 epsilon、历史值、默认值或人为下限分母伪造指标；门禁显式拒绝 `None`。
- 第 14 章高风险覆盖率改为真实风险样本数分母，空风险集合不适用；A2A lifecycle 明确要求非空 agent 和状态需求集合。其余四章补齐 `N`、分组集合和必需 trace 集合的定义域说明。
- 验证结果：第二十册 16 个 Python 围栏 AST/独立运行全通过；全库 659 个 Markdown 文件的 9622 对围栏配对通过，1290 个 Python 围栏 AST 全通过；`git diff --check` 通过。链接扫描保留既有代码样式误报，不修改无关内容。

## 2026-09-14：第二十册剩余指标定义域收口

- 第 13 章横向 Coding Agent 审计把 `C_risk` 改为真实 `|R_i|` 分母（要求 `|R_i|>0`），所有覆盖率空需求集合返回 `None/not_applicable`；总分在加权项无定义时返回 `None`，权限/评估/风险门禁显式拒绝 `None`，并保留空集合回归断言。
- 第 17 章恢复成功率、重复副作用率，第 18 章上下文折叠约束损失，第 21 章 workspace 可靠性均改为严格正分母公式，正文明确无样本时报告 `not_applicable`，禁止用 `max(1, …)` 伪造 0/1 分数。
- 第二十册 16 个 Python 围栏 AST 与独立运行通过；全库 659 个 Markdown、9622 对围栏配对、1290 个 Python 围栏 AST 通过；`git diff --check` 通过。全库逐块独立运行仍包含既有“片段依赖前文变量”的教学代码，已单独验证本轮受影响的第 13 章完整 demo。
- 2026-09-14 官方 OpenAI 模型页复访因 DNS 解析失败（curl 退出码 6）；未新增一手事实，GPT-6、Kimi K3、GLM-5.3 等待核验边界保持不变。
- 同日复读本地保存的 Z.ai GLM-5.3 官方文档快照，补记其发布方自报的 Terminal-Bench 3.0（4.6→28.3）、DeepSWE v1.1（46.2→66.9）和 Agents' Last Exam（23.8→28.5）前后数值，以及私有 Z.ai Code Bench 的双指标描述；已明确标注为页面快照/发布方自报，未升级为独立复现或基础模型单独增益。
- 复读本地 Kimi K3 官方发布页缓存，补记 Availability 与评测脚注：Kimi Work 3.1.0+、Kimi Code `/model`、API `kimi-k3`、$0.30/$3/$15 每百万 token 价格、Mooncake 分离式推理与编码工作负载缓存命中率“超过 90%”的发布方自报，以及 `max`、temperature=1.0、top-p=1.0 和按基准切换 Kimi Code/Claude Code/Codex harness 的条件。均绑定发布文章快照，未升级为长期费率、普遍性能保证或独立复现。
- 复读本地 DeepSWE v1.1 HTML 快照的数据对象，补记精确生成时间 `2026-09-03T22:24:37.984682+00:00`、统一 `mini-swe-agent` 仓库链接、GPT-6 Astra 最近作业时间及 Kimi K3 的 309/451、Pass@4≈89.4%、4 次重复运行和约 ±4.5% 区间字段；强调表格展示值经过四舍五入，引用时应保留分子/分母、重复次数和区间。

## 2026-09-14：第七册专项 Frontier 评测指标定义域复核

- `book-07-evaluation-experiments/chapters/15-specialized-frontier-eval-cluster.md` 的隐藏测试通过率从 `h/max(1,n)` 改为 `h/n, n>0`，代码新增统一 `ratio` 函数；空隐藏测试集合返回 `None/not_applicable`，trace 覆盖率和平均成本在空任务集上也不伪造数值。
- 补充 `ratio(1, 0)`、`ratio(0, 0)` 回归断言，并使空任务集进入 `remeasure_before_comparison` 门禁；该章 Python demo 已通过 AST、独立运行和 `git diff --check`。
- 额外按反引号与波浪号两种 Markdown 围栏重新扫描全库：1,698 个 Python 围栏全部通过 AST，未发现未闭合围栏；此前 1,290 个反引号 Python 围栏统计仍保留作历史可比口径。

## 2026-09-14：Claude Haiku 4.5 官方目录快照补充

- 从本地 Anthropic Models Overview 快照结构化字段核验 Claude Haiku 4.5：`claude-haiku-4-5-20251001`、别名 `claude-haiku-4-5`、2025-10-15 发布字段、200K context、64K 最大输出、extended thinking、`fastest` 延迟字段、平台 ID、价格及 2025-02/2025-07 cutoff。
- 新增 `research/model-update-2026-09/claude-haiku-4.5-source-notes.md`，并同步来源索引、候选解释、模型百科、术语、面试题、练习、论文、知识图谱和项目审计器。未读取的 announcement/system card、参数规模、训练架构、完整推理机制和独立 benchmark 继续标为待核验。
- Haiku 4.5 未出现在 2026-09-09 Artificial Analysis 采集切片中，因此不补写榜单日期；明确区分“官方目录发现”和“排行榜发现”。
- 已将 Haiku 4.5 入口同步到 `plan_v2.md`、`README.md`、`BOOK_SERIES.md` 和 `ROADMAP.md`，并完成受影响 Markdown 链接检查。

## 2026-09-14：Artificial Analysis 快照字段边界补强

- 从本地 `/tmp/artificialanalysis.html` 提取并复核 2026-09-09 快照中的配置级 Artificial Analysis Intelligence Index 示例：Fable 5.1 max with fallback 约 53.37、GPT-6 Astra max 约 52.81、Opus 5 max 约 50.70、GLM-5.3 max 约 44.86、Grok 4.6 high 约 44.41、Kimi K3 max 约 43.78、Gemini 3.8 Flash high 约 41.19。
- 已在 `inventory-interpretation.md` 和 `source-index.md` 标注：这些是第三方榜单快照配置，`releaseDate`/`parameters`/开放性字段不能替代官方发布日期、参数或许可证；指数不能与 DeepSWE 或其他 harness 分数直接合并。
- 已将该快照解释入口加入 `plan_v2.md`，后续引用排行榜时统一保留采集日期、effort/fallback、模型 revision 和 harness 条件。
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
- 新增第二十一册第 81 章，更新目录和第四册第 19 章；同步 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、README、`BOOK_SERIES.md`、`ROADMAP.md`、`plan_v2.md`、来源索引、候选解释和历史模型盘点说明。
- 模型卡自报 base/Agent 数字均保留 benchmark、effort、temperature/top-p、harness、工具、环境和 verifier 条件；未把 DeepSWE 74.2 等组合结果归因给基础模型。章节 demo 和全局格式、链接、AST 检查仍需在本轮末执行，完整 kernel、API 价格图片、线上路由实测和独立 benchmark 继续待核验。

## 2026-09-14：DeepSeek V4.1-Flash 技术报告逐页核验

- 通过 `10.24.27.134:7890` 重新获取固定 revision 的技术报告，51 页逐页文本提取成功；哈希仍为 `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`，并记录 PDF 书签与页码范围。
- 报告补充并已写入研究笔记/第 81 章：CED 的 `O(NL/2+n_win*L/2)` 复杂度、CSA2 的 `m=2/m=1` 层分组、`2048*8=16,384` HSI 候选池、Single-Pass mHC traffic、Engram 两模块设置、DSpark 五位置草稿器、RoPE 后 FP4 QAT、EPD 解耦、SWA 的 10% host-DRAM 短 TTL 池、45T/100.6M-token 训练设置和异步 RL/OPD 机制。
- 报告的 Table 1/3、effort 曲线、Dsec 容器密度和 cache 生命周期均保留为发布方内部设置与自报结果；完整 kernel source、所有参数分片、线上接受率、目标硬件 profiling、API 价格/限流和独立 benchmark 仍待核验。
- 清理第 81 章中由区间读取造成的重复段落，更新 `source-index.md`、`PAPERS.md`、`BOOK_SERIES.md`、`KNOWLEDGE_GRAPH.md` 和 `plan_v2.md` 的报告状态；全库结构检查随后执行。

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
- 其他厂商及其新模型暂不主动检索、核验或扩写；现有历史记录不删除、不回退，仅保留作为已有资料，直到用户明确通知重新纳入关注范围。该范围约束已写入 `plan_v2.md`。

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
| OpenAI | GPT-5.5、GPT-5.5 Pro、GPT-5.5 Instant | AA；基础 GPT-5.5 亦见 DS | GPT-5.5/Pro 资料级闭环；Instant 仅候选 | GPT-5.5/Pro 的模型页、专属指南、推理/工具/缓存/状态/后台文档和研究笔记已完成；Instant 没有进入本次 API 核验结论。 |
| OpenAI | GPT-5.4、GPT-5.4 Pro、GPT-5.4 mini/nano | AA；基础 GPT-5.4 亦见 DS | GPT-5.4 资料级闭环；Pro 为关联档位；mini/nano 仅候选 | GPT-5.4 的两个排行榜配置、官方模型页、专属指南、API 周边文档和研究笔记已完成；1M context、tool search、computer use、native compaction、custom tools/CFG、`phase` 和状态回放已核验；未公开架构、训练 recipe、独立技术报告和完整复现。 |
| Anthropic | Claude Fable 5.1 | AA | 资料级闭环 | 官方专属模型页、研究笔记和全套配套同步已完成；没有独立正式专题，参数、架构、训练细节和外部复现待补证。 |
| Anthropic | Claude Fable 5 | AA/DS | 资料级闭环 | Artificial Analysis 的 `max + Opus 4.8 Fallback`、DeepSWE 的 `xhigh + mini-swe-agent`、Anthropic 模型页、发布/重新部署公告、API/Agent 文档、system card 入口和研究笔记已完成；参数、架构、训练配方和独立 benchmark 待核验。 |
| Anthropic | Claude Opus 5 | AA/DS | 资料级闭环 | 两榜单配置、官方模型目录/专属页、开发者文档、发布公告、System Card 入口、研究笔记和配套同步已完成；参数、架构、训练细节和独立 benchmark 待核验。 |
| Anthropic | Claude Opus 4.8 | AA/DS | 资料级闭环 | Artificial Analysis 的 `max`、DeepSWE 的 `xhigh`/`max` + `mini-swe-agent`、Anthropic 发布公告、System Card 入口、Dynamic Workflows 博客、研究笔记和配套同步已完成；参数、架构、训练配方和独立 benchmark 待核验。 |
| Anthropic | Claude Sonnet 5 | AA/DS | 资料级闭环 | 两榜单配置、Anthropic 发布页/模型目录、完整 API/Agent 文档、System Card、Research/arXiv 定向检索、研究笔记和配套同步已完成；参数、架构、完整训练/后训练 recipe 和独立 benchmark 仍待核验。 |
| Anthropic | Claude Sonnet 4.6 | AA/DS | 资料级闭环 | 两榜单配置、Anthropic 发布页、真实模型目录、System Card 入口和 Agent 运行时资料已完成；1M context beta、adaptive/extended thinking、compaction、tool search、computer use 已核验；参数、架构、训练 recipe 和独立 benchmark 待核验。 |
| Z.ai（GLM） | GLM-5.3 | AA/DS | 内容专题闭环 | 官方 GLM-5.3/5.2 文档、研究笔记、第十六册第 20 章及配套同步已完成；SAO with compaction 原始定义、模型卡、权重和许可证待补证。 |
| Z.ai（GLM） | GLM-5.3-Flash | AA/DS | 内容专题闭环 | 已核验 Z.ai 文档/博客、固定 revision 模型卡与配置；已新增第二十一册第 84 章并同步书系配套。完整训练 recipe、生产 kernel、硬件 profiling、线上接受率和独立评测仍待补证。 |
| Z.ai（GLM） | GLM-5 | AA 单榜 | AA 单榜资料闭环 | Artificial Analysis 有精确 `glm-5` 条目；DataCurve 当前没有精确 `mini_swe_agent_glm_5_*` 行。已核验 Z.ai 模型卡、GLM-5 专属技术报告、DSA、744B/40B MoE、`slime` 异步 RL 和 Agentic Engineering；完整 DSA kernel、调度细节、训练 recipe、硬件 profiling 和独立复现待补证。 |
| Z.ai（GLM） | GLM-5.2 | AA/DS | 资料级闭环 | 已完成 AA/DataCurve high/max 复验、Z.ai 官方文档和研究笔记；1M/128K、长周期 Coding Agent、MCP、缓存和工作流已核验；参数、架构、SAO/compaction 原始定义、训练 recipe 和独立复现待补证。 |
| Qwen | Qwen3.8 Max、Qwen3.8 27B、Qwen3.8 2.4T A95B、Qwen3.8-Flash-Next | AA；Max 亦见 DS | 内容专题闭环 | 三份官方模型卡、Flash-Next GitHub/技术报告、Qwen Cloud 页面、研究笔记、第 83 章及配套同步已完成；完整生产 kernel、线上接受率、目标硬件 profiling、全系列训练配方和独立 benchmark 待核验。 |
| Kimi/Moonshot | Kimi K3 | AA/DS | 内容专题闭环 | 官方发布资料、Kimi K3 研究笔记、第十七册第 15 章及 KDA/AttnRes 配套同步已完成；完整技术报告、权重交付和精确配置待核验。 |
| Kimi/Moonshot | Kimi K2.7 Code | AA/DS | 资料级闭环 | 官方资源/API 文档、固定 revision 模型卡/配置/许可证、部署指南和研究笔记已完成；1T/32B active MoE、MLA、MoonViT、native INT4、thinking/preserve-thinking/tool-call 协议已核验；专属训练报告、完整后训练配方、线上接受率和独立 benchmark 待核验。 |
| DeepSeek | DeepSeek V4.1 Flash | AA | 内容专题闭环 | 官方发布页、模型卡、技术报告、研究笔记和第二十一册第 81 章已完成；完整 kernel、线上接受率、目标硬件 profiling 和独立 benchmark 待补证。 |
| DeepSeek | DeepSeek V4 Pro、V4 Flash | AA/DS | 内容专题闭环（V4 系列） | V4 官方公告、模型卡、技术报告资料、第五册第 18 章及架构专题已覆盖；不同快照行为、API 路由和独立复现仍待核验。 |
| DeepSeek | DeepSeek V4 Flash Vision | AA | 资料级闭环 | 已完成实验发布、Vision/Files/Responses/价格文档、研究笔记和第二十一册第 85 章；历史 alias、当前路由、图像预算和工具回灌已核验；视觉专属架构/训练报告、DataCurve 同名结果和独立评测仍待核验。 |
| Google（Gemini） | Gemini 3.8 Flash | AA/DS | 资料级闭环 | Google 官方模型页、DeepMind Model Card、发布博客、评测 PDF、研究笔记及工具/推理文档已入库；GA/stable、1M 输入、64K 输出、thinking、Interactions API 和工具闭环已核验；3.8 独立架构/训练报告和正式专题待补证。 |
| Google（Gemini） | Gemini 3.7 Flash | AA/DS | 资料级闭环 | 已完成 Google AI Developers 模型页、DeepMind Model Card、评测/安全入口、Interactions/工具/视频文档和 arXiv 定向检索；1M 输入、thinking、agentic video、step/state/signature 回放已核验；独立架构/训练报告待补证。 |
| Google（Gemini） | Gemini 3.6 Flash | AA/DS | 资料级闭环 | 已完成 Artificial Analysis/DataCurve 原始配置、Google AI Developers 模型页、DeepMind Model Card、Thinking/Interactions/工具组合/视频/缓存/Computer Use 文档和 arXiv 9 篇外部使用论文核验；默认 medium 与 `minimal/low/medium/high` thinking、agentic video、Model Card benchmark/safety 已记录，独立架构/训练报告待补证。 |
| Google（Gemini） | Gemini 3.5 Flash | AA/DS | 资料级闭环 | 已完成 Artificial Analysis/DataCurve 原始配置、Google AI Developers 模型页、What's New、DeepMind Model Card、Gemini 3 Flash 依赖 Model Card、Thinking/Interactions/工具组合/视频/缓存/Computer Use 文档和 arXiv 定向检索；默认 medium、thought preservation、工具上下文循环、Computer Use prompt-injection detection 和证据边界已记录；独立架构/训练报告待补证。 |
| Google（Gemini） | Gemini 3.1 Pro Preview | AA/DS | 资料级闭环 | 已完成三代理榜单复验、Google API 模型页、Release Notes、DeepMind Model Card、评测方法、Thinking/工具组合/Long context/Caching 文档和 arXiv 检索；1M/65K、多模态、`thinking_level`、`customtools` endpoint、signature/id 回放和 tool context circulation 已核验；独立架构/训练报告待补证。 |
| xAI（Grok） | Grok 4.6 | AA/DS | 资料级闭环 | xAI 官方发布页、模型/API 文档、四档 DataCurve 配置、研究笔记和配套同步已完成；官方发布日期、500K context、模型生成数据/SFT 轨迹筛选、agentic RL、self-testing/verification 和工具协议已核验；参数、架构、完整训练配方、专属技术报告和独立 benchmark 待核验。 |
| xAI（Grok） | Grok 4.5 | AA/DS | 资料级闭环 | 两个排行榜的 `high` 配置、xAI 发布公告、官方模型/API 文档、研究笔记和同步记录已完成；参数、架构、完整训练配方、专属技术报告和独立 benchmark 待核验。 |

### 后续逐项顺序

Grok 4.5、Grok 4.6、Claude Opus 5、Claude Fable 5.1、Claude Sonnet 5、Claude Sonnet 4.6、Gemini 3.7 Flash、Gemini 3.6 Flash、Gemini 3.5 Flash 和 Gemini 3.1 Pro Preview 均已完成“仅候选”到“资料级闭环”的升级，不新增专属架构章节。GLM-5 已完成 AA 单榜资料闭环，GLM-5.2 已完成双榜资料级闭环；GPT-5.4、GPT-5.5、GPT-5.6、Gemini 3.8 Flash、Claude Fable 5、Claude Opus 4.8 和 Kimi K2.7 Code 继续只补独立架构/训练报告、真实 API 行为和外部评测；已达到“内容专题闭环”的 GPT-6 Astra、GLM-5.3、GLM-5.3-Flash、Kimi K3、DeepSeek V4/V4.1、K2 Horizon 36B/A4B 和 Qwen3.8 只补待核验项，不重复建立全量模型档案。当前锚点为 `claude-sonnet-4-6`，已完成资料级闭环；下一主锚点继续从两个排行榜剩余重点条目选择。

## 2026-09-14：Qwen3.8 内容专题闭环

- Artificial Analysis 快照确认四类 Qwen3.8 条目：Max（2026-08-03）、2.4T-A95B（2026-08-12）、27B（2026-08-14）和 Flash-Next（2026-08-26）；DataCurve DeepSWE 当前快照只确认 Max。不同 `reasoning_effort`/Non-reasoning 行按配置归并，不按行数计基础模型。
- 官方核验完成：27B 和 A95B 模型卡、Flash-Next 模型卡与 GitHub、28 页技术报告、Qwen Cloud Max/27B/Flash 页面。研究笔记见 [`qwen3.8-source-notes.md`](research/model-update-2026-09/qwen3.8-source-notes.md)。
- 已整理面试主线：GDN + Gated Attention 的混合记忆、QSA 的 micro-block indexer/两阶段训练、四分支 Gated Residual、Layer 2 N-gram Embedding 的 host-memory prefetch、Muon/AdamW 分工与 batch-size warmup 取舍，以及 `enable_thinking`/`reasoning_effort`/`preserve_thinking` 协议。
- 新增第二十一册第 83 章，并同步第四册百科、目录、来源索引、模型盘点、候选解释、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、README、`BOOK_SERIES.md`、`ROADMAP.md`、`PROGRESS.md` 和本计划/进度文件。
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
- 新增研究笔记 [`gemini-3.8-flash-source-notes.md`](research/model-update-2026-09/gemini-3.8-flash-source-notes.md)，同步 `source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan_v2.md` 和本进度表；暂无 Gemini 专属正式章节，先映射已有 Reasoning、Agent、工具协议、Computer Use、长上下文/Serving 和评测章节。
- 该项闭环属于“资料级闭环”，不等于内部技术已全部公开；后续补证重点为独立架构/训练资料、真实 API 行为、生产工具安全边界和外部 benchmark 复现。

## 2026-09-14：GPT-5.6 资料级闭环

- Artificial Analysis 2026-07-09 快照记录 Sol/Terra/Luna 的多档 effort；DataCurve DeepSWE v1.1 快照记录 `gpt-5.6-sol` max（73% ±3%、约 $6.46、60K 输出 token、61 steps）和 `gpt-5.6-luna` max（67% ±4%、约 $0.61、73K 输出 token、102 steps）。不同 effort 行归并为同一 GPT-5.6 家族；DeepSWE 数字归因于模型配置 + `mini-swe-agent` + 工具/环境/verifier。
- OpenAI 三个模型页确认 Sol/Terra/Luna 的层级、`gpt-5.6` 到 Sol 的 alias、1,050,000 context、922,000 maximum input、128,000 maximum output、text/image input、text output、reasoning tokens 以及 `none`--`max` effort；Chat Completions、Responses、Batch 和 Responses 工具支持也已记录。
- Reasoning 文档补充 GPT-5.6 的 Responses `standard/pro` mode、effort 解耦、`reasoning.context=all_turns` 默认语义、同家族 reasoning 复用、function call 后保留 reasoning items 和 output budget/incomplete 边界。
- Prompt Caching 文档补充 GPT-5.6 的 1,024 可见 token 最小缓存前缀、implicit/explicit 断点、最多四次写入、0.1x 读取、1.25x 写入、30m TTL、自动路由和 compaction 可能降低 cache hit 的边界；Compaction/Agents/Tools 文档补充长期任务、工具搜索、MCP、Skills、Shell 和 harness 分层。
- 官方开发者博客 `Codex as a platform` 给出 retained reasoning + context compaction 使特定 GPT-5.6 Sol 工程流程分数从 13.3% 到 38.3%、输出 token 减少六倍的系统级例子；该数字不写成裸模型 benchmark。`Shell + Skills + Compaction` 与 `One year of Responses` 用于解释长期 Agent 的 runtime 组合。
- 本次没有找到 GPT-5.6 专属参数、架构、训练/后训练 recipe、system card、独立技术报告或可复现 benchmark。`latest-model.md?model=gpt-5.6` 当前实际解析为 GPT-6 Astra，已排除出 GPT-5.6 证据链。研究笔记见 [`gpt-5.6-source-notes.md`](research/model-update-2026-09/gpt-5.6-source-notes.md)。
- GPT-5.6 当前为“资料级闭环”，不新增专属正式章节；面试映射到第六、七、五、八、十六、十七、二十和二十四册。下一锚点转向 GPT-5.5。

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
- 新增研究笔记 [`grok-4.5-source-notes.md`](research/model-update-2026-09/grok-4.5-source-notes.md)，同步来源索引、模型盘点、榜单解释、`plan_v2.md` 和本进度表。Grok 4.5 当前为“资料级闭环”，没有独立架构/训练报告，因此不新增专属正式章节；当前锚点切换为 `grok-4.5`，下一锚点为 `grok-4.6`。

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
- 已同步 [`claude-fable-5.1-source-notes.md`](research/model-update-2026-09/claude-fable-5.1-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan_v2.md`](plan_v2.md) 和本文件。Fable 5.1 当前为“资料级闭环”，不新增独立架构章节；当前锚点切换为 `claude-fable-5.1`，下一锚点为 `claude-sonnet-5`。

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
- 已新增研究笔记 [`glm-5.3-flash-source-notes.md`](research/model-update-2026-09/glm-5.3-flash-source-notes.md) 和第二十一册第 84 章 [`glm-5.3-flash混合注意力与视觉闭环.md`](book-21-transformer-architecture-evolution/chapters/84-glm-5.3-flash混合注意力与视觉闭环.md)，并同步 `source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan_v2.md`、第四册百科、第二十一册目录/时间线以及论文、题库、练习、术语、项目和知识图谱。
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
- GPT-5.4 当前为“资料级闭环”：GPT-5.4/Pro 的关联关系已记录，但 Pro 未单独建立完整资料档案，mini/nano 仍为关联候选；没有公开参数规模、激活参数、层/专家结构、训练与后训练 recipe、system card、独立技术报告或完整 benchmark 复现，因此不新增 GPT-5.4 专属正式章节。研究笔记见 [`gpt-5.4-source-notes.md`](research/model-update-2026-09/gpt-5.4-source-notes.md)，当前锚点为 `gpt-5.4`；下一步仍从两个排行榜的剩余重点厂商候选中选择，不从 OpenAI 官方目录另发现模型。

## 2026-09-15：Gemini 3.7 Flash 断点恢复、联网复验与资料级闭环

- 针对用户所说的夜间 20:00—次日 09:00 可能造成代理中断，本轮重新发起联网搜索。沙箱内三条代理均无法连接；经受控联网重试后，`10.24.27.134:8098` 成功获取 Artificial Analysis 中文首页、DataCurve、DeepMind Model Card 和 DeepMind Research，`10.24.27.134:7890` 成功获取 Google AI Developers 模型页和 arXiv，`10.237.126.170:1234` 也成功获取 arXiv/DeepMind Research。7890 对 Artificial Analysis 大页面和部分 Google 页面存在超时，不能把单代理失败解释为网页不存在。
- Artificial Analysis high 详情页本轮复验为 3,605,373 bytes、SHA-256 `d0c5128baac580dbc983b403719f44730decf3bd6c3a24484d29633f13ffbe4c`；页面确认 `Gemini 3.7 Flash (high)`、`releaseDate: 2026-08-13`、Intelligence Index `39.4295316404896`、约 `292.3389 tokens/s`、约 `10.2161s` input/TTFT 字段、1M context、proprietary 且参数字段为空。输入/输出/cache hit 价格字段约为 `$0.75/$3.75/$0.075` 每百万 token；这些都是 Artificial Analysis 的第三方配置字段。
- DataCurve fresh 快照为 268,313 bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`，页面仍为 113 tasks、91 repositories、5 languages、统一 `mini-swe-agent`，包含 Gemini 3.7 Flash low/medium/high 四次运行配置。Pass@1 分别为 `53.7611%/65.4867%/65.2655%`，平均成本约 `$1.8323/$2.0251/$2.1763`；这些是配置 + harness + 工具 + 环境 + verifier 的系统结果，不是裸模型能力。
- Google AI Developers 模型页 fresh 快照 SHA-256 `53e8a497f27951f2d533efa4609a9a45e816e5bda697a4d7470489c88ed9243a`，确认模型 ID `gemini-3.7-flash`、输入 text/image/video/audio/PDF、text 输出、1,048,576 输入 token、65,536 输出 token、low/medium/high thinking、`minimal` 报错、caching/code execution/file search/function calling/Maps/Search grounding/structured outputs/URL context/Computer Use Preview，以及 Batch/Flex/Priority inference。
- DeepMind Model Card fresh 快照为 159,258 bytes、SHA-256 `b8051a19b7578abd1ab60e222f0afa829db38b415b66fbe91a24d6d89e3cdbc5`。官方只披露核心 reasoning foundation 的算法改进、agentic video understanding 和可调 thinking；架构、训练数据、硬件和软件信息指向 Gemini 3.6 Flash Model Card。没有把前代信息升级成 Gemini 3.7 独有事实。
- 已核验的面试技术线索包括：Interactions API 的 interaction/step、`previous_interaction_id`、store 保留/后台执行/implicit caching；tool context circulation、thought/tool signature 的 stateful/stateless 回放和 built-in/custom tool 执行边界；agentic video 的主动时间轴、transcript/帧/帧率/分辨率选择、`processing_call`/`processing_result` 以及 stateless processing step 回放。
- Google 官方评测 PDF 的发布方数字已单独记录：DeepSWE v1.1 65.3%、Terminal-Bench 2.1 85.8%、LVBench 85.4%、GDM-MRCR v2 128K 97.0%、OSWorld-2.0 47.9%。安全边界记录为 cybersecurity 达到 alert threshold 但未达到 CCL，CBRN 未达到 TCL/CCL；这些是 Google 自报安全/评测资料，不与 DataCurve 或 Artificial Analysis 拼接。
- arXiv 精确标题查询 [`title:"Gemini 3.7 Flash"`](https://arxiv.org/search/?query=%22Gemini+3.7+Flash%22&searchtype=title) 返回 0 个结果，快照 SHA-256 `a92b072d2026186568d09e3e42129649ee852886d6572d278cbb529f57b217a2`；全文查询找到 3 篇外部使用/评测论文：arXiv:2609.15983、2609.05232、2608.20563。它们分别研究多 Agent 研究 harness、执行资源契约和长时程安全 Agent 失败诊断，不是 Gemini 3.7 专属技术报告。
- 新增 [`gemini-3.7-flash-source-notes.md`](research/model-update-2026-09/gemini-3.7-flash-source-notes.md)，同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`plan_v2.md`](plan_v2.md) 和本进度表。Gemini 3.7 Flash 当前为“资料级闭环”，暂无独立正式章节；当前锚点切换为 `gemini-3.7-flash`，下一锚点继续从两个排行榜的剩余重点厂商条目中选择。

## 2026-09-15：Gemini 3.6 Flash 断点恢复、联网复验与资料级闭环

- 针对用户所说的夜间 20:00—次日 09:00 可能导致代理中断，本轮重新发起联网搜索。三条代理对 Artificial Analysis 中文首页均返回 HTTP 200；随后通过可用线路重新获取 Gemini 3.6 详情/provider 页、DataCurve、Google AI Developers 模型页、DeepMind Model Card、Thinking、Interactions、工具组合、视频理解、Context caching、Computer Use 和 arXiv 检索页。个别代理的瞬时连接失败没有被解释为页面不存在。
- Artificial Analysis 详情页快照 `/tmp/gemini36-recheck-aa-8098-20260915.html` 为 3,600,704 bytes，SHA-256 `362fd94a45955783974462edb3cdc99ed932625a56d8c0afe7b9bca4a7f2e721`；页面确认 `Gemini 3.6 Flash (high)`、`releaseDate: 2026-07-21`、Intelligence Index `34.3395678920714`、约 `192.7337 tokens/s`、TTFT `19.2239s`、1M context、约 `$0.15/$0.75/$3.75` cache-hit/input/output 每百万 token。provider 页当前只展示 `Google AI Studio` 一个 benchmark provider，不能与详情页 FAQ 的 provider 可用性数量混读。
- DataCurve 快照 `/tmp/gemini36-recheck-ds-8098-20260915.html` 为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；原始行是 `mini_swe_agent_gemini_3_6_flash_high`、`high`、4 runs、211/452、Pass@1 `46.6814%`、Pass@4 `75.2212%`、平均成本 `$2.2095066`、平均输出 `95,844.86` token、平均 Agent steps `116.73`、median peak context `151,398.5`。这是配置 + harness + 工具 + 环境 + verifier 的系统结果，不是裸模型分数。
- Google AI Developers 模型页快照 `/tmp/gemini36-recheck-ai-7890-20260915.html` 为 104,839 bytes，SHA-256 `e0d55ee75ca1c4e80c128284a907de9e2dd602d9a6a39e660599abea1bf43404`；确认 `gemini-3.6-flash`、text/image/video/audio/PDF 输入、text 输出、1,048,576 输入 token、65,536 输出 token、caching/code execution/File Search/function calling/Maps/Search grounding/structured outputs/thinking/URL context/Computer Use Preview，以及 Batch/Flex/Priority inference；稳定 alias 为 `gemini-3.6-flash`，latest update 为 July 2026。
- Thinking 文档明确 3.6 的默认档位为 `medium`，支持 `minimal/low/medium/high`；Interactions 文档模型表列出 3.6；视频文档将 3.6 列入 agentic video 模型，Context caching 文档给出 implicit caching 最低输入 `4,096` tokens。Gemini 3 工具组合文档记录 thought/tool signatures 与 tool context circulation；这些是 API/Agent 协议事实，不是内部 Transformer 机制。
- DeepMind Model Card 快照 `/tmp/gemini36-recheck-card-8098-20260915.html` 为 154,640 bytes，SHA-256 `c2e25fec9cc337856c4f612b729f8d1b3985a0627a7234e66091b53dddd9e1c3`。官方明确 3.6 based on Gemini 3.5 Flash，架构、训练数据、数据处理、硬件和软件资料指向 3.5 Model Card；公开 benchmark 包括 SWE-Bench Pro `58.7%`、DeepSWE v1.1 `49%`、Terminal-Bench 2.1 `78.0%`、GDPVal-AA v2 `1421`、OSWorld-Verified `83.0%`、GDM-MRCR v2 128K `91.8%`/1M `54.0%`。安全表和 Frontier Safety 结论按发布方/自动评测边界记录，未迁移成 3.6 独有训练事实。
- arXiv 精确标题检索快照 `/tmp/gemini36-recheck-arxiv-title-7890-20260915.html` 为 16,452 bytes、SHA-256 `03b6d5dabe226f9c165a6f725f670f2f846659fcc7308c037b9c0a4fe8db9127`，返回 0 篇；全文检索快照 `/tmp/gemini36-recheck-arxiv-all-7890-20260915.html` 为 63,191 bytes、SHA-256 `2e5c93fde5c302f8a1cd655425d9df2e885c31f4a904318c90f00db6ef372ffa`，返回 9 篇外部使用/评测论文，不是 Gemini 3.6 专属技术报告。9 篇的标题、摘要要点和证据边界已写入研究笔记。
- 新增 [`gemini-3.6-flash-source-notes.md`](research/model-update-2026-09/gemini-3.6-flash-source-notes.md)，并同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan_v2.md`](plan_v2.md) 和本进度表。Gemini 3.6 Flash 当前由“仅候选”升级为“资料级闭环”，暂无独立正式章节；当前锚点为 `gemini-3.6-flash`，下一锚点仍从两个排行榜剩余重点条目选择。

## 2026-09-15/16：Gemini 3.5 Flash 断点恢复、联网复验与资料级闭环

- 针对用户提出的 20:00—次日 09:00 代理中断规则，本轮重新发起联网收集。2026-09-15 工作时段重新抓取时，`10.237.126.170:1234` 与 `10.24.27.134:8098` 对 Gemini 3.5 Flash 的 Artificial Analysis 与 DataCurve 均返回 HTTP 200，两个代理取得的 AA 详情和 DataCurve 快照逐字节一致；7890 成功获取 Google AI Developers、DeepMind Model Card 和 Google API 周边文档。2026-09-16 轻量复探中，三个代理对 AA/DataCurve 均返回 HTTP 200，7890 对 Google 模型页返回 HTTP 200，8098/1234 对该大页面超时；代理差异仍不被解释成页面不存在。
- Artificial Analysis 快照为 3,607,081 bytes、SHA-256 `abdbed0cf8069ae81c272aea386724302a0659235fa5b950980a9a9a0d162ec7`；页面确认 `Gemini 3.5 Flash (high)`、第三方 `releaseDate: 2026-05-19`、Intelligence Index `32.9816033695905`、约 `221.8080 tokens/s`、约 `18.3631s` TTFT、1M context 和约 `$1.5625`/task。当前页面的 `deprecated: true`/`deprecatedTo: gemini-3-6-flash` 只作为 Artificial Analysis 目录状态记录，不能写成 Google 官方退役公告。
- DataCurve 快照为 268,313 bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`，仍为 113 tasks、91 repositories、5 languages、4 runs、统一 `mini-swe-agent`。`mini_swe_agent_gemini_3_5_flash_high` 原始行是 163/452、Pass@1 `36.0619%`、Pass@4 `63.7168%`、平均成本 `$3.4467`、平均输出 `75,730.19` token、平均 Agent steps `105.30`；这些是配置 + harness + 工具 + 环境 + verifier 的系统结果。
- Google 官方 API 页面确认 1,048,576 输入 token、65,536 输出 token、text/image/video/audio/PDF 输入、text 输出、caching/code execution/File Search/function calling/Maps/Search grounding/structured outputs/URL context、Computer Use Preview、Batch/Flex/Priority inference；What's New 页面补充 GA、默认 effort 从 high 改为 medium、`minimal/low/medium/high`、legacy `thinking_budget` 互斥、thought preservation 和 2025 年 1 月 knowledge cutoff。
- Interactions API 模型表明确支持 `gemini-3.5-flash`；工具组合文档的 `id`/`signature`、tool context circulation、stateful/stateless 回放和宿主执行责任已入库。Context caching 文档明确 implicit caching 最低输入 `4,096` tokens。Computer Use 文档明确 3.5 Flash 支持 opt-in screenshot prompt-injection detection，默认关闭。
- 视频证据保持保守：API 支持视频输入，但当前视频文档的 agentic processing 列表明确列出 3.8/3.7/3.6 和 3.5 Flash-Lite，没有明确列出 3.5 Flash；因此不把 3.6 的 `processing_call`/`processing_result` agentic video 结论迁移给 3.5 Flash。
- DeepMind Model Card 的发布方 benchmark、安全表和 Frontier Safety 结论已记录；Model Card 明确 architecture、training dataset、data processing、hardware 和 software 均指向 Gemini 3 Flash Model Card。arXiv 精确标题检索返回 0 篇，全文检索返回 30 篇外部使用/评测论文，没有检出 Gemini 3.5 Flash 专属技术报告。
- 新增 [`gemini-3.5-flash-source-notes.md`](research/model-update-2026-09/gemini-3.5-flash-source-notes.md)，同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan_v2.md`](plan_v2.md) 和本进度表。Gemini 3.5 Flash 当前由“仅候选”升级为“资料级闭环”，暂无独立正式章节；下一锚点优先选择 `gemini-3.1-pro-preview`，仍必须先由两个排行榜确认。

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
- 新增 [`glm-5-source-notes.md`](research/model-update-2026-09/glm-5-source-notes.md)，同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`plan_v2.md`](plan_v2.md) 和本进度表。GLM-5 当前为“AA 单榜资料闭环”，不新增独立正式章节；完整 DSA indexer 训练目标、生产 kernel、`slime` 调度、完整训练/后训练 recipe、硬件 profiling、线上接受率和独立复现仍待核验。

## 2026-09-16：GLM-5.2 双榜资料级闭环

- 按夜间 20:00—次日 09:00 可能中断的规则重新抓取 [Artificial Analysis GLM-5.2](https://artificialanalysis.ai/models/glm-5-2) 与 [DataCurve DeepSWE](https://deepswe.datacurve.ai/)，两个页面均返回 HTTP 200。AA 快照 `/tmp/glm52-aa-20260916.out` 为 `3,699,404` bytes，SHA-256 `25cf2438443a3ff1dac18964adb79ae98f24f41ed68d02ea4bd38e527a574244`；DataCurve 快照 `/tmp/glm52-ds-20260916.out` 为 `268,313` bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- 两个排行榜均有精确 GLM-5.2 配置：AA 为 `glm-5-2` 的 max/non-reasoning，DataCurve 为 `mini_swe_agent_glm_5_2_high` 和 `mini_swe_agent_glm_5_2_max`。DataCurve high/max 的 Pass@1 分别为 `36.2832%`/`43.7778%`，平均成本约 `$2.8355`/`$3.9199`，平均 Agent steps `121.88`/`129.13`；结果绑定 `mini-swe-agent`、4 runs、113 tasks、工具、环境和 verifier，不是裸模型能力。
- Z.ai 官方 [GLM-5.2 文档](https://docs.z.ai/guides/llm/glm-5.2) 快照为 `525,141` bytes，SHA-256 `d66a48a8a7654f1b033abca4fc9270a57d52961aace20d1335f36e9ab33ba6ba`，`dateModified` 为 `2026-09-03T14:49:41.429Z`。官方确认 1M context、128K 最大输出、thinking、function calling、context caching、structured output、MCP，以及项目级代码库、跨文件重构和分阶段验证的长周期工程工作流。
- “lossless context”、数月 Coding Agent 专项训练、开发者案例和长任务 benchmark 只按 Z.ai 发布方描述记录；不能解释成数学意义上的绝对无损，也不能把提示词中的 `/goal`、CLAUDE.md/Agent.md、ADB/logcat 或工作流案例写成内部算法。GLM-5.3 引用的 SAO with compaction 仍没有 GLM-5.2 原始定义。
- 已更新 [`glm-5.2-source-notes.md`](research/model-update-2026-09/glm-5.2-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan_v2.md`](plan_v2.md) 和本进度表。GLM-5.2 当前为“资料级闭环”，暂无独立正式章节；下一主锚点继续从两个排行榜的剩余重点候选中选择。

## 2026-09-16：Claude Sonnet 4.6 断点恢复、联网复验与资料级闭环

- 针对昨晚 20:00—今早 09:00 可能发生的代理中断，本轮重新使用三条代理抓取 Artificial Analysis 中文首页、DataCurve DeepSWE 和 Anthropic Sonnet 4.6 发布页，均返回 HTTP 200。三个代理取得的 Artificial Analysis 首页均为 `1,773,553` bytes、SHA-256 `60e491dc5eccaf2d032205a7380f30a24b2726583b5a9cfaccfe14c17d736476`；DataCurve 均为 `268,313` bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`，未发现截断或代理间差异。
- Sonnet 4.6 发布页三条线路均返回 HTTP 200、`281,283` bytes；动态响应哈希不同，但核心发布信息一致。`8098/1234` 访问 Anthropic 模型目录被重定向到区域不可用页，不能作为目录证据；`7890` 成功取得真实 `platform.claude.com` Models Overview，`728,089` bytes，SHA-256 `4c5ba69551075bbff00251943a1765c77be38f998d512fac67902067097cdfff`，确认 `Claude Sonnet 4.6`、`claude-sonnet-4-6`、1M context 和 128K output。
- Artificial Analysis 的 Sonnet 4.6 adaptive 快照为 `3,626,831` bytes，SHA-256 `5feda773b5ccefaaad7265b5f2d2401551d4d96e32cfa524a7fedd550db5b0de`；DataCurve 精确行 `mini_swe_agent_claude_sonnet_4_6_high` 为 135/451、Pass@1 `29.9335%`、Pass@4 `56.6372%`、平均成本约 `$5.5224`、平均输出 `76,160.31` token、平均 Agent steps `133.66`。这些属于榜单配置和 `mini-swe-agent` 系统结果，不能写成裸模型能力。
- Anthropic 官方资料确认 2026-02-17 发布、coding/computer use/long-context reasoning/agent planning/knowledge work/design 定位、1M context（发布时 beta）、adaptive/extended thinking、context compaction、tool search、web search/fetch、code execution、memory 和 programmatic tool calling。Computer-use prompt injection 风险虽称相较 Sonnet 4.5 改善，但不写成已解决。
- Claude Sonnet 4.6 当前为“资料级闭环”。没有公开参数规模、内部架构、完整训练/后训练 recipe 或独立技术报告；adaptive 预算、compaction 摘要格式、tool search 质量、线上 computer-use 接受率和独立复现仍待核验，不新增独立正式架构章节。研究笔记见 [`claude-sonnet-4.6-source-notes.md`](research/model-update-2026-09/claude-sonnet-4.6-source-notes.md)。
