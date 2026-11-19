# 第二十册第十九章精修审计

日期：2026-11-18  
章节：[第十九章《Harness-Aware Evaluation》](../../book-20-agent-harness-runtime/chapters/19-harness-aware-evaluation.md)  
状态：章级研究、正文重写、目录与配套同步完成；第二十册第二十至二十四章及全套书精修 goal 继续 active。

## 章节契约与边界

本章承接第九章 Trace/Artifact 证据和第十章 Evaluation Harness 设计。第十章主讲 task fixture、环境初始化、Runner、verifier、分母和成对比较；第十九章进一步回答：同一任务上的结果差异来自模型、Agent harness、环境，还是它们的配对交互？读者读完应能定义比较对象、构造模型×harness 的交叉矩阵、区分任务成功与 harness 履约、用 trace 提出并验证归因假设、分别呈现资源成本与泛化边界。

本章不重复主讲第十章的评测运行器和 verifier，也不把第七册统计设计搬入本册；置信区间、抽样和显著性检验链接第七册第十章。第十八章负责 context folding 机制，第十九章只说明怎样评估该机制的变化。

## 章级外部扫描与候选去向

检索与阅读日期：2026-11-18。范围限定为 Agent/harness 评估、模型与 harness 的统计交互、运行可靠性、trace 归因、成本/多 Agent 预算、harness 演化过拟合和相关课程组织。通过 web search 搜索论文题名与交叉设计关键词，并直接阅读固定版本论文、课程原始页面及 OpenAI 官方文档；未以既有 41 个小节作为检索清单。实际检索和来源证据保存在[Model–Harness 交互与归因研究记录](../../research/model-harness-interaction-attribution-2026-11-18.md)。

| 外部主题 | 直接证据与边界 | 本章安排 |
|---|---|---|
| 模型、harness 和环境会共同影响分数 | Han 与 Sun 对 Terminal-Bench 2.0 leaderboard 做加性和交互分析；105 个配置对中只有 3×9 的完整交叉子集用于交互估计，且研究是观察性分析。 | 19.1—19.4 定义配置、比较对象、交互与矩阵完整性，不把论文结果写成通用因果排名。 |
| Agent task outcome 与 harness 履约不是同一个结果 | Trial by Trace v1 在 95 个机制任务上分别评分 Agent 任务结果与 harness criteria；单个 cell 只有一个被选中执行。 | 19.5 分层说明，不把该论文的 31.1% 移植为普遍失败比例。 |
| Trace 给出过程证据，但候选根因需验证 | Trial by Trace 将候选机制链与人工核验传播链分开；Harvard 讲义也把 trajectory 与 outcome 区分。 | 19.6 要求核对事件、实际副作用、环境版本、验收输入并排除竞争解释。 |
| Harness 成本和收益需一起看 | Do More Agents Help 在固定 GPT-4.1、共享执行/评分接口和 token 记账下，展示不同任务集可能方向不一。 | 19.7 分别报告质量、token、工具、时间和人工接管成本，不设计单一效率分。 |
| Harness 演化可能吸收测试集特征或额外计算 | Harness-Delta Attribution v1 在其方法框架下分析 search-set gains，并将 overfit、test-time scaling 和残余上界分开。 | 19.8 讲 task-family/repository holdout 与资源对齐，明确作者数字依赖其 HDA 方法。 |
| 教学需要由对象定义过渡到实验与报告 | Harvard CS2680 讲义从 task/trial/trajectory/outcome、配置和 grader 展开；Buffalo CSE 410/510 把 task-level eval、metrics、CI/bootstrapping 接到包含 test cases、failure taxonomy 和 reproducible script 的 Lab 4。 | 19.4、19.9 取其组织思路；不把课程大纲说成教学效果研究。 |
| 供应商评测工作流 | OpenAI 当前指南建议调试时查看 trace，再用 datasets/eval runs 做重复比较。 | 19.6 仅作一家供应商的工作流例子，不推断 grader 效度或普遍标准。 |
| Provider/API 产品事实 | 属于版本合同，不是通用 harness 归因方法。 | Gemini Interactions、Responses 与 Claude 合同分别回链本册 23、20、24 章；产品发布分数仅留在对应 research/source notes。 |

未发现需要新增独立章节的新主题。关键缺口是：原稿前段与第十章重复，后段将多个发布方模型案例并列为方法课，未让读者看见它们的共同归因规则。

## 原稿内容去向

### 原 19.1—19.29

| 原稿范围 | 本轮处理 |
|---|---|
| 19.1—19.5：同模型成绩差异、函数表示、harness 组成、公平比较、成功判据 | 合并进 19.1—19.5；fixture、Runner、verifier 和评测分母交第十章主讲。 |
| 19.6、19.15、19.22：成本、重试和可靠性指标 | 方法保留于 19.7；报告 token、调用、时长和失败，不使用单一成本—质量总分。 |
| 19.7、19.14、19.18、19.23、19.25：Trace、回放和失败分析 | 原则合并到 19.6；Trace/Artifact schema 和回放系统回链第九、十章。 |
| 19.8、19.24：代码 Agent 评测表与 artifact 验收 | verifier 与 artifact 归第十章；保留本章的层次区分与版本绑定。 |
| 19.9—19.10：失败归因和常见误区 | 重组进 19.2—19.6，删除泛化清单式重复。 |
| 19.11：面试回答与练习 | 更新为[练习 35EN](../../EXERCISES.md)和[面试题 40AC](../../INTERVIEW_BANK.md)。 |
| 19.12—19.13、19.16—19.18、19.21：配置版本、Model×Harness×Environment、指标/矩阵和失败数据 | 按 reader question 重组至 19.1—19.6；删去不能定义分母或误作独立变量的自拟聚合指标。 |
| 19.19、19.22：retry worked example 与重试成本 | 收束为 19.7 的预算原则；不沿用缺乏配置和统计条件的“单位价值”总指标。 |
| 19.20、19.26、19.28—19.29：发布条件、反事实、回归包和阶段判断 | 报告所需材料写入 19.10；第十章继续主讲回归夹具。本章不保留固定发布 gate。 |
| 19.23、19.25：Replay 和失败轨迹价值 | 保留“观察与归因分开”的原则于 19.6；具体 side-effect replay 安全边界回第九、十章。 |
| 19.27：线上指标与离线分数 | 保留至 19.10，说明它们回答不同条件下的问题，不能直接拼成同一分数。 |

### 原 19.30—19.41 的发布案例

| 原小节 | 去向与决定 |
|---|---|
| 19.30 Grok 4.20 multi-agent | 一般性全系统成本比较改由 19.7 和 Do More Agents 研究例说明；xAI 当前 API/runtime 事实回到[Agent swarm runtime 来源记录](../../research/agent-swarm-runtime-evidence-2026-11.md)，不在通用评估章节重复 beta/Responses API 合同。 |
| 19.31 Gemini 3.8 Flash / Interactions | provider state 合同交本册第 23 章；上下文折叠交第 18 章。删除以 Gemini 发布参数组成的长 manifest 案例。 |
| 19.32 GLM-5.1 长周期任务 | 反馈与独立验收的一般原则归入 19.5—19.6；仓库无该小节所引的 GLM-5.1 source note，发布数字和机制主张未在本章保留。 |
| 19.33 Claude Sonnet 5 System Card | System Card 的发布方评测放入本册第 24 章相关 provider 合同，并与已存在的 [Claude Sonnet 5.5 来源记录](../../research/model-update-2026-09/claude-sonnet-5.5-source-notes.md)保持准确版本，不把 Sonnet 5 和 5.5 混称。 |
| 19.34 Grok 4.7 benchmark 与长轨迹 | 报告条件不能横向混排的规则归于 19.2、19.10；具体发布方数字留在[现有 Grok 4.7 来源记录](../../research/model-update-2026-09/grok-4.7-source-notes.md)。 |
| 19.35 Claude Opus 5.5 token 与责任 | 全系统成本原则归于 19.7；发布、安全及配置数字留在[现有 Opus 5.5 来源记录](../../research/model-update-2026-09/claude-opus-5.5-source-notes.md)。 |
| 19.36 DeepSeek Harness Preview | 一般配置 manifest 归于 19.10；原稿引用的 source note 与 audit script 均不存在，因此删除产品合同细节及“脚本运行结果”断言，避免把不可复核的 toy 输出写成产品证据。 |
| 19.37 GPT-5.6 Luna state replay/cache | 运行状态与成本的固定项归于 19.10；Responses/item 状态合同交本册第 20 章，上下文折叠交第 18 章；删除缺失来源链接。 |
| 19.38 Claude Opus 5 Prompt-Injection ASR | 任务成功与 harness/safety criteria 分层归于 19.5；Opus 5 发布与安全评估留在[现有来源记录](../../research/model-update-2026-09/claude-opus-5-source-notes.md)和第七册 safety evaluation，不在本章重写 ASR 分母。 |
| 19.39 Qwen3.6-27B benchmark | 通用发布方与第三方口径区分归于 19.2；仓库没有原稿所链的 Qwen3.6-27B source note，故不保留具体榜单数字。 |
| 19.40 Qwen3.7 Max agent RL | Search/holdout 与测试时计算原则归于 19.8；具体发布方数字和长程案例留在[现有 Qwen3.7 Max 来源记录](../../research/model-update-2026-09/qwen3.7-max-source-notes.md)及其已有章节入口。 |
| 19.41 Qwen3.6-35B-A3B harness/verifier RL | 任务 oracle 与泛化的通用问题归入 19.5、19.8；原稿引用的 Qwen3.6-35B-A3B source note 不存在，移除未经可复核来源支持的模型、训练和 benchmark 数字。 |

上述迁移保留能支撑评估方法的知识；provider release 事实继续由具体版本的研究记录或 provider 章节负责。删除缺失的本地文件链接时同时删除其无法验证的结果断言，并非只移除链接文字。原稿自拟的统一 score function、成功率连乘、质量/成本加权值、release gate，以及本地 toy script 的已运行输出均不再保留；章节没有用新公式或新 gate 替换。

## 正文主线与形式状态

- **reader_path：** 用同一模型分数变化引出完整配置 → 明确比较问题 → 以 2×2 教学矩阵看平均变化与交互 → 定义 task/trial/trajectory/outcome → 分开 task outcome、harness criteria 和评测装置故障 → 从 trace 提候选原因并验证 → 报告预算与多 Agent 成本 → 检查 benchmark overfit → 比较研究证据的适用范围 → 写可复查 manifest 与报告。
- **example_status：** 2×2 矩阵是明确的教学构造，四个任务、每格一次计数只用于解释主效应和配对交互，不代表实测、不支持统计显著性，也不估计随机运行波动。
- **evidence_status：** 每项论文结果均标固定版本、评测对象与限制；Harness-Bench 摘要页和固定 v2 正文任务数冲突时，正文与 research note 注明差异；Trial by Trace 的单 cell 单次选中运行未作为方差证据；HDA 54%/14% 限于该论文方法和显著 search-set 设置；OpenAI 文档限于供应商工作流。
- **formula_status / code_status：** 无作者自拟推导公式或可执行代码；将教学矩阵的成功比例、研究中的百分比/百分点/倍数等 13 项数值写成行内 MathJax，未添加综合得分或新 gate。移除原稿中未定义分母、未经来源支持的指标函数、加权聚合与固定 gate。
- **visual_status：** 2×2 结果表、比较条件表、结果分层表和 research comparison 表足以表达本章对照；没有流程关系或行为机制必须画 Mermaid 图才能理解的情形。
- **sync_status：** 更新目录第 19 章摘要，新增练习 35EN 和面试题 40AC，新增本审计和来源记录，更新第二十册架构地图及全局进度交接。简介没有单章摘要，无需改动。未改 plan.md、PDF、术语表或知识图谱。

## 静态验收与开放风险

- H2 编号 19.1—19.10 连续；13 个行内 MathJax 表达式共 26 个定界符，均成对；章节不含代码围栏、Python toy 声明或固定质量 gate。
- 正文、目录、研究记录、章节审计、练习/题库条目、架构审计和本交接共检查 78 个本地 Markdown 链接目标，缺失为 0；第 19 章—第 23 章另外检查 19 个目标，7 个缺失均属于待精修的第 20—23 章，已记入架构审计。章节中不存在旧 Grok 4.20、Gemini 3.8、GLM-5.1、DeepSeek Harness、GPT-5.6 Luna、Qwen3.6 等产品发布小节标题。
- git diff --check 通过。未做目标读者试读、GitHub 页面渲染或所有外链长期可达性检查。
- 未运行代码、模型、API、SDK、benchmark、自动化测试或 PDF 构建；未更新 PDF 或 plan.md。
- 旧稿中无本地来源记录的 GLM-5.1、Qwen3.6-27B 和 Qwen3.6-35B-A3B 数字已移除。若后续任务需要纳入，先保存可复核原始来源和条件，再决定是否服务本章读者目标。

## 连续复读补充（2026-11-19）

- 完整复读本章并复走 §19.3 的模型×harness 交叉教学矩阵，确认主效应与配对交互的解释、实验单位、条件对齐和外推边界相符。对照第九章 Trace、第十章 Evaluation Harness、第十八章 context folding 和第七册统计设计后，章节职责与交接仍成立。
- 复核 Han 与 Sun 的稀疏 leaderboard 分析、Trial by Trace 的单 cell 单次选择、Do More Agents Help 的任务/成本条件以及 HDA 的方法边界；沿用 2026-11-18 研究记录，没有引入新的外部事实。
- 按模块 04 将教学比例、百分点、倍数和论文百分数等 13 项数值改成行内 MathJax，保留原数值、分母和来源结论；未增加自拟汇总公式或固定门槛。静态复核：H2 `19.1`—`19.10` 连续；26 个数学定界符配成 13 对；无代码围栏；相关本地 Markdown 链接缺失 0；`git diff --check` 通过。未运行代码、模型、API、SDK、benchmark 或测试；GitHub MathJax 实际页面渲染留作整册终验。

## 下一入口

[第二十册第二十章《Responses API》](../../book-20-agent-harness-runtime/chapters/20-responses-api.md)。provider conversation/item、工具和续接合同由后续章节基于官方版本材料核验；第二十册第二十至二十四章与全套书其余未完成部分继续精修，全书 goal 保持 active。
