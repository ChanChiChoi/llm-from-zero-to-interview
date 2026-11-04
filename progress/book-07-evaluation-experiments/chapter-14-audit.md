# 第十四章审计：Frontier Model 证据等级

审阅日期：2026-11-03  
状态：章级扫描、正文精修、研究记录、配套同步和静态门禁完成；示例代码未运行；第七册及全套书修书 goal 继续进行。

## scope

- 审阅 `book-07-evaluation-experiments/chapters/14-frontier模型证据等级.md` 的 claim 拆分、证据分层、来源责任、benchmark 可比性、复现与版本更新流程、观察项、Claim Ledger demo 和前沿模型案例。
- 定向研究范围为截至 2026-11-03 可公开访问的 Grok 4.7、Claude Opus 5.5、Claude Fable 5.1 官方发布/API 资料、System Card、Artificial Analysis 与 DataCurve DeepSWE 页面。没有把排行榜缺项解释为零分，也没有迁移相邻模型的 Agent 结果。
- 保留本章“从来源证据到可辩护句子”的职责，不改变第七册章节骨架或本章位置。

## chapter_scan 与候选去向

| 缺口或候选 | 去向 | 决定与理由 |
| --- | --- | --- |
| 原 §14.11 将若干公开发布名称和社区流传名称一并称为“未确认” | §14.11 | 改为核对公开模型名、API ID、服务配置、别名和社区转述；分别确认名称、接口与未公开架构，避免查询未命中和名称已确认被混为一层。 |
| 证据分数、coverage 和 condition completeness 只有定义，没有输入到结果的手算 | §14.6.3 | 增加清楚标为教学假设的等权审计分数 `0.60`、加权覆盖率 `4/6≈0.667` 和条件完整率 `6/8=0.75`，解释三者各自回答的问题及其非概率含义。 |
| Claim Ledger demo 标出“运行输出”，但当前没有执行证据 | §14.19 | 标为“静态审阅，未运行（2026-11-03）”，将输出改为按分支静态推演的预期值，并逐项解释三个输入、四个信号、动作顺序和决策分支；不把布尔 toy 当成来源核验器。 |
| 前沿模型的发布方结果、第三方 provider 配置与 DataCurve harness 行容易被拼成一个裸模型排名 | §§14.22.1—14.22.3 | 补直接 DataCurve v1.1 链接和本次可见快照边界，加入 Grok 4.7 的发布方例分与 high-effort 标注，保留三类测量的任务、effort、fallback 与来源责任。 |
| Opus 5.5 发布页与 System Card 对 OSWorld 的版本标签不同 | §14.22.2 | 分别保留发布页的 OSWorld 2.1 partial `81.8%` 与 System Card 的 OSWorld 2.0 partial/strict `81.8%/48.7%`；不由同分推断任务修订相同，也不迁移 strict 分数。 |
| Fable/Mythos 关系容易从产品说明扩写成未经公开的权重事实 | §14.22.3 | 改为“发布页称同一模型、不同 safeguards 配置”，不写成已核实的权重共享或内部训练事实。 |
| 本章既有 Grok/Fable 来源笔记链接目标不存在 | `research/model-update-2026-09/` | 新建 Grok 4.7、Opus 5.5、Fable 5.1 三份来源记录，分别保存日期、阅读范围、官方/第三方事实和适用边界；不在 research 中写审阅进度。 |

本轮资料没有显示需要新增方法章节或重排。本章其余代表性研究比较、证据类型和五层证据链保留；本章章首定位、书系分工和简介/目录摘要仍准确。

## 正文修改与例子

- §14.6.3 的假设输入均为无量纲教学值：五维内部均分 `(1.0+0.8+0.5+0.3+0.4)/5=0.60`；四个子 claim 权重 `1、1、2、2`，前三项被支持，得到 `4/6≈0.667`；六个已记录字段除以八个必需字段，得到 `6/8=0.75`。文字说明评分人、权重、缺失字段和来源错误都不会被比率消除。
- §14.19 手工追踪 `ctx-interface`、`ctx-task` 和 `architecture` 三条合成输入：作用域不全为 false，存在独立任务证据为 true，架构直接来源为 false，未知项字段已登记为 true；逐项得到四条动作和 `publish_scoped_task_claim_and_hold_architecture`。代码正文未改可执行语句，也未运行。
- Grok 4.7 案例区分官方发布页、API model ID/500K context、Artificial Analysis xhigh 配置与 DataCurve 当前可见的 Grok 4.6 行；发布方 DeepSWE `71.0%` 明确标 high effort，引用时保留未披露条件。
- Opus 5.5 案例将 System Card 的 `66.36%` Terminal-Bench 条件、安全 fallback、DataCurve 缺少精确 5.5 行和 OSWorld 2.0/2.1 标签差异分开记录。
- Fable 5.1 案例保留 production safeguards、不同执行模型/配置、OSWorld partial/strict 口径，以及 DataCurve 只有 Fable 5 可见行的快照边界。

## length_status

按连续论证和可独立学习的辅助单元计解释性汉字；排除标题、表格、代码/文本围栏、链接 URL 和空白。静态计数按 Markdown 正文中的汉字字符执行，人工判断单元边界：

| 连续教学单元 | 实计汉字数 | 类型 | 下限 | 状态 |
| --- | ---: | --- | ---: | --- |
| §§14.1—14.8：最小 claim、证据层级、来源、维度与 NoPE 工作例 | 4,069 | 核心 | 1,800 | 通过 |
| §§14.9—14.15：比较协议、名称观察、不确定性、ledger、冲突与复现 | 2,207 | 核心 | 1,800 | 通过 |
| §§14.16—14.18：发布材料审计、观察项升级与维护回归 | 1,064 | 辅助 | 1,000 | 通过 |
| §14.19：Claim Ledger demo 与逐信号静态 walkthrough | 1,098 | 辅助 | 1,000 | 通过 |
| §§14.20—14.23：未知项、证据清单、三款模型案例与结论 | 2,017 | 核心 | 1,800 | 通过 |

## evidence、formula、code 与 link status

- 新模型来源、阅读范围和事实/边界分别记录于 `research/model-update-2026-09/grok-4.7-source-notes.md`、`claude-opus-5.5-source-notes.md` 和 `claude-fable-5.1-source-notes.md`。官方发布/API 页面、Artificial Analysis 与 DataCurve 当前页面均按 2026-11-03 访问快照记录；DataCurve 的 model list 和 provider 页面会继续变化。
- 公式：检查证据分数、Coverage、ConditionCompleteness 的变量与输入，手算结果均正确；数学围栏花括号为 34/34，无 `\operatorname`。
- 代码：唯一 Python block 位于 §14.19；状态为“静态审阅，未运行（2026-11-03）”。输出仅为按可见逻辑手工推演。代码围栏及相邻状态说明无 `progress/` 或 `research/` 内部路径。
- Chapter14 的 26 个围栏标记配成 13 对；Chapter14 所有本地 Markdown 链接目标存在。新加或改写的官方、AA 和 DataCurve 事实链接已定向核对；历史外链未作全量在线复核。无 Mermaid 图，因此不需要平台渲染验收。
- `git diff --check` 通过。没有运行代码、测试、模型、benchmark 或 PDF 构建，也没有更新 PDF。
- 扩展到 PAPERS、EXERCISES、INTERVIEW_BANK、KNOWLEDGE_GRAPH 的静态链接检查发现 77 个指向缺失本地研究/代码路径的链接实例，集中于其他模型研究记录；这些不是 Chapter14 的本地链接，保留为全书配套资料链接清理项，不用空壳来源笔记掩盖。

## sync_status 与 open_risks

- `PAPERS.md` 补入三份来源记录的入口和 Fable 5.1 发布方/第三方/DeepSWE 边界；`INTERVIEW_BANK.md` 的 49G 增加 Opus OSWorld 来源标签题；`EXERCISES.md` 的 6K 增加三种内部审计比率手算和版本标签审计；`KNOWLEDGE_GRAPH.md` 增加模型声明—证据—结论链。
- `GLOSSARY_EN_ZH.md` 已定义 Claim Ledger、Evaluation Card、HELM 与证据覆盖率；新增内容未引入需另立术语的稳定概念，因此无需改词表。第七册简介、目录和章间顺序未变化。
- 尚未全量复核本章历史外链、System Card PDF 可视渲染和 DataCurve 后续页面变化；重引用时应固定当时 leaderboard snapshot/配置。
- 本次全局静态链接扫描所见其他缺失 source-note/code 路径会在对应书册与配套资料审阅/最终门禁中处理；不得将其误记为 Chapter14 的完成缺陷或将 missing row/文件推断为模型事实。

## work_handoff

- 第十四章定向扫描、研究、正文精修、三份来源记录、配套同步和静态验收完成；无结构调整。第十五章逐章精修已完成，下一步按序连续复读 `book-07-evaluation-experiments/chapters/15-specialized-frontier-eval-cluster.md`，复核专业任务簇、benchmark 饱和、模型—harness—environment 归因和失败轨迹证据。
- 第七册仍需完成第十五章连续复读和最终门禁；全套书精修 goal 保持 active，不更新 PDF。

## 整册连续复读（2026-11-03）

- 按简介和目录路径完整复读本章，并核对第十三章长上下文专题与本章证据审计职责的衔接。第十三章给出特定任务、长度和预算下的能力曲线；本章进一步要求把相关发布句拆为最小 claim，分别绑定接口、方法、任务或生产层证据，未发现重复主讲或责任越界。
- 静态复核 §14.6.3 的三项合成审计量：等权证据分数 `0.60`、加权覆盖率 `4/6≈0.667`、条件完整率 `6/8=0.75`；复核 §14.19 的三条输入、四个布尔信号、四条动作和 `publish_scoped_task_claim_and_hold_architecture` 决策，均与代码逻辑和原审计一致。
- 将 §14.19 标题从“最小可运行的 Claim Ledger 审计 Demo”改为“Claim Ledger 审计 Demo”，与“静态审阅，未运行（2026-11-03）”状态一致；未改可执行语句，也未运行代码、测试、模型、benchmark 或 PDF。
- Grok 4.7、Opus 5.5、Fable 5.1 的来源分层、版本标签、effort/fallback 与 DataCurve 缺失行边界按原研究记录复核；没有把同分或相邻版本成绩合并。未发现需改简介、目录、导航或配套索引的问题；外链和页面动态快照未全量重查。
