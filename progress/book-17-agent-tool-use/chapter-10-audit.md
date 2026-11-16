# 第十章《Agent 评估》精修审计

审阅日期：2026-11-16。状态：本轮正文与配套同步完成；代码仅静态审阅、未运行，GitHub 公式渲染和外链全量可达性仍待最终门禁。正文保留 `10.0—10.27` 的章节结构。

## scope 与 reader_path

本章回答：如何将 Agent 的任务结果、动作轨迹、验收器、授权副作用与资源成本放进可复核的评测协议，并由首个失败事件定位模型、harness、工具、环境或 verifier 的问题？读者应能为一类任务建立可观测契约，选择有明确分母的指标，核查 verifier/judge 误差，比较模型与 harness，并沿 trace 解释结论。

读者路径从任务契约和 trace 开始，先区分成功率与部分分，再检查工具选择、observation、状态更新、claim 忠实性、恢复、安全和成本；随后讨论自动、人工、LLM judge 各自能证明什么，进入回归、harness 归因和多用户协作评估，最后用统一的知识库 Agent 设计案例与审计 demo 回收指标。

## coverage_map

| 知识点 | 读者作用与本次处理 | 解释层级/去向 |
| --- | --- | --- |
| 任务契约、trace、任务成功与部分成功 | 明确样本单位、外部验收和 rubric；补充任务配对及重复运行的不确定性边界 | 本章核心；区分任务抽样与同题运行噪声 |
| 工具、observation、状态、恢复与 trace 忠实性 | 给出可观察指标及不同分母，终止动作不计入后续决策分母 | 本章核心；状态更新机会单独界定 |
| 授权与高风险动作 | 分开未授权提案、执行器阻断、实际越权执行；确认率只看实际执行的高风险动作，并报告总体安全处置率 | 本章核心；动作执行生命周期链接第 11 章 |
| 自动 verifier | 增加误接收、误拒绝、弃权和覆盖率的独立校准，并用合成 confusion matrix 手算 | 本章核心；高风险场景保留错误成本不对称 |
| 人工评估与 LLM judge | 分开准确性、重复稳定性、候选顺序不变性和未评分覆盖；按领域和难度验证 | 本章核心；不把温度 0 或多数投票写成可靠性保证 |
| Harness-aware 评估 | 增加任务有效完成与任务正确性的区别、模型×harness 交叉对照、失败尝试资源账本 | 本章核心；来源为特定预印本，不称作统一行业标准 |
| Multi-Agent 多用户公平性 | 将群体目标、逐用户 ask/同意、提交时 ask 留存、未经同意覆盖和 claim 支持分别记录 | 本章支撑主线；共享资源案例承接第 9 章，安全授权交第 11 章 |
| 第 1 章迁移的 trace audit demo | §10.20 已承接原来的评估程序读者目标，增加执行/阻断状态、空集处理和分母说明 | 迁移关闭；代码未执行 |

没有新开章节或改变章节职责。本章内容仍由“怎样证明 Agent 做成任务、怎样解释失败”的中心问题贯穿；多用户、公平比较和 judge 校准均为评估协议的一部分，没有从其他章节重复搬入完整实现。

## chapter_scan 与 evidence

- 日期：2026-11-16。问题：Agent 成功/trace 指标的有效样本集合；verifier 误判和 LLM judge 稳定性；模型/harness 归因与有效完成；共享资源任务中的个体意图和群体结果。
- 时间范围和资料类型：2024—2026 的原始论文/预印本、ACL Anthology 论文、benchmark 及官方研究说明；不运行模型或 benchmark。
- 候选处理：AgentJudgeBench、RuVerBench、All Verdicts 用于提示准确性、难度、重复性、顺序和覆盖率应分开验证；When Agents Fail 用于任务成功/有效完成与模型—harness 交叉归因；Worse Together/MAMUBench 用于多用户意图和群体行为的互补观察；Towards Reproducible LLM Evaluation 用于强调固定 seed 不消除所有运行不确定性。各候选只按各自任务和实验协议表述，不迁移论文错误率或性能数字。
- 来源版本、阅读位置和事实边界已留在[统计推断与 LLM 评估资料](../../research/statistical-inference-llm-evaluation-2026-11-03.md)、[Judge 可靠性资料](../../research/llm-judge-protocol-reliability-and-resolution-2026-11-03.md)、[Agent 环境与多 Agent 资料](../../research/agent-environments-multi-agent-2026-11.md)、[Harness/benchmark 生命周期资料](../../research/evaluation-benchmark-lifecycle-and-agent-harness-2026.md)。本审计只记录章节取舍和修改，不把审阅过程写入 research。

## examples、formula 与 code 状态

- 核心案例：知识库 Agent 贯穿任务契约、检索证据、动作限制、外部状态和最终验收；第 10 章 §10.20 保留 5 条合成 trace，并在正文逐项解释失败维度。
- 新增 verifier 合成表：20 条人工复核样本；有效/无效各 10 条，接受 `8/2`、拒绝 `1/7`、弃权 `1/1`；手算已评分样本上的误接收率 `2/9=22.2%`、误拒绝率 `1/9=11.1%`、评分覆盖率 `18/20=90%`。这些值明确为教学数据。
- Multi-Agent 案例：三人日历的群体冲突数降为 0，但有一项用户约束未经同意被覆盖；用于展示群体结果不能替代逐用户 ask、同意与 claim 支持。明确标作合成例，不从单个论文场景外推。
- 指标分母已统一：`N` 为任务；`M` 为全部动作提案（含阻断）；observation/state 使用带可用 observation 和后续决策的子集；失败恢复只对失败工具调用定义；成本的标量和须处于同一单位/价格版本。未授权提案率、未授权执行率、阻断率和高风险安全处置率分开。
- 五条 trace 的预期输出手工重算：任务成功率 `0.4`、平均部分分 `0.65`、工具/参数合法率各 `0.944`、observation 使用和状态更新各 `11/13=0.846`、summary faithfulness `0.6`、claim support `0.7`、恢复率 `1.0`、重复动作率 `0.111`、未授权提案率 `1/18=0.056`、未授权执行率 `0/17=0.0`、未授权提案阻断率 `1.0`、高风险执行确认率 `0.0`、高风险安全处置率 `0.5`、平均归一化教学成本 `0.91`、P95 延迟 `5000 ms`、环境可复现率 `1.0`。删除提案被明确记录为 `blocked`，所以相应“危险动作未执行”claim 现在有轨迹字段支持；浏览器提交则仍是未确认执行。
- 代码位于 §10.20。新增 `observation_available`、`state_relevant`、`has_next_decision` 和 `execution_status` 字段及对应空集/分母处理；输出文本和门槛按上述手算同步。状态标记为“静态审阅，未运行（2026-11-16）”；没有执行代码、测试、模型或 benchmark。
- 公式状态：变量、样本集合和空分母语义已显式说明；verifier 错误率与配对/重复运行的不确定性补上解释。GitHub 页面 MathJax 尚未验收。

## link_status 与 sync_status

- §10.18.3 已链接到[第九章共享资源案例](../../book-17-agent-tool-use/chapters/09-multi-agent.md)；来源链接加入 §10.26。目录和简介第十章摘要已同步为覆盖用户意图、verifier、harness 成本与安全的评测契约。
- 第十三章精修时将本章误用的“AgentWorld 这类环境型模型”改为通用的“环境型 Agent 系统”，并补入指向第十三章环境契约的交叉链接。
- 第 1 章迁移状态已在[第一章审计](chapter-01-audit.md)关闭；[书册架构审计](architecture-audit.md)已推进到第十章并把下一入口改为第十一章。
- PAPERS.md 新增 AgentJudgeBench、All Verdicts 和 When Agents Fail；INTERVIEW_BANK.md 增加 verifier/judge 校准问题并扩展 Agent workflow eval 的多用户维度；EXERCISES.md 同步动作执行状态/有效分母及 verifier/judge 校准要求；GLOSSARY_EN_ZH.md 增加有效完成率、误接收/误拒绝和提案/执行/阻断指标定义。
- 研究事实已有对应 research 记录，本轮未把章节进度写入 research；`plan.md` 与 PDF 均未修改。

## open_risks 与下一步

- 本地源文件检查不能代替 GitHub MathJax 实际渲染；章节及来源外链没有进行全量可达性检查。
- Python demo 按本轮要求只做静态审阅和手算，没有实际运行；后续若用户要求执行验证，再按模块 04 记录环境、命令和边界。
- 第一至十章的目标平台 Mermaid/MathJax、旧外链全量验收，以及第 2—4、6—8 章既有代码复验仍在全册门禁中。
- 下一入口：第 11 章《Agent 安全》，重点核对策略授权、执行阻断、确认状态与本章新增指标的责任边界；全套书精修 goal 继续 active。按用户要求不更新 PDF。
