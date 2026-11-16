# 第十三章《Agent—环境闭环》精修审计

审阅日期：2026-11-16。状态：正文、章节命名、导航与跨章术语同步完成；§13.24 demo 仅静态审阅、未运行。GitHub Mermaid/MathJax 呈现、外链全量可达性和代码运行仍留在最终门禁。

## scope 与 reader_path

- **范围：** 完整复读第十三章正文（本轮文件 926 行），并核对本册简介、目录、架构审计、第十章评估/第十一章安全的接口及 `research/agent-environments-multi-agent-2026-11.md` 中的 OSWorld、AgentWorld 来源边界。保留既有文件路径 `13-agentworld模型与环境闭环.md`，把读者可见标题改为“Agent—环境闭环”。
- **读者前置：** 已有 Agent 工具调用、反馈循环和计划的基本图景；第十、十一章提供通用评估与安全指标。第十三章属于进阶环境专题，正文仍解释所需的状态、action、observation、reset 和 verifier，不依赖读者记住前章公式。
- **本章贡献：** 让读者能为代码、浏览器、数据库或业务工作流写出可复现环境契约；区分模型提议、执行器动作、环境状态与外部验收，并把不确定执行、部分成功、reset 失败和评测归因纳入报告。
- **阅读路径：** 从“文本说完成”与“世界状态确实完成”的区别出发，定义策略、环境、Harness、状态/动作/观察；再用契约、事件账本、reset、沙箱、反馈和 verifier 建立闭环；随后比较模型/工具/环境、归因长任务并固定版本；最后由数字案例和 workspace demo 手算一次安全成功与外部状态验收。

## chapter_scan 与 evidence

- **检索日期与问题：** 2026-11-13 至 2026-11-16；核对环境接口与任务状态如何定义，OSWorld 论文分数和 OSWorld-V2 release 如何区分，AgentWorld 专名、CCE 与通用 Agent—环境闭环如何区分，当前官方 Agent evaluation 文档可支持何种表述。
- **候选及去向：**

| 候选 | 本章作用与处理 | 证据范围 |
|---|---|---|
| MDP/POMDP、状态/动作/观察、任务契约 | 核心正文；解释状态投影、未知结果、停止与恢复，并由合同助手案例贯穿。 | 用 Gymnasium API 支持公开 reset/step 接口语义；不将其当作业务权限或安全标准。 |
| reset、事件账本、外部 verifier、环境隔离 | 核心正文；从独立初始状态到副作用复读，解释评测如何证明外部目标。 | 教学结构和工程设计建议；不声称任何 benchmark 或生产环境已自动满足。 |
| AgentWorld 多 Agent benchmark（arXiv:2609.31590） | 只作为专名案例；正文说明其 3—20 Agent 任务与 CCE，不把指标当通用 Agent 能力或环境定义。 | 预印本作者报告；CCE 的动作依赖归因含模型辅助标签，结论绑定其任务和标注口径。 |
| OSWorld 1.0、OSWorld 2.0 论文 v2、OSWorld-V2 `osworld-v2.1` release | 保留不同论文/仓库版本入口；正文明确论文 2.0 的结果不能改标为 v2.1。 | OSWorld 2.0 的 108 个工作流及发布快照信息来自论文 v2 和官方 release；不混用各自结果。 |
| AgentBench、WebArena、SWE-bench、BrowserGym、dm_env 与官方 Agent evals | 作为具体环境、接口或评估流程的资料入口，不扩成逐个 benchmark 综述。 | 分别绑定论文实验、官方接口文档或框架用途；外部状态 verifier、权限和 reset 仍需单独设计。 |

- **来源记录：** OSWorld 2.0、OSWorld-V2、AgentWorld 和其他相关来源的版本、阅读位置、证据边界及待验证项记录于[Agent 环境、长任务与多 Agent 评估资料](../../research/agent-environments-multi-agent-2026-11.md)。本轮未把章节进度写入 `research/`。
- **结构判断：** 章节虽覆盖多个环境类型，但它们共同回答“如何让 Agent 的成功绑定到可复现状态和外部验收”，且由统一契约与闭环案例连接；本轮未发现需拆成互不相依章节的读者结果，也未因保持原章数删除核心主题。

## coverage_map

| 知识点 | 对读者结果的作用 | 深度与去向 |
|---|---|---|
| 策略、环境、Harness 及状态/动作/观察 | 将模型输出与真实环境状态区分，说明观察投影和动作请求。 | 核心；有 MDP/POMDP、闭环 Mermaid 和贯穿合同助手例。 |
| 任务契约、权限、状态转移与 unknown | 说明初始快照、工具前置条件、拒绝/草稿/未知及成功条件。 | 核心；支持代码、浏览器、数据库和合同工作流。 |
| reset、沙箱、反馈、轨迹与外部 verifier | 判断样本是否独立、动作是否安全、环境变化是否真实、结论能否回放。 | 核心；逐层讲解并回链第十章评估和第十一章安全。 |
| 公平对照、模型/工具/环境归因、长任务与版本固定 | 避免把 Harness、观察权限、预算或环境快照带来的变化归因给模型。 | 核心；加入 OSWorld 论文/release 的版本案例；Runtime 机制更深处交第二十册。 |
| 常见漏洞、数字诊断和环境 demo | 把成功率拆成能定位的链条，并观察执行拒绝、验收和 reset。 | 核心实践；用 disjoint-set 假设的数字案例和标准库教学 demo 落地。 |
| 其他公开环境与 benchmark | 为后续查证提供明确入口。 | 资料入口；结果均限于各自任务、配置、Harness、权限与 verifier。 |

## example_status、formula_status 与 code_status

- **贯穿案例：** 合成合同助手任务检查合同与政策版本、付款期限阈值、证据充分性和权限；只能创建待审核工单，需回读状态才能确认。案例把输出文本、外部动作和目标状态分开。
- **数值案例：** §13.23 以 100 个任务为分母：90 个 patch 可应用、75 个通过公开测试、60 个通过隐藏测试；明确在 60 个隐藏测试通过样本中，3 个禁止文件修改和 2 个用户变更覆盖互不重叠，故安全成功数 `60-3-2=55`，比例 `0.55`。若失败类别重叠，按并集计数；没有假定事件独立。工单示例补充中间漏斗，不把请求接受当最终完成。
- **公式：** 环境转移、轨迹成功、三态 verifier、信念状态、reset 有效性和安全成功分解均在正文解释符号与作用；概率乘积分解明确为相关事件下的诊断近似，不是精确预测。公式静态检查中未发现 `\operatorname`、`\(...\)` 或尖括号下标风险；GitHub MathJax 实际呈现未验。
- **图示：** §13.2 Mermaid 图展示任务契约/权限、策略、action 请求、Harness 校验、环境转移、observation、verifier 和 reset 的关系；标签使用引号，文字箭头使用管道语法。源码已静态检查，GitHub 渲染仍待确认。
- **代码：** §13.24 的 Python 标准库 demo 在本轮未改可执行语句，状态保持“静态审阅，未运行（2026-11-16）”。按代码逻辑手工检查的输出为 good success=True、越界轨迹 success=False/`action_rejected`、hash 长度为 12、reset_clean=True；空轨迹和非法动作保持 unknown/rejected 且不改 workspace。没有运行代码、测试或 benchmark，以上不是运行日志；真实容器隔离、并发、网络和隐藏测试不在 demo 覆盖范围。

## link_status 与 sync_status

- 第十三章读者可见标题、目录 Mermaid 节点、目录行及简介摘要已统一为“Agent—环境闭环”；历史文件名保留以维持路径稳定。正文将 `AgentWorld` 留作 benchmark 专名，并在首次出现时写全 CCE 名称及限定范围。
- 新增第十章评估与第十一章安全的本地回链；同时把第十章 §10.20.2 中误用的“AgentWorld 这类环境型模型”改为“环境型 Agent 系统”，并补入指向本章的链接。相关修改记于[第十章审计](chapter-10-audit.md)。
- 同步 `GLOSSARY_EN_ZH.md` 区分 Agent—环境闭环与 AgentWorld benchmark；`PAPERS.md` 区分 arXiv:2609.31590 的 AgentWorld 多 Agent benchmark 与 arXiv:2606.24597 的 Qwen-AgentWorld，并增加本章 benchmark 入口。
- 已更新本册[架构审计](architecture-audit.md)的章节地图、命名决策、状态与下一入口；已更新全套修书进度中的本章交接。外链全量可达性尚未检查。

## open_risks 与 handoff

- §13.24 demo 未运行；如后续执行验证，需按模块 04 记录 Python 环境、命令、输入、边界和预期/实际结果，再更新正文状态。
- GitHub 对章节 Mermaid/MathJax 以及简介、目录 Mermaid 的目标平台呈现仍待确认；外部链接没有做全量可达性检查。
- OSWorld/AgentWorld 论文或仓库可能更新；本章引用必须继续保留论文版本、release tag、任务快照、Harness 和评分口径边界。
- **下一入口：** 第十四章《Agent Swarm 并行协作》，检查并行任务如何隔离 workspace、共享与合并 artifact、控制预算并恢复失败；保持它与本章环境状态、reset 和 verifier 契约之间的分工。第十七册全书仍需第 14—15 章精修、整册连续复读及最终门禁。
