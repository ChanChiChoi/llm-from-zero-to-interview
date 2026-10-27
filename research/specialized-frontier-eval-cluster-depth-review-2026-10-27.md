# Specialized Frontier Eval Cluster：代表性工作流评测方法比较

- 复核日期：2026-10-27
- 对应章节：[第七册第十五章：Specialized Frontier Eval Cluster](../book-07-evaluation-experiments/chapters/15-specialized-frontier-eval-cluster.md)
- 研究目标：把“列出 Agent benchmark”推进到方法比较。对每条路线记录它要测的对象、设计机制、公开证据、能支持的结论、主要限制和下一步补证。
- 范围：软件仓库与终端、浏览器/桌面、状态化工具、专业研究、长周期任务和 benchmark 合同审计。文献数字均只作为原研究设置的描述，不合并成排行榜。

## 方法比较

| 方法 | 它解决的评测缺口与机制 | 可支持的结论 | 限制及下一步补证 |
| --- | --- | --- | --- |
| [SWE-bench](https://arxiv.org/abs/2310.06770) | 将 GitHub issue、仓库快照、patch 与测试连接起来；原始集合含 12 个 Python 仓库的 2,294 个任务 | 在固定仓库和测试合同下，观察 Agent 是否能从 issue 推进到可通过目标测试的 patch | 任务与测试覆盖有限，公开 issue/PR 有暴露风险；需要固定 commit、runner、预算和测试，检查测试篡改、权限与隐藏行为 |
| [SWE-Gym](https://arxiv.org/abs/2412.21139) | 在 11 个 Python 仓库中提供 2,438 个 issue、可执行环境和 Agent 轨迹；兼具评测环境与训练/验证资源 | 可研究真实仓库交互环境、轨迹训练与 verifier/候选筛选 | 训练集和评测集的关系、轨迹来源、scaffold 与采样预算会影响迁移；用未参与训练的仓库/任务和等预算对照验证泛化 |
| [Terminal-Bench 2.0](https://arxiv.org/abs/2601.11868) | 89 个高难终端任务，各有独立环境、人工解法和测试，覆盖命令行交互与执行结果 | 在发布版本的任务、环境和预算内评估终端 Agent 完成度 | 样本量小，结果对依赖、容器、时限和 Agent harness 敏感；报告任务 revision、重复运行、资源条件及失败阶段 |
| [WebArena](https://arxiv.org/abs/2307.13854) | 以可复现的网站克隆和任务后的功能状态验证网页工作流 | 评估网页导航、跨页操作和站内知识使用能否达成任务目标 | 克隆站点不代表真实身份、权限、网络和生产副作用；用目标产品的状态合同和权限审计补齐 |
| [OSWorld](https://arxiv.org/abs/2404.07972) | 在真实桌面应用中执行任务，提供初始状态和执行后检查；原论文有 369 个任务 | 评估 GUI 感知、grounding、应用操作与跨应用流程 | 初始化和检查脚本也可能出错；绑定应用版本、分辨率、环境镜像和具体任务修订 |
| [BrowserGym](https://arxiv.org/abs/2412.05467) | 提供 Gym 式统一观察/动作环境、AgentLab 实验工具并接入多种网页 benchmark | 降低适配器、运行记录和实验管理的重复成本 | 它是研究/实验框架，不是统一任务集或统一评分标准；仍需分别报告各 benchmark 的任务与 grader |
| [τ-bench](https://arxiv.org/abs/2406.12045) | 用模拟用户、领域 API 和政策检查结束时数据库状态，并以 pass^k 观察重复运行可靠性 | 评估多轮澄清、业务规则遵循和状态变更可靠性 | 用户模拟器、有限政策和领域范围会影响结论；检查 simulator 偏差，并在目标 API 与澄清分布复测 |
| [ToolSandbox](https://arxiv.org/abs/2408.04682) | 1,032 个测试案例、34 个组合工具、多轮模拟用户；milestone DAG 和 minefield 事件分别表达必要过程与禁止动作 | 接受多种有效轨迹，同时度量依赖顺序、信息不足时的停止行为及最终状态 | milestone/minefield 依赖专家建模，模拟器和 grader 也可能错；对照人工审阅状态转移、模拟器错误和领域权重 |
| [AgentDojo](https://arxiv.org/abs/2406.13352) | 在工具任务中注入不可信内容，分别报告 benign utility、攻击下 utility 与针对性攻击成功率 | 将正常任务效用和 prompt injection 安全放进同一状态化环境 | 攻击模板、工具集和防御范围限定覆盖；需要目标系统权限、数据敏感度、外部副作用和真实红队样本 |
| [GAIA](https://arxiv.org/abs/2311.12983) | 466 个组合推理、网页、多模态和工具任务，按答案检查通用助理任务 | 观察系统能否为现实问题选择工具、搜集信息并作答 | 答案正确不等于轨迹可靠、证据支持或研究过程可复现；增加 trace、引用和产物验证 |
| [ScienceAgentBench](https://arxiv.org/abs/2410.05080) | 从 44 篇论文构造 102 个数据分析任务，覆盖四个学科并含专家参与；要求提交可执行 Python 产物 | 检查数据驱动分析代码能否运行、产生成果并满足 rubric | 作者限定于 Python 数据分析、有限学科和可在 10 分钟内完成的任务；不涵盖完整文献综合、假设提出和端到端科研 |
| [PaperBench](https://arxiv.org/abs/2504.01848) | 20 篇 ICML 2024 Spotlight/Oral 论文拆成 8,316 个层级 rubric 子结果，由作者共同设计，用于复现实验 | 观察 Agent 对方法理解、实现、运行、排错和部分复现的长程推进 | 21.0% 是论文报告的平均 replication score，不是完整复现论文的比例；judge、算力、论文选择和小规模人工基线限制解释 |
| [METR time horizon](https://metr.org/time-horizons/) | 以人类专家任务完成时间描述难度，拟合 Agent 成功率 logistic 曲线并报告 50%/80% 成功点 | 比较任务难度增加时可靠性怎样下降 | 人类时间是任务难度刻度，不代表 Agent 可连续自主运行的时长；任务领域和明确定义限制外推。页面说明数据集自 2026-09-08 起不再主动更新，高于 16 小时估计不可靠 |
| [Gaia2](https://arxiv.org/abs/2602.11964) | 让异步外部事件独立于 Agent 动作变化，加入时间约束、噪声、歧义与协作，并检查写动作 | 测环境中途变化时观察、改计划和正确改变状态的能力 | 仿真事件和 action verifier 绑定作者环境；应另报事件时间线、观察机会、状态证据与预算 |
| [Executable-contract audit](https://arxiv.org/abs/2609.37315) | 将工具文档/schema 与实现对照，在新鲜环境执行前置条件和状态变化，并追踪 evaluator 实际读取字段 | 审计 benchmark 宣称的合同能否由环境和 grader 实际执行 | 这是固定 commit 下的审计预印本，不是 benchmark。作者报告“可触达缺陷”不等同于 Agent 触发了问题或榜单改变；部分合同状态不可观察，需按可观测字段报告范围 |

## 证据与比较边界

1. 这些路线没有共同任务、预算、动作空间、初始状态或评分器。论文里的完成率、pass^k、部分分、时间 horizon 和安全率不能合并排名。
2. 环境框架与任务集要分清：BrowserGym 提供统一实验接口，但其接入的任务、grader 和难度没有因此统一。
3. 同一任务的结果仍依赖模型 revision、prompt、工具权限、重试、记忆、环境、grader、资源上限和 seed。报告应固定这些字段，并对工具 Agent 比较相同初始状态。
4. Agent benchmark 也需要被审计：验证工具合同、状态变更和 grader 读取的字段。但静态缺陷检查只表示实现存在潜在暴露面，不足以推断真实 Agent 触发或分数排序受影响。
5. 动态网页和滚动文档应保存固定 commit/release 或访问日期。METR 页面记录其停止更新日期；不把页面上的历史估计当成实时能力榜单。

## 本章公式与复现材料复核

- 切片宏平均仅对 n_k > 0 的非空切片取平均；所有切片都为空时结果为 not_applicable。原稿用 max(n_k, 1) 会把空切片悄悄变成零分，已修正。
- 配对差值定义 A=新版、B=旧版，并令 b=旧版对/新版错、c=旧版错/新版对，所以 (c-b)/n 明确为新版减旧版。
- 复现包包含 trace 但 trace 可能带提示、用户数据、令牌、路径和业务状态；加入凭证/标识符脱敏、最小权限、保留期限和可追溯引用要求，并要求脱敏后仍能检查动作—观察—状态—grader 的证据链。
- 章节 demo 是 toy 代码，本轮仅做静态复读与手算，不宣称运行通过。

## 来源链接复核

- BrowserGym 正确论文入口为 [The BrowserGym Ecosystem for Web Agent Research](https://arxiv.org/abs/2412.05467)，官方文档为 [browsergym.readthedocs.io](https://browsergym.readthedocs.io/latest/index.html)。原章节参考文献误链到 2305.17144（Ghost in the Minecraft），已更正。
- 其余论文及 METR 官方页面链接按章节引用列出。本轮未执行所有外链的 HTTP 状态扫描；引用的分数与数据都应回到各自论文版本和任务协议解释。

