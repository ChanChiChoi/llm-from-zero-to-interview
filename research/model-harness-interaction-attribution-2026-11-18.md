# Model–Harness 交互与归因研究记录

资料核验日期：2026-11-18。本记录只保存外部资料支持的研究事实、组织样本和证据边界。

## 检索范围与记录

- **研究问题：** Agent 任务结果怎样受模型与 harness 配对影响；哪些实验设计可以区分主效应、交互、执行可靠性、资源消耗和对 benchmark 的适配；公开课程怎样把这些问题组织成可学习、可复查的评估过程。
- **检索范围：** 2025—2026 年公开的 Agent/harness 评测研究；固定版本的论文正文、公开课程讲义与教学大纲，以及当前官方 Agent evaluation 文档。没有搜集供应商产品发布案例作为方法证据。
- **检索平台与检索词：** Web search，精确检索 “How good is your harness? Statistical evaluation of coding harnesses” 及 “When Agents Fail” Harness Terminal-Bench harness effect fully crossed 3 harness 9 models；再按原论文引用和已知固定页面打开原文。组织样本通过课程原始页面直接读取。
- **筛选与核验：** 以公开论文版本、课程原始页面和官方文档为主；记录版本、可读正文范围、任务与配置、来源实际支持内容及不能外推的边界。对动态页面注明访问日；没有把研究里的分解量当作通用指标。

## Harness 与模型的统计分解

### Han 与 Sun：Terminal-Bench 2.0 的统计评估

- **来源与版本：** Jiwoo Han、Yuekai Sun，*How good is your harness? Statistical evaluation of coding harnesses*，CTB@ICML 2026；[OpenReview PDF](https://openreview.net/pdf/99eabc2ce65fd2871a253a0a57954c934ea9e6b0.pdf)，2026-11-18 核验。阅读摘要、数据、加性模型、交互模型与讨论。
- **来源支持：** 研究对象为 Terminal-Bench 2.0：89 个终端任务，每个任务 5 次 trial。清理并正规化 leaderboard 后，作者排除与单一模型完全混杂的 harness，得到 105 个 harness–LLM 配对，涉及 28 个 harness 与 26 个 LLM；整体设计仍稀疏且不平衡。作者先用加性模型描述平均 harness 与 LLM 效应，再把交互分析限在完整交叉的 3 个 harness × 9 个 LLM 子矩阵，即 27 个配对。论文报告所选 leaderboard 中的 harness 与模型效应量级相近；部分配对交互也可与主效应相当。
- **证据边界：** 这是公开 leaderboard 的观察性分析，不是随机分配模型和 harness 的受控因果试验；结果受提交者选择、版本差异和稀疏设计影响。交互分析只涵盖完全交叉的小子矩阵。文中对个别交互机制的解释被作者明确作为推测。数据仅支持所测版本、任务与模型/harness 配对下的统计估计，不能据此形成普遍 harness 排名。

## 其他归因与分层结果研究

### Harness-Bench：完整工作流任务

- **来源与版本：** Yao et al., *Harness-Bench: Evaluating Agent Harnesses across Models and Realistic Workflows*，[固定 arXiv v2 正文](https://arxiv.org/html/2605.27922v2)，2026-10-15。2026-11-18 复核 §3—§5 与限制附录。
- **来源支持：** 固定 v2 正文以固定任务输入和初始环境，保留各 harness 原生行为，以任务 oracle 检查最终交付物，并记录结果、轨迹和资源；正文描述 300 个跨六类工作的任务、10 个模型后端和 7 个 harness。研究讨论完整 model–harness 配置的质量、完整完成和成本差异。
- **证据边界：** 附录写明每个 task–model–harness 配置只评估一次，无法从这些运行估计配置的随机波动；任务来自作者所选的日常工作流。2026-11-18 可见的 arXiv 摘要页写 106 个任务，而固定 v2 正文写 300 个；两处规模冲突已有[评测生命周期与 Harness-Bench 版本记录](evaluation-benchmark-lifecycle-and-agent-harness-2026.md)登记。描述规模时需固定页面和版本，不能把摘要与正文数字混写。

### Trial by Trace：任务结果与 harness 履约分层

- **来源与版本：** *Trial by Trace: Benchmarking AI Agent Harnesses for Reliability and Failure Attribution*，[arXiv v1](https://arxiv.org/html/2610.32285v1)，2026-10-23；2026-11-18 核验摘要、§3—§5 与复现说明。
- **来源支持：** TraceTrial-Bench 以 95 个机制任务比较 4 个 harness 和 9 个模型，报告 3,420 个被选中的 task–model–harness 执行。作者分开评分 Agent 任务结果与 harness criteria；摘要报告 31.1% 的运行完成了任务但至少违反一项 harness criterion。失败传播部分将轨迹中激活的候选机制链与人工验证后的传播链分开。
- **证据边界：** 每个 cell 使用一个被选中的执行；补录或替换运行不是独立的重复样本，因此不能据此估计该配置的随机方差。论文明确限制了 selector recall、Judge agreement、候选链 precision 等尚未建立的属性；候选链不能直接当根因，需核对事件顺序、中间状态、端点及竞争解释。结论属于预印本定义和任务设置。
- **补充边界：** 既有[Agent trace 诊断与在线评估记录](agent-trace-diagnosis-and-online-evaluation-2026.md)覆盖主要论文摘要和归因流程；本次进一步核对“一格一次被选中执行”的限制，以免把 3,420 次执行误读成每配置多次重复试验。

### Harness-Delta Attribution：Harness 演化后的收益来源

- **来源与版本：** Ding et al., *Harness-Delta Attribution: Dissecting Gains From Harness Evolution*，[arXiv v1](https://arxiv.org/html/2610.22466v1)，2026-10-15；2026-11-18 核验摘要、§3—§5 与附录。
- **来源支持：** 作者固定模型权重，分析围绕模型演化 prompts、skills、tools 和 orchestration 的收益，尝试区分 benchmark-specific overfitting、额外 test-time compute 与残余可泛化提升。研究覆盖 11 个执行模型和 3 个演化框架；在 21 个具有显著 search-set gains 的设置中，作者的方法平均将 54% 和 14% 的收益分别归于 overfitting 和 test-time scaling，残余项被称为泛化改善的上界。held-out 转移因任务、模型和框架而异，validation 选择也未稳定消除过拟合。
- **证据边界：** 54% 和 14% 是作者 HDA 分解在所选显著收益设置上的均值，不是所有 Agent benchmark 的常量，也不是直接可迁移的通用指标。剩余项只是该方法定义下的上界，不等于已证明因果的真实泛化收益；共同存在于搜索与 holdout 的 benchmark artifact 也会削弱 holdout 解释力。

### Do More Agents Help：固定协议下的质量与 token 成本

- **来源与版本：** *Do More Agents Help? Evaluating Accuracy and Cost in LLM Multi-Agent Workflows*，[arXiv v2](https://arxiv.org/html/2606.05670v2)，2026-10-09；2026-11-18 核验摘要、§3—§4 及附录协议。
- **来源支持：** BenchAgent 用共享执行底座、工具接口、回答合同、评分和 token 记账，比对固定 GPT-4.1 下的单 Agent 参考与六种 multi-agent workflow；主研究跨 10 个固定任务集，运行 3 次，每种配置每次 3,302 个实例。作者同时报告 benchmark-specific accuracy、汇总结果和 token 使用。例如 LLM-Debate 在 HumanEval 上高于单 Agent 9.67 个百分点，但汇总准确率低 1.96 个百分点、token 用量为 1.39 倍；ChatEval 在 IFEval 上高 22.25 个百分点，汇总准确率较低且 token 用量为 4.34 倍。
- **证据边界：** 结果绑定所测 GPT-4.1、固定实现、任务集合、采样和记账口径；不能据此判断所有 multi-agent 系统优劣。辅助的单次 GAIA 及外部 runtime 案例与三次重复的主实验不是同一种证据。

## 评测组织样本

### Harvard CS2680：Agent Evaluation and Benchmarks 课程讲义

- **来源：** [CS2680 Modern AI Systems 课程讲义](https://cs2680.seas.harvard.edu/notes/agent-evaluation)，2026-11-18 阅读章节概览与 evaluation guide。
- **来源支持：** 讲义从 task、trial、trajectory、outcome 区分评测对象；说明 Agent system 包括 model、prompt、memory、tools、control loop 和 execution environment；建议对齐条件，并一起读质量、可靠性、延迟和成本。其结构包括任务格式、grader、metrics、可靠性/成本、评测设计与报告。
- **适用边界：** 这是课程讲义和知识组织案例，可说明一种教学安排和定义，不是评测学习效果的因果研究，也不规定所有教材必须采用的章节顺序。

### University at Buffalo CSE 410/510：Fall 2026 Agentic Engineering

- **来源：** [课程原始大纲与进度](https://cse.buffalo.edu/courses/cse410/2026_fall/cse410ae/)，2026-11-18 阅读课程描述、暂定日程及 Lab 4。
- **来源支持：** 课程把 Agent Evaluation 放在从单 prompt 到 task-level evaluation 的路径上，并列出 metrics、benchmarks、confidence intervals 和 bootstrapping；Lab 4 要求为有界 Agent 任务设计 test cases、success criteria、metrics、failure taxonomy 和可复现评测脚本。大纲随后把安全、multi-agent、coding agent、trace 和可靠性统计分开安排。
- **适用边界：** 这是 Fall 2026 一门课程的暂定大纲与作业说明；只能证明该课程这样组织内容，不能单独证明教学效果或普遍课程结构。

## 动态官方文档

### OpenAI Agent Evals 指南

- **来源与核验日期：** [OpenAI Agent Evals 官方文档](https://developers.openai.com/api/docs/guides/agent-evals)，2026-11-18 阅读 trace grading 与 repeatable evaluation 流程。
- **来源支持：** 当前指南说明 trace 会记录单次运行中的 model calls、tool calls、guardrails 和 handoffs；在调试工作流时先查看 trace 并按明确标准评分；当 good outcome 已定义、需要重复对比时再转向 datasets 与 eval runs。
- **证据边界：** 这是供应商对其产品工作流的说明，不是独立研究；不证明某个 grader 有效、trace 足以解释因果，也不等同于跨供应商标准。文档和产品功能可能变化，应使用时核对目标版本。
