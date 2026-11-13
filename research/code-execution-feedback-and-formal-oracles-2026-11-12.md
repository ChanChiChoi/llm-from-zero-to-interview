# Code Execution Feedback and Formal Oracles：来源证据

核验日期：2026-11-12。本记录收录代码生成中执行反馈、过程监督、性能奖励和形式验证的新近研究证据；均为 source-backed，未在本地复现实验。

## CodePRM：执行反馈进入步骤级过程奖励

- **来源与版本：** Qingyao Li et al., [CodePRM: Execution Feedback-enhanced Process Reward Model for Code Generation](https://aclanthology.org/2025.findings-acl.428/)，Findings of ACL 2025，页 8169—8182。
- **阅读范围：** ACL Anthology 正式摘要、书目信息和论文链接；未逐表核对实验数值。
- **来源支持的事实：** 作者构造带代码片段和执行反馈的 thought trace 数据，将步骤关联代码的测试通过率作为步骤标签信息；PRM 同时读取推理过程与执行反馈，为中间步骤打分；推理时使用 Generate-Verify-Refine 流程，在代码搜索中定位并修正过程错误。
- **证据边界：** 上述是作者摘要所述方法与实验结论，不支持脱离其数据、模型、评测集和生成流程的普遍收益判断。步骤分数仍取决于测试集合、轨迹切分和标注协议；摘要未提供足够信息核验全部实验细节。

## StepCodeReasoner：结构化执行状态监督

- **来源与版本：** Hao Wang, Lei Sha, and Jie M. Zhang, [StepCodeReasoner: Aligning Code Reasoning with Stepwise Execution Traces via Reinforcement Learning](https://proceedings.mlr.press/v306/wang26gn.html)，ICML 2026，PMLR 306，页 129652—129672。
- **阅读范围：** PMLR 官方论文摘要和书目信息；未逐表核对全文实验。
- **来源支持的事实：** 作者把结构化、print-based 的执行状态锚点插入代码，训练模型预测逐步运行状态；随后提出 Dual-Granularity GRPO，分别利用跨轨迹比较和轨迹内下游正确性进行 credit assignment。摘要报告特定 7B 模型在 CRUXEval、LiveCodeBench、REval、HumanEval 和 MBPP 上的结果。
- **证据边界：** 结果绑定作者模型、数据、训练流程和各 benchmark。运行状态锚点是模型可见的执行监督，不证明其等同于模型内部推理，也不能据摘要外推到仓库维护和生产执行。

## RLPF：把正确性与正确后的性能反馈分阶段处理

- **来源与版本：** Huihao Jing et al., [RLPF: Reinforcement Learning from Performance Feedback for Code Generation](https://arxiv.org/abs/2607.27271v1)，arXiv v1，2026-07-29 提交。
- **阅读范围：** arXiv v1 摘要、方法中 failure-mode/success-mode 奖励说明、PerfCodeBench 切分与指标、主表、训练/评估成本及限制讨论。
- **来源支持的事实：** 论文指出运行时间只有在程序可提取、可编译/运行且通过 oracle 后才有含义，原始时间尺度也随任务变化。方法在未正确阶段提供执行管线进度反馈，在正确阶段以 baseline、expert reference 与候选运行时间构造任务归一化效率信号。PerfCodeBench 含 1,413 个训练任务和 306 个 family-disjoint 测试任务，覆盖 C、C++、CUDA、Go、Java 和 Python。作者报告 Qwen3-32B 的 correct-and-runnable rate 从 11.1% 提升至 54.6%，CGRE 从 8.1% 提升至 38.6%；评测运行时间取三次测量的中位数。
- **证据等级：** preprint / source-backed（作者论文报告，尚非同行评审结论）。
- **适用边界：** 数值只对应该模型、LoRA/RL 配置、PerfCodeBench harness 和其测试切分。作者把跨基准收益描述为 modest，并指出消融增益不构成统计显著性证明；运行时测量受硬件和执行条件影响，不支持把原始秒数作为跨任务通用奖励。

## SWE-Proof：仓库修复中的机器检查证明与规格瓶颈

- **来源与版本：** George Ma et al., [SWE-Proof: Can Language Models Resolve Real-World Issues with Machine-Checked Proofs?](https://arxiv.org/abs/2609.21190v2)，arXiv v2，2026-09-22 修订。
- **阅读范围：** v2 摘要、Benchproofer 构造方法与“Verify ⇒ Resolve”条件、SWE-bench Verified / Pro 规模表、Claude Opus 4.8 评测和讨论限制。
- **来源支持的事实：** SWE-Proof 为 SWE-bench Verified 的 500 个 issue 任务构造正式规格、可验证实现和证明，并使用 Nagini、Velvet、Lean 等验证后端；通过限制验证到新增/修改代码并为未修改函数提供公理摘要来控制规模。作者报告，在其 Claude Opus 4.8 设置中，约四分之一通过隐藏测试的补丁仍有形式缺陷；提供正确形式规格时解析率从 85.0% 升至 95.1%；只有 56% 的 agent 生成规格通过其对抗审计，未解析任务的规格更常失败。
- **证据等级：** preprint / source-backed（2026-09 arXiv v2 作者报告）。
- **适用边界：** 证明结论条件于形式规格、公理、验证后端及验证实现与仓库补丁等价的检查。论文显示规格忠实性本身是瓶颈；不能把形式证明当成已证明规格完整表达了自然语言需求。数字受其模型、任务集合和构造门禁限制，不能推广为所有仓库任务的比例。

## 综合边界

执行反馈可以从最终通过/失败细化到错误阶段、步骤级过程、正确后的效率和形式化性质，但这些信号各自依赖不同的 oracle。测试只覆盖抽样输入；过程分数依赖轨迹与测试质量；运行时反馈受正确性门槛和任务尺度影响；形式证明则受规格、公理与实现等价关系约束。上述论文提供互补设计证据，不构成一套统一训练配方，也未在此处进行本地复现。
