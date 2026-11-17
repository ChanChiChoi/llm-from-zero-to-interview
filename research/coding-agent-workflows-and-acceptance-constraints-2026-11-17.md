# Coding Agent 工作流与验收约束：来源证据

检索与阅读日期：2026-11-17。下列资料用于说明代码任务工作流、Agent 工具接口和功能测试之外的验收约束。产品文档会更新；论文结果绑定各自的版本、任务和评测设置。

## Anthropic Claude Code：Common Workflows

- 来源：[Common workflows](https://code.claude.com/docs/en/common-workflows)，在线产品文档，访问日期 2026-11-17。
- 阅读位置：代码库探索、定位相关代码、Refactor code、Work with tests、Plan before editing、Resume previous conversations、parallel sessions with worktrees 等小节。
- 来源支持：页面提供按任务组织的操作示例，包括探索代码、修复 bug、重构、测试、恢复会话、计划后再编辑，以及通过 worktree 隔离并行会话；重构示例建议分成小步并运行测试。
- 边界：这些是 Claude Code 产品工作流示例，不是独立对照实验，也不构成对其他 Agent、仓库或所有任务都最优的证明。并行 worktree 仅隔离不同 checkout；共享服务、数据库和外部资源仍需各自管理。
- 证据等级：官方产品文档，source-backed。

## OpenAI Codex：Using PLANS.md for multi-hour problem solving

- 来源：[Using PLANS.md for multi-hour problem solving](https://developers.openai.com/cookbook/articles/codex_exec_plans)，发布于 2025-10-07；访问日期 2026-11-17。页面注明该 recipe 已归档，并可能引用过时模型或 API。
- 阅读位置：Observable outcomes、Specify repository context、Validation、Capture evidence、Living plans and design decisions 等段落。
- 来源支持：计划示例要求写明用户可观察的结果、仓库路径、准确的测试命令、工作目录、预期输出及错误解释；长任务计划是可随证据和决定更新的文档。
- 边界：这是 Codex 的特定操作 recipe，且页面已归档。适合引用其计划和验证原则作为一个产品设计示例；不能引用其中的旧模型/API 建议作为当前产品事实或唯一工作流格式。
- 证据等级：官方归档 Cookbook，source-backed；模型和 API 相关内容不用于本章结论。

## SWE-agent：Agent-Computer Interface

- 来源：[SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793v3)，arXiv v3，最后修订于 2024-11-11。
- 阅读位置：摘要及版本记录；论文研究和实验结果按 v3 固定版本查看。
- 来源支持：论文把 Agent-Computer Interface（ACI）作为软件工程 Agent 的专用交互界面进行研究；报告其 ACI 设计影响论文所测任务的行为和表现，并在 SWE-bench、HumanEvalFix 上进行评估。
- 边界：结果受论文中的 Agent、接口、模型、基准和配置限制。它支持“接口设计是值得测量的变量”，不证明任何某种工具界面在所有任务上都会提升表现。
- 证据等级：同行评审前研究论文，source-backed。

## SWE-Gate：功能测试与 review constraints

- 来源：[SWE-Gate: Passing Functional Tests Is Not Enough for Software Engineering Agents](https://arxiv.org/abs/2609.04167v1)，arXiv v1，提交日期 2026-09-03。
- 阅读位置：摘要；基准构造、实验结果和局限章节；版本记录。
- 来源支持：SWE-Gate 从真实 PR review comments 中抽取约束，并为任务提供功能测试和约束测试；数据集包含 303 个仓库级修复任务，覆盖 75 个开源 Python 仓库。摘要报告在四个 LLM backends 的共同 coding-agent scaffold 上，644 个通过功能测试的 repairs 中有 221 个未满足给定 review constraints；221/644 约为 34.3%。
- 边界：该比例是论文指定的 Python 仓库、合成修复实例、约束测试和 Agent scaffold 下的结果，不是所有 Coding Agent 的一般失效率，也不能直接外推到生产仓库。功能测试未覆盖的 review constraints 依赖该基准定义及验证实现。
- 证据等级：预印本基准论文，source-backed。

## ANCHOR：轨迹与工程约束评估

- 来源：[ANCHOR: Assessing Naturalistic Constraint Hierarchies and Ordering in Repository-Grounded Agentic Coding](https://arxiv.org/abs/2610.16859v2)，arXiv v2，最后修订日期 2026-10-14。
- 阅读位置：摘要、基准设计、约束分类、trajectory-grounded evaluation、结果和 limitations。
- 来源支持：ANCHOR 在 358 个任务、8 个语言组和 19 个 Agent configuration 上研究工程约束遵循；基准把原子约束组合成 AND/THEN/PRIORITY 关系，并结合规则与 LLM judge 检查最终代码 diff 和执行轨迹。所列约束涵盖修改范围、工作流顺序、验证、工具使用和报告。
- 边界：这是以两种开源 scaffold 和指定基础模型配置为对象的基准研究；任务约束由研究流程生成和审核，评判器也有其错误边界。结果用于说明评估可同时看 artifact 与轨迹，不构成通用工作流规范或生产成功率。
- 证据等级：预印本基准论文，source-backed。

## 证据边界与未核验项

- Claude Code 页面是动态文档，所述行为与页面访问日绑定；如后续正文要陈述准确产品能力，需重新检查具体文档版本。
- OpenAI PLANS recipe 页面明确标注归档；本文仅取计划可观察结果、命令、工作目录和验证证据等稳定设计要点，不取旧模型或 API 建议。
- SWE-Gate 和 ANCHOR 均为近期预印本研究；其 benchmark 设计、任务抽样、自动评判及构造约束的有效性仍需后续复核，不能升级为普遍的产品规律。
- 本记录未执行本地 agent、测试、benchmark 或模型；所有章节教学轨迹和手算账本均为合成材料，不是论文复现或本地测量。
