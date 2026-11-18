# 评估平台与裁判可靠性来源

检索与阅读日期：2026-11-05。

本记录只保存联网发现的评估方法、框架事实及其证据边界，不记录章节审阅或修改过程。

## LLM 裁判 rubric 伪线索研究

- **来源：** [Judging LLM-as-a-Judge: Concerning Rubric Artifacts in LLM-based Automated Text Generation Evaluation](https://arxiv.org/abs/2609.02942v1)，arXiv v1，2026-08-31 提交；页面标明 accepted at EMNLP 2026。
- **阅读范围：** 摘要、§§3—5、§7 limitations、实验附录 A。
- **来源支持：** 论文在 HealthBench、HealthBench-Eval/Hard 与 ResearchRubrics 的特定样本、LLM 裁判和候选模型上训练 rubric-only probe；probe 不读取候选回答，却能高于随机水平预测部分裁判标签。反事实试验固定 rubric 而反转回答时，493 对有效样本中只有 37.7% 的裁判输出按预期翻转；另一个试验固定回答并反转 rubric 标准，所测候选/裁判组合的预期翻转率为 16.8% 或 32.2%。
- **证据等级：** source-backed（作者报告的实验设计、数据、模型与结果；未本地复现）。
- **边界：** 这是一项诊断性研究，使用有限的英语医疗与研究问答数据、特定模型和二元标签；作者明确指出尚不能确定全部因果机制，且结果未必推广到其他领域、语言或 rubric。它支持把 rubric 视为可能影响测量的变量并做反事实校准，不证明所有 LLM 裁判都不可靠。
- **未解决问题：** 需在客服事实核验、工具轨迹、不同语言、不同 judge 和人类标签上分别验证；rubric-only 探针的效能及成本也待独立评估。

## τ²-Bench 的双控制对话评测

- **来源：** [τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment](https://arxiv.org/abs/2506.07982)，arXiv，2025-06-09。
- **阅读范围：** 摘要、引言、任务构造、实验设置与结果、user simulator 质量审查、结论及局限。
- **来源支持：** 在原有 tool-agent-user 任务之外，论文构造 telecom 双控制环境：Agent 和模拟用户各自通过受限工具作用于共享动态状态；可组合任务由初始化、解决和断言函数描述，并比较默认协作、去除用户交互和提供 oracle plan 等模式。论文为 50 条 telecom 对话报告 3 次关键模拟用户错误，提醒仿真器本身也需计入评估可靠性。
- **证据等级：** source-backed（论文设置和报告结果；未本地复现）。
- **边界：** 任务是受控 telecom 环境，用户由模型模拟，工具、政策和任务由研究团队构造；作者指出模拟器仍有错误、扩展域依赖专家，且未直接建模客服中的 expert-novice gap。它可启发多方状态和模拟器质量的评测设计，不能代表真实客服成功率。
- **未解决问题：** 双控制与单控制差异在其他业务、真实用户和真实工具副作用下的外部效度尚未确定。

## Inspect AI 官方评估框架文档

- **来源：** [Tasks](https://inspect.aisi.org.uk/tasks)、[Scoring Policy](https://inspect.aisi.org.uk/scoring-policy.html)、[Scoring](https://inspect.aisi.org.uk/scoring.html)，UK AI Security Institute 官方在线文档；读取日期 2026-11-05，页面未固定软件版本。
- **阅读范围：** Task 最小组成及运行选项；模型输出判错、评分故障、unscored 及其分母行为；重评分和多 scorer 结果。
- **来源支持：** Inspect 将评估任务建模为 dataset、solver、scorer 的组合，可另设 sandbox、limits、model roles 和 metrics。评分政策把被测模型的不正确/不可用输出留在对应指标分母中，把 scorer 执行异常记录为 sample error；无法作出判定的 scorer 可产生带原因的 unscored 结果，不应与模型答错混为一类。
- **证据等级：** source-backed（特定开源评估框架的当前公开接口/文档）。
- **边界：** 这是框架设计示例，不规定通用生产服务的组件边界；在线文档未固定版本，具体 API 会变化。它不能证明某 scorer 有效或数据集具代表性。
- **未解决问题：** 目标项目若采用该框架，仍需绑定代码 revision、配置、插件和执行环境，并另行验证分母与错误分类是否符合业务决策。

## NIST 生成式 AI 风险管理画像

- **来源：** [NIST AI 600-1, Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile](https://doi.org/10.6028/NIST.AI.600-1)，2024-07-26 发布；NIST 页面显示 2026-04-08 更新。
- **阅读范围：** NIST 出版页元数据；PDF 中 Measure 2.13、相关 TEVV 与测量误差建议。
- **来源支持：** 报告将该画像定位为 AI RMF 1.0 的生成式 AI 跨行业 companion profile；Measure 2.13 建议对 TEVV 度量和流程的有效性进行评估与记录，其中包括建立测量误差模型以论证构念效度，并记录指标或结构化人工反馈的偏差与统计方差。
- **证据等级：** source-backed（NIST 发布的自愿风险管理框架及其建议）。
- **边界：** 这是治理建议，不是强制标准、产品认证、特定评分器或发布阈值；组织仍须按领域与风险定义具体指标。
- **未解决问题：** NIST 建议如何映射为某一企业客服系统的具体验收门槛，需由业务、风险和领域负责人共同确定。

## 2026-11-18 补充核验：评估产品生命周期与确定性测试边界

### OpenAI Evals 平台生命周期

- **来源与核验日期：** OpenAI 官方 [Working with evals](https://developers.openai.com/api/docs/guides/evals)，2026-11-18 访问。
- **来源支持的事实：** 当前动态指南说明 OpenAI 正在弃用 Evals 平台，已有 evals 从 2026-10-31 起只读，计划于 2026-11-30 关闭；指南建议新建评估的用户考虑 Datasets。
- **证据边界：** 这是单一供应商对平台生命周期的当前说明，不表示数据集、grader、运行与对照等一般评估概念失效，也不能据此推断其他平台或本地 harness 的生命周期。

### OpenAI Agents SDK 的内存态测试与集成测试边界

- **来源与核验日期：** OpenAI Agents SDK 官方 [Testing 文档](https://openai.github.io/openai-agents-python/testing/)，2026-11-18 访问；阅读测试范围、选择测试边界和 sandbox recipe 的说明。
- **来源支持的事实：** 文档描述 deterministic、provider-neutral 的 testing utilities 在内存中运行，不发出 model、sandbox-provider 或 Realtime API 请求；可测试 SDK/应用拥有的规范化 orchestration 交互，如工具执行、handoff、guardrail、重试、session 等。文档要求由真实 provider adapter 或 integration environment 覆盖真实模型、provider protocol、网络、sandbox provider、进程执行、文件系统保真、持久化、资源限制和隔离行为。
- **证据边界：** source-backed（OpenAI Agents SDK 动态文档）。这是该 SDK 当前测试工具的范围，不是通用测试金字塔，也不证明任何脚本测试已覆盖外部组件。使用时需要核验锁定 SDK 版本、instrumentation 配置和真实集成环境。
