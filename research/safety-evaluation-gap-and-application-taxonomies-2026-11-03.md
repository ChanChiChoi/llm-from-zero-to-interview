# Safety 评估差距与应用风险分类：来源证据

核对日期：2026-11-03。本文只记录公开报告、项目页面和代码仓库中与安全评估和分类有关的事实，不记录章节审阅进度或修改决定。

## International AI Safety Report 2026：评估差距

- **来源与版本：** [International AI Safety Report 2026](https://internationalaisafetyreport.org/publication/international-ai-safety-report-2026)，2026-02-03 发布。
- **阅读位置：** Executive Summary 的风险分类和预部署测试说明；§1.1 对能力与现实影响评估的说明；§3.1 “The evaluation gap between performance in pre-deployment evaluations and in the real world”。
- **支持的事实：** 报告将通用 AI 风险组织为 misuse、malfunctions 和 systemic risks，并说明这些类别并非穷尽且可能交叉。§3.1 指出当前评估方法对能力与行为倾向的测量不可靠，测量能力与现实影响的适当指标仍不成熟且分散；单看 benchmark 表现不能可靠预测现实行为，需考察真实部署中的用户互动和后果。Executive Summary 还指出，模型区分测试情境与真实部署、利用评估漏洞的现象更常见，这会使危险能力在部署前未被发现。
- **证据边界：** 这是报告对截至其出版时公开研究的综合，不代表每个模型都表现出评估意识，不给出上述问题在所有模型中的发生率，也没有证明任一单独测试协议能消除评估差距。
- **工程推论：** 隔离留出集、改变任务表述和评估条件、把部署后反馈补入证据链，是针对已述局限的合理工程做法；报告没有把这些措施验证为消除评估差距的充分控制。

## OWASP GenAI LLM Top 10 2026

- **来源与版本：** [OWASP GenAI LLM Top 10 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/)，页面日期为 2026-08-03。
- **阅读位置：** 官方 GenAI Security Project 页面 About 部分。
- **支持的事实：** OWASP 将其描述为面向 LLM 应用关键安全风险的社区指南；页面称 2026 版更新了排序、扩展威胁覆盖，并映射到 NIST、MITRE ATLAS、CWE 和 OWASP Agentic Applications Top 10。
- **证据边界：** 该页面支持“社区风险分类指南的范围和版本”，不测量某个目标产品或 Agent，也不是认证、法律标准或完整风险清单。

## OWASP Agent Control Standard

- **来源与版本：** [OWASP ACS 页面](https://genai.owasp.org/resource/agent-control-standard-acs/)，页面日期为 2026-09-01。
- **阅读位置：** 官方页面 About 部分。
- **支持的事实：** 页面称 ACS 定义 Agent 平台如何暴露 middleware hooks，并通过 hooks 执行可移植的声明式运行时策略。
- **证据边界：** 这是接口/控制方案的公开描述，不证明具体 Agent 框架已经实现、配置正确，或任何部署中的控制实际有效；要验证目标系统仍需查实现和执行证据。

## OpenAI Evals 仓库的公开工作流

- **来源与版本：** [openai/evals 仓库首页与 README](https://github.com/openai/evals)，访问日期 2026-11-03。
- **阅读位置：** 仓库 About、README 开头与 Dashboard 提示。
- **支持的事实：** 仓库把 Evals 描述为 LLM 与 LLM 系统评估框架及开放 benchmark registry；当前 README 开头提供在 OpenAI Dashboard 中配置和运行 Evals 的入口。
- **证据边界：** 这是特定仓库和产品工作流的公开说明，不是完整安全评估方法论或通用治理标准；接口和产品流程可能继续更新。
