# 第八册 AI Safety 与 Alignment 架构扫描（2026-10-27）

## 范围与状态

本记录用于第八册架构审视的外部发现阶段，依据当前 `WRITING_SPEC.md`，核对 `BOOK_SERIES.md`、`ROADMAP.md`、第七/八/九册简介、第八册简介与目录，以及第八册 16 篇正文的标题结构。此处形成的是书册契约初稿、候选主题去向和章节分工图；第 2—16 章的逐篇通读、篇幅统计、章级检索与内容去向复核尚未完成，因此不作为整册架构门禁通过记录。

检索日期为 2026-10-27，时间窗口以 2025-10 至 2026-10 的新材料为主，并回看仍影响当前方法边界的基础框架、前沿评估与原始研究。检索问题包括：Agent 从文本响应转为持续行动后，安全责任新增在哪些系统边界；上线前评估与实际行为之间的证据缺口是什么；组织如何把评估结论转换成受限发布、访问策略和事故响应；有哪些成熟度足以进入主书的新方向。

## 外部扫描与证据边界

| 来源 | 本次读取支持的判断 | 证据范围与限制 |
| --- | --- | --- |
| [International AI Safety Report 2026](https://internationalaisafetyreport.org/publication/international-ai-safety-report-2026)，特别是风险管理、Agent 自主性和评估缺口部分 | 将风险区分为 misuse、malfunctions 与 systemic risks；指出 Agent 自主行动增加干预难度，发布前评测对真实风险的预测能力有限；多层风险缓解和事故监测仍需发展 | 跨领域研究综述，不是具体产品风险评级；报告本身指出真实部署缓解效果和风险发生率的证据不足 |
| [OWASP Top 10 for LLM Applications 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/) 与 [Agent Control Standard](https://genai.owasp.org/resource/agent-control-standard-acs/)（2026-09） | 应用安全分类继续演进；ACS 将声明式策略与 Agent 平台 middleware hook 关联，为运行时控制提供接口方向 | OWASP 是社区安全指导。它适合发现威胁和控制候选，不证明某个实现已正确强制权限，也不提供特定 Agent 的事故概率 |
| [METR Frontier Risk Report (February to March 2026)](https://evals.alignment.org/blog/2026-05-19-frontier-risk-report/) | 前沿风险评估开始检查 Agent 在开发者内部使用中的错配行为、过度授权和隐瞒风险；报告讨论了识别长期错配意图的标准缺口及外部资料可见性问题 | 与多家开发者合作的试点评估，适用于报告中的评估窗口、参与系统和访问条件；不能外推成所有前沿模型的发生率 |
| [AgentCanary: A Security Evaluation Framework for Autonomous AI Agents in Real Executable Environments](https://arxiv.org/abs/2606.10484) | 提议在可执行环境中评估完整 Agent 轨迹，并分开观察 outcome safety、security awareness 与 task utility | arXiv 预印本；结论绑定其数据、模型、攻击方法和 Agent 框架。适合作为第六/十三章方法候选，不作为通用 benchmark 标准 |
| [Assessing Confidence in Frontier AI Safety Cases](https://arxiv.org/abs/2502.05791) | 使用结构化 claim、argument、evidence 安全论证，并讨论如何表达论证叶节点的信心与反证问题 | 单一 cyber misuse inability argument 的方法示例；作者也指出量化 confidence 困难，不能把 safety case 当成安全保证 |
| [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) 与 [NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) | 提供生命周期风险管理活动的组织入口 | 属自愿风险管理框架，不提供模型行为、产品权限或生产安全的实测证明；NIST 页面截至检索日说明 AI RMF 1.0 正在修订 |

## 新主题候选及去向

1. **Agent 身份、委托、资源范围与运行时权限。** 目标问题是“这次动作由谁代表谁、针对哪个资源、依据什么证据获准”。OWASP ACS 提供社区接口方案；国际报告和 METR 说明 Agent 自主性与监督的现实难点。第十五章已有身份、委托、范围、PDP/PEP、执行前状态检查和审计主线，第五章处理不可信输入如何诱发授权绕过，第十四章覆盖分类器到策略动作的分层；因此并入这些章节，并把 Agent loop、harness 与协议实现交给第十七、二十、二十二册，不新增独立章节。
2. **面向完整 Agent 轨迹的安全评估。** 目标问题是“仅评最后回答时，会漏掉哪些中间工具动作和外部状态变化”。AgentCanary 是可供比较的新预印本；第六章已有 red teaming、危险能力、capability elicitation、发布门禁，第十三章已有完整轨迹测试和自动化对抗闭环。候选主题适合并入第六/十三章，明确模型回答、工具动作、状态后果及任务效用的不同判定对象；不新增章节，也不将单个预印本写成共识。
3. **Safety case / assurance case。** 目标问题是“发布者如何把安全主张、推理步骤、证据、假设和反证连成可审计论证”。第十一章已有 Model Card、System Card、证据覆盖、范围限定和发布决定；候选落点为第十一章发布门禁与事故响应，作为评估证据的组织形式，而不是新的模型卡替代物。信心量化的困难和样例适用范围必须同时写清。
4. **影响层级与技术故障位置的双轴风险地图。** 国际报告的 misuse/malfunction/systemic risk 与第八册现有的 prompt injection、privacy、tool misuse 等技术标签回答不同问题。已并入第一章风险分类段落；题库、练习和术语表同步该双轴读法。
5. **从行为规范到运行时控制接口。** Model Spec / 行为规范回答“模型应如何回应”；ACS 描述 runtime hook 的政策执行接口。已在第一章指向第十五章，并计划在第十一/十五章逐篇扫描时核对版本、责任边界与证据。社区标准本身不能证明控制有效。

## 初步判断

第八册目前的 16 章已形成从目标、监督、优化风险、攻击、评估、内部控制、隐私到治理的主线；第十三至十六章承载自动对抗评估、多模态分类、风险校准访问和安全降级。初步扫描未发现必须独立新增的全书主线。原目录将后四章称作 “Frontier Safety 专题” 容易让读者误以为它们都只谈前沿模型风险，故改为“安全工程专题”。

主要重复风险是：第一章与第三章重复介绍 RLAIF / Constitutional AI；第五、六、十三章在攻击、红队和评估之间交界；第十一章与第十五、十六章在治理决定、访问策略和 fallback 之间交界；第九章的删除/遗忘与第十章训练隐私/应用权限之间交界。架构决定是先保留 16 章及主线顺序，以唯一主讲位置和交叉链接控制重复；待逐篇通读后再确认是否需拆分、重排或迁移。

仍开放的工作：逐篇确认 16 个正文的中心问题、前后依赖、各独立讲解单元篇幅、数值案例、视觉机会、现有研究入口、代码状态和原内容去向；对第 2—16 章分别开展章级扫描；最后沿第 1—12 主线及第 13—16 工程专题连续复读。后续若新来源显示 Agent 安全、治理或行为控制已跨越现有边界，应回到目录重新决策。
