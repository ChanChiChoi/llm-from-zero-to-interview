# 安全审核系统与对抗评估证据

**核验日期：** 2026-11-05  
**范围：** AI 对抗性威胁分类、OWASP 2026 LLM/Agent 安全指南、红队运行手册和提示注入 benchmark 覆盖。本文只记录来源事实、版本及证据边界，不记录章节审阅、修改或任务进度。

## NIST AI 100-2e2025

- **来源与版本：** NIST，*Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations*，AI 100-2e2025，2025-03-24 final；[NIST CSRC 正式页面](https://csrc.nist.gov/pubs/ai/100/2/e2025/final)，DOI [10.6028/NIST.AI.100-2e2025](https://doi.org/10.6028/NIST.AI.100-2e2025)。
- **实际阅读范围：** 正式页面的版本/摘要、报告 Executive Summary，以及生成式 AI 攻击 taxonomy 的 Figure 2 与页面说明。
- **来源支持的事实：** 报告以系统类型、机器学习生命周期阶段、攻击者目标/能力/知识组织 adversarial machine learning（AML）术语与攻击 taxonomy；覆盖预测式和生成式 AI、训练/测试/部署过程及嵌入模型的系统上下文。摘要说明报告不为具体组织或应用设定可接受风险阈值。
- **证据边界：** NIST 报告是术语与风险管理参考，不是法规、产品安全认证或某类攻击的完整穷举。页面记录了 2025-04-01 更正 PDF 和 2025-06-03 潜在更新说明；本记录以核验日公开版本页为准。
- **证据等级：** NIST 官方技术报告；支持 taxonomy、定义和作者范围，不证明某个安全控制有效。

## OWASP GenAI LLM Top 10 2026

- **来源与版本：** OWASP GenAI Security Project，*OWASP Top 10 for LLM Applications 2026*；[资源页](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/)，[2026 版 PDF](https://genai.owasp.org/download/56857/)。资源页显示日期 2026-08-03；所下载 PDF 标为 Version 2026，但其 revision history 仍写 `[2026 release date]`，存在发布日期元数据不一致。
- **实际阅读范围：** 资源页、PDF “What’s New in the 2026 Top 10”及 LLM01:2026 Prompt Injection（PDF §§开篇、LLM01 描述和缓解）；并核对其与 OWASP Agentic Top 10 的范围区分。
- **来源支持的事实：** 2026 版扩展风险覆盖并更新排名；PDF 将提示注入描述为直接/间接输入、工具输出、多模态内容和持久状态等输入边界问题，并指出应用应通过架构控制限制模型输出能影响的权限和后续组件。PDF 说明其排序结合社区投票和公开事件记录。
- **证据边界：** 这是社区指南和分类，不是强制标准、完整的事件总体或产品验收证据。PDF 的公开事件记录和专家投票方法不能直接给出部署系统的攻击率、误报率或安全阈值；发布日期按网页与 PDF 的可见元数据分别记载。
- **证据等级：** OWASP 社区项目正式指南；支持风险类别和项目建议，不独立证明缓解控制的效果。

## OWASP Top 10 for Agentic Applications 2026

- **来源与版本：** OWASP GenAI Security Project，*OWASP Top 10 for Agentic Applications 2026*，Version 2026，PDF 标示 2025-12；[正式资源页](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)，[PDF](https://genai.owasp.org/download/52117/)。
- **实际阅读范围：** 正式资源页及 PDF 前 14 页，包括版本、范围、风险概览、ASI01 Goal Hijack 与 ASI02 Tool Misuse/Exploitation 的描述和缓解指导。
- **来源支持的事实：** 风险清单包含 Agent Goal Hijack、Tool Misuse and Exploitation、Identity and Privilege Abuse、Supply Chain Vulnerabilities、Memory and Context Poisoning、Insecure Inter-Agent Communication、Cascading Failures、Human-Agent Trust Exploitation 等类别。缓解指导提到 least agency/least privilege、逐工具权限、动作级身份验证与高影响动作人工批准、执行前策略闸门、能力预算和工具调用监控；其 intent capsule 被称为 emerging pattern。
- **证据边界：** 该清单是社区指南，不是认证或控制有效性实验。Intent capsule 是文档举例的新兴模式，不是标准协议；具体权限、审批、预算和日志设计仍须按系统威胁模型验证。
- **证据等级：** OWASP 社区指南；支持风险分类与设计建议，不证明具体实现安全。

## OWASP GenAI Red Teaming Manual 1.0

- **来源与版本：** OWASP GenAI Security Project，*GenAI Red Teaming Manual Version 1.0*，资源页日期 2026-10-20；[正式资源页](https://genai.owasp.org/resource/genai-red-teaming-manual-version-1-0/)。
- **实际阅读范围：** 正式资源页的标题、日期和摘要；PDF 下载请求超时，本记录未据未读 PDF 声称其细节。
- **来源支持的事实：** 资源摘要称该手册提供八阶段方法：planning、reconnaissance、attack surface mapping、exploitation、persistence、impact assessment、reporting、remediation；覆盖模型、应用、多模态接口和 Agent，并强调授权/伦理测试、可复现发现、可衡量结果与防守运营集成。
- **证据边界：** 仅依据资源页摘要；未核验 PDF 中各阶段的详细操作、版本细节或实测效果。它是运行流程参考，不是特定系统的风险保证。
- **证据等级：** OWASP 项目官方发布页，支持页面公布的范围和流程概述。

## Nine Dimensions of Prompt Injection Attacks

- **来源与版本：** Jiamin Chang 等，*The Devil is in the Context: Nine Dimensions of Prompt Injection Attacks*，arXiv:2611.04672v1，2026-11-03 提交；[摘要页](https://arxiv.org/abs/2611.04672)，[HTML 全文](https://arxiv.org/html/2611.04672v1)。
- **实际阅读范围：** 摘要、§§I—III（九维 taxonomy 和标注信息需求）、§V（九个入选 benchmark 的覆盖分析）及结论。论文为预印本；核验时未确认同行评审状态。
- **来源支持的事实：** taxonomy 将样本与授权用户任务及系统上下文一起分类，维度包括来源信任、fragmentation、展示可见性、可解释性、条件触发、Agent 状态修改、主题关系、意图效果和原任务完成。作者在其选定的九个 benchmark 和标注方案中，计算 144 个两维类别组合后报告 19 个未覆盖（13.2%）；其特定较低覆盖类别组合为 15/36（41.7%）。作者明确称 taxonomy 可扩展，不是穷尽分类。
- **证据边界：** 这是新提交的预印本和作者自定 taxonomy；覆盖比例只绑定论文选择的九个 benchmark、taxonomy 和标注单元，不能外推到所有安全评测，也不是攻击成功率。该数字只能支持“样本量不等于组合覆盖”这一有限观察。
- **证据等级：** arXiv 预印本；支持作者提出的 taxonomy 和该研究协议下的覆盖分析，不代表领域共识。
