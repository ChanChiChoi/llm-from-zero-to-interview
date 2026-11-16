# Agent 身份、授权与运行时控制资料来源（2026-11）

记录日期：2026-11-16。本文只存联网发现的来源事实、版本/状态和证据边界；书册审视、修改决定与交接记录在 `progress/`。

## NIST Agent Standards Initiative

- 来源：[NIST AI Agent Standards Initiative](https://www.nist.gov/super-intelligence/si-agent-standards-initiative)。页面标注创建于 2026-02-17、更新于 2026-11-09；阅读日期：2026-11-16。
- 来源支持：页面将该倡议描述为促进产业主导标准、社区协议和身份/认证研究的计划，并提到自愿指南、差距分析及安全评测等工作方向。
- 证据等级：NIST 项目状态页面；支持该倡议公开目标和当前项目状态。
- 边界：该页面不等于发布了 Agent 身份或授权标准，也不规定企业实施的统一协议、产品或控制阈值。

## NIST 软件与 AI Agent 身份/授权项目

- 来源：[Comments on Software and Agentic AI Identity Concept Paper](https://www.nist.gov/news-events/news/2026/09/comments-software-and-agentic-ai-identity-concept-paper)；[Summary of Comments](https://pages.nist.gov/nccoe-ai-identity/summary-of-comments.html)。核对日期：2026-11-16。
- 来源支持：NIST 于 2026-09-29 表示，概念论文收到 600 多条意见，NCCoE 将以软件开发生命周期中的 DevSecOps 作为首个实现用例，并计划另行发布项目范围、用例、架构和标准的草案供反馈。评论汇总记录了身份绑定、短期凭证、任务级授权、委派链、审计、跨信任边界和多 Agent 等关注点；意见中偏向沿用现有身份标准并按场景扩展，具体实现方式没有共识。
- 证据等级：NIST 对项目状态的官方说明；评论汇总是公众意见的整理，不是正式标准或 NIST 对每条建议的背书。
- 边界：不能把意见者提出的短期凭证、签名意图、控制平面或协议扩展写成已发布要求。多 Agent、外部 Agent 和消费者 Agent 的实施范围仍待项目后续确定。

## OWASP Agentic Applications Top 10 2026

- 来源：[OWASP Top 10 for Agentic Applications for 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)。页面标注发布日期：2025-12-09；核对日期：2026-11-16。
- 来源支持：该资料是 OWASP GenAI Security Project 汇集专家审阅的 Agentic 应用风险分类与缓解参考，面向规划、行动及跨工作流的系统。
- 证据等级：社区安全项目发布的风险指南。
- 边界：风险分类不是法规、独立渗透测试结果或某产品的缺陷证明；个别系统仍需按数据、身份、权限、工具、运行环境和副作用建模与测试。

## OWASP Agent Control Standard（ACS）

- 来源：[Agent Control Standard (ACS)](https://genai.owasp.org/resource/agent-control-standard-acs/)；[项目仓库](https://github.com/GenAI-Security-Project/agent-control-standard)。页面标注加入 OWASP GenAI Security Project：2026-09-01；仓库内容核对日期：2026-11-16。
- 来源支持：ACS 把 Agent 运行时拦截点、策略决策和可追踪性作为开放控制接口方向。访问时仓库示例描述 `v0.1.0` 的 wire/schema，并展示 allow、deny、modify、ask、defer 等处置类型。
- 证据等级：社区项目说明和其公开源码/规范；仓库默认分支是动态资料，未按固定 commit 作全量规范审查。
- 边界：仓库在访问时明确声明其参考实现有缺口：wire 尚未认证、默认故障姿态为 proceed（fail-open），19 个 hook 中只有 2 个在该参考实现里做了 live evaluation。不能把参考实现演示、项目名称或 ACS 页面描述外推为生产成熟度、跨框架完整互操作性或经过独立验证的通用标准；采用时须固定版本并审阅规范、实现和威胁模型。
