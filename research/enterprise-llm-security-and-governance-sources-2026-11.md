# 企业 LLM 安全与治理来源记录

检索与阅读日期：2026-11-16。范围：企业资源访问、生成式 AI 风险治理、LLM 应用安全风险分类和一种官方 RBAC 实现示例。记录的是联网核验到的事实和证据边界，不记录书稿审阅进度或修改决定。

## 核验事实

### NIST SP 800-207：Zero Trust Architecture

- 来源：[NIST CSRC 发布页](https://csrc.nist.gov/pubs/sp/800/207/final)，最终版发布日期 2020-08-11；阅读发布页摘要、定义和文档状态。
- 支持事实：零信任将防护重点从静态网络边界转向用户、资产和资源；不能只因网络位置或资产归属而默认信任用户/设备；身份验证和授权是在建立对企业资源访问前分别执行的功能。
- 证据等级：NIST 官方架构出版物，适合支持零信任概念和架构原则。
- 边界：不是 LLM 应用的部署指南、渗透测试结果或具体授权策略；不能单独证明一个产品实现了零信任。

### NIST AI RMF 与生成式 AI Profile

- 来源：[NIST AI Risk Management Framework 状态页](https://www.nist.gov/itl/ai-risk-management-framework)；[NIST AI 600-1 出版物](https://doi.org/10.6028/NIST.AI.600-1)。AI 600-1 于 2024-07 发布；阅读简介、风险定义与资料范围。NIST 页面于 2026-11-16 标明 AI RMF 1.0 正在修订。
- 支持事实：AI 600-1 是面向生成式 AI 的跨行业 profile，作为 AI RMF 的配套资料；说明组织可按目标、风险容忍度和资源调整风险管理。AI RMF/该 Profile 是自愿使用资料，非自动满足法律义务或取得安全认证。
- 证据等级：NIST 官方框架与技术出版物。
- 边界：Profile 自身说明其讨论的风险、实证基础和建议范围有限；实际风险和合规适用性需按模型、系统、使用场景和司法辖区判断。AI RMF 1.0 页面已说明正在修订，引用需记录版本和日期。

### OWASP GenAI LLM Top 10 2026

- 来源：[OWASP GenAI LLM Top 10 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/)；项目公告：[2026 版本发布公告](https://genai.owasp.org/2026/09/01/owasp-genai-security-project-unveils-2026-top-10-for-llm-applications-new-agent-control-standard-and-sponsors-as-community-tops-30000-members/)。阅读资料页简介及公告中版本说明。
- 支持事实：截至检索日，2026 版是该项目最新的 LLM 应用风险指南；官方说明其为社区编制，覆盖风险、攻击场景与缓解措施，并与若干安全/风险框架建立映射。
- 证据等级：OWASP GenAI Security Project 社区指南；适合用于发现、分类和讨论应用安全风险。
- 边界：不是法规、产品认证或完整威胁模型；排名和风险覆盖不代表某组织的风险优先级。

### Microsoft Azure RBAC 角色分配

- 来源：[Microsoft Learn：Understand Azure role assignments](https://learn.microsoft.com/en-us/azure/role-based-access-control/role-assignments)。阅读 scope、role definition、principal 段落及其范围说明。
- 支持事实：Azure RBAC 角色分配把主体、角色定义和资源作用范围关联起来；文档建议将范围设为满足需求的最小资源范围。Microsoft 文档还说明主体类型和删除身份后的既有分配可能需要显式清理。
- 证据等级：供应商官方产品文档，支持 Azure 的实现行为。
- 边界：仅适用于 Azure RBAC 文档描述的实现；不能将 Azure 语义直接当作所有 IAM、企业 ACL 或多租户授权系统的标准定义。

## 未核验事项

- 未评估任何真实企业的 IAM 配置、日志质量、渗透测试、隔离实现、合同或供应商数据处理条款。
- 未对任何特定司法辖区的法规适用性作法律判断。
- 所有教学测试数值均为合成示例，不是来源报告的生产指标或安全结论。
