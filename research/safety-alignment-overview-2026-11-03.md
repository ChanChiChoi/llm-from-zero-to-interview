# Safety 与 Alignment 总览：来源证据

检索与阅读日期：2026-11-03。本文只保存第八册第一章定向扫描中新核验的公开事实、版本和证据范围；章节处理决定与修改记录见 `progress/book-08-ai-safety-alignment/chapter-01-audit.md`。供应商政策、风险分类和治理框架说明发布者的意图或流程，不等于独立评估或系统安全认证。

## Model Spec 的公开行为原则

- **来源**：[OpenAI Model Spec (2026-10-29)](https://model-spec.openai.com/2026-10-29.html) 与 [openai/model_spec CHANGELOG](https://github.com/openai/model_spec/blob/main/CHANGELOG.md)。
- **版本与阅读位置**：公开页面标注 October 29, 2026；阅读 Overview、五个组织价值、Chain of Command、行为规范和更新记录。
- **支持的事实**：当前公开版本按 faithfulness to human intent、minimize harm、honesty and transparency、facilitate human agency、act with care 五项价值组织原则；加入了支持用户理解/选择/行动、谨慎使用上下文与记忆、尊重隐私和用户控制等内容，并把 side effects 与授权/取舍纳入行为原则。OpenAI 页面明确称 Model Spec 描述其预期行为，公开版可能不包含所有细节，生产模型尚未完全反映该规范。
- **证据等级与边界**：source-backed（发布方规范与变更记录）。可支持“公开行为规范如何表述”和对应版本的原则，不能作为某个模型已满足规范的实验结果，也不能代替系统执行器与权限验证。

## NIST AI RMF 与关键基础设施 Profile

- **来源**：[NIST AI Risk Management Framework 页面](https://www.nist.gov/itl/ai-risk-management-framework)；[Trustworthy AI in Critical Infrastructure Profile 概念说明](https://www.nist.gov/programs-projects/concept-note-ai-rmf-profile-trustworthy-ai-critical-infrastructure)及[概念说明 PDF](https://www.nist.gov/system/files/documents/2026/04/08/Draft%20Concept%20Note_%20Development%20of%20the%20NIST%20AI%20RMF%20Trustworthy%20Use%20of%20AI%20in%20Critical%20Infrastructure%20Profile.pdf)。
- **版本与阅读位置**：访问日期 2026-11-03；AI RMF 页面 Overview；概念说明的 Description、发布日期和项目状态。概念说明日期为 2026-04-07，项目页更新日期为 2026-07-17，状态为 Ongoing。
- **支持的事实**：NIST 页面称 AI RMF 1.0 正在修订，且将其描述为自愿使用；NIST 于 2026-04-07 发布 Trustworthy AI in Critical Infrastructure Profile 概念说明，以发展关键基础设施场景中的风险管理实践。
- **证据等级与边界**：source-backed（官方框架与项目状态）。概念说明是正在开展的 Profile 的开发入口，不是已完成标准、法规清单或系统认证。

## OWASP Agent Control Standard

- **来源**：[OWASP GenAI Security Project 的 ACS 页面](https://genai.owasp.org/resource/agent-control-standard-acs/)；[项目仓库](https://github.com/GenAI-Security-Project/agent-control-standard)。
- **版本与阅读位置**：页面标注 2026-09-01；读取 About、仓库 README 的协议定位与治理边界。
- **支持的事实**：OWASP 页面将 ACS 描述为通过 Agent 平台 middleware hooks 执行声明式策略的可移植运行时控制接口。仓库称 ACS 是 wire specification，可由独立 Guardian 经认证通道在动作执行前检查并允许、拒绝或修改动作，留存可回放审计轨迹；该仓库同时提供参考实现。
- **证据等级与边界**：source-backed（OWASP GenAI 项目页面和公开项目仓库）。它是项目发布的开放接口/规范，不能仅凭规范或参考实现推出任意 Agent framework 已支持、策略有效或控制已部署。

## International AI Safety Report 风险分类

- **来源**：[International AI Safety Report 2026](https://internationalaisafetyreport.org/publication/international-ai-safety-report-2026)。
- **阅读位置**：Introduction 的风险分类段及 Chapter 2 开头。
- **支持的事实**：报告以 misuse、malfunctions、systemic risks 组织通用 AI 风险，并指出类别并非穷尽且可能重叠；报告还明确提醒，它使用的 systemic risk 含义与 EU AI Act 对高影响 GPAI 风险的术语用法不同。
- **证据等级与边界**：source-backed（跨学科国际报告）。该 taxonomy 是组织分析风险的框架，不是某产品风险等级或普遍监管术语定义；引用 systemic risks 时应保留报告自己的语境。

## 组织级安全治理框架的版本

- **OpenAI Preparedness**：[Preparedness Framework v2](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) 首页标注 Version 2，Last updated 2025-04-15；[更新说明](https://openai.com/index/updating-our-preparedness-framework/)同日发布。OpenAI 的[Frontier Governance Framework](https://openai.com/index/openai-frontier-governance-framework/)发表于 2026-05-28，称 Preparedness Framework 仍是管理最严重先进 AI 风险的基础；新文件说明其公开治理实践如何映射到相关法律义务。它们是 OpenAI 的治理文件，不是跨组织的认证或普遍阈值。
- **Anthropic Responsible Scaling Policy**：[官方政策页面](https://www.anthropic.com/responsible-scaling-policy)，访问日期 2026-11-03。页面当前列出 Version 3.4（2026-07-08 生效）及历史版本，并标注 2026-08-14 最近更新。该版本列表支持版本和发布日期，不证明外部组织应采用同一阈值。
- **Google DeepMind Frontier Safety Framework**：[官方框架页面](https://deepmind.google/frontier-safety/)，访问日期 2026-11-03。当前列出的版本为 3.1（2026-04-17）；[更新说明](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/)指出该版加入部分领域的 Tracked Capability Levels。该页面支持发布方框架版本及其公开流程，不是独立认证。

## METR Frontier Risk Report（2026 年 2—3 月）

- **来源**：[METR 官方报告页](https://metr.org/blog/2026-05-19-frontier-risk-report/)。
- **版本与阅读位置**：页面日期 2026-05-19，评估窗口为 2026-02-16 至 2026-03-16；阅读 Executive summary、pilot process 和 scope 说明。
- **支持的事实**：这是覆盖 Anthropic、Google、Meta、OpenAI 的组织级试点，而非单一公开模型排行榜；参与者提供了内部模型访问（包括原始 CoT）和非公开能力/内部使用/监控信息。METR 为参与组织编写私有报告，参与者批准哪些非公开信息可出现在公开报告中；报告称其过程具有比以往评估更大的 editorial independence。报告按 means、motive、opportunity 讨论具体时点的内部 Agent 风险。
- **证据等级与边界**：source-backed（第三方机构的公开试点评估）。结论依赖有限参与组织、内部访问、参与者提供及批准披露的材料、特定窗口和威胁模型；不可外推为所有模型、所有公司的状况或公开部署事故率。须保留其第三方评估过程与披露边界。
