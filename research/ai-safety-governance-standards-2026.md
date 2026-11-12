# AI 安全、治理与供应链标准：来源证据

检索与阅读日期：2026-10-30。本文只记录官方框架、标准与技术规范支持的事实；清单、标准和治理框架不能替代目标系统的评估，也不自动构成合规或安全认证。

## AI 风险治理

- **来源**：[NIST AI Risk Management Framework 官方页面](https://www.nist.gov/itl/ai-risk-management-framework)、[AI RMF 1.0（NIST AI 100-1）](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10)、[Generative AI Profile（NIST AI 600-1）](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)。
- **版本与阅读范围**：AI RMF 1.0 发布于 2023-01-26；生成式 AI Profile 发布于 2024-07-26。阅读官方页面、AI RMF 核心功能介绍及 Profile 摘要和引言。
- **来源支持的事实**：AI RMF 1.0 面向自愿使用，以 Govern、Map、Measure、Manage 四项功能组织风险管理；NIST AI 600-1 是面向生成式 AI 的跨行业 companion profile，提出与四项功能相对齐的风险管理行动。NIST 官方页面在 2026-10-30 标明 AI RMF 1.0 正在修订。
- **证据边界**：框架提供风险管理目标与行动建议，不是法规清单、认证或系统性能保证。Profile 的风险和行动须按具体使用场景选择，不能据其发布推断所有组织已实施。

## OWASP LLM 与 Agentic 风险分类

- **来源**：[OWASP GenAI LLM Top 10 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/)及其[官方 PDF](https://genai.owasp.org/download/56857/?tmstv=1785822482)，[OWASP Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)。
- **版本与阅读范围**：LLM Top 10 资源页日期为 2026-08-03，配套 PDF 标为 Version 2026；PDF 的 revision history 日期字段仍显示占位文本。Agentic Top 10 页面标注 2025-12-09。阅读 LLM Top 10 的目录、What's New、LLM01:2026 及直接/间接 prompt injection 定义，并阅读 Agentic Top 10 的范围说明。
- **来源支持的事实**：LLM Top 10 2026 保留 LLM01:2026 Prompt Injection；PDF 将以违反模型安全协议为目标的 jailbreak 归类为 direct prompt injection 子类。注入来源包括直接用户输入、检索内容、工具输出、多模态内容和持久记忆。该版明确区分模型作为应用组件的风险与 Agent 自主行动风险，并指向单独的 Agentic Top 10。
- **证据边界**：两份 OWASP Top 10 均是社区风险分类与缓解入口，不是穷尽的威胁目录、风险发生率研究、控制有效性实验证明或产品认证。LLM 2026 PDF 的发布日期尚未填写，故日期采用资源页面标注；不得从目录的排序直接推出攻击概率。

## 网络安全事件响应

- **来源与版本**：[NIST SP 800-61 Rev. 3 官方页面](https://csrc.nist.gov/pubs/sp/800/61/r3/final)，最终版发布于 2025-04，取代 2012 年发布的 Rev. 2；访问日期：2026-11-02。
- **阅读位置**：官方摘要、发布日期和 Supersedes 字段。
- **来源支持的事实**：Rev. 3 将事件响应建议纳入 CSF 2.0 网络安全风险管理活动，目标包括准备响应、降低事件数量与影响，并提高检测、响应和恢复活动的效率与效果。
- **证据边界**：该出版物提供组织级网络安全风险管理建议，不是某个 LLM 服务的性能故障手册、法律合规判定或特定事件处置命令集；组织应结合自身风险与适用政策制定响应流程。

## 内容来源凭证

- **来源**：[C2PA Specifications 2.5 index](https://spec.c2pa.org/specifications/specifications/2.5/index.html)与[Content Credentials Technical Specification 2.5](https://spec.c2pa.org/specifications/specifications/2.5/specs/C2PA_Specification.html)。
- **版本与阅读范围**：2.5 index 将 2.5 列为当前规格，并同时保留 2.4、2.3 等历史版本；规范的 claim generator、validator、manifest state 和 validation process 章节，版本历史附录记录 2.5（2026-10）。
- **来源支持的事实**：规范定义由 assertions、claim、签名和内容绑定构成的 C2PA manifest；validator 检查 claim/signature、签名者凭证/信任状态、assertions、ingredients 与 asset binding，并返回验证状态。规范提供 provenance 和完整性信号。
- **证据等级与边界**：source-backed。将其解释为“不裁决媒体陈述的真实性”是依据规范验证对象和 provenance 范围作出的边界说明；签名完整性或 signer trust 不证明 assertion 中事实为真，也不证明媒体未被误导性描述。用户界面仍须避免把“验证通过”表现成“内容真实”。

## 安全软件开发与供应链

- **来源**：[NIST SP 800-218 SSDF 1.1](https://csrc.nist.gov/pubs/sp/800/218/final)、[NIST SP 800-218A](https://csrc.nist.gov/pubs/sp/800/218/a/final)、[SLSA specification 1.2](https://slsa.dev/spec/v1.2/)。
- **版本与阅读范围**：SP 800-218 1.1 最终版发布于 2022-02；SP 800-218A 最终版发布于 2024-07，阅读官方摘要和使用关系；SLSA 1.2 页面标为 Approved，阅读版本说明、Build Track、Source Track 和 attestation/provenance 简介。
- **来源支持的事实**：NIST SP 800-218A 是补充 SSDF 1.1 的 community profile，增加适用于生成式 AI 与双用途基础模型安全开发的 practices、tasks 和 recommendations，并明确建议与 SSDF 1.1 合用。SLSA 1.2 描述逐步提升供应链安全保证的 levels/tracks，并包括 provenance 等 attestation 格式；它覆盖的具体保证以对应 track 要求为界。
- **证据边界**：这些资料支持流程与供应链控制规划，不自动证明任意模型权重、依赖或运行系统安全。单个 artifact 的 provenance 不自动传递到其全部依赖、数据来源或部署后行为。


## OWASP 2026 Prompt Injection 与 Jailbreak 分类口径复核（2026-11-02）

- **来源与版本**：[OWASP GenAI LLM Top 10 2026 资源页](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/)、[Prompt Injection 术语表](https://genai.owasp.org/glossary/prompt-injection/) 与 [Jailbreak Prompt 术语表](https://genai.owasp.org/glossary/jailbreak-prompt/)，访问日期 2026-11-02；并核对 OWASP GenAI Security Project 的 `LLM01:2026 Prompt Injection` 条目。
- **阅读位置**：2026 LLM Top 10 资源页版本说明；术语表对 direct/indirect prompt injection 和 jailbreak 目标的区分；LLM01:2026 对 direct injection 与 jailbreak 的关系说明。
- **来源支持的事实**：2026 Top 10 将 jailbreak 作为 direct prompt injection 的一个目标子类；当前 OWASP 术语表分别说明 prompt injection 的输入/信任边界和 jailbreak 的安全绕过目标，并指出两类攻击可能组合。分类用于组织风险和测试，不代表所有安全资料都采用相同本体。
- **证据等级与边界**：source-backed（OWASP 社区分类与动态术语表）。不能据分类推出不同风险必然互斥或一个检测器适用于所有类别；实际评估应按输入来源/信任边界、攻击目标、工具动作和最终影响多轴标记。
