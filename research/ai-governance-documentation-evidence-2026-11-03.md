# AI 治理文档与欧盟透明义务：来源证据

核对日期：2026-11-03。本文只记录联网核验的制度、指南和发布方文档事实；不记录书籍审阅进度、修改决定或法律意见。

## 欧盟通用目的 AI 模型义务

- 欧盟委员会的 [GPAI provider obligations FAQ](https://digital-strategy.ec.europa.eu/en/faqs/guidelines-obligations-general-purpose-ai-providers) 说明，AI Act 中通用目的 AI 模型供应商义务自 2025-08-02 起适用，覆盖向 AI Office 提供的技术文档、向下游系统提供者提供的信息、版权合规政策和训练内容公开摘要。具体义务取决于是否为法规意义上的 provider、模型是否进入欧盟市场、模型类别及例外条件；委员会 FAQ 是解释材料，不替代法规文本或个案判断。
- 同一委员会 FAQ 将适用时间与执法时间分开：2026-08-02 起委员会开始对 GPAI 供应商义务实施完整执法并可处以罚款；在 2025-08-02 前已投放市场的 GPAI 模型，其供应商最迟须在 2027-08-02 前履行相关义务。模型分类、供应商身份或系统性风险状态仍需按具体事实判断。
- 欧盟 [GPAI Code of Practice](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai) 是自愿的合规工具。Transparency 章节提供 Model Documentation Form；Safety and Security 章节针对系统性风险模型的附加义务。采用 Code 不会把其文档表格变成适用于所有组织的通用 Model Card 标准；底层法定义务仍由适用法规决定。委员会关于 GPAI 义务的[专题页](https://digital-strategy.ec.europa.eu/en/factpages/general-purpose-ai-obligations-under-ai-act)还说明满足条件的开源模型存在部分文档义务例外，但系统性风险模型不适用该例外，版权政策和训练摘要义务仍需分别核对。

## AI Act Article 50 透明义务

- 欧盟委员会于 2026-07-20 发布并于 2026-07-31 更新 [Article 50 透明义务指南页面](https://digital-strategy.ec.europa.eu/en/library/guidelines-transparency-obligations-providers-and-deployers-ai-systems)；指南用于解释提供者、部署者、适用内容、例外和合规示例。Article 50 义务自 2026-08-02 起适用。
- 委员会 [Article 50 FAQ](https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act) 区分角色：提供者需在适用条件下说明用户正在与 AI 交互，并为生成或操纵的合成内容提供机器可读标记和可检测性；部署者另需处理情绪识别/生物特征分类告知、深度伪造披露以及特定公共利益文本标注。公共利益文本的人工实质审查或编辑控制及相应编辑责任，影响法规规定的例外。
- 过渡期范围有限：在 2026-08-02 前已投放市场的 AI 系统，仅 Article 50(2) 的 AI 生成内容标记与检测义务可延至 2026-12-02；这不是整个 Article 50 的延期，也不是所有内容都需追溯标记。系统和内容的具体适用范围以指南及现行法规为准。
- 2026 年《AI 生成内容透明度 Code of Practice》是自愿工具，委员会与 AI Board 已评估其为 Article 50(2)、(4)、(5) 提供的一种适当合规路径；不签署者仍需通过其他适当方式证明合规。机器可读标记、面向用户的标签和内容真实性是不同问题，标记本身不证明内容真实、作者身份或传播链完整。
- 法规文本可从 [EUR-Lex 2026-07-27 consolidated text](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX%3A02024R1689-20260727) 核对；指南与 FAQ 用于核实委员会当前实施解释。本文不提供特定司法辖区的法律意见。

## 风险管理框架与机构政策

- NIST [AI RMF 页面](https://www.nist.gov/itl/ai-risk-management-framework) 当前说明 AI RMF 1.0 正在修订；NIST [Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) 于 2024-07-26 发布。NIST 将 RMF 定位为自愿风险管理框架，不是模型发布认证或适用所有司法辖区的法规。
- OpenAI 的 [Preparedness Framework v2](https://openai.com/index/updating-our-preparedness-framework/) 于 2025-04-15 发布；[Frontier Governance Framework](https://openai.com/index/openai-frontier-governance-framework/) 于 2026-05-28 发布。两者是该机构不同用途的公开治理材料。
- Anthropic 当前 [Responsible Scaling Policy 页面](https://www.anthropic.com/responsible-scaling-policy) 列出 Version 3.4（2026-07-08 生效）及 2026 年风险报告；Google DeepMind [Frontier Safety Framework 页面](https://deepmind.google/frontier-safety/)列出 Version 3.1（2026-04-17）。它们支持发布方公开了哪些流程、版本和承诺，不是独立评测、统一行业标准或对其他部署环境的保证。

## Claude System Card 的版本与评测证据

- Anthropic 的 [Claude Opus 5.5 System Card](https://www.anthropic.com/claude-opus-5-5-system-card) 将其 OSWorld 小节标为 2.0，并报告 108 个任务、1080p、最多 500 个动作、五次运行及超过 100K tokens 后的 server-side compaction；partial/strict 结果为 81.8%/48.7%。Anthropic 的[发布页](https://www.anthropic.com/claude-opus-5-5)则将 81.8% partial 标为 OSWorld 2.1。两个发布方页面的版本标签不一致；同一个分数不足以证明任务文件和协议相同，48.7% strict 应留在 System Card 自己的标签与设置下。
- [Claude Opus 5 System Card](https://www.anthropic.com/claude-opus-5-system-card) 的 2026-08-19 changelog 记录了 Cowork browser harness 不一致后重跑模型基线，并因 Cowork 不能关闭 thinking 而不再报告该条件。Anthropic 的 [Transparency Hub](https://www.anthropic.com/transparency) 对 Opus 5 的 Gray Swan IPI 摘要还给出 28 个场景、1,130 个高迁移攻击，以及一次/十五次尝试概率 0.2%/2.0%。这些是发布方在特定攻击集、harness 和配置下的自报结果，不是独立复现或实际部署风险率。
- 上述发布材料可用于研究版本、任务定义、harness、成功判据和缓解配置如何共同限定治理声明；不可据此推断未披露的内部机制或将结果迁移至其他产品配置。
