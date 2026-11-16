# 产品单位经济、计费与支付意愿资料记录

检索与核验日期：2026-11-16。本文只记录来源事实及其边界；书籍章节的审阅、取舍和修改另记在 `progress/`。

## FinOps Foundation：Unit Economics

- 来源：[FinOps Framework — Unit Economics](https://www.finops.org/framework/capabilities/unit-economics/)，FinOps Foundation 在线框架，访问日期 2026-11-16；页面未标明本文所用内容的固定版本。
- 阅读位置：Definition、Maturity Assessment、Functional Activities、Inputs & Outputs。
- 来源支持的事实：框架把 unit economics 描述为连接技术支出与技术使用带来价值的方法；举例包括每个服务请求、工作负载、客户、解决案例、座席或 token 的成本。页面也要求记录指标定义、假设和成本纳入范围，并在较成熟阶段计入适用的完整成本、制定分摊规则、把基线/干预/结果证据纳入决策。
- 证据等级：`source-backed`。这是 FinOps 实践框架，不是会计准则或独立因果研究；它支持成本可见性和指标设计，不规定 ROI 公式或统一的通过阈值。
- 适用边界：组织可按产品目标选不同分母；不能把资源效率指标（如每 token 成本）自动当作业务单位指标（如每个有效解决案例成本）。共享成本分摊需明确规则，分配结果不等于真实因果成本。

## OpenAI API：当前计费口径、批处理与提示缓存

- 来源：[API Pricing](https://developers.openai.com/api/docs/pricing)、[Batch API](https://developers.openai.com/api/docs/guides/batch)、[Cost optimization](https://developers.openai.com/api/docs/guides/cost-optimization)、[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)，OpenAI 官方开发者文档，均于 2026-11-16 访问。页面可变，未固定长期版本。
- 阅读位置：Pricing 的模型/计费类型、工具与 batch 表；Batch API 的 Overview；Cost optimization 的 Batch/Flex 段落；Prompt caching 的缓存原理、pricing 与 model differences。
- 来源支持的事实：计费可区分模型、输入/输出 token、缓存输入、批处理和内置工具等项目；当前价目表应按具体模型和计费类型查看。Batch API 是异步工作流，官方文档当前写明它相对同步 API 有更高成本效率，并列出 24 小时完成窗口。提示缓存复用未改变的前缀计算状态，是否命中受模型、前缀和路由条件影响，不保证每次命中；缓存价格随模型和缓存读写设置变化，部分保留选项存在写入费用。
- 证据等级：`verified`（已阅读上述官方页面）。
- 适用边界：供应商文档说明该供应商当前公开的计费/接口合同，不代表其他供应商，也不保证未来版本；任何预算仍需以决策时的正式价目表、组织账单、请求 usage 和合同为准。批处理不适合必须同步返回的任务；把单价优惠写入预测前还须核验质量、延迟、数据可用性和运维成本。

## NIST：生成式 AI 风险语境

- 来源：[Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile, NIST AI 600-1](https://nvlpubs.nist.gov/nistpubs/AI/NIST.AI.600-1.pdf)，NIST，访问日期 2026-11-16。
- 阅读位置：§2 风险概览（PDF 第 8 页附近，按 PDF 版式）；NIST 将风险说明为事件可能性与后果程度的组合，并指出具体情境中的风险证据可能不同、部分风险具有不确定性。
- 证据等级：`source-backed`。
- 适用边界：该表述不是财务预期损失公式的强制标准，也不提供具体业务的概率和损失参数；教学中使用“概率 × 单次损失”时必须标为特定场景下的估算假设，证据不足则保留未知，不能把框架引用当成风险下降证明。

## 支付意愿测量：Wertenbroch 与 Skiera（2002）

- 来源：Klaus Wertenbroch & Bernd Skiera, “Measuring Consumers’ Willingness to Pay at the Point of Purchase,” *Journal of Marketing Research*, 39(2), 228–241 (2002)，作者可访问稿：[Wharton PDF](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/05/Wertenbroch-and-Skiera-2002.pdf)，DOI：[10.1509/jmkr.39.2.228.19086](https://doi.org/10.1509/jmkr.39.2.228.19086)。
- 阅读位置：摘要、方法对比、三项研究与局限讨论。
- 来源支持的事实：文章指出 WTP 测量方法在激励受访者诚实揭示支付意愿及模拟购买场景方面不同；作者报告的三项研究比较了带激励约束的 BDM 方法和假设性/陈述性方法，样本涉及低价杂货与耐用品，BDM 得到的 WTP 估计较低。
- 证据等级：`source-backed`（同行评审的原始研究）。
- 适用边界：该结果展示了测量方法会影响 WTP 估计，不证明 BDM 对所有产品都更合适，也不能把低价消费品研究直接外推到企业软件、AI 服务或真实采购。客户表态、试用、实际报价接受、续约和使用数据的证据强弱仍取决于具体决策设计和样本。
