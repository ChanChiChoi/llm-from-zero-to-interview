# 产品需求发现与评估来源记录（2026-11）

记录日期：2026-11-16。本文只记录检索到的来源事实及其适用边界；章节取舍和修改结果记录在 `progress/`。

## GOV.UK 服务设计手册：发现阶段用户研究

- 来源：[User research in discovery](https://www.gov.uk/service-manual/user-research/user-research-in-discovery)，页面标注发布于 2016-11-18；[Understand users and their needs](https://www.gov.uk/service-manual/service-standard/point-1-understand-user-needs)，页面标注最后更新于 2022-05-30。阅读日期：2026-11-16。
- 来源支持：发现阶段应弄清潜在用户是谁、他们要完成什么、现在怎样做、遇到什么问题，以及需要什么结果；研究要看完整任务情境和支持服务的工作人员，可结合观察、访谈、既有数据、分析和过去研究。服务标准建议围绕用户问题而不是预设方案开展研究，并用研究、原型和现有数据尽早检验假设。
- 证据等级：官方服务设计指南，属于方法建议，不是受控实验或特定行业的效果证明。
- 边界：这些方法不能单独证明市场规模、支付意愿、统计代表性或某种 AI 方案优于现有方案；样本与结论要按具体服务、用户群和研究问题限定。收集真实工作材料和数据前需处理参与者同意、隐私及访问权限。

## Google HEART 与 Goals–Signals–Metrics

- 来源：Rodden、Hutchinson、Fu，*Measuring the User Experience on a Large Scale: User-Centered Metrics for Web Applications*，CHI 2010 作者版本：[Google Research PDF](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36299.pdf)。阅读日期：2026-11-16。
- 来源支持：HEART 将用户体验指标分成 Happiness、Engagement、Adoption、Retention、Task Success 类别；Goals–Signals–Metrics 流程先写产品/功能目标，再识别用户行为或态度信号，最后定义跟踪指标。论文指出各类指标不必全部使用；行为量应按用户或任务归一化；大规模指标不应脱离可用性和现场研究，也不是早期形成性用户研究的替代品。
- 证据等级：同行评审会议论文，作者描述其在 Google 多个产品中的框架实践；不是大模型产品评估的直接实证，也不证明某个指标适用于所有产品。
- 边界：HEART 类别是分类框架，不是通用成功阈值或场景评分公式。指标是否有效取决于目标、定义、分母、数据采集和使用情境。

## OpenAI Evals 指南与产品状态

- 来源：[Working with evals](https://developers.openai.com/api/docs/guides/evals)。阅读日期：2026-11-16。
- 来源支持：指南以任务说明、测试数据和评判标准组织模型输出评估，包含运行、分析和迭代流程；例子使用人工标注参考答案对照模型输出。
- 当前产品状态：指南在阅读日说明 OpenAI Evals 平台已进入弃用流程，既有 evals 于 2026-10-31 转为只读，计划于 2026-11-30 关闭；并建议新用户考虑 Datasets。这个时间表只描述该供应商平台，不能外推到通用评估方法或其他评估产品。
- 证据等级：供应商官方开发者文档，可支持其所示 API/平台合同与示例，不是独立方法学研究。
- 边界：离线模型输出评估不能替代真实用户研究，也不能单独证明任务完成、用户采用、经济收益或线上风险可接受。

