# Dataset Versioning 与 Governance：来源证据

检索与阅读日期：2026-11-03。本文记录数据集文档、机器可读元数据、provenance 工具和删除权文本的来源事实及其边界，不构成某项目的合规判断或法律意见。

## 数据集与模型文档框架

- **Datasheets for Datasets**：[arXiv:1803.09010 v8](https://arxiv.org/abs/1803.09010v8)，论文版本最后修订于 2021-12-01；阅读摘要和版本信息。作者提出为数据集记录动机、组成、采集过程和推荐用途等问题，以改善创建者与使用者之间的沟通。它是研究框架，不是法规或自动核验流程，也不能保证填写内容正确。
- **Model Cards for Model Reporting**：[arXiv:1810.03993 v2](https://arxiv.org/abs/1810.03993v2)，2019 年 1 月修订，发表于 FAT* 2019；阅读摘要和书目信息。论文建议随模型发布其适用场景、性能特征和评估程序，评测可按与应用有关的条件及群体拆分。它是模型报告框架，不是认证、数据集事实证明或通用合规结论。

## Croissant 数据集元数据

- **来源与版本**：[Croissant Format Specification 1.1](https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html)，规范页面标注版本 1.1、发布日期 2026-01-29；[MLCommons 发布说明](https://mlcommons.org/2026/02/croissant-1-1-standard/)日期为 2026-02-12。阅读规范的 introduction、provenance representation、external vocabularies 和 data-use-conditions 部分，以及发布说明的 1.1 新增能力段落。
- **来源支持的事实**：规范以机器可读元数据描述数据集及资源、记录结构和语义；可使用 PROV-O 的 `wasDerivedFrom`、`wasGeneratedBy`、`wasAttributedTo` 等关系表示数据来源、生成活动与责任主体，并可将 provenance 放在数据集、文件、RecordSet、Field 或数据值层。规范还展示用 `schema:usageInfo`、DUO 等词汇表示数据使用条件。1.1 发布说明称该版本扩展 provenance、外部词汇与使用条件表达。
- **证据边界**：规范能表达哪些字段，不证明字段内容真实、权利判断正确或策略在消费者系统中实际执行；“机器可读”不等于“机器已判定合法”。发布说明中的自动治理/完整血缘属于发布方对目标能力的描述，不替代具体系统验证。

## Provenance 与运行事件模型

- **W3C PROV**：[PROV Overview](https://www.w3.org/TR/prov-overview/) 是 2013-04-30 发布的 Working Group Note，概述 PROV 文档族。其摘要将 provenance 描述为参与生成某项数据或对象的实体、活动与人员信息，并说明 PROV 系列提供模型及序列化，以支持异构环境间交换；其中 PROV-O、PROV-DM 等文档的规范级别应分别查看。它提供语义互操作参考，不决定内容质量、授权或保留政策。
- **OpenLineage**：[官方文档](https://openlineage.io/docs/)在阅读时显示版本 1.55.0；阅读 About 与 Core Specification 的 Object Model。模型区分 Dataset、Job、Run，并支持运行时 RunEvent、设计时 JobEvent/DatasetEvent 及可扩展 facets。它用于跨系统 lineage metadata 采集与交换，不等同于不可变数据快照、字节完整性校验或访问控制。
- **DVC**：[官方文档入口](https://dvc.org/doc)当前重定向至 `doc.dvc.org`，导航提供 data/model versioning、data pipelines 和 experiment management 相关说明。该入口说明的是工具文档范围；本记录未据此作出任意部署具备完整可复现性的判断。

## Dataset Card 文档实现

- **Hugging Face Dataset Cards**：[官方文档](https://huggingface.co/docs/hub/datasets-cards)，访问日期 2026-11-03；阅读 What are Dataset Cards 与 metadata 部分。文档说明 dataset repository 的 `README.md` 会被 Hub 渲染为 dataset card；YAML 元数据可包含 license、language、size、tags 和 data-file configuration 等内容。
- **证据边界**：这是特定平台的发布和展示格式。卡片及元数据由发布者提供；页面渲染、可筛选标签或存在 license 字段，不足以证明标签内容正确、数据可用于任意用途或训练流水线执行了相应限制。

## GDPR Article 17 删除权文本

- **来源**：[EUR-Lex Regulation (EU) 2016/679 Article 17](https://eur-lex.europa.eu/eli/reg/2016/679/art_17/oj)；另核对 EUR-Lex 对法规正文的索引结果。阅读 Article 17(1)—(3) 关于权利适用条件、公开数据传播时的合理通知步骤，以及列明的例外项目。
- **来源支持的事实**：Article 17 将该权利命名为 right to erasure（right to be forgotten），规定在列举的若干条件满足时，数据主体有权请求删除个人数据；第 3 款规定在所列情形下该权利不适用。具体条件与例外必须回到法规原文判断。
- **证据边界**：条文不等于针对所有模型、所有数据请求的单一技术方案。由该文本推导“每个请求都必须执行同一种模型遗忘算法”没有来源支持；具体义务、身份、目的、例外和补救方法需按适用法律与事实由专业责任人判断。本记录不提供个案法律意见。

## 治理风险管理框架

- **NIST AI RMF**：[NIST 官方页面](https://www.nist.gov/itl/ai-risk-management-framework)，访问日期 2026-11-03；页面说明 AI RMF 1.0 于 2023-01-26 发布、供自愿使用，并于当前页面标注正在修订；页面还列出 2024-07-26 发布的 Generative AI Profile 和 2026-04-07 发布的关键基础设施 profile concept note。
- **证据边界**：NIST RMF 提供风险管理语言和建议，不是数据许可判断、系统认证或适用于所有司法辖区的法规。具体治理设计仍需和数据对象、用途及组织政策对应。
