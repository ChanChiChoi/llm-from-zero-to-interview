# RAG 产品与评估来源记录

检索与阅读日期：2026-11-16。范围：RAG 原始研究思路、检索 API 的当前公开接口、RAG 指标及评估器边界、LLM 应用威胁，以及评估平台时效。此记录只保存联网发现的知识和来源证据，不记录书稿审阅或修改过程。

## 检索增强生成的研究边界

- 来源：[Lewis 等，Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)，arXiv:2005.11401；阅读摘要、引言、模型定义和实验说明。论文于 2020-05-22 首次提交，发表于 NeurIPS 2020。
- 支持事实：论文讨论用可微访问的显式非参数记忆补充预训练模型参数记忆；其具体实验使用预训练 seq2seq 生成模型、Wikipedia dense vector index 和预训练检索器，并比较 RAG-Sequence 与 RAG-Token 公式。实验评估知识密集 NLP 任务，不能直接推广为企业知识库准确率或生产收益。
- 证据等级：原始研究论文。
- 边界：论文中的模型、语料、数据集与当前企业系统条件不同；“减少幻觉”“可更新”只能作为设计动机，需以目标产品数据验证。

## 检索和向量存储的供应商接口示例

- 来源：[OpenAI Retrieval API 指南](https://developers.openai.com/api/docs/guides/retrieval)，阅读语义搜索、vector store、attribute filtering 和 hybrid search 说明。接口页为持续更新的官方文档，未固定客户端或服务端版本。
- 支持事实：页面介绍语义检索由向量存储支持；文件加入 vector store 后会自动分块、向量化和索引；文件属性可用于检索过滤；文档当前还说明混合检索可调节语义向量与稀疏关键词信号的权重。
- 证据等级：供应商官方 API 文档，支持该产品公开行为。
- 边界：不能据此推断其他 RAG 实现具有相同行为；接口提供过滤选项不等于完成企业 ACL 授权；文档默认分块/索引机制不证明适合所有文件和任务。动态价格、文件限制和默认参数不作为长期正文常数。

## RAGAS 研究与当前 Ragas 指标文档

- 来源：[Es 等，RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)，arXiv:2309.15217；阅读摘要、指标背景、WikiEval 和结论部分。
- 支持事实：论文提出 reference-free RAG pipeline 自动评估框架，区分检索上下文相关性、生成答案忠实度和答案相关性；论文使用 WikiEval 人类偏好判断研究指标关系，并指出上下文相关性是较难评估的维度。
- 证据等级：原始研究论文。
- 边界：论文基准和人类判断设置不等于企业任务；自动指标是评估代理，不能代替业务准确性或目标用户评价。

- 来源：[Ragas Context Precision 当前文档](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)及 [Faithfulness 当前文档](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)；阅读指标定义、示例和 legacy API 状态说明。页面标注的近期维护时间可变，未固定 Python package revision。
- 支持事实：文档中的 Context Precision 衡量相关 context chunk 在检索排序中的位置/靠前程度；Faithfulness 通过识别回答 claim 并检查其能否由 retrieved context 推出，计算支持断言比例。当前文档建议新项目使用 collections-based API，而非标为 legacy 的 API。
- 证据等级：项目官方文档。
- 边界：实现版本、judge model、prompt 和 reference 配置会影响结果；将工具指标改写成不排序的简单片段占比时，需明确它不是同名 Ragas Context Precision。上线判断应对评估器做人工校准。

## OWASP GenAI LLM Top 10 2026 与提示注入

- 来源：[OWASP GenAI LLM Top 10 2026 资源页](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/)、[2026 PDF](https://genai.owasp.org/download/56857/?tmstv=1785822482)和 [Prompt Injection 词条](https://genai.owasp.org/glossary/prompt-injection/)。阅读 2026 版目录、LLM01 Prompt Injection 的描述/输入面/防护边界和资源页版本说明。
- 支持事实：2026 版清单包含 Prompt Injection、Sensitive Information Disclosure、Vector and Embedding Weaknesses 等应用风险分类。Prompt Injection 词条与 2026 PDF 说明，指令可来自被检索文档等间接输入；模型处理输入时缺少可靠的指令/数据结构分界，风险后果取决于系统赋予模型的访问和动作能力。
- 证据等级：OWASP GenAI Security Project 社区安全指南。
- 边界：该 Top 10 是风险识别与威胁讨论资料，不是法律、产品安全认证或完整的系统威胁模型。PDF 封面/修订历史仍把 2026 release date 标作待设定，而资源页将其列为当前最新指南；引用时写“2026 版”，不对出版日作更强断言。

## OpenAI Evals 平台时效

- 来源：[OpenAI Working with evals](https://developers.openai.com/api/docs/guides/evals)和 [OpenAI API deprecations](https://developers.openai.com/api/docs/deprecations)，官方文档在 2026-11-16 检索。
- 支持事实：Evals 平台于 2026-10-31 对现有用户转为只读，官方列出的 API、dashboard 和 graders 计划于 2026-11-30 关闭；指南建议正在建设评估者关注当前替代迁移资料。
- 证据等级：供应商官方产品生命周期文档。
- 边界：这是平台时间快照，可能随官方更新而变化；通用的任务定义、样本、指标、回归与人工复核流程不依赖此单一平台。
