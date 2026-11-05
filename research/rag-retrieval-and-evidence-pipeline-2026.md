# RAG 检索与证据链来源记录

检索日期：2026-10-30。本文记录第 7 章候选材料中经来源核验的事实与边界，不记录章节审阅或修改过程。

## GraphRAG 索引方法

- **来源与版本**：[Microsoft GraphRAG Indexing Methods](https://github.com/microsoft/graphrag/blob/main/docs/index/methods.md)，`main` 分支页面，访问日期 2026-10-30。
- **阅读位置**：Indexing Methods、Standard GraphRAG、FastGraphRAG、Choosing a Method。
- **来源支持的事实**：官方文档将 GraphRAG 定位为研究 RAG 索引方法的平台。标准方案使用语言模型抽取实体与关系、生成实体/关系描述和社区报告；FastGraphRAG 以 NLP 方法替代部分模型推理，索引成本更低，但图更嘈杂。文档估算标准图抽取约占其索引成本的 75%。
- **证据等级与边界**：source-backed（项目官方文档）；`main` 是动态页面，成本比例和方法选项只按访问时文档记录，不外推到其他语料、模型价格或实现版本。文档没有提供跨产品的受控成本比较；“额外生成阶段需要保真检查”是本书对该流水线的工程推论。

## Sentence Transformers 与模型快照

- **来源与版本**：[Sentence Transformers Usage](https://www.sbert.net/docs/sentence_transformer/usage/usage.html)、[`SentenceTransformer` API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)、[`CrossEncoder` API](https://www.sbert.net/docs/package_reference/cross_encoder/model.html)，官方文档访问日期 2026-10-30；[Sentence Transformers 6.2.0 PyPI release](https://pypi.org/project/sentence-transformers/)，版本 6.2.0 于 2026-10-21 发布。
- **阅读位置**：双编码器 usage、`encode_query` / `encode_document`、构造器 `revision` 参数、CrossEncoder 输入/输出与 `predict` activation 行为。
- **来源支持的事实**：Sentence Transformers 文档建议在 query/document 区分明确的检索任务中使用 `encode_query()` 和 `encode_document()`；没有专用 prompt/task 配置的模型中，它们可与通用 encode 路径相同。模型加载 API 可用 Hub commit id 固定 revision。CrossEncoder 联合处理一对文本，不能为候选独立预计算句向量；输出是分数或类别概率，activation 和尺度受模型配置及 API 参数影响。
- **证据等级与边界**：source-backed（官方 API 文档和 PyPI release 页面）；文档接口不证明当前章节示例在某个 Python、PyTorch、Transformers、NumPy、FAISS 组合中成功联调，本地未安装或执行模型示例。
- **固定模型快照**：`BAAI/bge-small-zh-v1.5` 使用 commit `7999e1d3359715c523056ef9478215996d62a620`；`BAAI/bge-reranker-base` 使用 commit `2cfc18c9415c912f9d8155881c133215df768a70`。提交页：[embedding revision](https://huggingface.co/BAAI/bge-small-zh-v1.5/commit/7999e1d3359715c523056ef9478215996d62a620)、[reranker revision](https://huggingface.co/BAAI/bge-reranker-base/commit/2cfc18c9415c912f9d8155881c133215df768a70)。模型内容固定不代表依赖栈、设备、输出值或排序质量固定；仍需在目标环境与评测集验证。

## FAISS 精确内积检索

- **来源与版本**：[FAISS MetricType and distances](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)、[FAISS indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)，官方项目 Wiki，访问日期 2026-10-30；[faiss-cpu 1.15.1 安装入口](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)，官方仓库当前安装说明。
- **阅读位置**：`METRIC_INNER_PRODUCT`、余弦相似度归一化要求、`IndexFlatIP` 精确内积索引表。
- **来源支持的事实**：内积本身不等于余弦相似度；要用内积排序实现 cosine，query 和库向量都须先归一化。`IndexFlatIP` 是 exhaustive / exact inner-product search；对大规模库应另按精确性、内存和吞吐选择索引。
- **证据等级与边界**：source-backed（FAISS 官方项目资料）；当前章节 API 参考 faiss-cpu 1.15.1，但没有本地运行/性能数据，不能据此给出目标平台安装保证或成本结论。

## 混合检索的倒数排名融合

- **来源与版本**：Gordon V. Cormack、Charles L. A. Clarke、Stefan Büttcher，*Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*，SIGIR 2009，DOI：[10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114)；[Elasticsearch Reciprocal Rank Fusion reference](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)，官方在线文档访问日期 2026-11-05。
- **阅读位置**：原论文的 RRF 分数聚合方法；Elastic 文档的公式、窗口和 rank constant 参数说明。
- **来源支持的事实**：RRF 把多个检索器中的名次按倒数项相加，降低候选在任一列表中的名次越靠后时的贡献；该分数融合使用排名而非原始相关性分数，因此不要求 BM25 与向量相似度处于同一量纲。
- **证据等级与边界**：source-backed（原始论文与官方实现文档）。论文标题中的基准比较只针对其数据、任务和对照方法；不能据此保证目标 RAG 的检索或最终答案质量都会提升。产品参数及实现以相应版本文档为准。

## 检索内容与提示注入

- **来源与版本**：[OpenAI, Understanding prompt injections](https://openai.com/safety/prompt-injections/)，官方安全说明，访问日期 2026-10-30。
- **阅读位置**：prompt injection 定义、外部来源恶意指令示例、分层防护说明。
- **来源支持的事实**：第三方内容可能把恶意指令混入模型上下文，意图误导模型偏离用户任务；来源以此说明 prompt injection 风险，并描述多层防护。
- **证据等级与边界**：source-backed（官方安全材料）；该说明不提供适用于所有 RAG 堆栈的检测率，不表示模型提示或文档隔离能单独阻断攻击。

## 引用证据强度与声明校准（FORCEBENCH）

- **来源与版本**：Pin Qian 等，*Relevant Is Not Warranted: Evidence-Force Calibration for Cited RAG*，arXiv:2605.28044 v1，提交日期 2026-05-27：[论文页面](https://arxiv.org/abs/2605.28044)。
- **阅读位置**：摘要中的任务构造、force axes、固定 198 对 locality-filtered 样本和 judge 对比结果。
- **来源支持的事实**：FORCEBENCH 固定被引用段落，并为同一段落配对一个证据强度匹配的 claim 与一个局部增强语气/强度的 claim；测试维度包括 relation、modality、scope、temporal validity 和 numeric specificity。论文报告的四个模型 judge 中，通用支持提示仍有较高 monotonicity violation；显式 warrant-strength 提示降低但没有消除该错误。
- **证据等级与边界**：source-backed（预印本摘要与论文）。198 对样本是诊断性压力测试，不估计真实系统中引用夸大声明的总体比例；结果绑定所测 judge、提示和文本任务，不能推成所有评估器的固定表现。

## RAG 安全控制条件（RAG-Safety-Bench）

- **来源与版本**：Adithiyan Rajan Indira Saravanan、Kathleen C. Fraser，*RAG-Safety-Bench: Reliable Evaluation of Retrieval-Augmented LLM Safety*，EMNLP 2026，[ACL Anthology](https://aclanthology.org/2026.emnlp-main.2246/)。
- **阅读位置**：ACL Anthology 论文摘要和元数据。
- **来源支持的事实**：benchmark 使用四个条件隔离 RAG 文档因素：非 RAG；包含有害请求答案的 oracle 文档；与请求相关但不含具体答案的文档；随机安全文档。论文报告五个开源 LLM 的结果。以固定文档条件比较，可以在该 benchmark 中减少 retriever quality 对模型安全效应判断的混淆。
- **证据等级与边界**：source-backed（ACL 正式会议论文元数据和摘要）；实验只覆盖该论文的五个模型、提示、文档条件和任务，不提供其他模型、私有企业语料或生产部署的安全保证，也不替代对实际 retriever 的测试。

## 检索增强间接提示注入（USENIX Security 2026）

- **来源与版本**：Hongyan Chang、Ergute Bao、Xinjian Luo、Ting Yu，*Overcoming the Retrieval Barrier: Indirect Prompt Injection in the Wild for LLM Systems*，35th USENIX Security Symposium 2026，[论文页与 PDF](https://www.usenix.org/conference/usenixsecurity26/presentation/chang-hongyan)。
- **阅读位置**：论文摘要、攻击构造、自然 query / 外部语料实验及讨论和限制部分。
- **来源支持的事实**：论文将恶意内容拆成用于提高检索命中概率的 trigger fragment 和编码后续目标的 attack fragment；报告了其 embedding retrieval 设置和 RAG/Agent 端到端攻击实验。该工作说明检索阶段需要作为独立安全边界测试，不能只在已构造的 prompt 上测试模型是否拒绝。
- **证据等级与边界**：source-backed（USENIX Security 2026 论文）。检索实验主要针对 embedding-based retrieval，没有完整评估 hybrid retrieval 或 reranker 组合；跨 embedding 架构迁移也不由实验保证。摘要中的攻击率和成本只适用于论文威胁模型、语料、query、embedding 服务与系统设置，不能当作任意 RAG 的风险率。

## GraphRAG 仓库当前维护状态

- **来源与版本**：[Microsoft GraphRAG 官方仓库 README](https://github.com/microsoft/graphrag)，`main` 分支，访问日期 2026-11-02。
- **阅读位置**：README 的项目定位、Repository Guidance、indexing 成本提示和版本迁移说明。
- **来源支持的事实**：仓库说明该项目是探索图结构增强问答的 research project，当前处于 largely in maintenance mode，不计划接受新功能 PR 或实现新功能，仍会按需要处理 bug 与依赖更新；README 另提醒 indexing 可能成本较高，建议先小规模试验，并指出示例代码不是正式支持的 Microsoft 产品。
- **证据等级与边界**：source-backed（项目官方仓库动态页面）；结论绑定 2026-11-02 访问时的 `main` README，不代表未来支持策略或每个版本的维护周期。论文关于 global sensemaking 的结果仍限于论文语料、任务和对照方法，仓库维护状态不改变论文实验结论。
