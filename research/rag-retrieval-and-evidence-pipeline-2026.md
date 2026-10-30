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

- **来源与版本**：[Elasticsearch Reciprocal Rank Fusion reference](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)，官方在线文档，访问日期 2026-10-30。
- **阅读位置**：公式、检索器组合示例、BM25 与向量检索示例。
- **来源支持的事实**：RRF 可把不同检索器的候选排名合成一个列表，不要求各自原始相关性分数处于相同尺度；Elasticsearch 页面给出其 API 参数和应用示例。
- **证据等级与边界**：source-backed（官方产品文档）；Elastic API 支持范围和默认参数依产品版本，不能当作 RRF 的通用实现规范或证明混合检索总会提升端到端答案。dense/sparse 的完整教学机制在第二册 RAG 章节。

## 检索内容与提示注入

- **来源与版本**：[OpenAI, Understanding prompt injections](https://openai.com/safety/prompt-injections/)，官方安全说明，访问日期 2026-10-30。
- **阅读位置**：prompt injection 定义、外部来源恶意指令示例、分层防护说明。
- **来源支持的事实**：第三方内容可能把恶意指令混入模型上下文，意图误导模型偏离用户任务；来源以此说明 prompt injection 风险，并描述多层防护。
- **证据等级与边界**：source-backed（官方安全材料）；该说明不提供适用于所有 RAG 堆栈的检测率，不表示模型提示或文档隔离能单独阻断攻击。
