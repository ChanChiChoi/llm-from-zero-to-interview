# 第 42 章 Feature Store、Embedding Store 与向量索引基础设施

上一章讲了评估平台。本章讲 Feature Store、Embedding Store 与向量索引基础设施。

这些系统在推荐、搜索、RAG、Agent、风控和个性化模型中都很常见。它们的共同点是：把模型需要的特征、向量和索引从临时计算变成可管理、可复用、可服务化的基础设施。

先记住一句话：

> Feature Store 管结构化特征，Embedding Store 管向量资产，向量索引负责高效相似度检索；三者共同支撑在线智能系统的数据访问层。

## 42.1 为什么需要这些基础设施

在小实验中，特征和 embedding 可能直接存在一个文件里：

```text
features.csv
embeddings.npy
```

但生产系统需要面对：

1. 特征来源多。
2. 特征更新频率不同。
3. 训练和线上特征要一致。
4. embedding 规模巨大。
5. 向量检索要低延迟。
6. 索引需要定期更新。
7. 多租户和权限隔离。
8. 特征和向量版本要可追溯。

没有基础设施，系统会变成各业务自己造一套特征和向量管道，难以治理。

## 42.2 Feature Store 是什么

Feature Store 是特征管理系统。

它负责：

1. 定义特征。
2. 计算特征。
3. 存储特征。
4. 版本化特征。
5. 在线服务特征。
6. 离线训练读取特征。
7. 保证训练和线上一致。
8. 记录特征血缘。

Feature Store 最初常见于推荐、排序、风控等机器学习系统。

在大模型时代，它仍然重要，尤其在个性化、用户画像、工具路由、Agent 记忆和检索增强中。

## 42.3 什么是特征

特征是模型或业务逻辑使用的结构化信号。

例如：

1. 用户最近 7 天活跃次数。
2. 商品点击率。
3. 文档热度。
4. 用户等级。
5. 历史购买次数。
6. 风险评分。
7. 会话长度。
8. 工具调用成功率。
9. 模型响应满意度。

特征可以用于：

1. 训练传统 ML 模型。
2. 训练排序模型。
3. 做模型路由。
4. 做个性化 prompt。
5. 做风控和安全策略。
6. 做实验分层。

## 42.4 Offline Store 和 Online Store

Feature Store 通常分为两层：

1. Offline Store。
2. Online Store。

Offline Store 用于训练和分析，通常存放历史特征，支持大规模扫描。

Online Store 用于线上实时服务，要求低延迟读取。

例如：

```text
Offline Store: Hive / Iceberg / Parquet / Data Lake
Online Store: Redis / RocksDB / Cassandra / DynamoDB / HBase
```

训练读 Offline Store，推理读 Online Store。

关键是两者特征定义要一致。

## 42.5 Training-Serving Skew

Training-serving skew 指训练时用的特征和线上服务时用的特征不一致。

例如：

1. 训练用 7 天点击率，线上用 3 天点击率。
2. 训练特征有延迟，线上实时更新。
3. 训练时缺失值填 0，线上填 -1。
4. 特征计算代码不同。
5. 时间窗口切错导致未来信息泄漏。

Feature Store 的核心价值之一就是减少 training-serving skew。

它通过统一特征定义、版本和计算逻辑来保证一致性。

## 42.6 Point-in-Time Correctness

特征训练中必须注意 point-in-time correctness。

意思是：训练样本在某个时间点，只能使用那个时间点之前可获得的特征。

如果使用未来信息，就会数据泄漏。

例如预测用户明天是否购买，不能用明天之后的点击数据。

Feature Store 要支持按时间回放历史特征。

这对推荐、风控和在线学习非常关键。

## 42.7 Embedding Store 是什么

Embedding Store 是向量资产管理系统。

它存储对象的 embedding，例如：

1. 文档 embedding。
2. 段落 embedding。
3. 图片 embedding。
4. 商品 embedding。
5. 用户 embedding。
6. 工具 embedding。
7. API embedding。
8. 代码片段 embedding。

Embedding Store 关心的不只是向量本身，还包括：

1. embedding model version。
2. 生成时间。
3. 原始对象 ID。
4. 元数据。
5. 权限。
6. 版本。
7. 索引状态。

## 42.8 Embedding 为什么要版本化

Embedding 是由模型生成的。

如果 embedding model 变了，同一个文本的向量也会变。

因此必须记录：

1. embedding model name。
2. embedding model version。
3. pooling 方法。
4. normalize 策略。
5. dimension。
6. chunking 策略。
7. preprocessing 规则。

不同版本的 embedding 通常不能混在同一个索引里直接比较。

否则检索结果会不可控。

## 42.9 向量索引是什么

向量索引用于高效相似度检索。

给定 query embedding，找到最相似的 top-k 向量。

常见应用：

1. RAG 文档召回。
2. 相似问题匹配。
3. 图片检索。
4. 推荐召回。
5. 工具选择。
6. 代码搜索。
7. 语义缓存。

如果向量数量很少，可以暴力计算相似度。

如果有百万、千万、十亿级向量，就需要索引。

## 42.10 ANN 检索

ANN 是 Approximate Nearest Neighbor，近似最近邻检索。

它牺牲一点召回精度，换取更快查询。

常见索引类型：

1. HNSW。
2. IVF。
3. PQ。
4. ScaNN。
5. DiskANN。
6. Faiss 索引。

选型要考虑：

1. 数据规模。
2. 向量维度。
3. 查询延迟。
4. 召回率。
5. 内存成本。
6. 更新频率。
7. 是否支持过滤。

向量索引是性能、成本和召回率之间的 trade-off。

## 42.11 HNSW 的直觉

HNSW 可以直观理解为多层图索引。

高层图用于快速跳转到大致区域，底层图用于精细搜索。

优点：

1. 查询速度快。
2. 召回率高。
3. 适合内存索引。
4. 支持增量插入。

缺点：

1. 内存占用较高。
2. 构建成本不低。
3. 删除和压缩复杂。

HNSW 常用于中大规模低延迟向量检索。

## 42.12 IVF 和 PQ 的直觉

IVF 是先把向量聚类到多个桶里，查询时只搜索部分桶。

PQ 是 Product Quantization，用压缩表示降低存储和计算成本。

IVF/PQ 更适合大规模向量和成本敏感场景。

代价是：

1. 召回率可能下降。
2. 参数调优复杂。
3. 构建和重建需要成本。

大规模向量系统常常在内存、磁盘、召回率和延迟之间做组合优化。

## 42.13 向量索引的元数据过滤

实际检索往往不只是向量相似。

还要按元数据过滤：

1. 租户。
2. 权限。
3. 文档类型。
4. 时间范围。
5. 语言。
6. 产品线。
7. 安全等级。

例如企业 RAG 中，用户只能检索自己有权限的文档。

如果只做向量相似而不做权限过滤，会造成数据泄露。

## 42.14 Chunking 和 Embedding

文档通常不能整体做 embedding，需要切 chunk。

Chunking 策略会影响检索质量。

常见策略：

1. 固定长度切分。
2. 按段落切分。
3. 按标题结构切分。
4. sliding window。
5. 语义切分。
6. 代码按函数或类切分。

需要记录：

1. chunking 规则版本。
2. chunk size。
3. overlap。
4. 原文档 ID。
5. chunk offset。

否则检索结果无法复现和调优。

## 42.15 Re-indexing

当 embedding 模型、chunking 策略或数据更新后，需要重新建索引。

触发原因：

1. 文档新增。
2. 文档删除。
3. 文档更新。
4. embedding model 升级。
5. chunking 策略变化。
6. 索引参数变化。

Re-indexing 要考虑：

1. 增量更新还是全量重建。
2. 旧索引和新索引切换。
3. 查询一致性。
4. 回滚。
5. 构建成本。

大规模索引重建可能需要数小时甚至更久。

## 42.16 双索引切换

为了降低索引更新风险，可以使用双索引切换。

流程：

```text
active index: v1
build shadow index: v2
validate v2
canary traffic to v2
switch active to v2
keep v1 for rollback
```

这样可以避免直接在生产索引上修改导致不可控风险。

向量索引也需要灰度和回滚。

## 42.17 Embedding Pipeline

一个 embedding 生产链路：

```text
Raw Object
  -> Preprocess
  -> Chunking
  -> Embedding Model
  -> Embedding Store
  -> Index Builder
  -> Vector Index
  -> Retrieval Service
```

每一步都要记录版本和状态。

如果检索效果变差，要能判断是文档变了、chunking 变了、embedding 模型变了，还是索引参数变了。

## 42.18 Retrieval Service

Retrieval Service 对外提供检索接口。

请求可能包括：

1. query text。
2. query embedding。
3. top-k。
4. filters。
5. rerank 配置。
6. tenant。
7. permission context。

返回包括：

1. object ID。
2. chunk ID。
3. score。
4. metadata。
5. source pointer。
6. index version。

服务必须记录 trace，方便排查“为什么召回了这些文档”。

## 42.19 向量检索质量指标

常见指标：

1. recall@k。
2. precision@k。
3. MRR。
4. NDCG。
5. hit rate。
6. latency p95/p99。
7. index build time。
8. index size。
9. filter selectivity。
10. empty result rate。

RAG 场景还要看：

1. answer correctness。
2. citation correctness。
3. groundedness。
4. hallucination rate。

检索质量不能只看向量相似分数。

## 42.20 Feature / Embedding 血缘

Feature 和 embedding 都需要血缘。

Feature 血缘：

```text
raw event -> feature transform -> feature version -> model training / online serving
```

Embedding 血缘：

```text
document -> chunk -> embedding model version -> embedding version -> index version -> retrieval result
```

血缘可以帮助定位线上效果变化。

例如 RAG 变差，可能是索引版本变化，而不是大模型退化。

## 42.21 多租户和权限

向量和特征基础设施必须支持权限隔离。

风险包括：

1. A 租户检索到 B 租户文档。
2. 用户检索到无权限知识库。
3. embedding 泄露敏感语义。
4. 特征被越权读取。
5. 索引构建时混入私有数据。

治理方式：

1. tenant ID 进入 metadata。
2. 检索时强制权限过滤。
3. index 按租户隔离或逻辑隔离。
4. 敏感数据单独索引。
5. 访问日志和审计。

不要把向量看成“不可还原所以不敏感”。embedding 仍然可能泄露信息。

## 42.22 Feature Store、Embedding Store 和 RAG

RAG 系统通常依赖：

1. 文档存储。
2. chunk 存储。
3. embedding store。
4. vector index。
5. metadata filter。
6. reranker。
7. retrieval trace。

Feature Store 也可能参与 RAG：

1. 用户画像特征影响检索策略。
2. 文档热度特征影响排序。
3. 权限特征影响过滤。
4. 历史反馈特征影响 rerank。

RAG 不只是一个向量数据库，而是一条完整检索和治理链路。

## 42.23 常见系统架构

一个基础设施架构：

```text
Data Sources
  -> Feature Pipeline / Embedding Pipeline
  -> Offline Store / Embedding Store
  -> Online Store / Vector Index
  -> Feature Serving / Retrieval Service
  -> Model Serving / RAG / Agent
  -> Monitoring / Lineage / Access Control
```

关键是把 batch 构建、在线服务、版本管理、权限和监控打通。

## 42.24 常见误区

误区一：向量数据库等于 RAG。

向量数据库只是检索组件，RAG 还需要文档处理、chunking、embedding、权限、rerank、prompt 组装和 trace。

误区二：embedding 不需要版本化。

embedding 强依赖模型版本、预处理和 chunking，不版本化会导致检索不可复现。

误区三：训练和线上特征分别算就行。

这会带来 training-serving skew，应该统一特征定义和版本。

误区四：向量相似度高就一定相关。

相似度只是召回信号，还需要 rerank、metadata filter 和任务评估。

误区五：embedding 不敏感。

embedding 可能泄露语义和私有信息，必须做权限和审计。

## 42.25 面试常见追问

问题一：Feature Store 解决什么问题？

可以回答：它统一特征定义、计算、存储、版本和在线服务，减少训练和线上特征不一致，支持特征复用、血缘和权限治理。

问题二：Embedding Store 和向量索引有什么区别？

可以回答：Embedding Store 管向量资产及其元数据、版本、来源和权限；向量索引负责高效相似度检索。前者偏资产管理，后者偏检索性能。

问题三：为什么 embedding 要版本化？

可以回答：embedding 依赖 embedding model、预处理、chunking、pooling 和 normalize 策略。版本变了，向量空间也可能变，不能混用。

问题四：如何保证企业 RAG 不越权？

可以回答：文档、chunk、embedding 和索引都记录租户和权限元数据，检索时强制权限过滤，敏感数据可独立索引，并记录 retrieval trace 和审计日志。

## 42.26 小练习

1. Feature Store 的 offline store 和 online store 有什么区别？
2. 什么是 training-serving skew？
3. Point-in-time correctness 为什么重要？
4. Embedding Store 应记录哪些元数据？
5. HNSW、IVF、PQ 分别适合什么场景？
6. Chunking 策略如何影响 RAG 质量？
7. 为什么向量索引更新需要灰度和回滚？
8. 如何设计一个支持权限过滤的企业向量检索服务？

## 42.27 本章小结

本章讲了 Feature Store、Embedding Store 与向量索引基础设施。

你需要记住：

1. Feature Store 管理结构化特征，核心价值是复用、版本、训练线上一致和血缘。
2. Embedding Store 管理向量资产，必须记录 embedding model、chunking、预处理、版本和权限。
3. 向量索引负责高效 ANN 检索，需要在召回、延迟、成本和更新复杂度之间权衡。
4. RAG 不是向量数据库本身，而是文档处理、embedding、索引、过滤、rerank、prompt 和 trace 的完整链路。
5. 特征和向量基础设施必须支持多租户、权限控制、版本化、血缘和质量监控。

下一章我们会讲 RAG/Agent 平台中的知识库、工具和 trace 存储。
