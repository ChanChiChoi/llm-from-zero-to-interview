# 第 42 章 Feature Store、Embedding Store 与向量索引基础设施

上一章讲了评估平台。本章讲 Feature Store、Embedding Store 与向量索引基础设施。

这些系统在推荐、搜索、RAG、Agent、风控和个性化模型中都很常见。它们的共同点是：把模型需要的特征、向量和索引从临时计算变成可管理、可复用、可服务化的基础设施。

先记住一句话：

> Feature Store 管结构化特征，Embedding Store 管向量资产，向量索引负责高效相似度检索；三者共同支撑在线智能系统的数据访问层。

## 42.0 本讲资料边界与第二轮精修口径

本讲按通用 AI Infra 数据访问层来写，不绑定某个 Feature Store、向量数据库、RAG 框架、搜索引擎、对象存储或内部平台实现。

资料校准时参考的公开口径包括：Feast 对 entity、feature view、offline store、online store、historical feature retrieval 和 point-in-time join 的抽象；Faiss 对相似度搜索、IVF、PQ、HNSW 等索引类型的公开说明；Milvus / Qdrant / Weaviate / Pinecone 等向量数据库文档对 ANN、metadata filter、payload、collection、多租户和索引参数的工程口径；以及前文数据版本、artifact、实验追踪和评估平台章节对版本、血缘、门禁和可观测性的要求。

本轮精修重点不是介绍某个产品 API，而是把 Feature Store、Embedding Store 和 Vector Index 抽象成可审计基础设施：

1. 用稳定数学符号描述特征定义、训练线上一致性、point-in-time correctness、embedding 版本、chunk 血缘、ANN 召回、过滤、索引切换、成本和最终门禁。
2. 增加一个 0 依赖 `MiniFeatureEmbeddingIndexAudit` demo，用 toy case 检查 16 个关键治理维度。
3. 同步第四册百科、题库、练习、术语表、项目任务和知识图谱，让本章和 RAG / Agent 平台章节自然衔接。

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

可以把本章讨论的基础设施抽象成一条数据访问链：

```math
A_i=(f_i,e_i,c_i,v_i,s_i,r_i,p_i,l_i,z_i)
```

其中 `f_i` 是 feature definition，`e_i` 是 embedding record，`c_i` 是 chunk / object metadata，`v_i` 是 vector index，`s_i` 是 serving request，`r_i` 是 retrieval result，`p_i` 是 permission context，`l_i` 是 lineage，`z_i` 是当前状态。

基础设施能力覆盖率可以写成：

```math
C_{\mathrm{infra}}=\frac{1}{N_{\mathrm{cap}}}\sum_{j=1}^{N_{\mathrm{cap}}}\mathbf{1}[a_j\ \mathrm{ready}]
```

如果只存了 `embeddings.npy`，但没有版本、权限、索引状态、trace 和回滚，`C_infra` 会很低。

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

一个特征定义可以写成：

```math
F_j=(n_j,v_j,k_j,s_j,t_j,w_j,\tau_j,o_j,p_j)
```

其中 `n_j` 是 feature name，`v_j` 是 definition version，`k_j` 是 entity key，`s_j` 是 source，`t_j` 是 transform version，`w_j` 是时间窗口，`\tau_j` 是 TTL 或 freshness SLA，`o_j` 是 owner，`p_j` 是 permission policy。

特征定义完整率可以写成：

```math
C_{\mathrm{feature}}=\frac{1}{N_{\mathrm{feature}}}\sum_{j=1}^{N_{\mathrm{feature}}}\mathbf{1}[F_j\ \mathrm{complete}]
```

面试里不要只说“有特征表”，要能说清每个特征的定义、版本、实体键、窗口、刷新频率和线上读取方式。

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

offline / online 一致性可以按同一批实体抽样检查：

```math
R_{\mathrm{skew}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\left[|x^{\mathrm{off}}_i-x^{\mathrm{on}}_i|>\epsilon_i\right]
```

其中 `x_off` 是离线训练读取到的特征值，`x_on` 是同一实体线上读取到的特征值，`\epsilon_i` 是允许的数值误差。`R_skew` 越高，说明训练和线上逻辑越不一致。

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

训练线上一致门禁可以写成：

```math
G_{\mathrm{skew}}=\mathbf{1}[R_{\mathrm{skew}}\le\rho \land C_{\mathrm{transform}}=1 \land C_{\mathrm{default}}=1]
```

其中 `C_transform` 表示离线和线上使用同一份或等价的 transform 版本，`C_default` 表示缺失值、截断、类型转换和默认值策略一致。

## 42.6 Point-in-Time Correctness

特征训练中必须注意 point-in-time correctness。

意思是：训练样本在某个时间点，只能使用那个时间点之前可获得的特征。

如果使用未来信息，就会数据泄漏。

例如预测用户明天是否购买，不能用明天之后的点击数据。

Feature Store 要支持按时间回放历史特征。

这对推荐、风控和在线学习非常关键。

Point-in-time correctness 的核心约束是：

```math
t_{\mathrm{feature\_available}}(i,j)\le t_{\mathrm{event}}(i)
```

也就是训练样本 `i` 在事件时间 `t_event` 上，只能使用当时已经可获得的第 `j` 个特征。

泄漏率可以写成：

```math
R_{\mathrm{pit\_leak}}=\frac{1}{N_{\mathrm{join}}}\sum_{i,j}\mathbf{1}[t_{\mathrm{feature\_available}}(i,j)>t_{\mathrm{event}}(i)]
```

`R_pit_leak` 必须接近 0，否则离线训练指标会被未来信息污染。

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

一个 embedding 记录可以写成：

```math
E_i=(o_i,m_i,v_i,d_i,c_i,g_i,u_i,h_i,z_i)
```

其中 `o_i` 是原始对象，`m_i` 是 embedding model，`v_i` 是 model version，`d_i` 是 dimension，`c_i` 是 chunking version，`g_i` 是生成时间，`u_i` 是权限和租户，`h_i` 是向量或源内容 checksum，`z_i` 是索引状态。

Embedding Store 的价值在于能回答：这个向量由哪个对象、哪个 chunk、哪个模型、哪个预处理和哪个权限上下文生成。

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

embedding 版本门禁可以写成：

```math
G_{\mathrm{emb}}=\mathbf{1}[C_{\mathrm{model}}=1 \land C_{\mathrm{dim}}=1 \land C_{\mathrm{chunk}}=1 \land C_{\mathrm{norm}}=1]
```

其中 `C_model`、`C_dim`、`C_chunk`、`C_norm` 分别表示 embedding 模型版本、向量维度、chunking 版本和 normalize 策略一致。只要其中一项变化，就要重新评估是否需要新索引或双索引切换。

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

暴力检索的计算量可以粗略写成：

```math
T_{\mathrm{brute}}\propto N d
```

其中 `N` 是向量数，`d` 是向量维度。`N` 很大时，即使用矩阵乘法优化，也会遇到延迟、内存带宽和成本压力。

常见余弦相似度写成：

```math
\mathrm{cos}(q,x)=\frac{q^\top x}{\lVert q\rVert_2\lVert x\rVert_2}
```

如果向量已经归一化，余弦相似度等价于点积排序。

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

ANN 质量通常用召回率对比精确检索：

```math
R@k=\frac{|S_{\mathrm{ann}}(k)\cap S_{\mathrm{exact}}(k)|}{|S_{\mathrm{exact}}(k)|}
```

其中 `S_ann(k)` 是 ANN 返回的 top-k 集合，`S_exact(k)` 是暴力精确检索 top-k 集合。ANN 不是“结果随便近似”，而是在明确召回阈值下换取延迟和成本收益。

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

HNSW 的内存可以粗略理解为向量本体加图边：

```math
M_{\mathrm{hnsw}}\approx Ndb + NM b_{\mathrm{id}}+M_{\mathrm{overhead}}
```

其中 `N` 是向量数，`d` 是维度，`b` 是每个向量元素字节数，`M` 是每个节点近邻连接数，`b_id` 是邻居 id 字节数。HNSW 召回高、延迟低，但内存通常比压缩索引更贵。

## 42.12 IVF 和 PQ 的直觉

IVF 是先把向量聚类到多个桶里，查询时只搜索部分桶。

PQ 是 Product Quantization，用压缩表示降低存储和计算成本。

IVF/PQ 更适合大规模向量和成本敏感场景。

代价是：

1. 召回率可能下降。
2. 参数调优复杂。
3. 构建和重建需要成本。

大规模向量系统常常在内存、磁盘、召回率和延迟之间做组合优化。

IVF 查询只扫描部分桶，扫描比例可以写成：

```math
R_{\mathrm{scan}}=\frac{n_{\mathrm{probe}}}{n_{\mathrm{list}}}
```

其中 `n_probe` 是查询时探测的桶数，`n_list` 是总桶数。`n_probe` 越大，召回通常越高，延迟也越高。

PQ 压缩后的存储可以粗略写成：

```math
M_{\mathrm{pq}}\approx \frac{Nm b_{\mathrm{code}}}{8}+M_{\mathrm{codebook}}
```

其中 `m` 是子向量段数，`b_code` 是每段编码 bit 数。PQ 适合降低存储和内存成本，但需要接受量化误差。

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

过滤选择率可以写成：

```math
R_{\mathrm{filter}}=\frac{N_{\mathrm{after}}}{N_{\mathrm{before}}}
```

其中 `N_before` 是向量相似召回候选数，`N_after` 是 metadata / permission filter 后的候选数。企业 RAG 中，过滤不是可选优化，而是安全边界。

权限泄露率可以写成：

```math
R_{\mathrm{leak}}=\frac{N_{\mathrm{unauth\_returned}}}{N_{\mathrm{returned}}}
```

这个指标必须为 0。

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

固定长度 sliding window 的 chunk 数可以粗略写成：

```math
N_{\mathrm{chunk}}=\left\lceil \frac{\max(0,L-s)}{s-o}\right\rceil+1
```

其中 `L` 是文档 token 长度，`s` 是 chunk size，`o` 是 overlap。增大 overlap 可能提升跨段召回，但也会增加 embedding 成本、索引规模和重复召回。

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

索引重建成本可以拆成：

```math
T_{\mathrm{reindex}}=T_{\mathrm{extract}}+T_{\mathrm{embed}}+T_{\mathrm{build}}+T_{\mathrm{validate}}+T_{\mathrm{switch}}
```

向量重建成本可以粗略写成：

```math
K_{\mathrm{embed}}=N_{\mathrm{chunk}}K_{\mathrm{per\_chunk}}
```

其中 `N_chunk` 是需要重新 embedding 的 chunk 数，`K_per_chunk` 是单 chunk embedding 成本。embedding model 或 chunking 变更通常比单纯新增文档更贵，因为可能触发全量重建。

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

双索引切换门禁可以写成：

```math
G_{\mathrm{switch}}=\mathbf{1}[C_{\mathrm{shadow}}=1 \land R@k\ge\tau_R \land L_{\mathrm{p95}}\le\tau_L \land C_{\mathrm{filter}}=1 \land C_{\mathrm{rollback}}=1]
```

其中 `C_shadow` 表示 shadow index 构建完成，`R@k` 是召回率，`L_p95` 是检索 P95 延迟，`C_filter` 表示权限过滤一致，`C_rollback` 表示旧索引仍可回滚。

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

Embedding pipeline 的版本指纹可以写成：

```math
h_{\mathrm{pipe}}=H(d_{\mathrm{src}},v_{\mathrm{prep}},v_{\mathrm{chunk}},v_{\mathrm{emb}},v_{\mathrm{index}})
```

其中 `H` 是稳定 hash。只要原始数据、预处理、chunking、embedding model 或索引参数变化，指纹就应该变化，避免把新旧产物混在一起。

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

一次检索请求可以写成：

```math
Q_i=(u_i,t_i,q_i,k_i,\phi_i,\rho_i,\eta_i,z_i)
```

其中 `u_i` 是用户或租户，`t_i` 是时间，`q_i` 是 query，`k_i` 是 top-k，`\phi_i` 是 metadata filter，`\rho_i` 是 rerank 配置，`\eta_i` 是权限上下文，`z_i` 是 trace id。

检索 trace 完整率可以写成：

```math
C_{\mathrm{trace}}=\frac{1}{N_{\mathrm{trace}}}\sum_{j=1}^{N_{\mathrm{trace}}}\mathbf{1}[\sigma_j\ \mathrm{captured}]
```

`sigma_j` 可以包括 query hash、embedding model、index version、filters、candidate ids、scores、rerank ids、permission decision、latency 和 source pointers。

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

常见排序指标可以写成：

```math
\mathrm{precision@k}=\frac{N_{\mathrm{relevant@k}}}{k}
```

```math
\mathrm{MRR}=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{\mathrm{rank}_i}
```

```math
\mathrm{NDCG@k}=\frac{\mathrm{DCG@k}}{\mathrm{IDCG@k}}
```

上线门禁通常要同时看召回、排序、权限、延迟、空结果、引用支持和最终回答质量。

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

Feature / embedding 血缘图可以写成：

```math
G_{\mathrm{fe}}=(V,E)
```

其中节点包括 raw event、feature transform、feature version、document、chunk、embedding model、embedding record、index version、retrieval request、retrieval result、RAG answer；边表示“生成、读取、索引、检索、引用、服务”。这张图能把线上 bad case 追到具体数据和索引版本。

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

多租户隔离门禁可以写成：

```math
G_{\mathrm{tenant}}=\mathbf{1}[C_{\mathrm{tenant}}=1 \land C_{\mathrm{acl}}=1 \land R_{\mathrm{leak}}=0 \land C_{\mathrm{audit}}=1]
```

其中 `C_tenant` 表示 tenant metadata 必填，`C_acl` 表示检索时强制 ACL 过滤，`R_leak` 是无权返回率，`C_audit` 表示访问日志和审计 trace 完整。

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

RAG 检索门禁可以写成：

```math
G_{\mathrm{rag\_retrieval}}=\mathbf{1}[R@k\ge\tau_R \land C_{\mathrm{citation}}\ge\tau_C \land R_{\mathrm{leak}}=0 \land L_{\mathrm{p95}}\le\tau_L]
```

这里 `C_citation` 是引用能否追到 source pointer、chunk id、doc version 和权限范围的覆盖率。RAG / Agent 平台如果丢了这些元数据，就很难解释答案来源。

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

## 42.24 特征向量基础设施审计指标和最小 demo

把本章内容合在一起，可以用 16 个维度审计 Feature Store、Embedding Store 和 Vector Index：

1. feature definition contract：特征名称、版本、实体键、来源、transform、窗口、TTL、owner 和权限是否完整。
2. offline online consistency：offline store 和 online store 是否使用同一特征定义、transform、默认值和类型策略。
3. point in time correctness：离线训练样本是否只读取当时已经可获得的历史特征。
4. embedding version contract：embedding model、版本、维度、pooling、normalize、预处理和 chunking 版本是否记录。
5. chunk embedding lineage：文档、chunk、offset、checksum、embedding record 和权限元数据是否可追踪。
6. vector index build readiness：索引版本、维度、metric、参数、构建状态、构建输入和 shadow index 是否完整。
7. ANN quality latency gate：ANN recall@k、P95/P99 延迟、空结果率和过滤选择率是否达标。
8. metadata permission filter：tenant、ACL、数据等级和敏感索引隔离是否在检索阶段强制执行。
9. shadow index switch readiness：双索引构建、验证、canary、切换和回滚是否完整。
10. retrieval trace completeness：query、embedding model、index version、filters、candidate ids、scores、rerank、权限决策和 source pointer 是否进入 trace。
11. feature embedding lineage graph：feature、document、chunk、embedding、index、retrieval、RAG answer 是否可双向追踪。
12. multi tenant isolation：租户元数据、配额、物理或逻辑隔离、跨租户命中阻断和审计是否完整。
13. quality monitoring metrics：recall、precision、latency、stale feature、skew、permission leak 和 empty result 是否监控。
14. cost capacity governance：向量内存、索引构建成本、embedding 成本、查询成本、预算和 owner 是否可治理。
15. RAG Agent integration：retrieval trace、rerank、citation、Agent memory scope 和 permission context 是否打通。
16. feature embedding index gate：最终门禁是否能证明覆盖率达标、无 P0 风险、可回滚、可观测、可审计。

下面的 demo 不依赖外部库。它用一个完整样本和 16 个单点坏样本说明：这类基础设施不能只看“能查向量”，而要同时证明特征、embedding、chunk、索引、过滤、trace、血缘、租户和 RAG / Agent 集成都可信。

```python
from math import sqrt


class MiniFeatureEmbeddingIndexAudit:
    REQUIRED_FEATURE_FIELDS = {
        "name",
        "version",
        "owner",
        "entity_key",
        "source",
        "transform_version",
        "dtype",
        "window",
        "ttl_minutes",
        "permission",
    }

    REQUIRED_EMBEDDING_MODEL_FIELDS = {
        "model_name",
        "model_version",
        "dimension",
        "pooling",
        "normalize",
        "preprocess_version",
        "chunking_version",
    }

    REQUIRED_TRACE_SIGNALS = {
        "query_hash",
        "embedding_model_version",
        "index_version",
        "filters",
        "candidate_ids",
        "scores",
        "rerank_ids",
        "permission_decision",
        "latency_ms",
        "source_pointers",
    }

    REQUIRED_QUALITY_METRICS = {
        "recall_at_10",
        "precision_at_5",
        "p95_ms",
        "empty_result_rate",
        "stale_feature_rate",
        "skew_rate",
        "permission_leak_rate",
    }

    def __init__(self, cases):
        self.cases = cases
        self.checks = [
            ("feature_definition_contract", self.feature_definition_contract),
            ("offline_online_consistency", self.offline_online_consistency),
            ("point_in_time_correctness", self.point_in_time_correctness),
            ("embedding_version_contract", self.embedding_version_contract),
            ("chunk_embedding_lineage", self.chunk_embedding_lineage),
            ("vector_index_build_readiness", self.vector_index_build_readiness),
            ("ann_quality_latency_gate", self.ann_quality_latency_gate),
            ("metadata_permission_filter", self.metadata_permission_filter),
            ("shadow_index_switch_readiness", self.shadow_index_switch_readiness),
            ("retrieval_trace_completeness", self.retrieval_trace_completeness),
            ("feature_embedding_lineage_graph", self.feature_embedding_lineage_graph),
            ("multi_tenant_isolation", self.multi_tenant_isolation),
            ("quality_monitoring_metrics", self.quality_monitoring_metrics),
            ("cost_capacity_governance", self.cost_capacity_governance),
            ("rag_agent_integration", self.rag_agent_integration),
            ("feature_embedding_index_gate", self.feature_embedding_index_gate),
        ]

    def feature_definition_contract(self, case):
        features = case["features"]
        return bool(features) and all(self.REQUIRED_FEATURE_FIELDS <= set(item) and item["ttl_minutes"] > 0 for item in features)

    def offline_online_consistency(self, case):
        consistency = case["offline_online"]
        offline = consistency["offline_values"]
        online = consistency["online_values"]
        if not consistency.get("same_transform_version") or not consistency.get("same_default_policy"):
            return False
        return all(abs(offline[name] - online.get(name, float("inf"))) <= 1e-9 for name in offline)

    def point_in_time_correctness(self, case):
        if not case["point_in_time"].get("point_in_time_join"):
            return False
        return all(item["feature_available_ts"] <= item["event_ts"] for item in case["point_in_time"]["training_rows"])

    def embedding_version_contract(self, case):
        model = case["embedding_model"]
        if not self.REQUIRED_EMBEDDING_MODEL_FIELDS <= set(model):
            return False
        dimension = model["dimension"]
        vectors = case["embedding_records"]
        if any(len(item["vector"]) != dimension for item in vectors):
            return False
        if model["normalize"]:
            return all(abs(vector_norm(item["vector"]) - 1.0) <= 1e-3 for item in vectors)
        return True

    def chunk_embedding_lineage(self, case):
        embedding_ids = {item["embedding_id"] for item in case["embedding_records"]}
        for chunk in case["chunks"]:
            required = {"doc_id", "chunk_id", "offset_start", "offset_end", "chunking_version", "embedding_id", "checksum", "tenant", "acl"}
            if not required <= set(chunk):
                return False
            if chunk["offset_start"] >= chunk["offset_end"] or chunk["embedding_id"] not in embedding_ids:
                return False
        return True

    def vector_index_build_readiness(self, case):
        index = case["index"]
        required = {"index_id", "version", "dimension", "metric", "index_type", "build_status", "index_params", "embedding_model_version", "chunking_version"}
        return required <= set(index) and index["build_status"] == "ready" and index["dimension"] == case["embedding_model"]["dimension"]

    def ann_quality_latency_gate(self, case):
        ann = case["ann_quality"]
        return (
            ann["recall_at_10"] >= 0.9
            and ann["p95_ms"] <= 50
            and ann["p99_ms"] <= 100
            and ann["empty_result_rate"] <= 0.05
        )

    def metadata_permission_filter(self, case):
        filt = case["permission_filter"]
        return (
            filt.get("tenant_filter")
            and filt.get("acl_filter")
            and filt.get("data_classification_filter")
            and filt.get("unauthorized_returned") == 0
            and filt.get("audit_log")
        )

    def shadow_index_switch_readiness(self, case):
        switch = case["shadow_switch"]
        return (
            switch.get("shadow_built")
            and switch.get("shadow_validated")
            and switch.get("canary_passed")
            and bool(switch.get("rollback_index"))
            and bool(switch.get("switch_plan"))
        )

    def retrieval_trace_completeness(self, case):
        return self.REQUIRED_TRACE_SIGNALS <= case["trace"].get("signals", set())

    def feature_embedding_lineage_graph(self, case):
        lineage = case["lineage"]
        required_nodes = {"feature", "document", "chunk", "embedding", "index", "retrieval", "rag_answer"}
        return required_nodes <= set(lineage["nodes"]) and len(lineage["edges"]) >= 9

    def multi_tenant_isolation(self, case):
        tenant = case["multi_tenant"]
        return (
            tenant.get("tenant_key_in_metadata")
            and tenant.get("per_tenant_quota")
            and tenant.get("sensitive_index_isolated")
            and tenant.get("no_cross_tenant_hits")
            and tenant.get("audit_ready")
        )

    def quality_monitoring_metrics(self, case):
        metrics = case["quality_monitoring"]["metrics"]
        return (
            self.REQUIRED_QUALITY_METRICS <= set(metrics)
            and metrics["recall_at_10"] >= 0.9
            and metrics["p95_ms"] <= 50
            and metrics["skew_rate"] <= 0.01
            and metrics["permission_leak_rate"] == 0
        )

    def cost_capacity_governance(self, case):
        cost = case["cost"]
        return (
            cost.get("owner")
            and cost["memory_gib"] <= cost["memory_budget_gib"]
            and cost["build_cost_usd"] <= cost["build_budget_usd"]
            and cost["query_cost_per_1k"] <= cost["query_budget_per_1k"]
        )

    def rag_agent_integration(self, case):
        integration = case["rag_agent"]
        required = {"retrieval_trace", "rerank_linked", "citation_source_pointers", "agent_memory_scope", "permission_context"}
        return required <= set(integration) and all(integration[name] for name in required)

    def feature_embedding_index_gate(self, case):
        gate = case["gate"]
        return gate.get("enabled") and gate.get("min_coverage", 0.0) >= 0.95 and gate.get("p0_risks", 1) == 0

    def metric_scores(self):
        total = len(self.cases)
        return {
            name: round(sum(1 for case in self.cases if check(case)) / total, 3)
            for name, check in self.checks
        }

    def failed_cases(self):
        failures = []
        for case in self.cases:
            failed = [name for name, check in self.checks if not check(case)]
            if failed:
                failures.append((case["case_id"], failed))
        return failures

    def examples(self):
        case = self.cases[0]
        offline = case["offline_online"]["offline_values"]
        online = case["offline_online"]["online_values"]
        skew_rate = sum(1 for name in offline if abs(offline[name] - online[name]) > 1e-9) / len(offline)
        pit_rows = case["point_in_time"]["training_rows"]
        pit_leak_rate = sum(1 for row in pit_rows if row["feature_available_ts"] > row["event_ts"]) / len(pit_rows)
        chunk_embedding_coverage = sum(1 for chunk in case["chunks"] if chunk.get("embedding_id")) / len(case["chunks"])
        brute_force = brute_force_search(
            query=case["query"]["vector"],
            records=case["embedding_records"],
            top_k=3,
            tenant=None,
            role=None,
        )
        filtered = brute_force_search(
            query=case["query"]["vector"],
            records=case["embedding_records"],
            top_k=3,
            tenant=case["query"]["tenant"],
            role=case["query"]["role"],
        )
        vector_bytes = case["index"]["vector_count"] * case["index"]["dimension"] * case["index"]["dtype_bytes"]
        vector_memory_gib = vector_bytes / (1024 ** 3)
        return {
            "feature_contract_coverage": round(sum(1 for item in case["features"] if self.REQUIRED_FEATURE_FIELDS <= set(item)) / len(case["features"]), 3),
            "offline_online_skew_rate": round(skew_rate, 3),
            "point_in_time_leak_rate": round(pit_leak_rate, 3),
            "embedding_dimension": case["embedding_model"]["dimension"],
            "embedding_norm_ok": self.embedding_version_contract(case),
            "chunk_embedding_coverage": round(chunk_embedding_coverage, 3),
            "bruteforce_top3": [item["chunk_id"] for item in brute_force],
            "permission_filtered_top3": [item["chunk_id"] for item in filtered],
            "cross_tenant_blocked": all(item["tenant"] == case["query"]["tenant"] for item in filtered),
            "ann_recall_at_10": case["ann_quality"]["recall_at_10"],
            "retrieval_p95_ms": case["ann_quality"]["p95_ms"],
            "vector_memory_gib": round(vector_memory_gib, 2),
            "shadow_index_ready": self.shadow_index_switch_readiness(case),
            "lineage_edge_count": len(case["lineage"]["edges"]),
            "estimated_build_cost_usd": case["cost"]["build_cost_usd"],
        }


def vector_norm(vector):
    return sqrt(sum(value * value for value in vector))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def brute_force_search(query, records, top_k, tenant=None, role=None):
    candidates = []
    for record in records:
        if tenant is not None and record["tenant"] != tenant:
            continue
        if role is not None and role not in record["acl"]:
            continue
        candidates.append((dot(query, record["vector"]), record))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in candidates[:top_k]]


def complete_case():
    return {
        "case_id": "complete_feature_embedding_index_ok",
        "features": [
            {
                "name": "user_activity_7d",
                "version": "v3",
                "owner": "growth",
                "entity_key": "user_id",
                "source": "events.daily_user_activity",
                "transform_version": "tx-activity-v3",
                "dtype": "float",
                "window": "7d",
                "ttl_minutes": 120,
                "permission": "tenant_private",
            },
            {
                "name": "doc_hotness",
                "version": "v2",
                "owner": "search",
                "entity_key": "doc_id",
                "source": "logs.doc_views",
                "transform_version": "tx-doc-hotness-v2",
                "dtype": "float",
                "window": "1d",
                "ttl_minutes": 60,
                "permission": "tenant_private",
            },
            {
                "name": "tool_success_rate",
                "version": "v5",
                "owner": "agent-platform",
                "entity_key": "tool_id",
                "source": "agent.tool_calls",
                "transform_version": "tx-tool-success-v5",
                "dtype": "float",
                "window": "30d",
                "ttl_minutes": 180,
                "permission": "platform_internal",
            },
        ],
        "offline_online": {
            "same_transform_version": True,
            "same_default_policy": True,
            "offline_values": {"user_activity_7d": 12.0, "doc_hotness": 0.82, "tool_success_rate": 0.91},
            "online_values": {"user_activity_7d": 12.0, "doc_hotness": 0.82, "tool_success_rate": 0.91},
        },
        "point_in_time": {
            "point_in_time_join": True,
            "training_rows": [
                {"entity": "user_1", "event_ts": 100, "feature_available_ts": 94},
                {"entity": "user_2", "event_ts": 130, "feature_available_ts": 125},
                {"entity": "doc_8", "event_ts": 180, "feature_available_ts": 170},
            ],
        },
        "embedding_model": {
            "model_name": "text-embedder",
            "model_version": "embed-v4",
            "dimension": 4,
            "pooling": "mean",
            "normalize": True,
            "preprocess_version": "prep-v2",
            "chunking_version": "chunk-v6",
        },
        "embedding_records": [
            {"embedding_id": "emb_policy", "chunk_id": "chunk_policy", "tenant": "acme", "acl": {"analyst"}, "vector": [1.0, 0.0, 0.0, 0.0]},
            {"embedding_id": "emb_pricing", "chunk_id": "chunk_pricing", "tenant": "acme", "acl": {"analyst"}, "vector": [0.8, 0.6, 0.0, 0.0]},
            {"embedding_id": "emb_beta", "chunk_id": "chunk_beta_private", "tenant": "beta", "acl": {"admin"}, "vector": [0.9487, 0.3162, 0.0, 0.0]},
            {"embedding_id": "emb_sensitive", "chunk_id": "chunk_sensitive", "tenant": "acme", "acl": {"security"}, "vector": [0.7, 0.0, 0.7141, 0.0]},
        ],
        "chunks": [
            {"doc_id": "doc_policy", "chunk_id": "chunk_policy", "offset_start": 0, "offset_end": 480, "chunking_version": "chunk-v6", "embedding_id": "emb_policy", "checksum": "sha256:p1", "tenant": "acme", "acl": {"analyst"}},
            {"doc_id": "doc_pricing", "chunk_id": "chunk_pricing", "offset_start": 480, "offset_end": 960, "chunking_version": "chunk-v6", "embedding_id": "emb_pricing", "checksum": "sha256:p2", "tenant": "acme", "acl": {"analyst"}},
            {"doc_id": "doc_beta", "chunk_id": "chunk_beta_private", "offset_start": 0, "offset_end": 420, "chunking_version": "chunk-v6", "embedding_id": "emb_beta", "checksum": "sha256:b1", "tenant": "beta", "acl": {"admin"}},
            {"doc_id": "doc_sensitive", "chunk_id": "chunk_sensitive", "offset_start": 0, "offset_end": 390, "chunking_version": "chunk-v6", "embedding_id": "emb_sensitive", "checksum": "sha256:s1", "tenant": "acme", "acl": {"security"}},
        ],
        "index": {
            "index_id": "kb-acme-shadow",
            "version": "index-v12",
            "dimension": 4,
            "metric": "cosine",
            "index_type": "hnsw",
            "build_status": "ready",
            "index_params": {"M": 32, "ef_search": 128},
            "embedding_model_version": "embed-v4",
            "chunking_version": "chunk-v6",
            "vector_count": 20_000_000,
            "dtype_bytes": 2,
        },
        "ann_quality": {"recall_at_10": 0.93, "p95_ms": 35, "p99_ms": 78, "empty_result_rate": 0.018},
        "permission_filter": {
            "tenant_filter": True,
            "acl_filter": True,
            "data_classification_filter": True,
            "unauthorized_returned": 0,
            "audit_log": True,
        },
        "shadow_switch": {
            "active_index": "index-v11",
            "shadow_index": "index-v12",
            "shadow_built": True,
            "shadow_validated": True,
            "canary_passed": True,
            "rollback_index": "index-v11",
            "switch_plan": "canary-then-promote",
        },
        "trace": {
            "signals": {
                "query_hash",
                "embedding_model_version",
                "index_version",
                "filters",
                "candidate_ids",
                "scores",
                "rerank_ids",
                "permission_decision",
                "latency_ms",
                "source_pointers",
            }
        },
        "lineage": {
            "nodes": {"feature", "document", "chunk", "embedding", "index", "retrieval", "rag_answer"},
            "edges": [
                ("raw_event", "feature"),
                ("document", "chunk"),
                ("chunk", "embedding"),
                ("embedding", "index"),
                ("index", "retrieval"),
                ("feature", "retrieval"),
                ("retrieval", "rag_answer"),
                ("permission", "retrieval"),
                ("rag_answer", "evaluation"),
            ],
        },
        "multi_tenant": {
            "tenant_key_in_metadata": True,
            "per_tenant_quota": True,
            "sensitive_index_isolated": True,
            "no_cross_tenant_hits": True,
            "audit_ready": True,
        },
        "quality_monitoring": {
            "metrics": {
                "recall_at_10": 0.93,
                "precision_at_5": 0.82,
                "p95_ms": 35,
                "empty_result_rate": 0.018,
                "stale_feature_rate": 0.004,
                "skew_rate": 0.0,
                "permission_leak_rate": 0,
            }
        },
        "cost": {
            "owner": "retrieval-platform",
            "memory_gib": 24.0,
            "memory_budget_gib": 64.0,
            "build_cost_usd": 780.0,
            "build_budget_usd": 1200.0,
            "query_cost_per_1k": 0.19,
            "query_budget_per_1k": 0.3,
        },
        "rag_agent": {
            "retrieval_trace": True,
            "rerank_linked": True,
            "citation_source_pointers": True,
            "agent_memory_scope": True,
            "permission_context": True,
        },
        "gate": {"enabled": True, "min_coverage": 0.96, "p0_risks": 0},
        "query": {"tenant": "acme", "role": "analyst", "vector": [1.0, 0.0, 0.0, 0.0]},
    }


def make_bad_case(case_id, mutate):
    case = complete_case()
    case["case_id"] = case_id
    mutate(case)
    return case


cases = [
    complete_case(),
    make_bad_case("feature_definition_missing_bad", lambda c: c["features"][0].pop("version")),
    make_bad_case("offline_online_skew_bad", lambda c: c["offline_online"]["online_values"].update({"doc_hotness": 0.61})),
    make_bad_case("point_in_time_leak_bad", lambda c: c["point_in_time"]["training_rows"][0].update({"feature_available_ts": 108})),
    make_bad_case("embedding_version_missing_bad", lambda c: c["embedding_model"].pop("model_version")),
    make_bad_case("chunk_lineage_missing_bad", lambda c: c["chunks"][0].pop("embedding_id")),
    make_bad_case("index_not_ready_bad", lambda c: c["index"].update({"build_status": "building"})),
    make_bad_case("ann_recall_low_bad", lambda c: c["ann_quality"].update({"recall_at_10": 0.82})),
    make_bad_case("permission_filter_missing_bad", lambda c: c["permission_filter"].update({"acl_filter": False})),
    make_bad_case("shadow_switch_missing_bad", lambda c: c["shadow_switch"].update({"rollback_index": None})),
    make_bad_case("retrieval_trace_missing_bad", lambda c: c["trace"]["signals"].remove("index_version")),
    make_bad_case("lineage_edge_missing_bad", lambda c: c["lineage"]["edges"].pop()),
    make_bad_case("multi_tenant_isolation_bad", lambda c: c["multi_tenant"].update({"no_cross_tenant_hits": False})),
    make_bad_case("quality_monitoring_missing_bad", lambda c: c["quality_monitoring"]["metrics"].pop("permission_leak_rate")),
    make_bad_case("cost_capacity_unbounded_bad", lambda c: c["cost"].update({"build_cost_usd": 1800.0})),
    make_bad_case("rag_agent_trace_missing_bad", lambda c: c["rag_agent"].update({"citation_source_pointers": False})),
    make_bad_case("infrastructure_gate_missing_bad", lambda c: c["gate"].update({"enabled": False})),
]

audit = MiniFeatureEmbeddingIndexAudit(cases)
examples = audit.examples()
failures = audit.failed_cases()
failed_cases = [case_id for case_id, _ in failures]
failed_gates = [gates[0] for _, gates in failures]

print("feature_embedding_index_examples=" + repr(examples))
print("metrics=" + repr(audit.metric_scores()))
print("hard_blocker_count=" + str(len(failures)))
print("failed_cases=" + repr(failed_cases))
print("failed_gates=" + repr(failed_gates))
print("feature_embedding_index_gate_pass=" + str(not failures))
```

参考输出：

```text
feature_embedding_index_examples={'feature_contract_coverage': 1.0, 'offline_online_skew_rate': 0.0, 'point_in_time_leak_rate': 0.0, 'embedding_dimension': 4, 'embedding_norm_ok': True, 'chunk_embedding_coverage': 1.0, 'bruteforce_top3': ['chunk_policy', 'chunk_beta_private', 'chunk_pricing'], 'permission_filtered_top3': ['chunk_policy', 'chunk_pricing'], 'cross_tenant_blocked': True, 'ann_recall_at_10': 0.93, 'retrieval_p95_ms': 35, 'vector_memory_gib': 0.15, 'shadow_index_ready': True, 'lineage_edge_count': 9, 'estimated_build_cost_usd': 780.0}
metrics={'feature_definition_contract': 0.941, 'offline_online_consistency': 0.941, 'point_in_time_correctness': 0.941, 'embedding_version_contract': 0.941, 'chunk_embedding_lineage': 0.941, 'vector_index_build_readiness': 0.941, 'ann_quality_latency_gate': 0.941, 'metadata_permission_filter': 0.941, 'shadow_index_switch_readiness': 0.941, 'retrieval_trace_completeness': 0.941, 'feature_embedding_lineage_graph': 0.941, 'multi_tenant_isolation': 0.941, 'quality_monitoring_metrics': 0.941, 'cost_capacity_governance': 0.941, 'rag_agent_integration': 0.941, 'feature_embedding_index_gate': 0.941}
hard_blocker_count=16
failed_cases=['feature_definition_missing_bad', 'offline_online_skew_bad', 'point_in_time_leak_bad', 'embedding_version_missing_bad', 'chunk_lineage_missing_bad', 'index_not_ready_bad', 'ann_recall_low_bad', 'permission_filter_missing_bad', 'shadow_switch_missing_bad', 'retrieval_trace_missing_bad', 'lineage_edge_missing_bad', 'multi_tenant_isolation_bad', 'quality_monitoring_missing_bad', 'cost_capacity_unbounded_bad', 'rag_agent_trace_missing_bad', 'infrastructure_gate_missing_bad']
failed_gates=['feature_definition_contract', 'offline_online_consistency', 'point_in_time_correctness', 'embedding_version_contract', 'chunk_embedding_lineage', 'vector_index_build_readiness', 'ann_quality_latency_gate', 'metadata_permission_filter', 'shadow_index_switch_readiness', 'retrieval_trace_completeness', 'feature_embedding_lineage_graph', 'multi_tenant_isolation', 'quality_monitoring_metrics', 'cost_capacity_governance', 'rag_agent_integration', 'feature_embedding_index_gate']
feature_embedding_index_gate_pass=False
```

这段 demo 的重点不是模拟真实 HNSW 或 IVF，而是把平台可审计字段讲清楚。真实生产中可以把 `brute_force_search` 换成 Faiss、Milvus、Qdrant、OpenSearch、pgvector 或自研服务，但版本、权限、trace、lineage、质量和回滚门禁不能省。

## 42.25 常见误区

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

## 42.26 面试常见追问

问题一：Feature Store 解决什么问题？

可以回答：它统一特征定义、计算、存储、版本和在线服务，减少训练和线上特征不一致，支持特征复用、血缘和权限治理。

问题二：Embedding Store 和向量索引有什么区别？

可以回答：Embedding Store 管向量资产及其元数据、版本、来源和权限；向量索引负责高效相似度检索。前者偏资产管理，后者偏检索性能。

问题三：为什么 embedding 要版本化？

可以回答：embedding 依赖 embedding model、预处理、chunking、pooling 和 normalize 策略。版本变了，向量空间也可能变，不能混用。

问题四：如何保证企业 RAG 不越权？

可以回答：文档、chunk、embedding 和索引都记录租户和权限元数据，检索时强制权限过滤，敏感数据可独立索引，并记录 retrieval trace 和审计日志。

问题五：为什么向量索引也要像模型发布一样做 shadow index、canary 和 rollback？

可以回答：embedding model、chunking、索引参数、metadata filter 或数据更新都会改变召回集合。直接替换生产索引可能导致召回退化、权限过滤失效、延迟上涨或空结果率升高。shadow index 先构建和评估，canary 小流量验证，保留旧索引回滚，才能把检索基础设施当成可治理生产系统。

## 42.27 小练习

1. Feature Store 的 offline store 和 online store 有什么区别？
2. 什么是 training-serving skew？
3. Point-in-time correctness 为什么重要？
4. Embedding Store 应记录哪些元数据？
5. HNSW、IVF、PQ 分别适合什么场景？
6. Chunking 策略如何影响 RAG 质量？
7. 为什么向量索引更新需要灰度和回滚？
8. 如何设计一个支持权限过滤的企业向量检索服务？
9. 写一个 0 依赖脚本，检查 embedding model version、chunking version、index version 和 retrieval trace 是否完整。
10. 扩展本章 demo，加入一个“旧索引 metadata filter 漏掉 tenant”的 bad case，并让它触发 metadata permission filter。

## 42.28 本章小结

本章讲了 Feature Store、Embedding Store 与向量索引基础设施。

你需要记住：

1. Feature Store 管理结构化特征，核心价值是复用、版本、训练线上一致和血缘。
2. Embedding Store 管理向量资产，必须记录 embedding model、chunking、预处理、版本和权限。
3. 向量索引负责高效 ANN 检索，需要在召回、延迟、成本和更新复杂度之间权衡。
4. RAG 不是向量数据库本身，而是文档处理、embedding、索引、过滤、rerank、prompt 和 trace 的完整链路。
5. 特征和向量基础设施必须支持多租户、权限控制、版本化、血缘和质量监控。
6. 生产级特征和向量平台必须有 point-in-time、training-serving skew、embedding 版本、shadow index、permission filter、retrieval trace、lineage、成本和回滚门禁。

下一章我们会讲 RAG/Agent 平台中的知识库、工具和 trace 存储。
