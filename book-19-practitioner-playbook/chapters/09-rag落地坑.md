# 第九章：RAG 落地坑

RAG 是大模型项目里最常见的落地形态之一，也是最容易被低估复杂度的系统。很多团队一开始以为 RAG 就是“向量数据库 + LLM”，上线后才发现答案错、引用假、权限漏、文档过期、检索召回不稳定、评估无法解释。RAG 的难点不在于 demo 能不能跑通，而在于能否稳定地把真实用户问题映射到正确、最新、可访问、可引用的证据上。

本章关注 RAG 落地中的真实坑：文档解析、chunk、embedding、混合检索、rerank、上下文构造、答案生成、引用归因、权限控制、索引更新、评估体系和线上事故排查。

## 0. 本讲资料边界与第二轮精修口径

本章第二轮精修时对照了 Retrieval-Augmented Generation 原论文、OpenAI File Search / vector store 资料、RAGAS 评估论文与实现口径、LlamaIndex RAG 评估文档，以及前序第六册部署、第七册评估、第十八册 RAG 产品落地相关内容。这里聚焦防御性的企业 RAG 落地排查和面试表达，不展开特定向量数据库配置、私有知识库真实数据治理制度、生产级检索平台架构或可复用的攻击提示词。

本章第二轮补强重点有三类：

1. 把 retrieval recall、MRR、context recall / precision、citation accuracy、unsupported claim rate、permission leak rate、stale evidence rate、abstention accuracy 和 RAG 上线门禁写成稳定公式。
2. 用一个 0 依赖 Python demo 复盘 RAG 事故：正确文档被 context 丢掉、错误码检索失败、越权证据进入上下文、旧版本文档被引用、多跳问题超预算、线上反馈为负。
3. 把本章和第四册百科、题库、练习、项目与知识图谱同步，确保 RAG 不再只被描述成“向量检索 + 生成”，而是可审计的证据系统。

## 9.1 核心观点

RAG 的失败大多不是单纯生成失败，而是检索、上下文构造、权限、引用和评估失败。

一个可靠 RAG 系统应该回答：

1. 正确文档是否入库。
2. 正确段落是否能被召回。
3. reranker 是否把正确证据排到前面。
4. 上下文是否包含足够且不冲突的证据。
5. 模型是否真正使用证据回答。
6. 引用是否支持答案中的每个关键声明。
7. 用户是否有权限看到这些证据。
8. 文档更新、删除和权限变化后索引是否同步。
9. 评估是否能定位错误发生在哪一环。

面试回答：

```text
我不会把 RAG 简化成向量库加 LLM。一个生产级 RAG 要拆成文档解析、chunk、embedding、召回、rerank、上下文构造、生成、引用、权限和评估。答案错时要做 error attribution，判断是文档没入库、chunk 不合理、retriever 没召回、reranker 排错、context 拼接问题、模型没用证据、引用不支持答案，还是权限和 freshness 问题。
```

## 9.2 常见问题

RAG 落地中常见问题包括：

1. 检索召回不到正确文档。
2. 检索到了正确文档，但模型不用证据。
3. chunk 太短丢上下文，太长浪费 token 并引入噪声。
4. embedding 模型和业务语义不匹配。
5. reranker 把语义相近但不回答问题的文档排到前面。
6. 文档权限控制缺失，导致越权泄露。
7. 引用看似可信，但实际不支持答案。
8. 知识更新后索引不同步，模型引用旧文档。
9. 表格、PDF、图片、代码块解析错误。
10. 用户问题需要多跳证据，但系统只取单段上下文。
11. 检索结果包含冲突证据，模型没有处理冲突。
12. 只评估最终答案，不评估检索、引用和权限。

RAG 系统链路长，每一环都可能失败。真实项目中，排查顺序比单点优化更重要。

## 9.3 先画清楚 RAG 链路

典型 RAG 包含离线链路和在线链路。

离线链路：

1. 文档采集。
2. 文档解析。
3. 清洗和结构化。
4. chunk 切分。
5. metadata 提取。
6. embedding 计算。
7. 索引构建。
8. 权限、版本和更新时间写入。

在线链路：

1. 用户 query 接入。
2. query rewrite 或 intent routing。
3. 权限过滤。
4. sparse retrieval、dense retrieval 或 hybrid retrieval。
5. reranker 精排。
6. context construction。
7. LLM 基于证据生成。
8. citation 和 attribution。
9. 日志、评估、反馈和回归样本沉淀。

排查 RAG 时，不要直接问“模型为什么答错”。更好的问题是：

```text
正确证据是否存在？是否入库？是否被召回？是否被排到前面？是否进入 prompt？是否被模型使用？引用是否真的支持答案？用户是否有权限看到？
```

这条问题链能把模糊的“RAG 不准”拆成可定位、可修复的问题。

## 9.4 文档没入库或解析错

RAG 的第一类事故是正确知识根本没有进入可检索系统。

常见现象：

1. 用户问的问题在原始文档里有答案，但检索永远找不到。
2. PDF 表格内容被解析成乱码或错位文本。
3. 标题、章节层级、列表、脚注丢失。
4. 图片中的 OCR 信息没有进入索引。
5. 代码块、配置项、公式被清洗掉。
6. 文档版本混乱，新旧内容同时存在。

可能原因：

1. 文档采集范围不完整。
2. parser 对 PDF、HTML、Word、表格、扫描件支持差。
3. 清洗规则过度删除。
4. metadata 丢失，导致后续过滤错误。
5. 入库任务失败但没有告警。
6. 文档更新后增量索引没有执行。

排查顺序：

1. 在原始文档中确认答案是否存在。
2. 检查解析后的文本是否保留该答案。
3. 检查 chunk 中是否包含该答案。
4. 检查 embedding 和索引是否成功生成。
5. 检查 metadata、权限、版本、时间字段是否正确。
6. 检查入库任务日志和失败重试。

经验法则：先确认知识是否真的进入系统，再讨论 embedding 和大模型。

## 9.5 Chunk 策略坑

chunk 是 RAG 中最容易被低估的工程决策。

chunk 太小：

1. 语义不完整。
2. 丢失标题和上文条件。
3. 表格、列表、步骤被切断。
4. 模型拿到片段但不知道适用范围。

chunk 太大：

1. 检索粒度粗。
2. 噪声多。
3. prompt 成本高。
4. reranker 难以判断相关性。
5. 多个无关主题混在一起。

常见事故：

```text
用户问“企业版 SSO 配置步骤”，系统召回了整篇管理员手册。手册里确实有答案，但 chunk 太大，里面同时包含计费、权限、API key、审计日志等内容。模型读到上下文后生成了混合答案，引用也不精确。
```

改进方式：

1. 按标题和语义结构切分，而不是只按固定 token 数切分。
2. 保留标题路径，例如 `产品文档 > 管理员设置 > SSO`。
3. 对表格、代码块、步骤列表使用特殊切分策略。
4. 适当 overlap，但不要让重复 chunk 占满 top-k。
5. 在 metadata 中保存文档、章节、版本和更新时间。
6. 用真实 query 做 chunk ablation，而不是凭感觉调大小。

面试表达：

```text
chunk 的目标不是越短越好，也不是越长越好，而是让每个 chunk 语义完整、可检索、可引用、成本可控。我会按文档结构切分，保留标题路径和 metadata，并通过 retrieval recall、rerank 后 context precision 和最终答案质量做 ablation。
```

## 9.6 Embedding 模型不匹配

embedding 决定第一阶段召回能力。如果 embedding 模型和业务语义不匹配，后续 rerank 和 LLM 很难补救。

常见现象：

1. 用户用业务黑话提问，检索不到正式文档。
2. 中文 query 检索英文文档效果差。
3. 代码、配置、错误日志检索效果差。
4. 数字、版本号、产品名、缩写被忽略。
5. 语义相似但答案不相关的文档排前面。

可能原因：

1. 通用 embedding 没覆盖领域术语。
2. query 和文档语言不一致。
3. 业务问题需要关键词匹配，但只用了向量检索。
4. embedding 对数字、符号、代码敏感性不够。
5. 文档 chunk 中缺少标题和上下文。

排查方法：

1. 构造 query-positive-doc 标注集。
2. 看 Recall@K、MRR、nDCG。
3. 分业务术语、代码、中文、英文、数字版本号评估。
4. 对比不同 embedding 模型。
5. 加 BM25 做 hybrid retrieval。
6. 必要时用领域数据微调 embedding 或训练 reranker。

不要只凭“向量相似度看起来合理”判断检索质量。RAG 检索评估必须有标注 query 和正负文档。

## 9.7 只用向量检索的坑

很多 RAG demo 只用 dense vector retrieval，但生产系统通常需要 hybrid retrieval。

向量检索擅长：

1. 语义相近表达。
2. 同义词和改写。
3. 问答式自然语言 query。

关键词检索擅长：

1. 产品名。
2. API 名称。
3. 错误码。
4. 版本号。
5. 配置项。
6. 人名、地名、缩写。

典型事故：

```text
用户搜索错误码 E1427。向量检索认为“登录失败排查”语义相关，排在前面；真正包含 E1427 的故障说明文档没有被召回。最后模型给出通用登录建议，但没有解决问题。
```

改进方式：

1. dense retrieval + BM25 混合召回。
2. 对错误码、API、产品名做精确匹配 boost。
3. 对 query 做实体识别和关键词抽取。
4. 合并多路召回后去重。
5. 用 reranker 做统一排序。
6. 分 query 类型选择检索策略。

真实企业知识库里，纯向量检索通常不够稳。

## 9.8 Reranker 排错

reranker 常用于从 top-50 或 top-100 召回结果中选出最适合放入 prompt 的 top-k。

常见现象：

1. retriever 已召回正确文档，但 reranker 没排到前面。
2. reranker 偏好包含相似关键词但不回答问题的 chunk。
3. 长 chunk 因为包含更多 query 词被误判相关。
4. reranker 延迟高，拖慢 TTFT。
5. reranker 训练数据和业务 query 分布不一致。

排查方式：

1. 单独评估 retrieval recall。
2. 单独评估 reranker top-k accuracy、MRR、nDCG。
3. 对比 rerank 前后正确证据的位置变化。
4. 抽样看 reranker 排错的 hard negative。
5. 按 query 类型分桶，例如事实问答、错误码、流程类、多跳类。
6. 评估 reranker 延迟和收益是否值得。

面试回答：

```text
如果 RAG 答错，我会先确认正确 chunk 是否在 retriever top-k 里。如果在，但没有进入最终 context，问题可能在 reranker 或 context selection。如果不在，问题在文档入库、chunk、embedding 或召回策略。这样可以避免把所有问题都归因给 LLM。
```

## 9.9 Context Construction 坑

检索和 rerank 之后，还要把证据拼成 prompt。很多 RAG 失败发生在 context construction。

常见问题：

1. top-k 直接拼接，重复 chunk 太多。
2. 证据顺序不合理，关键证据被放在中间或最后。
3. chunk 缺少标题、来源、时间和权限信息。
4. 多个版本文档混在一起。
5. 冲突证据没有显式标注。
6. 上下文超过 token budget，被截断掉关键证据。
7. prompt 指令没有要求基于证据回答和资料不足时拒答。

改进方式：

1. 按文档和主题去重。
2. 保留标题路径、来源 URL、版本和更新时间。
3. 对同一主题的多个 chunk 做合并或压缩。
4. 把高置信证据放在更显眼位置。
5. 明确标注冲突证据和新旧版本。
6. 控制 context precision，不要塞入太多弱相关内容。
7. 对资料不足问题要求模型拒答或请求更多信息。

RAG 的上下文不是检索结果的简单拼接，而是面向生成模型的证据组织。

## 9.10 检索到了但模型不用证据

这是 RAG 中很常见的现象：正确证据已经进入 prompt，但模型仍然根据参数记忆或常识回答。

常见原因：

1. prompt 没明确要求基于证据回答。
2. 证据太长或噪声太多，关键句不突出。
3. 模型已有参数知识和证据冲突。
4. 解码温度太高。
5. 问题需要多步推理，模型没有整合证据。
6. 证据格式不适合模型读取，例如表格被解析乱。

排查方法：

1. 把正确证据单独放入 prompt，看模型是否能答对。
2. 缩短上下文，只保留关键证据。
3. 要求模型逐条引用证据。
4. 对回答拆 atomic claims，检查每个 claim 是否有依据。
5. 比较不同 prompt 和解码参数。
6. 如果证据本身难读，回到解析和结构化环节。

常用 prompt 约束：

```text
请只根据给定资料回答。若资料不足以支持答案，请明确说“资料不足”。每个关键结论后必须标注引用编号。不要使用资料之外的常识补全。
```

但 prompt 不是万能的。如果上下文质量差，只靠提示词无法稳定解决幻觉。

## 9.11 引用看似可信但不支持答案

RAG 输出引用很容易让用户产生信任感，但引用本身也可能是错的。

常见引用错误：

1. 引用文档相关，但不支持具体结论。
2. 引用只支持部分结论，模型扩展出了无依据内容。
3. 引用旧版本文档。
4. 引用权限不该展示的文档。
5. 引用位置错了，链接到整篇文档而不是具体段落。
6. 引用编号和正文 claim 对不上。

治理方法：

1. 把答案拆成 atomic claims。
2. 检查每个 claim 是否有至少一个证据支持。
3. 评估 citation accuracy 和 unsupported claim rate。
4. 引用尽量指向段落、表格或章节，而不是整篇文档。
5. 对资料不足场景训练模型拒答。
6. 对高风险场景加人工审核或更严格的 grounding 检查。

面试表达：

```text
RAG 有引用不等于答案可信。我会做 claim-level attribution，把回答拆成原子声明，再检查每个声明是否被引用证据支持。指标上不仅看 answer correctness，还要看 citation accuracy、groundedness 和 unsupported claim rate。
```

## 9.12 权限控制坑

企业 RAG 最严重的事故之一是权限泄露。

常见现象：

1. 普通员工问到了管理层文档内容。
2. 离职员工仍能检索旧权限文档。
3. 跨租户检索返回了其他客户资料。
4. response cache 把高权限用户答案复用给低权限用户。
5. LLM 引用中暴露了用户没有权限打开的链接。

权限过滤应该尽量前置。

常见层级：

1. 文档入库时写入 ACL metadata。
2. 检索前根据用户身份过滤可见文档集合。
3. rerank 和 context construction 只处理有权限文档。
4. 引用链接返回前再次校验权限。
5. cache key 包含用户、租户、权限版本等信息。
6. 权限变更时触发索引和缓存失效。

危险做法：

```text
先全库检索和生成答案，最后再过滤引用。
```

这种做法可能已经把无权限信息泄露到模型上下文和输出中。权限控制必须在检索和上下文构造前就生效。

## 9.13 文档更新和索引同步

RAG 系统里的知识不是静态的。

常见事故：

1. 用户问最新政策，系统回答旧政策。
2. 文档删除后仍然能被检索。
3. 权限变更后旧权限仍生效。
4. 文档更新了，但 embedding 还是旧内容。
5. 同一文档多个版本同时出现，模型混合回答。

需要设计：

1. 文档版本号。
2. 更新时间。
3. 索引构建状态。
4. 删除和失效标记。
5. 增量索引任务。
6. 失败重试和告警。
7. 缓存失效策略。
8. 新旧版本冲突处理。

排查时要问：

1. 原文档什么时候更新。
2. parser 什么时候重新解析。
3. chunk 和 embedding 什么时候更新。
4. vector index 什么时候可见。
5. cache 是否仍命中旧结果。
6. 用户权限版本是否同步。

RAG 的 freshness 是产品能力，不是后勤细节。

## 9.14 多跳和综合问题

很多企业问题不是单段文档能回答的。

例如：

```text
如果我从专业版升级到企业版，并开启 SSO，账单周期和管理员权限会怎么变化？
```

这个问题可能需要：

1. 版本升级文档。
2. 计费文档。
3. SSO 配置文档。
4. 管理员权限文档。

常见失败：

1. 只召回其中一类证据。
2. 模型只回答最显眼的部分。
3. 多个证据之间存在条件依赖，模型没处理。
4. prompt token budget 不够，部分证据被截断。
5. 引用只覆盖局部结论。

改进方式：

1. query decomposition，把复杂问题拆成子问题。
2. 多路检索，分别找不同子问题证据。
3. context 中按子问题组织证据。
4. 让模型先列出依据，再综合回答。
5. 对多跳任务单独评估。

多跳 RAG 比单跳 FAQ 难很多，不能用简单 QA 测试集代表真实能力。

## 9.15 RAG 评估不能只看答案

RAG 评估至少要分三层。

第一层：检索评估。

1. Recall@K。
2. Precision@K。
3. MRR。
4. nDCG。
5. 正确证据是否进入 final context。

第二层：生成评估。

1. answer correctness。
2. faithfulness。
3. groundedness。
4. citation accuracy。
5. unsupported claim rate。
6. abstention accuracy。

第三层：系统评估。

1. TTFT。
2. TPOT。
3. 成本。
4. 索引更新延迟。
5. 权限泄露率。
6. cache 命中率。
7. 用户满意度。

只看最终回答准确率，会掩盖检索和引用的问题。一个答案可能碰巧对，但引用是错的；也可能检索对了，但模型没有使用证据。

## 9.16 RAG Error Attribution 表

建议为每个 bad case 标注错误归因。

```text
问题：用户原始 query
正确答案：人工标注答案
正确证据：文档 ID、chunk ID、段落位置
入库状态：原文是否存在、解析是否正确、chunk 是否存在
召回状态：retriever top-k 是否包含正确证据
重排状态：reranker 是否把正确证据排入 final context
上下文状态：final prompt 是否包含正确证据、是否有冲突证据
生成状态：模型是否使用证据、是否有 unsupported claim
引用状态：引用是否支持答案、是否指向正确位置
权限状态：用户是否有权访问引用证据
freshness：文档是否最新、索引是否同步
根因：解析 / chunk / embedding / retrieval / rerank / context / generation / citation / permission / freshness
修复：对应修复动作
```

有了这张表，团队才能知道下一步该改数据、检索、reranker、prompt、权限系统还是评估集。

## 9.17 典型事故：正确文档被召回但答案仍然错

现象：

```text
排查发现正确文档在 retriever top-5 里，但最终答案仍然错误。
```

可能原因：

1. 正确 chunk 被 reranker 排到后面，没有进入 final context。
2. final context 中有冲突旧文档，模型选错。
3. chunk 太大，关键句被噪声淹没。
4. prompt 没要求基于证据回答。
5. 模型使用参数知识覆盖了检索证据。
6. 引用正确但 answer claim 扩展过度。

排查：

1. 看 retriever top-k。
2. 看 reranker 后排序。
3. 看 final prompt 实际内容。
4. 把正确证据单独喂给模型。
5. 对答案做 claim-level attribution。
6. 检查文档版本和冲突证据。

修复方向：

1. 调整 reranker。
2. 改 context selection。
3. 压缩或突出关键证据。
4. 强化 grounding prompt。
5. 增加引用和拒答评估。

## 9.18 RAG 事故复盘模板

```text
现象：RAG 答案错误、引用错误、权限泄露、文档过期或召回失败
影响：影响哪些用户、文档集合、业务线和时间窗口
样本：问题、模型答案、正确答案、引用、正确证据
链路：解析、chunk、embedding、retrieval、rerank、context、generation、citation、permission、freshness
排查：正确证据是否入库、召回、重排、进入 prompt、被使用、被正确引用
根因：文档处理、检索、排序、prompt、权限、索引同步或评估缺失
修复：补索引、改 chunk、调召回、训练 reranker、改 prompt、加权限过滤、更新评估集
预防：RAG bad case 回归集、权限测试、freshness 监控、citation 检查和上线门禁
```

复盘时不要只写“模型幻觉”。如果答案没有被证据支持，要说明是证据没到、证据没用、证据冲突，还是引用校验缺失。

## 9.18.1 关键公式与 RAG 事故指标速查

**1. RAG 样本抽象**

把第 `i` 个 RAG 样本写成：

```math
q_i=(x_i,U_i,E_i,R_i,Z_i,A_i,C_i,F_i)
```

其中 `x_i` 是用户问题，`U_i` 是用户身份和权限，`E_i` 是标准证据集合，`R_i` 是第一阶段召回结果，`Z_i` 是 rerank 后进入最终上下文的证据，`A_i` 是生成答案，`C_i` 是答案中的 claim / citation 对齐表，`F_i` 是 freshness、latency、cost、线上反馈等系统字段。

这个抽象能把“RAG 答错”拆成四类证据问题：证据不存在、证据没召回、证据没进上下文、证据进了但没被正确使用。

**2. Retrieval Recall@K**

```math
\mathrm{Recall@K}_i=\frac{|R_i^{K}\cap E_i|}{|E_i|}
```

其中 `R_i^K` 是 retriever top-k 候选集合，`E_i` 是人工标注的标准证据集合。这个指标回答：正确证据有没有被第一阶段召回。

**3. MRR**

```math
\mathrm{MRR}=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{\mathrm{rank}_i}
```

其中 `rank_i` 是第一个正确证据在召回列表中的排名；如果没有正确证据，记为 0。MRR 比 Recall@K 更关注正确证据是否靠前。

**4. Context Recall 与 Context Precision**

```math
\mathrm{CR}_i=\frac{|Z_i\cap E_i|}{|E_i|}
```

```math
\mathrm{CP}_i=\frac{|Z_i\cap E_i|}{|Z_i|}
```

`CR_i` 回答“标准证据是否进入最终 prompt”，`CP_i` 回答“最终 prompt 里有多少是真相关证据”。RAG 不只要召回多，还要避免把大量噪声塞进上下文。

**5. Citation Accuracy**

```math
A_{\mathrm{cite}}=\frac{1}{M}\sum_{m=1}^{M}\mathbf{1}[c_m\Rightarrow z_m]
```

其中 `c_m` 是答案里的第 `m` 个 claim，`z_m` 是它引用的证据。这个指标要求引用真正支持 claim，而不是只和主题相关。

**6. Unsupported Claim Rate**

```math
R_{\mathrm{unsup}}=1-A_{\mathrm{cite}}
```

如果 unsupported claim rate 高，说明模型仍在用参数记忆或语言补全生成证据外内容。有引用不等于 grounded。

**7. Permission Leak Rate**

```math
R_{\mathrm{perm}}=\frac{\sum_i |\{z\in Z_i:z\notin \mathcal{A}(U_i)\}|}{\sum_i |Z_i|}
```

其中 `\mathcal{A}(U_i)` 是用户 `U_i` 可访问的证据集合。权限过滤必须发生在 retrieval / rerank / context construction 之前，不能生成后再过滤引用。

**8. Stale Evidence Rate**

```math
R_{\mathrm{stale}}=\frac{\sum_i |\{z\in Z_i:\mathrm{stale}(z)=1\}|}{\sum_i |Z_i|}
```

企业知识会更新。旧版本文档进入上下文时，模型可能给出非常自信但已经失效的答案。

**9. Abstention Accuracy**

```math
A_{\mathrm{abs}}=\frac{1}{N_{\mathrm{abs}}}\sum_i \mathbf{1}[\hat a_i=\mathrm{abstain}]
```

这个指标只在资料不足、权限不足、证据冲突或文档过期样本上计算。RAG 产品不是所有问题都要回答，正确拒答是能力的一部分。

**10. RAG 事故门禁**

```math
G_{\mathrm{rag}}=\mathbf{1}\left[
\bar R_{\mathrm{ret}}\ge\tau_{\mathrm{ret}}
\land \bar C_{\mathrm{rec}}\ge\tau_{\mathrm{crec}}
\land \bar C_{\mathrm{prec}}\ge\tau_{\mathrm{cprec}}
\land A_{\mathrm{cite}}\ge\tau_{\mathrm{cite}}
\land R_{\mathrm{perm}}=0
\land R_{\mathrm{stale}}\le\tau_{\mathrm{stale}}
\land A_{\mathrm{abs}}\ge\tau_{\mathrm{abs}}
\land P95(L)\le\tau_{\mathrm{lat}}
\land \bar F_{\mathrm{online}}>0
\right]
```

这个门禁把检索、上下文、引用、权限、freshness、拒答、延迟和线上反馈放到同一张表里。只要其中一项失败，就不能只凭一个“答案看起来不错”的样例上线。

## 9.18.2 最小可运行 RAG 事故审计 demo

下面的 demo 不依赖外部库。它故意构造 5 个 RAG bad case：旧版本退货政策被引用、SSO + 计费多证据样本正常、错误码检索失败、普通员工越权看到薪酬文档、多跳升级问题缺少当前退货政策且上下文超预算。

```python
from math import ceil

docs = {
    "return_v1": {"acl": {"employee", "admin"}, "tokens": 120, "stale": True},
    "return_v2": {"acl": {"employee", "admin"}, "tokens": 130, "stale": False},
    "sso_admin": {"acl": {"admin"}, "tokens": 180, "stale": False},
    "billing_enterprise": {"acl": {"admin"}, "tokens": 160, "stale": False},
    "error_e1427": {"acl": {"employee", "admin"}, "tokens": 90, "stale": False},
    "comp_private": {"acl": {"exec"}, "tokens": 140, "stale": False},
    "login_generic": {"acl": {"employee", "admin"}, "tokens": 100, "stale": False},
}

cases = [
    {
        "id": "return_policy_current",
        "role": "employee",
        "expected": ["return_v2"],
        "retrieved": ["return_v1", "return_v2", "login_generic"],
        "reranked": ["return_v1", "return_v2", "login_generic"],
        "context": ["return_v1", "login_generic"],
        "budget": 320,
        "claims": [
            {"claim": "Return window is 14 days", "citation": "return_v1", "support": "return_v2"},
        ],
        "should_abstain": False,
        "latency_ms": 980,
        "cost": 0.018,
        "online_delta": -0.20,
    },
    {
        "id": "sso_billing_admin",
        "role": "admin",
        "expected": ["sso_admin", "billing_enterprise"],
        "retrieved": ["sso_admin", "login_generic", "billing_enterprise"],
        "reranked": ["sso_admin", "billing_enterprise", "login_generic"],
        "context": ["sso_admin", "billing_enterprise"],
        "budget": 420,
        "claims": [
            {"claim": "SSO requires enterprise admin", "citation": "sso_admin", "support": "sso_admin"},
            {"claim": "Billing changes at next cycle", "citation": "billing_enterprise", "support": "billing_enterprise"},
        ],
        "should_abstain": False,
        "latency_ms": 1180,
        "cost": 0.023,
        "online_delta": 0.08,
    },
    {
        "id": "error_code_e1427",
        "role": "employee",
        "expected": ["error_e1427"],
        "retrieved": ["login_generic", "return_v1"],
        "reranked": ["login_generic", "return_v1"],
        "context": ["login_generic"],
        "budget": 250,
        "claims": [
            {"claim": "E1427 means generic login failure", "citation": "login_generic", "support": "error_e1427"},
        ],
        "should_abstain": False,
        "latency_ms": 920,
        "cost": 0.015,
        "online_delta": -0.15,
    },
    {
        "id": "private_comp_plan",
        "role": "employee",
        "expected": [],
        "retrieved": ["comp_private", "login_generic"],
        "reranked": ["comp_private", "login_generic"],
        "context": ["comp_private"],
        "budget": 220,
        "claims": [
            {"claim": "Compensation plan is visible", "citation": "comp_private", "support": "comp_private"},
        ],
        "should_abstain": True,
        "latency_ms": 1200,
        "cost": 0.019,
        "online_delta": -0.45,
    },
    {
        "id": "upgrade_multi_hop",
        "role": "admin",
        "expected": ["sso_admin", "billing_enterprise", "return_v2"],
        "retrieved": ["sso_admin", "billing_enterprise", "return_v1"],
        "reranked": ["sso_admin", "billing_enterprise", "return_v1"],
        "context": ["sso_admin", "billing_enterprise", "return_v1"],
        "budget": 400,
        "claims": [
            {"claim": "SSO setup needs admin", "citation": "sso_admin", "support": "sso_admin"},
            {"claim": "Billing changes next cycle", "citation": "billing_enterprise", "support": "billing_enterprise"},
            {"claim": "Return policy remains old", "citation": "return_v1", "support": "return_v2"},
        ],
        "should_abstain": False,
        "latency_ms": 1850,
        "cost": 0.041,
        "online_delta": -0.10,
    },
]


def mean(values):
    return sum(values) / max(1, len(values))


def percentile(values, pct):
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, ceil(len(ordered) * pct / 100) - 1))
    return ordered[idx]


def recall(items, expected):
    if not expected:
        return 1.0
    return len(set(items) & set(expected)) / len(expected)


def first_relevant_rank(items, expected):
    for idx, item in enumerate(items, start=1):
        if item in expected:
            return idx
    return None


retrieval_recalls = []
context_recalls = []
context_precisions = []
rr_scores = []
permission_leaks = []
stale_context = []
budget_overflows = []
claim_results = []
abstention_results = []
root_causes = {}

for case in cases:
    expected = case["expected"]
    if expected:
        retrieval_recalls.append(recall(case["retrieved"], expected))
        context_recalls.append(recall(case["context"], expected))
        rank = first_relevant_rank(case["retrieved"], expected)
        rr_scores.append(0.0 if rank is None else 1 / rank)

    relevant_context = [doc for doc in case["context"] if doc in expected]
    context_precisions.append(len(relevant_context) / max(1, len(case["context"])))

    context_tokens = sum(docs[doc]["tokens"] for doc in case["context"])
    if context_tokens > case["budget"]:
        budget_overflows.append(case["id"])

    for doc in case["context"]:
        if case["role"] not in docs[doc]["acl"]:
            permission_leaks.append((case["id"], doc))
        if docs[doc]["stale"]:
            stale_context.append((case["id"], doc))

    if case["should_abstain"]:
        abstention_results.append(len(case["claims"]) == 0)

    for claim in case["claims"]:
        ok = claim["citation"] == claim["support"] and not docs[claim["citation"]]["stale"]
        claim_results.append(ok)

    if expected and recall(case["retrieved"], expected) < 1:
        root_causes[case["id"]] = "retrieval_miss"
    elif expected and recall(case["context"], expected) < 1:
        root_causes[case["id"]] = "rerank_or_context_drop"
    elif any((case["id"], doc) in stale_context for doc in case["context"]):
        root_causes[case["id"]] = "stale_evidence"
    elif any((case["id"], doc) in permission_leaks for doc in case["context"]):
        root_causes[case["id"]] = "permission_leak"
    elif not all(claim_results[-len(case["claims"]):]):
        root_causes[case["id"]] = "citation_or_grounding"
    else:
        root_causes[case["id"]] = "pass"

citation_accuracy = mean([1 if ok else 0 for ok in claim_results])
unsupported_claim_rate = 1 - citation_accuracy
permission_leak_rate = len(permission_leaks) / sum(len(case["context"]) for case in cases)
stale_evidence_rate = len(stale_context) / sum(len(case["context"]) for case in cases)
abstention_accuracy = mean([1 if ok else 0 for ok in abstention_results])
budget_overflow_rate = len(budget_overflows) / len(cases)
avg_online_delta = mean([case["online_delta"] for case in cases])
p95_latency = percentile([case["latency_ms"] for case in cases], 95)
avg_cost = mean([case["cost"] for case in cases])

metrics = {
    "retrieval_recall": round(mean(retrieval_recalls), 3),
    "mrr": round(mean(rr_scores), 3),
    "context_recall": round(mean(context_recalls), 3),
    "context_precision": round(mean(context_precisions), 3),
    "citation_accuracy": round(citation_accuracy, 3),
    "unsupported_claim_rate": round(unsupported_claim_rate, 3),
    "permission_leak_rate": round(permission_leak_rate, 3),
    "stale_evidence_rate": round(stale_evidence_rate, 3),
    "abstention_accuracy": round(abstention_accuracy, 3),
    "budget_overflow_rate": round(budget_overflow_rate, 3),
    "p95_latency_ms": p95_latency,
    "avg_cost": round(avg_cost, 3),
    "avg_online_delta": round(avg_online_delta, 3),
}

failed_gates = []
if metrics["retrieval_recall"] < 0.80 or metrics["mrr"] < 0.70:
    failed_gates.append("retrieval")
if metrics["context_recall"] < 0.75 or metrics["context_precision"] < 0.60:
    failed_gates.append("context")
if metrics["citation_accuracy"] < 0.80 or metrics["unsupported_claim_rate"] > 0.10:
    failed_gates.append("citation_grounding")
if metrics["permission_leak_rate"] > 0:
    failed_gates.append("permission")
if metrics["stale_evidence_rate"] > 0:
    failed_gates.append("freshness")
if metrics["abstention_accuracy"] < 0.90:
    failed_gates.append("abstention")
if metrics["budget_overflow_rate"] > 0 or p95_latency > 1500 or avg_cost > 0.030:
    failed_gates.append("latency_cost_budget")
if avg_online_delta <= 0:
    failed_gates.append("online_feedback")

report = {
    "metrics": metrics,
    "permission_leaks": permission_leaks,
    "stale_context": stale_context,
    "budget_overflows": budget_overflows,
    "root_causes": root_causes,
    "failed_gates": failed_gates,
    "gate_pass": not failed_gates,
}

for key, value in report.items():
    print(f"{key}=", value)
```

一次输出示例：

```text
metrics= {'retrieval_recall': 0.667, 'mrr': 0.625, 'context_recall': 0.417, 'context_precision': 0.333, 'citation_accuracy': 0.625, 'unsupported_claim_rate': 0.375, 'permission_leak_rate': 0.111, 'stale_evidence_rate': 0.222, 'abstention_accuracy': 0.0, 'budget_overflow_rate': 0.2, 'p95_latency_ms': 1850, 'avg_cost': 0.023, 'avg_online_delta': -0.164}
permission_leaks= [('private_comp_plan', 'comp_private')]
stale_context= [('return_policy_current', 'return_v1'), ('upgrade_multi_hop', 'return_v1')]
budget_overflows= ['upgrade_multi_hop']
root_causes= {'return_policy_current': 'rerank_or_context_drop', 'sso_billing_admin': 'pass', 'error_code_e1427': 'retrieval_miss', 'private_comp_plan': 'permission_leak', 'upgrade_multi_hop': 'retrieval_miss'}
failed_gates= ['retrieval', 'context', 'citation_grounding', 'permission', 'freshness', 'abstention', 'latency_cost_budget', 'online_feedback']
gate_pass= False
```

这段输出说明：RAG 事故不能只看最终回答是否通顺。`return_policy_current` 的正确文档被召回了，但最终上下文丢掉了当前版本；`error_code_e1427` 是召回阶段失败；`private_comp_plan` 是权限过滤前置失败；`upgrade_multi_hop` 同时缺少当前证据、引用旧文档并超出上下文预算。修复顺序也应该按 root cause 分流：先补召回和 hybrid retrieval，再修 context selection / reranker，然后做权限前置、freshness 失效、citation gate 和拒答训练。

## 9.19 面试题：RAG 答错了怎么排查

回答要点：

```text
我会先找正确答案对应的证据，然后沿着 RAG 链路排查。第一，原文档是否存在并解析正确；第二，chunk 是否包含正确证据；第三，retriever top-k 是否召回；第四，reranker 是否排入 final context；第五，prompt 中是否包含足够证据和冲突信息；第六，模型是否基于证据回答；第七，引用是否真的支持答案；最后检查权限和文档 freshness。
```

## 9.20 面试题：如何设计企业 RAG 权限控制

回答要点：

```text
我会把权限控制前置到检索前，而不是生成后再过滤。文档入库时写入租户、用户组、角色、文档级和段落级 ACL metadata。在线查询时先根据用户身份过滤可访问文档集合，retrieval、rerank 和 context construction 都只能使用有权限证据。引用返回前再次校验权限，cache key 也要包含租户和权限版本。权限变更时要触发索引或缓存失效。
```

## 9.21 面试题：如何评估 RAG 系统

回答要点：

```text
我会分层评估。检索层看 Recall@K、MRR、nDCG 和正确证据是否进入 final context；生成层看答案正确性、faithfulness、groundedness、citation accuracy、unsupported claim rate 和资料不足时的拒答；系统层看 TTFT、成本、索引更新延迟、权限泄露率和线上用户反馈。同时要做 bad case error attribution，把错误归因到解析、chunk、retrieval、rerank、context、generation、citation、permission 或 freshness。
```

## 9.22 排查清单

核心清单：

1. 单独评估 retrieval recall。
2. 单独评估 reranker。
3. 检查 chunk 策略和标题路径。
4. 检查 prompt 是否强制基于证据回答。
5. 检查权限过滤是否在检索前生效。
6. 做 answer attribution 和 citation accuracy 评估。
7. 检查文档版本、更新时间和索引同步。
8. 建立 RAG bad case 回归集。

扩展清单：

1. 正确文档是否入库。
2. PDF、表格、图片 OCR 是否解析正确。
3. embedding 是否适配业务术语、代码、错误码和多语言。
4. 是否需要 BM25 + dense hybrid retrieval。
5. top-k 是否被重复 chunk 占满。
6. final context 是否有冲突证据。
7. 模型是否在资料不足时拒答。
8. cache 是否复用过期或越权答案。
9. 索引更新失败是否有告警。
10. 评估集是否覆盖真实线上问题。

## 9.23 经验法则

RAG 的经验可以总结为：

1. 先确认知识入库，再调检索模型。
2. 看最终答案，也看正确证据是否进入 prompt。
3. 看检索召回，也看 rerank 和 context precision。
4. 有引用不等于 grounded，必须做 claim-level 检查。
5. 权限必须检索前过滤，不能生成后补救。
6. 文档更新、删除和权限变化都要触发索引或缓存失效。
7. RAG 评估要分检索、生成和系统三层。
8. 每个 bad case 都要做 error attribution，而不是笼统说模型幻觉。

下一章会进入 Agent 落地坑。RAG 解决的是“基于外部知识回答”，Agent 还要进一步处理工具调用、任务分解、状态管理、执行安全和可恢复性。
