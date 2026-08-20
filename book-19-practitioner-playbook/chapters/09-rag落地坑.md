# 第九章：RAG 落地坑：把检索结果变成可审计证据

RAG 经常被介绍成“向量数据库加一个大模型”，但生产系统真正承担的是一条证据责任链：把用户问题映射到正确、最新、可访问、可引用的资料，再让模型在证据范围内回答，最后让用户能够核查答案的来源。

链路中的任何一环出错，最终都可能表现成一句“模型幻觉”。原文档可能没有入库，PDF 表格可能解析错，chunk 可能切断条件，embedding 可能漏掉错误码，reranker 可能把旧版本排在前面，权限过滤可能晚于生成，引用可能只和主题相关却不支持具体声明。若只修 prompt，就会把不同根因混成一个问题。

本章用一次企业知识库事故贯穿全文：客服问的是最新退货政策，系统召回了旧版本；管理员的多跳问题缺少一个关键证据；错误码查询被通用登录说明抢走；普通员工看到了薪酬文档。读者会沿着文档解析、chunk、embedding、混合检索、rerank、上下文构造、生成、claim 归因、引用、权限、freshness、多跳查询、拒答和评估逐段排查。

## 9.0 先看一场“回答很流畅”的事故

某企业把内部文档接入了 RAG。上线初期，答案很像产品宣传页：语气自然，引用链接齐全，用户感觉比直接问基础模型可靠。

一周后，客服发现三个问题。第一，系统把已经废止的退货政策作为最新政策引用；第二，普通员工询问薪酬计划时，答案泄露了只有高管可见的文档片段；第三，用户输入错误码 `E1427` 时，系统返回了泛化的“登录失败排查”而没有解释错误码。

团队最初只调大 top-k 和上下文长度，结果把更多旧文档、无关文档和越权文档塞进了 prompt，成本上升，问题反而更难定位。真正的修复需要把一次 RAG 请求拆成证据状态：

```text
原文档是否存在 -> 是否正确解析 -> 是否形成正确 chunk
-> 是否被召回 -> 是否被重排 -> 是否进入最终 context
-> 用户是否有权限 -> 证据是否最新 -> claim 是否被支持
-> 是否应该回答、拒答或转人工
```

对初学者来说，RAG 像一个带索引和引用的资料助理：先找资料，再根据资料回答。对专家来说，RAG 是一个带数据平面、权限平面、版本平面和生成平面的证据系统；“检索到了”只是中间事件，不是用户可采纳答案。

## 9.1 RAG 的对象边界和任务合同

RAG 至少包含两个记忆来源：模型参数中的参数记忆，以及外部文档、数据库或检索索引中的非参数记忆。外部资料可以提供更新能力和 provenance，但不自动保证正确性、权限和可用性。

在设计系统前，先写清楚任务合同：

```text
问题类型：事实查询、流程查询、比较、多跳、总结还是开放分析
证据要求：必须引用、允许概括，还是只返回可验证字段
时效要求：实时、按生效日期、按文档版本还是历史回溯
权限要求：租户、组织、角色、文档、段落和字段级边界
不足处理：资料不足时拒答、追问、转人工还是返回候选资料
成功定义：答案正确、引用支持、用户采纳和任务完成分别如何判断
```

如果产品没有定义“资料不足时应该怎样做”，模型就会倾向于继续生成一段完整文字。很多幻觉不是模型突然变坏，而是系统从未给它一个合法的“不知道”状态。

### 9.1.1 RAG 有四个不同的质量层

第一层是**资料层**：原文档、版本、解析文本、表格结构和 ACL 是否正确。

第二层是**检索层**：正确证据是否被召回、排序和过滤。

第三层是**生成层**：模型是否使用证据、是否完成任务、是否在证据不足时克制。

第四层是**产品层**：引用能否打开、信息是否最新、用户是否有权访问、延迟和成本是否可接受。

四层不能互相替代。检索 Recall@10 很高，不等于 final context 有正确证据；context 有正确证据，不等于 claim 被引用支持；答案碰巧正确，也不等于用户有权看到证据。

## 9.2 用证据账本记录每次请求

把第 `i` 个 RAG 请求抽象为：

$$
q_i=(x_i,u_i,e_i,r_i,z_i,a_i,c_i,p_i,f_i,o_i)
$$

其中：

- `x_i` 是用户问题和会话上下文；
- `u_i` 是用户、租户、角色和权限版本；
- `e_i` 是人工标注或程序验证得到的标准证据集合；
- `r_i` 是初始召回结果；
- `z_i` 是进入最终 context 的证据；
- `a_i` 是模型答案或完整生成 trace；
- `c_i` 是答案中的原子声明及其引用映射；
- `p_i` 是权限检查、过滤和拒绝记录；
- `f_i` 是文档版本、更新时间和索引同步状态；
- `o_i` 是延迟、成本、用户反馈和任务结果。

这张账本的目的不是保存更多日志，而是让“证据在哪里丢了”可以被回答。没有 `r_i`，无法区分召回失败和生成失败；没有 `z_i`，无法知道 reranker 或预算截断是否丢掉了证据；没有 `u_i`，无法复核权限；没有 `f_i`，无法判断旧文档是否应被淘汰。

### 9.2.1 最少要保留的版本字段

```text
dataset/document_id、document_version、effective_at、expires_at
parser_version、chunker_version、embedding_model/version
sparse_index、dense_index、reranker_version、filter_policy_version
query_rewrite_version、prompt_template、model_id、decoding
retrieved_ids、reranked_ids、context_ids、raw_output、claims
permission_snapshot、cache_key_version、latency、cost、feedback
```

如果只存最终答案和一个“引用链接”，后续不能判断问题来自文档、索引、过滤、上下文、模型还是引用映射。证据账本应与第七章的评估记录、第八章的推理 trace 关联起来。

## 9.3 文档采集和解析：正确知识可能一开始就消失

RAG 的第一类失败发生在模型之前。用户问题的答案可能存在于原始 PDF、网页、表格、图片或代码仓库中，但解析器没有把它转换成可检索对象。

### 9.3.1 不同文档类型有不同的损坏方式

PDF 可能丢失阅读顺序、页眉页脚和表格列关系；扫描件依赖 OCR，数字和符号容易错；HTML 可能把导航、广告和正文混在一起；Word 文档的标题层级和批注可能丢失；表格转成纯文本后，行列条件可能无法恢复；代码和配置被清洗时，缩进、版本号和参数名可能消失。

因此“解析成功”不能只看文件没有报错。需要抽样比较原文和结构化结果：

```text
标题路径是否保留
表格行列和单位是否保留
脚注、条件、例外和生效日期是否保留
图片 OCR、公式、代码块和错误码是否保留
文档 ID、版本、来源、ACL 和时间字段是否保留
```

### 9.3.2 文档入库的验证顺序

遇到“检索不到答案”，先按以下顺序验证：

1. 原始来源中是否确实有答案；
2. 采集任务是否拿到了正确版本；
3. 解析文本是否包含答案及其上下文；
4. chunk 是否保留了答案、标题和条件；
5. embedding、倒排索引和 metadata 是否生成；
6. 索引是否已发布到在线版本；
7. 权限过滤是否把它排除了；
8. cache 是否仍返回旧的检索结果。

只有确认知识进入在线索引后，才值得调 embedding 或 reranker。把一个没有入库的文档交给更强的模型，不能解决问题。

### 9.3.3 失败的入库任务必须可见

增量索引常见的危险状态是“部分成功”：一批文档中 99% 成功，1% 失败，但任务被标记为完成。产品看板若只显示任务状态，就会把缺失知识解释成模型能力不足。

每个文档应有明确状态：`discovered`、`parsed`、`chunked`、`embedded`、`published`、`failed`、`deleted`。状态转换应带版本和时间；失败要有重试、告警和人工查看入口。删除请求也必须传播到 chunk、dense index、sparse index、缓存和评估快照，而不是只删除原文件。

## 9.4 Chunk 不是切固定长度就结束

chunk 同时决定召回粒度、上下文噪声、引用精度和 token 成本。它不是一个单纯的 tokenizer 参数。

### 9.4.1 小 chunk 和大 chunk 的代价

小 chunk 的问题是语义不完整：标题、适用条件、例外条款可能在相邻块，模型只看到一个孤立步骤。大 chunk 的问题是相关性变粗：正确答案和计费、权限、历史版本等无关内容混在一起，reranker 难以判断，context 成本也上升。

例如用户问“企业版 SSO 的配置步骤”，整篇管理员手册虽然包含答案，却同时包含 API key、计费、审计和账户删除。模型可能把不同章节的条件拼成一套不存在的流程，引用也只能指向整篇手册。

### 9.4.2 结构化 chunk 的基本字段

```text
chunk_id、document_id、document_version
title_path、section、page/paragraph/table/line span
text、content_type、language、effective_at、expires_at
tenant、ACL、parent/neighbor chunk、source_uri
```

标题路径比一段孤立文本更能告诉模型“这句话适用于谁”。页码、行号、表格坐标和代码行范围让引用可以落到具体证据，而不是只给一个文档首页。

### 9.4.3 特殊结构要特殊切分

步骤列表应尽量保留顺序和前置条件；表格应保存列名、单位和行上下文；代码块应保持语言、缩进和版本；FAQ 可以把问题和答案作为一个语义单元；法律和政策条款需要把定义、例外、生效时间与主体绑定。

overlap 可以保留边界语义，但 overlap 太大时会让 top-k 被同一段重复内容占满。应通过真实 query 的证据召回、上下文精确率、引用粒度和 token 成本做 chunk ablation，而不是凭感觉选择 256 或 512 token。

## 9.5 Embedding、关键词和混合检索

embedding 召回擅长同义表达和语义改写，但企业问题常包含产品名、错误码、版本号、API、缩写和数字，这些内容不一定适合只用向量相似度判断。

### 9.5.1 业务语义不匹配的信号

```text
业务黑话 query 找不到正式术语文档
中文问题召回英文或机器翻译内容不稳定
错误码和版本号被泛化成“相似主题”
代码、配置键和 API 名称召回不准
同义但不回答问题的说明排在真正规范之前
```

排查要建立 query-positive-negative 数据集，按事实、流程、错误码、代码、多语言、数字和版本切片计算结果。只观察向量距离不能证明检索正确，因为相似度是模型空间中的距离，不是业务相关性。

### 9.5.2 Dense、sparse 和 hybrid 的职责

Dense retrieval 通常对自然语言改写、同义词和语义近似有帮助；BM25 或其他 sparse 方法对精确词、数字、产品名、错误码和配置项更稳。混合检索不是“两个结果简单拼接”，还要解决分数不可比、重复、权限过滤和合并排序。

一个常见路线是：多路召回扩大候选集，按文档和 chunk 去重，再使用 reranker 统一排序。不同 query 类型也可以路由到不同的召回组合，而不是对所有问题使用同一套 top-k。

### 9.5.3 Recall@K 的定义域

对第 `i` 个可评估问题，标准证据集合 `E_i` 非空，初始 top-k 集合为 `R_i^K`：

$$
Recall@K_i=\frac{\lvert R_i^K\cap E_i\rvert}{\lvert E_i\rvert}
$$

如果人工没有标注标准证据，结果是 `unknown`；如果该问题不需要外部证据，指标是 `not_applicable`；`E_i` 为空不能被当成 Recall 1。评估集应区分“无需检索的闲聊”和“必须找到规范的知识问题”。

## 9.6 Reranker：把候选变成可用证据

初始召回的目标通常是不要漏掉正确证据，reranker 的目标是把真正回答问题的内容排到有限上下文预算之内。两者不能用同一个指标替代。

### 9.6.1 先判断错误发生在哪一层

对每个 bad case 保存三个列表：

```text
retriever_top_k
reranker_order
final_context
```

如果正确证据不在第一列表，优先检查解析、chunk、embedding、sparse/hybrid 和过滤；如果在第一列表却不在第二列表，检查 reranker 和 hard negative；如果在第二列表却不在最终 context，检查预算、去重、版本、权限和 context selection。

这样做的价值是避免用更大的模型掩盖检索问题，也避免把 reranker 延迟误归因给 LLM。

### 9.6.2 Reranker 的常见捷径

长 chunk 因为包含更多 query 词而得到更高分；主题相关但结论不适用的文档排在前面；旧版本与新版本都很相似，reranker 没有理解生效日期；标题匹配很好但正文没有答案；安全或权限字段没有进入排序特征。

要构造 hard negative：它们应在表面上相似，但在版本、主体、地区、产品套餐或条件上不适用。只用随机负例训练和评估，会让排序器显得很好，却无法处理真实混淆项。

### 9.6.3 排序指标也有边界

若第一个标准证据在排名 `rank_i`，则单样本 reciprocal rank 为 `1/rank_i`；没有标准证据时记为 0，但只有当该样本确实应该有证据时才成立。平均倒数排名：

$$
MRR=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{rank_i},\qquad N>0
$$

如果 `N=0` 或样本标签缺失，MRR 没有定义。MRR 关注第一个正确证据，不能替代多证据覆盖、权限和版本正确性。

## 9.7 Context construction：把候选组织成证据

最终 prompt 不是把 top-k 文本用换行符拼起来。它要解决重复、顺序、版本冲突、权限、预算、来源和引用映射。

### 9.7.1 一个 context 单元至少告诉模型五件事

```text
它来自哪个文档、哪个版本和哪个位置
它适用于哪个产品、角色、地区或时间
它的 ACL 和过滤状态是什么
它与问题的关系是什么，是否是直接证据
答案中应如何引用它
```

把标题路径、来源、更新时间和 chunk ID放进结构化字段，通常比单纯加一段“请认真回答”更有帮助。模型需要看到证据的边界，而不是只有一串没有出处的文字。

### 9.7.2 预算是证据选择问题

上下文预算不足时，不能随机截断。应按任务和证据优先级选择：直接支持关键 claim 的 chunk、高置信且最新的版本、必要的定义和例外、完成多跳所需的互补证据。重复 chunk、低置信候选和过时版本应该被压缩或排除。

设标准证据集合为 `E_i`，进入 context 的集合为 `Z_i`。在 `E_i` 非空时，证据覆盖率为：

$$
ContextRecall_i=\frac{\lvert Z_i\cap E_i\rvert}{\lvert E_i\rvert}
$$

如果标准证据存在但 `Z_i` 为空，覆盖率应记为 0；如果样本本来不需要外部证据，则是 `not_applicable`。上下文精确率要求 `Z_i` 非空：

$$
ContextPrecision_i=\frac{\lvert Z_i\cap E_i\rvert}{\lvert Z_i\rvert}
$$

`Z_i` 为空时，这个数学比例没有定义，但“答案需要证据而没有选入任何证据”仍可作为单独失败事件记录，不能用非零分母伪造精确率。

### 9.7.3 冲突证据必须显式建模

新旧政策、不同地区、不同套餐或不同角色的文档可能同时召回。模型如果没有看到版本和适用条件，可能把两条规则拼成一条不存在的规则。

context 构造应保留冲突关系：

```text
冲突文档 ID、版本、生效时间和适用范围
冲突字段或相互矛盾的 claim
选择当前版本的规则和无法判断时的处理
```

当冲突无法由系统确定时，正确行为可能是说明冲突并请求人工，而不是强行选一条答案。

## 9.8 证据进入 prompt 以后，模型仍然可能不用它

检索正确并不等于生成正确。模型可能使用参数记忆、忽略上下文中的条件、把表格读错，或把多个证据组合成未经支持的结论。

排查可以做最小对照：

```text
无 context，只给问题
只给标准证据，不给噪声
给完整 final context
交换证据顺序、版本和答案选项
要求逐 claim 引用，再检查引用支持关系
```

如果只给标准证据就能答对，加入噪声后答错，问题更可能在 context precision、排序或冲突组织；如果标准证据单独给出仍答错，可能是解析、格式、模型能力或任务定义问题。

Prompt 可以明确要求“只依据给定资料，资料不足时说明不足”，但 prompt 不能替代证据质量、权限和版本控制。把无法支持的答案改成拒答，需要评估器和产品状态共同配合。

## 9.9 Claim-level grounding：答案不是一个原子标签

“答案正确”太粗。一个答案可能有五个声明，其中四个有证据，一个是模型补全；如果只给答案级正确标签，就无法定位这一个未支持声明。

把答案拆成原子 claim：

```text
claim_id、文本、类型、重要性和风险
支持证据 ID、引用 span、是否完整支持
是否包含条件、数字、时间、主体和例外
若没有支持，是否应拒答或降级为不确定表述
```

设有 `M>0` 个已完成标注的 claim，`s_m=1` 表示引用证据确实支持第 `m` 个声明，则 citation/grounding 准确率为：

$$
A_{cite}=\frac{\sum_{m=1}^{M}s_m}{M}
$$

当没有 claim、claim 标注未完成或引用关系无法复核时，指标分别是 `not_applicable` 或 `unknown`。unsupported claim rate 只有在同一批 claim 的支持标签完整时才能写成 `1-A_cite`。

### 9.9.1 引用存在不等于引用支持

常见的伪引用包括：

1. 引用文档主题相关，但段落没有该结论；
2. 引用只支持声明的前半句，后半句是模型扩展；
3. 引用旧版本，当前规则已经改变；
4. 引用链接对用户不可访问；
5. 多个 claim 共用一个宽泛链接，无法定位证据；
6. 引用编号和正文 claim 对不上。

高风险场景应把引用定位到段落、表格单元、代码行或政策条款，而不是只返回整篇文档首页。引用校验可以使用规则、字符串/数值比对、NLI 或 LLM judge，但关键结果应有人审或程序验证。

## 9.10 权限必须在检索前生效

企业 RAG 最严重的错误不是答得不够好，而是把用户不该知道的资料送进了模型或返回给了用户。

### 9.10.1 权限过滤的正确位置

一个基本的安全顺序是：

```text
解析用户身份和租户 -> 读取权限快照和版本
-> 过滤可检索文档/chunk -> dense/sparse retrieval
-> rerank 和 context construction
-> 生成、引用返回和缓存复核
```

“先全库检索和生成，最后过滤引用”是不够的。无权限文档已经进入模型上下文，模型可能在没有显式引用的情况下泄露其中内容；日志、缓存和 trace 也可能保存敏感信息。

### 9.10.2 权限数据和缓存数据要一致

权限至少需要考虑租户、组织、角色、用户组、文档 ACL、段落 ACL、字段级脱敏和权限版本。缓存 key 要包含影响可见性的字段；权限变更应使相关索引和缓存失效。

如果 context 集合为 `Z_i`，其中用户 `u_i` 无权访问的证据数为 `L_i`，在总 context 证据数大于零时，权限泄露率为：

$$
R_{perm}=\frac{\sum_iL_i}{\sum_i\lvert Z_i\rvert}
$$

没有 context 时比率 `0/0` 是 `not_applicable`；权限审计未执行时是 `unknown`；存在一条越权证据时，即使最终答案没有引用它，也应记录为安全事件。

## 9.11 Freshness：文档更新是在线行为

RAG 的外部记忆只有在更新链路可靠时才有价值。文档版本、索引版本和缓存版本必须能关联起来。

### 9.11.1 更新传播链

一次文档更新至少经过：

```text
源文档更新 -> 采集发现 -> 解析 -> chunk -> embedding/倒排
-> 索引发布 -> cache 失效 -> 在线可见 -> 评估和监控更新
```

任何一步失败，都可能产生旧知识。更新延迟可以按 `online_visible_at - source_updated_at` 记录，并按文档重要性、租户和索引分片观察。

### 9.11.2 旧证据率的定义域

若最终 context 中有 `S` 个证据被判定为过期，所有已知 freshness 的 context 证据数为 `Z_known`，且 `|Z_known|>0`：

$$
R_{stale}=\frac{S}{\lvert Z_{known}\rvert}
$$

如果文档时间字段缺失，不能把它当作新鲜；状态应为 `unknown`。旧证据率为零也不能证明系统正确，可能只是没有配置过期规则或没有覆盖更新场景。

### 9.11.3 新旧版本冲突要有选择规则

可以按生效时间、文档状态、适用地区和产品版本选择当前证据；如果多个文档仍然冲突，系统应把冲突传给模型或人工流程。最危险的做法是让向量相似度决定政策版本，因为旧文档可能和 query 更相似。

## 9.12 多跳、查询改写和 Agentic RAG

很多企业问题需要组合多个文档，例如“升级企业版并开启 SSO 后，账单什么时候变化、谁能配置、退货政策按哪个版本执行”。单次 top-k 检索很可能只覆盖其中一部分。

### 9.12.1 多跳查询的状态

多跳流程需要保存：

```text
原始问题、子问题、每跳查询、候选证据、已确认 claim
未解决的实体、版本和条件、停止原因、总 token/延迟/成本
```

如果系统在每一跳都无限扩大搜索，成本和 prompt 会失控；如果过早停止，最终答案会缺一块。停止条件可以是所有必需 claim 都有支持、预算耗尽、证据冲突、权限不足或需要人工确认。

### 9.12.2 查询改写可能产生 query drift

改写器把用户问题变成更“标准”的问题，可能提升召回，也可能丢掉产品名、否定词、时间条件、权限范围和数字。原始 query 必须保留，改写结果要和原始意图做对照评估。

### 9.12.3 Agentic RAG 的额外风险

主动检索、工具调用和多轮搜索增加了动作和状态。除了证据指标，还要评估工具选择、查询预算、重复检索、权限传递、外部数据写入和失败恢复。Agentic RAG 不是自动更准确，而是把检索策略从固定 pipeline 变成了一个需要治理的控制循环。

## 9.13 资料不足时，拒答是一个可评估结果

RAG 系统不应把每个 query 都映射成一段肯定回答。没有证据、证据冲突、权限不足、文档过期或问题超出范围时，拒答、追问或转人工可能是正确行为。

### 9.13.1 两类样本要分开

设 `A` 是应该回答的样本，`U` 是证据不足、无权限或应该受限处理的样本。对 `U`，正确拒答率为：

$$
R_{correct\_abstain}=\frac{N_{U,abstain}}{\lvert U\rvert},\qquad \lvert U\rvert>0
$$

对 `A`，误拒率为：

$$
R_{false\_abstain}=\frac{N_{A,abstain}}{\lvert A\rvert},\qquad \lvert A\rvert>0
$$

没有 `U` 或 `A` 时对应指标是 `not_applicable`，标签不确定时是 `unknown`。只报“拒答率”无法判断系统是更安全还是更无用。

### 9.13.2 高质量拒答要说明下一步

拒答不应泄露被保护文档的存在，也不应输出大段空泛免责声明。可以说明资料不足、请求用户补充版本或权限、提供公开帮助路径、转人工或返回可访问的替代资料。拒答本身也需要延迟、成本、用户采纳和误拒回归。

## 9.14 RAG 评估必须分层

### 9.14.1 检索层

关注正确证据是否进入候选，以及是否在有限 top-k 内排在前面：Recall@K、MRR、nDCG、实体/版本/错误码切片、权限过滤后的召回和多跳覆盖。

### 9.14.2 上下文层

关注最终 prompt 是否包含足够证据、噪声比例、重复、冲突、预算和引用映射。Context Recall 和 Context Precision 只在标准证据与集合定义清楚时有意义；它们不是最终答案质量。

### 9.14.3 生成层

关注答案正确性、完整性、claim 支持、引用位置、拒答、事实一致性和条件保留。开放式质量可以用人工或校准后的 judge 辅助，但关键数字和协议优先使用程序验证。

### 9.14.4 系统层

关注权限泄露、旧证据、索引更新延迟、TTFT、成本、缓存、失败恢复和线上反馈。RAG 的离线回答分数上升，如果权限事件或旧版本率上升，产品结论仍然可能是负面的。

## 9.15 Error Attribution：把 bad case 归因到第一处分歧

一个 bad case 可以有多个表面症状，但修复通常要找第一处分歧。例如正确文档没入库，就不应先修改生成 prompt；权限过滤发生在生成后，就不应只修引用链接。

建议把每个案例记录为：

```text
问题和用户权限
标准答案、标准证据和适用版本
解析文本、chunk、retriever top-k、reranker 排序
最终 context、预算截断和冲突关系
模型答案、原子 claim、引用和支持判断
权限、freshness、延迟、成本、反馈和风险
第一处分歧、根因、修复、回归样本和负责人
```

根因可以分为 `ingestion`、`parsing`、`chunking`、`embedding`、`retrieval`、`rerank`、`context`、`generation`、`citation`、`permission`、`freshness`、`abstention` 和 `product`。一个案例可以有多个标签，但报告要区分主因和伴随问题。

## 9.16 最小可运行的 RAG 事故审计

下面的代码只使用标准库，构造六个案例：旧版本进入 context、一个正常的多证据问题、错误码召回失败、越权薪酬文档、超预算多跳问题，以及正确拒答的无资料问题。

### 9.16.1 数据结构和定义域工具

```python
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Iterable, Optional


@dataclass(frozen=True)
class Metric:
    value: Optional[float]
    status: str
    reason: str = ""


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def safe_mean(values: Iterable[float], *, empty_status="not_applicable"):
    values = list(values)
    if not values:
        return Metric(None, empty_status, "empty sample set")
    checked = [finite_number(value, "mean.value") for value in values]
    return Metric(sum(checked) / len(checked), "valid")


def safe_ratio(numerator, denominator, *, name="ratio", empty_status="not_applicable"):
    numerator = finite_number(numerator, f"{name}.numerator")
    denominator = finite_number(denominator, f"{name}.denominator")
    if numerator < 0 or denominator < 0:
        raise ValueError(f"{name} cannot be negative")
    if denominator == 0:
        if numerator == 0:
            return Metric(None, empty_status, f"{name} has no denominator")
        raise ValueError(f"{name} has positive numerator and zero denominator")
    return Metric(numerator / denominator, "valid")


def percentile(values, percentage):
    values = [finite_number(value, "percentile.value") for value in values]
    percentage = finite_number(percentage, "percentile.percentage")
    if not values:
        return Metric(None, "not_applicable", "empty sample set")
    if not 0 <= percentage <= 100:
        raise ValueError("percentage must be between 0 and 100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return Metric(value, "valid")


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
        "id": "return_policy_current", "role": "employee",
        "expected": ["return_v2"],
        "retrieved": ["return_v1", "return_v2", "login_generic"],
        "reranked": ["return_v1", "return_v2", "login_generic"],
        "context": ["return_v1", "login_generic"], "budget": 320,
        "claims": [{"citation": "return_v1", "support": "return_v2"}],
        "answerable": True, "model_abstains": False,
        "latency_ms": 980, "cost": 0.018, "online_delta": -0.20,
    },
    {
        "id": "sso_billing_admin", "role": "admin",
        "expected": ["sso_admin", "billing_enterprise"],
        "retrieved": ["sso_admin", "login_generic", "billing_enterprise"],
        "reranked": ["sso_admin", "billing_enterprise", "login_generic"],
        "context": ["sso_admin", "billing_enterprise"], "budget": 420,
        "claims": [
            {"citation": "sso_admin", "support": "sso_admin"},
            {"citation": "billing_enterprise", "support": "billing_enterprise"},
        ],
        "answerable": True, "model_abstains": False,
        "latency_ms": 1180, "cost": 0.023, "online_delta": 0.08,
    },
    {
        "id": "error_code_e1427", "role": "employee",
        "expected": ["error_e1427"],
        "retrieved": ["login_generic", "return_v1"],
        "reranked": ["login_generic", "return_v1"],
        "context": ["login_generic"], "budget": 250,
        "claims": [{"citation": "login_generic", "support": "error_e1427"}],
        "answerable": True, "model_abstains": False,
        "latency_ms": 920, "cost": 0.015, "online_delta": -0.15,
    },
    {
        "id": "private_comp_plan", "role": "employee", "expected": [],
        "retrieved": ["comp_private"], "reranked": ["comp_private"],
        "context": ["comp_private"], "budget": 220,
        "claims": [{"citation": "comp_private", "support": "comp_private"}],
        "answerable": False, "model_abstains": False,
        "latency_ms": 1200, "cost": 0.019, "online_delta": -0.45,
    },
    {
        "id": "upgrade_multi_hop", "role": "admin",
        "expected": ["sso_admin", "billing_enterprise", "return_v2"],
        "retrieved": ["sso_admin", "billing_enterprise", "return_v1"],
        "reranked": ["sso_admin", "billing_enterprise", "return_v1"],
        "context": ["sso_admin", "billing_enterprise", "return_v1"], "budget": 400,
        "claims": [
            {"citation": "sso_admin", "support": "sso_admin"},
            {"citation": "billing_enterprise", "support": "billing_enterprise"},
            {"citation": "return_v1", "support": "return_v2"},
        ],
        "answerable": True, "model_abstains": False,
        "latency_ms": 1850, "cost": 0.041, "online_delta": -0.10,
    },
    {
        "id": "no_policy_found", "role": "employee", "expected": [],
        "retrieved": [], "reranked": [], "context": [], "budget": 220,
        "claims": [], "answerable": False, "model_abstains": True,
        "latency_ms": 460, "cost": 0.009, "online_delta": 0.02,
    },
]


def validate_case(case):
    required = ("id", "role", "expected", "retrieved", "reranked", "context",
                "budget", "claims", "answerable", "model_abstains", "latency_ms",
                "cost", "online_delta")
    if any(key not in case for key in required):
        raise ValueError("case is missing a required field")
    if not case["id"] or not case["role"] or not isinstance(case["id"], str):
        raise ValueError("case id and role must be non-empty strings")
    if type(case["answerable"]) is not bool or type(case["model_abstains"]) is not bool:
        raise ValueError("answerable and model_abstains must be real bools")
    if case["answerable"] != bool(case["expected"]):
        raise ValueError("answerable must agree with the expected evidence set")
    if not isinstance(case["budget"], int) or isinstance(case["budget"], bool) or case["budget"] <= 0:
        raise ValueError("budget must be a positive integer")
    finite_number(case["latency_ms"], "latency_ms")
    finite_number(case["cost"], "cost")
    finite_number(case["online_delta"], "online_delta")
    for key in ("expected", "retrieved", "reranked", "context"):
        if len(set(case[key])) != len(case[key]):
            raise ValueError(f"{key} contains duplicate document ids")
        for doc_id in case[key]:
            if doc_id not in docs:
                raise ValueError(f"unknown document id: {doc_id}")
    for claim in case["claims"]:
        if claim["citation"] not in docs or claim["support"] not in docs:
            raise ValueError("claim references an unknown document")


if len({case["id"] for case in cases}) != len(cases):
    raise ValueError("case ids must be unique")
for case in cases:
    validate_case(case)
```

这里显式检查 `answerable` 与标准证据集合是否一致，避免“没有证据却被标成可回答”的数据错误。`private_comp_plan` 是一个故意的安全事故：它不应该被普通员工回答，但系统仍把私有文档放进 context；这与“没有任何资料”的 `no_policy_found` 是两种不同的不可回答状态。

### 9.16.2 计算检索、上下文、引用和安全指标

```python
def retrieval_recall(items, expected):
    if not expected:
        return Metric(None, "not_applicable", "no evidence is required")
    return safe_ratio(len(set(items) & set(expected)), len(expected), name="recall")


def first_rank(items, expected):
    if not expected:
        return Metric(None, "not_applicable", "no evidence is required")
    for index, item in enumerate(items, start=1):
        if item in expected:
            return Metric(1 / index, "valid")
    return Metric(0.0, "valid", "evidence was not retrieved")


def context_recall(context, expected):
    if not expected:
        return Metric(None, "not_applicable", "no evidence is required")
    return safe_ratio(len(set(context) & set(expected)), len(expected), name="context_recall")


def context_precision(context, expected):
    if not context:
        return Metric(None, "not_applicable", "no context was selected")
    return safe_ratio(len(set(context) & set(expected)), len(context), name="context_precision")


answerable = [case for case in cases if case["answerable"]]
unanswerable = [case for case in cases if not case["answerable"]]
retrieval_recalls = [retrieval_recall(c["retrieved"], c["expected"]).value for c in answerable]
context_recalls = [context_recall(c["context"], c["expected"]).value for c in answerable]
context_precision_metrics = [context_precision(c["context"], c["expected"]) for c in answerable]
context_precisions = [m.value for m in context_precision_metrics if m.status == "valid"]
mrr_scores = [first_rank(c["retrieved"], c["expected"]).value for c in answerable]

claim_results = []
for case in cases:
    for claim in case["claims"]:
        claim_results.append(
            claim["citation"] == claim["support"]
            and not docs[claim["citation"]]["stale"]
        )
claim_accuracy = safe_ratio(sum(claim_results), len(claim_results), name="citation_accuracy")

context_doc_count = sum(len(case["context"]) for case in cases)
permission_leaks = [
    (case["id"], doc_id)
    for case in cases
    for doc_id in case["context"]
    if case["role"] not in docs[doc_id]["acl"]
]
stale_context = [
    (case["id"], doc_id)
    for case in cases
    for doc_id in case["context"]
    if docs[doc_id]["stale"]
]
permission_rate = safe_ratio(len(permission_leaks), context_doc_count, name="permission_leak_rate")
stale_rate = safe_ratio(len(stale_context), context_doc_count, name="stale_evidence_rate")

correct_abstentions = sum(case["model_abstains"] for case in unanswerable)
abstention_rate = safe_ratio(
    correct_abstentions, len(unanswerable), name="correct_abstention_rate"
)
false_abstentions = sum(case["model_abstains"] for case in answerable)
false_abstention_rate = safe_ratio(
    false_abstentions, len(answerable), name="false_abstention_rate"
)

budget_overflows = [
    case["id"] for case in cases
    if sum(docs[doc_id]["tokens"] for doc_id in case["context"]) > case["budget"]
]
budget_overflow_rate = safe_ratio(len(budget_overflows), len(cases), name="budget_overflow_rate")
```

注意几个定义域：检索 Recall 只对需要外部证据的样本计算；context precision 对空 context 是 `not_applicable`，但 answerable 样本的 context recall 会是 0；权限和旧证据率使用实际进入 context 的证据数作为分母；正确拒答和误拒答分别使用不可回答与可回答集合。

### 9.16.3 归因、延迟、成本和线上反馈

```python
def causes_for(case):
    causes = []
    context = case["context"]
    if any(case["role"] not in docs[doc_id]["acl"] for doc_id in context):
        causes.append("permission")
    if any(docs[doc_id]["stale"] for doc_id in context):
        causes.append("freshness")
    if case["answerable"]:
        if retrieval_recall(case["retrieved"], case["expected"]).value < 1:
            causes.append("retrieval_miss")
        elif context_recall(case["context"], case["expected"]).value < 1:
            causes.append("context_drop")
    if any(
        claim["citation"] != claim["support"]
        or docs[claim["citation"]]["stale"]
        for claim in case["claims"]
    ):
        causes.append("citation_or_grounding")
    if not case["answerable"] and not case["model_abstains"]:
        causes.append("abstention_failure")
    if not causes:
        causes.append("pass")
    return causes


root_causes = {case["id"]: causes_for(case) for case in cases}
latency_p95 = percentile([case["latency_ms"] for case in cases], 95)
average_cost = safe_mean([case["cost"] for case in cases])
average_online_delta = safe_mean([case["online_delta"] for case in cases])

metrics = {
    "retrieval_recall": round(safe_mean(retrieval_recalls).value, 3),
    "mrr": round(safe_mean(mrr_scores).value, 3),
    "context_recall": round(safe_mean(context_recalls).value, 3),
    "context_precision": round(safe_mean(context_precisions).value, 3),
    "citation_accuracy": round(claim_accuracy.value, 3),
    "unsupported_claim_rate": round(1 - claim_accuracy.value, 3),
    "permission_leak_rate": round(permission_rate.value, 3),
    "stale_evidence_rate": round(stale_rate.value, 3),
    "correct_abstention_rate": round(abstention_rate.value, 3),
    "false_abstention_rate": round(false_abstention_rate.value, 3),
    "budget_overflow_rate": round(budget_overflow_rate.value, 3),
    "p95_latency_ms": round(latency_p95.value, 1),
    "average_cost": round(average_cost.value, 3),
    "average_online_delta": round(average_online_delta.value, 3),
}

failed_conditions = []
if metrics["retrieval_recall"] < 0.80 or metrics["mrr"] < 0.70:
    failed_conditions.append("retrieval")
if metrics["context_recall"] < 0.75 or metrics["context_precision"] < 0.60:
    failed_conditions.append("context")
if metrics["citation_accuracy"] < 0.80 or metrics["unsupported_claim_rate"] > 0.10:
    failed_conditions.append("citation_grounding")
if metrics["permission_leak_rate"] > 0:
    failed_conditions.append("permission")
if metrics["stale_evidence_rate"] > 0:
    failed_conditions.append("freshness")
if metrics["correct_abstention_rate"] < 0.90:
    failed_conditions.append("abstention")
if metrics["budget_overflow_rate"] > 0 or metrics["p95_latency_ms"] > 1500:
    failed_conditions.append("latency_or_budget")
if metrics["average_online_delta"] <= 0:
    failed_conditions.append("online_feedback")

release_decision = "repair_evidence_pipeline" if failed_conditions else "expand_evidence"
```

这里的 `failed_conditions` 是本例的诊断列表，不是某种通用标准。真正的业务阈值需要按照文档风险、用户任务、权限事件和人工成本校准。一个 `permission` 事件不能被平均准确率抵消；它要进入独立的安全修复路径。

### 9.16.4 输出和边界测试

```python
report = {
    "metrics": metrics,
    "permission_leaks": permission_leaks,
    "stale_context": stale_context,
    "budget_overflows": budget_overflows,
    "root_causes": root_causes,
    "failed_conditions": failed_conditions,
    "release_decision": release_decision,
}

for key, value in report.items():
    print(f"{key}= {value}")

assert safe_mean([]).status == "not_applicable"
assert safe_ratio(0, 0).status == "not_applicable"
assert percentile([], 95).status == "not_applicable"
assert root_causes["return_policy_current"] == [
    "freshness", "context_drop", "citation_or_grounding"
]
assert "permission" in root_causes["private_comp_plan"]
assert root_causes["no_policy_found"] == ["pass"]

try:
    safe_ratio(1, 0)
except ValueError:
    pass
else:
    raise AssertionError("positive numerator with zero denominator is invalid")

try:
    validate_case({**cases[0], "model_abstains": 1})
except ValueError:
    pass
else:
    raise AssertionError("bool fields must not accept integer lookalikes")

try:
    finite_number(float("nan"), "latency")
except ValueError:
    pass
else:
    raise AssertionError("NaN must not enter an evidence report")

assert release_decision == "repair_evidence_pipeline"
```

运行结果的关键部分应类似：

```text
metrics= {'retrieval_recall': 0.667, 'mrr': 0.625, 'context_recall': 0.417, 'context_precision': 0.417, 'citation_accuracy': 0.625, 'unsupported_claim_rate': 0.375, 'permission_leak_rate': 0.111, 'stale_evidence_rate': 0.222, 'correct_abstention_rate': 0.5, 'false_abstention_rate': 0.0, 'budget_overflow_rate': 0.167, 'p95_latency_ms': 1687.5, 'average_cost': 0.021, 'average_online_delta': -0.133}
permission_leaks= [('private_comp_plan', 'comp_private')]
stale_context= [('return_policy_current', 'return_v1'), ('upgrade_multi_hop', 'return_v1')]
budget_overflows= ['upgrade_multi_hop']
release_decision= repair_evidence_pipeline
```

实际运行时，浮点分位数和平均值应以代码输出为准；关键结论不是某个小数，而是根因被拆开了：`return_policy_current` 同时有旧证据和 context 丢失；`error_code_e1427` 是初始召回失败；`private_comp_plan` 是权限前置失败并且没有正确拒答；`upgrade_multi_hop` 缺少当前证据、使用旧版本并超预算；`no_policy_found` 正确拒答。

这个 demo 还故意保留了一个容易误解的现象：`return_v2` 在 retriever 列表里，但没有进入最终 context。只看 Recall@K 会以为检索正常，只看最终答案会把问题归因给生成；只有同时保存三层列表，才能知道是 rerank/context selection 的问题。

## 9.17 RAG 事故的修复顺序

遇到“答案错误、引用错误或权限异常”时，按证据链修复。

第一步，冻结原始 query、用户权限快照、文档版本、retrieved/reranked/context 列表、prompt 和模型输出。不要在现场被覆盖后再凭记忆复现。

第二步，确认原文、解析文本、chunk、metadata 和索引发布状态。若知识根本不存在于在线索引，先修采集和入库。

第三步，先做权限过滤和缓存隔离。任何越权证据都要阻断、告警和追溯，不能等待生成质量优化。

第四步，按 query 类型检查 dense/sparse/hybrid Recall 和 hard negative，再检查 reranker 是否把标准证据排入 final context。

第五步，修 context 的版本、冲突、预算、去重和引用定位，确保模型看到的是可解释证据而不是一堆相似文本。

第六步，对 claim 做支持校验，补充资料不足、冲突和过期场景的拒答/追问路径。

第七步，把每个根因和代表性案例加入回归集，并在离线检索、生成、权限、freshness、延迟和线上反馈上复测。

## 9.18 资料来源与证据边界

### 9.18.1 研究论文

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) 提出结合参数记忆与显式非参数记忆的 RAG 路线，并讨论 provenance 和知识更新问题。论文实验使用特定 Wikipedia 索引、模型和任务，不能直接代表企业文档、权限和当前产品效果。
- RAG 领域后续关于 REALM、DPR、FiD、Self-RAG、corrective/agentic retrieval 的论文可用于理解检索训练、证据使用和多步搜索，但每个方法的指标、数据和成本边界都不同。
- [Ragas available metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) 说明 context precision、context recall、faithfulness 等评估概念的实现入口。Ragas 是评估框架和指标实现，不是对任意业务答案的独立真值来源；LLM-based metric 仍需要人工校准。

### 9.18.2 官方工具和安全资料

- [OpenAI File Search](https://developers.openai.com/api/docs/guides/tools-file-search) 说明托管文件搜索工具的接口和使用方式。托管检索减少了部分基础设施工作，但业务仍需自行验证权限、文档版本、引用支持、删除传播和任务质量。
- 向量数据库、embedding 模型、reranker 和搜索引擎的官方文档适合确认 API、过滤字段和索引语义，不等于目标领域的召回质量。部署时应锁定版本，并使用带标签的 query-positive-negative 集合实测。
- [OWASP GenAI Security Project](https://genai.owasp.org/) 提供生成式 AI 的风险资料入口。权限、数据泄露、间接提示注入和向量/embedding 风险需要结合目标数据流和权限模型验证，不能用一个安全分数替代审计。

资料来源要分清“论文提出了什么”“工具实现了什么”“本地系统测到了什么”。尤其不能因为某个托管服务自动做了检索，就推断它已经满足企业权限、时效和合规要求。

### 9.18.3 本地实验和业务证据

本章 demo 是教学构造，不是向量库或模型的 benchmark。真实 RAG 结论需要：文档来源、解析快照、索引版本、权限快照、query 分布、人工证据标注、模型输出、线上反馈和回归结果。

如果评估集只包含最终答案而没有标准证据，检索 Recall 和 citation accuracy 不能被可靠计算；如果文档 ACL 或更新时间缺失，权限和 freshness 只能报告 `unknown`。证据不足时降低结论强度，比编造一个完整百分比更专业。

## 9.19 本章小结

RAG 不是把文档塞进 prompt，而是建立一条可以追溯的证据链。先保证原文采集和解析正确，再选择适合业务术语、错误码、数字和多语言的检索组合；用 reranker 和 context selection 管理有限预算；把版本、时间、标题路径、权限和引用 span 随证据一起传递。

答案进入生成阶段后，仍然要做 claim-level grounding。引用存在不等于引用支持，检索 Recall 高不等于最终 context 正确，答案碰巧正确也不等于用户有权限看到。文档更新、删除和权限变化要传播到索引和缓存；多跳检索要管理 query drift、预算和停止原因；资料不足时正确拒答是系统能力的一部分。

最后，所有 bad case 都要沿着“证据存在、解析、召回、排序、context、权限、版本、生成、引用、拒答”的顺序找第一处分歧。只有当这条账本能够回放，团队才知道应该修数据、检索、排序、权限、prompt、模型还是产品流程。

下一章进入 Agent 落地坑。RAG 主要解决“基于外部证据回答”，Agent 还要在这些证据基础上规划、调用工具、改变外部状态，并处理失败和恢复。
