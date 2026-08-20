# 第六章：RAG 产品落地

## 0. 本章范围与资料

本章回答一个产品化问题：怎样把一个“能从文档找答案”的 RAG demo，建设成可评估、可引用、可治理、可运营的知识产品。正文会沿着文档进入系统、查询经过检索、证据进入上下文、答案返回用户以及反馈回到数据治理的完整链路展开；每一环都要同时考虑质量、权限、时效、延迟和成本。

资料核验主要参考 OpenAI retrieval / file search / evals 相关官方文档、RAGAS 等 RAG 评估论文、OWASP LLM Top 10 中关于向量与嵌入、提示注入和敏感信息泄露的风险分类，以及可靠性工程中关于 SLI、SLO 和尾延迟的通用实践。官方接口文档只能支持公开 API 和产品行为；论文中的评估指标不能自动证明目标企业知识库有效；本章的分数、阈值和 Python demo 都是教学构造。

本章不替代后续专门的 RAG 算法实现、向量数据库选型、安全合规审查或第十七册 Agentic RAG 深入章节。这里聚焦产品落地：怎么把企业知识库从“几个 PDF 的 demo”升级成可评估、可引用、可治理、可运营、可控成本的 RAG 产品。

RAG 是企业大模型落地中最常见的技术路线之一。它看起来简单：把文档切块、做向量检索、把检索结果喂给模型生成答案。但真正产品化时，难点往往不在“能不能检索”，而在知识是否最新、引用是否可信、权限是否正确、答案是否可评估、用户是否愿意使用，以及失败后如何持续改进。

本章系统讲 RAG 产品落地：企业文档治理、知识更新、文档解析、分块、检索、引用可信、权限控制、无法回答、反馈闭环、效果评估、上线运营和常见失败模式。它与第十八册第 5 章的关系是：第 5 章建立企业应用的身份、流程和治理边界，本章深入 RAG 知识链路；第十七册第 6 章再讨论 Agent 如何主动检索、多跳取证和使用工具。

## 6.1 RAG 为什么适合企业落地

企业有大量内部知识：制度、流程、产品说明、技术文档、FAQ、合同模板、工单记录、会议纪要、历史案例。这些知识经常分散在不同系统中，员工查找成本高。

RAG 的价值是：

1. 让模型基于企业知识回答。
2. 降低幻觉风险。
3. 给出引用来源。
4. 支持知识更新。
5. 避免频繁微调模型。
6. 接入权限控制。

RAG 适合企业落地，是因为它把变化较快的知识放在模型参数之外管理。文档更新后可以重新解析和索引，权限变化可以通过元数据和后端策略传播，答案还可以携带来源与版本。它并不保证答案一定正确：如果文档本身错误、检索遗漏证据或模型错误地解释证据，RAG 仍然会失败。RAG 的价值在于让失败更容易定位、证据更容易复查，而不是把模型变成不会犯错的知识库。

## 6.2 RAG Demo 和 RAG 产品的区别

RAG demo：

1. 几个 PDF。
2. 简单切块。
3. 向量检索 top-k。
4. 生成答案。

RAG 产品：

1. 多来源文档接入。
2. 文档版本管理。
3. 权限过滤。
4. 增量更新。
5. 引用和可追溯。
6. 评估集。
7. 用户反馈。
8. 监控和运营。

demo 证明技术能跑通，产品要面对真实企业知识的混乱和变化。一个 demo 常常只测“给定几个 PDF 能否回答一个问题”；一个产品还要回答：文档谁负责、版本如何冲突、离职后的权限何时失效、引用能否被用户打开、无答案时是否拒答、索引更新失败如何发现，以及模型或检索器升级后质量是否回退。

## 6.3 企业文档治理

RAG 的上限很大程度由文档质量决定。

常见文档问题：

1. 过期文档未删除。
2. 多个版本互相冲突。
3. 标题不清楚。
4. 文档没有负责人。
5. 权限不明确。
6. 格式复杂。
7. 图片和表格难解析。
8. 内容重复。

文档治理要解决：

1. 来源可信。
2. 版本清晰。
3. 负责人明确。
4. 更新时间可见。
5. 权限可继承。
6. 元数据完整。

RAG 不是数据治理的替代品。脏知识库会让模型更自信地输出错误答案。文档进入索引前，最好先经过所有者确认、版本与生效日期检查、权限标签检查、敏感信息分类和重复/冲突检测。无法确认来源或权限的文档可以进入隔离队列，但不应因为“内部来源”就默认可以被所有员工检索。

文档治理还要处理“同一事实有多个版本”的问题。系统需要定义优先级：按生效日期、业务系统权威性、文档状态或人工指定的主版本选择证据；如果冲突无法自动判断，应把冲突显式展示或拒答。把两个版本拼在同一个 prompt 中，往往会让模型自行编造折中结论，反而降低可追溯性。

## 6.4 文档解析

文档解析决定进入 RAG 系统的原始文本质量。

要处理：

1. PDF。
2. Word。
3. PPT。
4. Excel。
5. HTML。
6. Markdown。
7. 图片和扫描件。
8. 表格和图表。

解析难点：

1. 页眉页脚噪声。
2. 表格结构丢失。
3. 多栏排版错乱。
4. OCR 错字。
5. 图片中的关键信息。
6. 章节层级丢失。

解析质量差，后面 embedding 和检索再强也难补救。解析器应保留页码、标题层级、表格行列、图片位置和原文链接等定位信息，并对 OCR 置信度、乱码、空页和异常字符做统计。解析后的文本最好能够回链到原文片段，方便人工核验；否则即使答案看起来正确，引用也可能无法复查。

不同格式需要不同的验收方法。制度和合同重视段落、条款编号和版本；表格重视行列关系、单位和合并单元格；扫描件重视 OCR 错误和版面顺序；代码和接口文档重视符号、版本和 fenced code。不要用一个“解析成功”布尔值覆盖所有格式的质量差异。

## 6.5 分块策略

分块影响召回和答案质量。

分块太小：上下文不完整。分块太大：召回不精确，token 成本高。

常见策略：

1. 按段落分块。
2. 按标题层级分块。
3. 按语义边界分块。
4. 滑动窗口重叠。
5. 表格单独处理。
6. 保留标题和路径元数据。

好的 chunk 不只是文本片段，还应该带有文档标题、章节、更新时间、权限、来源链接等元数据。分块的边界应服务于回答任务：定义、例外条款、步骤和表格行不能被随意切开；但也不能为了保持完整而把整本手册塞进一个 chunk。可以把 chunk 看成“可检索的证据单元”，它既要足够小以便召回，也要包含让人理解这段话所需的标题、版本和上下文。

重叠窗口不是免费的修复。重叠过大，会让同一事实重复进入 top-k，浪费上下文预算并夸大召回；重叠过小，又可能丢掉跨段条件。分块实验应固定评估集，比较召回、上下文精确率、引用支持、token、延迟和重复率，而不是只看某个向量相似度。

## 6.6 检索策略

企业 RAG 通常不只用向量检索。

常见组合：

1. 向量检索。
2. 关键词检索。
3. 混合检索。
4. 元数据过滤。
5. reranking。
6. 查询重写。
7. 多轮检索。

向量检索适合语义相似，关键词检索适合精确术语、产品型号、错误码和人名。企业系统通常需要混合检索。检索器的职责是提供候选证据，不是直接决定答案；重排器可以改善候选顺序，但无法找回根本没有召回的证据。查询改写有助于把口语问题变成文档术语，也可能改变用户原意，因此应保留原问题、改写结果和最终使用的查询，便于失败复盘。

检索策略还要区分“找相关内容”和“找足以支持结论的内容”。一个文档可能主题相关，却没有回答问题所需的条件、例外或时间范围。评估集应标注最小支持证据，而不是只标注一堆相似文档；否则检索指标很高，生成仍然可能没有足够依据。

## 6.7 权限控制

企业 RAG 必须做权限控制。

关键原则：

1. 用户只能检索有权限的文档。
2. 检索前或检索后必须过滤权限。
3. 引用不能暴露无权限来源。
4. 摘要不能泄露无权限内容。
5. 日志不能保存敏感内容。
6. 权限变更要及时生效。

一个严重错误是：模型不直接展示原文，但把无权限文档内容总结出来。这仍然是数据泄露。更稳妥的顺序是先解析用户身份和租户，再将 ACL 作为检索条件；无权文档不应进入候选集、重排列表、prompt、引用列表、缓存或普通日志。检索后的过滤仍可作为第二道防线，但不能把它当成唯一防线。

产品上更稳的做法是把权限当成检索条件，而不是生成后的展示条件。换句话说，无权文档不应该进入候选集、rerank 列表、prompt 上下文、引用列表和原始日志。权限变更后，索引、缓存和向量库 metadata 也要同步失效，否则用户离职、转岗或项目权限收回后仍可能通过旧缓存看到答案。

## 6.8 知识更新

企业知识经常变化。

需要处理：

1. 新文档加入。
2. 老文档删除。
3. 文档内容修改。
4. 权限变化。
5. 文档过期。
6. 索引重建。
7. 增量 embedding。

产品上要显示答案依据的时间和版本。对于时效性强的问题，如果文档过期，系统应提示不确定，而不是强行回答。更新流程应记录抓取、解析、embedding、索引发布和缓存失效的状态；只有索引发布成功并通过抽样检查，新的文档版本才应成为默认证据。删除和权限撤销也需要同样的传播记录，否则主数据库已经更新，旧向量或缓存仍可能继续回答。

## 6.9 引用可信

引用是 RAG 产品的信任基础。

好的引用应该：

1. 支持对应结论。
2. 指向具体段落。
3. 用户有权限查看。
4. 文档版本正确。
5. 来源可信。
6. 能打开原文。

坏引用会严重破坏信任。最常见问题是答案正确但引用错，或引用文档根本不支持结论。

引用可信要按 claim 检查，而不是按整段回答检查。一个回答可能有 5 个关键断言，其中 4 个有证据、1 个是模型补出来的；如果只看“回答整体还行”，这个 unsupported claim 就会漏掉。企业 RAG 最好把答案拆成 atomic claims，再检查每个 claim 是否被引用段落直接支持、是否使用最新版本、是否有权限展示。引用的“存在”与“支持”是两个不同指标：链接能打开，不代表段落足以证明结论；引用正确，也不代表答案中的其他断言没有证据缺口。

## 6.10 无法回答

RAG 产品必须能说“不知道”。

无法回答的情况：

1. 检索不到相关文档。
2. 文档互相矛盾。
3. 用户无权限查看答案来源。
4. 问题超出知识库范围。
5. 证据不足。
6. 文档过期。

好的无法回答体验：

1. 说明原因。
2. 给出已检索范围。
3. 建议补充信息。
4. 引导用户联系负责人。
5. 允许反馈缺失文档。

强行回答会短期显得智能，长期损害信任。拒答也要有明确的原因码，例如 `no_evidence`、`permission_blocked`、`conflicting_versions`、`stale_source` 和 `out_of_scope`。不同原因对应不同修复：没有证据可能需要补文档，权限阻断不能靠扩大召回解决，版本冲突需要数据负责人处理，超出范围则需要改进产品边界。用户看到的文字可以友好，但系统内部应保留结构化原因。

## 6.11 用户反馈闭环

RAG 产品需要反馈闭环。

反馈可以包括：

1. 答案有用或无用。
2. 引用是否正确。
3. 是否解决问题。
4. 用户期望答案。
5. 缺失文档。
6. 错误文档。
7. 转人工原因。

反馈要进入改进流程：更新文档、调整分块、优化检索、补充评估集、改进 prompt 或权限规则。反馈不能直接作为训练标签使用：用户点“有用”可能只代表答案读起来顺眼，点“无用”也可能是权限不足或文档缺失。应把反馈与问题、证据、版本、用户角色、是否采用、人工复核和最终业务结果关联，再决定是数据问题、检索问题、生成问题还是产品交互问题。

## 6.12 评估指标

RAG 产品评估包括：

1. 检索召回率。
2. rerank 命中率。
3. 答案正确率。
4. 引用准确率。
5. 证据支持率。
6. 无法回答准确率。
7. 用户满意度。
8. 自助解决率。
9. 延迟。
10. 单次成本。

不同阶段关注不同指标。早期先看检索和答案正确，产品化后还要看使用率、满意度和业务收益。

### 6.12.1 把 RAG 评估拆成可解释的指标

可以把一次 RAG 查询样本写成：

```math
q_i=(x_i,u_i,D_i,K_i,E_i,A_i,C_i,P_i,L_i,B_i)
```

其中 `x_i` 是用户问题，`u_i` 是用户身份和角色，`D_i` 是该用户有权访问的文档集合，`K_i` 是检索候选，`E_i` 是最终放入上下文的证据，`A_i` 是答案，`C_i` 是引用，`P_i` 是权限判定，`L_i` 是延迟，`B_i` 是业务结果。评估集还需要为问题标注最小支持证据集合 `G_i`，以及答案中需要核验的原子断言集合。

这里有一个容易被忽略的规则：指标没有定义域时，不能偷偷把结果写成 0。比如某个问题没有任何标注证据，检索召回率没有分母；某个回答没有产生断言，证据支持率也没有分母；一个系统从未执行权限检查，权限通过率更不能因为“没有发现越权”就写成 1。工程系统应把这些情况标成 `unknown` 或“不适用”，并在汇总时保留这个信息。

**1. 检索召回率**

检索层首先要问：应该找到的证据，有多少被召回到了候选集中？这里的“应该找到”不能靠模型临时判断，而要由评估集中的人工标注、专家复核或可重复的规则给出。

```math
V_{\mathrm{ret}}=\{i\mid |G_i|>0\}
```

```math
R_{\mathrm{ret}}=
\frac{\sum_{i\in V_{\mathrm{ret}}}|K_i\cap G_i|}
{\sum_{i\in V_{\mathrm{ret}}}|G_i|}
\quad
\text{当分母}>0
```

其中 `G_i` 是人工标注或专家确认的支持证据集合，`V_{\mathrm{ret}}` 只包含有明确支持证据的问题。分母为 0 时结果是 `unknown`，不是 0；没有标准证据的问题应进入“无证据/应拒答”子集单独评估。上式是微平均，能反映证据总量；先求每个问题的召回率再平均，则是宏平均。两者差异较大时，报告中应同时展示，避免少数证据很多的问题主导结论。

召回率低时，重排器和生成器通常无法凭空找回缺失证据。它首先指向解析、分块、查询改写、关键词覆盖、向量模型或过滤条件的问题，而不是 prompt 写得不够漂亮。

**2. 上下文精确率**

召回很多不等于最后上下文质量高。进入 prompt 的证据要尽量相关。设 `V_{\mathrm{ctx}}=\{i\mid |E_i|>0\}`，即确实有证据进入上下文的问题集合：

```math
P_{\mathrm{ctx}}=
\frac{\sum_{i\in V_{\mathrm{ctx}}}|E_i\cap G_i|}
{\sum_{i\in V_{\mathrm{ctx}}}|E_i|}
\quad
\text{当分母}>0
```

如果 `E_i` 为空，但问题确实有 `G_i`，这不是上下文精确率为 0，而是“没有上下文”；它应由检索召回、答案策略和拒答结果共同解释。分母为 0 时结果是 `unknown`。精确率过低意味着无关 chunk、重复 chunk、旧版本或错误租户内容正在消耗上下文预算，可能增加延迟和引用错配。

**3. 证据支持率**

生成层要检查答案中的关键断言是否被证据支持。答案要先拆成可以逐条核验的断言。例如“报销上限是 500 元，且必须在 30 天内提交”至少包含金额、期限和条件三个可独立检查的事实。

```math
N_{\mathrm{claim}}=\sum_i n_i^{\mathrm{claim}},
\qquad
S_{\mathrm{ev}}=
\frac{\sum_i n_i^{\mathrm{support}}}{N_{\mathrm{claim}}}
\quad
\text{当 }N_{\mathrm{claim}}>0
```

其中 `n_i^{\mathrm{claim}}` 是第 `i` 个回答的关键断言数，`n_i^{\mathrm{support}}` 是被上下文证据直接支持的断言数。没有断言的样本不应进入这个比率的分母；分母为 0 时是 `unknown`。证据支持率比“答案带了链接”更严格，因为链接存在不代表链接中的段落足以支持断言。支持关系还要检查时间、适用范围、否定条件和用户权限，不能只用语义相似度代替人工或规则核验。

**4. 引用准确率**

引用准确率检查引用是否真的支持对应 claim。引用需要拆成“引用是否支持”和“需要引用的断言是否都被引用”两个问题。设 `N_{\mathrm{cited}}` 是带引用的断言数，`N_{\mathrm{cite\_ok}}` 是其中同时满足内容支持、版本正确、权限正确和链接可追溯的断言数：

```math
 A_{\mathrm{cite}}=
\frac{N_{\mathrm{cite\_ok}}}{N_{\mathrm{cited}}}
\quad
\text{当 }N_{\mathrm{cited}}>0
```

引用覆盖率则是：

```math
C_{\mathrm{cite}}=
\frac{N_{\mathrm{cited}}}{N_{\mathrm{claim}}}
\quad
\text{当 }N_{\mathrm{claim}}>0
```

没有任何引用时，`A_{\mathrm{cite}}` 是 `unknown`，但如果业务要求每个事实都可追溯，`C_{\mathrm{cite}}` 应为 0。把两个指标合并成一个“引用分数”，会掩盖“只引用了一条且恰好正确”或“每条都引用但引用都不支持”的不同失败模式。

**5. 拒答召回率和误拒答率**

“拒答准确率”这个名称容易混淆。产品至少要分别记录两类错误。设 `z_i=1` 表示根据评估标注，样本应该拒答或先澄清，`\hat z_i=1` 表示系统实际拒答或澄清。应拒答样本被正确处理的比例是拒答召回率：

```math
R_{\mathrm{abs}}=
\frac{\sum_i\mathbf{1}[z_i=1\land\hat z_i=1]}
{\sum_i\mathbf{1}[z_i=1]}
\quad
\text{当应拒答样本数}>0
```

在本来可以回答的样本中却拒答的比例是误拒答率：

```math
F_{\mathrm{abs}}=
\frac{\sum_i\mathbf{1}[z_i=0\land\hat z_i=1]}
{\sum_i\mathbf{1}[z_i=0]}
\quad
\text{当可回答样本数}>0
```

分母为 0 时对应指标是 `unknown`。合同、医疗或财务场景可能宁愿多澄清，也不能漏掉高风险拒答；内部 FAQ 则可能更关注误拒答造成的使用流失。阈值必须由风险和业务共同决定，不能把所有场景压成一个固定比例。

**6. 权限过滤通过率**

权限指标的分母不是“请求数”这么简单，而是系统实际执行的权限检查次数。一次查询可能检查候选文档、重排结果、上下文、引用、缓存和日志多个位置。设 `N_{\mathrm{check}}` 是已记录的检查次数，`N_{\mathrm{unauth}}` 是越权进入候选、上下文、引用、缓存或日志的事件数：

```math
P_{\mathrm{perm}}=
\frac{N_{\mathrm{check}}-N_{\mathrm{unauth}}}{N_{\mathrm{check}}}
\quad
\text{当 }N_{\mathrm{check}}>0
```

只要发现一条真实越权事件，就必须单独升级处理，不能让大量正常请求把平均值冲回 0.999。没有权限检查记录时，结果是 `unknown`；“没有日志”不是“没有越权”。权限覆盖率和权限正确率也应分开，前者问“多少路径做了检查”，后者问“做过的检查是否正确”。

**7. 过期证据率**

知识更新后，旧证据仍然进入答案会损害信任。设被检查的证据项数为 `N_{\mathrm{evidence}}`，其中已经过期、被替换或版本不一致的项数为 `N_{\mathrm{stale}}`：

```math
R_{\mathrm{stale}}=
\frac{N_{\mathrm{stale}}}{N_{\mathrm{evidence}}}
\quad
\text{当 }N_{\mathrm{evidence}}>0
```

分母为 0 或证据没有版本信息时，结果是 `unknown`。时效性强的业务还要按文档类型、更新时间窗口和风险等级分层统计，因为一份过期的营销 FAQ 和一份过期的支付政策不应承担相同后果。

**8. 多指标上线状态**

上线判断不应把所有指标相乘成一个看似精确的布尔值。对每个指标保留三种状态：

1. `passed`：数据完整，且满足当前场景阈值。
2. `failed`：数据完整，但未满足阈值，或发生了必须阻断的安全事件。
3. `unknown`：缺少分母、字段、版本或可靠证据，无法作出判断。

整体状态可以写成：

```math
\mathrm{status}=
\begin{cases}
\mathrm{failed},&\exists j:s_j=\mathrm{failed}\\
\mathrm{unknown},&\text{不存在 failed 且 }\exists j:s_j=\mathrm{unknown}\\
\mathrm{passed},&\text{所有必要指标都是 passed}
\end{cases}
```

其中 `s_j` 是第 `j` 个必要指标的状态。阈值 `r_0,p_0,s_0,c_0` 只是某个业务版本的验收参数，必须随风险、用户角色、文档时效和成本目标记录，不能写成 RAG 的普适常数。P95 延迟、单位成本、反馈闭环、业务指标和人工抽检也应进入必要指标集合。

### 6.12.2 为什么不能只看一个综合分

综合分适合做排查排序，不适合替代发布决策。若确实需要一个排序分，可以对有观测值的指标做加权平均：

```math
Q=
\frac{\sum_{j\in V}w_jm_j}{\sum_{j\in V}w_j},
\qquad
C_{\mathrm{score}}=
\frac{\sum_{j\in V}w_j}{\sum_jw_j}
```

其中 `V` 是有定义域且完成核验的指标集合，`m_j` 是归一化后的指标值，`w_j` 是业务权重，`C_{\mathrm{score}}` 是分数覆盖度。未知指标不能以 0 代入；否则一个只测了两项的 demo 可能因为“没有失败数据”而得到虚假的高分。

即使 `Q` 很高，只要权限状态是 `failed`，整体仍应是 `failed`；如果权限状态是 `unknown`，整体最多是 `unknown`。报告中应同时展示综合分、覆盖度、各层指标和失败原因，方便工程团队知道下一步该修文档、检索器、生成器还是治理流程。

## 6.13 评估集构建

RAG 评估需要问题集。

问题来源：

1. 历史搜索日志。
2. 客服工单。
3. 员工常见问题。
4. 业务专家整理。
5. 失败案例。
6. 新文档发布后的测试问题。

每个问题最好标注：

1. 标准答案。
2. 支持文档。
3. 关键段落。
4. 用户角色和权限。
5. 是否应拒答。

没有评估集，RAG 优化容易凭感觉。

## 6.14 上线运营

RAG 上线后要运营。

运营事项：

1. 监控热门问题。
2. 发现无答案问题。
3. 清理过期文档。
4. 维护文档负责人。
5. 分析低满意度回答。
6. 更新评估集。
7. 监控成本和延迟。
8. 处理权限异常。

RAG 产品不是一次建设，而是持续运营的知识系统。

## 6.15 常见失败模式

1. 文档质量差却怪模型。
2. 只用向量检索，忽略关键词和元数据。
3. 权限过滤不完整。
4. 引用不支持答案。
5. 文档更新后索引不同步。
6. 无法回答策略缺失。
7. 没有评估集。
8. 用户反馈没有进入改进流程。
9. 答案看起来流畅但没有证据。
10. 只做技术链路，不做知识运营。

RAG 落地的本质，是技术系统和知识管理系统一起建设。

## 6.16 从 demo 到产品：一次 RAG 方案评审

假设一个客服团队已经有了“上传 PDF 后可以提问”的内部 demo，现在希望把它接入全体员工。评审不能只问“top-k 取多少”或“换哪个 embedding 模型”，而应沿着责任链提出一组连续问题。

首先问知识是否可用：每份文档的所有者是谁，版本和生效日期在哪里，冲突版本如何处理，扫描件和表格是否经过抽样验收，删除和权限撤销怎样传播到索引与缓存。如果这些问题没有答案，继续调检索器只是在不稳定的数据上优化。

其次问证据是否足够：评估集中的问题是否来自真实工单和搜索日志，是否标注最小支持段落，是否包含无答案、过期、冲突和越权样本。一个只包含“文档中明确写着答案”的测试集，会高估系统能力；真正的用户问题常常有省略、错别字、时间范围和隐含条件。

再次问答案是否可追溯：回答中的每个关键断言是否能链接到具体版本的段落，引用是否和用户权限一致，模型是否会把多份冲突文档拼成一个折中结论。引用存在不等于引用正确，答案流畅也不等于证据充分。

最后问失败是否可运营：系统能否区分没有证据、没有权限、文档过期、版本冲突和超出范围，能否让用户反馈缺失文档，能否把失败样本分派给文档负责人、检索工程师或产品负责人。没有原因码和责任归属，线上反馈很快会变成一堆无法执行的“答案不好”。

因此，一个可接受的 RAG 产品方案通常同时提交四类材料：文档和权限治理说明、离线评估集及分层结果、线上延迟/成本/安全监控方案、失败后的修复流程。检索器只是其中一个组件，不能替代这四类产品责任。

## 6.17 如何设计一套可信的 RAG 评估

评估要从问题分层开始，而不是先选择一个综合分。至少应把问题分成有明确证据可回答、需要澄清、应该拒答、存在版本冲突、用户无权访问和超出知识库范围几类。每一类都要记录问题、用户角色、知识库版本、最小支持证据、期望行为和风险等级。

评估时先看检索层：候选集是否召回支持证据，最终上下文是否包含过多重复或无关内容。再看生成层：关键断言是否被证据支持，引用是否正确并覆盖需要追溯的断言，应该拒答的问题是否真的拒答，可回答的问题是否被过度拒答。最后看产品层：用户是否采纳答案、是否重复提问或转人工、延迟和成本是否落在目标内，权限和版本审计是否完整。

评估结果要按版本和分层保存。检索器升级可能提高总体召回，却伤害某个关键部门的术语；模型升级可能提高答案流畅度，却降低拒答召回；文档更新可能提高新版本问题的准确率，却让旧版本引用残留。只报告一个总体平均值，会把这些回归隐藏起来。

线上反馈不能直接等价为正确标签。“有用”可能只表示读起来顺，“无用”可能是用户没有权限、问题超范围或文档缺失。更可靠的闭环是抽取失败样本，补充人工判断，再把判断结果归因到解析、治理、检索、生成、权限或交互层。这样评估才会变成工程行动，而不是季度汇报中的一个分数。

## 6.18 最小可运行 RAG 产品审计 demo

下面这个 demo 用 0 依赖 Python 模拟 RAG 产品上线审计。它把检索召回、上下文精确、证据支持、引用准确、拒答、权限过滤、数据新鲜度、P95 延迟、单位成本、评估集、反馈闭环和业务指标放进同一张表。

```python
import math


UNKNOWN = "unknown"
PASSED = "passed"
FAILED = "failed"


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def ratio(numerator, denominator):
    if (
        not finite_number(numerator)
        or not finite_number(denominator)
        or denominator <= 0
        or numerator < 0
        or numerator > denominator
    ):
        return UNKNOWN
    return numerator / denominator


def minimum_status(value, threshold):
    if value == UNKNOWN:
        return UNKNOWN
    return PASSED if value >= threshold else FAILED


def maximum_status(value, threshold):
    if value == UNKNOWN:
        return UNKNOWN
    return PASSED if value <= threshold else FAILED


def overall_status(statuses):
    if FAILED in statuses:
        return FAILED
    if UNKNOWN in statuses:
        return UNKNOWN
    return PASSED


def invalid_product_fields(product):
    count_fields = [
        "retrieved_relevant",
        "relevant_total",
        "context_relevant",
        "context_total",
        "supported_claims",
        "total_claims",
        "citation_supported",
        "citation_total",
        "correct_abstentions",
        "expected_abstentions",
        "answerable_total",
        "incorrect_abstentions",
        "permission_checks",
        "unauthorized_events",
        "stale_evidence",
        "evidence_total",
    ]
    invalid = [
        field
        for field in count_fields
        if (
            not isinstance(product[field], int)
            or isinstance(product[field], bool)
            or product[field] < 0
        )
    ]
    for field in [
        "permission_coverage",
        "eval_ready",
        "feedback_loop",
        "business_metric_defined",
    ]:
        value = product[field]
        if not finite_number(value) or not 0 <= value <= 1:
            invalid.append(field)
    for field in ["p95_latency_ms", "latency_slo_ms", "cost_per_answer", "cost_slo"]:
        value = product[field]
        if not finite_number(value) or value < 0:
            invalid.append(field)
    return invalid


def audit_product(product):
    if not isinstance(product, dict):
        return {"name": "<invalid>", "status": UNKNOWN, "errors": ["not_a_mapping"]}

    name = product.get("name", "<unnamed>")
    required = [
        "name",
        "retrieved_relevant",
        "relevant_total",
        "context_relevant",
        "context_total",
        "supported_claims",
        "total_claims",
        "citation_supported",
        "citation_total",
        "correct_abstentions",
        "expected_abstentions",
        "answerable_total",
        "incorrect_abstentions",
        "permission_checks",
        "unauthorized_events",
        "permission_coverage",
        "stale_evidence",
        "evidence_total",
        "p95_latency_ms",
        "latency_slo_ms",
        "cost_per_answer",
        "cost_slo",
        "eval_ready",
        "feedback_loop",
        "business_metric_defined",
    ]
    missing = [field for field in required if field not in product]
    if missing:
        return {"name": name, "status": UNKNOWN, "errors": ["missing:" + field for field in missing]}

    invalid = invalid_product_fields(product)
    if invalid:
        return {
            "name": name,
            "status": UNKNOWN,
            "errors": ["invalid:" + field for field in invalid],
        }

    permission_rate = (
        UNKNOWN
        if product["permission_checks"] == 0
        or product["unauthorized_events"] > product["permission_checks"]
        else ratio(
            product["permission_checks"] - product["unauthorized_events"],
            product["permission_checks"],
        )
    )
    metrics = {
        "retrieval_recall": ratio(
            product["retrieved_relevant"], product["relevant_total"]
        ),
        "context_precision": ratio(
            product["context_relevant"], product["context_total"]
        ),
        "evidence_support": ratio(
            product["supported_claims"], product["total_claims"]
        ),
        "citation_accuracy": ratio(
            product["citation_supported"], product["citation_total"]
        ),
        "citation_coverage": ratio(
            product["citation_total"], product["total_claims"]
        ),
        "abstention_recall": ratio(
            product["correct_abstentions"], product["expected_abstentions"]
        ),
        "false_abstention_rate": ratio(
            product["incorrect_abstentions"], product["answerable_total"]
        ),
        "permission_filter_rate": permission_rate,
        "permission_coverage": product["permission_coverage"],
        "stale_evidence_rate": ratio(
            product["stale_evidence"], product["evidence_total"]
        ),
        "freshness": UNKNOWN,
        "eval_ready": product["eval_ready"],
        "feedback_loop": product["feedback_loop"],
        "business_metric_defined": product["business_metric_defined"],
    }
    metrics["freshness"] = (
        UNKNOWN
        if metrics["stale_evidence_rate"] == UNKNOWN
        else 1.0 - metrics["stale_evidence_rate"]
    )
    metric_status = {
        "retrieval_recall": minimum_status(metrics["retrieval_recall"], 0.80),
        "context_precision": minimum_status(metrics["context_precision"], 0.65),
        "evidence_support": minimum_status(metrics["evidence_support"], 0.85),
        "citation_accuracy": minimum_status(metrics["citation_accuracy"], 0.85),
        "citation_coverage": minimum_status(metrics["citation_coverage"], 0.80),
        "abstention_recall": minimum_status(metrics["abstention_recall"], 0.80),
        "false_abstention_rate": maximum_status(
            metrics["false_abstention_rate"], 0.20
        ),
        "permission_filter": (
            FAILED
            if product["unauthorized_events"] > 0
            else overall_status(
                [
                    minimum_status(metrics["permission_filter_rate"], 0.98),
                    minimum_status(metrics["permission_coverage"], 0.98),
                ]
            )
        ),
        "freshness": maximum_status(metrics["stale_evidence_rate"], 0.10),
        "p95_latency": maximum_status(
            product["p95_latency_ms"], product["latency_slo_ms"]
        ),
        "unit_cost": maximum_status(
            product["cost_per_answer"], product["cost_slo"]
        ),
        "eval_ready": minimum_status(metrics["eval_ready"], 0.80),
        "feedback_loop": minimum_status(metrics["feedback_loop"], 0.75),
        "business_metric": minimum_status(
            metrics["business_metric_defined"], 1.0
        ),
    }
    score_weights = {
        "retrieval_recall": 0.14,
        "context_precision": 0.10,
        "evidence_support": 0.16,
        "citation_accuracy": 0.12,
        "abstention_recall": 0.08,
        "permission_filter_rate": 0.12,
        "freshness": 0.06,
        "eval_ready": 0.07,
        "feedback_loop": 0.07,
        "business_metric_defined": 0.08,
    }
    observed_weight = sum(
        weight for name, weight in score_weights.items() if metrics[name] != UNKNOWN
    )
    score = (
        UNKNOWN
        if observed_weight == 0
        else sum(
            score_weights[name] * metrics[name]
            for name in score_weights
            if metrics[name] != UNKNOWN
        )
        / observed_weight
    )
    status = overall_status(list(metric_status.values()))
    return {
        "name": name,
        "status": status,
        "rag_score": UNKNOWN if score == UNKNOWN else round(score, 3),
        "score_coverage": round(observed_weight / sum(score_weights.values()), 3),
        "metrics": {
            key: value if value == UNKNOWN else round(value, 3)
            for key, value in metrics.items()
        },
        "metric_status": metric_status,
        "failed_gates": [
            key for key, value in metric_status.items() if value == FAILED
        ],
        "unknown_gates": [
            key for key, value in metric_status.items() if value == UNKNOWN
        ],
    }


products = [
    {
        "name": "support_policy_rag",
        "retrieved_relevant": 42,
        "relevant_total": 48,
        "context_relevant": 31,
        "context_total": 42,
        "supported_claims": 66,
        "total_claims": 72,
        "citation_supported": 61,
        "citation_total": 68,
        "correct_abstentions": 9,
        "expected_abstentions": 10,
        "answerable_total": 62,
        "incorrect_abstentions": 4,
        "permission_checks": 100,
        "unauthorized_events": 0,
        "permission_coverage": 0.99,
        "stale_evidence": 6,
        "evidence_total": 100,
        "p95_latency_ms": 1800,
        "latency_slo_ms": 2200,
        "cost_per_answer": 0.043,
        "cost_slo": 0.08,
        "eval_ready": 0.86,
        "feedback_loop": 0.80,
        "business_metric_defined": 1.0,
    },
    {
        "name": "legal_contract_rag",
        "retrieved_relevant": 34,
        "relevant_total": 45,
        "context_relevant": 28,
        "context_total": 44,
        "supported_claims": 58,
        "total_claims": 70,
        "citation_supported": 49,
        "citation_total": 64,
        "correct_abstentions": 8,
        "expected_abstentions": 12,
        "answerable_total": 58,
        "incorrect_abstentions": 11,
        "permission_checks": 100,
        "unauthorized_events": 0,
        "permission_coverage": 0.99,
        "stale_evidence": 16,
        "evidence_total": 100,
        "p95_latency_ms": 2600,
        "latency_slo_ms": 2500,
        "cost_per_answer": 0.091,
        "cost_slo": 0.10,
        "eval_ready": 0.82,
        "feedback_loop": 0.62,
        "business_metric_defined": 1.0,
    },
    {
        "name": "codebase_rag",
        "retrieved_relevant": 28,
        "relevant_total": 36,
        "context_relevant": 18,
        "context_total": 35,
        "supported_claims": 42,
        "total_claims": 54,
        "citation_supported": 37,
        "citation_total": 50,
        "correct_abstentions": 6,
        "expected_abstentions": 8,
        "answerable_total": 46,
        "incorrect_abstentions": 13,
        "permission_checks": 100,
        "unauthorized_events": 1,
        "permission_coverage": 0.93,
        "stale_evidence": 8,
        "evidence_total": 100,
        "p95_latency_ms": 1900,
        "latency_slo_ms": 1800,
        "cost_per_answer": 0.052,
        "cost_slo": 0.08,
        "eval_ready": 0.70,
        "feedback_loop": 0.68,
        "business_metric_defined": 1.0,
    },
    {
        "name": "generic_doc_chat",
        "retrieved_relevant": 18,
        "relevant_total": 44,
        "context_relevant": 14,
        "context_total": 48,
        "supported_claims": 30,
        "total_claims": 68,
        "citation_supported": 19,
        "citation_total": 55,
        "correct_abstentions": 2,
        "expected_abstentions": 11,
        "answerable_total": 44,
        "incorrect_abstentions": 22,
        "permission_checks": 100,
        "unauthorized_events": 3,
        "permission_coverage": 0.72,
        "stale_evidence": 31,
        "evidence_total": 100,
        "p95_latency_ms": 3300,
        "latency_slo_ms": 2200,
        "cost_per_answer": 0.115,
        "cost_slo": 0.08,
        "eval_ready": 0.35,
        "feedback_loop": 0.20,
        "business_metric_defined": 0.0,
    },
]

results = [audit_product(product) for product in products]
ranked = sorted(
    [(r["name"], r["rag_score"], r["status"]) for r in results],
    key=lambda item: item[1] if item[1] != UNKNOWN else float("-inf"),
    reverse=True,
)
rag_pass = [r["name"] for r in results if r["status"] == PASSED]
needs_rework = {
    r["name"]: {
        "failed": r["failed_gates"],
        "unknown": r["unknown_gates"],
    }
    for r in results
    if r["status"] != PASSED
}

print("ranked=", ranked)
print("rag_pass=", rag_pass)
print("sample_metrics=", results[0]["metrics"])
print("needs_rework=", needs_rework)


empty_product = audit_product({})
nan_product = dict(products[0])
nan_product["feedback_loop"] = float("nan")
zero_evidence = dict(products[0])
zero_evidence["evidence_total"] = 0
unauthorized_product = dict(products[0])
unauthorized_product["unauthorized_events"] = 1
assert empty_product["status"] == UNKNOWN
assert audit_product(nan_product)["status"] == UNKNOWN
assert audit_product(zero_evidence)["status"] == UNKNOWN
assert audit_product(unauthorized_product)["status"] == FAILED
assert all(result["status"] in {PASSED, FAILED} for result in results)
print("boundary_status=", {
    "empty": empty_product["status"],
    "nan": audit_product(nan_product)["status"],
    "zero_evidence": audit_product(zero_evidence)["status"],
    "unauthorized": audit_product(unauthorized_product)["status"],
})
```

一组典型输出是：

```text
ranked= [('support_policy_rag', 0.895, 'passed'), ('legal_contract_rag', 0.798, 'failed'), ('codebase_rag', 0.784, 'failed'), ('generic_doc_chat', 0.409, 'failed')]
rag_pass= ['support_policy_rag']
sample_metrics= {'retrieval_recall': 0.875, 'context_precision': 0.738, 'evidence_support': 0.917, 'citation_accuracy': 0.897, 'citation_coverage': 0.944, 'abstention_recall': 0.9, 'false_abstention_rate': 0.065, 'permission_filter_rate': 1.0, 'permission_coverage': 0.99, 'stale_evidence_rate': 0.06, 'freshness': 0.94, 'eval_ready': 0.86, 'feedback_loop': 0.8, 'business_metric_defined': 1.0}
needs_rework= {'legal_contract_rag': {'failed': ['retrieval_recall', 'context_precision', 'evidence_support', 'citation_accuracy', 'abstention_recall', 'freshness', 'p95_latency', 'feedback_loop'], 'unknown': []}, 'codebase_rag': {'failed': ['retrieval_recall', 'context_precision', 'evidence_support', 'citation_accuracy', 'abstention_recall', 'false_abstention_rate', 'permission_filter', 'p95_latency', 'eval_ready', 'feedback_loop'], 'unknown': []}, 'generic_doc_chat': {'failed': ['retrieval_recall', 'context_precision', 'evidence_support', 'citation_accuracy', 'abstention_recall', 'false_abstention_rate', 'permission_filter', 'freshness', 'p95_latency', 'unit_cost', 'eval_ready', 'feedback_loop', 'business_metric'], 'unknown': []}}
boundary_status= {'empty': 'unknown', 'nan': 'unknown', 'zero_evidence': 'unknown', 'unauthorized': 'failed'}
```

这个 demo 的重点是：RAG 产品不能只看“回答像不像对”。`legal_contract_rag` 有一定质量，但合同场景中证据过期、引用不足、拒答不足和反馈闭环缺失会影响责任边界；`codebase_rag` 有真实越权事件，即使综合分不低也必须失败；`generic_doc_chat` 则说明“文档聊天”如果没有评估、权限、引用、拒答和业务指标，很难算企业级 RAG 产品。空输入、非有限值、没有证据版本和越权事件分别展示了 `unknown` 与 `failed` 的不同含义。

## 6.19 资料入口与证据边界

- [OpenAI Retrieval 指南](https://developers.openai.com/api/docs/guides/retrieval)：官方开发者文档，可支持文件进入检索、搜索结果进入生成上下文等产品链路的说明；它不证明某个企业知识库的召回率或答案正确率。
- [OpenAI Evals 指南](https://developers.openai.com/api/docs/guides/evals)：官方评估入口，可支持把任务、样本、指标和回归评估纳入产品流程的讨论；本章的阈值和示例数据不是该文档给出的标准。
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)：论文，可支持把检索上下文和生成答案拆开评估的研究背景；论文指标的适用性仍取决于任务、标注和评估器。
- [Ragas 指标文档](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)：项目文档，可作为 context precision、faithfulness 等指标的实现参考；项目定义不等于企业业务验收规范。
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)：应用安全项目，可支持提示注入、敏感信息泄露和向量/嵌入弱点等风险分类；它不是某个 RAG 系统已经安全的证明。

上述资料能支持 RAG 的通用机制、评估思路和风险类别；本章的文档字段、阈值、状态汇总、场景参数、综合分和 Python demo 都是教学构造。真实项目还需使用自己的问题分层、权限矩阵、版本记录、人工复核、账单和线上 SLO。`unknown` 表示证据不足，不能自动改写成通过；`failed` 表示已有数据证明条件未满足或发生了阻断性事件。

## 6.20 本章小结

RAG 产品落地的关键，是让模型基于正确、最新、有权限、可追溯的知识回答问题。技术链路包括文档解析、分块、embedding、检索、rerank、生成和引用；产品链路还包括文档治理、权限、更新、评估、反馈和运营。

下一章会进入 Agent 产品落地，讨论当产品不只是回答问题，而是要执行任务、调用工具、操作系统时，如何设计体验、权限、成本和安全边界。
