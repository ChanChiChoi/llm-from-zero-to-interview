# 第四十五章：Retrieval-Augmented Architecture：把外部检索变成架构组件

## 45.1 RAG 不只是 prompt 拼接

最简单的 RAG 是“检索文档，拼进 prompt，调用 LLM”。当系统规模、上下文和工具复杂起来，检索会影响模型输入长度、证据路径、缓存、权限、引用和成本，因此它可以被视作模型架构之外的一个可组合组件。

把 retrieval 作为 architecture component，意味着明确三件事：模型何时请求外部 memory；检索返回什么形式的表示；模型如何验证、引用和更新检索结果。检索器不是一个黑盒外挂，错误召回会直接改变模型的条件分布。

## 45.2 外部记忆和条件生成

给定问题 q，检索器从语料库 D 选出 top-k 文档：

~~~math
R(q,D)=\operatorname{TopK}_{d\in D}s(q,d)
~~~

生成模型条件变为：

~~~math
P(y\mid q,D)\approx P_\theta(y\mid q,R(q,D))
~~~

若有 reranker、证据压缩或迭代检索，可以写成多阶段：

~~~math
R_1\rightarrow \operatorname{Rank}\rightarrow \operatorname{Compress}
\rightarrow R_2\rightarrow \operatorname{Generate}
~~~

每个阶段都有召回损失、延迟和 token 成本。RAG 能扩大可用知识，不代表生成模型一定正确使用了返回内容。

## 45.3 小白直觉：图书馆不是大脑

模型像一个会推理的读者，向量索引像图书馆目录。目录找不到书，读者再聪明也无法引用；目录给了十本相似但错误的书，读者可能被干扰；书找到了，读者仍要判断哪段支持答案。

所以要分别评估：召回是否包含证据、排序是否把证据放前面、上下文是否过长、模型是否正确引用，以及回答是否符合权限。

## 45.4 架构化检索的接口

一个结构化 retrieval result 不应只是文本字符串，至少可以包含：

~~~text
document_id
chunk_id
text
score
source_revision
access_scope
timestamp
provenance
~~~

模型输入可以把这些字段序列化，系统也可以在模型外部保留元数据。若只把 text 放入 prompt，后续引用、权限和版本冲突很难恢复。

## 45.5 检索与 attention 的分工

外部检索在大语料中做粗筛，attention 在候选上下文内做精细对齐。检索器的复杂度与索引规模相关，模型 attention 的复杂度与候选 token 数相关。可以通过降低候选集合，把昂贵的 global attention 限制在高价值文本上。

但检索器召回错误会造成硬门槛：模型无法从未返回的文档中生成可信证据。系统要支持 query rewrite、multi-hop、fallback、直接搜索和“证据不足”拒答。

## 45.6 一个最小检索审计

~~~python
def retrieval_gate(results, required_ids, top_k):
    selected = {item["id"] for item in results[:top_k]}
    recall = len(selected & set(required_ids)) / max(len(required_ids), 1)
    return {
        "top_k": top_k,
        "evidence_recall": recall,
        "has_all_required": set(required_ids) <= selected,
    }


docs = [{"id": "a"}, {"id": "wrong"}, {"id": "b"}]
print(retrieval_gate(docs, ["a", "b"], top_k=3))
~~~

这段代码只测 ID recall。真实系统还要测段落级支持、版本、权限、时间条件、重复片段、reranker 延迟和生成引用准确率。

## 45.7 何时把检索放入模型内部

把 retrieval 作为模型层/模块的一部分，可能采用可学习 query、memory cross-attention、检索 token、外部 key-value memory 或工具调用。优点是模型可以学习何时查、查什么；缺点是训练数据、索引更新、可复现性、权限和 serving 复杂度更高。

端到端联合训练可能让检索器适应生成目标，但也可能学会依赖训练语料中的 shortcut。生产系统需要可替换的索引和独立的检索回归集。

## 45.8 多跳和迭代检索

一个问题可能需要先找到实体定义，再找到相关版本，再比较条款。可以把每轮状态写成：

~~~math
q_{t+1}=U(q_t,e_t,y_t),\qquad e_t=R(q_t,D)
~~~

循环次数提高召回机会，却增加延迟、成本和错误累积。每轮都要有终止条件、证据去重、最大预算和失败回退。模型不能无限检索后再输出一个没有来源的答案。

## 45.9 训练和 serving trade-off

检索 query embedding、ANN search、rerank、文档压缩和 tokenization 会占用 CPU/GPU；长文档再送进模型会提高 TTFT 和 KV。缓存可以复用 query、文档 embedding、prefix 或 rerank 结果，但索引 revision 改变时必须失效。

应记录：

~~~text
retrieval latency / candidate count / evidence recall
prompt tokens / rerank cost / TTFT / generation cost
citation precision / access denial / stale index rate
~~~

## 45.10 常见失败模式

包括 chunk 切断条件、embedding 召回主题但不召回数字、reranker 把广告/旧版本排前、权限过滤发生在生成后、索引过期、重复证据挤占上下文、模型引用未使用的段落，以及检索失败后无证据回答。

排查要保存 query、候选 ID、分数、过滤原因、最终 prompt、模型引用和 source revision。不要只看最终答案猜检索是否成功。

## 45.11 机制与边界：检索改变的是信息 bottleneck

attention 的 bottleneck 是上下文内 token 交互，retrieval 的 bottleneck 是外部索引召回和证据排序。架构化检索通过改变输入分布，把无限/大规模历史转换成有限候选；它也把 recall error 变成不可逆损失。

研究应同时测 closed-book、oracle evidence、retrieved evidence 和 noisy evidence 四种条件。oracle 与 retrieved 的差距揭示检索器损失，retrieved 与生成答案的差距揭示模型使用证据的能力。

## 45.12 面试追问、误区与练习

**问：RAG 是否可以解决长上下文问题？**

标准回答：它通过先召回候选减少模型输入和干扰，可能改善成本与证据定位；但召回、排序、版本、权限和生成使用仍有风险，不能把 RAG 等同于无限有效上下文。

**问：如何定位 RAG 系统的错误？**

标准回答：先看 required evidence 是否进入候选，再看排序和最终 prompt，再看模型是否使用/引用，最后看权限、版本和后处理；每层都应有独立指标。

常见误区包括只测生成准确率、把相似度高当证据支持、在生成后才做权限过滤，以及把检索文本直接当可信事实。

练习：为合同问答设计四类评测：oracle evidence、ANN evidence、冲突版本、权限过滤，分别定义 recall、citation precision、拒答和延迟验收条件。

## 45.13 retrieval 是信息路径，不是外挂搜索框

检索加入模型后，系统的有效信息流变为 query、索引、候选、重排、上下文和生成的联合路径。任何一环错误都会表现为模型回答错误：索引没有新版本，重排漏掉否定条件，context 丢失来源，生成没有引用支持。

可以把证据成功率拆成：

~~~math
P(\mathrm{supported})
=P(\mathrm{retrieve})
\cdot P(\mathrm{preserve})
\cdot P(\mathrm{use})
\cdot P(\mathrm{cite})
~~~

这不是独立性定理，而是排查框架。

## 45.14 长上下文和 retrieval 的协同

长上下文适合把多个候选和原文放在一起比较，retrieval 适合控制输入规模和更新。高风险任务可以先检索候选，再保留来源范围和关键片段，最后由 verifier 检查每个主张。

评估同时改变索引版本、证据位置、相似干扰、冲突和时间有效性，避免把一次检索命中误写成模型具有稳定长记忆。

## 45.15 retrieval 的索引和模型协同

索引粒度、chunk 边界、embedding 模型、reranker 和上下文预算共同决定最终证据。chunk 太大增加噪声，太小丢失限定条件；只使用向量相似度可能找不到版本号和否定词。

长上下文模型可以接收更多候选，但候选越多也可能增加 lost-in-the-middle 和引用冲突。检索器应输出来源、分数、时间和权限，模型再按任务选择使用。

## 45.16 检索失败的可观测回退

如果 top-k 没有覆盖 gold evidence，系统应记录 query、索引版本、候选和过滤原因，并决定改写 query、扩大检索、回源或拒答。把空结果当作“没有相关信息”可能是正确结论，也可能是索引事故，必须区分。

## 45.17 证据路径的四个独立验收条件

一个完整的 retrieval 任务至少有四个可以分别失败的验收条件。第一是召回验收条件：所需文档是否进入候选；第二是保真验收条件：切块和上下文组装是否保留限定条件、表头和来源；第三是使用验收条件：模型是否真正依据证据而非凭记忆作答；第四是引用验收条件：答案中的每个主张是否能回到支持片段。

可以为每道门单独定义指标：

~~~math
G_1=R_{\mathrm{evidence}},\quad
G_2=R_{\mathrm{preserve}},\quad
G_3=R_{\mathrm{use}},\quad
G_4=P_{\mathrm{citation}}
~~~

最终质量低，不代表一定要换生成模型。若 G1 低，应查索引和 query；若 G2 低，应查 chunk 和 context builder；若 G3 低，应查冲突和提示结构；若 G4 低，应加 verifier 或引用约束。

## 45.18 版本、权限和时间有效性

检索架构处理的不是静态百科，而是会更新、撤回和分权限的数据。候选记录应至少带 document_id、revision、valid_from、valid_to、tenant_scope 和 source_uri。模型看到旧版本时，不能只依赖相似度决定是否使用；时间和权限是硬过滤条件。

权限过滤要发生在证据进入模型上下文之前，而不是生成后再删除答案。否则模型可能已经在隐藏上下文中读取了不应访问的信息。高风险场景还要记录过滤原因，方便区分“没有证据”和“证据存在但无权访问”。

## 45.19 检索与长上下文的预算关系

设候选文档 token 数为 C，系统指令和用户问题占用 P，输出预算为 O，上下文上限为 W，则：

~~~math
C+P+O\le W
~~~

但把 C 填满并不一定最好。过多候选会增加噪声、位置偏差和推理成本；过少候选可能漏掉跨文档条件。应在固定 W 下比较 top-k、摘要、原文回读和多轮检索，报告 evidence recall、citation support、TTFT 和单位成功成本。

## 45.20 Retrieval 应被当作架构路径

把 RAG 当成 prompt 拼接，会忽略它对模型信息流的改变。检索器决定候选证据，重排器决定进入上下文的顺序，模型决定如何融合，引用层负责把结论绑定回来源，权限层决定哪些证据可见。每一层都可能失败，不能只看最终生成文本。

可以把一次回答的证据链写成：

```text
query -> retrieve -> rerank -> pack context
      -> reason -> cite -> policy check -> answer
```

context packing 还要考虑 token budget、位置、冲突版本、摘要和来源多样性。检索结果越多不一定越好；冗余和相似错误证据会挤压真正关键片段，并放大模型的错误一致性。

## 45.21 RAG 与模型内部记忆的边界

参数记忆适合稳定常识，外部检索适合时效、私有、可更新和需要引用的知识。把所有信息都塞进权重会增加更新和删除成本，把所有问题都交给 RAG 又会增加检索延迟、权限和证据噪声。

评估要分检索 recall、evidence precision、citation support、答案正确率和权限违规率。一个模型答对但引用了无权文档，不能称为 RAG 成功；检索命中但模型没有使用证据，问题在 evidence utilization 而非 retriever。

## 45.22 RAG 架构的缓存和回滚

embedding、向量索引、重排器、文档摘要和 prompt cache 都是派生 artifact。文档更新或删除时，必须处理索引、缓存、摘要和引用；只更新主数据库会留下旧证据。缓存 key 还要包含租户、权限版本、索引 revision、模型/模板和 query policy。

上线新 retriever 时用固定 query、长文档、冲突文档、无答案和越权样本做 replay。回滚需要一起切换 index、reranker、packing 策略和引用解析器，不能只改一个模型 ID。

## 45.23 Retrieval 质量与 generation 质量要分开

一个 RAG 系统答错时，至少有三种可能：正确证据没有被召回，证据被召回但排序/打包时丢失，证据已经进入上下文但模型没有正确使用。对应的指标分别是 retrieval recall、context precision/coverage 和 evidence-grounded answer accuracy。把它们平均成一个答案准确率，会让优化方向失真。

构造一个可诊断样本时，保存 gold evidence ids、候选集合、排序位置、最终 context、引用跨度和模型主张。若 gold evidence 不在候选集合，生成模型没有责任；若在候选集合却没进 context，查 token budget 和 packing；若进了 context 仍引用错误，查实体绑定、冲突处理和 verifier。

还要加入无答案、越权和过期文档。RAG 的“召回率越高越好”并不成立：召回无权或过期资料可能增加风险。权限过滤、版本过滤和引用验证应在生成前后都存在，不能依赖模型自己遵守。

## 45.24 Retrieval 架构的反事实回放

当 RAG 答错时，至少回放四个版本：无证据、oracle evidence、真实检索证据和带噪证据。无证据与 oracle 的差距表示模型对外部知识的依赖；oracle 与真实检索的差距表示检索/排序损失；真实证据与带噪证据的差距表示模型的抗干扰和证据使用能力。

回放还要保存最终 context，而不是只保存 top-k ID。chunk 重排、摘要、权限过滤和 token 截断都可能改变模型实际看到的内容。若 gold evidence 在索引中却没有进入 context，问题不在生成模型；若证据完整进入而引用仍错，才进入实体绑定、冲突消解和 verifier 排查。

## 45.25 检索增强架构的适用条件

检索增强语言模型的早期代表包括 REALM（https://arxiv.org/abs/2002.08909）和 RETRO（https://arxiv.org/abs/2112.04426）；具体 RAG 组件和 API 取决于系统。本文把 retrieval 作为架构组件讨论，不把某个索引实现写成通用标准。

Retrieval-Augmented Architecture 的核心是把外部知识、证据路径、权限、版本和模型生成放在同一个可审计系统里。检索成功只是第一道门。

## 45.26 Retrieval 是可恢复的状态路径

一次 RAG 请求不应只保存最终 top-k。需要保存 query revision、embedding model、retriever/reranker、候选 source ids、权限过滤、packing 结果、引用跨度、模型 revision 和 verifier。这样检索错、组装错和生成错才能分别归因。

请求被取消或索引切换时，已提交的引用和工具动作不能被新版本静默替换。索引、摘要、缓存和文档删除要有版本和回滚关系；若无法恢复到同一证据集合，旧回答只能标记为旧版本，而不能伪装成当前事实。

## 45.27 Query rewrite 和 citation 的边界

query rewrite 可能提升召回，也可能把用户原意改掉；多查询可能增加覆盖，也可能引入更多无权和过期文档。评估要保留原 query、rewrite、每路候选、去重和最终 context，检查 rewrite 是否改变实体、时间、租户或否定词。

citation 也要做 span-level 支持：引用是否真的蕴含主张，是否来自当前权限和版本，是否覆盖数字、条件和例外。引用格式合法只说明 parser 通过，不说明证据支持答案。

## 45.28 RAG 与长上下文的路由

短且高置信问题可以直接检索少量证据；长文档、多证据或冲突版本请求可能需要更大的 context、rerank、分块阅读或多轮验证。路由应按 query 难度、召回置信度、证据数量、权限和成本选择路径，而不是所有请求都塞入同一个 prompt。

容量模型要同时计算 embedding/retrieval、rerank、context prefill、KV、generation 和 verifier。若 retrieval 提高了答案质量，却让长 context 造成 p99 超限，可采用候选摘要 + 原文回读的两级路径，但最终高风险引用仍需回到原文 artifact。
