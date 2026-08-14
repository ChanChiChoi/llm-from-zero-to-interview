# 第四十六章：Memory-Augmented Transformer 与长期记忆架构

## 46.1 上下文窗口不是长期记忆

上下文是一次请求中显式提供的工作区，长期记忆需要跨请求、跨会话保存、检索、更新和删除。把所有历史对话一直拼进 prompt 既昂贵，也容易包含过时、错误或敏感信息。

Memory-Augmented Transformer 的问题是：模型如何读取外部 memory；何时写入；如何判断一条记忆是否可信、过期或属于当前用户；如何在训练和 serving 中管理 memory state。

## 46.2 Key-value memory 的基本形式

外部 memory 可写成条目集合：

~~~math
\mathcal{M}=\{(k_i,v_i,m_i)\}_{i=1}^{N}
~~~

query 从 memory 中读取：

~~~math
a_i=\operatorname{softmax}(q^\top k_i),\qquad
r=\sum_i a_i v_i
~~~

若 memory 很大，需要先做 ANN/top-k：

~~~math
\mathcal{I}(q)=\operatorname{TopK}_{i}s(q,k_i)
~~~

再在候选内做精细 attention。memory 不只是向量库，还应包含 owner、时间、来源、版本、权限和删除状态。

## 46.3 小白直觉：工作记忆和档案记忆

模型上下文像桌面上的工作文件，读起来快但桌面有限；外部 memory 像档案室，能保存更久但需要索引和权限。把所有档案搬到桌面会拥挤，完全不查档案则会忘记用户偏好和历史事实。

长期记忆最难的不是“存下去”，而是以后在合适的时机找回来，并且不把错误或敏感信息当作事实。

## 46.4 Read、write、consolidate

记忆系统通常有三个操作：

1. **Read**：根据当前任务取候选事实。
2. **Write**：把新事件、结论或用户偏好写入。
3. **Consolidate**：去重、合并、衰减、纠错和删除。

写入策略可以有阈值：

~~~math
\operatorname{write}(e)=
\mathbb{1}[u(e)\ge \tau_u]\mathbb{1}[\mathrm{safe}(e)]
~~~

其中 u(e) 是事件的重要性/稳定性评分。阈值太低会记入噪声，太高会漏掉有价值事实。模型自动写入不能绕过权限和用户控制。

## 46.5 记忆与当前上下文的融合

可以把 memory result 作为额外 token、cross-attention 输入或结构化字段。一个融合形式是：

~~~math
h'_t=h_t+\lambda_t\operatorname{Read}(q_t,\mathcal{M})
~~~

λ_t 可以由模型或 policy 控制，但不能让低可信 memory 无条件覆盖用户当前明确输入。冲突时应按照 source revision、时间、权限和用户确认规则处理。

## 46.6 一个 toy memory

~~~python
class ToyMemory:
    def __init__(self):
        self.items = []

    def write(self, key, value, source):
        self.items.append({"key": key, "value": value, "source": source})

    def read(self, query, top_k=2):
        ranked = sorted(
            self.items,
            key=lambda item: sum(a * b for a, b in zip(query, item["key"])),
            reverse=True,
        )
        return ranked[:top_k]


memory = ToyMemory()
memory.write([1.0, 0.0], "偏好中文", "user-confirmed")
memory.write([0.0, 1.0], "旧地址", "unverified")
print(memory.read([0.9, 0.1]))
~~~

真实系统要做向量化、权限过滤、时间/版本排序、删除、加密和审计；这个 demo 只展示 key/value 读写。

## 46.7 写入错误比遗忘更危险

如果 memory 漏掉一次偏好，下一次可能只是没有个性化；如果把模型猜测写成用户事实，错误会跨会话传播。系统要区分：

~~~text
observed fact / user confirmed / model inferred / temporary state
~~~

不同类型的记忆拥有不同的保留、引用和删除策略。模型生成的摘要不能自动升级为可信事实。

## 46.8 长期记忆和隐私

memory 可能保存个人信息、企业机密、工具返回和敏感推断。访问控制要在检索前执行，删除要能够从索引、缓存、备份和派生摘要中传播。日志不能因为“只是 embedding”就默认无敏感性。

多租户系统要绑定 owner/tenant，不能用相似度检索跨用户泄露；缓存命中、prefix sharing 和 batch state 都要做隔离测试。

## 46.9 训练方式

可以从外部 memory 监督 read/write，使用检索结果做上下文；也可以让模型在任务奖励下学会查询和写入。联合训练的风险是 shortcut、memory leakage 和训练/线上索引不同。

离线训练应模拟 stale memory、冲突 memory、空 memory、恶意 memory 和删除事件。否则线上遇到过期偏好时，模型可能毫无拒答和校验能力。

## 46.10 与 RAG、KV cache 和 context window 的区别

KV cache 保存当前 forward 的历史表示，生命周期通常是请求级；RAG 从外部文档召回证据，通常是任务级；长期 memory 跨请求维护用户/系统状态；context window 是模型一次可见的 token 容量。它们可以组合，但不是同一个对象。

把 KV 当长期 memory 会造成版本、隐私和重启问题；把 RAG 当用户 memory 会把文档知识和个体事实混在一起；把 memory 全塞进 prompt 会重新遇到上下文成本。

## 46.11 机制与边界：memory consistency

长期 memory 要处理写后读一致性、并发写冲突、版本和 TTL。可为条目定义：

~~~math
\operatorname{valid}(m)=
\mathbb{1}[t_{\mathrm{now}}<t_{\mathrm{expire}}]
\mathbb{1}[\mathrm{revision}\in\mathcal{R}]
\mathbb{1}[\mathrm{scope}= \mathrm{request.scope}]
~~~

检索结果还要带 provenance，生成回答引用的不是向量距离，而是可审计的原始来源。memory compaction 可能改变语义，应做 compaction 前后回归。

## 46.12 面试追问、误区与练习

**问：Memory-Augmented Transformer 和 RAG 的区别是什么？**

标准回答：RAG 通常从外部知识库按当前问题召回文档；长期 memory 还包含跨请求读写、用户/系统状态、版本、权限和生命周期。二者都可能用检索，但数据语义和治理不同。

**问：为什么不能让模型把所有回答都写入 memory？**

标准回答：模型输出可能错误、敏感、临时或未经确认；无验收条件写入会造成错误传播和隐私风险。应区分事实类型、来源、用户确认、TTL 和删除策略。

常见误区包括把 embedding 当无敏感数据、忽略 stale/conflict、只做 read 不做 delete，以及不测试多租户隔离。

练习：设计一个带 source、scope、TTL、revision 和 deletion tombstone 的 memory schema，模拟写入冲突、过期和跨租户查询。

## 46.13 memory 写入的可信度验收条件

长期 memory 不能把每个生成结果都保存。写入应检查来源、时间、范围、冲突和用户权限；对于可变事实，记录有效期和覆盖策略；对于高风险事实，回到权威数据库或人工确认。

一个条目的写入条件可以写成：

~~~math
\mathrm{Write}(m)
=\mathrm{SourceValid}(m)
\land\mathrm{ScopeValid}(m)
\land\mathrm{ConflictChecked}(m)
\land\mathrm{Authorized}(m)
~~~

## 46.14 记忆读取与旧事实污染

读取 memory 后要带 revision 和 provenance。若当前文档版本与 memory 不同，系统应优先回源或把旧条目标记为过期。只根据相似度命中旧摘要，可能让模型在新政策上继续使用旧规则。

评估应包含事实更新、删除请求、租户隔离、冲突记忆和恢复。memory 命中率增加不等于质量增加。

## 46.15 memory 的写入和读取分离

写入器负责判断一个事实是否值得长期保存，读取器负责在当前任务中选择相关记忆。把两者合成一个相似度检索，会让未经验证的旧回答持续污染未来任务。

记忆条目要有来源、时间、scope、confidence、revision 和删除状态。读取结果带着这些元数据进入 prompt，模型才能区分“当前权威事实”和“历史经验”。

## 46.16 记忆一致性和删除

用户修改偏好、文档升级或权限撤销后，memory 必须失效。测试应覆盖写入、覆盖、删除、恢复、跨租户和并发读写；只验证命中率无法发现旧事实泄漏。

## 46.17 记忆条目的生命周期

长期记忆不是一条永不过期的向量。一个条目通常经历候选写入、验证、激活、更新、降级、删除和审计几个状态。偏好信息可以有较长 TTL，订单、政策和权限信息则需要短 TTL 或每次回源。把所有条目用同一个相似度分数排序，会把不同生命周期的数据混在一起。

可以把记忆条目抽象为：

~~~text
MemoryItem {
  content
  source
  scope
  created_at
  expires_at
  revision
  confidence
  deletion_state
}
~~~

读取时先做 scope、时间和删除状态过滤，再做语义排序。模型生成的摘要如果没有原始来源，最多是低置信候选，不能自动升级为权威事实。

## 46.18 记忆与上下文折叠的区别

上下文折叠通常压缩当前任务轨迹，目标是让任务继续执行；长期 memory 记录跨请求可复用的信息，目标是未来检索。前者强调任务状态、未完成动作和工具轨迹，后者强调来源、范围、TTL 和删除。两者都可能使用摘要，却不能共用同一写入验收条件。

如果把临时假设写进长期 memory，错误会跨任务传播；如果把长期约束只放在一次摘要里，下一次请求可能完全看不到。系统应明确哪些信息只存 session，哪些信息可跨 session，以及谁有权删除。

## 46.19 记忆质量的回放评估

记忆系统要用事件回放验证，而不是只看命中率。构造用户偏好变更、事实更新、删除、跨租户查询和冲突写入，比较无 memory、错误 memory、正确 memory 和回源策略。记录答案正确率、旧事实泄漏率、删除残留率、延迟和存储成本。

一个重要的验收条件是删除后不可再用，而不只是主表出现 tombstone。向量索引、缓存、摘要和备份副本都可能保留可读信息；高风险系统要能证明派生数据也已失效。

## 46.20 Memory 的读写策略必须分开评估

外部 memory 系统至少有写入、选择、读取、更新和删除五个动作。写入过于激进会把猜测、敏感信息和提示注入保存下来；写入过于保守又无法形成长期帮助。读取命中也不代表模型正确使用了内容，必须检查来源、时效和权限。

可以为 memory item 记录 `fact/hypothesis/preference/action` 类型、来源、确认状态、时间、tenant、版本和删除标记。模型生成的摘要不应自动升级为事实；用户明确纠正、外部权威来源或独立验证才可以改变状态。

## 46.21 Memory 与 context/cache 的区别

context 是当前请求可见的输入，KV cache 是当前或共享前缀的执行状态，memory 是跨请求保存的知识或任务状态。三者在生命周期、权限、删除、压缩和一致性上不同。把 KV cache 当 memory 会让短期隐状态长期泄露；把 memory 当普通 context 又会丢掉来源和生命周期。

系统应分别测 context folding、memory retrieval、prefix cache、state restore 和用户删除。删除一个 memory item 还要清理向量索引、摘要、缓存、引用和备份中的可删除副本。

## 46.22 Memory 的任务收益和副作用

评估 memory 不能只看多轮任务成功率。要增加记忆精确率、错误记忆率、过期使用率、删除后残留率、跨租户泄露率和用户纠正后的恢复率。长期任务还要测摘要漂移、版本冲突和上下文膨胀。

安全边界是：memory 可以帮助模型恢复已确认的状态，但不能代替当前授权、当前外部状态和高风险动作确认。每次写入和读取都应有 trace、policy version 和可解释来源。

## 46.23 Memory 写入验收条件和删除证明

跨请求 memory 的最大风险不是“没记住”，而是把猜测保存成事实、把敏感信息保存过久，或用户删除后仍能从派生索引读出。写入应经过类型、来源、确认状态、租户、过期时间和风险策略检查；模型自动生成的摘要默认是候选，不应直接成为可信事实。

删除测试要覆盖主表、向量索引、摘要、prefix/result cache、备份和离线导出。构造一个 memory item，执行删除，再用原问题、改写问题、相似实体和不同租户查询，检查命中率和引用是否降为零或进入明确的删除延迟窗口。只在主表看到 tombstone 不能证明不可再用。

读取也要重新检查当前权限和外部事实。memory 可以恢复用户确认过的偏好或任务状态，却不能替代当前授权、实时余额、最新库存和高风险动作确认。每次读写都保存 source、policy revision 和 trace，才能在错误记忆导致事故时回放。

## 46.24 记忆系统的完整验收链

一个长期 memory 请求至少要经过五道门：检索候选通过 scope/权限过滤，来源和 revision 可追溯，模型正确使用或拒绝记忆，写入经过类型和可信度审核，删除能传播到索引、缓存和备份。命中率只覆盖第一道门，不能证明系统安全或答案正确。

可以用一个带数字的回归集验收：100 条事实更新任务中，若 92 条找到了候选，84 条绑定到正确版本，80 条回答使用了证据，删除测试中仍有 3 条能从向量索引召回，那么系统的“召回率 92%”不能作为上线依据；删除残留和版本错用应是硬失败。

## 46.25 记忆增强架构的适用条件

外部检索增强的背景可参考 REALM（https://arxiv.org/abs/2002.08909）和 RETRO（https://arxiv.org/abs/2112.04426）；长期记忆系统的具体实现差异很大，不能由“memory augmented”名称推断。

Memory-Augmented Transformer 解决的是跨请求可读写状态，而不是简单扩大上下文。读取质量、写入可信度、版本/隐私治理和 serving 生命周期必须共同设计。

## 46.26 Memory item 的状态机

记忆条目不应只有存在/不存在两个状态。一个实用生命周期可以是 `candidate -> verified -> active -> superseded -> deleted`；候选来自模型摘要或用户输入，verified 表示来源/用户确认或外部验证通过，active 才能进入默认检索，superseded 保留版本关系但不再作为当前事实，deleted 进入删除传播。

每次状态转移要记录 actor、source、policy revision、timestamp、tenant 和 reason。模型生成的“用户喜欢某品牌”不能因为语言流畅就直接进入 active；当前请求中的新信息也不能无条件覆盖已确认事实。状态机能把记忆错误从“模型幻觉”具体化为写入验收条件或版本冲突问题。

## 46.27 记忆检索的评分和证据链

memory retrieval 可以使用语义相似、时间衰减、任务相关性、可信度和权限过滤。简单的教学评分为：

```math
S(m,q)=\alpha\,\mathrm{sim}(m,q)
 +\beta\,\mathrm{fresh}(m)
 +\gamma\,\mathrm{trust}(m)
 -\delta\,\mathrm{risk}(m).
```

实际系统不能把最高分直接当事实。返回结果要带 source、version、valid_from/valid_to、ACL 和 confidence；模型回答引用的是外部 artifact 还是 memory 摘要，要明确区分。过期、冲突或权限不足的条目可以作为候选提示，但不能在无标记的情况下写入最终答案。

## 46.28 多租户和删除的传播图

一个 memory item 常有向量、摘要、倒排索引、prefix cache、备份和离线训练候选等派生对象。删除不是改一行数据库，而是沿派生图传播 tombstone，并在删除延迟窗口内阻止新查询继续返回它。

可以为每条边保存 `parent_id -> derived_id`、生成版本、存储位置、权限域和删除状态。回归测试用原问题、改写、相似实体、跨语言和不同租户查询；同时检查主表、向量、摘要和缓存。若向量仍能召回，主表 tombstone 不能算删除完成。

## 46.29 Memory 与 Agent 长周期任务

长周期 Agent 需要保存的不只是“事实”，还可能包括任务状态、待办、用户偏好、工具结果和 checkpoint。不同类型的状态有不同可信度和生命周期：任务 checkpoint 需要可恢复，偏好可以让用户纠正，工具结果需要时间版本，安全策略和权限不能被 memory 覆盖。

因此 memory 写入必须区分 `fact`、`preference`、`task_state`、`observation` 和 `proposal`。Agent 恢复时先读取任务 checkpoint，再重新验证当前权限和外部状态；不能因为 memory 说“上次已授权”就跳过本次高风险确认。

Memory 系统的价值是降低重复上下文和长期任务的认知负担，但它增加了写入、读取、删除、版本和权限的治理面。只有把这些面作为独立实体评估，才不会把“记得更多”误写成“系统更可靠”。
