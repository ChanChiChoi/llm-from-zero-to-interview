# 第二十四章：Linear Attention 的基本思想和核技巧

## 24.1 它试图改写哪一步

标准 attention 对每个 query 和所有历史 key 做配对，产生 `T×T` 的 score。Linear Attention 的核心想法是把 attention score 写成 query/key 的特征映射内积，再利用乘法结合律先累计历史状态，从而把按 query 的全历史访问改成递归更新。

它不是简单把 softmax 删除。特征映射、归一化、因果计算、数值稳定和内容检索能力都决定最终质量。不同论文的线性 attention 公式可能不相同，必须区分教学抽象、核近似和具体模型。

## 24.2 从 softmax attention 到核分解

标准 causal attention 为：

~~~math
y_t=\frac{\sum_{i\le t}\exp(q_t^\top k_i)v_i}
{\sum_{i\le t}\exp(q_t^\top k_i)}
~~~

如果把相似度写成核 `K(q,k)`，并假设：

~~~math
K(q,k)\approx \phi(q)^\top\phi(k)
~~~

则分子可以重新排列：

~~~math
\sum_{i\le t}\phi(q_t)^\top\phi(k_i)v_i^\top
=\phi(q_t)^\top
\left(\sum_{i\le t}\phi(k_i)v_i^\top\right)
~~~

定义状态：

~~~math
S_t=\sum_{i\le t}\phi(k_i)v_i^\top,\qquad
z_t=\sum_{i\le t}\phi(k_i)
~~~

输出为：

~~~math
y_t=\frac{\phi(q_t)^\top S_t}
{\phi(q_t)^\top z_t+\epsilon}
~~~

先更新 `S_t,z_t`，再回答 query，状态更新时间与序列长度近似线性。

## 24.3 小白直觉：先做账本，再查账

full attention 是每次提问都翻遍历史账本；linear attention 是读到一条记录时，先把它写进一个汇总账本。之后的问题只查汇总账本，因此不会随着历史条目数线性增加查阅时间。

问题在于汇总账本不是原始账单。两条不同记录可能写入相似的统计量，精确地问“第 12 条记录是什么”就困难。线性 attention 换来的不是免费加速，而是把历史访问能力变成固定状态的压缩问题。

## 24.4 因果递归的最小实现

~~~python
import torch


def linear_attention(q, k, v, eps=1e-6):
    # q/k: [batch, seq, feature], v: [batch, seq, value_dim]
    b, t, f = q.shape
    value_dim = v.size(-1)
    state = torch.zeros(b, f, value_dim, device=q.device, dtype=q.dtype)
    normalizer = torch.zeros(b, f, device=q.device, dtype=q.dtype)
    outputs = []
    for step in range(t):
        key = k[:, step]
        value = v[:, step]
        state = state + key.unsqueeze(-1) * value.unsqueeze(-2)
        normalizer = normalizer + key
        numerator = torch.bmm(q[:, step:step + 1], state)
        denominator = (q[:, step] * normalizer).sum(-1, keepdim=True).unsqueeze(1)
        outputs.append(numerator / (denominator + eps))
    return torch.cat(outputs, dim=1)


q = k = torch.rand(2, 5, 8)
v = torch.rand(2, 5, 4)
print(linear_attention(q, k, v).shape)
~~~

这里假设 q/k 已经是非负特征，方便解释归一化；它不是标准 softmax attention 的等价实现。生产系统通常会用更适合的 feature map、门控、归一化和 fused scan。

## 24.5 复杂度和状态大小

若 feature 维度为 `r`，value 维度为 `d_v`，每一步更新矩阵状态的计算近似为 `O(rd_v)`，全序列为：

~~~math
O(T r d_v)
~~~

而不显式生成 `T×T` score。状态大小约为：

~~~math
O(r d_v+r)
~~~

如果 `r` 很大，或者多头状态乘上层数、batch 和 dtype，内存并不一定小。若状态更新是逐 token 递归，训练可以通过并行 scan 或 chunk 处理；推理则天然适合维护固定 state。

## 24.6 特征映射不是小细节

理想 softmax kernel `exp(q^T k)` 的特征映射可能需要无限维。实际 Performer 等方法使用随机特征近似，其他线性 attention 可能采用正值激活、ELU+1、门控或可学习映射。映射选择影响：

1. 是否能保持非负归一化。
2. 近似误差和方差。
3. 对内容相似度的分辨率。
4. kernel 是否可融合。
5. 长序列状态是否数值稳定。

feature map 的表达越丰富，状态更新可能越贵；表达太弱，检索和组合能力会退化。

## 24.7 为什么它不等于“免费替代 Transformer”

标准 attention 的 query 可以根据当前问题重新选择任意历史 token。linear attention 先把历史压成 `S_t,z_t`，之后所有 query 共享这份摘要。若任务需要复制一个罕见字符串、区分多个相似实体或根据后来的条件重新解释早期证据，固定状态可能丢失必要细节。

可以用两个历史序列 `H` 和 `H'` 构造相同或相近的状态，再用不同 query 测输出差异。若许多 query 无法区分它们，说明状态碰撞影响了内容寻址。

## 24.8 训练和推理的差别

训练时，递归形式看起来必须按时间循环；若更新满足合适的结合结构，可以用并行 prefix scan 或 chunked recurrence 加速。若 feature map 或 gate 让更新不可结合，训练并行度会下降。

推理时，固定 state 很有吸引力：每个新 token 只更新 state，不用读取完整 KV。但状态的 snapshot、reset、迁移和回滚成为 serving contract。speculative decoding 要区分候选 token 的临时 state 和已提交 state。

## 24.9 与 SSM、RNN 和低秩 attention 的边界

Linear attention 的状态常来自 `K^T V` 或其递归形式；RNN 使用一般的 `f(s,x)`；SSM 有状态转移和输入/输出投影；低秩 attention 可以近似 score 矩阵但不一定是递归状态。它们都可能拥有线性序列复杂度，却不代表信息流相同。

比较时要问：状态是向量还是矩阵；更新是否输入依赖；query 是否能直接访问历史；训练是否并行；是否有显式位置和门控；以及 kernel/生态是否成熟。

## 24.10 失败模式与诊断

常见问题包括 denominator 接近 0、state norm 爆炸、feature map 负值造成归一化异常、长序列误差累积、padding 更新状态、chunk 边界不一致和量化后状态漂移。短序列 loss 正常不代表长流稳定。

诊断可记录 feature 范围、normalizer 最小值、state Frobenius norm、不同长度 loss、精确复制率、冲突证据准确率和恢复前后 logits。若只有长序列失败，先检查数值和状态 reset，再讨论模型表达力。

## 24.11 机制与边界：近似核的质量不能只看平均误差

核近似在期望意义上接近 softmax，不代表每一个 query/key 对的相对排序都保持。长上下文检索依赖少量高分边，近似误差如果改变 top evidence 的排序，可能造成任务级突变。应按距离、相似度、稀有 token 和多证据组合测，而不是只测 perplexity。

此外，linear attention 的理论 FLOPs 可能被矩阵状态维度和 memory traffic 支配；若 state 更新无法充分并行，GPU 利用率可能低于成熟 dense attention。实际选型要做 roofline 和 kernel profile。

## 24.12 面试追问、误区与练习

**问：Linear Attention 为什么能把二次访问改成线性？**

标准回答：在可分解的核近似下，利用结合律先累计历史的 key-value 状态，再用当前 query 查询状态，避免显式构造所有 query-key 对；代价是核近似误差和固定状态的信息压缩。

**问：线性 attention 的状态是不是等价于完整 KV cache？**

标准回答：不是。它保存的是聚合统计或递归状态，不能保证任意恢复每个历史 token；KV cache 保存位置化的 K/V，内容寻址能力更强但随长度增长。

常见误区包括把线性复杂度当作全方位加速、忽略 feature map、把所有 SSM 都称为 linear attention，以及忘记 denominator、padding 和 state rollback。

练习：用不同 feature map 实现 toy causal linear attention，对照 dense attention 测平均误差、top evidence 排序、长距复制和 state 内存。

## 24.13 线性 attention 的状态容量

线性 attention 把历史的 key-value 关系累积进状态，计算复杂度下降的同时也引入表示瓶颈。不同历史如果产生近似相同的聚合矩阵，后续 query 就难以区分它们。

可以用任务驱动的容量测试代替抽象争论：增加实体数量、远距数字数量和相似干扰，记录 exact recall、冲突选择和状态维度。若状态维度增加才恢复精确引用，说明速度收益和记忆容量存在直接交换。

## 24.14 kernel 设计的真实边界

递归更新通常涉及小矩阵、归一化和状态读写。理论上是 O(T)，但 GPU 是否高效取决于 batch、state shape、融合程度和内存布局。一个未经融合的 Python 循环可能比成熟的 dense attention 慢很多。

工程验证需要做 kernel microbenchmark 和端到端 benchmark 两层。前者找出 update、projection 和 reduction 的时间，后者检验调度、cache、p99 和任务质量是否仍然满足要求。

## 24.15 分母不是实现细节

核化 attention 常把分子和分母分别累计：

~~~math
S_t=\sum_{i\le t}\phi(k_i)v_i^\top,\qquad
z_t=\sum_{i\le t}\phi(k_i)
~~~

输出近似为：

~~~math
y_t=\frac{\phi(q_t)^\top S_t}
{\phi(q_t)^\top z_t+\varepsilon}
~~~

如果 padding、reset 或 chunk continuation 只更新 S 而没有同步更新 z，短序列可能看不出问题，长序列和不同 batch 排列则会出现尺度漂移。分母还可能带来数值下溢，因此需要和状态本体一起做 reference 对照。

## 24.16 核近似与任务误差

线性 attention 改变了计算顺序，若同时用 feature map 近似 softmax，误差会经过状态累计和后续层放大。评估应分别比较 exact kernel、随机特征、确定性 feature map 和带 gate 的变体。

任务级指标要覆盖稀有数字、相似实体、长距复制和多证据合并。平均语言 loss 可能对少量关键字段不敏感，不能替代字段级 exact recall。

## 24.17 状态回滚和并发调度

线性 attention 的状态通常是矩阵或分子/分母对，推测解码、请求分叉和 beam search 都可能需要复制或回滚。若只保存输出 token，不保存 S 和 z，拒绝候选后后续 logits 会被污染。

高并发服务还要考虑 state update 的写带宽、batch 中不同请求的 shape 和状态分页。理论 O(T) 不能掩盖小矩阵更新、layout 转换和 kernel launch 的常数成本。

## 24.18 线性状态的容量与精确检索

线性 attention 把历史压进某种累积状态，复杂度因此可以随序列长度近似线性，但状态并不是无限大的数据库。设特征维度为 `r`，key/value 的维度为 `d`，一个常见状态形状近似是 `r\times d`；当 `r` 固定而输入长度继续增加，更多事实会竞争有限的表示容量。

应使用 capacity probe 测量它，而不是凭复杂度推断能力。把多个随机实体写入序列，再在不同位置查询；增加实体数量、干扰长度、重复次数和冲突版本，记录 exact recall、false positive、数值误差和 state bytes。一个方法可能在平均语言建模 loss 上很强，却在“从 100 个候选中找第 73 个版本号”任务上明显退化。

## 24.19 归一化和数值稳定性

线性 attention 常见的归一化形式是用累计 key 特征作为分母：

```math
y_t=\frac{\phi(q_t)^{\mathsf T}S_t}
{\phi(q_t)^{\mathsf T}z_t+\epsilon},qquad
S_t=S_{t-1}+\phi(k_t)v_t^{\mathsf T},
\quad z_t=z_{t-1}+\phi(k_t)
```

分母很小、特征值过大、长序列累积或低精度都会造成放大、下溢和漂移。实现应比较 full sequence、chunked、step-by-step 三条路径，检查分母分布、state norm、logit 差异和长度增长。`epsilon` 不是可以随便填的常数；它会改变小分母区域的输出，必须纳入回归。

## 24.20 并发、回滚与状态共享

线性 attention 的状态看起来比 KV 小，但它是顺序敏感的可变对象。请求 batch 合并、分叉候选、preemption 和取消都要定义 state owner、position、revision 和 copy-on-write 规则。两个请求共享相同前缀时，只有在后续写入能隔离且模型/模板/位置契约一致时才可共享。

speculative decoding 还要处理候选被拒绝时的 state 回滚。若 draft 已经更新了线性累积状态，target 拒绝后必须恢复到提交前的 state，不能只回退输出 token。服务 benchmark 应包含分叉、拒绝、重排和恢复，而不只是连续单请求。

## 24.21 查询条件决定状态压缩是否可接受

线性 attention 的状态是对历史做的统计汇总，而查询是在读取时才出现的。若未来 query 的类型在训练中稳定，固定统计可能足够；若 query 会临时改变关注对象，预先把所有历史压成同一个 `S_t` 就可能丢失区分信息。

可以构造一个很小的对照：前缀中有 100 个实体，每个实体有一个版本号；测试时随机询问任意实体，或者询问“版本号最大的实体”。前者要求内容寻址，后者要求聚合与比较。逐步增加实体数量、重复名称和冲突版本，记录 exact recall、false positive、state bytes 和恢复误差。若主题分类保持稳定但版本号召回下降，说明状态保留了语义总量，却没有保留可查询的原子结构。

状态压缩还改变了故障恢复方式。显式 KV 可以按 token page 重新取证，线性 state 若在一次更新中数值漂移，后续只能从最近 checkpoint 重算；因此 checkpoint 间隔会转化为恢复时间和重复计算成本。服务设计必须明确 state snapshot 的版本、hash、owner、position 和回滚点。

## 24.22 结合律带来的工程含义

线性 Transformer 的递归/核化思路可参考 *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*（https://arxiv.org/abs/2006.16236）；Performer 的随机特征路线见 https://arxiv.org/abs/2009.14794。不同模型对 feature map、gate 和 state 的实现差异很大。

Linear Attention 的核心是改变 attention 的计算顺序，把历史变成可更新状态。它适合愿意接受信息压缩的长流任务，但对任意精确检索和复杂多证据组合要保持证据边界。

## 24.23 结合律带来的训练/推理二重性

线性 attention 的关键是把相似度计算重排，使历史可以累积为 `S_t` 和 `z_t`。训练阶段可以利用并行 prefix scan 或矩阵乘法，推理阶段则按 token 更新固定状态。二者只有在初始化、padding、position、归一化和数值精度一致时才代表同一个模型。

对同一 prefix，必须比较 full sequence、chunk continuation、single-step 和 snapshot restore 的 logits。若输出最终相同但中间 state 差异随长度扩大，下一次工具调用、分叉或恢复仍可能不同。state schema、版本和回滚点应进入 serving manifest。

## 24.24 线性不是无条件的低成本

线性路径减少了显式 token-to-token 交互，却可能增加 feature projection、state 读写、归一化和 kernel launch。实际时间可粗略写成：

```math
T=T_{\mathrm{projection}}+T_{\mathrm{state\ update}}
 +T_{\mathrm{normalization}}+T_{\mathrm{launch}}+T_{\mathrm{fallback}}.
```

小 batch、短序列和低 state 复用时，固定开销可能超过 dense kernel；长流和高并发才可能摊薄。评估必须包含目标 GPU、变长请求、p99、state bytes、kernel path 和任务质量。

## 24.25 与外部检索的职责分工

线性 state 适合保存可压缩的上下文统计，RAG/外部 memory 适合保存可回读、可授权和可删除的原始证据。一个可靠组合可以让线性层扫描长文档，触发候选区间，再由索引返回原文，最终由显式 attention 或 decoder 做精确回答。

这时不能把最终引用归因给 linear state。日志要保存 candidate、source id、ACL、版本和 verifier 结果；若 state 提议了不存在的引用，应被归类为 provenance 失败而不是“检索命中”。
