# 第 55 章 Mistral 与 Mixtral：小模型效率、滑窗注意力和开源 MoE

## 55.1 Mistral 提出的真正问题

当模型规模不断增长时，工程师会遇到一个现实问题：在有限显存和延迟预算下，能否用更小的模型取得足够好的质量。Mistral 7B 的影响在于，它把注意力结构、KV cache、数据和训练效率放在一起考虑，说明参数量不是模型能力的唯一变量。

Mixtral 则把这个问题延伸到稀疏 MoE：总参数可以很大，但每个 token 只激活少数专家。两者共同推动了开源生态对推理成本、专家路由和高效架构的关注。

## 55.2 Grouped-Query Attention 的 cache 直觉

标准 MHA 为每个 query head 保存一组 key/value head。GQA 把多个 query head 分组，共享较少的 KV heads。设 query head 数为 H_q，KV head 数为 H_kv，则每个 token 的 KV payload 近似为：

~~~math
M_{\mathrm{KV/token}}
\propto 2H_{\mathrm{kv}}d_h b
~~~

当 H_kv 从 H_q 降低时，KV cache 和读取带宽下降；但 query head 仍然保持多头表达，质量和速度之间需要消融。MQA 是 H_kv=1 的极端情况，GQA 是中间折中。

GQA 的收益主要出现在 decode 和长上下文服务，不会自动降低 prefill 中所有 attention 计算。分析时要把权重 FLOPs、prefill、decode、cache 和 batch 分开。

## 55.3 Sliding Window Attention 的局部性

滑窗注意力让位置 t 只访问最近窗口 W 内的历史：

~~~math
\mathcal{N}(t)=\{i\mid \max(1,t-W+1)\le i\le t\}
~~~

单层的直接访问从整个前缀变为局部窗口，长序列成本和 cache 访问压力下降。多层堆叠后，信息仍可逐层传播到更远位置，但这不是一次 query 就能直接读取任意远处 token。

滑窗适合局部语言模式和连续流，但面对跨段精确引用、远距离变量绑定和多证据冲突，可能需要全局 token、检索或少量 global attention。

## 55.4 局部窗口不等于有效上下文

假设每层窗口宽度为 W，层数为 L，理论上信息传播范围可能随层数扩大；但是传播过程中会经历表示混合、归一化、噪声和训练分布限制。不能根据 L×W 直接宣称模型支持同等质量的上下文长度。

实验应比较：

1. 单个远距离 needle 是否能被找到。
2. 多个远距离证据能否同时合并。
3. 中间位置是否比首尾更差。
4. 精确数字和引用是否保留。
5. 窗口边界变化是否产生断裂。

## 55.5 Mixtral 的稀疏 MoE

设有 E 个专家，每个 token 选择 top-k 个专家。一个教学化的 MoE 层为：

~~~math
y_t=\sum_{e\in\mathrm{TopK}(g(x_t))}
g_e(x_t)f_e(x_t)
~~~

f_e 是第 e 个专家 FFN，g_e 是路由权重。与 dense FFN 不同，MoE 需要把 token 分发到不同专家，再把输出合并回来。

稀疏激活让总容量和每 token 计算部分解耦，但专家权重仍需存储，路由和通信仍需付费。跨 GPU 的 all-to-all 可能成为主要瓶颈，尤其当 token 分布不均或专家被分散到不同节点时。

## 55.6 负载均衡与容量因子

如果所有 token 都偏向少数专家，热门专家会溢出，其他专家闲置。常见路由辅助损失会鼓励 token 分布与路由概率更均衡。一个抽象形式是：

~~~math
\mathcal{L}_{\mathrm{aux}}
=\lambda E\sum_{e=1}^{E}f_e p_e
~~~

其中 f_e 是实际 token fraction，p_e 是路由概率 fraction。具体实现可能使用不同归一化和容量策略。

容量因子决定每个专家最多接收多少 token。容量过小会丢弃或转移 token，容量过大增加 padding 和内存。负载均衡不能只看辅助损失，还要看专家利用率、溢出率、通信、质量和尾延迟。

## 55.7 MoE 的一轮 dispatch

一个简化的执行过程是：

~~~text
hidden states
  -> router logits
  -> top-k expert ids
  -> capacity and token permutation
  -> all-to-all or local dispatch
  -> expert FFN
  -> combine by router weights
  -> restore token order
~~~

任何一个 index 或排序错误都会造成 token 输出错位。分布式实现还要检查不同 rank 的 token 数、空专家、通信超时和异常恢复。

## 55.8 一个最小 MoE 路由实验

下面的代码只演示 top-1 路由的形状，不包含分布式 dispatch：

~~~python
import torch


def top1_route(hidden, router):
    logits = router(hidden)
    expert_id = logits.argmax(dim=-1)
    load = torch.bincount(expert_id, minlength=logits.shape[-1])
    return expert_id, load


hidden = torch.randn(12, 8)
router = torch.nn.Linear(8, 4, bias=False)
ids, load = top1_route(hidden, router)
print(ids.tolist(), load.tolist())
~~~

实验时可以改变 router 初始化、输入分布和容量，观察 load variance；再比较均衡损失、路由随机性和最终任务质量。

## 55.9 Mistral 与 Mixtral 的共同工程启示

Mistral 说明局部注意力和 GQA 可以降低长上下文推理成本；Mixtral 说明稀疏专家可以提高总容量与每 token 激活计算的分离程度。两者都把“架构选择”与 serving 资源模型直接连接起来。

但它们解决的瓶颈不同。滑窗减少历史访问范围，GQA 减少 KV 头，MoE 改变 FFN 容量和路由。把它们统称为“稀疏化”会掩盖不同的质量和系统风险。

## 55.10 训练与推理的差异

训练时 MoE 的 token dispatch 可以在大 batch 中摊平通信和 kernel；在线 decode 时每轮 token 少、请求长度不同，路由开销和小矩阵效率可能更突出。滑窗 attention 在训练和推理都限制访问，但 cache layout 和窗口滚动仍需正确实现。

部署压测应分开：

| 阶段 | 重点 |
| --- | --- |
| Prefill | 输入吞吐、窗口 mask、路由通信、专家容量 |
| Decode | KV 读取、专家小 batch、p99、continuous batching |
| 长上下文 | 窗口边界、跨段 recall、cache bytes |
| 高并发 | 热门专家、all-to-all、排队和尾延迟 |

## 55.11 小模型效率不是只看参数

一个小模型可能通过更好的数据、tokenizer、训练 token、架构和后训练取得强能力，但也可能在知识覆盖、长推理和多语言上有明显边界。比较小模型时，报告：

~~~text
parameters
training_tokens
active_parameters
context_length
tokenizer
quantization
latency
task_success
~~~

否则“7B 超过 70B”之类的表述很容易脱离任务、提示和推理设置。

## 55.12 常见失败模式

GQA 头映射错误会使质量异常；滑窗 mask 的边界错误会泄漏未来或丢失当前 token；MoE token permutation 错误会造成输出错位；容量过小产生 token drop；热门专家造成 p99；量化后 router logits 改变路由；只测平均吞吐掩盖专家热点。

诊断时应保存 query/KV head mapping、window range、expert id、capacity、token count、通信时间和输出对齐信息。

## 55.13 GQA、滑窗和 MoE 解决的是不同瓶颈

GQA 主要减少 KV head 和 decode 读写；滑窗限制显式历史的可见范围；MoE 增加条件参数容量。三者可以组合，却不能把其中一个的收益归到另一个身上。GQA 仍可能保留全上下文，滑窗仍可能需要 global 路径，MoE 仍需加载和通信专家权重。

比较时把 prefill FLOPs、decode KV bytes、专家通信、窗口命中和任务精确引用分开记录。这样才能解释为什么一个模型在长 decode 上省内存，却在多卡专家 dispatch 上出现更高尾延迟。

## 55.14 MoE 的负载、通信和质量三角

专家分配的资源问题可以写成三项约束：负载均衡、通信开销和语义分工。强行均匀可能破坏专家 specialization，允许热点又会导致排队和 token overflow。容量因子、top-k、专家位置和并行拓扑共同决定结果。

一个简单的服务成本分解为：

~~~math
C_{\mathrm{MoE}}
=C_{\mathrm{router}}+C_{\mathrm{dispatch}}
+C_{\mathrm{expert}}+C_{\mathrm{combine}}
+C_{\mathrm{imbalance}}
~~~

最后一项包括热点、padding、丢 token、重试和尾延迟。只看 active parameters 会漏掉这些成本。

## 55.15 适用边界和回退设计

Mistral/Mixtral 路线适合希望在有限显存下获得较强质量、并且能承担相应 kernel 或分布式 dispatch 的系统。低并发单卡、严格 p99、专家通信受限或任务需要稳定 dense 路径时，MoE 的复杂度可能不划算。

回退可以包括 dense checkpoint、减少 top-k、专家副本、缩短窗口、检索原文或降低输出预算。回退触发必须进入评估和 trace，否则产品指标会把 fallback 成功和主路径成功混在一起。

## 55.16 发布数字如何落到容量规划

Mistral/Mixtral 的 total parameters、active parameters、窗口、KV heads 和专家 top-k 必须同时映射到容量模型。权重可能常驻，KV 随请求增长，专家通信随 token 分布变化，滑窗则改变不同层的历史长度。

面对高并发请求，应按输入长度、输出长度、专家热点和 fallback 分桶，记录峰值而不是只看平均值。一个小模型如果因为工具重试或专家排队产生更多失败，也可能不如更大但更稳定的 dense 模型。

## 55.17 Local attention 和 MoE 是两种不同的稀疏

Mistral 风格的 sliding window 减少 token 间的连接，Mixtral 风格 MoE 减少每个 token 激活的 FFN 专家数。前者改变历史信息路径和 KV 需求，后者改变参数激活、路由和通信；二者都叫“稀疏”但失败模式不同。

比较时要分别测长距证据、局部语法、expert load、overflow、reroute、通信、显存和 p99。MoE 的总参数可能很大，即使 active parameters 较小，权重驻留和跨卡通信仍会主导服务成本。

## 55.18 MoE router 的稳定性

router 需要处理负载均衡、capacity、token drop、top-k、专家崩溃和训练/推理分布漂移。某个 expert 长期过载会导致 token 被丢弃或 reroute，模型质量和尾延迟一起恶化。线上要记录 expert histogram、overflow、reroute、通信时间和请求级失败。

## 55.19 从模型论文到 serving engine

部署 MoE 需要权重分片、expert parallel、通信拓扑、batch 调度、量化和 fallback；部署 sliding window 需要 window offset、global token、KV page 和长上下文质量验收条件。只有在这些约束下的端到端 benchmark 才能支持“高效”。

## 55.20 资料范围与效率判断

Mistral 7B 论文公开了 GQA、滑窗注意力和小模型效率方向；Mixtral 8x7B 论文公开了稀疏 MoE 路线。具体开源权重的 tokenizer、训练 token、上下文、量化和实现版本应以模型卡和代码为准。

Mistral/Mixtral 的核心知识点不是背模块名称，而是理解两个资源交换：用局部和共享 KV 降低历史访问，用稀疏专家把总容量与激活计算分离，同时承担路由和通信成本。

## 55.21 面试问题与练习

**问：GQA 为什么能降低 KV cache？**

因为多个 query head 共享较少的 key/value head，历史只需保存较少 KV；但 query 侧表达和质量、prefill 与 decode 的收益需要分别评估。

**问：MoE 为什么不是免费增加参数？**

专家权重仍需存储，token dispatch、all-to-all、负载均衡和容量溢出都会产生成本；每 token 激活少只说明部分计算被稀疏化。

**练习：**给定 H_q=32、H_kv=8、d_h=128、序列长度 T，推导 GQA 相对 MHA 的 KV payload 比例，并设计一个含专家热点和窗口边界的压测。

资料入口：

- Mistral 7B: https://arxiv.org/abs/2310.06825
- Mixtral: https://arxiv.org/abs/2401.04088
- Switch Transformer: https://arxiv.org/abs/2101.03961

## 55.22 GQA、滑窗和 MoE 的预算不能相加后结束

GQA 主要减少每个 token 需要保存的 K/V head，sliding window 限制可访问历史，MoE 只激活部分 FFN expert。三者分别作用于 cache、attention 连接和参数计算，资源模型不能只写一个“稀疏率”。

对 decode 阶段，KV 近似随 `n_kv`、历史长度和层数增长；对 MoE，权重总容量随所有 expert 增长，active FLOPs 随 top-k 增长；对滑窗，远程信息路径和窗口边界决定质量。一个请求即便权重激活很少，也可能因为长窗口、all-to-all 或 expert 热点而变慢。

因此容量报告应分开列出权重驻留、KV、expert dispatch、通信、窗口执行和 fallback。只有在同一硬件和 batch 下测量，才知道哪一项是当前瓶颈。

## 55.23 Expert parallel 的通信与负载均衡

MoE token 先由 router 选择 expert，再可能通过 all-to-all 把 token 发送到对应设备，计算后再返回原序列顺序。路由熵低可能让少数 expert 过载，路由熵过高则增加通信和专家切换。容量因子、top-k、丢弃和 reroute 共同决定吞吐与质量。

可以记录每个 step 的 expert load 方差：

```math
\sigma^2_{\mathrm{expert}}
=\frac{1}{E}\sum_{e=1}^{E}(n_e-\bar n)^2.
```

它不是质量指标，却能解释为什么 active FLOPs 估算很好看、p99 仍很差。评估要按语言、代码、长上下文和租户切片，因为生产流量可能比训练平均分布更容易产生热点。

## 55.24 Local attention 与 MoE 的联合失败

局部 attention 可能让跨窗口的专家路由看不到所需上下文；MoE router 又可能把相似 token 聚到同一个热点 expert。若系统只做各自的单模块 ablation，无法发现联合失败。应比较 dense FFN + dense attention、MoE + dense attention、dense FFN + local attention 和二者联合，并固定训练和推理预算。

任务要覆盖局部语法、远距复制、专家负载热点、冲突实体、工具 JSON 和跨节点故障。关键指标包括 exact recall、overflow、all-to-all、KV bytes、TTFT/TPOT、p99、回退和单位成功成本。

## 55.25 选择 Mistral/Mixtral 路线的条件

这条路线适合希望用较小 active compute 获得较大总容量、且服务层能够承受 expert dispatch 与局部历史约束的 workload。若部署规模很小、跨卡通信昂贵、任务需要任意远程证据或 expert overflow 频繁，dense baseline 可能更稳。

最终决策要把模型卡的公开结构、目标 engine 的实现、实际 expert histogram、窗口质量和故障回退放在同一报告中。GQA、滑窗和 MoE 都是工程工具，不是“更现代所以必然更好”的结论。

## 55.26 router 热点的容量实验

MoE 容量不能只用 active parameters 估算。给定每个 expert 的 capacity factor `c` 和 batch token 数 `T`，单 expert 的可接纳 token 近似受 `cT/E` 约束；超过容量后可能丢弃、reroute 或 padding。测试应改变语言、代码、长度和 batch，记录 expert histogram、overflow、通信、p99 和质量。

## 55.27 一个专家容量的手算例子

设一轮共有 `T=4096` 个 token、`E=8` 个 expert、top-k 为 2，容量因子为 `c=1.25`。教学上每个 expert 的接纳上限约为 `c*T*k/E`；若 router 极度偏向一个 expert，该 expert 可能溢出，而其他 expert 空闲。系统要决定溢出 token 是丢弃、第二候选重路由、padding 还是回到 dense FFN。

压测不能只看 active FLOPs。还要记录 expert histogram、overflow、all-to-all bytes、跨节点时间、权重驻留、KV、窗口、TPOT 和任务质量。local attention 降低历史交互，MoE 降低每 token 激活，二者的稀疏维度不同；把它们相加成“更省计算”会遗漏通信和负载热点。

## 55.28 小结

Mistral/Mixtral 路线把局部历史访问、共享 KV、稀疏专家和通信成本放在同一个工程问题中。local attention 与 MoE 都叫稀疏，却分别改变信息路径和参数激活；必须按任务、硬件、专家负载、窗口和 fallback 做端到端比较。
