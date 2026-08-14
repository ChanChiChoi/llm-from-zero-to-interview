# 第 22 章 Dense 与 MoE：训练、服务和成本的完整取舍

## 22.1 这不是“哪个模型更先进”的问题

Dense Transformer 和 MoE Transformer 经常被放在一张参数量表里比较，但这种比较很容易误导。Dense 模型的每个 token 都经过相同的参数路径；MoE 模型拥有多个 expert，却只激活其中一部分。二者至少有四个不同的规模口径：总参数、每 token 激活参数、设备常驻参数和一次请求实际消耗的计算/通信。

因此，问题不应是“MoE 一定比 dense 强吗”，而应是：在给定质量目标、训练预算、GPU 拓扑、并发和服务 SLO 下，哪种结构的单位成功任务成本更低，风险更可控。

## 22.2 四本账：容量、计算、显存和通信

设 dense 模型参数量为 `N_d`。对于有 `N_e` 个 expert、每个 token 激活 `k` 个 expert 的 MoE，教学上可以写：

~~~math
N_{\mathrm{total}}
\approx N_{\mathrm{shared}}+N_eN_{\mathrm{expert}},
~~~

而一个 token 的 FFN 激活参数近似为：

~~~math
N_{\mathrm{active}}
\approx N_{\mathrm{shared}}+kN_{\mathrm{expert}}.
~~~

这两个数不等于显存和计算。推理时，若所有 expert 权重都驻留在 GPU，设备仍然要为总参数准备权重存储；即使 expert 权重分片到不同设备，路由和网络也会增加资源需求。可以把设备预算粗略写成：

~~~math
M_{\mathrm{device}}
\approx M_{\mathrm{resident\ weights}}
+M_{\mathrm{KV}}
+M_{\mathrm{workspace}}
+M_{\mathrm{communication}}.
~~~

Dense 的 `M_resident weights` 更直接，MoE 则取决于 expert placement、复制策略、量化、并发和是否允许冷 expert offload。说“active parameters 小，所以显存小”是最常见的错误之一。

## 22.3 训练 compute 为什么不能只按 active parameters 估算

在理想的矩阵乘模型里，MoE 只激活 `k` 个 expert，似乎可以用较低 FLOPs 获得较大容量。但训练还要支付 router、token permutation、capacity padding、通信和梯度同步。

一个教学式训练成本账本可以写成：

~~~math
C_{\mathrm{train}}
=C_{\mathrm{dense\ backbone}}
+C_{\mathrm{expert\ matmul}}
+C_{\mathrm{router}}
+C_{\mathrm{dispatch}}
+C_{\mathrm{combine}}
+C_{\mathrm{optimizer}}.
~~~

`C_expert_matmul` 与激活 token 数有关，`C_dispatch` 与 token 数、hidden size、top-k 和跨设备比例有关，`C_optimizer` 却要处理被更新的 expert 状态。低频 expert 在一个 step 中看到的 token 较少，但它仍然可能占据 optimizer state 和 checkpoint 空间。

训练吞吐的正确分母是有效 token 数和端到端 step 时间，而不是某个 expert kernel 的理论 TFLOPs。若 overflow 丢弃了一部分 token，必须报告“送入 expert 的 token”和“原始训练 token”两个分母。

## 22.4 设备拓扑改变取舍

Dense 模型通常可以用 tensor parallel 把矩阵切分到多张 GPU。MoE 还需要 expert parallel，把不同 expert 放到不同 GPU，再通过 all-to-all 发送 token。相同数量的 GPU，在 NVLink/NVSwitch 紧密互联和跨机柜网络下，可能有完全不同的 MoE 表现。

设本地 dispatch 字节为 `V_local`，跨节点 dispatch 字节为 `V_remote`，带宽分别为 `B_local` 和 `B_remote`，通信时间的粗略下界为：

~~~math
T_{\mathrm{comm}}
\gtrsim\frac{V_{\mathrm{local}}}{B_{\mathrm{local}}}
+\frac{V_{\mathrm{remote}}}{B_{\mathrm{remote}}}
+T_{\mathrm{collective\ overhead}}.
~~~

这个式子没有刻画拥塞、重叠和拓扑，但能解释为什么“相同 FLOPs”不意味着相同延迟。专家分布应尽可能与高速互联拓扑匹配；跨节点 expert routing 的比例、热点 expert 和 collective p99 都应进入容量规划。

## 22.5 Dense 的优势和边界

Dense 模型的主要优势是路径固定。每个 token 使用相同的层和权重，batch 组织、kernel 融合、张量并行、checkpoint、量化和服务调度都比较成熟。

固定路径还让评测更容易：同一输入在相同配置下通常具有更稳定的资源画像。模型扩展时，参数量、FLOPs 和显存虽然仍会变大，但不会增加 expert 选择和 all-to-all 这两个动态变量。

Dense 的代价是容量和计算绑定。要增加总参数，通常也要让每个 token 计算更多矩阵乘；当模型已经很大而目标又需要更多知识容量时，训练和推理预算可能无法接受。

Dense 也不是天然稳定。长上下文、KV cache、量化、并发和多模态 token 仍可能成为瓶颈。选择 dense 不能省略资源账本，只是减少了路由维度。

## 22.6 MoE 的优势和边界

MoE 的核心优势是条件容量：可以在类似每 token 计算预算下拥有更多参数子空间。不同 token 可能形成统计上的 specialization，低频领域不必和所有领域共享完全相同的 FFN 变换。

它的代价至少包括：

1. router 训练和容量 overflow；
2. expert load imbalance 和热点；
3. token dispatch、combine 和 all-to-all；
4. 总权重、optimizer state 和 checkpoint 更大；
5. 小 batch decode 下 expert kernel 不易饱和；
6. 动态路径使 p99、取消、重试和故障回退更复杂；
7. expert 版本、量化和并行布局需要额外兼容性管理。

所以 MoE 的优势更可能在规模化训练和容量受限的任务中显现；在小 batch、强低延迟、网络弱或部署资源有限的场景，dense 可能更有工程优势。

## 22.7 一个匹配预算的比较例子

假设团队有 64 张 GPU，需要在通用文本、代码和数学上达到同一质量目标。方案 A 是一个 dense 模型，方案 B 是一个总参数更大的 MoE，每个 token 只激活两个 expert。

不能只做“总参数 A < 总参数 B”的比较，而要建立三组实验：

| 实验组 | 主要控制 | 回答的问题 |
| --- | --- | --- |
| Active-compute match | 每 token 激活计算和训练 token 接近 | MoE 的额外容量是否带来质量收益 |
| Total-capacity match | 总参数或权重预算接近 | 两种结构的容量使用是否不同 |
| Wall-clock match | GPU 小时、网络和训练时间接近 | 工程预算下谁更划算 |

如果 MoE 在第一组提升，但在第三组因为通信和负载不均没有收益，结论应写成“在计算匹配的离线条件下有效，端到端训练收益待验证”，而不是“MoE 更高效”。

服务侧还要加入短输入、长输入、小 batch、大 batch、专家热点和请求取消等 workload。平均 tokens/s 很可能掩盖长请求触发的 expert p99。

## 22.8 训练阶段的并行策略

Dense 常见的组合是 data parallel、tensor parallel、pipeline parallel、FSDP/ZeRO。MoE 还会引入 expert parallel，实际布局可能是 TP、EP、DP 和 PP 的组合。

选择并行策略时要回答：

1. expert 是否在同一节点，还是跨节点；
2. token dispatch 是否能和 expert GEMM 重叠；
3. pipeline stage 中每层的 expert 数是否均衡；
4. checkpoint 是否按 expert 分片并可恢复；
5. 某个 GPU 失效时是否能重新放置 expert；
6. optimizer state 是否比模型权重更占空间。

在训练平台中，MoE 任务的资源申请不能只写“需要 N 张 GPU”。还应声明拓扑、最小带宽、expert placement、容错策略和可抢占 checkpoint 语义。

## 22.9 服务阶段的并行策略

推理服务的 Dense 模型可以通过多副本扩吞吐，或用 TP 将单副本拆开。MoE 还要决定 expert 是否复制、共享、远端调用或按热点动态放置。

expert 复制可以减少远程通信和热点，但增加权重显存；不复制可以节省存储，却可能让某个专家成为所有请求的队列瓶颈。请求路由器还要考虑 token 数、上下文长度、KV cache 和预计输出，而不是只按请求数 round-robin。

一次 decode step 的系统成本可以粗略写成：

~~~math
T_{\mathrm{step}}
=T_{\mathrm{queue}}
+T_{\mathrm{router}}
+T_{\mathrm{dispatch}}
+T_{\mathrm{expert}}
+T_{\mathrm{combine}}
+T_{\mathrm{KV}}.
~~~

不同 workload 的 `T_expert` 和 `T_dispatch` 分布不同，因此必须给出 p50、p95 和 p99，而非只给平均 step time。

## 22.10 KV cache 与 MoE 不是同一个问题

MoE 主要改变 FFN 参数路径，KV cache 主要由 attention 的层数、KV heads、head dimension、序列长度和 dtype 决定。一个 MoE 模型不会因为 active FFN 参数少就自动拥有较小 KV cache。

若模型同时使用 GQA、MQA 或 MLA，KV 预算才会因 attention 结构改变。比如同一个 MoE 路由配置，换成更低的 `H_kv` 可以减少 KV 读取；但这属于 attention/KV 优化，不应把收益错误归因给 MoE。

资源报告要把以下项目分开：

~~~text
resident expert weights
active expert compute
KV cache bytes
router and dispatch buffers
communication buffers
workspace and fragmentation
~~~

这样才能回答“模型为什么总参数很大却每 token 计算不大”和“为什么 MoE serving 仍然显存紧张”这两个不同问题。

## 22.11 质量退化的诊断路径

当 MoE 分数低于 dense baseline 时，不能直接得出“MoE 不适合”。先按层和任务检查 router 分布，再定位 overflow、专家输出、通信和训练数据。

一个可操作的顺序是：

1. 检查输入 token 数与 tokenizer，排除数据管道差异；
2. 对比 dense 与 MoE 的 shared backbone 输出；
3. 记录每层 expert histogram、route entropy 和 overflow；
4. 做 top-k、capacity factor 和 auxiliary loss 消融；
5. 按语言、代码、数学和长上下文切片；
6. 对热点/冷门 expert 做替换和输出相似度测试；
7. 单独测通信和 kernel，排除质量与系统问题混在一起；
8. 最后再决定是结构问题、训练问题还是服务配置问题。

如果只是某一类低频数据退化，可能需要调整 data mixture 或 expert 初始化；如果所有任务都掉分，优先查路由、模板、权重加载和训练稳定性。

## 22.12 成本模型：从 FLOPs 到单位成功任务

产品决策最终关心的不是模型每秒做多少乘加，而是完成一个有用任务要花多少资源。可以把单位成功任务成本写成：

~~~math
C_{\mathrm{success}}
=\frac{C_{\mathrm{GPU}}+C_{\mathrm{network}}+C_{\mathrm{storage}}+C_{\mathrm{retry}}}
{\max(N_{\mathrm{successful\ tasks}},1)}.
~~~

MoE 可能降低每 token 的矩阵计算，却因 router overflow 或 expert 热点增加重试；也可能提高离线质量，使一次任务更少调用工具或更少重试。只有把成功率、重试、延迟、网络和质量放到同一个分母，才能判断它是否真的便宜。

对于 reasoning 或 Agent workload，输出 token 数、工具调用次数和失败重试可能远大于普通聊天。Dense/MoE 的比较需要在目标业务分布上进行，不能从短问答 benchmark 直接外推。

## 22.13 何时优先 Dense，何时考虑 MoE

优先 Dense 的典型条件是：模型规模中等、batch 小、延迟 SLO 严格、GPU 间网络有限、团队没有成熟 expert parallel 栈，或者系统更重视行为稳定和部署简单。

考虑 MoE 的典型条件是：需要在固定计算预算下增加模型容量；训练 token 很多、batch 足够大；网络拓扑和 expert parallel 已成熟；任务分布多样且有证据说明条件容量有收益；团队可以监控 route、overflow、通信和回退。

两者还可以混合：只在部分层使用 MoE，保留 dense FFN 作为回退；对高风险或结构化任务限制动态路由；对热点 expert 复制，对冷 expert 分片。混合不是自动最优，而是把复杂度集中到最值得的层和任务。

## 22.14 常见反模式

### 只比较 total parameters

总参数可以说明容量上限，却不能说明每 token 计算、设备显存、训练时间或服务成本。

### 只比较 expert kernel FLOPs

忽略 dispatch、all-to-all、padding 和 combine，会把通信和调度成本藏起来。

### 只看平均负载

平均 expert load 可能正常，但 p99 请求都命中同一个热点 expert。要看分桶、时间序列和尾延迟。

### 训练 batch 直接外推线上

训练的 token 数、并发和 kernel 形状通常不同于在线 decode。必须用真实输入长度和输出长度分布压测。

### 把 active 参数当成 resident 参数

未激活的 expert 可能仍驻留在 GPU，或至少在某个 worker 上占据权重显存和通信带宽。

## 22.15 一个可复查的决策表

在模型选型会上，建议把下列字段写入决策记录：

~~~text
quality slices and safety gates
total / active / resident parameters
training GPU-hours and network bytes
expert load p50/p95/p99
overflow and drop policy
prefill / decode TTFT / TPOT
KV cache and workspace budget
replica / TP / EP layout
failure rollback and fallback
unit successful task cost
~~~

每一项都要绑定模型 revision、tokenizer、数据、硬件、runtime 和评测 harness。没有这些条件，表格只是一个不可复现的宣传摘要。

## 22.16 面试回答模板

被问到“Dense 和 MoE 怎么选”时，可以按四层回答：

第一层，定义差异：Dense 所有 token 走同一组参数，MoE 用 router 只激活少数 expert。

第二层，质量和容量：MoE 在相近 active compute 下增加总容量，但 specialization、overflow 和训练稳定性要实证。

第三层，系统代价：MoE 多了 expert parallel、dispatch、all-to-all、负载均衡、权重驻留和尾延迟问题。

第四层，决策证据：在相同质量、训练 GPU 小时、硬件拓扑、真实流量和单位成功任务成本下比较，而不是只看总参数或单个 benchmark。

## 22.17 资料范围与比较条件

MoE 基本路线可参照 GShard（https://arxiv.org/abs/2006.16668）、Switch Transformers（https://arxiv.org/abs/2101.03961）、Mixtral（https://arxiv.org/abs/2401.04088）和 DeepSeek-V2（https://arxiv.org/abs/2405.04434）。关于具体 serving engine 的 expert placement、通信融合、量化和故障回退，论文摘要不能替代源码、版本文档和目标硬件实测。本文的公式用于建立资源账本，不是任何厂商的性能承诺。

## 22.18 Dense/MoE 对比的阶段性结论

Dense 与 MoE 的真正差异，是是否用动态路由把模型容量、每 token 计算和设备资源拆成不同账本。Dense 以固定路径换取简单和稳定；MoE 以路由和通信复杂度换取条件容量。工程决策必须围绕质量、拓扑、SLO、故障边界和单位成功成本，而不是围绕一个更大的参数数字。

## 22.19 Dense/MoE 的容量不能只看 FLOPs

Dense 每个 token 经过固定参数，MoE 每个 token 激活少量 expert，但系统还要保存全部或部分 expert 权重、路由 metadata、通信 buffer、KV 和 workspace。训练中 all-to-all 和负载均衡可能主导 step time；推理中热点 expert 和长上下文可能主导 p99。

容量表至少分解 weight resident、active compute、expert dispatch、通信、KV、workspace、故障余量和回退。total parameters、active parameters 和每卡显存是三个不同口径，不能用一个数字代替。

## 22.20 负载均衡和质量的共同实验

router loss 降低不一定说明任务质量提高；强行均衡可能把语义相近 token 分散到不合适的 expert，丢弃或 reroute 又会改变 token 计算路径。实验要同时改变 capacity factor、top-k、overflow policy 和 expert placement，记录 load histogram、overflow、通信、主任务质量、长尾任务和 p99。

对生产流量，还要做语言、代码、长度、租户和 reasoning 切片。训练平均负载平衡不代表线上均衡，某些领域可能形成固定热点。热点触发的回退、expert replication 或动态调度要计入成本。

## 22.21 何时 Dense 仍是更好的系统

当模型规模中等、部署硬件有限、请求量低、任务需要稳定低尾延迟、工具协议多或团队缺少 expert-parallel 运维能力时，Dense 可能在全链路上更便宜。MoE 的条件容量只有在有效利用专家、通信可承受、故障可恢复和质量收益真实存在时才有价值。

因此选型不是参数规模竞赛，而是对同一 workload 比较：质量/安全验收条件、训练 GPU 小时、拓扑与通信、权重驻留、KV、TTFT/TPOT、p99、overflow、回退和单位成功任务成本。条件结论比“MoE 一定更省”更可靠。
