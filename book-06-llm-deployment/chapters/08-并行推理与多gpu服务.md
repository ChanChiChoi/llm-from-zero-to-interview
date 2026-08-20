# 第八章：并行推理与多 GPU 服务

当模型权重超过单卡显存时，多 GPU 是容量问题的解决方案；当单卡已经能装下模型但 QPS、尾延迟或故障冗余不达标时，多 GPU 又可能是服务编排问题的解决方案。这两个目标经常被混在一起：把八张卡放进同一个进程组，可能让一个模型跑起来，却不一定比四张卡两个副本更快、更便宜或更可靠。

初学者可以先把并行方式理解为四种切法。Tensor Parallel（TP）把一层里的矩阵或 head 切开，多个 GPU 一起完成同一层；Pipeline Parallel（PP）把不同层放到不同 stage，请求依次经过这些 stage；Expert Parallel（EP）把 MoE 的 expert 分散，token 通过 router 在 GPU 之间移动；多副本则不切一个模型，而是复制完整的执行组，让不同请求进入不同副本。

专家需要把问题再拆成四个维度：切分对象是什么，通信发生在哪里，请求的 KV/latent state 归谁，故障时要牺牲什么。TP 可能每层通信，PP 可能每个 stage 传 activation，EP 可能每层 all-to-all，多副本主要付出权重复制和路由成本。没有通信拓扑、请求长度分布和 SLO，单独讨论“TP 还是 PP 更好”没有稳定答案。

本章先解释推理并行与训练并行的目标差异，再按权重、KV Cache、activation、通信和副本容量建立账本，分别展开 TP、PP、EP、多副本、负载均衡、跨节点通信、状态迁移和故障恢复。代码 demo 使用 toy 数字比较容量和通信诊断，不能替代目标 GPU 上的 NCCL trace、kernel profiler 和真实 workload 压测。

## 本章资料边界

本章用 Megatron-LM 和 GPipe 论文解释张量/流水线并行的研究抽象，用 Switch Transformer 解释稀疏 MoE 路由，用 NVIDIA NCCL、TensorRT-LLM、DeepSpeed、vLLM 和 SGLang 官方文档核对当前公开的实现边界。论文实验、官方支持矩阵和本项目实测分别回答不同问题：不要用论文吞吐推断目标拓扑，也不要用 launcher 能启动推断状态迁移和质量正确。

## 推理并行和训练并行有什么不同

训练并行的主要状态包括权重、梯度、optimizer state、activation 和 checkpoint；它可以用 micro-batch、反向传播和较长的同步窗口摊薄通信。推理没有梯度和 optimizer state，但每个在线请求都有自己的 KV/latent/递归 state，且 decode 的每一轮都可能在用户可见路径上。

| 维度 | 训练并行常问 | 推理并行常问 |
| --- | --- | --- |
| 目标 | 收敛时间、训练吞吐、扩展效率 | TTFT、TPOT、goodput、容量和成本 |
| 状态 | 权重、梯度、优化器、checkpoint | 权重、KV/latent state、请求事件 |
| 通信 | 梯度同步、激活传输、参数分片 | 层内 collective、stage activation、expert all-to-all、state 迁移 |
| 调度 | micro-batch 和 pipeline bubble | 变长请求、continuous batching、取消、抢占和优先级 |
| 故障 | 训练作业恢复和 checkpoint 一致性 | 副本摘除、请求重试、状态恢复和副作用幂等 |

训练中一次同步慢一些，可能被后续计算隐藏；在线推理中一次跨节点 collective 可能直接变成用户的 TTFT 或 TPOT 尖峰。因此不能把训练时的并行度、micro-batch 或通信经验直接照搬到 serving。

## 1. 为什么需要多 GPU 推理

多 GPU 推理通常有两类原因，但决策顺序不同。

### 1.1 容量约束

单卡需要同时容纳权重、活动请求的 KV/latent state、激活、通信 buffer、CUDA graph/workspace 和 allocator 余量。一个粗略的容量不等式是：

~~~math
M_{\mathrm{weight}}
+M_{\mathrm{state}}
+M_{\mathrm{buffer}}
\le
\rho M_{\mathrm{gpu}},
\qquad 0<\rho<1.
~~~

M_weight 是本 rank 实际持有的权重和元数据，M_state 是 KV/latent/递归 state，M_buffer 是临时空间，M_gpu 是显存容量，\rho 是为碎片、突发和故障恢复留下的比例。它是容量账本，不是一个内部总开关：即使不等式成立，也可能因为 kernel、通信、P99 或协议不兼容而不适合上线。

容量不足时有几条路线：量化权重或 KV，减少上下文/并发，使用 TP/PP/EP 分片，换更大显存 GPU，或者将模型拆成多个服务阶段。选型时要说明减少的是哪一本账；例如 TP 分摊权重和某些状态，量化减少每元素 bytes，PP 分摊层，副本不会减少单副本容量。

### 1.2 吞吐和可用性约束

模型能装入单卡，不等于单卡能满足目标 QPS。此时增加副本通常比把一个副本继续切成更多 rank 更容易隔离故障，也更容易通过独立队列和负载均衡水平扩展。若单请求 latency 已经被单层计算限制，TP 可能有帮助；若瓶颈是队列或副本数量不足，TP 可能只增加通信。

应把容量与服务目标分开记录：

| 现象 | 第一判断 | 可能路线 |
| --- | --- | --- |
| 权重或 KV 放不下 | 容量问题 | 量化、TP、PP、EP、缩短上下文 |
| 单副本 P99 很高 | kernel/通信/调度问题 | profiling、batch、TP 拓扑、chunked prefill |
| 单副本吞吐不足但延迟可接受 | 水平容量问题 | 增加副本、token-aware routing |
| 某 rank 或某 expert 长尾 | 负载/拓扑问题 | 重分片、路由均衡、节点内通信 |
| 故障后请求重复或丢状态 | 状态契约问题 | 重算、迁移或副本内恢复设计 |

## 2. Tensor Parallel 推理

Tensor Parallel 把同一层的大矩阵、attention head 或相关 hidden 维度切到多个 rank。它不是简单地“每张卡跑一遍模型”，而是让 rank 在一次 forward 内共同完成一个逻辑层。

对线性层 Y=XW，列并行可以写成：

~~~math
W=[W_1,\ldots,W_g],
\qquad
Y=[XW_1,\ldots,XW_g].
~~~

每个 rank 得到一段输出，下一层可能直接消费这段分片，也可能需要 all-gather。行并行则把 W 按输入维度切分，每个 rank 产生部分和：

~~~math
W=
\begin{bmatrix}
W_1\\ \vdots\\ W_g
\end{bmatrix},
\qquad
Y=\sum_{j=1}^{g}X_jW_j,
~~~

这通常需要 all-reduce 或等价的聚合。具体通信点取决于 block 设计、attention/MoE 结构和 runtime。

若 TP 大小为 g_tp，权重切分近似为：

~~~math
M_{w,\mathrm{tp}}
\approx
\frac{M_w}{g_{\mathrm{tp}}}
+M_{\mathrm{replicated}}
+M_{\mathrm{metadata}}.
~~~

M_replicated 包括 norm、embedding、router 或未切分小矩阵，不能默认是零。KV head 若能均匀切分，某个 rank 的 cache 账本可近似为：

~~~math
M_{\mathrm{kv,tp}}
\approx
2LBT_{\mathrm{ctx}}
H_{\mathrm{kv,rank}}D_hb,
\qquad
H_{\mathrm{kv,rank}}\approx\frac{H_{\mathrm{kv}}}{g_{\mathrm{tp}}}.
~~~

L 是层数，B 是 active sequence 数，T_ctx 是上下文 token 数，H_kv 是 KV head 数，D_h 是 head dimension，b 是每个 K/V 元素的字节数。若 H_kv 不能被 g_tp 整除，或模型使用 MQA/GQA/MLA/递归 state，可能发生 head 复制、不同切分或 latent state 分片；必须用模型配置和 runtime layout 校验。

TP 的每层通信可以用一个下界直觉表示：

~~~math
T_{\mathrm{layer}}
\approx
T_{\mathrm{compute}}
+T_{\mathrm{collective}},
\qquad
T_{\mathrm{collective}}
\gtrsim
\frac{V_{\mathrm{comm}}}{B_{\mathrm{link}}}
+T_{\mathrm{launch}}.
~~~

V_comm 是实际 collective 搬运的字节数，B_link 是有效链路带宽，不是规格书峰值带宽。all-reduce、all-gather 的算法、rank 数、消息大小、拓扑、竞争流量和 overlap 都会改变结果。跨节点 TP 如果把 collective 放在每个 decode iteration 的关键路径上，网络尾延迟会直接变成 TPOT 尾延迟。

TP 更适合同一节点内有 NVLink/NVSwitch 等高速互联、模型层内矩阵足够大且单卡容量确实不足的场景。它的代价是每层通信、rank 同步和故障域扩大：一个 rank 退出，整个 TP group 往往都要摘除。选择 TP 时要同时报告每 rank 显存、collective 时间占比、节点内/跨节点链路和副本数量。

## 3. Pipeline Parallel 推理

Pipeline Parallel 按层把模型切成多个 stage。比如 stage 0 持有前半层，stage 1 持有后半层；一个请求的 hidden activation 依次经过这些 stage。若 stage 数为 g_pp，层数为 L，均匀切分时：

~~~math
L_{\mathrm{stage},s}
\approx
\left\lceil\frac{L}{g_{\mathrm{pp}}}\right\rceil.
~~~

权重和按层保存的 KV Cache 通常随 stage 分布：

~~~math
M_{w,\mathrm{pp},s}
\approx
M_{w,s},
\qquad
M_{\mathrm{kv,pp},s}
\approx
M_{\mathrm{kv},s}.
~~~

这里不应简单写成每张卡严格得到 M/g_pp，因为 embedding、lm_head、norm、MoE 层、不同层的 attention 状态和 stage 间 buffer 可能不均匀。PP 的容量价值来自把不同层的常驻状态分开，而不是让每个 rank 都复制完整模型。

单请求的串行路径可粗略表示为：

~~~math
T_{\mathrm{req}}
\approx
\sum_{s=1}^{g_{\mathrm{pp}}}T_s
+\sum_{s=1}^{g_{\mathrm{pp}}-1}T_{\mathrm{comm},s}
+T_{\mathrm{bubble}}.
~~~

T_s 是各 stage 的计算时间，T_comm,s 是传递 activation 的时间，T_bubble 是由于 stage 空闲、动态请求到达或长度不均造成的空洞。训练可用 micro-batch 填充 pipeline，但在线推理的请求会动态到达、输出长度不同，还会被取消或抢占；因此训练 benchmark 的 pipeline utilization 不等于在线单请求延迟。

PP 适合层数多、权重按层分割能解决容量、同一 stage 间通信不如层内 TP 紧密的场景。代价是 stage 顺序和故障域：某个 stage 失败通常使整个 PP replica 不可用；如果把不同租户请求混入 pipeline，还要保证取消和 backpressure 能沿 stage 传播。低并发、强 TTFT SLO 的服务应先测 stage 间通信和 bubble，再决定是否用 PP。

## 4. Tensor Parallel 和 Pipeline Parallel 怎么选

| 场景 | 倾向 |
| --- | --- |
| 单层矩阵很大 | Tensor Parallel |
| 模型层数很多且权重放不下 | Pipeline Parallel |
| 同机高速互联 | Tensor Parallel 更常见 |
| 跨节点通信较慢 | 减少跨节点 TP |
| 在线低延迟 | 尽量降低通信链路 |

选择不是只看“模型层数多不多”，而是看切分后哪种通信进入关键路径。TP 的通信发生在层内，通常频繁但消息形状稳定；PP 的通信发生在 stage 边界，次数较少但请求要顺序经过所有 stage；组合 TP×PP 可以同时切宽度和深度，却扩大 rank group、故障域和调试空间。

实用的判断顺序是：

1. 先用量化和单卡 profiler 确认模型是否能以目标并发放下。
2. 如果模型能放下但吞吐不足，先比较多副本和增大单副本 batch。
3. 如果单层矩阵或 attention kernel 是主要瓶颈，测同机 TP。
4. 如果按层切分是唯一能满足容量的方式，再测 PP 的 TTFT、bubble 和跨 stage latency。
5. 只有当单机 GPU 不够且网络、通信和故障演练都可接受时，才把 TP/PP/EP 扩到跨节点。

每个方案都要记录 gpus_per_replica、replica 数、单卡权重/状态峰值、collective 或 activation 传输时间、P95/P99 和故障恢复时间。理论上能放下只是必要条件，不是选型结论。

## 5. Expert Parallel for MoE

MoE 模型有多个 expert，但每个 token 通常只激活 top-k 个 expert。若 router 为 token x_i 选出集合 E_i，并给出权重 p_i,e，MoE 层可抽象为：

~~~math
y_i=\sum_{e\in E_i}p_{i,e}f_e(x_i).
~~~

Expert Parallel 把不同 expert 放到不同 GPU。一个 iteration 需要经历 dispatch、expert compute 和 combine 三步：先把 token 发到拥有目标 expert 的 rank，完成局部 expert 计算，再把结果发回原 token 的位置。若 expert 与 token 不在同一 rank，通信通常表现为 all-to-all 或其分解。

设本轮 token 数为 N，top-k 为 k，expert e 收到 n_e 个 token，则：

~~~math
\sum_{e=1}^{E}n_e=kN,
\qquad
\bar n=\frac{kN}{E},
\qquad
R_{\mathrm{imbalance}}=\frac{\max_e n_e}{\bar n}.
~~~

R_imbalance 越高，最热 expert 越可能决定这一轮的完成时间；但它还没有包含跨节点链路、capacity padding、expert kernel shape 和 combine 顺序。若启用 capacity factor c，某个 expert 可接收的上限通常可近似为 c\bar n；超出的 token 可能被丢弃、重路由、padding 或采用 dropless 实现，四种选择对质量和吞吐不同。

EP 的收益是让总参数量超过单卡容量，同时每个 token 只激活部分计算；难点是 router 动态改变通信和负载。推理时请求长度、语言和任务分布可能造成与训练不同的 expert 热点，不能只看训练期间的 load balance loss。线上应记录每个 expert 的 token 数、排队时间、dispatch/combine bytes、跨节点比例、丢 token/重路由率和 P99。

EP 适合 expert 数量大、单个 expert 权重无法在每卡复制、节点内或专用网络能承受 all-to-all 的场景。若 expert 很少、请求并发低或跨节点网络慢，复制部分 expert、使用副本或改变 TP/EP 组合可能比纯 EP 更稳。

## 6. 多副本服务

多副本的含义不是“多启动几个进程”，而是复制一组可以独立接收请求的执行单元。这个执行单元可能是一张 GPU，也可能是一个由 TP、PP 或 EP 组成的 GPU group。副本内部的 rank 必须协同完成一个请求，副本之间则尽量只通过路由层交换请求和结果。

例如，四卡 TP 副本在八卡机器上可以这样放置：

```text
replica 0: GPU 0-3, TP=4
replica 1: GPU 4-7, TP=4
```

这和一个八卡 TP 副本的差别，不只是副本数量不同。前者有两个独立的队列和两个故障单元，后者只有一个队列，但能把更大的模型或更高的单副本 batch 放在同一个并行组内。

### 6.1 副本数量、容量和吞吐

若每个副本需要 g_rep 张 GPU，机器上有 G 张可用 GPU，忽略预留卡和碎片时，物理副本上限为：

```math
R_{\mathrm{physical}}
=
\left\lfloor\frac{G}{g_{\mathrm{rep}}}\right\rfloor .
```

真正用于承载流量的副本数还要扣除故障或滚动升级所需的余量：

```math
R_{\mathrm{active}}
=
\max\left(
0,
\left\lfloor
\frac{G-G_{\mathrm{reserved}}}{g_{\mathrm{rep}}}
\right\rfloor
\right).
```

G_reserved 不是浪费掉的 GPU，而是为了节点维护、故障替换或流量突增保留的服务能力。没有冗余要求的离线批处理可以令它为零；在线服务若把所有卡都打满，任何一次故障都可能让可用容量突然跌到目标以下。

如果单副本在某个 workload 上的有效吞吐是 S_rep，理想情况下有：

```math
S_{\mathrm{aggregate}}
\approx
\sum_{r=1}^{R_{\mathrm{active}}}S_r
\approx
R_{\mathrm{active}}S_{\mathrm{rep}}.
```

这个近似只有在请求能够均匀分布、每个副本的模型和调度配置一致、网关不成为瓶颈时才成立。真实服务还要减去负载倾斜、冷启动、限流、网络和故障余量。对于流式生成，吞吐还应区分 output tokens/s、完成请求数/s 和满足 SLO 的 goodput；只报一个 aggregate TPS 容易掩盖长请求拖慢尾延迟的事实。

容量也不能只用副本数量相加。先把 token、KV 和队列限制都换算成同一种可接纳工作量，例如“等价 token 数”；若副本 r 的三种约束分别为 C_r,token、C_r,kv 和 C_r,queue，则：

```math
C_{\mathrm{service}}
=
\sum_{r=1}^{R_{\mathrm{active}}}C_r,
\qquad
C_r=\min\left(
C_{r,\mathrm{token}},
C_{r,\mathrm{kv}},
C_{r,\mathrm{queue}}
\right).
```

这里的 min 表示三个已经统一单位的数值约束中最先耗尽的那一项，而不是把三种容量直接相加。一个副本显存富余，但队列已达到 backpressure 上限时，继续把请求发过去仍然会损害延迟。

### 6.2 为什么副本常常比继续增大并行度简单

当单副本已经能装下模型时，增加副本有三个直接收益。第一，每个副本拥有独立的请求队列，短请求不会被同一个长请求完全挡住。第二，副本是天然的故障隔离单元，单个 TP group 失效时不必摘除所有 GPU。第三，横向扩容的边界清晰，路由器可以根据负载把新请求送往健康副本。

但副本不会降低单副本的权重显存，也不会让一条已经被单层矩阵计算限制的请求自动变快。把一张 80 GiB 的卡变成两张副本，得到的是两份约 80 GiB 的权重；它不能承载一份需要 120 GiB 的模型。反过来，把两个副本合成一个 TP group，可能解决容量问题，却会让通信和故障域变大。

可以用下面的现象区分两类决策：

| 现象 | 更像什么问题 | 首先比较 |
| --- | --- | --- |
| 权重或活动 KV 放不下 | 单副本容量问题 | 量化、TP、PP、EP、上下文预算 |
| 单副本已经满足延迟，但 QPS 不足 | 水平容量问题 | 多副本、队列上限、token-aware routing |
| 单层 GEMM 或 attention kernel 占主要时间 | 单请求计算问题 | 同机 TP、kernel、batch 形状 |
| 等待时间占 P99 的大部分 | 排队和路由问题 | 调度、负载均衡、backpressure |
| 一个 rank 或 expert 明显长尾 | 负载和拓扑问题 | 重新分片、路由均衡、节点内通信 |

### 6.3 副本一致性和生命周期

副本可以独立排队，但不能各自解释同一份模型协议。至少要绑定以下版本信息：

1. 权重和量化 artifact 的版本。
2. tokenizer、chat template 和 stop token 配置。
3. adapter、工具 schema、结构化输出 grammar 的版本。
4. sampling 默认值、最大上下文和最大输出限制。
5. runtime、kernel、CUDA/NCCL 和并行布局。

滚动升级时，先将旧副本标记为 draining，不再接收新请求，但允许已有请求在预算内完成；新副本通过健康检查后再加入路由池。健康检查不能只调用一个返回固定文本的接口，还应验证权重加载、一次短 prefill、一次 decode、stream 结束标记和结构化输出协议。否则“进程存活”可能被误认为“副本可服务”。

权重在多个副本之间重复保存是多副本最明显的成本。操作系统页缓存、容器镜像层或 GPU peer access 可能减少加载时间，但通常不能把独立 GPU 上的可执行权重当成一份共享显存。计划成本时仍应按副本账本估算，并把冷启动加载、CUDA graph capture 和 prefix cache 预热单独计入。

## 7. Load balancing

负载均衡的任务是把请求送到“能够在约束内尽快完成”的副本，而不是让每个副本收到相同数量的请求。LLM 请求的计算和显存成本差异很大：一个 100k token 的长上下文请求可能比数十个短请求更占用 prefill 算力和 KV block；一个正在生成的长输出又会长时间占住 decode 槽位。因此 round-robin 只能作为没有负载信息时的基线。

### 7.1 先估计请求成本

对请求 i，可以先用 prompt token 数 P_i、预计输出 token 数 Y_i、当前已占用的 KV token K_i 和工具/多模态附加成本构造一个粗略分数：

~~~math
\widehat C_i
=
\alpha P_i
+\beta \widehat Y_i
+\gamma K_i
+\delta C_{i,\mathrm{extra}}.
~~~

P_i 主要影响 prefill，Y_i 影响 decode 持续时间，K_i 反映已经占用的 state，C_extra 可以表示图像 token、检索重排或工具等待等额外资源。alpha、beta、gamma、delta 不是通用常数，应通过 trace 拟合或压测校准；如果没有可靠的输出长度预测，使用 max_new_tokens 只能提供保守上界，不能当成真实完成长度。

副本 r 的负载可以写成：

~~~math
\widehat L_r
=
\lambda_q Q_r
+\lambda_t A_{r,\mathrm{tokens}}
+\lambda_k K_{r,\mathrm{used}}
+\lambda_p P_{r,\mathrm{pending}}
+\lambda_f F_r.
~~~

Q_r 是排队请求数，A_tokens 是活动请求的预计剩余 token 负载，K_used 是 KV 使用量，P_pending 是等待 prefill 的 token 数，F_r 是故障、降级或冷启动惩罚。将不同单位相加前要做归一化；否则数值最大的字段会无意中支配路由。

最简单的选择规则是：

~~~math
r^\star
=
\arg\min_{r\in\mathcal H}\left(
\widehat L_r
+\eta\frac{\widehat C_i}{\widehat C_{r,\mathrm{remaining}}}
+\mu\widehat D_{i,r}
\right),
~~~

H 是当前健康且未超过队列/租户限制的副本集合，C_r,remaining 是用同一单位表示的副本 r 剩余容量，D_i,r 是请求 i 在副本 r 上的预计等待或数据搬运代价。eta 和 mu 体现服务对请求成本及 locality 的重视。路由器不必实现精确排队模型，但必须明确自己是在优化等待时间、KV 空间、prefix 命中率，还是它们的组合。

### 7.2 常见策略和适用边界

| 策略 | 看到的信息 | 优点 | 典型问题 |
| --- | --- | --- | --- |
| Round-robin | 副本编号 | 简单、均匀发请求 | 忽略 token 和队列，长请求会造成倾斜 |
| Least connections | 活动请求数 | 比请求轮询稍好 | 一个长请求和一个短请求权重相同 |
| Least queue time | 排队等待估计 | 直接对应延迟 | 依赖准确的服务时间估计 |
| Token-aware | prompt、剩余 token、KV | 更符合 LLM 资源消耗 | 需要统计和预测，估计错误会反噬 |
| Prefix-aware | 前缀缓存位置 | 可减少重复 prefill | 可能把热点集中到一个副本 |
| Tenant-aware | 租户配额和优先级 | 便于隔离资源 | 需要和公平调度、限流协同 |

真实系统经常组合使用这些策略。例如先按租户配额过滤副本，再在健康集合中用 token-aware score 选择，若请求前缀命中某个副本则给予有限的 locality bonus。locality bonus 必须有上限，否则一个缓存热点会持续吸走流量，最终形成“缓存命中率高但 P99 更差”的自激现象。

### 7.3 路由器和副本内调度器的边界

网关能知道副本队列和粗粒度 token 负载，但通常不知道下一次 iteration 中哪些请求会被抢占；副本内调度器知道 block table、prefill chunk 和 decode 槽位，却不适合承担全局租户配额。两层应约定同一套单位：

1. 网关报告 pending prompt tokens、active sequences、预计剩余输出和 KV 使用比例。
2. 副本在达到 token、KV 或队列上限时返回 backpressure，而不是继续接收后再超时。
3. 路由器对 backpressure 做短暂冷却，避免所有请求在多个副本之间来回试错。
4. 副本内调度器负责 iteration-level 的公平性，网关不应只按连接数反复重排流式请求。

重试是路由的一部分。只有尚未产生外部副作用的请求，才适合在网关层透明重试；流式响应已经发送一部分后，重试可能导致客户端看到重复文本。对于工具调用、写数据库或扣款请求，必须携带幂等键并由下游判断是否已经提交，不能把“HTTP 请求失败”简单等同于“业务没有发生”。

## 8. 跨节点通信瓶颈

多机推理的关键不是机器数量，而是哪些字节、以多高频率、在什么关键路径上跨过节点边界。节点内可能有 NVLink 或 NVSwitch，节点间则可能经过 PCIe、NIC、RDMA 或其他网络路径；具体能力取决于硬件、拓扑、驱动、NCCL 配置和集群竞争，不能从 GPU 型号单独推断。

### 8.1 用延迟和带宽拆开通信成本

对一次消息传输，一个有用的第一阶近似是：

~~~math
T_{\mathrm{comm}}(V)
\approx
\alpha_{\mathrm{path}}
+\frac{V}{B_{\mathrm{eff}}},
~~~

V 是实际传输字节数，alpha_path 是启动、同步和协议开销，B_eff 是考虑拓扑、协议和竞争后的有效带宽。collective 还要乘上与 rank 数、算法和通信模式有关的系数；因此这个式子是定位瓶颈的账本，不是 NCCL 的性能预测器。

TP 可能在每层触发 all-reduce 或 all-gather，EP 可能在每个 MoE 层触发 dispatch/combine all-to-all，PP 通常在 stage 边界传递 activation。若这些操作处于 decode 的关键路径，它们会反复贡献 TPOT；若能与计算重叠，表面上的 kernel 时间可能看不出网络成本，但尾延迟仍可能受到最慢 rank 影响。

可以把一个 decode iteration 的关键路径写成：

~~~math
T_{\mathrm{iter}}
\approx
\max_r T_{\mathrm{compute},r}
+T_{\mathrm{critical\ comm}}
+T_{\mathrm{sync}},
~~~

其中 max_r 表示最慢 rank 的计算，T_critical comm 只包含不能被计算隐藏的通信。GPU 利用率高并不代表系统高效：如果 rank 在等待 all-to-all 时仍有少量 kernel 活动，利用率看起来不低，用户的 TPOT 却已经变差。

### 8.2 拓扑映射比并行度数字更重要

常见的部署倾向是把强同步、频繁通信的 TP 尽量放在同一节点，把 PP 的粗粒度 stage 边界或副本边界放到跨节点位置；MoE 的 EP 则要看 expert 热点和 all-to-all 能否局部化。这个倾向不是规则：如果单层矩阵太大，跨节点 TP 可能是容量上唯一可行的方案；如果 PP stage 极度不均衡，减少网络次数也救不了尾延迟。

一个八卡、两节点的例子可以有两种映射：

~~~text
方案 A：每节点 4 卡组成一个 TP group，两个 TP group 作为两个副本。
方案 B：8 卡组成一个跨节点 TP group，只有一个副本。
~~~

方案 A 通常把层内 collective 留在节点内，并获得副本级故障隔离；方案 B 可能能容纳更大的单副本状态，但每个请求的层内通信都依赖节点间链路。最终选择要用真实拓扑和真实请求长度测量，而不是用 GPU 数量比较。

### 8.3 线上应该观察什么

至少应把 GPU 计算、collective、NIC、队列和用户指标放在同一条 trace 中：

1. 每种 collective 的调用次数、消息大小、耗时和 overlap 比例。
2. 节点内与跨节点流量，以及最慢 rank 与中位 rank 的差值。
3. dispatch/combine 的 token 数、丢弃/重路由数和 expert 热点。
4. prefill 与 decode 分别的 TTFT、TPOT、P95/P99。
5. 网络抖动、重传、NCCL error、rank restart 和请求重试次数。

只看平均带宽会漏掉小消息的启动开销，只看平均延迟会漏掉最慢 rank，只看 GPU utilization 会漏掉队列和网络等待。拓扑压测还要覆盖混合长度、低并发单请求、高并发 continuous batching 和一个 rank 被暂停的异常场景。

## 9. KV Cache 在多 GPU 中怎么处理

并行推理不仅要切权重，还要明确每一份请求状态由谁保存。权重是副本级或 rank 级的常驻 artifact，KV Cache 是请求级、随 token 增长的动态状态；两者的生命周期、复制成本和故障恢复方法完全不同。

### 9.1 不同并行方式的状态布局

| 并行方式 | 权重/计算布局 | KV 或 latent state 的常见归属 | 主要风险 |
| --- | --- | --- | --- |
| TP | 一个层的参数和计算跨 rank | 按 KV head、hidden 或 runtime block 分片，也可能复制部分 head | head 不能整除、复制比例和 collective 不一致 |
| PP | 不同层在不同 stage | 跟随对应层所在 stage | stage 间状态不能被错误拼接，stage 故障影响整组 |
| EP | expert 按 rank 分布，主干可能另行切分 | 通常由 attention/主干布局决定，不等于 expert 权重归属 | 把 expert 分片误当成 KV 分片 |
| 多副本 | 每个副本拥有完整执行组 | 每个副本私有；prefix cache 也通常是局部的 | 请求迁移需要重算或显式搬运 |

对标准 attention，如果一个 TP rank 持有 H_kv,rank 个 KV head，其粗略 cache 账本为：

~~~math
M_{\mathrm{kv},r}
\approx
2LBT_{\mathrm{ctx}}H_{\mathrm{kv},r}D_hb.
~~~

2 表示 K 和 V 两份，L 是层数，B 是活动 sequence 数，T_ctx 是每条 sequence 已占用的 token 数，D_h 是 head dimension，b 是每个元素的字节数。MHA、GQA、MQA、MLA 或递归 state 会改变 H_kv,r 和 state 的含义；不能因为 runtime 使用 TP 就默认 cache 一定按完整 hidden 维度均匀切分。

PP 下，stage s 只需维护自己负责的层的状态，但它仍然要为每一个经过该 stage 的请求保留位置和 block table。于是 stage 间的容量不一定均匀：长上下文和层数分布、embedding/lm_head 放置方式、不同 attention 结构都可能造成某个 stage 先满。工程上应看每 stage 的 cache 使用峰值，而不是只看全局平均值。

### 9.2 prefix cache 和正在生成的 state 不是一回事

可以复用的 prefix cache 通常对应已经确认的 token 前缀；正在生成的请求还包含当前位置、未完成的 block、采样状态、停止条件和可能的 speculative 分支。前者可以在相同模型版本和模板下作为只读共享候选，后者必须由一个明确的请求 owner 管理。

因此，负载均衡通常会在“最轻的副本”和“已有前缀的副本”之间做取舍。迁移一个已有 prefix 的请求，如果目标没有该 prefix，就会重新支付 prefill；为了命中缓存而把请求送到已经拥堵的副本，也可能使端到端延迟更差。这个选择应记录 cache hit、等待时间和重算 token，而不是只看命中率。

### 9.3 状态搬运的成本和边界

若需要迁移 V_state 字节的 KV 或 latent state，可以先用下面的账本估算：

~~~math
T_{\mathrm{move}}
\approx
T_{\mathrm{serialize}}
+\frac{V_{\mathrm{state}}}{B_{\mathrm{eff}}}
+T_{\mathrm{deserialize}}
+T_{\mathrm{verify}}.
~~~

只有在它小于从输入重新 prefill 的时间，并且目标布局兼容时，迁移才可能有收益：

~~~math
T_{\mathrm{move}}+T_{\mathrm{risk}}
<
T_{\mathrm{recompute}}.
~~~

T_risk 表示版本不一致、校验失败、并发写入和隐私擦除等工程风险的预留成本。实际系统不必把它当成精确概率，但必须把“迁移更快”从口号变成可观测的比较。

迁移前至少校验模型 revision、tokenizer/template revision、位置编码参数、dtype、KV block layout、adapter、采样 RNG 和 speculative tree。KV Cache 不是普通的无语义显存块；把不兼容的 cache 当成可复用结果，可能产生静默的错误文本，而不是立刻报错。

## 10. 多 GPU 服务的故障处理

多 GPU 服务的故障处理要先区分故障范围，再决定对请求做什么。GPU kernel hang、单 rank 进程退出、GPU 与 NIC 链路异常、整机断电、runtime OOM、网关超时和下游工具失败，不是同一种事件。它们的探测时间、可恢复状态和对用户的影响都不同。

### 10.1 从健康检查到摘除

应把 liveness、readiness 和 serving health 分开：

1. liveness 只说明进程仍能响应管理请求。
2. readiness 说明副本已加载正确 artifact，能够接收新请求。
3. serving health 还要覆盖一次短 prefill、一次 decode、流式结束和目标协议的结构化响应。

TP、PP 或 EP group 中任何一个 rank 失效，都可能使整个执行组无法继续完成 collective。此时不应只重启挂掉的 HTTP worker，而应由编排层将整个 group 标记为 draining 或 unavailable，停止分配新请求，并记录剩余请求的状态位置。否则新请求仍会进入一个永远等不到 collective 的队列。

### 10.2 在途请求的恢复语义

恢复动作取决于请求是否已经产生外部副作用：

| 请求阶段 | 可重试动作 | 需要向客户端说明 |
| --- | --- | --- |
| 尚未开始执行 | 换健康副本重新排队 | 通常不需要额外说明 |
| prefill 已完成、尚未返回 token | 可重算或迁移 state | TTFT 可能增加 |
| 已流式返回部分 token | 不能无条件拼接第二次结果 | 需要断点、重连或明确失败 |
| 已发起工具/写操作 | 先按幂等键查询下游 | 不能把超时当成未执行 |

对于纯文本请求，最可靠的恢复路径往往是保留已确认输入和输出前缀，在新副本重新 prefill；它牺牲计算换取协议简单。对于长上下文 Agent，状态迁移能减少重算，但需要更严格的版本和 checksum 约束。对于支付、写库、发消息等操作，模型生成和外部副作用应拆成两个可确认阶段，不能让推理层的自动重试直接再次调用工具。

重试还要有预算：每个请求最多重试次数、总重试时间和退避窗口必须有限。没有上限的重试会在 GPU 故障时把故障流量再次放大，形成 retry storm；客户端、网关和副本内部也应避免各自独立重试同一请求。

### 10.3 降级和恢复演练

当可用 GPU 减少时，系统可以采取降低最大输出、限制长上下文、关闭高成本 speculative 分支、把低优先级请求排队，或切换到已验证的小模型。降级的目标是减少资源消耗，但不能绕过安全策略、租户隔离和工具权限。

恢复能力应通过演练验证，而不是从进程重启成功推断。至少演练：

1. 一个 TP rank 退出，确认 group 摘除、请求重算和新副本加入。
2. 一个 PP stage 延迟，确认 backpressure 能传播而不是无限积压。
3. 网络短暂抖动，确认 collective 超时不会被无界重试放大。
4. cache checksum 或版本不匹配，确认系统丢弃 state 并回退重算。
5. 流式响应中断和工具调用超时，确认客户端和下游都不会收到重复副作用。

要记录恢复时间、丢失的已生成 token、重算 token、重试次数、恢复后的 P99 和故障期间的 goodput。这样才能区分“服务进程重新启动了”和“用户请求以可解释的语义恢复了”。

## 11. 状态所有权、请求迁移和故障恢复

多 GPU 部署最容易被忽略的是“请求状态属于谁”。权重可以在副本之间复制，KV Cache、递归 state 和 speculative 分支却属于某个具体请求和某个具体版本。对于请求 `r`，可以把状态写成：

```math
\Sigma_r=(\mathrm{replica},\mathrm{layer\_range},\mathrm{position},
\mathrm{kv\_blocks},\mathrm{state\_version},\mathrm{rng\_state})
```

只有这些字段与目标 worker 的模型、tokenizer 和协议版本一致时，迁移才是安全的。单纯把“剩余 prompt”重新发送给另一副本可能得到不同结果，也会重新支付 prefill；直接复制显存则受到设备拓扑、dtype、cache layout 和并发写入的限制。

### 11.1 三种恢复路径

1. **重试并重算**：只保留输入和已确认输出，在新副本重新 prefill。实现简单，适合短请求，但长上下文会显著增加 TTFT。
2. **迁移 cache/state**：转移 block table、KV 或递归 state。延迟较低，但要求源和目标的 layout、位置和版本完全兼容。
3. **副本内恢复**：只在同一并行组重新拉起 rank。数据移动少，但故障域较大，整组 GPU 仍可能不可用。

生产系统应根据请求类型选择路径：普通问答可以重算，长 RAG 或长周期 Agent 值得投资状态迁移，高风险工具事务则必须先冻结副作用状态，再决定是否重试。无论哪条路径，都要让用户知道是否发生了降级，并保证不会重复提交不可逆写操作。

### 11.2 拓扑选择的可验证条件

选择 TP、PP、EP 或多副本时，至少用真实拓扑测量四项：跨 GPU 通信时间、单请求延迟、混合长度吞吐和故障恢复时间。理论显存能放下模型不代表 SLO 可满足；同样，GPU 利用率高也可能是 all-to-all 或 PCIe 拥堵造成的假繁忙。上线前应做 rank 故障、网络抖动、cache checksum 错误、请求取消和 batch reorder 演练。

## 12. 并行推理选型流程

选型要先回答“单副本的瓶颈是什么”，再回答“切哪一维”。可以按下面的顺序收集证据：

1. 先建立单卡或单副本的权重、KV、buffer 和显存余量账本，确认目标并发下是否放得下。
2. 如果模型能放下但吞吐不足，比较增大单副本 batch、增加多副本和优化调度后的 goodput。
3. 如果单层矩阵或 attention kernel 占主要计算时间，测同机 TP，并同时记录 collective 占比。
4. 如果按层切分是解决容量的必要条件，测 PP 的 stage 平衡、activation 传输、TTFT 和低并发延迟。
5. 如果模型是 MoE，再把 expert token 分布、all-to-all、capacity 策略和热点 expert 纳入测量。
6. 只有在单节点 GPU 不够且跨节点通信、故障恢复和版本部署都可解释时，才扩展到跨节点 TP/PP/EP。

可以把方案记录成一张配置表，而不是只写“使用八卡推理”：

| 字段 | 必须回答的问题 |
| --- | --- |
| gpus_per_replica | 一个请求要占用多少张卡和哪些故障域 |
| TP/PP/EP | 哪些权重、state 和 token 在 rank 间移动 |
| replica 数 | 有多少独立队列，是否保留故障余量 |
| 显存峰值 | 权重、KV、workspace、通信 buffer 各占多少 |
| 通信路径 | collective 或 activation 是否跨节点，能否和计算重叠 |
| workload | prompt/output 分布、并发、租户和流式比例 |
| 结果 | TTFT、TPOT、goodput、P99、恢复时间和单位成本 |

选型结论必须附带 workload、硬件拓扑、runtime 版本和测量日期。否则“TP 比 PP 快”只能是脱离条件的句子，无法迁移到另一种模型或集群。

## 13. 最小 Python demo：多 GPU 推理选型的 toy 账本

下面的 0 依赖 demo 用 toy 数字比较单卡、多副本、TP 和 TP+PP 的显存、通信和容量。它不代表真实 benchmark，只演示如何把权重、KV Cache、通信和副本数放到同一张部署账本里。

```python
from math import ceil


MODEL = {
    "params_b": 70,
    "layers": 80,
    "hidden": 8192,
    "kv_heads": 8,
    "head_dim": 128,
    "bytes_weight": 2,
    "bytes_kv": 2,
    "bytes_activation": 2,
}

WORKLOAD = {
    "active_seq": 24,
    "ctx_tokens": 4096,
    "gpu_mem_gib": 80,
    "reserve_gib": 8,
    "num_gpus": 8,
}

PLANS = [
    {"name": "single_gpu", "tp": 1, "pp": 1, "replicas": 8},
    {"name": "tp4_two_replicas", "tp": 4, "pp": 1, "replicas": 2},
    {"name": "tp8_one_replica", "tp": 8, "pp": 1, "replicas": 1},
    {"name": "tp4_pp2_one_replica", "tp": 4, "pp": 2, "replicas": 1},
]


def gib(num_bytes):
    return num_bytes / (1024 ** 3)


def estimate(plan):
    tp, pp = plan["tp"], plan["pp"]
    gpus_per_replica = tp * pp
    gpu_feasible = plan["replicas"] * gpus_per_replica <= WORKLOAD["num_gpus"]
    weights = gib(MODEL["params_b"] * 1e9 * MODEL["bytes_weight"] / gpus_per_replica)
    layers_per_stage = ceil(MODEL["layers"] / pp)
    kv_heads_per_rank = ceil(MODEL["kv_heads"] / tp)
    kv = gib(
        2
        * layers_per_stage
        * WORKLOAD["active_seq"]
        * WORKLOAD["ctx_tokens"]
        * kv_heads_per_rank
        * MODEL["head_dim"]
        * MODEL["bytes_kv"]
    )
    total = weights + kv + WORKLOAD["reserve_gib"]
    fits = gpu_feasible and total <= WORKLOAD["gpu_mem_gib"]
    tp_comm_mib = 0.0
    if tp > 1:
        tp_comm_mib = 2 * MODEL["hidden"] * MODEL["bytes_activation"] * WORKLOAD["active_seq"] / 1024 ** 2
    pp_comm_mib = (pp - 1) * MODEL["hidden"] * MODEL["bytes_activation"] * WORKLOAD["active_seq"] / 1024 ** 2
    latency_proxy = round(1.0 + 0.06 * (tp - 1) + 0.10 * (pp - 1), 2)
    aggregate_capacity = plan["replicas"] * WORKLOAD["active_seq"] if fits else 0
    return {
        "gpus_per_replica": gpus_per_replica,
        "replicas": plan["replicas"],
        "weights_gib": round(weights, 2),
        "kv_gib": round(kv, 2),
        "total_gib": round(total, 2),
        "gpu_feasible": gpu_feasible,
        "fits": fits,
        "tp_comm_mib_per_decode": round(tp_comm_mib, 3),
        "pp_comm_mib_per_decode": round(pp_comm_mib, 3),
        "latency_proxy": latency_proxy,
        "aggregate_active_seq": aggregate_capacity,
    }


report = {plan["name"]: estimate(plan) for plan in PLANS}
for name, row in report.items():
    print(name, row)

feasible_names = [name for name, row in report.items() if row["fits"]]
capacity_choice = (
    max(feasible_names, key=lambda name: report[name]["aggregate_active_seq"])
    if feasible_names
    else None
)
print("capacity_choice=", capacity_choice)

replica_load = [
    {"id": "replica_0", "queued_tokens": 3200, "active_seq": 11},
    {"id": "replica_1", "queued_tokens": 900, "active_seq": 8},
]
new_request = {"prompt": 1200, "max_new": 256}
for replica in replica_load:
    available_slots = max(1, WORKLOAD["active_seq"] - replica["active_seq"])
    request_cost = 0.5 * new_request["prompt"] + new_request["max_new"]
    replica["score"] = (
        replica["queued_tokens"]
        + 32 * replica["active_seq"]
        + request_cost / available_slots
    )
print("route_to=", min(replica_load, key=lambda x: x["score"])["id"])
```

一组可能输出：

```text
single_gpu {'gpus_per_replica': 1, 'replicas': 8, 'weights_gib': 130.39, 'kv_gib': 30.0, 'total_gib': 168.39, 'gpu_feasible': True, 'fits': False, 'tp_comm_mib_per_decode': 0.0, 'pp_comm_mib_per_decode': 0.0, 'latency_proxy': 1.0, 'aggregate_active_seq': 0}
tp4_two_replicas {'gpus_per_replica': 4, 'replicas': 2, 'weights_gib': 32.6, 'kv_gib': 7.5, 'total_gib': 48.1, 'gpu_feasible': True, 'fits': True, 'tp_comm_mib_per_decode': 0.75, 'pp_comm_mib_per_decode': 0.0, 'latency_proxy': 1.18, 'aggregate_active_seq': 48}
tp8_one_replica {'gpus_per_replica': 8, 'replicas': 1, 'weights_gib': 16.3, 'kv_gib': 3.75, 'total_gib': 28.05, 'gpu_feasible': True, 'fits': True, 'tp_comm_mib_per_decode': 0.75, 'pp_comm_mib_per_decode': 0.0, 'latency_proxy': 1.42, 'aggregate_active_seq': 24}
tp4_pp2_one_replica {'gpus_per_replica': 8, 'replicas': 1, 'weights_gib': 16.3, 'kv_gib': 3.75, 'total_gib': 28.05, 'gpu_feasible': True, 'fits': True, 'tp_comm_mib_per_decode': 0.75, 'pp_comm_mib_per_decode': 0.0, 'latency_proxy': 1.28, 'aggregate_active_seq': 24}
capacity_choice= tp4_two_replicas
route_to= replica_1
```

这段 demo 的结论是：

1. 单卡方案虽然副本最多，但 70B FP16 权重和 KV Cache 放不下；gpu_feasible 只表示 GPU 数量够用，fits 还要经过显存账本。
2. `tp4_two_replicas` 每个副本 4 卡，可在 8 卡机器上放两个副本，toy 容量高于单个 8 卡副本。
3. `latency_proxy` 是人为构造的通信惩罚示意，不是目标 GPU 的实测延迟。
4. `route_to` 不按 round-robin，而是按 queued tokens、active sequence、可用槽位和新请求成本做简化路由。

## 14. 一个 8 卡集群的并行部署决策

考虑一台有 8 张 80 GiB GPU 的机器，准备部署一个 70B 模型。服务有两类 workload：短交互请求要求低 TTFT 和高可用性，长上下文请求需要 24 个活跃 sequence；峰值期间不能因为一张卡或一个进程失效就让所有请求同时中断。团队提出三个方案：4 卡 TP 放一个副本并部署两个副本；8 卡 TP 放一个副本；4 卡 TP 与 2 stage PP 组成一个副本。

先算容量，再谈速度。假设权重使用 FP16，70B 权重约为 130.4 GiB；在 demo 的 toy workload 下，每个完整执行组还需要约 30 GiB 的状态、workspace 和余量。单卡方案的权重本身就超过 80 GiB，不能通过增加副本解决；副本复制的是完整权重，不会把一份 130 GiB 的模型拆小。

TP4 双副本把权重和部分状态分摊到 4 张卡上，每张卡的 toy 总账约为 48.1 GiB，容量上可行，并且 8 张卡形成两个独立故障单元。一个 TP group 内的 all-reduce 或 all-gather 仍然可能进入每层关键路径，但如果 4 张卡在同一节点并由高速互联连接，通信代价可能可接受。两个副本各有独立队列，路由器可以把长请求和短请求分开，也能在一个 group draining 时保留另一个 group 服务部分流量。

TP8 单副本的每卡 toy 总账约为 28.05 GiB，容量余量更大，但所有请求共享一个 TP group。它可能让单个请求获得更高的层内并行度，却把 collective、队列和故障域集中到 8 张卡；一个 rank 失效通常会使整个副本不可用。若服务主要问题是单请求计算而不是容量，TP8 可能有价值；若问题是 QPS 和可用性，额外的 TP 并不等于额外副本。

TP4×PP2 也能在 toy 账本上放下模型，但请求要依次经过两个 pipeline stage，并在 stage 边界传递 activation。低并发时，stage 等待和 activation 传输可能直接增加 TTFT；高并发和合适的请求流量可以填充部分 stage 空洞。它只有在按层切分能显著改善容量或某些 stage 有独立扩展价值时才值得承担额外复杂度。必须测 stage 负载是否均衡，不能只用“每张卡 10 层”判断均衡，因为 embedding、lm_head、MoE 和不同 attention state 可能使各 stage 账本不同。

三个方案的比较应至少记录：

| 维度 | TP4 双副本 | TP8 单副本 | TP4×PP2 单副本 |
| --- | --- | --- | --- |
| 单副本 GPU | 4 | 8 | 8 |
| 独立队列 | 2 | 1 | 1 |
| 故障域 | 每个 4 卡 group | 整机 8 卡 group | 整机 8 卡 group |
| 主要通信 | TP collective | 更频繁/更大 TP collective | TP collective + stage activation |
| 适合的第一目标 | QPS、隔离、可用性 | 单请求计算和容量余量 | 层切分容量或特定拓扑 |
| 必测风险 | group 内通信、负载倾斜 | 单 group 故障、队列集中 | bubble、stage 不均衡、传输 |

若目标 workload 下 TP4 双副本已经满足 TTFT、TPOT、P99 和 goodput，它通常比把所有卡塞进一个 group 更容易运营；但不能从拓扑偏好直接下结论。如果单请求延迟仍不达标，才应比较 TP8 的 kernel 和 collective 收益；如果单卡层堆叠或 expert 权重无法合理放置，再测 PP 或 EP。

MoE 模型还要加一笔 expert 账。若每个 token 激活 top-k 个 expert，EP 会在 dispatch 和 combine 阶段产生 all-to-all；请求数均匀不代表 expert token 均匀。必须观察热 expert 的 `max_e / mean_e`、跨节点字节数、丢 token/重路由率和 P99。把 expert 权重放到不同 GPU 只能解决常驻容量，不能自动解决 router 偏斜。

然后处理路由和状态。路由器不应按请求数 round-robin，而应同时看 pending prompt tokens、active sequence、KV watermark、预计输出、prefix locality 和租户配额。已有 prefix 的请求可以获得有限的 locality bonus，但如果目标副本已经拥堵，重新 prefill 可能比排队更快。正在生成的 KV state 不能因为路由器看到更轻的副本就直接搬运；需要比较序列化、传输、校验和目标 layout 的成本，或者在新副本重算。

最后进行故障演练：关闭一个 TP rank，确认整个 group 被标记为 draining；在流式输出中断时，确认客户端不会收到重复 token；对已经发起工具写操作的请求，先用幂等键查询提交状态；对长上下文请求，比较 cache 迁移和重新 prefill 的恢复时间。只验证“进程重启成功”不够，还要报告恢复后的 goodput、重算 token、P99 和重复副作用次数。

这个案例的结论不是 TP4、TP8 或 PP2 中某一个永远正确，而是：容量问题决定能否放置模型，通信和 kernel 决定单请求性能，副本数量决定水平容量和故障隔离，状态所有权决定能否安全迁移。并行部署方案只有在真实拓扑、请求分布、协议、质量和故障语义都被测量后，才具有可运营的含义。

## 15. 资料与证据边界

1. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/2104.04473)：张量并行和流水线并行的经典系统抽象；论文主要面向训练，不能直接当作目标 serving 拓扑的性能结论。
2. [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)：流水线切分、micro-batch 和 bubble 的原始研究资料；在线推理的动态到达、取消和变长请求需要单独测量。
3. [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)：稀疏 MoE、router 和 token capacity 的研究背景；EP 的具体 dispatch/combine 实现取决于 runtime。
4. [NVIDIA NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)：核对 collective 操作、通信语义和拓扑相关的官方说明；有效带宽和尾延迟仍需目标集群实测。
5. [NVIDIA TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/)：当前官方项目入口，用于核对 TensorRT-LLM 的并行、量化和 serving 实现边界；版本变化可能影响支持矩阵。
6. [DeepSpeed Inference Tutorial](https://www.deepspeed.ai/tutorials/inference-tutorial/)：核对 DeepSpeed Inference 的 tensor parallel 和 kernel 优化路径；文档示例不等于目标 workload 的 P99。
7. [vLLM Distributed Inference and Serving](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)：核对 vLLM 当前 tensor/pipeline parallel serving 配置和多节点注意事项。
8. [SGLang Prefill-Decode Disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)：核对 prefill/decode 分离与状态传递的公开实现入口；具体迁移语义仍需结合版本和部署配置。
9. [PyTorch Distributed](https://docs.pytorch.org/docs/stable/distributed.html)：核对 process group、collective 和分布式通信的基础接口。

论文回答“方法为什么这样设计”，官方文档回答“当前组件公开支持什么”，源码和 runtime trace 才能回答“目标版本实际走了哪条路径”。尤其是跨节点 TP、EP all-to-all、KV state 迁移和故障恢复，不能用单机 benchmark 或进程启动成功替代实测。

## 16. 本章小结

本章核心结论：

1. 推理并行目标不同于训练并行，更重视延迟、吞吐、稳定性和成本。
2. Tensor Parallel 切层内矩阵，适合单层大计算和同机高速互联。
3. Pipeline Parallel 切模型层，能降低单卡权重显存，但可能增加在线延迟。
4. MoE 推理需要 expert parallel，并面对 all-to-all 和负载均衡问题。
5. 多副本服务是提高 QPS 和可用性的常见方式。
6. LLM 负载均衡要看 token、队列和 KV Cache 成本，而不是只看请求数。
7. 跨节点推理要谨慎评估通信瓶颈。
8. 多 GPU 服务必须配合健康检查、故障摘除、重试和降级。
