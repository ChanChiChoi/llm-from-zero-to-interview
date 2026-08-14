# 第三十三章：Mamba 的训练、推理和硬件效率优势

## 33.1 “线性”要落到哪一个阶段

Mamba 类模型常被描述为线性时间序列模型，但训练和推理的计算路径并不完全相同。训练要处理整段 token，需要并行 scan 或 chunk；推理每次只追加一个 token，主要维护递归 state。一个结构在训练上快，不代表在线 decode 也快；一个 state 很小，也不代表 GPU 利用率高。

工程评估要把 pretraining、batch inference、streaming decode、变长 batch 和状态恢复分别测。不能只拿论文中的单项 throughput 作为通用结论。

## 33.2 训练路径：从递归到 scan

输入依赖状态更新可以抽象为：

~~~math
s_t=A_t s_{t-1}+b_t
~~~

若把更新表示成仿射变换 `(A_t,b_t)`，则可以用结合运算做 prefix scan：

~~~math
(A_2,b_2)\circ(A_1,b_1)
=(A_2A_1,A_2b_1+b_2)
~~~

硬件实现可以在局部并行、分块累计和跨块合并之间折中。实际的 state 可能是按 batch、channel、state dimension 布置的向量，`A` 也可能是对角/广播结构，因此实现重点不是通用矩阵乘，而是 layout 和 fused kernel。

## 33.3 推理路径：固定 state 的价值

设每层 state 大小为 `d_state×d_inner` 的某种布局，新增 token 到来时，只需读取当前 token、更新 state、生成输出。与显式 KV 逐 token 读取长度 `T` 的历史相比，理论读取不随 `T` 增长。

显式 KV 的历史读取近似为：

~~~math
R_{\mathrm{KV}}\propto T H_{kv}d_h
~~~

递归 state 的读取更接近：

~~~math
R_{\mathrm{state}}\propto d_{\mathrm{state}}
~~~

这只比较了状态路径，不包含权重读取、激活、卷积、通信和 kernel 启动。state 小时，内存带宽可能更有利；state 更新细碎时，算术利用率可能反而较低。

## 33.4 一个相对成本模型

~~~python
def compare_decode(prompt, output, kv_per_token, state_size,
                   weight_read, update_cost):
    kv_reads = sum(prompt + i for i in range(output)) * kv_per_token
    state_reads = output * state_size
    return {
        "kv_path_units": round(kv_reads + output * weight_read, 2),
        "state_path_units": round(state_reads
                                  + output * (weight_read + update_cost), 2),
    }


print(compare_decode(
    prompt=32_000, output=256, kv_per_token=1.0,
    state_size=256, weight_read=500.0, update_cost=20.0,
))
~~~

这是硬件无关的相对模型，用来提醒读者：KV 读取随上下文增长，state 更新近似固定，但权重读取可能成为两者共同瓶颈。真实结论需用 profiler 和目标 batch 测。

## 33.5 为什么 kernel 融合很重要

如果每个 token 都执行多个小 kernel：投影、卷积、参数生成、state update、门控、输出投影，那么 kernel launch 和中间 tensor 写回可能吞掉理论优势。fused selective scan 将相关操作放进较少的 kernel，让 state 尽可能留在片上或寄存器/共享内存可访问范围。

融合也提高了调试难度。reference Python、CUDA kernel、不同 dtype、不同 batch 和不同序列长度必须逐步对比。一个 fused kernel 发生 fallback 时，服务性能可能突然下降而质量不变，监控必须记录实际 kernel 路径。

## 33.6 训练内存与推理内存

训练要保存中间激活、反向信息和优化器状态；scan 的并行实现可能需要额外 workspace。推理主要保存权重、state、临时激活和 batch 元数据。SSM 减少的是随历史长度增长的 KV 形式，不会消除模型权重。

可以把峰值内存写成：

~~~math
M_{\mathrm{peak}}=
M_{\mathrm{weight}}+
M_{\mathrm{state}}+
M_{\mathrm{activation}}+
M_{\mathrm{workspace}}+
M_{\mathrm{runtime}}
~~~

在长流小 batch 场景，`M_state` 的固定性质很有价值；在短 prompt 大 batch 场景，权重和 activation 可能主导，优势会变小。

## 33.7 变长 batch 和状态调度

Transformer serving 可以用 paged KV 管理不同长度；Mamba serving 要为每个请求分配对应层的 state，并在 batch 重排、请求完成、抢占和迁移时维护所有权。state 与 token offset 必须绑定，不能只绑定 batch slot。

如果请求 A 完成后 slot 被 B 复用，却忘记清零 A 的 state，B 会收到错误历史；如果 preemption 只保存文本长度而没有 state snapshot，恢复时必须重新处理前缀，延迟和行为都会改变。

## 33.8 训练/推理一致性

一个可靠测试是把同一序列分成两段：一次性运行得到输出 `Y_full`，先运行前缀保存 state，再继续运行后缀得到 `Y_resume`。比较对应 logits：

~~~math
\Delta_{\mathrm{resume}}=
\max_t\|z_t^{\mathrm{full}}-z_t^{\mathrm{resume}}\|_\infty
~~~

同时测试不同 chunk 长度、batch size、dtype、设备和 state serialization。若差异只在长序列出现，检查离散化、scan 累积和精度；若差异在短序列就存在，检查初始 state、卷积 padding 和位置/模板。

## 33.9 与 Transformer decode 的对比

Transformer decode 的每步计算包含 query 投影、对历史 KV 的 attention 读取和输出投影；Mamba decode 包含输入投影、局部卷积、selective parameter、state update 和输出投影。两者都可能被权重带宽和 batch 限制。

当上下文特别长且输出持续增长，Mamba 的固定状态有更明确的优势；当需要精确引用、prefix sharing、成熟 speculative decoding 或复杂多模态 cross-attention，Transformer 生态可能更成熟。实际系统也可能让两者按请求类型路由。

## 33.10 失败模式和指标

性能失败：kernel fallback、state layout 不连续、batch 太小、padding 浪费、跨卡同步、CPU/GPU copy、缓存未命中、热降频。正确性失败：state 泄露、resume mismatch、chunk 边界、padding 更新、speculative rollback 错误。

建议记录：prefill tokens/s、decode tokens/s、TTFT、TPOT、p50/p99、state bytes/request、峰值显存、kernel 时间占比、GPU occupancy、HBM read、state snapshot/restore 时间和失败率。

## 33.11 机制与边界：选择性 scan 的 roofline 视角

选择性 scan 往往拥有较低的算术强度，瓶颈可能是 state 读写和中间参数带宽，而不是 FLOPs。融合可以减少写回，但也可能降低 occupancy 或限制不同 batch 的并行。要使用实际 device counter 分析 memory-bound/compute-bound，而不是只从论文复杂度推断。

训练的 scan 并行度、推理的递归延迟和跨请求 batch 组织是三个不同优化问题。一个统一的“吞吐”数字没有足够信息，至少要注明长度、batch、硬件、dtype、是否包含 tokenizer 和比较 baseline。

## 33.12 面试追问、误区与练习

**问：Mamba 的推理为什么可能比 Transformer 更省内存？**

标准回答：它按请求保存固定大小的递归 state，不需要保存随上下文长度增长的完整 KV；但权重、激活、workspace 和 state layout 仍要计入，且实际速度取决于 kernel 和 batch。

**问：如何验证 Mamba 的流式 state 是正确的？**

标准回答：比较整段运行与前缀保存/后缀恢复的逐 token logits，覆盖不同 chunk、batch、dtype、迁移和 rollback；同时验证请求完成后 state 清零和跨租户隔离。

常见误区包括只看 state 大小、把理论线性直接写成实际低延迟、忽略 kernel fallback，以及没有把状态生命周期纳入 scheduler。

练习：在同一硬件上实现 toy KV decode 和 toy state decode，测不同上下文/输出长度下的读写量、p99 和恢复一致性。

## 33.13 训练吞吐和推理吞吐的区别

训练可以把多个序列组成大 batch，并利用并行 scan；推理 decode 每个请求每轮只前进少量 token，更容易受 state bandwidth、kernel launch 和 batch 不规则影响。不能拿训练 tokens/s 直接预测在线 TPOT。

部署应分别压测短请求、长请求、混合长度、低并发和高并发，并记录 state bytes、GPU 利用率、memory bandwidth、kernel time 和 p99。

## 33.14 Mamba 的服务状态契约

每个请求的 state 必须绑定 model revision、position/step、dtype、layer layout、owner 和 checksum。continuous batching 中请求加入、退出和重排都不能改变另一个请求的 state；preemption 和 speculative 必须支持 snapshot/rollback。

如果系统只把 Mamba 当成“没有 KV cache 的模型”，而没有实现状态生命周期，理论内存优势会被线上错误和恢复失败抵消。

## 33.15 用 roofline 解释“理论快、线上慢”

状态扫描常见的瓶颈不是矩阵乘的 FLOPs，而是每个 token 读写 state、输入相关参数和中间结果的带宽。可以用算术强度作第一层判断：

~~~math
I=\frac{\text{执行 FLOPs}}{\text{读写字节数}}
~~~

当 I 较低时，增加计算单元未必提高速度；需要减少中间写回、融合投影与 scan、改善 layout 或提高 batch。融合也不是无条件收益，过大的 kernel 可能降低 occupancy，混合长度还会增加 padding。

因此 profile 要同时看 HBM bandwidth、L2 命中、occupancy、kernel launch、CPU/GPU copy 和 fallback。论文中的理论复杂度只说明算法方向，不能替代设备计数器。

## 33.16 长流服务的调度问题

固定 state 让请求不必保存全部历史，却让每个请求成为一个持久状态对象。scheduler 需要在公平性、state 迁移和 batch 利用率之间取舍：长流请求占住 state slot，短请求可能难以插入；抢占可以释放计算资源，却要保存和恢复完整 state。

容量估算可写成：

~~~math
M_{\mathrm{state,total}}
=N_{\mathrm{active}}\times
\sum_{\ell=1}^{L}M_{\mathrm{state},\ell}
+M_{\mathrm{snapshot}}+M_{\mathrm{workspace}}
~~~

其中 active 请求数、层数、dtype 和 snapshot 频率都影响峰值。没有 state admission control，系统可能在 token 数看似不高时先耗尽显存。

## 33.17 从 toy benchmark 到生产验收条件

小 batch、整齐长度和连续运行只能验证理想路径。生产前应加入混合长度、请求取消、超时、迁移、模型升级、量化和 kernel fallback，并比较一次性与恢复后的 logits、任务答案和 p99。

一个实际准入条件可以要求：关键任务 exact recall 不下降，state restore 误差低于阈值，跨租户泄漏为零，fallback 比例有上限，长流 p99 满足 SLO。只有这些条件同时通过，固定 state 的内存优势才具有产品价值。

## 33.18 Roofline 视角下的“线性更快”

线性复杂度只说明随长度增长的阶数，不说明 kernel 位于计算受限还是带宽受限区域。对 Mamba 类 scan，应测每 token 的状态读写字节、FLOPs、有效带宽、occupancy、kernel launch 和融合程度。若状态维度很大，理论上不随 T 增长的状态更新仍可能受显存带宽限制。

可以用算术强度做初步判断：

```math
I=\frac{\mathrm{FLOPs}}{\mathrm{bytes\ moved}},qquad
P\le\min(P_{\mathrm{peak}},I\times BW_{\mathrm{peak}})
```

这不是完整性能模型，但能解释为什么同一个 Mamba kernel 在不同 GPU、batch 和 sequence length 上排名变化。比较 Transformer decode 时也要把 KV 读取、GQA/MLA、FlashAttention 和 batch 形状放在同一实验条件中。

## 33.19 长流服务的状态调度

固定 state 降低了随历史增长的缓存压力，却引入“每个活跃流都占一个 state”的调度问题。服务需要定义 state 的创建、迁移、冻结、恢复和释放；长时间无输出的流不能无限占用 GPU；抢占时保存 state 的成本要和重算成本比较。

对多租户服务，state key 必须包含请求、模型 revision、adapter、位置/step 和权限域。取消请求不仅要停止 decode，还要释放 state 和未完成的异步 kernel。若 state 只在 GPU 上存在，节点故障后的恢复路径必须明确是重算、迁移还是返回不可恢复。

## 33.20 状态迁移的成本模型

固定 state 降低了随 token 数增长的历史存储，却没有消除状态调度。假设每层 state 有 `s` 个元素、共有 `L` 层、每个元素占 `b` 字节，单请求快照大小近似为：

```math
M_state = L s b + M_metadata
```

迁移时还要支付序列化、网络传输、反序列化和重新 warmup 的时间。若 GPU 之间带宽为 `BW`，state transfer 的下界近似为 `M_state / BW`，实际还要加同步和排队。对长流服务，迁移 state 可能比重算最近 chunk 更便宜，也可能因为版本不兼容而更危险；这必须用相同输入和目标 SLO 比较。

## 33.21 训练吞吐不能替代在线容量

训练阶段的 scan 以长序列和大 batch 为主，适合融合和并行；在线 decode 的 batch 往往很小，state update、请求调度和网络输出占比上升。一个模型可以在训练吞吐上领先，却在短请求混合、随机取消和恢复场景下不满足 p99。

建议建立三层结果表：算子层记录 FLOPs、bytes、occupancy 和 kernel time；engine 层记录 TTFT、TPOT、active state、迁移和 fallback；任务层记录局部预测、精确复制、工具参数和单位成功成本。只有三层都通过，才能把“线性 scan”写成生产收益。

## 33.22 训练与硬件效率的联合评估

Mamba 论文（https://arxiv.org/abs/2312.00752）讨论了选择性状态和硬件感知 scan；具体性能数字必须绑定实现、GPU、batch、dtype 和 baseline。

Mamba 的硬件优势来自状态形式与 kernel 共同设计。固定 state 解决了一类长流内存问题，但只有当 scan、scheduler、snapshot 和真实 workload 一起验证，优势才是可用的工程优势。

## 33.23 Scan kernel 的边界条件

Mamba 的 scan 在理论上可以线性推进，但实际 kernel 还要处理变长 batch、padding、chunk、state reset、不同 dtype 和融合层。一个只在满长度、单一 batch 上通过的 kernel，不能直接证明线上服务等价。至少要比较 reference Python/框架实现、融合 kernel、chunked scan 和 single-step decode。

每个路径记录 hidden/state 最大误差、吞吐、显存、kernel 数和 fallback。若误差只在 chunk 边界出现，先查 state 传递；若误差随长度增长，查累计精度和参数稳定；若质量相同但 p99 恶化，查 workspace、调度和 batch fragmentation。

## 33.24 长流和短请求要分开容量

长流服务和普通聊天的资源画像不同。长流减少了频繁 prefill，却长期占用 state、连接和调度槽；短请求需要快速接入和释放，受 kernel launch、queue wait 和网络输出影响。把两者混在一个平均 tokens/s 中会掩盖公平性和尾延迟。

可以按 workload pool 记录 active state、state bytes、TTFT、TPOT、queue wait、取消率、迁移率和 p99。长流池需要 snapshot/迁移能力，交互池需要快速 admission 和严格的短请求 SLO。分池会损失资源共享效率，因此应通过真实流量压测决定是否值得。

## 33.25 Mamba 与 Transformer 的 matched benchmark

公平比较要固定模型参数量或明确 active compute、训练 token、tokenizer、context length、batch、dtype、硬件和质量任务。Transformer baseline 应使用成熟的 GQA/FlashAttention/KV 管理，Mamba 应使用目标版本的 fused scan；不能用未经优化的 dense reference 与优化后的 scan 对比。

质量任务至少包括局部语言、远距复制、流式状态延续、冲突版本、代码和工具结构化输出。系统指标包括 prefill/decode、KV/state、峰值显存、p99、snapshot、fallback 和单位成功成本。若某条路径依赖外部 retrieval 才能完成引用，retrieval 成本和延迟必须算入。

## 33.26 结论的可迁移范围

Mamba 论文和目标 runtime 的结果只能支持对应条件下的结论。不同 state size、层数、GPU、batch、量化、长度和服务调度可能改变排序。更谨慎的结论是：Mamba 提供了一种固定状态和硬件感知 scan 的路径，在长流和特定查询模式下有潜在资源优势；是否适合通用助手，要由精确证据、协议、恢复和端到端成本验证。
