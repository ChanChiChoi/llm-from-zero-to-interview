# 第八章：并行推理与多 GPU 服务

## 本章目标

理解模型太大或吞吐要求太高时如何进行多 GPU 推理。

## 核心议题

1. Tensor Parallel 推理。
2. Pipeline Parallel 推理。
3. Expert Parallel for MoE。
4. 多副本服务。
5. load balancing。
6. 跨节点通信瓶颈。

## 面试重点

部署中的并行目标和训练不同，常常更关注延迟、吞吐、稳定性和成本。

## 推理并行和训练并行有什么不同

训练并行关注的是：

1. 模型能不能训起来。
2. 梯度和优化器状态怎么同步。
3. 吞吐和收敛效率。

推理并行关注的是：

1. 模型能不能放下。
2. 单请求延迟是否可接受。
3. 多请求吞吐是否足够。
4. KV Cache 怎么分布。
5. 服务是否稳定。
6. 成本是否合理。

训练可以接受较高通信开销，只要总体 tokens/s 高；在线推理更敏感，因为通信可能直接增加用户请求延迟。

面试表达：训练并行关注训练效率和状态同步，推理并行更关注延迟、吞吐、KV Cache 和服务稳定性。

## 1. 为什么需要多 GPU 推理

多 GPU 推理通常有两类原因。

### 1.1 模型太大

单张 GPU 放不下模型权重和 KV Cache。

例如大模型权重本身就超过单卡显存，或者长上下文高并发导致 KV Cache 占用过大。

### 1.2 吞吐要求太高

模型能放进单卡，但单卡服务不了目标 QPS 或 tokens/s。

这时可以部署多个副本，或使用多 GPU 并行提高吞吐。

## 2. Tensor Parallel 推理

Tensor Parallel 把模型内部的大矩阵切到多张 GPU 上。

例如线性层：

```text
Y = X W
```

可以把 `W` 按列或行切分，让多张 GPU 共同计算。

### 2.1 优点

1. 能部署单卡放不下的大模型。
2. 单层计算可以并行。
3. 是大模型推理中常见并行方式。

### 2.2 代价

1. 每层都可能有通信。
2. 对 GPU 间带宽要求高。
3. 跨节点 TP 延迟更高。

### 2.3 适用场景

Tensor Parallel 更适合同一节点内多 GPU，例如 NVLink / NVSwitch 互联。

面试表达：TP 切的是层内矩阵，适合模型太大或单层计算太重，但通信频繁，所以更适合同机高速互联。

## 3. Pipeline Parallel 推理

Pipeline Parallel 把模型按层切成多个 stage。

例如：

```text
GPU 0: layer 1-10
GPU 1: layer 11-20
GPU 2: layer 21-30
GPU 3: layer 31-40
```

### 3.1 优点

1. 能把深模型放到多张 GPU 上。
2. 每张 GPU 只保存一部分层。

### 3.2 代价

1. 单请求要顺序经过多个 stage。
2. stage 间通信增加延迟。
3. 可能有 pipeline bubble。
4. 在线小 batch 场景利用率不一定好。

### 3.3 推理中的特点

训练中 pipeline 可以用 micro-batch 填满流水线。

但在线推理请求动态到达，输出长度不一致，pipeline 调度更复杂。

面试表达：PP 切的是层，能降低单卡权重显存，但会增加跨 stage 延迟，在线推理中要谨慎评估。

## 4. Tensor Parallel 和 Pipeline Parallel 怎么选

| 场景 | 倾向 |
| --- | --- |
| 单层矩阵很大 | Tensor Parallel |
| 模型层数很多且权重放不下 | Pipeline Parallel |
| 同机高速互联 | Tensor Parallel 更常见 |
| 跨节点通信较慢 | 减少跨节点 TP |
| 在线低延迟 | 尽量降低通信链路 |

真实系统可能组合 TP 和 PP，但复杂度会上升。

## 5. Expert Parallel for MoE

MoE 模型有多个 expert。

每个 token 由 router 分配到部分 expert。

Expert Parallel 把不同 expert 放到不同 GPU 上。

### 5.1 优点

1. 支持大规模 MoE 模型。
2. 每个 token 只激活部分 expert。
3. 总参数量可以很大。

### 5.2 难点

1. token 路由带来 all-to-all 通信。
2. expert 负载不均衡。
3. 某些 expert 可能成为瓶颈。
4. 部署和扩缩容更复杂。

面试表达：MoE 推理难点不只是参数多，而是 token 到 expert 的动态路由和 all-to-all 通信。

## 6. 多副本服务

如果模型能放进单卡或一组 GPU，可以部署多个副本。

例如：

```text
replica 1: GPU 0-1
replica 2: GPU 2-3
replica 3: GPU 4-5
```

请求由负载均衡器分发到不同 replica。

### 优点

1. 提高吞吐。
2. 提高可用性。
3. 便于水平扩展。
4. 单个副本故障不影响全部服务。

### 代价

1. 权重重复占显存。
2. 成本更高。
3. 需要负载均衡和健康检查。

面试表达：多副本是提高 QPS 和可用性的常见方式，但不能解决单副本模型放不下的问题。

## 7. Load balancing

负载均衡不是简单 round-robin。

因为 LLM 请求成本差异很大。

一个请求的成本取决于：

1. prompt 长度。
2. max_new_tokens。
3. 当前 decode 状态。
4. KV Cache 占用。
5. replica 当前队列长度。

常见策略：

1. round-robin。
2. least connections。
3. least queue time。
4. token-aware routing。
5. cost-aware routing。
6. tenant-aware routing。

面试表达：LLM 负载均衡最好看 token 和 KV Cache 成本，而不是只看请求数。

## 8. 跨节点通信瓶颈

多机推理比单机多卡更复杂。

原因是跨节点通信带宽和延迟通常不如节点内 NVLink / NVSwitch。

跨节点 TP 或 expert routing 可能显著增加延迟。

优化方向：

1. 尽量把强通信并行放在节点内。
2. 跨节点更多使用副本或较粗粒度切分。
3. 控制 batch 和并行度。
4. 监控通信耗时。

面试表达：跨节点并行推理要非常关注通信拓扑，因为通信延迟会直接影响用户请求延迟。

## 9. KV Cache 在多 GPU 中怎么处理

并行推理不仅要切权重，还要考虑 KV Cache。

在 TP 中，KV Cache 也可能按 head 或 hidden 维度分布在不同 GPU。

在 PP 中，不同层的 KV Cache 跟随对应 stage。

在多副本中，每个副本维护自己的 KV Cache。

这会影响：

1. 显存规划。
2. 调度策略。
3. 请求迁移。
4. 故障恢复。

通常一个正在生成的请求不适合频繁跨 replica 迁移，因为 KV Cache 已经在当前副本上。

## 10. 多 GPU 服务的故障处理

多 GPU 服务中，一个 GPU 故障可能导致整个副本不可用。

需要：

1. 健康检查。
2. 自动摘除故障副本。
3. 请求重试。
4. 限流和降级。
5. 快速拉起新副本。

如果是 TP/PP 组成的副本，任一 rank 挂掉通常会影响整个 replica。

## 11. 并行推理选型流程

可以按下面思路选择：

```text
1. 单卡能否放下模型和 KV Cache？
   能：优先单卡副本，简单稳定。
   不能：考虑 TP/PP/量化。

2. 是否需要提高 QPS？
   是：优先增加副本。

3. 是否单层矩阵太大？
   是：考虑 TP。

4. 是否模型层数太多或权重太大？
   是：考虑 PP。

5. 是否是 MoE？
   是：考虑 expert parallel 和路由开销。

6. 是否跨节点？
   是：谨慎评估通信延迟。
```

## 12. 面试官会怎么问

### 问法 1：推理中的 Tensor Parallel 解决什么问题？

可以这样答：

```text
Tensor Parallel 把模型层内的大矩阵计算切到多张 GPU 上，解决单卡放不下或单卡算力不足的问题。它常用于大模型推理，但每层可能需要通信，所以更适合同机 NVLink/NVSwitch 这类高速互联环境。
```

### 问法 2：为什么推理中多副本很常见？

可以这样答：

```text
如果单个副本已经能放下模型，多副本是提高 QPS、吞吐和可用性的简单方式。每个副本独立服务请求，负载均衡器分发流量，某个副本故障时可以摘除。但多副本会重复保存权重，成本更高。
```

### 问法 3：TP 和 PP 在推理中怎么选？

可以这样答：

```text
TP 切层内矩阵，适合单层计算或权重太大，通信频繁但在同机高速互联下效果好。PP 切模型层，能把深模型分到多卡，但单请求要经过多个 stage，增加延迟并可能有 pipeline bubble。低延迟在线服务通常要谨慎使用 PP。
```

### 问法 4：MoE 推理为什么难？

可以这样答：

```text
MoE 推理中 token 会被 router 动态分配到不同 expert，不同 expert 可能在不同 GPU 上。这会带来 all-to-all 通信和负载不均衡问题。某些 expert 热点会影响整体延迟，所以要处理 expert parallel、路由和负载均衡。
```

### 问法 5：为什么 LLM 负载均衡不能只按请求数？

可以这样答：

```text
因为不同请求的 token 数、输出长度和 KV Cache 占用差异很大。一个长上下文请求可能比很多短请求更贵。所以负载均衡最好考虑 queue length、active tokens、KV Cache 使用和预计生成长度，而不是只看请求数。
```

## 13. 本章小结

本章核心结论：

1. 推理并行目标不同于训练并行，更重视延迟、吞吐、稳定性和成本。
2. Tensor Parallel 切层内矩阵，适合单层大计算和同机高速互联。
3. Pipeline Parallel 切模型层，能降低单卡权重显存，但可能增加在线延迟。
4. MoE 推理需要 expert parallel，并面对 all-to-all 和负载均衡问题。
5. 多副本服务是提高 QPS 和可用性的常见方式。
6. LLM 负载均衡要看 token、队列和 KV Cache 成本，而不是只看请求数。
7. 跨节点推理要谨慎评估通信瓶颈。
8. 多 GPU 服务必须配合健康检查、故障摘除、重试和降级。
