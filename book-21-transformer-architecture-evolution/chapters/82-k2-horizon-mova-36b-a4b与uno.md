# K2 Horizon MoVA-36B-A4B：从 value routing 到长上下文 Serving

> 本章核验日期：2026-09-14。K2 候选由 [Artificial Analysis 中文榜单入口](https://artificialanalysis.ai/zh) 发现；模型事实主要来自 [IFM/K2-Horizon-MoVA-36B-A4B 官方模型卡](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B)、固定 revision `de2d2efb32ed7639b7140bccbefe131a0063a982` 的配置与实现，以及模型卡引用的 [SGLang K2 Horizon cookbook](https://docs.sglang.io/cookbook/autoregressive/IFM/K2-Horizon)。本章把发布方字段、代码语义、教学抽象和生产实测严格分开。

## 1. 先看一个真实问题：为什么 4B active 仍然可能很重

假设一个服务收到一段长文。模型对每个 token 并不需要把 36B 参数全部做一次矩阵乘法，而是可以从若干 FFN 专家和 value 专家中选择少数路径。于是模型卡给出“36B 总参数、约 4B 每 token active”的标签。

这个标签很有用，却很容易被误读成“只需要一张 4B 模型的显存”和“速度一定等于 4B dense 模型”。实际服务还要处理：

1. 36B 权重的加载、分片和副本。
2. 每个 token 到不同 expert 的 dispatch 与返回。
3. MoE router 和 MoVA router 的索引、权重与负载元数据。
4. 48 层的 KV cache，其中每层的 Q/K/V 结构并不等于专家权重。
5. prefill 激活、通信 buffer、workspace、padding 和碎片。

K2-Horizon-MoVA-36B-A4B 的价值正好在这里：它把“稀疏 FFN”推进到“value 路径也可以被路由”的组合，同时保留 GQA、RoPE 和 attention gate 等 Transformer 基础件。学习它时，不能只背 36B 和 4B 两个数字，要能从 token 路由一路推导到服务成本。

## 2. 模型卡和配置给了什么

### 2.1 一页规格表

| 字段 | K2-Horizon-MoVA-36B-A4B |
|---|---|
| 模型 ID | `IFM/K2-Horizon-MoVA-36B-A4B` |
| 总参数 | 约 36B |
| 每 token active | 约 4B，属于发布方的 active-parameter 口径 |
| 层数 | 48 decoder layers |
| hidden size | 2560 |
| attention heads | 32 query heads、8 KV heads、head dim 128 |
| FFN routed experts | 100，top-8 |
| shared expert | 1 |
| MoVA value experts | 64，top-4 |
| 上下文配置 | 524,288 tokens |
| 精度 | BF16 |
| 词表 | 250,624 |
| sliding window | `null` / disabled |
| 官方 revision | `de2d2efb32ed7639b7140bccbefe131a0063a982` |

这张表的第一条面试原则是：配置字段比模型名更可靠，但配置字段也只说明当前 checkpoint/实现的公开行为。`active parameters` 不能自动换算 FLOPs，`max_position_embeddings` 不能自动证明任意 512K 任务都能成功，许可证和模型卡也不能代替完整训练报告。

### 2.2 前三层和后四十五层

配置的 `mlp_only_layers=[0, 1, 2]` 使用零起始编号。实现中前 3 层是普通 dense attention + dense MLP；后 45 层因为 `decoder_sparse_step=1` 进入 sparse layer：attention 使用 MoVA value routing，FFN 使用 sparse MoE。

因此更准确的结构描述是：

```text
Layer 0-2:   dense attention + dense SwiGLU MLP
Layer 3-47:  MoVA attention + top-8 routed MoE FFN + shared expert
```

它不是“48 层全部把 100 个 FFN expert 算一遍”，也不是“全部层都是 sliding-window attention”。稀疏只减少主要专家计算，仍然留下路由、通信和缓存成本。

## 3. MoVA：把专家放到 value 路径

### 3.1 从普通 attention 开始

普通多头注意力可以写成：

`Q = X W_Q, K = X W_K, V = X W_V`

`A = softmax(Q K^T / sqrt(d))`

`Y = A V`

在 K2 的 GQA 中，Q 有 32 个 heads，K/V 有 8 个 heads。也就是说，多个 query head 共享同一组 KV head。MoVA 进一步让 value 变换不再只有一个固定的 `W_V`，而是先由 value router 为当前 token 选择 value experts：

`V(x) = sum_{e in Top4(x)} alpha_e(x) * SiLU(W_e x)`

这里的 `W_e` 将 hidden size 2560 映射到 `8 * 128` 的 value 表示。然后注意力仍然使用：

`Y = softmax(Q K^T / sqrt(128)) V(x)`

关键点是：MoVA 路由的是 value 变换，不是完整 attention head。它没有把 Q、K、attention score 和输出投影各自复制成 64 份。

### 3.2 一个 token 的路由过程

固定一个 token 的 hidden vector `x`，可以按下面的顺序理解：

1. value router 产生 64 个 logits。
2. 对 logits 做 sigmoid，得到非归一化 routing scores。
3. router bias 只加入用于 top-4 选择的 score。
4. 从选择结果取回原始 routing scores，而不是把带 bias 的数直接当混合权重。
5. top-4 权重归一化，再乘 `router_scaling_factor=2.5`。
6. 对 4 个 value expert 分别计算 `SiLU(W_e x)`，按权重相加。
7. 将混合 value 送入 GQA attention。

第 3、4 步是 checkpoint 兼容性的细节。下面这种看似自然的替换并不等价：

```text
错误简化：softmax(logits + bias) -> top-k -> 直接加权
公开实现：sigmoid(logits) -> (score + bias) 只用于选择
                       -> 取原始 score -> 归一化 -> 乘 2.5
```

### 3.3 和普通 MoE FFN 的关系

普通稀疏 MoE 把 expert 放在 FFN：

`FFN(x) = sum_{e in Top8(x)} beta_e(x) * FFN_e(x) + Shared(x)`

K2 先在 attention 的 value 路径有一次 top-4 路由，经过 attention 后又在 FFN 有一次 top-8 路由。因此一个 token 在后 45 层可能同时触发两类动态专家。两类 expert 的输出形状、路由数量和通信位置不同，不能合并成一个“top-12 MoE”标签。

| 路径 | 专家数 | 每 token 选择 | 输出所在阶段 |
|---|---:|---:|---|
| MoVA value expert | 64 | top-4 | 形成 attention 的 V |
| FFN routed expert | 100 | top-8 | attention 后的 SwiGLU FFN |
| FFN shared expert | 1 | 始终计算 | 与 routed FFN 输出相加 |

## 4. FFN MoE 和 active parameters

### 4.1 FFN 的计算

配置中 dense MLP 的中间维度是 6144，routed MoE expert 的中间维度是 768，激活函数是 SiLU。单个 expert 的教学表达为：

`FFN_e(x) = W_down,e ( SiLU(W_gate,e x) * W_up,e x )`

100 个 expert 的总权重决定模型文件大小；top-8 只决定某个 token 主要执行哪些 expert。shared expert 使用残差输入额外计算，不因为 top-k 而消失。

### 4.2 为什么 total 和 active 必须分开

可以把一次推理的资源分为三层：

```text
存储容量：36B total parameters
主要 token 计算：约 4B active-parameter proxy
系统成本：专家 dispatch + KV + workspace + 调度 + 通信
```

`4B` 更接近“每 token 参与主要专家矩阵计算的参数规模代理”，而不是严格的数学定义。至少有以下原因会让它和真实成本不同：

- router 仍要产生 logits，并执行 top-k、排序/索引和聚合。
- 多卡 expert parallel 需要 token-to-expert 通信。
- padding/capacity 和负载不均会让实际参与计算的 token 数增加。
- shared expert、dense 层和 attention 的参数不被简单的 active 标签完整描述。
- kernel launch、量化/反量化、内存访问和通信等待可能成为瓶颈。

面试时若被问“4B active 是否等价于 4B dense”，应回答：不是。它可以解释稀疏模型的主要计算规模，但必须结合专家布局、负载、通信、KV cache 和实测 TTFT/TPOT。

## 5. GQA、RoPE 和 attention gate 各自做什么

### 5.1 GQA：减少 KV 的 head 数

32 个 query heads 对应 8 个 KV heads，复用倍率是：

`groups = 32 / 8 = 4`

在相同序列、batch 和 head dimension 下，K/V cache 的 head 数理论上约是 MHA 的四分之一。这是 GQA 的收益，不是 MoVA 的收益。MoVA 生成的是最终 value states；它没有产生 64 份可以全部写进 KV cache 的 value cache。

### 5.2 RoPE：给 Q/K 加位置相位

配置使用 default RoPE，`rope_theta=10000000`，`rope_head_dim=128`。实现对 Q/K 施加旋转位置编码，再进入注意力。当前证据不支持把 K2 写成 YaRN 或 sliding-window 模型。

更长上下文来自训练阶段和位置分布的共同支持，而不是把 `max_position_embeddings` 从 8K 改成 512K 就结束。部署迁移时，tokenizer、position ids、RoPE 配置和 checkpoint branch 必须一起绑定。

### 5.3 attention gate：调节 attention 分支幅度

实现中 attention gate projection 从 hidden state 生成与 attention output 对齐的门控向量：

`Y_gate = Y * softplus(G(x), beta=ln(2))`

随后才进入 `o_proj`。它解决的是 attention 分支幅度控制，不负责选择 value expert，也不负责选择 FFN expert。

三个容易混淆的“门”可以这样分：

| 名称 | 输入 | 输出/作用 |
|---|---|---|
| MoVA router | hidden state | top-4 value expert 和 mixture weights |
| MoE router | hidden state | top-8 FFN expert 和 mixture weights |
| attention gate | hidden state 与 attention output | 对 attention 分支做幅度调节 |

## 6. 零依赖 toy：复现路由语义和资源账本

下面的代码不加载 K2 权重，也不声称实现生产 kernel。它只复现两个值得面试时手算的部分：

1. sigmoid 分数、bias 只用于选择、top-k 归一化和 scaling。
2. 用 token 数、两类 top-k、hidden size 和 head 数生成通信/KV 的数量级账本。

```python
import math


def route_with_selection_bias(logits, bias, top_k, scaling_factor):
    if len(logits) != len(bias):
        raise ValueError("logits and bias must have the same length")
    if not 0 < top_k <= len(logits):
        raise ValueError("top_k must be within the expert count")

    raw = [1.0 / (1.0 + math.exp(-value)) for value in logits]
    selection = [score + offset for score, offset in zip(raw, bias)]
    chosen = sorted(range(len(raw)), key=lambda index: selection[index], reverse=True)[:top_k]
    denominator = sum(raw[index] for index in chosen)
    weights = {index: scaling_factor * raw[index] / denominator for index in chosen}
    return chosen, weights


def k2_resource_ledger(batch, sequence, hidden, kv_heads, head_dim, layers, sparse_layers,
                       moe_top_k, mova_top_k, bytes_per_element, devices):
    tokens = batch * sequence
    expert_assignments = tokens * (moe_top_k + mova_top_k) * sparse_layers
    one_way_dispatch_bytes = expert_assignments * hidden * bytes_per_element
    kv_bytes = batch * sequence * layers * kv_heads * head_dim * 2 * bytes_per_element
    return {
        "tokens": tokens,
        "expert_assignments": expert_assignments,
        "one_way_dispatch_gib": one_way_dispatch_bytes / (1024 ** 3),
        "kv_gib": kv_bytes / (1024 ** 3),
        "devices": devices,
        "note": "toy payload estimate; excludes padding, overlap, workspace and weights",
    }


chosen, weights = route_with_selection_bias(
    logits=[0.2, 1.1, 0.7, -0.4],
    bias=[0.0, -0.3, 0.1, 0.0],
    top_k=2,
    scaling_factor=2.5,
)
print("chosen=", chosen)
print("weight_sum=", round(sum(weights.values()), 3))
print("ledger=", k2_resource_ledger(
    batch=2, sequence=4096, hidden=2560, kv_heads=8, head_dim=128,
    layers=48, sparse_layers=45, moe_top_k=8, mova_top_k=4,
    bytes_per_element=2, devices=2,
))
```

运行这段代码时，`weight_sum` 是 `2.5`，因为 top-k 原始 sigmoid 分数先归一化再乘 scaling factor。这个输出不是概率分布，不能再把它当作和为 1 的 softmax。`one_way_dispatch_gib` 也只是把 hidden payload 乘以 assignments 的教学上界；真实 SGLang/vLLM 还要考虑 local expert 命中、EP 分片、padding、all-to-all 算法、返回路径和通信计算 overlap。

## 7. 512K 长上下文：训练阶段比一个配置字段更重要

### 7.1 模型卡披露的阶段

模型卡把训练 token 写成每个阶段的额外预算，并说明后一个阶段从前一个阶段的最终 checkpoint 继续：

| 阶段 | 额外 token | 序列长度 | 公开目的/变化 |
|---|---:|---:|---|
| Pretraining | 22.9T | 8K | 基础预训练 |
| Midtraining Stage 1 | 1.1T | 32K | 上下文扩展 |
| Midtraining Stage 2 | 498B | 128K | 上下文扩展 |
| Midtraining Stage 3 | 110B | 512K | 上下文扩展 |
| Midtraining Stage 4 | 199B | 512K | 继续扩展，数据更偏 agentic/reasoning SFT |
| SFT Phase 1 | 219B | 512K | 领域覆盖 |
| SFT Phase 2 | 50B | 512K | Phase 1 高质量子集，learning-rate decay |

`50B` 是 Phase 2 的增量，不是和 `219B` 相互替代的总数。读取训练表时必须先问：这些是阶段增量还是累计量？否则会把同一条数据链错误地统计两次。

### 7.2 为什么长上下文训练是系统工程

序列长度从 8K 逐步扩展到 512K，会同时影响：

- position ids 和 RoPE 的训练覆盖。
- attention 的 pair 数、FlashAttention workspace 和 prefill 峰值。
- 数据 packing、长短样本混合和有效 token 利用率。
- 激活 checkpointing、并行切分和通信重叠。
- 评估时的长距离检索、lost-in-the-middle、工具轨迹和终止条件。

因此“配置允许 524,288”只说明模型接口/实现预留了这个长度；长任务是否稳定、检索是否有效、服务是否承受得住，要用固定 revision、任务集、batch、硬件和 harness 实测。

## 8. MoE/MoVA 的通信和显存账本

### 8.1 先算 token assignment

设 batch 为 `B`，序列长度为 `T`，后 45 层的 FFN top-k 为 8，MoVA top-k 为 4。每个 sparse layer 的专家 assignment 数量级是：

`N_assign = B * T * (8 + 4) * 45`

若把每个 assignment 的 hidden payload 粗略按 `H * bytes` 计，单向 payload 上界为：

`C_dispatch ~= N_assign * H * bytes`

这个公式不是 NCCL 或 SGLang 的精确通信模型，但能揭示一个重要事实：active parameters 变小了，token dispatch 未必变小。网络拓扑、expert placement 和负载倾斜可能决定最终吞吐。

### 8.2 再算 KV cache

只用模型结构做一个 BF16 K/V cache 估算，忽略 block metadata、page table 和压缩：

`M_KV ~= B * T * L * H_KV * d * 2 * bytes`

K2 的 `L=48`、`H_KV=8`、`d=128`，BF16 每元素 2 bytes。这里的 `2` 是 K 和 V 两份，不是 MoVA top-4。这个估算只覆盖 KV cache，不包含：

- 36B 权重及其 TP/EP 分片。
- MoVA/FFN router indices 和 weights。
- expert dispatch 的发送/返回 buffer。
- FlashAttention workspace、CUDA graph、allocator 碎片和 staging memory。
- prefill 激活、并行通信和长请求的调度预留。

完整显存账本应写成：

`HBM = weights + KV + activations + router_metadata + dispatch_buffers + workspace + fragmentation`

### 8.3 训练和 Serving 的差别

训练阶段还要加 optimizer states、gradients、activation checkpoint 和 all-reduce；Serving 阶段通常没有 optimizer states，却会承受 KV cache、continuous batching、专家通信和长尾请求。不能用训练显存数字直接回答线上并发，也不能用一个单请求 KV 公式替代训练容量规划。

## 9. MOPD：先分清同系列卡片的证据范围

### 9.1 0.9B 卡片公开了什么

同系列 [K2-Horizon-0.9B 卡片](https://huggingface.co/IFM/K2-Horizon-0.9B) 的 training overview 描述了另一条流程：RL 阶段从 midtraining checkpoint 分出 `math1`、`code1`、`math2a`、`math2b`、`code2`、`IF`、`stem` 七个领域专家，随后合并；之后有一个名为 `MOPD` 的阶段，卡片把目的描述为缓解权重合并造成的结构干扰与性能下降，并通过 on-policy distillation 在行为空间对齐多领域能力。

这给出一个很好的训练直觉：

```text
不同领域的 specialist RL
        -> 权重合并，得到一个统一 checkpoint
        -> 可能出现参数干扰
        -> 用当前 policy 的 on-policy 行为再做对齐/蒸馏
```

### 9.2 不能把 MOPD 直接套给 36B

36B MoVA 卡片的 training table 列出 pretraining、四个 midtraining stage 和两个 SFT phase，但没有列出 0.9B 卡片那样的七分支 RL expert merge + MOPD 阶段。因此本章只能做如下表达：

- “0.9B 同系列卡片公开了 MOPD 名称和高层目的。”
- “MOPD 说明了专家合并后进行行为层面恢复的一个公开案例。”
- “没有证据证明 36B MoVA 使用完全相同的 MOPD recipe。”

还要区分两种 expert：

| 概念 | 发生时间 | 发生什么 |
|---|---|---|
| 运行时 MoE/MoVA expert | 每次 forward | token 动态选择保留专家边界 |
| RL specialist merge | 训练后处理/发布流程 | 多个 checkpoint 的权重被合成一个 checkpoint |
| MOPD | 0.9B 卡片披露的后续阶段 | 以 on-policy 行为缓解合并后的干扰，细节未完整公开 |

“都有 expert”不等于“是同一个算法”。这是架构阅读和训练 recipe 阅读中很常见的偷换。

## 10. Serving：模型结构最终要落到启动参数

### 10.1 vLLM 路径

模型卡给出的 vLLM 主线包含 TP=2、expert parallel、BF16、remote code，以及 `k2_horizon` reasoning/tool parser：

```shell
vllm serve IFM/K2-Horizon-MoVA-36B-A4B \
  --revision main \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --trust-remote-code \
  --dtype bfloat16 \
  --reasoning-parser k2_horizon \
  --tool-call-parser k2_horizon \
  --enable-auto-tool-choice
```

这里的 `main` 是示例分支，不是永久 revision。评测应把 `--revision` 固定到具体 branch/commit，并保存 tokenizer、chat template、parser 和后端版本。

### 10.2 SGLang 路径

模型卡引用的 SGLang 配方在 2 张 H200 上验证，使用 TP=2、EP=2、FlashAttention-3 和路由 GEMM override：

```shell
python3 -m sglang.launch_server \
  --model-path IFM/K2-Horizon-MoVA-36B-A4B \
  --revision main \
  --tp 2 \
  --ep 2 \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --json-model-override-args '{"xllm_source_router_gemm_partitions":2}' \
  --reasoning-parser k2_horizon \
  --tool-call-parser k2_horizon \
  --host 0.0.0.0 --port 30000
```

`xllm_source_router_gemm_partitions=2` 是运行时路由 GEMM 分片的兼容设置，不是把 expert 数量改成两份。换 GPU、TP/EP 拓扑、FlashAttention 版本或通信库后，必须重新测：

1. 权重装载峰值和稳定显存。
2. expert dispatch 的通信占比与负载倾斜。
3. TTFT、TPOT、吞吐和长请求 p95。
4. reasoning/tool parser 的格式正确率。
5. 不同 revision 与不同后端的输出一致性。

### 10.3 请求协议也属于模型接入的一部分

模型卡推荐 `reasoning_effort="high"`、`temperature=1.0`、`top_p=0.95`，思考内容返回 `reasoning_content`，答案返回 `content`。工具格式支持 `json`、`xml`、`xml_typed`，默认是 `xml`。

这仍然是协议字段，不等于模型天然拥有网络、文件或终端权限。工具执行器、沙箱、审批、超时、回滚和审计由宿主系统负责。

## 11. Uno：K2 7B 周边的 diffusion adapter

### 11.1 它从哪里来

`K2-Horizon-7B-Uno` 不是本轮两个排行榜中的独立候选行。它是在 K2 7B 官方资料链上追到的关联 adapter，并由 [Uno 论文](https://arxiv.org/abs/2609.04010)解释其通用方法。本项目将它记录为“锚点周边技术”，而不是新模型发现。

这一点很重要：Artificial Analysis 的 K2 Horizon 条目负责告诉我们“值得围绕 K2 查什么”；官方模型卡和论文负责告诉我们“Uno adapter 做了什么”。两种来源的职责不能反过来。

### 11.2 AR pathway 和 diffusion pathway

Uno 的核心拆分是：

```text
冻结的 K2-Horizon-7B AR weights
          +
轻量 diffusion LoRA weights
          -> 并行 draft
          -> AR rejection verification
          -> 接受或回退
```

它不是把 36B MoVA 改成 diffusion model，也不是单独训练一个完全不同的 draft model。按照官方关联资料的描述，base AR 路径使用冻结的 K2-Horizon-7B，diffusion 路径通过 LoRA adapter 生成并行草稿；`Psi-Spec` 负责把 draft 和 AR 验证组合起来。

论文使用“lossless”时有明确的采样协议语境：目标是保持底层 AR 模型分布，而不是承诺任何硬件、batch、量化、parser 或工具 harness 下都没有质量损失。接受率、有效 token、回退次数、验证成本和 p95 仍需要在目标后端独立测量。

### 11.3 为什么它是 K2 的好周边技术

MoVA 主要回答“如何减少每个 token 的主要计算和扩大模型容量”；Uno 主要回答“如何让自回归 decode 更并行”。二者可以放在同一张 serving 设计图上，却不是同一层的优化：

| 技术 | 主要优化对象 | 需要绑定的 artifact |
|---|---|---|
| MoVA/MoE | token 到专家的计算路径和模型容量 | 主模型 revision、expert layout、runtime |
| Uno | draft 生成与 AR verification 的 decode 路径 | K2 7B base revision、Uno adapter revision、sampler、verifier |
| GQA/KV cache | 每层历史 K/V 的存储与读带宽 | model config、cache dtype、batch、context |

加载 Uno 时不能只记录 adapter 名称。至少要同时记录 base model、adapter revision、LoRA target modules、采样器、验证规则和后端版本；否则“同一个 Uno”可能对应不同的 base 和不同的接受率。

## 12. 证据边界和常见错误

### 12.1 当前可以说

- Artificial Analysis 2026-09-14 页面快照发现了 K2 Horizon 系列；快照字段不是官方发布日期或参数证明。
- `IFM/K2-Horizon-MoVA-36B-A4B` 官方卡片和固定 revision 支持本章的 36B/4B、48 层、GQA、MoVA、MoE、512K 配置及阶段训练表。
- 官方实现展示了 sigmoid router、bias 只用于选择、top-k 归一化/2.5 scaling、softplus attention gate 和关闭 sliding window 的语义。
- 0.9B 同系列卡片公开了 RL specialist merge 和 MOPD 的高层描述。
- Uno 是 K2 7B 的关联 adapter/论文技术，不是两个排行榜中的独立候选发现。

### 12.2 当前不能说

- 不能把 4B active 当作 4B dense 的显存、FLOPs 或端到端速度。
- 不能把 512K 配置写成“所有长文任务都可靠”或“使用 sliding window”。
- 不能把 0.9B 的 MOPD recipe 迁移成 36B MoVA 的已知训练事实。
- 不能把模型卡 benchmark、H200 吞吐或论文中的 Uno speedup 写成本项目独立复现。
- 不能把 HF 模型页、论文或 SGLang 文档出现的关联模型当成绕过排行榜的新增候选。
- IFM blog 正文本轮被 Cloudflare challenge 拦截，不能声称已经阅读正文。

## 13. 面试题

### 问题 1：MoVA 和普通 MoE 的区别是什么？

回答要点：普通 MoE 通常路由 FFN expert；MoVA 路由 attention 的 value 变换。K2 后 45 层同时有 64 value experts top-4 和 100 routed FFN experts top-8，二者的输出位置、通信和 cache 语义不同。

### 问题 2：为什么 K2 说 36B 总参数但约 4B active？

回答要点：总参数反映权重存储和模型容量；active 是每 token 主要走少数专家的计算规模代理。它不包含全部 router、shared expert、attention、通信、padding、workspace 和 KV，因此不能直接当作 4B dense 模型的延迟或显存。

### 问题 3：router bias 为什么不能直接加入最终 mixture weight？

回答要点：公开实现用 sigmoid 原始分数加 bias 做 top-k 选择，但取回原始分数作为混合权重，再归一化和乘 2.5。把 bias 带入最终权重会改变 checkpoint 的函数语义。

### 问题 4：GQA 和 MoVA 是否都在减少 KV cache？

回答要点：GQA 直接把 KV head 从 32 降到 8，带来结构性 cache head 数收益。MoVA 是 value 变换的专家路由；最终 value state 仍按层写入 cache，不能把 64 value experts 当成 64 份 cache。

### 问题 5：为什么 512K 不能只靠修改 max length？

回答要点：长上下文需要分阶段数据、位置分布、训练稳定性、packing、activation/通信预算和长距离评估共同支持。配置上限只说明接口/实现能力，不能证明任务成功率或并发。

### 问题 6：MOPD 能否证明 36B MoVA 做过七专家合并？

回答要点：不能。七专家 RL merge 和 MOPD 是 0.9B 同系列卡片明确描述的流程；36B 卡片的训练表没有列出它。可以借它解释 merge interference 和 on-policy behavior alignment，但必须标成同系列参考，不得迁移成 36B 事实。

### 问题 7：为什么 SGLang override 不是模型架构字段？

回答要点：`xllm_source_router_gemm_partitions=2` 是路由 GEMM 的运行时分片兼容设置；它改变 kernel/分片执行方式，不改变配置中的 100/64 expert 数量。换硬件和 backend 后要重新测性能与一致性。

### 问题 8：Uno 是不是 K2-Horizon-MoVA-36B 的 diffusion 版本？

回答要点：不是。Uno 是 K2-Horizon-7B 的关联 diffusion/LoRA adapter，冻结 AR base，新增轻量 diffusion path，用 `Psi-Spec` 做并行 draft 和 AR rejection verification。它属于 K2 周边 decode 加速技术，不是 36B MoVA 的架构改名，也不是排行榜独立候选。

## 14. 练习

1. 把 toy router 改成同时输出 FFN top-8 和 MoVA top-4 的 assignment histogram，分别报告两类 expert 的负载方差，不要把它们合并成 top-12。
2. 取 `B=1,T=131072` 和 `B=2,T=4096` 两种请求，按本章 KV 公式计算 BF16 K/V cache，分别加上 2-way dispatch 的 one-way payload，说明为什么 cache 和通信随不同变量增长。
3. 设计一个 `model_revision / base_revision / adapter_revision / tokenizer / parser / hardware / backend` manifest，分别用于 K2 36B 和 K2 7B Uno，缺任一字段时让评测门禁失败。
4. 为 0.9B 的 specialist merge + MOPD 画训练状态图，再为 36B 的公开训练表画状态图，标出哪些节点只有同系列参考证据。
5. 在相同任务、revision、硬件和 harness 下比较 `reasoning_effort=high` 的普通 decode 与 Uno draft/verify，至少报告有效 token、接受率、回退次数、质量、TTFT、TPOT 和 p95。

## 15. 本章来源

1. [Artificial Analysis K2 Horizon MoVA 36B A4B](https://artificialanalysis.ai/models/k2-horizon-mova-36b-a4b)：候选发现入口；页面字段按第三方快照处理。
2. [IFM/K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B)：模型卡、训练概览、checkpoint、Serving 和协议字段。
3. [固定 revision config.json](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/blob/de2d2efb32ed7639b7140bccbefe131a0063a982/config.json)：48 层、32/8 heads、64/4 MoVA、100/8 MoE、RoPE 和 gate 配置。
4. [固定 revision modeling_k2_horizon.py](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/blob/de2d2efb32ed7639b7140bccbefe131a0063a982/modeling_k2_horizon.py)：router 选择语义、MoVA value expert、GQA、attention gate、MoE 和 layer layout。
5. [K2-Horizon-0.9B model card](https://huggingface.co/IFM/K2-Horizon-0.9B)：同系列 specialist merge 和 MOPD 的高层披露；不作为 36B recipe 证据。
6. [K2-Horizon-7B-Uno model card](https://huggingface.co/IFM/K2-Horizon-7B-Uno)：K2 7B 关联 adapter 入口；不是两个排行榜的候选发现入口。
7. [Unlocking Lossless Speedups in LLMs via Discrete Diffusion](https://arxiv.org/abs/2609.04010)：Uno、Diffusion Distillation 和 Psi-Spec 的方法来源。
8. [SGLang K2 Horizon cookbook](https://docs.sglang.io/cookbook/autoregressive/IFM/K2-Horizon)：TP/EP、FlashAttention-3 和 router GEMM override 的部署入口。

