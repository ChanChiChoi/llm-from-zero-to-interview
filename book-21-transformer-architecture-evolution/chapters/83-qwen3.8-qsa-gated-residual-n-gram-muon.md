# Qwen3.8：QSA、Gated Residual、N-gram Embedding 与 Muon

> 本章核验日期：2026-09-14。Qwen3.8 的候选发现来自 [Artificial Analysis](https://artificialanalysis.ai/zh) 的具体模型页面；DataCurve DeepSWE 快照只检出 Qwen3.8 Max。模型规格主要来自 [Qwen3.8-27B 模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)、[Qwen3.8-2.4T-A95B 模型卡](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)、[Qwen3.8-Flash-Next 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) 和 [Flash-Next 技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)。报告 benchmark、kernel speedup、稳定性和训练效率数字均属于发布方自报。

## 83.1 先从一个长任务场景开始

设想一个 coding Agent 要处理一个接近百万 token 的代码仓库：它先扫描目录，修改配置，运行测试，再回到很早出现的接口定义。模型要解决两个相互冲突的问题：

1. 每个新 token 都保留全部历史，检索最精确，但 prefill 的注意力计算、KV 读取和服务延迟会随上下文迅速变重。
2. 把历史压缩成固定状态，解码便宜，但某个早期变量名、错误堆栈或工具结果可能无法被精确取回。

Qwen3.8 的架构线索可以看成一组互补动作：Gated DeltaNet（GDN）把大部分前缀压缩成递归状态；QSA 用轻量索引器挑出少量重要上下文；Gated Residual（GR）让网络沿深度保留多条不同历史；N-gram Embedding 在主干之外增加容量；Muon 与 AdamW 的分工则让这些结构能够在更大的 batch 和 learning rate 下训练。

这几项不是五个互不相关的名词。它们共同回答“容量放在哪里、历史如何保留、训练如何稳定、服务如何付费”四个问题。

## 83.2 榜单锚点与模型家族边界

### 83.2.1 发现证据不能代替官方规格

本轮具体 Artificial Analysis 页面和榜单日期是：

| 版本 | Artificial Analysis 记录 | DataCurve DeepSWE | 核验后的公开形态 |
|---|---:|---|---|
| Qwen3.8 Max | 2026-08-03 | 有 `qwen3.8-max` 条目 | Qwen Cloud hosted version；官方模型卡说明它基于 A95B |
| Qwen3.8 2.4T A95B | 2026-08-12 | 本地快照未检出 | 开源 text-only MoE checkpoint |
| Qwen3.8 27B | 2026-08-14 | 本地快照未检出 | 开源 dense native vision-language checkpoint |
| Qwen3.8-Flash-Next | 2026-08-26 | 本地快照未检出 | 开源实验性架构预览，官方称面向 Qwen4 的早期路线 |

`xhigh`、`medium`、`low` 和 `Non-reasoning` 是评测/请求配置，不是自动新增的基础模型。反过来，Cloud 页面也不能被写成一个下载到本地的独立权重版本。

### 83.2.2 一张规格表

| 版本 | 规模 | 层与 token mixing | 专家/额外容量 | 上下文和模态 |
|---|---|---|---|---|
| 27B | 27B dense | 64 层；每 4 层 3 GDN + 1 Gated Attention | 无 MoE；MTP 多步训练 | 原生 262,144，约可扩展到 1M；图像、视频 |
| 2.4T-A95B | 2.4T total、95B active | 92 层；每 4 层 3 GDN + 1 Gated Attention | 512 experts，10 routed + 1 shared | 原生 262,144，约可扩展到 1.01M；text-only |
| Flash-Next | 主模型 125B、6B/token active；另有 51B N-gram、4B MTP | 48 层；每 4 层 3 GDN + 1 QSA | 512 experts，10 routed + 1 shared；4 branch GR | 原生 262,144，约可扩展到 1M；视觉 |

27B 的 hidden size 为 5120，A95B 为 8192，Flash-Next 为 2560。这里的 active parameters 是模型卡或报告的口径，不等于单卡显存、严格 FLOPs 或端到端 tokens/s。下文所有资源计算都把 total weights、active compute、cache、通信和额外 table 分开。

## 83.3 GDN 与 Gated Attention：压缩历史，但不放弃检索

### 83.3.1 从普通注意力的成本出发

标准 causal attention 在位置 `t` 读取从 0 到 `t` 的全部 key/value。它的优势是“我还保留每个历史 token”，代价是长上下文 prefill 需要大量 query-key pair，并在 decode 时反复读越来越长的 KV。

线性注意力可以把历史写入一个固定大小状态 `S`。但固定状态不是无限记忆：不同 token 可能竞争同一组状态方向，模型必须决定什么该衰减、什么该覆盖。因此 Qwen3.8 使用的是带门控的 delta memory，而不是单纯把所有 `k v^T` 相加。

### 83.3.2 Gated DeltaNet 的更新

对一个 head，令 `S_t` 为 `d_k × d_v` 的状态：

```math
\widetilde{S}_{t-1}=\alpha_t S_{t-1}
```

```math
e_t=v_t-\widetilde{S}_{t-1}^{\top}k_t
```

```math
S_t=\widetilde{S}_{t-1}+\beta_t k_t e_t^{\top}
```

```math
y_t=S_t^{\top}q_t
```

逐步解释：

- `alpha_t` 是已有状态的生命周期控制器。它可以让旧信息整体衰减。
- `e_t` 是当前 key 对应的“还缺多少 value”。如果状态已经能回答当前 key，误差就小。
- `beta_t` 控制本次修正的写入力度。
- `k_t e_t^T` 只沿当前 key 的方向写入误差，避免无条件叠加重复关联。

所以 GDN 同时有遗忘、定向擦除和写入。它不是普通 RNN 的标量 hidden state，也不是一个把历史 token 做平均的池化器。

### 83.3.3 为什么要周期性显式注意力

GDN 的状态大小固定，适合持续处理长前缀；但它不能保证在任意长距离上逐 token 精确恢复。Qwen3.8-27B 和 A95B 的模型卡把层布局写成每四层三层 GDN、一层 Gated Attention：

```text
GDN -> GDN -> GDN -> Gated Attention
```

这是一种“多数层便宜记忆、少数层精确检索”的预算分配。Gated Attention 还用门控控制显式 attention 分支的幅度，避免它在深层 residual 中无限放大。

Flash-Next 在继续预训练时把这些 full-attention 层换成 QSA。这里要区分两个说法：Flash 的总体模型卡仍把 token mixing 描述为 GDN + QSA；QSA 的核心仍是显式 attention，只是先通过 indexer 选择稀疏上下文。不能把“有 GDN”简写成“整个模型已经没有 attention”。

## 83.4 QSA：micro-block 级别的稀疏显式注意力

### 83.4.1 先压缩 key，再决定看哪里

QSA 的目标是把“为每个 query 对一百万个 token 打分”变成“先对一批小 block 做便宜筛选，再只展开少数 token”。设 block 大小为 `r`，长度为 `n` 的 key 被划分为不重叠的 micro-block。

Flash-Next 的公开配置给出：

- indexer 使用 4 个 query heads 和 1 个 shared key head，即 MQA 形式。
- `r=4`，每四个 token 压成一个 block 表示。
- token budget `K=2048`，因此最多选择 `ceil(2048/4)=512` 个完整 block。
- 当前 query 所在的未完成尾部 token 额外保留。

上面的 `ceil(K/r)` 是报告语境下的完整 block 候选上限；本章 demo 为了验证“总返回 token 不超过预算”，把 causal tail 也计入 `token_budget`，所以会先扣除 tail，再用 `floor((K-|tail|)/r)` 计算可选完整 block 数。生产实现应以报告/源码定义的预算口径为准。

### 83.4.2 为什么平均池化必须早于 RoPE

一个 rotary position embedding 会给不同位置施加不同相位。如果先把每个 key 旋转，再平均四个不同相位的向量，结果既含内容平均，也含相位抵消。QSA 先做内容压缩，再给 block 的起始位置加 partial RoPE：

```math
\bar{k}_b=\mathrm{RMSNorm}\left(\mathrm{AvgPool}(k_{br:(b+1)r-1})\right)
```

```math
\bar{k}^{\mathrm{rope}}_b=\mathrm{PRoPE}(\bar{k}_b, br)
```

query 保留自己的 token position。这个顺序是一个很适合面试深挖的实现细节：压缩不只是减少数组长度，还要避免破坏位置语义。

### 83.4.3 block-causal scoring

对于 query `i` 和 block `b`，只有当这个 block 已经完整出现在 query 之前，indexer 才能对它评分。报告的抽象形式是：

```math
I_{i,b}=\sum_h\mathrm{ReLU}\left(\langle q_{i,h},\bar{k}_{b,h}\rangle\right),
\qquad br+r-1\leq i
```

ReLU 让负相似度不贡献重要性；block-causal 条件避免 query 偷看尚未完整暴露的 block。之后取分数最高的 block：

```math
B_i=\mathrm{TopK}_{\lceil K/r\rceil}\left(\{I_{i,b}\}_b\right)
```

展开 `B_i` 的 token index，合并尾部 token，才得到核心 attention 的可见集合 `S_i`。它不是“选 512 个 token”，而是先选最多 512 个 block，再展开至 token 预算；block 边界和尾部规则会影响检索召回。

### 83.4.4 两阶段训练解决候选选择和主干适应

如果直接在 full-attention checkpoint 上启用 QSA，indexer 可能选错上下文，主干也没有学过这种稀疏分布。Flash-Next 技术报告的两阶段流程是：

**Stage 1：dense distillation。** 先让 full-attention teacher 对每个 query 产生 token-level attention 分布。在每个 block 内使用 max pooling，而不是平均 pooling，以保留一个尖锐但重要的 token；再用 KL loss 让 indexer 的 block scores 接近 teacher 分布。报告设置约 1,000 steps、8 条 256K 序列、约 2B tokens。

**Stage 2：sparse training。** 启用稀疏核心 attention，联合更新 backbone 和 indexer，使主干逐渐适应被选出的 token 集合。报告设置约 8,000 steps、96 条 256K 序列、约 200B tokens。

这也解释了为什么“有一个 indexer”不足以复现 QSA。需要同时固定 teacher 分布、block 因果 mask、max pooling、top-k、尾部 token、联合训练和 kernel。

### 83.4.5 复杂度和报告数字的正确读法

报告把压缩 indexer 的复杂度写成约 `O(n^2/r)`；压缩倍率会降低平方项的常数，但不等于任意 `n` 都是线性。若只看固定 `K` 的核心 sparse attention，可以把它教学式地写成 `O(nK)`，但完整成本仍包含：

- indexer 的 query-key scoring。
- block top-k 和 token 展开。
- QSA core attention 的 sparse gather。
- 分页、padding、batch 和 kernel launch。
- MTP 多步中 index 复用的缓存一致性。

技术报告自报的对照结果是：full attention 与 QSA 的九项平均分从 75.9 到 76.8；512K--1M RULER 从 90.08 到 93.00；1M 的 8-needle MRCR 从 20.71 到 26.44；1M kernel-level prefill 约 7.6x、decode 约 4.9x。这里的 prefill 使用 16K chunk、batch size 1，decode 使用 batch size 4 和三个额外 MTP steps，并以 FlashInfer 实现作基线。它们是发布方自报，不是本项目独立测量。

## 83.5 Gated Residual：让四条历史通道各司其职

### 83.5.1 为什么加宽 residual stream

标准 residual 的优点是每个 block 都有一条直达输出的路径，缺点是所有层共享同一个累加器。早期写入的信息必须和后续所有写入混在一起，后面的 block 没有办法同时表达“我想保留第一层的一个 feature”和“我想使用最近一层的另一个 feature”。

把 residual 改成四个 branch 可以提供四个不同的积累轨迹，但如果每个 block 简单地把四个 branch 相加，又只是在增加内存和带宽。GR 的重点是让读取粒度足够细。

### 83.5.2 read gate 与 write gate

设四条 residual branch 为 `R_i`，每条 branch 先单独 RMSNorm：

```math
\widehat{R}_i=\mathrm{RMSNorm}(R_i;\gamma_i)
```

所有 branch 拼接后预测逐元素 read gate `G_i`，再形成 block 输入：

```math
G=\mathrm{unvec}\left(\sigma\left(W_u\,\mathrm{SiLU}\left(\frac{1}{n_r}W_d\,\mathrm{vec}(\widehat{R})\right)\right)\right)
```

```math
x=\frac{1}{n_r}\sum_i G_i\odot\widehat{R}_i
```

block 输出为 `y=F(x)`，写回每个 branch 时只使用 branch-level scalar：

```math
s=2\sigma\left(\frac{1}{n_r}W_w\mathrm{vec}(\widehat{R})\right),
\qquad R'_i=R_i+s_i y
```

于是 read 是 channel 粒度的，write 是 branch 粒度的。GR 删除 `Hres` branch mixing operator，避免每个 block 再把完整 residual state 读出来做矩阵混合；read 中已有 normalization，所以它可以直接承担 pre-norm 的作用。

### 83.5.3 和 AttnRes、mHC 的边界

这几个名字经常在面试中被放到同一张表，但它们优化的方向不同：

| 方法 | 主要对象 | 关键取舍 |
|---|---|---|
| Attention Residuals | 不同深度的历史 sublayer 输出 | 用 attention 选择“读哪一层”，表达力强，需管理深度方向的缓存/通信 |
| mHC | 多分支 residual 之间的流动 | 用双随机矩阵约束 branch mixing，强调稳定性和封闭性 |
| Gated Residual | 四分支 residual 的读写 | read 用逐元素 gate，write 用 branch scalar，删除 `Hres`，减少 memory traffic |

报告还评估了 FP8 residual storage 和 fused read/write。FP8 的收益来自减少 widened residual 的搬运字节，不是因为“低精度一定更快”；必须测量量化误差、激活范围、kernel 融合和 post-training 回归。

## 83.6 N-gram Embedding：容量、地址和带宽的重新分配

### 83.6.1 它到底查什么表

若当前 token 序列为 `a, b, c`，模型可以用 `b,c` 或 `a,b,c` 这样的局部 n-gram 形成地址，从 embedding table 取一个向量，并把它加入当前 token representation。它依赖当前上下文的局部离散索引，而不是外部文档。

Flash-Next 的公开配置给出 Layer 2、bigrams/trigrams 和 20,000,000 个 n-gram slots；模型卡和报告给出的额外参数规模约为 51B。由于地址是确定的，表可以从 accelerator offload 到 host memory，再用 asynchronous prefetch 与主干第一层计算重叠。

### 83.6.2 和 RAG、KV cache 的区别

| 机制 | 保存的东西 | 是否外部知识 | 读写时机 |
|---|---|---|---|
| N-gram Embedding | 随局部 token 地址索引的训练参数 | 否 | 每个 token 查表，训练后固定在模型中 |
| RAG | 检索到的文档片段 | 是 | 请求级检索，内容会变化 |
| KV cache | 当前请求已有层的 K/V | 否 | prefill 写入，decode 复用 |

N-gram 的主要工程问题不是额外的 dense FLOPs，而是表容量、随机读、预取命中、host-device 带宽和多租户隔离。它是“把参数放到算力之外”的容量扩展路线。

### 83.6.3 为什么 loss 下降不等于下游全赢

报告在固定总参数预算和增加总参数两种设置下都强调了 loss 与 benchmark 的分离：固定预算时，约 10x base vocabulary 的局部 loss 较优，但下游没有稳定的全面提升；增加 vocabulary 时 loss 近似单调下降，下游结果却会饱和或波动，中文评测的一些切片改善更明显。

所以实验报告至少要同时写：训练 loss、uncheatable/out-of-domain PPL、通用能力、代码、推理、长上下文和单位成功成本。只看 loss 会把一个更大的 lookup table 误判成普遍能力提升。

## 83.7 Muon + AdamW：优化器也属于架构设计

### 83.7.1 为什么不全用一种优化器

Muon 的核心是对矩阵参数的 momentum 做 Newton-Schulz orthogonalization，使更新方向更接近矩阵几何上的均衡方向。它更适合真正承担二维线性映射的权重，不一定适合词表、router 或极其细长的低秩矩阵。

Flash-Next 报告的参数分工如下：

| 参数类别 | 优化器 | 原因/证据边界 |
|---|---|---|
| attention q/k/v/o、GDN input/output、MoE expert FC、N-gram key/value projection | Muon | 矩阵作为线性映射；使用 8 次 Newton-Schulz |
| input embedding、LM head、MoE router、GR low-rank projections | AdamW | router 早期容易被 Muon 放大波动；细长低秩投影的消融更适合 AdamW |
| N-gram embedding table | Adam，关闭 weight decay | 查表参数不是同一种线性映射 |

Nesterov momentum 的 `mu=0.95` 和 8 次 Newton-Schulz 是报告配置。这里不把报告 OCR 中未稳定呈现的 shape scaling 精确公式写入结论；复现时应直接以原始报告/源码和 checkpoint 对照。

### 83.7.2 fused parameter 必须先按语义拆分

Megatron 风格实现可能把 QKV 存成一个 fused matrix，把 SwiGLU 的 gate/up 存成一个 `fc1`。但这些矩阵在语义上是多个独立线性算子。若直接对拼接矩阵做正交化，会发生两件坏事：

1. 不相关子块的奇异方向被混合。
2. 矩阵形状改变，更新缩放不再对应真实算子。

报告因此先在 per-head 或 gate/up 粒度拆分，分别做 Newton-Schulz，再 gather 回原布局。拆分后小矩阵很多，单步容易被 kernel launch 限制，所以用 CUDA graph 组织优化器步骤。

跨 TP/DP 分片时，还需要让每个逻辑 Muon 矩阵被完整重构，并均衡不同矩阵的正交化 FLOPs。报告把这一分布式优化器工作称为 Canzona。它提示一个很实用的面试结论：优化器选择会改变并行布局、通信和 kernel 设计，不只是替换一个 `optimizer.step()`。

### 83.7.3 scaling law 与 batch-size warmup

报告称新架构与 Muon 一起把近优 batch size 和 learning rate 推高，并重拟 scaling law。在一个 batch-size 实验中，逐渐从小 batch ramp 到目标 batch 没有优于一开始直接使用目标 batch；同样 token budget 下 warmup 还多约 18.8% optimizer steps。于是 Flash-Next 的生产 recipe 不使用传统 batch-size warmup。

这不是“所有 Muon 训练都不需要 warmup”的定理。正确表达是：在该报告的数据、模型规模、学习率、token budget 和稳定性测试下，作者选择了 constant target batch；迁移到新模型和硬件需重新测 loss、gradient norm、overflow、吞吐和 checkpoint resume。

## 83.8 Thinking protocol：把推理预算和模型身份分开

### 83.8.1 三个相似字段各自解决什么

- `enable_thinking`：是否输出 thinking block。27B 和 Flash-Next 可以按请求关闭；A95B 模型卡明确不能关闭。
- `reasoning_effort`：请求级推理深度/预算档位。`low`、`medium`、`xhigh` 是不同调用条件，不是新的 checkpoint。
- `preserve_thinking`：是否把历史消息中的 thinking block 保留到后续上下文。它是对话协议和上下文组织字段，不等于内部永久记忆。

27B、A95B 和 Flash-Next 都默认保留 thinking context，但 A95B 的 thinking 是强制的。Qwen Cloud 的参数封装方式可能与本地 Transformers/vLLM 的 `chat_template_kwargs` 不同，迁移时需以对应服务文档和实际响应回归为准。

### 83.8.2 为什么低 effort 可能让长任务更慢

对独立问答，减少 reasoning token 常常能降低单轮延迟。对 Agent 长任务，低 effort 可能造成：

1. 计划不完整，工具调用顺序错误。
2. 代码修改后没有充分运行测试。
3. 环境反馈没有被整合，下一轮重复失败。
4. harness 触发更多重试或上下文恢复。

所以应报告总任务耗时和单位成功成本，而不只是每轮 TPOT：

```math
\text{unit success cost}=\frac{\text{input cost}+\text{output/reasoning cost}+\text{tool/runtime cost}}{\text{successful tasks}}
```

若没有成功任务，分母为空，指标应为 `not_applicable`，不能写成零成本。

## 83.9 长上下文 Serving 的成本账本

### 83.9.1 三本参数账

Flash-Next 至少要同时记录：

```text
主干权重：125B total
主干每 token 激活：约 6B active proxy
N-gram table：约 51B，模型卡还列 20M slots
MTP：模型卡另列约 4B 参数
```

第一行决定权重存储和分片，第二行近似主要 token 计算，第三行决定额外 table 的存储/带宽，第四行影响 speculative draft 路径。它们不能相加后再把总数当 FLOPs，也不能拿 6B active 直接估算单卡显存。

### 83.9.2 QSA、GDN 和 cache 的不同账目

- GDN 解码状态是固定形状的每头状态，主要代价是状态读写和递归计算。
- QSA 仍要支付 indexer、top-k 和 sparse gather；它不是“零成本只取 2048 token”。
- MTP 可能复用 QSA top-k indices，但需要保证多个预测步的状态和位置语义一致。
- N-gram table 的表参数不等于 KV cache，host-memory prefetch 也会受到带宽和并发冲突影响。
- QSA 层的显式 attention 仍需保存/读取其实际 cache；不能因为 GDN 有固定状态，就把整模型 cache 当成常数。

### 83.9.3 一个教学级 KV/通信近似

对仍然保存显式 K/V 的层，可以先写一个粗略 cache 账本：

```math
M_{KV}\approx B\times T\times L\times H_{KV}\times d\times 2\times \text{bytes}
```

`B` 是 batch，`T` 是 token 数，`L` 是需要该 cache 的层数，`H_KV` 是 KV heads，`d` 是 head dimension，2 表示 K 和 V。它只覆盖显式 cache，不包括页表、indexer、tail、workspace、权重和 residual branches。

MoE 的 dispatch 也要单独估算：

```math
N_{assign}\approx B\times T\times k_{route}\times L_{sparse}
```

对于 Flash-Next，FFN 每 token 选择 10 个 routed experts，还要计算 shared expert；具体通信 payload 还取决于 hidden size、EP 拓扑、padding、local hit、all-to-all 实现和 overlap。active parameter 只是计算口径的起点。

### 83.9.4 Hosted Max 不能用 open checkpoint 账本替代

官方模型卡说明 Qwen3.8-Max 基于 Qwen3.8-2.4T-A95B，并给出视觉输入、非 thinking、默认 1M 和 built-in tools 等托管特性。它可以作为产品路线的证据，但不能据此假定：

- hosted 端暴露了全部 open checkpoint 参数。
- 线上 context、限流、缓存和工具实现与本地后端相同。
- 27B/Flash 的 QSA、GDN、MTP 细节自动适用于 Max。

面试中应把“基础模型关系”“服务端能力”“线上实测”列成三列。

## 83.10 零依赖 QSA/GR 教学 demo

下面的程序只用 Python 标准库。它做四件事：对 block 做压缩和因果选择，保留尾部 token；模拟四分支 GR 的 read/write；计算 total/active/N-gram 参数账本；用断言锁定教学不变量。它没有实现 RoPE、softmax、真实 kernel、MoE all-to-all 或 GPU offload，不应被当作生产实现。

```python
from math import sqrt


def dot(left, right):
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    return sum(x * y for x, y in zip(left, right))


def average_blocks(keys, block_size):
    if not keys or block_size <= 0:
        raise ValueError("keys and block_size must be positive")
    blocks = []
    for start in range(0, len(keys) - block_size + 1, block_size):
        block = keys[start:start + block_size]
        blocks.append([sum(row[col] for row in block) / block_size
                       for col in range(len(block[0]))])
    return blocks


def qsa_select(keys, query, query_index, block_size=4, token_budget=8):
    if not keys or not query:
        raise ValueError("keys and query must be non-empty")
    if not 0 <= query_index < len(keys):
        raise ValueError("query_index is out of range")
    if block_size <= 0 or token_budget <= 0:
        raise ValueError("block_size and token_budget must be positive")

    complete_blocks = average_blocks(keys, block_size)
    scored = []
    for block_id, block_key in enumerate(complete_blocks):
        block_end = (block_id + 1) * block_size - 1
        if block_end <= query_index:
            scored.append((max(0.0, dot(query, block_key)), block_id))

    # Reserve the incomplete causal tail before selecting complete blocks.
    tail_start = ((query_index + 1) // block_size) * block_size
    tail = list(range(tail_start, query_index + 1))
    if len(tail) > token_budget:
        raise ValueError("token_budget must fit the causal tail")
    complete_token_budget = token_budget - len(tail)
    block_budget = complete_token_budget // block_size
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = [block_id for _, block_id in scored[:block_budget]]
    selected = set()
    for block_id in chosen:
        start = block_id * block_size
        selected.update(range(start, min(start + block_size, len(keys))))

    # Keep the currently incomplete causal tail, as QSA does.
    selected.update(tail)
    ordered = sorted(selected)
    return ordered, chosen


def rms(vector, eps=1e-6):
    if not vector:
        raise ValueError("vector must be non-empty")
    return sqrt(sum(value * value for value in vector) / len(vector) + eps)


def gated_residual(branches, read_gates, write_gates, block_output):
    if len(branches) != 4 or len(read_gates) != 4 or len(write_gates) != 4:
        raise ValueError("the teaching model uses four branches")
    if not block_output:
        raise ValueError("block_output must be non-empty")
    width = len(block_output)
    if any(len(branch) != width for branch in branches):
        raise ValueError("all branches must have the same width")
    if any(len(gate) != width for gate in read_gates):
        raise ValueError("read gates must match branch width")

    normalized = []
    for branch in branches:
        scale = rms(branch)
        normalized.append([value / scale for value in branch])
    read = [sum(normalized[index][col] * read_gates[index][col]
                for index in range(4)) / 4.0
            for col in range(width)]
    written = [[branches[index][col] + write_gates[index] * block_output[col]
                for col in range(width)] for index in range(4)]
    return read, written


def parameter_ledger(main_total_b, active_b, ngram_b, mtp_b):
    values = {
        "main_total_b": main_total_b,
        "active_b": active_b,
        "ngram_b": ngram_b,
        "mtp_b": mtp_b,
    }
    if any(value < 0 for value in values.values()):
        raise ValueError("parameter values must be non-negative")
    values["storage_b_proxy"] = main_total_b + ngram_b + mtp_b
    return values


keys = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9],
        [1.0, 1.0], [0.8, 0.7], [0.2, 0.3], [0.4, 0.2],
        [0.9, 0.8], [0.1, 0.2], [0.7, 0.9]]
selected, blocks = qsa_select(
    keys, query=[1.0, 0.8], query_index=9,
    block_size=4, token_budget=6,
)
assert all(index <= 9 for index in selected)
assert len(selected) <= 6
assert selected[-2:] == [8, 9]

branches = [[1.0, 2.0], [2.0, 1.0], [0.5, 1.5], [1.5, 0.5]]
read, written = gated_residual(
    branches,
    read_gates=[[1.0, 0.5], [0.5, 1.0], [1.0, 1.0], [0.25, 0.75]],
    write_gates=[0.5, 0.0, 1.0, 0.25],
    block_output=[2.0, 3.0],
)
assert len(read) == 2 and len(written) == 4
ledger = parameter_ledger(125, 6, 51, 4)
assert ledger["storage_b_proxy"] == 180
print("qsa_blocks=", blocks)
print("qsa_tokens=", selected)
print("gr_read=", [round(value, 3) for value in read])
print("gr_branch_count=", len(written))
print("parameter_ledger=", ledger)
```

运行现象应该说明三件事：QSA 只返回因果可见的有限 token，GR 的 branch 数保持为四且 read/write 粒度不同，参数账本把 125B 主干、6B active、51B N-gram 和 4B MTP 分列。这个 demo 没有把 active 参数当成 storage，也没有把 QSA 选择当成真实 GPU latency。

## 83.11 面试追问与实验设计

### 追问 1：QSA 为什么不是简单的 top-k token attention？

回答应包含三步：先在 micro-block 上压缩 key，再用 block-causal indexer 选 block，最后展开为 token 并合并未完成尾部。`K=2048`、`r=4` 意味着最多 512 个完整 block，不是一次只算 512 个 token。复杂度还要加 indexer、top-k 和 sparse gather。

### 追问 2：为什么 QSA 要先池化再做 RoPE？

先旋转再平均会把不同位置的 rotary phase 混在一起，导致 block 表示的内容和位置信息相互抵消。先做内容平均，再给 block 起始位置做 partial RoPE，可以让压缩表示只承担一个明确的 block position。

### 追问 3：GDN 与 QSA 是竞争关系还是互补关系？

互补。GDN 以固定状态低成本吸收大部分前缀，适合递归处理；QSA 保留稀疏显式 token 检索，补回固定状态不能精确表达的远端证据。前者节省状态成本，后者维护检索能力。

### 追问 4：GR 和 mHC 的区别是什么？

mHC 的重点是约束 branch mixing 的双随机结构；GR 的重点是四分支 residual 上的逐元素 read gate 和 branch scalar write，并删除 `Hres`。二者都在稳定深层残差流，但读写自由度、通信和 kernel 代价不同。

### 追问 5：为什么 N-gram embedding 不是 RAG？

N-gram 只按当前 token 局部上下文索引训练好的内部 table，不从外部文档检索，不改变请求级知识来源；它也不是 KV cache。它把参数容量从 accelerator 主干计算中部分移到 table storage 和带宽。

### 追问 6：为什么 Muon 不能覆盖全部参数？

Muon 的几何假设更适合二维线性映射。embedding table、router 和某些细长低秩 projection 的结构不同，报告的消融发现 AdamW/Adam 更合适；fused matrix 还必须按语义拆分后再正交化。优化器分工是 checkpoint 与训练系统的一部分。

### 追问 7：A95B 和 Max 是不是两个独立模型？

A95B 是公开 text-only MoE checkpoint；Max 是官方页面说明基于它的托管版本，加入视觉、非 thinking、默认 1M 和工具等产品能力。不能把 Cloud service 当成另一个公开 open checkpoint，也不能把 hosted 行为全部外推到本地权重。

### 建议实验

1. **QSA 召回消融**：固定随机种子和 teacher，比较 `r=1/2/4/8`、max/average pooling、是否保留 tail，记录 salient-token recall、RULER/needle 检索、indexer latency 和核心 attention latency。
2. **QSA 训练阶段消融**：比较只做 dense distillation、只启用 sparse、两阶段联合训练，报告短上下文 loss、长上下文 recall、MTP accepted length 和 post-training 回归。
3. **GR 稳定性实验**：比较单 residual、四分支无 gate、四分支 scalar gate 和 GR，固定 batch 与 learning rate，记录 activation max、gradient p99.9、loss spikes、residual bytes 和 downstream accuracy。
4. **N-gram 容量账本**：固定总参数预算与 MoE 预算，改变 table vocabulary，分别报告 loss、out-of-domain PPL、代码/中文切片、host bandwidth、prefetch miss 和单位成功成本。
5. **Muon 分工实验**：比较全 AdamW、按报告分工、错误地对 fused matrix 直接正交化，检查 loss、router overflow、梯度尖峰、optimizer step time 和 resume 一致性。

所有实验都要固定模型 revision、tokenizer、后端、硬件、batch、context、effort、工具和评测 harness。报告发布方自报结果时，必须和本地结果分栏。

## 83.12 来源与待核验边界

- [Qwen3.8-27B 模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)：27B dense、视觉、64 层、GDN/Gated Attention、thinking 控制和 MTP 字段。
- [Qwen3.8-2.4T-A95B 模型卡](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)：2.4T/95B active、92 层、512 experts、text-only 和强制 thinking。
- [Qwen3.8-Flash-Next 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)：125B/6B active、51B N-gram、QSA/GR/MTP、视觉、配置和服务入口。
- [Qwen3.8-Flash-Next GitHub](https://github.com/QwenLM/Qwen3.8-Flash-Next)：实验性架构概览、代码和技术报告入口。
- [Flash-Next 技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)：GDN、QSA、GR、N-gram、Muon、scaling law、稳定性和评测；28 页报告中的速度、loss、benchmark 与稳定性为作者自报。
- [Artificial Analysis Qwen3.8 Max](https://artificialanalysis.ai/models/qwen3-8-max)、[27B](https://artificialanalysis.ai/models/qwen3-8-27b)、[A95B](https://artificialanalysis.ai/models/qwen3-8-2-4t-a95b)、[Flash-Next](https://artificialanalysis.ai/models/qwen3-8-flash-next)：本章的候选发现来源。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/)：本轮快照只辅助确认 Qwen3.8 Max 条目。

截至本章核验日期，完整生产 kernel、目标硬件 profiling、线上接受率、全系列训练配方、host-memory 预取在不同服务环境下的行为，以及 independent benchmark 仍待核验。没有这些证据，不能把报告中的 `7.6x`、`4.9x`、loss 或 Agent 结果写成普遍保证。

## 83.13 Qwen3.8 Max (0902)：把 revision 当作服务契约

Qwen3.8 Max (0902) 是本章架构主线的一个重要边界案例。Artificial Analysis 将 canonical `qwen3-8-max` 展示为 `Qwen3.8 Max (0902)`，Qwen Cloud 官方页则给出 alias `qwen3.8-max-2026-09-02`，并称它是 `qwen3.8-max` 的 upgraded snapshot。这里的“新”首先发生在 hosted service 的 revision 和接口契约上，而不是已公开一个新的本地权重架构。

官方页面给出的资源边界是：context `1M`、普通最大输入 `991K`、thinking 最大输入 `983K`、最大输出 `131K`。这几个数字不能直接相加来估计并发，因为 reasoning token、工具 schema/结果、缓存、KV、workspace、batch 和 provider 限流也会占用预算。AA 页面约 `984K` 的第三方 context 字段只能作为榜单目录字段，不能覆盖官方 API 的模式级上限。

0902 的高频面试点是协议组合：`reasoning_effort` 可选 `low/medium/xhigh`，默认 `xhigh`，且不能与 `thinking_budget` 同时设置；thinking 模式下 `tool_choice` 只能是 `auto` 或 `none`，需要强制选择某个工具时必须关闭 thinking。也就是说，effort 是请求级预算控制，tool choice 是接口约束，二者都不应被记录成新的模型架构。多模态调用还要遵循 `MultiModalConversation` 接口，不能把普通文本消息模板直接外推。

Context Cache 文档又把 Qwen Max 的缓存拆成 explicit、implicit 和 session 三类。它们的 owner、生命周期、命中、计费和失效语义不同，最小缓存长度为 `1,024` tokens；这不等于三种“永久 GPU KV cache”。在 serving 设计中，cache identity 至少要与模型 revision、tokenizer/template、租户、权限、输入前缀和会话状态关联，并在 trace 中记录命中、失效、重算和实际费用。

产品页对 coding、工程规模项目、长周期 autonomous development、多工具 Agent 和视觉理解的描述应转成可验收的任务契约：固定 0902 alias、effort、工具、环境、超时、verifier 和输出预算，测任务成功率、恢复率、TTFT、TPOT、p95、cache hit 和单位成功成本。它们不能直接被写成 QSA、GDN、MoE、训练数据或生产 kernel 的证据。

本轮 DataCurve 只有 `mini_swe_agent_qwen3_8_max_xhigh` 泛化行，没有 0902 精确行。因此其 Pass@1、成本、输出 token 和 Agent steps 只能作为 Qwen3.8 Max 系列的 harness 参考，不能迁移为 0902 revision 的独立模型分数。0902 的参数、层排布、专属训练报告、线上 acceptance rate 和目标硬件 profiling 仍待核验。

## 83.14 Qwen3.5-397B-A17B：Qwen3.8 架构演进的前置锚点

Qwen3.5-397B-A17B 是理解 Qwen3.8 的更早锚点。它已经出现在 Artificial Analysis，页面的 Reasoning/Non-reasoning 是同一基础模型的运行配置；DataCurve 当前没有精确的 `mini_swe_agent_qwen3_5_397b_a17b_*` 行，因此不能把其他 Qwen 模型的 Agent 评测迁移到这里。官方模型卡、固定 `config.json`、Qwen 官方仓库和发布博客则提供了 397B/17B、混合架构、原生视觉和 serving 的核验字段。

### 83.14.1 公开结构：Gated DeltaNet、Gated Attention 与 MoE

模型卡给出的最小结构账本如下：

| 字段 | Qwen3.5-397B-A17B 的公开口径 | 为什么重要 |
|---|---|---|
| 参数 | 397B total、17B activated | active 参数只描述 token 的计算路径，不等于总权重、KV/state cache 或并发显存 |
| 深度与宽度 | 60 层、hidden size 4096 | 可用于粗略 FLOPs 和通信账本，不能替代完整实现配置 |
| 层布局 | 15 次重复 `3 x (Gated DeltaNet -> MoE) + 1 x (Gated Attention -> MoE)` | 递归 state 和显式 attention 周期性交替，形成 hybrid token mixing |
| Gated DeltaNet | 64 个 value heads、16 个 query/key heads、head dimension 128 | 以固定大小 state 压缩前缀，推理时不能按普通 full-attention KV 处理 |
| Gated Attention | 32 Q heads、2 KV heads、head dimension 256、RoPE dimension 64 | 提供显式历史交互；2 KV heads 带来 GQA-like KV 共享，但不公开完整 kernel |
| MoE | 512 experts，10 routed + 1 shared，expert intermediate dimension 1024 | 总容量和每 token 计算量分离，路由、通信和专家驻留必须单独计账 |

这组字段与 Qwen3.8 的公开材料构成清晰的演进关系：Qwen3.8 README 称其建立在 Qwen3.5 的架构基础上；Qwen3.8/Flash-Next 再加入 QSA、Gated Residual、N-gram Embedding 和更具体的 Muon 训练分工。不能反向把这些 Qwen3.8 细节写成 Qwen3.5 已经公开的技术。

### 83.14.2 Gated DeltaNet 的状态更新直觉

教学上，可把 Gated DeltaNet 的状态写成：

```math
\widetilde{S}_{t-1}=\alpha_t S_{t-1}
```

```math
e_t=v_t-\widetilde{S}_{t-1}^{\top}k_t
```

```math
S_t=\widetilde{S}_{t-1}+\beta_t k_t e_t^{\top}
```

```math
y_t=S_t^{\top}q_t
```

`S_t` 是固定形状的递归 state，`alpha` 控制已有信息的保留，`beta` 控制本次 delta 写入；`e_t` 表示当前 value 与 state 读出之间的误差。它的面试重点不是背公式，而是说明这条路径把历史压缩为可更新 state，因而在长序列推理中拥有不同于显式 attention 的 cache 形态。

这里的公式是理解 Gated DeltaNet 的教学抽象。官方模型卡公开的是模块名称、head layout 和层排布，并没有在本轮公开 Qwen3.5 的完整 state layout、kernel、跨卡同步或硬件 profiling。不要把 Qwen3.8 Flash-Next 报告的 QSA、Gated Residual 或 N-gram 公式回填到 Qwen3.5。

### 83.14.3 原生多模态与 Agent 训练声明

Qwen3.5 是带 vision encoder 的 causal language model。固定配置公开了 vision encoder 的部分字段：27 层、hidden size 1152、16 heads、patch size 16、temporal patch size 2，输出 hidden size 4096。模型卡和博客进一步把它定位为 early-fusion multimodal foundation，并声明使用 trillions of multimodal tokens、扩展到 million-agent environments 的 RL、asynchronous RL framework 和 201 languages/dialects。

这些内容可转化为面试中的系统问题：视觉 token 如何与文本 token 对齐；视频 temporal patch 如何进入统一序列；rollout、环境、verifier 和 learner 如何异步解耦；policy version 与轨迹 freshness 如何管理。但它们仍是 Qwen 官方发布声明，不是完整训练 recipe 或独立复现。vision encoder 的公开配置也不能证明视觉预训练 loss、跨模态对齐损失、图像/视频 token 预算或生产吞吐。

### 83.14.4 MTP 与混合 cache 的 serving 边界

模型卡明确写出 `MTP: trained with multi-steps`，并给出 SGLang `NEXTN` 与 vLLM `qwen3_next_mtp` 的 speculative decoding 示例。完整 MTP 账本要分开：draft tokens、target verification、accepted length、rollback 和 committed KV。打开 `speculative` 参数本身不能证明线上加速，因为 acceptance rate、batch、后端 kernel、工具调用和视觉路径都会改变结果。

Qwen3.5 的 serving 示例还揭示一个容易忽略的边界：`--language-model-only` 可以跳过 vision encoder，释放显存给 KV cache。这是服务模式选择，不是新的模型 checkpoint；同一个模型在 multimodal mode 与 language-only mode 下，显存、输入协议、缓存容量和验收指标不同。官方示例建议 8-GPU tensor parallel，也只是发布方示例配置，不能当作所有硬件的最低要求。

### 83.14.5 与 Qwen3.8 的对照

| 维度 | Qwen3.5-397B-A17B | Qwen3.8 公开材料 |
|---|---|---|
| 主体 | 397B total / 17B active | 27B dense、2.4T/95B 和 Flash-Next 等多个形态 |
| 混合核心 | Gated DeltaNet + Gated Attention + sparse MoE | 延续 GDN/GA，并在 Flash-Next 展开 QSA、Gated Residual、N-gram、Muon 等路线 |
| 多模态 | 原生 vision-language，带 vision encoder | 不同 checkpoint 的视觉支持和 hosted/open 边界分别核验 |
| 长上下文 | 原生 262,144，可用 YaRN 扩展约 1,010,000 | 同样需要区分原生窗口、外推配置、有效召回和服务端上限 |
| MTP | 模型卡声明 multi-step training，提供 NEXTN/MTP serving 示例 | Flash-Next 的 MTP 与 QSA 结合更具体，但不能迁移回 Qwen3.5 |

面试中可以用一句话收束：Qwen3.5 已把“递归 state + 周期性显式 attention + sparse MoE + 原生视觉 + Agent RL”作为公开基础方向，Qwen3.8 再围绕稀疏显式检索、残差流、外置容量和优化器协同做更激进的工程化展开。

### 83.14.6 证据与待核验清单

- Artificial Analysis 的 Intelligence Index、context、速度和价格是第三方配置字段；DataCurve 没有 Qwen3.5-397B-A17B 精确行，不能迁移其他 Qwen 的 DeepSWE 结果。
- Qwen 官方的 multimodal token、million-agent RL、异步 RL 和 benchmark 数字是发布方自报；不能写成独立复现或完整训练配方。
- 参数、层排布和 serving 示例来自官方模型卡/配置；完整 GDN/GA kernel、state layout、MTP acceptance length、视觉独立复现、目标硬件 profiling 和 Qwen3.5-Plus hosted/open 精确服务差异仍待核验。
- 研究证据详见 [`Qwen3.5-397B-A17B 官方资料摘记`](../../research/model-update-2026-09/qwen3.5-397b-a17b-source-notes.md)。本节不新增 Qwen3.5 独立章节，以免与 Qwen3.8 的混合架构专题重复。
