# K2 Horizon：MoVA 36B/A4B 官方资料摘记

核验日期：2026-09-14。本文把两个排行榜的候选发现、官方模型卡、固定 revision、模型实现和部署资料分开记录。正式结论只使用已核验的字段；排行榜日期、参数和指数不替代官方资料。

## 1. 候选是怎样进入本轮的

本轮的模型候选来自 [Artificial Analysis 中文入口](https://artificialanalysis.ai/zh) 的页面快照。快照中的新增结构条目包括 `K2 Horizon MoVA 36B A4B`、`K2 Horizon 7B`、`K2 Horizon 3.7B` 和 `K2 Horizon 0.9B`，对应页面 slug 可能成对出现。页面的 `releaseDate=2026-09-03` 是第三方榜单字段，只能作为发现时间线，不能当作 IFM 的官方发布日期。

在已保存的 [DataCurve DeepSWE](https://deepswe.datacurve.ai/) v1.1 快照中，本轮没有检索到 K2 名称。这个结论只适用于本地保存的快照，不声称当前线上榜单一定没有 K2。因而 K2 本轮的候选来源应写成：Artificial Analysis；DeepSWE 作为第二个候选入口完成过交叉检查，但不是 K2 的发现证据。

候选发现和资料核验的关系如下：

```text
Artificial Analysis 发现 K2 Horizon
        |
        +--> IFM/Hugging Face 模型卡：确认公开 checkpoint 与配置
        +--> 固定 revision 的 config/modeling：确认架构语义
        +--> SGLang/vLLM 资料：确认部署接口与运行时约束
        +--> K2 7B Uno adapter/论文：记录关联技术，不新增榜单模型
```

## 2. 来源和证据等级

| 来源 | 用途 | 当前状态 |
|---|---|---|
| [Artificial Analysis K2 Horizon MoVA 36B A4B](https://artificialanalysis.ai/models/k2-horizon-mova-36b-a4b) | 候选发现、页面配置名和榜单入口 | 第三方发现证据 |
| [K2-Horizon-MoVA-36B-A4B model card](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B) | 公开规模、上下文、训练概览、checkpoint 和部署建议 | 官方模型卡，已读取 |
| [`config.json`](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/blob/de2d2efb32ed7639b7140bccbefe131a0063a982/config.json) | 模型类型、层数、头数、路由和 RoPE 字段 | 固定 revision 已读取 |
| [`modeling_k2_horizon.py`](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/blob/de2d2efb32ed7639b7140bccbefe131a0063a982/modeling_k2_horizon.py) | MoVA value routing、MoE routing、GQA、attention gate 的执行语义 | 固定 revision 已读取 |
| [SGLang K2 Horizon cookbook](https://docs.sglang.io/cookbook/autoregressive/IFM/K2-Horizon) | TP/EP、FlashAttention-3 和路由 GEMM override | 官方部署资料，已读取页面快照 |
| [vLLM IFM recipes](https://recipes.vllm.ai/IFM) | vLLM TP、expert parallel 和 parser 配方 | 模型卡给出的官方部署入口 |
| [IFM xllm](https://github.com/ifm-ai/xllm) | 训练/基础设施代码入口 | 模型卡标为进行中，不能写成已公开完整训练代码 |
| [IFM K2 blog](https://ifm.ai/blog/k2/) | 发布文章入口 | 本轮被 Cloudflare challenge 拦截，只记录链接，不声称已读取正文 |
| [K2-Horizon-7B model card](https://huggingface.co/IFM/K2-Horizon-7B) | Uno 的冻结 AR base 关联 | 官方关联模型卡，和 36B 不是同一架构规模 |
| [K2-Horizon-7B-Uno model card](https://huggingface.co/IFM/K2-Horizon-7B-Uno) | Uno adapter 的交付和使用入口 | 官方关联 adapter 资料，不是榜单发现 |
| [Uno paper](https://arxiv.org/abs/2609.04010) | 离散扩散、Diffusion Distillation、Psi-Spec 的一般机制 | arXiv:2609.04010v1，2026-09-03 |

本地证据文件只保存轻量文本、配置、代码和 header 校验结果，没有下载几十 GB 的模型权重：

| 文件 | SHA-256 |
|---|---|
| `/tmp/k2-horizon-mova-readme.md` | `80fe8535382e395e40228643c4700adbb801034e696681a0beb99c01af570f61` |
| `/tmp/k2-horizon-mova-config.json` | `3ee828b698d51aa23bbf8418823b7904bd35b0347dbf8a464b2aff5fe25b76fc` |
| `/tmp/k2-modeling.py` | `fb09e010956bd51cfa7d4055b4381cff34c9e06164066b49e3546f38b2e6242f` |
| `/tmp/k2-validation.json` | `903cbcdf156b1796712ad323aee33a8eb9665a4a75235220da5b38391aec3182` |
| `/tmp/uno-source.tar` | `bd7d0277c238495e4fe4e428fa1193a258c39b205f36de2cc643843b8ebb2cfe` |

## 3. K2-Horizon-MoVA-36B-A4B 已核验字段

### 3.1 规模和骨架

- 模型 ID：`IFM/K2-Horizon-MoVA-36B-A4B`。
- 许可证字段：Apache-2.0；模型卡把 checkpoint 标为已发布。
- 总参数约 36B，每 token 约 4B active parameters。这里的 active parameters 是路由后主要参与该 token 计算的参数量近似，不是整个请求的 FLOPs、显存或延迟保证。
- 48 个 decoder layer，hidden size 2560，词表 250,624，BF16。
- 32 个 query attention heads、8 个 key/value heads、head dimension 128，因此是 GQA，query 到 KV 的复用倍率为 `32 / 8 = 4`。
- `max_position_embeddings=524288`，即 524,288 token 的接口/配置上限。配置同时明确 `sliding_window=null`、`use_sliding_window=false`，不能把 K2 写成滑动窗口模型。
- 固定 revision 为 `de2d2efb32ed7639b7140bccbefe131a0063a982`。这条 revision 只绑定本次读取的实现和配置，不替代后续服务端 alias。

### 3.2 前三层 dense，后四十五层稀疏

配置的 `mlp_only_layers=[0,1,2]` 和实现中的 layer selection 表明前 3 层使用普通 dense attention + dense MLP；从第 4 层开始使用 sparse layer。由于 `decoder_sparse_step=1`，后 45 层逐层启用 MoVA attention 和 MoE FFN。

这个排布有一个容易错读的地方：36B/A4B 不是“每一层都同时把所有专家算一遍”，也不是“每层都使用滑动窗口”。后 45 层只对当前 token 选择的 value experts 和 routed FFN experts 做主要专家计算，但权重、路由元数据、跨卡 token dispatch 和共享专家仍然需要纳入系统账本。

### 3.3 MoVA：只路由 value，不路由整个 attention

普通 GQA 仍然先由 `q_proj` 生成 32 组 query、由 `k_proj` 生成 8 组 key。MoVA 改变的是 value 路径：配置有 64 个 value experts，每 token 选择 top-4；每个 value expert 把 hidden size 2560 映射到 8 个 KV head 乘 128 的 value 表示，再经过 SiLU，最后按路由权重相加。

对 token 表示 `x`，可以用下面的教学式理解 value mixture：

`v(x) = sum_{e in Top4(x)} alpha_e(x) * SiLU(W_e x)`

其中 `W_e` 是第 `e` 个 value expert，`alpha_e` 是选中路由的归一化权重。之后 attention 仍然计算 `softmax(q k^T / sqrt(d)) v`。所以 MoVA 的“专家”位于 value projection 一侧，不等于 64 个完整 attention heads，也不等于把 Q/K 也复制 64 份。

实现中 value router 使用 sigmoid 分数；router bias 只加到 top-k selection score，不加到最终参与 value mixture 的原始概率。选中的 top-4 权重归一化后乘 `router_scaling_factor=2.5`。这个细节会影响 checkpoint 兼容性，不能用一个泛化的 softmax router 替换。

### 3.4 FFN MoE 和 active parameters

后 45 层的 FFN 有 100 个 routed experts，每 token 选择 top-8，并额外计算 1 个 shared expert。routed expert 的中间维度为 768；shared expert 的实现使用相同基础中间维度乘 shared expert 数。FFN 使用 SwiGLU 形态：

`MLP(x) = W_down( SiLU(W_gate x) * W_up x )`

路由后，token 的输出是 8 个 routed expert 输出按权重聚合，再加上 shared expert 对残差输入的输出。`active=4B` 的直觉是“每 token 主要只走少数参数”，但实际运行仍需考虑：

1. 100 个专家的权重总量和分片布局决定 checkpoint 存储与加载压力。
2. top-8 dispatch 产生 token-to-expert 的 all-to-all 或等价通信。
3. 64 个 MoVA value experts 另有一次 value route 和聚合成本。
4. shared expert 不随 routed top-k 消失。
5. router、padding、capacity、通信 buffer、workspace 和 kernel launch 不包含在“4B”这个标签里。

## 4. Attention gate、RoPE 和缓存语义

### 4.1 GQA 的缓存收益

在相同 batch、序列长度和 head dimension 下，K/V cache 的 head 数从 32 降到 8，理论上的 K/V head 存储因子约为四分之一；这是 GQA 带来的结构性收益，而不是 MoVA 带来的。MoVA value expert 输出的最终 value 仍需要写入每层 cache，不能因为 value 由专家生成就把 cache 当作“64 份专家 cache”。

### 4.2 RoPE

配置使用 default RoPE，`rope_theta=10000000`，`rope_head_dim=128`。实现先对 Q/K 应用旋转位置编码，之后再进入 attention。没有证据支持把 K2 的位置机制写成 YaRN、滑动窗口或其他外推方法；模型卡对 512K 的说明来自分阶段 midtraining 和配置，而非 `sliding_window`。

### 4.3 Attention gate

`attention_gate_func=softplus`。实现由 hidden state 经过一个 gate projection，按 head dimension reshape，再用 `softplus(gate, beta=ln(2))` 乘在 attention output 上，最后交给 `o_proj`。它是 attention branch 内的可学习幅度控制，不是 MoE router，也不是 token 选择器。

因此三个“门”必须区分：

| 机制 | 作用对象 | 选择/调节什么 |
|---|---|---|
| MoE router | FFN routed experts | 每 token 的 top-8 专家及权重 |
| MoVA value router | value experts | 每 token 的 top-4 value 变换 |
| attention gate | attention output | 对已聚合的 attention 分支做幅度调节 |

## 5. 长上下文训练不是把 max length 改大

模型卡给出的增量 token 预算和序列长度如下：

| 阶段 | 额外 token | 序列长度 | 公开目的/说明 |
|---|---:|---:|---|
| Pretraining | 22.9T | 8K | 预训练 |
| Midtraining Stage 1 | 1.1T | 32K | 上下文扩展 |
| Midtraining Stage 2 | 498B | 128K | 上下文扩展 |
| Midtraining Stage 3 | 110B | 512K | 上下文扩展 |
| Midtraining Stage 4 | 199B | 512K | 延续扩展，并提高 agentic/reasoning SFT 数据比例 |
| SFT Phase 1 | 219B | 512K | 领域覆盖 |
| SFT Phase 2 | 50B | 512K | Phase 1 的高质量子集，带 learning-rate decay |

这些 token 是阶段增量，不是把每行相加后误称为“单一数据集规模”；每阶段从前一阶段最终 checkpoint 继续。对工程来说，长上下文阶段至少同时改变 attention 计算、position 分布、数据 packing、激活/通信峰值和训练稳定性。一个只修改 `max_position_embeddings` 的 toy 实验不能证明 512K 能力。

模型卡的 artifact 表还显示：最终 checkpoint 已有，训练日志有入口；技术报告和 xllm 代码在模型卡更新时仍标为进行中。因此不能写成“完整 recipe、完整训练代码和技术报告已公开”。

## 6. MoE/MoVA 的通信和显存账本

设 batch 为 `B`，序列长度为 `T`，hidden size 为 `H=2560`，后 45 层每 token 选 `k_moe=8` 个 FFN expert、`k_mova=4` 个 value expert，设备数为 `P`。一个不依赖具体 kernel 的容量草图是：

`tokens = B*T`

`routed_assignments = tokens * (8 + 4)`

`dispatch_bytes ~= routed_assignments * H * bytes_per_element`

这不是精确通信量，因为真实系统会有 expert parallel 分片、padding/capacity、局部命中、all-to-all 算法、融合 kernel 和 overlap。它的价值在于提醒面试者：active parameters 不能替代 token dispatch 账本。

显存也要分项：

`HBM = weights + KV_cache + activations + router_metadata + dispatch_buffers + workspace + fragmentation`

其中：

- `weights` 受 36B 总参数、BF16、tensor/expert parallel 分片和加载临时空间影响。
- `KV_cache` 受 48 层、8 个 KV heads、head dimension、序列长度、batch、缓存 dtype 和 prefix reuse 影响。
- `activations` 在 prefill 尤其受 `B*T`、checkpointing 和 layer pipeline 影响。
- `router_metadata` 至少包括 selected expert indices、weights、offsets 和负载统计。
- `dispatch_buffers` 受 top-k、EP 拓扑、跨节点带宽和 overlap 策略影响。

一个可审计的服务报告应分别给出权重、KV、专家 dispatch、workspace 和碎片，而不是把“4B active”直接换算成单卡显存。

## 7. MOPD 和专家合并的边界

同系列 [K2-Horizon-0.9B model card](https://huggingface.co/IFM/K2-Horizon-0.9B) 公开描述了另一条训练流程：RL 阶段从 midtraining checkpoint 分出 `math1`、`code1`、`math2a`、`math2b`、`code2`、`IF`、`stem` 七个专家模型，随后合并；其后有一个名为 `MOPD` 的阶段，模型卡把目的描述为缓解权重合并导致的结构干扰和性能下降，并通过 on-policy distillation 在行为空间对齐多领域能力。

这里有两个边界：

1. 这能解释“领域专家合并后为什么还需要行为层面的恢复”，但不能据此声称 MOPD 是一个已经公开完整定义的标准算法；模型卡没有给出完整 loss、采样配方、教师数量和所有超参数。
2. 0.9B 卡片的 RL/MOPD 流程不能自动迁移成 36B MoVA 的训练配置。36B 模型卡的训练表没有列出同样的七分支专家合并和 MOPD 阶段，正式章节只能分别标注“0.9B 同系列卡片披露”和“36B 已核验训练表”。

运行时 MoE routing 也和离线 expert merge 不同：前者每个 token 动态选择专家并保留专家边界；后者把多个训练分支的权重合成一个 checkpoint，可能出现参数方向干扰。两者都出现“expert”一词，但优化对象、发生时机、部署形态和验证方法完全不同。

## 8. Serving 和协议

### vLLM

模型卡给出的主线配置是 TP=2、expert parallel、BF16、`trust_remote_code`，并启用 `k2_horizon` reasoning/tool parser 和 auto tool choice。`--revision` 应固定到要评测的 checkpoint 分支，不能因为 `main` 指向后续更新就假设行为不变。

### SGLang

模型卡引用的 SGLang 配方在 2 张 H200 上验证，使用 TP=2、EP=2、FlashAttention-3，并传入 `xllm_source_router_gemm_partitions=2`。这个 override 是路由 GEMM 分片的运行时兼容条件，不是模型架构中的新专家数量；换 GPU、后端或 kernel 后仍要重新测显存、通信、TTFT、TPOT 和输出一致性。

### Transformers 和 chat template

模型卡写明验证环境为 Transformers 5.15.0、PyTorch 2.13.0、Safetensors 0.8.0；配置 metadata 的 `transformers_version` 为 5.13.0，二者是不同证据字段，不应擅自改写成一个版本。聊天请求推荐 `reasoning_effort="high"`、`temperature=1.0`、`top_p=0.95`；thinking 返回 `reasoning_content`，答案返回 `content`。工具格式支持 `json`、`xml`、`xml_typed`，默认是 `xml`。

部署迁移最少要回归：固定 revision、tokenizer、chat template、reasoning 参数、工具 parser、流式输出、tool call 参数、并行拓扑和错误恢复。模型卡的 parser 名称只说明接入接口，不授予网络、文件或终端权限。

## 9. Uno：K2 7B 的关联加速技术，不是 K2 主模型架构

`K2-Horizon-7B-Uno` 是从 K2 7B 的官方关联资料追到的 adapter，不是本轮 Artificial Analysis 或 DeepSWE 中单独发现的一行模型，也不应被写成 K2-Horizon-MoVA-36B-A4B 的变体。它和主模型的关系是：

- AR pathway 使用冻结的 K2-Horizon-7B 权重。
- diffusion pathway 使用轻量 LoRA adapter，并行地产生 draft。
- Uno 论文把方法称为 diffusion-augmented LLM；通过 Diffusion Distillation 训练 diffusion weights。
- `Psi-Spec` 用并行 draft 加 AR rejection verification；论文主张在其定义的采样协议下保持底层 AR 分布，而不是无条件保证所有部署场景都“零损失”。
- 交付的 Uno 仓库/模型卡是 adapter 入口，base model 需要单独获得；因此加载、量化、缓存和 license 记录必须同时绑定 base 与 adapter。

这条技术线值得作为 K2 周边知识学习：它把“提高 decode 并行度”的问题从更换完整 draft model，转成冻结 AR 主干、增加轻量 diffusion 参数和验证协议。但它的吞吐、接受率和质量结果必须固定 base revision、adapter revision、采样器、batch、硬件和 harness；不能把论文/模型卡自报数字直接当作本项目实测。

## 10. 当前不能确认的内容

- K2-Horizon-MoVA-36B-A4B 的完整技术报告、完整训练代码和完整数据 recipe 在本轮仍未公开可核验。
- 模型卡中的 benchmark、H200 性能和训练阶段说明属于发布方资料；本项目没有独立复现 36B 权重或线上吞吐。
- `active parameters` 不能推出 FLOPs、显存、通信或端到端速度。
- `MOPD` 的完整算法定义和它是否用于 36B MoVA 没有足够证据。
- Uno 的具体 base/adapter revision、接受率和跨后端性能需要单独固定并实测。
- IFM blog 正文被 Cloudflare challenge 拦截；本笔记没有把未读取的博客正文当成证据。

## 11. 复述模板

面试中可以这样概括：

> K2-Horizon-MoVA-36B-A4B 是一个 36B 总参数、约 4B/token active 的 BF16 稀疏模型。它在 48 层中前 3 层为 dense，后 45 层把 GQA attention 的 value 路径替换为 64-expert、top-4 的 MoVA，同时使用 100 routed FFN experts、top-8 和 1 个 shared expert。它的 512K 是分阶段 midtraining/SFT 后形成的配置与能力目标，不是 sliding-window attention。部署时除了 TP/EP，还要把两套路由的 dispatch、GQA KV cache、router metadata 和 kernel 兼容性列入账本。MOPD 只按 0.9B 同系列卡片的披露解释，不能反推 36B recipe；Uno 则是 K2 7B 的 diffusion/LoRA adapter 关联技术，不是排行榜新模型。

