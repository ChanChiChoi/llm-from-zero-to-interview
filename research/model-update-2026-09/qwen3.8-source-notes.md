# Qwen3.8 官方资料摘记

架构与报告核验日期：2026-09-14；serving/runtime 补证：2026-09-24。本文把两个排行榜作为候选发现入口，把 Qwen 官方模型卡、GitHub 仓库、技术报告和 Qwen Cloud 页面作为事实核验来源。模型卡和技术报告中的 benchmark、吞吐、稳定性与成本数字均保留为发布方自报，不能替代独立复现。

## 1. 榜单发现与证据分层

本轮 Artificial Analysis 快照确认了四类 Qwen3.8 条目：

| 候选/配置 | 榜单记录日期 | 发现页面 | 当前解释 |
|---|---:|---|---|
| Qwen3.8 Max | 2026-08-03 | [Artificial Analysis](https://artificialanalysis.ai/models/qwen3-8-max) | 托管产品/推理配置入口；与 Qwen3.8-2.4T-A95B 的关系需回到官方模型卡和 Cloud 页面解释 |
| Qwen3.8 2.4T A95B | 2026-08-12 | [Artificial Analysis](https://artificialanalysis.ai/models/qwen3-8-2-4t-a95b) | 官方开源模型卡可核验的 MoE checkpoint |
| Qwen3.8 27B | 2026-08-14 | [Artificial Analysis](https://artificialanalysis.ai/models/qwen3-8-27b) | 官方开源 dense vision-language checkpoint |
| Qwen3.8-Flash-Next | 2026-08-26 | [Artificial Analysis](https://artificialanalysis.ai/models/qwen3-8-flash-next) | 官方开源的实验性架构预览 |

DataCurve DeepSWE 当前快照只检出 `qwen3.8-max`，页面入口为 [DeepSWE](https://deepswe.datacurve.ai/)。因此本次只把 DataCurve 记为 Qwen3.8 Max 的辅助发现证据，不能写成 27B、2.4T-A95B 或 Flash-Next 也由 DeepSWE 发现。

快照身份如下：

- Artificial Analysis `/zh` 页面：`/tmp/artificialanalysis-live-2026-09-14.html`，SHA-256 `510bb1916a029700fbd78084a4b26cbe1e2b2690ce9e25179fde2d4fe650a7d4`。
- DataCurve DeepSWE 页面：`/tmp/deepswe-live-2026-09-14.html`，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。

排行榜中的 `xhigh`、`medium`、`low`、`Non-reasoning` 是同一基础模型的请求配置或评测条件，不应按行数计为不同架构。Qwen3.8-Max、Qwen3.8-Flash 等名称进入正式核验后，还必须区分 open checkpoint 与 hosted service。

## 2. 官方来源

- [Qwen3.8 GitHub 入口](https://github.com/QwenLM/Qwen3.8)：Qwen3.8 系列模型卡和相关发布入口。
- [Qwen3.8-27B 模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)：27B dense native vision-language checkpoint、结构配置、思考控制和部署提示。
- [Qwen3.8-2.4T-A95B 模型卡](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)：2.4T/95B active MoE checkpoint、强制 thinking 说明和配置。
- [Qwen3.8-Flash-Next 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)：125B/6B active、N-gram 参数、QSA、GR、MTP、视觉和部署入口。
- [Qwen3.8-Flash-Next 官方仓库](https://github.com/QwenLM/Qwen3.8-Flash-Next)：实验性架构说明、代码引用、技术报告和推理框架入口。
- [Qwen3.8-Flash-Next 技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)：*On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability*，日期为 2026-08-26。
- [Qwen 官方 Flash-Next README（固定提交）](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/README.md)：称 Flash-Next 是 Qwen4 架构的早期预览，并给出 Transformers、SGLang、vLLM、TokenSpeed 的部署命令；这不是另一个排行榜模型锚点。
- [vLLM v0.30.0 Qwen4Exp 实现](https://github.com/vllm-project/vllm/tree/v0.30.0/vllm/models/qwen4_exp) 与 [模型注册表](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/registry.py)：固定 release tag 下的 Qwen4Exp、QSA、PLE、Gated Residual 与 MTP serving 代码。
- [vLLM Qwen3.8-Flash-Next recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)：vLLM 发布的硬件配置和 serve 参数；动态页面标注 vLLM `0.29.0+`，不能等同本地安装或独立硬件复现。
- [SGLang v0.5.20 Qwen4Exp 实现](https://github.com/sgl-project/sglang/tree/v0.5.20/python/sglang/srt/models)：固定 release tag 下的 QSA 调度、MTP 和 PLE host-offload 实现。
- [Qwen3.8-Max 产品页](https://www.qwencloud.com/models/qwen3.8-max)：托管版本能力边界和产品配置。
- [Qwen3.8-27B 产品页](https://www.qwencloud.com/models/qwen3.8-27b)：27B hosted version 的产品入口。
- [Qwen3.8-Flash 产品页](https://www.qwencloud.com/models/qwen3.8-flash)：Flash-Next 对应的托管版本入口。
- [Qwen3.8 发布页路由](https://qwen.ai/blog?id=qwen3.8) 与 [Flash-Next 发布页路由](https://qwen.ai/blog?id=qwen3.8-flash-next)：Qwen 官方博客入口；页面内容可能由动态站点渲染，模型事实以模型卡和报告为准。

本轮保存的本地资料包括 `hf-qwen38-27b-card-2026-09-14.out`、`hf-qwen38-a95b-card-2026-09-14.out`、`hf-qwen38-flash-card-2026-09-14.out`、`qwen-flash-readme-2026-09-14.out` 和报告文本。技术报告 PDF 的 SHA-256 为 `04f263446d74a35cb7cea368574e0c561f3b05c133be2c777ac884404063655d`，正文共 28 页。

2026-09-24 新增固定来源快照：Qwen README 9,735 bytes / SHA-256 `34d45d3486c29dcc23dade1472b5cbf1347ffe0a1adc3334aec83b3dc4e08c50`（commit `69885871a64393807d988b27b1b5e380e8f28526`）；vLLM Recipe 页面 1,859,526 bytes / SHA-256 `6895396ab217c1e14d1832fa0006a6aadc9a25820f3868daa2989409b9d8a78a`。vLLM `v0.30.0` 固定 tag commit 为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`；SGLang `v0.5.20` 固定 tag commit 为 `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`。实现文件的 Git blob ID 与两 tag 的 tree listing 一致。

## 3. 模型家族边界

| 模型 | 公开形态 | 规模与结构 | 上下文/模态 | thinking 边界 |
|---|---|---|---|---|
| Qwen3.8-27B | Apache-2.0 开源模型卡中的 checkpoint | 27B dense；64 层；hidden 5120；每 4 层为 3 个 GDN + 1 个 Gated Attention；GDN 为 48 value heads/16 QK heads，Gated Attention 为 24Q/4KV | 原生 262,144，可扩展约 1M；图像和视频输入 | 默认开启；可按请求关闭；支持 `reasoning_effort` 与 `preserve_thinking` |
| Qwen3.8-2.4T-A95B | 开源 MoE checkpoint | 2.4T total、95B activated；92 层；hidden 8192；每 4 层为 3 个 GDN + 1 个 Gated Attention；512 experts，10 routed + 1 shared | 原生 262,144，可扩展约 1,010,000；text-only | 强制 thinking，不能关闭；支持 `reasoning_effort` 与 `preserve_thinking` |
| Qwen3.8-Flash-Next | 开源实验性架构预览 | 主模型 125B、6B/token active；额外 51B N-gram embedding，模型卡另列 4B MTP；48 层；每 4 层为 3 个 GDN + 1 个 QSA；512 experts，10 routed + 1 shared | 原生 262,144，可扩展约 1M；视觉输入 | 默认 thinking；模型卡提供关闭 thinking 和关闭历史 thinking 保留的接口 |
| Qwen3.8-Max | Qwen Cloud hosted version | 官方模型卡说明其基于 Qwen3.8-2.4T-A95B，并加入托管能力 | 默认 1M、视觉输入、内置工具等产品字段 | 支持非 thinking 等托管产品能力 |

Qwen Cloud 的 Max、Flash 和 27B 页面是服务形态，不能被写成额外下载得到的独立 open checkpoint。特别是 Max 不能仅凭产品页反推出与 A95B 完全相同的每层部署、权重 revision 或服务端路由；可以确认的是官方页面明确给出的基础关系和托管能力。

## 4. GDN 与周期性显式注意力

Qwen3.8 延续了 Qwen3.5 系列的混合 token mixing 思路。Gated DeltaNet 将前缀压缩成固定大小的 key-value 状态，周期性 Gated Attention 保留对历史 token 的直接检索。

技术报告采用如下状态约定。`S_t` 是 `d_k × d_v` 状态，`q_t`、`k_t`、`v_t` 分别是 query、key、value：

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

`alpha` 控制已有状态的生命周期，`beta` 控制当前 delta 修正强度。先读出当前 key 已经对应的 value，再只写入误差，区别于无条件累加 `k_t v_t^T` 的线性注意力。

27B 和 A95B 的模型卡直接公开了“每四层三层 GDN、一层 Gated Attention”的布局。Flash-Next 技术报告进一步说明：继续预训练时，full-attention 层被 QSA 替换；因此 Flash-Next 的部署核心是 GDN + QSA，而不能简单写成普通 GDN + full attention。QSA 仍承担稀疏的显式 token 检索，补偿固定状态的信息压缩。

## 5. QSA：micro-block 稀疏注意力

Qwen Sparse Attention 的目标不是直接丢弃全部远端 token，而是先用便宜的 indexer 在压缩后的 micro-block 上估计重要性，再展开少量候选 token 给核心注意力。

### 5.1 Indexer 流程

报告描述的 indexer 使用 MQA：每个 token 产生多个 query head，但共享一个 key head。设压缩倍率为 `r`，长度为 `n` 的 key 被切成非重叠 micro-block，并在 RoPE 之前平均池化：

```math
\bar{k}_b=\mathrm{RMSNorm}\left(\mathrm{AvgPool}(k_{br:(b+1)r-1})\right)
```

先池化再加位置编码很重要。不同 token 有不同 rotary phase，若先旋转再平均，平均值会混入相位差；QSA 先形成内容摘要，再给 block 起始位置分配一个相位。

对 query `i` 和 block `b`，indexer 使用 block-causal 约束，只允许 query 看到已经完整结束的 block：

```math
I_{i,b}=\sum_h\mathrm{ReLU}\left(\langle q_{i,h},\bar{k}_{b}\rangle\right),
\quad br+r-1\leq i
```

若 token budget 为 `K`，block budget 为 `ceil(K/r)`。选出的 block 再展开为原始 token index，并合并当前还未形成完整 block 的尾部 token。Flash-Next 的模型卡和配置给出 `K=2048`、`r=4`，即最多 512 个完整 block，再保留尾部 token。

### 5.2 为什么要两阶段训练

直接把一个随机或未适配的 indexer 接到 backbone 前面，会把“候选选错”与“主干不会在稀疏上下文上工作”混在一起。报告采用两阶段：

1. **Dense distillation**：从 full attention teacher 得到 token 级 attention 分布，在 block 内做 max pooling，避免一个尖锐的 salient token 被平均冲淡，再用 KL loss 训练 indexer。
2. **Sparse training**：启用 QSA 稀疏核心注意力，联合训练 indexer 与 backbone，让主干适应被选出的上下文。

报告给出的设置是：Stage 1 约 1,000 steps，每步 8 条 256K 序列，约 2B tokens；Stage 2 约 8,000 steps，每步 96 条 256K 序列，约 200B tokens。这些是报告设置，不是所有部署都必须照搬的配方。

### 5.3 复杂度与风险

报告把压缩 indexer 的复杂度写成约 `O(n^2/r)`。当 `r` 固定时，序列变长仍然会增加 indexer 成本；不能仅凭“最终只保留 K 个 token”就宣称整个 QSA 是严格线性。固定 token budget 下，核心稀疏 attention 可以教学式地理解为 `O(nK)`，但完整系统还包括 indexer、top-k、block 展开、尾部 token、分页和 kernel overhead。

报告自报的 QSA 对照结果包括：短任务平均 benchmark 从 full attention 的 75.9 到 76.8；512K--1M 的 RULER 从 90.08 到 93.00；1M 的 8-needle MRCR 从 20.71 到 26.44；1M kernel-level attention prefill 约 7.6x、decode 约 4.9x。条件包含 chunked prefill、batch size、MTP steps 和 FlashInfer baseline，必须标成发布方自报，不能写成独立复现。

QSA 还把 top-k index 复用于 MTP speculative decoding 的多个预测步。它减少 draft 阶段重复索引的成本，但不保证接受率提升；报告 Table 4 的平均 accepted length 变化很小，复现时仍要单独记录 acceptance length 和 target calls。

## 6. Gated Residual：四分支残差流

普通 pre-norm Transformer 只有一条 residual stream。增加分支可以保留不同时间尺度的历史，但也会带来额外内存流量。Flash-Next 的 Gated Residual（GR）采用四个 branch：

1. 每个 branch 使用自己的 RMSNorm。
2. 所有 branch 共同预测逐元素的 read gate，决定每个 channel 读多少。
3. 每个 branch 使用一个 scalar write gate，决定 block 输出写回多少。
4. 不再使用 `Hres` branch-mixing operator，避免每个 block 再完整读取并混合所有 branch。

教学公式可以写成：

```math
\widehat{R}_i=\mathrm{RMSNorm}(R_i;\gamma_i)
```

```math
G=\mathrm{unvec}\left(\sigma\left(W_u\,\mathrm{SiLU}\left(\frac{1}{n_r}W_d\,\mathrm{vec}(\widehat{R})\right)\right)\right)
```

```math
x=\frac{1}{n_r}\sum_i G_i\odot\widehat{R}_i
```

```math
s=2\sigma\left(\frac{1}{n_r}W_w\mathrm{vec}(\widehat{R})\right),\qquad R'_i=R_i+s_i y
```

这里 `G` 是 branch/channel 粒度的 read gate，`s_i` 是 branch 粒度的 write scalar。GR 的表达力主要放在“读哪些历史 channel”，而不是增加一个 branch mixing 矩阵。由于 read 已经包含归一化和 gate，GR 可以替代 block 前的普通 pre-normalization。

这和 Attention Residual、mHC 的区别是面试重点：AttnRes 对更早的层输出做深度方向 attention；mHC 保留 branch 读写标量并对 `Hres` 施加双随机约束；GR 则扩大 residual stream、用逐元素 read gate，并删除 `Hres`。它们都在处理跨层信息流，却不应写成同一个模块。

报告还测试了 FP8 residual storage 和 fused read/write kernel。FP8 能减少 widened residual 的搬运字节，但它依赖 gate 控制幅度、量化格式、kernel 和质量回归测试，不能从“4 branches”自动推出线上收益。

## 7. N-gram Embedding：把容量放到主干之外

N-gram Embedding 用以当前 token 结尾的短 n-gram 作为 table index，取回 embedding 向量并增强 token representation。Flash-Next 模型卡列出 Layer 2 的 20,000,000 个 bigram/trigram vocabulary，技术报告把它作为额外约 51B 参数的容量轴。

这个机制容易被误解成 RAG 或 KV cache，但它们的生命周期完全不同：

| 机制 | key 从哪里来 | 信息是否来自外部文档 | 主要成本 |
|---|---|---|---|
| N-gram Embedding | 当前 token 的局部 n-gram | 否，表随模型训练 | table lookup、embedding 读带宽和参数存储 |
| RAG | 用户/系统检索出的文档 | 是 | 检索、网络、文档 token 和上下文 |
| KV cache | 已处理的 query/key/value | 否，来自当前请求 | 显存/主存容量与读带宽 |

N-gram table 地址是确定的，因此可以放在 accelerator 外，通过 host-memory offload 与异步 prefetch 和第一层计算重叠。它增加大量参数，却不按比例增加每 token 的主干矩阵乘；代价转移到了存储容量、随机访问、预取命中和互连带宽。

报告的消融给出一个重要的实验纪律：固定总参数预算时，约 10x base vocabulary 是局部 loss 最优点，但下游结果并不总是同步；固定 MoE 预算而增加 N-gram vocabulary 时，loss 随词表增大下降，而下游性能会饱和或波动。局部 loss 变好不等于 Agent、代码和多语言能力都变好。

## 8. Muon、AdamW 与架构协同

Flash-Next 没有把所有参数都交给同一个优化器。报告中的分工是：

- Muon：真正作为二维线性映射的 attention projection、GDN projection、MoE expert FC，以及 N-gram key/value projection。
- AdamW：input embedding、LM head、MoE router 和 GR 的低秩 projection。
- N-gram embedding table：Adam，关闭 weight decay。

Muon 使用 Nesterov momentum，`mu=0.95`，并进行 8 次 Newton-Schulz orthogonalization。报告指出 fused QKV、GDN input 和 SwiGLU FC1 必须先按语义拆成独立子矩阵，再分别正交化；直接对拼接矩阵做正交化会混合不相关的奇异方向，也会用错误的矩阵形状计算缩放。

这带来两个系统问题：一是 TP/DP 分片不天然拥有完整矩阵，需要 Canzona 做负载均衡、跨 TP 重构和异步 Micro-Group；二是拆分后会产生大量小 kernel，因此报告使用 CUDA graph 降低 launch overhead。它们是生产实现细节，不能只背“Muon 比 AdamW 好”。

架构和优化器一起改变了 scaling law：报告观察到更大的 batch size 和 learning rate 更合适，并报告 batch-size warmup 在相同 token budget 下多约 18.8% optimizer steps，因而生产运行不采用该 warmup。Muon、GR、GDN/QSA 的稳定性与吞吐结论都来自报告自己的 stress test 和配置。

## 9. Thinking protocol 与托管版本

官方模型卡把 thinking 行为作为调用协议的一部分，不能把榜单的 `low/medium/xhigh` 直接写成不同模型：

| 版本 | 默认行为 | 可用控制 |
|---|---|---|
| 27B | thinking 默认开启 | `enable_thinking=False` 可请求 direct response；`reasoning_effort` 调整深度；`preserve_thinking` 控制历史 thinking 保留 |
| 2.4T-A95B | thinking 强制开启 | 可调 `reasoning_effort` 和 `preserve_thinking`，但不能关闭 thinking；text-only |
| Flash-Next | thinking 默认开启 | 支持 `enable_thinking`、`preserve_thinking`、`reasoning_effort`；视觉输入 |
| Max/Cloud | 产品页提供托管能力 | 以 Qwen Cloud 的直接字段和实时服务协议为准，不能把 open checkpoint 的模板参数原样外推 |

模型卡推荐的 `xhigh`、`medium`、`low` 是推理预算档位。多轮 Agent 中，低 effort 虽可能降低单轮响应时间，却可能引起失败和重试，导致总耗时、token 和单位成功成本上升。评测必须同时记每轮 token、总任务耗时、工具失败、重试和最终 artifact。

`preserve_thinking` 是历史消息协议和上下文组织策略，不是 KV cache 的替代品，也不能直接被解释成模型内部永久记忆。切换 hosted API 时还要检查字段嵌套方式、streaming、tool parser 和 chat template。

## 10. Serving/runtime 证据补充（2026-09-24）

这次仍沿 Artificial Analysis 已发现的 `Qwen3.8-Flash-Next` 推进，不从 runtime 仓库新增模型。Qwen 官方固定 README 把它描述为 Qwen4 的早期架构预览，并展示 262,144 context、TP=4 的 SGLang/vLLM 启动方式；这是发布方推荐命令，不代表这些命令在本地已运行。

| 固定来源 | 直接确认的 serving 实现 | 重要边界 |
|---|---|---|
| vLLM `v0.30.0`（commit `ced6857a…`） | registry 将 `Qwen4ExpForConditionalGeneration` / `Qwen4ExpMTP` 映射到独立实现；专属目录含 QSA、N-gram/PLE、Gated Residual、MTP 及对应测试。 | 这是 release 源码证据，不是本地 wheel、完整权重或线上实例验收。仓库还保留独立 `Qwen3Next` 路径，不能把两种架构混作一个实现。 |
| vLLM Qwen recipe（页面标注 `v0.29.0+`，2026-09-17 更新） | 提供 H100/H200、GB200/GB300、MI355X 等 profile；FP8 主权重验证配置记录在 4×H200 TP4。 | recipe 所述验证是发布方条件：`max-model-len=16384`、eager、16 sequences、8192 batched tokens，使用 vLLM commit `d1b4028d7e`；不能外推成 262K、吞吐承诺或本地复现。 |
| SGLang `v0.5.20`（commit `94602c9c…`） | 有独立 Qwen4Exp、MTP、PLE 模块；满足捕获模式与 token 阈值时，QSA indexer 可在 alternate CUDA stream 上与当前 stream 的 Q/K/V 准备工作并行，之后再启动 attention；MTP 可复用 draft-extend 对齐的索引。 | 这里不是 indexer 与最终 attention kernel 重叠。PLE 路径明确拒绝 two-batch overlap、N-gram speculation，且 speculative `topk` 只支持 1；是当前这条实现路径的组合限制。 |

可考的实现细节：

- vLLM QSA 将主模型 KV、raw-index-key circular state、压缩 indexer-key cache 与 top-k buffer 分开管理。当前 NVIDIA QSA 实现把 QSA 激活/QKV 与主 KV 路径限制在 BF16，且不接受 KV quantization 或 context parallelism；压缩 indexer cache 则允许 BF16 或无 scale 的 FP8 E4M3。这不排斥使用 **FP8 权重**：权重精度与 BF16 主 KV 是不同配置维度。
- QSA 先压缩索引候选、再展开成稀疏注意力 token；MTP 第 0 步选索引，后续 draft 步重用与 target 对齐的 top-k 行，同时仍维护 QSA side cache。它省掉重复索引工作，但接受长度/最终加速仍须实测。
- vLLM PLE/N-gram 可在 device table 与 pinned-host/UVA table 间选择，host lookup 使用独立 CUDA stream；SGLang 另提供 pinned-host 与 file-backed sparse `mmap`。约 47.7 GiB 的 FP8 PLE 表容量来自 SGLang 源码注释；其 file backend 在适用的 unified-memory 设备上通过 host page tables、按需 fault、`WILLNEED` prefetch 和 RSS trimmer 管理表容量。代码注释指出统一内存上的 pinned allocation 可能与权重争用同一容量，所以不能默认“pinned 一定省显存”。
- vLLM recipe 指出 H100 80GB/GPU 对 51B PLE/N-gram 表缺少足够余量；GB300 FP8 以 TP2 为最小部署，TP4/TEP4 有包含 MTP3 的配置，8×H200 则推荐 TEP8，plain TP8 与 128-wide quant block 不兼容。这些是该 recipe 的部署条件，不是跨硬件定律。
- vLLM tag 含 QSA reference、PLE shard/offload、Gated Residual 运算测试；SGLang tag 含同系列专属实现。本轮只审阅源码和测试文件，未执行 pytest、未安装 wheel、未加载权重、未做 GPU profile。

## 11. 待核验边界

以下结论截至本轮仍不应写成已独立确认的事实：

- Qwen3.8-Max、Qwen3.8-Flash 等 hosted service 的完整权重、revision、线上 kernel 和实际限流行为。
- QSA、GR、N-gram 和 Muon 在目标硬件上的端到端 TTFT、TPOT、并发、峰值显存和跨服务后端差异；本轮确认了 vLLM/SGLang 固定 tag 源码和测试文件，但没有运行测试或做目标硬件 profiling。报告/recipe 中的性能仍非本地独立复现。
- Qwen3.8 全系列的完整训练数据、奖励模型、后训练损失、MTP 训练配方和发布版本差异。
- 模型卡和技术报告中的 benchmark、loss、稳定性、训练 FLOPs 与成本数字之外的外部复现。
- 27B、A95B、Flash-Next 之间未公开的权重共享、服务路由和每层实现差异。公开的同系列基础关系不等于 checkpoint 完全相同。

本轮不下载几十 GB 权重作为研究前置条件。后续若要做性能实验，应固定模型 revision、tokenizer、后端、硬件、batch、context、effort、工具和 harness，并将发布方数字与本地结果分栏。

## 12. 书系落点

- 正式专题：[`第二十一册第 83 章`](../../book-21-transformer-architecture-evolution/chapters/83-qwen3.8-qsa-gated-residual-n-gram-muon.md)。
- 架构基础：第二十一册第 61、62、73、74、75、77、78 章，用于对比 KDA/GDN、Gated Attention、压缩注意力、AttnRes、CSA/HCA 和 mHC。
- 训练与优化：第五册的 MoE、长上下文、Muon、稳定性和后训练章节。
- Serving 与评测：第六册、第二十四册、第七册和第二十册，用于建立 KV/预取/通信/effort/harness 账本。
