# 第十六章：FP8 KV Cache：用低精度换长上下文并发

在大模型推理中，权重通常在服务启动时加载一次，而 KV Cache 会随着每个活跃请求
的上下文和输出不断增长。短请求时，权重显存可能是主要开销；到了长上下文、高
并发、RAG、多轮对话或 Agent 工具历史场景，KV Cache 往往变成更快增长的动态账本。

FP8 KV Cache 的目标是把 K/V 历史从更高精度 payload 换成 8 bit 浮点表示，从而
减少每 token 的缓存字节数，为更多请求、更多上下文或更少的 GPU 留出空间。但它
不是简单地把 `dtype` 改成 `float8`：K 和 V 的误差路径不同，scale 可能按层、head、
channel 或 block 保存，page 迁移和 prefix sharing 要携带这些 scale，runtime 还
可能因为硬件或 shape 不支持而回退到高精度或慢路径。

本章从 KV 的计算动机开始，建立显存公式，再展开 FP8 的数值和 scale、K/V 不对称、
静态/动态校准、分页 cache 生命周期、kernel 性能、长上下文评估、租户路由、回退和
发布。读者最终要能回答的不是“FP8 能省多少显存”，而是“在哪类请求上，付出什么
质量和工程代价后，显存收益是否值得”。

## 1. 为什么先量化 KV，而不是只量化权重

### 1.1 KV 是随请求增长的动态空间

自回归生成时，当前 token 的 query 需要和历史 token 的 key/value 交互。如果每个
decode step 都重新计算历史 K/V，会重复做大量工作；因此 runtime 会把已经计算的
K、V 保存起来，下一步只计算新 token 的状态并读取历史 cache。

权重显存近似固定，KV 则与以下因素相乘：

1. 层数；
2. 活跃请求数；
3. 每个请求的上下文和已生成 token 数；
4. KV head 数和 head dimension；
5. K 与 V 两份状态；
6. 每个元素的存储字节数、scale、page metadata 和对齐。

因此，服务即使能成功加载模型，也可能在长请求逐渐进入后耗尽显存。FP8 KV 的
第一目标常常不是让单请求 kernel 变快，而是让系统在同一 GPU 预算下保持更多
active sequence，减少抢占/重算，或者承载更长的有效上下文。

### 1.2 KV 量化和权重量化不是同一个实验

权重量化通常在服务启动前完成，量化误差对每个请求重复出现；KV 量化发生在请求
执行期间，scale 和 page 有生命周期，误差随着历史状态被后续 attention 读取。
两者的校准数据、回退方式和故障模式不同。

一个“FP8 模型”可能只表示权重使用 FP8，也可能表示权重、激活和 KV 都使用 FP8。
一个“FP8 KV”也不一定表示 K、V、scale、累加和输出 logits 都是 FP8。发布 manifest
必须把对象拆开记录，否则不同团队对同一个 precision 名称的理解会不一致。

### 1.3 低精度的收益有三种

FP8 KV 可能带来：

- **容量收益**：每个 KV token 占用更少显存，接纳更多并发或更长上下文；
- **带宽收益**：读取历史 K/V 的 payload 更小，某些硬件路径可能降低 decode 带宽
  压力；
- **调度收益**：较少的 OOM、抢占和重算，让 continuous batching 更稳定。

这三种收益不一定同时发生。如果 runtime 需要在每一步解包和转换，payload 变小
可能只改善容量而不改善 TPOT；如果质量回退增加重试，系统总成本可能反而上升。

## 2. KV Cache 显存公式和数量级

### 2.1 基本公式

设模型有 `L` 层，`B` 个活跃序列，每个序列平均保存 `T` 个 token，有 `H_kv` 个
KV head，每个 head 维度为 `D_h`，每元素占用 `b` 字节，则标准 MHA/GQA/MQA 路径
可以先用下面的下界估算：

~~~math
M_{\mathrm{KV}}
\approx
2LBTH_{\mathrm{kv}}D_hb
~~~

前面的 `2` 对应 K 和 V。MHA 的 `H_kv` 通常等于 query head 数，GQA 的 KV head
更少，MQA 可以进一步接近一个 KV head。`b=2` 对应 BF16/FP16 的理想 payload，
`b=1` 对应 8 bit payload 的理想情况。

这个式子没有计入 scale、page table、alignment、allocator 碎片、通信 buffer、
临时 workspace 和 runtime 预留，因此它只能帮助判断数量级。

### 2.2 一个 1M 上下文的计算

假设 `L=64`、`B=1`、`T=131072`、`H_kv=8`、`D_h=128`，使用 BF16：

~~~math
M_{\mathrm{BF16}}
\approx
2\times64\times1\times131072\times8\times128\times2
=32\ \mathrm{GiB}
~~~

把长度增加到 `T=1,048,576`，其他条件不变，BF16 KV 约为 256 GiB。FP8 payload 的
理想值约为 128 GiB，实际还要加 scale 和 metadata。

这些数字不是某个模型的容量承诺。真实模型可能使用 MLA、滑动窗口、混合 attention、
递归 state 或不同的 cache layout；即便公式成立，服务也必须用 runtime 实际分配的
bytes 和真实长度分布校准。

### 2.3 加上 scale 和 page metadata

若每个 block 有 `n` 个 K/V 元素，payload 每元素占 `b_q` 字节，每个 block 需要
`b_s` 字节 scale 和 `b_m` 字节 metadata，则平均 bytes/element 近似为：

~~~math
b_{\mathrm{avg}}
=
b_q+\frac{b_s+b_m}{n}
~~~

`n` 越小，scale 开销占比越大；`n` 越大，局部动态范围越难覆盖。对于 K 和 V
使用不同 scale 或不同 block 组织的系统，应分别建立账本，不能只把总 payload 除以
二就宣布“节省一半”。

## 3. FP8 表示、scale 和量化公式

### 3.1 FP8 不是一个唯一精度

OCP 的 FP8 二进制格式包括 E4M3 和 E5M2。两者都是 1 个符号位，但指数位和尾数位
分配不同：

| 格式 | 指数字段/尾数字段 | 最大有限绝对值 | 最小 normal | 最小 subnormal | 特殊值与取舍 |
| --- | --- | ---: | ---: | ---: | --- |
| E4M3 | 4 / 3 bit | 448 | `2^-6` | `2^-9` | 没有 Inf 编码，只保留 NaN 编码；精度较细、范围较窄 |
| E5M2 | 5 / 2 bit | 57,344 | `2^-14` | `2^-16` | 有 Inf 和 NaN；范围更大、相邻数间隔更粗 |

这些编码数值来自[OCP OFP8 v1.0](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)；
[FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) 解释了 E4M3/E5M2
为何按范围与精度分工。所以“FP8”本身不能回答溢出后饱和还是变成 NaN、scale 怎么算、数据写 cache 时
用什么 rounding，也不能证明 attention kernel 原生支持这种格式。OCP v1.0 规定
了编码和宽格式到 FP8 的舍入/饱和转换，却明确不规定 scale 的选择、表示和维护。
还要区分 OCP 编码与 FNUZ 变体：例如 ROCm 类型表分别列出 OCP 与 FNUZ，后者的
特殊值编码规则不同；PyTorch 也分别提供 `fn` 和 `fnuz` dtype。部署时要记录具体
variant、转换模式和 runtime/backend，而不是只写 `float8`。[ROCm 类型表](https://rocm.docs.amd.com/en/docs-7.2.3/reference/precision-support.html)
与 [PyTorch dtype 说明](https://docs.pytorch.org/docs/main/tensor_attributes.html)
可用于确认相应平台的类型名称和支持边界。

对一个 K/V block `G`，可以用抽象量化模型表示：

~~~math
q_i=Q\left(\frac{x_i}{s_G}\right),
\qquad
\hat{x}_i=s_GD(q_i),
\quad i\in G
~~~

`x_i` 是原始 K 或 V，`s_G` 是该组 scale，`q_i` 是 FP8 payload，`D` 是对应
FP8 码表的反量化函数。这里把 `s_G` 定义为反量化时相乘的 scale；有些 API 保存
其倒数，字段名不能代替公式。真实系统还要说明 scale 是否是静态、动态、per-tensor、
per-head、per-channel 或 per-block，以及是否在转换时饱和。

对于反量化 scale `s`，E4M3 的最大有限重构幅值约为 `448s`，最小 subnormal 为
`2^-9 s`；低于它的输入可能变成零，超过范围的输入则取决于饱和模式。以 E4M3
的最大有限值 448 为例，若某组最大绝对值为 500，采用 absmax scale 时
`s=500/448≈1.1161`，该最大值归一化后恰为 448。组内一个 `x=1` 的值会先变成
`1/s≈0.896`；E4M3 在这个区间最近的码值是 0.875，反量化得到约 `0.875×1.1161
≈0.9766`。这个算例展示 scale 与尾数格点如何共同决定重构误差；真实框架可能用
校准数据、不同 group 或预计算 scale，并不一定按每组实时 absmax 处理。

### 3.2 scale 粒度的取舍

**Per-tensor scale** 元数据少、kernel 简单，但一个异常值可能让整个 tensor 的
普通值只占很小的有效范围。

**Per-layer/head scale** 能适应层和 head 之间的分布差异，元数据仍相对可控；但
page 和 kernel 要知道每个 layer/head 的索引。vLLM 文档当前还区分 per-tensor 与
per-attention-head；在文档快照中，head scale 需要 `llm-compressor` 校准路径且只对
FlashAttention backend 开放。这里的 per-head scale 不是 KIVI 的 K-per-channel，也
不能推断其他 runtime 具有相同配置。

**Per-block scale** 让局部范围更紧，通常有更好的重构精度；代价是 scale 读取、
对齐和 page metadata 增加，block 太小还会影响带宽。

没有脱离模型、硬件和 workload 的绝对最佳粒度。应把 scale 开销、量化误差和
实际 attention kernel 时间同时测量。

### 3.3 静态、动态和校准 scale

静态 scale 在运行前从代表性校准集估计，cache 写入和读取比较简单；动态 scale
可以适应当前请求，却需要在写入或 block 完成时统计 range，并保证迁移、恢复和
共享前缀时使用同一解释。

scale 太小会导致异常值饱和或裁剪，scale 太大会让常见值的分辨率不足。可以用
分位数、最大值、滑动统计或 outlier-aware 规则估计，但每种方法都改变了误差、
metadata 和实现成本。校准集若只有短英文文本，不能证明长代码、数字、工具结果
和多模态 token 的 scale 足够稳定。

| scale 路径 | 何时确定 | 优点 | 主要代价或失效方式 |
| --- | --- | --- | --- |
| 固定默认值 | 运行时始终使用预设值，例如 vLLM 文档中的默认 `1.0` | 不需收集统计量，启动和写入逻辑简单 | 若量级不匹配会频繁饱和或浪费有效范围；只能作为明确记录的 baseline |
| 离线校准后冻结 | 用代表性样本估计，再固定在模型/runtime 配置中 | 不在每个 token 上求 amax，读写稳定；可以按 attention head 区分 | 请求长度、领域、RoPE、模态或模型版本变化会造成校准漂移 |
| 当前数据动态 amax | 在每个 tensor/组写入时统计当前最大值并求 scale | 可适应当前输入范围 | reduction 与额外内存流量有成本；如果旧值和新值共享被更新的 scale，必须重编码旧 payload |

NVIDIA Transformer Engine 2.20.2 的 FP8 Current Scaling 文档给出一般 tensor 量化
机制：计算 amax、把数据缩放到 FP8 范围、按 round-to-nearest-even 转换；该 recipe
需两次读 tensor（一次求 amax、一次转换）。这是理解动态 scale 代价的例子，不等于
Transformer Engine 提供同一套 KV-cache runtime。

还要区分“运行前校准”和“生成时更新”：如果一组已写入的 K/V 共用同一个 scale，
新增 token 导致 scale 改变时，旧 payload 也必须重编码，否则旧字节会按新 scale
反量化。另一种做法是固定 scale、把未完成的小组暂存在高精度，或按 token/更细的
单元各存 scale；它们分别付出饱和风险、临时内存或 metadata/kernel 成本。vLLM
当前文档公开了数据集校准和 per-head 路径；具体版本的计算、保存与加载细节仍须
按部署 release 核实。

## 4. K 和 V 为什么不能默认使用同一种策略

### 4.1 K 影响 attention 分数和权重分配

Key 进入 query-key 相似度：

~~~math
S_{t,j}=\frac{q_t k_j^{\mathsf{T}}}{\sqrt{D_h}}
~~~

量化 K 会扰动 query-key score；即使两个位置的排序没有互换，score 差值变化也会
改变 softmax 权重。少数高注意力 token、位置边界、数字和检索证据可能对这种变化
很敏感。

因此 K 的误差可能让模型把概率质量从一个证据位置挪到另一个位置，最终出现“看似
流畅，但引用选错位置”的失败。长上下文评估要记录 exact retrieval、evidence recall
和 citation support，不能只看生成文本的表面相似度。

### 4.2 V 影响聚合内容

Value 进入加权和：

~~~math
z_t=\sum_j a_{t,j}v_j
~~~

量化 V 会直接改变聚合后的表示。它未必改变 attention 的位置选择，却可能损失
数字、代码符号、工具参数或引用片段中的细节。K/V 的动态范围和敏感层可能不同，
因此相同 scale、相同 block 或相同质量阈值不是自然成立的选择。

把机制缩成一个两位置的标量例子：假设 query-key logits 是 `[1, 0]`，softmax
权重约为 `[0.731, 0.269]`。若 K 的量化误差把第一个 logit 从 1 改成 0.7，权重
变成约 `[0.668, 0.332]`；排序没变，但第一处证据的权重少了约 0.063。若
`V=[1, 0]`，聚合输出也从 0.731 降到 0.668。反过来，若 K 不变而第一个 V 值
有 `+0.1` 误差，输出增加约 `0.731×0.1=0.0731`。这是单头两位置的机制推演，
不是模型准确率预测；它说明 K 误差先改变选择权重，V 误差则按现有权重进入内容。

### 4.3 代表方案各自解决什么问题

KV 量化研究的共同经验不是“K 总比 V 敏感”，而是 K/V 分布、使用方式和在线写入
约束可能不同，需要实测后再决定分组轴与 scale。以下代表工作针对不同 bit 数和目标：

| 方案 | 要解决的问题与机制 | 作者报告的证据 | 代价、边界与对 FP8 的启示 |
| --- | --- | --- | --- |
| [KIVI，2-bit](https://arxiv.org/abs/2402.02750) | 作者在测试模型中发现 K 有少数固定 channel outlier、V 无同样模式；于是 K 沿 channel 量化，V 按 token 量化。K 的一组值跨多个 token，不能直接按生成顺序追加，因此未满组的数据暂留高精度，组满后再量化 | 在指定 Llama、Falcon、Mistral 与任务上，作者报告包含权重的峰值内存最多降低 2.6×、batch 最多扩大 4×、吞吐提高 2.35–3.47× | 这是极低 bit 的非对称方法，要求分组缓冲和专用读写路径；结果不能推成 FP8 必须采用相同轴，但支持先分别诊断 K/V 的理由 |
| [KVQuant，3-bit](https://arxiv.org/abs/2401.18079) | 处理超长上下文中的 key 通道 outlier 和 RoPE 后通道位置变化；结合 pre-RoPE per-channel K、按层敏感度分配的非均匀码本，以及逐向量 dense/sparse outlier | 作者在指定 LLaMA、Llama 2/3 和 Mistral 模型上报告，3-bit 时 Wikitext-2/C4 perplexity 增量低于 0.1，并给出百万级 context 容量实验 | 需要校准、特殊表示和自定义 CUDA kernel；论文的低 bit、语料和上下文容量不能与通用 FP8 serving 直接比较 |
| [vLLM FP8 KV 配置](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) | 当前文档公开 per-tensor 与 per-attention-head scale；head 粒度需校准路径且只支持指定 backend。FlashAttention 3 的 FP8 路径还把 Q 纳入 FP8 attention 计算 | 这是当前公开接口和实现约束，不是独立准确率结论 | per-head 并非 per-channel；不同后端的 Q/K/V 计算路径不同。评测时需固定 backend，否则不能把差异只归因于 cache payload |

这些方案形成的决策顺序是：先识别 K/V 的分布和误差传播，再选精度与量化轴；最后确认在线 append、scale 保存和 attention kernel 能否高效实现。不能只看论文的压缩倍数，也不能从 KIVI 的轴直接推导 FP8 的默认配置。

### 4.4 四组最小消融

至少比较：

1. BF16 K / BF16 V；
2. FP8 K / BF16 V；
3. BF16 K / FP8 V；
4. FP8 K / FP8 V。

每组在短聊天、长文档检索、代码、数字表格、工具历史和结构化输出上比较。若只有
FP8 K 造成 evidence recall 下降，可以保留 K 的高精度，而不是根据总平均分将 K/V
一起量化；若 V 对数字敏感，则回退 V 或只对敏感层回退。

## 5. KV Cache 是一个有生命周期的状态

### 5.1 从写入到回收

一个量化 KV page 可能经历：

~~~text
allocate
  -> append tokens
  -> attach scale metadata
  -> prefix hit / share
  -> decode read
  -> preempt / swap / migrate
  -> restore or recompute
  -> cancel / evict / free
~~~

每个阶段都必须保留 dtype、scale layout、logical position、model revision 和 owner
信息。只保存低精度 payload 而丢掉 scale，接收端可能得到数值上合法、语义上错误
的 cache；只复用 page 而忽略 position scheme，则可能把错误的历史位置送入 attention。

scale 是 payload 的解释条件，不是可随意覆盖的统计信息。假设某组 payload 中有
一个码值 `q`，其反量化值 `D(q)=10`；scale 为 0.1 时原值重构为 1。若新 token
到来后把整组 scale 改为 0.2，却不重新编码旧 payload，同一个 `q` 就会重构为 2。
因此一个量化 page 要么保持既有 scale，要么在改 scale 时重编码整组，要么把新 token
写入另一个独立 scale group。调度器、prefix share 和迁移协议都要遵守这条不变量。

### 5.2 Prefix cache 的版本和权限

prefix cache 命中时，复用的不只是 token id，还包括已经计算出的状态。cache key
至少应考虑：

~~~text
model_revision + template_hash + position_scheme
 + dtype + scale_revision + adapter + tenant_namespace
~~~

如果 BF16 请求命中 FP8 page，或旧模板命中新模板的 cache，服务可能不报错，却在
attention 中使用了不同的数值和位置语义。若 prefix 中包含租户私有文档，跨租户
共享更会造成数据泄露。因此 cache hit 不能绕过权限和版本检查。

### 5.3 抢占、swap 和重算

显存不足时，runtime 可能把低优先级请求的 KV swap 到 CPU/远端内存，或者释放后
重算 prefill。FP8 减少 GPU payload，却不会消除 swap 的网络/PCIe 带宽；量化 page
如果迁移前后 scale 和 layout 不一致，恢复结果会错误。

需要分别测：

- 正常 decode 的 TPOT；
- swap out/in 的传输时间；
- 重算的 prefill 时间；
- 恢复后的 logits 或任务一致性；
- 抢占请求与短请求的公平性。

只有在 OOM 减少和整体任务成本改善时，才算真正从量化中受益。

### 5.4 Speculative decoding 的临时 KV

draft model 可能产生临时 token 和对应 K/V。验证通过时，部分状态并入主路径；
验证拒绝时，临时 page、scale 和引用计数必须一起释放。否则服务可能出现隐性
显存泄漏，或者把 rejected token 的状态错误地保留在 target cache 中。

因此 speculative 路径的 trace 需要记录 proposed/accepted token、临时 KV bytes、
释放情况和回退原因，不能只记录最终 output tokens/s。

## 6. FP8 KV 与分页 allocator 的协作

### 6.1 分页解决位置，FP8 解决字节数

PagedAttention 或类似分页 allocator 把逻辑序列映射到物理 page，主要解决动态
KV 的碎片和按需分配；FP8 KV 改变每个 page 中 payload 的字节数，主要减少状态
存储和读取量。二者可以协作，但解决的是不同问题。

如果 page 很小，scale 和 page metadata 占比提高；如果 page 很大，异常值和不同
请求的动态范围更难局部适配。page table、scale layout 和 kernel 需要共同设计，
不能先独立选择一个 page size，再假定量化可无代价叠加。

### 6.2 page metadata 的最低内容

一个可迁移的量化 page 至少应携带：

~~~text
request_or_prefix_owner
logical_token_range
layer_head_layout
dtype
fp8_variant
scale_axis_and_shape
scale_revision
rounding_and_saturation_mode
model_revision
position_scheme
attention_compute_dtype
accumulation_dtype
reference_count
~~~

`reference_count` 对 prefix sharing 很重要；共享 page 被多个请求引用时，单个请求
取消不能直接释放它。租户 namespace 则决定公共前缀和私有前缀是否可以共享。

### 6.3 线上症状如何定位

如果显存下降但 TPOT 没改善，检查 dequant、scale load、attention kernel 和通信；
如果单请求通过而高并发失败，检查 page 分配、batch reorder、抢占恢复和引用计数；
如果只有 cache hit 请求出错，检查 prefix key、dtype、scale revision 和 position；
如果长上下文质量下降，做只量化 K/V 的消融并按层/head定位，同时确认 attention
backend 的累加精度、softmax rescaling 和长 contraction 维度；precision bug 不能
只用“FP8 误差”归因。

这样排查比直接说“FP8 精度不够”更有效，因为很多线上错误来自状态生命周期或
版本错配，而不是 payload 本身。

## 7. 容量收益和产品策略

### 7.1 一个数量级例子

设 `L=64`、`H_kv=8`、`D_h=128`、`B=1`、`T=131072`。BF16 的理想 KV 约为
32 GiB，FP8 payload 约为 16 GiB；如果 scale、metadata 和对齐额外占 8%，候选
约为 17.28 GiB。它可以使一个 GPU 多容纳请求，或在相同并发下保留更长上下文。

这只是容量层面的结果。若 FP8 让长上下文 citation support 从 0.70 降到 0.45，
审计问答不能启用它；若普通聊天从 0.98 降到 0.975，且满足质量底线，路由器可能
允许普通聊天使用 FP8、审计任务使用 BF16。这里的 0.70、0.45、0.98 和 0.975 都是
虚构的门限演示值，不是论文或生产测量；真实切换必须使用带误差区间的任务切片。

### 7.2 按任务路由，而不是全流量切换

一个实用策略可以按长度、风险和任务类型选择：

| 任务 | 默认 KV | 原因 |
| --- | --- | --- |
| 短聊天 | FP8 | 状态短，容量和并发收益明显 |
| 普通 RAG | FP8/混合 | 需要用引用回归集确认 |
| 数字审计 | BF16 或高精度 K | 排序和数字细节敏感 |
| 工具 Agent | 混合 | 工具历史和状态字段需要保护 |
| 超长文档 | FP8 试验池 | 需要长位置分桶和回退 |

路由条件要进入 trace。否则同一个模型 id 下，质量和延迟差异来自 dtype、任务、
回退还是缓存命中就无法解释。

### 7.3 单成功任务成本

若一次尝试的服务成本和验证成本分别为 `C_serve`、`C_verify`，一次尝试成功的
概率为 `p`，并且失败后按同一策略独立重试，最简单的期望成本可以先写成：

~~~math
C_{\mathrm{success}}
=
\frac{C_{\mathrm{serve}}+C_{\mathrm{verify}}}{p}
~~~

这里的 `p` 是在明确的任务切片、重试上限和停止条件下估计的成功概率；如果重试
会改用更大模型、人工复核或产生额外的固定费用，就应把这些费用写成独立的期望项，
例如 `C_fallback` 或 `C_human`，而不是把已经按期望计算的 `C_retry` 再除一次。这个
式子提醒我们：一次 FP8 请求节省的显存，如果被质量失败、重试、人工复核或回读
成本抵消，就没有真正降低单位成功任务成本。生产账本应记录尝试次数分布、成功率、
fallback 和人工处置，并以实际成功任务数校准这个近似。

## 8. 长上下文和 FP8 KV 的评估

### 8.1 短文本通过不代表长上下文通过

长上下文会让更多低精度历史状态参与后续 attention，局部误差也可能经层间计算影响
下游；退化不一定随 token 数线性累积。评测至少需要按长度分桶，例如短、中、长和
接近服务上限的输入；每个桶再分 cache hit/miss、短输出/长输出和 batch。

长上下文任务应测：

- 远距离 evidence recall；
- citation support 和 unsupported claim；
- 多轮工具历史是否完整；
- 代码和数字 token 是否保真；
- logits 漂移、格式合法率和拒答；
- TTFT、TPOT、P95/P99、OOM、抢占和恢复。

### 8.2 不只测最终文本

最终文本可能看起来流畅，却引用了错误的段落。可以在固定请求上保存 BF16 baseline
和 FP8 候选路径的中间诊断：K/V 反量化误差、attention score 差异、关键位置
的 top attention、最终 logits 差异和任务结果。中间诊断用于定位，最终任务指标
用于决定是否采用，二者不能互相代替。

### 8.3 K/V 消融和敏感层

四组 K/V 消融可以发现错误方向：

1. BF16 K / BF16 V：高精度基线；
2. FP8 K / BF16 V：观察 attention 排序敏感性；
3. BF16 K / FP8 V：观察聚合内容敏感性；
4. FP8 K / FP8 V：观察完整候选路径。

如果只有少数层或 head 对长程引用敏感，可以保留这些层的高精度，或者为高风险
任务使用不同 KV pool。混合精度复杂度会增加，但可能比全模型 BF16 更省资源、比
全模型 FP8 更稳。

### 8.4 一次真实 serving 验证揭示的精度与速度边界

2026 年 4 月，vLLM 团队发布了 Hopper/H100 与 Blackwell/B200 上的 FP8 KV/attention
验证报告。它特别有价值的地方是同时记录失败原因、kernel 修复和 workload 取舍。
但要读清实验对象：报告的 `--kv-cache-dtype fp8` 路径在 FlashAttention 3 上还会把
QK 与 score-value 运算放到 FP8 域；实验统一使用 per-tensor、未校准 scale=1.0，
因此它不是“只改变 KV 存储、其它计算完全不变”的单因素对照。所有数值都绑定报告
里的模型、backend、硬件和版本。[报告](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html)

| 报告中的问题 | 固定条件与观测 | 它说明什么 |
| --- | --- | --- |
| 128k 检索准确率骤降 | Llama-3.1-8B、H100、FlashAttention 3 的 Needle-in-a-Haystack case：BF16 91%，原 FP8 路径 13%；团队追到长 contraction 维度上的累加精度问题。加入 two-level accumulation，把部分和写入真正的 FP32 register 后恢复到 89% | FP8 出错不一定是 E4M3/KV 失真；QK/PV 累加、softmax 和长归约都属于数值路径。two-level accumulation 有额外寄存器成本，不能只看精度修复 |
| 单请求 decode 何时变快 | Llama-3.1-8B、H100、单请求：ITL 随输入长度的拟合斜率由 `4.37×10^-5` 降到 `2.37×10^-5 ms/token`，intercept 从 6.44 ms 变为 6.58 ms，拟合 break-even 约 7,010 tokens | 降低每个历史 token 的读/算成本不代表短上下文立刻更快；固定转换开销会形成长度门槛 |
| 混合 attention 的滑窗层是否都量化 | gpt-oss-20b、H100：旧 v0.10.2 FP8 ITL slope 为 BF16 的 96%、break-even 741,565 tokens；修复后的 v0.19.1 全 FP8 为 80%、跳过 128-token sliding-window 层后为 71%，对应 break-even 22,109 与 7,659 tokens | bounded window 的字节收益会饱和，量化/反量化成本却仍在；按层跳过有时比全模型统一 dtype 更合理 |
| decode 快是否代表整条请求都快 | Gemma-4-E2B、H100、`head_dim=256`：FP8 ITL slope 为 BF16 的 68%，但长输入 TTFT 拟合二次项约增至 1.6×，与 two-level accumulation 的寄存器压力相关 | prefill 与 decode 的瓶颈不同；必须分别测 TTFT 与 ITL，不能拿 decode 斜率代替端到端 latency |

在另一个并发 8、150 个请求、每个约 20k 输入和 2k 输出的 Llama-3.1-8B 测试中，
报告的输出吞吐从 450.3 增至 517.5 token/s（+14.9%），median ITL 从 15.18 ms
降至 12.93 ms。它说明同一硬件预算内的 cache 容量增加还可能让 scheduler 放入更多
请求；这个比例不能与上表单请求的 ITL slope 混成一个指标。

所以 FP8 发布矩阵至少要分开记录 cache payload bytes/token、attention compute dtype、
累加路径、prefill TTFT、decode ITL slope、并发吞吐、break-even context 和任务质量。
模型包含局部滑窗层或使用大 head dimension 时，还应实测按层跳过、不同 backend 和
两级累加配置的组合。

## 9. 量化 cache 的 manifest、灰度和回滚

### 9.1 manifest 字段

一个可复现的 FP8 KV artifact 至少应保存：

~~~text
model_revision
tokenizer_revision
template_hash
position_scheme
kv_dtype
k_dtype / v_dtype
fp8_variant            e4m3 / e5m2 / platform-specific variant
scale_scheme
scale_dtype
scale_axis_and_shape
scale_revision
rounding_mode
saturation_mode
attention_compute_dtype
accumulation_dtype
page_size
page_layout
kernel_revision
engine_revision
tested_hardware
calibration_revision
calibration_data_revision
fallback_policy
cache_schema_version
~~~

其中 `cache_schema_version` 不能省略。旧 page 是否可迁移，取决于 dtype、scale、
layout、position 和模型 revision 是否全部兼容。

### 9.2 灰度证据

离线 replay 可以控制输入和 baseline；shadow 可以观察真实长度、cache hit 和
租户分布；canary 才能测到真实连接、取消、page 回收和流式体验。三者都需要比较
BF16 baseline 与 FP8 候选路径，并保留失败样本。

发布时可以先让低风险短聊天进入 FP8，再逐步扩大到普通 RAG；数字审计、关键工具
和超长任务暂时留在 BF16，直到各自切片通过。灰度条件应包括质量、安全、P99、
OOM、fallback 和成本，不能只看显存下降。

### 9.3 回滚和排空

回滚不只是把一个配置值改回 BF16。旧 FP8 page 可能仍在共享池、swap 层或 worker
之间；scale、prefix hash、stream usage 和 allocator 统计也需要处理。稳妥的路径
是停止新 FP8 分配，等待或取消旧请求，排空/重算不兼容 page，再恢复 BF16 pool。

对于无副作用聊天，重算通常比跨版本迁移 KV 简单；对于长 Agent，必须优先保存
工作流状态和外部工具状态，不能把 KV 当成唯一的任务状态。

## 10. 一个可运行的 FP8 KV 容量与质量诊断示例

下面的 demo 使用标准库比较 BF16 和 FP8 payload 的教学容量，并模拟普通聊天和
数字审计两类任务的质量底线。它不会实现真实 FP8 编码或 GPU kernel；代码中的
`scale_overhead=0.08`、任务分数和质量门限都是教学假设，不是来源报告或生产数据；
这个参数把 scale、metadata 和对齐作为显式账本项，方便读者检查容量与质量门禁的
分离。代码只验证这本账和路由条件，不模拟实际 FP8 编码或 GPU kernel。

~~~python
import math
from pprint import pprint


def kv_payload_gib(layers, batch, tokens, kv_heads, head_dim, bytes_per_value):
    dimensions = (layers, batch, tokens, kv_heads, head_dim)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in dimensions):
        raise ValueError("dimensions must be positive integers")
    if (isinstance(bytes_per_value, bool)
            or not isinstance(bytes_per_value, (int, float))
            or not math.isfinite(bytes_per_value)
            or bytes_per_value <= 0):
        raise ValueError("bytes_per_value must be finite and positive")
    total = 2 * layers * batch * tokens * kv_heads * head_dim * bytes_per_value
    return total / (1024 ** 3)


def quantized_kv_gib(
    layers,
    batch,
    tokens,
    kv_heads,
    head_dim,
    payload_bytes,
    scale_overhead,
):
    if (isinstance(scale_overhead, bool)
            or not isinstance(scale_overhead, (int, float))
            or not math.isfinite(scale_overhead)
            or scale_overhead < 0):
        raise ValueError("scale_overhead must be finite and non-negative")
    payload = kv_payload_gib(
        layers, batch, tokens, kv_heads, head_dim, payload_bytes
    )
    return payload * (1 + scale_overhead)


def validate_task_scores(tasks):
    if not tasks:
        raise ValueError("tasks must not be empty")
    for name, row in tasks.items():
        if not name or "quality" not in row or "floor" not in row:
            raise ValueError("each task needs a name, quality, and floor")
        quality = row["quality"]
        floor = row["floor"]
        scores = (quality, floor)
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               for value in scores):
            raise ValueError("quality and floor must be numeric")
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in scores):
            raise ValueError("quality and floor must be finite values in [0, 1]")


layers = 64
batch = 1
tokens = 131072
kv_heads = 8
head_dim = 128

bf16_gib = kv_payload_gib(layers, batch, tokens, kv_heads, head_dim, 2)
fp8_gib = quantized_kv_gib(
    layers, batch, tokens, kv_heads, head_dim, payload_bytes=1, scale_overhead=0.08
)

tasks = {
    # Synthetic values demonstrate a release gate; they are not benchmark results.
    "chat": {"quality": 0.975, "floor": 0.970},
    "audit": {"quality": 0.890, "floor": 0.950},
}
validate_task_scores(tasks)
quality_failures = [
    name for name, row in tasks.items() if row["quality"] < row["floor"]
]

signals = {
    "memory_reduced": fp8_gib < bf16_gib,
    "chat_quality_floor": tasks["chat"]["quality"] >= tasks["chat"]["floor"],
    "audit_quality_floor": tasks["audit"]["quality"] >= tasks["audit"]["floor"],
    "quality_failures_are_visible": bool(quality_failures),
}
actions = []
if not signals["chat_quality_floor"]:
    actions.append("keep_bf16_for:chat")
if not signals["audit_quality_floor"]:
    actions.append("keep_bf16_for:audit")
if not signals["memory_reduced"]:
    actions.append("inspect_scale_and_kernel")

decision = "selective_enablement" if actions else "expand_fp8_cohort"
report = {
    "bf16_gib": round(bf16_gib, 2),
    "fp8_gib_with_overhead": round(fp8_gib, 2),
    "memory_saving_rate": round(1 - fp8_gib / bf16_gib, 3),
    "quality_failures": quality_failures,
    "signals": signals,
    "actions": actions,
    "decision": decision,
}
pprint(report, sort_dicts=False)
~~~

一组可复现输出如下：

~~~text
{'bf16_gib': 32.0,
 'fp8_gib_with_overhead': 17.28,
 'memory_saving_rate': 0.46,
 'quality_failures': ['audit'],
 'signals': {'memory_reduced': True,
             'chat_quality_floor': True,
             'audit_quality_floor': False,
             'quality_failures_are_visible': True},
 'actions': ['keep_bf16_for:audit'],
 'decision': 'selective_enablement'}
~~~

这个结果说明 FP8 的容量收益可以成立，但不应因为普通聊天通过就把数字审计任务
一起切过去。更细的系统还要加入 K/V 分开消融、长上下文位置分桶、cache 生命周期、
fallback 比例和真实 GPU kernel；教学程序只负责把“容量信号”和“质量信号”分开。

## 11. 常见误区与排查路径

### 11.1 误区：FP8 KV 只要占一半显存

payload 理论上从 2 字节降到 1 字节，但 scale、metadata、page、对齐、workspace、
通信和 allocator 仍然存在。应比较实际 `kv_bytes_per_token`、峰值显存和并发，而
不是只做位宽除法。

### 11.2 误区：K 和 V 使用同一个 scale 最简单

简单不代表合适。K 误差会改动 query-key logits 和 softmax 权重，V 误差进入加权
内容聚合；不同层/head 的动态范围
也可能不同。至少做 K-only、V-only 和 K/V 一起量化的消融。

### 11.3 误区：离线 perplexity 没变，所以线上安全

perplexity 不会告诉你引用是否支持、工具参数是否正确、page 是否错位、取消后
是否释放、prefix 是否跨租户复用。长上下文和高风险任务要有独立质量和安全样本。

### 11.4 误区：FP8 一定比 BF16 更快

如果转换、scale load、通信或 fallback 成本更大，TPOT 可能不变甚至变差。必须
记录实际 kernel 和慢路径比例。

### 11.5 误区：旧 KV 可以直接迁移到新量化版本

dtype、scale、layout、position、模型 revision 或 page schema 任一变化，都可能使
旧 cache 不再兼容。无法证明兼容时，排空并重算通常比静默复用更可靠。

### 11.6 排查顺序

长上下文质量下降时，可以按以下顺序定位：

1. 确认 FP8 路径确实生效，检查实际 dtype、kernel 和 fallback。
2. 比较 scale 分布、饱和比例、page layout 和 cache bytes。
3. 做只量化 K、只量化 V 的消融。
4. 按 layer、head、长度、位置和任务桶定位。
5. 检查 prefix hit、page migration、preemption restore 和取消回收。
6. 再决定是修改 scale、保留敏感层高精度，还是对某类 workload 回退。

## 12. 资料与证据边界

本章的公式用于解释容量和 scale 机制，不是任何具体 GPU 的性能保证。资料的用途
需要分层：

1. [OCP OFP8 v1.0](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)
   与[FP8 格式原论文](https://arxiv.org/abs/2209.05433)支持编码值、特殊值、舍入和
   scale 设计背景；格式规范不定义 KV runtime 的 scale 维护。
   [ROCm FP8 类型表](https://rocm.docs.amd.com/en/docs-7.2.3/reference/precision-support.html)
   和 [PyTorch dtype 表](https://docs.pytorch.org/docs/main/tensor_attributes.html)
   则显示 OCP/FNUZ 变体及对应 dtype 名称；实际兼容性仍需按设备和 kernel 核实。
2. [vLLM Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
   和 [LLM Compressor calibration recipe](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_kv_cache/)
   支持当前公开 scale/calibration/backend 用法；[TensorRT-LLM FP8 guide](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html)
   显示其 KV dtype 与模型量化分别配置。它们都是 rolling docs，需锁定部署 release。
3. [vLLM 2026 FP8 KV/attention 验证报告](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html)
   提供 H100/B200、FlashAttention/FlashInfer 上的故障归因和 workload 数据；其结果包含
   FP8 attention 计算，不能视作只换 KV payload 的纯对照。
4. [KIVI](https://arxiv.org/abs/2402.02750) 与 [KVQuant](https://arxiv.org/abs/2401.18079)
   用来比较 K/V 非对称分组、低比特码本、outlier 与流式写入；[PagedAttention](https://arxiv.org/abs/2309.06180)
   讨论分页管理，[KV Cache serving survey](https://arxiv.org/abs/2607.08057) 帮助
   区分压缩、迁移、调度与淘汰。它们的论文结果各自绑定模型、task 和实现。
5. [Transformer Engine 2.20.2 FP8 Current Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html)
   用来解释一般 tensor amax/scale/cast 与转换成本，不代表该训练库提供了本章所述的
   serving KV manager。[Hugging Face KV Cache 文档](https://huggingface.co/docs/transformers/en/kv_cache)
   是通用状态接口参考，不约束其他 runtime 的 page/scale layout。
6. 本章 Python demo 只计算教学账本与合成质量门禁，不实现 OCP FP8 cast、attention
   dequant kernel 或 GPU page allocator；其结果不代表 benchmark。

延伸阅读：

1. [OCP OFP8 v1.0](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)：FP8 二进制格式和转换规则。
2. [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)：格式设计动机及不同数值格式的训练/推理研究。
3. [vLLM Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) 与 [LLM Compressor KV calibration](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_kv_cache/)：公开 scale 和校准接口。
4. [The State of FP8 KV-Cache and Attention Quantization in vLLM](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html)：H100/FA3、B200/FlashInfer 的质量和速度验证及修复经过。
5. [TensorRT-LLM FP8 quantization](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html)：模型量化与 KV-cache dtype 的独立配置入口。
6. [Transformer Engine 2.20.2 FP8 Current Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html)：FP8 amax、scale 和 cast 流程示例。
7. [KIVI](https://arxiv.org/abs/2402.02750) 与 [KVQuant](https://arxiv.org/abs/2401.18079)：K/V 分组、通道 outlier 与超低 bit KV 量化对照。
8. [PagedAttention](https://arxiv.org/abs/2309.06180) 与 [System-Aware KV Cache Optimization survey](https://arxiv.org/abs/2607.08057)：分页、表示压缩、迁移、调度和淘汰的系统边界。
9. [Hugging Face KV Cache](https://huggingface.co/docs/transformers/en/kv_cache)：通用 KV cache 生命周期与接口参考。
10. [SmoothQuant](https://arxiv.org/abs/2211.10438)：相邻的 activation outlier 处理方法；不是 FP8 KV 的直接效果证据。

## 13. 本章小结

FP8 KV Cache 的本质是用更小的历史状态 payload 换取显存和调度空间，并可能降低
decode 读取流量；代价是数值误差、转换成本和状态生命周期进入 serving 主路径。

1. 先用 `2LBTH_kvD_hb` 估算 KV 数量级，再把 scale、page、workspace、对齐和余量
   加入实际账本。
2. E4M3/E5M2 的指数/尾数和特殊值不同；OCP 格式不定义 scale 规则，实际精度由
   payload、scale 粒度、转换模式、attention compute 和 accumulation 一起决定。
3. K 误差会先改变 attention logits/权重，V 误差进入加权输出；KIVI、KVQuant 与
   serving runtime 展示了不同量化轴、bit 数、校准和 kernel 约束。
4. 共享 page 的 payload 必须和 scale revision 配对；prefix sharing、抢占、swap、迁移、
   speculative rollback 和取消回收都要保留 dtype、scale、position、page 和 model revision。
5. 短文本或 perplexity 通过不代表长上下文引用、数字、代码和工具任务没有退化；长
   contraction 还可能暴露 kernel 累加精度问题。
6. 报告容量、TTFT、ITL slope、break-even context、并发吞吐和任务质量；payload 减半
   不等于 TPOT 或单位成功成本减半。
7. 量化发布应支持按任务/层路由、敏感 KV 保留高精度和可观察 fallback，依据实测逐步
   扩大请求范围。

当一个系统能说明“哪些请求使用 FP8、哪些请求保留 BF16、为什么这样分、发生异常
如何回退、旧 page 如何处理、质量和成本怎样证明”，FP8 KV 才真正从一个 dtype
选项变成可解释的工程设计。
