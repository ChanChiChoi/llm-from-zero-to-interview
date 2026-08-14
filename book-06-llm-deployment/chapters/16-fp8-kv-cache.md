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

FP8 通常至少要区分 exponent/mantissa 分配不同的格式，例如 E4M3 和 E5M2 一类
编码。不同格式在动态范围和有效精度之间取舍不同；具体特殊值、舍入和硬件支持
应以目标 GPU、Transformer Engine 或 serving runtime 的文档为准。

对一个 K/V block `G`，可以用抽象量化模型表示：

~~~math
q_i=Q\left(\frac{x_i}{s_G}\right),
\qquad
\hat{x}_i=s_GD(q_i),
\quad i\in G
~~~

`x_i` 是原始 K 或 V，`s_G` 是该组 scale，`q_i` 是 FP8 payload，`D` 是对应
FP8 码表的反量化函数。真实系统还要说明 scale 是否是静态、动态、per-tensor、
per-head、per-channel 或 per-block。

### 3.2 scale 粒度的取舍

**Per-tensor scale** 元数据少、kernel 简单，但一个异常值可能让整个 tensor 的
普通值只占很小的有效范围。

**Per-layer/head scale** 能适应层和 head 之间的分布差异，元数据仍相对可控；但
page 和 kernel 要知道每个 layer/head 的索引。

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

## 4. K 和 V 为什么不能默认使用同一种策略

### 4.1 K 影响 attention 排序

Key 进入 query-key 相似度：

~~~math
S_{t,j}=\frac{q_t k_j^{\mathsf{T}}}{\sqrt{D_h}}
~~~

量化 K 可能改变不同历史位置的 score 相对大小，进而改变 softmax 后的 attention
权重。少数高注意力 token、位置边界、数字和检索证据可能对这种变化很敏感。

如果 K 的误差主要影响排序，最终输出可能出现“看似流畅，但引用选错位置”的
失败。长上下文评估要记录 exact retrieval、evidence recall 和 citation support，
不能只看生成文本的表面相似度。

### 4.2 V 影响聚合内容

Value 进入加权和：

~~~math
z_t=\sum_j a_{t,j}v_j
~~~

量化 V 会直接改变聚合后的表示。它未必改变 attention 的位置选择，却可能损失
数字、代码符号、工具参数或引用片段中的细节。K/V 的动态范围和敏感层可能不同，
因此相同 scale、相同 block 或相同质量阈值不是自然成立的选择。

### 4.3 四组最小消融

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
scale_revision
model_revision
position_scheme
reference_count
~~~

`reference_count` 对 prefix sharing 很重要；共享 page 被多个请求引用时，单个请求
取消不能直接释放它。租户 namespace 则决定公共前缀和私有前缀是否可以共享。

### 6.3 线上症状如何定位

如果显存下降但 TPOT 没改善，检查 dequant、scale load、attention kernel 和通信；
如果单请求通过而高并发失败，检查 page 分配、batch reorder、抢占恢复和引用计数；
如果只有 cache hit 请求出错，检查 prefix key、dtype、scale revision 和 position；
如果长上下文质量下降，做只量化 K/V 的消融并按层/head定位。

这样排查比直接说“FP8 精度不够”更有效，因为很多线上错误来自状态生命周期或
版本错配，而不是 payload 本身。

## 7. 容量收益和产品策略

### 7.1 一个数量级例子

设 `L=64`、`H_kv=8`、`D_h=128`、`B=1`、`T=131072`。BF16 的理想 KV 约为
32 GiB，FP8 payload 约为 16 GiB；如果 scale、metadata 和对齐额外占 8%，候选
约为 17.28 GiB。它可以使一个 GPU 多容纳请求，或在相同并发下保留更长上下文。

这只是容量层面的结果。若 FP8 让长上下文 citation support 从 0.70 降到 0.45，
审计问答不能启用它；若普通聊天从 0.98 降到 0.975，且满足质量底线，路由器可能
允许普通聊天使用 FP8、审计任务使用 BF16。

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

若候选路径的服务成本、验证成本和重试成本分别为 `C_serve`、`C_verify`、`C_retry`，
成功概率为 `p`，可以先用：

~~~math
C_{\mathrm{success}}
=
\frac{C_{\mathrm{serve}}+C_{\mathrm{verify}}+C_{\mathrm{retry}}}{p}
~~~

这个式子提醒我们：一次 FP8 请求节省的显存，如果被质量失败、重试、人工复核或
回读成本抵消，就没有真正降低单位成功任务成本。

## 8. 长上下文和 FP8 KV 的评估

### 8.1 短文本通过不代表长上下文通过

KV 误差会随着历史长度和 attention 读取积累。评测至少需要按长度分桶，例如短、
中、长和接近服务上限的输入；每个桶再分 cache hit/miss、短输出/长输出和 batch。

长上下文任务应测：

- 远距离 evidence recall；
- citation support 和 unsupported claim；
- 多轮工具历史是否完整；
- 代码和数字 token 是否保真；
- logits 漂移、格式合法率和拒答；
- TTFT、TPOT、P95/P99、OOM、抢占和恢复。

### 8.2 不只测最终文本

最终文本可能看起来流畅，却引用了错误的段落。可以在固定请求上保存 BF16 baseline
和 FP8 candidate 的中间诊断：K/V 反量化误差、attention score 差异、关键位置
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
scale_scheme
scale_dtype
scale_revision
page_size
page_layout
kernel_revision
engine_revision
tested_hardware
calibration_revision
fallback_policy
cache_schema_version
~~~

其中 `cache_schema_version` 不能省略。旧 page 是否可迁移，取决于 dtype、scale、
layout、position 和模型 revision 是否全部兼容。

### 9.2 灰度证据

离线 replay 可以控制输入和 baseline；shadow 可以观察真实长度、cache hit 和
租户分布；canary 才能测到真实连接、取消、page 回收和流式体验。三者都需要比较
BF16 baseline 与 FP8 candidate，并保留失败样本。

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
`scale_overhead` 只是把 scale、metadata 和对齐作为显式假设，方便读者检查账本。

~~~python
from pprint import pprint


def kv_payload_gib(layers, batch, tokens, kv_heads, head_dim, bytes_per_value):
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
    payload = kv_payload_gib(
        layers, batch, tokens, kv_heads, head_dim, payload_bytes
    )
    return payload * (1 + scale_overhead)


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
    "chat": {"quality": 0.975, "floor": 0.970},
    "audit": {"quality": 0.890, "floor": 0.950},
}
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

简单不代表合适。K 影响 attention 排序，V 影响内容聚合；不同层/head 的动态范围
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

1. vLLM quantized KV cache、automatic prefix caching 和 TensorRT-LLM 文档用于
   确认当前 runtime 的公开入口；配置可用不等于目标模型、shape 和 GPU 都走快路径。
2. NVIDIA Transformer Engine FP8 primer 用于理解 FP8 数值和训练/推理相关背景；
   Transformer Engine 的能力不等于每个 serving engine 的能力。
3. Hugging Face KV Cache 文档用于确认通用 cache 接口和状态概念；自建 runtime 的
   page、scale 和迁移布局可能不同。
4. KIVI、KVQuant 和 PagedAttention 论文用于理解 KV 量化、低比特 cache 和分页
   管理的研究背景；论文 benchmark 不能直接替代线上长上下文压测。
5. 本章 Python demo 只计算教学账本，不实现真实 FP8 编码、dequant kernel 或 GPU
   page allocator。

延伸阅读：

1. [vLLM Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html)：vLLM 当前量化 KV Cache 的官方入口。
2. [vLLM Quantization](https://docs.vllm.ai/en/latest/features/quantization/)：vLLM 量化能力和实现索引；具体模型支持需看版本。
3. [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)：前缀 cache 与 KV 管理的工程边界。
4. [TensorRT-LLM Quantization](https://nvidia.github.io/TensorRT-LLM/features/quantization.html)：NVIDIA TensorRT-LLM 量化入口。
5. [TensorRT-LLM Precision](https://nvidia.github.io/TensorRT-LLM/reference/precision.html)：精度类型和 runtime 配置入口。
6. [NVIDIA Transformer Engine FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)：FP8 格式、缩放和训练/推理背景。
7. [Hugging Face KV Cache](https://huggingface.co/docs/transformers/en/kv_cache)：通用 KV Cache 接口和状态说明。
8. [KIVI](https://arxiv.org/abs/2402.02750)：低比特 KV Cache 量化研究入口。
9. [KVQuant](https://arxiv.org/abs/2401.18079)：极长上下文 KV 量化研究入口；论文结果需按目标模型复现。
10. [PagedAttention](https://arxiv.org/abs/2309.06180)：分页 KV Cache 管理研究背景。
11. [SmoothQuant](https://arxiv.org/abs/2211.10438)：激活异常值平滑和低精度量化的相邻方法。
12. [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)：FP8 等低精度训练/计算的软件实现入口。

## 13. 本章小结

FP8 KV Cache 的本质是用更小的历史状态 payload 换取显存、带宽和调度空间，但它
把数值误差和状态生命周期引入了 serving 主路径。

1. 先用 `2LBTH_kvD_hb` 估算 KV 数量级，再把 scale、page、workspace、对齐和余量
   加入实际账本。
2. FP8 格式、scale 粒度、动态范围、累加和硬件 kernel 必须作为一组配置理解。
3. K 影响 attention 排序，V 影响内容聚合，不能默认使用相同 scale、精度或回退。
4. prefix sharing、抢占、swap、迁移、speculative rollback 和取消回收都必须携带
   dtype、scale、position、page 和 model revision。
5. 短文本平均分通过不代表长上下文、引用、数字、代码和工具任务没有退化。
6. 存储、kernel、engine 和任务四层证据要同时记录，理论 payload 减半不等于 TPOT
   或单位成功成本减半。
7. 量化发布应支持按任务路由、敏感层高精度和可观察的 fallback，而不是全流量一键
   切换。

当一个系统能说明“哪些请求使用 FP8、哪些请求保留 BF16、为什么这样分、发生异常
如何回退、旧 page 如何处理、质量和成本怎样证明”，FP8 KV 才真正从一个 dtype
选项变成可解释的工程设计。
