# 第十五章：FP4、MXFP4 与 NVFP4：四位浮点不是一个统一开关

当模型规模、上下文长度和并发一起增长时，显存和内存带宽会越来越接近系统上限。
INT4 已经证明低比特权重可以显著降低存储，但整数刻度在激活分布、异常值和矩阵
计算路径上有自己的限制。于是，FP8、FP6、FP4 以及各种 block scaling 方案逐渐
进入训练和推理系统。

这组名词很容易造成误解：FP4 是一个位宽和浮点编码家族，MXFP4 是带有 microscaling
组织的格式路线，NVFP4 是 NVIDIA 硬件和软件栈中的具体低精度路径。它们都可能使
用 4 bit payload，却不一定使用相同的数值集合、scale 编码、block 大小、累加精度
或 kernel。把它们都叫“4 bit 浮点”，只能说明故事的开头，不能说明部署行为。

本章从数值表示开始，逐步解释 scale 和 metadata 为什么决定真实压缩率，为什么
异常值会吞掉有效精度，权重/激活/KV 为什么必须分开量化，以及如何从格式定义走到
kernel、runtime、评估和发布 manifest。涉及具体硬件时，正文会明确区分官方资料
披露、软件模拟和目标 workload 实测，避免把某个版本的结果写成永久规律。

## 1. 为什么低比特浮点会成为一个独立问题

### 1.1 从 FP16、BF16 到 INT4

FP16 和 BF16 通常提供比整数更大的动态范围，适合激活和累加；INT8、INT4 则能以
较低的存储和带宽成本保存权重。整数方案一般使用 scale 把浮点值映射到有限的整数
格点：

~~~math
q_i=\mathrm{round}\left(\frac{x_i}{s}\right),
\qquad
\hat{x}_i=sq_i
~~~

`x_i` 是原始值，`q_i` 是量化值，`s` 是 scale，`\hat{x}_i` 是反量化值。scale
可以按 tensor、channel、group 或 block 设置。粒度越细，越能适应局部范围，但
scale 元数据、读取和 kernel 复杂度也越高。

FP4 的动机是让低比特码保留某种浮点式的指数/尾数结构，而不是只使用等距整数
格点。对动态范围变化较大的权重或激活，非均匀码本可能比等距整数格点更合适；
但它并不会自动消除量化误差。scale、异常值、累加精度和硬件支持仍决定最终结果。

### 1.2 “4 bit”只描述 payload

一个量化 block 的存储不只有每个元素的 payload。可以写成近似账本：

~~~math
M_{\mathrm{block}}
\approx
n\frac{b_q}{8}
+\frac{b_s}{8}
+\frac{b_m}{8}
~~~

`n` 是 block 中元素数量，`b_q` 是每个元素的 payload bit 数，`b_s` 是 scale 的
存储 bit 数，`b_m` 是额外 metadata 的 bit 数。FP4 只把 `b_q` 设为 4，并没有
告诉我们 `n`、`b_s`、`b_m`、scale dtype、layout、累加 dtype 和未量化层。

例如每 128 个元素共享一个 16 bit scale 和一个 8 bit metadata，则平均每元素还
要承担 `24/128=0.1875 bit` 的额外存储。block 很小时，metadata 占比更大；block
很大时，scale 对异常值的适应能力更弱。

### 1.3 低比特系统至少有五个 dtype

一个生产量化路径中，经常同时出现：

1. payload dtype：权重或激活在存储中使用的低比特格式；
2. scale dtype：描述 block 动态范围的格式；
3. accumulation dtype：矩阵乘和归约使用的累加精度；
4. activation/KV dtype：中间激活或历史 K/V 的存储格式；
5. output/logits dtype：输出层和采样前 logits 使用的精度。

因此，“模型支持 FP4”不是完整配置。一个实际服务可能使用 FP4 权重、FP8 激活、
BF16 累加和 FP8 KV；也可能只对权重使用 FP4，attention 和 KV 仍然使用 BF16。
如果没有逐项记录，离线模拟结果和线上服务很容易对不上。

## 2. FP4、MXFP4 与 NVFP4 的概念边界

### 2.1 FP4 不是一个唯一的码表

普通“FP4”可能只是对 4 bit 浮点 payload 的泛称。一个具体编码需要说明 sign、
exponent、mantissa、特殊值、舍入方式、subnormal 是否存在以及 scale 如何作用。
常见讨论会提到 E2M1 一类布局，但即使 payload 布局相同，外部 scale、block 大小
和累加路径不同，也会产生不同部署行为。

所以，阅读一份模型卡或 runtime 配置时，不能只看 `fp4` 这个字符串。至少要继续
追问：

- payload 的实际码表是什么；
- scale 是 per-tensor、per-channel、per-group 还是 per-block；
- scale 本身是什么 dtype，放在什么内存布局；
- 矩阵乘是否直接消费 payload，还是先解包到更高精度；
- 哪些层、哪些 shape 和哪些 GPU 走原生路径；
- activation、KV 和累加是否仍然保持更高精度。

### 2.2 MXFP4：把 scale 组织也纳入格式

MXFP4 通常被放在 microscaling 格式谱系中理解。它的重点不只是使用 4 bit 元素，
而是规定一组元素共享某种 scale，并让 payload、scale 和 block 组织可以被硬件
批量处理。不同 profile 或 runtime 可能对 block 长度、scale 编码和布局有具体约定，
因此“MXFP4”应被理解为一条带有微缩放规则的格式路线，而不是所有 block-scaled
FP4 实现的通用别名。

microscaling 的直觉是：把一个大 tensor 切成许多小组，每组用自己的动态范围。它
比一个全局 scale 更能保留小值，但需要读取更多 scale，并且矩阵乘 kernel 必须知道
scale 的位置和广播方式。若软件只模拟 payload 而没有模拟真实 scale dtype 和布局，
得到的误差不能直接代表生产 kernel。

### 2.3 NVFP4：硬件路径优先于名字

NVFP4 通常指 NVIDIA 软件和硬件生态中面向 FP4 推理/训练的具体实现路径。它可能
包含特定的 payload、block scale、全局 scale、累加策略、校准流程和 Tensor Core
kernel。不同 GPU、驱动、TensorRT-LLM 或 Model Optimizer 版本的支持范围可能不同。

对部署工程师而言，NVFP4 最重要的问题不是“它是不是比 MXFP4 更先进”，而是：

1. 目标 GPU 是否有原生指令或高效 kernel；
2. 目标 TensorRT-LLM/runtime 版本是否支持目标模型形状；
3. scale 和 metadata 是否能被 engine 正确加载；
4. 不支持的层是否会回退到 FP8、FP16 或慢速解包路径；
5. 回退后显存、TTFT、TPOT 和质量是否仍满足服务目标。

因此，MXFP4 更适合从格式和微缩放规则讨论，NVFP4 更需要从硬件、kernel、engine
和版本契约讨论。两者都不能仅靠文件名互换。

### 2.4 一张比较表应该写什么

| 维度 | 泛称 FP4 | MXFP4 路线 | NVFP4 路线 |
| --- | --- | --- | --- |
| 讨论重点 | payload 编码和数值集合 | payload 与 microscale/block 组织 | 硬件、kernel、engine 和校准路径 |
| scale | 可能没有统一约定 | 通常是共享 scale 的格式化组织 | 由具体 NVIDIA profile 和 runtime 定义 |
| 性能证据 | 需要软件和硬件分别验证 | 需要 block/layout/kernel 验证 | 需要目标 GPU、驱动和 engine 压测 |
| 主要风险 | 名称过宽，配置不完整 | metadata 与 scale 开销 | 版本支持和 fallback 路径 |
| 可迁移结论 | 4 bit 不等于 4 倍端到端收益 | block 越细不一定越快 | 原生支持比文件后缀更重要 |

这张表的目的不是给三种格式排一个永久名次，而是提醒读者比较同一层次的对象。

## 3. Block scale 如何改变有效精度

### 3.1 从一个共享 scale 开始

把张量分成 block `B_g`，每个 block 使用 scale `s_g` 和低比特码 `q_i`，教学化
的量化与重构可以写成：

~~~math
q_i=Q\left(\frac{x_i}{s_g}\right),
\qquad
\hat{x}_i=s_gD(q_i),
\quad i\in B_g
~~~

`Q` 是量化或舍入函数，`D` 是从 payload 码表取值的反量化函数。若所有元素共享
一个全局 scale，元数据很少，但一个异常值可能支配整个 tensor；若每个元素独立
设置 scale，误差可以更小，但 scale 本身会抵消压缩收益，也很难被高效矩阵乘消费。
block/group scale 是两者之间的工程折中。

### 3.2 异常值为什么会消耗小值的精度

考虑一个 block 中 15 个值落在 `[-1,1]`，另一个值为 `20`。如果 scale 由最大
绝对值决定，则 `s=20`，普通值会被缩放到 `[-0.05,0.05]`。低比特码本的有限
等级中，许多普通值可能落到相同或相邻的格点，重构后的相对误差会明显增大。

可以把误差拆成两个方向：

~~~math
\epsilon_i=x_i-\hat{x}_i,
\qquad
\mathrm{MSE}(B_g)=\frac{1}{|B_g|}\sum_{i\in B_g}\epsilon_i^2
~~~

平均 MSE 仍然只是一个诊断量。若小值位于 attention score、router、归一化或输出
logits 的敏感路径，它们对任务的影响可能比大多数普通权重更大。反过来，某个层
的 MSE 较高，也不一定会造成可见的任务退化。因此需要按层、channel、token 类型
和任务切片观察误差。

### 3.3 三种处理异常值的办法

**缩小 block**：让异常值只影响一小组元素。代价是更多 scale、更多读取和更复杂
的 kernel。

**异常路径保留高精度**：把敏感 channel、layer 或 activation 置于 FP8/BF16，
其余部分使用 FP4。代价是混合精度配置、分支和发布复杂度。

**调整校准和缩放**：使用 clipping、percentile scale 或 outlier-aware calibration，
在极端值和普通值之间做折中。过度 clipping 可能损失真实的大值，不能只看校准集
上的平均误差。

没有一种处理对所有模型都最好。选择应由量化对象、任务、kernel 和显存预算共同
决定。

## 4. 量化对象不同，风险传播方式不同

### 4.1 权重 FP4

权重量化影响常驻显存和矩阵乘输入。错误会在每个请求中重复出现，但权重分布在
校准后相对稳定，适合做离线重构、敏感层分析和模型级回归。

权重 FP4 的收益通常首先体现在：

- 模型文件和常驻显存更小；
- 权重读取带宽可能降低；
- 单副本能承载更多 KV 或 active sequence；
- 同一 GPU 预算可以部署更多副本。

它不自动意味着端到端 latency 按相同比例下降。若 kernel 需要先解包到 FP16，
或者大量算子仍以高精度执行，payload 的节省可能主要转化为容量，而不是速度。

### 4.2 激活 FP4 或 FP8

激活随输入和 batch 变化，异常值分布比权重更难固定。激活量化需要更关注 calibration
数据、动态 range、token 类型和算子融合。相同的权重格式，在不同 batch、不同
prompt 长度、不同多模态 token 下，激活路径可能走不同的数值范围。

激活低精度的风险包括：

1. 某些 channel 饱和或下溢；
2. layernorm、attention score 或 router 的局部误差扩大；
3. 不同 shape 选择不同 kernel，导致性能和数值路径不一致；
4. 部分算子不支持低精度而回退，形成隐藏的性能长尾。

### 4.3 KV Cache 低精度

KV 量化影响的是历史状态。权重误差通常在每个请求开始时固定，KV 误差则会随着
上下文长度、重用和 decode 累积影响远距离依赖。长文档检索、工具历史、代码上下文
和引用任务都可能比普通短聊天更敏感。

KV 的粗略显存公式为：

~~~math
M_{\mathrm{kv}}
\approx
2N_{\mathrm{layer}}T_{\mathrm{active}}H_{\mathrm{kv}}D_{\mathrm{head}}b
~~~

将 `b` 从 2 字节降到 1 字节可能减少状态存储，但真实收益还受 scale、block、
对齐、读写和重建开销影响。KV 低精度必须单独测量长上下文、prefix reuse、抢占
重算和多轮对话，不能从 FP4 权重的结果外推。

### 4.4 累加和 logits 路径

矩阵输入使用低精度，不代表乘加结果也用低精度。高精度累加可以降低误差，但会
增加寄存器、带宽或 kernel 约束；低精度累加可能带来更高吞吐，却对 attention、
归一化、输出 logits 和结构化 token 更敏感。

一个完整 manifest 至少要分别记录 weight、activation、KV、scale、accumulation
和 logits dtype。`precision=fp4` 只能作为人类标签，不能作为机器配置的全部内容。

## 5. 从格式到 kernel：为什么理论压缩率不等于端到端加速

### 5.1 四层性能证据

低比特格式的性能应该分四层测量：

1. **存储层**：权重 payload、scale、metadata 和 checkpoint 实际字节数。
2. **kernel 层**：给定矩阵 shape、block layout、warm-up 和同步方式的矩阵乘时间。
3. **engine 层**：TTFT、TPOT、batch、并发、峰值显存、加载时间和 P99。
4. **任务层**：代码、数学、长上下文、结构化输出、RAG、工具和多模态质量。

只有第一层变好，说明压缩有效；第一、二层变好，说明 kernel 能消费；前三层变好，
说明 serving 受益；四层都没有回归，才有资格讨论生产采用。

### 5.2 解包和 scale 读取的隐性成本

如果硬件没有原生 FP4 matrix path，runtime 可能先把 payload 解包为 FP8、BF16 或
FP16，再执行矩阵乘。此时显存文件可能变小，但每次请求都支付了解包、scale 读取
和转换成本。即使有原生路径，不匹配的矩阵 shape、stride 或 group layout 也可能
触发 fallback kernel。

性能报告应记录实际执行路径：

~~~text
format: NVFP4 or MXFP4 profile
payload dtype: FP4
scale dtype: ...
accumulate dtype: ...
kernel: native / unpacked / fallback
hardware: GPU model and memory
runtime: engine and revision
~~~

没有 `kernel` 和 `runtime` 字段的“速度提升”很难复现。

### 5.3 block 大小的两面性

block 小，scale 更贴合局部分布，但 metadata、scale load、索引和 kernel 工作更多；
block 大，读取和布局更简单，但异常值影响范围更大。不存在脱离矩阵 shape、cache
行为和任务分布的“最佳 block size”。

工程上可以先做候选矩阵：

| block | 关注的收益 | 可能的代价 |
| --- | --- | --- |
| 小 | 局部误差和 outlier 影响较小 | scale/metadata 多，kernel 复杂 |
| 中 | 压缩和误差的折中 | 需要硬件 profile 配合 |
| 大 | 元数据少，布局简单 | 对异常值敏感，局部精度弱 |

最终结果必须由目标模型和硬件压测决定，不能从表格直接选一个数字。

## 6. PTQ、QAT 与校准集

### 6.1 PTQ 为什么便宜，为什么会失败

Post-training quantization 在训练完成后估计 scale、选择码本或做权重重构，不需要
重新训练全部模型，适合快速生成部署 artifact。它的风险是校准数据没有覆盖真实
输入，或敏感层的误差没有被模型训练过程适应。

PTQ 结果至少应保存：

- 模型和 tokenizer revision；
- calibration 数据版本和抽样规则；
- 每个对象的 scale、block 和 dtype；
- excluded modules 和 fallback modules；
- 量化工具、kernel 和 engine revision；
- 评测 harness、随机种子和任务切片。

只保存 `quantized.safetensors`，无法说明 scale 如何得到，也无法在质量退化时定位
是校准、格式、layout 还是 kernel 的问题。

### 6.2 QAT 解决的是什么

Quantization-aware training 在训练或微调过程中模拟量化误差，让模型参数有机会
适应低精度路径。它通常需要更多训练成本、稳定的 fake quant 实现和与部署 kernel
一致的模拟；如果训练时模拟的格式、scale 或累加路径与线上不同，QAT 的保证仍然
不成立。

对大模型来说，混合精度和少量敏感层训练可能比全模型 QAT 更现实，但这会增加
artifact 管理和验证矩阵。选择 PTQ、QAT 或混合方式时，应比较质量收益是否值得
额外训练、校准和运维成本。

### 6.3 校准集为什么不能只追求平均分

校准集应覆盖普通文本、代码、数学、数字、结构化 JSON、长上下文、多语言、工具
schema 和多模态 placeholder。若只用短自然语言，scale 可能对数字、代码缩进或
长距离 token 分布不合适。

校准和测试要隔离，避免通过重复调参记住评测样本。除了任务结果，还可以记录每层
的饱和比例、零值比例、scale 分布、最大误差和异常 channel，帮助解释质量变化。

## 7. 评估：平均质量接近不等于可以上线

### 7.1 任务切片

FP4 评估至少应覆盖：

| 切片 | 要观察的风险 |
| --- | --- |
| 普通聊天 | 流畅性、拒答和一般知识质量 |
| 代码 | 语法、可执行性、格式和边界条件 |
| 数学/数字 | 精确计算、符号和结构化数值 |
| 长上下文 | 远距离检索、引用和多轮状态 |
| RAG | 召回、citation support 和 unsupported claim |
| 工具 | schema、参数、授权和副作用 |
| 多模态 | OCR、视觉 token、音频/视频特征和融合 |
| 安全 | 漏放、误拦、注入、隐私和拒答 |

平均 perplexity 或平均 benchmark 分数只能描述一部分。高风险工具、长上下文和
少数语言样本可能只占很小比例，却决定事故严重度。

### 7.2 线上性能切片

每种精度配置还要按短/长输入、短/长输出、cache hit/miss、batch、并发和 GPU 测：

- 峰值显存和 allocator 碎片；
- model load 和 warm-up 时间；
- TTFT、TPOT、P95/P99；
- input/output tokens/s；
- KV 使用率、抢占和重算；
- fallback 比例和慢路径比例；
- 错误、超时、取消和流式中断。

如果低精度平均吞吐更高，却让长请求 P99、fallback 或工具 JSON 失败增加，不能
用平均吞吐覆盖协议失败。

### 7.3 接受率和推测解码

如果低精度 draft model 或 speculative path 与 target model 搭配，量化还可能改变
候选 token 的接受率。可以把接受率写成：

~~~math
\alpha
=
\frac{\text{accepted draft tokens}}
{\text{proposed draft tokens}}
~~~

低精度 draft 可能更快，却产生更多不被 target 接受的 token；target 验证成本和
额外调度会抵消收益。因此 FP4 量化后重新测 speculative decoding 的 acceptance、
TPOT、完整生成质量和失败切片，不能沿用 BF16 draft 的旧数据。

## 8. 一个异常值和存储账本的工作例子

### 8.1 异常值例子

考虑 16 个元素组成的 block，其中 15 个值在 `[-1,1]`，另一个值为 `20`。如果
scale 使用最大绝对值，普通值被除以 20，落在 `[-0.05,0.05]`。把 block 拆成
两个 8 元素 block，或把异常 channel 保留为更高精度，通常能给普通值更多有效格点，
但会增加 scale 和 kernel 操作。

这个例子不是某个 FP4 码表的完整模拟，而是帮助读者理解 block scale 的方向性。
真正的误差还依赖 payload codebook、rounding、scale dtype、累加和模型层的位置。

### 8.2 存储例子

设线性层有 `N=4096×4096` 个权重，FP16 每元素 2 字节，FP4 payload 每元素 0.5
字节；每 128 个权重共享一个 2 字节 scale，并有 1 字节 metadata，则：

~~~math
M_{\mathrm{fp4}}
=
N\times0.5
+\frac{N}{128}\times(2+1)
~~~

`N=16,777,216` 时，payload 约 8 MiB，scale 和 metadata 约 0.375 MiB，总量约
8.375 MiB；FP16 权重约 32 MiB。压缩比约为 `3.82`，不是简单把 2 字节除以 0.5
就声称有 4 倍真实压缩。

如果 runtime 还需要 FP16 workspace，峰值显存要再加解包缓冲、激活、通信副本和
allocator 余量。容量规划使用峰值，不使用模型文件大小替代运行时账本。

## 9. 一个可运行的 block 量化教学示例

下面的代码使用一个明确声明的教学 codebook，演示 block size、scale metadata、
异常值和存储估算的关系。它不是 MXFP4 或 NVFP4 的硬件实现，也不能用来证明任何
GPU 的 kernel 性能；它的价值是让读者看到“payload、scale 和 metadata 必须一起算”。

~~~python
from pprint import pprint


TEACHING_CODEBOOK = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0)


def quantize_blocks(values, block_size, codebook=TEACHING_CODEBOOK):
    reconstructed = []
    scales = []
    for start in range(0, len(values), block_size):
        block = values[start : start + block_size]
        if not block:
            continue
        scale = max(abs(value) for value in block) or 1.0
        scales.append(scale)
        for value in block:
            normalized = value / scale
            code = min(codebook, key=lambda candidate: abs(candidate - normalized))
            reconstructed.append(scale * code)
    return reconstructed, scales


def mse(left, right):
    return sum((a - b) ** 2 for a, b in zip(left, right)) / len(left)


def storage_bytes(element_count, block_size, payload_bits=4, scale_bytes=2, meta_bytes=1):
    block_count = (element_count + block_size - 1) // block_size
    return element_count * payload_bits / 8 + block_count * (scale_bytes + meta_bytes)


values = [0.9, 0.8, 0.02, -0.01, 20.0, 0.7, -0.6, 0.04]
results = {}
for block_size in (8, 4):
    restored, scales = quantize_blocks(values, block_size)
    results[block_size] = {
        "scales": scales,
        "mse": round(mse(values, restored), 5),
        "storage_bytes": storage_bytes(len(values), block_size),
        "restored": restored,
    }

pprint(results, sort_dicts=False)
~~~

这段代码的教学 codebook 只有 8 个值，故意没有声称自己是某个生产 FP4 码表。实际
运行输出如下：

~~~text
{8: {'scales': [20.0],
     'mse': 0.28776,
     'storage_bytes': 7.0,
     'restored': [0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0]},
  4: {'scales': [0.9, 20.0],
     'mse': 0.10776,
     'storage_bytes': 10.0,
     'restored': [0.9, 0.9, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0]}}
~~~

block 从 8 缩小到 4 后，异常值只影响后一个 block，MSE 下降，但 scale 和 metadata
数量增加。这个示例没有模拟真正 E2M1、E4M3、E8M0、Tensor Core 或 FP4 packing，
所以它只能说明 scale 粒度的机制，不能作为格式兼容性或速度报告。

## 10. 量化 artifact 和回退策略

### 10.1 manifest 应记录什么

一个可以复现的量化 artifact 至少要记录：

~~~text
model_revision
tokenizer_revision
format                 FP4 / MXFP4 / NVFP4 profile
payload_dtype
scale_dtype
block_size
accumulate_dtype
quantized_modules
excluded_modules
kv_dtype
calibration_revision
kernel_revision
engine_revision
tested_hardware
fallback_modules
~~~

`format` 只是人类可读名称，真正加载需要其余字段。回滚时必须同时恢复权重、scale、
kernel、engine 和服务端 dtype；只替换权重文件，可能导致 scale 被错误解释、算子
走慢路径，或者健康检查通过但输出质量异常。

### 10.2 为什么需要高精度回退

输出 head、router、位置相关投影、数字敏感层和某些 attention projection 可能对低
精度更敏感。可以保留 BF16/FP8、按层回退、按任务回退，或在质量和协议异常时把
请求路由到高精度副本。

回退条件应来自可观察信号：结构化格式失败、citation support 下降、工具参数
错误、长上下文切片退化、kernel 不支持、峰值显存异常或 P99 进入慢路径。回退本身
增加成本，所以要在 trace 中记录触发原因和回退后的任务结果。

### 10.3 Fallback 不能静默发生

如果目标 GPU 不支持某个 FP4 算子，runtime 可能回退到 FP8/BF16；这对功能是好事，
对性能报告却是关键信息。服务应该暴露实际 kernel、fallback 比例、加载时间和 dtype
路径。否则模型文件名写着 NVFP4，用户却实际得到一条混合慢路径，最终只能从 P99
异常中猜原因。

## 11. 四位量化部署的验证矩阵

### 11.1 存储、kernel、engine、任务四层

候选配置可以排成这样的矩阵：

| 配置 | 权重 | 激活 | KV | 累加 | 适合回答的问题 |
| --- | --- | --- | --- | --- | --- |
| A | BF16 | BF16 | BF16 | BF16 | 高精度基线是什么 |
| B | FP4 | BF16 | BF16 | BF16 | 权重压缩带来什么收益 |
| C | FP4 | FP8 | BF16 | BF16 | 激活量化是否改变性能/质量 |
| D | FP4 | FP8 | FP8 | BF16 | KV 压缩对长上下文有什么影响 |
| E | FP4 | FP8 | FP8 | 低精度 | 端到端低精度的风险在哪里 |

每行都应固定模型 revision、prompt、tokenizer、硬件、runtime、batch、输入输出
分布和采样参数，记录显存、加载、TTFT、TPOT、P99、吞吐、格式合法率、质量切片、
安全和 fallback。这样才知道收益来自哪一种量化对象。

### 11.2 哪些结论可以写，哪些不能写

| 已观察到的证据 | 可以写出的结论 | 不能直接推出 |
| --- | --- | --- |
| 文件字节数下降 | 存储/下载成本可能下降 | 端到端更快 |
| 目标 shape 的 kernel 变快 | 该 shape 的算子更快 | 所有 batch 都更快 |
| 平均质量接近基线 | 该评测集平均结果接近 | 长上下文和工具无损 |
| 峰值显存下降 | 该 workload 留出更多显存 | 并发一定翻倍 |
| fallback 比例为零 | 测试覆盖的路径未回退 | 未覆盖 shape 永不回退 |

工程判断必须保持在证据支持的范围内。尤其是闭源硬件格式和未来 runtime，官方
文档可以说明“支持某路径”，却不能替代目标任务的质量和性能测量。

## 12. 常见误区与排查路径

### 12.1 把四位格式混为一谈

看到 `FP4`、`MXFP4` 或 `NVFP4` 时，先拆 payload、scale、block、累加和 kernel。
如果资料只给了文件后缀，没有给出这些字段，结论应降低确定性。

### 12.2 只量化权重却宣称整条服务低精度

权重、激活、KV、累加和输出 logits 可能使用不同 dtype。部署报告要逐项列出，否则
读者会误以为低精度覆盖了整个数据流。

### 12.3 只看模型文件大小

解包 workspace、scale、通信、激活、KV 和 allocator 碎片都会影响峰值显存。容量
规划要测运行时峰值和并发，而不是拿 checkpoint 大小除以 GPU 显存。

### 12.4 只用短自然语言校准

短文本不能覆盖代码、数字、JSON、长上下文、多语言、工具和多模态 placeholder。
校准集应有版本、抽样规则和与测试集的隔离。

### 12.5 软件仿真和硬件 kernel 不一致

教学 codebook 或 Python 量化器可以帮助理解误差，但不能证明硬件 packing、scale
layout、累加和吞吐。软件模拟应明确标注“教学/离线近似”。

### 12.6 不检查 fallback

不支持的 shape 或算子可能静默回退到更高精度。监控实际 kernel、fallback 比例和
慢路径 P99，才能解释为什么文件更小但服务没变快。

### 12.7 用平均分掩盖协议失败

工具 JSON、引用支持、数字和安全样本往往比平均聊天分更重要。关键 slice 出现
退化时，应保留高精度层或回退，而不是用总体平均分抵消。

## 13. 资料与证据边界

本章中的 block scale、误差和存储公式是理解机制的教学模型，不是具体 GPU 格式
的完整规范。格式码表、scale 编码、block size、累加路径和 kernel 支持必须回到
对应版本的官方文档、源码或模型 artifact 核对。

资料的可信度和用途可以分开理解：

1. NVIDIA TensorRT-LLM precision/quantization 文档和 TensorRT Model Optimizer
   仓库用于确认 NVIDIA 软件栈公开的精度、校准和量化入口；不能直接证明目标模型
   在任意 GPU 上都走同一 kernel。
2. Hugging Face FP4/fine-grained FP4 文档用于确认 Transformers 生态中的配置和
   加载接口；接口存在不等于所有 serving engine 都支持。
3. OCP microscaling 规范用于理解 MX 类格式的标准化背景；具体 runtime profile
   仍需核对 block、scale 和布局。
4. GPTQ、AWQ、SmoothQuant 和 FP8 论文用于理解 PTQ、activation outlier、权重重构
   和低精度训练/推理的研究谱系；论文 benchmark 不能替代生产回放。
5. 本章的 Python 示例使用教学 codebook，不实现 MXFP4 或 NVFP4 的真实硬件路径，
   也不提供任何性能承诺。

延伸阅读：

1. [TensorRT-LLM Precision](https://nvidia.github.io/TensorRT-LLM/reference/precision.html)：NVIDIA TensorRT-LLM 当前精度类型入口。
2. [TensorRT-LLM Quantization](https://nvidia.github.io/TensorRT-LLM/features/quantization.html)：量化、校准和 engine 相关官方资料。
3. [Hugging Face FP4 Quantization](https://huggingface.co/docs/transformers/en/quantization/fp4)：Transformers FP4 配置和使用边界。
4. [Hugging Face Fine-Grained FP4](https://huggingface.co/docs/transformers/en/quantization/finegrained_fp4)：细粒度 FP4 和 scale 组织入口。
5. [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)：NVIDIA 官方模型优化工具仓库；版本和硬件支持需随 release 核对。
6. [OCP Microscaling Formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)：MX 类 microscaling 格式的规范入口；访问策略和版本以 OCP 当前页面为准。
7. [GPTQ](https://arxiv.org/abs/2210.17323)：基于近似二阶信息的后训练权重量化研究。
8. [AWQ](https://arxiv.org/abs/2306.00978)：activation-aware weight quantization 研究入口。
9. [SmoothQuant](https://arxiv.org/abs/2211.10438)：通过平滑激活异常值进行 W8A8 量化的研究入口。
10. [FP8-LM](https://arxiv.org/abs/2310.18313)：低精度浮点训练和推理的研究背景；不等同于 FP4 部署证明。

## 14. 本章小结

FP4、MXFP4 和 NVFP4 的共同点是降低低精度 payload 的存储或计算成本，差异则
藏在码表、scale、block、metadata、累加、kernel、硬件和 runtime 版本中。

1. FP4 是位宽/浮点编码层面的概念，不是一个唯一实现。
2. MXFP4 要重点理解 microscaling、block scale 和 metadata；block 越细不一定越快。
3. NVFP4 要重点核对硬件、kernel、engine、校准和 fallback；文件后缀不能证明实际路径。
4. 权重、激活、KV、累加和 logits 是不同量化对象，风险传播方式不同。
5. 异常值会让共享 scale 消耗普通值的有效精度，校准集和敏感层策略决定质量边界。
6. 端到端部署需要同时通过存储、kernel、engine 和任务四层证据，理论压缩率不等于
   端到端加速或成本下降。
7. manifest 要记录格式、scale、block、dtype、校准、kernel、engine、硬件和回退，
   量化才可复现、可诊断、可回滚。

当读者看到“FP4 显存减少、速度提升、质量接近 BF16”时，能够继续追问分母、硬件、
block、scale、累加、fallback 和任务切片，才真正理解了四位浮点的工程含义。
