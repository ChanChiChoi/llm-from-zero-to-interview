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
常见均匀 INT4 在反量化后是等间距格点；E2M1 则在小幅值区间放得更密、幅值增大后
间距变宽。它的收益取决于数据分布是否适合这组格点，不能仅凭“浮点”二字推导出
它比 INT4 更准确。

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

OCP MX 规范中的 E2M1 将 4 bit 分成 1 个符号位、2 个指数位和 1 个尾数位；它能表示
的非负幅值为 `0、0.5、1、1.5、2、3、4、6`，负值有对应幅值，最大幅值为 6，
没有 Inf 和 NaN 编码。一个量化值可抽象写成 `x̂ᵢ = s_g cᵢ`：`cᵢ` 从 E2M1
码本中选择，`s_g` 是该组的 scale。码本决定组内能选哪些数，scale 决定这组数落在
原张量的哪个范围；两者的舍入误差会叠加。

所以，阅读一份模型卡或 runtime 配置时，不能只看 `fp4` 这个字符串。至少要继续
追问：

- payload 的实际码表是什么；
- scale 是 per-tensor、per-channel、per-group 还是 per-block；
- scale 本身是什么 dtype，放在什么内存布局；
- 矩阵乘是否直接消费 payload，还是先解包到更高精度；
- 哪些层、哪些 shape 和哪些 GPU 走原生路径；
- activation、KV 和累加是否仍然保持更高精度。

### 2.2 MXFP4：把 scale 组织也纳入格式

OCP《Microscaling Formats (MX) v1.0》把 MXFP4 定义为 E2M1 元素、每 32 个元素共享
一个 8 bit E8M0 scale。E8M0 只有指数位、没有尾数位或符号位，因此有限 scale 是
2 的整数次幂。相比能够表示非 2 的整数次幂的 FP8 scale，这让尺度编码更简单，
却也会把 scale 本身的可选位置限制在更稀疏的集合里。OCP 将它定义成一种向量格式：
32 个 4 bit 元素与一个 8 bit scale 合起来为 `32×4+8=136` bit，即忽略补齐和布局
开销时平均 `4.25 bit/element`。定义以[OCP MX v1.0 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
为准。

microscaling 的直觉是：把一个大 tensor 切成小组，每组用自己的动态范围。与整个
tensor 共享一个 scale 相比，局部 scale 减少了某个大值支配所有小值的机会；代价是
每组都要存取 scale，矩阵乘 kernel 还要按约定的位置读取并广播它。OCP 规范定义了
数据类型、block size 和 scale 类型，但没有规定 scale 与 payload 必须如何物理排布；
具体 checkpoint、packing 和 kernel layout 仍由实现决定。

### 2.3 NVFP4：硬件路径优先于名字

NVIDIA Transformer Engine 2.20.2 的 [NVFP4 文档](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)
给出一种分层表示：E2M1 元素乘以
每 16 个连续元素共享的 E4M3 block scale，再乘以覆盖整个 tensor 的 FP32 global
scale：

~~~math
\hat{x}_i = c_i s_g s_{\mathrm{global}},
\qquad c_i\in\mathrm{E2M1},\quad i\in B_g,\quad |B_g|=16
~~~

E4M3 scale 的尾数位让它能表达非 2 的整数次幂的尺度；global scale 则把整张量先
映射到 E4M3 与 E2M1 可覆盖的范围。这个设计增加了 scale 精度和每组 scale 数量，
也需要保存 global scale。仅按一维分组计算，忽略 padding 与布局时为
`4 + 8/16 + 32/N = 4.5 + 32/N bit/element`；这里 `N` 是 tensor 元素数，`32/N`
来自一个 FP32 global scale 的摊销。

NVFP4 同样不等于一套固定不变的权重 recipe。Transformer Engine 2.20.2 文档说明，
其训练 recipe 默认对权重使用 16×16 的二维 scaling，对激活和梯度使用一维的
16-element scaling；其他工具和部署 artifact 可能采用不同 layout。后文的
`4.5 + 32/N` 只计算一维分组的 scale 账本，不可直接当作所有 NVFP4 checkpoint 的
文件大小。

还要区分“格式 recipe”和“量化图接口”。[TensorRT 10.x 的量化类型文档](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)
在显式 Q/DQ 量化定义中，把 NVFP4 的 block scale 参数写成 FP16、FP32 或 BF16；这与
Transformer Engine 对紧凑 NVFP4 表示的 E4M3 block scale + FP32 global scale 描述
处于不同接口层次。不能据 Q/DQ 图中 scale 张量的 dtype 推断 checkpoint 一定按同一
dtype 存储，也不能仅凭两份文档断言导出时的转换细节；应检查量化 artifact 和构建后的
engine。

对部署工程师而言，NVFP4 最重要的问题不是“它是不是比 MXFP4 更先进”，而是：

1. 目标 GPU 是否有原生指令或高效 kernel；
2. 目标 TensorRT-LLM/runtime 版本是否支持目标模型形状；
3. scale 和 metadata 是否能被 engine 正确加载；
4. 不支持的层是否会回退到 FP8、FP16 或慢速解包路径；
5. 回退后显存、TTFT、TPOT 和质量是否仍满足服务目标。

因此，MXFP4 更适合从格式和微缩放规则讨论，NVFP4 更需要从硬件、kernel、engine
和版本契约讨论。两者都不能仅靠文件名互换。

### 2.4 一张比较表应该写什么

| 项目 | E2M1 payload | OCP MXFP4 v1.0 | NVFP4 一维分组 |
| --- | --- | --- | --- |
| 元素码本 | 非负幅值 `0、0.5、1、1.5、2、3、4、6` 及其负值 | E2M1 | E2M1 |
| 局部 scale | 未由“FP4”泛称规定 | 每 32 个元素一个 E8M0（8 bit） | 每 16 个元素一个 E4M3（8 bit） |
| 全局 scale | 未由“FP4”泛称规定 | OCP MXFP4 定义中没有 | 每 tensor 一个 FP32 |
| 名义存储 | 仅 payload 为 4 bit | `4+8/32 = 4.25 bit/element` | `4+8/16+32/N bit/element` |
| 关键差别 | 只描述标量码本 | E8M0 scale 只有指数、scale 是 2 的幂 | E4M3 block scale 加 FP32 global scale |
| 仍需查证 | scale、packing、累加和 kernel | 实际 scale 布局、padding、kernel | 目标 recipe 的 1D/2D scaling、布局和 kernel |

表中的存储位数是便于预算的算术，不是 checkpoint 格式保证。NVFP4 的二维权重
scaling 会改变 scale 布局；对齐、padding、rowwise/columnwise 副本和未量化模块也会
改变实际字节数。比较时要先固定 recipe，再算同一层次的对象；不能拿一种格式的
payload bit 数去和另一种格式的硬件吞吐比较。

### 2.5 从支持矩阵读出真实边界

支持矩阵是部署决策中很有价值、但也很容易被误读的一类资料。以 NVIDIA Model
Optimizer 在 2026-10-27 访问的 [Linux 支持矩阵](https://nvidia.github.io/Model-Optimizer/guides/0_support_matrix.html)为例，FP4 被列为 per-block 的 FP4
权重与激活、Blackwell 及以后 GPU 的路径，部署栏列出 TensorRT 与 TensorRT-LLM。
这些信息可以证明该文档快照公开声明存在一条支持路径，不能证明任意模型、任意矩阵
形状或任意 KV 配置都能走同一条原生 kernel。该矩阵是滚动更新页面，应在选定软件
release 后重新核对。

读矩阵时应把三种问题分开：

1. **格式是否存在**：文档是否定义了 payload、scale、block 和累加语义。
2. **软件是否接入**：当前 Model Optimizer、TensorRT-LLM 或 Transformers 版本是否
   能导出、加载和构建目标 artifact。
3. **硬件路径是否可用**：目标 GPU、驱动、engine、模型 shape 和 batch 是否命中
   原生 kernel，还是在某些层回退到 FP8/BF16。

例如，支持矩阵中的“FP4”与模型文件名中的“NVFP4”不能简单互换：前者可能是
量化能力类别，后者还隐含厂商 profile、校准格式和 engine 约束。相反，一个模型
仓库提供了 NVFP4 checkpoint，也不能单凭文件存在推导出当前服务的 KV、视觉编码器、
MoE router 或输出头都使用 FP4。发布前应把矩阵中的声明转换成可执行的探针：加载
artifact，打印实际 dtype 和 kernel，运行覆盖目标 shape 的 warm-up，再测量 fallback
比例和任务切片。

本节的结论有时间边界。官方矩阵会随驱动、runtime 和模型支持更新；书中记录的
硬件范围应当带版本和访问日期，不能写成永远有效的兼容性承诺。

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
用 E2M1 码本可把这个结果算得更具体：`0.9/20=0.045`，离它最近的 E2M1 值是 0，
所以这个元素重构为 0、绝对误差为 0.9；若它单独使用 `s=0.9`，归一化值为 1，
则可重构回 0.9。这个计算只说明共享 scale 的影响方向，尚未加入 MX 的 E8M0
scale 舍入或 NVFP4 的 E4M3/global scale。

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

另一个可调变量是 scale 格式本身。MXFP4 的 E8M0 scale 以 2 的幂为刻度；NVFP4
一维 recipe 的 E4M3 scale 可在相邻的 2 的幂之间取值，但需要更多 block scale 和
tensor-level global scale。因而比较两种格式时，若模型质量不同，不能只归因于
E2M1 payload：scale 精度、block size 和量化算法都同时变化。

没有一种处理对所有模型都最好。选择应由量化对象、任务、kernel 和显存预算共同
决定。

### 3.4 为什么适配 INT4 的方法不一定适配 MXFP4/NVFP4

一个自然的想法是先旋转权重或激活以削弱 outlier，再用最邻近格点量化。但旋转
会改变数值在每个 block 中的分布；block size 较小、scale 又量化到离散格式时，
“outlier 被摊平”不保证每个 block 的新 scale 更合适。[Egiazarian 等人的
MR-GPTQ 研究](https://arxiv.org/abs/2509.23202)分析了这个问题：在其标准
round-to-nearest (RTN) 设置下，旋转有时改善 MXFP4、却可能损害 NVFP4；作者据此把
Hadamard 旋转缩到 block 内，并与 GPTQ 的二阶误差补偿及格式专属 scale search 结合，
提出 Micro-Rotated-GPTQ。作者在 Llama 3、Qwen 3 和指定 zero-shot 任务上报告，
MR-GPTQ 能显著改善 MXFP4；论文摘要把主实验概括为恢复到距 NVFP4 “within 1–2%”。
这个摘要概括不能当作每个 benchmark 都有同样差距的保证：附录 I 在另一组
PlatinumBench GPTQ 变体对比中，报告最佳 NVFP4 recovery 为 96.6%，最佳 MXFP4 为
92.3%，平均差距超过 4 个百分点。两处结果对应不同评测切片与统计口径，说明量化
方法的收益需要连同任务、模型和汇总方法一起读。
配套 QuTLASS kernel 在 B200 上报告最高 3.6× layer-wise、2.2× end-to-end 加速；在
RTX 5090 上报告最高 6× layer-wise、4× end-to-end。它们是论文中特定模型、层、shape、
baseline 与测试配置下的最高值，不是所有 FP4 负载的预测倍率。

这并不和[统一量化评测](https://arxiv.org/abs/2507.17417)“对 INT4 有效的 rotation
在 MXFP4/NVFP4 上收益有限”的结论简单冲突：通用旋转与 block-aware 旋转、RTN 与
GPTQ 误差回填、模型和评测集都不是同一个实验条件。可迁移的结论是先诊断某一格式
上的 scale 与误差，再选匹配它的校准/补偿方法；不能把“旋转有效”或“GPTQ 有效”
写成跨格式保证。

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

NVIDIA 研究团队的 [NVFP4 预训练报告](https://arxiv.org/abs/2509.25149)说明，低精度
训练结果依赖一套 recipe，而不只依赖格式名：其 12B hybrid Mamba-Transformer 在
10T tokens 训练中结合二维权重 scaling、部分路径的 Random Hadamard Transform、
梯度 stochastic rounding 和少量 BF16 层；作者报告稳定训练阶段的 NVFP4 validation
loss 相对误差低于 1%，学习率衰减末段略高于 1.5%，大多数下游切片接近 FP8，MBPP+
有例外。该报告证明的是这套训练方法和模型，不能用来声称任意 NVFP4 checkpoint 的
PTQ 质量接近 FP8。

### 6.3 校准集为什么不能只追求平均分

校准集应覆盖普通文本、代码、数学、数字、结构化 JSON、长上下文、多语言、工具
schema 和多模态 placeholder。若只用短自然语言，scale 可能对数字、代码缩进或
长距离 token 分布不合适。

校准和测试要隔离，避免通过重复调参记住评测样本。除了任务结果，还可以记录每层
的饱和比例、零值比例、scale 分布、最大误差和异常 channel，帮助解释质量变化。

### 6.4 固定 scale、补偿误差与学习 scale 是不同方案

PTQ 通常先从权重或校准样本计算每组 scale，再固定 scale 量化；若 scale 估计偏离
目标请求分布，量化误差会留在权重中。另一条路线是用 GPTQ 一类方法按层补偿权重
重构误差。MR-GPTQ 的例子说明，低 bit 格式需要把误差补偿步骤与 E2M1 码本、block
scale 和变换方式一起设计，不能只更换量化器名称。

还可以把 block scale 作为可训练参数。对一组权重 `w`，fake quantization 可写为
`ŵ = s·Q(w/s)`；前向仍模拟 NVFP4 的低精度表示，反向通过 straight-through
estimator (STE) 近似穿过离散的 round 操作。scale-only learning 固定权重、只优化
scale；full parameter scale learning 则同时更新权重和 scale。训练完导出时把学到的
scale 合并进量化 artifact，推理图不需要额外算子，但优化过程仍要做训练/蒸馏计算，
所以它不是免训练的校准 PTQ。

NVIDIA Model Optimizer 团队 2026-09-16 的 [scale-learning 说明](https://nvidia.github.io/Model-Optimizer/announcements/20260916-scale-learning.html)
给出了一个有边界的示例：在 Nemotron 3.5 Lightning 的 `W4A4 routed experts +
W8A8 attention/Mamba cache` 配置中，PTQ 后用 scale-only QAD 能在部分切片恢复质量：

| 评测（准确率，%） | BF16 reference | PTQ | scale-only learning |
| --- | ---: | ---: | ---: |
| GPQA Diamond | 76.45 | 73.30 | 75.00 |
| SciCode | 32.51 | 33.42 | 35.42 |
| Tau3 Banking | 9.897 | 7.216 | 7.216 |

同一组结果既显示前两项的变化，也显示 Tau3 Banking 没有比 PTQ 恢复；它不能被概括
成每个任务都获益。数据是发布方对一个模型与配置的报告，不是 NVFP4 普遍增益。
Model Optimizer 当前说明还限定该实现只训练 NVFP4 per-block weight scales。

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

再按两个具体一维格式 profile 计算同一矩阵。设 `N=4096×4096=16,777,216`，不需
padding，FP16 权重为 32 MiB，FP4 payload 本身为 8 MiB：

| 账本 | scale 字节 | 总大小 | 相对 FP16 的理论压缩比 |
| --- | ---: | ---: | ---: |
| OCP MXFP4：每 32 个值一个 E8M0 | `N/32 = 524,288`（0.5 MiB） | 8.5 MiB | `32/8.5 ≈ 3.76×` |
| NVFP4 一维：每 16 个值一个 E4M3，另有 FP32 global scale | `N/16 + 4 = 1,048,580`（约 1 MiB） | 约 9 MiB | 约 `3.56×` |

这两个账本只比较 E2M1 payload 和上述 scale，不含对齐、转置副本、未量化层、
engine workspace 或 KV。特别是 Transformer Engine 的 NVFP4 二维权重 scale 与一维
激活 scale 是不同 layout；部署必须直接检查导出的 artifact，而不是把本表当作
所有 NVFP4 权重的文件大小。

## 9. 一个可运行的 block 量化教学示例

下面的代码使用 E2M1 的离散码值和未量化的 Python 浮点 block scale，演示 block size、scale metadata、
异常值和存储估算的关系。它不是 MXFP4 或 NVFP4 的硬件实现，也不能用来证明任何
GPU 的 kernel 性能；它的价值是让读者看到“payload、scale 和 metadata 必须一起算”。

~~~python
import math
from pprint import pprint


E2M1_VALUES = (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
               0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def quantize_blocks(values, block_size):
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not values:
        raise ValueError("values must not be empty")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("values must be finite")

    reconstructed = []
    scales = []
    for start in range(0, len(values), block_size):
        block = values[start : start + block_size]
        scale = max(abs(value) for value in block) or 1.0
        scales.append(scale)
        for value in block:
            normalized = value / scale
            code = min(E2M1_VALUES, key=lambda code_value: abs(code_value - normalized))
            reconstructed.append(scale * code)
    return reconstructed, scales


def mse(left, right):
    if not left or len(left) != len(right):
        raise ValueError("inputs must be non-empty and have equal length")
    return sum((a - b) ** 2 for a, b in zip(left, right)) / len(left)


def storage_bytes(element_count, block_size, payload_bits=4, scale_bytes=2, meta_bytes=1):
    if element_count < 0 or block_size <= 0:
        raise ValueError("element_count must be non-negative and block_size positive")
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

这段代码使用 E2M1 可表示值，但 `scale=max(abs(block))` 是未量化浮点数；存储函数
里的 2-byte scale 和 1-byte metadata 也是通用账本假设。因此它不能模拟 OCP 的
E8M0 scale 舍入、NVFP4 的 E4M3/global/2D scaling、物理 packing 或 Tensor Core
运算。实际运行输出如下：

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
数量增加。这里使用真正的 E2M1 可表示值，却没有模拟 E4M3/E8M0 scale 量化、
Tensor Core 或 FP4 packing，所以它只能说明 scale 粒度的机制，不能作为格式兼容性
或速度报告。

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

本章的误差分析和存储推演用于解释机制；OCP MXFP4 v1.0 才定义此处列出的 E2M1、
32-element block 和 E8M0 scale 组合。NVFP4 的 block、global scale、二维权重 recipe
和布局则按 Transformer Engine 页面所指版本说明。模型 artifact 还需要结合实际
量化工具、checkpoint 与 engine 检查，不能只凭格式名还原完整 kernel 行为。

资料的可信度和用途可以分开理解：

1. NVIDIA TensorRT-LLM precision/quantization 文档和 TensorRT Model Optimizer
   仓库用于确认 NVIDIA 软件栈公开的精度、校准和量化入口；不能直接证明目标模型
   在任意 GPU 上都走同一 kernel。
2. Hugging Face Transformers 量化总览和量化选择文档用于确认 Transformers 生态中
   的配置与加载边界；页面是否列出某格式不等于所有 serving engine 都支持。
3. [NVIDIA Transformer Engine NVFP4 文档](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)
   当前页面标为 2.20.2；其中 1D scaling、global scale 和 2D weight scaling 是该文档
   所指 recipe，不应自动推广到所有 NVFP4 artifact。
4. [NVIDIA Model Optimizer 支持矩阵](https://nvidia.github.io/Model-Optimizer/guides/0_support_matrix.html)
   于 2026-10-27 访问，列出 Linux FP4 per-block weights/activations 与 Blackwell+、
   TensorRT/TensorRT-LLM 部署路径；它是会滚动变化的软件兼容声明，不是目标 workload
   的 native-kernel 或性能实测。
5. [A Comprehensive Evaluation on Quantization Techniques for LLMs](https://arxiv.org/abs/2507.17417)
   对比粒度、scale precision、旋转和误差补偿；结论绑定作者模型与量化设置。
   [Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization](https://arxiv.org/abs/2509.23202)
   分析 MXFP4/NVFP4 与通用旋转/GPTQ 的适配问题，并提出 MR-GPTQ；其准确率与 kernel
   结果绑定论文任务、模型、B200/RTX 5090 实现和测量方法。
6. [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149)
   报告结合二维权重 scaling、Hadamard transform、stochastic rounding 和选择性高精度
   层的训练 recipe；它证明的是作者训练设置，不是通用 post-training inference PTQ。
7. [Scale Learning: Training NVFP4 Block Scales to Recover Accuracy](https://nvidia.github.io/Model-Optimizer/announcements/20260916-scale-learning.html)
   解释冻结权重只训 scale 与联合更新权重/scale 的区别；数字是 Model Optimizer 团队
   在 Nemotron 3.5 Lightning 配置上的报告，当前实现范围有限，不能代表所有模型。
8. GPTQ、AWQ、SmoothQuant 和 FP8 论文用于理解 PTQ、activation outlier、权重重构
   和低精度训练/推理的研究谱系；论文 benchmark 不能替代生产回放。
9. 本章 Python 示例使用 E2M1 码值，但使用未量化浮点 scale 和通用 metadata 账本，
   不实现 MXFP4/NVFP4 的完整格式或硬件路径，也不提供性能承诺。

延伸阅读：

1. [TensorRT-LLM Precision](https://nvidia.github.io/TensorRT-LLM/reference/precision.html)：NVIDIA TensorRT-LLM 当前精度类型入口。
2. [TensorRT-LLM Quantization](https://nvidia.github.io/TensorRT-LLM/features/quantization.html)：量化、校准和 engine 相关官方资料。
3. [Hugging Face Transformers Quantization Overview](https://huggingface.co/docs/transformers/en/quantization/overview)：Transformers 量化配置与后端边界。
4. [Hugging Face Selecting a Quantization Method](https://huggingface.co/docs/transformers/en/quantization/selecting)：按模型、硬件和任务选择量化方法的入口。
5. [NVIDIA Model Optimizer Support Matrix](https://nvidia.github.io/Model-Optimizer/guides/0_support_matrix.html)：当前公开格式、模型格式、部署后端和 GPU 范围；矩阵需随版本复核。
6. [NVIDIA Model Optimizer Quantization Guide](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html)：PTQ/QAT 和导出流程入口。
7. [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)：NVIDIA 官方模型优化工具仓库；版本和硬件支持需随 release 核对。
8. [OCP Microscaling Formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)：MX 类 microscaling 格式的规范入口；访问策略和版本以 OCP 当前页面为准。
9. [GPTQ](https://arxiv.org/abs/2210.17323)：基于近似二阶信息的后训练权重量化研究。
10. [AWQ](https://arxiv.org/abs/2306.00978)：activation-aware weight quantization 研究入口。
11. [SmoothQuant](https://arxiv.org/abs/2211.10438)：通过平滑激活异常值进行 W8A8 量化的研究入口。
12. [FP8-LM](https://arxiv.org/abs/2310.18313)：低精度浮点训练和推理的研究背景；不等同于 FP4 部署证明。

## 14. 本章小结

FP4、MXFP4 和 NVFP4 的共同点是降低低精度 payload 的存储或计算成本，差异则
藏在码表、scale、block、metadata、累加、kernel、硬件和 runtime 版本中。

1. FP4 是位宽/浮点编码层面的概念，不是一个唯一实现。
2. MXFP4 v1.0 的 E8M0 scale 每 32 个 E2M1 值共享；NVFP4 一维 recipe 的 E4M3 scale
   每 16 个值共享并带 FP32 global scale，NVFP4 二维权重 scaling 又有不同存储账本。
3. Block size、scale 精度和量化误差补偿共同影响质量；MR-GPTQ 与可学习 scale 展示
   格式专属 PTQ 和训练时恢复两条路线，收益均受论文/发布方设置约束。
4. NVFP4 要重点核对硬件、kernel、engine、校准和 fallback；文件后缀不能证明实际路径。
5. 权重、激活、KV、累加和 logits 是不同量化对象，风险传播方式不同。
6. 异常值会让共享 scale 消耗普通值的有效精度，校准集和敏感层策略决定质量边界。
7. 端到端部署需要同时通过存储、kernel、engine 和任务四层证据，理论压缩率不等于
   端到端加速或成本下降。
8. manifest 要记录格式、scale、block、dtype、校准、kernel、engine、硬件和回退，
   量化才可复现、可诊断、可回滚。

当读者看到“FP4 显存减少、速度提升、质量接近 BF16”时，能够继续追问分母、硬件、
block、scale、累加、fallback 和任务切片，才真正理解了四位浮点的工程含义。
