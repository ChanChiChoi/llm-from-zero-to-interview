# 第 17 章 Native INT4：原生量化权重为什么不等于部署时再量化

“这个模型是 INT4”是一句信息量很低的话。它可能只表示权重文件用 4 bit
保存，也可能表示权重经过 GPTQ 或 AWQ 等后训练量化，还可能表示模型发布方
已经为某种 GPU、某种 group size、某种 scale 布局和某个推理引擎准备了融合
kernel。它们都可能被宣传为 4 bit 模型，但启动路径、显存账本、速度和质量
并不相同。

本章把 native INT4 用作一个工程分类：量化后的 artifact、scale 和 metadata
布局、计算累加精度、目标硬件以及 runtime kernel 是协同设计和发布的。这个
称呼不是一个所有厂商都采用、字段完全统一的正式标准，所以读者遇到模型卡
中的 native、pre-quantized、W4A16、AWQ INT4 或 INT4 weight-only 时，仍然
需要回到具体 manifest 和执行路径核对。

对初学者来说，可以把它想成把一套大型书籍压缩成小盒子：盒子占地变小，只
说明存储更省；如果阅读器每次都要先把整套书还原成原尺寸，阅读速度未必更快；
如果压缩时还改变了页码索引，引用和跳转也可能出错。对专家来说，需要同时
追踪 payload bit width、group axis、scale/zero-point、pack order、GEMM
kernel、累加 dtype、硬件 capability、并行切分和模型协议。

因此，部署 INT4 的问题不是“4 bit 能不能加载”，而是下面这条链是否闭合：

~~~text
高精度 checkpoint
  -> 量化方法与校准数据
  -> packed weight artifact
  -> scale / metadata layout
  -> native kernel 与累加路径
  -> runtime / hardware
  -> 质量、延迟、容量和回滚证据
~~~

## 17.1 先把“原生量化”说清楚

### 17.1.1 四个容易混在一起的概念

“4 bit”描述的是数值的存储位宽，却没有说明量化发生在什么时候、量化了
哪些张量以及计算时是否仍然使用高精度。至少要把下面四个概念分开：

| 概念 | 它真正描述的内容 | 它没有自动保证的事情 |
| --- | --- | --- |
| 4 bit payload | 某类张量的存储编码使用 4 bit | 不保证激活、KV 或累加也是 4 bit |
| PTQ artifact | 已有高精度模型经过校准或离线算法产生量化文件 | 不保证目标 runtime 有对应快路径 |
| QAT artifact | 训练过程让模型适应量化噪声 | 不保证文件布局和目标 kernel 已配套 |
| native execution path | runtime 用目标硬件支持的低比特 kernel 执行 | 不保证质量达到所有任务的高精度基线 |

PTQ 是 post-training quantization，通常先得到 BF16 或 FP16 checkpoint，再
用代表性数据估计 scale，或者用 GPTQ、AWQ 等方法优化权重。QAT 是
quantization-aware training，训练时让模型在前向过程中感受到近似的量化误差。
两者都可能产生高质量的 INT4 权重，但“native”在本章中强调的是 artifact
和执行系统的配套关系，不是训练方法的同义词。

### 17.1.2 小白视角：为什么文件能打开仍然不够

加载器能读出张量，只能证明文件的字节布局至少满足加载器的最低要求。后面
还会发生三件事：

1. runtime 要根据 scale 把 packed value 解释成近似权重；
2. kernel 要把这种布局送入矩阵乘，并用正确 dtype 累加；
3. 模型协议要和这个权重版本的 config、tokenizer、template 以及 adapter
   保持一致。

任何一步错了，结果都可能是“程序没有报错，但答案悄悄变差”。例如 group
size 读错时，张量 shape 仍然正确，输出却会持续偏离；kernel 不支持某个
矩阵形状时，服务也许只是慢很多。

### 17.1.3 专家视角：native 是一组联合版本

可以把一个可部署 artifact 表示为：

~~~math
\mathcal{A}
=
(\mathcal{W}_{\mathrm{pack}},
\mathcal{S},
\mathcal{Z},
\mathcal{L},
d_{\mathrm{acc}},
k_{\mathrm{kernel}},
h_{\mathrm{hw}},
r_{\mathrm{runtime}})
~~~

其中 W_pack 是压缩权重，S 是 scale，Z 是 zero point 或其他偏移信息，L 是
布局和 pack 规则，d_acc 是累加 dtype，k_kernel 是执行 kernel，h_hw 是硬件
能力，r_runtime 是 runtime 版本。这个元组中的任一字段变化都可能改变结果
或性能。

所以本章使用“原生 INT4”时，读者应该把它理解成“有明确执行契约的 INT4
artifact”，而不是把它当成一个神奇的模型标签。

## 17.2 为什么先量化权重，以及显存账本如何计算

### 17.2.1 权重是固定成本，KV 是动态成本

模型权重在 worker 启动时通常加载一次。请求到来后，KV Cache、临时激活、
通信 buffer 和调度 workspace 才会随着请求数、上下文长度和 batch 变化。
因此权重 INT4 的第一收益往往是让模型能够放进更少的 GPU，或者把原来被
权重占用的空间让给 KV Cache，而不是让所有 decode kernel 自动提速。

设模型有 P 个参数，每个参数使用 b_w bit 存储，理想权重 payload 为：

~~~math
M_{\mathrm{weight,payload}}
=
\frac{P b_w}{8}
~~~

这里的 P 是参数数量，b_w 是每个参数的存储 bit 数，结果的单位是 byte。
BF16 取 b_w=16，INT4 取 b_w=4。理想情况下，INT4 payload 是 BF16 的
四分之一，但这个比例还没有计入 scale、zero point、对齐和未量化模块。

实际 worker 的峰值工作集可以先写成：

~~~math
M_{\mathrm{peak}}
=
M_{\mathrm{weight}}
+M_{\mathrm{KV}}
+M_{\mathrm{workspace}}
+M_{\mathrm{comm}}
+M_{\mathrm{runtime}}
~~~

M_weight 包含 payload、scale、metadata 和高精度保留层；M_KV 是活跃请求
的历史状态；M_workspace 是 GEMM、attention 和编译 kernel 的临时空间；
M_comm 是 tensor parallel 或 pipeline parallel 的通信 buffer；M_runtime
是框架和 allocator 的常驻开销。为了应对峰值抖动，还可以另记策略性预算：

~~~math
M_{\mathrm{budget}}
=
M_{\mathrm{peak}}
+M_{\mathrm{reserve}}
~~~

这样，实测峰值和人为保留的余量不会被混在一个数字里。

### 17.2.2 一个实际的数量级例子

假设 BF16 权重占 400 GiB。仅从 payload 看，INT4 约占：

~~~math
M_{\mathrm{INT4,payload}}
=
400\times\frac{4}{16}
=
100\ \mathrm{GiB}
~~~

如果 scale、zero point、对齐和未量化层共同占 15 GiB，INT4 权重相关空间
就是 115 GiB，而不是 100 GiB。假设另有 20 GiB KV、16 GiB workspace、
8 GiB 通信 buffer 和 6 GiB runtime 常驻空间，则两个版本的粗略账本为：

| 项目 | BF16 权重 | INT4 权重 |
| --- | ---: | ---: |
| 权重相关 | 400 GiB | 115 GiB |
| KV Cache | 20 GiB | 20 GiB |
| workspace | 16 GiB | 16 GiB |
| 通信 buffer | 8 GiB | 8 GiB |
| runtime 常驻 | 6 GiB | 6 GiB |
| 合计 | 450 GiB | 165 GiB |

这个例子说明，INT4 确实可能显著改变模型并行规模，但它没有把 KV、workspace
和通信 buffer 一起变成四分之一。若长上下文使 KV 从 20 GiB 增加到 100 GiB，
系统瓶颈就可能再次转移到 KV。

### 17.2.3 GPU 数量也不是简单除法

设单卡显存为 C_gpu，实际允许给 worker 使用的比例为 u，则仅从容量得到
的最低卡数可以写成：

~~~math
N_{\mathrm{raw}}
=
\left\lceil
\frac{M_{\mathrm{peak}}}{uC_{\mathrm{gpu}}}
\right\rceil
~~~

u 通常小于 1，因为要给 CUDA context、通信、碎片和峰值留空间；这里已经
用 u 表示了余量。如果团队已经把明确的 M_reserve 加进 M_budget，就应改用：

~~~math
N_{\mathrm{raw,budget}}
=
\left\lceil
\frac{M_{\mathrm{budget}}}{C_{\mathrm{gpu}}}
\right\rceil
~~~

两种口径只能选一种，不能同时给同一份账本加 headroom。N_raw 还没有考虑
tensor parallel 的可用 degree、节点拓扑、故障冗余、跨卡带宽和发布策略。
INT4 把 M_weight 变小，只改变这个分式的一个组成部分。

### 17.2.4 小白视角：为什么显存省了，速度可能没有翻倍

如果程序先读取 INT4，再把每个权重恢复成 BF16，矩阵乘仍然用 BF16，那么
计算器看到的仍是高精度矩阵。此时得到的主要是存储收益；解包还可能增加
额外工作。

如果 GPU 有支持这种 pack layout 的融合 kernel，权重可以在更接近矩阵乘的
位置解包，减少显存带宽压力，才有机会改善速度。但实际 decode 可能由 KV
读取、调度或网络通信限制，所以“权重缩小四倍”和“请求吞吐提高四倍”之间
没有直接推导关系。

## 17.3 INT4 的量化数学：scale、zero point 与分组

### 17.3.1 对称量化

对一组浮点权重 w_i，对称量化可以抽象成：

~~~math
q_i
=
\operatorname{clip}
\left(
\operatorname{round}\left(\frac{w_i}{s}\right),
q_{\min},
q_{\max}
\right),
\qquad
\hat{w}_i=sq_i
~~~

q_i 是存储的整数码，s 是 scale，q_min 和 q_max 是该格式允许的整数范围，
hat{w}_i 是反量化后的近似值。二进制补码有符号 INT4 的整数码范围是
[-8, 7]；但许多对称量化器为了让正负两侧幅值相同，会选择 [-7, 7]，把
-8 留空。两种说法并不矛盾：前者是编码可表示的范围，后者是量化器选择使用
的码本。实际 artifact 还可能使用非对称范围或特殊的零点，必须以量化配置
为准。

用四个数手算一次对称量化。取权重
[-0.9, -0.2, 0.3, 1.1]，使用 [-7, 7]，令最大绝对值映射到 7：

~~~math
s=\frac{1.1}{7}\approx 0.15714,
\qquad
q_i=\operatorname{round}\left(\frac{w_i}{s}\right)
=[-6,-1,2,7]
~~~

反量化结果约为 [-0.9429, -0.1571, 0.3143, 1.1000]。例如
0.3 / 0.15714 ≈ 1.91，四舍五入到 2，再乘回 scale 得到约 0.3143。
最大值没有被裁剪，但 -0.9 被表示成 -0.9429；这组误差来自有限码点间距，
不是整数范围溢出。若同一 group 中出现更大的异常值，scale 会变大，普通值
可选的相邻码点间距也会变粗；裁剪可以恢复普通值分辨率，却会让被裁剪的大值
产生有意的饱和误差。

最简单的 scale 估计是：

~~~math
s
=
\frac{\max_i |w_i|}{q_{\max}}
~~~

它能避免最大值超出范围，却会让少数 outlier 决定整个 group 的分辨率。实践
中也会使用分位数、裁剪或专门的异常值处理，代价是可能牺牲少数大值的精确度。

### 17.3.2 非对称量化

如果权重分布明显偏离零点，可以引入 zero point z：

~~~math
q_i
=
\operatorname{clip}
\left(
\operatorname{round}\left(\frac{w_i}{s}\right)+z,
q_{\min},
q_{\max}
\right),
\qquad
\hat{w}_i=s(q_i-z)
~~~

非对称量化能更好利用有限的整数范围，但每个 group 需要额外保存 z，kernel
也要处理偏移。一个模型使用 symmetric 还是 asymmetric，不能由“INT4”四个
字符推断。

### 17.3.3 group size 是精度和开销的旋钮

把连续的权重划分为大小为 G 的 group，每个 group 保存一个或多个 scale。
用理想化的存储模型表示，单个 group 的字节数近似为：

~~~math
B_{\mathrm{group}}
\approx
\left\lceil\frac{4G}{8}\right\rceil
+b_s
+b_z
+b_m
~~~

其中第一项是 INT4 payload，b_s 是 scale 字节数，b_z 是 zero point 字节数，
b_m 是对齐或其他 metadata。平均每个参数的字节数为：

~~~math
b_{\mathrm{param}}
\approx
\frac{B_{\mathrm{group}}}{G}
~~~

G 越大，scale 的平均开销越小，但同一个 scale 要覆盖更多数值，异常值会
影响更多普通权重；G 越小，局部适应性通常更好，但 scale 读取、metadata
和 kernel 索引的成本增加。常见的 32、64、128 等 group size 只是工程选项，
不是脱离模型和硬件的质量结论。

### 17.3.4 pack layout 也是格式的一部分

两个文件都可以把数值称为 INT4，但它们可能在一个 byte 的高低 nibble 顺序、
矩阵转置、group axis、scale 顺序、padding 和 shard 边界上不同。loader 如果
把高 nibble 当低 nibble，数值仍然落在合法的 4 bit 范围，模型却会得到几乎
不可解释的输出。

因此 manifest 不能只写 dtype=int4。至少要同时描述：

~~~text
signed_or_unsigned
pack_order
group_axis
group_size
scale_axis
scale_dtype
zero_point_dtype
padding_rule
shard_boundary
~~~

## 17.4 PTQ、QAT 和 native artifact 的关系

### 17.4.1 PTQ：先有模型，再适配量化

PTQ 的典型链路是：

1. 读取 BF16/FP16 checkpoint；
2. 用代表性样本运行校准，观察权重或激活分布；
3. 确定 scale、group size、保留高精度模块和量化对象；
4. 生成 packed artifact；
5. 用目标 runtime 做质量和性能回归。

它的优势是灵活。同一个高精度 checkpoint 可以为不同 GPU、不同 group size 或
不同 workload 生成多个版本。它的风险是校准数据、量化算法和目标 kernel
之间可能不匹配。

### 17.4.2 GPTQ：用误差传播考虑权重重要性

逐元素 round-to-nearest（RTN）只看某个权重离哪个量化格点最近，不看这一格
误差通过当前层输入后会不会显著改变输出。GPTQ 把目标改成校准输入上的层输出
重建误差。对线性层 Y=WX，令 X 是收集到的输入激活，W_q 是量化矩阵：

~~~math
\mathcal{L}(W_q)
=
\left\|(W-W_q)X\right\|_F^2,
\qquad
H\approx 2XX^\top+\lambda I
~~~

这里 H 是这个局部二次重建目标的近似 Hessian，反映输入特征之间的相关性；
它不是整个语言模型任务损失对所有参数的 Hessian。lambda I 是阻尼项，用来
避免校准样本不足或数值误差造成矩阵病态。

每次把一个权重坐标 w_j 量化成 q_j 后，量化误差
e_j=w_j-q_j 不会被丢在原地。GPTQ 根据逆 Hessian 把它分摊给尚未量化的坐标：

~~~math
w_F
\leftarrow
w_F-
e_j
\frac{(H_F^{-1})_{F,j}}{(H_F^{-1})_{j,j}}
~~~

F 表示尚未量化的输入通道集合。分母把被量化坐标的局部代价归一化，分子则
告诉算法哪些剩余坐标能以较小的重建损失吸收误差。一个二通道的教学例子：

~~~math
H^{-1}=
\begin{bmatrix}
1 & 0.8\\
0.8 & 1
\end{bmatrix},
\qquad
w_1=0.62,\quad q_1=0.42,\quad w_2=0.50
~~~

此时 e_1=0.20，补偿后的第二个权重为
0.50-0.20\times 0.8/1=0.34。若第二通道与第一通道在校准输入上的关系很弱，
逆 Hessian 的非对角项接近 0，算法就不会把太多误差分给它。这个例子只演示
局部更新，不意味着 GPTQ 能让每个样本的输出完全不变。

原论文把 OBQ 的逐权重贪心更新扩展到大模型：所有输出行共享同一列顺序，因为
输入激活导出的 Hessian 对这些行相同；每次按一块连续列处理，再统一更新剩余
权重，以提高 GPU 利用率；逆 Hessian 的 Cholesky 重写和对角阻尼则缓解大矩阵
重复更新中的数值不稳定。原论文的量化设定采用均匀、逐行、非对称量化，并非
“GPTQ 必然使用某一种 INT4 对称码本”。

证据要和实验条件一起读。原论文使用 128 段、每段 2048 token 的 C4 校准输入，
在单张 80GB A100 上量化模型；OPT-175B 的整模型量化用时约 4.2 小时。在
WikiText-2 上，[原论文](https://arxiv.org/html/2210.17323v2)报告 FP16 PPL
为 8.34、4 bit GPTQ 为 8.37、同设定的 RTN 为 10.54。它说明误差补偿在这组
模型和校准条件下优于直接舍入，不保证
任意指令、代码、多语种或目标业务任务都同样退化。GPTQ 仍是 weight-only：
它压低权重搬运字节，但不自动减少矩阵乘的乘加次数；量化质量和生成速度还
依赖实际 kernel。

### 17.4.3 AWQ：激活感知的等价通道缩放

AWQ 先问“哪些权重误差会被当前输入放大”。对线性层 Y=XW，权重矩阵 W 的
某个输入通道若经常乘上较大的激活，该通道的量化误差更容易影响输出。AWQ
用校准激活的逐通道平均幅值估计这种敏感性，而不是只按权重绝对值挑异常项。

如果 D_s 是输入通道的对角缩放矩阵，未量化时可以做等价变换：

~~~math
XW=(XD_s^{-1})(D_sW)
~~~

AWQ 把敏感通道对应的权重放大，再把激活反向缩小；浮点计算仍是同一个线性
映射，但被量化的 D_sW 中这些权重相对 group scale 更容易获得细粒度表示。
量化后两侧不再严格抵消，留下的误差由校准数据上的层输出误差来衡量。若缩放
只照顾显著通道，也可能放大同一 group 内普通通道的误差，所以 AWQ 不会简单
地把显著通道无限放大。

论文把搜索范围限制为 s=s_X^alpha，其中 s_X 是逐输入通道的平均激活幅值，
alpha 在 [0,1] 的小网格上搜索；目标是让量化层输出接近原层输出，并可再搜索
weight clipping。D_s^{-1} 通常可以折入前一个算子，避免推理时多出独立缩放
操作。部署时目标仍是整组低比特权重；“保留 1% 权重为 FP16”是 AWQ 论文
讨论重要权重时的动机和对照，不是 AWQ 主方法的必需执行路径。部分综述把
AWQ 简写成保留 1% 高精度权重，谈具体算法时应以[AWQ 原论文](https://arxiv.org/html/2306.00978v6)
的等价缩放方法为准。

在 [AWQ 原论文](https://arxiv.org/html/2306.00978v6)中，OPT-6.7B、INT3、
group size 128、WikiText-2 上 FP16 PPL 为 10.86、RTN 为 23.54、AWQ 为
11.39；这是算法质量证据，不是任意模型的保证。
作者还在 PubMed 与 Enron 两个数据切片之间做校准/评测分布互换，报告该设定下
AWQ 的 PPL 增量为 0.5–0.6、GPTQ 为 2.3–4.9。它支持“该实验中 AWQ 对分布
变化较不敏感”，不等于 AWQ 无需校准或对所有业务域都更稳。AWQ 与 GPTQ
artifact 一样，仍要核验 group、scale、pack layout 和后端 kernel。

### 17.4.4 QAT：让模型在训练时适应噪声

QAT 在前向过程中模拟量化和反量化，但可训练参数仍保留高精度副本。一个
简化的 fake-quant 前向是：

~~~math
W_{\mathrm{fake}}
=
s\cdot\operatorname{clip}
\left(
\operatorname{round}\left(\frac{W}{s}\right),
q_{\min},q_{\max}
\right)
~~~

round 本身不可微，训练实现需要用替代梯度或其他近似方式传播更新；不同 QAT
配方在 scale 学习、剪裁、损失和权重/激活范围上可能不同，不能由“做了 QAT”
推断成统一算法。PyTorch 的 FakeQuantize 2.14 文档明确把该模块定义为训练时
模拟 quantize/dequantize；对应公式和观测器状态要与最终导出的 scale 区分。

一个面向 LLM 的代表方案是 LLM-QAT：先从原始模型采样生成文本，再用原模型
logits 作为软标签训练量化学生模型，避免依赖原始预训练语料；论文同时训练
weights、activations 和 KV cache 的量化路径。在 LLaMA-7B 的 W4A8KV4 设置中，
作者报告常识任务平均分从 RTN 的 41.2 提高到 60.7；数字绑定论文的模型、数据
生成、训练配置和任务集合，且这是一项训练研究，不证明任意 serving kernel
支持相同格式。QAT 以训练资源换取对量化噪声的适应，最终模型仍需验证 artifact
布局、scale、目标 kernel、任务切片和回滚兼容性。

### 17.4.5 代表方案解决的是不同层的问题

下面的代表方案不是四个互相替代的“INT4 格式”。RTN 是量化基线，GPTQ 和 AWQ
主要决定如何生成低比特权重，MARLIN 则决定一类兼容权重如何在 GPU 上执行。

| 方案 | 所在层与要解决的问题 | 关键机制 | 代表性证据 | 代价、限制与后续改进 |
| --- | --- | --- | --- | --- |
| RTN | 转换基线；以最低转换复杂度得到低比特权重 | 每个权重独立舍入到最近的量化格点，不补偿层输出误差 | GPTQ 原论文的 OPT-175B WikiText-2 4 bit 行，PPL 为 10.54 | 低 bit 时容易让一串误差叠加。先用固定校准集与任务切片测量是否需要更复杂 PTQ |
| GPTQ | PTQ 算法；降低校准输入上的线性层输出重建误差 | 用激活统计构造近似 Hessian，逐列量化并以逆 Hessian 补偿剩余权重 | 同一原论文中 OPT-175B 4 bit PPL 为 8.37，FP16 为 8.34；量化耗时约 4.2 小时/单张 80GB A100 | 校准分布、分组、阻尼和数值稳定性会影响结果；weight-only 不减少乘加。可增加目标域校准、任务回归，或在仍不达标时比较 QAT |
| AWQ | PTQ 算法；利用输入激活保护敏感通道，同时保持硬件友好的全低比特权重 | 等价地缩放权重/输入通道，在小范围搜索 scale 与 clipping，再导出低比特权重 | AWQ 原论文的 OPT-6.7B INT3-g128 WikiText-2 PPL：RTN 23.54、AWQ 11.39、FP16 10.86 | 激活统计与 clipping 仍来自校准集；该论文的分布偏移结果不能推广到所有领域。可用目标 workload 校准并做代码、数字、长上下文和结构化输出切片 |
| MARLIN | 执行 kernel；解决 W4A16 在批量 decode 下仍要高效读权重和使用 Tensor Core 的问题 | 异步访存、任务调度、流水线、量化专用布局与解码计算 | 作者在 A10 大矩阵层级上报告 batch 16–32 约 3.9×、batch 128 约 1.5× 对 FP16；vLLM 端到端收益因 attention 和 serving 开销更低 | 它不改量化质量；收益受 GPU、矩阵形状、group/pack 布局和 batch 影响。高 batch 进入 compute-bound 后，需评估 W4A8 等激活量化路径及其新增数值误差 |

这些数字分别来自不同论文、模型、校准、评测和 kernel 配置，不能据此给 GPTQ、
AWQ、MARLIN 排统一名次。MARLIN 的论文使用 GPTQ 生成的 INT4 权重来测 kernel，
正好说明“权重怎样量化”和“权重怎样执行”可以组合，但两层的贡献需要分开
报告。AWQ 原论文也把算法和 TinyChat 系统分开：TinyChat 的加速同时来自
on-the-fly dequant、平台相关 pack 和 kernel fusion，不能把系统速度都归因于
AWQ 的 scale 搜索。

先用统一任务评测确认量化质量，再检查 manifest 和 runtime 是否读取了兼容布局，
最后用目标 batch/长度 trace 确认命中预期 kernel。当前 vLLM 文档分别列出量化
方案和 linear backend 选择入口；支持矩阵会随 release 变化，文档项不能代替
对实际模型、shape 和硬件的启动日志与 profiler 检查。

所以以下说法都可能同时成立：

- 一个 native INT4 artifact 由 PTQ 产生；
- 一个 QAT 模型最终仍然使用 W4A16 的 serving path；
- 一个 PTQ 权重可以在某个 runtime 上走 native fused kernel，在另一个 runtime
  上只能走解包慢路径；
- 一个文件使用 INT4 payload，但 embedding、lm head、激活和 KV 仍然是高精度。

## 17.5 Native INT4 artifact 是一份联合契约

### 17.5.1 manifest 应该记录什么

一个可以交给部署团队的量化 artifact，至少要有类似下面的元数据。示例字段
是工程建议，不代表所有 runtime 使用相同的名称：

~~~text
base_model_revision
quant_method
weight_dtype
activation_dtype
accumulate_dtype
group_size
group_axis
scale_dtype
zero_point_dtype
pack_order
weight_layout
excluded_modules
calibration_revision
target_gpu_arch
kernel_revision
engine_revision
tensor_parallel_degree
tokenizer_revision
chat_template_hash
adapter_revision
cache_schema_version
known_fallbacks
~~~

base_model_revision 防止把同名但不同权重的模型混用；quant_method 说明是
GPTQ、AWQ、普通 round-to-nearest 还是其他算法；excluded_modules 说明哪些
层保留高精度；target_gpu_arch、kernel_revision 和 engine_revision 决定
能否走预期执行路径。

tokenizer_revision 和 chat_template_hash 也不能省略。量化误差之外，模板
变化可能改变输入边界、工具 schema 和生成格式，导致一次错误被误归因于 INT4。

### 17.5.2 血缘链比文件后缀更重要

量化版本应能沿着下面的链回溯：

~~~text
base checkpoint
  -> calibration set and method
  -> quantized artifact
  -> pack / scale conversion
  -> kernel and engine build
  -> golden evaluation
  -> deployment release
~~~

如果只保存最后一个 .safetensors 或某个引擎二进制，线上质量下降时就无法
判断是校准分布问题、pack 顺序问题、kernel 回退还是模板变化。artifact checksum、
转换脚本版本和评测数据 revision 应该和权重一起进入发布存储。

### 17.5.3 加载成功不等于语义兼容

兼容性至少有三层：

1. **字节兼容**：loader 能否解析 shape、dtype 和文件；
2. **数值兼容**：scale、zero point、pack order 和累加规则是否正确；
3. **任务兼容**：代码、数字、引用、工具调用和安全行为是否达到要求。

第一层通过，不能推导出后两层通过。实际系统应在启动后用固定输入比较若干
golden logits 和结构化输出，再用真实 workload 做性能测试。

## 17.6 从 packed weight 到矩阵乘：执行路径的差异

### 17.6.1 路径 A：先解包，再用高精度 GEMM

最容易实现的路径是：

~~~text
packed INT4
  -> unpack integer values
  -> apply scale / zero point
  -> BF16 or FP16 weight
  -> high-precision GEMM
~~~

它的优点是兼容性较好，很多高精度 GEMM 可以复用；缺点是需要额外的解包
workspace、转换带宽和临时张量。这样的 artifact 可以节省磁盘和常驻权重
显存，却未必节省计算时的读取量。

### 17.6.2 路径 B：融合的 W4A16

W4A16 通常表示权重为 4 bit，激活为 16 bit。kernel 在矩阵乘附近读取 packed
weight 和 scale，再完成必要的解包和乘法，累加可以使用更高精度：

~~~math
Y
\approx
\operatorname{GEMM}
\left(
X_{\mathrm{FP16/BF16}},
D(Q(W_{\mathrm{INT4}}),S)
\right)_{\mathrm{acc}}
~~~

X 是 16 bit 激活，Q(W) 是 packed INT4 权重，D 表示依照 scale 和 zero point
解码，acc 表示累加 dtype。它仍然不是 W4A4：只有权重使用 4 bit，激活和
累加的精度更高。

### 17.6.3 W4A8 和更低精度激活

如果激活也量化到 8 bit，常写作 W4A8。它减少激活带宽，但要额外处理激活
scale、动态范围和更低精度的矩阵乘。激活通常随 batch、输入内容和层变化，
校准和运行时统计比静态权重量化更复杂。

看到“INT4 inference”时，要先问它是 W4A16、W4A8，还是只把权重存储为
INT4 后在计算前恢复。权重 bit width、激活 bit width、KV dtype 和累加 dtype
必须分开报告。

### 17.6.4 kernel 覆盖决定真实收益

同一个 engine 可能为常见矩阵形状提供 fused INT4 kernel，却在以下情况回退：

- group size 不在支持集合；
- hidden size 或 output size 不能满足 tile 对齐；
- tensor parallel 切分后局部矩阵形状不合适；
- 某个模块没有对应的量化实现；
- GPU compute capability 不满足要求；
- scale dtype、pack order 或 activation dtype 不匹配。

回退可能表现为启动报错，也可能表现为解包后使用 BF16 GEMM。生产 trace
应记录实际 kernel 名称、量化模块覆盖率、dequant 时间和回退原因，而不能
只记录配置文件中的 int4。

## 17.7 group size、scale 布局与并行切分

### 17.7.1 scale 存储也有成本

假设每个 group 有 G 个权重，INT4 payload 每个参数占 0.5 byte，scale
占 b_s byte，zero point 占 b_z byte，额外 metadata 占 b_m byte，则：

~~~math
b_{\mathrm{avg}}
\approx
0.5+
\frac{b_s+b_z+b_m}{G}
~~~

当 G=128、scale 用 2 byte 且没有单独 zero point 时，scale 平均开销约为
0.015625 byte/parameter；如果还保存更多 metadata，实际比例会继续增加。
这解释了为什么“理论四分之一”通常会变成略高于四分之一。

### 17.7.2 tensor parallel 的边界

如果量化 group 沿着被 tensor parallel 切分的轴组织，必须明确 group 是否在
分片边界处重新开始。如果一个 group 被拆到两张卡，scale 是复制、分片还是
在通信后合并，都会影响正确性和通信量。

一个简单的约束是：

~~~math
\operatorname{shard\_size}
\bmod G
=0
~~~

它只是在 group 轴上对齐时的必要条件，不是全部兼容性条件。tile size、padding、
转置和 scale 的内存连续性仍然要由目标 kernel 验证。

### 17.7.3 pipeline parallel 和混合精度模块

pipeline parallel 按层切分，某一阶段可能包含保留高精度的 embedding、norm、
projector 或 output head。每个 stage 的量化配置不能假定相同；发布 manifest
应记录 module-level map，而不是只放一个全局 dtype。

跨 stage 的 hidden state 通常仍要使用约定的 activation dtype。若一边把
activation 当 BF16，另一边按 FP8 scale 解码，模型可能在通信层面成功运行，
却产生难以定位的数值漂移。

### 17.7.4 pack 顺序和 ABI

pack order、endian、矩阵转置、scale 的索引顺序和 padding 规则实际上构成
量化 kernel 的 ABI。更换 engine 版本时，即使模型参数和 group size 没有变，
也可能需要重新转换 artifact。一个安全的做法是把转换版本和 kernel 版本都
作为 cache schema 和发布版本的一部分。

## 17.8 校准数据决定量化误差落在哪里

### 17.8.1 校准集不是越大越好，而是要覆盖真实路径

校准集的作用不是让报告看起来有一个更大的样本数，而是让 scale 和误差补偿
看到未来会出现的激活分布。至少应分桶覆盖：

| 数据桶 | 需要观察的风险 |
| --- | --- |
| 中文、英文和低资源语言 | token 频率与隐藏状态分布不同 |
| 代码和配置文件 | 符号、缩进、长标识符和精确复制 |
| 数字、表格和金融字段 | 小数位、单位、排序和引用 |
| 长上下文 RAG | 远距离 evidence 与中间位置 |
| 工具 JSON | schema、枚举、参数边界和停止条件 |
| 多模态 placeholder | 图像、音频、视频 token 后的连接层 |
| 拒答和安全样本 | 安全边界及误拒答风险 |

如果校准集只有短英文聊天，不能据此断言代码、长文档和工具调用安全。
校准样本还要固定 tokenizer、template 和输入长度分布，否则量化实验与部署
请求并不是同一个输入协议。

### 17.8.2 层敏感性实验

可以对第 l 层做一次高精度恢复，比较任务质量变化：

~~~math
\Delta_l
=
Q_{\mathrm{mixed},l}
-
Q_{\mathrm{INT4}}
~~~

Q_INT4 是全量化候选在一个任务切片上的分数，Q_mixed,l 是只把第 l 层
恢复到较高精度后的分数。解释 Delta_l 时要按指标方向统一口径：准确率类
指标越高越好，PPL/错误率类指标越低越好。恢复后改善越明显，说明该层值得
进一步检查；它不是跨模型通用的固定敏感层名单。

如果有多个任务桶，可以写一个加权观察分数：

~~~math
Q_{\mathrm{weighted}}
=
\sum_{t=1}^{T}\alpha_t Q_t,
\qquad
\sum_{t=1}^{T}\alpha_t=1
~~~

Q_t 是任务桶 t 的质量，alpha_t 是业务权重。这个分数适合观察总体趋势，
但不能把安全、财务数字或工具副作用等高风险任务的失败用普通聊天的高分
抵消。高风险切片应单独报告和单独决定。

### 17.8.3 哪些模块可能需要保留高精度

很多模型的 embedding、归一化、attention projection、MoE router、跨模态
projector、lm head 或长上下文路径可能对量化更敏感，但这不是固定规则。
判断依据应该来自层消融、校准数据和目标任务，而不是照抄另一个模型的配置。

混合精度配置可能是 INT4 权重、FP8 激活、BF16 累加，并对少数层保留 BF16
权重。它通常能在质量和容量之间取得更平衡，但也增加 manifest、kernel 覆盖、
测试矩阵和运维复杂度。

## 17.9 质量评估：不要只看 perplexity

### 17.9.1 高精度基线和候选路径必须使用同一请求

比较 BF16 和 INT4 时，固定模型 revision、tokenizer、chat template、随机种子、
采样参数、stop 条件和最大输出长度。否则输出差异可能来自采样设置，而不是
量化误差。

可以先观察固定前缀上的 logits 差异。对同一位置的两个分布 p 和 q，KL 散度可
写作：

~~~math
D_{\mathrm{KL}}(p\parallel q)
=
\sum_{v}p(v)\log\frac{p(v)}{q(v)}
~~~

v 遍历词表，p 可以是 BF16 基线分布，q 可以是 INT4 候选分布。logits 差异
适合定位数值变化，但它不能替代真实任务成功率：一个小的概率变化可能恰好
改变 JSON 的一个括号，也可能完全不影响最终任务。

### 17.9.2 任务切片和指标分母

| 任务 | 更应该观察的指标 |
| --- | --- |
| 普通聊天 | helpfulness、拒答率、长度和延迟 |
| 代码生成 | 测试通过率、编译率、补丁正确率 |
| 数字问答 | exact match、数字复制和单位正确率 |
| RAG | evidence recall、citation support、unsupported claim |
| 工具调用 | schema 合法率、参数准确率、重复提交率 |
| 结构化输出 | JSON/schema 合法率和字段级正确率 |
| 安全任务 | unsafe completion、误拒答和攻击成功率 |
| 长上下文 | 远距离检索、位置分桶和多轮状态保持 |

每个指标都要写清分母。例如 JSON 合法率是“可解析响应数 / 需要 JSON 的响应
数”，工具参数准确率不能把没有发起工具调用的普通聊天混进分母。量化后
如果模型更早停止，平均输出长度变短不能直接当成性能提升。

### 17.9.3 质量失败和 fallback 要分开

一条请求失败可能来自五个不同地方：

1. INT4 数值误差；
2. kernel 回退导致超时；
3. schema 或 tokenizer/template 不兼容；
4. 资源不足造成取消或重试；
5. 外部工具或检索系统失败。

日志需要把这些原因分开，否则一个总成功率无法回答“量化是否值得”。对于
能够回退到 BF16 的请求，还要记录回退前后使用的权重版本、kernel、显存峰值
和最终任务结果。

### 17.9.4 单位成功任务成本

成本分子和成功率必须使用同一批原始请求作分母。令 C_all 是按候选策略处理
一个原始请求所产生的期望总成本，已经包含首次推理、所有重试、验证、fallback
和人工复核；令 p_final 是这批原始请求最终成功的比例，则：

~~~math
C_{\mathrm{success}}
=
\frac{\mathbb{E}[C_{\mathrm{all\ per\ original\ request}}]}
{p_{\mathrm{final}}}
~~~

例如 100 个原始请求合计花费 120 个成本单位，最终 90 个成功，则单位成功成本
为 120/90=1.33 个成本单位/成功请求。重试和复核的开销已经计入 120，不能
再把它们重复加一次；成功率也不能用“成功尝试数 / 所有尝试数”替代原始请求
成功率。若不同阶段的账单与人工工时不同，应从 trace 分阶段汇总到每个原始
请求，再除以最终成功比例。

INT4 减少 GPU 显存并不必然减少这个数。如果 JSON 失败让请求重试两次，或者
代码任务需要人工修复，存储收益可能被质量成本抵消。

## 17.10 INT4 对容量、并发和延迟的真实影响

### 17.10.1 并发由多个约束共同决定

对一个 worker，可以把可承载的活跃请求数粗略写为：

~~~math
N_{\mathrm{active}}
\le
\min\left(
\max\left(
0,
\left\lfloor
\frac{B_{\mathrm{worker}}-M_{\mathrm{weight}}-M_{\mathrm{fixed}}}
{M_{\mathrm{KV,req}}+M_{\mathrm{workspace,req}}}
\right\rfloor
\right),
N_{\mathrm{scheduler}},
N_{\mathrm{network}}
\right)
~~~

这里 B_worker 是分配给单个 worker 的显存预算，M_weight 是常驻权重，M_fixed
是 runtime、保留余量和固定 workspace；M_KV,req 与 M_workspace,req 是一个
活跃请求带来的 KV 和增量 workspace 估算，N_scheduler 与 N_network 则是调度
和网络上限。这个式子是线性近似：真实 workspace 可能随 batch、shape 和 kernel
变化，KV 也要按请求长度/模型结构计算，最终值应由目标 runtime profile 校正。

如果服务原来受权重显存限制，INT4 释放的空间可以增加 KV 预算；如果服务
原来受 KV 或网络限制，权重变小可能只让模型更容易加载，在线并发不会同步
增加。权重是 worker 的固定占用，不能写成“权重预算 / 权重大小”来估算请求数；
只有请求到来时额外增长的 KV 和 workspace 才进入每请求分母。

### 17.10.2 prefill 和 decode 的瓶颈不同

prefill 一次处理较长输入，矩阵乘占比通常较高；decode 每步处理少量新 token，
却要读取所有历史 KV，常常更受内存带宽、cache 访问和调度影响。W4A16 可能
改善权重读取或矩阵乘，但不能从权重压缩公式直接推出 TPOT 改善。

压测应至少分开报告：

- batch 1 和连续 batching；
- 短输入/短输出与长输入/长输出；
- prefill TTFT、decode TPOT/ITL 和端到端 P99；
- cache hit/miss、量化 kernel 和 fallback；
- 峰值显存、功耗、GPU 利用率和单位成功任务成本。

### 17.10.3 “最低卡数”与生产卡数不是一回事

假设每张卡为 80 GiB，只允许使用 85%，可用预算为 68 GiB。上一节的工作
负载用 450 GiB BF16 时，容量下界为 7 张；但实际可能因为 TP degree、拓扑
和冗余选择 8 张。INT4 工作负载为 165 GiB，容量下界为 3 张；实际可能
选择 4 张以满足拓扑和故障恢复。

所以容量报告要同时写：

1. 由显存分式得到的 raw minimum；
2. runtime 支持的并行 degree；
3. 节点内互联和跨节点带宽；
4. N+1 或滚动发布所需的备用空间；
5. 长上下文和峰值请求的实际余量。

只报告“从 8 卡降到 4 卡”而不说明这是哪一种卡、哪一种 batch 和哪一种
上下文分布，会误导采购和容量规划。

## 17.11 Native INT4 与其他低精度对象的组合

### 17.11.1 权重 INT4 不等于 KV INT4

权重量化和 KV 量化发生在不同生命周期。权重通常启动时固定，KV 在请求期间
按 token 增长；前者主要影响常驻空间，后者还影响 cache 读写、page 迁移和
长上下文质量。发布 manifest 应分别记录：

~~~text
weight_dtype
activation_dtype
accumulate_dtype
kv_k_dtype
kv_v_dtype
kv_scale_scheme
~~~

一个服务完全可能使用 INT4 weight、BF16 activation、BF16 accumulate 和 FP8
KV。把这些都简称为“INT4 推理”会掩盖实际误差路径。

### 17.11.2 speculative decoding

draft model 生成候选，target model 验证候选。若 target 使用 native INT4，
需要重新测接受率；如果量化误差改变 target logits，接受 token 数可能变化，
导致原本的加速收益下降。

可以先记录：

~~~math
a
=
\frac{N_{\mathrm{accepted}}}{N_{\mathrm{proposed}}}
~~~

a 是平均接受率。它只是加速诊断，不是质量指标；还要比较 rejection 后的
分布正确性、最终任务成功率、临时 KV 回收和 target 的真实 kernel。

draft 和 target 不必使用完全相同的权重精度，但 hidden state、tokenizer、
停止规则和验证协议必须兼容。工具调用、结构化输出和多模态 token 的候选
路径尤其需要单独回归。

### 17.11.3 多模态 encoder 和 projector

图像 encoder、音频 encoder、视频采样器和 projector 的张量形状与语言模型
不同。语言模型主体有 INT4 kernel，不代表 projector 或跨模态连接层也支持
相同的 group size 和 pack layout。

多模态评估要把输入处理、视觉/音频 token 数、projector、语言层、输出格式
分别计时。否则图像任务变慢或字段丢失时，无法判断问题来自 INT4 语言权重
还是前面的媒体处理链。

### 17.11.4 LoRA、adapter 和权重合并

把 LoRA adapter 叠加到 packed INT4 权重上有三种常见方式：

1. 发布前把 adapter 合并到高精度权重，再重新量化；
2. 保留 INT4 base，运行时用高精度低秩分支补偿；
3. 为每个 adapter 单独生成已经量化的 artifact。

三种方式在加载时间、显存、切换速度和质量上不同。若 adapter 合并后没有重新
校准，原来的 scale 可能不再适合；若运行时叠加，额外 GEMM 可能抵消 INT4
带来的收益。adapter revision 应进入 prefix cache key 和发布 manifest。

## 17.12 兼容性检查与可解释回退

### 17.12.1 启动阶段检查

启动时可以按下面的顺序验证，而不是等第一条线上请求暴露问题：

1. 读取 checksum 和 base model revision；
2. 检查 group size、axis、scale/zero-point dtype 和 pack order；
3. 检查每个模块是否有目标 GPU 支持的 kernel；
4. 检查 tensor parallel 分片是否满足 group 和 tile 对齐；
5. 用固定输入比较 golden logits 和结构化输出；
6. 运行一组短、长、代码、工具和多模态 warmup；
7. 记录实际 kernel、解包时间、峰值显存和 fallback。

“检查”在这里是工程验证过程的自然名称；它不是一个把复杂证据压成单个
布尔值的内部标签。不同失败应当保留不同原因。

### 17.12.2 运行时回退的几种类型

| 回退类型 | 典型原因 | 需要记录的证据 |
| --- | --- | --- |
| 模块级回退 | embedding、lm head 或 projector 无 kernel | module、dtype、kernel |
| shape 回退 | tile 或 group 对齐不满足 | shape、batch、parallel degree |
| 硬件回退 | compute capability 不匹配 | GPU 型号、engine 版本 |
| 数值回退 | overflow、异常 scale 或 logits 异常 | scale 分布、异常位置 |
| 任务回退 | 结构化或高风险任务质量不达标 | task slice、质量指标 |
| 资源回退 | KV 或 workspace 造成 OOM/抢占 | 显存、队列、cache 状态 |

如果所有回退都记成“INT4 failed”，团队会错误地重做量化；如果所有回退
都静默发生，团队又会把一个混合路径当成 native benchmark。

### 17.12.3 观测字段

一次请求的 trace 至少应能关联：

~~~text
request_id
model_revision
quant_artifact_revision
weight_dtype
activation_dtype
accumulate_dtype
kernel_path
fallback_reason
group_size
kv_dtype
prompt_tokens
output_tokens
ttft_ms
tpot_ms
peak_memory_gib
quality_slice
~~~

这些字段让人能够回答三个实际问题：请求用了哪条计算路径，花费了多少资源，
最终质量是否属于某个已知风险切片。没有它们，INT4 的线上收益和退化都很难
被归因。

## 17.13 发布、灰度与回滚

### 17.13.1 从离线 replay 到真实流量

发布过程可以按证据逐步增加真实因素：

1. 固定 replay：同一批请求比较 BF16 与 INT4 的 logits、任务结果和资源；
2. 压力测试：加入不同长度、batch、抢占、prefix hit 和并行拓扑；
3. shadow：复制真实请求但不返回 INT4 结果，观察真实输入分布和 fallback；
4. 低风险 canary：让短聊天或低风险请求使用候选路径；
5. 分任务扩大：依据代码、RAG、工具和长上下文的独立结果逐类扩大。

灰度不是为了得到一个漂亮的平均值。每个切片都要同时保留高精度基线，记录
请求分布、cache 状态、kernel 路径和失败样本。

### 17.13.2 回滚要回滚整套协议

假设 INT4 版本的 JSON 合法率从 98% 降到 91%，正确处理不只是替换模型路径。
至少需要：

~~~text
停止新 INT4 分配
  -> 保留已经提交的外部副作用记录
  -> 排空或拒绝不兼容的 prefix / KV cache
  -> 切回已验证的 model + engine + tokenizer/template
  -> 重新运行 golden replay
  -> 核对 usage、成本和失败请求
~~~

如果只替换权重而继续复用旧 cache，旧 cache 的 model revision、position、dtype
或 scale schema 可能已经不兼容。对于有工具副作用的 Agent，请先处理提交
状态，再处理可重算的语言状态。

### 17.13.3 版本组合

可以把发布版本表示为：

~~~math
R
=
(
r_{\mathrm{model}},
r_{\mathrm{quant}},
r_{\mathrm{kernel}},
r_{\mathrm{engine}},
r_{\mathrm{tokenizer}},
r_{\mathrm{template}},
r_{\mathrm{cache}}
)
~~~

R 中的任何一项变化都要经过兼容性判断。旧 INT4 page、prefix cache 或
speculative 临时状态不能因为模型名字相同就直接复用。无法证明兼容时，清理
并重算通常比静默迁移更可靠。

## 17.14 一个可运行的容量与质量诊断 demo

下面的程序使用标准库建立一个教学账本。它不实现真正的 INT4 pack、CUDA
kernel 或模型评测，只把权重压缩、其他显存、卡数和任务质量分开列出。这样
读者可以先验证算术，再把实际 profiler 和回归结果替换进来。

~~~python
from math import ceil
from pprint import pprint


def worker_cards(total_gib, card_gib, usable_fraction):
    usable_gib = card_gib * usable_fraction
    return ceil(total_gib / usable_gib)


def total_worker_memory(weight_gib, common_memory):
    return weight_gib + sum(common_memory.values())


bf16_weight_gib = 400.0
int4_payload_gib = bf16_weight_gib * 4 / 16
scale_metadata_gib = 15.0
int4_weight_gib = int4_payload_gib + scale_metadata_gib

common_memory = {
    "kv_cache_gib": 20.0,
    "workspace_gib": 16.0,
    "communication_gib": 8.0,
    "runtime_gib": 6.0,
}

bf16_total_gib = total_worker_memory(bf16_weight_gib, common_memory)
int4_total_gib = total_worker_memory(int4_weight_gib, common_memory)

tasks = {
    "chat": {"quality": 0.975, "floor": 0.970},
    "code_json": {"quality": 0.965, "floor": 0.970},
}
quality_failures = [
    name for name, row in tasks.items() if row["quality"] < row["floor"]
]

diagnostics = {
    "bf16_total_gib": round(bf16_total_gib, 2),
    "int4_total_gib": round(int4_total_gib, 2),
    "int4_weight_reduction": round(1 - int4_weight_gib / bf16_weight_gib, 3),
    "bf16_min_cards": worker_cards(
        bf16_total_gib, card_gib=80.0, usable_fraction=0.85
    ),
    "int4_min_cards": worker_cards(
        int4_total_gib, card_gib=80.0, usable_fraction=0.85
    ),
    "native_kernel_ratio": 0.98,
    "quality_failures": quality_failures,
}

actions = []
if "code_json" in quality_failures:
    actions.append("retain_high_precision_for:code_json")
if diagnostics["native_kernel_ratio"] < 0.95:
    actions.append("inspect_kernel_fallback")
if diagnostics["int4_min_cards"] < 4:
    actions.append("check_topology_before_scaling_down")

recommendation = "selective_int4_rollout" if actions else "expand_int4_cohort"
report = {
    "diagnostics": diagnostics,
    "actions": actions,
    "recommendation": recommendation,
}
pprint(report, sort_dicts=False)
~~~

一组可复现输出如下：

~~~text
{'diagnostics': {'bf16_total_gib': 450.0,
                 'int4_total_gib': 165.0,
                 'int4_weight_reduction': 0.713,
                 'bf16_min_cards': 7,
                 'int4_min_cards': 3,
                 'native_kernel_ratio': 0.98,
                 'quality_failures': ['code_json']},
 'actions': ['retain_high_precision_for:code_json',
             'check_topology_before_scaling_down'],
 'recommendation': 'selective_int4_rollout'}
~~~

这里的 int4_weight_reduction 只是权重相关空间的减少比例，不是整台机器
的显存减少比例。bf16_min_cards 和 int4_min_cards 是按 85% 可用显存得到
的容量下界，不代表 runtime 支持的并行度。代码任务的质量失败也没有被聊天
任务的通过结果抵消，因此建议是选择性灰度，而不是全流量切换。

## 17.15 一个企业代码助手的完整取舍

假设一个企业代码助手有如下约束：

- 使用 80 GiB GPU，单 worker 希望保留 15% 余量；
- 需要支持 16K 输入、2K 输出和连续 batching；
- 代码补丁必须通过编译和单元测试；
- 工具 JSON 必须可解析，外部副作用必须可追踪；
- 长上下文 RAG 需要保留引用证据；
- 发布时要能在滚动升级期间回退。

高精度版本的 400 GiB 权重会让容量下界接近 7 张卡，实际可能选择 8 卡
tensor parallel。INT4 artifact 的权重相关空间为 115 GiB，连同公共开销
约 165 GiB，容量下界为 3 张卡，实际可能选择 4 卡。INT4 释放出的空间
可以提高 KV 预算或减少单 worker 的 GPU 数，但它没有自动证明代码质量。

离线回放发现普通聊天质量为 0.975，代码 JSON 合法率为 0.965，而代码
切片最低要求为 0.970。此时有三种合理路径：

1. 把所有请求留在 BF16，优先保证质量，接受更高硬件成本；
2. 只让低风险聊天使用 INT4，代码和工具请求使用高精度路径；
3. 用层敏感性实验恢复少数模块，重新校准并验证混合精度 artifact。

第三条路径的代价不是只增加几个配置字段。它会增加 artifact 组合、kernel
覆盖、容量计算、发布测试和回滚复杂度。如果恢复 output head、MoE router
或 tool-sensitive projection 后代码切片恢复到要求以上，额外复杂度可能
值得；如果质量仍然不稳，第二条路径更容易解释和维护。

上线时应分别观察：

| 维度 | 需要回答的问题 |
| --- | --- |
| 资源 | 实际峰值显存是否接近账本，KV 是否成为新瓶颈 |
| 执行 | native kernel 比例是否稳定，是否出现 shape 回退 |
| 延迟 | TTFT、TPOT、P99 是否受解包或调度影响 |
| 质量 | 代码测试、JSON、数字复制和引用是否退化 |
| 安全 | prompt injection、工具权限和拒答是否改变 |
| 恢复 | 回退时旧 cache 和外部工具状态如何处理 |
| 成本 | 单位成功任务成本是否真的下降 |

这个案例中，“INT4 能否部署”的答案不是一个二元结论。更准确的答案是：
某个 artifact 在某个硬件和 runtime 上可以走预期 kernel，容量收益成立，
但代码 JSON 切片仍需保留高精度或继续做混合精度校准。

## 17.16 常见误区与定位路径

### 17.16.1 误区：四 bit 就等于显存只剩四分之一

只有权重 payload 接近这个比例。scale、zero point、未量化模块、workspace、
KV、通信和 allocator 仍然存在。报告应把权重、KV、workspace 和其他请求
空间分列，并说明 GiB、GB 和峰值统计口径。

### 17.16.2 误区：native 代表训练时已经适配量化

“native”在不同项目中可能指预量化 artifact、原生格式、融合 kernel 或
训练感知量化。除非模型卡明确说明 QAT、校准过程和运行时路径，否则不能
从名称推断训练方法。

### 17.16.3 误区：文件能加载就说明 group size 正确

错误的 scale axis 或 pack order 不一定触发 shape error。固定 prompt 的
golden logits、数字复制、JSON 解析和长上下文引用比“加载完成”更能发现
这类静默错误。

### 17.16.4 误区：batch 1 的速度代表线上吞吐

低比特 kernel 可能只覆盖某些矩阵形状。连续 batching、不同序列长度、tensor
parallel 分片或长上下文会改变 kernel 选择。应报告 batch、长度、P50/P95/P99、
TTFT、TPOT 和 native kernel 比例。

### 17.16.5 误区：量化后输出更短就是更快

输出变短可能来自提前 EOS、结构化格式失败或拒答变化。吞吐要同时记录
生成 token 数、有效任务数、停止原因和重试次数。

### 17.16.6 误区：回滚只替换权重文件

量化 artifact 与 scale、kernel、engine、tokenizer、template、adapter 和
cache schema 相关。回滚时还要处理正在运行的请求、旧 prefix cache 和已提交
的工具副作用。

### 17.16.7 从症状反推原因

| 症状 | 优先检查 |
| --- | --- |
| 显存下降但速度不变 | 是否先解包到 BF16，dequant 是否占用时间 |
| 单请求正常，高并发 OOM | KV、workspace、page 碎片和 batch 重排 |
| 只有代码/数字失败 | 敏感层、校准桶、output head 和格式约束 |
| 启动快但 TPOT 变慢 | fused kernel 覆盖、scale load 和内存带宽 |
| 只有某个并行度失败 | shard boundary、group 对齐和 tile shape |
| 新版本 cache 命中后异常 | model/quant/kernel/cache schema 版本 |
| 平均质量正常但用户投诉 | 任务切片、fallback 请求和高风险分母 |

排查时先确认实际执行路径，再做 K/V 或层级质量消融，最后才决定更换量化
算法。否则容易把 runtime 兼容问题误判为 INT4 数值问题。

## 17.17 资料、证据边界与延伸阅读

本章的显存公式是容量估算模型，不能替代目标 GPU、engine 和真实长度分布的
压测。官方文档可以确认某个 runtime 的配置入口、支持列表或硬件要求，但
“文档中存在选项”不等于每个模型、每个 shape 和每个 batch 都会走快路径。

本章按层次引用 GPTQ、AWQ、LLM-QAT 和 MARLIN 原论文：前几篇讨论量化权重或
训练方案，MARLIN 讨论执行 kernel。TACL 综述用于建立 PTQ/QAT、
weight-only/weight-activation/KV cache 的分类地图，不代替原论文；其中对
AWQ 的简短描述与 AWQ 原论文的主方法并不完全一致，细节以原论文提出的全低
比特等价缩放为准。各论文 benchmark 使用的模型、校准数据、硬件和 kernel
不能直接外推为线上 SLO。模型卡、转换脚本、engine profile 和生产 trace 才能
共同说明一个具体 artifact 的实际能力。

本章 demo 只计算教学账本，不实现 4 bit nibble pack、scale 解码、CUDA
kernel 或真实模型评测。示例中的 400 GiB、15 GiB 和质量分数都是显式假设，
读者应替换为自己的 artifact manifest 和回归数据。

延伸阅读：

1. [TensorRT-LLM Quantization](https://nvidia.github.io/TensorRT-LLM/features/quantization.html)：NVIDIA TensorRT-LLM 的量化入口和 runtime 配置说明。
2. [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)：量化 kernel、engine 和版本实现的源码入口。
3. [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)：NVIDIA 模型优化、量化和导出工具入口。
4. [vLLM Quantization](https://docs.vllm.ai/en/latest/features/quantization/)：vLLM 当前量化能力索引，具体支持需按版本和模型核对。
5. [PyTorch AO Documentation](https://docs.pytorch.org/ao/stable/)：PyTorch 原生量化工具和配置的官方入口，具体 API 需按当前版本核对。
6. [Transformers bitsandbytes Quantization](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)：Hugging Face Transformers 中 4 bit 加载和量化接口的说明。
7. [llama.cpp](https://github.com/ggml-org/llama.cpp)：另一类本地推理 runtime 的量化格式和实现入口，不能与 GPU engine 的 pack layout 直接混用。
8. [AWQ](https://arxiv.org/abs/2306.00978)：激活感知权重量化研究，关注重要权重通道和低 bit 推理。
9. [GPTQ](https://arxiv.org/abs/2210.17323)：基于近似二阶信息的后训练权重量化研究。
10. [SmoothQuant](https://arxiv.org/abs/2211.10438)：激活异常值平滑和 W8A8 量化的研究背景，不等同于 W4A16。
11. [LLM.int8()](https://arxiv.org/abs/2208.07339)：大模型混合精度和异常值处理的研究背景。
12. [QLoRA](https://arxiv.org/abs/2305.14314)：4 bit 基座与低秩微调的研究入口，训练/微调语境不等于 serving kernel 契约。
13. [A Survey on Model Compression for Large Language Models](https://aclanthology.org/2024.tacl-1.85/)：用作压缩路线和评测指标的分类地图；遇到算法细节或与原论文表述不一致时回到一手论文核对。
14. [LLM-QAT](https://aclanthology.org/2024.findings-acl.26/)：阅读从教师模型生成校准文本、logits 蒸馏以及权重/激活/KV 联合量化的训练路线；实验结论绑定 LLaMA 模型、数据生成和评测配置。
15. [MARLIN](https://arxiv.org/abs/2408.11743)：阅读 W4A16 混合精度 kernel 如何用异步加载、调度、流水线和量化布局提高 batched decode；速度数字绑定作者 GPU、层形状、batch 和 vLLM 设置。
16. [PyTorch FakeQuantize 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)：核对 fake quant 的量化/反量化模拟公式及 observer 状态；API 文档不等同于某个 LLM 的完整 QAT 配方。

阅读这些资料时要区分三类证据：官方文档说明“这个 runtime 当前声称支持
什么”；论文说明“某种方法在特定实验中得到什么”；生产压测说明“你的
artifact 在目标 workload 上实际发生什么”。三类证据互相补充，不能用其中
任意一类替代另外两类。

## 17.18 本章小结

Native INT4 的价值不在文件名里的“4”，而在压缩权重能否沿着正确的 scale、
layout、kernel 和模型协议进入真实服务。

1. INT4 payload、PTQ、QAT、预量化 artifact 和 native kernel 是不同概念。
2. 理想权重压缩比例不等于整台 GPU 的显存比例，更不等于并发和吞吐比例。
3. group size、scale、zero point、pack order 和 shard boundary 是数值与性能
   的共同接口。
4. W4A16、W4A8、BF16 累加、FP8 激活和 INT4 KV 必须分别记录，不能统称为
   “INT4 推理”。
5. 校准数据和层敏感性决定误差落在哪些任务；perplexity 不能代替代码、数字、
   引用、工具和安全评估。
6. 实际部署要观察 native kernel、dequant、fallback、KV、P99、质量和单位
   成功任务成本。
7. 回滚需要同时处理模型、量化 artifact、kernel、engine、tokenizer、template、
   adapter 和 cache schema。

当一个团队能够说明某个 INT4 artifact 的生成方法、数值布局、目标硬件、
执行 kernel、未量化模块、任务质量、容量下界和回滚路径时，“原生量化”才是
一个可验证的部署事实，而不是一个令人误解的标签。
