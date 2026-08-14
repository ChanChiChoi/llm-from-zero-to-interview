# 第 66 章 Total Parameters 与 Active Parameters：两个数字如何同时为真

## 66.1 “模型有多少参数”为什么不是一个答案

介绍稀疏 MoE 或混合模型时，经常同时看到 total parameters 和 active parameters。初学者会问：到底是 500 亿还是 5000 亿？答案是两个数字描述不同对象。

total parameters 是 checkpoint 中可训练权重的总量，包含所有 experts、共享层和路由未选中的参数；active parameters 是一次 token 前向实际参与计算的参数量，通常包含被选中的 experts 和共享路径。它更接近单 token 的计算量，但不一定等于显存占用或服务成本。

## 66.2 MoE 的简单公式

设有 `E` 个 experts，每个 token 选择 `k` 个，单个 expert 的参数量为 `P_e`，共享参数为 `P_s`，则理想化：

```math
P_{\mathrm{total}}=P_s+E P_e
```

```math
P_{\mathrm{active}}\approx P_s+kP_e
```

如果 `E=64,k=2`，active 的 expert 部分约为总 expert 部分的 `1/32`。但真实模型还要计算 router、shared expert、embedding、输出层、top-k capacity、padding 和通信。

## 66.3 active 不等于显存

推理时通常要把所有 expert 权重放在 GPU、CPU 或远端存储中，或者在请求到来时加载。因此显存需求更接近常驻权重和 cache，而不是单 token active 参数。

训练时所有 expert 也可能需要 optimizer state、梯度和 checkpoint。active FLOPs 降低了计算，但 all-to-all 通信、expert load imbalance 和路由 buffer 仍然存在。

## 66.4 一个容量估算

假设共享参数 20B，64 个 expert 每个 10B，每 token 选 2 个，则：

```math
P_{\mathrm{total}}=20+64\times10=660\mathrm{B}
```

```math
P_{\mathrm{active}}\approx20+2\times10=40\mathrm{B}
```

若 BF16 每参数 2 字节，仅权重总存储约为 `660B\times2=1.32TB`，active 计算路径约使用 40B 参数的矩阵乘，但服务仍可能需要容纳 1.32TB 权重或采用 expert offload。把 40B 写成“只需 80GB 显存”是错误的，因为 KV、workspace、复制和并发没有算。

## 66.5 路由和负载均衡

router 为每个 token 计算 expert 分数并选择 top-k。若大量 token 选择同一个 expert，该 expert 过载，其他 expert 空闲；系统可能丢 token、padding 到 capacity 或增加通信等待。

训练中常加入 load balancing loss。一个教学化形式可以惩罚 token 分配比例 `f_e` 与 router probability `p_e` 的偏差：

```math
L_{\mathrm{balance}}\propto E\sum_{e=1}^{E}f_ep_e
```

具体正则和实现依模型而定。路由均衡不是纯训练指标，还影响线上专家并行和 p99。

## 66.6 active FLOPs 与 token 级成本

MoE 可以降低每 token 的 expert FLOPs，但 prefill 中的大 batch、padding、通信和 shared path 可能改变结果。decode batch 小时，专家矩阵乘可能难以填满 GPU，active 参数少却不一定更快。

容量规划要同时记录：total weight GiB、active FLOPs、expert dispatch bytes、all-to-all time、KV cache、batch throughput 和 p99。参数量只是一项输入，不是完整成本模型。

## 66.7 训练和推理的差异

训练要为所有 experts 保存梯度和 optimizer state，checkpoint 也包含全量参数；推理可以使用 expert parallel、offload、量化和按需加载。训练中的 token 路由分布可能和线上请求不同，不能直接把训练 active ratio 当作生产负载。

量化还要问是所有专家同样精度，还是按专家/层使用不同 precision。冷门 expert 被量化后可能只在少数任务上退化，平均分不一定能发现。

## 66.8 参数口径的审计

看到发布材料的“总参数”和“激活参数”，要核对是否包含 embedding、LM head、shared expert、router、adapter 和稀疏比例；还要记录 token-level 还是 sequence-level active、top-k 是否固定、不同模态是否使用不同路径。

如果来源只给一个营销数字而没有 config、层数、expert 数和 top-k，就只能把它当发布方口径，不能反推出完整架构。

## 66.9 常见误区

把 active 参数当常驻显存；把 total 参数当每 token FLOPs；忽略 shared layer；忽略专家通信；只测平均负载不测热点；将 total、active 和 dense-equivalent 混用；从参数量猜测模型质量。

## 66.10 面试回答与练习

回答“MoE 500B 为什么能用较小 active 参数运行”时，应区分总权重和每 token 激活路径，说明 top-k、shared expert、专家并行、通信、权重常驻和 KV cache 的差异。active 参数更接近单 token 计算，却不能单独决定显存、吞吐或成本。

练习一：给定 `P_s=20B,E=64,P_e=10B,k=2`，计算 total 和 ideal active。

练习二：列出为什么 active FLOPs 低但 p99 仍可能高的四个原因。

练习三：设计一个专家热点监控面板，至少包括路由比例、capacity、丢 token、通信和 p99。

### 66.10.1 参数量的教学分解

对含 shared 参数 `P_s`、`E` 个 expert、每个 expert 参数 `P_e`、每 token 激活 `k` 个 expert 的 MoE，可以写：

```math
P_{\mathrm{total}}=P_s+E P_e
```

理想 active 参数近似为：

```math
P_{\mathrm{active}}=P_s+kP_e
```

如果存在 shared expert、router、embedding、output head 和多个层，公式需要逐层相加。它的作用是说明 total 和 active 关注不同问题，不是提供任何具体模型的真实参数统计。

### 66.10.2 active 参数不等于单 token 成本

一个 token 即使只激活两个 expert，也可能需要把 token 通过 all-to-all 发到远端 GPU；专家权重可能已经常驻显存，通信和负载不均衡会决定延迟。router 热点会造成某个 expert 满载、capacity overflow 或 token drop。长上下文请求还要为 KV/state 分配资源，与 active FLOPs 没有直接比例关系。

因此容量规划至少同时看 total weight、active FLOPs、expert traffic、权重驻留、KV bytes、batch 和 p99。发布材料中“active 参数更小”不能直接翻译成“显存更小或成本更低”。

### 66.10.3 worked example：理想和现实

设 `P_s=20B`、`E=64`、`P_e=10B`、`k=2`，则理想 total 为 `660B`，active 为 `40B`。如果 token 均匀分配，计算路径可能接近 40B 的量级；如果 20% token 集中到少数 expert，热点 GPU 的处理时间和通信会决定尾延迟，平均 active 参数没有反映这个问题。

### 66.10.4 量化和参数口径

total/active 还要注明参数是否包含 embedding、共享层、量化后的存储格式和 router。一个模型的参数数量可以按训练精度统计，部署显存却按量化 payload 统计；两个数字没有矛盾，但不能混在同一列比较。

## 66.11 参数量、FLOPs 和带宽的三角关系

total parameters 主要回答模型权重容量，active parameters 近似回答一个 token 经过多少专家计算；二者都不能单独回答实际速度。一个 MoE 请求还要支付 router、token permutation、通信、padding 和权重读取。

可以用三个账本并列：

~~~math
N_{\mathrm{total}}\rightarrow M_{\mathrm{weight}},
\qquad
N_{\mathrm{active}}\rightarrow C_{\mathrm{matmul}},
\qquad
C_{\mathrm{comm}}\rightarrow \mathrm{latency\ and\ bandwidth}
~~~

如果权重分散在多个设备，active 计算很小也可能等待远端专家；如果专家集中在一个设备，热点会让 p99 变差。参数口径必须和部署拓扑一起解释。

## 66.12 训练 token 的专家分布

专家是否学到分工，不能只看路由概率。应比较不同数据域、语言、代码和长度 bucket 的 expert assignment、负载和质量。一个专家很热门，可能代表它承担通用功能，也可能说明 router 塌缩。

可以计算专家负载熵：

~~~math
H_{\mathrm{load}}
=-\sum_{e=1}^{E}p_e\log p_e
~~~

负载熵高不一定更好，因为均衡路由可能损害语义分工；负载熵低也不一定失败，因为少数专家可能确实适合通用 token。它必须和 token drop、任务质量、通信和梯度一起分析。

## 66.13 面向选型的参数报告模板

一份可比较的模型卡至少给出：total parameters、active parameters、非专家参数、专家数量、top-k、权重 dtype、KV/state dtype、最大 batch、上下文长度、prefill/ decode 吞吐和硬件。

如果厂商只公开其中一部分，应将未知字段标为未知。不要用 total 参数除以专家数猜 active，也不要用单次 API 价格反推模型实际 FLOPs。

## 66.14 参数口径与权重常驻

MoE 的 active parameters 描述一次 token 经过的计算路径，不表示只有这些权重驻留在 GPU。若专家权重分片在不同设备，token 仍需通信；若采用 offload，active FLOPs 低也可能等待权重搬运。

可以把单 token 的服务成本拆成：

~~~math
C_{\mathrm{token}}
=C_{\mathrm{dense}}
+C_{\mathrm{active\ expert}}
+C_{\mathrm{router}}
+C_{\mathrm{communication}}
+C_{\mathrm{weight\ movement}}
~~~

因此 total/active 只能作为第一层口径，实际吞吐还要绑定拓扑、batch、专家热点、dtype 和 kernel。

## 66.15 专家负载的可观测性

路由器输出的概率和实际执行的 token fraction 可能不同。容量限制、token drop、padding、top-k 和跨 rank dispatch 都会改变最终负载。应记录每个专家的输入 token、溢出、等待时间、输出质量和梯度。

负载均衡损失下降不等于服务均衡；某些专家可能承担高价值但低频任务，强行均匀会损害 specialization。质量、负载熵、通信和 p99 要一起分析。

## 66.16 面试中的完整参数回答

回答“一个 100B MoE 模型到底多大”时，应先问 total 还是 active，再问非专家参数、专家数量、top-k、共享专家、权重 dtype、常驻位置、KV/state 和通信。若资料缺少字段，就明确标为未知。

对比模型时，最好同时给出：权重存储、单 token 计算、每请求 KV/state、专家通信、量化和 batch 影响。这样才能把两个看似矛盾的数字放回同一资源账本。

## 66.17 从参数量走向成本模型

Total parameters 描述模型拥有多少权重，active parameters 描述一个 token 走过多少参数路径。二者都重要，但还必须和权重常驻、KV、通信、负载均衡、量化和 batch 一起解释。发布数字要保留口径和模型版本。

本章的 MoE 参数关系是教学模型；具体 expert 数、shared path、top-k 和 active 统计以官方 config、技术报告和模型卡为准。

## 66.18 三种“参数量”口径

面试和模型卡中常见的参数数字至少有三种含义：

1. total parameters：所有共享层和所有 expert 的权重总量。
2. active parameters：一次 token 路由实际参与矩阵乘法的参数量。
3. resident parameters：某一设备或某一时刻实际驻留在显存中的参数量。

这三个数字可能完全不同。MoE 可能拥有很大的 total，但每个 token 只激活少量 expert；量化和 offload 可能降低 resident bytes，却不改变数学上的 parameter count；shared attention、embedding、router 和 output head 是否计入，也会改变公布口径。

看到“某模型是 400B，但每 token 只用 20B”时，不能直接把 20B 当成显存需求或总计算量。至少要问：20B 是否包含 shared path？是否按 BF16、FP8 还是量化权重统计？top-k 是按 token 还是按 layer？通信和 router overhead 是否计入？

## 66.19 MoE 参数公式

设有 E 个 expert，每个 expert 的参数量为 P_e，共享部分参数量为 P_s，router 和其他小模块为 P_r，则：

~~~math
P_{\mathrm{total}}
=P_s+E P_e+P_r.
~~~

如果每个 token 在每层选择 k 个 expert，理想化的 active 参数为：

~~~math
P_{\mathrm{active}}
\approx P_s+kP_e+P_r.
~~~

这只是参数访问量的近似，不等于 FLOPs 的精确值。不同 expert 的 hidden size、shared expert、top-k 组合、capacity overflow、token dropping 和 fused kernel 都会改变实际计算。若 attention 占了很大比例，active FFN 参数的减少也不会按相同比例降低全模型计算。

## 66.20 参数量与 FLOPs 的关系

对一个 dense linear layer，输入 token 数为 T 时，矩阵乘法成本大约与输入维度、输出维度和 T 的乘积有关。MoE 的 expert 参数稀疏化降低的是每个 token 访问的专家矩阵，不代表所有设备都只保存 active expert：

~~~math
F_{\mathrm{token}}
\approx F_{\mathrm{shared}}
+kF_{\mathrm{expert}}
+F_{\mathrm{router}}
+F_{\mathrm{attention}}.
~~~

训练时还要乘以 forward/backward 的常数，并考虑 all-to-all 通信；推理时 decode 的单 token 计算、prefill 的矩阵利用率和 batch 形状又不同。因此 active parameters 适合作为“每 token 计算路径”的解释，不应直接当成吞吐预测公式。

## 66.21 为什么 total 大仍可能昂贵

total parameters 影响至少四类成本：

1. 权重存储：所有 expert 权重可能需要在集群中保存。
2. 加载和初始化：启动服务需要读取、反量化和建立权重布局。
3. 路由与通信：token 发送到远端 expert 需要 all-to-all。
4. 容错与升级：更多分片、更多 expert 增加重启和灰度复杂度。

一个 token active 20B 不代表一张 GPU 只需放 20B。若请求混合会访问所有 expert，服务仍需让这些 expert 可被快速调度；若 expert 采用按需加载，磁盘和网络延迟可能成为新瓶颈。

## 66.22 负载均衡改变有效 active 成本

理论上每个 token 选择 k 个 expert，实际可能出现热门 expert 过载、冷门 expert 空闲。令第 e 个 expert 收到 n_e 个 token，则负载均衡可以观察：

~~~math
\mathrm{CV}
=\frac{\operatorname{std}(n_e)}
{\operatorname{mean}(n_e)}.
~~~

CV 高时，平均 active 参数仍然没有变，但尾部设备的等待、显存和通信时间会变差。capacity factor 会限制每个 expert 能接收的 token 数，超出的 token 可能被丢弃、送回 shared path 或重新路由。评估必须记录 token drop、expert overflow、all-to-all bytes 和最慢 rank，而不是只看平均 GPU 利用率。

## 66.23 一个数字例子

假设共享部分 10B，有 E=64 个 expert，每个 expert 5B，router 0.1B，每个 token 选择 k=2，则：

~~~math
P_{\mathrm{total}}
=10+64\times5+0.1
=330.1\mathrm{B},
~~~

理想 active 参数约为：

~~~math
P_{\mathrm{active}}
=10+2\times5+0.1
=20.1\mathrm{B}.
~~~

这个例子说明为什么模型可以同时拥有很大的容量和较小的 token 路径。但如果共享 attention、embedding、通信和未融合的 router kernel 占据大量成本，实际 throughput 可能远低于一个 20B dense 模型的直觉估计。

## 66.24 参数量、显存和量化

权重显存的粗略估算是：

~~~math
M_{\mathrm{weight}}
\approx P_{\mathrm{resident}}\times s_{\mathrm{weight}}
+M_{\mathrm{scale}}
+M_{\mathrm{layout}}.
~~~

BF16 每参数通常按 2 字节估算，FP8/INT4 的平均字节数更低，但 scale、zero point、padding、混合精度层和临时 buffer 会增加额外占用。量化降低的是存储和可能的带宽，不自动改变 total/active parameter 的定义。

MoE 还要区分全量加载、expert sharding 和按需加载。全量加载启动快但占用更多显存；按需加载节省 resident memory，却可能增加 cache miss、网络读取和尾延迟。

## 66.25 参数量与上下文成本要分开

长上下文系统的主要增量可能来自 KV/state，而不是权重。服务容量模型至少写成：

~~~math
M_{\mathrm{request}}
=M_{\mathrm{weights}}
+M_{\mathrm{KV/state}}
+M_{\mathrm{activation}}
+M_{\mathrm{workspace}}.
~~~

一个 total 参数更大的 MoE 模型，如果 active 路径和 KV/state 设计更节省，可能在特定 batch 下更容易服务；反过来，一个 total 较小的 dense 模型如果使用较大的 MHA KV cache，也可能受长上下文限制。不能只按参数排序推断推理成本。

## 66.26 对比模型时的统一报告格式

为了避免营销数字互相误导，建议每次对比都给出：

~~~text
model revision
total parameters
active parameters definition
shared parameters
expert count and top-k
resident weight bytes
KV/state bytes per request
precision and quantization
FLOPs or measured tokens/s
communication bytes
batch and sequence length
quality harness and date
~~~

如果某个字段未公开，应写 unknown，而不是用同系列模型或社区猜测补齐。尤其是闭源模型，产品页可能公布上下文和价格，却没有 total/active 的内部结构证据。

## 66.27 训练、推理和更新的不同账本

训练 active 参数影响前向、反向和 optimizer state；推理 active 参数影响 token 路径；checkpoint 和发布包仍要处理 total 权重。训练还需要为 expert 梯度、optimizer states、通信和 checkpoint 保存空间付费。

因此一个架构选择可能在推理上很划算，在训练上却因为 all-to-all 和负载不均衡变贵；也可能训练吞吐不错，但线上小 batch decode 的 expert 路由不规则，尾延迟很差。架构评估必须分别测 pretraining、prefill、decode 和模型升级。

## 66.28 active 参数的三个口径

“active”至少可能指每个 token 实际参与矩阵乘的 expert 参数、一次请求实际被调度到的参数，或某个产品宣传中的激活规模。三者不一定相同。MoE 的 router 可能选择 top-k expert，但 resident weight、通信 buffer、共享层和 runtime workspace 仍然占用显存。

因此应同时报告：总参数 `N_total`、每 token active 参数 `N_active`、常驻权重 `N_resident`、临时激活 `M_activation` 和通信量 `B_comm`。一个模型可以在 FLOPs 上看起来很小，却因所有 expert 都要常驻而无法在目标 GPU 上部署。

## 66.29 一个容量账本例子

假设共有 64 个 expert，每个 token 选择 2 个，expert 总参数为 64B，共享参数为 8B。理想 active 参数约为 `8+64×2/64=10B`，但若所有 expert 都以 BF16 常驻，权重仍需要约 `64B×2 bytes=128GB`，还没有计入量化 scale、通信和 workspace。

这个例子说明 active 参数适合描述 token 级计算，不适合单独描述显存和故障域。若采用 expert offload，计算成本、PCIe/NVLink 流量和尾延迟又会改变，必须用真实硬件压测验证。

## 66.30 用统一分母比较模型

跨模型比较时，至少绑定相同的输入/输出长度、batch、硬件、精度、并行方式、cache、工具和 reasoning effort。报告质量时用每个成功任务的 active FLOPs、延迟和显存，而不是用“参数量越小越快”的口号。MoE、稠密模型和混合架构的成本账本可以不同，但分母必须可解释。

## 66.31 active FLOPs 也有误差

理论 active 参数假设 router 均匀、kernel 满载、没有 padding 和通信；真实系统还会受到 token drop、expert capacity、batch 形状、跨卡 all-to-all 和量化回退影响。profile 时要同时保存 router histogram、实际 matmul shape、空转时间和通信占比。

## 66.32 模型发布表的最小字段

比较模型时至少列 `N_total`、`N_active/token`、`N_resident`、精度、KV/state、通信、上下文、训练/推理硬件、batch、输出长度、质量和单位成功成本。缺失字段不能用 active 参数猜出来，应标为未知或待核验。

## 66.33 训练更新和服务加载不是同一账本

训练需要 optimizer state、gradient、activation 和通信 buffer；服务需要权重、KV/state、workspace、batch 和网络。一个模型适合训练并不意味着同样的 active 参数适合在线服务，容量规划必须分别建账。

## 66.34 小结

Total parameters 说明模型容量和权重总账，active parameters 说明 token 经过的条件路径，resident parameters 说明当前设备真正承担的存储。它们都不能单独代表质量、吞吐或成本。只有把参数口径、量化、KV/state、通信、负载均衡、batch 和评测条件放到同一张账本，模型之间的比较才有意义。
