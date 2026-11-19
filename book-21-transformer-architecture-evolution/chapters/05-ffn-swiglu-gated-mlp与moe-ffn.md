# 第五章：FFN、SwiGLU、Gated MLP 与 MoE FFN

## 5.1 注意力之后，token 表示还要经过什么变换

Transformer block 中的 attention 会让一个位置读取其他 token 的表示。读完上下文后，每个位置还要对自己的特征做非线性变换，才能把刚收集到的信息加工成下一层可用的表示。这个按位置分别应用的网络就是前馈网络（feed-forward network，FFN）；现代模型里也常叫 MLP。

先用一个简单分工建立方向：attention 负责 token 之间的信息交换，FFN 负责把每个 token 当前的通道表示变换到新的表示。这个分工说明两者处理的轴不同，不代表 FFN 会把事实或概念存放在某个可直接定位的固定单元里。

本章先从原始 Transformer 的两层 FFN 出发，再解释 SwiGLU 怎样在通道维加入门控，最后说明 MoE 怎样让不同 token 只计算一组专家中的一小部分。理解后，读者应能按相同口径比较普通 FFN 与 SwiGLU 的参数量，并区分 MoE 的总参数、每 token 使用的参数和实际服务成本。

## 5.2 Dense FFN：每个位置共用参数，分别计算

设输入张量为 $X\in\mathbb{R}^{B\times T\times D}$： $B$ 是 batch 大小， $T$ 是序列长度， $D$ 是模型宽度。取其中一个 token 的向量 $x_{b,t}\in\mathbb{R}^{D}$，FFN 先把它投影到宽度为 $M$ 的中间表示，再投影回 $D$：

~~~math
h_{b,t}=\phi(x_{b,t}W_1+b_1),\qquad
y_{b,t}=h_{b,t}W_2+b_2.
~~~

这里 $W_1\in\mathbb{R}^{D\times M}$， $W_2\in\mathbb{R}^{M\times D}$， $\phi$ 是逐元素非线性函数；原始 Transformer 使用 ReLU。每个位置都执行这两步，但所有位置共用同一组 $W_1$、 $W_2$ 和偏置。换言之，FFN 不是给每个 token 单独存一套参数，而是把同一个函数分别应用到各位置。

从张量形状看，计算是 $[B,T,D]\rightarrow[B,T,M]\rightarrow[B,T,D]$。中间投影只改变每个 token 的通道维度，不在序列位置之间混合信息；token 间的读取由 attention 完成。原始定义可见 [Attention Is All You Need §3.3](https://arxiv.org/abs/1706.03762)。

## 5.3 用参数量看中间层为什么会变宽

忽略偏置时，两层 FFN 的参数量约为：

~~~math
P_{\mathrm{dense}}\approx DM+MD=2DM.
~~~

若 $M=4D$，则矩阵参数约为 $8D^2$。原始 Transformer 选择了 $D=512$、 $M=2048$；后续模型也常用约 $4D$ 的中间宽度，但 $4D$ 是常见配置，不是每种模型都必须遵循的规则。

若把一次乘加计作两个浮点运算，两次矩阵乘法对每个 token 约需 $4DM$ FLOPs，尚未计入激活函数和其他开销。这个估算帮助读者比较不同宽度的 FFN；实际整层成本还会受序列长度、batch、数据类型、算子实现和硬件影响。

## 5.4 GLU：让一条通道表示调节另一条

普通 FFN 先对一条投影结果应用激活函数。门控线性单元（gated linear unit，GLU）改为从同一个输入产生两条宽度为 $M$ 的支路：一条形成候选表示，另一条形成门控值，再逐元素相乘：

~~~math
u=xW_u+b_u,\qquad
g=\phi(xW_g+b_g),\qquad
h=u\odot g,\qquad
y=hW_d+b_d.
~~~

$u$ 和 $g$ 都是长度为 $M$ 的向量， $\odot$ 表示对应位置相乘， $W_d$ 再把结果投回模型宽度。门控支路改变了每个通道的幅度：当某个位置的门控值较小时，对应候选特征的贡献也较小。它是可学习的连续调制，不是一个人工写死的开关，也不意味着某个通道永远对应某个固定语义。

原始 GLU 通常用 sigmoid 产生门控值；GLU 变体会替换门控支路的激活函数。SwiGLU 在门控支路使用 SiLU/Swish；它因此不只是把 ReLU 换成另一个激活函数，而是包含 gate、up、down 三组投影的门控 MLP。

## 5.5 SwiGLU 的三组投影

令 $\mathrm{SiLU}(z)=z\,\sigma(z)$，其中 $\sigma$ 是 sigmoid 函数。对一个 token，SwiGLU 可写成：

~~~math
g=\mathrm{SiLU}(xW_g),\qquad
u=xW_u,\qquad
y=(g\odot u)W_d.
~~~

这里 gate projection 和 up projection 都把宽度从 $D$ 变到 $M$，down projection 再从 $M$ 变回模型宽度。常见实现可以省略偏置；是否使用偏置由具体模型配置决定。PaLM 和 LLaMA 都公开采用 SwiGLU，但采用本身只能说明它是这些模型的结构选择，不能单独证明质量提升由这一处改动造成。

## 5.6 参数匹配时，不要固定相同的中间宽度

普通两矩阵 FFN 忽略偏置时约有 $2DM$ 个参数；SwiGLU 有 gate、up、down 三个矩阵，约有 $3DM$ 个参数。若两者采用相同的 $M$，SwiGLU 参数会多约一半。因此要比较近似相同参数量的配置，必须把 SwiGLU 中间宽度调小。

以普通 FFN 的 $M=4D$ 为参照，令两种网络的矩阵参数接近：

~~~math
3DM_{\mathrm{SwiGLU}}\approx2D(4D),\qquad
M_{\mathrm{SwiGLU}}\approx\frac{8}{3}D.
~~~

例如 $D=4096$ 时，普通 FFN 取 $M=16384$，有 $134{,}217{,}728$ 个矩阵参数。SwiGLU 取 $M=11008$，有 $135{,}266{,}304$ 个矩阵参数，约为前者的 $1.008$ 倍。这个数值只说明参数量接近；它不保证两种模型有相同质量、训练速度或推理延迟。

## 5.7 SwiGLU 的实验证据能说明到哪里

[Shazeer 的 GLU 变体研究](https://arxiv.org/abs/2002.05202)在 T5 序列到序列模型中比较了 ReGLU、GEGLU、SwiGLU 等 FFN 变体，并通过缩小门控版本的中间宽度匹配参数量和计算量。论文报告，GEGLU 和 SwiGLU 在 C4 span-filling 目标的 held-out log-perplexity 上较优；下游微调结果有噪声。这个证据支持“门控变体在该实验设置中表现较好”，不支持“所有 decoder-only LLM 都必然从 SwiGLU 获益”。

| 资料 | 公开内容 | 证据边界 |
| --- | --- | --- |
| [Shazeer, 2020](https://arxiv.org/abs/2002.05202) | 在 T5 序列到序列设置中等参数/等计算比较多种 GLU 变体；GEGLU、SwiGLU 的 held-out C4 指标较优 | 结论绑定该模型、C4 span-filling 训练和论文评测；微调结果并非每项都稳定领先 |
| [PaLM, 2022](https://arxiv.org/abs/2204.02311) | 报告模型采用 SwiGLU，并说明门控 MLP 使用三组矩阵 | 多个架构改动同时存在，不能将整模型表现归因于 SwiGLU |
| [LLaMA, 2023](https://arxiv.org/abs/2302.13971) | 报告使用 SwiGLU，并将中间宽度设为 PaLM 常用 $4D$ 的约 $2/3$ | 这是模型配置，不是对所有模型精确等质量或等延迟的证明 |

## 5.8 用短代码对应 SwiGLU 的张量形状

下面的 PyTorch 层接收形状为 $[B,T,D]$ 的输入。两条上投影产生 $[B,T,M]$，逐元素相乘后由 down projection 返回 $[B,T,D]$。它展示的是 FFN 子层，不包含 attention、残差或归一化。

**代码状态：静态审阅，未运行。** 这是一个依赖 PyTorch 的最小模块片段，不是可独立运行的训练脚本。

~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
~~~

公式把 token 表示写成行向量，因此映射矩阵的形状记作 $D\times M$；PyTorch 的 `nn.Linear(D, M)` 将权重存成 $M\times D$，执行的是同一个宽度映射。`nn.Linear` 作用于输入的最后一维，所以 batch 和序列维会保留；gate、up 的形状都是 $[B,T,M]$，输出形状是 $[B,T,D]$。代码中的乘法与上一节公式里的逐元素乘法一一对应。

## 5.9 MoE：每个 token 只经过一部分 FFN 专家

Dense FFN 对每个 token 使用同一个函数。若要增加模型可选择的 FFN 参数，又不让每个 token 都计算所有 FFN，可以准备 $E$ 个 expert，并由 router 根据当前 token 选择其中 $k$ 个。每个 expert 通常是一个 MLP。

对 token 表示 $x_i$，MoE 层的常见形式是：

~~~math
y_i=\sum_{e\in\mathcal{S}(x_i)}w_e(x_i)F_e(x_i),\qquad
|\mathcal{S}(x_i)|=k.
~~~

$\mathcal{S}(x_i)$ 是 router 为该 token 选出的 expert 集合， $F_e$ 是第 $e$ 个 expert 的输出， $w_e$ 是合并各 expert 输出时使用的权重。许多实现会在选中的专家之间归一化权重，但路由和权重处理存在不同方案。

可以把一次计算分成三步：

1. router 读取 token 表示，为可选 experts 打分并选出一小部分。
2. 被选中的 experts 分别处理这个 token；其他 experts 在该 token 的这一步计算中不执行。
3. router 权重把选中 experts 的结果合并，输出仍与输入 token 表示处于同一模型宽度。

不同 token 可以选择不同的 expert 子集。这就是“稀疏激活”的含义：专家库很大，但单个 token 只走其中几条计算路径。

用两个 token 看 top-2 路由。下面假设 router 为它们选出不同的专家组合；专家编号和权重只是教学示意，不对应某个模型的实际配置。

| token | 选中的 experts | 加权后的输出 |
| --- | --- | --- |
| $x_1$ | $E_1,E_3$ | $w_1(x_1)F_1(x_1)+w_3(x_1)F_3(x_1)$ |
| $x_2$ | $E_2,E_4$ | $w_2(x_2)F_2(x_2)+w_4(x_2)F_4(x_2)$ |

两行展示了两件事：不同 token 可走不同的专家路径；每个 token 的输出仍由它选中的专家结果按 router 权重合并。实际路由和权重由模型计算，不是固定表格。

## 5.10 总参数与每 token 使用参数是两本账

假设有 $E$ 个大小相同、每个含 $P_e$ 个参数的 routed experts；模型其余总是参与计算的部分有 $P_{\mathrm{shared}}$ 个参数，router 有 $P_r$ 个参数。按这个简化口径，模型总参数为：

~~~math
P_{\mathrm{total}}=P_{\mathrm{shared}}+EP_e+P_r.
~~~

若每个 token 选择 $k$ 个 experts，每 token 参与的参数量（常称 active parameters）可近似记为：

~~~math
P_{\mathrm{active}}(x)=P_{\mathrm{shared}}+kP_e+P_r.
~~~

若各 expert 大小不同，则把 $kP_e$ 换成被选中 experts 的参数量之和。具体模型卡可能采用不同的计数口径，尤其对共享层和 router 的处理应按来源核对。

[Mixtral 8x7B 论文](https://arxiv.org/abs/2401.04088)报告每层有 8 个 FFN experts、每个 token 选择 2 个，并给出约 47B 的 sparse total 参数和约 13B 的 active 参数。这里的整模型 active/total 比例不能简单用 $2/8$ 代替，因为 attention、其他共享权重和计数口径也参与整模型统计。论文指出，active 参数数与推理计算开销相关，但它不包括 sparse total 参数对应的服务内存，也不反映设备利用率；路由和同一设备运行多个 experts 时的额外内存加载也会带来开销。服务时仍需保存所有可选 experts 的权重，但可以通过模型并行或 Expert Parallelism 分布到多张 GPU；这不等于每张 GPU 都存一整份。

因此至少要分开读三项：总参数描述整套权重规模；active 参数描述一个 token 参与的权重规模；FLOPs、显存、通信和延迟则要结合执行方式与硬件测量。 $P_{\mathrm{active}}(x)$ 不是端到端成本的替代指标。

## 5.11 SwiGLU 与 MoE 是两条不同的架构轴

SwiGLU 说明“一个 FFN 内部怎样形成通道表示”；MoE 说明“多个 FFN 中哪些会处理当前 token”。两者可以组合：一个 MoE expert 本身也可以是 SwiGLU MLP。Mixtral 就采用 SwiGLU 作为专家函数。

MoE 常放在 Transformer 的 FFN 子层，是因为 FFN 对每个 token 独立计算，router 可以按 token 选择计算路径，同时保留 attention 的 token 间信息交换结构。GShard、Switch 和 Mixtral 都采用了这类设计。这是常见架构选择，不是“attention 在理论上不能使用专家”的定理；一旦改动 attention 的路由，就会进一步改变 token 间的信息可见性和计算路径。

| 路线 | 论文披露的设计 | 阅读时保留的边界 |
| --- | --- | --- |
| [GShard](https://arxiv.org/abs/2006.16668) | 在 Transformer 中隔层用 position-wise MoE 替换 FFN，采用 top-2 路由并配套自动分片 | 论文的规模、吞吐和质量结果绑定多语翻译及 TPU 集群设置 |
| [Switch Transformer](https://www.jmlr.org/papers/v23/21-0998.html) | 用 top-1 选择简化稀疏路由，并研究容量、负载均衡和低精度训练 | 不同路由实现对超容量 assignment 的处理不同；细节见第 21 章 |
| [Mixtral](https://arxiv.org/abs/2401.04088) | 8 个 experts 中 top-2；单个 expert 是 SwiGLU FFN | 47B/13B 是作者模型的参数报告，不等于普遍质量、显存或延迟比例 |
| [DeepSeekMoE](https://aclanthology.org/2024.acl-long.70/) | 研究更细粒度的 routed experts 与 shared experts | 论文实验不证明细粒度或共享专家在所有任务和设备上都更好 |

路由细节、capacity 与 token-expert assignment、负载均衡、expert parallel 和 all-to-all 通信会改变训练和 serving 成本，本章只建立基础概念。具体路由路线见[第 21 章](./21-MoE路线Switch-GShard-Mixtral-DeepSeek.md)，Dense 与 MoE 的训练/服务账本见[第 22 章](./22-Dense与MoE训练服务trade-off.md)；模型案例见 [LLaMA 第 54 章](./54-LLaMA现代开源LLM基线.md)、[Mixtral 第 55 章](./55-Mistral-Mixtral高效与MoE.md)与[DeepSeek 第 58 章](./58-DeepSeek-V2-V3-R1架构训练.md)。

## 5.12 自测与本章小结

读模型配置时，可以按这个顺序核对：它是 dense FFN 还是 MoE？如果是 gated MLP，中间宽度是否与参数比较口径相符？如果是 MoE，每 token 选几个 expert，模型报告的 total/active 参数包括哪些部分？最后再查看容量、通信、服务内存和实测延迟，而不要从 active 参数直接猜结论。

**面试回答示例：**

> Transformer 的 FFN 对每个 token 独立地做通道变换，参数在各位置间共享。SwiGLU 在 FFN 中增加 gate 和 up 两条投影，用逐元素乘法调节中间通道，因此相同中间宽度时它比两矩阵 FFN 参数多；比较时通常会缩小中间宽度。MoE 则在多个 FFN experts 中按 token 选择少数几个，所以总参数和每 token 使用参数需要分别统计。active 参数不等于 FLOPs、显存或延迟，路由、通信和硬件利用率还要单独看。

**练习：**

1. 对输入形状 $[B,T,D]$，分别写出两层 dense FFN 和 SwiGLU 的中间张量形状。
2. 取 $D=4096$，计算 dense $M=4D$ 与 SwiGLU $M=11008$ 的近似矩阵参数量，并说明它们匹配的是什么、没有匹配什么。
3. 用自己的话区分 GLU 的通道门控和 MoE 的 expert 路由；说明一个 MoE expert 为什么可以本身采用 SwiGLU。
4. 阅读 Mixtral 论文，解释 47B total 与 13B active 分别描述什么，并列出 active 参数不能回答的两类 serving 成本问题。
5. 需要了解 capacity、assignment overflow 和 all-to-all 时，沿第 21、22 章继续阅读，不把这些成本写成一个通用的通过/不通过阈值。

本章从共享参数的 position-wise FFN 讲起，推到三矩阵 SwiGLU 和它的近似等参数宽度，再介绍 MoE 如何让每个 token 选择少量 FFN experts。下一章转向残差和归一化：FFN 与 attention 的输出怎样通过 residual stream 进入深层网络，并由 LayerNorm 或 RMSNorm 稳定处理，见[第 6 章](./06-residual-layernorm-rmsnorm-prenorm与训练稳定性.md)。此前 attention 的信息路由和 Q/K/V 前置概念见[第 2 章](./02-self-attention的信息路由本质.md)与[第 3 章](./03-qkv-attention-head与表示子空间.md)。
