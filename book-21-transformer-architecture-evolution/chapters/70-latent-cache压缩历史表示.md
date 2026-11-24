# 第 70 章 Latent Cache：压缩历史表示

上一章说明，混合架构可以在不同层保存普通 K/V、窗口内 K/V、逐 token latent 或递归状态。本章把镜头收窄到 latent cache：每个历史位置仍保留一个能被当前 query 寻址的表示，但该表示不再直接保存完整的多头 key 与 value。读完后，你应能解释 latent 怎么形成、attention 怎样使用它、它和其他压缩方式改变了什么，以及为什么“缓存字节变少”还不能单独证明模型质量和生成速度都更好。

## 70.1 从显式 K/V 到逐 token latent

### 70.1.1 显式 K/V cache 的逐 token 账本

在常规自回归 Transformer 中，每层为历史位置保留 key 和 value，后续生成可以直接复用。若每层有 $H_{\mathrm{kv}}$ 个独立 KV head/group、每个向量宽度为 $d_h$、每个元素占 $b$ 字节，则一个 token 在一层的逻辑 payload 为：

```math
M_{\mathrm{explicit}}^{(\mathrm{token,layer})}
= 2H_{\mathrm{kv}}d_hb.
```

这里的 $H_{\mathrm{kv}}$ 已经把 MHA、GQA、MQA 的 head/group 差异纳入记账；公式只数 K/V 元素，不包含页表、对齐、临时工作区或服务运行时元数据。完整的显式 cache 容量会随着层数、同时运行的请求和每个请求保留的 token 数继续增加。第 11 章从生成流程讲解 KV cache，第 69 章则把普通 KV payload 放入混合模型的逐层账本。

### 70.1.2 一个共享 latent 联合承载 key 与 value 信息

Multi-Head Latent Attention（MLA）让模型先把当前位置的 hidden state 投影到较窄的共同表示，再由这个表示生成 key 与 value：

```math
c_t^{KV}=W^{DKV}h_t,\qquad
k_t^{C}=W^{UK}c_t^{KV},\qquad
v_t^{C}=W^{UV}c_t^{KV}.
```

若 $h_t\in\mathbb{R}^{d}$，则 $c_t^{KV}\in\mathbb{R}^{d_c}$；$W^{DKV}$ 把 hidden state 下投影到 latent 空间，$W^{UK}$ 与 $W^{UV}$ 分别把同一个 latent 映射到 key/value 所需的表示。MLA 的关键不是先算出完整 K/V 再用外部压缩器编码，而是让注意力层围绕这个较窄的表示工作，推理时为每个位置缓存 $c_t^{KV}$。

DeepSeek-V2 论文将它称为 key/value joint compression。论文的 cache 公式、矩阵形状和具体模型中的位置分支见[原始报告](https://arxiv.org/html/2405.04434v5#sec_2.1.2)；[第 63 章](63-gated-mla门控潜变量注意力.md)进一步讲解 MLA 的计算和 Kimi K3 的 Gated MLA 配置。本章只取理解“每个位置存什么、之后怎样读”的最小机制。

### 70.1.3 每个位置仍可寻址，缓存仍随长度增长

Latent cache 不是把前文汇总成一份固定大小的摘要。位置 $s$ 有自己的 $c_s^{KV}$，当前 query 仍可以对各个历史位置计算 attention 权重；压缩的是每个位置的表示宽度，而不是历史位置的数量。因此，在层数和 latent 宽度固定时，cache 随上下文长度 $T$ 仍近似按 $O(Td_c)$ 增长。

这点也把 MLA 与递归序列状态分开：递归层逐 token 更新一个状态，通常不保留每个历史位置的独立地址。两者都可能降低某种 KV 存储压力，却保留了不同的读取能力和误差来源。[第 69 章](69-显式kv-cache在混合架构中的角色.md)比较了这些状态类型；[第 71 章](71-recursive-state-cache递归状态缓存.md)负责递归状态的更新与生命周期。

## 70.2 Attention 怎样直接读取 latent

### 70.2.1 把 key 上投影折入 query 路径

如果先从缓存 latent 重建每个历史 key，再与 query 做点积，仍能得到标准 attention 的分数，但会重新付出构造整段 K 的计算和数据搬运。MLA 的低秩结构允许改变乘法次序。对第 $i$ 个 head，$W^{UK}_i$ 表示 key 上投影中属于该 head 的部分：

```math
q_{t,i}^{\mathsf T}k_{s,i}^{C}
= q_{t,i}^{\mathsf T}W^{UK}_ic_s^{KV}
= \left((W^{UK}_i)^{\mathsf T}q_{t,i}\right)^{\mathsf T}c_s^{KV}.
```

于是可以先把当前 query 映射到与 latent 对齐的空间，再与已缓存的各位置 latent 打分，而不必为了求分数先展开每个 $k_{s,i}^{C}$。这是矩阵结合律给出的代数重排，不是近似注意力；[DeepSeek-V2 §2.1.2](https://arxiv.org/html/2405.04434v5)把这种重排称为将上投影吸收到 query 路径。

代数上允许吸收不等于所有 kernel 都以同一种方式执行。具体实现可能融合投影、分块读取或在某处物化中间量；若实现仍重建整段 K，缓存载荷减少未必能按相同比例减少读取流量或延迟。[Lior Sinai 的 MLA 讲解与 Julia 实现](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html)专门分析了未吸收投影时的 decode 代价以及吸收后的计算路径。

### 70.2.2 把 value 上投影与输出投影合并

注意力权重确定后，每个 head 要对历史 value 做加权求和。由于 value 是 latent 的线性上投影，可以先对 latent 按该 head 的权重求和，再把结果映射回 value 空间：

```math
o_{t,i}
= \sum_{s\le t} a_{t,s,i}W^{UV}_ic_s^{KV}
= W^{UV}_i\sum_{s\le t}a_{t,s,i}c_s^{KV}.
```

随后还可以把 value 上投影与 attention 输出投影合并。不同 head 的权重一般不同，所以仍须为各 head 形成相应的加权 latent 结果；合并投影并不会把所有 head 的 attention 变成同一组权重。DeepSeek-V2 §2.1.2 给出的吸收路径避免必须生成完整历史 V 张量，同时保留原有的加权汇总关系。

这也说明低维 cache 不能被理解为“读一个摘要向量就结束”。每个 query 仍要从它能访问的位置中算出权重、汇总内容，再投影为 attention 输出；如果要讨论推理速度，必须进一步看实现如何组织这些操作。

### 70.2.3 RoPE 位置路径影响 latent 定义与字节数

RoPE 会按 token 位置旋转 query/key 的通道。旋转与低秩投影通常不能随意交换，因此“缓存 latent”还要说明位置如何进入 key。DeepSeek-V2 将内容相关的 K/V latent 与共享的 RoPE key 分开；推理缓存包含这两类逐位置信息，不能只把 $d_c$ 当作它的完整 cache 宽度。按论文定义，MLA 每 token、每层需计入 $d_c+d_h^R$ 个元素，其中 $d_h^R$ 是额外位置 key 的维度。

并非所有 latent attention 都使用同一位置方案。[第 63 章](63-gated-mla门控潜变量注意力.md)区分 DeepSeek-V2 的 decoupled RoPE 与 Kimi K3 报告中的 NoPE 配置；后者是具体模型设计，不是 MLA 的通用定义。另一条路线是 KV-Latent：它缩减既有注意力头维度，并针对低维 query/key 修改 RoPE 频率采样。其作者在 [ACL 2025 论文](https://aclanthology.org/2025.acl-long.77/)中报告了特定模型上的继续训练和评测结果，不能据此推断任意模型都可直接采用同样的频率表。

### 70.2.4 预训练联合学习与 checkpoint 缩维是两种路径

DeepSeek-V2 式 MLA 把 latent 投影和模型其他参数一起训练，让网络学习哪些维度适合未来的 attention 读写。KV-Latent 则从已有 LLaMA checkpoint 出发，缩减 Q/K 与 V/O 的 head 维度，再用两阶段训练恢复能力。两者都能减少 cache 载荷，却不是同一个压缩算法：参数初始化、位置处理、训练信号和压缩误差都不同。

KV-Latent 的两阶段方案包括层内蒸馏与继续语言建模；当 latent/head 维度继续变窄时，少量追加训练未必能恢复原能力。这类结果提醒我们，表示维度必须结合训练过程和任务要求评价，而不能只看张量 shape。[KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding](https://aclanthology.org/2025.acl-long.77/)给出了方法和实验细节。

## 70.3 不同压缩方法改变不同的轴

“压缩 KV cache”可能表示改变每个数的位宽、每个位置的表示维度、保留的 token 集合，也可能表示不再保存逐 token 历史。它们可以组合，但应按改变对象分别记账：

| 方法 | 主要改变什么 | 历史是否仍按位置寻址 |
| --- | --- | --- |
| Latent / 低秩表示 | 每个 token 存储的特征维度 | 是，通常每位置保留 latent |
| MQA/GQA | 独立 K/V head 或 group 数 | 是，按 KV 组存储 |
| 量化 | 每个数值占用的 bit 数及数值误差 | 是，向量位置和维度仍在 |
| Token selection / eviction | 被读取或保留的历史位置集合 | 取决于是否实际删除位置 |
| 递归状态 | 用逐步更新的状态承接前缀历史 | 否，不为所有历史位置保留独立地址 |

### 70.3.1 MQA/GQA 共享 head 与 latent 压缩维度

MQA/GQA 减少独立 key/value 表示的组数；latent cache 则把 K/V 的信息投到较窄的坐标空间。前者改变的是 head/group 轴，后者改变的是每个位置的表示方式和宽度。两者有可能组合，因此比较模型时要同时记录 KV group 数、latent 宽度、每值精度和任何额外位置分量。

不能把两种方法简单写成彼此替代的“cache 压缩倍数”。GQA 中某个 K/V 组仍保存其向量；MLA latent 中每个位置也仍有独立 code。它们减少的是不同的乘数，实际能否叠加以及叠加后的模型质量要由具体结构和训练验证。[第 4 章](04-mha-mqa-gqa-mla的演进与trade-off.md)讨论 MHA/MQA/GQA/MLA 的历史关系。

### 70.3.2 量化减少每个数值的位宽

量化通常保留相同的向量维度和 token 位置，只用较少 bit 表示每个数值；由此引入的误差与 latent 降维导致的信息容量限制不是一回事。量化还可能需要 scale、反量化和适配 kernel，这些都会影响实际字节数或推理代价。

两者可以串接，例如先把每 token 的 latent 维度减小，再降低 latent 数值精度。但组合后的误差不能当成两个独立数字机械相乘，必须在目标模型和任务上一起测。具体 KV 量化格式和 runtime 支持由第二十四册处理。

### 70.3.3 Token selection 与 eviction 改变位置集合

稀疏读取让 query 只计算部分历史位置的 attention；淘汰则实际不再保留某些历史表示。前者可能减少计算，却不必然减少已分配的 cache；后者减少存储，但会让被删除的位置无法直接回读。两种情况都与“每个保留位置变窄”不同。

SALS 将两种轴结合：它先把 pre-RoPE 多头 key 投到低秩 latent 空间，再在 latent 空间选出 token，只重建入选 key，并对 value 使用通道量化。该论文用特定模型和任务测试这种组合；它说明维度压缩与 token 选择能够协同，却不表示每种 latent cache 都只保存选中的 token。[SALS 正式论文](https://papers.neurips.cc/paper_files/paper/2025/hash/00a0ebcad584c59dbc439c2af8793638-Abstract-Conference.html)展开了它的稀疏选择机制；本书的第 26 章及第 74、77 章承担稀疏与压缩 attention 的详细讨论。

### 70.3.4 递归状态把逐 token 历史改成时间更新

递归状态随着新 token 到来而更新，空间通常由模型状态维度决定；它不保留可直接索引的每个历史位置。Latent cache 则仍保留位置集合，只把每个位置的 K/V 表示变窄。因此，两者可能都降低显式 KV 载荷，却不能把 token latent 称为“固定大小记忆”，也不能把递归状态看成“把所有 KV 向量压缩成一长串 token latent”。[第 71 章](71-recursive-state-cache递归状态缓存.md)讨论 state 的表示与生命周期。

## 70.4 表示容量的瓶颈必须用任务保真来判断

### 70.4.1 信息瓶颈是容量直觉，不是 MLA 的通用训练目标

较窄的 latent 给每个位置设置了表示预算：经由这个 code，后续 query 能使用的信息受投影结构和维度约束。从这个角度说它形成了“瓶颈”，但这只是描述容量受限的直观说法。[经典 Information Bottleneck 方法](https://arxiv.org/abs/physics/0004057)另有明确的信息论目标；DeepSeek-V2 的 MLA 章节与 KV-Latent 的方法定义并没有把该互信息目标列为通用 cache 压缩训练目标。

MLA 也不要求逐维重建原 hidden state 或原始 full-rank K/V。端到端训练的模型可以在 latent 空间中学习有用表示，而 checkpoint 后处理方案则要面对原模型能力能否恢复的问题。评估时应问“目标任务还能否正确读取所需信息”，而不是只问“latent 能否还原某个中间张量”。

### 70.4.2 Key 与 Value 可以采用不同的维度预算

Key 参与 query-key 匹配，影响哪些历史位置获得较大权重；Value 提供按这些权重汇总的内容。改变二者维度的直接后果不同，因此某个模型可以分别分配 key 和 value 的预算。KV-Latent 研究了 $d_{qk}$ 与 $d_{vo}$ 的不同组合，并报告其测试中的性能对 $d_{vo}$ 更敏感。

这个发现不能提升为“Value 总比 Key 重要”或“Key 总能压得更多”的普遍规则。它来自特定 checkpoint、训练预算和基准；在其他模型中，关键证据可能正好需要精确的 key 匹配，或者模型能从上游状态中补偿部分误差。实际设计应通过分层消融来验证。

### 70.4.3 按信息类型、距离和干扰测量保真

单一的长上下文平均分容易掩盖 cache 压缩对不同信息的影响。一个面向长文本问答的评测可以分别检查精确实体、数字、日期、版本、否定词、相互冲突的事实、跨段组合与同义改写；再改变事实的位置、距 query 的 token 数、干扰内容和上下文长度。代码或检索场景也应加入目标任务常见的标识符、路径或精确字符串。

还要分清中间代理和最终任务结果。若方案从一个已训练 checkpoint 转换而来，可以比较 attention logits、attention 输出或教师模型蒸馏损失；这些分数有助于发现误差从哪一层产生，却不能代替最终问答、复制或生成任务。若 MLA 与模型一起训练，内部表示不必和原始 full K/V 数值相似，因而更应使用与读者目标一致的任务指标。

## 70.5 Cache 字节数、读取成本与任务质量需要一起比较

### 70.5.1 以 latent 和附加位置项计算逻辑 payload

若每个历史 token 缓存 $d_c$ 个 latent 元素，以及 $d_{\mathrm{pos}}$ 个位置相关元素，分别以 $b_c$ 与 $b_{\mathrm{pos}}$ 字节存储，则一层每 token 的 latent payload 为：

```math
M_{\mathrm{latent}}^{(\mathrm{token,layer})}
= d_cb_c+d_{\mathrm{pos}}b_{\mathrm{pos}}.
```

与同一配置下普通 K/V 对齐后，可写成逻辑 payload 比例：

```math
\rho_{\mathrm{payload}}
= \frac{d_cb_c+d_{\mathrm{pos}}b_{\mathrm{pos}}}
       {2H_{\mathrm{kv}}d_hb_{\mathrm{explicit}}}.
```

若各部分使用相同精度，$b_c=b_{\mathrm{pos}}=b_{\mathrm{explicit}}$，字节数中的 $b$ 可以约去。DeepSeek-V2 的位置项是 decoupled RoPE key，对应 $d_{\mathrm{pos}}=d_h^R$；没有独立位置分支的方案也应依据其公开结构填写，而不是套用该配置。总账还需乘以层数、请求数和实际保留的位置数，并将 scale、页元数据、对齐和工作区另行计入。

这个比例只说明给定形状下的逻辑载荷；它不包含质量损失、projection FLOPs、运行时暂存、通信或 kernel 效率。latent 每位置依然有记录，所以单纯减小 $d_c$ 不会把 $T$ 方向的线性增长消掉。

### 70.5.2 分开节省的带宽与新增的投影计算

Decode 时每产生一个 token，都要读取历史表示并计算可见位置的 attention。latent 更窄可以降低每步读取的数据量，但每个位置的 latent 还要参与 key 打分和值汇总；如果实现先完整重建所有 K/V，重建本身可能增加计算和内存流量。投影吸收、专用融合 kernel 或选择性重建会改变这笔成本，但不能只凭代数推导给出实测速度。

因此至少要分别观察 cache payload/峰值设备内存、实际内存流量、prefill 时间、time to first token、逐 token decode 延迟、吞吐和目标任务质量。短序列可能还未从节省的历史读取中获益，却已经承担额外映射成本；长序列则可能更受历史读取带宽约束。究竟何时交叉，要在目标模型、batch、精度、硬件和实现上测量，不能用“更少元素”直接推出统一的加速比例。

### 70.5.3 固定预算消融才能解释表示方案的收益

有意义的压缩比较需要固定或明确报告 checkpoint、训练数据与 token 数、上下文范围、精度、batch、硬件、baseline 和测量 harness。随后分开改变 latent width、key/value 维度、位置处理方式和 token selection，才能初步判断哪项变化带来收益或损失。如果模型同时改变 MoE、层结构、训练数据和 cache 表示，结果只支持这套配置的联合表现。

比较“同等 cache 字节下谁的质量更高”和“同等质量下谁占用更少”回答的是两个不同问题。两种口径都应报告完整任务指标与上下文长度，而不是只挑最好的压缩率、一个平均 benchmark 分数或一次峰值吞吐。

## 70.6 公开结果能证明什么

### 70.6.1 DeepSeek-V2 的整模型数字不等于 MLA 单因素吞吐

DeepSeek-V2 的架构章节支持 MLA 的 joint latent、key/value 上投影吸收和 decoupled RoPE cache 账本。论文摘要还报告，相比 DeepSeek 67B，DeepSeek-V2 的 KV cache 减少 93.3%，最大 generation throughput 达到 5.76 倍。但 DeepSeek-V2 同时改变了模型架构，采用 DeepSeekMoE，并包含自己的训练和部署设置；这些是整模型比较数字，不能据此说 MLA 单独带来同等幅度的 cache/吞吐收益。

### 70.6.2 KV-Latent 显示维度与任务、训练预算相关

ACL 2025 的 KV-Latent 在 LLaMA-3-8B GQA 配置中，将 $d_{qk}$ 与 $d_{vo}$ 从 128 降到 64，并按作者方案继续训练。其 Table 1 报告：4K 序列下 BF16 cache 从 491 MB 降到 245 MB；MMLU、OBQA 与 AI2-ARC 的平均分由 42.1 变为 41.3，论文所报 Needle-in-a-Haystack pass ratio 保持在 92%。当两种维度进一步降到 16 时，cache 为 64 MB、所报 NIAH pass ratio 为 6%。这组结果能说明该论文设置中存在质量与维数的关系；不能推出统一最低 rank，也不能把 3,840-token、随机插入短句、50 次试验的 NIAH 结果等同于所有长文档检索能力。

### 70.6.3 SALS 的结果属于低秩、选择与量化的组合

NeurIPS 2025 的 SALS 将低秩 key latent、token selection、入选 key reconstruction 和 value quantization 组合起来，并在 LLaMA2-7B-Chat、Mistral-7B-v0.2 及特定长上下文评测设置中报告质量与吞吐。作者指出，较短序列的重建开销会抵消一部分稀疏读取收益，较长序列下其特定配置的加速更明显。这个例子说明稀疏选择能与 latent compression 协作，也说明 throughput 不能归因于某一个轴；论文的硬件、模型、选择比例和 baseline 都是结果条件。

判断任何 latent cache 方案时，先问每个位置具体缓存什么、query 如何从中读取、位置编码是否另占空间；再问哪些任务经过了评估、拿什么 baseline 比、增加了哪些计算。压缩比是容量账本的一项输入，只有和任务保真及实际读取成本放在一起，才能支持工程选择。
