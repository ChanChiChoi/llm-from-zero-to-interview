# 第三十三章：Mamba 的训练、推理和硬件效率优势

[第三十二章](32-Mamba与Selective-Scan.md)说明，输入依赖的状态更新仍可由并行前缀 scan 计算。本章接着问：这条计算路径怎样映射到 GPU，固定状态又会改变长序列推理的哪项成本？答案不能只看“线性”或 state 大小。训练与自回归 decode 使用不同路径；算子速度、整模型吞吐和显存也是不同的测量对象。原论文的硬件与 benchmark 结果能说明一种实现如何获得优势，但每个数字都绑定了具体模型、序列长度、batch、数据类型、设备和对照实现。

## 33.1 复杂度、阶段与实际速度

当模型宽度、状态维度和 batch 固定时，Selective Scan 的工作量随序列长度近似线性增长。这个复杂度描述序列长度如何改变运算规模，不直接说明 GPU 墙钟时间。实际执行还要考虑内存数据搬运、kernel launch、并行度、batch 形状和模型权重读取。尤其是把短序列的理论算量与长序列的实际延迟放在一起比较，会掩盖不同瓶颈。

同一模型在不同阶段也走不同路径。训练需要处理整段序列及其梯度；整段 prompt 的 prefill 可以使用并行 scan；自回归 decode 每次收到一个新 token，只能根据前一步状态推进当前步。因此训练 tokens/s 不能代替 decode 的每 token 延迟，scan 算子的微基准也不能代替整模型吞吐。

GPU 上还存在速度和容量不同的存储层级。HBM 是 GPU 使用的高带宽显存；SRAM 是容量较小、位于芯片上的高速存储。算法即使只做线性数量的计算，如果反复把庞大的中间张量写回 HBM，再读回来继续处理，仍可能受内存带宽限制。Mamba 的 Selective Scan 设计同时考虑了递归的并行计算和这些数据搬运。[Mamba 原论文 §3.3](https://arxiv.org/abs/2312.00752v2)

## 33.2 训练路径：硬件感知 Selective Scan

对每个位置展开输入依赖状态时，离散系数与中间状态会涉及形状为 $(B,L,D,N)$ 的数据：$B$ 是 batch、$L$ 是序列长度、$D$ 是通道维度、$N$ 是状态维度。若把全部展开结果写入 HBM，状态维度会将序列张量放大，数据搬运成为主要成本之一。

Mamba 原论文把离散化、并行 scan 和状态读出融合起来：从 HBM 读取 $\Delta,A,B,C$ 等参数，在 SRAM 中形成离散系数、组合逐位置更新并由 $C_t$ 读出，最后只把模型宽度大小的输出写回 HBM。并行前缀 scan 处理 token 之间的递归依赖；融合则避免反复将扩展状态写出再读入。论文分析认为，相比先物化扩展张量再单独扫描的实现，这条路径可将相应的 HBM 读写量按状态维 $N$ 的量级减少。具体算法和硬件分析见[Mamba 原论文 §3.3、Appendix D](https://arxiv.org/abs/2312.00752v2)。

~~~mermaid
flowchart LR
  params["HBM 读取 u、Δ、A、B、C"] --> disc["SRAM 中离散化"]
  disc --> scan["SRAM 中并行 Selective Scan"]
  scan --> read["SRAM 中按 C_t 读出"]
  read --> output["HBM 写出序列输出"]
  saved["反向阶段读取已保存输入"] -.-> recompute["重算梯度所需中间状态"]
~~~

图中区分了前向扫描和反向重计算：扩展状态在融合路径的片上计算中生成并使用，反向阶段从保留的输入重建所需中间量。若序列太长，不能一次把相关计算安排在片上存储中，论文采用分块并延续块间状态。分块改变执行安排，不改变原序列的状态更新顺序。Selective Scan 的结合关系见前章；本章关注它怎样减少慢速存储之间的往返，不重推 scan 的代数。

## 33.3 训练内存：反向阶段重计算状态

计算梯度通常需要前向时的中间状态。朴素实现会把展开后的状态保存在 HBM 供反向传播读取；Mamba 选择保存必要输入，在反向阶段重新执行相应离散化和扫描，重建梯度计算所需的中间量。它用额外算术换取较少的中间张量读写，适用于 scan 这种容易受内存带宽限制的计算。重计算减少的是特定激活存储与搬运，不会消除参数、输入输出和其他激活。

因此，要区分 scan 层的激活估算与整模型训练峰值。原论文按混合精度的一种层级核算，估计 selective SSM 层每 token 保存约 16 字节激活；FlashAttention attention 层加相邻 MLP 层则约 32 字节。这个估算描述特定架构组件的保存需求，不含优化器状态等整模型成本。[Mamba 原论文 Appendix D](https://arxiv.org/abs/2312.00752v2)

原论文还报告了完整训练显存表：125M 模型、长度 2048、单张 A100 80GB，Transformer 对照使用 torch.compile 与 FlashAttention-2。batch 为 1 时，表中 Mamba 为 4.8GB、Transformer 为 4.6GB；batch 为 16 时分别为 23.1GB 与 20.7GB。该设置下两者显存处于相近量级，Transformer 的记录还略低。因此不能从 selective scan 单层的激活估算推出“整个 Mamba 模型训练显存必然更低”。[Mamba 原论文 Appendix E.5](https://arxiv.org/abs/2312.00752v2)

## 33.4 自回归推理：固定状态与 KV cache

attention decode 每一步要读取历史 KV；历史越长，每层缓存的内容越多。Mamba 每步则读取并更新各层的递归状态，还维护短卷积所需的局部缓存。以单请求、各层宽度一致、每个元素占 $b$ 字节作粗略容量估算：

~~~math
M_{\mathrm{KV}}\approx 2L_{\mathrm{layer}}T H_{\mathrm{kv}}d_hb,\qquad
M_{\mathrm{SSM}}\approx L_{\mathrm{layer}}D_{\mathrm{inner}}(N+k_c)b
~~~

这里 $L_{\mathrm{layer}}$ 是层数，$T$ 是历史 token 数，$H_{\mathrm{kv}}$ 是 KV head 数，$d_h$ 是每个 head 的宽度，$D_{\mathrm{inner}}$ 是 Mamba 的内层宽度，$N$ 是状态维，$k_c$ 是每通道的卷积缓存长度。公式只比较常驻的主要序列状态，未计权重、临时激活、对齐与分配器开销；实际模型也可能在不同层采用不同尺寸或精度。

在模型配置固定时，Mamba 的递归状态和卷积缓存不随 $T$ 线性增长；它们仍随层数、内层宽度和状态维增大。每生成一个 token，状态更新也需要读写相应状态。固定容量因此不能直接解释成所有设置下 Mamba 都更快或状态都更小。对于小 batch decode，权重读取、kernel 启动及 GPU 利用率可能主导速度；较长上下文下，不随 $T$ 扩大的状态容量则可能让同一设备容纳更多活动序列。Mamba 原论文将其较高推理吞吐部分归因于不用存储 KV cache，因而能够采用更大的 batch；这个归因对应作者的测量设置，不是对所有服务负载的保证。[Mamba 原论文 §4.5、Appendix E.5](https://arxiv.org/abs/2312.00752v2)

## 33.5 原论文效率结果与可迁移边界

同一篇论文分别测量了 scan 算子、端到端 decode 吞吐和训练显存。三者回答的问题不同，不能把一类数字当成另一类结论。

| 测量对象 | 论文设置 | 作者报告 | 适用边界 |
| --- | --- | --- | --- |
| Selective Scan 算子 | A100 80GB PCIe；BF16；batch 1；$D=1024,N=16$；长度从 512 增至约 500K；对照包含因果 FlashAttention-2 | 长度超过 2K 时快于作者当时采用的 FlashAttention-2；相对普通 PyTorch scan 报告最高约 20—40 倍；附录称长度 32K 时相对 attention 算子最高约 7 倍 | 不含 attention 的 QKV 投影、卷积核构建或其他模型运算，属于核心算子比较 |
| 端到端推理 | A100 80GB PCIe；prompt 2048、生成 128；batch 1—128；Mamba 1.4B/6.9B 对比 GPT-3 结构 Transformer 1.3B/6.7B；标准 Hugging Face Transformers 实现；重复三次取平均 | 论文报告相近模型规模下 Mamba 吞吐约为 Transformer 的 4—5 倍 | 结果绑定该设备、长度、batch 扫描、模型和对照实现；6.9B Mamba 示例未训练，不能推广为当代优化服务引擎的速度比例 |
| 完整训练显存 | 125M 模型；长度 2048；A100 80GB；Transformer 对照使用 torch.compile 与 FlashAttention-2 | 表中 Mamba 与优化 Transformer 显存相近，多个 batch 记录略高 | 只覆盖论文给出的模型与精度/实现设置，不代表所有大小、长度或训练配方 |

这些数字说明，硬件感知状态扫描可以让输入依赖递归在指定 GPU 和基准中高效执行；固定状态也会改变长上下文 decode 的容量曲线。它们不证明 Mamba 在任意 GPU、任何 batch 或当前推理引擎上都有相同优势。FlashAttention 的对照尤其提醒读者：精确 attention 仍可通过 IO-aware tiling 减少 HBM 与片上 SRAM 间的搬运，而保持 dense attention 计算语义；比较时必须写清对手的实现水平，而不是只比较“二次”与“线性”。[FlashAttention 原论文 v2](https://arxiv.org/abs/2205.14135v2)；其注意力内核的数据路径见[第二十章](20-FlashAttention与PagedAttention.md)。

本章的结论限于计算路径和成本口径：Select Scan 的并行性、融合和重计算共同决定训练效率；固定递归状态改变随上下文长度增长的常驻存储；性能数据仍需绑定测量对象、设备、模型和 baseline。模型能否精确利用远处信息及不同架构的任务表现分别由[第三十四章](34-Mamba局限与证据边界.md)和[第三十六章](36-SSM替代Transformer的证据与争议.md)承接；第六册和第二十四册负责完整服务引擎与部署性能评测。
