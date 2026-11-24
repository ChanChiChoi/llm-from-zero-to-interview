# Mamba 硬件感知扫描与效率证据

研究日期：2026-11-23。研究问题：核对 Mamba 的 Selective Scan 怎样避免输入依赖状态展开造成的大量 HBM 访问，训练反向的状态重计算承担什么作用，自回归时固定状态相对 KV cache 改变哪项容量，以及原论文 scan、端到端推理与显存结果分别绑定什么实验条件。

## 检索范围

- 读者起点：理解 Mamba 输入依赖状态更新与仿射前缀 scan（第二十一册第三十二章）。本记录限于原始 Mamba 的训练算子、GPU 内存路径、推理状态容量和论文效率实验；不评价当前 Mamba 版本、服务引擎、kernel 库或其他模型的现行速度。
- 检索日期与平台：2026-11-23，Web 搜索；arXiv 原始论文 HTML/PDF 页面和课程讲义 PDF。
- 搜索词：Mamba selective scan hardware-aware state expansion HBM SRAM kernel fusion recomputation；Mamba paper inference throughput A100 batch size memory benchmark; FlashAttention IO-aware tiling SRAM HBM; Mamba hardware-aware scan lecture notes.
- 筛选方法：以 Mamba 作者论文原文核对算法机制、复杂度和实验协议；以 FlashAttention 原论文核对注意力 IO 优化的对照含义；大学课程讲义仅用于检查组织顺序和示意图，不提供效率数据证据。没有运行代码、模型或 benchmark。

## 主要发现

1. Mamba 原论文指出，朴素递归展开的中间状态形状为 $(B,L,D,N)$，比输入/输出 $(B,L,D)$ 多状态维 $N$；朴素递归工作量记为 $O(BLDN)$，而卷积形式有 FFT 相关开销。输入依赖参数使系统失去共享固定卷积核，因此需要在 scan 路径中处理内容选择。
2. 硬件感知 scan 组合三种方法：kernel fusion 减少 HBM 往返；并行前缀 scan 处理递归的顺序依赖；backward recomputation 不把完整中间状态保存到 HBM，而从保留输入重算梯度所需状态。作者描述的路径是把 $\Delta,A,B,C$ 从 HBM 载入片上 SRAM，在 SRAM 中离散化、扫描并用 $C$ 读出，只将 $(B,L,D)$ 输出写回。原论文称这使 Selective SSM 层的内存需求与使用 FlashAttention 的 Transformer 层相当；这一单层结论不等于完整 Mamba 模型显存一定更低。
3. 原论文 §4.5 报告 scan 运算在状态维 $N=16$ 时，序列长度超过 2K 可快于作者当时所知的 FlashAttention-2 实现，并称相对普通 PyTorch scan 可达 20—40 倍；附录 D 报告长度 32K 时相对 attention scan 算子可达约 7 倍。Appendix E.5 的 scan 比较固定 A100 80GB PCIe、BF16、$D=1024,N=16$、batch 1，并明确不含注意力 QKV projection 或卷积模型的 kernel 构建等其他操作；不应把它写成完整模型推理加速。
4. 原论文端到端推理吞吐测试使用 A100 80GB PCIe，prompt 长度 2048、生成长度 128，batch 从 1 到 128；比较 Mamba 1.4B/6.9B 与 GPT-3 结构 Transformer 1.3B/6.7B，Transformer baseline 为 Hugging Face Transformers 标准实现，重复测量三次取平均。论文报告相似规模下 4—5 倍吞吐，但该值绑定这一硬件、batch、输入输出长度、模型和实现，不支持当前任意 serving 引擎或 workload 的普遍速度结论。
5. 论文全模型显存实验使用 125M 模型、序列长度 2048、A100 80GB，Transformer baseline 配置 torch.compile 与 FlashAttention-2。表中 Mamba 的显存略高，例如 batch 1 时 4.8GB 对 4.6GB，batch 16 时 23.1GB 对 20.7GB；因此“选择性 scan 层激活量与优化后的 Transformer 相当”不能扩展成“整个 Mamba 训练显存更低”。作者也指出显存随激活张量规模变化。
6. FlashAttention 原论文的 IO-aware exact attention 通过 tiling 减少 HBM 与 GPU SRAM 间读写，同时保持 dense attention 数学计算。它说明 wall-clock 优势可来自存储路径优化，不必改变架构或渐近配对复杂度；比较 Mamba 时须区分算法结构、内核 I/O 和模型级工作量。

## 来源与证据边界

| 来源 | 版本、阅读位置 | 支持内容 | 不支持的结论 / 使用限制 |
| --- | --- | --- | --- |
| Albert Gu、Tri Dao，[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752v2)，arXiv v2，2024-05-31 | §3.3.2、Appendix D 的状态扩展、HBM/SRAM、融合和重计算；§4.5、Appendix E.5 的 scan、端到端推理与训练显存设置 | 原论文算法描述、shape/复杂度记法、作者报告的实现及实验结果 | 不支持脱离 A100、baseline、batch/长度/精度、模型规模的通用速度或显存保证；不是 2026 年当前服务实现的比较 |
| Tri Dao 等，[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135v2)，arXiv v2，2022-06-23 | Abstract 与 IO-aware tiling 说明 | exact attention 可通过减少 HBM/SRAM 往返提升执行效率，作为 IO 优化对照 | 不支持其他 attention 实现的速度排名；论文不是 Mamba 对比基准 |
| Stanford CS277，[Lecture 15: State Space Models](https://web.stanford.edu/class/cs277/assets/lectures/L15.pdf)，课程讲义 | Mamba hardware parallelism、SRAM/DRAM 传输、kernel fusion 与 parallel scan 页 | 作为教学组织参考，检查从扫描并行性过渡到 memory hierarchy 和融合的讲解 | 讲义不是运行记录或独立性能证据；硬件数字与现行设备规格不在本文采用 |
| DAU AIDT Lab，[04 Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://aidtlab-dau.github.io/lectures/Vision%20Transformer/04%20Mamba%20-%20Linear-Time%20Sequence%20Modeling%20with%20Selective%20State%20Spaces.pdf) | §3.3 / hardware-aware algorithm slides | 辅助检查 kernel fusion、parallel scan、recomputation 的概念组合和图示组织 | 主要转述原论文，不作为独立证据或性能复现 |

## 未验证边界

- 未运行 Mamba、PyTorch scan、FlashAttention、GPU profiler 或原论文实验；正文证据状态为 source-backed，不表示本地复现。
- 原论文基准不是当前框架/内核版本的匹配基线；没有据此宣称当前 Mamba、Transformer 或服务 engine 的速度排序。
- 不测请求调度、状态迁移、p99、能耗或单位成功成本；完整 serving 与 workload 评测由第六册和第二十四册承接。
