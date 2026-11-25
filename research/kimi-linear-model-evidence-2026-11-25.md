# Kimi Linear 模型级架构与评测证据

研究日期：2026-11-25

## 研究问题与检索范围

核对 Kimi Linear 的 3:1 KDA/全局 MLA 排布、token-mixing 与 MoE channel-mixing 的模块顺序、NoPE 选择、1.4T 架构对照、5.7T 发布 checkpoint、长上下文分数、cache 与速度主张。重点区分机制、模型层配置、内部受控比较、发布模型比较和 decode 测量口径。

## 主要来源

### Kimi Linear: An Expressive, Efficient Attention Architecture

- 来源：Kimi Team，arXiv:2510.26692 v2；首次提交 2025-10-30，v2 更新于 2025-11-01。
- 入口：[arXiv v2 HTML](https://arxiv.org/html/2510.26692v2)；[arXiv v2 PDF](https://arxiv.org/pdf/2510.26692v2)。
- 论文是模型定义与作者实验的主要一手来源。未将官方 README、第三方模型目录或课程幻灯片用作性能证据。

## 研究发现与证据位置

### 模型结构与参数

- 阅读位置：HTML §4，尤其 Figure 3 和 §4.1–4.2；PDF 第 5 页。
- Figure 3 说明每个 block 先进行 token mixing，再进行 MoE channel mixing；token mixing 堆叠 $N$ 层 KDA 后接 1 层 MLA，具体实现设 $N=3$。这是 Kimi Linear 的层安排，不是 KDA 定义或普遍最优比例。
- 48B Kimi Linear 配置为 48B 总参数、3B activated parameters。§5.4 说明对照中的 MLA、GDN-H 和 Kimi Linear 共享模型架构、参数量和训练设置；MoE 激活 256 个专家中的 8 个（包含一个 shared expert），首层是 dense。论文脚注的 activated-parameter 口径排除 embedding。
- §4.1 的 KDA 使用 $d_k=d_v=128$；$q/k/v$ 的具体 ShortConv、Swish、L2Norm、逐通道 decay 参数化和低秩 output gate 属于 Kimi Linear 版本事实。KDA 状态更新及变量职责由[2026-11-24 KDA 研究记录](kimi-delta-attention-kda-2026-11-24.md)和第 61 章承担，此记录不重复其机制推导。
- §4.2 表明 Kimi Linear 对全部全局 MLA 层使用 NoPE，并将位置感知/近期性建模的设计责任主要交给 KDA。此为作者对该模型分工的解释；NoPE 一般性质和 K3 的位置路径不由此外推，分别见第 64、88 章。

### 3:1 比例消融

- 阅读位置：HTML §5.2 “Hybrid ratio”，Table 1；PDF 第 6 页。
- 表中训练/验证 PPL（越低越好）：0:1 为 9.45/5.77，1:1 为 9.29/5.66，3:1 为 9.23/5.65，7:1 为 9.23/5.70，15:1 为 9.34/5.82。
- 在被测比例中，3:1 的验证 PPL 最低，且 1:1 接近；作者选择 3:1 作为质量/推理开销折中。表格只支持这组实验配置，不能推出其他模型、语料、任务或 runtime 下均以 3:1 最佳。

### 1.4T 同规模对照

- 阅读位置：HTML §5.4–5.5；§5.4.1 的训练 recipe；Table 3–5；PDF 第 9–11 页。
- MLA、GDN-H、Kimi Linear 对照共享参数规模和 training setup。主预训练采用 4,096-token context、共同 1.4T token 的 K2 pretraining corpus、MuonClip、WSD、learning rate $1.1\times10^{-3}$ 和 32M-token global batch；报告另称这些模型采用相同的 annealing schedule 和 long-context activation phase。该组是报告内较受控的模型比较，但 Kimi Linear 同时改变 KDA、hybrid ratio、NoPE 等设计，不能拆成单因素因果结论。
- Table 3 的 Pretrain MMLU-Pro：MLA 47.2，GDN-H 47.9，Kimi Linear 51.0。Table 5 的 128k long-context RULER：81.3、80.5、84.3；长上下文平均分别 52.2、51.2、54.5。Kimi Linear 并非每个任务均领先，例如 Table 5 的 LongBench V2 为 35.0，低于 MLA 的 36.1；Frames 58.8，低于 MLA 的 60.5。
- Table 4 是相同 SFT recipe 的另一阶段，不与 base pretraining 混排；存在反例，例如 LiveBench (Pass@1) Kimi Linear 45.2，低于 GDN-H 46.4；EvalPlus 为 61.0，低于 MLA 62.6 和 GDN-H 62.5。
- Figure 1(a) 将 MMLU-Pro@4k 与 RULER@128k 的性能/加速放在同一图中，RULER 标出 84.3 与 3.98× acceleration。该 acceleration 读数不能和 Figure 7 的 prefill、batch-one TPOT 或 Figure 1(b) 的更大 batch TPOT 混为同一指标。

### 5.7T 发布 checkpoint 与 Moonlight 比较

- 阅读位置：HTML §5.4.1 与 Appendix D；Table 8–9；PDF 第 9、17 页。
- 论文说明最终 released Kimi Linear checkpoint 扩展至 5.7T token，并支持最多 1M context；这与 1.4T matched architecture runs 是不同训练版本。
- Table 8 的 Kimi Linear Base 与 Moonlight Base 都训练 5.7T token，activated parameters 都为 3B，但 total parameters 分别为 48B 和 16B。该表是 release-model 比较，不是同总参数、只改 attention 的消融。
- Table 9 的 Moonlight-Instruct 对超过 8k context 的任务没有分数；Kimi Linear-Instruct 的 RULER@1M 为 94.8。不能把该分数当成 matched full-attention 1M 对照或所有 1M-context 任务的能力证明。
- 论文报告的是作者评测，不等同于独立复现；窗口支持、某项 RULER 得分和任意长文理解需要分开陈述。

### Cache 与推理速度

- 阅读位置：HTML Abstract、§5.6、Figure 1 与 Figure 7、§6.3；PDF 第 1、12–13、15 页。
- 3:1 表示 token-mixing 层中每四层有一层全局 MLA，论文摘要称长序列生成时 KV cache 使用量最多减少 75%。这个比例是 token-level attention cache 的结构直觉；不等于整模型显存减少 75%，也不计 KDA recurrent state、ShortConv state、MLA latent 宽度、metadata、workspace 和运行时 buffer。
- Figure 7 以相同 48B setting、相同 attention layer/head 数作对照，caption 明示 batch size = 1。论文报告 prefill 在 512k 为 2.3×、1M 为 2.9× 快于 MLA；decode 曲线也只适用于该图的 TPOT/长度/模型测量条件。Figure 7 图中标有 2.2× 与 1.8× 区域值，正文另将 1M decode 叙述为 6×；不把未完全说明的单个曲线标注解释成普遍吞吐数字。
- Figure 1(b) 另报告 1M 场景 6.3× TPOT（1.84 ms vs 11.48 ms），并把它联系到较低 cache 占用允许更大 batch。它与 Figure 7 的 batch-one 曲线属于不同测量口径，不能混在同一效率叙述里。
- 在报告设置中能找到 batch、48B model 和 sequence-length 信息；未找到足以复现服务性能所需的完整硬件/驱动/runtime/调度说明。因此正文保留作者报告的限定性数字，不声称目标系统可复现。

## 教学与章节边界

- 第 61 章：KDA 状态方程、delta correction、逐通道衰减、chunkwise 计算和固定状态边界。
- 第 63 章：Gated MLA 与 latent cache。
- 第 64 章：NoPE 的一般位置性质。
- 第 65 章：不同混合注意力结构的横向比较。
- 第 69 章：各层历史表示与 cache 账本。
- 第 76 章：Kimi Linear 将这些组件组成单一模型的案例、1.4T/5.7T 版本差异、质量和效率证据。
- 第 88 章：Kimi K3 全模型、KDA/Gated MLA 版本与其他组件边界。

本记录只存联网研究所得的事实、来源、版本、阅读位置和证据边界；本章改写决策、进度和验收状态记入 `progress/`。
