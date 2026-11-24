# 周期性 RoPE 与局部/全局位置路径研究记录

检索日期：2026-11-24。研究对象是 `P-RoPE`（Periodic RoPE）和 `HP-RoPE`（Harmonic Periodic RoPE），以及它们在 MiniWin 中与滑动窗口注意力、全局 NoPE 层的组合。这里的 `P-RoPE` 不等同于有些资料所称的 partial/ fractional RoPE；后者通常指只对 head 的一部分维度施加旋转。

## 核心机制与可支持主张

### RoPE 的位置差与窗口可见范围是不同对象

标准 RoPE 将查询与键按位置旋转，使注意力分数依赖两位置的相对偏移；因果滑动窗口注意力（SWA）则由 mask 限制每个 query 实际可读的位置。修改旋转索引不能自动扩展局部 mask。常规 RoPE 在训练长度之外的表现取决于模型和任务；“位置索引超出训练长度”是外推风险的描述，不应改写成所有模型在某一确定长度必然崩溃。

### P-RoPE 的周期索引与边界误差

Huo 与 Li 的 arXiv v2 将 P-RoPE 定义为在宽度为 $W$ 的因果滑动窗口内使用周期位置索引 $\varphi(t)=t\bmod W$，因此位置表只需包含 $W$ 个相位。对未跨周期边界的 query/key 对，编码偏移仍为真实间距 $d$；跨边界时编码为 $d-W$，而标准 RoPE 的相对旋转一般满足 $R_{d-W}\ne R_d$。只有所有频率均使 $W\theta_i$ 为 $2\pi$ 的整数倍时，边界额外旋转才为单位矩阵。P-RoPE 因而消除了无界增长的位置索引，却引入依赖窗口相位的旋转不连续；它本身不改变 mask 的可见集合。

### HP-RoPE 的边界条件

v2 以 $\theta_i'=2\pi m_i/W$、$m_i\in\mathbb{Z}$ 选取频率，使 $R_W=I$，从而 $R_{d-W}=R_d$，在旋转算子层面消除 P-RoPE 的 wrap mismatch。论文配置示例为 $W=256$、$d_h=96$、$n_p=d_h/2=48$，并令 $m_i=n_p-i$，即频率编号从 48 到 1。该代数性质只说明位置旋转在周期边界一致，不保证模型一定保留任意远距离事实，也不等同于更好的所有任务指标。

### MiniWin 的局部/全局组合

论文中的 MiniWin 将带 P-RoPE/HP-RoPE/标准 RoPE 的局部 SWA 层与不施加显式位置变换的全局 NoPE 层交错。v2 配置是 8 层、`SSSL × 2`（六个局部层、两个全局层）、$W=256$、$d_h=96$；变体间仅改变局部层的位置编码，NoPE 全局路径保持不变。该架构是理解实验的比较底座，不代表 P-RoPE 单独提供跨窗口读取能力。NoPE 不表示整网没有顺序线索；因果 mask 与前序 hidden state 都可能提供线索，具体能力依任务和训练而定。

全局 attention 仍访问完整可见前缀，故 P-RoPE 不会自动令模型的注意力计算、KV 存储或可用内存变成常数，也不能把“无旋转索引耗尽”解释为实际无限上下文。

### v2 实验及边界

v2 报告基于 MiniMind3 训练 MiniWin：长度 768、约 840 万文档（先留出 5%）、AdamW/bf16；另将约 10% packed tokens 用于 1536 长度的继续预训练。评测包括长度 512 至 16k 的 packed hold-out、约 76 万 token 且未用于训练的《西游记》连续文本，以及英/中文 format-free passkey（每长度 200 次试验）。比较模型使用同一架构、数据与训练日程，变化点是局部窗口层的 RoPE/P-RoPE/HP-RoPE；NTK、YaRN 等对照在测试期处理。

在 768 训练模型的 16k 评测中，论文报告：packed hold-out perplexity 为 RoPE 62.9、YaRN 20.7、P-RoPE 8.07、HP-RoPE 7.15；未见连续文本上分别为 390、421、118、113。英/中文 passkey top-1 中 P-RoPE 为 99%/100%，HP-RoPE 为 100%/99.5%；对应 RoPE 为 13%/14%，YaRN 的报告值最高为 22%/42%。这些是该小型模型和该评测协议下的作者报告结果，不是大模型、生产负载或所有长文本任务的普遍结论。

v2 还通过固定 Q/K/V 并移动周期表起点、以及跨边界双 needle 诊断，观察 P-RoPE 的 attention logits/分布受边界位置影响，而 HP-RoPE 在浮点误差范围内基本不变。这些诊断支持 wrap error 的机制解释，不能替代更广泛的端到端任务评测。论文结尾明确把更大模型和更广泛 benchmark 留作后续验证；截至检索日期是 arXiv v2 预印本，不能写成独立复现或既有大模型采用事实。

## 来源与证据边界

| 来源 | 本轮查阅内容 | 可支持 | 边界 |
| --- | --- | --- | --- |
| Huo & Li, [Periodic and Harmonic Periodic RoPE for Length Extrapolation in LLMs, arXiv v2](https://arxiv.org/html/2605.27980v2), 2026-11-19 | 方法 §§II-A—II-D；实验 §§III-A—III-C；结论 | P-RoPE/HP-RoPE 公式、MiniWin 结构、具体实验设置和作者报告的结果 | 预印本；仅支持文中模型/数据/任务/窗口条件，不能外推到更大模型或无限上下文理解 |
| [arXiv 版本历史](https://arxiv.org/abs/2605.27980) | v1/v2 日期与当前题名 | 当前应以 2026-11-19 的 v2 为本轮证据版本；v2 增加 HP-RoPE 和边界诊断 | v1 摘要不能代替 v2 方法和实验 |
| Huo & Li, [Cominder/miniwin](https://github.com/Cominder/miniwin), README | 项目介绍、模型与代码入口 | 作者关联项目将 MiniWin 定位为 P-RoPE、SWA、NoPE 的小型语言模型实现 | `main` 可变；README 不是训练结果的独立复现，不用于确认 v2 的 HP-RoPE 实验细节 |
| [SWAN-GPT: An Efficient and Scalable Approach for Long-Context Language Modeling](https://arxiv.org/abs/2504.08719) | 摘要、架构段和局部/全局 RoPE/NoPE 结构 | 说明局部 SWA-RoPE 与全局 NoPE 交错是相关研究路线，可作为架构前置 | 不包含本研究所述 P-RoPE/HP-RoPE，结果不可与 MiniWin 合并或互相归因 |
| Haviv et al., [Transformer Language Models without Positional Encodings Still Learn Positional Information](https://aclanthology.org/2022.findings-emnlp.99/), Findings EMNLP 2022 | 摘要与结论 | 因果 LM 可在特定训练条件下从因果可见性获得隐式位置信息 | 不证明每个 NoPE 模型都有充分位置能力或可无限长度泛化 |
| Kazemnejad et al., [The Impact of Positional Encoding on Length Generalization in Transformers](https://papers.neurips.cc/paper_files/paper/2023/file/4e85362c02172c0c6567ce593122d31c-Paper-Conference.pdf), NeurIPS 2023 | 摘要与实验总结 | NoPE/其他 PE 的长度泛化结果依赖下游任务与训练设置；不能只凭术语判断表现 | 其模型与任务不等于 MiniWin，也不能直接为其结果背书 |
| Dong, [Understanding RoPE (Rotary Position Embeddings)](https://simondong1.github.io/rope.html), 2026-06；Seyfi, [Position, part 2: RoPE in depth](https://mseyfi.github.io/llm-vlm/foundations/position-part-2-rope-in-depth-and.html) | RoPE 直观讲解、公式与相对位移说明 | 提供适合教学的 query/key 旋转、频率对和相对位置讲解线索 | 教学解释材料，不用于支撑 P-RoPE 的新公式、实验结果或普遍效果 |
| Khan et al., [Fractional Rotation, Full Potential? Investigating Performance and Convergence of Partial RoPE](https://arxiv.org/abs/2603.11611), arXiv v2 | 摘要/术语 | 确认另一条文献路线使用 partial RoPE 指只对 head 子维度旋转，有助区分名称 | 与周期索引 P-RoPE 是不同设计，不将其结论挪用到本章 |

本轮搜索词包括 `"Periodic RoPE"`, `"P-RoPE" sliding window`, `"harmonic periodic RoPE"`, `MiniWin P-RoPE NoPE`, `SWAN-GPT NoPE sliding window`, `partial RoPE` 与 `RoPE tutorial/lecture notes`。研究者博客和课程 notes 提供了 RoPE 直觉讲解入口；针对 2026 年 11 月新增的 HP-RoPE 和 wrap 诊断，直接证据仍来自论文 v2。
