# 优化器与学习率策略精修证据

检索日期：2026-11-24。用于第五册第6章的正文事实核对。本文只记录本次检索得到的算法定义、接口语义和公开实验条件，不记录逐段审阅进度。

## 证据等级与边界

- **A：原始论文或正式会议论文。** 可支持论文定义、作者报告的实验结果和明确实验条件；不自动支持跨任务的普遍结论。
- **A-：官方框架文档。** 可支持指定版本的接口行为和状态加载约定；不支持其他框架或版本的默认行为。
- 本章的手算与教学代码属于解释工具，不是论文复现、框架兼容性保证或生产训练性能结论。

## 本次核对的主要证据

| 主题 | 来源与版本 | 可支持的正文主张 | 未验证边界 |
| --- | --- | --- | --- |
| Adafactor | Shazeer & Stern, arXiv:1804.04235 v1, 2018-04-11；[原论文](https://arxiv.org/abs/1804.04235) | 对矩阵参数用行/列统计估计二阶矩；原文也提出 update clipping、递增二阶衰减和相对步长。 | 论文的 WMT14 英德实验不能证明其在任意 LLM 训练上的质量或吞吐优势。 |
| Lion | Chen et al., arXiv:2302.06675 v4, 2023-05-08；[原论文](https://arxiv.org/abs/2302.06675) | 通过符号更新减少状态；作者指出它通常需要低于 Adam 的学习率，也观察到收益随 batch 变化，且一些任务收益小或不显著。 | 图像、对比学习和扩散结果不可直接变成语言模型配方；学习率/衰减不可从 AdamW 机械继承。 |
| Sophia | Liu et al., arXiv:2305.14342 v4, 2024-03-05；[原论文](https://arxiv.org/abs/2305.14342) | 使用对角曲率估计、动量和逐坐标裁剪；曲率估计间隔更新。作者在 125M–1.5B GPT 预训练设置报告相对 Adam 的最高约 2×效率。 | 加速是论文设置与评估口径下的作者结果；估计器、刷新频率、裁剪系数和模型规模改变时要重测。 |
| GaLore | Zhao et al., arXiv:2403.03507 v2, 2024-06-02；[原论文](https://arxiv.org/abs/2403.03507) | 双侧低秩子空间投影用于梯度与优化器状态，权重仍全量更新。作者报告在 LLaMA 1B/7B、C4、最多 19.7B tokens 下 optimizer-state memory 最多下降 65.5%；8-bit GaLore 的总训练内存结果绑定 BF16 baseline 与论文测量条件。 | 低秩不表示每个参数更新都是固定秩，也不表示任意梯度/阶段都低秩；峰值训练显存还包括基、梯度、临时量与激活。24GB RTX 4090 的可行性也仅是论文配方下的案例。 |
| Warmup 的架构与更新尺度 | Xiong et al., arXiv:2002.04745 v2, 2020-06-29；Kosson et al., arXiv:2410.23922 v1, 2024-10-31；[Xiong et al.](https://arxiv.org/abs/2002.04745)、[Kosson et al.](https://arxiv.org/abs/2410.23922) | Xiong 等人的 mean-field 分析与实验支持其所研究的 Pre-LN 可去除 warmup、Post-LN 初始化梯度较易造成不稳定；Kosson 等人在小规模 GPT/AdamW-Lion 设置中通过修改 update normalization 显著减少或消除 warmup 需求。 | 两者均不能证明任意生产规模架构可移除 warmup；后一研究还改变了 optimizer 的 update 归一化方式。 |
| WSD / WSD-S | Wen et al., arXiv:2410.05192 v3, 2024-12-02；[原论文](https://arxiv.org/abs/2410.05192)；MiniCPM, arXiv:2404.06395 v3, 2024-06-03；[MiniCPM](https://arxiv.org/abs/2404.06395) | WSD 先维持稳定主干，再从主干分出短程 decay 候选；WSD-S 将 cooldown 放在同一训练链后再升回稳定学习率。MiniCPM 在 36M 模型、40N/60N/80N 稳定训练检查点上报告约 10% token decay 足够而 2.5% 不足。River Valley 是在论文假设下解释曲线的一种模型。 | WSD-S / River Valley 论文的河谷解释带有假设；MiniCPM 的 decay 比例绑定其小模型与数据设置。训练集、模型规模、batch 和比例都限制外推。WSD 与 WSD-S 的 checkpoint 语义不能混同。 |
| Muon 大模型缩放 | Liu et al., arXiv:2502.16982 v1, 2025-02-24；[原论文](https://arxiv.org/abs/2502.16982) | 作者报告 compute-optimal scaling-law 对照下约使用 AdamW 52% 的训练 FLOPs；Moonlight MoE 的激活/总参数约 2.24B/15.29B，完整训练报告为 5.7T tokens。 | 约 52% 是 scaling-law 与该报告对照的估算，不是所有任务上的墙钟倍率；模型报告不可与小模型 token 效率等同。 |
| 矩阵优化器跨尺度超参迁移 | Qiu et al., NeurIPS 2025；[会议论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/bdcd9f6327db5877dee502cdec183159-Paper-Conference.pdf) | 论文报告 μP 学习率迁移、独立 weight decay 按宽度缩放后，Muon、SOAP、Shampoo 在 FineWeb、Llama 架构、190M–1.4B 设置下约 1.4× compute multiplier；实验 batch 为 128 条 1024-token 序列，约 0.12M tokens。 | 主要计算效率口径按 Transformer FLOPs 和验证损失估算，不应读成包含全部 optimizer 核、通信和墙钟开销的端到端倍速。 |
| 公平 optimizer 比较 | Wen et al., arXiv:2509.02046 v3, 2026-10-25；[论文版本](https://arxiv.org/abs/2509.02046)，后发表于 ICLR 2026 | 论文在 0.1B–1.2B、1–8× Chinchilla 数据预算下强调每个优化器需单独调参、应在预算终点比较；其矩阵优化器在 1.2B 设置报告约 1.1× 而非小模型约 1.4×。 | 仍是 C4/英文验证、Llama-2 架构及其训练预算/实现下的作者研究；不是跨数据、硬件或模型的最终排行榜。 |
| 大 batch 下 Muon/SOAP | Khona et al., arXiv:2607.20548 v1, 2026-07-13；[原论文](https://arxiv.org/abs/2607.20548) | 覆盖 8B dense、30B-A3B 和 72B-A8B MoE；基础 global batch 为 3072×8192，约 25M tokens。论文摘要报告实验中最高 100M-token batch；72B 图中 Muon 的对照为基础 batch 的 2×，约 50M。 | 作者结果绑定其 Megatron 系统、WSD、batch ramp、学习率缩放和模型配置；摘要的“最高 100M”不能和 72B 图中的 50M 对照混为同一实验。 |
| 跨阶段 optimizer 选择 | Liu et al., arXiv:2605.06654 v1, 2026-05-07；Zhao et al., arXiv:2610.22650 v1, 2026-10-15；[研究一](https://arxiv.org/abs/2605.06654)、[研究二](https://arxiv.org/abs/2610.22650) | 第一项在 GPT-2 与 Llama-2-7B 的特定 full-SFT/遗忘指标中报告同族 optimizer 的 Pareto 优势；第二项以 1.2B Llama-3 和最高 14B 开源模型比较 AdamW/Muon，报告匹配可减少遗忘，但最终 benchmark 的 optimizer 交互较弱，调优 LR 后继续预训练中的表观切换代价基本消失。 | 两者关注的任务阶段、模型、遗忘指标和最终 benchmark 不同，不能合并成 optimizer 切换普遍有害或普遍无代价的结论；第一项没有 Muon 预训练的 7B 对照。 |
| Optimizer checkpoint 加载 | PyTorch 2.15 官方文档；[`Optimizer.state_dict`](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Optimizer.state_dict.html)、[`Optimizer.load_state_dict`](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Optimizer.load_state_dict.html)、[`LRScheduler`](https://docs.pytorch.org/docs/2.15/generated/torch.optim.lr_scheduler.LRScheduler.html) | optimizer state 按参数组顺序与当前参数关联，默认不对名称做额外验证；scheduler 应先初始化，再加载 optimizer state，以免覆盖已恢复学习率。 | 仅限 PyTorch 2.15 对应接口；其他框架、分片优化器和未来版本需查各自文档并验证恢复轨迹。 |

## 正文修订采用的表达约束

1. 将论文中的倍率写成“作者在特定配置下报告”，同时保留模型、数据、预算和衡量单位。
2. 区分 token efficiency、估算 FLOPs、单步成本、吞吐、墙钟和质量门槛；不能互换。
3. 说明 warmup/scheduler 的时间基准与 checkpoint 分支；不把 WSD 与 WSD-S 的恢复点合并。
4. 不把 optimizer state 压缩等同于总峰值显存下降；不把矩阵 optimizer 描述为 AdamW 的通用替换。
5. 本轮未运行第6章教学代码，也未执行训练、测试或 PDF 构建。
