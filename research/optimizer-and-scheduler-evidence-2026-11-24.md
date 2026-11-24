# 优化器与学习率调度证据（2026-11-24）

本文汇总优化器对照、矩阵预条件器、WSD 以及 PyTorch 状态恢复接口的公开证据。经验倍率只适用于来源明确给出的模型、数据、调参和计量方式。

## 优化器对照与矩阵预条件器

| 知识点 | 来源与证据 | 适用边界 |
| --- | --- | --- |
| ICLR 2026 对 11 种优化器的系统对照覆盖约 0.13B、0.3B、0.52B、1.2B Llama-2 架构模型、C4 英文验证集及 1–8 倍 Chinchilla 数据预算。逐优化器调参后，矩阵方法相对 AdamW 的报告收益随模型规模降低；1.2B、8 倍预算下约 1.1×。论文主要按达到同一 loss 所需 token 数衡量，并引用既有工作认为适当实现可将矩阵变换额外耗时控制在约 10%；这不是端到端墙钟加速的直接测量。中间 checkpoint 排名可随学习率衰减翻转。 | [Fantastic Pretraining Optimizers and Where to Find Them](https://proceedings.iclr.cc/paper_files/paper/2026/file/ea4f0a6ad7088c9b93f22d74c0c9e8c2-Paper-Conference.pdf)，ICLR 2026 论文，§1–3、Figures 1/4/6，复核于 2026-11-24。 | 数据、硬件、架构、超参搜索和主指标均绑定该实验；1.1×不能直接解释成运行时间提高 10%。论文不同部分报告的设置分别覆盖 0.1B–1.2B 及 1–16 倍数据预算，不能混为同一实验表。 |
| NeurIPS 2025 的超参迁移研究在 190M–1.4B Llama 模型、FineWeb、GPT-2 tokenizer、1024 context、128 序列 batch（约 0.12M token）下，使用 μP 学习率缩放与约 $1/D$ 的独立 weight decay；Muon、SOAP、Shampoo 报告约 1.4× compute multiplier。其效率计量统计 Transformer 前向/反向 FLOPs，不计优化器变换本身。错误缩放下收益迅速减小，1.4B Muon 对照可降到约 1.1× 以下。 | [Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales](https://proceedings.neurips.cc/paper_files/paper/2025/file/bdcd9f6327db5877dee502cdec183159-Paper-Conference.pdf)，NeurIPS 2025 论文，§4、Appendix H，复核于 2026-11-24。 | 报告的是给定 FLOPs 计量和小 token batch 下的缩放实验；不含优化器变换 FLOPs/端到端吞吐。这里的 $D$ 与论文定义的模型宽度/尺度有关，不应简化为任意训练配置的通用 decay 配方。 |
| NVIDIA 2026 报告覆盖 8B dense GPT、30B-A3B Transformer MoE 与 72B-A8B hybrid Mamba-Transformer MoE。论文摘要称测试 next-token global batch 至多 100M tokens；标准基线配置为 3072 个 8192-token 序列，约 25M token。72B Figure 2 的 Muon 对照使用 2× 基线，即约 50M token。大批量对照还涉及 batch ramp-up、平方根 LR scaling、WSD、update RMS 匹配及 Megatron layer-wise 分布式优化器。 | [SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales](https://arxiv.org/html/2607.20548)，arXiv:2607.20548v1，§5、Figure 2、Appendix C，复核于 2026-11-24。 | 100M 是论文扫描到的 batch 上限，不代表标准基线或所有架构都使用该 batch。稳定性结论绑定模型、数据、调度、Muon/SOAP 实现和系统配置；SOAP 的稳定性改动包括每步 QR/eigenbasis 更新及协方差修正，不应概括成未经修正的任意 SOAP 实现。 |

## 优化器切换与后训练证据

| 知识点 | 来源与证据 | 适用边界 |
| --- | --- | --- |
| 2026 年 5 月预印本报告：预训练与 SFT 使用相同 optimizer，在其受控设置下可改善遗忘与新任务表现的 tradeoff；但 Muon 用于推理类 SFT 时表现较差的案例也被报告。 | [Optimizer-Model Consistency: Full Finetuning with the Same Optimizer as Pretraining Forgets Less](https://arxiv.org/abs/2605.06654)，arXiv:2605.06654v1，摘要与作者实验，复核于 2026-11-24。 | 单篇预印本结果，受模型、SFT 数据和任务影响；不能推出跨阶段必须保持同一 optimizer。 |
| 2026 年 10 月 workshop 论文研究 AdamW/Muon 在预训练、继续预训练和 SFT 间切换：从头训练的 1.2B Llama-3 混合比例实验未发现显著的普遍 mismatch penalty；另在最多 14B 的 Llama/Qwen 开放权重模型上，预训练用 AdamW 的模型以 Muon 做一般 SFT 仍具竞争力。作者承认匹配预训练 optimizer 可减少部分遗忘，但报告这一点不必然转化为更好的最终 benchmark 表现。 | [The Ephemeral Cost of Switching Optimizers Across LLM Training Stages](https://arxiv.org/abs/2610.22650)，arXiv:2610.22650v1，2026-10-15，摘要，NeurIPS 2026 workshop，复核于 2026-11-24。 | 与前项的结论存在张力；比较需看各自任务、模型、数据、混合比例与遗忘定义。两篇均为公开论文/预印本证据，不足以形成跨阶段 optimizer 选择定律。 |

## PyTorch 2.15 接口语义

| 知识点 | 来源与证据 | 适用边界 |
| --- | --- | --- |
| `AdamW` 的 `state_dict` 保存按参数关联的优化器状态及参数组，不保存参数张量本身；载入时以组内顺序配对参数状态，默认不额外核验名称或对象语义。官方说明在创建学习率调度器之后加载 optimizer state，以避免覆盖已载入的学习率。 | [PyTorch AdamW 2.15 文档](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html)，在线文档，复核于 2026-11-24。 | 文档是 PyTorch 当前接口说明，不证明其他框架的恢复规则；参数组重排、模型变更或跨版本载入仍需显式映射与恢复对照。 |
| `clip_grad_norm_` 对所有参数梯度视为拼接后的单一向量计算总范数，并原地修改梯度。 | [PyTorch clip_grad_norm_ 2.15 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.clip_grad_norm_.html)，在线文档，复核于 2026-11-24。 | 分布式训练中的跨 rank 总范数是否成立，还取决于具体并行与归约实现；调用该接口本身不等于全局范数同步。 |
