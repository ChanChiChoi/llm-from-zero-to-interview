# 第 52 章 Chinchilla、PaLM 与 Compute-Optimal Training

第九章介绍了 loss 如何随模型、数据和计算规模变化。本章把 scaling 问题落到一个事前决策：如果预训练计算预算已经给定，应把预算分给更大的模型，还是更多训练 token？答案取决于优化目标和成本口径；“compute-optimal”不是脱离条件的最强模型。

## 52.1 Compute-optimal：先固定目标和预算

设 $N$ 表示模型参数量，$D$ 表示训练过程中处理的 token 数，$C_{\mathrm{budget}}$ 表示预训练计算预算。Compute-optimal 研究问的是：在这个预算内，怎样选择模型和训练长度，使目标预训练损失最低？用数学语言写，问题是

~~~math
\min_{N,D} L_{\mathrm{pretrain}}(N,D)
\quad \text{subject to} \quad
\widehat{C}_{\mathrm{train}}(N,D)\leq C_{\mathrm{budget}}.
~~~

这个目标锁定了优化对象：预训练 loss。它并不直接求解每项下游任务的最高分，也没有把指令微调、偏好优化、评测或在线服务成本都放进同一个预算。把“最优”写完整，至少要说明最小化什么、计算预算如何记账，以及模型和数据来自什么实验范围。

## 52.2 模型参数和训练 token 的预算关系

对 dense Transformer，常用的训练计算粗估是

~~~math
C_{\mathrm{train}}\approx 6ND.
~~~

这里的 $N$ 是参数量，$D$ 是训练中处理的 token exposure 总数，$C_{\mathrm{train}}$ 以 FLOPs 计。这个近似把每个 token 的前向和反向计算折算到参数规模上；Chinchilla 论文附录将详细的 embedding、attention、dense block 与 logits 计算同 $6ND$ 比较，认为在其模型范围里差异较小。[论文的 FLOPs 计算](https://arxiv.org/abs/2203.15556)

在预算固定时，增大 $N$ 通常意味着能处理的 $D$ 变少；缩小模型则能把更多预算用于训练 token。因而“更大的模型”与“训练更久”是资源分配的两端，而不是两个可以各自无限增加的旋钮。比如参数量减半、训练 token 数加倍，在这条粗略账本下总 FLOPs 大致不变，但所得 loss 是否更低仍要由实测 scaling 曲线判断。

$D$ 记录模型训练时处理了多少 token，不等于语料里有多少唯一文档，也不等于一次样本的上下文长度。重复一个 token 会增加训练 exposure，但不会创造同等数量的新文本信息。$6ND$ 也不是设备上的墙钟时间或功耗公式：attention 的序列长度项、重计算、通信和实际利用率都会改变真实计算；attention 的细分成本见[第十章](10-o-n2-attention的计算与显存瓶颈.md)。MoE 的总参数、激活参数与路由通信也不能直接塞进 dense 公式，见[第二十一章](21-MoE路线Switch-GShard-Mixtral-DeepSeek.md)和[第二十二章](22-Dense与MoE训练服务trade-off.md)。

## 52.3 Chinchilla：把同一算力从参数转向 token

Hoffmann 等人不是只比较两个已经训练好的大模型。他们先训练超过 400 个不同规模和训练长度的模型，覆盖约 $70\,\mathrm{M}$ 至 $16\,\mathrm{B}$ 参数、约 $5\,\mathrm{B}$ 至 $500\,\mathrm{B}$ 训练 token，再据此估计固定 FLOPs 下的 loss 前沿。实验结果显示，在其研究范围里，模型参数和训练 token 应随预算大致同步增加：模型规模加倍时，训练 token 也大致加倍。[原始论文](https://arxiv.org/abs/2203.15556)

研究随后把估计应用到 Gopher 规模的训练预算，并训练 Chinchilla 验证预测：

| 模型 | 参数量 | 训练 token | 这组数字说明什么 |
| --- | ---: | ---: | --- |
| Gopher | $280\,\mathrm{B}$ | 约 $300\,\mathrm{B}$ | 当时的大模型基准之一 |
| Chinchilla | $70\,\mathrm{B}$ | $1.4\,\mathrm{T}$ | 以更小模型和更多 token 使用相近训练计算预算 |

作者报告 Chinchilla 在大多数被测下游任务上超过 Gopher。这说明在相近训练预算下，投入更多 token 可以比单纯增大参数更有效。$70\,\mathrm{B}$ 参数配 $1.4\,\mathrm{T}$ token 对应的参数—训练 token 比（$D/N\approx20$）常被概括为约 $20:1$ tokens/parameter；它是该研究前沿上的一个配置，不是对所有模型、语料或目标生效的定律。

这组结果也不是只改变 $N$ 和 $D$ 的完全隔离对照。Chinchilla 与 Gopher 的数据、优化器等设置并不完全相同；Hoffmann 等人指出，昂贵的大规模预算仅有 Chinchilla 与 Gopher 两个可比较的训练点。因此，更稳妥的结论是：该研究用大量较小训练拟合出一个预算建议，并以大模型案例支持它；它没有证明任何新模型只需复制该比例就会达到相同效果。

## 52.4 PaLM：系统规模与预算配置是两类扩展

PaLM 展示了另一种“大规模”的含义：模型、数据和多设备训练系统如何一起扩展。论文报告 PaLM 是约 $540\,\mathrm{B}$ 参数的 dense 自回归 Transformer，在约 $780\,\mathrm{B}$ token 上训练，并使用 Pathways 跨两个 TPU v4 Pod 部署 $6{,}144$ 个芯片训练。[PaLM 原始论文](https://arxiv.org/abs/2204.02311)

PaLM 报告的系统利用率指标也说明了“效率”需要说清口径。其两项指标分别是 model-FLOPs utilization（$46.2\%$）和 hardware-FLOPs utilization（$57.8\%$）。前者只计模型有效计算，后者还把重计算（rematerialization）产生的 FLOPs 计入硬件实际执行量；两者都相对芯片理论峰值计算量报告。[PaLM 原始论文](https://arxiv.org/abs/2204.02311) 这两个数值描述训练系统把设备峰值转成计算工作的情况，并不回答固定预算下该选多少参数、训练多少 token。

因此，不能把 PaLM 的参数/token 组合直接当成 Chinchilla 的同预算对照。两项研究的预算、训练数据、模型规模和评测条件不同：Chinchilla 用来验证模型—数据预算前沿，PaLM 说明更大模型可以依靠多 Pod 系统进行训练。PaLM 的系统吞吐或下游 benchmark 成绩不能单独证明某个 token/parameter 比例普遍最优。

## 52.5 结论何时迁移：实验边界与目标变化

Compute-optimal 前沿是由实验估计出来的，成本和训练协议都会影响它。NeurIPS 2024 的复现研究在 OpenWebText2 与 RefinedWeb 上分析了 Kaplan 与 Chinchilla 估计差异，指出末层 FLOPs 是否计入、学习率 warmup 长度和按模型规模调节优化器都会改变前沿；在其设置中，修正这些因素后两条 scaling 趋势更接近。[复现研究](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b6341525cd84f3be0ef203e4d7cd8556-Abstract-Conference.html)。因此不能只背一个比例，还应检查参数定义、FLOPs 口径、训练长度与超参数如何选择。

数据池受限时，继续增加 $D$ 会包含更多重复 exposure；此时“更多 token”等于“更多新信息”的假设不再成立。2025 年的一项数据受限 scaling 研究单独建模唯一 token 与重复训练，并报告在其受控实验中，重复数据的边际价值会随重复次数下降。[JMLR 研究](https://www.jmlr.org/papers/volume26/24-1000/24-1000.pdf)这补充了 Chinchilla 的非重复数据范围，不能被压成一个跨语料通用的最佳 epoch 数。数据质量与 mixture 怎样改变 loss 曲线，见[第九章](09-transformer的并行性表达力和scaling优势.md)及第二十一册之外的数据工程专题。

最后要把预训练目标与后续目标分开。更低的预训练 loss 可能帮助多个下游任务，却不保证指令跟随、代码执行、安全表现或产品成本同步改善。若目标改成某类下游分数，或把推理调用成本纳入预算，就应重新定义优化问题和验证数据；dense 预训练估算也不能直接替代 MoE 的激活与通信账本。助手后训练的机制由[第五十三章](53-InstructGPT-ChatGPT-RLHF.md)继续展开。
