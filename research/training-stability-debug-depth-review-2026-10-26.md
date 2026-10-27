# 第五册第九章：训练稳定性与 debug 深度精修研究记录

日期：2026-10-26  
正文：[`book-05-llm-training/chapters/09-训练稳定性与debug.md`](../book-05-llm-training/chapters/09-训练稳定性与debug.md)

## 研究问题与本轮范围

本轮从 `progress_v2.md` 的第五册第八章之后继续，精修第五册第九章；没有回头重做第四册或更早的正文。原章已有的健康基线、异常时间轴、loss spike、NaN/Inf、梯度过大/过小、数据与 mask、学习率和 optimizer state、混合精度、分布式同步、checkpoint 连续性、诊断 toy、复盘和练习均予保留。研究新增内容用于解释这些排查分支何时成立、不同研究方案改动了什么，以及各自证据的适用范围。

核心问题拆为：

1. 有限 loss spike、梯度峰值和 NaN/Inf 是哪些不同现象，如何找第一处变化？
2. Adam 不稳定和 spike 是否有多个可检验机制？不同论文的状态量、假设和证据如何区分？
3. clipping、降低 Adam `beta2`、SPAM、GradientStabilizer 各自修改什么，代价和失败条件是什么？
4. 当层间梯度/激活异常与归一化、残差路径和初始化有关时，Pre-LN、DeepNorm、Peri-LN 的问题设定和实验证据有何区别？
5. 教学诊断程序如何避免把告警阈值、候选坏 batch、健康 step 和可恢复 checkpoint 混为一谈？

## 读过的主要来源与正文支持范围

### Adam 时间相关梯度估计

- Molybog et al., [A Theory on Adam Instability in Large-Scale Machine Learning](https://arxiv.org/abs/2304.09871), 2023 预印本 v2；阅读 abstract、模型设置、时间相关更新分布、loss instability 机制及 mitigation discussion。
- 论文从 Adam 更新分量的时间相关结构解释大范数、与下降方向弱相关的更新；报告 7B、30B、65B、546B 运行观测。原文指出较小的 7B/30B 运行没有同样的爆炸式 loss divergence，较大模型中不同层的更新分布变化也不一致。
- 论文讨论的缓解路线包括降低 learning rate、调整 epsilon、减小 batch、改变 beta 参数和回滚/跳过 batch。每条都有成本：较低学习率增加训练时间；较小 epsilon 与低精度数值风险冲突；更小 batch 影响大规模数据并行效率；修改 Adam 平滑状态会改变正常训练时的信息响应。
- 正文把它作为研究提出的一种机制和一组实验观察，没有写成所有 spike 的统一根因。研究提出的架构、数据、算法因素未被彻底分离，且文章本身指出仍需要不同优化算法的相似大规模对照。

### Adaptive-preconditioner decoupling

- Bai et al., [Adaptive Preconditioners Trigger Loss Spikes in Adam](https://arxiv.org/abs/2506.04805v2), arXiv v2；阅读机制分析、gradient-directional curvature、五阶段分析、实验验证和 limits。
- 论文研究 Adam 二阶矩 `v_t` 未能跟上当前 `g_t^2` 的情况；作者描述梯度骤降后 `v_t` 可近似按 EMA 自身衰减，而后续梯度上升时两者脱节，预条件 Hessian 的最大特征值可能在 `2/eta` 阈值上方持续，从而出现 spike。
- 小模型验证为约 10M 参数、8 层 Transformer、合成推理数据 900k 序列、batch 2048，观察到七次 spike。另一实验使用 187M 参数的 LLaMA 结构模型，在 SlimPajama 上训练 100B tokens；比较 beta2 0.999、0.9、0.85，论文报告较低 beta2 减少 spike 频率。
- 论文用方向曲率量 `g^T Hhat g / ||g||^2` 预测 spike；stochastic mini-batch 会扰动预测，作者用持续越阈值的量作改进。Hessian-vector product 的规模化开销仍明显，论文也指出其严格理论是一维二次情形，高维非凸迁移基于实验而非形式证明。
- 正文用一维二次函数做手算：`eta=0.001`、`h=2400` 时 `eta*h=2.4`，梯度下降后的损失比为 `(1-2.4)^2=1.96`。这只解释稳定阈值直觉，不声称复现真实 Adam spike。

### SPAM

- Huang et al., [SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training](https://arxiv.org/abs/2501.06842v2), arXiv v2；阅读 GSS 定义、Adam 在线代理、momentum reset、实验/消融、模型配置和运行时间表。
- 论文回顾性 GSS 按某坐标跨训练步的梯度幅值历史归一化；完整历史不能在线保存，因此实现用 Adam 二阶矩构造 `g_i^2 / V_i` 近似，并对命中坐标缩放为 `sign(g_i)*sqrt(theta*V_i)`。两者不是相同定义，阈值不可以直接互换。
- Momentum Reset 周期重置 Adam 的一阶、二阶状态，并在重置后做 cosine warmup；论文 LLM 预训练设置示例为 `DeltaT=500`、warmup 150 步、theta=5000。消融中重置间隔过短或检测阈值过低会伤害质量，显示“清除异常状态”可能连同有用历史/真实梯度一起丢弃。
- 主表在 C4 LLaMA 预训练中报告 PPL：60M Adam 34.09、SPAM 30.46；1B Adam 16.13、SPAM 14.66。该结果绑定作者的模型、数据、调参和评测设置。论文主表与附录对 60M 训练 token 数有 1.1B/1.3B 的不同表述，正文不引用该 token 数。
- 单 H100 上平均 100 次迭代，60M Adam 0.3666 秒/step、SPAM 0.3814 秒/step；该数值是作者特定实现的运行测量，不代表其他硬件/框架的开销。

### GradientStabilizer

- Huang et al., [GradientStabilizer: Fix the Norm, Not the Gradient](https://proceedings.mlr.press/v306/huang26bb.html), ICML 2026/PMLR 306；阅读正式论文页面摘要与方法定位。
- 方法保留即时梯度方向，用运行梯度范数统计替换/稳定更新幅值。论文证明其幅值在 spike step 有界，并报告 LLM FP16 预训练、FP4 QAT 及其他任务中的稳定性改善和更宽稳定学习率区域。
- 正文没有凭摘要添加未核实的额外提升数字。方向保留也意味着错误方向不会因此变正确；target quality、throughput、cost 仍需本地同口径评测。

### 归一化、残差和深层信号传播

- Xiong et al., [On Layer Normalization in the Transformer Architecture](https://proceedings.mlr.press/v119/xiong20b.html), ICML 2020：mean-field 分析 Post-LN 初始化时靠近输出层的较大梯度，以及 Pre-LN 较平稳梯度路径；作者在机器翻译和 BERT 设置中验证。正文限定为论文设置，不保证任意 decoder-only recipe 可免 warmup。
- Wang et al., [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555)：DeepNorm 联合设计残差尺度与初始化尺度；论文包含 1,000 层实验和多语翻译任务上的深度/参数规模对照。正文指出这不是可摘取系数应用到训练中 checkpoint 的热修复。
- Kim et al., [Peri-LN: Revisiting Normalization Layer in the Transformer Architecture](https://proceedings.mlr.press/v267/kim25u.html), ICML 2025/PMLR 267：分析外围归一化位置及激活方差、梯度传播，报告最高 3.2B Transformer 实验。与 Pre-LN、DeepNorm 的模型和训练 protocol 不同，不将结果作直接排名。
- 正文将架构路线放在观测到层间梯度/激活系统性偏移后的下一轮实验中；对现有 checkpoint 先查数据、mask、归一化实现和统计口径，避免把架构改动误作单步 debug 开关。

### PyTorch 2.14 框架语义

- [autograd anomaly detection](https://docs.pytorch.org/docs/2.14/autograd.html)：支持在 backward NaN 时报告相应 forward 操作 traceback；它是慢速诊断工具。
- [`clip_grad_norm_`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html)：`error_if_nonfinite` 可检测总范数异常，但不能定位第一个坏 tensor。
- [AMP](https://docs.pytorch.org/docs/2.14/amp.html)：核对 scaler/unscale 流程；[FSDP](https://docs.pytorch.org/docs/2.14/fsdp.html)要求按 sharded gradient 语义使用自身 clipping 接口。
- [DDP 内部机制](https://docs.pytorch.org/docs/2.14/notes/ddp.html)：用于 rank reducer/collective 一致性解释。
- [Distributed Checkpoint](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html)：核对分布式状态 save/load 接口；临时目录、manifest/checksum 验证和版本指针的原子发布是应用级流程，文档接口本身不保证该事务语义。

以上框架文档链接均于 2026-10-26 按 PyTorch 2.14 固定版本打开核验；链接可达仅记录访问情况，不代表目标集群已完成实测。

## 正文修改与原知识点去向

1. 保留原有训练基线、时间轴和证据链；补代表研究后仍以“先定位第一处异常，再用反事实检验”的诊断步骤为主线。
2. 在 loss spike 处加入 Adam-instability、adaptive-preconditioner decoupling、SPAM、GradientStabilizer 的机制/证据/局限表，并加入 global norm clipping 工程基线，明确方案处理层次不同。
3. 增加方向曲率公式和一维手算，说明 `2/eta` 的直觉、所需非零梯度条件及 HVP 成本；补 SPAM 在线 proxy 的零二阶矩处理要求、参数消融和实测 step time。
4. 新增 Pre-LN、DeepNorm、Peri-LN 比较，提供层间 activation/gradient/update 指标；保留架构与训练中间状态的区别。
5. 保留 NaN/Inf、梯度爆炸/消失、mask、loss reduction、learning rate、optimizer resume、AMP、DDP 和 checkpoint 主题；为 PyTorch API 标版本并厘清 API 检测能力、FSDP clipping、rank collective 和应用原子发布边界。
6. 诊断代码保留纯标准库 toy 形式；把异常 batch 改成“候选”，将已知 healthy step 与可验证完整 checkpoint 分离；检查阈值域、结构对齐、rank key、checkpoint 必需字段和 bool/负梯度范数等输入边界。
7. 保留原七项练习，新增对照 Adam spike 理论、spike 缓解方案和归一化架构的三项研究型练习；已同步面试题、论文索引、术语表与练习索引。

## 验证与剩余边界

- 第 9.12 节标准库 toy 程序成功执行；step 4 spike ratio 为 2.66，step 6 nonfinite，step 4 grad norm 7.5，step 4/6 为异常 batch 候选，step 6 rank mismatch，缺失 checkpoint fields 为 data position/RNG/scaler/scheduler，last healthy step 为 5，但完整 rollback checkpoint 为 `None`。
- §9.12 代码正常路径通过；另拒绝 6 类无效输入：bool step、bool checkpoint step、空 history、负 gradient norm、NaN ratio 和少于两 rank 的 rank-loss 列表。手算边界 `(1 - 0.001×2400)^2 = 1.96`。
- 结构核验：`9.0—9.16` 标题连续，62 个 fenced markers 成对，18 个 math blocks 花括号和环境配对通过，10 项练习存在，唯一内部链接目标存在；`git diff --check` 通过。面试题、练习、论文索引与术语表同步已核对。
- 未运行真实模型、GPU、HVP 曲率跟踪、分布式集群、FSDP/DDP 故障注入或 checkpoint 事故演练。论文数字是作者报告，toy 代码不是论文复现。
- 不同研究的参数规模、token 量、数据集、optimizer recipe 和指标不同；正文避免跨研究算统一排名。SPAM 主表与附录训练 token 计数差异按缺口留档。
