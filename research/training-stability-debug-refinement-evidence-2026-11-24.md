# 训练稳定性与 debug 证据：来源、版本与适用边界

本记录保存第五册训练稳定性章节所用研究事实、接口行为和证据范围。论文结果绑定各自的模型、数据、训练配置与评价定义；框架 API 事实绑定所列文档版本。

## Loss spike 与优化器研究

### Adam instability

- **来源与版本：** [A Theory on Adam Instability in Large-Scale Machine Learning](https://arxiv.org/abs/2304.09871)，Molybog 等，arXiv v2。核阅摘要、实验结果和讨论部分。
- **来源支持：** 论文讨论梯度估计的时间相关性与 Adam 更新范数/方向之间的关系；覆盖 7B、30B、65B 与 546B 模型，并报告较小规模实验未出现与最大规模相同程度的爆炸式 divergence。作者讨论学习率、$\epsilon$、batch size 和 $\beta_1/\beta_2$ 等缓解方向。
- **证据等级与边界：** source-backed（预印本的机制分析、特定规模实验和作者讨论）。该工作不证明所有 loss spike 都由 Adam 单一因素造成，也不提供跨模型通用超参数处方。

### Adaptive Preconditioners Trigger Loss Spikes in Adam

- **来源与版本：** [arXiv v2](https://arxiv.org/abs/2506.04805v2)，Bai 等，2026-05-25 修订。核阅 §§3–6、附录中的实验与证明说明。
- **来源支持：** 论文用二阶矩估计器与瞬时平方梯度的 decoupling 解释 Adam 中的 loss spike，并分析预条件 Hessian 与梯度方向曲率。合成推理任务采用约 10M 参数的 8 层 Transformer、900,000 条序列和 batch size 2048，报告 7 次 spike；另一项在 SlimPajama 上训练 187M 参数 LLaMA 结构模型、使用 100B tokens，并比较 $\beta_2=0.999,0.9,0.85$，较低 $\beta_2$ 在这些设置中降低 spike。
- **证据等级与边界：** source-backed（预印本的局部二次/线性稳定性分析及论文实验）。分析机制依赖论文简化假设；高维非凸模型中的观测属于实验支持，不是普遍的因果定理。
- **正文公式口径：** 沿普通固定学习率梯度下降的更新路径积分 Hessian 得到的一步损失条件，是章节明确标作普通梯度下降的精确微积分恒等式；它不应归因成 Adam 的完整稳定性条件。论文对 Adam 使用局部二次近似、预条件 Hessian 和梯度方向曲率代理，且该代理的估计需要二阶计算。

### SPAM

- **来源与版本：** [SPAM v2](https://arxiv.org/abs/2501.06842v2)，核阅方法、主实验、消融、时间开销与附录 D。
- **来源支持：** 论文区分回顾性 GSS（某坐标当前梯度绝对值相对该坐标跨训练步梯度均值的比值）与训练时在线代理 $g_i^2/V_i$；在线限幅为 $\operatorname{sign}(g_i)\sqrt{\theta V_i}$。C4 常规预训练配置采用 $\Delta T=500$、reset 后 warmup 150 steps、$\theta=5000$。主表报告 LLaMA-60M Adam/SPAM PPL 34.09/30.46，LLaMA-1B 为 16.13/14.66；60M 的主表 token 数是 1.1B，而附录 D 表 7 是 1.3B。单张 H100 的 100-iteration 测量中，60M Adam/SPAM 每步为 0.3666/0.3814 秒。
- **证据等级与边界：** source-backed（单一预印本版本的论文方法和作者报告实验）。在线代理不等同于回顾性 GSS；阈值、训练 token 数和运行成本不能脱离论文实现、数据与设备外推。60M 的 token 预算按原文报告保留差异。

### GradientStabilizer

- **来源与版本：** [PMLR 306:47732–47749, ICML 2026](https://proceedings.mlr.press/v306/huang26bb.html)。核阅正式论文摘要与出版信息。
- **来源支持：** 方法保持梯度方向，以运行统计替换/控制梯度范数；摘要报告了 LLM FP16 预训练、FP4 量化感知训练等设置中的稳定性及学习率范围结果。
- **证据等级与边界：** source-backed（同行评审论文摘要）。结果仍绑定论文实验和 recipe；稳定幅值不意味着梯度方向正确，也不能替代目标模型的质量、吞吐和成本评估。

### AdaGC

- **来源与版本：** [arXiv v3](https://arxiv.org/abs/2502.11034v3)，2026-06-09 修订；正式出版为 [ICML 2026, PMLR 306:127545–127572](https://proceedings.mlr.press/v306/wang26dg.html)。核阅方法、实验指标、表 5、消融和并行通信说明。
- **来源支持：** AdaGC 按每个参数张量的历史裁剪梯度范数 EMA 设动态阈值；默认前 100 步使用 GlobalGC warmup。论文的 spike score 是时间序列中偏离此前 1,000 个值滚动均值至少 10 个标准差的点所占比例。v3 表 5 报告：Llama-2 7B（9K steps）GlobalGC/AdaGC score 0.0333%/0；Mixtral 8×1B（36K）0.0144%/0；ERNIE 10B-A1.4B（21K）0.0100%/0。论文称计算逐张量局部范数时在 TP 组内 all-reduce，并与全 DP/TP/PP 全局范数收集作区分。
- **原文数值不一致：** 同一表将 Mixtral GlobalGC 记为 52 次 spike、36,000 steps、0.0144%。按文中 score 定义，以次数除以步数再乘 100%，重算为 0.1444%；表列值相差 10 倍。章节保留作者表列值并标出算术差异，不将表列值当作重算结果。
- **证据等级与边界：** source-backed（arXiv v3 完整正文及 ICML 正式出版信息）。结论绑定模型、训练步数、数据、benchmark 和并行策略；score 为零只表示没有观测点达到该判据。

## 归一化与深层信号传播

- **Pre-LN/Post-LN：** [Xiong 等，ICML 2020，arXiv v2](https://arxiv.org/abs/2002.04745v2)。mean-field 分析和所测翻译/BERT 设置支持 Post-LN 初始化时靠近输出层的期望梯度较大、Pre-LN 梯度传播更平稳并可在这些实验中移除 warmup。此证据不保证任意 decoder-only 训练都无需 warmup。
- **DeepNet/DeepNorm：** [Wang 等，2022](https://arxiv.org/abs/2203.00555)。论文报告在其定义的架构和初始化下扩展到 1,000 层；其多语言翻译实验比较 200 层 3.2B 与 48 层 12B 模型。结果不能支持把缩放系数直接迁移到任意已有 checkpoint。
- **Peri-LN：** [PMLR 267:30400–30436, ICML 2025](https://proceedings.mlr.press/v267/kim25u.html)。摘要报告最高 3.2B 参数实验中更平衡的 variance growth、steadier gradient flow 与 convergence stability；与 Pre-LN 或 DeepNorm 的实验条件并非完全相同。
- **证据等级与边界：** 均为原始论文 source-backed 证据。架构比较绑定论文中的深度、初始化、任务和 recipe；不能据此认定现有 checkpoint 可安全热切换归一化结构。

## PyTorch 2.15 诊断与恢复接口

- [Autograd 2.15](https://docs.pytorch.org/docs/2.15/autograd.html)：`detect_anomaly(check_nan=True)` 可在 backward 运算产生 NaN 时抛错，并显示创建失败 backward function 的 forward traceback；文档提示该模式会显著减慢程序，只应用于调试。它不是 forward 中首个非有限张量检测器，也不以该开关检查 Inf。
- [`clip_grad_norm_` 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.clip_grad_norm_.html)：`error_if_nonfinite=True` 在总梯度范数为 NaN、Inf 或 -Inf 时抛错；它提供全局范数检查，不定位源张量。
- [DDP 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.parallel.DistributedDataParallel.html)：构造、forward 和输出反向传播属于同步点；各 rank 的参数注册顺序必须一致。`no_sync()` 内累计本地梯度，第一次离开 context 后的 forward-backward 再同步，且 forward 必须在 context 内。
- [Debugging environment variables 2.15](https://docs.pytorch.org/docs/2.15/debugging_environment_variables.html)：`TORCH_DIST_NAN_CHECK=1` 检查任意 backend 的 collective 输入（发送）缓冲区中的 NaN；文档没有把它描述为 Inf 检查、首个坏算子定位或 collective 顺序诊断。
- [Distributed Checkpoint 2.15](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html)：支持多 rank 并行保存/加载和 load-time resharding。应用层事务式发布和恢复轨迹验证需要另外设计，不能从有保存 API 推出。
- **证据等级与边界：** 固定版本官方文档，支持 PyTorch 2.15 的接口范围。文档复核不是在目标模型、硬件或分布式集群上运行验证。
