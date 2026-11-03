# 训练稳定性与调试：来源证据

记录日期：2026-11-02

本记录只保存新核验的研究和框架事实、来源版本及其证据范围。书内取舍、改写和验收状态写入 progress/。

## PyTorch 2.15 调试接口

- **来源与版本：** PyTorch 2.15 官方 API 文档；2026-11-02 查阅 [autograd anomaly detection](https://docs.pytorch.org/docs/2.15/autograd.html)、[clip_grad_norm_](https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.clip_grad_norm_.html)、[debugging environment variables](https://docs.pytorch.org/docs/2.15/debugging_environment_variables.html)、[DistributedDataParallel](https://docs.pytorch.org/docs/2.15/generated/torch.nn.parallel.DistributedDataParallel.html) 和 [Distributed Checkpoint](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html)。
- **实际阅读范围：** autograd 的 “Debugging and anomaly detection”；clip_grad_norm_ 的参数、返回值；environment variables 的 TORCH_DIST_NAN_CHECK 条目；DDP 对构造、forward/backward 同步点和参数注册顺序的说明；DCP 的 state save/load 与 resharding 入口。
- **来源支持的事实：**
  - torch.autograd.detect_anomaly(check_nan=True) 可为导致失败 backward function 的 forward 操作打印 traceback；check_nan=True 时，backward 产生 NaN 会报错。文档明确提示该模式会减慢程序，应限于调试。
  - clip_grad_norm_(..., error_if_nonfinite=True) 在所有参数梯度拼接后计算总范数；若该总范数为 NaN、正 Inf 或负 Inf，会抛错。
  - TORCH_DIST_NAN_CHECK=1 检查所有 backend 的 torch.distributed collective 输入（发送）buffer 是否含 NaN，由 backend-agnostic 的 ProcessGroup collective hooks 驱动。
  - DDP 文档说明构造、forward 和输出反向传播可能形成同步点，且所有进程的参数注册顺序须相同。
- **证据等级与边界：** source-backed（固定版本的官方 API 文档），不代表本地运行复核。Anomaly detection 的说明只保证 backward NaN 检测和 forward traceback，不等于 forward first-bad-tensor 或 Inf 检查；梯度范数检查只能指出总范数非有限；TORCH_DIST_NAN_CHECK 的接口描述限于 NaN collective buffer，不能据此推断其能定位上游算子、检查 Inf 或诊断 collective 顺序/hang。框架接口不保证应用层数据游标、随机状态或跨文件原子发布。

## Adam loss spike 机制及适用范围

### Molybog 等：Adam update 与训练不稳定

- **来源与版本：** Molybog et al., “A Theory on Adam Instability in Large-Scale Machine Learning,” arXiv:2304.09871 v2（2023-04-25），[HTML 正文](https://arxiv.org/html/2304.09871)。
- **实际阅读范围：** 摘要、§1、Table 1、规模对照段、更新方向/梯度相关性分析与讨论。
- **来源支持的事实：** 作者提出 Adam 更新向量可能出现较大范数，并与训练损失的下降方向弱相关；论文比较 7B、30B、65B 和 546B 训练。该实验中 7B/30B 没有观察到同等的爆炸式 loss divergence，65B 表现为中度不稳定，546B 有明显 spike。不同模型的深度、宽度、batch 和学习率也随规模改变。
- **证据等级与边界：** source-backed（作者的大规模训练观测与理论分析）。该论文承认数据和架构会影响严重程度，并指出还需在类似设置下比较其他优化算法；它不能证明所有 spike 都由 Adam 引起，也不能把单个规模的现象外推到所有训练任务。

### Bai 等：自适应预条件器与 loss spike

- **来源与版本：** Bai et al., “Adaptive Preconditioners Trigger Loss Spikes in Adam,” arXiv:2506.04805 v2（2026-05-25），[HTML 正文](https://arxiv.org/html/2506.04805v2)。
- **实际阅读范围：** §4.1–4.3 的 Hessian 稳定条件和梯度方向曲率，§6 的合成与语言模型实验，Appendix D 的证明，Appendix G 的模型配置。
- **来源支持的事实：** 对固定学习率普通梯度下降，论文的严格单步条件使用沿参数更新路径的加权平均 Hessian；在梯度有限且非零时，loss 增加当且仅当梯度方向曲率超过 2/eta。真实网络中的 Hessian 随参数变化；论文使用当前步 Hessian/Adam 预条件 Hessian 的曲率作为实际监控量，并以 Hessian-vector product 估计，而不是声称该代理与路径平均严格条件等价。实验包括约 10M 参数、8 层、900K 序列、batch 2048 的合成任务，以及在 100B SlimPajama token 上训练的 187M LLaMA 结构模型；后者对照 beta2=0.999/0.9/0.85，报告在这些设置中降低 beta2 会减少 spike 频率。
- **证据等级与边界：** source-backed（理论及作者实验）。严格推导依赖论文规定的简化优化设定；HVP 监控有额外二阶计算，不是普通训练日志的免费指标；较低 beta2 的结果绑定该论文模型、数据、学习率、实现和 token 配方。

## Spike-aware 梯度方法

### SPAM

- **来源与版本：** Huang et al., “SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training,” arXiv:2501.06842 v2（2025-02-28），[HTML 正文](https://arxiv.org/html/2501.06842v2)。
- **实际阅读范围：** §3 方法，§4.1 C4 预训练，§5 的 reset interval 消融，Appendix D Table 7 的训练配置，Appendix L 的运行时间。
- **来源支持的事实：** SPAM 用 Adam 二阶矩构造在线坐标级 spike 代理，并对异常梯度做 spike-aware clipping；另周期重置动量并在 reset 后 warmup。论文分析的回顾性 GSS 使用每个坐标跨步的历史梯度统计，而训练代理 g_i^2/V_i 是 online proxy，两者不具代数等价性。常规 C4 预训练表中 Delta T=500、theta=5000、warmup 150 步；LLaMA-130M 消融在 Delta T 从 2500 缩到 500 时改善，继续缩短则退化。主表报告 LLaMA-60M 的 Adam/SPAM PPL 为 34.09/30.46，LLaMA-1B 为 16.13/14.66。Appendix L 的单 H100、100 次迭代测量中，60M Adam/SPAM 分别为 0.3666/0.3814 秒每步。
- **来源内分歧与边界：** LLaMA-60M 的 token 数在主表列 1.1B、Appendix D Table 7 列 1.3B；PPL 应以论文作者报告呈现，不能声称两处给出了统一 token 预算。所有性能、阈值和速度结果绑定论文训练配方、硬件与指标定义；不是本项目复现或通用默认值。

### GradientStabilizer

- **来源与版本：** Huang et al., “GradientStabilizer: Fix the Norm, Not the Gradient,” ICML 2026, PMLR 306:47732–47749，[PMLR 官方页面](https://proceedings.mlr.press/v306/huang26bb.html)。
- **实际阅读范围：** PMLR 官方摘要与论文元数据。
- **来源支持的事实：** 摘要称该梯度变换保持瞬时梯度方向，使用运行梯度范数统计得到稳定化幅值，并报告在 LLM FP16 预训练、FP4 量化感知预训练及其他任务上的稳定性实验。
- **证据等级与边界：** source-backed（同行评审论文摘要）。摘要支持方法概述和实验任务范围，不足以复现所有算法细节或把作者的稳定性、学习率范围和性能结果推广到其他模型、recipe、硬件或任务。论文作者报告不等于本地复现。

### AdaGC 的既有来源记录

- AdaGC 的论文版本、张量级 EMA 阈值、全局裁剪 warmup、作者定义的 spike score、表 7 结果和并行归约边界已记录于 [initialization-and-training-stability-2026.md](initialization-and-training-stability-2026.md)。本记录复用该来源证据，不重复复制实验数字。

## 输出头有限精度与迟发 loss spike

- **来源与版本：** Chen et al., “The Role of Quantization in Loss Spikes,” arXiv:2610.38277，2026-10-28 发布；[arXiv 正文](https://arxiv.org/abs/2610.38277)，2026-11-02 查阅。
- **实际阅读范围：** §§1、2.3、3.1–3.4、结论及 Appendix C.1–C.3；包括模型/数据/硬件配置、BF16 mixed/BF16 true/FP32 对照、LM-head SGD-M、z-loss、μ-centering 与误差测量。
- **来源支持的事实：** 作者在 28M–1.4B 参数的现代化 GPT-2 结构、FineWeb/GPT-2 tokenizer 与 QK-Norm 设置中研究高学习率下的输出 logits 增长。主精度对照将 28M 模型训练 16B tokens（约 570 token/parameter、约 28 倍 Chinchilla token 预算），peak LR 为 0.1；运行使用 JAX/Flax，最多 32 个 Google Cloud TPU。五个 seed 的报告中，BF16 true 最早出现迟发 time-bomb spike，BF16 mixed 后出现，FP32 在测试 horizon 内保持稳定。作者测同一 hidden states 的 BF16/FP32 输出 logits RMS 差异和 softmax 分布 KL，发现误差随 logit 均值增长，并提出 Adam 对词表行的坐标级预条件可打破 LM-head 交叉熵梯度的零和结构、造成输出 embedding/logit 均值漂移；精确算术下共同平移不改 softmax，有限精度则可能损失 logits 间差值。作者还报告 LM-head SGD-M、z-loss 和 μ-centering 在其设置中抑制误差及迟发 spike。
- **证据等级与边界：** source-backed（单篇预印本的作者实验）。模型规模、过高学习率和约 28 倍计算最优 token 的长程过训练是重要条件；作者明确承认结果对更自然的大模型训练的迁移尚未确定。方法效果不构成通用处方，也不证明普遍生产 spike 都起因于 LM head 量化。BF16/FP32 同 hidden-state logits 比较和分布 KL 是诊断性研究测量，成本高于普通训练日志，不应视为低成本默认监控。
