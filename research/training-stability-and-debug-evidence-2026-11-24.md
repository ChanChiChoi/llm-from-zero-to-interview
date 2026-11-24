# 训练稳定性与调试：研究证据（2026-11-24）

## 范围与检索

围绕大模型训练中的 loss spike、Adam 不稳定、异常梯度裁剪、分布式非有限值检测、checkpoint 原子性核验正文主张。2026-11-24 以英文关键词检索并直接阅读原论文、PMLR 论文页及 PyTorch 2.15 官方文档；检索词包括 `Adam loss spike instability`、`SPAM gradient spike score`、`AdaGC spike score`、`PyTorch 2.15 detect_anomaly`、`TORCH_DIST_NAN_CHECK`、`DDP synchronization points` 和 `Distributed Checkpoint atomic`。本条记录仅存新核验的事实及其来源边界。

## 核心证据

### Adam 不稳定与自适应预条件器

- Molybog 等人的论文报告了 7B、30B、65B 和 546B 参数模型的训练观察；不稳定程度依模型和训练设置变化，7B 与 30B 实验未出现与最大规模相同的爆炸式 loss divergence，546B 的不稳定最显著。作者讨论降低学习率、调整 ε、降低 batch size、修改 β₁/β₂ 等候选处理方式，并明确说明吞吐、低精度数值和正常训练阶段响应方面的取舍。这些是论文中的机制解释与建议，不构成跨任务的通用修复处方。
  - 来源：Igor Molybog et al., *A Theory on Adam Instability in Large-Scale Machine Learning*, arXiv v2, https://arxiv.org/html/2304.09871v2 ，第 1 节、表 1、第 7.1 节。
  - 支持：论文设置下 Adam 更新分布与时间相关性、模型规模相关的 loss instability 观察和作者建议。
  - 不支持：所有 loss spike 均由 Adam 单独造成，或任何单一超参数调整都能普遍修复。
- Bai 等人把其机制归因于 Adam 二阶矩估计与即时梯度平方失配：二阶矩可在梯度变小时按 β₂ 衰减，梯度回升时响应滞后。论文区分预条件 Hessian 最大特征值越过 $2/\eta$ 的稳定性信号，与沿当前梯度方向的曲率用于判断 spike 的预测量；短时越界不等于一定出现 spike。理论证明限于简化的一维二次情形，高维模型结论依赖实验验证。实验包括约 10M 参数、8 层合成推理 Transformer（900k 序列、batch 2048，观察到 7 次 spike）以及 187M 非 embedding 参数的 LLaMA 结构模型（100B SlimPajama tokens）；在后者中比较了 β₂ 为 0.999、0.9 和 0.85 的设置，较低 β₂ 在该实验中降低 spike 频率。Hessian-vector product 估计需要额外计算，论文也指出显著大于 200M 参数时扩展这种分析仍昂贵。
  - 来源：Zhiwei Bai et al., *Adaptive Preconditioners Trigger Loss Spikes in Adam*, arXiv v2, https://arxiv.org/html/2506.04805v2 ，摘要、第 4–6 节、附录 G。
  - 支持：该论文特定机制、参数规模、训练数据、测量与作者报告的 β₂ 对照。
  - 不支持：将 $2/\eta$ 作为所有 Adam 实现或非凸训练的严格普适判据。

### SPAM 指标、实验与边界

- SPAM 论文中的回顾性 gradient spike score 是一个坐标在当前步的梯度绝对值除以该坐标跨训练步梯度绝对值的平均值；训练时无法预先获得完整历史，因此论文采用 $g_i^2/V_i$ 作为在线近似代理。论文明确区分两种定义，二者不是代数等价式，回顾性 GSS 的阈值不能直接移作在线代理阈值。
- 常规预训练设置报告 $\Delta T=500$、momentum reset 后 150 步 warmup、$\theta=5000$；这些是论文配置，不是推荐的通用默认值。
- C4 预训练结果主表报告 LLaMA-60M 上 Adam/SPAM 的困惑度分别为 34.09/30.46，LLaMA-1B 为 16.13/14.66；论文主表将 60M 训练 token 数列为 1.1B，附录 D 表 7 列为 1.3B。该差异限制了对 60M 数字进行 token 预算严格对齐比较的能力。论文另报告单张 H100、100 次迭代下 LLaMA-60M Adam/SPAM 平均每步分别为 0.3666/0.3814 秒；这不是跨硬件或实现的成本估计。
  - 来源：Tianjin Huang et al., *SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training*, arXiv v2, https://arxiv.org/html/2501.06842v2 ，定义 2.1、第 3 节、表 1、附录 C、D、L。
  - 支持：论文定义、设置与作者报告值。
  - 不支持：将在线代理等同于回顾性 GSS，或把单一 H100/模型配置上的时间外推成普遍开销。

### AdaGC 与可引用范围

- ICML 2026/PMLR 正式论文摘要确认 AdaGC 按 tensor 维护历史 clipped-gradient norm 的指数移动平均并据此设置自适应裁剪界限；论文摘要报告 Llama-2 7B、Mixtral 8×1B 和 ERNIE 10B-A1.4B 设置下 spike score 降为 0，并报告相对 GlobalGC 的下游准确率增量分别为 1.32%、1.27% 和 2.48%。这些结果绑定论文的模型、训练及评测条件。
- 作者的 arXiv v4（页面标记为 ICML 2026 接收稿）源文件进一步确认：spike score 是时间序列中偏离前 1,000 个值滚动均值至少 10 个标准差的点所占百分比；GlobalGC/AdaGC score 分别为 Llama-2 7B 0.0333%/0、Mixtral 8×1B 0.0144%/0、ERNIE 10B-A1.4B 0.0100%/0；表中步数分别为 9K、36K、21K。Qwen3-1.7B 的 AdaGC score 为 0.0053%，并非 0。方法默认前 100 步使用 GlobalGC warm-up。不同论文版本的表序号可能改变，正文无需引用未核对的表号；“score 为 0”也不能解释成不存在任何训练异常。
  - 来源：Guoxia Wang et al., *AdaGC: Enhancing LLM Pretraining Stability via Adaptive Gradient Clipping*, ICML 2026, PMLR 306, https://proceedings.mlr.press/v306/wang26dg.html 。arXiv v4 页面/源文件：https://arxiv.org/abs/2502.11034v4 。
  - 支持：正式摘要及作者 v4 源文件中的方法轮廓、score 定义、逐模型统计和 warm-up 设置。
  - 不支持：把该论文的 loss-point score 与本章 loss-window ratio 直接比较，或把零 score 推广为训练没有任何异常。

### 工程排障与教学组织

- NVIDIA NeMo AutoModel 0.7.0（26.10）排障文档描述一种 checkpoint 输入 embedding 故障：少数 embedding 行的范数可能为零、近零或非有限；训练可在对应 token ID 出现前看似正常，该 token 出现后会产生极大 embedding gradient 并主导 global gradient norm。文档将 embedding 行修复限定在确认该 checkpoint 缺陷后，不把它作为坏数据、不稳定 loss、精度模式或 optimizer state 问题的通用修复。
  - 来源：NVIDIA, *NeMo AutoModel 0.7.0 Troubleshooting*, https://docs.nvidia.com/nemo/automodel/v0.7/model-coverage/troubleshooting ，页面“Repair Damaged Input-Embedding Rows”。
  - 支持：该版本所记录的故障形态及特定修复边界，不支持其对所有训练框架/模型普遍发生的推断。
- Berkeley Scalable AI Spring 2026 pretraining lecture 将 loss pathologies 分成 divergence 与 data spike 两种可观察签名，联合查看 loss、grad norm、数值状态、数据来源/分片边界及调度变化；并指出稳定性还涉及数据工程和系统上下文。这是课程材料的组织方式，可作为排障分类参考，不证明该结构对所有学习者有普遍效果。
  - 来源：Berkeley Scalable AI, *Lecture 6: Introduction to Pre-Training*, Spring 2026, https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture6.pdf ，slides 42–46。

### PyTorch 2.15 调试与分布式接口

- `torch.autograd.detect_anomaly(check_nan=True)` 可在 backward 运算产生 NaN 时报错，并显示创建该 backward 函数的 forward traceback；官方文档将其定位为调试功能并提示其会拖慢程序。此说明不包括对 Inf 的检测保证。
- `torch.nn.utils.clip_grad_norm_(..., error_if_nonfinite=True)` 对拼接意义下的总梯度范数检查 NaN/Inf；它不会指出第一个坏张量。
- 使用分片 FSDP 策略时，应调用 FSDP 的 `clip_grad_norm_`，该方法按全局向量范数视角处理跨 rank 分片，并要求所有 rank 调用；`NO_SHARD` 可直接使用标准工具。
- `TORCH_DIST_NAN_CHECK=1` 检查任意 backend 的 `torch.distributed` collective 发送缓冲区是否含 NaN；官方描述不包含 Inf 检测，也不声称定位最初产生 NaN 的算子。
- DDP 官方文档把构造、forward 和输出（或输出函数）的 differentiation 列为分布式同步点，并要求所有进程的参数注册顺序相同。
- DCP `FileSystemWriter` 的文档假设包含单文件创建原子性；它不等于训练应用跨 rank、多文件 checkpoint 的整体事务提交。manifest、完整性校验及版本发布协议仍需由应用明确定义。
  - 来源：PyTorch 2.15 官方文档：https://docs.pytorch.org/docs/2.15/autograd.html ；https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.clip_grad_norm_.html ；https://docs.pytorch.org/docs/2.15/fsdp.html ；https://docs.pytorch.org/docs/2.15/debugging_environment_variables.html ；https://docs.pytorch.org/docs/2.15/generated/torch.nn.parallel.DistributedDataParallel.html ；https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html 。
  - 支持：文档所列版本中的接口行为及 API 限制。
  - 不支持：对任意 PyTorch 版本、后端、应用级 checkpoint 事务及模型故障根因作通用推断。

### 归一化与深层训练

- Pre-LN 的 mean-field 研究在其初始化分析中报告了与 Post-LN 不同的梯度行为并解释 warmup 的关联；DeepNet 的深层结果绑定论文给出的架构/初始化和翻译实验；Peri-LN 的实验结论也受其模型、参数规模和任务设置限制。不可将这些论文的缩放/归一化方案直接描述成可安全切换到任意现成 checkpoint 的通用修复。
  - 来源：Xiong et al. (2020), https://proceedings.mlr.press/v119/xiong20b.html ；Wang et al. (2022), https://arxiv.org/abs/2203.00555 ；Kim et al. (2025), https://proceedings.mlr.press/v267/kim25u.html 。
