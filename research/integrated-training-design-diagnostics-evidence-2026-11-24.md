# 训练系统综合设计与故障诊断：研究证据（2026-11-24）

本记录用于精修《训练系统综合设计与故障诊断》。范围是核对本章的综合账本案例、数据混合、训练故障机制和评测污染证据。它记录可复用的研究事实及其边界，不记录章节审阅进度。

## 可复用事实与证据

### DoReMi：域级 excess loss 与候选采样比例

- DoReMi 先训练 reference model，再用 Group DRO 更新 proxy model 与域权重，最后按平均域权重重采样并训练更大的主模型。论文的 excess loss 用 reference loss 校正 proxy loss；这不是把域原始 loss 直接当作采样权重。
- The Pile 实验用 280M reference/proxy 模型得到 8B 主模型的候选权重。论文报告：五个生成式 one-shot 下游任务的平均准确率提高 6.5 个百分点；达到 baseline 准确率时，DoReMi 用 75k step，而 baseline 用 200k step（2.6 倍更快）。训练两份 280M 模型寻找权重的 FLOPs 约为训练主模型 FLOPs 的 8%。这些数字属于该语料、模型和实验设定；代理模型尺度会改变学出的权重。
- 原始论文：[arXiv HTML v4](https://arxiv.org/html/2305.10429)，特别是算法与实验段落。

### Adam loss spike：机制及实验对象的边界

- Molybog 等提出时间相关的梯度估计可能使 Adam 更新向量范数较大、却与下降方向近乎不相关，并展示 7B、30B、65B、546B 训练轨迹。这是作者的机制解释和观测，不是每类 loss spike 的统一诊断。
- Bai 等论文的 187M LLaMA 结构模型在 100B SlimPajama tokens 设置中出现多次 spike；降低 β₂ 可减少该设置的 spike 频率。论文关于增大 ε、抑制 spike 的直接实验图使用的是两层全连接函数逼近任务；不应把该 ε 结果和 187M 语言模型实验混成一个证据范围。
- 二者均未证明可以不经目标配置验证而直接采用固定 β₂、ε 作为大模型生产配方。
- 原始论文：[Molybog 等](https://arxiv.org/html/2304.09871)；[Bai 等，arXiv v2](https://arxiv.org/html/2506.04805)。

### 输出头有限精度与 time-bomb spike

- Chen 等 2026 年论文 v2（2026-11-02）把一类延迟、终末性 loss spike 限定为其研究的 time-bomb 现象。代表性精度对照使用 28M 参数模型、16B tokens（约 570 token/parameter，约为其采用的 Chinchilla 参照预算的 28 倍）和五个 seed；作者在相同 hidden states 上以 FP32 重算 LM head logits，并比较 logit RMS 误差和 softmax 分布 KL。BF16 true 组较早失败、BF16 mixed 组较晚，FP32 组在所测训练区间保持稳定。
- BF16 在数值绝对值 1024—2048 区间的相邻表示间距为 8。对 logits 同时加或减常数在精确 softmax 中不改分布；若量化已抹去 logits 间差异，之后做稳定 softmax 不能找回这些差异。这个数值例说明有限精度机制，不预测真实训练的发生概率。
- 结论边界：论文讨论的是特定小模型、较长训练预算和精度设置下的 time-bomb；不代表所有 loss spike 都由该机制引起。作者把 AdamW 的坐标预条件、输出头漂移和量化误差连在一个可检验解释中，仍需要目标模型对照。
- 原始论文：[arXiv HTML v2](https://arxiv.org/html/2610.38277)。

### Per-domain loss 与评测污染

- PALOMA 面向 546 个英文和代码文本域的语言模型 fit；其对照建议覆盖训练去污染、训练顺序、分层子采样、词表以及 perplexity 输入格式。它支持查看不同文本域的 LM fit，不衡量推理、安全或工具能力。
- PaCoST 对每个 benchmark 样本生成语义对应的改写，分别计算原题与改写题答案的 P(True) 自我置信度，再对配对差做单侧配对 t 检验。显著差异是 benchmark 级别的潜在污染行为证据，不是训练语料溯源、受影响样本定位或分数膨胀因果估计。测试大量 benchmark 时还要说明多重检验等统计口径。
- 原始论文：[PALOMA arXiv HTML v2](https://arxiv.org/html/2312.10523)；[PaCoST Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.97/)。

## 未验证边界

- 没有对本书或目标项目的模型、数据、拓扑做训练运行、benchmark 或 checkpoint 恢复实验；论文中的数字仍是对应论文的报告值。
- DoReMi 配比迁移到本书示例 7B 模型前，仍需固定总有效 token、数据版本、主模型和评估协议做目标验证。
- Bai 与 Chen 等的 loss-spike 机制应在目标训练的时间线、optimizer state、有效 token 分母、rank 数据和精度对照上验证，不从症状名称直接选修复。
- 污染审计只能覆盖可访问语料、选定匹配器与阈值；无法从未命中推断隐藏语料中绝无重叠。
