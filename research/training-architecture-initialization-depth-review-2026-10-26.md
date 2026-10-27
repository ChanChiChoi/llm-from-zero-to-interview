# 第五册第四章：模型架构与初始化深度研究记录（2026-10-26）

## 研究问题与范围

本轮按 `WRITING_SPEC.md` 对《模型架构与初始化》4.0—4.18 复读；目标不是把 RoPE、RMSNorm、GQA、SwiGLU、MoE 等名词再定义一遍，而是补足它们分别解决的约束、机制成立条件、代表研究的实际证据和适用边界，并检查章节中的参数/缓存账本是否会让读者误读。正文仍维持一个文件；当前问题可以在原有 19 个主节内讲清，不需要拆章。

筛出的核心问题：

1. 参数预算中，attention projection、FFN、embedding 和 KV cache 的比例不能混为一谈。
2. RoPE 的频率表、相对位移性质和长上下文扩展方法需要连接起来解释。
3. RMSNorm、Pre-Norm/Post-Norm、DeepNorm 与 μP 分别处理不同训练问题，不能归为一个可互换的“初始化技巧”。
4. SwiGLU 质量比较必须控制参数量；三投影门控结构不自动代表相同延迟。
5. MQA/GQA 的质量、缓存、checkpoint 上转换和设备拓扑需要一起比较。
6. MoE 的参数容量、每 token active 计算、负载均衡、router 数值稳定和 all-to-all 通信是不同问题。
7. 架构代码必须说明统计的是 dense reference 还是 MoE 子账本，不能在输出名中暗示其包含实际模型没有计算的参数。

## 代表研究与工程方案对照

| 来源 | 解决的问题与方法 | 结果及证据范围 | 限制/后续改进问题 |
| --- | --- | --- | --- |
| [RoFormer](https://arxiv.org/abs/2104.09864) | 将位置写入 Q/K 的二维旋转；绝对 position id 经过旋转后在点积中体现相对位移。 | 提供 RoPE 的频率旋转形式、相对位置性质和原始长文本分类实验入口。 | “相对位置”代数性质不意味着任意长度外推；不同 checkpoint 的 base、部分旋转维度、通道 layout、频率缩放必须对齐。 |
| [Position Interpolation](https://arxiv.org/abs/2306.15595) | 直接把新位置外推到训练范围以外可能引起相位/attention score 异常；PI 将位置缩回旧范围并少量继续训练。 | 作者在 LLaMA 7B–65B 设置中报告扩展到 32K，微调最多约 1,000 steps。 | 线性缩放会同时压缩短距离相位；应复测原长度质量和目标长文检索，而非只看最大 position。 |
| [YaRN](https://arxiv.org/abs/2309.00071) | 统一缩放所有频率会损害短距离与长距离之间的折中；YaRN 按波长做平滑分段插值，并配合注意力温度校正与继续训练。 | 论文报告其比较设置下较既有方法减少约 10× token、2.5× steps。 | 这是同论文设置内的预算对照，不是跨模型、recipe 或硬件的固定效率倍率。 |
| [LongRoPE](https://arxiv.org/abs/2402.13753) | 维度和位置的外推需求不均匀，极长扩展还可能损害短上下文。 | 搜索维度相关插值因子，分阶段从 256K 延至 2,048K，并用 8K 再调整恢复短上下文；报告基于 LLaMA 2/Mistral。 | 论文的 2M 支持受模型、继续训练和任务设计约束，不代表 2M 输入的通用有效检索/推理。 |
| [RMSNorm](https://arxiv.org/abs/1910.07467) | 检验 LayerNorm 的 re-centering 是否必要，以较简单的 RMS 重缩放替代中心化。 | 作者在多种网络和任务上报告与 LayerNorm 可比，并按具体模型报告 7%–64% running-time reduction。 | 旧论文的运行时间不能直接迁移到当代 LLM 或具体 fused kernel；RMSNorm 没有 LayerNorm 的平移不变性，收益需要固定实现实测。 |
| [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) | 解释 Post-LN warmup 需求与 Norm 位置的关系。 | mean-field 分析指出 Post-LN 初始化时靠近输出层梯度偏大，Pre-LN 初始化梯度相对平稳；在作者任务中测试了不使用 warmup。 | 解释的是特定初始化分析与任务实验，不推出所有现代模型的 Pre-Norm recipe 都无需 warmup。 |
| [DeepNet / DeepNorm](https://arxiv.org/abs/2203.00555) | 极深 Transformer 的 model update 可能失控；联合修改 residual connection 与特定层初始化来约束更新。 | 作者报告扩展到 1,000 层；多语翻译中 200 层、3.2B 参数模型超过 48 层、12B 参数基线 5 BLEU。 | 重要证据来自 encoder-decoder NMT；将其用于现代 decoder-only LLM 需重建适配与稳定性实验。 |
| [Tensor Programs V / μTransfer](https://arxiv.org/abs/2203.03466) | 扩宽模型后重新调每个超参成本很高；μP 用宽度一致参数化使部分超参从小模型迁移至大模型。 | 论文在 Transformer、ResNet 上验证了小模型 HP 调优到大模型的 transfer。 | μP 是一整套参数化与 optimizer/shape 约束，不是一个可脱离架构应用的初始化标准差。 |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | 比较门控前馈子层中的非线性变体。 | 在 seq2seq Transformer 中发现若干 GLU 变体可胜过 ReLU/GELU。 | 原论文没有解释现代 decoder-only LLM 的普遍收益机制；对比必须控制 FFN 参数量与训练预算，吞吐还依赖 kernel。 |
| [GQA](https://arxiv.org/html/2305.13245v3) | MQA 缓存/带宽更省但可能损害质量；提出 GQA 并从 MHA checkpoint 上转换。 | 将组内 K/V 矩阵 mean-pool 后以原配方继续预训练。T5-XXL 约 5% uptraining 时，GQA-8 平均分 47.1、0.28 秒/样本/TPUv4 芯片；MHA-XXL 为 47.2、1.51 秒；MQA-XXL 为 46.6、0.24 秒。 | 任务平均包含 ROUGE、BLEU、F1；计时受 TPU、batch 和并行设置影响；主要模型是 encoder-decoder，作者也承认没有与从头训练的 GQA 做同等对照。 |
| [Llama 2](https://arxiv.org/html/2307.09288v2) | 检验 MHA/MQA/GQA 在目标模型质量、batch throughput 和 tensor-parallel 分片中的实际折中。 | 在 34B/70B 选用 GQA；论文报告 attention 结构 ablation，并说明多 GPU 情况下 MQA 需要 KV 复制或改用 batch 切分。 | 同时变更多个模型与系统条件，不能仅靠表中分数归因架构；其 8×A100/TP 配置不等同于其他 serving 形态。 |
| [Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538) / [GShard](https://arxiv.org/abs/2006.16668) | 从条件计算中扩展参数容量；GShard 进一步把模型分片扩展做成可编程机制。 | GShard 报告 600B 级多语翻译模型在 2,048 TPU v3 上训练。 | 证明的是特定翻译任务与 TPU/XLA 环境下的可扩展性；通信和 load balance 是算法/系统共同约束。 |
| [Switch Transformer](https://arxiv.org/abs/2101.03961) | 简化 top-2 稀疏路由的复杂度、通信和稳定性问题。 | 用 top-1 routing；T5 Base/Large 实验报告同计算资源下最多 7× 预训练速度，并报告 BF16 稀疏训练。 | 这不是任意 decoder-only LLM 的推理倍速；capacity、token dropping、路由失衡仍需实测。 |
| [ST-MoE](https://arxiv.org/html/2202.08906) | 稀疏模型训练的 loss spike 和低精度 router 不稳定，及 sparse model 微调的 recipe 差异。 | Router z-loss 惩罚 softmax 前 logits 的 log-sum-exp 平方；论文稳定性表中 baseline 4/6 稳定，z-loss 3/3 稳定且报告质量无损。 | z-loss 控制数值尺度而非专家均衡；系数依赖 sweep，不能替代负载辅助项或服务日志。 |
| [Expert Choice Routing](https://arxiv.org/abs/2202.09368) | token-choice top-k 让 expert 负载偏斜、产生 under/over-specialization。 | expert 选择固定 bucket 的 top-k tokens，作者在其资源对照中报告较 Switch top-1/GShard top-2 收敛速度超过 2×。 | token 激活 expert 的数量变成可变；同等收敛 speedup 绑定论文模型、任务与训练设置。 |
| [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v2) | 经典 auxiliary load loss 可能与主任务质量产生拉扯，batch-level balance 也无法保证单序列内部均衡。 | 每 expert routing bias 调整 top-k 选择，但 gate weight 仍由原 affinity 计算；另外有 sequence-wise balance loss。技术报告称模型 671B total/37B active 并报告其整个训练过程稳定。 | 这是模型、算法、精度、通信 kernel 和 2048 H800 集群共同设计的结果；不能拆出单一 router bias 直接声称复现同等质量或吞吐。 |

## 原有章节知识覆盖图

| 原位置/主题 | 本轮处理及修订后位置 |
| --- | --- |
| 4.0 架构预算表、四个账本与冲突关系 | 保留，继续作为全章问题地图。 |
| 4.1 目标约束与 decoder-only/encoder-decoder 边界 | 保留；没有把通用生成主流说成其他结构无价值。 |
| 4.2 causal attention、统一序列接口和任务 mask 区别 | 保留。 |
| 4.3 参数公式、embedding、FFN 和层宽取舍 | 保留；新增 4.3.6 数值手算，说明 GQA attention projection 与 cache 降幅不同，并把 FFN 与 attention 参数做比例比较。 |
| 4.4 6PD、token/parameter 和部署预算 | 保留；Dense 近似和 MoE 口径仍分开。 |
| 4.5 RoPE 旋转、相对位移、长度外推 | 保留并补足频率表和 `d_rot=8` 数值例；增加 PI、YaRN、LongRoPE 的问题、机制、报告结果和边界比较。 |
| 4.6 LayerNorm/RMSNorm、Pre/Post-Norm、残差尺度 | 保留；增加 LayerNorm 公式、Norm 位置的梯度因果证据、RMSNorm 研究范围；明确 residual heuristic 的独立分支假设；补 DeepNorm 与 μP 的不同目标。 |
| 4.7 SwiGLU 机制、参数与 FFN 边界 | 保留；增加等参数推导、4096 hidden 的整数宽度手算、Shazeer 研究证据与质量/延迟比较条件。 |
| 4.8 MHA/MQA/GQA 的 cache 和质量折中 | 保留；增加 5% checkpoint uptraining 机制与论文表中 GQA-8/MQA/MHA 指标，绑定其 T5/TPUv4 限制；补 Llama 2 的结构和并行拓扑理由。 |
| 4.9 长上下文计算和有效利用评估 | 保留。与 4.5 的区分是：4.5 解释 RoPE 相位变换，4.9 解释训练数据、任务能力与系统成本。 |
| 4.10 MoE 总参数、active 参数、router、capacity、overflow | 保留并细化 routed/shared/router 账目，区分经典 auxiliary loss、router z-loss、capacity 和 routing bias；新增六类代表方案对照。 |
| 4.11 Xavier、残差初始化、新 token embedding、checkpoint 续训 | 保留；与 4.6 区分初始化直觉和深度/宽度缩放研究。 |
| 4.12 稳定性指标和小实验 | 保留；增加它们与归一化位置、MoE router 数值稳定之间的研究依据。 |
| 4.13 推理成本变量 | 保留。 |
| 4.14 标准库架构账本 | 保留并修正名字与文档：主账本标为 dense reference，MoE FFN 单层子账本独立；明确 tied embedding、norm、输出头遗漏；RMSNorm 计算改为数值缩放版，RoPE helper 改名并限定为固定频率的单 2D pair。 |
| 4.15 故障排查路径 | 保留；RoPE、GQA、router 诊断已有更清楚的机制和区分。 |
| 4.16 来源、资料边界和阅读清单 | 保留并新增 LongRoPE、DeepNorm、μP、GShard、ST-MoE、Expert Choice、Llama 2 和 DeepSeek-V3 一手入口。 |
| 4.17 六项练习 | 保留；练习已覆盖参数、GQA、RoPE、初始化、MoE 和 checkpoint 迁移。 |
| 4.18 小结 | 保留章节结构；重要机制增加证据入口和验证要求，没有移除既有问题。 |

## 本地手算与静态复核

- `d=4096, H_q=32, d_h=128, H_kv=8`：MHA Q/K/V/O 投影为 `67,108,864` 参数；GQA 投影 `41,943,040`，占 62.5%；KV cache 则占 MHA 的 25%。
- `d_ff=4d` 普通 FFN 为 `134,217,728` 参数；SwiGLU 取整数 `d_ff=10,923` 时为 `134,221,824`，相差约 0.003%。
- `d_rot=8, base=10,000` 的四个 RoPE 频率按公式为 `1, 0.1, 0.01, 0.001`。
- RMSNorm helper 采用按最大绝对值缩放的等价计算，避免对有限但很大的输入直接平方而溢出。helper 不含 affine scale/bias，是数值机制示例而不是完整框架层。
- MoE 参数公式中的 `E` 明确定义为 routed experts；shared experts 与 router 分开计。DeepSeek-V3 的 bias strategy 不能误读为完全没有辅助损失：论文还保留 sequence-wise auxiliary balance loss。
- 本轮未执行章节 Python 代码；上述数字是公式手算，代码状态不标记为运行验证。

## 来源核验状态与开放风险

主要一手来源以 arXiv 作者论文及技术报告为主。RoFormer、PI、YaRN、LongRoPE、RMSNorm、Pre-LN/Post-LN、μP、SwiGLU、GShard、Switch 和 Expert Choice 本轮核读了论文页面摘要/方法摘要；GQA 读了方法、主要结果、timing、限制与 stability appendix；Llama 2 核对 attention ablation 表和并行/吞吐说明；DeepNet 阅读 practitioner summary、DeepNorm 定义和多语实验摘要；ST-MoE 核读 router z-loss、precision stability 与对应消融；DeepSeek-V3 核对路由/负载均衡公式、序列级辅助损失和系统章节。没有声称逐篇逐页复现全部论文，也没有运行仓库代码。

开放项：

- PI/YaRN/LongRoPE 的最大长度和训练成本来自不同 checkpoint、任务和训练设置，不能横向排名；没有用目标模型复现。
- RMSNorm、Pre-LN、DeepNorm 和 μP 的论文任务/架构不完全相同，正文用来解释解决的问题，不构成架构排行榜。
- GQA 时间与质量是 T5-XXL/TPUv4 结果；Llama 2 的选择也是自身模型与 8×A100 拓扑下的决策，不代表目标集群。
- MoE 文献中训练加速、benchmark 质量、route balance 和吞吐口径差异大；目标集群的 all-to-all、p99、overflow 与 expert 权重驻留仍未验证。
- 本轮未运行 §4.14 Python；未运行真实 GPU、训练栈或 serving profile；未构建 PDF。
