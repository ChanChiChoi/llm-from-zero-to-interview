# K2-Horizon 架构与 Uno 证据记录

研究日期：2026-11-25  
研究问题：为第二十一册第 82 章核对 K2-Horizon-3.7B 与 MoVA-36B-A4B 的架构、参数口径、长上下文训练证据和 Uno 适配器边界。读者已有基础 Transformer、GQA、MoE 和自回归生成知识；本记录仅保存外部事实、来源和证据边界。

## 检索范围

- 关键词：`K2-Horizon MoVA Mixture-of-Value Attention`、`K2-Horizon 3.7B dense configuration`、`Grouped RMSNorm K2-Horizon`、`K2-Horizon technical report 2611.21003`、`Uno discrete diffusion adapter Psi-Spec`、`MoVA value routing tutorial`。
- 首要来源：IFM 作者技术报告、IFM 官方 Hugging Face 模型卡/固定 revision 与模型实现、Uno 原论文。技术博客和可视化材料只用于寻找解释顺序；事实均回到一手来源核验。
- 固定对象：K2-Horizon 报告 arXiv:2611.21003v1（2026-11-16 提交）；IFM/K2-Horizon-MoVA-36B-A4B 当前仓库提交 `3fb5edfa484350459fb6b7461f80b5701b7dd6b4`（API 记录 lastModified 2026-11-22）；IFM/K2-Horizon-3.7B 当前仓库提交 `364e884e27fad049a325b9246902a28d25f2da11`（lastModified 2026-11-22）；Uno 原论文 arXiv:2609.04010v1（2026-09-03 提交）。

## 已核实事实

### K2-Horizon-3.7B 与 MoVA-36B-A4B 的结构

| 项目 | 3.7B dense | 36B-A4B MoE+MoVA |
| --- | ---: | ---: |
| 论文中的总参数（含 embedding/head） | 5.058B | 37.445B |
| backbone 参数（不含 input embedding 与 output projection） | 3.775B | 36.162B |
| 每 token active backbone 参数 | 3.775B | 4.665B |
| Transformer 层数 | 36 | 48 |
| hidden size | 2,560 | 2,560 |
| query heads / KV heads / head dim | 32 / 8 / 128 | 32 / 8 / 128 |
| dense FFN intermediate size | 10,240 | 6,144（仅初始 dense FFN 层） |
| grouped RMSNorm 分组数 | 2 | 2 |
| attention gate | 关闭 | softplus |
| routed FFN experts | 0 | 每稀疏层 100，top-8，并有 1 个 shared expert |
| value experts | 0 | 每 MoVA 层 64，top-4 |
| `max_position_embeddings` | 524,288 | 524,288 |

论文 Table 1 说明模型名是 nominal size class；总参数、backbone 参数和 active backbone 参数是不同口径。模型卡使用的“36B/约 4B active”是发布标签/四舍五入表达，不能直接等同于 36.162B/4.665B 精确计数或显存、FLOPs、吞吐。

36B 当前配置有 `mlp_only_layers: [0, 1, 2]`、`decoder_sparse_step: 1`、`num_experts: 100`、`num_experts_per_tok: 8` 和 `mova_num_experts: 64`。同一固定 revision 的 `modeling_k2_horizon.py` 将 `is_sparse_layer` 同时用于选择 attention 与 MLP：前 3 层使用普通 attention 和 dense MLP；后 45 层使用 MoVA attention 与 sparse MoE FFN。把 MoVA 仅从论文 Table 1 的“initial dense FFN layers”行外推为所有 48 层都启用不成立；模型实现明确保留前三层 dense attention。

3.7B 固定配置将 `mlp_only_layers` 列为全部 36 层，并将 `num_experts`、`num_experts_per_tok`、`mova_num_experts`、`mova_num_experts_per_tok` 设为 0；其 `attention_gate_func` 为 null。它与 36B 共享 hidden size、32Q/8KV GQA 和 524,288 位置配置，但不是 36B 的缩小版，也不是 IFM 进行 dense-vs-MoVA 主比较时的 matched dense 对照。论文 §2.2 与 §3.4 的主对照是 32B dense 模型。

### MoVA 的输入路径、路由语义与 cache

IFM 论文 §2.2 定义 MoVA 为把 MoE 风格稀疏化放到 attention 的 value projection：每个 token 的 hidden state 经 router 选取若干 value projections，合并成一个 value 表示，随后仍沿常规 GQA attention 计算。该改动不复制完整 Q/K/attention head，也不改变 query-key attention 计算。

固定模型实现显示：36B 的 router 使用 sigmoid scores；router bias 只加入 top-k 选择分数。被选专家的混合权重从原 sigmoid scores 取回、归一化后乘 `router_scaling_factor=2.5`。因此最终权重和为 2.5，不是概率和为 1 的 softmax，也不是凸组合。每个 value expert 是从 2,560 维 hidden state 到 `8 × 128`（共 1,024）维值向量的无 bias 线性层；每路输出经过 SiLU 后加权求和。

实现先完成 value expert 的路由和合并，再把单个混合 value tensor 交给 attention/cache。IFM 论文称其 cache 大小与普通 GQA 相同；MoVA 不把 64 个 expert 输出或路由索引作为每个历史 token 的额外 KV 条目保存。这里的“无额外 MoVA cache”不表示总 cache 小：官方 config 是 48 层、8 个 KV heads、head dim 128、关闭 sliding window。按 batch=1、BF16、未量化、显式保存 K 与 V 计算，512K token 的结构账本约 96 GiB；这是公式推算，未计 page metadata、其他状态、workspace 或运行时压缩。3.7B 有 36 层，同样假设下约 72 GiB。

由配置与实现推得，每层 value experts 权重为 `64 × 2,560 × 1,024` 个参数；45 个 MoVA 层合计约 7.55B 存储参数，top-4 对应约 0.472B 每 token active value-expert 参数。该数是根据公开权重形状的结构计数，不是独立权重文件校验，也不等同于实测 FLOPs 或 decode 延迟。

### Grouped RMSNorm 与 attention gate

IFM 报告 §2.1 说明 3.7B、7B、32B 和 36B-A4B 使用 Grouped RMSNorm。它在每组特征上单独计算 RMS；省略可学习缩放参数的归一化结果中，每个分量受 `sqrt(group_size)` 的上界约束。36B 与 3.7B 的 hidden size 为 2,560、分组数为 2。作者报告称这一设计提升 K2-Horizon 多个规模的训练稳定性；该结果属于作者报告，不能据此推断任意模型或训练配置都稳定性更高。

IFM 报告 §2.3 与 36B 当前 config/code 记录 softplus attention gate。门控向量由 hidden state 生成，逐元素乘在 attention 输出上，并位于 output projection 前；`beta=ln(2)` 时 `softplus(0; beta)=1`。与有界在 `(0,1)` 的 sigmoid gate 相比，softplus 可衰减也可放大。3.7B config 关闭 attention gate。该 gate、MoVA value router 和 FFN MoE router 作用在不同位置，不能混成同一个 router。

### 512K 训练与 RoPE 配置

IFM 36B 模型卡称 524,288-token context 从 midtraining 阶段起支持。报告 §3.1 的 3.7B 与 36B schedule 均从 8K pretraining 开始，之后依次为 32K / 1.1T tokens、128K / 498B、512K / 110B 和 512K / 199B；报告的 midtraining 表把 RoPE base 从 pretraining 的 500K 调到 Stage 1 的 1M，再在后续阶段用 10M。当前 HF config 中 `max_position_embeddings=524288`、`rope_theta=10,000,000`。卡片把训练阶段 token budget 说明为增量，不能将其误读为各阶段互不关联的独立数据集。context field 和长上下文训练证明配置/训练经历，不证明任意 512K 任务上的检索或推理质量。

### Dense 对照与效果结论

论文 §2.2/§3.4 将 MoVA-36B-A4B 与 32B dense 放在相同数据和训练设置下比较，MoVA 预训练 loss 紧跟 dense 但 dense 保持小幅领先；在报告的 1.1M step checkpoint 上，MoVA 在 MMLU、GSM8K、HellaSwag、HumanEval 等列举指标比 dense 低 1.2–3.7 分，同时 active backbone 参数约为 32B dense 的七分之一。该是模型组合级、作者报告级比较。36B 同时改变了 FFN 稀疏性、value 路径、层数/宽度、Grouped RMSNorm 分组和 attention gate；它没有隔离 MoVA 单一组件的因果贡献，也不能从 active parameter 比值推出一般性吞吐收益。

3.7B 的同级 benchmark 表是模型 family 的另一项结果，不是上述 MoVA 消融。模型卡 benchmark 使用各自声明的 reasoning/benchmark protocol，协议不完全相同；不从 3.7B 与 36B 的榜单行推断 MoVA 的效果。

### Uno diffusion adapter 与精确验证

IFM K2-Horizon 报告 §9.4 将 Uno adapters 加到训练完成的 0.9B 与 7B 模型，不是 3.7B 或 36B-A4B 的架构组成。报告的 adapters 为 rank-128 LoRA、LoRA alpha 256，挂接 attention Q/K/V/O 与 MLP gate/up/down 投影；训练时冻结 AR base weights，只训练 diffusion adapters。公开训练使用 7B tokens 的 SFT 数据样本，包括 16,384 context 下逐步扩展 block size，随后 65,536 context、block size 8。该 adapter 改变 draft 路径，不把 base model 改成无自回归语言模型。

IFM 报告与 Uno 原论文描述的机制是：diffusion path 一次并行提出一组 draft tokens，AR base 在另一次前向中给出目标分布并验证；使用 Ψ-Spec 的标准 rejection sampling 逐 token 接受候选，首个拒绝处按归一化正残差分布抽取修正 token，全部通过时再抽一个 bonus token。该采样程序可精确保留 base AR 的输出分布。这里的“lossless”是分布意义，不保证随机采样的两次运行逐字相同，也不保证任意实现、硬件和批量下均加速。论文中的吞吐结果绑定模型、sampler、batch、context、硬件和 runtime。

## 研究材料

### 一手来源

1. K2-Horizon 作者报告：[K2-Horizon: A Frontier Fleet of Open Source Language Models](https://arxiv.org/abs/2611.21003v1)，Institute of Foundation Models，arXiv v1，2026-11-16。重点阅读 §2 / Table 1（Grouped RMSNorm、MoVA、gate 与参数配置）、§3.1（pretraining/midtraining schedule）、§3.4（36B vs 32B 训练比较）、§7.4（小模型评测）、§9（Uno）。支持作者公开的方法、配置、训练和实验叙述；不构成第三方复现或独立 benchmark。
2. IFM 官方 36B 模型卡：[IFM/K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B)，仓库提交 [`3fb5edfa484350459fb6b7461f80b5701b7dd6b4`](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/tree/3fb5edfa484350459fb6b7461f80b5701b7dd6b4)，API 记录 lastModified 2026-11-22。支持官方 checkpoint 身份、模型卡字段、长上下文和公开阶段信息；字段不证明任意任务效果。
3. 固定 36B 配置：[config.json](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/blob/3fb5edfa484350459fb6b7461f80b5701b7dd6b4/config.json)；固定实现：[modeling_k2_horizon.py](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/blob/3fb5edfa484350459fb6b7461f80b5701b7dd6b4/modeling_k2_horizon.py)。核对层范围、expert 数、route weights、value shape、gate 位置、KV 更新和 grouped norm；仅支持此 revision 的实现语义。
4. IFM 官方 3.7B 模型卡：[IFM/K2-Horizon-3.7B](https://huggingface.co/IFM/K2-Horizon-3.7B)，仓库提交 [`364e884e27fad049a325b9246902a28d25f2da11`](https://huggingface.co/IFM/K2-Horizon-3.7B/tree/364e884e27fad049a325b9246902a28d25f2da11)，API 记录 lastModified 2026-11-22。固定配置：[config.json](https://huggingface.co/IFM/K2-Horizon-3.7B/blob/364e884e27fad049a325b9246902a28d25f2da11/config.json)。支持该 dense checkpoint 的参数与层配置，不能据此反推 36B MoVA 性能。
5. IFM 首发说明：[Introducing K2 Horizon: Frontier Performance, Radically Open](https://ifm.ai/blog/k2/)，2026-09-03。提供 MoVA 的发布方设计动机和家族介绍；事实性细节优先用后续论文和固定配置核验。
6. Uno 原论文：[Unlocking Lossless Speedups in LLMs via Discrete Diffusion](https://arxiv.org/abs/2609.04010v1)，2026-09-03。支持 Uno、diffusion adapters、Diffusion Distillation 和 Ψ-Spec 的方法定义；该论文的 benchmark 设置不能外推到其他 K2 checkpoint/runtime。

### 教学材料（不作为独立事实证据）

- [K2 Horizon · Visual Learn](https://sghael.github.io/visual-learn/k2-horizon/)：按 dense FFN、MoE 和 MoVA 图示模型层分工；其 2026-09 叙述早于 IFM 技术报告，采用图示顺序时不沿用旧结论。
- [K2-Horizon-MoVA-36B: routing the value vectors, and why the KV cache does not notice](https://ai.thesatyajit.com/articles/k2-horizon-mova)，2026-09-22：沿一个 token 的 route→weighted value→cache update 排列机制，适合发现“路由发生在 value 生成而 cache 只收到混合值”的解释路径；其 snapshot 和计算须按当前官方报告/配置重新核验。

## 未支持的外推

- 36B-A4B 的 nominal “36B / 4B” 标签不能推出 4B dense 的存储、FLOPs 或端到端速度。
- 512K 配置与分阶段训练不等于所有 512K 输入都有效，也不保证 512K 下的固定延迟或并发量。
- 3.7B dense、32B dense、36B-A4B 的 benchmark 行不能视为 MoVA 单项消融；32B 是技术报告中的主要 dense 对照，模型间仍同时存在多项结构差异。
- Uno 的精确性是 Ψ-Spec 算法在相应分布/采样条件下的性质；速度取决于草稿接受情况和实测负载，不是模型卡保证。
- K2 0.9B 的 MOPD、K2 7B/0.9B Uno、K2 3.7B 和 36B-A4B 是不同 artifact/训练或推理对象；不可合并为同一模型 recipe。
- serving 命令、parser、TP/EP、kernel 调优、H200 实测和产品/API 协议不属于本章架构结论，需在推理/服务书册中按固定版本核对。
