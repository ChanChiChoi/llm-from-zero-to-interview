# Recursive state cache：递归语义、执行形态与实现边界

## 研究问题与范围

本轮为第二十一册第 71 章研究递归状态怎样表示前缀、逐步更新并服务于后续 token；训练/整段处理与自回归解码是否必须采用相同执行形式；固定状态的内存收益及信息访问代价；模型定义的数学状态与运行时实际缓存表示之间的边界。范围限于架构机制和模型状态语义，不把服务端调度、状态池、分页、租户隔离或跨机迁移写成本章的通用要求。

检索日期：2026-11-24。检索平台：Web 搜索、arXiv HTML、官方 GitHub 源码、Waterloo STAT 946 课程 wiki、方法团队博客。检索词包括 “RetNet recurrent/parallel/chunkwise recurrence notes”、 “Mamba recurrent mode parallel training selective scan”、 “Mamba2 allocate_inference_cache conv_state ssm_state”、 “SSM cache inputs not state ReplaySSM”。直接核对论文、固定版本源码和教学材料中的对应段落；不使用搜索摘要作为正文技术证据。

## 主要发现

1. 显式 KV cache 按历史位置保留可寻址的 key/value；递归模型则用状态转移把当前输入并入固定结构的表示，并从状态产生输出。递归状态不是自然语言摘要，也不意味着所有模型都只存一个张量。RetNet 的递归 retention 可作为具体矩阵状态例子：$S_n=\gamma S_{n-1}+K_n^{\mathsf T}V_n$，输出为 $Q_nS_n$；此式仅描述该机制，不能当作所有 SSM/递归模型的统一公式。
2. 递归表示不必规定整个序列都逐 token 串行执行。RetNet 明确给出 parallel、recurrent、chunkwise recurrent 三种计算形式，并描述块内并行、块间 recurrent 传递状态。Mamba 论文描述整段 convolutional/parallel 计算用于训练、autoregressive recurrent 计算用于逐步推理，并提出硬件感知 selective scan；这些说法各自绑定论文方法，不能推广为所有递归架构的固定执行策略。
3. 官方 Mamba 仓库 tag v2.2.4 中，Mamba2 的 step 接受一个 token，并接收/返回 conv_state 与 ssm_state 两项；allocate_inference_cache 分别创建卷积窗口状态和 SSM 状态。形状依赖模型配置。这证明此实现的运行时状态是多个组件的组合，不证明所有模型或框架都采用这两个对象，也不规定跨请求缓存管理方式。
4. ReplaySSM arXiv v1（2026-10-09）将 Mamba-2/Gated DeltaNet 递归状态表示为 checkpoint 加近期输入缓冲，在缓冲刷新时折叠更新；论文称其输出与常规解码在浮点差异范围内匹配。它说明递归数学状态和内存中的缓存布局可以不同。该工作仍是预印本，性能结论属于其指定 vLLM 实现、模型、GPU、批量与测量设置；本章不使用其性能数字，也不把其方案写成通用做法。
5. Waterloo STAT 946 W25 课程材料用 RetNet 递归式、并行式及 chunkwise recurrent 小节串联计算形态，适合作为教学组织参考。其一般性能及方法评价文字不作为本章技术证据；关键机制仍以 RetNet 原论文为准。

## 来源与核对位置

### 原始论文

- Yutao Sun et al., “Retentive Network: A Successor to Transformer for Large Language Models,” arXiv:2307.08621v1, 2023-07-17. https://arxiv.org/html/2307.08621v1
  - §2.1 “The Recurrent Representation of Retention”：式 (6)，状态更新与 query 读出。
  - §2.1 “The Chunkwise Recurrent Representation of Retention”：式 (7)，块内 parallel 与块间 recurrent 的组合。
  - §2.3 Training / Inference：训练使用 parallel 或 chunkwise，推理采用 recurrent 表示。
  - 支持：RetNet 的具体状态机制及其三种计算表示。实验速度和内存主张只支持论文内给定模型、A100、序列长度、实现与测量条件，不作通用加速结论。
- Albert Gu and Tri Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces,” arXiv:2312.00752v2, 2024-05-31 revision. https://arxiv.org/html/2312.00752v2
  - §2 “Structured State Space Sequence Models”：离散递归状态及输出关系；§2 Computation 说明 recurrence 与 global convolution 两种计算形式，并描述论文常用训练/自回归推理路径。
  - §3.3.2 “Overview of Selective Scan”：并行扫描、kernel fusion 与重计算的硬件感知实现动机。
  - §3.5 “Boundary Resetting”：说明独立序列拼接时需在边界隔离状态的语义问题。
  - 支持：Mamba 论文中所述机制与执行选择。不得把论文中的理论复杂度或实验吞吐外推至任意模型和硬件。
- Ze-Wei Liou et al., “ReplaySSM: Cache SSM Inputs, Not State,” arXiv:2610.13778v1, 2026-10-09. https://arxiv.org/html/2610.13778v1
  - §1 introduction、§2.1 与 §3：checkpoint 加近期 SSM 输入缓冲，分离输出计算与 checkpoint 更新。
  - 论文说明方案针对 Mamba-2 和 Gated DeltaNet，报告的等价性带浮点误差边界。
  - 证据等级：新预印本。支持“缓存实现可选择重建数学状态”的机制例证；不单独支持普遍生产收益、任意递归形式可直接套用或跨系统性能结论。

### 固定版本实现

- State Spaces, official Mamba repository, tag v2.2.4, mamba_ssm/modules/mamba2.py. https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mamba2.py
  - step（约 L1415 起）：输入仅支持一个 decoding token，显式接收卷积状态与 SSM 状态并原位推进。
  - allocate_inference_cache（约 L1534 起）：按配置分别分配 conv_state 与 ssm_state；后者维度包括 batch、head、head dimension 与 d_state。
  - 支持：该 tag 中 Mamba2 block 的实际状态组成和接口。源码不能代表其他版本、其他 SSM 或服务框架中的状态生命周期。

### 教学与实践解释

- University of Waterloo, STAT 946 W25 course wiki, “Retentive Network (RetNet): A Successor to Transformer for Large Language Models.” https://wiki.math.uwaterloo.ca/statwiki/index.php?title=stat946W25
  - RetNet 小节约 L822–854：依次组织递归式、并行式和 chunkwise recurrent 表示，提供机制串讲的教学路径。
  - 只用于检查解释顺序和术语衔接；其中对效率和模型优劣的概括不用于正文证据。
- Dao AI Lab, “ReplaySSM: Cache SSM Inputs, Not State,” 2026. https://dao-lab.ai/blog/2026/replayssm/
  - §3–4 用“近期输入缓冲 + checkpoint”解释状态重建、flush 与回滚的关系。
  - 作者团队直接参与该预印本，博客适合作为机制讲解入口；实验结论以预印本的模型、实现和测量条件为界，不作独立复现证据。

## 正文采用边界

- 用 RetNet 矩阵递归说明状态如何累积、衰减和被 query 读取，随后明确这是一个机制例子。
- 用 Mamba2 tag v2.2.4 说明单个 block 的推理状态可以包含多种内部记忆对象；逐版本、逐配置说明。
- 用 ReplaySSM 说明“数学状态”不等于某一种唯一的 device-memory layout；只作为 2026 年预印本案例，不放性能比较数字。
- 固定状态只表示状态张量尺寸不随已处理 token 数直接增长；总内存仍取决于层数、状态维度、精度、卷积历史、batch 并发及实现。逐 token 递归也不保证序列计算低延迟或整体更快。
- 能力比较须用同任务、同模型/训练预算及质量指标对齐；不能仅由状态大小推出精确检索能力、长上下文质量或生成速度。批处理和服务状态池的运行策略归系统书讨论。
