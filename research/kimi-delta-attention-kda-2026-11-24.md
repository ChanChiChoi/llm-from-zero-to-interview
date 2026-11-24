# Kimi Delta Attention（KDA）研究记录

研究日期：2026-11-24

## 研究问题与检索范围

核对 Kimi Linear 所称 Kimi Delta Attention 的状态更新、逐通道衰减、delta correction 与 chunkwise 训练；区分该机制与 Gated DeltaNet、Gated MLA 的职责；追踪 Kimi Linear 与 Kimi K3 报告之间的版本差异；检查 KDA 实验和 kernel 速度主张能够支持的范围。

检索词覆盖中英文：Kimi Delta Attention / KDA equation / KDA explained / Kimi Linear chunkwise WY UT / Gated Delta Networks scalar decay / Kimi K3 log decay g_min / FlashKDA chunk / KDA lecture notes / KDA tutorial. 资料类型包括原始技术报告、官方实现说明、研究者教程和大学课程讲义。

## 研究发现

### Kimi Linear: An Expressive, Efficient Attention Architecture

- 来源：Moonshot AI，arXiv:2510.26692，v2；论文最初提交于 2025-10-30，v2 日期 2025-11-01。
- 入口：[arXiv v2 摘要](https://arxiv.org/abs/2510.26692v2)；[HTML 全文](https://arxiv.org/html/2510.26692v2)。
- 阅读位置：§3 的 KDA 定义与状态方程；§3.1 的 chunkwise WY/UT 训练形式；§4 的模型组合；§4.4 的 output gate；§5.1 的合成任务；§5.6 的模型速度比较。
- 支持的主张：KDA 将 Gated DeltaNet 的标量衰减替换为逐 key channel 的对角衰减。令 $S_t\in\mathbb{R}^{d_k\times d_v}$、$k_t,q_t\in\mathbb{R}^{d_k}$、$v_t\in\mathbb{R}^{d_v}$，报告给出的递归为
  $S_t=(I-\beta_t k_tk_t^{\mathsf T})\mathrm{Diag}(\alpha_t)S_{t-1}+\beta_t k_tv_t^{\mathsf T}$，读出为 $o_t=S_t^{\mathsf T}q_t$。等价读写视图是先衰减 $S_{t-1}$，再用当前 key 的 value 预测误差做 rank-one correction。$\alpha_t$ 是 key-channel retention vector；$\beta_t$ 是每 token/head 的 scalar write strength。
- §3.1 支持 KDA 以 chunkwise WY/UT 形式进行批量训练计算，同时保留边界状态传递的思路；正文不需要把论文全部矩阵推导重印。
- §4.1 的模型组合是特定 Kimi Linear 配置，使用三层 KDA 配一层全局 MLA。它不是所有 KDA 模型的强制比例。
- §4.4 的 output gate 在 Kimi Linear 中采用低秩参数化；output gate 与状态衰减 $\alpha_t$、写入强度 $\beta_t$ 不是同一作用。
- §5.1 的 Palindrome、MQAR、Stack 合成任务使用 2 层、2 头、head dimension 128 的小模型，序列长度 256—2048，最多 20,000 个训练 step 并搜索 learning rate。该实验设置只能支持相应合成任务与配置中的观察，不能直接证明真实长文档检索或一般模型规模的优劣。
- §5.6 的模型速度比较绑定论文中的模型、序列长度、batch 和硬件条件；不能当作 KDA 普遍端到端加速结论。
- §6.1、§6.2 讨论 KDA 的位置解释、DPLR 关系与 chunk 内计算性质；不能据此把对角衰减写成标准位置 embedding 表。

### Gated Delta Networks

- 来源：原论文预印本，arXiv:2412.06464，v3，2025-03-06。
- 入口：[arXiv v3](https://arxiv.org/abs/2412.06464v3)；[HTML 全文](https://arxiv.org/html/2412.06464v3)。
- 阅读位置：§2—§3 的 gated delta rule 与 hardware-efficient chunkwise training。
- 支持的主张：Gated Delta Networks 将 delta correction 与按时间/head 的 scalar decay 结合，并发展适合硬件的 chunkwise 训练实现。该来源用于有限对照 scalar decay 与 KDA per-key-channel diagonal gate。
- 不支持：不应把一个 GDN 参数化当作所有叫作 gated DeltaNet 的实现；也不支持把 GDN 和 KDA 写成张量公式、输出门或状态布局完全相同的模块。

### Kimi K3 technical report

- 来源：Moonshot AI，arXiv:2607.24653，v2，2026-08-07。
- 入口：[arXiv v2 摘要](https://arxiv.org/abs/2607.24653v2)；[HTML 全文](https://arxiv.org/html/2607.24653v2)；[官方 Kimi K3 仓库](https://github.com/MoonshotAI/Kimi-K3)。
- 阅读位置：§2.1.1 的 KDA 状态转移及模型架构说明。
- 支持的主张：K3 v2 给出与 Kimi Linear 同类型的 KDA 状态递归，同时报告了实现参数化变化：将 Kimi Linear 使用的可能无下界负 Softplus log-decay 改为有下界的 scaled sigmoid，给出 $g_{\min}=-5$；报告将其与控制累计衰减的倒数范围和 dense Tensor Core tile compute 联系起来。K3 也将 Kimi Linear 的低秩 output gate 改为 headwise RMSNorm 后的输入相关 full-rank output gate。
- §2.1.1 的层安排属于 K3 模型事实：每三层 KDA 接一层 Gated MLA，并在末尾另加 Gated MLA。
- 不支持：上述 decay 参数化、$g_{\min}$、output gate 和层比例不能变成 KDA 的通用定义，也不能将 K3 整体报告结果归因于 KDA 单一组件。

### FlashKDA

- 来源：Moonshot AI 官方仓库 [FlashKDA](https://github.com/MoonshotAI/FlashKDA)；[2026-04-20 deep dive](https://github.com/MoonshotAI/FlashKDA/blob/master/docs/20260420-flashkda-v1-deep-dive.md)。
- 阅读位置：deep dive 的 chunk 选择、kernel 分工、数值范围和性能部分；仓库 README 的设备及依赖支持说明。
- 支持的主张：该实现文档说明特定 v1 kernel 选择 CHUNK=16，结合其门值下界和 bf16 数值范围，并将预处理/块内工作与递归传递分开；文档报告相对于其 one-kernel prototype 的速度变化。仓库 README 和配置支持当前实现声明的硬件/张量限制。
- 边界：这是某实现、kernel 切分和对照原型下的工程观察，仓库默认分支会更新；不能推导成 KDA 算法的固有加速，也不能替代 Kimi Linear/K3 整体模型吞吐或端到端服务数据。

### 教学与课程材料

- Simon Dong, “Linear Attention, From Scratch to Kimi 3 (Kimi Delta Attention)”, 2026-08：[文章](https://simondong1.github.io/linear-attention.html)。从 key-value association matrix、delta correction、逐通道 forget gate 到 KDA 完整递归逐步展开，并对照 Kimi Linear/K3。它用于检查初学者解释和张量轨迹；公式与模型事实仍回到原始报告核验。
- CUHK Shenzhen CSC6203 Fall 2026, Lecture 04 “Transformer and Beyond”：[课程 slides PDF](https://nlp-course.cuhk.edu.cn/materials/2026_fall_slides/Lecture04_Transformer_and_Beyond.pdf)。相关页面在 KDA、MLA 与混合层安排中组织顺序递归、可学习的通道衰减及 Kimi 的 3:1 案例。课程材料用于教学组织与概念关系，不替代模型的版本化来源。

## 可采用的机制表述与边界

1. KDA 的精确教学式可按“衰减旧状态 → 用当前 key 读取预测 → 计算 value 误差 → 沿当前 key 做 rank-one 写入 → 用 query 读出”组织；此表达与 Kimi Linear v2 论文方程等价。
2. $\alpha_t$ 沿 key/state 行维逐通道缩放旧状态；$\beta_t$ 是写入误差的标量强度。二者不能混称一个 gate，也不能与 output gate 混为一谈。
3. KDA 的核心矩阵 state 大小不按历史 token 数线性增长；信息地址因此从 token 级位置变成有限矩阵关联。固定状态不等于保存全部历史、精确回溯来源或普遍胜过 attention。
4. 递归定义与训练 kernel 是两个层次。KDA 的 chunkwise WY/UT 形式说明一种训练并行排布；特定 FlashKDA kernel 性能不能外推到所有实现。
5. Kimi Linear 和 Kimi K3 报告共享 KDA 主递归思路，但 decay parameterization、output gate 和 layer schedule 有版本差异；正文应逐项绑定来源。

