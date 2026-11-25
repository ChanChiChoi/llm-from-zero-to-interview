# 第 82 章 K2-Horizon：3.7B Dense、MoVA 36B-A4B 与 Uno 扩散适配器

本章围绕同一模型家族中三种不同层次的设计对象展开：3.7B 是 dense decoder，36B-A4B 同时在 attention value 与 FFN 上采用稀疏专家，Uno 则是在 0.9B 和 7B 模型上附加的 diffusion draft/AR verification 路径。中心问题是：稀疏路由具体改动了 Transformer 哪条计算路径、怎样改变参数与 KV 状态账本，以及公开比较究竟支持哪些架构结论。依据 IFM 的 K2-Horizon 技术报告 v1、2026-11-22 固定模型配置与实现、Uno 原论文区分名义模型规模与精确参数口径，避免把 3.7B 错当成 36B MoVA 的 matched dense baseline，也不把 Uno 写成 36B 模型变体。第 81 章承接 DeepSeek V4.1 的近期模型结构，第 21 章承接 MoE 路由基础，第 6、24 册承接服务引擎与推测解码的完整操作/评测。

## 82.1 固定 K2-Horizon 的模型身份与参数口径

本节先把家族名、具体 checkpoint、模型卡和技术报告分开，再解释 3.7B、36B-A4B 这些名义标签与 total/backbone/active 参数计数的差异。IFM 报告 Table 1 将 3.7B 的总参数记为 5.058B、backbone 记为 3.775B；36B-A4B 对应 37.445B total、36.162B backbone、4.665B active backbone/token。读者应知道这些值各自包含什么，并在后续结构比较中使用一致口径。

### 82.1.1 报告、模型卡与固定 revision 分别支持什么

以 2026-11-16 的 IFM 作者报告和 2026-11-22 的官方模型仓库提交为当前证据对象，区分报告给出的训练/评测结论、配置提供的结构字段、实现提供的前向路径。说明固定 revision 能复核该配置与代码，不代表训练权重、全部运行后端或之后提交都与其等价。

### 82.1.2 名义规模、存储参数和 active 参数不能互换

比较 3.7B 与 36B-A4B 的名义名称、total、backbone 和 active backbone 数量；说明 36B-A4B 的“约 4B active”是发布标签，而报告精确表记录 4.665B active backbone。总参数涉及权重驻留，active 参数近似表示单 token 激活的模型计算容量，二者都不能直接换算显存、FLOPs 或生成速度。

## 82.2 3.7B Dense 与 36B-A4B 的 block 排布

本节按固定配置和模型实现建立两种前向路径的对照。3.7B 为 36 层 dense decoder；36B-A4B 为 48 层模型，前 3 层是普通 attention 加 dense FFN，后 45 层同时启用 MoVA attention 与 sparse MoE FFN。两个模型共享 hidden size、GQA 头数和长上下文位置配置，但深度与 FFN 路径不同；3.7B 是家族中的 dense sibling，36B 的主要 dense 训练比较对象是 32B 模型。

### 82.2.1 3.7B 的 dense attention、GQA 与 FFN

按 3.7B 固定 config 说明 36 层、hidden size 2,560、32 个 query heads、8 个 KV heads、head dimension 128 和 dense FFN intermediate size 10,240；`num_experts=0`、`mova_num_experts=0` 且 attention gate 关闭。该路径用于认识同家族 dense decoder，不等于 36B-A4B 除规模外完全相同的模型。

### 82.2.2 36B-A4B 的前三层与后四十五层

依据 `mlp_only_layers=[0,1,2]` 与 `K2HorizonDecoderLayer` 实现解释层号范围。前三层走普通 GQA attention 与 dense FFN；其后每层以 MoVA 生成 value、以 FFN MoE 变换 token 表示。指出 `mlp_only_layers` 不能被误读成“前三层只换 MLP、attention 仍启用 MoVA”：实现对两种子层使用同一 sparse-layer 条件。

### 82.2.3 3.7B 不是 36B 的 matched dense baseline

比较 3.7B 与 36B 的共同结构字段和不同层数/FFN 尺度，限定 3.7B 的用途是 dense 家族案例。IFM 报告用于比较 MoVA 模型训练轨迹的 dense 模型是 32B；不能用 3.7B 与 36B 的榜单分数差归因 MoVA。

## 82.3 Grouped RMSNorm：在特征组内控制归一化尺度

本节介绍 3.7B 与 36B-A4B 均使用的 Grouped RMSNorm，说明它保留 RMS normalization 的中心化缺省方式，同时按特征分组计算均方根。分组数控制每组宽度，报告据此讨论归一化输出幅度上界与训练稳定性；结论限定为 IFM 的模型设计和报告结果，不能推广为任意模型训练保证。

### 82.3.1 标准 RMSNorm 与组内均方根

说明 RMSNorm 对特征向量按均方根缩放，再把分组索引引入计算；定义组大小、epsilon 和逐元素输出，使读者理解从全隐藏维度归一化到分组归一化改变的统计范围。

### 82.3.2 上界和稳定性主张的适用边界

在省略可学习缩放参数的归一化值上，报告推导单个分量的幅度界与组宽度平方根相关；配置中 3.7B 和 36B-A4B 均使用两组。说明作者报告观察到的稳定性改善属于其训练规模与 recipe，不等于架构级稳定性证明或普遍效果。

## 82.4 MoVA：稀疏专家改变 attention 的 value 生成路径

本节把 Mixture-of-Value Attention（MoVA）放在标准 attention 数据流中的准确位置：query-key attention 仍决定从哪些历史 token 读取，MoVA 则在每个 token 的 value 形成之前选择 value projection experts。每个被选专家输出一组 GQA value heads，经非线性变换与路由权重合并后，才交给常规 attention；这不是复制完整 attention heads，也不是路由历史位置。

### 82.4.1 从单一 value projection 到 per-token value routing

回顾 Q、K、V 的职责和 32Q/8KV GQA 共享关系，再说明普通 `V=XW_V` 如何被 64 个候选 value projections 及 token-conditioned router 取代。定位 MoVA 改的是 value 表示生成，不改 query/key 的内容选择公式。

### 82.4.2 sigmoid score、selection bias、top-4 与混合权重

依据固定实现区分三步：sigmoid 原始分数生成，router bias 只参与 top-4 选择，选择后取回原始分数并归一化、乘 2.5。说明该 scaling 让最终权重和为 2.5，因此不能把它们称作和为 1 的概率，也不能将 bias 后的选择分数直接用作混合权重。

### 82.4.3 value experts 的形状、SiLU 与单 token 输出

说明每个 value expert 将 2,560 维输入映射为 8×128 维输出，64 个专家中每 token 执行 4 个；各输出经 SiLU 后按归一化并缩放的路由分数求和。该非线性专家混合产生一个与普通 GQA value tensor 同形状的向量，数量与机制分别绑定固定 config 和代码 revision。

### 82.4.4 MoVA 与 query-key attention 的连接

解释合成后的 value 按 KV head 进入 causal attention，由 query-key scores 汇总不同历史位置的 value。读者应能区分“决定当前 token 的 value 如何编码”和“决定当前 query 读取哪些历史位置”两种选择；两者沿同一 attention block 衔接但不是同一个 routing operation。

## 82.5 FFN MoE：与 MoVA 并列的第二条稀疏路径

本节说明 36B-A4B 在 attention 之后的 FFN 路径又使用一套不同的专家路由：每稀疏层有 100 个 routed FFN experts、每 token top-8，并额外执行 1 个 shared expert。该路径输出 token 的变换结果，与 MoVA 在 attention 前生成 value 的阶段、形状与计算任务均不同；两类 top-k 不可相加成“top-12 expert”。

### 82.5.1 routed expert 与 shared expert 怎样共同生成 FFN 输出

解释 FFN router 依据当前层表示选择 top-8 routed experts，所选输出加权聚合后与 shared expert 输出相加；明确 100 是候选容量、8 是每 token routed 数、shared expert 是始终参与的公共路径。

### 82.5.2 一个 token 怎样依次走过 MoVA 与 FFN MoE

沿一层 36B-A4B 后段 block 的结构说明：输入表示先形成 Q/K 与路由后的 V，attention 读取历史后产生当前表示，再进入 FFN 的另一套路由。指出两个 router 的输入状态、专家参数、阶段顺序和输出作用不同，不能把两处选择简化为一个专家池。

### 82.5.3 stored、active、attention 与 dispatch 成本分账

用官方参数表区分 37.445B total、36.162B backbone 与 4.665B active backbone/token，并说明参数数字没有纳入完整设备工作集和端到端延迟。MoVA 每层 value expert 权重及每 token top-4 的数量级可由固定形状推算；通信、负载、batch 重用、padding 与 kernel 会另行影响实际成本。

## 82.6 Softplus Attention Gate：调节 attention 分支输出幅度

本节介绍 36B-A4B 使用的 softplus attention gate，并与 MoVA router、FFN router 区分。门控从当前 hidden state 生成逐元素系数，乘在 attention 汇总结果上、位于 output projection 之前；配置使用 `beta=ln(2)`，使零附近的 gate 约为 1，既能衰减也能放大 attention 分支。

### 82.6.1 gate 在 attention block 的位置与函数

说明 query-key attention 得到的输出如何乘以从 hidden state 生成的 gate，再进入 output projection；给出 softplus 关系及 `beta=ln(2)` 的初始化尺度性质，并保持 per-head/per-feature 的张量语义。

### 82.6.2 三类路由/门控的输入与作用对象

对齐 MoVA router、FFN router 与 attention gate：前两者在候选专家中选择并加权，后者缩放 attention 输出特征。指出 3.7B 配置将 attention gate 关闭，不能把 36B 的 gate 视为所有 K2-Horizon 模型共有机制。

## 82.7 512K 上下文由阶段训练与位置配置共同支持

本节将 `max_position_embeddings` 与模型训练过程结合。官方报告给出从 8K pretraining 到 32K、128K、512K 的连续 midtraining 阶段，并同步调整 RoPE base；3.7B 与 36B-A4B 的模型卡均标示 524,288 上下文。配置值和阶段记录解释公开支持范围，仍不能替代针对特定长文检索任务的有效性评估。

### 82.7.1 8K、32K、128K 与 512K 阶段的 token 预算

分别说明 pretraining 与四个 midtraining stages 的 sequence length 和增量 token budget，并指出 checkpoint 之间按训练链继续。强调“阶段 token 数”不是互不相关的数据集规模，避免把上下文长度写成只由某一次配置修改获得。

### 82.7.2 RoPE base 随上下文扩展的变化

对照报告中 pretraining、Stage 1 与后续扩展阶段的 RoPE base，以及当前 3.7B/36B config 的最终 theta 值。解释位置长度和频率尺度是同一训练路线中的配置因素；不能从当前 `rope_theta` 单独推导模型在任意长度上的能力。

### 82.7.3 配置上限、训练经历和任务质量是三种证据

将模型卡的 512K 声明、训练序列长度以及长上下文 benchmark 分开说明。报告/模型卡支持“公开模型按阶段训练并配置到 512K”；对任意证据位置、任务复杂度、质量退化与服务并发仍需专门评测。

## 82.8 GQA KV cache：MoVA 不增加 expert 副本，绝对 cache 仍很大

本节从配置计算显式 K/V cache 的结构大小，区分相对变化与绝对容量。MoVA 将专家输出在 cache update 前合并，因此每层仍缓存普通 GQA 的单份 K/V；但 48 层、8 KV heads、128 head dimension 与 512K 序列共同形成很大的未压缩缓存。

### 82.8.1 从层数、KV heads 和 head dimension 推导 cache 公式

定义 batch、context token 数、层数、KV heads、head dimension、每元素字节数，以及 K 和 V 两份状态，给出 BF16 显式 cache 的容量关系。说明该式不包括权重、allocator、page metadata 和 workspace。

### 82.8.2 36B 与 3.7B 的 512K 静态估算

在 batch=1、BF16、不量化、不使用滑窗的假设下，分别按 48 层与 36 层计算 36B 和 3.7B 的理论 KV 容量。标出结果是静态 shape 推算，不是实测峰值显存，也不含框架压缩和并发状态。

### 82.8.3 “MoVA 不加 cache”不等于“512K cache 很小”

说明 MoVA 只避免把每个 value expert 的中间结果都存入历史；它不压缩 GQA 已产生的 K/V，也不缩短历史长度。将该机制优势和长上下文模型的绝对 KV 需求分开陈述。

## 82.9 Dense 对照能支持的模型级结论

本节依据 IFM 技术报告解释 MoVA-36B-A4B 与 32B dense 的公开比较。两者在相同训练数据与设定下比较时，作者报告 36B-A4B 的 pretraining loss 接近但略落后，若干 1.1M-step benchmark 相差 1.2–3.7 分，active backbone 参数约为 dense 模型的七分之一。该结果属于完整模型组合层级，不能把差异归因于 MoVA 单一部件。

### 82.9.1 为什么真实 dense 比较对象是 32B

区分小型 3.7B dense sibling 与报告的 32B dense matched comparator；说明报告对 32B/36B 的数据、训练设置和 checkpoint 评估提供了比较依据。模型规模和多个结构因素仍不同，匹配训练条件不等于单变量因果实验。

### 82.9.2 预训练 loss、benchmark 差距与 active-parameter 比例

按报告原文核对 1.1M-step 处作者列出的指标和差值，说明比较的是哪个 checkpoint、何种评测阶段，以及这些观察支持“较少 active parameters 下接近该 dense run”这一有限主张。不得将它改写为所有任务、后训练 checkpoint 或实际服务速度的普遍等价。

### 82.9.3 不能从整体比较推出 MoVA 单项贡献

指出 36B 模型同时包含 FFN MoE、MoVA、softplus attention gate、不同深度/宽度及 grouped norm 设定；报告没有在该对照中隔离所有组件的单项因果效果。3.7B final benchmark 行同样不构成 MoVA 消融或架构性能证明。

## 82.10 Uno：在 0.9B 与 7B 上附加并行 diffusion draft

本节把 Uno 作为 K2-Horizon 家族报告中的旁支解码机制，而不是 3.7B/36B-A4B 的内部 block。IFM 报告 §9 只说明它把 diffusion LoRA adapters 加到完整训练后的 0.9B 与 7B 模型；冻结 AR base 定义目标输出分布，轻量 adapter 产生并行 draft，`Ψ-Spec` 负责精确拒绝采样验证。

### 82.10.1 Uno 的 model scope 与 MoVA backbone 边界

固定 Uno 对应 K2-Horizon-0.9B 与 K2-Horizon-7B 的报告/adapter release，明确报告未将其应用到 3.7B 或 36B-A4B。说明一个方法沿 K2-Horizon 家族发布，不表示同一 adapter 属于所有模型变体。

### 82.10.2 冻结 AR 权重与 diffusion LoRA 分支

说明 AR 主干仍用 next-token prediction 分布，LoRA adapter 接入 attention 和 MLP 投影，训练时冻结 AR 参数，只训练 diffusion 权重。报告给出 rank-128、alpha 256、7B-token distillation 和递增 block/context schedule；这些字段只描述报告中的 Uno 训练对象。

### 82.10.3 一次并行 draft、一次 AR 验证与 rejection sampling

用目标分布 `p`、draft 分布 `q` 和一个短 token 例子展示候选接受率 `min(1,p/q)`、拒绝后的正残差重采样，以及全接受时增加一个 bonus token。说明 draft 并行提出多个位置，AR verifier 决定最终提交序列，不能把 draft 直接当作已验证输出。

### 82.10.4 “lossless”的分布含义与吞吐条件

说明 `Ψ-Spec` 在所述采样算法下保持 base AR 输出分布，并不保证两次随机生成逐字相同，也不保证任意 batch、硬件或引擎都有速度提升。报告吞吐必须绑定模型、sampler、block size、batch、context、硬件与 runtime；本章不复写 serving engine 命令或生产测试手册。

## 82.11 面试讨论：把结构、容量和评测分层回答

本节将本章重点转成有边界的架构解释：读者应能沿 token 路径说出 MoVA 与 MoE 各自选择什么、解释参数和 cache 账本，并检查报告对比的因果范围。回答要求说明固定对象与前提，而不是背诵模型名或单项 benchmark 分数。

### 82.11.1 MoVA 与 FFN MoE 分别路由什么

围绕 value projection、FFN expert、top-k、共享专家、Q/K attention 与 cache update 的相对位置，要求说明两条稀疏路径为什么不是一个 top-12 层。

### 82.11.2 36B-A4B 与 3.7B/32B 的比较口径

要求区分 nominal size、total/backbone/active 参数，指出 3.7B 是 dense sibling、32B 是作者报告中的主要 dense comparator，并说明训练条件匹配不等于 MoVA 单项消融。

### 82.11.3 为什么 512K 和 Uno speedup 都需要额外条件

讨论配置上限与任务级能力的区别，以及 Uno 算法精确性与吞吐的区别；回答中应给出需要固定的 checkpoint、sampler、context、batch、hardware 和 evaluation protocol。

## 82.12 练习：复原路由路径、缓存和 verifier 分布

练习聚焦公开配置与算法机制，不要求下载完整权重或运行模型。计算与假设值应标作结构推算或教学设定，不冒充 K2-Horizon 实测。

### 82.12.1 复原两种模型的层与参数表

从固定 config/code 列出 3.7B 全 dense 路径和 36B 前 3/后 45 层路径；把 MoVA、FFN MoE、shared expert、attention gate 与 group norm 分别填入表格，并标注 total/backbone/active 参数来源。

### 82.12.2 计算 MoVA value 权重与 512K KV shape

用 sigmoid raw score、selection bias、top-4、归一化和 2.5 scaling 手算一次混合权重；另按 batch=1/BF16 计算 3.7B 与 36B 的全长 KV 静态估算，并列出未包含项。

### 82.12.3 用小型分布走查 Ψ-Spec 接受与回退

给定一组教学用 `p` 与 `q`，对 draft token 计算接受概率，并在拒绝时从正残差重新抽样；验证最终单位置分布与 `p` 一致，再解释该结论不等于每次生成字符串一致或系统吞吐必然提高。

## 82.13 来源、版本与前后章节

本章主要事实依据 IFM 作者报告 arXiv:2611.21003v1、36B/3.7B 官方模型卡及各自固定 config、36B 固定模型实现和 Uno 原论文。参数形状、gate/router 语义和 cache 推算要绑定具体版本；模型卡 benchmark、报告自报效果、实现字段和静态账本分别标注证据等级。第 21 章承接 MoE，相关系统与 serving 交给第 6、24 册；0.9B MOPD 及其他 K2 family 训练分支不迁移为 36B 架构事实。
