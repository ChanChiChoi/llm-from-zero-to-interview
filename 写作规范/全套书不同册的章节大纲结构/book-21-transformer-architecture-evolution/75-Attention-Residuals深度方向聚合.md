# 第 75 章 Attention Residuals：让网络沿深度选择历史表示

本章围绕 Transformer 的残差流如何汇总网络深度上的子层结果，解释固定加法的优点与 PreNorm 中可能出现的贡献稀释，再说明 AttnRes 如何以深度 softmax 选择历史表示。重点是把 token embedding、attention/MLP 子层增量、可学习伪查询、RMSNorm key 和加权 value 组成同一个可追踪过程；随后说明 Block AttnRes 用块内 partial sum 与块间历史表示减少存储和通信，并厘清 Full、Block、Kimi K3 配置及后续研究各自的范围。读者需能复走一个 token 的 Full 与 Block 深度读取，解释 zero-init、成本和证据条件，并区分深度聚合、token attention、early-exit 与多路 residual。KDA 的递归状态机制由第 61 章承担，Kimi Linear 中 KDA 与全局 MLA 的组合案例见第 76 章；mHC 的多流约束由第 78 章承担，Kimi K3 的全模型结构由第 88 章承担。

## 75.1 标准残差把历史计算累积到同一条状态

本节把 Transformer 的一次 attention 或 FFN 残差写入当作一项深度更新，明确每次更新的输入、输出增量和累积状态。固定单位权重让信息与梯度有直接路径；论文提出的 PreNorm dilution 是特定机制与实验观察，不能写成所有深层模型必然失稳。

### 75.1.1 Transformer block 内有 attention 与 FFN 两次残差写入

说明同一 Transformer block 的 attention 子层和 FFN 子层会先读取当前表示，再各自把输出增量写回残差流；后续推导用“深度步骤/子层”作为索引单位，避免把 decoder block 数与 residual sublayer 数混为一谈。

### 75.1.2 固定单位系数让旧增量无法在读入时单独调节

标准残差可展开为 embedding 加所有先前子层输出增量的和。当前层可以改变自己写入什么，但传统加法本身不能按当前内容为不同历史增量选择独立系数；这解释了方法动机，不预设固定残差在所有任务上都不足。

### 75.1.3 AttnRes 改的是深度信息汇总，不是 token 注意力或执行层数

本节限定作用轴：每个 token 的当前子层在深度上读取此前子层/块表示，而常规 self-attention 仍沿 token 位置建立交互。深度权重会调节信息混合，却不会自动跳过未运行的层，也不是 early-exit。

## 75.2 Full AttnRes：以深度 softmax 读取历史来源

Full AttnRes 将 embedding 和每个先前子层写入保留为来源，目标子层先用伪查询与各来源计算深度分数，再将归一化后的历史值组成自己的输入。中心是“同一个历史来源如何既参与打分、又向输出贡献内容”的对应关系，而非仅列出 softmax 公式。

### 75.2.1 source bank 包含 embedding 与子层输出增量

定义 $v_0$ 为 token embedding、$v_i$ 为第 $i$ 个子层产生的增量；当前读入位置 $l$ 只能访问 $i<l$ 的已生成来源。区分累积后的 hidden state 与写入 source bank 的单项增量。

### 75.2.2 每个读入位置用一个可学习伪查询

说明 $q_l=w_l$ 是目标深度步骤对应的可学习向量，对同一读入位置的各 token 共享；它不是标准 token attention 中从当前 token hidden state 投影得到的 query。

### 75.2.3 同一历史向量同时作为 key 与 value

定义 $k_i=v_i$ 的来源关系，说明 RMSNorm 进入打分路径、未归一化的 $v_i$ 进入加权输出。RMSNorm 限制仅由 source 幅度造成的 score 支配，不等于对 residual output 做相同归一化。

### 75.2.4 Softmax 沿深度来源归一化

给出 $w_l^\top \mathrm{RMSNorm}(k_i)$ 生成 score，$\alpha_{i\to l}$ 在当前可见的深度来源上归一化，再形成 $h_l$ 的数学关系。指出权重随 token 输入变化的原因是 key/value 来源随 token 变化，而不是 query 逐 token 生成。

## 75.3 用一个 token 复走 Full AttnRes 的读入

本节用同一组历史来源走完 score、softmax 和 value 聚合，让读者能从单项 source 追到目标子层输入。数值仅用于演示归一化和内容混合，不冒充论文实验。

### 75.3.1 从三条历史来源算出深度权重与混合结果

采用三个二维 value 与明确标注的教学分数，手算 softmax 权重及加权输出，指出分数、归一化权重和向量结果分别对应 attention 的哪一环。

### 75.3.2 静态伪查询仍会产生 token 相关的权重

保持同一 $w_l$，改变该 token 的来源向量，使 key 分数和权重发生变化；再与另一个 token 的来源对照，分清“query 参数共享”与“路由结果相同”不是一回事。

### 75.3.3 零初始化伪查询得到均匀平均，不等于恒等残差初始化

伪查询全零时 logits 相同，初始 $\alpha$ 对所有可用来源均匀分配。说明它与标准 residual 的固定加法在函数上不同，不能把其他讲解材料中的 one-hot/identity warm start 当作原论文设置。

## 75.4 Block AttnRes：块间保留摘要，块内保留 partial sum

Block AttnRes 在表示来源的粒度与传统加法之间折中：一个 block 内照常累加新生成的子层增量，block 完成后才把其 partial sum 纳入可被后续深度读取的历史。需要清楚交代首层、块内后续层、块边界和输出层各自可见哪些来源。

### 75.4.1 连续子层增量形成一个已完成块表示

定义块 $n$ 由哪些深度步骤组成，以及 $b_n=\sum_{j\in\mathcal{B}_n}v_j$ 如何由块内输出形成；embedding $b_0$ 单独保留，不并进第一个块增量。

### 75.4.2 首个子层与后续子层看到不同候选集合

第一个子层读取 embedding 和全部已完成块；当本块已有输出后，partial sum 作为额外来源加入。通过块内执行轨迹说明每次读取发生在当前 delta 产生之前，避免把本层输出泄漏回自己的输入。

### 75.4.3 Attention 与 FFN 分别读取，输出头也需最终聚合

说明 attention/FFN 各自有读入参数并分别写入本块 partial sum；最后的输出读入将所有已完成块表示与 embedding 汇总后交给最终 normalization/head。它不应被简化成“每个 decoder block 只做一次 Attention Residual”。

### 75.4.4 区分 block 数、block 宽度与 residual sublayer 数

定义 $N$ 为块数、$S$ 为每块深度步骤宽度，说明每个 Transformer block 通常包含 attention 与 FFN 两个 residual sublayer。论文 scaling sweep、48B Kimi Linear 实验和 Kimi K3 的具体设置分开陈述，不让一个数字替代另一组配置。

## 75.5 让深度读取在训练与推理中可承受

本节将数学候选数映射到保存状态、读取和跨设备通信，解释 Block AttnRes 的工程动机与论文的调度方案。复杂性要按 token、层、块、并行切分和阶段分开，不能把大 O 直接说成端到端加速。

### 75.5.1 Full AttnRes 的算术与历史存储随深度增长

按每 token 估算 Full 的 $O(L^2d)$ 分数/加权运算和 $O(Ld)$ source 保存，再说明 batch/sequence token 数怎样扩展张量规模；区分算术量与重计算、激活存留、内存带宽和 pipeline communication。

### 75.5.2 Block AttnRes 以块级历史源降低访存压力

说明 block 机制把主历史 bank 从 $L$ 项压到约 $N$ 项并保留当前 partial，读入仍需对这些候选形成权重。归纳计算量、状态量与 block width 的关系，不将“更少 source”直接等同为相同倍数的 latency 降低。

### 75.5.3 两阶段计算批量处理已完成块并在线合并 partial

第一阶段将同块伪查询并行计算对已完成 block representations 的注意力与 softmax 统计；第二阶段按子层顺序加入新 partial source，并用 online-softmax 的 max、归一化因子和加权值统计合并，得到与整组候选一次 softmax 等价的输出。

### 75.5.4 Pipeline cache 与 GPU kernel 决定实际代价

交代跨 stage 缓存只传递新完成块的作用；再以 Kimi K3 on MI355X 工程报告说明单遍读取、token-row 分片、TP collectives 融合和 register pressure。论文的低于 4% training/低于 2% typical inference 与厂商 kernel 对照均绑定各自条件。

## 75.6 论文证据：损失、梯度、下游表现和开销分别判断

本节对齐 baseline、compute、训练 token、模型和 benchmark，解释论文给出的 Full/Block 证据，以及哪些信息仍不能从结果推导。重点是分清机制诊断、同预算比较和产品模型结果。

### 75.6.1 Scaling law 比较报告特定预算下的 compute 等效

说明论文在多个小型尺度、8192-token context 和基线选定超参数下比较 PreNorm、Full 和约 8 blocks 的 Block AttnRes；1.25× 是指定拟合/损失点上的 compute 等效，不能泛化为所有训练都节省 20% 计算。

### 75.6.2 16-layer 消融分离 Full、Block 与 baseline

引用论文 16-layer 消融中的 validation loss 设置，比较 Full、指定 block size 和 baseline；明确它只支持该消融中的相对关系，不能证明任意规模、块宽或数据都保持同等排序。

### 75.6.3 Kimi Linear 48B 实验报告训练动态和下游指标

交代 48B 总参数/3B activated、1.4T-token training recipe、baseline 对照和被测任务。用少量指标呈现改善并包括未变项目，使“论文报告在被测集合中占优”与“每项能力都普遍改善”保持区别。

### 75.6.4 成本数字和 PreNorm dilution 解释都保留实验范围

分别标注论文所报 pipeline training overhead 与典型 inference latency overhead；隐藏态幅度、梯度范数和验证损失是其设定下的观察与解释，不宣称残差稀释是深度模型的唯一故障来源或 AttnRes 的唯一机制收益。

## 75.7 Kimi K3：架构采用与总体模型成绩分开归因

Kimi K3 报告和固定 revision/config 让 AttnRes 获得一个较大规模模型实例，但 K3 同时修改序列注意力、MoE、精度与系统实现。本节只将报告与配置支持的结构事实归到 AttnRes，模型总体性能留作系统级证据。

### 75.7.1 K3 报告给出 8 个块、12-layer size 与 embedding 来源

读取 K3 report v2 的 Block AttnRes 设置，并与固定 config 中 `attn_res_block_size=12` 相核对。解释 partial final block 和 embedding 独立来源；这组配置不同于 AttnRes 原论文的 block-count sweep 与 Kimi Linear 48B 实验。

### 75.7.2 K3 整体能力不能单独证明 AttnRes 因果收益

列出 K3 同时涉及的 KDA/MLA、Stable LatentMoE、量化、数据与训练改动，说明全模型对比不隔离 residual 机制；如需归因，必须有同架构、同预算、只改变 AttnRes 的消融。

### 75.7.3 推理 kernel 的具体优化回链实现证据

将 AMD MI355X 文章中的融合与分片经验限定为对应硬件和 runtime；Kimi K3 全模型结构与公开配置详见第 88 章，K3 生产服务路径不由本章重讲。

## 75.8 后续变体与残差路线的选择边界

2026 后续预印本继续拆分 depth routing 的 key/value 角色、压缩 score 维数或引入多流来源。它们说明 AttnRes 是仍在变化的研究方向，不应被回写成原始 Kimi Team 定义，也不能把不同基线下的 headline 数字直接排序。

### 75.8.1 Low-Rank AttnRes 将路由 key 维数与 value 宽度拆开

说明低秩工作保留 full-dimensional value、改用低维 key 计算 routing，并注明论文中的 $r$、$N$、硬件和 throughput 条件；不推广成固定节省比例。

### 75.8.2 RD-AttnRes 将匹配路径与 value 内容路径解耦

概述该预印本共享 Q/K depth route、另学 V route 的改动，限制其配对 seed、模型规模和 token budget 结果，区分作者提出的问题与已证明的普适规律。

### 75.8.3 Multi-Stream AttnRes 为不同来源视图增加多流读取

说明其 causal depthwise convolution 如何形成多尺度 stream，再各自做 depth retrieval 并轻量融合；限制在作者的 18B/28B MoE 配置、任务与论文证据。

### 75.8.4 与 KDA、mHC 对齐问题后再决定采用哪条路线

AttnRes 选择跨深度来源，KDA 递归保存序列历史，mHC 维护并约束多条同深度残差流。比较时应固定模型、数据、预算、并行和任务，衡量质量及存储、通信和时延，不因共享“改残差流”一词混为同种机制。
