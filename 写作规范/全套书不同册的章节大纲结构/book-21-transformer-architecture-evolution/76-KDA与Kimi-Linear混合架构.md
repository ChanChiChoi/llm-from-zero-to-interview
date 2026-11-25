# 第 76 章 Kimi Delta Attention：从可更新记忆到线性注意力

本章承接第 61 章的 KDA 递归机制，转而回答模型级问题：KDA 如何与全局 MLA、NoPE 和 MoE 组成 Kimi Linear；三层递归状态与一层全局读取的实验证据是什么；论文中的 1.4T 受控比较、5.7T 发布 checkpoint、长上下文质量与推理速度分别支持什么结论。第 63 章负责 Gated MLA 的 latent cache 机制，第 64 章负责 NoPE 的一般性质，第 65 章负责混合注意力的横向比较。正文不重复推导 KDA 状态方程，也不把 Kimi Linear 的配置推广为通用架构配比。

## 76.1 先把 Kimi Linear 拆成三条结构轴

本节区分序列混合、通道混合和位置信息三类设计。KDA/MLA 决定 token 历史怎样参与当前计算；MoE 决定通道混合层的稀疏专家容量；NoPE 是全局 MLA 层的位置处理选择。它们同时出现于一个模型，但改变的对象不同。

### 76.1.1 48B 总参数和 3B 激活参数描述不同的模型规模

绑定到 Kimi Linear v2 的模型设置说明总参数、每 token 激活参数以及 MoE 结构；指出报告脚注对 activated parameters 的统计边界。参数规模不能单独证明架构质量或服务显存。

### 76.1.2 每个 block 先做 token mixing，再做 MoE channel mixing

按论文 Figure 3 说明一个 block 的基本顺序：KDA 或 MLA token-mixing layer，随后是 MoE channel-mixing layer。不要把 KDA/MLA 的 3:1 比例误读成 MoE 的专家数或整模型 Transformer block 的其他计数。

### 76.1.3 KDA、MLA 与 NoPE 改变的对象不同

对比 KDA 每层的固定形状递归 state 与 MLA 按 token 保存 latent cache；指出 global MLA 保留按位置读取的路径，KDA 则将历史写入状态。细节分别回链第 61、63、69 章。

## 76.2 一条远程事实怎样经过四层周期

本节用“前缀较早处定义一个版本号，末尾问题询问其精确值”走查 Kimi Linear 的层间信息流。例子说明可用路径，不预设模型一定成功。

### 76.2.1 三个 KDA 层分别更新自己的递归状态

每个 KDA 层在读入 token 时更新本层状态；不同层不是共享一个全模型摘要。用状态写入解释有限状态怎样持续传递历史特征，并指出原 token 不再有独立地址。

### 76.2.2 全局 MLA 层提供按位置读取前缀的入口

说明周期中的 MLA 层可以让当前位置 query 与 causal mask 允许的前缀 latent 交互。它补上显式全局读取路径，但访问到历史位置不等于准确选中、保留或输出版本号。

### 76.2.3 后续状态继续处理全局读取产生的表示

描述 MLA 输出继续经过对应通道混合层，再进入后续 token-mixing 层；后续递归不会把 MLA cache 变成无损索引。强调路径存在、训练形成能力和任务成绩是不同层级的证据。

## 76.3 为什么选择每三层 KDA 接一层全局 MLA

本节从纯线性状态的精确回读压力引出 hybrid 设计，再阅读论文的层比消融。结果限定在 Kimi Linear 的报告配置，不能写成跨模型的最优常数。

### 76.3.1 有限状态路径会遇到精确复制和检索压力

承接第 61 章的状态容量边界，说明需要逐字复制、冲突实体辨别和远程键值召回等任务会形成压力；引用论文的合成任务只作机制动机，不重讲其实验细节。

### 76.3.2 比例消融在被测设置中选择了 3:1

列出论文测试的 0:1、1:1、3:1、7:1、15:1 KDA/MLA 比例及训练/验证困惑度，指出 3:1 的验证值最低且与 1:1 接近。明确作者将该比例选作质量—吞吐折中；表格不能证明所有规模、语料或任务都应采用 3:1。

### 76.3.3 3:1 是模型选择，不是 KDA 的定义

区分 KDA 递归定义、具体模型层表和模型对照结果。跨模型复用该比例前，需要重新检查模型宽度、层数、上下文任务、训练预算、cache 表示和 kernel。

## 76.4 1.4T token 对照支持哪些质量判断

本节读取报告主实验中最接近结构因果对照的一组结果，列清共同训练设置和模型版本，再以少量有代表性的指标呈现总体趋势与例外。

### 76.4.1 三种 token mixer 使用相同模型规模和训练设置

说明 MLA、GDN-H、Kimi Linear 的相同架构规模、参数量与训练 setup；预训练使用 1.4T token、4096-token context、共同语料及优化配方。指出这是报告内部控制较好的结构比较，但复合改变了 KDA、hybrid ratio 和 NoPE 等设计，不能拆成单因素因果结论。

### 76.4.2 MMLU-Pro 与 RULER 展示两个不同长度切片

报告 MMLU-Pro@4k 与 RULER@128k 三种模型的分数，并给出长上下文表格中的整体平均和任务例外。解释“整体平均领先”不等于每个任务、每个能力或任何长度都领先。

### 76.4.3 SFT 分数需与预训练阶段分开阅读

选取 SFT 表中的少数结果和反例，区分 base pretraining、instruction tuning 与后续 RL 结果。禁止将不同训练阶段分数放在同一列作直接排名。

## 76.5 5.7T 发布 checkpoint 回答的是另一类问题

本节将研究型 1.4T 对照与最终发布 checkpoint 分开，避免把模型成熟化、额外训练和层结构的贡献混为一谈。

### 76.5.1 1M context 是 checkpoint 支持和特定评测，不是所有任务的保证

依据论文 v2 说明最终发布模型的训练 token 数和支持窗口；公开报告能确认窗口与指定 benchmark 结果，不能证明任意 1M 输入都能被完整利用。

### 76.5.2 与 Moonlight 的 5.7T 比较不是同总参数消融

说明 Appendix D 的 Kimi Linear 与 Moonlight 都训练 5.7T token、均为 3B activated parameters，但总参数分别为 48B 和 16B。两者可作发布模型比较，不能当作只改 attention 的同规模消融。

### 76.5.3 1M RULER 指标不能替代匹配的全注意力基线

报告 Instruct checkpoint 的 RULER@1M 分数，并说明 Moonlight-Instruct 在超过 8k 的任务没有评测。把长窗口成绩绑定其评测协议和 checkpoint，不外推到所有长文本检索或真实 agent 任务。

## 76.6 分开核算 cache、prefill 与 decode 效率

本节把架构导致的历史表示变化和论文报告的效率曲线分别计量，说明 token KV 比例、固定 state、吞吐和 TPOT 不是同一个指标。

### 76.6.1 3:1 层表解释 KV cache 主张，但不等于总显存比例

从四层周期中只有一层全局 MLA 需要逐 token latent cache 推出理想比例直觉；再说明 KDA 的递归 state、短卷积状态、MLA latent 宽度、元数据和 workspace 不包含在简单比例里，不能把“最多减少 75% KV cache”写成整模型总显存减少 75%。

### 76.6.2 Figure 7 区分 prefill latency 和单 token decode

标明被比较模型的层数/head 条件与 batch size=1，报告 512k/1M prefill latency 的相对结果。区分 prefill latency、单 token decode latency 与并发吞吐。

### 76.6.3 Figure 1 的 6.3× TPOT 依赖另一种 batch 口径

报告作者所示 1M 场景的 6.3× TPOT 数字与相应毫秒数，并注明论文将收益联系到较低内存占用允许更大 batch；该图的测量口径与 Figure 7 的 batch-one 曲线不同。硬件/运行时与调度未充分披露时，不将其写成通用线上加速承诺。

## 76.7 用证据而不是层比回答“混合架构是否更好”

本节收束模型选择与因果归属：同一模型的组件消融、不同训练阶段的模型分数、cache 账本和 runtime latency 分别回答不同问题。

### 76.7.1 设计评测时按信息访问需求和上下文长度分桶

要求分别覆盖精确复制、远程实体/数值召回、干扰下检索、长文本理解与 generation-heavy decode；同时记录窗口、目标距离、任务成功标准、模型 checkpoint 和输入/输出长度。

### 76.7.2 只有匹配模型和执行条件才能比较质量与速度

列明参数与激活口径、训练 token、数据、post-training 阶段、上下文、batch、硬件、kernel、cache dtype 与 baseline。多个组件共同变化时将结论写成整套设计的联合结果。

### 76.7.3 Kimi Linear 是一个具体实现案例，不是架构配比处方

回链第 61 章 KDA 机制、第 63 章 Gated MLA、第 64 章 NoPE、第 65 章混合架构比较、第 69 章 cache 表示和第 88 章 Kimi K3 案例；以“递归状态与全局逐位置读取互补，但任务能力须实测”结束。
