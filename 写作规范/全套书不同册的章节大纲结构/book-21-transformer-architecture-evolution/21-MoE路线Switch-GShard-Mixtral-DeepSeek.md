# 第二十一章：MoE 路线：Switch、GShard、Mixtral 与 DeepSeek-MoE

## 章节契约

- **前置：** 读者已经理解 Transformer 中 position-wise FFN 的作用与标准 block 数据流（本册第五章）；第五册第四章提供 MoE 总参数、active 计算、负载均衡与 capacity 的训练口径。
- **离章结果：** 读者能解释 token 怎样由 router 送往若干 expert、expert 输出如何回到原 token；能区分 GShard 的 top-2 变体、Switch 的 top-1、Mixtral 的每层 8 expert/top-2 以及 DeepSeekMoE 的细粒度切分和 shared expert；能避免把路由策略、专家的语义专业化和模型性能宣传混为一谈。
- **本章职责：** 解释 Transformer MoE 的 FFN 路由机制和代表性结构路线，并说明每项方法主张的模型与实验边界。
- **边界：** 第五册第四章主讲 router 训练、辅助 loss、z-loss、capacity 公式和训练参数账本；第五册第七章与第二十二章承担 expert parallel 和通信细节；第二十二章完整比较 Dense/MoE 的训练、推理、显存、通信、SLO 和成本。本章不重复这些训练 recipe 或系统取舍，也不把 MLA、服务 runtime 或 kernel 归作 MoE 路由。
- **研究依据：** 旧稿含 21 个 H2，其中容量、负载均衡、训练/服务账本、p99、发布门禁、迁移与回滚多处重复第二十二章及第五册第四章；代码审计也不属于本章读者任务。依据章节研究后将主题收束为六个相互承接的方面。事实与教学材料来源见[MoE 路由与专家结构研究记录](../../../research/moe-routing-and-specialist-architecture-2026-11-20.md)。

## 主题分配

| 主题 | 本章所需认识 | 完整主讲位置 |
| --- | --- | --- |
| MoE FFN 定位 | MoE 将一个或多个逐 token FFN 变为多个带不同参数的候选路径；router 按 token hidden state 选路径 | 21.1；FFN 前置见本册第五章 |
| 单 token 路径 | router score、expert 集合、dispatch、expert batch、combine 与原 token 顺序恢复是一条连贯的数据路径 | 21.2；跨设备和 kernel 成本见第二十二章 |
| 路由选择 | top-1、top-2 与 expert-choice 决定一 token 有多少候选 expert；容量溢出策略随论文/实现而异 | 21.3；loss 与 capacity 数学见第五册第四章 |
| 代表性路线 | GShard、Switch、Mixtral 展示不同 top-k 路由选择和模型/系统落点 | 21.4；系统/实验比较见第二十二章 |
| DeepSeekMoE | 细粒度切分与 shared expert 分别改变 routed expert 组合粒度与所有 token 共有的计算路径 | 21.5；DeepSeek-V2 的 MLA 由本册第四、十一章主讲 |
| 专家专业化证据 | 参数路由不自动产生人类可读的领域专家；论文中的消融和评测应限定于其模型、训练与任务 | 21.6；跨 Dense/MoE 公平对比见第二十二章 |

## 正文编号大纲

### 21.1 稀疏 MoE 把 FFN 容量变成条件路径

本节围绕“哪些模型输入使用哪些 FFN 参数”界定本章中的 MoE Transformer。与每个 token 共用同一个 dense FFN 不同，MoE 在一个或多个 position-wise FFN 位置放置多个参数不同但接口相同的 expert，由 router 对每层的 token hidden state 作条件选择。专家是训练得到的参数分工，不是预先命名的学科模块；本节只唤回理解结构所需的 FFN 与 residual 关系，总参数和 active 参数的账本留在第五册第四章及第二十二章。

### 21.2 跟踪一个 token 如何经过 router 与 expert

本节说明一个 token 的 hidden state 怎样从 router scores 形成 selected expert set，经 dispatch 按 expert 聚合，分别通过 expert FFN 后，再按该架构定义的 gate/combination weights 汇回原 token 位置并进入 block 后续路径。正文需要把 token identity、expert batch 和逆置换关系说清；单设备与跨设备部署的传输细节、all-to-all 实现和 kernel 调度不在本节展开。组合权重的归一化口径按模型定义，不视为 MoE 的统一规则。

### 21.3 Top-1、Top-2 与容量策略如何改变路由

本节比较 token-choice top-k 与 expert-choice 两种选择方向，并说明 top-1 与 top-2 分别改变每 token 选中的 expert 数量和合并路径。固定 expert capacity 可能令所选 assignment 遇到溢出；丢弃、改路由或其他处理方式取决于具体论文和实现。负载均衡目标、capacity 公式、overflow 指标和 router 稳定化不在这里重复推导，回链第五册第四章。

### 21.4 GShard、Switch 与 Mixtral 的路由落点

本节按原始论文区分三个设计：GShard 在其 Transformer 翻译系统中使用 position-wise MoE 与一种 top-2 gating 变体，并把自动分片/编译作为系统路线的一部分；Switch 将路由简化为每 token 选一个 expert；Mixtral 报告每层八个 FFN experts、每 token 选两个并组合输出。它们不是对同一训练、硬件和评测条件的横向实验，论文中规模、质量或吞吐数据只支持对应设置。

### 21.5 DeepSeekMoE：细粒度切分与共享路径

本节解释 DeepSeekMoE 的两项结构选择。细粒度切分把较宽的 expert FFN 拆成更多较窄的子 expert，并在论文的设计中相应增加每 token 激活数，以保持规定的总参数与计算口径；shared experts 则让每个 token 固定经过一部分 expert，再由 router 选择其余 routed experts。论文的实验支持其指定模型和消融条件下的观察，正文不把“shared expert 学通用知识”写成架构自动保证。

### 21.6 专家分工的证据边界与章节边界

本节收束专家专业化和公开模型路线的解释边界：expert id、token histogram 或 router 概率本身不足以证明专家各自对应数学、代码或语言领域；要读作者的分析方法、消融对象与任务范围。DeepSeek-V2 同时采用 MoE 和 MLA 时，两项技术分别作用于 FFN 路由与 attention/KV 表示，不能把联合结果归为单一路线。Dense 与 MoE 的成本、训练/服务选择及容量预算由第二十二章完整主讲。
