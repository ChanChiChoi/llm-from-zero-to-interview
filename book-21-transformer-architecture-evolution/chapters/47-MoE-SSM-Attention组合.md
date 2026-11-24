# 第四十七章：MoE、SSM 与 Attention 的组合设计：架构位置与证据

## 47.1 Attention/SSM 与 MoE 改变的是不同结构位置

一个 Transformer block 通常由序列混合部分和逐 token 的 FFN 组成。序列混合部分决定当前位置怎样使用序列中的其他信息：Attention 根据当前 query 读取可见的 key/value，SSM 则沿序列更新并读取递归状态。FFN 对每个位置的表示作非线性变换。

MoE 通常改造的是 FFN 这条参数路径：router 根据 token 表示选择一个或多个 expert，再将选中 expert 的输出合并。它改变“当前位置经过哪组参数”，本身不规定历史信息如何跨 token 传播。因此，Attention/SSM 和 MoE 不是三个同类的序列算子；前两者主要决定序列信息路径，MoE 通常决定条件参数路径。

MoE 也可以进入 SSM 内部投影。此时它不再只是替换 FFN，而会改变状态更新模块内部由哪些参数处理 token。判断一个组合架构，至少要问两件事：序列由什么模块混合，专家化落在 FFN 还是序列模块内部。只列出“MoE、SSM、Attention”三个名称，还不能看出实际数据流。

## 47.2 Jamba：在 Attention–Mamba 主干的 MLP 位置加入 MoE

Jamba 展示了一个清楚的三者组合：层间交错 Attention 和 Mamba；每层的序列混合模块之后接 MLP；部分 MLP 被 MoE 替换。token 先经过所在层的 Attention 或 Mamba，再由普通 MLP 或被选中的 experts 变换。MoE 的插入不改变该层究竟采用 Attention 还是 Mamba。[AI21 的架构说明图](https://www.ai21.com/blog/announcing-jamba/)可以帮助辨认这两个层级；精确配置和实验结论以[Jamba 正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/a9ed43fa31dc8b4a7d7a673d713dcb5f-Paper-Conference.pdf)为准。

论文把几个选择分开报告：一个重复 block 有八层，其中 Attention 与 Mamba 的比例为 1:7；每隔两层，普通 MLP 替换为 MoE；每个 MoE 层有 16 个 experts，每个 token 选择 top-2。这里的 1:7 不是其他模型必须照抄的比例，而是该论文在自身比较中选择的配置。

作者先用 1.3B 参数模型训练 250B tokens，对比没有 MoE 的不同 Attention/Mamba 比例。1:3 与 1:7 在报告的指标上差别不大，1:7 计算更省，因此作者在后续较大模型中采用它。接着，作者用 7B、50B-token 的实验比较纯 Attention、纯 Mamba 和不带 MoE 的 hybrid；在这些设置下 hybrid 结果较好。另一个消融再在 hybrid 上加入 MoE，作者报告多个基准和 log-prob 指标改善。这样分开做的消融能说明论文设置下各个改动提供了什么信息；它不能证明每种规模、训练配方或任务中都会出现同样增益。

## 47.3 BlackMamba：Mamba 序列路径与 routed MLP 的组合

BlackMamba 去掉显式 Attention 路径：序列经过 Mamba 状态更新后，再进入 routed MLP；router 据当前位置的表示选择 FFN experts。与 Jamba 相比，这个设计把问题收窄为“SSM 主干能否和条件 FFN 容量共同训练”，不再同时检验显式 Attention 对模型行为的贡献。

论文报告了 340M/1.5B 和 630M/2.8B 的前向激活/总参数配置，并在自建语料上训练 300B tokens。作者对这些模型报告了语言建模、下游任务以及训练 FLOPs 和生成时间的比较。应把这些结果留在其模型规模、语料、实现和计时方式内解读；它们不是与 Jamba 的同条件对照，也不能推出所有 SSM-MoE 模型都比 Transformer 或 hybrid 更快。[BlackMamba 原论文](https://arxiv.org/abs/2402.01771)给出的首先是一个不同组合位置的研究实例。

## 47.4 Routing Mamba：把专家化移入 SSM 投影，并扩展到混合主干

Mamba block 内有多个功能不同的投影。把每个投影都独立改成 MoE，不保证得到一组彼此协调的 expert 路径。Routing Mamba 的做法是在选定投影上建立专家，并共享相关投影之间的路由决定；对原始 Mamba 的一组实验还让部分较小的投影与卷积参数由 experts 共享。这样做改变的是 SSM 内部状态计算的参数路径，不是把整个 Mamba block 换成普通 FFN。

这篇 NeurIPS 2025 论文先比较独立专家化若干 Mamba 投影的方案，再评估其共享路由方法。作者报告，简单地逐个投影套用 MoE 在其设置下经常没有改善、部分配置还会退化；RoM 的选择性专家化和共享路由则在其 Mamba 及 hybrid 实验中得到较好的验证困惑度结果。这是对具体 Mamba 参数化和路由设计的发现，不表示所有 SSM 内部 MoE 都必须采用同一种共享方式。

为检验它能否进入混合主干，作者以包含 Mamba 与滑动窗口 Attention 的 Samba 为对象。在一组 SlimPajama 训练对照中，数据为 20B tokens、硬件为 8 张 A100；RoM 配置使用 421M 前向激活参数、1.0B 总参数，和 511M 参数的 Samba expand=4 比较。论文表中两者在 4K 验证长度的困惑度分别约为 9.77 和 9.78，一次 4K forward 的 FLOPs 分别为 4.74T 和 6.13T。作者据此报告约 23% 的 FLOPs 差值。这是特定结构和一次 forward 口径下的比较，不能改写成端到端训练时长或线上延迟节省。[Routing Mamba 的 NeurIPS 2025 正式论文](https://papers.nips.cc/paper_files/paper/2025/file/79bc44e53e8a981ac1510d4c24ba3d71-Paper-Conference.pdf)还单独比较 RoM 与 FFN-MoE 的组合；在相近的激活参数设置下，两种方案的平均任务准确率接近，并未显示组合必然提高平均分。

## 47.5 比较专家化位置与序列混合路径

| 组合 | 序列混合路径 | MoE 的位置 | 这项设计主要检验什么 |
| --- | --- | --- | --- |
| Jamba | Attention 与 Mamba 层间交错 | 部分层的 MLP | 在混合序列主干上加入条件 FFN 容量后，论文指标怎样变化 |
| BlackMamba | Mamba | Mamba 后的 routed MLP | 不使用显式 Attention 时，SSM 与 FFN-MoE 能否组成有竞争力的模型 |
| Routing Mamba on Samba | Mamba 与滑动窗口 Attention | 选定 Mamba 投影；另有与 FFN-MoE 联合的实验 | 专家化 SSM 内部路径、共享路由及其与混合主干的关系 |

这三种位置不能被压成一个“统一 MoE 层”。在 Jamba 和 BlackMamba 中，MoE 替换 FFN；在 Routing Mamba 中，expert 参与生成 SSM 状态路径所需的投影。一个模型是否还有 Attention、Attention 采用什么范围、router 读到哪一层表示，都由具体架构定义。即使两篇论文使用同一个名称，也不能据此假设它们有相同的路由输入、expert 功能或通信路径。

组合设计带来的研究问题也随插入位置而变：FFN-MoE 主要改变每个 token 的非线性参数变换；SSM 投影专家化会进入状态更新模块内部；若多个子模块共享路由，还要检验共享决定是否适合这些子模块的计算关系。论文里的结构图、层配置和消融表应一起阅读，不能只凭模型名或总参数量推断路径。

## 47.6 组合收益需要能隔离主效应与交互的消融

若要判断三类结构选择各自贡献了什么，实验应包含能拆开它们的控制模型：固定序列主干后比较 dense FFN 与 FFN-MoE；固定参数化后比较 Attention、SSM 和混合序列路径；如果研究 MoE 进入 SSM 投影，再把它与只改 FFN 的版本比较。若还要声称两处专家化彼此协同，就需要直接比较联合方案与各自单独加入的方案。只有完整三组件模型而没有这些对照，无法分辨收益来自哪个模块，也无法证明存在交互增益。

对照还要交代数据、训练 token、模型结构、前向激活参数与总参数、计算口径、序列长度、任务和实现版本。参数量、计算量、训练速度、生成速度和上下文行为回答的是不同问题，应分别报告。Jamba 的拆分消融、BlackMamba 的 Mamba-MoE 实验和 Routing Mamba 在 Samba 上的投影/FFN 对照提供了不同强度、不同范围的证据；它们不构成跨论文总排名。

现有证据留下了几个仍需实验回答的问题：专家更适合放在 FFN 还是 SSM 投影；不同位置是否应该共享路由；最优插入密度是否随主干、任务和规模改变；在更大训练预算和不同检索/长上下文任务上能否复现收益。本章只归纳这些具体组合的证据，不预测一个所有模型都会采用的配方。更广的模型演进判断由第五十章承担。
