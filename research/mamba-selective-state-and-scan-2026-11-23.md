# Mamba 选择性状态更新与 Selective Scan

研究日期：2026-11-23。研究问题：核对 Mamba 如何从固定参数 LTI SSM 转向输入依赖的状态更新，$\Delta$、$\mathbf{B}$、$\mathbf{C}$ 分别影响什么，仿射状态更新如何通过 associative scan 并行组合，以及 Mamba block 怎样连接局部卷积、选择性 SSM 和门控分支。

## 检索范围

- 读者起点：已了解固定参数 LTI SSM 的状态递归、卷积表示和 S4 核生成。本记录只研究原始 Mamba（Mamba-1）的选择机制、扫描代数与 block 数据流；不复述第 29—31 章基础 SSM/S4，不研究 Mamba-2/3、当前 API、服务状态恢复、GPU kernel 调优或跨模型 benchmark。
- 检索日期与平台：2026-11-23，Web 搜索；原始论文 arXiv 页面与 PDF、官方项目资料入口、课程讲义 PDF。
- 搜索词：Mamba Linear-Time Sequence Modeling selective SSM Algorithm 2；Mamba selective copying input dependent Delta B C affine scan；Mamba selective scan associative affine transformation；Mamba block architecture course slides；state-spaces/mamba official selective_scan_interface。
- 筛选方法：先从方法作者原论文核对机制、公式、任务和算法，再查作者/项目来源及课程讲义对关键概念的呈现顺序。技术结论以论文为准；讲义只用于检查教学组织。没有依据单一仓库当前主分支推断固定版本 API 或普遍实现。

## 主要发现

1. Mamba 原论文指出，固定的 LTI 动态对所有时间位置采用同一转移与读写规律，缺少基于 token 内容选择信息的机制。Selective Copying 将待记忆 token 的位置随机变化，并混入无关 token，因而区别于只要求追踪固定时间间隔的 copying 任务。
2. 原论文 Algorithm 2 让 $\Delta$、$\mathbf{B}$、$\mathbf{C}$ 成为输入函数：$\mathbf{B}_t$ 与 $\mathbf{C}_t$ 由当前层输入投影产生，$\Delta_t$ 经正值变换；$\mathbf{A}$ 仍为结构化可学习参数。原论文给出的形状记法中，输入为 $(B,L,D)$，$\mathbf{A}$ 为 $(D,N)$，输入依赖的 $\mathbf{B}_t,\mathbf{C}_t$ 为 $(B,L,N)$，$\Delta_t$ 为 $(B,L,D)$。离散展开后，状态系数具有位置、通道和状态维度。
3. 论文使用零阶保持离散化展示 $\bar{\mathbf A}_t=\exp(\Delta_t\mathbf A)$ 及对应的 $\bar{\mathbf B}_t$。对 $\mathbf A=-1,\mathbf B=1$ 的单状态教学特例，更新化为 $h_t=e^{-\Delta_t}h_{t-1}+(1-e^{-\Delta_t})u_t$；小 $\Delta_t$ 保留旧状态、少写入当前输入，大 $\Delta_t$ 衰减旧状态并更偏向当前输入。该解释受特例和稳定模态约束；一般 Mamba 还使用内容依赖的 $\mathbf{B}_t$、$\mathbf{C}_t$。
4. 输入依赖离散参数使每个时间位置对应一个状态更新，不再共享单一固定 LTI 卷积核。令 $b_t=\bar{\mathbf B}_tu_t$，单步更新为仿射映射 $h\mapsto\bar{\mathbf A}_th+b_t$。按序复合两步更新的运算满足结合律，因此可以通过 parallel prefix scan 得到多个位置的状态；结合律不表示交换律，原始 token 顺序仍然重要。
5. 原论文在 Selective Copying、Induction Heads 等合成任务上评估选择性机制，并提供语言、音频和基因组实验。合成结果只支持指定任务、序列构造和训练设置下的行为结论；不能推出任意历史 token 的显式寻址、任意 ICL 或跨任务的普遍优势。论文的硬件复杂度与性能结果不进入本章普遍结论。
6. Mamba block 把 SSM 路径与门控 MLP 风格的结构组合：输入扩展后分为主支路和门控支路，主支路经短卷积及 SiLU 后进入 selective SSM，SSM 输出由门控支路调制，再投影回模型维度；堆叠时配合残差连接和归一化。该结构支持解释整体数据路径，具体配置须区分变体和实现版本。

## 来源与证据边界

| 来源 | 版本、阅读位置 | 支持内容 | 不支持的结论 / 使用限制 |
| --- | --- | --- | --- |
| Albert Gu、Tri Dao，[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752v2)，arXiv:2312.00752 v2，2024-05-31 | §2 的离散化公式；§3.1 selective copying/induction-head 动机；§3.2 与 Algorithm 2 的输入依赖参数及 shape；§3.3 selective scan；§3.4 Figure 3 的 Mamba block；§3.5.1—3.5.2 的门控关系与参数作用；§4.1 合成任务 | 选择机制定义、$\Delta/B/C$ 的输入依赖、零阶保持写入、仿射 scan 路径、block 结构、作者报告的任务与实验设置 | 不支持脱离模型/数据/任务/硬件的普遍性能、精确检索保证或“等价 Transformer”的结论；本文不复用摘要中的吞吐数字。 |
| Minjia Zhang 课程，[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://minjiazhang.github.io/courses/sp24-resource/Mamba-pre.pdf)，2024 年课程讲义 | “Motivation: Selection as a Means of Compression”“Selective SSM”“Selective Scan: Dynamic Matrices + Parallel Scan”“Mamba Block Design”页 | 作为教学组织参考，检查从任务动机、参数变化、扫描代数到 block 图示的讲解路线 | 讲义不是独立实验或技术事实证据；所有正文事实均回核 Mamba 作者论文。 |
| Mamba 作者/项目的官方代码入口 | 搜索并查看项目 README 与 selective scan interface；未用浮动主分支为正文断言固定版本行为 | 确认官方论文与代码仓库的关联入口 | 未按固定 revision 复核代码路径，因此不用于正文版本/API、shape 细节或性能结论。 |

## 未验证边界

- 未运行 Mamba、selective scan、参考递归、CUDA kernel、梯度或合成任务；正文是 source-backed，不代表本地复现。
- 未复现论文训练或 benchmark，也未检查 GitHub MathJax/Mermaid 页面渲染；不对硬件吞吐、状态恢复或生产服务作结论。
- 研究限定于 Mamba-1 原论文的机制；Mamba-2 的 SSD 算法与后续改进留给第三十五章。
