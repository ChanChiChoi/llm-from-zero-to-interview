# 第二十五章：Performer、Linear Transformer 与注意力近似

## 章节契约

- **前置：** 理解 scaled dot-product attention、query/key/value、softmax 归一化、因果 attention，以及有限特征 kernel 下的结合律重排和前缀状态；后者由第二十四章建立。
- **离章结果：** 能推导正值随机特征对 softmax kernel 的无偏估计；解释 FAVOR+ 中正值特征和正交随机方向分别承担的作用；区分 kernel 估计无偏、归一化 attention 输出及多层 Transformer 结果；按定理假设和实验设置准确描述近似保证；估算固定特征数时的主要成本，并与精确 attention 路线区分。
- **本章职责：** 以 Performer/FAVOR+ 为对象，说明怎样用有限随机特征近似 softmax kernel，并分析这种近似的误差、理论条件、实证范围和计算代价。
- **边界：** 第二十四章负责线性 attention 的特征分解、基本因果状态和状态访问边界；本章只在说明 FAVOR+ 如何代入该结构时简要回顾，不重复其递推推导。第二十章负责 FlashAttention 的精确 attention 数学与 IO 优化；第二十六至二十八章负责稀疏 attention 与高效路线比较。本章不提供随机特征实现代码、部署回退方案、benchmark 流程或产品选择清单。
- **研究依据：** 旧稿 25 个 H2 反复讲解 Linear Transformer 状态、近似误差和特征数，并混有泛化随机 Fourier 代码、部署/回退决策、运行 trace、benchmark 清单和实验伪代码。修订结构以“精确重排与 softmax 近似的区别—正值特征估计—FAVOR+ 设计—归一化输出误差—保证与实验边界—成本口径”形成完整路线；证据见[Performer/FAVOR+ 随机特征研究记录](../../../research/performer-favor-plus-random-features-2026-11-20.md)。

## 正文编号大纲

### 25.1 从精确重排到随机近似

本节区分第二十四章的代数重排与 Performer 的 softmax kernel 估计。内容包括标准缩放点积 softmax 所定义的相似度、有限特征内积对该相似度的近似，以及 Performer 虽沿用特征分解计算、却引入采样误差的原因；本章将据此聚焦误差与保证的适用范围。

### 25.2 正值随机特征如何估计 softmax kernel

本节围绕高斯随机向量的矩母函数说明正值随机特征的来源，界定缩放后的 query/key、单个随机特征和有限样本核估计之间的关系。内容覆盖估计对随机方向取期望时为何恢复指数内积，以及一组有限采样只给出该 kernel 的随机估计；一般面向平移不变 kernel 的随机 Fourier 特征不自动成为 softmax 的正值特征。

### 25.3 FAVOR+ 中正值与正交各自解决什么

本节说明 FAVOR+ 如何将正值随机特征（PRF）与正交随机特征（ORF）组合。知识范围包括正值分量对估计分数和归一化分母符号的作用，正交采样在各向同性分布与论文条件下对估计方差的影响，以及正交维数限制和分块扩展；不把正值称为数值稳定性的充分保证，也不把有限维空间外的方向说成彼此全正交。

### 25.4 无偏 kernel 估计不等于无偏 attention 输出

本节从 attention 的归一化加权 value 和出发，界定随机估计同时进入分子与分母后造成的比值误差。内容包括分子、分母误差的组合关系，分母大小对输出偏差和扰动的影响，以及局部估计误差经过多头、多层计算传播的可能性；kernel 单项均方误差不能单独代表最终任务误差。

### 25.5 理论保证和实验结果各自覆盖什么

本节对齐 FAVOR+ 论文中估计器特定的方差与收敛结论及有限实验。内容区分正值 ORF 的 MSE 结果、带 query/key 范数上界的三角特征 ORF 注意力矩阵一致收敛结论，以及不同对象之间不可互换的保证；实验结论保留序列长度、维度、样本数量、模型、数据集和 feature redrawing 等条件，不从中推出所有模型或任务的普遍质量保证。

### 25.6 固定特征数下的计算成本与精确 attention 边界

本节解释 feature 数量、query/key 维度和 value 维度如何共同决定主要计算量及因果流中的状态大小，并说明线性随序列长度增长依赖 feature 数量固定。内容还区分随机特征带来的 softmax 近似与 FlashAttention 对 dense attention 数学形式的 IO 优化；渐近复杂度、论文设备实验与通用 wall-clock 性能属于不同证据层次。
