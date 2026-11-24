# 第三十八章：RetNet：Retention 机制与并行/递推统一

本章围绕 RetNet 如何让同一 retention 映射既能按整段序列并行计算，也能按位置递归更新状态，并进一步支持块内并行、块间递归。读者从已理解的 causal attention、线性代数记号和第三十七章的递归状态出发，读完应能推导 retention 的衰减加权和与状态更新之间的关系，辨认 multi-scale retention block 的构成，并判断固定衰减、有限矩阵状态及论文效率结果各自支持什么结论。

## 38.1 RetNet 要处理的训练与生成约束

本节说明整段训练的并行计算与自回归生成的逐步状态之间为何会形成架构取舍，并交代 RetNet 用带衰减的内容交互和可复用状态回应这一问题。重点界定本章解释的是 2023 年 RetNet 论文中的 retention 机制，不把它写成所有线性注意力或后续递归架构的通用定义；复杂度和性能判断须带上序列长度、状态宽度与实验条件。

## 38.2 因果衰减交互如何写成递归状态

本节说明 retention 如何用 causal decay matrix 对 query-key 交互按相对距离加权，再与 value 相乘；通过展开递归式，连接每个历史 key-value 外积的衰减贡献、固定形状的状态矩阵和当前位置的 query 读出。内容包括 query/key/value 的形状、因果下三角边界、逐位置衰减、位置相位变换的作用，以及一个状态如何随当前输入更新并产生当前输出。

## 38.3 Multi-Scale Retention block 的组成与数值尺度

本节说明不同 head 的固定衰减率如何形成多个时间尺度，以及 retention 输出如何经逐头 GroupNorm、Swish 门控与输出投影组成 multi-scale retention。内容覆盖论文中的相对位置相位、头间衰减安排、retention score 的缩放与稳定化、残差和 FFN 结构；明确 score normalization 与 GroupNorm 不等于 softmax 概率归一化，也不把简化的分子/分母递推误称为 RetNet 的完整公式。

## 38.4 Parallel、recurrent 与 chunkwise recurrent 三种执行形态

本节说明三种形态为何来自同一衰减状态映射：parallel 在整段位置上计算因果交互，recurrent 逐 token 读写状态，chunkwise recurrent 在 chunk 内并行并在 chunk 间传递边界状态。内容包括块内历史贡献、块间输入状态衰减、块结束状态更新、固定 chunk 长度下随序列长度增长的成本，以及在相同初始状态、位置处理、mask、衰减和数值规范下比较三种计算的条件；不把位置依赖误写为相互独立，也不把理论一致性等同于任意实现的逐位相等。

## 38.5 Retention 状态的表达边界与相邻路线

本节以可访问历史的方式区分 retention、softmax attention、一般线性注意力和 RWKV：RetNet 的 query 仍读取 key-value 聚合状态，但历史项受固定衰减并累积到有限矩阵；不同机制在权重归一化、状态更新、逐位置适应性和历史寻址方式上并不相同。内容说明多尺度衰减带来的归纳偏置及其容量边界，并将全面的跨架构比较留给第四十二章，服务状态生命周期留给推理与系统书册。

## 38.6 原论文的结果与效率主张边界

本节概述原论文用何种模型规模、训练语料、token 预算、设备、序列长度和 chunk 设置报告语言建模、训练与推理结果，并区分机制推导、论文实验观察和普遍性能结论。读者应能说明固定状态相对历史长度的存储含义，识别不同 baseline、kernel 与硬件对吞吐比较的影响，不从 2023 年论文的设定推出 RetNet 对所有现代 Transformer、任务或设备都更快或更强。
