# 第四十章：Mega、Gated State Space、H3 等早期混合路线

本章围绕早期序列模型如何在平滑历史、状态传播和内容比较之间分配职责展开。读者从 SSM 的卷积/递归视角和 attention 的逐位置交互出发，读完应能辨认 Mega、GSS 与 H3 的实际运算关系，说明它们分别怎样将 EMA、状态空间滤波、门控和注意力组合，以及根据原论文设置解释各自证据。章节聚焦三种设计谱系与有限的混合变体；一般卷积家族由第四十一章展开，Hyena 的统一算子视角见第三十九章，横向架构决策由第四十二章承担。

## 40.1 语言建模缺口如何引出早期混合设计

本节从语言建模中的序列回忆和跨位置比较问题说明早期 SSM 路线的研究动机。内容涉及 induction-head 与 associative-recall 合成任务所分别检验的能力、S4D/GSS 在 H3 论文任务中的表现，以及这些小型合成任务能说明和不能说明的范围；它们为后续结构设计提供问题线索，不构成对所有 SSM 的普遍否定。

## 40.2 Mega：多维 EMA 怎样进入 gated attention

本节说明 Mega 的多维 damped EMA 如何递归整合带有不同衰减尺度的历史，并将 EMA 表示用于构造共享 attention 特征、query/key 与门控。内容覆盖原始输入形成的 value 路径、attention 后 reset/update gate 的作用，以及完整 Mega 与 Mega-chunk 在全长 attention、固定长度分块和跨块压缩上下文上的差异；EMA 是 attention 的时间结构偏置，不是与 attention 输出简单二选一的独立分支。

## 40.3 GSS：以门控缩小状态空间上下文路径

本节围绕 GSS 如何把输入映射到状态路径与门控路径展开，说明 DSS 只处理较窄的上下文表示，随后投影并与另一输入流逐点相乘。内容覆盖该乘法门控降低昂贵状态空间/FFT 计算维度的机制、DSS 的固定状态参数化，以及论文中的 GSS-Hybrid 变体；不得把 GSS 概括成由 token 动态生成转移矩阵的 selective SSM。

## 40.4 H3：shift 与 diagonal SSM 怎样组合 key-value 记忆

本节说明 H3 如何用 shift SSM 保留先前 token 的对齐线索，以输入乘法建立局部 key-value 关联，再由 diagonal SSM 累积长程交互并供 query 读取。内容覆盖 $Q/K/V$ 投影、单头形式与多头外积状态的对应、状态层在 Transformer block 中与 MLP/残差的关系，以及它针对 induction-head 和 associative-recall 能力的设计依据；合成任务不等于自然语言中的一般检索保证。

## 40.5 三条路线的混合位置与 Hyena 谱系

本节对齐 Mega、GSS、H3 的混合发生位置及保留的计算路径：Mega 将 EMA 放入 gated attention 表示路径，GSS 在状态空间上下文流外做乘法门控并可插入 Transformer 层，H3 可与少量完整 attention 层组成 Hybrid H3。内容还说明 Hyena 论文将 GSS/H3 映射为特定参数化下的低阶算子特例，同时划清算子关系与完整模型、训练方案及结果的边界；与第三十九、四十一、四十二章分工衔接。

## 40.6 原论文结果及其适用条件

本节按机制任务、语言模型质量与实现速度区分三类证据。内容整理 Mega 的 LRA 多任务结果、GSS 对 DSS 的特定模型规模/TPUv4 比较、H3 的合成任务与 H3-attention 语言模型结果，并注明任务长度、词表、模型大小、训练数据与预算、硬件和实现条件。结论只落在论文实际测量范围内，不把混合层的效果归因给单一子算子，也不据此推出架构家族的普遍胜负。
