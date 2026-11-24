# 第 61 章 KDA：把历史表示成可门控的递归状态

本章专讲 Kimi Linear 报告中的 Kimi Delta Attention（KDA），围绕一个问题展开：怎样在固定形状的 key-value 状态中保留、衰减并修正历史关联。结构主线依次说明递归状态相对显式 token 历史的寻址变化、delta rule 的误差写入、KDA 的逐 key 通道衰减与标量写入强度、状态读写过程、chunkwise 并行训练、信息容量和资源边界，以及 Kimi Linear/Kimi K3 的版本差异。第 24 章负责线性注意力基础，第 62 章主讲通用 Gated DeltaNet 与 delta-rule 训练实现，第 63 章主讲 Gated MLA latent cache，第 65 章比较混合架构，第 88 章负责 Kimi K3 的完整模型案例；本章不扩写 serving 状态生命周期或通用推理系统验收。

## 61.1 显式历史与递归状态

本节将 KDA 放在线性状态路线的历史表示问题中理解：标准 causal attention 保留 token 级 key/value 以便按位置读取，递归矩阵把多条 key-value 关联合并到固定形状状态。说明状态矩阵的 key 轴和值轴，以及 query 怎样读取它；同时明确有限状态不等同于无损压缩，也不提供每个旧 token 的显式地址。它承接第 24 章的线性注意力基础，不重讲核特征方法。

## 61.2 Delta Rule：让状态按当前 key 纠正预测

本节聚焦 delta memory 怎样避免只对旧状态不断追加：从状态对当前 key 的 value 预测、观察 value、预测误差到沿当前 key 方向的秩一写入，说明各对象的关系及写入幅度。此机制是 KDA 的重要前置；正文可用矩阵状态把误差写入具体化，但通用 DeltaNet 的完整推导、门控变体和训练数值问题交由第 62 章。

## 61.3 KDA 的逐通道衰减和写入公式

本节给出 Kimi Linear 报告中 KDA 的精确状态转换，将逐 key 通道对角保留系数作用于旧状态，再从衰减状态计算预测误差，最后由每 head/token 标量写入强度控制 rank-one correction。中心是 $\mathrm{Diag}(\alpha_t)$ 与 $\beta_t$ 的轴和职责：前者逐 key 特征通道缩放旧状态行，后者调节当前误差写入；二者不应与读出端 output gate 混同。用其展开式确认这是一种对角加低秩更新，并与 Gated DeltaNet 的标量衰减设定作有限、版本绑定的对照。

## 61.4 一次完整读写的状态轨迹

本节让读者沿同一个小型矩阵状态追踪：通道衰减如何先改变旧关联，当前 key 如何从新状态形成预测，预测误差如何沿 key 方向写入，以及 query 如何读出更新后的 value 向量。中心是把矩阵行/列、衰减、预测、误差和输出一一对应，并据此看出 $\alpha_t$ 与 $\beta_t$ 作用对象不同。例中数值只作机制演示，不能写成模型参数或实验结果。

## 61.5 递归更新如何用于 chunkwise 并行训练

本节区分 token-by-token 的递归语义与训练时的并行排布。Kimi Linear 报告的 chunkwise WY/UT 形式将块内多步关系组织成批量矩阵运算，并在块间传递边界状态；正文说明其解决串行训练路径的目标和不可丢失的跨块状态关系，不重做完整矩阵推导。若引用 FlashKDA 的工程选择或速度观察，必须绑定文档、kernel 原型与设置，不能把局部实现结果写成算法保证。

## 61.6 固定状态的资源收益和信息代价

本节把状态表示连到成本和能力边界：KDA 核心矩阵状态的存储主要由层数、head 数和 key/value 状态维度决定，而 token 级 K/V payload 随上下文长度增长；实际混合模型仍可能有 MLA/full-attention cache、卷积状态、workspace 和 metadata。由有限状态的合并与不可寻址性说明精确远程引用、冲突字段和出处恢复为何需要单独评估。Kimi Linear 的 Palindrome、MQAR、Stack 合成任务设置只能支持相应小模型、长度、优化过程下的观察，不外推为真实长文档能力或普遍速度结论。

## 61.7 Kimi Linear 与 Kimi K3 的版本边界

本节将算法结构、版本参数化和模型层间组合分开：Kimi Linear v2 以逐 key 通道衰减描述 KDA，报告三层 KDA 配一层全局 MLA；Kimi K3 v2 报告对 log-decay 下界和 output gate 参数化的调整，并采用三层 KDA、一层 Gated MLA、末端附加 Gated MLA 的版本化布局。说明这些差异来自特定技术报告，不是 KDA 通用要求；Gated MLA 机制与 K3 完整架构分别交由第 63、88 章。

