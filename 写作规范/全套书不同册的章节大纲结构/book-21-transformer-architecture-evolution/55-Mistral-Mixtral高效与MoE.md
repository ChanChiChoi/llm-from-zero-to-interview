# 第五十五章大纲：Mistral/Mixtral：GQA、Sliding Window Attention 与稀疏 MoE

## 本章定位

本章以 Mistral 7B v0.1 和 Mixtral 8x7B 两个论文版本为案例，比较 GQA、sliding-window attention 与稀疏 MoE 各自改变的信息或资源路径，并核对论文对基准表现的条件化结论。第十七章负责逐版本读取模型配置，第十九章负责局部 attention 的可见性和传播，第十一章负责 KV cache，第 21/22 章负责 MoE 路由和资源成本；本章不复述这些机制细节或部署工作流。

## 55.1 两个固定版本提出的架构问题

明确范围是 2023 年发表论文的 Mistral 7B v0.1，以及 2023 年 12 月发布权重、2024 年 1 月发表论文的 Mixtral 8x7B。概括前者组合 GQA/SWA、后者在 Mistral 主干的 FFN 使用稀疏专家；论文成绩绑定各自 benchmark 和对照模型。

## 55.2 GQA 与滑窗分别改变什么

用简表区分 GQA 的 K/V head 共享与 SWA 的单层历史可见范围。说明 GQA 影响 KV 状态，SWA 影响直接 attention 路径；理论层间传播范围不等于任务有效上下文。机制分别回链第四、十一、十七、十九章。

## 55.3 Mixtral 8x7B：在前馈层使用稀疏专家

说明 Mixtral 论文报告与 Mistral 7B 相同主干、每层 8 个 FFN experts、每 token 每层 top-2 路由；论文按约 $47\mathrm{B}$ 总参数和每 token 约 $13\mathrm{B}$ 激活参数概括规模，并报告 $32\mathrm{k}$ 训练上下文。强调总权重、激活计算、路由和通信是不同账本，详细机制回链第二十一、二十二章。

## 55.4 论文结果如何与结构证据对应

以表格对照 Mistral 7B 和 Mixtral 8x7B 的论文实验主张、比较对象与外推边界。说明跨论文、不同基座/提示/解码条件的成绩不是同一排名，也不能直接归因给单个模块。

## 55.5 把“效率”落回具体成本路径

总结 GQA、SWA 和 MoE 的不同作用及各自不能证明的结论；提示锁定 checkpoint、license 和模型卡版本，随后按任务验证。Mistral 后续模型的具体案例交给本册第七十九章。
