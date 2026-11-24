# 训练总览：Scaling Law 与训练口径证据（2026-11-24）

本记录支持《训练总览》中 scaling-law 历史、token/compute 预算及近年研究的表述。仅记录可复用事实、来源版本和适用边界。

## Scaling law 与训练预算

- Chinchilla 论文报告在 70M 至超过 16B 参数、5B 至 500B 训练 token 的 400 多个模型上拟合预算前沿；其 70B Chinchilla 使用 1.4T token，并与同训练计算预算的 Gopher 对照。论文在其设置下建议模型参数量与训练 token 随计算预算近似同比例增长；不能据此把某一固定 token/parameter 数当作跨语料、目标和训练配方的普适常数。
- 数据受限研究报告：在其固定计算与数据环境下，重复至多 4 个 epoch 的数据，对 loss 的变化相对独立数据较小；重复进一步增加后，追加计算的收益会衰减。此结果是数据受限条件下的研究证据，不表示重复采样普遍无害或能补出新信息。
- Inference-aware scaling 论文在其模型族和成本假设下把预期推理需求加入训练选择，报告对于约 1B 请求的情形，训练更小、更久可能降低生命周期成本；实验覆盖 47 个模型，外推 token/parameter 可达 10,000。请求量、成本函数和模型族均是结论边界。
- 原始来源：[Chinchilla](https://arxiv.org/html/2203.15556)、[数据受限 Scaling](https://arxiv.org/html/2305.16264)、[Inference-aware Scaling，arXiv v3](https://arxiv.org/html/2401.00448v3)。

## 拟合可复现性与近年扩展

- 2025 年 scaling-law 综述表格列出 51 篇研究，其中 45 篇量化 scaling law；其清单还显示，报告训练设置、方程变量、评估、拟合方法、分析代码与分数/检查点的研究数逐层减少。正文的“51 项中 45 项拟合幂律”与该文统计一致；综述指出未充分报告设定会影响拟合结果，不能把不同论文中的指数直接并作同一规律。
- ICLR 2026 的生成式评估 scaling-law 研究比较了以训练 compute、参数与 token、参考解 log-likelihood 为协变量的三类拟合，并指出 pass@k 中的 k 会影响 scaling 参数和可预测性；其稳定性与预测结论限定在研究评估任务、模型范围和判分设置。
- ICLR 2026 的文本质量干预研究使用 23 个 QualityPajama 数据集、超过 2,000 个 100M—8B 参数模型，训练 token 覆盖 100M—200B；论文报告去重、启发式过滤和 LLM 改写会改变拟合系数与指数，计算最优 token/parameter 比例可随干预跨数量级变化。结论不提供通用质量分数或跨语料排序。
- 原始来源：[(Mis)Fitting 综述](https://arxiv.org/html/2502.18969)、[ICLR 2026 生成式评估研究](https://proceedings.iclr.cc/paper_files/paper/2026/hash/55d7fbeddfd9bb6dace146cc4fa02ff0-Abstract-Conference.html)、[ICLR 2026 文本质量干预研究](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6544c0b53cc1ea51d0c7086b536830a8-Abstract-Conference.html)。

## 训练口径与未验证边界

- 一次更新的理想 token 数使用 data-parallel world size：每个 DP rank 贡献不同样本；TP、PP rank 协作处理同一数据副本，不应把所有 GPU 数直接当作数据副本数。
- Dense 的 $6ND$ 是早期数量级估算，仍需按模型架构、有效参数和 token 定义检查，并由目标配置 profiler/集群吞吐修正。
- 本记录没有对书中示例执行训练、评测、benchmark 或集群测量；示例数值均不是目标项目的实测结论。
