# 训练架构配置与稳定性证据复核

## 范围与方法

- 研究问题：复核第5册第4章把架构选择映射到训练计算、KV cache、MoE 路由和初始化稳定性时使用的事实与经验主张。
- 日期：2026-11-24。
- 资料类型：原始研究论文及技术报告的 arXiv HTML/摘要版本；按正文实际引用的章节、公式和配置核对，不把单篇实验结果推广为通用配方。
- 本文件只记录新核实的研究事实、来源版本及外推边界；章节审阅顺序、改动和静态验收记录在 `progress/`。

## 核验结果

### 参数、训练计算与注意力缓存

- [Training Compute-Optimal Large Language Models（Chinchilla）](https://arxiv.org/html/2203.15556)的 Appendix F 明确把 embedding、attention、dense block 和 final logits 等项纳入其 FLOPs 统计。正文将 `6PD` 限定为第一阶量级估算，并要求长序列 attention、通信、重计算和其他阶段另计，符合该核算边界；这不是所有架构和实现都精确满足的恒等式。
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/html/2305.13245v3)介绍 MQA/GQA 的 K/V head 组织，并在论文设定中以原预训练计算约 5% 的额外训练预算将现有 checkpoint uptrain。该比例和质量/速度结果对应论文的 T5 模型、数据和实验配置，不能视为任意 decoder-only checkpoint 转换的通用成本保证。
- [LongRoPE](https://arxiv.org/abs/2402.13753)报告通过非均匀位置插值搜索，在免微调情形下扩展 8 倍，并以先扩至 256K、再渐进扩至 2,048K 的方案达到论文所述结果，最后另做 8K 调整以恢复短上下文表现。实验覆盖论文中的 LLaMA 2 与 Mistral 配置；“最大窗口”本身不证明任意模型能可靠使用该长度。

### MoE 负载定义与数值稳定

- [Switch Transformers](https://arxiv.org/html/2101.03961)的负载均衡损失为 `α · N · Σ_i f_i P_i`：其中 `f_i` 是该论文 top-1 路由配置下实际分发给 expert i 的 token 比例，`P_i` 是 batch 内平均路由概率，`α` 为辅助损失系数。不同 top-k、容量裁剪和路由实现对分母及实际接收 token 的口径可能不同；第4章因此明确要求说明统计定义，不能仅凭相同符号假设实现一致。
- [ST-MoE](https://arxiv.org/html/2202.08906)定义 router z-loss 为 batch 中 softmax 前 router logits 的 log-sum-exp 平方均值，并将其与主损失及 load-balance 项分别加权。在该论文特定稳定性实验中，baseline 为 4/6 次稳定，router z-loss 条件为 3/3；论文同时报告该条件的任务质量没有下降。update clipping 条件也为 3/3 稳定，但质量明显变差。样本数、模型、优化器和实验配方均限定在论文设置；这些结果既不证明所有 MoE 都必须使用同一系数，也不表示 z-loss 会自动均衡 expert 负载。

### 结构变化与长序列训练日程的案例边界

- [On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability，arXiv v1](https://arxiv.org/html/2608.30320v1)（2026-08-31）报告一个 28 层、`25B-A3B` MoE 压力测试：在最优学习率 3 倍处保持 AdamW、结构和数据顺序，单独切换 GatedNorm；loss spike 由每 10k 步 32.0 降至 3.2，超过 0.5 梯度裁剪阈值的次数由 256 降至 20。论文正文该句将 AdamW 拼作 “AdmaW”，但同页图注和实验设置使用 AdamW。数字来自单变量压力测试，不等同于完整生产训练 recipe 的收益，也不是其他模型的默认超参数。
- [DeepSeek-V4.1-Flash 技术报告，arXiv v1](https://arxiv.org/html/2609.19969v1)（2026-09-17）描述其 45T 多模态 token 训练：从头在 64K 序列长度训练稀疏 attention，并在处理到 34T token 时扩展至 1M。该节同时列出固定 batch、学习率变化和负载平衡设置；64K→1M 是该模型的具体训练日程，不构成其他任务或模型的通用扩窗方案。

## 对正文的支持边界

- 章节中的参数量、KV cache、attention score、`6PD`、MoE active 参数及初始化关系用于手算和候选筛查。最终资源结论应以配置导出、目标实现 profiler 和目标硬件数据为准。
- 论文中的训练稳定性、质量、长上下文和转换成本结果，都必须保留模型规模、训练阶段、数据、优化器、硬件/实现与指标口径；不能仅凭一个结构名推断跨项目收益。
- 数学表达核对了原始论文或对应模块定义；本轮未执行代码、模型训练、benchmark 或目标平台公式渲染。
