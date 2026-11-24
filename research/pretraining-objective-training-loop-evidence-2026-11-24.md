# 预训练目标与训练循环证据复核

## 范围与方法

- 研究问题：核对第5册第5章从自回归目标、有效 label 分母、梯度累积与混合精度，到其他预训练目标时所需的原始研究和框架语义。
- 日期：2026-11-24。
- 资料类型：原始论文及 arXiv 指定版本、PyTorch 官方文档、Hugging Face 指定版本/当前文档。
- 本文件仅记录事实来源与适用边界；章节审阅顺序、正文修改和静态验收写入 `progress/`。

## 核验结果

### Cross entropy、梯度累积与混合精度

- [PyTorch `CrossEntropyLoss` 2.15 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.modules.loss.CrossEntropyLoss.html)规定输入为未归一化 logits；整数类别目标可以使用 `ignore_index`。普通未加权类别目标的 mean reduction 按未忽略目标数归一化；若提供类别权重，分母改为未忽略目标对应权重之和。第5章因此把简单 token mean 公式限定在未加权目标，并明确加权 loss 需复核 reduction 分母。
- [PyTorch AMP recipe 2.15](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)说明 `GradScaler.step` 在梯度含 Inf/NaN 时会跳过 optimizer step；如需检查或裁剪梯度，应先 `unscale_`；保存并恢复 AMP 训练还应保存 scaler state。scheduler 是否推进取决于调度器对 step 的定义，正文用“按成功 optimizer update 计数时才推进”作条件限定。
- [Accelerate v1.15.0 gradient accumulation 指南](https://huggingface.co/docs/accelerate/v1.15.0/usage_guides/gradient_accumulation)给出变长 token batch 的示例：在完整累积窗口预取样本、跨进程汇总有效目标数，并按该指南封装的累积步数与 DDP rank 平均缩放 loss。该缩放只对应其 `Accelerator`/DDP 示例，不能再与其他框架的归一化叠加。
- [Transformers 当前 gradient accumulation 文档](https://huggingface.co/docs/transformers/grad_accumulation)说明 causal LM 内部 shift 时，目标数量应对 shifted labels 计数，文档示例使用 `labels[..., 1:]`；若 collator 已提供 `shift_labels`，则按该张量计数。该页为滚动文档，具体 Trainer 语义应按安装版本核对。

### 目标函数的原始研究边界

- [BERT v2](https://arxiv.org/abs/1810.04805)以双向 Transformer 表示、jointly conditioning on left and right context 为模型设定。第5章用它说明 MLM 与 causal LM 的可见上下文不同，不将 BERT 的任务效果推广到开放生成。
- [T5 v4](https://arxiv.org/html/1910.10683)在其 C4/模型与评测设置下固定 15% corruption，对比平均 span 长度 2、3、5、10；论文报告长度 3 相对 i.i.d. corruption 在多数非翻译任务上有小幅且统计显著的改善。该结果不是跨架构、数据和预算的普遍最优 span 长度。
- [UL2 v3](https://arxiv.org/html/2205.05131)提出 7 种 denoiser 的均匀混合。在论文的小模型消融中，167M decoder 与 335M encoder-decoder 在 C4 上预训练约 32B tokens；作者报告 encoder-decoder UL2 对 T5 基线的跨任务归一化总分相对提高 43.6%，对 GPT-like decoder 基线相对提高 76.1%。分数是论文按 9 项任务归一化后的总体比较，不是单项准确率提升，也不构成生产规模普遍排名。
- [FIM v1](https://arxiv.org/html/2207.14255)用 6 个大模型、每个 50B tokens 比较 0%、25%、50%、75%、90%、100% FIM rate；论文报告最高 90% 时左到右测试 loss 未见明显退化，100% 出现退化，同时 infilling pass rate 随 FIM rate 提高，而 FIM perplexity 变化较小。context-level 与 document-level FIM 的评测差异也依赖其代码模型、采样温度和基准。
- [Gloeckle et al. MTP v1](https://arxiv.org/html/2404.19737)以共享 trunk 加多个独立未来预测 head；[DeepSeek-V3 v2 技术报告](https://arxiv.org/html/2412.19437)使用串联预测并保留各预测深度之间的因果链。二者是不同架构实现；额外预测 head 能否提升能力或用于推测解码，仍取决于目标模型和部署验证。
- [Masked Diffusion Language Models v2](https://arxiv.org/html/2406.07524)将离散 masked diffusion 的训练目标化为带权 masked-LM cross entropy，并从变分目标推导；其权重、噪声时间步和采样程序与普通 causal token mean 不同。正文只作目标类别概览，机制留给架构专题章节。

## 对正文的支持边界

- 普通 causal token mean、ignore index、token 累积分母与内部 label shift 是不同层次：公式须与实际 label API、reduction 和有效目标张量一致。
- 梯度累积与一次 full-batch 更新的代数等价要求目标函数与权重一致；随机层、状态统计、分布式缩放和框架调用边界可能改变实际轨迹。
- 本轮仅核对源公式和文档行为；没有运行章内代码、测试、模型、benchmark 或训练，也没有在目标阅读平台渲染公式。
