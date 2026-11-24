# 预训练目标与损失归一化证据（2026-11-24）

本文记录可复核的预训练目标、token 分母与混合精度接口事实；实验结论只适用于来源中明确给出的模型、数据、实现和评测条件。

## 框架接口

| 知识点 | 来源与证据 | 适用边界 |
| --- | --- | --- |
| Transformers Trainer 的梯度累积自定义损失可用整个累积窗口的 `num_items_in_batch` 作分母；因果 LM 的目标数按移位后的标签统计，默认 shift 留下的序列首位不计入。自定义 `compute_loss` 若已自行按目标数缩放，应设置 `loss_is_scaled_for_ga`，避免 Trainer 再按累积步数缩放。 | [Transformers 梯度累积文档](https://huggingface.co/docs/transformers/grad_accumulation)，在线文档，复核于 2026-11-24。 | `main` 文档会变动；自定义 loss、padding-free collator、label shift 和版本差异都应按安装版本确认。 | 
| Accelerate 的变长样本例子预先统计本地累积窗口的非 ignore label 数，再跨进程汇总；为配合 Accelerate 对 accumulation 的缩放和 DDP 的 rank 平均，反传系数包含累积步数、进程数与全局目标数。 | [Accelerate 梯度累积指南](https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation)，在线文档，复核于 2026-11-24。 | 该系数属于指南中的特定调用路径；手写 DDP、其他 helper、不同 reduction 或内部 label shift 不能直接叠加照搬。 | 
| 类别索引目标下，PyTorch `CrossEntropyLoss(reduction="mean")` 会忽略 `ignore_index`；若提供类别权重，分母按未忽略目标的类别权重之和计算。`reduction="sum"` 则返回未归一化损失和。 | [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.CrossEntropyLoss.html)，在线文档，复核于 2026-11-24。 | 普通 token mean 公式假设无类别权重；加权目标需依其 reduction 定义重写分母。 | 
| FP16 loss scaling 的梯度在检查或裁剪前须先 `unscale_`；GradScaler 在非有限梯度时可跳过 optimizer 更新，checkpoint 如需恢复缩放状态应保存 scaler state。 | [PyTorch AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)，在线文档，复核于 2026-11-24。 | 设备、dtype、算子和 scaler 配置影响具体行为；BF16 通常不依赖相同动态 scaling 流程。 |

## 目标函数与论文实验

| 知识点 | 来源与证据 | 适用边界 |
| --- | --- | --- |
| T5 比较固定 15% corruption 下的平均 span 长度 2、3、5、10；平均长度 3 在多数非翻译 benchmark 上相对 i.i.d. corruption 有小幅且显著的优势，同时 span corruption 使目标序列较短。 | [T5 §3.3.4](https://arxiv.org/html/1910.10683)，原始论文，复核于 2026-11-24。 | 受其 C4、模型、训练配方和评测设置限制，不表示 span corruption 普遍胜过 causal LM。 |
| UL2 的 Mixture-of-Denoisers 混合 7 个 denoiser 配置：R 的短 span/15% corruption、S 的 prefix-to-suffix 形式、X 的较长 span或高 corruption；论文将任务近似均匀混合。32B-token、167M decoder 与 335M encoder-decoder 消融报告 UL2 相对 T5 基线平均高 43.6%、相对 GPT-like 基线高 76.1%，总分对异质任务等权归一化。 | [UL2 §3.1、§4.2–4.3](https://arxiv.org/html/2205.05131)，原始论文，复核于 2026-11-24。 | 43.6%/76.1% 是论文综合指标的相对差异，不是准确率百分点或任一单任务提升；结果绑定论文模型与 benchmark。 |
| FIM rate 消融以 6 个大模型、每个 50B tokens 比较 0%、25%、50%、75%、90%、100%；该实验中最高到 90% 未见明显左到右能力退化，100% 普通自回归 loss 退化。上下文级相对文档级 FIM 的采样评测优势不能由约 0.001 nats/token 的近乎相同 PPL 充分反映。 | [FIM §4.2、§4.4](https://arxiv.org/html/2207.14255)，原始论文，复核于 2026-11-24。 | 限于论文代码模型、数据、训练预算与 infilling benchmark；不构成跨语料 FIM rate 的通用推荐。 |
| 独立未来预测头的 MTP 让共享主干表示并行预测多个未来 token；预测距离末端位置必须按有效目标 mask 处理。该论文讨论自推测解码作为可选复用方式，不能由辅助训练目标本身推出端到端加速。 | [Gloeckle et al.](https://arxiv.org/html/2404.19737)，原始论文，复核于 2026-11-24。 | MTP 头结构、损失权重与推理利用方式随实现变化；不可与串联式 MTP 结构混为一谈。 |
| MDLM 将掩码扩散训练目标化为随噪声/时间权重变化的 masked cross-entropy 损失平均；它不是无权重的标准 causal token mean。 | [MDLM §3.5](https://arxiv.org/html/2406.07524)，原始论文，复核于 2026-11-24。 | 具体 loss 权重与噪声日程依参数化和采样方案，不应把其数值直接与普通 causal PPL 排序。 |

