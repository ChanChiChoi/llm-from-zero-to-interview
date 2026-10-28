# 因果训练目标、监督遮罩与生成前缀：事实和证据

检索日期：2026-10-28。

## 因果可见性与标签错位

- Vaswani 等人的 Transformer 原论文 v7（2023-08-02）§3.1 说明 decoder 的输出嵌入错开一位，并与自注意力遮罩共同保证位置 i 的预测只能依赖先前已知输出；§3.2.3 说明对非法的后续位置在 softmax 前设为负无穷。来源：[论文 HTML §3.1](https://arxiv.org/html/1706.03762v7#S3.SS1)、[§3.2.3](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS3)。证据等级：原始论文中的架构定义。边界：支持原论文 decoder 的设计和机制，不表示所有框架公开的 mask API 使用相同张量形状或布尔语义。
- 本记录中以 C_{ij}=0（j≤i）、C_{ij}=-∞（j>i）表示的加性因果遮罩，是对上述机制的数学写法；softmax 后被屏蔽位置的权重为零。它不规定 padding 位置或目标标签是否计入 loss。

## Causal LM 与 Masked LM

- Devlin 等人的 BERT 论文 v2（2019-05-24）摘要与 §3.1 说明，BERT 通过 masked language modeling 选择输入中的部分 token 作为预测目标，并使用左右上下文恢复其原词；论文将该目标用于双向表示预训练。来源：[论文 HTML §3.1](https://arxiv.org/html/1810.04805v2#S3.SS1)。证据等级：原始论文对 BERT 目标及其设计的定义。边界：说明 BERT 的训练目标，不等同于自回归下一个 token 生成；具体遮盖替换比例属于该论文的输入构造方案，不应据此概括所有 Masked LM。

## Teacher forcing 与训练/生成前缀

- Bengio 等人的 Scheduled Sampling 论文 v3（2015-09-23）§1、§2.1–2.4 描述序列预测中的训练/推理差异：训练条件使用真实的前一目标 token，推理时改用模型选择或采样出的 token；论文讨论偏离后的生成状态可能导致后续错误，并在 RNN 序列任务上提出 scheduled sampling。来源：[论文 HTML §1](https://arxiv.org/html/1506.03099v3#S1)、[§2.4](https://arxiv.org/html/1506.03099v3#S2.SS4)。证据等级：原始论文的研究动机与方法描述。边界：论文研究的是其 RNN 序列预测任务；它支持“真实前缀与模型生成前缀不同”这一概念来源，不证明现代 LLM 的每次长文本失败都由 exposure bias 导致，也不证明该论文方法可普遍修复此问题。

## 对话模板与 assistant-only loss 的版本行为

- Hugging Face Transformers v5.17.0 官方聊天模板文档说明，不同模型可把同一组角色/内容消息格式化为不同控制 token 序列；模板可能处理 BOS/EOS 特殊 token，且添加 generation prompt 会改变 assistant 响应的起始边界。来源：[Chat templates](https://huggingface.co/docs/transformers/v5.17.0/chat_templating)，阅读位置：“Chat templates”概述、特殊 token 提示及 add_generation_prompt。证据等级：固定版本官方文档。边界：适用于该文档所描述的 Transformers 接口，不定义任意模型服务的停止条件或所有聊天协议。
- Hugging Face TRL v0.29.0 SFTTrainer 文档说明，assistant_only_loss=True 可用于仅对 assistant 响应计算 loss、忽略 system/user 内容；该功能要求聊天模板能借助 generation 标记返回 assistant token mask。该版本指标定义将 loss 描述为对未被 mask 的 token 计算的交叉熵平均。来源：[SFTTrainer 文档](https://huggingface.co/docs/trl/v0.29.0/en/sft_trainer)，阅读位置：“Train on assistant messages only”及训练指标字段。证据等级：固定版本官方文档。边界：这是 TRL v0.29.0 的配置与模板要求示例，不构成其他训练库的统一参数、标签移位约定或 reduction 规则。

## 未验证边界

- 未运行任何训练代码、框架 API 或模型服务；未验证特定 tokenizer 的结束 token、实际停止配置、padding 约定、位置 ID、label shift 所在层级或分布式 loss reduction。
- 本记录中的手算概率、token ID 和变长 batch 是构造的教学输入，不是论文实验、模型测量或生产行为证据。
