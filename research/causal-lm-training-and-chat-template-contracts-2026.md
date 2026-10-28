# 因果语言模型训练与聊天模板：来源证据

记录日期：2026-10-28

本记录保存联网核验得到的机制与接口说明，不记录书册审阅、修改决定或工作进度。

## 序列预测中的训练/推理前缀差异

### 来源

- Samy Bengio et al., “Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks,” arXiv:1506.03099v3，[论文页面](https://arxiv.org/abs/1506.03099v3)。
  - 实际阅读范围：arXiv 页面摘要和版本信息；摘要说明训练时按真实前序 token 最大化序列似然，推理时前序 token 来自模型，并指出这种差异可能导致错误沿序列累积；论文提出 scheduled sampling 作为从真实 token 逐步转向模型生成 token 的训练课程。
  - 证据等级：source-backed（特定循环神经网络序列预测方法及论文实验）。
  - 适用边界：摘要和论文实验不能证明该训练策略对所有架构、所有 Transformer 生成任务均有效；“exposure bias”不能作为所有生成失败的单一解释。

## 因果语言模型的标签对齐

### 来源

- Hugging Face Transformers, “Causal language modeling,” 文档版本 v5.17.0，[官方文档](https://huggingface.co/docs/transformers/v5.17.0/tasks/language_modeling)。
  - 实际阅读范围：因果语言建模说明、数据拼接/切块示例、`labels = input_ids.copy()`、`tokenizer.pad_token = tokenizer.eos_token` 以及 `DataCollatorForLanguageModeling(mlm=False)` 段落。教程描述 causal LM 会将输入作为 labels，并在训练目标中错开一个 token。
- Hugging Face Transformers v5.17.0 固定版本源码：[DataCollatorForLanguageModeling](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/data/data_collator.py#L3414-L3422)、[GPT2LMHeadModel](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/gpt2/modeling_gpt2.py#L3248-L3253)。
  - 实际阅读范围：`mlm=False` 分支将输入复制为 labels 并把 padding label 设为 `-100`；GPT-2 causal LM 头的文档说明 labels 在模型内移位。
  - 证据等级：source-backed（固定 revision 的公开实现路径和接口说明）。
  - 适用边界：任务教程的概括把“右移后的训练目标”与 data collator 放在同一流程中；固定 revision 源码表明 collator 的职责是生成同长度 labels 和忽略 padding，GPT-2 模型头负责内部移位。不可把这一责任划分推广到所有 Transformers 模型头、版本或其他框架；实际实现须检查具体 loss 契约。该文档页面提示已有 v5.21.0 更新版本，但本次浏览工具未能打开该版本页面，因此不把 v5.17.0 描述为最新文档。
  - 条件推论：教程示例把 pad token ID 配成 EOS token ID；该版本 collator 又将所有等于 `pad_token_id` 的 label 设为 `-100`。若实际训练直接采用这两个条件，真实序列中的 EOS labels 也会被忽略。该风险是从固定教程配置和 collator 实现推得，是否影响具体训练取决于数据整理和 mask 逻辑，不可外推为通用行为。

## 聊天格式与特殊 token

### 来源

- Hugging Face Transformers, “Chat templates,” 文档版本 v5.17.0，[官方文档](https://huggingface.co/docs/transformers/v5.17.0/chat_templating)。
  - 实际阅读范围：聊天消息转 token 序列的说明、由同一 Mistral-7B 基础模型微调得到的 Mistral 与 Zephyr 格式示例、特殊 token 去重警告、生成提示和训练模板段落。
  - 来源支持：因果语言模型继续生成 token 序列；不同模型的聊天控制 token/模板可能不同，即使基础模型相同；错误控制 token 会降低文档所述模型的表现；模板已有 BOS/EOS 时再额外添加特殊 token 可能造成重复并损害表现。
  - 证据等级：source-backed（该版本 Transformers 文档展示的格式示例和使用建议）。
  - 适用边界：示例不意味着所有聊天模型都使用相同控制 token，也不定义通用的 prompt loss mask 策略；训练格式需匹配具体模型/tokenizer 约定。

## 生命周期

因果 next-token 训练、标签移位与模型特定聊天格式：active。具体张量长度、loss 移位位置、特殊 token 和 mask 行为取决于模型、tokenizer、训练目标、框架及版本；需要按运行配置核验。
