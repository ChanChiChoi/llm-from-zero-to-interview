# 第三章：Tokenizer 与格式坑

Tokenizer 和格式问题，是大模型训练、SFT、对齐、RAG 和 Agent 项目中最隐蔽的坑之一。它们不像显存溢出那样直接报错，也不像 loss 爆炸那样明显；很多时候，模型只是“怪怪的”：不停生成、复述用户输入、不听 system prompt、工具调用格式错、多轮对话角色混乱、中文和代码效率异常。

本章系统讲 tokenizer 与格式坑：special token、chat template、BOS/EOS/PAD、label mask、训练推理格式一致性、中文/代码/数学 tokenization、多模态特殊 token，以及排查 checklist。

## 3.1 核心观点

很多“模型不听话”的问题，本质是格式和 label mask 错误。

例如：

1. 模型总是复述用户问题，可能是 prompt 部分参与了 loss。
2. 模型不停生成，可能是 EOS 没学到或推理时 EOS 配错。
3. 模型忽略 system prompt，可能是 chat template 和训练格式不一致。
4. 工具调用 JSON 总坏，可能是训练数据格式和推理约束不一致。
5. 多模态模型看不到图，可能是图像 token 位置和模型预期不一致。

面试回答：

```text
大模型训练里 tokenizer 和格式问题非常常见。排查时我会打印原始样本、token ids、decode 后文本、special token、attention mask 和 label mask，确认训练和推理 chat template 完全一致。很多 SFT 不听话、复述用户输入、不停生成和工具调用格式错误，都不是模型能力问题，而是 BOS/EOS/PAD、role token 或 loss mask 配错。
```

## 3.2 常见问题

1. 特殊 token 配错导致训练和推理不一致。
2. chat template 改动后历史 checkpoint 行为变化。
3. EOS 处理错误导致模型不停生成。
4. PAD token 参与 loss 导致训练异常。
5. prompt 部分没有 mask，SFT 学会复述用户输入。
6. 中文、代码、数学符号 tokenization 效率低。
7. 多模态特殊 token 和图像 token 对齐错误。

这些问题的共同特征是：模型还能训练、还能推理，但行为会悄悄偏掉。

## 3.3 Special Token 配错

常见 special token：

1. BOS：序列开始。
2. EOS：序列结束。
3. PAD：填充。
4. UNK：未知 token。
5. system/user/assistant role token。
6. tool call token。
7. image/audio/video placeholder token。

配错会导致：

1. 模型不知道什么时候结束。
2. padding 被当成真实内容学习。
3. role 边界混乱。
4. 推理时 prompt 格式和训练不同。
5. 多模态输入位置错。

经验：任何更换 tokenizer、chat template 或 special token 的操作，都应该视为高风险变更。

## 3.4 BOS 和 EOS 坑

EOS 错误最常见。

表现：

1. 模型不停生成。
2. 生成到最大长度才停。
3. 多轮对话把下一轮 user 也生成出来。
4. 回答后继续输出模板符号。
5. stop words 不生效。

排查：

1. 训练样本是否包含 EOS。
2. label 中 EOS 是否参与 loss。
3. 推理配置的 eos_token_id 是否正确。
4. tokenizer decode 后 EOS 是否可见。
5. chat template 是否在 assistant 结束处加 EOS。

BOS 也要注意。有些模型训练时需要 BOS，有些不需要。重复加 BOS 或漏加 BOS 都可能改变行为。

## 3.5 PAD Token 参与 Loss

PAD 参与 loss 是训练事故。

表现：

1. loss 异常。
2. 模型学会生成 padding 相关 token。
3. 长短样本 loss 分布异常。
4. batch 内 padding 多时训练不稳定。

正确做法通常是：

```text
input_ids: 包含 pad
attention_mask: pad 位置为 0
labels: pad 位置为 -100
```

排查时不要只看 input_ids，要看 labels 和 attention_mask。

## 3.6 Label Mask 坑

SFT 中，通常只希望 assistant 部分参与 loss。

如果 user prompt 也参与 loss，模型会学会复述用户输入或模仿 prompt。

错误样例：

```text
User: 解释 Transformer
Assistant: Transformer 是...
```

如果整段都算 loss，模型被训练去预测 User 部分。这会污染指令跟随。

正确思路：

1. system 和 user 部分 labels 设为 -100。
2. assistant 答案部分参与 loss。
3. EOS 是否参与 loss 要明确。
4. 多轮对话每轮 assistant mask 都要正确。

## 3.7 Chat Template 不一致

训练和推理 chat template 不一致，是 SFT 常见问题。

训练格式可能是：

```text
<|system|>...
<|user|>...
<|assistant|>...
```

推理格式却是：

```text
### Instruction:
...
### Response:
```

模型当然会不稳定。

表现：

1. 不听 system prompt。
2. 回答开头奇怪。
3. 多轮对话角色错乱。
4. 生成模板残留符号。
5. 指令跟随变差。

排查：训练前后都打印完整 prompt，逐字符对比模板。

## 3.8 历史 Checkpoint 和 Template 变更

一个危险操作是：训练中途或上线后修改 chat template。

后果：

1. 旧 checkpoint 行为变差。
2. 新旧评估不可比。
3. 线上 prompt 不兼容。
4. 工具调用格式破坏。

如果必须改 template：

1. 记录版本。
2. 跑回归评估。
3. 明确 checkpoint 兼容范围。
4. 灰度上线。
5. 保留回滚路径。

chat template 是模型协议，不是普通字符串。

## 3.9 多轮对话格式坑

多轮对话比单轮更容易错。

问题：

1. role 顺序错。
2. assistant 答案缺 EOS。
3. user 和 assistant 边界不清。
4. 历史轮次过长。
5. 只 mask 最后一轮，前面 assistant 未参与训练。
6. 所有轮次都参与 loss，但格式不一致。

排查时要手工 decode 一条多轮样本，看它是否和推理时完全一致。

## 3.10 工具调用格式坑

Tool calling 对格式更敏感。

常见问题：

1. JSON 示例不一致。
2. 工具名训练和线上不同。
3. 参数 schema 改了但数据没改。
4. 工具调用和自然语言回答混在一起。
5. 缺少 tool observation role。
6. 多工具调用顺序不清。

表现：

1. 模型生成不可解析 JSON。
2. 调错工具。
3. 参数缺失。
4. 工具结果不被使用。

工具调用要有结构化 schema、严格校验和格式回归测试。

## 3.11 中文 Tokenization 效率

不同 tokenizer 对中文效率差异很大。

表现：

1. 同样中文文本 token 数过多。
2. 上下文窗口有效容量变小。
3. 推理成本上升。
4. 中文长文任务更容易截断。

排查：

1. 统计中英文 token/char 比例。
2. 对领域中文术语抽样。
3. 看长文截断比例。
4. 评估中文任务成本。

中文产品不能只按英文 token 经验估算上下文和成本。

## 3.12 代码和数学符号 Tokenization

代码和数学对 tokenizer 也敏感。

问题：

1. 缩进和空格被切得很碎。
2. 常见代码符号 tokenization 低效。
3. Unicode 数学符号处理差。
4. LaTeX 公式被切碎。
5. 特殊字符 decode 不一致。

影响：

1. 代码上下文容量下降。
2. 生成格式更容易错。
3. 数学表达不稳定。
4. 成本上升。

代码模型和数学模型要单独看 tokenization 统计，而不是只看自然语言。

## 3.13 多模态 Special Token 坑

多模态模型通常用特殊 token 表示图像、音频或视频位置。

常见问题：

1. 图像 placeholder 数量不匹配。
2. 图像 token 插入位置错。
3. 多图顺序错。
4. 图文对齐标签错。
5. 训练和推理的 image token 格式不同。
6. 截断时把图像 token 或问题截掉。

表现：

1. 模型像没看到图。
2. 多图问答混淆。
3. 回答引用错误图片。
4. 图像理解能力突然下降。

多模态排查一定要打印文本 token 序列和图像特征位置。

## 3.14 截断策略坑

上下文过长时需要截断。

常见坑：

1. 截掉 system prompt。
2. 截掉用户最新问题。
3. 截掉 assistant 答案开头。
4. 截掉图像 token。
5. 截断后 label mask 错位。
6. RAG 文档挤掉任务指令。

截断策略必须明确优先级。通常最新用户问题、系统约束和必要工具 schema 不能被截掉。

## 3.15 Decode 检查

最简单有效的排查方法：decode 回来看。

每次数据处理后都应该抽样检查：

1. 原始文本。
2. token ids。
3. decode 后文本。
4. attention mask。
5. labels。
6. label 为 -100 的位置。
7. special token 位置。

很多格式 bug 一 decode 就能发现。

## 3.16 排查清单

1. 打印原始文本、token ids、decode 后文本。
2. 检查 BOS、EOS、PAD、system、user、assistant token。
3. 检查 label mask。
4. 检查训练和推理 chat template 是否完全一致。
5. 对一条样本手动计算 loss 范围。

扩展清单：

1. 检查 eos_token_id 和 pad_token_id 是否正确。
2. 检查 PAD 是否 label 为 -100。
3. 检查 assistant 部分是否参与 loss。
4. 检查 prompt 部分是否被 mask。
5. 检查多轮对话 role 顺序。
6. 检查 tool call 格式。
7. 检查推理 stop token。
8. 检查 truncation 是否截掉关键内容。
9. 检查多模态 placeholder 和特征数量是否匹配。
10. 检查新旧 tokenizer 是否兼容。

## 3.17 最小复现样本

遇到格式问题，应该构造最小样本。

例如：

```text
System: 你是助手
User: 只回答一个字：好
Assistant: 好
```

检查：

1. decode 后是否包含预期 role。
2. user 部分是否 mask。
3. assistant 的“好”和 EOS 是否参与 loss。
4. 推理时是否能停止。

一个极简样本跑通后，再扩展到多轮、工具、多模态。

## 3.18 事故复盘例子

例子：SFT 后模型总复述用户问题。

排查路径：

1. 先看生成样例，发现回答前重复 user 内容。
2. 检查训练样本 decode，格式看似正常。
3. 打印 labels，发现 user 部分没有设为 -100。
4. 修复 collator 的 mask 逻辑。
5. 用小样本重新训练验证。
6. 回归多轮对话和指令跟随评估。

复盘结论：不是模型能力问题，而是 label mask 事故。

## 3.19 面试题：SFT 后模型复述用户输入怎么办

回答要点：

```text
我会先怀疑 label mask 和 chat template。具体会打印一条训练样本的原文、token ids、decode 文本和 labels，检查 user/system 部分是否被设为 -100，assistant 部分是否参与 loss，EOS 是否正确。如果 prompt 部分参与了 loss，模型就会学习预测用户输入，表现为复述用户问题。
```

## 3.20 面试题：模型不停生成怎么排查

回答要点：

```text
我会先检查 EOS。包括训练样本中是否有 EOS，EOS 是否参与 loss，推理配置里的 eos_token_id 是否正确，stop token 是否和 chat template 对齐，以及 decode 后是否能看到 assistant 结束边界。如果 EOS 配错或训练时没学到结束符，模型就可能一直生成到最大长度。
```

## 3.21 经验法则

很多“模型不听话”的问题，本质是格式和 label mask 错误。

更具体地说：

1. 先 decode，再猜原因。
2. 先看 labels，再看 loss。
3. 训练和推理 template 必须一致。
4. EOS/PAD 错误会造成非常诡异的行为。
5. chat template 是协议，不是普通 prompt。
6. 工具调用和多模态比普通对话更依赖格式。

下一章会进入训练不稳定排查。很多 loss 异常和训练崩溃也可能从格式问题开始，但还会涉及学习率、梯度、混合精度、数据 batch 和分布式系统。
