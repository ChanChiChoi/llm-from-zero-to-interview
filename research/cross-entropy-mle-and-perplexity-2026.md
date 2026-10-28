# 交叉熵、最大似然与困惑度：来源证据

记录日期：2026-10-28

本记录保存通过原始数学定义和官方文档核验的知识点与证据边界，不记录书册审阅、修改历史或工作状态。

## 交叉熵与忽略标签

### 来源

- PyTorch, `torch.nn.CrossEntropyLoss`, v2.14，[官方 API 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)。
  - 实际阅读范围：未归一化 logits 和类别索引输入约定、逐样本公式、`ignore_index` 掩码，以及 `reduction='mean'` / `'sum'` 定义。
  - 来源支持：交叉熵接口可以接收未归一化 logits；对类别索引目标，单项为 log-softmax 后的负目标 log 概率；忽略 index 不进入损失；默认均值的分母按有效权重归一。
  - 证据等级：source-backed（固定 API 版本的公开接口与数学定义）。
  - 适用边界：自定义训练循环、加权目标、token/sequence 聚合和其他框架可能采用不同 reduction 或 mask 约定；不能只凭函数名推断有效 token 分母。

## 困惑度与固定上下文评估

### 来源

- Hugging Face Transformers, “Perplexity of fixed-length models”，官方文档页面，[文档链接](https://huggingface.co/docs/transformers/perplexity)（访问日期：2026-10-28；页面 URL 未固定版本）。
  - 实际阅读范围：PPL 定义、tokenization 对指标的影响、固定长度模型的上下文限制、不重叠分块与滑动窗口比较，以及 GPT-2/WikiText-2 示例中的 target mask、stride 和 token 归一化。
  - 来源支持：自回归/因果语言模型的 PPL 是序列平均负对数似然的指数；tokenization 会直接影响 PPL；有限上下文长度下，互不重叠的块会减少多数预测可见的上下文，滑动窗口或较小 stride 可在计算量与上下文利用率间取舍。
  - 证据等级：source-backed（官方教程中的定义、方法说明和 GPT-2 示例）。
  - 适用边界：文档示例使用 GPT-2 和 WikiText-2，不代表其分数可与不同 tokenizer、模型上下文长度、语料或 mask 口径直接比较；该页面没有固定 revision，记录时不把它称为版本锁定接口。文档对 masked language model 的说明指标准左到右 PPL 定义，不排除另行定义的 pseudo-perplexity 等指标。
  - 实现阅读备注：该页示例先按 `trg_len` 将重叠窗口中的上下文 labels 设为 `-100`，随后又用 `num_valid_tokens - batch_size` 估计模型内部 shift 后的目标数。对后续重叠窗口，首位 label 已在 mask 前缀内，直接再减一个可能使分母少计一个有效目标；这是根据示例代码的静态推导，未本地运行。记录标准定义和窗口方法时不沿用页面报告的具体 PPL 数值；自建实现应从 shift 后的有效目标 mask 计算分母。

## 数学关系

在 one-hot 监督下，交叉熵退化为真实类别的负对数概率；对一个序列的条件概率取乘积后再取负对数，会得到各位置负对数概率之和。按有效 token 平均并使用自然对数时，PPL 等于平均负对数似然的指数。它们是定义之间的代数关系；教学例子可使用合成概率，但不能把这些数值表述成模型实测结果。

## 生命周期

交叉熵、最大似然和标准自回归 perplexity：active。PPL 的可比性依赖 tokenizer、上下文窗口、stride、目标 mask、有效 token 计数与聚合方式；跨配置报告需给出这些条件。
