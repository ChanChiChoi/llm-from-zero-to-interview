# 第 4 章（第二篇）：把对话变成监督信号

基座模型已经能按固定模板生成回答。现在要决定训练时究竟教它什么：用户消息通常是条件，assistant 的回答通常是预测目标，但一条对话会被模板展开成一整串 token。读完这一篇，你应能检查模板边界、回答 mask、截断和 padding，知道每个有效 label 对应哪一个回答 token。

## 4.2 把对话变成监督信号

### 4.2.1 SFT 学习的是条件分布

先看一条最简单的样本：用户问“天气如何？”，assistant 回答“晴天”。模型必须同时看到问题和回答，才能学到“在这个问题下怎样接着说”；但如果训练目标是 assistant-only，问题 token 只是上下文，真正进入 loss 的是回答 token 和约定的结束符。监督微调（supervised fine-tuning, SFT）因此学习的是给定对话条件时的回答分布。

这类样本至少包含条件和目标：

```text
条件：system、user 以及此前的对话历史。
目标：assistant 希望生成的回答。
```

单轮样本可以写成：

```math
(\mathrm{prompt}_i,\mathrm{response}_i)
```

完整训练序列是：

```math
x_i
=
\mathrm{Tokenize}
(\mathrm{prompt}_i\oplus\mathrm{response}_i\oplus\mathrm{EOS})
```

只把 response 喂给模型会丢掉任务条件，只把 prompt 喂给模型又没有目标。SFT 的目标是提高：

```math
p_\theta(\mathrm{response}\mid\mathrm{prompt})
```

传统 instruction/input/output 数据可以写成：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
example = {
    "instruction": "把下面的句子翻译成英文。",
    "input": "我喜欢机器学习。",
    "output": "I like machine learning.",
}
```
现代聊天数据常写成 messages：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
messages = [
    {"role": "system", "content": "你是一个严谨的助手。"},
    {"role": "user", "content": "什么是梯度下降？"},
    {
        "role": "assistant",
        "content": "梯度下降沿负梯度方向更新参数以降低损失。",
    },
]
```
两种表示最终都要变成 token id。决定模型行为的不是字段名称，而是模板渲染后的序列、特殊 token 和参与 loss 的位置。

### 4.2.2 模板属于训练分布

一个简单模板可以这样渲染：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
def format_instruction(example):
    instruction = example["instruction"].strip()
    extra_input = example.get("input", "").strip()
    if extra_input:
        prompt = (
            "### Instruction:\n"
            + instruction
            + "\n\n### Input:\n"
            + extra_input
            + "\n\n### Response:\n"
        )
    else:
        prompt = (
            "### Instruction:\n"
            + instruction
            + "\n\n### Response:\n"
        )
    return prompt, example["output"].strip()
```
推理时必须复用同一套边界：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
def build_prompt(instruction, extra_input=""):
    if extra_input.strip():
        return (
            "### Instruction:\n"
            + instruction.strip()
            + "\n\n### Input:\n"
            + extra_input.strip()
            + "\n\n### Response:\n"
        )
    return (
        "### Instruction:\n"
        + instruction.strip()
        + "\n\n### Response:\n"
    )
```
聊天模型应优先使用自己的 tokenizer 提供的 chat template：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
prompt_messages = [
    {"role": "system", "content": "你是一个严谨的助手。"},
    {"role": "user", "content": "解释过拟合。"},
]

prompt = tokenizer.apply_chat_template(
    prompt_messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

add_generation_prompt=True 适合推理，把序列渲染到 assistant 即将开始的位置；训练样本已经含有 assistant 回答时，通常使用 False。一个模型的 role 标记、换行和结束 token 不能移植到另一个模型。模板的微小差异也会改变 token 边界和监督位置。
### 4.2.3 labels mask 的数学定义

设完整序列为 $x_0,\ldots,x_{T-1}$，回答区域由 $m_t$ 标记：

```math
m_t=
\begin{cases}
0,& t\in\mathrm{prompt\ or\ padding}\\
1,& t\in\mathrm{assistant\ response}
\end{cases}
```

labels 写成：

```math
y_t=
\begin{cases}
-100,&m_t=0\\
x_t,&m_t=1
\end{cases}
```

PyTorch 的交叉熵默认把 -100 当作 `ignore_index`。prompt token 仍然保留在 `input_ids` 中，作为回答的上下文；它们只是不会成为监督目标。参见 [PyTorch 2.14.0 CrossEntropyLoss 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)。

若模型内部负责 causal shift，labels 与 input_ids 等长，位置 $t-1$ 的 logits 预测 labels 位置 $t$。令 $S=\sum_{t=1}^{T-1}m_t$ 为这个序列中真正参与监督的回答 token 数。下面的 assistant-only 目标只在 $S>0$ 时有定义；一个被截断到只剩 prompt 的样本不应进入 loss 的平均：

```math
\mathcal{L}_{\mathrm{assistant}}
=
-
\frac{
\sum_{t=1}^{T-1}
m_t\log p_\theta(x_t\mid x_{0:t-1})
}{
\sum_{t=1}^{T-1}m_t
}
```

回答结尾的 EOS 是否参与监督必须明确。把 EOS 纳入回答区域，模型更容易学到停止位置；如果模板把 EOS 放在 assistant 区域之外，mask 可能把它漏掉。应把有效 labels 解码出来检查。

### 4.2.4 用 offset mapping 构造回答 mask

对简单文本模板，可以用 fast tokenizer 的字符偏移定位 prompt 结束位置：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
def tokenize_sft_example(example, tokenizer, max_length=512):
    if not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("offset mapping requires a fast tokenizer")
    prompt, response = format_instruction(example)
    if not response:
        raise ValueError("response must contain at least one character")
    if tokenizer.eos_token is None:
        raise ValueError("tokenizer.eos_token is required for this example")
    full_text = prompt + response + tokenizer.eos_token
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )

    labels = []
    prompt_end = len(prompt)
    for token_id, (start, end) in zip(
        encoded["input_ids"],
        encoded["offset_mapping"],
    ):
        # Token boundaries cannot be split; mask any token that overlaps the prompt.
        labels.append(
            -100 if start < prompt_end else token_id
        )
    if not labels or not any(label != -100 for label in labels):
        raise ValueError(
            "truncation removed every supervised token; reserve space for response"
        )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }
```
不能默认把 tokenize(prompt) 得到的 token 数当作完整序列的切分点。BPE 或 byte-level tokenizer 可能在边界处合并片段，两个独立 tokenize 的边界不一定相同。offset mapping 描述 token 覆盖的字符区间，通常更可靠；但它要求 fast tokenizer，特殊 token 的 offset 仍需实际检查。若 token 横跨 prompt 与 response 的分界，tokenizer 无法把它拆成半个 token；这里保守地把整个边界 token mask 掉，避免把 prompt 字符纳入目标，代价是可能少监督回答开头的一小段。截断后仍须检查至少有一个回答 token 留在监督区。

聊天模板可以尝试使用 assistant mask：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    add_generation_prompt=False,
    return_assistant_tokens_mask=True,
)

if "assistant_masks" not in encoded:
    raise RuntimeError(
        "this tokenizer/template did not return assistant_masks; "
        "inspect the template before constructing labels"
    )
input_ids = encoded["input_ids"]
assistant_mask = encoded["assistant_masks"]
if len(input_ids) != len(assistant_mask) or not any(assistant_mask):
    raise ValueError("assistant mask must align with input_ids and mark a response")
labels = [
    token_id if flag else -100
    for token_id, flag in zip(input_ids, assistant_mask)
]
```
这个接口依赖 Transformers 版本和模板中的 `{% generation %}` 标记。Transformers v5.17.0 文档规定，只有支持该标记的 chat template 才能返回 assistant mask；运行时应打印 `encoded.keys()`，确认返回的 mask 真的覆盖回答。不支持 mask 时，应经过验证后使用 offset mapping 或框架提供的 completion-only 方案，不能无声地把整条序列都当成回答。参见 [Transformers v5.17.0 tokenizer 文档](https://huggingface.co/docs/transformers/v5.17.0/main_classes/tokenizer)。

使用 TRL `SFTTrainer` 时，参数名还会随数据格式选择不同监督范围。TRL v1.14.1 把 `completion_only_loss` 用于 prompt-completion 数据，把 `assistant_only_loss` 用于 conversational 数据；后者依赖 chat template 暴露 assistant generation 区间。二者不是“开哪个都一样”：prompt-completion 的 completion 边界与多轮对话中的 assistant 消息边界并不相同。若数据已预先构造 `labels=-100`，该版本会直接采用这些 labels；仍要核对 mask 是否与预期监督范围一致。参见 [TRL v1.14.1 SFTTrainer 文档](https://huggingface.co/docs/trl/v1.14.1/sft_trainer)。

### 4.2.5 截断与有效监督 token

回答通常位于序列尾部，普通 truncation 又常从尾部截断，因此可能出现：

```text
prompt 被保留。
response 被完全截掉。
labels 全是 -100。
训练循环仍运行，但没有有效监督。
```

有效监督 token 数为：

```math
n_{\mathrm{sup}}
=
\sum_{t=0}^{T-1}\mathbf{1}[y_t\neq-100]
```

样本至少要满足 $n_{\mathrm{sup}}>0$。如果任务要求完整 JSON、代码或回答，还要检查 EOS 是否存在、结构是否被截断。

> 代码状态：静态审阅，未运行（2026-10-29）

```python
def has_supervision(feature):
    return any(label != -100 for label in feature["labels"])


def supervision_count(feature):
    return sum(label != -100 for label in feature["labels"])
```
更稳妥的做法是先限制过长对话历史，再为回答保留空间；不能只把 max_length 调大而不检查显存和模型上下文上限。

### 4.2.6 三个字段必须同步 padding

batch 中应有：

```math
X,A,Y\in\mathbb{Z}^{B\times T_{\max}}
```

```text
X：input_ids，padding 使用 pad_token_id。
A：attention_mask，padding 使用 0。
Y：labels，padding 使用 -100。
```

最小 collator：

本实现固定使用右侧 padding，符合下面 `pad_sequence` 的输出方式。上一节生成示例中的左侧 padding 服务于 decoder-only 批量生成，两种场景的 padding 契约不能混用。

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import torch


class SFTDataCollator:
    def __init__(self, tokenizer, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        if tokenizer.pad_token_id is None:
            raise ValueError("SFT padding requires tokenizer.pad_token_id")
        if tokenizer.padding_side != "right":
            raise ValueError("this collator requires right-side padding")
        if label_pad_token_id != -100:
            raise ValueError("this chapter uses -100 as the loss ignore index")

    def __call__(self, features):
        if not features:
            raise ValueError("cannot collate an empty feature list")
        required = {"input_ids", "attention_mask", "labels"}
        for index, item in enumerate(features):
            missing = required.difference(item)
            if missing:
                raise ValueError(
                    f"feature {index} is missing fields: {sorted(missing)}"
                )
            lengths = {len(item[key]) for key in required}
            if len(lengths) != 1 or not lengths or min(lengths) == 0:
                raise ValueError(
                    f"feature {index} has empty or misaligned sequence fields"
                )
            vocab_size = len(self.tokenizer)
            if any(
                value < 0 or value >= vocab_size
                for value in item["input_ids"]
            ):
                raise ValueError(
                    f"feature {index} contains an input id outside the tokenizer vocabulary"
                )
            if any(value not in {0, 1} for value in item["attention_mask"]):
                raise ValueError(f"feature {index} has a non-binary attention mask")
            if any(
                value < 0 and value != self.label_pad_token_id
                for value in item["labels"]
            ):
                raise ValueError(f"feature {index} has an invalid label value")
            if any(
                value >= vocab_size
                for value in item["labels"]
                if value != self.label_pad_token_id
            ):
                raise ValueError(
                    f"feature {index} contains a target id outside the tokenizer vocabulary"
                )
            if not any(value != self.label_pad_token_id for value in item["labels"]):
                raise ValueError(
                    f"feature {index} has no supervised token after truncation"
                )
            if any(
                mask_value == 0 and label_value != self.label_pad_token_id
                for mask_value, label_value in zip(
                    item["attention_mask"], item["labels"]
                )
            ):
                raise ValueError(
                    f"feature {index} supervises a padding position"
                )
        input_ids = [
            torch.tensor(item["input_ids"], dtype=torch.long)
            for item in features
        ]
        attention_mask = [
            torch.tensor(item["attention_mask"], dtype=torch.long)
            for item in features
        ]
        labels = [
            torch.tensor(item["labels"], dtype=torch.long)
            for item in features
        ]
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                attention_mask,
                batch_first=True,
                padding_value=0,
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=self.label_pad_token_id,
            ),
        }
```
训练前做断言：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
assert batch["input_ids"].shape == batch["attention_mask"].shape
assert batch["input_ids"].shape == batch["labels"].shape
for mask_row, label_row in zip(
    batch["attention_mask"],
    batch["labels"],
):
    for mask_value, label_value in zip(
        mask_row.tolist(),
        label_row.tolist(),
    ):
        if mask_value == 0:
            assert label_value == -100
```
这种检查能在梯度计算前发现字段错位，比训练数小时后猜测 loss 异常原因更可靠。还要把三个 mask 概念分开：`attention_mask` 标记哪些输入位置是有效上下文；causal mask 限制每个位置能看见哪些未来 token；`labels=-100` 表示该目标位置不计 loss。padding 的 `attention_mask` 和 label 都应屏蔽，prompt 通常保留为上下文、只在 labels 中屏蔽。过滤或截断后若一个 batch 的 labels 全为 `-100`，平均 loss 没有有效分母，collator 应在更新前拒绝它。实际训练还要确认模型 embedding 行数覆盖 tokenizer 的全部 id；collator 不能替代模型侧检查。

监督边界已经画清楚，但样本是否独立、是否被截断、来自哪一版模板，还决定评估能否解释训练结果。下一篇先做数据审计，再进入全参数 SFT。
