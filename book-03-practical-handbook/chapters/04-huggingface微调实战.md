# 第四部分：Hugging Face 微调实战

## 第 19 讲：加载和运行一个开源 Causal LM

### 本讲目标

学完本讲，你应该能做到六件事：

1. 使用 Hugging Face Transformers 加载一个开源 causal LM。
2. 理解 tokenizer、model、config 三者的关系。
3. 写出最小可运行推理代码。
4. 掌握 `device_map`、`torch_dtype`、`eval()`、`no_grad()` 的作用。
5. 理解 `generate` 的基本参数。
6. 能排查模型下载、显存不足、tokenizer 不匹配等常见问题。

前面第三部分我们从零训练了一个小 GPT。

从这一讲开始，我们进入 Hugging Face 微调实战。

真实工作中，很少从零训练大模型。

更常见的是：

```text
加载一个开源基座模型。
构造指令数据。
做 SFT 或 LoRA 微调。
评估微调前后行为变化。
```

本讲先完成第一步：加载并运行一个开源 causal language model。

---

### 一、什么是 Causal LM

Causal LM 是 causal language model，也就是自回归语言模型。

它的训练目标是：

```text
根据当前位置及之前的 token，预测下一个 token。
```

GPT、LLaMA、Qwen、Mistral、Yi 等 decoder-only 模型都属于 causal LM。

Hugging Face 中常用类是：

```python
AutoModelForCausalLM
```

它表示：

```text
自动根据模型配置加载适合 causal language modeling 的模型结构。
```

对应 tokenizer 常用：

```python
AutoTokenizer
```

---

### 二、安装依赖

基础依赖：

```bash
pip install torch transformers accelerate
```

如果后续要做 LoRA、QLoRA，还会用到：

```bash
pip install peft bitsandbytes datasets trl
```

本讲只需要：

```text
torch
transformers
accelerate
```

如果没有 GPU，也可以先用很小的模型在 CPU 上跑通流程。

---

### 三、选择一个适合教学的小模型

为了避免显存压力，本讲建议使用小模型演示。

例如：

```text
sshleifer/tiny-gpt2
distilgpt2
gpt2
```

其中：

```text
sshleifer/tiny-gpt2：非常小，适合测试代码流程。
distilgpt2：比 GPT-2 小，适合轻量实验。
gpt2：经典小模型，但仍比 tiny-gpt2 大。
```

如果你有 GPU，也可以换成中文或开源指令模型，例如 Qwen 系列小模型。

但首次跑通建议先用小模型。

---

### 四、最小加载代码

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "sshleifer/tiny-gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()

prompt = "Hello, my name is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
print(logits.shape)
```

输出 shape 通常是：

```text
[batch, seq_len, vocab_size]
```

例如：

```text
torch.Size([1, 5, 50257])
```

这和我们前面从零实现的小 GPT 完全一致。

---

### 五、tokenizer 做了什么

```python
inputs = tokenizer(prompt, return_tensors="pt")
```

通常返回：

```python
{
    "input_ids": tensor(...),
    "attention_mask": tensor(...),
}
```

其中：

```text
input_ids：token id 序列。
attention_mask：哪些位置是有效 token。
```

打印：

```python
print(inputs)
print(tokenizer.decode(inputs["input_ids"][0]))
```

你会看到 tokenizer 把字符串变成了整数 id。

模型只认识 id，不直接认识字符串。

---

### 六、用 generate 生成文本

Hugging Face 模型自带 `generate`。

```python
generated_ids = model.generate(
    **inputs,
    max_new_tokens=50,
)

generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)
```

`max_new_tokens` 表示最多新生成多少个 token。

注意它不是总长度。

如果 prompt 长度是 5，`max_new_tokens=50`，最终最多是 55 个 token。

---

### 七、加入采样参数

上一讲我们手写过 temperature、top-k、top-p。

Hugging Face `generate` 也支持这些参数。

```python
generated_ids = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
)
```

关键参数：

```text
do_sample=True：启用随机采样。
temperature：控制分布尖锐程度。
top_k：只从概率最高的 k 个 token 采样。
top_p：只从累计概率达到 p 的候选集合采样。
```

如果不设置 `do_sample=True`，很多采样参数不会生效。

贪心生成：

```python
generated_ids = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=False,
)
```

---

### 八、使用 GPU 和 dtype

如果有 GPU：

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

inputs = tokenizer(prompt, return_tensors="pt").to(device)
```

对于较大模型，常用半精度：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to(device)
```

如果使用 bf16：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
).to(device)
```

经验：

```text
新一些的 NVIDIA GPU 通常支持 bf16。
老一些的 GPU 可能只适合 fp16。
CPU 上通常不要强行用 fp16。
```

---

### 九、device_map="auto"

加载稍大的模型时，可以使用：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
```

`device_map="auto"` 会让 accelerate 自动决定模型放在哪里。

可能是：

```text
单张 GPU
多张 GPU
部分 CPU offload
```

对于入门实战，你可以先记住：

```text
小模型：model.to(device) 就够。
较大模型：device_map="auto" 更方便。
```

但不要混用得太随意。

如果用了 `device_map="auto"`，通常不要再手动 `model.to(device)`。

---

### 十、完整推理脚本

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = "sshleifer/tiny-gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    prompt = "Hello, my name is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )
    print(generated_text)


if __name__ == "__main__":
    main()
```

这个脚本完成了：

```text
加载 tokenizer
加载 causal LM
编码 prompt
生成 token ids
解码为文本
```

---

### 十一、`model.eval()` 和 `torch.no_grad()`

推理时应该写：

```python
model.eval()
with torch.no_grad():
    ...
```

`model.eval()` 的作用：

```text
关闭 dropout 等训练时随机行为。
```

`torch.no_grad()` 的作用：

```text
不构建计算图，节省显存和计算。
```

如果忘记 `no_grad()`，推理也能跑，但会浪费显存。

如果忘记 `eval()`，生成可能受 dropout 影响，不够稳定。

---

### 十二、attention_mask 和 pad_token_id

很多 decoder-only 模型没有默认 padding token。

生成时可能看到警告：

```text
Setting pad_token_id to eos_token_id
```

可以显式设置：

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

生成时也可以传：

```python
pad_token_id=tokenizer.eos_token_id
```

`attention_mask` 用来告诉模型哪些 token 是有效输入。

对于单条不 padding 的 prompt，影响不大。

对于 batch 推理，它很重要。

---

### 十三、batch 推理

多个 prompt 一起推理：

```python
prompts = [
    "Hello, my name is",
    "The capital of France is",
]

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
).to(device)

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
    )

for ids in generated_ids:
    print(tokenizer.decode(ids, skip_special_tokens=True))
```

batch 推理时要注意：

```text
padding=True
attention_mask
pad_token_id
```

否则不同长度 prompt 可能处理不正确。

---

### 十四、查看模型配置

```python
print(model.config)
```

你可以看到：

```text
vocab_size
n_layer 或 num_hidden_layers
n_head 或 num_attention_heads
n_embd 或 hidden_size
max_position_embeddings
bos_token_id
eos_token_id
```

这些配置对应我们前面手写 GPT 时的超参数。

例如：

```text
hidden_size 对应 d_model。
num_attention_heads 对应 num_heads。
num_hidden_layers 对应 num_layers。
vocab_size 对应 tokenizer 词表大小。
```

理解 config 很重要。

微调、LoRA、量化和推理优化都会用到它。

---

### 十五、常见错误排查

#### 错误 1：模型下载失败

可能原因：

```text
网络不可用。
模型需要登录授权。
模型名写错。
```

解决：

```text
检查 model_name。
使用本地模型路径。
提前下载模型。
需要 gated access 的模型先登录并申请权限。
```

#### 错误 2：CUDA out of memory

解决：

```text
换更小模型。
使用 fp16/bf16。
减小 max_new_tokens。
减小 batch size。
使用 device_map="auto"。
后续使用 4bit 量化。
```

#### 错误 3：tokenizer 和 model 不匹配

表现：

```text
生成乱码。
embedding id 越界。
输出质量异常。
```

解决：

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

尽量使用同一个 `model_name` 加载 tokenizer 和 model。

#### 错误 4：输入和模型不在同一 device

表现：

```text
Expected all tensors to be on the same device
```

解决：

```python
inputs = tokenizer(prompt, return_tensors="pt").to(device)
model = model.to(device)
```

#### 错误 5：采样参数不生效

如果设置了 temperature、top_p，但输出仍像贪心，检查是否设置：

```python
do_sample=True
```

---

### 十六、面试怎么讲加载开源模型

如果面试官问“怎么用 Hugging Face 跑一个 causal LM”，可以这样回答：

```text
我会使用 AutoTokenizer 和 AutoModelForCausalLM 从同一个 model name 或本地路径加载 tokenizer 和模型。tokenizer 把 prompt 编码成 input_ids 和 attention_mask，模型 forward 输出 [B, T, vocab_size] logits；生成时使用 model.generate，设置 max_new_tokens、do_sample、temperature、top_k、top_p 等参数，最后用 tokenizer.decode 把 token ids 转回文本。
```

如果追问“`model.eval()` 和 `torch.no_grad()` 有什么作用”，可以回答：

```text
model.eval() 会关闭 dropout 等训练态行为，让推理稳定；torch.no_grad() 不构建计算图，可以节省显存和计算。推理时通常两者都要使用。
```

如果问“tokenizer 和模型为什么要匹配”，可以回答：

```text
模型 embedding 和输出 head 都是基于特定 vocab 训练的。如果 tokenizer 不匹配，同一个 token id 对应的文本片段可能不同，甚至 id 超出 embedding 范围，生成结果会异常，所以 tokenizer 和 model 必须来自同一个 checkpoint 或同一套训练资产。
```

如果问“显存不够怎么办”，可以回答：

```text
可以换更小模型、使用 fp16/bf16、减小 batch size 和生成长度、使用 device_map="auto" 做自动放置，或者在后续微调和推理中使用 8bit/4bit 量化。
```

---

### 十七、小练习

#### 练习 1

用 `sshleifer/tiny-gpt2` 跑通最小推理脚本。

#### 练习 2

打印 `inputs["input_ids"]`，再用 tokenizer decode 回文本。

#### 练习 3

分别测试 `do_sample=False` 和 `do_sample=True` 的生成差异。

#### 练习 4

调整 `temperature=0.7`、`top_p=0.9`，观察输出变化。

#### 练习 5

打印 `model.config`，找到 hidden size、层数、头数和 vocab size。

---

### 本讲总结

这一讲完成了开源 causal LM 的加载和运行。

核心结论如下：

1. Causal LM 是自回归语言模型，用于 next-token prediction。
2. Hugging Face 中常用 `AutoTokenizer` 和 `AutoModelForCausalLM`。
3. tokenizer 把文本转成 token ids，model 只处理 token ids。
4. 模型 forward 输出 logits，shape 是 `[B, T, vocab_size]`。
5. `generate` 封装了自回归生成过程。
6. 推理时应使用 `model.eval()` 和 `torch.no_grad()`。
7. tokenizer 和 model 必须匹配。
8. 显存不足时可以使用小模型、半精度、`device_map="auto"` 或量化。

下一讲，我们构造指令微调数据集，为 SFT 做准备。

## 第 20 讲：构造指令微调数据集

### 本讲目标

学完本讲，你应该能做到六件事：

1. 理解指令微调数据集的基本结构。
2. 构造 `instruction/input/output` 格式样本。
3. 构造 chat 格式的多轮对话样本。
4. 使用 tokenizer 把样本变成 `input_ids`、`attention_mask`、`labels`。
5. 理解为什么要 mask 掉 prompt 部分的 loss。
6. 能排查 SFT 数据中的常见质量问题。

上一讲我们加载并运行了开源 causal LM。

本讲开始准备 SFT 数据。

SFT 是 supervised fine-tuning，也就是监督微调。

它的核心是：

```text
给模型看一批“用户输入 -> 理想回答”的样本，让模型学习按照指令回答。
```

很多微调失败，不是模型或 LoRA 配置问题，而是数据格式、labels mask、chat template 或数据质量出了问题。

所以这一讲非常关键。

---

### 一、什么是指令微调数据

指令微调样本通常包含三部分：

```text
instruction：用户希望模型做什么。
input：可选的额外上下文。
output：期望模型输出的答案。
```

例如：

```json
{
  "instruction": "把下面这句话翻译成英文。",
  "input": "我喜欢机器学习。",
  "output": "I like machine learning."
}
```

如果没有额外上下文，也可以是：

```json
{
  "instruction": "解释什么是过拟合。",
  "input": "",
  "output": "过拟合是指模型在训练集上表现很好，但在未见数据上表现较差的现象。"
}
```

模型训练时看到的是拼接后的文本。

例如：

```text
### Instruction:
解释什么是过拟合。

### Response:
过拟合是指模型在训练集上表现很好，但在未见数据上表现较差的现象。
```

---

### 二、为什么不是直接喂 output

SFT 的目标不是让模型背答案。

而是让模型学会：

```text
在给定用户指令和上下文时，生成合适回答。
```

所以训练样本必须包含 prompt 和 response。

prompt 告诉模型任务是什么。

response 是模型要学习生成的内容。

如果只训练 output，模型不知道这个回答对应什么问题。

---

### 三、Alpaca 风格格式

早期很多开源 SFT 数据使用 Alpaca 格式。

有 input 时：

```python
def format_alpaca(example):
    if example.get("input", ""):
        prompt = (
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Input:\n"
            f"{example['input']}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Response:\n"
        )

    response = example["output"]
    return prompt, response
```

完整文本是：

```python
prompt, response = format_alpaca(example)
text = prompt + response
```

这种格式直观、容易调试。

缺点是不同模型可能有自己的 chat template。

真实微调时，应优先使用目标模型推荐的对话模板。

---

### 四、Chat 格式数据

很多现代指令模型使用 chat 格式。

一条样本可能是：

```python
messages = [
    {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
    {"role": "user", "content": "解释什么是梯度下降。"},
    {"role": "assistant", "content": "梯度下降是一种通过沿负梯度方向更新参数来最小化损失函数的优化方法。"},
]
```

多轮对话：

```python
messages = [
    {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
    {"role": "user", "content": "什么是过拟合？"},
    {"role": "assistant", "content": "过拟合是模型在训练集上表现好但泛化差的现象。"},
    {"role": "user", "content": "怎么缓解？"},
    {"role": "assistant", "content": "可以使用更多数据、正则化、dropout、早停或减小模型容量。"},
]
```

chat 格式更接近真实聊天模型。

---

### 五、使用 chat_template

Hugging Face tokenizer 支持：

```python
tokenizer.apply_chat_template(...)
```

示例：

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
```

`tokenize=False` 表示先返回字符串。

`add_generation_prompt=False` 表示这是训练样本，最后已经包含 assistant 答案。

推理时通常使用：

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

因为推理时只给到 user，后面要让模型生成 assistant。

注意：

```text
不同模型 chat_template 不同。
不要随便把 A 模型的模板套到 B 模型上。
```

---

### 六、SFT 的 labels 应该是什么

causal LM 训练需要：

```text
input_ids
attention_mask
labels
```

通常：

```text
labels 和 input_ids 一样长。
```

但不一定所有位置都参与 loss。

对于指令微调，我们希望模型学习回答部分。

不希望它花主要精力学习 prompt 模板和用户问题。

所以常见做法是：

```text
prompt 部分 labels 设为 -100。
response 部分 labels 保留真实 token id。
```

PyTorch 的 `CrossEntropyLoss` 会忽略 label 为 `-100` 的位置。

这叫 loss mask。

---

### 七、为什么要 mask prompt loss

假设训练文本是：

```text
### Instruction:
解释什么是过拟合。

### Response:
过拟合是指...
```

如果不 mask prompt，模型也会被训练去预测：

```text
### Instruction:
解释什么是过拟合。
```

这不是我们最关心的目标。

我们真正关心的是：

```text
给定 instruction 后，模型能生成 response。
```

所以更合理的 labels 是：

```text
prompt tokens:   -100 -100 -100 ...
response tokens: 真实 token id
```

这会让 loss 只在回答部分计算。

---

### 八、构造单条 SFT 样本

下面用 Alpaca 风格演示。

```python
def preprocess_example(example, tokenizer, max_length=512):
    prompt, response = format_alpaca(example)
    full_text = prompt + response + tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    full = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    labels = input_ids.copy()
    prompt_len = len(prompt_ids)
    labels[:prompt_len] = [-100] * min(prompt_len, len(labels))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
```

这里的核心是：

```python
labels[:prompt_len] = [-100] * min(prompt_len, len(labels))
```

它把 prompt 部分从 loss 中排除。

---

### 九、处理截断问题

如果 `max_length` 太短，可能把 response 截没了。

例如：

```text
prompt 很长，response 很短。
max_length 截断后只剩 prompt。
labels 全是 -100。
```

这样的样本没有训练价值。

可以过滤掉：

```python
def has_response_labels(example):
    return any(label != -100 for label in example["labels"])
```

或者在预处理时检查：

```python
if all(label == -100 for label in labels):
    return None
```

实际用 Hugging Face `Dataset.map` 时，返回 `None` 不总是方便。

更常见做法是先 map，再 filter。

---

### 十、构造一个小数据集

```python
raw_data = [
    {
        "instruction": "解释什么是过拟合。",
        "input": "",
        "output": "过拟合是指模型在训练集上表现很好，但在未见数据上表现较差的现象。",
    },
    {
        "instruction": "把下面这句话翻译成英文。",
        "input": "我喜欢机器学习。",
        "output": "I like machine learning.",
    },
    {
        "instruction": "给出三个缓解过拟合的方法。",
        "input": "",
        "output": "可以增加数据、使用正则化、加入 dropout、早停或减小模型容量。",
    },
]
```

转成 Hugging Face Dataset：

```python
from datasets import Dataset


dataset = Dataset.from_list(raw_data)
```

预处理：

```python
tokenized_dataset = dataset.map(
    lambda x: preprocess_example(x, tokenizer, max_length=512),
    remove_columns=dataset.column_names,
)
```

过滤无效样本：

```python
tokenized_dataset = tokenized_dataset.filter(
    lambda x: any(label != -100 for label in x["labels"])
)
```

---

### 十一、Data Collator

不同样本长度不同，需要 padding。

SFT 中要同时 pad：

```text
input_ids
attention_mask
labels
```

其中 labels 的 padding 应该是 `-100`，避免 padding 参与 loss。

一个简单 collator：

```python
import torch


class SFTDataCollator:
    def __init__(self, tokenizer, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.label_pad_token_id,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
```

使用：

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

collator = SFTDataCollator(tokenizer)
```

---

### 十二、用 DataLoader 检查 batch

```python
from torch.utils.data import DataLoader


dataloader = DataLoader(
    tokenized_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collator,
)

batch = next(iter(dataloader))

print(batch["input_ids"].shape)
print(batch["attention_mask"].shape)
print(batch["labels"].shape)
```

三者 shape 应该一致：

```text
[batch_size, seq_len]
```

检查 labels：

```python
print(batch["labels"][0])
```

你应该看到 prompt 部分是 `-100`，response 部分是真实 token id。

---

### 十三、解码检查样本

数据预处理后一定要人工检查。

```python
example = tokenized_dataset[0]

print("full text:")
print(tokenizer.decode(example["input_ids"]))

response_ids = [
    token_id for token_id, label in zip(example["input_ids"], example["labels"])
    if label != -100
]

print("response only:")
print(tokenizer.decode(response_ids))
```

你要确认：

```text
full text 包含 instruction 和 response。
response only 只包含答案部分。
```

这一步能发现大部分 labels mask bug。

---

### 十四、数据质量检查

SFT 数据不是越多越好。

低质量数据会直接污染模型行为。

检查维度包括：

```text
instruction 是否清楚。
input 是否缺失或错位。
output 是否回答了问题。
是否有乱码、HTML、无意义内容。
是否有过长样本。
是否有重复样本。
是否有安全风险内容。
是否有答案泄漏或格式混乱。
```

简单去重：

```python
seen = set()
deduped = []

for ex in raw_data:
    key = (ex["instruction"].strip(), ex.get("input", "").strip(), ex["output"].strip())
    if key not in seen:
        seen.add(key)
        deduped.append(ex)
```

过滤空答案：

```python
raw_data = [ex for ex in raw_data if ex["output"].strip()]
```

---

### 十五、训练集和验证集划分

指令数据也需要验证集。

```python
split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
```

如果数据很少，可以用 10% 验证集。

如果数据很多，1% 到 5% 也可以。

注意：

```text
相似或重复样本不要同时出现在 train 和 eval。
```

否则验证 loss 会虚高可信度。

更严谨的做法是按任务、来源或文档分组划分。

---

### 十六、完整预处理脚本骨架

```python
from datasets import Dataset
from transformers import AutoTokenizer


model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def format_alpaca(example):
    if example.get("input", ""):
        prompt = (
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Input:\n"
            f"{example['input']}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Response:\n"
        )
    return prompt, example["output"]


def preprocess_example(example, max_length=512):
    prompt, response = format_alpaca(example)
    full_text = prompt + response + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels = input_ids.copy()

    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


raw_data = [
    {"instruction": "解释什么是过拟合。", "input": "", "output": "过拟合是模型在训练集上表现好但泛化差的现象。"},
    {"instruction": "翻译成英文。", "input": "我喜欢机器学习。", "output": "I like machine learning."},
]

dataset = Dataset.from_list(raw_data)
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = split_dataset["train"].map(
    preprocess_example,
    remove_columns=dataset.column_names,
)
eval_dataset = split_dataset["test"].map(
    preprocess_example,
    remove_columns=dataset.column_names,
)

train_dataset = train_dataset.filter(lambda x: any(label != -100 for label in x["labels"]))
eval_dataset = eval_dataset.filter(lambda x: any(label != -100 for label in x["labels"]))
```

这个脚本的输出可以直接用于下一讲全参数 SFT。

---

### 十七、Chat Template 版本预处理思路

如果模型有 chat template，可以使用更标准的 messages 格式。

```python
def build_messages(example):
    user_content = example["instruction"]
    if example.get("input", ""):
        user_content += "\n" + example["input"]

    return [
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"]},
    ]
```

训练文本：

```python
messages = build_messages(example)
full_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
```

为了 mask prompt，需要构造不含 assistant 内容的 prompt messages：

```python
prompt_messages = messages[:-1]
prompt_text = tokenizer.apply_chat_template(
    prompt_messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

然后按前面的方式计算 `prompt_len` 并 mask。

这是 chat 模型 SFT 中非常常见的处理方式。

---

### 十八、常见工程坑

#### 坑 1：labels 全等于 input_ids

这会让模型同时学习 prompt 和 response。

有些场景可以这么做，但指令微调通常更希望只训练回答部分。

#### 坑 2：labels 全是 -100

这说明 response 被截断掉了，或者 prompt_len 算错了。

这种样本没有 loss。

#### 坑 3：没有 eos token

如果不加 eos，模型可能不知道回答在哪里结束。

#### 坑 4：pad labels 用了 pad_token_id

labels 的 padding 应该用 `-100`，不是 pad token id。

#### 坑 5：chat template 和模型不匹配

这会让模型学到错误格式。

指令模型微调时应使用该模型自己的模板。

#### 坑 6：训练集和验证集有重复样本

验证 loss 会过于乐观。

#### 坑 7：直接相信数据集质量

开源指令数据常有重复、错答、格式混乱和安全问题。

必须抽样检查。

---

### 十九、面试怎么讲 SFT 数据构造

如果面试官问“指令微调数据怎么构造”，可以这样回答：

```text
我会把原始样本整理成 instruction、input、output 或 messages 格式。然后根据目标模型的 prompt template 或 chat template 拼接成训练文本，用 tokenizer 编码成 input_ids 和 attention_mask。labels 通常复制 input_ids，但把 prompt 部分和 padding 部分设为 -100，只在 assistant response 部分计算 loss。最后做长度截断、过滤无效样本、去重和 train/eval 划分。
```

如果追问“为什么 prompt 部分 labels 要设为 -100”，可以回答：

```text
因为 SFT 的目标是让模型在给定用户指令后学习生成助手回答，而不是学习复述用户问题或模板。把 prompt 部分设为 -100 后，CrossEntropyLoss 会忽略这些位置，只在 response token 上计算 loss。
```

如果问“chat template 为什么重要”，可以回答：

```text
不同聊天模型在预训练或指令微调时使用的 system/user/assistant 标记不同。微调时如果模板不匹配，模型会学到和原始对话格式不一致的分布，影响推理效果。因此应优先使用 tokenizer 自带的 apply_chat_template。
```

如果问“SFT 数据质量怎么检查”，可以回答：

```text
我会检查 instruction 是否清楚、output 是否回答问题、是否有空答案、重复样本、过长样本、乱码和安全风险；还会 decode tokenized 样本，确认 prompt 和 response 拼接正确，labels 中只有 response 部分参与 loss。
```

---

### 二十、小练习

#### 练习 1

构造 10 条 `instruction/input/output` 样本，并转成 Hugging Face Dataset。

#### 练习 2

实现 `format_alpaca`，打印拼接后的 prompt 和 response。

#### 练习 3

实现 `preprocess_example`，确认 prompt 部分 labels 是 `-100`。

#### 练习 4

写一个 collator，确认 labels 的 padding 是 `-100`。

#### 练习 5

用 `tokenizer.apply_chat_template` 重写预处理逻辑。

---

### 本讲总结

这一讲构造了指令微调数据集。

核心结论如下：

1. SFT 数据通常包含 instruction、input、output，或 chat messages。
2. 模型训练看到的是 prompt 和 response 拼接后的 token 序列。
3. labels 通常复制 input_ids，但 prompt 和 padding 部分应设为 `-100`。
4. `-100` 会被 CrossEntropyLoss 忽略。
5. chat 模型应优先使用目标 tokenizer 的 chat template。
6. 数据预处理后必须 decode 抽查，确认格式和 labels mask 正确。
7. 数据质量直接决定 SFT 效果，清洗、去重和验证集划分很重要。

下一讲，我们使用这个数据集对小模型做全参数 SFT。

## 第 21 讲：全参数 SFT 小模型

### 本讲目标

学完本讲，你应该能做到六件事：

1. 理解全参数 SFT 和参数高效微调的区别。
2. 使用 Hugging Face `Trainer` 对小模型做 SFT。
3. 写出全参数 SFT 的核心训练配置。
4. 理解显存占用、batch size、gradient accumulation 的关系。
5. 保存和加载 SFT 后的模型。
6. 能排查全参数微调中的常见问题。

上一讲我们构造了指令微调数据集。

本讲把数据喂给模型，做一次全参数 SFT。

所谓全参数 SFT，就是：

```text
模型所有可训练参数都参与梯度更新。
```

这和后面要讲的 LoRA 不同。

LoRA 只训练少量低秩适配器参数。

全参数 SFT 更直接，但显存和存储成本更高。

所以本讲只建议对小模型做实验。

---

### 一、什么是全参数 SFT

SFT 是 supervised fine-tuning。

目标是让模型学习指令数据中的回答风格和任务能力。

全参数 SFT 表示：

```text
embedding、attention、MLP、norm、lm_head 等所有参数都更新。
```

优点：

```text
表达能力强。
模型可以充分适配新数据。
实现直接，不需要额外适配器结构。
```

缺点：

```text
显存占用大。
训练成本高。
容易灾难性遗忘。
每个任务都要保存一份完整模型。
```

对于几百 MB 到几 GB 的小模型，可以尝试全参数 SFT。

对于 7B、14B、70B 模型，通常更常用 LoRA、QLoRA 或其他参数高效方法。

---

### 二、准备依赖

```bash
pip install torch transformers datasets accelerate
```

本讲使用：

```text
AutoTokenizer
AutoModelForCausalLM
Trainer
TrainingArguments
Dataset
```

如果 GPU 显存有限，建议先用非常小的模型跑通流程。

例如：

```text
sshleifer/tiny-gpt2
distilgpt2
```

真实中文 SFT 可以换成更合适的中文或多语言小模型。

---

### 三、整体流程

全参数 SFT 流程如下：

```text
1. 加载 tokenizer。
2. 加载 causal LM。
3. 构造 instruction 数据。
4. tokenize，并构造 labels mask。
5. 构造 data collator。
6. 配置 TrainingArguments。
7. 创建 Trainer。
8. trainer.train()。
9. 保存模型和 tokenizer。
10. 加载微调后模型做推理测试。
```

这条流程是 Hugging Face 微调的基础。

后面的 LoRA 和 QLoRA 也是在这条流程上改造。

---

### 四、加载 tokenizer 和模型

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "sshleifer/tiny-gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id
```

为什么设置 `pad_token`？

因为训练 batch 中不同样本长度不同，需要 padding。

decoder-only 模型常常没有默认 pad token。

小实验中可以用 eos token 兼作 pad token。

---

### 五、准备小型 SFT 数据

```python
raw_data = [
    {
        "instruction": "解释什么是过拟合。",
        "input": "",
        "output": "过拟合是指模型在训练集上表现很好，但在未见数据上表现较差的现象。",
    },
    {
        "instruction": "给出三个缓解过拟合的方法。",
        "input": "",
        "output": "可以增加数据、使用正则化、加入 dropout、早停或减小模型容量。",
    },
    {
        "instruction": "把下面这句话翻译成英文。",
        "input": "我喜欢机器学习。",
        "output": "I like machine learning.",
    },
    {
        "instruction": "解释什么是梯度下降。",
        "input": "",
        "output": "梯度下降是一种通过沿负梯度方向更新参数来最小化损失函数的优化方法。",
    },
]
```

转成 Dataset：

```python
from datasets import Dataset


dataset = Dataset.from_list(raw_data)
split_dataset = dataset.train_test_split(test_size=0.25, seed=42)
train_raw = split_dataset["train"]
eval_raw = split_dataset["test"]
```

真实项目中，数据至少应该有几千到几十万条。

这里的小数据只用于跑通流程。

---

### 六、格式化 prompt

```python
def format_alpaca(example):
    if example.get("input", ""):
        prompt = (
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Input:\n"
            f"{example['input']}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Response:\n"
        )
    return prompt, example["output"]
```

这和上一讲保持一致。

后续如果使用 chat 模型，建议改成 `apply_chat_template`。

---

### 七、tokenize 并构造 labels

```python
max_length = 512


def preprocess_example(example):
    prompt, response = format_alpaca(example)
    full_text = prompt + response + tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    full = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels = input_ids.copy()

    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
```

预处理数据集：

```python
train_dataset = train_raw.map(
    preprocess_example,
    remove_columns=train_raw.column_names,
)
eval_dataset = eval_raw.map(
    preprocess_example,
    remove_columns=eval_raw.column_names,
)

train_dataset = train_dataset.filter(lambda x: any(label != -100 for label in x["labels"]))
eval_dataset = eval_dataset.filter(lambda x: any(label != -100 for label in x["labels"]))
```

---

### 八、Data Collator

```python
class SFTDataCollator:
    def __init__(self, tokenizer, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.label_pad_token_id,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


data_collator = SFTDataCollator(tokenizer)
```

这里最重要的是：

```text
labels padding 用 -100。
```

否则 padding token 会参与 loss。

---

### 九、检查一个 batch

在训练前先检查 batch。

```python
from torch.utils.data import DataLoader


loader = DataLoader(
    train_dataset,
    batch_size=2,
    collate_fn=data_collator,
)

batch = next(iter(loader))
print(batch["input_ids"].shape)
print(batch["attention_mask"].shape)
print(batch["labels"].shape)
print(batch["labels"][0])
```

确认三件事：

```text
input_ids、attention_mask、labels shape 一致。
prompt 部分 label 是 -100。
response 部分 label 是真实 token id。
```

不要跳过这一步。

SFT 训练异常，很多时候就是 labels 构造错了。

---

### 十、TrainingArguments

```python
from transformers import TrainingArguments


training_args = TrainingArguments(
    output_dir="outputs/full_sft_tiny_gpt2",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=20,
    save_steps=20,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
)
```

如果你的 transformers 版本较旧，参数名可能是：

```python
evaluation_strategy="steps"
```

而不是：

```python
eval_strategy="steps"
```

根据本地版本调整即可。

---

### 十一、gradient accumulation 是什么

如果显存只能放下 batch size 2，但你想要等效 batch size 8，可以设置：

```text
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
```

等效 batch size 约为：

```text
2 * 4 = 8
```

如果是多卡，还要乘以 GPU 数：

```text
effective_batch_size = per_device_batch_size * gradient_accumulation_steps * num_gpus
```

它的作用是：

```text
分多次 forward/backward 累积梯度，再执行一次 optimizer step。
```

这是显存不足时非常常用的技巧。

---

### 十二、创建 Trainer 并训练

```python
from transformers import Trainer


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
```

训练结束后保存：

```python
trainer.save_model("outputs/full_sft_tiny_gpt2/final")
tokenizer.save_pretrained("outputs/full_sft_tiny_gpt2/final")
```

保存目录会包含：

```text
model weights
config.json
tokenizer files
generation config
```

推理时可以直接从这个目录加载。

---

### 十三、完整训练脚本骨架

```python
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


model_name = "sshleifer/tiny-gpt2"
output_dir = "outputs/full_sft_tiny_gpt2"
max_length = 512

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id

# raw_data = [...]
# dataset split
# preprocess_example
# SFTDataCollator

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=20,
    save_steps=20,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(f"{output_dir}/final")
tokenizer.save_pretrained(f"{output_dir}/final")
```

实际使用时，把前面数据构造代码补完整即可。

---

### 十四、加载 SFT 后模型推理

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


sft_dir = "outputs/full_sft_tiny_gpt2/final"

tokenizer = AutoTokenizer.from_pretrained(sft_dir)
model = AutoModelForCausalLM.from_pretrained(sft_dir)
model.eval()

prompt = (
    "### Instruction:\n"
    "解释什么是过拟合。\n\n"
    "### Response:\n"
)

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
```

注意：

```text
推理 prompt 格式要和训练时一致。
```

如果训练时用 Alpaca 格式，推理也要用 Alpaca 格式。

如果训练时用 chat template，推理也要用 chat template。

---

### 十五、全参数 SFT 的显存开销

全参数训练需要保存：

```text
模型参数
梯度
优化器状态
激活值
```

AdamW 优化器通常还要保存一阶和二阶动量。

粗略理解：

```text
训练显存远大于推理显存。
```

同一个模型，推理能跑，不代表全参数 SFT 能跑。

如果 OOM，可以尝试：

```text
减小 per_device_train_batch_size。
增大 gradient_accumulation_steps。
减小 max_length。
使用 fp16/bf16。
开启 gradient checkpointing。
换更小模型。
改用 LoRA 或 QLoRA。
```

开启 gradient checkpointing：

```python
model.gradient_checkpointing_enable()
```

它会用更多计算换更低激活显存。

---

### 十六、手写训练循环版本

虽然 `Trainer` 很方便，但你也应该知道底层逻辑。

```python
from torch.utils.data import DataLoader
from torch.optim import AdamW


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()

loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=data_collator,
)

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"epoch {epoch}, step {step}, loss {loss.item():.4f}")
```

`Trainer` 内部做的事情，本质也是：

```text
forward
loss
backward
optimizer step
logging
evaluation
checkpoint
```

理解手写循环有助于排查问题。

---

### 十七、如何确认参数都在训练

全参数 SFT 中，所有参数默认 `requires_grad=True`。

可以检查：

```python
total_params = 0
trainable_params = 0

for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()

print("total params:", total_params)
print("trainable params:", trainable_params)
print("trainable ratio:", trainable_params / total_params)
```

全参数 SFT 中比例应该接近：

```text
1.0
```

后面 LoRA 中，这个比例会非常小。

---

### 十八、常见工程坑

#### 坑 1：数据 labels 全是 -100

训练 loss 会异常，模型学不到东西。

训练前必须检查 batch。

#### 坑 2：pad token 没设置

decoder-only 模型常没有 pad token。

需要设置：

```python
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
```

#### 坑 3：推理模板和训练模板不一致

训练用 Alpaca，推理却用裸问题，效果会变差。

#### 坑 4：学习率过大

全参数 SFT 更新所有参数，学习率过大容易破坏模型原有能力。

常见起点是：

```text
1e-5 到 5e-5
```

#### 坑 5：小数据训练太久

容易过拟合和灾难性遗忘。

#### 坑 6：以为能推理就能训练

训练需要梯度、优化器状态和激活值，显存远高于推理。

---

### 十九、面试怎么讲全参数 SFT

如果面试官问“全参数 SFT 怎么做”，可以这样回答：

```text
我会先加载 tokenizer 和 AutoModelForCausalLM，然后把 instruction 数据按目标模型的 prompt 或 chat template 拼接，tokenize 成 input_ids、attention_mask 和 labels。labels 中 prompt 和 padding 部分设为 -100，只在 assistant response 上计算 loss。训练时所有模型参数 requires_grad=True，用 AdamW 或 Trainer 做监督微调，并根据验证集 loss 保存 checkpoint。
```

如果追问“全参数 SFT 和 LoRA 的区别”，可以回答：

```text
全参数 SFT 会更新模型所有参数，适配能力强，但显存、存储和训练成本高，也更容易灾难性遗忘；LoRA 冻结基座模型，只训练低秩适配器参数，成本更低，适合大模型微调。
```

如果问“为什么全参数训练显存比推理大很多”，可以回答：

```text
推理主要保存模型参数和少量 KV cache；训练除了参数，还要保存梯度、优化器状态以及反向传播需要的激活值。AdamW 还会保存一阶和二阶动量，所以训练显存远高于推理。
```

如果问“SFT 学习率怎么选”，可以回答：

```text
全参数 SFT 一般使用较小学习率，比如 1e-5 到 5e-5 起步。学习率太大容易破坏预训练能力，太小则适配慢。具体要结合数据规模、模型大小和验证集表现调参。
```

---

### 二十、小练习

#### 练习 1

用 `sshleifer/tiny-gpt2` 跑通全参数 SFT。

#### 练习 2

训练前打印一个 batch，确认 labels mask 正确。

#### 练习 3

打印 total params 和 trainable params，确认比例接近 1。

#### 练习 4

把 `learning_rate` 从 `5e-5` 改成 `5e-4`，观察 loss 是否更不稳定。

#### 练习 5

训练后用同样 prompt 格式测试微调前后模型输出差异。

---

### 本讲总结

这一讲完成了小模型的全参数 SFT。

核心结论如下：

1. 全参数 SFT 会更新模型所有参数。
2. SFT 数据必须正确构造 `input_ids`、`attention_mask` 和 `labels`。
3. prompt 和 padding 部分的 labels 通常设为 `-100`。
4. Hugging Face `Trainer` 可以快速完成训练、评估和保存。
5. 全参数训练显存远高于推理显存。
6. 小数据全参数训练容易过拟合和遗忘。
7. 推理时 prompt 模板必须和训练时一致。

下一讲，我们实现 LoRA 微调，用更少参数完成指令适配。

## 第 22 讲：LoRA 微调

### 本讲目标

学完本讲，你应该能做到六件事：

1. 理解 LoRA 为什么能降低微调成本。
2. 使用 PEFT 给 causal LM 注入 LoRA adapter。
3. 配置 LoRA 的 `r`、`alpha`、`dropout` 和 `target_modules`。
4. 使用 Hugging Face `Trainer` 训练 LoRA。
5. 保存、加载和合并 LoRA adapter。
6. 能解释 LoRA 和全参数 SFT 的区别、优缺点和适用场景。

上一讲我们做了全参数 SFT。

全参数 SFT 很直接，但成本高。

对于大模型，更新全部参数通常不现实。

LoRA 是最常用的参数高效微调方法之一。

它的核心思想是：

```text
冻结原始模型参数，只训练少量低秩矩阵，让模型学到任务增量。
```

本讲在上一讲 SFT 流程基础上，把全参数微调改造成 LoRA 微调。

---

### 一、LoRA 解决什么问题

大模型参数量很大。

如果全参数训练，需要保存：

```text
参数
梯度
优化器状态
激活值
```

显存成本非常高。

LoRA 的目标是：

```text
尽量不动原始大模型参数，只训练很少的新增参数。
```

这样带来几个好处：

```text
显存更低。
训练更快。
保存文件更小。
多个任务可以保存多个 adapter。
更不容易破坏原始模型能力。
```

代价是：

```text
适配能力可能弱于全参数 SFT。
需要选择 target_modules。
推理时要加载 adapter 或合并权重。
```

---

### 二、LoRA 的核心公式

一个线性层原本是：

```text
y = W x
```

LoRA 冻结原始 `W`，增加一个低秩增量：

```text
y = W x + ΔW x
```

其中：

```text
ΔW = B A
```

如果 `W` 是 `[out_dim, in_dim]`，则：

```text
A: [r, in_dim]
B: [out_dim, r]
```

`r` 是低秩维度，通常远小于 `in_dim` 和 `out_dim`。

所以 LoRA 训练参数量是：

```text
r * in_dim + out_dim * r
```

而不是：

```text
out_dim * in_dim
```

这就是省参数的来源。

---

### 三、LoRA 常见超参数

#### 1. `r`

低秩维度。

常见值：

```text
4, 8, 16, 32, 64
```

`r` 越大，可训练参数越多，表达能力越强，但显存和过拟合风险也更高。

#### 2. `lora_alpha`

缩放系数。

LoRA 输出通常会乘：

```text
lora_alpha / r
```

常见设置：

```text
lora_alpha = 2r 或 4r
```

#### 3. `lora_dropout`

LoRA 分支上的 dropout。

小数据微调时可以用：

```text
0.05 或 0.1
```

#### 4. `target_modules`

指定哪些线性层注入 LoRA。

常见目标是 attention 投影层：

```text
q_proj, k_proj, v_proj, o_proj
```

有些模型命名不同，例如 GPT-2 使用：

```text
c_attn, c_proj
```

这是 LoRA 实战里最容易踩坑的地方。

---

### 四、安装依赖

```bash
pip install peft transformers datasets accelerate
```

PEFT 是 Hugging Face 的 Parameter-Efficient Fine-Tuning 库。

LoRA、Prefix Tuning、Prompt Tuning 等都在这个库里。

本讲主要使用：

```python
from peft import LoraConfig, get_peft_model, TaskType
```

---

### 五、加载模型和 tokenizer

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "sshleifer/tiny-gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id
```

这里仍然用小模型演示。

如果换成 LLaMA/Qwen/Mistral 类模型，LoRA 配置里的 `target_modules` 需要相应调整。

---

### 六、查看模型模块名

在配置 `target_modules` 前，先打印模型模块名。

```python
for name, module in model.named_modules():
    if "attn" in name or "proj" in name or "c_" in name:
        print(name)
```

对 GPT-2 类模型，常见模块名包括：

```text
transformer.h.0.attn.c_attn
transformer.h.0.attn.c_proj
transformer.h.0.mlp.c_fc
transformer.h.0.mlp.c_proj
```

对 LLaMA/Qwen 类模型，常见模块名包括：

```text
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
mlp.gate_proj
mlp.up_proj
mlp.down_proj
```

所以不要盲目复制别人的 `target_modules`。

要先看当前模型的模块命名。

---

### 七、配置 LoRA

对 GPT-2 tiny 模型，可以先用：

```python
from peft import LoraConfig, TaskType, get_peft_model


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

`print_trainable_parameters()` 会输出类似：

```text
trainable params: 1,024 || all params: 102,714 || trainable%: 0.99
```

实际数字取决于模型。

你应该看到：

```text
可训练参数比例远小于 100%。
```

这就是 LoRA 和全参数 SFT 的直接区别。

---

### 八、LLaMA/Qwen 类模型的 target_modules

如果使用 LLaMA、Qwen、Mistral 类架构，常见配置是：

```python
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
```

更轻量的配置可以只训练 attention：

```python
target_modules = ["q_proj", "v_proj"]
```

经验：

```text
只训 q/v：参数少，成本低。
训 q/k/v/o：attention 适配更充分。
再加 MLP：表达能力更强，但参数更多。
```

具体选择取决于任务、数据量和显存。

---

### 九、复用上一讲 SFT 数据

LoRA 微调的数据处理和全参数 SFT 一样。

仍然需要：

```text
input_ids
attention_mask
labels
```

仍然要：

```text
prompt labels = -100
padding labels = -100
response labels = token ids
```

也就是说，LoRA 只改变“训练哪些参数”。

它不改变 SFT 数据目标。

你可以直接复用第 20 讲和第 21 讲的数据预处理代码。

---

### 十、TrainingArguments

LoRA 通常可以使用比全参数 SFT 稍大的学习率。

例如：

```python
from transformers import TrainingArguments


training_args = TrainingArguments(
    output_dir="outputs/lora_tiny_gpt2",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.0,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=20,
    save_steps=20,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
)
```

如果 transformers 版本较旧，使用：

```python
evaluation_strategy="steps"
```

---

### 十一、创建 Trainer 并训练

```python
from transformers import Trainer


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
```

保存 LoRA adapter：

```python
model.save_pretrained("outputs/lora_tiny_gpt2/adapter")
tokenizer.save_pretrained("outputs/lora_tiny_gpt2/adapter")
```

这里保存的通常不是完整模型权重。

而是 LoRA adapter 权重和配置。

这也是 LoRA 文件小的原因。

---

### 十二、完整 LoRA 训练脚本骨架

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model


model_name = "sshleifer/tiny-gpt2"
output_dir = "outputs/lora_tiny_gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# train_dataset, eval_dataset, data_collator 复用上一讲

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=20,
    save_steps=20,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained(f"{output_dir}/adapter")
tokenizer.save_pretrained(f"{output_dir}/adapter")
```

---

### 十三、加载 LoRA adapter 做推理

推理时需要先加载 base model，再加载 adapter。

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


base_model_name = "sshleifer/tiny-gpt2"
adapter_dir = "outputs/lora_tiny_gpt2/adapter"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()
```

然后正常 generate：

```python
prompt = (
    "### Instruction:\n"
    "解释什么是过拟合。\n\n"
    "### Response:\n"
)

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
```

注意：

```text
adapter 必须配合训练时的 base model 使用。
```

不要把 A 模型训练出来的 LoRA adapter 加到 B 模型上。

---

### 十四、合并 LoRA 权重

如果想把 LoRA 合并进 base model，方便部署，可以使用：

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("outputs/lora_tiny_gpt2/merged")
tokenizer.save_pretrained("outputs/lora_tiny_gpt2/merged")
```

合并后得到的是普通 causal LM。

推理时不再需要 PEFT adapter。

加载方式：

```python
model = AutoModelForCausalLM.from_pretrained("outputs/lora_tiny_gpt2/merged")
```

合并的优点：

```text
部署简单。
推理链路更普通。
```

合并的缺点：

```text
失去 adapter 灵活切换能力。
每个任务要保存一份合并后的模型。
```

---

### 十五、LoRA 参数量估算

假设某个线性层：

```text
W: [4096, 4096]
```

全参数训练参数量：

```text
4096 * 4096 ≈ 16.8M
```

LoRA 设置 `r=8`：

```text
A: [8, 4096]
B: [4096, 8]
参数量: 8*4096 + 4096*8 = 65,536
```

相比 16.8M，少很多。

这就是为什么 LoRA 适合大模型微调。

---

### 十六、LoRA 和全参数 SFT 对比

```text
全参数 SFT：
更新全部参数。
适配能力强。
显存和存储成本高。
更容易遗忘。

LoRA：
冻结基座模型。
只训练低秩 adapter。
成本低，文件小。
适合多任务和大模型。
适配能力可能略弱于全参数。
```

实际工作中：

```text
小模型、强适配、充足资源：可以尝试全参数 SFT。
大模型、资源有限、多任务适配：优先 LoRA/QLoRA。
```

---

### 十七、常见工程坑

#### 坑 1：target_modules 写错

如果模块名不匹配，LoRA 可能没有注入成功，或者报错。

训练前一定要：

```python
model.print_trainable_parameters()
```

确认可训练参数不是 0。

#### 坑 2：把 adapter 当完整模型加载

LoRA adapter 不是完整模型。

加载时需要：

```python
base_model + adapter
```

或者先 merge 后再按普通模型加载。

#### 坑 3：base model 不一致

adapter 必须和训练时的 base model 匹配。

#### 坑 4：学习率照搬全参数 SFT

LoRA 通常可以用更高学习率，比如 `1e-4` 到 `3e-4`。

但仍要根据数据和 loss 调整。

#### 坑 5：以为 LoRA 不会过拟合

LoRA 参数少，但小数据上仍然会过拟合。

#### 坑 6：保存时忘记 tokenizer

adapter 推理仍需要 tokenizer。

应该一起保存。

---

### 十八、面试怎么讲 LoRA

如果面试官问“LoRA 是什么”，可以这样回答：

```text
LoRA 是一种参数高效微调方法。它冻结原始模型权重，在部分线性层旁边增加低秩矩阵分支，用 BA 近似权重增量 ΔW。训练时只更新这些低秩矩阵，从而大幅减少可训练参数和优化器状态，降低显存和存储成本。
```

如果追问“LoRA 为什么省参数”，可以回答：

```text
原始线性层权重 W 的参数量是 out_dim * in_dim。LoRA 用两个低秩矩阵 A 和 B 表示增量，参数量是 r * in_dim + out_dim * r。当 r 远小于输入输出维度时，参数量会小很多。
```

如果问“LoRA 一般加在哪些层”，可以回答：

```text
常见做法是在 attention 的 q_proj、k_proj、v_proj、o_proj 上加 LoRA，也可以加到 MLP 的 gate_proj、up_proj、down_proj。更轻量的配置只加 q_proj 和 v_proj。具体 target_modules 要根据模型结构命名确认。
```

如果问“LoRA 怎么部署”，可以回答：

```text
可以推理时加载 base model 再加载 LoRA adapter，也可以把 adapter merge 到 base model 中保存成普通模型。前者方便多任务切换，后者部署链路更简单。
```

---

### 十九、小练习

#### 练习 1

打印模型模块名，找到适合注入 LoRA 的 target modules。

#### 练习 2

在 `sshleifer/tiny-gpt2` 上跑通 LoRA SFT。

#### 练习 3

比较全参数 SFT 和 LoRA 的 trainable parameter ratio。

#### 练习 4

把 `r` 从 8 改成 16，观察可训练参数量变化。

#### 练习 5

保存 adapter 后，重新加载 base model + adapter 做推理。

---

### 本讲总结

这一讲实现了 LoRA 微调。

核心结论如下：

1. LoRA 冻结原始模型，只训练低秩 adapter。
2. LoRA 用 `ΔW = BA` 表示权重增量。
3. `r` 越大，可训练参数越多，适配能力越强。
4. `target_modules` 必须和模型真实模块名匹配。
5. LoRA 训练数据和普通 SFT 一样，仍然需要正确 labels mask。
6. LoRA adapter 不是完整模型，推理时要配合 base model 或先 merge。
7. LoRA 适合资源有限的大模型任务适配。

下一讲，我们实现 QLoRA，在 4bit 量化基座模型上训练 LoRA adapter。

## 第 23 讲：QLoRA 微调

### 本讲目标

学完本讲，你应该能做到六件事：

1. 理解 QLoRA 和 LoRA 的区别。
2. 使用 bitsandbytes 以 4bit 方式加载基座模型。
3. 配置 `BitsAndBytesConfig`。
4. 使用 `prepare_model_for_kbit_training` 准备量化模型训练。
5. 在 4bit 量化模型上训练 LoRA adapter。
6. 能解释 QLoRA 的显存优势、限制和常见工程坑。

上一讲我们实现了 LoRA。

LoRA 冻结基座模型，只训练少量 adapter。

但基座模型本身仍然要以 fp16、bf16 或 fp32 形式加载。

对于 7B、14B 甚至更大的模型，光加载基座模型就可能占满显存。

QLoRA 的核心思路是：

```text
把冻结的基座模型用 4bit 量化加载，只训练 LoRA adapter。
```

这样能进一步降低显存，让单卡微调更大的模型成为可能。

---

### 一、QLoRA 解决什么问题

LoRA 已经减少了可训练参数。

但 LoRA 中冻结的 base model 仍然要占显存。

例如一个 7B 模型：

```text
fp16 参数约 14GB。
```

训练时还需要激活值、adapter、优化器状态等。

显存压力仍然很大。

QLoRA 进一步把 base model 量化到 4bit。

粗略理解：

```text
fp16 base model -> 4bit base model
LoRA adapter 保持可训练
```

这样可以显著降低基座模型显存占用。

---

### 二、QLoRA 和 LoRA 的区别

LoRA：

```text
base model 通常 fp16/bf16 加载。
base model 冻结。
训练 LoRA adapter。
```

QLoRA：

```text
base model 以 4bit 量化加载。
base model 冻结。
训练 LoRA adapter。
计算时使用合适的 compute dtype。
```

共同点：

```text
都只训练 adapter。
都需要 target_modules。
都可以保存 adapter。
```

关键区别：

```text
QLoRA 的 base model 是量化加载的，显存更低。
```

---

### 三、安装依赖

```bash
pip install transformers peft accelerate datasets bitsandbytes
```

`bitsandbytes` 用于 8bit/4bit 量化加载。

注意：

```text
bitsandbytes 对 CUDA 环境比较敏感。
CPU 或不兼容 CUDA 环境可能无法正常使用 4bit。
```

如果本地环境不支持，可以先阅读代码逻辑，实际运行时换到支持 CUDA 的机器。

---

### 四、BitsAndBytesConfig

4bit 加载通常使用：

```python
from transformers import BitsAndBytesConfig
import torch


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

参数解释：

```text
load_in_4bit=True：以 4bit 方式加载模型权重。
bnb_4bit_quant_type="nf4"：使用 NormalFloat4，QLoRA 常用量化类型。
bnb_4bit_compute_dtype：计算时使用的 dtype，常用 bf16 或 fp16。
bnb_4bit_use_double_quant=True：使用双重量化，进一步节省显存。
```

如果 GPU 不支持 bf16，可以改成：

```python
bnb_4bit_compute_dtype=torch.float16
```

---

### 五、加载 4bit 基座模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.pad_token_id = tokenizer.pad_token_id
```

这里用 `device_map="auto"`。

原因是量化模型加载通常依赖 accelerate 自动放置。

如果你使用的是 tiny-gpt2，可能没有必要用 QLoRA。

QLoRA 的价值主要体现在较大模型上。

---

### 六、准备 k-bit training

PEFT 提供了：

```python
prepare_model_for_kbit_training
```

用法：

```python
from peft import prepare_model_for_kbit_training


model = prepare_model_for_kbit_training(model)
```

它会做一些适合 k-bit 训练的准备，例如：

```text
处理 norm 层精度。
确保输入梯度相关设置正确。
为量化模型训练做兼容处理。
```

实际工程中，QLoRA 通常写成：

```python
model = prepare_model_for_kbit_training(model)
```

然后再注入 LoRA。

---

### 七、配置 LoRA adapter

```python
from peft import LoraConfig, TaskType, get_peft_model


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

对于 Qwen/LLaMA/Mistral 类模型，这组 target modules 很常见。

但仍然建议先打印模块名确认。

```python
for name, module in model.named_modules():
    if any(key in name for key in ["q_proj", "v_proj", "gate_proj", "up_proj"]):
        print(name)
```

---

### 八、QLoRA 数据处理

QLoRA 的数据处理和 SFT/LoRA 一样。

仍然使用：

```text
input_ids
attention_mask
labels
```

仍然需要：

```text
prompt labels = -100
padding labels = -100
response labels = token ids
```

也就是说：

```text
QLoRA 改变的是模型加载和训练参数方式，不改变 SFT 数据目标。
```

第 20 讲的数据预处理代码可以直接复用。

---

### 九、TrainingArguments

```python
from transformers import TrainingArguments


training_args = TrainingArguments(
    output_dir="outputs/qlora_qwen_0_5b",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    weight_decay=0.0,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,
    bf16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    report_to="none",
)
```

如果 GPU 不支持 bf16，可以用：

```python
fp16=True
```

不要同时随意开启 fp16 和 bf16。

根据硬件选择一个。

---

### 十、创建 Trainer 并训练

```python
from transformers import Trainer


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
```

保存 adapter：

```python
model.save_pretrained("outputs/qlora_qwen_0_5b/adapter")
tokenizer.save_pretrained("outputs/qlora_qwen_0_5b/adapter")
```

和 LoRA 一样，保存的主要是 adapter。

base model 仍然来自原模型。

---

### 十一、完整 QLoRA 脚本骨架

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


model_name = "Qwen/Qwen2.5-0.5B"
output_dir = "outputs/qlora_qwen_0_5b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.pad_token_id = tokenizer.pad_token_id

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# train_dataset, eval_dataset, data_collator 复用 SFT 数据处理代码

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,
    bf16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained(f"{output_dir}/adapter")
tokenizer.save_pretrained(f"{output_dir}/adapter")
```

这个脚本是典型 QLoRA SFT 骨架。

---

### 十二、加载 QLoRA adapter 推理

推理时同样需要 base model + adapter。

如果想保持 4bit 加载：

```python
from peft import PeftModel


base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(
    base_model,
    "outputs/qlora_qwen_0_5b/adapter",
)
model.eval()
```

然后正常 generate。

如果要部署，也可以考虑合并到高精度 base model。

但注意：

```text
4bit 量化模型上 merge_and_unload 的行为和可用性会受库版本与模型影响。
```

工程中常见做法是：

```text
训练时用 QLoRA。
部署时根据需求选择 adapter 加载或合并到 fp16/bf16 模型。
```

---

### 十三、QLoRA 显存为什么更低

QLoRA 降显存主要来自三点：

```text
1. base model 以 4bit 存储。
2. base model 冻结，不保存其优化器状态。
3. 只为 LoRA adapter 保存梯度和优化器状态。
```

相比全参数 SFT：

```text
不需要为全部参数保存梯度和 AdamW 状态。
```

相比普通 LoRA：

```text
base model 权重本身更省显存。
```

所以 QLoRA 是单卡微调大模型时非常常用的方法。

---

### 十四、QLoRA 的限制

QLoRA 不是免费午餐。

常见限制：

```text
依赖 bitsandbytes 和 CUDA 环境。
4bit 量化会带来一定数值误差。
训练速度不一定比 LoRA 更快。
某些模型结构或算子兼容性可能有问题。
部署链路比全参数模型复杂。
```

实践中要关注：

```text
loss 是否正常下降。
生成质量是否稳定。
是否出现 NaN。
adapter 是否能正常加载。
```

---

### 十五、LoRA、QLoRA、全参数 SFT 对比

```text
全参数 SFT：
base model 高精度加载，全部参数训练。
成本最高，适配能力强。

LoRA：
base model 高精度加载，冻结 base，只训练 adapter。
成本较低，文件小。

QLoRA：
base model 4bit 加载，冻结 base，只训练 adapter。
显存更低，适合单卡微调较大模型。
```

选择建议：

```text
小模型 + 资源充足：全参数 SFT。
中大模型 + 有一定显存：LoRA。
大模型 + 显存紧张：QLoRA。
```

---

### 十六、常见工程坑

#### 坑 1：bitsandbytes 安装或 CUDA 不匹配

表现为导入失败或 4bit 加载失败。

解决方式是检查 CUDA、PyTorch 和 bitsandbytes 版本兼容性。

#### 坑 2：忘记 `prepare_model_for_kbit_training`

量化模型训练前通常需要调用它。

#### 坑 3：target_modules 不匹配

和 LoRA 一样，必须根据模型真实模块名配置。

#### 坑 4：bf16 硬件不支持

如果 GPU 不支持 bf16，改用 fp16 compute dtype。

#### 坑 5：以为 4bit 模型所有计算都是 4bit

4bit 主要是权重存储，计算会使用指定 compute dtype，例如 fp16/bf16。

#### 坑 6：adapter 和 base model 不匹配

QLoRA adapter 仍然必须配合训练时的 base model。

#### 坑 7：小模型上看不出 QLoRA 优势

QLoRA 主要为大模型省显存。

在 tiny 模型上只是演示流程。

---

### 十七、面试怎么讲 QLoRA

如果面试官问“QLoRA 是什么”，可以这样回答：

```text
QLoRA 是 LoRA 的量化版本。它把冻结的基座模型以 4bit 量化方式加载，同时只训练 LoRA adapter。这样既保留了 LoRA 只训练少量参数的优点，又显著降低了 base model 的显存占用，适合资源受限场景下微调较大模型。
```

如果追问“QLoRA 和 LoRA 区别是什么”，可以回答：

```text
LoRA 通常以 fp16 或 bf16 加载基座模型，然后冻结基座并训练 adapter；QLoRA 则以 4bit 方式加载基座模型，再训练 adapter。二者训练的参数都是 LoRA adapter，但 QLoRA 的基座权重存储更省显存。
```

如果问“QLoRA 需要哪些关键配置”，可以回答：

```text
关键配置包括 BitsAndBytesConfig 中的 load_in_4bit、量化类型 nf4、compute dtype、double quant；加载模型时传 quantization_config 和 device_map；然后用 prepare_model_for_kbit_training 准备模型，再通过 get_peft_model 注入 LoRA adapter。
```

如果问“QLoRA 有什么风险”，可以回答：

```text
主要风险包括 bitsandbytes 和 CUDA 兼容性、4bit 量化带来的数值误差、某些模型结构不兼容、训练或合并 adapter 链路复杂，以及 adapter 必须和 base model 严格匹配。
```

---

### 十八、小练习

#### 练习 1

检查本机是否能成功导入 bitsandbytes。

#### 练习 2

用 `BitsAndBytesConfig` 以 4bit 加载一个小模型。

#### 练习 3

调用 `prepare_model_for_kbit_training` 后注入 LoRA。

#### 练习 4

比较 LoRA 和 QLoRA 加载同一模型时的显存占用。

#### 练习 5

保存 QLoRA adapter，并重新加载 base model + adapter 做推理。

---

### 本讲总结

这一讲实现了 QLoRA 微调。

核心结论如下：

1. QLoRA = 4bit 量化 base model + LoRA adapter 训练。
2. QLoRA 进一步降低了基座模型显存占用。
3. `BitsAndBytesConfig` 是 4bit 加载的核心配置。
4. QLoRA 通常需要 `prepare_model_for_kbit_training`。
5. 数据处理和普通 SFT/LoRA 相同，仍要正确构造 labels mask。
6. QLoRA adapter 仍然必须和 base model 匹配。
7. QLoRA 适合显存有限但想微调较大模型的场景。

下一讲，我们评估 SFT 前后模型行为变化，判断微调是否真的有效。

## 第 24 讲：评估 SFT 前后模型行为变化

### 本讲目标

学完本讲，你应该能做到六件事：

1. 设计 SFT 前后对比评测集。
2. 对 base model 和 SFT model 使用同一套 prompt 做生成对比。
3. 从格式遵循、任务正确性、幻觉、稳定性等维度分析行为变化。
4. 写出简单的自动化评估脚本。
5. 理解 loss、人工评估和自动指标之间的关系。
6. 能把 SFT 评估结果整理成项目报告和面试表达。

前面几讲我们完成了：

```text
加载开源 causal LM
构造 SFT 数据集
全参数 SFT
LoRA 微调
QLoRA 微调
```

但训练完成不等于任务完成。

你必须回答一个问题：

```text
微调真的让模型变好了吗？
```

本讲专门讲 SFT 前后模型行为评估。

---

### 一、为什么要评估 SFT 前后行为

SFT 的目标不是单纯降低训练 loss。

而是改变模型行为。

例如：

```text
更愿意按指令回答。
输出格式更稳定。
领域知识更贴合数据。
回答风格更符合要求。
```

如果 loss 降了，但模型生成质量没有改善，甚至变差，说明微调并不成功。

所以评估要同时看：

```text
训练/验证 loss
生成样例
人工偏好
任务正确性
格式遵循
安全性和幻觉
```

---

### 二、准备对比模型

我们需要两个模型：

```text
base model：微调前模型。
sft model：微调后模型，可以是全参数模型、LoRA adapter 或 QLoRA adapter。
```

加载 base model：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


base_model_name = "sshleifer/tiny-gpt2"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
base_model.eval()
```

加载全参数 SFT 后模型：

```python
sft_dir = "outputs/full_sft_tiny_gpt2/final"
sft_tokenizer = AutoTokenizer.from_pretrained(sft_dir)
sft_model = AutoModelForCausalLM.from_pretrained(sft_dir)
sft_model.eval()
```

如果是 LoRA adapter：

```python
from peft import PeftModel


adapter_dir = "outputs/lora_tiny_gpt2/adapter"
base_for_lora = AutoModelForCausalLM.from_pretrained(base_model_name)
sft_model = PeftModel.from_pretrained(base_for_lora, adapter_dir)
sft_model.eval()
```

注意：

```text
评估时 base 和 SFT 应尽量使用同一 tokenizer 和同一 prompt 模板。
```

---

### 三、构造评测集

评测集不要只用训练集样本。

应该包含：

```text
训练分布内样本。
训练分布外但相似的样本。
格式要求样本。
容易诱发幻觉的样本。
安全边界样本。
```

示例：

```python
eval_prompts = [
    {
        "id": "overfit_def",
        "instruction": "解释什么是过拟合。",
        "input": "",
        "expected_points": ["训练集", "泛化", "未见数据"],
    },
    {
        "id": "regularization_methods",
        "instruction": "给出三个缓解过拟合的方法。",
        "input": "",
        "expected_points": ["正则化", "dropout", "早停"],
    },
    {
        "id": "translation",
        "instruction": "把下面这句话翻译成英文。",
        "input": "我喜欢机器学习。",
        "expected_points": ["machine learning"],
    },
    {
        "id": "format_json",
        "instruction": "用 JSON 格式输出三个机器学习术语。",
        "input": "",
        "expected_points": ["{", "}", "术语"],
    },
]
```

这里的 `expected_points` 是非常粗糙的自动检查依据。

真实评估可以更复杂。

---

### 四、保持 prompt 模板一致

如果训练时使用 Alpaca 模板，评估时也要用同样模板。

```python
def build_prompt(example):
    if example.get("input", ""):
        return (
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Input:\n"
            f"{example['input']}\n\n"
            "### Response:\n"
        )
    return (
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Response:\n"
    )
```

如果训练时使用 chat template，评估时使用：

```python
messages = [
    {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
    {"role": "user", "content": user_content},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

模板不一致会严重影响评估结论。

---

### 五、统一生成函数

为了公平比较，base 和 SFT 使用同样生成参数。

```python
import torch


@torch.no_grad()
def generate_text(model, tokenizer, prompt, device="cpu"):
    model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text
```

这里先用 `do_sample=False`。

原因是贪心解码更稳定，便于对比。

如果要评估多样性，可以再用采样生成多次。

---

### 六、批量生成对比

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

results = []

for item in eval_prompts:
    prompt = build_prompt(item)

    base_output = generate_text(base_model, tokenizer, prompt, device=device)
    sft_output = generate_text(sft_model, sft_tokenizer, prompt, device=device)

    results.append({
        "id": item["id"],
        "prompt": prompt,
        "base_output": base_output,
        "sft_output": sft_output,
        "expected_points": item["expected_points"],
    })
```

打印对比：

```python
for r in results:
    print("=" * 80)
    print("ID:", r["id"])
    print("PROMPT:")
    print(r["prompt"])
    print("BASE:")
    print(r["base_output"])
    print("SFT:")
    print(r["sft_output"])
```

观察 SFT 是否更像训练目标。

---

### 七、评估维度

建议从六个维度评估。

#### 1. 指令遵循

模型是否理解用户要求。

例如要求“给三个方法”，是否真的给三个。

#### 2. 格式遵循

要求 JSON、列表、表格时，是否按格式输出。

#### 3. 内容正确性

回答是否事实正确、逻辑合理。

#### 4. 风格一致性

是否符合 SFT 数据中的回答风格。

例如简洁、专业、中文回答。

#### 5. 幻觉程度

是否编造不存在的信息。

#### 6. 稳定性

多次生成是否稳定，是否容易跑偏或重复。

---

### 八、一个简单打分表

可以人工打分：

```text
0：完全错误
1：部分正确
2：基本正确
3：很好
```

表格：

```text
样本 ID | base 指令遵循 | sft 指令遵循 | base 正确性 | sft 正确性 | 备注
```

或者用 Python 保存：

```python
scores = [
    {
        "id": "overfit_def",
        "base_follow": 0,
        "sft_follow": 2,
        "base_correct": 0,
        "sft_correct": 2,
        "note": "SFT 能回答过拟合定义，base 输出不稳定。",
    }
]
```

小项目里人工评估比复杂自动指标更直观。

---

### 九、关键词自动检查

可以写一个非常简单的关键词命中率。

```python
def keyword_score(output, expected_points):
    hit = 0
    for point in expected_points:
        if point.lower() in output.lower():
            hit += 1
    return hit / max(len(expected_points), 1)
```

应用：

```python
for r in results:
    base_score = keyword_score(r["base_output"], r["expected_points"])
    sft_score = keyword_score(r["sft_output"], r["expected_points"])
    print(r["id"], "base", base_score, "sft", sft_score)
```

注意：

```text
关键词分数很粗糙，只能辅助，不能替代人工判断。
```

例如模型可能用了同义表达，但关键词没命中。

也可能关键词命中，但整体回答是错的。

---

### 十、保存评估结果

```python
import json
from pathlib import Path


out_path = Path("outputs/eval_sft_outputs.jsonl")

with out_path.open("w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

JSONL 的好处是：

```text
一行一个样本。
方便后续追加字段。
方便人工检查和脚本处理。
```

也可以保存成 CSV。

但长文本字段里有换行时，JSONL 更稳。

---

### 十一、评估 loss 和行为评估的关系

验证 loss 低，不一定表示回答更好。

原因包括：

```text
验证集太像训练集。
loss 只衡量 token 级预测，不直接衡量任务成功。
格式、事实性、安全性很难只靠 loss 反映。
```

行为评估也有缺点：

```text
人工成本高。
容易受样本选择影响。
采样随机性会影响结论。
```

所以实践中要结合：

```text
验证 loss
固定 prompt 对比
人工评分
自动指标
真实业务测试
```

---

### 十二、微调后变差的常见原因

#### 原因 1：数据质量差

错答、重复、格式混乱会直接污染模型。

#### 原因 2：模板不一致

训练和推理 prompt 格式不同，模型不知道该如何回答。

#### 原因 3：学习率过大

模型原有能力被破坏。

#### 原因 4：训练太久

小数据上过拟合，模型变得机械或只会背训练样本。

#### 原因 5：labels mask 错误

如果 response 没有参与 loss，模型学不到回答。

如果 prompt 也参与大量 loss，模型可能过度学习模板。

#### 原因 6：评估 prompt 不合理

评估问题和训练目标完全不一致，不能说明微调失败。

---

### 十三、对比报告怎么写

一个简单报告结构：

```text
1. 模型与微调方法
2. 数据集规模与格式
3. 训练配置
4. 评估集设计
5. SFT 前后样例对比
6. 指标结果
7. 主要改进
8. 失败案例
9. 后续改进方向
```

示例结论：

```text
微调后模型在训练领域问题上的指令遵循明显增强，能更稳定地输出中文解释，并按模板回答。关键词命中率从 0.25 提升到 0.70。但在未覆盖的复杂推理问题上提升有限，部分回答仍存在泛化不足和重复表达，后续需要扩大高质量指令数据并加入更严格的验证集。
```

这样的结论比“loss 降了”更有说服力。

---

### 十四、自动化评估脚本骨架

```python
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(example):
    if example.get("input", ""):
        return (
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Input:\n"
            f"{example['input']}\n\n"
            "### Response:\n"
        )
    return (
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Response:\n"
    )


@torch.no_grad()
def generate_text(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def keyword_score(output, expected_points):
    output_lower = output.lower()
    hits = sum(1 for p in expected_points if p.lower() in output_lower)
    return hits / max(len(expected_points), 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_name = "sshleifer/tiny-gpt2"
    sft_dir = "outputs/full_sft_tiny_gpt2/final"

    base_tokenizer = AutoTokenizer.from_pretrained(base_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_name).to(device).eval()

    sft_tokenizer = AutoTokenizer.from_pretrained(sft_dir)
    sft_model = AutoModelForCausalLM.from_pretrained(sft_dir).to(device).eval()

    eval_prompts = [
        {
            "id": "overfit_def",
            "instruction": "解释什么是过拟合。",
            "input": "",
            "expected_points": ["训练集", "泛化", "未见数据"],
        }
    ]

    results = []
    for ex in eval_prompts:
        prompt = build_prompt(ex)
        base_output = generate_text(base_model, base_tokenizer, prompt, device)
        sft_output = generate_text(sft_model, sft_tokenizer, prompt, device)

        results.append({
            "id": ex["id"],
            "prompt": prompt,
            "base_output": base_output,
            "sft_output": sft_output,
            "base_keyword_score": keyword_score(base_output, ex["expected_points"]),
            "sft_keyword_score": keyword_score(sft_output, ex["expected_points"]),
        })

    out_path = Path("outputs/eval_sft_outputs.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
```

如果评估 LoRA/QLoRA adapter，把 SFT 模型加载部分改成 `PeftModel.from_pretrained` 即可。

---

### 十五、面试怎么讲 SFT 评估

如果面试官问“怎么评估 SFT 是否有效”，可以这样回答：

```text
我会准备一套独立评估 prompt，覆盖训练分布内、相似泛化、格式遵循、事实性和安全边界样本。然后用同样 prompt 和生成参数对比 base model 和 SFT model 的输出，从指令遵循、内容正确性、格式稳定性、幻觉和重复等维度做人工和自动评估。同时结合验证 loss，判断模型是否过拟合或遗忘。
```

如果追问“只看 validation loss 可以吗”，可以回答：

```text
不够。validation loss 是 token 级预测指标，能反映模型对验证数据的拟合，但不一定等价于任务成功。SFT 更关注模型行为变化，例如是否按指令回答、格式是否正确、是否减少幻觉，所以需要结合生成样例和任务级评估。
```

如果问“微调后模型变差怎么办”，可以回答：

```text
我会先检查数据质量、prompt template 是否一致、labels mask 是否正确；再看学习率、训练步数和是否过拟合；最后分析失败样例，判断是数据覆盖不足、模型容量不足还是采样参数问题。
```

---

### 十六、小练习

#### 练习 1

构造 20 条 SFT 评估 prompt，至少包含 5 条训练集中没见过的问题。

#### 练习 2

用同一套 prompt 对比 base model 和 SFT model 输出。

#### 练习 3

为每条输出人工打分：指令遵循、正确性、格式遵循各 0-3 分。

#### 练习 4

实现关键词命中率，并和人工评分对比。

#### 练习 5

找 3 个 SFT 后仍失败的样例，写出可能原因和改进方案。

---

### 本讲总结

这一讲完成了 SFT 前后模型行为评估。

核心结论如下：

1. SFT 的目标是改变模型行为，不只是降低 loss。
2. 评估应对比 base model 和 SFT model 在同一批 prompt 上的输出。
3. prompt 模板和生成参数要保持一致，避免评估不公平。
4. 评估维度包括指令遵循、格式遵循、正确性、幻觉、风格和稳定性。
5. validation loss 有参考价值，但不能替代生成行为评估。
6. 微调后变差通常和数据质量、模板不一致、labels mask、学习率和过拟合有关。
7. 好的 SFT 项目应该包含评估集、样例对比、指标结果和失败案例分析。

至此，第三册第四部分“Hugging Face 微调实战”正文第一版完成。
