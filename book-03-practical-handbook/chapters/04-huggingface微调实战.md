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

资料边界说明：本讲第二轮精修时按 `WRITING_PLAN.md` 核对 Hugging Face Transformers 的 `PreTrainedModel.from_pretrained`、tokenizer、text generation / `generate`、`dtype` / `device_map` 文档，以及 PyTorch `Module.eval()` 和 `torch.no_grad()` 文档。这里以教学推理脚本为主，重点讲清 tokenizer、model、config、forward logits shape、生成长度和常见工程边界，不展开服务化推理、流式输出或量化加载细节。Hugging Face 当前主线文档中模型加载示例更常写 `dtype=...`，不少稳定版本和历史示例仍写 `torch_dtype=...`；二者表达的是加载权重时使用的目标 dtype，实际代码应以本机安装的 `transformers` 版本文档为准。

---

### 一、什么是 Causal LM

Causal LM 是 causal language model，也就是自回归语言模型。

它的训练目标是：

```text
根据当前位置及之前的 token，预测下一个 token。
```

更形式化地说，给定 token 序列 `x_0,x_1,...,x_{T-1}`，decoder-only causal LM 把联合概率分解为：

```math
p(x_0,\ldots,x_{T-1})
=
\prod_{t=0}^{T-1}p(x_t\mid x_0,\ldots,x_{t-1})
```

当 `t=0` 时，条件上下文可以理解为空上下文或 BOS token。实际工程中很多模型会用 BOS、special token 或 chat template 明确告诉模型“序列从哪里开始”。

训练时常用 next-token 负对数似然：

```math
L=
-\frac{1}{T-1}
\sum_{t=0}^{T-2}
\log p(x_{t+1}\mid x_0,\ldots,x_t)
```

推理时，`generate` 做的事情就是不断把已生成 token 拼回上下文，再预测下一个 token。

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

用符号写就是：

```math
X\in\mathbb{Z}^{B\times T}
```

```math
Z=f_\theta(X),\qquad Z\in\mathbb{R}^{B\times T\times V}
```

其中 `B` 是 batch size，`T` 是输入 token 数，`V` 是 `model.config.vocab_size`，也应和 tokenizer 可产生的 token id 范围匹配。`Z[b,t,:]` 是第 `b` 条样本第 `t` 个位置预测下一个 token 的 logits。

把 logits 转成概率后，第 `t` 个位置预测下一个 token 的分布是：

```math
p_\theta(x_{t+1}=v\mid x_0,\ldots,x_t)
=
\mathrm{softmax}(Z_{b,t,:})_v
```

其中 `v` 是候选 token id。`generate` 每一步基本就是取最后一个有效位置的 logits，按 greedy 或 sampling 策略选出下一个 token id，再把它拼回上下文。

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

tokenizer 和模型配置必须一致，核心约束是 token id 不能越界：

```math
0\le x_{b,t}<V,\qquad V=\mathrm{config.vocab\_size}
```

如果新增了特殊 token，例如 `<tool>`、`<image>` 或新的 chat role token，tokenizer 的长度可能变大。此时模型侧的 embedding matrix 和 lm head 也要同步 resize，否则轻则新 token 没有可训练表示，重则出现 token id 越界。

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

更一般地说：

```math
T_{\mathrm{out}}\le T_{\mathrm{prompt}}+M
```

其中 `M` 是 `max_new_tokens`。如果生成过程中提前遇到 EOS token，实际输出会更短。

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

如果你的 `transformers` 版本已经采用新文档中的参数名，也可能写成：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
).to(device)
```

如果本机版本不认识 `dtype`，就使用旧写法 `torch_dtype`。面试或项目文档里更重要的是讲清楚 dtype 的含义：它控制权重加载和计算时使用的数值类型，影响显存、速度和数值稳定性。

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

`device_map="auto"` 依赖 `accelerate` 做模块放置。它适合“模型大到单卡放不下或不想手动切分”的场景；如果只是 tiny-gpt2、distilgpt2 这类小模型，手动 `model.to(device)` 更直观。

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

如果当前环境还没有安装 `transformers` 或不能下载模型，可以先用下面这个纯 Python toy demo 体会 Hugging Face 推理接口背后的数据流。它不是真实模型，只模拟 `tokenizer(...)`、`outputs.logits.shape`、batch padding、`attention_mask` 和 `generate` 的核心 shape 关系：

```python
class ShapeOnly:
    def __init__(self, shape):
        self.shape = shape


class ToyOutput:
    def __init__(self, logits_shape):
        self.logits = ShapeOnly(logits_shape)


class ToyTokenizer:
    def __init__(self, padding_side="left"):
        self.tokens = [
            "<pad>", "<eos>", "Hello", "my", "name", "is",
            "The", "answer", "I", "am", "a", "tiny", "demo", ".",
        ]
        self.stoi = {token: idx for idx, token in enumerate(self.tokens)}
        self.itos = {idx: token for token, idx in self.stoi.items()}
        self.pad_token_id = self.stoi["<pad>"]
        self.eos_token_id = self.stoi["<eos>"]
        self.padding_side = padding_side

    def __len__(self):
        return len(self.tokens)

    def encode(self, text):
        return [self.stoi.get(token, self.eos_token_id) for token in text.split()]

    def __call__(self, texts, padding=False):
        if isinstance(texts, str):
            texts = [texts]
        input_ids = [self.encode(text) for text in texts]
        max_len = max(len(ids) for ids in input_ids)
        if padding:
            padded = []
            for ids in input_ids:
                pads = [self.pad_token_id] * (max_len - len(ids))
                if self.padding_side == "left":
                    padded.append(pads + ids)
                else:
                    padded.append(ids + pads)
            input_ids = padded
        attention_mask = [
            [0 if token_id == self.pad_token_id else 1 for token_id in ids]
            for ids in input_ids
        ]
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=True):
        pieces = []
        for idx in ids:
            token = self.itos[idx]
            if skip_special_tokens and token in {"<pad>", "<eos>"}:
                continue
            pieces.append(token)
        return " ".join(pieces)


class ToyConfig:
    def __init__(self, vocab_size, hidden_size, num_hidden_layers):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers


class ToyCausalLM:
    def __init__(self, vocab_size):
        self.config = ToyConfig(vocab_size, hidden_size=16, num_hidden_layers=2)
        self.training = True
        self.next_token = {5: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 1}

    def eval(self):
        self.training = False
        return self

    def __call__(self, input_ids, attention_mask=None):
        batch_size = len(input_ids)
        seq_len = len(input_ids[0])
        return ToyOutput((batch_size, seq_len, self.config.vocab_size))

    def generate(self, input_ids, max_new_tokens, pad_token_id=None):
        outputs = [list(ids) for ids in input_ids]
        for ids in outputs:
            for _ in range(max_new_tokens):
                last_id = ids[-1]
                next_id = self.next_token.get(last_id, 1)
                ids.append(next_id)
                if next_id == 1:
                    break
        return outputs


def shape(batch):
    return (len(batch), len(batch[0]))


tokenizer = ToyTokenizer(padding_side="left")
model = ToyCausalLM(vocab_size=len(tokenizer)).eval()
single = tokenizer("Hello my name is")
batch = tokenizer(["Hello my name is", "Hello"], padding=True)
single_outputs = model(**single)
batch_outputs = model(**batch)
generated_ids = model.generate(single["input_ids"], max_new_tokens=4)
max_token_id = max(max(ids) for ids in batch["input_ids"])

print("vocab_match=", model.config.vocab_size == len(tokenizer))
print("token_ids_in_range=", max_token_id < model.config.vocab_size)
print("single_input_shape=", shape(single["input_ids"]))
print("single_logits_shape=", single_outputs.logits.shape)
print("generated_shape=", shape(generated_ids))
print("new_tokens_added=", len(generated_ids[0]) - len(single["input_ids"][0]))
print("decoded=", tokenizer.decode(generated_ids[0]))
print("batch_input_ids=", batch["input_ids"])
print("batch_attention_mask=", batch["attention_mask"])
print("batch_logits_shape=", batch_outputs.logits.shape)
```

参考输出：

```text
vocab_match= True
token_ids_in_range= True
single_input_shape= (1, 4)
single_logits_shape= (1, 4, 14)
generated_shape= (1, 8)
new_tokens_added= 4
decoded= Hello my name is I am a tiny
batch_input_ids= [[2, 3, 4, 5], [0, 0, 0, 2]]
batch_attention_mask= [[1, 1, 1, 1], [0, 0, 0, 1]]
batch_logits_shape= (2, 4, 14)
```

这段 demo 只验证接口和 shape，不验证真实模型能力。真实使用时仍应优先运行上面的 `AutoTokenizer` 和 `AutoModelForCausalLM` 示例。

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

如果 `input_ids` 的 shape 是 `[B,T]`，那么 `attention_mask` 通常也是 `[B,T]`：

```math
m_{b,t}=
\begin{cases}
1, & \mathrm{valid} \\
0, & \mathrm{padding}
\end{cases}
```

直觉上，模型应该关注 `m=1` 的真实 token，而不是把补齐长度用的 padding 当成语义内容。后续做 SFT 时，padding 位置的 labels 也通常要设为 `-100`，让 loss 忽略这些位置。

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

tokenizer.padding_side = "left"

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
decoder-only 模型批量生成时通常优先使用 left padding
```

否则不同长度 prompt 可能处理不正确。原因是 decoder-only 生成通常从每条样本的最后一个有效 token 继续生成；如果 batch 里短 prompt 被 right padding 到同一长度，最后位置可能是 padding token，容易让生成起点和 `attention_mask` 语义变得不一致。具体 padding side 仍要以模型文档、tokenizer 默认设置和运行时 warning 为准。

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

资料边界说明：本讲第二轮精修时按 `WRITING_PLAN.md` 核对 Hugging Face Transformers chat template / `apply_chat_template` 文档、TRL `SFTTrainer` 关于 `assistant_only_loss` 和 `completion_only_loss` 的说明，以及 PyTorch `CrossEntropyLoss(ignore_index=-100)` 文档。这里重点讲数据构造和 loss mask 机制，不展开下一讲的 Trainer 参数、LoRA 配置或分布式训练。

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

用符号写，假设一条样本被拼成 token 序列 `x_0,...,x_{T-1}`，其中回答部分从位置 `s` 开始。可以定义一个 assistant mask：

```math
m_t=
\begin{cases}
0, & t<s \\
1, & t\ge s
\end{cases}
```

训练用的 labels 是：

```math
y_t=
\begin{cases}
-100, & m_t=0 \\
x_t, & m_t=1
\end{cases}
```

直觉上，`m_t=0` 的 prompt token 仍然在 `input_ids` 里作为上下文输入模型，但不会作为监督目标参与 loss。

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

如果模型内部或训练框架按 causal LM 方式自动 shift labels，那么传入模型的 `labels` 通常和 `input_ids` 等长。Hugging Face 的 `AutoModelForCausalLM` 系列模型一般会在 forward 里处理这种 shift。若你自己手写 loss，则要显式使用 `logits[:, :-1, :]` 去预测 `labels[:, 1:]`，不要把同一位置的 token 当成自己的标签。

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
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

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

这是因为常见 tokenizer 截断会保留序列前部、截掉尾部；而 SFT 的回答通常在尾部。如果 prompt 已经占满 `max_length`，response token 会被截掉，训练时就没有任何有效监督信号。

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

从 shape 上看，collator 的目标是：

```math
X,A,Y\in\mathbb{Z}^{B\times T_{\max}}
```

其中 `X` 是 `input_ids`，`A` 是 `attention_mask`，`Y` 是 `labels`。对 padding 位置，应满足 `A_{b,t}=0` 且 `Y_{b,t}=-100`，这样模型既不会把 padding 当作真实上下文，也不会在 padding 位置计算训练损失。

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

### 十七、0 依赖最小预处理 demo

如果当前环境没有安装 `datasets` 或 `transformers`，可以先用下面这个纯 Python demo 验证 SFT 数据构造的核心逻辑。它不是真实 tokenizer，但能展示 prompt/response 拼接、assistant-only labels、padding labels 和截断后无效样本过滤。

```python
import re


class ToyTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.vocab = {self.pad_token: 0, self.eos_token: 1}
        self.inv_vocab = {0: self.pad_token, 1: self.eos_token}
        self.pad_token_id = 0

    def _pieces(self, text):
        return re.findall(r"\n|[^\s]+", text)

    def encode(self, text):
        ids = []
        for piece in self._pieces(text):
            if piece not in self.vocab:
                idx = len(self.vocab)
                self.vocab[piece] = idx
                self.inv_vocab[idx] = piece
            ids.append(self.vocab[piece])
        return ids

    def decode(self, ids):
        pieces = []
        for idx in ids:
            token = self.inv_vocab[idx]
            if token not in {self.pad_token, self.eos_token, "\n"}:
                pieces.append(token)
        return " ".join(pieces)


def format_alpaca(example):
    if example.get("input", "").strip():
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
    return prompt, example["output"].strip()


def preprocess_example(example, tokenizer, max_length=64):
    prompt, response = format_alpaca(example)
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(prompt + response + " " + tokenizer.eos_token)
    input_ids = full_ids[:max_length]
    attention_mask = [1] * len(input_ids)
    labels = input_ids.copy()
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def has_response_labels(example):
    return any(label != -100 for label in example["labels"])


def collate(features, pad_token_id, label_pad_token_id=-100):
    max_len = max(len(f["input_ids"]) for f in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for feature in features:
        pad_len = max_len - len(feature["input_ids"])
        batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * pad_len)
        batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
        batch["labels"].append(feature["labels"] + [label_pad_token_id] * pad_len)
    return batch


def shape(matrix):
    return (len(matrix), len(matrix[0]))


raw_data = [
    {"instruction": "解释 什么是 过拟合。", "input": "", "output": "过拟合 是 训练集 好 但 泛化 差。"},
    {"instruction": "翻译 成 英文。", "input": "我 喜欢 机器学习。", "output": "I like machine learning."},
    {"instruction": "解释 什么是 过拟合。", "input": "", "output": "过拟合 是 训练集 好 但 泛化 差。"},
    {"instruction": "空 答案 样本。", "input": "", "output": "   "},
]

seen = set()
cleaned = []
for ex in raw_data:
    key = (ex["instruction"].strip(), ex.get("input", "").strip(), ex["output"].strip())
    if not ex["output"].strip() or key in seen:
        continue
    seen.add(key)
    cleaned.append(ex)

tokenizer = ToyTokenizer()
tokenized = [preprocess_example(ex, tokenizer, max_length=64) for ex in cleaned]
valid = [ex for ex in tokenized if has_response_labels(ex)]
batch = collate(valid, tokenizer.pad_token_id)
response_ids = [
    token_id
    for token_id, label in zip(valid[0]["input_ids"], valid[0]["labels"])
    if label != -100
]
truncated = preprocess_example(cleaned[0], tokenizer, max_length=6)
pad_label_values = [
    label
    for row_mask, row_labels in zip(batch["attention_mask"], batch["labels"])
    for mask, label in zip(row_mask, row_labels)
    if mask == 0
]

print("cleaned_count=", len(cleaned))
print("valid_count=", len(valid))
print("input_shape=", shape(batch["input_ids"]))
print("label_shape=", shape(batch["labels"]))
print("attention_shape=", shape(batch["attention_mask"]))
print("first_prompt_mask_ok=", all(label == -100 for label in valid[0]["labels"][:8]))
print("first_response_label_count=", sum(label != -100 for label in valid[0]["labels"]))
print("pad_labels=", pad_label_values)
print("pad_labels_all_minus100=", all(label == -100 for label in pad_label_values))
print("decoded_response=", tokenizer.decode(response_ids))
print("truncated_has_response=", has_response_labels(truncated))
```

参考输出：

```text
cleaned_count= 2
valid_count= 2
input_shape= (2, 24)
label_shape= (2, 24)
attention_shape= (2, 24)
first_prompt_mask_ok= True
first_response_label_count= 8
pad_labels= [-100, -100, -100, -100, -100]
pad_labels_all_minus100= True
decoded_response= 过拟合 是 训练集 好 但 泛化 差。
truncated_has_response= False
```

这个 demo 说明四件事：重复样本和空答案会被过滤；`input_ids`、`attention_mask`、`labels` 的 batch shape 对齐；prompt 和 padding 位置的 label 都是 `-100`；如果 `max_length` 过短导致回答被截掉，样本应被过滤。

---

### 十八、Chat Template 版本预处理思路

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

有两个边界要注意：

1. `apply_chat_template(..., add_generation_prompt=False)` 适合已经包含 assistant 答案的训练样本。
2. `apply_chat_template(..., add_generation_prompt=True)` 更适合推理或构造“只到 assistant 开始标记为止”的 prompt，用来计算 prompt token 长度。

如果 tokenizer 的 chat template 不支持标记 assistant token mask，就需要像上面这样分别构造 prompt text 和 full text，再用两者 token 长度差来得到回答区间。若模板本身支持 assistant mask，训练框架也可能直接用 assistant mask 做 `assistant_only_loss`。

---

### 十九、常见工程坑

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

#### 坑 8：忘记区分框架内部 shift 和手写 loss shift

如果使用 Hugging Face causal LM 的 `labels` 参数，通常传入等长 `labels` 即可；如果自己手写 loss，要显式把 logits 和 labels 错开一位。

---

### 二十、面试怎么讲 SFT 数据构造

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

### 二十一、小练习

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

资料边界说明：本讲第二轮精修时按 `WRITING_PLAN.md` 核对 Hugging Face Transformers 的 `Trainer`、`TrainingArguments`、模型保存加载和 gradient checkpointing 文档，以及 PyTorch optimizer / 训练循环相关接口。这里以小模型教学 SFT 为主，强调训练目标、batch / gradient accumulation、保存加载和排错边界；大模型 LoRA、QLoRA 和高性能分布式训练放到后续章节。

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

如果用上一讲构造好的 assistant mask `m_{b,t}`，全参数 SFT 的目标可以写成：

```math
L_{\mathrm{sft}}(\theta)=
-
\frac{
\sum_{b=1}^{B}\sum_{t=1}^{T-1}m_{b,t}\log p_\theta(x_{b,t}\mid x_{b,0:t-1})
}{
\sum_{b=1}^{B}\sum_{t=1}^{T-1}m_{b,t}
}
```

全参数的意思不是 loss 变了，而是优化时所有模型参数 `theta` 都允许更新。和 LoRA 相比，它的可训练参数集合更大：

```math
\Theta_{\mathrm{train}}=\Theta_{\mathrm{model}}
```

后面的 LoRA 会变成“冻结原模型参数，只训练 adapter 参数”。

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

```math
B_{\mathrm{eff}}
=
B_{\mathrm{device}}\times G\times N_{\mathrm{gpu}}
```

它的作用是：

```text
分多次 forward/backward 累积梯度，再执行一次 optimizer step。
```

这是显存不足时非常常用的技巧。

更具体地说，假设累积 `G` 个 micro-batch 后更新一次参数，那么更新方向近似是：

```math
g=
\frac{1}{G}
\sum_{i=1}^{G}\nabla_\theta L_i(\theta)
```

如果你增大 `gradient_accumulation_steps`，显存压力通常下降，但 optimizer step 频率也会下降；学习率、warmup steps 和 logging/eval steps 都要按 optimizer step 的视角重新理解。

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

较新的 Transformers 版本中，`Trainer` 更推荐使用 `processing_class=tokenizer` 来保存 tokenizer / processor 相关信息；很多旧教程仍使用 `tokenizer=tokenizer`。如果本地运行时看到 deprecation warning，可以把上面参数改成：

```python
processing_class=tokenizer
```

如果本机版本不认识 `processing_class`，继续使用 `tokenizer=tokenizer` 即可。

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

如果当前 `transformers` 版本提示 `tokenizer` 参数将被替换，可以把 `Trainer(...)` 里的 `tokenizer=tokenizer` 改成 `processing_class=tokenizer`。

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

粗略估算时可以先把模型参数量记作 `P`，权重、梯度和 AdamW 两个动量状态都会占显存：

```math
M_{\mathrm{state}}
\approx
P\cdot(s_w+s_g+2s_o)
```

其中 `s_w` 是单个权重元素字节数，`s_g` 是梯度字节数，`s_o` 是 AdamW 一阶或二阶状态的单元素字节数。真实训练还要加上激活值、临时 buffer、通信 buffer 和框架开销，所以这个公式只是理解“为什么训练比推理贵”的下限估算。

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

### 十七、0 依赖 toy 全参数训练循环

如果当前环境不能安装 `torch`、`transformers` 或 `datasets`，可以用下面这个纯 Python demo 先理解全参数 SFT 的三个核心动作：只在非 `-100` label 上算 loss，累积多个 micro-batch 梯度后再更新，一次更新会修改模型的全部参数。

这个 demo 用一个 bigram causal LM 代替真实 Transformer。它太小，不具备真实语言能力，但训练循环和 assistant-only loss 的逻辑是一样的。

```python
import math
import random


random.seed(7)
vocab_size = 12
train_data = [
    {"input_ids": [2, 3, 4, 5, 6, 7, 1], "labels": [-100, -100, -100, 5, 6, 7, 1]},
    {"input_ids": [2, 8, 4, 9, 10, 11, 1], "labels": [-100, -100, -100, 9, 10, 11, 1]},
]
logits = [
    [random.uniform(-0.02, 0.02) for _ in range(vocab_size)]
    for _ in range(vocab_size)
]


def softmax(row):
    max_logit = max(row)
    exps = [math.exp(x - max_logit) for x in row]
    total = sum(exps)
    return [x / total for x in exps]


def empty_grad():
    return [[0.0 for _ in range(vocab_size)] for _ in range(vocab_size)]


def add_grad(dst, src):
    for i in range(vocab_size):
        for j in range(vocab_size):
            dst[i][j] += src[i][j]


def loss_and_grad(batch):
    grad = empty_grad()
    total_loss = 0.0
    valid_tokens = 0
    for item in batch:
        input_ids = item["input_ids"]
        labels = item["labels"]
        for t in range(1, len(input_ids)):
            target = labels[t]
            if target == -100:
                continue
            context = input_ids[t - 1]
            probs = softmax(logits[context])
            total_loss -= math.log(max(probs[target], 1e-12))
            valid_tokens += 1
            for token_id, prob in enumerate(probs):
                grad[context][token_id] += prob
            grad[context][target] -= 1.0
    if valid_tokens == 0:
        return 0.0, grad, 0
    for i in range(vocab_size):
        for j in range(vocab_size):
            grad[i][j] /= valid_tokens
    return total_loss / valid_tokens, grad, valid_tokens


def apply_update(grad, lr):
    for i in range(vocab_size):
        for j in range(vocab_size):
            logits[i][j] -= lr * grad[i][j]


def evaluate_loss():
    loss, _, valid_tokens = loss_and_grad(train_data)
    return loss, valid_tokens


initial_loss, valid_tokens = evaluate_loss()
lr = 1.2
grad_accum_steps = 2
optimizer_steps = 0

for epoch in range(25):
    accum = empty_grad()
    accum_count = 0
    for example in train_data:
        micro_loss, micro_grad, _ = loss_and_grad([example])
        add_grad(accum, micro_grad)
        accum_count += 1
        if accum_count == grad_accum_steps:
            for i in range(vocab_size):
                for j in range(vocab_size):
                    accum[i][j] /= accum_count
            apply_update(accum, lr)
            optimizer_steps += 1
            accum = empty_grad()
            accum_count = 0

final_loss, _ = evaluate_loss()
trainable_params = vocab_size * vocab_size
prediction_after_marker = max(range(vocab_size), key=lambda k: softmax(logits[4])[k])

print("valid_supervised_tokens=", valid_tokens)
print("initial_loss=", round(initial_loss, 4))
print("final_loss=", round(final_loss, 4))
print("loss_decreased=", final_loss < initial_loss)
print("optimizer_steps=", optimizer_steps)
print("trainable_params=", trainable_params)
print("trainable_ratio=", 1.0)
print("predict_after_response_marker=", prediction_after_marker)
```

参考输出：

```text
valid_supervised_tokens= 8
initial_loss= 2.4823
final_loss= 0.6518
loss_decreased= True
optimizer_steps= 25
trainable_params= 144
trainable_ratio= 1.0
predict_after_response_marker= 5
```

这里的 `trainable_ratio=1.0` 对应全参数训练；如果是 LoRA，这个比例会远小于 1。`valid_supervised_tokens=8` 说明只有 labels 非 `-100` 的回答 token 参与监督。

---

### 十八、如何确认参数都在训练

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

### 十九、常见工程坑

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

#### 坑 7：Trainer 版本参数名不一致

不同 Transformers 版本中，`eval_strategy` / `evaluation_strategy`、`tokenizer` / `processing_class` 可能有差异。遇到报错或 warning 时，先查本地版本文档，不要机械复制旧教程。

---

### 二十、面试怎么讲全参数 SFT

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

### 二十一、小练习

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

资料边界说明：本讲第二轮精修时按 `WRITING_PLAN.md` 核对 LoRA 原论文、Hugging Face PEFT `LoraConfig` / adapter 保存加载 / merge 文档，以及 Transformers `Trainer` 版本边界。这里重点讲低秩增量、参数量估算、`target_modules` 选择和 adapter 生命周期；QLoRA 的 4bit 量化基座训练放到下一讲。

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

```math
y=W_0x
```

LoRA 冻结原始权重 `W_0`，增加一个低秩增量：

```math
y=W_0x+\frac{\alpha}{r}BAx
```

其中：

```math
\Delta W=\frac{\alpha}{r}BA
```

如果原始权重形状是 `W_0: [d_out, d_in]`，则：

```math
A\in\mathbb{R}^{r\times d_{\mathrm{in}}},\qquad
B\in\mathbb{R}^{d_{\mathrm{out}}\times r}
```

`r` 是低秩维度，通常远小于 `d_in` 和 `d_out`。

所以 LoRA 训练参数量是：

```math
N_{\mathrm{lora}}
=
r(d_{\mathrm{in}}+d_{\mathrm{out}})
```

而不是：

```math
N_{\mathrm{full}}
=
d_{\mathrm{out}}d_{\mathrm{in}}
```

这就是省参数的来源。

LoRA 常见初始化方式会让增量分支一开始接近 0，这样刚注入 adapter 时模型行为接近原始模型；随后训练只更新 `A` 和 `B`，让它们学习任务相关的权重增量。

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

```math
\frac{\alpha}{r}
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

如果输出的 trainable params 是 0，通常说明 `target_modules` 没有匹配到任何模块，或者模型结构和你复制的配置不一致。

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

和上一讲一样，如果当前 `transformers` 版本提示 `tokenizer` 参数将被替换，可以改用 `processing_class=tokenizer`。如果版本较旧不支持 `processing_class`，继续保留 `tokenizer=tokenizer`。

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

```math
W_0\in\mathbb{R}^{4096\times4096}
```

全参数训练参数量：

```math
N_{\mathrm{full}}=4096\times4096\approx 16.8\mathrm{M}
```

LoRA 设置 `r=8`：

```math
N_{\mathrm{lora}}
=
8\times4096+4096\times8
=
65536
```

相比 16.8M，少很多。

这就是为什么 LoRA 适合大模型微调。

---

### 十六、0 依赖 LoRA 机制 demo

如果当前环境没有安装 `peft`，可以先用下面这个纯 Python demo 理解 LoRA 的低秩增量。它用均方误差训练一个小线性层的 LoRA adapter：原始权重 `W_0` 保持不变，只更新低秩矩阵 `A` 和 `B`。

```python
import random


random.seed(3)
d_in = 16
d_out = 16
rank = 2
alpha = 4
scale = alpha / rank

W = [[random.uniform(-0.05, 0.05) for _ in range(d_in)] for _ in range(d_out)]
W_before = [row[:] for row in W]
A = [[random.uniform(-0.02, 0.02) for _ in range(d_in)] for _ in range(rank)]
B = [[0.0 for _ in range(rank)] for _ in range(d_out)]

data = []
for input_idx, target_idx in [(0, 3), (1, 4), (2, 5), (3, 6)]:
    x = [0.0] * d_in
    target = [0.0] * d_out
    x[input_idx] = 1.0
    target[target_idx] = 0.8
    data.append((x, target))


def matvec(matrix, vector):
    return [sum(w * x for w, x in zip(row, vector)) for row in matrix]


def forward(x):
    base = matvec(W, x)
    ax = matvec(A, x)
    bax = matvec(B, ax)
    y = [base_i + scale * delta_i for base_i, delta_i in zip(base, bax)]
    return y, ax


def loss_only():
    total = 0.0
    for x, target in data:
        y, _ = forward(x)
        total += sum((yi - ti) ** 2 for yi, ti in zip(y, target)) / d_out
    return total / len(data)


def train_step(lr):
    grad_A = [[0.0 for _ in range(d_in)] for _ in range(rank)]
    grad_B = [[0.0 for _ in range(rank)] for _ in range(d_out)]
    for x, target in data:
        y, ax = forward(x)
        error = [yi - ti for yi, ti in zip(y, target)]
        for out_idx in range(d_out):
            for j in range(rank):
                grad_B[out_idx][j] += scale * error[out_idx] * ax[j] / d_out
        for j in range(rank):
            upstream = sum(error[out_idx] * B[out_idx][j] for out_idx in range(d_out))
            for in_idx in range(d_in):
                grad_A[j][in_idx] += scale * upstream * x[in_idx] / d_out

    n = len(data)
    for j in range(rank):
        for in_idx in range(d_in):
            A[j][in_idx] -= lr * grad_A[j][in_idx] / n
    for out_idx in range(d_out):
        for j in range(rank):
            B[out_idx][j] -= lr * grad_B[out_idx][j] / n


initial_loss = loss_only()
for _ in range(500):
    train_step(lr=8.0)
final_loss = loss_only()

full_params = d_out * d_in
lora_params = rank * (d_in + d_out)

print("initial_loss=", round(initial_loss, 4))
print("final_loss=", round(final_loss, 4))
print("loss_decreased=", final_loss < initial_loss)
print("full_params=", full_params)
print("lora_params=", lora_params)
print("param_ratio=", round(lora_params / full_params, 4))
print("scale=", scale)
print("base_weight_changed=", W != W_before)
```

参考输出：

```text
initial_loss= 0.0416
final_loss= 0.0185
loss_decreased= True
full_params= 256
lora_params= 64
param_ratio= 0.25
scale= 2.0
base_weight_changed= False
```

这个 demo 的重点不是任务本身，而是三个机制：LoRA 参数量是 `r(d_in+d_out)`，低秩分支可以降低 loss，原始权重 `W_0` 没有被更新。

---

### 十七、LoRA 和全参数 SFT 对比

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

### 十八、常见工程坑

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

#### 坑 7：合并前后没有做回归测试

`merge_and_unload()` 后推理链路更简单，但仍要用同一批 prompts 检查合并前后的输出、困惑度或业务指标是否符合预期。

---

### 十九、面试怎么讲 LoRA

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

### 二十、小练习

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

资料边界说明：本讲第二轮精修时按 `WRITING_PLAN.md` 核对 QLoRA 论文、Hugging Face bitsandbytes 4bit 量化文档、PEFT `prepare_model_for_kbit_training` 文档和 Transformers 量化加载接口。这里重点讲 QLoRA 的工程机制、显存来源和配置边界，不展开 bitsandbytes kernel 细节或生产部署量化策略。

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

```math
W_0 \rightarrow Q_4(W_0)
```

其中 `Q_4` 表示 4bit 量化存储。前向时使用量化权重的反量化近似值，再叠加 LoRA 增量：

```math
y\approx \mathrm{dequant}(Q_4(W_0))x+\frac{\alpha}{r}BAx
```

`Q_4(W_0)` 冻结，LoRA adapter 保持可训练。

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

NF4 可以理解为更适合近似正态分布权重的 4bit 数据类型；double quant 的直觉是连量化常数也继续量化，从而进一步压缩量化元数据。它们都是为了降低冻结基座模型的存储成本，但不会改变“只训练 LoRA adapter”的核心训练目标。

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

和前两讲一样，如果当前 `transformers` 版本提示 `tokenizer` 参数将被替换，可以改用 `processing_class=tokenizer`。如果版本较旧不支持 `processing_class`，继续保留 `tokenizer=tokenizer`。

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

如果只看冻结基座权重，理想情况下：

```math
M_{\mathrm{fp16}}\approx 2P
```

```math
M_{\mathrm{4bit}}\approx 0.5P + M_{\mathrm{meta}}
```

其中 `P` 是参数量，单位是 byte 级粗略估算；`M_meta` 是量化 scale、zero point 或 double quant metadata 等额外开销。真实显存还包括 LoRA 权重、梯度、优化器状态、激活值和临时 buffer。

---

### 十四、0 依赖量化基座 + LoRA demo

如果当前环境没有 CUDA、bitsandbytes 或 peft，可以先用下面这个纯 Python demo 理解 QLoRA 的核心结构：基座权重量化后冻结，只训练 LoRA adapter。这里用简单均匀 4bit 量化演示存储和误差直觉；真实 QLoRA 常用 NF4 和 double quant，数值细节更复杂。

```python
import random


random.seed(5)
d = 16
rank = 2
alpha = 4
scale = alpha / rank
W_fp = [[random.uniform(-0.3, 0.3) for _ in range(d)] for _ in range(d)]


def quantize_row(row, levels=16):
    lo, hi = min(row), max(row)
    step = (hi - lo) / (levels - 1) if hi != lo else 1.0
    q = [round((x - lo) / step) for x in row]
    return q, lo, step


q_rows = []
metadata = []
for row in W_fp:
    q, lo, step = quantize_row(row)
    q_rows.append(q)
    metadata.append((lo, step))


def dequantize():
    return [
        [lo + q * step for q in row]
        for row, (lo, step) in zip(q_rows, metadata)
    ]


W_q = dequantize()
W_q_before = [row[:] for row in W_q]
quant_mse = sum(
    (a - b) ** 2
    for fp_row, q_row in zip(W_fp, W_q)
    for a, b in zip(fp_row, q_row)
) / (d * d)

A = [[random.uniform(-0.02, 0.02) for _ in range(d)] for _ in range(rank)]
B = [[0.0 for _ in range(rank)] for _ in range(d)]
data = []
for input_idx, target_idx in [(0, 5), (1, 6), (2, 7), (3, 8)]:
    x = [0.0] * d
    target = [0.0] * d
    x[input_idx] = 1.0
    target[target_idx] = 0.8
    data.append((x, target))


def matvec(matrix, vector):
    return [sum(w * x for w, x in zip(row, vector)) for row in matrix]


def forward(x):
    base = matvec(W_q, x)
    ax = matvec(A, x)
    bax = matvec(B, ax)
    return [b + scale * delta for b, delta in zip(base, bax)], ax


def loss():
    total = 0.0
    for x, target in data:
        y, _ = forward(x)
        total += sum((yi - ti) ** 2 for yi, ti in zip(y, target)) / d
    return total / len(data)


def train_step(lr):
    grad_A = [[0.0 for _ in range(d)] for _ in range(rank)]
    grad_B = [[0.0 for _ in range(rank)] for _ in range(d)]
    for x, target in data:
        y, ax = forward(x)
        error = [yi - ti for yi, ti in zip(y, target)]
        for out_idx in range(d):
            for j in range(rank):
                grad_B[out_idx][j] += scale * error[out_idx] * ax[j] / d
        for j in range(rank):
            upstream = sum(error[out_idx] * B[out_idx][j] for out_idx in range(d))
            for in_idx in range(d):
                grad_A[j][in_idx] += scale * upstream * x[in_idx] / d
    n = len(data)
    for j in range(rank):
        for in_idx in range(d):
            A[j][in_idx] -= lr * grad_A[j][in_idx] / n
    for out_idx in range(d):
        for j in range(rank):
            B[out_idx][j] -= lr * grad_B[out_idx][j] / n


initial_loss = loss()
for _ in range(500):
    train_step(lr=8.0)
final_loss = loss()

print("quant_mse=", round(quant_mse, 6))
print("initial_loss=", round(initial_loss, 4))
print("final_loss=", round(final_loss, 4))
print("loss_decreased=", final_loss < initial_loss)
print("fp16_base_bytes=", d * d * 2)
print("int4_base_bytes_ideal=", round(d * d * 0.5, 1))
print("lora_trainable_params=", rank * (d + d))
print("base_quantized_changed=", W_q != W_q_before)
```

参考输出：

```text
quant_mse= 9.6e-05
initial_loss= 0.0632
final_loss= 0.02
loss_decreased= True
fp16_base_bytes= 512
int4_base_bytes_ideal= 128.0
lora_trainable_params= 64
base_quantized_changed= False
```

这段 demo 对应 QLoRA 的核心直觉：量化基座有小的量化误差，理想 4bit 存储显著小于 fp16，训练过程中被冻结的量化基座不变，只有 LoRA adapter 在学习。

---

### 十五、QLoRA 的限制

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

### 十六、LoRA、QLoRA、全参数 SFT 对比

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

### 十七、常见工程坑

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

#### 坑 8：把 4bit 存储误解成 4bit 全流程计算

QLoRA 的核心收益来自冻结基座权重的低比特存储；矩阵乘法、adapter、norm 和部分中间计算仍会使用 fp16/bf16 等 compute dtype。

---

### 十八、面试怎么讲 QLoRA

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

### 十九、小练习

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

资料边界说明：本讲第二轮精修时按 `WRITING_PLAN.md` 核对 Hugging Face Transformers 的 text generation / `generate`、`Trainer.evaluate` / `predict` 和 Hugging Face Evaluate 指标库文档。这里重点讲离线 SFT 行为评估的基本闭环：固定评测集、固定 prompt 模板、固定生成参数、任务级指标、人工复核和坏例归因；不展开大规模 leaderboard、LLM-as-a-judge 生产评测、在线 A/B 实验或安全红队评估。

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

可以把一次离线行为评估写成：

```math
\mathcal{D}_{\mathrm{eval}}=\{(p_i,c_i,e_i)\}_{i=1}^{N}
```

其中 `p_i` 是第 `i` 条评估 prompt，`c_i` 是评估类别，例如格式遵循、知识问答或安全边界，`e_i` 是期望检查点、参考答案或评分 rubric。

对同一条样本，base 和 SFT 的输出分别是：

```math
y_i^{\mathrm{base}}=G(\theta_{\mathrm{base}},p_i,g),
\qquad
y_i^{\mathrm{sft}}=G(\theta_{\mathrm{sft}},p_i,g)
```

其中 `G` 表示生成过程，`g` 表示固定的生成参数，例如 `max_new_tokens`、`do_sample`、`temperature`、`top_p`。公平对比的核心是：只改变模型参数，不随意改变 prompt 模板和生成参数。

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

如果 SFT 是 LoRA 或 QLoRA adapter，通常仍然复用 base model 的 tokenizer。不要把“不同基座模型之间的能力差异”混进“SFT 是否有效”的结论里；如果必须换 tokenizer、换 base 或换 chat template，需要在报告里单独说明。

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

评测集设计至少要记录三类信息：

```text
样本来源：训练集、验证集、人工新增、线上坏例。
样本类别：知识、格式、推理、安全、风格、拒答边界。
期望行为：关键词、参考答案、格式规则、人工评分 rubric。
```

如果评估集里训练样本占比过高，结果会偏乐观。一个简单的分桶统计可以写成：

```math
N=\sum_{c\in\mathcal{C}}N_c
```

其中 `N_c` 是类别 `c` 下的样本数。报告里不要只写总分，还要写每个类别的结果，否则格式类样本、幻觉类样本和普通问答样本会互相掩盖。

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
    prompt_len = inputs["input_ids"].shape[-1]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=pad_token_id,
    )
    new_tokens = outputs[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
```

这里先用 `do_sample=False`。

原因是贪心解码更稳定，便于对比。

如果要评估多样性，可以再用采样生成多次。

注意这里返回的是新生成的 response，而不是 prompt + response 的整体文本。否则关键词检查可能把 prompt 里的词误算成模型输出命中。

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

更一般地，关键词命中率可以写成：

```math
K_i=
\frac{1}{M_i}
\sum_{j=1}^{M_i}
\mathbf{1}[q_{i,j}\subset y_i]
```

其中 `M_i` 是第 `i` 条样本的检查点数量，`q_{i,j}` 是第 `j` 个关键词或检查点，`y_i` 是模型输出。`K_i` 越高，说明粗粒度检查点命中越多，但它不等价于语义正确。

如果样本有明确格式要求，可以统计格式通过率：

```math
A_{\mathrm{fmt}}=
\frac{1}{N_{\mathrm{fmt}}}
\sum_{i\in\mathcal{I}_{\mathrm{fmt}}}
\mathbf{1}[\mathrm{valid}(y_i)]
```

其中 `\mathcal{I}_{\mathrm{fmt}}` 是有格式要求的样本集合，`\mathrm{valid}` 表示 JSON 解析成功、字段完整、表格列数正确等检查函数。

对比 base 和 SFT 时，也可以统计胜率和回归率：

```math
W_{\mathrm{sft}}=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[s(y_i^{\mathrm{sft}})>s(y_i^{\mathrm{base}})]
```

```math
R_{\mathrm{reg}}=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[s(y_i^{\mathrm{sft}})<s(y_i^{\mathrm{base}})]
```

其中 `s` 是人工分或自动综合分。回归率很重要，因为平均分提升可能掩盖某些类别明显变差。

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

用 assistant-only labels 评估时，验证 loss 通常类似：

```math
L_{\mathrm{val}}=
-
\frac{1}{\sum_{i,t}m_{i,t}}
\sum_{i,t}
m_{i,t}\log p_{\theta}(x_{i,t}\mid x_{i,1:t-1})
```

其中 `m_{i,t}=1` 表示该 token 参与 loss，`m_{i,t}=0` 表示 prompt 或 padding 被 mask 掉。Hugging Face `Trainer.evaluate` 在数据里有 `labels` 时通常会返回 `eval_loss`，但这个值仍然是 token 级预测指标。

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
    prompt_len = inputs["input_ids"].shape[-1]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=pad_token_id,
    )
    new_tokens = outputs[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


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

### 十五、0 依赖最小评估 demo

下面这个 demo 不依赖 `transformers`，只模拟 base 和 SFT 输出已经生成好的情况。它演示四件事：关键词命中率、格式通过率、SFT 胜出样本数、回归样本数和坏例桶统计。

```python
import json


EVAL_SET = [
    {
        "id": "overfit_def",
        "expected_points": ["训练集", "泛化", "未见数据"],
        "format": None,
        "needs_source": False,
    },
    {
        "id": "regularization_json",
        "expected_points": ["正则化", "dropout", "早停"],
        "format": "json",
        "needs_source": False,
    },
    {
        "id": "unknown_paper",
        "expected_points": ["无法确认"],
        "format": None,
        "needs_source": True,
    },
    {
        "id": "translation",
        "expected_points": ["machine learning"],
        "format": None,
        "needs_source": False,
    },
]

BASE_OUTPUTS = {
    "overfit_def": "过拟合是模型记住训练集。",
    "regularization_json": "可以多训练，也可以多调参。",
    "unknown_paper": "这篇论文提出了革命性的训练算法。",
    "translation": "I like study.",
}

SFT_OUTPUTS = {
    "overfit_def": "过拟合是模型在训练集表现好，但在未见数据上泛化差。",
    "regularization_json": '{"methods": ["正则化", "dropout", "早停"]}',
    "unknown_paper": "无法确认这篇论文是否存在；需要提供来源后再总结。",
    "translation": "I like machine learning.",
}


def keyword_score(output, expected_points):
    output_lower = output.lower()
    hits = sum(1 for point in expected_points if point.lower() in output_lower)
    return hits / max(len(expected_points), 1)


def format_pass(output, rule):
    if rule == "json":
        try:
            json.loads(output)
            return True
        except json.JSONDecodeError:
            return False
    return True


def has_unsupported_claim(output, example):
    risky_words = ["提出了", "证明了", "实验表明"]
    cautious_words = ["无法确认", "需要提供来源", "不知道"]
    risky = any(word in output for word in risky_words)
    cautious = any(word in output for word in cautious_words)
    return example["needs_source"] and risky and not cautious


def evaluate(outputs):
    rows = []
    bad_cases = {"format_fail": 0, "low_keyword": 0, "unsupported_claim": 0}
    format_checks = []

    for example in EVAL_SET:
        output = outputs[example["id"]]
        k_score = keyword_score(output, example["expected_points"])
        ok_format = format_pass(output, example["format"])
        unsupported = has_unsupported_claim(output, example)

        if example["format"] is not None:
            format_checks.append(ok_format)
        if not ok_format:
            bad_cases["format_fail"] += 1
        if k_score < 0.5:
            bad_cases["low_keyword"] += 1
        if unsupported:
            bad_cases["unsupported_claim"] += 1

        rows.append({
            "id": example["id"],
            "keyword": round(k_score, 3),
            "format": ok_format,
            "unsupported": unsupported,
        })

    avg_keyword = sum(row["keyword"] for row in rows) / len(rows)
    format_rate = sum(format_checks) / max(len(format_checks), 1)
    return {
        "avg_keyword": round(avg_keyword, 3),
        "format_pass": round(format_rate, 3),
        "bad_cases": bad_cases,
        "rows": rows,
    }


def row_value(row):
    return row["keyword"] + 0.1 * int(row["format"]) - 0.5 * int(row["unsupported"])


base_report = evaluate(BASE_OUTPUTS)
sft_report = evaluate(SFT_OUTPUTS)

wins = sum(
    row_value(sft_row) > row_value(base_row)
    for base_row, sft_row in zip(base_report["rows"], sft_report["rows"])
)
regressions = sum(
    row_value(sft_row) < row_value(base_row)
    for base_row, sft_row in zip(base_report["rows"], sft_report["rows"])
)

print(f"base_avg_keyword={base_report['avg_keyword']}")
print(f"sft_avg_keyword={sft_report['avg_keyword']}")
print(f"base_format_pass={base_report['format_pass']}")
print(f"sft_format_pass={sft_report['format_pass']}")
print(f"sft_wins={wins}")
print(f"regressions={regressions}")
print(f"base_bad_cases={base_report['bad_cases']}")
print(f"sft_bad_cases={sft_report['bad_cases']}")
```

输出示例：

```text
base_avg_keyword=0.083
sft_avg_keyword=1.0
base_format_pass=0.0
sft_format_pass=1.0
sft_wins=4
regressions=0
base_bad_cases={'format_fail': 1, 'low_keyword': 4, 'unsupported_claim': 1}
sft_bad_cases={'format_fail': 0, 'low_keyword': 0, 'unsupported_claim': 0}
```

这个 demo 的重点不是证明 SFT 一定更好，而是展示评估脚本应该输出可解释的诊断信息。真实项目里要替换成真实模型输出，并加入人工复核，尤其要检查关键词命中但语义错误的样本。

---

### 十六、面试怎么讲 SFT 评估

如果面试官问“怎么评估 SFT 是否有效”，可以这样回答：

```text
我会准备一套独立评估 prompt，覆盖训练分布内、相似泛化、格式遵循、事实性和安全边界样本。然后用同样 prompt 模板和生成参数对比 base model 与 SFT model 的新生成输出，从指令遵循、内容正确性、格式稳定性、幻觉和重复等维度做人工和自动评估。同时记录关键词命中率、格式通过率、胜率、回归率和坏例桶，并结合验证 loss 判断模型是否过拟合或遗忘。
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

### 十七、小练习

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

#### 练习 6

在评估脚本里加入 `regression` 标记：只要 SFT 输出比 base 输出人工分更低，就把样本写入单独的坏例文件。

---

### 本讲总结

这一讲完成了 SFT 前后模型行为评估。

核心结论如下：

1. SFT 的目标是改变模型行为，不只是降低 loss。
2. 评估应对比 base model 和 SFT model 在同一批 prompt 上的输出。
3. prompt 模板和生成参数要保持一致，避免评估不公平。
4. 评估维度包括指令遵循、格式遵循、正确性、幻觉、风格和稳定性。
5. 自动指标可以包括关键词命中率、格式通过率、胜率、回归率和坏例桶。
6. validation loss 有参考价值，但不能替代生成行为评估。
7. 微调后变差通常和数据质量、模板不一致、labels mask、学习率和过拟合有关。
8. 好的 SFT 项目应该包含评估集、样例对比、指标结果和失败案例分析。

至此，第三册第四部分“Hugging Face 微调实战”正文第一版完成。
