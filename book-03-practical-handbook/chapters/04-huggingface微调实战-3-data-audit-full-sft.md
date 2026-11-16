# 第 4 章（第三篇）：先审计 SFT 数据，再更新全部参数

上一篇把对话写成 input、attention mask 和 labels；开始训练前，还要确认切分没有泄漏、过滤与截断没有留下空目标、模板和 tokenizer 版本可以复原。本篇先完成这份数据审计，再把同一组 labels 交给全参数 SFT。这样，模型结果才能归因到训练方法，而不是数据处理中的隐形变化。

## 4.2（续）：数据质量与版本

### 4.2.7 数据质量、去重与分组划分

SFT 清洗至少应覆盖：

```text
任务是否清楚，输入和输出是否错位。
回答是否真正回应问题，而不是复制提示。
空答案、乱码、模板残片、HTML 噪声。
重复样本和近似重复样本。
代码、JSON、表格和公式是否能解析。
来源、许可证、隐私和安全限制。
```

把同一文档切成相邻片段后随机拆分，会让验证集与训练集高度相似。更可靠的方式是按文档、来源、用户或任务族分组。每条记录应保存元数据：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
record = {
    "id": "example-0001",
    "source": "human_written",
    "task": "translation",
    "language": "zh-en",
    "license": "recorded_in_dataset_card",
    "messages": messages,
}
```
这样坏例能够追溯到数据来源和任务桶，而不是只知道某个 token 预测错误。

### 4.2.8 纯 Python 预处理实验

下面的程序模拟模板、mask、截断、去重和 padding：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import re


class ToyTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.vocab = {self.pad_token: 0, self.eos_token: 1}
        self.inverse = {0: self.pad_token, 1: self.eos_token}

    def encode(self, text):
        pieces = re.findall(r"\n|[^\s]+", text)
        ids = []
        for piece in pieces:
            if piece not in self.vocab:
                index = len(self.vocab)
                self.vocab[piece] = index
                self.inverse[index] = piece
            ids.append(self.vocab[piece])
        return ids

    def decode(self, ids):
        return " ".join(
            self.inverse[index]
            for index in ids
            if index not in {self.pad_token_id, 1}
        )


def format_example(example):
    prompt = (
        "Instruction: "
        + example["instruction"]
        + "\nResponse: "
    )
    return prompt, example["output"].strip()


def preprocess(example, tokenizer, max_length=32):
    if not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    prompt, response = format_example(example)
    if not response:
        raise ValueError("response must not be empty")
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(
        prompt + response + " " + tokenizer.eos_token
    )
    input_ids = full_ids[:max_length]
    labels = input_ids[:]
    labels[:min(len(prompt_ids), len(labels))] = [
        -100
    ] * min(len(prompt_ids), len(labels))
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def valid(feature):
    required = {"input_ids", "attention_mask", "labels"}
    if not required.issubset(feature):
        return False
    lengths = {len(feature[key]) for key in required}
    if len(lengths) != 1 or not lengths or min(lengths) == 0:
        return False
    return any(label != -100 for label in feature["labels"])


def collate(features, pad_token_id):
    if not features:
        raise ValueError("cannot collate an empty feature list")
    if pad_token_id is None:
        raise ValueError("pad_token_id is required")
    width = max(len(item["input_ids"]) for item in features)
    output = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in features:
        pad_count = width - len(item["input_ids"])
        output["input_ids"].append(
            item["input_ids"] + [pad_token_id] * pad_count
        )
        output["attention_mask"].append(
            item["attention_mask"] + [0] * pad_count
        )
        output["labels"].append(
            item["labels"] + [-100] * pad_count
        )
    return output


raw = [
    {"instruction": "解释过拟合。", "output": "训练集好但泛化差。"},
    {"instruction": "解释过拟合。", "output": "训练集好但泛化差。"},
    {"instruction": "翻译。", "output": "I like machine learning."},
]

tokenizer = ToyTokenizer()
seen = set()
features = []
for example in raw:
    key = (example["instruction"], example["output"])
    if key in seen:
        continue
    seen.add(key)
    item = preprocess(example, tokenizer)
    if valid(item):
        features.append(item)

if not features:
    raise ValueError("all examples lost their supervised response during truncation")
batch = collate(features, tokenizer.pad_token_id)
response_ids = [
    token_id
    for token_id, label in zip(
        features[0]["input_ids"],
        features[0]["labels"],
    )
    if label != -100
]

print("unique_valid_examples=", len(features))
print("batch_width=", len(batch["input_ids"][0]))
print("supervised_tokens=", sum(
    label != -100 for label in features[0]["labels"]
))
print("response=", tokenizer.decode(response_ids))
print("padding_labels_ok=", all(
    label == -100
    for mask_row, label_row in zip(
        batch["attention_mask"],
        batch["labels"],
    )
    for mask, label in zip(mask_row, label_row)
    if mask == 0
))
```
真实 tokenizer 还要处理 BPE 边界和特殊 token，这个 toy 实验只负责让监督范围可见。

### 4.2.9 数据处理结果也要版本化

划分数据集时，随机种子只是必要条件。还应固定去重规则、分组键、tokenizer、chat template、max_length、截断策略和无监督样本过滤规则。原始文本不变，tokenizer 版本变化也可能改变 token 数、截断位置和有效监督 token 数。因此应保存可抽查的中间样本：

```text
渲染后的完整文本。
input_ids。
attention_mask。
labels。
有效监督 token 数。
来源与任务元数据。
```

考虑序列“用户问：天气？助手答：晴天 EOS”。若模板渲染成 8 个 token，前 4 个是问题，后 4 个是回答和 EOS，assistant-only labels 为 $[-100,-100,-100,-100,r_1,r_2,r_3,\mathrm{EOS}]$。input_ids 仍保留全部 8 个 token；若模型内部做 causal shift，位置 3 的 logits 预测 $r_1$。回答四个目标的 token loss 若为 1、2、1、2，平均值是 $6/4=1.5$；误把 prompt 的四个位置也纳入后，整体平均可能降到 $7/8=0.875$，看似更好，回答监督却没有改善。检查有效 labels 解码出的文本，并和 logits 的预测位置对照，才能确认监督边界。

多轮对话中，assistant 可能先发起工具调用，再收到工具结果，最后给用户答复。是否训练工具调用格式、拒答标记或所有 assistant 消息，取决于目标；工具返回通常是外部程序生成的内容，不应无意中变成模型要复述的目标。用角色时间线标出 input token、目标 token 和 mask，能看清模型到底被要求生成什么。completion-only 的显式 prompt/completion 边界与 assistant-only 的多轮消息边界也不能混用。

截断会直接改变目标。若 prompt 长 900 token、回答长 200 token，`max_length=1024` 的右侧截断只留下 124 个回答 token；对完整 JSON 任务，这可能留下没有闭合括号和 EOS 的样本。可以把历史压到 800，为回答预留 224 个位置，并统计截断率；超过预算的样本按任务策略裁剪、拆分或丢弃。不同策略会改变数据分布，必须写进数据版本。

经过切分、版本和有效目标审计，数据才适合进入训练。下面先用全参数更新建立基线，再讨论怎样减少可训练状态。

## 4.3 全参数 SFT：监督目标、显存与可恢复训练

### 4.3.1 全参数训练的含义

全参数 SFT 使用上一节的 masked causal loss，只把可更新参数集合设为整个模型：

```math
\Theta_{\mathrm{train}}
=
\Theta_{\mathrm{model}}
```

参数更新可抽象为：

```math
\theta_{k+1}
=
\theta_k-\eta\widehat{g}_k
```

全参数的适配自由度大，但基座模型原来的能力也更容易被新数据覆盖。小数据、大学习率或过多训练轮次会放大遗忘和过拟合。

### 4.3.2 显存估算

设可训练参数量为 $P$，权重、梯度和一个优化器状态元素分别占 $b_w,b_g,b_o$ 字节，一个粗略训练内存表达为：

```math
M_{\mathrm{train}}
\approx
P(b_w+b_g+2b_o)
+
M_{\mathrm{activation}}
+
M_{\mathrm{temporary}}
```

AdamW 通常保存一阶和二阶状态，所以出现 $2b_o$。实现还可能保留 fp32 master weights、梯度 scaler、通信 buffer 和 allocator 缓存。激活内存又随 batch、序列长度、层数、隐藏维度和 checkpointing 改变。

fp16 权重约占 $2P$ 字节只是推理权重的量级；模型能生成不代表同一设备能全参数训练。

### 4.3.3 Trainer 与有效 batch

Trainer 封装的仍是：

```text
collator → forward → masked loss → backward
→ optimizer.step → scheduler → logging/eval/save
```

下面的 `TrainingArguments` 和 `Trainer` 片段假设 `model`、`train_dataset`、`eval_dataset` 和 `data_collator` 已分别按 4.1、4.2 节准备好；它们不是可以脱离上下文直接运行的完整训练脚本。

小模型配置：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import torch
from transformers import TrainingArguments


training_args = TrainingArguments(
    output_dir="outputs/full_sft_tiny",
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
旧版 Transformers 可能使用 evaluation_strategy，新版 Trainer 也可能用 processing_class 保存 tokenizer，而旧版使用 tokenizer 参数。版本差异应通过本地签名确认。

> 代码状态：静态审阅，未运行（2026-10-29）

```python
from transformers import Trainer


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model("outputs/full_sft_tiny/final")
tokenizer.save_pretrained("outputs/full_sft_tiny/final")
```
在数据并行、各设备 micro-batch 大小相同的条件下，梯度累积后的有效 batch size 近似为：

```math
B_{\mathrm{eff}}
=
B_{\mathrm{device}}
\times
N_{\mathrm{gpu}}
\times
G
```

但语言模型更应记录有效监督 token：

```math
N_{\mathrm{token,step}}
=
\sum_i\sum_t\mathbf{1}[y_{i,t}\neq-100]
```

当样本长度不同，逐 micro-batch 平均后再累积不一定等价于所有有效 token 的全局平均。严谨比较时要记录每步 token 数，必要时显式做 token-level normalization。

### 4.3.4 checkpointing 与恢复

gradient checkpointing 用重新计算换激活显存：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
model.config.use_cache = False
model.gradient_checkpointing_enable()
```
KV cache 为逐 token 推理优化，通常不应与训练 checkpointing 同时开启。

可恢复 checkpoint 可能包含：

```text
模型参数、optimizer state、学习率调度器 state。
随机数状态、训练步数和 Trainer state。
```

只加载权重可以继续推理，却不一定能恢复原优化轨迹：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
trainer.train(
    resume_from_checkpoint="outputs/full_sft_tiny/checkpoint-100"
)
```
恢复前应确认数据顺序、batch、梯度累积、随机种子和版本没有改变。改变有效 batch 后，旧 optimizer state 对新 token 语义未必合适。

### 4.3.5 手写训练循环

这个循环同样依赖已经构造好的 `model`、`train_dataset` 和 `data_collator`。它展示训练状态如何流动；真实任务还要先确认数据集非空，并为保存、恢复和评估定义明确的目录与策略。

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


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
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0:
            print("epoch=", epoch, "step=", step, "loss=", loss.item())
```
这段示例已经做梯度裁剪和非有限 loss 检查。更完整的训练流程还需按实验目标配置混合精度、学习率 scheduler、周期性评估、checkpoint 保存与恢复，并确认梯度累积时 loss 和有效 token 分母的约定。理解底层状态流，可以在 Trainer 的默认行为不符合预期时逐层定位。

### 4.3.6 纯 Python 的 masked loss 更新实验

下面用 bigram logits 代替 Transformer，只验证目标 token 的梯度和全参数更新：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import math
import random


random.seed(7)
vocab_size = 10
if vocab_size <= 1:
    raise ValueError("vocab_size must be greater than one")
logits = [
    [random.uniform(-0.02, 0.02) for _ in range(vocab_size)]
    for _ in range(vocab_size)
]
data = [
    {"ids": [2, 3, 4, 5, 6, 1], "labels": [-100, -100, -100, 5, 6, 1]},
    {"ids": [2, 7, 4, 8, 9, 1], "labels": [-100, -100, -100, 8, 9, 1]},
]
if not data:
    raise ValueError("masked-loss toy experiment requires non-empty data")
for item in data:
    if len(item["ids"]) != len(item["labels"]):
        raise ValueError("ids and labels must have equal lengths")
    if not item["ids"]:
        raise ValueError("each sequence must contain at least one token")
    if any(token_id < 0 or token_id >= vocab_size for token_id in item["ids"]):
        raise ValueError("input token id is outside the toy vocabulary")
    if any(
        label != -100 and (label < 0 or label >= vocab_size)
        for label in item["labels"]
    ):
        raise ValueError("supervised label is outside the toy vocabulary")


def softmax(row):
    maximum = max(row)
    values = [math.exp(value - maximum) for value in row]
    total = sum(values)
    return [value / total for value in values]


def zeros():
    return [[0.0] * vocab_size for _ in range(vocab_size)]


def loss_grad(batch):
    grad = zeros()
    loss = 0.0
    count = 0
    for item in batch:
        for pos in range(1, len(item["ids"])):
            target = item["labels"][pos]
            if target == -100:
                continue
            row = item["ids"][pos - 1]
            probabilities = softmax(logits[row])
            loss -= math.log(max(probabilities[target], 1e-12))
            count += 1
            for token_id, probability in enumerate(probabilities):
                grad[row][token_id] += probability
            grad[row][target] -= 1.0
    if count == 0:
        raise ValueError("masked loss is undefined when no label is supervised")
    for row in grad:
        for column in range(vocab_size):
            row[column] /= count
    return loss / count, grad, count


def add(left, right):
    for row in range(vocab_size):
        for column in range(vocab_size):
            left[row][column] += right[row][column]


def update(grad, learning_rate):
    for row in range(vocab_size):
        for column in range(vocab_size):
            logits[row][column] -= learning_rate * grad[row][column]


initial, _, valid_tokens = loss_grad(data)
for _ in range(30):
    total_grad = zeros()
    for item in data:
        _, grad, _ = loss_grad([item])
        add(total_grad, grad)
    for row in total_grad:
        for column in range(vocab_size):
            row[column] /= len(data)
    update(total_grad, 1.2)

final, _, _ = loss_grad(data)
print("valid_tokens=", valid_tokens)
print("initial_loss=", round(initial, 4))
print("final_loss=", round(final, 4))
print("loss_decreased=", final < initial)
print("trainable_params=", vocab_size * vocab_size)
```
loss 下降只证明这组目标被优化了，不证明模型学会了自然语言或获得了泛化能力。后者必须用独立数据和行为评估确认。

### 4.3.7 全参数 SFT 的边界

应同时观察：

```text
训练和验证的 masked loss。
有效监督 token 数和实际训练 token 数。
固定 prompt 的输出。
独立任务、格式和安全样本。
```

学习率过大、数据量小而 epoch 过多、领域风格覆盖通用风格、训练集与验证集重复，都可能让 loss 看起来很好而行为变差。

把公式代入 7B 参数作一次量级估算：BF16 权重和梯度各约需 $7\times10^9\times2=14$ GB；Adam 的两个 FP32 moment 合计约需 $7\times10^9\times8=56$ GB。仅这四项就约 84 GB 十进制，若另有 FP32 master weights 再加约 28 GB。它们还没有计入激活、临时 workspace、通信 buffer 和碎片；框架、分片与 offload 会改变单卡分布，但不会让总状态凭空消失。84 GB 约为 78.2 GiB，报告时要说明单位以及测量口径。

参数状态之外，激活随 batch、序列长度和层数增长。attention 的朴素 score 矩阵在序列长度从 2048 增至 4096 时约扩大 4 倍。Activation checkpointing 少保存中间激活、反向时再计算，通常以额外计算换显存；梯度累积降低每个 micro-batch 的激活峰值，却不减少权重、梯度和 Adam 状态。比如每卡 micro-batch=1、序列长 2048、累积 8 次、4 卡数据并行，有效 batch 约 32 条序列，但每次前向仍只持有每卡当前 micro-batch 的激活。

动态 padding 或分布式训练下，还要看 loss 的分母。若 3 条回答分别有 5、10、20 个有效 token，先算每条均值再平均会让三条回答权重相同；把 35 个 token loss 求和后除以 35，则长回答权重更高。训练器的 reduction 与实验目标必须一致。warm-up 时应使用计划中的 batch 和序列长度，记录 peak allocated、peak reserved，并为 optimizer step、evaluation 和 checkpoint 留显存余量。

“全参数”也要通过参数名和数量核对，而不是只看 Trainer 开始运行。记录 `requires_grad` 参数总数、各模块计数及 optimizer 参数组；若 embedding、输出头或 norm 意外被冻结，实际就不是预期的更新范围。ZeRO/FSDP 可以分片参数、梯度和优化器状态，但会引入通信及 checkpoint 管理成本；本章的小模型示例用于讲清流程，不能直接当作 7B 生产训练配置。

学习率、训练轮数与数据覆盖共同影响遗忘风险。若数据集中在某种格式，重复训练过久可能改善格式指标却损害通用问答或安全行为；比较 epoch 时应固定验证数据，分开看目标格式、通用能力和安全切片，并保存按预定规则选出的 checkpoint。可复核报告至少写有效监督 token、全局 batch、optimizer updates、wall-clock、峰值显存、保存策略及恢复时加载了哪些状态。权重能加载、loss 能下降和完整续训是三个不同结论。

全参数训练提供了更新完整模型的基线。若训练状态或显存负担过大，可以在不改监督目标的前提下转向 LoRA。
