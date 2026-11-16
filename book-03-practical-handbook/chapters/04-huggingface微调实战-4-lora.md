# 第 4 章（第四篇）：用 LoRA 限定可训练参数

全参数 SFT 会更新每一层权重。LoRA（low-rank adaptation，低秩适配）冻结基座矩阵，只在选定线性层旁增加可训练的低秩分支；它改变参数更新方式，不改变上一篇定义的对话监督目标。本篇从低秩表示推到 adapter 的训练、保存和容量边界。

## 4.4 LoRA：用低秩增量适配基座模型

### 4.4.1 为什么冻结基座

全参数 SFT 的成本来自三部分：所有权重都需要梯度，所有可训练权重都可能需要优化器状态，反向传播还要保存激活。LoRA 的基本策略是冻结基座模型，只增加少量可训练参数。

对一个线性层：

```math
y=W_0x
```

LoRA 改写为：

```math
y
=
W_0x
+
\frac{\alpha}{r}BAx
```

LoRA 的参数化来自原始论文；论文报告的效果与效率结论绑定其中的模型、任务和实验配置，不能直接推广成任意任务的保证。参见 [LoRA 原论文](https://arxiv.org/abs/2106.09685)。

其中 $d_{\mathrm{in}}>0$、$d_{\mathrm{out}}>0$、$r$ 是正整数；当 $r\ll\min(d_{\mathrm{in}},d_{\mathrm{out}})$ 时，才把它称为低秩增量。缩放因子 $\alpha$ 可以为零，但这会让 adapter 在该层没有有效增量，通常不是有意义的训练配置。这里的矩阵维度为：

```math
A\in\mathbb{R}^{r\times d_{\mathrm{in}}},
\qquad
B\in\mathbb{R}^{d_{\mathrm{out}}\times r},
\qquad
\Delta W=\frac{\alpha}{r}BA
```

原始权重 $W_0$ 不更新，只有 $A$ 和 $B$ 参与优化。若 $r$ 远小于输入输出维度，新增参数量为：

```math
N_{\mathrm{LoRA}}
=
r(d_{\mathrm{in}}+d_{\mathrm{out}})
```

LoRA 的低秩假设不是“所有任务变化都天然低秩”，而是一个容量与成本之间的工程近似。rank 太小可能表达不足，rank 太大则增加显存、训练时间和过拟合风险。最终选择应由独立验证结果决定。

### 4.4.2 初始化与超参数

PEFT v0.18.0 的默认初始化将 $B$ 置零，所以初始增量 $BA$ 为零，尚未训练的 adapter 不改变基座输出；其他初始化方式可能不同。常规缩放是 $\alpha/r$，启用 Rank-Stabilized LoRA 后为 $\alpha/\sqrt{r}$。$\alpha$ 控制增量幅度，`lora_dropout` 作用在适配分支上；把 alpha 设为 rank 的 2 倍或 4 倍只是常见搜索起点，不是定律。行为取决于配置和库版本，参见 [PEFT v0.18.0 LoRA 配置文档](https://huggingface.co/docs/peft/v0.18.0/package_reference/lora)。

重要配置包括：

```text
r：低秩维度。
lora_alpha：增量缩放。
lora_dropout：适配器分支 dropout。
target_modules：注入 LoRA 的真实模块名。
bias：是否训练偏置。
```

target_modules 不能靠模型名称猜。先打印模块名：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
for name, module in model.named_modules():
    if any(key in name for key in [
        "q_proj", "k_proj", "v_proj",
        "o_proj", "c_attn", "c_proj",
    ]):
        print(name, type(module).__name__)
```
Llama、Qwen、Mistral 类模型通常有 q_proj、k_proj、v_proj、o_proj，MLP 中还可能有 gate_proj、up_proj、down_proj。GPT-2 常见 `c_attn` 和 `c_proj`，其中 Conv1D 的权重布局为 `(fan_in, fan_out)`，PEFT 的 `fan_in_fan_out=True` 用于适配此布局；普通 `Linear` 通常不需要该选项。参见 [PEFT v0.18.0 LoRA 配置文档](https://huggingface.co/docs/peft/v0.18.0/package_reference/lora)。

### 4.4.3 用 PEFT 注入 adapter

GPT-2 类结构的示例：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
from peft import LoraConfig, TaskType, get_peft_model


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],
    bias="none",
    fan_in_fan_out=True,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```
Llama、Qwen 或 Mistral 类结构可以从更保守的 attention 配置开始：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
```
如果需要更高适配容量，再比较 q/k/v/o 或加入 MLP 投影。不要把“目标层越多”直接等同于“效果越好”；可训练参数、优化器状态和过拟合风险都会增加。

注入后必须检查参数集合：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
total_params = 0
trainable_params = 0
unexpected_trainable = []
for name, parameter in model.named_parameters():
    total_params += parameter.numel()
    if parameter.requires_grad:
        trainable_params += parameter.numel()
    if parameter.requires_grad and "lora" not in name.lower():
        unexpected_trainable.append(name)

if total_params == 0:
    raise ValueError("model has no parameters")
if trainable_params == 0:
    raise ValueError("no trainable LoRA parameters were found")
if unexpected_trainable:
    raise ValueError(
        "base parameters remain trainable: "
        + ", ".join(unexpected_trainable[:3])
    )
print("total=", total_params)
print("trainable=", trainable_params)
print("ratio=", trainable_params / total_params)
```
如果 trainable 为 0，通常是 target_modules 没匹配到真实结构；如果大量基座参数仍可训练，则没有实现预期的冻结。

### 4.4.4 LoRA 不改变 SFT 的数据目标

LoRA 只改变 $\Theta_{\mathrm{train}}$，不改变 input_ids、attention_mask 和 labels。prompt 位置仍应使用 -100，padding 位置仍应使用 -100，回答位置仍然使用真实 token id。

可以复用 4.2 的 collator 和预处理，只替换模型包装和训练参数：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
from transformers import Trainer, TrainingArguments


training_args = TrainingArguments(
    output_dir="outputs/lora_tiny",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=20,
    save_steps=20,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)
trainer.train()
```
LoRA 常使用比全参数 SFT 更大的学习率作为起点，但这只是搜索范围，不是固定答案。应在相同评估集上比较 rank、alpha、学习率和训练步数。

### 4.4.5 Adapter 的保存、加载与合并

adapter 文件通常只保存低秩参数和配置，不包含完整基座：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
adapter_dir = "outputs/lora_tiny/adapter"
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
```
推理时先加载训练时相同的 base model，再加载 adapter：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


base_name = "sshleifer/tiny-gpt2"
base_revision = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"
adapter_dir = "outputs/lora_tiny/adapter"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
base_model = AutoModelForCausalLM.from_pretrained(
    base_name,
    revision=base_revision,
)
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()

inputs = tokenizer("### Instruction:\n解释过拟合。\n\n### Response:\n",
                   return_tensors="pt")
with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```
如果部署链路不需要动态切换 adapter，可以合并权重：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("outputs/lora_tiny/merged")
tokenizer.save_pretrained("outputs/lora_tiny/merged")
```
合并后可以按普通 CausalLM 加载，但失去了多个任务 adapter 快速切换的灵活性。合并前后应使用同一批 prompt 做回归比较；不同版本的 PEFT、量化模型和特殊层可能对 merge 的支持不同。

### 4.4.6 LoRA 的容量与资源对比

假设同一个线性层的维度为 $4096\times4096$：

```math
N_{\mathrm{full}}
=
4096^2
\approx
16.8\mathrm{M}
```

rank 为 8 时：

```math
N_{\mathrm{LoRA}}
=
8\times4096+4096\times8
=
65536
```

这个单层例子中，可训练权重从约 16.8M 降到 65,536，约少 256 倍。但 LoRA 不是零成本：基座仍需载入并参与前向，激活仍沿完整网络传播，adapter 自身也要保存梯度和优化器状态。减少的是可训练状态与 adapter 文件大小，不代表整模型显存或计算量也缩小 256 倍。

### 4.4.7 纯 Python 的低秩更新实验

下面冻结 W，只更新 A、B，用均方误差验证低秩分支能够降低目标误差：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import random


random.seed(3)
d_in = 12
d_out = 12
rank = 2
scale = 2.0
if d_in <= 0 or d_out <= 0:
    raise ValueError("linear dimensions must be positive")
if rank <= 0 or rank > min(d_in, d_out):
    raise ValueError("rank must be in [1, min(d_in, d_out)]")
if scale <= 0:
    raise ValueError("scale must be positive")
W = [
    [random.uniform(-0.05, 0.05) for _ in range(d_in)]
    for _ in range(d_out)
]
W_before = [row[:] for row in W]
A = [
    [random.uniform(-0.02, 0.02) for _ in range(d_in)]
    for _ in range(rank)
]
B = [[0.0 for _ in range(rank)] for _ in range(d_out)]
data = []
for input_index, target_index in [(0, 3), (1, 4), (2, 5), (3, 6)]:
    vector = [0.0] * d_in
    target = [0.0] * d_out
    vector[input_index] = 1.0
    target[target_index] = 0.8
    data.append((vector, target))
if not data:
    raise ValueError("LoRA toy experiment requires non-empty data")


def matvec(matrix, vector):
    return [
        sum(weight * value for weight, value in zip(row, vector))
        for row in matrix
    ]


def forward(vector):
    base = matvec(W, vector)
    low_rank = matvec(B, matvec(A, vector))
    return [
        base_value + scale * delta
        for base_value, delta in zip(base, low_rank)
    ]


def loss():
    if not data:
        raise ValueError("loss is undefined for an empty dataset")
    total = 0.0
    for vector, target in data:
        prediction = forward(vector)
        total += sum(
            (value - expected) ** 2
            for value, expected in zip(prediction, target)
        ) / d_out
    return total / len(data)


def train_step(learning_rate):
    grad_a = [[0.0] * d_in for _ in range(rank)]
    grad_b = [[0.0] * rank for _ in range(d_out)]
    for vector, target in data:
        prediction = forward(vector)
        error = [value - expected for value, expected in zip(prediction, target)]
        ax = matvec(A, vector)
        for out_index in range(d_out):
            for rank_index in range(rank):
                grad_b[out_index][rank_index] += (
                    scale * error[out_index] * ax[rank_index] / d_out
                )
        for rank_index in range(rank):
            upstream = sum(
                error[out_index] * B[out_index][rank_index]
                for out_index in range(d_out)
            )
            for in_index in range(d_in):
                grad_a[rank_index][in_index] += (
                    scale * upstream * vector[in_index] / d_out
                )
    for rank_index in range(rank):
        for in_index in range(d_in):
            A[rank_index][in_index] -= (
                learning_rate * grad_a[rank_index][in_index] / len(data)
            )
    for out_index in range(d_out):
        for rank_index in range(rank):
            B[out_index][rank_index] -= (
                learning_rate * grad_b[out_index][rank_index] / len(data)
            )


initial = loss()
for _ in range(500):
    train_step(8.0)
final = loss()
print("initial_loss=", round(initial, 4))
print("final_loss=", round(final, 4))
print("loss_decreased=", final < initial)
print("full_params=", d_in * d_out)
print("lora_params=", rank * (d_in + d_out))
print("base_changed=", W != W_before)
```
base_changed 应为 False。这个结果只说明冻结矩阵和低秩分支的优化逻辑成立，不说明 rank=2 对任何真实任务都足够。

### 4.4.8 LoRA 的适用边界

LoRA 适合基座模型保留、任务变化相对局部、需要保存多个任务适配器的场景。需要警惕：

```text
target_modules 不匹配导致没有参数被训练。
adapter 与 base model、词表或模板不一致。
小数据上仍然会过拟合。
rank 太小造成能力不足，rank 太大增加成本。
merge 后未做输出回归。
```

因此训练报告应同时保存 base revision、adapter 配置、可训练参数量、有效训练 token 数和评估结果。

rank 描述可表示的增量空间，不是越大越好。举一个能看清容量的 $2\times2$ 层：rank 1 时，$A\in\mathbb{R}^{1\times2}$、$B\in\mathbb{R}^{2\times1}$，所以 $\mathrm{rank}(BA)\leq1$，不能表示满秩更新；rank 2 虽能表示任意 $2\times2$ 更新，因子却需要 8 个参数，多于直接学习矩阵的 4 个。真实网络的层宽通常远大于 rank，低秩才可能节省参数，同时也会限制可表达方向。

比较 rank=4 与 rank=16 时，应固定目标模块、数据、有效训练 token 和学习率搜索范围，并看独立验证切片。`target_modules` 也决定更新发生在哪里：只注入 q/v 投影与注入 q/k/v/o 及 MLP 投影是不同实验。训练前后都应核对实际模块名、可训练/冻结参数数和 adapter 范数；若增大 rank 只改善训练样本，可能是过拟合或数据不足。

默认初始化下，刚注入的 adapter 应让基座输出保持不变或接近不变；可用同一输入比较 logits。LoRA dropout 是训练时适配分支上的正则化，不是推理温度；评估模式下应关闭随机丢弃。若输出意外改变，检查初始化、目标层、权重加载和 adapter 是否重复合并。量化基座上的 merge 可能涉及反量化再写回，需检查当前库支持和数值回归，不能把合并前后的 shape 一致当成输出等价。

后续研究也尝试压缩低秩适配中的优化器状态。ICLR 2026 的 LoFT 摘要描述了把 Adam 一阶、二阶矩投影到低秩子空间，并在论文设置下缩小 adapter 与全参数微调的效果差距；这不代表常规 LoRA 在一般任务上必然落后。参见 [LoFT 官方论文摘要](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7428310c0f97f1c6bb2ef1be99c1ec2a-Abstract-Conference.html)。

LoRA 缩小了可训练参数集合，却仍要载入基座权重。下一篇在此基础上量化冻结基座，并检查它换来了哪些资源空间、又带来哪些精度边界。
