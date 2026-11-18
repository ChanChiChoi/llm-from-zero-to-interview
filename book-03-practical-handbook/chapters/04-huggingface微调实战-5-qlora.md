# 第 4 章（第五篇）：在量化基座上训练 LoRA

LoRA 减少了需要训练的参数，但基座权重仍要放进内存。QLoRA 进一步用低比特格式存储冻结的基座，同时以较高精度计算并训练 LoRA adapter。存储位宽、计算 dtype、量化误差和激活内存是不同的问题，不能当成同一个开关。本篇沿用相同的 SFT labels，只看权重的存储方式改变后，显存和计算开销会怎样变化。

## 4.5 QLoRA：量化基座上的低秩训练

### 4.5.1 QLoRA 在 LoRA 上增加了什么

普通 LoRA 通常以 fp16、bf16 或 fp32 加载冻结基座；QLoRA 将冻结基座以低比特形式存储，仍然只训练 LoRA 分支：

```math
y
\approx
\mathrm{dequant}(Q_4(W_0))x
+
\frac{\alpha}{r}BAx
```

Q_4 表示 4bit 量化存储，dequant 表示计算时恢复到某种计算 dtype。量化的是冻结基座权重，不意味着所有矩阵乘法、中间激活和 adapter 都以 4bit 计算。

QLoRA 的显存收益来自：

```text
冻结基座的权重存储更小。
基座不保存训练梯度和 optimizer state。
只有 LoRA adapter 需要反向更新。
```

### 4.5.2 NF4、double quant 与 compute dtype

NF4 是一种 4-bit 权重编码：每个权重可表示为 16 个离散等级之一。与把数值范围等宽切成 16 份的 int4 不同，NF4 的等级按近似正态的权重分布安排，让常见区域有更细的表示。double quantization 再压缩这些等级所需的 scale 等元数据；论文还讨论 paged optimizer 来缓解训练显存峰值。三者分别涉及权重编码、元数据和峰值管理，不应混成一个“更低精度”开关。参见 [QLoRA 原论文](https://arxiv.org/abs/2305.14314) 和 [Transformers v5.17.0 bitsandbytes 文档](https://huggingface.co/docs/transformers/v5.17.0/quantization/bitsandbytes)。

常见配置：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import torch
from transformers import BitsAndBytesConfig


if not torch.cuda.is_available():
    raise RuntimeError("4bit training needs a supported CUDA environment")

supports_bf16 = torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)
```
compute dtype 决定许多计算阶段使用 fp16 或 bf16，不是量化存储位数。GPU 不支持 bf16 时，应选择硬件支持的 fp16；CPU 环境不能因为配置写对了就获得 bitsandbytes 的 CUDA kernel。

### 4.5.3 加载量化基座并准备训练

> 代码状态：静态审阅，未运行（2026-10-29）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "Qwen/Qwen2.5-0.5B"
model_revision = "b937cc073b55e980e6f21d5f8b7a77818efcf0eb"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    revision=model_revision,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    revision=model_revision,
    quantization_config=bnb_config,
)
model.config.pad_token_id = tokenizer.pad_token_id
```
该示例固定到 Qwen2.5-0.5B 的公开快照；对应模型卡使用 Transformers `AutoTokenizer` 和 `AutoModelForCausalLM` 加载，没有打开 `trust_remote_code`。这是单 GPU 训练示例，因此省略 `device_map`，让训练框架管理设备放置。Transformers v5.17.0 的 bitsandbytes 指南明确指出 `device_map="auto"` 只应用于推理；多卡训练应改用所选训练框架支持的并行配置。参见 [Qwen2.5-0.5B 模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/b937cc073b55e980e6f21d5f8b7a77818efcf0eb) 和 [bitsandbytes 微调说明](https://huggingface.co/docs/transformers/v5.17.0/quantization/bitsandbytes#finetuning)。

PEFT 提供量化训练准备函数：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
from peft import prepare_model_for_kbit_training


model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
```
它会处理适合 k-bit 训练的若干精度和梯度设置。随后注入 LoRA：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
from peft import LoraConfig, TaskType, get_peft_model


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```
target_modules 仍需根据 model.named_modules() 验证。QLoRA 不会修复 LoRA 的模块选择错误。

### 4.5.4 QLoRA 训练配置

数据处理与普通 SFT、LoRA 相同：

```text
input_ids：完整模板序列。
attention_mask：有效 token。
labels：prompt/padding 为 -100，assistant 为真实 token id。
```

训练参数示例：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
from transformers import TrainingArguments


training_args = TrainingArguments(
    output_dir="outputs/qlora_model",
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
    bf16=supports_bf16,
    fp16=torch.cuda.is_available() and not supports_bf16,
    gradient_checkpointing=True,
    report_to="none",
)
```
然后复用 Trainer：

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
model.save_pretrained("outputs/qlora_model/adapter")
tokenizer.save_pretrained("outputs/qlora_model/adapter")
```

### 4.5.5 显存的量级理解

只看冻结基座权重时，理想化估算为：

```math
M_{\mathrm{fp16}}
\approx
2P
```

```math
M_{\mathrm{4bit}}
\approx
0.5P+M_{\mathrm{quant\ metadata}}
```

其中 $P$ 是参数量，按字节估算。真实训练显存还包括量化元数据、LoRA 参数及其梯度/optimizer state、激活和临时 buffer；若同一进程还做生成评估，才另有 KV cache。因此不能把 $0.5P$ 当作程序实际显存。

QLoRA 降低的是冻结基座的存储和训练状态，不会消除长序列激活，也不会保证吞吐一定高于普通 LoRA。量化 kernel 的速度取决于硬件、库版本和模型结构。

### 4.5.6 量化 adapter 的加载与部署

推理时可以继续以 4bit 加载 base，再挂载 adapter：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
from peft import PeftModel


base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    revision=model_revision,
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(
    base_model,
    "outputs/qlora_model/adapter",
)
model.eval()
```
如果需要合并，应根据部署目标加载合适精度的 base，并验证当前 PEFT/bitsandbytes 版本对该模型和量化布局的支持。不能假设任意 4bit 模型都能安全地原地 merge 后再保存。

### 4.5.7 纯 Python 的量化加低秩实验

下面用均匀量化模拟冻结基座和 LoRA。它不是 NF4 实现，只用于展示量化误差与可训练分支的关系：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import random


random.seed(5)
dimension = 12
rank = 2
scale = 2.0
weight = [
    [random.uniform(-0.3, 0.3) for _ in range(dimension)]
    for _ in range(dimension)
]


def quantize_row(row, levels=16):
    if not row:
        raise ValueError("cannot quantize an empty row")
    if not isinstance(levels, int) or levels < 2:
        raise ValueError("levels must be an integer greater than one")
    low = min(row)
    high = max(row)
    step = (high - low) / (levels - 1) if high != low else 1.0
    quantized = [round((value - low) / step) for value in row]
    return quantized, low, step


quantized_rows = []
metadata = []
for row in weight:
    quantized, low, step = quantize_row(row)
    quantized_rows.append(quantized)
    metadata.append((low, step))

quantized_weight = [
    [low + value * step for value in row]
    for row, (low, step) in zip(quantized_rows, metadata)
]
if not weight:
    raise ValueError("toy QLoRA experiment requires non-empty weight")
if rank <= 0 or rank > dimension:
    raise ValueError("rank must be in the interval [1, dimension]")
if scale <= 0:
    raise ValueError("scale must be positive")
frozen_before = [row[:] for row in quantized_weight]
quantization_error = sum(
    (left - right) ** 2
    for left_row, right_row in zip(weight, quantized_weight)
    for left, right in zip(left_row, right_row)
) / (dimension * dimension)

A = [
    [random.uniform(-0.02, 0.02) for _ in range(dimension)]
    for _ in range(rank)
]
B = [[0.0 for _ in range(rank)] for _ in range(dimension)]
data = []
for input_index, target_index in [(0, 4), (1, 5), (2, 6), (3, 7)]:
    vector = [0.0] * dimension
    target = [0.0] * dimension
    vector[input_index] = 1.0
    target[target_index] = 0.8
    data.append((vector, target))
if not data:
    raise ValueError("toy QLoRA experiment requires non-empty data")


def matvec(matrix, vector):
    return [
        sum(weight * value for weight, value in zip(row, vector))
        for row in matrix
    ]


def forward(vector):
    base = matvec(quantized_weight, vector)
    adapter = matvec(B, matvec(A, vector))
    return [left + scale * right for left, right in zip(base, adapter)]


def loss():
    if not data:
        raise ValueError("loss is undefined for an empty dataset")
    values = []
    for vector, target in data:
        prediction = forward(vector)
        values.append(
            sum((left - right) ** 2 for left, right in zip(prediction, target))
            / dimension
        )
    return sum(values) / len(values)


def train_step(learning_rate):
    grad_a = [[0.0] * dimension for _ in range(rank)]
    grad_b = [[0.0] * rank for _ in range(dimension)]
    for vector, target in data:
        prediction = forward(vector)
        error = [left - right for left, right in zip(prediction, target)]
        ax = matvec(A, vector)
        for out_index in range(dimension):
            for rank_index in range(rank):
                grad_b[out_index][rank_index] += (
                    scale * error[out_index] * ax[rank_index] / dimension
                )
        for rank_index in range(rank):
            upstream = sum(
                error[out_index] * B[out_index][rank_index]
                for out_index in range(dimension)
            )
            for in_index in range(dimension):
                grad_a[rank_index][in_index] += (
                    scale * upstream * vector[in_index] / dimension
                )
    for rank_index in range(rank):
        for in_index in range(dimension):
            A[rank_index][in_index] -= (
                learning_rate * grad_a[rank_index][in_index] / len(data)
            )
    for out_index in range(dimension):
        for rank_index in range(rank):
            B[out_index][rank_index] -= (
                learning_rate * grad_b[out_index][rank_index] / len(data)
            )


initial = loss()
for _ in range(500):
    train_step(8.0)
final = loss()
print("quantization_error=", round(quantization_error, 6))
print("initial_loss=", round(initial, 4))
print("final_loss=", round(final, 4))
print("loss_decreased=", final < initial)
print("frozen_base_changed=", quantized_weight != frozen_before)
print("adapter_params=", rank * (dimension + dimension))
```
这里的量化方式比 NF4 简单得多，不能用来复现 QLoRA 论文数值；它只说明量化基座可以保持不变，而低秩分支仍能学习一部分任务增量。

### 4.5.8 QLoRA 的工程边界

应明确记录：

```text
bitsandbytes、CUDA、PyTorch 和 Transformers 版本。
GPU 是否支持所选 compute dtype。
量化类型、double quant 和 device_map。
base revision、adapter 配置和训练 token 数。
训练与部署阶段是否使用相同量化布局。
```

4-bit 训练失败不一定是 labels 问题，kernel、驱动、模型层类型、dtype 和设备放置都可能导致失败。按依赖顺序定位：先单独加载量化基座并查看实际量化模块，再注入 adapter 并核对可训练参数，最后跑一个很小 batch。单卡训练由训练进程管理设备；多卡须使用训练框架支持的并行策略，不能把推理用的 `device_map="auto"` 当作 DDP/ZeRO 分片。

以 7B 参数为例，裸 FP16/BF16 权重约为 14 GB，4-bit 权重约为 3.5 GB，理想节省 10.5 GB；实际量化还需 scale 等元数据。若每 64 个权重共享一个 FP16 scale，每个权重平均还要约 $2/64=0.03125$ byte，7B 权重约有 1.094 亿组，scale 就约 219 MB。量化块、scale dtype、零点、未量化层和打包对齐会改变实际文件与显存，故这只是量级估算，不是 bitsandbytes 的精确预测。

均匀量化 toy 例可展示误差怎样传播：权重 $[0.1,0.4,0.9,-0.2]$ 映射到范围 $[-1,1]$ 的 16 个等距等级时，步长约为 $2/15=0.1333$；每个值被舍入后，误差会经矩阵乘法进入输出。这不是 NF4 的网格，不能拿来复现 QLoRA 数值。真实比较应固定 hidden states，先看线性层输出误差，再看完整任务回归；单层误差小不代表多层行为不变。

量化误差与 adapter 更新也不是互相抵消。量化矩阵 $\hat W$ 近似原权重 $W$，训练目标更新的是适配分支 $\Delta W$，不是专门拟合误差 $W-\hat W$；loss 下降不代表基座量化误差已被修复。比较同一 base 的 LoRA 与 QLoRA，应固定 adapter、数据和有效 token 预算，分别记录任务质量、峰值显存与吞吐。QLoRA 同时涉及 4-bit 基座、较高精度的 adapter、计算激活和可能更高精度的累加，排查数值问题应查看各模块实际 dtype，而不是笼统称模型“是 4-bit”。

实验记录至少保存量化类型、block size、double-quant 开关、compute dtype、跳过量化的模块、量化库版本、设备、模型 revision 和 adapter 配置。若输入输出头或敏感层保持较高精度，量化权重数就不等于参数总数乘 0.5 byte。配置文件、实际量化模块列表和 checkpoint 大小应一起留档。

量化和 adapter 解决的是资源与参数更新问题，仍不能说明回答变好。最后一篇用固定评估协议检查行为收益、退化和证据边界。
