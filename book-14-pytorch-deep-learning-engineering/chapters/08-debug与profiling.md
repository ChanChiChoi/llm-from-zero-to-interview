# 第八章：Debug 与 Profiling

深度学习工程能力的一大分水岭，不是能不能把模型跑起来，而是模型出问题时能不能快速定位。真实训练中常见问题包括 shape 不对、dtype 不对、device 不一致、loss 不下降、梯度为 None、NaN、OOM、DataLoader 太慢、GPU 利用率低、吞吐不稳定、验证和训练结果不一致、分布式卡住。Debug 和 profiling 的目标，就是把“感觉模型有问题”变成“定位到具体层、具体张量、具体 batch 或具体系统瓶颈”。

本章不追求列出所有 PyTorch 调试工具，而是建立一套优先级清楚的排查方法：先确认数据和 shape，再确认 loss 和梯度，再确认数值稳定，再看显存，再看性能瓶颈。很多时候，系统化排查比背 API 更重要。

## 8.1 Debug 的基本原则

遇到训练问题时，不要一上来改模型结构或调学习率。更稳的顺序是：

1. 先复现。
2. 再缩小问题范围。
3. 打印关键张量元信息。
4. 验证数据和 label。
5. 验证 loss 是否合理。
6. 验证梯度是否存在且有限。
7. 再考虑优化器、学习率、混合精度和分布式。

一个好的 debug 习惯是：每次只改一个变量，并记录修改前后的现象。否则你可能让问题暂时消失，但不知道真正原因。

面试回答：

```text
我 debug 训练问题时会先缩小范围：确认数据样本和 label 是否正确，再检查 tensor 的 shape、dtype、device，然后看 loss 是否依赖参数、梯度是否为 None 或 NaN，最后再看优化器、学习率、混合精度、显存和分布式通信。不要一开始就盲目改模型或调参。
```

## 8.2 打印 tensor 元信息

很多 bug 一打印 tensor 元信息就能定位。

```python
import torch


def debug_tensor(name, x):
    if not torch.is_tensor(x):
        print(name, type(x))
        return
    print(
        name,
        "shape=", tuple(x.shape),
        "dtype=", x.dtype,
        "device=", x.device,
        "requires_grad=", x.requires_grad,
        "is_contiguous=", x.is_contiguous(),
    )
```

使用：

```python
debug_tensor("input_ids", batch["input_ids"])
debug_tensor("labels", batch["labels"])
debug_tensor("logits", logits)
debug_tensor("loss", loss)
```

最先检查：

1. shape 是否符合预期。
2. dtype 是否符合模块要求。
3. device 是否一致。
4. loss 是否是标量。
5. logits 和 labels 是否能对齐。

## 8.3 Shape debug

Shape 错误是最常见的 PyTorch bug。

语言模型里常见 shape：

```text
input_ids: [B, T]
labels: [B, T]
hidden_states: [B, T, d]
logits: [B, T, V]
```

计算 loss 前：

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]

debug_tensor("shift_logits", shift_logits)
debug_tensor("shift_labels", shift_labels)
```

然后展平：

```python
loss = torch.nn.functional.cross_entropy(
    shift_logits.reshape(-1, shift_logits.size(-1)),
    shift_labels.reshape(-1),
    ignore_index=-100,
)
```

常见错误：

1. logits 是 `[B, T, V]`，labels 是 `[B, T]`，但忘记展平。
2. 展平后 logits 的第 0 维和 labels 的第 0 维不一致。
3. attention scores 期望 `[B, H, T, T]`，mask 写成了 `[B, T, 1]`。
4. `view` 用在非 contiguous tensor 上报错。

调试建议：把每个模块的输入输出 shape 都打印一次，直到发现第一个不符合预期的位置。

## 8.4 Dtype debug

Dtype 错误常见于 embedding、loss、mask 和混合精度。

常见规则：

1. `input_ids` 通常是 `torch.long`。
2. `labels` 通常是 `torch.long`。
3. logits 通常是浮点类型。
4. attention mask 可以是 bool，也可以转换成加性浮点 mask。
5. 模型参数可能是 fp32、fp16 或 bf16。

典型报错：

```text
expected scalar type Long but found Float
```

通常表示你把 float tensor 传给了需要索引的模块，例如 `nn.Embedding`。

检查代码：

```python
for key, value in batch.items():
    debug_tensor(key, value)
```

混合精度下，某些操作可能因为 dtype 不一致报错。优先确认：

1. 模型参数 dtype。
2. 输入 dtype。
3. autocast 作用域是否正确。
4. loss 是否在合理 dtype 下计算。

## 8.5 Device debug

Device mismatch 是另一个高频错误。

典型报错：

```text
Expected all tensors to be on the same device
```

排查：

```python
print("model device:", next(model.parameters()).device)
for key, value in batch.items():
    if torch.is_tensor(value):
        print(key, value.device)
```

常见原因：

1. 模型在 GPU，batch 还在 CPU。
2. 某个 buffer 没有注册，`model.to(device)` 没移动它。
3. 在 forward 中新建 tensor 时没指定 device。
4. loss 里用到的 class weight 还在 CPU。

错误示例：

```python
mask = torch.ones(seq_len, seq_len)  # 默认在 CPU
scores = scores.masked_fill(mask == 0, -1e9)
```

更安全：

```python
mask = torch.ones(seq_len, seq_len, device=scores.device)
```

或者注册为 buffer。

## 8.6 NaN 和 Inf debug

NaN 是训练中最危险的问题之一。先判断 NaN 出现在：

1. 输入数据。
2. logits。
3. loss。
4. 梯度。
5. 参数。

检查函数：

```python
def check_finite(name, x):
    if torch.is_tensor(x) and not torch.isfinite(x).all():
        print("non-finite:", name)
        print("nan:", torch.isnan(x).any().item())
        print("inf:", torch.isinf(x).any().item())
```

训练中使用：

```python
check_finite("logits", logits)
check_finite("loss", loss)
```

检查梯度：

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        check_finite("grad " + name, param.grad)
```

常见原因：

1. 学习率太大。
2. fp16 溢出。
3. softmax 前 logits 过大。
4. mask 产生整行 `-inf`。
5. loss 分母为 0。
6. 数据中有异常值。
7. 梯度爆炸。

## 8.7 anomaly detection

PyTorch 提供 anomaly detection，能帮助定位 backward 报错来源。

```python
torch.autograd.set_detect_anomaly(True)
```

或者：

```python
with torch.autograd.detect_anomaly():
    loss.backward()
```

它会让 PyTorch 尝试报告导致反向异常的前向位置。

注意：

1. 它会明显降低速度。
2. 只适合 debug，不适合长期训练。
3. 它不一定能定位所有 NaN 根因，但对 in-place 和 backward 异常很有帮助。

## 8.8 梯度 debug

loss 不下降时，先确认梯度是否存在。

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        if param.grad is None:
            print("grad is None:", name)
        else:
            print(name, param.grad.norm().item())
```

常见问题：

1. 参数没有进入 optimizer。
2. 参数 `requires_grad=False`。
3. loss 不依赖该参数。
4. 中间用了 `detach()`。
5. forward 分支没有用到该模块。

梯度全 0 可能说明：

1. 激活饱和。
2. mask 把 loss 全忽略了。
3. loss 写错。
4. 学习率不是原因，梯度本身就没有信号。

梯度非常大可能说明：

1. 学习率过大。
2. 初始化不稳定。
3. 数据异常。
4. loss scale 或混合精度异常。
5. 需要梯度裁剪。

## 8.9 hook 调试中间激活

如果怀疑某一层输出异常，可以用 forward hook。

```python
activations = {}


def save_activation(name):
    def hook(module, inputs, output):
        if torch.is_tensor(output):
            activations[name] = output.detach().cpu()
    return hook


handle = model.layers[0].register_forward_hook(save_activation("layer0"))
```

运行一次 forward 后：

```python
print(activations["layer0"].shape)
handle.remove()
```

注意：

1. 保存激活时要 `detach()`。
2. 用完 hook 要 `remove()`。
3. 不要长期保存大激活。
4. 分布式和 compile 场景下 hook 可能增加复杂度。

## 8.10 OOM debug

OOM 需要区分是“瞬间峰值太高”还是“显存持续增长”。

打印显存：

```python
if torch.cuda.is_available():
    print("allocated", torch.cuda.memory_allocated() / 1024**2)
    print("reserved", torch.cuda.memory_reserved() / 1024**2)
```

常见原因：

1. batch size 太大。
2. sequence length 太长。
3. 没开混合精度。
4. 没开 activation checkpointing。
5. 保存了带图 tensor。
6. 验证时没关梯度。
7. DataLoader 或评估代码缓存过多输出。

排查顺序：

1. 减小 batch size。
2. 减小 sequence length。
3. 关掉额外输出，例如 hidden states 和 attentions。
4. 检查是否保存了 loss、logits、activation tensor。
5. 开启 AMP 和 checkpointing。

## 8.11 DataLoader 性能 debug

GPU 利用率低，不一定是模型慢，可能是 DataLoader 喂不饱 GPU。

现象：

1. GPU 利用率周期性掉到很低。
2. 每步训练时间波动很大。
3. CPU 占用很高。
4. 数据读取线程成为瓶颈。

排查：

1. 增加 `num_workers`。
2. 使用 `pin_memory=True`。
3. 减少 `__getitem__` 中的重预处理。
4. 把可离线处理的 tokenization 提前做掉。
5. 检查远程存储或磁盘 IO。

简单计时：

```python
import time

start = time.time()
for step, batch in enumerate(loader):
    if step == 100:
        break
print("data time", time.time() - start)
```

如果纯 DataLoader 迭代都很慢，瓶颈不在模型。

## 8.12 性能 profiling 的基本思路

性能优化前要先 profiling。不要凭感觉优化。

先区分瓶颈：

1. DataLoader 慢。
2. CPU 到 GPU 拷贝慢。
3. forward 慢。
4. backward 慢。
5. optimizer step 慢。
6. 分布式通信慢。

最简单的计时方式：

```python
torch.cuda.synchronize()
start = time.time()

outputs = model(**batch)
loss = outputs["loss"]
loss.backward()
optimizer.step()

torch.cuda.synchronize()
elapsed = time.time() - start
```

为什么要 `torch.cuda.synchronize()`？因为 CUDA 操作默认异步，如果不同步，Python 计时可能不准确。

## 8.13 torch.profiler 入门

PyTorch 提供 `torch.profiler` 用于分析 CPU/GPU 时间。

简化示例：

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for step, batch in enumerate(train_loader):
        if step >= 5:
            break
        train_step(batch)
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

你可以看到：

1. 哪些算子耗时最多。
2. CPU 和 CUDA 时间分布。
3. shape 信息。
4. 显存分配情况。

profiler 本身有开销，不要在完整训练中一直开。

## 8.14 常见性能瓶颈

### 8.14.1 小 batch 导致 GPU 利用率低

小 batch 下 GPU 可能吃不满。解决思路：

1. 增大 batch。
2. 梯度累积。
3. 合并小 kernel。
4. 使用更高效算子。

### 8.14.2 DataLoader 慢

解决思路：

1. 预处理离线化。
2. 调整 `num_workers`。
3. 使用 pinned memory。
4. 减少 Python 层复杂逻辑。

### 8.14.3 CPU-GPU 拷贝慢

解决思路：

1. `pin_memory=True`。
2. `.to(device, non_blocking=True)`。
3. 避免每步创建大量小 tensor。

### 8.14.4 通信慢

分布式中常见。解决思路：

1. 增大计算通信比。
2. 梯度累积配合 `no_sync()`。
3. 检查网络和 NCCL。
4. 减少不必要同步。

## 8.15 训练结果异常的系统排查

### 8.15.1 loss 不下降

排查：

1. 数据和 labels 是否正确。
2. loss 是否对齐任务。
3. 参数是否进入 optimizer。
4. 梯度是否存在。
5. 学习率是否合理。
6. mask 是否把有效 token 忽略。
7. 模型是否处于 train 模式。

### 8.15.2 train loss 降，val loss 不降

可能原因：

1. 过拟合。
2. 训练集和验证集分布不同。
3. 验证代码有 bug。
4. train/eval 模式切换错误。
5. 数据泄漏或评估指标错误。

### 8.15.3 训练很快但效果很差

可能原因：

1. loss mask 错误。
2. labels 全是 `-100` 或大部分被忽略。
3. 数据被截断得太严重。
4. tokenizer 或 chat template 不一致。
5. 学习率或 batch 配置不合理。

## 8.16 分布式 debug

分布式问题最重要的是确认每个 rank 的状态。

打印：

```python
print(
    "rank", dist.get_rank(),
    "world_size", dist.get_world_size(),
    "device", torch.cuda.current_device(),
)
```

常见问题：

1. 某些 rank 没进入同一个 collective。
2. 某个 rank 数据加载失败。
3. 只有 rank 0 创建了文件，其他 rank 读不到。
4. 每个 rank batch 数不一致。
5. 日志混在一起看不清。

建议：

1. 日志带 rank。
2. 只让 rank 0 保存 checkpoint。
3. 用小数据、小 batch 先复现。
4. 必要时用 `dist.barrier()` 定位卡点。

## 8.17 一个实用 debug checklist

遇到训练异常，可以按这个顺序查：

1. 打印一个原始样本。
2. 打印 collate 后 batch。
3. 检查 input、labels、mask 的 shape 和 dtype。
4. 跑一个 batch 的 forward。
5. 检查 logits shape。
6. 检查 loss 是否有限。
7. backward 后检查梯度是否存在。
8. 检查 optimizer 是否包含目标参数。
9. 检查学习率和 scheduler。
10. 检查 train/eval/no_grad 使用是否正确。
11. 检查显存是否持续增长。
12. 如果慢，再做 profiler。

这个顺序的核心是从数据到模型、从正确性到性能，不要先优化性能再修正确性。

## 8.18 一个最小可复现 bug 的写法

当你遇到复杂 bug，最有效的方法是构造最小复现。

步骤：

1. 固定随机种子。
2. 使用极小模型。
3. 使用 1-2 个 batch 的假数据。
4. 去掉分布式和混合精度。
5. 保留能复现 bug 的最少代码。

示例：

```python
torch.manual_seed(0)

model = TinyModel()
batch = {
    "input_ids": torch.randint(0, 100, (2, 8)),
    "labels": torch.randint(0, 100, (2, 8)),
}

outputs = model(**batch)
loss = outputs["loss"]
loss.backward()
```

如果最小例子不复现，说明问题可能来自数据、分布式、混合精度、DataLoader 或训练循环外围逻辑。

## 8.19 面试官会怎么问

### 问题一：loss 不下降你怎么排查？

回答模板：

```text
我会先看数据和 label 是否正确，再看 logits 和 labels 的 shape 是否对齐，loss 是否有限且依赖参数。然后检查梯度是否为 None、0、NaN，参数是否进入 optimizer，学习率是否合理，mask 是否把有效 token 忽略。最后再看模型容量、初始化和数据分布问题。
```

### 问题二：训练出现 NaN 怎么排查？

回答模板：

```text
先定位 NaN 出现在输入、logits、loss、梯度还是参数。常见原因包括学习率过大、fp16 溢出、softmax 或 mask 产生异常、loss 分母为 0、数据异常和梯度爆炸。我会记录 loss、grad norm、lr，并用 check_finite 或 anomaly detection 缩小范围。
```

### 问题三：GPU 利用率低怎么排查？

回答模板：

```text
先区分是 DataLoader 慢、CPU-GPU 拷贝慢、模型计算小、频繁同步还是分布式通信慢。可以单独测试 DataLoader 迭代速度，用 cuda synchronize 做分段计时，再用 torch.profiler 看 CPU/CUDA 时间分布。
```

### 问题四：OOM 怎么排查？

回答模板：

```text
先确认是训练还是验证 OOM，再看 batch size、sequence length、是否开启 AMP、是否保存了带图 tensor、验证是否关闭梯度、是否输出 hidden states 或 attentions。然后考虑 activation checkpointing、梯度累积和减少缓存。
```

### 问题五：torch.profiler 能看什么？

回答模板：

```text
torch.profiler 可以统计 CPU 和 CUDA 算子的耗时、调用次数、shape 和显存信息，帮助定位耗时最多的算子、CPU/GPU 时间分布、DataLoader 或拷贝瓶颈。但 profiler 有额外开销，通常只在少量 step 上开启。
```

## 8.20 常见误区

1. 一看到 loss 不下降就改模型结构，不先查数据和 loss。
2. 只打印 loss，不打印 shape、dtype、device。
3. 把中间 tensor 存进 list，导致显存泄漏。
4. 验证时忘记 `no_grad`，误以为模型太占显存。
5. CUDA 异步计时不加 `synchronize`，导致性能判断错误。
6. profiler 一直开，导致训练本身变慢。
7. 分布式日志不带 rank，导致错误信息无法定位。
8. 用 `find_unused_parameters=True` 掩盖模型分支 bug。

## 8.21 小练习

1. 写一个 `debug_tensor` 函数，打印 shape、dtype、device 和 contiguous 状态。
2. 构造一个 shape mismatch 的 loss 例子，并修复它。
3. 构造一个 device mismatch 的例子，并修复它。
4. 写一个 `check_finite` 函数，检查 logits、loss 和 grad。
5. 写一个 forward hook，保存某层输出并在使用后 remove。
6. 对同一个训练 step 用 `torch.cuda.synchronize()` 做分段计时。
7. 用 `torch.profiler` 跑 5 个 step，打印耗时最多的算子。
8. 写一份 OOM 排查 checklist。
9. 写一份 loss 不下降排查 checklist。

## 8.22 本章总结

Debug 和 profiling 的核心是把问题分层。先确认数据、shape、dtype、device，再确认 loss、梯度和参数更新，然后看 NaN、OOM 和分布式同步，最后才做性能 profiling。

需要记住：

1. 大多数训练 bug 可以先从一个 batch 的 shape、dtype、device 查起。
2. loss 不下降要先查数据、loss、梯度和 optimizer。
3. NaN 要定位到输入、logits、loss、梯度或参数中的哪一层。
4. OOM 要区分显存峰值过高和持续泄漏。
5. CUDA 计时要考虑异步执行，必要时 `synchronize()`。
6. `torch.profiler` 用于定位性能瓶颈，但不要长期打开。
7. 分布式 debug 要按 rank 打日志，确认所有 rank 进入相同同步点。

下一章会进入实现 Transformer 组件，用前面学到的 tensor、Module、训练循环和 debug 方法手写 attention、MLP、block、mask 和最小 decoder-only 模型。
