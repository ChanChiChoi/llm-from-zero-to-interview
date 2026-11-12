# 第八章：Debug 与 Profiling

深度学习工程能力的一大分水岭，不是能不能把模型跑起来，而是模型出问题时能不能快速定位。真实训练中常见问题包括 shape 不对、dtype 不对、device 不一致、loss 不下降、梯度为 None、NaN、OOM、DataLoader 太慢、GPU 利用率低、吞吐不稳定、验证和训练结果不一致、分布式卡住。Debug 和 profiling 的目标，就是把“感觉模型有问题”变成“定位到具体层、具体张量、具体 batch 或具体系统瓶颈”。

本章不追求列出所有 PyTorch 调试工具，而是建立一套优先级清楚的排查方法：先确认数据和 shape，再确认 loss 和梯度，再确认数值稳定，再看显存，再看性能瓶颈。很多时候，系统化排查比背 API 更重要。

## 0. 本章范围与资料

本章参考已核对 PyTorch 官方 `torch.autograd.detect_anomaly` / `set_detect_anomaly`、CUDA semantics、CUDA memory management、`torch.cuda.memory_allocated` / `memory_reserved`、`torch.cuda.synchronize`、`torch.profiler`、PyTorch profiler recipe、Performance Tuning Guide 和 DataLoader 相关文档口径。

本章聚焦 PyTorch 训练调试和性能定位中最常见的问题：shape / dtype / device、loss 与 label 对齐、NaN / Inf、梯度为 None 或异常、forward hook、OOM、DataLoader 瓶颈、CPU-GPU 拷贝、CUDA 异步计时、`torch.profiler` 基础用法、分布式日志和最小可复现 bug。

本章给出 `torch.compile` 的入门核验流程：与 eager 对照输出、预热后测稳态、查看 graph break 与重编译；不展开 Inductor 内部 profiler、CUDA kernel 级优化、Nsight Systems / Nsight Compute、NCCL 深度 profiling、生产级 observability 平台、分布式 trace 聚合或复杂性能建模。这些内容会在 AI Infra、推理框架和性能优化章节中继续展开。

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

可以把一次排查写成四列记录：

| 假设 | 最小实验 | 观测 | 下一步 |
| --- | --- | --- | --- |
| 标签错位 | 固定一个 batch，打印 shift 前后的一对 token | 目标 token 与输入不对应 | 修正 collate 或 shift |
| 梯度断开 | 只跑 forward/backward，检查参数 grad | 某层 grad 为 `None` | 查 `detach`、分支和 optimizer |
| 数据管线慢 | 不跑模型，只遍历 100 个 batch | batch 等待时间占比高 | 查 worker、IO、collate |
| CUDA kernel 慢 | 固定输入，分别测 forward/backward，并同步设备 | 某阶段占据墙钟时间 | 用 profiler 细分算子 |

这张表的价值在于让“怀疑”变成一次有对照组的实验。没有对照组时，改变 batch、学习率、精度和模型结构后看到 loss 变化，无法判断哪一个改变产生了效果。

排查不是“把所有检查都打开”，而是为一个可证伪的假设收集最小证据。例如“loss 不下降”至少可能来自标签错位、loss 没连到参数、学习率不合适或验证口径错误；每一次实验都应只改变一个因素，并记录样本、随机种子、配置和结果。修复后还要用原始最小复现和一个回归 batch 重跑，确认问题消失且没有引入新问题。

### 8.1.1 关键公式与 debug/profiling 速查

第一，调试一个 tensor 时不要只看数值，还要看元信息：

~~~math
m(X)=(\mathrm{shape}(X),\mathrm{dtype}(X),\mathrm{device}(X),\mathrm{requires\_grad}(X),\mathrm{stride}(X))
~~~

shape / dtype / device 错误往往比模型结构错误更常见。

第二，causal LM loss 展平前后必须满足：

~~~math
N_{\mathrm{logit}} = B(T-1),\qquad N_{\mathrm{label}} = B(T-1)
~~~

也就是 `shift_logits.reshape(-1, V)` 的第 0 维必须和 `shift_labels.reshape(-1)` 的长度一致。

第三，非有限值检查可以抽象为：

~~~math
I_{\mathrm{finite}}(X)=\prod_i I(|x_i|<\infty)
~~~

如果 logits、loss、grad 或参数中任一关键张量 `I_finite=0`，应先定位非有限值第一次出现的位置，再调学习率或改模型。

第四，梯度检查常看每个参数的梯度范数：

~~~math
G_l=\|\nabla_{\theta_l} L\|_2
~~~

`G_l=None`、`G_l=0`、`G_l` 非有限或异常大，分别对应断图、无信号、数值错误或梯度爆炸等不同问题。

第五，OOM 要区分峰值过高和持续增长：

~~~math
\Delta M_t=M_t-M_{t-1}
~~~

如果每步结束后 `M_t` 持续上升，常见原因是保存了带计算图的 tensor、验证没关梯度或评估缓存过多输出；如果只有峰值超限，常见方向是 batch、sequence length、activation、optimizer state 和临时 workspace。

第六，训练吞吐要先约定分子。若目标是衡量实际参与监督的 token，可以写成：

~~~math
R_{\mathrm{valid\ token/s}}=\frac{N_{\mathrm{valid\ target}}}{t_{\mathrm{step}}}
~~~

其中 `N_valid_target` 是经过 causal shift 和 ignore mask 后实际参与 loss 的目标数，`t_step` 是同一统计窗口的墙钟时间。若只想报 padded token/s，可另用 `B×T/t_step`，但不要把它称为有效 token/s。优化前要先区分数据加载、拷贝、forward、backward、optimizer 和通信分别占多少。

第七，CUDA 计时需要同步：

~~~math
t_{\mathrm{cuda}}=t_{\mathrm{end\ after\ sync}}-t_{\mathrm{start\ after\ sync}}
~~~

因为 CUDA kernel 默认异步执行，不同步时 Python 侧计时可能只测到 launch 开销。

第八，profiler 可按算子汇总 self time 和调用次数；CUDA 的峰值显存要另按同一观测窗口取 high-water mark：

~~~math
T^{\mathrm{self}}_{k}=\sum_{j=1}^{n_k}t^{\mathrm{self}}_{k,j},\qquad M_{\mathrm{peak}}=\max_{t\in[t_0,t_1]}M_{\mathrm{allocated}}(t)
~~~

其中 `k` 是某个算子 key，`n_k` 是该 key 在 trace 中的调用数。不同算子的 CUDA 时间可能重叠，`key_averages()` 的行也可能显示 inclusive total；不能把所有行相加当作 step 墙钟时间。真正要看的是瓶颈归因：DataLoader、CPU-GPU copy、forward、backward、optimizer、通信，还是少数高频小算子。若要判断峰值显存，应在目标区间前重置 peak、执行同一批工作并同步后读取；profiler 的 allocation 事件不是显存峰值的替代品。

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
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


使用：

```python
debug_tensor("input_ids", batch["input_ids"])
debug_tensor("labels", batch["labels"])
debug_tensor("logits", logits)
debug_tensor("loss", loss)
```
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


最先检查：

1. shape 是否符合预期。
2. dtype 是否符合模块要求。
3. device 是否一致。
4. loss 是否是标量。
5. logits 和 labels 是否能对齐。

元信息检查应尽量在错误第一次出现的边界完成，而不是等到 loss 抛异常才检查。例如 embedding 前检查 token dtype，attention 前检查 `Q/K/V` shape 和 mask broadcast 形状，loss 前检查 shift 后有效 label 数。越晚检查，越多中间结果可能已经掩盖根因。

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
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


然后展平：

```python
loss = torch.nn.functional.cross_entropy(
    shift_logits.reshape(-1, shift_logits.size(-1)),
    shift_labels.reshape(-1),
    ignore_index=-100,
)
```
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


常见错误：

1. logits 是 `[B, T, V]`，labels 是 `[B, T]`，但忘记展平。
2. 展平后 logits 的第 0 维和 labels 的第 0 维不一致。
3. attention scores 期望 `[B, H, T, T]`，mask 写成了 `[B, T, 1]`。
4. `view` 用在非 contiguous tensor 上报错。

调试建议：把每个模块的输入输出 shape 都打印一次，直到发现第一个不符合预期的位置。

把一个小 batch 的 shape contract 手算一遍，能分清“张量长度匹配”和“监督语义匹配”。取 `B=1`、`T=4`、词表大小 `V=6`，输入和原始 labels 都是 `[2,5,3,0]`，其中最后一个 label 是 padding，改为 `-100`。logits 形状是 `[1,4,6]`；丢弃最后一个预测位置后，`shift_logits` 是 `[1,3,6]`；labels 向左移一位后得到 `[5,3,-100]`，形状 `[1,3]`。展平以后 logits 是 `[3,6]`，labels 是 `[3]`，因此行数相同，但只有前两个位置参加 loss，第三个是忽略项。

再检查 loss 的分母：假设前两行 softmax 对各自目标类别的概率分别为 `0.5` 和 `0.25`，交叉熵是 `-ln(0.5)=0.6931` 与 `-ln(0.25)=1.3863`；`-100` 位置不计入均值，最终是 `(0.6931+1.3863)/2=1.0397`。如果误把输入长度 4 当目标数，或把 padding 位置也纳入分母，算出来的标量会偏小；shape 全部能广播并不能证明 label 对齐正确。

不要长期在生产 forward 中打印全部张量。更可维护的做法是把 shape contract 写成断言，并在最小复现中打开：

```python
assert input_ids.ndim == 2, input_ids.shape
assert logits.ndim == 3, logits.shape
assert logits.shape[:2] == input_ids.shape, (logits.shape, input_ids.shape)
assert shift_logits.reshape(-1, logits.size(-1)).shape[0] == shift_labels.numel()
```
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


断言失败时保留 batch id、序列长度和 tokenizer 版本，避免只得到一条没有上下文的 shape 错误。

把这些断言放到数据入口、模型边界和 loss 入口时，日志能指出错误第一次出现的层次：若 collate 后 labels 已错，模型不应背锅；若 batch 正确而 logits 时间维改变，再查 attention、cache 或模型输出契约；若 logits 对但 shift 后 target 数变为 0，就检查 padding 与 ignore mask。在线上每步对整个大 tensor 做 `.all()` 可能触发设备同步，因此正式训练通常保留廉价的 shape / dtype 检查，并把逐值检查放在小批量复现中。

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
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


混合精度下，某些操作可能因为 dtype 不一致报错。优先确认：

1. 模型参数 dtype。
2. 输入 dtype。
3. autocast 作用域是否正确。
4. loss 是否在合理 dtype 下计算。

dtype 正确也不意味着数值正确。`torch.long` 的 label 可以合法地包含越界类别 id，bool mask 也可能在广播后覆盖了错误的维度；所以 dtype 检查应和取值范围、shape contract 一起做。对类别 label，可以加上：

```python
valid = labels.ne(-100)
assert bool(((labels[valid] >= 0) & (labels[valid] < vocab_size)).all())
```
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


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
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


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
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


更安全：

```python
mask = torch.ones(seq_len, seq_len, device=scores.device)
```
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


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
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


训练中使用：

```python
check_finite("logits", logits)
check_finite("loss", loss)
```
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


检查梯度：

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        check_finite("grad " + name, param.grad)
```
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


常见原因：

1. 学习率太大。
2. fp16 溢出。
3. softmax 前 logits 过大。
4. mask 产生整行 `-inf`。
5. loss 分母为 0。
6. 数据中有异常值。
7. 梯度爆炸。

排查 NaN 时要找“第一次出现”，而不是只在最终 loss 处打印。可以把 forward 拆成 embedding、norm、attention、MLP、logits 五个边界；对每个边界记录 `isfinite`、最小值、最大值和均值。若 logits 已经非有限，loss 函数不是根因；若 logits 有限但 loss 非有限，检查 mask、label 范围和 reduction；若 loss 有限但 backward 后 grad 非有限，检查低精度、scale、异常梯度和裁剪顺序。

实际操作时不必每一层都保存完整 tensor。固定同一个失败 batch，在五个边界各记录一次“是否有限、首个异常索引、有限元素的最小/最大值”；从输入向后移动检查点，直到找到第一个由全有限变成非有限的位置。比如输入和 attention 前的 hidden 都有限，而某个 attention row 的 scores 全是 `-inf`，随后 softmax 才出现 NaN，那么问题在 mask 构造或可见 key 为空，降低学习率并不能修复这个结构性错误。CUDA 上把 `.all()` 转成 Python bool 会等待设备结果，逐层检查会扰动时序，所以只在最小复现中打开。

全屏蔽 attention 是一个特殊反例：如果一整行 logits 都被填成 `-inf`，softmax 可能产生 NaN；如果一个样本的所有 label 都是 `-100`，mean reduction 可能出现无有效分母。debug 代码应明确区分“没有有效监督”与“loss 恰好为 0”，前者通常应跳过或报错，不应静默参与更新。

## 8.7 anomaly detection

PyTorch 提供 anomaly detection，能帮助定位 backward 报错来源。

```python
torch.autograd.set_detect_anomaly(True)
```
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


或者：

```python
with torch.autograd.detect_anomaly():
    logits = model(batch["input_ids"])
    loss = compute_loss(logits, batch["labels"])
    loss.backward()
```
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


它会让 PyTorch 尝试报告导致反向异常的前向位置。

注意：

1. 它会明显降低速度。
2. 只适合 debug，不适合长期训练。
3. 它不一定能定位所有 NaN 根因，但对 in-place 和 backward 异常很有帮助。

`detect_anomaly` 更适合定位“哪个前向操作为 backward 保存了坏值”或“哪个 backward Function 抛了异常”，不是性能 profiler，也不是所有非有限值的自动修复器。一个实用流程是先用 `check_finite` 缩小到某个 batch/阶段，再只在该最小复现上打开 anomaly detection；否则全量训练会被它的额外检查拖慢，日志也会淹没真正的错误。

上下文必须包住前向和反向：异常检测需要在前向时记录创建 backward 节点的位置，若只在已构造好的 `loss.backward()` 外套 context，通常拿不到前向 traceback。它的 `check_nan=True` 针对反向计算产生的 NaN；前向 logits 已经非有限时，仍要用前面的逐阶段有限值检查定位。先用普通代码确认输入和输出，再对该 batch 临时启用，不要将 anomaly mode 留在整轮训练里。

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
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


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

可用一个两参数梯度向量校准范数的含义：若 `grad=[3,4]`，二范数是 `sqrt(3²+4²)=5`；这不是“参数变化量”，而是当前 loss 对该参数块的局部敏感度。`None` 表示当前反向没有为该参数生成 `.grad`，全零表示 `.grad` 存在但这一批给出的信号为零；两种情况都要再查该参数是否在 optimizer 的 param groups、是否被冻结以及本轮 optimizer step 是否真的执行。只看整体 loss 有限或梯度范数非零，无法证明每个目标参数都被更新。

还应把“梯度为 `None`”和“梯度为全 0”分开记录。前者通常说明参数没有参与当前计算图，后者可能是有效的零梯度，也可能是 mask、饱和或数值下溢。检查完梯度后再比较 optimizer step 前后的参数差异：有梯度不代表参数一定会变，参数组、冻结状态、学习率和 GradScaler 是否跳过 step 都会影响更新。

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
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


运行一次 forward 后：

```python
print(activations["layer0"].shape)
handle.remove()
```
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


注意：

1. 保存激活时要 `detach()`。
2. 用完 hook 要 `remove()`。
3. 不要长期保存大激活。
4. 分布式和 compile 场景下 hook 可能增加复杂度。

hook 是观察工具，不是模型逻辑。它可能改变内存峰值和执行时序，因此用 hook 定位到异常后，应移除 hook 再重跑最小复现；如果问题消失，先检查 hook 是否保存了未 detach 的输出或改变了引用生命周期。对大模型，优先记录摘要统计和少量样本，不要把每一层完整激活搬到 CPU。

上面的 `output.detach().cpu()` 适合只采一次的小模型观察；放到每步训练会复制激活并可能等待 GPU，测到的时间和峰值显存就不再是原始训练。若只关心 shape，可在 hook 中只保存 `tuple(output.shape)`；若要检查具体值，采样一个 batch 后记录有限值和极值，立即移除 handle 并释放保存的对象。对于返回 tuple/dict 的模块，hook 还要按实际输出结构取目标 tensor，不能假设每层都直接返回单个 tensor。

## 8.10 OOM debug

OOM 需要区分是“瞬间峰值太高”还是“显存持续增长”。

打印显存：

```python
if torch.cuda.is_available():
    print("allocated MiB", torch.cuda.memory_allocated() / 1024**2)
    print("reserved MiB", torch.cuda.memory_reserved() / 1024**2)
```
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


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

把 OOM 报告中的三个时刻分开：当前分配量 `allocated`、allocator 保留量 `reserved` 和历史峰值 `peak`。若 `allocated` 在每个 iteration 结束后上升，优先查计算图引用；若 `allocated` 稳定但 `reserved` 较高，可能只是缓存；若只在某个长序列 batch 突然峰值超限，优先查样本长度、padding、attention 中间量和临时 workspace。每次实验都应固定数据顺序，否则不同长度 batch 可能让结论失真。

用张量元素数估一个显存下界，能先判断是不是尺寸账本问题。一个 FP32 logits tensor 若形状为 `[B=4,T=2048,V=32000]`，含 `4×2048×32000=262,144,000` 个元素，每个 4 bytes，占 `1,048,576,000` bytes，约 `0.977 GiB`。若代码把 10 个带图 logits 留在 list 中，仅这些输出张量就约 `9.77 GiB`，还没算 Autograd 保存的中间激活和 allocator 开销。若只要记标量，应存 `loss.detach().item()`；若需复用完整预测，应先明确是否可以移到 CPU、降采样或只保留评估窗口。这个算术预算不等于 CUDA 峰值，但能快速指出“缓存对象本身已太大”。

测峰值时要重置窗口，并在结束后等待 CUDA 完成：

```python
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
train_step(fixed_batch)
torch.cuda.synchronize()
peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
print("step peak MiB", peak_mib)
```
> 代码片段，静态审阅，未运行；需要 CUDA 设备和本节的 `fixed_batch`、`train_step`。


逐步比较显存时，先规定采样点，例如 optimizer step 后、梯度释放后，再对同一组固定 shape 的 batch 重复记录。每步结尾的 allocated 连续上涨才是引用累积的信号；单独看到 reserved 比 allocated 大，通常只说明缓存分配器留有可复用内存，不足以证明泄漏。`max_memory_allocated` 只涵盖 PyTorch allocator 管理的 tensor，不包括所有外部 CUDA 分配。

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

测稳态 DataLoader 等待时间时，先启动 iterator 并丢掉少量预热 batch，再固定测量窗口；这样可避免把 worker 首次启动混进每个配置的稳态值，也避免 off-by-one：

```python
import time
from itertools import islice

iterator = iter(loader)
for _ in islice(iterator, 10):
    pass

start = time.perf_counter()
measured_batches = sum(1 for _ in islice(iterator, 100))
elapsed = time.perf_counter() - start
if measured_batches:
    print("batches", measured_batches, "mean ms", elapsed * 1000 / measured_batches)
```
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


如果纯 DataLoader 迭代都很慢，瓶颈不在模型。

这测的是 batch 产出间隔，不是每个 worker 的 CPU 利用率，也不代表端到端训练一定能按相同比例提速。若 iterator 总共不到 110 个 batch，实际测得的数目会更少，必须同时报告 `measured_batches`；若随机裁剪或变长样本使 batch 成本差别大，应固定数据顺序并分长度桶统计。增加 `num_workers`、打开 `pin_memory` 或 `non_blocking=True` 都是候选实验，不是默认的加速保证；worker 启动、共享内存、RAM 和设备复制可能反而成为新成本。

端到端 step 时间可以拆成：

~~~math
t_{\mathrm{step}}=t_{\mathrm{wait}}+t_{\mathrm{copy}}+t_{\mathrm{compute}}+t_{\mathrm{sync}}+t_{\mathrm{log}}.
~~~

其中各项可能重叠，所以这是诊断账本而不是严格可加的物理定律。CPU 计时测到的是提交和等待行为；GPU 计时必须在正确边界同步。要比较 DataLoader 配置，至少固定 batch 数、预处理逻辑、存储位置和 worker 生命周期，避免把第一次启动 worker 的开销误认为稳定吞吐。

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
for _ in range(warmup_steps):
    train_step(fixed_batch)

torch.cuda.synchronize()
start = time.perf_counter()

for _ in range(measured_steps):
    train_step(fixed_batch)

torch.cuda.synchronize()
elapsed = time.perf_counter() - start
mean_ms = elapsed * 1000 / measured_steps
```
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


为什么要 `torch.cuda.synchronize()`？因为 CUDA 操作默认异步，如果不同步，Python 计时可能不准确。

一个常见误区是把 `synchronize()` 加在每个小算子周围，然后用得到的时间判断真实训练性能。同步会破坏原本可能存在的 kernel overlap，只适合做局部测量；端到端吞吐应使用稳定的 warmup 后窗口，并只在窗口边界同步。上面的片段固定重复同一个 batch，量的是模型训练 step，不含 DataLoader 等待；若要测端到端吞吐，应在另一组计时中遍历真实 loader，并明确是否包含数据读取、拷贝、日志和分布式同步。CPU-only 环境不能用 CPU 时间推断 CUDA kernel 性能，必须在目标 GPU 上复测。

把 token 分子也带入一次计算：假设 batch 有 8 条、每条 padding 后长度 1,024、单步墙钟时间为 0.20 秒。按 padded slot 算是 `8×1,024/0.20=40,960` token/s；如果 shift 和 padding mask 后只有 75% 的位置是有效目标，则有效目标数为 `8×1,024×0.75=6,144`，有效 token 吞吐是 `6,144/0.20=30,720` token/s。两种数都能用于诊断，但比较不同 padding 策略时应优先报告有效 token/s，并固定序列长度分布与 batch 组成。

## 8.13 torch.profiler 入门

PyTorch 提供 `torch.profiler` 用于分析 CPU/GPU 时间。

简化示例：

```python
import torch.profiler

activities = [torch.profiler.ProfilerActivity.CPU]
sort_by = "self_cpu_time_total"
if torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)
    sort_by = "cuda_time_total"

with torch.profiler.profile(
    activities=activities,
    record_shapes=True,
    profile_memory=True,
) as prof:
    for step, batch in enumerate(train_loader):
        if step >= 5:
            break
        train_step(batch)
        prof.step()

print(prof.key_averages().table(sort_by=sort_by, row_limit=20))
```
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


你可以看到：

1. 哪些算子耗时最多。
2. CPU 和 CUDA 时间分布。
3. shape 信息。
4. 显存分配情况。

profiler 本身有开销，不要在完整训练中一直开。

一个 profiler trace 只能说明被记录窗口里的行为。应先 warmup，再用 `schedule(wait=..., warmup=..., active=...)` 采集少量稳定 step，并在每个 step 调用 `prof.step()`；如果没有 step 标记，调度器可能一直停留在错误阶段。`record_shapes=True` 和 `profile_memory=True` 会增加开销，只有需要对应证据时才打开。Profiler 还可能显示算子耗时，却无法单独证明“应该换哪个模型结构”；优化结论仍需在关闭 profiler 后用独立 benchmark 验证。

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
) as prof:
    for step, batch in enumerate(loader):
        train_step(batch)
        prof.step()
        if step >= 3:
            break
```
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


按 `wait=1, warmup=1, active=2` 展开，一共至少需要四个 `prof.step()`：第 0 步不记录，第 1 步做预热并丢弃，第 2、3 步生成 trace。把 active 设成 20 却只跑 4 步，不会得到完整采样；若要比较多次运行，还应把 trace 保存到不同目录并记录 PyTorch、设备、shape、dtype、batch 和是否编译。读取 trace 时先看调用数、self time 和等待区间，再回到独立计时确认热点是否真的影响墙钟。

## 8.14 常见性能瓶颈

### 8.14.1 小 batch 导致 GPU 利用率低

小 batch 下 GPU 可能吃不满。解决思路：

1. 增大 batch。
2. 梯度累积。
3. 合并小 kernel。
4. 使用更高效算子。

小 batch 常让 kernel launch 和 Python 调度成本占比变大，但“GPU 利用率低”也可能来自每步工作本来就少、输入等待或同步空隙。先固定模型与有效 token 总量，比较单步算子时间、每秒有效 token 和每次更新的样本/token 数；梯度累积能改变更新看到的数据量，却不一定把小矩阵变成高效的大矩阵，也可能增加每轮 backward 的调度成本。只有同一工作量下的稳态测量显示计算吞吐提升，才有依据保留 batch 改动。

### 8.14.2 DataLoader 慢

解决思路：

1. 预处理离线化。
2. 调整 `num_workers`。
3. 使用 pinned memory。
4. 减少 Python 层复杂逻辑。

将数据准备从模型计算中剥离出来测量：若预热后纯 loader 的中位 batch 等待时间已接近或超过模型 step 的时间，才优先优化解码、读取和 collate；如果 loader 很快而整步慢，应继续拆分 device copy、forward/backward 和通信。worker 数从 0 加到 8 并不必然更快：小数据集的进程启动和 IPC 可能超过并行收益，随机预处理还可能受 CPU 线程争用影响。每次只改 worker、pinning 或预处理中的一个因素，并报告相同 batch 数和样本长度分布。

### 8.14.3 CPU-GPU 拷贝慢

解决思路：

1. `pin_memory=True`。
2. `.to(device, non_blocking=True)`。
3. 避免每步创建大量小 tensor。

`pin_memory=True` 为 host-to-device 搬运提供 pinned source；配合 `.to(device, non_blocking=True)` 允许异步提交，但是否能与计算重叠还取决于 stream、依赖和输入准备时间。若复制后立即在 CPU 上读取结果，或默认 stream 的依赖迫使设备等待，异步参数不会自动缩短总墙钟。测量时至少分别记录拷贝和完整 step，并确认 pinned memory 没有让主机 RAM 压力升高。

### 8.14.4 通信慢

分布式中常见。解决思路：

1. 增大计算通信比。
2. 梯度累积配合 `no_sync()`。
3. 检查网络和 NCCL。
4. 减少不必要同步。

多卡 step 时间受最慢 rank 限制：即使平均通信时间不高，单个 rank 的长序列、慢数据读取或额外分支也会让其他 rank 等在 collective。先逐 rank 对齐 `data_wait`、forward、backward、通信尾部和 optimizer step；只有通信占比确实高，才实验累积窗口、bucket 或拓扑。细节回看第七章的 reducer、`no_sync()` 和 rank 日志主讲；Profiler trace 中某个 NCCL kernel 的 self time不能替代全局同步尾部。

### 8.14.5 `torch.compile` 的正确性与性能核验

`torch.compile` 把可捕获的 PyTorch 计算交给编译路径执行。默认遇到部分无法跟踪的 Python 代码时，Dynamo 可以结束当前图、在 eager 中运行该段，再继续追踪；这通常不是程序报错，但会减少一部分融合和优化机会。graph break 的影响并不相同：发生在模型计算中段、频繁返回 Python 的断点可能比输入预处理处的单个断点更显著，所以排查目标是能解释的性能热点，不是追求 trace 上的 break 数为零。

先分开验证三个问题：第一，在固定输入上 eager 与 compiled 输出是否满足当前精度的容差；第二，代表性的 shape、dtype、stride 是否引发重编译；第三，排除编译冷启动后稳态 step 是否更快。初始化和前几次调用可能包含编译成本，若把它们算进一次短 benchmark，就可能得出相反结论。序列长度或其他 guard 输入变化时也可能触发新编译，因此预热和计时要覆盖实际会出现的 shape 桶，而不是只测一个最短样本。

下面用一个冻结参数的前向示例隔离编译路径，测试 eager/compiled 的输出并用 `torch.utils.benchmark.Timer` 做稳态对照。该计时适合小型前向微基准；要决定训练是否变快，还需用相同 batch、梯度、optimizer、数据管线和运行窗口测完整训练 step。

```python
import torch
from torch import nn
import torch.utils.benchmark as benchmark

device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(
    nn.Linear(128, 256),
    nn.GELU(),
    nn.Linear(256, 10),
).to(device).eval()
model.requires_grad_(False)
x = torch.randn(64, 128, device=device)
compiled_model = torch.compile(model)

eager_output = model(x)
compiled_output = compiled_model(x)  # 首次调用可能包含编译时间。
torch.testing.assert_close(compiled_output, eager_output, rtol=1e-4, atol=1e-5)

compiled_model(x)  # 同 shape 再运行一次，避免把首次编译算入计时。
eager_time = benchmark.Timer(
    stmt="model(x)",
    globals={"model": model, "x": x},
    num_threads=1,
).blocked_autorange(min_run_time=0.5)
compiled_time = benchmark.Timer(
    stmt="compiled_model(x)",
    globals={"compiled_model": compiled_model, "x": x},
    num_threads=1,
).blocked_autorange(min_run_time=0.5)
print("eager median ms", eager_time.median * 1000)
print("compiled median ms", compiled_time.median * 1000)
```
> 代码已验证（2026-11-11；Python 3.12.9 / PyTorch 2.12.0+cpu；CPU；eager 与 compiled 输出断言通过。玩具前向计时不代表一般加速）。


在 PyTorch 2.15 中，`fullgraph=True` 可作为诊断开关：遇到 graph break 时让该函数报错，从而定位必须检查的边界；它不表示模型一定更快，也不一定是所有生产代码都该采用的模式。小例子可用 `TORCH_LOGS="graph_breaks,recompiles"` 查看断点位置/原因和 guard 失效触发的重编译。先在 eager 路径运行同一输入排除普通 Python 错误，再用日志确认是数据相关分支、暂不支持的代码还是代码本身错误；修复后重新比较数值与目标 shape 的编译次数。

若模型已由 DDP/FSDP 等分布式 wrapper 包装，2.15 troubleshooting 文档建议先评估编译内部 module，而不是把 wrapper 本身直接交给 `torch.compile`；这一建议是版本化入口，不是所有组合的兼容性保证。第七章负责说明 wrapper 的通信边界，本节只要求在相同的分布式拓扑和训练输入下重新验证输出、collective 顺序和稳态性能。

手算一个只用于说明指标的假设结果：如果同一个稳态窗口中 eager median 是 `1.20 ms`，compiled median 是 `0.90 ms`，speedup 为 `1.20/0.90≈1.33×`，延迟下降比例则是 `(1.20-0.90)/1.20=25%`。这是由假设测量值算出的指标关系，不是本章机器实测或 PyTorch 的加速承诺；还必须记录模型、输入 shape、设备、dtype、编译 backend、预热、计时次数及误差范围。

```bash
TORCH_LOGS="graph_breaks,recompiles" python train.py
```
> 命令片段，静态审阅，未运行；需在项目训练入口和 PyTorch 2.15 环境中执行。


日志选项属于定位工具，可能产生大量输出并增加运行开销；诊断结束后应关闭，再以无 profiler、无 verbose 编译日志的配置测量性能。若 compiled 输出不满足容差，先用固定输入、检查随机层/训练状态和 eager 结果，再缩小到模型子模块；不要仅凭 profiler 更快或 trace 较短就宣称训练正确或整体提速。

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

把判断顺序落到一个固定小实验里：抽 1–8 个样本，保存输入、labels 和 mask，关闭随机增强和复杂 scheduler，连续做少量真实 optimizer step，并记录每步 loss、grad norm、学习率及一两个参数的变化。如果样本数很少而 loss 仍不动，先检查 loss 对参数的梯度、参数是否在 optimizer、`zero_grad/backward/step` 顺序和 mask；如果 loss 明显下降而验证不升，再转去检查切分、正则、训练/验证模式和指标定义。这个小集拟合实验用于验证计算路径，不用来证明泛化能力。

先做小数据过拟合实验：固定 1--8 个样本，关闭随机增强和复杂调度，观察模型能否把训练 loss 明显压低。若不能，优先怀疑数据、loss、梯度或 optimizer；若能而验证集不升，才进入泛化、切分和正则化分析。每一步保存固定 batch 的 logits 摘要和参数差异，防止把评估代码的旧输出误认为训练结果。

### 8.15.2 train loss 降，val loss 不降

可能原因：

1. 过拟合。
2. 训练集和验证集分布不同。
3. 验证代码有 bug。
4. train/eval 模式切换错误。
5. 数据泄漏或评估指标错误。

验证不升还可能是 reduction 口径不同：训练按有效 token 平均，验证却按 batch 平均；或分布式验证只打印 rank 0 的本地结果。先统一分母和切片，再讨论过拟合。对生成任务，固定 prompt、检索结果、temperature、top-p 和随机种子；否则每次验证的生成差异会被误判成模型回退。

例如，验证 batch A 有 2 个有效目标、mean loss 为 2；batch B 有 6 个有效目标、mean loss 为 1。若把两个 batch mean 等权平均，报告 `(2+1)/2=1.5`；若目标是按 token 统计，正确值是 `(2×2+6×1)/(2+6)=1.25`。两种算法都可能“稳定下降”，但前者给短 batch 的单个 token 更大权重。比较 train/val 前应确认二者对 ignore mask、shift、权重和全局分子/分母采用相同口径，并记录有效项数。

### 8.15.3 训练很快但效果很差

可能原因：

1. loss mask 错误。
2. labels 全是 `-100` 或大部分被忽略。
3. 数据被截断得太严重。
4. tokenizer 或 chat template 不一致。
5. 学习率或 batch 配置不合理。

“训练很快”也要先问一个 update 实际看到了多少有效监督。padding 过多、labels 大量为 `-100` 或 batch sampler 重复样本，都会让 step/s 看起来不错却使有效 token/s 很低；若数据和 loss 确认无误，再核对训练 token 总量、更新次数、目标 tokenizer/chat template 和评测集是否对应同一任务。报告应同时给 optimizer step、有效 token、loss 分母和训练时长，不能只说跑了多少个 epoch。

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
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


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

更可靠的日志事件至少包含 `rank`、`epoch`、`global_step`、`micro_step`、batch id、collective 名称和设备。遇到 hang 时先确认哪个 rank 最后打印了什么事件，再判断是数据读取、Python 异常、CUDA kernel 还是 collective 等待。不要在每个路径随意加 barrier：它可能把一个原本暴露的顺序 bug 隐藏起来，也会改变通信与计算重叠。

用一条短轨迹演练判断：假设 rank 0、1、3 都打印 `step=20 before_backward`，rank 2 的最后一条则是 `step=20 decode_error`，其余 rank 随后在 reducer 同步等待。最早可观察的异常在 rank 2 的数据路径；先核查该 batch 与异常退出是否被所有进程看见，再处理通信超时。若四个 rank 都打印了相同 collective 的 before 但没有 after，才进一步查 backend、设备映射、tensor shape 和网络。第七章[分布式训练入门](07-分布式训练入门.md)主讲了 `join()` 与 collective 顺序；本节沿用同一日志字段做定位，不把工具 debug 日志当作业务错误记录的替代品。

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

按顺序检查并不意味着每项都要打开最重的工具。第一步打印一个样本即可；只有发现 shape/label 违约才拓展到整批统计；loss 正常但更新异常，再检查梯度和 optimizer；正确性都通过后才量时间和显存。这样可限制每个实验的成本，也减少同步、hook 或 profiler 改变问题现场的机会。若第一次失败已经定位到数据 decode，就先修复并用原 batch 回归，不需要同时跑完整 profile。

完成修复后再做三类回归：

1. 同一个最小复现应不再失败。
2. 正常 batch 的 loss、梯度和参数更新仍满足原 contract。
3. 关闭 debug hook、anomaly detection 和 profiler 后，吞吐与显存回到可接受范围。

否则“debug 通过”可能只是因为打开了某个工具改变了执行路径。

## 8.18 一个最小可复现 bug 的写法

当你遇到复杂 bug，最有效的方法是构造最小复现。

步骤：

1. 固定随机种子。
2. 使用极小模型。
3. 使用 1-2 个 batch 的假数据。
4. 去掉分布式和混合精度。
5. 保留能复现 bug 的最少代码。

缩小过程要保留失败的必要条件，而不是盲目把数据改成“更简单”。例如错位 bug 依赖 `labels[:,1:]` 的 mask，就不能把 padding 全删掉；多卡步调问题依赖 rank2 缺一个 batch，就不能在复现里让所有 rank 都迭代相同次数。每删去一项依赖都重跑，只有现象还在才继续删，最终把环境版本、随机种子、输入 shape、预期、实际与触发命令记在同一文件中。

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
> 代码片段，静态审阅，未运行；按本节上下文和输入约定使用，目标 API 为 PyTorch 2.15。


如果最小例子不复现，说明问题可能来自数据、分布式、混合精度、DataLoader 或训练循环外围逻辑。

最小复现还应把“预期”和“实际”写成断言，而不是只贴一段日志。例如固定 `loss.shape == []`、有效 label 数大于 0、某个参数梯度非空、参数更新量非零；这样环境升级后，示例会在行为变化的位置失败，而不是继续输出看似正常的数字。

用一个可复算的断言说明它和 traceback 的差别：若固定输入为 `[B=2,T=8]`、忽略标签数为 3，则 shift 后目标位置是 `2×7=14`，有效标签应为 `14-3=11`。若程序得到 10，shape 仍可能完全合法，但说明 mask 或切片少保留了一个监督目标。把 `assert valid_count == 11` 放进最小复现，得到的是稳定的语义回归检查；anomaly detection 则用于 autograd 在 backward 中抛错或生成 NaN 时追溯前向节点。两者解决不同问题。

## 8.19 最小可运行 debug/profiling 审计 demo

下面这个 demo 在 CPU 上即可运行。它把本章的核心检查点串起来：tensor 元信息、causal LM shift loss 对齐、finite 检查、forward hook、梯度范数、optimizer 参数覆盖、简单 step 计时，以及可选的 `torch.profiler` 入口。把完整代码保存为 `debug_audit.py` 后，默认用 `python debug_audit.py` 执行；将 `RUN_TORCH_PROFILER` 设为 `1` 会在 CPU 上多采一小段 profiler trace。模型 vocab size 是 16，固定输入与 labels 都落在这个范围内，第二行最后一个 label 用 `-100` 表示忽略。

当前机器如果 CUDA driver 不匹配，`torch.profiler` 可能产生底层驱动探测日志。为了让章节 demo 输出稳定，下面默认不启动 profiler；需要真实 profiler 时，把环境变量 `RUN_TORCH_PROFILER=1` 打开即可。

这个例子首先回答“一个固定 batch 是否按约定经过模型并产生有效梯度”，而不是测模型能力或真实训练速度。它含 2 条长度为 4 的样本；shift 后每条最多 3 个监督位置，第二条最后一项被 mask，因此有效目标数应为 5。运行后，metadata 的 shape 从 `[2,4]` 到 `[2,4,16]`，再到 `[2,3,16]` 和 `[2,3]`；flatten 对齐条件为 `6` 行，loss 分母则是 5 个有效 label。读者可先手算这两个数量，再对照输出定位是 shape 还是 mask 出错。

```python
import os
import time
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore", message="CUDA initialization.*")

import torch
import torch.nn.functional as F
from torch import nn


def debug_tensor(name, x):
    return {
        "name": name,
        "shape": tuple(x.shape),
        "dtype": str(x.dtype).replace("torch.", ""),
        "device": x.device.type,
        "requires_grad": bool(x.requires_grad),
        "contiguous": bool(x.is_contiguous()),
    }


def finite_status(name, x):
    finite = torch.isfinite(x)
    return {
        "name": name,
        "all_finite": bool(finite.all()),
        "nan": bool(torch.isnan(x).any()),
        "inf": bool(torch.isinf(x).any()),
    }


class TinyLM(nn.Module):
    def __init__(self, vocab_size=16, hidden_size=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        hidden = self.norm(self.embed(input_ids))
        return self.head(hidden)


torch.manual_seed(0)
model = TinyLM()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
batch = {
    "input_ids": torch.tensor([[1, 2, 3, 4], [2, 3, 4, 0]], dtype=torch.long),
    "labels": torch.tensor([[1, 2, 3, 4], [2, 3, 4, -100]], dtype=torch.long),
}
activation_shapes = {}


def hook(module, inputs, output):
    activation_shapes["norm"] = tuple(output.shape)


handle = model.norm.register_forward_hook(hook)
optimizer.zero_grad(set_to_none=True)
logits = model(batch["input_ids"])
shift_logits = logits[:, :-1, :]
shift_labels = batch["labels"][:, 1:]
loss = F.cross_entropy(
    shift_logits.reshape(-1, shift_logits.size(-1)),
    shift_labels.reshape(-1),
    ignore_index=-100,
)
loss.backward()
grad_norms = {
    name: round(float(param.grad.norm()), 4)
    for name, param in model.named_parameters()
    if param.grad is not None
}
optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
all_trainable_in_optimizer = all(
    id(param) in optimizer_param_ids
    for param in model.parameters()
    if param.requires_grad
)
handle.remove()

bad_logits = torch.tensor([[0.0, float("inf")], [float("nan"), 1.0]])
nonfinite_report = finite_status("bad_logits", bad_logits)

timing_start = time.perf_counter()
for _ in range(5):
    timed_logits = model(batch["input_ids"])
    timed_loss = timed_logits.sum()
    timed_loss.backward()
    model.zero_grad(set_to_none=True)
elapsed_ms = round((time.perf_counter() - timing_start) * 1000, 3)

profiler_enabled = os.environ.get("RUN_TORCH_PROFILER") == "1"
profiler_summary = {"enabled": profiler_enabled, "event_count_positive": False, "has_addmm": False}
if profiler_enabled:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        prof_logits = model(batch["input_ids"])
        prof_loss = prof_logits.sum()
        prof_loss.backward()
    events = prof.key_averages()
    profiler_summary = {
        "enabled": True,
        "event_count_positive": len(events) > 0,
        "has_addmm": any("addmm" in event.key for event in events),
    }

valid_label_count = int((shift_labels != -100).sum())
metadata = [
    debug_tensor("input_ids", batch["input_ids"]),
    debug_tensor("logits", logits),
    debug_tensor("shift_logits", shift_logits),
    debug_tensor("shift_labels", shift_labels),
]
checks = {
    "loss_is_scalar": loss.ndim == 0,
    "shift_flat_match": shift_logits.reshape(-1, shift_logits.size(-1)).shape[0]
    == shift_labels.reshape(-1).shape[0],
    "valid_labels_exist": valid_label_count > 0,
    "loss_finite": bool(torch.isfinite(loss)),
    "grads_present": set(grad_norms)
    == {"embed.weight", "norm.weight", "norm.bias", "head.weight", "head.bias"},
    "optimizer_covers_trainable": all_trainable_in_optimizer,
    "hook_removed": len(model.norm._forward_hooks) == 0,
    "detects_nonfinite": not nonfinite_report["all_finite"],
    "timing_positive": elapsed_ms > 0,
}

print("metadata=", metadata)
print("valid_label_count=", valid_label_count)
print("loss=", round(float(loss.detach()), 4))
print("activation_shapes=", activation_shapes)
print("grad_norms=", grad_norms)
print("nonfinite_report=", nonfinite_report)
print("elapsed_ms_positive=", elapsed_ms > 0)
print("profiler_summary=", profiler_summary)
print("checks=", checks)
```
> 代码已验证（2026-11-11；Python 3.12.9 / PyTorch 2.12.0+cpu；CPU；默认路径与 `RUN_TORCH_PROFILER=1` 检查通过；未验证 CUDA）。


期望输出：

```text
metadata= [{'name': 'input_ids', 'shape': (2, 4), 'dtype': 'int64', 'device': 'cpu', 'requires_grad': False, 'contiguous': True}, {'name': 'logits', 'shape': (2, 4, 16), 'dtype': 'float32', 'device': 'cpu', 'requires_grad': True, 'contiguous': True}, {'name': 'shift_logits', 'shape': (2, 3, 16), 'dtype': 'float32', 'device': 'cpu', 'requires_grad': True, 'contiguous': False}, {'name': 'shift_labels', 'shape': (2, 3), 'dtype': 'int64', 'device': 'cpu', 'requires_grad': False, 'contiguous': False}]
valid_label_count= 5
loss= 3.1516
activation_shapes= {'norm': (2, 4, 8)}
grad_norms= {'embed.weight': 0.2852, 'norm.weight': 0.2617, 'norm.bias': 0.3396, 'head.weight': 1.7006, 'head.bias': 0.5929}
nonfinite_report= {'name': 'bad_logits', 'all_finite': False, 'nan': True, 'inf': True}
elapsed_ms_positive= True
profiler_summary= {'enabled': False, 'event_count_positive': False, 'has_addmm': False}
checks= {'loss_is_scalar': True, 'shift_flat_match': True, 'valid_labels_exist': True, 'loss_finite': True, 'grads_present': True, 'optimizer_covers_trainable': True, 'hook_removed': True, 'detects_nonfinite': True, 'timing_positive': True}
```

这段输出要检查几件事：

1. `shift_logits` 和 `shift_labels` 第一维展平后能对齐，并且有效 label 数不是 0。
2. `loss_finite=True` 说明当前 batch 的 loss 没有 NaN / Inf。
3. `grad_norms` 覆盖了所有可训练参数，说明 loss 确实连到了参数。
4. `nonfinite_report` 能发现手工构造的 NaN / Inf，说明 finite 检查函数有效。
5. `hook_removed=True` 避免 forward hook 长期挂在模型上。
6. `elapsed_ms_positive=True` 说明至少可以做基础 step timing；真实 GPU timing 还需要 CUDA synchronize。
7. `profiler_summary.enabled=False` 是默认稳定运行模式；需要真实 profiler 时打开环境变量再看 CPU 算子事件。

固定种子下，当前 Python 3.12.9 / PyTorch 2.12.0+cpu 环境的参考 loss 是 `3.1516`；其他 PyTorch 版本或设备可能因初始化与浮点运算细节略有差异，所以回归优先比较 shape、有效标签数、finite 状态、参数覆盖和容差内结果，不要求跨版本逐位相同。`elapsed_ms_positive=True` 只是确认计时代码执行，不是吞吐指标；计时循环包含 5 次 CPU forward/backward，重复固定 batch，也没有 optimizer step、DataLoader 或 CUDA kernel。

默认 profiler 输出应是 `enabled=False`；开启 `RUN_TORCH_PROFILER=1` 后，在本地 CPU 环境应看到正的 event count 和 `aten::addmm` 事件。进程日志可能包含 profiler 提示，不属于稳定的章节输出。即使 CPU trace 正常，也只证明 profiling API 在这个小例子中能返回事件，不代表设备 kernel 的真实时间、CUDA 路径或完整训练 loop 的瓶颈。

## 8.20 常见误区

1. 一看到 loss 不下降就改模型结构，不先查数据和 loss。
2. 只打印 loss，不打印 shape、dtype、device。
3. 把中间 tensor 存进 list，导致显存泄漏。
4. 验证时忘记 `no_grad`，误以为模型太占显存。
5. CUDA 异步计时不加 `synchronize`，导致性能判断错误。
6. profiler 一直开，导致训练本身变慢。
7. 分布式日志不带 rank，导致错误信息无法定位。
8. 用 `find_unused_parameters=True` 掩盖模型分支 bug。
9. 把 `torch.compile` 第一次调用的编译冷启动算进 steady-state，或把每个 graph break 都当成必须清零的错误。
10. 把 profiler 汇总的 operator time 相加当作整步墙钟，或把 profiler trace 的运行时间当正式 benchmark。

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
10. 用一个故意错位的 shift loss 构造最小复现，并让 shape/label 断言在错误位置失败。
11. 构造全 `-100` labels，区分“没有有效监督”与“有限的零损失”。
12. 用 hook 找到第一个非有限激活，再移除 hook 验证问题仍可复现。
13. 对 DataLoader、forward、backward 和 optimizer 分段计时，比较 warmup 前后结果。
14. 用 profiler schedule 采集 2 个 active step，并说明 trace 不能代表完整训练。
15. 对固定模型和输入比较 eager / `torch.compile` 输出，给出容差；在预热后用 `torch.utils.benchmark.Timer` 测稳态，再用 `TORCH_LOGS="graph_breaks,recompiles"` 解释变化的 shape 是否导致重编译。

## 8.22 资料与证据边界

本章关于调试、异常检测、CUDA 计时、Profiler 和性能调优的 API 语义，优先依据 PyTorch 官方资料：

1. [Autograd anomaly detection（PyTorch 2.15）](https://docs.pytorch.org/docs/2.15/autograd.html#anomaly-detection)
2. [Profiler API（PyTorch 2.15）](https://docs.pytorch.org/docs/2.15/profiler.html)
3. [Profiler recipe（PyTorch 2.15 tutorial）](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
4. [Benchmark Utils 与 `torch.utils.benchmark.Timer`（PyTorch 2.15）](https://docs.pytorch.org/docs/2.15/benchmark_utils.html)
5. [CUDA semantics 与异步执行（PyTorch 2.15）](https://docs.pytorch.org/docs/2.15/notes/cuda.html#asynchronous-execution)
6. [CUDA memory management（PyTorch 2.15）](https://docs.pytorch.org/docs/2.15/notes/cuda.html#memory-management)
7. [`torch.compile` troubleshooting（PyTorch 2.15）](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_troubleshooting.html)
8. [`fullgraph=True` graph-break 诊断（PyTorch 2.15）](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html)
9. [DataLoader API（PyTorch 2.15）](https://docs.pytorch.org/docs/2.15/data.html)

这些资料可以确认工具调用约定；计时、吞吐、GPU 利用率、DataLoader 瓶颈和显存峰值仍必须在目标硬件、模型、数据和编译配置上实测。章节中的 CPU demo 验证的是 shape/loss/finite/hook/梯度和 profiler 接口逻辑，不代表 CUDA kernel 性能。教程构造的 NaN、OOM 和 rank hang 只是用于理解因果链，不能代替目标系统的故障复盘。

## 8.23 本章总结

Debug 和 profiling 的核心是把问题分层。先确认数据、shape、dtype、device，再确认 loss、梯度和参数更新，然后看 NaN、OOM 和分布式同步，最后才做性能 profiling 或编译优化。

需要记住：

1. 大多数训练 bug 可以先从一个 batch 的 shape、dtype、device 查起。
2. loss 不下降要先查数据、loss、梯度和 optimizer。
3. NaN 要定位到输入、logits、loss、梯度或参数中的哪一层。
4. OOM 要区分显存峰值过高和持续泄漏。
5. CUDA 计时要考虑异步执行，必要时 `synchronize()`。
6. `torch.profiler` 用于定位性能瓶颈，但不要长期打开。
7. 分布式 debug 要按 rank 打日志，确认所有 rank 进入相同同步点。
8. `torch.compile` 要先核对数值，再覆盖代表性 shape 预热和测稳态；graph break 与重编译是诊断信号，不是性能结论。

下一章会进入实现 Transformer 组件，用前面学到的 tensor、Module、训练循环和 debug 方法手写 attention、MLP、block、mask 和最小 decoder-only 模型。
