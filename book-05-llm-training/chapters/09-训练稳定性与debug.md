# 第九章：训练稳定性与 Debug

## 本章目标

掌握大模型训练中常见异常的定位方法。

## 核心议题

1. loss spike。
2. NaN 和 Inf。
3. 梯度爆炸和梯度消失。
4. 数据异常导致训练异常。
5. 学习率过大或 warmup 不足。
6. 混合精度溢出。
7. 分布式同步错误。

## Debug 顺序

1. 检查数据 batch。
2. 检查 loss mask。
3. 检查学习率和 optimizer。
4. 检查梯度 norm。
5. 检查数值精度。
6. 检查分布式通信和 checkpoint 恢复。

## 面试重点

训练 debug 题要结构化回答，不要只说“调学习率”。

## 为什么训练 debug 很难

大模型训练的问题通常不是单点问题。

同一个现象可能来自不同层：

```text
loss spike
  可能是坏数据
  可能是学习率过大
  可能是混合精度溢出
  可能是分布式同步异常
  可能是 checkpoint 恢复错误
```

所以训练 debug 的关键不是背几个修复技巧，而是建立排查顺序。

一个可靠原则是：

```text
先定位问题发生在哪一层，再决定怎么修。
```

训练系统可以分成这些层：

1. 数据层。
2. Tokenization 和 sample 构造层。
3. Loss mask 层。
4. 模型 forward/backward 层。
5. 优化器和学习率层。
6. 数值精度层。
7. 分布式通信层。
8. checkpoint 和恢复层。

面试表达：训练 debug 要从数据、loss、梯度、优化器、精度、分布式和恢复状态逐层排查，而不是直接调学习率。

### 训练稳定性 Debug 的资料边界

按照 `WRITING_PLAN.md` 的要求，本章第二轮精修前核对了 PyTorch autograd anomaly detection、`clip_grad_norm_` 的 non-finite 检查、PyTorch AMP / `GradScaler`、Hugging Face Trainer 日志指标，以及前序训练循环、优化器、分布式训练、显存和数值精度章节资料。

本章聚焦训练稳定性排查的工程闭环：loss spike、NaN / Inf、grad norm、数据 batch、label mask、学习率、optimizer state、mixed precision、分布式 rank 状态和 checkpoint 恢复。它不展开特定训练平台的监控系统、生产告警规则、NCCL 深层故障诊断和模型架构级稳定性理论。

### 训练健康指标口径

每个训练 step 可以记录一组健康指标：

```math
h_t=
(L_t,L_{\mathrm{val},t},G_t,\eta_t,S_t,R_t,U_t)
```

其中 `L_t` 是 train loss，`L_val,t` 是 validation loss，`G_t` 是 grad norm，`eta_t` 是 learning rate，`S_t` 是 loss scale，`R_t` 是 tokens/sec，`U_t` 是 GPU utilization 或类似系统利用率。debug 的目标是从时间序列 `h_t` 中找到第一次异常：

```math
t^\*=
\min\{t\mid G_{\mathrm{health}}(h_t)=0\}
```

一个简化健康门禁可以写成：

```math
G_{\mathrm{health}}=
g_{\mathrm{loss}}\,
g_{\mathrm{grad}}\,
g_{\mathrm{data}}\,
g_{\mathrm{mask}}\,
g_{\mathrm{lr}}\,
g_{\mathrm{precision}}\,
g_{\mathrm{rank}}\,
g_{\mathrm{ckpt}}
```

这些 `g` 都是 0/1 检查项：loss 有限且无异常 spike、梯度范数可控、batch 数据正常、label mask 正确、学习率和 warmup 符合预期、精度没有 overflow / underflow、分布式 rank 状态一致、checkpoint 恢复字段完整。

## 1. 先建立健康训练的基线

Debug 的前提是知道“正常训练”应该长什么样。

一个健康训练过程通常有这些特征：

1. train loss 总体下降，允许小幅波动。
2. validation loss 同步下降或稳定改善。
3. learning rate 按 schedule 正常 warmup 和 decay。
4. grad norm 在合理范围内波动。
5. 没有持续 NaN / Inf。
6. tokens/sec 相对稳定。
7. GPU utilization 没有长时间掉到很低。
8. checkpoint 能保存并恢复。

如果你没有这些基线指标，出现问题时就很难判断异常来自哪里。

## 2. Debug 总顺序

推荐的排查顺序：

```text
1. 复现问题：确认是偶发还是稳定复现。
2. 定位时间：找到第一次异常的 step。
3. 定位层级：数据、loss、梯度、优化器、精度、分布式。
4. 缩小规模：用小模型、小 batch、单卡复现。
5. 对比基线：和上一个健康配置或 checkpoint 对比。
6. 单变量实验：一次只改一个因素。
7. 修复后回归：确认旧问题消失且没有引入新问题。
```

不要一上来同时改学习率、batch size、数据、精度和并行策略。这样即使训练好了，也不知道是哪一个因素起作用。

## 3. Loss spike

Loss spike 是大模型训练中非常常见的异常。

表现：

```text
loss 突然升高
grad norm 同步变大
有时会恢复，有时进入 NaN
```

### 3.1 可能原因

| 原因 | 说明 |
| --- | --- |
| 坏数据 batch | 极长、乱码、异常 token、错误 label |
| 学习率过大 | 更新步子太大，参数跳到不稳定区域 |
| warmup 太短 | 训练初期还不稳定就给了高 lr |
| 梯度爆炸 | grad norm 突然变大 |
| FP16 overflow | 数值溢出导致 loss 异常 |
| checkpoint 恢复不完整 | optimizer/scheduler 状态错位 |

### 3.2 排查方法

1. 找到 spike 的 step。
2. 保存该 step 的 batch id 或样本。
3. 检查该 batch 的长度、token 分布、特殊 token、label mask。
4. 查看 grad norm 是否同步升高。
5. 查看 learning rate 是否刚到 peak。
6. 检查是否从 checkpoint 恢复后不久发生。

一个实用的 spike 判定不要只看单点 loss，而要和前面一小段窗口比较。设最近 `k` 个健康 step 的 loss 中位数为基线：

```math
r_t=
\frac{L_t}
{\mathrm{median}(L_{t-k},\ldots,L_{t-1})+\epsilon}
```

当下面条件成立时，可以先把该 step 标成候选异常点：

```math
r_t\ge \tau_{\mathrm{spike}}
```

其中 `tau_spike` 是 spike 阈值，工程上常用 2 到 3 倍作为初筛，再结合 batch、grad norm、lr 和 loss scale 复核。这个公式不是为了替代人工分析，而是为了让告警和日志定位有稳定口径。

### 3.3 处理方法

1. 过滤异常数据。
2. 降低 peak learning rate。
3. 延长 warmup。
4. 开启或调小 gradient clipping。
5. 使用 BF16 替代 FP16。
6. 从 spike 前的健康 checkpoint 恢复。

面试表达：loss spike 要先判断是数据驱动、优化器驱动还是数值驱动，不能直接假设是学习率问题。

## 4. NaN 和 Inf

NaN / Inf 说明数值已经失控。

常见现象：

```text
loss = nan
grad_norm = inf
某些 parameter norm 变成 nan
optimizer state 出现 inf
```

### 4.1 常见原因

1. FP16 overflow。
2. 学习率过大。
3. warmup 太短。
4. 梯度爆炸。
5. attention logits 过大。
6. 输入中有异常值。
7. 自定义 loss 中除以 0。
8. 分布式 reduce 前某个 rank 已经 NaN。

### 4.2 排查顺序

```text
1. 找第一次出现 NaN 的 step。
2. 检查前一个 step 的 loss、grad norm、lr。
3. 检查 batch 是否异常。
4. 检查 mixed precision 和 loss scaling。
5. 检查模型输出 logits 是否过大。
6. 检查是所有 rank NaN，还是某个 rank 先 NaN。
```

有限性检查可以写成：

```math
g_{\mathrm{finite}}=
\mathbf{1}[\mathrm{isfinite}(L_t)]
\mathbf{1}[\mathrm{isfinite}(G_t)]
```

如果 `g_finite=0`，就不要继续让 optimizer 更新参数，而应该保存现场、定位第一个坏张量，并回滚到上一个健康 checkpoint。PyTorch 里常见做法是对关键张量显式检查 finite，在必要时打开 autograd anomaly detection；梯度裁剪接口也可以要求遇到 non-finite total norm 时直接报错。

### 4.3 处理方法

1. 使用 BF16。
2. 降低 learning rate。
3. 延长 warmup。
4. 开启 gradient clipping。
5. 跳过异常 batch。
6. 回滚到健康 checkpoint。
7. 对关键张量加 NaN/Inf 检查。

面试表达：NaN 的核心是定位第一个坏掉的 step 和第一个坏掉的张量。

## 5. 梯度爆炸和梯度消失

### 5.1 梯度爆炸

表现：

```text
grad norm 突然变大
loss spike
参数范数异常
可能进入 NaN
```

常见原因：

1. learning rate 太大。
2. warmup 不足。
3. 初始化不稳定。
4. loss mask 错误导致异常 loss。
5. 数据 batch 异常。

常见处理：

1. gradient clipping。
2. 降低 lr。
3. 延长 warmup。
4. 检查异常 batch。
5. 检查 loss 是否平均方式错误。

全局梯度范数通常按所有可训练参数的梯度拼接后计算：

```math
G_t=
\sqrt{\sum_i \lVert g_{t,i}\rVert_2^2}
```

如果最大允许范数为 `c`，global norm clipping 可以抽象成：

```math
g'_t=
g_t
\min\left(1,\frac{c}{G_t+\epsilon}\right)
```

这说明 clipping 只在 `G_t>c` 时缩小梯度方向的长度，不会在正常梯度范围内改变更新。面试里要能说清楚：clipping 是稳定训练的保护机制，不是修复坏数据、错误 label mask 或过大学习率的根本替代品。

### 5.2 梯度消失或过小

表现：

```text
loss 几乎不下降
grad norm 很小
参数更新很小
```

可能原因：

1. learning rate 太小。
2. loss mask 把大部分 token 忽略了。
3. 模型某些参数没有参与计算。
4. optimizer 参数分组漏了参数。
5. mixed precision underflow。

面试表达：梯度异常要结合 grad norm、参数更新幅度、loss mask 和 optimizer 参数分组一起看。

## 6. 数据异常导致训练异常

数据问题是训练异常的高频根因。

常见数据异常：

1. 空样本。
2. 极长样本。
3. 乱码。
4. 错误编码。
5. 特殊 token 缺失。
6. label 全部为 ignore。
7. prompt 和 answer 错位。
8. 重复样本比例过高。
9. benchmark 或测试集泄漏。

对 SFT 或指令微调，异常 batch 的一个关键指标是有效 label 比例：

```math
q_{\mathrm{label}}=
\frac{\sum_{j=1}^{T}\mathbf{1}[y_j\ne -100]}{T}
```

如果 `q_label` 接近 0，说明这一批几乎没有监督 token；如果 prompt token 也参与了 loss，则模型可能在学复读 prompt。稳定训练不只要求 batch shape 正确，还要求有效监督区域正确。

### 6.1 怎么检查 batch

建议打印或保存异常 step 的：

```text
sample id
source
input length
input_ids 前后若干 token
decoded text
labels 中有效 token 数
attention_mask sum
loss mask span
```

很多问题一 decode 就能看出来，例如回答被截断、角色 token 错、全是 padding、文本乱码。

### 6.2 数据 bug 的典型现象

| 现象 | 可能数据问题 |
| --- | --- |
| loss 不下降 | label 错位、有效 label 太少 |
| loss 很低但模型很差 | 数据重复、泄漏、任务太简单 |
| loss spike | 异常 batch、极端长度或坏样本 |
| 模型学会复读 prompt | SFT label mask 错误 |
| 模型输出格式混乱 | chat template 不一致 |

## 7. Loss mask 和 label 错误

Loss mask 错误非常隐蔽。

训练脚本可能正常运行，loss 也会下降，但模型行为不对。

常见错误：

1. user token 参与 SFT loss。
2. assistant token 没有参与 loss。
3. padding token 参与 loss。
4. 多轮对话只训练最后一轮。
5. label 没有 shift。
6. 截断后只剩 prompt，没有 answer。

loss mask 的门禁可以简化为：

```math
g_{\mathrm{mask}}=
\mathbf{1}[q_{\mathrm{label}}\ge q_{\min}]
\mathbf{1}[\Delta_{\mathrm{shift}}=1]
\mathbf{1}[\Delta_{\mathrm{pad}}=1]
```

其中 `Delta_shift=1` 表示 causal LM 的 input / label shift 正确，`Delta_pad=1` 表示 padding token 没有参与 loss。这个门禁不追求覆盖所有格式错误，但能拦住最常见、最隐蔽的训练目标错误。

检查方式：

```text
decode input_ids
decode labels != -100 的部分
确认模型到底在学什么
```

面试表达：训练 debug 时我一定会 decode 一批 input 和 label mask，因为很多格式问题只看 tensor shape 看不出来。

## 8. 学习率和 optimizer 问题

学习率相关问题非常常见。

### 8.1 学习率过大

表现：

1. loss spike。
2. grad norm 爆炸。
3. NaN。
4. 训练初期不稳定。

处理：

1. 降低 peak lr。
2. 延长 warmup。
3. 加强 gradient clipping。

### 8.2 学习率过小

表现：

1. loss 下降很慢。
2. 参数更新幅度很小。
3. validation 指标长期不动。

处理：

1. 增大学习率。
2. 检查 scheduler 是否配置错误。
3. 检查 optimizer 是否包含所有可训练参数。

### 8.3 optimizer state 恢复错误

如果从 checkpoint 恢复时只加载模型权重，没有加载 optimizer 和 scheduler，训练可能突然不稳定。

特别是 AdamW 的动量状态和学习率 schedule，如果不连续，loss 曲线可能跳变。

## 9. 混合精度溢出

FP16 训练容易遇到 overflow 或 underflow。

表现：

1. loss scale 频繁下降。
2. grad norm 变 Inf。
3. loss 变 NaN。

处理：

1. 改用 BF16。
2. 调整 loss scaling。
3. 降低 lr。
4. 对高风险操作使用 FP32。
5. 检查 softmax、norm 和自定义 kernel。

loss scaling 的核心关系是：

```math
\tilde{L}_t=
S_tL_t
```

```math
\tilde{g}_t=
S_tg_t
```

```math
g_t=
\frac{\tilde{g}_t}{S_t}
```

如果缩放后的梯度出现 Inf / NaN，scaler 通常会跳过本次 optimizer step 并调整 scale。调试时要记录 `S_t` 是否频繁下降，以及跳过更新是否和 loss spike、坏 batch 或 rank 异常发生在同一段时间。

面试表达：混合精度问题不只是 dtype 选择，也和 loss scaling、kernel 实现和梯度尺度有关。

## 10. 分布式同步错误

分布式训练中的 bug 通常更难定位。

常见现象：

1. 训练 hang 住。
2. 某些 rank loss 不一致。
3. 某些 rank OOM。
4. all-reduce 报错。
5. checkpoint 恢复后 rank 状态不一致。

### 10.1 常见原因

1. 某个 rank 数据长度不同。
2. collective 调用顺序不一致。
3. 某个 rank 先 NaN 或报错。
4. gradient accumulation 边界不一致。
5. dataloader 没有正确 shard。
6. 随机种子或 dropout 状态不一致。

### 10.2 排查方法

1. 先单卡复现。
2. 再单机多卡复现。
3. 最后多机复现。
4. 打印每个 rank 的 step、loss、batch size。
5. 确认所有 rank collective 调用一致。

每个 rank 的 loss 至少要做一致性观察：

```math
\Delta_{\mathrm{rank}}=
\max_r L_{t,r}-\min_r L_{t,r}
```

对应的分布式健康门禁可以写成：

```math
g_{\mathrm{rank}}=
\mathbf{1}[\Delta_{\mathrm{rank}}\le \tau_{\mathrm{rank}}]
\prod_r \mathbf{1}[\mathrm{isfinite}(L_{t,r})]
```

如果某个 rank 先变 NaN，其他 rank 可能只是被 collective 卡住。排查时要看 per-rank loss、batch id、有效 token 数和 collective 调用位置，而不是只看主进程日志。

面试表达：分布式 debug 要先缩小到单卡和单机多卡，确认问题是模型/数据问题还是通信/并行问题。

## 11. Checkpoint 恢复问题

Checkpoint 恢复错误很容易导致训练曲线异常。

完整恢复通常需要：

1. model weights。
2. optimizer state。
3. scheduler state。
4. scaler state。
5. random state。
6. dataloader position。
7. distributed state。
8. tokenizer 和 config。

常见错误：

1. 只恢复模型，不恢复 optimizer。
2. scheduler step 从 0 重新开始。
3. tokenizer 版本不一致。
4. FSDP/ZeRO checkpoint 加载方式不匹配。
5. 数据继续位置错乱。

完整 checkpoint 的恢复门禁可以写成：

```math
G_{\mathrm{ckpt}}=
g_{\mathrm{model}}\,
g_{\mathrm{opt}}\,
g_{\mathrm{sched}}\,
g_{\mathrm{scaler}}\,
g_{\mathrm{rng}}\,
g_{\mathrm{data}}
```

只要其中一项为 0，就只能说“加载了部分状态”，不能说“恢复了训练”。对训练恢复来说，optimizer 动量、scheduler step、loss scaler、随机状态和 dataloader 位置都可能影响接下来的 loss 曲线。

面试表达：checkpoint 能用于推理不代表能无缝恢复训练。恢复训练需要 optimizer、scheduler、scaler、随机状态和数据位置一致。

## 12. 最小可运行训练健康审计 demo

下面的 demo 用标准库模拟一次训练健康审计。输入是 step 级训练日志、batch 元信息、per-rank loss 和 checkpoint 字段；输出是 spike、NaN、梯度爆炸、坏 batch、rank 不一致、checkpoint 缺项和建议动作。

```python
import math
from statistics import median

history = [
    {"step": 1, "loss": 2.40, "grad_norm": 0.8, "lr": 1e-5, "loss_scale": 4096, "tokens_sec": 980},
    {"step": 2, "loss": 2.18, "grad_norm": 0.9, "lr": 2e-5, "loss_scale": 4096, "tokens_sec": 1005},
    {"step": 3, "loss": 2.05, "grad_norm": 1.0, "lr": 3e-5, "loss_scale": 4096, "tokens_sec": 996},
    {"step": 4, "loss": 5.80, "grad_norm": 7.5, "lr": 4e-5, "loss_scale": 4096, "tokens_sec": 991},
    {"step": 5, "loss": 2.02, "grad_norm": 1.1, "lr": 5e-5, "loss_scale": 4096, "tokens_sec": 997},
    {"step": 6, "loss": float("nan"), "grad_norm": float("inf"), "lr": 6e-5, "loss_scale": 2048, "tokens_sec": 990},
]

batches = {
    1: {"id": "batch_001", "max_len": 2048, "valid_label_ratio": 0.82, "has_nan_token": False, "source": "web_clean"},
    2: {"id": "batch_002", "max_len": 2048, "valid_label_ratio": 0.80, "has_nan_token": False, "source": "web_clean"},
    3: {"id": "batch_003", "max_len": 2048, "valid_label_ratio": 0.79, "has_nan_token": False, "source": "code_clean"},
    4: {"id": "batch_004_bad_mask", "max_len": 8192, "valid_label_ratio": 0.03, "has_nan_token": False, "source": "sft_mixed"},
    5: {"id": "batch_005", "max_len": 2048, "valid_label_ratio": 0.78, "has_nan_token": False, "source": "web_clean"},
    6: {"id": "batch_006_nan", "max_len": 2048, "valid_label_ratio": 0.81, "has_nan_token": True, "source": "math_synth"},
}

rank_losses = {4: [5.8, 5.7, 5.9, 5.8], 6: [2.1, float("nan"), 2.0, 2.1]}

checkpoint = {
    "model": True,
    "optimizer": True,
    "scheduler": False,
    "scaler": False,
    "random_state": True,
    "data_position": False,
}

seq_limit = 4096
valid_label_min = 0.2
grad_threshold = 5.0
spike_ratio = 2.0

def is_finite(x):
    return math.isfinite(x)

def finite_losses_before(step):
    return [r["loss"] for r in history if r["step"] < step and is_finite(r["loss"])]

spike_steps = []
nan_steps = []
grad_explosion_steps = []
bad_batches = []
rank_mismatch_steps = []

for record in history:
    step = record["step"]
    loss = record["loss"]
    grad_norm = record["grad_norm"]

    if not is_finite(loss) or not is_finite(grad_norm):
        nan_steps.append(step)

    previous = finite_losses_before(step)
    if previous and is_finite(loss):
        baseline = median(previous[-3:])
        if loss / baseline >= spike_ratio:
            spike_steps.append((step, round(loss / baseline, 2)))

    if is_finite(grad_norm) and grad_norm > grad_threshold:
        grad_explosion_steps.append((step, grad_norm))

    batch = batches[step]
    reasons = []
    if batch["max_len"] > seq_limit:
        reasons.append("too_long")
    if batch["valid_label_ratio"] < valid_label_min:
        reasons.append("low_valid_label_ratio")
    if batch["has_nan_token"]:
        reasons.append("nan_token")
    if reasons:
        bad_batches.append((step, batch["id"], reasons))

for step, losses in rank_losses.items():
    finite = [x for x in losses if is_finite(x)]
    has_nonfinite = len(finite) != len(losses)
    spread = max(finite) - min(finite) if finite else float("inf")
    if has_nonfinite or spread > 0.5:
        rank_mismatch_steps.append((step, has_nonfinite, round(spread, 3) if is_finite(spread) else "inf"))

missing_checkpoint = [name for name, ok in checkpoint.items() if not ok]
healthy_steps = [
    r["step"]
    for r in history
    if r["step"] not in nan_steps and r["step"] not in [s for s, _ in spike_steps]
]
rollback_step = max(step for step in healthy_steps if step < min(nan_steps or [10**9]))

recommendations = []
if bad_batches:
    recommendations.append("quarantine_bad_batches")
if grad_explosion_steps:
    recommendations.append("lower_lr_or_enable_clipping")
if nan_steps:
    recommendations.append("switch_bf16_or_check_loss_scaling")
if missing_checkpoint:
    recommendations.append("resume_only_from_complete_checkpoint")
if rank_mismatch_steps:
    recommendations.append("compare_rank_batches_and_collectives")

checks = {
    "spike_detected": spike_steps == [(4, 2.66)],
    "nan_detected": nan_steps == [6],
    "bad_batch_detected": bad_batches[0][1] == "batch_004_bad_mask",
    "checkpoint_gap_detected": missing_checkpoint == ["scheduler", "scaler", "data_position"],
    "rank_mismatch_detected": rank_mismatch_steps == [(6, True, 0.1)],
    "rollback_before_nan": rollback_step == 5,
}

print("spike_steps=", spike_steps)
print("nan_steps=", nan_steps)
print("grad_explosion_steps=", grad_explosion_steps)
print("bad_batches=", bad_batches)
print("rank_mismatch_steps=", rank_mismatch_steps)
print("missing_checkpoint=", missing_checkpoint)
print("rollback_step=", rollback_step)
print("recommendations=", recommendations)
print("checks=", checks)
print("gate_pass=", all(checks.values()))
```

期望输出：

```text
spike_steps= [(4, 2.66)]
nan_steps= [6]
grad_explosion_steps= [(4, 7.5)]
bad_batches= [(4, 'batch_004_bad_mask', ['too_long', 'low_valid_label_ratio']), (6, 'batch_006_nan', ['nan_token'])]
rank_mismatch_steps= [(6, True, 0.1)]
missing_checkpoint= ['scheduler', 'scaler', 'data_position']
rollback_step= 5
recommendations= ['quarantine_bad_batches', 'lower_lr_or_enable_clipping', 'switch_bf16_or_check_loss_scaling', 'resume_only_from_complete_checkpoint', 'compare_rank_batches_and_collectives']
checks= {'spike_detected': True, 'nan_detected': True, 'bad_batch_detected': True, 'checkpoint_gap_detected': True, 'rank_mismatch_detected': True, 'rollback_before_nan': True}
gate_pass= True
```

这个 demo 的重点不是模拟真实训练，而是把训练 debug 的日志口径固定下来：先定位 step，再关联 batch、grad、rank 和 checkpoint。真实项目里可以把这些检查接到 Trainer callback、训练日志、数据样本落盘和告警系统里。

## 13. 一个实用 debug checklist

遇到训练异常，可以按下面清单走：

```text
1. 异常类型是什么：loss spike、NaN、OOM、hang、吞吐下降？
2. 第一次出现在哪个 step？
3. 是否能稳定复现？
4. 该 step 的 batch 是否异常？
5. input_ids / labels / mask decode 后是否正确？
6. lr、warmup、grad norm 是否异常？
7. mixed precision 是否 overflow？
8. 单卡是否复现？
9. 从上一个 checkpoint 恢复是否正常？
10. 最近改动了什么：数据、代码、超参、并行、精度？
```

这比“感觉是学习率问题”可靠得多。

## 14. 面试官会怎么问

### 问法 1：训练 loss 突然 spike，你怎么排查？

可以这样答：

```text
我会先定位 spike 第一次出现的 step，然后看该 step 的 batch、loss、grad norm、learning rate 和 mixed precision 状态。先检查是否是坏数据或 label mask 错误，再看 warmup 是否太短、peak lr 是否过大、grad clipping 是否生效。如果出现 NaN/Inf，还要检查 FP16 overflow 和 optimizer state。必要时从 spike 前 checkpoint 恢复，并做单变量实验确认原因。
```

### 问法 2：NaN 怎么定位？

可以这样答：

```text
NaN 要定位第一个坏掉的 step 和第一个坏掉的张量。我会检查 loss、logits、grad norm、参数范数和 optimizer state，确认是 forward 就 NaN，还是 backward 后梯度 NaN。然后检查 batch、学习率、mixed precision、loss scaling 和 gradient clipping。分布式场景还要看是不是某个 rank 先 NaN。
```

### 问法 3：为什么 train loss 下降但模型效果很差？

可以这样答：

```text
可能是数据重复、评估污染、训练目标和评估目标不一致、label mask 错误或模型只学到了格式。我要看 validation loss、per-domain loss、去重和污染检测，也会 decode 样本检查模型到底在学哪些 token。不能只看 train loss 判断模型能力。
```

### 问法 4：分布式训练 hang 住怎么办？

可以这样答：

```text
先看是否某个 rank 提前报错或 OOM，其他 rank 在 collective 通信处等待。然后检查所有 rank 的 step、batch size、dataloader 长度和 collective 调用顺序是否一致。排查时先单卡，再单机多卡，最后多机，逐步缩小问题范围。
```

### 问法 5：恢复 checkpoint 后 loss 突然异常怎么办？

可以这样答：

```text
我会检查是否完整恢复了 model、optimizer、scheduler、scaler、random state 和数据位置。如果只恢复模型权重，AdamW 动量和学习率 schedule 不连续，loss 可能跳变。还要确认 tokenizer、config、并行切分和 checkpoint 格式一致。
```

## 15. 本章小结

本章核心结论：

1. 训练 debug 要先定位层级，再修复症状。
2. Loss spike 可能来自数据、学习率、梯度、精度或恢复状态。
3. NaN/Inf 要定位第一个坏 step 和第一个坏张量。
4. 梯度异常要结合 grad norm、loss mask 和 optimizer 参数分组看。
5. 数据 batch 和 label mask 是最常见、也最容易被忽略的根因。
6. 学习率过大、warmup 不足和 gradient clipping 缺失会导致训练不稳定。
7. 混合精度问题要关注 FP16 overflow、loss scaling 和高风险 kernel。
8. 分布式问题要先缩小到单卡/单机，再排查通信和 rank 状态。
9. Checkpoint 恢复必须包含 optimizer、scheduler、scaler、随机状态和数据位置。
10. 面试中要用结构化 checklist 回答训练 debug，而不是只说“调参”。
