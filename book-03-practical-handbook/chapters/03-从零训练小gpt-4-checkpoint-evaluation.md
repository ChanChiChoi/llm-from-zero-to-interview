# 第 3 章（第四篇）：Checkpoint、评估与训练证据

训练曲线和生成文本观察的是不同现象。Checkpoint 负责保存“训练到哪里、下一步怎样继续”，评估则要说明 loss 与生成行为在什么数据和协议下变化。本篇把状态保存、best/last 分工、loss/PPL、失败定位和报告组织成一条证据链。

## 3.4 Checkpoint、日志与可恢复评估

### 3.4.1 checkpoint 不是只有模型权重

一个训练状态快照可以写成：

```math
C_t=(\theta_t,s_t,t,L^*_{\mathrm{val}},h,\rho_t,\tau)
```

其中 $\theta_t$ 是模型参数，$s_t$ 是优化器状态，$t$ 是更新计数，$L^*_{\mathrm{val}}$ 是截至当前最优验证 loss，$h$ 是配置，$\rho_t$ 是随机数或数据迭代状态，$\tau$ 是 tokenizer 与特殊 token 元数据。

如果只保存 $\theta_t$，可以做某种形式的推理，却不能严格恢复训练轨迹。AdamW 至少维护一阶和二阶统计量：

```math
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
```

```math
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
```

不恢复 $m_t,v_t$，下一次更新的方向和尺度都会变化。若使用学习率 scheduler、AMP scaler、梯度累积、分布式 sampler 或随机数据增强，还要把对应状态加入恢复协议。

教学级字符模型至少保存：

```text
model_state_dict
optimizer_state_dict
step
best_val_loss
config
tokenizer 或 stoi/itos
```

生产训练还应考虑：scheduler state、scaler state、随机数状态、数据 shard 与 offset、分布式 rank 信息、代码版本、数据版本、设备和 dtype。

### 3.4.2 best 与 last 是两个不同对象

last.pt 表示最近一次保存的训练状态，适合中断恢复；best.pt 表示验证集指标最好的状态，适合最终评估。若评估 step 集合为 $\mathcal{T}$：

```math
t^*=\arg\min_{t\in\mathcal{T}}L_{\mathrm{val}}(t)
```

```math
L^*_{\mathrm{val}}=\min_{t\in\mathcal{T}}L_{\mathrm{val}}(t)
```

把最近模型误称为最佳模型，会把“可恢复性”和“泛化选择”混成一个概念。对于多指标任务，best 还需要固定选择规则，例如验证 loss、格式通过率和安全违规率的组合，而不能训练结束后挑一个看起来最好的样例。

### 3.4.3 保存与加载的实现

下面的实现使用临时文件再替换目标文件，减少进程在写入中途退出时留下半个 checkpoint 的概率。它假设 `model`、`optimizer`、`config` 和 tokenizer 元数据只包含张量、基本类型及普通容器，并以 PyTorch 2.12 的加载接口为目标版本。

> 代码状态：静态审阅，未运行（2026-10-29）

```python
from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    step,
    best_val_loss,
    config,
    tokenizer_meta,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "best_val_loss": float(best_val_loss),
        "config": config,
        "tokenizer_meta": tokenizer_meta,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
```

本例显式指定 `weights_only=True`，与 [PyTorch 2.12 `torch.load` 文档](https://docs.pytorch.org/docs/2.12/generated/torch.load.html)一致；该模式限制可反序列化对象范围，但不构成“任何来源都安全”的证明。PyTorch 文档仍提醒不要加载不可信来源。旧版本若不支持该参数，不应靠捕获任意 `TypeError` 自动退回普通 pickle 加载；应按目标版本单独审查兼容路径，并只加载可信文件。
### 3.4.4 step 的定义必须写进代码

本章把 step 定义为“已经完成的 optimizer update 数量”。因此初始模型是 step 0，先评估，再完成一次更新；完成后状态变成 step 1。恢复时从记录的 step 继续，下一轮会重新评估当前状态，然后执行下一次更新。另一种系统可以把 step 定义为“下一次待执行的更新编号”，但保存端和加载端必须使用同一种定义。

下面的循环依赖前文 3.2.4/3.2.6 定义的 `model`、`estimate_loss`、`get_batch`、`optimizer` 和 `device`，以及 3.4.3 定义的 `save_checkpoint`；把它接在这些代码之后即可运行。它显式记录评估/保存 step 和已完成的 optimizer update 数，checkpoint 写入当前工作目录下的 `runs/mini-gpt/step-semantics/`：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
from pathlib import Path

run_dir = Path("runs/mini-gpt/step-semantics")
run_dir.mkdir(parents=True, exist_ok=True)
best_path = run_dir / "best.pt"
last_path = run_dir / "last.pt"
config = {"batch_size": 2, "block_size": 16}
tokenizer_meta = {"type": "character"}
eval_iters = 5
eval_interval = 1
max_steps = 2
best_val_loss = float("inf")
start_step = 0
saved_steps = []
optimizer_updates = 0
log_rows = []

for step in range(start_step, max_steps + 1):
    if step % eval_interval == 0 or step == max_steps:
        metrics = estimate_loss(model, get_batch, eval_iters, device)
        train_loss = metrics["train"]
        val_loss = metrics["val"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_path,
                model,
                optimizer,
                step,
                best_val_loss,
                config,
                tokenizer_meta,
            )
        save_checkpoint(
            last_path,
            model,
            optimizer,
            step,
            best_val_loss,
            config,
            tokenizer_meta,
        )
        tokens_seen = step * config["batch_size"] * config["block_size"]
        saved_steps.append(step)
        log_rows.append((step, train_loss, val_loss, best_val_loss, tokens_seen))

    if step == max_steps:
        break

    input_ids, targets = get_batch("train")
    _, loss = model(input_ids.to(device), targets.to(device))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    optimizer_updates += 1

print("saved_steps=", saved_steps)
print("optimizer_updates=", optimizer_updates)
```
这个循环会在 step 0 和最后一个 step 做评估。恢复训练时，如果 last.pt 中 step 已经是 max_steps，应直接结束，而不是再做一次更新。重复恢复时还要处理日志去重，否则同一个 step 会在 CSV 中出现两次。

### 3.4.5 日志和生成样例

最小 CSV 行可以是：

```text
step,train_loss,val_loss,best_val_loss,tokens_seen,elapsed_seconds
0,3.46,3.47,3.47,0,0.8
100,2.41,2.58,2.58,51200,18.2
```

日志至少要能回答三个问题：训练进行了多少更新、模型在训练和验证上的变化是什么、某一行指标对应哪一个 checkpoint。生成样例还应记录 prompt、采样配置、随机种子和 checkpoint step；只保存一段没有元数据的文本，无法进行可靠对比。

下面是标准库版本的日志函数：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import csv
from pathlib import Path


def init_log(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow([
            "step",
            "train_loss",
            "val_loss",
            "best_val_loss",
            "tokens_seen",
            "elapsed_seconds",
        ])


def append_log(
    path,
    step,
    train_loss,
    val_loss,
    best_val_loss,
    tokens_seen,
    elapsed_seconds,
):
    with Path(path).open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow([
            step,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{best_val_loss:.6f}",
            tokens_seen,
            f"{elapsed_seconds:.3f}",
        ])


log_path = Path("runs/mini-gpt/mini_char_gpt_log.csv")
init_log(log_path)
append_log(log_path, 0, 3.46, 3.47, 3.47, 0, 0.8)
print(log_path.read_text(encoding="utf-8").strip())
```
示例把日志写到当前工作目录的 `runs/mini-gpt/`；真实实验应为不同 run 使用独立目录和明确的保留策略，日志和 checkpoint 不能靠文件名猜测相互对应。


### 3.4.6 保存策略与恢复验收

Checkpoint 保存的是“下一步如何继续”。两个运行即使模型参数相同，只要 Adam 的一阶/二阶矩不同，下一次更新也可能不同；scheduler、AMP scaler、随机数状态和数据采样位置同样会影响续训。因此，断点恢复要加载完整训练状态；只加载权重后重启优化器，是一次新的微调起点。前面的教学代码展示了模型、优化器和 step 的基本保存方式，尚未保存 scheduler、随机状态或数据迭代位置，不能据此声称可以逐步精确复现。

保存频率是在 I/O 开销与可接受的重算量之间取舍。若每步耗时 2 秒、每 100 步保存一次，均匀发生的中断平均会丢失约 100 秒，最大接近 200 秒，不含保存耗时。通常让 `last` 服务于恢复、`best` 由预先指定的验证指标选择，周期性归档保留训练阶段；最终 checkpoint 只表示预算结束，不一定是最佳状态。若覆盖旧文件，先确认新文件可读，并在日志中记下保留或清理的 step。

临时文件写完后再替换目标路径，可以减少进程中断时留下半个 checkpoint 的风险，但不能替代空间、权限、文件校验和来源检查。metadata 可包含格式版本、模型配置、数据/tokenizer hash 和关键张量 shape；加载时先核对元数据，再检查状态字典。对不可信文件，应使用目标 PyTorch 版本的安全加载接口，不要把 checkpoint 当作普通文本数据。

恢复验收可做一个小对照：保存 step 10 的状态，继续用同一批数据更新一次得到 A；重新加载 step 10，再用相同数据和随机状态更新得到 B。相同环境且确定性设置完整时比较 A/B 的参数；若算子存在非确定性，预先说明比较容差。只加载权重得到的 C 可能不同，因为它没有 Adam 历史。恢复后的首条日志还要核对模型 step、optimizer/scheduler 计数、tokens_seen 和数据位置之间的对应关系；这些数可以不同，但关系必须有定义。

日志与 checkpoint 应通过唯一 `run_id` 关联。恢复时若重写同一个 step，要么去重，要么记录 resume event，避免曲线出现伪重复。首次恢复评估应复用固定验证 batch、相同 `model.eval()` 状态和相同生成配置；如果指标不一致，先查模型权重、数据位置和评估协议，再继续训练。

## 3.5 训练曲线与生成行为：从 loss 到证据

### 3.5.1 loss 的定义、估计与 perplexity

设训练评估集合中有 $N_{\mathrm{train}}>0$ 个有效预测位置，验证集合中有 $N_{\mathrm{val}}>0$ 个有效预测位置，则：

```math
L_{\mathrm{train}}(t)=\frac{1}{N_{\mathrm{train}}}\sum_{i=1}^{N_{\mathrm{train}}}\ell_i(\theta_t)
```

```math
L_{\mathrm{val}}(t)=\frac{1}{N_{\mathrm{val}}}\sum_{i=1}^{N_{\mathrm{val}}}\ell_i(\theta_t)
```

token-level perplexity 是：

```math
\mathrm{PPL}=\exp(L)
```

只有在 tokenizer、loss mask、平均方式和评估分布一致时，两个 perplexity 才适合直接比较。字符级和 BPE 级的 token 单位不同，不能只看 PPL 数字大小就判断哪一个模型更好。

PPL 可以粗略理解为模型在每个 token 位置上的“等效候选数”：loss 为 $L$ 时，PPL 是 $e^L$。这个数仍依赖 token 如何切分、哪些位置进入平均以及评估文本的分布；字符级与 BPE 级的 PPL 不能脱离这些条件直接比较。

前文的 `estimate_loss` 通常只抽取若干个 batch，因此得到的是集合平均的抽样估计；当 `eval_iters=0` 时没有定义，代码会拒绝这个输入。抽样数量少时，验证 loss 可能因为起点不同而波动。比较很小的改进时，应固定评估窗口、报告有效 token 数，或完整遍历验证集。

当前实现每个 batch 都有相同的 $B\times T$ 个有效目标，所以平均 batch loss 等于按 token 汇总后的均值。若变长 batch 含 padding，不能直接等权平均各批的标量 loss。例如一批只有 1 个有效 token、loss 为 3，另一批有 9 个有效 token、平均 loss 为 1；等权平均是 2，按 token 汇总则是 $(1\times3+9\times1)/10=1.2$。这时应累计 loss 总和与有效 token 数，再做一次除法。

训练—验证差距可以写为：

```math
G(t)=L_{\mathrm{val}}(t)-L_{\mathrm{train}}(t)
```

差距变大常见于过拟合，但也可能来自 train/val 分布不同、验证集过小或两个 split 的 tokenization 方式不同，不能只凭 gap 一个数字下结论。

### 3.5.2 四种曲线形态与不同原因

| train loss | val loss | 可能解释 | 下一步证据 |
| --- | --- | --- | --- |
| 下降 | 下降 | 模型正在学习且验证分布同步改善 | 固定 prompt 的生成和更多评估窗口 |
| 下降 | 先降后升 | 过拟合、重复数据或训练过久 | best checkpoint、数据去重、按文档切分 |
| 高且下降慢 | 高且下降慢 | 容量不足、步数不足、学习率太低或数据难度高 | 小规模学习率/模型消融 |
| 剧烈振荡或 NaN | 剧烈振荡或 NaN | 学习率、梯度、mask、label 或 dtype 异常 | 第一处非有限值、梯度范数、label 范围 |

初始 loss 接近 $\log V$ 只是均匀 logits 下的 sanity check。loss 高于这个量不一定是 bug，可能是随机初始化产生了偏置；loss 低于它也不自动说明模型学会了语言，尤其在数据泄漏或重复样本场景下。

### 3.5.3 生成样例和 loss 观察不同对象

loss 是平均的局部概率指标；生成样例是一次采样路径。一个模型可能降低常见 token 的 loss，却在较长生成中反复、提前结束或破坏格式。反过来，一次看起来漂亮的样例也可能只是随机得到的高概率路径。

生成评估至少应固定：

1. prompt 集合及其来源；
2. checkpoint step；
3. 最大新 token 数和停止规则；
4. temperature、top-k、top-p；
5. 随机种子或明确使用 greedy；
6. tokenizer 与 decode 版本。

可以记录以下行为指标：

```math
\mathrm{repeat\_rate}=
\frac{\#\{i:w_i=w_{i-1}\}}{n-1},\qquad n\ge2
```

其中 $w_i$ 是按空格切出的词或按 token 切出的单位，且要求 $n\ge2$。当样本少于两个单位时，这个指标没有定义，代码返回 `None`，不能把“没有足够观测”伪装成零重复；还可以加入 distinct-n、训练集 n-gram 重叠、格式通过率、停止条件命中率和任务特定指标。

### 3.5.4 一个零依赖的曲线诊断实验

下面的 toy 数据有意包含“训练 loss 继续下降、验证 loss 反弹、生成重复率上升”的阶段。它用于演示如何把多个证据放在一起，不是对真实模型质量的评估器。

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import csv
import math
from io import StringIO


LOG = """
step,train_loss,val_loss
0,3.47,3.46
100,2.80,2.90
300,1.95,2.35
600,1.10,2.60
1000,0.55,3.10
"""

SAMPLES = {
    0: "qzae lmx trrpp oo",
    300: "hello model the text",
    1000: "hello hello hello hello language model",
}


def read_log(text):
    return [
        {
            "step": int(row["step"]),
            "train_loss": float(row["train_loss"]),
            "val_loss": float(row["val_loss"]),
        }
        for row in csv.DictReader(StringIO(text.strip()))
    ]


def adjacent_repeat_rate(text):
    words = text.split()
    if len(words) < 2:
        return None
    repeated = sum(left == right for left, right in zip(words, words[1:]))
    return repeated / (len(words) - 1)


rows = read_log(LOG)
best = min(rows, key=lambda row: row["val_loss"])
last = rows[-1]
best_seen = rows[0]["val_loss"]
overfit_step = None
for previous, row in zip(rows, rows[1:]):
    if row["val_loss"] < best_seen:
        best_seen = row["val_loss"]
    elif (
        row["train_loss"] < previous["train_loss"]
        and row["val_loss"] > best_seen
    ):
        overfit_step = row["step"]
        break

print("uniform_loss_for_vocab_32=", round(math.log(32), 4))
print("best_step=", best["step"])
print("last_gap=", round(last["val_loss"] - last["train_loss"], 2))
print("first_overfit_signal=", overfit_step)
for step, sample in SAMPLES.items():
    rate = adjacent_repeat_rate(sample)
    print(
        f"sample_{step}_repeat_rate=",
        None if rate is None else round(rate, 2),
    )
```
这个实验应得到 best step 300、第一处过拟合信号 600，以及末期样例较高的相邻重复率。它没有证明 600 一定是生产意义上的过拟合起点，因为阈值、样本和评估量都由教学代码构造；它展示的是证据组合方式。

在这组记录里，step 300 的 validation loss 为 2.35，低于 step 600 的 2.60，因此按预先选定的 loss 规则保存 step 300。因为同一 validation 集被反复用于挑 checkpoint，2.35 不能再被解释成独立测试估计；最终比较应在未参与调参和选择的测试集上进行，或把结论限定为“在这套 validation 协议中的最优点”。

### 3.5.5 从异常现象回到第一分歧点

当 loss 不下降时，可以沿数据路径回退，而不是立即扩大模型：

1. 打印一个 x/y 样本，确认右移关系；
2. 检查 input_ids 为 long，label 范围为 $[0,V)$；
3. 对一个 batch 检查 logits 为 $[B,T,V]$，loss 是有限标量；
4. 检查 causal mask 的未来权重是否为零；
5. 检查梯度是否存在、是否在第一步就溢出；
6. 把学习率降低一个数量级做对照；
7. 在极小数据上尝试过拟合几个 batch，验证模型是否有学习能力；
8. 再检查 train/val 切分、tokenizer 和生成 decode。

如果极小数据也无法过拟合，问题更可能在实现或数据契约；如果极小数据可以过拟合而真实数据不行，才进一步研究数据规模、容量、优化预算和分布差异。这个顺序能把“模型能力不足”和“代码目标错误”分开。

### 3.5.6 报告一次训练而不是只报告一个数字

一份可复现实验记录至少应包含：

```text
数据：原始文本版本、hash、切分规则、训练/验证 token 数
tokenizer：类型、词表大小、版本、特殊 token
模型：D、层数、head 数、最大长度、参数量
优化：batch、token budget、学习率、优化器、梯度裁剪
环境：PyTorch、CUDA、设备、dtype、随机种子
结果：train/val loss、PPL、best step、生成配置和样例
归因：最强证据、剩余不确定性、下一项消融
```

“loss 降了”是结果的一部分，不是完整结论。读者需要知道它在哪个 token 单位、哪套数据、哪种评估协议下下降。

总体平均会掩盖样本差异。比如验证集里的常见短句 loss 较低、罕见长句 loss 较高，最终 token 平均值会按有效 token 数加权，并不等于先算每句 loss 再做平均。若部署流量以长指令为主，总平均可能不代表目标场景；应按长度、来源和任务切片，并报告每片有效 token 数。把平均 loss 从 2 降到 1.8，PPL 会从约 7.39 降到 6.05，比例变化并不相同；更不能把不同 tokenizer 的 2.0 与 1.8 直接排成胜负，因为 token 单位和分母可能不同。

验证 loss 本身也有抽样波动。若两 checkpoint 相差 0.01，而抽样窗口会让结果波动 0.03，就只能说当前测量分辨不出稳定差异。可以固定窗口、完整遍历验证集或做分层重复抽样；若用 bootstrap 区间，要说明抽样单位是 token、文档还是样本。完整遍历能减少抽样噪声，却不能修复数据分布偏差或标签泄漏。图表应标出 checkpoint step、有效 token 数和数据版本。

生成样例是另一类证据，应与 loss 一起解释而不是互相替代。loss 下降但固定 prompt 的重复率上升，可能来自过拟合、采样配置改变或重复文本；loss 暂时升高而格式通过率提高，则可能说明 token 平均目标和任务目标不一致。每条样例都要保留 checkpoint、prompt、解码参数和判定规则，失败样例也要留下。

异常定位先找“第一个分歧点”：若 loss 首次成为 NaN，记录对应 batch、logit 极值、梯度范数和学习率；若 train loss 降而 val loss 升，先查切分、重复文本和样本长度；若 loss 都下降而生成变重复，再固定解码设置检查高频模式。每个诊断都应有一个能推翻当前猜测的最小对照，而不是仅凭曲线形状下结论。



训练和评估流程已经建立后，字符级 tokenizer 的局限就能用同一组预算来比较。最后一篇会说明 BPE 改变了哪些数量、哪些模型接口必须一起迁移，以及怎样避免不公平的结论。
