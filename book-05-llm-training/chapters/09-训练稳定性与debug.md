# 第九章：训练稳定性与调试：从异常现象到根因定位

## 9.0 稳定训练不是一条光滑的 loss 曲线

训练开始后，loss 通常总体下降，但不会沿一条光滑直线前进。某个 batch 可能让 loss 短暂升高，某次浮点运算可能产生 Inf，某个 rank 可能在 collective 前退出，checkpoint 恢复后 scheduler 也可能跳到错误的阶段。

真正难的地方在于：日志里看到的往往是最后一个症状，而不是最早发生的错误。一个错误的 label mask 可能先改变有效 token 分母，再改变梯度，随后表现为 loss spike，最后才变成模型能力下降。如果一看到 spike 就降低学习率，可能只是在改变症状，仍然没有修复目标数据。

训练稳定性调试可以沿着这条因果链展开：

~~~text
数据与 token
  -> input/label/mask
  -> forward 与 loss
  -> backward 与 gradient
  -> optimizer/scheduler
  -> dtype 与数值状态
  -> distributed collective
  -> checkpoint 与恢复
~~~

初学者可以先记住一个原则：先找“第一次异常出现在哪里”，再决定改什么。专家还要追问：

1. 这个指标的分母是什么？
2. 异常发生在 micro-step 还是 optimizer step？
3. 第一个坏值是输入、logits、loss、gradient、parameter 还是 optimizer state？
4. 异常是所有 rank 共有，还是某个 rank 私有？
5. 修复后是否只让 loss 好看了，还是验证质量、吞吐和恢复连续性也没有回归？

本章建立一套可以复现、比较和回滚的 debug 方法。它不是“把所有超参数都调一遍”的清单，而是把每一次修改对应到一个可检验的假设。

## 9.1 健康基线：先知道什么是正常

### 9.1.1 基线不能只有一条 loss 曲线

不同模型、数据混合和 tokenizer 的 loss 绝对值不能直接比较。一个没有数据和分母背景的 loss=2.0，既不能自动证明训练正常，也不能自动证明训练异常。

一个可用的健康基线至少包括：

~~~text
train loss and validation loss
effective label tokens
learning rate and scheduler step
raw and clipped grad norm
relative parameter update
loss scale and skipped updates
tokens/sec and step time
per-rank loss and collective position
peak memory
checkpoint write/read result
~~~

train loss 和 validation loss 应按有效 token 口径汇总；grad norm 要区分裁剪前和裁剪后；learning rate 要绑定 optimizer step；tokens/sec 要说明是单 rank、单机还是全局。相同名称的指标，如果计算口径不同，不能放在同一张图上直接比较。

### 9.1.2 用训练账本定义一个 step

每个 optimizer step 可以记录：

~~~math
h_t=
(
L_{\mathrm{train},t},
L_{\mathrm{val},t},
G_t,
\eta_t,
S_t,
T_t,
R_t,
U_t,
\Delta_{\mathrm{rank},t}
)
~~~

其中：

- L_train,t 和 L_val,t 是训练与验证 loss；
- G_t 是全局梯度范数；
- eta_t 是当前学习率；
- S_t 是 loss scale；
- T_t 是真实参与监督的有效 label token 数；
- R_t 是吞吐；
- U_t 是 GPU 利用率或类似系统信号；
- Delta_rank,t 是需要进一步解释的 rank 间差异。

把 T_t 放进日志很重要。两个 batch 的平均 loss 可能相同，但一个只有几百个有效 token，另一个有几万个有效 token，它们对参数更新的统计可靠性不同。

### 9.1.3 基线应保存配置

基线不仅是数值，还包括：

~~~text
model and tokenizer version
data manifest and mixture
random seeds
micro-batch and accumulation
parallel topology
parameter/gradient/optimizer dtypes
kernel and framework versions
evaluation script version
checkpoint policy
~~~

没有这些上下文，后续的“恢复”“加速”或“更稳定”都可能只是比较了不同实验。

## 9.2 第一次异常：把时间轴作为主索引

### 9.2.1 有效 token loss 的分母

令第 t 个 batch 展平后有 n_t 个 token 槽位，m_j 表示第 j 个位置是否参与监督，则：

~~~math
T_t
=
\sum_{j=1}^{n_t}m_j
~~~

~~~math
L_t
=
-\frac{1}{T_t}
\sum_{j=1}^{n_t}
m_j
\log p_\theta(y_j\mid x_{<j})
~~~

当 T_t=0 时，这个平均值没有定义。工程上应记录并跳过空监督 batch，不能把它默认为 loss=0，因为这会伪造一条正常曲线。

这里 m_j 属于 {0,1}，n_t 是非负整数；只有 T_t>0 时才可以把未归一化 token loss 除以它。若某个 rank 的局部 T_t 为 0，但全局仍有有效 token，需要按全局未归一化 loss 和 token count 聚合，不能先把该 rank 的局部 mean 虚构为 0。

### 9.2.2 异常指示量

可以用一个指示量把不同信号投影到 step 轴：

~~~math
a_t
=
\max
\left\{
\mathbf{1}[\neg\operatorname{finite}(L_t)],
\mathbf{1}[\neg\operatorname{finite}(G_t)],
\mathbf{1}[T_t=0],
\mathbf{1}[r_t\ge\tau_{\mathrm{spike}}],
\mathbf{1}[\Delta_{\mathrm{rank},t}>\tau_{\mathrm{rank}}]
\right\}
~~~

第一个候选异常 step 是：

~~~math
t^\*
=
\min\{t\mid a_t=1\}
~~~

a_t=1 只表示需要保存现场和继续分析，不表示根因已经确定。rank loss 不同，可能只是各 rank 处理了不同难度的数据；只有在有效 token、数据 shard 和聚合方式可比时，差异才更支持同步问题。

### 9.2.3 四个阶段的有限性检查

一次更新应分别检查：

~~~text
forward output and loss
backward gradients
unscaled and clipped gradients
updated parameters and optimizer state
~~~

forward 已经出现 NaN，优先看输入、mask、logits 和算子；forward 有限而 backward 变坏，优先看反向 kernel、loss scaling 和梯度路径；只有 optimizer step 后变坏，才把学习率、weight decay、optimizer state 和恢复状态放在首位。

## 9.3 从异常曲线建立证据链

### 9.3.1 时间、层级和反事实

看到异常时先保留现场：

| 问题 | 证据 |
| --- | --- |
| 第一个异常何时出现 | global step、micro-step、optimizer step、phase |
| 哪一层先异常 | batch、forward、backward、optimizer、collective |
| 哪个 rank 先异常 | per-rank loss、finite flag、collective position |
| 能否复现 | 固定 batch、checkpoint、seed、最小配置 |
| 修复是否完整 | 同口径 loss、质量、吞吐和恢复对照 |

每次修改都应对应一个假设。例如：

- 降低 peak lr：检验更新步长过大；
- 替换异常 batch：检验样本触发了错误；
- 切换 BF16：检验 FP16 动态范围；
- 单卡重放：检验问题是否依赖 collective；
- 从上一个 checkpoint 恢复：检验参数是否已被污染。

如果同时修改数据、学习率、batch、精度和并行策略，即使训练恢复，也无法知道是哪一个变量起作用。

### 9.3.2 异常组合

| loss | grad norm | 有效 token/rank 状态 | 先验证什么 |
| --- | --- | --- | --- |
| 有限但突然升高 | 正常 | batch 来源或难度变化 | 数据分布、模板、评估口径 |
| 升高且同步变大 | 变大 | token 数正常 | 学习率、warmup、label 或更新步长 |
| NaN/Inf | NaN/Inf | 单个 rank 先出现 | forward/backward 溢出、坏输入 |
| 训练正常、验证恶化 | 正常 | validation 固定 | 过拟合、评估模式、污染 |
| loss 正常、吞吐下降 | 正常 | rank 等待或利用率低 | dataloader、通信、kernel 回退 |
| 一个 rank 异常 | 该 rank 异常 | 其他 rank 等待 | rank 私有 batch、OOM、控制流分叉 |

这些组合是候选原因排序，不是自动诊断规则。每一行都应转成一个最小反事实实验。

## 9.4 Loss spike：有限的异常也需要解释

### 9.4.1 spike 和 NaN 不是同一件事

有限的短时 spike 可能来自困难 batch 或数据分布切换；NaN/Inf 说明至少有一个数值已经失去有限性。训练集 spike 和验证集 spike 也需要分开看：

- 只有训练集 spike：可能是某个 batch、监督区域或更新步长；
- 训练和验证一起跳变：可能是参数更新、恢复状态或全局数据切换；
- 只有验证集恶化：可能是评估实现、分布变化或过拟合。

### 9.4.2 基于窗口的候选检测

设最近 k 个健康 step 的 loss 中位数为基线：

~~~math
b_t
=
\operatorname{median}(L_{t-k},\ldots,L_{t-1}),
\qquad
r_t
=
\frac{
L_t
}{
b_t
}
~~~

当 r_t >= tau_spike 时，可以把它标为候选 spike。2 到 3 倍可以作为某个项目的告警起点，但不是跨模型的标准；窗口长度、loss 方差、有效 token 数和数据混合都会改变解释。

这个比值只有在窗口至少含一个有限的健康 loss 且其中位数严格大于 0 时才有定义。若没有健康窗口，或 baseline=0，应把 spike 比值标为 `not_applicable` 并改用绝对 loss、梯度和数据证据，而不是在分母加一个极小数制造任意大的“告警”。tau_spike 也应为正的、预先记录的项目阈值。

### 9.4.3 三组对照

数据对照：保存 spike batch 的 sample id、来源、长度、token 分布、特殊 token 和 label mask；在健康 checkpoint 上重放。如果旧参数对这批数据也高 loss，问题更像样本、模板或标签；只有新参数高 loss，数据可能只是触发器。

状态对照：把 spike 前后的 grad norm、learning rate、warmup、loss scale、optimizer step 和恢复时间放在同一时间轴。loss 和 grad norm 同时升高，更支持更新或监督信号异常；loss 升高但其他状态平稳，要优先检查 batch 难度和聚合口径。

验证对照：固定 validation batch 和评估脚本，确认是训练信号变化还是评估路径变化。修复后要同时观察下一段训练、validation、目标能力和吞吐。

### 9.4.4 修复动作的边界

如果 decode 确认是错误模板、乱码或 label 错位，应修数据管线；只跳过单个 batch 不能修复同源样本。如果 spike 与 scheduler 或 checkpoint 恢复同步出现，应先检查 optimizer/scheduler state。clipping 可以限制一次异常更新，但不能修复错误分母或坏数据。参数已被 NaN 污染时，应从污染前 checkpoint 恢复。

## 9.5 NaN、Inf 和第一个坏张量

### 9.5.1 常见原因

1. FP16/FP8 overflow。
2. 学习率过大或 warmup 太短。
3. 梯度爆炸。
4. attention logits、softmax 或 norm 数值异常。
5. 输入含非法值。
6. 自定义 loss 中除以 0。
7. 分布式 reduce 前某个 rank 已经非有限。

最终 loss=nan 只说明错误传播到了 loss，不说明错误起源就在 loss。

### 9.5.2 保留第一个坏值

调试时记录：

~~~text
first non-finite step
first non-finite phase
tensor name and layer
rank
dtype and shape
batch/sample id
loss scale
learning rate
~~~

torch.autograd.detect_anomaly(check_nan=True) 可以在小规模调试中追踪产生异常梯度的反向节点，但速度开销很大，不应常驻生产训练。clip_grad_norm_ 的 error_if_nonfinite=True 可以把非有限 total norm 直接暴露出来，属于保护和定位手段，不是根因修复。

### 9.5.3 累积窗口中的异常

如果异常发生在 gradient accumulation 的中间，已经累积的梯度不能与下一个窗口继续拼接。通常应丢弃整个 accumulation window，并记录丢弃的有效 token 数、异常 batch 和当前 optimizer step；否则一个不完整梯度可能把错误传播到下一次更新。

## 9.6 梯度爆炸与梯度过小

### 9.6.1 全局梯度范数

所有可训练参数梯度组成的全局范数为：

~~~math
G_t
=
\sqrt{\sum_i\lVert g_{t,i}\rVert_2^2}
~~~

给定阈值 c，global norm clipping 为：

~~~math
g'_t
=
g_t
\min\left(
1,
\frac{c}{G_t+\epsilon}
\right)
~~~

它只在 G_t>c 时缩短梯度，不改变梯度方向。clipping 的统计应包括裁剪前范数、裁剪后范数和触发比例。

### 9.6.2 相对参数更新量

梯度大不一定等于参数更新大。记录：

~~~math
u_t
=
\frac{
\lVert\theta_{t+1}-\theta_t\rVert_2
}{
\lVert\theta_t\rVert_2+\epsilon
}
~~~

G_t 大但 u_t 正常，可能是参数尺度或 optimizer 预条件器的作用；G_t 正常但 u_t 突然升高，要检查学习率、optimizer state、参数分组和 weight decay。

这个相对量要求 theta_t 的范数严格大于 0，且两次参数都是有限值。零初始化的小张量、尚未初始化的 adapter 或被置零的参数组不应靠 epsilon 伪装成“相对更新极大”；此时应另报绝对更新范数和参数组状态。

### 9.6.3 梯度消失与零梯度

loss 不下降、grad norm 很小、参数更新很小，可能来自：

- learning rate 太小；
- loss mask 忽略了大部分 token；
- 参数没有参与 forward；
- 参数没有加入 optimizer；
- FP16 underflow；
- 某个 MoE expert 很少收到 token。

要区分“很小”和“完全为零”。分层梯度、有效 label 数、参数 group 元数据和局部 FP32 对照比一个总 grad norm 更有信息。

### 9.6.4 梯度累积的最终梯度

若第 b 个 micro-batch 有 T_b 个有效 token、按 token 汇总的梯度为 G_b，一次 update 的目标应近似为：

~~~math
g_{\mathrm{update}}
=
\frac{
\sum_{b=1}^{A}G_b
}{
\sum_{b=1}^{A}T_b
}
~~~

直接平均每个 micro-batch 的 mean gradient，隐含每个 T_b 相同。debug 要记录 accumulation boundary、每个 T_b、累计梯度和 clipping 时机。

## 9.7 数据、token 和 loss mask

### 9.7.1 常见数据异常

~~~text
empty sample
extreme length
garbled text
missing special token
wrong role template
shifted or truncated answer
duplicate or leaked sample
invalid numeric value
~~~

只看 tensor shape 通常发现不了这些问题。异常 batch 应能追溯到 sample id、source、原文和 tokenized 版本。

### 9.7.2 有效监督比例

对 SFT 或对话训练，attention mask 为 a_j，label mask 为 m_j：

~~~math
T_{\mathrm{attn}}
=
\sum_j a_j,
\qquad
T_{\mathrm{label}}
=
\sum_j m_j
~~~

~~~math
q_{\mathrm{label}}
=
\frac{T_{\mathrm{label}}}
{T_{\mathrm{attn}}+\epsilon}
~~~

T_label=0 表示没有监督目标，应显式处理；q_label 很低不一定是错误，但需要结合任务和长度分布解释。

q_label 的定义要求 T_attn>0，且 label mask 是 attention mask 的子集；空样本或所有 attention 都被 mask 时该比例未定义。若 label 位落在 a_j=0 的位置，问题不是“比例偏低”，而是数据格式或 mask 协议已经不一致。

### 9.7.3 逐样本 decode

异常 batch 至少保存：

~~~text
sample id and source
input length
decoded input
decoded labels where label != ignore_index
attention_mask sum
label_mask sum
role and special-token spans
packing document boundaries
~~~

例如，模型学会复读 user prompt，可能不是优化器不稳定，而是 user token 错误地参与了 loss；回答全部为空，可能是截断只保留了 prompt；packed 文档互相泄漏，可能是 document boundary 没有进入 attention 规则。

### 9.7.4 按来源和长度切片

总体平均值会掩盖一个坏 source。至少按来源、长度桶、语言、任务和合成/人工标记统计：

- loss；
- 有效 label ratio；
- 长度分布；
- 非有限值；
- spike 频率；
- 重复和污染风险。

这能区分偶发 outlier 与某一数据分支的系统性错误。

## 9.8 学习率、优化器和 scheduler

AdamW 的简化更新为：

~~~math
\theta_{t+1}
=
\theta_t
-
\eta_t
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
-
\eta_t\lambda\theta_t
~~~

因此 loss spike 不能只看配置里的 peak lr，还要看：

- 当前 optimizer step；
- 偏置修正后的动量和二阶矩；
- weight decay；
- effective token batch；
- scheduler 是否按 optimizer step 前进；
- checkpoint 恢复后 state 是否连续。

这条简化式使用 0<=beta_1,beta_2<1、epsilon>0、eta_t>=0，并把 decoupled decay 写成一阶近似的更新形式。实际框架对 decay 的精确施加顺序、foreach/fused kernel 和参数组排除规则仍需以其实现为准；NaN 的 m/v 或无效 learning rate 不会由该公式自行恢复。

### 9.8.1 学习率过大

若 loss、grad norm 和相对更新量同时升高，且异常靠近 warmup 结束或 scheduler 切换，优先做：

1. 保持数据和 optimizer 不变，降低 peak lr；
2. 保持 peak lr 不变，延长 warmup；
3. 记录 clipping ratio；
4. 比较达到同一质量所需的 token 与时间。

这三步是不同实验，不能同时改完再归因。

### 9.8.2 学习率过小

loss 下降慢、更新量很小可能是 lr 过小，也可能是 loss mask 或 optimizer 参数组错误。先比较有效 token 数、梯度范数和相对更新量；梯度本身为零与梯度正常但 lr 小是不同问题。

### 9.8.3 optimizer state 恢复错误

只恢复模型权重而不恢复 optimizer、scheduler 或 scaler，会改变下一步训练轨迹。恢复后打印：

~~~text
global optimizer step
consumed tokens
group learning rates
optimizer state step
loss scale
data cursor
~~~

如果只加载权重是有意的新实验，应明确把它称为新的微调或继续训练起点，而不是无缝恢复。

## 9.9 混合精度溢出

### 9.9.1 FP16、BF16 和 FP8

FP16 动态范围较小，更容易 overflow/underflow；BF16 动态范围接近 FP32，但尾数位少；FP8 还要引入 scale、amax、格式选择和硬件 kernel。数值问题不能只靠改配置文件中的 dtype 判断，必须记录实际参数、激活、累积、optimizer state 和 kernel 路径。

### 9.9.2 Loss scaling

FP16 loss scaling 为：

~~~math
\tilde L_t=S_tL_t,
\qquad
\tilde g_t=S_tg_t,
\qquad
g_t=\frac{\tilde g_t}{S_t}
~~~

它能降低小梯度 underflow 概率，却不能修复错误 loss，也不能防止缩放后的值 overflow。要记录：

~~~text
current scale
scale growth/backoff
skipped optimizer steps
overflow count
first bad rank
batch and length
~~~

### 9.9.3 unscale 后再 clipping

典型顺序是：

~~~text
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
scaler.step(optimizer)
scaler.update()
~~~

在 unscale 前 clipping，阈值对应的是 scaled gradient。BF16 通常不需要 FP16 式动态 loss scaling，但仍要检查有限性、累积精度和异常 kernel。

## 9.10 分布式同步错误

### 9.10.1 控制流与数值差异

控制流错误通常表现为 hang：

- rank 提前异常或 OOM；
- collective 调用顺序不同；
- dataloader 长度不同；
- PP send/recv micro-batch id 不匹配。

数据/数值差异通常先表现为曲线不同：

- rank batch 重叠或遗漏；
- 有效 token 分母不同；
- mask/loss reduction 不一致；
- 某个 rank 先出现 NaN。

随机种子不同会导致本地随机路径不同，但不自动等于通信错误；是否要求逐位一致要由训练目标和实现决定。

### 9.10.2 逐级缩小

先单卡重放固定 batch，再单机多卡，最后多机。每个 rank 记录：

~~~text
global step
micro-step
batch id
valid tokens
local loss
grad norm
current collective
last completed collective
finite flags
~~~

若一个 rank 先 NaN，其他 rank 的 hang 可能只是等待 collective；必须找出第一个坏 rank，而不是只看主进程最后一行。

### 9.10.3 全局加权 loss

第 r 个 rank 有 T_{t,r} 个有效 token、local mean loss 为 L_{t,r}，全局 loss 应为：

~~~math
T_t=\sum_r T_{t,r},
\qquad
L_t=
\frac{\sum_r T_{t,r}L_{t,r}}{T_t}
~~~

rank 间 spread：

~~~math
\Delta_{\mathrm{rank},t}
=
\max_r L_{t,r}-\min_r L_{t,r}
~~~

spread 必须结合不同数据难度和有效 token 解释，不能单独设一个跨任务固定阈值。

全局 loss 要求所有 T_{t,r} 为非负计数且 T_t>0；Delta_rank 还要求参与比较的 L_{t,r} 都有限、数据分片和 reduction 口径可比。某个 rank 的 loss 为 NaN 时，首先记录有限性失败和 collective 位置，不能继续把 spread 当作普通的数值差异。

## 9.11 Checkpoint 恢复与连续性

### 9.11.1 完整状态

可恢复训练通常需要：

~~~math
C_{\mathrm{resume}}
=
\{
\mathrm{model},
\mathrm{optimizer},
\mathrm{scheduler},
\mathrm{scaler},
\mathrm{rng},
\mathrm{data},
\mathrm{distributed},
\mathrm{tokenizer},
\mathrm{config},
\mathrm{step}
\}
~~~

只加载 model weights 只能说明权重可用于推理，不能说明训练轨迹连续。

### 9.11.2 连续性对照

在小模型或固定 batch 上：

1. 连续运行到 step 20；
2. 在 step 10 保存所有状态；
3. 从 step 10 恢复并再运行 10 步；
4. 比较第 11 步之后的输入、学习率、loss、梯度和参数。

硬件或 kernel 不同可能无法逐位一致，但差异应被量化。至少要比较有效 token、loss 窗口、参数范数、scheduler 和 checkpoint step。

### 9.11.3 checkpoint 完整性

分布式 checkpoint 应先写临时目录，完成 shard、manifest、大小和 checksum 校验后再发布版本指针。manifest 至少包含：

~~~text
tensor name
global shape
dtype
shard offsets
file name
checksum
world size and topology
creation step
~~~

“目录能打开”是文件可读性证据；“恢复后曲线连续”是行为证据，两者不能混为一谈。

## 9.12 最小可运行训练诊断示例

下面的示例只用 Python 标准库。它把 step 日志、batch 元信息、rank loss 和 checkpoint 字段对齐，检测候选 spike、非有限值、梯度异常、坏 batch、rank 差异和恢复缺项。它不模拟真实 autograd 或通信，只展示证据如何落到同一条时间轴上。

~~~python
import math
from statistics import median


history = [
    {"step": 1, "loss": 2.40, "grad_norm": 0.8},
    {"step": 2, "loss": 2.18, "grad_norm": 0.9},
    {"step": 3, "loss": 2.05, "grad_norm": 1.0},
    {"step": 4, "loss": 5.80, "grad_norm": 7.5},
    {"step": 5, "loss": 2.02, "grad_norm": 1.1},
    {"step": 6, "loss": float("nan"), "grad_norm": float("inf")},
]

batches = {
    1: {"id": "batch_001", "max_len": 2048, "valid_ratio": 0.82, "invalid": False},
    2: {"id": "batch_002", "max_len": 2048, "valid_ratio": 0.80, "invalid": False},
    3: {"id": "batch_003", "max_len": 2048, "valid_ratio": 0.79, "invalid": False},
    4: {"id": "batch_004_bad_mask", "max_len": 8192, "valid_ratio": 0.03, "invalid": False},
    5: {"id": "batch_005", "max_len": 2048, "valid_ratio": 0.78, "invalid": False},
    6: {"id": "batch_006_invalid", "max_len": 2048, "valid_ratio": 0.81, "invalid": True},
}

rank_losses = {4: [5.8, 5.7, 5.9, 5.8], 6: [2.1, float("nan"), 2.0, 2.1]}
checkpoint = {
    "model": True,
    "optimizer": True,
    "scheduler": False,
    "scaler": False,
    "data_position": False,
}

sequence_limit = 4096
valid_ratio_min = 0.2
grad_threshold = 5.0
spike_ratio = 2.0


def finite(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def ratio(name, value):
    if not finite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite ratio in [0, 1]")
    return float(value)


def validate_history(records):
    if not isinstance(records, list) or not records:
        raise ValueError("history must be a nonempty list")
    previous_step = 0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each history record must be a mapping")
        step = positive_int("step", record.get("step"))
        if step <= previous_step:
            raise ValueError("history steps must be strictly increasing")
        for name in ("loss", "grad_norm"):
            value = record.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
        previous_step = step


def validate_batch(batch):
    if not isinstance(batch, dict) or not isinstance(batch.get("id"), str):
        raise ValueError("batch must include a string id")
    positive_int("max_len", batch.get("max_len"))
    ratio("valid_ratio", batch.get("valid_ratio"))
    if not isinstance(batch.get("invalid"), bool):
        raise ValueError("invalid must be a boolean")


def validate_rank_losses(losses):
    if not isinstance(losses, list) or not losses:
        raise ValueError("rank losses must be nonempty")
    for value in losses:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("rank losses must be numeric")


validate_history(history)
for batch in batches.values():
    validate_batch(batch)
for losses in rank_losses.values():
    validate_rank_losses(losses)
if not isinstance(checkpoint, dict) or not checkpoint:
    raise ValueError("checkpoint must be a nonempty mapping")
if any(not isinstance(present, bool) for present in checkpoint.values()):
    raise ValueError("checkpoint fields must be booleans")
sequence_limit = positive_int("sequence_limit", sequence_limit)
valid_ratio_min = ratio("valid_ratio_min", valid_ratio_min)
if not finite(grad_threshold) or grad_threshold <= 0.0:
    raise ValueError("grad_threshold must be positive")
if not finite(spike_ratio) or spike_ratio <= 1.0:
    raise ValueError("spike_ratio must be greater than 1")


spikes = []
nonfinite = []
grad_explosions = []
bad_batches = []
rank_mismatches = []

for index, record in enumerate(history):
    step = record["step"]
    loss = record["loss"]
    grad_norm = record["grad_norm"]

    if not finite(loss) or not finite(grad_norm):
        nonfinite.append(step)

    previous = [
        item["loss"]
        for item in history[:index]
        if finite(item["loss"])
    ]
    if previous and finite(loss):
        baseline = median(previous[-3:])
        if baseline > 0.0 and loss / baseline >= spike_ratio:
            spikes.append((step, round(loss / baseline, 2)))

    if finite(grad_norm) and grad_norm > grad_threshold:
        grad_explosions.append((step, grad_norm))

    batch = batches[step]
    reasons = []
    if batch["max_len"] > sequence_limit:
        reasons.append("too_long")
    if batch["valid_ratio"] < valid_ratio_min:
        reasons.append("low_valid_ratio")
    if batch["invalid"]:
        reasons.append("invalid_value")
    if reasons:
        bad_batches.append((step, batch["id"], reasons))

for step, losses in rank_losses.items():
    finite_values = [value for value in losses if finite(value)]
    has_nonfinite = len(finite_values) != len(losses)
    spread = max(finite_values) - min(finite_values) if finite_values else float("inf")
    if has_nonfinite or spread > 0.5:
        rank_mismatches.append(
            (step, has_nonfinite, round(spread, 3) if finite(spread) else "inf")
        )

missing_checkpoint = [
    name for name, present in checkpoint.items() if not present
]
healthy_steps = [
    item["step"]
    for item in history
    if item["step"] not in nonfinite
    and item["step"] not in [step for step, _ in spikes]
]
rollback_step = max(
    step for step in healthy_steps
    if step < min(nonfinite or [10 ** 9])
)

recommendations = []
if bad_batches:
    recommendations.append("quarantine_bad_batches")
if grad_explosions:
    recommendations.append("lower_lr_or_enable_clipping")
if nonfinite:
    recommendations.append("inspect_dtype_and_loss_scaling")
if missing_checkpoint:
    recommendations.append("repair_checkpoint_continuity")
if rank_mismatches:
    recommendations.append("compare_rank_batches_and_collectives")

diagnostics = {
    "spike_detected": spikes == [(4, 2.66)],
    "nonfinite_detected": nonfinite == [6],
    "grad_explosion_detected": grad_explosions == [(4, 7.5)],
    "bad_batch_detected": bad_batches[0][1] == "batch_004_bad_mask",
    "rank_mismatch_detected": rank_mismatches == [(6, True, 0.1)],
    "checkpoint_gap_detected": missing_checkpoint == [
        "scheduler", "scaler", "data_position"
    ],
    "rollback_before_nonfinite": rollback_step == 5,
}

print("spikes=", spikes)
print("nonfinite=", nonfinite)
print("grad_explosions=", grad_explosions)
print("bad_batches=", bad_batches)
print("rank_mismatches=", rank_mismatches)
print("missing_checkpoint=", missing_checkpoint)
print("rollback_step=", rollback_step)
print("recommendations=", recommendations)
print("diagnostics=", diagnostics)

assert all(diagnostics.values())
print("training debug toy: ok")
~~~

示例的阈值和 batch 都是教学构造。真实项目要把 decode 文本、有效 token、数据来源、dtype、loss scale、rank、collective 位置和 checkpoint id 一并保存；示例只负责展示这些字段如何被组合成候选诊断。

## 9.13 把排查结果写成可复盘记录

一次高质量 debug 至少包含：

| 维度 | 记录内容 | 作用 |
| --- | --- | --- |
| 现象 | spike、NaN、OOM、hang、吞吐下降 | 明确要解释什么 |
| 时间 | 第一个异常 step、micro-step、phase | 找到先后关系 |
| 输入状态 | batch、decode、mask、lr、grad、scale、rank | 判断哪一层先异常 |
| 反事实 | 健康 checkpoint、单卡、小模型、替换 batch | 排除候选原因 |
| 变更回归 | 修复前后 loss、质量、吞吐、恢复 | 确认不是只改变症状 |

“把 lr 调小后恢复了”不是完整结论。完整结论应该说明：在哪个 step 发现什么证据，哪一个假设被哪一个对照支持，修复改变了什么，以及修复后是否保持了训练目标和质量口径。

## 9.14 资料边界与延伸阅读

本章把框架文档用于确认 API 行为，把原始论文用于确认机制，把 toy 示例用于固定日志和诊断口径。它们不能替代目标模型、数据、硬件和分布式配置的实测。

| 资料类型 | 能支持的结论 | 不能自动支持的结论 |
| --- | --- | --- |
| PyTorch autograd 文档 | anomaly detection 的调试接口 | 所有 NaN 的根因 |
| PyTorch clipping 文档 | global norm 与非有限值检查语义 | 通用 clipping threshold |
| PyTorch AMP 文档 | autocast、unscale、scaler 行为 | 任意模型的数值稳定性 |
| PyTorch DDP 文档 | 梯度同步与 rank 协作约束 | 目标集群的通信性能 |
| Distributed Checkpoint 文档 | 分布式保存加载接口 | 恢复后模型质量连续 |
| 梯度爆炸论文 | 梯度裁剪的机制背景 | 所有 LLM 的固定阈值 |
| 混合精度论文 | loss scaling 与 master weight 的方法 | 任意 FP8/FP16 recipe |

推荐阅读：

- PyTorch autograd anomaly detection：
  https://docs.pytorch.org/docs/stable/autograd.html
- PyTorch clip_grad_norm_：
  https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
- PyTorch automatic mixed precision：
  https://docs.pytorch.org/docs/stable/amp.html
- PyTorch DistributedDataParallel：
  https://docs.pytorch.org/docs/stable/notes/ddp.html
- PyTorch Distributed Checkpoint：
  https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
- On the difficulty of training Recurrent Neural Networks：
  https://arxiv.org/abs/1211.5063
- Mixed Precision Training：
  https://arxiv.org/abs/1710.03740

阈值、warmup 长度、是否跳过异常样本、是否回滚以及恢复后要求逐位一致还是统计一致，都属于目标系统的实验配置。资料可以说明机制和接口，不能替项目做这些选择。

## 9.15 章节练习：把现象变成证据

### 练习一：建立基线

为一次 1000 step 的训练设计日志字段，要求同时支持 loss、有效 token、grad norm、学习率、loss scale、吞吐、rank 差异和 checkpoint 恢复分析。说明每个字段的分母和 step 口径。

### 练习二：定位第一个坏张量

设计一个小模型实验，分别制造 forward NaN、backward NaN 和 optimizer step NaN。写出每种情况下应保存的 tensor、phase、dtype、shape 和 batch 信息。

### 练习三：loss spike 反事实

给定一个 spike batch、一个健康 batch 和 spike 前 checkpoint，设计三次重放来区分数据问题、学习率问题和恢复状态问题。

### 练习四：mask 审计

构造一个多轮对话，分别制造 user token 参与 loss、assistant token 被全部 mask、padding 参与 loss 和 label shift 错位。decode 每种情况的有效 label，并预测 loss 与模型行为。

### 练习五：变长梯度累积

两个 micro-batch 的有效 token 数分别为 100 和 300，未归一化梯度范数分别为 2 和 6。比较 token-level 累积与简单 batch mean 的权重差异，并说明 clipping 应放在什么位置。

### 练习六：分布式 hang

设计一个两 rank toy collective，其中 rank 1 在第 3 个 micro-step 跳过 all-reduce。记录每个 rank 的 collective 序号，说明主进程日志为什么不能直接告诉你根因。

### 练习七：恢复连续性

让 toy optimizer 连续运行 20 step，在第 10 步保存 model、optimizer、scheduler、scaler、RNG 和 data cursor。比较连续运行与恢复运行的第 11 步 loss、学习率、参数和有效 token。

## 9.16 本章小结

训练 debug 的第一原则是定位第一次异常，而不是先修改最显眼的超参数。健康基线应包含有效 token、loss 分母、梯度、更新量、学习率、数值状态、rank 状态、吞吐和 checkpoint 行为。

Loss spike 需要和 batch、mask、grad norm、学习率、warmup 和恢复状态对照；NaN/Inf 需要找到第一个坏张量；梯度爆炸和过小要结合全局/分层范数、相对更新量、有效监督和参数分组；数据问题要通过 decode、来源切片和 label mask 审计还原；混合精度问题要看 dtype、loss scale、unscale、kernel 和跳过的 update。

分布式异常要区分 collective 控制流和 rank 数据/数值差异，checkpoint 问题要区分文件可读性与恢复后轨迹连续性。一次可靠的修复必须能够被最小复现、反事实对照和同口径回归验证。

下一章进入 checkpoint 评估与实验管理，继续说明如何从训练过程中选择版本、组织实验和保存可以被复核的模型证据。
