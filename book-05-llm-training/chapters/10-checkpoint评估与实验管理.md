# 第十章：Checkpoint、评估与实验管理

## 本章目标

理解如何保存、恢复、比较和选择模型 checkpoint。

## 核心议题

1. checkpoint 保存内容：model、optimizer、scheduler、rng state、训练进度。
2. checkpoint 频率和存储成本。
3. 训练恢复和容错。
4. validation loss、perplexity、下游 benchmark。
5. 实验追踪、版本管理和可复现性。

## 面试重点

大规模训练不是跑一个脚本，而是长期实验系统。评估和实验管理决定结果是否可信。

## 为什么 checkpoint 和实验管理很重要

大模型训练成本很高，一次训练可能持续数天、数周甚至更久。

如果没有可靠 checkpoint，任何机器故障、网络故障、代码 bug 或 NaN 都可能让训练前功尽弃。

如果没有实验管理，即使跑出一个好模型，也很难回答：

1. 这个模型是用哪批数据训练的。
2. 训练了多少 token。
3. 用了什么学习率和 batch size。
4. 哪个 checkpoint 最好。
5. 相比上一个版本到底提升在哪里。
6. 是否有能力退化或安全风险。

面试表达：训练工程不只是把 loss 跑下来，还要保证训练过程可恢复、结果可比较、结论可复现。

### Checkpoint 与实验管理的资料边界

按照 `WRITING_PLAN.md` 的要求，本章第二轮精修前核对了 PyTorch 保存/加载通用 checkpoint、PyTorch Distributed Checkpoint、Hugging Face Transformers `Trainer` checkpoint 与 best model 配置、perplexity 评估口径，以及实验追踪和 artifact versioning 的官方实践。

本章聚焦训练工程中最小可落地的闭环：checkpoint 保存内容、恢复连续性、存储保留策略、validation / benchmark / regression 评估、实验记录、版本管理和可复现性。它不展开生产级对象存储、跨机房灾备、完整 MLOps 平台、权限审计系统和所有评测框架 API。

### 实验对象与总门禁

一次训练实验可以抽象成：

```math
e=
(c,d,m,h,a,r)
```

其中 `c` 是代码版本，`d` 是数据版本，`m` 是模型和 tokenizer 配置，`h` 是训练超参和硬件环境，`a` 是 checkpoint artifact，`r` 是评估结果。一个实验能不能被采信，可以先用下面的总门禁判断：

```math
G_{\mathrm{exp}}=
g_{\mathrm{ckpt}}\,
g_{\mathrm{eval}}\,
g_{\mathrm{repro}}\,
g_{\mathrm{compare}}\,
g_{\mathrm{safety}}
```

这些检查分别对应：checkpoint 是否完整、评估是否充分、记录是否可复现、对比是否公平、安全和回归风险是否可接受。

## 1. Checkpoint 应该保存什么

一个完整训练 checkpoint 不只是模型权重。

通常至少包含：

| 内容 | 作用 |
| --- | --- |
| model weights | 模型参数 |
| optimizer state | AdamW 的 m/v 等状态 |
| scheduler state | learning rate 进度 |
| scaler state | FP16 mixed precision 的 loss scale |
| RNG state | Python、NumPy、PyTorch、CUDA 随机状态 |
| global step | 当前训练步数 |
| consumed tokens | 已训练 token 数 |
| data state | dataloader 或 shard 位置 |
| model config | 架构配置 |
| tokenizer | tokenizer 文件和 special token |
| training config | batch、lr、并行、精度等配置 |

只保存 model weights 可以用于推理，但通常不能完整恢复训练。

如果 optimizer 或 scheduler 没恢复，继续训练时 loss 可能跳变。

恢复训练所需字段可以写成一个集合：

```math
C_{\mathrm{resume}}=
\{model,opt,sched,scaler,rng,data,step,tok,config,tokenizer\}
```

完整性门禁是：

```math
G_{\mathrm{ckpt}}=
\prod_{u\in C_{\mathrm{resume}}} g_u
```

其中 `g_u=1` 表示字段 `u` 存在、版本匹配且能被当前训练脚本读取。只要 `G_ckpt=0`，这个 checkpoint 就只能算部分可用，不能直接宣称能无缝恢复训练。

## 2. 保存频率怎么定

Checkpoint 保存太少，故障时损失大；保存太频繁，会浪费存储和训练时间。

需要权衡：

1. 训练成本。
2. 故障概率。
3. checkpoint 大小。
4. 存储带宽。
5. 恢复时间。
6. 评估频率。

常见策略：

```text
每 N steps 保存一次 latest checkpoint
每 M tokens 保存一个里程碑 checkpoint
保留最近 K 个 checkpoint
对关键 checkpoint 做长期归档
```

例如：

```text
latest: 用于故障恢复，只保留最近几个
milestone: 用于评估和对比，按 token 数保存
best: 根据评估指标选择，用于后训练或发布
```

如果每 `f` 个 step 保存一次，总训练步数为 `S`，保存次数约为：

```math
N_{\mathrm{save}}=
\left\lceil \frac{S}{f}\right\rceil
```

一次故障最多会损失最近一次保存后的 token：

```math
T_{\mathrm{lost,max}}=
fB_{\mathrm{tok}}
```

其中 `B_tok` 是每 step 消耗的 token 数。保存开销占比可以粗略估成：

```math
R_{\mathrm{save}}=
\frac{T_{\mathrm{save}}}{fT_{\mathrm{step}}}
```

`f` 越小，容错越好，但 `R_save` 和存储压力越高。

面试表达：checkpoint 策略要同时考虑容错、评估和存储成本。

## 3. Checkpoint 存储成本

大模型 checkpoint 非常大。

如果是 70B 模型，单模型权重就可能上百 GB。若再保存 optimizer state，体积可能数倍增加。

因此要区分两类 checkpoint：

1. 训练恢复 checkpoint：包含 optimizer、scheduler、scaler 等完整状态。
2. 推理/评估 checkpoint：只包含模型权重、config 和 tokenizer。

训练恢复 checkpoint 更大，但能继续训练。

推理 checkpoint 更小，适合评估、部署和归档。

如果使用 FSDP 或 ZeRO，checkpoint 还可能是 sharded 格式，需要专门的保存和加载逻辑。

设模型参数量为 `P`，权重每个参数占 `b_w` 字节，则推理权重大小近似为：

```math
M_{\mathrm{eval}}=
Pb_w+M_{\mathrm{meta}}
```

如果训练恢复还保存梯度、AdamW 一阶/二阶状态和 master weights，恢复 checkpoint 可以粗略估成：

```math
M_{\mathrm{resume}}
\approx
P(b_w+b_g+b_m+b_v+b_{\mathrm{master}})
+M_{\mathrm{meta}}
```

保留策略的总存储可以写成：

```math
M_{\mathrm{retain}}=
K_{\mathrm{latest}}M_{\mathrm{resume}}
+K_{\mathrm{best}}M_{\mathrm{eval}}
+K_{\mathrm{mile}}M_{\mathrm{eval}}
```

这里 `latest` 偏容错，`best` 偏下游使用，`mile` 偏实验复盘。真实项目还要考虑 sharded checkpoint 合并、对象存储带宽和跨节点读取失败。

## 4. 训练恢复和容错

恢复训练时，要验证三件事：

1. 能否成功加载。
2. 恢复后 loss 是否连续。
3. 恢复后学习率和 step 是否正确。

一个健康恢复应该表现为：

```text
恢复前后 loss 曲线连续
learning rate 不从头开始
optimizer 动量状态存在
global step 和 token count 正确
```

常见恢复错误：

1. 只加载 model weights。
2. scheduler 从 0 开始。
3. optimizer state 缺失。
4. tokenizer 版本不一致。
5. 分布式 shard 数变化导致加载失败。
6. 数据迭代位置变化，重复或跳过大量数据。

恢复后要做连续性检查。设恢复前最后一个健康日志为 `-`，恢复后第一个日志为 `+`：

```math
\delta_L=
\frac{|L^+-L^-|}{L^-+\epsilon}
```

```math
\delta_{\eta}=
\frac{|\eta^+-\eta^-|}{\eta^-+\epsilon}
```

```math
\delta_{\mathrm{step}}=
s^+-s^- - 1
```

一个简化恢复门禁是：

```math
G_{\mathrm{resume}}=
\mathbf{1}[\delta_L\le \tau_L]\,
\mathbf{1}[\delta_{\eta}\le \tau_{\eta}]\,
\mathbf{1}[\delta_{\mathrm{step}}=0]\,
\mathbf{1}[T^+>T^-]
```

如果 load 成功但 `G_resume=0`，说明训练状态可能不连续，必须先定位 scheduler、optimizer、random state 或 data position 的问题。

面试表达：checkpoint 恢复不是“能 load 成功”就结束，还要检查 loss、lr、step 和数据状态是否连续。

## 5. 为什么不能只看最后一个 checkpoint

很多初学者默认最后一步最好，但真实训练不一定。

可能出现：

1. 后期过拟合某些数据。
2. 某些能力退化。
3. 安全拒答变差。
4. checkpoint 附近发生 loss spike。
5. 后训练中风格过拟合。

所以 checkpoint 选择是多目标决策，不是只看 step 最大。

需要比较：

1. validation loss。
2. per-domain loss。
3. 下游 benchmark。
4. 人工评测。
5. safety eval。
6. regression eval。
7. 推理延迟和成本。

## 6. Validation loss 和 perplexity

预训练阶段最基础的指标是 validation loss。

Perplexity 可以理解为 cross entropy loss 的指数形式：

```text
perplexity = exp(loss)
```

对 causal LM，带 mask 的 validation loss 可以写成：

```math
L_{\mathrm{val}}=
-
\frac{
\sum_i\sum_t m_{i,t}\log p_{\theta}(x_{i,t}\mid x_{i,<t})
}{
\sum_i\sum_t m_{i,t}
}
```

对应 perplexity 为：

```math
PPL=
\exp(L_{\mathrm{val}})
```

loss 越低，perplexity 越低，说明模型对验证集 token 的预测越好。

但要注意：

1. validation set 必须干净且不泄漏。
2. 不同 tokenizer 的 loss/perplexity 不宜直接比较。
3. 总 loss 可能掩盖某些 domain 退化。
4. loss 下降不代表所有能力提升。

面试表达：validation loss 是基础健康指标，但不是完整能力评估。

## 7. Per-domain loss

只看整体 validation loss 不够。

应该按 domain 分开看：

| Domain | 可能观察 |
| --- | --- |
| general web | 通用语言能力 |
| code | 代码建模能力 |
| math | 数学符号和推理模式 |
| zh | 中文能力 |
| en | 英文能力 |
| academic | 论文和专业文本 |
| dialogue | 对话格式能力 |

例如整体 loss 下降，但代码 loss 上升，说明模型可能在代码能力上退化。

这种情况在数据配比调整后很常见。

设验证集按 domain 分成 `D_k`，每个 domain 的 loss 为：

```math
L_k=
-
\frac{
\sum_{(i,t)\in D_k}m_{i,t}\log p_{\theta}(x_{i,t}\mid x_{i,<t})
}{
\sum_{(i,t)\in D_k}m_{i,t}
}
```

整体 loss 可以看成加权平均：

```math
L_{\mathrm{all}}=
\sum_k w_kL_k
```

其中 `w_k` 来自各 domain 的有效 token 占比。checkpoint 对比时，不仅要看 `L_all` 是否下降，也要看是否存在：

```math
L_k^{new}-L_k^{old}>\tau_k
```

这类 domain regression。

面试表达：per-domain loss 能帮助判断模型是全面变好，还是只在某些数据分布上变好。

## 8. 下游 benchmark

Base model 训练阶段也需要下游 benchmark。

常见评估维度：

1. 知识问答。
2. 数学推理。
3. 代码生成。
4. 阅读理解。
5. 多语言能力。
6. 长上下文能力。
7. 安全和毒性。

但 benchmark 有风险：

1. 可能被污染。
2. 可能和真实目标不一致。
3. 可能只反映某一类能力。
4. 分数提升可能没有统计显著性。

所以 benchmark 要和 validation loss、人工评测、回归测试一起看。

## 9. Regression eval

模型新版本不能只看新能力是否提升，还要看旧能力是否退化。

Regression eval 的目标是发现：

```text
旧模型答对，新模型答错
```

这类样本叫 loss cases。

设旧模型正确标记为 `z_i^{old}`，新模型正确标记为 `z_i^{new}`，回归率可以写成：

```math
R_{\mathrm{reg}}=
\frac{
\sum_i \mathbf{1}[z_i^{old}=1]\mathbf{1}[z_i^{new}=0]
}{
\sum_i \mathbf{1}[z_i^{old}=1]+\epsilon
}
```

如果样本带业务权重 `w_i`，高风险回归率可以写成：

```math
R_{\mathrm{reg,w}}=
\frac{
\sum_i w_i\mathbf{1}[z_i^{old}=1]\mathbf{1}[z_i^{new}=0]
}{
\sum_i w_i\mathbf{1}[z_i^{old}=1]+\epsilon
}
```

常见回归维度：

1. 格式遵循。
2. 数学和代码。
3. 安全拒答。
4. 多语言。
5. 企业业务规则。
6. 长上下文引用。

面试表达：模型评估不能只看平均分提升，要重点看高风险场景有没有退化。

## 10. 实验追踪应该记录什么

一个训练实验至少要记录：

| 类别 | 内容 |
| --- | --- |
| 代码 | git commit、分支、diff |
| 数据 | 数据版本、配比、过滤规则、token 数 |
| 模型 | 架构、参数量、tokenizer、context length |
| 训练 | lr、batch、optimizer、schedule、precision、parallelism |
| 环境 | GPU 类型、数量、驱动、框架版本 |
| 日志 | loss、grad norm、lr、tokens/sec、OOM/NaN |
| checkpoint | 保存路径、step、token count、评估结果 |
| 结论 | 提升、退化、异常、下一步 |

没有这些记录，实验就不可复现。

实验记录可以抽象为：

```math
R_{\mathrm{run}}=
(v_{\mathrm{code}},v_{\mathrm{data}},v_{\mathrm{tok}},v_{\mathrm{cfg}},h,s,A,M)
```

其中 `v_code` 是代码版本，`v_data` 是数据版本，`v_tok` 是 tokenizer 版本，`v_cfg` 是模型和训练配置版本，`h` 是硬件和依赖环境，`s` 是随机种子，`A` 是 artifact 列表，`M` 是指标表。最小可复现门禁为：

```math
G_{\mathrm{repro}}=
g_{\mathrm{code}}\,
g_{\mathrm{data}}\,
g_{\mathrm{tok}}\,
g_{\mathrm{cfg}}\,
g_{\mathrm{env}}\,
g_{\mathrm{seed}}\,
g_{\mathrm{eval}}
```

面试表达：实验管理的核心是让别人能复现你的训练结果，也能理解每次变化来自哪里。

## 11. 实验命名和版本管理

实验命名要能表达关键信息。

坏命名：

```text
run1
test_new
final_v2
```

更好的命名：

```text
7b_pretrain_lr3e-4_bs4m_tok2t_data-v5_bf16_fsdp
```

不一定要很长，但至少应该能看出：

1. 模型规模。
2. 训练阶段。
3. 数据版本。
4. 关键超参。
5. 时间或 run id。

版本管理包括：

1. 代码版本。
2. 数据版本。
3. tokenizer 版本。
4. config 版本。
5. checkpoint 版本。

其中数据版本尤其重要。很多训练差异来自数据变化，而不是模型或优化器变化。

## 12. 可复现性

大模型训练完全 bit-level 复现很难，但工程上至少要做到结论可复现。

影响复现的因素：

1. 随机种子。
2. 数据顺序。
3. 分布式并行策略。
4. kernel 非确定性。
5. mixed precision。
6. checkpoint 恢复状态。
7. 依赖库版本。

可复现性最低要求：

```text
同样代码 + 同样数据 + 同样配置，可以得到相近 loss 曲线和相近评估结论。
```

不要承诺所有大规模分布式训练都能逐 bit 复现。

## 13. 实验对比的基本原则

比较两个实验时，要尽量做到单变量控制。

例如你想比较学习率：

```text
只改 learning rate
其他数据、模型、batch、seed、训练 token 尽量一致
```

如果同时改了数据、学习率和 tokenizer，就很难判断提升来自哪里。

常见错误：

1. 不同训练 token 数直接比较。
2. 不同 tokenizer 直接比较 loss。
3. 不同数据版本却归因到 optimizer。
4. 只看一个 benchmark。
5. 没有统计波动和多次实验。

如果两个实验配置分别是 `h_A` 和 `h_B`，想验证的变量集合是 `V`，公平对比要求除 `V` 外的关键字段一致：

```math
D_{\mathrm{diff}}=
\{u\mid h_A(u)\ne h_B(u),u\notin V\}
```

```math
G_{\mathrm{compare}}=
\mathbf{1}[D_{\mathrm{diff}}=\varnothing]\,
\mathbf{1}[T_A=T_B]\,
\mathbf{1}[E_A=E_B]
```

其中 `T_A/T_B` 是训练 token 数，`E_A/E_B` 是评估集和评估脚本版本。真实实验中很难做到所有变量完全一致，但至少要把差异记录清楚，不能把多变量变化包装成单变量结论。

面试表达：实验结论可信的前提是对照清晰、变量可控、评估充分。

## 14. Checkpoint 选择流程

一个实用选择流程：

```text
1. 过滤掉训练异常 checkpoint。
2. 按 validation loss 选候选。
3. 看 per-domain loss，排除明显偏科。
4. 跑核心 benchmark。
5. 做 regression eval。
6. 做 safety eval。
7. 抽样人工评测。
8. 评估推理成本。
9. 选择进入后训练或部署的 checkpoint。
```

可以把 checkpoint 选择写成多目标打分：

```math
J(c)=
-\alpha L_{\mathrm{val}}(c)
+\beta S_{\mathrm{bench}}(c)
+\gamma S_{\mathrm{safety}}(c)
-\lambda R_{\mathrm{reg}}(c)
-\mu C_{\mathrm{infer}}(c)
```

这里 `S_bench` 是 benchmark 汇总分，`S_safety` 是安全评估分，`R_reg` 是回归风险，`C_infer` 是推理成本。权重不是固定真理，而是要和项目目标绑定：base model 可能更重视 validation / benchmark，面向企业部署的模型可能更重视 regression / safety / latency。

这比“最后一个 checkpoint”稳健很多。

## 15. 最小可运行实验管理审计 demo

下面的 demo 用标准库模拟一次 checkpoint 与实验管理审计。输入是 3 个 checkpoint、一次恢复探针、一个实验记录和两组对照实验；输出 checkpoint 完整性、PPL、domain regression、候选打分、保留存储、恢复连续性、实验记录完整性和对照是否公平。

```python
import math
from statistics import mean

required_resume_fields = [
    "model", "optimizer", "scheduler", "scaler", "rng", "data_position",
    "global_step", "consumed_tokens", "config", "tokenizer",
]

checkpoints = [
    {
        "id": "ckpt_1000", "step": 1000, "tokens_b": 52.4, "train_loss": 2.44, "val_loss": 2.31,
        "domain_loss": {"general": 2.20, "code": 2.90, "math": 2.70},
        "bench": {"qa": 0.52, "math": 0.31, "code": 0.28},
        "safety": 0.95, "regression_failures": 5, "stable": True, "size_gib": 180,
        "resume_fields": {name: True for name in required_resume_fields},
    },
    {
        "id": "ckpt_2000", "step": 2000, "tokens_b": 104.9, "train_loss": 2.29, "val_loss": 2.18,
        "domain_loss": {"general": 2.08, "code": 2.74, "math": 2.52},
        "bench": {"qa": 0.56, "math": 0.38, "code": 0.34},
        "safety": 0.94, "regression_failures": 4, "stable": True, "size_gib": 180,
        "resume_fields": {name: True for name in required_resume_fields},
    },
    {
        "id": "ckpt_3000", "step": 3000, "tokens_b": 157.3, "train_loss": 2.20, "val_loss": 2.16,
        "domain_loss": {"general": 2.05, "code": 2.98, "math": 2.41},
        "bench": {"qa": 0.57, "math": 0.40, "code": 0.29},
        "safety": 0.89, "regression_failures": 11, "stable": False, "size_gib": 180,
        "resume_fields": {**{name: True for name in required_resume_fields}, "scaler": False, "data_position": False},
    },
]

resume_probe = {
    "checkpoint_id": "ckpt_2000",
    "loss_before": 2.19,
    "loss_after": 2.20,
    "lr_before": 2.8e-4,
    "lr_after": 2.8e-4,
    "step_before": 2000,
    "step_after": 2001,
    "tokens_before_b": 104.90,
    "tokens_after_b": 104.95,
}

run_manifest = {
    "git_commit": "abc1234", "data_version": "data_v5", "tokenizer_version": "tok_v2",
    "model_config": "7b_gqa_rope", "train_config": "lr3e-4_bs4m_bf16_fsdp",
    "env": "64xa100_cuda12", "seed": 42, "eval_script": "eval_v3.py",
    "checkpoint_id": "ckpt_2000", "metrics": {"val_loss": 2.18}, "notes": "lr sweep winner",
}

run_a = {"data_version": "data_v5", "tokenizer": "tok_v2", "tokens_b": 104.9, "lr": 3e-4, "batch_tokens": 4_000_000, "seed": 42}
run_b = {"data_version": "data_v5", "tokenizer": "tok_v2", "tokens_b": 104.9, "lr": 2e-4, "batch_tokens": 4_000_000, "seed": 42}
run_bad = {"data_version": "data_v6", "tokenizer": "tok_v2", "tokens_b": 157.3, "lr": 2e-4, "batch_tokens": 4_000_000, "seed": 42}

base = checkpoints[0]
base_val_loss = base["val_loss"]
storage_budget_gib = 400
latest_k = 2
save_interval_steps = 1000
step_seconds = 8
checkpoint_write_seconds = 60

def missing_fields(ckpt):
    return [name for name in required_resume_fields if not ckpt["resume_fields"].get(name, False)]

def perplexity(loss):
    return round(math.exp(loss), 2)

def domain_regressions(ckpt, threshold=0.05):
    return [
        domain
        for domain, loss in ckpt["domain_loss"].items()
        if loss - base["domain_loss"][domain] > threshold
    ]

def score_checkpoint(ckpt):
    bench_avg = mean(ckpt["bench"].values())
    domain_penalty = 0.10 * len(domain_regressions(ckpt))
    regression_penalty = 0.02 * ckpt["regression_failures"]
    stability_penalty = 0 if ckpt["stable"] else 0.50
    score = (base_val_loss - ckpt["val_loss"]) + bench_avg + ckpt["safety"]
    score -= domain_penalty + regression_penalty + stability_penalty
    return round(score, 4)

def changed_keys(a, b, ignore=("lr",)):
    return [key for key in a if key in b and a[key] != b[key] and key not in ignore]

completeness = {ckpt["id"]: missing_fields(ckpt) for ckpt in checkpoints}
ppl = {ckpt["id"]: perplexity(ckpt["val_loss"]) for ckpt in checkpoints}
domain_report = {ckpt["id"]: domain_regressions(ckpt) for ckpt in checkpoints}
score_table = [
    {"id": ckpt["id"], "score": score_checkpoint(ckpt), "stable": ckpt["stable"]}
    for ckpt in checkpoints
]
best = max(
    [ckpt for ckpt in checkpoints if not missing_fields(ckpt)],
    key=score_checkpoint,
)

latest_retained = sorted(checkpoints, key=lambda ckpt: ckpt["step"], reverse=True)[:latest_k]
retained_ids = sorted({best["id"], *(ckpt["id"] for ckpt in latest_retained)})
retained_storage_gib = sum(ckpt["size_gib"] for ckpt in checkpoints if ckpt["id"] in retained_ids)
save_overhead = round(checkpoint_write_seconds / (save_interval_steps * step_seconds), 4)
max_lost_tokens_b = round((checkpoints[1]["tokens_b"] - checkpoints[0]["tokens_b"]), 1)

loss_jump = abs(resume_probe["loss_after"] - resume_probe["loss_before"]) / resume_probe["loss_before"]
resume_checks = {
    "loss_continuous": loss_jump < 0.02,
    "lr_continuous": resume_probe["lr_before"] == resume_probe["lr_after"],
    "step_continuous": resume_probe["step_after"] == resume_probe["step_before"] + 1,
    "tokens_increase": resume_probe["tokens_after_b"] > resume_probe["tokens_before_b"],
}

manifest_required = [
    "git_commit", "data_version", "tokenizer_version", "model_config", "train_config",
    "env", "seed", "eval_script", "checkpoint_id", "metrics", "notes",
]
manifest_missing = [name for name in manifest_required if name not in run_manifest]
fair_lr_compare = changed_keys(run_a, run_b) == []
bad_compare_changed = changed_keys(run_a, run_bad)

checks = {
    "best_complete": completeness[best["id"]] == [],
    "resume_continuous": all(resume_checks.values()),
    "storage_under_budget": retained_storage_gib <= storage_budget_gib,
    "manifest_complete": manifest_missing == [],
    "fair_lr_compare": fair_lr_compare,
    "bad_compare_detected": bad_compare_changed == ["data_version", "tokens_b"],
    "safety_ok": best["safety"] >= 0.92,
    "regression_ok": best["regression_failures"] <= 5,
}

print("completeness=", completeness)
print("ppl=", ppl)
print("domain_report=", domain_report)
print("score_table=", score_table)
print("best_checkpoint=", best["id"])
print("retained_ids=", retained_ids)
print("retained_storage_gib=", retained_storage_gib)
print("save_overhead=", save_overhead)
print("max_lost_tokens_b=", max_lost_tokens_b)
print("resume_checks=", resume_checks)
print("manifest_missing=", manifest_missing)
print("fair_lr_compare=", fair_lr_compare)
print("bad_compare_changed=", bad_compare_changed)
print("checks=", checks)
print("gate_pass=", all(checks.values()))
```

期望输出：

```text
completeness= {'ckpt_1000': [], 'ckpt_2000': [], 'ckpt_3000': ['scaler', 'data_position']}
ppl= {'ckpt_1000': 10.07, 'ckpt_2000': 8.85, 'ckpt_3000': 8.67}
domain_report= {'ckpt_1000': [], 'ckpt_2000': [], 'ckpt_3000': ['code']}
score_table= [{'id': 'ckpt_1000', 'score': 1.22, 'stable': True}, {'id': 'ckpt_2000', 'score': 1.4167, 'stable': True}, {'id': 'ckpt_3000', 'score': 0.64, 'stable': False}]
best_checkpoint= ckpt_2000
retained_ids= ['ckpt_2000', 'ckpt_3000']
retained_storage_gib= 360
save_overhead= 0.0075
max_lost_tokens_b= 52.5
resume_checks= {'loss_continuous': True, 'lr_continuous': True, 'step_continuous': True, 'tokens_increase': True}
manifest_missing= []
fair_lr_compare= True
bad_compare_changed= ['data_version', 'tokens_b']
checks= {'best_complete': True, 'resume_continuous': True, 'storage_under_budget': True, 'manifest_complete': True, 'fair_lr_compare': True, 'bad_compare_detected': True, 'safety_ok': True, 'regression_ok': True}
gate_pass= True
```

这个 demo 的关键是把“选哪个 checkpoint”拆成可审计步骤：先过滤不完整和不稳定 checkpoint，再看 validation / PPL / domain / benchmark / safety / regression，最后检查实验记录是否完整、对照是否公平、保留策略是否超过存储预算。

## 16. 面试官会怎么问

### 问法 1：checkpoint 里应该保存什么？

可以这样答：

```text
如果只是推理，保存 model weights、config 和 tokenizer 基本够用。如果要恢复训练，还要保存 optimizer state、scheduler state、mixed precision scaler、global step、consumed tokens、random state、data state 和分布式状态。否则恢复后 loss、lr 或数据顺序可能不连续。
```

### 问法 2：怎么选择最好的 checkpoint？

可以这样答：

```text
不能只选最后一步。我会先看 validation loss 和 per-domain loss，再跑目标 benchmark、regression eval 和 safety eval。对于候选 checkpoint，还要看训练是否稳定、是否有能力退化、推理成本是否可接受，必要时做人工评测。
```

### 问法 3：为什么 train loss 降低不代表模型更好？

可以这样答：

```text
train loss 降低可能来自数据重复、训练集泄漏、过拟合或更容易的数据分布。它不一定代表下游能力、安全性或泛化提升。所以要看 validation loss、per-domain loss、benchmark、人工评测和 regression eval。
```

### 问法 4：如何保证实验可复现？

可以这样答：

```text
我会记录代码 commit、数据版本、tokenizer、模型 config、训练超参、硬件环境、随机种子、checkpoint 和评估脚本版本。大规模训练不一定能 bit-level 复现，但至少要保证同样配置能得到相近 loss 曲线和相近评估结论。
```

### 问法 5：两个实验怎么公平比较？

可以这样答：

```text
尽量做单变量对照，保持数据、模型、训练 token、batch、评估集和脚本一致，只改变想验证的因素。比较时不能只看单个平均分，要看 per-domain、regression、safety 和统计波动。
```

## 17. 本章小结

本章核心结论：

1. Checkpoint 是训练容错、评估和版本管理的核心产物。
2. 恢复训练需要 model、optimizer、scheduler、scaler、随机状态和数据状态。
3. 保存频率要在容错、评估和存储成本之间平衡。
4. 不能默认最后一个 checkpoint 最好。
5. Validation loss 是基础指标，但不是完整能力评估。
6. Per-domain loss 能发现模型偏科和能力退化。
7. Benchmark 要结合污染检测、回归测试和人工评测使用。
8. 实验追踪要记录代码、数据、模型、训练配置、环境和结论。
9. 可复现性要求结论可复现，不一定要求 bit-level 完全一致。
10. 面试中要把 checkpoint、评估和实验管理讲成一个长期训练系统。
