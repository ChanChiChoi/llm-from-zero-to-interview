# 第 9 章（5）：微调后的能力回归

训练稳定不代表微调没有代价。一个模型可以更会回答新业务问题，同时更不会遵守原有格式、处理通用问题或遵守安全边界。这一篇把目标能力和保留能力分开测量，再比较回放、降低更新强度、输出锚定、参数正则与任务路由等解决路线。

## 9.5 微调后的能力退化：从目标任务提升到整体能力衡量

### 9.5.1 一个模型可以同时变好和变坏

微调完成后，目标任务分数上涨并不意味着模型整体能力上涨。一个客服模型可能更会使用企业术语，却更容易把普通问题拒答；一个代码模型可能更擅长某种仓库风格，却不再遵守 JSON 协议；一个领域模型可能在专业问答上提高，却忘记通用数学、语言或安全边界。

这种现象通常被称为灾难性遗忘，但实际项目中的退化不止一种：

1. 记忆遗忘：新分布的梯度改变了原有参数表示。
2. 任务干扰：新任务与旧任务要求不同的输出策略。
3. 格式回归：模型学会目标内容，却破坏结构化协议、停止符或工具 schema。
4. 安全回归：窄领域数据覆盖了原有拒答、隐私和权限行为。
5. 分布回归：训练数据与线上输入的长度、语言、噪声或多轮形式不同。
6. 解码回归：模型参数变化不大，但最佳温度、停止条件和长度分布发生变化。

因此，评估微调不能只保留一个 target score。至少要同时测量目标任务、通用能力、格式、安全、长短输入、旧版本回归和资源代价。

### 分开看新任务和原有任务

第一张是“它有没有学会新任务”，第二张是“它有没有破坏原来会做的事情”。如果只看第一张，任何过拟合都可能被误认为成功；如果只看第二张，又会错过有价值的领域适配。

把旧能力样本记为 $D_{\mathrm{retain}}$，新任务样本记为 $D_{\mathrm{target}}$，至少报告：

```text
target_score_before -> target_score_after
retain_score_before -> retain_score_after
format_validity_before -> format_validity_after
safety_error_before -> safety_error_after
latency_and_cost_before -> latency_and_cost_after
```

目标任务提升和保留能力下降同时出现时，问题不是“模型到底好还是坏”，而是需要判断业务愿意支付多少退化代价，或者是否应采用路由、混合数据和更小的参数更新。

### 目标任务与保留能力是两类目标

设目标任务损失为 $\mathcal{L}_{\mathrm{target}}$，保留集损失为 $\mathcal{L}_{\mathrm{retain}}$，协议和安全约束分别为 $\mathcal{L}_{\mathrm{format}}$ 与 $\mathcal{L}_{\mathrm{safety}}$，一种带锚定项的抽象目标是：

```math
\mathcal{L}_{\mathrm{total}}=
\mathcal{L}_{\mathrm{target}}
+\lambda_{\mathrm{retain}}\mathcal{L}_{\mathrm{retain}}
+\lambda_{\mathrm{format}}\mathcal{L}_{\mathrm{format}}
+\lambda_{\mathrm{safety}}\mathcal{L}_{\mathrm{safety}}
+\lambda_{\mathrm{anchor}}D\bigl(p_\theta\Vert p_{\theta_0}\bigr)
+\Omega(\theta)
```

$\theta_0$ 是微调前模型，$D$ 可以是输出分布的 KL 距离，$\Omega$ 可以是参数或 adapter 的正则项。实际训练未必显式使用这一个总目标，但这个分解提醒我们：只优化新任务 loss，就没有机制保护未被采样的旧能力。

目标提升量和保留退化量分别为：

```math
\Delta_{\mathrm{target}}=S_{\mathrm{target}}(\theta)-S_{\mathrm{target}}(\theta_0)
```

```math
\Delta_{\mathrm{retain}}=S_{\mathrm{retain}}(\theta)-S_{\mathrm{retain}}(\theta_0)
```

当 $\Delta_{\mathrm{target}}>0$ 而 $\Delta_{\mathrm{retain}}<0$ 时，应报告这两个方向，而不是把它们混成一个平均分。平均分掩盖切片退化的方式，与[第一篇讨论的 micro average](09-评估与debug实战-1-评测协议与benchmark.md)风险相同。

### 9.5.2 先排除“评估看起来退化”的假象

微调后分数下降，不一定是遗忘。首先检查以下条件是否保持一致：

```text
模型调用是否使用同一个 chat template
system、user、assistant 角色是否一致
tokenizer 是否改变，特殊 token 是否映射一致
temperature、top_p、max_tokens 是否一致
停止词和 JSON 解析规则是否一致
上下文长度、截断方向和 padding 是否一致
评估数据是否发生时间、权限或文档版本变化
```

例如，模型本身能够生成正确 JSON，但新的 tokenizer 把结束符处理不同，runner 在等待停止符时截断了输出；这应该归入协议或服务回归，而不是直接写成“模型遗忘 JSON”。

### 9.5.3 退化曲线与有效训练预算

训练步数、学习率和数据重复次数共同决定模型被新分布推动多远。设训练 token 数为 $T$，每个样本平均长度为 $\ell$，样本重复或 epoch 数为 $e$，则粗略关系为：

```math
T\approx e\cdot N\cdot\ell
```

当数据集很小而 epoch 很多时，模型在有限表达模式上反复更新，容易提高目标集分数并损失多样性。学习率过大时，单步参数变化更显著；对 LoRA 等参数高效微调，rank、alpha 和目标模块也会改变可表达的更新空间。

可以记录相对更新量：

```math
r_{\mathrm{update}}=\frac{\lVert\theta-\theta_0\rVert_2}{\lVert\theta_0\rVert_2},\quad \lVert\theta_0\rVert_2>0
```

这个比值只在 $\lVert\theta_0\rVert_2>0$ 时有定义；若基座参数范数为 0，应报告相对更新量不适用，不能用任意 $\varepsilon$ 掩盖分母为 0。它不能单独预测质量，但能帮助比较不同微调方案是否施加了相近强度的改变。对 LoRA 等参数高效微调，还应记录 adapter 参数范数、合并前后行为以及目标层集合。

### 9.5.4 解决退化的几条路线

#### 路线一：混合回放数据

在新任务 batch 中混入旧能力、通用、安全和格式样本，让梯度持续看到需要保留的行为。混合比例不应只按样本数量决定，因为安全和结构化失败的代价可能远高于普通问答。

设新任务 batch 比例为 $\alpha$，回放比例为 $1-\alpha$：

```math
\mathcal{L}_{\mathrm{mix}}=
\alpha\mathcal{L}_{\mathrm{target}}+
(1-\alpha)\mathcal{L}_{\mathrm{replay}}
```

这里 $0\le\alpha\le1$。新任务比例 $\alpha$ 太低，适配可能较慢或不足；太高则对旧能力的保护减弱。上述损失混合等价于按该比例采样，需要两类损失的归一化口径一致；按样本采样但按 token 平均时，还应考虑长度差异。对回放数据要保留版本和去重信息，避免把测试样例重新混入训练。

#### 路线二：降低更新强度

降低学习率、减少 epoch、使用 warmup、冻结更多层或限制 adapter 目标模块，通常能减少大范围漂移。它们可能牺牲目标任务峰值，但有助于保留广泛能力。

#### 路线三：输出分布锚定

在保留样本上让微调模型不要偏离基座模型过远，可使用 KL 或 logits 蒸馏：

```math
\mathcal{L}_{\mathrm{KL}}=
\frac{1}{|T|}\sum_{t\in T}
D_{\mathrm{KL}}\left(
p_{\theta_0}(\cdot\mid x_{1:t-1})
\Vert
p_{\theta}(\cdot\mid x_{1:t-1})
\right)
```

这里要求有效位置集合 $T$ 非空、两模型使用同一词表和上下文，且学生分布在教师具有正概率的位置也为正。此处是教师到学生的 KL，前面的总目标只示意一种可能的锚定方向；交换方向会改变优化行为，不能混用。它能约束输出分布，却不自动保证事实、安全或格式；如果基座模型在某个任务上本来就错误，盲目锚定会保留错误。

#### 路线四：参数正则与重要性保护

Elastic Weight Consolidation 使用参数重要性估计，惩罚关键参数偏离旧值：

```math
\mathcal{L}_{\mathrm{EWC}}=
\mathcal{L}_{\mathrm{target}}
+\frac{\lambda}{2}\sum_i F_i(\theta_i-\theta_{0,i})^2
```

$F_i$ 常由 Fisher 信息近似。它的直觉是：对旧任务重要的参数不应被新任务随意改变；但重要性估计本身有成本，且不同任务之间可能共享参数，不能保证完全消除干扰。

#### 路线五：路由和分工

如果新任务与通用任务冲突明显，可以使用场景路由、adapter 路由或多个专门模型，而不是要求一个模型在所有目标上用同一组参数达到最优。路由会增加服务复杂度、缓存和评估组合数，但有时比继续扩大单一模型的训练约束更可控。
### 9.5.5 用一个退化矩阵理解方案取舍

假设三种微调方案得到如下结果，分数均归一化到 0--1：

```text
方案 A：目标 0.86，通用 0.80，格式 0.98，安全 0.97，成本 1.00
方案 B：目标 0.91，通用 0.72，格式 0.94，安全 0.95，成本 1.08
方案 C：目标 0.88，通用 0.84，格式 0.99，安全 0.98，成本 1.03
```

方案 B 的目标分最高，却在多个关键切片下降；方案 C 可能更适合产品发布。若只取四个分数的平均，B 与 C 的差距可能被掩盖。正确做法是先定义不可接受的退化条件，再在满足条件的方案中比较收益和成本。

可以用一个简化的效用表示：

```math
U_{\mathrm{deploy}}=w_tS_{\mathrm{target}}+w_gS_{\mathrm{general}}+w_fS_{\mathrm{format}}+w_sS_{\mathrm{safety}}-\lambda_c(C-C_0)
```

这里的各项分数应在同一约定的 $[0,1]$ 范围内，$w_t,w_g,w_f,w_s\ge0$ 且权重和为 1；$C$ 是候选方案的成本，$C_0$ 是比较基线成本，两者应使用相同单位且为有限非负数，$\lambda_c\ge0$ 用于把成本差映射到分数尺度。这只是决策账本，不是训练目标。权重应由业务风险和用户后果决定；安全和权限类指标往往不适合仅靠加权平均处理。

例如使用 §9.5.6 `adapter_large` 的结果，权重为 $(0.40,0.20,0.15,0.25)$，候选与基线成本分别为 1.08 和 1.00、$\lambda_c=0.05$：效用为 $0.40\times0.90+0.20\times0.73+0.15\times0.94+0.25\times0.95-0.05\times(1.08-1.00)=0.8805$。相对基线，它的通用、格式和安全分数分别下降 0.09、0.04 和 0.03；如果发布规则要求这些维度最多下降 0.01，即使效用排序可接受，也必须先按硬门槛拒绝该方案。

### 9.5.6 一个零依赖退化分析器

下面的程序输入基座模型和微调模型在不同切片上的结果，计算相对变化、加权效用和最差切片。它不替代人工分析，但能防止只打印一个总平均分。

> 代码状态：静态审阅，未运行（2026-10-30）

```python
import json
import math


BASELINE = {
    "target": 0.70,
    "general": 0.82,
    "format": 0.98,
    "safety": 0.98,
}

VARIANTS = {
    "adapter_small": {
        "target": 0.84,
        "general": 0.81,
        "format": 0.98,
        "safety": 0.98,
        "cost": 1.02,
    },
    "adapter_large": {
        "target": 0.90,
        "general": 0.73,
        "format": 0.94,
        "safety": 0.95,
        "cost": 1.08,
    },
    "mixed_replay": {
        "target": 0.87,
        "general": 0.84,
        "format": 0.99,
        "safety": 0.985,
        "cost": 1.04,
    },
}

WEIGHTS = {
    "target": 0.40,
    "general": 0.20,
    "format": 0.15,
    "safety": 0.25,
}


def validate_inputs():
    if not BASELINE or not VARIANTS:
        raise ValueError("baseline and variants must both be non-empty")
    if set(BASELINE) != set(WEIGHTS):
        raise ValueError("baseline metrics and weight metrics must match")
    if not math.isclose(sum(WEIGHTS.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("metric weights must sum to 1")
    for key, weight in WEIGHTS.items():
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"invalid weight for {key}")
    for key, value in BASELINE.items():
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"baseline score for {key} must be in [0, 1]")
    for name, result in VARIANTS.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("variant names must be non-empty strings")
        if set(result) != set(WEIGHTS) | {"cost"}:
            raise ValueError(f"variant {name} has an unexpected metric set")
        for key in WEIGHTS:
            value = result[key]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"variant {name} score for {key} must be in [0, 1]")
        cost = result["cost"]
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError(f"variant {name} cost must be finite and positive")


def analyze(name, result):
    changes = {
        key: round(result[key] - BASELINE[key], 4)
        for key in WEIGHTS
    }
    utility = sum(WEIGHTS[key] * result[key] for key in WEIGHTS)
    utility -= 0.05 * (result["cost"] - 1.0)
    return {
        "name": name,
        "changes": changes,
        "utility": round(utility, 4),
        "worst_change": min(changes.values()),
        "safety_ok": changes["safety"] >= -0.01,
        "format_ok": changes["format"] >= -0.01,
    }


validate_inputs()
report = [analyze(name, result) for name, result in VARIANTS.items()]
report.sort(key=lambda row: row["utility"], reverse=True)
print(json.dumps(report, ensure_ascii=False, indent=2))
```
如果 adapter_large 的加权效用因为目标任务权重较高而排在前面，仍要注意它的通用、格式和安全切片显著下降。代码把变化和约束分开打印，提醒我们“排序最高”不等于“适合使用”。

### 9.5.7 如何区分遗忘、任务干扰和评估协议变化

可以设计四组对照：

```text
同一基座 + 原始评估协议
同一基座 + 新评估协议
微调模型 + 原始评估协议
微调模型 + 新评估协议
```

若两种模型在新协议下都下降，优先检查模板、解析器和解码；若只有微调模型在原始协议下下降，再分析参数更新和数据；若只有某个切片下降，检查训练分布、样本混合和任务冲突；若所有切片都下降，检查 checkpoint、tokenizer 和服务加载。

再配合行为探针：

```text
同义改写是否保持答案
不同长度是否保持格式
添加无关上下文后是否改变事实
需要拒答时是否仍然拒答
工具未执行时是否声称完成
旧任务与新任务交替输入时是否互相污染
```

这些探针不是通用能力的替代品，而是帮助把“分数下降”拆成可解释行为。

### 9.5.8 资料与边界

[EWC 论文](https://arxiv.org/abs/1612.00796)给出了通过参数重要性减轻灾难性遗忘的经典路线；[LoRA 论文](https://arxiv.org/abs/2106.09685)说明了低秩更新如何降低可训练参数量，但参数少不等于没有能力退化；输出锚定、回放和多任务混合的效果依赖数据比例、任务冲突、学习率、目标模块和评估集。

因此，微调方案不能只用“目标分数最高”选择。必须把新能力、旧能力、协议、安全、成本和不同输入切片放在同一份版本化报告里。

能力变化有了分层数据，开放式回答仍需要解释质量差异。接下来读[人工评测与生成质量](09-评估与debug实战-6-人工评测.md)，学习怎样组织人工判断并校准自动裁判。
