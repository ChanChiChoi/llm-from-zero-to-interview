# 第二章：Chain-of-Thought

## 0. 本讲资料边界与第二轮精修口径

本章第二轮精修时，重点参考 Chain-of-Thought Prompting、Zero-shot CoT、scratchpad intermediate computation、CoT faithfulness / unfaithfulness，以及公开 reasoning model 对隐藏推理、可见解释和 test-time compute 的讨论。这里把 CoT 当作 reasoning 系统中的一种推理时计算和中间状态表达方法，而不是把“输出更长解释”直接等同于“模型真实推理能力更强”。

本章聚焦：

1. CoT 为什么可能提升数学、逻辑、代码和多步任务表现。
2. few-shot CoT、zero-shot CoT、scratchpad、隐藏推理和可见解释的边界。
3. 如何用最终答案准确率、步骤准确率、答案一致性、成本和回归样本审计 CoT。
4. 为什么长 CoT 不一定更好，以及工程上为什么要做 CoT 路由。
5. 面试中如何回答“CoT 是否等于真实推理过程”这个高频追问。

本章不展开 self-consistency、多候选搜索、process reward model、Tree-of-Thought 或 test-time compute scaling 的完整算法细节，这些会在后续章节单独讲。本章也不提供诱导模型泄露隐藏推理、系统策略或安全规避路径的技巧；产品视角只讨论如何给用户展示简洁、可验证、必要的解释。

Chain-of-Thought，简称 CoT，是 reasoning model 中最基础也最容易被误解的概念。它的表面形式很简单：让模型在给出最终答案之前，先写出中间推理步骤。但它背后的问题很深：CoT 到底是在帮助模型“真正推理”，还是只是在生成看起来像推理的解释？为什么 CoT 对数学题、逻辑题有效，对某些简单任务反而没必要？长 CoT 是否一定更好？训练时应该监督完整 CoT 吗？推理时应该向用户展示 CoT 吗？

本章目标是讲清 CoT 的作用、使用方式、机制争议、有效场景、训练数据、风险和面试表达。

## 2.1 CoT 是什么

CoT 指模型在回答前生成一段中间推理过程。

普通回答：

```text
答案是 10。
```

CoT 回答：

```text
每分钟进水 3 升，漏水 1 升，所以净增加 2 升。
水箱容量是 20 升。
20 / 2 = 10。
答案是 10 分钟。
```

CoT 的核心作用是把复杂问题拆成多个小步骤，让模型有更多 token 空间进行中间计算和状态保存。

面试回答：

```text
Chain-of-Thought 是让模型在最终答案前生成中间推理步骤的方法。它可以把复杂问题拆解成多个子步骤，给模型更多计算和中间状态表达空间，因此常能提升数学、逻辑和多步推理任务表现。但 CoT 不保证每一步真实可靠，需要配合验证和评估。
```

### 2.1.1 关键公式与 CoT 审计指标速查

把一个 CoT 样本写成：

```math
c_i=(x_i,z_i,\hat y_i,y_i^\star,m_i,b_i)
```

其中 `x_i` 是问题，`z_i` 是中间推理链，`hat y_i` 是模型最终答案，`y_i^star` 是标准答案，`m_i` 是任务元信息，`b_i` 是预算或安全边界。

带 CoT 的自回归生成可以粗略写成：

```math
P(z_i,\hat y_i\mid x_i)
=
\left(
\prod_{t=1}^{L_i}P(z_{i,t}\mid x_i,z_{i,1:t-1})
\right)
\left(
\prod_{u=1}^{A_i}P(\hat y_{i,u}\mid x_i,z_i,\hat y_{i,1:u-1})
\right)
```

其中 `L_i` 是推理链长度，`A_i` 是答案长度。这个式子强调：CoT 不是免费能力，推理链越长，推理时 token 成本越高。

直答准确率：

```math
A_{\mathrm{direct}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[\hat y_i^{\mathrm{direct}}=y_i^\star]
```

CoT 准确率：

```math
A_{\mathrm{cot}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[\hat y_i^{\mathrm{cot}}=y_i^\star]
```

步骤准确率：

```math
A_{\mathrm{step}}
=
\frac{\sum_i\sum_{j=1}^{M_i}q_{ij}}
{\sum_i M_i}
```

其中 `q_ij=1` 表示第 `i` 个样本第 `j` 个步骤被人工、规则、工具或 verifier 判定为正确。

最终答案与推理链一致率：

```math
C_{\mathrm{final}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[g(z_i)=\hat y_i]
```

其中 `g(z_i)` 表示从推理链中抽取出的结论。这个指标只能说明“步骤和答案是否一致”，不能证明答案正确。

CoT 回归率：

```math
R_{\mathrm{reg}}
=
\frac{
\sum_i
\mathbb{1}[
\hat y_i^{\mathrm{direct}}=y_i^\star
\land
\hat y_i^{\mathrm{cot}}\ne y_i^\star
]
}{N}
```

它衡量“直答本来正确，但 CoT 反而带错”的比例。这个指标在简单事实问答、分类题和低延迟产品里很重要。

CoT token 成本：

```math
C_{\mathrm{cot}}
=
\sum_{i=1}^{N}(T_i^{\mathrm{prompt}}+T_i^{\mathrm{cot}}+T_i^{\mathrm{answer}})
```

一个简化 CoT 路由门禁：

```math
G_{\mathrm{cot}}
=
\mathbb{1}[
A_{\mathrm{route}}\ge A_{\mathrm{cot}}
\land
A_{\mathrm{step}}\ge \alpha
\land
R_{\mathrm{reg}}\le \rho
\land
C_{\mathrm{route}}\le C_{\max}
\land
S_{\mathrm{leak}}\le \epsilon
]
```

其中 `A_route` 是按任务难度选择直答、CoT 或工具后的准确率，`S_leak` 是可见解释泄露隐藏策略或安全细节的风险分。面试中要强调：CoT 是否上线，不只看 `A_cot`，还要看回归、步骤质量、成本和安全。

## 2.2 为什么 CoT 可能有效

CoT 可能有效有几个直觉原因。

### 2.2.1 增加中间计算空间

如果模型直接输出答案，它必须在很短上下文中完成所有隐式计算。CoT 允许模型把中间变量写出来。

例如：

```text
净速度 = 3 - 1 = 2
时间 = 20 / 2 = 10
```

这类似人在草稿纸上解题。

### 2.2.2 分解复杂问题

多步问题可以拆成子问题。

例如：

1. 读题。
2. 提取条件。
3. 选择公式。
4. 计算。
5. 检查答案。

CoT 让模型显式走这些步骤。

### 2.2.3 匹配训练语料格式

互联网上有大量“逐步解答”的文本。模型学过这种格式后，在 prompt 中要求 step-by-step，可能激活类似解题模式。

### 2.2.4 便于后续验证

如果只有最终答案，verifier 很难判断错在哪里。CoT 提供中间步骤，便于 process verifier 或人类检查。

## 2.3 Few-shot CoT

Few-shot CoT 是在 prompt 中给几个带推理过程的示例，让模型模仿。

示例：

```text
Q: 小明有 3 个苹果，又买了 2 个，一共有几个？
A: 小明原来有 3 个，又买了 2 个，所以 3 + 2 = 5。答案是 5。

Q: 一本书 10 元，买 3 本多少钱？
A:
```

模型看到示例后，更可能生成步骤：

```text
每本 10 元，买 3 本，所以 10 * 3 = 30。答案是 30 元。
```

Few-shot CoT 的优点：

1. 不需要训练。
2. 对大模型效果明显。
3. 可以通过示例控制输出风格。

缺点：

1. 占用上下文。
2. 示例质量影响大。
3. 对小模型可能不稳定。

## 2.4 Zero-shot CoT

Zero-shot CoT 不给示例，只通过一句提示诱导模型逐步思考。

经典形式：

```text
Let's think step by step.
```

中文可以是：

```text
请一步一步思考。
```

这类提示能让模型生成更长推理过程。

但要注意：

1. 它可能提升复杂题表现。
2. 对简单题可能增加冗余。
3. 对不擅长的题可能生成更长但错误的解释。
4. 不应把“更长回答”当作“更强推理”。

## 2.5 CoT 和 Scratchpad

Scratchpad 可以理解为模型的草稿纸。它不一定是给用户看的最终解释，而是模型内部或中间生成的工作区。

区别：

1. CoT 常指自然语言推理链。
2. Scratchpad 更强调中间工作区，可以包含计算、代码、表格、草稿。
3. 最终回答可以只展示结论和简洁理由。

在产品中，很多系统不会直接展示完整 CoT，而是展示简洁解释。这是因为完整 CoT 可能冗长、错误、泄露策略或让用户误以为每一步都可靠。

## 2.6 CoT 是否等于真实推理

这是一个重要争议。

支持 CoT 有效的观察：

1. CoT 提升很多数学和逻辑 benchmark。
2. 中间步骤可以被检查。
3. 错误常能在步骤中定位。

质疑点：

1. CoT 可能是事后合理化。
2. 模型可能先决定答案，再编解释。
3. 有些 CoT 步骤看似合理但不因果影响答案。
4. CoT 可能掩盖错误，让答案更有迷惑性。

更稳妥的看法是：CoT 是一种有用的推理外显和计算扩展方式，但不能直接等同于模型真实内部机制。需要通过干预实验、过程验证和工具校验评估。

面试回答：

```text
CoT 不一定等于模型真实内部推理。它可能帮助模型分解问题，也可能只是生成看起来合理的解释。判断 CoT 是否可靠，需要看最终答案、步骤正确性、对步骤干预是否影响结论，以及是否能通过 verifier 或工具验证。
```

## 2.7 CoT 何时有效

CoT 常对这些任务有效：

1. 多步数学题。
2. 逻辑推理。
3. 多跳问答。
4. 规划任务。
5. 需要中间变量的代码题。
6. 复杂文本分析。

共同特点：问题不能靠单步模式匹配解决，需要中间状态。

CoT 不一定适合：

1. 简单事实问答。
2. 分类任务。
3. 低延迟场景。
4. 用户只需要短答案的场景。
5. 高安全风险场景中公开完整推理。

工程上可以动态决定是否启用长推理，而不是所有问题都强制 CoT。

## 2.8 长 CoT 是否一定更好

不一定。

长 CoT 的好处：

1. 有更多推理空间。
2. 可以分解复杂问题。
3. 便于检查。

长 CoT 的风险：

1. 成本高。
2. 延迟高。
3. 更容易跑偏。
4. 可能出现冗余和自相矛盾。
5. 可能生成更有说服力的错误解释。

合理做法是根据任务难度分配推理预算。简单题短答，复杂题长思考。

## 2.9 CoT 数据如何构造

CoT 训练数据通常包含：

1. 问题。
2. 推理步骤。
3. 最终答案。

例如：

```json
{
  "question": "小明有 3 个苹果，又买了 2 个，一共有几个？",
  "reasoning": "小明原来有 3 个，又买了 2 个，所以 3 + 2 = 5。",
  "answer": "5"
}
```

高质量 CoT 数据要求：

1. 步骤正确。
2. 不跳步。
3. 不自相矛盾。
4. 最终答案和步骤一致。
5. 格式稳定。
6. 难度分布合理。

低质量 CoT 数据会训练模型生成错误但自信的推理链。

## 2.10 CoT 和最终答案格式

Reasoning 任务常需要明确分离推理和最终答案。

例如：

```text
推理：...
答案：10
```

这样有几个好处：

1. 方便自动抽取答案。
2. 方便 verifier 检查。
3. 方便评估。
4. 减少格式混乱。

数学评估中，答案抽取非常重要。如果模型推理正确但答案格式不规范，自动评估可能判错。

## 2.11 Hidden CoT 和可见解释

产品中常区分内部推理和外部解释。

内部推理：模型或系统用于解决问题的草稿过程。

外部解释：给用户看的简洁、可靠、必要的说明。

为什么不总展示完整 CoT？

1. 完整 CoT 可能很长。
2. 可能包含错误中间想法。
3. 可能泄露系统策略。
4. 用户更需要可验证结论。
5. 安全场景下可能暴露规避路径。

因此很多系统会输出简洁解释，而不是完整内部推理。

## 2.12 CoT 与 Verifier 的关系

CoT 生成候选推理路径，verifier 负责评估。

组合方式：

1. 生成多条 CoT。
2. 抽取最终答案。
3. 用 verifier 或规则打分。
4. 选择最可信答案。

Verifier 可以看：

1. 最终答案是否正确。
2. 中间步骤是否合理。
3. 是否有计算错误。
4. 是否使用了题目中不存在的条件。

CoT 没有 verifier 时，容易生成“有步骤但错”的答案。

## 2.13 CoT 与工具使用

有些推理不应该只靠自然语言。

例如：

1. 大数计算。
2. 代码执行。
3. 符号运算。
4. 数据查询。
5. 单元测试。

这时更好的做法是：CoT 负责规划，工具负责精确执行。

例子：

```text
先把题目转成 Python 表达式，再运行得到结果，最后解释。
```

工具使用可以显著降低计算错误，但也带来工具选择、参数、安全和执行失败问题。

## 2.14 常见 CoT 失败模式

1. 步骤看似合理但计算错。
2. 中间步骤和最终答案不一致。
3. 使用题目中没有给出的条件。
4. 过度解释简单问题。
5. 推理链很长但偏离题目。
6. 模型先猜答案再编理由。
7. 格式混乱导致答案抽取失败。
8. 在安全问题上给出规避细节。

这些失败说明 CoT 必须配合评估、格式约束和 verifier。

## 2.15 最小可运行 CoT 质量 / 成本审计 demo

下面这个 demo 不调用任何模型，而是用 toy case 模拟三种策略：

1. `direct`：直接给答案。
2. `cot`：生成推理链后给答案。
3. `routed`：只在复杂数学题和代码题上启用 CoT，简单题和安全边界题保持短答或拒答。

它展示一个面试中很重要的结论：CoT 整体可能比直答更好，但盲目 CoT 仍会在简单事实题上回归，也可能在干扰条件题中生成看似合理但错误的步骤。

```python
from collections import Counter

cases = [
    {
        "id": "water_tank",
        "route": "hard_math",
        "gold": "10",
        "direct": "12",
        "cot": "10",
        "direct_tokens": 5,
        "cot_tokens": 55,
        "steps": [True, True, True],
        "unsupported": False,
        "visible_cot_allowed": True,
    },
    {
        "id": "apple_simple",
        "route": "simple",
        "gold": "5",
        "direct": "5",
        "cot": "5",
        "direct_tokens": 4,
        "cot_tokens": 30,
        "steps": [True, True],
        "unsupported": False,
        "visible_cot_allowed": True,
    },
    {
        "id": "distractor_math",
        "route": "hard_math",
        "gold": "12",
        "direct": "99",
        "cot": "111",
        "direct_tokens": 5,
        "cot_tokens": 45,
        "steps": [False, False],
        "unsupported": True,
        "visible_cot_allowed": True,
    },
    {
        "id": "code_loop",
        "route": "code",
        "gold": "pass",
        "direct": "fail",
        "cot": "pass",
        "direct_tokens": 6,
        "cot_tokens": 60,
        "steps": [True, True, True],
        "unsupported": False,
        "visible_cot_allowed": True,
    },
    {
        "id": "capital_lookup",
        "route": "simple",
        "gold": "tokyo",
        "direct": "tokyo",
        "cot": "kyoto",
        "direct_tokens": 4,
        "cot_tokens": 48,
        "steps": [False, False],
        "unsupported": False,
        "visible_cot_allowed": True,
    },
    {
        "id": "safety_boundary",
        "route": "safety",
        "gold": "refuse",
        "direct": "refuse",
        "cot": "refuse",
        "direct_tokens": 8,
        "cot_tokens": 24,
        "steps": [True],
        "unsupported": False,
        "visible_cot_allowed": False,
    },
]


def accuracy(items, key):
    return sum(case[key] == case["gold"] for case in items) / len(items)


def route_answer(case):
    if case["route"] in {"hard_math", "code"}:
        return case["cot"], case["cot_tokens"]
    return case["direct"], case["direct_tokens"]


direct_acc = accuracy(cases, "direct")
cot_acc = accuracy(cases, "cot")
routed_correct = 0
routed_tokens = 0
route_counts = Counter()

for case in cases:
    answer, tokens = route_answer(case)
    routed_correct += answer == case["gold"]
    routed_tokens += tokens
    route_counts[case["route"]] += 1

all_steps = [ok for case in cases for ok in case["steps"]]
step_accuracy = sum(all_steps) / len(all_steps)
regression_ids = [
    case["id"]
    for case in cases
    if case["direct"] == case["gold"] and case["cot"] != case["gold"]
]
unsupported_ids = [case["id"] for case in cases if case["unsupported"]]
visible_cot_blocked = [
    case["id"] for case in cases if not case["visible_cot_allowed"]
]
simple_cases = [case for case in cases if case["route"] == "simple"]
simple_waste_or_regression = sum(
    case["direct"] == case["gold"]
    and case["cot_tokens"] > case["direct_tokens"]
    for case in simple_cases
) / len(simple_cases)

report = {
    "direct_accuracy": round(direct_acc, 3),
    "cot_accuracy": round(cot_acc, 3),
    "routed_accuracy": round(routed_correct / len(cases), 3),
    "step_accuracy": round(step_accuracy, 3),
    "avg_cot_tokens": round(
        sum(case["cot_tokens"] for case in cases) / len(cases), 3
    ),
    "cost_per_cot_correct": round(
        sum(case["cot_tokens"] for case in cases)
        / sum(case["cot"] == case["gold"] for case in cases),
        3,
    ),
    "cost_per_routed_correct": round(routed_tokens / routed_correct, 3),
    "simple_waste_or_regression": round(simple_waste_or_regression, 3),
    "route_counts": dict(route_counts),
    "cot_regression_ids": regression_ids,
    "unsupported_step_ids": unsupported_ids,
    "visible_cot_blocked": visible_cot_blocked,
}

gates = {
    "cot_beats_direct": report["cot_accuracy"] > report["direct_accuracy"],
    "routing_beats_cot": report["routed_accuracy"] > report["cot_accuracy"],
    "step_quality_ok": report["step_accuracy"] >= 0.65,
    "no_cot_regression": len(regression_ids) == 0,
    "no_unsupported_steps": len(unsupported_ids) == 0,
    "budget_ok": report["cost_per_routed_correct"] <= 40,
}
report["gates"] = gates
report["gate_pass"] = all(gates.values())

for key, value in report.items():
    print(f"{key}={value}")
```

期望输出：

```text
direct_accuracy=0.5
cot_accuracy=0.667
routed_accuracy=0.833
step_accuracy=0.692
avg_cot_tokens=43.667
cost_per_cot_correct=65.5
cost_per_routed_correct=35.2
simple_waste_or_regression=1.0
route_counts={'hard_math': 2, 'simple': 2, 'code': 1, 'safety': 1}
cot_regression_ids=['capital_lookup']
unsupported_step_ids=['distractor_math']
visible_cot_blocked=['safety_boundary']
gates={'cot_beats_direct': True, 'routing_beats_cot': True, 'step_quality_ok': True, 'no_cot_regression': False, 'no_unsupported_steps': False, 'budget_ok': True}
gate_pass=False
```

这段 demo 的面试表达是：

1. `cot_accuracy` 高于 `direct_accuracy`，说明 CoT 对复杂任务有价值。
2. `routed_accuracy` 又高于 `cot_accuracy`，说明工程上不应所有题都强制长 CoT。
3. `capital_lookup` 是 CoT 回归样本，说明简单事实题可能被过度推理带偏。
4. `distractor_math` 是 unsupported step 样本，说明 CoT 会把题目外的干扰条件写进推理链。
5. `safety_boundary` 不允许展示可见 CoT，说明产品中要区分内部推理和用户可见解释。
6. `gate_pass=False` 不是代码失败，而是在提醒：上线前还需要 regression guard、unsupported-step verifier 和安全展示策略。

## 2.16 面试官会怎么问

### 问题一：CoT 为什么能提升推理能力？

回答模板：

```text
CoT 给模型更多中间计算空间，把复杂问题分解成步骤，并匹配训练语料中逐步解题的模式。它尤其适合数学、逻辑和多步任务，也便于后续 verifier 检查中间过程。但它不保证每一步都真实可靠。
```

### 问题二：Zero-shot CoT 和 few-shot CoT 有什么区别？

回答模板：

```text
Few-shot CoT 在 prompt 中给几个带推理过程的示例，让模型模仿；zero-shot CoT 不给示例，只用类似“请一步一步思考”的提示诱导模型生成推理。Few-shot 更可控但占上下文，zero-shot 更简单但稳定性依赖模型能力。
```

### 问题三：CoT 是否代表模型真实推理过程？

回答模板：

```text
不一定。CoT 可能帮助模型推理，也可能是事后合理化或看起来合理的解释。判断它是否可靠，需要看最终答案、步骤正确性、对中间步骤干预是否影响结论，以及是否能被 verifier 或工具验证。
```

### 问题四：什么时候不应该使用长 CoT？

回答模板：

```text
简单事实问答、低延迟场景、用户只需要短答案的任务，或者安全敏感场景中，不一定适合展示长 CoT。长 CoT 会增加成本和延迟，也可能生成更有说服力但错误的解释。
```

### 问题五：如何构造高质量 CoT 数据？

回答模板：

```text
高质量 CoT 数据要保证步骤正确、逻辑连贯、不跳步、不自相矛盾，最终答案和推理一致，并且格式稳定、难度分布合理。低质量 CoT 会训练模型生成错误但自信的推理链。
```

## 2.17 小练习

1. 给一道小学应用题写普通答案和 CoT 答案。
2. 比较 zero-shot CoT 和 few-shot CoT 的 prompt。
3. 构造一个 CoT 看似合理但计算错误的样本。
4. 设计一个答案抽取格式，例如 `答案：...`。
5. 写一个 CoT 数据质量检查表。
6. 说明为什么产品中不一定展示完整 CoT。
7. 设计一个 CoT + Python 工具校验流程。
8. 用上面的 demo 增加一个简单题 CoT 回归样本，并观察 `routed_accuracy` 是否继续优于 `cot_accuracy`。

## 2.18 本章总结

CoT 是 reasoning model 的基础工具。它通过显式中间步骤给模型更多计算空间，帮助复杂问题分解，也便于 verifier 和人类检查。但 CoT 不是推理正确性的保证，更不一定等于模型真实内部机制。

需要记住：

1. CoT 常对数学、逻辑、多步任务有效。
2. Few-shot CoT 用示例诱导，zero-shot CoT 用指令诱导。
3. 长 CoT 不一定更好，要看任务难度和成本。
4. CoT 可能是伪解释，需要 verifier 或工具验证。
5. 高质量 CoT 数据很重要，错误 CoT 会污染模型。
6. 产品中通常更适合展示简洁解释，而不是完整内部推理。

下一章会进入 self-consistency 与采样，讲清为什么多采样、多路径投票和答案标准化能提升推理可靠性。
