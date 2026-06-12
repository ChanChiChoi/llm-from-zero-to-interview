# 第十章：Reasoning 评估

Reasoning model 的评估比普通问答评估更难。因为我们不只关心最终答案对不对，还关心推理过程是否可靠、模型是否真的泛化、是否只是背过 benchmark、是否能在更难变体上保持稳定，以及提升是否来自更多推理计算而不是模型本身能力变化。

本章系统讲 reasoning 评估：数学和代码 benchmark、最终答案评估、过程评估、污染检测、泛化评估、鲁棒性、推理时计算成本、人工评估和常见陷阱。

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修参考公开评估资料，包括 [HELM](https://arxiv.org/abs/2211.09110)、[BIG-bench](https://arxiv.org/abs/2206.04615)、[MMLU](https://arxiv.org/abs/2009.03300)、[GPQA](https://arxiv.org/abs/2311.12022)、GSM8K、MATH、HumanEval、LiveCodeBench、SWE-bench、OpenAI Evals 和 EleutherAI lm-evaluation-harness 等资料。它们共同提示：reasoning 评估不能只看单一 benchmark 平均分，而要同时记录任务版本、prompt、采样参数、候选数、verifier、工具、污染检查、切片表现、统计不确定性和成本。

本章不把评估写成“刷榜技巧”。第二轮精修重点是建立可复现、可审计的评估报告结构：同一批样本上比较 baseline 和 candidate；明确 original、variant、contaminated、hard slice；同时看最终答案、过程步骤、第一处错误、public / hidden test、test-time compute、预算归一化指标和 paired significance。面试中要能说明：分数提升是否来自模型能力、推理预算、候选选择器、污染、prompt 调参，还是统计噪声。

## 10.1 Reasoning 评估为什么难

普通分类任务可以直接看准确率。Reasoning 任务更复杂，因为：

1. 最终答案可能对，但过程错误。
2. 过程看起来合理，但答案错误。
3. 开放题没有唯一答案。
4. 模型可能见过测试题。
5. 多次采样结果不稳定。
6. 推理成本不同，分数不可直接比较。
7. Verifier 本身可能有偏差。

面试回答：

```text
Reasoning 评估不能只看最终准确率。需要同时评估最终答案、推理过程、泛化能力、鲁棒性、推理成本和数据污染。尤其是当模型使用多样本采样、搜索或 verifier reranking 时，必须报告计算预算，否则不同模型的分数不可公平比较。
```

### 10.1.1 关键公式与 Reasoning 评估指标速查

把第 `i` 个 reasoning 评估样本写成：

```math
e_i=(x_i,y_i^\star,V_i,P_i,z_i,q_i,g_i,r_i,b_i)
```

其中 `x_i` 是原题，`y_i^\star` 是标准答案或评估 oracle，`V_i` 是扰动 / 变体集合，`P_i` 是候选输出集合，`z_i` 是被评推理过程，`q_i` 是步骤标签，`g_i` 是题型、难度、语言、领域等切片标签，`r_i` 是污染、泄漏、格式异常等风险标记，`b_i` 是推理预算。

最终答案准确率：

```math
A_{\mathrm{ans}}=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[\hat y_i=y_i^\star]
```

切片准确率：

```math
A_g=
\frac{
\sum_{i:g_i=g}
\mathbb{1}[\hat y_i=y_i^\star]
}{
\sum_{i:g_i=g}1
}
```

切片指标用于发现平均分掩盖的问题，例如 proof、geometry、hard code、long context 或中文题表现很差。

过程步骤准确率：

```math
A_{\mathrm{step}}=
\frac{
\sum_{i,j}\mathbb{1}[\hat q_{ij}=q_{ij}]
}{
\sum_i L_i
}
```

第一处错误定位准确率：

```math
A_{\mathrm{first}}=
\frac{1}{N_{\mathrm{err}}}
\sum_{i\in \mathcal{E}}
\mathbb{1}[\hat e_i=e_i^\star]
```

扰动鲁棒性下降：

```math
D_{\mathrm{robust}}=
A_{\mathrm{orig}}-A_{\mathrm{var}}
```

如果原题正确率高、变体正确率低，常见原因是记忆原题、依赖表面模板、对无关信息敏感或没有真正理解约束。

污染率：

```math
R_{\mathrm{contam}}=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[r_i^{\mathrm{contam}}=1]
```

污染检查要覆盖题面、答案、解析、代码签名、测试样例、参数化模板和近似改写，而不是只做字符串 exact match。

test-time compute 成本：

```math
C_i=
c_{\mathrm{tok}}T_i
+c_{\mathrm{cand}}K_i
+c_{\mathrm{ver}}V_i
+c_{\mathrm{tool}}U_i
+c_{\mathrm{lat}}R_i
```

其中 `T_i` 是 token 数，`K_i` 是候选数，`V_i` 是 verifier 调用，`U_i` 是工具调用，`R_i` 是延迟。不同模型或策略比较时，必须报告预算，否则 1 次 greedy 和 64 次采样 rerank 的准确率不可直接比较。

单位成本正确率：

```math
E_{\mathrm{cost}}=
\frac{
\sum_i\mathbb{1}[\hat y_i=y_i^\star]
}{
\sum_i C_i
}
```

成对提升：

```math
\Delta_{\mathrm{pair}}=
\frac{1}{N}
\sum_{i=1}^{N}
(\mathbb{1}[\hat y_i^{new}=y_i^\star]
-
\mathbb{1}[\hat y_i^{base}=y_i^\star])
```

同一批样本上比较新旧模型时，优先使用 paired evaluation，因为它能消除样本难度差异。小样本下还要给 bootstrap confidence interval 或 McNemar 类检验，而不是只报一个提升点。

一个简化 reasoning 评估门禁：

```math
G_{\mathrm{eval}}=
\mathbb{1}[
A_{\mathrm{ans}}\ge \alpha
\land
A_{\mathrm{step}}\ge \beta
\land
A_{\mathrm{first}}\ge \gamma
\land
R_{\mathrm{contam}}\le \rho
\land
D_{\mathrm{robust}}\le \delta
\land
E_{\mathrm{cost}}\ge \eta
]
```

这个门禁表达的是评估可信度，而不是单纯模型强弱：答案、过程、错误定位、污染、鲁棒性和成本都过线，评估结论才更值得相信。

## 10.2 数学 Benchmark

数学 benchmark 常用于评估长链推理能力。

常见类型：

1. 小学和初中应用题。
2. 高中数学题。
3. 竞赛数学题。
4. 证明题。
5. 符号计算题。
6. 合成数学题。

GSM 类任务偏自然语言应用题，MATH 类任务通常更难，包含代数、数论、组合、几何等多种题型。

数学 benchmark 的优势是答案明确、便于自动评测。局限是公开题容易污染，且最终答案准确率不能完全反映推理过程质量。

## 10.3 代码 Benchmark

代码 benchmark 常评估程序生成和问题求解能力。

常见指标：

1. Pass@1。
2. Pass@k。
3. 编译通过率。
4. 单元测试通过率。
5. 隐藏测试通过率。
6. 平均修复轮数。
7. 超时率。
8. 安全违规率。

代码评估的优势是可以执行验证。缺点是测试覆盖不完整，通过测试不等于完全正确，且不同语言、依赖和运行环境会影响结果。

代码 benchmark 还要区分一次生成能力和执行反馈利用能力。一个模型 Pass@1 高，说明直接生成强；一个模型 self-debug 后提升大，说明利用反馈能力强。

## 10.4 最终答案评估

最终答案评估是最基础的层面。

数学中可以看：

1. 数值是否相等。
2. 表达式是否等价。
3. 单位是否正确。
4. 格式是否符合要求。
5. 是否满足题目约束。

代码中可以看：

1. 是否通过测试。
2. 是否超时。
3. 是否内存超限。
4. 是否符合接口。
5. 是否产生副作用。

最终答案评估简单清晰，但无法说明模型为什么对或为什么错。

## 10.5 过程评估

过程评估关注中间推理步骤。

常见维度：

1. 每一步是否正确。
2. 步骤之间是否连贯。
3. 是否存在跳步。
4. 是否遗漏条件。
5. 是否第一处错误很早出现。
6. 最终答案是否由过程支持。

过程评估可以由人工、LLM judge、process verifier 或规则系统完成。它比最终答案评估更细，但成本更高，也更容易受到评估器偏差影响。

一个模型如果答案正确但过程胡编，在高风险场景中不能被视为可靠。

## 10.6 First Error Evaluation

First error evaluation 关注推理链中第一处错误在哪里。

它比“整条链对不对”更有诊断价值。

例如：

```text
step 1: correct
step 2: correct
step 3: wrong
step 4: follows from wrong step
```

如果只看最终答案，模型是错的；如果看第一处错误，可以知道问题出在 step 3。这对训练 process verifier、构造纠错数据和分析模型短板很有帮助。

## 10.7 污染检测

数据污染是 reasoning 评估的大问题。模型可能在预训练或微调中见过测试题、答案或解析。

污染来源：

1. 公开 benchmark。
2. 题解网站。
3. 教材和论坛。
4. 代码仓库。
5. 合成模板重复。
6. 训练集和测试集去重不充分。

检测方式：

1. 字符串和语义去重。
2. 检查题目近似重复。
3. 使用新构造测试集。
4. 参数化改写题目。
5. 比较原题和变体题表现。
6. 分析模型是否输出标准题解措辞。

污染不能完全消除，但必须被报告和控制。

## 10.8 泛化评估

真正的 reasoning 能力应该能泛化到新题、变体题和更难题。

泛化评估可以包括：

1. 题面改写。
2. 数值替换。
3. 条件扰动。
4. 跨题型组合。
5. 更长推理链。
6. 未见过的工具环境。
7. 新 benchmark。

如果模型只在原题上高分，在轻微改写后大幅下降，说明可能依赖记忆或模板匹配。

## 10.9 鲁棒性评估

Reasoning 模型还要评估鲁棒性。

常见扰动：

1. 无关信息干扰。
2. 题目表述顺序变化。
3. 多余条件。
4. 缺失条件。
5. 对抗性提示。
6. 单位变化。
7. 格式变化。
8. 长上下文干扰。

鲁棒模型应该能识别哪些信息重要，哪些信息无关，而不是被表面文本带偏。

## 10.10 推理成本评估

Reasoning 评估必须报告推理成本。

需要记录：

1. 采样数量。
2. 输出 token 数。
3. 搜索深度。
4. verifier 调用次数。
5. 工具调用次数。
6. 平均延迟。
7. GPU 成本。
8. 超时率。

两个模型如果一个是单次采样，一个是 64 次采样加 reranking，直接比较准确率是不公平的。应同时报告 quality-cost curve，也就是不同预算下的质量曲线。

## 10.11 多次采样稳定性

Reasoning 模型输出常有随机性。因此需要评估稳定性。

常见方法：

1. 多次运行同一题。
2. 计算答案一致率。
3. 计算正确率方差。
4. 比较不同温度下表现。
5. 分析错误答案分布。

稳定性很重要。一个模型平均分高但波动巨大，在实际系统中可能不如一个平均分稍低但稳定的模型。

## 10.12 LLM Judge 的使用

LLM judge 可以评估开放式 reasoning，但要小心。

优点：

1. 覆盖开放题。
2. 成本低于大量人工评审。
3. 能给解释。
4. 可用于过程评分。

风险：

1. 偏好长答案。
2. 偏好格式漂亮的答案。
3. 被错误但流畅的推理欺骗。
4. 与被评模型同源导致偏差。
5. 评判标准不稳定。

使用 LLM judge 时，最好结合标准答案、rubric、多评委一致性和人工抽检。

## 10.13 人工评估

人工评估适合高难度、开放式和过程质量评估。

人工评估要注意：

1. 明确评分 rubric。
2. 标注者需要领域能力。
3. 进行多标注者一致性检查。
4. 区分答案正确和过程正确。
5. 抽样覆盖不同难度和题型。
6. 记录不确定案例。

人工评估成本高，但对发现 benchmark 盲区非常有价值。

## 10.14 常见评估陷阱

1. 只报最高分，不报预算。
2. 只看最终答案，不看过程。
3. 公开 benchmark 污染严重。
4. 使用不可靠 judge。
5. 题型覆盖太窄。
6. 忽略失败案例分析。
7. 忽略延迟和成本。
8. 把格式更长误认为推理更强。
9. 忽略多次运行波动。

Reasoning 评估的目标不是得到一个漂亮分数，而是理解模型在哪些任务上真的会推理，在哪些任务上只是看起来会推理。

## 10.15 一个完整评估报告应该包含什么

一个合理的 reasoning 评估报告至少包含：

1. Benchmark 名称和版本。
2. 数据去重和污染检查说明。
3. Prompt 和解码参数。
4. 是否使用 CoT、self-consistency、search、verifier 或工具。
5. 推理预算。
6. 最终答案指标。
7. 过程质量指标。
8. 分题型和分难度结果。
9. 成本和延迟。
10. 失败案例分析。

如果没有这些信息，评估结果很难复现，也很难判断改进来自哪里。

## 10.16 最小可运行 reasoning 评估审计 demo

这个 demo 模拟 6 个 toy reasoning 样本，对比 baseline 和 candidate。它同时记录原题、变体、过程步骤、第一处错误、污染、切片、成本和 paired bootstrap 区间。它故意让 `gate_pass=False`，因为候选模型虽然平均准确率更高，但统计区间不稳、过程质量不足、存在污染样本、变体鲁棒性下降明显。

```python
from random import Random


SAMPLES = [
    {
        "id": "math_easy",
        "slice": "math",
        "contaminated": False,
        "baseline_correct": True,
        "candidate_correct": True,
        "variant_correct": True,
        "steps": [True, True, True],
        "first_error_gold": None,
        "first_error_pred": None,
        "cost": 120,
    },
    {
        "id": "math_contam",
        "slice": "math",
        "contaminated": True,
        "baseline_correct": False,
        "candidate_correct": True,
        "variant_correct": False,
        "steps": [True, True],
        "first_error_gold": None,
        "first_error_pred": None,
        "cost": 160,
    },
    {
        "id": "logic_distractor",
        "slice": "logic",
        "contaminated": False,
        "baseline_correct": False,
        "candidate_correct": True,
        "variant_correct": False,
        "steps": [True, False, True],
        "first_error_gold": 1,
        "first_error_pred": 1,
        "cost": 260,
    },
    {
        "id": "code_hidden",
        "slice": "code",
        "contaminated": False,
        "baseline_correct": False,
        "candidate_correct": True,
        "variant_correct": True,
        "steps": [True, True, True, True],
        "first_error_gold": None,
        "first_error_pred": None,
        "cost": 260,
    },
    {
        "id": "proof_hard",
        "slice": "proof",
        "contaminated": False,
        "baseline_correct": False,
        "candidate_correct": False,
        "variant_correct": False,
        "steps": [True, False, False, False],
        "first_error_gold": 1,
        "first_error_pred": 2,
        "cost": 420,
    },
    {
        "id": "simple_regression",
        "slice": "simple",
        "contaminated": False,
        "baseline_correct": True,
        "candidate_correct": False,
        "variant_correct": False,
        "steps": [True, False],
        "first_error_gold": 1,
        "first_error_pred": 1,
        "cost": 170,
    },
]


def ratio(numerator, denominator):
    return round(numerator / denominator, 3) if denominator else 0.0


def accuracy(key):
    return ratio(sum(sample[key] for sample in SAMPLES), len(SAMPLES))


def paired_bootstrap_ci(diffs, rounds=2000, seed=7):
    rng = Random(seed)
    n = len(diffs)
    values = []
    for _ in range(rounds):
        values.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    values.sort()
    return (
        round(values[int(0.025 * rounds)], 3),
        round(values[int(0.975 * rounds)], 3),
    )


baseline_accuracy = accuracy("baseline_correct")
candidate_accuracy = accuracy("candidate_correct")
variant_accuracy = accuracy("variant_correct")
diffs = [
    int(sample["candidate_correct"]) - int(sample["baseline_correct"])
    for sample in SAMPLES
]
paired_lift = round(sum(diffs) / len(diffs), 3)
bootstrap_ci = paired_bootstrap_ci(diffs)

all_steps = [step for sample in SAMPLES for step in sample["steps"]]
error_cases = [
    sample for sample in SAMPLES
    if sample["first_error_gold"] is not None
]
process_step_accuracy = ratio(sum(all_steps), len(all_steps))
first_error_accuracy = ratio(
    sum(sample["first_error_gold"] == sample["first_error_pred"] for sample in error_cases),
    len(error_cases),
)
contaminated = [sample["id"] for sample in SAMPLES if sample["contaminated"]]
robustness_drop = round(candidate_accuracy - variant_accuracy, 3)

slice_accuracy = {}
for slice_name in sorted({sample["slice"] for sample in SAMPLES}):
    group = [sample for sample in SAMPLES if sample["slice"] == slice_name]
    slice_accuracy[slice_name] = ratio(
        sum(sample["candidate_correct"] for sample in group),
        len(group),
    )

total_cost = sum(sample["cost"] for sample in SAMPLES)
candidate_correct = sum(sample["candidate_correct"] for sample in SAMPLES)
summary = {
    "baseline_accuracy": baseline_accuracy,
    "candidate_accuracy": candidate_accuracy,
    "paired_lift": paired_lift,
    "bootstrap_ci": bootstrap_ci,
    "variant_accuracy": variant_accuracy,
    "robustness_drop": robustness_drop,
    "process_step_accuracy": process_step_accuracy,
    "first_error_accuracy": first_error_accuracy,
    "contamination_rate": ratio(len(contaminated), len(SAMPLES)),
    "cost_per_correct": round(total_cost / max(1, candidate_correct), 3),
    "p95_cost": sorted(sample["cost"] for sample in SAMPLES)[-1],
}
gates = {
    "accuracy_ok": candidate_accuracy >= 0.65,
    "lift_ci_nonnegative": bootstrap_ci[0] >= 0.0,
    "process_ok": process_step_accuracy >= 0.8,
    "first_error_ok": first_error_accuracy >= 0.8,
    "contamination_ok": len(contaminated) == 0,
    "robustness_ok": robustness_drop <= 0.2,
    "cost_ok": summary["cost_per_correct"] <= 400,
}

print(f"summary={summary}")
print(f"slice_accuracy={slice_accuracy}")
print(f"contaminated={contaminated}")
print(f"regressions={[sample['id'] for sample in SAMPLES if sample['baseline_correct'] and not sample['candidate_correct']]}")
print(f"variant_failures={[sample['id'] for sample in SAMPLES if sample['candidate_correct'] and not sample['variant_correct']]}")
print(f"gates={gates}")
print(f"gate_pass={all(gates.values())}")
```

预期输出：

```text
summary={'baseline_accuracy': 0.333, 'candidate_accuracy': 0.667, 'paired_lift': 0.333, 'bootstrap_ci': (-0.333, 0.833), 'variant_accuracy': 0.333, 'robustness_drop': 0.334, 'process_step_accuracy': 0.722, 'first_error_accuracy': 0.667, 'contamination_rate': 0.167, 'cost_per_correct': 347.5, 'p95_cost': 420}
slice_accuracy={'code': 1.0, 'logic': 1.0, 'math': 1.0, 'proof': 0.0, 'simple': 0.0}
contaminated=['math_contam']
regressions=['simple_regression']
variant_failures=['math_contam', 'logic_distractor']
gates={'accuracy_ok': True, 'lift_ci_nonnegative': False, 'process_ok': False, 'first_error_ok': False, 'contamination_ok': False, 'robustness_ok': False, 'cost_ok': True}
gate_pass=False
```

这个结果的关键解释是：candidate 的平均准确率从 `0.333` 到 `0.667`，但 bootstrap interval 跨过 0，说明样本太少时不能把提升说成稳健结论；`math_contam` 说明污染样本会抬高分数；`variant_failures` 说明原题正确不代表变体鲁棒；`simple_regression` 说明 reasoning 策略可能伤害简单题；`proof` 切片为 0 说明平均分掩盖 hard slice。真正的评估报告必须把这些证据一起展示。

## 10.17 面试题：如何评估一个 Reasoning Model

回答要点：

```text
我会从最终答案、过程质量、泛化能力、鲁棒性和推理成本几个维度评估。数学任务可以看 GSM、MATH 类 benchmark 和变体题；代码任务可以看 Pass@1、Pass@k、隐藏测试通过率和 self-debug 后提升。还要做数据污染检测，报告采样数、token、verifier 和工具调用预算。对于过程推理，要看步骤正确率和第一处错误位置，而不只看最终准确率。
```

## 10.18 面试题：为什么不能只看 Benchmark 分数

回答要点：

```text
Benchmark 分数可能受数据污染、prompt 设置、推理预算、采样次数和评估器偏差影响。一个模型分数高，可能是因为见过题、用了更多 test-time compute，或者 verifier 选择了更好的候选。Reasoning 模型还需要看过程是否正确、变体题是否泛化、成本是否可接受，以及失败案例是什么。
```

## 10.19 小练习

1. 给一个 reasoning 评估集设计 original、variant、contaminated、hard slice 四类样本标记。
2. 修改本章 demo，让 `math_contam` 不计入最终分数，观察 accuracy、bootstrap interval 和 gate 如何变化。
3. 设计一个过程评估 rubric，分别评估步骤正确、条件覆盖、第一处错误和最终答案支持度。
4. 解释为什么 pass@k、verifier accuracy、final accuracy 和 cost per correct 必须分开报告。
5. 用 3 句话说明 paired evaluation 为什么比两个模型分别报平均分更适合比较小幅提升。

## 10.20 本章小结

Reasoning 评估要从“最终答案对不对”扩展到“过程是否可靠、是否泛化、是否鲁棒、是否公平比较、成本是否可接受”。数学和代码 benchmark 很重要，但不能成为唯一依据。污染检测、过程评估、变体题、鲁棒性测试、推理预算报告和失败案例分析同样关键。

下一章会讨论 reasoning model 的安全与局限，包括幻觉、过度自信、伪推理、reward hacking、工具误用和高风险场景中的可靠性边界。
