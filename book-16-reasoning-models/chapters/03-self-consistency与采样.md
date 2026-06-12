# 第三章：Self-Consistency 与采样

## 0. 本讲资料边界与第二轮精修口径

本章第二轮精修时，重点参考 Self-Consistency Improves Chain of Thought Reasoning in Language Models、Chain-of-Thought Prompting、nucleus sampling / top-p 采样，以及 HumanEval 中 pass@k 的评估口径。这里把 self-consistency 当作最小可落地的 test-time compute 方法：同一道题生成多个候选推理链，抽取并标准化最终答案，再通过多数投票、加权投票或 verifier 选择答案。

本章聚焦：

1. 为什么 greedy decoding 无法提供多路径多样性。
2. temperature、top-k、top-p 如何影响候选分布。
3. 答案抽取、答案标准化和等价类合并为什么是工程关键。
4. majority vote、weighted vote、pass@k 和 verifier rerank 的区别。
5. 如何用准确率、候选多样性、pass@k、成本和失败样本审计 self-consistency。

本章不展开 verifier / reward model 的训练细节，不展开 Tree-of-Thought / MCTS 搜索，也不把“采样更多次”写成无条件更强。self-consistency 只有在模型能生成一定比例正确候选、答案能标准化、采样多样性适中且成本可接受时才值得上线。

Self-consistency 是 reasoning 中最简单也最实用的 test-time compute 方法之一。它的核心思想是：同一个问题不要只生成一条推理链，而是采样多条不同推理路径，再从多个答案中聚合出更可靠的最终答案。对数学、逻辑、代码和复杂问答任务来说，多条独立路径往往能覆盖不同解法，降低单次生成偶然错误的影响。

但 self-consistency 不是免费午餐。它会增加推理成本，需要合理设置 temperature、top-p、采样数量、答案抽取、答案标准化和聚合策略。本章系统讲清这些工程细节。

## 3.1 为什么需要多路径采样

单次 CoT 生成容易受随机性和局部错误影响。

例如同一道题，模型可能生成：

```text
路径 1：步骤正确，答案 42
路径 2：中间计算错，答案 36
路径 3：步骤正确，答案 42
路径 4：理解题意错，答案 40
```

如果只看一次，可能抽到错误答案。如果采样多次并投票，答案 42 更可能被选中。

面试回答：

```text
Self-consistency 的思想是对同一个问题采样多条推理路径，抽取每条路径的最终答案，然后通过投票或打分选择最一致的答案。它利用了模型在不同采样路径上的多样性，常能提升数学和逻辑推理任务的可靠性，但代价是推理成本增加。
```

## 3.2 Self-Consistency 的基本流程

完整流程：

1. 给定问题。
2. 用非贪心采样生成 `k` 条 CoT。
3. 从每条 CoT 中抽取最终答案。
4. 对答案做标准化。
5. 投票或加权聚合。
6. 输出最终答案。

伪代码：

```python
answers = []

for _ in range(k):
    response = model.generate(prompt, temperature=0.7, top_p=0.95)
    answer = extract_final_answer(response)
    answer = normalize_answer(answer)
    answers.append(answer)

final_answer = majority_vote(answers)
```

关键不是“多生成几次”这么简单，而是答案抽取和标准化要可靠。

### 3.2.1 关键公式与 Self-Consistency 指标速查

把第 `i` 道题的候选集合写成：

```math
\mathcal{C}_i=
\{(z_{ij},\hat y_{ij},s_{ij},T_{ij})\}_{j=1}^{K_i}
```

其中 `z_ij` 是第 `j` 条推理链，`hat y_ij` 是候选最终答案，`s_ij` 是可选的 verifier 或规则分数，`T_ij` 是该候选消耗的 token 数。

温度采样把 logits `l_v` 变成概率：

```math
p_v(\tau)=
\frac{\exp(l_v/\tau)}
{\sum_{u\in\mathcal{V}}\exp(l_u/\tau)}
```

其中 `tau` 是 temperature。`tau` 越小，分布越尖锐；`tau` 越大，候选越分散。

Top-k 采样集合：

```math
\mathcal{V}_k=
\{v: l_v \mathrm{\ in\ top\ } k\}
```

Top-p / nucleus sampling 先按概率从大到小排序，选择最小前缀集合：

```math
\mathcal{N}_p=
\{v_{(1)},\ldots,v_{(m)}\},
\qquad
\sum_{j=1}^{m}p_{(j)}\ge p
```

多数投票：

```math
\hat y_i^{\mathrm{maj}}
=
\arg\max_a
\sum_{j=1}^{K_i}
\mathbb{1}[\nu(\hat y_{ij})=a]
```

其中 `nu` 是答案标准化函数，例如把 `42.0`、`42 minutes` 和 `答案：42` 合并成同一个答案等价类。

加权投票：

```math
\hat y_i^{\mathrm{w}}
=
\arg\max_a
\sum_{j=1}^{K_i}
s_{ij}\mathbb{1}[\nu(\hat y_{ij})=a]
```

Self-consistency accuracy：

```math
A_{\mathrm{sc}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[\hat y_i^{\mathrm{maj}}=y_i^\star]
```

Pass@k 不衡量“最终选对”，而衡量候选集合中至少有一个正确候选的概率。给定 `n` 个候选、其中 `c` 个正确，不放回估计为：

```math
\mathrm{pass@}k
=
1-
\frac{\binom{n-c}{k}}
{\binom{n}{k}}
```

平均候选多样性可以粗略用标准化答案去重比例表示：

```math
D_{\mathrm{ans}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\frac{|\{\nu(\hat y_{ij})\}_{j=1}^{K_i}|}
{K_i}
```

Self-consistency token 成本：

```math
C_{\mathrm{sc}}
=
\sum_{i=1}^{N}
\sum_{j=1}^{K_i}T_{ij}
```

一个简化上线门禁：

```math
G_{\mathrm{sc}}
=
\mathbb{1}[
A_{\mathrm{sc}}\ge A_{\mathrm{greedy}}
\land
\mathrm{pass@}k\ge \alpha
\land
D_{\mathrm{ans}}\ge \delta
\land
C_{\mathrm{sc}}\le C_{\max}
]
```

面试中要强调：`A_sc`、`pass@k` 和 verifier rerank accuracy 是三种不同指标。`pass@k` 高说明候选集合里有正确解，但如果 majority vote 或 verifier 选错，线上最终答案仍然可能错。

## 3.3 Greedy Decoding 为什么不适合 self-consistency

Greedy decoding 每一步都选择概率最高的 token。

如果模型和输入完全相同，greedy 通常每次都生成同一条路径。

```text
sample 1 -> same answer
sample 2 -> same answer
sample 3 -> same answer
```

这没有多样性，不能发挥 self-consistency。

Self-consistency 通常需要 sampling，例如设置 temperature、top-p 或 top-k，让模型探索不同推理路径。

## 3.4 Temperature

Temperature 控制采样分布的平滑程度。

低 temperature：

1. 输出更稳定。
2. 多样性较低。
3. 更接近 greedy。

高 temperature：

1. 输出更多样。
2. 可能探索不同解法。
3. 错误和跑偏风险更高。

reasoning 中常需要中等 temperature。太低没有多样性，太高会生成不可靠推理。

工程上需要根据任务调参，而不是固定一个值适用所有任务。

## 3.5 Top-p 和 Top-k

Top-k 限制每一步只从概率最高的 `k` 个 token 中采样。

Top-p 选择累计概率达到 `p` 的候选 token 集合。

在 reasoning 任务中：

1. top-p 可以保留合理多样性。
2. top-k 可以避免长尾低质量 token。
3. 过强截断可能让不同路径不够多样。
4. 过弱截断可能导致推理跑偏。

常见组合是 moderate temperature + top-p。

## 3.6 答案抽取

Self-consistency 的一个关键工程问题是：如何从一段长 CoT 中抽取最终答案。

推荐让模型使用固定格式：

```text
答案：42
```

或者：

```text
Final answer: 42
```

抽取函数可以基于规则：

```python
def extract_final_answer(text):
    marker = "答案："
    if marker in text:
        return text.split(marker)[-1].strip().split("\n")[0]
    return text.strip().split("\n")[-1]
```

真实项目中要处理：

1. 多个答案标记。
2. 单位。
3. 分数和小数。
4. 多选题选项。
5. LaTeX。
6. 中文数字。
7. 模型输出解释和答案混在一起。

## 3.7 答案标准化

同一个答案可能有不同表达。

例如：

```text
42
42.0
答案是 42
四十二
```

如果不标准化，投票会被分散。

标准化包括：

1. 去掉空格和标点。
2. 统一大小写。
3. 数值转标准格式。
4. 分数化简。
5. 单位归一。
6. 多选题提取选项字母。

数学任务中，答案标准化会显著影响评估结果。

## 3.8 Majority Vote

最简单聚合是多数投票。

```python
from collections import Counter


def majority_vote(answers):
    return Counter(answers).most_common(1)[0][0]
```

优点：

1. 简单。
2. 不需要额外模型。
3. 对答案空间明确的任务有效。

缺点：

1. 无法判断少数正确、多数错误的情况。
2. 对开放式答案不稳定。
3. 依赖答案抽取和标准化。

## 3.9 加权投票

如果每条路径有置信度或 verifier 分数，可以加权投票。

```text
answer A: 0.8 + 0.7
answer B: 0.9
最终 A 分数 1.5，B 分数 0.9
```

分数来源可以是：

1. 模型 log probability。
2. Verifier score。
3. Reward model。
4. 工具校验结果。
5. 规则检查。

加权投票比多数投票更强，但依赖评分器质量。

## 3.10 Pass@k

Pass@k 常用于代码生成和数学候选生成评估。

直觉：如果生成 `k` 个候选，只要其中一个正确，就认为 pass。

例如代码题：

1. 生成 10 个解法。
2. 对每个解法运行单元测试。
3. 只要有一个通过，就 pass@10 成功。

Pass@k 反映模型“生成正确候选的能力”，不等于模型“自动选择正确候选的能力”。如果没有 verifier 或测试器，生成了正确候选也未必能选出来。

面试中要区分：

1. pass@k：候选集合里是否有正确答案。
2. self-consistency accuracy：聚合后最终答案是否正确。
3. verifier accuracy：能否选中正确候选。

## 3.11 成本和质量曲线

采样数 `k` 越大，通常准确率会先提升，但边际收益递减。

```text
k=1  -> baseline
k=4  -> 明显提升
k=16 -> 继续提升但成本高
k=64 -> 可能收益很小
```

要画成本-质量曲线：

1. 横轴：平均 tokens、延迟或调用次数。
2. 纵轴：准确率。
3. 比较不同 k、temperature、verifier 策略。

工程上不能只追求最高准确率，还要考虑用户能接受的延迟和成本。

## 3.12 什么时候 self-consistency 有效

有效场景：

1. 答案可标准化。
2. 存在多条推理路径。
3. 单次生成容易偶然出错。
4. 模型有一定基础能力。
5. 多数正确路径能压过错误路径。

典型任务：

1. 数学应用题。
2. 逻辑题。
3. 选择题。
4. 代码候选生成。
5. 可执行工具验证任务。

不适合场景：

1. 开放创意写作。
2. 答案难以标准化。
3. 模型系统性错误。
4. 低延迟强约束场景。
5. 高风险任务中无 verifier 的投票。

## 3.13 多样性和正确性的平衡

Self-consistency 需要多样性，但不是越随机越好。

太保守：

```text
所有样本几乎一样，投票没有意义。
```

太随机：

```text
推理路径发散，错误答案增多。
```

理想状态是：候选路径有足够多样性，但仍围绕合理解法。

调参方向：

1. temperature。
2. top-p。
3. prompt 约束。
4. 最大推理长度。
5. verifier 过滤。

## 3.14 Self-Consistency 和 Verifier 结合

更强方案是采样 + verifier。

流程：

1. 采样多个候选。
2. 抽取答案和推理链。
3. verifier 给每个候选打分。
4. 选择最高分或加权投票。

代码题中 verifier 可以是单元测试。

数学题中 verifier 可以是：

1. 规则检查。
2. 计算器。
3. 符号系统。
4. 训练的 reward model。

Verifier 的引入能解决“多数错、少数对”的问题，但也可能引入 verifier 偏差。

## 3.15 一个完整实验设计

如果做 self-consistency 项目，可以这样设计：

1. 选择数据集，例如 GSM8K。
2. 设置 baseline：greedy CoT。
3. 设置采样策略：k=4、8、16。
4. 调 temperature 和 top-p。
5. 做答案抽取和标准化。
6. 比较 majority vote 和 verifier rerank。
7. 统计 accuracy、tokens、latency、cost。
8. 分析 bad cases。

bad case 分类：

1. 题意理解错。
2. 计算错。
3. 答案抽取错。
4. 多数投票选错。
5. 所有候选都错。
6. verifier 选错。

## 3.16 最小可运行 self-consistency / 采样审计 demo

下面这个 demo 用 0 依赖 Python 模拟 5 道 toy reasoning 题。它同时比较：

1. `greedy`：单次直出答案。
2. `majority`：对多个采样候选做答案标准化后多数投票。
3. `weighted`：按 verifier / 规则分数做加权投票。
4. `pass@2`：候选集合里至少有一个正确答案的概率估计。

重点不是模拟真实模型，而是把 self-consistency 项目里最容易漏掉的工程点一次跑通：答案标准化、候选多样性、majority failure、weighted rescue、pass@k 和 token 成本。

```python
from collections import Counter, defaultdict
from math import comb
import re

problems = [
    {
        "id": "water_tank",
        "gold": "10",
        "greedy": "12",
        "candidates": [
            {"answer": "10 minutes", "score": 0.86, "tokens": 58},
            {"answer": "10", "score": 0.78, "tokens": 54},
            {"answer": "12", "score": 0.42, "tokens": 46},
            {"answer": "10.0", "score": 0.81, "tokens": 61},
        ],
    },
    {
        "id": "distractor_math",
        "gold": "12",
        "greedy": "99",
        "candidates": [
            {"answer": "111", "score": 0.35, "tokens": 52},
            {"answer": "111", "score": 0.40, "tokens": 49},
            {"answer": "12", "score": 0.91, "tokens": 57},
            {"answer": "111", "score": 0.37, "tokens": 50},
            {"answer": "12 units", "score": 0.88, "tokens": 59},
        ],
    },
    {
        "id": "code_loop",
        "gold": "pass",
        "greedy": "fail",
        "candidates": [
            {"answer": "fail", "score": 0.20, "tokens": 42},
            {"answer": "pass", "score": 0.83, "tokens": 64},
            {"answer": "pass", "score": 0.79, "tokens": 68},
            {"answer": "pass", "score": 0.77, "tokens": 63},
        ],
    },
    {
        "id": "logic_grid",
        "gold": "blue",
        "greedy": "red",
        "candidates": [
            {"answer": "blue", "score": 0.72, "tokens": 47},
            {"answer": "green", "score": 0.41, "tokens": 43},
            {"answer": "blue", "score": 0.74, "tokens": 45},
            {"answer": "red", "score": 0.30, "tokens": 44},
        ],
    },
    {
        "id": "capital_lookup",
        "gold": "tokyo",
        "greedy": "Tokyo",
        "candidates": [
            {"answer": "tokyo", "score": 0.82, "tokens": 20},
            {"answer": "kyoto", "score": 0.25, "tokens": 35},
            {"answer": "Tokyo.", "score": 0.81, "tokens": 22},
            {"answer": "tokyo city", "score": 0.76, "tokens": 24},
        ],
    },
]

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_answer(answer):
    text = answer.strip().lower()
    if text in {"pass", "fail", "blue", "green", "red", "tokyo", "kyoto"}:
        return text
    if text.endswith("."):
        text = text[:-1]
    if text == "tokyo city":
        return "tokyo"
    numbers = NUMBER_RE.findall(text)
    if numbers:
        value = float(numbers[-1])
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return text


def majority_vote(answers):
    counts = Counter(answers)
    return counts.most_common(1)[0][0]


def weighted_vote(candidates):
    scores = defaultdict(float)
    for cand in candidates:
        scores[normalize_answer(cand["answer"])] += cand["score"]
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


rows = []
for item in problems:
    normalized = [
        normalize_answer(cand["answer"])
        for cand in item["candidates"]
    ]
    n = len(normalized)
    c = sum(ans == item["gold"] for ans in normalized)
    rows.append(
        {
            "id": item["id"],
            "greedy": normalize_answer(item["greedy"]),
            "majority": majority_vote(normalized),
            "weighted": weighted_vote(item["candidates"]),
            "correct_candidates": c,
            "unique_answers": len(set(normalized)),
            "tokens": sum(cand["tokens"] for cand in item["candidates"]),
            "pass_at_2": round(pass_at_k(n, c, 2), 3),
        }
    )

greedy_accuracy = sum(
    row["greedy"] == item["gold"]
    for row, item in zip(rows, problems)
) / len(problems)
majority_accuracy = sum(
    row["majority"] == item["gold"]
    for row, item in zip(rows, problems)
) / len(problems)
weighted_accuracy = sum(
    row["weighted"] == item["gold"]
    for row, item in zip(rows, problems)
) / len(problems)
pass_at_1_est = sum(
    row["correct_candidates"] / len(item["candidates"])
    for row, item in zip(rows, problems)
) / len(problems)
pass_at_2_est = sum(row["pass_at_2"] for row in rows) / len(rows)
total_tokens = sum(row["tokens"] for row in rows)
total_candidates = sum(len(item["candidates"]) for item in problems)
avg_unique_ratio = sum(
    row["unique_answers"] / len(item["candidates"])
    for row, item in zip(rows, problems)
) / len(rows)
majority_failures = [
    row["id"]
    for row, item in zip(rows, problems)
    if row["majority"] != item["gold"]
]
weighted_rescues = [
    row["id"]
    for row, item in zip(rows, problems)
    if row["majority"] != item["gold"]
    and row["weighted"] == item["gold"]
]

report = {
    "greedy_accuracy": round(greedy_accuracy, 3),
    "majority_accuracy": round(majority_accuracy, 3),
    "weighted_accuracy": round(weighted_accuracy, 3),
    "pass_at_1_est": round(pass_at_1_est, 3),
    "pass_at_2_est": round(pass_at_2_est, 3),
    "avg_unique_answer_ratio": round(avg_unique_ratio, 3),
    "total_candidates": total_candidates,
    "total_tokens": total_tokens,
    "cost_per_majority_correct": round(
        total_tokens / (majority_accuracy * len(problems)), 3
    ),
    "majority_failures": majority_failures,
    "weighted_rescues": weighted_rescues,
    "per_problem": rows,
}
gates = {
    "majority_beats_greedy": (
        report["majority_accuracy"] > report["greedy_accuracy"]
    ),
    "weighted_beats_majority": (
        report["weighted_accuracy"] > report["majority_accuracy"]
    ),
    "pass_at_2_ready": report["pass_at_2_est"] >= 0.8,
    "diversity_not_too_low": report["avg_unique_answer_ratio"] >= 0.45,
    "budget_ok": report["cost_per_majority_correct"] <= 260,
}
report["gates"] = gates
report["gate_pass"] = all(gates.values())

for key, value in report.items():
    print(f"{key}={value}")
```

期望输出：

```text
greedy_accuracy=0.2
majority_accuracy=0.8
weighted_accuracy=1.0
pass_at_1_est=0.63
pass_at_2_est=0.907
avg_unique_answer_ratio=0.53
total_candidates=21
total_tokens=1003
cost_per_majority_correct=250.75
majority_failures=['distractor_math']
weighted_rescues=['distractor_math']
per_problem=[{'id': 'water_tank', 'greedy': '12', 'majority': '10', 'weighted': '10', 'correct_candidates': 3, 'unique_answers': 2, 'tokens': 219, 'pass_at_2': 1.0}, {'id': 'distractor_math', 'greedy': '99', 'majority': '111', 'weighted': '12', 'correct_candidates': 2, 'unique_answers': 2, 'tokens': 267, 'pass_at_2': 0.7}, {'id': 'code_loop', 'greedy': 'fail', 'majority': 'pass', 'weighted': 'pass', 'correct_candidates': 3, 'unique_answers': 2, 'tokens': 237, 'pass_at_2': 1.0}, {'id': 'logic_grid', 'greedy': 'red', 'majority': 'blue', 'weighted': 'blue', 'correct_candidates': 2, 'unique_answers': 3, 'tokens': 179, 'pass_at_2': 0.833}, {'id': 'capital_lookup', 'greedy': 'tokyo', 'majority': 'tokyo', 'weighted': 'tokyo', 'correct_candidates': 3, 'unique_answers': 2, 'tokens': 101, 'pass_at_2': 1.0}]
gates={'majority_beats_greedy': True, 'weighted_beats_majority': True, 'pass_at_2_ready': True, 'diversity_not_too_low': True, 'budget_ok': True}
gate_pass=True
```

这段 demo 的面试表达是：

1. `majority_accuracy` 高于 `greedy_accuracy`，说明多路径采样能降低单次偶然错误。
2. `distractor_math` 的 majority vote 选错，但 weighted vote 选对，说明 pass@k 高不等于自动聚合后一定对。
3. `pass_at_2_est` 高于 `pass_at_1_est`，说明增加候选数能提高“至少包含一个正确解”的概率。
4. `avg_unique_answer_ratio` 不能太低，否则多次采样只是重复同一条路径。
5. `cost_per_majority_correct` 必须和准确率一起看，否则 self-consistency 会变成简单堆 token。

## 3.17 面试官会怎么问

### 问题一：Self-consistency 是什么？

回答模板：

```text
Self-consistency 是对同一个问题采样多条推理路径，抽取最终答案，并通过多数投票或加权投票选择答案的方法。它利用不同 CoT 路径的多样性提升推理可靠性，常用于数学和逻辑任务。
```

### 问题二：为什么 self-consistency 需要 sampling？

回答模板：

```text
因为如果使用 greedy decoding，同一个输入通常每次生成同样路径，没有多样性。Self-consistency 需要通过 temperature、top-p 等采样方式生成不同推理路径，才能让投票发挥作用。
```

### 问题三：pass@k 和 majority vote 有什么区别？

回答模板：

```text
pass@k 只看 k 个候选中是否至少有一个正确，衡量生成正确候选的能力；majority vote 是从多个候选答案中投票选最终答案，衡量自动聚合后的准确率。没有 verifier 时，pass@k 高不代表最终能选对。
```

### 问题四：Self-consistency 的成本问题怎么处理？

回答模板：

```text
采样 k 条路径会让成本和延迟大致随 k 增长。工程上要画成本-质量曲线，选择合适 k，并可结合早停、verifier 过滤、动态预算或只对困难问题启用 self-consistency。
```

### 问题五：Self-consistency 为什么可能失败？

回答模板：

```text
如果模型系统性理解错题，多数采样也会错；如果答案抽取或标准化错误，投票会失真；如果 temperature 太高，路径可能跑偏；如果正确候选是少数，没有 verifier 时 majority vote 也可能选错。
```

## 3.18 小练习

1. 对一道数学题采样 5 条 CoT，并手动投票。
2. 写一个 `extract_final_answer` 函数。
3. 写一个答案标准化函数，处理整数、小数和单位。
4. 比较 greedy、temperature=0.3、temperature=0.8 的输出差异。
5. 设计一个 pass@k 评估流程。
6. 画出 k 从 1 到 16 的成本-质量曲线。
7. 构造一个 majority vote 选错但 verifier 能选对的例子。
8. 修改上面的 demo，把 `distractor_math` 的 verifier 分数调低，观察 `weighted_accuracy` 如何变化。

## 3.19 本章总结

Self-consistency 是 test-time compute 的入门方法。它通过多路径采样和答案聚合提升推理可靠性，尤其适合答案可标准化、存在多种解法的数学和逻辑任务。

需要记住：

1. Greedy decoding 没有多样性，不适合 self-consistency。
2. Temperature 和 top-p 控制候选多样性。
3. 答案抽取和标准化是工程关键。
4. Majority vote 简单但可能选错。
5. Pass@k 衡量候选中是否有正确答案，不等于最终选择能力。
6. Verifier 可以提升候选选择质量。
7. 采样数量增加会带来成本和延迟，需要画成本-质量曲线。

下一章会进入 verifier 与 reward model，系统讲 outcome verifier、process verifier、reward model、候选重排和验证器训练。
