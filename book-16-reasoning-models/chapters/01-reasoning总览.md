# 第一章：Reasoning 总览

Reasoning model 是当前大模型能力竞争的重要方向之一。传统 chat model 更强调对话流畅、指令遵循和通用知识，而 reasoning model 更强调多步推理、数学证明、代码求解、规划、验证和复杂任务分解。它不仅要“答得像”，还要在有标准答案或可验证目标的任务上“答得对”。

Reasoning 的核心不只是让模型输出更长的解释。真正的 reasoning 涉及训练数据、Chain-of-Thought、self-consistency、verifier、process supervision、search、tool execution、test-time compute scaling 和评估体系。本章先建立全局地图，后续章节再逐个展开。

## 0. 本讲资料边界与第二轮精修口径

本章第二轮精修前，重点参考了 Chain-of-Thought、Zero-shot CoT、Self-Consistency、Training Verifiers to Solve Math Word Problems、Let's Verify Step by Step、Tree of Thoughts、HumanEval / pass@k、MATH / GSM8K，以及 OpenAI 关于 reasoning models 和 test-time compute 的公开资料。

本章是第十六册总览章，只建立 reasoning model 的全局地图和面试主线，不展开后续各章的完整技术细节。这里不把“输出很长解释”直接等同于真实推理，也不把闭源 reasoning 模型的内部训练 recipe 写成公开事实。第二轮精修重点是：

1. 区分 chat model、reasoning model、CoT prompt、训练出来的 reasoning 能力和推理时搜索 / 验证系统。
2. 用公式解释 self-consistency、best-of-n、pass@k、verifier reranking、process supervision、test-time compute 成本和上线门禁。
3. 用 0 依赖 demo 展示 greedy、self-consistency、verifier、pass@k、过程步骤准确率和 token 成本之间的关系。
4. 明确 reasoning 的安全边界：更强推理可能提升数学 / 代码 / 规划能力，也可能提升攻击规划、工具滥用和看似严谨的错误解释。

## 1.1 什么是 Reasoning

Reasoning 可以理解为模型从已知信息出发，通过一系列中间步骤得到结论的能力。

典型任务：

1. 数学题。
2. 代码题。
3. 逻辑题。
4. 规划任务。
5. 多跳问答。
6. 复杂工具使用。
7. 科学问题推理。

例如：

```text
一个水箱每分钟进水 3 升，每分钟漏水 1 升，容量 20 升，从空开始多久装满？
```

模型不能只凭语言模式猜答案，而要计算净流入速度、容量和时间。

面试回答：

```text
Reasoning 指模型基于问题条件进行多步推理、计算、验证和决策的能力。它常出现在数学、代码、逻辑、多跳问答和规划任务中。和普通聊天不同，reasoning 更强调中间步骤是否合理、最终答案是否可验证，以及推理时如何使用额外计算和验证器提升正确率。
```

## 1.2 Reasoning Model 和 Chat Model 的区别

Chat model 主要优化：

1. 指令遵循。
2. 对话自然度。
3. 通用知识问答。
4. 安全拒答。
5. 用户体验。

Reasoning model 更强调：

1. 多步推理正确性。
2. 数学和代码能力。
3. 长思维链。
4. 自我检查。
5. verifier 或 reward model 选择答案。
6. test-time compute 扩展。

区别不是绝对的。很多强 chat model 也有 reasoning 能力，很多 reasoning model 也可以聊天。但训练目标、数据分布、推理策略和评估重点不同。

一个 chat model 可能回答流畅但算错；一个 reasoning model 可能花更多 token 思考，但在可验证任务上更准确。

## 1.3 为什么 Next-Token Prediction 也能产生推理

LLM 的基础训练目标是预测下一个 token。看起来这只是语言建模，为什么能推理？

原因包括：

1. 训练语料中包含大量推理过程。
2. 数学、代码、证明和解释文本本身有结构。
3. Transformer 能从上下文中组合模式。
4. 大规模参数和数据带来泛化能力。
5. 指令微调进一步强化了解题格式。

但 next-token prediction 本身不保证推理正确。模型可能生成“看起来像推理”的文本，但中间步骤错误，最终答案也错。

这就是为什么 reasoning 方向需要 verifier、过程监督、工具执行和 test-time compute。

### 1.3.1 关键公式与 Reasoning 总览速查

把一个 reasoning 样本抽象为：

```math
r_i=(x_i,y_i^\star,C_i,V_i,B_i)
```

其中 `x_i` 是题目，`y_i^star` 是参考答案，`C_i` 是候选推理链集合，`V_i` 是 verifier 或测试器，`B_i` 是推理预算。

**自回归生成与中间步骤**

语言模型仍然按自回归方式生成推理链和答案：

```math
p_\theta(y_{1:T}\mid x)=
\prod_{t=1}^{T}
p_\theta(y_t\mid x,y_{1:t-1})
```

Reasoning 的特殊之处不在于目标函数突然变成符号证明，而在于训练数据、提示、搜索、验证器和推理预算让模型更倾向于生成可验证的中间状态。

**单次回答准确率**

如果只生成一次，准确率为：

```math
A_1=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[\hat y_i=y_i^\star]
```

面试里要强调：greedy / temperature=0 的单次回答不等于模型的全部 reasoning 潜力。

**Self-consistency**

对同一道题采样 `K` 条推理链，抽取最终答案 `a_{ik}`。多数投票可以写成：

```math
\hat y_i^{\mathrm{sc}}=
\mathrm{mode}(a_{i1},\ldots,a_{iK})
```

对应准确率：

```math
A_{\mathrm{sc}}=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[\hat y_i^{\mathrm{sc}}=y_i^\star]
```

self-consistency 是用更多采样成本换候选多样性和答案稳定性，但如果模型系统性误解题意，多数投票也可能错。

**Verifier / Best-of-N**

生成 `K` 个候选后，用 verifier 分数 `s_{ik}` 选择最高分候选：

```math
\hat y_i^{\mathrm{ver}}=
y_{ij},
\qquad
j=\mathrm{index\ of\ max}_{k}(s_{ik})
```

对应准确率：

```math
A_{\mathrm{ver}}=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[\hat y_i^{\mathrm{ver}}=y_i^\star]
```

这里的 verifier 可以是规则、单元测试、代码执行器、数学检查器、reward model 或 LLM judge。最关键的面试点是：生成器负责提出候选，verifier 负责筛选。

**pass@k**

代码和可执行验证任务常看 pass@k。设一题生成 `n` 个候选，其中 `c` 个正确，不放回抽 `k` 个至少命中一个正确解的估计为：

```math
\mathrm{pass@}k=
1-
\frac{\binom{n-c}{k}}
{\binom{n}{k}}
```

当 `n-c<k` 时，pass@k 记为 1。它衡量候选集合里是否存在正确解，不等于一次生成就可靠。

**Process supervision**

如果每条推理链有步骤标签 `z_{ij}`，步骤级准确率可写成：

```math
A_{\mathrm{step}}=
\frac{\sum_i\sum_j z_{ij}}
{\sum_i M_i}
```

其中 `z_ij=1` 表示第 `i` 个样本的第 `j` 个推理步骤正确。过程监督能定位中间错误，但标注成本高，正确过程也可能有多种写法。

**Test-time compute 成本**

若每个样本采样 `K_i` 条候选，第 `k` 条使用 token 数为 `T_{ik}`，总推理成本可近似为：

```math
C_{\mathrm{ttc}}=
\sum_{i=1}^{N}
\sum_{k=1}^{K_i}
T_{ik}
```

更真实的系统还要加入 verifier、工具执行、搜索节点、队列等待和人工复核成本。

**Reasoning 上线门禁**

一个简化 reasoning 门禁可以写成：

```math
G_{\mathrm{reason}}=
\mathbb{1}[
A_{\mathrm{ver}}\ge \alpha
\land
A_{\mathrm{step}}\ge \beta
\land
C_{\mathrm{ttc}}\le C_{\max}
\land
R_{\mathrm{unsafe}}\le \rho
]
```

这说明 reasoning 系统不能只追求更高正确率，还要受成本、延迟、安全和可解释失败分析约束。

## 1.4 Chain-of-Thought

Chain-of-Thought，简称 CoT，是让模型显式生成中间推理步骤。

示例：

```text
先计算净流入速度：3 - 1 = 2 升/分钟。
水箱容量是 20 升。
时间 = 20 / 2 = 10 分钟。
答案是 10 分钟。
```

CoT 的作用：

1. 给模型更多中间计算空间。
2. 让复杂问题分解成步骤。
3. 便于人或 verifier 检查。
4. 可以提升数学和逻辑题表现。

局限：

1. CoT 可能看起来合理但其实错误。
2. 长推理会增加成本和延迟。
3. 公开 CoT 可能带来安全和隐私问题。
4. 不是所有任务都需要长推理。

## 1.5 Test-Time Compute

Test-time compute 指推理阶段额外花费计算来提升答案质量。

常见方式：

1. 生成更长思维链。
2. 采样多个解法。
3. self-consistency 投票。
4. verifier 选择最佳答案。
5. search 或 tree-of-thought。
6. 工具执行和反馈修正。

直觉：训练时模型参数固定，但推理时可以花更多计算探索更多候选、检查错误和选择更可靠答案。

面试回答：

```text
Test-time compute 是在推理阶段增加计算预算来提升推理质量，例如生成多条 CoT、self-consistency 投票、用 verifier 打分、搜索不同解法或执行代码验证。它的核心 trade-off 是准确率提升与延迟、成本增加之间的权衡。
```

## 1.6 Self-Consistency

Self-consistency 是一种简单有效的推理增强方法。

流程：

1. 对同一个问题采样多条推理路径。
2. 抽取每条路径的最终答案。
3. 对答案投票。
4. 选择出现最多或 verifier 分数最高的答案。

例子：

```text
sample 1 -> answer A
sample 2 -> answer B
sample 3 -> answer A
sample 4 -> answer A
最终选择 A
```

优点：

1. 实现简单。
2. 不需要重新训练模型。
3. 对数学和推理题常有效。

缺点：

1. 成本随采样数线性增长。
2. 如果模型系统性错误，投票也会错。
3. 答案抽取和标准化很重要。

## 1.7 Verifier

Verifier 是用来判断候选答案或推理过程质量的模型或程序。

类型：

1. Outcome verifier：只判断最终答案好不好。
2. Process verifier：判断每一步推理是否合理。
3. Programmatic verifier：用规则、单元测试、执行器验证。
4. Reward model：给候选答案打分。

在数学题中，verifier 可以检查最终答案是否正确或步骤是否成立。在代码题中，最强 verifier 往往是执行测试用例。

Verifier 的价值：生成模型负责提出候选，verifier 负责筛选和纠错。

## 1.8 Process Supervision

过程监督不是只监督最终答案，而是监督中间步骤。

例如数学题：

```text
step 1: 正确
step 2: 正确
step 3: 错误
```

优势：

1. 能更早发现错误。
2. 给模型更细粒度训练信号。
3. 有助于复杂推理。

挑战：

1. 标注成本高。
2. 中间步骤可能有多种正确写法。
3. 过程正确不一定最终答案正确，反之亦然。
4. 容易出现“看起来规范”的伪推理。

## 1.9 Search 和 Tree-of-Thought

Search 方法把推理看成搜索过程。

模型不是一次生成完整答案，而是在多个中间状态中探索。

常见思路：

1. 生成多个下一步。
2. 用 verifier 或 heuristic 评分。
3. 保留较好的分支。
4. 继续展开。

Tree-of-Thought 是这种思想的代表：把 reasoning steps 组织成树，而不是单条链。

优点：

1. 能探索多个思路。
2. 可结合 verifier 剪枝。
3. 对规划和复杂问题有帮助。

缺点：

1. 成本高。
2. 搜索空间大。
3. 评分器质量决定效果。

## 1.10 数学推理

数学推理是 reasoning model 最重要的评估场景之一。

能力包括：

1. 算术。
2. 代数。
3. 几何。
4. 概率统计。
5. 证明。
6. 多步应用题。

常见增强方式：

1. CoT 数据。
2. Self-consistency。
3. Verifier。
4. 过程监督。
5. Python 工具校验。
6. 题目难度 curriculum。

数学题的好处是答案通常可验证，因此适合研究 reasoning。

## 1.11 代码推理

代码推理的优势是可以执行验证。

模型可以：

1. 读题。
2. 写代码。
3. 运行测试。
4. 根据错误信息修复。
5. 迭代直到通过。

这比纯文本推理更可验证。单元测试、编译器、解释器都可以作为外部 verifier。

代码 reasoning 常见任务：

1. 算法题。
2. Bug 修复。
3. 单元测试生成。
4. 程序综合。
5. 代码解释。
6. 工具调用和执行反馈。

## 1.12 Reasoning 评估

Reasoning 评估不能只看回答是否长。

需要看：

1. 最终答案正确率。
2. 中间步骤正确性。
3. 鲁棒性。
4. 分布外泛化。
5. 成本和延迟。
6. 是否使用工具。
7. 是否能发现自己不确定。

常见 benchmark：

1. GSM8K。
2. MATH。
3. AIME 类竞赛题。
4. HumanEval。
5. MBPP。
6. Big-Bench Hard。

但要警惕 benchmark contamination。模型可能见过题目或类似解答，导致评估高估真实推理能力。

## 1.13 Reasoning 的安全和局限

Reasoning 能力增强也带来风险。

风险：

1. 更强的攻击规划。
2. 更会绕过规则。
3. 更强代码生成可能带来恶意代码风险。
4. 长 CoT 可能泄露敏感推理或系统策略。
5. 模型可能给出看似严谨但错误的解释。

局限：

1. 长思维链不等于真实推理。
2. Verifier 也可能错。
3. 搜索成本高。
4. 工具执行依赖环境。
5. 复杂现实问题没有标准答案。

## 1.14 训练时计算和推理时计算的权衡

提升 reasoning 有两条路：

1. 训练时投入更多计算，让模型本身更强。
2. 推理时投入更多计算，让模型多想、多试、多验证。

训练时计算的优点：

1. 单次推理更快。
2. 能力内化到参数里。

推理时计算的优点：

1. 可按任务难度动态分配预算。
2. 不需要重新训练就能提升部分任务表现。
3. 可以结合 verifier 和工具。

核心 trade-off：准确率、延迟、成本和可靠性。

## 1.15 Reasoning 项目路线

适合简历的项目：

1. 数学推理 self-consistency 实验。
2. GSM8K/MATH 评估和错误归因。
3. Python 工具校验数学答案。
4. 代码生成 + 单元测试反馈。
5. Verifier 训练和候选重排。
6. Process supervision 数据构造。
7. Tree-of-Thought 搜索 demo。
8. Test-time compute 成本-质量曲线分析。

项目表达要强调：任务定义、数据、方法、评估、成本、bad case 和改进方向。

## 1.16 最小可运行 reasoning 成本 / 正确率审计 demo

下面这个 demo 不调用真实模型，而是模拟 4 道 reasoning 题的候选答案。它比较 greedy 单次回答、self-consistency 多数投票、verifier reranking 和 pass@k，同时统计过程步骤准确率与 token 成本。

```python
from collections import Counter
from math import comb


problems = [
    {
        "id": "water_tank",
        "gold": "10",
        "prompt_tokens": 40,
        "candidates": [
            {"answer": "12", "correct": False, "verifier": 0.25, "tokens": 42, "steps": [1, 0]},
            {"answer": "10", "correct": True, "verifier": 0.93, "tokens": 50, "steps": [1, 1, 1]},
            {"answer": "10", "correct": True, "verifier": 0.88, "tokens": 48, "steps": [1, 1]},
        ],
    },
    {
        "id": "code_sum",
        "gold": "pass",
        "prompt_tokens": 40,
        "candidates": [
            {"answer": "fail", "correct": False, "verifier": 0.45, "tokens": 55, "steps": [1, 0, 0]},
            {"answer": "pass", "correct": True, "verifier": 0.91, "tokens": 60, "steps": [1, 1, 1]},
            {"answer": "pass", "correct": True, "verifier": 0.87, "tokens": 58, "steps": [1, 1, 0]},
        ],
    },
    {
        "id": "logic_grid",
        "gold": "blue",
        "prompt_tokens": 40,
        "candidates": [
            {"answer": "blue", "correct": True, "verifier": 0.86, "tokens": 35, "steps": [1, 1]},
            {"answer": "blue", "correct": True, "verifier": 0.84, "tokens": 37, "steps": [1, 1]},
            {"answer": "blue", "correct": True, "verifier": 0.82, "tokens": 36, "steps": [1, 1]},
        ],
    },
    {
        "id": "distractor_math",
        "gold": "ignore",
        "prompt_tokens": 40,
        "candidates": [
            {"answer": "use_extra", "correct": False, "verifier": 0.55, "tokens": 45, "steps": [1, 0]},
            {"answer": "use_extra", "correct": False, "verifier": 0.51, "tokens": 44, "steps": [0, 0]},
            {"answer": "ignore", "correct": True, "verifier": 0.89, "tokens": 52, "steps": [1, 1, 1]},
        ],
    },
]


def majority_answer(candidates):
    counts = Counter(candidate["answer"] for candidate in candidates)
    max_count = max(counts.values())
    winners = {answer for answer, count in counts.items() if count == max_count}
    for candidate in candidates:
        if candidate["answer"] in winners:
            return candidate["answer"]
    raise AssertionError("unreachable")


def pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def audit_reasoning_budget(problems, k=2):
    greedy_correct = 0
    sc_correct = 0
    verifier_correct = 0
    pass1_values = []
    passk_values = []
    total_candidate_tokens = 0
    total_prompt_tokens = 0
    step_correct = 0
    step_total = 0
    per_problem = {}

    for problem in problems:
        candidates = problem["candidates"]
        gold = problem["gold"]
        greedy = candidates[0]["answer"]
        sc = majority_answer(candidates)
        best = max(candidates, key=lambda item: item["verifier"])["answer"]
        correct_count = sum(candidate["correct"] for candidate in candidates)
        n = len(candidates)

        greedy_correct += greedy == gold
        sc_correct += sc == gold
        verifier_correct += best == gold
        pass1_values.append(pass_at_k(n, correct_count, 1))
        passk_values.append(pass_at_k(n, correct_count, k))
        total_candidate_tokens += sum(candidate["tokens"] for candidate in candidates)
        total_prompt_tokens += problem["prompt_tokens"] * n
        step_correct += sum(sum(candidate["steps"]) for candidate in candidates)
        step_total += sum(len(candidate["steps"]) for candidate in candidates)
        per_problem[problem["id"]] = {
            "greedy": greedy,
            "self_consistency": sc,
            "verifier": best,
            "correct_candidates": correct_count,
        }

    total_tokens = total_prompt_tokens + total_candidate_tokens
    n_problems = len(problems)
    report = {
        "greedy_accuracy": round(greedy_correct / n_problems, 3),
        "self_consistency_accuracy": round(sc_correct / n_problems, 3),
        "verifier_accuracy": round(verifier_correct / n_problems, 3),
        "pass_at_1_est": round(sum(pass1_values) / n_problems, 3),
        "pass_at_2_est": round(sum(passk_values) / n_problems, 3),
        "avg_candidates": round(sum(len(p["candidates"]) for p in problems) / n_problems, 3),
        "process_step_accuracy": round(step_correct / step_total, 3),
        "total_tokens": total_tokens,
        "cost_per_verified_correct": round(total_tokens / max(1, verifier_correct), 3),
        "per_problem": per_problem,
    }
    gates = {
        "verifier_beats_greedy": report["verifier_accuracy"] > report["greedy_accuracy"],
        "self_consistency_beats_greedy": report["self_consistency_accuracy"] > report["greedy_accuracy"],
        "pass_at_2_ready": report["pass_at_2_est"] >= 0.85,
        "process_quality_ok": report["process_step_accuracy"] >= 0.7,
        "budget_ok": total_tokens <= 1200,
    }
    report["gates"] = gates
    report["gate_pass"] = all(gates.values())
    return report


report = audit_reasoning_budget(problems)
for key, value in report.items():
    print(f"{key}={value}")
```

参考输出：

```text
greedy_accuracy=0.25
self_consistency_accuracy=0.75
verifier_accuracy=1.0
pass_at_1_est=0.667
pass_at_2_est=0.917
avg_candidates=3.0
process_step_accuracy=0.759
total_tokens=1042
cost_per_verified_correct=260.5
per_problem={'water_tank': {'greedy': '12', 'self_consistency': '10', 'verifier': '10', 'correct_candidates': 2}, 'code_sum': {'greedy': 'fail', 'self_consistency': 'pass', 'verifier': 'pass', 'correct_candidates': 2}, 'logic_grid': {'greedy': 'blue', 'self_consistency': 'blue', 'verifier': 'blue', 'correct_candidates': 3}, 'distractor_math': {'greedy': 'use_extra', 'self_consistency': 'use_extra', 'verifier': 'ignore', 'correct_candidates': 1}}
gates={'verifier_beats_greedy': True, 'self_consistency_beats_greedy': True, 'pass_at_2_ready': True, 'process_quality_ok': True, 'budget_ok': True}
gate_pass=True
```

这个 demo 展示了 reasoning 系统的几个面试要点：

1. 单次 greedy 回答可能很弱，但候选集合里可能已有正确解。
2. self-consistency 能提升稳定性，但遇到系统性干扰时多数投票仍会错。
3. verifier 质量足够好时，best-of-n 可以显著提升正确率。
4. pass@k 衡量的是候选集合潜力，不等于线上一次调用体验。
5. reasoning 上线必须同时看准确率、步骤质量、token 成本和安全门禁。

## 1.17 面试官会怎么问

### 问题一：Reasoning model 和 chat model 有什么区别？

回答模板：

```text
Chat model 更强调指令遵循、对话自然和通用问答；reasoning model 更强调数学、代码、规划和多步推理正确性。Reasoning model 通常会使用更强 CoT 数据、verifier、过程监督、搜索和 test-time compute，在可验证任务上追求更高准确率。
```

### 问题二：为什么 test-time compute 重要？

回答模板：

```text
因为推理阶段可以通过生成多条思路、投票、搜索、verifier 选择和工具执行来提升复杂任务正确率。它让模型按任务难度动态花费计算，但代价是延迟和成本增加。
```

### 问题三：CoT 一定代表模型真的推理吗？

回答模板：

```text
不一定。CoT 可能只是生成看起来合理的解释，步骤和真实内部计算未必一致。它对很多任务有效，但可能出现伪推理、错误步骤和事后合理化。因此需要最终答案评估、过程验证、工具校验和鲁棒性测试。
```

### 问题四：Verifier 在 reasoning 中有什么作用？

回答模板：

```text
Verifier 用来评估候选答案或推理过程的质量。生成模型负责提出多个候选，verifier 负责打分、筛选或发现错误。数学中可以检查答案，代码中可以执行测试，复杂任务中可以训练 reward model 或 process verifier。
```

### 问题五：如何评估 reasoning 能力？

回答模板：

```text
要看最终答案正确率、中间步骤正确性、鲁棒性、分布外泛化、工具使用能力、成本和延迟。数学和代码任务适合评估，因为答案或程序行为可验证。同时要注意 benchmark contamination，避免高估真实推理能力。
```

## 1.18 本章总结

Reasoning model 的核心不是输出更长文本，而是在复杂、可验证任务中通过多步推理、搜索、验证和额外计算提升正确率。

需要记住：

1. Reasoning 强调数学、代码、逻辑、规划和多步任务。
2. CoT 提供中间推理空间，但不保证真实正确。
3. Self-consistency 用多样采样和投票提升可靠性。
4. Verifier 和 process supervision 提供更强监督和筛选。
5. Search 把推理变成候选路径探索。
6. Test-time compute 是准确率、延迟和成本之间的权衡。
7. Reasoning 评估要关注正确性、鲁棒性、成本和污染风险。

下一章会深入 Chain-of-Thought，讲清 CoT 的提出背景、prompt 形式、训练数据、显式/隐式推理、长思维链和常见误区。
