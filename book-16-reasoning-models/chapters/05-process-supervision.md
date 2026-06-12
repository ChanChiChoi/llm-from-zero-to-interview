# 第五章：Process Supervision

Process supervision，过程监督，是 reasoning model 中非常关键的一条路线。普通监督通常只看最终答案对不对；过程监督进一步要求监督中间推理步骤是否正确。对于复杂数学、代码、规划和多步任务来说，一个答案可能最后错了，但前面很多步骤是对的；也可能最终答案碰巧对，但中间推理是错的。只看结果会浪费大量训练信号，也难以定位错误来源。

本章系统讲过程监督的动机、步骤标注、PRM、推理轨迹、错误步骤定位、标注成本、和 search/verifier 的结合，以及常见风险。

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修参考公开资料中的 process supervision 和 PRM 线索，重点包括 OpenAI 的 [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)、OpenAI 对过程监督的公开介绍、PRM800K 数据集说明，以及更早的 [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)。这些资料共同说明了一点：在数学等可拆步骤任务上，只监督最终答案会丢失大量中间错误信息；如果能标注或评估每一步，模型和搜索过程就能更早发现坏路径。

但本章不把 process supervision 写成所有 reasoning 任务的通用最优解。真实系统还要同时看最终答案、搜索效果、标注成本、标注一致性、PRM 偏差、hidden reasoning 边界和安全展示策略。本章 demo 也只模拟 toy 级审计闭环，用于训练面试表达和工程指标意识，不代表真实 PRM 训练配方。

## 5.1 为什么需要过程监督

Outcome supervision 只监督最终答案。

例如：

```text
问题 -> 最终答案：错
```

但它不知道哪里错。

Process supervision 会看每一步：

```text
step 1: 正确
step 2: 正确
step 3: 错误
step 4: 基于错误继续推理
```

这样可以给模型更细粒度反馈。

面试回答：

```text
过程监督不是只看最终答案，而是监督推理过程中的每一步是否正确。它能更早发现错误，给模型更密集的训练信号，也能训练 process reward model 用于搜索和候选筛选。但它的主要代价是步骤标注成本高、步骤粒度难统一。
```

### 5.1.1 关键公式与 Process Supervision 指标速查

把第 `i` 个过程监督样本写成：

```math
p_i=(x_i,z_i,y_i^\star,\ell_i,e_i,c_i)
```

其中 `x_i` 是题目，`z_i=(z_{i1},\ldots,z_{iM_i})` 是推理步骤序列，`y_i^star` 是标准答案，`ell_i` 是步骤标签集合，`e_i` 是错误类型或第一处错误位置，`c_i` 是标注成本元信息。

步骤正确性和相关性可以写成：

```math
q_{ij}\in\{0,1\},
\qquad
r_{ij}\in\{0,1\}
```

其中 `q_ij=1` 表示第 `j` 步正确，`r_ij=1` 表示这一步和解题相关。过程监督不能只鼓励“正确废话”，否则模型可能生成大量无关但看似无害的步骤。

Outcome accuracy 仍然要保留：

```math
A_{\mathrm{out}}=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[\hat y_i=y_i^\star]
```

步骤分类准确率衡量 PRM 或过程分类器是否判断对每一步：

```math
A_{\mathrm{step}}=
\frac{
\sum_{i=1}^{N}
\sum_{j=1}^{M_i}
\mathbb{1}[\hat q_{ij}=q_{ij}]
}{
\sum_{i=1}^{N}M_i
}
```

第一处错误位置是过程监督里最重要的诊断信号之一：

```math
j_i^{\mathrm{err}}=
\min\{j:q_{ij}=0\}
```

如果样本没有错误步骤，可以把 `j_i^err` 记为无错误。第一处错误检测准确率可写成：

```math
A_{\mathrm{first}}=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[\hat j_i^{\mathrm{err}}=j_i^{\mathrm{err}}]
```

PRM 通常给“题目 + 已有步骤前缀 + 当前步骤”打分：

```math
s_{ij}=f_{\phi}(x_i,z_{i,1:j})
```

整条推理链的过程分数可以取平均，也可以取最薄弱步骤：

```math
S_{\mathrm{avg}}(z_i)=
\frac{1}{M_i}
\sum_{j=1}^{M_i}s_{ij}
```

```math
S_{\mathrm{min}}(z_i)=
\min_{1\le j\le M_i}s_{ij}
```

平均分适合整体质量排序，最小分适合“只要有一步明显错就要降级”的数学推理和代码推导任务。

PRM 接入搜索时，一个简化 beam 更新可以写成：

```math
B_{t+1}=
\mathrm{TopK}
(
\{b+a:b\in B_t,\ a\in\mathcal{A}(b)\},
S_{\mathrm{avg}}
)
```

其中 `B_t` 是第 `t` 步保留的中间状态集合，`A(b)` 是从状态 `b` 可生成的下一步动作集合。这个公式表达的是：每一层生成多个下一步，再用过程分数剪枝。

标注侧要同时看自动标注覆盖率和人工标注成本：

```math
C_{\mathrm{auto}}=
\frac{
\sum_i\sum_j a_{ij}
}{
\sum_i M_i
}
```

```math
C_{\mathrm{label}}=
\sum_i\sum_j(1-a_{ij})c_{ij}
```

其中 `a_ij=1` 表示第 `i` 个样本第 `j` 步可由程序、规则、测试或工具自动标注，`c_ij` 是需要人工复核时的成本。

一个上线前过程监督门禁可以写成：

```math
G_{\mathrm{proc}}=
\mathbb{1}[
A_{\mathrm{step}}\ge \alpha
\land
A_{\mathrm{first}}\ge \beta
\land
C_{\mathrm{auto}}\ge \gamma
\land
A_{\mathrm{search}}\ge \eta
\land
C_{\mathrm{label}}\le C_{\max}
]
```

面试里要强调：`A_out`、`A_step`、`A_first`、`A_search` 和 `C_label` 不能互相替代。最终答案正确但过程错，说明 outcome supervision 有盲区；过程正确但最终格式错，说明系统还需要答案规范化或格式 verifier。

## 5.2 Outcome Supervision 的局限

只看最终答案有几个问题：

1. 不知道哪一步错。
2. 正确中间步骤没有被奖励。
3. 错误中间步骤可能被忽略。
4. 最终答案碰巧对时，错误过程可能被当成好样本。
5. 对长推理任务训练信号太稀疏。

例如数学题中，模型可能因为计算错一个符号导致最终答案错。如果只看最终答案，整条推理都被判错；但实际上前面题意理解和公式选择可能是正确的。

## 5.3 什么是推理步骤

过程监督首先要定义“步骤”。

一个步骤可以是：

1. 提取条件。
2. 写出公式。
3. 代入数值。
4. 做一次计算。
5. 得到中间结论。
6. 检查答案。

示例：

```text
Step 1: 净速度 = 3 - 1 = 2。
Step 2: 总容量是 20 升。
Step 3: 时间 = 20 / 2 = 10。
```

步骤粒度太粗，定位不了错误；粒度太细，标注成本太高。合理粒度通常是一个可检查的逻辑单元。

## 5.4 过程标注格式

一种简单格式：

```json
{
  "question": "...",
  "steps": [
    {"text": "净速度 = 3 - 1 = 2", "label": "correct"},
    {"text": "时间 = 20 / 2 = 10", "label": "correct"}
  ],
  "answer": "10"
}
```

也可以用分数：

```json
{"step": "...", "score": 1.0}
```

或者标注错误类型：

```json
{"step": "...", "label": "wrong", "error_type": "calculation_error"}
```

错误类型有助于后续分析和数据改进。

## 5.5 PRM：Process Reward Model

PRM 是给推理步骤或中间状态打分的模型。

输入：

```text
question + previous steps + current step
```

输出：

```text
当前步骤正确概率或分数
```

PRM 可以用于：

1. 训练模型偏好正确步骤。
2. 在搜索中给中间节点打分。
3. 早停错误路径。
4. 分析模型哪里出错。

PRM 和 ORM 的区别：PRM 评价过程，ORM 评价结果。

## 5.6 正确步骤和有用步骤

过程监督中要区分两件事：

1. 这一步是否正确。
2. 这一步是否有助于解题。

有些步骤正确但无关：

```text
题目说有 3 个苹果。苹果是一种水果。
```

这句话正确，但对解题没帮助。

高质量过程监督应该鼓励：

1. 正确。
2. 相关。
3. 必要。
4. 不冗余。

否则模型可能学会生成很多正确但无用的废话。

## 5.7 错误步骤定位

一个好的 process supervision 数据集应该能定位第一处错误。

例如：

```text
Step 1: 正确
Step 2: 正确
Step 3: 错误
Step 4: 因为 step 3 错而继续错
```

训练时要特别关注第一处错误，因为后面的错误可能只是连锁反应。

错误类型：

1. 题意理解错。
2. 条件提取错。
3. 公式选择错。
4. 计算错。
5. 单位错。
6. 逻辑跳步。
7. 引入不存在条件。

## 5.8 标注成本

过程监督最大问题是标注贵。

标最终答案相对简单；标每一步需要标注者理解题目和推理过程。

成本来自：

1. 步骤切分。
2. 判断每步正确性。
3. 标注错误类型。
4. 多种正确解法。
5. 领域知识要求。

降低成本的方法：

1. 用规则或工具自动校验部分步骤。
2. 用 LLM 初标，人类复核。
3. 只标注关键步骤。
4. 重点标 hard cases。
5. 使用程序化任务，例如代码测试。

## 5.9 自动过程监督

某些任务可以自动获得过程监督。

例如代码任务：

1. 每次修改后运行测试。
2. 编译错误定位到某个步骤。
3. 单元测试反馈是否通过。

数学任务中也可以用：

1. Python 计算器。
2. 符号计算。
3. 单位检查。
4. 公式规则。

但自动监督覆盖有限。自然语言推理步骤是否“合理”很难完全规则化。

## 5.10 Process Supervision 与 Search

搜索需要中间节点评分，PRM 很适合。

流程：

1. 模型生成多个下一步。
2. PRM 给每个下一步打分。
3. 保留高分步骤。
4. 继续展开。

相比只用最终答案评分，PRM 可以更早剪掉错误路径，节省计算。

风险：如果 PRM 错误地低估了正确路径，搜索会错过好解。

## 5.11 Process Supervision 与 RL

过程奖励可以用于强化学习。

普通 outcome reward：只在最终答案给奖励。

过程 reward：每个步骤都可能给奖励。

优点：

1. 奖励更密集。
2. Credit assignment 更容易。
3. 对长推理任务更友好。

挑战：

1. PRM 质量决定训练方向。
2. 模型可能优化表面步骤。
3. 奖励 hacking 风险。

## 5.12 过程监督的风险

风险包括：

1. 模型学会写“看起来正确”的步骤。
2. 步骤格式僵化。
3. 标注者偏好影响推理风格。
4. 多种正确解法被误判。
5. PRM 被 reward hacking。
6. 长步骤导致成本上升。

因此过程监督不能只看步骤评分，还要看最终任务正确率和泛化能力。

## 5.13 过程监督和可解释性

过程监督会让模型输出更结构化的推理过程，看起来更可解释。

但注意：

1. 可读步骤不一定是真实内部机制。
2. 步骤正确不一定代表模型真正理解。
3. 解释可能只是训练出的格式。

过程监督能提升可检查性，但不能完全解决可解释性问题。

## 5.14 数据集设计建议

设计 process supervision 数据集时，要覆盖：

1. 简单题和复杂题。
2. 多种解法。
3. 常见错误类型。
4. 迷惑性错误步骤。
5. 正确但冗余步骤。
6. 最终答案正确但过程错误的样本。
7. 过程正确但最后格式错误的样本。

数据越贴近模型真实错误分布，PRM 越有用。

## 5.15 评估 Process Supervision

评估维度：

1. Step classification accuracy。
2. First-error detection accuracy。
3. PRM ranking accuracy。
4. Search 后最终准确率。
5. 成本和延迟。
6. 对 hard cases 的提升。

最终还是要看：使用过程监督后，reasoning 系统是否在真实任务上更准确、更稳定。

## 5.16 最小可运行 process supervision / PRM 审计 demo

这个 demo 不训练真实 PRM，而是模拟一张上线前审计表：同一批 toy reasoning 样本同时记录最终答案、步骤标签、PRM 预测、自动标注覆盖、人工标注成本和搜索选择结果。它刻意保留两个 bad case：`lucky_answer` 是最终答案碰巧正确但过程错误，`format_error` 是过程正确但最终格式不合格。

输入：

1. toy 样本的最终答案、标准答案和步骤标签。
2. 每一步的 PRM 预测、是否相关、是否可自动标注和人工成本。
3. toy 搜索状态的候选动作与 PRM 选择。

输出：

1. outcome accuracy、step accuracy、first-error accuracy。
2. relevant step ratio、auto label coverage、human label cost。
3. outcome supervision 盲区样本、冗余步骤和 search failure。
4. process supervision gate。

```python
def first_error(labels):
    for idx, label in enumerate(labels, start=1):
        if label == 0:
            return idx
    return None


cases = [
    {
        "id": "clean_solve",
        "gold": "10",
        "answer": "10",
        "steps": [
            {"id": "extract_rate", "q": 1, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "net_rate", "q": 1, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "volume", "q": 1, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "time_formula", "q": 1, "pred": 1, "relevant": 1, "auto": 0, "cost": 2},
            {"id": "compute_time", "q": 1, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "final_answer", "q": 1, "pred": 1, "relevant": 1, "auto": 0, "cost": 2},
        ],
    },
    {
        "id": "lucky_answer",
        "gold": "42",
        "answer": "42",
        "steps": [
            {"id": "read_question", "q": 1, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "wrong_formula", "q": 0, "pred": 0, "relevant": 1, "auto": 0, "cost": 3},
            {"id": "bad_substitution", "q": 0, "pred": 1, "relevant": 1, "auto": 0, "cost": 0},
            {"id": "redundant_plan", "q": 1, "pred": 1, "relevant": 0, "auto": 0, "cost": 2},
            {"id": "lucky_arithmetic", "q": 0, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "unsupported_finish", "q": 0, "pred": 1, "relevant": 1, "auto": 0, "cost": 2},
        ],
    },
    {
        "id": "format_error",
        "gold": "6",
        "answer": "six hours",
        "steps": [
            {"id": "extract_distance", "q": 1, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "extract_speed", "q": 1, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "divide", "q": 1, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "unit_check", "q": 1, "pred": 1, "relevant": 1, "auto": 0, "cost": 3},
            {"id": "numeric_result", "q": 1, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "format_instruction", "q": 1, "pred": 1, "relevant": 1, "auto": 0, "cost": 2},
        ],
    },
]

search_states = [
    {
        "id": "distractor_state",
        "choices": [
            {"name": "follow distractor", "score": 0.22, "good": False},
            {"name": "ignore distractor", "score": 0.81, "good": True},
        ],
    },
    {
        "id": "logic_state",
        "choices": [
            {"name": "branch_red", "score": 0.45, "good": False},
            {"name": "branch_blue", "score": 0.76, "good": True},
        ],
    },
    {
        "id": "hard_negative_state",
        "choices": [
            {"name": "plain_correct", "score": 0.62, "good": True},
            {"name": "polished_wrong", "score": 0.84, "good": False},
        ],
    },
]

outcome_correct = [case["answer"] == case["gold"] for case in cases]
all_steps = [step for case in cases for step in case["steps"]]

step_matches = [step["q"] == step["pred"] for step in all_steps]
outcome_accuracy = round(sum(outcome_correct) / len(cases), 3)
step_accuracy = round(sum(step_matches) / len(step_matches), 3)

true_first_errors = [first_error([step["q"] for step in case["steps"]]) for case in cases]
pred_first_errors = [first_error([step["pred"] for step in case["steps"]]) for case in cases]
first_error_accuracy = round(
    sum(t == p for t, p in zip(true_first_errors, pred_first_errors)) / len(cases),
    3,
)

relevant_step_ratio = round(sum(step["relevant"] for step in all_steps) / len(all_steps), 3)
auto_label_coverage = round(sum(step["auto"] for step in all_steps) / len(all_steps), 3)
human_label_cost = sum(0 if step["auto"] else step["cost"] for step in all_steps)

final_correct_bad_process = [
    case["id"]
    for case in cases
    if case["answer"] == case["gold"] and any(step["q"] == 0 for step in case["steps"])
]
process_correct_bad_final = [
    case["id"]
    for case in cases
    if case["answer"] != case["gold"] and all(step["q"] == 1 for step in case["steps"])
]
redundant_step_ids = [step["id"] for step in all_steps if step["relevant"] == 0]

search_choices = []
for state in search_states:
    choice = max(state["choices"], key=lambda item: item["score"])
    search_choices.append({"id": state["id"], "choice": choice["name"], "good": choice["good"]})

search_top1_accuracy = round(
    sum(choice["good"] for choice in search_choices) / len(search_choices),
    3,
)
search_failures = [choice["id"] for choice in search_choices if not choice["good"]]

gates = {
    "step_quality_ok": step_accuracy >= 0.8,
    "first_error_ok": first_error_accuracy >= 0.9,
    "auto_coverage_ok": auto_label_coverage >= 0.5,
    "search_ready": search_top1_accuracy >= 0.8,
    "cost_ok": human_label_cost <= 20,
    "no_outcome_blind_spot": not final_correct_bad_process and not process_correct_bad_final,
}
gate_pass = all(gates.values())

print(f"outcome_accuracy={outcome_accuracy}")
print(f"step_accuracy={step_accuracy}")
print(f"first_error_accuracy={first_error_accuracy}")
print(f"relevant_step_ratio={relevant_step_ratio}")
print(f"auto_label_coverage={auto_label_coverage}")
print(f"human_label_cost={human_label_cost}")
print(f"final_correct_bad_process={final_correct_bad_process}")
print(f"process_correct_bad_final={process_correct_bad_final}")
print(f"redundant_step_ids={redundant_step_ids}")
print(f"search_top1_accuracy={search_top1_accuracy}")
print(f"search_choices={search_choices}")
print(f"search_failures={search_failures}")
print(f"gates={gates}")
print(f"gate_pass={gate_pass}")
```

预期输出：

```text
outcome_accuracy=0.667
step_accuracy=0.833
first_error_accuracy=1.0
relevant_step_ratio=0.944
auto_label_coverage=0.556
human_label_cost=16
final_correct_bad_process=['lucky_answer']
process_correct_bad_final=['format_error']
redundant_step_ids=['redundant_plan']
search_top1_accuracy=0.667
search_choices=[{'id': 'distractor_state', 'choice': 'ignore distractor', 'good': True}, {'id': 'logic_state', 'choice': 'branch_blue', 'good': True}, {'id': 'hard_negative_state', 'choice': 'polished_wrong', 'good': False}]
search_failures=['hard_negative_state']
gates={'step_quality_ok': True, 'first_error_ok': True, 'auto_coverage_ok': True, 'search_ready': False, 'cost_ok': True, 'no_outcome_blind_spot': False}
gate_pass=False
```

这个输出刻意让 `gate_pass=False`。它不是 demo 失败，而是在说明 process supervision 的真实工程判断：步骤质量和第一处错误检测看起来不错，但 outcome supervision 有盲区，PRM 搜索也会被 polished hard negative 误导。上线前不能只看 `A_step`，还要补 hard negative、格式 verifier、搜索切片和人工复核策略。

## 5.17 面试官会怎么问

### 问题一：Process supervision 是什么？

回答模板：

```text
Process supervision 是对推理中间步骤进行监督，而不是只监督最终答案。它可以标注每一步是否正确，训练 process reward model，用于错误定位、搜索剪枝和更细粒度的 reasoning 训练。
```

### 问题二：它和 outcome supervision 有什么区别？

回答模板：

```text
Outcome supervision 只看最终答案对不对，训练信号稀疏，无法定位中间错误。Process supervision 会检查每一步推理，能提供更密集的反馈，但标注成本更高，步骤粒度也更难定义。
```

### 问题三：PRM 有什么作用？

回答模板：

```text
PRM 是 process reward model，用来给推理步骤或中间状态打分。它可以用于搜索时选择更好的下一步，也可以用于训练模型偏好正确推理过程。相比 ORM，PRM 更适合中间过程评估。
```

### 问题四：过程监督为什么成本高？

回答模板：

```text
因为标注者不仅要知道最终答案，还要理解每一步推理是否正确、是否相关、错误发生在哪里，以及错误类型是什么。多种正确解法和不同步骤粒度也会增加标注难度。
```

### 问题五：过程监督有什么风险？

回答模板：

```text
模型可能学会生成格式漂亮但不真实的推理步骤，也可能被标注风格限制，或者 reward hacking PRM。过程正确性也不等于真实内部机制。因此要同时评估最终准确率、步骤正确性和泛化能力。
```

## 5.18 小练习

1. 给一道数学题写出逐步标注格式。
2. 标出一条错误 CoT 中第一处错误。
3. 设计 5 类推理错误类型。
4. 构造一个最终答案正确但过程错误的样本。
5. 设计一个 PRM 输入输出格式。
6. 比较 PRM 和 ORM 在搜索中的作用。
7. 设计一个 process supervision 数据质量 checklist。
8. 运行本章 demo，并新增一个“最终答案正确但第一步读题错误”的 hard case，观察 outcome accuracy、first-error accuracy 和 gate 如何变化。

## 5.19 本章总结

Process supervision 通过监督中间步骤，为 reasoning model 提供比最终答案更细粒度的训练信号。它能帮助错误定位、训练 PRM、支持搜索和提升复杂任务表现，但也带来高标注成本、步骤粒度争议和 reward hacking 风险。

需要记住：

1. Outcome supervision 看最终答案。
2. Process supervision 看中间步骤。
3. PRM 给步骤或中间状态打分。
4. 第一处错误定位非常重要。
5. 过程监督适合 search 和长推理任务。
6. 标注成本和标注质量是最大瓶颈。
7. 最终要看下游 accuracy 是否提升。
8. 工程上必须同时看 outcome、step、first-error、search、标注成本和 hard negative。

下一章会进入 search 与 Tree-of-Thought，讲清如何把推理建模成搜索、如何展开候选、如何用 verifier 剪枝，以及 MCTS 等方法的直觉。
