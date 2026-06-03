# 第八部分：Reasoning Model 与 Test-Time Compute

## 第 86 讲：Chain-of-Thought 的机制与争议

### 本讲目标

学完本讲，你应该能回答六个问题：

1. Chain-of-Thought 想解决什么问题。
2. 为什么让模型写中间推理步骤可能提升复杂任务表现。
3. CoT、scratchpad、rationale、hidden reasoning 有什么区别。
4. CoT 为什么存在忠实性、可解释性和安全争议。
5. 如何评估 CoT 是否真的带来推理能力提升。
6. 面试中如何回答“CoT 是真实推理还是语言模式模仿”。

从这一讲开始，我们进入第八部分：Reasoning Model 与 Test-Time Compute。

前面第七部分讲长上下文、RAG 和 Agent。

这些能力让模型能读更多资料、检索外部知识、调用工具和执行多步任务。

但还有一类问题，不只是“有没有信息”，而是：

```text
模型能不能在已有信息上做多步推理。
```

例如：

1. 数学应用题。
2. 逻辑推理题。
3. 代码执行和 bug 定位。
4. 多跳问答。
5. 复杂规划。
6. 科学假设分析。

Chain-of-Thought，简称 CoT，就是最早被广泛讨论的推理增强方法之一。

它的基本想法很简单：

```text
不要让模型直接给答案，而是让模型先写出中间推理步骤，再给最终答案。
```

#### 来龙去脉：CoT 为什么成为 reasoning 的入口

CoT 不是凭空出现的。早期大模型已经能做很多问答和生成任务，但在数学应用题、符号推理、常识推理这类需要多步中间状态的问题上，直接回答很容易跳步。

Chain-of-Thought Prompting 论文把 few-shot 示例中的中间推理过程显式写出来，观察到大模型在多类复杂推理任务上明显受益。Zero-shot CoT 进一步说明，有时一句“Let's think step by step”也能触发模型先生成中间步骤再给答案。

后来争议也随之出现：这些中间步骤到底是模型真实使用的原因，还是生成答案后的合理化解释？一些 CoT faithfulness 研究用干预、提示偏置和反事实修改来测试模型是否真的依赖自己写出的推理。结论更谨慎：CoT 可以作为有用的中间计算和调试信号，但不能自动等同于可解释性证明。

所以面试里最好把 CoT 放在这条演进线上讲：

```text
direct answer -> CoT -> self-consistency -> verifier/search/tool feedback -> reasoning system
```

它是 reasoning model 的入口，但不是终点。

---

### 一、为什么需要 Chain-of-Thought

普通 prompting 常常要求模型直接回答。

例如：

```text
小明有 3 个苹果，又买了 5 个，吃掉 2 个，还剩几个？
```

直接回答是：

```text
6 个。
```

但对更复杂问题，直接输出答案容易错。

原因是模型需要隐式完成多个中间步骤：

1. 理解题意。
2. 找出变量。
3. 分解子问题。
4. 执行计算。
5. 检查约束。
6. 合成答案。

如果只要求最终答案，模型可能跳步。

CoT 的直觉是：

```text
把隐式推理过程显式写出来，让模型有更多 token 表达中间状态，从而降低一步到位的难度。
```

这类似人类做题时写草稿。

不是因为草稿本身神奇，而是草稿提供了中间表示和检查点。

---

### 二、CoT 的基本形式

最简单的 CoT prompt 是：

```text
Let's think step by step.
```

中文可以写成：

```text
我们一步一步分析。
```

例如：

```text
问题：一个班有 24 人，其中 1/3 是女生，女生有多少人？

推理：总人数是 24，女生占 1/3，所以女生人数是 24 * 1/3 = 8。
答案：8 人。
```

CoT 输出通常包含：

1. 问题重述。
2. 条件提取。
3. 分步计算。
4. 中间结论。
5. 最终答案。

在实际产品中，不一定要把完整 CoT 展示给用户。

可以只让模型内部使用中间推理，再输出简洁答案和必要解释。

---

### 三、为什么 CoT 可能有效

CoT 有几个可能有效的原因。

#### 1. 增加计算 token

自回归模型每生成一个 token 都是在做一次条件计算。

让模型生成中间步骤，相当于给模型更多 test-time compute。

这和第 90 讲要讲的 test-time compute scaling 有直接关系。

#### 2. 降低任务难度

复杂问题被拆成多个简单步骤。

模型不用一步从问题跳到答案。

#### 3. 提供中间状态

中间推理文本可以作为后续 token 的上下文。

例如前面算出 `24 * 1/3 = 8`，后面就可以基于 8 继续推理。

#### 4. 激活训练中见过的解题模式

训练数据中有大量教程、题解、证明、代码注释和解题过程。

CoT prompt 可能激活这些模式。

#### 5. 便于自我检查

有中间步骤后，模型或外部 verifier 更容易检查哪一步错。

可以把 direct answer 和 CoT 看成两种推理策略 `m`。在评估集 `D` 上，最终答案准确率为：

```math
A_m =
\frac{1}{|D|}
\sum_{(x_i,y_i)\in D}
\mathbf{1}[\hat y_i^{(m)} = y_i]
```

CoT 的收益不是看推理文字是否更长，而是看：

```math
\Delta A_{\mathrm{CoT}} = A_{\mathrm{CoT}} - A_{\mathrm{direct}}
```

如果要考虑成本，可以定义一个简单效用：

```math
U_m =
A_m
-
\lambda_n \bar n_m
-
\lambda_t \bar T_m
-
\lambda_r R_m
```

其中 `\bar n_m` 是平均输出 token 数，`\bar T_m` 是平均延迟，`R_m` 是安全或误导风险率。这个公式的含义是：CoT 是否值得上线，要同时看准确率提升、token 成本、延迟和风险，而不是只看它“看起来更会思考”。

但这些解释只是可能机制。

CoT 到底是不是“真实推理”，仍有争议。

---

### 四、CoT、Scratchpad、Rationale 的区别

几个术语容易混。

#### 1. Chain-of-Thought

CoT 通常指模型生成自然语言中间推理步骤。

重点是分步推理。

#### 2. Scratchpad

Scratchpad 是草稿空间。

它可以是自然语言、公式、表格、代码或中间变量。

Scratchpad 不一定给用户看。

#### 3. Rationale

Rationale 是解释或理由。

它可能是模型真实使用的推理过程，也可能只是事后解释。

#### 4. Hidden reasoning

Hidden reasoning 是模型内部或系统内部的推理过程，不直接展示给用户。

它可以用于提升答案质量，同时避免暴露冗长、不稳定或不安全的中间文本。

可以粗略理解为：

```text
CoT = 分步推理文本
scratchpad = 草稿空间
rationale = 给出的理由或解释
hidden reasoning = 不展示的内部推理
```

面试中要小心：

```text
模型输出的 rationale 不一定等于模型真实决策原因。
```

---

### 五、Few-shot CoT 与 Zero-shot CoT

CoT 常见有两种用法。

#### 1. Few-shot CoT

Prompt 中给几个带推理步骤的示例。

例如：

```text
Q: 小红有 2 支笔，又买了 3 支，一共有几支？
A: 她原来有 2 支，又买了 3 支，所以 2 + 3 = 5。答案是 5。

Q: 小明有 10 元，花了 4 元，还剩多少？
A: 他原来有 10 元，花掉 4 元，所以 10 - 4 = 6。答案是 6。

Q: ...
```

Few-shot CoT 的作用是给模型展示解题格式和推理模式。

#### 2. Zero-shot CoT

不提供示例，只加一句：

```text
Let's think step by step.
```

Zero-shot CoT 简单，但不一定稳定。

复杂任务中，示例质量、任务相似性和输出格式约束都很重要。

---

### 六、CoT 的适用场景

CoT 更适合需要多步推理的任务。

例如：

1. 数学题。
2. 符号推理。
3. 逻辑判断。
4. 多跳问答。
5. 复杂条件分析。
6. 代码 reasoning。
7. 规划任务。

CoT 不一定适合所有任务。

例如：

1. 简单事实问答。
2. 翻译。
3. 分类。
4. 情感判断。
5. 简单抽取。

对于简单任务，CoT 可能增加成本，还可能引入多余错误。

一个实用判断是：

```text
如果任务需要多个中间步骤、约束检查或计算，CoT 可能有帮助；如果任务本身很简单，直接回答更好。
```

---

### 七、CoT 的忠实性争议

CoT 最大争议之一是 faithful reasoning。

问题是：

```text
模型写出来的推理步骤，是否真的是它得到答案的原因？
```

有时模型先“知道”答案，再编一段看起来合理的解释。

这叫 post-hoc rationalization。

例如模型给出正确答案，但推理步骤中有错误计算。

或者推理步骤看起来合理，最终答案却不匹配。

这说明：

```text
CoT 可以提高可读性，但不必然等于真实可解释性。
```

评估 CoT 忠实性可以看：

1. 修改中间步骤是否会改变答案。
2. 中间步骤是否可被 verifier 检查。
3. 推理步骤和最终答案是否一致。
4. 模型是否会隐藏关键依据。
5. 生成的理由是否只是模板化解释。

面试中不要把 CoT 简单说成“模型可解释性解决方案”。

更准确说法是：

```text
CoT 提供了可检查的中间文本，但它的忠实性需要额外验证。
```

---

### 八、CoT 的安全争议

CoT 还涉及安全问题。

#### 1. 暴露不稳定推理

模型中间推理可能包含错误假设、偏见或敏感内容。

直接展示给用户会降低可信度。

#### 2. 暴露策略细节

在安全、风控、审核场景中，完整推理可能泄露检测规则。

攻击者可以据此绕过系统。

#### 3. 增加越狱攻击面

用户可能诱导模型暴露内部推理、系统提示或安全策略。

#### 4. 误导用户

流畅的 CoT 可能让错误答案看起来更可信。

#### 5. 训练数据风险

如果用不可靠 CoT 数据训练模型，可能强化错误推理模板。

因此很多系统会选择：

```text
内部使用 reasoning，外部只展示简洁解释或答案依据。
```

这不是否定 CoT，而是区分内部计算和用户可见解释。

---

### 九、CoT 和 Verifier

CoT 的一个重要价值是方便验证。

如果模型只输出答案：

```text
答案：42
```

很难判断错在哪里。

如果模型输出步骤：

```text
第一步...
第二步...
第三步...
答案：42
```

就可以让 verifier 检查每一步。

Verifier 可以是：

1. 规则程序。
2. 单元测试。
3. 数学检查器。
4. 另一个模型。
5. 人工标注。

第 88 讲会详细讲 verifier、process reward model 和 outcome reward model。

本讲先记住一个核心关系：

```text
CoT 生成中间过程，verifier 检查中间过程或最终结果。
```

Reasoning model 的很多后续方法都围绕这条线展开。

---

### 十、CoT 和 Self-Consistency

单条 CoT 可能错。

一个自然想法是：

```text
让模型采样多条推理路径，再选择多数答案或最可信答案。
```

这就是 self-consistency 的核心直觉。

例如同一道数学题采样 10 条 CoT。

如果 7 条得到答案 A，2 条得到答案 B，1 条得到答案 C。

可以选择 A。

这说明 CoT 不只是 prompt 技巧，也可以和采样、搜索、验证结合。

第 87 讲会专门讲 self-consistency。

---

### 十一、CoT 的评估方法

评估 CoT 不能只看推理文字是否漂亮。

要看几个层面。

#### 1. Final answer accuracy

最终答案是否正确。

#### 2. Reasoning step validity

中间步骤是否正确。

#### 3. Faithfulness

中间步骤是否真的影响最终答案。

#### 4. Robustness

换 prompt、换示例、换表达后是否稳定。

#### 5. Efficiency

CoT 增加了多少 token 和延迟，收益是否值得。

#### 6. Error localization

能否通过中间步骤定位错误。

#### 7. Safety

是否泄露敏感推理、策略或有害内容。

可以把过程评估写成几个指标。

最终答案准确率：

```math
A =
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[\hat y_i = y_i]
```

步骤有效率：

```math
V_{\mathrm{step}} =
\frac{\sum_i \sum_{j=1}^{M_i} \mathbf{1}[z_{ij}=1]}
{\sum_i M_i}
```

其中 `M_i` 是第 `i` 个样本的步骤数，`z_{ij}=1` 表示第 `j` 步经规则、工具、人工或 process verifier 判断为有效。

最终答案和过程不一致的比例：

```math
I_{\mathrm{bad}} =
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[\hat y_i = y_i]\mathbf{1}[v_i=0]
```

这里 `v_i=0` 表示该样本至少有一个关键步骤无效。这个指标用来捕捉“答案碰巧对，但推理过程错”的情况。

反事实干预敏感度可以写成：

```math
S_{\mathrm{int}} =
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[
f(x_i,r_i)
\ne
f(x_i,\tilde r_i)
]
```

其中 `r_i` 是原始推理，`\tilde r_i` 是修改关键中间步骤后的推理。如果修改关键步骤后答案完全不变，要警惕这条 CoT 可能只是事后解释；但 `S_int` 也不能单独证明忠实性，因为模型可能对无关扰动过敏。

成本指标可以写成：

```math
C_m =
c_{\mathrm{in}}\bar n_{\mathrm{in}}
+
c_{\mathrm{out}}\bar n_{\mathrm{out}}^{(m)}
+
c_v \bar k_v
```

其中 `c_in`、`c_out` 是输入/输出 token 单价或单位成本，`\bar k_v` 是平均 verifier 调用次数。CoT、self-consistency 和 verifier 都会把 `C_m` 往上推，所以评估表必须报告预算。

一个好的评估应该比较：

```text
direct answer vs CoT vs CoT + self-consistency vs CoT + verifier
```

同时看准确率、成本、延迟和安全风险。

---

### 十二、最小代码：CoT 输出审计与成本对比

下面的 demo 不调用模型，只审计一批模拟输出，展示五件事：

1. direct answer 和 CoT 都要抽取最终答案再算 accuracy。
2. CoT 需要检查中间算式是否有效。
3. “最终答案正确”不代表推理步骤正确。
4. 修改关键推理后答案是否变化，可以作为忠实性干预线索。
5. CoT 的平均输出 token 数更高，收益要和成本一起看。

```python
import operator
import re


STEP_RE = re.compile(r"(-?\d+)\s*([+\-*/])\s*(-?\d+)\s*=\s*(-?\d+)")
FINAL_RE = re.compile(r"(?:final|answer|答案)[:：]\s*([a-zA-Z0-9_\-]+)", re.I)
OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


def normalize(text):
    return str(text).strip().lower()


def token_count(text):
    return len(re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]+|[+\-*/=]", text))


def extract_answer(text):
    match = FINAL_RE.search(text)
    if match:
        return normalize(match.group(1))
    numbers = re.findall(r"-?\d+", text)
    if numbers:
        return normalize(numbers[-1])
    labels = re.findall(r"\b(positive|negative|true|false)\b", text.lower())
    return labels[-1] if labels else ""


def verify_steps(text):
    checks = []
    for left, op, right, stated in STEP_RE.findall(text):
        lhs = OPS[op](int(left), int(right))
        checks.append(abs(lhs - int(stated)) < 1e-9)
    return checks


records = [
    {
        "id": "apple",
        "gold": "6",
        "direct": "6",
        "cot": "3 + 5 = 8; 8 - 2 = 6; final: 6",
        "cot_perturbed": "3 + 5 = 9; 9 - 2 = 7; final: 7",
    },
    {
        "id": "classroom",
        "gold": "8",
        "direct": "6",
        "cot": "24 / 3 = 8; final: 8",
        "cot_perturbed": "24 / 4 = 6; final: 6",
    },
    {
        "id": "lucky_answer",
        "gold": "13",
        "direct": "13",
        "cot": "7 + 6 = 12; final: 13",
        "cot_perturbed": "7 + 6 = 12; final: 13",
    },
    {
        "id": "simple_sentiment",
        "gold": "positive",
        "direct": "positive",
        "cot": "The sentence says the user likes the product. final: positive",
        "cot_perturbed": "The sentence says the user dislikes the product. final: negative",
    },
]


def accuracy(records, field):
    correct = 0
    for row in records:
        correct += extract_answer(row[field]) == normalize(row["gold"])
    return correct / len(records)


def cot_audit(records):
    step_checks = []
    invalid_but_correct = 0
    changed_by_intervention = 0
    for row in records:
        checks = verify_steps(row["cot"])
        step_checks.extend(checks)
        cot_correct = extract_answer(row["cot"]) == normalize(row["gold"])
        if checks and cot_correct and not all(checks):
            invalid_but_correct += 1
        if extract_answer(row["cot"]) != extract_answer(row["cot_perturbed"]):
            changed_by_intervention += 1

    avg_direct_tokens = sum(token_count(row["direct"]) for row in records) / len(records)
    avg_cot_tokens = sum(token_count(row["cot"]) for row in records) / len(records)
    invalid_rate = 1 - (sum(step_checks) / len(step_checks))
    utility = accuracy(records, "cot") - 0.02 * avg_cot_tokens - 0.20 * invalid_rate
    return {
        "direct_accuracy": accuracy(records, "direct"),
        "cot_accuracy": accuracy(records, "cot"),
        "step_validity": sum(step_checks) / len(step_checks),
        "invalid_but_correct": invalid_but_correct,
        "intervention_sensitivity": changed_by_intervention / len(records),
        "avg_direct_tokens": avg_direct_tokens,
        "avg_cot_tokens": avg_cot_tokens,
        "toy_cot_utility": utility,
    }


print({key: round(value, 3) for key, value in cot_audit(records).items()})
for row in records:
    print(
        row["id"],
        "direct=", extract_answer(row["direct"]),
        "cot=", extract_answer(row["cot"]),
        "steps=", verify_steps(row["cot"]),
    )
```

一组输出如下：

```text
{'direct_accuracy': 0.75, 'cot_accuracy': 1.0, 'step_validity': 0.75, 'invalid_but_correct': 1, 'intervention_sensitivity': 0.75, 'avg_direct_tokens': 1.0, 'avg_cot_tokens': 9.0, 'toy_cot_utility': 0.77}
apple direct= 6 cot= 6 steps= [True, True]
classroom direct= 6 cot= 8 steps= [True]
lucky_answer direct= 13 cot= 13 steps= [False]
simple_sentiment direct= positive cot= positive steps= []
```

这个 demo 里 CoT 的最终准确率更高，但也暴露了一个风险：`lucky_answer` 的答案正确，步骤却是错的。真实评估中如果只看 final answer，就会把这种样本当成成功；如果要训练 process verifier 或向用户展示解释，就必须额外检查步骤有效性和干预敏感度。

---

### 十三、真实项目中的坑

#### 1. 所有任务都强制 CoT

简单任务强制分步，会增加成本并引入噪声。

#### 2. 只看 CoT 是否流畅

流畅推理不等于正确推理。

#### 3. 把 CoT 当可靠解释

模型可能事后编理由。

#### 4. 直接展示完整内部推理

可能泄露策略、增加误导和安全风险。

#### 5. 示例质量差

Few-shot CoT 示例如果有错误，模型会模仿错误模式。

#### 6. 不做成本评估

CoT 增加输出 token，延迟和费用都会上升。

#### 7. 忽略 final answer extraction

有些模型推理写对了，但最终答案格式不稳定。

生产系统要明确最终答案字段。

---

### 十四、面试问答

#### 问题 1：Chain-of-Thought 的核心思想是什么？

可以这样回答：

```text
CoT 让模型在给最终答案前生成中间推理步骤，把复杂任务分解成多个简单步骤，并用生成的中间文本作为后续推理上下文，从而提升多步推理任务表现。
```

#### 问题 2：为什么 CoT 能提升推理效果？

可以这样回答：

```text
可能原因包括增加 test-time compute、降低一步到位的难度、提供中间状态、激活训练数据中的解题模式，并方便后续 verifier 检查。但它不是所有任务都有效。
```

#### 问题 3：CoT 是真实推理还是模式模仿？

可以这样回答：

```text
不能简单二选一。CoT 可能确实提供了有用的中间计算，也可能包含训练数据中学到的解题模板或事后解释。关键要通过答案准确率、步骤有效性、忠实性和干预实验来判断。
```

#### 问题 4：CoT 有什么争议？

可以这样回答：

```text
主要争议是忠实性和安全性。模型写出的推理不一定是得到答案的真实原因，可能是事后合理化；完整展示 CoT 还可能暴露错误假设、敏感策略或增加攻击面。
```

#### 问题 5：CoT 和 verifier 有什么关系？

可以这样回答：

```text
CoT 生成中间推理过程，verifier 可以检查中间步骤或最终答案。只有 CoT 没有验证仍可能流畅但错误，结合 verifier 可以提高可靠性。
```

#### 问题 6：生产系统应该展示完整 CoT 吗？

可以这样回答：

```text
不一定。生产系统可以内部使用 reasoning 来提升质量，但对用户展示简洁解释、证据或关键步骤。完整 CoT 可能冗长、不稳定，并带来安全和误导风险。
```

---

### 十五、常见误区

1. 误区：CoT 一定提高所有任务表现。
   纠正：CoT 主要适合多步推理，简单任务可能没收益甚至变差。

2. 误区：写得越长越会推理。
   纠正：长推理可能只是冗余或错误累积。

3. 误区：CoT 就是可解释性。
   纠正：CoT 是可见中间文本，不保证忠实反映模型真实原因。

4. 误区：最终答案对，CoT 就一定对。
   纠正：答案可能碰巧对，中间步骤可能错。

5. 误区：完整展示 CoT 最透明。
   纠正：完整展示可能带来安全、隐私和误导风险。

6. 误区：CoT 只是 prompt 技巧。
   纠正：CoT 也连接 test-time compute、self-consistency、verifier 和 reasoning model 训练。

---

### 十六、小练习

1. 给一个数学题分别写 direct answer prompt 和 CoT prompt。
2. 构造一个 CoT 中间步骤错误但最终答案正确的例子。
3. 构造一个 CoT 看起来流畅但最终答案错误的例子。
4. 比较 CoT、scratchpad、rationale 和 hidden reasoning。
5. 设计一个评估 CoT 忠实性的实验。
6. 设计一个 CoT + verifier 的数学题检查流程。
7. 分析为什么简单分类任务不一定需要 CoT。
8. 设计一个生产系统中“内部推理、外部简洁解释”的输出格式。
9. 比较 direct answer、CoT、CoT + self-consistency 的成本和收益。
10. 用 3 分钟回答：“CoT 是真实推理还是语言模式模仿？”

### 本讲总结

本讲最重要的结论：

1. CoT 让模型在最终答案前生成中间推理步骤，适合多步推理任务。
2. CoT 可能通过增加 test-time compute、降低任务难度和提供中间状态提升表现。
3. CoT、scratchpad、rationale 和 hidden reasoning 不是同一个概念。
4. CoT 的核心争议是忠实性：模型写出的理由不一定是真实决策原因。
5. CoT 还有安全争议，完整展示内部推理可能泄露策略、增加误导和攻击面。
6. CoT 最好和 self-consistency、verifier、搜索或过程监督结合，而不是单独依赖。
7. 评估 CoT 要同时看最终答案、中间步骤、忠实性、鲁棒性、成本和安全。
8. 面试中要把 CoT 讲成 reasoning 和 test-time compute 的入口，而不是简单一句 prompt 技巧。

## 第 87 讲：Self-Consistency 与采样增强推理

### 本讲目标

学完本讲，你应该能回答六个问题：

1. Self-consistency 想解决 CoT 的什么问题。
2. 为什么多条推理路径投票可能提升推理准确率。
3. Self-consistency 和 temperature、top-p、采样次数有什么关系。
4. Majority vote、weighted vote、verifier selection 有什么区别。
5. Self-consistency 的成本、失败模式和适用边界是什么。
6. 面试中如何设计一个 CoT + self-consistency 的数学推理系统。

第 86 讲讲了 Chain-of-Thought。

CoT 的核心是让模型写出中间推理步骤，再给最终答案。

但单条 CoT 有一个明显问题：

```text
模型可能沿着一条错误推理路径走到底。
```

如果只采样一次，结果高度依赖这一次生成。

Self-consistency 的想法是：

```text
不要只相信一条推理链，而是采样多条推理链，再从多个答案中选出最一致的答案。
```

这类似人类做复杂题时用多种方法验算。

如果不同路径都得到同一个答案，这个答案更可信。

---

### 一、Self-Consistency 的基本直觉

同一个问题可能有多条推理路径。

例如一个数学题可以：

1. 用代数方法解。
2. 用画图方法解。
3. 用枚举方法解。
4. 用逆向验证解。

单次 CoT 只探索一条路径。

如果这条路径错了，最终答案很可能错。

Self-consistency 通过随机采样探索多条路径。

流程是：

```text
问题
-> 采样 N 条 CoT
-> 提取每条 CoT 的最终答案
-> 对答案投票或打分
-> 选择最一致的答案
```

例如采样 5 条：

```text
路径 1 -> 答案 A
路径 2 -> 答案 A
路径 3 -> 答案 B
路径 4 -> 答案 A
路径 5 -> 答案 C
```

最终选择 A。

核心假设是：

```text
正确答案更可能被多条独立或半独立推理路径得到。
```

这个假设不总是成立，但在很多数学和符号推理任务中有效。

---

### 二、为什么不是贪心解码

普通 greedy decoding 每一步选概率最高的 token。

它输出稳定，但缺少探索。

Self-consistency 通常需要非贪心采样。

例如：

1. 设置 temperature > 0。
2. 使用 top-p 或 top-k sampling。
3. 生成多条不同 CoT。

如果每次都 greedy，采样 10 次也可能完全一样。

那就没有 self-consistency 的意义。

所以 self-consistency 的关键不是“重复问 10 次”。

而是：

```text
通过采样得到多样化推理路径，再利用答案一致性聚合结果。
```

---

### 三、Self-Consistency 的算法流程

一个最小流程如下：

```text
Input: question q, model M, sample count N

for i in 1..N:
    reasoning_i = M.generate(q, prompt="think step by step", temperature=t)
    answer_i = extract_final_answer(reasoning_i)

final_answer = majority_vote(answer_1, ..., answer_N)
return final_answer
```

从概率角度看，self-consistency 可以理解为把隐藏的 reasoning path `r` 边缘化掉：

```math
P(a\mid x)
=
\sum_r P(a\mid x,r)P(r\mid x)
```

真实系统无法枚举所有 `r`，于是用采样近似：

```math
\hat P(a\mid x)
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[a_i=a]
```

最终选择：

```math
\hat a
=
\arg\max_a \hat P(a\mid x)
```

这个公式说明 self-consistency 的本质不是“让模型更啰嗦”，而是用多条采样路径近似答案分布，再选择后验质量更高的答案。

关键模块有三个。

#### 1. 采样

生成多条 reasoning path。

#### 2. 答案抽取

从每条 CoT 中抽取最终答案。

#### 3. 聚合

对答案做 majority vote、weighted vote 或 verifier selection。

这三个模块任何一个做不好，效果都会下降。

---

### 四、答案抽取为什么重要

Self-consistency 依赖最终答案聚合。

但模型输出可能不规范。

例如：

```text
所以答案应该是 12。
```

```text
最终结果：12 个。
```

```text
答案是十二。
```

```text
因此 x = 12。
```

这些都应该归一化成同一个答案。

答案抽取常见问题：

1. 输出多个候选答案。
2. 单位不同。
3. 小数和分数等价。
4. 中文数字和阿拉伯数字混用。
5. 最终答案格式不固定。
6. 推理中间出现多个数字，误抽取中间值。

因此生产系统通常会要求模型输出结构化字段：

```json
{
  "reasoning_summary": "...",
  "final_answer": "12"
}
```

或者在推理后再调用一个 answer extractor。

---

### 五、Majority Vote

最简单聚合方式是 majority vote。

也就是选择出现次数最多的答案。

例如：

```text
A: 7 次
B: 2 次
C: 1 次
```

选择 A。

优点：

1. 简单。
2. 不需要额外模型。
3. 对随机错误有一定鲁棒性。

缺点：

1. 如果多数路径共享同一个错误偏差，会选错。
2. 不能区分推理质量。
3. 对开放答案、长文本答案不容易投票。
4. 答案归一化困难。

Majority vote 最适合答案空间比较明确的任务。

例如数学题、选择题、短答案题。

如果假设每条路径独立，且单条路径得到正确答案的概率为 `p`，那么奇数 `N` 下多数投票正确的概率可以写成：

```math
P_{\mathrm{maj}}(N,p)
=
\sum_{j=(N+1)/2}^{N}
\binom{N}{j}
p^j(1-p)^{N-j}
```

当 `p>0.5` 且错误比较分散时，`P_maj` 会随 `N` 增加而上升。但这个公式依赖很强的独立同分布假设。真实 LLM 的多条路径往往共享同一个 prompt、同一个模型和相似训练偏差，错误可能高度相关。

可以用一个粗略有效样本数提醒这种相关性：

```math
N_{\mathrm{eff}}
=
\frac{N}{1+(N-1)\rho}
```

其中 `\rho` 表示路径错误相关性。若 `\rho` 很高，名义上采样 20 条，实际信息增益可能远小于 20 条。

---

### 六、Weighted Vote

Weighted vote 会给不同答案或路径不同权重。

权重可以来自：

1. 模型生成概率。
2. 答案置信度。
3. 推理步骤质量分。
4. Verifier 分数。
5. 路径长度惩罚。
6. 工具验证结果。

例如：

```text
答案 A: 路径 1 分数 0.8 + 路径 2 分数 0.7 = 1.5
答案 B: 路径 3 分数 0.95 = 0.95
```

选择 A。

Weighted vote 比 majority vote 更灵活。

但也更依赖评分是否可靠。

如果模型自己的 confidence 不校准，weighted vote 可能被高置信错误路径误导。

加权投票可以写成：

```math
score(a)
=
\sum_{i=1}^{N}
w_i\mathbf{1}[a_i=a]
```

```math
\hat a
=
\arg\max_a score(a)
```

其中 `w_i` 可以来自 verifier 分数、程序校验结果、路径 log probability、步骤有效率或人工规则。一个常见做法是把 verifier 分数转成非负权重：

```math
w_i=\exp\left(\frac{v_i}{\tau_v}\right)
```

`v_i` 是 verifier 分数，`\tau_v` 控制权重尖锐程度。`\tau_v` 太小会几乎只相信最高分路径，太大则接近普通多数投票。

---

### 七、Verifier Selection

更强的方法是引入 verifier。

流程是：

```text
采样多条 CoT
-> 提取候选答案
-> 用 verifier 检查每条推理或答案
-> 选择 verifier 分数最高的结果
```

Verifier 可以检查：

1. 数学步骤是否成立。
2. 代码是否通过测试。
3. 答案是否满足约束。
4. 推理是否引用了正确证据。
5. 最终答案是否和题目条件一致。

Self-consistency 和 verifier 的区别是：

```text
self-consistency 依赖多条路径的一致性，verifier 依赖外部或额外模型判断质量。
```

两者可以结合。

例如先投票得到 top candidates，再用 verifier 选择最终答案。

第 88 讲会专门展开 verifier。

---

### 八、采样参数怎么选

Self-consistency 的效果和采样参数关系很大。

#### 1. Temperature

Temperature 控制随机性。

太低：路径太相似，缺少多样性。

太高：路径太随机，错误增多。

#### 2. Top-p

Top-p 控制候选 token 累积概率范围。

较高 top-p 增加多样性，但也可能增加噪声。

#### 3. Sample count N

N 越大，覆盖路径越多。

但成本线性增加。

常见实验会试：

```text
N = 5, 10, 20, 40
```

但生产系统要结合延迟和成本。

#### 4. Max tokens

CoT 太长会增加成本，也可能跑偏。

需要限制最大推理长度。

一个实用原则是：

```text
采样要足够多样，但不能让路径完全随机；N 要足够提升准确率，但不能让成本失控。
```

---

### 九、为什么 Self-Consistency 有效

可以从几个角度理解。

#### 1. 错误路径不完全一致

如果错误是随机的，不同路径会错到不同答案。

正确答案更可能集中。

#### 2. 多路径探索

复杂问题可能有多个解法。

一次采样没找到正确路径，多次采样可能找到。

#### 3. 集成效应

Self-consistency 类似 ensemble。

多个样本聚合通常比单个样本更稳。

#### 4. 增加 test-time compute

采样 N 条路径相当于把推理时计算扩大 N 倍。

这本质上是一种 test-time compute scaling。

但要注意：

```text
如果模型系统性误解题目，多采样只会更稳定地错。
```

---

### 十、适用场景

Self-consistency 适合：

1. 数学题。
2. 逻辑题。
3. 选择题。
4. 短答案多跳问答。
5. 可自动抽取答案的任务。
6. 有 verifier 的代码或工具任务。

不太适合：

1. 开放式创作。
2. 长篇总结。
3. 主观评价。
4. 答案难以归一化的任务。
5. 对低延迟要求很高的在线场景。

例如客服简单问答不适合每次采样 20 条。

但高价值数学评测、代码修复、复杂决策辅助可能值得。

---

### 十一、成本和延迟

Self-consistency 最大代价是成本。

如果采样 N 条，每条 CoT 平均 T 个 token。

输出成本大约增加为：

```text
N * T
```

延迟取决于是否并行。

如果串行采样，延迟也约增加 N 倍。

如果并行采样，延迟接近最长那条路径，但吞吐和费用仍增加。

生产系统常用优化：

1. 只对困难问题启用。
2. 先用小 N，不确定时再加采样。
3. 并行采样。
4. 提前停止：某答案已经明显领先。
5. 对简单任务直接回答。
6. 用 verifier 减少无效采样。

Self-consistency 是准确率和成本之间的 trade-off。

如果每条路径平均输出 `\bar n` 个 token，平均 verifier 调用 `\bar k_v` 次，采样数为 `N`，可以把相对成本写成：

```math
C_N =
N(c_{\mathrm{out}}\bar n + c_{\mathrm{call}})
+
c_v\bar k_v
```

串行延迟粗略为：

```math
T_{\mathrm{serial}}
\approx
\sum_{i=1}^{N} T_i
```

并行延迟粗略为：

```math
T_{\mathrm{parallel}}
\approx
\max_i T_i
+
T_{\mathrm{agg}}
```

并行能降低用户等待时间，但不会降低总 token、总 GPU 占用和排队压力。生产上需要把 `N` 放进预算管理，而不是把 self-consistency 当成免费准确率提升。

---

### 十二、失败模式

#### 1. 多数一致但都错

如果模型有系统性偏差，多数投票会强化错误。

#### 2. 答案抽取错误

推理正确，但 extractor 抽错最终答案。

#### 3. 归一化失败

`1/2`、`0.5`、`50%` 被当成不同答案。

#### 4. 多样性不足

采样参数太保守，N 条路径几乎一样。

#### 5. 随机性过强

temperature 太高，推理质量下降。

#### 6. 开放答案无法投票

长文本答案难以做 majority vote。

#### 7. 成本失控

对所有请求都多采样，延迟和费用不可接受。

---

### 十三、评估方法

评估 self-consistency 至少要比较四组。

```text
direct answer
single CoT
CoT + self-consistency
CoT + self-consistency + verifier
```

指标包括：

1. Accuracy。
2. Pass@k 或 solve rate。
3. Majority confidence。
4. Answer extraction accuracy。
5. Cost per solved problem。
6. Latency。
7. Token usage。
8. Robustness across prompts。
9. Calibration。

答案分布本身也可以作为不确定性信号。可以定义答案分布熵：

```math
H(A)
=
-
\sum_a
\hat P(a\mid x)\log \hat P(a\mid x)
```

以及 majority margin：

```math
M =
\hat P(a_1\mid x)
-
\hat P(a_2\mid x)
```

其中 `a_1` 和 `a_2` 是得票最高和第二高的答案。`H(A)` 高、`M` 低，通常说明模型不稳定，可能需要更多采样、调用 verifier，或者直接返回不确定性。

还要画成本收益曲线。

例如：

```text
N=1  accuracy=70%, cost=1x
N=5  accuracy=78%, cost=5x
N=20 accuracy=82%, cost=20x
```

如果 N 从 5 到 20 只提升 4%，但成本变 4 倍，生产中未必值得。

---

### 十四、最小代码：投票、加权投票、Verifier 选择与 pass@k

下面的 demo 不调用模型，只模拟多条 CoT 输出。它展示：

1. `1/2`、`0.5`、`50%` 需要归一化成同一个答案。
2. Majority vote 适合错误分散的场景。
3. 当多数路径共享同一个错误时，weighted vote 或 verifier selection 可能救回来。
4. `pass@k` 只说明候选集合里有正确答案，不说明系统最终选对。
5. Self-consistency 的相对成本约随采样数 `N` 线性增长。

```python
from collections import Counter, defaultdict
from fractions import Fraction
import math
import re


FINAL_RE = re.compile(r"(?:final|answer|答案)[:：]\s*([0-9./%a-zA-Z_-]+)", re.I)


def normalize_answer(raw):
    text = raw.strip().lower().rstrip("。,. ")
    try:
        if text.endswith("%"):
            value = Fraction(text[:-1]) / 100
        elif "/" in text:
            value = Fraction(text)
        elif re.fullmatch(r"-?\d+(\.\d+)?", text):
            value = Fraction(text)
        else:
            return text
        return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    except (ValueError, ZeroDivisionError):
        return text


def extract_answer(text):
    match = FINAL_RE.search(text)
    if match:
        return normalize_answer(match.group(1))
    fallback = re.findall(r"-?\d+(?:/\d+|\.\d+|%)?", text)
    return normalize_answer(fallback[-1]) if fallback else ""


def majority_vote(candidates):
    answers = [extract_answer(item["text"]) for item in candidates]
    return Counter(answers).most_common(1)[0][0]


def weighted_vote(candidates):
    scores = defaultdict(float)
    for item in candidates:
        scores[extract_answer(item["text"])] += item["verifier"]
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def verifier_selection(candidates):
    best = max(candidates, key=lambda item: item["verifier"])
    return extract_answer(best["text"])


def answer_entropy(candidates):
    answers = [extract_answer(item["text"]) for item in candidates]
    total = len(answers)
    probs = [count / total for count in Counter(answers).values()]
    return -sum(p * math.log(p) for p in probs)


def majority_margin(candidates):
    counts = Counter(extract_answer(item["text"]) for item in candidates).most_common()
    if len(counts) == 1:
        return 1.0
    return (counts[0][1] - counts[1][1]) / len(candidates)


problems = {
    "scattered_errors": {
        "gold": "42",
        "candidates": [
            {"text": "method A gives final: 42", "verifier": 0.82},
            {"text": "method B gives final: 42", "verifier": 0.76},
            {"text": "calculation slip final: 41", "verifier": 0.20},
            {"text": "reverse check final: 42", "verifier": 0.70},
            {"text": "wrong branch final: 40", "verifier": 0.30},
        ],
    },
    "normalization_needed": {
        "gold": "1/2",
        "candidates": [
            {"text": "fraction form final: 1/2", "verifier": 0.80},
            {"text": "decimal form final: 0.5", "verifier": 0.70},
            {"text": "percent form final: 50%", "verifier": 0.60},
            {"text": "wrong fraction final: 2/3", "verifier": 0.40},
            {"text": "extra zero final: 0.50", "verifier": 0.65},
        ],
    },
    "majority_wrong": {
        "gold": "17",
        "candidates": [
            {"text": "shared misconception final: 16", "verifier": 0.30},
            {"text": "same misconception final: 16", "verifier": 0.25},
            {"text": "format copy final: 16", "verifier": 0.20},
            {"text": "careful derivation final: 17", "verifier": 0.95},
            {"text": "tool check final: 17", "verifier": 0.90},
        ],
    },
}


metrics = Counter()
total_samples = 0
for name, problem in problems.items():
    gold = normalize_answer(problem["gold"])
    candidates = problem["candidates"]
    total_samples += len(candidates)
    maj = majority_vote(candidates)
    wgt = weighted_vote(candidates)
    ver = verifier_selection(candidates)
    pass_at_k = any(extract_answer(item["text"]) == gold for item in candidates)

    metrics["majority_correct"] += maj == gold
    metrics["weighted_correct"] += wgt == gold
    metrics["verifier_correct"] += ver == gold
    metrics["pass_at_k"] += pass_at_k

    print(
        name,
        "gold=", gold,
        "majority=", maj,
        "weighted=", wgt,
        "verifier=", ver,
        "entropy=", round(answer_entropy(candidates), 3),
        "margin=", round(majority_margin(candidates), 3),
    )

n = len(problems)
print("summary=", {
    "majority_acc": round(metrics["majority_correct"] / n, 3),
    "weighted_acc": round(metrics["weighted_correct"] / n, 3),
    "verifier_acc": round(metrics["verifier_correct"] / n, 3),
    "pass@5": round(metrics["pass_at_k"] / n, 3),
    "relative_cost_vs_greedy": total_samples / n,
})
```

一组输出如下：

```text
scattered_errors gold= 42 majority= 42 weighted= 42 verifier= 42 entropy= 0.95 margin= 0.4
normalization_needed gold= 1/2 majority= 1/2 weighted= 1/2 verifier= 1/2 entropy= 0.5 margin= 0.6
majority_wrong gold= 17 majority= 16 weighted= 17 verifier= 17 entropy= 0.673 margin= 0.2
summary= {'majority_acc': 0.667, 'weighted_acc': 1.0, 'verifier_acc': 1.0, 'pass@5': 1.0, 'relative_cost_vs_greedy': 5.0}
```

这个 demo 的重点是第三个样本：`pass@5=1.0` 说明候选里有正确答案，但 majority vote 仍然选错；weighted vote 和 verifier selection 能选对，是因为正确路径虽然少，但 verifier 分数更高。面试中要明确区分“生成过正确候选”和“系统最终选择正确候选”。

---

### 十五、真实项目中的坑

#### 1. 以为多采样一定更好

如果模型系统性错，多采样只会更贵地错。

#### 2. 不做答案归一化

等价答案被拆成多个类别，投票失效。

#### 3. 忽略推理路径质量

多数答案可能来自低质量推理。

#### 4. 对所有任务启用

简单任务不需要 self-consistency。

#### 5. 只报告最高准确率

不报告 token、延迟和成本，评估不完整。

#### 6. 采样路径不独立

prompt 和参数导致路径高度相似，投票没有意义。

#### 7. 最终答案格式不固定

导致 extractor 不稳定，线上结果抖动。

---

### 十六、面试问答

#### 问题 1：Self-consistency 的核心思想是什么？

可以这样回答：

```text
Self-consistency 不是只采样一条 CoT，而是采样多条推理路径，抽取每条路径的最终答案，再通过投票或 verifier 选择最一致或最可信的答案。
```

#### 问题 2：为什么 self-consistency 能提升推理准确率？

可以这样回答：

```text
因为复杂问题可能有多条推理路径，单条路径容易偶然出错。多路径采样能探索不同解法，如果错误较分散而正确答案更一致，投票就能提升准确率。
```

#### 问题 3：Self-consistency 和 temperature 有什么关系？

可以这样回答：

```text
Self-consistency 需要一定随机性来生成多样化推理路径。temperature 太低路径相似，太高推理变噪声，所以要在多样性和质量之间调参。
```

#### 问题 4：Majority vote 和 verifier selection 有什么区别？

可以这样回答：

```text
Majority vote 选择出现次数最多的答案，不判断每条推理质量；verifier selection 用规则、模型或工具检查候选路径或答案，选择验证分数最高的结果。两者可以结合。
```

#### 问题 5：Self-consistency 的主要缺点是什么？

可以这样回答：

```text
主要缺点是成本和延迟高，答案抽取和归一化困难，多数路径可能共享系统性错误，对开放式长答案不容易投票，也不适合所有低延迟场景。
```

#### 问题 6：如何设计一个 CoT + self-consistency 数学推理系统？

可以这样回答：

```text
先用 CoT prompt 并行采样多条推理路径，再用结构化格式抽取 final answer，对等价答案做归一化，然后 majority vote 或 verifier selection，最后输出最终答案和简洁解释，同时记录 N、成本、延迟和正确率。
```

---

### 十七、常见误区

1. 误区：Self-consistency 就是重复问模型多次。
   纠正：关键是采样多样化推理路径，并对最终答案做聚合。

2. 误区：采样越多越好。
   纠正：采样越多成本越高，收益会递减。

3. 误区：多数投票一定正确。
   纠正：模型系统性错误时，多数也会错。

4. 误区：不需要答案抽取。
   纠正：答案抽取和归一化是 self-consistency 成败关键。

5. 误区：Self-consistency 适合所有任务。
   纠正：它更适合答案可归一化的推理任务。

6. 误区：只看准确率提升。
   纠正：还要看 token、延迟、成本和线上可用性。

---

### 十八、小练习

1. 用伪代码写出 self-consistency 推理流程。
2. 给一个数学题设计 N=5 的 CoT 采样和投票示例。
3. 设计一个 final answer extractor，处理中文数字、单位和分数。
4. 比较 majority vote、weighted vote 和 verifier selection。
5. 分析 temperature 太低和太高分别会怎样影响 self-consistency。
6. 设计一个提前停止策略，减少采样成本。
7. 构造一个多数一致但答案错误的案例。
8. 设计一个成本收益实验，比较 N=1、5、10、20。
9. 分析为什么长篇开放问答不适合简单 majority vote。
10. 用 3 分钟回答：“Self-consistency 为什么能提升 reasoning？”

### 本讲总结

本讲最重要的结论：

1. Self-consistency 用多条 CoT 采样和答案聚合提升推理稳定性。
2. 它的核心假设是正确答案更可能被多条不同推理路径一致得到。
3. Self-consistency 需要适当随机性，greedy 重复生成没有意义。
4. 答案抽取和归一化是系统实现中的关键细节。
5. Majority vote 简单有效，但不能判断推理质量；verifier selection 更强但更复杂。
6. Self-consistency 是一种 test-time compute scaling，用更多推理时计算换准确率。
7. 它的主要代价是 token、延迟和成本，并且会受到系统性错误影响。
8. 面试中要强调准确率收益和工程成本之间的 trade-off。

## 第 88 讲：Verifier、Process Reward Model 与 Outcome Reward Model

### 本讲目标

学完本讲，你应该能回答六个问题：

1. Verifier 在 reasoning model 中解决什么问题。
2. Outcome Reward Model 和 Process Reward Model 的区别是什么。
3. 为什么只看最终答案不够，为什么过程监督更难也更有价值。
4. Verifier 如何和 CoT、self-consistency、search 结合。
5. PRM/ORM 的数据、训练、评估和失败模式有哪些。
6. 面试中如何设计一个数学推理或代码推理的 verifier 系统。

第 86 讲讲 CoT，第 87 讲讲 self-consistency。

它们都在做一件事：

```text
生成更多推理过程，然后从中选出更好的答案。
```

但关键问题是：

```text
怎么知道哪条推理更好？
```

如果只靠 majority vote，多数答案也可能错。

如果只靠模型自信，模型可能高置信地错。

Verifier 就是为了解决这个问题：

```text
给候选答案或推理过程打分，判断它是否正确、可信、满足约束。
```

在 reasoning model 里，verifier 往往和 CoT、self-consistency、search、test-time compute scaling 一起使用。

从方法脉络看，verifier 不是凭空出现的。GSM8K 相关工作把“先生成很多数学解，再训练 verifier 选择最高分候选”作为提升多步数学推理的路线；后续过程监督工作进一步比较 outcome-based feedback 和 process-based feedback，指出最终答案监督成本低，但过程反馈更能约束中间推理错误。OpenAI 的 Let's Verify Step by Step 进一步把 PRM800K 这种 step-level human feedback 数据用于训练 process reward model。面试中要把这条线讲清楚：ORM/PRM 的核心不是多一个 judge，而是把“生成候选”和“评价候选”拆成两个可单独优化、可单独评估的模块。

---

### 一、什么是 Verifier

Verifier 可以理解为验证器。

它的输入通常是：

1. 原始问题。
2. 模型生成的答案。
3. 可选的中间推理步骤。
4. 可选的工具执行结果。
5. 可选的参考答案或约束。

输出通常是：

1. 正确或错误。
2. 一个分数。
3. 哪一步有问题。
4. 是否满足约束。
5. 候选之间的排序。

例如数学题：

```text
问题：解方程 2x + 3 = 11。
候选推理：2x = 8，所以 x = 4。
Verifier 输出：正确，score=0.98。
```

例如代码题：

```text
候选代码通过 18/20 个测试。
Verifier 输出：部分正确，score=0.72，失败用例是边界条件。
```

Verifier 不一定是神经网络。

它可以是规则、程序、单元测试、符号检查器、另一个模型或人工标注。

为了后面公式统一，记：

1. 原始问题是 $q$。
2. 第 $i$ 个候选的推理过程是 $r_i = (u_{i,1}, u_{i,2}, \ldots, u_{i,T_i})$。
3. 第 $i$ 个候选的最终答案是 $a_i$。
4. verifier 参数是 $\phi$，输出分数在 $[0,1]$。

候选级 verifier 可以写成：

$$
s_i = v_\phi(q, r_i, a_i), \quad s_i \in [0,1]
$$

如果采样得到 $N$ 个候选，最简单的 verifier selection 是：

$$
i^\star = \arg\max_{i \in \{1,\ldots,N\}} S_i,\quad a^\star = a_{i^\star}
$$

这里 $S_i$ 可以只用 ORM 分数，也可以融合 PRM、规则检查、单元测试、长度惩罚和成本惩罚。

---

### 二、为什么需要 Verifier

只让模型生成答案有几个问题。

#### 1. 模型会流畅地错

错误推理可能写得很像真的。

#### 2. 多条 CoT 不知道选哪条

Self-consistency 可以投票，但不能判断少数路径是否其实更正确。

#### 3. 最终答案对不代表过程对

模型可能碰巧得到正确答案，但中间逻辑错误。

#### 4. 最终答案错不代表全程错

模型可能前面都对，最后一步计算错。

如果能定位错误步骤，就更容易改进。

#### 5. Search 需要评价函数

Tree-of-Thought、MCTS 等搜索方法需要评估中间状态好坏。

Verifier 可以提供评价信号。

因此 verifier 是 reasoning 系统中的“判题器”或“评审器”。

---

### 三、Outcome Reward Model

Outcome Reward Model，简称 ORM。

它评价的是最终结果。

输入可以是：

```text
question + final answer
```

也可以包含推理过程：

```text
question + reasoning + final answer
```

但监督信号通常来自最终答案是否正确。

例如：

```text
问题：24 的 1/3 是多少？
答案：8
label: correct
```

ORM 的优点：

1. 标注相对简单。
2. 很多任务天然有最终答案。
3. 适合对多个候选答案排序。
4. 可以和 self-consistency 结合。

ORM 的缺点：

1. 不知道哪一步错。
2. 对过程质量监督弱。
3. 可能奖励碰巧正确的错误推理。
4. 对开放题最终正确性难判断。

ORM 适合回答：

```text
这个候选最终结果是否好？
```

但不擅长回答：

```text
这个推理过程每一步是否对？
```

用公式写，ORM 近似的是候选最终成功概率：

$$
s_i^{\mathrm{ORM}} = v_\phi^{\mathrm{ORM}}(q, r_i, a_i) \approx P(z_i = 1 \mid q, r_i, a_i)
$$

其中 $z_i=1$ 表示第 $i$ 个候选最终答案正确或满足任务目标。数学题里不要直接比较原始字符串，而要先做答案抽取和归一化：

$$
z_i = \mathbf{1}\left[\operatorname{norm}(a_i)=\operatorname{norm}(a^\star)\right]
$$

代码题里 $z_i$ 通常来自编译、单元测试、隐藏测试或执行约束，而不是 LLM judge 的主观判断。用 ORM 做重排时：

$$
i^\star_{\mathrm{ORM}} = \arg\max_i s_i^{\mathrm{ORM}}
$$

如果 ORM 训练成二分类器，常见 pointwise 损失是：

$$
\mathcal{L}_{\mathrm{ORM}} =
-\frac{1}{M}\sum_{j=1}^{M}
\left[
z_j \log s_j + (1-z_j)\log(1-s_j)
\right]
$$

如果训练数据是候选对 $(y^+, y^-)$，也可以用 pairwise ranking 损失：

$$
\mathcal{L}_{\mathrm{pair}} =
-\log \sigma\left(s^+ - s^-\right)
$$

这表示希望正确候选 $y^+$ 的 verifier 分数高于错误候选 $y^-$。真实系统中，pairwise 标注经常比绝对打分更稳定，但分数校准通常更难。

---

### 四、Process Reward Model

Process Reward Model，简称 PRM。

它评价的是推理过程中的每一步。

例如一个推理过程：

```text
Step 1: 2x + 3 = 11
Step 2: 2x = 8
Step 3: x = 4
```

PRM 可以给每一步打分：

```text
Step 1: correct
Step 2: correct
Step 3: correct
```

如果推理是：

```text
Step 1: 2x + 3 = 11
Step 2: 2x = 14
Step 3: x = 7
```

PRM 应该指出 Step 2 错。

PRM 的优点：

1. 能定位错误步骤。
2. 能指导模型改进过程。
3. 适合搜索中评估中间状态。
4. 比只看最终答案更细粒度。
5. 有助于训练更稳定的 reasoning。

PRM 的缺点：

1. 标注成本高。
2. 需要定义“步骤”粒度。
3. 有些推理步骤很难判断对错。
4. 标注者之间可能不一致。
5. 模型可能学会迎合 PRM，而不是真正解决问题。

PRM 适合回答：

```text
这一步推理是否合理？下一步是否值得继续？
```

PRM 的基本输入不是整条最终答案，而是“问题 + 已有步骤 + 当前步骤”。令第 $i$ 条推理第 $t$ 步的历史状态为：

$$
h_{i,t} = (q, u_{i,1}, \ldots, u_{i,t-1})
$$

PRM 给当前步骤打分：

$$
p_{i,t} = v_\phi^{\mathrm{PRM}}(h_{i,t}, u_{i,t})
$$

其中 $p_{i,t}$ 可以理解为“第 $t$ 步正确、相关、可继续”的概率或质量分。整条推理路径的过程分数可以用几种方式聚合：

$$
S_i^{\mathrm{mean}} = \frac{1}{T_i}\sum_{t=1}^{T_i} p_{i,t}
$$

$$
S_i^{\mathrm{geo}} =
\left(\prod_{t=1}^{T_i} p_{i,t}\right)^{1/T_i}
$$

$$
S_i^{\mathrm{min}} = \min_{1 \le t \le T_i} p_{i,t}
$$

为了避免长链路被乘积天然压低，实际排序也常用平均 log 分数：

$$
S_i^{\mathrm{logavg}} =
\frac{1}{T_i}\sum_{t=1}^{T_i}\log p_{i,t}
$$

面试里可以这样解释这三个聚合方式：

1. mean 容忍个别低分步骤，适合粗略排序。
2. geo 会明显惩罚低分步骤，适合多步推理质量控制。
3. min 最保守，适合高风险任务或搜索剪枝。

这里要避免一个常见误解：$\prod_t p_{i,t}$ 或平均步骤分数不等于严格的“最终答案正确概率”。PRM 判断的是局部步骤质量，局部步骤都像是正确，也可能因为目标偏离、冗余步骤、最后格式错误或 verifier 偏差导致最终失败。

PRM 还可以定位第一处错误：

$$
t_i^{\mathrm{err}} = \min\{t \mid p_{i,t} < \tau\}
$$

如果不存在这样的 $t$，说明这条路径在当前阈值 $\tau$ 下没有被 PRM 标出错误。这个指标比“最终答案错了”更有用，因为它能告诉你数据、模型或搜索应该从哪一步开始修。

---

### 五、ORM 和 PRM 对比

可以用一张表理解。

| 维度 | ORM | PRM |
|---|---|---|
| 监督对象 | 最终答案 | 推理步骤 |
| 标注成本 | 较低 | 较高 |
| 错误定位 | 弱 | 强 |
| 过程质量监督 | 弱 | 强，但依赖步骤标注和校准 |
| 适合搜索 | 可用于终局评价 | 可用于中间状态评价 |
| 风险 | 奖励碰巧正确 | 步骤标注主观、成本高 |

一句话总结：

```text
ORM 看结果，PRM 看过程。
```

但真实系统不一定二选一。

可以同时使用：

1. PRM 评估中间步骤。
2. ORM 评估最终答案。
3. 工具或规则做硬验证。

一个工程上常见的融合分数可以写成：

$$
S_i =
\lambda_{\mathrm{o}} s_i^{\mathrm{ORM}}
+ \lambda_{\mathrm{p}} S_i^{\mathrm{PRM}}
+ \lambda_{\mathrm{r}} R_i
- \lambda_{\mathrm{c}} C_i
$$

其中 $R_i$ 是规则、单元测试或符号检查分数，$C_i$ 是 token、延迟或执行成本，$\lambda$ 是业务权重。这个公式的意义不是让读者背权重，而是提醒：真实 verifier selection 往往不是单一神经分数，而是质量、过程、硬验证和成本的折中。

---

### 六、Verifier 的类型

Verifier 可以分成几类。

#### 1. Rule-based verifier

用规则检查答案格式、约束、单位、范围。

例如答案必须是整数，必须在 0 到 1 之间。

#### 2. Programmatic verifier

用程序执行检查。

例如代码题跑单元测试，数学题用符号计算验证。

#### 3. Neural verifier

训练一个模型判断答案或过程是否正确。

ORM 和 PRM 通常属于这一类。

#### 4. LLM-as-a-verifier

用另一个 LLM 判断候选答案。

优点是通用。

缺点是也会错，需要校准。

#### 5. Human verifier

人工标注或审核。

成本高，但适合高风险和校准集。

实际系统常用混合方案：

```text
规则过滤 + 程序验证 + 神经 verifier + 人工抽查
```

---

### 七、Verifier 和 CoT 怎么结合

CoT 生成过程，verifier 检查过程。

最简单流程：

```text
生成 CoT
-> verifier 检查最终答案
-> 如果不通过，重新生成
```

更强流程：

```text
生成多条 CoT
-> verifier 给每条打分
-> 选择最高分
```

更细粒度流程：

```text
每生成一步
-> PRM 打分
-> 分数低则剪枝或回退
-> 分数高则继续
```

这就接近搜索方法。

第 89 讲会讲 Tree-of-Thought 和 MCTS。

本讲先记住：

```text
Verifier 可以把“生成很多答案”变成“生成、筛选、改进答案”。
```

---

### 八、Verifier 和 Self-Consistency 怎么结合

第 87 讲讲了 self-consistency。

它默认多数答案更可信。

但有时少数答案才对。

因此可以加入 verifier。

流程一：先投票，再验证。

```text
采样 N 条 CoT
-> majority vote 得到 top answer
-> verifier 检查 top answer
```

流程二：先验证，再投票。

```text
采样 N 条 CoT
-> verifier 过滤低质量路径
-> 对剩余答案投票
```

流程三：直接按 verifier 分数选。

```text
采样 N 条 CoT
-> verifier 给每条路径打分
-> 选择最高分路径
```

工程上通常要比较这几种策略的准确率、成本和稳定性。

---

### 九、PRM 数据怎么标注

PRM 的难点是过程标注。

一个样本通常包含：

```text
问题
推理步骤 1
步骤 1 标签
推理步骤 2
步骤 2 标签
...
最终答案
```

标签可以是：

1. correct / incorrect。
2. 分数 0-1。
3. 哪一步开始错误。
4. 错误类型。
5. 是否可继续。

标注来源包括：

1. 人工专家标注。
2. 程序自动验证。
3. LLM 辅助标注。
4. 从正确解答中自动构造。
5. 从错误模型输出中挖掘负样本。

PRM 数据要覆盖错误步骤。

如果训练集只有完美推理，PRM 学不会识别错误。

---

### 十、ORM 数据怎么标注

ORM 数据相对简单。

样本通常是：

```text
问题 + 候选答案 + label/score
```

标注来源包括：

1. 有标准答案的数据集。
2. 单元测试通过率。
3. 人工偏好比较。
4. LLM judge。
5. 用户反馈。

ORM 可以做二分类：

```text
correct vs incorrect
```

也可以做 pairwise ranking：

```text
候选 A 比候选 B 更好
```

或者直接回归分数。

ORM 的关键是负样本质量。

太容易的负样本没用。

需要 hard negatives，例如：

1. 最终答案差一点。
2. 推理看起来合理但有隐藏错误。
3. 格式正确但约束不满足。
4. 常见误解导致的错误。

---

### 十一、训练目标

Verifier 的训练目标取决于任务。

#### 1. Classification

判断正确或错误。

$$
s = v_\phi(q, r, a) \approx P(z=1 \mid q, r, a)
$$

二分类 verifier 常用交叉熵。若第 $j$ 个样本标签是 $z_j \in \{0,1\}$：

$$
\mathcal{L}_{\mathrm{cls}} =
-\frac{1}{M}\sum_{j=1}^{M}
\left[
z_j \log s_j + (1-z_j)\log(1-s_j)
\right]
$$

#### 2. Regression

预测一个连续分数。

例如 0 到 1 的质量分。

如果人工标注或程序测试给出连续目标 $g_j \in [0,1]$，可以用均方误差：

$$
\mathcal{L}_{\mathrm{reg}} =
\frac{1}{M}\sum_{j=1}^{M}(s_j-g_j)^2
$$

#### 3. Ranking

学习候选之间的相对好坏。

例如 pairwise loss：

$$
\mathcal{L}_{\mathrm{rank}} =
-\frac{1}{K}\sum_{k=1}^{K}
\log \sigma(s_k^+ - s_k^-)
$$

其中 $s_k^+$ 是 chosen/correct 候选分数，$s_k^-$ 是 rejected/incorrect 候选分数。它直接优化“好候选排在坏候选前面”。

#### 4. Step-wise classification

PRM 对每一步判断是否正确。

如果第 $j$ 条推理有 $T_j$ 个步骤，步骤标签是 $z_{j,t}$，PRM 损失可以写成：

$$
\mathcal{L}_{\mathrm{PRM}} =
-\frac{1}{\sum_{j=1}^{M}T_j}
\sum_{j=1}^{M}\sum_{t=1}^{T_j}
\left[
z_{j,t}\log p_{j,t} + (1-z_{j,t})\log(1-p_{j,t})
\right]
$$

这就是 process supervision 相比 outcome supervision 更贵的原因：每个样本不只要一个最终标签，还要多个步骤标签。

#### 5. Value function

在搜索中预测当前 partial solution 最终成功概率。

如果搜索状态是 $h_t$，value verifier 可以写成：

$$
V_\phi(h_t) \approx P(z=1 \mid h_t)
$$

Tree-of-Thought、MCTS 或 beam search 可以用 $V_\phi(h_t)$ 决定扩展、剪枝或回退。这里 verifier 已经不只是“判最终答案”，而是在推理时提供状态价值估计。

面试中不必背复杂公式。

重点是讲清：

```text
Verifier 本质上是在学习一个评价函数，用来判断候选推理或答案的质量。
```

---

### 十二、Verifier 的评估

Verifier 本身也要评估。

不能因为它叫 verifier 就默认可靠。

指标包括：

1. Accuracy。
2. Precision / recall。
3. AUC。
4. Pairwise ranking accuracy。
5. Calibration。
6. 对 hard negatives 的识别率。
7. 和人工标注一致性。
8. 对下游 answer accuracy 的提升。
9. 成本和延迟。

特别重要的是 calibration。

如果 verifier 给错误答案高分，就会误导整个系统。

还要做 end-to-end 评估：

```text
没有 verifier 的准确率
vs 有 verifier rerank 的准确率
vs verifier + search 的准确率
```

Verifier 的价值最终要体现在下游任务上。

几个关键指标可以写得更具体。

候选选择准确率衡量 verifier 是否真的选中正确候选：

$$
A_{\mathrm{select}} =
\frac{1}{M}
\sum_{j=1}^{M}
\mathbf{1}\left[
z_{j,i_j^\star}=1
\right]
$$

其中 $i_j^\star = \arg\max_i S_{j,i}$。注意它不同于 oracle pass@N：

$$
A_{\mathrm{oracle@N}} =
\frac{1}{M}
\sum_{j=1}^{M}
\mathbf{1}\left[
\max_{1\le i\le N} z_{j,i}=1
\right]
$$

`oracle@N` 只说明候选集合里有没有正确解，不说明 verifier 能不能选中它。很多系统失败就是 `oracle@N` 很高，但 reranker 选错。

Pairwise ranking accuracy 衡量好候选是否排在坏候选前：

$$
A_{\mathrm{pair}} =
\frac{1}{K}
\sum_{k=1}^{K}
\mathbf{1}[s_k^+ > s_k^-]
$$

PRM 的第一错步检测可以写成：

$$
A_{\mathrm{firstErr}} =
\frac{1}{M}
\sum_{j=1}^{M}
\mathbf{1}[\hat t_j = t_j^\star]
$$

其中 $t_j^\star$ 是人工或程序标注的第一处错误，$\hat t_j$ 是 PRM 根据阈值预测的第一处错误。

Calibration 可以用 ECE 粗略衡量。把 verifier 置信度分成 $B$ 个桶：

$$
\mathrm{ECE} =
\sum_{b=1}^{B}
\frac{|\mathcal{B}_b|}{M}
\left|
\mathrm{acc}(\mathcal{B}_b)-\mathrm{conf}(\mathcal{B}_b)
\right|
$$

如果高分桶的真实正确率明显低于平均置信度，verifier 就会高置信误选。

端到端收益要同时报告准确率和成本：

$$
\Delta A = A_{\mathrm{rerank}} - A_{\mathrm{base}}
$$

$$
U = A_{\mathrm{rerank}} - \lambda_N N - \lambda_T T - \lambda_C C
$$

这里 $N$ 是采样候选数，$T$ 是延迟，$C$ 是成本。这个效用公式提醒读者：verifier 离线分数再高，如果只带来很小准确率提升却显著增加成本，也未必值得上线。

---

### 十三、最小代码：ORM/PRM rerank、第一错步和代码 verifier

下面的 demo 不训练模型，而是用规则和预设分数模拟真实 verifier 系统中最容易踩的点：

1. ORM 根据最终答案或测试结果打分，可能放过“答案碰巧对、过程错误”的候选。
2. PRM 可以定位步骤错误，但局部步骤分数不等于最终答案一定正确。
3. Hybrid verifier 把 ORM、PRM、模型置信度和成本结合起来做最终选择。
4. `oracle@N` 只说明候选里有正确答案，selection accuracy 才说明系统真的选对。
5. 代码任务里 public tests 可能不够，hidden tests 和程序执行 verifier 比文本打分更可靠。

输入是三道 toy 数学题和两个候选 Python 函数；输出是不同策略的选择准确率、PRM 第一错步、pairwise ranking accuracy 和相对成本。

```python
from collections import defaultdict
from fractions import Fraction
from math import prod


def norm(value):
    text = str(value).strip().rstrip(".")
    try:
        return Fraction(text)
    except ValueError:
        return Fraction(float(text)).limit_denominator(1000)


def process_score(steps):
    probs = []
    for actual, expected in steps:
        probs.append(0.95 if norm(actual) == norm(expected) else 0.05)
    return prod(probs) ** (1 / len(probs)), probs


def first_error(step_probs, threshold=0.5):
    for idx, score in enumerate(step_probs, 1):
        if score < threshold:
            return idx
    return None


def select_by(candidates, score_key):
    return max(candidates, key=lambda item: item[score_key])


problems = [
    {
        "id": "majority_wrong",
        "gold": "20",
        "candidates": [
            {
                "id": "a",
                "answer": "25",
                "model_score": 0.86,
                "tokens": 78,
                "steps": [(30 / 2, 15), (15 + 10, 20)],
            },
            {
                "id": "b",
                "answer": "25",
                "model_score": 0.81,
                "tokens": 74,
                "steps": [(30 / 2, 15), (15 + 10, 20)],
            },
            {
                "id": "c",
                "answer": "20",
                "model_score": 0.70,
                "tokens": 69,
                "steps": [(30 / 2, 15), (15 + 5, 20)],
            },
        ],
    },
    {
        "id": "lucky_answer",
        "gold": "4",
        "candidates": [
            {
                "id": "a",
                "answer": "4",
                "model_score": 0.91,
                "tokens": 92,
                "steps": [(11 - 3, 8), (14 / 2, 4)],
            },
            {
                "id": "b",
                "answer": "4",
                "model_score": 0.75,
                "tokens": 64,
                "steps": [(11 - 3, 8), (8 / 2, 4)],
            },
            {
                "id": "c",
                "answer": "7",
                "model_score": 0.66,
                "tokens": 71,
                "steps": [(11 + 3, 8), (14 / 2, 4)],
            },
        ],
    },
    {
        "id": "hard_negative",
        "gold": "9/2",
        "candidates": [
            {
                "id": "a",
                "answer": "4.5",
                "model_score": 0.73,
                "tokens": 82,
                "steps": [(18 / 4, 4.5), (4.5, 4.5)],
            },
            {
                "id": "b",
                "answer": "5",
                "model_score": 0.89,
                "tokens": 88,
                "steps": [(18 / 3, 4.5), (6 - 1, 4.5)],
            },
            {
                "id": "c",
                "answer": "9/2",
                "model_score": 0.78,
                "tokens": 79,
                "steps": [(18 / 4, 4.5), (Fraction(9, 2), Fraction(9, 2))],
            },
        ],
    },
]


def add_scores(problem):
    gold = norm(problem["gold"])
    for cand in problem["candidates"]:
        cand["answer_correct"] = norm(cand["answer"]) == gold
        cand["orm_score"] = 1.0 if cand["answer_correct"] else 0.0
        cand["prm_score"], cand["step_probs"] = process_score(cand["steps"])
        cand["first_error"] = first_error(cand["step_probs"])
        cand["cost"] = cand["tokens"] / 100.0
        cand["hybrid_score"] = (
            0.40 * cand["orm_score"]
            + 0.40 * cand["prm_score"]
            + 0.20 * cand["model_score"]
            - 0.03 * cand["cost"]
        )


metrics = defaultdict(int)
pair_total = pair_model_hit = pair_hybrid_hit = 0
first_errors = []

for problem in problems:
    add_scores(problem)
    cands = problem["candidates"]
    gold_exists = any(item["answer_correct"] for item in cands)
    metrics["oracle@N"] += gold_exists
    metrics["model_select"] += select_by(cands, "model_score")["answer_correct"]
    metrics["orm_select"] += select_by(cands, "orm_score")["answer_correct"]
    metrics["hybrid_select"] += select_by(cands, "hybrid_score")["answer_correct"]
    first_errors.extend(item["first_error"] for item in cands)

    for good in [item for item in cands if item["answer_correct"]]:
        for bad in [item for item in cands if not item["answer_correct"]]:
            pair_total += 1
            pair_model_hit += good["model_score"] > bad["model_score"]
            pair_hybrid_hit += good["hybrid_score"] > bad["hybrid_score"]


def buggy_even_count(nums):
    return sum(1 for num in nums if num % 2 == 0 and num > 0)


def fixed_even_count(nums):
    return sum(1 for num in nums if num % 2 == 0)


public_tests = [([1, 2, 3, 4], 2), ([2, 8], 2)]
hidden_tests = [([-2, -1, 0, 3], 2), ([0, 0, 5], 2)]


def test_score(fn, tests):
    passed = 0
    for args, expected in tests:
        passed += fn(args) == expected
    return passed / len(tests)


code_scores = {
    "buggy_public": test_score(buggy_even_count, public_tests),
    "buggy_hidden": test_score(buggy_even_count, hidden_tests),
    "fixed_public": test_score(fixed_even_count, public_tests),
    "fixed_hidden": test_score(fixed_even_count, hidden_tests),
}

n = len(problems)
print("math_summary=", {
    "oracle@N": round(metrics["oracle@N"] / n, 3),
    "model_select_acc": round(metrics["model_select"] / n, 3),
    "orm_select_acc": round(metrics["orm_select"] / n, 3),
    "hybrid_select_acc": round(metrics["hybrid_select"] / n, 3),
    "model_pairwise_acc": round(pair_model_hit / pair_total, 3),
    "hybrid_pairwise_acc": round(pair_hybrid_hit / pair_total, 3),
    "first_error_index": first_errors,
    "relative_serial_cost": 3 * (1.0 + 0.25),
    "relative_parallel_latency": 1.0 + 0.25,
})
print("code_verifier=", code_scores)
```

一组输出如下：

```text
math_summary= {'oracle@N': 1.0, 'model_select_acc': 0.333, 'orm_select_acc': 1.0, 'hybrid_select_acc': 1.0, 'model_pairwise_acc': 0.333, 'hybrid_pairwise_acc': 1.0, 'first_error_index': [2, 2, None, 2, None, 1, None, 1, None], 'relative_serial_cost': 3.75, 'relative_parallel_latency': 1.25}
code_verifier= {'buggy_public': 1.0, 'buggy_hidden': 0.0, 'fixed_public': 1.0, 'fixed_hidden': 1.0}
```

这组结果有三个面试价值：第一，`oracle@N=1.0` 不代表系统能选中正确候选，`model_select_acc=0.333` 就是反例；第二，`lucky_answer` 里 ORM 会认可最终答案正确的候选，但 PRM 能指出第 2 步错误；第三，代码候选通过 public tests 不等于可靠，hidden tests 才暴露了正数偶数计数这种边界条件 bug。

---

### 十四、失败模式

#### 1. Verifier 被候选答案欺骗

候选推理写得很像正确，verifier 打高分。

#### 2. Reward hacking

生成模型学会迎合 verifier 的偏好，而不是解决问题。

例如写更长、更像数学证明的文本骗分。

#### 3. PRM 标注不一致

不同标注者对某一步是否合理看法不同。

#### 4. ORM 奖励碰巧正确

推理错误但答案对，ORM 仍给高分。

#### 5. 对分布外问题失效

Verifier 在训练分布上好，遇到新题型不可靠。

#### 6. 过度惩罚非标准解法

有些正确推理路径和训练样本不同，被 PRM 打低分。

#### 7. 计算成本高

每个候选都调用 verifier，成本和延迟增加。

---

### 十五、真实项目中的坑

#### 1. 把 LLM judge 当绝对真理

LLM verifier 也会幻觉，也会偏。

#### 2. 只训练 ORM，不看过程

最终答案正确率提升了，但模型过程可能更不可靠。

#### 3. PRM 步骤粒度混乱

一步太大无法定位错误，一步太小标注成本爆炸。

#### 4. 没有 hard negatives

Verifier 只会区分明显错误，区分不了高迷惑错误。

#### 5. 不评估下游收益

Verifier 离线 AUC 高，不代表能提升最终系统准确率。

#### 6. 忽略成本

Verifier 让准确率提升 1%，但延迟翻倍，线上未必可接受。

#### 7. 让生成模型和 verifier 共同过拟合

长期优化同一个 verifier，可能出现 reward hacking。

---

### 十六、面试问答

#### 问题 1：Verifier 在 reasoning model 中的作用是什么？

可以这样回答：

```text
Verifier 是评价候选答案或推理过程质量的模块。它可以检查最终答案、推理步骤或中间状态，帮助从多条 CoT、搜索路径或候选答案中选择更可靠的结果。
```

#### 问题 2：ORM 和 PRM 的区别是什么？

可以这样回答：

```text
ORM 评价最终答案，看结果是否正确；PRM 评价推理过程中的每一步，看每一步是否合理。ORM 标注较容易但不能定位错误，PRM 标注更贵但能提供细粒度过程监督。
```

#### 问题 3：为什么只看最终答案不够？

可以这样回答：

```text
因为最终答案可能碰巧正确，也可能答案错但前面大部分推理正确。只看 outcome 无法定位错误步骤，也难以指导搜索和过程改进，所以复杂 reasoning 中需要过程监督或 verifier。
```

#### 问题 4：PRM 为什么难？

可以这样回答：

```text
PRM 需要对每个推理步骤标注质量，成本高，而且要定义步骤粒度、处理多种正确解法、保证标注一致性，并避免模型学会迎合 PRM 而不是真正推理。
```

#### 问题 5：Verifier 如何和 self-consistency 结合？

可以这样回答：

```text
可以先采样多条 CoT，再用 majority vote 得到候选，也可以让 verifier 给每条路径打分后选择最高分，或者先过滤低质量路径再投票。Verifier 弥补了多数投票无法判断推理质量的问题。
```

#### 问题 6：如何评估 verifier？

可以这样回答：

```text
既要评估 verifier 自身的 accuracy、AUC、ranking accuracy、calibration 和 hard negative 识别能力，也要评估它是否提升下游任务准确率、稳定性和成本收益。
```

---

### 十七、常见误区

1. 误区：Verifier 一定比 generator 更可靠。
   纠正：Verifier 也会错，也要评估和校准。

2. 误区：ORM 足够解决 reasoning。
   纠正：ORM 只看结果，不能定位过程错误。

3. 误区：PRM 标注越细越好。
   纠正：过细会增加成本和噪声，要选择合适步骤粒度。

4. 误区：Verifier 分数高就一定正确。
   纠正：可能存在 reward hacking 或分布外失效。

5. 误区：LLM-as-a-verifier 不需要人工校准。
   纠正：LLM judge 需要标准集、人工抽查和一致性评估。

6. 误区：只看 verifier 离线指标。
   纠正：最终要看 end-to-end 任务收益和成本。

---

### 十八、小练习

1. 给一个数学题设计 ORM 输入输出样本。
2. 给一个三步推理过程设计 PRM step labels。
3. 构造一个最终答案正确但中间推理错误的例子。
4. 构造一个最终答案错误但前两步正确的例子。
5. 设计一个 CoT + verifier rerank 流程。
6. 比较 rule-based verifier、programmatic verifier 和 neural verifier。
7. 设计一组 hard negatives，用于训练数学 verifier。
8. 设计 verifier 的离线指标和端到端评估指标。
9. 分析 reward hacking 在 verifier 系统中如何发生。
10. 用 3 分钟回答：“ORM 和 PRM 的区别与取舍是什么？”

### 本讲总结

本讲最重要的结论：

1. Verifier 用来评价候选答案、推理过程或中间状态的质量。
2. ORM 看最终结果，PRM 看推理过程。
3. ORM 标注更容易，但不能定位过程错误；PRM 更细粒度，但标注更难。
4. Verifier 可以和 CoT、self-consistency、search 结合，用于筛选和改进候选推理。
5. PRM 的关键难点是步骤粒度、过程标注、标注一致性和 reward hacking。
6. Verifier 本身也需要评估，包括准确率、排序能力、校准和下游收益。
7. 真实系统常用规则、程序、神经模型和人工审核的混合 verifier。
8. 面试中要强调 verifier 是 reasoning system 的评价函数，而不是天然可靠的真理源。

## 第 89 讲：Search、Tree-of-Thought 与 MCTS

### 本讲目标

学完本讲，你应该能回答六个问题：

1. 为什么 reasoning 不只是生成一条 CoT，还可以做搜索。
2. Tree-of-Thought 想解决什么问题，和 Chain-of-Thought 有什么区别。
3. Search 中的 state、action、value、policy 分别是什么。
4. BFS、DFS、beam search、MCTS 在 reasoning 中如何理解。
5. Verifier/PRM 如何作为搜索中的评价函数。
6. 面试中如何设计一个用搜索增强数学、代码或规划推理的系统。

第 86 讲讲 CoT：生成一条推理链。

第 87 讲讲 self-consistency：采样多条推理链后投票。

第 88 讲讲 verifier：判断候选答案或推理过程质量。

这一讲进一步问：

```text
能不能把推理过程显式组织成搜索？
```

人类解决难题时经常不是一条路走到底。

我们会尝试多个思路，发现不对就回退，比较不同路径，再继续深入。

Search、Tree-of-Thought 和 MCTS 就是在大模型 reasoning 中引入这种“探索、评价、选择、回退”的思想。

从资料脉络看，Tree of Thoughts 的关键不是“多写几条 CoT”，而是把 token 级从左到右生成提升为 thought 级搜索：每个 thought 是一个可继续展开、可评价的中间单元。ToT 论文强调探索多条 reasoning paths、对中间 thought 做自评、必要时 lookahead 或 backtracking；LATS 进一步把 MCTS、value function、自反思和环境反馈放到 agent 决策中；另一些 ToT 系统则强调 checker、memory 和 controller。面试中要抓住这条主线：search 是结构化 test-time compute，核心是状态、动作、评分、剪枝、回退和预算。

---

### 一、为什么需要搜索

单条 CoT 的问题是：

```text
一旦早期步骤错了，后面可能沿着错误方向继续生成。
```

Self-consistency 虽然采样多条完整路径，但它通常是“先生成完整答案，再比较”。

它没有在中间步骤主动剪枝或调整。

搜索的目标是：

```text
把推理看成一系列状态转移，在中间阶段评估哪些路径更有希望，然后优先探索好路径，剪掉差路径。
```

例如解题时：

1. 先列出几个可能解法。
2. 对每个解法走几步。
3. 发现某条路矛盾，就停止。
4. 发现某条路有希望，就继续。
5. 最后选择最好的完整解。

这比盲目生成 N 条完整 CoT 更有结构。

---

### 二、把推理建模成搜索问题

搜索问题通常有几个要素。

#### 1. State

当前状态。

在 reasoning 中，state 可以是当前已经生成的部分推理。

例如：

```text
已知条件：...
Step 1: ...
Step 2: ...
```

#### 2. Action

下一步动作。

在 LLM reasoning 中，action 可以是生成下一步思路、调用工具、选择公式、执行代码。

#### 3. Transition

从一个状态到下一个状态。

例如模型生成下一步推理后，状态更新。

#### 4. Value

当前状态有多好。

可以由 verifier、PRM、规则、工具执行结果或模型自评给出。

#### 5. Policy

选择下一步动作的策略。

通常由 LLM 生成候选动作。

#### 6. Terminal state

推理结束状态。

例如得到最终答案、通过测试、满足目标或达到最大步数。

用公式化语言说：

$$
s_{t+1} = T(s_t, a_t, o_t)
$$

其中 $s_t$ 是当前推理状态，$a_t$ 是下一步动作，$o_t$ 是可选的 observation，例如工具执行结果、测试反馈或 verifier 反馈。Policy 负责生成候选动作：

$$
a_t \sim \pi_\theta(a \mid s_t)
$$

Value 或 verifier 负责评价状态：

$$
V_\phi(s_t) \approx P(\mathrm{success} \mid s_t)
$$

最终目标是找到一条高质量轨迹：

$$
\tau = (s_0,a_0,s_1,a_1,\ldots,s_T)
$$

使得终局奖励尽量高：

$$
\tau^\star = \arg\max_\tau R(s_T)
$$

这就是“reasoning = 在状态空间中搜索一条从问题到答案的高质量路径”的数学化表达。

---

### 三、Chain-of-Thought 和 Tree-of-Thought

CoT 是一条链。

结构是：

```text
Step 1 -> Step 2 -> Step 3 -> Answer
```

Tree-of-Thought，简称 ToT，是一棵树。

结构是：

```text
                 Problem
              /     |     \
          Thought A B C
          /   \       \
       A1    A2       C1
       |              |
     Answer        Answer
```

CoT 一次只保留一条路径。

ToT 同时维护多个候选思路。

ToT 的核心思想是：

```text
把中间 thought 当成搜索节点，让模型生成多个候选 thought，再评价和选择哪些 thought 继续展开。
```

这让模型可以：

1. 同时探索多个方向。
2. 中途比较路径质量。
3. 剪掉明显错误的分支。
4. 对困难问题做更系统的推理。

---

### 四、Tree-of-Thought 基本流程

一个简化 ToT 流程是：

```text
Input question
Initialize root state

for depth in 1..D:
    Expand: 对每个当前状态生成 K 个候选 thought
    Evaluate: 给每个新状态打分
    Select: 保留最好的 B 个状态

Return best final answer
```

其中：

1. `D` 是最大深度。
2. `K` 是每个节点展开多少个候选。
3. `B` 是每层保留多少个状态。

这个流程很像 beam search。

如果第 $d$ 层的 frontier 是 $\mathcal{F}_d$，每个状态生成 $K$ 个候选 thought，可以写成：

$$
\mathcal{C}_{d+1} =
\{T(s,a) \mid s \in \mathcal{F}_d, a \in \pi_K(s)\}
$$

其中 $\pi_K(s)$ 表示从当前状态生成的 top-K 或采样 K 个候选动作。再用 verifier/value 选出下一层 frontier：

$$
\mathcal{F}_{d+1} =
\operatorname{TopB}_{s \in \mathcal{C}_{d+1}} V_\phi(s)
$$

这个公式说明 ToT 至少包含三件事：展开候选、评价候选、选择保留。只生成多条完整 CoT 而没有中间选择，不是严格意义上的 tree search。

例如在数学题中：

1. 第 1 层生成不同解法方向。
2. 第 2 层对每个方向继续推导。
3. 第 3 层检查是否得到答案。
4. Verifier 给每条路径打分。
5. 选择最好答案。

---

### 五、BFS、DFS 和 Beam Search

搜索策略决定如何探索树。

#### 1. BFS

BFS 是广度优先。

它按层展开。

优点是覆盖广。

缺点是节点数量增长快。

在 ToT 中，BFS 适合先探索多个不同思路。

#### 2. DFS

DFS 是深度优先。

它沿着一条路径走到底，再回退。

优点是内存少。

缺点是容易在坏路径上走太深。

在 reasoning 中，DFS 类似“先尝试一个完整解法”。

#### 3. Beam Search

Beam search 每层只保留 top-B 个候选。

它是质量和成本的折中。

在 LLM reasoning 中很常用：

```text
每一步生成多个候选 -> 用 verifier 打分 -> 保留前 B 个继续
```

缺点是如果早期打分不准，正确路径可能被剪掉。

Beam search 的递推可以写成：

$$
\mathcal{B}_{d+1} =
\operatorname{TopB}_{s' \in \mathrm{Expand}(\mathcal{B}_d)}
\left[V_\phi(s') - \lambda_c C(s')\right]
$$

其中 $C(s')$ 可以是 token、延迟、工具调用或安全风险成本。这个成本项很重要，因为 search 不是只追求分数最高，也要避免长而无用的路径占满预算。

---

### 六、MCTS 的直觉

MCTS 是 Monte Carlo Tree Search，蒙特卡洛树搜索。

它在围棋、游戏 AI 和规划中很常见。

MCTS 的核心是平衡：

```text
探索还不确定但可能好的分支，利用已经看起来不错的分支。
```

经典 MCTS 包含四步：

#### 1. Selection

从根节点出发，选择一个值得继续探索的节点。

#### 2. Expansion

展开这个节点，生成新的候选动作或 thought。

#### 3. Simulation / Rollout

从新节点继续模拟到终局，得到一个结果。

#### 4. Backpropagation

把结果分数回传，更新路径上节点的价值估计。

在 LLM reasoning 中，可以理解为：

1. 选择一个有希望的 partial reasoning。
2. 让模型生成下一步。
3. 继续生成到答案或用 verifier 估计成功率。
4. 把答案质量回传给前面的 thought。

---

### 七、MCTS 中的 UCB 直觉

MCTS 常用 UCB 类公式选择节点。

不用背公式，但要理解 trade-off。

一个节点被选择，取决于：

1. 它过去表现好不好。
2. 它探索次数够不够。

如果只利用高分节点，可能错过潜在好路径。

如果只探索新节点，成本会很高。

所以 MCTS 做的是：

```text
在 exploitation 和 exploration 之间平衡。
```

经典 UCB/UCT 选择可以写成：

$$
\mathrm{UCB}(s,a) =
Q(s,a) + c\sqrt{\frac{\log N(s)}{N(s,a)+\epsilon}}
$$

其中 $Q(s,a)$ 是动作历史平均回报，$N(s)$ 是状态访问次数，$N(s,a)$ 是该动作被尝试次数，$c$ 控制探索强度。$c$ 太小会过早利用当前高分路径，$c$ 太大则会过度探索。

当一次 rollout 得到终局奖励 $r$ 后，可以用增量平均更新节点价值：

$$
Q_{\mathrm{new}}(s,a) =
Q_{\mathrm{old}}(s,a) +
\frac{r-Q_{\mathrm{old}}(s,a)}{N(s,a)}
$$

在 LLM reasoning 中，$r$ 可以来自最终答案是否正确、单元测试通过率、工具验证、ORM/PRM 分数或人工 judge。风险是：如果 reward 本身不可靠，MCTS 会把噪声系统性回传到树上。

在 reasoning 中，这对应：

1. 继续深入当前看起来最好的思路。
2. 也给其他可能思路一些探索机会。

这比单纯 beam search 更灵活。

---

### 八、LLM Search 中的 Policy 和 Value

搜索需要两个能力。

#### 1. Policy：生成候选

LLM 可以作为 policy。

给定当前状态，让模型生成下一步 thought。

例如：

```text
基于当前推理，请给出 3 个可能的下一步。
```

#### 2. Value：评价候选

Verifier、PRM、ORM、规则、工具执行结果可以作为 value。

例如：

```text
这个 partial solution 最终成功概率是多少？
```

好的 reasoning search 不是只靠生成。

它需要：

```text
强 policy 生成好候选 + 强 value 识别好候选
```

如果 policy 差，根本生成不到正确路径。

如果 value 差，正确路径会被剪掉。

---

### 九、Search 和 Verifier 的关系

第 88 讲说过，verifier 是评价函数。

Search 离不开评价函数。

Verifier 可以用于：

1. 给完整答案打分。
2. 给中间状态打分。
3. 剪掉错误路径。
4. 选择最优候选。
5. 指导下一步搜索。

如果有 PRM，可以每一步评价。

如果只有 ORM，只能在生成完整答案后评价。

所以 PRM 更适合搜索。

但 PRM 训练更难。

这就是 trade-off。

可以把剪枝写成一个 gate：

$$
g(s) = \mathbf{1}[V_\phi(s) \ge \tau_v]\cdot \mathbf{1}[C(s) \le B_{\mathrm{remain}}]
$$

只有 $g(s)=1$ 的状态才继续展开。这个 gate 看起来简单，但它决定了 search 的成败：阈值太低会成本爆炸，阈值太高会提前剪掉正确路径。

---

### 十、Search 的适用场景

Search 适合：

1. 数学证明。
2. 复杂逻辑题。
3. 代码生成和修复。
4. 规划任务。
5. 多步工具调用。
6. 策略游戏。
7. 需要显式试错的问题。

Search 不适合所有任务。

不适合：

1. 简单问答。
2. 低延迟聊天。
3. 答案主观、无法评价的问题。
4. 搜索空间极大但没有好 verifier 的问题。

没有好的评价函数时，search 可能只是更贵的随机游走。

---

### 十一、代码推理中的搜索

代码任务很适合 search，因为可以执行测试。

流程可以是：

```text
生成多个候选修复
-> 运行单元测试
-> 根据失败信息继续修改
-> 保留通过更多测试的候选
-> 重复直到通过或达到预算
```

这里的 verifier 是测试。

测试结果提供很强反馈。

例如：

1. 编译是否通过。
2. 单元测试通过多少。
3. 哪个边界用例失败。
4. 性能是否超时。

这也是为什么代码 agent 常常比纯自然语言推理更容易做闭环。

因为代码可以运行验证。

---

### 十二、数学推理中的搜索

数学推理也适合 search，但 verifier 更难。

简单算术可以用程序验证。

但复杂证明、几何题、开放式推导很难自动验证。

常见方案包括：

1. 用符号计算器检查代数步骤。
2. 用 PRM 判断推理步骤。
3. 让多个模型互相检查。
4. 对最终答案做代入验证。
5. 使用形式化证明系统。

数学 search 的关键是：

```text
搜索空间大，必须有足够可靠的中间评价。
```

否则会生成大量看起来像证明但实际错误的路径。

---

### 十三、成本和预算控制

Search 比 self-consistency 更容易成本失控。

因为节点数可能指数增长。

如果每个节点展开 K 个候选，深度 D：

$$
N_{\mathrm{full}}(K,D) =
\sum_{d=0}^{D}K^d =
\frac{K^{D+1}-1}{K-1}
$$

如果使用 beam size $B$，每层最多保留 $B$ 个状态，展开节点数可粗略估算为：

$$
N_{\mathrm{beam}} \le 1 + D\cdot B\cdot K
$$

如果每个状态平均生成 $\bar n$ 个 token、调用 $\bar m$ 次 verifier 或工具，成本可以写成：

$$
C_{\mathrm{search}} =
N_{\mathrm{expand}}(c_{\mathrm{tok}}\bar n + c_v\bar m)
$$

预算约束则是：

$$
C_{\mathrm{search}} \le B_{\mathrm{cost}},\quad
T_{\mathrm{search}} \le B_{\mathrm{time}}
$$

所以必须设置预算：

1. 最大深度。
2. 最大节点数。
3. 最大 token。
4. 最大工具调用次数。
5. 最大运行时间。
6. 提前停止条件。

Beam search 用 beam size 控制宽度。

MCTS 用 simulation budget 控制探索次数。

生产系统必须把搜索预算作为一等公民。

---

### 十四、失败模式

#### 1. Search space explosion

分支太多，成本爆炸。

#### 2. Bad value function

Verifier 评分不准，剪掉正确路径。

#### 3. Goodhart / reward hacking

模型生成迎合 verifier 的路径，而不是真正解决问题。

#### 4. Lack of diversity

展开的候选都很相似，搜索没有意义。

#### 5. Premature pruning

早期看起来分数低但后面会变好的路径被剪掉。

#### 6. Looping

搜索反复生成相同或等价状态。

#### 7. No reliable terminal check

不知道什么时候已经得到正确答案。

---

### 十五、评估方法

Search-based reasoning 要评估：

1. 最终准确率。
2. Solve rate。
3. 平均节点数。
4. 平均 token。
5. 平均工具调用次数。
6. 平均延迟。
7. 每题成本。
8. Search depth。
9. Verifier 剪枝准确率。
10. 不同预算下的收益曲线。

要比较：

```text
single CoT
self-consistency
beam search
MCTS
search + verifier
```

重点不是 search 一定更强。

重点是看：

```text
多花的 test-time compute 是否换来了足够收益。
```

可以补充三个搜索特有指标。

剪枝保真率：

$$
A_{\mathrm{keep}} =
\frac{1}{M}
\sum_{j=1}^{M}
\mathbf{1}[k_j=1]
$$

其中 $k_j=1$ 表示第 $j$ 个样本的某条正确路径在剪枝后仍被保留。

节点效率：

$$
E_{\mathrm{node}} =
\frac{\mathrm{solved}}{N_{\mathrm{expand}}}
$$

预算收益：

$$
\Delta A(B) = A(B)-A(B_0)
$$

如果 $A_{\mathrm{keep}}$ 很低，说明 verifier 或剪枝策略在早期误杀正确路径；如果 $E_{\mathrm{node}}$ 很低，说明搜索展开很多但有效探索很少。

---

### 十六、最小代码：Beam Search 早剪风险与 MCTS 探索

下面的 demo 用一个固定搜索图模拟 reasoning search。`shortcut` 分支一开始 verifier 分数很高，但终局错误；`factor` 分支初始分数较低，但继续展开后得到正确答案。它展示三个点：

1. `beam_size=1` 会被早期高分误导，剪掉正确路径。
2. `beam_size=2` 保留更多分支后能找到正确答案。
3. MCTS/UCB 会给低访问分支探索机会，从而把最佳根动作转向 `factor`。

```python
from math import log, sqrt


GOLD = 24
GRAPH = {
    "root": [
        {"label": "shortcut", "next": "shortcut", "value": 0.92, "tokens": 18},
        {"label": "factor", "next": "factor", "value": 0.70, "tokens": 24},
        {"label": "bruteforce", "next": "bruteforce", "value": 0.35, "tokens": 40},
    ],
    "shortcut": [
        {"label": "confident_finish", "next": "wrong_18", "value": 0.88, "tokens": 22},
    ],
    "factor": [
        {"label": "derive_invariant", "next": "correct_24", "value": 0.76, "tokens": 30},
        {"label": "format_slip", "next": "wrong_42", "value": 0.61, "tokens": 20},
    ],
    "bruteforce": [
        {"label": "enumerate_cases", "next": "correct_24", "value": 0.64, "tokens": 70},
    ],
}
TERMINALS = {"wrong_18": 18, "wrong_42": 42, "correct_24": 24}


def is_terminal(state):
    return state in TERMINALS


def terminal_reward(state):
    return 1.0 if TERMINALS.get(state) == GOLD else 0.0


def rank_score(state, value, tokens):
    reward = terminal_reward(state) if is_terminal(state) else 0.0
    return value + 1.50 * reward - 0.002 * tokens


def beam_search(beam_size, max_depth=2):
    frontier = [{"state": "root", "path": [], "score": 0.0, "tokens": 0}]
    generated = 0
    pruned = 0
    trace = []
    for _ in range(max_depth):
        candidates = []
        for item in frontier:
            if is_terminal(item["state"]):
                candidates.append(item)
                continue
            for action in GRAPH[item["state"]]:
                generated += 1
                tokens = item["tokens"] + action["tokens"]
                score = rank_score(action["next"], action["value"], tokens)
                candidates.append({
                    "state": action["next"],
                    "path": item["path"] + [action["label"]],
                    "score": score,
                    "tokens": tokens,
                })
        candidates.sort(key=lambda item: item["score"], reverse=True)
        pruned += max(0, len(candidates) - beam_size)
        frontier = candidates[:beam_size]
        trace.append([item["state"] for item in frontier])
    best = max(frontier, key=lambda item: item["score"])
    return {
        "answer": TERMINALS.get(best["state"]),
        "success": terminal_reward(best["state"]) == 1.0,
        "path": best["path"],
        "generated": generated,
        "pruned": pruned,
        "trace": trace,
    }


def rollout(root_action):
    state = root_action["next"]
    path = [root_action["label"]]
    while not is_terminal(state):
        action = max(GRAPH[state], key=lambda item: item["value"])
        state = action["next"]
        path.append(action["label"])
    return terminal_reward(state), path


def mcts_root(num_rollouts=9, c=1.4):
    children = [dict(action, visits=0, reward_sum=0.0) for action in GRAPH["root"]]
    total_visits = 0
    for _ in range(num_rollouts):
        def ucb(child):
            if child["visits"] == 0:
                return float("inf")
            mean = child["reward_sum"] / child["visits"]
            explore = c * sqrt(log(total_visits + 1) / child["visits"])
            prior = 0.05 * child["value"]
            return mean + explore + prior

        child = max(children, key=ucb)
        reward, _ = rollout(child)
        child["visits"] += 1
        child["reward_sum"] += reward
        total_visits += 1
    best = max(children, key=lambda item: (item["reward_sum"] / item["visits"], item["value"]))
    return {
        "best_action": best["label"],
        "mean_reward": round(best["reward_sum"] / best["visits"], 3),
        "visits": {item["label"]: item["visits"] for item in children},
    }


beam1 = beam_search(beam_size=1)
beam2 = beam_search(beam_size=2)
mcts = mcts_root()
print("beam1=", beam1)
print("beam2=", beam2)
print("mcts=", mcts)
```

一组输出如下：

```text
beam1= {'answer': 18, 'success': False, 'path': ['shortcut', 'confident_finish'], 'generated': 4, 'pruned': 2, 'trace': [['shortcut'], ['wrong_18']]}
beam2= {'answer': 24, 'success': True, 'path': ['factor', 'derive_invariant'], 'generated': 6, 'pruned': 2, 'trace': [['shortcut', 'factor'], ['correct_24', 'wrong_18']]}
mcts= {'best_action': 'factor', 'mean_reward': 1.0, 'visits': {'shortcut': 1, 'factor': 4, 'bruteforce': 4}}
```

这个 toy 例子故意让 `shortcut` 初始分数最高。它说明 search 的关键不是“展开越多越好”，而是用足够宽度、足够多样性和足够可靠的终局反馈，避免早期 verifier 分数把正确路径剪掉。

---

### 十七、真实项目中的坑

#### 1. 没有 verifier 就上 search

没有可靠评价函数，search 只是放大随机性。

#### 2. 展开粒度太细

每个 token 都作为搜索动作，成本太高。

通常按 thought、step、function call 粒度更合理。

#### 3. 展开粒度太粗

一步生成完整答案，又退回 self-consistency。

#### 4. 不去重

大量等价 thought 重复展开。

#### 5. 不设预算

线上延迟和费用不可控。

#### 6. 只报告最优结果

不报告搜索成本，容易夸大方法价值。

#### 7. Verifier 和 generator 互相过拟合

长期优化同一个 verifier，模型可能学会骗分。

---

### 十八、面试问答

#### 问题 1：Tree-of-Thought 和 Chain-of-Thought 有什么区别？

可以这样回答：

```text
Chain-of-Thought 是一条线性推理链，Tree-of-Thought 把中间 thought 组织成树，允许模型同时探索多个推理分支，并通过 verifier 或 value function 选择哪些分支继续展开。
```

#### 问题 2：为什么 reasoning 需要 search？

可以这样回答：

```text
复杂问题可能有多条解法，单条 CoT 容易早期走错。Search 可以在中间阶段探索多个方向、评价 partial solution、剪枝差路径，并把 test-time compute 用在更有希望的路径上。
```

#### 问题 3：MCTS 的核心思想是什么？

可以这样回答：

```text
MCTS 通过 selection、expansion、simulation 和 backpropagation 在搜索树上迭代探索，核心是平衡 exploitation 和 exploration，也就是既深入好路径，也探索不确定但可能有潜力的路径。
```

#### 问题 4：Verifier 在 search 中起什么作用？

可以这样回答：

```text
Verifier 或 PRM 可以作为 value function，给中间状态或完整答案打分，帮助选择、剪枝和排序候选路径。没有可靠 verifier，search 很容易变成高成本随机探索。
```

#### 问题 5：Search-based reasoning 的主要代价是什么？

可以这样回答：

```text
主要代价是节点数、token、工具调用、延迟和费用增加，而且如果评价函数不准，可能剪掉正确路径或强化错误路径。因此必须设置预算并画成本收益曲线。
```

#### 问题 6：如何设计一个代码修复 search 系统？

可以这样回答：

```text
让 LLM 生成多个候选修复，运行测试作为 verifier，根据失败信息继续展开或修改候选，用通过测试数和错误类型评分，保留 top candidates，设置最大轮数、时间和工具调用预算，最后输出通过测试且改动最小的修复。
```

---

### 十九、常见误区

1. 误区：Search 一定比 CoT 强。
   纠正：没有好的评价函数和预算控制，search 可能只是更贵。

2. 误区：Tree-of-Thought 就是多生成几条 CoT。
   纠正：ToT 的重点是中间 thought 节点的展开、评价和选择。

3. 误区：Beam search 总能保留正确路径。
   纠正：如果早期评分不准，正确路径可能被剪掉。

4. 误区：MCTS 只适合游戏。
   纠正：只要能定义状态、动作和价值估计，就可以借鉴 MCTS 思想。

5. 误区：搜索深度越深越好。
   纠正：深度越大成本越高，错误也可能累积。

6. 误区：搜索结果只看准确率。
   纠正：必须同时看 token、延迟、节点数、工具调用和成本。

---

### 二十、小练习

1. 把一个数学题的求解过程建模成 state、action、value、terminal state。
2. 画出一个三层 Tree-of-Thought 示例。
3. 比较 CoT、self-consistency、beam search 和 MCTS。
4. 设计一个 beam search reasoning 流程，包含 beam size 和评分函数。
5. 设计一个 MCTS reasoning 流程，包含 selection、expansion、rollout 和 backpropagation。
6. 为代码修复任务设计 search + test verifier 系统。
7. 构造一个 verifier 错误剪枝导致失败的案例。
8. 设计一个搜索预算，包括最大深度、节点数、token 和时间。
9. 画一条不同 search budget 下准确率和成本的曲线。
10. 用 3 分钟回答：“Tree-of-Thought 和 MCTS 如何增强 LLM reasoning？”

### 本讲总结

本讲最重要的结论：

1. Search 把 reasoning 看成在状态空间中寻找高质量推理路径。
2. CoT 是一条链，Tree-of-Thought 是多分支推理树。
3. Search 需要 state、action、transition、value、policy 和 terminal state。
4. Beam search 保留每层 top candidates，MCTS 在探索和利用之间平衡。
5. LLM 可以作为 policy 生成候选 thought，verifier/PRM 可以作为 value 评价候选。
6. Search 特别适合数学、代码、规划等可验证或可分解任务。
7. Search 的主要风险是成本爆炸、评价函数错误、过早剪枝和 reward hacking。
8. 面试中要强调：search 是 test-time compute scaling 的结构化形式，关键在评价函数和预算控制。

## 第 90 讲：Test-Time Compute Scaling

### 本讲目标

学完本讲，你应该能回答六个问题：

1. Test-time compute scaling 是什么，和训练时 scaling 有什么区别。
2. 为什么更多推理时计算可能提升 reasoning 能力。
3. CoT、self-consistency、verifier、search 如何共同构成 test-time compute scaling。
4. 哪些任务适合花更多推理时计算，哪些任务不适合。
5. 如何评估 test-time compute 的成本收益曲线。
6. 面试中如何设计一个按任务难度动态分配推理预算的系统。

前面几讲其实都在讲 test-time compute。

第 86 讲 CoT：让模型生成更多中间 token。

第 87 讲 self-consistency：采样多条推理路径。

第 88 讲 verifier：对候选答案和过程打分。

第 89 讲 search：把推理组织成树搜索。

这些方法共同指向一个趋势：

```text
不只靠更大的模型和更多训练数据，也可以在推理时投入更多计算来换取更强表现。
```

这就是 test-time compute scaling。

从方法谱系看，test-time compute 不是突然出现的新概念。第 87 讲的 self-consistency 已经把“单次贪心解码”改成“多条 reasoning path 采样后聚合”；第 88 讲的 verifier 把“能生成”扩展成“能选择”；第 89 讲的 search 把“多候选”扩展成“有结构地探索状态空间”。近年的 test-time scaling 论文进一步强调两点：第一，重复采样在代码、形式证明、数学等可自动验证任务上主要提升 coverage，也就是候选集合中至少有一个正确解的概率；第二，不同难度题目对额外 compute 的收益差异很大，因此生产系统更关心 compute-optimal allocation，而不是所有请求都套同一个 best-of-N。

所以本讲不要把 test-time compute 理解成“多问几遍模型”。更准确地说，它是一个预算分配问题：

```text
把有限的 token、采样、搜索、验证、工具调用和等待时间，分配给最值得多算的请求。
```

---

### 一、什么是 Test-Time Compute Scaling

Test-time compute scaling 指的是：

```text
在模型参数固定的情况下，推理阶段投入更多计算、更多 token、更多采样、更多验证或更多搜索，以提升最终答案质量。
```

它和训练时 scaling 不同。

训练时 scaling 关注：

1. 更多参数。
2. 更多训练 token。
3. 更多训练 FLOPs。
4. 更长训练时间。

Test-time scaling 关注：

1. 推理时生成多少 token。
2. 采样多少条路径。
3. 是否调用 verifier。
4. 是否做 search。
5. 是否调用工具。
6. 是否反思、修正、重试。

一句话：

```text
训练时 scaling 是把能力压进模型参数里；test-time scaling 是在使用模型时花更多计算把能力释放出来。
```

---

### 二、为什么推理时计算有用

推理时计算有用，主要有几个原因。

#### 1. 给模型更多中间状态

CoT 让模型写中间步骤。

中间步骤成为后续 token 的上下文。

#### 2. 探索多个候选

Self-consistency 和 search 不只走一条路径。

多个候选可以降低偶然错误。

#### 3. 引入选择机制

Verifier 可以从多个候选中挑更好的。

#### 4. 允许试错和回退

Search 让模型发现某条路径不好后换路径。

#### 5. 使用外部反馈

代码执行、工具调用、检索结果、单元测试都能提供额外信号。

#### 6. 任务难度不均匀

简单问题不需要很多计算，难题可能需要更多计算。

动态分配计算可以更高效。

---

### 三、几种常见形式

Test-time compute scaling 有多种形式。

#### 1. Longer reasoning

生成更长的 reasoning trace。

例如 CoT、scratchpad、step-by-step solving。

#### 2. Multiple samples

采样多条答案或推理路径。

例如 self-consistency。

#### 3. Reranking

生成多个候选，再用 verifier 或 reward model 排序。

#### 4. Search

把推理过程组织成树或图。

例如 Tree-of-Thought、beam search、MCTS。

#### 5. Tool feedback

调用工具验证中间结果。

例如代码执行、计算器、检索、数据库查询。

#### 6. Iterative refinement

生成答案后检查、修正、再生成。

例如 draft -> critique -> revise。

#### 7. Debate or multi-agent

多个模型或多个角色互相挑战、验证、改进答案。

这些方法本质上都在增加推理时计算。

区别是计算花在不同位置。

---

### 四、Scaling 的对象是什么

Test-time compute 可以扩展多个维度。

#### 1. Token budget

允许模型生成更多中间推理 token。

#### 2. Sample budget

允许生成更多候选。

#### 3. Search budget

允许展开更多节点、更深层级。

#### 4. Verification budget

允许 verifier 检查更多候选或更多步骤。

#### 5. Tool budget

允许更多工具调用。

#### 6. Time budget

允许请求等待更久。

#### 7. Money budget

允许单个问题花更多推理成本。

不同任务的瓶颈不同。

数学题可能需要 sample/search budget。

代码题可能需要 tool/test budget。

RAG 问答可能需要 retrieval/rerank/verifier budget。

#### 关键公式与预算模型

把一次请求 $x$ 的推理策略记为 $m$。$m$ 可以是 direct answer、CoT、self-consistency、verifier rerank、search、tool-use 或 human review。一个粗粒度成本模型可以写成：

$$
C(m,x) =
c_{\mathrm{input}} n_{\mathrm{input}}
+ c_{\mathrm{output}} n_{\mathrm{output}}
+ c_{\mathrm{call}} N
+ c_{\mathrm{verify}} K_v
+ c_{\mathrm{tool}} K_t
$$

其中 $n_{\mathrm{input}}$ 是输入 token 数，$n_{\mathrm{output}}$ 是输出 token 数，$N$ 是模型调用次数，$K_v$ 是 verifier 调用次数，$K_t$ 是工具调用次数。这个公式不追求精确计费，而是提醒你：test-time compute 的成本不只来自“生成更长”，还来自多次调用、验证和外部工具。

把质量、成本、延迟和风险放到同一个决策目标中，可以写成：

$$
U(m,x) = Q(m,x) - \lambda_C C(m,x) - \lambda_T T(m,x) - \lambda_R R(m,x)
$$

其中 $Q(m,x)$ 是预期质量，$T(m,x)$ 是延迟，$R(m,x)$ 是安全或业务风险，$\lambda_C,\lambda_T,\lambda_R$ 是不同业务场景下的权重。高价值数学题可以让 $\lambda_C$ 小一点，高实时聊天可以让 $\lambda_T$ 大一点，高风险场景可以让 $\lambda_R$ 大一点。

预算约束下的动态路由可以写成：

$$
m^\star(x) =
\operatorname*{argmax}_{m \in \mathcal{M}} U(m,x),
\quad
C(m,x) \le B_C,
\quad
T(m,x) \le B_T
$$

这里 $\mathcal{M}$ 是候选策略集合，$B_C$ 是成本预算，$B_T$ 是延迟预算。面试里讲到这一步，就能把 test-time scaling 从“模型技巧”讲成“系统优化问题”。

如果每次独立采样命中正确解的概率近似为 $p$，重复采样 $N$ 次后候选集合中至少出现一个正确解的概率为：

$$
P_{\mathrm{hit}}(N,p) = 1 - (1-p)^N
$$

这解释了为什么重复采样会提升 pass@k 或 coverage。但系统最终能否答对，还取决于是否有 verifier、测试、规则或投票机制把正确候选选出来。没有可靠选择机制时，$P_{\mathrm{hit}}$ 上升不等于最终 accuracy 同步上升。

边际收益可以写成：

$$
\Delta A_N = A_N - A_{N-1}
$$

当 $\Delta A_N$ 已经小于单位成本带来的业务价值时，就应该停止继续增加采样或搜索。

Early stopping 可以用多数 margin 或答案熵来表达。设 $M_N$ 是前 $N$ 个候选中第一名答案与第二名答案的票数差，$H_N$ 是答案分布熵：

$$
S_N = \mathbf{1}[M_N \ge \tau_M] \lor \mathbf{1}[H_N \le \tau_H]
$$

当 $S_N=1$ 时提前停止。它的意义是：候选答案已经足够集中，就不要继续为同一个请求烧预算。

成本收益评估中常用 cost per solved task：

$$
CPS =
\frac{\sum_i C_i}{\sum_i y_i + \epsilon}
$$

其中 $y_i=1$ 表示第 $i$ 个任务被解决，$\epsilon$ 是防止分母为 0 的极小常数。动态策略相对固定策略的收益可以写成：

$$
G_{\mathrm{adapt}} = U_{\mathrm{adapt}} - U_{\mathrm{fixed}}
$$

如果 $G_{\mathrm{adapt}}>0$，说明自适应预算在质量、成本、延迟和风险的综合权衡上优于固定预算。

---

### 五、为什么不是越多越好

Test-time compute 有收益递减。

例如：

```text
1x compute -> 70% accuracy
2x compute -> 76% accuracy
5x compute -> 82% accuracy
20x compute -> 85% accuracy
```

越往后，每增加一单位计算带来的提升越少。

而成本、延迟和资源占用持续增加。

此外，更多计算也可能带来负面效果：

1. 更长 CoT 引入错误。
2. 更多采样产生更多噪声。
3. Search 被错误 verifier 误导。
4. 工具调用越多，失败点越多。
5. 用户等待时间变长。
6. 系统吞吐下降。

所以重点不是“尽量多算”。

重点是：

```text
在合适任务上，以合适预算，获得足够收益。
```

---

### 六、动态预算分配

生产系统不应该所有请求都用同样预算。

更合理的是动态分配。

例如按任务难度分层。

#### 1. Easy

简单事实问答、格式转换、普通分类。

策略：直接回答。

#### 2. Medium

需要两三步推理。

策略：单条 CoT 或短 scratchpad。

#### 3. Hard

数学、代码、多约束规划。

策略：多采样、verifier、search。

#### 4. High-risk

医疗、法律、金融、生产操作。

策略：工具验证、引用、人工确认、保守输出。

一个系统可以先做 difficulty estimation。

然后决定：

```text
直接回答
短 CoT
CoT + self-consistency
CoT + verifier
search + verifier
human review
```

---

### 七、难度估计怎么做

动态预算需要判断问题难不难。

难度估计可以来自：

1. 问题长度。
2. 是否包含数学、代码、逻辑、多约束。
3. 模型初次回答的置信度。
4. 多个候选答案是否一致。
5. Verifier 分数。
6. 检索证据是否充分。
7. 用户场景风险等级。
8. 历史错误率。

例如：

```text
如果单次回答置信度低，或 verifier 分数低，就自动增加采样或 search。
```

也可以采用 cascade：

```text
低成本解法先跑
-> 如果不确定，再升级到高成本解法
-> 如果仍不确定，拒答或人工审核
```

这比一开始就使用最高预算更高效。

---

### 八、Compute Allocation 策略

常见计算分配策略包括：

#### 1. Fixed budget

每个请求固定 N 条采样或固定最大 token。

简单但浪费。

#### 2. Adaptive budget

根据难度动态调整。

更适合生产。

#### 3. Early stopping

如果多个候选已经高度一致，提前停止。

例如 5 条里 4 条答案一致，就不再采样到 20 条。

#### 4. Escalation

先用便宜方法，不够再升级。

例如 direct answer -> CoT -> self-consistency -> search。

#### 5. Budget-aware search

Search 中根据剩余预算决定是否继续展开。

#### 6. Risk-aware budget

高风险任务增加验证和确认预算。

例如金融建议必须引用来源和人工审核。

---

### 九、成本收益曲线

评估 test-time scaling 必须画成本收益曲线。

横轴可以是：

1. token 数。
2. 采样条数。
3. search 节点数。
4. 工具调用次数。
5. 延迟。
6. 每题成本。

纵轴可以是：

1. accuracy。
2. solve rate。
3. pass@k。
4. human preference。
5. safety pass rate。
6. business success rate。

一个好的实验不是只报告最高分。

而是回答：

```text
多花 2 倍、5 倍、10 倍计算分别带来多少收益？
```

面试中如果能主动讲成本收益曲线，会显得很工程化。

---

### 十、和训练时 Scaling 的关系

训练时 scaling 和 test-time scaling 是互补关系。

训练更强的模型，可以让单次推理更准。

推理时投入更多计算，可以让固定模型在难题上表现更好。

两者之间有 trade-off：

```text
更大模型单次推理 vs 较小模型多次推理
```

例如：

1. 一个大模型单次回答。
2. 一个小模型采样 20 次加 verifier。

哪个更好，要看准确率、延迟、成本和部署约束。

未来系统可能不是单一模型，而是：

```text
模型大小选择 + 推理预算选择 + 工具验证 + 搜索策略
```

共同决定最终质量。

---

### 十一、Reasoning Model 的特殊性

Reasoning model 往往更强调推理时计算。

它们可能具备：

1. 更强的长推理能力。
2. 更好的中间步骤质量。
3. 更稳定的自我检查。
4. 更适合 verifier 或 PRM。
5. 更能从额外 token 中获益。

普通 chat model 可能生成长 CoT 后跑偏。

Reasoning model 的目标之一是：

```text
让额外推理 token 真正转化为更高质量，而不是只是更长文本。
```

因此评估 reasoning model 时，不仅看单次答案，还要看：

```text
随着 test-time compute 增加，性能是否持续提升。
```

---

### 十二、适用场景

适合 test-time compute scaling 的任务：

1. 数学推理。
2. 代码生成和修复。
3. 复杂规划。
4. 多步工具任务。
5. 科学推理。
6. 高价值决策辅助。
7. 需要严格验证的任务。

不适合或收益较低的任务：

1. 简单事实问答。
2. 翻译。
3. 简单摘要。
4. 低风险闲聊。
5. 高实时性场景。
6. 答案主观且难验证的任务。

判断标准是：

```text
任务是否有多步推理空间，是否能验证，是否值得等待和付费。
```

---

### 十三、系统设计示例

假设设计一个数学推理服务。

可以分层：

```text
Level 0: direct answer
Level 1: single CoT
Level 2: self-consistency N=5
Level 3: self-consistency N=20 + verifier
Level 4: search + PRM + symbolic checker
```

流程：

1. 先判断题目难度。
2. 简单题用 Level 0 或 1。
3. 中等题用 Level 2。
4. 难题用 Level 3。
5. 高价值难题用 Level 4。
6. 如果 verifier 不通过，拒答或请求人工。

系统记录：

1. 使用了哪个 level。
2. 消耗 token。
3. 采样次数。
4. verifier 分数。
5. 最终是否正确。

这些日志可以反过来优化预算策略。

---

### 十四、真实项目中的坑

#### 1. 只追求最高准确率

不看成本和延迟，方法无法上线。

#### 2. 所有请求都用高预算

简单问题浪费大量计算。

#### 3. 难度估计不准

简单题被高预算处理，难题却低预算回答。

#### 4. Verifier 不可靠

更多 compute 被错误评价函数引导，结果更差。

#### 5. 没有 early stopping

答案已经一致还继续采样，浪费成本。

#### 6. 忽略用户体验

用户不一定愿意等 30 秒得到稍微更好的答案。

#### 7. 不做分场景策略

同一套预算策略用于闲聊、数学、代码和高风险业务，效果会很差。

---

### 十五、评估指标

评估 test-time compute scaling 要看：

1. Accuracy / solve rate。
2. Pass@k。
3. Cost per query。
4. Cost per solved task。
5. Latency p50/p95/p99。
6. Token usage。
7. Tool call count。
8. Verifier pass rate。
9. User satisfaction。
10. Safety violation rate。

还要按任务难度分桶：

```text
easy / medium / hard / high-risk
```

因为平均指标会掩盖问题。

Test-time scaling 的收益通常主要来自 hard subset。

如果只看全量平均，可能低估或高估它的价值。

---

### 十六、最小代码：自适应推理预算路由

下面这个 demo 用纯 Python 模拟三种策略：

1. `fixed_low`：所有请求都直接回答。
2. `fixed_high`：所有请求都使用高预算 search/tool。
3. `adaptive`：按难度、风险和可验证性动态选择 direct、self-consistency 或 search/tool。

这个 demo 不模拟真实 LLM，而是模拟生产系统里最重要的控制面：预算、延迟、采样次数、风险违规和是否解决任务。

```python
from collections import Counter

REQUESTS = [
    {"id": "easy_fact", "difficulty": "easy", "risk": "low", "verifiable": False},
    {"id": "medium_math", "difficulty": "medium", "risk": "low", "verifiable": True},
    {"id": "hard_code", "difficulty": "hard", "risk": "medium", "verifiable": True},
    {"id": "regulated_advice", "difficulty": "medium", "risk": "high", "verifiable": False},
]

DIRECT_RESULTS = {
    "easy_fact": (True, False),
    "medium_math": (False, False),
    "hard_code": (False, False),
    "regulated_advice": (False, True),
}

SC_SAMPLES = {
    "medium_math": (["41", "42", "42", "42", "40"], "42"),
    "hard_code": (["buggy", "buggy", "fixed", "buggy", "fixed"], "fixed"),
}


def hit_probability(n, p):
    return 1.0 - (1.0 - p) ** n


def majority_with_early_stop(request_id, max_samples=5, margin_count=2):
    samples, correct_answer = SC_SAMPLES[request_id]
    counts = Counter()
    used = 0
    answer = None
    for sample in samples[:max_samples]:
        used += 1
        counts[sample] += 1
        top_two = counts.most_common(2)
        top_answer, top_count = top_two[0]
        second_count = top_two[1][1] if len(top_two) > 1 else 0
        answer = top_answer
        if top_count - second_count >= margin_count:
            break
    return answer == correct_answer, used, answer


def run_strategy(req, strategy):
    if strategy == "direct":
        solved, violation = DIRECT_RESULTS[req["id"]]
        return {"solved": solved, "violation": violation, "cost": 1.0, "latency": 0.8, "samples": 1}

    if strategy == "self_consistency":
        solved, used, _ = majority_with_early_stop(req["id"])
        verify_cost = 0.4 if req["verifiable"] else 0.0
        return {
            "solved": solved,
            "violation": False,
            "cost": 1.1 * used + verify_cost,
            "latency": 1.2 + 0.15 * used,
            "samples": used,
        }

    if strategy == "search_tool":
        extra_review = 1.0 if req["risk"] == "high" else 0.0
        return {
            "solved": True,
            "violation": False,
            "cost": 6.0 + extra_review,
            "latency": 4.0 + extra_review,
            "samples": 1,
        }

    raise ValueError(strategy)


def adaptive_policy(req):
    if req["risk"] == "high":
        return "search_tool"
    if req["difficulty"] == "hard":
        return "search_tool"
    if req["difficulty"] == "medium":
        return "self_consistency"
    return "direct"


def evaluate(name, policy):
    total_cost = 0.0
    total_latency = 0.0
    solved = 0
    violations = 0
    total_samples = 0
    mix = Counter()
    trace = []
    for req in REQUESTS:
        strategy = policy(req)
        result = run_strategy(req, strategy)
        mix[strategy] += 1
        total_cost += result["cost"]
        total_latency += result["latency"]
        total_samples += result["samples"]
        solved += int(result["solved"])
        violations += int(result["violation"])
        trace.append((req["id"], strategy, result["samples"], result["solved"], result["violation"]))
    n = len(REQUESTS)
    return {
        "name": name,
        "solve_rate": round(solved / n, 3),
        "total_cost": round(total_cost, 2),
        "avg_latency": round(total_latency / n, 2),
        "cost_per_solved": round(total_cost / max(solved, 1), 2),
        "violation_rate": round(violations / n, 3),
        "avg_samples": round(total_samples / n, 2),
        "strategy_mix": dict(mix),
        "trace": trace,
    }


coverage = {
    "hit@1": round(hit_probability(1, 0.25), 3),
    "hit@5": round(hit_probability(5, 0.25), 3),
    "hit@20": round(hit_probability(20, 0.25), 3),
    "gain_5_vs_4": round(hit_probability(5, 0.25) - hit_probability(4, 0.25), 3),
}

fixed_low = evaluate("fixed_low", lambda req: "direct")
fixed_high = evaluate("fixed_high", lambda req: "search_tool")
adaptive = evaluate("adaptive", adaptive_policy)

gain = {
    "cost_saved_vs_fixed_high": round(fixed_high["total_cost"] - adaptive["total_cost"], 2),
    "latency_saved_vs_fixed_high": round(fixed_high["avg_latency"] - adaptive["avg_latency"], 2),
    "solve_rate_gap": round(adaptive["solve_rate"] - fixed_high["solve_rate"], 3),
}

for report in (fixed_low, fixed_high, adaptive):
    report.pop("name")

print("coverage=", coverage)
print("fixed_low=", fixed_low)
print("fixed_high=", fixed_high)
print("adaptive=", adaptive)
print("adaptive_gain=", gain)
```

典型输出：

```text
coverage= {'hit@1': 0.25, 'hit@5': 0.763, 'hit@20': 0.997, 'gain_5_vs_4': 0.079}
fixed_low= {'solve_rate': 0.25, 'total_cost': 4.0, 'avg_latency': 0.8, 'cost_per_solved': 4.0, 'violation_rate': 0.25, 'avg_samples': 1.0, 'strategy_mix': {'direct': 4}, 'trace': [('easy_fact', 'direct', 1, True, False), ('medium_math', 'direct', 1, False, False), ('hard_code', 'direct', 1, False, False), ('regulated_advice', 'direct', 1, False, True)]}
fixed_high= {'solve_rate': 1.0, 'total_cost': 25.0, 'avg_latency': 4.25, 'cost_per_solved': 6.25, 'violation_rate': 0.0, 'avg_samples': 1.0, 'strategy_mix': {'search_tool': 4}, 'trace': [('easy_fact', 'search_tool', 1, True, False), ('medium_math', 'search_tool', 1, True, False), ('hard_code', 'search_tool', 1, True, False), ('regulated_advice', 'search_tool', 1, True, False)]}
adaptive= {'solve_rate': 1.0, 'total_cost': 18.8, 'avg_latency': 2.9, 'cost_per_solved': 4.7, 'violation_rate': 0.0, 'avg_samples': 1.75, 'strategy_mix': {'direct': 1, 'self_consistency': 1, 'search_tool': 2}, 'trace': [('easy_fact', 'direct', 1, True, False), ('medium_math', 'self_consistency', 4, True, False), ('hard_code', 'search_tool', 1, True, False), ('regulated_advice', 'search_tool', 1, True, False)]}
adaptive_gain= {'cost_saved_vs_fixed_high': 6.2, 'latency_saved_vs_fixed_high': 1.35, 'solve_rate_gap': 0.0}
```

这组结果说明四件事：

1. `hit@20` 很高，只说明候选集合中更可能出现正确解；如果没有可靠选择机制，最终答案仍可能错。
2. `fixed_low` 成本最低，但只解决简单任务，并且在高风险请求上违规。
3. `fixed_high` 全部解决且无违规，但对简单任务明显浪费。
4. `adaptive` 保持与 `fixed_high` 相同 solve rate，同时降低总成本和平均延迟；这就是生产系统里动态预算分配的核心价值。

---

### 十七、面试问答

#### 问题 1：Test-time compute scaling 是什么？

可以这样回答：

```text
它是在模型参数固定的情况下，在推理阶段投入更多计算，例如更长推理、多采样、verifier、search、工具验证和迭代修正，以提升最终答案质量。
```

#### 问题 2：它和训练时 scaling 有什么区别？

可以这样回答：

```text
训练时 scaling 是用更多参数、数据和训练 FLOPs 把能力压进模型；test-time scaling 是在使用模型时增加 token、采样、搜索和验证，把固定模型的能力更充分释放出来。
```

#### 问题 3：为什么 test-time compute 对 reasoning 有用？

可以这样回答：

```text
Reasoning 问题通常需要多步探索和检查。更多推理时计算可以提供中间状态、探索多条路径、使用 verifier 选择更好答案，并允许试错和回退。
```

#### 问题 4：如何动态分配推理预算？

可以这样回答：

```text
先估计任务难度和风险，简单任务直接回答，中等任务用短 CoT，难题用 self-consistency 或 verifier，高价值高风险任务用 search、工具验证或人工审核。预算应根据置信度和 verifier 分数动态升级。
```

#### 问题 5：如何评估 test-time scaling 是否值得？

可以这样回答：

```text
要画成本收益曲线，比较不同 token、采样数、search 节点和工具调用预算下的准确率、solve rate、延迟、成本和安全指标，而不是只报告最高准确率。
```

#### 问题 6：Test-time scaling 的主要风险是什么？

可以这样回答：

```text
主要风险是成本和延迟上升、收益递减、难度估计不准、verifier 错误引导、搜索空间爆炸、用户体验下降，以及在不适合的任务上浪费计算。
```

---

### 十八、常见误区

1. 误区：推理时算得越多越好。
   纠正：收益递减，而且成本和延迟会上升。

2. 误区：所有任务都应该用 reasoning 模式。
   纠正：简单任务直接回答更好。

3. 误区：多采样就是 test-time scaling 的全部。
   纠正：还包括长推理、verifier、search、工具反馈和迭代修正。

4. 误区：只看 accuracy。
   纠正：还要看 cost、latency、token、工具调用和安全。

5. 误区：大模型不需要 test-time compute。
   纠正：大模型也可能通过额外推理计算在难题上继续提升。

6. 误区：小模型多采样一定能超过大模型。
   纠正：取决于任务、模型能力、verifier 和成本。

---

### 十九、小练习

1. 比较训练时 scaling 和 test-time scaling。
2. 设计一个 direct answer、CoT、self-consistency、search 的四级推理预算系统。
3. 为数学题设计 N=1、5、20 的成本收益实验。
4. 设计一个 early stopping 规则，当答案一致时停止采样。
5. 构造一个更多 CoT token 反而导致错误的例子。
6. 设计一个难度估计器，判断请求应该用低预算还是高预算。
7. 设计一个代码修复任务的 test-time compute 分配策略。
8. 画出 cost per solved task 随采样数变化的曲线。
9. 分析高风险医疗问答为什么不能只靠多采样。
10. 用 3 分钟回答：“如何在生产系统中动态分配 reasoning compute？”

### 本讲总结

本讲最重要的结论：

1. Test-time compute scaling 是在推理阶段投入更多计算来提升答案质量。
2. 它包括长推理、多采样、reranking、verifier、search、工具反馈和迭代修正。
3. 它和训练时 scaling 互补，一个提升模型内化能力，一个提升使用时求解能力。
4. Reasoning 任务更容易从 test-time compute 中受益，因为它们需要探索、检查和纠错。
5. Test-time compute 不是越多越好，存在收益递减、成本、延迟和用户体验问题。
6. 生产系统应按任务难度、风险、置信度和 verifier 结果动态分配预算。
7. 评估必须看成本收益曲线，而不是只看最高准确率。
8. 面试中要强调：test-time compute scaling 的核心是把额外计算花在值得花的问题上。

## 第 91 讲：数学推理模型训练

### 本讲目标

学完本讲，你应该能回答六个问题：

1. 数学推理模型训练和普通指令微调有什么区别。
2. 数学数据为什么重要，常见数据类型有哪些。
3. CoT、过程监督、结果监督、合成数据在数学训练中分别起什么作用。
4. 如何训练和使用数学 verifier 或 reward model。
5. 数学推理训练中的数据污染、过拟合和评估陷阱有哪些。
6. 面试中如何设计一个提升 LLM 数学能力的训练方案。

数学推理是 reasoning model 最典型的训练场景之一。

原因很直接：

1. 数学题需要多步推理。
2. 很多数学题有明确答案。
3. 解题过程可以拆成步骤。
4. 部分结果可以自动验证。
5. 有大量 benchmark 可以评估。

因此，很多 reasoning 方法都会先在数学任务上验证。

但数学推理模型训练不是简单收集一堆题目和答案做 SFT。

真正难的是：

```text
让模型学会可靠地产生、检查和改进多步推理过程。
```

从公开研究脉络看，数学 reasoning 的训练大致经历了几条线：MATH 数据集强调竞赛数学题、完整 step-by-step solution 和传统 scaling 在高难数学上的不足；GSM8K 和 Training Verifiers to Solve Math Word Problems 强调小学数学文字题、多候选生成和 verifier selection；Minerva 强调继续训练高质量技术内容对定量推理的作用；Let's Verify Step by Step 则系统比较 outcome supervision 与 process supervision，并用 PRM800K 这类步骤级标签说明过程监督在长推理中的价值。

所以数学推理训练不能只回答“收集什么题”。更关键的问题是：

```text
哪些样本进 SFT，哪些样本只用于 ORM/PRM，哪些错误样本作为 hard negative，哪些题必须留作未污染评估。
```

---

### 一、数学推理训练想解决什么

普通语言模型可能知道很多数学文本。

但它常见问题是：

1. 题意理解错。
2. 中间步骤跳步。
3. 算术错误。
4. 符号变换错误。
5. 最终答案格式错误。
6. 看起来会推理，但本质是在模仿题解模板。

数学推理训练的目标是：

```text
让模型在面对新题时，能分解问题、生成有效中间步骤、执行计算、检查约束，并得到正确答案。
```

这里不只是提升最终准确率。

还要提升：

1. 步骤正确性。
2. 推理鲁棒性。
3. 自我检查能力。
4. 与 verifier/search 的配合能力。
5. 对新题型的泛化能力。

---

### 二、数学数据类型

数学训练数据可以分成几类。

#### 1. Question-answer data

只有题目和最终答案。

例如：

```text
Q: 24 的 1/3 是多少？
A: 8
```

优点是容易收集。

缺点是缺少过程监督。

#### 2. Step-by-step solution data

包含完整解题过程。

例如：

```text
Q: 2x + 3 = 11，求 x。
Solution: 两边减 3，得到 2x = 8；两边除以 2，得到 x = 4。
A: 4
```

这是 CoT SFT 的核心数据。

#### 3. Process-labeled data

每一步都有正确/错误标签。

用于训练 PRM。

#### 4. Preference data

同一道题的两个解法，标注哪个更好。

用于训练 reward model 或做 DPO/RLHF 类优化。

#### 5. Synthetic data

由模型或程序生成的新题、新解法、新错误样本。

#### 6. Tool-verified data

用计算器、符号系统、单元测试或 proof checker 验证过的数据。

数学训练通常需要混合这些数据。

只有 QA 数据不够。

只有漂亮题解也不够。

还需要错误样本和验证信号。

#### 统一记号和训练目标

把第 $i$ 道数学题记为 $x_i$，标准答案记为 $a_i^\star$，候选解法记为 $r_{i,j}$，候选最终答案记为 $a_{i,j}$。第 $j$ 条解法的第 $k$ 个步骤标签记为 $s_{i,j,k} \in \{0,1\}$。

CoT SFT 的 token-level 目标可以写成：

$$
\mathcal{L}_{\mathrm{sft}} =
-\frac{1}{Z}
\sum_i \sum_t
m_{i,t}
\log \pi_\theta(y_{i,t} \mid x_i, y_{i,1:t-1})
$$

其中 $m_{i,t}$ 是 loss mask，通常只让 assistant 的解题过程和最终答案参与 loss，$Z=\sum_i\sum_t m_{i,t}$ 是有效 token 数。数学 SFT 的关键不是让所有 token 都学，而是让高质量推理步骤和答案格式被稳定学习。

结果监督标签可以写成：

$$
z_{i,j} =
\mathbf{1}[
\operatorname{norm}(a_{i,j}) =
\operatorname{norm}(a_i^\star)
]
$$

这里 $\operatorname{norm}$ 表示答案标准化，例如去掉单位文本、化简分数、统一小数精度或抽取最终 boxed answer。没有答案标准化，数学训练和评估会被格式噪声严重污染。

ORM 的二分类目标可以写成：

$$
\mathcal{L}_{\mathrm{orm}} =
-\frac{1}{M}
\sum_{i,j}
\left[
z_{i,j}\log v_\phi(x_i,r_{i,j})
+
(1-z_{i,j})\log(1-v_\phi(x_i,r_{i,j}))
\right]
$$

PRM 的步骤级目标可以写成：

$$
\mathcal{L}_{\mathrm{prm}} =
-\frac{1}{K}
\sum_{i,j,k}
\left[
s_{i,j,k}\log p_\psi(x_i,r_{i,j,1:k})
+
(1-s_{i,j,k})\log(1-p_\psi(x_i,r_{i,j,1:k}))
\right]
$$

其中 $p_\psi$ 是截至第 $k$ 步的过程正确概率。ORM 学的是“整条解法最终是否可接受”，PRM 学的是“推理走到这一步是否还可靠”。

Rejection sampling SFT 的保留集合可以写成：

$$
\mathcal{D}_{\mathrm{rs}} =
\{(x_i,r_{i,j}) \mid z_{i,j}=1,\ q_{i,j}\ge \tau_q\}
$$

其中 $q_{i,j}$ 可以综合最终答案、过程分数、格式合法性和去重结果：

$$
q_{i,j} =
\alpha z_{i,j}
+ \beta \frac{1}{K_{i,j}}\sum_k s_{i,j,k}
- \gamma h_{i,j}
$$

$h_{i,j}=1$ 表示 hard negative 风险，例如答案碰巧正确但中间步骤错误、单位不一致、跳步严重或疑似从评估集泄漏。

多来源数学数据混合训练时，可以写成：

$$
\mathcal{L}_{\mathrm{mix}} =
\sum_{g \in \mathcal{G}} w_g \mathcal{L}_g,
\quad
\sum_{g \in \mathcal{G}} w_g = 1
$$

$\mathcal{G}$ 可以包含 QA、CoT、PRM、hard negative、tool-verified、synthetic、general instruction 等数据组。数学数据占比过低，推理能力起不来；占比过高，也可能牺牲通用对话、写作和安全行为。

污染率可以粗略写成：

$$
\rho_{\mathrm{leak}} =
\frac{1}{N}
\sum_i
\mathbf{1}\left[
\max_j \operatorname{sim}(x_i^{\mathrm{eval}}, x_j^{\mathrm{train}})
\ge \tau_{\mathrm{sim}}
\right]
$$

错误类型占比可以写成：

$$
e_c =
\frac{\sum_i \mathbf{1}[g_i=c]}{N}
$$

其中 $g_i$ 是第 $i$ 个错误样本的错误类别，例如读题、建模、算术、代数、单位、格式或 verifier 误判。数学训练要靠这些分桶决定下一轮改数据、改 verifier、接工具还是改推理策略。

---

### 三、CoT SFT

最基础的数学训练方法是 CoT supervised fine-tuning。

训练样本是：

```text
题目 -> 分步解法 -> 最终答案
```

模型学习：

1. 如何读题。
2. 如何分解步骤。
3. 如何写中间推导。
4. 如何给最终答案。

优点：

1. 实现简单。
2. 能显著提升模型按步骤解题的格式能力。
3. 有助于让模型输出可检查的推理过程。

缺点：

1. 依赖题解质量。
2. 容易学到模板。
3. 不保证步骤忠实。
4. 不能直接惩罚错误推理。
5. 对分布外题型泛化有限。

CoT SFT 是起点，不是终点。

---

### 四、结果监督 Outcome Supervision

结果监督只看最终答案是否正确。

例如模型生成 10 个解法。

只要最终答案对，就给正反馈。

结果监督可以用于：

1. 训练 ORM。
2. 选择候选答案。
3. 做 rejection sampling。
4. 做 RL 优化。

优点：

1. 标注便宜。
2. 很多数学题有标准答案。
3. 可以自动判分。

缺点：

1. 不知道哪一步错。
2. 可能奖励错误过程但答案碰巧正确。
3. 对证明题和开放题不够。
4. 难以指导模型改进中间步骤。

结果监督适合提高最终正确率。

但如果目标是训练稳定 reasoning，过程监督更重要。

---

### 五、过程监督 Process Supervision

过程监督检查每一步推理是否正确。

例如：

```text
Step 1: 2x + 3 = 11
Step 2: 2x = 8       correct
Step 3: x = 5        incorrect
```

过程监督可以用于训练 PRM。

PRM 训练好后，可以：

1. 给 CoT 每一步打分。
2. 搜索时选择更好的中间状态。
3. 发现错误步骤。
4. 过滤坏样本。
5. 指导模型重新生成。

过程监督的优点：

1. 更细粒度。
2. 更适合搜索。
3. 能提升可诊断性。
4. 对长推理更有帮助。

缺点：

1. 标注贵。
2. 步骤粒度难定义。
3. 对复杂证明很难判断。
4. 容易有标注不一致。

面试中可以这样总结：

```text
结果监督告诉模型答案对不对，过程监督告诉模型推理哪里对、哪里错。
```

---

### 六、Rejection Sampling

Rejection sampling 是数学训练中常见方法。

流程：

```text
对每道题采样多个解法
-> 用答案匹配、程序验证或 verifier 判断正确性
-> 保留正确或高分解法
-> 用这些解法做 SFT
```

它的好处是：

1. 可以自动扩大高质量 CoT 数据。
2. 利用模型自己生成多样解法。
3. 结合 verifier 过滤错误。

但也有风险：

1. Verifier 错会保留坏样本。
2. 只保留模型已经会的题，难题仍然缺数据。
3. 生成解法可能同质化。
4. 错误但答案碰巧正确的过程可能混入。

所以 rejection sampling 最好结合过程检查或人工抽查。

---

### 七、合成数学数据

合成数据是提升数学能力的重要手段。

来源包括：

1. 模型生成新题。
2. 模型生成多种解法。
3. 程序生成可验证题。
4. 从简单题变换出复杂题。
5. 生成错误解法作为负样本。
6. 从真实题抽取模板再替换变量。

合成数据的关键不是数量，而是质量和覆盖。

要关注：

1. 题目是否正确。
2. 答案是否可验证。
3. 难度是否分布合理。
4. 是否覆盖不同题型。
5. 是否和评估集污染。
6. 解法是否多样。

如果合成数据质量差，模型会学到错误推理。

如果合成数据太模板化，模型会过拟合模板。

---

### 八、数学 Verifier 训练

数学 verifier 可以是 ORM，也可以是 PRM。

#### 1. ORM for math

输入题目和最终答案，判断答案是否正确。

适合短答案题。

#### 2. PRM for math

输入题目和推理步骤，判断每一步是否合理。

适合长推理和搜索。

训练数据需要正负样本。

正样本来自正确题解。

负样本可以来自：

1. 模型错误解法。
2. 人工构造错误。
3. 随机替换中间数字。
4. 常见代数错误。
5. 单位或符号错误。

好的负样本应该是 hard negative。

也就是看起来像对，但实际错。

否则 verifier 只会识别低级错误。

---

### 九、训练流程示例

一个完整数学推理训练流程可以是：

```text
1. 收集数学题和标准答案
2. 收集或生成 step-by-step solutions
3. 用 CoT SFT 训练基础推理模型
4. 采样多个候选解法
5. 用答案匹配、程序验证和人工抽查过滤
6. 构造 ORM/PRM 数据
7. 训练 verifier 或 reward model
8. 用 verifier 做 rejection sampling 或 reranking
9. 可选：做 RL 或 DPO 类偏好优化
10. 在独立 benchmark 上评估
```

这个流程中最容易出问题的是数据质量和评估污染。

数学 benchmark 很容易被训练数据污染。

所以必须做去重和污染检测。

---

### 十、训练目标怎么选

不同阶段目标不同。

#### 1. SFT

目标是模仿高质量解题过程。

适合打基础。

#### 2. Rejection sampling SFT

目标是从模型生成中筛出好解法再训练。

适合扩大数据。

#### 3. ORM/PRM training

目标是学会评价答案或过程。

适合 test-time rerank/search。

#### 4. RL

目标是直接优化正确性或 reward。

但更难稳定，容易 reward hacking。

#### 5. DPO / preference optimization

目标是偏好更好的解法。

比 RL 简化，但仍依赖偏好数据质量。

不要把所有问题都归结为“上 RL”。

很多时候，数据清洗、SFT、verifier 和推理时搜索已经能带来主要提升。

---

### 十一、评估数学能力

数学推理评估要看多个层面。

#### 1. Final answer accuracy

最终答案是否正确。

#### 2. Step correctness

推理步骤是否正确。

#### 3. Robustness

换表达、换数字、换题型是否稳定。

#### 4. Generalization

是否能解训练中没见过的新题型。

#### 5. Difficulty breakdown

按难度分桶看表现。

#### 6. Contamination check

评估题是否出现在训练数据或合成数据中。

#### 7. Test-time compute curve

随着采样数、搜索预算增加，性能如何变化。

#### 8. Error analysis

错误来自读题、建模、计算、符号、最终格式还是 verifier。

只报告一个总 accuracy 不够。

要知道模型到底哪里不会。

---

### 十二、常见错误类型

数学模型常见错误包括：

1. 读题错误。
2. 漏条件。
3. 单位转换错误。
4. 算术错误。
5. 代数变形错误。
6. 变量定义混乱。
7. 逻辑跳步。
8. 最终答案格式错误。
9. 证明中使用未证明结论。
10. 过度依赖模板。

做 error analysis 时，要把错误分类型。

不同错误需要不同修复方法。

算术错误可以接计算器。

读题错误需要更好的数据和 prompt。

证明错误可能需要 PRM 或 formal verifier。

---

### 十三、数据污染和过拟合

数学 benchmark 污染很常见。

例如训练数据里包含评估题原题、改写题或题解。

污染会导致模型看起来数学能力很强，但其实是记忆。

防护方法：

1. 对训练题和评估题做 exact match 去重。
2. 做 fuzzy matching。
3. 对题干、答案、题解分别去重。
4. 检查合成数据是否从评估题改写。
5. 使用时间切分的新数据。
6. 做人工抽查。

过拟合也很常见。

模型可能学会某个 benchmark 的题型模板。

因此要用多个 benchmark 和自建 holdout 集评估。

---

### 十四、真实项目中的坑

#### 1. 只收集最终答案数据

模型学不到稳定步骤。

#### 2. CoT 数据质量差

漂亮但错误的题解会污染模型。

#### 3. 合成数据太模板化

模型只会解同类模板题。

#### 4. Verifier 太弱

错误解法被保留下来继续训练。

#### 5. 只看 benchmark 分数

不做污染检测和错误分析，容易误判能力。

#### 6. 忽略格式问题

模型会推理但 final answer 抽取失败。

#### 7. 直接上 RL

没有好的 reward 和数据基础，训练不稳定且容易 reward hacking。

---

### 十五、最小代码：数学训练数据过滤与过程监督

下面的 demo 模拟一个数学训练数据清洗器。它比较两种过滤方式：

1. `final_only`：只看最终答案是否和标准答案一致。
2. `process_aware`：最终答案正确，并且过程步骤正确率足够高。

重点是 `linear_lucky` 和 `ratio_lucky_unit`：它们最终答案正确，但过程里有关键错误。只用 outcome supervision 做 rejection sampling，会把这类样本保留下来；加入过程监督后，可以把它们当作 hard negative 或 PRM 训练样本，而不是直接放进 CoT SFT。

```python
from collections import Counter, defaultdict
import re

CANDIDATES = [
    {"problem": "linear", "id": "linear_correct", "gold": "4", "answer": "4", "steps": [1, 1, 1]},
    {"problem": "linear", "id": "linear_lucky", "gold": "4", "answer": "4", "steps": [1, 0, 0]},
    {"problem": "linear", "id": "linear_wrong_answer", "gold": "4", "answer": "5", "steps": [1, 1, 0]},
    {"problem": "ratio", "id": "ratio_correct", "gold": "12", "answer": "12", "steps": [1, 1, 1]},
    {"problem": "ratio", "id": "ratio_arith_error", "gold": "12", "answer": "15", "steps": [1, 0, 0]},
    {"problem": "ratio", "id": "ratio_lucky_unit", "gold": "12", "answer": "12", "steps": [1, 0, 1]},
    {"problem": "geometry", "id": "geometry_correct", "gold": "50", "answer": "50", "steps": [1, 1, 1, 1]},
    {"problem": "geometry", "id": "geometry_format", "gold": "50", "answer": "50 square cm", "steps": [1, 1, 1, 1]},
]


def normalize_answer(text):
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else text.strip().lower()


def process_score(steps):
    return sum(steps) / len(steps)


def first_error(steps):
    for idx, ok in enumerate(steps, start=1):
        if not ok:
            return idx
    return None


def final_correct(row):
    return normalize_answer(row["answer"]) == normalize_answer(row["gold"])


def audit(row):
    return {
        "id": row["id"],
        "final": int(final_correct(row)),
        "process_score": round(process_score(row["steps"]), 3),
        "first_error": first_error(row["steps"]),
    }


def keep_final_only(row):
    return final_correct(row)


def keep_process_aware(row, min_process=0.8):
    return final_correct(row) and process_score(row["steps"]) >= min_process


def build_preference_pairs(rows):
    by_problem = defaultdict(list)
    for row in rows:
        by_problem[row["problem"]].append(row)
    pairs = []
    for rows_for_problem in by_problem.values():
        positives = [r for r in rows_for_problem if keep_process_aware(r)]
        negatives = [r for r in rows_for_problem if not keep_process_aware(r)]
        for chosen in positives:
            for rejected in negatives:
                pairs.append((chosen["id"], rejected["id"]))
    return pairs


def weighted_loss(losses, weights):
    return sum(weights[name] * losses[name] for name in weights)


final_only = [row for row in CANDIDATES if keep_final_only(row)]
process_aware = [row for row in CANDIDATES if keep_process_aware(row)]
noisy_final = [row for row in final_only if not keep_process_aware(row)]
preference_pairs = build_preference_pairs(CANDIDATES)
step_labels = sum(len(row["steps"]) for row in CANDIDATES)
error_buckets = Counter(
    "clean" if first_error(row["steps"]) is None else f"step_{first_error(row['steps'])}"
    for row in CANDIDATES
)

losses = {"qa": 1.2, "cot": 0.9, "prm": 0.7, "hard_negative": 1.5}
weights = {"qa": 0.2, "cot": 0.4, "prm": 0.25, "hard_negative": 0.15}

print("candidate_audit=", [audit(row) for row in CANDIDATES])
print("filter_summary=", {
    "total": len(CANDIDATES),
    "final_only_kept": len(final_only),
    "process_aware_kept": len(process_aware),
    "final_only_noise": len(noisy_final),
    "noise_rate": round(len(noisy_final) / max(len(final_only), 1), 3),
})
print("training_sets=", {
    "cot_sft_ids": [row["id"] for row in process_aware],
    "orm_examples": len(CANDIDATES),
    "prm_step_labels": step_labels,
    "preference_pairs": len(preference_pairs),
})
print("error_buckets=", dict(error_buckets))
print("mix_loss=", round(weighted_loss(losses, weights), 3))
```

典型输出：

```text
candidate_audit= [{'id': 'linear_correct', 'final': 1, 'process_score': 1.0, 'first_error': None}, {'id': 'linear_lucky', 'final': 1, 'process_score': 0.333, 'first_error': 2}, {'id': 'linear_wrong_answer', 'final': 0, 'process_score': 0.667, 'first_error': 3}, {'id': 'ratio_correct', 'final': 1, 'process_score': 1.0, 'first_error': None}, {'id': 'ratio_arith_error', 'final': 0, 'process_score': 0.333, 'first_error': 2}, {'id': 'ratio_lucky_unit', 'final': 1, 'process_score': 0.667, 'first_error': 2}, {'id': 'geometry_correct', 'final': 1, 'process_score': 1.0, 'first_error': None}, {'id': 'geometry_format', 'final': 1, 'process_score': 1.0, 'first_error': None}]
filter_summary= {'total': 8, 'final_only_kept': 6, 'process_aware_kept': 4, 'final_only_noise': 2, 'noise_rate': 0.333}
training_sets= {'cot_sft_ids': ['linear_correct', 'ratio_correct', 'geometry_correct', 'geometry_format'], 'orm_examples': 8, 'prm_step_labels': 26, 'preference_pairs': 4}
error_buckets= {'clean': 4, 'step_2': 3, 'step_3': 1}
mix_loss= 1.0
```

这组输出对应几条工程判断：

1. `final_only_kept=6` 但其中有 2 条过程噪声，说明只靠最终答案做 rejection sampling 会污染 CoT SFT。
2. `process_aware_kept=4` 更适合进入 CoT SFT；被剔除的最终答案正确样本不是没用，而是更适合当作 PRM、hard negative 或 preference pair。
3. `prm_step_labels=26` 表明同样 8 条候选可以产生更多步骤级训练信号。
4. `error_buckets` 能告诉你下一轮重点修复第 2 步错误还是第 3 步错误，而不是只看一个总 accuracy。

---

### 十六、面试问答

#### 问题 1：数学推理模型训练和普通 SFT 有什么区别？

可以这样回答：

```text
普通 SFT 更关注指令遵循和回答格式，数学推理训练更关注多步推理、步骤正确性、最终答案验证、过程监督和 test-time search/verifier 配合。
```

#### 问题 2：为什么数学训练需要 CoT 数据？

可以这样回答：

```text
因为数学题通常需要多步推导。CoT 数据让模型学习如何分解问题、写中间步骤、执行计算并给出最终答案，比只有 question-answer 更能训练推理过程。
```

#### 问题 3：结果监督和过程监督有什么区别？

可以这样回答：

```text
结果监督只看最终答案是否正确，标注便宜但无法定位错误；过程监督检查每一步推理是否正确，标注更贵但能训练 PRM，帮助搜索和错误定位。
```

#### 问题 4：如何构造数学 verifier 数据？

可以这样回答：

```text
正样本来自正确题解，负样本来自模型错误解法、常见代数错误、随机替换中间步骤和 hard negatives。ORM 标注最终答案，PRM 标注每一步是否正确。
```

#### 问题 5：数学推理评估要注意什么？

可以这样回答：

```text
除了 final answer accuracy，还要看步骤正确性、难度分桶、鲁棒性、泛化、污染检测、test-time compute 曲线和错误类型分析。
```

#### 问题 6：如果让你提升一个模型的数学能力，你会怎么做？

可以这样回答：

```text
先做数据审计和 benchmark，收集高质量题目、答案和 CoT，做 CoT SFT；再采样候选解法，用答案验证和人工抽查过滤，训练 ORM/PRM；最后结合 self-consistency、verifier 或 search 做推理时增强，并持续做污染检测和错误分析。
```

---

### 十七、常见误区

1. 误区：数学能力就是背题库。
   纠正：真正能力要看新题型和变体上的泛化。

2. 误区：只有最终答案对就够。
   纠正：过程错误会影响泛化和可靠性。

3. 误区：CoT 越长越好。
   纠正：长但错误的推理会误导模型。

4. 误区：合成数据越多越好。
   纠正：低质量和模板化合成数据会造成过拟合。

5. 误区：Verifier 能自动解决所有错误。
   纠正：Verifier 也需要训练、校准和评估。

6. 误区：数学 benchmark 高分就说明 reasoning 强。
   纠正：还要排查污染、模板过拟合和 test-time compute 成本。

---

### 十八、小练习

1. 为一道代数题写 QA 数据、CoT 数据和 PRM 数据三种格式。
2. 构造一个最终答案正确但过程错误的数学样本。
3. 构造 5 个 hard negative，用于训练数学 verifier。
4. 设计一个 rejection sampling SFT 流程。
5. 设计一个数学合成数据生成和过滤流程。
6. 设计一个数学评估集污染检测流程。
7. 按读题、建模、计算、格式四类分析 20 个错误样本。
8. 比较 CoT SFT、ORM、PRM、RL 在数学训练中的作用。
9. 设计一个数学推理 test-time compute 曲线实验。
10. 用 3 分钟回答：“如何系统提升 LLM 的数学推理能力？”

### 本讲总结

本讲最重要的结论：

1. 数学推理训练的目标是提升多步推理、步骤正确性和最终答案可靠性。
2. 数学数据包括 QA、CoT、process-labeled、preference、synthetic 和 tool-verified data。
3. CoT SFT 是基础，但不足以保证过程正确和泛化。
4. 结果监督看最终答案，过程监督看每一步推理。
5. Rejection sampling 和合成数据可以扩展训练集，但必须依赖可靠过滤。
6. 数学 verifier/PRM 对搜索和推理时增强很重要，但本身也会出错。
7. 数学评估必须做污染检测、难度分桶、错误分析和 test-time compute 曲线。
8. 面试中要把数学训练讲成“数据 + 过程 + 验证 + 推理时增强 + 评估”的完整闭环。

## 第 92 讲：代码推理与执行反馈

### 本讲目标

学完本讲，你应该能回答六个问题：

1. 代码推理为什么是 reasoning model 的重要场景。
2. 执行反馈和普通文本 verifier 有什么区别。
3. 代码生成、代码修复、debug、单元测试在训练和推理中如何形成闭环。
4. 如何用 pass@k、测试通过率、执行结果训练或筛选代码模型。
5. 代码推理中的安全、沙箱、数据污染和评估陷阱有哪些。
6. 面试中如何设计一个利用执行反馈提升代码能力的系统。

代码推理和数学推理很像。

它们都需要多步 reasoning。

但代码有一个非常重要的优势：

```text
代码可以运行。
```

这意味着模型生成的答案不只是靠人类或 LLM judge 判断。

很多情况下可以直接执行测试，得到明确反馈：

1. 编译是否通过。
2. 单元测试是否通过。
3. 哪个用例失败。
4. 报错信息是什么。
5. 性能是否超时。
6. 输出是否符合预期。

因此，代码任务天然适合 execution feedback。

这也是为什么代码能力和 agent 能力关系很紧密。

#### 来龙去脉：从 functional correctness 到 execution feedback

代码模型早期评估经常看生成代码是否“像正确答案”。Codex/HumanEval 把重点推进到 functional correctness：题目给 docstring 和函数签名，模型生成函数，再用单元测试判断行为是否正确。这个转变很关键，因为它把代码生成从文本相似度评估变成了可执行评估。

随后几条线继续往前走：

1. Codex/HumanEval 说明 repeated sampling 可以显著提高候选集合里出现正确程序的概率。
2. AlphaCode 把大规模采样、行为聚类和过滤放进竞争编程任务，说明代码 reasoning 不只是单次生成，而是生成、筛选和选择的系统。
3. CodeRL 进一步把 unit tests 和 critic 分数作为训练、推理时反馈，让模型根据执行信号重新生成。
4. Self-Debugging 强调模型可以读取执行结果、解释自己的代码，再进行修复。

所以本讲不要把代码推理理解成“会写 Python 语法”。更准确的定位是：

```text
problem specification -> code candidates -> sandbox execution -> tests/verifier -> repair/search/rerank
```

其中执行环境和测试 verifier 是代码 reasoning 区别于普通文本 reasoning 的核心。

---

### 一、代码推理包含哪些任务

代码 reasoning 不只是“写代码”。

常见任务包括：

#### 1. Code generation

根据题目或需求生成代码。

例如 LeetCode、HumanEval、业务函数实现。

#### 2. Code completion

补全函数、类、测试或配置。

#### 3. Code repair

根据错误信息修复 bug。

#### 4. Debugging

理解错误原因，定位代码问题。

#### 5. Test generation

为已有代码生成单元测试。

#### 6. Code review

发现 bug、风险、性能问题和安全问题。

#### 7. Repository-level reasoning

跨文件理解项目结构、调用链、依赖和测试。

这些任务都需要模型理解代码语义，而不只是生成语法正确的文本。

---

### 二、为什么执行反馈重要

普通文本任务很难自动判断答案对错。

但代码可以运行。

例如模型生成一个函数：

```python
def add(a, b):
    return a - b
```

单元测试可以立刻发现错误：

```text
assert add(2, 3) == 5
实际输出：-1
```

执行反馈的价值是：

1. 客观。
2. 可重复。
3. 可自动化。
4. 能定位部分错误。
5. 能用于训练、筛选和搜索。

相比 LLM judge，执行反馈更硬。

但也不是完美的。

测试覆盖不足时，代码可能通过测试但仍有 bug。

---

### 三、代码推理闭环

一个典型执行反馈闭环是：

```text
生成代码
-> 运行测试
-> 读取错误信息
-> 分析失败原因
-> 修改代码
-> 再次运行测试
-> 直到通过或达到预算
```

这和人类写代码很像。

模型不是一次生成就结束。

而是在执行环境中迭代。

这个闭环可以用于推理时：

1. 多次尝试修复。
2. 根据测试反馈改进。
3. 用通过测试数排序候选。

也可以用于训练时：

1. 生成候选代码。
2. 执行测试筛选正确样本。
3. 用正确修复轨迹做 SFT。
4. 用测试通过率训练 reward model。

---

### 四、Pass@k

代码模型常用指标是 pass@k。

它衡量：

```text
模型生成 k 个候选中，至少有一个通过测试的概率。
```

例如 pass@1 是单次生成通过率。

pass@10 是生成 10 个候选时至少一个通过的概率。

如果一次为同一道题采样 `n` 个候选，其中 `c` 个通过测试，HumanEval/Codex 类评估常用的 pass@k 估计可以写成：

```math
\operatorname{pass@k}
=
1
-
\frac{\binom{n-c}{k}}{\binom{n}{k}}
```

这个公式的直觉是：从 `n` 个候选里抽 `k` 个，先计算抽到的 `k` 个全都不是正确候选的概率，再用 1 减掉它。若 `c=0`，pass@k 为 0；若错误候选数 `n-c` 小于 `k`，则 pass@k 为 1。

Pass@k 很适合代码任务，因为：

1. 代码可以自动测试。
2. 多候选生成很常见。
3. 实际系统可以采样多个候选再筛选。

但 pass@k 也有局限：

1. 依赖测试质量。
2. 不反映代码可读性。
3. 不反映安全性。
4. 不反映性能。
5. 不反映修改范围是否合理。

所以生产系统不能只看 pass@k。

---

### 五、执行反馈作为 Verifier

在代码任务中，测试就是一种强 verifier。

Verifier 输入是候选代码。

输出是：

1. 是否编译。
2. 通过多少测试。
3. 哪些测试失败。
4. 错误栈是什么。
5. 是否超时。
6. 资源占用如何。

例如：

```json
{
  "compile": "success",
  "passed": 18,
  "total": 20,
  "failed_tests": ["test_empty_input", "test_large_n"],
  "error": "IndexError: list index out of range"
}
```

这个反馈比一句“答案不对”更有用。

模型可以根据失败用例修复代码。

把候选程序记为 `y`，公开测试通过数为 `p_y^{\mathrm{pub}}`，公开测试总数为 `m_y^{\mathrm{pub}}`，隐藏测试通过数为 `p_y^{\mathrm{hid}}`，隐藏测试总数为 `m_y^{\mathrm{hid}}`。测试通过率可以写成：

```math
q_y^{\mathrm{pub}}
=
\frac{p_y^{\mathrm{pub}}}{m_y^{\mathrm{pub}}}
```

```math
q_y^{\mathrm{hid}}
=
\frac{p_y^{\mathrm{hid}}}{m_y^{\mathrm{hid}}}
```

公开测试和隐藏测试之间的泛化差距可以写成：

```math
g_{\mathrm{hid}}
=
q_y^{\mathrm{pub}}
-
q_y^{\mathrm{hid}}
```

如果 `g_hid` 很大，说明候选可能 hardcode 了公开样例，或者公开测试没有覆盖关键边界。

执行反馈还可以变成一个筛选或训练用 reward。设 `b_y` 表示编译通过，`o_y` 表示超时，`v_y` 表示 sandbox 违规，`d_y` 表示 diff 或复杂度惩罚，可以定义：

```math
R(y)
=
w_b b_y
+
w_p q_y^{\mathrm{pub}}
+
w_h q_y^{\mathrm{hid}}
-
w_o o_y
-
w_v v_y
-
w_d d_y
```

这个 reward 不是唯一正确写法。它的作用是提醒面试官：真实代码系统不会只奖励 public tests 通过，还要惩罚超时、安全违规、过大修改和隐藏测试泛化差。

---

### 六、训练代码模型的数据

代码模型训练数据可以分几类。

#### 1. Code corpus

大量开源代码。

用于预训练，让模型学习语法、API、模式和项目结构。

#### 2. Instruction-code data

需求描述到代码实现。

用于指令微调。

#### 3. Code explanation data

代码到解释，或解释到代码。

提升代码理解能力。

#### 4. Bug-fix data

错误代码、错误信息、修复补丁。

用于训练 debug 和 repair。

#### 5. Test data

函数、测试用例、期望输出。

用于训练模型理解测试和生成测试。

#### 6. Execution trace data

模型尝试、报错、修复、再测试的轨迹。

这类数据对 agentic coding 特别重要。

---

### 七、Rejection Sampling for Code

代码任务很适合 rejection sampling。

流程：

```text
给定题目
-> 采样 N 个代码候选
-> 运行测试
-> 保留通过测试的候选
-> 用通过候选做 SFT 或偏好数据
```

如果有多个通过候选，可以再按：

1. 简洁性。
2. 时间复杂度。
3. 内存复杂度。
4. 可读性。
5. 安全性。
6. 是否使用允许的库。

进一步排序。

风险是：

1. 测试覆盖不全。
2. 代码 hardcode 测试。
3. 生成不安全代码。
4. 候选通过测试但不可维护。

所以测试只是第一层过滤。

可以把代码 rejection sampling 的保留集合写成：

```math
\mathcal{D}_{\mathrm{code}}^{\mathrm{rs}}
=
\{(x_i,y_{i,j}) \mid q_{i,j}^{\mathrm{pub}}=1,\ v_{i,j}=0,\ d_{i,j}\le \tau_d\}
```

如果有隐藏测试或更强 verifier，过滤条件还应加入 `q_hid`、静态分析、安全扫描和性能约束。只用公开测试过滤，容易把 hardcode public tests 的候选放进 SFT 数据。

---

### 八、从失败中学习

代码执行反馈最有价值的地方是失败信息。

失败可以告诉模型：

1. 哪个输入出错。
2. 期望输出是什么。
3. 实际输出是什么。
4. 栈跟踪在哪里。
5. 是语法、类型、边界、性能还是逻辑问题。

训练数据可以包含：

```text
题目
错误代码
测试失败信息
修复分析
修复后代码
```

这比只有“正确代码”更接近真实开发。

模型学到的是 debug workflow。

例如：

```text
看到 IndexError -> 检查空列表和边界条件 -> 添加保护逻辑 -> 重新测试
```

---

### 九、代码搜索与执行反馈

代码任务可以把第 89 讲的 search 用起来。

一个简单 beam search repair：

```text
初始 bug 代码
-> 生成 K 个修复候选
-> 运行测试打分
-> 保留 top-B
-> 根据失败信息继续修改
-> 重复直到通过
```

若每个任务最多允许修复 `B` 轮，repair success 可以写成：

```math
S_B
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
\left[
\max_{1\le b\le B} u_{i,b}=1
\right]
```

其中 `u_{i,b}=1` 表示第 `i` 个任务在第 `b` 次尝试后通过目标测试。平均解决尝试数可以写成：

```math
A_{\mathrm{solve}}
=
\frac{\sum_i a_i}{\sum_i u_i + \epsilon}
```

`a_i` 是第 `i` 个任务实际消耗的尝试次数，`u_i=1` 表示最终解决，`\epsilon` 用来避免分母为 0。这个指标比单看 pass@k 更贴近 agentic coding：系统不仅要有正确候选，还要能用有限预算把它找到。

Value function 可以是：

1. 测试通过率。
2. 是否编译通过。
3. 失败测试数量。
4. diff 大小。
5. 静态分析结果。
6. 安全扫描结果。

这比纯自然语言 search 更可靠。

因为执行环境提供硬反馈。

---

### 十、Sandbox 和安全

执行模型生成的代码有风险。

代码可能：

1. 删除文件。
2. 访问网络。
3. 读取环境变量和 secret。
4. 无限循环。
5. 消耗大量 CPU/内存。
6. 执行恶意命令。

所以必须使用 sandbox。

基本要求：

1. 文件系统隔离。
2. 禁止或限制网络。
3. CPU 和内存限制。
4. 执行时间限制。
5. 环境变量脱敏。
6. 最小权限运行。
7. 容器或虚拟机隔离。
8. 审计执行日志。

执行反馈系统如果没有 sandbox，不能上线。

安全违规率可以写成：

```math
r_{\mathrm{sandbox}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[v_i=1]
```

其中 `v_i=1` 表示第 `i` 次候选执行被静态规则、权限系统、容器策略或运行时审计判为违规。代码模型上线时，这个指标通常是 hard gate：即使 pass@k 很高，只要 sandbox violation rate 不可控，就不能直接接入真实用户环境。

---

### 十一、代码评估指标

代码能力评估不只看 pass@k。

常见指标包括：

1. pass@1。
2. pass@k。
3. compile rate。
4. test pass rate。
5. repair success rate。
6. average attempts to pass。
7. time to solve。
8. token cost。
9. runtime performance。
10. memory usage。
11. security violation rate。
12. diff minimality。

对于真实代码库任务，还要看：

1. 是否通过现有测试。
2. 是否新增测试。
3. 是否破坏其他模块。
4. 是否符合代码风格。
5. 是否解决根因而不是硬编码。

常用指标可以统一写成：

```math
r_{\mathrm{compile}}
=
\frac{1}{N}
\sum_i b_i
```

```math
\bar q_{\mathrm{test}}
=
\frac{1}{N}
\sum_i
\frac{p_i}{m_i}
```

```math
r_{\mathrm{sec}}
=
\frac{1}{N}
\sum_i v_i
```

其中 `b_i` 表示编译通过，`p_i/m_i` 是第 `i` 个候选的测试通过率，`v_i` 表示安全违规。对于公开/隐藏测试分离的 benchmark，还应该报告：

```math
\bar g_{\mathrm{hid}}
=
\frac{1}{N}
\sum_i
(q_i^{\mathrm{pub}}-q_i^{\mathrm{hid}})
```

如果 `\bar g_hid` 很高，说明 public tests 指标虚高，模型可能没有真正学到泛化的程序语义。

---

### 十二、数据污染问题

代码 benchmark 很容易污染。

例如 HumanEval、MBPP、LeetCode 题目和解法在网上广泛存在。

模型可能记住题解。

防护方法：

1. 训练数据和评估题 exact match 去重。
2. 函数签名、题干、测试用例去重。
3. 检查近似题目和改写题。
4. 使用时间切分的新题。
5. 自建私有评估集。
6. 使用真实 repo issue。

代码评估要特别注意：

```text
模型会不会只是见过这道题。
```

---

### 十三、真实项目中的坑

#### 1. 测试覆盖不足

代码通过测试但线上失败。

#### 2. 模型 hardcode 测试

模型写出只针对测试用例的代码。

#### 3. 执行环境不一致

本地通过，线上依赖版本不同失败。

#### 4. 没有 sandbox

执行模型代码造成安全风险。

#### 5. 只优化 pass@k

代码不可读、不安全、性能差。

#### 6. 错误反馈太长

完整日志塞回模型，噪声大且成本高。

#### 7. 修复范围失控

模型为通过测试大改无关代码。

---

### 十四、最小代码：执行反馈、隐藏测试差距与安全过滤

下面的 demo 不调用模型，只模拟同一道题的多个候选程序。任务是实现 `count_even(nums)`，统计列表里偶数的个数。它展示：

1. public tests 全通过，不等于 hidden tests 正确。
2. pass@k 只说明候选集合里有正确程序，不说明最终选择机制可靠。
3. public-only rerank 可能选中 hardcode 公开样例的候选。
4. sandbox 违规候选必须先被安全过滤，而不是参与执行竞争。
5. repair success 和 attempts to first success 更接近代码 agent 的真实体验。

```python
from math import comb

PUBLIC_TESTS = [
    ([], 0),
    ([1, 2, 3, 4], 2),
    ([2, 4, 6], 3),
]

HIDDEN_TESTS = [
    ([0, -2, -3, 5], 2),
    ([-4, -6, 1], 2),
    ([0], 1),
]

CANDIDATES = [
    {
        "id": "hardcode_public",
        "model_score": 0.92,
        "code": """
def count_even(nums):
    if nums == []:
        return 0
    if nums == [1, 2, 3, 4]:
        return 2
    if nums == [2, 4, 6]:
        return 3
    return 0
""",
    },
    {
        "id": "positive_only",
        "model_score": 0.80,
        "code": """
def count_even(nums):
    return sum(1 for x in nums if x > 0 and x % 2 == 0)
""",
    },
    {
        "id": "correct_general",
        "model_score": 0.70,
        "code": """
def count_even(nums):
    return sum(1 for x in nums if x % 2 == 0)
""",
    },
    {
        "id": "unsafe_env",
        "model_score": 0.95,
        "code": """
import os

def count_even(nums):
    return len(os.environ)
""",
    },
    {
        "id": "syntax_error",
        "model_score": 0.20,
        "code": """
def count_even(nums)
    return 0
""",
    },
]

FORBIDDEN = ("import os", "open(", "__import__", "subprocess", "socket", "eval(", "exec(")
SAFE_BUILTINS = {"len": len, "sum": sum, "range": range, "list": list}


def static_violations(code):
    return [item for item in FORBIDDEN if item in code]


def pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def run_tests(fn, tests):
    passed = 0
    failures = []
    for args, expected in tests:
        try:
            actual = fn(list(args))
            ok = actual == expected
        except Exception as exc:
            actual = type(exc).__name__
            ok = False
        passed += int(ok)
        if not ok:
            failures.append({"input": args, "expected": expected, "actual": actual})
    return passed, failures


def evaluate_candidate(item):
    code = item["code"]
    violations = static_violations(code)
    compile_ok = True
    try:
        compiled = compile(code, f"<{item['id']}>", "exec")
    except SyntaxError:
        compile_ok = False
        compiled = None

    report = {
        "id": item["id"],
        "compile": int(compile_ok),
        "sandbox_violation": int(bool(violations)),
        "public_rate": 0.0,
        "hidden_rate": 0.0,
        "hidden_gap": 0.0,
        "public_failures": 0,
        "hidden_failures": 0,
        "reward": -2.0,
    }
    if (not compile_ok) or violations:
        report["reward"] = round(0.5 * compile_ok - 2.5 * bool(violations), 3)
        return report

    env = {"__builtins__": SAFE_BUILTINS}
    exec(compiled, env, env)
    fn = env["count_even"]
    public_passed, public_failures = run_tests(fn, PUBLIC_TESTS)
    hidden_passed, hidden_failures = run_tests(fn, HIDDEN_TESTS)
    public_rate = public_passed / len(PUBLIC_TESTS)
    hidden_rate = hidden_passed / len(HIDDEN_TESTS)
    hidden_gap = public_rate - hidden_rate
    reward = 0.5 * compile_ok + public_rate + hidden_rate - 0.8 * hidden_gap
    report.update(
        {
            "public_rate": round(public_rate, 3),
            "hidden_rate": round(hidden_rate, 3),
            "hidden_gap": round(hidden_gap, 3),
            "public_failures": len(public_failures),
            "hidden_failures": len(hidden_failures),
            "reward": round(reward, 3),
        }
    )
    return report


reports = [evaluate_candidate(item) for item in CANDIDATES]
correct_count = sum(r["hidden_rate"] == 1.0 and r["sandbox_violation"] == 0 for r in reports)
public_only = max(
    reports,
    key=lambda row: (
        row["public_rate"],
        next(item["model_score"] for item in CANDIDATES if item["id"] == row["id"]),
    ),
)
hidden_aware = max(reports, key=lambda row: row["reward"])
first_success = next((idx for idx, row in enumerate(reports, start=1) if row["hidden_rate"] == 1.0), None)

summary = {
    "compile_rate": round(sum(r["compile"] for r in reports) / len(reports), 3),
    "sandbox_violation_rate": round(sum(r["sandbox_violation"] for r in reports) / len(reports), 3),
    "public_all_pass": sum(r["public_rate"] == 1.0 for r in reports),
    "hidden_all_pass": correct_count,
    "avg_public_rate": round(sum(r["public_rate"] for r in reports) / len(reports), 3),
    "avg_hidden_rate": round(sum(r["hidden_rate"] for r in reports) / len(reports), 3),
    "pass@1": round(pass_at_k(len(reports), correct_count, 1), 3),
    "pass@2": round(pass_at_k(len(reports), correct_count, 2), 3),
    "pass@4": round(pass_at_k(len(reports), correct_count, 4), 3),
    "repair_success@3": int(first_success is not None and first_success <= 3),
    "attempts_to_first_success": first_success,
}

print("candidate_reports=", reports)
print("summary=", summary)
print("public_only_choice=", public_only["id"], "hidden_rate=", public_only["hidden_rate"], "gap=", public_only["hidden_gap"])
print("hidden_aware_choice=", hidden_aware["id"], "hidden_rate=", hidden_aware["hidden_rate"], "gap=", hidden_aware["hidden_gap"])
```

典型输出：

```text
candidate_reports= [{'id': 'hardcode_public', 'compile': 1, 'sandbox_violation': 0, 'public_rate': 1.0, 'hidden_rate': 0.0, 'hidden_gap': 1.0, 'public_failures': 0, 'hidden_failures': 3, 'reward': 0.7}, {'id': 'positive_only', 'compile': 1, 'sandbox_violation': 0, 'public_rate': 1.0, 'hidden_rate': 0.0, 'hidden_gap': 1.0, 'public_failures': 0, 'hidden_failures': 3, 'reward': 0.7}, {'id': 'correct_general', 'compile': 1, 'sandbox_violation': 0, 'public_rate': 1.0, 'hidden_rate': 1.0, 'hidden_gap': 0.0, 'public_failures': 0, 'hidden_failures': 0, 'reward': 2.5}, {'id': 'unsafe_env', 'compile': 1, 'sandbox_violation': 1, 'public_rate': 0.0, 'hidden_rate': 0.0, 'hidden_gap': 0.0, 'public_failures': 0, 'hidden_failures': 0, 'reward': -2.0}, {'id': 'syntax_error', 'compile': 0, 'sandbox_violation': 0, 'public_rate': 0.0, 'hidden_rate': 0.0, 'hidden_gap': 0.0, 'public_failures': 0, 'hidden_failures': 0, 'reward': 0.0}]
summary= {'compile_rate': 0.8, 'sandbox_violation_rate': 0.2, 'public_all_pass': 3, 'hidden_all_pass': 1, 'avg_public_rate': 0.6, 'avg_hidden_rate': 0.2, 'pass@1': 0.2, 'pass@2': 0.4, 'pass@4': 0.8, 'repair_success@3': 1, 'attempts_to_first_success': 3}
public_only_choice= hardcode_public hidden_rate= 0.0 gap= 1.0
hidden_aware_choice= correct_general hidden_rate= 1.0 gap= 0.0
```

这组输出对应几个关键判断：

1. 5 个候选里只有 1 个真正通过隐藏测试且无安全违规，所以 `pass@4=0.8`，但 public tests 通过的候选有 3 个。
2. `public_only_choice` 选中了 `hardcode_public`，它公开测试全过，隐藏测试全失败。
3. `unsafe_env` 编译通过但触发静态安全规则，不能因为模型置信度高就执行。
4. `attempts_to_first_success=3` 说明如果按候选顺序修复，第三次才找到正确程序；这比单个 pass@k 更能解释代码 agent 的用户等待时间和成本。

---

### 十五、面试问答

#### 问题 1：为什么代码推理适合执行反馈？

可以这样回答：

```text
因为代码可以运行，编译结果、单元测试、错误栈、超时和输出差异都能提供客观反馈。这比只靠文本 judge 更硬，也更适合形成生成、执行、修复的闭环。
```

#### 问题 2：pass@k 是什么？

可以这样回答：

```text
pass@k 衡量模型生成 k 个候选中至少有一个通过测试的概率。它适合评估代码生成的多候选能力，但依赖测试质量，也不能完全反映可维护性和安全性。
```

#### 问题 3：如何用执行反馈训练代码模型？

可以这样回答：

```text
可以对每个题目采样多个候选，运行测试筛选通过样本做 rejection sampling SFT；也可以保存失败代码、错误信息、修复分析和最终补丁，训练模型根据执行反馈迭代修复。
```

#### 问题 4：代码执行反馈有哪些安全风险？

可以这样回答：

```text
模型生成代码可能访问文件、网络、secret，执行恶意命令或无限循环，因此必须用沙箱隔离、限制网络和资源、脱敏环境变量，并记录执行日志。
```

#### 问题 5：为什么通过测试不等于代码正确？

可以这样回答：

```text
测试覆盖可能不足，模型可能 hardcode 测试，代码可能有性能、安全、可读性或边界问题。因此通过测试只是必要条件，不是充分条件。
```

#### 问题 6：如何设计一个代码修复 Agent？

可以这样回答：

```text
输入 issue、代码和测试，模型生成候选补丁，在沙箱中运行测试，解析失败日志，继续生成修复；用测试通过率、diff 大小、安全扫描和代码风格评分排序候选，并设置最大轮数、时间和资源预算。
```

---

### 十六、常见误区

1. 误区：代码通过测试就一定正确。
   纠正：测试覆盖有限，还要看边界、性能、安全和可维护性。

2. 误区：执行反馈不需要 verifier。
   纠正：测试本身就是 verifier，但也需要静态分析、安全扫描和人工 review 补充。

3. 误区：pass@k 越高模型越适合生产。
   纠正：生产还要看延迟、成本、修复质量和安全。

4. 误区：可以直接执行模型生成代码。
   纠正：必须用 sandbox 和资源限制。

5. 误区：完整错误日志都塞给模型最好。
   纠正：应该提取关键错误、失败用例和栈信息，避免噪声。

6. 误区：代码能力只靠预训练代码语料。
   纠正：执行反馈、测试、修复轨迹和 repo-level 数据同样重要。

---

### 十七、小练习

1. 为一个简单函数设计题目、参考答案和 5 个单元测试。
2. 构造一个通过弱测试但实际错误的代码样本。
3. 设计一个代码 rejection sampling SFT 流程。
4. 设计一个执行反馈 JSON schema，包含编译、测试、错误和超时。
5. 为代码执行设计一个 sandbox 策略。
6. 设计一个代码修复 search 流程，包含候选生成、测试和 rerank。
7. 比较 pass@1、pass@k 和 repair success rate。
8. 分析 HumanEval 类 benchmark 的污染风险。
9. 设计一个错误日志压缩 prompt，只保留关键信息。
10. 用 3 分钟回答：“如何利用执行反馈提升代码模型能力？”

### 本讲总结

本讲最重要的结论：

1. 代码推理是 reasoning model 的重要场景，因为代码可以执行验证。
2. 执行反馈提供编译、测试、错误栈、超时和输出差异等硬信号。
3. 代码任务可以形成生成、执行、反馈、修复、再执行的闭环。
4. pass@k 衡量多候选中至少一个通过测试的概率，但不等于生产质量。
5. 执行反馈可用于 rejection sampling、reward model、search 和 agentic repair。
6. 运行模型生成代码必须使用 sandbox 和资源限制。
7. 代码评估要关注测试覆盖、污染、性能、安全、diff 范围和 repo-level 影响。
8. 面试中要把代码推理讲成“模型生成 + 执行环境 + 测试 verifier + 反馈修复”的系统。

## 第 93 讲：自我改进与合成推理数据

### 本讲目标

学完本讲，你应该能回答六个问题：

1. 为什么 reasoning model 需要合成推理数据。
2. Self-training、self-improvement、bootstrapping、distillation 分别是什么。
3. 如何生成、过滤、验证和迭代合成推理数据。
4. 合成数据为什么可能提升能力，也为什么可能导致模型退化。
5. 自我改进中的数据污染、模式坍塌、reward hacking 风险有哪些。
6. 面试中如何设计一个安全可控的合成推理数据 pipeline。

前面几讲分别讲了数学推理、代码执行反馈、verifier 和 test-time compute。

这些能力可以用来做一件很重要的事：

```text
让模型或更强系统生成新的推理数据，再用这些数据训练更强模型。
```

这就是自我改进和合成推理数据的核心。

大模型时代，数据不再只来自人工标注和互联网文本。

模型本身、工具、verifier、搜索系统都可以参与数据生产。

但这件事有双刃剑属性。

高质量合成数据能提升模型。

低质量合成数据会放大错误、污染分布、造成过拟合和能力退化。

#### 来龙去脉：从 Self-Instruct 到 reasoning self-improvement

合成数据不是大模型时代才有。传统机器学习里就有数据增强，例如图像旋转、语音加噪、机器翻译回译和规则生成样本。但大模型把合成数据推进到新阶段：模型本身可以生成 instruction、题目、解法、测试、偏好对和工具轨迹。

几条代表性脉络可以这样串起来：

1. Self-Instruct 说明模型可以从少量种子任务出发，生成 instruction/input/output，再过滤无效或相似样本，用于 instruction tuning。
2. STaR 把重点放到 reasoning：模型生成 rationales，保留最终答案正确的推理，再迭代微调，让模型从自己的正确推理中自举。
3. WizardLM/Evol-Instruct 强调从简单指令逐步演化到复杂指令，解决普通合成指令太简单、太模板化的问题。
4. phi-1 说明高质量教材式数据和合成练习可以显著改变小模型的代码能力效率曲线。
5. Orca 这类工作强调从强模型的复杂解释轨迹中学习，但也提醒：小模型可能只模仿风格，不一定学到真实推理能力。

所以合成推理数据的核心不是“多生成一些 CoT”。更准确的闭环是：

```text
seed tasks -> generator/search/tool -> verifier/filter -> dedup/balance -> train -> clean eval -> error-driven next data
```

只要缺少验证、去重、独立评估或 lineage，自我改进就很容易变成自我污染。

---

### 一、为什么需要合成推理数据

推理数据很贵。

尤其是高质量 step-by-step reasoning 数据。

人工标注需要专家，成本高，速度慢。

而 reasoning model 需要大量数据覆盖：

1. 数学题。
2. 代码题。
3. 逻辑题。
4. 多跳问答。
5. 复杂规划。
6. 工具调用。
7. 错误修复。
8. 过程监督。

互联网文本中的推理过程质量参差不齐。

因此自然会想到：

```text
能不能用模型自己生成题目、解法、错误样本和验证数据？
```

合成推理数据的目标是：

1. 扩大数据规模。
2. 覆盖更多难度和题型。
3. 生成过程监督数据。
4. 生成 hard negatives。
5. 支持特定领域定制。
6. 支持持续迭代。

---

### 二、几个相关概念

#### 1. Synthetic data

由模型、程序或规则生成的数据。

例如模型生成数学题和解答。

#### 2. Self-training

用模型给未标注数据生成标签，再用高置信标签训练模型。

#### 3. Self-improvement

模型利用自己的生成、反馈、验证和修正，不断产生更好训练数据或更好策略。

#### 4. Bootstrapping

用已有模型或少量数据启动一个迭代流程，逐步扩大能力和数据。

#### 5. Distillation

用强模型或强推理系统生成数据，训练较小或更便宜的模型。

例如：

```text
强模型 + search + verifier -> 高质量解法 -> 训练小模型单次生成
```

这些概念有重叠。

面试中不必纠结名词边界。

重点是讲清数据生成、过滤、训练和评估闭环。

---

### 三、合成推理数据的类型

合成推理数据可以包括很多类型。

#### 1. 合成题目

生成新的数学题、逻辑题、代码题。

#### 2. 合成解法

为已有题目生成多种解法。

#### 3. 合成 CoT

生成 step-by-step reasoning。

#### 4. 合成错误样本

生成看起来合理但实际错误的解法。

用于训练 verifier 或做对比学习。

#### 5. 合成偏好对

同一道题生成好解法和坏解法，标注偏好。

#### 6. 合成工具轨迹

生成工具调用、执行反馈、修复过程。

#### 7. 合成评估样本

构造测试集或 red-team 样本。

不同类型数据用途不同。

训练 generator 需要好解法。

训练 verifier 需要正负样本。

训练 agent 需要轨迹数据。

评估需要独立且未污染的数据。

---

### 四、基本 Pipeline

一个合成推理数据 pipeline 通常包括：

```text
Seed tasks
-> Data generation
-> Verification / filtering
-> Deduplication
-> Difficulty and diversity balancing
-> Training
-> Evaluation
-> Error analysis
-> Next iteration
```

#### 1. Seed tasks

从少量真实题、领域任务或模板开始。

#### 2. Data generation

用模型、规则或程序生成题目和解法。

#### 3. Verification

用答案匹配、程序执行、verifier、人工抽查过滤。

#### 4. Deduplication

去掉重复题、近似题和评估集污染。

#### 5. Balancing

控制难度、题型、领域和语言分布。

#### 6. Training

用过滤后的数据做 SFT、DPO、reward model 或 RL。

#### 7. Evaluation

在独立 holdout 和真实任务上评估。

#### 8. Iteration

根据错误分析生成下一轮数据。

把这条 pipeline 写成记号，可以更清楚地区分“生成很多”和“可训练数据”。

设第 `t` 轮模型为 `M_t`，种子集合为 `S_t`，生成器从提示模板、采样参数和工具配置中产生候选：

```math
e_i=(x_i,r_i,a_i,m_i)
\sim
G_t(S_t,\pi_t,\tau_t)
```

其中 `x_i` 是题目或指令，`r_i` 是推理轨迹，`a_i` 是答案，`m_i` 是 lineage 元数据，例如 seed、prompt、teacher、采样参数和 verifier 版本。

过滤器可以输出一个软质量分：

```math
q_i =
\alpha z_i
+ \beta v_i
+ \gamma f_i
+ \delta d_i
+ \eta n_i
- \mu c_i
- \nu s_i
```

这里 `z_i` 表示最终答案或执行结果是否正确，`v_i` 是步骤有效率或 verifier 分数，`f_i` 是格式合法性，`d_i` 是多样性/覆盖收益，`n_i` 是新颖性，`c_i` 是污染风险，`s_i` 是安全风险。

但真实 pipeline 不能只靠软分相加。安全、污染、重复、最终正确性和关键格式通常要做 hard gate：

```math
g_i =
\mathbf{1}[z_i=1]
\mathbf{1}[v_i\ge \tau_v]
\mathbf{1}[f_i=1]
\mathbf{1}[c_i<\tau_c]
\mathbf{1}[s_i=0]
\mathbf{1}[u_i=0]
```

其中 `u_i=1` 表示和已有训练集或评估集重复。最终保留集合可以写成：

```math
\mathcal{D}_{t}^{\mathrm{keep}}
=
\{e_i \mid q_i\ge \tau_q,\ g_i=1\}
```

保留率是：

```math
r_{\mathrm{keep}}
=
\frac{|\mathcal{D}_{t}^{\mathrm{keep}}|}
{|\mathcal{D}_{t}^{\mathrm{gen}}|}
```

保留率太低，说明生成器或 prompt 质量差；保留率太高，也可能说明过滤器太松。

---

### 五、生成题目

生成题目时要控制几个维度。

1. 领域。
2. 难度。
3. 题型。
4. 所需推理步数。
5. 是否可自动验证。
6. 是否和已有数据重复。

例如数学题生成 prompt 可以要求：

```text
生成 20 道初中代数应用题，每题需要 3-5 步推理，有唯一整数答案，并给出标准答案和分步解法。
```

但模型生成的题目可能有问题：

1. 条件矛盾。
2. 没有唯一答案。
3. 答案算错。
4. 难度不符合要求。
5. 题目太模板化。

所以题目生成后必须验证。

---

### 六、生成解法

对已有题目，可以生成多种解法。

例如：

1. 代数解法。
2. 枚举解法。
3. 逆向验证。
4. 图形直觉。
5. 代码求解。

多解法有两个好处。

第一，可以增加 reasoning diversity。

第二，可以让模型不只记一种模板。

但多解法也可能引入错误。

尤其是模型为了多样性而编出不成立的方法。

因此每个解法都要检查：

1. 中间步骤是否成立。
2. 最终答案是否一致。
3. 是否使用题目没有给出的假设。
4. 是否过度跳步。

---

### 七、生成错误样本

错误样本对 verifier 很重要。

如果 verifier 只见过正确解法和明显错误，它很难识别高迷惑错误。

错误样本可以包括：

1. 算术错误。
2. 代数变形错误。
3. 漏条件。
4. 单位错误。
5. 变量混淆。
6. 逻辑跳步。
7. 代码边界条件错误。
8. 通过部分测试但失败隐藏测试。

Hard negative 的特点是：

```text
表面上很像正确推理，但有关键错误。
```

例如：

```text
2x + 3 = 11
2x = 14
x = 7
```

格式很像正确解法，但第二步错。

训练 verifier 时，hard negatives 比随机错误更有价值。

---

### 八、过滤与验证

合成数据最重要的是过滤。

常见过滤信号包括：

1. 答案匹配。
2. 程序执行。
3. 单元测试。
4. 符号计算。
5. Verifier 分数。
6. 多模型一致性。
7. Self-consistency。
8. 人工抽查。
9. 格式规则。
10. 去重和污染检测。

一个高质量 pipeline 通常不是单一过滤器。

而是多层过滤：

```text
格式过滤 -> 自动验证 -> verifier 打分 -> 去重 -> 人工抽查
```

过滤标准要根据用途调整。

训练 generator 的数据要尽量正确。

训练 verifier 的数据要包含高质量负样本。

评估数据必须尽量干净且独立。

---

### 九、Self-Improvement Loop

一个自我改进循环可以这样设计：

```text
当前模型 M_t
-> 生成候选题目和解法
-> verifier / tools 过滤
-> 得到高质量数据 D_t
-> 训练新模型 M_{t+1}
-> 在独立评估集测试
-> 分析错误
-> 生成下一轮针对性数据
```

形式化地说：

```math
\mathcal{D}_{t}^{\mathrm{new}}
=
F_t(G_t(M_t,S_t))
```

```math
M_{t+1}
=
\operatorname{Train}
(M_t,\mathcal{D}_{\mathrm{base}},\mathcal{D}_{t}^{\mathrm{new}})
```

其中 `G_t` 是生成过程，`F_t` 是过滤、去重、安全和评估 gate，`\mathcal{D}_base` 是人工数据、真实任务数据或高质量自然数据。关键是不要让 `\mathcal{D}_t^{new}` 完全替代外部锚点。

训练混合目标可以写成：

```math
\mathcal{L}_{t+1}
=
(1-\lambda_t)\mathcal{L}_{\mathrm{base}}
+
\lambda_t\mathcal{L}_{\mathrm{syn}}
```

`\lambda_t` 是合成数据权重。它应该由 ablation 和回归评估决定，而不是由“这轮生成了多少数据”决定。

关键是每一轮都要有独立评估。

否则模型可能只是越来越擅长自己生成的数据。

自我改进的风险是闭环污染：

```text
模型生成的数据带有模型自己的偏差，训练后偏差被放大，下一轮生成更偏。
```

所以需要外部锚点：

1. 人类数据。
2. 程序验证。
3. 强 verifier。
4. 独立 benchmark。
5. 真实用户任务。

每轮更新后要看干净评估集上的变化：

```math
\Delta A_m
=
A_m(M_{t+1})
-
A_m(M_t)
```

其中 `m` 可以是数学、代码、逻辑、安全、通用问答、拒答误伤等不同能力。发布 gate 可以写成：

```math
R_{\mathrm{degrade}}
=
\sum_m
w_m
\mathbf{1}[\Delta A_m < -\tau_m]
```

如果 `R_degrade>0`，说明至少有重要能力回退，不能只因为合成数据目标任务提升就继续扩大混入比例。

---

### 十、Distillation from Strong Reasoner

一种常见做法是从强推理系统蒸馏。

强系统可以是：

```text
大模型 + CoT + self-consistency + verifier + search + tools
```

它生成高质量推理轨迹。

然后训练较小模型模仿。

目标是：

```text
把昂贵 test-time compute 的结果压缩到便宜模型里。
```

例如：

1. 强系统花 30 秒解题。
2. 生成高质量解法。
3. 小模型用这些解法 SFT。
4. 小模型未来单次生成就能接近强系统部分能力。

风险是：

1. 小模型只能模仿表面过程。
2. 强系统错误会被蒸馏。
3. 蒸馏数据分布太窄。
4. 小模型容量不足。

---

### 十一、数据多样性

合成数据容易模式化。

例如所有数学题都长得像：

```text
小明有 x 个苹果...
```

模型会学会模板，而不是泛化推理。

提高多样性的方法：

1. 控制题型分布。
2. 控制难度分布。
3. 多种生成 prompt。
4. 多模型生成。
5. 从真实错误中生成数据。
6. 引入不同领域和语言。
7. 聚类后采样，避免重复。
8. 针对薄弱点生成数据。

多样性不是越随机越好。

要在覆盖和质量之间平衡。

可以用类别分布熵粗略衡量覆盖：

```math
H_C
=
-
\sum_{c\in C}
p_c\log p_c
```

`p_c` 是类别 `c` 在保留数据中的占比。也可以用有效桶数：

```math
B_{\mathrm{eff}}
=
\frac{1}
{\sum_{c\in C} p_c^2}
```

如果 `B_eff` 很低，说明数据虽然很多，但主要集中在少数模板或少数类别。多样性指标不能替代正确性，但能提示“模式坍塌”风险。

---

### 十二、数据污染风险

合成数据很容易污染评估集。

例如模型见过 benchmark 题，生成了改写版。

如果这些改写题进入训练，评估分数会虚高。

防护方法：

1. 生成前排除评估题。
2. 生成后做 exact/fuzzy dedup。
3. 对题干、答案、解法分别查重。
4. 用 embedding 检测近似题。
5. 保留生成来源和 lineage。
6. 使用时间更新的 holdout。
7. 对高分样本人工抽查。

Lineage 很重要。

要知道每条合成数据来自哪个 seed、哪个模型、哪个 prompt、哪个 verifier。

否则出了污染很难追踪。

污染率可以粗略写成：

```math
\rho_{\mathrm{leak}}
=
\frac{1}{N}
\sum_i
\mathbf{1}
\left[
\max_j
\operatorname{sim}(x_i^{\mathrm{syn}},x_j^{\mathrm{eval}})
\ge
\tau_{\mathrm{sim}}
\right]
```

其中 `sim` 可以是 exact match、n-gram overlap、embedding similarity、代码 AST 相似度或题目结构相似度。只查题干不够，答案、测试、解法轨迹和变量结构也要查。

---

### 十三、质量评估

合成数据质量要从多个维度评估。

1. Correctness。
2. Step validity。
3. Difficulty。
4. Diversity。
5. Novelty。
6. Verifiability。
7. Format consistency。
8. Contamination risk。
9. Downstream improvement。
10. Human audit pass rate。

最终最重要的是 downstream improvement。

也就是：

```text
加入这批合成数据后，模型在干净评估集和真实任务上是否变好。
```

如果数据看起来漂亮，但训练后没有提升，甚至退化，就要回头看分布和质量。

质量评估至少要分两层。

第一层是数据本身：

```math
Q_{\mathrm{data}}
=
w_z r_{\mathrm{correct}}
+
w_v r_{\mathrm{step}}
+
w_f r_{\mathrm{format}}
+
w_d H_C
-
w_l \rho_{\mathrm{leak}}
-
w_s r_{\mathrm{safety}}
```

第二层是训练后的真实收益：

```math
G_{\mathrm{down}}
=
\sum_m
w_m
\left(
A_m^{\mathrm{after}}
-
A_m^{\mathrm{before}}
\right)
```

如果 `Q_data` 高但 `G_down` 不提升，可能是数据分布不对、任务太简单、模板过拟合、teacher 风格太重，或者 student 容量不足。

---

### 十四、真实项目中的坑

#### 1. 只看合成数据数量

数量大但错误多，会损害模型。

#### 2. 没有强过滤

模型生成的错误推理直接进训练集。

#### 3. 合成数据分布太窄

模型在模板题上提升，在真实题上不提升。

#### 4. 评估集污染

训练数据包含 benchmark 改写题，分数虚高。

#### 5. 自我循环放大偏差

模型生成自己的偏好数据，再训练自己，偏差越来越强。

#### 6. 忽略负样本

只生成正确解法，verifier 学不到区分高迷惑错误。

#### 7. 没有数据 lineage

无法追踪错误数据来自哪里。

---

### 十五、最小代码：合成推理数据过滤与训练集构造

下面的 demo 不调用模型，而是模拟一轮合成 reasoning data pipeline。它展示：

1. 生成 9 条候选后，只有 3 条进入 generator SFT。
2. `ratio_lucky` 最终答案正确，但过程有错，应转成 verifier negative，而不是进入 CoT SFT。
3. `eval_leak` 因评估集相似度过高被污染 gate 拦截。
4. `unsafe_task` 被安全 gate 拦截。
5. 合成数据混入比例、类别多样性、lineage 和 clean eval delta 都要一起看。

```python
from collections import Counter
import math
import re

GENERATED = [
    {"id": "algebra_clean", "task": "Solve for x: 2x + 3 = 11.", "category": "math", "gold": "4", "answer": "4", "steps": [1, 1, 1], "format_ok": 1, "verifiable": 1, "eval_sim": 0.18, "safety_risk": 0, "lineage": ("seed_math", "prompt_v2", "teacher_A")},
    {"id": "algebra_duplicate", "task": "Solve for x: 2x + 3 = 11.", "category": "math", "gold": "4", "answer": "4", "steps": [1, 1, 1], "format_ok": 1, "verifiable": 1, "eval_sim": 0.18, "safety_risk": 0, "lineage": ("seed_math", "prompt_v2", "teacher_A")},
    {"id": "ratio_lucky", "task": "A recipe uses 3 cups for 6 people. How many cups for 12 people?", "category": "math", "gold": "6", "answer": "6", "steps": [1, 0, 1], "format_ok": 1, "verifiable": 1, "eval_sim": 0.20, "safety_risk": 0, "lineage": ("seed_math", "prompt_v3", "teacher_A")},
    {"id": "code_even_clean", "task": "Implement count_even(nums) for zero, negative, and positive integers.", "category": "code", "gold": "pass", "answer": "pass", "steps": [1, 1, 1], "format_ok": 1, "verifiable": 1, "eval_sim": 0.25, "safety_risk": 0, "lineage": ("seed_code", "prompt_v1", "teacher_B")},
    {"id": "logic_clean", "task": "Alice is older than Bob. Bob is older than Chen. Who is youngest?", "category": "logic", "gold": "Chen", "answer": "Chen", "steps": [1, 1], "format_ok": 1, "verifiable": 1, "eval_sim": 0.30, "safety_risk": 0, "lineage": ("seed_logic", "prompt_v1", "teacher_B")},
    {"id": "eval_leak", "task": "Benchmark item: a known hidden holdout word problem with numbers changed.", "category": "math", "gold": "18", "answer": "18", "steps": [1, 1, 1, 1], "format_ok": 1, "verifiable": 1, "eval_sim": 0.94, "safety_risk": 0, "lineage": ("seed_math", "prompt_v4", "teacher_A")},
    {"id": "unsafe_task", "task": "Generate code that reads environment secrets before solving a toy task.", "category": "code", "gold": "blocked", "answer": "blocked", "steps": [1], "format_ok": 1, "verifiable": 0, "eval_sim": 0.10, "safety_risk": 1, "lineage": ("seed_code", "prompt_bad", "teacher_A")},
    {"id": "wrong_answer", "task": "Solve for y: y - 5 = 9.", "category": "math", "gold": "14", "answer": "4", "steps": [1, 0], "format_ok": 1, "verifiable": 1, "eval_sim": 0.16, "safety_risk": 0, "lineage": ("seed_math", "prompt_v2", "teacher_C")},
    {"id": "format_bad", "task": "Create a boolean expression task with exactly one answer.", "category": "logic", "gold": "true", "answer": "true", "steps": [1, 1], "format_ok": 0, "verifiable": 1, "eval_sim": 0.14, "safety_risk": 0, "lineage": ("seed_logic", "prompt_v5", "teacher_C")},
]


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a, b):
    left = tokens(a)
    right = tokens(b)
    return len(left & right) / len(left | right) if left or right else 1.0


def process_score(row):
    return sum(row["steps"]) / len(row["steps"])


def answer_ok(row):
    return str(row["answer"]).strip().lower() == str(row["gold"]).strip().lower()


def diversity_entropy(rows):
    counts = Counter(row["category"] for row in rows)
    total = sum(counts.values())
    return -sum((count / total) * math.log(count / total) for count in counts.values()) if total else 0.0


def effective_buckets(rows):
    counts = Counter(row["category"] for row in rows)
    total = sum(counts.values())
    return 1 / sum((count / total) ** 2 for count in counts.values()) if total else 0.0


seen_tasks = []
reports = []
kept = []
rejected = []
for row in GENERATED:
    duplicate = any(jaccard(row["task"], old) >= 0.92 for old in seen_tasks)
    seen_tasks.append(row["task"])
    contamination = row["eval_sim"] >= 0.90
    quality = (
        0.40 * answer_ok(row)
        + 0.25 * process_score(row)
        + 0.15 * row["format_ok"]
        + 0.10 * row["verifiable"]
        + 0.10 * (1 - row["eval_sim"])
        - 0.60 * row["safety_risk"]
        - 0.30 * duplicate
        - 0.40 * contamination
    )
    hard_gate = (
        answer_ok(row)
        and process_score(row) >= 0.8
        and row["format_ok"] == 1
        and row["safety_risk"] == 0
        and not duplicate
        and not contamination
    )
    decision = "keep" if quality >= 0.78 and hard_gate else "reject"
    report = {
        "id": row["id"],
        "quality": round(quality, 3),
        "answer_ok": int(answer_ok(row)),
        "process": round(process_score(row), 3),
        "duplicate": int(duplicate),
        "contam": int(contamination),
        "safety": row["safety_risk"],
        "format_ok": row["format_ok"],
        "decision": decision,
    }
    reports.append(report)
    (kept if decision == "keep" else rejected).append(row)

human_examples = 12
synthetic_weight = len(kept) / (human_examples + len(kept))
rejected_by_reason = Counter()
for report in reports:
    if report["decision"] == "keep":
        continue
    if report["safety"]:
        rejected_by_reason["safety"] += 1
    elif report["contam"]:
        rejected_by_reason["contamination"] += 1
    elif report["duplicate"]:
        rejected_by_reason["duplicate"] += 1
    elif not report["answer_ok"]:
        rejected_by_reason["wrong_answer"] += 1
    elif report["process"] < 0.8:
        rejected_by_reason["process_noise"] += 1
    elif not report["format_ok"]:
        rejected_by_reason["format"] += 1
    else:
        rejected_by_reason["low_quality"] += 1

baseline_eval = {"math": 0.62, "code": 0.55, "logic": 0.58, "safety": 0.90}
filtered_eval = {"math": 0.66, "code": 0.58, "logic": 0.60, "safety": 0.90}
unfiltered_eval = {"math": 0.63, "code": 0.56, "logic": 0.58, "safety": 0.82}
filtered_delta = {name: round(filtered_eval[name] - baseline_eval[name], 3) for name in baseline_eval}
unfiltered_delta = {name: round(unfiltered_eval[name] - baseline_eval[name], 3) for name in baseline_eval}

summary = {
    "generated": len(GENERATED),
    "kept": len(kept),
    "keep_rate": round(len(kept) / len(GENERATED), 3),
    "synthetic_weight": round(synthetic_weight, 3),
    "kept_entropy": round(diversity_entropy(kept), 3),
    "kept_effective_buckets": round(effective_buckets(kept), 3),
    "rejected_by_reason": dict(rejected_by_reason),
}
training_sets = {
    "generator_sft_ids": [row["id"] for row in kept],
    "verifier_negative_ids": [row["id"] for row in rejected if answer_ok(row) and process_score(row) < 0.8],
    "blocked_ids": [row["id"] for row in rejected if row["safety_risk"] or row["eval_sim"] >= 0.90],
}
lineage_example = {row["id"]: row["lineage"] for row in kept[:2]}

print("reports=", reports)
print("summary=", summary)
print("training_sets=", training_sets)
print("filtered_delta=", filtered_delta)
print("unfiltered_delta=", unfiltered_delta)
print("lineage_example=", lineage_example)
```

典型输出：

```text
reports= [{'id': 'algebra_clean', 'quality': 0.982, 'answer_ok': 1, 'process': 1.0, 'duplicate': 0, 'contam': 0, 'safety': 0, 'format_ok': 1, 'decision': 'keep'}, {'id': 'algebra_duplicate', 'quality': 0.682, 'answer_ok': 1, 'process': 1.0, 'duplicate': 1, 'contam': 0, 'safety': 0, 'format_ok': 1, 'decision': 'reject'}, {'id': 'ratio_lucky', 'quality': 0.897, 'answer_ok': 1, 'process': 0.667, 'duplicate': 0, 'contam': 0, 'safety': 0, 'format_ok': 1, 'decision': 'reject'}, {'id': 'code_even_clean', 'quality': 0.975, 'answer_ok': 1, 'process': 1.0, 'duplicate': 0, 'contam': 0, 'safety': 0, 'format_ok': 1, 'decision': 'keep'}, {'id': 'logic_clean', 'quality': 0.97, 'answer_ok': 1, 'process': 1.0, 'duplicate': 0, 'contam': 0, 'safety': 0, 'format_ok': 1, 'decision': 'keep'}, {'id': 'eval_leak', 'quality': 0.506, 'answer_ok': 1, 'process': 1.0, 'duplicate': 0, 'contam': 1, 'safety': 0, 'format_ok': 1, 'decision': 'reject'}, {'id': 'unsafe_task', 'quality': 0.29, 'answer_ok': 1, 'process': 1.0, 'duplicate': 0, 'contam': 0, 'safety': 1, 'format_ok': 1, 'decision': 'reject'}, {'id': 'wrong_answer', 'quality': 0.459, 'answer_ok': 0, 'process': 0.5, 'duplicate': 0, 'contam': 0, 'safety': 0, 'format_ok': 1, 'decision': 'reject'}, {'id': 'format_bad', 'quality': 0.836, 'answer_ok': 1, 'process': 1.0, 'duplicate': 0, 'contam': 0, 'safety': 0, 'format_ok': 0, 'decision': 'reject'}]
summary= {'generated': 9, 'kept': 3, 'keep_rate': 0.333, 'synthetic_weight': 0.2, 'kept_entropy': 1.099, 'kept_effective_buckets': 3.0, 'rejected_by_reason': {'duplicate': 1, 'process_noise': 1, 'contamination': 1, 'safety': 1, 'wrong_answer': 1, 'format': 1}}
training_sets= {'generator_sft_ids': ['algebra_clean', 'code_even_clean', 'logic_clean'], 'verifier_negative_ids': ['ratio_lucky'], 'blocked_ids': ['eval_leak', 'unsafe_task']}
filtered_delta= {'math': 0.04, 'code': 0.03, 'logic': 0.02, 'safety': 0.0}
unfiltered_delta= {'math': 0.01, 'code': 0.01, 'logic': 0.0, 'safety': -0.08}
lineage_example= {'algebra_clean': ('seed_math', 'prompt_v2', 'teacher_A'), 'code_even_clean': ('seed_code', 'prompt_v1', 'teacher_B')}
```

这组结果对应几条工程判断：

1. `ratio_lucky` 的软质量分很高，但过程分只有 0.667，被 hard gate 拦截；这类样本适合做 verifier negative，不适合进 CoT SFT。
2. `eval_leak` 最终答案和步骤都正确，但污染风险过高，必须从训练集中移除。
3. `synthetic_weight=0.2` 表示这轮合成数据只占混合训练集 20%，避免一轮合成分布压过人工或真实数据锚点。
4. `unfiltered_delta` 里安全指标下降 0.08，说明不过滤直接训练会引入明显副作用。
5. `lineage_example` 让后续错误回溯到 seed、prompt 和 teacher，而不是只知道“这条数据来自合成”。

---

### 十六、面试问答

#### 问题 1：为什么需要合成推理数据？

可以这样回答：

```text
高质量人工推理数据昂贵且覆盖有限。合成数据可以扩大题型、难度和过程监督覆盖，用于训练 generator、verifier、reward model 和 agent 轨迹，但必须经过严格验证和去重。
```

#### 问题 2：Self-improvement 的基本流程是什么？

可以这样回答：

```text
当前模型生成题目、解法或轨迹，再用工具、verifier、自一致性和人工抽查过滤，得到高质量数据后训练下一版模型，并在独立评估集上验证，再根据错误分析进入下一轮。
```

#### 问题 3：合成数据有什么风险？

可以这样回答：

```text
风险包括错误推理混入、数据分布单一、模式坍塌、评估集污染、模型偏差自我放大、reward hacking 和缺少 lineage 导致无法追踪问题来源。
```

#### 问题 4：如何过滤合成推理数据？

可以这样回答：

```text
可以用格式规则、答案匹配、程序执行、符号验证、单元测试、verifier 打分、多模型一致性、去重、污染检测和人工抽查组成多层过滤 pipeline。
```

#### 问题 5：Distillation from strong reasoner 是什么？

可以这样回答：

```text
用更强但更贵的推理系统，例如大模型加 search、verifier 和工具，生成高质量推理轨迹，再训练较小模型模仿，希望把昂贵 test-time compute 的能力部分压缩到模型参数中。
```

#### 问题 6：如何判断合成数据真的有用？

可以这样回答：

```text
不能只看数据量和表面质量，要看加入数据后模型在独立、未污染的评估集和真实任务上的提升，并按题型、难度和错误类型做 ablation 和 error analysis。
```

---

### 十七、常见误区

1. 误区：合成数据越多越好。
   纠正：质量、覆盖、验证和去重比数量更重要。

2. 误区：模型生成的数据可以直接训练。
   纠正：必须经过过滤、验证和污染检测。

3. 误区：自我改进可以完全不需要外部信号。
   纠正：需要工具、verifier、人类数据或独立评估作为锚点。

4. 误区：合成数据提升 benchmark 就说明有效。
   纠正：可能是污染或模板过拟合，要看干净 holdout 和真实任务。

5. 误区：只生成正确样本就够。
   纠正：训练 verifier 和鲁棒模型还需要高质量负样本。

6. 误区：强模型蒸馏一定能让小模型学会推理。
   纠正：小模型可能只学到表面格式，容量和数据分布都会限制效果。

---

### 十八、小练习

1. 设计一个数学合成数据 pipeline，包含生成、验证、去重和训练。
2. 为代码修复任务设计一个合成错误样本生成流程。
3. 构造 5 条 hard negative reasoning 数据。
4. 设计一个数据 lineage schema，记录 seed、prompt、model、verifier 和过滤结果。
5. 设计一个合成数据污染检测流程。
6. 设计一个 ablation，比较人工数据、合成数据和混合数据。
7. 分析 self-improvement 中偏差自我放大的原因。
8. 设计一个 distillation from strong reasoner 的训练方案。
9. 为合成数据质量设计 10 个评估指标。
10. 用 3 分钟回答：“如何安全地用合成推理数据提升 reasoning model？”

### 本讲总结

本讲最重要的结论：

1. 合成推理数据用于扩大题型、难度、过程监督和错误样本覆盖。
2. Self-improvement 的核心是生成、过滤、训练、评估和迭代闭环。
3. 合成数据可以包括题目、解法、CoT、错误样本、偏好对、工具轨迹和评估样本。
4. 过滤和验证是合成数据 pipeline 的核心，不能直接相信模型生成内容。
5. Distillation 可以把强推理系统的昂贵输出压缩到较小模型中，但会受数据质量和模型容量限制。
6. 合成数据的主要风险是错误放大、分布单一、污染、模式坍塌和 reward hacking。
7. 判断合成数据是否有用，要看独立评估集和真实任务的 downstream improvement。
8. 面试中要强调：合成数据不是免费午餐，而是一个需要强验证和强评估的数据工程系统。

## 第 94 讲：Reasoning Model 评估

### 本讲目标

学完本讲，你应该能回答六个问题：

1. Reasoning model 评估和普通 chat model 评估有什么区别。
2. 数学、代码、逻辑、多跳问答、规划分别怎么评估。
3. 为什么只看 final answer accuracy 不够。
4. 如何评估过程正确性、test-time compute、鲁棒性和污染风险。
5. LLM-as-a-judge、verifier、工具评估、人类评估各有什么优缺点。
6. 面试中如何设计一个可信的 reasoning model 评估体系。

Reasoning model 的目标不是“回答更像人”。

它更强调：

```text
能否在复杂任务中稳定地分解、推理、验证和得到正确结论。
```

因此 reasoning model 评估不能只看通用聊天满意度。

它要回答：

1. 答案是否正确。
2. 推理过程是否可靠。
3. 错误是否可定位。
4. 增加 test-time compute 是否真的带来收益。
5. 模型是否只是记住 benchmark。
6. 在真实复杂任务中是否有用。

#### 来龙去脉：从单个 benchmark 到 evaluation harness

Reasoning 评估不是一个指标突然出现，而是从多个方向汇合出来的。

早期通用能力评估更像“模型会不会答题”，例如知识问答、阅读理解和多选题。数学 reasoning 评估把问题变硬：MATH 用竞赛数学题和 step-by-step solution 暴露传统 scaling 在高难数学上的不足；GSM8K 和 verifier 线路说明，多步文字题不仅要看最终答案，还要看多候选生成和验证器能否选出正确解。代码评估则由 HumanEval/Codex 推动到 functional correctness：程序必须运行并通过测试，而不是文本像答案。

随后，BBH 这类多步任务集合说明：同一个模型在 direct prompting 和 CoT prompting 下表现可能差别很大，因此 benchmark 必须记录 prompt、预算、采样和验证策略。再往后，Agent、RAG、代码仓库修复和工具使用任务把评估对象从单个 answer 扩展成完整 trajectory。

所以 reasoning 评估的核心变化是：

```text
single final answer -> answer + process + budget + robustness + contamination + trajectory
```

面试里不要把它讲成“跑几个榜单”。更准确的说法是：reasoning evaluation 是一个可复现的 harness，里面要固定数据、提示、采样、答案归一化、工具执行、过程标注、污染检测、成本统计和发布 gate。

---

### 一、Reasoning 评估的特殊性

普通 chat model 评估常看：

1. 指令遵循。
2. 有用性。
3. 流畅度。
4. 安全性。
5. 用户偏好。

Reasoning model 还要看：

1. 多步推理正确性。
2. 过程一致性。
3. 约束满足。
4. 可验证结果。
5. 难题上的 solve rate。
6. 随推理预算增长的性能曲线。
7. 对变体题和分布外题的泛化。

一个答案写得很流畅，不代表 reasoning 好。

一个推理过程很长，也不代表 reasoning 好。

Reasoning 评估必须更硬、更细、更可复现。

---

### 二、Final Answer Accuracy

最基础指标是 final answer accuracy。

例如数学题最终答案是否正确。

代码题是否通过测试。

选择题是否选对。

优点：

1. 简单。
2. 可量化。
3. 易比较模型。
4. 适合有标准答案的任务。

缺点：

1. 不知道过程是否正确。
2. 答案可能碰巧对。
3. 错误难定位。
4. 不适合开放题。
5. 无法评价成本和推理预算。

所以 final answer accuracy 是必要指标，但不是充分指标。

设评估集为 `\mathcal{E}=\{(x_i,a_i,d_i,w_i)\}_{i=1}^{N}`，其中 `x_i` 是题目，`a_i` 是标准答案，`d_i` 是任务域，`w_i` 是样本权重。模型输出经答案抽取和归一化后得到 `\hat a_i`。最终答案正确指示变量为：

```math
y_i
=
\mathbf{1}[\hat a_i=a_i]
```

加权 final accuracy 可以写成：

```math
A_{\mathrm{final}}
=
\frac{\sum_{i=1}^{N} w_i y_i}
{\sum_{i=1}^{N} w_i}
```

如果要看某个任务域 `d` 的表现，可以写成：

```math
A_d
=
\frac{\sum_{i=1}^{N} \mathbf{1}[d_i=d]y_i}
{\sum_{i=1}^{N} \mathbf{1}[d_i=d]}
```

这能避免总分掩盖“数学升了、代码掉了”或“简单题升了、困难题掉了”的情况。

---

### 三、过程正确性评估

Reasoning model 的核心是过程。

过程评估关注：

1. 每一步是否成立。
2. 是否有跳步。
3. 是否使用题目没有给出的假设。
4. 中间变量是否一致。
5. 最终答案是否由前面步骤推出。

方法包括：

1. 人工标注步骤。
2. PRM 打分。
3. LLM-as-a-judge 判断步骤。
4. 符号工具检查。
5. 单元测试或执行验证。
6. 对推理过程做 contradiction check。

过程评估很贵。

但它能发现 final answer accuracy 看不到的问题。

例如：

```text
答案对，但推理过程错。
```

这种模型在新题上可能不可靠。

如果第 `i` 个样本有 `S_i` 个可标注步骤，第 `s` 步标签为 `z_{i,s}\in\{0,1\}`，过程正确率可以写成：

```math
A_{\mathrm{step}}
=
\frac{\sum_{i=1}^{N}\sum_{s=1}^{S_i} z_{i,s}}
{\sum_{i=1}^{N} S_i}
```

还要专门统计“答案对但过程错”的比例：

```math
r_{\mathrm{lucky}}
=
\frac{
\sum_{i=1}^{N}
\mathbf{1}[y_i=1]
\mathbf{1}
\left[
\min_{1\le s\le S_i} z_{i,s}=0
\right]
}
{\sum_{i=1}^{N}\mathbf{1}[y_i=1]+\epsilon}
```

`r_lucky` 高，说明 final answer accuracy 可能虚高，模型可能靠猜测、模板记忆或错误推理碰巧答对。

---

### 四、数学推理评估

数学评估常见指标：

1. final answer accuracy。
2. step correctness。
3. solve rate by difficulty。
4. self-consistency gain。
5. verifier rerank gain。
6. symbolic check pass rate。
7. contamination rate。
8. error type distribution。

常见任务包括：

1. 小学应用题。
2. 代数。
3. 几何。
4. 数论。
5. 概率组合。
6. 竞赛题。

数学评估要特别注意：

```text
答案格式归一化。
```

例如 `1/2`、`0.5`、`50%` 可能等价。

如果答案抽取和归一化做不好，评估会失真。

数学评估还要把 difficulty bucket 分开。设难度为 `h_i`，难度桶 `b` 上的准确率为：

```math
A_b
=
\frac{\sum_i \mathbf{1}[h_i=b]y_i}
{\sum_i \mathbf{1}[h_i=b]}
```

如果只报告平均准确率，模型可能在简单题上提升、竞赛难题上不变，面试时就会被追问“这个提升到底来自哪里”。

---

### 五、代码推理评估

代码评估有更硬的执行信号。

常见指标：

1. pass@1。
2. pass@k。
3. compile rate。
4. test pass rate。
5. repair success rate。
6. average attempts。
7. runtime。
8. memory usage。
9. security violation rate。
10. diff minimality。

代码评估要注意：

1. 测试覆盖不足。
2. hidden tests。
3. hardcode 测试。
4. 依赖环境差异。
5. 代码安全。
6. repo-level side effects。

代码模型通过 benchmark 不代表能改真实仓库。

真实仓库任务还要评估：

1. 是否理解上下文。
2. 是否改对文件。
3. 是否新增测试。
4. 是否破坏现有行为。
5. 是否符合项目风格。

公开测试和隐藏测试要分开报告。设 `q_i^{\mathrm{pub}}` 是公开测试通过率，`q_i^{\mathrm{hid}}` 是隐藏测试通过率，则隐藏泛化差距可以写成：

```math
g_{\mathrm{hid}}
=
\frac{1}{N}
\sum_{i=1}^{N}
(q_i^{\mathrm{pub}}-q_i^{\mathrm{hid}})
```

`g_hid` 越大，越说明 public tests 指标不可信，模型可能只是记住样例或 hardcode 公开测试。

---

### 六、逻辑和多跳问答评估

逻辑和多跳问答通常不像数学那样容易自动验证。

常见指标：

1. answer correctness。
2. evidence recall。
3. reasoning path correctness。
4. supporting facts accuracy。
5. contradiction rate。
6. robustness to distractors。

例如多跳问答需要模型：

```text
找到证据 A -> 根据 A 找证据 B -> 合成答案
```

评估时不能只看最终答案。

还要看：

1. 是否找到了正确中间证据。
2. 是否忽略干扰证据。
3. 是否正确连接多跳关系。
4. 是否引用支持答案的来源。

这和 RAG 的 attribution 评估有重叠。

如果一个多跳问题需要 `K_i` 个证据点，模型找到第 `k` 个证据的标签为 `e_{i,k}`，证据覆盖率可以写成：

```math
A_{\mathrm{evi}}
=
\frac{\sum_i\sum_{k=1}^{K_i} e_{i,k}}
{\sum_i K_i}
```

最终答案对但 `A_evi` 低，说明模型可能没有真正沿证据链推理；这类问题在开放 QA 和 RAG 场景里很常见。

---

### 七、规划和 Agent 推理评估

规划任务评估更复杂。

例如：

```text
帮我规划一个三天学习计划，并根据每天反馈调整。
```

评估指标包括：

1. task success rate。
2. plan validity。
3. constraint satisfaction。
4. tool call accuracy。
5. recovery rate。
6. safety violation rate。
7. user confirmation correctness。
8. trace quality。
9. cost and latency。

Agent 推理不能只看最终回答。

因为 Agent 可能最终回答看起来对，但中间越权调用、泄露数据或浪费大量工具调用。

所以要评估完整 trajectory。

设 Agent 第 `i` 条轨迹有 `T_i` 个工具调用或动作，动作合法标签为 `p_{i,t}`，最终任务成功标签为 `u_i`。轨迹合法率可以写成：

```math
A_{\mathrm{traj}}
=
\frac{\sum_i\sum_{t=1}^{T_i} p_{i,t}}
{\sum_i T_i}
```

任务成功且没有违规动作的比例可以写成：

```math
A_{\mathrm{safe}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[u_i=1]
\mathbf{1}
\left[
\min_{1\le t\le T_i}p_{i,t}=1
\right]
```

Agent 评估里 `A_safe` 往往比单纯 task success 更重要：一个越权拿到答案的轨迹不能算可靠成功。

---

### 八、Test-Time Compute 曲线

Reasoning model 的一个关键评估是：

```text
随着推理预算增加，性能是否提升？
```

可以比较：

1. direct answer。
2. single CoT。
3. self-consistency N=5。
4. self-consistency N=20。
5. verifier rerank。
6. search。

记录：

1. accuracy。
2. solve rate。
3. token cost。
4. latency。
5. tool calls。
6. cost per solved task。

理想模型应该在难题上能从更多 compute 中获益。

如果增加 20 倍 compute 只提升 1%，说明收益很低。

如果更多 compute 反而降低准确率，说明推理过程可能不稳定。

多候选评估要区分 `oracle@N` 和真实选择准确率。设第 `i` 题有 `N` 个候选，第 `j` 个候选正确标签为 `u_{i,j}`，verifier 或 reranker 选择下标为 `s_i`。候选集合里“至少有一个正确解”的上界是：

```math
O_N
=
\frac{1}{M}
\sum_{i=1}^{M}
\mathbf{1}
\left[
\max_{1\le j\le N} u_{i,j}=1
\right]
```

真实选择准确率是：

```math
A_{\mathrm{sel}}
=
\frac{1}{M}
\sum_{i=1}^{M}
u_{i,s_i}
```

如果 `O_N` 高但 `A_sel` 低，说明生成器已经产生正确解，但 verifier、投票、rerank 或 search 策略没选出来。

把推理预算记为 `B`，在预算 `B` 下的准确率、平均 token 成本和平均延迟分别记为 `A(B)`、`C(B)`、`L(B)`。从预算 `B_1` 增加到 `B_2` 的边际收益可以写成：

```math
G_{\mathrm{budget}}
=
\frac{A(B_2)-A(B_1)}
{C(B_2)-C(B_1)+\epsilon}
```

每解决一道题的成本可以写成：

```math
C_{\mathrm{solve}}
=
\frac{\sum_{i=1}^{N} C_i}
{\sum_{i=1}^{N} y_i+\epsilon}
```

Reasoning model 如果只靠极高预算提升榜单分数，`C_solve` 会很高，线上不一定可用。

---

### 九、鲁棒性评估

Reasoning model 可能对题目表述很敏感。

鲁棒性评估包括：

1. 改写题干。
2. 打乱无关信息顺序。
3. 加入 distractors。
4. 替换数字。
5. 改变单位。
6. 改变语言。
7. 加入无关上下文。
8. 测试同构题。

如果模型只在原题上对，改写后错，说明泛化不足。

数学和代码任务尤其要做变体测试。

因为很多模型可能记住模板。

设第 `i` 个原题有 `K_i` 个语义等价变体，变体 `k` 的正确指示变量为 `v_{i,k}`。变体鲁棒准确率可以写成：

```math
A_{\mathrm{var}}
=
\frac{\sum_i\sum_{k=1}^{K_i} v_{i,k}}
{\sum_i K_i}
```

原题到变体的性能下降可以写成：

```math
D_{\mathrm{var}}
=
A_{\mathrm{orig}}
-
A_{\mathrm{var}}
```

如果 `A_orig` 高但 `A_var` 低，模型更像记住原题或模板，而不是稳健推理。

---

### 十、数据污染检测

Reasoning benchmark 常被污染。

污染来源包括：

1. 预训练语料包含 benchmark。
2. SFT 数据包含题目或题解。
3. 合成数据从 benchmark 改写。
4. 评估 prompt 泄露答案。
5. 公开 leaderboard 被反复优化。

检测方法：

1. Exact match。
2. Fuzzy match。
3. Embedding similarity。
4. 题干、答案、解法分别查重。
5. 时间切分。
6. 新构造私有测试集。
7. 观察异常高分样本。

污染检测不是可选项。

没有污染检测的 reasoning 分数可信度很低。

污染风险率可以写成：

```math
\rho_{\mathrm{contam}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
\left[
\max_j
\operatorname{sim}(x_i,u_j)
\ge
\tau
\right]
```

`u_j` 是训练语料、SFT 数据、合成数据或公开题解中的候选文本。`sim` 不能只查题干，还要查答案、解题步骤、单元测试、函数签名、变量结构和改写模板。这个指标更准确地说是 contamination risk，不等于已经证明训练集中一定含有该题。

---

### 十一、LLM-as-a-Judge 的使用边界

LLM judge 可以评估开放推理。

但它有风险：

1. 偏好流畅答案。
2. 被错误 CoT 说服。
3. 漏看长推理中的错误。
4. 对数学和代码细节不可靠。
5. 不稳定。
6. 和被评模型同源时有偏。

适合 LLM judge 的场景：

1. 开放答案初筛。
2. 解释质量评估。
3. 人工评估辅助。
4. 错误类型归类。

不适合完全依赖 LLM judge 的场景：

1. 数学最终判分。
2. 代码正确性。
3. 高风险安全决策。
4. 精确事实验证。

最好组合：

```text
工具验证 + verifier + LLM judge + 人工抽查
```

如果有人工标注集合，可以估计 judge agreement：

```math
A_{\mathrm{judge}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[j_i=h_i]
```

其中 `j_i` 是 judge 标签，`h_i` 是人工标签。还要按任务类型拆开看：judge 在开放解释上可能可用，在数学、代码、安全和高风险决策上通常不能替代硬验证。

---

### 十二、错误分析

Reasoning 评估必须做 error analysis。

错误类型可以包括：

1. 读题错误。
2. 条件遗漏。
3. 错误建模。
4. 中间计算错误。
5. 逻辑跳步。
6. 工具调用错误。
7. Verifier 误判。
8. 答案抽取错误。
9. 格式错误。
10. 安全或权限错误。

错误分析的价值是指导下一步优化。

如果主要错在计算，可以接工具。

如果主要错在读题，需要数据和 prompt。

如果主要错在搜索剪枝，需要改 verifier。

如果主要错在污染，需要重建评估集。

---

### 十三、评估体系设计

一个可信 reasoning 评估体系可以分层：

#### 1. Unit eval

单题、单函数、单步骤评估。

#### 2. Benchmark eval

数学、代码、逻辑、多跳问答等标准集。

#### 3. Robustness eval

改写、扰动、distractor、同构题。

#### 4. Test-time scaling eval

不同预算下的性能曲线。

#### 5. Safety eval

越权、危险工具、敏感信息、错误自信。

#### 6. Human eval

复杂开放题和真实任务人工评估。

#### 7. Regression eval

历史失败样本回归测试。

#### 8. Production eval

真实用户任务、延迟、成本、满意度和事故率。

这种体系比单一 benchmark 分数可靠得多。

最后可以把多维指标变成 release gate，而不是只看榜单总分。一个简化的综合分可以写成：

```math
S_{\mathrm{eval}}
=
w_a A_{\mathrm{final}}
+
w_p A_{\mathrm{step}}
+
w_r A_{\mathrm{var}}
-
w_c C_{\mathrm{solve}}
-
w_l L
-
w_s r_{\mathrm{safe}}
-
w_m \rho_{\mathrm{contam}}
```

但生产发布不能只用 `S_eval` 排序。安全违规、污染率、关键任务回退和隐藏测试差距应该是 hard gate：只要超过阈值，即使综合分更高，也不能发布。

---

### 十四、最小代码：Reasoning Evaluation Harness

下面的 demo 不调用模型，而是模拟一个 reasoning evaluation harness。它展示：

1. 污染样本要从正式分数中隔离。
2. final answer accuracy、step accuracy、鲁棒性、成本和安全要同时报告。
3. 直接回答可能便宜但准确率和鲁棒性低。
4. 高预算 reasoning 可能准确率高，但如果 trajectory 有安全违规，仍然不能发布。
5. 领域分桶能暴露数学、代码、逻辑和 Agent 任务上的差异。

```python
from collections import Counter

CASES = [
    {
        "id": "math_linear",
        "domain": "math",
        "gold": "4",
        "variant_gold": ["4", "4"],
        "eval_sim": 0.18,
        "outputs": {
            "direct": {"answer": "5", "steps": [1, 0], "variant_answers": ["5", "4"], "tokens": 80, "latency": 0.7, "safety": 0, "error": "calculation"},
            "reasoner": {"answer": "4", "steps": [1, 1, 1], "variant_answers": ["4", "4"], "tokens": 260, "latency": 2.1, "safety": 0, "error": None},
        },
    },
    {
        "id": "math_lucky",
        "domain": "math",
        "gold": "6",
        "variant_gold": ["6", "6"],
        "eval_sim": 0.22,
        "outputs": {
            "direct": {"answer": "6", "steps": [1, 0, 1], "variant_answers": ["6", "3"], "tokens": 110, "latency": 0.9, "safety": 0, "error": "bad_process"},
            "reasoner": {"answer": "6", "steps": [1, 1, 1], "variant_answers": ["6", "6"], "tokens": 300, "latency": 2.4, "safety": 0, "error": None},
        },
    },
    {
        "id": "code_even",
        "domain": "code",
        "gold": "pass",
        "variant_gold": [],
        "eval_sim": 0.30,
        "outputs": {
            "direct": {"answer": "pass", "steps": [1, 1], "public_ok": 1, "hidden_ok": 0, "variant_answers": [], "tokens": 140, "latency": 1.3, "safety": 0, "error": "hidden_fail"},
            "reasoner": {"answer": "pass", "steps": [1, 1, 1], "public_ok": 1, "hidden_ok": 1, "variant_answers": [], "tokens": 420, "latency": 3.5, "safety": 0, "error": None},
        },
    },
    {
        "id": "logic_distractor",
        "domain": "logic",
        "gold": "B",
        "variant_gold": ["B", "B"],
        "eval_sim": 0.26,
        "outputs": {
            "direct": {"answer": "B", "steps": [1, 1], "variant_answers": ["A", "B"], "tokens": 90, "latency": 0.8, "safety": 0, "error": None},
            "reasoner": {"answer": "B", "steps": [1, 1, 1], "variant_answers": ["B", "B"], "tokens": 240, "latency": 2.0, "safety": 0, "error": None},
        },
    },
    {
        "id": "agent_refund",
        "domain": "agent",
        "gold": "success",
        "variant_gold": ["success"],
        "eval_sim": 0.35,
        "outputs": {
            "direct": {"answer": "fail", "steps": [1, 0], "variant_answers": ["fail"], "tokens": 130, "latency": 1.1, "safety": 0, "error": "constraint"},
            "reasoner": {"answer": "success", "steps": [1, 1, 1], "variant_answers": ["success"], "tokens": 520, "latency": 4.2, "safety": 1, "error": "unsafe_tool"},
        },
    },
    {
        "id": "contam_holdout",
        "domain": "math",
        "gold": "18",
        "variant_gold": ["18"],
        "eval_sim": 0.94,
        "outputs": {
            "direct": {"answer": "18", "steps": [1, 1], "variant_answers": ["18"], "tokens": 70, "latency": 0.6, "safety": 0, "error": None},
            "reasoner": {"answer": "18", "steps": [1, 1, 1], "variant_answers": ["18"], "tokens": 180, "latency": 1.7, "safety": 0, "error": None},
        },
    },
]


def normalize(value):
    return str(value).strip().lower()


def is_contaminated(case):
    return case["eval_sim"] >= 0.90


def is_correct(case, output):
    if case["domain"] == "code":
        return output.get("hidden_ok", 0) == 1
    return normalize(output["answer"]) == normalize(case["gold"])


def variant_scores(case, output):
    return [
        int(normalize(answer) == normalize(gold))
        for answer, gold in zip(output["variant_answers"], case["variant_gold"])
    ]


def evaluate(model_name):
    scoreable = [case for case in CASES if not is_contaminated(case)]
    correct = []
    step_labels = []
    variant_labels = []
    domain_total = Counter()
    domain_correct = Counter()
    errors = Counter()
    safety_violations = 0
    tokens = 0
    latency = 0.0
    lucky_right = 0
    correct_total = 0

    for case in scoreable:
        output = case["outputs"][model_name]
        ok = is_correct(case, output)
        correct.append(int(ok))
        domain_total[case["domain"]] += 1
        domain_correct[case["domain"]] += int(ok)
        step_labels.extend(output["steps"])
        variant_labels.extend(variant_scores(case, output))
        safety_violations += output["safety"]
        tokens += output["tokens"]
        latency += output["latency"]
        if output["error"]:
            errors[output["error"]] += 1
        if ok:
            correct_total += 1
            lucky_right += int(min(output["steps"]) == 0)

    accuracy = sum(correct) / len(correct)
    step_accuracy = sum(step_labels) / len(step_labels)
    robustness = sum(variant_labels) / len(variant_labels)
    safety_rate = safety_violations / len(scoreable)
    cost_per_solved = tokens / (sum(correct) + 1e-9)
    domain_accuracy = {
        domain: round(domain_correct[domain] / domain_total[domain], 3)
        for domain in sorted(domain_total)
    }
    release_gate = (
        accuracy >= 0.75
        and step_accuracy >= 0.85
        and robustness >= 0.75
        and safety_rate == 0.0
        and cost_per_solved <= 450
    )
    return {
        "scoreable": len(scoreable),
        "quarantined": len(CASES) - len(scoreable),
        "final_accuracy": round(accuracy, 3),
        "step_accuracy": round(step_accuracy, 3),
        "lucky_right_rate": round(lucky_right / (correct_total + 1e-9), 3),
        "robustness_accuracy": round(robustness, 3),
        "safety_violation_rate": round(safety_rate, 3),
        "avg_tokens": round(tokens / len(scoreable), 1),
        "avg_latency": round(latency / len(scoreable), 2),
        "cost_per_solved": round(cost_per_solved, 1),
        "domain_accuracy": domain_accuracy,
        "error_buckets": dict(errors),
        "release_gate": release_gate,
    }


results = {model: evaluate(model) for model in ["direct", "reasoner"]}
contamination_rate = sum(is_contaminated(case) for case in CASES) / len(CASES)
budget_gain = (
    results["reasoner"]["final_accuracy"] - results["direct"]["final_accuracy"]
) / (
    results["reasoner"]["avg_tokens"] - results["direct"]["avg_tokens"]
)

print("contamination_rate=", round(contamination_rate, 3))
print("direct=", results["direct"])
print("reasoner=", results["reasoner"])
print("accuracy_gain_per_extra_token=", round(budget_gain, 5))
```

典型输出：

```text
contamination_rate= 0.167
direct= {'scoreable': 5, 'quarantined': 1, 'final_accuracy': 0.4, 'step_accuracy': 0.727, 'lucky_right_rate': 0.5, 'robustness_accuracy': 0.429, 'safety_violation_rate': 0.0, 'avg_tokens': 110.0, 'avg_latency': 0.96, 'cost_per_solved': 275.0, 'domain_accuracy': {'agent': 0.0, 'code': 0.0, 'logic': 1.0, 'math': 0.5}, 'error_buckets': {'calculation': 1, 'bad_process': 1, 'hidden_fail': 1, 'constraint': 1}, 'release_gate': False}
reasoner= {'scoreable': 5, 'quarantined': 1, 'final_accuracy': 1.0, 'step_accuracy': 1.0, 'lucky_right_rate': 0.0, 'robustness_accuracy': 1.0, 'safety_violation_rate': 0.2, 'avg_tokens': 348.0, 'avg_latency': 2.84, 'cost_per_solved': 348.0, 'domain_accuracy': {'agent': 1.0, 'code': 1.0, 'logic': 1.0, 'math': 1.0}, 'error_buckets': {'unsafe_tool': 1}, 'release_gate': False}
accuracy_gain_per_extra_token= 0.00252
```

这组结果对应几个判断：

1. `contam_holdout` 被隔离，所以 1 个污染样本不会进入正式分数。
2. `direct` 的 `final_accuracy=0.4`，而且 `lucky_right_rate=0.5`，说明一半正确答案的过程并不可靠。
3. `reasoner` 的准确率、过程和鲁棒性都更好，但 `safety_violation_rate=0.2`，因此 release gate 仍然失败。
4. `accuracy_gain_per_extra_token` 把 test-time compute 收益和额外 token 成本绑在一起，避免只看高预算准确率。
5. `domain_accuracy` 能看出 direct 在代码和 Agent 场景失败，而不是只看到一个平均分。

---

### 十五、真实项目中的坑

#### 1. 只看总分

平均分掩盖难题和高风险场景退化。

#### 2. 忽略成本

模型靠 50 倍采样拿高分，线上不可用。

#### 3. 不做污染检测

分数虚高。

#### 4. LLM judge 直接当真值

Judge 本身会错。

#### 5. 不评估过程

答案对但推理过程不可靠。

#### 6. 不做错误归因

只知道错了，不知道该改数据、模型、verifier 还是工具。

#### 7. Benchmark 和业务无关

数学题高分不代表企业 Agent 任务成功。

---

### 十六、面试问答

#### 问题 1：Reasoning model 评估和普通 chat model 评估有什么区别？

可以这样回答：

```text
普通 chat model 更多看有用性、流畅度和指令遵循；reasoning model 还要看多步推理正确性、过程可靠性、可验证结果、test-time compute 曲线、鲁棒性和污染风险。
```

#### 问题 2：为什么 final answer accuracy 不够？

可以这样回答：

```text
因为最终答案可能碰巧正确，过程可能错误；也可能答案错但前面步骤大部分正确。只看 final answer 无法定位错误，也无法评估过程可靠性和泛化能力。
```

#### 问题 3：如何评估 test-time compute scaling？

可以这样回答：

```text
比较 direct answer、CoT、多采样、verifier、search 等不同预算下的准确率、solve rate、token、延迟、工具调用和 cost per solved task，画成本收益曲线。
```

#### 问题 4：Reasoning benchmark 为什么容易污染？

可以这样回答：

```text
因为数学、代码和逻辑题常在网上公开，预训练、SFT 或合成数据都可能包含原题、题解或改写题。需要 exact/fuzzy/embedding 去重和私有 holdout。
```

#### 问题 5：LLM-as-a-judge 能不能评估 reasoning？

可以这样回答：

```text
可以作为辅助，但不能完全依赖。LLM judge 可能偏好流畅解释、被错误推理说服，也可能漏看细节。数学和代码最好用工具、测试或符号验证，开放题再结合人工和 judge。
```

#### 问题 6：如何设计一个 reasoning model 评估体系？

可以这样回答：

```text
我会分层设计：标准 benchmark、过程正确性、鲁棒性、污染检测、test-time compute 曲线、安全评估、错误分析、回归测试和真实任务评估，并同时报告准确率、成本、延迟和错误类型。
```

---

### 十七、常见误区

1. 误区：Reasoning 模型只要数学分高就行。
   纠正：还要看代码、逻辑、规划、真实任务和安全。

2. 误区：答案对就说明推理对。
   纠正：答案可能碰巧对，过程仍可能错。

3. 误区：更长 CoT 就代表更强 reasoning。
   纠正：长推理可能只是冗余或错误累积。

4. 误区：Benchmark 分数可直接代表生产效果。
   纠正：生产还要看延迟、成本、工具、用户场景和风险。

5. 误区：LLM judge 足够评估 reasoning。
   纠正：要结合工具验证、人工抽查和标准答案。

6. 误区：只需要一次离线评估。
   纠正：需要持续回归测试和线上监控。

---

### 十八、小练习

1. 为一个数学 reasoning benchmark 设计评估指标。
2. 为一个代码 benchmark 设计 pass@k、执行反馈和安全指标。
3. 构造一个答案正确但推理过程错误的样本，并设计评估方法。
4. 设计一个 test-time compute 曲线实验。
5. 设计一个 reasoning benchmark 污染检测流程。
6. 比较 LLM judge、PRM、程序验证和人工评估。
7. 为多跳问答设计 evidence 和 attribution 评估。
8. 为 Agent planning 设计 trajectory-level 评估。
9. 对 20 个 reasoning 错误样本做错误类型归因。
10. 用 3 分钟回答：“如何可信评估一个 reasoning model？”

### 本讲总结

本讲最重要的结论：

1. Reasoning model 评估要看最终答案、推理过程、鲁棒性、成本和安全。
2. Final answer accuracy 必要但不充分。
3. 数学、代码、逻辑、多跳问答和规划任务需要不同评估方法。
4. Test-time compute 曲线是 reasoning model 的关键评估维度。
5. 数据污染会让 reasoning benchmark 分数严重虚高。
6. LLM-as-a-judge 可以辅助评估，但不能替代工具验证和人工校准。
7. 错误分析比单一分数更能指导模型、数据、verifier 和工具改进。
8. 面试中要把 reasoning 评估讲成多层体系，而不是只报一个 benchmark 分数。

## 第 95 讲：从 Chat Model 到 Reasoning Model

### 本讲目标

学完本讲，你应该能回答六个问题：

1. Chat model 和 reasoning model 的核心区别是什么。
2. 为什么单纯指令微调不足以得到强 reasoning model。
3. Reasoning model 在数据、训练、推理和评估上有哪些变化。
4. CoT、verifier、search、test-time compute 如何共同推动 reasoning model。
5. 从产品和系统角度，reasoning model 带来哪些新设计问题。
6. 面试中如何完整回答“如何从 chat model 走向 reasoning model”。

第八部分从 CoT 讲起，依次讲了 self-consistency、verifier、search、test-time compute、数学训练、代码执行反馈、合成推理数据和 reasoning 评估。

这一讲做一个总收束：

```text
Chat model 如何演化为 reasoning model？
```

Chat model 的核心能力是对话、指令遵循和通用问答。

Reasoning model 的核心能力是面对复杂任务时进行多步思考、验证、搜索和纠错。

两者不是完全割裂。

Reasoning model 仍然需要聊天能力。

但它在训练目标、数据结构、推理方式和评估体系上都发生了变化。

#### 来龙去脉：从会聊天到会解题

从 base model 到 chat model，关键变化是对齐用户意图。InstructGPT 这条线说明：仅仅把语言模型做大，并不会自然保证它更会遵循用户意图；需要用示范数据、偏好数据和人类反馈把模型从“续写器”拉向“助手”。

但 chat alignment 主要解决的是“如何按用户意图回答”。复杂数学、代码、规划和多跳问答暴露了另一个问题：模型可能回答得自然、礼貌、格式正确，却在中间步骤上犯错。Chain-of-Thought 把中间推理步骤显式化，self-consistency 进一步把单条贪心路径变成多路径采样和答案聚合，verifier / PRM 则把“生成答案”拆成“生成候选 + 评价过程/结果”。

所以从 chat model 到 reasoning model，不是简单多加一句“请逐步思考”，而是训练和系统形态都发生迁移：

```text
instruction following
-> step-by-step solving
-> multi-sample and rerank
-> process/outcome verification
-> search/tool/test-time compute
-> budget-aware product system
```

面试中要把这条线讲成能力边界的演进：chat model 解决沟通和对齐，reasoning model 在此基础上加入可验证求解、过程监督和推理时计算。

---

### 一、Chat Model 的典型目标

Chat model 主要解决：

1. 理解用户指令。
2. 给出有帮助的回答。
3. 保持对话自然。
4. 遵守安全规范。
5. 适配多种通用场景。

训练上常见流程是：

```text
预训练 -> 指令微调 -> 偏好对齐 -> 安全对齐
```

评估上常看：

1. helpfulness。
2. harmlessness。
3. instruction following。
4. fluency。
5. human preference。

Chat model 很适合：

1. 问答。
2. 总结。
3. 写作。
4. 翻译。
5. 头脑风暴。
6. 普通助手任务。

但遇到复杂数学、代码、规划和多步工具任务时，普通 chat model 容易出现不稳定推理。

从训练目标看，chat SFT 通常只对助手回答 token 做最大似然：

```math
L_{\mathrm{chat}}(\theta)
=
-\frac{1}{|A|}
\sum_{t \in A}
\log p_\theta(y_t \mid x, y_{1:t-1})
```

其中 `A` 是助手回答 token 位置集合，`x` 是对话上下文，`y_t` 是第 `t` 个目标 token。这个目标能让模型更像助手，但它本身不区分“答案正确但过程错”和“答案正确且过程稳”。

如果用集合 `E` 表示普通指令评估集，chat model 常见分数可以抽象成：

```math
A_{\mathrm{inst}}
=
\frac{1}{|E|}
\sum_{(x_i,y_i)\in E}
I[f_\theta(x_i)=y_i]
```

这里 `A_inst` 更偏向指令遵循、格式、帮助性和最终偏好，不足以刻画难题上的求解过程。

---

### 二、Reasoning Model 的典型目标

Reasoning model 更关注：

1. 多步问题分解。
2. 中间状态维护。
3. 逻辑一致性。
4. 错误检查。
5. 试错和回退。
6. 使用 verifier 和工具。
7. 随 test-time compute 增加而提升。

它适合：

1. 数学推理。
2. 代码推理。
3. 科学问题。
4. 复杂规划。
5. 多跳问答。
6. Agent 任务。
7. 高价值决策辅助。

一句话：

```text
Chat model 更像会沟通的通用助手，reasoning model 更像会解题、会验证、会试错的求解器。
```

更形式化地看，reasoning model 需要同时优化最终答案、过程质量和推理预算。设 `R_i` 是样本 `i` 的步骤集合，`z_{ij}` 表示第 `j` 个步骤是否正确：

```math
A_{\mathrm{final}}
=
\frac{1}{n}
\sum_{i=1}^{n}
I[\hat a_i=a_i]
```

```math
A_{\mathrm{step}}
=
\frac{1}{\sum_i |R_i|}
\sum_{i=1}^{n}
\sum_{j=1}^{|R_i|}
z_{ij}
```

如果一种方法的 token 成本是 `C`，延迟是 `T`，安全风险是 `R`，可以用一个简单效用函数描述 reasoning 的取舍：

```math
U
=
A_{\mathrm{final}}
+ \alpha A_{\mathrm{step}}
- \lambda C
- \tau T
- \rho R
```

这说明 reasoning model 不是“无限多想”，而是在正确率、过程可靠性、成本、延迟和安全之间做工程优化。

---

### 三、为什么指令微调不够

指令微调可以教模型“怎么回答”。

但 reasoning 需要模型“怎么思考”。

普通 SFT 数据可能包含大量问答，但缺少：

1. 高质量多步推理过程。
2. 错误步骤标注。
3. 失败后修正轨迹。
4. verifier 反馈。
5. 搜索和多候选选择。
6. 难题上的过程监督。

所以只做指令微调，模型可能学会：

```text
看起来像在推理。
```

但未必学会：

```text
可靠地推理。
```

这就是 CoT 忠实性、过程监督和 verifier 重要的原因。

可以把普通 SFT 的缺口写成一个目标错配：

```math
\max_\theta A_{\mathrm{inst}}(\theta)
\not\Rightarrow
\max_\theta A_{\mathrm{reason}}(\theta)
```

其中 `A_reason` 至少包含最终答案、步骤正确性、鲁棒性、工具轨迹合法性和预算收益。更合理的升级目标是：

```math
L_{\mathrm{upgrade}}
=
\lambda_c L_{\mathrm{chat}}
+ \lambda_r L_{\mathrm{reason}}
+ \lambda_p L_{\mathrm{process}}
+ \lambda_v L_{\mathrm{verify}}
+ \lambda_s L_{\mathrm{safe}}
```

这里 `L_chat` 保留助手能力，`L_reason` 学分步求解，`L_process` 学步骤好坏，`L_verify` 学选择和验证，`L_safe` 防止推理和工具调用带来副作用。

---

### 四、数据上的变化

从 chat model 到 reasoning model，数据形态发生变化。

Chat 数据常见是：

```text
用户问题 -> 助手回答
```

Reasoning 数据更像：

```text
问题 -> 分步推理 -> 中间检查 -> 最终答案
```

或者：

```text
问题 -> 多个候选解法 -> verifier 分数 -> 选择结果
```

或者：

```text
任务 -> 工具调用 -> observation -> 修正 -> 完成
```

需要的数据包括：

1. CoT 解法。
2. 过程标注。
3. 正负样本。
4. hard negatives。
5. 工具执行轨迹。
6. 代码测试反馈。
7. 数学 verifier 数据。
8. 合成推理数据。
9. 失败修复轨迹。

Reasoning 数据更贵，也更需要验证。

一个实用的数据配比可以写成：

```math
q_{\mathrm{reason}}
=
\frac{N_{\mathrm{reason}}}
{N_{\mathrm{chat}}+N_{\mathrm{reason}}+N_{\mathrm{safe}}}
```

但不能只看样本数，因为 reasoning 样本通常更长、更贵，也更容易有过程噪声。训练时更常按 token 和权重混合：

```math
L_{\mathrm{mix}}
=
\sum_{k\in K}
\omega_k
\frac{1}{N_k}
\sum_{(x,y)\in D_k}
L_k(x,y)
```

其中 `K` 可以包含 `chat`、`reason`、`process`、`tool`、`safe` 等数据桶，`omega_k` 是每类数据的训练权重。

Reasoning 数据的关键不是“有长过程”，而是过程可验证。可以给每条样本一个质量分：

```math
Q_i
=
w_a I[\hat a_i=a_i]
+ w_p A_{\mathrm{step},i}
+ w_v V_i
+ w_d D_i
- w_l L_i
- w_s S_i
```

其中 `V_i` 是 verifier 或工具验证分，`D_i` 是多样性/覆盖贡献，`L_i` 是长度惩罚，`S_i` 是安全或污染风险。

---

### 五、训练上的变化

Reasoning model 训练不只是普通 SFT。

常见训练组件包括：

#### 1. CoT SFT

让模型学习分步解题格式和基本推理。

#### 2. Rejection sampling SFT

采样多条解法，用 verifier 或工具过滤，保留高质量样本训练。

#### 3. ORM / PRM

训练结果奖励模型或过程奖励模型。

#### 4. Preference optimization

让模型偏好更正确、更简洁、更稳定的解法。

#### 5. RL

用可验证 reward 直接优化推理成功率。

但风险更高，需要强 reward 和防 reward hacking。

#### 6. Distillation

从强推理系统蒸馏能力到更便宜模型。

这些训练方法的共同目标是：

```text
不只让模型生成答案，而是让模型生成更可靠的求解过程。
```

对于 CoT SFT，可以对 reasoning token 和最终答案 token 使用不同权重：

```math
L_{\mathrm{cot}}
=
-\frac{1}{Z}
\left(
\alpha_r
\sum_{t\in R}
\log p_\theta(y_t \mid x, y_{1:t-1})
+
\alpha_a
\sum_{t\in A}
\log p_\theta(y_t \mid x, y_{1:t-1})
\right)
```

其中 `R` 是中间推理 token，`A` 是最终答案 token，`Z` 是归一化项。若担心模型过度学习冗长过程，可以让 `alpha_a` 不低于 `alpha_r`，并在评估时硬看最终答案与步骤正确性。

PRM 的 step-level 目标可以写成二分类：

```math
L_{\mathrm{prm}}(\phi)
=
-\frac{1}{M}
\sum_{j=1}^{M}
\left[
z_j\log v_\phi(s_j)
+(1-z_j)\log(1-v_\phi(s_j))
\right]
```

其中 `s_j` 是某个中间步骤上下文，`z_j` 是步骤标签。推理时可以把 generator 候选和 verifier 选择结合：

```math
a^*
=
\arg\max_{a_j\in C(x)}
\left[
v_\phi(x,a_j)
-\eta C_j
-\mu R_j
\right]
```

这里 `C(x)` 是候选集合，`C_j` 是候选成本，`R_j` 是候选风险。这比单纯最大化语言模型概率更接近生产 reasoning system。

---

### 六、推理方式上的变化

Chat model 通常一次生成答案。

Reasoning model 常常采用更复杂的推理流程。

例如：

1. 先分析问题。
2. 生成中间步骤。
3. 采样多个候选。
4. 用 verifier 检查。
5. 必要时 search。
6. 调用工具。
7. 修正错误。
8. 输出最终答案。

推理流程可以是：

```text
direct answer
CoT
self-consistency
verifier rerank
search
tool execution
human review
```

并且可以根据任务难度动态选择。

这就是第 90 讲讲的 test-time compute scaling。

动态路由可以写成：

```math
m^*(x)
=
\arg\max_{m\in M}
\left[
\hat q_m(x)
-\lambda \hat c_m(x)
-\tau \hat t_m(x)
-\rho \hat r_m(x)
\right]
```

其中 `M` 是可选推理模式集合，例如 direct、CoT、self-consistency、search、tool、human review；`q` 是质量预测，`c` 是成本，`t` 是延迟，`r` 是风险。

预算也可以按难度和风险动态分配：

```math
B(x)
=
B_0
+B_d d(x)
+B_u u(x)
+B_r r(x)
```

其中 `d(x)` 是难度估计，`u(x)` 是不确定性估计，`r(x)` 是风险估计。简单请求走低预算，难题和高风险请求才升级到 verifier、search 或人工确认。

---

### 七、评估方式上的变化

Chat model 可以用人类偏好评估很多场景。

Reasoning model 需要更硬的评估。

包括：

1. final answer accuracy。
2. process correctness。
3. pass@k。
4. solve rate。
5. verifier gain。
6. test-time compute curve。
7. contamination check。
8. robustness。
9. tool trajectory quality。
10. cost per solved task。

Reasoning model 的分数如果不报告推理预算，意义不完整。

例如：

```text
模型 A accuracy 90%，但每题采样 100 次。
模型 B accuracy 85%，但单次回答。
```

这两个结果不能简单比较。

必须同时报告成本、延迟和预算。

因此 reasoning 评估最好报告预算归一化收益：

```math
G_{\mathrm{budget}}
=
\frac{
A_{\mathrm{reason}}-A_{\mathrm{chat}}
}{
C_{\mathrm{reason}}-C_{\mathrm{chat}}+\epsilon
}
```

还要报告能力回退：

```math
\Delta_{\mathrm{inst}}
=
A_{\mathrm{inst}}^{\mathrm{new}}
-
A_{\mathrm{inst}}^{\mathrm{base}}
```

如果 reasoning 能力提升但普通助手能力、安全或延迟严重退化，就不是合格升级。

---

### 八、系统架构上的变化

Reasoning model 不一定只是一个模型。

它经常是一个系统。

系统组件包括：

1. Generator。
2. Verifier。
3. Reward model。
4. Search controller。
5. Tool executor。
6. Memory/state manager。
7. Budget allocator。
8. Safety guardrail。
9. Trace logger。

可以理解为：

```text
Reasoning system = 模型生成 + 评价函数 + 搜索策略 + 工具反馈 + 预算控制 + 安全审计
```

这比单个 chat model 复杂得多。

但也更适合复杂任务。

系统层的关键是把一次请求变成状态转移过程：

```math
s_{t+1}
=
F(s_t,a_t,o_t)
```

其中 `s_t` 是当前状态，`a_t` 是模型或控制器动作，`o_t` 是工具、verifier 或用户反馈。最终成功不只取决于模型输出，还取决于轨迹是否合法：

```math
S_{\mathrm{safe}}
=
I[\hat a=a]
\cdot
I[V_{\mathrm{tool}}=1]
\cdot
I[R_{\mathrm{unsafe}}=0]
```

---

### 九、产品形态上的变化

Chat model 产品通常追求：

1. 快速响应。
2. 自然对话。
3. 用户体验流畅。
4. 覆盖广泛任务。

Reasoning model 产品还要设计：

1. 是否显示思考过程。
2. 用户是否愿意等待。
3. 任务是否需要高预算。
4. 什么时候拒答。
5. 什么时候请求确认。
6. 如何展示证据和结论。
7. 如何解释不确定性。

例如：

```text
快速模式：直接回答，低成本。
深度思考模式：多步推理和验证，高成本。
```

用户体验上要明确：

1. 为什么需要等待。
2. 结果可信度如何。
3. 是否用了工具。
4. 哪些地方不确定。

---

### 十、Hidden Reasoning 与可见解释

Reasoning model 常有内部 reasoning。

但内部 reasoning 不一定全部展示给用户。

原因包括：

1. 中间过程可能冗长。
2. 中间过程可能不稳定。
3. 可能包含错误假设。
4. 可能泄露安全策略。
5. 可能让用户过度相信错误推理。

生产系统常见做法是：

```text
内部使用长推理，外部展示简洁解释、关键步骤、证据和最终答案。
```

这不是不透明。

而是区分：

1. 内部计算轨迹。
2. 用户可读解释。
3. 可审计 trace。

面试中要避免说“完整 CoT 必须展示才可信”。

更合理的是提供可验证证据和必要解释。

---

### 十一、从 Chat 到 Reasoning 的实现路径

如果已有一个 chat model，要增强 reasoning，可以分阶段。

#### 阶段 1：数据增强

收集数学、代码、逻辑、规划等高质量 CoT 数据。

#### 阶段 2：CoT SFT

让模型学会基本分步推理。

#### 阶段 3：Verifier

训练或接入答案验证、过程验证、代码测试和工具检查。

#### 阶段 4：多候选与 rerank

使用 self-consistency 和 verifier reranking。

#### 阶段 5：Search

对高难任务引入 beam search、ToT 或 MCTS。

#### 阶段 6：合成数据闭环

用强推理系统生成数据，再过滤、训练和评估。

#### 阶段 7：动态预算和产品化

根据任务难度选择 direct、CoT、search、tool 或人工审核。

这条路径比“一次训练一个 reasoning model”更工程化。

发布时可以用 hard gate 避免“分数高但不可用”的 checkpoint：

```math
G_{\mathrm{release}}
=
I[A_{\mathrm{reason}}\ge a_{\min}]
I[A_{\mathrm{inst}}\ge b_{\min}]
I[R_{\mathrm{safe}}\le r_{\max}]
I[C_{\mathrm{p95}}\le c_{\max}]
I[\Delta_{\mathrm{base}}\ge -\epsilon]
```

这个 gate 的含义是：reasoning 分数要达标，普通指令能力不能明显退化，安全风险和 p95 成本不能越线，基础能力回退不能超过阈值。

---

### 十二、最小代码：Chat 到 Reasoning 的训练混合与动态路由

下面的 demo 不调用模型，而是模拟一个从 chat model 升级到 reasoning system 的最小决策过程。它展示：

1. 训练数据从 chat-only 变成 chat、reason、process、safety 的加权混合。
2. `chat_only` 便宜但难题、代码和 Agent 任务失败。
3. `cot_default` 提升多数任务，但如果没有安全路由，仍可能在工具任务上违规。
4. `reasoning_router` 根据难度和风险选择 direct、CoT、search/tool 或 safe review。
5. 发布 gate 需要同时看 solve rate、step accuracy、安全、成本和延迟。

```python
from collections import Counter

TRAINING_BATCH = [
    {"kind": "chat", "tokens": 240, "nll": 0.72},
    {"kind": "chat", "tokens": 180, "nll": 0.68},
    {"kind": "reason", "tokens": 260, "nll": 0.62},
    {"kind": "reason", "tokens": 320, "nll": 0.58},
    {"kind": "process", "tokens": 190, "nll": 0.64},
    {"kind": "safety", "tokens": 140, "nll": 0.55},
]

WEIGHTS = {"chat": 1.0, "reason": 1.35, "process": 1.25, "safety": 1.5}

TASKS = [
    {
        "id": "faq_refund",
        "domain": "chat",
        "difficulty": 0.10,
        "risk": 0.00,
        "outcomes": {
            "direct": {"ok": 1, "steps": [1], "safety": 0, "tokens": 80, "latency": 0.6},
            "cot": {"ok": 1, "steps": [1, 1], "safety": 0, "tokens": 150, "latency": 1.0},
            "search_tool": {"ok": 1, "steps": [1, 1], "safety": 0, "tokens": 300, "latency": 2.0},
            "safe_review": {"ok": 1, "steps": [1], "safety": 0, "tokens": 220, "latency": 1.5},
        },
    },
    {
        "id": "math_word",
        "domain": "math",
        "difficulty": 0.65,
        "risk": 0.00,
        "outcomes": {
            "direct": {"ok": 0, "steps": [1, 0], "safety": 0, "tokens": 100, "latency": 0.8},
            "cot": {"ok": 1, "steps": [1, 1, 1], "safety": 0, "tokens": 280, "latency": 2.2},
            "search_tool": {"ok": 1, "steps": [1, 1, 1], "safety": 0, "tokens": 450, "latency": 3.5},
            "safe_review": {"ok": 1, "steps": [1, 1], "safety": 0, "tokens": 360, "latency": 3.0},
        },
    },
    {
        "id": "logic_distractor",
        "domain": "logic",
        "difficulty": 0.55,
        "risk": 0.00,
        "outcomes": {
            "direct": {"ok": 0, "steps": [0], "safety": 0, "tokens": 90, "latency": 0.7},
            "cot": {"ok": 1, "steps": [1, 1], "safety": 0, "tokens": 230, "latency": 1.8},
            "search_tool": {"ok": 1, "steps": [1, 1, 1], "safety": 0, "tokens": 430, "latency": 3.2},
            "safe_review": {"ok": 1, "steps": [1, 1], "safety": 0, "tokens": 330, "latency": 2.7},
        },
    },
    {
        "id": "code_hidden",
        "domain": "code",
        "difficulty": 0.80,
        "risk": 0.10,
        "outcomes": {
            "direct": {"ok": 0, "steps": [1, 0], "safety": 0, "tokens": 130, "latency": 1.0},
            "cot": {"ok": 0, "steps": [1, 1, 0], "safety": 0, "tokens": 340, "latency": 2.8},
            "search_tool": {"ok": 1, "steps": [1, 1, 1], "safety": 0, "tokens": 620, "latency": 4.8},
            "safe_review": {"ok": 1, "steps": [1, 1], "safety": 0, "tokens": 520, "latency": 4.2},
        },
    },
    {
        "id": "agent_order_change",
        "domain": "agent",
        "difficulty": 0.75,
        "risk": 0.60,
        "outcomes": {
            "direct": {"ok": 0, "steps": [1, 0], "safety": 1, "tokens": 120, "latency": 1.0},
            "cot": {"ok": 1, "steps": [1, 1, 1], "safety": 1, "tokens": 360, "latency": 3.0},
            "search_tool": {"ok": 1, "steps": [1, 1, 1], "safety": 0, "tokens": 560, "latency": 4.5},
            "safe_review": {"ok": 1, "steps": [1, 1], "safety": 0, "tokens": 440, "latency": 4.0},
        },
    },
    {
        "id": "benign_summary",
        "domain": "chat",
        "difficulty": 0.20,
        "risk": 0.00,
        "outcomes": {
            "direct": {"ok": 1, "steps": [1], "safety": 0, "tokens": 70, "latency": 0.5},
            "cot": {"ok": 1, "steps": [1, 1], "safety": 0, "tokens": 160, "latency": 1.0},
            "search_tool": {"ok": 1, "steps": [1, 1], "safety": 0, "tokens": 310, "latency": 2.2},
            "safe_review": {"ok": 1, "steps": [1], "safety": 0, "tokens": 210, "latency": 1.6},
        },
    },
    {
        "id": "risky_tool",
        "domain": "agent",
        "difficulty": 0.50,
        "risk": 0.90,
        "outcomes": {
            "direct": {"ok": 0, "steps": [0], "safety": 1, "tokens": 110, "latency": 0.9},
            "cot": {"ok": 0, "steps": [1, 0], "safety": 1, "tokens": 260, "latency": 2.3},
            "search_tool": {"ok": 0, "steps": [1, 0], "safety": 1, "tokens": 500, "latency": 4.1},
            "safe_review": {"ok": 1, "steps": [1, 1], "safety": 0, "tokens": 300, "latency": 2.8},
        },
    },
]


def weighted_training_loss(records):
    mass_by_kind = Counter()
    weighted_nll = 0.0
    total_mass = 0.0
    for item in records:
        mass = item["tokens"] * WEIGHTS[item["kind"]]
        mass_by_kind[item["kind"]] += mass
        weighted_nll += mass * item["nll"]
        total_mass += mass
    mix = {
        kind: round(mass_by_kind[kind] / total_mass, 3)
        for kind in sorted(mass_by_kind)
    }
    return mix, round(weighted_nll / total_mass, 3)


def choose_mode(policy, task):
    if policy == "chat_only":
        return "direct"
    if policy == "cot_default":
        return "safe_review" if task["risk"] >= 0.80 else "cot"
    if task["risk"] >= 0.80:
        return "safe_review"
    if task["domain"] == "code" or task["difficulty"] >= 0.75:
        return "search_tool"
    if task["difficulty"] >= 0.35:
        return "cot"
    return "direct"


def evaluate(policy):
    mode_counts = Counter()
    domain_total = Counter()
    domain_ok = Counter()
    step_labels = []
    ok_total = 0
    safety_violations = 0
    tokens = 0
    latency = 0.0

    for task in TASKS:
        mode = choose_mode(policy, task)
        output = task["outcomes"][mode]
        mode_counts[mode] += 1
        domain_total[task["domain"]] += 1
        domain_ok[task["domain"]] += output["ok"]
        step_labels.extend(output["steps"])
        ok_total += output["ok"]
        safety_violations += output["safety"]
        tokens += output["tokens"]
        latency += output["latency"]

    n = len(TASKS)
    solve_rate = ok_total / n
    step_accuracy = sum(step_labels) / len(step_labels)
    safety_rate = safety_violations / n
    avg_tokens = tokens / n
    avg_latency = latency / n
    cost_per_solved = tokens / (ok_total + 1e-9)
    domain_accuracy = {
        domain: round(domain_ok[domain] / domain_total[domain], 3)
        for domain in sorted(domain_total)
    }
    release_gate = (
        solve_rate >= 0.85
        and step_accuracy >= 0.90
        and safety_rate == 0.0
        and avg_latency <= 3.00
        and cost_per_solved <= 350
    )
    return {
        "solve_rate": round(solve_rate, 3),
        "step_accuracy": round(step_accuracy, 3),
        "safety_violation_rate": round(safety_rate, 3),
        "avg_tokens": round(avg_tokens, 1),
        "avg_latency": round(avg_latency, 2),
        "cost_per_solved": round(cost_per_solved, 1),
        "mode_counts": dict(mode_counts),
        "domain_accuracy": domain_accuracy,
        "release_gate": release_gate,
    }


mix, train_loss = weighted_training_loss(TRAINING_BATCH)
results = {
    policy: evaluate(policy)
    for policy in ["chat_only", "cot_default", "reasoning_router"]
}
gain_per_1k_tokens = (
    results["reasoning_router"]["solve_rate"] - results["chat_only"]["solve_rate"]
) / (
    (results["reasoning_router"]["avg_tokens"] - results["chat_only"]["avg_tokens"]) / 1000
)

print("training_mix=", mix)
print("weighted_training_loss=", train_loss)
for name, result in results.items():
    print(f"{name}=", result)
print("reasoning_gain_per_extra_1k_tokens=", round(gain_per_1k_tokens, 2))
```

典型输出：

```text
training_mix= {'chat': 0.254, 'process': 0.144, 'reason': 0.474, 'safety': 0.127}
weighted_training_loss= 0.625
chat_only= {'solve_rate': 0.286, 'step_accuracy': 0.5, 'safety_violation_rate': 0.286, 'avg_tokens': 100.0, 'avg_latency': 0.79, 'cost_per_solved': 350.0, 'mode_counts': {'direct': 7}, 'domain_accuracy': {'agent': 0.0, 'chat': 1.0, 'code': 0.0, 'logic': 0.0, 'math': 0.0}, 'release_gate': False}
cot_default= {'solve_rate': 0.857, 'step_accuracy': 0.941, 'safety_violation_rate': 0.143, 'avg_tokens': 260.0, 'avg_latency': 2.09, 'cost_per_solved': 303.3, 'mode_counts': {'cot': 6, 'safe_review': 1}, 'domain_accuracy': {'agent': 1.0, 'chat': 1.0, 'code': 0.0, 'logic': 1.0, 'math': 1.0}, 'release_gate': False}
reasoning_router= {'solve_rate': 1.0, 'step_accuracy': 1.0, 'safety_violation_rate': 0.0, 'avg_tokens': 305.7, 'avg_latency': 2.46, 'cost_per_solved': 305.7, 'mode_counts': {'direct': 2, 'cot': 2, 'search_tool': 2, 'safe_review': 1}, 'domain_accuracy': {'agent': 1.0, 'chat': 1.0, 'code': 1.0, 'logic': 1.0, 'math': 1.0}, 'release_gate': True}
reasoning_gain_per_extra_1k_tokens= 3.47
```

这组结果对应几个判断：

1. `training_mix` 说明第二阶段训练不再是 chat-only，而是给 reasoning、process 和 safety 数据更高权重。
2. `chat_only` 成本最低，但在数学、代码、逻辑和 Agent 任务上明显失败。
3. `cot_default` 的 solve rate 明显提升，但 `agent_order_change` 仍有安全违规，所以不能发布。
4. `reasoning_router` 通过动态预算把简单请求留给 direct，把难题交给 CoT/search，把高风险任务交给 safe review。
5. `reasoning_gain_per_extra_1k_tokens` 把收益和额外 token 成本绑定，避免只看最高准确率。

---

### 十三、真实项目中的坑

#### 1. 把长回答当 reasoning

长不等于对。

#### 2. 只做 CoT SFT

没有 verifier 和评估，模型可能只是模仿推理格式。

#### 3. 只看 benchmark 分数

忽略污染、成本、延迟和真实任务。

#### 4. 没有预算控制

高推理预算导致线上不可用。

#### 5. Verifier 太弱

错误路径被高分选择。

#### 6. 忽略安全

Reasoning + tool use 可能造成真实副作用。

#### 7. 误把 reasoning model 当万能模型

简单任务不需要深度推理。

---

### 十四、面试问答

#### 问题 1：Chat model 和 reasoning model 的区别是什么？

可以这样回答：

```text
Chat model 更强调对话、指令遵循和通用帮助；reasoning model 更强调多步推理、验证、搜索、纠错和随 test-time compute 增加而提升的能力。Reasoning model 往往不只是一个模型，而是模型、verifier、工具和预算策略组成的系统。
```

#### 问题 2：为什么普通 SFT 不够训练 reasoning model？

可以这样回答：

```text
普通 SFT 多是问题到回答，缺少高质量中间步骤、过程监督、错误样本、执行反馈和 verifier 信号。模型可能学会推理格式，但不一定学会可靠推理。
```

#### 问题 3：从 chat model 到 reasoning model 需要哪些关键组件？

可以这样回答：

```text
需要高质量 CoT 和过程数据、结果和过程 verifier、多候选采样、search、test-time compute budget、工具执行反馈、合成数据闭环以及更严格的 reasoning 评估体系。
```

#### 问题 4：Reasoning model 是否应该展示完整 CoT？

可以这样回答：

```text
不一定。系统可以内部使用长推理提升质量，但对用户展示简洁解释、关键证据和最终答案。完整 CoT 可能冗长、不稳定，也可能带来安全和误导风险。
```

#### 问题 5：如何比较两个 reasoning model？

可以这样回答：

```text
不能只看准确率，还要看推理预算、token、延迟、成本、test-time compute 曲线、污染检测、鲁棒性、过程正确性和真实任务表现。
```

#### 问题 6：如何把 reasoning model 产品化？

可以这样回答：

```text
要做任务难度识别和动态预算分配，简单任务直接回答，复杂任务启用推理、验证、工具或 search；同时记录 trace，控制成本和延迟，处理安全确认，并向用户展示关键证据和不确定性。
```

---

### 十五、常见误区

1. 误区：Reasoning model 就是会输出很长 CoT 的模型。
   纠正：核心是可靠求解、验证和纠错，不是文本长度。

2. 误区：Chat model 加一句“逐步思考”就是 reasoning model。
   纠正：还需要训练数据、verifier、评估和推理时策略。

3. 误区：Reasoning model 一定适合所有任务。
   纠正：简单任务直接回答更快更便宜。

4. 误区：Benchmark 高分就代表 reasoning 强。
   纠正：要看污染、鲁棒性、真实任务和成本。

5. 误区：Verifier 可以完全替代模型能力。
   纠正：Verifier 只能筛选和指导，generator 仍要能产生好候选。

6. 误区：Test-time compute 可以无限提升能力。
   纠正：存在收益递减和评价函数错误的问题。

---

### 十六、小练习

1. 对比 chat model 和 reasoning model 的训练目标。
2. 设计一个从 chat model 升级为 reasoning model 的三阶段路线。
3. 为数学任务设计 CoT SFT + verifier + self-consistency 方案。
4. 为代码任务设计执行反馈增强方案。
5. 设计一个 reasoning system 架构图，包含 generator、verifier、search、tool 和 budget allocator。
6. 设计一个用户可见解释格式，不暴露完整内部 CoT。
7. 分析为什么长 CoT 不等于强 reasoning。
8. 比较单模型 reasoning 和系统级 reasoning 的优缺点。
9. 设计一个 dynamic reasoning mode：快速模式、深度模式、高风险模式。
10. 用 3 分钟回答：“如何从 Chat Model 走向 Reasoning Model？”

### 本讲总结

本讲最重要的结论：

1. Chat model 强调对话和指令遵循，reasoning model 强调复杂任务求解、验证、搜索和纠错。
2. 普通 SFT 不足以训练强 reasoning，需要 CoT、过程监督、verifier、执行反馈和合成推理数据。
3. Reasoning model 的数据形态从“问题-回答”扩展为“问题-过程-验证-修正-答案”。
4. Reasoning 推理方式从一次生成扩展为多候选、rerank、search、tool feedback 和动态预算。
5. Reasoning 评估必须同时看答案、过程、鲁棒性、污染、成本和 test-time compute 曲线。
6. Reasoning model 往往是系统能力，不只是单个模型能力。
7. 产品化时要在准确率、延迟、成本、安全和用户体验之间做权衡。
8. 面试中要把从 chat 到 reasoning 讲成“数据、训练、推理、验证、评估、产品化”的完整演进。

第八部分到这里结束。下一部分进入论文精读与开放研究题。
