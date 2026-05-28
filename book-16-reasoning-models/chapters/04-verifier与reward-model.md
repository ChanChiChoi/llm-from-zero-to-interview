# 第四章：Verifier 与 Reward Model

Reasoning model 的一个核心趋势是“生成”和“验证”分工。生成模型负责提出候选答案、推理链或代码；verifier 或 reward model 负责判断哪个候选更可能正确。这个范式非常重要，因为复杂推理中，模型一次生成就答对很难，但生成多个候选再筛选，常常能显著提升准确率。

本章系统讲 verifier、reward model、outcome verifier、process verifier、reranking、pairwise/listwise 训练、生成-验证范式、工程使用方式和常见风险。

## 4.1 为什么需要 Verifier

单个生成模型容易出现：

1. 中间步骤错误。
2. 最终答案错误。
3. 多条候选质量不一。
4. 推理链看似合理但不可靠。
5. 自信输出错误答案。

如果能额外训练或使用一个 verifier，就可以让系统从多个候选中选择更好的答案。

直觉：

```text
Generator: 给出多个可能答案
Verifier: 判断哪个答案更可信
```

面试回答：

```text
Verifier 的作用是评估候选答案或推理过程的质量。Reasoning 中生成模型负责提出多个候选，verifier 负责打分、筛选或重排。这样可以把“会生成正确答案”和“能选出正确答案”拆开，提升复杂推理任务的可靠性。
```

## 4.2 Outcome Verifier

Outcome verifier 只看最终答案是否正确或更好。

输入可能是：

```text
question + final answer
```

或者：

```text
question + reasoning + final answer
```

输出：

```text
score
```

它不一定逐步检查过程，只判断最后结果质量。

优点：

1. 标注相对容易。
2. 适合答案可验证任务。
3. 可以用于候选重排。

缺点：

1. 无法指出哪一步错。
2. 对答案格式敏感。
3. 如果最终答案碰巧对，过程错误也可能高分。

## 4.3 Process Verifier

Process verifier 检查中间步骤。

输入：

```text
question + step_1 + step_2 + ...
```

输出可以是：

1. 每一步正确/错误。
2. 每一步分数。
3. 整条推理链分数。

优点：

1. 能更早发现错误。
2. 给搜索提供中间节点评分。
3. 适合 process supervision。

缺点：

1. 标注成本高。
2. 正确步骤可能有多种表达。
3. 步骤粒度难定义。
4. 评分器本身也可能错。

面试中可以说：outcome verifier 看结果，process verifier 看过程。

## 4.4 Reward Model 和 Verifier 的关系

Reward model 通常给模型输出打一个偏好分数。

在 RLHF 中，reward model 学的是人类偏好；在 reasoning 中，reward model 可以学候选答案是否正确、推理过程是否可靠、是否更符合题目要求。

关系：

1. Verifier 更强调正确性验证。
2. Reward model 更泛化，可能评分 helpfulness、format、safety、reasoning quality。
3. 在很多系统中，两者都可以作为候选重排器。

可以把 reasoning reward model 看成一种 learned verifier。

## 4.5 生成-验证范式

生成-验证范式流程：

1. Generator 生成多个候选。
2. Verifier 给每个候选打分。
3. 选择最高分候选。
4. 必要时继续搜索或修正。

伪代码：

```python
candidates = []

for _ in range(k):
    answer = generator.generate(question)
    score = verifier.score(question, answer)
    candidates.append((score, answer))

best = max(candidates, key=lambda x: x[0])
```

这种方法适合：

1. 数学题。
2. 代码题。
3. 规划任务。
4. 工具使用。
5. 多候选问答。

## 4.6 Reranking

Reranking 是 verifier 最常见的用法之一。

流程：

1. 生成 `N` 个候选答案。
2. verifier 对每个候选打分。
3. 按分数排序。
4. 返回 top-1 或 top-k。

和 self-consistency 的区别：

1. Self-consistency 通常按答案投票。
2. Reranking 按候选质量打分。
3. Reranking 可以选中少数但高质量的答案。

如果多数候选错但有一个正确，majority vote 可能失败；好的 verifier rerank 可能成功。

## 4.7 Pairwise Ranking

Pairwise 训练让模型判断两个候选哪个更好。

样本：

```json
{
  "question": "...",
  "chosen": "正确解法",
  "rejected": "错误解法"
}
```

训练目标：让 chosen 分数高于 rejected。

优点：

1. 偏好标注比绝对打分容易。
2. 和 RLHF/DPO 数据形式类似。
3. 适合训练 reward model。

缺点：

1. 只知道相对偏好。
2. 数据覆盖不足时泛化差。
3. 可能学到长度、格式等偏差。

## 4.8 Pointwise 和 Listwise

除了 pairwise，还有 pointwise 和 listwise。

Pointwise：每个候选单独打分。

```text
candidate -> score
```

Listwise：一次输入多个候选，直接学习排序。

```text
[candidate_1, candidate_2, candidate_3] -> ranking
```

比较：

1. Pointwise 简单，但分数校准难。
2. Pairwise 标注容易，常用。
3. Listwise 更贴近排序目标，但训练复杂。

## 4.9 Verifier 训练数据

Verifier 数据来源：

1. 人工标注。
2. 规则自动标注。
3. 数学答案校验。
4. 代码单元测试。
5. LLM-as-Judge 初筛。
6. 生成模型产生错误候选。

高质量 verifier 数据需要包含：

1. 正确候选。
2. 常见错误候选。
3. 迷惑性错误。
4. 格式不同但正确的答案。
5. 中间步骤错误但最终碰巧对的样本。
6. 中间步骤合理但最终算错的样本。

如果 negative 太简单，verifier 只会学会区分明显错误，无法处理真实候选。

## 4.10 Verifier 的偏差

Verifier 可能学到错误偏差。

常见偏差：

1. 偏好更长答案。
2. 偏好格式更整齐的答案。
3. 偏好更自信的语气。
4. 偏好训练集中常见模式。
5. 不能识别隐蔽计算错误。

这会导致 reward hacking：生成模型学会骗过 verifier，而不是真的提高正确性。

解决方向：

1. 增加 hard negatives。
2. 做 calibration。
3. 用工具验证补充。
4. 定期更新 verifier 数据。
5. 分开评估 verifier 和 generator。

## 4.11 Programmatic Verifier

程序化 verifier 是最可靠的一类验证器之一。

例子：

1. 数学表达式用 Python 计算。
2. 代码题运行单元测试。
3. SQL 查询执行结果。
4. JSON schema 校验。
5. 形式化证明检查器。

优点：

1. 客观。
2. 可复现。
3. 不容易被语言风格欺骗。

缺点：

1. 只适用于可执行或可规则验证任务。
2. 测试用例可能不完整。
3. 执行环境有安全风险。
4. 需要处理超时和沙箱。

## 4.12 Verifier 与 Search

Search 需要中间状态评分。Process verifier 可以给每一步打分，帮助剪枝。

流程：

1. 当前推理状态生成多个下一步。
2. verifier 给每个下一步评分。
3. 保留高分分支。
4. 继续展开。

这比只在最后评分更高效，因为错误路径可以早停。

但如果 verifier 过早误杀正确分支，搜索也会失败。

## 4.13 ORM 和 PRM

Reasoning 中常见两个缩写：

1. ORM：Outcome Reward Model。
2. PRM：Process Reward Model。

ORM 评价最终答案或整条输出。

PRM 评价每一步过程。

对比：

```text
ORM: 这道题最终答对了吗？
PRM: 这一步推理对吗？下一步是否合理？
```

PRM 更适合搜索和过程监督，但标注成本更高。

## 4.14 评估 Verifier

Verifier 本身也要评估。

指标：

1. Pairwise accuracy。
2. Ranking accuracy。
3. Calibration。
4. Top-1 selection accuracy。
5. Hard negative accuracy。
6. 下游任务提升。

最关键的是下游提升：使用 verifier 后，最终 reasoning accuracy 是否提高，成本是否可接受。

不要只看 verifier 在验证集上的分类准确率，因为它可能在真实生成候选分布上失效。

## 4.15 工程使用建议

使用 verifier 时建议：

1. 先建立 greedy baseline。
2. 再做 self-consistency baseline。
3. 加 verifier rerank。
4. 对比 accuracy、tokens、latency、cost。
5. 分析 verifier 选错的样本。
6. 检查是否偏好长答案或格式。
7. 对代码和数学尽量加入程序验证。

如果 verifier 带来的收益小于成本，就不一定值得上线。

## 4.16 面试官会怎么问

### 问题一：Verifier 和 generator 的关系是什么？

回答模板：

```text
Generator 负责生成候选答案或推理路径，verifier 负责评估候选质量并选择更可靠的输出。这个生成-验证范式能把“提出可能解”和“判断哪个解更好”分开，常用于数学、代码和复杂推理任务。
```

### 问题二：Outcome verifier 和 process verifier 有什么区别？

回答模板：

```text
Outcome verifier 主要看最终答案或整条输出是否正确，标注相对容易但无法定位中间错误。Process verifier 会检查每一步推理是否合理，可以用于搜索和过程监督，但标注成本高、步骤粒度也更难定义。
```

### 问题三：Reward model 和 verifier 有什么区别？

回答模板：

```text
Verifier 更强调正确性验证，reward model 更泛化，可以学习人类偏好、helpfulness、format、safety 和 reasoning quality。在 reasoning 场景中，reward model 可以作为 learned verifier 给候选答案打分和重排。
```

### 问题四：Verifier 为什么可能被 reward hacking？

回答模板：

```text
如果 verifier 学到的是长度、格式、自信语气等表面特征，generator 可能优化这些特征来骗过 verifier，而不是真的提高正确性。解决方法包括 hard negatives、工具验证、calibration、定期更新数据和独立评估下游正确率。
```

### 问题五：代码任务中最好的 verifier 是什么？

回答模板：

```text
通常是执行测试用例。代码是否正确可以通过编译、运行单元测试、检查输出和性能约束来验证。相比语言 reward model，程序执行更客观，但需要沙箱、超时控制和足够覆盖的测试集。
```

## 4.17 小练习

1. 构造一个 question/chosen/rejected 的 verifier 训练样本。
2. 比较 outcome verifier 和 process verifier 的适用场景。
3. 写一个简单 rerank 伪代码。
4. 设计一个代码题 programmatic verifier。
5. 列出 5 种 verifier 偏差。
6. 设计一个 verifier 评估表。
7. 构造一个 majority vote 失败但 verifier 成功的例子。

## 4.18 本章总结

Verifier 和 reward model 是 reasoning 系统中非常重要的组件。它们让系统不再只依赖一次生成，而是可以生成多个候选、打分、重排、搜索和验证。

需要记住：

1. Outcome verifier 看最终结果。
2. Process verifier 看中间步骤。
3. Reward model 可以作为 learned verifier。
4. Reranking 能从多个候选中选择更好答案。
5. Programmatic verifier 在代码和可执行任务中非常强。
6. Verifier 也会有偏差和 reward hacking 风险。
7. 最终要看 verifier 是否提升下游 accuracy，并且成本可接受。

下一章会进入 process supervision，进一步讲如何监督中间步骤、如何构造过程标注，以及 PRM 在 search 和 reasoning 中的作用。
