# 第三章：Self-Consistency 与采样

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

## 3.16 面试官会怎么问

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

## 3.17 小练习

1. 对一道数学题采样 5 条 CoT，并手动投票。
2. 写一个 `extract_final_answer` 函数。
3. 写一个答案标准化函数，处理整数、小数和单位。
4. 比较 greedy、temperature=0.3、temperature=0.8 的输出差异。
5. 设计一个 pass@k 评估流程。
6. 画出 k 从 1 到 16 的成本-质量曲线。
7. 构造一个 majority vote 选错但 verifier 能选对的例子。

## 3.18 本章总结

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
