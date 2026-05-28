# 第一章：Reasoning 总览

Reasoning model 是当前大模型能力竞争的重要方向之一。传统 chat model 更强调对话流畅、指令遵循和通用知识，而 reasoning model 更强调多步推理、数学证明、代码求解、规划、验证和复杂任务分解。它不仅要“答得像”，还要在有标准答案或可验证目标的任务上“答得对”。

Reasoning 的核心不只是让模型输出更长的解释。真正的 reasoning 涉及训练数据、Chain-of-Thought、self-consistency、verifier、process supervision、search、tool execution、test-time compute scaling 和评估体系。本章先建立全局地图，后续章节再逐个展开。

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

## 1.16 面试官会怎么问

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

## 1.17 本章总结

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
