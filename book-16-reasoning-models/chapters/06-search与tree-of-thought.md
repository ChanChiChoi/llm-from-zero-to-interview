# 第六章：Search 与 Tree-of-Thought

前面几章讲了 CoT、self-consistency、verifier 和过程监督。它们共同指向一个更一般的思想：把推理看成搜索。模型不必一次生成完整答案，而是可以在多个中间状态中探索、评分、剪枝和回溯。Tree-of-Thought、beam search、MCTS、规划式推理都属于这条路线。

本章系统讲 search reasoning：为什么需要搜索、如何定义状态和动作、Tree-of-Thought 如何工作、beam search 和 MCTS 有什么区别、verifier 如何参与剪枝、工程上如何控制成本，以及面试中如何把这类方法讲清楚。

## 6.1 为什么需要 Search

普通 CoT 是一条路径：

```text
question -> step 1 -> step 2 -> step 3 -> answer
```

复杂任务的问题在于，某一步选错后，后续推理可能全部建立在错误基础上。Search 的思想是不要只走一条路，而是保留多个候选路径，逐步探索、评分和筛选。

```text
question
  -> thought A -> thought A1 -> answer
  -> thought B -> thought B1 -> answer
  -> thought C -> pruned
```

面试回答：

```text
Search 把推理过程看成状态空间探索。模型从当前状态生成多个候选下一步，用 verifier、reward model、规则或工具反馈评分，保留更有希望的分支继续展开。它比单条 CoT 更能处理多解法和长链条任务，但计算成本更高，也更依赖评分函数质量。
```

## 6.2 状态、动作和转移

把推理建模为搜索，至少要定义五件事：

1. State：当前推理状态。
2. Action：下一步可选操作。
3. Transition：执行 action 后如何得到新状态。
4. Score：状态或动作的质量如何评估。
5. Termination：什么时候停止。

在数学题中：

```text
state = 当前已写出的推理步骤
action = 下一步数学变换或子结论
transition = 把下一步加入推理链
score = verifier 对当前路径的评分
termination = 得到最终答案或达到深度上限
```

在代码题中：

```text
state = 当前代码版本和测试结果
action = 修改代码或新增测试
transition = 应用修改并运行检查
score = 测试通过率、错误减少程度、代码质量评分
termination = 全部测试通过或预算耗尽
```

搜索能否有效，通常不是取决于“会不会展开树”，而是取决于状态表示是否完整、动作粒度是否合适、评分信号是否可靠。

## 6.3 Tree-of-Thought

Tree-of-Thought，简称 ToT，是把推理过程组织成一棵树。

普通 CoT 只有一条链，ToT 有多条分支。基本流程是：

1. 从原始问题出发。
2. 生成多个候选 thought。
3. 对每个 thought 或路径打分。
4. 选择若干分支继续展开。
5. 重复直到得到答案或达到预算上限。

这里的 thought 不一定是一句话，也不一定是一整段答案。它可以是一个中间结论、一个子问题分解、一个计划步骤、一段代码修改、一个数学变换。关键是它必须可评估、可继续展开。

粒度太小，搜索树会爆炸；粒度太大，无法定位和纠正错误。工程上常用“一个自然推理步骤”作为 thought 粒度。

## 6.4 Beam Search

Beam search 是最常见的搜索方法之一。它的做法是每一层只保留分数最高的 `beam_size` 个候选。

```text
start
generate candidates
keep top B
expand each candidate
keep top B
repeat
```

优点：

1. 简单。
2. 成本可控。
3. 容易和 verifier 结合。
4. 适合分层展开的推理任务。

缺点：

1. 可能过早丢掉潜在正确路径。
2. 结果高度依赖评分函数。
3. 候选容易同质化。
4. 如果 beam 太大，成本仍然很高。

在 reasoning 场景中，beam search 常用于“生成多个下一步，再用 verifier 或 process reward model 选择前几个”。

## 6.5 Best-First Search

Best-first search 每次优先展开当前评分最高的节点。

它的直觉是：

```text
哪个路径当前最有希望，就先探索哪个路径
```

当 verifier 分数比较可靠时，best-first search 可以快速深入高质量路径。但如果评分器偏差很大，它也可能一直沿着看似合理但实际错误的方向走。

Beam search 更像按层推进；best-first search 更像从全局候选池中挑最优节点推进。

## 6.6 MCTS 的直觉

MCTS，Monte Carlo Tree Search，常用于游戏和规划。它通常包含四步：

1. Selection：选择要展开的节点。
2. Expansion：扩展新节点。
3. Simulation：模拟后续结果。
4. Backpropagation：把结果回传更新节点价值。

在 LLM reasoning 中，可以把模型生成的 thought 当成 action，把 verifier 分数、工具验证结果或最终答案正确性当成 reward。

MCTS 的优势是能在探索新路径和利用高分路径之间做平衡。它适合规划、博弈、代码修复、复杂数学推理等任务。但它的实现和调参比 beam search 更复杂，并且需要可靠 reward，否则回传的价值估计会被噪声污染。

## 6.7 Verifier 在搜索中的作用

Search 离不开评分。常见评分来源包括：

1. LLM 自评。
2. Outcome verifier。
3. Process verifier。
4. Programmatic verifier。
5. 规则 heuristic。
6. 工具执行反馈。

Process verifier 特别适合中间节点评分，因为它能判断当前步骤是否合理。Outcome verifier 更适合终局评分。Programmatic verifier 最可靠，但只适用于答案可自动检查的任务，例如代码测试、数学计算、格式校验。

好的 verifier 可以显著减少无效搜索；差的 verifier 会误导搜索，把正确路径剪掉，把错误路径留下。

## 6.8 剪枝

剪枝是搜索中控制成本的关键。常见策略包括：

1. 分数低于阈值就丢弃。
2. 每层只保留 top-k。
3. 重复状态合并。
4. 超过最大深度停止。
5. 发现明显错误立即停止。
6. 工具验证失败则剪枝。
7. 达到 token 或延迟预算后停止。

剪枝太弱，成本爆炸；剪枝太强，可能丢掉正确路径。一个实用原则是：早期保留多样性，后期加强筛选。

## 6.9 搜索成本

如果每一步生成 `b` 个分支，深度是 `d`，最坏情况下节点数量会随 `b^d` 增长。因此 search reasoning 不能无节制开启。

工程上通常限制：

1. 最大深度。
2. 每步候选数。
3. beam size。
4. 总 token budget。
5. verifier 调用次数。
6. 工具执行次数。
7. 总延迟。

Search 更适合高价值复杂任务，例如数学竞赛题、代码修复、规划、严肃问答和需要高可靠性的决策任务。对普通闲聊或简单问答，搜索的收益通常抵不过成本。

## 6.10 Search 和 Self-Consistency 的区别

Self-consistency 是独立生成多条完整推理路径，最后投票。中间过程不干预。

Search 是分步生成、分步评分、分步剪枝，可以回溯和继续探索。

```text
self-consistency = 多条完整链 + 最后聚合
search = 多步展开 + 中间筛选 + 动态探索
```

Self-consistency 更简单，适合快速提升稳定性；search 更强，但工程复杂度和成本更高。

## 6.11 Search 和工具使用

工具可以作为搜索环境的一部分。

代码任务：

```text
state = 当前代码
action = 修改代码
tool = 运行测试
score = 通过测试数量和错误变化
```

数学任务：

```text
state = 当前推导
action = 写出下一步表达式
tool = Python 或符号计算器
score = 是否和约束一致
```

工具反馈让 search 更可靠，因为它提供外部校验。但也会带来超时、沙箱、安全、环境不一致和反馈稀疏等问题。

## 6.12 常见失败模式

1. 分支太多，成本爆炸。
2. Verifier 误剪正确路径。
3. LLM 生成的候选缺乏多样性。
4. 搜索陷入局部最优。
5. 状态表示不完整，导致评分失真。
6. 评分函数偏好格式而不是正确性。
7. 工具反馈不充分或不可用。
8. 最终答案整合错误。

很多失败不是搜索算法本身的问题，而是状态、动作、评分和预算设计的问题。

## 6.13 一个简单 ToT 伪代码

```python
frontier = [(0.0, initial_state)]

for depth in range(max_depth):
    candidates = []
    for _, state in frontier:
        thoughts = generator.generate_next_thoughts(state, n=num_branches)
        for thought in thoughts:
            new_state = state.add(thought)
            score = verifier.score(new_state)
            candidates.append((score, new_state))

    candidates.sort(key=lambda item: item[0], reverse=True)
    frontier = candidates[:beam_size]

    for score, state in frontier:
        if state.is_finished():
            return state.final_answer()

return select_best_answer(frontier)
```

这个伪代码体现了展开、评分、剪枝和停止。真实系统还需要去重、缓存、失败重试、预算控制、工具调用和日志追踪。

## 6.14 什么时候值得用 Search

适合使用 search 的场景：

1. 解空间很大。
2. 存在多种解法。
3. 中间步骤可以被评分。
4. 错误路径能被较早发现。
5. 任务价值足以覆盖额外成本。

不适合的场景：

1. 简单问答。
2. 低延迟强约束任务。
3. 无法评估中间状态的开放生成。
4. Verifier 不可靠且没有工具反馈。
5. 用户只需要快速草稿。

## 6.15 面试题：Tree-of-Thought 相比 CoT 的优势是什么

回答要点：

```text
CoT 通常是一条推理链，容易受早期错误影响。Tree-of-Thought 把推理组织成树，允许模型生成多个中间 thought，并用评分器或规则选择更好的分支继续展开。它的优势是探索更多解法、支持剪枝和回溯，适合复杂规划和数学推理。代价是计算成本更高，并且效果依赖中间评分函数。
```

## 6.16 面试题：Beam Search 和 MCTS 的区别

回答要点：

```text
Beam search 通常按层展开，每层保留 top-k 候选，简单、稳定、成本容易控制，但容易早剪正确路径。MCTS 会通过 selection、expansion、simulation 和 backpropagation 逐步估计节点价值，更强调探索和利用的平衡，适合规划类任务，但实现复杂、成本更高，也更依赖 reward 质量。
```

## 6.17 本章小结

Search reasoning 的核心是把推理从“一次生成一条答案”改成“在状态空间中探索多个候选路径”。Tree-of-Thought 提供了直观框架，beam search 提供了简单可控的实现，MCTS 提供了更强的规划能力，verifier 和工具反馈提供了剪枝依据。

真正落地时，最重要的不是把搜索树画出来，而是设计好状态、动作、评分、剪枝和预算。下一章会继续讨论 test-time compute scaling，也就是如何在推理阶段用更多计算换取更高质量答案。
