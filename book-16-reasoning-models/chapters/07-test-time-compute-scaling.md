# 第七章：Test-Time Compute Scaling

前面几章讲了 CoT、self-consistency、verifier、process supervision 和 search。它们背后都有一个共同现象：模型在推理阶段花更多计算，往往能得到更好的答案。Test-time compute scaling，推理时计算扩展，就是研究如何在推理阶段分配更多计算预算，并把这些预算转化为更高准确率、更强鲁棒性或更好可验证性的技术路线。

本章重点讲推理时计算扩展的动机、常见方式、预算分配、自适应计算、延迟和成本权衡、工程系统设计，以及面试中如何回答这类问题。

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修参考公开资料中的 inference-time / test-time compute scaling 线索，重点包括 [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)、[Large Language Monkeys](https://arxiv.org/abs/2407.21787)、Tree of Thoughts、self-consistency、verifier reranking 和 OpenAI o1 system card 中关于推理时计算、reasoning 与安全评估的公开表述。这些资料共同说明：模型参数固定后，推理阶段的采样数、候选数、搜索深度、verifier 调用和工具反馈仍然可以显著改变任务表现。

本章不把“更多 test-time compute”写成无条件更强。推理时计算扩展必须同时看准确率、边际收益、延迟、成本、候选相关性、verifier 偏差、安全风险和任务价值。尤其要避免两种误区：第一，把长 CoT 或更多采样等同于真实推理能力；第二，只汇报高预算准确率，而不汇报单位正确样本成本、P95 延迟和低价值请求的预算浪费。

## 7.1 什么是 Test-Time Compute Scaling

传统 scaling law 主要关注训练阶段：参数更多、数据更多、训练 FLOPs 更多，模型能力通常更强。

Test-time compute scaling 关注的是另一件事：模型已经训练好了，在推理时能不能通过更多计算得到更好的答案。

例如：

```text
普通推理：生成 1 条答案
更多推理计算：生成 32 条候选 + verifier 选择
搜索推理：逐步展开 + 剪枝 + 工具验证
长思考：允许模型生成更长 reasoning trace
```

面试回答：

```text
Test-time compute scaling 指在模型参数固定的情况下，通过增加推理阶段计算来提升输出质量，例如更长思考、多样本采样、自一致性投票、verifier reranking、tree search、工具调用和反思修正。它的核心问题是如何把额外计算预算分配到最有价值的样本和步骤上，同时控制延迟、成本和稳定性。
```

### 7.1.1 关键公式与 TTC Scaling 指标速查

对第 `i` 个请求，把推理时预算写成一个向量：

```math
b_i=(K_i,L_i,D_i,V_i,U_i,T_i)
```

其中 `K_i` 是候选数或采样数，`L_i` 是允许的推理长度，`D_i` 是搜索深度，`V_i` 是 verifier 调用数，`U_i` 是工具调用数，`T_i` 是总 token 数。Test-time compute scaling 的核心不是让所有分量都变大，而是按任务价值和难度分配这些预算。

给定一种推理策略 `m`，第 `i` 个请求的抽象成本可以写成：

```math
C_i(m)=
c_{\mathrm{tok}}T_i
+c_{\mathrm{ver}}V_i
+c_{\mathrm{tool}}U_i
+c_{\mathrm{lat}}R_i
```

其中 `R_i` 是用户可感知延迟或服务占用时间。不同业务可以调整各项权重，例如交互式产品更重视 `R_i`，批处理评测更重视总 token 和 verifier 调用。

固定预算下的准确率曲线可以写成：

```math
A(B)=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[\hat y_i(B)=y_i^\star]
```

边际收益衡量多花一段预算换来多少准确率提升：

```math
g(B_1,B_2)=
\frac{A(B_2)-A(B_1)}
{C(B_2)-C(B_1)}
```

如果每个候选独立命中正确解的概率粗略为 `p_i`，生成 `K` 个候选时“至少出现一个正确候选”的理想化概率为：

```math
P_i(K)=1-(1-p_i)^K
```

这个公式只表达直觉：更多候选提高覆盖正确解的概率，但真实候选高度相关，增长会更快进入边际收益递减。

并行候选生成和串行搜索的延迟口径不同。一个简化并行口径是：

```math
R_i^{\mathrm{parallel}}=
\max_{1\le k\le K_i}R_{ik}
+R_i^{\mathrm{verify}}
```

搜索或工具循环更接近串行：

```math
R_i^{\mathrm{loop}}=
\sum_{t=1}^{H_i}
(R_{it}^{\mathrm{gen}}+R_{it}^{\mathrm{tool}}+R_{it}^{\mathrm{verify}})
```

因此同样的 token 数，在用户体验上可能完全不同。

自适应计算可以写成一个路由函数：

```math
m_i=\pi(x_i,d_i,v_i,u_i)
```

其中 `d_i` 是难度估计，`v_i` 是任务价值，`u_i` 是可验证性或工具可用性。路由函数决定该请求走 direct、self-consistency、verifier、search 还是 tool loop。

一个简化上线门禁：

```math
G_{\mathrm{ttc}}=
\mathbb{1}[
A_{\mathrm{adapt}}\ge \alpha
\land
C_{\mathrm{adapt}}\le C_{\max}
\land
R_{95}\le R_{\max}
\land
W_{\mathrm{waste}}\le \omega
]
```

其中 `W_waste` 表示高预算但仍失败、或者低价值请求消耗高预算的浪费率。面试里要强调：TTC scaling 的目标不是“所有请求都更慢”，而是把额外计算投给低置信度、高价值、可验证的请求。

## 7.2 为什么推理时计算有用

很多复杂任务不是模型完全不会，而是第一次采样不一定走到正确路径。增加推理计算可以带来几类收益：

1. 提高探索范围。
2. 降低单次采样偶然性。
3. 给 verifier 更多候选可选。
4. 允许模型检查和修正错误。
5. 让工具反馈参与决策。
6. 对困难样本投入更多预算。

一个直观例子是数学题。模型一次生成可能犯错，但生成多条推理路径后，其中可能有一条正确。如果有可靠 verifier 或最终答案投票，就能选出更好的答案。

## 7.3 常见推理时扩展方式

常见方法包括：

1. Long CoT：允许模型生成更长推理。
2. Self-consistency：采样多条推理链并聚合。
3. Best-of-N：生成 N 个候选，用评分器选择。
4. Verifier reranking：用 verifier 重排候选。
5. Tree-of-Thought：逐步展开推理树。
6. MCTS：用树搜索估计路径价值。
7. Reflection：先回答，再自查并修正。
8. Tool-use loops：生成、执行工具、观察反馈、继续推理。

这些方法本质上都是用更多推理计算换更高答案质量，只是计算花在不同位置。

## 7.4 Best-of-N

Best-of-N 是最简单的 test-time scaling 方法。

流程：

```text
输入问题
生成 N 个候选答案
给每个候选打分
选择最高分答案
```

它的优点是实现简单、并行友好、容易扩展。缺点是需要评分器。如果评分器不可靠，N 越大可能越容易选到“看起来好但实际错”的答案。

Best-of-N 的效果通常取决于两个因素：候选中是否存在正确答案，以及评分器是否能识别它。

## 7.5 Self-Consistency

Self-consistency 也是一种推理时计算扩展。它生成多条推理路径，然后对最终答案投票。

适用条件：

1. 最终答案形式比较明确。
2. 多条路径可以独立采样。
3. 正确答案在采样中出现概率不低。
4. 错误答案分散，正确答案更容易聚集。

它不需要训练额外 verifier，但对开放式问答、创意写作、主观任务不一定适用。

## 7.6 Verifier Reranking

Verifier reranking 是 reasoning model 中非常常见的推理时扩展方法。

基本流程：

```text
generator 生成多个候选
verifier 给候选评分
系统返回分数最高的候选
```

如果 verifier 是 outcome verifier，它主要看最终答案是否正确。如果 verifier 是 process verifier，它还可以看中间步骤是否合理。

Reranking 的关键不是生成更多文本，而是让评分器参与选择。很多时候，generator 已经能生成正确答案，但单次采样不稳定；reranking 可以把正确候选挑出来。

## 7.7 Adaptive Compute

所有问题都用同样推理预算并不合理。简单问题一次回答即可，困难问题才需要更多计算。Adaptive compute 的目标是根据样本难度动态分配预算。

常见策略：

1. 先用低成本模式回答。
2. 如果置信度低，再增加采样。
3. 如果候选分歧大，调用 verifier。
4. 如果 verifier 仍不确定，启动搜索。
5. 如果工具可验证，调用工具。
6. 达到预算上限后停止。

一个实用流程：

```text
easy query -> direct answer
medium query -> self-consistency
hard query -> search + verifier + tools
```

Adaptive compute 的难点是如何判断“这题难不难”和“现在是否应该继续花钱”。

## 7.8 置信度估计

自适应计算依赖置信度估计。常见信号包括：

1. 模型 log probability。
2. 多样本答案一致性。
3. Verifier 分数。
4. 候选之间的分歧程度。
5. 工具验证结果。
6. 历史错误模式。
7. 输入任务类型。

但这些信号都不完美。模型可能高置信度地犯错，verifier 也可能被表面合理的答案欺骗。因此工程系统通常会组合多个信号，而不是只依赖单一置信度。

## 7.9 Latency Trade-off

推理时计算扩展最直接的代价是延迟。

如果生成一个答案需要 2 秒，生成 16 个候选即使并行，也会带来更高排队、显存、吞吐和 verifier 成本。如果还要搜索和工具调用，延迟可能进一步上升。

工程上要区分：

1. 用户可感知延迟。
2. 后台总计算成本。
3. GPU 吞吐损失。
4. 工具调用等待时间。
5. verifier 服务延迟。
6. 超时和重试成本。

高质量模式可以慢，但不能不可控。系统必须设置硬预算和降级策略。

## 7.10 Cost Trade-off

推理计算扩展也会显著增加成本。

成本来源：

1. 更多输出 token。
2. 多次模型调用。
3. Verifier 调用。
4. 工具执行。
5. 搜索树展开。
6. 缓存和日志存储。

业务上需要回答：额外准确率是否值得额外成本。对于医疗、法律、代码修复、金融分析等高价值任务，更多推理计算可能值得；对于简单客服问答，可能不值得。

## 7.11 Budget Allocation

预算分配比“预算越多越好”更重要。

常见分配方式：

1. 增加候选数量。
2. 增加每条推理长度。
3. 增加搜索深度。
4. 增加 verifier 精度。
5. 增加工具调用。
6. 增加反思轮数。

不同任务适合不同分配方式。数学题可能更适合多路径采样和 verifier；代码题可能更适合工具执行和迭代修复；规划任务可能更适合搜索和约束检查。

## 7.12 Diminishing Returns

推理时计算通常存在边际收益递减。N 从 1 增加到 8，提升可能明显；N 从 64 增加到 128，提升可能很小。

原因包括：

1. 候选开始重复。
2. Verifier 成为瓶颈。
3. 模型能力上限限制。
4. 错误模式高度相关。
5. 任务本身缺少可验证信号。

因此系统应该测量不同预算下的质量曲线，而不是盲目增加推理次数。

## 7.13 工程系统设计

一个支持 test-time compute scaling 的系统通常包含：

1. Router：判断任务类型和难度。
2. Generator：生成候选或下一步。
3. Verifier：评估候选质量。
4. Search Controller：控制展开、剪枝和停止。
5. Tool Executor：执行外部工具。
6. Budget Manager：管理 token、时间、调用次数。
7. Aggregator：聚合候选并输出最终答案。
8. Logger：记录路径、评分和失败原因。

这类系统的核心不是单个 prompt，而是一个受预算约束的推理控制器。

## 7.14 常见失败模式

1. 花了更多计算，但候选高度重复。
2. Verifier 选错，把错误答案排第一。
3. 搜索过深，延迟不可控。
4. 低价值请求消耗高预算。
5. 置信度估计错误，难题被当成简单题。
6. 反思轮数增加，但模型只是在重复原错误。
7. 工具调用失败后系统没有降级。
8. 日志不完整，无法分析预算浪费在哪里。

推理时计算扩展不是“多生成几次”这么简单，而是要把额外计算用在能提升正确率的位置。

## 7.15 最小可运行 TTC scaling / 动态预算审计 demo

这个 demo 模拟 6 个 toy 请求，在四种固定策略和一个 adaptive routing 策略之间比较：

1. `direct`：低预算直接回答。
2. `self_consistency`：多样本采样与聚合。
3. `verifier`：候选生成加 verifier。
4. `search`：搜索加 verifier / tool feedback。
5. `adaptive`：按难度、任务价值和可验证性路由。

它刻意保留 `adversarial_math`：adaptive 给它分配了高于 direct 的预算，但仍然失败，所以 `gate_pass=False`。这个失败用来提醒读者：动态预算系统不能只看平均准确率，也要看高算力浪费和 hard slice。

```python
CASES = [
    {
        "id": "easy_lookup",
        "difficulty": 0.15,
        "value": 0.20,
        "verifiable": False,
        "modes": {
            "direct": {"correct": True, "tokens": 32, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 120},
            "self_consistency": {"correct": True, "tokens": 180, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 260},
            "verifier": {"correct": True, "tokens": 240, "verifier_calls": 3, "tool_calls": 0, "latency_ms": 430},
            "search": {"correct": True, "tokens": 520, "verifier_calls": 5, "tool_calls": 0, "latency_ms": 900},
        },
    },
    {
        "id": "factual_short",
        "difficulty": 0.25,
        "value": 0.30,
        "verifiable": False,
        "modes": {
            "direct": {"correct": True, "tokens": 36, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 130},
            "self_consistency": {"correct": True, "tokens": 210, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 280},
            "verifier": {"correct": True, "tokens": 260, "verifier_calls": 3, "tool_calls": 0, "latency_ms": 460},
            "search": {"correct": True, "tokens": 540, "verifier_calls": 5, "tool_calls": 0, "latency_ms": 920},
        },
    },
    {
        "id": "algebra_hard",
        "difficulty": 0.62,
        "value": 0.80,
        "verifiable": True,
        "modes": {
            "direct": {"correct": False, "tokens": 90, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 220},
            "self_consistency": {"correct": True, "tokens": 520, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 680},
            "verifier": {"correct": True, "tokens": 430, "verifier_calls": 4, "tool_calls": 0, "latency_ms": 620},
            "search": {"correct": True, "tokens": 760, "verifier_calls": 6, "tool_calls": 0, "latency_ms": 1180},
        },
    },
    {
        "id": "code_patch",
        "difficulty": 0.90,
        "value": 0.95,
        "verifiable": True,
        "modes": {
            "direct": {"correct": False, "tokens": 120, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 260},
            "self_consistency": {"correct": False, "tokens": 650, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 760},
            "verifier": {"correct": False, "tokens": 610, "verifier_calls": 5, "tool_calls": 0, "latency_ms": 820},
            "search": {"correct": True, "tokens": 940, "verifier_calls": 5, "tool_calls": 2, "latency_ms": 1420},
        },
    },
    {
        "id": "logic_puzzle",
        "difficulty": 0.78,
        "value": 0.70,
        "verifiable": False,
        "modes": {
            "direct": {"correct": False, "tokens": 100, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 240},
            "self_consistency": {"correct": True, "tokens": 560, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 740},
            "verifier": {"correct": True, "tokens": 620, "verifier_calls": 5, "tool_calls": 0, "latency_ms": 860},
            "search": {"correct": True, "tokens": 980, "verifier_calls": 8, "tool_calls": 0, "latency_ms": 1500},
        },
    },
    {
        "id": "adversarial_math",
        "difficulty": 0.82,
        "value": 0.60,
        "verifiable": True,
        "modes": {
            "direct": {"correct": False, "tokens": 110, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 240},
            "self_consistency": {"correct": False, "tokens": 620, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 760},
            "verifier": {"correct": False, "tokens": 590, "verifier_calls": 5, "tool_calls": 0, "latency_ms": 840},
            "search": {"correct": False, "tokens": 900, "verifier_calls": 7, "tool_calls": 0, "latency_ms": 1450},
        },
    },
]

TOKEN_COST = 1.0
VERIFIER_COST = 45.0
TOOL_COST = 100.0


def unit_cost(stats):
    return round(
        TOKEN_COST * stats["tokens"]
        + VERIFIER_COST * stats["verifier_calls"]
        + TOOL_COST * stats["tool_calls"],
        3,
    )


def choose_route(case):
    if case["difficulty"] <= 0.35:
        return "direct"
    if case["verifiable"] and case["value"] >= 0.90:
        return "search"
    if case["verifiable"] and case["difficulty"] <= 0.70:
        return "verifier"
    if case["difficulty"] >= 0.60:
        return "self_consistency"
    return "direct"


def summarize(mode):
    chosen = [case["modes"][mode] for case in CASES]
    correct = sum(item["correct"] for item in chosen)
    cost = sum(unit_cost(item) for item in chosen)
    latencies = sorted(item["latency_ms"] for item in chosen)
    return {
        "accuracy": round(correct / len(chosen), 3),
        "total_cost": round(cost, 3),
        "cost_per_correct": round(cost / max(1, correct), 3),
        "p95_latency_ms": latencies[-1],
    }


fixed_modes = ["direct", "self_consistency", "verifier", "search"]
fixed_summary = {mode: summarize(mode) for mode in fixed_modes}

adaptive_records = []
for case in CASES:
    route = choose_route(case)
    stats = case["modes"][route]
    adaptive_records.append(
        {
            "id": case["id"],
            "route": route,
            "correct": stats["correct"],
            "cost": unit_cost(stats),
            "latency_ms": stats["latency_ms"],
        }
    )

adaptive_correct = sum(item["correct"] for item in adaptive_records)
adaptive_cost = sum(item["cost"] for item in adaptive_records)
adaptive_latencies = sorted(item["latency_ms"] for item in adaptive_records)
adaptive_summary = {
    "accuracy": round(adaptive_correct / len(adaptive_records), 3),
    "total_cost": round(adaptive_cost, 3),
    "cost_per_correct": round(adaptive_cost / max(1, adaptive_correct), 3),
    "p95_latency_ms": adaptive_latencies[-1],
}

marginal = {}
base_acc = fixed_summary["direct"]["accuracy"]
base_cost = fixed_summary["direct"]["total_cost"]
for mode in ["self_consistency", "verifier", "search"]:
    acc_gain = fixed_summary[mode]["accuracy"] - base_acc
    cost_gain = fixed_summary[mode]["total_cost"] - base_cost
    marginal[mode] = round(acc_gain / cost_gain, 6) if cost_gain else 0.0

wasted_high_compute = [
    item["id"]
    for item in adaptive_records
    if item["route"] != "direct" and not item["correct"]
]

gates = {
    "adaptive_accuracy_ok": adaptive_summary["accuracy"] >= 0.8,
    "adaptive_cheaper_than_search": adaptive_summary["total_cost"] < fixed_summary["search"]["total_cost"],
    "latency_ok": adaptive_summary["p95_latency_ms"] <= 1500,
    "wasted_high_compute_ok": len(wasted_high_compute) == 0,
}

print(f"fixed_summary={fixed_summary}")
print(f"adaptive_records={adaptive_records}")
print(f"adaptive_summary={adaptive_summary}")
print(f"marginal_accuracy_per_cost={marginal}")
print(f"wasted_high_compute={wasted_high_compute}")
print(f"gates={gates}")
print(f"gate_pass={all(gates.values())}")
```

预期输出：

```text
fixed_summary={'direct': {'accuracy': 0.333, 'total_cost': 488.0, 'cost_per_correct': 244.0, 'p95_latency_ms': 260}, 'self_consistency': {'accuracy': 0.667, 'total_cost': 2740.0, 'cost_per_correct': 685.0, 'p95_latency_ms': 760}, 'verifier': {'accuracy': 0.667, 'total_cost': 3875.0, 'cost_per_correct': 968.75, 'p95_latency_ms': 860}, 'search': {'accuracy': 0.833, 'total_cost': 6460.0, 'cost_per_correct': 1292.0, 'p95_latency_ms': 1500}}
adaptive_records=[{'id': 'easy_lookup', 'route': 'direct', 'correct': True, 'cost': 32.0, 'latency_ms': 120}, {'id': 'factual_short', 'route': 'direct', 'correct': True, 'cost': 36.0, 'latency_ms': 130}, {'id': 'algebra_hard', 'route': 'verifier', 'correct': True, 'cost': 610.0, 'latency_ms': 620}, {'id': 'code_patch', 'route': 'search', 'correct': True, 'cost': 1365.0, 'latency_ms': 1420}, {'id': 'logic_puzzle', 'route': 'self_consistency', 'correct': True, 'cost': 560.0, 'latency_ms': 740}, {'id': 'adversarial_math', 'route': 'self_consistency', 'correct': False, 'cost': 620.0, 'latency_ms': 760}]
adaptive_summary={'accuracy': 0.833, 'total_cost': 3223.0, 'cost_per_correct': 644.6, 'p95_latency_ms': 1420}
marginal_accuracy_per_cost={'self_consistency': 0.000148, 'verifier': 9.9e-05, 'search': 8.4e-05}
wasted_high_compute=['adversarial_math']
gates={'adaptive_accuracy_ok': True, 'adaptive_cheaper_than_search': True, 'latency_ok': True, 'wasted_high_compute_ok': False}
gate_pass=False
```

这个输出说明：固定 search 准确率最高，但成本也最高；adaptive routing 用接近一半的成本达到同样的准确率。不过 `adversarial_math` 暴露了一个上线前必须处理的问题：系统识别到它难，却没有选对有效策略。真实工程里要继续补 hard-slice detector、更强 verifier、工具校验或人工复核，而不是只看平均准确率达标。

## 7.16 面试题：什么是 Test-Time Compute Scaling

回答要点：

```text
Test-time compute scaling 是指模型参数固定时，通过增加推理阶段计算来提升答案质量。典型方法包括长 CoT、多样本采样、self-consistency、best-of-N、verifier reranking、tree search、工具调用和反思修正。它的关键挑战是预算分配和成本控制，因为更多计算会带来延迟和费用，且存在边际收益递减。
```

## 7.17 面试题：如何设计 Adaptive Compute

回答要点：

```text
我会先设计一个任务路由和置信度估计模块。简单任务直接回答；中等难度任务使用多样本采样或 self-consistency；高难度任务启动 verifier、搜索或工具调用。系统需要设置 token、延迟、调用次数等预算上限，并记录每一步收益。核心是把计算集中在低置信度、高价值、可验证的样本上，而不是所有请求平均加预算。
```

## 7.18 小练习

1. 给一个在线问答产品设计 direct、self-consistency、verifier 和 search 四档预算。
2. 写出你自己的 TTC 成本函数，至少包含 token、verifier 调用、工具调用和延迟。
3. 修改本章 demo，让 `adversarial_math` 走 search，观察准确率、成本和 gate 如何变化。
4. 设计一个 hard-slice 报告，说明哪些题应该升级到人工复核而不是继续增加采样数。
5. 用 3 句话解释为什么 test-time compute scaling 会有边际收益递减。

## 7.19 本章小结

Test-time compute scaling 是 reasoning model 的重要方向。它说明模型能力不只来自训练阶段，也来自推理阶段如何使用计算。Long CoT、self-consistency、best-of-N、verifier reranking、search、tools 和 reflection 都是推理时扩展的具体形式。

真正的工程问题是：哪些请求值得更多计算，额外预算应该用于采样、搜索、验证还是工具，什么时候停止，以及如何在质量、成本和延迟之间取得平衡。下一章会进入数学推理训练，讨论 reasoning model 在数学任务上如何构造数据、训练和评估。
