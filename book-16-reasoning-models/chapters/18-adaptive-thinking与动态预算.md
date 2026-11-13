# 第 18 章 自适应思考（Adaptive Thinking）：把推理预算变成可校准的控制环

## 18.1 为什么所有请求都用最高预算并不合理

把每个请求都交给最高思考档位，实施上很简单，却把三类任务混在了一起。简单问题可能已经能够可靠回答，额外 token（词元）只增加费用和排队；困难问题可能真正缺少文档、工具观察或人工授权，延长内部生成并不会提供这些信息；高风险问题即使模型愿意继续思考，也不能因此获得更多权限。

自适应思考的目标不是让系统猜测模型“聪不聪明”，而是让控制器根据可观察进展决定下一步资源。它可以增加推理 token、采样候选、调用检索、运行测试、请求人工或直接停止。每次升级都要有触发原因，每次停止都要有可解释状态，预算和权限还必须分开管理。

可以把一次请求看成一个有限资源决策问题：在当前证据、剩余时间和风险下，继续花费多少计算，是否能带来足够的任务价值？这和单纯把 `max_output_tokens` 调大不同，因为新增资源可能被分配给工具、验证、人工或恢复。

本章沿用[第七章](07-test-time-compute-scaling.md)对测试时计算策略和质量—成本曲线的分析，以及[第十三章](13-reasoning-effort与推理预算.md)定义的资源单位与请求账本，进一步讨论单个任务运行时如何根据新证据调整预算。

## 18.2 控制器的状态

令 \(t\) 为非负整数决策步，\(x\) 为原始任务，\(o_t\) 为截至当前已确认的观察和证据，\(a_t\) 为已有产物或候选，\(b_t\) 为剩余预算向量，\(d_t\) 为以统一时间单位计量的有限非负剩余时长，\(r_t\) 为风险状态，\(c_t\) 为服务容量状态。状态 \(s_t\) 应保存可观测的验证器（verifier）、工具和权限事实，而不能只记录模型自报信心：

~~~math
s_t=(x,o_t,a_t,b_t,d_t,r_t,c_t)
~~~

这组字段让控制器可以依据证据、产物、预算、时限、风险和容量选择下一步动作；模型自报信心最多是候选信号，不能代替外部验证事实。

设 \(b_{\mathrm{reason}}\)、\(b_{\mathrm{tool}}\)、\(b_{\mathrm{verify}}\)、\(b_{\mathrm{time}}\) 和 \(b_{\mathrm{side}}\) 分别表示剩余推理 token、工具调用、验证调用、时间和副作用额度；每个分量都是有限非负量，并保留自己的单位。\(b_{\mathrm{side}}=0\) 表示不允许外部副作用。剩余预算向量可写成：

~~~math
\mathbf{b}_t
=(b_{\mathrm{reason}},b_{\mathrm{tool}},b_{\mathrm{verify}},b_{\mathrm{time}},b_{\mathrm{side}})
~~~

不能把 token、毫秒和写操作次数直接相加。权限属于策略状态，不因预算向量变大而自动变化。

控制器的动作集合可以写成：

~~~text
keep -> upgrade_reason -> call_tool -> verify
     -> ask_clarification -> abstain -> human -> terminal
~~~

这些动作不是模型文本，而是运行时可以审计和计费的状态转换。`call_tool` 需要独立的允许列表（allowlist）和授权；`human` 可能增加等待时间，但不是“模型失败”的同义词；`abstain` 表示当前证据不足以满足成功契约。

## 18.3 难度、风险与可验证性是三个维度

输入长度、任务难度、任务风险和可验证性经常被错误地压成一个分数。长日志可能只需抽取一个字段，短问题可能要求复杂证明；数学题可能难但风险低，支付确认可能简单却风险高；工具返回空结果可能不是难度变大，而是证据状态未知。

可用信号包括：任务类型、历史同类失败率、引用覆盖、候选分歧、验证器结果、工具错误、剩余截止时间、数据新鲜度和容量水位。模型自报“我很有把握”最多是一个候选信号，不能直接授权危险动作。

令 \(d\) 为按目标任务口径计算的有限难度分数，\(\hat p_{\mathrm{fail}}(d)\in[0,1]\) 为用固定任务集校准得到的失败概率估计；\(P(\mathrm{fail}\mid d)\) 表示在该难度分数下的条件失败概率。可将二者的校准关系写为：

~~~math
\hat p_{\mathrm{fail}}(d)
\approx P(\mathrm{fail}\mid d)
~~~

这只是校准关系，不是模型内部真实难度。校准必须按任务类型、语言、领域、输入长度和风险分组；未覆盖的组不能因为缺少历史失败就默认为简单。

## 18.4 边际收益：下一单位预算换来什么

设 \(s\) 为当前可观测任务状态，\(\Delta\mathbf{b}\) 为有限非负的候选资源增量。\(Q\in[0,1]\) 是由独立评估定义的质量指标，\(\Delta Q(\Delta\mathbf{b}\mid s)\in[-1,1]\) 是同一任务契约下投入这些资源后的质量变化；\(\Delta C\) 与 \(\Delta R\) 分别是有限成本和风险增量。\(V\ge0\) 是有限的任务价值，\(\lambda,\mu\ge0\) 是有限权重，分别把成本和风险换算成与 \(V\Delta Q\) 相同的效用单位。由此得到一个教学用的边际效用：

~~~math
\Delta U(\Delta\mathbf{b}\mid s)
=V\,\Delta Q(\Delta\mathbf{b}\mid s)
-\lambda\Delta C
-\mu\Delta R
~~~

若安全事件属于不可接受的硬约束，不能用很大的 \(V\) 抵消它，而应直接拒绝动作。

质量提升可能来自不同资源：更多推理 token、一次新的检索、一次独立测试或一次人工确认。控制器要估计的是当前状态下的边际收益，而不是把历史上高档位平均分提高的百分点直接套到每个请求。

简单问题的边际收益通常很快饱和；困难问题在第一次获得关键证据后可能突然改善。若新增 token 只产生重复表达而没有新增证据，$\Delta Q$ 应接近 0；若验证器尚未运行，最有价值的动作可能是验证而不是继续生成。

用一组假设值比较三个动作。本例把 $Q$ 取为任务成功概率，当前估计值为 0.70；效用价值系数 $V=100$（每增加 1.0 的成功概率对应 100 个效用点），成本惩罚系数 $\lambda=2$（每个归一化成本单位扣 2 点），风险惩罚系数 $\mu=10$（每个归一化风险单位扣 10 点）。这些都是本例中的标度，不是线上测量值。继续生成的成功概率估计升到 0.73，成本增加 1、风险增加 0.10，因此 $\Delta U=100\times(0.73-0.70)-2\times1-10\times0.10=3-2-1=0$。运行独立验证后，估计概率升到 0.78，成本增加 3、风险增加 0.05，效用为 $100\times0.08-2\times3-10\times0.05=1.5$。读取一份新资料后，概率估计升到 0.80，成本增加 4、风险增加 0.10，效用为 $100\times0.10-2\times4-10\times0.10=1$。按这组假设，验证的边际效用最高；若三个动作的效用都不大于 0，就停止或转入其他路径。估计必须来自按任务切片校准的历史数据，且硬性安全约束要先于效用排序。

## 18.5 从低到高预算（low 到 high）的升级路径

一种可审计的策略是分级升级。低预算（low）路径做有限生成和格式检查；中预算（medium）路径允许一次只读检索、自检或测试；高预算（high）路径允许更多候选、工具观察和独立验证。每一级都要定义资源上限、成功条件和失败状态。

升级不能只由“回答不够长”触发。更有意义的触发信号包括：引用缺失、候选结论冲突、结构化模式（schema）失败、测试返回新错误、证据过期、验证器返回 `unknown`（未知）或风险策略要求人工。

每次升级写入可复盘轨迹（trace）：

~~~text
decision_id: d-17
from: low
to: medium
reason: citation_missing
budget_added: tool=1, verify=1
deadline_remaining_ms: 3800
policy_revision: route-v4
~~~

这样系统可以解释成本变化，也能在升级比例突然上升时区分模型退化、数据漂移、工具故障和容量不足。

## 18.6 选择性预测：不知道时要能停

选择性预测的核心是允许模型在无法满足质量条件时拒答、请求更多信息或转交人工，而不是强行输出。设校准后的自动提交分数为 $p\in[0,1]$；自动提交阈值为 $\tau\in[0,1]$，只有满足：

~~~math
p\ge\tau
~~~

才允许进入自动路径。$p$ 必须先在与线上相似的任务集上校准，$\tau$ 按风险、业务损失和人工容量设定；不应把原始 logits（未归一化模型分数）或语气强度直接当作概率。将选择性预测与转人工纳入目标函数的代表性工作包括 [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) 和 [Predict Responsibly](https://arxiv.org/abs/1711.06664)；它们提供的是特定学习设定的方法与实验，不替代目标业务上的阈值校准。

在同一评估样本和任务定义下，令 \(\Pr(\mathrm{wrong})\)、\(\Pr(\mathrm{defer})\) 与 \(\Pr(\mathrm{extra\ compute})\) 分别表示错误、转交和额外计算事件的概率，均位于 \([0,1]\)；\(L_{\mathrm{wrong}}\)、\(L_{\mathrm{defer}}\) 与 \(L_{\mathrm{cost}}\) 是有限非负且使用同一效用单位的单次损失。将这些惩罚相加，可写成期望动作损失：

~~~math
L
=L_{\mathrm{wrong}}\Pr(\mathrm{wrong})
+L_{\mathrm{defer}}\Pr(\mathrm{defer})
+L_{\mathrm{cost}}\Pr(\mathrm{extra\ compute})
~~~

不同业务的损失权重不同。客服 FAQ 可以允许较多自动回答，支付和医疗建议要更重视错误动作；如果人工队列已满，盲目降低阈值会把容量问题伪装成效率提升。

拒答不是成功，也不是失败的简单标签。它是系统明确承认当前证据不足的一种可观察结果。评估时要分别报告覆盖率、选择性准确率、拒答率、人工接管和单位成功成本。

## 18.7 工具和验证器是预算升级的不同用途

当问题缺少事实时，工具调用可能比更多 token 更有价值；当候选已经产生但无法判断时，验证器比再次采样更有价值；当外部动作风险高时，人工确认比两倍推理预算更重要。

研究问答可以先从索引读取摘要；如果用户要求比较两篇论文的实验条件，控制器应升级为获取原文表格和版本，而不是只生成更长的解释。代码任务可以先读取错误堆栈，再运行最小测试；测试失败后，新增观察决定是否值得继续。工具返回空结果、超时或版本冲突时，状态必须保留为 `unknown`（未知）或 `stale`（过期），不能当作“没有证据所以问题不存在”。

每次工具和验证动作都要进入同一任务账本，记录输入版本、调用时间、结果状态、资源使用和是否产生副作用。这样才能区分“预算花在思考”与“预算花在获得证据”。

## 18.8 预算控制器的决策状态机

控制器可分为 LOW、VERIFY、MEDIUM、HIGH、ABSTAIN、HUMAN 和 TERMINAL。一个示意状态机是：

~~~text
LOW
  -> VERIFY       when evidence_gap or schema_failure
VERIFY
  -> MEDIUM       when new_evidence and budget_remains
MEDIUM
  -> HIGH         when new_evidence and budget_remains
VERIFY / MEDIUM / HIGH
  -> TERMINAL     when verified
  -> ABSTAIN      when no_progress
any nonterminal state
  -> ABSTAIN      when budget_exhausted
  -> HUMAN        when permission_unknown
~~~

VERIFY 是一个真实的状态，不是升级前的一句提示。它允许系统先检查已有证据，再决定额外计算是否值得。状态转移应由事件触发：新证据、验证器通过、重复调用、权限变化、截止时间接近和容量拒绝都要有明确处理。下方的教学状态机只实现其中一条精简路径；生产控制器还需为图中未编码的事件定义合同。

为了避免在阈值附近来回升降，可以使用迟滞：连续 m 次有新证据才升级，连续 n 次无进展才停止或人工。m、n 必须是正整数，并且每次状态变更写入原因和状态版本。

## 18.9 取消、租约与未知外部状态

动态预算不能只扣减一个 token 计数器。每次升级都应获得一个租约，包含剩余时间、reason token、工具轮数、验证额度、容量配额和允许的副作用。请求取消、用户撤回权限、工作区（workspace）改变或外部状态失效时，租约立即失效。

取消时要区分已提交和未提交状态。未提交的候选可以丢弃；已经开始的工具动作需要查询执行状态；已经提交的副作用不能靠删除模型文本回滚。若客户端超时，不能直接把工具标为失败后重新发送。

租约还防止重试时重复累加预算。恢复流程读取原有 decision_id 和租约版本，确认剩余额度后再继续；旧租约不能在新请求中自动复活。高风险未知状态应暂停并请求人工，而不是用更多推理 token 猜测。

推理与工具事件的协议顺序、工具结果和恢复合同见[第十九章](19-interleaved-thinking与工具协议.md)；本节聚焦控制器持有的任务状态、预算租约与取消边界。

## 18.10 容量：自适应升级会反过来改变系统

动态路由影响的不只是单请求质量，也影响队列和键值缓存（KV cache）。设 \(N\) 为非负整数并发请求数；当 \(N>0\) 时，\(q\in[0,1]\) 表示其中升级到深路径的比例。\(M_{\mathrm{base}}\) 是基础状态的单请求占用，\(M_{\mathrm{shallow}}\) 和 \(M_{\mathrm{deep}}\) 是浅、深路径相对基础状态的单请求增量；三者均为有限非负的 GiB/请求。\(M_{\mathrm{state}}\) 表示总状态需求；若 \(N=0\)，总需求为 0 且无需定义 \(q\)。当 \(N>0\) 时可作如下粗略估算：

~~~math
M_{\mathrm{state}}
\approx N\left(
M_{\mathrm{base}}
+qM_{\mathrm{deep}}
+(1-q)M_{\mathrm{shallow}}
\right)
~~~

这不是完整的图形处理器（GPU）容量模型，还会受到批处理（batch）、KV 复用、模型并行和序列长度影响，但能说明升级比例上升会增加状态持有和尾延迟。

例如，假设 $N=100$ 个并发请求，$M_{\mathrm{base}}=0.20$ GiB/请求，浅路径增量 $M_{\mathrm{shallow}}=0.40$ GiB/请求，深路径增量 $M_{\mathrm{deep}}=1.50$ GiB/请求。升级比例 $q$ 从 0.20 增至 0.70 时，前者占用约为 $100\times(0.20+0.20\times1.50+0.80\times0.40)=82$ GiB；后者约为 $100\times(0.20+0.70\times1.50+0.30\times0.40)=137$ GiB，增加 55 GiB。数值是为演算构造的容量快照，不是特定 GPU 或模型的实测；它说明路由比例会改变内存需求，不能据此直接推算完整集群容量。

生产系统可以设置全局和租户级 high 配额、最大升级深度、推理 token budget、熔断、异步队列和容量拒绝。容量不足时要明确返回降级、延迟或人工状态；不能静默减少验证后仍声称使用了 high 路径。

## 18.11 公平性：谁被动态路由器判为难题

按历史失败率分配预算可能把新领域、新用户、低资源语言和不熟悉的表达方式系统性判为难题，也可能因为样本太少把真正危险的请求判为简单。控制器应按语言、领域、任务类型、输入长度、租户和风险分组校准。

公平性不意味着所有请求消耗相同资源，而是要确认资源差异来自任务和风险证据，而不是无关的用户属性。若某个分组的漏升级率显著更高，应先修复信号和数据，再谈节省成本。

预算升级和访问权限必须分离。high 路径可以允许更多只读验证，但不能因此读取更多租户数据、绕过人工确认或写入生产。风险策略可以要求某类动作永远人工确认，即使其难度估计很低。

## 18.12 如何评估自适应思考策略

离线评估至少比较三条基线：`always-low`、`always-high` 和自适应策略。任务集、模型版本、工具允许列表、成功定义、时间上限和总成本口径要对齐。自适应策略若使用了额外的隐性人工或更高总预算，不能只报告它的准确率。

报告以下指标：总体和切片成功率、覆盖率、选择性准确率、升级率、漏升级率、无效升级率、平均延迟和第95/99百分位延迟（P95/P99）、模型/工具/验证成本、人工接管、重复副作用和安全事件。还要记录触发原因，否则无法知道升级是因为证据缺失、模型版本回归还是容量过载。

如果自适应策略接近 `always-high` 的质量、平均成本接近 `always-low`，且高风险切片没有回归，它才体现出系统价值。若升级比例接近 100%，先检查初始预算、分类器和工具质量；若升级比例很低但高风险漏升级增加，应立即停止自动发布。

必须防止离线数据泄漏。路由器只能使用决策时已经可见的状态，不能在看到最终答案后用它选择预算。回放时要保存每次决策的信号快照（signal snapshot）和预算账本（budget ledger）。

## 18.13 一个研究问答的端到端案例

用户先问某论文的结论，任务可以走 low：读取已索引的摘要，要求答案包含来源。若来源缺失，进入 VERIFY，检查索引版本和引用覆盖；如果用户随后要求与另一篇论文比较实验设置，升级为 medium，读取两篇原文的实验表。

如果两篇论文的样本、指标或时间窗口不同，控制器不能继续把两个数字直接比较，而应把差异写入产物，并要求更高预算的核验或返回“不足以比较”。如果工具返回过期页面，状态是 `stale`（过期）；如果请求已经发送但没有响应，状态是 `unknown`（未知）。两者都不是普通的“检索没有结果”。

评估这个案例时，不能只看最终答案。要检查引用是否支持结论、版本是否正确、升级是否在证据缺口出现后触发、是否出现无效重复搜索、第95百分位延迟是否超过服务目标（SLO），以及恶意网页中的指令是否被当作系统命令。

## 18.14 常见失败模式

第一，把输入长度直接当作难度。长文本可能只是重复，短文本可能包含高风险写操作。

第二，把模型自报置信度当作自动提交许可。高推理档位可能只产生更长的错误解释。

第三，升级没有硬上限。验证器连续失败时，控制器在档位之间循环，最终耗尽队列和工具资源。

第四，升级只增加输出长度。若没有新的证据、工具观察或独立验证，额外 token 的边际收益应被视为可疑。

第五，降级没有安全边界。高风险任务在 high 失败后，不能自动退回未经验证的 direct 答案。

第六，把未知状态当作失败并重试。外部动作可能已经发生，恢复需要查询和幂等。

第七，只看平均成本。少量高风险任务的漏升级和 P99 退化可能被大量简单请求掩盖。

## 18.15 一个可运行的动态预算审计器

下面的标准库示例不调用模型或网络，用固定事件模拟一个预算控制器。它检查任务状态、预算扣减、升级原因、无进展停止、未经授权工具请求拦截和高风险人工确认路径。示例的目的不是预测真实难度，而是展示如何把动态策略变成可回放的状态机。

代码状态：已验证（2026-11-13，CPython 3.12.9，Python 标准库）。

~~~python
from dataclasses import dataclass, field


TASKS = {"research", "code", "payment"}
RISKS = {"low", "high"}
EVENTS = {"evidence_gap", "schema_failure", "new_evidence", "verified", "no_progress", "permission_unknown", "budget_exhausted"}
STATES = {"LOW", "VERIFY", "MEDIUM", "HIGH", "ABSTAIN", "HUMAN", "TERMINAL"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass
class Budget:
    reason: int
    tool: int
    verify: int

    def __post_init__(self):
        for name in ("reason", "tool", "verify"):
            value = getattr(self, name)
            require(type(value) is int and value >= 0, f"{name} must be a non-negative integer")

    def spend(self, name, amount):
        require(type(name) is str and name in {"reason", "tool", "verify"}, "unknown budget bucket")
        require(type(amount) is int and amount > 0, "amount must be a positive integer")
        remaining = getattr(self, name)
        require(amount <= remaining, f"{name} budget exhausted")
        setattr(self, name, remaining - amount)


@dataclass
class Controller:
    task: str
    risk: str
    budget: Budget
    state: str = "LOW"
    trace: list = field(default_factory=list)

    def __post_init__(self):
        require(type(self.task) is str and self.task in TASKS, "unknown task")
        require(type(self.risk) is str and self.risk in RISKS, "unknown risk")
        require(isinstance(self.budget, Budget), "budget must be a Budget")
        require(type(self.state) is str and self.state in STATES, "unknown state")
        self.trace.append(f"state:{self.state}")

    def step(self, event):
        require(type(event) is str and event in EVENTS, "unknown event")
        if self.state in {"TERMINAL", "HUMAN", "ABSTAIN"}:
            raise ValueError("controller is already terminal")
        if event == "permission_unknown":
            self.state = "HUMAN"
            self.trace.append("human:permission_unknown")
            return self.state
        if event == "budget_exhausted":
            self.state = "ABSTAIN"
            self.trace.append("abstain:budget_exhausted")
            return self.state
        if self.state == "LOW" and event in {"evidence_gap", "schema_failure"}:
            self.budget.spend("verify", 1)
            self.state = "VERIFY"
            self.trace.append("upgrade:LOW->VERIFY")
        elif self.state == "VERIFY" and event == "new_evidence":
            self.budget.spend("reason", 1)
            self.state = "MEDIUM"
            self.trace.append("upgrade:VERIFY->MEDIUM")
        elif self.state == "MEDIUM" and event == "new_evidence":
            self.budget.spend("reason", 1)
            self.state = "HIGH"
            self.trace.append("upgrade:MEDIUM->HIGH")
        elif self.state in {"VERIFY", "MEDIUM", "HIGH"} and event == "verified":
            self.state = "TERMINAL"
            self.trace.append("terminal:verified")
        elif self.state in {"VERIFY", "MEDIUM", "HIGH"} and event == "no_progress":
            self.state = "ABSTAIN"
            self.trace.append("abstain:no_progress")
        else:
            raise ValueError(f"event {event} is invalid in state {self.state}")
        return self.state

    def request_tool(self, authorized, human_approved=False):
        require(type(authorized) is bool, "authorized must be a boolean")
        require(type(human_approved) is bool, "human_approved must be a boolean")
        if self.state in {"TERMINAL", "HUMAN", "ABSTAIN"}:
            raise ValueError("controller is already terminal")
        if not authorized:
            self.trace.append("tool_blocked:unauthorized")
            return False
        if self.risk == "high" and not human_approved:
            self.trace.append("tool_blocked:human_approval_required")
            return False
        self.budget.spend("tool", 1)
        self.trace.append("tool:authorized")
        return True


research = Controller("research", "low", Budget(reason=2, tool=1, verify=1))
research.step("evidence_gap")
research.step("new_evidence")
research.step("new_evidence")
research.step("verified")

payment = Controller("payment", "high", Budget(reason=2, tool=1, verify=1))
payment.request_tool(authorized=False)
payment.request_tool(authorized=True)
payment.step("permission_unknown")

stalled = Controller("code", "low", Budget(reason=1, tool=0, verify=1))
stalled.step("evidence_gap")
stalled.step("no_progress")

print(research.state, research.trace)
print(payment.state, payment.budget.tool, payment.trace)
print(stalled.state, stalled.trace)
~~~

预期输出为：

~~~text
TERMINAL ['state:LOW', 'upgrade:LOW->VERIFY', 'upgrade:VERIFY->MEDIUM', 'upgrade:MEDIUM->HIGH', 'terminal:verified']
HUMAN 1 ['state:LOW', 'tool_blocked:unauthorized', 'tool_blocked:human_approval_required', 'human:permission_unknown']
ABSTAIN ['state:LOW', 'upgrade:LOW->VERIFY', 'abstain:no_progress']
~~~

这个控制器只模拟状态和资源扣减，不判断答案是否真的正确，也没有容量或权限服务。`request_tool` 接收由外部策略给出的授权结果；授权与人工批准是两个独立布尔值，不能由模型输出代替。该方法只演示门控与预算扣减，不会执行真实工具。它刻意把 `permission_unknown` 事件直接送入人工状态，把 `no_progress` 事件变成 `ABSTAIN` 状态，避免用更多 token 掩盖证据缺失。

边界测试包括：负预算、非整数预算、未知或不可哈希的任务、风险、状态、事件值和预算桶名称、重复终态迁移、验证预算耗尽、未知权限、未经授权工具调用、高风险但无人批准、工具预算耗尽和非法授权值。未经授权或未获高风险人工批准时，工具预算不应扣减。未知外部状态不能通过增加预算自动变成已成功。

## 18.16 从审计器回到生产系统

生产控制器需要把模型、工具、验证器、容量和策略服务的事件汇合到同一 `task_id`。每次决策至少保存 `decision_id`、状态版本、信号快照、预算前后值、动作、`policy_revision`、停止原因和外部状态。这样才能回放“为什么升级”以及“升级后是否真的带来新证据”。

还要为模型和工具版本变化建立基准任务组（golden tasks）。新模型可能改变自报置信度，新检索器可能改变证据覆盖，新验证器可能改变未知结果比例；这些变化都可能让旧阈值失效。上线前应做影子流量（shadow）或受限金丝雀发布（canary），逐步观察升级率、漏升级率、第95百分位延迟和安全事件。

冷启动时没有可靠历史数据，不能把未知当成简单。可以使用保守初始预算、少量影子抽样（shadow sampling）和人工标注建立校准集；数据分布变化后重新估计边际收益和容量，而不是永久沿用旧阈值。

## 18.17 练习

**练习一：升级条件。** 为客服问答、代码修复和支付审核分别定义 `low`、`verify`、`high`、`human` 状态的进入条件，并写出每一级的硬资源上限。

**练习二：信号校准。** 设计一个实验，比较输入长度、模型自报置信度、引用覆盖和验证器失败对实际失败率的预测能力，按语言和风险切片。

**练习三：容量保护。** 假设升级比例从 10% 突然升到 70%，列出可能的模型、工具、数据和流量原因，并设计一个不会静默越权的降级路径。

**练习四：选择性预测。** 给定自动提交阈值和人工容量，画出覆盖率—风险曲线（coverage–risk curve），说明什么时候增加验证比降低阈值更合理。

## 18.18 资料边界与本章结论

选择性预测的代表性方法包括 [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) 与 [Predict Responsibly](https://arxiv.org/abs/1711.06664)；[FrugalGPT](https://arxiv.org/abs/2305.05176) 讨论特定设置下的模型级联，[Adaptive Computation Time](https://arxiv.org/abs/1603.08983) 则研究循环神经网络按序列步数分配计算。它们支持各自论文设置中的方法讨论，不能直接证明某个闭源模型的隐含置信度、自动升级阈值或内部路由。任务状态、预算账本、租约、容量保护和权限分离是工程设计，需要在目标系统上通过回放、压测和故障演练验证。

自适应思考的核心是把“多想一会儿”变成一个有状态、可校准、受预算和风险约束的控制环。难题应在证据缺失或验证器失败时得到新的信息和验证，简单题应及时结束；未知状态应保持未知，高风险动作应进入独立授权或人工路径。真正的成功不是让平均答案更长，而是在固定资源下提高可验证质量，同时控制尾延迟、成本、容量和公平性。

多模态视觉推理与 Qwen3-VL 的工具奖励案例见[第二十一册第九十二章](../../book-21-transformer-architecture-evolution/chapters/92-qwen3-vl-interleaved-mrope-deepstack与视频时间戳.md)；多模态训练阶段和数据设计见[第五册第十二章](../../book-05-llm-training/chapters/12-多模态训练.md)。本章聚焦跨任务的动态预算控制，不重复视觉模型结构和训练路线。


## 18.19 服务提供方 API：动态控制的接口合同示例

前面的状态机由应用控制器依据证据、预算和策略做决定；模型服务提供方的 API（application programming interface，应用程序接口）只提供模型侧的配置入口。字段名称相似，不代表额度单位、默认值、跨轮持久化或内部实现相同。以下例子用于说明如何把 API 配置纳入轨迹；档位细节见[第十四章](14-thinking-levels.md)，跨轮状态管理见[第十五章](15-preserve-thinking与推理状态.md)。

### Anthropic：投入档位（effort）不是精确 token 配额

截至 2026-11-13，Claude Opus 5.5 的官方模型页将 `adaptive thinking` 列为始终开启，并通过 `output_config.effort` 调整行为；旧式 `thinking.type=disabled` 和手工 `budget_tokens` 不属于该模型的有效控制方式。同步请求的 `max_tokens` 仍是 thinking 与可见输出共享的响应上限。这个合同说明 effort 标签不是预分配的固定 token 数，应用仍要单独设置整项任务的成本和时间上限，并用实际 usage 校准观察。字段行为见 [Opus 5.5 模型页](https://platform.claude.com/docs/en/models/opus-5-5/overview) 与[Thinking 文档](https://platform.claude.com/docs/en/build-with-claude/thinking)；它们描述公开 API，不揭示模型内部调度方式。

### OpenAI：`configuration_update` 是会话历史中的预算控制项

GPT-6 的 Responses API 标准单智能体会话可用一条历史项（item）更新后续推理 effort，例如：

~~~json
{
  "type": "configuration_update",
  "reasoning": {"effort": "high"}
}
~~~

更新持续到后续 item 再次覆盖，因此客户端重放完整历史时要保留它原来的位置。官方指南还规定它不能与自动上下文压缩（compaction）或自动截断（truncation）并用；独立 `/responses/compact` 不接受带此更新的历史，显式压缩后若仍需新 effort，须在下一条用户消息前重新放置更新。响应中的 `reasoning.effort` 仍反映请求级设置，不能单独用来还原更新后的有效值。应由轨迹按历史 item 重建实际生效的档位，并同时保存请求级配置与实际 usage。合同见 [OpenAI Reasoning 指南](https://developers.openai.com/api/docs/guides/reasoning) 和 [GPT-6 Sol 模型页](https://developers.openai.com/api/docs/models/gpt-6-sol)。

把这些接口放回控制环，假设一次研究任务以中档预算开始。验证器返回未知后，控制器先比较继续生成、再检索和人工确认的预期边际效用；若选择提高 GPT-6 的 effort，就把配置更新作为新的历史 item 保存，再发起后续响应。模型返回后，应用比较实际 token 用量与原估计，并运行独立验证器：得到新证据才进入下一状态，连续无进展则停止或转人工。此处的更新只改变模型响应配置，不执行检索、不授予读写权限，也不证明结果变好。

若运行时为了上下文限制压缩会话，不能只保留可读摘要而丢掉配置更新的状态；必须按官方合同恢复有效 effort，并把压缩前后的历史版本和配置来源留在轨迹中。Anthropic 的 `effort` 也不应被套用成同一种会话 item：控制器需按每个新请求使用对应模型文档的字段，再依据响应 usage 校准成本。两家的请求都要在模型调用之后接独立的任务验证与停止判断，才能把“接口配置已应用”与“任务已经成功”分开。

这两个例子分别展示“请求内行为档位”和“历史中持续生效的会话更新”。应用控制器仍要定义什么时候升级、用什么证据判定升级有效、何时停止，以及哪些动作必须授权；API 接受配置或返回更高 `usage` 都不等于任务成功。用于复盘的记录至少包含 `model`/`snapshot`、请求级 `effort`、历史更新及其位置、`usage`、工具与验证结果、成本和停止原因。服务提供方文档会更新，部署时应按实际模型与 API 版本重新核对，不把上面的合同推广成统一标准。

跨服务评测也不能把同名的 `high` 当作同一预算臂。先固定同一任务集、成功谓词、外部时间/成本上限和工具权限，再按各自 API 合同设置档位；记录实际推理用量、上下文压缩、工具、验证和重试成本，分别报告成功率与单位成功成本。若请求跨越多轮，配置历史和各轮 usage 都要归入同一个 task_id；不然可能把会话更新误记成模型能力差异，或漏算重试、工具和人工成本。

实际部署时可把状态拆成两个相连的记录：应用层保存任务目标、总预算、权限、外部观察和验证结论；provider 层保存模型 ID、请求字段、会话历史与响应 usage。每次调用前由控制器先判断是否允许增加推理，再按 provider 合同编码；每次响应后由应用更新总账本并检查停止条件。这样既不会把 API 的会话续接当成完整任务状态，也不会让一个更高的请求档位绕开工具授权或人工审批。

## 参考资料

- Geifman and El-Yaniv, [SelectiveNet: A Deep Neural Network with an Integrated Reject Option](https://proceedings.mlr.press/v97/geifman19a.html).
- Madras et al., [Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer](https://arxiv.org/abs/1711.06664).
- Chen et al., [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176).
- Graves, [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983).
- OpenAI, [Reasoning Models Guide](https://developers.openai.com/api/docs/guides/reasoning).
- Anthropic, [Claude Opus 5.5 model page](https://platform.claude.com/docs/en/models/opus-5-5/overview) and [Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking).
