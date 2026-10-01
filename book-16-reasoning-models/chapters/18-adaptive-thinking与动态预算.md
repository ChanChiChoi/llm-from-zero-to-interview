# 第 18 章 Adaptive Thinking：把推理预算变成可校准的控制环

## 18.1 为什么所有请求都用最高预算并不合理

把每个请求都交给最高思考档位，实施上很简单，却把三类任务混在了一起。简单问题可能已经能够可靠回答，额外 token 只增加费用和排队；困难问题可能真正缺少文档、工具观察或人工授权，延长内部生成并不会提供这些信息；高风险问题即使模型愿意继续思考，也不能因此获得更多权限。

Adaptive thinking 的目标不是让系统猜测模型“聪不聪明”，而是让控制器根据可观察进展决定下一步资源。它可以增加推理 token、采样候选、调用检索、运行测试、请求人工或直接停止。每次升级都要有触发原因，每次停止都要有可解释状态，预算和权限还必须分开管理。

可以把一次请求看成一个有限资源决策问题：在当前证据、剩余时间和风险下，继续花费多少计算，是否能带来足够的任务价值？这和单纯把 max_output_tokens 调大不同，因为新增资源可能被分配给工具、验证、人工或恢复。

## 18.2 控制器的状态

设第 t 个决策时刻的任务状态为：

~~~math
s_t=(x,o_t,a_t,b_t,d_t,r_t,c_t)
~~~

x 是原始任务，o_t 是截至当前已确认的观察和证据，a_t 是已有 artifact 或候选，b_t 是剩余预算向量，d_t 是剩余 deadline，r_t 是风险状态，c_t 是服务容量状态。状态不应只保存模型的自然语言自报信心，还要保存 verifier、工具和权限的事实。

预算可以表示成：

~~~math
\mathbf{b}_t
=(b_{\mathrm{reason}},b_{\mathrm{tool}},b_{\mathrm{verify}},b_{\mathrm{time}},b_{\mathrm{side}})
~~~

每个分量有自己的单位，必须是有限非负量；b_side=0 表示不允许外部副作用。不能把 token、毫秒和写操作次数直接相加。权限属于策略状态，不因预算向量变大而自动变化。

控制器的动作集合可以写成：

~~~text
keep -> upgrade_reason -> call_tool -> verify
     -> ask_clarification -> abstain -> human -> terminal
~~~

这些动作不是模型文本，而是 runtime 可以审计和计费的状态转换。call_tool 需要独立的 allowlist 和授权；human 可能增加等待时间，但不是“模型失败”的同义词；abstain 表示当前证据不足以满足成功契约。

## 18.3 难度、风险与可验证性是三个维度

输入长度、任务难度、任务风险和可验证性经常被错误地压成一个分数。长日志可能只需抽取一个字段，短问题可能要求复杂证明；数学题可能难但风险低，支付确认可能简单却风险高；工具返回空结果可能不是难度变大，而是证据状态未知。

可用信号包括：任务类型、历史同类失败率、引用覆盖、候选分歧、verifier 结果、工具错误、剩余 deadline、数据新鲜度和容量水位。模型自报“我很有把握”最多是一个候选信号，不能直接授权危险动作。

对难度分数 d，可以在固定任务集上估计失败概率：

~~~math
\hat p_{\mathrm{fail}}(d)
=P(\mathrm{fail}\mid d)
~~~

这只是校准关系，不是模型内部真实难度。校准必须按任务类型、语言、领域、输入长度和风险分组；未覆盖的组不能因为缺少历史失败就默认为简单。

## 18.4 边际收益：下一单位预算换来什么

设当前状态为 s，继续投入预算向量 Delta b 后，任务质量提升为 Delta Q(Delta b | s)，成本和风险增量分别为 Delta C、Delta R，则可以使用教学上的效用：

~~~math
\Delta U(\Delta\mathbf{b}\mid s)
=V\,\Delta Q(\Delta\mathbf{b}\mid s)
-\lambda\Delta C
-\mu\Delta R
~~~

V 是任务成功的业务价值，Q 必须由独立评估定义，C 与 R 使用可比较的成本和风险尺度，lambda、mu 大于等于 0。若安全事件属于不可接受的硬约束，不能用很大的 V 抵消它，而应直接拒绝动作。

质量提升可能来自不同资源：更多 reasoning token、一次新的检索、一次独立测试或一次人工确认。控制器要估计的是当前状态下的边际收益，而不是把历史上 high 档平均分提高的百分点直接套到每个请求。

简单问题的边际收益通常很快饱和；困难问题在第一次获得关键证据后可能突然改善。若新增 token 只产生重复表达而没有新增证据，Delta Q 应接近 0；若 verifier 尚未运行，最有价值的动作可能是验证而不是继续生成。

## 18.5 从 low 到 high 的升级路径

一种可审计的策略是分级升级。low 路径做有限生成和格式检查；medium 路径允许一次只读检索、自检或测试；high 路径允许更多候选、工具观察和独立验证。每一级都要定义资源上限、成功条件和失败状态。

升级不能只由“回答不够长”触发。更有意义的触发信号包括：引用缺失、候选结论冲突、结构化 schema 失败、测试返回新错误、证据过期、verifier 返回 unknown 或风险策略要求人工。

每次升级写入 trace：

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

选择性预测的核心是允许模型在无法满足质量条件时拒答、请求更多信息或转交人工，而不是强行输出。设校准后的自动提交分数为 `p`，其中 `p\in[0,1]`；自动提交阈值为 `\tau\in[0,1]`，只有满足：

~~~math
p\ge\tau
~~~

才允许进入自动路径。`p` 必须先在与线上相似的任务集上校准，`\tau` 按风险、业务损失和人工容量设定；不应把原始 logits 或语气强度直接当作概率。

可以把动作损失写成：

~~~math
L
=L_{\mathrm{wrong}}\Pr(\mathrm{wrong})
+L_{\mathrm{defer}}\Pr(\mathrm{defer})
+L_{\mathrm{cost}}\Pr(\mathrm{extra\ compute})
~~~

不同业务的损失权重不同。客服 FAQ 可以允许较多自动回答，支付和医疗建议要更重视错误动作；如果人工队列已满，盲目降低阈值会把容量问题伪装成效率提升。

拒答不是成功，也不是失败的简单标签。它是系统明确承认当前证据不足的一种可观察结果。评估时要分别报告覆盖率、选择性准确率、拒答率、人工接管和单位成功成本。

## 18.7 工具和 verifier 是预算升级的不同用途

当问题缺少事实时，工具调用可能比更多 token 更有价值；当候选已经产生但无法判断时，verifier 比再次采样更有价值；当外部动作风险高时，人工确认比两倍 reasoning budget 更重要。

研究问答可以先从索引读取摘要；如果用户要求比较两篇论文的实验条件，控制器应升级为获取原文表格和版本，而不是只生成更长的解释。代码任务可以先读取错误堆栈，再运行最小测试；测试失败后，新增观察决定是否值得继续。工具返回空结果、超时或版本冲突时，状态必须保留为 unknown 或 stale，不能当作“没有证据所以问题不存在”。

每次工具和 verifier 动作都要进入同一任务账本，记录输入版本、调用时间、结果状态、资源使用和是否产生副作用。这样才能区分“预算花在思考”与“预算花在获得证据”。

## 18.8 预算控制器的决策状态机

控制器可分为 LOW、VERIFY、MEDIUM、HIGH、ABSTAIN、HUMAN 和 TERMINAL。一个示意状态机是：

~~~text
LOW
  -> VERIFY       when evidence_gap or schema_failure
  -> MEDIUM       when verifier_conflict and budget_remains
  -> HUMAN        when side_effect_risk
VERIFY
  -> TERMINAL     when verified
  -> MEDIUM       when new_evidence and budget_remains
  -> ABSTAIN      when unknown or no_progress
MEDIUM
  -> HIGH         when calibrated_gain_positive
  -> ABSTAIN      when no_new_evidence
HIGH
  -> TERMINAL     when verified
  -> ABSTAIN      when budget_exhausted
  -> HUMAN        when permission_changed
~~~

VERIFY 是一个真实的状态，不是升级前的一句提示。它允许系统先检查已有证据，再决定额外计算是否值得。状态转移应由事件触发：新证据、verifier 通过、重复调用、权限变化、deadline 接近和容量拒绝都要有明确处理。

为了避免在阈值附近来回升降，可以使用迟滞：连续 m 次有新证据才升级，连续 n 次无进展才停止或人工。m、n 必须是正整数，并且每次状态变更写入原因和状态版本。

## 18.9 取消、租约与未知外部状态

动态预算不能只扣减一个 token 计数器。每次升级都应获得一个租约，包含剩余时间、reason token、工具轮数、验证额度、容量配额和允许的副作用。请求取消、用户撤回权限、workspace 改变或外部状态失效时，租约立即失效。

取消时要区分已提交和未提交状态。未提交的候选可以丢弃；已经开始的工具动作需要查询执行状态；已经提交的副作用不能靠删除模型文本回滚。若客户端超时，不能直接把工具标为失败后重新发送。

租约还防止重试时重复累加预算。恢复流程读取原有 decision_id 和租约版本，确认剩余额度后再继续；旧租约不能在新请求中自动复活。高风险未知状态应暂停并请求人工，而不是用更多 reasoning token 猜测。

## 18.10 容量：自适应升级会反过来改变系统

动态路由影响的不只是单请求质量，也影响队列和 KV cache。设并发请求数为 N，其中升级到深路径的比例为 q，浅路径和深路径每个请求的状态占用分别为 M_shallow、M_deep，基础占用为 M_base，则粗略状态需求为：

~~~math
M_{\mathrm{state}}
\approx N\left(
M_{\mathrm{base}}
+qM_{\mathrm{deep}}
+(1-q)M_{\mathrm{shallow}}
\right)
~~~

这里 N 是非负整数，q 属于 [0,1]，各个内存量是有限非负数。这不是完整的 GPU 容量模型，还会受到 batch、KV 复用、模型并行和序列长度影响，但能说明升级比例上升会增加状态持有和尾延迟。

生产系统可以设置全局和租户级 high 配额、最大升级深度、reasoning token budget、熔断、异步队列和容量拒绝。容量不足时要明确返回降级、延迟或人工状态；不能静默减少验证后仍声称使用了 high 路径。

## 18.11 公平性：谁被动态路由器判为难题

按历史失败率分配预算可能把新领域、新用户、低资源语言和不熟悉的表达方式系统性判为难题，也可能因为样本太少把真正危险的请求判为简单。控制器应按语言、领域、任务类型、输入长度、租户和风险分组校准。

公平性不意味着所有请求消耗相同资源，而是要确认资源差异来自任务和风险证据，而不是无关的用户属性。若某个分组的漏升级率显著更高，应先修复信号和数据，再谈节省成本。

预算升级和访问权限必须分离。high 路径可以允许更多只读验证，但不能因此读取更多租户数据、绕过人工确认或写入生产。风险策略可以要求某类动作永远人工确认，即使其难度估计很低。

## 18.12 如何评估 adaptive thinking

离线评估至少比较三条基线：always-low、always-high 和 adaptive。任务集、模型版本、工具 allowlist、成功定义、时间上限和总成本口径要对齐。若 adaptive 使用了额外的隐性人工或更高总预算，不能只报告它的准确率。

报告以下指标：总体和切片成功率、覆盖率、选择性准确率、升级率、漏升级率、无效升级率、平均和 P95/P99 延迟、模型/工具/验证成本、人工接管、重复副作用和安全事件。还要记录触发原因，否则无法知道升级是因为证据缺失、模型版本回归还是容量过载。

如果 adaptive 接近 always-high 的质量，平均成本接近 always-low，且高风险切片没有回归，它才体现出系统价值。若升级比例接近 100%，先检查初始预算、分类器和工具质量；若升级比例很低但高风险漏升级增加，应立即停止自动发布。

必须防止离线数据泄漏。路由器只能使用决策时已经可见的状态，不能在看到最终答案后用它选择预算。回放时要保存每次决策的 signal snapshot 和 budget ledger。

## 18.13 一个研究问答的端到端案例

用户先问某论文的结论，任务可以走 low：读取已索引的摘要，要求答案包含来源。若来源缺失，进入 VERIFY，检查索引版本和引用覆盖；如果用户随后要求与另一篇论文比较实验设置，升级为 medium，读取两篇原文的实验表。

如果两篇论文的样本、指标或时间窗口不同，控制器不能继续把两个数字直接比较，而应把差异写入 artifact，并要求更高预算的核验或返回“不足以比较”。如果工具返回过期页面，状态是 stale；如果请求已经发送但没有响应，状态是 unknown。两者都不是普通的“检索没有结果”。

评估这个案例时，不能只看最终答案。要检查引用是否支持 claim、版本是否正确、升级是否在证据缺口出现后触发、是否出现无效重复搜索、P95 是否超过 SLO，以及恶意网页中的指令是否被当作系统命令。

## 18.14 常见失败模式

第一，把输入长度直接当作难度。长文本可能只是重复，短文本可能包含高风险写操作。

第二，把模型自报置信度当作自动提交许可。高 effort 可能只产生更长的错误解释。

第三，升级没有硬上限。verifier 连续失败时，控制器在档位之间循环，最终耗尽队列和工具资源。

第四，升级只增加输出长度。若没有新的证据、工具观察或独立验证，额外 token 的边际收益应被视为可疑。

第五，降级没有安全边界。高风险任务在 high 失败后，不能自动退回未经验证的 direct 答案。

第六，把 unknown 当作失败并重试。外部动作可能已经发生，恢复需要查询和幂等。

第七，只看平均成本。少量高风险任务的漏升级和 P99 退化可能被大量简单请求掩盖。

## 18.15 一个可运行的动态预算审计器

下面的标准库示例不调用模型或网络，用固定事件模拟一个预算控制器。它检查任务状态、预算扣减、升级原因、无进展停止、工具未知状态和高风险人工路径。示例的目的不是预测真实难度，而是展示如何把动态策略变成可回放的状态机。

~~~python
from dataclasses import dataclass, field


TASKS = {"research", "code", "payment"}
RISKS = {"low", "high"}
EVENTS = {"evidence_gap", "new_evidence", "verified", "no_progress", "permission_unknown", "budget_exhausted"}


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
        require(name in {"reason", "tool", "verify"}, "unknown budget bucket")
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
        require(self.task in TASKS, "unknown task")
        require(self.risk in RISKS, "unknown risk")
        require(self.state in {"LOW", "VERIFY", "MEDIUM", "HIGH", "ABSTAIN", "HUMAN", "TERMINAL"}, "unknown state")
        self.trace.append(f"state:{self.state}")

    def step(self, event):
        require(event in EVENTS, "unknown event")
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
        if self.state == "LOW" and event == "evidence_gap":
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


research = Controller("research", "low", Budget(reason=2, tool=1, verify=1))
research.step("evidence_gap")
research.step("new_evidence")
research.step("verified")

payment = Controller("payment", "high", Budget(reason=2, tool=0, verify=1))
payment.step("permission_unknown")

stalled = Controller("code", "low", Budget(reason=1, tool=0, verify=1))
stalled.step("evidence_gap")
stalled.step("no_progress")

print(research.state, research.trace)
print(payment.state, payment.trace)
print(stalled.state, stalled.trace)
~~~

预期输出为：

~~~text
TERMINAL ['state:LOW', 'upgrade:LOW->VERIFY', 'upgrade:VERIFY->MEDIUM', 'terminal:verified']
HUMAN ['state:LOW', 'human:permission_unknown']
ABSTAIN ['state:LOW', 'upgrade:LOW->VERIFY', 'abstain:no_progress']
~~~

这个控制器只模拟状态和资源扣减，不判断答案是否真的正确，也没有容量服务和权限服务。它刻意把 permission_unknown 直接送入人工状态，把 no_progress 变成 ABSTAIN，避免用更多 token 掩盖证据缺失。

边界测试应包括：负预算、非整数预算、未知任务、未知风险、未知事件、重复终态迁移、验证预算耗尽、high 风险无权限和未授权工具调用。未知外部状态不能通过增加预算自动变成已成功。

## 18.16 从审计器回到生产系统

生产控制器需要把模型、工具、verifier、容量和策略服务的事件汇合到同一 task_id。每次决策至少保存 decision_id、状态版本、signal snapshot、预算前后值、动作、policy revision、停止原因和外部状态。这样才能回放“为什么升级”以及“升级后是否真的带来新证据”。

还要为模型和工具版本变化建立 golden tasks。新模型可能改变自报置信度，新检索器可能改变证据覆盖，新 verifier 可能改变 unknown 比例；这些变化都可能让旧阈值失效。上线前应做 shadow 或受限 canary，逐步观察升级率、漏升级率、P95 和安全事件。

冷启动时没有可靠历史数据，不能把未知当成简单。可以使用保守初始预算、少量 shadow 采样和人工标注建立校准集；数据分布变化后重新估计边际收益和容量，而不是永久沿用旧阈值。

## 18.17 练习

**练习一：升级条件。** 为客服问答、代码修复和支付审核分别定义 low、verify、high、human 的进入条件，并写出每一级的硬资源上限。

**练习二：信号校准。** 设计一个实验，比较输入长度、模型自报置信度、引用覆盖和 verifier 失败对实际失败率的预测能力，按语言和风险切片。

**练习三：容量保护。** 假设升级比例从 10% 突然升到 70%，列出可能的模型、工具、数据和流量原因，并设计一个不会静默越权的降级路径。

**练习四：选择性预测。** 给定自动提交阈值和人工容量，画出 coverage—risk 曲线，说明什么时候增加验证比降低阈值更合理。

## 18.18 资料边界与本章结论

SelectiveNet、Learning to Defer、FrugalGPT 和 Adaptive Computation Time 等研究支持选择性预测、转人工、模型级联和动态计算分配的通用讨论；它们不能直接证明某个闭源模型的 hidden confidence、自动升级阈值或内部 routing。任务状态、预算账本、租约、容量保护和权限分离是工程设计，需要在目标系统上通过回放、压测和故障演练验证。

Adaptive thinking 的核心是把“多想一会儿”变成一个有状态、可校准、受预算和风险约束的控制环。难题应在证据缺失或 verifier 失败时得到新的信息和验证，简单题应及时结束；未知状态应保持未知，高风险动作应进入独立授权或人工路径。真正的成功不是让平均答案更长，而是在固定资源下提高可验证质量，同时控制尾延迟、成本、容量和公平性。

## 18.19 Claude Sonnet 5：从 effort 到 System Card 证据

Sonnet 5 把 adaptive thinking 和 `output_config.effort` 放在请求协议中。`low`、`medium`、`high`、`xhigh`、`max` 是行为控制信号，不是预先分配好的固定 thinking token 数；`max_tokens` 才是单次响应对 thinking、工具调用和可见文本共享的硬上限。因而一次实验至少要记录 effort、`max_tokens`、工具轮数、thinking/output token、超时和任务级预算，不能只记录“使用了 max”。

这个区分也解释了 DeepSWE 的五档结果：同一 Sonnet 5 在 max 与 low 下的 Pass@1、成本和 Agent steps 不同，但变化来自模型配置、`mini-swe-agent`、工具、仓库环境、预算和 verifier 的组合。它可以说明 test-time compute 的系统取舍，不能证明出现了五个 checkpoint，也不能把 max 分数迁移到其他 harness。

System Card 还提供了一个更严格的读表方式。发布方能力结果通常使用 adaptive + max、多个 trials 和特定工具环境；BrowseComp 的 10M token limit、约 200K compaction 触发和 safeguards 状态都可能改变结果。安全评测中 Claude Code 恶意请求拒答率 92.37%、computer-use 恶意任务拒答率 84.68%，并不等于模型具备一个独立的安全分类器；它们是给定策略和执行环境中的行为结果。

面试时可以用下面的三层预算回答：

```text
effort       -> 当前请求倾向于投入多少推理/工具行为
max_tokens   -> 当前响应允许生成的硬上限
task budget  -> 整个 Agent loop 可消耗的思考、工具、重试和验证资源
```

三层都需要独立的停止条件。若 verifier 没有新证据、工具状态未知或权限不明确，增加 effort 不能自动把 unknown 变成 success；控制器应转入验证、人工确认或安全停止。

## Qwen3-VL：视觉推理的 reward 也要分层

Qwen3-VL 的 Thinking with Images 训练把 answer accuracy、multi-turn reasoning 和 tool-calling reward 分开。这个拆分把 reasoning 的“想得对”与 Agent 的“取证和行动过程可靠”区分开：最终答案正确，不能证明视觉证据读取充分；调用工具更多，也不能证明探索策略更好。

SAPO/General RL 和视觉 Agent tool-integrated RL 的完整超参、reward 权重和 rollout recipe 仍未公开到可独立复现的程度。面试或实验中应固定模型 variant/revision、工具、权限、任务环境、verifier 和统计方法，并把论文/发布方结果与本地轨迹实验分开。

参考资料：

- Geifman and El-Yaniv, SelectiveNet: A Deep Neural Network with an Integrated Reject Option：<https://arxiv.org/abs/1901.09149>
- Madras et al., Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer：<https://arxiv.org/abs/1711.06664>
- Chen et al., FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance：<https://arxiv.org/abs/2305.05176>
- Graves, Adaptive Computation Time for Recurrent Neural Networks：<https://arxiv.org/abs/1603.08983>
- OpenAI, Reasoning Models Guide：<https://platform.openai.com/docs/guides/reasoning>

## 18.20 Claude Opus 5.5：把 token efficiency 纳入 effort 控制

Anthropic 对 Claude Opus 5.5 的发布描述把重点从“回答更长”转向“同一项长任务使用更少的 token、工具轮次和重试”。这不能证明模型内部采用了某种新的推理算法，但给出了一个很适合面试的系统指标：在固定任务和 verifier 下，比较单位成功成本，而不是只比较最终答案。

```text
successful_task_cost
  = input_cost + cache_read/write_cost + output_cost
  + tool_cost + retry_cost + fallback_cost

quality_per_cost = verified_successes / successful_task_cost
```

`effort`、`max_tokens` 和 Agent 任务预算仍然是三个不同控制面。发布方多数 Opus 5.5 benchmark 使用 adaptive + max，Terminal-Bench 使用 xhigh，而成本曲线会使用 default/medium；如果把这些结果放在同一行，就会把推理预算差异误读成模型能力差异。

部署时还要把同步请求和批处理合同分开：Anthropic 当前模型页给出同步输出上限 128K；Message Batches API 在 `output-300k-2026-03-24` beta header 下才支持最多 300K。模型页同时列出 1M context、512-token 最小可缓存 prompt，以及 cache read 和 5 分钟/1 小时 cache write 的独立价格。预算账本应记录 endpoint、effort、输出上限、cache read/write 和 verifier 成本；不能把 Batch 的上限当成同步 API 的模型能力。

因此评测记录至少需要保存 model/revision、effort、fallback、thinking/output token、工具轮数、任务预算、verifier、成功 artifact 和真实成本。若 Cyber 或 biology safeguard 触发 fallback，最终结果必须写成带路由的系统结果，不能归因给 Opus 5.5 单体。

这也改变了“更强 reasoning”的验收方式：模型先取得完整上下文、减少局部重复修改，再由测试、静态检查或领域 verifier 判断是否成功；没有新证据时增加 effort 不能把 unknown 变成 success。

Opus 5.5 的官方 API contract 还给出一个重要迁移约束：adaptive thinking 始终开启，`thinking.type=disabled` 和手工 `budget_tokens` 都会失败，推理深度由 `output_config.effort` 控制。thinking block 可能先于 text block 返回，且会绑定产生它的模型、conversation 和消息前缀；模型切换、工具定义变化或 compaction 后，harness 必须验证 block 是否仍可回放。这里的“thinking”是 API state protocol，不等于服务向用户暴露完整思维链。

因此 reasoning evaluator 应把请求验证和响应解析也纳入测试：旧客户端若按位置取第一个 text block、把 `any` 当成通用强制工具，或在消息前缀变化后盲目重放 thinking block，都会在模型分数不变时产生线上回归。官方模型页还要求将同步 128K 输出、带 beta header 的 Batch 300K、always-on thinking、默认 medium 和 cache 账本分别处理。资料依据：[Claude Opus 5.5 model page](https://platform.claude.com/docs/en/models/opus-5-5/overview.md) 与 [What's new](https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5.md)。

对应的教学实现见 [`claude_opus55_protocol_audit.py`](../../research/model-update-2026-09/code/claude_opus55_protocol_audit.py)：脚本按 `block.type` 解析 thinking/tool/result/compaction，检查 producer 与 prefix binding，并用负例验证 disabled thinking、手工 budget、forced tool choice、旧 computer tool 和错误进度块会被拒绝。输出仅是 `local_protocol_toy`，不等于 Anthropic 服务端行为或模型 reasoning 质量。

资料边界：Opus 5.5 当前有 Artificial Analysis 精确条目和 Anthropic 官方 API contract、发布页/System Card 入口，但 DataCurve 没有精确 Agent 行，参数、架构、完整训练 recipe、adaptive thinking 内部实现和独立复现仍未确认。

### 18.20.1 System Card 让 effort 评测回到条件曲线

System Card 的能力数字不能脱离 effort 和 harness。Terminal-Bench 4.0 `66.36%` 使用 xhigh、Claude Code `--bare`、5 trials；OSWorld 2.0 partial/strict `81.8%/48.7%` 绑定 108 tasks、500 actions 上限和超过 100K tokens 后的 compaction；ProgramBench `91.2%` 绑定 166 golden tasks。因而 adaptive thinking 的评估至少应同时保存 effort、max token、compaction、工具动作、verifier 和实际模型。

CoBench 2.1 `55.8%` 和 AECI `169.36` 是 System Card 的发布方 fit 结果，不是“推理 token 越多就越自治”的证明；AECI 新 fit 的 374 benchmarks、7,985 observations、732 models 也使它不能和旧 fit 直接纵向比较。多 Agent 的约 `2.7x`/`2.8x` speedup 使用 derived latency，说明编排可以改变测试时计算路径，但不等于某个新的 reasoning checkpoint。

## 18.21 GPT-6 Sol：mode、effort 与会话内预算更新

GPT-6 Sol 很适合用来说明“模型档位”和“推理预算”不是同一个字段。OpenAI 官方模型页确认 `reasoning.effort` 支持 `none`、`low`、`medium`、`high`、`xhigh`、`max`，默认是 `medium`；Reasoning 文档进一步把 GPT-6 family 的 `reasoning.mode` 分成 `standard` 与 `pro`。前者选择执行模式，后者控制所选模式内投入多少推理，不能把 `pro + low` 或 `standard + high` 简化成一个统一的“模型等级”。

这会改变评测 manifest。除了模型和 snapshot，还要保存：

```text
model, snapshot, mode, effort, max_output_tokens
reasoning_tokens, visible_output_tokens, tool_tokens
task_budget, timeout, retries, verifier, final_artifact
```

Reasoning token 不直接暴露为可读思维链，但会占用 context 并计入 output token。响应可能在可见文本出现前因为 `max_output_tokens` 或 context limit 变成 `incomplete`，所以“没有最终答案”不能简单归因于模型不会做题；也可能是预算、工具或恢复策略先耗尽。实验开始时应为 reasoning 和 visible output 预留足够空间，再用 usage object 校准。

### 18.21.1 `configuration_update` 是会话状态，不是换模型

GPT-6 family 在标准单 Agent 会话中支持 `configuration_update`。例如第一轮用 low 生成草案，下一轮在用户消息前插入：

```json
{
  "type": "configuration_update",
  "reasoning": {"effort": "high"}
}
```

它只改变后续响应的 effort；request-level `reasoning.effort` 仍可保持原值。effective effort 会一直延续到后续响应，直到另一条 update 覆盖。更新项必须随 `previous_response_id` 或完整 history replay 保留在原位置，两个相邻 update 会被拒绝。它不能和自动 compaction/automatic truncation 组合；独立 `/responses/compact` 也拒绝含 update 的历史；显式 compact 后要在下一条用户消息之前重新放置所需 update。该 item 可放入 Responses 请求或 WebSocket `response.create`。

有个监控陷阱：官方文档说明响应的 `reasoning.effort` 字段仍反映 request-level setting，不代表 `configuration_update` 选出的 effective effort。trace 应同时保存请求级值、按历史 item 还原的当前值和 token usage。保持 request-level 设置不变、只追加 update，可以保留原 prompt prefix，利于 prompt-cache 复用；这不保证服务端一定 cache hit。

因此 runtime trace 应区分：

```text
model identity -> request-level effort -> configuration update
-> actual response usage -> tool/verifier result
```

`configuration_update` 改变的是 test-time compute policy，不证明权重、训练阶段或内部 attention 发生变化。无新证据时把 effort 从 low 提高到 max，也不能把 verifier 失败自动变成成功。

### 18.21.2 和榜单结果如何对齐

Artificial Analysis 的 `GPT-6 Sol (max)` 是一个配置级第三方测量；DataCurve 当前没有精确 `mini_swe_agent_gpt_6_sol_*` 行。因而不能把 GPT-6 Astra 或 GPT-5.6 的 Agent 结果移植给 Sol，也不能把 AA 的 Intelligence Index 当成“max 模型裸分”。公平实验至少固定 mode、effort、工具、harness、任务环境、压缩策略和 verifier，并报告 quality、reasoning/output token、steps、延迟和单位成功成本。

资料依据：[GPT-6 Sol model page](https://developers.openai.com/api/docs/models/gpt-6-sol.md)、[Reasoning models](https://developers.openai.com/api/docs/guides/reasoning.md)。这些资料公开的是 API/runtime 契约；参数、dense/MoE、attention 变体和完整训练 recipe 仍未知。

### 18.21.3 用 toy 验证配置更新的失败路径

配套 [`gpt6_sol_contract_audit.py`](../../research/model-update-2026-09/code/gpt6_sol_contract_audit.py) 将 `configuration_update` 当作历史 item，而不是普通的请求参数：相邻 update、`pro`/multi-agent、automatic compaction/truncation 和 standalone compact 都拒绝；显式 compaction 后必须在下一条用户消息前重新插入 update。脚本同时检查 reasoning/output/context 的 `incomplete`、272K whole-request pricing 和 verifier/幂等门禁。运行结果为 `ok=true`、`network_called=false`，证据等级仅为 `local_protocol_toy`。

## 18.22 GPT-6 Luna：family runtime 与 sibling budget

Luna 的 `reasoning.effort` 同样支持 `none`、`low`、`medium`、`high`、`xhigh`、`max`，但“高效率”是 focused/high-volume 产品定位，不是一个可推导的模型架构。GPT-6 family 的 `reasoning.mode=standard/pro` 与 effort 仍是两个控制面，`configuration_update` 只改变后续会话预算，不切换 Luna/Sol checkpoint。

Artificial Analysis 有精确 `GPT-6 Luna (max)`，DataCurve 没有精确 `mini_swe_agent_gpt_6_luna_*` 行；因此不能把 Sol/Astra/GPT-5.6 的 Agent 结果移植给 Luna。公平对照还要绑定 Luna 自己的 model ID、2026-05-18 cutoff、价格、provider、工具、harness、压缩策略和 verifier。证据见 [`gpt-6-luna-source-notes.md`](../../research/model-update-2026-09/gpt-6-luna-source-notes.md)。
