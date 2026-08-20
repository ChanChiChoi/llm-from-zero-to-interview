# 第 13 章 Reasoning Effort 与推理预算：多花计算不等于只生成更长文本

## 13.1 同一个模型，为什么有时要多想几秒

让模型回答“北京是哪个国家的首都”，和让模型定位一个跨文件的并发 bug，表面上都是一次对话，实际需要的计算完全不同。前一个问题可以在一次前向和一次解码中结束；后一个问题可能需要阅读多个文件、提出假设、运行测试、观察失败、修改方案，再重新验证。

如果所有请求都强制使用最高推理预算，简单问题会承担不必要的延迟和费用；如果所有请求都使用最低预算，困难问题又没有足够的搜索和验证机会。`reasoning_effort` 这类参数出现的背景，正是把“愿意为这次任务投入多少额外计算”变成请求级策略。

这里的“推理”不应狭义理解为用户看得到的一段思考文字。它可能包括内部 token、候选答案、树搜索分支、verifier 调用、代码执行、工具轮次和失败恢复。产品接口把这些复杂动作隐藏在 `low`、`medium`、`high` 或数值预算之后，工程师则必须知道这些档位究竟换来了什么。

## 13.2 三个容易混淆的量

第一个量是随机性。`temperature` 和 `top_p` 改变采样分布，让模型更愿意探索不同候选。它们不保证模型会进行更长的验证；高 temperature 甚至可能让候选更不稳定。

第二个量是计算预算。预算决定模型可以产生多少内部推理 token、尝试多少候选、调用多少工具、运行多少验证或进行多少次恢复。预算增大，可能提高成功率，也可能只是让错误重复更久。

第三个量是时间预算。它是 wall-clock deadline，包含排队、网络、工具和 GPU 调度时间。高 effort 请求可以因为工具超时而没有生成很多 token，低 effort 请求也可能因队列拥堵而等待很久。

因此，下面三句话的含义不同：

- “把 temperature 调低”是在减少采样随机性；
- “允许 8K reasoning token”是在增加内部计算上限；
- “最多等待 5 秒”是在设置服务级 deadline。

一个健壮的系统会分别记录这三个量，而不是把它们都叫作“思考深度”。

## 13.3 从 test-time compute 到请求级预算

早期的大语言模型主要把计算集中在训练阶段。模型参数固定以后，推理时只沿着一条自回归路径生成答案。后来研究者发现，对于数学、代码和规划任务，可以在推理阶段生成多个候选，再用一致性、规则或验证器选择更好的结果。self-consistency、树搜索、程序执行和 verifier-guided decoding 都属于把额外计算放在测试时的思路。

这类方法有一个共同的成本曲线。较小的计算预算可能覆盖大部分简单样本；超过某个难度后，额外候选或验证显著提高成功率；再继续增加预算，收益逐渐饱和，甚至因为搜索空间太大而引入新的错误。`reasoning_effort` 的工程任务，就是把这个曲线暴露给路由器，而不是把所有请求推到曲线最右端。

可以把一次任务的资源拆成：

```math
B=B_{\mathrm{reason}}+B_{\mathrm{tool}}+B_{\mathrm{verify}}+B_{\mathrm{recover}}
```

只有在各分量已经换算到同一资源单位时，才可以把它们写成一个标量 `B`。更常见的记录方式是预算向量 `\mathbf B=(B_reason,B_tool,B_verify,B_recover)`，其中每一项都应是有限的非负数，并记录单位和硬上限。`B_reason` 可以是内部 token，`B_tool` 可以是工具调用次数或工具时间，`B_verify` 可以是测试、规则检查或第二模型的次数，`B_recover` 可以是失败后的重试额度；调用次数、毫秒和 token 不能未经换算直接相加。

若某项预算没有启用，应记录为明确的 0；若尚未测量，应记录为缺失值而不是 0。否则报告会把“没有工具调用”和“工具调用成本没有被记录”混成同一种情况。

如果任务价值用 `V` 表示，成本用 `C(B)` 表示，时间和风险分别用 `T(B)`、`R(B)` 表示，一个简化的效用可以写为：

```math
U(B)=V\,P_{\mathrm{success}}(B)
-\lambda C(B)-\mu T(B)-\nu R(B)
```

这里 `0\le P_success(B)\le1` 是在固定任务分布、成功定义和预算策略下的成功概率；`V` 是一次成功的业务价值；`C(B)`、`T(B)` 和 `R(B)` 分别是成本、时间和风险暴露或期望损失；`\lambda、\mu、\nu` 是把它们换算到同一效用单位的非负权重。所有量都应是有限数，且成本、时间和风险通常不应为负。若风险是不可接受的硬约束，就不能只把它乘一个较大的 `\nu`，还要在路由前直接拒绝不满足约束的策略。

公式的作用不是精确预测内部模型，而是提醒我们：预算策略不能只追求成功率。医疗、支付和生产变更任务还必须考虑错误副作用和人工复核成本。若某个量没有观测到，`U(B)` 应报告为未计算，而不是用 0 假装没有成本或风险。

## 13.4 effort 如何进入一次请求

一次请求可以经过下面的生命周期：

```text
request
  -> classify task and risk
  -> choose budget
  -> reason / propose
  -> verify or call tool
  -> observe result
  -> continue, recover, downgrade, or answer
```

路由器先根据任务类型、用户等级、风险和剩余容量选择预算。模型 runtime 负责生成候选或 reasoning item；工具执行器只执行通过权限检查的动作；verifier 检查代码、数学或结构化输出；调度器在 deadline 和资源压力下决定是否继续。

若 runtime 只是把 `high` 映射为更大的 `max_tokens`，系统可能得到一段更长的解释，却没有得到新的证据、测试或验证。相反，一个有效的高 effort 策略可能表现为内部思考 token 增加不多，但多运行了一次测试或比较了两个候选 patch。

## 13.5 从一个代码修复任务看预算分配

假设有三类请求。第一类是 API 名称查询；第二类是一个文件中的空指针修复；第三类是跨服务的竞态问题，需要修改代码并运行集成测试。

对第一类请求，direct 路径足够。系统可以给极小的 reasoning 预算，不启用代码工具，答案经过格式和拒答规则检查即可。对第二类请求，可以允许一次定位、一次 patch 和一次单元测试。对第三类请求，则需要给出多个候选假设、跨文件检索、测试失败后的恢复和更长的 deadline。

一个实验可以记录如下结果：

| 路径 | 成功率 | reasoning token | 工具轮次 | p95 | 单位成功成本 |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct | 0.72 | 1.0x | 0.1 | 1.1 s | 1.4 |
| verify | 0.80 | 1.6x | 0.8 | 3.4 s | 2.5 |
| deep | 0.84 | 3.1x | 2.1 | 8.2 s | 4.8 |

如果所有请求都是简单查询，`deep` 显然不划算；如果第三类生产变更失败一次的代价远高于几倍推理成本，那么深路径可能是合理的。关键不是给档位贴上“聪明”标签，而是让预算和任务价值对应起来。

## 13.6 什么时候继续，什么时候停止

模型或 Agent 继续推理应该有可观察的进展信号：候选之间存在分歧、verifier 分数提高、测试从失败变为通过、工具返回了新证据，或者当前答案仍缺少问题要求的字段。若连续几轮重复同一计划、工具结果没有变化、候选质量没有改善，就应该停止、换路径或请求人工。

可以把继续条件写成一个工程验收条件：

```math
G_{\mathrm{continue}}
=\mathbf{1}[B_{\mathrm{used}}<B_{\max}]
\mathbf{1}[P_{\mathrm{gain}}>\delta]
\mathbf{1}[T_{\mathrm{remaining}}>0]
```

其中 `P_gain` 表示下一轮获得有效进展的估计概率，`\delta` 是最低收益阈值。真正的系统可能用启发式、学习到的控制器或 bandit 策略估计它，但三个约束不能消失：预算要有硬上限、继续要有理由、deadline 要能被满足。

该式要求 `B_used` 和 `B_max` 使用同一资源口径，`P_gain\in[0,1]`，`\delta\in[0,1]`，并且 `T_remaining` 是有限的非负时间。若进展概率未知、预算上限缺失或时间已超时，系统应返回“不继续”或进入人工处理，而不是把缺失值当作满足条件。三个指示函数相乘的结果只表示是否允许继续，不表示继续一定成功。

高风险工具还要有独立的动作上限。即便 reasoning 预算未用完，删除文件、发送邮件、修改生产配置等动作也不能因为模型选择了 high effort 就自动放行。计算预算是资源控制，权限是安全控制，二者必须分离。

## 13.7 如何证明 effort 真的带来了能力

评估时固定模型 revision、system prompt、数据、工具权限和输出协议，只改变 effort。每个 bucket 同时记录：答案质量、验证通过率、reasoning token、工具调用、verifier 调用、TTFT、TPOT、p95/p99、重试和安全事件。

不要只看平均准确率。把简单题和难题混在一起，high effort 可能只是帮助少数困难样本，却让全部用户多付钱。更有信息量的图是按任务难度画 `success-versus-cost` 曲线，并报告在固定 deadline 下的成功率。

还要做行为回归。例如高 effort 是否更容易输出没有依据的长解释？是否因为探索更多工具而越过权限边界？是否在工具失败时重复调用造成外部副作用？推理能力上升不能抵消安全和协议回归。

## 13.8 一个最小控制器的伪代码

下面的伪代码表达的是策略边界，不是某个供应商的内部实现。它使用 `python` 风格语法帮助读者阅读，但依赖未定义的业务对象，不能直接运行；后面会给出一个不依赖外部服务的最小可运行控制器。

```text
budget = policy.choose(task_type, risk, deadline, capacity)
state = start(task)

while budget.remaining() > 0 and not deadline.expired():
    proposal = model.reason(state, budget.reason_slice())
    decision = verifier.check(proposal, state)

    if decision.accepted:
        return commit(proposal)
    if decision.needs_tool:
        if not auth.allow(decision.tool_call):
            return ask_for_confirmation(decision.tool_call)
        observation = tool.execute(decision.tool_call)
        state = state.observe(observation)
        budget.charge_tool(observation)
        continue
    if not decision.has_progress:
        return fallback_or_escalate(state)
    state = state.refine(proposal, decision)

return timeout_or_partial_result(state)
```

这里最重要的不是循环本身，而是 `verifier`、`auth`、`budget.charge_tool` 和 `fallback_or_escalate` 都是显式模块。没有它们，所谓 effort 很容易退化成一个没有可解释性的输出长度开关。

伪代码还有一个重要前提：`decision.accepted` 不能直接等同于“可以改变外部状态”。即使 verifier 判断候选内容正确，最终提交仍要经过独立的 policy、权限和确认检查；`budget.charge_tool()` 也必须在调用前预留资源、在调用后按实际消耗结算，不能让失败重试绕过硬上限。

### 13.8.1 最小可运行的预算控制器

下面的例子把一个任务简化为三个阶段：直接回答、验证和升级。它不调用模型或工具，只用确定性的教学事件模拟控制器如何根据“有新证据”和“无进展”消耗预算。代码中的 `effort` 是本地策略标签，不代表任何供应商字段的内部语义。

```python
from dataclasses import dataclass


@dataclass
class Budget:
    reason: int
    verify: int
    tool: int

    def __post_init__(self):
        for name in ("reason", "verify", "tool"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def charge(self, kind, amount=1):
        if kind not in {"reason", "verify", "tool"}:
            raise ValueError("unknown budget kind")
        if type(amount) is not int or amount < 0:
            raise ValueError("amount must be a non-negative integer")
        remaining = getattr(self, kind)
        if amount > remaining:
            raise ValueError(f"{kind} budget exhausted")
        setattr(self, kind, remaining - amount)


def choose_effort(task_kind, risk):
    if risk not in {"low", "high"}:
        raise ValueError("risk must be low or high")
    if task_kind not in {"lookup", "code"}:
        raise ValueError("unknown task kind")
    if risk == "high":
        return "verify"
    return "direct" if task_kind == "lookup" else "verify"


def run_controller(task_kind, risk, events):
    if not isinstance(events, list) or not all(isinstance(event, str) for event in events):
        raise ValueError("events must be a list of strings")
    effort = choose_effort(task_kind, risk)
    budget = Budget(reason=2 if effort == "verify" else 1, verify=1 if effort == "verify" else 0, tool=0)
    trace = [f"start:{effort}"]

    if effort == "direct":
        budget.charge("reason")
        trace.append("answer:direct")
        return {"status": "answered", "trace": trace, "remaining": budget}

    budget.charge("reason")
    trace.append("answer:proposal")
    for event in events:
        if event == "new_evidence":
            budget.charge("verify")
            trace.append("verify:passed")
            return {"status": "answered", "trace": trace, "remaining": budget}
        if event == "no_progress":
            trace.append("escalate:human")
            return {"status": "escalated", "trace": trace, "remaining": budget}
        raise ValueError("unknown event")
    trace.append("timeout:partial")
    return {"status": "partial", "trace": trace, "remaining": budget}


lookup = run_controller("lookup", "low", [])
code = run_controller("code", "low", ["new_evidence"])
high_risk = run_controller("lookup", "high", ["no_progress"])
print(lookup["status"], code["status"], high_risk["status"])
print(lookup["trace"])
print(code["trace"])
print(high_risk["trace"])
```

预期输出：

```text
answered answered escalated
['start:direct', 'answer:direct']
['start:verify', 'answer:proposal', 'verify:passed']
['start:verify', 'answer:proposal', 'escalate:human']
```

这个例子只验证控制流和预算扣减，不验证答案正确性，也没有真正的权限服务。生产实现还必须为候选、工具参数、外部状态、确认和日志定义独立 schema，并把超时、重试和部分成功纳入账本。

## 13.9 常见失败

最常见的失败是把 effort 当 temperature。这样既没有得到更多验证，也不一定得到更稳定的结果。

第二个失败是高档位只让回答变长。可以检查高档位与低档位的工具次数、验证通过率和可引用证据是否真的增加；如果只有输出 token 增加，说明资源没有花在解决问题上。

第三个失败是预算没有硬上限。工具不可用、测试一直失败或外部服务返回相同结果时，Agent 可能无限循环，最终同时消耗模型和工具资源。

第四个失败是不同 provider 的同名字段被当成同一语义。一个服务的 `reasoning_effort=high` 可能影响隐藏 token，另一个服务可能只调整路由优先级。迁移时必须比较实际 token、事件、工具和计费。

## 13.10 面试回答与练习

面试官问“reasoning effort 怎么设计”，可以回答：它是请求级的 test-time compute 策略，不等于 temperature，也不等于可见思考文本。系统要把推理、工具、验证、恢复和时间拆开计量，根据任务难度和风险路由；每个请求设置预算和停止条件，用成功率、单位成功成本、尾延迟和安全事件评估收益，不把 high 档位全量开启。

练习一：为客服问答、代码修复和研究 Agent 设计 low、medium、high 三档预算，写出每档允许的工具、验证和最大时间。

练习二：一个 high effort 路径把成功率从 0.72 提高到 0.80，但单位成本变为 3 倍。分别从低风险聊天、支付风控和生产发布任务判断是否启用，并说明缺失哪些数据。

练习三：设计一个“无进展”检测器，至少使用重复工具调用、候选相似度和 verifier 分数三个信号。

### 13.10.1 预算曲线如何测出来

不要先给 `low`、`medium`、`high` 贴上固定 token 数，再假设档位有意义。更可靠的做法是固定任务集，逐步增加 reasoning token、候选数、工具次数或 deadline，得到一条质量—成本曲线。对每个点记录成功率、引用支持、验证通过、p95、单位成功成本和危险副作用。

若成功率从 0.70 提升到 0.82 只需要把预算从 2K 增到 6K，之后增到 20K 只提升到 0.83，那么 6K 附近可能是该任务集的收益拐点。这个拐点不能直接迁移到其他模型或任务，但能帮助路由器避免默认使用最高档位。

### 13.10.2 任务分类器的风险

动态预算通常需要一个任务分类器判断问题难度。分类器可能把“输入很长”误认为“推理很难”，也可能把短但高风险的支付确认判为简单任务。输入长度、任务难度、风险等级和输出长度是四个不同维度，不能用一个启发式代替。

分类器的错误应有安全上限：难度判断错，可以让任务转入更高预算或返回不确定；风险判断错，不能自动获得更多权限。对高风险动作，策略应直接要求证据和审批，而不是用 high reasoning 作为补偿。

### 13.10.3 一个可解释的升级例子

代码 Agent 初始获得 3 次工具调用和 4K reasoning budget。若第一次测试失败，且失败日志包含新的栈信息，系统可以增加一次诊断预算；若连续两次调用完全相同的测试并得到相同错误，则不再升级，而是返回人工或换用检索路径；若 patch 已通过隐藏测试但尚未确认写入，增加的不是思考 token，而是审批状态。

这个例子说明升级信号来自状态变化。没有新证据的重复计算不应被当作有效 reasoning；预算控制器必须同时看 token、工具观察、verifier 进展和剩余时间。

## 13.11 把预算从 token 数变成资源账本

把推理预算写成“最多生成 8K token”很容易，但这不是一个完整的资源模型。内部 reasoning token 只是模型计算的一部分。一次代码修复可能只生成 2K 个内部 token，却读取了 30 个文件、运行了三次测试，并让 GPU 在长上下文上重复做了大量 KV 读取；另一次数学题可能生成 8K token，但不调用任何外部工具。两者的成本和风险不能用一个数字比较。

可以把每个请求的预算记录成向量：

```math
\mathbf{B}=(B_{\mathrm{reason}},B_{\mathrm{tool}},B_{\mathrm{verify}},B_{\mathrm{time}},B_{\mathrm{side}})
```

其中 `B_reason` 是内部或显式推理 token 上限，`B_tool` 是工具调用次数或工具时间，`B_verify` 是测试、规则检查和独立 verifier 的额度，`B_time` 是 wall-clock deadline，`B_side` 是允许的外部副作用次数。每一项都必须与实际消耗使用同一单位，且是有限的非负数；`B_side=0` 应明确表示不允许副作用。最后一项尤其重要：读取日志和发送邮件都可能只占一次工具调用，但它们对系统的影响完全不同。

单位成本也应按资源拆开：

```math
C= c_r N_r+c_t T_{\mathrm{tool}}+c_v N_v+c_g T_{\mathrm{gpu}}+c_h T_{\mathrm{human}}
```

`N_r`、`N_v` 分别表示 reasoning 和 verifier 的消耗，`T_tool`、`T_gpu`、`T_human` 表示工具、GPU 和人工时间；系数由账单、容量成本或业务估值给出。这里要求各资源量和系数都是有限的非负数，并且每一项的单位已经换算到同一成本单位；没有发生的资源取 0，未记录的资源保留缺失状态。若要计算单位成功成本，还需要定义同一批任务中的成功事件数；成功数为 0 时，单位成功成本未定义，而不是把总成本除以 1。这个式子不是要求把每个 token 都精确计价，而是防止系统只盯着模型 API 的 token 账单，忽略工具服务、排队、存储和人工审核。

对初学者来说，可以把预算理解成旅行预算：车票只是总花费的一部分，换乘、等待和行李也会消耗资源。对工程师来说，更关键的是定义每种资源的硬上限和软上限。软上限用于动态控制，硬上限用于防止异常循环；高风险副作用还要由独立权限系统控制，不能因为剩余 token 很多就自动放行。

## 13.12 预算控制器的状态机

一个可生产化的 effort 控制器通常不是一次选择后就不再变化，而是随着证据和验证结果改变状态。可以把它分成五个状态：`direct` 直接回答，`diagnose` 诊断和检索，`verify` 执行测试或规则检查，`escalate` 升级预算或人工，`terminal` 完成、拒答或超时。

状态转移必须由事件触发，而不是由模型说“我还需要想一想”触发。例如，工具返回新的栈信息可以从 `diagnose` 转到 `verify`；连续两次相同失败且没有新证据，应从 `verify` 转到 `escalate` 或 `terminal`；高风险写操作永远需要 policy gate，而不是从 `verify` 直接转到 `terminal`。

```text
direct --evidence_missing--> diagnose
diagnose --proposal_ready--> verify
verify --passed------------> terminal
verify --new_failure-------> diagnose
verify --no_progress-------> escalate
any    --deadline----------> terminal
```

为了避免控制器在阈值附近反复升降，可以使用迟滞。设最近窗口的有效收益为 `g_t`，上升阈值为 `\delta_{\mathrm{up}}`，下降阈值为 `\delta_{\mathrm{down}}`，并满足 `\delta_{\mathrm{down}}<\delta_{\mathrm{up}}`：

```math
L_{t+1}=\begin{cases}
L_t+1,&g_t>\delta_{\mathrm{up}}\ \text{且连续满足 }m\text{ 次}\\
L_t-1,&g_t<\delta_{\mathrm{down}}\ \text{且连续满足 }n\text{ 次}\\
L_t,&\mathrm{otherwise}
\end{cases}
```

这里的 `g_t` 可以由 verifier 分数改善、新证据数量、测试状态变化和候选分歧组成。它应是有限的可比较量，不是模型自报的信心分数。`L_t` 应是有界整数级别，`m、n` 是正整数，`\delta_down<\delta_up` 且两个阈值使用同一收益单位；更新时还要将 `L_t+1` 和 `L_t-1` 截断到合法档位范围。工程上还应把每次升级原因写入 trace，例如 `reason=evidence_missing`、`reason=test_new_signal` 或 `reason=deadline_near`，否则线上成本突然升高时无法区分任务变难和控制器失灵。

## 13.13 用实验证明 high effort 的收益来源

高 effort 只有在固定条件下提高了目标能力，才有理由进入默认路径。实验至少要有 `always-low`、`always-high` 和 `adaptive` 三组；模型 revision、system prompt、工具权限、采样参数、任务集和成功定义必须相同。对 reasoning 模型，还要明确是否把内部 token、可见摘要和工具事件分别计量。

任务集不能只取平均难度。应按任务类型、难度、风险、输入长度和是否可验证分层。每层报告成功率、单位成功成本、p95/p99、有效工具比例、verifier 通过率、重复副作用率和人工接管率。若 high effort 只帮助最难的 5% 样本，却让全部请求延迟翻倍，那么产品策略应是难题升级，而不是全量 high。

一个简单的统计报告可以给成功率 `\hat p` 配置二项置信区间：

```math
\hat p=\frac{k}{n},\qquad
\mathrm{SE}(\hat p)=\sqrt{\frac{\hat p(1-\hat p)}{n}}
```

这里要求 `n` 是正整数，`k` 是 `0` 到 `n` 之间的整数，因而 `\hat p\in[0,1]`。`n=0` 时成功率和标准误都未定义；小样本或 `k=0/n` 时，Wald 标准误只是近似，正式报告更适合使用 Wilson 区间或精确方法。

实际报告可使用 Wilson 区间或 bootstrap，而不是只比较两个小样本的百分点。还要做 paired replay：让同一个任务在不同预算下运行，并记录每一次从错误变正确、从无验证变有验证的转折。这样才能回答“收益来自更多计算、额外工具、验证器，还是只是随机采样差异”。

对外部 provider，只能依据公开文档确认字段的语义。例如 OpenAI 的 reasoning 文档说明了 reasoning effort、reasoning token 和 Responses API 中的状态处理入口；Anthropic 的 extended thinking 文档说明了其产品协议中的思考预算与工具使用边界。它们可以支持“档位是请求协议的一部分”这一事实，但不能据此推断所有闭源模型内部使用了同一种搜索或 verifier。

## 13.14 一个预算决策的完整例子

设一个文档审查任务有三个候选策略：低预算只做一次提取，中预算增加交叉检查，高预算允许多次候选和工具验证。低预算耗时 2 秒、成功率 0.72；中预算耗时 5 秒、成功率 0.84；高预算耗时 13 秒、成功率 0.86。若只看准确率，高预算似乎更好；若任务的 SLO 是 p95 小于 8 秒，高预算根本不能进入默认路径。

更合理的路由会估算边际收益：

```math
\mathrm{value}_{k}
=\frac{\Delta Q_k}{\Delta C_k}
-\lambda\frac{\Delta L_k}{L_{\max}}
-\mu\frac{\Delta R_k}{R_{\max}}.
```

其中 `Q` 是独立评测质量，`C` 是单位成本，`L` 是延迟，`R` 是风险；要求 `\Delta C_k\ne0`、`L_max>0`、`R_max>0`，并说明各增量是在同一配对任务上测得。只有 `value_k` 为正且不违反硬性条件时，才升级到下一档。若成本没有增加，第一项未定义；若风险或延迟基线未知，不能用 0 代替。这个公式不是要求在线服务精确知道真实概率，而是强迫设计者把“多想一会儿”说成一个可测的资源决策。

## 13.15 预算和权限不能共用一个开关

增加 reasoning token 只增加计算，不应自动获得更多文件、网络或写入权限。一个常见的错误是把 `high effort` 和“允许执行更多工具”绑定在一起，导致模型在困难任务上同时拥有更长思考和更大的副作用范围。安全设计应分别维护 compute budget、tool budget、data budget 和 approval budget。

例如，模型可以使用高预算读取三份公开文档，但仍然没有发送邮件权限；也可以拥有发送邮件权限，却必须在低预算摘要通过人工确认后执行。预算状态机和授权状态机分开，才能在超时、重试和恢复时知道哪些资源已经消耗，哪些动作从未授权。

## 13.16 预算实验的最小报告

每次比较不同 effort，至少固定模型 revision、prompt、工具集合、最大输出、采样策略、并发、缓存和评测集。报告平均值之外，还要给出 p50/p95、成功任务成本、停止位置、升级比例、工具调用次数和失败原因。对同一任务做 paired replay，才能减少题目难度差异造成的假象。

如果高预算只让答案更长，却没有提高引用正确率、隐藏测试通过率或恢复成功率，就应降低默认预算或改进 verifier。推理预算的收益必须落到任务结果，而不是落到可见的思考文本长度。

## 13.17 预算消耗的可观测账本

服务端应把输入 token、输出 token、隐藏 reasoning token、候选数、工具时间、验证时间、排队时间和人工等待分开记录。否则“high 比 low 贵多少”只能靠账单猜测，无法知道增加的预算到底用于模型计算还是等待资源。

## 13.18 预算用尽时的最后一步

达到预算上限时，系统应先保存当前状态，尝试做一次轻量验证，再选择返回部分结果、明确不确定、转人工或安全降级。不能直接截断在一个半成品 JSON、未闭合代码块或未经确认的外部动作上。预算边界本身也是协议边界。

## 13.19 小结与资料边界

推理预算的本质是把额外计算花在可验证的进展上。它可以表现为内部 token，也可以表现为候选、搜索、工具、测试和恢复。好的系统同时管理质量、成本、时间和风险，并且将预算与权限分离。

本章涉及的 test-time compute、self-consistency、verifier 和 adaptive inference 概念可由公开论文和官方 reasoning API 文档支持；具体 `reasoning_effort` 字段的映射、可见性和计费属于 provider 版本行为，不能仅凭参数名推断。
