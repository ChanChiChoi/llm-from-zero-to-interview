# 第 13 章 Reasoning Effort 与推理预算：多花计算不等于只生成更长文本

## 13.1 同一个模型，为什么有时要多想几秒

让模型回答“北京是哪个国家的首都”，和让模型定位一个跨文件的并发 bug，表面上都是一次对话，实际需要的计算完全不同。前一个问题可以在一次前向和一次解码中结束；后一个问题可能需要阅读多个文件、提出假设、运行测试、观察失败、修改方案，再重新验证。

如果所有请求都强制使用最高推理预算，简单问题会承担不必要的延迟和费用；如果所有请求都使用最低预算，困难问题又没有足够的搜索和验证机会。工程上需要请求级的计算策略，把“这项任务值得投入多少额外资源”变成可观测、可比较的选择。`reasoning.effort`、`output_config.effort` 等是特定 API 暴露的字段名；它们的档位、默认值和实际作用取决于供应商、模型和 endpoint，不能当成通用标准。

这里的“推理”不应狭义理解为用户看得到的一段思考文字。它可能包括内部 token、候选答案、树搜索分支、verifier 调用、代码执行、工具轮次和失败恢复。有些产品接口用 `low`、`medium`、`high` 这样的档位，有些提供数值预算；字段和档位不是共同刻度，工程师必须知道目标 endpoint 实际公开了什么。

## 13.2 四种容易混淆的控制量

第一个量是随机性。`temperature` 和 `top_p` 改变采样分布，让模型更愿意探索不同候选。它们不保证模型会进行更长的验证；高 temperature 甚至可能让候选更不稳定。

第二个量是 effort 控制。它是模型/API 提供的请求级控制，可能影响模型愿意投入的工作量、生成策略或工具行为；档位名称本身不承诺固定 token 数、GPU 时间或质量提升。

第三个量是显式预算或输出上限。部分接口提供 thinking budget，部分接口只提供总输出 token 上限，还有接口把它们组合起来。thinking budget 的目标值、内部资源限制和总输出硬上限不是同一个概念；具体语义要按当前模型文档确认。

第四个量是时间预算。它是 wall-clock deadline，包含排队、网络、工具和 GPU 调度时间。高 effort 请求可能因为工具超时而没有生成很多 token，低 effort 请求也可能因队列拥堵而等待很久。

因此，下面四句话的含义不同：

- “把 temperature 调低”是在减少采样随机性；
- “选择 high effort”是在请求某个模型/API 使用其定义的高 effort 行为；
- “把 thinking budget 设为 8K”是在设置特定接口定义的思考预算，而不一定是总输出硬上限；
- “最多等待 5 秒”是在设置服务级 deadline。

一个健壮的系统会分别记录这四种控制量和实际消耗，而不是把它们都叫作“思考深度”。

## 13.3 从 test-time compute 到请求级预算

早期的大语言模型主要把计算集中在训练阶段。模型参数固定以后，推理时只沿着一条自回归路径生成答案。后来研究者发现，对于数学、代码和规划任务，可以在推理阶段生成多个候选，再用一致性、规则或验证器选择更好的结果。self-consistency、树搜索、程序执行和 verifier-guided decoding 都属于把额外计算放在测试时的思路。

这类方法有一个共同的成本曲线。较小的计算预算可能覆盖大部分简单样本；超过某个难度后，额外候选或验证显著提高成功率；再继续增加预算，收益逐渐饱和，甚至因为搜索空间太大而引入新的错误。工程任务是测出目标模型、任务和策略下的曲线，并让路由器据此选择资源；不是把所有请求推到曲线最右端。

令 `B_reason`、`B_tool`、`B_verify`、`B_recover` 分别表示推理、工具、验证和失败恢复的资源上限；每个分量都是有限非负值，且须注明单位和硬上限。只有在这些分量已换算到同一资源单位时，才可把它们相加得到标量预算：

```math
B=B_{\mathrm{reason}}+B_{\mathrm{tool}}+B_{\mathrm{verify}}+B_{\mathrm{recover}}
```

更常见的记录方式是预算向量 `\mathbf B=(B_reason,B_tool,B_verify,B_recover)`。`B_reason` 可以是可计量的 reasoning token，`B_tool` 可以是工具调用次数或工具时间，`B_verify` 可以是测试、规则检查或第二模型的次数，`B_recover` 可以是失败后的重试额度；调用次数、毫秒和 token 不能未经换算直接相加。

例如，仅为演示标量加法，假设某套内部成本模型已经把四类资源都换算为“资源点”，一项任务的四个分量分别是 4、3、2、1 点，则 `B=4+3+2+1=10` 点。这个换算系数是系统自己的估值，不是 token、工具调用和毫秒之间存在天然等价关系；若没有可靠换算，就应保留预算向量。

若某项预算没有启用，应记录为明确的 0；若尚未测量，应记录为缺失值而不是 0。否则报告会把“没有工具调用”和“工具调用成本没有被记录”混成同一种情况。

令 `P_success(B)` 为固定任务分布、成功定义和预算策略下的成功概率，满足 `0\le P_success(B)\le1`；`V` 是有限的单次成功业务价值，`C(B)`、`T(B)`、`R(B)` 分别是有限非负的成本、时间和风险暴露；`\lambda、\mu、\nu` 是有限非负的效用换算权重。一个简化的效用可以写为：

```math
U(B)=V\,P_{\mathrm{success}}(B)
-\lambda C(B)-\mu T(B)-\nu R(B)
```

若风险是不可接受的硬约束，就不能只把它乘一个较大的 `\nu`，还要在路由前直接拒绝不满足约束的策略。

代入一个纯教学数例：假设 `V=100` 效用点、`P_success=0.8`、`C=2` 成本单位、`T=4` 秒、`R=0.01`，并取 `\lambda=3` 效用点/成本单位、`\mu=0.5` 效用点/秒、`\nu=200` 效用点/风险单位，则 `U=100×0.8−3×2−0.5×4−200×0.01=80−6−2−2=70` 效用点。数值是为展示代入步骤而构造的；线上使用时，成功定义、成本口径和风险量都必须来自同一任务与测量协议。

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

下面对照截至 **2026-11-13** 查阅的三家官方 API 文档。表中描述的是文档公开的接口合同，不代表模型内部计算方式；模型支持范围和字段默认值可能继续变化。

| API 文档 | effort / thinking 控制 | 输出上限及读账注意点 |
| --- | --- | --- |
| [OpenAI Reasoning models](https://developers.openai.com/api/docs/guides/reasoning) | `reasoning.effort` 的取值和默认值按模型而异；文档列出 `none`、`minimal`、`low`、`medium`、`high`、`xhigh`、`max` 等可用值。支持时，`reasoning.mode` 的 `standard` / `pro` 与 effort 是不同控制项。 | `max_output_tokens` 限制生成总量，涵盖 reasoning、可见输出和格式内容；usage 在支持时可提供 reasoning token 统计，但不能假设每个 endpoint 都暴露相同细节。 |
| [Anthropic Effort](https://platform.claude.com/docs/en/build-with-claude/effort) 与 [Extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) | `output_config.effort` 的支持范围依模型而异；`adaptive` 是 thinking 模式而不是 effort 档位。`thinking.type` 和 `budget_tokens` 也依模型及请求协议而定。 | 在支持手动 thinking budget 的模型上，`budget_tokens` 是目标预算；`max_tokens` 仍是总输出硬上限。不要把 effort 和 budget 值直接当成同一计量单位。 |
| [Google Gemini thinking](https://ai.google.dev/gemini-api/docs/generate-content/thinking) | Gemini 3/4 文档以模型支持情况定义 `thinkingLevel`；Gemini 2.5 使用数值 `thinkingBudget`。文档提示 Gemini 4 应使用 `thinkingLevel`，而旧 budget 字段已弃用。 | `max_output_tokens` 是包含 thought tokens 的基础设施输出上限；它会造成硬截断，但不会据此改变思考分配。Thought token 的使用也会计入用量/计费。 |

表中的字段名、档位和支持状态只是 2026-11-13 的文档快照。迁移 provider 或模型时，应在目标 endpoint 重新核对字段合同，并测量实际质量、usage、工具事件、延迟和费用；同名 `high` 不是跨模型的共同算力单位。

若 runtime 只是把 `high` 映射为更大的 `max_tokens`，系统可能得到一段更长的解释，却没有得到新的证据、测试或验证。相反，一个有效的高 effort 策略可能表现为内部思考 token 增加不多，但多运行了一次测试或比较了两个候选 patch。

## 13.5 从一个代码修复任务看预算分配

假设有三类请求。第一类是 API 名称查询；第二类是一个文件中的空指针修复；第三类是跨服务的竞态问题，需要修改代码并运行集成测试。

对第一类请求，direct 路径足够。系统可以给极小的 reasoning 预算，不启用代码工具，答案经过格式和拒答规则检查即可。对第二类请求，可以允许一次定位、一次 patch 和一次单元测试。对第三类请求，则需要给出多个候选假设、跨文件检索、测试失败后的恢复和更长的 deadline。

为说明报告格式，下面是假想固定测试集上的**合成教学数据**，不是实测或文献结论；成功率使用同一成功定义，token 和成本为相对基线倍数，成本单位是任意教学单位：

| 路径 | 成功率 | reasoning token | 工具轮次 | p95 | 单位成功成本 |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct | 0.72 | 1.0x | 0.1 | 1.1 s | 1.4 |
| verify | 0.80 | 1.6x | 0.8 | 3.4 s | 2.5 |
| deep | 0.84 | 3.1x | 2.1 | 8.2 s | 4.8 |

如果所有请求都是简单查询，`deep` 显然不划算；如果第三类生产变更失败一次的代价远高于几倍推理成本，那么深路径可能是合理的。关键不是给档位贴上“聪明”标签，而是让预算和任务价值对应起来。

## 13.6 硬预算门槛只判断能不能继续

先把策略价值和资源可行性分开。第七章讨论额外计算的边际收益，第十八章讨论控制器怎样根据反馈选择下一步；本节只给账本一个硬门槛：预算未超限、deadline 尚未到、当前动作获准，三项都满足时才允许消耗下一笔资源。

设 `B_used` 和 `B_max` 是同一资源口径下的有限非负数，`T_remaining` 是有限非负秒数，`policy_allows` 是独立策略/权限检查返回的布尔值；`\mathbf{1}[condition]` 在条件成立时为 1，否则为 0。硬可行标记为：

```math
G_{\mathrm{allow}}
=\mathbf{1}[B_{\mathrm{used}}<B_{\max}]
\mathbf{1}[T_{\mathrm{remaining}}>0]
\mathbf{1}[\mathrm{policy\_allows}]
```

例如已用预算为 7、上限为 10，剩余 2 秒且策略允许，则 `G_allow=1`，表示资源上仍可继续；预算已到 10、时间已到 0 或权限检查返回 false 时，至少一个指示函数为 0，系统不得调用下一工具或生成下一轮。若剩余预算或权限状态未知，应按不可继续处理，而不能把缺失当作 0 消耗或默认授权。`G_allow=1` 只说明可行，不证明继续有价值，也不证明结果正确。

高风险动作始终需要独立的授权门槛。即便 reasoning 预算未用完，删除文件、发送邮件、修改生产配置等动作也不能因为模型选择了 high effort 就自动放行。计算预算是资源控制，权限是安全控制，二者必须分离。

## 13.7 如何证明 effort 真的带来了能力

评估时固定模型 revision、system prompt、数据、工具权限和输出协议，只改变 effort。每个 bucket 同时记录：答案质量、验证通过率、reasoning token、工具调用、verifier 调用、TTFT、TPOT、p95/p99、重试和安全事件。TTFT（time to first token）是请求到首个输出 token 的延迟；TPOT（time per output token）是生成阶段平均每个输出 token 的时间。应说明它们的测量端点与流式口径；供应商不公开内部 token 时，不能用可见输出 token 冒充内部 reasoning token。

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

**代码状态：已验证（2026-11-13，Python 3.12.9；仅标准库）。**

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
        if not isinstance(kind, str) or kind not in {"reason", "verify", "tool"}:
            raise ValueError("unknown budget kind")
        if type(amount) is not int or amount < 0:
            raise ValueError("amount must be a non-negative integer")
        remaining = getattr(self, kind)
        if amount > remaining:
            raise ValueError(f"{kind} budget exhausted")
        setattr(self, kind, remaining - amount)


def choose_effort(task_kind, risk):
    if not isinstance(risk, str) or risk not in {"low", "high"}:
        raise ValueError("risk must be low or high")
    if not isinstance(task_kind, str) or task_kind not in {"lookup", "code"}:
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

第四个失败是把不同 provider 的字段名或同名档位当成统一语义。公开文档只能确认特定模型和 endpoint 的接口合同，不揭示完整内部实现。迁移时应核对目标文档，并比较实际 usage、事件、工具、延迟和计费；没有公开的数据应标为不可观测，不能凭字段名猜测。

## 13.10 面试回答与练习

面试官问“reasoning effort 怎么设计”，可以回答：它是请求级的 test-time compute 策略，不等于 temperature，也不等于可见思考文本。系统要把推理、工具、验证、恢复和时间拆开计量，根据任务难度和风险路由；每个请求设置预算和停止条件，用成功率、单位成功成本、尾延迟和安全事件评估收益，不把 high 档位全量开启。

练习一：为客服问答、代码修复和研究 Agent 设计 low、medium、high 三档预算，写出每档允许的工具、验证和最大时间。

练习二：一个 high effort 路径把成功率从 0.72 提高到 0.80，但单位成本变为 3 倍。分别从低风险聊天、支付风控和生产发布任务判断是否启用，并说明缺失哪些数据。

练习三：设计一个“无进展”检测器，至少使用重复工具调用、候选相似度和 verifier 分数三个信号。

### 13.10.1 预算曲线如何测出来

不要先给 `low`、`medium`、`high` 贴上固定 token 数，再假设档位有意义。更可靠的做法是固定任务集，逐步增加 reasoning token、候选数、工具次数或 deadline，得到一条质量—成本曲线。对每个点记录成功率、引用支持、验证通过、p95、单位成功成本和危险副作用。

例如，一组**合成教学数值**假设成功率从 0.70 提升到 0.82 只需要把预算从 2K 增到 6K，之后增到 20K 只提升到 0.83，那么 6K 附近可能是该任务集的收益拐点。它只演示如何读曲线，不是实测结果；这个拐点也不能直接迁移到其他模型或任务。

### 13.10.2 任务分类器的风险

动态预算通常需要一个任务分类器判断问题难度。分类器可能把“输入很长”误认为“推理很难”，也可能把短但高风险的支付确认判为简单任务。输入长度、任务难度、风险等级和输出长度是四个不同维度，不能用一个启发式代替。

分类器的错误应有安全上限：难度判断错，可以让任务转入更高预算或返回不确定；风险判断错，不能自动获得更多权限。对高风险动作，策略应直接要求证据和审批，而不是用 high reasoning 作为补偿。

### 13.10.3 一个可解释的升级例子

代码 Agent 初始获得 3 次工具调用和 4K reasoning budget。若第一次测试失败，且失败日志包含新的栈信息，系统可以增加一次诊断预算；若连续两次调用完全相同的测试并得到相同错误，则不再升级，而是返回人工或换用检索路径；若 patch 已通过隐藏测试但尚未确认写入，增加的不是思考 token，而是审批状态。

这个例子说明升级信号来自状态变化。没有新证据的重复计算不应被当作有效 reasoning；预算控制器必须同时看 token、工具观察、verifier 进展和剩余时间。

## 13.11 把预算从 token 数变成资源账本

把推理预算写成“最多生成 8K token”很容易，但这不是一个完整的资源模型。内部 reasoning token 只是模型计算的一部分。一次代码修复可能只生成 2K 个内部 token，却读取了 30 个文件、运行了三次测试，并让 GPU 在长上下文上重复做了大量 KV 读取；另一次数学题可能生成 8K token，但不调用任何外部工具。两者的成本和风险不能用一个数字比较。

设 `B_reason`、`B_tool`、`B_verify`、`B_time`、`B_side` 分别是推理 token 上限、工具调用/时间上限、验证额度、wall-clock 时限和允许的外部副作用次数；每个分量都是有限非负上限，需记录单位和硬边界。可以把每个请求的预算记录成向量：

```math
\mathbf{B}=(B_{\mathrm{reason}},B_{\mathrm{tool}},B_{\mathrm{verify}},B_{\mathrm{time}},B_{\mathrm{side}})
```

例如，某代码任务的预算可设为 `(4000 token, 3 次调用, 2 次验证, 30 秒, 0 次副作用)`；当前已消耗 `(2600 token, 2 次调用, 1 次验证, 12 秒, 0 次副作用)`，逐分量相减后剩余 `(1400 token, 1 次调用, 1 次验证, 18 秒, 0 次副作用)`。这里只能对向量同一位置使用相同单位做减法，不能把不同分量相加。`B_side=0` 明确表示不允许副作用；读取日志和发送邮件都可能只占一次工具调用，但它们对系统的影响完全不同。

单位成本也应按资源拆开。令 `N_r`、`N_v` 分别为非负整数的 reasoning token 数和 verifier 次数；`T_tool`、`T_gpu`、`T_human` 分别为以分钟计的有限非负工具、GPU 和人工耗时。系数 `c_r` 的单位为成本单位/token，其余系数分别为成本单位/分钟或成本单位/次；所有系数都应为有限非负数，换算结果使用共同成本单位。则：

```math
C= c_r N_r+c_t T_{\mathrm{tool}}+c_v N_v+c_g T_{\mathrm{gpu}}+c_h T_{\mathrm{human}}
```

系数可来自账单、容量成本或业务估值；每一项都换算到同一成本单位。以纯教学美元数例为例：`N_r=2000`、`c_r=0.000002 美元/token`，故推理项为 `$0.004`；`T_tool=1` 分钟且 `c_t=$0.10/分钟`，工具项 `$0.10`；`N_v=2`、`c_v=$0.03/次`，验证项 `$0.06`；`T_gpu=0.5` 分钟、`c_g=$0.50/分钟`，GPU 项 `$0.25`；`T_human=0.2` 分钟、`c_h=$1.00/分钟`，人工项 `$0.20`。总计 `C=$0.004+$0.10+$0.06+$0.25+$0.20=$0.614`。这些系数不是价格建议；真实账本要说明计费与容量归属。没有发生的资源取 0，未记录的资源保留缺失状态。若要计算单位成功成本，还需要定义同一批任务中的成功事件数；成功数为 0 时，单位成功成本未定义，而不是把总成本除以 1。这个式子防止系统只盯着模型 API 的 token 账单，忽略工具服务、排队、存储和人工审核。

对初学者来说，可以把预算理解成旅行预算：车票只是总花费的一部分，换乘、等待和行李也会消耗资源。对工程师来说，更关键的是定义每种资源的硬上限和软上限。软上限用于动态控制，硬上限用于防止异常循环；高风险副作用还要由独立权限系统控制，不能因为剩余 token 很多就自动放行。

## 13.12 在每个动作边界结算资源

策略控制器决定下一步做什么；资源账本则记录动作前预留了多少、动作后实际消耗多少。两个职责分开，才能避免把请求上限误当成真实成本。可靠的结算顺序是：调用前按硬上限预留资源，动作返回后按实际用量结算，取消或失败时也记录已发生的调用和时间，并释放未使用的预留。

| 账本时点 | 记录内容 | 例子 |
| --- | --- | --- |
| 动作开始前 | 请求的最大额度与本次预留量 | 一个任务最多 3 次工具调用，本次先预留 1 次 |
| 动作结束后 | 成功、失败或超时状态，以及已发生的真实用量 | 工具超时仍记录 1 次调用和 2.4 秒耗时 |
| 剩余额度 | 硬上限减去已结算量；未使用的预留归还 | 3 次上限、累计消耗 2 次，剩 1 次 |
| 无法观测时 | 保留 `unknown` / 缺失状态，不伪装成零 | 外部写操作超时后副作用状态未知 |

例如任务上限为 3 次工具调用、10 秒工具时间。第一次调用成功并耗时 1.6 秒，第二次超时并耗时 2.4 秒，则已消耗是 2 次和 4 秒，剩余额度是 1 次和 6 秒；超时不抹去已经发生的调用。若第二次调用可能已完成外部写入，但响应丢失，账本要把副作用状态记为未知，并由恢复/幂等策略处理，不能把它当作“零副作用”后直接重试。

## 13.13 用实验证明 high effort 的收益来源

高 effort 只有在固定条件下提高了目标能力，才有理由进入默认路径。固定配置对照至少应有 `always-low` 与 `always-high`；如加入 `dynamic-router` 组，应固定并记录应用层路由策略的版本。这里的 `dynamic-router` 指在线预算控制器，不是某个 provider 的 `adaptive` thinking mode。模型 revision、system prompt、工具权限、采样参数、任务集和成功定义必须相同。对 reasoning 模型，还要明确是否把内部 token、可见摘要和工具事件分别计量；控制环的设计见[第十八章](./18-adaptive-thinking与动态预算.md)。

任务集不能只取平均难度。应按任务类型、难度、风险、输入长度和是否可验证分层。每层报告成功率、单位成功成本、p95/p99、有效工具比例、verifier 通过率、重复副作用率和人工接管率。若 high effort 只帮助最难的 5% 样本，却让全部请求延迟翻倍，那么产品策略应是难题升级，而不是全量 high。

设固定评测协议下有 `n` 个独立试验，其中 `k` 个成功；`n` 为正整数，`k` 是 `0` 到 `n` 之间的整数。在正态近似条件下，成功率估计、标准误和 95% Wald 近似区间为：

```math
\hat p=\frac{k}{n},\qquad
\mathrm{SE}(\hat p)=\sqrt{\frac{\hat p(1-\hat p)}{n}},\qquad
\mathrm{CI}_{\mathrm{Wald}}\approx\hat p\pm1.96\,\mathrm{SE}(\hat p)
```

例如 `k=82`、`n=100`，则 `\hat p=82/100=0.82`，`SE=\sqrt{0.82×0.18/100}=\sqrt{0.001476}≈0.0384`；Wald 95% 区间约为 `0.82±1.96×0.0384=[0.745,0.895]`（四舍五入到三位小数）。`n=0` 时成功率和标准误都未定义；小样本或 `k=0/n` 时，Wald 近似可能失真，正式报告更适合使用 Wilson 区间或精确方法。这个例子只演示计算，不代表某模型的评测结果。

实际报告可使用 Wilson 区间或 bootstrap，而不是只比较两个小样本的百分点。还要做 paired replay：让同一个任务在不同预算下运行，并记录每一次从错误变正确、从无验证变有验证的转折。这样才能回答“收益来自更多计算、额外工具、验证器，还是只是随机采样差异”。

对外部 provider，只能依据公开文档确认具体字段的合同。第 13.4 节的官方资料支持对应 endpoint 公开的 effort、thinking budget 和输出上限差异；它们不揭示隐藏推理、搜索或 verifier 的完整内部机制，也不能据此推断其他模型使用相同策略。

## 13.14 把单位成功成本算出来

设同一批固定任务有 `N_success` 个通过预先定义的成功检查，累计资源成本为 `C_total`；两者都来自相同任务集、模型配置和计量口径，且 `N_success>0`。单位成功成本为：

```math
C_{\mathrm{success}}=\frac{C_{\mathrm{total}}}{N_{\mathrm{success}}}
```

继续用一组独立的合成教学数据（不与 §13.5 的代码修复表合并）：100 个请求中，低预算策略通过 72 个，累计成本为 144 个成本单位；则 `C_success=144/72=2`。中预算策略通过 84 个、累计成本 210，单位成功成本为 `210/84=2.5`；高预算通过 86 个、累计成本 430，单位成功成本为 `430/86=5`。这些数字表示多花预算后成功率略有提高，但每个成功任务的平均成本也升高；它们不是实测结果。若 `N_success=0`，单位成功成本未定义，应如实报告“没有成功样本”，不能用 1 替代分母。

还要检查硬约束。假设三种策略的 P95 延迟分别为 2、5、13 秒，服务目标要求 P95 小于 8 秒，则高预算策略不能进入这个服务的默认可用集合，即使它的成功率最高。SLO 不满足时，不能用单位成功成本或效用分数抵消；是否对少量特定任务另行升级，属于第十八章的动态控制问题。

## 13.15 预算和权限不能共用一个开关

增加 reasoning token 只增加计算，不应自动获得更多文件、网络或写入权限。一个常见的错误是把 `high effort` 和“允许执行更多工具”绑定在一起，导致模型在困难任务上同时拥有更长思考和更大的副作用范围。安全设计应分别维护 compute budget、tool budget、data budget 和 approval budget。

例如，模型可以使用高预算读取三份公开文档，但仍然没有发送邮件权限；也可以拥有发送邮件权限，却必须在低预算摘要通过人工确认后执行。预算状态机和授权状态机分开，才能在超时、重试和恢复时知道哪些资源已经消耗，哪些动作从未授权。

## 13.16 预算实验的最小报告

每次比较不同 effort，至少固定模型 revision、prompt、工具集合、最大输出、采样策略、并发、缓存和评测集。报告平均值之外，还要给出 p50/p95、成功任务成本、停止位置、升级比例、工具调用次数和失败原因。对同一任务做 paired replay，才能减少题目难度差异造成的假象。

如果高预算只让答案更长，却没有提高引用正确率、隐藏测试通过率或恢复成功率，就应降低默认预算或改进 verifier。推理预算的收益必须落到任务结果，而不是落到可见的思考文本长度。

## 13.17 预算消耗的可观测账本

服务端应把输入 token、输出 token、reasoning token（若接口提供）、候选数、工具时间、验证时间、排队时间和人工等待分开记录。reasoning token 的字段名、可见性和是否计入输出用量因供应商、模型与 API surface 而异；没有返回该字段表示“未观测”，不能记成 0。否则“high 比 low 贵多少”只能靠账单猜测，无法知道增加的预算到底用于模型计算还是等待资源。

## 13.18 预算用尽时的最后一步

达到预算上限时，系统应先保存当前状态，尝试做一次轻量验证，再选择返回部分结果、明确不确定、转人工或安全降级。不能直接截断在一个半成品 JSON、未闭合代码块或未经确认的外部动作上。预算边界本身也是协议边界。

## 13.19 小结与资料边界

推理预算的本质是把额外计算花在可验证的进展上。它可以表现为内部 token，也可以表现为候选、搜索、工具、测试和恢复。好的系统同时管理质量、成本、时间和风险，并且将预算与权限分离。

test-time compute 的策略分类和质量—成本曲线已在[第七章](./07-test-time-compute-scaling.md)主讲；在线反馈控制见[第十八章](./18-adaptive-thinking与动态预算.md)。本章补充请求级 API 控制字段、资源计量与成本账本。具体 API 字段是有日期的接口合同，而不是统一算力刻度。

本章使用的官方接口资料（查阅日期：2026-11-13）：[OpenAI Reasoning models](https://developers.openai.com/api/docs/guides/reasoning)、[Anthropic Effort](https://platform.claude.com/docs/en/build-with-claude/effort)、[Anthropic Extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)、[Google Gemini thinking](https://ai.google.dev/gemini-api/docs/generate-content/thinking)。文档只支持其公开字段及适用范围；部署时须按目标模型、API surface 和 revision 复核。
