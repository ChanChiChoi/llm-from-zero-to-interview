# 第 8 章 Agent 项目实战（2）：ReAct 状态循环

一次工具调用只能完成一步。用户问“6,000 元报销是否需要审批”，系统可能要先查询当前制度，再比较金额，最后引用对应条款。Agent 的工作不是“让模型多说几轮”，而是把每一步的目标、动作、外部观察和终止条件接起来。

如果模型生成“超过 5,000 元，需要审批”，它可能只生成了一句话；如果执行器真的查询到生效版本的条款、计算器返回 true，系统再根据两条证据组织结论，我们才有一条可检查的任务轨迹。

## 8.2 从一次调用走到有界循环

### 8.2.1 用轨迹表示一轮 Agent 任务

ReAct 把模型决策与环境观察交替组织。[原论文](https://arxiv.org/abs/2210.03629)研究模型交错生成任务相关轨迹与动作、再从外部知识库或环境取得观察的做法。对一条包含 $K$ 次动作的轨迹，可记为：

$$
\tau=(q,a_1,o_1,a_2,o_2,\ldots,a_K,o_K,y)
$$

其中 $q$ 是用户目标，$a_k$ 是第 $k$ 步提出的动作，$o_k$ 是执行器返回的观察，$y$ 是最终结果。状态可以写成 $s_{k+1}=s_k\oplus(a_k,o_k)$：把新动作和观察加入已有状态，再让模型基于更新后的状态决定下一步。

这里的“思考”指决定下一步的过程，不意味着必须展示或保存完整内部推理。系统可以记录结构化计划、工具提议、校验结果和必要的理由代码；决定外部世界发生什么的，是执行器返回的观察和回执。

一次报销判断可以按顺序走：目标是判断是否触发审批；先查询当前制度；观察到“单笔超过 5,000 元需额外审批”；再调用受限比较器判断 6,000 是否大于 5,000；得到真后，给出带制度来源的结论。每个 observation 都来自外部工具，不能由模型自己补写。

### 8.2.2 把状态存成事件，而不是一段不断变长的文字

最简单的 demo 会把历史拼成 scratchpad。短任务容易看懂；长任务则不断重复旧文本、增加 token 成本，也难以定位事实来自用户、模型还是工具。更稳妥的状态至少区分目标、事件、已确认事实、待解决问题、预算和终态。

> 代码状态：静态审阅，未运行（2026-10-30）

```python
state = {
    "task_id": "expense_001",
    "user_query": "6000 元报销是否需要额外审批？",
    "events": [
        {
            "kind": "tool_call",
            "call_id": "call_1",
            "name": "search_docs",
            "arguments": {"query": "报销金额审批规则"},
        },
        {
            "kind": "tool_result",
            "call_id": "call_1",
            "status": "ok",
            "content": "超过 5000 元需要部门负责人额外审批。",
        },
    ],
    "budget": {"steps": 2, "tool_calls": 1, "tokens": 180},
}
print("event_kinds=", [event["kind"] for event in state["events"]])
```


事件保存后，系统可以只把下一步需要的信息重新组织给模型，而不是无条件复制全部历史。比如“当前制度版本 P-17 的阈值是 5,000 元”应附来源与版本；“需要审批”是基于金额比较得到的结论。二者证据类型不同，不能只留一个混合摘要。

### 8.2.3 Tool Calling 是协议，ReAct 是循环

| 概念 | 解决的问题 | 典型内容 |
| --- | --- | --- |
| Tool Calling | 如何表达并执行一条工具调用 | 工具名、参数、调用 ID、工具结果 |
| ReAct | 如何把多步动作组织成任务 | 状态更新、行动、观察、重规划、停止 |

可以在每一步都使用结构化 Tool Calling：模型提出调用，应用校验并执行，再把结果写入状态，模型基于新状态决定继续还是结束。这样保留多步任务能力，同时避免依赖脆弱的自由文本解析。

如果模型没有原生工具协议，教学实现可能让它输出：

> Action: search_docs  
> Action Input: 报销金额审批规则

解析器再把文本转成动作：

> 代码状态：静态审阅，未运行（2026-10-30）

```python
import re


def parse_react(text):
    if not isinstance(text, str):
        return {"kind": "error", "raw": text}
    final = re.search(r"Final Answer:\s*(.*)", text, flags=re.S)
    if final:
        return {"kind": "final", "answer": final.group(1).strip()}
    action = re.search(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)", text)
    argument = re.search(r"Action Input:\s*(.*)", text, flags=re.S)
    if action and argument:
        return {
            "kind": "action",
            "name": action.group(1),
            "input": argument.group(1).strip(),
        }
    return {"kind": "error", "raw": text}


print(parse_react("Action: search_docs\nAction Input: 年假制度"))
print(parse_react("Final Answer: 资料不足。"))
```


自由文本会受空格、标点、代码块和多行参数影响；解析失败必须进入明确错误状态，不能把不完整字段猜成有效调用。能用结构化协议时应优先使用结构化协议；ReAct 的重点是循环和状态，并非某种文本格式。

### 8.2.4 给循环设硬预算和终止状态

循环需要有限预算，例如最大决策步数、工具调用数、累计输入与输出 token、总耗时和金额成本：

$$
B=(B_{\mathrm{steps}},B_{\mathrm{calls}},B_{\mathrm{tokens}},B_{\mathrm{time}},B_{\mathrm{cost}})
$$

执行器每一步检查已用预算；触及任一上限，任务进入 budget_exhausted。Prompt 中写“请不要调用太多工具”只是软提示，不能代替程序里的硬限制。

循环可以因目标完成、证据不足、权限拒绝、工具明确失败、外部状态未知、用户取消、无进展或预算耗尽而停止。不要把这些终态压成“成功/失败”两个值：final 表示目标条件有证据支持，refused 表示策略阻断，tool_error 表示工具明确失败，unknown 表示副作用状态不能确认，partial 表示只完成部分目标。

下面的无依赖 demo 用固定 mock policy 展示查询、比较和总结；它只帮助看循环顺序，不是实际模型评估。

> 代码状态：静态审阅，未运行（2026-10-30）

```python
import ast
import math
import operator
import re


DOC = "报销制度：单笔超过 5000 元需要部门负责人额外审批。"


def search_docs(query):
    return DOC if "报销" in query or "审批" in query else "未找到资料"


def compare(expression):
    operators = {ast.Gt: operator.gt, ast.GtE: operator.ge}
    tree = ast.parse(expression, mode="eval")
    node = tree.body
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return "错误：只允许单个比较"
    values = [node.left.value, *[item.value for item in node.comparators]] if all(
        isinstance(item, ast.Constant) for item in [node.left, *node.comparators]
    ) else []
    if not values or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not math.isfinite(value))
        for value in values
    ):
        return "错误：只允许数字"
    fn = operators.get(type(node.ops[0]))
    if fn is None:
        return "错误：不支持的比较"
    return str(fn(values[0], values[1]))


def mock_policy(scratchpad):
    if "Observation:" not in scratchpad:
        return "Action: search_docs\nAction Input: 报销金额审批规则"
    if "compare" not in scratchpad:
        return "Action: compare\nAction Input: 6000 > 5000"
    return "Final Answer: 需要额外审批；6000 元超过 5000 元。"


TOOLS = {"search_docs": search_docs, "compare": compare}


def parse(text):
    if not isinstance(text, str):
        return {"kind": "error"}
    final = re.search(r"Final Answer:\s*(.*)", text, flags=re.S)
    if final:
        return {"kind": "final", "answer": final.group(1).strip()}
    action = re.search(r"Action:\s*(\w+)", text)
    argument = re.search(r"Action Input:\s*(.*)", text)
    if not action or not argument:
        return {"kind": "error"}
    return {"kind": "action", "name": action.group(1), "input": argument.group(1)}


def run(max_steps=4, policy=mock_policy):
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer")
    scratchpad = ""
    events = []
    seen = set()
    for step in range(max_steps):
        if not callable(policy):
            raise TypeError("policy must be callable")
        model_output = policy(scratchpad)
        parsed = parse(model_output)
        events.append({"kind": "model", "step": step, "parsed": parsed})
        if parsed["kind"] == "final":
            events.append({"kind": "final", "answer": parsed["answer"]})
            return {"answer": parsed["answer"], "events": events, "stop": "final"}
        if parsed["kind"] != "action" or parsed["name"] not in TOOLS:
            return {"answer": "动作解析或工具校验失败。", "events": events, "stop": "error"}
        key = (parsed["name"], parsed["input"])
        if key in seen:
            return {"answer": "检测到重复动作。", "events": events, "stop": "repeat"}
        seen.add(key)
        observation = TOOLS[parsed["name"]](parsed["input"])
        events.append({
            "kind": "observation",
            "step": step,
            "name": parsed["name"],
            "content": observation,
        })
        scratchpad += model_output + "\nObservation: " + observation + "\n"
    return {"answer": "达到步数上限，任务未完成。", "events": events, "stop": "budget"}


result = run()
short = run(max_steps=1)
print("answer=", result["answer"])
print("stop=", result["stop"])
print("observation_count=", sum(event["kind"] == "observation" for event in result["events"]))
print("short_stop=", short["stop"])
```


max_steps=1 的结果应是未完成，而不是把尚未比较的任务包装成答案。mock policy 也不代表真实模型能稳定选对工具，教学中的状态转换与生产 Agent 的任务质量要分开评价。

### 8.2.5 “有进展”要能被检查

每轮调用工具不等于任务向前走了一步。一次观察只有在补齐字段、关闭子目标、改变可信外部状态，或让证据从不足变为充分时，才算有进展。如果 Agent 连续对同一查询得到同一空结果，再提出相同调用，调用次数增加了，信息却没有增加。

执行器可以对规范化的“工具名 + 参数”计算动作指纹，再结合观察摘要和无进展计数识别重复。允许一次有理由的查询改写，和无限重复同一请求，是两件事。重复判定还要考虑动作语义：同一个只读查询有时可以重试，但扣款、退款或发邮件不能因为“还没看到回执”而无条件重发。

### 8.2.6 持久化状态，让任务能在中断后恢复

长任务要保存当前业务状态，而不只是整段对话：目标及版本、已确认事实及来源、待办子目标、最近一次工具提议、外部执行状态、累计预算和停止原因。服务重启后，恢复程序先读取这些字段和最近回执，再判断应继续、查状态还是交给人工。

查询制度失败通常是只读操作，可以在期限内受限重试；写请求已经送到外部服务但响应丢失，就要用原幂等键查询结果。不能因为模型没看到回执，就生成新的幂等键再做一遍。如果工具已完成而模型尚未消费结果，可以把原 call/result 事件接回轨迹，并区分“执行器已完成”和“Agent 已观察”。

用户取消任务也不一定撤销已发出的副作用。系统需分别记录取消请求、执行器是否停止、外部动作是否提交及最终能否确认状态。可恢复意味着恢复真实业务状态，不是把同一段 prompt 重新发送一次。

持久化边界也要有选择：保存足以恢复任务的目标、结构化事件、工具回执、预算和终态，不必把每轮自由文本推理都当成业务状态。不可逆动作应分别记录执行意图和外部回执；若服务在两者之间崩溃，恢复流程根据幂等键查询外部状态，而不是猜测最后一步。只读任务通常可以从最近一个已确认观察继续；写任务则要先解决“动作是否已经生效”，再让模型规划后续。

### 8.2.7 用一个可核算例子估算循环成本

假设三次模型调用分别读入 700、900、1,050 token，分别输出 120、45、80 token；模型耗时为 420、580、360 ms，两次工具耗时为 90 和 2 ms。累计输入是 $700+900+1{,}050=2{,}650$ token，输出是 $120+45+80=245$ token，模型耗时 1,360 ms，工具耗时 92 ms，整轮约 1.452 秒。

这是一个假设账本，不是模型或系统的实测成绩。后续请求往往再次携带问题、工具观察和状态摘要；只记录最后一次请求的 1,050 token 会低估整轮输入。实际比较还要分别记录模型等待、工具等待、排队、重试和失败。

下一篇会把循环里的“允许执行”变成具体控制：谁能读什么、哪些参数可执行、风险动作如何确认，以及超时后怎样避免重复副作用。
