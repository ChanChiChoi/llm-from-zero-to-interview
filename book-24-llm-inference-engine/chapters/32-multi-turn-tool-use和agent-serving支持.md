# 第 32 章 Multi-turn、Tool Use 和 Agent Serving 支持

上一章讲了 Speculative Decoding：它通过 draft/verify/accept 减少主模型 decode 轮数，主要优化 TPOT 和 output tokens/s。

本章回到 SGLang 的问题意识：复杂 LLM 应用往往不是一次 `prompt -> answer`，而是多轮对话、工具调用、外部环境交互、agent 轨迹和多步程序执行。

这些场景从应用层看是 agent framework，从 runtime 层看则是：多次 generation、共享上下文、结构化工具参数、工具结果回灌、动态分支、状态管理、cache 复用和调度公平性。

一句话概括：

> Multi-turn、tool use 和 agent serving 的核心不是“模型会调用工具”这么简单，而是 runtime 能否高效、可靠地执行多步有状态 LLM 程序，并在每一轮复用上下文、约束工具格式、管理请求状态和控制尾延迟。

## 32.1 本章目标

读完本章，你应该能讲清：

1. Multi-turn chat 在 runtime 中为什么不是普通单请求。
2. Tool use 的请求、生成、解析、执行、回灌流程是什么。
3. SGLang tool parser、structured output、tool_choice 分别解决什么问题。
4. Agent serving 为什么会产生多次 generation 和树状 prefix sharing。
5. RadixAttention 如何帮助多轮和 agent trajectory 复用 KV cache。
6. Scheduler 在 agent workload 中要面对哪些新问题。
7. 面试中如何从 serving runtime 视角解释 agent 支持。

## 32.2 从单轮 chat 到多轮状态

单轮 chat 很简单：

```text
user prompt -> model answer
```

多轮 chat 则是：

```text
system message
user turn 1
assistant turn 1
user turn 2
assistant turn 2
user turn 3
assistant turn 3
...
```

每一轮请求都要包含历史上下文，模型才能理解当前问题。

从 OpenAI-compatible API 看，请求是 `messages` 列表。

从 runtime 看，关键是这些 messages 会被 chat template 渲染成 token 序列：

```text
messages
  -> chat template
  -> prompt text
  -> token ids
  -> prefill / prefix cache / decode
```

多轮的性能机会在于：下一轮通常共享上一轮完整历史。

例如：

```text
turn 2 prompt: [S U1 A1 U2]
turn 3 prompt: [S U1 A1 U2 A2 U3]
```

`turn 3` 可以复用 `turn 2` 已经计算过的大段 KV cache。

这就是 RadixAttention 在 multi-turn chat 中的价值。

## 32.3 多轮 chat 的 runtime 状态

一个多轮会话至少有两类状态。

应用层状态：

1. session id。
2. message history。
3. 用户身份。
4. 工具配置。
5. 业务上下文。
6. 安全策略。

Runtime 层状态：

1. tokenized history。
2. prefix cache 命中路径。
3. KV cache 引用。
4. 当前 request state。
5. sampling params。
6. stop 条件。
7. streaming offset。
8. grammar state。

很多系统会把应用层状态放在上层服务或 agent framework 中，而 SGLang Runtime 负责高效执行每次 generation。

但 runtime 仍然需要看到足够稳定的 token prefix，才能复用 cache。

## 32.4 Chat template 为什么影响 cache

多轮对话的 prefix cache 命中依赖 token prefix 完全一致。

如果 chat template 变化，token 序列就变化。

例如：

```text
<user>Hello</user><assistant>Hi</assistant>
```

和：

```text
User: Hello
Assistant: Hi
```

语义相同，但 token 序列不同，cache 无法共享。

工程建议：

1. 同一模型使用稳定 chat template。
2. 工具说明顺序稳定。
3. system prompt 不要插入随机字段。
4. 历史裁剪策略稳定。
5. 上层不要每轮改变无关格式。

Multi-turn serving 的 cache 命中率，很多时候不是 runtime 算法问题，而是 prompt 拼接稳定性问题。

## 32.5 Tool use 的基本流程

Tool use 指模型在回答过程中选择调用外部工具。

典型流程：

```text
1. 用户提问
2. 请求中包含可用 tools schema
3. 模型生成 tool call
4. runtime 或上层解析 tool name 和 arguments
5. 应用层执行工具
6. 工具结果作为 role=tool message 回灌
7. 模型基于工具结果继续生成最终回答
```

展开成 messages：

```text
user: What's the weather in Boston?
assistant: tool_call get_weather({"city":"Boston"})
tool: {"temperature":"85F","condition":"cloudy"}
assistant: It is 85F and cloudy in Boston.
```

注意：工具执行通常不在 LLM runtime 内部完成。

Runtime 负责生成和解析工具调用，上层应用负责真正执行工具、处理权限、超时和结果回灌。

## 32.6 Tool schema 是什么

工具通常用 schema 描述。

例如：

```json
{
  "type": "function",
  "function": {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
      },
      "required": ["city", "unit"]
    }
  }
}
```

这个 schema 有三层作用：

1. 告诉模型有哪些工具。
2. 约束工具参数格式。
3. 给 parser 或 validator 提供校验依据。

在 SGLang 中，tool use 可以结合 tool parser、chat template、structured output 和 tool choice 来实现更可靠的函数调用。

## 32.7 Tool parser 的作用

不同模型的工具调用格式不同。

有的输出 JSON。
有的输出特殊 tag。
有的输出 Python-like function call。
有的用模型专属 token。

Tool parser 的作用是把模型原始输出解析成统一的 tool call 对象：

```text
raw generated text
  -> tool parser
  -> normal_text
  -> tool_calls: [{name, arguments}]
```

SGLang 文档中列出多种 parser，例如 DeepSeek、GLM、GPT-OSS、Kimi、Llama、Mistral、Qwen、pythonic 等。

这说明一个工程事实：tool calling 不是完全统一的模型能力，它强依赖模型训练格式、chat template 和 parser。

如果 parser 和模型格式不匹配，工具调用可能解析失败。

## 32.8 Tool choice

Tool choice 用来控制模型是否必须调用工具，或必须调用哪个工具。

常见模式：

1. auto：模型自行决定是否调用。
2. required：必须至少调用一个工具。
3. specific function：必须调用指定函数。
4. none：不允许调用工具。

从 runtime 视角看，tool choice 可以被转化为更强的输出约束。

例如 required 模式要求模型输出 tool call，而不是普通文本。

SGLang 文档提到 tool_choice 可通过 EBNF grammar 等方式实现更可靠的工具调用行为。

这和第 30 章 constrained decoding 是同一条线：用 grammar 约束模型输出，而不是只靠 prompt 希望模型听话。

## 32.9 Tool call 的结构化约束

工具调用最怕格式不稳定。

例如你希望：

```json
{"city":"Boston","unit":"fahrenheit"}
```

模型却输出：

```text
city is Boston, use Fahrenheit please
```

人能看懂，程序不好执行。

因此 tool call 通常需要 structured output 支持：

1. JSON schema 约束 arguments。
2. Structural tag 约束 begin/end 标签。
3. EBNF 约束 tool_choice。
4. Parser 解析模型专属格式。
5. Validator 做最终参数校验。

推荐理解为分层：

```text
prompt / chat template: 告诉模型工具语义
constrained decoding: 限制工具调用格式
tool parser: 把文本解析成对象
validator: 校验业务参数
application: 执行工具
```

不要把所有责任都放在模型输出上。

## 32.10 工具执行不等于模型推理

Tool use 往往跨越 LLM runtime 和应用系统边界。

LLM Runtime 负责：

1. 接收 tools schema。
2. 渲染 chat template。
3. 生成 tool call。
4. 支持 constrained decoding。
5. 解析或返回 tool call。
6. 继续处理工具结果后的 generation。

应用层负责：

1. 判断工具是否允许调用。
2. 执行 HTTP、数据库、搜索、代码解释器等工具。
3. 处理工具超时和错误。
4. 清洗工具结果。
5. 把结果作为 tool message 回灌。
6. 做审计和安全控制。

这条边界很重要。

如果把真实工具执行塞进推理 runtime，runtime 会变得难以隔离、难以扩缩容，也更难保证安全。

## 32.11 Agent loop 是什么

Agent loop 是多轮 tool use 的泛化。

典型 ReAct-like loop：

```text
while not done:
    model generates thought/action
    if action is tool call:
        execute tool
        append observation
    else if action is final answer:
        return answer
```

从 runtime 看，它会产生多次 generation：

```text
gen action 1
tool result 1
gen action 2
tool result 2
gen final answer
```

每次 generation 都共享之前的 trajectory。

这正是 SGLang 强调复杂 LLM programs 的原因。

## 32.12 Agent trajectory 的 prefix sharing

Agent trajectory 通常长这样：

```text
system prompt
tool descriptions
user task
thought 1
action 1
observation 1
thought 2
action 2
observation 2
final answer
```

后续每一步都共享前面的所有历史。

如果每次工具返回后都从头 prefill，成本很高。

RadixAttention 可以复用：

```text
[system prompt][tool descriptions][user task][thought/action/observation history]
```

只对新增 observation 和下一轮 user/tool message 后的 suffix 做计算。

对于多分支 agent search，也会出现树状共享：

```text
root task
  -> plan A
      -> tool result A1
      -> tool result A2
  -> plan B
      -> tool result B1
```

Radix tree 可以自然表达这些共享路径。

## 32.13 Agent serving 和普通 chat serving 的差异

普通 chat serving：

```text
一个用户请求 -> 一个模型回答
```

Agent serving：

```text
一个用户任务 -> 多次模型调用 + 多次工具调用 + 动态停止
```

差异包括：

1. 请求生命周期更长。
2. 生成次数不固定。
3. 工具耗时不可控。
4. 上下文不断增长。
5. 中间状态需要保存。
6. 可能有并行分支。
7. 失败点更多。
8. cache reuse 机会更多。

因此 agent serving 不能只按单次 chat completion 的 QPS 来评估。

要看的是整个 task 的端到端成功率、工具轮数、总 tokens、总延迟和成本。

## 32.14 Scheduler 面临的新问题

Agent workload 会给 scheduler 带来新问题。

第一，请求会分阶段到达。

模型生成 tool call 后，请求会等待工具结果；工具结果回来后，又产生下一次 generation。

第二，工具耗时不稳定。

一个工具可能 50ms 返回，也可能 5s 超时。

第三，agent 轨迹长短差异大。

有的任务 1 轮完成，有的任务 10 轮还没完成。

第四，cache locality 很重要。

同一个 agent session 的后续 generation 如果打到同一个 runtime，更容易命中 prefix cache。

第五，公平性更复杂。

一个长 agent 任务不能无限占用调度资源，短 chat 请求也不能被完全饿死。

所以 agent serving 常需要：

1. session-aware routing。
2. max tool rounds。
3. max context length。
4. max total tokens。
5. per-step timeout。
6. priority queue。
7. cache-aware scheduling。

## 32.15 Session-aware routing

在多副本 serving 中，同一个 session 如果每轮都打到不同 runtime，prefix cache 命中会变差。

例如：

```text
turn 1 -> replica A
turn 2 -> replica B
turn 3 -> replica C
```

每个副本都有自己的 GPU KV cache，后续轮次可能无法复用前面轮次的 cache。

Session-aware routing 的思路是：

```text
同一个 session 尽量路由到同一个 runtime replica
```

好处：

1. 提高 RadixAttention 命中率。
2. 降低多轮 TTFT。
3. 减少重复 prefill。

代价：

1. 负载均衡更难。
2. 热门 session 可能压垮单副本。
3. 副本故障时 cache 丢失。
4. 扩缩容时迁移复杂。

这是平台层和 runtime 层的协同问题。

## 32.16 Context growth 和裁剪

多轮和 agent 的上下文会不断增长。

问题包括：

1. prompt 越来越长。
2. prefill 和 cache 成本上升。
3. KV cache 占用上升。
4. 超过模型 max context。
5. 工具结果可能很长。
6. 无关历史影响模型质量。

常见处理策略：

1. 滑动窗口保留最近历史。
2. 总结旧历史。
3. 只保留关键 tool results。
4. 对工具结果做压缩。
5. RAG 化历史。
6. 设置 max tool rounds 和 max tokens。

但裁剪会影响 prefix cache。

如果每轮裁剪策略不稳定，token prefix 会变化，cache 命中下降。

所以裁剪策略要尽量确定、可复现。

## 32.17 Tool result 回灌

工具结果通常作为 `role=tool` message 回到模型上下文。

例如：

```text
assistant tool_call: get_weather({"city":"Boston"})
tool result: {"temperature":"85F"}
assistant final: Boston is 85F.
```

回灌时要注意：

1. 工具结果长度限制。
2. 工具结果格式稳定。
3. 错误信息如何表达。
4. 是否暴露敏感字段。
5. 是否需要引用来源。
6. 多工具结果顺序。

工具结果是 prompt 的一部分，会进入 tokenizer 和 KV cache。

如果工具结果含随机字段、时间戳、trace id，prefix cache 复用会变差。

## 32.18 Tool errors 和恢复

工具调用可能失败。

失败类型：

1. 工具不存在。
2. 参数校验失败。
3. 权限不足。
4. 网络超时。
5. 下游返回 500。
6. 结果为空。
7. 结果过长。

处理方式：

1. 把错误作为 tool message 回灌，让模型决定下一步。
2. 直接终止 agent。
3. 重试工具。
4. 切换备用工具。
5. 请求用户补充信息。

Serving runtime 需要支持 abort、timeout 和资源清理。

如果 agent 等工具时不释放或暂停相应资源，很容易造成长尾和泄漏。

## 32.19 Parallel tool calls

有些模型或 agent 会一次提出多个工具调用。

例如：

```text
call get_weather(city="Paris")
call get_hotels(city="Paris")
call get_flights(destination="Paris")
```

应用层可以并行执行这些工具。

Runtime 视角的影响：

1. 模型需要输出多个 tool calls。
2. Parser 要能处理多个 calls。
3. 工具结果回灌顺序要稳定。
4. 后续 generation 共享相同前缀。
5. 工具等待阶段不应占用 GPU decode slot。

Parallel tool calls 能降低 agent wall-clock latency，但会增加外部系统压力和错误处理复杂度。

## 32.20 Agent search 和分支

有些 agent 不只线性执行，还会搜索多个候选计划。

例如：

```text
task
  -> plan A
      -> execute A
      -> score A
  -> plan B
      -> execute B
      -> score B
  -> choose best
```

这会产生多分支 generation。

SGLang 的优势在这里更明显：

1. 多个分支共享 root prompt。
2. 每个分支可能继续扩展。
3. RadixAttention 可以复用共享前缀。
4. Scheduler 可以批量执行多个 branch 的 prefill/decode。
5. Structured output 可以约束 plan/score 格式。

这类 workload 比普通 chat 更接近 SGLang 论文里的复杂 LLM programs。

## 32.21 Agent serving 的指标

Agent serving 不能只看单次请求 QPS。

建议看：

任务级指标：

1. task success rate。
2. task E2E latency。
3. tool rounds per task。
4. model calls per task。
5. total input tokens per task。
6. total output tokens per task。
7. cost per task。

Runtime 指标：

1. TTFT per generation。
2. TPOT。
3. prefix cache hit length。
4. RadixAttention saved tokens。
5. waiting queue length。
6. running requests。
7. KV cache usage。
8. grammar mask latency。

Tool 指标：

1. tool call count。
2. tool latency。
3. tool error rate。
4. timeout rate。
5. retry count。

这些指标要分层看，否则很难判断慢在模型、runtime、工具还是 agent 逻辑。

## 32.22 安全边界

Tool use 和 agent serving 必须考虑安全。

常见风险：

1. Prompt injection 诱导调用危险工具。
2. 工具参数包含恶意输入。
3. 模型泄露工具结果中的敏感数据。
4. 工具调用越权。
5. 无限循环调用工具。
6. 代码执行工具造成破坏。
7. SSRF 或外部请求滥用。

Runtime 层能做一部分格式约束，但不能替代安全策略。

应用层必须做：

1. 工具 allowlist。
2. 参数 validation。
3. 权限校验。
4. 速率限制。
5. 沙箱执行。
6. 审计日志。
7. 用户确认机制。

不要因为有 structured output，就默认工具调用是安全的。

## 32.23 和 speculative decoding 的关系

Agent 场景中 speculative decoding 可能有帮助，也可能收益有限。

有帮助的情况：

1. 每轮输出较长。
2. 最终答案较长。
3. 工具结果总结较长。
4. 低 temperature、格式稳定。

收益有限的情况：

1. 每轮只输出短 tool call。
2. 工具等待时间主导 E2E latency。
3. 高不确定性规划，draft 接受率低。
4. structured output 和 tool constraints 使 draft 更复杂。

所以 agent serving 中要分开看：

```text
模型 decode 时间
工具执行时间
调度排队时间
prefix cache 命中
```

如果主要慢在工具 API，speculative decoding 解决不了根因。

## 32.24 和 structured output 的关系

Tool use 几乎天然需要 structured output。

原因是工具参数必须被程序解析。

常见组合：

1. JSON schema 约束 function arguments。
2. Structural tag 包住 tool call。
3. Tool parser 解析模型原生格式。
4. Tool choice 约束是否必须调用。
5. Validator 做最终业务校验。

Structured output 让工具调用格式更稳定，但不能保证工具选择一定正确。

例如模型可能合法地调用 `get_weather`，但用户其实问的是酒店价格。

所以还需要工具选择策略、系统 prompt、业务规则和评测。

## 32.25 和 RadixAttention 的关系

Multi-turn、tool use 和 agent serving 都是 RadixAttention 的典型收益场景。

原因：

1. 多轮历史共享前缀。
2. 工具说明通常固定。
3. Agent trajectory 逐步增长。
4. 多分支计划共享 root。
5. Self-consistency 和 agent search 共享问题描述。

可以把三者关系记成：

```text
Multi-turn 提供长共享历史
Tool use 提供固定工具说明和结构化调用
Agent serving 提供多步/分支/回灌轨迹
RadixAttention 负责复用这些共享 prefix 的 KV cache
```

如果 prompt template 稳定，收益会更明显。

## 32.26 常见工程坑

坑一：工具说明每轮顺序变化。

后果：token prefix 变化，cache 命中率下降。

坑二：工具结果太长。

后果：上下文快速膨胀，KV cache 占用上升。

坑三：tool parser 和模型格式不匹配。

后果：模型生成了工具调用，但系统解析不出来。

坑四：streaming tool call 参数碎片处理错误。

后果：增量 arguments 拼接失败，JSON parse 错误。

坑五：工具等待期间占用 GPU slot。

后果：running queue 被阻塞，吞吐下降。

坑六：agent 无限循环。

后果：成本失控，用户等待时间过长。

坑七：跨副本路由打散 session。

后果：多轮 prefix cache 命中率低。

## 32.27 面试官会怎么问

问题一：SGLang 为什么适合 agent serving？

回答要点：Agent workload 包含多次 generation、工具调用、结构化输出、多轮历史和分支控制流。SGLang 的 frontend/runtime 协同、RadixAttention、structured output、scheduler 能更好地表达和执行这些复杂 LLM programs。

问题二：Tool use 的完整链路是什么？

回答要点：请求带 tools schema，模型生成 tool call，tool parser 解析 name/arguments，应用层执行工具，把结果作为 tool message 回灌，模型继续生成最终答案。

问题三：Tool parser 解决什么问题？

回答要点：不同模型工具调用格式不同，parser 把模型原始输出转换成统一的 tool call 对象，便于应用层执行工具。

问题四：Multi-turn chat 为什么能受益于 RadixAttention？

回答要点：后续轮次共享之前完整历史，RadixAttention 可以复用历史 token 对应 KV cache，只计算新增 turn 的 suffix。

问题五：Agent serving 的性能瓶颈怎么拆？

回答要点：拆成模型 TTFT/TPOT、prefix cache 命中、scheduler queue、KV cache、工具执行延迟、工具错误率、agent 轮数和总 tokens。

## 32.28 标准回答模板

如果面试官问“SGLang 如何支持 multi-turn、tool use 和 agent serving”，可以这样回答：

```text
从 runtime 视角看，multi-turn、tool use 和 agent serving 都是复杂 LLM program。它们不是一次简单 chat completion，而是多次 generation、共享上下文、结构化工具调用、工具结果回灌和动态停止。

Multi-turn chat 中，后续轮次会共享前面完整历史。只要 chat template 和消息拼接稳定，RadixAttention 可以复用历史 token 的 KV cache，只对新增用户消息和后续生成做计算，从而降低多轮 TTFT。

Tool use 中，请求会携带 tools schema，模型生成 tool call，SGLang 可以通过 tool parser 把模型原始输出解析成统一的 tool name 和 arguments，也可以结合 JSON schema、structural tag、EBNF 和 tool_choice 做 constrained decoding，提高工具调用格式可靠性。真正的工具执行通常在应用层完成，结果再作为 tool message 回灌给模型。

Agent serving 则会把这个过程循环起来：模型生成 action，外部工具返回 observation，再继续生成下一步。这个轨迹会不断增长，也可能出现多分支搜索。SGLang 的价值在于用 RadixAttention 复用共享 trajectory，用 scheduler 管理多次 generation，用 structured output 保证工具参数可解析，并通过 runtime 指标拆解模型、cache、工具和 agent 逻辑的瓶颈。
```

## 32.29 小练习

1. 画出一次 tool use 的完整链路：user、model、tool parser、tool execution、tool result、final answer。
2. 解释为什么 tool schema 顺序变化会影响 prefix cache 命中。
3. 设计一个 multi-turn chat 的 session-aware routing 策略。
4. 给出 5 个 agent serving 的关键指标。
5. 解释 tool parser 和 constrained decoding 的区别。
6. 设计一个防止 agent 无限循环的策略。
7. 画一个 agent search 的 prefix sharing tree，并标出 RadixAttention 可以复用的部分。

## 32.30 本章总结

Multi-turn、tool use 和 agent serving 是 SGLang 问题意识的集中体现：LLM 应用正在从单次生成变成多步、有状态、有工具、有分支的程序。

在这些场景中，runtime 需要处理的不只是模型 forward，还包括对话历史、工具 schema、tool parser、structured output、工具结果回灌、agent trajectory、cache locality、scheduler fairness 和安全边界。

SGLang 的 RadixAttention 可以复用多轮和 agent 轨迹中的共享 prefix，structured output 可以让工具调用更可靠，scheduler 可以管理多次 generation 的资源竞争。下一章会对比 SGLang 和 vLLM 的架构差异，把前面几章的 runtime、cache、scheduler、structured output 和 agent workload 放到同一张对比表里。
