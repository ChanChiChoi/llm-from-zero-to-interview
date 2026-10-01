# 第 32 章 Multi-turn、Tool Use 和 Agent Serving 支持

上一章讲了 Speculative Decoding：它通过 draft/verify/accept 减少主模型 decode 轮数，主要优化 TPOT 和 output tokens/s。

本章回到 SGLang 的问题意识：复杂 LLM 应用往往不是一次 `prompt -> answer`，而是多轮对话、工具调用、外部环境交互、agent 轨迹和多步程序执行。

这些场景从应用层看是 agent framework，从 runtime 层看则是：多次 generation、共享上下文、结构化工具参数、工具结果回灌、动态分支、状态管理、cache 复用和调度公平性。

一句话概括：

> Multi-turn、tool use 和 agent serving 的核心不是“模型会调用工具”这么简单，而是 runtime 能否高效、可靠地执行多步有状态 LLM 程序，并在每一轮复用上下文、约束工具格式、管理请求状态和控制尾延迟。

## 32.0 本讲范围与资料

本章参考六类公开资料：

1. SGLang 论文对 structured language model programs、frontend language、fork / branch、RadixAttention 和 runtime 协同的系统说明。
2. SGLang Frontend Language 文档对 `gen`、multi-turn、Python control flow、fork、choices / regex constrained decoding、batching、streaming 和 multi-modal prompt 的公开口径。
3. SGLang OpenAI-compatible API、Tool Parser、Structured Outputs 和 server arguments 文档，对 tools schema、tool parser、tool choice、EBNF / JSON schema / structural tag、chat template、streaming 和 metrics 的说明。
4. ReAct、Toolformer、MRKL 等公开论文，对 agent loop、工具调用、observation 回灌和 reasoning / acting 交替的基本问题背景提供依据。
5. Agent serving / LLM serving 相关公开论文和文档，对多次 generation、prefix sharing、tool latency、session-aware routing、KV cache locality 和 task-level metrics 的工程边界提供参考。
6. 前面第 28 到 31 章关于 RadixAttention、scheduler、structured generation 和 speculative decoding 的本书内部口径。

本章边界也要先说清：

1. 本章讲 SGLang-like runtime 如何支持 multi-turn、tool use 和 agent serving，不绑定某个 SGLang 版本的真实 tool parser 类名、OpenAI-compatible 字段全集、server 参数全集或内部源码路径。
2. 本章把真实工具执行放在应用层或 agent framework，不把数据库、HTTP、代码解释器、浏览器、MCP server 等外部系统塞进模型 runtime。
3. 本章 demo 是教学版 agent serving audit，只模拟 session routing、prefix cache、tool parser、validator、工具结果回灌、GPU slot 释放和指标验收条件；不实现真实网络工具、真实权限系统或生产 trace 后端。
4. Tool parser、structured output 和 validator 不是同一个东西：parser 负责把模型文本转成对象，structured output 负责约束生成格式，validator 负责业务参数和权限检查。
5. Agent serving 的性能结论必须看 task-level E2E latency、工具耗时、model calls、prefix reuse、KV pressure、scheduler queue 和失败恢复，不能只看单次 chat completion 的 QPS。

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

## 32.29 Agent Serving 公式、状态验收条件和可运行 demo

从 serving 视角看，一个 agent task 可以抽象为：

```math
\tau_i=(s_i,u_i,g_{i,1},c_{i,1},o_{i,1},\ldots,g_{i,K_i},a_i)
```

其中 `s_i` 是 session，`u_i` 是用户任务，`g_{i,k}` 是第 `k` 次模型 generation，`c_{i,k}` 是可选 tool call，`o_{i,k}` 是 tool observation，`a_i` 是最终回答。`K_i` 不是固定常数，这也是 agent serving 比单次 chat serving 更难调度的原因。

一次 task 的端到端延迟可以拆成：

```math
L_i=\sum_{k=1}^{K_i}\left(L^{\mathrm{queue}}_{i,k}+L^{\mathrm{prefill}}_{i,k}+L^{\mathrm{decode}}_{i,k}+L^{\mathrm{stream}}_{i,k}\right)+\sum_{k=1}^{K_i}L^{\mathrm{tool}}_{i,k}
```

这个公式的直觉是：agent 慢不一定慢在模型，也可能慢在工具、排队、prefix cache miss、streaming 或 agent loop 轮数。

Multi-turn / agent prefix reuse 可以用：

```math
R_{\mathrm{reuse}}=\frac{\sum_{i,k}H_{i,k}}{\max(1,\sum_{i,k}P_{i,k})}
```

其中 `P_{i,k}` 是第 `i` 个 task 第 `k` 次 generation 的 prompt token 数，`H_{i,k}` 是该 generation 命中的 prefix token 数。Session-aware routing 追求的是让后续轮次尽量落到已有 cache 的 replica 上：

```math
R_{\mathrm{affinity}}=\frac{N_{\mathrm{same\_replica}}}{\max(1,N_{\mathrm{followup}})}
```

Tool use 还要看 parser、validator 和 executor：

```math
C_{\mathrm{parse}}=\frac{N_{\mathrm{parsed}}}{\max(1,N_{\mathrm{tool\_request}})}
```

```math
C_{\mathrm{block}}=\frac{N_{\mathrm{blocked}}}{\max(1,N_{\mathrm{tool\_request}})}
```

`C_parse` 低说明模型格式、chat template 或 parser 不匹配；`C_block` 不是越低越好，高风险工具被权限系统正确阻断时，block 是安全证据。

Agent serving 的 runtime 准入条件可以形式化为：

```math
G_{\mathrm{agent}}=G_{\mathrm{session}}G_{\mathrm{cache}}G_{\mathrm{parser}}G_{\mathrm{tool}}G_{\mathrm{slot}}G_{\mathrm{round}}G_{\mathrm{metric}}
```

这些因子分别要求 session affinity、prefix cache、tool parser、工具权限、等待工具时释放 GPU slot、agent round 上限和 task-level metrics 都能被观测。

下面是一个 0 依赖 toy demo。它模拟三个 task：

1. 第一个 task 生成 `get_weather` tool call，应用层执行工具，再回灌 tool result 生成最终回答。
2. 第二个 task 是同一 session 的后续轮次，用 session-aware routing 命中同一个 replica，并复用前一轮历史。
3. 第三个 task 试图调用需要确认的 `send_email`，validator 阻断后生成安全回复。

```python
from dataclasses import dataclass
import json


@dataclass
class ToolSpec:
    name: str
    required: tuple
    needs_confirmation: bool = False


@dataclass
class AgentTask:
    task_id: str
    session_id: str
    user_tokens: list
    steps: list


class ToyReplica:
    def __init__(self, name):
        self.name = name
        self.cached_prefixes = []
        self.model_calls = 0

    def lookup(self, tokens):
        best = 0
        for cached in self.cached_prefixes:
            n = 0
            for a, b in zip(tokens, cached):
                if a != b:
                    break
                n += 1
            best = max(best, n)
        return best

    def insert(self, tokens):
        key = tuple(tokens)
        if key not in self.cached_prefixes:
            self.cached_prefixes.append(key)


class ToyAgentServingAudit:
    def __init__(self, tasks, tools):
        self.tasks = tasks
        self.tools = {tool.name: tool for tool in tools}
        self.replicas = [ToyReplica("replica_a"), ToyReplica("replica_b")]
        self.affinity = {}
        self.base_tokens = ["SYS", "TOOLS", "get_weather", "send_email"]
        self.histories = {}

    def _route(self, session_id):
        if session_id in self.affinity:
            return self.affinity[session_id]
        replica = min(self.replicas, key=lambda item: item.model_calls)
        self.affinity[session_id] = replica
        return replica

    def _parse_call(self, fragments):
        raw = "".join(fragments)
        if not raw.startswith("CALL "):
            return None, {}, "bad_prefix"
        name, payload = raw[5:].split(" ", 1)
        try:
            args = json.loads(payload)
        except json.JSONDecodeError:
            return name, {}, "bad_json"
        return name, args, None

    def _validate_call(self, name, args, confirmed):
        tool = self.tools.get(name)
        if tool is None:
            return False, "unknown_tool"
        missing = [field for field in tool.required if field not in args]
        if missing:
            return False, "schema_missing_" + ",".join(missing)
        if tool.needs_confirmation and not confirmed:
            return False, "needs_confirmation"
        return True, "ok"

    def _execute_tool(self, name, args):
        if name == "get_weather":
            return 120, ["weather", args["city"], "85F", "cloudy"]
        if name == "send_email":
            return 250, ["email", "sent"]
        return 0, ["unknown"]

    def run(self):
        rows = []
        metrics = {
            "tasks": len(self.tasks),
            "model_calls": 0,
            "tool_calls_requested": 0,
            "tool_calls_executed": 0,
            "tool_calls_blocked": 0,
            "parse_success": 0,
            "validation_failures": 0,
            "tool_latency_ms": 0,
            "tool_wait_gpu_slots_held": 0,
            "naive_prefill_tokens": 0,
            "run_prefill_tokens": 0,
            "saved_prefill_tokens": 0,
            "output_tokens": 0,
            "stream_events": 0,
            "max_rounds_seen": 0,
        }
        for task in self.tasks:
            history = self.histories.setdefault(task.session_id, [])
            history.extend(task.user_tokens)
            task_rows = []
            for round_id, step in enumerate(task.steps, start=1):
                replica = self._route(task.session_id)
                prompt = self.base_tokens + history
                hit_len = replica.lookup(prompt)
                run_prefill = len(prompt) - hit_len
                metrics["naive_prefill_tokens"] += len(prompt)
                metrics["run_prefill_tokens"] += run_prefill
                metrics["saved_prefill_tokens"] += hit_len
                metrics["model_calls"] += 1
                replica.model_calls += 1
                metrics["max_rounds_seen"] = max(metrics["max_rounds_seen"], round_id)
                replica.insert(prompt)

                if step["kind"] == "tool_call":
                    metrics["tool_calls_requested"] += 1
                    name, args, parse_error = self._parse_call(step["fragments"])
                    generated_tokens = ["CALL", name or "parse_error"] + list(args.keys())
                    metrics["output_tokens"] += len(generated_tokens)
                    metrics["stream_events"] += len(step["fragments"])
                    history.extend(["ASSISTANT_CALL", name or "parse_error"] + list(args.keys()))
                    replica.insert(self.base_tokens + history)
                    if parse_error is None:
                        metrics["parse_success"] += 1
                    valid, reason = self._validate_call(name, args, step.get("confirmed", False))
                    if valid:
                        latency, result_tokens = self._execute_tool(name, args)
                        metrics["tool_calls_executed"] += 1
                        metrics["tool_latency_ms"] += latency
                        history.extend(["TOOL_RESULT", name] + result_tokens)
                    else:
                        metrics["tool_calls_blocked"] += 1
                        metrics["validation_failures"] += 1
                        history.extend(["TOOL_ERROR", name or "parse_error", reason])
                    task_rows.append(
                        {
                            "round": round_id,
                            "replica": replica.name,
                            "prompt_len": len(prompt),
                            "hit_len": hit_len,
                            "run_prefill": run_prefill,
                            "tool": name,
                            "valid": valid,
                            "reason": reason,
                        }
                    )
                else:
                    answer_tokens = step["output_tokens"]
                    metrics["output_tokens"] += len(answer_tokens)
                    metrics["stream_events"] += 1
                    history.extend(["ASSISTANT_FINAL"] + answer_tokens)
                    replica.insert(self.base_tokens + history)
                    task_rows.append(
                        {
                            "round": round_id,
                            "replica": replica.name,
                            "prompt_len": len(prompt),
                            "hit_len": hit_len,
                            "run_prefill": run_prefill,
                            "tool": None,
                            "valid": True,
                            "reason": "final",
                        }
                    )
            rows.append({"task": task.task_id, "session": task.session_id, "rows": task_rows})

        metrics["reuse_ratio"] = round(metrics["saved_prefill_tokens"] / max(1, metrics["naive_prefill_tokens"]), 3)
        metrics["avg_tool_latency_ms"] = round(metrics["tool_latency_ms"] / max(1, metrics["tool_calls_executed"]), 1)
        metrics["model_calls_per_task"] = round(metrics["model_calls"] / max(1, metrics["tasks"]), 3)
        metrics["blocked_tool_rate"] = round(metrics["tool_calls_blocked"] / max(1, metrics["tool_calls_requested"]), 3)
        metrics["session_affinity"] = {key: value.name for key, value in sorted(self.affinity.items())}
        metrics["replica_model_calls"] = {replica.name: replica.model_calls for replica in self.replicas}

        gates = {
            "session_affinity_kept": len(set(metrics["session_affinity"].values())) == 2
            and metrics["session_affinity"]["weather_session"] == "replica_a",
            "prefix_reuse_visible": metrics["saved_prefill_tokens"] > 0 and metrics["reuse_ratio"] > 0.35,
            "tool_parser_ok": metrics["parse_success"] == metrics["tool_calls_requested"],
            "permission_block_visible": metrics["tool_calls_blocked"] == 1,
            "gpu_slot_released_while_waiting": metrics["tool_wait_gpu_slots_held"] == 0,
            "agent_rounds_bounded": metrics["max_rounds_seen"] <= 3,
            "metrics_ready": metrics["model_calls"] == 5 and metrics["stream_events"] >= 5,
        }
        gates["agent_serving_gate"] = all(gates.values())
        return rows, metrics, gates


tools = [
    ToolSpec("get_weather", ("city", "unit")),
    ToolSpec("send_email", ("to", "body"), needs_confirmation=True),
]

tasks = [
    AgentTask(
        task_id="weather_then_answer",
        session_id="weather_session",
        user_tokens=["USER", "weather", "Boston"],
        steps=[
            {"kind": "tool_call", "fragments": ["CALL get_weather ", '{"city":"Boston",', '"unit":"fahrenheit"}']},
            {"kind": "final", "output_tokens": ["Boston", "is", "85F", "and", "cloudy"]},
        ],
    ),
    AgentTask(
        task_id="followup_same_session",
        session_id="weather_session",
        user_tokens=["USER", "what", "to", "wear"],
        steps=[{"kind": "final", "output_tokens": ["Wear", "light", "clothes", "and", "carry", "water"]}],
    ),
    AgentTask(
        task_id="unsafe_email_blocked",
        session_id="ops_session",
        user_tokens=["USER", "email", "customer"],
        steps=[
            {"kind": "tool_call", "fragments": ["CALL send_email ", '{"to":"customer",', '"body":"discount"}']},
            {"kind": "final", "output_tokens": ["I", "need", "confirmation", "before", "sending"]},
        ],
    ),
]

rows, summary, gates = ToyAgentServingAudit(tasks, tools).run()
print("agent_serving_rows=", rows)
print("agent_serving_summary=", summary)
print("agent_serving_gates=", gates)
```

运行后可以看到类似输出：

```text
agent_serving_summary= {'tasks': 3, 'model_calls': 5, 'tool_calls_requested': 2, 'tool_calls_executed': 1, 'tool_calls_blocked': 1, 'parse_success': 2, 'validation_failures': 1, 'tool_latency_ms': 120, 'tool_wait_gpu_slots_held': 0, 'naive_prefill_tokens': 72, 'run_prefill_tokens': 27, 'saved_prefill_tokens': 45, 'output_tokens': 24, 'stream_events': 9, 'max_rounds_seen': 2, 'reuse_ratio': 0.625, 'avg_tool_latency_ms': 120.0, 'model_calls_per_task': 1.667, 'blocked_tool_rate': 0.5, 'session_affinity': {'ops_session': 'replica_b', 'weather_session': 'replica_a'}, 'replica_model_calls': {'replica_a': 3, 'replica_b': 2}}
agent_serving_gates= {'session_affinity_kept': True, 'prefix_reuse_visible': True, 'tool_parser_ok': True, 'permission_block_visible': True, 'gpu_slot_released_while_waiting': True, 'agent_rounds_bounded': True, 'metrics_ready': True, 'agent_serving_gate': True}
```

这个 demo 要表达的工程重点是：

1. Multi-turn 和 agent serving 的单位是 task / session，不只是单次 completion。
2. 同一个 session 的后续 generation 应尽量命中同一个 runtime replica，提升 prefix cache 复用。
3. Tool parser 只负责解析格式，权限和确认仍要靠 validator / application。
4. 工具等待期间不应该占用 GPU decode slot。
5. 高风险工具被阻断不是失败，而是安全条件生效。
6. Agent serving 的 dashboard 必须同时看 model calls、tool latency、prefix saved tokens、stream events、blocked tool rate 和 task-level latency。

## 32.30 小练习

1. 画出一次 tool use 的完整链路：user、model、tool parser、tool execution、tool result、final answer。
2. 解释为什么 tool schema 顺序变化会影响 prefix cache 命中。
3. 设计一个 multi-turn chat 的 session-aware routing 策略。
4. 给出 5 个 agent serving 的关键指标。
5. 解释 tool parser 和 constrained decoding 的区别。
6. 设计一个防止 agent 无限循环的策略。
7. 画一个 agent search 的 prefix sharing tree，并标出 RadixAttention 可以复用的部分。

## 32.31 本章总结

Multi-turn、tool use 和 agent serving 是 SGLang 问题意识的集中体现：LLM 应用正在从单次生成变成多步、有状态、有工具、有分支的程序。

DeepSeek V3.2 的 `thinking with tools` 进一步提醒我们：模型的 reasoning、tool call、tool result 和 final answer 需要在消息/encoding 层保持边界。报告说明，如果后续轮次只有 tool message，历史 reasoning 会保留；只有引入新的 user message 时才丢弃历史 reasoning，避免每次工具返回后重复思考整个问题。Roo Code 或 Terminus 这类把工具交互模拟成 user message 的 harness 可能无法获得该收益，官方建议这类架构使用 non-thinking 模型。独立 parser 只能把输出转换成候选事件，不能授予模型工具权限，也不能证明工具已经执行。生产链路仍要经过 schema 校验、权限/确认门禁、超时取消、结果回灌和 trace 审计；对于 V3.2-Speciale，官方明确其不支持 tool calling，相关工具指标应记为 `not_applicable`，而不是失败率为零。

在这些场景中，runtime 需要处理的不只是模型 forward，还包括对话历史、工具 schema、tool parser、structured output、工具结果回灌、agent trajectory、cache locality、scheduler fairness 和安全边界。

SGLang 的 RadixAttention 可以复用多轮和 agent 轨迹中的共享 prefix，structured output 可以让工具调用更可靠，scheduler 可以管理多次 generation 的资源竞争。下一章会对比 SGLang 和 vLLM 的架构差异，把前面几章的 runtime、cache、scheduler、structured output 和 agent workload 放到同一张对比表里。

## 32.32 Qwen3.8 Max (0902) 的 revision 与工具协议门禁

Qwen3.8 Max (0902) 的服务接入说明了为什么模型 alias、effort、工具选择和缓存必须同时进入 Agent serving manifest。Qwen Cloud 将 0902 标为 `qwen3.8-max` 的 upgraded snapshot，alias 为 `qwen3.8-max-2026-09-02`；因此 router 日志至少要保存用户请求的 alias、解析后的 revision、provider、tokenizer/template 版本和实际 mode。

请求预算要分开写：context `1M`，普通最大输入 `991K`，thinking 最大输入 `983K`，最大输出 `131K`。`reasoning_effort` 的 `low/medium/xhigh` 与 `thinking_budget` 互斥；不要让客户端同时发送两个字段后由不同 provider 自行解释。请求账本还要加入 reasoning token、工具 schema、工具结果、重试、压缩、缓存命中和最终 artifact，不能只把 input+output 与 context window 做加法。

工具门禁尤其容易在迁移时出错：thinking 模式下 `tool_choice` 只有 `auto`/`none`，强制指定工具需要关闭 thinking；多模态请求使用 `MultiModalConversation`。因此 adapter 应在请求进入模型前做 capability matrix 检查，在 parser 后做 schema/权限/确认检查，在 executor 后记录实际执行回执。模型输出 call、宿主接受 call 和工具真正产生副作用是三个不同状态。

缓存也要分层记录。Qwen 文档区分 explicit、implicit、session cache，最小缓存长度为 `1,024` tokens，三者的命中、计费和有效期语义不同。router 不应把 provider cache 当作可跨 revision、跨 tokenizer 或跨租户任意共享的 GPU KV；cache key、失效、session affinity、权限和失败重算都必须进入 trace。一次安全的迁移回归至少覆盖：

1. `reasoning_effort` 与 `thinking_budget` 同时出现时是否在客户端拒绝；
2. thinking + forced tool 时是否在 capability gate 处拒绝或切换到明确的非 thinking 请求；
3. 0902 与旧 revision 的 prefix/cache 是否被错误复用；
4. 多模态消息、tool result、取消、超时和重试能否保持同一 session 的状态一致；
5. DataCurve 泛化 Agent 结果是否被错误写入 0902 的模型级 dashboard。

### 32.32.1 QwenCloud Context Cache 的协议门禁

Context Cache 的实现不能只保留一个 `cache_hit=true` 布尔值。Explicit cache 要校验 `cache_control.type=ephemeral`、`1,024` token 资格门槛、最多 4 个 marker 和 marker 向前最多 20 个 content block 的 lookback；超过 4 个 marker 时只有最后 4 个生效。达到 token 门槛不保证实际命中。并行工具调用如果把每个结果拆成多个 content block，可能把待复用前缀推过该窗口。

文档另给出一个不同的 follow-up 规则：相对既有 cache block A 的 `other messages` 不超过 20 条时，可以命中 A、刷新 TTL 并建立扩展 block；超过 20 条时 A 不复用，服务按完整上下文建新 block。它按 message 计数，不能和 marker 向前检查的 20 个 content block 合并成一个阈值。Chat Completions、DashScope、Anthropic-compatible API 中 explicit/implicit cache 互斥；Responses 未开启 session cache 时，支持的模型仍可能自动使用 implicit cache。

Explicit/session 的 cache block 有 5 分钟有效期，命中会刷新；implicit cache 没有固定 TTL，命中概率也不保证。文档中的典型计费因子为 explicit/session 创建 `1.25x`、命中 `0.10x`，implicit 创建 `1.00x`、命中 `0.20x`；产品页的实际美元价格字段必须单独绑定到当前 alias 和时点，不能把两套口径混成 GPU 成本。

Responses session cache 还要把 `x-dashscope-session-cache: enable`、`previous_response_id`、account 和 model 绑定到同一 session lineage。Responses API 的命中字段是 `usage.input_tokens_details.cached_tokens`；Chat Completions cache 示例则用 `usage.prompt_tokens_details.cached_tokens`，不能跨 endpoint 套字段。Responses API 会延续前轮 input/output 并追加当前 input，但前轮 `instructions` 不会自动继承，调用方要逐轮重发；`previous_response_id` 不能与 `conversation` 同用，`store=false` 的 response 不能续接。总 token、cache creation 和 hit 也不能用简单减法推导 prefill、KV 字节或账单。

本轮的 [`qwen_max0902_cache_contract_audit.py`](../../research/model-update-2026-09/code/qwen_max0902_cache_contract_audit.py) 用标准库验证这些 host-side invariants，并覆盖 thinking + forced tool、`reasoning_effort`/`thinking_budget` 冲突、implicit 非保证命中和 session continuation。它是 `local_protocol_toy`，不代表 QwenCloud 的生产调度、实际命中率或结算结果。

### 32.32.2 QwenCloud 动态限流与 endpoint 迁移

2026-09-21 的 QwenCloud 文档示例已经从 `dashscope-intl.aliyuncs.com` 迁移到 `maas.qwencloudapi.com`。例如 OpenAI-compatible mode 使用 `https://maas.qwencloudapi.com/compatible-mode/v1`，DashScope API 使用 `https://maas.qwencloudapi.com/api/v1`，Realtime WebSocket 也迁移到同一 host。这个变化首先影响 provider adapter、认证、路由和 transport 回归；它不能证明模型权重、架构或能力升级。历史日志可以保留旧 endpoint，但当前 manifest 必须明确 endpoint 生效时间和文档版本。

QwenCloud 的 Dynamic Rate Limiting 还把服务配额和模型吞吐分开。限流粒度是 `account + model`：同一账户下多个 workspace/API key 的请求量聚合；workspace 可以为单个模型设置 override，没有 override 时继承 account-level quota。TPM tier 根据自然月消费调整，平台达到保证 TPM 后如果仍有余量可能继续接受请求，所以保证值是 soft limit；文档的月度节奏是每月 10 日通知、15 日生效，tier 只升档或保持不变。`qwen3.8-max-0902` 当前表中的保证 TPM 为 `1,500,000 / 1,500,000 / 1,500,000`。

因此 Agent serving manifest 至少应分开记录：

- `account`、`workspace`、model alias、resolved model revision、provider endpoint、region 和 API mode；
- `guaranteed_tpm`、`observed_tpm`、并发、queue latency、429 数量、`Retry-After`、重试次数和单位成功成本；
- tokenizer/template、reasoning mode/effort、cache 类型与命中、工具 schema、执行回执和最终 artifact。

`guaranteed_tpm` 只能回答 provider 对账户/模型组合承诺的服务配额，不能替代模型 tokens/s、GPU capacity、KV cache 容量或目标硬件 profiling。Qwen3.8 Max (0902) 的 hosted quota 与 Artificial Analysis 的第三方速度/成本字段、DataCurve 的泛化 `qwen3_8_max_xhigh` Agent 结果完全分账；DataCurve 没有精确 0902 行，不能把任何 quota 或泛化 Pass@1 写成 revision 级能力。

## 32.33 GPT-5.3 Codex：Responses-only Agent Runtime 门禁

GPT-5.3 Codex 把本章的 runtime 边界推到更具体的 Coding Agent 场景。OpenAI 官方模型页确认它只支持 Responses API，并提供 function calling、web search、hosted shell 和 skills；官方 Codex Prompting Guide 又把 `apply_patch`、固定工作目录、代码库探索、并行工具调用和长时间自治放在 harness 设计中心。这里的关键不是“模型多了几个工具”，而是 adapter、session、workspace、executor 和 verifier 必须共同形成可恢复的状态机。

一次 Codex-style rollout 可以写成：

```text
Responses input
  -> model reasoning / assistant phase=commentary
  -> tool call proposal
  -> schema + permission + approval gate
  -> sandboxed shell / patch / skill executor
  -> tool receipt + artifact diff
  -> next Responses input or phase=final_answer
```

`phase=commentary` 是中间进度或工具前说明，`phase=final_answer` 才是最终输出协议状态。phase 不能当作 chain-of-thought，也不能代替 executor 回执；模型说“已修复”不等于测试通过、文件已写入或 artifact 已提交。

### 32.33.1 Responses-only 的协议影响

若 runtime 原本只支持 Chat Completions，迁移 GPT-5.3 Codex 不能只替换 model ID。至少要重新设计：

1. input/output item 而不是只有 message 文本；
2. tool call、tool result、加密 reasoning item 和 assistant `phase` 的完整 replay；
3. `previous_response_id` 与无状态 input-array 两种链路的状态语义；
4. stream event、取消、重试、幂等和最终 artifact 的 trace；
5. 400K context、272K maximum input、128K maximum output、reasoning tokens、工具 schema/results 和 workspace 的统一预算账本。

只保留最终文本会造成一种隐蔽故障：表面上历史可读，下一轮却缺少继续工具循环所需的 opaque state 或 phase，Agent 可能重复执行、提前结束或错误地把中间 preamble 当最终答案。

### 32.33.2 Compaction、prefix cache 与工作区

GPT-5.3 Codex 的 first-class compaction 适合多小时 coding task，但 compaction 不是普通摘要，prompt cache 也不是永久会话记忆。服务端 compaction 返回的 encrypted item 和 standalone compact endpoint 的 canonical context 都应按协议原样保存；应用还要在 workspace 中保存 diff、测试结果、artifact hash 和未完成副作用。

缓存账本与压缩账本必须分开：cache 关注稳定 rendered prefix 的 KV state 复用，compaction 关注继续任务所需状态的重新表示。压缩后 prefix 变化可能造成 cache miss；cache hit 也不能证明目标、权限或 artifact 状态正确恢复。

### 32.33.3 生产验收门禁

针对 Codex-style Agent serving，建议把以下条件作为一次 task-level gate：

| 层 | 必须证明的事实 | 失败示例 |
|---|---|---|
| 模型/API | ID、snapshot、effort、端点和预算可追溯 | 把 GPT-5.4 的 tool_search 字段迁移到 GPT-5.3 Codex |
| 协议状态 | output item、phase、tool result 和 compaction 可 replay | 只保存最终文本 |
| 工具执行 | schema、权限、沙箱、超时、幂等和回执完整 | 模型生成 shell 就算执行成功 |
| 工作区 | patch、测试、diff、artifact hash 和回滚可核验 | “看起来改好了”但没有测试证据 |
| 评测 | 模型、harness、工具、环境和 verifier 分栏 | 把 AA 指数或其他模型 DeepSWE 结果当裸能力 |

因此，Agent serving 的核心指标至少包括 task success、model calls per task、tool rounds、tool error/timeout、prefix saved tokens、compaction recovery、duplicate side effects、E2E p95 和 unit-success cost；不能只看单轮 TTFT 或 tokens/s。

本节的 GPT-5.3 Codex 资料来自 Artificial Analysis 候选页与 OpenAI 官方模型/API/harness 文档。官方没有公开其参数、注意力结构、训练 recipe、生产 kernel 或线上 acceptance rate，不能把本节的协议门禁写成模型内部实现。

## 32.34 GPT-OSS：MoE dispatch、MXFP4 与 Harmony serving

gpt-oss 的 serving manifest 至少要同时保存 `gpt-oss-120b`/`20b`、固定 revision、`low/medium/high` effort、Harmony tokenizer/template、MXFP4 kernel、专家并行拓扑、窗口和工具 harness。`total parameters`、`active parameters`、驻留/分片权重、top-4 assignment、padding、all-to-all、GQA KV、workspace 和并发请求必须分栏，不能只记录一个“5.13B active”。

Harmony tool call 的生产链路可以写成：

```text
Harmony render -> model channel/recipient -> parser/schema gate
-> permission/approval -> sandbox browser/Python executor
-> tool receipt/result replay -> next generation -> artifact verifier
```

`analysis`、`commentary`、`final` 是协议 channel，不等于宿主已经执行工具；`low/medium/high` 是同一权重的推理配置，也不应被 router 当作不同 checkpoint。部署压测要把 sliding/full attention 层、MXFP4 误差、专家负载、cache bytes、TTFT/TPOT、工具轮数、恢复率和单位成功成本一起报告。DataCurve 当前没有 gpt-oss 精确行，不能把其他 OpenAI/Codex Agent 结果写进该模型 dashboard。

### 32.34.1 gpt-oss 的 raw CoT 与 provider 兼容性门禁

gpt-oss 的工具调用可能发生在 raw CoT 内部，因此 serving 层不能只回传最终文本。Responses 兼容层应保留 `reasoning.content[]` 中的 `reasoning_text`，并正确发出 `response.reasoning_text.delta`/`done` 事件；下一轮将 reasoning item、tool call 和 tool result 按原有 lineage 重放到 Harmony。Chat Completions 兼容层可采用 `reasoning` 消息字段和 delta 约定，但这属于 gpt-oss provider 兼容方案，不应冒充托管模型的通用公开契约。

raw CoT 默认不能展示给终端用户，因为其中可能出现有害内容或 developer instruction 泄露。应用应把可见 summary 与 raw CoT 分开存储，按 `item_id`、index、turn lineage 做顺序/去重校验；final 产生后清理 analysis，工具循环未结束时保留上一 final 前的 analysis，重试还要防止重复副作用。

验收分三层：

```text
API shape/channel/schema
        -> compatibility-test tool-call smoke test
        -> raw CoT + tool-result replay
        -> AIME/GPQA/HealthBench quality eval
```

官方 `compatibility-test` 的 0 invalid requests、`pass@k` 与 `pass^k` 均超过 90% 只能作为“很可能兼容”的信号，不能替代 kernel 数值、MXFP4 误差、MoE dispatch、目标硬件 profiling、线上 acceptance 和 artifact verifier。serving dashboard 应把 API 兼容、模型质量、状态恢复、工具副作用和单任务成功成本分开记录。

## 32.35 Claude Opus 4.6：协议状态与工具发现门禁

Opus 4.6 的 serving manifest 至少要保存模型 snapshot、effort、thinking 类型、工具目录 revision、compaction beta、权限策略和 verifier 版本。`effort`、`max_tokens`、1M context 和 128K/300K output 是不同约束；容量规划还要加入 thinking、工具 schema、工具结果、重试和 compaction 的 token 与 KV 账本。

一次多轮调用可抽象为：

```text
request -> adaptive thinking/signature -> tool call
-> host schema + permission gate -> executor receipt
-> tool result -> compaction block if needed -> next turn -> artifact verifier
```

thinking signature 和 compaction block 不能由业务层改写成任意摘要。Tool search 的 `tool_reference` 只是降低 schema 上下文；权限、审批、沙箱、allowlist 和实际副作用仍必须由宿主控制。Computer use 仍使用 `computer_20251124`，动作执行、截图、prompt-injection 防护和人工接管不能由模型输出替代。

生产门禁至少检查：完整 item replay、signature/compaction 完整性、tool search 选择与权限隔离、未知超时状态、重复副作用、artifact 完整率和单位成功成本。DataCurve 没有精确 Opus 4.6 行，因此 dashboard 不得迁移 Opus 4.8/5 的 Pass@1、成本或 Agent steps。

## 32.36 Claude Opus 4.7：三层预算、高分辨率视觉与安全控制面

Opus 4.7 的 serving manifest 不能只保存模型 ID 和 context window。至少应记录 snapshot、effort、task budget、`max_tokens`、thinking/tool/result/output usage、tokenizer 版本、视觉尺寸、visual tokens、工具目录、compaction 状态、策略版本、权限决定和 verifier 版本。

### 32.36.1 三层预算账本

请求级容量可以抽象为：

```text
step policy: effort
loop budget: task budget
request cap: max_tokens
```

`effort` 影响单步的投入倾向，不是严格 token 上限；task budget 约束整个 Agent loop，并包含 thinking、工具调用、工具结果和最终输出；`max_tokens` 约束单次响应。服务端 compaction 只改变继续任务所需的上下文表示，不重置当前 turn 已消耗的 task budget。调度器因此应分别记录：

```text
input + cached_prefix + thinking + visible_output
+ tool_schema + tool_result + retry + compaction
```

不能把 1M context、128K output、task budget 和 effort 直接相加推导并发。并发估算还需要考虑 KV/cache 命中、workspace、工具等待期间是否占用 GPU slot、重试和取消清理。

### 32.36.2 Tokenizer 迁移和视觉输入

Opus 4.7 的更新 tokenizer 可能使同一输入变成旧版本约 `1.0-1.35x` token。迁移压测要固定请求内容和模板，比较输入 token、稳定前缀、缓存命中、thinking/output、工具结果、TTFT、TPOT 和单位成功成本，而不是只把旧预算乘一个经验比例。

高分辨率视觉 tier 的接口边界是最长边 `2576 px`、最多 `4784` visual tokens；普通 tier 为 `1568 px`、`1568` visual tokens。视觉 serving manifest 应保存原始像素、缩放后的像素、patch/token 估计、媒体顺序、坐标变换和缓存 key。对 computer/browser 截图，应用在回灌前自行缩放；否则一个大截图可能同时改变输入成本、上下文剩余空间和延迟。

可以把视觉请求的可观测账本写成：

```text
image_bytes -> resize policy -> visual tokens -> prompt tokens
-> cache lookup -> model/tool turn -> coordinate transform
-> executor receipt -> visual/text verifier -> artifact
```

高分辨率接口说明的是输入契约和 serving 成本，不说明视觉编码器的层数、训练数据或内部 patch 实现。

### 32.36.3 Cyber safeguard 不是执行权限

发布页描述的自动 cyber safeguards 和 Cyber Verification Program 属于控制面。生产请求仍应经过：

```text
model proposal -> policy classification -> user/org authorization
-> tool schema gate -> sandbox/network isolation
-> human escalation if needed -> executor receipt -> verifier/audit
```

策略拦截、授权拒绝、工具失败和模型不会生成调用是不同状态，不能都记成同一个 refusal。对于高风险工具，审计日志要保存策略版本、命中规则、授权主体、工具参数摘要、网络范围、执行回执、未知状态和最终 artifact。研究例外也应通过独立验证路径，不应把例外入口写成模型默认权限。

### 32.36.4 生产验收表

| 层 | 必须记录 | 常见错误 |
|---|---|---|
| 模型/配置 | `claude-opus-4-7`、snapshot、effort、预算 | 把 `xhigh` 当作新模型 |
| token/缓存 | tokenizer、输入 token、稳定前缀、cache hit | 沿用旧版本 token budget |
| 视觉 | 原图、缩放、visual tokens、坐标映射 | 只保存截图，不保存缩放契约 |
| Agent 状态 | thinking、tool result、compaction、remaining budget | compaction 后重置 loop 预算 |
| 安全控制 | policy、授权、沙箱、网络、人工升级、回执 | 模型生成 tool call 就算已执行 |
| 评测归因 | 模型、provider、harness、任务、verifier | 迁移其他 Claude 的 DeepSWE 分数 |

Artificial Analysis 的 Intelligence Index 和速度字段是第三方配置观察；DataCurve 当前没有精确 Opus 4.7 行，不能迁移 Opus 4.6、4.8 或 5 的 Pass@1、成本和 Agent steps。官方资料公开了运行时和控制面契约，但没有公开参数、注意力结构、训练 recipe、生产 kernel 或线上 acceptance rate。

## 32.37 Gemini 3.5 Flash-Lite：长上下文、视频处理与低成本 serving

Gemini 3.5 Flash-Lite 的 serving manifest 至少应保存 model code、请求级 `thinking_level`、输入模态、媒体 file handle、视频 processing mode、缓存状态、工具 schema、权限决定和 verifier 版本。Google API 页面给出 1,048,576 输入 token、65,536 输出 token，并列出 caching、code execution、File Search、function calling、Search/Maps grounding、structured output 和 Computer Use Preview；这些是接口容量和能力契约，不等于 GPU 显存或有效上下文召回。

### 32.37.1 Static 与 agentic video 的两本账

视频可以分为两种 serving 路径：

```text
static: media -> fixed frame sampling -> one context build -> generation
agentic: media -> processing_call -> selected segment/transcript/audio
        -> processing_result -> next step -> generation
```

static 路径易于预估输入 token 和延迟，适合短视频或需要全片固定采样的请求；agentic 路径只读取与问题相关的时间轴内容，可能降低长视频的 token 和噪声，但要额外维护处理事件、媒体状态、超时、取消、重试和回放。`processing_call`/`processing_result` 应像工具事件一样进入 trace，且必须与模型回答、宿主授权和真实副作用分开。

### 32.37.2 Thinking 与成本路由

Thinking 文档把 Lite 的默认档位列为 `minimal`，支持 `minimal/low/medium/high`。router 不应把这些 level 当成四个模型，也不能只用输出 token 估算成本。请求账本至少包括输入 token、thinking token、可见输出、媒体处理 token、工具 schema/result、缓存命中、处理步骤、重试和等待时间。对 subagent/文档解析优先低 level 的策略，应通过固定任务和 verifier 验证质量是否仍在目标门槛；复杂任务升级 level 时要保留触发原因和剩余预算。

### 32.37.3 版本和证据边界

DeepMind Model Card 明确 Gemini 3.5 Flash-Lite based on Gemini 3.1 Flash-Lite，并把架构、训练数据、硬件和软件资料指向 3.1 Model Card。因此 serving 层可以据官方文档实现模态、预算、视频事件和工具门禁，但不能据此推导 Lite 的层数、专家数、attention kernel 或硬件利用率。`whats-new-gemini-3.5` 页面正文属于 Gemini 3.5 Flash，其默认 `medium`、GA 和 thought preservation 不能写到 Lite。

完整快照和待核验项见 [`gemini-3.5-flash-lite-source-notes.md`](../../research/model-update-2026-09/gemini-3.5-flash-lite-source-notes.md)。DataCurve 当前没有精确 Lite 行，不能把 Gemini 3.5 Flash 或其他 Gemini 版本的 Agent 分数迁移到 Lite。

## 32.38 Kimi K2.6：MLA、native INT4 与 Vendor Verifier

K2.6 的 serving manifest 不能只保存模型名和 256K context。官方配置还给出 `q_lora_rank=1536`、`kv_lora_rank=512`、384 experts/top-8、MoonViT 和 compressed-tensors group-size-32 INT4；实际调度要把 latent/position cache、专家 dispatch、视觉 token、`reasoning_content`、工具 schema/result、workspace 和并发余量一起记账。

部署验收应分层：

```text
weights/revision -> quantization/kernel -> sampling gate
-> tool/reasoning parser -> multimodal preprocessing
-> Agent harness/sandbox -> verifier/artifact
```

Kimi Vendor Verifier 的 pre-flight、OCRBench、MMMU-Pro、AIME2025、K2VV ToolCall 和 SWE-Bench 分别暴露参数约束、视觉管线、长输出 KV/量化、工具 schema 与 Agent sandbox 的问题。它们是实现和 harness 的验收，不是 K2.6 裸模型能力分数。DataCurve 当前没有精确 K2.6 行，不能把其他 Kimi 版本的 Agent 结果迁移到本节。

## 32.39 GPT-5.4 mini/nano：小模型路由、能力探测与显式契约

GPT-5.4 mini/nano 说明了一个常见 serving 误区：同属一个产品家族，不代表可以共享一份模型 manifest。OpenAI 官方页给出的精确 snapshot 是 `gpt-5.4-mini-2026-03-17` 和 `gpt-5.4-nano-2026-03-17`；两者都是 text/image → text、400K context、272K maximum input、128K maximum output，并支持 `none/low/medium/high/xhigh` reasoning effort。它们不能直接套用 GPT-5.4 base 的 1.05M context 或 base 的工具清单。

### 32.39.1 按任务形状路由

官方定位把两个 sibling 放在不同的工作负载位置：mini 偏高吞吐 coding、computer use 和仍需要较强 reasoning 的 Agent workflow；nano 偏 classification、extraction、ranking 和窄任务 sub-agent。router 不应只按价格选择，而应先判断：

```text
request
  -> task-shape classifier
  -> capability probe(model_id, snapshot, endpoint)
  -> mini / nano / base route
  -> fixed effort + explicit prompt contract
  -> verifier -> escalation / retry
```

nano 的默认任务应有清晰边界、固定 schema、有限工具和可判定停止条件。开放式规划、隐含消歧、多工具依赖和高失败代价任务，应升级到更强模型，或由更强模型规划、nano 执行固定子任务。把 nano 的 `reasoning_effort` 调高不能替代任务切分、输入约束和独立 verifier。

### 32.39.2 Prompt contract 和能力矩阵

小模型更依赖显式契约。生产 prompt 至少要写清目标、关键规则、前置依赖、工具调用顺序、输入缺失如何 abstain、工具失败如何恢复、输出 schema、长度、停止条件和一个正确示例。`reasoning_effort` 是请求级行为/投入旋钮，不是可直接拿来做并发容量规划的硬 token 上限；容量账本仍要拆开 input、reasoning、visible output、tool schema、tool result、retry、cache hit 和 executor wait。

能力矩阵必须按精确页面读取：mini 当前页面列出 `tool_search` 与 `computer_use`；nano 当前页面未列出这两项。即使两者都有 function calling、web search、file search、code interpreter、hosted shell、apply patch、skills 和 MCP，实际是否可用仍取决于 endpoint、provider、宿主授权、沙箱、审批和超时策略。模型返回 tool call 不是工具已获授权，更不是副作用已经提交。

### 32.39.3 成本与验收账本

官方价格快照为 mini input `$0.75/M`、cached input `$0.075/M`、output `$4.50/M`；nano input `$0.20/M`、cached input `$0.02/M`、output `$1.25/M`。router 可以用价格做候选排序，但成功路由的成本应写成：

```text
unit_success_cost = (input + cached_input + reasoning + output
                     + tool/result + retry + executor_wait)
                    / verified_successes
```

DataCurve 当前只有 `mini_swe_agent_gpt_5_4_xhigh`，没有 mini/nano 精确行；该 base + `mini-swe-agent` 系统结果不能填入两个 sibling 的 Agent dashboard。验收至少固定 model ID、snapshot、effort、prompt contract、工具目录、harness、任务集、verifier、provider 和重试策略，并分别报告 task success、工具错误、升级率、延迟、tokens 和单位成功成本。

本节对应的完整快照、官方来源和负面证据见 [`gpt-5.4-mini-nano-source-notes.md`](../../research/model-update-2026-09/gpt-5.4-mini-nano-source-notes.md)。官方没有公开两个 sibling 的参数、层/专家/注意力结构、训练 recipe、独立技术报告、production kernel 或线上 acceptance rate；本节讨论的是模型/API 契约和 Agent serving 责任边界。

## 32.40 Grok 4.20：多 Agent、Compaction 与工具状态的服务账本

Grok 4.20 的官方文档把普通 `grok-4.20-0309-reasoning`、non-reasoning 和 `grok-4.20-multi-agent-0309` 分成不同服务对象。普通模型页写 1M maximum prompt；Artificial Analysis 当前目录写 2M context。serving manifest 不能只保存一个展示名称，而应绑定：

```text
model_id + alias + endpoint + region + capability snapshot
-> context limit + long-context billing threshold
-> effort semantics + tool catalog + rate limit
-> cache/compaction policy + verifier
```

`grok-4.20-multi-agent` 的 `reasoning.effort` 语义特殊：low/medium 对应 4 个协作 Agent，high/xhigh 对应 16 个；它控制并行编排规模，不是把普通模型的 reasoning token 上限简单调大。调度器要为每个子 Agent 保存上下文、角色、工具、来源和失败状态，并为 leader 保存合并依据和引用。

### 32.40.1 Compaction 是协议对象，不是摘要字符串

xAI Context Compaction API 返回一个单独的 `type=compaction` item，其中 encrypted content 是 opaque blob。继续请求时要把该 item 原样放在新用户输入前面，不能解析、裁剪或重排。服务端 compaction 只能压缩尚未超出 context 的会话；它不重置已经使用的工具轮数、task budget、权限决定或外部副作用。

因此长任务状态至少分成：模型/加密 reasoning state、compaction item、prompt cache 前缀、工具调用与回执、宿主 workspace、artifact 和 verifier。只保留 leader 最终文本会丢失子 Agent 的证据链；只保留 compaction blob 又无法替代宿主的权限和执行日志。

### 32.40.2 混合工具的边界

xAI server-side Web Search、X Search、Code Execution 和 Collections Search 可由服务器自动执行；client-side function call 则在模型提出调用后暂停，把权限和副作用交给宿主。Remote MCP 通过 Streaming HTTP/SSE 接入外部工具；`allowed_tools` 既是最小权限开关，也是减少 tool schema context overhead 的成本开关。`max_turns` 限制单次请求的 assistant/server-side tool turns；client-side tool 形成 checkpoint 后，后续请求会重新计数。

工程实现要让每个事件带 `call_id`、工具定义 hash、参数摘要、授权决定、执行回执、重试/幂等键和来源。Function calling 默认允许 parallel calls，但并行只适合独立或幂等操作；涉及写入时仍应按依赖拆成多轮并经过 verifier。多 Agent、compaction、cache 和工具不是模型权限，不能因为 API 返回成功就跳过最终 artifact gate。

## 32.41 Gemini 3.8 Flash：Interactions、1M 上下文与 Computer Use serving

Gemini 3.8 Flash 的 serving 状态不应只保存最终文本。Interactions API 的 `thought`、`tool_call`、`tool_result`、`model_output` 和 SSE 事件需要进入事件日志；stateful 模式可以用 `previous_interaction_id` 继续，stateless 模式则必须按协议回放此前 steps、签名和工具回执。服务端状态、应用会话、prompt cache 和 GPU KV cache 是四个不同层次，不能互相替代。

一次工具请求的成本账本应拆成输入/输出/thinking token、tool schema/result、Search/URL/File/Code 处理、cache hit/miss、executor wait、重试和 verifier。1M input limit 只是接口上限，实际吞吐还受 attention/KV、并发、长文档召回、缓存前缀和输出预算影响；本模型页面的 `$0.75/$3.75` 与 AA 测量字段也要按来源和时间保存。

Computer Use 的模型输出是 action intent，不是执行结果。serving 层要在 action 前做 capability/permission probe、域名与资源 allowlist、`require_confirmation`/`blocked` 策略、幂等和高风险拦截；执行后回传截图/结果，再由模型继续。坐标归一化解决接口格式，不解决 UI 漂移、遮挡、竞态和错误点击。当前没有 Gemini 3.8 独立架构或生产 kernel 证据，因此本节只讨论公开 API/runtime 行为和可复现的审计边界。

2026-09-23 文档复验补充了 serving manifest 的生命周期字段：Interactions 默认保存 55 天（paid）或 1 天（free），支持按 ID 删除；`store=false` 不能用于后续 `previous_interaction_id`。`previous_interaction_id` 只恢复 history，tools、system instruction 和 generation config 必须每轮重新写入；跨模型链还要检查输出模态兼容性。Thinking 的 `max_output_tokens` 是 thought + visible output 的硬上限，触顶状态可能为 `incomplete`，因此调度器不能把它当作单纯可见输出长度。

另外，官方 Thinking 与 Tool combination 页面对 standard function-call signature 的字段范围并不完全一致。适配器应保存原始 step/schema、实际出现的 opaque signature、call/result `id` 和 capability-probe 版本；不能为了统一内部类型而补造或删除 signature。这个字段门禁与缓存、重试、幂等和 verifier 一起决定是否允许恢复执行。

## 32.42 DeepSeek V4 Pro：压缩 KV、混合状态与 serving manifest

V4 Pro 的 serving manifest 必须同时保存“模型身份”“参考实现”“状态布局”和“Agent 协议”四类信息。只写 `deepseek-v4-pro` 与 `1M` 会丢掉 revision、量化、indexer、局部窗口和工具解析的关键约束。

```yaml
model_id: deepseek-v4-pro
hf_revision: b5968e9190ef611bbf34a7229255be88a0e937c1
architecture: DeepseekV4ForCausalLM
config: {layers: 61, hidden_size: 7168, routed_experts: 384,
  routed_per_token: 6, shared_experts: 1, index_topk: 1024,
  max_position_embeddings: 1048576}
inference: {window_size: 128, index_n_heads: 64, index_head_dim: 128,
  hc_mult: 4, hc_sinkhorn_iters: 20, expert_dtype: fp4,
  compression_ratios: [128, 4]}
conversion_example: {experts: 384, model_parallel: 8}
state: [compressed_kv, indexer_candidates, local_window, overlap_tail,
  mla_latent_or_position, moe_dispatch, hc_stream]
protocol: {reasoning_effort: [low, high, max], responses: stateless,
  parallel_tool_calls: true}
```

`compressed_kv`、indexer candidate 和 `local_window` 不能合并成一个固定 bytes/token 公式：不同层的压缩倍率、未完成块、top-k 索引、量化 scale 和 cache eviction 都不同。`hc_stream` 也不是 KV cache，它属于残差流混合；`moe_dispatch` 还要计入专家通信与 workspace。恢复一个长会话时，manifest 必须知道这些状态是否可以从 prefix cache 重建，还是需要随请求持久化。

API 的 `low/high/max` 是请求级 reasoning effort，Responses 当前是 stateless；`function tools`、`apply_patch` 和并行工具调用只描述协议能力。宿主仍负责会话历史、schema/权限校验、工具执行、幂等、回执、重试和 artifact verifier。官方 encoding 中的 DSML、`<think>` 和 `reasoning_effort` 是 prompt/parser 协议字段，不是可以直接观察的隐藏思维链。

DataCurve 的 `mini_swe_agent_deepseek_v4_pro_max` 结果必须单独挂在 manifest 的评测段，绑定 `n_runs=4`、工具、任务集、环境和 verifier；不能把 Pass@1/4 写成模型字段，也不能迁移给 V4 Flash 或 V4.1-Flash。完整权重、生产 kernel、硬件 profiling 和线上 tool acceptance 仍是待验收项。

## 32.43 DeepSeek V4.1-Flash：EPD、分层 cache 与 reference runtime

V4.1-Flash 的请求 manifest 要把服务 alias、排行榜锚点、权重 revision 和参考实现分开。一个教学版可以写成：

```yaml
requested_model: deepseek-flash
catalog_anchor: deepseek-v4-1-flash
hf_revision: dba1be0a40aa45a94ad051997016db3960a90277
architecture: DeepseekV41ForCausalLM
backbone: {layers: 40, causal_encoder_layers: 20, decoder_layers: 20,
  max_position_embeddings: 1048576}
moe: {routed_experts: 384, routed_per_token: 6, shared_experts: 1}
attention: {window_size: 128, index_topk: 512,
  candidate_topk_blocks: 2048, candidate_block_size: 8,
  main_kv: fp4_e2m1, scale_group: 16}
runtime: {hc_mult: 4, sinkhorn_iters: 20, engram_layers: [1, 14],
  mtp_layers: 3, dspark_targets: [37, 38, 39]}
cache: {global_kv_bytes_per_token: 890,
  swa: bounded_replay, swa_persistent: false}
evaluation: {datacurve_exact_row: null, harness: not_applicable}
```

其中 `890 bytes/token` 仍只是模型卡对 global KV 的口径；FP4 scale、indexer K、candidate mask、SWA ring、未封存尾块、workspace、通信 buffer 和权重不在这个单值中。`cache.swa_persistent: false` 也不能理解为“没有恢复成本”：技术报告描述的部署方案把 SWA 状态放入每台机器约 10% host DRAM 的短 TTL 分布式池，global KV 另放持久 cache，缺失时再做 bounded replay；约 10% 与至少 72 小时是报告配置，不是所有后端的默认 SLO。

EPD（Encoder–Prefill–Decode）把视觉编码、prefill 和 decode 变成可独立扩缩、可重叠的服务边界。生产调度器至少需要传递媒体顺序、encoder/global KV handle、压缩块位置、SWA replay metadata、CSA2 mode/index metadata、reasoning effort 和工具权限；只传一串 prompt 文本无法安全恢复这些状态。

固定 revision 的 [reference inference tree](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277/inference) 公开了 compressor、candidate/indexer、sparse attention、mHC/Sinkhorn、Engram、TileLang FP4/FP8 kernel 和 DSpark forward path，但其 README 明确是 readable reference implementation；`generate.py` 走普通 autoregressive loop，不能把 `forward_spec` 的存在写成已经接入 draft/verify/rollback 的生产 speculative serving。没有完整权重、CUDA/TileLang、目标硬件和 acceptance-length trace 时，manifest 的 `runtime_verified` 应为 `false`。

DataCurve 当前没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，所以评测段应使用 `null`/`not_applicable`，不能迁移 `mini_swe_agent_deepseek_v4_pro_max` 或 `deepseek_v4_flash` 的分数。请求端还要保存 requested alias 与响应 `model` 字段：旧 `deepseek-v4-flash`、`deepseek-v4-flash-vision-exp` 以及带日期的 `deepseek-v4-pro` 兼容路由都不能反向证明当时使用了哪个 checkpoint。

## 32.44 DeepSeek `deepseek-recipe`：协议适配、流式状态与图像安全

V4.1 的 serving manifest 还应固定官方 [`deepseek-recipe`](https://github.com/deepseek-ai/deepseek-recipe/tree/8cadfede7063c896b944e7bae05daa3549ae97ea) commit `8cadfede7063c896b944e7bae05daa3549ae97ea`。该 commit 的源码归档 SHA-256 为 `1116ca33e9dc62a913fb9214578c400f1704e6bca33487f4a4c31b32c67a21a6`，职责是协议转换、V4/V4.1 prompt/tokenizer、流式输出解析和图像预处理，不是 inference engine。

建议把协议链写成：

```text
API schema -> ConversationRequest -> V4.1 renderer -> tokenizer
-> external backend -> InferenceChunk -> StreamProcessor -> protocol event
```

`ConversationRequest`、`StateMachine` 和 `StreamProcessor` 分别属于规范化、跨 chunk 解析和事件生成层。流状态要保留未闭合的 `<think>`、DSML、JSON/stop marker；解析出 tool call 以后仍要经过 schema、租户/资源权限、幂等、执行回执和 verifier。token-ID 流必须附带匹配 tokenizer，不能让缺失 tokenizer 静默降级。

图像 resolver 的 URL/data URL/bytes、并发和字节预算也应进入请求 manifest。固定 commit 的默认限制为 600 张图、32 MiB/图、64 MiB/请求、8 个并发源，HTTP fetcher 默认 5 次重定向、10 秒连接和 60 秒请求超时；但源码明确不做 private/loopback/link-local 过滤，故 SSRF、DNS rebinding、MIME、出站网络和审计仍是宿主责任。`server-py`/`server-rs` 的 mock inference 只能验收协议接线，不能验收权重、CUDA/TileLang、工具副作用或线上 SLO。

README 的未支持项（如 `logprobs`、server-side web search、JSON Schema/strict、`n>1`、Responses storage 和 encrypted thinking）是该 adapter 的版本边界；Serving 层不能把它们直接解释为 V4.1 模型能力不存在。完整来源、哈希和证据边界见 [`deepseek-v4.1-flash-source-notes.md`](../../research/model-update-2026-09/deepseek-v4.1-flash-source-notes.md)。

## 32.45 Kimi K3：固定权重清单与双状态 cache

K3 的 hybrid serving 需要先做 artifact gate，再做 runtime gate。固定 HF revision `f831ab66814297da540d832a5235f8e904f29d06` 的 `model.safetensors.index.json` 记录 `497,220` 个 tensor 和 96 个连续分片；零依赖审计脚本 [`kimi_k3_manifest_audit.py`](../../research/model-update-2026-09/code/kimi_k3_manifest_audit.py) 会检查分片连续性、93 层映射、69/24 attention partition、896 experts 覆盖和 MXFP4 packed/scale 配对。

```yaml
model: kimi-k3
revision: f831ab66814297da540d832a5235f8e904f29d06
artifact: {shards: 96, tensors: 497220, full_weights_downloaded: false}
architecture: {layers: 93, kda_layers: 69, full_attention_layers: 24}
moe: {routed_experts: 896, experts_per_token: 16, shared_experts: 2}
quantization: {format: mxfp4-pack-quantized, bits: 4, packed_scale_pairs: 247296}
cache:
  full_attention: [key_cache, value_cache]
  kda: [conv_states, recurrent_states]
```

这个 manifest 特意保留 `full_weights_downloaded: false`。HF API 的 `safetensors.total` 是 U8/BF16/F32 dtype 统计，index 的 `metadata.total_size` 是 packed 文件口径；它们都不能替代真实下载、校验和加载日志。

源码显示两条 cache 生命周期不同：full-attention/Gated MLA 通过 `past_key_values.update()` 追加 K/V；KDA 在带 cache 的单 token decode 中走 `fused_recurrent_kda`，把短卷积和递归 state 写回动态 cache；prefill/chunk 走 `chunk_kda`，并用 `cu_seqlens` 描述变长输入。因而 scheduler、prefix cache、beam reorder 和故障恢复不能只传一个 KV length。

部署验收应至少注入：MLA cache 丢失、KDA state 丢失、prefix-match unit 改变、KV dtype/backend 改变、revision 不一致、变长 batch 边界和 tool-call parser 漂移。任何恢复都要在 schema、权限、幂等、执行回执和 artifact verifier 之后才允许产生外部副作用。FlashKDA 的仓库 benchmark 仍是局部 kernel 结果，不能替代 hybrid cache 正确性、端到端吞吐或线上 Agent acceptance。

## 32.46 Kimi K3：docs/API、registry、recipe 与 stable wheel 的证据层级

K3 的 vLLM 资料已经出现多个“看起来像完成”的信号，但它们回答的是不同问题。面试或上线审计不能把它们压缩成一个 `supported=true`：

| 证据 | 能证明什么 | 不能证明什么 |
|---|---|---|
| stable supported-models 页面 | stable 文档目录知道 `KimiK3ForConditionalGeneration`、`Kimi-K3` 和 `moonshotai/Kimi-K3` | 当前 stable wheel 已在目标卡加载完整 MXFP4 权重、cache 恢复和工具调用都通过 |
| stable K3 API 页面 | 文档 API 暴露 `KimiK3ForConditionalGeneration`、`KimiK3MTP` | 当前安装包版本、NVIDIA/ROCm 分支覆盖、MTP/DSpark 的真实 acceptance |
| v0.29.0 / main `registry.py` | PyPI stable `0.29.0` 对应 tag 与 main 源码 registry 已登记 K3、MTP、DSpark 等入口 | stable release 中有代码就等于每个 registry class 都能在目标硬件端到端运行 |
| `kimi_k3/__init__.py` | package 入口根据 `current_platform` 分流 NVIDIA 与 ROCm，并避免 TPU 主动加载 GPU 实现 | NVIDIA/ROCm kernel、通信拓扑、权重加载和故障恢复已经完成 profiling |
| vLLM recipe YAML | 给出特定硬件、CUDA/driver、TP/TEP/DEP/PP、DCP、hybrid cache 和 parser 风险的部署组合 | recipe 已进入 stable wheel、所有硬件通过 SLO、线上 tool-call acceptance 已知 |
| FlashKDA README/Atom | KDA kernel 的依赖、state API、变长输入和局部 benchmark；当前 commit 仍为 `7afb9f454f160a6c4bbc0999beca0a8c40a38934` | K3 的完整端到端吞吐、MLA/KDA 双状态恢复或生产 Agent 成功率 |

因此 K3 的 serving gate 应按顺序拆开：

```text
docs/API discoverability
    -> source registry and platform branch
    -> pinned recipe and dependency gate
    -> stable wheel/version gate
    -> full-weight load and tensor-shape gate
    -> MLA cache + KDA recurrent/conv-state recovery
    -> target GPU profiling and fault injection
    -> tool schema/retry/idempotency/permission/verifier acceptance
```

`KimiK3MTP` 和 `K3DSparkModel` 出现在 registry，结合 PyPI `vllm 0.29.0` 与 v0.29.0 tag，可以说明 stable release/source 中有对应代码入口；它们不能让我们把 MTP 的 draft/verify/rollback，或 DSpark 的 speculative acceptance，写成 K3 已经线上验证。尤其是 K3 recipe 仍明确写成 `Pre-release` 并要求 K3-enabled nightly/image，因此“stable release 有实现”与“目标环境 stable runtime 已验收”必须分栏记录。

面试追问可以这样回答：如果只拿到 stable API 页面，我会报告“实现可发现”；如果拿到 main registry，我会报告“源码入口已登记”；如果拿到 recipe，我会报告“存在特定环境的部署路径”；只有在固定 wheel、revision、权重加载日志、目标硬件 profiling、双状态恢复回归和线上 verifier 通过后，才会报告“生产 serving 闭环”。

PyPI 的 `vllm 0.29.0` 还增加了一层可审计证据：release metadata 给出 wheel 文件、大小、上传时间和 SHA-256，v0.29.0 tag 给出 registry/package/model 源码。它比“网页列出类名”更强，但仍是 artifact/source evidence；没有安装日志、完整权重、目标 GPU profile 和线上回归，不能升级成 runtime acceptance。

## 32.47 GLM-5.3-Flash：双状态池、MTP 与 multimodal serving gate

GLM-5.3-Flash 是一个很适合面试 serving 设计的例子，因为它把“模型结构里的混合状态”和“Agent 请求的多阶段状态”同时带进了 runtime。一个只记录 `model_id`、`context_length` 和 `kv_tokens` 的 manifest 不够用。

### 32.47.1 两个 cache/state pool 要分开调度

SGLang 的 Flash recipe 同时描述 paged KV pool 和 KDA state pool：前者服务显式 attention 的分页 K/V，后者服务 KDA/线性路径的递归状态。两者的分配、回收、prefix reuse、dtype/backend 和故障恢复并不相同；页面还提示 KDA state pool 可能先成为并发瓶颈。

因此一个 session 的 runtime 状态至少应抽象为：

```yaml
model_revision: pinned
kv: {pool: paged, dtype: bf16_or_fp8, page_table: opaque}
kda_state: {pool: state, handle: opaque, length: integer}
mtp: {enabled: boolean, position: integer, acceptance: unknown}
session: {id: opaque, prefix_key: stable_or_miss}
```

这里的 `kda_state` 不是把 KV cache 换一个名字。它是递归状态，可能有不同的 shape、写回时机和 batch 合并规则。恢复时如果只恢复 KV page table，不恢复 KDA state，服务可以继续吐出 token，却不再具有相同的历史状态；只恢复 KDA state 而丢失 sparse/indexer metadata，则远程精确检索也可能改变。

### 32.47.2 MTP 需要 capability、policy 和 acceptance 三层

SGLang cookbook 给出的低延迟路线使用 MTP `5/1/6`，高吞吐路线可以关闭 speculative decoding。runtime manifest 应区分：

| 层 | 要回答的问题 |
|---|---|
| capability | 当前 model/revision 与 runtime 是否暴露 MTP 接口？ |
| policy | 这个请求的延迟、显存和工具循环是否值得启用 speculative？ |
| acceptance | 固定硬件、batch、长度和 verifier 后，draft token 实际接受率是多少？ |

工具调用和多轮 Agent 尤其不能只看 token 接受率。一个被接受的 draft 仍可能在 tool-call boundary、结构化 JSON、停止条件或权限检查处触发回滚；因此要把 accepted tokens、rollback tokens、tool parser errors、执行等待和最终 task success 分开统计。

Transformers 的 GLM5-Next 文档明确不包含 MTP layer，而 vLLM/SGLang 可以在 serving 路径提供 MTP 支持。由此得到一个通用结论：框架的 speculative feature surface 不能反向证明 checkpoint 自身含有同名 layer。

### 32.47.3 KV dtype 和 DSA backend 的成对约束

Flash recipe 给出的硬件组合是 Blackwell 的 FP8 KV + TRT-LLM DSA，以及 H100/H200 的 BF16 KV + TileLang DSA；FP8 KV + TileLang DSA 被标为无效组合。

调度器的 capability probe 不应只问 `supports_fp8=true`，还要联合检查：

1. GPU family/compute capability；
2. KV dtype、scale dtype、page/block layout；
3. DSA backend、kernel 编译版本和 runtime dependency；
4. MTP、prefix cache、quantized KV 与 KDA state 的组合；
5. long-context recall、工具回执恢复、TTFT/TPOT 和 p99。

把 KV dtype 作为一个孤立的低精度开关，可能得到“能启动但召回/数值不对”的错误部署。recipe 页面报告的 GB300 throughput、KV capacity 和 GSM8K 差异属于发布方结果，不能直接填进本地 SLO。

### 32.47.4 EPD 与 PD 的状态传输

Flash 的多模态 serving 还要处理 encode 阶段：

```text
Encode pool -> visual representation
Prefill pool -> KV/state metadata
Decode pool -> text/tool events
```

跨池消息不能只有 prompt 字符串，至少还要带 `request_id`、model revision、媒体顺序、视觉 token 布局、position offset、KV/state handle、cache dtype、取消/超时、失败重算和权限上下文。

SGLang 页面报告视频默认 2 FPS、最多约 240K visual tokens，并给出 4x GB300 encoder disaggregation 的 decode-gap 结果；这些数值说明了阶段隔离的目标。页面同时标出 PD 当前只有 dummy weights 机械验证，没有 load/accuracy 验证，普通 speculative decoding 与不同 TP 数值正确性也未完成验证。

因此部署状态要按 gate 记录：

```text
recipe discoverable
  -> dummy-weight wiring
  -> full-weight load
  -> numerical correctness
  -> EPD/PD recovery
  -> target hardware profile
  -> tool/schema/permission/verifier acceptance
```

“有命令”只通过第一层；不能在 dashboard 中写成 production-ready。

### 32.47.5 FlashX 和三种实现证据不要混成模型发现

`GLM-5.3-FlashX` 是 Z.ai 文档列出的关联服务入口，约 200 tokens/s；它没有在两个排行榜中形成新的 canonical 模型条目。因此 inventory 只保留 GLM-5.3-Flash，FlashX 的速度、套餐配额和 endpoint 差异进入 service manifest。

同样要把证据分成四层：

| 证据 | 可以说明 | 不能说明 |
|---|---|---|
| 模型卡/config | 模型层数、MoE、attention 类型和公开字段 | 目标后端已通过 SLO |
| Transformers docs | 基础模型接入和前向能力边界 | MTP serving acceptance |
| vLLM/SGLang recipe | 某版本/硬件路线存在部署入口 | 本机完整权重、数值和故障恢复 |
| 本地/线上 gate | 目标 revision、硬件和 workload 的实测结果 | 可迁移到其他 provider/版本 |

面试时如果被问“GLM-5.3-Flash 是否支持 MTP”，稳妥回答应带上限定：模型公开配置、框架实现和目标 serving acceptance 是三个问题；SGLang/vLLM recipe 的实现入口不能替代固定硬件上的接受率、回滚、tool-call 和 verifier 回归。

### 32.47.6 stable/main/source evidence 的 capability matrix

GLM-5.3-Flash 的实现证据应按下表汇报，而不是只报一个“框架支持”布尔值：

| 来源 | 当前能证明什么 | 仍不能证明什么 |
|---|---|---|
| SGLang `v0.5.20` tag | 固定 stable source 中存在 `glm5_next.py`、配置、NextN 和相关测试入口 | 本机依赖、完整权重、目标 GPU 的数值/性能 |
| SGLang `main` | projection/KDA/mHC/量化演进，以及 B200/H200 的测试门禁定义 | mutable main 已进入发布 wheel，或测试已由本轮实际运行 |
| vLLM `main` | `glm5next` 的 indexer/tail/KDA/MTP runtime 结构与源码级状态边界 | v0.29.0 stable tag/wheel 已含该专属路径，或目标硬件 acceptance |
| vLLM recipe | 特定版本、硬件和依赖的部署路线 | recipe 所列结果可迁移到其他卡、版本或 provider |
| full-weight target run | 固定 revision 的 load、数值、恢复和性能结果 | 没有同时固定 workload/verifier 时的生产 SLO |

面试中应优先追问四个字段：源码/recipe 对应的 commit 或版本、权重 revision、cache/state manifest、验收日志。尤其是 vLLM stable 的负证据要明确写成“当前快照未检出专属路径”，不能扩大为“vLLM 永远不支持”；同样，main 的源码存在也不能缩写成“stable 已支持”。

## 32.48 Claude Sonnet 5：adaptive thinking 与 Agent serving 的状态账本

Sonnet 5 的 serving 不应把 `effort` 当成一个固定 batch shape 或固定 thinking token 配额。`effort` 是行为信号，`max_tokens` 是单次响应硬上限，而一个 Agent loop 还要额外记录工具结果、重试、compaction、缓存和 verifier 成本。调度器应把这些维度分别写入请求 manifest：

```text
model/revision/provider/effort/max_tokens
input/thinking/output/tool tokens
cache prefix/hit, context trigger, compaction item
tool latency, retries, permission result, verifier result
task success, p95 latency, unit-success cost
```

Anthropic System Card 的 BrowseComp 配置使用 10M token limit 并在约 200K 触发 compaction，说明长任务结果同时受模型、上下文管理和工具 harness 影响。评测 serving 时要对比 compaction 前后 TTFT/TPOT、prefix cache 命中、恢复后的工具协议完整性和重复副作用；1M context 只代表接口上限，不代表 KV cache 已无限扩展或远程证据检索必然可靠。

另外，programmatic tool calling 可以减少模型往返，但工具执行仍由宿主控制；computer toolset、工具 schema、沙箱和权限策略也不能由模型页字段自动推导。只有固定 provider、runtime、工具版本、权限、硬件和 verifier 后的端到端回归，才可以作为 serving acceptance。

## 32.49 Grok 4.7：opaque state replay 与 compaction serving

Grok 4.7 的 Responses serving 需要把 `reasoning.encrypted_content` 当作不可解释、不可编辑的协议 item 原样回放。它不等价于可见 CoT、prompt cache 或 GPU KV cache；服务适配器应至少校验 model ID/revision、item 顺序、`call_id`、工具回执和 provider 是否支持同一状态格式。Chat Completions 没有同样 ciphertext 时，不能静默复用 Responses 的恢复逻辑。

Reasoning 文档同时提供可见的 `response.reasoning_text.delta` 和 `response.reasoning_summary_text.delta` 流事件；它们是开发者观察面，不是可以替换 encrypted item 的完整状态。Grok 4.7 还会把服务端工具的加密输出放入同一回放责任域，因此 trace manifest 应区分 `encrypted_state`、`summary_event`、server-side tool receipt 和 client-side tool receipt。`store` 决定 `previous_response_id` 的服务端存储行为，不能据此省略客户端事件账本。

`POST /v1/responses/compact` 返回单个 opaque compaction item。调度器应在 request 仍未超出 context limit 时触发它，并把压缩触发原因、旧事件索引、工具副作用、权限结果、artifact handle 和新 item digest 写进 manifest。compaction 不能挽救已经超限的请求，也不能把已消耗预算或已提交副作用清零。

一个最小 serving gate 可以写成：

```text
provider/model capability probe
  -> encrypted-state compatibility
  -> compaction replay test
  -> tool receipt/idempotency test
  -> independent verifier and artifact recovery
  -> target hardware/latency/cost profile
```

官方 500K context、150 RPS、50M TPM 和价格字段只是服务合同；它们不能替代固定 provider、请求长度、工具 schema、缓存命中、p99 和任务成功率下的本地验收。详细来源和哈希见 [`grok-4.7-source-notes.md`](../../research/model-update-2026-09/grok-4.7-source-notes.md)。

## 32.50 Qwen3-Omni：Thinker/Talker 的双状态 serving

Qwen3-Omni 的 serving 不能只维护文本 KV cache。Thinker 侧要保存多模态 token、TM-RoPE timestamp、chunk offset、RAG/tool/policy/verifier 事件；Talker 侧还要保存首码本/残差码本状态、Code2Wav 的因果上下文和 audio packet sequence。取消、重试、断点恢复和 prefix hit 都要同时检查两类状态。

可以把服务 gate 写成：

```text
media decode / timestamp validation
  -> AuT + vision preprocessing
  -> Thinker chunked prefill
  -> tool/policy/verifier gate
  -> Talker codebooks
  -> Code2Wav waveform
  -> packet streaming / cancel / replay
```

官方 README 当前说明 vLLM 主要支持 Thinker，Instruct 音频输出仍在实现推进中；因此仓库加载、Thinker 服务、Talker code、Code2Wav 和目标硬件流式验收不能合并为一个“已支持”字段。报告的 `234/547 ms` 首包数字必须与本机 TTFT、首包、TPOT、p95/p99 和并发条件分栏。

## Qwen3-VL：视觉 prefill 与 DeepStack serving gate

Qwen3-VL 的 serving 不能只记录文本 KV cache。请求还要保存视觉 grid、temporal/height/width position、Video Timestamp、DeepStack residual、MoE expert placement、tool schema 和 GUI observation。一个可审计的门禁是：

```text
processor / visual preprocessing
  -> full-weight load and numerical correctness
  -> M-RoPE / DeepStack / timestamp replay
  -> visual prefill + TP/EP + KV/cache profile
  -> tool permission / executor / verifier
```

HF config 或 Transformers raw main 只能证明实现入口和字段存在，不能证明完整权重、生产视觉/稀疏 kernel、目标硬件 profiling 或 GUI/tool acceptance。AA 的 `235B total / 22B active` 也不能直接替代 resident expert、KV、workspace、通信和并发账本。完整证据见 [`qwen3-vl-source-notes.md`](../../research/model-update-2026-09/qwen3-vl-source-notes.md)。

## 32.51 Qwen3.7 Plus：托管多模态与区域化 tool serving

Qwen3.7 Plus 没有公开权重或专属 serving kernel，本节讨论的是官方托管合同和部署验收方法。请求进入推理层后，至少要分开记录：文本 token、图像/视频预处理、媒体 token 估算、tool schema、thinking/output reservation、prefix/cache 状态，以及 GUI/mobile action 的执行回执。

一个 source-aware serving gate 可以写成：

```text
alias/snapshot/provider capability probe
  -> region/scope/API-key check
  -> media decode/resize/timestamp manifest
  -> visual/text prefill + context budget
  -> function/structured/search schema validation
  -> permission/executor/observation/idempotency
  -> verifier/artifact + cost/latency profile
```

官方页面给出 1M context、991,808 max input、131,072 max output，thinking max input `983,616` 和 max chain-of-thought length `262,144`。调度器不能把这些数相加当成一个输入容量；视觉 token、工具结果、缓存元数据和输出预留都占用请求账本。输入超过 256K 后，部分 region 的价格也会切换档位，因而成本报告必须绑定 region、scope、cache、batch 和 provider。

Function Calling 和 Structured Outputs 是协议门，不是执行门；Web Search、Prefix Completion 和 Context Caching 是按 region/scope 开关的服务能力。Context cache 也不是 GPU KV cache：前者复用 provider 前缀，后者服务于当前执行图，媒体 revision、tool schema 或 compaction 的变化可能分别导致不同层失效。

Qwen3.7 Plus 的视觉 Agent serving 最终要验收四个结果：媒体是否正确预处理，模型是否产出合法且有证据的 action proposal，宿主是否在权限范围内执行并回灌 observation，以及独立 verifier 是否确认真实 artifact。AA 约 1M context 和产品页的 GUI/mobile 定位不能替代目标设备的 TTFT、TPOT、p95、失败恢复、越权率和任务成功率实测。

## 32.52 Kimi K3：SGLang stable/main 的双状态与 EP serving

Kimi K3 的 SGLang 证据要与 vLLM 证据分开记账。`v0.5.20` tag 已有 K3 文本/视觉入口；`main` 的文本实现继续演进 shared-expert TP、DP/EP token shard、ModelSlim fused weight mapping 和 KDA overlap。它们都不能替代完整权重、目标硬件和 Agent acceptance 验收。

### 32.52.1 版本和实现入口

固定源码快照：

| 版本 | K3 文本源码 | K3 视觉源码 | 含义 |
|---|---|---|---|
| `v0.5.20` | 168,114 bytes，blob `b0ede48c88264d518351a66abf623f1bcf8a730e` | 32,790 bytes，blob `c423027994645e4a839ec47bd26bcc1b2a7cb012` | stable tag 的实现入口 |
| `main` | 171,101 bytes，blob `383a6f47812bccd1cb91b76814cd0730ff945dd7` | 同上，逐字节一致 | 当前分支的文本 runtime 演进 |

`v0.5.20` tag commit 为 `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`。`main` 是可变分支，所以部署 manifest 必须保存抓取时间、blob 和 SHA-256，不能只保存分支名称。SGLang 类名或文件存在，只能证明 source entry；stable wheel、完整权重和目标卡仍需单独验证。

### 32.52.2 KDA state、MLA cache 和 kernel fallback

K3 是 hybrid attention 模型。服务状态至少分为：

```yaml
model: {revision: pinned, sglang_blob: pinned}
mla: {key_cache: separate, value_cache: separate, dtype: pinned}
kda: {conv_state: separate, recurrent_state: separate, dtype: pinned}
execution: {mode: prefill_or_chunk_or_decode_or_verify, backend: pinned}
acceptance: {weight_load: unverified, recovery: unverified, profile: unverified}
```

SGLang `KimiK3DeltaAttention` 在 load/post-load 阶段检查卷积、`A_log`、`dt_bias`、head layout 和 dtype，命中条件后才把静态参数交给 fused KDA decode；否则回退普通链路。decode、prefill/chunk 和 target verify 的 beta/gate 处理不能混为一个“attention path”。MLA `key_cache/value_cache` 与 KDA `conv_states/recurrent_states` 也不能压缩成一个 `kv_length` 字段。

故障注入至少包括：只恢复 MLA cache、只恢复 KDA state、修改 state dtype、改变 prefix-match unit、切换 backend、改变 TP/EP topology，以及让 fused capability gate 失配。每一种情况都要检查拒绝、回退、数值一致性和是否会错误复用跨请求 buffer。

### 32.52.3 LatentMoE 和 EP/A2A 的通信账本

K3 routed expert 的 serving 路径可以写成：

```text
full hidden
  -> FP32 router / grouped Top-K
  -> latent down projection
  -> MegaMoE/DeepEP/Mooncake/Ascend-FuseEP/MoRI A2A
  -> expert GEMM in latent width
  -> latent reduce + RMSNorm
  -> full-width up projection
  -> shared experts + tail add
```

在 A2A/SP-MoE 路径，attention TP 的 reduce-scatter 先形成 token shard，MoE 只 dispatch 当前 rank 的 rows，尾部再 all-gather；如果把普通 full batch 误送给所有 EP rank，就会重复 dispatch 和放大通信。`main` 还允许 shared experts 使用独立 TP group，在该分支执行 gather、shared MLP 和 reduce-scatter；`v0.5.20` 的兼容路径更专用，不能用一个开关名称代替实际 group/shape 条件。

SBO 会把 shared expert 计算放到 side stream，与 routed A2A/latent tail 重叠。验收指标不能只看总 tok/s，还要记录 A2A bytes、shared gather/reduce-scatter、stream wait、p99、显存峰值、空闲 rank、capture/replay 和 tool-call 边界的回滚。

### 32.52.4 视觉和 Agent 层仍是独立门禁

K3 视觉文件在 stable/main 快照中一致，包含 MoonViT3d、2D RoPE、变长 `cu_seqlens`、按形状选择的 SDPA/Triton/FA4/FlashInfer-CuDNN、fused RoPE 和可选 CUDA graph。这能证明源码设计路径存在，不能证明视觉权重、grid/patch 对齐、视觉数值、图像 owner 分片或多模态 stream 已验收。

最后一层仍是 Agent runtime：K3 偶发输出宿主 parser 不期望的 tool-call 格式时，schema validation、权限、retry、幂等、executor 和 verifier 必须在模型之外完成。应把源码 entry、full-weight load、dual-state recovery、target profile、tool acceptance 和 artifact verifier 分列，不能用 `supported=true` 一项覆盖全部状态。

### 32.52.5 K3 upstream 提交与部署门禁

SGLang 的 K3 commit history 给出了一条实用的部署审计顺序：先确认 KDA deferred gate 的投影责任，再确认 CUDA graph 的 side-stream fork/join，随后检查 expert weight loader 的命名映射和 routing dtype，最后核对 PP/PD/DCP/DSpark 的 process group 与硬件 backend。近期提交包括 8ac19cc（deferred KDA gate）、c4d3770（CUDA graph stream）、c2c3629（O(1) expert lookup）、f4c2563（PP prefill + DCP decode + DSpark）、8ac39c6（Ascend A5）和 cb32dbc（ROCm fused KDA projection）。

这组提交不能被压缩为“性能优化列表”。它们分别影响状态语义、异步执行、权重加载、跨阶段调度和设备分支。部署 manifest 应保存源码 commit/blob、model revision、quantization mapping、hardware/backend、TP/EP/PP/DCP group、prefill/decode/verify mode、MLA/KDA state、fallback 和 tool-call parser version。只有在完整权重加载、数值对照、目标卡 profile、故障恢复和 tool/verifier 测试都通过后，才可以从 source evidence 进入 runtime acceptance。

## 32.53 GLM-5.3 标准 DSA：MLA、Indexer Cache 与 serving gate

标准 GLM-5.3 的 serving 路径要和本章前面的 GLM-5.3-Flash 分开。标准版的固定 HF config 是 `glm_moe_dsa` / `GlmMoeDsaForCausalLM`，当前 vLLM 和 SGLang 都把它接到 DeepSeek-V3.2 DSA 共用 runtime；它不是 `glm5_next` 的 `RadixLinearAttention`/KDA/视觉路径。

### 32.53.1 状态账本

标准 DSA 请求至少需要拆出以下状态：

```yaml
model:
  revision: aca966e4e02791568aa6a4ced368624b3d897f42
  class: GlmMoeDsaForCausalLM
  runtime_entry: deepseek_v32
attention:
  mla_latent_cache: pinned_dtype_and_layout
  rope_state: interleaved_indexer_rope
indexer:
  full_layers: 21
  shared_layers: 57
  top_k: 2048
  topk_indices: per_layer_or_reused_state
  causal_offset: per_request
mtp:
  index_share_for_iteration: true
acceptance:
  full_weight_load: unverified
  numerical_correctness: unverified
  index_recall: unverified
  evidence_recall: unverified
  hardware_profile: unverified
  tool_verifier_slo: unverified
```

`topk_indices` 不能被压缩成普通 `kv_length`。Full indexer 层重新产生候选，Shared 层复用前一 Full 层的选择；prefix reuse、batch reorder、chunked prefill、decode 和 speculative verify 都可能改变这些索引的生命周期。恢复时要同时检查 latent KV、RoPE offset、候选索引、block table、MTP iteration 和请求版本。

### 32.53.2 stable/main 不是同一个能力等级

当前可以观察到的证据分层如下：

| 层级 | 可证明 | 不能证明 |
|---|---|---|
| HF config/revision | 模型类、层数、专家和公开 indexer 字段 | 完整权重已下载、召回和生产性能 |
| Transformers main | interleaved indexer RoPE、Full/Shared top-k、sparse mask 的基础前向语义 | vLLM/SGLang kernel、量化和硬件正确性 |
| vLLM `v0.29.0`/main | `GlmMoeDsaForCausalLM -> deepseek_v32` registry 与 runtime source entry | 本机 wheel、目标卡、main 专属优化或线上 SLO |
| SGLang `v0.5.20`/main | `is_glm_moe_dsa`、DeepSeek-V3.2 路径和跨层 top-k 状态 | stable/main 的数值等价、完整权重和生产 acceptance |
| 本地/线上 gate | 固定 revision、硬件和 workload 下的真实结果 | 结果自动迁移到 Flash、其他 provider 或其他版本 |

vLLM main 的 PCP/DCP、`SparseCacheRole.INDEXER`、HiSparse 和 logical top-k 应记录为 main 分支差异；不能因为 recipe 或 registry 写着 v0.29.0+，就宣称每个 v0.29.0 wheel 都具备相同实现。

### 32.53.3 稀疏 serving 的质量门禁

DSA 的性能报告至少要同时包含：

1. index recall：相关历史位置是否进入候选集合；
2. evidence recall：最终显式 attention 实际读到的候选是否包含任务证据；
3. 数值对照：dense reference、sparse path、不同 dtype 和不同 batch 的 logits/输出误差；
4. serving 指标：TTFT、TPOT、p95/p99、显存峰值、索引计算开销和通信开销；
5. Agent 指标：工具回执恢复、长轨迹状态保真、任务成功率、verifier 通过率和失败重试。

只报告 `index_topk=2048` 或理论 attention FLOPs，不足以证明模型服务质量。Shared layer 复用会减少 indexer 计算，也可能放大候选漏检；因此应对 full/shared 比例、tail 保留、dense fallback 和 chunk 边界做消融。

### 32.53.4 与 Flash 版的边界

标准版的状态重点是 MLA latent cache、indexer top-k 和 Full/Shared 跨层选择；Flash 版才需要额外维护 KDA/linear state、视觉 token、paged KV 与 EPD 的跨池 metadata。两套账本不能合并为一个“GLM-5.3 cache”。

最终 gate 仍应保持：

```text
source entry
  -> full-weight load
  -> dense/sparse numerical check
  -> index/evidence recall
  -> cache and MTP recovery
  -> target hardware profile
  -> tool/schema/permission/verifier/SLO
```

本节所有 `unverified` 字段都必须在实测后替换为带 revision、硬件、dtype、batch、上下文和失败样本的结果；不能用源码存在、模型卡字段或排行榜分数代替 serving 验收。

## 32.54 Claude Opus 5.5：长任务成本与安全 fallback 的 serving 账本

Anthropic 对 Opus 5.5 的公开描述同时包含更少 token/步骤、更低 cache read 成本和 Cyber/Life Sciences/Distillation safeguards。Serving 层不能把这些信息压缩为一个“模型更便宜”的字段，而要把实际执行路径展开：

```yaml
request:
  model: claude-opus-5-5
  effort: medium | high | xhigh | max
  fallback_policy: default | restricted | none
  tools: versioned_allowlist
  cache_prefix: digest_and_ttl
  task_budget: tokens/steps/wall_clock
execution:
  actual_model: primary_or_fallback
  refusal_category: null_or_class
  tool_calls: count_and_receipts
  retries: classified_and_bounded
  verifier: version_and_result
outcome:
  artifact_digest: required
  success: verified_only
  cost: input/cache/output/tool/retry/fallback
```

Opus 5.5 的 benchmark 还显示了一个常见陷阱：同一发布页中，max effort 的质量结果、xhigh 的 Terminal-Bench 结果和 medium/default 的成本曲线并存。调度器或评测器必须把 effort、provider、工具和 fallback 绑定到 trace；不能将 `max with fallback` 当成独立 checkpoint，也不能把 fallback 完成的任务写成 primary model 的 acceptance。

WANDR、OSWorld、Terminal-Bench 和 AutomationBench 的共同点是模型、工具宿主、权限、环境和 verifier 形成闭环。模型没有因为“研究能力”描述而自动获得网络访问；web search/fetch、code execution、sandbox、来源审计、超时、幂等和最终引用检查仍属于 harness。当前 Opus 5.5 没有精确 DataCurve Agent 行，不能迁移 Opus 5 或 Fable 5.1 的 serving 结果。

Opus 5.5 的 serving adapter 还必须处理官方迁移契约，而不是只替换 model ID：adaptive thinking always-on，`thinking.type=disabled/enabled` 需要在客户端提前阻止；`tool_choice=any/tool` 应转成 `auto` 加 strict schema 或 structured outputs；Claude API/Google Cloud 需要使用 `computer_toolset_20260801`。响应解析按 block type 进行，并保存 thinking binding、消息前缀 hash、工具 schema 版本和 compaction block；tool-call 间的进度默认可能不在 text block 中。

fast mode 是同一模型的更快 inference configuration，不是新的权重。它使用独立 beta header 和限流，`usage.speed` 是实际采用的速度证据，且 premium price 与 prompt-cache/data-residency modifier 叠加。Serving SLO 应同时记录 TTFT、output tokens/s、speed、429/529、input/output/cache/tool/fallback cost；不能只用一个平均 latency 或把 fast mode 当成新的榜单模型。来源：[Claude Opus 5.5 model page](https://platform.claude.com/docs/en/models/opus-5-5/overview.md)、[Fast mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode)。

System Card 数字也应进入 serving manifest，而不是单独放在 marketing 表格：Terminal-Bench 4.0 `66.36%` 绑定 xhigh 和 Claude Code `--bare`；OSWorld partial/strict `81.8%/48.7%` 绑定 action 上限、截图、runs 和 >100K tokens compaction；Cyber ACE `301/410 = 73.4%` 明确是在关闭 safeguards 下取得。一个可回放的记录至少需要：

```text
model/revision/actual_model/fallback_reason
effort/safeguard_state/tools/permissions/environment
input/cache/reasoning/output/tool/retry tokens
compaction/screenshot/handoff/verifier/artifact
TTFT/TPOT/derived_latency/wall_clock/429/529/cost
```

这样才能区分 inference 配置改变、上下文压缩、工具宿主、风险 fallback 和真正的模型能力变化。System Card 报告的多 Agent 约 `2.7x`/`2.8x` speedup 是 derived latency，不能替代目标硬件、并发和真实 wall-clock profile。

## 32.55 GPT-6 Sol：长上下文 serving 与 whole-request 成本

GPT-6 Sol 的官方模型页给出 `1,050,000` context、`922,000` maximum input 和 `128,000` maximum output。Serving 层不能把这三个数字合成一个“可输入 1M token”的开关：reasoning tokens、工具 schema、工具结果、历史 output items 和可见输出共同占用窗口，`max_output_tokens` 耗尽时甚至可能在可见文本前返回 incomplete。

成本账本还必须处理 whole-request threshold。Sol 的 input `$2/M`、cached input `$0.20/M`、cache write `$2.50/M`、output `$10/M` 是模型页价格快照；当 input 超过 `272K` 时，整次请求的 input/cache 费率乘 2、output 费率乘 1.5，而不是只对超出的 token 加价。Batch/Flex 价格为标准的 50%，Fast mode 为适用价格的 2x。一个简单的服务端估算可以先按请求分桶：

```text
if input_tokens > 272_000:
    input_rate *= 2
    cache_rate *= 2
    output_rate *= 1.5

cost = input_tokens * input_rate
      + cached_tokens * cache_rate
      + cache_write_tokens * cache_write_rate
      + output_tokens * output_rate
      + tool_call_cost
      + retry_cost
```

这段是账本模型，不是 OpenAI 账单 API 的替代品；还要记录 regional processing、数据驻留、Batch/Flex、Fast mode、工具调用和失败重试的实际合同。

### 32.55.1 Tool search、compaction 与 cache 的交互

工具目录支持不等于每轮把所有 schema 注入上下文。tool search 可以延迟加载工具定义，减少初始 schema token，但加载后的 namespace、schema hash、权限和工具版本会成为会话状态。compaction 又会替换早期上下文并产生 opaque item；stateless chain 要原样回放 output items，`previous_response_id` chain 不要手工裁剪。Serving trace 至少保存：

```text
model/snapshot/mode/effort
input/output/cached/cache_write/reasoning tokens
tool registry/search calls/schema hash/permission/executor receipt
compaction trigger/item/cache prefix/replay policy
TTFT/TPOT/p95/429/incomplete/verifier/artifact
```

OpenAI 的 Agents 文档还要求把 Agents API、Agents SDK 和 Responses API 的状态所有权分开。因而部署 GPT-6 Sol 时，不能仅凭模型页的 `hosted_shell` 或 `computer_use` 就声称目标硬件、sandbox、网络访问和生产 SLO 已通过；这些必须由对应 executor、权限和 verifier 实测。

资料边界：GPT-6 Sol 当前是 AA 单榜资料级闭环 + 当前时点复验 + local protocol toy，DataCurve 没有精确 Agent 行。本节的成本公式和 serving trace 是官方模型/API 合同驱动的工程账本，不是模型内部架构或真实 provider 性能复现。可运行的 [`gpt6_sol_contract_audit.py`](../../research/model-update-2026-09/code/gpt6_sol_contract_audit.py) 仅验证 whole-request threshold、cache prefix 变化、幂等和 verifier 的本地逻辑。

### 32.55.2 配置更新、compaction 与成本门禁的最小回放

服务端实现或本地 harness 至少应拒绝三类隐性错误：把 `configuration_update` 当成换 checkpoint，把 compaction 后的旧 cache prefix 当成必然命中，把 permission-approved 误记为 artifact 已验证。toy 先保留 canonical compaction item，再用新的 prefix 做 cache lookup；同时把 `reasoning_tokens + visible_output_tokens` 计入 output/context，并对超过 `272K` 的整次请求应用价格倍率。这样得到的是可解释账本，不是 provider profiling。

### 32.55.3 GPT-6 Sol/Luna：EU data residency 与 processing gate

OpenAI 官方 [data residency guide](https://developers.openai.com/api/docs/guides/your-data.md) 明确说明：GPT-6 Sol 与 Luna 的 EU data residency 只对 **Standard processing 的 Responses API 和 Chat Completions** 可用。模型页所说的 `Standard processing` 是数据驻留/服务处理资格；它不同于 GPT-6 `reasoning.mode=standard`，后者是 test-time execution mode。部署 manifest 应分别记录 `processing_mode` 与 `reasoning.mode`，不可因为同名 `standard` 就推导出 residency eligibility。

Regional storage 与 regional processing 也不是一回事：前者约束合同定义的 customer content 静态存储位置；只有 endpoint/model 的 regional-processing 支持表明确列出的组合，才能说明推理在该区域执行。system data（账号、用量、计费与支持元数据、structured-output schema 等）不属于 customer-content residency 保证，Remote MCP 的请求数据由第三方自己的政策管辖；客户或终端用户基础设施的位置还可能导致区域外传输。非美国地区需满足适用的 abuse-monitoring/retention controls，regional processing 在可用区域有 10% uplift。

Batch/Flex 的价格折扣与 Fast mode 的加价不是 residency 资格声明；若 endpoint × region × processing mode 的官方表格没有明确支持，就记为 `not_established`，先验证服务合同再上线。建议对 capability tuple 做显式门禁：

```text
model × endpoint × region × processing_mode
    -> storage_eligible / inference_region / retention_controls / cost_uplift
```

这是平台服务边界，不证明 Sol/Luna 的内部架构、权重路由或法律合规；外部工具、日志、用户设施和合同条款仍需独立审计。

## 32.56 GPT-6 Luna：sibling serving 与全请求计费

GPT-6 Luna 的官方合同给出 `1,050,000` context、`922,000` maximum input、`128,000` maximum output，以及 input `$0.10/M`、cached input `$0.01/M`、cache writes `$0.125/M`、output `$0.50/M`。超过 `272K` input tokens 后，整次请求 input/cache 按 2x、output 按 1.5x；Batch/Flex 为 50%，Fast mode 为 2x。Serving 账本必须把 reasoning、可见输出、工具 schema/result、缓存、compaction、retry 和 verifier 失败成本一起计算。

Luna 与 Sol 共享上述 EU data-residency processing gate：仅 Standard processing 下的 Responses/Chat Completions 有明确支持；`Standard processing` 不等于 `reasoning.mode=standard`。regional storage/processing、system-data exclusions、Remote MCP 第三方边界与 retention/cost controls 统一见 32.55.3。Luna 的 Fast mode、Batch/Flex 价格不能自动视作 EU-residency eligible；部署仍应逐项校验 `model × endpoint × region × processing_mode`，并分别计算它自己的 token 与地域成本。

Luna 与 Sol 不能共享 provider benchmark 或 DataCurve Agent 结果。DataCurve 当前没有精确 `mini_swe_agent_gpt_6_luna_*` 行；本节的 sibling routing、cost ledger 和 replay trace 只建立在官方模型/API 合同上，不证明 Luna 的内部架构、生产 kernel、目标硬件性能或生产 SLO。详见 [`gpt-6-luna-source-notes.md`](../../research/model-update-2026-09/gpt-6-luna-source-notes.md)。

## 32.57 GPT-5.6 Luna：persisted reasoning、compaction 与 cache replay

GPT-5.6 Luna 的 serving trace 需要把模型档位、effort、`reasoning.context`、reasoning item、function call/output、tool-search 结果、compaction item 和 cache 统计分开。`all_turns` 是同家族 opaque reasoning 的可用范围，不是可见 CoT、永久记忆或 KV cache；`current_turn` 可以减少旧 reasoning 的渲染，但旧 item 仍可能留在协议 payload 中。

GPT-5.6 的 prompt caching 以至少 `1,024` 个 visible input token 为门槛，支持 explicit/implicit breakpoint，单请求最多四次 explicit cache write，`30m` 是文档给出的 TTL。compaction 替换早期上下文后，逻辑上相同的下一请求也可能失去旧 cache prefix；因此 serving 账本必须同时记录 `cached_tokens`、`cache_write_tokens`、compaction 次数、恢复延迟、重复工具调用和单位成功成本。

工具搜索的 hosted/client 两种路径还要分别记 `execution` 和 `call_id`。schema 被加载不等于权限授予；permission、sandbox、executor receipt、幂等键和 artifact verifier 仍由宿主负责。无网络 [`gpt56_luna_state_replay_audit.py`](../../research/model-update-2026-09/code/gpt56_luna_state_replay_audit.py) 已覆盖这些拒绝路径，证据等级固定为 `local_protocol_toy`，不代表生产 endpoint 或硬件 profiling。

## 32.58 Kimi K3：adaptive DSpark、cache pointer 与 ROCm 版本门禁

K3 的 hybrid state 让 speculative serving 同时管理 MLA/context KV、KDA recurrent state 和 draft/target verification metadata。vLLM main 的 adaptive DSpark path 用 confidence head 为每个位置输出 draft 接受概率；一个 `k+1` CUDA graph capture 可以承载批内不同 verify 长度，设备 offsets/masks 记录每个 request 的实际边界。由于 adaptive scheduler 会重写 scheduled-token 数，verify row 必须依据 request 状态分类；不能把固定长度 scheduler 的等式继续当成通用规则。

DSpark context KV 还要考虑 buffer 重新绑定。vLLM 2026-09-28 合入的 K3 修复在 cache 层 `data_ptr()` 改变时清掉并重建 context pointer cache，并限制该更新不能在 graph capture 内发生。服务端恢复/重分配测试应显式覆盖：cache owner 变化、地址重用、图重捕获和拒绝 token rollback。固定 `v0.30.0` tag 早于这项修复与 9 月 23 日的 variable-length commit，不能把 main 功能反写成 stable 支持。

ROCm 部署还需要精确匹配 SiTU 权重布局和 AITER kernel。最新 MI355X recipe 选择 SiTUv2 a4w4 FlyDSL 路径，需要相应 vLLM/AITER 版本；带 `A8W4` 的旧 flag 名是兼容 alias，不代表实际执行 a8w4。独立设置 AITER 的 a8w4 dispatch flag而没有同步 vLLM weight-shuffle flag，可能使 kernel 与 packed weight layout 不一致，且没有运行时错误；相关修复 PR 在本次快照仍 open。AITER MLA 对 DSpark 的非因果 draft block 同样有 query-length/dtype capability gate，不支持时应拒绝或切至 `TRITON_MLA`。

版本和结果必须分层：vLLM `v0.30.0` 仍是 stable，含 ROCm non-causal MLA PR #55966，但不含 SiTUv2 a4w4 PR #53940；9 月 29 日最新 `v0.31.0rc1` 与 K3 无关。固定 recipe 指向 ROCm nightly，并引用 SemiAnalysis MI355X AgentX lane。lane 的真实评测使用 block rejection；吞吐专用 sweep 的 `synthetic` 模式用固定 golden acceptance length 跳过 target verification，因此只能讨论受控接受长度假设下的吞吐，不能证明准确性或真实 acceptance rate。权重加载、硬件数值、cache recovery 与生产 SLO 仍待目标机器验收。细节见[Kimi K3 研究笔记](../../research/model-update-2026-09/kimi-k3-source-notes.md#2026-09-29-vllm-recipeadaptive-dspark-rocm-serving-)。
