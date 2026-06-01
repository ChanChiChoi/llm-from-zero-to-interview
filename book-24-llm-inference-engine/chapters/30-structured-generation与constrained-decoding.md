# 第 30 章 Structured Generation 与 Constrained Decoding

上一章讲了 SGLang scheduler：它在每个 engine step 里平衡 prefill、decode、RadixAttention 命中、KV cache、token budget、structured output 开销和公平性。

本章进入 SGLang 另一个非常重要的能力：Structured Generation，也就是结构化生成。

很多生产应用不希望模型自由写一段自然语言，而是希望模型输出严格满足格式：JSON、正则表达式、EBNF grammar、固定 choices、工具调用标签、函数参数 schema 等。

如果只靠 prompt 里写“请输出 JSON”，模型仍然可能输出解释文字、多余 markdown、缺字段、错引号、尾逗号、类型不对。生产系统通常不得不 parse、修复、重试，延迟和失败率都会上升。

Constrained decoding 的思路是：不要等生成完再检查格式，而是在每一步选 token 之前，根据 grammar state 计算哪些 token 合法，把非法 token 的 logits mask 掉，让模型只能在合法 token 集合中采样。

一句话概括：

> Structured generation 把输出格式约束变成 runtime decoding 的一部分；constrained decoding 在每个 decode step 根据 JSON schema、regex、EBNF 或 structural tag 维护 grammar state，并用 token mask 限制 sampler 只能选择合法 token。

## 30.1 本章目标

读完本章，你应该能讲清：

1. 为什么“prompt 里要求输出 JSON”不等于可靠结构化输出。
2. Structured generation 和后处理 parse/retry 的区别。
3. Constrained decoding 在 logits 到 token 的哪个阶段生效。
4. JSON schema、regex、EBNF、structural tag 分别适合什么场景。
5. Grammar state、valid token mask、sampler 的关系。
6. Structured output 为什么会影响 scheduler 和性能。
7. 面试中如何解释 SGLang 的 structured outputs。

## 30.2 为什么结构化输出重要

LLM 在应用里经常不是最终用户直接阅读的聊天机器人，而是系统链路中的一个组件。

它的输出可能要被程序继续消费。

典型场景：

1. 信息抽取：输出 JSON。
2. 工具调用：输出函数名和参数。
3. 分类：输出固定 label。
4. 路由：输出某个 agent 或 workflow 名称。
5. SQL 生成：输出语法受限的查询片段。
6. 配置生成：输出 YAML 或 JSON。
7. 表单填写：输出字段值。
8. 评测打分：输出固定 schema。

如果输出格式不稳定，后续系统会出问题。

例如你希望模型输出：

```json
{
  "city": "Paris",
  "country": "France",
  "population": 2148000
}
```

但模型可能输出：

```text
Sure, here is the JSON: {"city": "Paris", "country": "France", "population": "about 2.1 million"}
```

这个结果对人类可读，但对程序不可靠。

问题包括：

1. 多了自然语言前缀。
2. 字段类型不符合 schema。
3. 可能缺字段。
4. 可能多字段。
5. 可能 JSON 语法不合法。
6. 可能把枚举值写成近义词。

结构化生成就是为了解决这类问题。

## 30.3 后处理和重试的问题

最朴素的方案是生成后 parse：

```text
model output -> parse -> validate -> fail -> retry
```

这在小规模 demo 中常见，但生产问题很多。

第一，重试增加延迟。

一次失败可能让请求多跑一遍，TTFT 和 E2E latency 都变差。

第二，重试增加成本。

每次 retry 都要重新执行模型生成，除非 runtime 能很好地复用 cache。

第三，失败不是确定可修复。

模型可能连续输出不合法内容，尤其是 schema 复杂、字段多、上下文长时。

第四，后处理修复可能改错语义。

例如模型输出了字符串类型的 `population`，后处理强行转整数，可能把不确定表达错误地变成确定值。

第五，错误发生太晚。

生成完整答案后才发现不合法，已经浪费了全部 decode 成本。

Constrained decoding 的目标就是把格式约束提前到每一步 token 选择阶段。

## 30.4 Constrained decoding 的核心流程

普通 sampling 流程是：

```text
logits
  -> penalty
  -> temperature
  -> top-k / top-p
  -> sample
  -> token
```

Constrained decoding 会插入 grammar mask：

```text
logits
  -> penalty
  -> grammar mask
  -> temperature
  -> top-k / top-p
  -> sample
  -> token
  -> update grammar state
```

Grammar mask 的作用是：

```text
合法 token: 保留 logits
非法 token: logits = -inf
```

这样 softmax 后，非法 token 的概率就是 0。

伪代码：

```python
def constrained_sample(logits, grammar_state, sampling_params):
    valid_token_ids = grammar_state.valid_next_tokens()
    mask = build_vocab_mask(valid_token_ids, vocab_size=logits.shape[-1])
    logits = logits.masked_fill(~mask, float("-inf"))
    token = sample_with_temperature_topk_topp(logits, sampling_params)
    grammar_state.accept(token)
    return token
```

这就是 structured output 属于 runtime 的原因：它直接参与 token selection。

## 30.5 Grammar state 是什么

Grammar state 表示当前已经生成的 token prefix 在约束规则中的位置。

例如约束输出 JSON object：

```json
{"name":"Paris","population":2148000}
```

生成过程中的状态可能是：

```text
step 0: 还没开始，只允许 "{" 或空白
step 1: 已生成 "{"，接下来允许字段名字符串
step 2: 已生成字段名，接下来允许 ":"
step 3: 已生成冒号，接下来允许字段值
step 4: 已生成字段值，接下来允许 "," 或 "}"
```

Grammar state 每接收一个 token，就向前推进。

它要回答的问题是：

```text
在当前状态下，下一个 token 哪些是合法的？
```

这个问题看起来简单，但实际很复杂，因为 tokenizer 的 token 不一定和字符边界对齐。

例如一个 token 可能是：

```text
"Paris"
"\"name"
"ation"
"}"
```

Grammar backend 必须在 tokenizer token 和语法状态之间建立映射。

## 30.6 Token-level 约束比字符级约束更难

很多 grammar 本来定义在字符层面。

例如 regex：

```text
(Paris|London)
```

字符级别看，只允许输出 `Paris` 或 `London`。

但模型采样的是 token id，不是字符。

Tokenizer 可能把 `Paris` 切成一个 token，也可能切成多个 token；不同模型 tokenizer 不一样。

所以 constrained decoding 需要做的是：

```text
字符级 grammar
  -> 和 tokenizer vocabulary 对齐
  -> 每个 grammar state 下计算合法 token ids
```

这就是 grammar backend 的价值。

它不只是 parse schema，而是要高效回答每一步合法 token 集合。

## 30.7 SGLang 支持哪些结构化约束

SGLang structured outputs 常见约束包括：

1. JSON schema。
2. Regular expression。
3. EBNF grammar。
4. Structural tag。
5. Choices 或固定候选。
6. Tool call schema。

SGLang 文档中提到的 grammar backend 包括：

1. XGrammar。
2. Outlines。
3. llguidance。

不同 backend 支持的约束类型和性能特点可能不同。

学习时先抓住抽象：

```text
constraint spec -> grammar backend -> grammar state -> valid token mask
```

## 30.8 JSON schema

JSON schema 适合定义结构化对象。

例如：

```json
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "population": {"type": "integer"}
  },
  "required": ["name", "population"]
}
```

它适合：

1. 信息抽取。
2. 表单填充。
3. 函数参数。
4. 评测结果。
5. Agent 中间状态。

JSON schema 的优点：

1. 和工程系统兼容好。
2. 可以表达字段类型。
3. 可以表达 required 字段。
4. 可以表达 enum、array、object。
5. 可以用 Pydantic 等工具生成。

局限：

1. 复杂 schema 可能增加 decoding 开销。
2. 只能保证语法和部分类型约束，不保证事实正确。
3. 约束越死，模型表达空间越小。
4. 对长文本字段，schema 只能约束外形，很难约束语义质量。

## 30.9 Regex

Regex 适合比较短、形式明确的输出。

例如：

```text
(positive|negative|neutral)
```

或者：

```text
[A-D]
```

适合场景：

1. 分类标签。
2. 多选题答案。
3. 简单 ID 格式。
4. 固定前缀后缀。
5. 小范围枚举。

Regex 的优点是简单直接。

缺点是：

1. 表达嵌套结构不方便。
2. 可读性容易变差。
3. 复杂 regex 可能难以维护。
4. 不适合复杂 JSON object。

如果只是要模型输出 A/B/C/D，regex 或 choices 比 JSON schema 更轻。

## 30.10 EBNF grammar

EBNF 适合自定义语法。

例如：

```text
root ::= city | description
city ::= "London" | "Paris" | "Berlin"
description ::= city " is capital of " country
country ::= "England" | "France" | "Germany"
```

EBNF 适合：

1. DSL 生成。
2. 简化 SQL 片段。
3. 命令语言。
4. 固定格式推理步骤。
5. 比 regex 更结构化、比 JSON schema 更自由的语法。

优点：

1. 表达力强。
2. 能定义递归或层级结构。
3. 适合领域特定语言。

缺点：

1. 编写门槛高。
2. 语法错误调试成本高。
3. 复杂 grammar 可能影响性能。
4. 不同 backend 支持的 grammar 格式可能有差异。

## 30.11 Structural tag

Structural tag 常用于工具调用或函数调用格式。

它的直觉是：当模型触发某个 begin tag 时，中间内容必须满足某个 schema，然后以 end tag 结束。

例如：

```text
<function=get_weather>{"city":"Paris","unit":"celsius"}</function>
```

这种方式适合：

1. Tool calling。
2. 多工具选择。
3. XML-like 标记输出。
4. Agent 动作格式。
5. 需要触发式结构化片段的场景。

它比单纯 JSON schema 更适合混合输出：模型可能先自然语言解释，然后在特定触发标签后输出结构化工具调用。

但工程上要明确：

1. 哪些 tag 可以触发。
2. 每个 tag 对应什么 schema。
3. 是否允许多个 tool call。
4. 触发后是否必须停止。
5. 标签之间是否允许普通文本。

## 30.12 Choices

Choices 是最简单的 constrained generation。

例如：

```text
choices = ["yes", "no"]
```

或者：

```text
choices = ["A", "B", "C", "D"]
```

它本质上也是约束输出只能落在候选集合内。

Choices 适合：

1. 分类。
2. 多选题。
3. 路由选择。
4. 小范围枚举。
5. 判断题。

如果任务可以表达成 choices，不要上来就用复杂 JSON schema。越简单的约束通常越稳定、越快、越容易调试。

## 30.13 Structured output 在请求里怎么传

SGLang 支持 OpenAI-compatible API 和 native API 两类常见方式。

OpenAI-compatible API 中，JSON schema 常通过 `response_format` 表达：

```python
response = client.chat.completions.create(
    model="model-name",
    messages=[{"role": "user", "content": "Extract city info."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "city_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "population": {"type": "integer"}
                },
                "required": ["name", "population"]
            }
        }
    },
)
```

Native `/generate` API 中，约束通常放在 `sampling_params`：

```python
payload = {
    "text": "Give city information in JSON.",
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 128,
        "json_schema": json_schema,
    },
}
```

Regex 和 EBNF 也类似：

```python
sampling_params = {
    "temperature": 0,
    "max_new_tokens": 32,
    "regex": "(Paris|London)",
}
```

```python
sampling_params = {
    "temperature": 0,
    "max_new_tokens": 32,
    "ebnf": 'root ::= "Hello" | "Hi"',
}
```

关键是：这些参数最终都会进入 runtime 的 grammar backend，而不是只作为 prompt 文本。

## 30.14 Runtime 内部执行链路

一个 structured output 请求进入 SGLang Runtime 后，可以这样理解：

```text
request
  -> parse sampling params
  -> detect json_schema / regex / ebnf / structural_tag
  -> compile or initialize grammar object
  -> create grammar state for request
  -> scheduler schedules prefill/decode
  -> model runner returns logits
  -> grammar backend computes valid token mask
  -> sampler masks logits and samples token
  -> grammar state accepts token
  -> output layer detokenizes and checks finish
```

这里有两类状态：

1. 模型生成状态：input ids、KV cache、output tokens。
2. 语法约束状态：当前 grammar position、合法 token 集合、是否完成。

这两类状态必须同步推进。

如果 token 生成了，但 grammar state 没更新，下一步 mask 就会错。

如果请求 abort 或 finish，grammar state 也要清理。

## 30.15 Grammar mask 和 sampler 的顺序

一个实际 sampler 可能包含很多处理：

1. repetition penalty。
2. frequency penalty。
3. presence penalty。
4. custom logit processor。
5. grammar mask。
6. temperature。
7. top-k。
8. top-p。
9. min-p。
10. random sampling 或 greedy。

常见直觉是：grammar mask 应该在最终采样前生效，保证非法 token 不会被采到。

例如：

```text
raw logits
  -> penalties
  -> custom logit processors
  -> grammar mask invalid tokens to -inf
  -> sampling filters
  -> sample next token
```

顺序细节不同框架可能不同，但必须满足一个原则：

```text
最终候选集合不能包含 grammar state 下非法的 token。
```

## 30.16 如果合法 token 集合为空怎么办

理论上，grammar backend 应该避免进入无合法 token 状态。

但工程上仍可能出现：

1. schema 不合法。
2. grammar 写错。
3. tokenizer 和 grammar 编译不兼容。
4. prompt 强烈诱导模型进入冲突状态。
5. backend bug。
6. custom logit processor 把合法 token 又全部屏蔽了。

如果合法集合为空，runtime 不能继续正常采样。

常见处理方式：

1. 返回错误。
2. 终止请求。
3. fallback 到 unconstrained decoding。
4. 放宽约束。
5. 记录指标和日志。

生产系统中一般不建议静默 fallback，因为用户以为拿到的是严格结构化输出，实际上不是。

## 30.17 Constrained decoding 不保证事实正确

Constrained decoding 只能保证形式合法，不保证内容正确。

例如 JSON schema 要求：

```json
{"city": "string", "population": "integer"}
```

模型输出：

```json
{"city": "Paris", "population": 999999999}
```

这在语法和类型上可能合法，但事实错误。

所以 structured output 解决的是：

```text
格式可解析、字段可验证、工具调用可执行
```

它不自动解决：

```text
事实正确、推理正确、业务规则正确
```

业务侧仍然需要 validation、retrieval grounding、规则校验或人工审核。

## 30.18 Prompt 仍然重要

即使用了 constrained decoding，也应该在 prompt 中明确说明输出格式。

原因是 grammar mask 只限制合法 token，不负责告诉模型“应该填什么内容”。

如果 prompt 不清楚，模型可能生成形式合法但语义差的结果。

例如 schema 允许：

```json
{"name": "string", "population": "integer"}
```

但 prompt 没说要抽取哪个城市，模型可能随便填。

正确做法是：

1. Prompt 说明任务语义。
2. Schema 约束输出形式。
3. 后置 validator 检查业务规则。

三者配合，才是生产可用的结构化输出。

## 30.19 Structured output 对性能的影响

Structured output 不是免费功能。

它可能带来这些开销：

1. 请求开始时编译 schema 或 grammar。
2. 每个请求维护 grammar state。
3. 每个 decode step 计算 valid token mask。
4. mask logits 需要额外操作。
5. 复杂约束可能减少可选 token，影响模型自然生成路径。
6. batch 中不同请求 grammar 不同，增加调度和采样复杂度。

性能表现可能是：

1. TTFT 轻微上升，因为要初始化 grammar。
2. TPOT 上升，因为每步 decode 多了 mask 计算。
3. CPU 利用率上升，如果 grammar backend 主要在 CPU 侧工作。
4. p99 上升，如果复杂 schema 集中出现。

因此线上要监控 structured output 请求占比、grammar compile latency、mask latency 和失败率。

## 30.20 对 scheduler 的影响

第 29 章讲过，structured output 会影响 scheduler。

原因是每个 structured request 可能有额外 per-step 成本。

Scheduler 如果完全把它当普通 decode 请求，可能低估 decode step latency。

例如：

```text
batch A: 64 个普通 decode 请求
batch B: 64 个复杂 JSON schema decode 请求
```

二者 decode token 数一样，但 batch B 的采样和 mask 可能更贵。

Scheduler 可以考虑：

1. 限制单 batch 中复杂 grammar 请求数量。
2. 统计 grammar mask latency。
3. 对超复杂 schema 做 admission control。
4. 把 structured workload 和普通 chat workload 分队列。
5. 在成本模型中加入 grammar complexity。

是否需要这些策略取决于业务规模和 p99 要求。

## 30.21 Streaming 下的结构化输出

Structured output 和 streaming 可以一起使用，但要注意用户看到的是逐步形成的结构。

例如 JSON streaming：

```text
{
{"name"
{"name":"Paris"
{"name":"Paris","population"
...
```

中间片段可能不是完整 JSON。

这对前端和调用方有影响：

1. 不能每个 chunk 都当完整 JSON parse。
2. 需要等 finish 后再做完整 validation。
3. 如果要增量解析，需要 streaming parser。
4. stop condition 要和 grammar completion 协同。

因此结构化 streaming 的最佳实践通常是：

```text
流式展示可以逐 chunk 显示
程序消费最好等完整对象结束后再 parse
```

除非你专门设计了增量协议。

## 30.22 Tool call 和 agent 的关系

Agent 场景中，structured output 很关键。

模型需要输出：

1. 调用哪个工具。
2. 工具参数是什么。
3. 是否继续思考。
4. 是否返回最终答案。

如果工具调用格式不稳定，agent runtime 就无法可靠执行。

Structural tag 或 JSON schema 可以让工具调用更可控：

```text
<function=get_weather>{"city":"Paris","unit":"celsius"}</function>
```

这类输出被 constrained decoding 约束后，parser 更容易稳定识别工具名和参数。

但仍要校验：

1. 工具是否存在。
2. 参数是否在允许范围内。
3. 权限是否允许调用。
4. 参数值是否安全。
5. 是否需要用户确认。

Constrained decoding 保证格式，不替代工具安全策略。

## 30.23 和 RadixAttention 的关系

Structured output 和 RadixAttention 是不同层面的优化。

RadixAttention 解决：

```text
如何复用共享 prefix 的 KV cache，减少重复 prefill？
```

Structured output 解决：

```text
如何让 decode 输出满足格式约束？
```

它们可以同时工作。

例如一个固定 schema 的信息抽取任务：

```text
shared prompt: instruction + schema explanation + document template
request-specific: document content
output: JSON schema constrained
```

RadixAttention 可以复用固定 instruction 和 schema 说明的 prefix。

Constrained decoding 可以保证输出 JSON 格式。

Scheduler 则要同时考虑 cache hit 和 grammar mask 成本。

## 30.24 常见错误

错误一：只在 prompt 里说“输出 JSON”，没有 constrained decoding。

结果是 demo 可能成功，线上各种边界情况失败。

错误二：schema 过度复杂。

过度嵌套、字段太多、枚举太长，会增加生成难度和 runtime 开销。

错误三：以为 constrained decoding 保证事实正确。

它只保证格式，不保证内容真实。

错误四：streaming 时每个 chunk 都 parse JSON。

中间 chunk 往往不是完整对象。

错误五：custom logit processor 和 grammar mask 冲突。

如果自定义处理器把所有合法 token 屏蔽掉，请求会失败。

错误六：忽略 tokenizer 差异。

Grammar backend 必须和当前模型 tokenizer 对齐，否则 valid token mask 可能不正确。

## 30.25 如何选择约束方式

可以按复杂度从低到高选择：

1. 固定分类：choices。
2. 简单格式：regex。
3. 标准对象：JSON schema。
4. 自定义语法：EBNF。
5. 工具调用混合文本：structural tag。

选择原则：

```text
能用 choices 就不用 regex
能用 regex 就不用复杂 grammar
标准结构优先用 JSON schema
自定义语言再考虑 EBNF
工具调用考虑 structural tag 或 JSON schema
```

越简单的约束越容易调试，性能也通常更好。

## 30.26 调试 structured output

排查结构化输出问题，可以按下面顺序：

1. 确认请求是否真的传入 `json_schema`、`regex`、`ebnf` 或 `structural_tag`。
2. 确认 grammar backend 是否支持该约束类型。
3. 确认 schema 或 grammar 本身合法。
4. 确认 tokenizer 与模型一致。
5. 降低 temperature，先用 deterministic decoding 测试。
6. 简化 schema，逐步增加字段。
7. 查看 grammar compile latency 和 mask latency。
8. 检查是否 custom logit processor 冲突。
9. 检查 finish reason 和 max_new_tokens 是否过小。
10. 最后再考虑模型能力问题。

常见现象：

1. 一直不结束：可能 schema 允许很长输出，或 max_new_tokens 太大。
2. 过早截断：可能 max_new_tokens 太小。
3. 内容合法但语义差：prompt 不清楚或模型能力不足。
4. 延迟上升：grammar mask 或 compile 开销高。
5. 请求失败：schema/grammar/tokenizer/backend 不兼容。

## 30.27 面试官会怎么问

问题一：Constrained decoding 是什么？

回答要点：它是在每一步 decode 时根据 grammar state 计算合法 token 集合，把非法 token logits mask 成负无穷，让 sampler 只能选择满足 JSON schema、regex、EBNF 等约束的 token。

问题二：为什么不直接生成后 parse JSON？

回答要点：后处理失败发生太晚，会增加重试、延迟和成本；constrained decoding 把格式约束前移到 token selection 阶段，减少非法输出。

问题三：Structured output 能保证答案正确吗？

回答要点：不能。它主要保证格式和部分类型约束，不能保证事实正确、业务规则正确或工具调用安全。

问题四：JSON schema、regex、EBNF 怎么选？

回答要点：分类和枚举用 choices 或 regex；标准对象用 JSON schema；自定义 DSL 或复杂语法用 EBNF；工具调用可用 structural tag 或 JSON schema。

问题五：Structured output 为什么会影响 scheduler？

回答要点：因为每个请求需要 grammar state，每步 decode 要计算 valid token mask，复杂 schema 会增加 per-step 开销，影响 batch latency 和 p99。

## 30.28 标准回答模板

如果面试官问“SGLang structured generation 怎么实现”，可以这样回答：

```text
SGLang 的 structured generation 不是简单靠 prompt 要求模型输出 JSON，而是把结构化约束放进 runtime 的 constrained decoding 过程。用户可以通过 JSON schema、regex、EBNF 或 structural tag 指定输出格式，runtime 会把这些约束交给 grammar backend，比如 XGrammar、Outlines 或 llguidance，生成每个请求的 grammar state。

在每一步 decode 时，model runner 先输出 logits。Sampler 在采样前会查询当前 grammar state 下哪些 token 合法，然后构造 valid token mask，把非法 token 的 logits 置为负无穷。经过 temperature、top-p、top-k 等采样处理后，只会从合法 token 中选择下一个 token。选出 token 后，grammar state 会接受这个 token 并推进到下一个状态。

这样做可以减少 JSON parse 失败、格式错误和重试，把输出合法性从生成后检查前移到生成过程中。但它只保证格式和部分类型约束，不保证事实正确。它还会带来 grammar compile、valid token mask、per-request grammar state 等开销，所以 scheduler 和性能监控也要把 structured output 请求单独看。
```

## 30.29 小练习

1. 写一个 JSON schema，要求输出 `name`、`age`、`is_student` 三个字段。
2. 用 regex 约束模型只能输出 `positive`、`negative`、`neutral`。
3. 写一个简单 EBNF，让模型只能输出 `Hello` 或 `Hi`。
4. 画出 constrained decoding 从 logits 到 token 的流程图。
5. 解释为什么 grammar state 要随每个请求单独维护。
6. 举例说明 constrained decoding 不能保证事实正确。
7. 设计一个 structured output dashboard，包含 compile latency、mask latency、失败率、schema 类型分布和 p99。

## 30.30 本章总结

Structured generation 是复杂 LLM 应用走向生产的关键能力。它解决的不是“模型会不会说人话”，而是“模型输出能不能被程序可靠消费”。

Constrained decoding 的核心是在每一步采样前，根据 JSON schema、regex、EBNF 或 structural tag 维护的 grammar state 计算合法 token mask，屏蔽非法 token，让输出天然满足格式约束。

它比生成后 parse/retry 更靠前、更稳定，但也不是万能：它不能保证事实正确，复杂约束会带来性能开销，streaming 下也要注意中间 chunk 不一定是完整对象。

下一章会进入 Speculative Decoding 在 SGLang 中的角色，讨论如何用草稿模型、并行验证或其他机制减少主模型 decode 步数，从而进一步优化 TPOT 和吞吐。
