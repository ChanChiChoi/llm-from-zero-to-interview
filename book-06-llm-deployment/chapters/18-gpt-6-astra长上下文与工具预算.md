# GPT-6 Astra：长上下文、推理档位与工具预算

> 资料来源：[GPT-6 Astra 模型页](https://developers.openai.com/api/docs/models/gpt-6-astra)、[GPT-6 family 指南](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra)、[Reasoning](https://developers.openai.com/api/docs/guides/reasoning)、[API 定价](https://developers.openai.com/api/docs/pricing)、[Fast mode](https://developers.openai.com/api/docs/guides/fast-mode)、[Ultrafast mode](https://developers.openai.com/api/docs/guides/ultrafast-mode)、[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)、[Conversation state](https://developers.openai.com/api/docs/guides/conversation-state)、[Compaction](https://developers.openai.com/api/docs/guides/compaction)、[Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search)、[Async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling)、[Mid-turn steering](https://developers.openai.com/api/docs/guides/steering)、[Misalignment monitoring](https://developers.openai.com/api/docs/guides/safety-checks/misalignment-monitoring) 和 [Rethinking skills and prompts](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)。GPT-5.3 Codex、GPT-6 Sol、GPT-6 Luna、GPT-6.1 Sol 的对照分别链接在对应小节。核验日期：2026-10-27。价格、可用模型、参数兼容性和安全能力会变化，接入前应重查官方页面。本章只陈述公开的接口和运行时契约，不据此推断模型参数、内部架构或训练过程。

## 1. 先分清总上下文、输入预算与输出预算

GPT-6 Astra 官方模型页明确列出 <code>1,050,000</code> context window 和 <code>128,000</code> max output tokens。当前模型页没有单独列出 maximum input tokens。因此，<code>922,000 = 1,050,000 - 128,000</code> 只是“假设本轮把最大输出额度全部预留”时的算术余量，不是官方另行承诺的 maximum input，也不是每个请求都能使用的固定输入上限。([模型页](https://developers.openai.com/api/docs/models/gpt-6-astra))

可以先用一个简化账本理解单次响应：

~~~math
I + O \leq C,\quad O \leq O_{\max}
~~~

其中 <code>C</code> 是 context window，<code>I</code> 是本次请求输入的 token，<code>O</code> 是本次响应产生的 token，<code>O_max</code> 是 API 的最大输出限制。实际的 <code>I</code> 不只有用户刚输入的文字，还可能包含系统指令、工具定义、历史消息、工具回执和多模态输入；<code>O</code> 则可能包含推理 token、可见文本和结构化输出。上下文里还需容纳运行时状态。多轮 Agent 不能把每一轮的 1.05M 相加，当作一个可无限累积的“任务窗口”。

这一区分用于做容量规划，而不是证明请求会被接受。更实用的做法是为每个请求预留输出和状态空间，再给输入设一个低于理论余量的上限：

~~~text
input allowance
  = context window
  - reserved reasoning and visible output
  - tool / protocol state
  - safety margin
~~~

例如，长代码任务若把 <code>max_output_tokens</code> 设为 128K，就不能再把 <code>922K</code> 当作无条件可用的输入额度；系统指令、工具 schema、工具返回值、推理输出和安全余量仍会占位置。对于尚未量过的任务，先从较低的输入目标开始，记录实际 <code>usage</code>、截断原因和成功率，再逐步调高。

## 2. 推理档位是可测的预算设置，不是质量等级

GPT-6 Astra 模型页列出 <code>reasoning.effort</code> 的 <code>low</code>、<code>medium</code>、<code>high</code>、<code>xhigh</code> 和 <code>max</code>。这些字符串是该模型的 API 选项，不能与其他厂商或其他模型的同名选项作数值等价比较。GPT-6 家族还提供 <code>reasoning.mode</code> 的 <code>standard</code> 与 <code>pro</code>：mode 选择执行模式，effort 控制该模式中的推理投入，两者彼此独立。官方把 <code>pro</code> 描述为适合需要更多模型工作、可容忍更高延迟和 token 用量的任务。([Astra 模型页](https://developers.openai.com/api/docs/models/gpt-6-astra)，[Reasoning 指南](https://developers.openai.com/api/docs/guides/reasoning))

推理 token 不会作为普通文字展示，但它们会占 context、计入输出 token 并产生输出费用。官方说明，复杂度不同的请求可能生成几百到数万 reasoning tokens；若 reasoning 加可见输出触及 <code>max_output_tokens</code> 或窗口边界，响应可能以 <code>incomplete</code> 结束，甚至还没产生可见答案。官方建议初次摸索 GPT-6 类模型时至少给 reasoning 和输出预留 25K tokens，再按本地观测调节；这是起步建议，不是模型的固定需求。([Reasoning 指南](https://developers.openai.com/api/docs/guides/reasoning))

因此，<code>max_output_tokens</code> 不是“答案最多写多少字”，而是 reasoning 与可见输出共同受限的硬上限。若任务要先思考、再调用工具、最后产出 JSON 或代码，输出预算必须覆盖整个响应过程。排查“模型没回答”时，应先检查 <code>status</code>、<code>incomplete_details.reason</code> 和 <code>usage.output_tokens_details.reasoning_tokens</code>，不能只看空的 <code>output_text</code> 就断定模型没有运行。

<code>effort</code> 是否值得升档，应由任务结果决定。至少同时看成功率、reasoning/output tokens、工具次数、TTFT、TPOT、超时率和单位成功成本。若失败是检索漏资料、工具返回错误、权限不够或验证器不准确，单纯调高 effort 可能只增加推理消耗，不会修复失败路径。

## 3. 模态与工具支持不等于执行权限

GPT-6 Astra 官方页列出文本输入/输出、图像输入，以及 web search、file search、image generation、code interpreter、hosted shell、apply patch、skills、computer use、MCP 和 tool search 等工具。它把图像列为输入模态、文本列为输出模态；这不等于模型原生输出图像 token。模型页也说明，工具类模型可能按调用另行计费。([模型页](https://developers.openai.com/api/docs/models/gpt-6-astra))

一次工具调用至少要分清四个事件：

~~~text
模型提出调用
  -> 应用或平台校验 schema 与参数
  -> 策略层授权
  -> executor 执行并回传结果
  -> 模型消费结果，或 verifier 检查产物
~~~

模型页上出现 <code>hosted_shell</code> 或 MCP，不能据此推导它能访问任意文件、网络或终端。对应用自定义工具，执行器负责身份、路径、网络、超时、幂等和审计；对托管工具，也要根据平台公开的工具权限与计费条款单独验收。一次“模型返回了 tool call”既不代表宿主已授权，也不代表外部副作用已成功提交。

## 4. 长上下文费用要按整次请求和缓存类别拆开

官方定价页按每百万 token 报价，并区分输入、缓存读取、缓存写入时长、输出和处理档位。对 Astra，输入超过 272K token 后，整次请求的 input/cache 费率按 2 倍、output 按 1.5 倍计费；不是只对超过 272K 的那部分加价。缓存写入分 30 分钟与 12 小时两档。下表是 2026-10-27 核对的美元价格，不包括区域处理/FedRAMP 的附加费、托管工具逐次费用、应用自建工具成本和账户合同折扣。([API 定价页](https://developers.openai.com/api/docs/pricing)，[模型页](https://developers.openai.com/api/docs/models/gpt-6-astra))

| 处理档位 | 输入范围 | 未缓存输入 | 缓存读取 | 缓存写入 30 分钟 | 缓存写入 12 小时 | 输出 |
|---|---:|---:|---:|---:|---:|---:|
| Standard | ≤272K | $10 | $1 | $12.50 | $25 | $50 |
| Standard | >272K | $20 | $2 | $25 | $50 | $75 |
| Batch / Flex | ≤272K | $5 | $0.50 | $6.25 | $12.50 | $25 |
| Batch / Flex | >272K | $10 | $1 | $12.50 | $25 | $37.50 |
| Fast | ≤272K | $20 | $2 | $25 | $50 | $100 |
| Fast | >272K | $40 | $4 | $50 | $100 | $150 |
| Ultrafast | ≤272K | $60 | $6 | $75 | $150 | $300 |
| Ultrafast | >272K | $120 | $12 | $150 | $300 | $450 |

以上各列单位都是美元/百万 token。Batch/Flex 是 Standard 的 50%；Fast 是适用费率的 2 倍；Ultrafast 是 6 倍。地区处理和 FedRAMP 端点还可能额外加价。Fast 与 Ultrafast 是服务档位，不是每个请求的延迟 SLA；官方说明 Astra Fast 没有 latency SLA。Ultrafast 有单独的速率限制，官方建议频繁工具调用的 Agent 使用持久 WebSocket 以降低网络往返开销；Astra Ultrafast 目前只支持 US data residency 和 global processing，不支持 EU data residency。价格高不等于每个任务都更快，应在自己的工具链和负载上比较端到端延迟。([Fast mode](https://developers.openai.com/api/docs/guides/fast-mode)，[Ultrafast mode](https://developers.openai.com/api/docs/guides/ultrafast-mode))

为避免把输入类别混算，教学估算把每轮输入拆成三段：普通未缓存输入 <code>U</code>、缓存读取 <code>H</code> 和新写入缓存 <code>W</code>；三者互不重叠，<code>U+H+W</code> 才是本轮输入。推理 token 已在输出用量内，不要再重复记一次。模型 token 费用可写成：

~~~math
C_{\mathrm{tokens}} = U r_{\mathrm{input}} + H r_{\mathrm{cached}} + W r_{\mathrm{write}} + O r_{\mathrm{output}}
~~~

完整 Agent 任务成本还要把多轮响应、托管工具逐次费用、自建 executor 基础设施、验证器、失败重试和人工接管纳入。并行只能降低可并行工作部分的墙钟时间，不会自动消除工具调用费或模型 token 消耗。

下面的零依赖示例演示 Astra 的阈值倍率、服务档位和缓存读写口径。它只估算模型 token 费用，不含工具调用、区域附加费和账户合同；生产账单应以 API 返回的 usage 与当前价格页对账。

~~~python
TIERS = {
    "standard": 1.0,
    "batch": 0.5,
    "flex": 0.5,
    "fast": 2.0,
    "ultrafast": 6.0,
}
WRITE_TTL_MULTIPLIER = {"30m": 1.25, "12h": 2.5}


def estimate_astra_token_cost(
    uncached_input_tokens,
    cached_input_tokens,
    cache_write_tokens,
    output_tokens,
    *,
    write_ttl="30m",
    tier="standard",
):
    counts = (
        uncached_input_tokens,
        cached_input_tokens,
        cache_write_tokens,
        output_tokens,
    )
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("token counts must be non-negative integers")
    if write_ttl not in WRITE_TTL_MULTIPLIER:
        raise ValueError("write_ttl must be '30m' or '12h'")
    if tier not in TIERS:
        raise ValueError(f"unknown tier: {tier}")

    total_input = (
        uncached_input_tokens + cached_input_tokens + cache_write_tokens
    )
    is_long = total_input > 272_000
    input_rate = 10.0 * TIERS[tier] * (2.0 if is_long else 1.0)
    cached_rate = input_rate * 0.1
    write_rate = input_rate * WRITE_TTL_MULTIPLIER[write_ttl]
    output_rate = 50.0 * TIERS[tier] * (1.5 if is_long else 1.0)

    return (
        uncached_input_tokens * input_rate
        + cached_input_tokens * cached_rate
        + cache_write_tokens * write_rate
        + output_tokens * output_rate
    ) / 1_000_000


print(round(
    estimate_astra_token_cost(
        140_000, 100_000, 60_000, 20_000,
        write_ttl="30m",
        tier="standard",
    ),
    4,
))
~~~

~~~text
6.0
~~~

本例输入共 300K，落入 long-context 费率：140K 普通输入按 <code>$20/M</code>、100K 缓存读取按 <code>$2/M</code>、60K 30 分钟写入按 <code>$25/M</code>、20K 输出按 <code>$75/M</code>，合计 <code>$6.00</code>。如果把缓存写入误算成普通输入，这个估值会少算；若把 reasoning tokens 加到输出 token 后又单独加一遍，也会多算。真实的工具费用和 executor 资源账本要在 token 估算之外追加。

## 5. “工具预算”要覆盖完整任务，不只是一轮 Responses

一次 API 请求的 <code>max_output_tokens</code> 和平台工具限制，只约束当前响应。长任务可能包含多轮 <code>responses.create</code>、异步工具、自动 steering continuation、重试、compaction 与独立 verifier；若只限制单次调用，每次都在上限以内，整项任务仍可能持续消耗资源。因此，应用应维护一个跨响应的 job budget，并明确哪些上限由 API 提供、哪些必须由自己的 runtime 强制执行。

| 预算项 | 需要记录的限制 | 典型执行方 |
|---|---|---|
| 每响应上下文 | 输入 token 目标、输出预留、context margin | 请求构造器与 API |
| 每响应生成 | <code>max_output_tokens</code>、effort、mode | 请求构造器与 API |
| 整项任务模型费 | 所有 response 的输入/输出/缓存费用总额 | job coordinator |
| 工具资源 | 每类工具调用数、并发数、单次 timeout、外部 API 费用 | tool executor |
| 重试与恢复 | 重试上限、允许重试的错误、幂等键、恢复截止时间 | job coordinator 与状态库 |
| 验证 | 测试/查询/独立评审器调用次数和资源 | verifier |
| 总时限与副作用 | deadline、人工审批点、可补偿操作和不可重复操作 | policy / approval 层 |

一个跨轮账本可以表示为：

~~~math
C_{\mathrm{job}} = \sum_i C_{\mathrm{model},i} + \sum_j C_{\mathrm{hosted\ tool},j}
        + C_{\mathrm{app\ executor}} + C_{\mathrm{verification}} + C_{\mathrm{retry}} + C_{\mathrm{human\ review}}
~~~

这也是为什么应报告单位成功任务成本，而不只报一次响应价格。失败重试、验证器和人工接管可能提高单项成功成本；若统计分母只包含已调用的请求而不含失败任务，成本对比会偏低。评测须先定义“成功”：例如产物通过测试、业务约束和权限审查，而不是模型说“已完成”。

下面是一份纯教学限制示例：最多 8 次工具调用、最多 2 个并发只读调用、单项任务 10 分钟、幂等只读失败最多重试一次、外部写操作需审批、模型与托管工具总额封顶 <code>$3</code>、最后最多执行 2 组独立验证。这些数字不是 OpenAI API 限额或推荐值；示例要说明的是预算应有所有者和拒绝路径。async 可让两个独立只读调用重叠等待，但并发器仍要数调用数、占用时长和费用。若用户中途 steer 触发第二个响应，第二个响应的 token/tool 限制独立生效，job coordinator 仍要把它计入同一个 8-call、10 分钟和 <code>$3</code> 总预算。

长上下文代码 Agent 还需要一个外部权威状态记录：model ID/snapshot、mode/effort、输入与输出用量、cache read/write TTL、每个工具的 <code>call_id</code> 和 task handle、schema 版本、权限决定、执行状态、重试次数、compaction/steering lineage、最终 artifact hash、验证结果与总账单。没有这条记录，就很难判断超支来自推理、重复工具调用、过量验证还是失败恢复。

## 6. 状态回放、动态 effort 与工具搜索

长任务的上下文不只是可读的聊天文本。Responses item 还可能包含 function call / function result、加密的 reasoning item、assistant <code>phase</code>、tool-search 加载结果以及 compaction item。无状态 replay 必须依照 API 协议保留后续生成需要的 output items；只把最终可见文本拼回去，可能丢失工具链和继续任务所需的状态。<code>previous_response_id</code> 可以引用服务端响应链，但引用状态不等于历史输入免费，也不免除应用维护业务任务状态。([Conversation state](https://developers.openai.com/api/docs/guides/conversation-state)，[Compaction](https://developers.openai.com/api/docs/guides/compaction))

GPT-6 支持在对话中追加 <code>configuration_update</code> 改变后续响应的 reasoning effort。要尽量保留缓存前缀，官方建议保持顶层 <code>reasoning.effort</code> 原值，再把新的 effort 作为追加 item；若直接改顶层参数，隐藏系统指令可能变化并打断前缀复用。这个 item 是会话输入协议，不是改权重或切换 checkpoint。官方还给出兼容性限制：配置更新适用于标准模式的单 Agent 会话；不能与自动 compaction/truncation 或 standalone <code>/responses/compact</code> 混用，更新 item 不能相邻堆叠。应把每次更新及其生效顺序写进 replay trace，而不是只记录最终的 effort 值。([Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)，[GPT-6 family guide](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra))

工具目录本身也会占上下文。工具很少且常用时，把 schema 一开始全部提供，少一次发现步骤；目录大而每个任务只用少数工具时，可 defer 部分 function schema，通过 tool search 按需加载。Hosted tool search 由平台从本次声明的 deferred tools 中找候选；client-executed tool search 则由应用决定如何检索。加载后工具会追加到上下文，较容易保留已有前缀，但新增的 discovery step 仍可能带来延迟和漏选风险。namespace 的描述会用于发现，已 defer function 的名称和描述仍可能对模型可见；搜索到工具不是执行授权，schema、来源、版本和权限仍要检查。([Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search)，[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching))

因此，动态 effort、tool search 和 compaction 是三个不同的控制点：effort 改变后续推理预算，tool search 改变可用工具 schema 的加载集合，compaction 改变继续任务的上下文表示。日志中要分别记录，否则看见缓存命中率或任务成功率变化时，无法判断是哪个控制点产生影响。

## 7. 异步工具、steering 与后台响应是不同状态机

三者容易被混称为“异步 Agent”，但它们管理的是不同对象：

| 能力 | 在处理什么等待/变化 | 调用方仍要负责什么 |
|---|---|---|
| Async tool calling | 应用执行的慢 function/custom tool；模型可继续处理独立部分 | 启动 job、注册 task handle、保存原 <code>call_id</code>、timeout/重试/幂等、回灌结果 |
| Mid-turn steering | 同一个 WebSocket 响应运行时追加用户约束 | 保留 steering id/response lineage；核对 continuation 与原响应状态 |
| Background mode | Responses 的生成任务在后台运行 | 轮询/接收完成事件、截止时间、取消和结果恢复 |

<code>async: true</code> 是应用执行工具的协议。应用在完整 tool call item 到达后启动任务，并将原始 <code>call_id</code> 与 <code>task_handle</code> 登记在 job registry；<code>task_handle</code> 应在整条对话中唯一，即使之前的任务已经完成。任务结果回来后，应用用原 <code>call_id</code> 回传，并延续最新的 <code>previous_response_id</code>。工具任务“已完成”和模型“已读到结果”是两个状态；后者必须等模型消费回执。托管内建工具不因此变为应用异步任务，官方也不建议把 async tool calling 与 multi-agent parallel tool calls 混用。([Async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling))

<code>response.steer.accepted</code> 表示输入排队，不表示模型已经按新要求行动。若不需要应用先给工具结果或审批，API 会自动创建 continuation；在此之前服务端完成当前 output item 和已经在跑的 hosted tool work。若 steering 中断原响应，它以 <code>incomplete_details.reason="steered"</code> 结束；若原响应先正常结束，则仍为 completed，但后面也可以出现 continuation。自动 continuation 继承请求设置，不过 token 和 tool-call 上限按每个 response 分别计算。不要再为同一 steering 手动额外发一个 <code>response.create</code>，也不要把每响应上限误当作 job 全局预算。([Mid-turn steering](https://developers.openai.com/api/docs/guides/steering))

steering 是追加控制输入，不是撤回操作。已经发出的文本无法从用户界面收回，已经执行的工具或外部写入也不会因新约束自动回滚。若 steering 改变了目标，应在下一步重新检查权限与产物状态，并对已执行副作用采用显式补偿或人工确认。

## 8. Skills、AGENTS.md 与安全监控的适用范围

OpenAI 的技能文章建议用简短描述引导模型判断何时加载 skill，并采用“最小路由文档 + 按需读取支持文档”的渐进披露结构。由此得到的工程建议是：根级 <code>AGENTS.md</code> 应优先保留适用于所有任务的规则，将细节放到按需读取的支持文件；过多无关内容会占上下文，重复或冲突规则也会增加选错路径的机会。技能和指令应作为可评估的上下文路由设计，而不是 GPT-6 Astra 的隐藏记忆机制。评测时固定模型、harness、工具目录和权限，比较工具选择、无关 token、成功率、停止原因、压缩次数与单位成功成本。([Rethinking skills and prompts](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra))

Misalignment monitoring 是额外的安全控制面，不能当作通用的工具授权或独立 verifier。当前文档的覆盖范围取决于 API 和对话状态：

| 请求路径 | 文档描述的监控行为 | 实际边界 |
|---|---|---|
| Responses + persisted reasoning、WebSocket 或 OpenAI compaction | 可识别对话续接并自动阻断后续执行 | 不撤销此前已执行的工具或副作用 |
| Responses，不使用上述状态机制 | 可监控，并通过配置 webhook 告警 | webhook 告警本身不会自动停止对话 |
| Chat Completions | 不在该监控系统覆盖范围内 | 其他安全检查仍适用 |

被拦截的请求可能返回 <code>403</code> 和 <code>misalignment_policy_violation</code>；流式请求也要处理已经开始输出后出现的错误。收到该错误应冻结自动重试，保存 request/response/tool ID 和已发生副作用，再进入审计或人工处置。监控可能误报或漏报；权限、沙箱、审批与产物验证仍要独立部署。([Misalignment monitoring](https://developers.openai.com/api/docs/guides/safety-checks/misalignment-monitoring))

## 9. Compaction 是状态协议，不是可读摘要

server-side compaction 在 Responses 请求中设置 <code>context_management</code> 与 <code>compact_threshold</code>；达到阈值后，服务端会发出 opaque compaction item 并压缩此前上下文。采用无状态 input-array replay 时，应将该 item 连同 response output 一起保留；为减少延迟，可以丢弃最近 compaction item 之前的旧 items。若使用 <code>previous_response_id</code>，则只传新消息，不应自己裁剪服务端链路。独立 <code>/responses/compact</code> 会返回供下一次 <code>/responses</code> 原样使用的 canonical context window，其中可能既有 compaction item 也有仍保留的旧 items；不要再对其手动 prune。([Compaction](https://developers.openai.com/api/docs/guides/compaction))

这类压缩结果是不透明的，文档没有公开其内部算法或可供人读的完整摘要。它不能替代应用自己的任务状态：当前目标、权限、已提交副作用、未完成 job、工具结果、artifact hash 和 verifier 结论仍应存到外部账本。验收压缩时不要只问“摘要是否通顺”，而要问压缩前后的任务能否继续：模型是否还知道下一步、是否会重复执行已提交动作、工具结果是否仍能关联、权限是否仍有效、最终产物能否验证。

长上下文、tool search 和 compaction 各自解决不同问题：

| 方案 | 主要解决的问题 | 代价或失败边界 |
|---|---|---|
| 保留较长原始上下文 | 让本轮仍可访问更多原始信息 | token、延迟和费用上升；不能保证模型正确检索或综合远距离内容 |
| Deferred tool search | 不在首轮装入全部不常用工具 schema | 增加发现步骤；候选目录或描述不准时会漏选 |
| Compaction | 把旧上下文压成较短的可续接状态 | 内容变为 opaque 协议状态；需要保留 canonical item 并验证业务状态 |
| 外部任务/产物账本 | 在模型上下文之外保存权威状态和副作用证据 | 需要应用负责一致性、权限、清理、恢复和审计 |

面试中应先根据瓶颈选择机制：prompt 太长但工具目录小，先分析检索与裁剪；工具 schema 占比大，评估 defer；多轮历史持续增长，评估 compaction；副作用和恢复需要确定性时，外部账本仍不可省。把几个能力都称作“扩大记忆”会掩盖它们不同的输入、输出和恢复契约。

## 10. 本章相关 GPT-6 模型与 GPT-5.3 Codex：只比较已公开合同

本节只对照旧稿已涉及的 model IDs，不是 GPT-6 家族的完整目录。本次核验的官方定价页也列出 GPT-6.1 Terra 等相邻模型；它们的接口和能力不能由本节 Astra/Sol/Luna 对照推断。新部署应按实际 model ID 查看当日模型目录、模型页与参数兼容文档。([API 定价页](https://developers.openai.com/api/docs/pricing))

GPT-5.3 Codex 可用作历史对照，但当前官方页面已标为 Deprecated。它列出 400K context window、128K max output，没有单独列 maximum input；<code>400K - 128K = 272K</code> 同样只是把全部输出额度预留后得到的算术余量。页面列出 Chat Completions 与 Responses 端点及 function calling，不能再写成 “Responses-only”；迁移或新部署应先确认模型是否仍可用以及目标功能的 endpoint 兼容性。([GPT-5.3 Codex 模型页](https://developers.openai.com/api/docs/models/gpt-5.3-codex))

GPT-6 Astra、GPT-6 Sol、GPT-6 Luna 与 GPT-6.1 Sol 的当前模型页都列出约 1.05M context window 和 128K max output，但模型页未分别列出 maximum input。模型之间不能只按窗口大小或产品定位推断质量、速度、架构或成本；reasoning options、tool calling endpoint、费率和区域处理各自要按精确 model ID 核验。([Astra](https://developers.openai.com/api/docs/models/gpt-6-astra)，[GPT-6 Sol](https://developers.openai.com/api/docs/models/gpt-6-sol)，[GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)，[GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol))

当前 GPT-6 family guide 说明：Astra 不支持 <code>none</code> effort，Astra 的 reasoning tool calling 需要 Responses；Sol/Luna 在 <code>reasoning_effort="none"</code> 时才支持 Chat Completions function calling。Sol/Luna 模型页列出的 endpoints 与功能还应逐个按当前 model ID 检查，不能由端点列表推断全部参数组合都兼容。<code>reasoning.mode</code> 的 <code>standard/pro</code> 与 <code>effort</code> 是两个不同维度。GPT-6.1 Sol 另有 <code>reasoning.context="all_turns"</code> 等状态选项；不要把它们无条件迁移到 Astra。([GPT-6 family guide](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra)，[Reasoning](https://developers.openai.com/api/docs/guides/reasoning)，[GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol))

<code>gpt-6-sol</code> 的当前官方页指向较新的 <code>gpt-6.1-sol</code>；精修部署文档时应把 <code>gpt-6-sol</code> 当作较早模型页下的合同快照，而非整个 Sol 产品线的最新推荐。GPT-6 Luna 在既有模型目录中被列为高吞吐/高频使用定位；这是产品路由线索，不是参数量或 kernel 结构证据。对 Sol、Luna 的第三方评测应绑定自己的模型 ID、effort、provider、harness、任务集、工具、verifier 和时间，不迁移到 Astra 或 GPT-6.1 Sol。([GPT-6 Sol](https://developers.openai.com/api/docs/models/gpt-6-sol)，[GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol)，[GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna))

旧稿中的 [GPT-6 Sol 合同审计 toy](../../research/model-update-2026-09/code/gpt6_sol_contract_audit.py) 保留为历史教学材料，展示如何把配置、预算、缓存、工具权限和副作用写成可失败的状态门禁；它不是 Astra 的示例。脚本按 2026-09-23 的 Sol 合同快照写入了 `922K` 输入上限和当时的价格常量，而当前官方模型页没有单列该 maximum input，费率也可能变化。现在只应把它用于理解协议审计的组织方式；若要复用代码，必须先按目标 model ID 的当前文档重写常量并复核状态约束。([当时的 GPT-6 Sol 研究记录](../../research/model-update-2026-09/gpt-6-sol-source-notes.md))

本章与开放权重部署不能混用一张账本。gpt-oss-120b/20b 的部署成本取决于参数驻留/分片、专家路由、MXFP4 权重布局、KV cache、通信、kernel 与 workspace；API 模型则还要记录缓存档位、服务 tier、托管工具费用和账户价格。两类系统都需固定 model/revision、tokenizer/template、任务集、硬件/runtime 和 verifier 后再比较延迟、吞吐、质量与单位成功成本。

## 11. 面试追问与自查练习

**问：1.05M context window 能直接输入 1.05M token 吗？**

不能这样回答。官方单独列出最大输出 128K，但没有在 Astra 模型页列出独立 maximum input。922K 只是预留满额输出后的理论减法结果；工具定义、系统指令、历史状态和推理输出还要共享窗口。应按当前接口与响应状态测可用输入空间。

**问：把 effort 调到 <code>max</code> 会不会一定提高质量？**

不会。它只调节该模型接受的推理预算，不是跨模型的质量等级。若任务瓶颈在数据、工具、权限或 verifier，升 effort 不会自动补足这些缺口，还可能加大 token、延迟和任务费用。

**问：为何请求设置了较大的 <code>max_output_tokens</code>，结果仍然没有答案？**

reasoning token 也会消耗输出额度。若 reasoning 已触及输出上限，响应可能以 <code>incomplete</code> 结束而没有可见文本；要结合 <code>status</code>、<code>incomplete_details</code> 和 usage 判断。下一次可减少输入、给可见输出留空间，或重新设计任务/验证路径。

**问：工具调用并行是否一定降低成本？**

不会。并行可能降低等待时间，但工具请求费、执行器资源和模型处理工具结果的 token 仍然消耗。并发还可能增加峰值资源与外部写入冲突；必须分别统计墙钟延迟、调用数、费用、结果 token 和成功率。

**问：缓存读写怎样进入预算？**

把缓存读取、30 分钟写入、12 小时写入和普通未缓存 token 分开；超过 272K 输入时按整次请求切到 long-context 费率。缓存命中减少 token 价格不等于缓存无成本，cache write 和上下文变化也可能影响复用。完整费用还须加入工具、重试、验证与人工接管。

**问：服务端 compaction 是不是生成了一段普通摘要？**

不应这样假设。compaction item 是 opaque 的协议状态；standalone endpoint 返回的 canonical window 可能还包含其他 item，必须按文档原样交给下一请求。业务目标、副作用和 artifact 仍由应用自己的持久状态负责。

**问：Misalignment monitoring 能代替权限检查吗？**

不能。它只在文档说明的请求类型和对话状态下监测；一些 Responses 请求只有 webhook 告警而不会自动停止，Chat Completions 不在该系统覆盖内。即使拦截成功，也不会撤销此前已经发生的副作用。

把一次真实任务建成可比较的实验时，至少固定 model ID/snapshot、mode、effort、context reserve、服务档位、工具 schema/权限、最大调用数、deadline、重试/幂等规则、compaction 策略、verifier 和成功定义。报告 p50/p95 延迟、reasoning 与可见输出 token、cache read/write、工具数、超时/恢复率、任务成功率、成本/成功任务及失败样本。没有这些字段，“长上下文更强”或“高 effort 更聪明”都无法指导工程选择。

## 小练习

1. 给定 Astra 的 <code>1,050,000</code> context 与 <code>128,000</code> max output，推导 <code>922,000</code> 为什么只是算术余量；把工具 schema、历史输入、reasoning 和输出预留画进一个请求账本。
2. 用本节 Python 示例计算 300K 输入、不同 cached/read/write 分布和 30m/12h 缓存写入；覆盖阈值恰好为 272K 与超过 272K、非法 tier、负 token、浮点 token 等边界。
3. 对同一个任务比较 <code>standard</code> 与 <code>pro</code>、<code>low</code>/<code>high</code>/<code>max</code> 的组合；固定工具、输入、超时和验证器，报告成功率、reasoning token、可见输出、TTFT/TPOT 与单位成功成本。
4. 设计一个整项 Agent job budget，分别给出模型总额、每类工具调用数、最大并发、单工具 timeout、重试条件、verifier 预算、deadline 和副作用审批门禁。
5. 为 hosted shell、MCP、apply patch 和 computer use 画出 model、API、宿主 executor、沙箱、策略、审批和审计日志的责任边界。
6. 比较无状态完整 output item replay、只保留最终文本、<code>previous_response_id</code> 三种做法；说明各自保留什么状态、缓存与恢复如何核验。
7. 为 deferred tool search 设计 schema 版本、namespace、工具移除和权限撤销的审计流程；对比 eager 与 deferred 的上下文、发现步骤和漏选风险。
8. 模拟 async function tool：并发启动两个只读 job，再发生 timeout、重复完成回传和响应 lineage 更新；分别记录 original <code>call_id</code>、unique task handle、执行器状态和模型是否已消费结果。
9. 画出 <code>response.create -> response.created -> response.steer -> response.steer.accepted -> continuation</code> 的状态图；另覆盖原响应先 <code>completed</code>、steer 中断原响应、需要 app tool result/approval 三条路径。
10. 扩展安全故障演练：<code>misalignment_policy_violation</code> 到达后停止自动 retry，审计已启动副作用，再决定人工检查或补偿；明确 webhook 告警不等于执行阻断。
11. 对比服务器 compaction、standalone <code>/responses/compact</code>、长原始上下文和应用外部任务账本；说明每项解决什么问题、损失/边界是什么、需如何验证状态连续。
12. 把一个过长的 skill/<code>AGENTS.md</code> 拆成最小路由入口和渐进披露支持文件，固定同一任务集比较无关上下文 token、工具选择、停止原因、压缩次数与成功率。

配套的零依赖状态机示例见 [gpt6_agent_protocol_demo.py](../../research/model-update-2026-09/code/gpt6_agent_protocol_demo.py)。它验证 tool job、steering、断线恢复与安全阻断的教学状态转换，不调用真实 API，也不证明服务端事件时序、监控准确率或生产吞吐。逐节来源、旧内容去向及证据范围见[研究记录](../../research/gpt6-astra-long-context-tool-budget-depth-review-2026-10-27.md)。
