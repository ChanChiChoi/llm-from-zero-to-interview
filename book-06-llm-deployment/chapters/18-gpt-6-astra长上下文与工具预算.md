# GPT-6 Astra：长上下文、推理档位与工具预算

> 资料来源：[OpenAI 官方 GPT-6 Astra 模型页](https://developers.openai.com/api/docs/models/gpt-6-astra.md)、[OpenAI API 定价页](https://developers.openai.com/api/docs/pricing.md)、[Using GPT-6 Astra](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra.md)、[Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md)、[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md)、[Conversation state](https://developers.openai.com/api/docs/guides/conversation-state.md)、[Compaction](https://developers.openai.com/api/docs/guides/compaction.md)、[Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search.md)、[Async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling.md)、[Mid-turn steering](https://developers.openai.com/api/docs/guides/steering.md)、[Misalignment monitoring](https://developers.openai.com/api/docs/guides/safety-checks/misalignment-monitoring.md) 和 [Rethinking skills and prompts](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)，核验日期 2026-09-23，2026-09-28 复验。9 月 23 日 latest-model guide 已从 Astra 专属指南漂移为 GPT-6 family 迁移指南；9 月 28 日其哈希未变，开发者博客 Markdown 正文复核亦未发现新方法。本章只讨论官方明确公开的接口、容量和运行时能力，不根据产品功能推断参数量、MoE 结构或训练算法。

## 先分清三个“长度”

官方模型页同时列出 1,050,000 context window、922,000 maximum input tokens 和 128,000 maximum output tokens。初学者很容易把它们加在一起，认为一次请求可以输入 922K 再输出 128K，还剩 0；实际上它们是不同层次的约束，具体请求还会受到 Responses item、工具结果、系统消息和接口实现限制。

可以把一次请求想成一个容量有限的工作台。context window 是工作台总容量，maximum input 是输入区域的上限，maximum output 是输出区域的上限。工作台还要放工具调用、文件引用、状态元数据和可能的推理 token；因此“页面上的最大值”不能直接换算成并发数。

## 推理档位是预算旋钮

GPT-6 Astra 官方页列出 `reasoning.effort` 支持 `low`、`medium`、`high`、`xhigh` 和 `max`。这些名称是该模型的预算接口，不应与其他厂商同名档位直接比较。

实际服务应把推理预算、工具预算、验证预算和恢复预算分开记录。任务越复杂，增加 reasoning token 可能提高成功率；但如果工具调用或环境反馈才是瓶颈，只升档位不一定有效。评测时必须同时记录成功率、推理 token、总输出 token、工具次数、TTFT、TPOT、p95 成本和超时率。

## 官方页确认的模态与工具边界

模型页确认输入支持文本和图像，输出模态为文本；同时列出 web search、file search、image generation、code interpreter、hosted shell、apply patch、skills、computer use、MCP 和 tool search 等工具。

“支持 image_generation”表示模型可以通过 Responses API 调用图像生成工具，不等同于该模型原生输出图像 token；“支持 hosted_shell”也不等同于模型获得宿主机任意权限。真正的文件、网络、终端和 MCP 权限必须由执行器、沙箱和策略层判定。

## 长上下文成本不是线性直觉

截至 2026-09-28，官方模型页与定价页列出的费率如下，金额均为每百万 token，字段次序为 input / cached input / cache writes / output：

| 计费模式 | Short context | Long context |
|---|---|---|
| Standard | `$10 / $1 / $12.50 / $50` | `$20 / $2 / $25 / $75` |
| Batch | `$5 / $0.50 / $6.25 / $25` | `$10 / $1 / $12.50 / $37.50` |
| Flex | `$5 / $0.50 / $6.25 / $25` | `$10 / $1 / $12.50 / $37.50` |
| Fast mode | `$20 / $2 / $25 / $100` | `$40 / $4 / $50 / $150` |

模型页说明，输入超过 272K token 时，整次请求的 input/cache 费率按 2 倍、output 按 1.5 倍计算，而不是只对超出部分加价；Standard long-context 行与该规则相符。Batch/Flex 为 Standard 的 50%，Fast mode 为适用费率的 2 倍。价格是 2026-09-28 的官方文档快照，正式系统应继续读取当前[定价页](https://developers.openai.com/api/docs/pricing.md)并记录日期。GPT-6 Astra Fast mode 没有 latency SLA，且 EU data residency 不支持该模式，因此价格倍数不等于速度保证。

一个预算函数可以写成：

```math
C=C_{\mathrm{input}}+C_{\mathrm{cache}}+C_{\mathrm{output}}+C_{\mathrm{tool}}.
```

它只是成本分解，不是完整计费公式。缓存命中、缓存写入和工具调用需要分项；Batch、Flex 与 Fast mode 的公开价格不同，超过阈值后则是整次请求改变费率。下方示例只演示 Standard 费率，不估算工具费、Batch/Flex/Fast 或账户级折扣。

## 零依赖预算计算示例

```python

def estimate_cost(input_tokens, output_tokens,
                  input_price=10.0, output_price=50.0,
                  cache_hit_tokens=0, cache_hit_price=1.0):
    """Prices are dollars per million tokens; teaching estimate only."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    if cache_hit_tokens < 0 or cache_hit_tokens > input_tokens:
        raise ValueError("cache hits must be within input tokens")
    uncached = input_tokens - cache_hit_tokens
    multiplier_in = 2.0 if input_tokens > 272_000 else 1.0
    multiplier_out = 1.5 if input_tokens > 272_000 else 1.0
    cost_in = (uncached / 1_000_000) * input_price * multiplier_in
    cost_cache = (cache_hit_tokens / 1_000_000) * cache_hit_price * multiplier_in
    cost_out = (output_tokens / 1_000_000) * output_price * multiplier_out
    return cost_in + cost_cache + cost_out


print(round(estimate_cost(300_000, 20_000, cache_hit_tokens=100_000), 4))
```

这个示例只演示页面所述的阈值和输入/输出分解，没有加入账户等级、Batch、Flex、Fast mode 或工具调用费用。真实系统还应把缓存命中率、工具 token、失败重试和人工审核成本加到单位成功成本中。

## 与 Agent Runtime 的关系

长上下文模型仍然需要外部状态管理。代码 Agent 的 workspace、diff、测试结果、权限和 artifact 不能全部依赖 prompt；上下文折叠也不能替代可追踪的文件和 trace。MCP、skills、apply patch 和 computer use 都是协议/工具层能力，模型只是提出调用，运行时负责授权、执行、回执、超时和回滚。

一个稳健的请求记录至少包括：模型 ID、reasoning effort、输入 token、输出 token、缓存读写、工具调用、工具结果、权限决策、上下文压缩事件、错误码和最终 artifact。缺少这些字段时，无法解释成本、延迟或失败来自模型、工具还是运行时。

## 状态回放、动态 effort 与工具搜索

长任务不只是把更多文本塞进窗口。OpenAI 的官方运行时文档把状态拆成几类 item：可见消息、工具调用与结果、加密 reasoning item、assistant `phase`、compaction item，以及工具搜索后加载的工具集合。无状态 replay 时要保留 Responses 返回的全部 output items；只复制最终文本，可能丢失后续生成所需的状态。`previous_response_id` 和 Conversations API 可以引用服务端状态，但不会使历史输入免费。

GPT-6 及后续模型还支持在对话中追加 `configuration_update`，调整后续响应的 reasoning effort。它是请求协议中的配置更新，不是重新加载模型或修改权重。公平评测应固定初始配置，或把每次升档事件记录为实验变量，并分别统计 reasoning token、visible output、TTFT、工具轮数、恢复率和单位成功成本。

官方 latest-model guide 目前按 GPT-6 family 给出迁移矩阵：Astra 不支持 `none` effort，Sol/Luna 支持；Astra 的 reasoning tool calling 需要 Responses，Sol/Luna 只有在 `reasoning_effort="none"` 时支持 Chat Completions function calling；三者在 EU data residency 下只使用 Standard processing。这个矩阵是接口/部署契约，必须进入 capability manifest，不能推导成三个模型的内部架构或训练差异。

当工具数量较多时，可用 deferred tool search 只把 namespace/MCP 的概览放进初始上下文，在模型需要时再加载具体 schema。加载工具到上下文末端有助于复用稳定前缀，但工具集合也因此成为会话状态的一部分。工具 schema 的版本、权限和来源必须审计；不能因为模型“搜索到工具”就自动授予执行权限。

## 异步工具调用与中途 steering

GPT-6 Astra 的模型指南把长任务的等待问题拆成两个不同协议。`async: true` 适用于应用执行的 function/custom tool：模型发出工具调用后可以继续处理独立问题，应用立即启动慢任务，等任务完成后用原始 `call_id` 在新的 Responses 请求中回传结果。模型继续生成不等于工具已经完成，应用需要维护任务句柄、状态、超时、重试、幂等和最新 `previous_response_id`；工具结果最终还要经过模型消费和 verifier。托管内建工具不因此变成异步应用任务，multi-agent parallel tool calls 也不能直接与 async tool 混用。

在 WebSocket Responses 模式下，GPT-6 Astra 还支持 `response.steer`。客户端在收到 `response.created` 后，使用原响应 ID 发送新的用户约束；服务端先确认输入已排队，再在当前 output item 或正在运行的 hosted tool 工作结束后创建 continuation。原响应可能以 `incomplete_details.reason = "steered"` 结束，后续响应继承原请求设置。这个机制只是追加新的控制输入：已经发送的文本、已经启动的工具和已经产生的外部副作用不会被撤销，因此需要把 steering 事件、响应 lineage 和补偿动作写入审计轨迹。

面试中要把三件事分开：async tool calling 解决“模型等待工具时能否继续工作”，steering 解决“用户能否在同一轮改变约束”，background mode 解决“响应生成任务是否在后台运行”。三者都会影响状态机和成本，但都不授予模型宿主权限，也不保证外部副作用可回滚。

## Skills、AGENTS.md 与安全监控

OpenAI 的模型指南和开发者博客把技能描述、`AGENTS.md` 和任务提示视为上下文路由层。技能名称与描述需要让模型知道何时加载；描述过长或互相冲突会降低选择质量，复杂技能应把根文档做成最小路由器，再按需读取支持文档和脚本。仓库规则也应按任务引用，而不是要求每次编辑都读取整套文档。长任务提示需要定义完成条件、可自主推进的范围和停止点，否则模型可能过早询问，也可能在小任务上执行过宽的验证。

这是一条可迁移的 Agent 工程知识，不是 GPT-6 Astra 的内部记忆机制。评测技能或 `AGENTS.md` 时，应固定模型、harness、工具目录和权限，测工具选择、无关上下文 token、任务成功率、压缩次数、停止原因和单位成功成本；不能把“更会遵循指令”写成参数或训练配方。

Misalignment monitoring 属于平台安全控制面。官方文档说明它异步检查高后果上下文中的推理与动作，可能发出告警或阻断后续执行；被阻断时应识别 `misalignment_policy_violation`，停止自动重试，保留 response/tool/request ID 并检查已经发生的副作用。监控可能误报或漏报，且不会撤销早已完成的工具动作，所以它不能替代权限、人工审批、沙箱和 verifier。

因此失败路径必须是显式状态，而不是异常后“再发一次请求”。任务注册表应拒绝重复 `task_handle`、错误 `call_id` 和 pending job 的提前消费；steering 应拒绝同一 response 的第二次 steering，WebSocket 断开后也不能把旧 steering 隐式带到新连接；安全阻断后应冻结自动 retry，保留已启动副作用并交给审计/补偿流程。配套 toy 只验证这些状态转换，不证明真实服务端的事件时序或安全检测准确率。

## Compaction 的 canonical context

Server-side compaction 在 `context_management` 中设置 `compact_threshold`，跨过阈值后服务端在流中发出加密 compaction item，并用压缩后的状态继续推理。独立 compact endpoint 返回的不只是一个摘要字符串，而是下一轮应使用的 canonical context window；手工再删掉它之前的 item，可能破坏恢复所需的状态。对 `previous_response_id` 链式调用也不应同时手工 prune。

面试时应把 compaction 视为状态协议和可恢复性问题：保留目标、工具回执、权限、未完成副作用和 artifact hash，验证压缩前后的任务状态等价性，并对错误摘要、重复执行、工具集合变化和缓存断点失效做回归。官方文档没有公开压缩算法或摘要内容，因此不能把它描述成内部 memory architecture。

## 局限与面试追问

**问：1,050,000 context window 能否直接支持同等长度输入？** 不能。官方还分别列出 maximum input 和 maximum output；工具、系统消息和运行时状态也占用容量。

**问：`max` 是否一定比 `low` 更好？** 不一定。它通常提供更高推理预算，但会增加 token、延迟和成本，且任务瓶颈可能在工具或环境。

**问：模型支持 hosted shell 是否意味着可以执行任意命令？** 不意味着。权限、沙箱、网络和文件范围由执行器与策略层控制。

**问：如何比较两个模型的长上下文能力？** 固定输入长度、任务分布、工具集、压缩策略和输出预算，同时测有效检索、成功率、TTFT、TPOT、显存或 API 成本以及失败类型。

## 小练习

1. 将预算示例扩展为缓存命中、工具调用和失败重试成本。
2. 设计一个 300K、500K 和 1M 输入的延迟/成本实验，记录 p50、p95 和成功率。
3. 为 Responses API 请求画出模型、工具、MCP、沙箱和审计日志的责任边界。
4. 设计 low/high/max 的同任务评测，固定工具与超时，比较单位成功成本。
5. 设计一条 Responses replay 回归：比较完整 output item 回放、只回放最终文本和 `previous_response_id` 三种方式，检查 reasoning/phase、工具调用和状态恢复差异。
6. 设计 deferred tool search 的 schema 版本与权限审计，验证工具集合改变后缓存前缀是否仍可复用。
7. 设计 async function tool 的任务注册表，模拟工具延迟、重复回传、超时和响应 lineage 变化，验证原始 `call_id`、幂等键和 verifier 的关系。
8. 用 WebSocket 事件序列模拟 `response.create -> response.created -> response.steer -> response.incomplete(steered) -> response.created`，检查已发送文本、已启动工具和外部副作用不会被假设为自动回滚。
9. 将一套冗长技能说明改成“最小路由器 + 渐进披露”结构，比较无关上下文 token、技能选择、压缩次数和任务完成率；单独记录 misalignment 告警，不把它当作质量分数。
10. 扩展协议 toy 的故障矩阵：重复 task handle、错误 call ID、pending consume、重复 steering、WebSocket 断线恢复和 `misalignment_policy_violation`；要求安全阻断禁止自动 retry，且已启动副作用必须进入审计/补偿账本。

配套的零依赖状态机示例见 [`gpt6_agent_protocol_demo.py`](../../research/model-update-2026-09/code/gpt6_agent_protocol_demo.py)。它验证 `call_id`/task handle 失败门禁、pending/重复回传、响应血缘、steering 断线恢复、misalignment 后禁止自动重试、既有副作用保留和渐进披露账本；不调用真实 API，也不代表 GPT-6 Astra 的真实延迟、事件时序或质量。

## GPT-5.3 Codex 对照：更小窗口，不同 Agent 契约

GPT-5.3 Codex 与本章的 GPT-6 Astra 不能只按模型代际比较。官方页面显示 Codex 为 400K context、272K maximum input、128K maximum output，并且只支持 Responses；GPT-6 Astra 的窗口和端点更宽。更重要的差异在于 Codex 的公开定位与 harness 约束：Codex Prompting Guide 明确强调长时自治、first-class compaction、代码库探索、`apply_patch`、固定工作目录和工具并行。

面试回答应把两者放在同一张预算表中，同时保留协议差异：

| 维度 | GPT-5.3 Codex | GPT-6 Astra 本章已核验内容 |
|---|---|---|
| context / input / output | 400K / 272K / 128K | 1.05M / 922K / 128K |
| endpoint | Responses-only | Responses、Chat Completions、Batch |
| effort | low/medium/high/xhigh | low/medium/high/xhigh/max |
| Agent 状态重点 | Codex harness、phase、workspace、patch、compaction | configuration update、phase、tool search、canonical compaction |
| 不能推断 | 参数、MoE/稠密结构、训练 recipe、kernel | 参数、架构、训练 recipe、system card |

这个对照不是质量排名。GPT-5.3 Codex 没有精确 DataCurve DeepSWE 行，不能把 GPT-6 Astra 或 GPT-5.4/5.5 的 Agent 结果迁移给它。评测时应固定模型 ID、snapshot、effort、工具、harness、任务集、verifier 和压缩策略，分别报告成功率、reasoning/output token、工具轮数、TTFT、TPOT、恢复率和单位成功成本。

## GPT-6 Sol 对照：服务预算与 mode/effort 分离

Artificial Analysis 已记录 `gpt-6-sol` canonical 条目；OpenAI 官方模型页确认它的 context window 为 `1,050,000`、maximum input 为 `922,000`、maximum output 为 `128,000`，支持 `none`、`low`、`medium`、`high`、`xhigh`、`max` effort，默认 `medium`。它与本章 GPT-6 Astra 的共同点是都应按长上下文预算、工具宿主和 reasoning token 记账；不能因为同属 GPT-6 就把两个模型的 provider 指标、DataCurve 行或内部机制互相迁移。

GPT-6 Sol 的部署 manifest 还要区分 `reasoning.mode` 与 `reasoning.effort`。OpenAI Reasoning 文档把 GPT-6 family 的 mode 定义为 standard/pro，把 effort 定义为模式内的推理投入；`configuration_update` 可以在同一标准单 Agent 会话中改变后续 effort。它是会话控制协议，不是 checkpoint 切换，也不改变已产生的 tool side effect。配置更新、compaction item、tool schema 和 executor receipt 都应进入 replay trace。

Sol 的价格合同包含一个 whole-request threshold：超过 `272K` input tokens 后，整次请求 input/cache 按 2x，output 按 1.5x；Batch/Flex 为 50%，Fast mode 为 2x。生产容量规划应把 input、cached input、cache write、reasoning/output、tool call、retry、compaction 和 verifier 失败分别计量。模型页列出的 hosted shell、computer use 或 MCP 只是入口能力，不等于目标 sandbox、网络、权限或硬件已经验收。

因此本章的部署结论保持一致：模型合同、推理预算、工具宿主、缓存/压缩、权限和 artifact verifier 是不同层。GPT-6 Sol 当前没有公开参数规模、内部架构、完整训练 recipe 或独立生产 benchmark；本节只把官方可观察 API/runtime 字段纳入服务账本。

### Sol 合同审计 toy：把预算和副作用变成门禁

配套的 [`gpt6_sol_contract_audit.py`](../../research/model-update-2026-09/code/gpt6_sol_contract_audit.py) 用零依赖状态机验证这些边界：`standard/pro` 与 effort 正交；`configuration_update` 只能在 standard single-agent 中调整后续 effort，不能和 automatic compaction/truncation 或 standalone compact 混用；超过 `272K` input 后按整次请求放大 input/cache 与 output 费率；工具必须经过 permission、executor、idempotency 和 artifact verifier。toy 还模拟 opaque compaction 后的 cache prefix miss，并输出 `local_protocol_toy`，不代表真实 API、模型质量或目标硬件性能。

## GPT-OSS 对照：active 参数、MXFP4 与 Harmony serving

gpt-oss-120b/20b 的部署账本与 GPT-6 Astra 的 API 预算账本不同：前者是开放权重 MoE，必须分开记录 total/active parameters、专家驻留或分片、top-4 dispatch、GQA KV cache、MXFP4 weight storage、通信和 workspace；不能用 5.13B 或 3.61B active 参数直接决定显存。

官方 gpt-oss 仓库的 Triton/PyTorch 路径、Harmony chat template 和 Cookbook 是可运行入口，但不是跨 GPU/后端的性能保证。部署验收至少要固定模型 revision、tokenizer/template、MXFP4 kernel、硬件、batch、专家并行、窗口、reasoning effort 和工具 harness，并分别测 prefill/decode、专家负载、量化误差、cache bytes、TTFT/TPOT、任务成功率和单位成功成本。

## GPT-6 Luna 对照：高吞吐定位不等于架构披露

Artificial Analysis 的 `gpt-6-luna` 与 OpenAI model page 是独立证据：Luna 的 focused/high-volume 定位、1.05M/922K/128K、2026-05-18 cutoff 和 `$0.10/$0.50` 合同不能覆盖 GPT-6 Sol 的价格、provider 指标或 Agent 行。部署 manifest 仍要固定 model ID、snapshot、effort、mode、tool schema、compaction、executor 和 verifier。

“高效率”只能指导 task-shape routing 和成本账本，不能反推参数、dense/MoE、kernel 或 FLOPs。DataCurve 当前没有 `mini_swe_agent_gpt_6_luna_*` 精确行，因此本章只纳入官方可观察合同；完整权重、目标硬件 profiling、生产 kernel 和线上 SLO 保持 `unverified`。详见 [`gpt-6-luna-source-notes.md`](../../research/model-update-2026-09/gpt-6-luna-source-notes.md)。

## GPT-6.1 Sol：近 Astra 能力的成本与状态账本

Artificial Analysis 的实时榜单新增 OpenAI `gpt-6-1-sol` 及多个 effort 变体；DataCurve 没有精确 `mini_swe_agent_gpt_6_1_sol_*` 行，因此不能迁移其他 GPT 的 Agent 分数。OpenAI 官方模型页将它定位为较低成本的复杂 coding、computer use 和 professional work 模型，公开合同为 1,050,000 context、922,000 maximum input、128,000 maximum output，输入支持 text/image，输出为 text。

它支持 `low/medium/high/xhigh/max` reasoning effort，默认 `medium`，但不支持 `none/minimal`。GPT-6.1 Sol 的 `reasoning.mode`（`standard/pro`）和 `reasoning.effort` 是两个独立字段；mode 选择执行档位，effort 控制该档位中的推理预算。

官方 Reasoning 文档允许 `reasoning.context=all_turns` 复用兼容的历史 opaque reasoning items，`current_turn` 则只让当前回合 reasoning 可用；两者都不暴露原始思维链。Compaction 文档要求把加密 compaction item 作为下一轮输入的一部分保留；stateless input-array chaining 与 `previous_response_id` chaining 的裁剪规则不同。模型合同、推理状态、compaction、工具执行和业务 verifier 应在 replay ledger 中分栏。

本节只写官方可观察 API/runtime 合同。GPT-6.1 Sol 的参数、架构、训练配方、完整权重、真实 endpoint 行为和生产硬件验收仍未公开核验。
