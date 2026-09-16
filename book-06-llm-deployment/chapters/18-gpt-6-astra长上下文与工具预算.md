# GPT-6 Astra：长上下文、推理档位与工具预算

> 资料来源：[OpenAI 官方 GPT-6 Astra 模型页](https://developers.openai.com/api/docs/models/gpt-6-astra.md)，核验日期 2026-09-09。本章只讨论模型页明确公开的接口、容量和工具能力，不根据产品功能推断参数量、MoE 结构或训练算法。

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

模型页本次标示每百万 token 输入 10 美元、缓存输入 1 美元、缓存写入 12.5 美元、输出 50 美元；输入超过 272K token 时，整次请求的输入和缓存费率按 2 倍、输出费率按 1.5 倍计算。价格是文档快照，正式系统应读取当前价格页并在实验记录中保存日期。

一个预算函数可以写成：

```math
C=C_{\mathrm{input}}+C_{\mathrm{cache}}+C_{\mathrm{output}}+C_{\mathrm{tool}}.
```

它只是成本分解，不是完整计费公式。缓存命中、缓存写入、工具调用、批处理和 fast mode 可能有不同规则；超过阈值后是整次请求改变费率，而不是只对超出的 token 加价。

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
