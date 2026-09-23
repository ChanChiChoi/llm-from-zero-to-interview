# 第 90 章 Kimi K2.6：Native Multimodal、Agent Swarm 与推理验收

> 本章核验日期：2026-09-20。模型入口来自 [Artificial Analysis 的 Kimi K2.6 条目](https://artificialanalysis.ai/models/kimi-k2-6)；DataCurve DeepSWE 当前没有精确的 `mini_swe_agent_kimi_k2_6_*` 行。模型规格、配置、部署和 Agent 技术来自 [Kimi K2.6 官方模型卡](https://huggingface.co/moonshotai/Kimi-K2.6)、[固定配置](https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/config.json)、[部署指南](https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/docs/deploy_guidance.md)、[官方博客](https://www.kimi.com/blog/kimi-k2-6) 和 [Kimi Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier.html)。榜单字段和官方 benchmark 均绑定各自条件，不是本地独立复现。

## 90.1 先把“模型、运行配置、Agent 系统”拆开

Kimi K2.6 最容易被误读的地方，是它同时公开了大模型配置、长周期 coding 案例、Agent Swarm 和部署验收工具。它们属于不同证据层：

| 层 | K2.6 可确认内容 | 不能直接推出的结论 |
|---|---|---|
| 榜单条目 | Artificial Analysis 有 K2.6 和 Non-reasoning 配置；AA 页面标记 deprecated 并指向 K3 | AA 指数是裸模型能力；deprecated 是官方退役公告 |
| 模型 artifact | 1T/32B、61 层、384 experts、top-8、MLA、MoonViT、256K、native INT4 | 完整训练 recipe、所有 kernel、线上 p99 |
| API/协议 | thinking/instant、`preserve_thinking`、`reasoning_content`、多步工具调用 | 暴露了完整隐藏 chain-of-thought；模型拥有工具权限 |
| Agent harness | 300 sub-agents、4,000 coordinated steps、长周期案例、Kimi Code CLI | 模型内部有 300 个专家；Agent 结果只由基础模型决定 |
| verifier | KVV 检查参数、视觉、长输出、tool call 和 SWE-Bench 部署 | KVV 分数就是模型 benchmark |

因此本章的主问题不是“它的 Intelligence Index 是多少”，而是：同一份开源权重如何通过架构账本、低精度部署、状态协议、Agent 编排和供应商验收形成可复现系统。

## 90.2 1T/32B MoE：total、active、resident 和 latency

官方模型卡摘要给出 1T total parameters、32B activated parameters、61 layers、1 个 dense layer、384 experts、每 token 选择 8 个 routed experts 和 1 个 shared expert。面试时至少要拆开四本账：

1. **容量账**：1T 描述所有专家和共享权重的总规模。
2. **计算账**：32B/Top-8 描述一个 token 的主要激活路径，但还要加 attention、router、shared expert 和实现开销。
3. **驻留账**：专家权重可能分片或按后端策略驻留；不能把 32B 当作“只需装载 32B 权重”。
4. **服务账**：专家 dispatch/combine、通信、KV/cache、媒体状态、工具结果、workspace 和并发都会影响 p99。

可以用一个审计式近似表示请求显存：

```text
resident weights + KV/cache + expert dispatch buffers
  + media/tool state + workspace + scheduler headroom
```

这比“总参数乘 bit 数”更接近真实部署问题。尤其在 MoE 和多模态 Agent 中，单 token 激活路径与整个服务进程的内存压力不是一件事。

## 90.3 MLA、YaRN 和 256K 上下文的 cache 账本

固定配置给出 `q_lora_rank=1536`、`kv_lora_rank=512`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64` 和 `v_head_dim=128`。这些字段能帮助我们理解 MLA 的低秩潜变量方向：服务系统可以尝试压缩历史表示，减少显式 KV 的带宽和存储压力。

但“有 MLA”不等于“长上下文免费”。实际系统还要核算：

```text
latent KV / positional component
  + top-k expert dispatch
  + multimodal tokens
  + tool results and reasoning state
  + prefix-cache metadata and workspace
```

配置的 YaRN `factor=64`、原始位置 `4096` 和最大位置 `262,144` 是公开配置字段。它们不能证明任意长度、任意媒体比例、任意工具轨迹都具有同样的精确召回率。面试回答应把 context window、有效检索、TTFT、TPOT、cache bytes 和任务成功率分开测。

## 90.4 MoonViT 与 native INT4：多模态和低精度是联合 serving 问题

README 将视觉编码器称为 MoonViT，约 400M 参数；配置公开了视觉塔、patch、spatial-temporal video attention 和统一视觉 chunk。它支持文本、图像和视频输入，但没有公开完整的视觉训练 recipe、对齐损失或生产视觉 kernel。

配置的 native INT4 路线是 `compressed-tensors` 的 pack-quantized 权重，group size 为 32，部分 attention、shared expert、lm head、vision tower 和 projector 被列为忽略项。于是部署验收必须同时检查：

- 权重反量化和计算 dtype 是否符合后端假设；
- 被忽略的模块是否造成显存峰值；
- 图像/video token 是否改变 prefill 峰值和 cache key；
- 量化后长输出、视觉输入、tool call JSON 是否出现质量回归；
- CPU/GPU 异构、TP/EP、batch 和并发改变后，吞吐数字是否仍然成立。

官方部署指南给出 vLLM、SGLang 和 KTransformers 路线，并要求使用 `kimi_k2` tool/reasoning parser。KTransformers 的示例性能绑定 8 张 L20、Intel 6454S、48-way concurrency 和具体配置，不能泛化为生产 SLO。

## 90.5 K2.5 架构复用：版本差异不能靠名称猜

Kimi K2.6 部署指南明确称它与 K2.5 使用相同架构路线，部署方法可以直接复用。这条证据支持“复用路线”的判断，但不支持以下更强结论：

- K2.6 与 K2.5 的训练数据、后训练、权重和所有层细节完全相同；
- K2.5 技术报告中的每个 benchmark 都是 K2.6 复现；
- Kimi K3 的 KDA、Gated MLA、Block AttnRes 或 Stable LatentMoE 已经存在于 K2.6。

一个合格的版本账本应把“模型类名/配置复用”“架构路线复用”“权重相同”“训练 recipe 相同”分成四个字段，而不是都写成“same architecture”。

## 90.6 Agent Swarm：sub-agent 数量属于系统，不属于 MoE

K2.6 官方博客描述 Agent Swarm：最多 300 个 sub-agents，4,000 个 coordinated steps；并展示了长周期的 Mac/Zig 优化和 `exchange-core` coding 案例。正确的系统图是：

```text
K2.6
  -> plan / decompose / tool-call proposal
  -> coordinator creates domain-specialized sub-agents
  -> isolated workspace + tools + sandbox execute
  -> tests / verifier inspect intermediate artifacts
  -> coordinator merges, retries, cancels or delivers
```

这里的 sub-agent 可以是相同模型、不同 prompt、不同工具权限或不同上下文的多个运行实例。它不是 300 个 MoE expert，也不是模型一次生成 4,000 个 token。系统设计必须补充：

| 控制面 | 面试应追问 |
|---|---|
| 任务图 | 哪些子任务可并行，哪些需要前置 artifact？ |
| 上下文 | 子 Agent 看到完整历史、摘要还是只读证据？ |
| 权限 | 谁可以读写 workspace、联网、执行命令或提交副作用？ |
| 预算 | 每个 worker 的 token、工具轮数、墙钟和 GPU slot 如何限制？ |
| 恢复 | 超时、取消和未知外部状态如何查询与重试？ |
| 合并 | 代码、文档、表格等 artifact 如何版本化并由谁验收？ |

官方案例中的 4,000+ tool calls、12 小时、14 iterations 和 1,000+ tool calls、4,000+ 行代码修改都是发布方案例数字。面试时必须继续追问任务版本、工具、权限、环境、失败重试和最终验收，而不能只复述数量。

## 90.7 长周期 coding 的 context 和 artifact 门禁

长周期 Agent 的瓶颈经常不是一次生成质量，而是状态能否跨数小时保留。K2.6 README 的评测脚注明确绑定了 context management：有的任务只保留最近一轮工具消息，有的使用 hide tool result，有的把超出上下文的任务直接计为失败。

这说明“256K context”不能直接等价为“能够可靠处理 256K 的完整轨迹”。一个长任务 manifest 至少应保存：

```text
model/revision
  + thinking mode and sampling parameters
  + tool schema/version and permissions
  + workspace revision and artifact owner
  + context-management policy
  + tool receipts, tests, retries and unknown states
  + final artifact digest and verifier result
```

最终完成条件也不能是模型输出“done”。应检查测试是否在目标工作区执行、工具结果是否真实回传、artifact 是否存在、关键状态是否满足、是否有 no-op 或 shortcut，以及 verifier 是否独立于模型。

## 90.8 Thinking、preserve_thinking 和多步工具调用

K2.6 的 thinking/instant 是服务运行配置。`preserve_thinking` 默认关闭；开启后，历史 assistant message 中的 `reasoning_content` 需要按协议回传。这个机制的工程意义是状态连续性，而不是让业务层随意修改或公开隐藏推理。

典型循环可以写成：

```text
user request
  -> reasoning state + tool call
  -> host parser and permission gate
  -> executor receipt
  -> tool result
  -> preserved state + next call/final answer
```

模型输出 tool call 只是提案；schema 校验、授权、沙箱、网络范围、超时、幂等和副作用验证都属于宿主系统。`reasoning_content` 还会占用上下文和 token，开启 preserve 模式时要把它加入 cache、成本和截断账本。

## 90.9 Kimi Vendor Verifier：六道“链信任”检查

KVV 的重要性在于它把“模型能力失败”和“供应商实现偏差”分开。六类公开检查分别覆盖：

1. API 参数 pre-flight；
2. OCRBench 多模态 smoke test；
3. MMMU-Pro vision preprocessing；
4. AIME2025 长输出压力；
5. K2VV ToolCall 的 F1 和 JSON Schema accuracy；
6. SWE-Bench 的完整 Agentic coding。

它们对应不同故障面：

```text
sampling constraint -> visual preprocessing -> cache/quantization
       -> tool parser/schema -> sandbox/agent harness
```

KVV 博客称完整流程在两台 H20 8-GPU 服务器上约需 15 小时，并加入流式推理、自动重试和 checkpoint resume。这个数字说明验收本身是工程系统，不说明 K2.6 的推理速度。一次合理的部署报告要按下表分层：

| 观察 | 可能归属 | 需要补的证据 |
|---|---|---|
| thinking 参数被后端改写 | API/serving | pre-flight 日志和请求回放 |
| 长输出突然退化 | KV/cache/量化/kernel | AIME 长输出和精度对照 |
| 图片答案异常 | vision preprocessing | OCRBench/MMMU-Pro 切片 |
| 工具 JSON 解析失败 | parser/schema/harness | ToolCall F1、schema validator |
| coding 任务失败 | 模型、工具、沙箱或 verifier | SWE-Bench 全链路 trace |

## 90.10 与 K2.7 Code、K3 的差异

| 对象 | 可确认主线 | 不应混写 |
|---|---|---|
| Kimi K2.6 | K2.5 架构路线、1T/32B、MLA、MoonViT、native INT4、Agent Swarm、KVV | K2.7 的 coding 专项改进、K3 的 KDA/AttnRes |
| Kimi K2.7 Code | 以 K2.6 为基础的 coding-focused agentic model，always-on/preserved thinking 等协议 | K2.7 的 benchmark 和协议不自动等于 K2.6 |
| Kimi K3 | KDA、Gated MLA、Block AttnRes、Stable LatentMoE、XTM 和技术报告 | K3 的新架构不能回写到 K2.6 |

版本归因的核心规则是：先锁定排行榜条目和官方 revision，再将技术字段绑定到来源；不能因为三个模型都叫 Kimi K2/K3 就把后续技术倒灌到前代。

## 90.11 面试追问

**问：1T total、32B active 是否意味着 K2.6 只需要 32B 参数显存？**

答：不意味着。active 描述每 token 的主要计算路径；驻留专家、shared expert、attention、router、通信 buffer、KV/cache、媒体和并发 workspace 仍会占用资源。需要固定 dtype、TP/EP、batch、上下文和并发后实测。

**问：300 个 sub-agents 是不是 300 个 MoE experts？**

答：不是。384 experts/top-8 是模型内部路由；300 sub-agents 是 Agent harness 在外部任务图中创建的运行实例，两者数量、权限、上下文和生命周期都不同。

**问：native INT4 如何证明部署正确？**

答：先检查权重格式、group scale、未量化模块和后端 kernel，再用 KVV 的 pre-flight、视觉、长输出和 tool-call 测试比较质量与协议，最后绑定硬件、并发、TTFT/TPOT 和任务成功率。不能只看模型能否启动。

**问：为什么 `preserve_thinking` 不是简单的字符串拼接？**

答：它关系到多轮请求中 reasoning state、tool call/result、采样参数和上下文成本的回放。宿主还要验证 schema、权限、执行回执和副作用，不能只把一段文本追加到 prompt。

**问：为什么 K2.6 没有 DataCurve 精确行仍可做资料闭环？**

答：资料闭环的含义是榜单发现、官方来源、技术拆解和证据边界完整；它不等于双榜评测闭环。没有精确行就明确缺失，不迁移 K2.7/K3 的系统结果。

## 90.12 当前 Kimi Code harness 补证：128 与 300 不能合并

Kimi K2.6 的博客案例和当前 Kimi Code CLI 文档提供了两组不同层次的 Agent 证据：

| 证据 | 可确认内容 | 面试时的边界 |
|---|---|---|
| K2.6 官方博客 | 最多 300 个 sub-agents、4,000 个 coordinated steps，以及长周期 coding 案例 | 发布方案例和产品能力描述，不是模型内部专家数，也不是 CLI 的固定上限 |
| Kimi Code CLI `AgentSwarm` | 最多 128 个 subagent、默认 2 小时超时、`resume_agent_ids`、模型池、聚合报告、并发爬坡和唯一工具调用约束 | 当前 harness 的工具协议，不能改写成 K2.6 专属 coordinator 或训练机制 |

[Kimi Code 模型配置](https://www.kimi.com/code/docs/kimi-code/models.html) 当前列出 `k3`、`k3-256k`、`kimi-for-coding`（K2.8 Preview）和 `kimi-for-coding-highspeed`（K2.7 Code HighSpeed）四个 ID。[Agent/subagent 文档](https://www.kimi.com/code/docs/kimi-code-cli/customization/agents.html) 说明 main agent 与 subagent 的上下文隔离、内置 `coder/explore/plan` profile、权限继承和自定义 Agent；[内置工具文档](https://www.kimi.com/code/docs/kimi-code-cli/reference/tools.html) 说明 `AgentSwarm` 的模板/恢复、128 上限、2 小时超时、模型池、`KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY` 和派发前权限复核。

这组文档还把“可恢复会话”具体化：[会话文档](https://www.kimi.com/code/docs/kimi-code-cli/guides/sessions.html)公开 `state.json`、`wire.jsonl`、`--continue`、`--session`、`/compact`、`/fork` 和导出。它说明长周期 Agent 的状态不仅是历史 prompt，还包括工具 schema、请求参数、MCP 清单、权限、workspace 和可回放事件。面试时应追问：compact 后哪些 artifact 仍可验证、resume 如何处理已发生但未知结果的副作用、聚合报告是否保留每个 worker 的证据，以及权限变更是否会重新校验。

K2.8 Preview 虽出现在上述官方产品文档中，但 2026-09-20 的 Artificial Analysis/DataCurve 快照没有精确 K2.8 条目；它不进入本项目候选清单，也不新增 K2.8 章节。K2.6 本章只把这些文档作为通用 harness 补证，不把 K2.8/K2.7/K3 的模型能力或架构倒灌到 K2.6。

## 90.13 小结

Kimi K2.6 的面试价值不在于再背一个参数规模，而在于把一条开源模型路线拆成可审计的系统：1T/32B 的 MoE 容量账本、MLA/YaRN 的上下文状态、MoonViT 的多模态输入、native INT4 的低精度部署、Agent Swarm 的并行任务图、`preserve_thinking` 的状态协议，以及 KVV 的供应商验收。

当前公开资料支持这些接口和系统事实，但没有公开完整训练 recipe、所有生产 kernel、线上 acceptance rate 或独立 K2.6 DataCurve 结果。面试回答如果能持续区分模型、运行配置、harness、推理实现和 verifier，就不会把 Agent 系统的失败或成功简单归因给基础模型。

## 90.14 来源

- [Artificial Analysis: Kimi K2.6](https://artificialanalysis.ai/models/kimi-k2-6)
- [Kimi K2.6 官方模型卡](https://huggingface.co/moonshotai/Kimi-K2.6)
- [Kimi K2.6 配置](https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/config.json)
- [Kimi K2.6 部署指南](https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/docs/deploy_guidance.md)
- [Kimi K2.6 官方博客](https://www.kimi.com/blog/kimi-k2-6)
- [Kimi Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier.html)
- [Kimi Thinking Models API guide](https://platform.moonshot.ai/docs/guide/use-kimi-k2-thinking-model)
- [Kimi Code 模型配置](https://www.kimi.com/code/docs/kimi-code/models.html)
- [Kimi Code Agent/subagent](https://www.kimi.com/code/docs/kimi-code-cli/customization/agents.html)
- [Kimi Code 内置工具与 AgentSwarm](https://www.kimi.com/code/docs/kimi-code-cli/reference/tools.html)
- [Kimi Code 会话与上下文](https://www.kimi.com/code/docs/kimi-code-cli/guides/sessions.html)
