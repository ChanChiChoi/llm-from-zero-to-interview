# Kimi K2.6：排行榜锚点、原生 INT4 与 Agent Swarm 资料摘记

核验日期：2026-09-20。本文只记录已经出现在 Artificial Analysis 的 Kimi K2.6，再沿 Moonshot/Kimi 官方模型卡、配置、部署指南、技术博客和 API 资料扩展面试相关技术。DataCurve DeepSWE 当前没有 Kimi K2.6 的精确 Agent 行，因此不把 Kimi K2.7 Code 或 Kimi K3 的结果迁移过来。

## 1. 锚点和证据边界

### 1.1 Artificial Analysis

- [Kimi K2.6](https://artificialanalysis.ai/models/kimi-k2-6) 是本轮的模型发现入口。2026-09-20 复核的三个代理快照逐字节一致，文件 `/tmp/kimi26-aa-1234.html` 为 `3,938,140` bytes，SHA-256 为 `f24ad7d9cd3ddbb253470750dcf9f47aea3fc0d675c9f8f7eb8bdbcd50dcfe18`。
- 页面日期字段为 2026-04-20；第三方字段约为 1T total/32B active、256K context、约 35.83 output tokens/s、约 3.00s TTFT，Intelligence Index 为 `26.9791999742844`。这些字段只用于确认榜单条目和复现索引，不替代 Moonshot 官方模型卡。
- 页面同时存在 `Kimi K2.6 (Non-reasoning)` 条目，并标记模型已 deprecated、指向 Kimi K3。这里的 deprecated 和迁移目标是 Artificial Analysis 的目录状态，不能改写成 Moonshot 的官方退役公告，也不能把 K3 的架构回写给 K2.6。

### 1.2 DataCurve DeepSWE

- [DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/) 的复核快照为 `268,571` bytes，SHA-256 为 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。
- 当前页面没有精确的 `mini_swe_agent_kimi_k2_6_*` 行。因而本轮不记录 K2.6 的 Pass@1、Pass@4、成本、输出 token 或 Agent steps，也不迁移 K2.7 Code/K3 的同类数据。
- 这使 K2.6 的当前状态成为 **AA 单榜资料级闭环**，而不是双榜 Agent 评测闭环。AA 指数、官方 benchmark 和 DeepSWE 结果必须分栏保存。

## 2. 官方模型卡和固定配置

主要来源：

- [Hugging Face 模型卡](https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/README.md)，固定 revision `7eb5002f6aadc958aed6a9177b7ed26bb94011bb`，`lastModified` 为 `2026-05-19T09:01:54Z`；本地快照 `/tmp/kimi26-readme-7890.md` 为 `29,325` bytes，SHA-256 为 `95db3be1d0473e482c7aa901f237ad7af8bb7929625a777b27428081bb8ea9f7`。
- [固定 config.json](https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/config.json)，本地快照 `/tmp/kimi26-config-7890.json` 为 `5,416` bytes，SHA-256 为 `85825ca6e18cbe539eb83ee09eedfb3f4222265929f06e9f535a6d9364f55899`。

README 的模型摘要公开了以下口径：

| 字段 | Kimi K2.6 官方公开值 | 证据边界 |
|---|---:|---|
| 架构 | MoE | README 的模型摘要 |
| 总参数 | 1T | 发布方模型卡口径，不等于一次请求的 resident bytes |
| Activated 参数 | 32B | 发布方激活参数口径，不等于所有 serving 显存 |
| 层数 | 61，其中 1 个 dense layer | 模型摘要和配置一致 |
| hidden size | 7168 | 文本子配置 |
| routed experts | 384 | 文本子配置 |
| selected experts/token | 8 | 文本子配置 |
| shared experts | 1 | 文本子配置 |
| context | 256K / 262,144 positions | README 与配置字段 |
| attention | MLA | README 模型摘要；完整生产 kernel 未公开 |
| vision | MoonViT，约 400M 参数 | README 模型摘要；视觉训练 recipe 未公开 |
| vocabulary | 160K / `vocab_size=163840` | README 的十进制摘要和配置字段 |

配置还公开了几个容易被误读的字段：

- 顶层 `architectures` 是 `KimiK25ForConditionalGeneration`，顶层 `model_type` 是 `kimi_k25`；`auto_map` 指向 `configuration_kimi_k25` 和 `modeling_kimi_k25`。
- `text_config` 的 architecture 是 `DeepseekV3ForCausalLM`，其 `model_type` 是 `kimi_k2`。这是当前实现的复用/兼容标识，不能据此把 K2.6 归为 DeepSeek 发布模型，也不能反推 K2.6 与 DeepSeek V3 的训练 recipe 相同。
- 文本配置有 `q_lora_rank=1536`、`kv_lora_rank=512`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`v_head_dim=128`。这些字段足以支持 MLA 和 cache 账本的面试讨论，但不足以还原完整 attention kernel 或实际每 token cache bytes。
- RoPE 配置为 YaRN，`factor=64`、`original_max_position_embeddings=4096`、`rope_theta=50000`；最大位置字段不等于每种长文本任务都有相同的召回率。
- 量化配置是 `compressed-tensors` 的 `pack-quantized`，权重 `num_bits=4`、group size `32`、对称 group quantization；配置把 attention、shared experts、若干 MLP、lm head、vision tower 和 projector 列为忽略项。不能把“4 bit”简单乘总参数得到最终端到端显存。
- 视觉配置给出 27 层视觉塔、`vt_hidden_size=1152`、16 个 attention heads、patch size 14 和 `spatial_temporal` video attention。它是配置级实现证据，不等于完整 MoonViT 论文或训练说明。

## 3. K2.5 架构复用的正确读法

[官方部署指南](https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/docs/deploy_guidance.md) 明确写出 K2.6 与 K2.5 使用同一架构、部署方法可以直接复用。部署指南快照 `/tmp/kimi26-deploy.md` 为 `3,648` bytes，SHA-256 为 `103a660460f6ce9b43a68b5c7a98a76271b5b3121dc57afd521fe11953abc530`。

这条证据支持三件事：

1. K2.6 的配置和 serving 入口可以沿 K2.5 的实现路线理解。
2. K2.6 的新能力重点更像是在同一架构路线上的模型、后训练和 Agent 工作流升级，而不是公开资料已经证明的全新 attention 家族。
3. K2.5 的技术报告、训练配方、实验数字不能无条件改名为 K2.6 专属事实；只有 K2.6 自己的模型卡、配置或官方博客支持的字段才能这样记录。

尤其不能把 Kimi K3 的 KDA、Gated MLA、Block Attention Residuals 或 Stable LatentMoE 自动迁移给 K2.6。K3 是后续独立模型路线，已有自己的技术报告和配置。

## 4. Native INT4 和部署账本

README 明确称 K2.6 采用与 Kimi K2 Thinking 相同的 native INT4 quantization 方法；配置则给出 `compressed-tensors`、4 bit、group size 32 的固定字段。native INT4 的面试重点不是“所有矩阵都变成 4 bit”，而是：

```text
权重存储精度 + group scale/zero-point + 未量化模块
        + dequant/compute dtype + KV/cache + 通信和 workspace
        -> 端到端显存与吞吐
```

官方部署路线包括：

- vLLM：建议使用 `kimi_k2` tool-call/reasoning parser；文档给出 H200 单机 TP8 示例，并说明启用 thinking 时需要 reasoning parser。
- SGLang：文档给出 `sglang>=0.5.10.post1` 的示例，同样要求 `kimi_k2` tool-call/reasoning parser。
- KTransformers：提供 CPU/GPU 异构和 RAWINT4 示例；文档的吞吐数字绑定 8 张 L20、Intel 6454S、并发和具体参数，不能泛化成所有机器的 SLO。

这里至少要分四层：模型权重的量化格式、后端对量化的加载和 kernel 支持、KV/媒体/工具状态、以及调度并发。仅凭总参数和 bit 数不能证明“能在某硬件上稳定运行”。

## 5. Agent Swarm 和长周期 coding

官方 README 和 [Kimi K2.6 技术博客](https://www.kimi.com/blog/kimi-k2-6) 将 K2.6 定位为 native multimodal agentic model，公开主线包括 long-horizon coding、coding-driven design、proactive/background agents 和 swarm-based orchestration。博客快照 `/tmp/kimi26-blog-7890.html` 为 `293,044` bytes，SHA-256 为 `d98ec25f85e628534bf0a6ab7472ec23b4a52d763dadfdf65e0488572629150a`。

博客给出的 Agent Swarm 描述是：最多 300 个 sub-agents、4,000 个 coordinated steps，动态拆分并行且领域专门化的子任务。它还描述了：

- `Qwen3.5-0.8B` 的 Mac/Zig inference optimization 案例，4,000+ tool calls、超过 12 小时连续执行、14 次迭代；
- `exchange-core` coding showcase，约 13 小时、1,000+ tool calls、4,000+ 行代码修改；
- 能够持续运行的 proactive/background agent，用于日程、代码和跨平台操作。

这些数字和案例是 Kimi 官方博客自报，必须绑定博客版本、任务、工具、环境和验收方式。它们不是 K2.6 模型内部有 300 个专家，也不是 4,000 个 token 的推理深度。更合理的系统分层是：

```text
K2.6 生成计划/子任务/工具调用
        -> Agent harness 创建并调度 sub-agent
        -> 工具、沙箱、工作区和外部服务执行
        -> 测试/验证器检查中间和最终 artifact
        -> coordinator 汇总、重试、停止或交付
```

因此面试中要追问 sub-agent 的上下文、权限、预算、取消、重试、共享状态和 artifact owner，而不是只问“模型支持多少 Agent”。

## 6. 官方 benchmark 与 harness 条件

README 的官方横向表包含 K2.6、K2.5、GPT-5.4、Claude Opus 4.6 和 Gemini 3.1 Pro。K2.6 的部分发布方数字包括 HLE-Full with tools `54.0`、BrowseComp `83.2`、BrowseComp with Agent Swarm `86.3`、DeepSearchQA F1 `92.5`，以及编码和视觉 benchmark。它们必须按发布方条件读取：

- K2.6 使用 thinking，默认实验参数为 `temperature=1.0`、`top-p=1.0`、context `262,144`；
- HLE、BrowseComp、DeepSearchQA、WideSearch 使用 search/code interpreter/web browsing 等工具，并采用不同 context management；
- Terminal-Bench 2.0 使用 Terminus-2 和 JSON parser，并在 preserve thinking mode；
- SWE-Bench 系列使用 Moonshot 的 in-house framework，报告平均 10 次独立运行；
- 视觉评测使用 max tokens `98,304`，有工具的设置另有 step/token 上限。

这组数字可以用来练习“发布方 benchmark 如何审计”，但不能和 AA Intelligence Index、DataCurve Pass@1 或其他模型的不同 harness 结果拼接成统一排名。

## 7. Thinking、preserve_thinking 和工具协议

K2.6 API 资料确认官方入口为 `https://platform.moonshot.ai`，提供 OpenAI-compatible 和 Anthropic-compatible API，并支持文本、图像、视频输入。Thinking Models 文档快照 `/tmp/kimi26-api-docs.html` 为 `622,179` bytes，SHA-256 为 `a9a977883c3520b0334582b9554a61405de8d7d8f047245f1177fe4d6ceb8f9d`。官方示例区分 Thinking 和 Instant 两种服务模式：

- Thinking 推荐 `temperature=1.0`；Instant 推荐 `temperature=0.6`。
- `preserve_thinking` 默认关闭；开启时，多轮请求需要保留完整 `reasoning_content`，以维持后续决策所需的协议状态。
- K2.6 沿用 Interleaved Thinking 和 Multi-Step Tool Call 设计；模型产生 tool call 后由宿主执行，tool result 再回灌给模型。
- Kimi Code CLI 是官方推荐的 coding-agent harness。它是运行时和工具编排层，不是 K2.6 的 Transformer 层名称。

API 资料还提醒：`reasoning_content` 会计入上下文和 token 消耗。面试中应把它描述成“可回放的思考状态协议”，而不是公开完整隐藏 chain-of-thought 的许可证。至少要记录：model/revision、thinking mode、preserve flag、tool schema version、tool call/result、权限、执行回执、错误和最终 artifact。

## 8. Kimi Vendor Verifier：把模型、推理实现和 harness 分开验收

[Kimi Vendor Verifier 官方博客](https://www.kimi.com/blog/kimi-vendor-verifier.html) 随 K2.6 开源，快照 `/tmp/kimi-vendor-verifier.html` 为 `126,942` bytes，SHA-256 为 `6d6afb1f8f850b91241badd67e35b6955c873f1c14fbee6aeaf1e63cff1416b4`。它的核心问题是：第三方部署结果异常时，问题来自模型能力、推理引擎、量化、参数约束、tool parser 还是评测 harness？

公开的六类检查是：

1. API 参数 pre-flight：检查 thinking 模式的 temperature/top_p 等约束，以及 reasoning 内容是否正确回传。
2. OCRBench：约 5 分钟的多模态管线 smoke test。
3. MMMU-Pro vision：检查不同视觉输入和预处理。
4. AIME2025：长输出压力测试，暴露 KV cache 或量化问题。
5. K2VV ToolCall：检查 trigger consistency/F1 和 JSON Schema accuracy。
6. SWE-Bench：完整 Agentic coding test，沙箱依赖使其没有完全开源。

官方博客说明完整流程在两台 NVIDIA H20 8-GPU 服务器上顺序执行约 15 小时，并支持流式推理、自动重试和 checkpoint resume。这里的测试结果是 verifier/harness 的验收结果，不是 K2.6 模型能力分数；它的价值在于帮助发现“模型没有变差，但供应商实现偏离了协议”的问题。

可以把一次部署验收写成：

```text
model weights/revision
  -> tokenizer/chat template
  -> quantization and kernel
  -> sampling/reasoning parameter gate
  -> tool/reasoning parser
  -> multimodal preprocessing
  -> Agent harness and sandbox
  -> benchmark/verifier
```

任何一层失败，都不应直接归因给基础模型。

## 9. 论文、技术报告和负面证据

截至本轮，没有找到 Kimi K2.6 专属的完整公开技术报告或独立架构论文。官方 README、配置、部署指南和博客足以支持模型摘要、Agent Swarm、native INT4、API 协议和 KVV，但不能填补以下空白：

- 完整 pre-training 数据、优化器、token 配比和后训练/RL recipe；
- 61 层的完整层排布、每层 attention/MLP 细节、路由负载均衡和生产 kernel；
- MoonViT 的完整训练与融合方案；
- native INT4 的误差消融、不同 GPU 的端到端 profiling 和线上 acceptance rate；
- Agent Swarm 的 coordinator、sub-agent model、共享记忆、权限和 verifier 的完整实现；
- 独立第三方 benchmark 复现以及 K2.6 的精确 DataCurve 行。

K2.5 的资料可以作为“同一架构路线”的背景，K2.7 Code 的资料可以作为后续 coding-focused model 的差异对照，K3 报告可以作为后续模型的独立架构路线；三者都不能被改名成 K2.6 专属技术报告。

## 10. 面试知识映射

| 面试主题 | K2.6 证据 | 面试回答应强调 |
|---|---|---|
| MoE 容量 | 1T total/32B activated、384 experts、top-8、1 shared | active 参数不是 resident memory，也不是端到端 latency |
| MLA serving | `q_lora_rank=1536`、`kv_lora_rank=512` | 压缩 cache 仍要测 index、KV、gather、通信和并发 |
| 长上下文 | 256K/262,144 positions、YaRN | context window 不等于均匀召回率 |
| 量化 | native INT4、group size 32、compressed-tensors | 量化格式、scale、未量化模块、kernel 和 compute dtype 要一起看 |
| Agent Swarm | 300 sub-agents、4,000 steps 的官方案例 | sub-agent 数量属于 harness/系统，不是 MoE expert 数量 |
| 思考状态 | `preserve_thinking`、`reasoning_content` | 是协议回放和上下文成本问题，不等于任意暴露隐藏思维 |
| 部署验收 | KVV 六类测试 | 模型、引擎、参数、parser、harness、verifier 分层归因 |
| 评测公平性 | 官方 benchmark 条件和 AA/DataCurve 分栏 | 固定 revision、effort、工具、环境、预算和 verifier |

## 11. 当前状态与待核验项

Kimi K2.6 当前为 **AA 单榜资料级闭环**：已具备两个排行榜的复核、官方模型卡/配置、部署指南、Agent Swarm 博客、API 协议、KVV 验收资料、研究笔记和正式书籍章节。当前不新增 K2.6 专属技术报告条目，也不把 K2.7/K3 的架构或 DeepSWE 结果迁移过来。

仍待核验：完整训练/后训练 recipe、完整层排布和 production kernel、INT4 端到端误差与硬件 profiling、MoonViT 训练细节、Agent Swarm 的内部 coordinator 与 verifier、线上 tool acceptance、K2.6 精确 DataCurve Agent 行和独立复现。下一锚点仍只能从 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜中的八家重点厂商条目选择。

## 12. 当前 Kimi Code harness 补证（2026-09-20）

本节补充的是当前 Kimi Code 产品和 CLI 的公开 harness 文档，不是 Kimi K2.6 专属的 coordinator 实现，也不能把其中的模型 ID、subagent 上限或 session 机制回写成 K2.6 的 Transformer 结构。它的价值在于补齐 K2.6 相关 coding-agent 面试中的运行时、权限和可恢复性证据。

### 12.1 当前模型配置与候选边界

[Kimi Code 模型配置](https://www.kimi.com/code/docs/kimi-code/models.html) 当前写明 Kimi Code 提供 K3、K2.8 Preview 和 K2.7 Code HighSpeed 三类模型、共 4 个模型 ID：

| Model ID | 官方文档中的版本 | 文档级运行信息 |
|---|---|---|
| `k3` | K3 | 最高 1M 上下文 |
| `k3-256k` | K3 | 256K 上下文版本 |
| `kimi-for-coding` | K2.8 Preview | 文档称综合性能接近 K3，支持 `low/high/max` |
| `kimi-for-coding-highspeed` | K2.7 Code HighSpeed | 文档称为 K2.7 Code 高速版，256K 上下文 |

这份官方产品表是 Kimi Code 的服务配置说明，不是本项目的模型发现入口。2026-09-20 的 Artificial Analysis 快照没有 `K2.8`/`kimi_k2_8` 条目，DataCurve 快照也没有精确的 K2.8 Agent 行。因此 K2.8 Preview 只记录为“官方周边文档中的关联版本”，不加入候选盘点、不建立 K2.8 独立研究笔记，也不把 K2.8 的 `effort` 或上下文字段迁移给 K2.6。当前 Kimi Code 表中的 K3/K2.7 配置同样不能反向证明 K2.6 使用了它们的协议或实现。

### 12.2 Agent、AgentSwarm 与权限控制

[Kimi Code 内置工具文档](https://www.kimi.com/code/docs/kimi-code-cli/reference/tools.html) 和 [Agent/subagent 文档](https://www.kimi.com/code/docs/kimi-code-cli/customization/agents.html) 给出了当前 CLI harness 的可审计边界：

- `AgentSwarm` 从 `prompt_template` 和 `items` 创建 subagent，也可以用 `resume_agent_ids` 恢复已有 subagent；新 subagent 可以从配置的模型池选择模型，恢复的 subagent 保留原模型。
- 一次 swarm 最多 128 个 subagent，等待全部完成后返回聚合报告；每个 subagent 默认 2 小时超时，可用 `[swarm] timeout_ms` 或 `KIMI_CODE_SWARM_TIMEOUT_MS` 调整。
- 如果模型响应调用了 `AgentSwarm`，它必须是该响应中的唯一工具调用；默认并发从 5 个开始，之后每 700ms 增加 1 个，可用 `KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY` 设置正整数上限。
- `Agent`/`AgentSwarm` 的 `tools`、`disallowedTools` 和 `subagents` 既影响模型可见的工具/类型列表，也会在真正派发前再次校验；权限审批是另一层控制面。subagent 默认继承 main agent 的权限规则，内置 `coder`、`explore`、`plan` profile 不能继续递归派发。

这组 128、2 小时、恢复、聚合、并发和权限规则属于当前 Kimi Code CLI harness。K2.6 官方博客的“最多 300 个 sub-agents、4,000 个 coordinated steps”属于另一份产品/发布方案例，二者必须按来源、产品形态和时间分别保存，不能取最大值写成“K2.6 的 Agent 上限”，也不能把 128 写成 K2.6 模型内部结构。

### 12.3 会话状态、事件流与上下文管理

[Kimi Code 会话文档](https://www.kimi.com/code/docs/kimi-code-cli/guides/sessions.html) 将会话持久化为工作目录和 `sessionId` 下的状态文件与 `agents/*/wire.jsonl` 事件流。事件流用于恢复和回放，还记录发给模型的工具 schema、请求参数和 MCP 工具清单。公开的操作包括 `kimi --continue`、`kimi --session <id>`、`/compact`、`/fork` 和 `kimi export`。

因此面试中可以把 Kimi Code 的 coding agent 描述成“模型 + 可恢复事件流 + 工具/权限快照 + workspace”的系统，而不是把 session 简化为 prompt 字符串。`/compact`、fork、resume 和聚合报告都需要记录 artifact owner、工具回执、上下文策略和最终 verifier；文档没有公开 K2.6 专属 coordinator 的内部状态机，不能继续推断。

本轮 Kimi Code 官方快照用于复核：README `/tmp/kimi-code-readme-7890.md` 为 `5,334` bytes、SHA-256 `50ea102c7003c746244c69cb78cbd5b1e9158311e5e4e028aee8a2afee698c4c`；模型、Agent、工具、会话和供应商文档快照分别为 `/tmp/kimi-code-models-7890.html`（`68,326` bytes，SHA-256 `21e4975dbb5e6ac5ac90a024580817ec998fa85d7613b5dc0524acc04b27ac7e`）、`/tmp/kimi-code-agents-1234.html`（`77,630` bytes，SHA-256 `227ff82f499a981659c8f7dbae78744d7256e7b52a6697d9c34e9ff726543e41`）、`/tmp/kimi-code-tools-1234.html`（`80,291` bytes，SHA-256 `687a02447dc09831aec88a87fdfa79b6c0ff244805eaf81f7cbf16138a528e68`）、`/tmp/kimi-code-sessions-1234.html`（`63,043` bytes，SHA-256 `38cc2cc87176871243f211f9044402b8286b2a42b0eb1aa275f2d65517b20c51`）和 `/tmp/kimi-code-providers-1234.html`（`72,766` bytes，SHA-256 `3d3b790e2edbd8054b438edb25ccc3e61db0dbd764364aabd7366a6661932153`）。

## 13. 来源清单

- [Artificial Analysis: Kimi K2.6](https://artificialanalysis.ai/models/kimi-k2-6)
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/)
- [Kimi K2.6 Hugging Face 模型卡](https://huggingface.co/moonshotai/Kimi-K2.6)
- [Kimi K2.6 固定 config.json](https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/config.json)
- [Kimi K2.6 部署指南](https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/docs/deploy_guidance.md)
- [Kimi K2.6: Advancing Open-Source Coding](https://www.kimi.com/blog/kimi-k2-6)
- [Kimi Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier.html)
- [Kimi Thinking Models API guide](https://platform.moonshot.ai/docs/guide/use-kimi-k2-thinking-model)
- [Kimi Code CLI](https://www.kimi.com/code)
- [Kimi Code 模型配置](https://www.kimi.com/code/docs/kimi-code/models.html)
- [Kimi Code Agent 与 subagent](https://www.kimi.com/code/docs/kimi-code-cli/customization/agents.html)
- [Kimi Code 内置工具与 AgentSwarm](https://www.kimi.com/code/docs/kimi-code-cli/reference/tools.html)
- [Kimi Code 会话与上下文](https://www.kimi.com/code/docs/kimi-code-cli/guides/sessions.html)
- [Kimi Code 平台与模型](https://www.kimi.com/code/docs/kimi-code-cli/configuration/providers.html)
- [MoonshotAI/kimi-code](https://github.com/MoonshotAI/kimi-code)
- [Kimi K2.6 官方 API](https://platform.moonshot.ai)
