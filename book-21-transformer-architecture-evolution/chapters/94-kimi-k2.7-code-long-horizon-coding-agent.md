# 第 94 章 Kimi K2.7 Code：MoE/MLA、INT4 与长周期编码 Agent

> 核验日期：2026-09-24；固定 revision 部署资料复验：2026-09-29。模型锚点来自 Artificial Analysis 的 [Kimi K2.7 Code](https://artificialanalysis.ai/models/kimi-k2-7-code) 与 DataCurve DeepSWE 的精确配置 mini_swe_agent_kimi_k2_7_code_default。架构和量化摘要来自固定 revision 的 [Kimi K2.7 Code 模型卡](https://huggingface.co/moonshotai/Kimi-K2.7-Code/tree/74797c9c62378b951a1f6fcf5c4631024e9b8bef) 与 [配置](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/config.json)；当前 API 字段来自 [Kimi 官方快速开始](https://platform.kimi.com/docs/guide/kimi-k2-7-code-quickstart)。官方技术细节与榜单成绩均保留各自来源责任，不代表本项目下载权重、调用真实 API 或复现生产性能。

## 94.1 一个修复任务，为什么不只是“让模型多写几行代码”

假设 Agent 要修复一个跨模块 bug：它先读仓库、定位调用链，再打开测试、运行工具、接收报错，改动多个文件，最后还得重跑验证。模型每次只看到一段上下文；工具实际做了什么、推理状态是否保留下来、任务预算是否用完、最终改动是否真的通过测试，都由模型以外的运行系统决定。

Kimi K2.7 Code 值得研究的地方，正好横跨几层：一个沿 K2.6 路线的稀疏 MoE/MLA 权重；面向长程编码的产品定位；强制 thinking 的 API 合同；多轮工具状态回传；以及绑定具体 CLI、参数、任务集与 verifier 的评测结果。面试时先问“哪一层在起作用”，往往比背一个排行榜分数更有价值。

## 94.2 先读榜单：同一模型的身份、配置与 Agent 结果

| 证据来源 | 2026-09-24 可观察值 | 能说明什么 |
|---|---|---|
| Artificial Analysis | canonical slug kimi-k2-7-code；当前 Intelligence Index 25.8121062401836，页面展示约 26；context 字段 256K | 第三方评测和 provider/configuration 快照，不是参数量或官方基准 |
| DataCurve DeepSWE v1.1 | mini_swe_agent_kimi_k2_7_code_default；113 tasks、4 runs、452 attempts、138 passed；Pass@1 30.53%、Pass@4 61.06% | mini-swe-agent、工具、仓库环境、预算与 verifier 组成的 Agent 系统结果 |
| Kimi 官方 API 文档 | 默认 max_tokens=32768；支持 256K context；thinking 默认启用且不可关闭 | 32K 是默认单次生成上限，256K 是上下文窗口，两者不是同一个预算 |

DataCurve 该行平均成本约 $2.82、平均输出约 59.3K tokens、平均输入约 13.01M tokens、平均 Agent steps 149.1。这些均绑定页面所用的 provider、mini-SWE harness、任务和运行次数；不能与 Artificial Analysis 的 Intelligence Index 拼成一个“裸模型总分”。

从本轮 K2.7 Code 详情页提取的 Intelligence Index 为 25.8121062401836。和历史快照中约 26.27 的观测值相比，数字变化本身只能先记为页面/评测时点变化；没有权重 revision、官方发布或训练变更证据时，不能叫作模型升级。

## 94.3 1T 总参数、32B 激活参数和显存账本

固定 revision 的模型卡给出约 1T total parameters、32B activated parameters、61 层、384 个 routed experts（每 token 选 8 个）和 1 个 shared expert。文本 hidden size 为 7,168，expert intermediate size 为 2,048，attention heads 为 64，词表约 160K。这里的 total 和 active 是不同口径：

- **Total parameters**描述模型所有专家与共享权重的总容量。
- **Activated parameters**描述一个 token 经过主要路由路径参与计算的参数规模；它不表示其余专家权重不需要存储或读取。
- **Resident parameters**是某个部署实例当前驻留在 HBM/CPU/NVMe 的权重部分；它由分片、offload、并行方式和运行时决定。
- **Active compute**还要支付 attention、router、shared expert、dispatch/combine 和通信开销。

所以不能用 32B 直接估算模型显存。可审计的权重账本至少是：

~~~text
resident weight bytes
  = packed quantized weights
  + non-quantized/excluded weights
  + scales / packing metadata / alignment
  + loader and allocator overhead
~~~

模型卡称 K2.7 Code 使用与 K2 Thinking 相同的 native INT4 方法；固定配置包含 compressed-tensors 的 pack-quantized、4-bit、group size 32 和对称 group quantization。与此同时，attention、shared experts、部分 MLP、lm head、vision tower 与 projector 出现在量化忽略项中。因此“native INT4”不是“整份 checkpoint 每个参数都恰为 4 bit”。

只为建立量级直觉：如果假设 1T 参数全部以 4 bit 紧密打包，权重原始数据下限为

~~~text
1e12 parameters × 4 bits / 8
  = 500,000,000,000 bytes
  ≈ 465.7 GiB
~~~

这个是假设的**全量打包权重原始字节数**，不是 K2.7 Code 实测文件大小或 GPU 峰值。被忽略的高精度模块、group scale、张量切片 padding、kernel workspace、KV cache、并发请求和视觉输入都会改变实际资源账本。

## 94.4 MLA 与 256K：缓存表示不是上下文质量

配置披露 MLA 以及 kv_lora_rank=512、qk_rope_head_dim=64、qk_nope_head_dim=128、v_head_dim=128。MLA 的关键思路是把历史 key/value 表示压缩到较低维 latent，再结合位置相关部分计算注意力；它改变 KV cache 的表示和带宽账本，但不会让长上下文免费，也不能由“窗口 256K”推出任意任务都能等质量检索。

若仅为教学而假设某实现每层、每 token 缓存一个 512 维 latent 和一个 64 维 RoPE key，并采用 2-byte 元素，则：

~~~text
proxy_bytes_per_token
  = 61 layers × (512 + 64) dimensions × 2 bytes
  = 70,272 bytes/token
~~~

乘上 262,144 tokens，代理估算约为 17.16 GiB/序列。这个演算只展示“层数 × 每层缓存宽度 × dtype × token 数”的增长关系；它不是 vLLM/SGLang 的已验证 K2.7 cache layout、真实 batch 显存或 256K 长度可服务证明。部署时仍需固定 backend、cache dtype、page size、并发和精确权重 revision 实测。

还有一个常见混淆：context window 是一次请求可容纳的输入/状态范围；API 的 max_tokens 则是本次生成上限。官方快速开始当前给出默认 max_tokens=32768，不能将其误读成 256K 输出，更不能把最大上下文当成有效记忆或 KV 显存承诺。

## 94.5 Code 专项的改进声明应带上条件

官方模型卡把 K2.7 Code 定位为沿 K2.6 路线的 coding-focused agentic model，并称其在复杂软件工程任务上更强，同时约减少 30% thinking-token 使用。当前 API 快速开始还称 Kimi Code Bench v2、Program-Bench、MLS Bench Lite 相对 K2.6 分别提升 21.8%、11%、31.5%；Kimi Claw 24/7、MCP Atlas、MCP Mark Verified 则约提升 10%。

这些是发布方的对照与摘要，不是本地 profile。要把“少 30% thinking token”转成可验证的工程结论，至少要问：两版是否使用同一任务集、CLI/harness、工具、context policy、超时和成功判据？token 是否只算 thinking block，还是包含工具调用、重试和最终回答？最终要比较的是每次请求 token，还是每个成功交付任务的成本？

因此完整的成本目标更接近：

~~~text
cost_per_success
  = total model + tool + retry + verifier cost
    / number of independently verified successful artifacts
~~~

减少内部 reasoning-token 不保证总 API 成本下降；如果 Agent 需要更多工具轮次或失败重试，单位成功成本仍可能变高。

## 94.6 强制 thinking：API 参数是合同，不是架构说明

当前官方快速开始规定 K2.7 Code 不支持关闭 thinking。相关参数和默认值如下：

| 字段 | 文档约束 | 工程含义 |
|---|---|---|
| thinking | 默认 {"type":"enabled"}；关闭会报错 | 不要把它当作可切换的 non-reasoning checkpoint |
| max_tokens | 默认 32,768 | 单次输出上限；与 256K context 分开记 |
| temperature | 固定 1.0；指定其他值报错 | 服务端限制，不是公开训练超参数 |
| top_p | 固定 0.95 | 不代表可以从静态字段推出采样质量 |
| n | 固定 1 | 单请求候选数量受 API 限制 |
| presence_penalty / frequency_penalty | 固定 0.0 | 其他值可能返回错误 |
| tool_choice | 仅 auto 或 none | 不能要求模型必须调用某个函数 |

官方建议不要手动设置上述参数，优先采用默认值。面试时要分清“接口可调用字段”“线上服务实际允许值”和“模型内部算法”：API 合同可以说明如何发请求，不能证明模型里的 thinking 实现、RL 目标或解码器细节。

## 94.7 多轮工具调用：保存状态，也要保存执行事实

一次工具循环可以抽象为：

~~~text
assistant(reasoning_content, tool_call[id])
  -> host validates schema and permission
  -> executor runs side effect and records receipt
  -> tool(tool_call_id, result)
  -> assistant(next reasoning_content, next tool_call or final)
~~~

tool_call 是模型提出的动作，不是已授权或已执行的事实。宿主仍要检查 schema、权限、路径/域名 allowlist、超时、幂等、回执和最终 artifact。

Kimi 文档特别建议在多步调用中把 assistant message 里的 reasoning_content 原样留在上下文中；不回传通常不会报错，但可能影响后续推理连贯性和工具效果。这是一个**协议层状态连续性建议**，不是把任意隐藏思维交给业务应用，也不代表模型拥有持久记忆。重放日志应至少关联 assistant 消息、tool-call ID、工具版本、授权决定、执行回执和 verifier 结果。

断线重试时还有一个副作用陷阱：如果 tool_call 已执行但调用方没收到结果，直接重新执行可能重复写文件、发邮件或部署。正确系统要由宿主维护幂等键与执行账本，先查询未知状态，再决定是否重试；模型 API 的 reasoning state 不能代替这个账本。

## 94.8 多模态 coding agent：让屏幕与视频进入工具循环

K2.7 Code 的公开输入包括文本、图像和视频，模型卡列有约 400M 参数的 MoonViT vision encoder。官方快速开始展示了一个视频分析工具：宿主接收模型给出的片段起止时间，用 ffprobe 读取时长，再用 ffmpeg 截取片段，随后以多模态 tool result 回灌视频与描述。

这类设计不是“模型凭空看到了机器上的视频”。模型只能看到宿主实际提供的视觉内容。把它落到 coding Agent 时，必须记录：

1. 输入媒体如何上传、抽帧或裁剪；
2. 每段视觉内容用了多少 token；
3. 工具返回的是 URL、base64 还是 file handle；
4. 结果是否被裁剪、转码或重复回灌；
5. 模型建议的时间段是否被允许访问；
6. 最终回答或代码修改是否由独立 verifier 验证。

当前文档称图像/视频 token 按内容动态计算；视频由多张关键帧组成，分辨率和关键帧数量都会改变 token 消耗。官方提供 token-estimate 接口，建议开始理解前估算视觉请求的 token。快速开始列出最大 100 MB request body、图片不支持 URL（当前需 base64）、大视频建议走文件上传，并推荐图片不超过 4K、视频不超过 1080p。这些是当前 API 入口的服务限制，部署前仍要检查账户和文件接口的最新合同。

因此视觉 coding 的上下文预算不是简单的“代码 token + 一张图”：它还包括视频关键帧、tool result、thinking state、历史代码、重试和最终输出。把一个 20 分钟视频完整塞进每一轮通常既浪费预算，也让状态恢复更脆弱；按需选取片段并把提取参数写入 trace，更容易审计。

## 94.9 Highspeed 是服务变体，不是模型发现入口

2026-09-24 的 Kimi API 快速开始新增说明：kimi-k2.7-code-highspeed 与 kimi-k2.7-code 是同一个模型，但官方宣称输出速度约为普通版 5–6 倍；常规编程输入长度中位数场景约 180 tokens/s，短上下文场景可达约 260 tokens/s。文档同时注明资源有限、体验可能波动，并在逐步扩容。

本项目只允许从 Artificial Analysis 与 DataCurve DeepSWE 发现模型。因而这里把 Highspeed 记录为**已发现 Kimi K2.7 Code 的服务/路由变体**，不新增 canonical model、不复制一行模型盘点，也不把发布方速度当作本地测量。真实选型还需固定 provider、输入长度、输出长度、并发、TTFT、TPOT、错误率和单位成功成本。

## 94.10 教学审计器：把请求限制和近似账本变成可检查代码

零依赖脚本 [kimi_k27_contract_budget_toy.py](../../research/model-update-2026-09/code/kimi_k27_contract_budget_toy.py) 做三件事：核对文档列出的请求默认/固定值、检查 tool-call ID 与 tool result 是否对应、估算全量 1T INT4 原始字节数和一个假设性的 MLA cache shape。

运行：

~~~bash
python3 research/model-update-2026-09/code/kimi_k27_contract_budget_toy.py
~~~

它会拒绝关闭 thinking、非默认 temperature 或 tool_choice="required"；但如果下一轮缺少 reasoning_content，只输出连续性警告，因为官方文档称省略通常不会报错。内存结果采用 1T 全量 INT4、以及“每层每 token 一个 512 维 latent + 一个 64 维 RoPE key、元素 2 bytes”的假设。实际模型含不量化模块，实际缓存布局也需要 runtime 源码和目标硬件确认。

因此脚本输出的 local_protocol_toy 只说明审计逻辑和算术代理可运行；不调用真实 API、不下载/加载权重、不验证实际生成质量、吞吐、缓存正确性或生产接受率。

## 94.11 异构推理与 LoRA SFT：把发布方吞吐放回硬件账本

2026-09-29 经代理重取的固定 revision [部署指南](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/docs/deploy_guidance.md) 展示了三种部署路径。vLLM 部分称 nightly wheel 仍属实验性，同时称 vLLM 0.19.1 已人工验证；SGLang 部分称 v0.5.10 及以后 stable 可用；两者都给出 H200 单机 TP8 示例并要求 `kimi_k2` tool-call/reasoning parser。

KTransformers 示例更能体现异构资源分工：K2.7 Code + SGLang 使用 RAWINT4、CPU expert inference 与部分专家 GPU 驻留。发布方给出的命令参数包含 `--kt-cpuinfer 96`、`--kt-num-gpu-experts 30`、tensor parallel 4、prefill 阈值 400 和 50K `max-total-tokens`。指南报告在 8×NVIDIA L20 + 2×Intel 6454S、48 路并发下 prefill 640.12 tokens/s、decode 24.51 tokens/s。

同一指南还提供 KTransformers + LLaMA-Factory 的 LoRA SFT 示例，发布方报告端到端吞吐 44.55 tokens/s，硬件为 2×RTX 4090 + Intel 8488C，并写有 `1.97T RAM`、`200G swap`。`1.97T RAM` 是原文记法，来源没有解释单位；不将它擅自规范成 TB 或 GB。SFT tokens/s 与 serving prefill/decode tokens/s 的任务、统计口径和并发不同，不能横向比较。

面试重点不是背这些速度，而是追问：RAWINT4 覆盖哪些张量、CPU/GPU 专家分工如何改变 dispatch 与 decode latency、48-way 的并发定义、prefill/decode 的计时边界、SFT 的序列长度与有效 token 口径、以及 swap/主存容量是否为 bottleneck。指南没有给出足够的完整 workload 与计时方法用于独立复现；这些仍是 publisher-reported 场景结果，不是本项目硬件验收。

## 94.12 面试中值得追问的边界

1. 为什么 1T total / 32B active 不能直接换算成 32B 权重显存？
2. native INT4 与配置里的 quantization ignore list 如何共同影响 checkpoint 和 HBM 账本？
3. MLA 的低维 cache proxy 为什么不是实际 KV cache profile？
4. 32K 默认 max_tokens 与 256K context 有什么区别？
5. reasoning_content 回传改善连续性，为什么仍不能替代 tool receipt、幂等和 verifier？
6. 视频抽帧数量与分辨率怎样同时影响 token、延迟、成本和长任务恢复？
7. 官方宣称 thinking tokens 减少 30%，还缺哪些任务级成本证据？
8. Highspeed 的“同一模型”说法怎样与 leaderboard model identity、provider route 和实测吞吐分栏？

## 94.13 当前证据等级

当前状态为**双榜内容专题闭环**：AA 与 DataCurve canonical 条目、固定 revision 模型卡/配置/部署指南、官方 API 资料、研究笔记、正式章节和本地协议教学代码均已对齐。部署指南的吞吐/训练数字只是发布方给定硬件场景，不代表本地复现；KTransformers RAWINT4 命令与同提交通用支持矩阵尚未对齐，K2.7 兼容性未实测。专属技术报告/完整训练配方、真实 API probe、完整权重加载、目标硬件 profile、独立 benchmark、线上工具接受率和生产 SLO 仍未确认。闭环仅表示面试知识专题所需的公开材料已整理，不表示完整模型或生产验收完成。

## 94.14 KTransformers：部署命令与通用支持矩阵要分开读

Kimi 固定 revision 的 K2.7 部署指南称 K2.7 Code 与 K2.5/K2.6 架构相同、部署方法可复用，并给出 KTransformers + SGLang 的 RAWINT4 命令。该指南还明确说示例未必是最优配置。命令配置 `--kt-cpuinfer 96`、`--kt-threadpool-count 2`、`--kt-num-gpu-experts 30`、`--kt-gpu-prefill-token-threshold 400` 和 TP=4；吞吐数字绑定指南指定的 8×L20、2×Intel 6454S 与 48 路并发。

但 KTransformers 固定提交 [c40722b](https://github.com/kvcache-ai/ktransformers/commit/c40722bf04c494f2492b7eb9e86ef01a4ede45b3) 的 [Native Precision 支持矩阵](https://github.com/kvcache-ai/ktransformers/blob/c40722bf04c494f2492b7eb9e86ef01a4ede45b3/doc/en/kt-kernel/Native-Precision-Tutorial.md)只把 `RAWINT4` 列在 Kimi-K2-Thinking 名下，当前 `kt-cli` 名单也没有 K2.7/K2.5；[K2.5 专页](https://github.com/kvcache-ai/ktransformers/blob/c40722bf04c494f2492b7eb9e86ef01a4ede45b3/doc/en/Kimi-K2.5.md)则提供相邻型号的 RAWINT4 例子。这是文档覆盖不一致，不足以裁定“不支持”或“已支持”。严谨记录是：K2.7 有发布方部署命令，仓库通用矩阵未明确覆盖，缺少本地 wheel/权重加载和数值测试。

通用教程建议 `kt-cpuinfer` 约取物理核数的 90%，`kt-threadpool-count` 对应 NUMA 节点数；`kt-num-gpu-experts` 表示每个 MoE layer 驻留在 GPU 的专家数。Dual prefill 文档将 `< threshold` 分给 CPU-GPU hybrid，将 `>= threshold` 分给 layerwise prefill；layerwise 路径会把 CPU 权重传到 GPU，增加额外显存需求。因此阈值 400 是需要按 prompt 长度分布和 VRAM profile 调优的配置，不是普适最优值。K2.7 示例没有显式启用 `--kt-enable-dynamic-expert-update`，不能将该命令说成启用了动态专家重排。

## 94.15 LoRA SFT：不能把 RAWINT4 serving 等同于 INT4 training

K2.7 部署指南报告 KTransformers + LLaMA-Factory 的 LoRA SFT 命令和指定硬件吞吐，但没有在该段说明基座权重训练精度或 INT4 到训练格式的转换。KTransformers 同一固定提交的旧 [K2.5 source-install SFT 教程](https://github.com/kvcache-ai/ktransformers/blob/c40722bf04c494f2492b7eb9e86ef01a4ede45b3/doc/en/SFT_Installation_Guide_KimiK2.5.md)顶部特别注明：`0.7.0.post4` 应使用新的 release tutorial，后面的 BF16 转换步骤属于旧版本，不能直接当作当前发行配方。

固定提交中的 [K2.5 `0.7.0.post4` release tutorial](https://github.com/kvcache-ai/ktransformers/blob/c40722bf04c494f2492b7eb9e86ef01a4ede45b3/.github/release/examples/kimi-k25/README.md)要求使用完整原始 K2.5 权重、不转成全 BF16；配套 YAML 同时有 `kt_backend: RAWINT4`、`kt_expert_weight_format: rawint4` 和 `bf16: true`。这把权重存储/后端格式与训练精度字段分开了，文档没有逐算子解释所有 dtype。该教程是 NekoQA 风格微调，使用 K2.5 固定 revision、8×RTX 5090 参考配置、LoRA rank 8/alpha 16、最长 4096 tokens、每卡 batch 1 和梯度累积 8；配置覆盖 attention 与 fused expert LoRA。checkpoint 分开保存普通和 expert adapter 及训练状态，resume 验收还比较 LoRA tensor、optimizer、RNG、scheduler 与恢复后的 loss；训练 adapter 随后转换为 SGLang 格式。以上均是 K2.5 `0.7.0.post4` 的发行文档说明，不是本地复现。这里的 K2.5 数字与 K2.7 部署指南报告的 2×RTX 4090 场景不是同一个 recipe，不能横向比较。

K2.5 是相邻型号证据，不是 K2.7 的专属训练合同。在没有 K2.7 专属训练 YAML、权重 revision 或实测前，不断言 K2.7 RAWINT4 checkpoint 可直接按 K2.5 配方做 SFT；也不把旧 source-install 教程的全 BF16 转换步骤当成 `0.7.0.post4` 要求。复现还需锁定 K2.7 权重 revision、LoRA target modules、sequence length、有效 token 统计、显存/主存和优化器设置。
