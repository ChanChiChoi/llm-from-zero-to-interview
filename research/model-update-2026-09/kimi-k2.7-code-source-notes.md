# Kimi K2.7 Code：长周期编码 Agent、MoE/MLA 与思考状态协议

初次核验日期：2026-09-15；榜单/官方资料复核日期：2026-09-24、2026-09-29。本笔记只把 Artificial Analysis 与 DataCurve DeepSWE 的条目作为 Kimi K2.7 Code 的模型锚点，再沿 Moonshot/Kimi 官方资源页、API 文档、Hugging Face 模型卡、配置和部署指南提取面试相关知识。排行榜结果、官方模型卡 benchmark 和产品运行时字段分别承担不同证据责任，不能互相替代。

## 1. 榜单锚点与评测边界

### Artificial Analysis

- [Kimi K2.7 Code 详情页](https://artificialanalysis.ai/models/kimi-k2-7-code) 的 canonical slug 是 `kimi-k2-7-code`，页面 `releaseDate` 字段为 2026-06-12。该日期是 Artificial Analysis 的第三方目录字段，不当作 Moonshot 官方发布日期。
- 本次恢复后的详情页快照保存为 `/tmp/aa-kimi-k27-20260915-rerun.html`，大小 3,605,694 bytes，SHA-256 为 `e37ce8e8f224233a95ca93771fe9f2130e544ef071703bae475295bd1ac527db`。
- 页面当前给出约 26.27 的 Artificial Analysis Intelligence Index、约 48.46 output tokens/s、256K context 和约 `$0.54`/Intelligence Index task。它们是第三方在其固定任务和供应商条件下的测量，不是基础模型的裸能力、永久价格或官方 benchmark。

### DataCurve DeepSWE

- [DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/) 快照页面注明 113 个任务、91 个仓库、5 种语言，并对模型统一使用 `mini-swe-agent` harness。
- `kimi-k2-7-code` 的配置是 `mini_swe_agent_kimi_k2_7_code_default`，`reasoning_effort:null`，`n_runs:4`。
- 该行结果为 `138/452`，Pass@1 为 `0.3053097345`（约 30.53%），Pass@4 为 `0.6106194690`（约 61.06%），平均成本 `$2.8155`，平均输出 59,297 tokens，平均 149.1 个 Agent steps。95% run-to-run 区间的半宽约 0.5 个百分点。
- DataCurve 快照保存为 `/tmp/deepswe-20260915-rerun.html`，大小 268,313 bytes，SHA-256 为 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；它与本地此前保存的 2026-09-03 页面字节一致。

DeepSWE 的观测对象应写成：

```text
result = F(base_model, model_revision, harness, tools, task_set,
           verifier, timeout, retry, context_policy, provider)
```

因此不能把 Artificial Analysis 的指数、DeepSWE 的 Pass@1 或 Agent steps 拼成一个总排名，也不能把 `mini-swe-agent` 的工具执行结果直接归因于 Kimi K2.7 Code 本身。

## 2. 官方资料与模型身份

- 官方资源页：[Kimi K2.7 Code: Open-Source Agentic Coding Model](https://www.kimi.ai/resources/kimi-k2-7-code)。本次快照大小 208,201 bytes，SHA-256 为 `36bdf7c4676551a5384a4ffaa1cc0c04ecf5cbcdd5da858bc04b142043d96945`。
- API 快速开始：[Kimi K2.7 Code](https://platform.kimi.com/docs/guide/kimi-k2-7-code-quickstart)。本次 HTML 快照大小 433,442 bytes，SHA-256 为 `10a7875220daccec89d64108631200cd5379f3f60991934f7f13b218ce5fb09a`。
- 官方模型卡：[moonshotai/Kimi-K2.7-Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code)。本轮使用固定 revision `74797c9c62378b951a1f6fcf5c4631024e9b8bef` 的 raw README，大小 15,061 bytes，SHA-256 为 `c78bbe5af19636b180eda956abf1f30a79027ec04d6642ebddc44edbc4470879`。
- 固定配置：[config.json](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/config.json)。配置快照大小 5,420 bytes，SHA-256 为 `ffbb57bff844e024f6c112640b63da80af228ceb8fb3e2e21908a21febbcdffb`。
- 部署指南：[docs/deploy_guidance.md](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/docs/deploy_guidance.md)。2026-09-29 经 `10.24.27.134:7890` 固定 revision 复取 HTTP 200，3,689 bytes / SHA-256 `b1bc4c5fb7c8b1da727663d4da83999afc902f6166ff862fbaef514b86de2c65`；与工作区留存的 2026-09-28 快照逐字节一致。

官方模型卡把 Kimi K2.7 Code 描述为基于 Kimi K2.6 的 coding-focused agentic model，重点是现实世界的长周期编码任务、复杂软件工程工作流的端到端完成，以及相对 Kimi K2.6 约 30% 的 thinking-token 使用量下降。这里的“约 30%”是发布方自述，不是独立 profiling 结论。

### 2.1 公开架构账本

模型卡公开的摘要字段如下：

| 字段 | 官方公开值 | 面试解释 |
|---|---:|---|
| 架构 | MoE | 总参数与每 token 激活参数分离，不能只报一个参数数字 |
| 总参数 | 1T | 稀疏专家集合的总容量 |
| 激活参数 | 32B | 单 token 路由到的计算规模口径，不等于显存占用 |
| 层数 | 61 | 包含 1 个 dense layer |
| routed experts | 384 | 每个 token 选 8 个 |
| shared experts | 1 | 共享路径不随 token 路由到单个专家 |
| attention hidden dimension | 7168 | 文本主干 hidden size |
| expert hidden dimension | 2048 | 每个 MoE expert 的中间维度 |
| attention heads | 64 | 与 MLA 配置共同决定 Q/K/V 投影账本 |
| vocabulary | 160K | 模型卡的四舍五入表示 |
| context length | 256K | 配置为 `max_position_embeddings=262144` |
| attention | MLA | 以低秩 KV 表示降低长上下文 KV cache 账本 |
| activation | SwiGLU | 门控 FFN 路线 |
| vision encoder | MoonViT | 400M 参数；支持图像和视频输入 |

固定 `config.json` 还公开了 `q_lora_rank=1536`、`kv_lora_rank=512`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、YaRN `factor=64`、原始位置长度 4096 和 `rope_theta=50000`。这些字段支持对 MLA/长上下文实现进行代码级核对，但不应扩写成训练方法或性能保证。

配置顶层的 Transformers 类名仍是 `KimiK25ForConditionalGeneration`，文本配置的 `model_type` 为 `kimi_k2`，这与模型卡“沿用 Kimi K2.5/K2.6 架构路线”的描述一致。类名和复用路径是实现身份，不等于 K2.7 与前代在权重、后训练或数据上完全相同。

### 2.2 Native INT4 与部署

模型卡明确写明 Kimi K2.7 Code 采用与 Kimi K2 Thinking 相同的 native INT4 方法。固定配置还显示：

- `compressed-tensors` 的 `pack-quantized` 格式；
- 权重 4 bit、group size 32、对称 group quantization；
- self-attention、shared experts、部分 MLP、lm head、vision tower 和 projector 列在量化忽略项中。

因此面试回答要区分“模型卡宣称的 native INT4 路线”和“配置中哪些模块实际进入压缩权重范围”，不要把 1T 总参数简单乘以 0.5 bytes 当成完整运行显存。量化还要和专家驻留、权重加载、KV cache、通信及 kernel 支持一起核算。

官方部署指南覆盖三条路线：

- vLLM：使用 `kimi_k2` tool-call/reasoning parser；文档给出 nightly wheel，并称 vLLM 0.19.1 已人工验证；
- SGLang：文档给出 `sglang>=0.5.10.post1`，同样需要 `kimi_k2` parser；
- KTransformers：提供 CPU/GPU 异构、RAWINT4 和 LoRA SFT 示例。

部署指南中的 KTransformers+SGLang 示例将 RAWINT4 专家权重和 CPU/GPU 异构执行结合：发布方报告在 8×NVIDIA L20 + 2×Intel 6454S、48 路并发下 prefill 640.12 tokens/s、decode 24.51 tokens/s；命令同时绑定 `--kt-cpuinfer 96`、`--kt-num-gpu-experts 30`、TP=4、prefill 阈值 400 和 `--max-total-tokens 50000` 等配置。另一条 KTransformers+LLaMA-Factory LoRA SFT 示例报告端到端 44.55 tokens/s，硬件为 2×RTX 4090 + Intel 8488C，原文还写 `1.97T RAM` 与 `200G swap`；单位未进一步解释，保留原文，不擅自换算为 GB/TB。两组都是官方指南给出的场景数字，不是本项目复现，也不能互相比较或外推到其他硬件/任务。

这些示例适合面试时追问权重驻留与计算位置、prefill/decode 分账、专家 offload、LoRA 训练数据/序列长度和测量分母。公开指南未同时给出足以独立复现这些数字的完整 workload、batch/序列分布、功耗与计时定义，故性能结论仍按 publisher-reported 保留。

## 3. K2.7 Code 的 Agent 运行时协议

### 3.1 Always-on thinking 与 preserve thinking

官方 API 快速开始明确：Kimi K2.7 Code 只支持思考模式，关闭 thinking 会报错；模型卡也写明会强制 `thinking` 和 `preserve_thinking` 为 true。文档给出的接口约束包括：

- `temperature` 固定为 1.0；
- `top_p` 固定为 0.95；
- `n` 固定为 1；
- `presence_penalty` 和 `frequency_penalty` 固定为 0；
- `tool_choice` 只支持 `auto` 与 `none`；
- 多步工具调用中建议完整回传 assistant message 的 `reasoning_content`。

`preserve_thinking` 的核心不是让应用读取任意隐藏思维，而是把模型协议要求的 reasoning state 连同多轮 assistant message 一起传回，使后续工具调用和问题追问保持连续。面试中应把它表述为“可回放的推理状态协议”，不要把它等同于公开完整内部 chain-of-thought 或模型记忆。

### 3.2 Interleaved thinking 与多步工具循环

模型卡把 K2.7 Code 与 K2 Thinking 放在同一 Interleaved Thinking/Multi-Step Tool Call 路线上；API 快速开始给出了视觉输入、工具 schema、模型 tool call、宿主执行工具、再把 tool result 回传的完整循环。抽象流程是：

```text
user request
  -> reasoning_content + tool_call
  -> host executes tool
  -> tool result
  -> reasoning_content + next tool_call or final answer
```

这意味着 coding agent 的成功率不仅取决于模型，还取决于工具 schema、权限、执行环境、超时、错误回传、上下文截断和最终验证器。Kimi Code CLI 是官方推荐的 agent framework；它是 harness，不是 K2.7 Code 的网络结构名称。

### 3.3 多模态 coding agent

K2.7 Code 支持文本、图像和视频输入，文本输出。官方 API 文档特别说明视频聊天当时只在官方 API 中属于实验能力；图片和视频 token 按视觉内容动态计算，视频由关键帧组成，输入体积还受到限制。一个可迁移的面试问题是：视频理解的 token 预算、抽帧策略、工具结果和代码修改如何共同占用 256K 上下文，而不是只问“是否支持视频”。

## 4. 官方 benchmark 与条件

模型卡的表格由 Moonshot 自报，包含 K2.6、K2.7 Code、GPT-5.5 和 Claude Opus 4.8 的横向数字：

| Benchmark | K2.6 | K2.7 Code | GPT-5.5 | Claude Opus 4.8 |
|---|---:|---:|---:|---:|
| Kimi Code Bench v2 | 50.9 | 62.0 | 69.0 | 67.4 |
| Program Bench | 48.3 | 53.6 | 69.1 | 63.8 |
| MLS Bench Lite | 26.7 | 35.1 | 35.5 | 42.8 |
| Kimi Claw 24/7 Bench | 42.9 | 46.9 | 52.8 | 50.4 |
| MCP Atlas | 69.4 | 76.0 | 79.4 | 81.3 |
| MCP Mark Verified | 72.8 | 81.1 | 92.9 | 76.4 |

脚注给出的条件是：K2.6/K2.7 Code 在 Kimi Code CLI 中启用 thinking，temperature=1.0、top-p=0.95、context=262,144；GPT-5.5 在 Codex `xhigh`，Opus 4.8 在 Claude Code `xhigh`。Program Bench 还绑定无网络、编译后二进制行为测试和 248,000+ fuzz tests；Kimi Claw 24/7 使用 OpenClaw、17 个场景、610 个评价点、3 次运行；MCP 类测试使用 100 tool-call/step budget 和 32K max tokens per step。

这些数字适合用来训练“如何读 benchmark”：先固定模型 snapshot、effort、CLI/harness、工具集合、网络权限、上下文长度、预算、重试、硬件和 verifier，再谈差异。它们不支持“某个模型架构一定更强”的结论。

## 5. Kimi API 周边技术的归属边界

Kimi 官方 API 文档还提供动态加载工具、自动 Context Caching、Partial Mode 和流式断线重连等资料。这些是平台/运行时知识，可以作为面试周边，但不能全部写成 K2.7 Code 的独有新技术：

- 动态加载工具文档明确说明当前仅 `kimi-k3` 支持，不能把它直接归因于 K2.7 Code；
- Context Caching 是 API 层自动前缀缓存，文档称 prompt tokens 大于 256 时才可能命中，缓存命中还依赖固定前缀；
- Partial Mode 可以用 assistant 前缀续写被截断输出，思考模式下需要同时传回 `reasoning_content`；
- 自动断线重连是客户端/harness 策略，通常和 Partial Mode 配合，不是模型训练机制。

面试上最值得组合的问题是：动态工具注入如何保持前缀缓存、工具声明变化如何导致 cache miss、Partial Mode 如何避免重复生成，以及 reasoning state 在重试时怎样防止协议污染。

## 6. 论文、技术报告与负面证据

本轮对 Kimi 官方资源页、官方模型卡、固定配置、官方 GitHub 组织检索结果和定向 arXiv 搜索进行了复核：

- 官方 K2.7 Code 资源页链接到模型/API/代码入口，没有链接专属 arXiv 或技术报告；
- 固定 revision 的 README 公开架构摘要、benchmark、INT4 和部署说明，但没有训练 token、优化器、后训练 recipe 或专属论文链接；
- 定向搜索 `site:arxiv.org "Kimi K2.7 Code"` 未返回 K2.7 Code 专属论文结果，结果主要回到 Hugging Face、Kimi API 和资源页；
- MoonshotAI 官方 GitHub repository API 对 `Kimi-K2.7-Code` 查询返回 `total_count: 0`。这只说明本次查询没有检出同名官方仓库，不证明没有其他实现或后续发布。

因此当前不能把 Kimi Linear、Attention Residuals、Mooncake 或 K2 Thinking 的论文/实现直接改名为 K2.7 Code 的专属训练报告。它们可以作为 Kimi 技术路线的关联背景，但必须单独标注来源和与 K2.7 的证据距离。

## 2026-09-24 两榜与官方 API 快速开始复核

本轮仍沿用榜单中已存在的 Kimi K2.7 Code canonical 条目；没有从 Kimi 文档或 arXiv 新增模型候选。

| 来源 | 当前快照 | 本轮核对 |
|---|---|---|
| Artificial Analysis Kimi K2.7 Code | HTTP 200，4,047,429 bytes，SHA-256 1a8818a6ee812da88ba7b9a95b691ae7481dac7b55d7126084fbd1d5e964e56b | canonical slug kimi-k2-7-code 仍在榜；Intelligence Index 25.8121062401836、256K context 是第三方字段 |
| DataCurve DeepSWE | HTTP 200，268,036 bytes，SHA-256 14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1 | 精确 mini_swe_agent_kimi_k2_7_code_default 行仍在：113 tasks、4 runs、452 attempts、138 passed；Pass@1 30.53%、Pass@4 61.06%、平均成本约 $2.82、平均输出约 59.3K tokens、平均 149.1 steps |
| Kimi 官方资源页 | HTTP 200，270,859 bytes，SHA-256 f3696d9842d595fa8777bfedb7dcd2b9251186ac965fe4e5d0cc0c3ba9a38f51 | 用于核验已发现模型，不产生新候选 |
| Kimi 官方 API 快速开始 HTML | HTTP 200，455,287 bytes，SHA-256 2613331b1ace91d15f9babd118aa854285e523cec4a598fab78dddf94a7530c0 | 与 Markdown 快照交叉检查 |
| Kimi 官方 API 快速开始 Markdown | HTTP 200，12,445 bytes，SHA-256 7c1ecbbfb1d2103772a00ca99b6e058c7a6f018cb00744ae241042f7f8071ec7 | 补充默认 max_tokens 32,768、视觉 token estimate、媒体上传和 highspeed 服务变体 |
| arXiv 精确检索 | HTTP 200，16,442 bytes，SHA-256 a305e3e8bb51f7e9e64e44266a1dc58a7df5c6aae879b818f0c59654186af4fa | 精确查询 “Kimi K2.7 Code” 返回 no results；只说明本次检索，不证明未来不会发布报告 |

本轮官方文档补充：

- 256K 是 context window；API 快速开始另给默认 max_tokens=32,768，两者不是同一限制。
- thinking 默认 enabled，关闭会报错；temperature 固定 1.0、top_p 固定 0.95、n 固定 1、presence/frequency penalty 固定 0；tool_choice 只接受 auto/none。
- 多步工具调用建议保留 assistant 消息中的 reasoning_content；省略通常不报错，但可能降低连续性。它是协议状态提示，不代替工具授权、执行回执、幂等账本或 verifier。
- 图像/视频 token 动态计算；视频由关键帧组成，分辨率和帧数影响预算。文档建议使用 token-estimate 接口；当前 quickstart 还写有 request body 100 MB 限制、图片 URL 不支持（需 base64）、大视频推荐文件上传、图片不超过 4K/视频不超过 1080p。
- 示例由宿主工具用 ffprobe/ffmpeg 按时间片段准备视频，再将视觉结果回灌。这是 API/harness 能力，不代表模型能直接读取本机文件或拥有工具权限。
- 文档新列出 kimi-k2.7-code-highspeed，并明确它与 kimi-k2.7-code 是同一个模型。官方宣称输出速度约为普通版 5–6 倍、常规编程中位输入场景约 180 tokens/s、短上下文约 260 tokens/s，同时注明资源有限、体验可能波动。它只作为已发现 K2.7 Code 的服务/路由变体记录，不另建模型条目，也不把发布方速度当作独立测量。
- 快速开始称 Kimi Code Bench v2、Program-Bench、MLS Bench Lite 相对 K2.6 分别提升 21.8%、11%、31.5%，Agent 类基准约提升 10%；均按发布方对照声明记录，不等于本项目独立复现。2026-09-29 经 7890 重取固定 HF revision `74797c9c62378b951a1f6fcf5c4631024e9b8bef` 的 README（15,061 bytes / `c78bbe5af19636b180eda956abf1f30a79027ec04d6642ebddc44edbc4470879`）与 config（5,420 bytes / `ffbb57bff844e024f6c112640b63da80af228ceb8fb3e2e21908a21febbcdffb`），均与原固定快照逐字节一致；2026-09-24 曾记录的代理超时现已成功复验，不再是当前访问阻塞。部署指南的新鲜快照与异构推理/LoRA SFT 数字见本节。

## 7. 状态与面试映射

Kimi K2.7 Code 当前为 **双榜内容专题闭环**：排行榜身份、官方来源、研究笔记、第二十一册第 94 章和配套面试/练习资料均已同步。没有独立专属训练报告或外部复现；完整权重、目标硬件 profile、真实 API 行为、线上 acceptance 和生产 SLO 仍未确认。闭环表示公开面试知识专题已整理，不是模型或生产验收闭环。

推荐面试主线：

1. 1T/32B active MoE 如何影响路由、通信、显存和吞吐；
2. MLA 与 256K context 如何改变 KV cache 账本；
3. native INT4 为什么不能只用参数量乘 bit 数估显存；
4. always-on thinking、preserve_thinking、reasoning_content 与多步 tool call 如何组成状态协议；
5. Kimi Code CLI、mini-SWE-agent、验证器和模型分数如何拆开归因；
6. multimodal tool result、Partial Mode、context caching 和长周期 coding agent 如何共同管理上下文预算。

研究笔记中的后续锚点曾切换为 Grok 4.5；K2.7 Code 当前的未确认项仍是线上接受率、目标硬件 profiling、完整训练/后训练配方和独立 benchmark。新增正式落点为第二十一册第 94 章。

## 8. 来源清单

- [Artificial Analysis: Kimi K2.7 Code](https://artificialanalysis.ai/models/kimi-k2-7-code)
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/)
- [Kimi K2.7 Code 官方资源页](https://www.kimi.ai/resources/kimi-k2-7-code)
- [Kimi K2.7 Code API 快速开始](https://platform.kimi.com/docs/guide/kimi-k2-7-code-quickstart)
- [Kimi API 工具调用](https://platform.kimi.com/docs/guide/use-kimi-api-to-complete-tool-calls)
- [动态加载工具](https://platform.kimi.com/docs/guide/use-dynamic-tool-loading)
- [Context Caching](https://platform.kimi.com/docs/guide/use-context-caching-feature-of-kimi-api)
- [Partial Mode](https://platform.kimi.com/docs/guide/use-partial-mode)
- [官方 Hugging Face 模型卡](https://huggingface.co/moonshotai/Kimi-K2.7-Code)
- [固定 revision 配置](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/config.json)
- [模型部署指南](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/docs/deploy_guidance.md)
- [arXiv 定向搜索](https://arxiv.org/search/?query=Kimi+K2.7+Code&searchtype=all)

## 9. 2026-09-29 KTransformers 部署复核：版本支持与文档冲突

本轮继续沿两榜已发现的 Kimi K2.7 Code 锚点，没有从 serving 仓库新增模型。固定 HF revision 的 K2.7 部署指南（`3,689` bytes，SHA-256 `b1bc4c5fb7c8b1da727663d4da83999afc902f6166ff862fbaef514b86de2c65`）称 K2.7 Code 与 K2.5/K2.6 架构相同、部署方法可直接复用，并给出 KTransformers + SGLang 的 RAWINT4 命令；指南同时提醒这些只是示例命令，不保证最优。

为核对它引用的实现，本轮经 `10.24.27.134:7890` 读取 KTransformers `main` Atom。feed 的最新提交是 `c40722bf04c494f2492b7eb9e86ef01a4ede45b3`（2026-09-23；feed `21,972` bytes，SHA-256 `04d2f2e3ddfb18a5e03ebb93100569656db843f038eb4cad9eb275d67c0fca78`）。以下 raw 文档均固定在该 commit，HTTP 200，并与同日已取快照逐字节一致：

| 固定文档 | bytes / SHA-256 | 对本锚点的证据 |
|---|---|---|
| `doc/en/Kimi-K2.5.md` | `5,536` / `07f8d0c56ab503235c6b040da1a52a207d1cfb8104845593a3c45f7c49d18204` | 给出 K2.5 的 RAWINT4 CPU/GPU 异构推理示例；不是 K2.7 权重加载测试 |
| `doc/en/kt-kernel/Native-Precision-Tutorial.md` | `9,597` / `aed37372f057ef24d726cfb4c45010721afeb719156bcfc83fc9ab90b64aa93e` | RAWINT4 支持矩阵列 Kimi-K2-Thinking；示例说明当前 `kt-cli` 支持名单也没有列 K2.7/K2.5 |
| `doc/en/kt-kernel/experts-sched-Tutorial.md` | `8,002` / `f7714175e6d3d7f3c64a25a45f66d9533f47252fcffa1f89609ff107da365ced` | `kt-num-gpu-experts` 是每个 MoE layer 的驻留专家数；动态更新需要显式开关 |
| `doc/en/SFT_Installation_Guide_KimiK2.5.md` | `6,858` / `e6d792a8a7383340b24bc3675c7e94c31c29b42588e43a610f532561b511e6e0` | 文档顶部说明其 source-install 正文属于旧版本，`0.7.0.post4` 应使用下列 release tutorial |
| `.github/release/examples/kimi-k25/README.md` | `19,251` / `c3345dccdccf43c3df8ad708a0e0644500bf00368aaef88ca17b85eb0df75d02` | KTransformers `0.7.0.post4` K2.5 LoRA recipe：使用原始完整权重、不转全 BF16；不是 K2.7 recipe |
| `.github/release/examples/kimi-k25/README_EN.md` | `20,724` / `8c2f5a5c5c05bf3313a3da5caaee3af8fa0a14574b7c672979387089105b1ff1` | 英文版与中文版描述一致 |
| `.github/release/examples/kimi-k25/train-neko.yaml` | `1,660` / `a42b713c57790afc192660e7419533d81781400f3f0bfa9fe9bb6b0639e7529a` | 同时列 `kt_backend: RAWINT4`、`kt_expert_weight_format: rawint4` 与 `bf16: true`；权重格式和训练精度是不同配置维度 |

因此 RAWINT4 结论必须保持为**文档未对齐、K2.7 兼容性未实测**：Kimi 部署指南提供可执行命令，KTransformers 的同提交通用支持矩阵没有列 K2.7；K2.5 专页是相邻型号的部署依据，不等于 K2.7 的独立兼容证明。命令给出 `--kt-cpuinfer 96`、`--kt-threadpool-count 2`、`--kt-num-gpu-experts 30`、`--kt-gpu-prefill-token-threshold 400`、TP=4、RAWINT4；通用教程建议 CPU infer 线程约为物理核的 90%、thread pool 数按 NUMA 节点数设置、GPU experts 按每个 MoE layer 计。K2.7 示例没有传 `--kt-enable-dynamic-expert-update`，故不能把它描述成已启用动态专家重排。

Native Precision 教程把 dual prefill 写为：输入 token 数 `< threshold` 使用 CPU-GPU hybrid，`>= threshold` 使用 layerwise prefill，并将 CPU 权重传到 GPU；后者会增加显存压力。K2.7 命令设置阈值 400，但这是通用教程描述与发布方配置，不是本机已验证的 K2.7 分支边界或性能 profile。

SFT 文档存在版本差异：旧的 K2.5 source-install guide 顶部明确说 `0.7.0.post4` 应使用 release tutorial，后续 BF16 转换步骤不应冒充当前发行配方。固定提交里的 `0.7.0.post4` K2.5 release tutorial 使用固定模型 revision `54383e83fa343a1331754112fb9e3410c55efa2f`，要求完整原始模型、不转成全 BF16；随附 YAML 同时设置 `kt_backend: RAWINT4`、`kt_expert_weight_format: rawint4` 和 `bf16: true`。它说明量化权重后端与训练精度字段可以并存，不等于把基座权重转换成 BF16；文档没有逐算子解释所有 dtype。该 K2.5 配方是 NekoQA 风格微调，参考训练配置为 8×RTX 5090、LoRA rank 8/alpha 16、最大序列长度 4096、每卡 batch 1、梯度累积 8，并训练 attention 与 fused expert 两类 LoRA。checkpoint 分开保存普通与 expert adapter、优化器等状态，再转换成 SGLang adapter；教程还用 LoRA tensor、optimizer、RNG、scheduler 和恢复后的 loss 核对 resume。它们是 K2.5 发行版发布的流程/验收说明，不是本地复现，也不是 K2.7 配方。K2.5 的训练与 K2.7 部署指南 2×RTX 4090 的 SFT 数字属于不同 recipe/硬件口径，不能横向比较。

K2.7 部署指南报告一条 KTransformers + LLaMA-Factory LoRA SFT 命令和 2×RTX 4090 场景吞吐，但没有给 K2.7 专属训练 YAML、权重 revision 或精度/转换说明。K2.5 release tutorial 是相邻型号的积极证据，仍不足以证明 K2.7 对应 checkpoint 可按该路径训练；完整权重加载、目标环境版本、数值正确性和本地吞吐仍未验证。

本轮代理结果按端点区分：Google 首页先返回 HTTP 302（372 bytes），跟随跳转的定向搜索 HTTP 200（91,749 bytes）；GitHub commit 页面 HTTP 200（283,724 bytes），固定 raw 文档 HTTP 200。GitHub API tree 返回 HTTP 403 rate-limit，两个初始猜测的 raw 路径返回 404 后改用 README 中的实际路径成功取得文件。API 限流与路径 404 都不是代理断网证据。

- [KTransformers 固定提交 `c40722b`](https://github.com/kvcache-ai/ktransformers/commit/c40722bf04c494f2492b7eb9e86ef01a4ede45b3)
- [K2.5 serving guide at fixed commit](https://github.com/kvcache-ai/ktransformers/blob/c40722bf04c494f2492b7eb9e86ef01a4ede45b3/doc/en/Kimi-K2.5.md)
- [Native Precision tutorial at fixed commit](https://github.com/kvcache-ai/ktransformers/blob/c40722bf04c494f2492b7eb9e86ef01a4ede45b3/doc/en/kt-kernel/Native-Precision-Tutorial.md)
- [Expert scheduling tutorial at fixed commit](https://github.com/kvcache-ai/ktransformers/blob/c40722bf04c494f2492b7eb9e86ef01a4ede45b3/doc/en/kt-kernel/experts-sched-Tutorial.md)
- [K2.5 SFT tutorial at fixed commit](https://github.com/kvcache-ai/ktransformers/blob/c40722bf04c494f2492b7eb9e86ef01a4ede45b3/doc/en/SFT_Installation_Guide_KimiK2.5.md)
- [K2.5 `0.7.0.post4` release SFT tutorial at fixed commit](https://github.com/kvcache-ai/ktransformers/blob/c40722bf04c494f2492b7eb9e86ef01a4ede45b3/.github/release/examples/kimi-k25/README.md)
- [K2.5 release training YAML at fixed commit](https://github.com/kvcache-ai/ktransformers/blob/c40722bf04c494f2492b7eb9e86ef01a4ede45b3/.github/release/examples/kimi-k25/train-neko.yaml)
