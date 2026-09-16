# Kimi K2.7 Code：长周期编码 Agent、MoE/MLA 与思考状态协议

核验日期：2026-09-15。本笔记只把 Artificial Analysis 与 DataCurve DeepSWE 的条目作为 Kimi K2.7 Code 的模型锚点，再沿 Moonshot/Kimi 官方资源页、API 文档、Hugging Face 模型卡、配置和部署指南提取面试相关知识。排行榜结果、官方模型卡 benchmark 和产品运行时字段分别承担不同证据责任，不能互相替代。

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
- 部署指南：[docs/deploy_guidance.md](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/docs/deploy_guidance.md)。

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

部署指南中的吞吐数字绑定 8 张 L20、Intel 6454S、并发和具体 KTransformers 参数，是发布方环境示例，不是普遍性能承诺。

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

## 7. 状态与面试映射

Kimi K2.7 Code 当前为 **资料级闭环**：两个排行榜中的可追溯条目、官方资源/API/模型卡/配置/部署资料、研究笔记和进度同步均已具备；没有独立专属训练报告或外部复现，因此暂不新增 Kimi K2.7 Code 专属正式章节。

推荐面试主线：

1. 1T/32B active MoE 如何影响路由、通信、显存和吞吐；
2. MLA 与 256K context 如何改变 KV cache 账本；
3. native INT4 为什么不能只用参数量乘 bit 数估显存；
4. always-on thinking、preserve_thinking、reasoning_content 与多步 tool call 如何组成状态协议；
5. Kimi Code CLI、mini-SWE-agent、验证器和模型分数如何拆开归因；
6. multimodal tool result、Partial Mode、context caching 和长周期 coding agent 如何共同管理上下文预算。

后续锚点切换为 `grok-4.5`。Kimi K2.7 Code 仍可补证线上接受率、目标硬件 profiling、完整训练/后训练配方和独立 benchmark，但这些缺口不阻止当前资料级闭环。

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

