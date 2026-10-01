# 论文路线

## Gemini 4 Argon（官方发布资料，非论文）

[Google The Keyword: Introducing Gemini 4 Argon](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-4-argon/)。记录 1M 输出预算、长程 coding/enterprise/cyber 定位、发布方评测和分阶段安全措施。Google 未公开 Argon 专属技术报告或完整训练 recipe；文章中的评测均按发布方结果处理，不当作独立复现。

## GPT-6.1 Sol（2026-10-01）

暂无 GPT-6.1 Sol 专属公开论文或技术报告。本轮使用 OpenAI 官方模型页、Reasoning、Agents、Compaction 文档记录 API/runtime 合同；不把文档当作内部架构或训练论文。

## 入门必读

1. Attention Is All You Need
2. GPT 系列论文
3. BERT
4. InstructGPT
5. Scaling Laws for Neural Language Models
6. Chinchilla
7. LoRA
8. Direct Preference Optimization
9. FlashAttention
10. CLIP
11. PagedAttention / vLLM 相关论文与技术报告
12. Mamba / Selective State Space Models
13. Constitutional AI / RLAIF 相关论文
14. ReAct / Toolformer / Function Calling 与 Agent 工具使用论文线
15. SimPO / ORPO / KTO 等 reference-free 或轻量偏好优化论文线
16. DeepSeek-R1 / GRPO / RLVR / DAPO 等 reasoning 后训练论文线
17. vAttention / FlashInfer 等 inference serving 内存与 kernel 论文线
18. GraphRAG 等结构化 RAG 论文线

## 核心方向

### 架构

Transformer、GPT、LLaMA、Mistral、MoE、SSM、S4、Mamba、RWKV、RetNet、Hyena、Linear Attention、Attention + SSM 混合架构。

建议补读论文线：

1. Efficiently Modeling Long Sequences with Structured State Spaces / S4。
2. Mamba: Linear-Time Sequence Modeling with Selective State Spaces。
3. RWKV: Reinventing RNNs for the Transformer Era。
4. Retentive Network / RetNet。
5. Hyena Hierarchy。
6. Linear Transformers / Performer 等线性注意力路线。

### 训练

Scaling Law、Chinchilla、数据配比、分布式训练、训练稳定性。

### 对齐

InstructGPT、RLHF、DPO、Constitutional AI、RLAIF、KTO、ORPO。

建议补读论文线：

1. Direct Preference Optimization / DPO。
2. KTO、ORPO、SimPO 等不显式训练 reward model 或不依赖 reference model 的偏好优化路线。
3. DeepSeekMath / GRPO、DeepSeek-R1 和 DAPO 等 reasoning RL 与 RLVR 路线。
4. Constitutional AI / RLAIF 与 AI feedback 相关路线。
5. Reward overoptimization、reward hacking、scalable oversight 与偏好评估校准相关论文。

### 推理优化

FlashAttention、PagedAttention、Speculative Decoding、量化论文。

建议补读论文线：

1. FlashAttention / FlashAttention-2 / FlashAttention-3。
2. PagedAttention and vLLM。
3. vAttention：把动态 KV 显存管理放到虚拟内存 / 按需映射视角下理解 PagedAttention 的替代路线。
4. FlashInfer：从 serving kernel、batching、attention backend 和框架集成角度理解推理性能优化。
5. Speculative Decoding / Speculative Sampling。
6. Medusa、EAGLE 等多 token 预测和推测解码路线。
7. GPTQ、AWQ、SmoothQuant、KV Cache Quantization 等量化路线。
8. [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)：target hidden feature 的跨层融合、逐层 K/V 注入、block diffusion draft 与 target verification。

### Agent、工具协议与 Coding Agent

ReAct、Toolformer、function calling、MCP、A2A、agent evaluation、coding agent runtime、sandbox、trace/replay 和工具安全。

建议补读资料线：

1. ReAct: Synergizing Reasoning and Acting in Language Models。
2. Toolformer: Language Models Can Teach Themselves to Use Tools。
3. OpenAI function calling / tools 官方文档。
4. Model Context Protocol 官方规范和 SDK 文档。
5. Agent benchmark、tool-use benchmark 和 SWE-bench 相关资料。
6. GraphRAG / RAPTOR / Agentic RAG 等结构化、多跳和多轮检索增强路线。

### AI Infra 与 Serving Engine

GPU 集群、分布式训练系统、LLM serving engine、调度、KV Cache 管理、observability、成本治理和平台工程。

建议补读资料线：

1. Megatron-LM、DeepSpeed、FSDP、Ray 等训练系统资料。
2. vLLM、SGLang、TensorRT-LLM、TGI 等推理框架文档和源码导读。
3. Kubernetes GPU 调度、NVIDIA GPU operator、MIG、NCCL 和集群网络资料。
4. LLMOps、模型仓库、评估平台、实验追踪和 artifact 管理资料。

### 多模态

CLIP、Flamingo、BLIP、LLaVA、Stable Diffusion、DALL·E、Whisper、视频生成路线。

### Safety 与 Interpretability

Red Teaming、Mechanistic Interpretability、SAE、Model Editing、Unlearning、Privacy。

## 2026-08 Frontier Model Release 与技术报告入口

以下资料用于模型卡、产品接口和架构信号核对；它们不是同一口径的 benchmark，引用时应保留来源类型和访问日期。

### 官方模型目录与产品文档

1. [OpenAI Models](https://developers.openai.com/api/docs/models)：GPT-5.5、GPT-5.6 系列、上下文/输出、工具与 reasoning 控制入口。
2. [Anthropic Models Overview](https://docs.anthropic.com/en/docs/about-claude/models/overview)：Claude 系列、长上下文和 thinking/Agent 能力说明。
3. [Gemini API Models](https://ai.google.dev/gemini-api/docs/models)：Gemini 模型、thinking levels、多模态和工具能力入口。
4. [Qwen3.8-Max Product Page](https://www.qwencloud.com/models/qwen3.8-max)：一方产品页；参数、1M context 和 Agent 宣称需标注产品页/待核验。
5. [GLM Model Guide](https://docs.bigmodel.cn/cn/guide/models)：GLM 已发布模型目录；GLM-5.5 若无可核验 ID/模型卡，应留在观察项。

### 模型卡与开源实现

1. [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)：CSA/HCA、1M context、参数和 reasoning 接口核对入口。
2. [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)：Flash 正式版本与 Agent post-training 变化入口。
3. [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)：Gated DeltaNet/Gated Attention、原生多模态和长上下文。
4. [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)：hybrid attention、长上下文和 speculative decoding 入口。
5. [Qwen-AgentWorld-35B-A3B](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B)：world model/Agent environment 训练和评测边界。
6. [Kimi K3 官方发布文章](https://www.kimi.com/en/blog/kimi-k3)：KDA、Gated MLA、规模与长上下文的发布披露入口；NoPE、层比例和 total/active 参数已由 K3 技术报告与固定 config 补证，serving/kernel 仍须看实现资料。
7. [Mistral Small 4](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603)：MoE、EAGLE、NVFP4 和统一 Instruct/Reasoning/Devstral 路线。
8. [Leanstral 1.5](https://huggingface.co/mistralai/Leanstral-1.5-119B-A6B)：Lean 4 形式化证明 coding agent。
9. [Shieldstral 1.0](https://huggingface.co/mistralai/Shieldstral-1.0-3B)：policy-adaptive multimodal safety classifier。
10. [Step 3.7 Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash)：MoE、视觉 encoder、reasoning levels、MTP/EAGLE 与低精度入口。
11. [North Mini Code](https://huggingface.co/CohereLabs/North-Mini-Code-1.0)：sliding-window RoPE/global NoPE、MoE 和 coding agent。

### 本轮新增章节的证据使用规则

1. 模型名、参数、上下文、架构字段和 API 能力优先引用官方模型卡、官方模型目录或官方开发者文档；模型卡能证明公开字段，不能证明未披露的训练细节。
2. Qwen3.8-Max 只使用 QwenCloud 一方产品页作为产品信号；GLM 模型目录当前可核对到 GLM-5.2，GLM-5.5 继续列为待核验观察项。
3. `CSA/HCA`、`KDA`、`Gated DeltaNet`、`Gated MLA`、`p-RoPE`、`MetaP`、`NoPE` 等章节把模型卡事实和教学公式分开；公式用于理解机制，不冒充厂商源码。
4. `AgentWorld`、`Shieldstral` 和 DeepSeek-V4 的论文链接用于核对任务闭环、安全分类器和技术路线；产品页、模型卡、论文和引擎文档的证据等级不混用。

### 论文与技术报告补充入口

1. [DeepSeek-V4 Technical Report](https://arxiv.org/abs/2606.19348)：CSA/HCA、百万 token 上下文、训练与推理路线。
2. [Qwen-AgentWorld](https://arxiv.org/abs/2606.24597)：模型、环境、工具和 verifier 的闭环评测。
3. [Shieldstral](https://arxiv.org/abs/2607.25857)：policy-adaptive multimodal safety classifier。
4. [EAGLE](https://arxiv.org/abs/2401.15077) 与 [EAGLE-3](https://arxiv.org/abs/2503.01840)：推测解码 draft head 路线。
5. [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)：推理预算、候选和验证成本的实验基础。

### 2026-09 新模型架构专题

1. [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)：Kimi Delta Attention、门控线性状态、DPLR 简化与混合 MLA。
2. [Attention Residuals](https://arxiv.org/abs/2603.15031)：沿深度选择历史表示、Block AttnRes、流水线通信与两阶段推理。
3. [DeepSeek-V4 Technical Report](https://arxiv.org/abs/2606.19348)：CSA/HCA、mHC、Muon、FP4/FP8、GRPO 与 on-policy distillation。
4. [Step 3.5 Flash Technical Report](https://arxiv.org/abs/2602.10604)：稀疏 MoE、滑动窗口/全注意力混合与 MTP-3；模型卡中的部署和评测字段仍需绑定 revision、硬件与后端。
5. [xAI Models](https://docs.x.ai/developers/models)：Grok 4.6 的 500K context、reasoning effort、结构化输出和工具接口；这是官方模型文档，不是架构论文。

6. [Anthropic Models Overview](https://platform.claude.com/docs/en/models/overview)：Claude Opus 5 的模型 ID、1M context、最大输出、adaptive thinking、平台与价格字段；这是官方模型目录，不是架构论文。
7. [Claude Opus 5 system card](https://www.anthropic.com/claude-opus-5-system-card)：官方安全/系统资料入口；本轮仅记录入口，不把未读取的完整内容扩写成已核验训练或架构事实。
8. [Claude Opus 5 Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)、[Effort](https://platform.claude.com/docs/en/build-with-claude/effort) 与 [Refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback)：官方运行时资料，支持 thinking block/signature 回放、`display: "omitted"`、effort 能力边界、refusal/fallback 和 fallback credit 的面试讨论；不是内部架构或训练报告。
9. [Claude Fable 5.1 overview](https://platform.claude.com/docs/en/models/fable-5-1/overview)：官方模型页，记录 1M context、adaptive always-on thinking、preserved thinking、thinking block 兼容/历史编辑失效、forced tool capability error、per-message effort、turn-scoped system messages、`display: "updates"` 和 content provenance；不是架构论文。
10. [Claude Sonnet 5 overview](https://platform.claude.com/docs/en/models/sonnet-5/overview)、[system card](https://www.anthropic.com/claude-sonnet-5-system-card)：官方模型页和 System Card，记录 1M context、128K/300K 输出、Adaptive、平台与成本字段，以及 2026-09-21 深读的训练公开性、安全和评测边界；仍不把未公开训练或架构细节写成事实。
11. [Claude Haiku 4.5 overview](https://platform.claude.com/docs/en/models/haiku-4-5/overview)、[system card](https://www.anthropic.com/claude-haiku-4-5-system-card)：官方模型页和系统资料入口，记录 200K context、64K 输出、extended thinking、fastest 延迟字段、平台与成本字段；本轮尚未读取 system card 全文，不把未核验训练或架构细节写成事实。
12. [DeepSeek-R1-0528 Release](https://api-docs.deepseek.com/news/news250528)：官方 API 发布页，记录 2025/05/28、JSON/function calling、API 兼容性承诺和开源权重入口；benchmark 图片、模型卡、架构与训练细节仍待核验。

### GPT-6 Astra 官方接口资料

1. [GPT-6 Astra model page](https://developers.openai.com/api/docs/models/gpt-6-astra.md)：`gpt-6-astra` 的输入/输出模态、context window、maximum input/output、reasoning effort、Responses 工具与端点支持。
2. [Using GPT-6 Astra](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra.md)、[Async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling.md)、[Mid-turn steering](https://developers.openai.com/api/docs/guides/steering.md) 和 [Misalignment monitoring](https://developers.openai.com/api/docs/guides/safety-checks/misalignment-monitoring.md)：模型指南、应用侧异步工具、WebSocket 中途约束和平台安全控制面。
3. [Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md)、[Conversation state](https://developers.openai.com/api/docs/guides/conversation-state.md)：`configuration_update`、reasoning/`phase` item replay、`previous_response_id` 和 reasoning/visible output 的容量边界；`phase` 专节当前以 GPT-5.5/GPT-5.4 为示例，不作为 GPT-6 专属能力。
4. [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md)、[Compaction](https://developers.openai.com/api/docs/guides/compaction.md)、[Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search.md) 和 [Rethinking skills and prompts](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)：KV state 前缀缓存、deferred schema、hosted/client tool search、canonical compaction context 以及技能/`AGENTS.md`/任务提示的上下文治理。
5. 本资料只证明官方公开的接口、状态协议、Agent 工程建议和容量字段；参数规模、训练架构、数据、发布日期和技术报告仍需独立的一手来源，不从榜单条目反推。

### GLM-5.3 官方文档与长任务环境

1. [GLM-5.3 Guide](https://docs.z.ai/guides/llm/glm-5.3)：模型版本关系、1M context、128K output、推理档位、可执行环境与验证器流程描述。
2. [GLM-5.2 Guide](https://docs.z.ai/guides/llm/glm-5.2)：长任务定位和前代接口背景；正文没有展开 SAO/compaction 的数学定义，具体算法由下列 SAO 论文补证。
3. [GLM-5.2 官方博客](https://z.ai/blog/glm-5.2)：补充 IndexShare、MTP speculative decoding、长上下文 serving、critic-based PPO 与 coding-agent anti-hack；正文资源和哈希见研究笔记。
4. [SAO: Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508)：公开 DIS、single-rollout、value-model、双侧 token clipping 和 Skip-Observation GAE；摘要称部署到 GLM-5.2（750B-A40B）Agent RL pipeline。
5. 证据边界：上述资料支持接口、高层训练流程和 SAO 公开算法，不支持参数量、完整训练配方、GLM-5.3 专属 compaction 序列化或把论文实验改写为 GLM-5.3 独有结果。

### GLM-5.3 官方博客正文与评测脚注

1. [GLM-5.3 官方博客](https://z.ai/blog/glm-5.3)：同一 GLM-5.2 基座上的 post-training scaling、Z.ai Code Bench、可执行长周期环境、reference-free verifier、reward-shortcut audit、CyberGym/ExploitBench/ExploitGym 和合作代码库漏洞统计。
2. [GLM-5.3 正文资源](https://z.ai/blog/assets/glm-5.3-BIDw01m9.js) 与 [source bundle](https://z.ai/blog/assets/src-GO5ZQO2t.js)：2026-09-21 固定正文资源，哈希分别为 f809bf586f74b40a01ecd8d674f9ec3de469a5abeae125e2a849c9805605d0f3 和 e431f2c5e2590dd61672b259c32f49a81d6a0a0304e1311ea29b4b5a1e7b7d69。
3. 评测脚注：DeepSWE、Terminal-Bench 3.0、ALE、CyberGym、ExploitGym、ExploitBench 的 harness、temperature/top-p、上下文、超时、隔离、域名白名单和 verifier 条件。

这些材料是官方发布方资料和页面脚注，不是 GLM-5.3 独立技术报告。50% 提升、84.5%、54.4%、105/130 及 2,436/269/1,097 等数字必须连同任务定义和运行条件引用，不能与 DataCurve 或 Artificial Analysis 拼成裸模型结论。

### Kimi K3 发布与长任务 Harness

1. [Kimi K3 官方发布文章](https://www.kimi.com/en/blog/kimi-k3)：KDA、AttnRes、Stable LatentMoE、量化、视觉、长上下文、Agent 限制和评测脚注入口。
2. K3 文章中的技术和 benchmark 需绑定发布时的模型、harness、硬件、effort 与 fallback；Kimi Linear、Attention Residuals 论文只用于解释一般机制，不证明 K3 的完整配置。

### DeepSWE v1.1 评测快照

1. [DeepSWE](https://deepswe.datacurve.ai/)：长周期软件工程 benchmark 入口；本项目保存了页面更新时间为 2026-09-03 的 `/tmp/deepswe.html` 快照。
2. [`deepswe-snapshot-notes.md`](research/model-update-2026-09/deepswe-snapshot-notes.md)：整理 113 个任务、91 个仓库、5 种语言、统一 `mini-swe-agent` harness，以及 21 个配置的 Pass@1、区间、成本、输出 token 和 Agent steps。
3. 证据边界：这些是模型 revision + effort + Agent harness + 工具 + verifier + 运行策略的组合结果；不能直接与 Artificial Analysis、SWE-bench 或其他 harness 的结果拼接，也不能反推参数量、架构或训练方法。

### DeepSeek V4.1-Flash

1. [DeepSeek-V4.1-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)：固定 revision `dba1be0a40aa45a94ad051997016db3960a90277` 的模型卡，披露 CED、CSA2、FP4 KV、Engram、DSpark、原生多模态、训练规模和 Agent 评测协议；它是模型卡，不等于独立论文复现。
2. [DeepSeek-V4.1-Flash Technical Report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)：官方 51 页技术报告；本轮已逐页提取并核对 CED、CSA2、HSI、mHC、Engram、DSpark、FP4 KV、训练基础设施、SWA replay 和 Agent 评测边界。完整 production kernel、线上接受率和独立 profiling 仍待核验；仓库 reference kernel 另按源码 artifact 记录。
3. [DeepSeek-V4.1-Flash API Release](https://api-docs.deepseek.com/news/news260910)：API alias、兼容路由和服务端发布日期入口；路由说明带日期，不能替代权重身份。
4. [DeepSeek-V4.1-Flash fixed-revision inference tree](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277/inference)：`model.py`、TileLang `kernel.py`、权重转换、Engram/vision 和 DSpark forward path 的 reference implementation；官方 README 明确它不是生产 serving engine，`generate.py` 仍走普通自回归生成。
5. [DeepSeek `deepseek-recipe` pinned commit](https://github.com/deepseek-ai/deepseek-recipe/tree/8cadfede7063c896b944e7bae05daa3549ae97ea)：固定源码归档 SHA-256 `1116ca33e9dc62a913fb9214578c400f1704e6bca33487f4a4c31b32c67a21a6`；记录协议转换、V4.1 encoding/tokenizer、增量流式 parser、图像 quota 和 mock server 接线。它不是推理后端、工具执行器或 verifier。
6. [vLLM v0.30.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0) 与 [stable registry](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/model_executor/models/registry.py)：tag commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，registry 已登记 `DeepseekV41ForCausalLM`/`DSparkV41DraftModel`；这是 V4.1 stable release/source evidence，不是完整权重、硬件 profiling 或生产 acceptance。
7. [PyPI vLLM 0.30.0 metadata](https://pypi.org/pypi/vllm/0.30.0/json)：固定 release artifact、wheel 和 sdist 的大小/SHA-256；它证明可分发包存在，不证明当前环境安装成功、target/draft verify、FP4 质量或线上 SLO。
8. [DeepSeek Harness quickstart](https://deepseek-harness.github.io/deepseek-harness/en/guide/quickstart)、[providers](https://deepseek-harness.github.io/deepseek-harness/en/guide/providers)、[architecture/reference](https://deepseek-harness.github.io/deepseek-harness/en/reference/)、[MCP memory](https://deepseek-harness.github.io/deepseek-harness/en/guide/mcp-memory) 和 [GitHub review](https://deepseek-harness.github.io/deepseek-harness/en/guide/github-review)：官方 Preview runtime 文档，覆盖 provider/session/plugin/MCP/webhook 合同；它不是 V4.1 技术报告或内部架构证据。
9. [SGLang v0.5.20 release notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.20) 与 [PR #39116](https://github.com/sgl-project/sglang/pull/39116)、[#38192](https://github.com/sgl-project/sglang/pull/38192)、[#37764](https://github.com/sgl-project/sglang/pull/37764)：DeepSeek-V4 AMD/HIP 路径中的 DSpark graph replay、unified-KV/SWA ring 容量记账、FP4 indexer schedule fusion。发布数字绑定指定实现/负载，不是模型能力或独立复现；具体边界见 V4.1 研究笔记。

该协议 artifact 的面试价值在于把 `model -> protocol -> transport -> executor -> verifier` 分层。README 列出的 `logprobs`、server-side web search、JSON Schema/strict enforcement、`n>1`、Responses context storage 和 encrypted thinking 未支持项属于该适配器 commit 的能力边界，不能改写成 V4.1 模型能力的负面证明；图像 fetcher 的 SSRF 警告也必须保留。

### K2 Horizon MoVA 与 Uno

1. [K2-Horizon-MoVA-36B-A4B model card](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B)：固定 revision 下核对 MoVA value routing、稀疏 MoE、GQA、长上下文、训练阶段和部署字段。
2. [K2 Horizon modeling implementation](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/blob/de2d2efb32ed7639b7140bccbefe131a0063a982/modeling_k2_horizon.py)：核对 value router 的 top-k 选择语义、attention gate 和专家执行路径；源码阅读不能替代完整训练报告。
3. [Uno paper, arXiv:2609.04010](https://arxiv.org/abs/2609.04010)：学习冻结 K2 7B AR base、LoRA diffusion path、Diffusion Distillation 和 `Psi-Spec` rejection verification；Uno 是关联 adapter/论文技术，不是排行榜新增模型。

### Qwen3.8 官方模型卡与 Flash-Next 技术报告

1. [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)：核对 27B dense native vision-language、64 层、GDN/Gated Attention、MTP 和 thinking 控制。
2. [Qwen3.8-2.4T-A95B model card](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)：核对 2.4T total/95B active、92 层、512 experts、text-only 和强制 thinking。
3. [Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)：核对 125B/6B active、约 51B N-gram、4B MTP、GDN + QSA、四分支 Gated Residual 和视觉字段。
4. [Qwen3.8-Flash-Next GitHub](https://github.com/QwenLM/Qwen3.8-Flash-Next)：实验性架构预览、实现入口和技术报告索引。
5. [Flash-Next 技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)：QSA 的 micro-block indexer/两阶段训练、GR、N-gram host-memory prefetch、Muon/AdamW 分工与评测设置。
6. [vLLM `v0.30.0` Qwen4Exp](https://github.com/vllm-project/vllm/tree/v0.30.0/vllm/models/qwen4_exp) 与 [model registry](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/registry.py)：固定版本 QSA、PLE、Gated Residual、MTP 实现和测试入口；本项目只做静态源码核对。
7. [SGLang `v0.5.20` model sources](https://github.com/sgl-project/sglang/tree/v0.5.20/python/sglang/srt/models) 与 [vLLM Qwen recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)：用于核对 indexer 调度、PLE offload 与特定部署条件；性能/容量数据属于来源方条件，不是本地验收。
8. [Artificial Analysis Qwen3.8 条目](https://artificialanalysis.ai/models/qwen3-8-flash-next)：候选发现来源；[DataCurve DeepSWE](https://deepswe.datacurve.ai/) 当前快照只检出 `qwen3.8-max`。

模型卡和报告可证明公开字段与作者披露的技术路线，不能自动证明完整生产 kernel、目标硬件 profiling、线上 acceptance rate、全系列训练配方或独立 benchmark。Flash-Next 报告中的速度、loss、稳定性和 benchmark 数字必须保留为发布方自报；Qwen3.8-Max 是官方说明基于 A95B 的 hosted version，不作为新的 open checkpoint。

### GLM-5.3-Flash 官方资料与混合 Serving

1. [GLM-5.3-Flash 官方模型文档](https://docs.z.ai/guides/vlm/glm-5.3-flash)：输入模态、thinking/tool streaming、模型定位和接入边界。
2. [GLM-5.3-Flash 官方博客](https://z.ai/blog/glm-5.3-flash)：hybrid linear+sparse attention、IndexPool、视觉 coding loop、SGLang/ReplaySSM、混合 cache 和 EPD 的发布方描述。
3. [GLM-5.3-Flash 模型卡](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a) 与 [固定配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json)：`320B/18B`、45 层、`34/11` layer types、MoE、IndexPool、mHC 和 vision 字段。
4. [Artificial Analysis 条目](https://artificialanalysis.ai/models/glm-5-3-flash) 与 [DataCurve DeepSWE](https://deepswe.datacurve.ai/)：分别提供 `max` 配置字段和 `mini-swe-agent` Agent 结果；两者不是同一 benchmark，也不是裸模型分数。
5. [SGLang GLM-5.3-Flash cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash.md)：2026-09-21 页面，公开 paged KV/KDA state 双 pool、MTP、KV/DSA backend pairing、视频/EPD 与 PD dummy-weight/数值门禁。
6. [vLLM GLM-5.3-Flash recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)：v0.29.0+、native FP8/MTP、hybrid KDA+sparse MLA 和硬件/依赖路线；recipe 声明不等于本机权重加载或 profiling。
7. [Transformers GLM5-Next 文档](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/glm5_next.md)：基础接入文档明确不包含 MTP layer，用于区分 checkpoint/framework 与 serving runtime 能力。

8. [SGLang `v0.5.20` GLM5Next source](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/models/glm5_next.py) 与 [SGLang main source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/glm5_next.py)：分别作为固定 stable entry 和 mutable upstream 演进证据。
9. [vLLM v0.30.0 GLM5Next fixed tree](https://github.com/vllm-project/vllm/tree/v0.30.0/vllm/models/glm5next)、[fixed main commit](https://github.com/vllm-project/vllm/tree/81d7293c2167e39f3ffddc9a82d633f94e8a1eaa/vllm/models/glm5next) 与 [v0.29.0 tree](https://github.com/vllm-project/vllm/tree/v0.29.0)：stable source 已逐文件确认 cache/KDA/MTP/multimodal 路径；attention/indexer、KPool stride-aware tail kernel、Quark weight-loading 与 fixed main 有差异，具体 blob 和结论见 [研究笔记](research/model-update-2026-09/glm-5.3-flash-source-notes.md)。源码进入 stable 不等于 wheel、权重、硬件或生产 acceptance 通过。

这些资料中没有公开 GLM-5.3-Flash 的完整训练报告或生产 kernel。`3.01x/4.44x` attention/KV、约 `3x` serving 和视觉 self-verification 必须标作发布方自报；研究笔记与正式章节见 [`glm-5.3-flash-source-notes.md`](research/model-update-2026-09/glm-5.3-flash-source-notes.md) 和 [`第二十一册第 84 章`](book-21-transformer-architecture-evolution/chapters/84-glm-5.3-flash混合注意力与视觉闭环.md)。

### DeepSeek V3.2 官方资料与工具推理

1. [Artificial Analysis DeepSeek V3.2](https://artificialanalysis.ai/models/deepseek-v3-2)：本轮模型发现锚点，当前页面为 `Non-reasoning` 配置；Artificial Analysis 的指数、参数和上下文字段只作为第三方榜单/目录字段。
2. [DeepSeek V3.2 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)：官方资料明确列出 DeepSeek Sparse Attention、scalable RL framework、large-scale agentic task synthesis pipeline，以及 `thinking with tools` 和新的工具调用模板。
3. [DeepSeek V3.2 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf)：官方报告入口；已核对 DSA 两阶段训练、2,048 KV top-k、约 2.1B/943.7B tokens、GRPO 稳定化和 Agent 任务合成规模。Figures 1–7 已从 arXiv v1 单独取图/面板并视觉核验；arXiv v1 HTML 中全部编号公式 Eq. (1)–(9) 的 TeX annotations 已文本核对但未在 PDF 页面视觉核验；未编号行内数学表达式未做穷尽审计。
4. [DeepSeek V3.2-Exp 仓库](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)：模型卡指向的本地运行与结构入口；`V3.2-Exp`、`V3.2-Speciale` 是关联版本/变体，不作为新的排行榜基础模型。

DataCurve 当前快照没有精确的 `mini_swe_agent_deepseek_v3_2_*` 行，所以不迁移其他 DeepSeek 版本的 Pass@1、成本或 Agent steps。DSA 的完整 indexer 训练目标、kernel、KV 字节账本、scalable RL 的完整 recipe、任务合成过滤器和线上 tool-call acceptance rate 均待核验。

#### 2026-09-20 官方实现与 serving 补证

5. [V3.2-Exp inference demo](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference)：参考 `model.py`/`kernel.py` 展示 FP8 indexer、non-interleaved indexer RoPE、`fp8_index`、causal/top-k、prefill MHA、decode MQA、latent KV/position cache 和 FP8 KV cache。
6. [DeepGEMM PR #200](https://github.com/deepseek-ai/DeepGEMM/pull/200) 与 [FlashMLA PR #98](https://github.com/deepseek-ai/FlashMLA/pull/98)：分别提供 FP8 MQA logits/paged MQA logits，以及 sparse prefill/sparse FP8 decode/SM90 sparse MLA 的高性能 CUDA 入口。
7. [TileLang DeepSeek V3.2 examples](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32)：将执行链拆为 Lightning Indexer、radix/histogram top-k selector 和 sparse MLA，并展示 pipelined double buffering 与 FP8 sparse MLA；这是研究实现证据，不是本地性能复现。
8. [vLLM DeepSeek-V3.2-Exp recipe](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html)：核对 DeepGEMM 的 MoE/MQA logits 使用、`DP=8, EP=8, TP=1`、TP fallback、FP8/BF16 KV cache、`max-num-seqs` 与 GSM8K recipe 结果。5-shot `0.9591`/20-shot `0.9538` 绑定 V3.2-Exp + vLLM + lm-eval，不能写成裸模型分数。
9. 正式落点为第二十一册第 19 章的 [`19.29--19.43`](book-21-transformer-architecture-evolution/chapters/19-Sliding-Window与稀疏注意力.md)：补充最终 V3.2 与 V3.2-Exp artifact 分层、FP8 indexer/sparse MLA 执行链、kernel/recipe/模型能力分账、`thinking with tools`/DSML encoding 协议边界、Search Agent context management，以及 Figures 1–7 与 DSA/scalable RL Eq. (1)–(9) 的架构、评测、训练和证据边界。
10. [DeepSeek V3.2 arXiv v1 HTML](https://arxiv.org/html/2512.02556v1)：§4.4 对比 `Summary`、`Discard-75%`、`Discard-all` 与 `Parallel-fewest-step`；报告给出 80% context trigger、128K 上限下约 20%+ 案例超限，以及 BrowseComp Pass@1 无管理 51.4/带管理 67.6*。这些是商业 Search API + Agent harness 的论文结果；“up to 60.2”原文未标百分号，不能改写成 `+60.2%`。
11. Figure 3 的 [prefill cost panel](https://arxiv.org/html/2512.02556v1/cost_prefilling.svg) 与 [decode cost panel](https://arxiv.org/html/2512.02556v1/cost_decoding.svg) 已视觉核验：token position 越长，V3.2 的每百万 token 成本曲线越平缓；极短位置存在交叉。报告口径为 H800 实际部署服务并按 `$2/GPU-hour` 估算，不是 API 单价或独立复现；不从无数值标签曲线估读点值。
12. [Figure 4 thinking-retention illustration](https://arxiv.org/html/2512.02556v1/figures/template.JPEG)：同一轮中追加工具消息时保留历史 thinking；新的 user message 到来后清除旧 thinking，但保留 tool-call/result 与前轮 answer。该图说明消息类别与上下文回放策略的耦合，不是永久记忆或通用客户端保证。
13. [Figure 5 合成 Agent 数据 RL 曲线](https://arxiv.org/html/2512.02556v1/figures/synthesis-rl-plot.png)：对比 synthetic-data RL、V3.2-SFT 与 search/code RL 的 V3.2-Exp 基线。曲线属于发布方训练消融，不是最终模型的独立 benchmark，也不能证明真实环境泛化因果、数据无污染或完整 RL recipe 已公开。
14. Figure 7 的 [MLA-MHA panel](https://arxiv.org/html/2512.02556v1/MLA-MHA.svg) 与 [MLA-MQA panel](https://arxiv.org/html/2512.02556v1/MLA-MQA.svg) 展示 per-head K/V 投影与 shared latent KV 两种形式。V3.1-Terminus 阶段安排与 V3.2-Exp inference demo 必须分别归因；图示不单独证明 cache bytes、吞吐、kernel 覆盖或质量等价。
15. [Figure 1 benchmark chart](https://arxiv.org/html/2512.02556v1/v32_performance.svg) 分隔 reasoning/agentic 指标，且 Codeforces 使用不同纵轴。HMMT 2025 February、HLE text-only 与 HLE 模板变体应显式记录；V3.2-Thinking 在通用模板与 HLE 官方模板下的 HLE 数字不同（图中 25.1，正文另报 23.9）。作者报告的内部 tool-use 环境结果不等于 AA/DataCurve 或独立复现。
16. DSA Eq. (1)–(4) 的 HTML TeX 文本说明：index score 不等于概率/attention output；Top-k 只选 latent KV，主 attention 再算结果；dense warm-up 与 sparse stage 使用不同范围的 indexer KL。Eq. (4) 没有明确写子集后的 target normalization，训练代码待核验；本轮 PDF 下载成功但未视觉核验。
17. [DeepSeek-V3.2-Exp 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)：本轮 7890 可读网页与 raw README，仓库明确公开 inference demo 并链接 TileLang/DeepGEMM/FlashMLA；可见根目录没有 trainer/loss 源码入口。该负证据只限当前仓库，不能推断其他官方或私有训练代码不存在，也没有解答 Eq. (4) 的 target normalization。
18. Eq. (5)–(9) 的 arXiv v1 HTML TeX annotations 已文本核对：Eq. (5) 嵌套 group response mean 与 per-response token mean；Eq. (6) 展示 outcome reward 减 group mean；Eq. (7) 用 current/old importance weight 修正 KL estimator；Eq. (8) mask 只乘 clipped policy 项；Eq. (9) 仅屏蔽负 advantage 且 response-mean divergence 超阈值的序列。Keep Routing 与 Keep Sampling Mask 是作者披露的额外稳定化策略。仅为 HTML 源文本检查，未在 PDF 页面视觉核验，不代表完整训练 recipe。

配置边界纠正：固定官方 V3.2 `config.json` 与 V3.2-Exp inference config 都是 `q_lora_rank=1536`，但属于不同 revision/artifact；不能把实验实现字段、kernel 或 recipe 合并成最终 V3.2 的完整生产配置或独立 benchmark。

#### 2026-09-21 当前页面与 README 复核

Artificial Analysis 当前详情快照为 3,638,730 bytes / SHA-256 4dbd4f6cb11f25092a221ef0490a910f91f43885330fe34cc38c212c4d7470c3；结构化字段为 685B total/37B active、128K、Intelligence Index 16.043537719683、0.28/0.42 美元每百万输入/输出 token。9 月 20 日页面的 648B 只作为历史第三方目录字段保留；没有 DataCurve 精确 V3.2 行，也不补写当前 output speed/TTFT。

V3.2-Exp README 当前快照为 6,899 bytes / SHA-256 dffcdf358a42599945d49293a4f210dbe589141085207b76c081c9ace1f8fd74。新增核验点是 V3.1-Terminus 对齐对照、indexer non-interleaved RoPE 与 MLA layout 修复、TileLang/DeepGEMM/FlashMLA 分层和 SGLang dsv32 启动入口。vLLM recipe 当前 URL 返回 404，只是链接/线路负证据，不改变已有 recipe 记录。正式映射扩展到第二十一册第 19 章 19.34。

### Qwen3.8 Max (0902) 服务 revision 与上下文缓存

1. [Artificial Analysis Qwen3.8 Max](https://artificialanalysis.ai/models/qwen3-8-max)：本轮候选发现入口；当前页面标题为 `Qwen3.8 Max (0902)`，canonical 页面为 `qwen3-8-max`，release slug 为 `qwen3-8-max-0902`。
2. [Qwen3.8-Max-0902](https://www.qwencloud.com/models/qwen3.8-max-0902)：官方产品页；alias 为 `qwen3.8-max-2026-09-02`，明确称其为 `qwen3.8-max` 的 upgraded snapshot，并给出 1M/991K/983K/131K 请求边界和价格快照。
3. [Qwen Cloud Thinking](https://docs.qwencloud.com/developer-guides/text-generation/thinking)：核对 `reasoning_effort=low/medium/xhigh`、默认 `xhigh` 和与 `thinking_budget` 的互斥关系。
4. [Qwen Cloud Function Calling](https://docs.qwencloud.com/developer-guides/tool-calling/function-calling)：核对 thinking 模式下 `tool_choice` 只能为 `auto`/`none`，以及 `MultiModalConversation` 接口。
5. [Qwen Cloud Context Cache](https://docs.qwencloud.com/developer-guides/run-and-scale/context-cache)：核对 explicit、implicit、session cache、最小 1,024-token 长度及不同命中/计费/有效期语义；7890 复核还确认 `ephemeral` marker、最多 4 个且只取最后 4 个、20 content-block lookback、5 分钟 TTL/reset、账户/模型隔离、`previous_response_id` lineage 和 `usage.input_tokens_details.cached_tokens`。
6. [Qwen Cloud Dynamic Rate Limiting](https://docs.qwencloud.com/developer-guides/administration/dynamic-rate-limits)：核对 account+model 聚合、workspace override、月度 TPM tier、soft limit 和 `qwen3.8-max-0902` 的 `1,500,000 / 1,500,000 / 1,500,000` 保证 TPM；这是 hosted quota 证据，不是模型吞吐或 GPU capacity。
7. [DataCurve DeepSWE](https://deepswe.datacurve.ai/)：只有泛化 `mini_swe_agent_qwen3_8_max_xhigh` 行；没有 0902 精确行，不迁移 Pass@1、成本或 Agent steps。

本轮没有检出 0902 专属论文、独立技术报告、公开新权重或完整架构说明。QwenCloud 文档示例已从 `dashscope-intl.aliyuncs.com` 迁移到 `maas.qwencloudapi.com`，这是 provider adapter/transport 配置变更；产品页当前价格还区分 input/output/implicit cache `$2/$6/$0.25` 与 explicit cache creation/read `$2.50/$0.17` 每百万 token，Context Cache 文档的典型相对口径则为 explicit/session 创建 `125%`、命中 `10%`，implicit 创建 `100%`、命中 `20%`。新增 [`qwen_max0902_cache_contract_audit.py`](research/model-update-2026-09/code/qwen_max0902_cache_contract_audit.py) 仅提供 `local_protocol_toy` 证据。0902 的 coding、长周期 autonomous development、多工具 Agent 和视觉理解按官方产品定位记录，不升级为训练/架构事实；Qwen3.8 Max 继续作为基于 A95B 的 hosted service 记录，而非新的 open checkpoint。

### GPT-5.3 Codex：官方运行时与 Agent harness 资料

1. [Artificial Analysis GPT-5.3 Codex](https://artificialanalysis.ai/models/gpt-5-3-codex)：候选发现入口，页面为 `GPT-5.3 Codex (xhigh)`；DataCurve 当前没有精确 GPT-5.3 Codex 行。
2. [GPT-5.3-Codex model page](https://developers.openai.com/api/docs/models/gpt-5.3-codex.md)：核对 `gpt-5.3-codex`、400K context、272K maximum input、128K maximum output、Responses-only、reasoning effort、function calling、web search、hosted shell 和 skills。
3. [Codex Prompting Guide](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide.md)：核对较少 reasoning tokens、medium/high/xhigh 选择、长时自治、first-class compaction、代码库探索、`apply_patch`、工具 schema、并行调用和 phase。
4. [Compaction](https://developers.openai.com/api/docs/guides/compaction.md)、[Conversation state](https://developers.openai.com/api/docs/guides/conversation-state.md)、[Tools](https://developers.openai.com/api/docs/guides/tools.md)、[Agents](https://developers.openai.com/api/docs/guides/agents.md) 和 [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md)：核对完整 output item replay、加密 reasoning item、`previous_response_id`、canonical context、KV prefix cache 和模型/宿主/执行器边界。

这些资料是 API、协议和 harness 证据，不是 GPT-5.3 Codex 的架构论文。官方没有公开参数规模、MoE/稠密结构、训练 recipe、system card、生产 kernel 或线上 acceptance rate；不要把其他 GPT/Codex 配置的 DeepSWE 结果迁移到该锚点。

### OpenAI gpt-oss：开放权重 MoE、Harmony 与可变推理

1. [gpt-oss Model Card](https://arxiv.org/abs/2508.10925)：OpenAI 官方模型卡/技术报告，核对 `gpt-oss-120b`/`gpt-oss-20b` 的 MoE 规模、交替 sliding/full attention、GQA、RoPE/YaRN、MXFP4、预训练/后训练方向、工具、评测和开放权重安全边界。
2. [OpenAI gpt-oss repository](https://github.com/openai/gpt-oss)：核对 MXFP4 参考实现、Triton/PyTorch 路径、本地推理和 serving 入口；仓库实现不能替代完整生产 profiling。
3. [OpenAI Harmony](https://github.com/openai/harmony)：核对角色层级、`analysis`/`commentary`/`final` channel、recipient、工具调用和 structured output 的渲染/解析协议。
4. [gpt-oss Cookbook](https://developers.openai.com/cookbook/topic/gpt-oss)、[Transformers](https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers)、[vLLM](https://developers.openai.com/cookbook/articles/gpt-oss/run-vllm)：核对 chat template、Harmony、函数调用和本地后端接入边界。

这些资料支持模型卡公开的架构、量化和协议事实，不支持完整 optimizer、CoT RL 奖励/rollout recipe、router 负载均衡、MXFP4 kernel 误差消融、硬件 profiling 或线上接受率。Artificial Analysis 的两个锚点是候选发现来源；DataCurve 当前没有精确 gpt-oss 行，不能迁移其他 OpenAI/Codex 的 Agent 评测。

### Claude Opus 4.6：长任务协议与 Agent harness

1. [Artificial Analysis Claude Opus 4.6 adaptive](https://artificialanalysis.ai/models/claude-opus-4-6-adaptive) 与 [基础配置](https://artificialanalysis.ai/models/claude-opus-4-6)：模型发现入口；两个条目按同一基础模型归并。
2. [Claude Opus 4.6 发布页](https://www.anthropic.com/news/claude-opus-4-6) 与 [System Card](https://www.anthropic.com/claude-opus-4-6-system-card)：发布方模型定位、评测和安全资料。
3. [模型页](https://platform.claude.com/docs/en/models/opus-4-6/overview.md)、[Adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking)、[Effort](https://platform.claude.com/docs/en/build-with-claude/effort)、[Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)：1M/128K/300K、effort、thinking signature 与 compaction 协议。
4. [Tool search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)、[Computer use](https://platform.claude.com/docs/en/agents-and-tools/computer-use)、[Advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use)：schema 按需发现和宿主执行器边界。
5. [外部论文：AI control monitors](https://arxiv.org/abs/2604.13069)、[外部论文：poisoned identifiers](https://arxiv.org/abs/2604.04289)：只作为外部使用/评测证据，不是 Anthropic 官方 Opus 4.6 技术报告。

DataCurve 当前没有精确 `mini_swe_agent_claude_opus_4_6_*` 行，不能迁移其他 Claude 版本的 Pass@1、成本或 Agent steps。完整核验与待核验项见 [`claude-opus-4.6-source-notes.md`](research/model-update-2026-09/claude-opus-4.6-source-notes.md)。

### Claude Opus 4.7：Task budget、视觉输入与长任务控制面

1. [Artificial Analysis Claude Opus 4.7 adaptive/max](https://artificialanalysis.ai/models/claude-opus-4-7) 与 [non-reasoning/high](https://artificialanalysis.ai/models/claude-opus-4-7-non-reasoning)：候选发现入口；两个配置按同一基础模型归并。
2. [Claude Opus 4.7 发布页](https://www.anthropic.com/news/claude-opus-4-7) 与 [System Card](https://www.anthropic.com/claude-opus-4-7-system-card)：发布方定位、评测和安全控制面资料。
3. [模型页](https://platform.claude.com/docs/en/models/opus-4-7/overview.md) 与 [Effort](https://platform.claude.com/docs/en/build-with-claude/effort)：`claude-opus-4-7`、1M context、128K synchronous output、adaptive thinking、`high/xhigh/max` 行为档位。
4. [Task budgets](https://platform.claude.com/docs/en/build-with-claude/task-budgets)、[Vision](https://platform.claude.com/docs/en/build-with-claude/vision)、[Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction) 和 [Tool search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)：Agent loop 预算、`2576 px/4784 visual tokens`、状态压缩和按需工具 schema。

这些资料支持 API、Agent 和部署控制面的面试结论，不支持参数量、内部架构、完整训练 recipe、生产 kernel 或线上 acceptance rate。DataCurve 当前没有精确 Opus 4.7 行，不得迁移 Opus 4.6、4.8、5 的 Agent 结果。完整证据见 [`claude-opus-4.7-source-notes.md`](research/model-update-2026-09/claude-opus-4.7-source-notes.md)。

### Claude Opus 4.8：Dynamic Workflows 与多 Agent benchmark

1. [Artificial Analysis Claude Opus 4.8](https://artificialanalysis.ai/models/claude-opus-4-8) 与 [DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/) 的 `xhigh/max + mini-swe-agent` 行：仅用于锚点及配置条件下的榜单结果；AA 当前已将模型标为 deprecated。
2. [Claude Opus 4.8 System Card](https://www.anthropic.com/claude-opus-4-8-system-card)：§8.11 描述 BrowseComp/ProgramBench 的 blocking orchestrator、fixed team 和 async-subagent harness 及 score/token/派生 latency 评估；§6.3.6 描述 coding diligence 与状态总结诚实度。
3. [Introducing dynamic workflows in Claude Code](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code)：长任务的动态外部编排、并行 subagents、独立检查/反驳、checkpoint 恢复，以及成本、首次确认和管理员控制。

System Card 分数是特定 Anthropic harness 的发布方结果；其 latency 由固定 prefill/decode rate、token 数与工具耗时推导，不是生产 wall-clock。Dynamic Workflows 产品不能与 System Card benchmark harness 视为同一实现。完整证据及页面 hash、勘误和限制见 [`claude-opus-4.8-source-notes.md`](research/model-update-2026-09/claude-opus-4.8-source-notes.md)。没有独立技术报告、内部架构或完整训练配方，不新增第二十一册架构章节。

## 2026-09 Kimi K3 技术报告

1. [Kimi K3 官方仓库](https://github.com/MoonshotAI/Kimi-K3) 与 [`k3_tech_report.pdf`](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)：2.8T/104B activated、KDA/Gated MLA 混合、Block AttnRes、Stable LatentMoE、Quantile Balancing、MXFP4/MXFP8 QAT、长轨迹 RL 和 XTM 协议。
2. [Kimi Linear](https://arxiv.org/abs/2510.26692) 与 [Attention Residuals](https://arxiv.org/abs/2603.15031)：用于解释 KDA、AttnRes 的通用论文机制；论文实验配置不自动替换 K3 report 配置。
3. [固定 HF revision/config](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)：确认 `KimiK3ForConditionalGeneration`、93 layers、69 KDA + 24 Gated MLA、`q_lora_rank=1536`、`kv_lora_rank=512`、896 experts/top-16/2 shared 和 1,048,576 context；完整权重 metadata 不等于本地下载。
4. [FlashKDA](https://github.com/MoonshotAI/FlashKDA)：master commit `7afb9f454f160a6c4bbc0999beca0a8c40a38934`、`chunk_kda` backend、recurrent state、SM90+/CUDA 12.9+/PyTorch 2.4+ 和 H20/GB200 仓库 benchmark。
5. [vLLM K3 recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3)：hybrid KV manager、MLA/KDA 双状态、prefix caching、DCP/TP/TEP/DEP/PP 拓扑、Blackwell FP8 KV 与 tool-call parser 兼容性提示。

K3 report 与固定 config 是官方配置证据，不等于完整训练 recipe、目标硬件 profiling 或独立复现报告；完整权重未下载，FlashKDA 已固定公开 commit，vLLM recipe 已公开但 stable upstream merge、线上 acceptance 和端到端 serving 收益仍待核验。

## 2026-09 Kimi K3 fixed artifact/runtime evidence

1. [固定 HF safetensors index](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/model.safetensors.index.json)：`497,220` tensor mappings、96 shards、`metadata.total_size=1,560,860,324,864` bytes；这是 packed artifact 证据，不是本地权重加载证明。
2. [固定 `modeling_kimi_linear.py`](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py)：`KimiDynamicCache` 分离 full-attention K/V 与 KDA conv/recurrent state；chunk/prefill 和 single-token recurrent decode 走不同路径。
3. [`kimi_k3_manifest_audit.py`](research/model-update-2026-09/code/kimi_k3_manifest_audit.py)：零依赖教学审计器，用于验证 revision/config/index 的一致性；通过不等于完整权重、CUDA kernel、目标硬件 profiling 或线上 acceptance。

## Qwen3.5-397B-A17B：原生多模态与混合 state 架构

1. [Qwen3.5-397B-A17B 模型卡](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)：397B total/17B active、Gated DeltaNet/Gated Attention、512-expert MoE、vision encoder、MTP、262K native context 和 serving 示例。
2. [Qwen3.5 官方博客](https://qwen.ai/blog?id=qwen3.5)：early-fusion、多模态 token、million-agent RL、异步 RL 和 201 languages/dialects 的发布方说明。
3. [Qwen3.8 官方仓库 README](https://github.com/QwenLM/Qwen3.8)：Qwen3.8 建立在 Qwen3.5 架构基础上的系列关系与 Qwen3.5 引用入口；QSA/Gated Residual/N-gram/Muon 仍按 Qwen3.8/Flash-Next 证据记录。

本项目把 Qwen3.5-397B-A17B 作为 Qwen3.8 专题的前置锚点；模型卡和博客的 benchmark、训练效率与 RL 规模数字均为 Qwen 官方自报，不等同于独立技术报告或复现。

## GLM-5：DSA、MoE、slime 与 Agentic Engineering

1. [Artificial Analysis GLM-5](https://artificialanalysis.ai/models/glm-5)：精确 `GLM-5 (Reasoning)` 榜单条目；2026-09-20 页面提示已有更新模型 `GLM-5.1`，因此保留为历史 AA 锚点。
2. [Z.ai GLM-5 博客](https://z.ai/blog/glm-5) 与 [GLM-5 API 文档](https://docs.z.ai/guides/llm/glm-5)：模型定位、DSA、长周期 Agent 和部署入口；本轮博客资源快照可复验，文档 Markdown 端点的代理 503 只作为线路失败记录。
3. [GLM-5 模型卡](https://huggingface.co/zai-org/GLM-5)、[固定 README](https://huggingface.co/zai-org/GLM-5/raw/main/README.md) 与 [固定 config.json](https://huggingface.co/zai-org/GLM-5/raw/main/config.json)：`GlmMoeDsaForCausalLM`、744B/40B、78 层、256 routed experts/top-8/1 shared、`index_topk=2048`、低秩字段和约 202K positions。
4. [GLM-5 技术报告](https://arxiv.org/abs/2602.15763)：*GLM-5: from Vibe Coding to Agentic Engineering*；用于解释 DSA、异步 RL 基础设施 `slime`、长轨迹 Agent 和从代码生成到可验证工程 artifact 的转向。

本项目新增第二十一册第 89 章 [`GLM-5：DSA、MoE、slime 与 Agentic Engineering`](book-21-transformer-architecture-evolution/chapters/89-glm-5-dsa-slime与agentic-engineering.md)。DataCurve 当前没有精确 `mini_swe_agent_glm_5_*` 行，不能迁移 GLM-5.2/5.3 的 DeepSWE 结果；完整 indexer loss/recall、生产 kernel、异步 freshness 控制、训练 recipe、硬件 profiling 和独立复现仍待核验。

## Gemini 3.5 Flash-Lite：模型卡与视频 Agent 资料

1. [Artificial Analysis Gemini 3.5 Flash-Lite](https://artificialanalysis.ai/models/gemini-3-5-flash-lite)：模型发现入口；`releaseDate`、`isReasoning` 和 Intelligence Index 只作为第三方配置字段。DataCurve 当前没有精确 `mini_swe_agent_gemini_3_5_flash_lite_*` 行。
2. [Gemini 3.5 Flash-Lite API model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)：`gemini-3.5-flash-lite`、text/image/video/audio/PDF、1M/65K、thinking、缓存、代码执行、File Search、function calling、Search/Maps grounding 和 Computer Use Preview。
3. [Gemini 3.5 Flash-Lite Model Card](https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/) 与 [PDF](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-5-Flash-Lite-Model-Card.pdf)：低延迟/高吞吐定位、based on Gemini 3.1 Flash-Lite、发布方 benchmark、价格、安全和限制；architecture/训练/软硬件字段指向 3.1 Model Card。
4. [Thinking](https://ai.google.dev/gemini-api/docs/thinking) 与 [Video understanding](https://ai.google.dev/gemini-api/docs/video-understanding)：Lite 默认 `On (minimal)`、四档 thinking、static/agentic video、按需时间轴读取及 `processing_call/result` 处理事件。

当前没有检出 Gemini 3.5 Flash-Lite 独立架构论文或完整训练报告。因此这些来源支持的是 API、推理预算、视频处理和证据分层的面试知识，不支持参数、层结构、训练 recipe、生产 kernel 或线上 acceptance rate 的结论；也不能把 Gemini 3.5 Flash 的 `medium` 默认值或 GA 页面迁移给 Lite。

## Kimi K2.6：模型卡、Agent Swarm 与推理验收

1. [Artificial Analysis Kimi K2.6](https://artificialanalysis.ai/models/kimi-k2-6)：模型发现入口；DataCurve 当前没有精确 `mini_swe_agent_kimi_k2_6_*` 行，不迁移 K2.7 Code/K3 的 Agent 结果。
2. [Kimi K2.6 Hugging Face 模型卡](https://huggingface.co/moonshotai/Kimi-K2.6) 与 [固定 config.json](https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/config.json)：1T/32B、61 层、384 experts/top-8/1 shared、MLA、YaRN 256K、MoonViT 和 native INT4 的公开字段。
3. [Kimi K2.6 部署指南](https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/docs/deploy_guidance.md)：K2.5 架构路线复用、vLLM/SGLang/KTransformers 和 `kimi_k2` tool/reasoning parser。
4. [Kimi K2.6 官方博客](https://www.kimi.com/blog/kimi-k2-6) 与 [Kimi Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier.html)：long-horizon coding、Agent Swarm、proactive agents，以及参数、视觉、长输出、ToolCall 和 SWE-Bench 的部署验收分层。

正式落点为第二十一册第 90 章 [`Kimi K2.6：Native Multimodal、Agent Swarm 与推理验收`](book-21-transformer-architecture-evolution/chapters/90-kimi-k2.6-native-multimodal-agent-swarm与推理验收.md)。官方资料没有公开完整 K2.6 训练 recipe、生产 kernel 或独立技术报告；不要把 K3 的 KDA/AttnRes 或 K2.7 Code 的协议/评测回写为 K2.6 专属事实。

## GPT-5.4 mini/nano：精确 API sibling 与小模型 serving

1. [Artificial Analysis GPT-5.4 mini](https://artificialanalysis.ai/models/gpt-5-4-mini) 与 [GPT-5.4 nano](https://artificialanalysis.ai/models/gpt-5-4-nano)：模型发现入口；DataCurve 当前只有 GPT-5.4 base 的 `mini_swe_agent_gpt_5_4_xhigh`，没有 mini/nano 精确 Agent 行。
2. [GPT-5.4 mini 模型页](https://developers.openai.com/api/docs/models/gpt-5.4-mini.md) 与 [GPT-5.4 nano 模型页](https://developers.openai.com/api/docs/models/gpt-5.4-nano.md)：各自 `2026-03-17` snapshot、400K/272K/128K、模态、effort、定位、价格和能力矩阵。
3. [Using GPT-5.4](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.4.md) 与 [Reasoning models](https://developers.openai.com/api/docs/guides/reasoning.md)：小模型 prompt contract、task-shape routing、reasoning effort 和 Agent/tool 运行时边界。

本轮未检出 GPT-5.4 mini/nano 独立参数、架构论文、完整训练/后训练报告、公开权重或精确 DataCurve benchmark。因此这些来源支持的是精确 API sibling、路由、预算、能力探测和评测归因的面试知识，不支持内部 Transformer 结构或训练 recipe 结论；完整摘记见 [`gpt-5.4-mini-nano-source-notes.md`](research/model-update-2026-09/gpt-5.4-mini-nano-source-notes.md)。

## DeepSeek V4：CSA/HCA、mHC 与百万上下文系统

1. [DeepSeek V4 技术报告（arXiv:2606.19348）](https://arxiv.org/abs/2606.19348)：V4 的 CSA/HCA、mHC、Muon、长上下文训练/serving 与后训练流程；正文实验仍应按论文基线、精度、硬件和实现条件引用。
2. [DeepSeek-V4-Pro 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) 与 [配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json)：1.6T/49B、1M context、MoE、61 层、384 routed/6 selected/1 shared、低秩字段、`index_topk=1024`、YaRN 和量化配置。
3. [V4 Pro GA 公告](https://api-docs.deepseek.com/news/news260813)、[Responses API](https://api-docs.deepseek.com/guides/responses_api)、[Thinking](https://api-docs.deepseek.com/guides/thinking) 和 [Tool Calls](https://api-docs.deepseek.com/guides/tool_calls)：`low/high/max`、Codex 优化、stateless Responses、工具协议和 reasoning/tool loop 的官方运行时证据。
4. [Artificial Analysis V4 Pro](https://artificialanalysis.ai/models/deepseek-v4-pro) 与 [DataCurve DeepSWE](https://deepswe.datacurve.ai/)：榜单配置与 Agent harness 结果；DataCurve 的 Pass@1 `62.831858%`、Pass@4 `88.495575%` 只适用于 `max + mini-swe-agent + 工具/环境/verifier`，不写成裸模型分数。

本轮没有把 V4 Pro 资料写成全新的架构章节，因为 CSA/HCA、mHC、MoE、低精度和后训练已有第二十一册第 77/78 章及 DeepSeek V4/V4.1 专题承载；生产 kernel、硬件 profiling、完整 recipe、线上 acceptance 和独立复现仍待核验。

### DeepSeek V4 Flash：SGLang serving implementation artifacts

以下是已确认 DeepSeek V4 锚点的框架实现资料，不是新的模型发现入口，也不是学术论文：

1. [SGLang v0.5.20 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.20)：stable tag 与发布日期入口；release metadata 固定值和快照哈希见 [`deepseek-v4-source-notes.md`](research/model-update-2026-09/deepseek-v4-source-notes.md)。
2. [PR #34565：SWA branch-point caching](https://github.com/sgl-project/sglang/pull/34565)：混合注意力下保留共享前缀分支点的 SWA 状态；PR 中 token hit rate、TTFT 等数字绑定其 V4-Flash-0731 workload。
3. [PR #30805：TRT-LLM CSA/HCA attention for SM100/103](https://github.com/sgl-project/sglang/pull/30805) 与 [PR #29927：DeepSeek V4 on SM120](https://github.com/sgl-project/sglang/pull/29927)：分别对应 B200 单元 kernel 对照与 RTX PRO 6000 sparse-indexer/MoE 路径；不同 baseline/硬件/测量范围不能横向合并成一个加速倍数。
4. [PR #39171：V4.1 FlashMLA fork rebase](https://github.com/sgl-project/sglang/pull/39171)：只能证明相关依赖 pin 更新，不等同于 V4.1 完整 vision/runtime 或生产验收。

源码与发布方 benchmark 均不替代完整权重加载、目标硬件独立复现、准确性/召回和生产流量端到端 SLO；证据细节见 [`deepseek-v4-source-notes.md`](research/model-update-2026-09/deepseek-v4-source-notes.md) 与第二十一册第 77 章。

## GLM-5.1：长周期 Agent 与过程质量资料入口

1. [Artificial Analysis GLM-5.1](https://artificialanalysis.ai/models/glm-5-1)：模型发现入口；DataCurve 当前没有精确 `mini_swe_agent_glm_5_1_*` 行。
2. [Z.ai GLM-5.1 文档](https://docs.z.ai/guides/llm/glm-5.1)：200K/128K、长周期 Agent、实验—分析—优化、thinking、function calling、MCP、structured output 和 context cache 能力入口。
3. [Hugging Face GLM-5.1 模型卡](https://huggingface.co/zai-org/GLM-5.1) 与 [config.json](https://huggingface.co/zai-org/GLM-5.1/blob/main/config.json)：`GlmMoeDsaForCausalLM`、78 层、256 routed/top-8/1 shared、前三层 dense、低秩 q/kv 字段、`index_topk=2048` 和 202752 positions。
4. [Z.ai GLM-5.1 官方博客](https://z.ai/blog/glm-5.1)：补充 VectorDBBench 的外层迭代优化、KernelBench Level 3 的正确性/反作弊审计，以及无单一数值目标的 8 小时 Linux desktop 自评 harness；正文资源快照为 2026-09-21 官方 JS，不能当作独立技术报告或独立复现。
5. [Z.ai Release Notes](https://docs.z.ai/release-notes/new-released.md)：multi-turn SFT、RL、process-quality evaluation framework，以及 GLM-5.1 的长任务定位。
6. [GLM-5 技术报告](https://arxiv.org/abs/2602.15763)：模型卡引用的关联报告；不能改名为 GLM-5.1 专属报告。arXiv 精确检索本轮没有 GLM-5.1 专属技术报告。

GLM-5.1 的 SWE-Bench Pro `58.4`、655 次 Linux desktop 迭代/6.9× 吞吐和 KernelBench Level 3 `3.6×` 对比 `torch.compile` max-autotune `1.49×` 均是发布方自报，不能与 Artificial Analysis 或 DataCurve 拼成统一模型能力排名。完整证据和待核验项见 [`glm-5.1-source-notes.md`](research/model-update-2026-09/glm-5.1-source-notes.md)。

## Grok 4.20：Multi-agent runtime、状态协议与负面论文证据

1. [Artificial Analysis Grok 4.20](https://artificialanalysis.ai/models/grok-4-20)：本轮唯一模型发现入口；页面为 `Grok 4.20 0309 v2 (Reasoning)`，记录第三方 Intelligence Index、2M context、价格和性能字段。页面当前的 deprecated 状态只代表第三方目录，不替代 xAI API 身份。
2. [xAI Grok 4.20 模型页](https://docs.x.ai/developers/models/grok-4.20) / [Markdown](https://docs.x.ai/developers/models/grok-4.20.md)：核对 `grok-4.20-0309-reasoning`、non-reasoning、`grok-4.20-multi-agent-0309`、1M maximum prompt、function calling、structured output、reasoning、Batch、价格阈值和限流字段。
3. [xAI Multi-agent](https://docs.x.ai/developers/model-capabilities/text/multi-agent.md) 与 [Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning.md)：核对 4/16 Agent-count、leader synthesis、子 Agent opaque/encrypted state、内置 web/X/code/collections 工具，以及普通 reasoning 与 multi-agent `effort` 语义不能混用的边界。
4. [xAI Context Compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction.md)、[Tools](https://docs.x.ai/developers/tools/overview.md)、[Remote MCP](https://docs.x.ai/developers/tools/remote-mcp.md) 与 [Release Notes](https://docs.x.ai/developers/release-notes.md)：核对 opaque `compaction` item 原样回放、server/client tool 责任、`max_turns`、MCP `allowed_tools` 和 Grok 4.20/Multi-agent live 的发布入口。
5. [DataCurve DeepSWE](https://deepswe.datacurve.ai/)：当前没有精确 `mini_swe_agent_grok_4_20_*` 行；页面中的 Grok 4.5/4.6 结果不能迁移为 Grok 4.20 的 Pass@1、成本、输出 token 或 Agent steps。
6. [arXiv 标题精确检索 “Grok 4.20”](https://arxiv.org/search/?query=%22Grok+4.20%22&searchtype=title)：本轮无精确标题结果；全文命中只是外部评测或使用 Grok 4.20，没有检出 xAI 发布的 Grok 4.20 专属技术报告。因此不能补写参数量、MoE/注意力架构、训练 recipe、生产 kernel 或独立 benchmark。

这些资料支持的是模型目录、API/runtime、工具协议和评测证据分层，不支持把 multi-agent 编排写成新的 Transformer 结构。完整快照、哈希、负面证据和待核验项见 [`grok-4.20-source-notes.md`](research/model-update-2026-09/grok-4.20-source-notes.md)。

## Gemini 3.8 Flash：Thinking、Interactions 与工具协议资料

1. [Artificial Analysis Gemini 3.8 Flash high/medium/low](https://artificialanalysis.ai/models/gemini-3-8-flash)：榜单发现入口；三个 effort 行归并为一个基础模型。DataCurve high 精确行绑定 `mini-swe-agent`、工具、环境和 verifier，不能迁移到其他 effort 或 Gemini 版本。
2. [Gemini 3.8 Flash 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)、[Thinking](https://ai.google.dev/gemini-api/docs/thinking) 与 [Thought signatures](https://ai.google.dev/gemini-api/docs/thought-signatures)：1M 输入、65,536 输出、thinking level、共同 output budget、thought summary/signature 和回放边界；模型页列出视频输入。
3. [Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview) 与[当前正式 API reference](https://ai.google.dev/api/interactions-api)：stateful/stateless continuation、`previous_interaction_id`、step 类型和 SSE trace；用于 Agent 状态协议，而非内部架构证明。旧 reference 路径 `/api/interactions` 当前返回 404。
   [Google Gen AI Python SDK pinned revision](https://github.com/googleapis/python-genai/tree/6d012889752f65c1a51d0ad6e5970fc97d19c4ca)（2026-09-26 main commit）：custom `FunctionCallStep`/`FunctionResultStep` 未声明 signature，`ThoughtStep.signature` optional；`extra="allow"`、lenient union 和 `UnknownStep.raw` 提供 forward-compatible 保留路径，但不代表 endpoint 字段禁令。正式 schema 使用 `function_call.id` → `function_result.call_id`；Thinking/Tool-combination prose 的签名范围和 `function_response` 命名仍有冲突。
4. [工具组合](https://ai.google.dev/gemini-api/docs/tool-combination)、[Video Understanding](https://ai.google.dev/gemini-api/docs/video-understanding)、[Google Search grounding](https://ai.google.dev/gemini-api/docs/google-search)、[URL Context](https://ai.google.dev/gemini-api/docs/url-context)、[File Search](https://ai.google.dev/gemini-api/docs/file-search) 和 [Code Execution](https://ai.google.dev/gemini-api/docs/code-execution)：覆盖检索/索引/沙箱结果回灌，以及静态抽帧与按需 `processing_call` / `processing_result` 媒体读取；token/质量收益属于发布方文档声明。
5. [Computer Use](https://ai.google.dev/gemini-api/docs/computer-use) 与 [Structured outputs](https://ai.google.dev/gemini-api/docs/structured-output)：action intent、坐标/审批/宿主执行边界，以及 JSON Schema 的结构约束边界。
6. [Google DeepMind Model Card](https://deepmind.google/models/model-cards/gemini-3-8-flash/)、[官方发布博客](https://blog.google/innovation-and-ai/models-and-research/gemini-models/3-8-flash-and-3-8-flash-cyber/) 和 [官方评测 PDF](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-8_flash_model_evaluation.pdf)：发布方定位、安全/评测和对前代资料的继承关系。
7. arXiv 对精确 `Gemini 3.8 Flash` 标题未检出独立技术报告；因此本轮不补写参数量、MoE/注意力结构、训练 recipe、RL 或 verifier 实现。完整核验和 2026-09-20 哈希见 [`gemini-3.8-flash-source-notes.md`](research/model-update-2026-09/gemini-3.8-flash-source-notes.md)。

## DeepSeek V4 Pro：公开 inference artifact 与协议实现补证

1. [DeepSeek-V4-Pro 固定 HF revision](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/b5968e9190ef611bbf34a7229255be88a0e937c1)：`config.json`、`encoding/encoding_dsv4.py`、`inference/config.json`、`inference/model.py` 和 `inference/kernel.py` 的文件级证据；64 个权重分片只做 metadata 记录，完整权重未下载。
2. `inference/model.py` 将 gated KV compression、overlap state、causal/top-k indexer、MLA、128-token local window、hash/score routing、top-6 + shared expert、MTP 和 Hyper-Connections 串为参考执行路径；这补充论文的架构解释，但不等于生产 kernel 或线上服务实现。
3. `inference/kernel.py` 的 TileLang 路径覆盖 `[128,128]` block FP8/FP4 quantization、GEMM、sparse online softmax 和 HC Sinkhorn。FP4/FP8 误差与吞吐必须绑定 GPU、batch、commit、cache 状态和并行拓扑测量。
4. `encoding/encoding_dsv4.py` 是 DSML prompt encoder/parser，包含 tool role、string/JSON parameter、`<think>` 和严格 malformed-output 行为；它是协议层证据，不是隐藏思维链或内部 reasoning 算法。

本轮仍不把 `MP=8` 转换示例、HF metadata、DataCurve max harness 行或模型卡自报比例写成硬件无关的生产结论；完整权重加载、独立 profiling、index/evidence recall、线上 tool acceptance 和完整训练 recipe 待核验。

## Kimi K3：vLLM upstream/runtime 资料（2026-09-21）

- [vLLM stable supported models](https://docs.vllm.ai/en/stable/models/supported_models/)：稳定文档目录中的 K3 支持入口；快照 `738,169` bytes，SHA-256 `5420be70a431de9d6e25ba8322c27c1397239fc11643544b354591b8de80d069`。它是文档 discoverability 证据，不是 stable wheel 或完整权重加载证明。
- [vLLM stable K3 API](https://docs.vllm.ai/en/stable/api/vllm/models/kimi_k3/)：`KimiK3ForConditionalGeneration`/`KimiK3MTP` API reference；快照 `799,430` bytes，SHA-256 `d17b052921ae5ac2216453edf2034854f45a7d1e46a37ef90eff13e25e5cab5b`。API 类名不替代目标硬件 runtime acceptance。
- [vLLM main model registry](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/registry.py) 与 [K3 package init](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/kimi_k3/__init__.py)：分别记录 K3/MTP/DSpark registry entry 与 NVIDIA/ROCm platform branch；main 源码入口不等于 stable release。
- [K3 recipe YAML](https://raw.githubusercontent.com/vllm-project/recipes/main/models/moonshotai/Kimi-K3.yaml)：`Pre-release`/K3-enabled nightly 的部署账本，SHA-256 `5b2bcd21b8dfe210ea3f3d514c6a5448e395847d35fcb1f137d3567233e9d227`；用于研究 hybrid KV、并行拓扑、CUDA/driver 和 parser 风险，不当作稳定版本或生产 SLO。
- [FlashKDA](https://github.com/MoonshotAI/FlashKDA)：README/Atom 和 commit `7afb9f454f160a6c4bbc0999beca0a8c40a38934`；局部 kernel API/benchmark 与 K3 端到端 serving、cache recovery 和 tool-call acceptance 分开记录。

这些资料适合面试中的“证据分层”问题：文档/API -> stable release/source entry -> 预发布优化 recipe -> 目标硬件和线上验收。任何后一层都不能由前一层自动推导。

PyPI `vllm 0.29.0` 的 x86_64 wheel 于 2026-09-09 发布，大小 `315,961,042` bytes、SHA-256 `09d48617fc2be9c6cdcd5db480651ab0d84817b257204f2cc2e3ecbb70bbb635`；v0.29.0 tag 的 K3 registry/model source 已固定。它证明 stable release 中存在实现入口，但不替代安装日志、完整权重加载、目标硬件 profiling、hybrid cache recovery 和线上 acceptance。

### GPT-5.6 Luna 当前榜单与运行时资料

1. [GPT-5.6 Luna Artificial Analysis](https://artificialanalysis.ai/models/gpt-5-6-luna)：2026-09-23 当前 `max` 详情页快照为 `3,996,959` bytes、SHA-256 `4246b96416b4aaf4f8bbcacf72f8c45787944e24f02c59c6e3db1fed4de93c41`；Intelligence Index `37.3244239690841`、当前速度 `142.271568203031 tokens/s`、cost/task `0.17829726152289094` 和 1M context 都是第三方配置/provider 字段，速度变化按采集时点漂移记录。
2. [GPT-5.6 官方模型页族](https://developers.openai.com/api/docs/models/gpt-5.6-luna.md)、[Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md)、[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md)、[Tools](https://developers.openai.com/api/docs/guides/tools.md) 和 [Compaction](https://developers.openai.com/api/docs/guides/compaction.md)：既有官方快照支持 effort/mode、persisted reasoning、显式缓存断点、tool search、工具宿主责任和长任务压缩。
3. DataCurve 精确 `mini_swe_agent_gpt_5_6_luna_max` 行为 `301/448`、Pass@1 `67.1875%`、Pass@4 `90.2655%`、平均成本约 `$0.6056`、平均输出约 `73.4K` token、101.68 steps；它绑定 `mini-swe-agent`、工具、环境和 verifier，不能与 AA 指数合并。
4. 2026-09-23 通过 `10.24.27.134:7890` 取得当前官方 Markdown；Luna/Reasoning/Prompt caching/Compaction/Tools 的大小与哈希见 [`gpt-5.6-source-notes.md`](research/model-update-2026-09/gpt-5.6-source-notes.md)。页面恢复可读但没有新的公开架构、训练 recipe 或 system card。
5. [`gpt56_luna_state_replay_audit.py`](research/model-update-2026-09/code/gpt56_luna_state_replay_audit.py) 是无网络 local protocol toy，覆盖 opaque reasoning 的 turn scope、完整 item replay、function lineage、canonical compaction、cache miss、hosted/client tool search 和幂等 verifier；不代表真实 endpoint、模型质量或生产 SLO。

## Claude Sonnet 5：System Card、Adaptive Thinking 与 Agent 运行时

1. [Claude Sonnet 5 System Card](https://www.anthropic.com/claude-sonnet-5-system-card)：训练数据公开性、安全风险分级、Agentic safety、cyber 评测和发布方 benchmark 的主要证据。它没有公开参数规模、完整训练 recipe 或内部 adaptive-thinking 机制。
2. [Sonnet 5 模型页](https://platform.claude.com/docs/en/models/sonnet-5/overview)、[Adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/thinking) 和 [Effort](https://platform.claude.com/docs/en/build-with-claude/effort)：区分 adaptive、effort、`max_tokens`、thinking block/signature 和请求契约。
3. [Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)、[Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing) 和 [Programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)：长任务、工具状态和模型往返优化的 API 边界。
4. [Artificial Analysis Claude Sonnet 5](https://artificialanalysis.ai/models/claude-sonnet-5) 与 [DataCurve DeepSWE](https://deepswe.datacurve.ai/)：分别是第三方配置测量和 `mini-swe-agent` 系统结果；不能与 Anthropic 发布方分数合成裸模型排名。

本轮没有检出精确标题为 Sonnet 5 的 arXiv 技术报告。面试应把 `effort -> max_tokens -> task budget`、发布方 safeguards/harness、compaction/state replay 和“参数/训练 recipe 未公开”作为证据分层问题，而不是猜测内部结构。

## Claude Sonnet 5.5：Thinking State、Agent Runtime 与分层安全

1. [Claude Sonnet 5.5 System Card](https://www.anthropic.com/claude-sonnet-5-5-system-card)：训练过程公开边界、RSP 风险分级、三阶段 cyber safeguard、类别化 fallback、Agent prompt-injection 与 capability benchmark。评测数字绑定 snapshot、harness、effort、工具、safeguards、fallback 和 verifier。
2. [官方模型页](https://platform.claude.com/docs/en/models/sonnet-5-5/overview)、[What's New](https://platform.claude.com/docs/en/models/sonnet-5-5/whats-new-sonnet-5-5)、[Migration guide](https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide) 与 [Prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5-5)：`between_tools` 约束、forced-tool 不支持、thinking block 的 model/prefix/account binding、computer toolset 迁移、progress-block 呈现和 effort 重新校准。
3. [Artificial Analysis 条目](https://artificialanalysis.ai/models/claude-sonnet-5-5) 是第三方 max + fallback 配置测量；[DataCurve DeepSWE](https://deepswe.datacurve.ai/) 当前没有精确 Sonnet 5.5 Agent 行。不迁移其他 Claude 版本的 Agent 分数。
4. 2026-09-29 arXiv 精确查询返回 `totalResults=0`；本轮以 Anthropic System Card/API 文档为官方证据，不把被引用的 IPI 背景论文说成 Sonnet 5.5 技术报告。

研究笔记：[claude-sonnet-5.5-source-notes.md](research/model-update-2026-09/claude-sonnet-5.5-source-notes.md)；正式教材：[第二十册第 24 章](book-20-agent-harness-runtime/chapters/24-claude-sonnet-5.5-thinking-state与安全路由.md)与[第八册第 16.22 节](book-08-ai-safety-alignment/chapters/16-fallback-routing与安全降级.md#1622-claude-sonnet-55多阶段-cyber-gate-与类别化-fallback)。

## Grok 4.7：论文与官方资料边界

1. [Grok 4.7 官方发布页](https://x.ai/news/grok-4-7)：更大 base、更长 RL run、困难长任务混合、自验证、长上下文管理和 safeguard stack；CursorBench、DeepSWE、Terminal-Bench、Harvey、HealthBench、GDPval、LatchBio 和 HackerBench 数字均需保留发布方 harness 标签。
2. [Grok 4.7 模型/API 文档](https://docs.x.ai/developers/models/grok-4.7)、[Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning)、[Context Compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction)：核对 500K、effort、encrypted reasoning state、opaque compaction item 和服务合同；这些不是内部架构论文。
3. [Remote MCP](https://docs.x.ai/developers/tools/remote-mcp)、[Function Calling](https://docs.x.ai/developers/tools/function-calling) 与 [Structured Outputs](https://docs.x.ai/developers/features/structured-outputs)：核对工具 schema、`allowed_tools`、宿主授权与结构化输出边界。
4. [arXiv 精确标题检索](https://arxiv.org/search/?query=%22Grok+4.7%22&searchtype=title)：本轮没有检出 Grok 4.7 专属论文；因此不补写参数规模、MoE/稠密结构、完整训练 recipe、生产 kernel 或独立复现结论。
5. 本轮 Reasoning/Compaction/Remote MCP 刷新页不是论文，但提供了可复现的协议问题：encrypted reasoning 与 server-side tool output 的回放、summary delta 的观察、`store`/`previous_response_id` 的服务端状态引用，以及 Responses/原生 SDK 的 MCP 字段差异。面试和实验应将这些 API 合同与内部推理算法严格分开。

## K2 Horizon 3.7B：dense 长上下文与 artifact migration

1. [Artificial Analysis K2 Horizon 3.7B](https://artificialanalysis.ai/models/k2-horizon-3-7b)：模型发现入口；DataCurve 当前没有精确 `mini_swe_agent_k2_horizon_3_7b_*` 行，不迁移相邻 K2 或其他模型的 Agent 结果。
2. [IFM/K2-Horizon-3.7B model card](https://huggingface.co/IFM/K2-Horizon-3.7B)：核对 `K2HorizonForCausalLM`、36 层 dense、32Q/8KV GQA、524K position、BF16、分阶段长上下文训练、RL expert merge 和中间 checkpoint。
3. [vLLM recipe](https://recipes.vllm.ai/IFM/K2-Horizon-3.7B) 与 [SGLang PR #37654](https://github.com/sgl-project/sglang/pull/37654)：核对 5.06B dense/H200/`k2_horizon` parser、TP1/BF16/FlashAttention-3 和发布方性能口径；不能替代本机 profiling。
4. `/tmp/k2-migration-fixed-20260922.out`：核对 `k2_aurora -> k2_horizon`、copy、`weights_reencoded=false`、BF16、36 shards/327 tensors；这是 artifact 迁移证据，不是重新训练证明。
5. 旧 `APPENDIX.md` 的 `XllmForCausalLM`/FP32 与当前 `K2HorizonForCausalLM`/BF16 发生 revision 冲突；`3.78B core / 5.06B including embeddings` 可作为参数口径差异记录，不能覆盖当前固定 config/index。

正式映射为第二十一册第 82 章的 dense-vs-MoVA 对照；完整证据见 [`k2-horizon-3.7b-source-notes.md`](research/model-update-2026-09/k2-horizon-3.7b-source-notes.md)。

## Qwen3-Omni Technical Report：多模态 Thinker-Talker 与流式语音

1. [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)（arXiv:2509.17765，2025-09-22）：系统阅读 Thinker/Talker、AuT、TM-RoPE、多码本 Talker、MTP、Code2Wav、异步 chunked prefill、三阶段预训练和 Thinker/Talker 后训练。
2. [Qwen3-Omni 官方 GitHub](https://github.com/QwenLM/Qwen3-Omni)：对照模型卡、推理依赖、`disable_talker()`、Thinker serving 与当前 vLLM 音频输出边界。
3. 阅读重点：把 12.5 Hz/约 80 ms 音频 token 粒度、首包延迟、Talker residual codebook 并行度、声码器实时率和端到端流式 SLO 分开建账，避免把论文数字直接当成硬件实测。
4. 面试练习：解释为什么 Talker 使用多模态特征而不是只读取 Thinker 文本；画出 RAG/function calling/safety/verifier 插入 Thinker-Talker 的状态流；设计跨 audio/video timestamp 的 TM-RoPE 和 packet replay 验收。

本条对应正式专题 [`第二十一册第 91 章`](book-21-transformer-architecture-evolution/chapters/91-qwen3-omni-thinker-talker-aut与流式多模态.md) 和证据底稿 [`qwen3-omni-source-notes.md`](research/model-update-2026-09/qwen3-omni-source-notes.md)。

## Qwen3-VL Technical Report：三轴位置、视觉层级与视频时间戳

1. [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)：阅读 SigLIP2 vision encoder、两层 MLP merger、Qwen3 MoE decoder、Interleaved-MRoPE、DeepStack、Video Timestamp、S0-S3 curriculum、square-root normalized per-token loss、SAPO 和 Thinking with Images。
2. [Qwen3-VL 官方 GitHub](https://github.com/QwenLM/Qwen3-VL)：对照模型卡、Cookbook、GUI/手机 Agent、OCR、视频理解、grounding、multimodal coding 和部署入口。
3. [Qwen3-VL Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct) 与 [Thinking model card](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking)：固定 revision/config，区分同一基础模型的 artifact/运行配置，不把 AA 的 instruct/reasoning 页面算成两个模型。
4. [Transformers Qwen3-VL modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)：核对三维 `get_rope_index`、timestamp placeholder 和 DeepStack 前层注入；当前只固定 raw main 快照，不写成生产 kernel 或独立复现。
5. 阅读问题：为什么 tool-call reward 必须与 answer accuracy、multi-turn reasoning 分开？为什么 DeepStack 不增加视觉 token 长度却仍增加 serving 成本？如何区分 AA `235B/22B`、论文数据和本地 full-weight profiling？

本条对应正式专题 [`第二十一册第 92 章`](book-21-transformer-architecture-evolution/chapters/92-qwen3-vl-interleaved-mrope-deepstack与视频时间戳.md) 和证据底稿 [`qwen3-vl-source-notes.md`](research/model-update-2026-09/qwen3-vl-source-notes.md)。

## Qwen3.7 Plus：多模态交互式混合 Agent 的官方托管合同

1. [Artificial Analysis Qwen3.7 Plus](https://artificialanalysis.ai/models/qwen3-7-plus)：候选发现入口；当前第三方字段为约 1M context、`25.1622215821984` Intelligence Index、`68.5428061089526 tokens/s` 和约 `$0.40/$1.60` input/output，不能当作 Alibaba 的统一价格或裸模型能力。
2. [Alibaba Cloud Qwen3.7 Plus](https://www.alibabacloud.com/help/en/model-studio/qwen3-7-plus)：官方定位、`qwen3.7-plus-2026-05-26` snapshot 关系、text/image/video input、GUI/移动端/视觉参考生成代码、Function Calling、Structured Outputs、Web Search、Prefix Completion、Context Caching 和 context limits。
3. [Alibaba Cloud Recommended models](https://www.alibabacloud.com/help/en/model-studio/models)：2026-09-22 推荐模型页，在 text generation 和 vision model 列表中列出 `qwen3.7-plus`。
4. [OpenAI-compatible Chat](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope)：region-specific endpoint、region-bound API key、workspace domain、SDK/tool calling 和跨区域错误语义。
5. 研究底稿：[`qwen3.7-plus-source-notes.md`](research/model-update-2026-09/qwen3.7-plus-source-notes.md)。当前 DataCurve 没有精确 Qwen3.7 Plus 行；不能迁移其他 Qwen 的 Agent 分数，也没有公开专属技术报告、参数、权重或训练 recipe。

面试阅读重点是把模型 action proposal、schema、region/scope、宿主权限、GUI/mobile executor、observation、幂等、缓存和独立 verifier 分层。官方托管文档足以支持 API/runtime 合同，不能证明内部视觉 encoder、GUI policy、生产 kernel 或真实移动设备成功率。

## Qwen3.7 Max：Agent 环境扩展与跨框架 RL（官方博客，非技术论文）

1. [Qwen Team《Qwen3.7：智能体新前沿》](https://qwen.ai/blog?id=qwen3.7)：文章 API `path=qwen3.7` 于 2026-09-28 返回 HTTP 200；Qwen 自述训练环境质量/多样性扩展、留出 OOD benchmark、`Task × Harness × Verifier` 可组合 rollout、跨 harness/verifier RL、35 小时 M890 kernel 优化和 RL trajectory reward-hacking monitor。
2. 关键口径：M890 PPU 的 `10.0x` 相对 SGLang Triton 多 workload 几何平均；KernelBench L3 `1.98x/96%` 是 H100 50 题的两种不同统计量；两者不可混写。奖励监控的 13 条规则/1,618 个案例是发布方计数，不是 precision/recall。
3. 后续找到 [arXiv:2609.27321v1《Verifiable Hidden Dynamics Play》](https://arxiv.org/abs/2609.27321v1)，页面评论标为 “Qwen Technical Report”。它与 environment scaling 主题相关，但训练主体是 Qwen3.6-35B-A3B；Qwen3.7-Max 只在 setter/benchmark 对照出现。论文未声明 VHD-Play 实现了博客的 `Task × Harness × Verifier` 组合或 Max 的内部训练 recipe，不作该归属推断。
4. 仍未披露 Qwen3.7 Max 的参数量、内部 Transformer 架构或完整专属训练 recipe；VHD-Play 的作者结果尚无本项目独立复现。
5. 本地 Qwen 博客响应快照 `/tmp/qwen37-article-api-8098-20260928.json`：`122,153` bytes / `41e1f58384b99f1d2495111c3f6f55d01850283daa7ce5d07d38a764eea9108c`（含动态 request ID）；完整博客边界见 [`qwen3.7-max-source-notes.md`](research/model-update-2026-09/qwen3.7-max-source-notes.md)。

## GLM-5.3 标准 DSA：实现与 serving 证据

1. [GLM-5.3 模型卡与固定 config](https://huggingface.co/zai-org/GLM-5.3/tree/aca966e4e02791568aa6a4ced368624b3d897f42)：阅读 `GlmMoeDsaForCausalLM`、78 层、前三层 dense、256 routed/top-8/1 shared、`q_lora_rank`/`kv_lora_rank`、`index_topk`、Full/Shared indexer 类型和 MTP index sharing；不要把 config 当成完整训练账本。
2. [Transformers GLM-MoE-DSA configuration](https://github.com/huggingface/transformers/blob/0bc252863a4e5c0e709893664cc57369ebcd7353/src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py) 与 [modeling](https://github.com/huggingface/transformers/blob/0bc252863a4e5c0e709893664cc57369ebcd7353/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py)：核对 interleaved indexer RoPE、Full 层 top-k、Shared 层 `prev_topk_indices` 和 sparse attention mask。
3. [vLLM registry](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/models/registry.py)、[DeepSeek-V3.2 attention](https://github.com/vllm-project/vllm/tree/v0.29.0/vllm/model_executor/models/deepseek_v32) 与 [main 对照](https://github.com/vllm-project/vllm/tree/81d7293c2167e39f3ffddc9a82d633f94e8a1eaa/vllm/model_executor/models)：理解 `GlmMoeDsaForCausalLM -> deepseek_v32` 的 runtime 复用，以及 stable/main、PCP/DCP、HiSparse 和 logical top-k 的证据边界。
4. [SGLang v0.5.20](https://github.com/sgl-project/sglang/tree/v0.5.20/python/sglang/srt/models) 与 [main 固定 ref](https://github.com/sgl-project/sglang/tree/861b11f087af2822cb545ea1895059a721831014/python/sglang/srt/models)：核对 `is_glm_moe_dsa`、`DeepseekV32ForCausalLM` 和跨层 top-k 状态；mutable main 不能替代 stable wheel 或本机验收。
5. [IndexCache](https://arxiv.org/abs/2603.12201) 是 DSA lightning indexer 的关联 serving 论文；[SAO](https://arxiv.org/abs/2607.07508) 的直接对象是 GLM-5.2。阅读时区分“关联技术背景”和“GLM-5.3 专属已证实算法”，并单独设计 index/evidence recall、cache recovery、MTP acceptance 和目标硬件 profiling 实验。
6. [SAO HTML 正文](https://arxiv.org/html/2607.07508v1) 本地快照为 `190,454` bytes / SHA-256 `953b8968fa30d5f579a9650cc17d95521515ccac6cc7408b2a45c8126651822d`；PDF 为 `664,828` bytes / SHA-256 `44c695be0428c666d06c914ba76c037e3ac77eeb5db0a81bbe239719c21bda48`。论文的 Qwen3-30B-A3B 实验和 GLM-5.2 部署声明不能改写成 GLM-5.3 独有结果。

## Claude Opus 5.5：官方资料与评测边界

1. [Anthropic Claude Opus 5.5 发布页](https://www.anthropic.com/claude-opus-5-5)：记录 2026-09-22 发布、token/cache 成本、长任务 coding、benchmark 条件、Cyber/Life Sciences Verification、fallback 和 preserved thinking anti-distillation。它是发布方资料，不是独立技术论文。
2. [Claude Opus 5.5 System Card](https://www.anthropic.com/claude-opus-5-5-system-card)：PDF `17,795,106` bytes，SHA-256 `7311c9c6bbb16d012f1c12c7418b05949fcf7ae3e30d2c40f22050074b2a7378`；已用 Node.js `zlib` 与 ToUnicode/CMap 只读解析正文。正文覆盖训练数据边界、RSP、CoBench/AECI、Cyber、Agent safety、prompt injection、alignment、OSWorld/ProgramBench/Terminal-Bench 和 multi-agent 评测；具体数字必须绑定 snapshot、safeguards、fallback、工具、环境和 verifier。
3. [Artificial Analysis Claude Opus 5.5](https://artificialanalysis.ai/models/claude-opus-5-5)：第三方 canonical 条目；DataCurve 当前没有精确 `mini_swe_agent_claude_opus_5_5_*` 行，不能迁移其他 Claude 的 DeepSWE 结果。

4. [Claude Opus 5.5 model page](https://platform.claude.com/docs/en/models/opus-5-5/overview.md)、[What's new](https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5.md) 和 [Migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide.md)：官方 API contract 资料，记录 always-on adaptive thinking、thinking block binding、forced tool choice/computer toolset 变更、progress blocks、compaction 和 inline tools；不是模型架构论文。
5. [Fast mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode)：同一模型的更快 inference configuration，记录 `speed`、`usage.speed`、独立限流和 premium pricing；不能当作新的 checkpoint 或独立 benchmark 模型。

阅读重点：把 token efficiency、effort、fallback、thinking binding、工具宿主、verifier 和实际执行模型写进评测 manifest；不要从发布方成本下降或 API 行为推导参数、架构或训练方法。

System Card 的面试重点包括：Terminal-Bench 4.0 `66.36%`（xhigh、Claude Code `--bare`、5 trials）、OSWorld 2.0 partial/strict `81.8%/48.7%`、CoBench 2.1 `55.8%`、AECI `169.36`、关闭 cyber safeguards 后的 ACE `301/410 = 73.4%`、Gray Swan IPI k=1/10/15 `0.1%/0.7%/1.0%`，以及五 Agent team 的 derived-latency speedup。它们是发布方系统评测证据，不是独立论文复现或生产 SLO；System Card 也明确列出早期 snapshot、长轨迹、多 Agent、语言差异和模拟环境真实性等限制。

## GPT-6 Sol：官方模型与运行时资料

1. [Artificial Analysis GPT-6 Sol](https://artificialanalysis.ai/models/gpt-6-sol)：两个排行榜中的 AA canonical 锚点；max 配置的 Intelligence Index、速度、成本和 context 是第三方/provider 字段。DataCurve 当前没有精确 `mini_swe_agent_gpt_6_sol_*` 行，不能迁移其他 GPT 的 Agent 结果。
2. [GPT-6 Sol model page](https://developers.openai.com/api/docs/models/gpt-6-sol.md)：确认 `gpt-6-sol`、复杂 coding/agentic workflows、text/image input、1.05M context、922K maximum input、128K maximum output、`none`--`max` effort、Responses/Chat Completions/Batch、工具目录和 272K whole-request pricing threshold。
3. [Reasoning models](https://developers.openai.com/api/docs/guides/reasoning.md)：GPT-6 family 的 `standard/pro` mode、mode/effort 独立、reasoning token、`incomplete` 预算边界、会话内 `configuration_update` 及其与 compaction/truncation 的约束。
4. [Agents](https://developers.openai.com/api/docs/guides/agents.md)：Agents API 托管 Codex harness，Agents SDK 管理应用侧 loop/storage/approval/runtime，Responses API 直接管理 response/history/tool loop；三类 session/conversation/sandbox 不可混同。
5. [Using tools](https://developers.openai.com/api/docs/guides/tools.md)：function calling、web/file search、MCP、skills、shell、computer use、tool search 和 programmatic tool calling 的通用平台接口。工具列表是 capability surface，不是模型架构证据。
6. [Compaction](https://developers.openai.com/api/docs/guides/compaction.md)：server-side `context_management.compact_threshold`、encrypted compaction item、stateless output replay、`previous_response_id` chaining 和 standalone `/responses/compact` canonical context。

7. [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md) 与 [`gpt6_sol_contract_audit.py`](research/model-update-2026-09/code/gpt6_sol_contract_audit.py)：1,024 token minimum、30m toy TTL、最多四个 explicit breakpoints、configuration update 保持原 prefix、compaction 后可能 cache miss；toy 还验证 permission/executor/verifier、幂等 replay、预算 incomplete 和 272K whole-request pricing。
8. [API data residency guide](https://developers.openai.com/api/docs/guides/your-data.md)：GPT-6 Sol/Luna 的 EU residency 仅对 Standard processing 的 Responses/Chat Completions 有明确支持；regional storage 与 regional processing、system data 与 customer content、OpenAI endpoint 与 Remote MCP 第三方策略必须分账。

当前没有 GPT-6 Sol 专属参数、架构、完整训练/后训练 recipe、system card、公开权重、独立技术报告或生产 benchmark 复现；本条目服务于模型合同、长上下文预算、Agent 状态和 serving 评测方法。toy 的证据等级是 `local_protocol_toy`，不是 API 验收或模型质量结果。

## GPT-6 Luna：官方模型合同与 family runtime

1. [Artificial Analysis GPT-6 Luna](https://artificialanalysis.ai/models/gpt-6-luna)：AA canonical `gpt-6-luna`、max 配置和第三方 Intelligence/速度/成本字段；当前 DataCurve 没有精确 `mini_swe_agent_gpt_6_luna_*` 行，不能迁移相邻 GPT 的 Agent 结果。
2. [GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna.md)：确认 focused/high-volume 定位、`gpt-6-luna`、1.05M context、922K maximum input、128K maximum output、2026-05-18 knowledge cutoff、六档 `reasoning.effort`、Responses/Chat Completions/Batch、Responses 工具目录和 `$0.10/$0.50` input/output 价格合同；这些是服务合同，不是参数或架构证据。
3. [Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md)、[Agents](https://developers.openai.com/api/docs/guides/agents.md)、[Using tools](https://developers.openai.com/api/docs/guides/tools.md)、[Compaction](https://developers.openai.com/api/docs/guides/compaction.md)：复用 GPT-6 family 的 mode/effort、`configuration_update`、tool capability surface、runtime ownership 和 opaque compaction replay 证据。

当前没有 GPT-6 Luna 专属参数、架构、完整训练/后训练 recipe、system card、公开权重、独立技术报告、精确 DataCurve Agent 结果或生产 benchmark 复现；本条目服务于 sibling 路由、长上下文预算、单位成功成本、Agent 状态和 serving 评测方法。

## 2026-09-24 Kimi K3：vLLM v0.30.0 stable/runtime 来源

- [vLLM v0.30.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)：release commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，2026-09-22 发布；[PyPI metadata](https://pypi.org/pypi/vllm/json) 为 13,218 bytes / SHA-256 `43020551808911e4cabfca5ea71951766c25101b3817a10d306d88fe42d860b8`。x86_64 wheel metadata 为 314,883,777 bytes / SHA-256 `ef52ee58c410ead0b8afb190838fa4cbcb52075596f67862a03859d984966ac4`；本轮未下载/安装。
- 固定 [v0.30.0 registry](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/registry.py)（64,420 bytes / `a08a98aaae52ced32226aa647f58d682ac600b9572651a9b377b97846bc99212`）、[NVIDIA K3 model](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/models/kimi_k3/nvidia/model.py)（88,517 bytes / `27cbd7f0dceb493cf20e29056c3e1f00cddfae2074a5bd4bde52d6a6b73a38fe`）和 [K3 DSpark MLA](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/models/kimi_k3/nvidia/dspark_mla.py)（20,001 bytes / `ba5c555c8e3e392a0b69af91a371488e857cdef7685749853793bd9eda4aa793`）。对照 v0.29.0 tag：K3 stable registry/model/DSpark entry 已存在，故 v0.30.0 属实现演进，不是首次支持。
- 技术点：MegaMoE IPC transformed-weight 零拷贝复用和 raw packed-weight 释放；多次 streamed `load_weights` 后在 post-load hook finalize；PP auxiliary hidden state 与 AttnRes 边界 gate；DSpark grouped context-KV dtype/scale/layout gate；KDA SSM cache dtype。全部是 vLLM runtime source evidence，不是 K3 新训练算法或目标硬件验收。

## Claude Opus 5：Prompt-Injection 评测方法相关论文

- [How vulnerable are AI agents to indirect prompt injections? Insights from a Large-Scale Public Competition](https://arxiv.org/abs/2603.15714)：System Card 引用的 Gray Swan 大规模公开 red-team competition 背景；卡片据此构造 IPI 场景/攻击集。它不是 Anthropic 的 Opus 5 技术报告，也不代表 Opus 5 作者发布的模型论文。
- [The attacker moves second: Stronger adaptive attacks bypass defenses against LLM jailbreaks and prompt injections](https://arxiv.org/abs/2510.09023)：用于理解 adaptive attacker 与静态攻击集的差异；模型安全结果仍要绑定攻击预算、scenario set、产品 safeguards 和 harness。

Opus 5 研究笔记的 arXiv 精确标题检索仍未发现 Opus 5 专属技术报告；以上为 System Card 所引用的评测背景资料，不作为模型架构或训练证据。

## Qwen3.5-Omni Technical Report：ARIA 与实时多模态语音

1. [Qwen3.5-Omni Technical Report](https://arxiv.org/abs/2604.15804v2)（Qwen Team，v2，2026-04-17）：沿 Artificial Analysis 已有的 Qwen3.5-Omni-Plus/Flash 条目，精读 Hybrid MoE Thinker-Talker、AuT 6.25 Hz 编码、TM-RoPE + 秒级显式 timestamp、RVQ/MTP/Code2Wav 与 ARIA。
2. 阅读 ARIA 时抓住其核心约束：text 与 speech token 合成一个交错序列，任意前缀的累计 speech:text token ratio 不超过样本级全局 ratio；对照固定 interleave rate 和 MFA alignment，讨论 tokenizer 速率跨语言不均衡时的流式同步。
3. 对照 `S1 encoder alignment -> S2 ~4T multimodal tokens -> S3 262,144 context` 与 Thinker specialist/on-policy distillation、interaction-aligned RL、Talker DPO/GSPO/speaker fine-tuning。报告的 100M+ 小时总体音视频、AuT 40M 小时和 Talker 20M+ 小时不可简单相加。
4. 将作者报告的 215 项任务、内部 vLLM 首包数据、AA provider 指标与本地/独立实测分开；Plus 与 Flash 使用不同部署资源，不做无条件横向性能结论。
5. 官方 API 入口：[Alibaba Cloud Model Studio Qwen-Omni](https://help.aliyun.com/zh/model-studio/user-guide/qwen-omni)。示例模型 ID `qwen3.5-omni-plus` 与 `stream=True` 是托管 API 使用合同，不证明模型内部机制或已通过真实 endpoint probe。
6. 完整证据索引见 [`qwen3.5-omni-source-notes.md`](research/model-update-2026-09/qwen3.5-omni-source-notes.md) 和[第二十一册第 93 章](book-21-transformer-architecture-evolution/chapters/93-qwen3.5-omni-aria-aut-timestamp与流式语音.md)。DataCurve 当前无精确 Qwen3.5-Omni Agent 行。

## Qwen3.6-35B-A3B：混合注意力与 Thinking Preservation

1. [Artificial Analysis Qwen3.6-35B-A3B Reasoning](https://artificialanalysis.ai/models/qwen3-6-35b-a3b) 与 [Non-reasoning](https://artificialanalysis.ai/models/qwen3-6-35b-a3b-non-reasoning)：榜单发现入口，两种配置合并为一个模型锚点。DataCurve 当前快照没有精确 Qwen3.6 Agent 行。
2. [QwenLM/Qwen3.8 官方仓库固定 README](https://github.com/QwenLM/Qwen3.8/blob/2ea10dc725823bf7c3e21ce8557cbe15245132ae/README.md)：仓库发布记录将 Qwen3.6-35B-A3B 标为 2026-04-16 发布，并链接官方博客。
3. [Qwen3.6-35B-A3B ModelScope 模型卡](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B)：固定 README revision `913c459c5c83fa016a0e54a52e5b95f6c894e0fe`。固定 [config](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B/resolve/1a5ae24e867f8d82388070d3f61590158a01d15c/config.json) 与 [chat template](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B/resolve/1a5ae24e867f8d82388070d3f61590158a01d15c/chat_template.jinja)：前者核对架构字段，后者明确 `preserve_thinking` 与 `enable_thinking` 的渲染边界。
4. 模型卡称 35B total/3B activated、40 层、每组 3×Gated DeltaNet + 1×Gated Attention、256 experts（8 routed + 1 shared），native context 262,144、YaRN 可扩展约 1,010,000，并声明 MTP multi-step training。
5. `preserve_thinking` 让历史 assistant reasoning block 继续进入 prompt；它不是跨会话 memory。Qwen 对减少重复推理/KV cache 利用率的描述是发布方 claim，保留历史也可能增加 token/KV 成本；须与 `enable_thinking` 的本轮生成控制分开。
6. [Qwen 官方发布博客](https://qwen.ai/blog?id=qwen3.6-35b-a3b) 正文于 2026-09-28 通过 Qwen 文章 API 取得：JSON 94,678 bytes / SHA-256 `d287402f27a6ffa3226466aa57407310f670eb73a6d72bcfef03597a94e35d49`（含动态 request_id）；HTML 正文 91,941 bytes / `706889d145ea17b8c8234c4cda35b00fdecc0b6bcb9e1f5f20d2ed3ff9e15ed1`。博客报告发布方 benchmark，并给出 SWE-bench Pro 修订、Terminal-Bench 资源/重复次数、SkillsBench 子集、TAU3/VITA judge、MCPMark/MCP-Atlas 工具和评测器条件；这些适合作为 evaluator/harness 审计素材，不是独立复现或新架构披露。
7. 博客将开源 checkpoint 的百炼 API 名称写作 `qwen3.6-flash`，并给出 OpenClaw `contextWindow=131072` / `maxTokens=16384` 示例；alias 与客户端预算不等于新增模型候选、native context 或实测 API 上限。文章元数据日期为 2026-04-15 +08，与榜单/仓库标注 2026-04-16 并列保留；本项目未做真实 endpoint probe。

8. [arXiv:2609.27321v1《Verifiable Hidden Dynamics Play: Generating Agentic RL Environments from Solved Mechanisms》](https://arxiv.org/abs/2609.27321v1)，评论标为 “Qwen Technical Report”，2026-09-23。先采样并求解数学机制，冻结 optimum/default references 与 normalized reward，再由 frozen setter 合成 corpus-grounded stateful tools；论文报告 3,300 environments、Qwen3.6-35B-A3B 的 34-step GRPO 与 agentic diagnostic mean `0.204 → 0.815`。Replay audit 覆盖 8/11 families；one training run / one evaluation seed，结果均为论文作者报告。

本轮取得了 Agent RL 环境方法技术报告，但不是完整基础模型预训练 recipe。早期 arXiv 搜索快照没有列出此报告，现已由 v1 论文页核验并纠正记录。完整证据与未知项见 [`qwen3.6-35b-a3b-source-notes.md`](research/model-update-2026-09/qwen3.6-35b-a3b-source-notes.md)，教学映射为第二十一册第 83 章 83.15、第二十册第 19 章 19.41 / 第 21 章 21.30、第二十四册第 61 章 61.32。

## Qwen3.6-27B：GDN Tree-Scan serving 预印本

1. [Artificial Analysis Qwen3.6-27B](https://artificialanalysis.ai/models/qwen3-6-27b) 的 Reasoning / Non-reasoning 条目合并为同一模型锚点；[DataCurve DeepSWE](https://deepswe.datacurve.ai/) 当前无精确 Qwen3.6-27B Agent 行。
2. [Qwen 官方 ModelScope 模型卡](https://www.modelscope.cn/models/Qwen/Qwen3.6-27B)、固定 config 与 chat template 支持 27B dense、3×Gated DeltaNet + 1×Gated Attention 周期，以及共享的 preserve_thinking 序列化接口。
3. [GDN Tree-Scan: Served Tree Verification for Recurrent-Hybrid Language Models](https://arxiv.org/abs/2609.23900v1) 是 Zhiyuan Ma 的外部单作者预印本（2026-09-20），不是 Qwen 技术报告。其 vLLM 路线把 FA2 tree-bias、branch-local GDN scan/replay、MTP tree draft 和 accepted-chain-only state publication 合为 serving verifier。
4. 预印本在 Qwen3.6-27B-FP8、B=1、temperature 0.6、四个 SWE/Codex tasks 上报告 +17.2% committed tokens/event、+27.0% token-weighted decode TPS，但 per-request-equal 仅 +4.0%；没有给出可推广的 task-wall 提速。40-turn p-rescore 只是与 native recurrent-oracle flip floor 的有限样本比较，不是 full distribution-distance proof；B=4、更多 seeds 与 request-cluster bootstrap 仍待补。
5. 实现/测量仓库的固定入口为 [Lumo_FlyWheel commit 55f55854328b37f262e97d57b5863d8fadd7ff76](https://github.com/MaCoredroid/Lumo_FlyWheel/tree/55f55854328b37f262e97d57b5863d8fadd7ff76)。本项目未运行 pinned code、加载权重或独立复现；所有数字须保留为作者报告。
6. [Qwen3.6-27B 官方发布博客](https://qwen.ai/blog?id=qwen3.6-27b) 的正文于 2026-09-28 经官方文章 API 恢复（嵌入 HTML 91,758 bytes / SHA-256 `7748e75a7c5a47943d6abe4e6415cb6b8d4749713eeff39323eb831f2d8ae367`）。博客提供发布方 benchmark、`preserve_thinking` 和 Agent 客户端示例，不是独立技术报告；OpenClaw 的 128K/16K 是 harness 配置，不是 native context/output 规格。正文/API 证据与未做真实 endpoint probe 的边界见研究笔记。

arXiv title-field 查询 ti:Qwen3.6 当前返回 0 条，但 all-field 查询有多篇把 Qwen3.6-27B 当测试模型的工作；这不是“没有相关研究”的证据。完整技术摘录和边界见 [qwen3.6-27b-source-notes.md](research/model-update-2026-09/qwen3.6-27b-source-notes.md)，教学映射为第二十一册第 83 章 83.16.4–83.16.5、第二十四册第 61 章 61.31–61.32。
