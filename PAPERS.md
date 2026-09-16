# 论文路线

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
6. [Kimi K3 官方发布文章](https://www.kimi.com/en/blog/kimi-k3)：KDA、Gated MLA、规模与长上下文的发布披露入口；NoPE、层比例和 total/active 参数仍待模型卡/技术报告核验。
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
8. [Claude Fable 5.1 overview](https://platform.claude.com/docs/en/models/fable-5-1/overview)：官方模型页，记录 1M context、adaptive always-on thinking、preserved thinking、beta 状态协议和平台/价格字段；不是架构论文。
9. [Claude Sonnet 5 overview](https://platform.claude.com/docs/en/models/sonnet-5/overview)、[system card](https://www.anthropic.com/claude-sonnet-5-system-card)：官方模型页和系统资料入口，记录 1M context、128K/300K 输出、Adaptive、平台与成本字段；本轮尚未读取系统卡全文，不把未核验训练或架构细节写成事实。
10. [Claude Haiku 4.5 overview](https://platform.claude.com/docs/en/models/haiku-4-5/overview)、[system card](https://www.anthropic.com/claude-haiku-4-5-system-card)：官方模型页和系统资料入口，记录 200K context、64K 输出、extended thinking、fastest 延迟字段、平台与成本字段；本轮尚未读取 system card 全文，不把未核验训练或架构细节写成事实。
11. [DeepSeek-R1-0528 Release](https://api-docs.deepseek.com/news/news250528)：官方 API 发布页，记录 2025/05/28、JSON/function calling、API 兼容性承诺和开源权重入口；benchmark 图片、模型卡、架构与训练细节仍待核验。

### GPT-6 Astra 官方接口资料

1. [GPT-6 Astra model page](https://developers.openai.com/api/docs/models/gpt-6-astra.md)：`gpt-6-astra` 的输入/输出模态、context window、maximum input/output、reasoning effort、Responses 工具与端点支持。
2. 本资料只证明模型页公开的接口和容量字段；参数规模、训练架构、数据、发布日期和技术报告仍需独立的一手来源，不从榜单条目反推。

### GLM-5.3 官方文档与长任务环境

1. [GLM-5.3 Guide](https://docs.z.ai/guides/llm/glm-5.3)：模型版本关系、1M context、128K output、推理档位、可执行环境与验证器流程描述。
2. [GLM-5.2 Guide](https://docs.z.ai/guides/llm/glm-5.2)：长任务定位和前代接口背景；正文未提供 SAO with compaction 的数学定义。
3. 证据边界：上述文档支持接口与高层训练流程，不支持参数量、完整训练配方或 SAO 全称/实现的确定结论。

### Kimi K3 发布与长任务 Harness

1. [Kimi K3 官方发布文章](https://www.kimi.com/en/blog/kimi-k3)：KDA、AttnRes、Stable LatentMoE、量化、视觉、长上下文、Agent 限制和评测脚注入口。
2. K3 文章中的技术和 benchmark 需绑定发布时的模型、harness、硬件、effort 与 fallback；Kimi Linear、Attention Residuals 论文只用于解释一般机制，不证明 K3 的完整配置。

### DeepSWE v1.1 评测快照

1. [DeepSWE](https://deepswe.datacurve.ai/)：长周期软件工程 benchmark 入口；本项目保存了页面更新时间为 2026-09-03 的 `/tmp/deepswe.html` 快照。
2. [`deepswe-snapshot-notes.md`](research/model-update-2026-09/deepswe-snapshot-notes.md)：整理 113 个任务、91 个仓库、5 种语言、统一 `mini-swe-agent` harness，以及 21 个配置的 Pass@1、区间、成本、输出 token 和 Agent steps。
3. 证据边界：这些是模型 revision + effort + Agent harness + 工具 + verifier + 运行策略的组合结果；不能直接与 Artificial Analysis、SWE-bench 或其他 harness 的结果拼接，也不能反推参数量、架构或训练方法。

### DeepSeek V4.1-Flash

1. [DeepSeek-V4.1-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)：固定 revision `dba1be0a40aa45a94ad051997016db3960a90277` 的模型卡，披露 CED、CSA2、FP4 KV、Engram、DSpark、原生多模态、训练规模和 Agent 评测协议；它是模型卡，不等于独立论文复现。
2. [DeepSeek-V4.1-Flash Technical Report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)：官方 51 页技术报告；本轮已逐页提取并核对 CED、CSA2、HSI、mHC、Engram、DSpark、FP4 KV、训练基础设施、SWA replay 和 Agent 评测边界。完整 kernel source、线上接受率和独立 profiling 仍待核验。
3. [DeepSeek-V4.1-Flash API Release](https://api-docs.deepseek.com/news/news260910)：API alias、兼容路由和服务端发布日期入口；路由说明带日期，不能替代权重身份。

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
6. [Artificial Analysis Qwen3.8 条目](https://artificialanalysis.ai/models/qwen3-8-flash-next)：候选发现来源；[DataCurve DeepSWE](https://deepswe.datacurve.ai/) 当前快照只检出 `qwen3.8-max`。

模型卡和报告可证明公开字段与作者披露的技术路线，不能自动证明完整生产 kernel、目标硬件 profiling、线上 acceptance rate、全系列训练配方或独立 benchmark。Flash-Next 报告中的速度、loss、稳定性和 benchmark 数字必须保留为发布方自报；Qwen3.8-Max 是官方说明基于 A95B 的 hosted version，不作为新的 open checkpoint。

### GLM-5.3-Flash 官方资料与混合 Serving

1. [GLM-5.3-Flash 官方模型文档](https://docs.z.ai/guides/vlm/glm-5.3-flash)：输入模态、thinking/tool streaming、模型定位和接入边界。
2. [GLM-5.3-Flash 官方博客](https://z.ai/blog/glm-5.3-flash)：hybrid linear+sparse attention、IndexPool、视觉 coding loop、SGLang/ReplaySSM、混合 cache 和 EPD 的发布方描述。
3. [GLM-5.3-Flash 模型卡](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a) 与 [固定配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json)：`320B/18B`、45 层、`34/11` layer types、MoE、IndexPool、mHC 和 vision 字段。
4. [Artificial Analysis 条目](https://artificialanalysis.ai/models/glm-5-3-flash) 与 [DataCurve DeepSWE](https://deepswe.datacurve.ai/)：分别提供 `max` 配置字段和 `mini-swe-agent` Agent 结果；两者不是同一 benchmark，也不是裸模型分数。

这些资料中没有公开 GLM-5.3-Flash 的完整训练报告或生产 kernel。`3.01x/4.44x` attention/KV、约 `3x` serving 和视觉 self-verification 必须标作发布方自报；研究笔记与正式章节见 [`glm-5.3-flash-source-notes.md`](research/model-update-2026-09/glm-5.3-flash-source-notes.md) 和 [`第二十一册第 84 章`](book-21-transformer-architecture-evolution/chapters/84-glm-5.3-flash混合注意力与视觉闭环.md)。
