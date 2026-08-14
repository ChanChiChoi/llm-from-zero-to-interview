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
6. [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3)：KDA、Gated MLA、NoPE 和 total/active 参数入口。
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
