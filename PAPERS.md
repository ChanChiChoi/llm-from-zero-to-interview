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
