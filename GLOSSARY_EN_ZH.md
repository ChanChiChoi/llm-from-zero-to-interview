# 中英文术语表

## Core Terms

Language Model：语言模型

Large Language Model / LLM：大语言模型

Next-Token Prediction：下一个 token 预测

Autoregressive Model：自回归模型

Token：词元 / 记号

Tokenizer：分词器 / token 化器

Vocabulary：词表

Token ID：token 编号

Byte Pair Encoding / BPE：字节对编码

WordPiece：WordPiece 子词切分方法

SentencePiece：SentencePiece 分词方法 / 工具

Byte-Level Tokenization：字节级 tokenization

Special Token：特殊 token

Beginning of Sequence / BOS：序列开始标记

End of Sequence / EOS：序列结束标记

Padding Token / PAD：填充 token

Out-of-Vocabulary / OOV：词表外

Embedding：嵌入 / 向量表示

Embedding Model：嵌入模型 / 向量表示模型

Bi-Encoder：双塔编码器

Cross-Encoder：交叉编码器

Contrastive Learning：对比学习

In-Batch Negatives：批内负样本

Hard Negative：困难负样本

False Negative：伪负样本

Cosine Similarity：余弦相似度

Dot Product：点积 / 内积

L2 Distance：L2 距离 / 欧氏距离

Approximate Nearest Neighbor / ANN：近似最近邻

Hierarchical Navigable Small World / HNSW：分层可导航小世界图索引

Inverted File Index / IVF：倒排文件索引

Product Quantization / PQ：乘积量化

Token Embedding：token 嵌入

Embedding Matrix：嵌入矩阵

Position Embedding：位置嵌入

Positional Encoding：位置编码

Sinusoidal Positional Encoding：正弦位置编码

Rotary Position Embedding / RoPE：旋转位置编码

RoPE Scaling：RoPE 缩放

Long Context：长上下文

Context Window：上下文窗口

Position Extrapolation：位置外推

Position Interpolation：位置插值

Lost in the Middle：中间信息遗失现象

Long Context Continued Pretraining：长上下文继续预训练

Context Parallelism：上下文并行

Sequence Parallelism：序列并行

Needle-in-a-Haystack：长上下文针尖测试

Position Robustness：位置鲁棒性

Evidence Position：证据位置

Multi-Evidence Reasoning：多证据推理

Attribution：归因 / 证据归属

Citation Accuracy：引用准确率

Faithfulness：忠实性 / 基于证据的一致性

Unsupported Claim：无依据声明

Groundedness：基于证据的程度 / groundedness

Atomic Claim：原子声明

Abstention：拒答 / 放弃回答

Abstention Accuracy：拒答准确率

Unanswerable Question：不可回答问题

Claim-Level Evaluation：声明级评估

Citation Hallucination：引用幻觉

Error Attribution：错误归因

Benchmark Contamination：基准污染

Memory-Augmented LLM：记忆增强大模型

Weight Tying：权重共享

Contextual Representation：上下文表示

Decoder-Only Transformer：仅解码器 Transformer

MiniGPT：最小 GPT 教学模型

Language Modeling Head / LM Head：语言模型输出头

Generation：生成

Greedy Decoding：贪心解码

Sampling：采样

Checkpoint：检查点

Pretraining：预训练

Base Model：基础模型

Instruct Model：指令模型

Corpus：语料库

Data Cleaning：数据清洗

Deduplication：去重

Benchmark Contamination：基准污染

Data Packing：数据打包

Scaling Law：规模定律

Chinchilla Scaling：Chinchilla 规模定律

Compute-Optimal Training：计算最优训练

Data Mixture：数据配比

Data Quality：数据质量

Synthetic Data：合成数据

Near Deduplication：近重复去重

Validation Loss：验证损失

Training Throughput：训练吞吐

Distributed Training：分布式训练

Data Parallelism：数据并行

Distributed Data Parallel / DDP：分布式数据并行

ZeRO：零冗余优化器

Fully Sharded Data Parallel / FSDP：全分片数据并行

Tensor Parallelism：张量并行

Pipeline Parallelism：流水线并行

Activation Checkpointing：激活检查点 / 激活重计算

All-Reduce：全归约

Reduce-Scatter：归约散射

All-Gather：全收集

Model FLOPs Utilization / MFU：模型 FLOPs 利用率

Hidden Size：隐藏维度

Memorization：记忆

Generalization：泛化

Hallucination：幻觉 / 编造

Conditional Probability：条件概率

Joint Probability：联合概率

Chain Rule of Probability：概率链式法则

Teacher Forcing：教师强制

Label Shift：标签错位 / 标签右移

Logits：未归一化分数

Softmax：归一化指数函数

Causal Language Model：因果语言模型

Decoding：解码

Gradient：梯度

Gradient Descent：梯度下降

Backpropagation：反向传播

Chain Rule：链式法则

Learning Rate：学习率

Learning Rate Schedule：学习率调度

Warmup：预热

Cosine Decay：余弦衰减

Optimizer：优化器

Stochastic Gradient Descent / SGD：随机梯度下降

Momentum：动量

Adam：Adam 优化器

AdamW：AdamW 优化器

Weight Decay：权重衰减

Decoupled Weight Decay：解耦权重衰减

Gradient Norm：梯度范数

Gradient Clipping：梯度裁剪

Vanishing Gradient：梯度消失

Exploding Gradient：梯度爆炸

Cross Entropy：交叉熵

Maximum Likelihood Estimation / MLE：最大似然估计

Negative Log-Likelihood / NLL：负对数似然

KL Divergence：KL 散度 / 相对熵

Perplexity：困惑度

Entropy：熵

Self-Attention：自注意力

Query：查询向量

Key：键向量 / 索引向量

Value：值向量 / 内容向量

Attention Weight：注意力权重

Contextual Representation：上下文表示

Causal Self-Attention：因果自注意力

Long-Range Dependency：长距离依赖

Scaled Dot-Product Attention：缩放点积注意力

Full Attention：全注意力

Local Attention：局部注意力

Sliding Window Attention：滑动窗口注意力

Sparse Attention：稀疏注意力

Linear Attention：线性注意力

State Space Model / SSM：状态空间模型

Structured State Space Model / S4：结构化状态空间模型 S4

Mamba：选择性状态空间序列模型 Mamba

Selective Scan：选择性扫描机制

RWKV：结合 RNN 式状态递推与 Transformer 式训练目标的序列模型路线

RetNet / Retentive Network：保留网络 / RetNet 架构

Hyena：基于隐式长卷积的长序列建模架构

Hybrid Architecture：混合架构，如 Attention + SSM + Convolution

FlashAttention：FlashAttention 高效注意力实现

Attention Complexity：注意力复杂度

Multi-Head Attention / MHA：多头注意力

Attention Head：注意力头

Head Dimension：头维度

Multi-Query Attention / MQA：多查询注意力

Grouped-Query Attention / GQA：分组查询注意力

Output Projection：输出投影

Transformer Block：Transformer 模块 / Transformer 块

Feed-Forward Network / FFN：前馈网络

MLP：多层感知机

Residual Connection：残差连接

Layer Normalization / LayerNorm：层归一化

RMSNorm：均方根归一化

Pre-LN：前置归一化

Post-LN：后置归一化

SwiGLU：门控激活结构 SwiGLU

Causal Mask：因果掩码

Padding Mask：填充掩码

Causal LM：因果语言模型

Masked Language Model / MLM：掩码语言模型

Information Leakage：信息泄漏

Key-Value Cache / KV Cache：键值缓存

Post-Training：后训练

Instruction Tuning：指令微调

Supervised Fine-Tuning / SFT：监督微调

Full Fine-Tuning：全参微调

Parameter-Efficient Fine-Tuning / PEFT：参数高效微调

Low-Rank Adaptation / LoRA：低秩适配

Quantized LoRA / QLoRA：量化低秩适配

Chat Template：聊天模板

Loss Mask：损失掩码

Ignore Index：忽略索引

System Prompt：系统提示词

User Message：用户消息

Assistant Response：助手回答

End-of-Turn Token：轮次结束 token

Synthetic Instruction Data：合成指令数据

Catastrophic Forgetting：灾难性遗忘

Overfitting：过拟合

Padding：填充

Truncation：截断

Packing：打包

Reinforcement Learning from Human Feedback / RLHF：基于人类反馈的强化学习

Preference Data：偏好数据

Chosen Response：被选择回答 / 偏好回答

Rejected Response：被拒绝回答 / 非偏好回答

Reward Model：奖励模型

Pairwise Ranking Loss：成对排序损失

Policy Model：策略模型

Reference Model：参考模型

Proximal Policy Optimization / PPO：近端策略优化

KL Penalty：KL 惩罚项

Reward Score：奖励分数

Pairwise Accuracy：成对排序准确率

Reward Calibration：奖励校准

Distribution Shift：分布偏移

Goodhart's Law：古德哈特定律

Adversarial Evaluation：对抗式评估

Over-Optimization：过度优化

Direct Preference Optimization / DPO：直接偏好优化

DPO Loss：DPO 损失

Beta in DPO：DPO 中的 beta 超参数

Length Bias：长度偏差

Over-Refusal：过度拒答

Reward Hacking：奖励黑客 / 奖励作弊

Benchmark Contamination：基准污染

Mechanistic Interpretability：机制可解释性

Speculative Decoding：推测解码

Inference：推理

Decoding：解码

Greedy Decoding：贪心解码

Beam Search：束搜索

Temperature：温度参数

Top-k Sampling：Top-k 采样

Top-p Sampling / Nucleus Sampling：Top-p 采样 / 核采样

Repetition Penalty：重复惩罚

Presence Penalty：存在惩罚

Frequency Penalty：频率惩罚

Stop Token：停止 token

Best-of-N：N 个候选中选优

Self-Consistency：自一致性

Speculative Decoding：推测解码

Speculative Sampling：推测采样

Draft Model：草稿模型 / 提案模型

Target Model：目标模型

Acceptance Rate：接受率

Acceptance-Rejection Sampling：接受-拒绝采样

Corrected Distribution：修正分布

Multi-Token Prediction：多 token 预测

Medusa：Medusa 多 token 解码方法

Medusa Heads：Medusa 预测头

EAGLE：EAGLE 推测解码方法

Feature-Level Prediction：特征层预测

Tree Attention：树状注意力

Candidate Tree：候选树

KV Cache：键值缓存

KV Cache Quantization：KV Cache 量化

K Cache：Key 缓存

V Cache：Value 缓存

Per-Token Quantization：按 token 量化

Per-Head Quantization：按注意力头量化

KV Scale：KV 缩放系数

Mixed-Precision KV Cache：混合精度 KV Cache

LLM Serving：大模型在线推理服务

API Gateway：API 网关

Request Router：请求路由器

Scheduler：调度器

Inference Worker：推理工作节点

Streamer：流式返回模块

Rate Limit：限流

Quota：配额

Backpressure：反压

Fallback Model：降级模型 / 备用模型

Observability：可观测性

Tracing：链路追踪

Tail Latency：尾延迟

Capacity Planning：容量规划

Training Cost：训练成本

Inference Cost：推理成本

GPU Hour：GPU 小时

Token Cost：token 成本

Input Token Cost：输入 token 成本

Output Token Cost：输出 token 成本

Cost per Token：每 token 成本

Cost per Successful Answer：每个成功回答成本

Unit Economics：单位经济模型

Marginal Cost：边际成本

Redundancy Factor：冗余系数

Cache Hit Rate：缓存命中率

On-Device Deployment：端侧部署

Edge Deployment：边缘部署

Small Language Model / SLM：小语言模型

Knowledge Distillation：知识蒸馏

Pruning：剪枝

Structured Pruning：结构化剪枝

LoRA Merge：LoRA 权重合并

On-Device Runtime：端侧运行时

Neural Processing Unit / NPU：神经网络处理单元

WebGPU：浏览器 GPU 计算接口

Core ML：Apple Core ML 框架

ONNX Runtime：ONNX 运行时

GGUF：GGUF 模型文件格式

Cold Start：冷启动

Power Consumption：功耗

Thermal Throttling：过热降频

Hybrid Edge-Cloud Inference：端云协同推理

Prefill：预填充阶段

Decode：解码阶段

Time to First Token / TTFT：首 token 延迟

Time per Output Token / TPOT：每输出 token 延迟

Throughput：吞吐量

Latency：延迟

Dynamic Batching：动态批处理

Continuous Batching：连续批处理

In-Flight Batching：飞行中批处理 / 运行中批处理

Iteration-Level Batching：迭代级批处理

Static Batching：静态批处理

Token Budget：token 预算

Chunked Prefill：分块预填充

PagedAttention：分页注意力

KV Block：KV 缓存块

Block Table：块映射表

Logical Block：逻辑块

Physical Block：物理块

Internal Fragmentation：内部碎片

External Fragmentation：外部碎片

Prefix Cache：前缀缓存

Prompt Cache：提示词缓存

FlashAttention：FlashAttention 高效注意力实现

IO-Aware Optimization：IO 感知优化

Tiling / Blocking：分块计算

Online Softmax：在线 softmax

High Bandwidth Memory / HBM：高带宽显存

SRAM：静态随机存取存储器 / 片上高速存储

Fused Kernel：融合算子

Memory Bandwidth Bottleneck：显存带宽瓶颈

Quantization：量化

Post-Training Quantization / PTQ：后训练量化

Quantization-Aware Training / QAT：量化感知训练

Fake Quantization：伪量化

Straight-Through Estimator / STE：直通估计器

Scale：量化缩放系数

Zero Point：量化零点

Symmetric Quantization：对称量化

Asymmetric Quantization：非对称量化

Per-Tensor Quantization：按张量量化

Per-Channel Quantization：按通道量化

Group-Wise Quantization：分组量化

Mixed-Precision Quantization：混合精度量化

GPTQ：GPTQ 后训练量化方法

Activation-Aware Weight Quantization / AWQ：激活感知权重量化

Calibration Data：校准数据

Outlier：异常大值 / 离群值

Dequantization：反量化

FP32：32 位浮点数

FP16：16 位浮点数

BF16：Brain Float 16 位浮点数

FP8：8 位浮点数

INT8：8 位整数

INT4：4 位整数

Weight-Only Quantization：仅权重量化

Multimodal Learning：多模态学习

Vision-Language Model / VLM：视觉语言模型

Retrieval-Augmented Generation / RAG：检索增强生成

Document Parsing：文档解析

Chunking：文档切分

Metadata：元数据

Vector Database：向量数据库

Vector Index：向量索引

Hybrid Retrieval：混合检索

BM25：BM25 关键词检索算法

Query Rewrite：查询改写

Reranker：重排序模型

Reranking：重排序

Pointwise Ranking：逐点评分排序

Pairwise Ranking：成对排序

Listwise Ranking：列表级排序

Hit@k：前 k 命中率

Context Precision：上下文精确性

Metadata-Aware Reranking：元数据感知重排序

Context Construction：上下文构造

Grounding：基于证据生成 / 锚定

Access Control：访问控制

Document Freshness：文档新鲜度

Recall@k：前 k 召回率

Precision@k：前 k 精确率

Mean Reciprocal Rank / MRR：平均倒数排名

Normalized Discounted Cumulative Gain / nDCG：归一化折损累计增益

Agent：智能体

Hallucination：幻觉

Factuality：事实性

Benchmark：基准测试

Human Evaluation：人工评估

LLM-as-a-Judge：用大模型作评审

Verifier：验证器

Citation Grounding：引用依据对齐

Robustness：鲁棒性

AI Safety：AI 安全

Alignment：对齐

Helpful：有帮助

Honest：诚实

Harmless：无害

Jailbreak：越狱攻击

Prompt Injection：提示注入

Red Teaming：红队测试

Model Card：模型卡
