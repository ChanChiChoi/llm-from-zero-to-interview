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

InfoNCE：对比预测编码损失 / InfoNCE 损失

CLIP：Contrastive Language-Image Pre-training / 图文对比预训练

CLIP Loss：CLIP 双向图文对比损失

Logit Scale：logit 缩放系数

In-Batch Negatives：批内负样本

Hard Negative：困难负样本

False Negative：伪负样本

Cosine Similarity：余弦相似度

Dot Product：点积 / 内积

L2 Distance：L2 距离 / 欧氏距离

Zero-Shot Classification：零样本分类

Recall@K：Top-K 召回率

Linear Algebra：线性代数

Vector：向量

Vector Norm：向量范数

Matrix Multiplication：矩阵乘法

Projection：投影

Orthogonality：正交性

Eigenvalue：特征值

Eigenvector：特征向量

Matrix Rank：矩阵秩

Singular Value：奇异值

Singular Value Decomposition / SVD：奇异值分解

Low-Rank Approximation：低秩近似

Low-Rank Compression：低秩压缩

Principal Component Analysis / PCA：主成分分析

Explained Variance Ratio：解释方差比例

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

Web-Scale Data Collection：Web 规模数据采集

Source Registry：数据源登记表 / 来源登记系统

Robots Exclusion Protocol / robots.txt：机器人排除协议 / robots.txt 访问规则

Web ARChive / WARC：网页归档格式

Provenance：数据血缘 / 来源追踪

Dataset Versioning：数据集版本管理

Data Governance：数据治理

Data Lineage：数据血缘

Dataset Manifest：数据集清单 / Manifest

Immutable Snapshot：不可变快照

Shard Checksum：分片校验和

Data Schema：数据 schema / 数据字段协议

Schema Evolution：Schema 演进

Deletion Request：删除请求

Dataset Datasheet：数据集说明书 / Datasheet

Dataset Card：数据集卡片

Model Card：模型卡片

Audit Log：审计日志

Terms of Service / ToS：服务条款

License Review：许可证审查

Code Data：代码数据

Math Data：数学数据

Domain Data：领域数据 / 专业领域数据

Specialized Data：专项数据

Unit Test Data：单元测试数据

Functional Correctness：功能正确性

Docstring-Code Pair：文档字符串与代码配对样本

Verifier Data：验证器数据

Answer Verification：答案验证

Process Supervision：过程监督

Domain Authority：领域权威性

Recency：时效性

Citation Metadata：引用元数据 / 出处元数据

Expert Audit：专家审计

Data Cleaning：数据清洗

Quality Score：质量分

Quality Classifier：质量分类器

Quality Filter：质量过滤器

Filtering False Positive：过滤误删 / 过滤假阳性

Filtering False Negative：过滤漏删 / 过滤假阴性

PII Filtering：个人可识别信息过滤

Secret Scanning：密钥扫描

Safety Filtering：安全过滤

Toxicity Filtering：毒性过滤

Data Filtering Audit：数据过滤审计

Threshold Calibration：阈值校准

Deduplication：去重

Exact Deduplication：精确去重 / 完全去重

Document-Level Deduplication：文档级去重

Substring Deduplication：子串级去重

Near Duplicate：近重复样本

Near Deduplication：近重复去重

MinHash：最小哈希

Locality-Sensitive Hashing / LSH：局部敏感哈希

SimHash：相似哈希

Jaccard Similarity：Jaccard 相似度

Hamming Distance：汉明距离

Train-Eval Overlap：训练集与评估集重叠

Canary：金丝雀探针 / 受控记忆探针

Memorization Risk：记忆风险

Benchmark Contamination：基准污染

Data Packing：数据打包

Scaling Law：规模定律

Chinchilla Scaling：Chinchilla 规模定律

Compute-Optimal Training：计算最优训练

Data Mixture：数据配比

Data Sampling：数据采样

Sampling Weight：采样权重

Temperature Sampling：温度采样 / 平滑采样

Upsampling：上采样

Downsampling：下采样

Effective Epoch：有效训练轮数 / 有效重复轮数

Mixture Ablation：数据配比消融实验

Dynamic Mixture：动态数据配比

Curriculum Schedule：课程式训练调度 / 阶段式采样计划

Data Quality：数据质量

Synthetic Data：合成数据

Validation Loss：验证损失

Training Throughput：训练吞吐

PyTorch Tensor：PyTorch 张量

Tensor Shape：张量形状

Dtype / Data Type：数据类型

Device：设备

CPU Tensor：CPU 张量

CUDA Tensor：CUDA 张量 / GPU 张量

Broadcasting：广播机制

Tensor Stride：张量步幅

Contiguous Tensor：连续内存张量

View：视图

Reshape：改变形状

Flatten：展平

Transpose：转置

Permute：维度重排

Unsqueeze：增加单维度

Squeeze：删除单维度

Expand：广播扩展视图

Repeat：重复拷贝

Matmul：矩阵乘法

Batched Matrix Multiplication / BMM：批量矩阵乘法

Einsum：爱因斯坦求和约定

Tensor Mask：张量掩码

Causal Mask：因果掩码

Padding Mask：填充掩码

Device Mismatch：设备不一致

Shape Mismatch：形状不匹配

Tensor Shape Audit：张量形状审计

Distributed Training：分布式训练

Rank：分布式进程全局编号

Local Rank：本机进程 / GPU 编号

World Size：分布式训练总进程数

Process Group：进程组 / 通信进程组

Data Parallelism：数据并行

Distributed Data Parallel / DDP：分布式数据并行

Gradient Synchronization：梯度同步

Communication Overlap：通信与计算重叠

Collective Operation：集合通信操作

torchrun：PyTorch 分布式启动命令

Rank 0 Checkpoint：rank 0 检查点保存

DDP no_sync：DDP 跳过梯度同步上下文

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

True Risk / Expected Risk：真实风险 / 期望风险

Empirical Risk：经验风险

Empirical Risk Minimization / ERM：经验风险最小化

Generalization Gap：泛化差距

Bias-Variance Trade-off：偏差-方差权衡

Underfitting：欠拟合

Regularization：正则化

Validation Set：验证集

Test Set：测试集

Hidden Test Set：隐藏测试集

Generalization Audit：泛化审计

Hallucination：幻觉 / 编造

Probability Theory：概率论

Random Variable：随机变量

Probability Distribution：概率分布

Conditional Probability：条件概率

Joint Probability：联合概率

Marginal Probability：边缘概率

Chain Rule of Probability：概率链式法则

Bayes' Theorem：贝叶斯公式

Prior：先验

Likelihood：似然

Posterior：后验

Evidence：证据 / 归一化因子

Bayesian Update：贝叶斯更新

Maximum A Posteriori / MAP：最大后验估计

Aleatoric Uncertainty：偶然不确定性 / 数据噪声不确定性

Epistemic Uncertainty：认知不确定性 / 模型知识不确定性

Confidence：置信度

Expected Calibration Error / ECE：期望校准误差

Brier Score：Brier 分数 / 概率预测均方误差

Selective Prediction：选择性预测

Abstention：拒答 / 暂不自动回答

Expectation：期望

Variance：方差

Teacher Forcing：教师强制

Label Shift：标签错位 / 标签右移

Logits：未归一化分数

Softmax：归一化指数函数

Causal Language Model：因果语言模型

Decoding：解码

Gradient：梯度

Gradient Descent：梯度下降

Backpropagation：反向传播

PyTorch Autograd：PyTorch 自动求导

PyTorch Module / nn.Module：PyTorch 模块 / 模型组织基类

Parameter / nn.Parameter：模型参数

Module Registration：模块注册

Submodule：子模块

ModuleList：模块列表

ModuleDict：模块字典

ParameterList：参数列表

ParameterDict：参数字典

Buffer：缓冲状态 / 非训练模型状态

Register Buffer：注册 buffer

Persistent Buffer：持久化 buffer

state_dict：状态字典 / 权重字典

load_state_dict：加载状态字典

Missing Key：缺失权重键

Unexpected Key：多余权重键

Strict Loading：严格加载

Train Mode：训练模式

Eval Mode：评估模式 / 推理模式

Forward Hook：前向 hook

Module Audit：模块组织审计

Dataset / PyTorch Dataset：数据集 / PyTorch 数据集

Map-style Dataset：可索引数据集

IterableDataset：流式可迭代数据集

DataLoader：数据加载器 / 批加载器

Collate Function / collate_fn：批组装函数

Default Collate：默认批组装逻辑

Sampler：采样器

Batch Sampler：批采样器

DistributedSampler：分布式采样器

Length Bucket：长度分桶

Padding Waste：padding 浪费率

Attention Mask：注意力掩码

Ignore Index：忽略标签索引

DataLoader Worker：数据加载 worker

worker_init_fn：worker 初始化函数

num_workers：DataLoader worker 数量

pin_memory：页锁定内存

persistent_workers：持久化 worker

prefetch_factor：预取因子

Data Pipeline Audit：数据管线审计

Automatic Differentiation：自动微分

Dynamic Computation Graph：动态计算图

Define-by-Run：运行时定义计算图

Requires Grad：需要梯度标记

Leaf Tensor：叶子张量

Non-Leaf Tensor：非叶子张量

Grad Function / grad_fn：梯度函数 / 反向函数

Backward：反向传播调用

Vector-Jacobian Product / VJP：向量-雅可比积

Retain Graph：保留计算图

Retain Grad：保留中间梯度

Chain Rule：链式法则

Learning Rate：学习率

Learning Rate Schedule：学习率调度

Warmup：预热

Cosine Decay：余弦衰减

Learning Rate Decay：学习率衰减

Optimizer：优化器

Optimizer State：优化器状态

Stochastic Gradient Descent / SGD：随机梯度下降

Momentum：动量

Adam：Adam 优化器

AdamW：AdamW 优化器

Weight Decay：权重衰减

Decoupled Weight Decay：解耦权重衰减

Gradient Norm：梯度范数

Gradient Clipping：梯度裁剪

Global Batch Size：全局批大小

Micro Batch Size：微批大小

Gradient Accumulation：梯度累积

Zero Grad：梯度清零

Set Grad to None：将梯度设为 None

Training Loop：训练循环

Training Step：训练步 / 参数更新步

Micro Step：微步 / 微批次反传步

Optimizer Step：优化器更新步

Scheduler Step：学习率调度更新步

Evaluation Loop：验证循环 / 评估循环

Checkpoint Resume：检查点恢复训练

Global Step：全局训练步

Raw Loss：原始 loss / 未缩放 loss

Scaled Loss：缩放后 loss

Non-Finite Loss：非有限 loss / NaN 或 Inf loss

Training Loop Audit：训练循环审计

Tensor Metadata：张量元信息

Debug Tensor Metadata：张量元信息调试

Finite Check：有限值检查

NaN / Not a Number：非数值

Inf / Infinity：无穷值

Forward Hook：前向钩子

Gradient Debug：梯度调试

Step Timing：训练步计时

CUDA Synchronize：CUDA 同步

torch.profiler / PyTorch Profiler：PyTorch 性能分析器

Profiler Activity：Profiler 活动类型

Record Shapes：记录算子输入形状

Profile Memory：记录内存分配

DataLoader Bottleneck：DataLoader 瓶颈

Debug / Profiling Audit：调试与性能分析审计

PyTorch Engineering Interview Readiness：PyTorch 工程面试准备度

Engineering Interview Gate：工程面试门禁

Topic Coverage：主题覆盖率

Formula Coverage：公式覆盖率

Debug Coverage：排查路径覆盖率

Red Flag：风险表述 / 危险信号

Revision Plan：修订计划

Detach：切断计算图

No Grad：不记录梯度上下文

Inference Mode：推理模式

In-place Operation：原地操作

Anomaly Detection：异常检测

Autograd Hook：自动求导钩子

Autograd Audit：自动求导审计

Hessian：Hessian 矩阵 / 二阶导数矩阵

Loss Landscape：损失曲面

Loss Spike：损失尖峰 / loss 突增

Vanishing Gradient：梯度消失

Exploding Gradient：梯度爆炸

Information Theory：信息论

Entropy：熵

Conditional Entropy：条件熵

Mutual Information / MI：互信息

Cross Entropy：交叉熵

Cross-Entropy Loss：交叉熵损失

Maximum Likelihood Estimation / MLE：最大似然估计

Negative Log-Likelihood / NLL：负对数似然

KL Divergence：KL 散度 / 相对熵

Perplexity：困惑度

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

Transformer Components：Transformer 组件

Transformer Component Audit：Transformer 组件审计

Multi-Head Attention / MHA：多头注意力

Multi-Head Self-Attention：多头自注意力

Attention Head：注意力头

Head Dimension：头维度

Multi-Query Attention / MQA：多查询注意力

Grouped-Query Attention / GQA：分组查询注意力

Output Projection：输出投影

Transformer Block：Transformer 模块 / Transformer 块

Decoder Block：解码器模块 / Decoder 块

Pre-Norm Decoder Block：前置归一化解码器模块

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

Causal LM Shift Loss：因果语言模型右移损失

LM Head Weight Tying：语言模型输出头权重绑定

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

Distillation Data：蒸馏数据

Teacher Model：教师模型

Student Model：学生模型

Teacher-Student Distillation：教师-学生蒸馏

Self-Instruct：自指令生成 / 自举指令生成

Evol-Instruct：演化式指令生成

Prompt Registry：提示词登记表 / Prompt 版本登记系统

Teacher Authorization：教师模型输出授权

Synthetic Ratio：合成数据占比

Generated Data Recursion：生成数据递归训练

Model Collapse：模型坍缩 / 模型退化

Catastrophic Forgetting：灾难性遗忘

Model Editing：模型编辑 / 参数级局部修改

Knowledge Editing：知识编辑

ROME / Rank-One Model Editing：秩一模型编辑

MEMIT / Mass-Editing Memory in a Transformer：Transformer 记忆批量编辑

MEND / Model Editor Networks with Gradient Decomposition：基于梯度分解的模型编辑网络

Machine Unlearning：机器遗忘 / 机器反学习

LLM Unlearning：大模型遗忘

Forget Set：遗忘集 / 待删除目标集

Retain Set：保留集

Editing Locality：编辑局部性

Edit Success：编辑成功率

Unlearning Robustness：遗忘鲁棒性

Membership Inference Risk：成员推断风险

Training Data Extraction：训练数据抽取 / 训练数据复现风险

Membership Inference：成员推断

PII Leakage：个人可识别信息泄露

Secret Leakage：密钥泄露

Differential Privacy / DP：差分隐私

DP-SGD / Differentially Private SGD：差分隐私随机梯度下降

Privacy Budget：隐私预算

Privacy Gate：隐私门禁

Watermarking：水印

Watermark Detection：水印检测

Watermark Gate：水印门禁

Green Token：绿色 token / 水印候选 token

Watermark Z-Score：水印 z 分数

SynthID-Text：SynthID 文本水印

Content Credentials：内容凭证 / 内容来源凭证

C2PA / Coalition for Content Provenance and Authenticity：内容来源与真实性联盟

Editing Gate：模型编辑门禁

Unlearning Gate：遗忘门禁

Overfitting：过拟合

Padding：填充

Truncation：截断

Packing：打包

Reinforcement Learning from Human Feedback / RLHF：基于人类反馈的强化学习

Markov Decision Process / MDP：马尔可夫决策过程

State：状态

Action：动作

Transition：状态转移

Reward：奖励

Return：回报 / 累计奖励

Discount Factor：折扣因子

Trajectory：轨迹

Policy：策略

Value Function：价值函数

Q Function / Action-Value Function：动作价值函数

Advantage：优势函数 / 优势

Policy Gradient：策略梯度

Baseline：基线

Preference Data：偏好数据

Chosen Response：被选择回答 / 偏好回答

Rejected Response：被拒绝回答 / 非偏好回答

Reward Model：奖励模型

Pairwise Ranking Loss：成对排序损失

Policy Model：策略模型

Reference Model：参考模型

Proximal Policy Optimization / PPO：近端策略优化

PPO Ratio：PPO 新旧策略概率比

Clipped Surrogate Objective：裁剪代理目标

KL Penalty：KL 惩罚项

Reward Score：奖励分数

Reward Model Pairwise Loss：奖励模型成对偏好损失

Pairwise Accuracy：成对排序准确率

Reward Calibration：奖励校准

Distribution Shift：分布偏移

Goodhart's Law：古德哈特定律

Alignment Problem：对齐问题 / 目标对齐问题

Outer Alignment：外部对齐

Inner Alignment：内部对齐

Proxy Objective：代理目标

Objective Specification：目标规范

Specification Gaming：目标规范漏洞利用 / 规则钻空子

Goal Misgeneralization：目标误泛化

Deceptive Alignment：欺骗性对齐 / 伪装式对齐

Goodhart Gap：古德哈特缺口 / 代理目标缺口

Alignment Gate：对齐上线门禁

Scalable Oversight：可扩展监督

Human Feedback：人类反馈

AI Feedback：AI 反馈

Debate：辩论式监督

Iterated Amplification：迭代放大 / 递归放大监督

Recursive Reward Modeling：递归奖励建模

Oversight Gate：监督质量门禁

Human Audit：人工审计 / 人工复核

Verifier Coverage：验证器覆盖率

Evidence Support Rate：证据支持率

High-Risk Audit Coverage：高风险人审覆盖率

Weak-to-Strong Generalization：弱到强泛化

Adversarial Evaluation：对抗式评估

Over-Optimization：过度优化

Direct Preference Optimization / DPO：直接偏好优化

DPO Loss：DPO 损失

Beta in DPO：DPO 中的 beta 超参数

Evaluation Statistics：评估统计

Sample Mean：样本均值

Sample Variance：样本方差

Standard Error / SE：标准误

Confidence Interval / CI：置信区间

p-value：p 值

Null Hypothesis：零假设

Alternative Hypothesis：备择假设

Effect Size：效果大小

Statistical Significance：统计显著性

Practical Significance：实际显著性

Bootstrap：自助法 / bootstrap 重采样

Bootstrap Confidence Interval：bootstrap 置信区间

Paired Evaluation：成对评估

Paired Difference：成对差异

Discordant Pair：不一致样本对

McNemar Test：McNemar 检验

Sample Size：样本量

Statistical Power：统计功效

Minimum Detectable Effect / MDE：最小可检测效果

Multiple Comparisons：多重比较

Family-Wise Error Rate / FWER：族错误率

False Discovery Rate / FDR：错误发现率

Bonferroni Correction：Bonferroni 校正

Benjamini-Hochberg Procedure：Benjamini-Hochberg 程序

Experiment Gate：实验上线门禁

Math Interview Readiness：数学面试准备度

Math Interview Rubric：数学面试评分规则

Formula Coverage：公式覆盖率

Formula Accuracy：公式准确率

Formula Audit：公式审计

Demo Coverage：代码 demo 覆盖率

Weak Question：薄弱题

Revision Plan：修订计划 / 复盘计划

Length Bias：长度偏差

Over-Refusal：过度拒答

False Refusal：误拒

Unsafe Leak：不安全泄漏 / 漏拒

Safety Data：安全数据

Boundary Allowed Data：边界允许数据

Safety Refusal Data：安全拒答数据

Red Team Data：红队数据

Red Teaming：红队测试

Dangerous Capability Evaluation：危险能力评估

Capability Elicitation：能力激发

Red Team Regression Suite：红队回归测试集

Red Team Gate：红队门禁

Release Gate：发布门禁

Risk Taxonomy：风险分类体系

Unsafe Compliance Rate：不安全服从率 / 漏拒率

Refusal Accuracy：拒答准确率

Attack Success Rate：攻击成功率

Unauthorized Tool Call Rate：未授权工具调用率

Safe Completion Quality：安全替代回答质量

Severity-Weighted Risk：严重度加权风险

Safety Gate：安全上线门禁

Policy Coverage：策略覆盖度

System Card：系统卡 / 系统说明卡

Model Governance：模型治理

Governance Gate：治理门禁

Model Card Completion：模型卡完整度

System Card Completion：系统卡完整度

Risk Disclosure：风险披露

Responsible Scaling：负责任扩展 / 负责任规模化

Release Decision Record：发布决策记录

Approval Quorum：审批法定人数 / 审批覆盖要求

Annotator Agreement：标注者一致性

Preference Margin：偏好间隔

Rubric：评分规范 / 标注规范

Reward Hacking：奖励黑客 / 奖励作弊

Reward-Human Gap：奖励分数与人工质量差距

Proxy Mismatch：代理目标错配

Reward Hacking Rate：奖励漏洞利用率 / 奖励作弊率

High-Reward-Low-Quality：高奖励低质量样本

Best-of-N Overoptimization：Best-of-N 过度优化

Reward Hacking Gate：奖励漏洞门禁 / reward hacking 门禁

Benchmark Contamination：基准污染

Interpretability：可解释性

Mechanistic Interpretability：机制可解释性

Circuit：机制回路 / 子网络

Activation Patching：激活修补

Causal Tracing：因果追踪

Sparse Autoencoder / SAE：稀疏自编码器

Superposition：叠加表示

Polysemanticity：多语义性

Monosemanticity：单语义性

Feature Purity：特征纯度

Reconstruction Fidelity：重构保真度

Interpretability Gate：可解释性门禁

Steering：行为引导 / Steering

Representation Engineering：表示工程

Steering Vector：引导向量

Activation Steering：激活引导

Activation Intervention：激活干预

Contrastive Activation Addition / CAA：对比激活加法

Projection Shift：投影变化

Steering Gate：Steering 上线门禁

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

Distillation Temperature：蒸馏温度

Soft Target：软目标

Hard Target：硬目标

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

Mixed Precision Training：混合精度训练

Automatic Mixed Precision / AMP：自动混合精度

Autocast：自动类型转换上下文 / autocast

GradScaler：梯度缩放器

Loss Scaling：损失缩放

Unscale Gradients：反缩放梯度

Activation Memory：激活显存

Optimizer State Memory：优化器状态显存

CUDA Memory Stats：CUDA 显存统计

Memory Allocated：已分配显存

Memory Reserved：已保留显存

Peak Memory：峰值显存

Out of Memory / OOM：显存不足 / 内存不足

OOM Audit：OOM 排查审计

INT8：8 位整数

INT4：4 位整数

Weight-Only Quantization：仅权重量化

Multimodal Learning：多模态学习

Vision-Language Model / VLM：视觉语言模型

Vision Encoder：视觉编码器

Image Encoder：图像编码器

Vision Tower：视觉塔

Multimodal Data：多模态数据

Visual Token Budget：视觉 token 预算

Visual Token：视觉 token

Patch Embedding：图像 patch 嵌入

Patch Token：patch token / 图像块 token

CLS Token：CLS token / 全局汇聚 token

Vision Hidden Size：视觉隐藏维度

Vision Encoder Shape Audit：视觉编码器 shape 审计

Multimodal Context Budget：多模态上下文预算

Modality Connector：模态连接器 / 多模态桥接模块

VLM Connector Audit：VLM 连接器审计

Image Placeholder：图片占位符

Visual Token Compression：视觉 token 压缩

Assistant-Only Multimodal Loss：仅 assistant 位置计算的多模态损失

Multimodal SFT Data Audit：多模态 SFT 数据审计

Multimodal Task Coverage：多模态任务覆盖率

Missing Refusal Rate：漏拒率 / 应拒未拒率

Diffusion Model：扩散模型

Denoising Diffusion Probabilistic Models / DDPM：去噪扩散概率模型

Forward Diffusion：前向扩散 / 加噪过程

Reverse Denoising：反向去噪过程

Noise Scheduler：噪声调度器

Noise Prediction Loss：噪声预测损失

Classifier-Free Guidance / CFG：无分类器引导

Guidance Scale：引导强度

Latent Diffusion：潜空间扩散

Diffusion Transformer / DiT：扩散 Transformer

Stable Diffusion：稳定扩散 / Stable Diffusion 文生图系统

VAE Compression Ratio：VAE 压缩比例

Text-to-Image Pipeline Audit：文生图 pipeline 审计

Negative Prompt：负向提示词

ControlNet：条件控制网络 / ControlNet

Image-to-Image：图生图

Inpainting：局部重绘

DALL-E：DALL-E 文生图系列

Autoregressive Image Tokens：自回归图像 token

Video Generation：视频生成

Spatiotemporal Patch：时空 patch

Video Diffusion：视频扩散模型 / 视频 diffusion

Temporal Consistency：时序一致性

Identity Drift：身份漂移

Flickering：画面闪烁 / 帧间闪烁

World Model：世界模型

Physics Consistency：物理一致性

Video Generation Evaluation：视频生成评估

Fréchet Video Distance / FVD：弗雷歇视频距离

Video Token Audit：视频 token 审计

Cross-Attention Adapter：交叉注意力适配器

Q-Former：查询 Transformer / Query Transformer

Perceiver Resampler：Perceiver 重采样器 / 视觉 token 重采样器

Multimodal Cost Audit：多模态成本审计

Image-Text Pair：图文对

Image-Text Alignment：图文对齐

Caption Data：描述文本数据 / Caption 数据

OCR Data：OCR 数据 / 图像文字识别数据

Audio-Transcript Pair：音频转录对

Automatic Speech Recognition / ASR：自动语音识别

Text-to-Speech / TTS：文本转语音

Speech-to-Speech：语音到语音

Waveform：波形

Log-Mel Spectrogram：对数梅尔频谱图

Vocoder：声码器

Voice Cloning：声音克隆

Audio Codec：音频编解码器

Speech Token：语音 token

Codec Language Model：编解码器 token 语言模型

Voice Activity Detection / VAD：语音活动检测

Word Error Rate / WER：词错误率

Character Error Rate / CER：字符错误率

Mean Opinion Score / MOS：平均主观意见分

Audio Token Audit：音频 token 审计

Unified Multimodal Model：统一多模态模型

Any-to-Any Multimodal Model：任意模态到任意模态多模态模型

Unified Tokenization：统一 token 化

Early-Fusion Multimodal Transformer：早期融合多模态 Transformer

Center-LLM Multimodal System：以 LLM 为中心的多模态系统

Multimodal Router：多模态路由器

Multimodal Loss Mixture：多模态损失混合

Modality Conflict：模态冲突

Unified Multimodal Audit：统一多模态审计

Multimodal Evaluation：多模态评估

VQA Accuracy：视觉问答准确率

Chart Relaxed Accuracy：图表问答宽松准确率

Grounding IoU：视觉定位交并比

Multimodal Hallucination Rate：多模态幻觉率

Multimodal Prompt Injection：多模态提示注入

Biometric Identity Safety：生物身份安全

Content Provenance：内容来源与溯源

Multimodal Safety Audit：多模态安全审计

Temporal Alignment：时间对齐 / 时序对齐

Video Caption：视频描述文本

Grounded Answer：有证据支撑的回答

Grounded Multimodal QA：有媒体证据支撑的多模态问答

Visual Privacy：视觉隐私

Data Attribution：数据归因

Data Valuation：数据估值

Data Shapley：数据 Shapley 估值

Influence Function：影响函数

Gradient Similarity：梯度相似度

Source-Level Ablation：数据源级消融

Proxy Model：代理模型 / 替代模型

Negative Value Data：负价值数据

Active Learning：主动学习

Data Selection：数据选择

Marginal Contribution：边际贡献

Utility Function：效用函数

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

Indirect Prompt Injection：间接提示注入

Instruction Hierarchy：指令层级

Untrusted Content Boundary：不可信内容边界

Prompt Injection Gate：提示注入门禁

Data Exfiltration Rate：数据外泄率

Tool Result Injection：工具结果注入

Model Card：模型卡

Data Interview Readiness：数据工程面试准备度

Answer Coverage：回答覆盖度

Metric Coverage：指标覆盖度

Interview Rubric：面试评分规约

Mock Interview Retrospective：模拟面试复盘

Safety Interview Readiness：安全面试准备度

Safety Interview Rubric：安全面试评分规约

Safety Platform：安全平台 / 安全治理平台

Gate Coverage：门禁覆盖度

Trade-off Coverage：取舍覆盖度

Unsafe Detail Penalty：危险细节扣分

Readiness Gate：准备度门禁
