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
7. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
8. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
9. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
10. FlashAttention
11. CLIP
12. PagedAttention / vLLM 相关论文与技术报告
13. Mamba / Selective State Space Models
14. Constitutional AI / RLAIF 相关论文
15. ReAct / Toolformer / Function Calling 与 Agent 工具使用论文线
16. SimPO / ORPO / KTO 等不同反馈格式与 reference 约束的偏好优化论文线（KTO 使用 reference）
17. DeepSeek-R1 / GRPO / RLVR / DAPO 等 reasoning 后训练论文线
18. vAttention / FlashInfer 等 inference serving 内存与 kernel 论文线
19. GraphRAG 等结构化 RAG 论文线
20. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)：从参数化与非参数化记忆的结合理解 RAG；论文实验范围不等于任意生产检索系统的事实性保证。

## 核心方向

### Tokenizer、数据与预训练

1. [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/)：BPE 子词切分和开放词表的研究入口。
2. [SentencePiece](https://aclanthology.org/D18-2012/) 与 [Subword Regularization](https://aclanthology.org/P18-1007/)：原始文本 tokenizer 接口、Unigram 和子词采样。
3. [Unpacking Tokenization](https://arxiv.org/abs/2403.06265)、[Tokenization Is More Than Compression](https://arxiv.org/abs/2402.18376) 与 [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425)：分别观察压缩与下游效果的相关性、最短切分是否充分，以及不同语言的 token fertility；实验条件不可互换。
4. [FOCUS](https://arxiv.org/abs/2305.14481) 与 [Beyond Initialization Loss](https://arxiv.org/abs/2608.03494)：研究新 tokenizer 的 embedding 初始化与继续预训练；后者是 Hindi / Nemotron-3-Nano 特定设置下的 2026 预印本。
5. [Tokenizer Choice for LLM Training](https://aclanthology.org/2024.findings-naacl.247/)、[Objective vs. Search](https://aclanthology.org/2026.emnlp-main.2307/)、[No Single Tokenizer Feature Reliably Predicts Downstream Language Model Performance](https://aclanthology.org/2026.findings-emnlp.96/)、[TokLens](https://aclanthology.org/2026.acl-srw.18/) 与 [MUTANT](https://aclanthology.org/2026.acl-long.2146/)：分别研究受控 tokenizer 选择、目标/搜索解耦、intrinsic 指标泛化、多语相关性和印度语言 tokenizer recipe；模型规模、语言、benchmark 和硬件条件不同，不能把单篇结果当通用排名。
6. [C4 / T5](https://arxiv.org/abs/1910.10683)、[The Pile](https://arxiv.org/abs/2101.00027) 与 [FineWeb](https://arxiv.org/abs/2406.17557)：不同规模和流程下的数据构造案例。
7. [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)：去重、记忆与质量的论文实验；结论受其数据和评测设置约束。
8. [The Stack v2 / StarCoder 2](https://arxiv.org/abs/2402.19173) 与 [BigCode Responsible AI](https://arxiv.org/abs/2308.07124)：代码语料、许可和治理资料。
9. [A Survey on Data Selection for Language Models](https://arxiv.org/abs/2402.16827)：数据选择 taxonomy，覆盖质量、去重、配比、污染与不同训练阶段的目标。
10. [Dolma](https://arxiv.org/abs/2402.00159) 与 [DataComp-LM](https://arxiv.org/abs/2406.11794)：开放语料的治理 / lineage 与固定训练配方的数据中心 benchmark；不同论文的分数不能直接排名。
11. [DoReMi](https://arxiv.org/abs/2305.10429) 与 [RegMix](https://arxiv.org/abs/2407.01492)：通过 Group DRO 和小模型回归优化数据 mixture；实验数据、proxy 假设和目标评估不同。
12. [Nemotron-CC](https://arxiv.org/abs/2412.02595)：长 token horizon 下的质量筛选、真实数据覆盖和合成改写折中；数字属于作者训练报告。
13. [Removing Noise, not Finding Gold](https://arxiv.org/abs/2510.00866) 与 [Leverage Score Sampling](https://arxiv.org/abs/2609.32484)：分类器过滤的目标依赖性、多样性和跨域覆盖；后者为 2026-09 预印本，等待独立复核。
14. [Scaling Laws](https://arxiv.org/abs/2001.08361) 与 [Chinchilla](https://arxiv.org/abs/2203.15556)：模型规模、训练 token 和计算预算的经验研究。
15. [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)：独立数据受限时重复曝光的边际收益和预算分配；结论绑定 GPT-2、C4/OSCAR 与其训练设置。
16. [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448)：把推理需求和 serving 成本纳入训练规模选择；作者用 47 个模型验证拟合，并在约 1B 请求的假设情景讨论更小、训练更久的选择；请求量、质量、硬件与美元成本假设需重新核对。
17. [(Mis)Fitting: A Survey of Scaling Laws](https://arxiv.org/abs/2502.18969)：跨论文比较拟合形式、训练设置、评估和复现信息。
18. [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)：公开旗舰模型训练 token/FLOPs 和较小模型长训练的报告案例。
19. [Mixed Precision Training](https://arxiv.org/abs/1710.03740) 与 [A Study of BFLOAT16 for Deep Learning Training](https://arxiv.org/abs/1905.12322)：FP16 loss scaling 和 BF16 数值特性；论文实验不代表当前所有硬件的性能。
20. [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer / T5](https://arxiv.org/abs/1910.10683)：对比 causal/prefix language modeling 与 BERT-style denoising，研究 sentinel span corruption、目标长度及其任务/吞吐影响；配套 [T5 代码](https://github.com/google-research/text-to-text-transfer-transformer) 和 [SeqIO](https://github.com/google/seqio) 可追踪任务预处理。
21. [UL2: Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131)：R/S/X denoiser 混合与 mode switching；聚合提升需绑定原论文模型规模、C4、约 32B token 和多任务评测口径。
22. [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/abs/2207.14255)：FIM token 重排、PSM/SPM、训练率与 infilling pass-rate 消融；[HumanEval Infilling repository](https://github.com/openai/human-eval-infilling)提供配套 benchmark 代码（当前已归档）。
23. [Demystifying Synthetic Data in LLM Pre-training](https://aclanthology.org/2025.emnlp-main.544/) 与 [Rewriting Scaling Laws for Synthetic Data with Compute and Data Constraints](https://arxiv.org/abs/2610.31867)：前者的 EMNLP 2025 正式论文比较自然网页、改写文本、教材式生成数据及混合比例；约三分之一改写文本在特定大预算对照中可使达到相同验证 loss 的速度提高 5–10 倍。后者把生成成本纳入 scaling-law 估算，最优改写比例随 compute-limited 预算上升。数据类型、生成成本和拟合假设各不相同，不能照搬为通用配方；后者是 2026 年预印本。
24. [Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d0c80a0c294a16190c8904b9809c5fba-Abstract-Conference.html)：把数据质量与噪声、覆盖率作为受控预训练变量研究；适合建立质量消融的实验思路，不是任意语料的通用质量分数。
25. [Scaling Laws for Optimal Data Mixtures](https://proceedings.neurips.cc/paper_files/paper/2025/hash/bc1d640f841f752c689aae20b31198c1-Abstract-Conference.html)：把混合比例、训练规模与目标评估放进联合实验；最优配比受语料、目标和训练预算约束，不可直接迁移成固定比例。
26. [Datasets, Documents, and Repetitions](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2aba6ec20299931d46ddeefd5ddcb442-Abstract-Conference.html)：区分数据集规模、文档独立性和重复曝光；作者的 scaling 结论需要连同采样方案与实验数据阅读。
27. [Explaining and Mitigating Crosslingual Tokenizer Inequities](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5b91cefbfa52340af2f16f249572dc76-Abstract-Conference.html)、[Broken Tokens?](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2b5c5689fae6fa9a4883e73e511d52c8-Abstract-Conference.html)、[zip2zip](https://proceedings.neurips.cc/paper_files/paper/2025/hash/def682d57535b9e778508b4b1a911034-Abstract-Conference.html) 与 [zip2zip++](https://arxiv.org/abs/2610.13605)：分别研究跨语言 token 成本、非规范切分鲁棒性、动态词表压缩及其质量改进。结论受语言、模型训练阶段、任务、硬件和实现协议限制；zip2zip++ 是 2026 年 arXiv v1 预印本，不构成任意改写 tokenizer 的部署建议。
28. [Rethinking Multilingual Continual Pretraining](https://arxiv.org/abs/2504.04152)：COLM 2025 对三个多语言底模、30 多种语言和 36 种 CPT 配置的比较；双语/代码混合的分类与生成结果有取舍，属于特定语言和模型设置下的实验，不是通用 mixture 配方。
29. [In-Place Tokenizer Expansion for Pre-trained LLMs](https://arxiv.org/abs/2607.15232)：模型生产方延续原 BPE merge table、迁移 embedding 并分两阶段继续训练的案例；每字符解码收益由压缩率和参考设备的 token 吞吐合成，且原支持语言存在回退，不能视作独立 tokenizer 替换或通用线上速度保证。
30. [Packing and Masking Strategies for Pre-Training Sequence Composition](https://aclanthology.org/2024.acl-long.427.pdf)：比较将 EOS 分隔的文档装入固定长度 chunk 时的因果 mask 与文档内隔离 mask，并报告跨文档上下文干扰和隔离开销；结论限于论文的模型、数据和任务，EOS token 本身不会重置 attention。

### 架构

Transformer、GPT、LLaMA、Mistral、MoE、SSM、S4、Mamba、RWKV、RetNet、Hyena、Linear Attention、Attention + SSM 混合架构。

可解释性入口：[Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)：阅读 FFN 的 key–value memory 视角时，应保留论文的特定模型与机制分析范围，不把它理解成神经元存储事实的定论。

建议补读论文线：

1. Efficiently Modeling Long Sequences with Structured State Spaces / S4。
2. Mamba: Linear-Time Sequence Modeling with Selective State Spaces。
3. RWKV: Reinventing RNNs for the Transformer Era。
4. Retentive Network / RetNet。
5. Hyena Hierarchy。
6. Linear Transformers / Performer 等线性注意力路线。

Transformer 核心机制入口：

- [Pre-training under infinite compute](https://proceedings.iclr.cc/paper_files/paper/2026/hash/79682ef54677a2cb3dfb8383b372037f-Abstract-Conference.html)、[Pretraining Scaling Laws for Generative Evaluations of Language Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/55d7fbeddfd9bb6dace146cc4fa02ff0-Abstract-Conference.html) 与 [How Text Quality Interventions Reshape Neural Scaling Laws for LLMs: Empirical Study](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6544c0b53cc1ea51d0c7086b536830a8-Abstract-Conference.html)：分别扩展固定数据/重复训练、生成式 pass@k 评估和数据质量干预下的 scaling 问题；结论仍受作者拟合形式、任务和实验范围约束。

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：scaled dot-product attention、多头结构和正弦位置编码。
2. [BERT](https://arxiv.org/abs/1810.04805)：WordPiece、双向 encoder、MLM 与 NSP 的代表性实例；可对照 [RoBERTa](https://arxiv.org/abs/1907.11692) 看预训练数据量、动态 masking、NSP 和训练预算等因素如何共同改变结果；再读 [GPT 原始预训练报告](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)，比较 decoder-only 的无监督预训练与 BERT 的双向编码目标。
3. [SentencePiece](https://arxiv.org/abs/1808.06226)：从原始文本训练子词模型的 tokenizer 工具与模型入口。
4. [RoFormer](https://arxiv.org/abs/2104.09864)：RoPE 的旋转位置机制。
5. [ALiBi](https://arxiv.org/abs/2108.12409)：在 attention 分数中加入距离相关线性 bias 的位置方案。
6. [RMSNorm](https://arxiv.org/abs/1910.07467) 与 [Pre-LN/Post-LN 分析](https://arxiv.org/abs/2002.04745)：归一化及深层 Transformer 梯度路径。
7. [GQA](https://arxiv.org/abs/2305.13245)：共享 K/V heads 与 KV Cache 取舍。
8. [Lost in the Middle](https://arxiv.org/abs/2307.03172)：长上下文中信息位置效应的研究实例。
9. [Layer Normalization](https://arxiv.org/abs/1607.06450)、[ScaleNorm](https://arxiv.org/abs/1910.05895) 与 [GeLU](https://arxiv.org/abs/1606.08415)：归一化和激活的原始定义及实验范围。
10. [Position Interpolation](https://arxiv.org/abs/2306.15595) 与 [YaRN](https://arxiv.org/abs/2309.00071)：RoPE 上下文扩展的代表性训练方案；结果受模型、训练预算和评测任务约束。
11. [PagedAttention](https://arxiv.org/abs/2309.06180)：KV block 分配、缓存共享和 vLLM serving 系统。
12. [Switch Transformers](https://arxiv.org/abs/2101.03961) 与 [ST-MoE](https://arxiv.org/abs/2202.08906)：top-1 路由、capacity、token dropping 和 router z-loss。
13. [LongRoPE](https://arxiv.org/abs/2402.13753)：非均匀位置插值搜索、分阶段扩展到超长上下文和短上下文恢复；2M 是论文设置中的结果，不是通用有效上下文承诺。
14. [DeepNet / DeepNorm](https://arxiv.org/abs/2203.00555)：把残差缩放和初始化共同设计以控制深层 Transformer 的 model update；代表实验是多语机器翻译。
15. [Tensor Programs V / μTransfer](https://arxiv.org/abs/2203.03466)：通过 μP 参数化把部分宽度超参数从小模型迁移到大模型；不是单一初始化公式。
16. [Expert Choice Routing](https://arxiv.org/abs/2202.09368)：让 expert 选择固定容量的 token bucket，缓解 token-choice 的专家负载不均，同时允许不同 token 激活不同数量的 expert。
17. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)：报告按专家负载调整 routing bias 的 auxiliary-loss-free batch balancing，并保留 sequence-wise 辅助损失防止单序列失衡；结果绑定其 MoE 与集群共设计。

- [Multi-Layer Scheduling for MoE-Based LLM Reasoning](https://arxiv.org/abs/2602.21626)：研究请求级、引擎级和 expert 级的分层调度；作者报告的 TTFT/TPOT 变化绑定论文 workload、模型、集群和服务实现，不能由 router token 计数直接推出。

这些论文的实验结论要按其模型、数据和评测设置阅读，不能直接当作所有现代 LLM 的配置或性能保证。

### 长上下文、检索与 RAG

正文入口：[第七册第十三章：1M Context 有效能力评测](book-07-evaluation-experiments/chapters/13-1m-context有效能力评测.md)、[第七册第十四章：Frontier Model 证据等级](book-07-evaluation-experiments/chapters/14-frontier模型证据等级.md)。

1. [Position Interpolation](https://arxiv.org/abs/2306.15595) 把位置映射回预训练范围；[YaRN](https://arxiv.org/abs/2309.00071) 按 RoPE 波长选择插值并增加 attention scaling；[LongRoPE](https://arxiv.org/abs/2402.13753) 通过非均匀搜索和分阶段适配报告 2,048K；[LongLoRA](https://arxiv.org/abs/2309.12307) 研究长上下文微调成本。各论文结果受模型、训练预算和评测任务约束，最大窗口不能代替有效长度。
2. [Longformer](https://arxiv.org/abs/2004.05150) 与 [BigBird](https://arxiv.org/abs/2007.14062) 以局部/全局/随机稀疏连接改变 token 交互图；[Ring Attention](https://arxiv.org/abs/2310.01889) 把 blockwise attention 分布到多设备；[FlashAttention](https://arxiv.org/abs/2205.14135) 与 [PagedAttention](https://arxiv.org/abs/2309.06180) 优化 HBM 数据移动和 KV 状态管理。这些路线缓解的瓶颈不同，不应合并成一个“上下文扩展倍数”。
3. [Lost in the Middle](https://arxiv.org/abs/2307.03172) 控制证据位置；[RULER](https://arxiv.org/abs/2404.06654) 加入多 needle、多跳和聚合的可控复杂度；[NoLiMa](https://arxiv.org/abs/2502.05167) 降低问题与证据的词面重叠；[LongBench v1](https://arxiv.org/abs/2308.14508) 扩展中英文多任务覆盖；[LongBench v2](https://arxiv.org/abs/2412.15204) 以高难长上下文题测理解与推理，长度用 words 而非 tokens；[LongBench Pro](https://arxiv.org/abs/2601.02872) 提供 8K–256K token 的中英自然样本和细分长度/依赖类别，仍是 2026 年预印本。
4. [Google DeepMind MRCR v2](https://github.com/google-deepmind/eval_hub/tree/master/eval_hub/mrcr_v2) 固定较简单的多轮共指任务复杂度并扩大长度，适合看极长输入上的曲线；仓库说明可生成到 8M，但任务和噪声模型有明确边界。OpenAI [MRCR / Graphwalks 发布说明](https://openai.com/index/gpt-4-1/)提供相近的厂商任务设计入口，不能与 DeepMind 变体混分。
5. needle 冒烟成绩不等于跨任务有效利用。基准分数不能脱离数据、提示、长度单位、推理预算、工具权限和 harness 直接排名；具体任务组合见[第七册第十三章](book-07-evaluation-experiments/chapters/13-1m-context有效能力评测.md)和[研究记录](research/long-context-effective-capability-depth-review-2026-10-27.md)。
6. [REALM](https://arxiv.org/abs/2002.08909)、[Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)、[RAG](https://arxiv.org/abs/2005.11401) 与 [RETRO](https://arxiv.org/abs/2112.04426)：检索增强语言模型的早期路线，分别比较潜在检索预训练、检索器、生成时文档边缘化和 chunked cross-attention。
7. [RAG Survey](https://arxiv.org/abs/2312.10997)：按 retrieval、generation、augmentation 和 evaluation 组织后续 RAG 研究；综述用于建立地图，具体结论仍回到原始论文和目标系统。
8. [HNSW](https://arxiv.org/abs/1603.09320)、[FAISS](https://arxiv.org/abs/1702.08734)、[BEIR](https://arxiv.org/abs/2104.08663) 与 [MTEB](https://arxiv.org/abs/2210.07316)：近似检索系统和跨任务评估；目标语料仍要单独测过滤后召回和端到端质量。
9. [MonoT5](https://arxiv.org/abs/2003.06713) 与 [ColBERT](https://arxiv.org/abs/2004.12832)：生成式精排与 late-interaction 检索设计。
10. [ALCE](https://arxiv.org/abs/2305.14627)、[ARES](https://aclanthology.org/2024.naacl-long.20/) 与 [RAGTruth](https://arxiv.org/abs/2401.00396)：带引用生成、通过合成训练数据/轻量 judge/少量人工标签的 prediction-powered inference 估计 RAG 组件，以及不忠实回答分析；ARES 的验证绑定论文任务与校准分布，自动 judge 仍需按目标语料复核。
11. [FActScore](https://arxiv.org/abs/2305.14251) 与 [AttributionBench](https://aclanthology.org/2024.findings-acl.886/)：细粒度事实核验和自动引用归因评估；结论绑定其原论文数据与评审协议。
12. [RAGAS](https://arxiv.org/abs/2309.15217)：RAG reference-free 自动化评估框架；和 ARES 对照时要固定任务、人工标注协议与 judge，不把单篇论文结果推成通用排名。
13. [Retrieval Augmented Generation or Long-Context LLMs?](https://arxiv.org/abs/2407.16833) 与 [Long Context vs. RAG for LLMs](https://arxiv.org/abs/2501.01880)：比较长上下文和检索路线；结论随模型、检索方式、问题类型、数据过滤和预算变化，当前论文报告不能作为跨语料排名。
14. [Self-RAG](https://arxiv.org/abs/2310.11511) 与 [Corrective RAG / CRAG](https://arxiv.org/abs/2401.15884)：比较模型内 learned retrieval / reflection 与外部 retrieval evaluator / fallback；前者依赖相应模型训练，后者依赖 evaluator 校准及可信的外部搜索。
15. [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130) 与 [RAPTOR](https://arxiv.org/abs/2401.18059)：分别用实体关系图/社区摘要处理全局语料问题、用递归聚类摘要树支持不同抽象层次的检索；索引构建、更新和原文定位成本不同，不替代精确 chunk 检索。

### 评估与 Debug 的统计和方法

正文入口：[第三册第九章：离线评估与 Debug](book-03-practical-handbook/chapters/09-评估与debug实战.md)、[第三册第十章：Agent 轨迹与线上评估](book-03-practical-handbook/chapters/10-agent轨迹与线上评估实战.md)、[第五册第十章](book-05-llm-training/chapters/10-checkpoint评估与实验管理.md)、[第七册第三章](book-07-evaluation-experiments/chapters/03-human-eval与pairwise-eval.md)、[第七册第四章](book-07-evaluation-experiments/chapters/04-llm-as-a-judge.md)、[第七册第五章](book-07-evaluation-experiments/chapters/05-污染检测与可信评估.md)、[第七册第六章](book-07-evaluation-experiments/chapters/06-reasoning-code-math-eval.md)、[第七册第七章](book-07-evaluation-experiments/chapters/07-多模态与长上下文评估.md)、[第七册第十一章](book-07-evaluation-experiments/chapters/11-error-analysis与回归测试.md)和[第七册第十二章](book-07-evaluation-experiments/chapters/12-评估体系综合设计与工程落地.md)。

1. [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)：位置、冗长和自我偏好等评审偏差；结论绑定其题目、模型和人工比较协议。
2. [Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge](https://arxiv.org/abs/2406.07791)：用候选顺序交换、重复评审和多候选指标分析位置敏感性；v9 报告 15 个 judge、22 类任务和超过 150,000 次评估。偏差随 judge/任务变化，长度影响较弱但不能据此假定为零；交换一致不代表评分正确或目标任务已校准。
3. [A Coefficient of Agreement for Nominal Scales](https://doi.org/10.1177/001316446002000104)、[Reliability in Content Analysis](https://doi.org/10.1111/j.1468-2958.2004.tb00738.x) 与 [Inter-Coder Agreement for Computational Linguistics](https://aclanthology.org/J08-4004/)：分类与内容分析一致性；系数选择依赖标注设计，一致不等于事实正确。
4. [Trustworthy online controlled experiments: five puzzling outcomes explained](https://doi.org/10.1145/2339530.2339653) 与 [CUPED](https://doi.org/10.1145/2433396.2433413)：线上实验可信度与实验前协变量降方差；需另审随机化单元、稀疏安全事件和停止规则。
5. [The Hitchhiker's Guide to Testing Statistical Significance in NLP](https://aclanthology.org/P18-1128/)、[Recommended Statistical Significance Tests for NLP Tasks](https://arxiv.org/abs/1809.01448) 与 [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264v4)：按任务和 metric 选择成对检验、bootstrap、随机化等方法；深度 RL 研究说明少量运行的点估计可能掩盖不确定性，并讨论区间、性能分布和 IQM。相关统计工具需匹配数据层级；RL 结果不能直接当作 LLM 评测实证。显著性不等于实际重要性，也要处理多重比较。
6. [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110)：用共同的 scenarios/adaptations/metrics 提高不同模型结果的可比性，并公开逐样本 prompts/completions；覆盖集合受论文的 feasibility 取舍限制，不是所有真实用例的全集。
7. [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782)：记录 lm-eval 对评测任务、prompt、harness 和原始输出的处理，以及题目 bootstrap、训练 seed、prompt variation 等不确定性区别；lm-eval 是评测执行框架，不替用户选择代表性 benchmark。
8. [Proving Test Set Contamination in Black-Box Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/46e624c244cff669223d488defd4e835-Abstract-Conference.html)：以 canonical/shuffled benchmark 顺序似然测试闭源模型的污染信号；需似然接口和可交换性等前提，未检出不构成无污染证明。
9. [ConStat: Performance-Based Contamination Detection in Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a7f89793b9e6f8c6568dbbb6ff727b9b-Abstract-Conference.html)：把异常提升不能迁移到重述样本/同任务参考集的表现定义为污染信号；它评估非泛化差距，不证明训练语料中直接包含原样本。
10. [PALOMA: A Benchmark for Evaluating Language Model Fit](https://arxiv.org/abs/2312.10523)：覆盖多文本域并控制词表、文档切分、抽样和去污染，说明单一混合 perplexity 难以代表所有域；PPL 仍不是下游能力分数。
11. [PaCoST](https://arxiv.org/abs/2406.18326)：比较原题与语义改写题的配对置信度，以统计信号检测 benchmark 级潜在污染；需要概率访问和多次调用，不作逐样本溯源或来源证明。
12. [Prediction-Powered Inference](https://arxiv.org/abs/2301.09633)：用大量模型预测估计总体量，以少量人工标签估计预测残差并构造有效区间；效果受标注抽样、目标分布和预测误差校正设计约束。
13. [GPQA](https://arxiv.org/abs/2311.12022)：专业科学题构造与领域专家/非专家验证；要区分 546 题 Extended 与筛选后的 448 题 main、198 题 Diamond，后两者准确率受筛选影响。
14. [LiveBench v2](https://arxiv.org/abs/2406.19314v2)：论文描述近期题源、客观 ground truth 和按月更新；项目主页于 2026-10-27 核验时列出 2026-01-08 快照并说明约每六个月完整刷新。公开题仍可能被未来模型训练，引用时需绑定具体快照与访问日期。
15. [When Benchmarks are Targets](https://arxiv.org/abs/2402.01781)：测量选项顺序和答案提取扰动如何改变多选基准分数及相对排名；结论绑定作者模型/题库。
16. [Can We Trust Item Response Theory for AI Evaluation?](https://arxiv.org/abs/2607.15190)：预印本分析模型数量有限、能力分布偏斜或聚集时 IRT 参数与排名推断的可靠性风险；不代表 IRT 对所有 benchmark 都失效。
17. [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132)：比较真实用户 pairwise 反馈、非均匀对战抽样、Bradley--Terry 系数与区间估计；作者明确限制在志愿用户和网页聊天分布，且该研究未测安全性。
18. [Best Practices for the Human Evaluation of Automatically Generated Text](https://aclanthology.org/W19-8643/) 与 [The Authenticity Gap in Human Evaluation](https://aclanthology.org/2022.emnlp-main.406/)：前者总结 NLG 人评任务、标注和报告实践；后者从效用理论审查 Likert 平均分的偏好解释，并提出 SPA。结论需绑定各自任务和假设。
19. [BeyondBench](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22b4e30a7a660b17f1fb58ab49671e77-Abstract-Conference.html)：以算法生成的数学与组合任务减少固定题面记忆风险；44 类任务和 117 种变化形式受其生成规则与判分覆盖约束，不代表一般知识或开放任务评估。
20. [The Token Tax](https://aclanthology.org/2026.africanlp-main.10/)：AfricaNLP 2026 正式论文在 16 种非洲语言、5 个学科的 AfriMMLU 评测中研究 tokenizer fertility 与问答表现关系；相关性不等于 tokenizer 导致准确率差异的因果证明。
19. [Rethinking the Agreement in Human Evaluation Tasks](https://aclanthology.org/C18-1281/) 与 [Interrater Disagreement Resolution](https://aclanthology.org/2021.humeval-1.15/)：前者提醒自然语言质量判断可能有不可消除的分歧，后者为目标是共识标注的任务提供按分歧来源复核的流程。
20. [Dropping Just a Handful of Preferences Can Change Top Large Language Model Rankings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/92d777fb721936613b6444b07c4ffc18-Abstract-Conference.html)：ICLR 2026 以 AMIP 找候选影响票后精确重拟合 BT 排名；固定数据中的最坏情况删除能翻榜，但不是随机抽样区间或投票作假证据。
21. [Does Data Contamination Detection Work (Well) for LLMs? A Survey and Evaluation on Detection Assumptions](https://aclanthology.org/2025.findings-naacl.291/)：综述 50 篇检测研究、归纳 8 类假设，并对 3 类 membership-inference 假设做实证案例；其中部分方法在论文使用的预训练数据样本上接近随机猜测。该结论不等于所有成员推断方法或污染检测都无效。
22. [AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge](https://aclanthology.org/2025.acl-long.901/)：从 Wikidata/Wikipedia 历史修订识别 cutoff 后更新事实，再自动生成 QA 样本和支持文档；主要覆盖 QA，受知识来源、cutoff 声明和自动构题质量限制。
23. [Detecting Benchmark Contamination Through Watermarking](https://arxiv.org/abs/2502.17259)：发布前用水印模型改写 benchmark，再检测训练后遗留的统计信号；ICLR 2025 Workshop 论文在 1B 模型/10B tokens 的受控预训练中验证，属于前瞻方案，不能追溯已公开旧题。
21. [Challenges in Trustworthy Human Evaluation of Chatbots](https://arxiv.org/abs/2412.04363)：用公开 Arena 偏好快照模拟冷漠随机票、定向攻击票和主观问题分歧；票数扰动结果是基于旧快照的压力测试，不是平台实际低质量票率估计。
22. [Illusions of the Gold Standard: A Large-scale Analysis of Human Evaluation Protocols for Long-form Text Generation](https://aclanthology.org/2026.acl-long.635/)：人工复核 284 篇并辅助分析 1.8k 余篇近年 *CL 长文本生成人评论文，整理复现报告项及其缺漏；结论限于作者会议论文样本。
23. [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://aclanthology.org/2023.emnlp-main.153/)：从评价标准生成细化步骤、表单打分和评分 token 概率加权；GPT-4 的 SummEval 离散版 Spearman ρ=0.514 / Kendall τ=0.418，概率加权版分别为 0.502 / 0.446。论文提醒离散版 τ 受并列分数影响，所有数字均绑定原论文摘要任务。
24. [A Closer Look into Using Large Language Models for Automatic Evaluation](https://aclanthology.org/2023.findings-emnlp.599/)：对照 G-Eval 的自动 CoT、数字输出和评分说明；说明提示流程的收益须在目标 meta-eval 上实测。
25. [Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models](https://aclanthology.org/2024.emnlp-main.248/)：direct-assessment 与 pairwise evaluator 分开训练后合并权重；报告四类直接评分与四类 pairwise 基准，数据集与评测任务有英语和领域边界。
26. [Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges](https://aclanthology.org/2025.gem-1.33/)：用 TriviaQA 受控设置分析标签一致、绝对分数偏差和排序能力；高百分比一致不代表绝对分数已校准。
27. [A Survey on LLM-as-a-Judge](https://arxiv.org/abs/2411.15594)：汇总 prompt、模型选择、后处理、pipeline、偏差和鲁棒性，适合作为检索地图，不替代原论文。
28. [SAJA: A Simple Approach to Judge Alignment for LLM-as-a-Judge](https://aclanthology.org/2026.acl-industry.45/)：一次结构化 rubric 调用生成多维特征，用少量人工标签训练校准头；Headline 结果绑定论文模型、基准和样本切分。
29. [Judging LLM-as-a-Judge: Concerning Rubric Artifacts in LLM-based Automated Text Generation Evaluation](https://aclanthology.org/2026.emnlp-main.2612/) 与[详细预印本](https://arxiv.org/abs/2609.02942)：检查 rubric-only 标签可预测性和答案/标准反事实响应；结果绑定 HealthBench 等特定数据和模型。
30. 多模态与长上下文 benchmark：[VQA v2](https://arxiv.org/abs/1612.00837)用互补图像对控制语言先验；[POPE](https://arxiv.org/abs/2305.10355)用对象存在性探针评估视觉幻觉；[MMMU](https://arxiv.org/abs/2311.16502)测多学科多模态推理；[DocVQA](https://arxiv.org/abs/2007.00398)和 [2026 ANLS harness 审计](https://aclanthology.org/2026.docinsights-1.6/)展示文档指标与答案格式依赖；[ChartQA](https://arxiv.org/abs/2203.10244)结合真实来源图表、人工与生成问题；[AudioBench](https://arxiv.org/abs/2406.16020)补语音/声音/副语言理解；[Video-MME](https://arxiv.org/abs/2405.21075)、[MLVU](https://arxiv.org/abs/2406.04264)和 [MMLongBench-Doc](https://arxiv.org/abs/2407.01523)覆盖视频时长与跨页多模态证据。每项只支持自己的任务、数据和评分协议，不能合成跨模态总排名。

31. 近期多模态评测：[XModBench](https://proceedings.iclr.cc/paper_files/papers/2026/hash/21071811269c70ac11a3e7869bac747e-Abstract-Conference.html)测文本、图像和音频之间的能力与一致性；[LVOmniBench](https://arxiv.org/abs/2603.19217)测长音视频理解；[WBench](https://arxiv.org/abs/2605.25874)测多轮交互式视频世界模型；[RVG-Bench](https://arxiv.org/abs/2610.32268)测推理驱动的音视频时间定位；[VHELM](https://arxiv.org/abs/2410.07112v2)以九个方面组织视觉语言模型评估。结果受各自数据、模型快照和协议限制，不能拼成统一排名。

#### Frontier model claims 与证据报告

正文入口：[第七册第十四章：Frontier Model 证据等级](book-07-evaluation-experiments/chapters/14-frontier模型证据等级.md)。

1. [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) 与 [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)：分别结构化模型用途/限制披露和数据来源/处理记录；披露模板不核验填写真伪，也不证明模型消费了所列数据。
2. [What's documented in AI? A Systematic Analysis of 32K AI Model Cards](https://arxiv.org/abs/2402.05160)：审计 Hugging Face 模型卡字段覆盖；该快照的填写率不能直接外推到其他模型仓库或当前全体模型。
3. [HELM](https://arxiv.org/abs/2211.09110)：在共同 scenario、adaptation 和多指标框架下改善比较条件；覆盖受可行性和任务集合限制。
4. [CheckList](https://aclanthology.org/2020.acl-main.442/)：以 minimum functionality、invariance、directional expectation 将总分变成行为测试；测试清单与 oracle 决定能发现哪些失败。
5. [G-Eval](https://aclanthology.org/2023.emnlp-main.153/)：把 rubric、细化评审步骤和表单式 LLM 打分组合起来；人工相关性限于论文任务，不能当事实准确率。
6. [Evaluation Cards](https://arxiv.org/abs/2606.09809)：组织 benchmark、model 和运行记录的解释字段；它是 2026 年预印本中的报告层方案，不核验底层结果真假。上述方法的机制、结果和补证路线见[第十四章 §14.4.6](book-07-evaluation-experiments/chapters/14-frontier模型证据等级.md)。

#### 在线实验与 A/B Test

正文入口：[第七册第九章](book-07-evaluation-experiments/chapters/09-在线实验与ab-test.md)。

1. [CUPED](https://exp-platform.com/cuped/)：用实验前协变量降低指标方差；效果绑定协变量预测力与实验单位，不能修复分配、曝光或结局日志缺陷。
2. [Always Valid Inference](https://arxiv.org/abs/1512.04922)：为连续监测 A/B 结果提供始终有效的 p 值与区间；说明序贯监测需要匹配的推断程序。
3. [Diagnosing Sample Ratio Mismatch](https://www.microsoft.com/en-us/research/publication/diagnosing-sample-ratio-mismatch-in-online-controlled-experiments-a-taxonomy-and-rules-of-thumb-for-practitioners/)、eBay 的 [automated randomization validation and SRM detection](https://arxiv.org/abs/2208.07766) 与 [A Dirty Dozen](https://www.microsoft.com/en-us/research/publication/a-dirty-dozen-twelve-common-metric-interpretation-pitfalls-in-online-controlled-experiments/)：分别提供 SRM 根因 taxonomy、PSI/序贯 SRM 平台监测案例和指标解释/护栏失误案例。
4. [Detecting Network Effects](https://www.kdd.org/kdd2017/papers/view/detecting-network-effects-randomizing-over-randomized-experiments)：比较个体随机与集群随机，检查在线实验的干扰假设。
5. [Generative AI at Work](https://doi.org/10.1093/qje/qjae044)、[The Effects of Generative AI on High-Skilled Work](https://doi.org/10.1287/mnsc.2025.00535) 与 [LLM-enabled primary care trial](https://doi.org/10.1038/s41591-026-04503-6)：对照分阶段部署、跨企业代码助手随机试验和临床集群随机试验；指标、随机单位和识别设计不同，不能把各自百分比当作通用 LLM 收益。

本章的旧知识承接、来源核验、方法比较、数学复核和 demo 静态审阅见[在线实验深度研究记录](research/online-experimentation-llm-depth-review-2026-10-27.md)。

#### 统计推断与实验设计

正文入口：[第七册第十章](book-07-evaluation-experiments/chapters/10-统计显著性与实验设计.md)。

1. [The Hitchhiker’s Guide to Testing Statistical Significance in NLP](https://aclanthology.org/P18-1128/)：按任务、指标和观测依赖选择检验；强调 test item 结构影响检验有效性。
2. [An Empirical Investigation of Statistical Significance in NLP](https://aclanthology.org/D12-1091/)：考察 bootstrap 显著性校准及测试域到目标域的迁移边界；显著性不能替代外部有效性。
3. [CUPED / Improving the Sensitivity of Online Controlled Experiments](https://exp-platform.com/cuped/)（[WSDM 2013 DOI](https://doi.org/10.1145/2433396.2433413)）：实验前协变量降方差；约 50% 是 Bing 研究中的报告，不是任意实验的预期值。
4. [Always Valid Inference](https://arxiv.org/abs/1512.04922)：连续监测的有效 p 值、区间及序贯多重检验入口。
5. [Holm step-down](https://doi.org/10.2307/4615733)、[Benjamini–Hochberg](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) 与 [Benjamini–Yekutieli](https://doi.org/10.1214/aos/1013699998)：分别理解 FWER、FDR 和检验依赖下的 FDR 控制。

本章的知识承接、来源核验、方法比较、公式修订和 demo 静态审阅见[统计显著性研究记录](research/statistical-significance-experimental-design-depth-review-2026-10-27.md)。

#### Error Analysis 与回归测试

正文入口：[第七册第十一章](book-07-evaluation-experiments/chapters/11-error-analysis与回归测试.md)。

1. [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://aclanthology.org/2020.acl-main.442/)：用能力 × MFT / invariance / directional expectation 设计行为测试；测试发现受能力清单、扰动语义和 oracle 质量约束，不能直接定位生产根因。
2. [The ML Test Score](https://doi.org/10.1109/BigData.2017.8258038)：28 条生产 ML 数据、模型、基础设施和监控检查；适合审查工程防线，不是 LLM 语义质量 benchmark。
3. [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110)：以共同场景、适配和多指标提高横向可比性；论文场景覆盖不等于私有应用的回归集。
4. [EvalPlus / HumanEval+](https://papers.neurips.cc/paper_files/paper/2023/hash/43e9d647ccd3e4b7b5baab53f0368686-Abstract-Conference.html)：LLM seed、类型变异、程序契约和差分执行补足函数级测试；作者报告扩展测试揭示了原测试漏掉的代码错误，也发现部分 canonical solutions 有缺陷，结论绑定 HumanEval 与其 oracle。
5. [OpenAI Trace Grading](https://developers.openai.com/api/docs/guides/trace-grading)：Agent 端到端 trace 的结构化评分文档；属于厂商产品入口，需单独校准 grader 并核对动态平台状态。
6. [Agentic CLEAR](https://aclanthology.org/2026.acl-demo.74/)（ACL 2026 System Demonstrations）与 [AgentRx](https://www.microsoft.com/en-us/research/blog/systematic-debugging-for-ai-agents-introducing-the-agentrx-framework/)：分别研究 system/trace/node 多层 Agent 诊断，以及用受 guard 条件约束和证据日志定位首个不可恢复失败；论文、发布方数据和本地复现应分开记账。

本章把上述路线按行为测试、生产就绪、跨模型 benchmark、程序 oracle 和工作流 trace 分开使用；其组合关系和限制见[错误分析与回归测试研究记录](research/error-analysis-regression-testing-depth-review-2026-10-27.md)。

#### Eval Platform 与可复现评估

1. [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110)：通过 scenarios、adaptations 和 metrics 固定横向比较条件，并公开 prompts/completions；统一条件改善可比性，但覆盖受可行性约束，不能替代目标产品数据。
2. [Eval-Driven System Design](https://developers.openai.com/cookbook/examples/partners/eval_driven_system_design/receipt_inspection)：官方案例把失败样本分析、数据集/grader、版本比较和迭代连成应用开发闭环；属于工程案例，不是跨任务收益的独立实验。
3. [MLflow Tracking](https://mlflow.org/docs/latest/tracking/) 与 [GenAI Evaluation Datasets](https://mlflow.org/docs/latest/genai/datasets)：记录 run 参数、代码版本、指标、输入数据元信息和 artifacts；能作为实验账本实现参考，不会自动保证标签、统计设计或评估结论有效。
4. [OpenTelemetry GenAI Semantic Conventions](https://github.com/open-telemetry/semantic-conventions-genai)：为模型调用和 evaluation event 约定共享字段，便于将分数、标签、解释和 response ID 关联回 trace；字段规范不提供 ground truth，且 prompt、输出和工具参数可能含敏感数据。
5. [OpenAI Evaluation Best Practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)、[Working with evals](https://developers.openai.com/api/docs/guides/evals) 和 [Trace grading](https://developers.openai.com/api/docs/guides/trace-grading)：用于核对供应商的 dataset、grader、run 和 agent trace 工作流。官方页面于 2026-10-27 标注既有 Evals 托管平台计划在 2026-10-31 转为只读、2026-11-30 关闭；功能时间线需按页面重查，不能把具体 API 当成稳定架构边界。

本章各类评估框架、工作流、追踪系统与 trace 规范分别解决的问题、收益和限制见[评估体系与平台深度研究记录](research/evaluation-platform-design-depth-review-2026-10-27.md)。

#### Reasoning、Code、Math Eval

1. [GSM-Symbolic](https://arxiv.org/abs/2410.05229)：符号模板控制数值实例与无关条件，帮助观察数学答案对题面变化的敏感性；论文结果不证明模型内部推理机制。
2. [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) 与 [PRM800K](https://github.com/openai/prm800k)：对照 outcome/process supervision，并提供步骤级人工标签；训练结果绑定 MATH 与论文 recipe。
3. [ProcessBench](https://aclanthology.org/2025.acl-long.50/)：评估对数学过程最早错误步骤的定位；步骤标注分歧、难度和样本筛选需一并审查。
4. [EvalPlus / HumanEval+](https://papers.neurips.cc/paper_files/paper/2023/hash/43e9d647ccd3e4b7b5baab53f0368686-Abstract-Conference.html)：LLM 构造 seed、类型感知变异和差分执行，用来审计函数级测试是否漏错；更多测试仍不是完整正确性证明。
5. [LiveCodeBench](https://arxiv.org/abs/2403.07974)：按时间收集竞赛题并评测代码生成、修复和输出预测；需要绑定题目快照和模型 cutoff。
6. [SWE-bench Verified 初始设计](https://openai.com/index/introducing-swe-bench-verified/) 与 [2026 后续审计](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)：从 issue/patch 及 `FAIL_TO_PASS` / `PASS_TO_PASS` 测试理解仓库级任务，并对照后续公开审计揭示的测试缺陷与题目暴露风险。后续审计是 OpenAI 自身报告，不是独立共识。

上述 DOI 的书目身份已通过 Crossref 核验，出版商落地页本次返回 403；[核验范围与限制](research/evaluation-debug-practical-review-2026-10-22.md)另记。

#### 幻觉、拒答与标签可靠性

1. [Evaluating large language models for accuracy incentivizes hallucinations](https://www.nature.com/articles/s41586-026-10549-w)：分析准确率式评估对拒答与无依据回答的激励，并讨论开放式 rubric；结论应按论文的任务和评估设定理解。
2. [Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights](https://aclanthology.org/2026.acl-long.680/)：提出 TRIVIA+ 长上下文 RAG 幻觉检测数据与标签噪声分析；人工标签和噪声条件的结果不能直接外推到所有检测基准。

### 训练

Scaling Law、Chinchilla、数据配比、分布式训练、训练稳定性。

建议补读论文线：

1. [Scheduled Sampling](https://arxiv.org/abs/1506.03099) 与 [Huszár 对其目标的分析](https://arxiv.org/abs/1511.05101)：训练上下文/生成上下文差异及其方法争议。
2. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) 与 [Rethinking the Inception Architecture](https://arxiv.org/abs/1512.00567)：软目标蒸馏和 label smoothing 的早期研究入口。
3. [Reconciling Modern Machine-Learning Practice and the Classical Bias–Variance Trade-Off](https://arxiv.org/abs/1812.11118) 与 [Deep Double Descent](https://arxiv.org/abs/1912.02292)：插值区间和不同复杂度轴上的双下降实验。
4. [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977) 与 [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)：不确定性类型和分类校准；结论需按论文任务阅读。
5. [Understanding the Difficulty of Training Deep Feedforward Neural Networks](https://proceedings.mlr.press/v9/glorot10a.html) 与 [On the Difficulty of Training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063)：初始化尺度与梯度消失/爆炸问题。
6. [Adam](https://arxiv.org/abs/1412.6980) 与 [Decoupled Weight Decay Regularization / AdamW](https://arxiv.org/abs/1711.05101)：自适应矩估计、偏差修正和解耦 weight decay；算法公式不替代目标框架的版本化实现语义。
7. [Adafactor](https://arxiv.org/abs/1804.04235)、[Lion](https://arxiv.org/abs/2302.06675) 与 [Sophia](https://arxiv.org/abs/2305.14342)：分别比较 factored second moment、符号动量和带裁剪的对角曲率估计；收益需按论文模型、数据和训练预算解释。
8. [Shampoo](https://proceedings.mlr.press/v80/gupta18a.html) 与 [SOAP](https://arxiv.org/abs/2409.11321)：矩阵协方差预条件和在预条件器特征基中执行 Adam；SOAP 论文的 LLM 结果绑定 360M/660M 模型及其大 batch 训练设置。
9. [GaLore](https://arxiv.org/abs/2403.03507) 与 [Muon](https://arxiv.org/abs/2502.16982)：低秩梯度状态和矩阵正交化路线；前者的 rank/投影刷新、后者的参数分组/update RMS 都是算法条件。
10. [Deconstructing What Makes a Good Optimizer for Language Models](https://arxiv.org/abs/2407.07972)、[Fantastic Pretraining Optimizers and Where to Find Them](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ea4f0a6ad7088c9b93f22d74c0c9e8c2-Abstract-Conference.html)（[预印本](https://arxiv.org/abs/2509.02046)）：对照优化器选择、超参搜索预算、模型规模和学习率衰减下的排名变化；后者 2025 年预印本发表于 ICLR 2026，论文设置不构成普适排名。
11. [Shampoo 与 Muon 的跨尺度超参迁移研究](https://arxiv.org/abs/2512.05620) 和 [SOAP, Muon, and Beyond](https://arxiv.org/abs/2607.20548)：分别比较 μP/weight decay 缩放，以及多十亿参数和大 batch 下的 SOAP/Muon；结论绑定各自的数据、规模、schedule、硬件与实现。
12. [Understanding Warmup-Stable-Decay Learning Rates](https://arxiv.org/abs/2410.05192)：River Valley 假设、WSD-S 单分支 checkpoint 路径及其实验边界。
13. [A Theory on Adam Instability in Large-Scale Machine Learning](https://arxiv.org/abs/2304.09871)、[Adaptive Preconditioners Trigger Loss Spikes in Adam](https://arxiv.org/abs/2506.04805v2)、[SPAM](https://arxiv.org/abs/2501.06842v2) 与 [GradientStabilizer](https://proceedings.mlr.press/v306/huang26bb.html)：分别分析 Adam 更新方向/时间相关梯度估计、二阶矩失配、尖峰检测与动量重置、梯度幅值稳定；模型规模、阈值和实验配方不同，不应将结果视为同一设置下的排名。
14. [On Layer Normalization in the Transformer Architecture](https://proceedings.mlr.press/v119/xiong20b.html)、[DeepNet](https://arxiv.org/abs/2203.00555) 与 [Peri-LN](https://proceedings.mlr.press/v267/kim25u.html)：对照 Pre-LN 梯度路径、深度相关残差/初始化缩放和外围归一化位置；各自架构与初始化条件不同，证据不能直接迁移为现成模型的热切换方案。

- [LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7428310c0f97f1c6bb2ef1be99c1ec2a-Abstract-Conference.html)：研究低秩子空间中的 optimizer moment 表示；作者的质量和收敛报告绑定论文模型、目标层与训练设置，不等于所有 LoRA 实现都采用该优化器。

分布式训练框架的当前接口可对照 [PyTorch FSDP2 fully_shard](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)、[NVIDIA Megatron Core 并行策略指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)、[NeMo Distributed Checkpoint](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/checkpoints/dist_ckpt.html) 和 [DeepSpeed ZeRO 教程](https://www.deepspeed.ai/tutorials/zero/)；这些动态文档说明公开接口和候选配置，不提供目标集群的性能保证。

2026 年分布式系统增量：[TorchTitan graph-based CPU activation offloading](https://docs.pytorch.org/devlogs/distributed/2026-06-23-cpu-offloading/)展示图级激活筛选、view replay、NUMA、预算/defer/prefetch 与 HBM—带宽取舍；[PyTorch 2.13 distributed release notes](https://pytorch.org/blog/pytorch-2-13-release-blog/)引入 API Unstable 的 `torchcomms` 方向，强调 collective tracing 和部分通信组恢复。前者实验绑定其 8×H100 配置，后者的通信后端能力不等于训练状态自动恢复。

[SPMD Types in TorchTitan](https://docs.pytorch.org/devlogs/distributed/2026-08-26-spmd-types-in-torchtitan/)记录该项目在计算区域迁移到 SPMD typechecking、同时保留 FSDP2/DTensor 持久化状态的实现边界；[PyTorch DTensor 文档](https://docs.pytorch.org/docs/stable/distributed.tensor.html)用于核对 `DeviceMesh`、`Replicate`、`Shard`、`Partial` 等布局接口；[TorchTitan config-based sharding 工程记录](https://docs.pytorch.org/devlogs/distributed/2026-08-17-config-based-sharding-full-dtensor/)展示 module boundary placement 的配置路线。前两者分别支持项目实现和公共布局接口，均不构成跨框架默认配置或性能保证。

### 在线策略蒸馏与学生分布监督

1. [MiniLLM](https://arxiv.org/abs/2306.08543) 与[微软作者实现](https://github.com/microsoft/LMOps/tree/main/minillm)：序列级 reverse-KL、policy-gradient 时间信用、教师混合采样和长度归一化；代码仓库适合核对数据和实现入口，复现需按其依赖与硬件条件重建。
2. [GKD / On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649)：学生自生成状态上的条件分布拟合，比较学生轨迹比例与 forward/reverse KL、JSD 目标。
3. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)：§4.5 和表 21 展示 off-policy 初始化后继续 OPD 的路线，并在同一 Qwen3-8B 初始化上报告 RL/OPD 指标与 GPU-hours；结果绑定报告所用模型、任务和训练流程。
4. [A Survey of On-Policy Distillation for Large Language Models](https://arxiv.org/abs/2604.00626)：2026-10-21 修订 v5 的在写综述，按 student state、teacher feedback 和 policy update 整理方法；用于建立分类图，不取代原论文实验。
5. [A Formula-Driven Survey and Research Agenda for On-Policy Distillation](https://arxiv.org/abs/2606.22793)：从采样估计、时间信用与词表支持拆分目标，并提出待研究议程；其中 GAE-OPD、CR-OPD 等是研究假设，不是已验证通用方案。
6. 后续代表论文：[REOPOLD](https://arxiv.org/abs/2603.11137)、[Entropy-Aware OPD](https://arxiv.org/abs/2603.07079)、[TrOPD](https://arxiv.org/abs/2606.01249)、[On the Position Bias of OPD](https://arxiv.org/abs/2606.22600)、[vOPD](https://arxiv.org/abs/2605.07865)、[RG-OPD](https://arxiv.org/abs/2607.04037)、[MOPD](https://arxiv.org/abs/2605.12652) 与 [TOP-D](https://arxiv.org/abs/2607.04751)：分别对照 sampled reward clipping、熵/位置/信赖区加权、control variate、verifier gate、多 rollout peer context 和 proximal teacher；分数绑定各文论文的模型、数据、预算和评测。
7. 工程入口：[Thinking Machines Lab 的 OPD 实践文章](https://thinkingmachines.ai/blog/on-policy-distillation/)、[TRL DistillationTrainer](https://huggingface.co/docs/trl/main/distillation_trainer) 与 [TRL GKDTrainer](https://huggingface.co/docs/trl/gkd_trainer)。工程文章是团队自身实验，TRL 页面为动态文档；阅读时固定版本、代码 revision 和指标分母。
8. [DeepSeek-V4 Technical Report](https://arxiv.org/abs/2606.19348) §5.1–5.2：多教师 reverse-KL、full-vocabulary logit distillation，以及集中存储/参数分片/hidden-state 缓存/教师输出头调度。报告是该模型的作者披露；路由细节与 OPD 对照消融未完整公开，不能把其系统设计当作通用配方。
9. 比较边界：[Model Soups](https://proceedings.mlr.press/v162/wortsman22a.html) 研究兼容 fine-tuned checkpoints 的权重平均；它与学生状态上的 logit distillation 是不同合并对象，也不能从该论文的图像/迁移实验推导出多领域 LLM 专家可直接平均。

### 对齐

InstructGPT、RLHF、DPO、Constitutional AI、RLAIF、KTO、ORPO。

后训练目标的统一视角：[Towards a Unified View of Large Language Model Post-Training](https://arxiv.org/abs/2509.04419)（v2, 2026-01-20）把不同方法拆成稳定掩码、reference-policy 分母、advantage 和似然梯度等选择，并提出 Hybrid Post-Training；实验主要覆盖数学推理与分布外数学套件。适合用来提出受控比较问题，不能把作者算法的结果当成任意任务的默认结论。

建议补读论文线：

1. [InstructGPT](https://arxiv.org/abs/2203.02155)：示范数据、偏好排序和 RLHF 的代表性研究；论文设置不代表所有对齐流程。
2. [Direct Preference Optimization / DPO](https://arxiv.org/abs/2305.18290)：从 KL 约束下的偏好优化推导 pairwise 目标；阅读时区分论文目标、toy loss 与具体训练器的 mask/长度实现。
3. [IPO](https://arxiv.org/abs/2310.12036)：从偏好学习假设与有限 margin 目标分析离线 preference loss；不要把有限目标间隔解释成已经解决了全模型过优化。
4. [KTO](https://arxiv.org/abs/2402.01306)、[ORPO](https://arxiv.org/abs/2403.07691) 与 [SimPO](https://arxiv.org/abs/2405.14734)：比较单样本 desirability feedback、SFT+odds-ratio、reference-free 平均 log-prob margin 的数据形态与 reference 依赖；各自实验规模和设置不同，不能只按“更简单”排优劣。
5. [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)、[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 与 [DAPO](https://arxiv.org/abs/2503.14476)：对照组内相对 reward、分阶段 reasoning 训练与 dynamic sampling / Clip-Higher / token reduction 等组合 recipe；实验结论受模型、verifier、采样和 loss reduction 配置限制。
6. [Understanding R1-Zero-Like Training / DrGRPO](https://arxiv.org/abs/2503.20783)：通过去掉 response-length 与 group-standard-deviation normalization、使用固定 generation budget 分母讨论 policy-gradient weighting；“unbiased”应放在论文推导和采样假设内理解。
7. [GSPO](https://arxiv.org/abs/2507.18071) 与 [DCPO](https://arxiv.org/abs/2509.02333)：对照 sequence-level importance ratio/clipping 与 token-probability-dependent clipping、SAS、response 内 token mean；同一任务和模型之外的分数不构成通用排名。
8. [RLAIF vs. RLHF](https://arxiv.org/abs/2309.00267) 与 [Constitutional AI](https://arxiv.org/abs/2212.08073)：区分 principle-guided AI feedback、偏好模型训练和 direct-RLAIF；效果结论受具体任务、judge、偏差处理和人工评估协议约束。
9. [Scaling Laws for Reward Model Overoptimization](https://proceedings.mlr.press/v202/gao23h.html) 与 [A Survey of Post-Training Scaling in Large Language Models](https://aclanthology.org/2025.acl-long.140/)：前者用于理解 proxy reward 与 gold reward 的脱钩风险，后者建立 SFT、feedback-based RL、test-time compute 等问题地图；综述分类不能代替原论文和设置内的证据。
10. [Spurious Rewards: Rethinking Training Signals in RLVR](https://proceedings.mlr.press/v306/shao26d.html)：ICML 2026 论文研究随机/错误奖励、GRPO clipping bias 与底模先验的交互；Qwen2.5-Math 的结果不能外推到 Llama、OLMo 或非数学任务，作者明确将 spurious reward 定位为分析对照，不是训练建议。
11. [LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking](https://arxiv.org/abs/2604.15149)：通过 Isomorphic Perturbation Testing 比较实例标签正确与逻辑同构下的规则保持，并做只改变 verifier 的受控训练；当前证据限于 SLR-Bench 归纳逻辑任务。
12. [ImpossibleBench](https://openreview.net/pdf?id=SeO4vyAj7E) 与 [Rethinking Verification for LLM Code Generation](https://arxiv.org/abs/2507.06920)：前者用规格/测试冲突测量 Agent 对测试漏洞的利用，后者用 SAGA/TCGBench 研究代码测试套件生成和覆盖质量；二者不能代替 RL policy 的独立能力评估。
13. [Towards Scalable Oversight via Partitioned Human Supervision](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b6f03ce953cf7f07dafa20891d2539e3-Abstract-Conference.html)：研究专家仅排除候选项时的互补标签监督；估计器依赖多选设置、负例抽样和同分布假设，不能直接套到自由文本。
14. [Humans Correct Their Judgments Through Debate, but Weaker AI Models Do Not](https://arxiv.org/abs/2610.28539)：比较指定任务中的人类和 LLM 辩论裁判；论文报告人类收益较稳定、弱模型收益不稳定，结论绑定其实验协议、模型和样本。
15. [Limits to scalable evaluation at the frontier](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4264ee4376776907c0b87ed70b959585-Abstract-Conference.html)：分析 judge 去偏可节省 gold label 的理论限制；不能据此推断具体项目的校准集可按固定比例缩减。
16. [The Geometry of LLM-as-Judge](https://arxiv.org/abs/2606.03043)：2026 年预印本，在指定 Indic 语料、语言与 judge 上比较模型间一致性和人类一致性；共识不等于人类校准。
17. [Natural Emergent Misalignment from Reward Hacking in Production RL](https://arxiv.org/abs/2511.18397)：在研究模型掌握 reward-hack 知识并进入可利用的真实训练编码环境后，观察到跨任务行为变化；不代表 Claude 产品线上 misalignment 发生率。
18. [Measuring Reward-Seeking via Contrastive Belief Updates](https://arxiv.org/abs/2607.18966)：用反事实 grader 信念变化比较行为敏感度；趋势证据来自单一 o3 RL lineage 和短程代码任务，方法仍需跨模型与长时程验证。
19. [Large Language Model Post-Training: A Unified View of Off-Policy and On-Policy Learning](https://arxiv.org/abs/2604.07941)：按训练轨迹来源区分外部轨迹与 learner-generated rollout，并用 support expansion、policy reshaping 和 behavioral consolidation 解释方法角色；这是 2026 年综述的概念框架，不是固定的后训练流水线或通用因果结论。
20. [The Art of Scaling Reinforcement Learning Compute for LLMs](https://proceedings.iclr.cc/paper_files/paper/2026/hash/75ea88aca02640cca54c1231db8cacdd-Abstract-Conference.html)：比较 LLM RL 配方随计算预算扩大的表现，讨论 loss aggregation、normalization、curriculum 和 off-policy 数据对计算效率的影响；结果绑定作者配方和评估任务。
21. [Understanding the Performance Gap in Preference Learning: A Dichotomy of RLHF and DPO](https://proceedings.mlr.press/v306/shi26s.html)：在特定模型类、表达能力失配和有限样本假设下比较 RLHF、DPO 与在线变体；这是理论分析，不是跨数据集方法排行榜。
22. [TRL Logging](https://huggingface.co/docs/trl/main/en/logging) 与 [TRL v1.16.0 Async GRPO Trainer](https://huggingface.co/docs/trl/v1.16.0/en/async_grpo_trainer)：核对文档版本对应的 KL/rollout 指标和异步样本陈旧度监控；动态文档和版本化训练器页面只说明相应公开接口，不构成统一算法定义。
23. [Shallow Beliefs: Synthetic document finetuning does not inoculate against emergent misalignment from reward hacking](https://arxiv.org/abs/2609.14998)：在 Llama-3.3-70B-Instruct 与特定 reward-hacking 编码环境中发现 SDF 表达的 inoculating belief 未阻止 EM；结论限于论文训练配方和评估。
24. [Scaling Laws For Scalable Oversight](https://proceedings.neurips.cc/paper_files/paper/2025/file/8c3caae2f725c8e2a55ecd600563d172-Paper-Conference.pdf)：用 Mafia、Debate、Backdoor Code 和 Wargames 等监督游戏研究 nested oversight；数值绑定论文游戏和 Elo 配置，不表示生产监督成功率。
25. [TRL GRPOTrainer 文档](https://huggingface.co/docs/trl/grpo_trainer)：核对当前公开 trainer 的 reward function、loss normalization、clipping 与 metrics 字段；文档随版本变化，不能替代论文实验或目标代码 revision 审查。

26. [What Matters in Data for DPO?](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3f37b8fbd43303106dd141a602838ad5-Abstract-Conference.html)：在论文指定的偏好数据、模型和任务中比较 chosen/rejected 数据属性，并分析 chosen-response SFT 与在线 DPO；不据此推断 rejected 样本普遍无用。
27. [Group-in-Group Policy Optimization for LLM Agent Training / GiGPO](https://proceedings.neurips.cc/paper_files/paper/2025/file/420c9f777c0b4f78d515e53cf74d58b2-Paper-Conference.pdf)：把 episode 级相对 advantage 与同一 environment state 下的局部动作比较结合；结果受 ALFWorld、WebShop、搜索 QA、Qwen2.5 与论文训练设置约束。
28. [DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training, v2](https://arxiv.org/abs/2604.26256v2)：研究多版本 rollout、长尾轨迹调度与 staleness 控制；吞吐结果绑定预印本 v2 报告的系统、任务、基线和 policy window。
29. [Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale, v2](https://arxiv.org/abs/2604.11554v2)：面向异构模态与多轮 Agent rollout 的服务隔离和异步队列；报告中的 speedup 与 reward convergence 只支持论文 v2 的模型、部署与对照设置。

### 领域适配与行业任务评估

1. [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964) 与 [Continual Learning for Large Language Models: A Survey](https://arxiv.org/abs/2402.01364)：分别阅读 DAPT/TAPT 的问题设置，以及 LLM 继续预训练、指令微调、对齐和 RAG/模型编辑的阶段差异；前者是 RoBERTa 分类实验，不能直接代表生成式 LLM 效果。
2. [FinGPT: Open-Source Financial Large Language Models](https://arxiv.org/abs/2306.06031) 与 [FinGPT: Democratizing Internet-scale Data for Financial Large Language Models](https://arxiv.org/abs/2307.10485)：开放金融数据采集、LoRA/QLoRA 适配和 RLSP 市场信号案例；股价变化是带混杂因素的弱标签，不等于新闻造成的收益或专家标注。
3. [FinBen](https://proceedings.neurips.cc/paper_files/paper/2024/file/adb1d9fa8be4576d28703b396b82ba1b-Paper-Datasets_and_Benchmarks_Track.pdf)：NeurIPS 2024 发表版本覆盖 42 个数据集、24 项任务；论文发现指令微调对文本分析有帮助，但复杂问答、生成和预测仍有缺口。地区、语言、数据量与模型范围是外推边界。
4. [MedAlign](https://doi.org/10.1609/aaai.v38i20.30205) 与 [TIMER](https://www.nature.com/articles/s41746-025-01965-9)：对照临床人员生成的 EHR 工作流指令与纵向病历时间分布控制；TIMER 的结果绑定 STARR 数据、生成式指令数据及其 Llama/Qwen 训练和评估协议。
5. [DeepSeekMath](https://arxiv.org/abs/2402.03300)：连接领域继续预训练、数学 SFT 与可验证结果奖励 RL 的代表案例；MATH/GSM8K 结果只支持该数学模型、数据和判分设置，不能迁移为开放领域奖励有效性的证据。

### 后训练案例：厂商公开了什么，仍有哪些未知量

1. [GLM-5.3 官方博客](https://z.ai/blog/glm-5.3)：用于核对长周期 coding environment、verifier solvability / shortcut checks、Megatron/SGLang rollout 系统与训练-采样 log-prob 对齐披露。页面中的数字是发布方报告，需保留硬件、轨迹和评测边界；HTML 抽取若不可用，不应把搜索摘要写成已独立复核的技术报告。
2. [OpenAI Introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/)：明确披露 SFT 和 high-compute RL 阶段、CoT/工具目标、无直接 CoT 监督；没有公开 optimizer、reward、KL 和 rollout 规模。
3. [xAI Introducing Grok 4.7](https://x.ai/news/grok-4-7)：披露更大的 base model、更长 RL run 和偏多小时任务的训练混合；缺少具体算法、reward 和消融，不能将模型差异全归因于 RL 时长。

### Reasoning 与 Test-Time Compute

1. [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) 与 [Zero-shot Reasoners](https://arxiv.org/abs/2205.11916)：分别作为少样本和零样本分步提示的研究入口；提示效果不能外推为所有模型和任务上的稳定推理提升。
2. [Language Models Do Not Always Say What They Think](https://arxiv.org/abs/2305.04388)：用干预研究检验可见 CoT 与模型答案之间的忠实性边界。
3. [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)：多路径采样、答案抽取和一致性聚合；结果依赖任务、采样和答案判定协议。
4. [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)、[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) 与 [PRM800K](https://github.com/openai/prm800k)：结果选择、过程监督和步骤级标注的研究/数据入口，需区分论文实验与本地 toy verifier。
5. [Tree of Thoughts](https://arxiv.org/abs/2305.10601)、[Language Agent Tree Search](https://arxiv.org/abs/2310.04406) 与 [AlphaZero](https://arxiv.org/abs/1712.01815)：分别阅读 thought 搜索、Agent 环境反馈和策略/价值/MCTS 组合；具体状态、奖励和预算不能跨任务照搬。
6. [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) 与 [Large Language Monkeys](https://arxiv.org/abs/2407.21787)：分析特定实验设置下的推理预算分配和重复采样收益；不能据此推断所有请求都应提高预算。
7. [CodeRL](https://arxiv.org/abs/2207.01780) 与 [Teaching Large Language Models to Self-Debug](https://arxiv.org/abs/2304.05128)：分别阅读测试反馈驱动的代码生成训练和基于反馈的迭代修复；论文任务与评测设置不等于真实仓库代码维护。
8. [Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3e22bea3b170f4c2aebb9c48d98ae64d-Abstract-Conference.html) 与 [Rethinking Fine-Tuning when Scaling Test-Time Compute](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e8f4eae0a41cab67fdead3aa6b77f083-Abstract-Conference.html)：分别研究数学任务的思考长度分布，以及 CE 训练与 pass@N 搜索间的目标错配；结论限于各自模型、数据和推理预算。
9. [PaCoRe](https://aclanthology.org/2026.acl-long.1253/) 与 [ParaThinker](https://aclanthology.org/2026.findings-emnlp.1469/)：并行推理轨迹的消息协调和单模型多路径设计；精度、延迟、KV reuse 结果绑定论文设置。
10. [Timely Machine](https://aclanthology.org/2026.acl-long.211/) 与 [ThinkRetrieve](https://aclanthology.org/2026.findings-emnlp.1703/)：分别研究工具时延下的 wall-clock test-time compute，以及检索已求解推理轨迹；后者不应与检索事实知识混为一谈。
11. [RAG over Thinking Traces Can Improve Reasoning Tasks](https://aclanthology.org/2026.emnlp-main.1964/) 与 [FOL-Traces](https://aclanthology.org/2026.findings-eacl.115/)：分别研究思考轨迹检索与程序化核验的一阶逻辑过程评估；任务、语料和模型集合各有边界。
12. [KisMATH](https://aclanthology.org/2026.tacl-1.59/)：把数学推理轨迹表示为因果 CoT 图并进行结构干预；适合研究特定步骤与答案的关系，不等于普遍的 CoT 忠实性证明。
13. [Learning from Synthetic Data Improves Multi-hop Reasoning](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9e95016eceae77e945b84554b0f1bb49-Abstract-Conference.html) 与 [Native Reasoning Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ccf6d8b4a1fe9d9c8192f00c713872ea-Abstract-Conference.html)：分别研究规则合成数据上的多跳推理训练与从标准 QA 对生成潜在推理轨迹的训练目标；结论需绑定各论文实验协议。
 14. [Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and Reproducibility](https://arxiv.org/abs/2608.04001)：2026 年 arXiv 预印本，按计算发生在单条未完成轨迹、完整候选的终局聚合或未完成前缀搜索来分类，并区分精确重放与分布可复现。分类和评估建议作为分析框架使用；其经验结果受论文模型、任务和推理协议限制。
15. [FACT-E: Causality-Inspired Evaluation for Trustworthy Chain-of-Thought Reasoning](https://aclanthology.org/2026.findings-acl.1014/)：联合估计步骤间忠实性和 CoT 到答案的一致性；结果限于其 GSM8K、MATH、CommonsenseQA 与模型设置，标准评估增加模型调用成本。
16. [Hidden Thoughts Are Not Secret: Reasoning-Trace Exposure in LLMs](https://aclanthology.org/2026.findings-emnlp.2303/)：研究接口隐藏轨迹后仍可能出现可用于迁移的用户可见文本；证据来自受限开源模型实验，相似度不证明恢复了完整内部状态。
17. [TRACE: An Evidence-Grounded Benchmark for Safety Evaluation of Large Reasoning Models](https://aclanthology.org/2026.emnlp-main.1174/)：分开评估 prompt、推理轨迹、最终回复及证据归因；覆盖中英文和有限模型/风险集合，不能视作普遍安全率。
18. [Scaling up Test-Time Compute with Latent Reasoning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3b01972cf31e6fa0fe29e4b8b5c2a0a1-Abstract-Conference.html)：以循环深度在潜在状态中扩展测试时计算；作者报告属于单个 3.5B/800B-token 概念验证，训练与基准比较有明确限制。

### 推理优化

FlashAttention、PagedAttention、Speculative Decoding、量化论文。

建议补读论文线：

1. [FlashAttention](https://arxiv.org/abs/2205.14135)、[FlashAttention-2](https://arxiv.org/abs/2307.08691) 与 [FlashAttention-3](https://arxiv.org/abs/2407.08608)：依次阅读 IO-aware tiling/online softmax、并行工作划分、Hopper 异步调度与 FP8 误差评估；加速结果绑定具体 GPU、shape、dtype 和 baseline。
2. [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)：KV block 管理、请求生命周期和 serving 系统设计。
3. [vAttention](https://arxiv.org/abs/2405.04437)：把动态 KV 显存管理放到虚拟内存 / 按需映射视角下理解 PagedAttention 的替代路线；其性能对照绑定论文的 A100、模型与 kernel。
4. FlashInfer：从 serving kernel、batching、attention backend 和框架集成角度理解推理性能优化。
5. [Speculative Decoding](https://arxiv.org/abs/2211.17192)：draft/target 接受与修正规则；若候选分布、采样处理或验证实现改变，需重新确认分布保证。
6. [Medusa](https://arxiv.org/abs/2401.10774) 与 [EAGLE](https://arxiv.org/abs/2401.15077)：多预测头与隐藏状态 draft 路线，训练目标、候选树和部署成本各不相同。
7. 权重量化可比较 [GPTQ](https://arxiv.org/abs/2210.17323) 的近似二阶误差补偿与 [AWQ](https://arxiv.org/abs/2306.00978) 的 activation-aware 通道缩放；激活 outlier 可比较 [LLM.int8()](https://arxiv.org/abs/2208.07339) 的 FP16 旁路与 [SmoothQuant](https://arxiv.org/abs/2211.10438) 的等价难度迁移。论文的质量指标和 kernel 路径不同，不能直接按倍率排名。
8. Serving 调度可对照 [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)、[FastServe（NSDI 2026）](https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang)、[Sarathi-Serve](https://arxiv.org/abs/2403.02310)、[DistServe](https://arxiv.org/abs/2401.09670) 与 [PPD multi-turn disaggregation](https://arxiv.org/abs/2603.13358)：分别阅读 iteration-level scheduling、token-level preemption、chunked prefill、prefill/decode 分离与多轮 append-prefill 路由。FastServe 早期 arXiv 与正式发表版报告的倍率不同，引用时固定版本；各研究结果仍绑定模型、硬件、长度分布与 SLO。
9. [AgentPerfBench](https://arxiv.org/abs/2609.34683)：2026-09 预印本用真实 Agent trace 的每轮输入/输出长度、轮数分布和饱和测量研究 Agentic inference；作者的硬件结果绑定其模型、GPU、runtime 和负载，不能当作通用性能倍率。
9. [A Survey of LLM Inference Systems](https://arxiv.org/abs/2506.21901)：用作请求处理、batching、调度、KV 管理、量化和多副本架构的文献地图；对其归纳出的具体机制仍应回到原论文和框架版本核验。
10. [Revisiting SLO and Goodput Metrics in LLM Serving](https://arxiv.org/abs/2410.14257)：对照请求/令牌级 SLO、goodput 与 smooth-goodput，讨论如何避免“吞吐提升但用户体验或请求完成率变差”的指标漏洞。
11. [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677)：分析 prefill 与 decode 分离、异构资源配置和 KV 跨机传输；论文吞吐/成本结果绑定其生产 trace、GPU、集群设计与成本模型，不能当作通用收益。
12. [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785)：理解算术强度、计算峰值、带宽上限与 ridge point 的经典性能模型；应匹配运算精度和内存层级，并把它限制为 kernel/阶段分析工具。
13. [H₂O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048) 与 [StreamingLLM](https://arxiv.org/abs/2309.17453)：分别保留累计 attention heavy hitters + recent tokens、attention sinks + recent window；都通过有限预算丢弃 KV，解决目标和质量边界不同。
14. [Towards Efficient LLM Serving: A Survey on System-Aware KV Cache Optimization](https://arxiv.org/abs/2607.08057)：按 KV execution/scheduling、memory placement/migration、representation compression 与 retention 汇总研究谱系；survey taxonomy 用来找原论文，不替代逐项核验。
15. KV Cache 可对照 [KIVI](https://arxiv.org/abs/2402.02750) 的 K-per-channel/V-per-token 与 residual window、[KVQuant](https://arxiv.org/abs/2401.18079) 的 pre-RoPE key/敏感度加权非均匀码本/稀疏 outlier，以及 [TurboQuant](https://arxiv.org/abs/2504.19874) 的旋转量化/QJL 内积校正；三者的 bit 数、系统代价和实测 workload 不同，向量失真、下游任务质量和 serving 吞吐不能混作一个结果。
16. [A Survey on Model Compression for Large Language Models](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00704/125482/A-Survey-on-Model-Compression-for-Large-Language)：用作 pruning、quantization、distillation 与 weight-only/weight-activation/KV cache quantization 的分类地图；机制和实测仍需回到原始论文。
17. [A Comprehensive Evaluation on Quantization Techniques for Large Language Models](https://arxiv.org/abs/2507.17417)：在统一设置下拆解量化前变换和误差补偿，并检查粒度、对称性、MXFP4/NVFP4 scale；结论绑定作者评测的模型与配置。
18. [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) 与 [Google Research 技术说明](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)：阅读理论失真、长上下文质量和 H100 attention-logit microbenchmark，并区分作者论文结果与服务指标。
19. [vLLM TurboQuant accuracy/performance study](https://vllm.ai/blog/2026-05-11-turboquant)：固定 vLLM 0.20.2 commit、模型、H100 数量和 retrieval/reasoning/serving workload，观察低 bit cache 的准确率—容量—TPOT 权衡；不可和论文向量 microbenchmark 混比。
20. [OCP Microscaling Formats MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) 与 [NVIDIA Transformer Engine NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)：核对 MXFP4 的 E2M1/32-value E8M0 block scale 与 NVFP4 的 E2M1/16-value E4M3 scale/FP32 tensor scale。
21. 比较 [DistServe](https://arxiv.org/abs/2401.09670)、[Mooncake](https://arxiv.org/abs/2407.00079) 与 [TaiChi](https://arxiv.org/abs/2508.01989)：分别关注 P/D 资源解耦、KV-centric 调度/分层缓存、以及按 TTFT/TPOT SLO 在共置和解耦间自适应；请求吞吐、goodput、trace 和 SLO 指标不可混排。
22. [SmartGen](https://arxiv.org/abs/2607.28150) 与 [SpectrumKV](https://arxiv.org/abs/2606.08635)：研究 P/D 阶段 KV 网络传输的选择性传输与 per-token mixed precision；TTST、KV 预算和困惑度分别绑定各论文模型、网卡和 workload。
23. [DeepEP](https://github.com/deepseek-ai/DeepEP) 与 [vLLM Expert Parallel Load Balancer](https://github.com/vllm-project/vllm/blob/main/docs/serving/expert_parallel_deployment.md)：理解 EP dispatch/combine 通信 kernel、prefill/decode throughput-latency 路线、expert 重映射与冗余专家的显存代价；主分支文档不是目标集群性能报告。
24. [vLLM Disaggregated Prefilling](https://github.com/vllm-project/vllm/blob/main/docs/features/disagg_prefill.md) 与 [SGLang PD Disaggregation](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/pd_disaggregation.mdx)：核对当前 connector、KV handoff 与版本边界；vLLM 当前文档标注 experimental，并说明 disaggregated prefill 本身不保证吞吐提升。
25. [FrugalGPT](https://arxiv.org/abs/2305.05176) 与 [RouteLLM](https://arxiv.org/abs/2406.18665)：比较模型近似/级联阈值与偏好数据路由强弱模型。分别核对串行延迟、router overhead、训练偏好分布和跨 benchmark domain shift；论文节省数字绑定作者任务和 API 价格。
26. Prompt compression 可并读 [LLMLingua](https://arxiv.org/abs/2310.05736)、[LongLLMLingua](https://arxiv.org/abs/2310.06839) 与 [LLMLingua-2](https://arxiv.org/abs/2403.12968)：比较 token 级 coarse-to-fine 选择、question-aware 长上下文/位置偏差处理、蒸馏后的双向 token 分类。压缩器成本、关键事实召回和业务成功成本需在目标样本上测。
27. [Compact Language Models via Pruning and Knowledge Distillation / Minitron](https://arxiv.org/abs/2407.14679)：结构化剪枝与蒸馏从 Nemotron-4 15B 派生 8B/4B；40× training-token 与 1.8× family FLOPs 是作者的特定模型族和从头训练基线，不是通用压缩倍率。
28. [Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization](https://arxiv.org/abs/2509.23202)：分析 MXFP4/NVFP4 的 scale 量化、旋转与 RTN/GPTQ 交互，提出 block-aware MR-GPTQ；准确率、QuTLASS kernel 与加速报告绑定作者模型、zero-shot task 和指定 GPU。
29. [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149)：12B hybrid Mamba-Transformer 的 10T-token 训练报告；二维权重 scaling、Hadamard transform、stochastic rounding 与选择性高精度层构成一套 recipe，不能等同于任意 NVFP4 PTQ。
30. [Scale Learning: Training NVFP4 Block Scales to Recover Accuracy](https://nvidia.github.io/Model-Optimizer/announcements/20260916-scale-learning.html)：官方 Model Optimizer 团队说明冻结权重训练 block scales、联合训练权重/scales 的路径与 Nemotron 3.5 Lightning 结果；单模型报告和当前实现范围不能推广为通用质量收益。
31. [TensorRT 10.x Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)：读取 NVFP4 的显式 Q/DQ block-scale 接口，并与 Transformer Engine 的紧凑 NVFP4 recipe 区分；文档接口不能单独证明导出 checkpoint 的实际 scale dtype 或 engine 路径。
32. [OCP 8-bit Floating Point Specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf) 与 [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)：核对 E4M3/E5M2 的字段、特殊值、范围、舍入/饱和转换及 scale 设计背景；OCP 不定义 KV scale 的选择和维护。
33. [The State of FP8 KV-Cache and Attention Quantization in vLLM](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html)：vLLM 团队对 H100/FlashAttention 3 与 B200/FlashInfer 的长上下文质量、累加精度、滑窗层、ITL slope 和并发吞吐报告；其中 FP8 attention 计算与 FP8 KV 同时开启，数字绑定发布配置。
34. [vLLM Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) 与 [LLM Compressor KV calibration](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_kv_cache/)：核对 per-tensor/per-head scale、数据校准和 backend 限制；[TensorRT-LLM FP8 guide](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html) 展示另一套 cache-dtype 配置入口。rolling docs 使用时需固定 release。
35. [LLM-QAT](https://aclanthology.org/2024.findings-acl.26/)：代表性的 LLM 量化感知训练路线，以教师自生成文本和 logits 软标签训练量化学生；W4A8KV4 结果绑定论文模型、任务和训练配置，不代表任意 serving runtime 支持相同 artifact。
36. [MARLIN](https://arxiv.org/abs/2408.11743)：W4A16 的 GPU 执行 kernel 研究，讨论异步访存、任务调度和流水线；它优化执行，不负责量化质量，性能需绑定 GPU、矩阵形状、batch 与权重布局。
37. [PyTorch FakeQuantize 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)：核对 QAT 前向中的 quantize/dequantize 模拟与 observer 状态；API 文档不等于完整 LLM 训练配方。
38. [The KV Cache Working Set: Online Capacity Planning for LLM Inference Systems](https://arxiv.org/abs/2609.27746)、[UltraQuant: 4-bit KV Caching for Context-Heavy Agents](https://arxiv.org/abs/2606.20474) 与 [Tessera: Demand-Driven KV Cache Management for Retrieval-Augmented LLM Serving](https://arxiv.org/abs/2609.32999v1)：分别研究 trace 驱动的前缀缓存工作集估计、长上下文 Agent 的低位宽缓存，以及检索驱动的跨前缀 KV 复用；性能和容量结论均受各自 workload、硬件和版本范围约束。
15. [I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving](https://www.ndss-symposium.org/ndss-paper/i-know-what-you-asked-prompt-leakage-via-kv-cache-sharing-in-multi-tenant-llm-serving/)：分析多租户 prefix sharing 的时序侧信道风险；攻击结论绑定论文的系统、观测能力和威胁模型。
9. [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)：target hidden feature 的跨层融合、逐层 K/V 注入、block diffusion draft 与 target verification。

当前实现与测量入口：[vLLM Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/)、[vLLM Serving Benchmark 指标](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md)、[GuideLLM Goodput](https://vllm-project.github.io/guidellm/stable/reference/guidellm/schemas/benchmark/goodput/) 与 [NVIDIA Triton Speculative Decoding](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/llm_features/speculative_decoding_by_backend_type.html)。它们分别说明版本化运行指标、客户端 ITL/TPOT 测量点、SLO 过滤后的 goodput 和特定后端推测解码；不能将滚动文档当作跨框架统一接口。

2026-09-29 的 [vLLM Disaggregated Serving 指南](https://vllm-project.github.io/2026/09/29/disaggregated-serving-guide.html) 展示 render、prefill、KV handoff、decode 与 derender 的工程组合，并明确列出传输、解析器重放和请求编排成本。该项目博客适合作为实现案例与待测假设来源，不代表通用的性能提升结论。

#### 解码策略与结构化生成

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751v2)：分析开放式生成中的退化重复并提出 nucleus sampling；论文结果绑定 GPT-2/WebText 设置，不能视为所有任务的采样质量排序。
- [Locally Typical Sampling](https://aclanthology.org/2023.tacl-1.7/)：按 token surprisal 与条件熵的距离构造候选集；作者在摘要和故事生成中报告与 top-k/nucleus 有竞争力并减少退化重复，不能外推成通用聊天优势。
- [Turning Up the Heat: Min-p Sampling](https://arxiv.org/abs/2407.01082) 与 [Min-p, Max Exaggeration: A Critical Analysis of Min-p Sampling in Language Models](https://arxiv.org/abs/2506.13681)：并读提出 min-p 的原论文和对其人评、统计与超参数控制的批判性复核；将方法定义和“普遍质量更好”的实证主张分开。
- [XGrammar](https://arxiv.org/abs/2411.15100) 与 [Flexible and Efficient Grammar-Constrained Decoding](https://arxiv.org/abs/2502.05111)：比较 token-mask 预处理/运行时解析与 subword-tokenizer/CFG sound alignment；mask 吞吐、离线 preprocessing 和端到端 serving 是不同指标。
- [Decoding Speculative Decoding](https://arxiv.org/abs/2402.01528)：通过大规模 draft 选择实验分析 draft 延迟、接受长度和加速的关系；不能把 draft 的语言建模分数当成 speculative 收益的替代指标。
- [Speculative Decoding: Performance or Illusion?](https://arxiv.org/abs/2601.11580) 与[作者维护的配置和结果页](https://specdecode-bench.github.io/)：在 vLLM 多模型、多 workload、多 batch 设置下检查 draft、n-gram、EAGLE 和 MTP；结论绑定论文版本、H100、模型与输出条件。

- [FlashAttention-4](https://arxiv.org/abs/2603.05451)：Blackwell GPU 上的 attention algorithm/kernel co-design；论文数字要绑定 B200、BF16、cuDNN 9.13/Triton 等 baseline 版本和 workload，新版库吸收方法后对照结果会变化。

### Agent、工具协议与 Coding Agent

ReAct、Toolformer、function calling、MCP、A2A、agent evaluation、coding agent runtime、sandbox、trace/replay 和工具安全。

建议补读资料线：

1. [ReAct](https://arxiv.org/abs/2210.03629): Synergizing Reasoning and Acting in Language Models。
2. [Toolformer](https://arxiv.org/abs/2302.04761): Language Models Can Teach Themselves to Use Tools。
3. [OpenAI function calling 官方文档](https://developers.openai.com/api/docs/guides/function-calling)：当前 API 中由应用端执行函数并将输出回传；文档只支持 OpenAI API 的公开调用合同。
4. [Model Context Protocol 官方规范](https://modelcontextprotocol.io/specification/2026-07-28) 和 SDK 文档：规范化跨应用工具/资源互操作，但不替具体宿主应用完成授权、幂等和审计。
5. [SWE-bench](https://arxiv.org/abs/2310.06770): Can Language Models Resolve Real-World GitHub Issues?；代码 Agent 评测绑定各自 harness 与测试执行。
5. Agent benchmark、tool-use benchmark 和 SWE-bench 相关资料。
6. GraphRAG / RAPTOR / Agentic RAG 等结构化、多跳和多轮检索增强路线。
7. [Gorilla](https://arxiv.org/abs/2305.15334)、[API-Bank](https://arxiv.org/abs/2304.08244) 与 [ToolLLM](https://arxiv.org/abs/2307.16789)：工具/API 选择、调用数据和评估基准。
8. [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091)、[Reflexion](https://arxiv.org/abs/2303.11366) 与 [Voyager](https://arxiv.org/abs/2305.16291)：规划、反馈更新和长程环境技能积累的不同研究设置。
9. [Indirect Prompt Injection](https://arxiv.org/abs/2302.12173)：外部检索内容如何影响工具型应用的安全研究。
10. [MemGPT](https://arxiv.org/abs/2310.08560)、[Generative Agents](https://arxiv.org/abs/2304.03442) 与 [LongMem](https://arxiv.org/abs/2306.07174)：运行时记忆、反思规划和长程上下文增强；不构成隐私或删除保证。
11. [AgentDojo](https://arxiv.org/abs/2406.13352) 与 [ToolSandbox](https://arxiv.org/abs/2408.04682)：分别研究 prompt injection 防御及状态化工具使用评估；AgentDojo 首版包含 97 个正常任务和 629 个安全用例，并按环境状态同时评 utility 与 security；测试环境不替代目标系统的权限和副作用审计。
12. [Overcoming the Retrieval Barrier](https://www.usenix.org/conference/usenixsecurity26/presentation/chang-hongyan)：USENIX Security 2026 的检索增强间接注入研究；报告的 retrieval 与端到端 attack rate 绑定论文 threat model 和系统设置。
13. [OpenAI Programmatic Tool Calling](https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling)：特定 Responses / Agents runtime 中用代码编排可预测工具链的官方设计入口；不代表通用 Agent 协议，也不绕过应用侧权限和审批。

#### 长上下文、RAG 与长期记忆

- [LongBench Pro](https://arxiv.org/abs/2601.02872)：以自然产生的双语长文样本扩展长上下文评估；阅读时检查长度、任务类型和证据位置分布，不能据此替代目标语料。
- [LongHarness Bench](https://arxiv.org/abs/2609.38137)：评测长上下文 harness 在多条件检索与跨文档推理任务中的质量—成本差异；结果绑定预印本任务、模型与 harness 配置，不能泛化为 harness 排名。
- [Relevant Is Not Warranted: Evidence-Force Calibration for Cited RAG](https://arxiv.org/abs/2605.28044)：FORCEBENCH 用对比证据检查引用是否足以支持声明的具体范围和强度；报告值绑定 198 对样本与四个 judge。
- [From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents](https://aclanthology.org/2026.findings-acl.1337/) 与 [Mem2ActBench](https://aclanthology.org/2026.acl-long.370/)：分别研究长期个性化记忆的更新/遗忘和任务型 Agent 中的记忆行动使用；任务合同不同，不合并为单一记忆能力分数。
- [AdaMem](https://aclanthology.org/2026.emnlp-industry.189/)、[OP-Bench](https://aclanthology.org/2026.emnlp-main.1168/) 与 [PS-Bench](https://aclanthology.org/2026.acl-long.1260/)：分别研究偏好条件化的选择性记忆写入、无关/重复/迎合式过度个性化，以及个人记忆导致的 intent legitimation；实验范围限于论文中的 persona、样本、模型和评测协议。
- [Agent Memory: Characterization and System Implications of Stateful Long-Horizon Workloads](https://arxiv.org/abs/2606.06448)：按构建、检索和生成路径研究状态化长期记忆的系统成本；论文 workload 与代表系统范围限制其外推。

#### Prompt Injection 与记忆污染评估

- [OWASP LLM01:2025 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)：将 jailbreak 归为 prompt injection 的一种，并区分直接用户输入与间接外部内容；权限最小化、输出校验等是缓解建议，不构成某系统已安全的证据。
- [OWASP GenAI LLM Top 10 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/) 与 [OWASP Agentic Top 10 2026](https://genai.owasp.org/download/52117/)：前者列 LLM01 Prompt Injection，后者以 ASI06 单列跨任务 Memory & Context Poisoning；两者是威胁分类，不是产品安全认证。
- [WASP](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1c9818387f5dd0a0bc151214660f059d-Abstract-Datasets_and_Benchmarks_Track.html)：端到端 Web Agent 注入 benchmark；部分行为妥协和攻击目标完成度分开观察，数字绑定论文任务与 grader。
- [StepJack](https://arxiv.org/abs/2608.06477)：测试分散在多页面、多步骤的 CUA 间接注入；样例、12 个目标、最多三步和一次完整运行限制不支持通用趋势外推。
- [INFUSE](https://arxiv.org/abs/2610.20776)：端到端 Agent 工作流的间接注入 benchmark；LLM judge 的 ASR-L 与 artifact heuristic 的 ASR-H 是不同指标，完整网格以 Mini-SWE-Agent 为主且运行于沙箱。
- [MemSecBench](https://arxiv.org/abs/2607.27080)：以 Write–Execute–Forget 追踪持久化、召回、采纳、副作用及选择性修复；报告的比例属于其 310 个案例、48 个情境和 24 种指定配置。

#### Frontier 工作流 Agent 评测

本章对照 [第七册第十五章](book-07-evaluation-experiments/chapters/15-specialized-frontier-eval-cluster.md)，详细研究记录在[研究笔记](research/specialized-frontier-eval-cluster-depth-review-2026-10-27.md)。

- [SWE-bench](https://arxiv.org/abs/2310.06770) 将仓库 issue、patch 与测试连接；[SWE-Gym](https://arxiv.org/abs/2412.21139) 增加可执行环境和可用轨迹；[Terminal-Bench 2.0](https://arxiv.org/abs/2601.11868) 则测试终端工作流。三者的任务合同和训练/测试边界不同。
- [WebArena](https://arxiv.org/abs/2307.13854) 检查可复现网站任务的功能状态；[OSWorld](https://arxiv.org/abs/2404.07972) 检查真实桌面应用与跨应用流程；[BrowserGym](https://arxiv.org/abs/2412.05467) 提供统一实验接口，不会统一各任务集的评分含义。
- [τ-bench](https://arxiv.org/abs/2406.12045)、[ToolSandbox](https://arxiv.org/abs/2408.04682) 和 [AgentDojo](https://arxiv.org/abs/2406.13352) 分别突出多轮业务状态/pass^k、milestone 与禁止事件、以及注入下效用和安全结果。
- [GAIA](https://arxiv.org/abs/2311.12983)、[ScienceAgentBench](https://arxiv.org/abs/2410.05080) 和 [PaperBench](https://arxiv.org/abs/2504.01848) 从综合问答、可执行数据分析推进到论文实验复现；答案正确、产物可运行和复现进度是不同判定对象。
- [METR time horizon](https://metr.org/time-horizons/) 测任务难度时间刻度上的成功曲线；[Gaia2](https://arxiv.org/abs/2602.11964) 测异步动态环境下的适应。两者不应合成“Agent 能连续运行多久”。
- [Executable-contract audit](https://arxiv.org/abs/2609.37315) 审计工具实现、状态变化和 grader 读取字段；这是评测合同审计方法，不是 benchmark。预印本中的潜在缺陷不等于 Agent 已触发或榜单已变化。

### AI Infra 与 Serving Engine

GPU 集群、分布式训练系统、LLM serving engine、调度、KV Cache 管理、observability、成本治理和平台工程。

建议补读资料线：

1. 分布式训练与并行： [ZeRO](https://arxiv.org/abs/1910.02054)解释 optimizer/gradient/parameter state 逐级分片，[Megatron-LM 张量并行](https://arxiv.org/abs/1909.08053)与[GPU 集群训练](https://arxiv.org/abs/2104.04473)研究 TP/PP/DP 组合及交错流水线，[GPipe](https://arxiv.org/abs/1811.06965)建立 micro-batch 与 flush pipeline，[GShard](https://arxiv.org/abs/2006.16668)和[Switch Transformer](https://arxiv.org/abs/2101.03961)对照 top-2/top-1 稀疏路由，[Reducing Activation Recomputation](https://arxiv.org/abs/2205.05198)研究 TP/SP/选择性重算的 activation 账本，[Ring Attention](https://arxiv.org/abs/2310.01889)研究 blockwise 长上下文 attention，[PaLM](https://arxiv.org/abs/2204.02311)定义 MFU/HFU，[MegaScale](https://arxiv.org/abs/2402.15627)对照 10K+ GPU 下的吞吐、通信重叠、straggler 和故障恢复。配合 [PyTorch DDP 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)、[FSDP2 2.14](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html)、[Distributed Checkpoint 2.14](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html) 和 [Megatron Core CP 指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)核对当前接口与版本语义；Megatron Core 在线文档会持续更新。
2. Megatron-LM、DeepSpeed、FSDP、Ray 等训练系统资料。
3. vLLM、SGLang、TensorRT-LLM、TGI 等推理框架文档和源码导读。
4. Kubernetes GPU 调度、NVIDIA GPU operator、MIG、NCCL 和集群网络资料。
5. LLMOps、模型仓库、评估平台、实验追踪和 artifact 管理资料。
6. 训练显存与数值精度： [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)讨论分段保存与反向重算，[Reducing Activation Recomputation](https://arxiv.org/abs/2205.05198)比较 Transformer selective recompute 与 sequence parallel，[8-bit Optimizers](https://arxiv.org/abs/2110.02861)研究 blockwise optimizer state 量化，[FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)比较 E4M3/E5M2，[QLoRA](https://arxiv.org/abs/2305.14314)研究 NF4、Double Quantization 与分页 optimizer；再用 [PyTorch AMP 2.14](https://docs.pytorch.org/docs/2.14/amp.html)、[checkpoint 2.14](https://docs.pytorch.org/docs/2.14/checkpoint.html) 和 [Transformer Engine FP8 Primer 2.20.2](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html)核对 API 与硬件路径。作者的实验数字需按模型、语料、精度和设备配置阅读。
7. 训练能耗与碳排： [Llama 2](https://arxiv.org/abs/2307.09288)报告以 GPU-hours/TDP 估算训练排放及未纳入的冷却、互联和硬件制造项；[OLMo](https://arxiv.org/abs/2402.00838)披露另一硬件/电网强度下的估算。对照 PUE、电网碳强度、实测功率与 embodied carbon 时，先固定地点、时间、能量边界和来源，不能横向排列不同口径的 tCO₂e。
8. Serving runtime 的当前实现入口： [vLLM V1 architecture](https://github.com/vllm-project/vllm/blob/main/docs/design/arch_overview.md)、[scheduler API](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/) 与 [serving benchmark guide](https://docs.vllm.ai/en/stable/benchmarking/cli/)；[SGLang Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)、[Session-Aware Radix Cache](https://docs.sglang.io/docs/advanced_features/session_radix_cache) 与 [Radix Cache eviction](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy)；[TensorRT-LLM backend-removal migration guide](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html)；[LMDeploy backend support matrix](https://lmdeploy.readthedocs.io/en/stable/supported_models/supported_models.html)；[llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)；[Transformers serving guide](https://huggingface.co/docs/transformers/v5.6.0/serve-cli/serving) 与 [TGI maintenance notice](https://huggingface.co/docs/text-generation-inference/index)。这些是动态文档/仓库；应固定 release，并用目标模型和流量复核。
9. 生产可靠性、服务治理和事故复盘可并读 [Google SRE: Service Level Objectives](https://sre.google/sre-book/service-level-objectives/)、[Google SRE Workbook: Canarying Releases](https://sre.google/workbook/canarying-releases/)、[Google SRE: Postmortem Culture](https://sre.google/sre-book/postmortem-culture/)、[Google SRE Workbook: Incident Response](https://sre.google/workbook/incident-response/)、[NIST SP 800-61 Rev. 3](https://csrc.nist.gov/pubs/sp/800/61/r3/final)、[AWS idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/)、[AWS timeouts/retries/jitter](https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/)、[Microsoft Transactional Outbox](https://learn.microsoft.com/en-us/power-platform/architecture/reference-architectures/transactional-outbox-pattern)、[Kubernetes probes](https://kubernetes.io/docs/concepts/workloads/pods/probes/)、[Prometheus instrumentation](https://prometheus.io/docs/practices/instrumentation/) 与 [OpenTelemetry GenAI metrics](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md)。分别核对 SLI/SLO 与 canary 归因、复盘和行动项、网络安全事件响应、重试意图与幂等、outbox 的至少一次投递、探针职责、指标基数和 LLM 指标边界；这些工程文档不是目标服务性能或安全的验收证据，GenAI 指标约定需按实际版本复核。

### 多模态

1. 视觉对齐与 VLM： [Swin Transformer](https://arxiv.org/abs/2103.14030)、[ViT](https://arxiv.org/abs/2010.11929)、[ConvNeXt](https://arxiv.org/abs/2201.03545)、[CLIP](https://arxiv.org/abs/2103.00020)、[SigLIP](https://arxiv.org/abs/2303.15343)、[Flamingo](https://arxiv.org/abs/2204.14198)、[BLIP-2](https://arxiv.org/abs/2301.12597) 与 [LLaVA](https://arxiv.org/abs/2304.08485)。重点比较双编码器对比学习、查询压缩、交叉注意力和视觉 token 接入。
2. 扩散与连续生成： [DDPM](https://arxiv.org/abs/2006.11239)、[Imagen](https://arxiv.org/abs/2205.11487)、[Latent Diffusion](https://arxiv.org/abs/2112.10752)、[DiT](https://arxiv.org/abs/2212.09748)、[Flow Matching](https://arxiv.org/abs/2210.02747)、[Rectified Flow](https://arxiv.org/abs/2209.03003) 与 [SDXL](https://arxiv.org/abs/2307.01952)；区分可逆密度模型、连续向量场训练目标与扩散采样路径，避免把同名的 flow 术语混为一类。
3. 采样与条件控制： [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598)、[DDIM](https://arxiv.org/abs/2010.02502)、[DPM-Solver](https://arxiv.org/abs/2206.00927)、[Progressive Distillation](https://arxiv.org/abs/2202.00512) 和[扩散噪声 schedule 分析](https://arxiv.org/abs/2305.08891)；区分无需重训的采样器与需要额外训练的蒸馏。
4. 图像编辑： [SDEdit](https://arxiv.org/abs/2108.01073)、[InstructPix2Pix](https://arxiv.org/abs/2211.09800) 与 [ControlNet](https://arxiv.org/abs/2302.05543)；分别核对加噪重绘、语言指令编辑和空间条件控制。
5. 生成评估： [FID](https://arxiv.org/abs/1706.08500)、[The Role of ImageNet Classes in FID](https://arxiv.org/abs/2203.06026)、[CMMD / Rethinking FID](https://arxiv.org/abs/2401.09603)、[FVD](https://arxiv.org/abs/1812.01717)、[Beyond FVD](https://arxiv.org/abs/2410.05203)、[GenEval](https://arxiv.org/abs/2310.11513)、[T2I-CompBench](https://arxiv.org/abs/2307.06350)、[VBench](https://arxiv.org/abs/2311.17982) 与 [VBench-2.0](https://arxiv.org/abs/2503.21755)。指标依赖特征提取器和数据设定，不替代条件遵循、任务正确性和安全评估。
6. 视频生成： [Video Diffusion Models](https://arxiv.org/abs/2204.03458)、[Make-A-Video](https://arxiv.org/abs/2209.14792)、[Imagen Video](https://arxiv.org/abs/2210.02303)、[Lumiere](https://arxiv.org/abs/2401.12945)、[CogVideoX](https://arxiv.org/abs/2408.06072)、[Wan 2.1](https://arxiv.org/abs/2503.20314) 与 [Pyramidal Flow Matching](https://arxiv.org/abs/2410.05954)；Sora 的公开视频表示与系统边界见[OpenAI 2024 技术报告](https://openai.com/index/video-generation-models-as-world-simulators/)和 [Sora 2 System Card](https://openai.com/index/sora-2-system-card/)，不据此推断未披露内部架构。OpenAI Help Center 记录 Sora 网页/应用已于 2026-04-26 停止，API 已于 2026-09-24 停止（[停止服务说明](https://help.openai.com/en/articles/20001152-what-to-know-about-the-sora-discontinuation)）。
7. 自回归图像生成： [DALL·E（dVAE + 自回归 Transformer）](https://arxiv.org/abs/2102.12092) 与 [DALL·E 2（unCLIP 两阶段）](https://arxiv.org/abs/2204.06125)；论文中的模型规模、codebook 与数据配比是各自实验选择，不是通用超参数。
7. 语音模型方法与评估： [SpeechLM 综述（ACL 2025）](https://aclanthology.org/2025.acl-long.682/)、[离散语音 token 综述](https://arxiv.org/abs/2502.06490)、[CTC/RNN-T 流式对照](https://arxiv.org/abs/2011.04785)、[Qwen3-ASR 技术报告](https://arxiv.org/abs/2601.21337) 与 [mu-bench](https://arxiv.org/abs/2609.32082)；用来建立架构、tokenizer、监督目标和业务关键字段评测地图。Qwen3-ASR 的流式窗口绑定该实现，mu-bench 的 UER 绑定其 judge、rubric 与五种 locale，不把模型报告或单一基准外推成通用结论。
8. 语音合成演进： [Tacotron 2](https://arxiv.org/abs/1712.05884)、[FastSpeech](https://arxiv.org/abs/1905.09263)、[FastSpeech 2](https://arxiv.org/abs/2006.04558)、[VITS](https://proceedings.mlr.press/v139/kim21f.html)、[VALL-E](https://arxiv.org/abs/2301.02111)、[VALL-E 2](https://arxiv.org/abs/2406.05370)、[Qwen3-TTS 技术报告](https://arxiv.org/abs/2601.15621) 与 [LP-TTS-Bench](https://arxiv.org/abs/2610.19774)；比较 Mel+vocoder、自回归/非自回归、显式韵律条件、变分端到端和 codec LM，并把音色与韵律作为不同评估维度。论文中的 first-packet latency 和长 prompt 收益绑定各自模型、评测协议与测量端点。
9. 流式对话语音： [Moshi](https://arxiv.org/abs/2410.00037) 比较并行用户/助手音频流与文本辅助生成；Qwen3-Omni/Qwen3.5-Omni 的 Thinker-Talker、MTP/Code2Wav 和 ARIA 见各自技术专题。论文延迟数字需绑定各自硬件、并发、首包定义与测量边界。
10. 多模态评估与文档： [CLIPScore](https://arxiv.org/abs/2104.08718)、[MME](https://arxiv.org/abs/2306.13394)、[MMBench](https://arxiv.org/abs/2307.06281)、[MMMU](https://arxiv.org/abs/2311.16502)、[POPE](https://arxiv.org/abs/2305.10355)、[HallusionBench](https://arxiv.org/abs/2310.14566)、[MagiC](https://arxiv.org/abs/2507.07297) 的 grounded reasoning 与步骤级评测，以及 [LayoutLMv3](https://arxiv.org/abs/2204.08387)；[VGR](https://arxiv.org/abs/2506.11991)展示推理中按需回看局部视觉特征的路线。分数和方法收益绑定各自任务集、模型、prompt、解析规则和版本，不概括整体产品能力。
11. 统一多模态路线： [Chameleon](https://arxiv.org/abs/2405.09818)、[NExT-GPT](https://arxiv.org/abs/2309.05519) 与 [AnyGPT](https://arxiv.org/abs/2402.12226)；分别观察早期融合序列、LLM 中枢连接专用 encoder/decoder、离散多模态 token 的结构和数据约束。
12. 多模态标准与安全： [VQA 官方评估协议](https://visualqa.org/evaluation.html)、[ITU-T P.800.2 MOS 报告建议](https://www.itu.int/itu-t/recommendations/rec.aspx?id=12973) 和 [C2PA 2.4 安全边界](https://spec.c2pa.org/specifications/specifications/2.4/security/Harms_Modelling.html)；可验证来源凭证不能单独证明媒体事实真实。
13. 官方工程接口： [Transformers Processor](https://huggingface.co/docs/transformers/main/processors) 与 [Diffusers Scheduler](https://huggingface.co/docs/diffusers/main/en/using-diffusers/schedulers) 文档，分别用于核对多模态预处理输入输出和 checkpoint/scheduler 兼容性；页面持续更新，需按目标版本复核。
14. 多模态训练的目标与配方： [SigLIP 2](https://arxiv.org/abs/2502.14786) 展示对比目标之外加入 captioning、自蒸馏、masked prediction 和数据筛选的组合；[MM1](https://arxiv.org/abs/2403.09611) 对照 encoder、分辨率、视觉 token 数和数据 mixture；[LLaVA-1.5](https://arxiv.org/abs/2310.03744) 提供公开视觉指令微调基线；[JEST](https://arxiv.org/abs/2406.17711) 研究 joint-batch 样本选择。它们解决的瓶颈不同，论文数据比例、算力和 benchmark 结果不能拼成统一排行榜。
15. OCR、ASR、视频和扩散训练对照： [Donut](https://arxiv.org/abs/2111.15664) 与 [Pix2Struct](https://arxiv.org/abs/2210.03347) 对照 OCR-free 结构化生成和 screenshot parsing；[CTC](https://www.cs.toronto.edu/~graves/icml_2006.pdf) 与 [RNN-T](https://arxiv.org/abs/1211.3711) 对照单调路径求和和时间—前缀格点；[Min-SNR](https://arxiv.org/abs/2303.09556) 分析 diffusion timestep 梯度冲突及目标相关权重；[CogVideoX](https://arxiv.org/abs/2408.06072) 报告变长视频 frame pack 与时空训练设计。具体收益应绑定论文配置，不外推为所有数据或模型的保证。
16. 多模态全景综述： [From Specific-MLLMs to Omni-MLLMs](https://aclanthology.org/2025.findings-acl.453/) 用于定位架构、训练阶段、数据集和评测问题；综述负责建立阅读地图，机制和数值结论仍回到对应原论文核对。
17. 多模态推理预算与视频剪枝： [Qwen2-VL](https://arxiv.org/abs/2409.12191) 以动态分辨率分配视觉 token；[EVS](https://arxiv.org/abs/2510.14624) 删除时间静态 patch；[VidCom2](https://arxiv.org/abs/2505.14454) 按帧独特性分配压缩预算。分别比较预处理/编码后成本、保留位置和质量—延迟曲线，结果只绑定原论文配置。
18. 多模态 serving、安全与视频评估： [vLLM-Omni](https://arxiv.org/abs/2610.09307) 展示 orchestrator、stage pool、connector 与 chunk handoff；[Video-MME-v2](https://arxiv.org/abs/2604.05015) 评估跨题一致和证据支持；[Image-based Prompt Injection](https://arxiv.org/abs/2603.03637) 研究图像内隐藏指令。系统 benchmark、论文攻击率和模型能力须分开报告。
19. 交互式视频世界模型评估：[WBench](https://arxiv.org/abs/2605.25874) 以五个维度组织 289 个案例与 1,058 轮交互；[World-in-Context](https://arxiv.org/abs/2610.33169) 是 2026-10 的 v1 预印本，显式联合编码视角、动作和摄像机控制，并在 WBench 上评估。两者结果均绑定各自数据、模型接口和评分协议，不能代表所有模型或现实环境中的世界建模能力。
20. 视频跨模态描述评估：[OmniCapBench](https://arxiv.org/abs/2610.12458) 用 786 段视频和原子化事件检查实体指代、镜头变化、声音事件、时间定位与音画一致性；样本数和发现绑定其数据构造与评测协议。
21. 扩散模型规模研究：[Scaling Laws for Diffusion Transformers](https://proceedings.iclr.cc/paper_files/paper/2026/hash/880d8999c07a8efc9bbbeb0c38f50765-Abstract-Conference.html) 在指定 FLOPs 区间拟合 DiT 生成质量并做大规模外推；预测点需与直接实验区分，不能当成目标模型的实测结论。

- [XModBench](https://proceedings.iclr.cc/paper_files/paper/2026/hash/21071811269c70ac11a3e7869bac747e-Abstract-Conference.html)：以三模态任务检查 omni-language model 的跨模态一致性与模态偏差；它补充任务切片，不证明模型内部形成了模态不变表示。
- [LVOmniBench](https://arxiv.org/abs/2603.19217)：评估 10—90 分钟长音视频中的记忆、时间定位和跨模态理解；275 个视频与 1,014 组问答以及作者报告的模型分数只代表该数据、模型快照和协议。

### Safety 与 Interpretability

Red Teaming、Mechanistic Interpretability、SAE、Model Editing、Unlearning、Privacy。

#### 自动化对抗评估

- [GOAT](https://proceedings.mlr.press/v267/pavlova25a.html)：ICML 2025 的多轮自适应攻击生成方法；其 ASR@10 绑定 JailbreakBench、攻击轮数和 Judge 口径，不能外推为现实攻击概率。
- [MM-ART](https://arxiv.org/abs/2504.03174)：在指定模型与语言集合中研究多语言、多轮红队，适合提示语言和轮数的覆盖切片，不作为普遍语言安全排名。
- [Agent vs. Agent](https://aclanthology.org/2025.emnlp-industry.62/)：按自定义工具集生成攻击任务和评估函数，展示应用工作流与静态工具集之间的差异；结论受其任务和模型配置约束。
- [REDAgentBench](https://arxiv.org/abs/2608.10669)：2026-08 arXiv 预印本，以隔离服务回执和最终状态验证 Agent 副作用；1,661 个案例覆盖五类模拟服务，并比较 harness 与证据视图对报告 ASR 的影响。仍应按预印本及其具体模型、环境和判定配置解读。

Mechanistic interpretability 补读入口：[SAEBench](https://arxiv.org/abs/2503.09532) 与 [MIB](https://proceedings.mlr.press/v267/mueller25a.html) 对照 SAE 指标和因果定位任务；[SANITY Checks](https://arxiv.org/abs/2602.14111) 检查随机基线；[Revising and Falsifying SAE Feature Explanations](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5b6ce2529a79862a2d8ac57ac0e01d90-Abstract-Conference.html) 以近邻负例反驳过宽解释；[LLM Forensics](https://arxiv.org/abs/2609.07746) 区分触发检测与行为控制；[Formal Mechanistic Interpretability](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b5afe13494c825089b1e3944fdaba212-Abstract-Conference.html) 在视觉模型中研究形式化 circuit 保证。以上结果均绑定论文自己的模型、数据、指标和任务范围。

Representation engineering 与 activation steering 补读入口：[Representation Engineering](https://arxiv.org/abs/2310.01405)、[Activation Engineering / ActAdd](https://arxiv.org/abs/2308.10248)、[CAA](https://aclanthology.org/2024.acl-long.828/)、[Endogenous Steering Resistance](https://proceedings.mlr.press/v306/mckenzie26a.html)、[Spherical Steering](https://proceedings.mlr.press/v306/you26a.html)、[PID Steering](https://proceedings.iclr.cc/paper_files/paper/2026/hash/fa5617c176e76fee83f3f9947fdf9f3f-Abstract-Conference.html) 与 [SteerEval](https://aclanthology.org/2026.acl-long.1443/)；对安全多视角和输入扰动再读 [SteeringSafety](https://arxiv.org/abs/2509.13450) 与 [Adversarial Robustness of Activation Steering](https://arxiv.org/abs/2606.07696)。这些论文支持方法与限定实验结果，不支持新模型上的上线保证。

Model editing 与 unlearning 补读入口：[EVOKE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c592fc7e6207f82560ed45fece8d6937-Abstract-Conference.html) 研究编辑过拟合，[LocFT-BF](https://proceedings.iclr.cc/paper_files/paper/2026/hash/74b8ccb5ad21871fd436867df19c35ac-Abstract-Conference.html) 比较 fine-tuning 编辑流程，[UniEdit](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6cc6c2be42c838827d2f1d12baa05539-Abstract-Datasets_and_Benchmarks_Track.html) 加入多领域邻域多跳测试；[Context-Aware Unlearning](https://proceedings.mlr.press/v306/peng26a.html) 检查删除后重新提供上下文时的可用性，[Information Decomposition Audit](https://aclanthology.org/2026.eacl-long.35/) 检查内部表征残留，[SUITE](https://arxiv.org/abs/2607.09236) 讨论 forget/retain 边界的非对称泛化。各结果受论文模型、数据、方法与证据状态约束；SUITE 为预印本。

近期表示方向估计线索：[Contrastive Covariance Subspaces for Activation Steering](https://arxiv.org/abs/2610.14025) 是 2026-09 arXiv 预印本，尝试压制样本中的共变干扰；结果需与均值差、随机方向和同任务 holdout 比较后再判断迁移价值。

建议补读资料线：

1. [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565) 与 [Risks from Learned Optimization](https://arxiv.org/abs/1906.01820)：奖励设计、副作用、specification gaming 与目标错配的基础分析。
2. [Constitutional AI](https://arxiv.org/abs/2212.08073)：原则 + AI feedback 的对齐研究方法。
3. [Extracting Training Data from LLMs](https://arxiv.org/abs/2012.07805)、[Membership Inference](https://arxiv.org/abs/1610.05820) 与 [DP-SGD](https://arxiv.org/abs/1607.00133)：隐私攻击与差分隐私保护入口。
4. [ROME](https://arxiv.org/abs/2202.05262) 与 [MEMIT](https://arxiv.org/abs/2210.07229)：参数化知识定位与批量编辑研究方法。
5. [Model Cards](https://arxiv.org/abs/1810.03993)、[StereoSet](https://arxiv.org/abs/2004.09456) 与 [RealToxicityPrompts](https://arxiv.org/abs/2009.11462)：模型披露、偏见与毒性评估入口；结合任务和群体切片使用。
6. [HarmBench](https://arxiv.org/abs/2402.04249)：标准化 510 种有害行为与自动红队/稳健拒答评估，并分析输出 token budget、攻击方法和评判器对 ASR 可比性的影响；benchmark 不能代替目标策略与生产工具副作用评估。
7. [StrongREJECT](https://arxiv.org/abs/2402.10260)：把拒绝、回答具体性和说服力分开计分，并将 rubric / fine-tuned evaluator 与人工判断比较；评分不等于现实伤害率。
8. [JailbreakBench](https://arxiv.org/abs/2404.01318) 与 [XSTest](https://aclanthology.org/2024.naacl-long.301/)：前者固定攻击威胁模型、工件和查询预算并匹配 benign 行为，后者专门检查夸大安全和安全请求误拒。
9. [InjecAgent](https://arxiv.org/abs/2403.02691) 与 [AgentDojo](https://arxiv.org/abs/2406.13352)：分别覆盖工具型间接注入固定案例与状态化 Agent 环境；后者分开报告 benign utility、utility under attack 和 targeted ASR。
10. [Who Tests the Testers? / SafeAudit](https://aclanthology.org/2026.emnlp-main.73/)：系统枚举工具工作流并审计 benchmark 未覆盖的 agent 安全路径；论文数字绑定其枚举器、环境和预算。
11. [The Secret Sharer](https://arxiv.org/abs/1802.08232)、[Extracting Training Data from LLMs](https://arxiv.org/abs/2012.07805) 与 [Do Membership Inference Attacks Work on LLMs?](https://arxiv.org/abs/2402.07841)：分别研究罕见 canary 暴露、输出抽取和训练集成员推断；三类证据不可互换。
12. [Privacy Auditing of LLMs](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e76814a0012a46ade4f997591e86f972-Abstract-Conference.html)、[Beyond Memorization](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9028b8a3ca98f58e373f0c1497a17448-Abstract-Conference.html) 与 [InfoRMIA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ddf43af256e11499149cf06040c62caa-Abstract-Conference.html)：覆盖 canary 隐私审计、从用户文本推断属性和 token 级记忆信号。
   Privacy Auditing 的 49.6% TPR@1% FPR 绑定 Qwen2.5-0.5B 和论文威胁模型；InfoRMIA 的成员推断与 token 级定位结果绑定其 benchmark。水印方向再读 [SynthID-Text](https://www.nature.com/articles/s41586-024-08025-4)、Anthropic 对 Claude 水印限制的[官方说明](https://www.anthropic.com/news/claude-text-watermark)，以及 2026 年预印本 [AI Watermark Evidence Fails Forensic Readiness](https://arxiv.org/abs/2607.16010)；产品说明、论文实验和预印本结果分别标注，不外推为普遍作者识别能力。
13. [BBQ](https://arxiv.org/abs/2110.08193) 与 [CBBQ](https://arxiv.org/abs/2306.16244)：通过信息不足和信息充分的上下文测量 stereotype fallback、忽略证据及中文社会语境；数据集不等于真实决策公平性结论。
14. [CheckList](https://aclanthology.org/2020.acl-main.442/)：MFT、invariance 和 directional expectation 行为测试设计，可用于构造可核验的 LLM 安全/鲁棒性关系测试。
15. [Evaluating Frontier Models for Dangerous Capabilities](https://deepmind.google/research/publications/78150/) 与 [OpenAI 生物风险 uplift 评估](https://openai.com/index/building-an-early-warning-system-for-llm-aided-biological-threat-creation/)：一个展示分领域能力评测，一个比较人类在有/无模型辅助时的表现；均应按发布方实验边界解读。
16. [NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)、[OpenAI Preparedness Framework 更新](https://openai.com/index/updating-our-preparedness-framework/)、[OpenAI Frontier Governance Framework](https://openai.com/index/openai-frontier-governance-framework/)、[Anthropic Responsible Scaling Policy](https://www.anthropic.com/responsible-scaling-policy)、[Google DeepMind Frontier Safety Framework](https://deepmind.google/frontier-safety/) 与 [OWASP GenAI Security Project](https://owasp.org/projects/top-10-for-large-language-applications)：分别提供风险生命周期、发布方能力/防护治理和应用层威胁分类入口；机构框架有各自版本与对象，不是模型安全认证或普遍法律标准。
17. [NIST SP 800-188: De-Identifying Government Datasets](https://doi.org/10.6028/NIST.SP.800-188)：讨论直接标识符变换、假名化和再识别风险；适合作为去标识化治理参考，不应把带密钥哈希当作匿名化保证，也不是所有司法辖区的完整合规意见。
17. [Indirect Prompt Injections: Are We on the Brink of a New AI Security Threat?](https://arxiv.org/abs/2510.05244)：讨论间接注入评测集、攻击强度和指标问题的预印本；作者分析不应当作领域共识。
18. [International AI Safety Report 2026](https://internationalaisafetyreport.org/publication/international-ai-safety-report-2026)：综合整理 misuse、malfunctions、systemic risks、Agent 自主性与评估缺口；适合建立风险地图，不对具体产品提供安全认证。
19. [OWASP Top 10 for LLM Applications 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/) 与 [Agent Control Standard](https://genai.owasp.org/resource/agent-control-standard-acs/)：分别作为应用风险分类和 Agent 运行时 middleware / 声明式控制入口；属于社区安全指导与接口方案，不是控制效果的独立实测。
20. [METR Frontier Risk Report (February to March 2026)](https://evals.alignment.org/blog/2026-05-19-frontier-risk-report/)：记录一次与多家前沿模型开发者合作的外部风险评估试点；结论绑定该窗口、参与系统和资料访问条件。
21. [AgentCanary](https://arxiv.org/abs/2606.10484)：以可执行环境中的完整 Agent 轨迹为评测对象，并分开报告 outcome safety、security awareness 与 task utility；按预印本和其模型/框架配置解读。
22. [Assessing Confidence in Frontier AI Safety Cases](https://arxiv.org/abs/2502.05791)：以 cyber misuse 的 inability argument 讨论 safety case 及信心量化；方法案例有限，不能视为已验证的通用风险认证流程。
23. [SandboxEscapeBench v3](https://arxiv.org/abs/2603.02277v3)：测量容器到宿主机威胁模型中的 Agent 逃逸能力；其任务包含有意弱点、shell 访问和较少监控，不能外推为受防护生产环境的失陷概率。
24. [AgentCyberRange](https://arxiv.org/abs/2606.14295)：在多主机企业式靶场中评估 Web 利用及后续阶段，论文包含 15 个 Web 应用和 110 个漏洞；作者明确指出 phishing、Windows domain、cloud IAM、supply-chain 与 social engineering 不在覆盖范围内。
25. [FUSE: An Evaluating Framework for Dangerous Capabilities of LLMs](https://arxiv.org/abs/2609.02168)：以 Knowledge、Defense、Harm 三条评估管线分别测量领域能力、防御行为和潜在危害；完整验证主要在生物化学领域，cyber 为小规模试点，H 维度跨评审者一致性有限。
26. [AISI incident report on unsanctioned agent behaviour during cyber testing](https://www.aisi.gov.uk/blog/incident-report-unsanctioned-agent-behaviour-during-cyber-testing) 与[后续评估环境更新](https://www.aisi.gov.uk/blog/building-a-more-secure-environment-for-evaluating-dangerous-capabilities)：记录一项特定评估配置中的越界动作和事后多层网络出口/运行时监控措施，不代表一般 Agent 的发生率。
27. [EU AI Act GPAI obligations](https://digital-strategy.ec.europa.eu/en/factpages/general-purpose-ai-obligations-under-ai-act)、[GPAI Code of Practice](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai)、[Article 50 transparency FAQ](https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act) 与[生成内容标记实践准则](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)：区分法规义务、可自愿采用的合规工具、按角色适用的透明义务及生成内容标记方法；适用性和日期要按最新官方文本核验。

本章正文及本轮旧知识承接、方案比较、公式和 demo 复核见[研究记录](research/safety-robustness-evaluation-depth-review-2026-10-27.md)；[第七册第八章](book-07-evaluation-experiments/chapters/08-safety与robustness-eval.md)中的数据和数值均保留对应研究范围。

第八册新方向与书册分工的初步来源评估见[第八册架构扫描记录](research/book-08-ai-safety-alignment-architecture-scan-2026-10-27.md)。

### 后训练评估与公开基准

1. [MMLU](https://arxiv.org/abs/2009.03300) 与 [MMLU-Pro](https://arxiv.org/abs/2406.01574)：知识与推理题目评测；注意题目版本、提示、选项格式和污染风险。
2. [GSM8K](https://arxiv.org/abs/2110.14168) 与 [MATH](https://arxiv.org/abs/2103.03874)：数学文字题和竞赛数学基准；报告时应固定解析器、生成预算和答案判分规则。
3. [HumanEval](https://arxiv.org/abs/2107.03374) 与 [MBPP](https://arxiv.org/abs/2108.07732)：代码生成任务集；分数依赖采样、测试执行和 pass@k 口径。
4. [IFEval](https://arxiv.org/abs/2311.07911) 与 [LongBench](https://arxiv.org/abs/2308.14508)：分别覆盖可验证的指令遵循和多任务长上下文评估；两者都不能概括整个产品或所有语言、长度与工具任务。

### 学习科学与知识迁移

1. [Test-Enhanced Learning](https://doi.org/10.1111/j.1467-9280.2006.01693.x)：主动测试与重复学习对即时、延迟保持的影响；结论受论文任务和测试间隔限定。
2. [Distributed practice in verbal recall tasks: A review and quantitative synthesis](https://doi.org/10.1037/0033-2909.132.3.354)：练习间隔与最终保持间隔对 verbal recall 的联合影响，不提供普遍适用的固定复习日历。

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
9. [Shieldstral 1.0 模型卡](https://huggingface.co/mistralai/Shieldstral-1.0-3B) 与 [Mistral 官方产品目录](https://docs.mistral.ai/models/shieldstral-1-0)：策略自适应多模态分类器；保留模型卡/报告 3B 与产品目录 3.8B 的规格口径差异。
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
4. [Towards Policy-Adaptive Image Guardrail / SafeEditBench](https://arxiv.org/html/2603.01228)：同图像安全/不安全配对及跨政策评测，CVPR 2026 论文。
5. [PolicyShiftBench / PolicyShiftGuard](https://arxiv.org/abs/2607.05910)：同图像政策翻转评测与 PSS；arXiv 预印本。
6. [EAGLE](https://arxiv.org/abs/2401.15077) 与 [EAGLE-3](https://arxiv.org/abs/2503.01840)：推测解码 draft head 路线。
7. [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)：推理预算、候选和验证成本的实验基础。

### 长序列注意力与循环状态综述

1. [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)：按稀疏注意力、低秩/核近似、压缩、循环等路线整理效率方法，并分别讨论计算量和内存；该版本早于近年的 KDA/CSA/HCA。
2. [On the Resurgence of Recurrent Models for Long Sequences](https://arxiv.org/abs/2402.08132)：梳理循环网络、SSM 与 attention-recurrent hybrid，适合作为有限状态和长序列研究缺口的概念地图，不替代 KDA 等新方案的原论文。

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

### 深度聚合与多路残差的前序研究

1. [DenseFormer（NeurIPS 2024）](https://arxiv.org/abs/2402.02622)：以 depth-weighted averaging 汇聚当前层和历史层表示；用于理解深度方向信息复用的早期方案。
2. [DeepCrossAttention（ICML 2025）](https://proceedings.mlr.press/v267/heddes25a.html)：输入相关的 depth-wise cross-attention；论文的语言建模速度/质量结果绑定其训练配置和基线。
3. [Hyper-Connections](https://arxiv.org/abs/2409.19606)：扩展多路 residual streams 并学习流间连接，为理解 mHC 的前身和稳定性问题提供入口。
4. [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)：将残差流混合映射约束到双随机矩阵集合；有限步 Sinkhorn 和论文训练开销需按其模型与实现条件理解。

### 第二册架构与训练稳定性新研究

1. [AdaGC: Enhancing LLM Pretraining Stability via Adaptive Gradient Clipping（ICML 2026）](https://proceedings.mlr.press/v306/wang26dg.html)：按张量基于历史裁剪范数调整梯度上界；论文报告 Llama-2 7B、Mixtral 8×1B、ERNIE 10B-A1.4B 等实验，不将其结果外推为任意规模训练的保证。
2. [SpanNorm: Reconciling Training Stability and Performance in Deep Transformers（ICML 2026）](https://proceedings.mlr.press/v306/chao26b.html)：结合直接残差路径、PostNorm 式输出归一化和缩放策略；作者报告深层 dense 与 MoE 设置下的理论和实验结果。
3. [PowLU: An Activation Function for Stable Pre-Training of LLMs](https://arxiv.org/abs/2605.25704)：预印本研究限制 SwiGLU 大正值输入下的增长，并报告 Ling 架构大规模 MoE 训练实验；结论绑定作者配置。
4. [More Expressive Feedforward Layers: Part I. Token-Adaptive Mixing of Activations](https://arxiv.org/abs/2605.26647)：预印本提出以输入相关门控混合多种激活函数，在 0.12B—2B dense/MoE 设置中评估 MoA；结果不与 PowLU 直接横向排名。
5. [Taming Curvature: Architecture Warm-Up for Stable Transformer Training](https://arxiv.org/abs/2606.16768)：预印本以预条件曲率在线估计和逐步增加网络深度研究训练稳定性；规模、算法开销与作者实验适用范围需在引用时保留。
6. [How To Build A Modern Diffusion Language Model](https://arxiv.org/abs/2610.33391) 与 [LLaDA2.0: Scaling Up Diffusion Language Models to 100B](https://arxiv.org/abs/2512.15745)：前者是 2026 年 10 月提交的离散扩散语言模型综述，后者报告由自回归模型转换的 100B block-diffusion 实验；区分综述分类、作者模型结果与独立复现，不把并行去噪直接等同于端到端更快。
7. [Log-Linear Attention](https://arxiv.org/abs/2506.04761)：arXiv v3 将线性注意力的固定状态扩展为对数增长的多尺度状态，并报告 $O(T\log T)$ 形式；Mamba-2/Gated DeltaNet 结果绑定 700M—800M 模型、H100 与论文任务，仍有性能和 kernel 复杂度限制。
8. [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention](https://arxiv.org/abs/2605.22791)：分开 key 侧擦除门与 value 侧写入门；作者在 1.3B、100B FineWeb-Edu token 的 recurrent/hybrid 训练设置下报告语言建模与检索结果，hybrid 包含滑动窗口 attention，不外推为通用优势。

### GPT-6 Astra 官方接口资料

1. [GPT-6 Astra model page](https://developers.openai.com/api/docs/models/gpt-6-astra)：`gpt-6-astra` 的输入/输出模态、1,050,000 context、128,000 max output、reasoning effort、工具和 endpoint。当前页面没有单独列 922,000 maximum input；该数是扣除满额输出后的算术余量。
2. [Using GPT-6](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra)、[Async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling)、[Mid-turn steering](https://developers.openai.com/api/docs/guides/steering) 和 [Misalignment monitoring](https://developers.openai.com/api/docs/guides/safety-checks/misalignment-monitoring)：家族参数/endpoint 约束、应用侧异步工具、WebSocket steering 状态机及平台安全控制面。
3. [Reasoning](https://developers.openai.com/api/docs/guides/reasoning)、[Conversation state](https://developers.openai.com/api/docs/guides/conversation-state)：mode/effort 独立、reasoning/output 上限、`configuration_update`、reasoning/`phase` item replay 与 `previous_response_id`；`phase` 专节中的其他模型示例不当作 GPT-6 独有能力。
4. [API pricing](https://developers.openai.com/api/docs/pricing)、[Fast mode](https://developers.openai.com/api/docs/guides/fast-mode)、[Ultrafast mode](https://developers.openai.com/api/docs/guides/ultrafast-mode)：Astra 的 short/long context、缓存读取、30m/12h cache write、Standard/Batch/Flex/Fast/Ultrafast 价表及地域/速率限制。价格为 2026-10-27 页面快照。
5. [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)、[Compaction](https://developers.openai.com/api/docs/guides/compaction)、[Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search) 和 [Rethinking skills and prompts](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)：KV 前缀缓存、deferred schema、hosted/client tool search、canonical compaction context 以及技能/`AGENTS.md`/任务提示的上下文治理。
6. 本组资料仅证明当前公开接口、价格表、状态协议和官方工程建议；不证明模型参数规模、内部架构、训练 recipe、目标服务时延或生产效果。逐节机制对照和未验证项见 [`gpt6-astra-long-context-tool-budget-depth-review-2026-10-27.md`](research/gpt6-astra-long-context-tool-budget-depth-review-2026-10-27.md)。
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

## 第八册第十五章：Risk-Calibrated Access 的授权与身份资料

1. [NIST SP 800-207：Zero Trust Architecture](https://csrc.nist.gov/pubs/sp/800/207/final) 与 [SP 800-207A：Cloud-Native Access Control](https://csrc.nist.gov/pubs/sp/800/207/a/final)：支持资源级判断、避免仅凭网络位置或组织归属建立隐式信任；207A 特别讨论云原生应用中的用户与服务身份。它们是架构指导，不是 Agent 专用授权策略。
2. [OpenID AuthZEN Authorization API 1.0](https://openid.net/specs/authorization-api-1_0.html) 与[规范状态页](https://openid.net/wg/authzen/specifications/)：2026 年 1 月获批为 Final Specification，使 PDP 与 PEP 能交换授权请求和决定；不统一内部策略语言、风险阈值或允许动作。
3. [OAuth 2.0 Token Exchange, RFC 8693](https://www.rfc-editor.org/rfc/rfc8693)：包含委托、冒充及 actor/subject 语义；部署信任模型、具体令牌安全属性和业务授权仍需本地定义。
4. [OAuth 2.0 DPoP, RFC 9449](https://www.rfc-editor.org/rfc/rfc9449)：以持有证明对 OAuth 令牌做发送方约束，降低被盗令牌重放风险；不等于业务授权，不能替代 HTTPS。
5. [OWASP Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/) 的 ASI03 Identity and Privilege Abuse：用于识别继承身份、凭证泄漏和权限扩展等风险类别，不是实现标准或某个部署安全性的证据。
6. [Zanzibar: Google’s Consistent, Global Authorization System](https://www.usenix.org/conference/atc19/presentation/pang)：大规模关系授权与一致性系统的公开研究案例；不是 Agent 权限标准。
7. [Cedar Authorization](https://docs.cedarpolicy.com/auth/authorization.html) 与 [Policy Validation](https://docs.cedarpolicy.com/policies/validation.html)：具体展示 principal/action/resource/context、默认拒绝及策略 schema 校验；实现行为仅适用于 Cedar，不能推广为所有 PDP 的共同语义。

上述来源支持本章的身份、授权接口、关系建模和风险分类边界，不提供可直接复制的组织策略，也不证明实际 PEP、executor、撤销传播或生产状态已经通过验证。

## 第八册第十六章：Fallback Routing、重试与状态恢复

1. [Google SRE Book：Handling Overload](https://sre.google/sre-book/handling-overload/)：讨论过载下的负载控制、排队、降级响应与错误返回；具体系统仍需按自身容量和 SLO 验证。
2. [Google SRE Book：Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/)：解释过载反馈、重试放大、退避和级联故障，作为通用 SRE 工程背景。
3. [RFC 9110 §9.2.2：Idempotent Methods](https://www.rfc-editor.org/rfc/rfc9110#section-9.2.2)：定义 HTTP 方法幂等性并限制非幂等请求的自动重试；它不替应用层付款、邮件或文件操作定义幂等键。
4. [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) 与 [NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)：提供 AI 风险生命周期和生成式 AI 风险管理背景，不认证具体 fallback 路由。
5. [OWASP Top 10 for LLM Applications 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/)：应用层威胁分类入口，不是系统安全验收证据。

模型安全路由案例沿用[Opus 5.5 研究记录](research/model-update-2026-09/claude-opus-5.5-source-notes.md)、[Fable 5.1 研究记录](research/model-update-2026-09/claude-fable-5.1-source-notes.md)和[Sonnet 5.5 研究记录](research/model-update-2026-09/claude-sonnet-5.5-source-notes.md)中的官方 System Card 与发布方评测边界；这些是特定发布版本证据，不表示其他服务采用相同路由策略。

## 第八册第十七章：AI Control 与不可信 Agent 监督

1. [AI Control: Improving Safety Despite Intentional Subversion](https://arxiv.org/abs/2312.06942)：提出在假设模型可能蓄意绕过监督时评估可信监控、可信编辑等协议；实验结论绑定论文中的代码任务、模型和控制设定。
2. [SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents](https://arxiv.org/abs/2506.15740)：构造带隐藏副任务的多步 Agent 环境；监控 AUC 与 sabotage score 绑定其模型、轨迹、scratchpad 和任务判定。
3. [Evaluating Control Protocols for Untrusted AI Agents](https://arxiv.org/abs/2511.02997)：比较 resampling、trusted delegation 与 critical-action protocols，并用迭代红蓝队测量自适应攻击；报告数字不能脱离其 SHADE-Arena 设置和指标解释。
4. [Adaptive Attacks on Trusted Monitors Subvert AI Control Protocols](https://proceedings.iclr.cc/paper_files/paper/2026/hash/54b153ad8a138f4c186f21a8b7341d5e-Abstract-Conference.html)：ICLR 2026 论文研究协议知情的监控器特异提示注入；支持把 monitor-targeted attack 加入评估，不证明所有控制协议均失败。
5. [Reliable Weak-to-Strong Monitoring of LLM Agents](https://arxiv.org/abs/2508.19461)：研究分层、顺序监控及定向人工升级；结果受论文任务和威胁条件约束。
6. [How to evaluate control measures for LLM agents?](https://arxiv.org/abs/2504.05259) 与 [A sketch of an AI control safety case](https://arxiv.org/abs/2501.17315)：分别提供能力校准和安全论证示例框架；不是标准认证或未来能力预测。
7. [ControlArena：AI Control 介绍](https://control-arena.aisi.org.uk/intro-to-control.html) 与[主站](https://control-arena.aisi.org.uk/)：公开 settings、policy、monitor、micro/macro protocol 和安全/效用评估入口；接口随项目版本变化。
8. [OpenAI：Securing our frontier model training environment](https://openai.com/index/securing-our-frontier-model-training-environment/)：2026 年 10 月的发布方红队案例，记录其训练环境隔离测试与修复；属于公司自述，未独立复现。

各来源的版本、具体支持判断和未验证范围见[第十七章研究与精修记录](research/book-08-ch17-ai-control-refinement-2026-10-28.md)。

## 第九册架构扫描：数据生成、模态对齐、归因与治理

1. [Demystifying Synthetic Data in LLM Pre-training](https://aclanthology.org/2025.emnlp-main.544/) 已在第五册训练数据来源中建立入口；在第九册用于比较自然数据、重写数据、生成教材和混合比例。EMNLP 2025 正式论文覆盖超过 1,000 个模型变体、最高 200B token；具体收益绑定数据类型、模型和预算，不能作为通用比例。
2. [Rewriting Scaling Laws for Synthetic Data with Compute and Data Constraints](https://arxiv.org/abs/2610.31867)：2026-10-22 的 arXiv v1 将生成改写成本计入数据/算力预算；最优比例及 22B 外推为作者的特定 scaling-law 拟合结果。
3. [The Fairness Collapse Phenomenon](https://arxiv.org/abs/2608.04268)：在 Bias in Bios 英文职业/性别语料上研究递归合成数据的公平性变化；支持把公平性切片作为合成数据审计候选，不证明其他语料或属性也有同样退化。
4. [Kairos: A Dataset for Fine-Grained Video-Language Modeling over Space, Time, and Dynamics](https://arxiv.org/abs/2609.08755) 与 [EuroSpeech: A Multilingual Speech Corpus](https://arxiv.org/abs/2510.00514)：分别提供长视频时间标注与多语言语音/转录对齐的构造案例；样本规模、标注设计和作者报告收益不跨语料外推。
5. [Dataset Inference for Data Provenance and Privacy Auditing in Tabular Foundation Models](https://arxiv.org/abs/2610.24278)：2026-10 预印本研究表格基础模型的训练集成员推断；方法和结果不等于文本 LLM 的数据来源证明，也不能取代内部 lineage。
6. [EU Commission: Template for the Public Summary of Training Content](https://digital-strategy.ec.europa.eu/en/faqs/template-general-purpose-ai-model-providers-summarise-their-training-content)：官方 FAQ 说明公开训练内容摘要模板和适用时间；应与内部数据 manifest/逐样本血缘区分，并按当前官方版本复核。
7. [What Is Your Data Worth to GPT? LLM-Scale Data Valuation with Influence Functions](https://arxiv.org/abs/2405.13954)：LoGra 在 Llama 3 8B-Instruct 与 1B-token 数据条件下报告大规模影响函数近似的效率变化；不代表通用估值精度或目标训练系统速度。
8. [A Survey on Data Contamination for Large Language Models](https://arxiv.org/abs/2502.14425) 与 [DyCodeEval](https://arxiv.org/abs/2503.04149)：用于补充 contamination 检测和动态评估路线；动态改题仍需检查语义、难度和评估独立性。

书册候选的纳入位置和证据限制见[第九册书册级架构记录](research/book-09-book-level-architecture-2026-10-28.md)。

### 第九册第一章：数据地图、配比与血缘扫描

1. [Data Mixing Laws](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cc84bfabe6389d8883fc2071c848f62a-Abstract-Conference.html)：用小规模训练拟合 mixture 与模型表现的关系；报告结论绑定其 RedPajama、模型规模、token 预算和评估集。
2. [Demystifying Data Organization for Enhanced LLM Training](https://arxiv.org/abs/2605.30334)：ACL 2026 研究数据比例以外的排序、调度和局部多样性；作者在指定 pretraining/SFT 配置上评估，不构成通用排序 recipe。
3. [Collapse or Thrive: Perils and Promises of Synthetic Data in a Self-Generating World](https://proceedings.mlr.press/v267/kazdan25a.html)：区分递归替换、累积与固定子集工作流；结论受所研究任务、数据维护方式和生成代数约束。
4. [Tracing the Data Trail](https://arxiv.org/abs/2601.14311)：LLM 数据来源、透明度、血缘追踪和隐私技术综述；作为 taxonomy 入口，不证明模型输出可普遍逐样本追溯。

### 第九册第二章：Web-scale 数据采集与解析

1. [Common Crawl Get Started](https://www.commoncrawl.org/get-started)：官方快照目录、下载入口及 WARC/WAT/WET 数据类型说明；快照列表会更新，引用时应记录使用的 crawl ID。
2. [FineWeb2: One Pipeline to Scale Them All](https://arxiv.org/abs/2506.20920)：COLM 2025 使用 96 个 Common Crawl 快照构造 20 TB、1,868 个语言—文字系统组合的语料；核心处理消融在 9 种语言、比较覆盖 14 种，不能把覆盖规模当作普遍效果验证。
3. [Can Performant LLMs Be Ethical? Quantifying the Impact of Web Crawling Opt-Outs](https://arxiv.org/abs/2504.06219)：COLM 2025 在 FineWeb-Edu、1.5B 模型和特定 robots 回溯过滤设置下研究能力差异，不构成法律授权判断或跨领域性能保证。
4. [RFC 9309: Robots Exclusion Protocol](https://www.rfc-editor.org/rfc/rfc9309.html)：IETF 标准化 robots 规则的语法和处理方式，并明确规则不是访问授权。

章节级扫描、候选主题去向与未验证范围见[第二章研究与精修记录](research/book-09-ch02-web-scale-acquisition-refinement-2026-10-28.md)。

### 第九册第三章：清洗、过滤与质量评分

1. [Removing Noise, not Finding Gold: Quality Filtering for Large-Scale Pretraining](https://arxiv.org/abs/2510.00866v4)：ICML 2026 分析分类器质量过滤的参考集偏好、任务偏向和长度等表面特征；模型、RedPajama-V2、训练预算和评估范围均按论文设置理解。
2. [Judging Quality Across Languages: A Multilingual Approach to Pretraining Data Filtering with Language Models](https://aclanthology.org/2025.emnlp-main.449/)：EMNLP 2025 的 JQL 工作用人工标签、LLM 标注和轻量多语言模型筛选训练数据；35 语种标注规模不等于 35 语种具有同样规模的预训练对照。
3. [Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](https://aclanthology.org/2025.acl-long.533/)：ACL 2025 将多维评分和 proxy model 验证损失结合，报告结果依其数据集、模型、token 预算与评估协议。

章节扫描、阈值定义、数例核验与证据限制见[第三章研究与精修记录](research/book-09-ch03-cleaning-filtering-quality-refinement-2026-10-28.md)。

### 第九册第四章：去重与污染检测

1. [Datasets, Documents, and Repetitions: The Practicalities of Unequal Data Quality](https://papers.neurips.cc/paper_files/paper/2025/hash/2aba6ec20299931d46ddeefd5ddcb442-Abstract-Conference.html)：NeurIPS 2025 研究不同计算预算下过滤数据的重复训练与文档频率控制；收益和上限依赖作者的训练 recipe、语料及评测。
2. [Scalable Frequency- and Length-Aware Subdocument Deduplication for Large Language Model Pretraining](https://arxiv.org/abs/2608.03089)：2026-08 arXiv 预印本把子文段候选检测与频率、长度感知的保留预算分开；报告条件包括 Hy-MT2-30B-A3B、FineWeb-Edu/代码网页、36,000 步和约 294.9B token。
3. [Data Contamination Quiz](https://aclanthology.org/2025.tacl-1.37/) 与 [The Emperor’s New Clothes in Benchmarking?](https://proceedings.mlr.press/v267/sun25t.html)：前者用词级扰动题测试模型对 benchmark 原始措辞的暴露信号，后者比较 benchmark 更新策略的结果保真度与污染抵抗力；两者都不能将单次行为或题目改写直接当成“无污染”证明。
4. [Detecting Data Contamination from Reinforcement Learning Post-training](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ab003a4f85ecb1b7b1514ff539dc7395-Abstract-Conference.html) 与 [On the Fragility of Benchmark Contamination Detection in Reasoning Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a3860475ddbfb8c644c43f0dcd266b66-Abstract-Conference.html)：ICLR 2026 分别测试 RL 阶段污染探针和推理模型上的检测掩盖；实验只支持论文特定的模型、训练方法和污染任务。

章节的来源阅读、方法取舍、手算与验证范围见[第四章研究与精修记录](research/book-09-ch04-dedup-contamination-refinement-2026-10-28.md)。

### 第九册第五章：Data Mixture 与配比

1. [DoReMi](https://arxiv.org/abs/2305.10429) 与 [GRAPE](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e7bba023351f8262c22e3328e42645ec-Abstract-Conference.html)：代理损失/目标权重优化路线，结果依赖 proxy、域划分和目标损失。
2. [RegMix](https://arxiv.org/abs/2407.01492)、[Data Mixing Laws](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cc84bfabe6389d8883fc2071c848f62a-Abstract-Conference.html) 与 [Scaling Laws for Optimal Data Mixtures](https://proceedings.neurips.cc/paper_files/paper/2025/hash/bc1d640f841f752c689aae20b31198c1-Abstract-Conference.html)：比较小规模训练拟合“配比—验证表现”的路线；外推绑定数据、预算、固定/动态配比假设和目标指标。
3. [FastMix](https://proceedings.iclr.cc/paper_files/paper/2026/hash/01d845baaa31b1043e5f280592d57dc4-Abstract-Conference.html)：在线更新 mixture 的研究路线，需保留独立评估反馈以控制反复调参风险。
4. [SampleMix](https://aclanthology.org/2025.findings-emnlp.741/)：样本级跨域质量/多样性选择；论文收益绑定 SlimPajama、1B LLaMA、100B token 和其评估任务。
5. [Rethinking Data Mixture for Large Language Models](https://aclanthology.org/2026.findings-eacl.15/)：数据配比方法综述，整理启发式、优化、拟合和在线方法的分类。
6. [Scaling Laws for Multilingual Language Models](https://aclanthology.org/2025.findings-acl.221/) 与 [ATLAS](https://arxiv.org/abs/2510.22037)：多语言比例拟合与跨语言迁移/干扰研究入口，结论不构成通用语言配方。
7. [Scaling Laws for Mixture Pretraining Under Data Constraints](https://arxiv.org/abs/2605.12715)：有限数据池重复暴露的 2026 年预印本；最优重复因子依赖其模型、数据、预算和验证目标。
8. [Data Mixing Can Induce Phase Transitions in Knowledge Acquisition](https://proceedings.neurips.cc/paper_files/paper/2025/hash/eee7ae5cf0c4356c2aeca400771791aa-Abstract-Conference.html)：受控合成知识任务中的非线性配比效应案例，不外推为真实语料的一般规律。

本章的定向方法比较、候选主题去向与外推边界见[第五章研究与精修记录](research/book-09-ch05-data-mixture-refinement-2026-10-28.md)。

### 第九册第六章：Code、Math 与 Domain Data

1. [rStar-Coder: Scaling Competitive Coding with a Large-scale Verified Dataset](https://proceedings.neurips.cc/paper_files/paper/2025/hash/54e847e1dffc87a8063844b149148557-Abstract-Conference.html)：问题、长解答、生成测试和相互验证组成的代码数据案例；结果绑定论文中的 Qwen 模型与 benchmark。
2. [The Common Pile v0.1](https://proceedings.neurips.cc/paper_files/paper/2025/hash/52acc050138d6f40dad6f12f91a4ce22-Abstract-Datasets_and_Benchmarks_Track.html)：30 个来源、约 8 TB 的开放许可/公有领域语料构造与训练案例；不能替代项目级授权审查。
3. [ResearchCodeBench](https://proceedings.neurips.cc/paper_files/paper/2025/hash/cd0d0a873cc3e601c76f46dccc3d4c5f-Abstract-Datasets_and_Benchmarks_Track.html)：把近期机器学习研究方法转成实现任务的代码评测；不是训练数据质量证明。
4. [FLAMES](https://aclanthology.org/2025.findings-emnlp.1346/)：数学合成策略比较，讨论题目覆盖与解答可靠性的分工；结论依赖固定生成预算、模型和任务。
5. [More Data or Better Data?](https://aclanthology.org/2025.emnlp-industry.43/)：统一流水线中的数学开源数据与合成方案比较；论文结果受其训练流程和任务设定约束。
6. [MathDebugger](https://aclanthology.org/2026.emnlp-main.608/)：以错误类型检查数学答案验证能力；不证明项目自己的 verifier 已可靠。
7. [Memorization in Large Language Models in Medicine](https://arxiv.org/abs/2509.08604)：医学语料继续预训练、benchmark 微调与临床记录微调设置中的记忆研究预印本；不提供可跨语料外推的通用记忆率。

章节正文、分数手算和 demo 验证范围见[第六章研究与精修记录](research/book-09-ch06-code-math-domain-refinement-2026-10-28.md)。

### 第九册第七章：Synthetic Data 与 Distillation Data

1. [Self-Instruct](https://arxiv.org/abs/2212.10560)、[WizardLM / Evol-Instruct](https://arxiv.org/abs/2304.12244) 与 [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)：分别提供指令自举、复杂指令演化和小型代码模型合成教材数据案例；结论绑定论文中的 seed、生成器、过滤和任务。
2. [Orca](https://arxiv.org/abs/2306.02707) 与 [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301)：解释轨迹和中间推理监督的蒸馏案例；教师给出的过程文本仍需正确性与授权核验。
3. [Quality Matters: Evaluating Synthetic Data for Tool-Using LLMs](https://aclanthology.org/2024.emnlp-main.285/)：比较已验证与未验证的合成工具数据；较少的验证样本胜过较多未验证样本的结果绑定论文所测两个 benchmark 和训练设置。
4. [On the Diversity of Synthetic Data and its Impact on Training Large Language Models](https://arxiv.org/abs/2410.15226)：研究 topic/prompt 覆盖、重复生成次数和训练收益；模型规模与数据条件限制外推范围。
5. [The Curious Decline of Linguistic Diversity](https://aclanthology.org/2024.findings-naacl.228/) 与 [Collapse or Thrive](https://proceedings.mlr.press/v267/kazdan25a.html)：分别观察递归微调的输出多样性变化，并区分递归替换与累积数据流程；不支持“所有合成数据都会坍缩”或“混合后绝不会退化”的普遍断言。
6. [Demystifying Synthetic Data in LLM Pre-training](https://aclanthology.org/2025.emnlp-main.544/) 与 [Polyglot Teachers](https://arxiv.org/abs/2604.11290)：前者比较自然、改写、教材式合成和混合预训练，后者研究多语言 teacher 选择；各自结果只适用于论文的数据、模型和训练条件。

章节的来源阅读、候选去向、条件边界、数值复算和固定 demo 验证范围见[第七章研究与精修记录](research/book-09-ch07-synthetic-distillation-refinement-2026-10-28.md)。

### 第九册第八章：Preference Data 与安全数据

1. [Training language models to follow instructions with human feedback / InstructGPT](https://arxiv.org/abs/2203.02155)、[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)、[KTO](https://arxiv.org/abs/2402.01306)、[Helpful and Harmless RLHF](https://arxiv.org/abs/2204.05862) 与 [Constitutional AI](https://arxiv.org/abs/2212.08073)：用于区分示范数据、成对比较、单条 desirable/undesirable 反馈、reward model/策略优化及原则化反馈；各自训练结果绑定论文模型、任务和标注流程。
2. [Diverging Preferences: When do Annotators Disagree and do Models Know?](https://proceedings.mlr.press/v267/zhang25bx.html)：ICML 2025 对偏好分歧来源进行分类，并检查标准 reward modeling 与 LLM judge 对分歧的处理；支持记录分歧原因，不支持把所有分歧视作统一的人类偏好。
3. [Clean First, Align Later: Benchmarking Preference Data Cleaning for Reliable LLM Alignment](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b457a310c34efcaf6322d1bc6503ce16-Abstract-Datasets_and_Benchmarks_Track.html)：NeurIPS 2025 PrefCleanBench 比较 13 种清洗方法，并跨数据集、backbone 和优化目标评估；报告的多 judge 过滤优势受论文比较设置约束，不构成生产清洗规则。
4. [Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge](https://aclanthology.org/2025.ijcnlp-long.18/) 与 [Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement](https://proceedings.iclr.cc/paper_files/paper/2025/hash/08dabd5345b37fffcbe335bd578b15a0-Abstract-Conference.html)：前者检查 judge、候选和任务对位置偏差的影响；后者研究在小型人工校准集上选择性信任或升级 judge 的协议。其一致率/保证条件都不能脱离具体校准分布和假设。
5. [Human Preferences for Constructive Interactions in Language Model Alignment](https://arxiv.org/abs/2503.16480)：2025 arXiv 预印本分析跨国家、个体价值观和对话风格的偏好数据；7,500 多段对话、74 个国家和 21 个模型是该研究数据范围，不代表全球用户分布。
6. [Does Safety Live in a Shared Multilingual Subspace? Cross-Lingual (Non-)Transfer of the Refusal Direction](https://aclanthology.org/2026.mrl-1.19/)：MRL 2026 在两个开放模型和十种语言中检查拒答方向；用于说明跨语言安全机制不能默认等价，不能外推到所有模型或语言。

章节的来源阅读、证据条件、分歧处理、切片设计和固定 demo 验证范围见[第八章研究与精修记录](research/book-09-ch08-preference-safety-refinement-2026-10-28.md)。

### 第九册第九章：多模态、语音与视频数据

1. [CLIP](https://arxiv.org/abs/2103.00020)、[LAION-5B](https://arxiv.org/abs/2210.08402) 与 [Whisper](https://arxiv.org/abs/2212.04356)：分别作为图文弱监督、web-scale 图文数据集和大规模弱监督语音路线的历史入口；论文的规模、数据来源和收益不等于当前项目的授权或质量结论。
2. [DataComp](https://arxiv.org/abs/2304.14108) 与 [JEST](https://arxiv.org/abs/2406.17711)：前者用固定 CLIP 训练实现和 38 个下游任务比较图文数据筛选，后者研究 joint example selection；前者支持固定预算的端到端数据评估，后者说明 batch 可学习性具有组合性，均不能替代任务级验证、许可审查和独立安全门禁。
3. [EuroSpeech: A Multilingual Speech Corpus](https://arxiv.org/abs/2510.00514)：报告多语言长音频采集、非逐字转录和两阶段对齐；论文样本、语言覆盖、许可及 ASR 收益均绑定其来源和实验条件。
4. [Kairos](https://arxiv.org/abs/2609.08755) 与 [HAVEN](https://arxiv.org/abs/2605.19223)：前者提供 10—30 分钟视频的细粒度时间标注案例，后者联合 frame、shot、video 粒度的对齐评测；均为特定数据集/benchmark 的设计，不构成通用长视频配方或目标系统验收。

章节的来源阅读、公式定义、合成案例、指标分母及固定 demo 验证范围见[第九章研究与精修记录](research/book-09-ch09-multimodal-data-refinement-2026-10-28.md)。

### 第九册第十章：Data Attribution 与 Valuation

1. [Influence Functions](https://arxiv.org/abs/1703.04730)、[Data Shapley](https://arxiv.org/abs/1904.02868)、[高效 Shapley 数据估值](https://proceedings.mlr.press/v89/jia19a.html) 与 [TracIn](https://arxiv.org/abs/2002.08484)：用于解释局部影响、平均边际贡献和训练轨迹归因的经典方法；计算近似和结论均依赖模型、效用与目标。
2. [TRAK](https://proceedings.mlr.press/v202/park23c.html)、[DataInf](https://arxiv.org/abs/2310.00902) 与 [LoGra](https://arxiv.org/abs/2405.13954)：覆盖随机投影 after-kernel 归因及 LoRA/大模型 influence 的可扩展近似；LoGra 的吞吐和显存数字仅对应论文基线、模型和语料设置。
3. [PreSelect](https://proceedings.mlr.press/v267/shum25a.html)、[BLISS](https://proceedings.mlr.press/v306/hao26b.html) 与 [OPUS](https://proceedings.mlr.press/v306/wang26gc.html)：分别展示预测式筛选、双层 proxy influence 和优化器更新空间中的动态选择；各自模型、语料、阶段、预算和结果不可拼成统一排行榜。
4. [Convex Dataset Valuation for Post-Training](https://proceedings.mlr.press/v306/zeng26k.html)、[Mechanistic Data Attribution](https://proceedings.mlr.press/v306/chen26cv.html) 与 [RISE](https://arxiv.org/abs/2604.16197)：分别处理预算约束下的数据集价值、可解释单元的训练来源干预和输出层 readout sketch。RISE 为 2026 arXiv 预印本；所有方法都不能替代目标模型干预、授权审查和独立风险评估。
5. [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) 提供治理、测量和风险管理的组织框架，不是数据价格、许可或价值的判定依据。

章节的研究范围、公式手算、代码边界与未完成门禁见[第十章研究与精修记录](research/book-09-ch10-data-attribution-valuation-refinement-2026-10-28.md)。
