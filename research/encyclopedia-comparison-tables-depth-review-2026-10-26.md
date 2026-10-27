# 第四册第十五章《方法比较与工程取舍》深度精修研究记录（2026-10-26）

## 研究目标与边界

- 对象：`book-04-llm-encyclopedia/chapters/15-面试高频对比表.md`，15.1—15.43，修订前约 795 行，修订后 832 行。
- 目标：沿用全部 43 组比较及 Adam/AdamW 教学代码，不把章节改成只列方案名称的目录；补足方法层次、公式变量、研究证据、比较条件、失败诊断和适用边界。
- 研究问题：哪些方法解决同一类约束，哪些实际上位于不同层；论文声称的收益在什么模型、任务、预算和硬件下观察到；哪些后续方法处理了前一方案的具体不足；比较结果为何不能直接当作普遍排名。
- 来源处理：本轮打开并阅读所列原论文的摘要、方法/实验或限制章节；Anthropic agent 文章仅作为工程经验来源，PyTorch/NVIDIA 文档仅支持其公开接口/数值格式。仅浏览摘要的来源不用于支撑正文中的精细实验结论。论文作者报告的结果不视为独立复现。
- 引用校正：最终来源审计发现 arXiv `1801.06146` 是 ULMFiT，而不是 GPT；章节已改用 [OpenAI GPT 原始预训练报告](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)。arXiv 元数据显示 `2501.01880` 于 2024-12-27 提交；正文保留其预印本身份，不因 arXiv 编号年份推断提交年份。

## 形成的关键理解与正文处理

### 优化器、归一化、位置和 FFN（15.2—15.11）

- [AdamW 原论文](https://arxiv.org/abs/1711.05101) 将 Adam 中的 L2 梯度惩罚与 decoupled weight decay 区分开来。其 CIFAR-10 / ImageNet32x32 实验支持特定 ResNet、学习率日程及数据设置中的泛化改善，不支持“AdamW 对每个模型和任务必然更好”。正文补上这条证据范围，并保留参数分组、学习率、weight decay 和搜索预算的控制条件。
- [LayerNorm 原论文](https://arxiv.org/abs/1607.06450) 指出统计量按单样本层内神经元计算，因此训练与测试计算一致；[RMSNorm 原论文](https://arxiv.org/abs/1910.07467) 把简化假设明确为中心化不必要，并在若干不同架构/任务中报告可比质量和耗时变化。正文把“少一次中心化”与“实际训练/端到端更快”分开，不把单篇论文的 7%—64% 运行时间差异外推到所有硬件。
- [Pre-LN/Post-LN 分析](https://arxiv.org/abs/2002.04745) 将 Post-LN 初始化时输出附近梯度较大、warmup 稳定训练的结论与其 mean-field 理论和实验设置绑定。正文说明 Pre-LN 的直接残差梯度路径及其训练稳定性收益，不把它表述为所有深度/初始化下的最优结构。
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864) 和 [ALiBi](https://arxiv.org/abs/2108.12409) 支持旋转 Q/K 与 attention score 距离 bias 的机制定义；ALiBi 的 train-short/test-long 实验只说明具体模型由 1,024 外推到 2,048 的结果，不能据此保证任意长上下文质量。正文进一步区分位置外推、窗口上限和有效信息利用。
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) 对比 FFN 门控变体，支持“门控可能提升所测序列任务”的限定表述。正文要求控制中间宽度、参数量、训练 token 和计算预算，再判断激活/门控差别。

### 后训练、偏好、采样与参数高效微调（15.12—15.22）

- [InstructGPT](https://arxiv.org/abs/2203.02155) 的流程是示范 SFT、偏好数据训练 reward model、再用 PPO 优化 policy；[DPO](https://arxiv.org/abs/2305.18290) 由 KL 约束的偏好优化推得离线 pairwise loss，论文的实验和讨论并未证明离线 DPO 对分布外行为普遍稳健。正文区分“SFT/RLHF/DPO 是阶段、反馈形式还是优化算法”，并把“不主动采集新策略样本”限定为经典离线 DPO，提示在线 DPO/RL 变体存在。
- [Constitutional AI](https://arxiv.org/abs/2212.08073) 采用原则引导的自我批评/修订和 AI feedback；[RLAIF vs. RLHF](https://arxiv.org/abs/2309.00267) 的 human-preference 比较覆盖摘要、helpful dialogue、harmless dialogue 等具体任务，不能推出 AI feedback 在其他领域就是人类偏好的等价替代。该研究还实际测量了候选顺序偏差，并通过倒序评审处理；正文补充评审器校准和高风险人工金标准。
- [Nucleus sampling 原研究](https://arxiv.org/abs/1904.09751) 将解码质量变化与同一语言模型的解码规则联系起来。正文将 beam 的局部候选搜索与采样的多样性用途分开，并限制在特定生成任务，不把开放对话中的风格观察泛化为所有 beam 实现。
- [LoRA](https://arxiv.org/abs/2106.09685)、[Adapter](https://arxiv.org/abs/1902.00751)、[Prefix-Tuning](https://arxiv.org/abs/2101.00190) 分别支持低秩权重更新、层内模块和连续前缀的机制。原论文“低 rank/少量参数也可达可比质量”的结论依赖其任务、骨干和数据设置；正文把参数量、优化器状态、推理时延、上下文/KV 占用和多 adapter 路由成本分账。
- [量化与整数训练](https://arxiv.org/abs/1712.05877) 联合讨论整数推理和训练中模拟量化误差；正文说明 QAT 通常在训练期引入 fake-quant/量化误差模拟，PTQ 的校准需求依赖量化对象/方法，不写成所有 PTQ 都必须校准。量化是否更快取决于目标硬件和低比特 kernel。
- FP16/BF16 的字段与范围以 [NVIDIA CUDA 浮点格式文档](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html) 为格式依据；正文补 exponent/mantissa 数值，不把指数范围与端到端速度混为一谈。

### 注意力实现、推理服务和训练并行（15.23—15.27）

- [FlashAttention](https://arxiv.org/abs/2205.14135) 是针对标准 exact attention 的 IO-aware tiling/recomputation，减少 HBM 访问；不会改变 dense attention 的二次 token-pair 计算量。[BigBird](https://arxiv.org/abs/2007.14062) 代表稀疏连接图以降低长序列复杂度的路线。正文说明稀疏 mask 的输出可以相对该 mask 精确，但它相对全连接注意力省略了边；质量风险来自被省略的依赖，稀疏图也必须由硬件友好 kernel 实现才会获得 wall-clock 加速。
- [PagedAttention/vLLM](https://arxiv.org/abs/2309.06180) 同时包含分页 KV 管理、缓存共享、调度和系统实现，论文报告的 2—4 倍吞吐提升绑定其模型、负载、系统和对比 baseline。正文避免将 PagedAttention 缩减成“allocator”或将论文性能外推；说明与 FlashAttention 是可共存的不同瓶颈，但具体 kernel 必须支持对应 KV block layout。
- [GQA](https://arxiv.org/abs/2305.13245) 的方法包含从 MHA checkpoint 转换及额外 uptraining；论文报告以原预训练计算量 5% uptrain 的设置，GQA 质量接近 MHA、速度接近 MQA。正文将这组结果标成论文配方，而非任意模型的保证。
- [ZeRO](https://arxiv.org/abs/1910.02054) 分别分片 optimizer states、gradients、parameters；当前 [PyTorch FSDP stable 文档](https://docs.pytorch.org/docs/stable/fsdp.html) 说明 FULL_SHARD 的参数 all-gather、梯度 reduce-scatter 和 NO_SHARD 复制语义。正文新增 ZeRO Stage 1/2/3 表格，并说明 FSDP_FULL_SHARD 与 ZeRO-3 概念相近但 API、包装粒度和运行时不能视为同一实现。`S/N` 只估算均匀分片后的持久状态，峰值仍含临时 gather/buffer。

### RAG、上下文、检索与重排（15.28—15.31）

- [原始 RAG](https://arxiv.org/abs/2005.11401) 以 BART、DPR、Wikipedia 向量索引和知识密集型 NLP 任务为研究对象，且有 RAG-Sequence 与 RAG-Token 两种结构。正文把知识可更新/可审计、行为可塑性作为工程选型方向，并补充访问控制要由检索管线强制执行，不能只写在 prompt 中。
- [DPR](https://arxiv.org/abs/2004.04906) 说明 dense retriever 的双编码器机制和其在特定开放域 QA 数据上的检索表现；正文不把语义 dense / lexical sparse 写成互斥类别，补充 query/document 分布、实体/数字、跨语种和 hybrid merge 的评估切片。
- [BERT Passage Re-ranking](https://arxiv.org/abs/1901.04085) 作为 cross-encoder rerank 入口；正文定义 recall@K 与条件 rerank success@M 的链式概率，要求分别标注 gold evidence，避免仅凭最终生成错误归因。
- 两项 RAG/long-context 研究报告了有范围差异的结果：[Google Research 的 2024 对照](https://arxiv.org/abs/2407.16833) 使用 GPT-4o、Gemini 1.5 Pro、GPT-3.5 Turbo 和 LongBench/∞Bench 的部分数据集；其结果平均偏向 LC，但超出 GPT-3.5 context window 的数据有例外，且其检索器是特定 dense retriever/chunk/top-k 设置。[2024-12-27 提交的后续再评估预印本](https://arxiv.org/abs/2501.01880) 扩展并过滤 QA 题，报告结果随问题类型和检索策略变化，摘要指出 summarization-based retrieval 可接近 LC、chunk retrieval 落后，RAG 在 dialogue/general queries 有优势。两者都不是所有应用的 universal ranking。正文在“长上下文与 RAG”一节补研究对照和公平比较条件。
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) 使用多文档 QA 与 key-value retrieval 展示证据位置影响；正文明确其为一类任务下的现象，不把它当成所有长上下文模型/任务恒定失效。

### Agent、推理、评估、可靠性与 MoE（15.32—15.41）

- [Anthropic Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) 是特定工程团队在 2024-12 发布的经验，明确将预定义代码路径 workflow 与模型动态控制 agent 区分；正文将其作为可操作定义和经验来源，注明“agent/workflow”在社区口径中不统一，workflow 内仍可有动态 LLM 判断。
- [Chain-of-Thought](https://arxiv.org/abs/2201.11903) 与 [ReAct](https://arxiv.org/abs/2210.03629) 分别代表中间文本推理与推理/动作交替获取环境反馈。[CoT 忠实性研究](https://arxiv.org/abs/2305.04388) 在特定受控扰动中发现解释可能合理但未忠实报告决策影响因素。正文把 CoT 当作生成的中间文本，不能默认它是忠实内部思维或外部证据。
- [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) 研究报告了位置、冗长、自我增强等偏差；某些题型/相近质量回答更易出现位置效应。正文加入顺序交换、rubric、人工校准集、分歧切片及任务成功客观指标。
- [HumanEval / pass@k](https://arxiv.org/abs/2107.03374) 给出候选集上的无偏估计；正文解释估计目标、`n/c/k` 定义域和候选生成预算，增加小型计算示例。样本通过率仍受 test coverage 和生成分布制约。
- [FActScore](https://arxiv.org/abs/2305.14251) 将长回答拆为 atomic facts 并逐项与可靠知识来源核对；正文用它支持 factuality/grounding 的不同评测对象，同时明确“有来源支撑”不等于对现实事实永久正确。
- [Jailbroken](https://arxiv.org/abs/2307.02483) 讨论 safety training 的 competing objectives/mismatched generalization；[Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) 讨论外部检索内容被当成指令的系统攻击链。正文说明 jailbreak 与 prompt injection 的定义有重叠：一个按越狱目标/策略描述，另一个按不可信指令进入上下文/数据路径描述；直接与间接入口应另标，工具权限是独立执行边界。
- [Switch Transformers](https://arxiv.org/abs/2101.03961) 展示 sparse MoE 的参数容量与 token 计算分离，也报告通信、复杂度和训练不稳定性是实际采用障碍。正文定义 total/active parameters，并说明总权重仍需存储、跨设备 all-to-all/负载不均会影响端到端延迟。

## 原章节知识点覆盖映射（coverage_map）

本轮保留全部比较关系，下面记录各节进入修订后章节的位置与解释层级；不将本表当作正文替代。

| 原节 | 原有主题 | 本轮处理和解释层级 |
| --- | --- | --- |
| 15.1 | 比较指标、成本和失败模式 | 保留；明确测量对象、Pareto 支配与成本/延迟单位，避免同名指标跨任务直接排榜。 |
| 15.2 | Adam / AdamW 和 toy update | 保留代码；限定 AdamW 论文实验范围，复核更新式和参数/超参控制。 |
| 15.3 | SGD / Adam | 保留 optimizer-state 与收敛/泛化口径；改成条件性陈述。 |
| 15.4 | BatchNorm / LayerNorm | 保留统计轴差异；引用原论文并区分训练/推理统计状态。 |
| 15.5 | LayerNorm / RMSNorm | 保留公式；补原假设、论文报告范围和硬件性能外推边界。 |
| 15.6 | Pre-Norm / Post-Norm | 保留梯度路径；引入 warmup、初始化、深度等条件。 |
| 15.7 | BERT / GPT | 保留 encoder/decoder 目标和接口差异；为 MLM/causal LM 附原始论文入口。 |
| 15.8 | Encoder-decoder / decoder-only | 保留 cross-attention 与统一 next-token 路线；补输出长度/输入复用的选型因素。 |
| 15.9 | MHA / MQA / GQA | 保留 KV 公式；补 GQA uptraining 方法和结果边界。 |
| 15.10 | Absolute / RoPE / ALiBi | 保留三种注入机制；区分训练外推与有效利用。 |
| 15.11 | ReLU / GELU / SwiGLU | 保留 FFN 门控；强调参数/FLOPs 公平控制。 |
| 15.12 | SFT / RLHF / DPO | 保留三阶段路线；术语拆分并标示可组合的后训练阶段。 |
| 15.13 | RLHF / DPO | 保留 reward model、PPO 与离线 pairwise loss；把 offline 边界限定于 canonical DPO。 |
| 15.14 | RLHF / RLAIF | 保留反馈来源差异和错误率；补 AI judge 校准、位置偏差和任务域限制。 |
| 15.15 | Top-k / Top-p | 保留；补累计概率集合定义和结构约束边界。 |
| 15.16 | Temperature / Top-p | 保留联合调参；增加 logits 例子说明变换顺序。 |
| 15.17 | Greedy / Beam / Sampling | 保留；补局部/序列分数目标、长度偏差和任务匹配。 |
| 15.18 | LoRA / full fine-tuning | 保留公式；定义矩阵尺寸并区分训练内存、部署成本和可塑性。 |
| 15.19 | LoRA / Adapter / Prefix | 保留已有表格；补经典方法出处、缓存/序列开销及多任务部署因素。 |
| 15.20 | Quantization / Distillation | 保留并组合视角；定义量化取整/裁剪和目标硬件限制。 |
| 15.21 | PTQ / QAT | 保留校准与训练期误差适应；不把校准要求泛化到所有算法。 |
| 15.22 | FP16 / BF16 | 保留；列有效数值字段并说明硬件/kernel 决定速度。 |
| 15.23 | FlashAttention / Sparse Attention | 保留 exact IO optimization 与 sparse mask 差异；解释 sparse 精确性相对 mask 的含义。 |
| 15.24 | PagedAttention / FlashAttention | 保留分配/调度与算子 IO 的层次差异；避免将 PagedAttention 简化为 allocator。 |
| 15.25 | KV cache / activation checkpoint | 保留 inference/train 分期差异；增加单 token 重算对照与 RNG/dropout caveat。 |
| 15.26 | DP / TP / PP | 保留并行轴；定义 data-parallel group 中全局 batch 的口径。 |
| 15.27 | DDP / ZeRO / FSDP | 保留；展开 ZeRO 三阶段、FSDP 全分片轨迹、近似内存式和 transient peak。 |
| 15.28 | RAG / fine-tuning | 保留且可组合；把事实更新/权限与行为适配分开。 |
| 15.29 | dense / sparse retrieval | 保留 hybrid；补误差切片及分数融合不能生硬相加。 |
| 15.30 | retriever / reranker | 保留条件概率关系；定义召回率、条件排序成功率，增加手算例子。 |
| 15.31 | long context / RAG | 保留组合可能；加入两篇有范围差异的研究证据和总成本模型限制。 |
| 15.32 | agent / workflow | 保留动态决策与固定控制流；标注定义口径、人工审批和工程文章证据级别。 |
| 15.33 | CoT / tool use | 保留内部分解与外部观察；增加 CoT 忠实性边界和可核验工具结果。 |
| 15.34 | human eval / LLM judge | 保留人工/judge；增添盲评、顺序交换和人工标定做法。 |
| 15.35 | accuracy / pass@k | 保留公式、输入边界；加入手算和候选成本解释。 |
| 15.36 | factuality / grounding | 保留并行指标；增加 atomic facts 评测映射与来源条件。 |
| 15.37 | robustness / generalization | 保留区别；注明术语在不同文献中并非互斥，评测用受控语义保持扰动。 |
| 15.38 | over-/under-refusal | 保留指标；明确 confusion matrix 正类与手算分母。 |
| 15.39 | jailbreak / prompt injection | 保留两类风险；澄清分类维度重叠并拆文本策略与系统执行边界。 |
| 15.40 | helpful / harmless | 保留双目标；连到偏好数据与安全动作边界，不把拒绝率当完整安全指标。 |
| 15.41 | dense / MoE | 保留路由公式；明确定义 total/active 参数及通信和部署驻留成本。 |
| 15.42 | 比较表写法 | 保留模板，新增一个 RAG/微调选型实例，示范从需求到验收指标的推导。 |
| 15.43 | 来源与证据边界 | 保留并扩展索引；将代表性来源放到支持该主张的正文附近。 |

## 尚未完成与风险

- 40 余组比较不等于本轮对所有方法全文综述；来源覆盖优先给到有代表性的对照和直接支持关键机制/评测的材料。
- 本轮没有对量化、分布式训练、检索、解码或 Agent runtime 做本地复现实验；性能结论仍须绑定论文与系统配置。
- 人工/LLM judge、安全拒答和 factuality 的指标依赖任务标注与风险分类；通用公式需在正文定义正类和分母。
- RAG/long-context 的代表性比较研究目前仍绑定特定模型、数据、retriever、chunk 与预算；部署选型需要目标语料/模型实测。
- AdamW 教学 Python 代码和其它存量示例属于已存在代码；本轮按项目指令静态审阅，不把未执行代码记成运行通过。
- 配套检查：`INTERVIEW_BANK.md` 已含 Adam/AdamW、temperature/top-k/top-p、BF16、DDP/FSDP、FlashAttention/sparse attention、Retriever/Reranker、Pass@k 与 Agent/RAG 相关题目；`GLOSSARY_EN_ZH.md` 已收录 AdamW、FSDP、FlashAttention、PagedAttention、MoE、Beam Search、Top-p、RAG 和 Reranker 等术语；本轮在 `PAPERS.md` 补入 GPT 原始报告，并保留 RAG/长上下文比较论文。现有索引覆盖主要高频主题，无需重复新增练习或术语条目。
