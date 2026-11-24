# SSM 是否替代 Transformer：跨架构证据研究

研究日期：2026-11-23  
研究问题：公开实验能否支持“SSM 会替代 Transformer”这一跨架构判断；证据对纯 SSM、特定任务、模型版本及混合模型分别支持到哪里。  
范围：以 Mamba 系列为主要可比案例，同时查找 SSM 任务能力的正反结果、状态记忆理论分析和适合初学者的解释材料。Serving 引擎、产品生态迁移与通用架构选择流程不属于本研究主题。

## 检索与筛选

检索平台包括 arXiv、ICLR Proceedings、PMLR、研究者/实验室博客及公开课程。检索词围绕 “Mamba empirical comparison 8B 3.5T tokens”, “Mamba retrieval ICL copying”, “Mamba-3 retrieval capabilities ICLR 2026”, “SSM Transformer tradeoffs”, “state space models lecture notes” 展开，并沿论文、会议页面及作者教学材料追查。

下列一手来源用于核验方法、实验条件和作者报告结果；Albert Gu 的文章和 Eva Koroleva 的课程用于发现解释路径，不把其中的类比或教学样例当作跨模型定律。没有运行模型、复现 benchmark 或重新计算论文结果。

## 外部主题与证据

### 纯 SSM 的匹配规模比较

**Roger Waleffe 等，An Empirical Study of Mamba-based Language Models，arXiv:2406.07887。**  
原文：[arXiv HTML](https://arxiv.org/html/2406.07887)，重点阅读 §§2.1–2.3、3.1–3.3.4、4.2–4.3.2 和结论。页面标示所读版本为 v1，2024-06-12。

- 研究比较约 8B 的 Mamba、Mamba-2 和 Transformer；架构参数约为 8.15B、8.24B 和 8.53B，使用同一训练数据系列和相近训练超参数。
- 1.1T tokens 设置中，纯 Mamba/Mamba-2 在五项常见任务（不计 MMLU）的平均成绩高于 Transformer；但 5-shot MMLU 分别低约 18.3 和 17.1 个百分点。
- 3.5T tokens 设置只训练纯 Mamba-2 和 Transformer，没有训练纯 Mamba。增加训练 token 后，Mamba-2 的 5-shot MMLU 与 Transformer 的差距缩至 1.37 个百分点。不能把这一结果概括成三种架构在 3.5T 下全部匹配比较。
- MMLU 结果会随选项呈现格式变化；文中 cloze 格式下，1.1T 纯 SSM 与 Transformer 的相对差异不同于标准选项格式。它说明评测任务构造会影响观察到的能力，不等于任意 prompt 都有相同效应。
- 8B Mamba-2-Hybrid 在 3.5T tokens 的 12 项常见任务上高于对应 Transformer，平均高 2.65 点。这是 hybrid 结果，不是纯 SSM 替代 Transformer 的证据。
- 论文报告的长上下文“近 8 倍”是 batch 32 下、长输入生成单 token 的预测时间比；短上下文预测差异较小。该数值不代表端到端服务实测或一般硬件保证。
- 长上下文结论随任务构造变化：LongBench 与 RULER 中 HotpotQA 的干扰段落生成方式不同，hybrid 相对 Transformer 的结果方向也不一致。基准名称相同或题型相近，不足以保证输入分布等价。

**支持范围：** 匹配规模和较多训练 tokens 下，纯 Mamba-2 能在部分标准语言任务接近或超过同研究 Transformer；同研究中仍存在任务差异。纯 SSM 与 hybrid 必须分开归因。该研究不覆盖所有 SSM、所有模型规模、所有训练流程或现实服务工作负载。

### 精确复制与上下文检索

**Samy Jelassi 等，Repeat After Me: Transformers are Better than State Space Models at Copying，arXiv:2402.01032。**  
原文：[arXiv HTML](https://arxiv.org/html/2402.01032)，重点阅读摘要、§§2–4 与讨论。

- 理论部分在其 generalized state space model（GSSM）定义和合成精确复制任务下分析固定大小输入相关状态；其结论依赖这一形式化模型、任务和记忆假设。
- 实验包括合成复制的样本效率与长度外推，以及预训练 Pythia/Mamba 的 phonebook 查找。作者观察到 Transformer 在这些复制/查找设置中领先。
- 论文也承认，大状态理论上可能装下整个输入；其结论不能不加限定地改写成“任意 SSM 无法复制”或“SSM 无法完成自然语言任务”。

**支持范围：** 显式逐位置历史在精确复制与任意上下文寻址类任务中具有直接优势；其理论下界不是对每个 SSM 变体、每个任务或每个实际配置的普遍性能定律。

### ICL 不是单一能力

**Riccardo Grazzi 等，Is Mamba Capable of In-Context Learning?，AutoML 2024，PMLR 256。**  
原文：[PMLR 页面](https://proceedings.mlr.press/v256/grazzi24a.html)，阅读摘要和出版信息。论文报告 Mamba 在简单函数近似及若干 NLP ICL 任务中接近 Transformer；它只支持这些任务和设置下的比较，不能推广到所有 ICL 任务。

**Marco Bondaschi 等，From Markov to Laplace: How Mamba In-Context Learns Markov Chains，ICLR 2026。**  
原文：[ICLR 2026 页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8fef8a350b54e18758ea546bc984ba28-Abstract-Conference.html)。作者在 Markov 链 ICL 任务中报告单层 Mamba 能学到 Bayes/minimax 最优的 Laplacian smoothing estimator，并给出相应理论分析。

**支持范围：** “ICL”包括函数/规则推断、序列统计预测、样例适配和关联回忆等不同任务。某一任务上的成功或失败不能代表所有 ICL。

### 后续版本改善部分证据，但没有形成普遍胜出

**Aakash Sunil Lahoti 等，Mamba-3: Improved Sequence Modeling using State Space Principles，ICLR 2026。**  
原文：[ICLR 2026 正式页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html)；[arXiv HTML 全文](https://arxiv.org/html/2603.15569)，重点阅读 §§4.1.2–4.3 和结论。

- 论文在 1.5B 预训练模型上评估 retrieval；真实数据任务采用 2K token 截断和 cloze 格式。Mamba-3 在 TQA、SQuAD 的关联回忆/问答上有竞争力，但在半结构化 SWDE 与非结构化 FDA 信息抽取上较弱；合成 NIAH 多数任务可匹配或超过基线，并优于 Mamba-2 的部分分布外检索结果。
- 作者指出 Transformer baseline 在汇总 retrieval 成绩上仍明显更强；对长序列 NIAH 的解释也受位置编码差异影响。
- Mamba-3 的 hybrid 实验使用线性层与 NoPE attention 交错，比例为 5:1。该结果不能归到纯 Mamba-3。
- “1.5B 下平均下游准确率高 0.6 点、MIMO 再增 1.2 点”的比较对象是论文列出的最佳 Gated DeltaNet 基线，不是对所有 Transformer 的结论。

**支持范围：** SSM 版本、参数化、状态大小、任务格式和数据分布会改变结果；某些能力改进可与某些检索短板同时存在。该工作不能证明整体语言模型任务或整个 SSM 家族已经普遍取代 Transformer。

### 机制权衡与讲解组织材料

**Albert Gu，On the Tradeoffs of SSMs and Transformers，Goomba Lab，2025。**  
原文：[作者文章](https://goombalab.github.io/blog/2025/tradeoffs/)，阅读状态记忆、token cache 与 tradeoff 部分。作者用压缩状态与逐 token cache 解释在线处理和细粒度检索的机制张力；文章明确包含作者的启发式和观点，适合作为教学组织线索，不单独支撑普遍能力结论。

**Eva Koroleva，Beyond attention with Mamba, state space models and hybrids，公开课程第 10 课，2026-10-19 发布、2026-11-10 更新。**  
原文：[课程讲义与 notebook](https://blog.korolevalab.com/blog/2026/transformer-architectures-10-mamba/)，阅读“Same data, different machinery”“Hybrids”等部分。它将状态递归、长度、复制任务、混合结构按对象和任务连接，适合作为讲解顺序参照；课程中的小模型实验和模型配置观察只支持其声明的版本与环境。

## 综合判断与写作边界

现有证据更适合支持条件化判断，而非架构家族的胜负宣言：理论复杂度说明了不同的信息保存和计算路径；受控模型实验显示任务、训练 token、prompt 格式和模型版本会改变相对结果；特定复制/检索任务暴露纯固定状态路径的困难；ICL 正结果和后续版本的部分 retrieval 改进说明这些任务不能压成“会/不会”二分。混合模型需单列，不能把它们的提升算作纯 SSM 的能力。

这是跨来源综合推断，不是任一论文直接证明的总体定理。章节应保留任务、参数规模、token 预算、题型/输入分布和纯/混合模型的证据边界；无需重写通用实验协议、Pareto 选择、serve 测量或 hybrid 接口，分别由第 42、43 章及服务相关书册承担。新增证据可由后续大规模 matched study 或跨任务复现改变，维护时需按论文版本复查。
