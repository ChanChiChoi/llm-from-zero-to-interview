# 长上下文评测与有效能力：来源证据

检索与阅读日期：2026-11-03；新增论文核对日期：2026-11-20。本记录保存论文和官方 benchmark 文档支持的任务设计、版本、数字与证据边界；书册审阅决定与去向只写在 `progress/`。

## 教学与组织材料（2026-11-20）

- **Stanford CS224V Fall 2026 课程安排：** [课程日程](https://web.stanford.edu/class/cs224v/schedule.html)，核读 11/11 的 “Long-Context Long-Horizon Deep Researcher” 和 11/16 的 “Agent Context Management”。日程把跨文档阅读、分析、比较，与上下文压缩、外部存储、长期记忆分作相邻课题；可作为从模型内上下文利用过渡到外部状态管理的组织样本。它只记录课程实际编排，不证明这种顺序普遍更优；系统/harness 细节仍由相关章节主讲。
- **技术博客（页面署名 wh）：** [Evaluating Long Context (Reasoning) Ability](https://nrehiew.github.io/blog/long_context/)，2026-11-20 核读 “Existing Long Context Evaluation” 与 “What makes a good long context benchmark?”。文章将长上下文任务拆成相关信息定位与后续任务推理，并指出只扩大搜索空间不足以代表真实长程推理；这是作者的基准设计观点和自建 LongCodeEdit 案例，不作为普遍 benchmark 结论。作者页面未提供足以独立判断身份的详细介绍，因此仅借其解释路线作教学参照，核心事实仍以原始论文核验。

## NoLiMa v3：降低词面捷径

- **来源与版本：** Modarressi et al., [NoLiMa: Long-Context Evaluation Beyond Literal Matching](https://arxiv.org/abs/2502.05167v3)，arXiv v3，2025-07-09。
- **阅读范围：** 摘要、§3 任务构造、§4 实验与主要结果、附录 E 延伸设置。
- **来源支持的事实：** NoLiMa 通过降低问题与 needle 之间的词面重合，要求模型使用潜在关联完成检索。v3 摘要报告评测 13 个自称支持至少 128K 的模型；32K 时 11 个模型跌到各自短上下文基线的一半以下；GPT-4o 从 99.3% 基线降到 69.7%。论文以基线比例阈值描述 effective length，并在附录分析更新模型及更长长度。
- **证据边界：** 数字绑定论文的 needle/haystack、prompt、模型 revision、任务子集与长度设置；单一关联检索任务不代表开放式多文档推理，也不能作为产品上下文上限。

## LongHarness Bench v1：模型与长上下文 Harness 的质量—成本

- **来源与版本：** Pham et al., [LongHarness Bench: Stress-Testing Language Model Harnesses for Long-Context Reasoning](https://arxiv.org/abs/2609.38137v1)，arXiv v1，提交于 2026-09-29。更具体的任务与结果核验也见 `research/long-context-harness-and-agent-memory-2026.md`。
- **阅读范围：** arXiv 摘要、任务构造 §3.1—§3.4、模型/harness 设置和结果 §§4.1—4.5。
- **来源支持的事实：** 论文提出四项需要多样检索、语义干扰和步骤依赖检索的任务，比较直接推理与四种 Agent harness，覆盖五个模型。Constraint Solving Search 的一个样本约含 120K tokens、160 人材料和 3—5 个条件。作者报告最佳模型—harness 配置的四任务 macro accuracy 为 68%，没有一种 harness 在四项任务上都领先；具体成本随 harness 显著变化。
- **证据边界：** 这是单版预印本。任务、模型、effort、harness 实现和 API 价格都限定结果；它支持在长上下文系统评测中记录质量与计算成本，不支持跨产品的 harness 排名，也不是 1M 通用任务。

## LongBench Pro v1：自然双语长文与多维任务分层

- **来源与版本：** Chen et al., [LongBench Pro: A More Realistic and Comprehensive Bilingual Long-Context Evaluation Benchmark](https://arxiv.org/abs/2601.02872v1)，arXiv v1，提交于 2026-01-06；截至本次核验未见后续版本。
- **阅读范围：** 摘要、数据构造与质量检查 §§3—4、evaluation settings 与结果 §§5.1—5.2。
- **来源支持的事实：** 数据集含 1,500 条中英文自然长上下文样本，11 个主任务、25 个次任务，长度为 8K—256K tokens；按 full/partial context dependency、六档长度和四档难度分层。作者用前沿模型起草问题、答案和解题材料后由专家复核；从 300 条样本抽查的属性正确率为 99.3%、答案正确率为 97.3%。论文报告比较 46 个模型。
- **证据边界：** 论文是单版预印本；模型辅助构题、专家抽样核验和论文内的长度/输出预算设定都属于 benchmark 合同的一部分。最高 256K 的数据不证明 1M 表现，论文比较也不能直接预测其他业务语料。

## RULER：可控长度与复杂度的合成诊断

- **来源与版本：** Hsieh et al., [RULER: What’s the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654v3)，arXiv v3，2024-08-06。
- **阅读范围：** 摘要、§3 benchmark 设计、§4 实验结果与 limitations。
- **来源支持的事实：** RULER 提供可配置长度和任务复杂度，覆盖多类型/多数量 needle、variable tracking、多词聚合和 QA。作者对 17 个长上下文模型运行 13 项代表任务，并报告虽然模型在 vanilla NIAH 上接近满分，但只有约一半模型能在其 32K “satisfactory performance” 标准下维持表现。
- **证据边界：** 这是合成任务套件；“满意”由作者在该论文中定义。论文明确指出位置控制、现实长任务相关性和 prompt robustness 等限制，不能把 32K 结果当成统一产品门槛。

## LURCH：长度之外的证据布局与推理难度

- **来源与版本：** Qin et al., [Beyond Context Length: Characterising the Difficulty of Long-Context Reasoning](https://arxiv.org/abs/2610.22067v1)，arXiv v1，提交于 2026-10-15。
- **阅读范围：** 摘要、§1、§§4.1—4.4、§§5.1—5.4、结论及附录 E 的 reasoning-graph annotation validation。
- **来源支持的事实：** LURCH 从 17 个来源保留 6,550 个自然长上下文英文 QA 样本，原文上下文长度为 32K—256K tokens。作者提出 evidence scope（需要找到多少信息）、evidence dispersion（证据在上下文中分布多广）和 reasoning depth（证据到答案图中的依赖步数）作为补充长度的任务属性；在所测 4 个模型和该数据集内，三者与较低表现相关，证据分散样本随上下文变长退化更明显。
- **证据边界：** 这是单版预印本；结果是 LURCH 任务中的条件相关而非因果效应。evidence 扩展与 reasoning graph 含自动标注；两位模型标注者的 depth 完全一致率为 65.6%、Spearman 相关为 0.48，最终分析选用其中一个标注者的图。样本来源、任务构成、图标注和 4 模型集合都限制外推，三个属性不是被证明适用于全部任务的完备度量。

## CRE：推理任务中的位置、填充内容与长度控制

- **来源与版本：** Zhang et al., [Positional Failures in Long-Context LLMs: A Blind Spot in Reasoning Benchmarks](https://arxiv.org/abs/2605.23170v1)，arXiv v1，提交于 2026-05-22。
- **阅读范围：** 摘要、§1、§§3.1—3.5、§4 结果和 Limitations。
- **来源支持的事实：** Context Rot Evaluation (CRE) 在 GSM8K 与 ARC-Challenge 任务中交叉控制 target-task 位置、filler 内容和上下文长度；论文使用 8K/32K/64K 三档长度、三类 filler、end/middle 位置和每条件 50 个样本，覆盖 9 个经不同 provider API 运行的模型。作者报告位置影响随模型、filler 和长度而变化，并主张推理任务也需要位置控制。
- **证据边界：** 这是单版预印本和受限任务样本；每格 N=50，许多 64K 结果只有一个 seed；provider/API 异质性影响跨模型幅度比较，ARC-Challenge 结果功效有限。该设计展示一种因子控制方法，不是已验证的通用 benchmark 规范；文中 duplicate-at-end probe 也不隔离内部机制或构成产品修复方案。

## Lost in the Middle：关键信息位置影响上下文使用

- **来源与版本：** Liu et al., [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/html/2307.03172)，arXiv v3（2023-11-20），后发表于 TACL。
- **阅读范围：** 摘要、§§2.1–2.3 的多文档问答设置与结果、§§3.1–3.2 的键值检索设置与结果。
- **来源支持的事实：** 作者在多文档问答和合成键值检索中控制上下文长度及相关信息位置；在论文测试的模型和设置中，性能往往在相关信息靠近开头或结尾时较好，位于长上下文中间时会下降。作者还发现，扩展上下文窗口本身并不必然提高同长度输入上的信息使用表现。
- **证据等级：** 原始论文的实验协议和作者报告结果。
- **证据边界：** 结果绑定论文使用的 2023 年模型/API 版本、提示、任务和长度；其中键值任务是合成检索，多文档问答也按特定检索语料构造。不能把 U 形曲线视为所有当前模型的固定性质，也不能由此认定 attention 本身造成该现象。

## LongBench v1 与 v2：任务范围和长度单位

- **LongBench v1 来源与版本：** Bai et al., [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508v2)，arXiv v2，2024-06-19。摘要和任务构造段落报告 21 个数据集、六类任务、英语平均 6,711 words、中文平均 13,386 characters；任务涵盖单/多文档 QA、摘要、few-shot、合成任务和代码补全。长度均值不是百万 token 指标，汇总也不消除任务异质性。
- **LongBench v2 来源与版本：** Bai et al., [LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks](https://arxiv.org/abs/2412.15204v2)，arXiv v2，2025-01-03。摘要和数据统计报告 503 道高难多选题，覆盖六类任务，长度为 8K—2M **words**；近 100 位参与者参与采集，论文报告 15 分钟时限的人类专家准确率 53.7%、直接作答最佳模型 50.1%、o1-preview 较长推理 57.7%。
- **证据边界：** 两版本均有各自固定的任务、语言、评测器和模型集合；LongBench v2 的长度单位是 words，不得改写成 tokens。人工难题分数不代表普通业务样本总体。

## Google DeepMind MRCR v2：固定共指任务的长度曲线

- **来源与版本：** Google DeepMind [`eval_hub/eval_hub/mrcr_v2/README.md`](https://github.com/google-deepmind/eval_hub/blob/master/eval_hub/mrcr_v2/README.md)，`master` 分支，2026-11-03 阅读；仓库会滚动变化，引用或复现实验应记录 commit。
- **来源支持的事实：** 多轮对话重复相近请求、每次回答不同，评测要求数出目标实例并复现对应回答。仓库当时说明会报告 8-needle 的 upto_128K cumulative 与 at_1M pointwise 变体，生成尺度可达 8M；在其随机复制噪声模型下，2/4/8 needles 的噪声估计约为 51%/27%/15%。README 还指出若给予代码工具，任务会明显更简单，比较时必须声明工具条件。
- **证据边界：** 这些噪声值取决于 README 明示的随机猜测模型；任务的推理复杂度保持较低，主要测长程共指/定位，不能代表一般多证据推理。`master` 代码、数据、metric 和官方报告变体可能更新。
