# 第五册第十四章：训练系统综合设计与故障诊断研究记录

日期：2026-10-26  
正文：[`book-05-llm-training/chapters/14-训练系统综合设计与故障诊断.md`](../book-05-llm-training/chapters/14-训练系统综合设计与故障诊断.md)  
依据：当前 [`WRITING_SPEC.md`](../WRITING_SPEC.md)、`progress_v2.md`

## 本章定位与问题图谱

本章是第五册的综合设计章，承担“把前文概念放进一个训练系统并从观测追到机制”的任务，不取代前面章节对优化器、数据工程、分布式训练、后训练和多模态训练的完整讲解。原稿已有 7B 训练方案账本、loss spike、平均 loss 与能力的差异、benchmark contamination、batch/LR、并行、显存、SFT、VLM、成本、模型阶段、DPO/PPO、一个 Python 审计账本和练习。本轮保留这些入口，重点补充三条曾停留在概括层的证据链：

1. 数据 mixture 如何由“文件比例”变成可检验的优化问题，DoReMi 的代理模型如何产生域权重，证据在哪些数据和规模成立。
2. loss spike 不是单一故障。Adam 时间相关更新假说与自适应二阶矩失配假说分别解释什么、观察什么，以及为何不能把其中一个当作所有 spike 的答案。
3. 评估污染的“文本重叠”“模型行为异常”和“分数被抬高”是不同判断。PALOMA 的分域、控制变量协议和 PaCoST 的配对置信度检测各自能提供什么证据、有什么盲区。

同时将 DCP 的版本化接口能力与训练语义恢复边界落到 PyTorch 2.14 文档，并检查综合账本代码有没有把评估和风险藏在其他字段中。

## 已阅读来源与证据

| 来源 | 本轮阅读内容 | 支持的判断 | 边界 |
| --- | --- | --- | --- |
| [DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429) | 方法/Group DRO 目标、The Pile 与 GLaM 实验、规模消融、局限 | 先训练 reference，再以 proxy 的域 excess loss 做 Group DRO 得到权重，最终在大模型训练中按该权重重采样；280M→8B Pile 实验报告 5 个 one-shot 任务平均准确率提升 6.5 个百分点、75k 步达到 baseline 200k 步结果，代理/权重优化开销约占大模型计算 8% | Pile 的 22 域、tokenizer、模型、训练配方和 5 个任务绑定；280M 与 1B proxy 得到不同权重，论文指出权重空间可能存在不同局部解；不是数据质量、许可或所有下游任务的自动优化器 |
| [A Theory on Adam Instability in Large-Scale Machine Learning](https://arxiv.org/abs/2304.09871) | 论文的时间相关梯度分析、大模型曲线和讨论/缓解方式 | 论文把某类大 batch Adam spike 与梯度估计的时间相关、更新分量分布变化联系起来；报告从 7B 到 546B 规模观察，并强调没有适用于所有设置的单一补救 | 是一组理论和训练轨迹分析，不等于已证明每个 spike 都由该机制导致；降低 batch 或动量系数会改变吞吐、更新噪声和跟踪性 |
| [Adaptive Preconditioners Trigger Loss Spikes in Adam](https://arxiv.org/abs/2506.04805) | 预条件 Hessian 分析、五阶段机制、语言模型实验和消融 | 提出当当前梯度平方与 EMA 二阶矩脱节时，`v_t` 可能按 `beta_2` 衰减、有效预条件增强，并在特定稳定条件被破坏时触发 loss spike；在约 187M LLaMA 结构模型、SlimPajama 100B tokens 设置下观察 spike，降低 `beta_2` 或增大 `epsilon` 在其试验中抑制 spike | 规模和系统与旗舰生产训练不同；曲率与方向性统计计算并非一般在线训练日志都有；作者机制是重要解释之一，不能单因相关症状推断根因，也不能不经验证直接改 optimizer |
| [PALOMA: A Benchmark for Evaluating Language Model Fit](https://arxiv.org/abs/2312.10523) 与 [NeurIPS 2024 论文页](https://papers.nips.cc/paper_files/paper/2024/hash/760b2d94398aa61468aa3bc11506d9ea-Abstract-Datasets_and_Benchmarks_Track.html) | 论文评估指南、域级结果和评测控制 | 分域评估比单一混合 perplexity 更能暴露模型对不同文本域的拟合差异；PALOMA 论文给出 546 个 English/code 域，并控制去污染、训练数据顺序、分层抽样、词表和文档截断/拼接等评测条件 | 语言模型 fit/perplexity 不等于下游任务能力、安全性或泛化；域数量和数据范围以论文版本为准，不能把其单一数字当成所有 PALOMA release 的固定描述 |
| [PaCoST: Paired Confidence Significance Testing for Benchmark Contamination Detection](https://arxiv.org/abs/2406.18326) | 方法、配对置信度检验、实验与限制 | 将原题和语义保持的改写题配对，比较模型对自身答案的 P(True) 置信度并做 paired t-test，尝试识别 benchmark-level 暴露信号；作者用人为污染的 Llama/Mistral 设置验证方案 | 需要概率分布访问，使用多次模型调用，不能定位具体被污染样本；改写是否等难、置信度是否校准会影响判断；统计显著提示“可能污染”而非证明语料来源或分数因果提升 |
| [PyTorch Distributed Checkpoint 2.14 文档](https://docs.pytorch.org/docs/2.14/distributed.checkpoint) 与 [DCP 2.14 教程](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html) | 并行 save/load、跨拓扑 reshard、文件和状态字典语义 | DCP 支持多 rank 并行保存/加载，并可在加载时按不同集群拓扑重分片；它以已分配的状态对象为目标原地加载，和单文件 `torch.save` 不同 | DCP 的 reshard/API 成功不自动证明 RNG、optimizer、scheduler、数据游标、训练步和应用层 manifest 一致；后者仍须由训练程序保存并做恢复演练 |

## 方案对照与正文落点

### 数据 mixture

DoReMi 要解决的是“手工 mixture 或针对已知下游任务反复训多个比例，成本高且可能对选定任务过拟合”。它以 reference model 为域难度基线，在 proxy 上优化域间 excess loss 的 Group DRO 权重，再把权重用于大模型采样。该方法把代理训练成本换成更快达到论文内 baseline 质量的结果。正文会同时说明：采样权重只是 exposure policy，不自动判断授权、去重、隐私、污染和质量；proxy 尺度变化可能换出另一组域权重，迁移到目标任务要固定总 token 做消融。

### Loss spike

两篇 Adam 研究均提醒不能把 spike 归因于“学习率高”或“坏数据”之一，但侧重机制不同：

| 假说 | 论文指出的内部变化 | 可加入的日志/对照 | 主要限制 |
| --- | --- | --- | --- |
| 时间相关梯度与 Adam 更新退化（Molybog et al., 2023） | 大 batch 条件下梯度估计的时间相关与 `m/sqrt(v)` 更新分量分布变化相关 | 首次异常前的 layer-wise 梯度/更新范数、`m/v` 状态、固定 batch 重放；验证减 batch 或动量设置的对照分支 | 不同运行的机制和恢复过程可能不同；改变 batch/动量也改吞吐及优化动力学 |
| 自适应预条件器失配（Bai et al., 2025） | 某些层当前梯度平方明显小于历史 `v_t`，导致 `v_t` 自身衰减，预条件后的有效步长与曲率条件恶化 | 对照 `g_t^2`、`v_t`、逐层 update norm、`beta_2`/`epsilon` 分支；检查异常发生在 optimizer update 前还是后 | 论文的 187M/100B-token 实验与真实大型训练的规模、数据、内核不同；所需曲率方向指标不能只凭 loss 曲线恢复 |

正文将研究解释转成“候选机制及可证伪实验”，不会把某个日志信号写成充分诊断。修复应从异常前 checkpoint 分叉，一次只改一个变量，比较训练与分层验证结果；如果缺少 optimizer、RNG 或数据游标，恢复后的 run 要记录为语义不完整。

### 评估与污染

PALOMA 的重要贡献在于控制语言模型 fit 评测的分布和口径：对多域分别报告，并避免把 tokenization、样本顺序、文档拼接和子文档重复差异误当作模型变化。这帮助解释“平均 validation loss 下降但代码/领域能力下降”，但并不声称用 PPL 替代能力评估。

污染证据则需要分级：训练语料可访问时，exact/近似匹配找候选重叠；PaCoST 以原题—改写题配对置信度差检验 benchmark 级信号；有条件时再用新鲜、隐藏、可执行或动态任务检验能力。候选重叠不等于模型学会了答案，置信度差不等于找到了来源，benchmark-level 信号也不能说明哪几题受影响。报告应分开覆盖率、候选命中、人工确认、未知项和分数敏感性，不把 unknown 写成零污染。

## 旧稿知识点覆盖图

| 原有问题 | 本轮处理 | 正文位置 |
| --- | --- | --- |
| 7B 训练目标、数据、模型、优化、系统、状态、评估账本 | 保留，补足“计划/证据/实测”的区分 | §1—§3、§15 |
| mixture 比例与有效 token | 保留，增加 DoReMi 的 objective→proxy→resampling→大模型结果及迁移边界 | §2.2 |
| loss spike、坏 batch、NaN、rank 差异、resume | 保留，增加两种代表机制对照、可观测量和固定 checkpoint 反事实排查 | §4 |
| validation loss 与能力分层 | 保留，增加 PALOMA 的多域控制案例及 PPL 范围 | §5.2 |
| benchmark exact/near overlap、改写与隔离 | 保留，区分候选匹配、置信度信号、确认污染和实际分数影响 | §6 |
| batch/LR、并行切分、显存账本、SFT、VLM、成本、Base/Instruct/Chat、DPO/PPO | 保留其章节导航作用；这些主题的机制主讲文件仍是本册前文，综合章按目标、状态、评测与系统边界建立回链 | §7—§14 |
| 综合 Python 账本 | 保留并单独表示目标、数据、模型、优化、系统、评估、风险和证据字段 | §15 |

## 正文修订落点与本地验收

- §2.2 新增 DoReMi 推导链、论文内 280M→8B 案例、代理成本、不同 proxy 尺度会产出不同权重的观察和项目内验证方法。
- §4.5—§4.6 增 Adam 机制对照、二阶矩递推的小数值推演、异常发生阶段分叉和固定 checkpoint 反事实步骤。
- §5.2 加 PALOMA 的多域 fit 与评测控制案例；§6 将污染检测方法按证据功能区分，把 candidate overlap 和人工 confirmed 项分别计数。
- §2.7 加 PyTorch 2.14 DCP 的多 rank 存取、跨拓扑 reshard、原地 state load 和应用训练状态连续性的边界。
- §15 账本把 `evaluation`、`risk` 从其它字段拆开，增加评测 protocol/split isolation、污染/许可/隐私状态门禁；标准输出保持原约定。
- 更新 `PAPERS.md`、`GLOSSARY_EN_ZH.md`、`INTERVIEW_BANK.md` 与 `EXERCISES.md`；第五册目录已有第十四章映射，不需改动。
- 章节 0—19 标题连续，54 个 fenced marker 配对，24 个 `math` block 花括号平衡；22 个 Markdown 链接的本地目标全部存在。
- 教学脚本执行后 6 行 stdout 与正文预期逐字相同；缺少 evaluation/risk 字段和非法污染状态被拒绝，未知 split isolation 进入不完整评估状态。`git diff --check` 通过。

## 尚未验证的事项

- 论文数字只按论文作者的公开设置记述；未复现 DoReMi、Adam spike 或 contamination 检测实验。
- 未在目标集群、目标 checkpoint、目标 benchmark 或厂商黑盒模型上执行排错、污染审计或恢复演练。
- PyTorch 2.14 文档用于说明接口；没有安装或运行 PyTorch 多卡 DCP/FSDP。
- 正文 Python 脚本是资源账本/结构审计 toy，不实现分布式训练、模型质量评估或论文算法复现。
