# 长上下文评测与有效能力：来源证据

检索与阅读日期：2026-11-03。本记录保存论文和官方 benchmark 文档支持的任务设计、版本、数字与证据边界；书册审阅决定与去向只写在 `progress/`。

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

## LongBench v1 与 v2：任务范围和长度单位

- **LongBench v1 来源与版本：** Bai et al., [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508v2)，arXiv v2，2024-06-19。摘要和任务构造段落报告 21 个数据集、六类任务、英语平均 6,711 words、中文平均 13,386 characters；任务涵盖单/多文档 QA、摘要、few-shot、合成任务和代码补全。长度均值不是百万 token 指标，汇总也不消除任务异质性。
- **LongBench v2 来源与版本：** Bai et al., [LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks](https://arxiv.org/abs/2412.15204v2)，arXiv v2，2025-01-03。摘要和数据统计报告 503 道高难多选题，覆盖六类任务，长度为 8K—2M **words**；近 100 位参与者参与采集，论文报告 15 分钟时限的人类专家准确率 53.7%、直接作答最佳模型 50.1%、o1-preview 较长推理 57.7%。
- **证据边界：** 两版本均有各自固定的任务、语言、评测器和模型集合；LongBench v2 的长度单位是 words，不得改写成 tokens。人工难题分数不代表普通业务样本总体。

## Google DeepMind MRCR v2：固定共指任务的长度曲线

- **来源与版本：** Google DeepMind [`eval_hub/eval_hub/mrcr_v2/README.md`](https://github.com/google-deepmind/eval_hub/blob/master/eval_hub/mrcr_v2/README.md)，`master` 分支，2026-11-03 阅读；仓库会滚动变化，引用或复现实验应记录 commit。
- **来源支持的事实：** 多轮对话重复相近请求、每次回答不同，评测要求数出目标实例并复现对应回答。仓库当时说明会报告 8-needle 的 upto_128K cumulative 与 at_1M pointwise 变体，生成尺度可达 8M；在其随机复制噪声模型下，2/4/8 needles 的噪声估计约为 51%/27%/15%。README 还指出若给予代码工具，任务会明显更简单，比较时必须声明工具条件。
- **证据边界：** 这些噪声值取决于 README 明示的随机猜测模型；任务的推理复杂度保持较低，主要测长程共指/定位，不能代表一般多证据推理。`master` 代码、数据、metric 和官方报告变体可能更新。
