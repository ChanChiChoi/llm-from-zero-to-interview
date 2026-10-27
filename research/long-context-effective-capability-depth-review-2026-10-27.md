# 第七册第十三章：1M Context 有效能力研究记录

核对日期：2026-10-27  
正文：[13-1m-context有效能力评测.md](../book-07-evaluation-experiments/chapters/13-1m-context有效能力评测.md)  
研究目的：按当前 `WRITING_SPEC.md` 补足代表研究的具体问题、机制、证据和限制；保留章节原有的上下文合同、RoPE、训练、KV/成本、任务设计、有效长度、RAG、线上调度、证据等级和练习主线。

## 1. 旧稿核心知识点承接

本轮没有将原章节重写成 benchmark 清单。以下既有内容继续作为章节主干：

- 界面、计算、训练、任务和生产五种“支持”口径；请求经过网关、tokenizer、模板和 runtime 的完整性审计。
- 长上下文局部/全局任务区别、prefill/decode 与尾延迟；位置表示、RoPE、RoPE scaling、PI/YaRN/LongRoPE、稀疏/混合 attention、context parallelism、FlashAttention、PagedAttention 和状态压缩。
- 长上下文训练的长度覆盖与任务覆盖；KV cache 预算、显存与 prefill 计算成本；单位成功任务成本。
- needle 到多证据、干扰、版本冲突和归因任务的递进；条件化有效长度 `L_eff`、回答/证据/引用/拒答/运行指标。
- 分层实验、RAG 配合、生产服务、故障定位、证据等级、对外声明、demo、练习、模型版本比较和 Qwen2.5-1M 历史阅读示例。

重心新增在原有 13.6、13.9、13.14、13.17 和 13.20：加入方法分层对照、benchmark 研究逐项比较、可推导队列数值、demo 完整信号/动作映射和来源边界。VQA/DocVQA/ChartQA/AgentBench 保留为相邻多模态/Agent 评测入口，但明确它们不构成文本 1M context 直接证据。

## 2. 位置与系统路线：论文各解决什么

### Position Interpolation

[Position Interpolation](https://arxiv.org/abs/2306.15595) 观察到直接外推 RoPE 位置可能产生超出预训练分布的 attention score；方法将长序列位置映射回训练位置范围，再以短时长序列微调适配。论文对 LLaMA 7B—65B 报告 2K→32K、最多 1,000 步，并测 passkey retrieval、语言建模、长文摘要和原窗口任务。它解决的是位置外推不稳定和扩展微调效率，不能提供多证据推理训练，也不能据此保证产品模型的 32K/1M 任务表现。

### YaRN

[YaRN](https://arxiv.org/abs/2309.00071) 将 PI 对所有 RoPE 维度同等压缩的弱点拆开处理：按波长分辨频率维度，在局部相对位置信号保留与长位置插值之间平滑过渡（NTK-by-parts），并对 attention logits 使用 scaling。原文 v3 的作者摘要报告，相较既有扩展方法约使用 1/10 训练 token、1/2.5 训练 steps；同一论文的实验还包含 LLaMA/Llama 2 多尺度上下文和 64K/128K checkpoint。上述倍率绑定其模型、基线、语料和训练/评测 recipe；正文保留“作者报告”，不写成所有长上下文微调的固定成本优势。

### LongRoPE

[LongRoPE](https://arxiv.org/abs/2402.13753) 针对极长上下文中频率维度和 token 位置呈非均匀性的问题，搜索分维、分位置的插值因子；按 256K 长度微调，再对扩展 checkpoint 做二次插值至 2,048K，并在 8K 再调整以恢复短上下文质量。作者在 LLaMA 2/Mistral 上报告不微调最高 8×扩展和分阶段 2,048K 结果。研究的关键贡献是处理极端扩展时的数据稀缺、训练成本与原长度退化；窗口数字仍不等于开放式任务有效长度。

### Sparse attention 与分布式/执行优化

- [Longformer](https://arxiv.org/abs/2004.05150)：局部滑窗加任务相关 global attention，改每个 token 的交互范围。
- [BigBird](https://arxiv.org/abs/2007.14062)：局部、随机和 global token 的稀疏连接，提供不同的信息传播图。
- [Ring Attention](https://arxiv.org/abs/2310.01889)：把序列分块分布到设备，环形交换 K/V block 并重叠通信/计算，缓解单设备内存瓶颈；标准 dense attention 的总两两交互工作仍是二次规模。
- [FlashAttention](https://arxiv.org/abs/2205.14135)：分块、在线 softmax 和 IO-aware 计算减少 HBM 中间读写。
- [PagedAttention](https://arxiv.org/abs/2309.06180)：分页式 KV 管理降低碎片并改善 serving 的状态分配/共享。

读法：PI/YaRN/LongRoPE 是位置表示与训练 recipe；Longformer/BigBird 改变 token 连边；Ring Attention 改变并行放置和通信；FlashAttention/PagedAttention 改变执行/状态管理。它们解决不同层的问题，不能用单一“扩展倍数”对比。稀疏连接降低计算并可能限制证据传播路径；分布式 dense attention 改变每设备负担而不自动减少全序列两两交互；执行优化不等价于任务理解提升。正文将对应收益映射到任务质量、硬件 trace 和成本三类独立验证。

## 3. Benchmark 对照与组合方式

| 工作 | 设计贡献 | 作者证据（按论文/仓库口径） | 主要限制 |
| --- | --- | --- | --- |
| NIAH | 单条可精确匹配 needle，可控制长度和位置。 | 低成本暴露输入截断、位置丢失和显著字符串检索失败。 | 词面捷径强；不能替代多证据、版本消歧、冲突与引用归因。 |
| Lost in the Middle | 在多文档 QA 与 key-value 检索中控制相关信息位置。 | 作者观察到多个模型对中间位置证据显著敏感，提供“同材料只改证据位置”的测量协议。 | 主要是位置探针，不能覆盖所有任务或 1M 长度。 |
| RULER | 可配置 context length/task complexity；增加多 needle、multi-hop tracing、aggregation，共 13 tasks。 | 作者评估 17 个 long-context models；vanilla NIAH 近满分并不阻止长度/复杂度提升后的退化，论文按自身满意阈值报告只有约一半声明 ≥32K 的模型在 32K 达标。 | 合成任务和作者阈值；不代表任意生产任务或通用发布门槛。 |
| NoLiMa | 降低 query-needle 词面重叠，要求通过潜在关联找证据。 | 作者评估 12 个至少声明 128K 的模型；10 个模型在 32K 低于各自短上下文 baseline 一半，GPT-4o 报告从 99.3% 短程 baseline 降至 69.7%。 | 关联检索任务而非一般因果推理；受 needle、提示、模型版本限制。 |
| LongBench v1 | 21 个中英文数据集、六类任务统一评测；英文平均 6,711 words，中文平均 13,386 characters。 | 扩大单 needle 之外的多任务覆盖，并比较位置扩展、长序列微调和检索压缩。 | 任务异质且平均长度不是百万 token；总分会掩盖任务/语言差异。 |
| LongBench v2 | 503 道多选题、六类单/多文档/ICL/对话/代码/结构化数据题；采集近 100 位多领域受教育参与者并经自动/人工审核。 | 上下文长度 8K—2M words；作者报告 15 分钟 human expert 53.7%、直接作答最佳模型 50.1%、o1-preview 长推理 57.7%。 | words != tokens；困难题筛选、选择题形式和推理预算影响分数；不能代表目标业务总体。 |
| LongBench Pro | 1,500 条中英文自然长上下文样本、11 primary/25 secondary tasks，8K—256K tokens；full/partial dependency、六档长度、四档难度；模型起草后专家验证。 | 2026-01 v1 预印本评估 46 个模型，报告有效长度常短于声明并观察跨语言错位等。 | 最高 256K，不能证明 1M；单版本预印本结论待独立复核；人机协作样本构造仍有分布选择。 |
| Google DeepMind MRCR v2 | 重复相似请求/回答的合成多轮对话，要求按目标请求序号返回特定回答；控制相对任务复杂度并提高长度。 | GitHub README 在核对日称公开长度曲线数据可至 8M；8-needle 对随机复制某个相关回答的噪声估计约 15%。 | 特意保持任务简单，适合长程定位/共指而不是多文档推理；工具会显著改变难度；README 是滚动 revision。 |
| OpenAI MRCR / Graphwalks | OpenAI 官方发布说明分别描述多轮合成共指与结构化图 BFS 多跳任务。 | 作为厂商公开任务和方法入口，可帮助区分“找第 i 个相似回答”和跨位置图遍历。 | 发布说明中的模型表现是供应方报告；与 Google DeepMind MRCR v2 在任务、数据和计分上不等同，不拼接分数。 |

由此得到的组合路径是：NIAH 做输入/位置冒烟；Lost in the Middle 做位置受控实验；RULER/NoLiMa 隔离复杂度和词面捷径；LongBench 系列补任务分布与自然样本；MRCR v2 提供超长范围的固定复杂度曲线；最终再做目标领域的多证据、版本冲突、不可回答和引用归因样本。对于 LongBench v2、LongBench Pro 与 MRCR v2，要把长度单位和来源版本写进 manifest；不同 benchmark、模型 API、推理预算和工具条件不能合成同一排行。

## 4. 队列推导复核

采用 M/G/1 FCFS Pollaczek–Khinchine 平均排队时长：`u = λE[S] < 1`，`E[Wq] = λE[S²]/(2(1-u))`。参考 [Stanford EE384X M/G/1 讲义](https://web.stanford.edu/class/ee384x/EE384X/handouts/H10.pdf)。

- 长短混合：`λ=2/s`；短服务时间 `0.1s`，概率 `0.98`；长服务时间 `10s`，概率 `0.02`。`E[S]=0.98×0.1+0.02×10=0.298s`；`u=0.596`；`E[S²]=0.98×0.1²+0.02×10²=2.0098s²`；`E[Wq]=2×2.0098/[2×(1−0.596)]≈4.9748s`。
- 全短对照：`E[S]=0.1s`，`u=0.2`，`E[S²]=0.01s²`，`E[Wq]=2×0.01/[2×0.8]=0.0125s`。

这说明服务时间二阶矩和到达利用率会共同影响均值等待。它只是在 Poisson、单服务台、FCFS、独立服务时间下的理论示例，不是 GPU continuous batching、chunked prefill 或 P/D 分离的线上模型；不把 4.97 秒外推成部署延迟。

## 5. Demo 静态复核

章节 demo 输入保持为 32K/256K/1M 三个 toy 桶。按首末桶差值 `0.20` 手算：`evidence_degrades=True`（0.55 < 0.94−0.20）、`answer_degrades=True`（0.39 < 0.88−0.20）、`citation_degrades=True`（0.31 < 0.91−0.20）；所有 ingest=1.00，故 `ingest_unstable=False`；P95 48 > 20，`interactive_slo_miss=True`；cost 9.6 > 4，`cost_budget_miss=True`。预期 primary decision 是 `route_long_tasks_to_async`，同时保留三项质量动作。

代码静态结构现在覆盖每个失败类：输入不稳定触发核对 tokenizer/gateway/runtime 并重跑；输入稳定后 evidence/answer/citation 各自触发诊断；SLO 与成本异常各有路由/优化动作。决策优先级是先修输入，再满足运行预算，最后处理质量主路径。该 demo 仍以首末点差作 toy 阈值，不估中间趋势、置信区间、统计显著性或生产风险；没有执行 Python。

## 6. 配套与验收

- `PAPERS.md` 长上下文部分增加第七册第十三章入口，并按机制层拆分 PI/YaRN/LongRoPE、稀疏注意力、Ring、FlashAttention/PagedAttention 与 benchmark 系列。
- `EXERCISES.md` 增加 6J：围绕 1M 研究助手组合评测并固定长度单位、工具和运行预算。
- `INTERVIEW_BANK.md` 增加 49F：解释 RULER、NoLiMa、LongBench v2/Pro、MRCR v2 各自补上的证据缺口。
- `GLOSSARY_EN_ZH.md` 增 NoLiMa、LongBench、LongBench Pro、MRCR v2 和有效上下文长度的操作性定义。
- 第七册目录原已列本章；无需更改章节导航。

本轮静态复核章节一级标题顺序、代码 fenced block 配对、公式符号、预期输出和本地索引链接；未运行 Python、测试套件、外部公式渲染、完整外链状态扫描或 PDF 构建。公开论文结论绑定其作者模型、benchmark、提示和预算；DeepMind README 是滚动文档；LongBench Pro 是预印本；无目标模型、GPU、真实语料、线上 trace 或业务标注，不能得出目标系统有效 1M 能力或生产容量结论。
