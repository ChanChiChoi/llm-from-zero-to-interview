# Error Analysis 与回归测试深度精修研究记录

- 日期：2026-10-27
- 正文：[第七册第十一章](../book-07-evaluation-experiments/chapters/11-error-analysis与回归测试.md)
- 依据：当前 [WRITING_SPEC.md](../WRITING_SPEC.md) 与 [progress_v2.md](../progress_v2.md)
- 范围：原章知识承接、错误 taxonomy、根因验证、behavioral testing、回归集、代表性评测研究、样本级版本 diff 和示例代码复核。
- 执行边界：打开 ACL Anthology、arXiv、NeurIPS、IEEE/Google Research 和 OpenAI 官方文档核对方法；未运行 Python、测试套件、外部 benchmark、目标模型/API 或生产工作流。

## 1. 原章节知识承接

原章的 17 个一级主题全部保留：从总体分数和误差向量、失败证据与 taxonomy，经过人工/LLM 标注、聚类和主动挖掘、根因分析、regression suite、版本差异、修复复验和自动化闭环，再到 toy 审计、常见坑、适用边界、benchmark 扩展、练习和资料。没有因补充研究而删除旧主题。

| 原知识点 | 本轮处理位置 | 承接和深化 |
| --- | --- | --- |
| 汇总分数会掩盖重要切片，错误频次与严重度不同 | §1 | 保留误差向量、加权损失和总体分数的业务局限 |
| trace/输入/标签/模型和 harness 版本要可复核 | §2 | 保留失败、无效、评估器分歧、缺证据和隐私边界 |
| taxonomy 区分症状、能力、根因假设与动作 | §3 | 保留通用、RAG、Agent、代码/数学分层和多标签统计 |
| 人工/LLM judge 校准与分歧 | §4 | 保留标注单位、反例指南、kappa 和分歧归因，并与第七册第四章互补 |
| 手工/embedding 聚类、LLM 辅助归因、主动挖掘 | §5 | 保留频次、严重度、可修复性；明确主动样本不估总体率 |
| Five Whys、近因/根因和 Model/Harness/Protocol/Evidence | §6 | 全部保留；新增 §6.7，用固定组件、oracle evidence 对照和 holdout 把假设转成可区分的调试实验 |
| benchmark/golden/regression/exploration 的区别，case schema、oracle、维护、去污染和覆盖 | §7 | 全部保留；补强覆盖率的解释边界，新增 §7.9 对照代表性方案的研究问题、机制和限制 |
| 版本账本、四象限、切片 diff、不确定性 | §8 | 保留；配对统计方法承接第十章，不重复展开推断理论 |
| 数据、prompt、RAG/工具、训练、judge/harness 修复与确认 | §9 | 保留 targeted intervention → holdout → regression → fresh exploration 链 |
| 自动化、人工边界和线上反馈入集 | §10 | 保留全流程和敏感/高风险人工处理 |
| toy 回归审计代码、优先级、覆盖和发布信号 | §11 | 保留 10 个样本及输出逻辑；把根因计数更名为候选根因假设计数，加入静态手算的关键输出和阈值边界 |
| 常见坑、方法边界、从 case 扩展为能力评估、练习 | §12—§15 | 全部保留；无根因分析被简化成 taxonomy 标签 |
| 资料与结语 | §16—§17 | 更正 CheckList 入口，增加 EvalPlus 和 trace grading 来源，并解释各来源承担的评测层次 |

## 2. 代表性方法分别解决什么问题

| 来源 | 解决的问题 / 方法 | 本章使用方式 | 证据边界 |
| --- | --- | --- | --- |
| Ribeiro et al., [CheckList](https://aclanthology.org/2020.acl-main.442/), ACL 2020 | held-out accuracy 可能忽略具体行为；用能力 × 测试类型矩阵组织 Minimum Functionality Tests、Invariance Tests 和 Directional Expectation Tests，并以模板扩展样例。论文用户研究中，使用 CheckList 的 NLP 从业者构造了约两倍测试，发现接近三倍 bug；商业情感分析团队也发现之前测试未揭示的可行动问题。 | §7.9 将其用于定向行为测试，再把经确认的行为缺陷沉淀成回归 case。 | 用户研究数量不是所有团队的保证；效果依赖能力选择、扰动语义、预期关系和标注质量。它暴露行为，不自动定位线上系统组件根因。 |
| Breck et al., [The ML Test Score](https://doi.org/10.1109/BigData.2017.8258038), IEEE Big Data 2017 | 把生产 ML 常见的数据、模型、基础设施和监控风险整理为 28 项测试/监控要求和 readiness rubric；论文说明其经验来自生产系统实践并访谈 36 个系统维护者。 | §7.9 用来提醒读者，回归集之外还有数据血缘、切片质量、训练/服务 skew、监控和回滚等系统防线。 | 论文面向 2017 年生产 ML，未定义开放生成的语义 oracle、Agent trace 或今天的 judge calibration；其 rubric 不是 LLM 产品质量评分。 |
| Liang et al., [HELM](https://arxiv.org/abs/2211.09110) | 解决不同模型测量场景不重叠、只看 accuracy 隐藏 trade-off 的问题；原论文以 16 个核心场景和 7 类指标构成公共矩阵，发布 prompts/completions。 | §7.9 作为较广 benchmark/风险覆盖与横向比较层，帮助发现业务回归集之外的能力变化。 | 场景选择受可行性和覆盖取舍限制；公共 benchmark 不含目标组织的分布，也不提供单条私有 trace 的根因。论文数字绑定原研究的模型与场景。 |
| Liu et al., [EvalPlus / HumanEval+](https://papers.neurips.cc/paper_files/paper/2023/hash/43e9d647ccd3e4b7b5baab53f0368686-Abstract-Conference.html), NeurIPS 2023 | 原有代码 benchmark 手工 test 数量有限；用 LLM 生成语义有效 seed、类型感知变异扩展输入、程序 contract 过滤无效输入，再用 differential testing 对照 reference implementation。HumanEval 每题平均测试数由 9.6 增至 774.8；论文还报告原始 canonical solutions 中超过 10% 有缺陷。 | §7.9 用作“加强可执行 oracle”的代码任务案例，并提醒 reference 本身也可能出错。 | 生成/变异输入仍可能漏掉路径或误解 contract；通过有限测试不等于对全部合法输入证明正确。数字绑定 HumanEval 任务、参考实现和论文配置。 |
| OpenAI [Trace Grading](https://developers.openai.com/api/docs/guides/trace-grading) | 官方产品文档介绍对 Agent end-to-end trace 做结构化评分，以便查看模型调用、工具调用、handoff 和 workflow 行为，补足只看最终答案的黑盒结果。 | §7.9 与 §2、§9、§10 的 Agent trace/归因路径相连；trace grading 发现候选失效阶段后，仍要人工/程序确认并转为稳定 case。 | 这是厂商文档，不是独立效能研究；trace completeness 不自动说明 grader 正确，LLM grader 需要校准；平台/API 状态要按使用时版本复核。 |

这些来源不能合成为一个质量分数。一个有用的分层是：HELM/benchmark 提示“哪些能力或场景变化”；CheckList 测试“某个行为关系是否成立”；trace 记录“Agent 工作流在哪里表现异常”；ML Test Score 检查“生产系统防线是否齐全”；EvalPlus 说明“程序型任务的 oracle 如何通过更多有效输入变强”。产品级回归集只收录重要、可复现、判定可审计的 case，并和新样本、holdout、线上监控并行。

## 3. 根因诊断新增内容

原章已有“症状—假设—根因—动作”区分和通用组件旁路列表，但还缺一个完整例子说明如何依据干预结果区分竞争假设。§6.7 用年份约束 RAG 回答展示了这条路径：

1. 固定模型、prompt、解码、权限、索引快照，复现原请求并核对证据/标签。
2. 仅将生产检索出的上下文替换为核验后的 2023 年证据，判断失败是否可被 evidence intervention 救回。
3. 读取 top-k、ACL 过滤、rerank、context packing 和截断状态，细分 evidence path 的失败阶段。
4. 用未见公司/年份检验修复，联合观察证据召回、最终答案和引用支持。

“50 条失败 trace 中 31 条在 oracle context 下修正”是教学算例。它表示被选中失败样本中对 evidence replacement 的响应，不是总体 lift、线上 accuracy 或对 production retriever 单一根因的证明。章节在段落中明确说明了抽样条件和 oracle context 的外推边界。

## 4. demo 静态核算

示例的 10 个 case 手算得到：稳定正确 1、修复 2、回归 4、仍未解决 3；回归 id 为 `json_schema`、`safety_refusal`、`citation`、`privacy`；修复 id 为 `rag_finance`、`tool_call`。回归集覆盖 6 项预设能力中的 4 项，`coverage=4/6≈0.667`，缺 `reasoning` 和 `tool`。

候选根因映射计数为：`prompt_schema=1`、`policy_boundary=2`、`reasoning_data=1`、`rag_grounding=1`、`rag_retrieval=1`、`code_tests=1`。映射仅从 `error` 标签机械得到，没有读取 trace 或做干预，因此研究记录和代码输出都称为 `root_hypothesis_counts`，不称为 root-cause counts。教学动作包括 review high severity、恢复安全/隐私行为、补能力覆盖及扩大流量前重测；`0.80` 覆盖门槛和 hold 规则仅为示例 policy。

priority 手算次序为 `privacy 3.5`、`citation_error 3.2`、`safety 3.0`、`retrieval_miss 2.8`、`formatting 2.7`、`edge_case 2.4`、`reasoning 2.0`。这一乘积只用于当前 toy case 排序，不能让低风险修复抵消高严重度隐私/安全回归。

## 5. 配套同步与检查

- `PAPERS.md`：第七册正文入口增加第十一章；补 CheckList、ML Test Score、HELM、EvalPlus、Trace Grading 的方法层次和边界，并链接本研究记录。
- `EXERCISES.md`：增加 87F 根因干预与 oracle context 外推题、87G 代表性方法比较题。
- `INTERVIEW_BANK.md`：增加 5J RAG 根因定位和 5K 方法分工面试题。
- `GLOSSARY_EN_ZH.md`：在回归术语旁补 Error Analysis、Root Cause Hypothesis、MFT、INV、DIR；既有 Regression Suite、Metamorphic Testing 保持不动。
- 第七册 `目录.md` 已有第十一章入口，不需改动。
- 只手算 demo 关键输出；没有执行 Python，也没有将手算写成运行结果。
- 本轮引用链接逐项打开了 CheckList ACL 页面、ML Test Score Google Research PDF、HELM arXiv 页面、EvalPlus arXiv/NeurIPS 页面、OpenAI 官方 trace grading 文档；未对全章所有旧链接做 HTTP 状态扫描、未做外部公式渲染或 PDF 构建。

## 6. 未验证事项

没有目标业务 trace、抽样框、专家标签、人工 judge 校准结果、生产检索器或 Agent executor。根因实验、风险权重、模型变化和覆盖阈值都是教学设计；论文数值仅适用于作者的模型、样本、任务、prompt、oracle 和 harness。`compare_versions` 仍显式跳过未对齐 case，因此实际系统需单独报告 old/new 对齐率、缺失 case 和筛选对目标分布的影响。OpenAI hosted eval/tracing 产品与文档会变化，应在部署前复核当前生命周期和版本。
