# 第七册第四章：LLM-as-a-Judge 资料驱动精修记录

## 范围与保留的知识点

本轮依据当前 [`WRITING_SPEC.md`](../WRITING_SPEC.md)，复读并精修
[`04-llm-as-a-judge.md`](../book-07-evaluation-experiments/chapters/04-llm-as-a-judge.md)
§1—§19。保留原有测量对象与任务契约、judge 形式、prompt 与 rubric、结构化
输出、偏见和攻击面、人工 gold 校准、位置交换、统计稳定性、多 judge 路由、RAG
案例、诊断 demo、生产 pipeline、Human Eval 衔接、失败模式、练习和证据边界；
本轮是在现有主线上补机制、研究比较和范围约束，没有删去原有核心主题。

## 研究问题

1. Judge 的分数在什么意义上是测量结果，而非样本固有属性？
2. 代表性方案分别针对 prompt 设计、专用 evaluator 能力和人工量尺校准中的什么缺口？
3. 一致率、模型排序、绝对分数和自动放行为什么需要不同的验收指标？
4. 顺序交换能检验什么，又有哪些答案敏感性与 rubric 敏感性需要单独测试？
5. 论文里的数字可以支持到什么范围，哪些仍需目标任务的人工 holdout？

## 主要研究来源与证据范围

| 来源 | 问题与方法 | 可支持的结论 | 不能外推的部分 |
| --- | --- | --- | --- |
| [G-Eval, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.153/) | 为 NLG 任务生成具体评价步骤，使用表单打分，并按评分 token 概率加权 | SummEval 上 GPT-4 离散评分版 Spearman ρ=0.514；概率加权版 ρ=0.502、Kendall τ=0.446（离散版 τ=0.418）。论文提醒离散版产生较多并列分数，其 Kendall τ 不能作为公平的直接比较 | 不是代码、RAG 或安全任务上的普遍有效性；概率加权的小数分仍不是完整评分分布或绝对分数校准；API token 概率的可用性和调用成本依版本变化 |
| [Chiang & Lee, Findings EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.599/) | 对 CoT、数字输出和评分说明做自动评估组件分析 | 自动 CoT 不总能提高人类一致性；仅输出数值会损失部分信息；让模型解释评分在作者的两个 meta-eval 数据集上改善相关性 | 评分理由可作为测量输入，不证明理由忠实表达模型决策过程，也不能替代人工 gold 和反事实测试 |
| [Prometheus 2, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.248/) | 分别训练 direct-assessment 和 pairwise evaluator，再合并权重；允许使用自定义标准 | 在四个 direct-assessment 与四个 pairwise 基准上，该论文测试的开源 evaluator 中相关或一致表现最佳 | 论文基准、英语数据和训练配方有限；和人工或商业 judge 的比较仍是近似；合并权重的优势是实验结果，不是普适定理 |
| [Shi et al., position bias study](https://arxiv.org/abs/2406.07791) | 用重复、顺序交换和多候选比较测量位置稳定性、位置一致性与偏好公平性 | arXiv v9 报告 15 个 judge、MT-Bench 与 DevBench 的 22 类任务、约 40 个候选模型和超过 150,000 次评估；偏差随 judge/任务变化，答案质量差距影响较强，prompt 组成长度影响较弱 | 作者数据、候选和提示协议特定；不能据此认定目标 judge 的位置偏差大小，也不能把“长度影响较弱”改写成“长度没有影响” |
| [Thakur et al., GEM 2025](https://aclanthology.org/2025.gem-1.33/) | 在 TriviaQA 上比较 13 个 judge 与 9 个作答模型，并检查与人工标签/分数的一致 | 百分比一致可能掩盖 10–20 分的分数差；Scott's π 较高的模型与人工分数仍可相差最高约 5 分；低成本 judge 仍可能给出有用排序信号 | 受控 TriviaQA/候选模型设置；不能把论文误差值当作目标任务误差，也不能用一个一致率替代任务所需的分数校准或高风险漏报指标 |
| [A Survey on LLM-as-a-Judge](https://arxiv.org/abs/2411.15594) | 汇总 in-context 方法、模型选择、后处理、评估流程、偏差与鲁棒性 | 作为概念地图帮助找检索入口和缺项 | 综述 taxonomy 不替代原论文对具体机制和数字的证明 |
| [SAJA, ACL Industry Track 2026](https://aclanthology.org/2026.acl-industry.45/) | 一次固定结构化 rubric 调用提取多维分数，再以少量人工标签训练轻量校准头 | 通常 100–500 条人工标签；论文报告 MT-Bench pairwise F1 0.78→0.86、SummEval 多维特征 Spearman ρ 0.60→0.74；MT-Bench confidence triage 以 44% 覆盖率达到 99.6% 准确率 | 数据拆分、候选 judge、标注分布和基准特定；不表示少量标签在任意领域足够，也不允许把选择性自动化准确率当全量准确率 |
| [Bagaria et al., EMNLP 2026](https://aclanthology.org/2026.emnlp-main.2612/)；[公开预印本](https://arxiv.org/abs/2609.02942) | 训练 rubric-only classifier，并做答案与评价标准的配对反事实 | HealthBench 等数据中 rubric 文本可预测部分 judge 标签；在 493 对经检查的答案反事实里，Qwen2.5-7B-Instruct judge 的预期标签翻转率为 37.7%；评价标准反转的翻转率按 Gemma/LLaMA 候选族分别为 16.8%/32.2% | 医疗/研究 rubric、指定 judge 和合成且人工定性检查的配对样本；可预测性不证明 judge 忽略回答，结果不是所有 judge 的失败率 |
| [LangSmith evaluation types](https://docs.langchain.com/langsmith/evaluation-types)；[OpenAI Graders API](https://developers.openai.com/api/reference/resources/graders) | 当前工程文档展示 offline/online 评估、LLM/code/composite evaluator 和 grader API 接口 | 可用于核对公开接口、字段和评估工作流 | 动态文档只说明公开产品接口，不证明目标 judge 的效度、人工一致或生产质量 |

正文数字以对应论文实验定义为准。研究笔记保留论文版本、对象和限制；综述与
官方文档用于发现框架和接口，不单独支撑模型质量结论。

## 代表性路线如何互补

| 路线 | 主要解决的问题 | 关键机制 | 仍需的补强 |
| --- | --- | --- | --- |
| G-Eval | 自然语言 rubric 太抽象，直接打整数分时分数离散、容易同分 | 任务标准生成成具体评价步骤；按评分 token 概率计算加权分 | 给目标任务做独立人工 meta-eval；检查概率接口/重复采样成本；验证理由与目标证据对应 |
| Prometheus 2 | 通用开源 evaluator 不易同时做绝对评分和 pairwise，也难注入自定义标准 | 两类数据分别训练 evaluator，再合并权重；使用 criterion、reference 和 verbal feedback | 检查领域/语言适配与输出尺度；目标任务单独留出校准集；检查训练数据重叠和 judge 共享偏差 |
| SAJA | 复杂 prompt 搜索、多次 API 调用和 logits 依赖不适合闭源 API 部署 | 单次调用提取结构化多维特征，小型校准器映射到人工标签 | 人工标签要代表目标分布；校准与 holdout 分开；监控 rubric/模型/数据漂移；报告 coverage-risk 曲线 |
| Rubric/答案反事实审计 | 仅看相关性或顺序稳定，可能漏掉 judge 对标准/答案不敏感 | 固定上下文，只改答案证据或 rubric 方向，检查输出是否按预期变化；可加 rubric-only baseline | 反事实要由领域人员检查为最小对照；结果只支持构造样本范围；不要从模型理由推断因果机制 |

这几条路线回答的是不同问题，不能用各自的 headline score 排成通用排行榜：
prompt 方法改变当前评分流程，专用 judge 改变被评估器，校准头改变输出到人工
量尺的映射，反事实审计检查输入发生明确变化时 evaluator 是否相应响应。
最终选型还要看是否有人工标签、API 是否开放 token 概率、是否需要复现、每条
评估成本、目标语言/风险与输出用途。

## 关键计算和正文落点

- G-Eval 分数按照允许评分 token 的概率质量加权：
  `score = Σ_k p(s_k) × s_k`。例如评分 3、4 的概率质量为 0.6、0.4，示例得分是 3.4；这是概率加权示例，不表示该概率等于人类准确率。论文的 SummEval 表中，离散版 Spearman ρ=0.514、Kendall τ=0.418，概率加权版分别为 0.502 和 0.446；论文提醒离散版的 τ 受大量并列分数影响，不宜直接做公平比较。不能笼统说概率加权让所有相关指标变好。
- 校准头学习 `ŷ=fθ(x)`，其中 `x` 是 rubric 各维度的 judge 分数，`y` 是人工标签。训练、调参、概率校准和最终 holdout 应分开；校准不能修复缺失证据或偏斜 gold set。
- 评估指标按决策区分：标签复现看混淆矩阵/少数类召回；模型排序看 Spearman、Kendall 或 pairwise 指标；分数替代看 MAE/RMSE、系统偏差和分层校准；自动放行另看高风险 FNR 与 coverage-risk。单一相关系数或一致率不覆盖这些目标。
- 顺序交换、答案反事实和标准反事实分别测位置不变性、候选证据敏感性与标准敏感性。答案与 rubric 反事实需检查只改变目标变量，不能把生成样本中的长度、风格等共变因素误归因于 judge。
- §12 教学 demo 的七个样本和阈值为合成输入。静态逐行核对显示：14 次输出均有效；judge 新模型偏好率 `6/7≈0.857`、人工折半偏好率 `4.5/7≈0.643`、差 `≈0.214`；一致率 `5/7≈0.714`；原顺序与交换均映射回同一候选，位置 A/B 各 7 胜；更长回答获胜 `7/7=1.0`；高风险 RAG 分歧进入专家复核。此处为源码检查和手算，不是执行输出验证。

## 本轮正文与配套同步

- §3.6 比较 G-Eval、Prometheus 2、SAJA 各自解决的问题、机制、指标和边界；补充 G-Eval token 概率加权式与 Prometheus 2 数据/训练分路。
- §4.4 加入 Chiang 与 Lee 对 G-Eval CoT、数字输出与评分说明的对照结果，并限制理由文本的解释力。
- §7.8 补 rubric-only baseline 与最小答案/标准反事实设计；加入论文的 493 对、37.7%、16.8%/32.2% 结果与数据边界。
- §8.7 区分类别标签一致、排序能力、分数校准和自动放行；以 Thakur 等人的结果说明这些量不能互相替代。
- §9.5 按 arXiv v9 更新位置偏差研究规模和长度影响表述：15 个 judge、22 类任务、约 40 个候选模型、超过 150,000 次评估；提示组件长度影响较弱，质量差距影响较强。
- §18 更新为 ACL Anthology 论文、综述与当前官方评估文档，并保留 HELM、OpenAI Evals、Hugging Face Evaluate 等原有入口。
- `EXERCISES.md` 增加路线选择与 holdout/反事实审计题；`INTERVIEW_BANK.md` 增加方案比较追问；`PAPERS.md` 增加核心论文与证据限制。

## 代码、链接与未完成验证

- §12 Python demo 本轮只做源码静态复读和手算，没有执行 Python；章节已将输出标记为按代码逻辑手工核对的预期值，未声称本轮运行通过。补充了上游 `valid` 标记的语义、无有效 pair/无非平局时返回 `None`，并确保 rubric、解析率、交换稳定性和位置偏差等每项门禁失败都会产生对应 action。七样本的分子、分母、标签映射和展示输出均按代码路径人工核对。
- 公式中评分期望、概率校准头和 demo 汇总按定义手算；没有使用外部公式渲染器。
- ACL Anthology 论文页面/论文 PDF、arXiv 论文和当前 LangSmith/OpenAI API 文档已核对；框架/API 文档会变，部署时需再按 release 检查。
- 没有目标业务请求、目标语言标注、专家 gold、候选模型版本或线上流量。所有论文数字与合成 demo 都不能证明目标任务的 judge 精度、位置/长度偏差或生产放行可靠性。
