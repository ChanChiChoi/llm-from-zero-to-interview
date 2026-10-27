# 第七册第二章：Benchmark 设计资料驱动复核

核验日期：2026-10-27。依据 `WRITING_SPEC.md` 逐行复读 `book-07-evaluation-experiments/chapters/02-benchmark设计.md` §1—§16。旧章已包含任务契约、数据血缘、采样、难度、标注、判定器、指标、污染防护、版本、统计和企业 RAG 案例；本轮保留整条结构，补充代表性 benchmark 的设计机制、分层抽样后的目标分布估计、项目分析统计边界、指标的空分母处理、运行样例的失败门禁，并复核 OpenAI Evals 当前平台生命周期。

## 研究问题

1. benchmark 是怎样从业务目标转成可验证构念、样本、判定器和行动决策的？
2. 专家题、可执行约束题、滚动新题和协议扰动分别解决什么问题？
3. 总体候选模型得分跨度与题目区分度有何区别，IRT 在 LLM benchmark 上有哪些假设风险？
4. 哪些污染结论来自数据重合，哪些只是测试集在新题上的行为迁移信号？
5. 当前章节 demo 是否把预计算相似度误写成真实污染检测，或存在 gate 没有生成动作的遗漏？
6. 如何区分 benchmark 样本比例和目标总体比例；指标在没有有效分母时应如何报告？

## 来源与证据映射

| 来源 | 阅读/核验内容 | 支持的判断 | 不支持的判断 |
| --- | --- | --- | --- |
| [GPQA](https://arxiv.org/abs/2311.12022) | 阅读数据集构造、筛选和结果表：546 题 Extended、448 题 main、198 题 Diamond；Extended 专家/非专家原始正确率 64.8%/34.1%，main 为 71.9%/30.4% 且受筛选影响；作者另估计约 74% 专家可确认客观性 | 四阶段专家/非专家验证检查题目是否足够难且有可确认答案；区分候选池与筛选子集可以暴露选择效应 | “Google-proof”意味着训练集无污染；main/Diamond 的验证准确率是未受筛选的自然总体估计；三门学科代表所有领域 |
| [IFEval](https://arxiv.org/abs/2311.07911) | 正文报告 25 类、541 prompts；prompt-level / instruction-level 与 strict / loose 四种指标；loose 会删除 Markdown 强调符号、首/末行等变体，论文指出它减少部分 false negative、也会引入 false positive | 规则评分把可形式化指令变成可重复判定，同时区分逐约束和整题通过 | 程序通过等同于答案有帮助、事实正确或语义上满足复杂用户目标；strict/loose 可视为同一个精确量尺 |
| [LiveBench v2](https://arxiv.org/abs/2406.19314v2) | arXiv v2 于 2025-04-18 修订；近期来源、客观答案和滚动更新；论文描述月度新增/更新。另查看[项目主页](https://livebench.github.io/)：本轮访问显示标注的最新快照为 2026-01-08，并写有约每六个月完整刷新策略 | 新题和客观答案降低静态公开题过时和主观 judge 的部分风险；论文方案与发布主页可作为不同版本的生命周期材料 | 永久“免污染”；月度论文策略与当前网站所示刷新计划可不标版本地混写；不同快照可直接当作同一固定试卷 |
| [When Benchmarks are Targets v2](https://arxiv.org/abs/2402.01781v2) | ACL 2024 camera-ready；摘要报告多选选项顺序和答案选择方法扰动可使该研究设置中模型排名变化最多 8 位，并建议 hybrid scoring | 评分协议是测量对象的一部分；微小扰动可揭示某些榜单名次不稳定，目标 benchmark 应保存主协议并做敏感性分析 | 论文报告的名次变化幅度适用于所有模型、所有基准或当前 leaderboard；任意扰动都保持任务构念不变 |
| [Can We Trust Item Response Theory for AI Evaluation?](https://arxiv.org/abs/2607.15190) | 2026 预印本摘要：benchmark 模型数较少、能力分布可能偏斜/聚集，与传统教育心理测量中被假设的数据结构不同；分析估计方式的计算与推断可靠性 | IRT/潜在能力参数应用到 AI model response matrix 时要核查模型集合、拟合诊断和样本条件；参数可作诊断但需谨慎解释 | IRT 对所有模型集都不可靠，或该预印本已形成全领域共识；这是仍需独立复核的方法证据 |
| [ONS sample design and estimation](https://www.ons.gov.uk/methodology/methodologytopicsandstatisticalconcepts/sampledesignandestimation) | 英国国家统计局说明抽样设计、总体估计/加权与抽样方差；复杂抽样需要对应设计下的精度估计 | benchmark 刻意按切片过采样时，应区分样本构成与目标总体权重，并把不确定性和抽样计划一起报告 | ONS 的人口/经济调查权重可直接复制到任意 LLM benchmark；权重能修正切片比例以外的覆盖或选择偏差 |
| [Cohen kappa](https://doi.org/10.1177/001316446002000104)、[Artstein & Poesio inter-coder agreement review](https://aclanthology.org/J08-4004/) 与 [scikit-learn classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) | 查阅 kappa 的观察/偶然一致定义、计算语言学一致性测量比较及分类指标库的 `zero_division` 选项 | 一致性系数和空分母约定都依赖标注设计/评分协议，必须显式写出定义、分子分母和边缘情况 | 某一个系数或库默认值适用于所有 judge、标签空间和评测任务；零分母时用 0 可解释为性能失败 |
| [HELM](https://arxiv.org/abs/2211.09110) 与 [lm-eval 复现经验](https://arxiv.org/abs/2405.14782) | 前者用共同场景和多指标组织覆盖，后者由 harness 维护者总结评测设置敏感性、比较和复现问题；lm-eval 另有[官方实现](https://github.com/EleutherAI/lm-evaluation-harness) | benchmark 设计需要区分“测量覆盖”与“运行协议”；多个任务复用同一 harness 仍需固定 model/task/prompt/runner revision | 框架本身能让任务代表产品，或能消除所有不公平和污染 |
| [OpenAI evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices) | 官方文档把工程 eval 拆成目标、数据、指标、运行比较和持续评测 | 一种厂商提出的持续评测开发循环 | 独立研究结论、唯一评估流程或通用工具优选 |
| [OpenAI API deprecations](https://developers.openai.com/api/docs/deprecations) 和[迁移指南](https://developers.openai.com/cookbook/examples/evaluation/moving-from-openai-evals-to-promptfoo) | 截至 2026-10-27，托管 Evals 平台计划 2026-10-31 起只读、2026-11-30 关闭；提供迁移至 Promptfoo 的路径 | 本章不把 hosted Evals 视为长期稳定依赖，建议让评估数据、grader 和结果可迁移 | 开源 `github.com/openai/evals` 仓库已被同时弃用，或 Promptfoo 对每种团队都是最佳替代 |

## 设计结论：样本跨度不是量尺的质量

旧 §5.3 的 `max(score)-min(score)` 是候选模型集的总体分数跨度。它能提醒团队
所有模型是否都接近满分/零分，却不是 item discrimination。题目区分度针对单道
题是否能区分候选总体能力不同的系统；其估计依赖评分、模型集组成、响应矩阵和
拟合条件。候选模型不是从目标能力总体独立抽来的“受试者”，同系列模型还会
相关。因此，本轮把跨度改名为 `R_score`，补上 CTT/IRT 诊断边界，并将 IRT 参数
定位为 pilot 中的坏题线索，而非自动删题或唯一排名机制。

## 设计结论：样本分布与目标分布不是同一个权重

分层抽样会改变各任务在 benchmark 中的比例。若直接取全体样本平均，估计的是
benchmark 自身的 q 分布；若目标是线上总体，且已知各互斥切片的目标比例 p_g，
可以先估每个切片的结果，再计算 `Σ_g p_g Q̂_g`。这一步只能修正切片构成差异：
切片内如果是主观挑出的 bad case、覆盖框不全或当前线上 mix 已漂移，权重不会
自动消除相应偏差。总分应同时保留原始样本量、每切片结果、目标权重与抽样方法。

## 代表性 benchmark 的设计取舍

| 方法 | 所测构念/设计操作 | 主要收益 | 主要代价和后续改进 |
| --- | --- | --- | --- |
| GPQA | 专家构造高难科学多选题，并邀请 domain / non-domain validators 尝试 | 验证难度不是由“题目看起来专业”主观决定；能观察专家监督瓶颈 | 题库小且学科集中，专家答案也可有错误；保留审题意见、仲裁和修订版本 |
| IFEval | 将显式格式/计数约束变成可程序验证条件 | 无需让 LLM judge 主观打“遵循程度”分，失败能定位到具体约束 | 只覆盖易形式化指令；与任务质量分开报告，并对语言/边界/多条件冲突补测试 |
| LiveBench | 以时间新近的公开来源、客观答案和滚动更新降低静态污染风险 | 不必完全依赖 crowdsourced preference 或 LLM judge 给硬题判分 | 发布后仍可能被训练；月度题集变化改变了测量表面；同时保留版本化 frozen core 和 dated fresh suite |
| Leaderboard sensitivity study | 改变选项次序和答案提取细节后比较名次 | 找到模型排名对实现协议的脆弱处 | 只在所测题库/模型/扰动上成立；对目标项增加有解释的扰动复测和排名区间 |

GPQA 的来源表区分了 Extended 与 main/Diamond 选择后的准确率；IFEval 的 loose
规则也记录了 false positive 代价。LiveBench 的论文更新计划与项目主页页面状态
分别对应论文方法和当前公开入口，不能不带时间/版本地合并。以上细节已在正文
§5.5 展开，说明论文方法、实验子集和实际项目版本都属于 benchmark 证据的一部分。

这些方法分别优化题目难度、规则可判定性、新鲜度和协议稳健性，不能用一个
方法代替完整 benchmark 设计。实验选择应从目标任务和错误代价反推，而不是因为
某种题型更容易快速出分。

## 旧正文继承与本轮修订

| 旧正文知识点 | 处理方式 |
| --- | --- |
| construct validity、效度/信度/决策价值、任务契约和被测系统边界 | 保留并继续作为全章骨架 |
| 样本 schema、证据、数据来源、采样、过采样、训练/调参/test 隔离 | 保留；不把自然流量平均效果与风险样本发现混为一谈 |
| 难度、覆盖率、区分度和 pilot | 保留；区分候选集总分跨度与 item-level discrimination，补 IRT 使用条件及 GPQA 筛选前后准确率口径 |
| rubric、多答案、标注员一致性、程序判定器与 sandbox | 保留；新增 GPQA 和 IFEval 作为不同判分设计示例 |
| pass@k、RAG、Agent、安全、协议、污染、manifest、版本生命周期、统计分析和客服知识库案例 | 保留原公式与案例；新研究资料只补方法边界 |
| OpenAI Evals 作为工具入口 | 保留开源 repository 的历史代码入口，明确 hosted dashboard/API 是另一对象并记录当前弃用安排 |
| Benchmark 诊断 demo | 明确 `train_similarity` 是合成的预计算输入而非语料扫描；缺字段时使用行号回报并仅用 schema-valid 行计算切片，空数据显式失败；增加所有 failed signals 到 action 的映射，防止有 gate 失败却仍输出 ready；保留原 7 条样本、任务 mix、相似度风险和预期动作 |
| 代表性 benchmark | 旧稿只有 benchmark 名称和链接；本轮逐个解释 GPQA 验证流水线、IFEval 四种评分粒度、LiveBench 论文/主页版本与扰动敏感性研究的问题、证据、边界和改进方向 |
| 线上比例估计与风险过采样 | 保留自然分布/均衡比较/风险发现三种用途；增加互斥切片约束、按目标比例加权的算例，并说明非概率 bad-case 采样不能只靠权重变成无偏线上估计 |
| 指标分母与人类一致性 | 补充 kappa 的观察/偶然一致解释、系数的设计边界、空分母时的 `N/A` 语义；人工评测完整方法留给下一章 |

## 手工推导与代码审阅记录

- demo 观察任务比例为 `qa=2/7≈0.286`，其他五类各 `1/7≈0.143`；和目标分布的 total variation 距离约为 `0.079`。
- `leak_001` 的固定字段为 `0.91`，高于示例 `0.80`，因此唯一失败 signal 是 `leakage_clear`，唯一 action 是 `remove_leak_flags_from_final_holdout`。
- 本轮没有执行 Python 代码；仅做静态阅读并手算上述输出字段，不记录运行通过。
- `D_mix=0.5×Σ|p_g-q_g|` 是两个分布在当前类别划分上的 total variation distance。它衡量切片占比差异，不衡量单条文本的语义距离。
- 抽样 toy 中 p=(0.70,0.15,0.10,0.05)，q=(0.40,0.25,0.20,0.15)，故 D_mix=0.30；四个切片分数为 (0.92,0.78,0.60,0.80)，q 加权平均 0.803，p 加权估计 0.861。
- kappa 的 `p_o` 是观测一致率，`p_e` 是由两位标注员标签边际分布算出的机会一致率；`p_e=1` 时分母为 0，不能输出伪造的数值。

## 未解决边界

1. 没有目标产品真实请求、授权训练语料、专家标注、线上回放或候选模型答题结果；章节里的客服场景、分层权重例子和 demo 均为教学设计。
2. 0.25 的分布距离阈值、0.80 的相似度门槛、风险样本数以及 demo 所有权重均为演示值，不是统计或生产建议。
3. LiveBench 论文的按月更新描述、项目主页所列最新快照和约每六个月完整刷新说明属于不同版本/时间层；后续使用前要重查当前页面。GPQA/IFEval 数字绑定作者题目、评分程序、抽样和版本；不得迁移成通用的“无污染”或能力分数。
4. IRT 预印本是最新方法警示，不等于 IRT 完全不适用于 benchmark；更稳妥的做法是报告假设、拟合、模型集合构成和不确定性。
5. OpenAI Evals 托管服务时间表会变，实施时要重查官方弃用页；开源 repository 与托管 dashboard/API 不应混称。
6. 目标比例加权要求切片互斥、覆盖目标总体且切片内抽样可支持该总体估计；风险题、历史 bad case 和人工精选样本通常不满足，需按压力测试解释。没有试跑响应矩阵和盲审标注，无法估计真实题目难度、区分度或人类一致性。
