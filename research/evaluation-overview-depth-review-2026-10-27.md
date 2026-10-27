# 第七册第一章：评估总览资料驱动复核

核验日期：2026-10-27。依据 `WRITING_SPEC.md` 对 `book-07-evaluation-experiments/chapters/01-评估总览.md` 复读，重点检查“评估层次、代表性框架、分数可比性、judge、污染、持续评测”的行业证据与正文解释是否对应。正文原有 14 个大节、benchmark 地图、指标公式、统计区间、RAG 案例和多层 demo 均保留；本轮是在现有主线中补方法对照、最新产品状态和必要的边界说明，没有删掉已有主题。

## 研究问题

1. 代表性评估路线分别解决评估链上的哪个故障，机制如何工作？
2. 哪些结论是论文方法证据，哪些只是框架实现或厂商工程建议？
3. 开放式评判与 benchmark 污染如何避免被“分数”掩盖？
4. 本章将 OpenAI Evals 作为工程入口时，当前托管产品生命周期是否改变了推荐方式？
5. 现有章节中的指标诊断代码是否把 case-level score comparison 误称为 pairwise preference，或让演示规则看起来像发布门禁？

## 来源与证据映射

| 来源 | 阅读/核验内容 | 能支持的判断 | 不能支持的判断 |
| --- | --- | --- | --- |
| [HELM 原论文](https://arxiv.org/abs/2211.09110) | scenario、adaptation、model、metric 的整体评估设计；其覆盖共同场景和多种指标的出发点 | 共同任务和多维指标可改善模型间结果透明度与可比性；HELM 选择的是一组明确场景，不是所有真实任务全集 | 所有产品场景均被覆盖，或模型级分数能代表带有检索、工具、策略的完整系统 |
| [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782) 与 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | arXiv 页面显示该论文 2026-05-31 修订至 v3；维护者总结评测设置敏感性、比较方法、复现和透明度困难。仓库将任务执行实现为可共享的 harness，并列出不同 backend、task 和 generation 配置 | 固定执行框架能减少各自重写任务、prompt、request、post-processing 与 metric 带来的实现差异；重复实验要记录任务、prompt、模型、tokenizer、生成参数和代码版本 | harness 选出的任务能代表目标用户；统一 runner 自动保证模型间公平；维护者经验等同于所有场景都成立的实验定律 |
| [MT-Bench / Chatbot Arena 论文](https://arxiv.org/abs/2306.05685) | 论文讨论用强 LLM judge 评开放对话、众包 Arena 比较人类偏好，并列出位置、冗长、自我偏好和 judge 推理能力限制 | 当答案不唯一时，成对比较和偏好标签能提供 exact match 无法给出的质量信号；judge 可以扩展评估覆盖 | 偏好就是事实、正确性或安全；论文中的 judge-human agreement 可直接迁移到任意领域、模型或 rubric |
| [Judging the Judges: A Systematic Study of Position Bias](https://arxiv.org/abs/2406.07791) | 研究在多个 judge、任务和答案组合下交换候选位置，分析重复稳定性、位置一致性和偏好公平性 | 交换顺序是可操作的位置偏差诊断；偏差大小依赖 judge 和任务 | 交换顺序后结果一致就证明裁判正确，或某一 judge 在目标任务中已校准 |
| [ConStat: Performance-Based Contamination Detection](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a7f89793b9e6f8c6568dbbb6ff727b9b-Abstract-Conference.html) | 论文把污染相关异常操作化为公开 benchmark 成绩不能迁移到 rephrased / synthetic 同分布样本或同任务参考评估上的表现差距 | 行为上的非泛化差距可作为可疑污染信号，用来补充语料重合审计 | 证明训练集逐字包含 benchmark 原题，或给出完整污染样本溯源 |
| [OpenAI evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices) | 官方工程文档建议定义目标、收集数据、定义指标、运行比较、持续评测，并把新线上案例用于迭代 | 厂商提出的一种 eval-driven 开发循环，可作为工程流程参考 | 学术有效性、独立评估平台比较或任意产品都应使用 OpenAI 工具 |
| [OpenAI API deprecations](https://developers.openai.com/api/docs/deprecations) 与[迁移说明](https://developers.openai.com/cookbook/examples/evaluation/moving-from-openai-evals-to-promptfoo) | 官方页面在 2026-10-27 显示：Evals platform 于 2026-10-31 起只读，dashboard/API 计划于 2026-11-30 关闭；官方提供迁移到 Promptfoo 的说明 | 当前章节不应把托管 Evals 平台写成长期稳定依赖；数据和 grader 应可迁移 | `github.com/openai/evals` 开源仓库已同步弃用，或任何特定替代工具适用于所有团队 |

## 代表性方法的解题关系

本章 §4.7 采用“覆盖—执行—开放式判分—污染审计—持续回归”的区分：

- HELM 主要解决评测覆盖零散和模型间报告不对齐的问题。
- lm-eval 主要解决同一批任务在不同模型/后端上执行协议难复用的问题。
- MT-Bench / Arena 处理开放回答没有唯一 gold string 的情况；human preference 与 judge preference 是质量观测，不是事实真值。
- overlap audit 与 ConStat 行为检测观察不同对象：前者寻找可访问语料中的字面/近似重合；后者捕捉公开题成绩无法迁移的行为信号。两个结果都需写明假设和不可见范围。
- 持续 eval 将新版本回归和线上失败样本纳入开发循环；它的工具平台不改变数据、指标和判分本身的有效性。

因此，不能用“选一个评测框架”代替评估设计。一个较稳妥的系统可先用 HELM 风格矩阵检查覆盖缺口，用锁定版本的 harness 执行固定样本，用程序 verifier 处理客观条件，以人工校准的 pairwise/judge 补充开放质量，再用数据审计和改写题观察污染，最后在经过风险门控的线上实验中验证产品结果。

## 旧正文继承与本轮修订

| 旧正文知识点 | 处理方式 |
| --- | --- |
| 评估不是单一分数；能力、协议、安全、效率和产品价值分开 | 保留原向量化表达，新增 §4 六层不是互斥类别的说明 |
| Offline、online、human、automatic、safety、regression eval | 保留六节定义；增加维度交叉解释，避免误读为互斥 taxonomy |
| HELM、lm-evaluation-harness、MT-Bench、污染检测 | 保留 §5 benchmark 地图和原延伸阅读；新增 §4.7 方法谱系解释各自问题、收益、限制与组合方式 |
| A/B 顺序、judge 人工锚点、偏置 | 保留现有 rubric/gold-set 流程；补充针对同一样本交换 A/B 顺序的诊断作用和“稳定不等于正确”边界 |
| 指标、bootstrap、线上比例和 RAG 案例 | 保留原公式与案例，未改成只有框架名的宏观介绍 |
| OpenAI Evals 工程入口 | 保留开源 repository 链接但区分托管平台；新增截至核验日的只读/关闭时间及官方迁移来源 |
| 多层评估诊断 demo | 将 case-level baseline/candidate 分数比较从 `pairwise` 更名为 `case_score_comparison`，避免与人工/judge preference 混淆；将质量、slice、延迟、成本等失败信号都映射为 action/failed gate；说明分数必须共享量表。输出中的旧/新加权分、5 次候选胜出、3 次平局、`0.8125`、z 值与原始教学数据一致性按手算核对；本轮没有执行代码。

## 未解决边界

1. 未连接目标产品日志、隐私治理、在线实验平台或生产数据；正文的 RAG 与持续评估案例是教学设计。
2. 各论文结论分别绑定作者数据、judge、任务、评测脚本和版本，不能据此排名当前商用模型。
3. OpenAI Evals 时间表会随官方通知变化；读者实施前应重查官方 deprecation 页面。这里的日期只表示 2026-10-27 的官方状态。
4. demo 中的安全样本仅 3 条、线上比例使用独立样本近似公式，`z>=1.64` 只是演示信号，不是释放门槛；真实样本量、随机化单位、置信区间、多重检验和高风险审核需要按项目设计。
5. 章节末 benchmark 名单会变化；此处修订重心是方法的证据角色，而非宣称列举了 2026 年所有主流 benchmark。
