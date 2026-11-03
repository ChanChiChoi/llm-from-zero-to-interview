# Benchmark 设计与测量协议来源

核验日期：2026-11-03。本文只记录来源事实、版本、阅读范围和证据边界。

## IRT 在 AI benchmark 中的估计边界

- **来源与版本：** Jiang et al., [Can We Trust Item Response Theory for AI Evaluation?](https://arxiv.org/abs/2607.15190)，arXiv v1，2026-07-16。
- **阅读范围：** 摘要、模拟方法和结果概述、结论与局限。
- **来源支持的事实：** 研究以 6 个常用 LLM benchmark 的题目参数和能力分布为基础，在 3 类 IRT 模型、4 种估计器和 18,000 个模拟条件中比较推断。作者报告，传统估计器在较大 benchmark 上可能难以计算；可扩展估计器在模型数量较少或能力分布非正态时，仍可能给出不可靠的题目级或排名推断。文中也指出，长题池可在部分设置下稳定模型聚合分数，但不足以自动保证题目质量推断和排名恢复。
- **证据边界：** 这是预印本中的模拟研究，结论绑定所选 benchmark 参数、生成机制、估计器和单维 IRT 假设；未覆盖的 30—100 模型样本规模、硬件差异及题目语义信息仍是限制。不能据此说 IRT 一概无效，也不能只凭拟合参数自动筛选最终测试集。

## 难题与可核验指令的设计例子

- **GPQA 来源与版本：** Rein et al., [GPQA: A Graduate-Level Google-Proof Q&A Benchmark](https://arxiv.org/abs/2311.12022)，arXiv v1，2023-11-20。
- **阅读范围：** 全文摘要、数据收集与数据划分、题目客观性分析。
- **来源支持的事实：** 论文报告 GPQA Extended 为 546 题；主集为 448 题，筛选条件为至少一名领域专家验证者同意且三名非专家中不超过两人答对；Diamond 为 198 题，要求两名专家验证者都同意且非专家最多一人答对。Extended 的专家/非专家正确率分别为 64.8%/34.1%；主集表格为 71.9%/30.4%，并注明主集与 Diamond 的验证者准确率受筛选影响。摘要另报专家原始正确率约 65%，在事后扣除专家明确确认的疏忽后估计约 74%。
- **证据边界：** 这些正确率来自指定学科、题集、招募和验证协议。调整后的 74% 是专家准确率估计，不是“74% 的题客观”这一题目占比；不同筛选子集的正确率不能当作同一总体横向比较。

- **IFEval 来源与版本：** Zhou et al., [Instruction-Following Evaluation for Large Language Models](https://arxiv.org/abs/2311.07911)，arXiv v1，2023-11-14。
- **阅读范围：** 全文摘要、数据构造和评价指标定义。
- **来源支持的事实：** 论文构造 25 类可验证指令和 541 个 prompts；每个 prompt 可包含一条或多条指令。它分别报告 prompt-level 与 instruction-level strict accuracy，以及对应的 loose accuracy。loose 规则组合去除 Markdown 强调标记、首行和末行等变换，可降低某些字符串格式导致的假阴性，也可能引入假阳性；作者将其定位为 strict 规则的补充。
- **证据边界：** 这是论文当时构造的特定英语指令集和规则实现。显式约束可编程核验，不等于事实正确、任务有用或通用 instruction following；语言计数、Unicode、冲突指令和语义质量需要另行测量。

## 更新型题集与执行协议敏感性

- **LiveBench 论文来源与版本：** White et al., [LiveBench: A Challenging, Contamination-Limited LLM Benchmark](https://arxiv.org/abs/2406.19314v2)，arXiv v2，2025-04-18。
- **阅读范围：** 论文摘要和版本维护设计；另于 2026-11-03 访问[项目主页](https://livebench.github.io/)核对页面当前公开快照说明。
- **来源支持的事实：** v2 描述近期题源、客观答案和周期性更新，用以限制静态题集的污染/陈旧风险。访问日项目主页仍把 LiveBench-2026-01-08 列为最新公开版本，并称题目定期更新、约每 6 个月完整刷新，近期更新题目会延迟公开。
- **证据边界：** 论文设计中的月度增加/替换节奏和项目主页当前的完整刷新说明属于不同时间层。此处记录主页访问时可见的版本状态，不证明题目未泄漏，也不说明下一次发布时间；评测报告应固定下载版本和评分代码。

- **来源与版本：** Alzahrani et al., [When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards](https://arxiv.org/abs/2402.01781v2)，arXiv v2（ACL 2024 camera-ready，2024-07-03）。
- **阅读范围：** 摘要与研究设置概述。
- **来源支持的事实：** 在作者测试的多选 benchmark、模型和扰动设置中，改变选项顺序或答案选择方法等细节可导致名次最多移动 8 位；论文讨论了适用设置中的 hybrid answer scoring。
- **证据边界：** “最多 8 位”是所测榜单和扰动下的观察，不是对任意 benchmark 的预期变化量。要把此结果迁移到新任务，需固定主协议并单独设计语义保持的扰动审计。

## 相关动态文档

OpenAI 托管 Evals 的生命周期与 2026-11-03 官方时间表复核，另记录于[评估设计、不确定性与污染来源](evaluation-design-uncertainty-and-contamination-2026.md)。此处不把托管平台状态外推为评测方法或开源 `openai/evals` 仓库的生命周期。
