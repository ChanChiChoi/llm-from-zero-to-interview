# 第七册第十四章：Frontier Model 证据等级研究记录

- 研究日期：2026-10-27。
- 正文：[第十四章](../book-07-evaluation-experiments/chapters/14-frontier模型证据等级.md)。
- 任务：在保留原有证据等级、claim ledger、冲突处理、复现、未知项和案例的基础上，补出代表性研究分别解决了什么缺口，并把新模型案例改写成有条件、有来源、有边界的证据审计。

## 代表性方法：解决的问题、机制和边界

| 方法 | 缺口与设计 | 能支持的结论 | 不能替代的证据 / 下一步 |
| --- | --- | --- | --- |
| [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) | 模型发布材料缺少用途、评测条件、群体切片和限制的统一结构；提出随模型发布的说明卡 | 使用者可以按共同字段理解发布者声明的用途和测试范围 | 卡片是披露格式，不审计内容真伪。后续对 2022 年 Hugging Face 快照 32,111 张卡片的[系统分析](https://arxiv.org/abs/2402.05160)发现 evaluation 和 limitations 字段分别只在 15.4% 与 17.4% 卡片中填写；仍需回查原始测试及独立证据 |
| [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) | 数据来源、组成、收集、处理、用途及限制难以追溯；为数据集生命周期提供结构化文档 | 可审查数据集来源与处理过程，并提出治理问题 | 数据集说明本身不能证明某模型训练时实际使用了该数据；需训练清单、数据快照、过滤/去重日志和模型 revision |
| [HELM](https://arxiv.org/abs/2211.09110) | 各家模型在不同任务、prompt 与指标下单独报分，比较条件不一致；以 scenario、adaptation、多指标和逐样本记录组织统一评测 | 原论文覆盖 30 个模型、42 个场景；共同核心场景覆盖率从此前平均 17.9% 提升至 96%，并展示准确率、校准、鲁棒、公平、毒性和效率的取舍 | 可行性选择仍限制场景覆盖；共同 harness 不保证代表目标流量。应补私有 holdout、目标切片和线上观测 |
| [CheckList](https://aclanthology.org/2020.acl-main.442/) | 总分掩盖行为失败；以能力清单配合 minimum functionality、invariance、directional expectation 生成可解释测试 | 用户研究中，一组从业者用它写出约两倍测试并发现近三倍 bug；另一项商业情感分类应用发现可行动缺陷 | 覆盖依赖能力清单、扰动有效性和 oracle 质量，不直接估计线上错误率；把发现转为目标语料回归用例 |
| [G-Eval](https://aclanthology.org/2023.emnlp-main.153/) | n-gram/参考答案指标难评价开放式文本；让 LLM 根据 rubric 生成细化步骤、按表单评分，并比较评分 token 概率加权 | GPT-4 在论文 SummEval 摘要任务上的离散评分与人工判断 Spearman ρ 为 0.514，展示 rubric 和聚合方式会影响 judge 相关性 | 相关性不等于事实正确性；结果限于论文模型和摘要数据，且观察到对 LLM 文本的偏好风险。需要目标任务人工校准、顺序交换、反事实与独立 holdout |
| [Evaluation Cards](https://arxiv.org/abs/2606.09809) | 评测运行、benchmark 元数据、模型资料分散，汇总分难追溯；把三类元数据组合成结构化记录和读者视图 | 论文由 52 篇工作和 10 次访谈提炼 schema，并以复现性、文档完整度、来源/风险、分数可比性组织信息；其工具分析 5,816 个模型、635 个 benchmark、101,843 条结果 | 这是 2026 年 arXiv 预印本及报告层方案，不检验底层结果真假；依赖记录与抽取质量。它组织评测运行，Claim Ledger 则追踪单条命题及支持/反证 |

这些工作不是由弱到强的证据排行榜，而是互补的环节：Datasheet 说明数据，Model Card 说明发布者披露，HELM 统一部分比较条件，CheckList 暴露行为反例，G-Eval 提供开放文本评分方法，Evaluation Card 组织一次运行的可解释字段。它们都不能代替目标产品 holdout 或线上验证。

## 本章示例和模型发布审计

- 把 `unknowns` 作为列表记录，避免把未知项数量误当成未知项本身；架构 claim 的直接来源检查改为逐条遍历，并要求技术报告、官方配置或官方代码直接支持该命题。
- 观察项从产品名、官方 API/模型 ID、artifact/config/code、技术报告逐级升级；找不到精确名称或 leaderboard 行只表示当前来源/快照没有对应记录，不代表模型不存在或得分为零。
- Grok 4.7：将 xAI 发布方任务结果、Artificial Analysis 的 provider/configuration 测量和 DataCurve DeepSWE 的特定 harness 行分开。DeepSWE 页面在 2026-09-22 快照列出 113 个任务、91 个仓库、5 种语言；没有 Grok 4.7 精确行时，不借用 Grok 4.6 或邻近版本结果。官方发布页说明较大 base model、更长 RL 训练、长任务偏置和 Grok Bot harness 适配，但没有完整参数与训练配方。
- Claude Opus 5.5：发布方 benchmark、第三方 provider/config 指标和特定 Agent harness 结果分账。System Card 的 OSWorld 2.0 partial/strict `81.8%/48.7%`（108 tasks、1080p、最多 500 actions、5 runs、超 100K token server-side compaction）与当前 Anthropic 页面 OSWorld 2.1 partial `81.8%` 是不同 revision；相同数字不能合并。Terminal-Bench 4.0 `66.36%` 绑定 xhigh、Claude Code `--bare`、5 trials；ProgramBench `91.2%` 绑定 166 个 golden tasks。
- Claude Fable 5.1：记录 System Card 中 benchmark revision、partial/strict 定义、safeguards、fallback 和实际执行模型；Fable 5.1 与 Mythos 5.1 的关系及安全配置要按发布方披露表述。不同 safeguards、provider 配置与 DeepSWE 结果不能合成无条件能力排名；DataCurve 2026-09-22 快照无 Fable 5.1 精确 Agent 行。

模型发布源与可复核提取值继续以三份带版本/来源边界的记录为准：[Grok 4.7](model-update-2026-09/grok-4.7-source-notes.md)、[Claude Opus 5.5](model-update-2026-09/claude-opus-5.5-source-notes.md)、[Claude Fable 5.1](model-update-2026-09/claude-fable-5.1-source-notes.md)。章节中的现时页面和榜单快照须按引用日期复核。

## Demo 与验收边界

- Demo 只展示 claim 作用域、独立任务证据、架构来源和未知字段的教学门禁；本轮静态复读并手算预期，不执行 Python。
- 当前案例不能推出未公开的参数、架构、训练配方、目标产品准确率或生产安全性；厂商测量、provider 测量、固定 benchmark 与独立复现要分别记账。
- 未做全章 URL HTTP 扫描、外部公式渲染、PDF 构建或生产模型评估。
