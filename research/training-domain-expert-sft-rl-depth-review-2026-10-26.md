# 第五册第十五章：领域专家 SFT 与 RL 研究记录

日期：2026-10-26  
正文：[`book-05-llm-training/chapters/15-领域专家sft与rl.md`](../book-05-llm-training/chapters/15-领域专家sft与rl.md)  
依据：当前 [`WRITING_SPEC.md`](../WRITING_SPEC.md)、`progress_v2.md`

## 本章定位与问题图谱

本章回答的不是抽象的“怎么让模型懂金融/医疗”，而是：当通用模型做一项专业工作失败时，如何辨认缺口对应的训练信号；代表性行业研究实际改变了什么；如何证明改变了工作流能力，而不是只让 loss、偏好分或格式分变好。

本轮将核心问题整理为：

1. 领域继续预训练、SFT、偏好优化、RLVR、RAG/工具和独立 policy 分别改变什么对象；什么故障不应靠训练去修。
2. 领域专家如何把事实、证据、假设、计算、版本、动作和分歧写入样本，而不是只标最终答案。
3. 市场结果、专家偏好与程序 verifier 的监督语义不同；统称“反馈”会掩盖混杂、偏好代理和 verifier 漏洞。
4. 领域评估必须贴近实际工作流，并覆盖时间、患者/实体、版本、地区和风险切片；测试集自身也会有覆盖偏差。
5. 训练收益必须与回归、权限风险、人工复核和单位成功成本共同验收。

## 已阅读来源与证据

| 来源 | 本轮阅读内容 | 支持的判断 | 边界 |
| --- | --- | --- | --- |
| [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964) | 四个领域、八个分类任务的 DAPT/TAPT 设计与实验 | 区分领域无标注语料适配与更窄任务语料适配；研究报告 RoBERTa 在多个领域/任务上获得收益 | 使用 encoder 与分类任务，是方法谱系证据，不是当前生成式 LLM 的直接效果估计 |
| [Continual Learning for Large Language Models: A Survey](https://arxiv.org/abs/2402.01364) | LLM 继续预训练、指令微调、对齐的阶段分类，以及 RAG/模型编辑对照 | 动态事实更新、语言/领域能力和任务行为不是同一种更新目标；跨阶段训练会引入遗忘问题 | 综述用于问题地图，不能替代各算法或产品训练配方的原始证据 |
| [FinGPT: Open-Source Financial Large Language Models](https://arxiv.org/abs/2306.06031) | 金融数据获取、自动清洗、LoRA/QLoRA 和股价方向反馈章节 | 金融数据更新快且高质量人工标签稀缺；一个论文配置报告可训练参数从 6.17B 降至 3.67M | 参数节省不等于领域任务准确率提高；数字只绑定论文配置，不能当作所有 LoRA 的固定比例 |
| [FinGPT: Democratizing Internet-scale Data for Financial Large Language Models](https://arxiv.org/abs/2307.10485) | 34 个网络数据源、实时采集和 RLSP 方法/讨论 | RLSP 用相对股价变化产生市场标签，数据采集可扩展；作者明确指出市场趋势过拟合风险，股价受新闻外因素影响 | 市场价格是有混杂的弱监督信号，不是新闻因果标签、专家偏好，也不能证明可盈利；要用时间隔离与交易成本进行独立评估 |
| [FinBen: A Holistic Financial Benchmark for Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/adb1d9fa8be4576d28703b396b82ba1b-Paper-Datasets_and_Benchmarks_Track.pdf) | NeurIPS 2024 最终论文任务统计、21 个模型评估发现和限制 | 最终发表版本含 42 个数据集、24 个任务；作者报告指令微调模型在文本分析有提升，但复杂 QA、文本生成和预测增益有限 | 发表版数据规模与早期 arXiv 摘要不同，正文使用发表版；任务数据规模、英语/美国市场和当时模型范围限制外推，benchmark 不是部署安全或微调因果证明 |
| [MedAlign: A Clinician-Generated Dataset for Instruction Following with Electronic Medical Records](https://doi.org/10.1609/aaai.v38i20.30205) | AAAI 2024 论文的任务构造、医生评审和错误率 | 15 位临床人员/7 个专科、983 条 EHR 指令、276 份纵向病历、303 条临床参考答复；六个通用模型错误率为 35%—68% | 错误率绑定论文所测模型和任务；这是评测，不是领域微调效果；EHR 数据受研究使用协议限制 |
| [TIMER: Temporal Instruction Modeling and Evaluation for Longitudinal Clinical Records](https://www.nature.com/articles/s41746-025-01965-9) | npj Digital Medicine 2025 全文的方法、表 1/2/5、评测生成、局限与数据可用性 | 同底模时间戳指令微调改善病历时间推理；Llama 3.1 8B 的 MedAlign correctness 30.69%→34.32%，TIMER-Eval 45.02%→48.51%；训练分布与均匀时间线测试匹配时相对近期事件训练有 +6.5% win advantage | 基于 Stanford STARR EHR 和 Gemini 1.5 Pro 生成的时间对齐指令；每种时间分布 5,000 条、训练 6 epochs；测试使用不同患者时间线。只抽审 100 条生成指令、原始 EHR 不公开，亚群公平性和输出校准未覆盖；不是生产临床安全证据 |
| [DeepSeekMath](https://arxiv.org/html/2402.03300) | 数学语料的采集与去污染、SFT 数据、GRPO/结果监督和模型评估 | 120B math-related token 继续预训练后再做数学 SFT 和 RL；7B 从 SFT 到 RL 的 MATH 46.8%→51.7%、GSM8K 82.9%→88.2% | 数学答案可由规则核验，不能外推到开放式医疗/法律/金融；分数依赖作者模型、数据、采样和判分协议。GRPO 机制主讲在本册第十一章，本章引用其“奖励何时可验证”的事实 |

## 代表方案如何分工

| 缺口 | 对应信号 | 代表证据 | 为什么不能用其它信号替代 |
| --- | --- | --- | --- |
| 领域文本和术语分布不足 | DAPT/CPT | DAPT/TAPT 研究与 LLM continual learning survey | 语言建模 loss 下降不等于工作流遵循、事实引用正确或任务成功 |
| 领域任务输入/输出协议和步骤不会做 | SFT / 轨迹 SFT | TIMER 的时间戳指令样本；FinGPT 的低成本 LoRA 路线 | token imitation 不独立验证计算、引用支持、动作权限或泛化 |
| 两个答案都可接受但偏好不同 | 偏好数据 / DPO / RM | 本册第十一章 DPO、RLHF 和偏好方法比较 | 偏好标签表达相对选择，不提供绝对事实或规范真值 |
| 明确、稳定、可程序判定的结果 | RLVR / 环境反馈 | DeepSeekMath | verifier 只覆盖所写规则；开放判断不能被虚构成精确 reward |
| 最新事实、法规、财报 revision | RAG/工具与版本控制 | LLM continual learning survey 的阶段区别；本章数据版本链 | 权重不容易做逐条撤销，持续训练也可能遗忘；新鲜资料应先由权威来源提供并可引用 |
| 隐私、越权、生产副作用 | policy engine、执行器、人工确认 | 本章 `read -> draft -> review -> execute` 状态边界 | 模型训练分数不具备授权语义，硬约束不能被软 reward 抵消 |

FinGPT 与 DeepSeekMath 特别适合面试时说明两种不同的结果反馈。FinGPT/RLSP 试图利用市场规模化标签，但市场变化不是新闻的纯因果结果；DeepSeekMath 的最终答案可由规则判分，信号更容易与任务成功对齐，却只适用于结构化数学/可验证任务。若面试者只说“RL 用 reward 优化”，缺少的关键判断是 reward 的来源、可观测性、混杂、可钻空子程度和对应的业务目标。

TIMER 的价值也不只是一个医疗模型分数。它把临床任务具体化为“对病人时间线上多个时点做查询和趋势推断”，通过时间戳、证据定位和训练分布消融把数据设计映射到能力；其训练/测试按患者时间线隔离，且报告 correctness 与 completeness。其不足（合成数据偏差、有限专家审阅、单数据环境、公平性/校准未测）恰好说明：一篇强结果也要说明支持什么、尚不支持什么。

## 原有知识点覆盖图

| 原有问题/知识点 | 本轮处理 | 修订后位置 |
| --- | --- | --- |
| SFT、偏好优化、RLVR 和 policy 的路线差异 | 保留原路线表，加入 DAPT/TAPT、RAG/时效事实和真实论文对照，明确对应缺口 | §15.0、§15.0.1 |
| 通用能力与领域工作流差异 | 保留，仍用财报和临床工作流讲证据、动作和责任 | §15.1、§15.6、§15.7 |
| assistant token loss、空 mask 和 token/任务目标区别 | 保留机制和边界；不拿文献表代替 label/mask 解释 | §15.2、§15.18 |
| 专家样本分层、专家参与、分歧和条件版本 | 保留事实/过程/交付三层、标注链、alternative answer 和 disagreement policy | §15.3、§15.4、§15.10.4、§15.19、§15.25 |
| 金融、医疗、工具和安全案例 | 保留原示例与权限状态机，研究案例补具体数据构造、指标和失败边界 | §15.6、§15.7、§15.10.3、§15.17、§15.23、§15.28 |
| 专家反馈、reward、verifier、hard gate | 保留并区分偏好、市场弱反馈和精确结果 verifier；硬风险仍是独立 policy | §15.5、§15.14、§15.20、§15.21 |
| 评估桶、时间切分、分布、回归和成本 | 保留原指标，加入 FinBen/MedAlign/TIMER 对行业评测覆盖和时间分布的实证 | §15.0.2、§15.8、§15.12、§15.15、§15.22、§15.24、§15.27 |
| 数据刷新、遗忘、完整金融流水线、发布账本 demo 和练习 | 均保留；补数据配比/反事实设计和 source revision 追溯入口 | §15.11、§15.22—§15.28 |

本章既有多处相邻主题会反复提到 verifier、评估、回归、权限和版本。由于它们分别回答“训练目标”“如何做评估”“如何限制策略”“如何发布”的问题，本轮保留这些主讲位置，不把重复出现的关键词误删为某一主题已经被覆盖；新增研究案例集中放在 15.0，后文只在需要处使用证据，不再罗列相同论文结论。

## 正文修订落点与本地验收

- §15.0 增加研究驱动的适配路线选择图谱；对 DAPT/TAPT、FinGPT/RLSP、TIMER、DeepSeekMath 说明研究目标、机制、观察到的结果和外推限制。
- §15.0.2 加入 FinBen/MedAlign 评测证据，并通过 TIMER 的 55.3% recent-quarter skew 展示“临床真实题目仍会有评测覆盖偏差”。
- §15.5 加入市场弱标签、偏好比较、结果 verifier 三种反馈的监督语义对照，避免把名称相同的“feedback/RL”误认为相同目标。
- §15.8 将领域评估深化为任务来源、时间线覆盖、患者/机构隔离、关键字段与未知切片检查。
- §15.30 增代表性论文和评测集，并在章节文中为关键数值放直接来源入口。
- 同步 `PAPERS.md`、`GLOSSARY_EN_ZH.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`；第五册目录已有第十五章路径，不需要调整。
- 原 §15.28 Python 审计 demo 保留。执行、输出对照、异常输入检查和 Markdown/链接/公式/标题核验完成后，在此记录具体结果。

## 本地验收结果与仍未验证的范围

- §15.28 标准库 Python demo 提取执行成功。正常路径输出毛利率 `0.36`、单位成功任务成本 `7.5` 和发布通过；缺 revision、零收入、越权写入、零成功任务、负复核成本均触发预期错误，rollback drill 缺失时发布阻断。
- 第 15.0—15.30 一级章节标题连续；17 对 Markdown 围栏成对，13 个 `math` 围栏的括号成对；章节和本研究笔记中的相对 Markdown 目标存在；第 15 章所新增的 EXERCISES/INTERVIEW_BANK 入口存在。
- `git diff --check` 通过。
- 未训练模型、未执行真实医疗或金融数据、未运行生产权限系统、未用 GPU 复现 TIMER/FinGPT/DeepSeekMath。公开论文数字仅按各自模型、语料、任务和指标范围引用；示例代码只验证本地教学审计逻辑。
