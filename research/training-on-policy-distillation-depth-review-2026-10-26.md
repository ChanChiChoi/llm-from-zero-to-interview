# 第五册第十七章 On-Policy Distillation 深度精修研究记录

- `date`：2026-10-26
- `chapter`：[`book-05-llm-training/chapters/17-on-policy-distillation.md`](../book-05-llm-training/chapters/17-on-policy-distillation.md)
- `spec`：当前 `WRITING_SPEC.md`；重点落实重要知识点的研究证据、代表方法的机制/收益/代价对照、旧内容去向、公式手算、可执行 demo 和配套题库同步。
- `research_questions`：学生自生成状态究竟解决什么分布错位；sampled log-ratio 应作为何种梯度信号；局部条件分布拟合与完整序列目标怎样区分；近期方法分别修复哪个环节；教师反馈的可靠性与独立任务正确性如何验证；公开训练器把论文目标映射成哪些采样、显存、吞吐与版本选择。
- `source_policy`：原始论文用于核对机制和数据，厂商技术报告用于确认发布方公开的训练阶段及报告结果，官方工程文档/仓库用于检查接口或实现入口，工程文章用于案例启发并明确标为作者/实验室自述；不同证据类型不混作同等级结论。

## 研究问题图谱

| 主问题 | 需分开的环节 | 本章落点 |
| --- | --- | --- |
| 离线蒸馏为何在学生自己的错误上失效 | 教师示范状态分布、学生部署状态分布、专家可查询性与教师在偏离状态上的可靠性 | §17.0—17.1；用 DAgger 做问题动机但明确其假设不构成 OPD 保证 |
| 学生在训练中究竟学什么 | token 分布、动作、artifact、verifier、学生状态来源和状态快照 | §17.2—17.8；记录 student/teacher/verifier/environment revision、工具结果、成本、隐私和缓存身份 |
| KL 与 sampled log-ratio 如何变成梯度 | full-vocabulary local KL、sampled token score-function、序列 reverse KL、return-to-go 和 baseline | §17.3、§17.9；新增二动作手算，区分期望等式、单个样本和真正的参数梯度 |
| 训练为何不稳或负迁移 | 长轨迹位置偏差、学生与教师支持不匹配、高教师熵/多模态解、重尾 reward、近零信号、估计方差和 verifier 错误 | §17.10—17.12、§17.17、§17.20；方法按修复环节对照，不把所有方法概括为“更稳的 KL” |
| OPD 是否带来可迁移能力 | 新题、失败恢复、长轨迹、工具权限、verifier 独立性、通用能力回归、成本和多 seed 波动 | §17.13—17.15、§17.21；区分模仿损失、教师调用率和任务成功/恢复能力 |
| 论文方法如何进入实际训练 | generation worker、teacher service、完整 logits 或 sampled log-prob、token mask、loss reduction、version pinning、生成/教师成本 | §17.16、§17.20.1、§17.23；结合 TRL 文档与 MiniLLM GitHub 入口，未声称本地跑过真实训练器 |

## 读过的来源与证据增量

| 来源 | 本轮核对的机制、数值或接口 | 适用范围与限制 |
| --- | --- | --- |
| [DAgger 原论文](https://proceedings.mlr.press/v15/ross11a.html) | learner 先访问自身状态，再向专家查询动作并迭代聚合数据；支持用“学生所到达状态”解释离线示范的错位来源。 | 理论保证依赖在线学习、可查询专家及代价界等前提。LLM 教师会在极端前缀失准，查询成本/隐私也不同，所以正文只用作概念先例。 |
| [MiniLLM](https://arxiv.org/abs/2306.08543) | 学生序列分布对教师的 sequence-level reverse KL；分析 full-vocabulary 项与 policy-gradient 未来信用；作者报告 teacher mixing、length normalization 和多个随机种子。正文保留原有 Dolly 消融：移除 length normalization 后 ROUGE-L 24.6→14.7。 | 实验是早期指令跟随模型和 Dolly 等任务；不能直接外推到现代工具 Agent、数学推理或当前训练栈。逐 token importance-ratio 处理、长度项等都要看具体实现。 |
| [GKD 原论文](https://arxiv.org/abs/2306.13649) | 将学生生成序列比例和条件分布 divergence 作为可调轴；比较 KL/JSD 等目标，优化时把离散生成作为采样状态处理。论文实验含 T5-XL 教师及 77M/250M/800M 学生，任务覆盖摘要、翻译、算术推理和 instruction distillation。 | 任务间对 divergence 的偏好不同；token-local 条件拟合不自动等于完整序列 reverse-KL 的时间信用。不同数据/模型设置不能直接汇总排名。 |
| [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) | §4.5：small-model pipeline 分 off-policy response distillation 与 on-policy student sequence logits 对齐，teacher 为 Qwen3-32B 或 Qwen3-235B-A22B。§4.7/Table 21 从相同 off-policy-distilled Qwen3-8B 起点比较：AIME'24 初始/RL/OPD 为 55.0/67.6/74.4（Pass@64 90.0/90.0/93.3），AIME'25 为 42.8/55.5/65.5（83.3/83.3/86.7），MATH500 为 92.4/94.8/97.0，LiveCodeBench v5 为 42.0/52.9/60.3，MMLU-Redux 为 86.4/86.9/88.3，GPQA-Diamond 为 55.6/61.3/63.3；报告列出的 GPU-hours 为 RL 17,920、OPD 1,800。 | 这是一个技术报告的作者结果。可支持“该 Qwen3-8B pipeline 中 OPD 结果更高、报告 GPU-hours 更低”；不能推出 OPD 普遍优于所有 RL、得到相同算力下因果性强结论或所有指标可比。正文的约 10 倍只绑定该表。 |
| [A Survey of On-Policy Distillation for LLMs, v5](https://arxiv.org/html/2604.00626v5) | 2026-10-21 修订；页面标为 Work in Progress。将 OPD 放在 student state / teacher feedback / policy update 的交互里；区分 GKD 的 flexible conditional matching 与 MiniLLM 的 sequence-level reverse KL/temporal credit；指出 stale rollout policy lag、local token estimator 不等价于序列目标等问题。 | 是在写综述，不是独立 benchmark 或算法标准。正文用于搭研究地图，关键机制和结果仍引用各原论文。 |
| [A Formula-Driven Survey and Research Agenda for OPD](https://arxiv.org/abs/2606.22793) | 2026-06-22 v1；以 sampled action weighting、temporal credit 和 vocabulary routing 区分估计器偏差；提出 GAE-OPD、CR-OPD 等后续研究方向。 | 单篇预印本/研究议程；提出的方法名不意味着已有可靠大规模实验。正文明确把其假设作为待研究方向。 |
| [REOPOLD](https://arxiv.org/abs/2603.11137) | 将 stop-gradient sampled log-ratio 与 policy gradient 联系；针对严重负重尾、近零 token 和熵塌缩采用混合 reward clipping、熵分位筛 token、探索→精炼两阶段。报告数学 600 steps 与 ProRL 约 2,000 steps 的结果对照，并在视觉推理报告 7B student / 32B teacher 的约 3.32× inference efficiency。 | 结果限作者各自 baseline、任务、训练步数与 test-time K/latency 设置。论文页、Microsoft Research 页面标题/venue 元数据存在变化；本章按题名/arXiv ID 引用，研究记录不推断官方仓库不存在或已发布。未本地复现。 |
| [MOPD 原论文](https://arxiv.org/html/2605.12652) | 同题多条学生 rollout 中，向 self-teacher 提供成功/失败 peer。Qwen3-4B AIME'24 mean@8：base 16.89、SDPO 7.29、MOPD 28.54；LiveCodeBench-v6 mean@8：base 28.80、SDPO 48.84、MOPD 57.01。作者的 peer-context 消融里，2 个成功 + 1 个失败 rollout 在部分 teacher-signal 指标表现较强；运行代价相对 SDPO 每步在 LCB +9.1%、QA-Chemistry +33.3%。 | 成功/失败需可靠 verifier；额外 rollout/context 增加成本。作者还报告 Qwen3-4B LCB 中 57% prompts 没有成功 rollout，hard bucket 缺正例。论文是 preprint 结果，不可泛化到不含同类成功轨迹的学生。注意 MOPD 还和其他模型报告缩写重名，配套术语已按全称区分。 |
| [Thinking Machines Lab：On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) | 文章说明每 token reverse-KL/log-prob query 与 RL loss 的工程拼接，并提出通过旧模型教师恢复 mid-training 后行为；其算法说明将 `γ=0`，即当前 token immediate feedback，不纳入后续 return-to-go，并称 `γ>0` 在其测试中没有收益。在内部 QA eval / IF-Eval：Qwen3-8B 18%/85%；100% document mid-train 43%/45%；70% 文档+30% chat 为 36%/79%；后续 OPD 为 41%/83%。 | 实验与 QA set 由实验室自有工程文章报告，不是外部审计 benchmark。可作“行为恢复”工作流示例，不可宣称普遍无遗忘或跨模型效果；`γ=0` 是作者实现选择，不是序列级最优的一般结论。该文关于高效倍数的其他小实验也未外推为行业成本比例。 |
| [TRL DistillationTrainer 官方文档](https://huggingface.co/docs/trl/main/distillation_trainer) | 当前 `main` 文档称学生 on-policy completions 与教师 full next-token distribution 做 chunked JSD，避免完整 teacher logits 常驻；可用 vLLM，可见 async HTTP teacher 路径。当前页注明 `main` 需源码安装并指向稳定版 v1.16.0。 | 动态官方文档，不是固定 commit 代码；本轮未安装/运行。用户真实实现应 pin TRL/Transformers/vLLM revision，并查 loss、mask、logit processing 和 token normalization。 |
| [TRL GKDTrainer 官方文档](https://huggingface.co/docs/trl/gkd_trainer) | 当前文档 `trl.experimental.gkd` 将 `lmbda` 解释为学生自生成数据比例，`beta` 为 Generalized JSD 插值，另有 sequence KD 选项。 | API 与训练器代码可能变化。参数名不等同论文符号的全部语义；应用时检查固定版本的 config、数据源和 loss reduction。 |
| [Microsoft LMOps MiniLLM GitHub](https://github.com/microsoft/LMOps/tree/main/minillm) | 官方实现仓库含 training/eval/data-processing 脚本、teacher-mixed sampling 和模型并行。README 记录论文代码改动基于 Transformers 分支，实验采用 16×32 V100。 | 仅作为代码/数据入口和实现证据，不是现代框架的开箱即用 recipe；未下载、运行或宣称复现。 |

## 代表方法比较与诊断映射

| 被诊断的失败 | 代表方法 | 改动位置/机制 | 论文证据 | 后续验证和风险 |
| --- | --- | --- | --- | --- |
| 示范状态与学生部署历史错位 | DAgger → GKD/OPD | 让教师在学生 rollout 到达的状态上监督，而非只看固定参考轨迹 | GKD 在多个生成任务比较自生成状态比例和 divergence；Qwen3 公布两阶段强到弱蒸馏 | 需要确保生成数据确属当前/目标策略分布；学生偏离时 teacher 质量可能下降 |
| 长轨迹中早期错误改变后续状态，局部 KL 不给前面决策未来信用 | MiniLLM sequence-level objective | 用 sequence-level reverse KL 与 policy gradient return-to-go，并控制 teacher mixing/length | Dolly 结果和 length-normalization 消融；实验范围限早期指令蒸馏 | 序列梯度方差更高、重要性比率近似和长度 normalization 影响 estimator |
| sampled signal 重尾、近零 token 贡献低、熵快速下降 | REOPOLD | reward clipping、熵筛选 token、阶段切换 | 作者报告 sample efficiency 与 visual test-time scaling | clipping floor、entropy percentile 和 switch step 需独立消融；不能同时称 unbiased 与性能定律 |
| 教师高熵位置多解被 reverse-KL 压成单一 mode | EOPD | 以 teacher entropy 选择 reverse/forward KL | Qwen3 0.6/1.7/4B 的六个数学 benchmark Pass@8 增量已在正文按规模列举 | 高 entropy 不代表候选都正确；要一起看 Avg@k、Pass@k 和 verifier quality |
| 前缀逐渐脱离 teacher support 后，后段 token 被同权重学习 | IW-OPD / position weighting | 累积 teacher-student discrepancy 给后段 token 调权 | AIME'25 step 10 差距 +6.9pp、收敛时 +1.7pp 的范围已在正文表述 | discrepancy 是 proxy，不是精确 density-ratio correction；报告最终步、长度和种子 |
| 单 token log-ratio estimator 方差高 | vOPD | 用 action-independent detached baseline，top-k 仅近似 baseline 而非砍目标动作支持 | 作者报告不同 Qwen3/Olmo-3 设置的准确率、wall-clock | 降低局部 estimator 方差不自动带来 sequence-level temporal credit，也不验证答案正确 |
| 教师在学生离轨状态上错误或越过学生可信支持 | TrOPD / RG-OPD | 信赖区使用一种 loss，异常 token 改用 top-k forward KL；或按 verifier outcome gate teacher logits | 各论文针对特定任务报告准确率 | 阈值/判别器/verifier 本身会错；正例 outcome 不一定能定位正确 token |
| 同题多个 rollout 的成功和失败证据未被利用 | MOPD | 成功和失败 peer 同时进入 self-teacher context | AIME'24 与 LCBv6 mean@8 及 teacher-signal 对照 | 受成功 rollout 可得性、标签噪声、context 和 teacher compute 限制 |

上述映射是以论文机制与文中症状做的诊断索引，不是证明把多个方法串联就会叠加收益。每个新组件都应先明确目标失败，固定底模、教师、prompt、verifier、rollout 和预算，再用一项消融检验对应机制是否改变了行为。此处不得跨论文裸分排名。

## 原有问题/章节覆盖图

章节没有删除原有 17.0—17.23 主体问题；§17.20 的“研究进展如何读”已展开为研究问题轴、方法矩阵和行业/框架证据。逐项映射如下：

| 原知识/读者问题 | 处理层级 | 本轮位置 | 处理说明 |
| --- | --- | --- | --- |
| 17.0 离线 KD、exposure bias 与错误累积 | 核心 | §17.0 | 保留错误累积公式、LLM 具体错误状态，补 DAgger 源头及假设边界 |
| 17.1 OPD 解决/不解决什么，多来源混合 | 核心 | §17.1 | 保留；强调 on-policy 是状态来源，不代替可靠教师、verifier、安全和任务迁移 |
| 17.2 样本状态契约与版本追溯 | 核心 | §17.2 | 保留 task/student/teacher/verifier/environment/resource 字段 |
| 17.3 token、reverse-KL、序列/动作/artifact 目标 | 核心 | §17.3—17.3.3 | 保留多粒度目标和风险；补齐 local conditional vs sequence-level 区别 |
| 17.4 训练环路、教师查询和状态保存 | 核心 | §17.4 | 保留学生 rollout→teacher→verifier→update→independent eval 因果链 |
| 17.5 教师可靠性与硬权限边界 | 核心 | §17.5 | 保留可信度来源和 executor 权限不被 teacher override |
| 17.6 Coding Agent 工作示例 | 核心例子 | §17.6 | 保留工作区 patch、测试、错误恢复和环境 state 例子 |
| 17.7 教师触发、预算与调用成本 | 核心 | §17.7 | 保留按异常/置信/风险触发，补充失败恢复与单位成功成本 |
| 17.8 缓存、版本和隐私 | 核心工程风险 | §17.8 | 保留 cache identity、permission、删除请求与低熵哈希风险 |
| 17.9 KL→token reward 数学 | 核心且加深 | §17.9 | 保留原题意，修复单样本 loss 混淆；新增二动作完整手算和 score-function baseline 解释 |
| 17.10 长轨迹后段信号 | 核心 | §17.10 | 保留累计分歧、token weight 和位置偏差研究入口 |
| 17.11 trust region、熵、裁剪、采样、baseline | 核心 | §17.11 | 保留子机制并分别说明信号位置、目标变化和失败条件 |
| 17.12 多 rollout 与 verifier | 核心 | §17.12 | 保留并加 MOPD 的具体 peer-success/failure 机制与结果 |
| 17.13 数据/环境安全和 teacher API 成本 | 核心 | §17.13 | 保留脱敏、权限隔离、跨租户和 API 数据边界 |
| 17.14 学生恢复能力的评估 | 核心 | §17.14 | 保留独立新题、失败类别、恢复、成本和安全指标 |
| 17.15 成本收益 | 核心 | §17.15 | 保留定义和分母；Qwen3 与 Tinker 结果另作来源受限案例，不替代目标部署测量 |
| 17.16 轻量 teacher-call Python demo | 核心可执行示例 | §17.16 | 保留选择规则；cache key 纳入 confidence、failed tests、action 和 risk；新增超大整数检查 |
| 17.17 失败症状诊断表 | 核心 | §17.17 | 保留 replay、崩溃/退化、位置、调用触发、安全和多 seed 条目 |
| 17.18 SFT/offline KD/OPD/verifier/evaluation 阶段组合 | 核心 | §17.18 | 保留流程和逐阶段状态分布观测 |
| 17.19 imitation/outcome、安全、长度、成本冲突 | 核心 | §17.19 | 保留 multi-objective 向量，强调权限不是可抵消的软惩罚 |
| 17.20 研究进展阅读方法 | 核心扩展 | §17.20—17.20.1 | 从概念性研究概览改为解决问题、机制、具体结果、边界和工程实现表；新增 Qwen3、Tinker、TRL、MiniLLM repo evidence |
| 17.21 训练 stale rollout、RL 区别、teacher mismatch、触发率、能力评估、tokenizer | 核心 | §17.21.1—17.21.6 | 逐项保留；新增 policy lag 的显式解释已同步术语表 |
| 17.22 总结、17.23 来源入口 | 核心维护 | §17.22—17.23 | 总结纳入两阶段工业管线和方法层次；阅读入口按原研究、厂商报告、综述、工程文档分组 |
| 练习与题库迁移 | 支撑 | §17.24 / `EXERCISES.md` / `INTERVIEW_BANK.md` | 新增手算、时间信用、论文比较、Qwen3 表格差值、版本化框架审计练习和面试问答 |

## 公式、例子和数值复核

- 新增二动作例子：`p_S=(0.5,0.5)`、`p_T=(0.8,0.2)`；`r_A=ln(1.6)≈0.470`，`r_B=ln(0.4)≈-0.916`，`E_{pS}r≈-0.223`，`D_KL(pS||pT)≈0.223`；动作无关 baseline `b=-0.223` 得优势约 `(+0.693,-0.693)`。具体高低方向与 surrogate `-sg(r-b) log p_S(a)` 一致。
- Qwen3 表 21：OPD-RL AIME'24 `74.4-67.6=+6.8pp`，AIME'25 `65.5-55.5=+10.0pp`，MATH500 `+2.2pp`，LCB v5 `+7.4pp`，MMLU-Redux `+1.4pp`，GPQA `+2.0pp`；GPU-hours `17,920/1,800≈9.96×`。正文仅选择 AIME、LCB 与成本作窄表，报告完整四组表格的对应值并限定来源范围。
- MOPD：AIME'24 mean@8 比 Qwen3-4B base 高 `28.54-16.89=11.65pp`，比 SDPO 高 `21.25pp`；LCBv6 mean@8 比 base 高 `28.21pp`，比 SDPO 高 `8.17pp`。正文保留作者原数值和 benchmark 名称；不同 benchmark 的差值不互相比较。
- MiniLLM/GKD 的核心区分依综述 §2 与原论文：MiniLLM sequence-level reverse-KL temporal credit；GKD 的 student-generated prefix conditional matching/divergence selection。新增手算解释固定前缀的 KL 期望，不将其误说成序列级等价。
- 所有数值为论文/技术报告/工程文章结果，公式与手算为本地定义复核；未使用外部 LaTeX 渲染器或声称论文复现。

## 代码、链接与配套同步

- `code_status`：提取执行 §17.16 纯标准库 Python 教学 demo；执行结果与正文触发类别、cache key 变化和非法输入说明一致。只验证本地 demo，不代表训练器、teacher API、训练模型或生产 cache 系统。
- `link_status`：本章新增 DAgger、Qwen3、Tinker、TRL 与 Microsoft LMOps 链接均已通过相应原始/官方页面核对；其余既有方法链接按 17.20 对照表中 arXiv 原文入口核验；本地 research/chapter/book links 目标存在。
- `sync_status`：`PAPERS.md` 新增 OPD 专题阅读入口；`GLOSSARY_EN_ZH.md` 增 Student-Generated Rollout、Policy Lag、Sampled Log-Ratio Reward、Temporal Credit Assignment、GKD、MiniLLM、Multi-Rollout OPD，并区分 K2-Horizon 报告中的同名缩写；`INTERVIEW_BANK.md` 增 20H 研究比较题；`EXERCISES.md` 增第五册第十七章五项练习；本册目录已经有第十七章链接，无需改动。
- `validation`：逐节核对 §17.0—17.24 连续、标题无跳号；fenced code/math 配对，表格列数一致；检查本地相对链接、章节入口、数学例子与表格计算；运行 §17.16 demo；`git diff --check` 通过。未运行 TRL、真实 LLM、GPU、论文 benchmark、外部公式渲染或 PDF 构建。

## 遗留与证据边界

1. Qwen3 Table 21 为同一发布报告的模型/benchmark/GPU-hours 记录；硬件配置和完整 RL/OPD 预算细节不足以外推跨机构成本比，GPU-hours 也不是货币/总研发成本。
2. Tinker 的 QA eval 是自有评测，不能与公开 benchmark 或同行评审论文横向排名；“行为恢复”要在读者自己的模型、数据和任务切片上验证。
3. TRL 动态文档对应版本需 pin 到项目实际 commit；本轮只读文档，没有安装或执行 TRL trainer。
4. 除 §17.16 的 deterministic Python 调度 toy 外，没有在真实学生/教师、外部 API、GPU、verifier sandbox 或 Agent 环境训练 OPD。
5. REOPOLD 的 arXiv 与 Microsoft Research venue/title metadata 变化没有足够信息在本记录中归因；章节用原论文 ID/title 作引用，不声明源码未发布。方法结果按作者报告记录。
6. MOPD 成功/失败 peer 依赖现存成功样本和 verifier；对没有成功 rollout 的 prompt，它不能从自身生成数据中构造有效成功对照。
