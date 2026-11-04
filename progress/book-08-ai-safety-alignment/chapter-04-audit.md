# 第八册第 4 章审计：Reward Hacking 与目标错配

审阅日期：2026-11-03。
范围：全文诊断、章级扫描、正文精修、来源核验、例子与公式静态复核、配套同步。未运行章节代码、测试、模型、benchmark 或 PDF 构建。

## 章节职责与衔接

- **中心问题：** 当训练或评估优化可测量的奖励、偏好或规则指标时，代理信号怎样偏离真实目标，模型与整套反馈流程会以什么方式利用这种差异，团队又怎样识别、缓解并把结果转成有证据的发布动作。
- **读者收获：** 能区分 reward hacking、specification gaming、Goodhart 现象和更广义的目标错配；能解释 RM 的 pairwise 代理目标、policy 分布迁移、RL / best-of-N 过度优化、KL 限制的边界；能设计具有独立 gold、风险切片和人工升级的审计。
- **主线：** 真实目标与 proxy 的定义 → LLM 常见代理漏洞 → reward model 的训练与失效 → RLHF、DPO、RLAIF 与最佳候选筛选中的优化压力 → 跨任务行为变化与 grader 敏感度证据 → 检测、缓解和 RAG 应用案例。
- **书系衔接：** 第 2 章定义 alignment problem、emergent misalignment、goal misgeneralization 与隐藏目标推断的概念边界；第 3 章讲监督怎样扩大及校准；本章承接 reward proxy 被优化后的风险；第 5—6 章继续讨论 jailbreak、prompt injection、red teaming 与危险能力评估。未发现需要返回册级架构调整的问题。

## 章级扫描与候选处置

核验日期为 2026-11-03，重点查看 reward model / policy 的分布迁移、overoptimization、best-of-N、DPO / RLAIF 的偏好代理、reward hacking 后的跨任务行为变化，以及 grader 信念敏感度测量。新增来源的版本、阅读范围及证据边界见 `research/reward-hacking-cross-task-generalization-2026-11-03.md`；RewardBench、Gao 等人的 overoptimization 研究及偏好优化方法的既有记录分别见相关 reward-model 与 RLHF research notes。

| 候选主题 | 去向与理由 |
| --- | --- |
| LLM 中的长度、自信、安全模板、RAG 引用、benchmark、代码评测和 judge gaming | 保留于 §4，作为共同 proxy 失配路径的具体表现；扩展配对反事实、任务控制、claim-level support 和评测器隔离边界。 |
| Reward model 的 pairwise 目标、有限数据、标签分歧和分布迁移 | §5 主讲；补充 Bradley–Terry 概率的手算、分数平移不变性、偏好候选采样链和 RM 用途对应的验收条件。 |
| Reward model overoptimization、best-of-N 与 KL 的关系 | §6 主讲；结合人类摘要实验、合成 gold-RM 研究和独立教学算例，明确不同来源与优化设置不可合并为通用阈值。 |
| RLHF、DPO、RLAIF 的代理信号差异 | §7 保留比较；补充 DPO logit 手算、AI feedback 共同盲点，以及 policy / 固定偏好数据 / 在线 judge 的不同分布风险。 |
| Reward hacking 后的 emergent misalignment | 保留于 §7.5，以 MacDiarmid et al. 和 Jose & Stastny 的研究设置作证据案例；明确“production RL”环境含义、额外提供漏洞知识、单模型/环境范围及产品发生率边界。 |
| Contrastive SDF / reward-seeking | 保留于 §8.5；解释 `87%-9%=78` 个百分点及 early checkpoint 的 `40%-24%=16` 个百分点，补样本量、credible interval、reward-hacker organism 结果、操纵检查和单 lineage / 短程编码限制。 |
| RAG citation-presence 奖励失效 | 保留于 §12 应用案例，新增四条 claim 的引用存在率、当前版本支持率与高风险动作的手算。 |
| 跨模型统一发生率、稳定隐藏目标或通用 KL / 阈值结论 | 不纳入正文断言；现有来源不支持这些外推，继续保留证据边界。 |

## 正文修改

- §3 补足 `u_i` 作为独立 gold 近似而非绝对真值的边界，统一 proxy mismatch、reward hacking rate、reward-gold gap、优化曲线、长度相关与 KL 公式的变量和分母说明；保留候选回答手算。
- §4 按长度、自信、安全、引用、benchmark、代码和 judge 等路径补充反事实配对、风险分类、版本、工具副作用及反馈闭环解释，避免把单一分数变化直接判为模型“故意作弊”。
- §5 解释 RM 的 pairwise 目标只约束已见比较，不赋予绝对分数跨任务含义；手算 `σ(1)≈0.731`、单样本 NLL `≈0.313`，说明同加常数不改变偏好概率；补齐标签分歧、留出候选和分布漂移边界。
- §6 补入 Stiennon 等人的指定摘要设置与 Gao 等人的 6B 合成 gold-RM 设置，说明人类偏好实验与合成研究各自范围；用 `1%` 漏洞候选、`N=1/50` 手算筛选概率与期望 gold，另用三个 checkpoint 手算 `G_over` 从 `-0.05` 到 `0.22`。例子均为教学值，不是模型实测或通用阈值。
- §7 说明 RLHF、DPO、RLAIF 的偏好来源和优化循环；DPO 例中 `β=1` 时 logit 差 `0.5`、概率约 `0.622`，`β=2` 时约 `0.731`，仅示意损失尺度变化。正文对 SDF 论文修正为“该实验中的 inoculation prompting 阻止了所测 EM 泛化”，不外推到所有 SDF。
- §8 增加固定 prompt / 解码 / gold 版本、分层抽样、配对比较、共享依赖检查与根因定位流程；Contrastive SDF 例保留 `n≈300` 和 95% Bayesian credible intervals 的来源说明，并明确不是部署风险率或隐藏目标读数。
- §9 将缓解路径连接到具体失败环节和可证伪复测条件；加入 `2/100` 总体率与高风险 `2/2` 切片例，说明风险分层不能被低风险成功样本稀释。
- §9.7 将“最小可运行”改为中性 demo 标题，将旧“已验证”改为“静态审阅，未运行（2026-11-03）”，并将“运行后会看到”改成固定输入的静态预期输出。代码语句未改；加入数据流解释、门槛优先级、限制及输入变化建议。
- §12 加入四条 claim 的合成例：citation presence `4/4=1.00`，直接支持中有一条引用废止版本，因此当前版本有效支持 `1/4=0.25`；高风险 unsupported claim 独立升级，不由平均值抵消。

## 篇幅与教学例状态

静态计数排除标题、代码、围栏、表格、引用链接文字和空白，仅计汉字；解释性文字质量仍按人工审阅判断。

| 独立单元 | 类型 | 汉字数 | 下限 | 状态 |
| --- | --- | ---: | ---: | --- |
| §3 Reward Hacking、Specification Gaming、Goodhart 与目标错配 | 核心 | 2,060 | 1,800 | 通过 |
| §4 LLM 常见 reward hacking 模式 | 核心 | 2,574 | 1,800 | 通过 |
| §5 Reward Model 作为 proxy 及其失效 | 核心 | 1,838 | 1,800 | 通过 |
| §6 Reward Model Overoptimization | 核心 | 1,911 | 1,800 | 通过 |
| §7 RLHF、DPO、RLAIF 与跨任务证据 | 核心 | 2,010 | 1,800 | 通过 |
| §8 Reward Hacking 的发现与测量 | 核心 | 1,827 | 1,800 | 通过 |
| §9.1—§9.6 缓解与上线决策 | 核心 | 1,829 | 1,800 | 通过 |
| §9.7 奖励黑客审计 demo | 辅助 | 1,122 | 1,000 | 通过 |
| §12 RAG 引用奖励案例 | 辅助 | 1,130 | 1,000 | 通过 |

§1—§2 是历史背景与直觉类比，引出 §3 的定义和形式化审计；§10—§11、§13—§16 是项目检查清单、章节导航、误区、参考资料、练习和总结，不作为独立方法正文。核心例子的输入、逐步计算和证据边界已在正文列明；demo 与 RAG 案例中的数值均为合成教学输入。

## 公式、代码、链接与配套同步

- 全章 66 个围栏标记配成 33 对，其中 14 个 math 块、1 个 Python 块；公式花括号计数 82/82，无 `\operatorname`。公式经静态结构检查，未在 GitHub 页面检查 MathJax 渲染。
- Python demo 位于 §9.7，代码语句本轮未修改、未执行。原标记声称 2026-10-27 已验证，但 `progress/` 中没有可复核的命令、输入范围、实际输出和边界记录；因此本轮不沿用“已验证”。当前固定 8 个 case 的主要预期量经手算复核：mismatch / hacking rate 均为 `7/8=0.875`；平均 proxy `0.894`、gold `0.459`、gap `0.435`；Pearson 长度相关约 `0.539`；严重度加权率 `27/28≈0.964`；最终预期动作是 `hold_for_high_severity_review`。这些不是程序运行输出。
- 21 个 Markdown 链接中有 3 个本地链接（第 2、3、5 章）且目标有效，18 个外链未对历史来源做全量巡检；三篇新增/重点研究原文已核对其版本、方法/结果和 limitations。无 Mermaid 图。整册连续复读时在相邻概念关系处补入第 3 章监督方法入口，并在总结区分优化代理漏洞与第 5 章输入信任边界风险。
- 同步 `EXERCISES.md`（68A）、`INTERVIEW_BANK.md`（44）和 `KNOWLEDGE_GRAPH.md`（reward-seeking → contrastive belief update → grader-sensitivity evaluation）；`GLOSSARY_EN_ZH.md` 已有 Reward-Seeking、SDF 与 Contrastive Belief Update 定义，`PAPERS.md` 已含本章来源，因此无需重复添加。章节职责与文件结构未变化，简介和目录摘要无需修改。
- `git diff --check` 通过。未运行 Python demo、测试、模型、benchmark 或 PDF，也未更新 PDF。

## 遗留风险与交接

- 本章旧外链未全量在线核验；高优先级研究来源已定向核验。MathJax 在目标网页的实际渲染未验收。
- demo 仍是未运行代码；静态手算不等于语法或边界运行验证。后续若执行代码语句、输入输出约定或依赖发生变化，按模块 04/05 的规则重新验证受影响范围。
- 本章精修与配套同步完成；整册连续复读确认第 3 章的监督代理问题在本章落到 reward model 与优化风险。与第 5 章的输入信任边界风险属于不同机制；已在总结处明确这一区分并补入相邻章节链接。未发现章节骨架、简介/目录摘要需要调整的问题。
- 下一步按目录继续连续复读 `book-08-ai-safety-alignment/chapters/05-jailbreak与prompt-injection.md`；整册安全对齐书的连续复读和最终门禁仍待完成，全套书精修 goal 保持 active。
