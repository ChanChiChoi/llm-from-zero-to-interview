# 第八册第 7 章审计：Interpretability 与 Mechanistic Interpretability

审阅日期：2026-11-03。

状态：正文与配套资料的静态精修已完成；代码执行、GitHub Mermaid/MathJax 渲染和全章外链全量复核未完成，因此不记为完整门禁通过。未运行章节代码、测试、模型、benchmark 或 PDF 构建。

## 章节职责与衔接

- **中心问题：** 怎样从输出相关性逐步建立内部表示、因果干预和机制假设，并判断这些证据能支持什么安全结论。
- **读者收获：** 能区分归因、可读出性、因果干预和机制解释；手算 QK/OV 与 patching 指标；辨别 SAE 重构、稀疏度、feature 标签和行为因果证据的不同作用；把局部机制结果接回 holdout、行为回归及外部权限控制。
- **主线：** 黑箱问题与证据层次 → feature、head、circuit → activation patching、ablation 和 causal tracing → SAE 表示与评估 → 安全应用和机制边界 → grokking 与工程路径 → 受控案例和审计 demo。
- **书系衔接：** 承接第 6 章危险能力评估中“观测到行为后如何形成可检验解释”，为第 8 章 steering 与 representation engineering 提供干预证据边界。训练与参数编辑实现交第五册及第 9 章；评估统计交第七册。没有发现改变书册骨架、章节关系或本章目录的理由。

## 章级扫描与候选处置

检索日期 2026-11-03，重点时间窗 2025—2026。扫描问题：SAE 架构和评估的近期变化是否改变 feature 解释标准；是否有独立基准说明 SAE/patching 的相对能力边界；自动生成的 feature 解释怎样证伪；检测与因果控制是否分离；形式化保证、潜在知识诱发与安全用途分别能支持何种结论。原始来源、版本、阅读范围和证据边界见 `research/mechanistic-interpretability-evidence-2026-11-03.md`。

| 候选 | 证据与章节去向 | 决定 |
| --- | --- | --- |
| Gated SAE 与 sign-aware gated SAE | Gated SAE 是 shrinkage 与方向选择/幅度估计的架构改进；SA-GSAE 的 2026 arXiv v3 在 Pythia-1B、SmolLM3-3B 的六个设置报告双侧 gate 结果 | 纳入 SAE 方法演进，并限定模型、hook、$L_0$、重构和预印本边界；不把较低 dead-feature 比例等同语义正确或安全有效 |
| SAEBench 与 MIB | SAEBench 包含 8 项评估、7 种 SAE 架构和 200 多个 SAE；MIB 分为 circuit 与 causal-variable localization 两条 track | 纳入 §8.9 比较，说明 proxy 排名不能稳定预测下游结果，且不同任务的优胜方法不同；不归并成通用排名 |
| SANITY Checks for SAEs | 2026 arXiv v1 的独立重尾合成设置与随机基线结果挑战单看 explained variance 的做法，作者同时说明设定和架构覆盖限制 | 纳入 §8.9 作匹配基线提醒；保留独立 feature 假设、未覆盖 transcoder/crosscoder 等限制，不解读为 SAE 普遍失效 |
| DAME / Rethinking SAE Evaluation | NeurIPS 2026 接收稿以 dominance 分层检查高计算贡献 feature；LLM judge 得分存在分层差异 | 纳入 §8.9 和参考资料，说明抽样分层及 judge proxy 的边界；不把低四分位差异外推为所有模型规律 |
| SAE feature explanation 的反例检验 | NeurIPS 2025 工作使用相似负例修订和证伪自动解释 | 纳入 §8.8，补充近邻反例流程；自然语言标签仍是可反驳假设 |
| LLM Forensics | Findings of EMNLP 2026 在 1B/8B 的无害语言切换任务中分别测试 trigger detection 与干预效果 | 纳入 §9.2，突出检测不等于控制；不外推至恶意后门、工具调用或生产系统 |
| Formal Mechanistic Interpretability | ICLR 2026 在视觉模型上研究连续输入域鲁棒性、robust patching 与 circuit minimality | 纳入 §13.4，说明形式化保证绑定输入域、性质和验证器可扩展性；不称为前沿 LLM 安全保证 |
| MechELK 潜在知识诱发 | arXiv v1 提出 SAE feature、patching、CKS 和推理时表示干预的三阶段流程；作者整理的 DAB 及表格/正文汇总数存在未解释的算术冲突，arXiv 页面日期也与编号月份不一致 | **暂缓纳入正文。** 只在研究记录保存其方法、作者报告和不一致项；目前证据不足以把内部读出称为潜在信念或欺骗意图检测。后续出现可复核修订、独立复现并明确指标分母时再评估 |

扫描结论：候选研究加强了 SAE 质量评价、因果干预和安全用途边界；没有出现必须增加新讲次或改变章节骨架的成熟主题。

## 正文修改

- 在 §5.2 补上 induction-head 与 GPT-2 Small IOI 电路研究入口，并限定任务、模型和分析路径，避免把 head 名称写成普适功能标签。
- 在 §5.4 补一维 QK/OV 手算：分数 $[0,\ln 3]$ 得权重 $[0.25,0.75]$，value 加权为 8，投影系数 0.5 后写入 4；将第二个 value 置零后写入 0.25，差值 3.75。说明注意力权重不是最终因果贡献比例。
- §8 增加 SA-GSAE、SANITY、DAME 的机制、固定实验范围与限制；纠正 SAEBench 架构数为 7，避免把稀疏度/重构等 proxy 解释成下游效用。
- §8.8 增补相似负例和 feature 语义、行为相关性、安全因果性的区分；§9.2 将 LLM Forensics 引用改为 ACL Anthology 正式页面，并限于无害语言切换任务。
- §10.1 增加明确标注为教学假设值的 modular-addition progress-measure 轨迹：$p=7,a=4,b=5$ 得答案 2；候选方向读出 $0.15,0.35,0.68$，留出准确率 $50\%,50\%,75\%$。说明这些数不是论文结果，也未证明预警能力。
- §18 将代码状态改为“静态审阅，未运行（2026-11-03）”，删除无法追溯的旧运行声明；对全为 `None` 的 patch/path 分数增加过滤，按未定义指标、失败指标和全部达到 toy 阈值设置不同 next-actions，并补全静态复算、缺测分支、阈值舍入及真实实验边界。

## 篇幅与例子

汉字数由静态脚本逐单元计数：排除标题、表格、代码围栏、行内代码和公式，仅计解释性中文汉字；数字不是质量判断的替代。

| 独立单元 | 类型 | 汉字数 | 下限 | 状态 |
| --- | --- | ---: | ---: | --- |
| 开篇与 §§1—3 | 辅助 | 1,394 | 1,000 | 通过 |
| §§4—7 | 核心 | 3,222 | 1,800 | 通过 |
| §§8—9 | 核心 | 3,869 | 1,800 | 通过 |
| §§10—11 | 辅助 | 1,298 | 1,000 | 通过 |
| §§12—13 | 核心 | 1,937 | 1,800 | 通过 |
| §14 受控案例 | 辅助 | 1,511 | 1,000 | 通过 |
| §18 审计 toy demo 说明、推演与边界 | 辅助 | 1,119 | 1,000 | 通过 |

- **核心算例：** §5.4 的 QK/OV 手算输入、权重、加权 value、投影后写入和反事实差异均在正文给出；其数值是教学假设。
- **进展指标案例：** §10.1 用模 7 加法明确区分手造读出轨迹与任务准确率，不把示例写成论文测量或有效预警结论。
- **Demo：** §18 静态手算 patch 分数为 `0.4, 0.2, 0.8, 0.7`；ablation 最大差为 `1.6`；path patch 最大分为 `0.75`；SSE=`0.09`、SST≈`13.028`、重构分≈`0.993`；平均激活 feature 数为 `1.0`，toy top-2 purity 均为 `1.0`。七个示例阈值均通过，但输出仍要求 holdout 和行为副作用回归。代码未运行。

## 公式、代码、图示与链接

- 66 个围栏标记组成 33 对：1 Mermaid、11 text、20 math、1 Python；20 个 math 块的花括号计数为 57/57，未发现 `\\operatorname`。公式和 Mermaid 仅做源文件静态核对，GitHub MathJax/Mermaid 页面渲染未验收。
- §18 Python demo 中可执行语句本轮有改动，且没有可复用的运行记录；依用户持续的“不运行章节代码”约束，本轮只做静态审阅和手算，状态如实标注未运行。代码与正文承诺的一致性尚待允许运行后验证。
- 全章 30 个 Markdown 链接中 2 个为有效本地链接（第 6、8 章），另有 28 个外链；新增及修订的 SAEBench、MIB、SANITY、DAME、SA-GSAE、LLM Forensics、Formal MI 等重点来源已定向查阅，旧外链未逐一在线巡检。第 6、8 章链接在整册连续复读时补入，分别交代从外部行为证据到内部机制假设、从机制解释到推理时干预的转换。
- `git diff --check` 通过。

## 配套同步

- `PAPERS.md` 增加/更新 Gated SAE、SAEBench、DAME、SA-GSAE、SANITY、LLM Forensics 等阅读入口；未收录尚待复核的 MechELK。
- `EXERCISES.md` 扩展题 35EB，要求辨别 feature dominance 抽样和 DAME 的 LLM-judge 边界。
- `INTERVIEW_BANK.md` 补充高计算贡献 feature 与大模型范围不足的证据限定。
- `GLOSSARY_EN_ZH.md` 增补 Gated SAE、Feature Dominance、DAME；`KNOWLEDGE_GRAPH.md` 增加 SAE feature evaluation → feature dominance → DAME 的关系。
- 本章职责和目录未调整，无需改简介或目录；连续复读时补入第 6、8 章导航，见链接状态。

## 遗留风险与交接

- §18 demo 的 Python 语句发生修改但未执行；这阻止本章完整正文门禁通过，需在允许运行代码时只验证受影响代码块，并将实际命令、环境、输出和边界记录在本文件及正文状态标记中。
- GitHub Mermaid/MathJax 实际渲染和 28 个历史外链全量核验尚未完成；新增的第 6、8 章相对链接目标均存在。
- MechELK 暂缓纳入：arXiv 页面提交日与编号月份互相冲突；表 1 六项 accuracy 的算术均值与摘要/正文汇总不一致；其作者整理 DAB 和 latent-knowledge 定义也不支持把结果解释为欺骗意图检测。若未来修订版或独立复现澄清这些问题，再决定是否纳入。
- 第八册整册连续复读及最终门禁未完成。下一步按目录审阅 `book-08-ai-safety-alignment/chapters/08-steering与representation-engineering.md`，并检查第 7 章与第 8 章对“解释、干预、安全效果”的分工。

## 整册连续复读记录

- 2026-11-03 按目录完成第 7 章全篇连续复读。章节从输入归因和表示读出，推进到 patching、ablation、SAE、局部机制假设及安全用途；手算指标和 toy demo 静态预期与既有审计一致。它承接第 6 章的外部行为证据，并将可解释的表示/因果线索交给第 8 章评估推理时干预；解释、干预效果和系统安全边界保持区分。
- 未发现需改章节骨架、简介摘要、目录或顺序的问题。新增第 6 章回链及第 8 章前向链接；第 8 章开篇也补回第 7 章，明确机制证据是干预假设的来源而不是效果证明。未修改第 7 章代码或执行实验。
- 下一步按目录完整复读第 8 章《Steering 与 Representation Engineering》，重点检查机制解释到主动控制的边界，以及推理时 steering 与第 9 章持久参数编辑/遗忘的交接。
