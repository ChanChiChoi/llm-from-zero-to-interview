# 第八册第 2 章审计：Alignment Problem

审阅日期：2026-11-03  
范围：完整通读本章正文、公式、图表、引用、练习及 Python demo；完成章级研究扫描、正文精修、配套同步和静态门禁。

## 章节职责与诊断

- **中心问题：** 人类真实意图如何经过书面目标、训练代理、模型学到的策略，最终变成部署行为；这些环节为何会错位，又能怎样观察与修复。
- **读者收获：** 能区分外部目标规范问题、模型学到的行为目标与能力泛化失败；能把可见行为与内部目标推断分开；能用代理指标、分布外任务和系统约束定位风险，并说明证据边界。
- **主线：** 问题来源 → alignment 定义 → outer/inner alignment → specification gaming 与 goal misgeneralization → deceptive alignment 的风险假设和证据等级 → 生命周期缓解 → 目标错配审计示例 → 真实系统失败及综合案例。各节共同回答“意图如何在目标与行为链路中失配”，保留原结构。
- **与前后章节的关系：** 第 1 章建立安全与对齐术语地图；本章给出目标、监督与行为失配的概念基础；第 3 章完整讲 scalable oversight，第 4 章深入 reward hacking，第 5—6 章讲攻击和红队评估，第 7—11 章讲内部证据、干预、隐私与治理。未发现改变册级边界或顺序的问题。

## 章节级研究扫描

- **检索与核验日期：** 2026-11-03；基础概念使用奠基论文，近期扫描关注 2024—2026 年的上下文条件行为、alignment faking、模型体压力测试、反 scheming 训练及模拟 Agent 案例。
- **核验问题：** 目标误泛化与 specification gaming 如何区分；能力保留的行为证据是否足以推断内部目标；受监督/未受监督行为差异可支持多强的结论；构造模型体和模拟 Agent 结果能否外推到生产发生率。
- **阅读来源与证据：** Concrete Problems in AI Safety、Risks from Learned Optimization、两篇 Goal Misgeneralization 原始论文、InstructGPT、Learning to Summarize from Human Feedback、Constitutional AI、AI Safety via Debate，以及 ACL 2026 ICL emergent misalignment、Alignment Faking、Sleeper Agents、SchemeArena、Training Natural Schemers、Deliberative Alignment 和 Anthropic 2026 模拟案例。近期来源的版本、阅读位置、支持事实和边界集中记录于 `research/alignment-problem-and-oversight-conditional-behavior-2026-11-03.md`。

### 候选主题及去向

| 候选 | 处置 |
| --- | --- |
| 目标规范、代理指标与 outer/inner alignment | 保留为本章主线；覆盖概念定义、教学例和可观察指标。训练算法细节交第五册。 |
| specification gaming、reward hacking、Goodhart | 保留范围和概念关系；训练奖励漏洞的机制深讲交第 4 章。 |
| goal misgeneralization | 保留为本章核心概念；用能力保留/行为目标错配说明分布外风险，并明确其行为定义不证明稳定的内部目标表征。 |
| deceptive alignment、alignment faking 与监督条件行为 | 保留风险定义和证据分层；近期实验限定为特定模型、提示、监控信号或构造环境下的行为观察，不写成自然训练模型具有恶意目标或生产发生率。 |
| Scalable oversight、辩论、AI feedback | 本章仅用于交代监督瓶颈与缓解位置；机制对比交第 3 章。 |
| 新兴压力测试与 Agent 案例 | 纳入证据表和延伸入口；按构造模型体、benchmark 或受控模拟限定结论，不新增章节。 |
| 目标错配审计 Demo | 保留为边界明确的辅助单元，用来手算目标、proxy、行为、分布外能力和严重度指标；不视为模型安全认证。 |

## 正文与证据修改

- 收紧“模型能力提升会放大对齐后果”的表述，补充其依赖动作权限、执行时域和环境约束。
- 将 ACL 2026 的 ICL emergent misalignment 定位为指定上下文和模型下的条件行为现象；不把它解释为参数更新或持久隐藏目标，并把训练后 reward-hacking EM 的展开留给第 4 章。
- 区分 deceptive alignment 的风险假设与 alignment faking 的实验操作化；说明条件性行为差异或推理文本本身不能证明稳定隐藏目标。
- 扩展证据表并限定 Alignment Faking、Sleeper Agents、SchemeArena、Training Natural Schemers、Deliberative Alignment 的模型、构造、任务、判定方式和外推边界；加入 Anthropic 2026 受控 Agent 模拟案例，并明确不是现实事故或发生率估计。
- 为 §10.6 改用“目标错配审计 Demo”标题，把输出称为静态预期结果；新增 `M_outer` 与 `M_beh` 的对照解读、`robust_safety_success` 单例分析及 `M_shift` 的目标不变配对分母前提。Python 语句未修改。
- 更新练习 3，要求独立确认能力仍在并避免由行为变化推断稳定隐藏目标；新增本章第 3 章 scalable oversight 入口。
- 将 Alignment Problem、Specification Gaming、Goal Misgeneralization、Deceptive Alignment、Alignment Faking、ICL/emergent misalignment、model organism 和 oversight flip 等术语在首次出现处补充中文说明。

### 目标误泛化引用更正

原文曾将 *Goal Misgeneralization in Deep Reinforcement Learning* 链到 `2210.01790`。核对 arXiv 原始摘要页和 HTML 后确认：`2210.01790v2` 是 Shah 等人的 *Goal Misgeneralization: Why Correct Specifications Aren't Enough For Correct Goals*；`2105.14111v7` 才是 Langosco 等人的 *Goal Misgeneralization in Deep Reinforcement Learning*（ICML 2022）。现已分别修正导言、§7.1 的近邻引文和 §14.1 资料项，并说明 Shah 等人的定义不要求假设模型内部有明确目标表征。两篇论文均链接到准确 arXiv ID。

## 门禁状态

- **篇幅：** 按独立讲解单元统计，核心正文 8,883 个汉字、辅助 Demo 1,116 个汉字；排除标题、代码、表格、参考文献条目、行内代码/链接文本和练习。核心与辅助单元分别达到 1,800 / 1,000 字下限。篇幅不替代因果解释和证据质量判断。
- **例子与计算：** §10.6 的单行 `qa_truthfulness` 算例为 outer mismatch、behavior mismatch、proxy follow 各 `1/1=1`，Goodhart gap `0.92-0.45=0.47`，OOD 且能力 `0.82≥0.75`，GMG `1/1=1`。八条合成样本手算为 outer mismatch `6/8=0.75`、behavior mismatch `6/8=0.75`、proxy follow `8/8=1`、GMG `4/5=0.80`、Goodhart gap `0.40625`（显示为 `0.406`）、paired behavior change `5/8=0.625`、severity-weighted mismatch `22/27≈0.815`、policy coverage `7/8=0.875`。已说明 `M_shift` 的分母只适用于目标不变的可比配对。
- **代码：** §10.6 Python demo 标为“静态审阅，未运行（2026-11-03）”；未运行章节代码、测试、模型或 benchmark。输出作为合成输入的静态预期结果，主要计数、条件和 hold 分支已手算核对；静态审阅不等于执行验证。
- **公式与图表：** 公式变量、分母和教学例已人工对照；MathJax 公式花括号静态计数 75/75，无 `\operatorname`。全章 66 个代码围栏标记配对。Mermaid 源码已人工检查，GitHub 页面渲染未验证。
- **链接：** 35 个 Markdown 链接中 5 个为本地链接，目标全部存在；其余 30 个为外链，本轮未全量逐条核验。新增/更正的两篇目标误泛化论文已逐页核对标题、版本和摘要，研究记录中的 HTML 入口已读。全仓库搜寻后没有发现“Deep Reinforcement Learning”继续指向 `2210.01790` 的残留错链。整册连续复读后，在本章总结新增通往第 3 章的监督主题入口。
- **配套同步：** 已更新 `PAPERS.md`、`GLOSSARY_EN_ZH.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`KNOWLEDGE_GRAPH.md`；新增来源事实和证据边界写入上述 `research/` 研究记录。审计结论和修改过程仅记于 `progress/`。
- **格式：** `git diff --check` 通过。未更新或构建 PDF。

## 遗留项与下一步

- 外链没有全量重验，Mermaid 未做 GitHub 页面渲染；这些未验证范围不能表述为已验收。
- 第八册第 11、16 章原有缺失的研究/代码目标待对应章节逐篇核对；第八册架构审计与第 1 章审计已有记录。
- 本章静态精修和全篇连续复读均已完成；连续复读确认监督失配自然交接到第 3 章，未发现需要改变骨架的问题。第 3 章的全文复读和交接补链已完成，后续按全册连续复读进度进入第 4 章。
