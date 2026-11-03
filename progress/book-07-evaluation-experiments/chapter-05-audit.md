# 第五章审计：污染检测与可信评估

审阅日期：2026-11-03  
状态：定向扫描、正文精修和静态门禁完成；第七册及全套修书仍在进行中。

## 范围与章节职责

- 完整审阅 `book-07-evaluation-experiments/chapters/05-污染检测与可信评估.md`。本章负责解释评估信息如何经训练、调参、评估器、检索和反馈链路造成泄漏，如何检测可观察信号，以及如何设计 holdout 和报告残余风险；第 2 章负责 benchmark 构造全流程，第 4 章负责 LLM judge 的偏差校准，第 14 章负责更广的模型发布证据审计。
- 全文按一篇连续核心教学单元处理：污染威胁与血缘 → 重合/行为检测 → 时间、canary 与私有 holdout → 按任务报告风险。新增小节未改变章节边界或目录顺序。
- 未发现需要新增章节或调整第七册骨架的缺口。评估器与合成数据生成器的关系属于污染威胁模型，应与第 4 章 judge 偏差章节交叉链接；RL 阶段专用检测仍属于本章行为证据范围。

## 章级扫描与候选去向

扫描日期：2026-11-03；覆盖 benchmark overlap、黑盒原文/扰动检测、评估器独立性、RL 后训练检测以及未检出信号的解释。来源事实、阅读范围和证据边界见 `research/evaluation-design-uncertainty-and-contamination-2026.md`。

| 候选主题 | 证据与去向 | 决定与边界 |
| --- | --- | --- |
| Data Contamination Quiz（DCQ） | TACL 2025 正式论文；纳入 §8.6.7 | 加入原文与词级扰动选择、BDQ/BCQ 位置校正及逐字暴露边界。行为命中不等于训练来源证明，改写质量和检索暴露仍需控制。 |
| RL 阶段专用污染检测 | ICLR 2026 Self-Critique/RL-MIA；EMNLP 2026 LaRA；ICLR 2026 reasoning-detector fragility；纳入 §8.6.8 | 并列呈现 token 熵、逐层表示和检测信号被 GRPO/CoT 场景遮蔽的结果。各论文的模型、污染构造和接口不可互换，不能合并排名；LaRA 需要内部表示访问。 |
| Preference leakage | ICLR 2026 正式论文；纳入 §1.3、§3.7，并在第 4 章 §7.3 互链 | 将生成器与 judge 同模型、继承或同家族造成的学生模型偏好列入评估器相关风险。它不同于同一测试项进入训练，也不同于候选顺序偏差；闭源模型关系未知时需标明未知。 |
| BeyondBench 等生成式新题评测 | 既有 research 来源记录其算法/数学生成任务与 verifier 范围 | 不作为污染检测器加入本章方法节；其重点是特定结构任务的 benchmark 构造，与第 2 章设计和第 15 章专业任务簇相邻。 |

## 正文修改与例子

- 将原信息流文本换成 Mermaid 图，展示公开题、评估样本、RAG/cache、反馈回流和合成数据生成器—judge 关系。源码节点均加引号，文字箭头用管道标签；本轮人工检查了语法，未在 GitHub 页面确认实际渲染。
- 将原“评估器泄漏”细化为含合成数据血缘的评估器相关泄漏，并补充 preference leakage 机制、记录字段与第 4 章偏差治理交接。
- 增加可复算例：normalized exact match 的字符串/哈希路径；两组二词 shingle 的 Jaccard `2/6≈0.333`；经验分位数映射的 ConStat `q=0.80`、预期分数 `0.75`、`δ=0.15`；时间截止筛选；canary `1/5=0.20`；原题/扰动题 `0.91-0.52=0.39`。
- 修正 embedding 相似度公式中的禁用 `\operatorname`，改用 `\mathrm`。
- Python demo 的可执行语句未改。标题不再暗示本轮执行，代码块旁标注“静态审阅，未运行（2026-11-03）”，关键输出称为手算预期；按控制流人工核对 exact、near、cutoff、tuning、canary、perturbation 信号与决策动作。本轮未执行章节代码。

## 长度、公式、代码与链接

- 解释性汉字静态计数：9,806。按单篇连续核心单元计，高于 1,800 字下限；计数排除标题、表格、代码/文本/公式/Mermaid 围栏、行内代码、行内数学、链接和代码状态标记。
- 核心例子覆盖输入、计算步骤、结果和含义；所有数值均为教学示例，不是模型实测或通用阈值。
- 围栏共 50 个标记、25 对：11 个 math、12 个 text、1 个 Mermaid、1 个 Python block；围栏配对，math 花括号平衡，行内 `$` 共 12 个且配对。未发现 `\operatorname`。
- 本章共 30 个 Markdown 链接：29 个外链、1 个指向第 4 章的本地链接；本地目标存在。新增论文链接已对照 ACL Anthology、ICLR 官方页面或正式 OpenReview 论文核验；历史外链未逐项全量检查。
- Mermaid 源码经过人工语法检查；目标 GitHub 页面渲染未复核。闭源训练语料、后训练和线上数据可见性仍是证据边界，未检出不构成无污染证明。
- `git diff --check` 通过。未运行代码、测试、模型或 benchmark，未构建或更新 PDF。

## 配套同步

- `PAPERS.md` 增加 DCQ、Self-Critique/RL-MIA、检测脆弱性、LaRA 和 Preference Leakage 入口及适用边界。
- `GLOSSARY_EN_ZH.md` 增加 DCQ 与 Preference Leakage；`KNOWLEDGE_GRAPH.md` 增加污染审计链和生成器—judge 关系链。
- `EXERCISES.md` 增加 Jaccard、DCQ 位置校正、ConStat、RL 阶段检测和 judge 偏好泄漏练习。
- `INTERVIEW_BANK.md` 更新 overlap、Oren、ConStat、DCQ、RL 专用检测和 preference leakage 的比较题。
- 第 4 章 §7.3 增加 preference leakage 来源与第 5 章 §3.7 互链；本章简介、目录摘要和阅读顺序未变，无需同步。

## 下一步

第 5 章精修完成。下一步按序进入第七册第 6 章《Reasoning/Code/Math Eval》。第 6—15 章、整册连续复读和最终门禁尚未完成；全套书修书 goal 保持 active。
