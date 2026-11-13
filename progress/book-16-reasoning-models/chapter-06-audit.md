# 第六章《搜索与 Tree-of-Thought》精修记录

审阅日期：2026-11-12  
状态：章节全文审阅、定向扫描、正文修改和单篇门禁完成；不更新 PDF。

## scope

- 完整复读第六章并对照第十六册契约，保留“单链局限 → 状态与动作 → ToT/搜索策略 → 环境反馈 → 评分与剪枝 → 成本/评估/安全”的主线；未拆章或改编号。
- 复核第三章多候选聚合、第四章 verifier / reward model、第五章过程监督的前置关系，以及第七章测试时计算和第十六章并行推理的交接。
- 修复 8 升与 5 升水壶的无效示例路径；正文状态表现示两个有效 3 升终态，并明确另一条路径实际留下 2 升。教学 demo 的 `water_jug` 金标准同步为 3。
- 把固定候选上的单层比较从“MCTS 实验”收窄为 beam top-1 与 UCT 式选择，明确 demo 未实现完整的 selection / expansion / simulation / backpropagation，避免把预制四题结果写成完整算法比较。

## chapter_scan

- **日期与范围：** 2026-11-12；围绕 thought 搜索、候选状态、beam / best-first / A* / MCTS/UCT、环境反馈、剪枝和搜索评估定向复核。检查窗口覆盖经典搜索论文、ToT/LATS 原始论文及 2025—2026 test-time scaling 分类和并行推理来源。
- **核验来源：** ToT arXiv v2；LATS arXiv v3；Hart、Nilsson 与 Raphael 的 A* 原始论文 DOI；Kocsis 与 Szepesvári 的 UCT 原始论文 DOI；Browne 等人的 MCTS 综述 DOI；AlphaZero arXiv v1；Self-Consistency arXiv v4；Let's Verify Step by Step arXiv v1；Hariri 等人的 test-time scaling 分类 arXiv v2。正文将原论文实验结论限定于对应模型、任务和预算条件。
- **候选主题及去向：** 2026 年 inference-regime 分类将完整候选聚合与前缀搜索分别称为 leaf-level 与 prefix-level scaling；正文用一段连接 Self-Consistency、Search 与第七章，说明该分类是描述预算施加位置而非互斥算法。PaCoRe、ParaThinker 等并行推理工作已在既有 `research/test-time-compute-and-evaluation-2026.md` 中记录；本章不展开多路径协调，交给第十六章。Snell 与 adaptive allocation 研究交给第七/十八章分析。未发现需要变更第十六册骨架或增设独立章节的候选。
- **资料纠错：** 原资料索引将 arXiv:2007.14554 标成 MCTS review；该标识实际对应量子多假设检验。已删除错链，改为 Browne 等人的 *A Survey of Monte Carlo Tree Search Methods* DOI，并补充 UCT 与 A* 原始来源；修正 AlphaZero 论文题名及 LATS 版本号。

## length_status

- 按一个连续核心教学单元统计 §0—19；排除标题、围栏内代码/文本/公式、表格、行内代码、Markdown 链接及其目标、状态标记和练习/总结/资料索引后，解释性正文实计 **11,749 个汉字**，高于核心单元 1,800 字下限。计数仅是门槛，不替代机制、证据与例子审阅。

## evidence

- ToT 研究背景引用 arXiv:2305.10601v2，并把 Game of 24、创意写作和 Mini Crosswords 结果限定在原论文设置。
- LATS 引用固定 v3，说明其环境反馈和 agent/value/reflection 设计属于特定研究方法，不推断闭源产品内部实现。
- MCTS 四阶段引用 Browne 等人的 2012 综述；UCT 平滑选择公式与经典假设引用 Kocsis 与 Szepesvári 原始论文。正文明确 `+1` 是本章教学平滑约定，并与未访问节点置为无穷大的惯例区分。
- A* 的 `g+h` 与启发式条件引用 Hart、Nilsson 与 Raphael 原始论文；PRM 分数未被当作可采纳的剩余代价估计。
- AlphaZero、Self-Consistency、过程监督的论文实验均保留任务、模型与方法边界。2026 inference-regime taxonomy 明确标为描述性分类，不作为收益证据。
- 未新增 `research/` 文件；新发现的书籍审阅与修改决定只记录于本目录。

## example_status

- 水壶状态以 `(8 升壶, 5 升壶)` 表示，逐步列出 `(0,0) → (8,0) → (3,5) → (3,0) → (0,3)`；另一条路径终态为 `(3,0)`，错误路径留下 `(8,2)`。
- 树大小以 `B=4,D=6` 手算 5,461 个节点；有效分支、根节点/子节点、token、工具美元账本及 token/成功均有代入值，并明确单位不能直接混加。
- ToT frontier 和 beam top-k 使用具体候选及分数；路径平均分与 `γ=0.5` 折扣分、A* 两节点 `g+h`、同状态路径合并和 UCT 三个子节点分数均逐步复算。
- 环境观察用合同第 7 页与 6% 税率演示状态如何写入证据版本；组合评分、top-score margin、剪枝错误率、支配剪枝、多样性/重复率、准确率差和正确路径覆盖率均给出具体输入与结果。
- 四题 demo 的 0.25/0.75/1.0 及其平均多样性、16 个预制节点、980 候选 token、326.667 token/beam 成功题、剪枝任务和 UCT 选择均被明确标为玩具数据。

## code_status

- §17 的独立 Python demo 已按修改后的内容从 Markdown 提取并运行。
- **状态：已验证。** 2026-11-12，Python 3.12.9，标准库。实际 stdout 与正文预期输出逐字匹配；8 项数值边界断言通过：浮点可表示的大整数、超大整数、bool、NaN、无穷、普通有限小数、超大整数成本平稳拒绝、980/3 单位成本舍入。另确认 `hard_negative` 三个 UCT 分数为 `[0.606, 1.155, 1.143]`。
- `finite_number` 先拒绝 bool 和非数值，再尝试转成有限 float，并捕获超范围整数的 `OverflowError`，避免验证器意外崩溃。代码只读取预置 toy 候选，不调用模型、API、工具或生产环境。
- 代码状态标记紧邻代码块；完整 MCTS 不在代码验证范围内，正文已明示。

## formula_status

- 27 个 display-math 公式均检查了围栏、变量、定义域和正文解释；所有原 `\operatorname` 已替换为 `\mathrm`。
- 搜索规模、有效分支、资源账本、ToT frontier、beam、路径聚合、A*、UCT、状态合并、环境转移、组合评分、margin、剪枝、支配、diversity/dup、accuracy、增益与路径覆盖均有手算输入或具体状态案例。
- `C_success` 说明仅能对同单位成本求和；不同资源无可靠换算率时使用资源向量。

## link_status

- 本章 26 个 Markdown 链接中的 5 个本地链接均存在，指向第三、四、五、七和十六章；本章、简介、目录的 Markdown 围栏均成对。
- 简介 33 个本地链接、目录 51 个本地链接均检查通过。核心 arXiv 与 DOI 论文入口已在浏览核对；未对仓库全部外链做批量 HTTP 爬取。
- 移除了作者联网访问过程文字；资料索引固定到具体版本或正式 DOI。

## visual_status

- 新增水壶状态轨迹表，逐步呈现动作和 `(8 升壶,5 升壶)` 状态。
- 新增 ToT、beam、best-first/A*、MCTS/UCT 方法层次与适用边界表。
- 关系由状态表、方法表和数值例子充分表达；未用 Mermaid。

## sync_status

- 已更新第十六册简介第六章摘要，以及目录的推荐路径和全部章节导航中的第六章描述。
- 正文入口明确回链第三至第五章，并交接第七章预算分析和第十六章并行推理；章节编号、目录结构未变化。
- 未修改 `plan.md`，未更新 PDF。

## open_risks

- demo 只有四个预制任务和一层固定候选；UCT 评分不是完整 MCTS，也没有真实模型或线上成本测量。正文和审计均明确此边界。
- 2026 inference-regime taxonomy 是预印本中的分析分类；正文仅用于说明 leaf-level 聚合与 prefix-level 搜索的区别，不视为穷尽分类或方法效果结论。
- 论文实验数据均依各自任务和配置；本章未声称在统一基准上比较 ToT、LATS、A*、beam 与 MCTS。

## gate

章节扫描、核心篇幅、具体状态推演、公式手算、证据边界、代码执行与边界检查、本地链接、视觉机会和简介/目录同步均通过。第六章精修完成；下一入口为第七章《推理时计算扩展》。
