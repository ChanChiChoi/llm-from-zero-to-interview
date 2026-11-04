# 第七章《Synthetic Data 与 Distillation Data》精修审计

审阅日期：2026-11-03

## 范围与章节判断

- 完整审阅 book-09-data-engineering/chapters/07-synthetic-data与distillation-data.md 开篇、§§0—22、公式、Mermaid、Python demo、参考来源和章节交叉链接；完成定向研究、静态精修与配套同步。
- 本章职责是把数据生成、teacher-student 信号、验证、混合比例和来源/风险治理连成数据工程流程。第 3、5 章分别提供通用质量筛选和 mixture/预算接口；偏好数据交第 8 章，版本、授权和治理交第 11 章。没有发现需要改变册级边界、章节顺序或分工的证据，未增删章节标题。
- 三个独立核心教学单元按论证连续性分组：开篇—§§0—3 建立来源/关系区分、损失和审计量；§§4—13 从数据类型、生成方式推进到 pipeline、验证、多样性和反馈风险；§§14—20 处理训练阶段、自然数据锚点、配比、分布编辑和实施决策。§19—20 是后一单元的决策收束，不单独承担完整新主题；§21—22 为来源和结语。

## 章级扫描与候选去向

- **日期与时间窗：** 2026-11-03；重点检查 2024—2026 年的多语言 teacher 评估、预训练改写比例与生成成本、样本正确性/过程验证、多样性和模型退化条件。扫描为定向检查，不声称穷尽。
- **研究记录：** Polyglot Teachers 的版本、阅读位置、统计定义和边界见 [multilingual-synthetic-teacher-selection-2026.md](../../research/multilingual-synthetic-teacher-selection-2026.md)；计入生成成本的 rewriting scaling law 见 [scaling-laws-data-mixtures-2026.md](../../research/scaling-laws-data-mixtures-2026.md)。本审计只记录纳入决定与正文处理。

| 候选方向 | 处置及理由 |
| --- | --- |
| 多语言 SFT teacher 选择 | 在 §8 纳入 Polyglot Teachers arXiv v3 的有界结果：区分“参数规模/benchmark 的回归不显著”“内在指标 PCA 解释方差”和“预测 student 的留出集 $R^2$”，不再合并成“数据特征普遍胜过模型规模”。六语种和评测设置限制外推。 |
| 计入重写生成成本的 synthetic-data scaling | 在 §16 纳入 Rewriting Scaling Laws arXiv v1 的 compute-limited 与 data-limited 框架，以及特定 FineWeb-Edu/FinePhrase、Paloma 和重写成本假设；给出作者报告的比例估计与高预算对照，并说明未计重写模型预训练成本。不是通用比例 recipe。 |
| 答案正确性与过程监督的分别验证 | 在 §11 新增教学假设例：40 道题中 34 道答案正确、其中 29 道过程有效；手算 34/40=0.85 与 29/40=0.725，说明 answer-only 与 process supervision 的准入目标不同。 |
| 合成多跳推理的规则数据 | 将 ICLR 2026 “Learning from Synthetic Data Improves Multi-hop Reasoning” 保留为候选入口；公开摘要描述规则生成的虚构知识与 RL 训练，但本章重点是数据管线和治理，现有规则/数学/推理轨迹已有讲解。未凭摘要扩写训练算法；如未来纳入，需读全文并确认新增数据工程机制。 |
| 数据退化与合成混合比例 | 保留现有的 EMNLP 2025、递归微调和 Collapse or Thrive 条件化说明；不把某篇论文比例写成门槛，也不将递归替换的风险外推到所有混合或单轮合成训练。 |

## 正文修改、篇幅与例子

- 收紧 Polyglot Teachers 的量化表述，并把引用固定到 arXiv v3；正文明确 93.3% 指论文报告的内在质量指标 PCA 方差解释率，$R^2=0.664$ 来自 student 表现回归的留出数据，不据此断言数据特征普遍优于模型规模。研究记录注明 §2.3 公式列四项聚合度量、Table 4 报告六个主成分，因此不推断 PCA 输入特征数。
- §16 新增 compute/data-limited 区分，交代预估重写比例、Paloma 四档对照及生成器成本估算不含预训练开销；source-backed 事实仅绑定该预印本的模型、seed、rewriter、任务和拟合。
- §11 新增假设性过程验证手算，区分最终答案标签与推理过程监督。假设值不是实测结论。
- **length_status：** 解释性汉字实计，排除标题、表格、代码、行内代码、数学公式、链接文本和空白：开篇—§§0—3 核心 **2,821**；§§4—13 核心 **2,906**；§§14—20 核心 **2,859**。三组均达到第 3.3 节 1,800 字核心下限；分组按“概念与目标—生成及验证闭环—训练配比与治理决策”的连续教学链确定。
- **example_status：** 手核 SFT token loss $0.768$、温度为 2 的 KL 蒸馏损失 $0.366$、质量代理分 $0.873$；审计 demo 的原始/保留 token 为 8,350/4,560，保留率约 0.546，synthetic/distillation 并集占 2,940/4,560≈0.645，distilled 标签占 780/4,560≈0.171，目标覆盖 10/10，残余风险 0/6，逐条重复标记率 1/12≈0.083。新样例 34/40=0.85、29/40=0.725。均为正文输入上的手算或教学假设，不是实验结果。

## 公式、代码、图示与链接

- **formula_status：** 11 个 math 围栏逐块校准数学下标大括号；源码花括号 136/136，未见未加大括号的数学下标或 operatorname。补明 keep 标记的取值域和 pairwise 重复率样本数。变量含义、分母和无定义边界已静态复核；未完成 GitHub MathJax 页面渲染。
- **code_status：** §18.1 旧“已验证（2026-10-28）”在本册/本章进度中找不到可追溯执行记录；相邻 PAPERS.md 还指向不存在的旧 research 维护文件。本轮将状态改为“静态审阅，未运行（2026-11-03；Python 3.12.9，标准库）”，补运行方法，移除旧精确运行输出，改列手算预期值。逐块与 HEAD 对照，Python 可执行代码 17,186 字符完全相同；未运行章节代码、断言、测试、模型或 benchmark。
- **visual_status：** 原有 1 个 Mermaid 流程图未改；围栏配对通过静态检查，GitHub 页面渲染未验收。
- **link_status：** 全章 26 个 Markdown 链接，3 个本地目标均存在，23 个外链未做全量巡检。新增/更新的 arXiv v1、v3 和 EMNLP 链接已核对正式来源。
- **sync_status：** 更新 PAPERS.md 中第九册配比和第七章条目，固定研究版本、加入成本边界，并删除指向不存在 research 维护文件的链接。新增多语言 teacher 来源记录并补写已有 scaling-law 研究笔记。章节职责和目录未改变，因此未改简介、目录、plan.md 或题库/术语/知识图谱。

## 未关闭事项与交接

- 当前章节审计只覆盖静态内容；GitHub Mermaid/MathJax 目标页面渲染、代码动态运行和既有外链全量巡检仍未完成。按当前范围不运行代码、测试、模型、benchmark 或 PDF 构建。
- 第九册连续复读按既定顺序在第十二章后进行；第八至十二章仍须逐章精修。
- **下一步：** 按目录进入第八章《Preference Data 与安全数据》，重新按当前任务阶段读取模块 02，并在定向研究、公式/图表/代码及维护验收阶段加载模块 03—05。
