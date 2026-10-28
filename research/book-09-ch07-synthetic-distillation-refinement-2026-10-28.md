# 第九册第七章 Synthetic Data 与 Distillation Data 研究与精修记录

## 复核范围与章节职责

按当前 `WRITING_SPEC.md` 从头复读本章，结合第九册简介、目录、书册契约和前六章职责，完成本章定向扫描、正文修改、数例复算、固定 Python 示例执行和配套资料同步。此前版本只作为研究线索，不作为本轮改稿或验收证据。

本章保留单篇结构，回答“合成与蒸馏数据何时能帮助训练，如何以可追溯、可验证的方式构造和评估”。第五章负责一般数据池配比与样本选择，第六章负责代码、数学和领域数据的类型专项验证；本章聚焦来源构造、教师信号、生成版本、任务验证、合成并集比例和递归训练边界。未发现需要新增独立章节的主题。

## 章级研究扫描与候选去向

- 复核 Self-Instruct、WizardLM/Evol-Instruct、phi-1、Orca、Distilling Step-by-Step 和经典 teacher-student 蒸馏作为方法谱系入口；保留其生成/监督机制，结论限定在论文的 seed、模型、任务、过滤和实验配置。
- 阅读 EMNLP 2024 的 Quality Matters。该研究以人工定义的正确性条件和模型辅助审查评估工具调用数据；作者在所测两个 benchmark 和训练设置中报告，经过验证的较少样本优于较多未验证样本。正文只据此支持“验证是独立数据质量环节”，不推成所有任务的普遍增益。
- 阅读 arXiv 预印本 On the Diversity of Synthetic Data（350M/1.4B 配置）。正文保留 topic/prompt 覆盖与重复生成次数是候选审计因素，并说明结果绑定论文语料、模型和预算。
- 阅读 Polyglot Teachers（2026 预印本）。作者比较 10 个 teacher、6 种语言、约 1.4M 条 SFT 样本和 240 个 student 模型；论文报告的 teacher 选择信号绑定其语言、prompt 和评估任务。本章将其作为“参数规模不等于目标 teacher 质量”的限定案例，不据此给出通用选型规则。
- 对照 Findings NAACL 2024 The Curious Decline of Linguistic Diversity 与 ICML 2025 Collapse or Thrive。前者在递归英文生成微调任务中观察多样性下降；后者明确区分逐代以合成数据替换原始数据、与既有数据累积及固定子集等工作流。正文改为条件化表述，不把 collapse 写成合成数据必然结果，也不把累积流程写成无退化保证。
- 阅读 EMNLP 2025 Demystifying Synthetic Data in LLM Pre-training。其预训练比较区分自然网页、改写文本、教材式生成和混合训练；文中比例与速度结果只适用于论文模型、数据类型和预算。数据/计算成本论文的拟合比例未写成通用配方。

新增候选主题均并入原有“方法、评估、决策边界”主线；模型 collapse 不另立章节，代码/数学/领域数据验证仍链接第六章，通用 mixture 优化仍交给第五章。

## 正文修改与旧内容去向

- 将章首文本流程替换为标准 Mermaid 闭环，说明目标、来源授权、生成、任务验证、隐私/污染/安全门禁、配比和训练反馈的先后关系。
- 明确授权与样本质量是不同门禁；拆开 synthetic 来源维度和 distillation teacher-student 关系，要求分别保存 origin、generator、teacher、distilled 标记和 student objective。
- 给 SFT token loss、温度 KL 蒸馏、样本质量代理分、合成/蒸馏 token 并集、能力覆盖、pairwise 重复率及残余风险标记率补充变量/分母说明与可复算数值。
- 修正 pairwise 重复率分母为 `n-1`；将 demo 的逐条上游 duplicate 标记率另行命名，避免把标记统计当作语义相似度测量。
- 将 demo 的 synthetic ratio 门槛改为未舍入的精确比较；输入校验覆盖 schema、非空唯一 ID、origin、task、token 整数、有限质量分、布尔标记、多样性标签集合、teacher 和阈值。
- 空输入与全拒绝输入分别显示未定义指标并暂停；示例新增 `distilled` 独立布尔字段，避免把来源和蒸馏关系误设为互斥类别。
- 改写自然数据锚点表述：采集样本保留渠道观察到的非生成表达和尾部样例，但不等于无偏世界分布，需配合独立时间、来源或人工 holdout。
- 更新常见决策与证据边界，合并 Self-Instruct、Evol-Instruct、蒸馏监督、比例消融和递归数据流程说明。

## 公式与手算复核

- SFT 示例：三个参与 loss 的 token 条件概率为 `0.5、0.25、0.8`，权重均为 1；`-(ln 0.5+ln 0.25+ln 0.8)/3≈0.768`。
- KL 蒸馏示例：`τ=2`，`p_T=(0.8,0.2)`、`p_S=(0.6,0.4)`；`D_KL≈0.0915`，乘 `τ²` 后约 `0.366`。
- 样本质量代理分：给定 correctness/evidence/diversity/safety 与 hallucination/duplicate/contamination 特征，按明示权重逐项乘加得 `q=0.8725≈0.873`；此分不是概率或准入授权。
- 固定 demo：原始 token `8,350`、保留 `4,560`，保留率 `0.546`。其中非蒸馏合成样本 `2,160` token 与同时为 synthetic/distilled 的教师样本 `780` token 构成并集 `2,940`，故 `R_gen=2,940/4,560≈0.645`；蒸馏标签单算 `780/4,560≈0.171`。自然 token 占比约 `0.355`，任务覆盖 `10/10=1.0`，上游重复标记率 `1/12≈0.083`，保留样本残余风险标记率 `0/6=0`。
- 比例边界：`72/100` 精确到达 `0.72` 上限而通过；`7204/10000=0.7204` 即使报告舍入为 `0.72` 仍超过上限。

## 固定代码块执行记录

- 位置：§18.1 唯一 Python 围栏。
- 状态：可执行语句和输入输出约定在本轮修改后重新验证。
- 环境：Python 3.12.9，标准库；执行 `python3 /tmp/book09ch07_fixed_demo.py`。该文件由章节 Python 围栏直接抽取。
- 输入范围：12 条 toy 教学记录，混合自然/合成来源、一个同时 synthetic 与 distilled 的 teacher 轨迹及授权、答案验证、PII、安全、重复、污染、质量、多样性和残余风险字段。
- 输出：6 条保留、6 条拒绝；token 保留率 `0.546`，自然/合成来源比例 `0.355/0.645`，distillation token 比例 `0.171`，synthetic/distillation 并集比例 `0.645`，多样性覆盖 `1.0`，重复标记率 `0.083`，残余风险率 `0.0`，决策 `continue_to_ablation`；与正文输出块一致。
- 已随主示例运行的断言覆盖：精确比例上限及略超边界、空输入、全拒绝、空分母、低质量样本、重复 ID、布尔 token 计数、NaN 分数、非法 origin/多样性类型、错误布尔字段、空 ID、缺字段、非法阈值和空 target tags。断言属于独立教学 demo 的内嵌一致性检查；没有新增测试文件或运行测试套件。
- 教学边界：只处理预先标注的合成记录；不调用 teacher、验证器、代码沙箱、检索器、隐私检测器或训练系统，不能证明真实授权、事实正确、模型收益或生产风险。

## 篇幅、图表、链接与配套同步

- `length_status`：本章为单篇核心讲解单元，按排除标题、围栏、表格、资料列表、代码状态行、行内代码/公式和链接引用文字后的汉字口径计 **8,162 字**，高于 1,800 字下限；篇幅统计不替代连续阅读判断。
- `visual_map`：章首为标准 `mermaid` 三反引号围栏；节点标签使用引号，带文字的虚线箭头使用 `-.->|标签|`。源码结构与箭头语法已静态核对，GitHub 页面实际渲染未验证。
- `link_status`：本轮新增的论文页面为 ACL Anthology、PMLR 或 arXiv 主页面，并已阅读核对其所支持的论断。尚未对本章所有历史外链做完整 HTTP 检查；本地相对链接静态确认目标存在。
- `sync_status`：更新 `PAPERS.md` 第九册第七章来源组；更新 `EXERCISES.md` 的样本 schema、demo 边界、教师验证、比例消融与递归训练练习；修订 `INTERVIEW_BANK.md` Q29–34 的术语边界和 collapse 答法；在 `GLOSSARY_EN_ZH.md` 补充 Synthetic Data Lineage、Synthetic Ratio 定义和 Residual Risk Rate。
- `validation`：固定 demo 执行成功，输出块与实际结果逐行一致；检查章节围栏配对、Mermaid 规范和本地链接；章节正文对进度文件名及“复核记录见”工作流语句搜索结果为零。`git diff --check` 通过。未构建 PDF、未做全章外链 HTTP 扫描、未验证 GitHub 页面渲染。

## 未解决风险与后续

论文比例、teacher ranking 和递归数据结果各自依赖数据来源、模型、任务、生成代数及训练预算；没有目标训练语料和模型实验，不能据此宣称最优比例或产品收益。残余风险和质量分都依赖标注协议，不能替代授权、隐私和硬性安全门禁。

本章完成了正文和已声明代码/数例范围内的精修，但第九册连续阅读、GitHub Mermaid/MathJax 渲染、全章外链扫描仍未通过；书册门禁保持开放，下一章按目录进入第九册第八章。
