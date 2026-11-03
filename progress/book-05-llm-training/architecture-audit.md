# 第五册：大模型训练全流程——册级架构审视

## 审视范围与当前状态

- 审视日期：2026-10-30。
- 阶段：已完成书册定位、简介/目录审视、章节地图、册级研究扫描和骨架决定；开始逐章精修前，骨架保持现状。
- 阅读范围：`BOOK_SERIES.md`、`ROADMAP.md`、第四/第五/第六册简介与目录；第五册 18 个章节文件的标题和二级标题；因异步后训练候选与既有覆盖关系不清，定向检索了第 11、14、17、18 章的相关片段。未把本轮当作逐章精读或章级扫描。
- 研究记录：新知识和来源证据写入 [`research/llm-training-and-post-training-systems-2026.md`](../../research/llm-training-and-post-training-systems-2026.md)。
- PDF：按用户要求跳过，未读取或更新生成 PDF。

## `book_contract`

- **目标读者与前置：** 已会 Python，理解反向传播、交叉熵和 Transformer 基本数据流；希望把训练工程讲清楚。无需预先具备多卡训练经验。
- **读者问题：** 如何从训练目标出发，组织数据、模型、优化、并行、评估和恢复，并能从训练异常定位原因。
- **读完能力：** 能制定端到端训练方案；解释关键更新与资源账本；比较预训练、后训练和多模态路线；评估 checkpoint、能力回归和实验风险；用证据排查 loss、数值、吞吐及系统故障。
- **纳入范围：** 训练项目与数据协议、tokenizer、架构和初始化、预训练循环、优化器、并行和数值稳定性、checkpoint/实验、对齐后训练、多模态训练、成本及训练故障诊断。
- **不承担范围：** 数据治理的深层体系由第九册主讲；数学/PyTorch基础由第十三、十四册补充；通用评估科学由第七册主讲；多模态任务深讲由第十五册主讲；Reasoning/Agent 算法专题由第十六、十七册主讲；部署服务由第六、二十四册主讲；AI Infra 平台设计由第二十三册主讲。
- **书册类型与独特价值：** 以线性训练工程教材为主，并在末段提供可选专项案例。独特价值是把模型能力形成过程、系统执行、测量、恢复和事故排查连成一条训练闭环；相邻册各自只覆盖其中的原理、项目、专题或平台切面，不能替代本册。
- **交接路径：** 从第一册核心训练主线进入；数学与实现不足时先/同步使用第十三、十四册；练习实现接第三册，评估接第七册，数据治理接第九册，多模态接第十五册，Reasoning/RLVR/Agent 接第十六、十七册，部署和平台分别接第六/二十四册与第二十三册。

## `intro_audit`

- 现有定位、前置、能力目标、边界和阅读建议与书系分工基本一致。
- 原简介缺少逐章问题、主线作用和前后衔接的短摘要。已新增“章节地图”，分列 1—14 章训练闭环和 15—18 章选读专题；后续每章精修完成时同步复核对应摘要。
- 本轮未改动已有能力目标及跨册分工表述。

## `toc_audit`

- 现有四阶段顺序可执行：1—5 章建立目标、数据、协议、模型和最小更新；6—9 章处理更新效率与稳定性；10—14 章完成 checkpoint、评估、后训练、多模态、资源和综合诊断；15—18 章作为后训练专题与公开案例。
- 目录已说明主线/选读、先修材料和各专题用途；章节文件均在目录中有入口。本轮不调整章节顺序、数量或文件名。

## `chapter_map`

| 章 | 作用与依赖 | 交接 |
| --- | --- | --- |
| 01 训练总览 | 定义目标、端到端训练链和交付证据；为其余章节建立共同账本 | 02—14 主线；专题读者可回看总流程 |
| 02 数据工程 | 从训练侧处理数据契约、有效 token、领域曝光、混合和消融解释；第九册主讲数据工程全流程方法 | 03 的序列化；09/10 的训练诊断；第九册各环节深讲 |
| 03 Tokenizer 与数据格式 | 约定文本/多模态序列、模板、标签和训练协议 | 04/05 模型输入与监督；11/12 专项数据协议 |
| 04 模型架构与初始化 | 把能力与资源目标转换成网络、规模、位置和初始化方案 | 05/06/08 的训练配置；第二十一册架构深讲 |
| 05 预训练目标与训练循环 | 从概率目标和 mask 推演更新、验证、日志与 checkpoint | 06—10 运行、稳定和实验闭环 |
| 06 优化器与学习率策略 | 比较 optimizer state、参数分组、学习率和 batch 预算 | 08/09 数值与稳定性；13 资源预算 |
| 07 分布式训练 | 组织数据并行、状态分片、TP/PP/CP/EP、通信与恢复 | 08/10/13；第二十三册平台深讲 |
| 08 显存优化与数值精度 | 解释训练状态/激活/峰值显存、混合精度与 OOM | 09 根因诊断；13 资源预算；第六册部署量化另行展开 |
| 09 训练稳定性与 debug | 以时间线和证据链排查 loss、梯度、数据、精度和同步异常 | 10 checkpoint 选择；14 综合诊断 |
| 10 Checkpoint、评估与实验管理 | 使训练状态可恢复、实验可比较、结论可复核 | 11/12 阶段决策；第七册评估方法深讲 |
| 11 后训练与对齐训练 | 建立 SFT、偏好与 RL 方法地图及数据/奖励边界 | 15—18 选读专题；第十六、十七册算法深讲 |
| 12 多模态训练 | 组织视觉、文档、语音、视频和生成任务的训练数据与目标 | 第十五册多模态专题；第七册评估 |
| 13 训练成本与资源规划 | 将 FLOPs、token、吞吐、GPU、存储和重跑变成阶段预算 | 14 综合设计；第六、二十三册系统约束 |
| 14 训练系统综合设计与故障诊断 | 综合前述账本完成方案，并从故障/回归反向定位 | 收束 1—14 主线并引向专题分支 |
| 15 领域专家 SFT 与 RL | 处理专家数据、领域工作流、奖励和回归 | 16/17/18；第九、十六、十七册 |
| 16 RLVR 与可验证奖励 | 定义可验证任务、verifier、安全沙箱和泛化检查 | 第十六册 reasoning RL；第十七册 Agent 环境 |
| 17 On-Policy Distillation | 讲 student rollout、教师信号、分布覆盖、稳定和成本 | 18 公开案例；第十六/十七册相关方法 |
| 18 DeepSeek V4 领域专家与统一蒸馏 | 将公开模型训练披露映射到领域 RL 与多教师 OPD | 第六册 serving、第七册评估、第二十三册平台边界 |

## `discovery_scan`

- **时间与问题：** 2026-10-30；重点查阅 2025—2026 年公开材料，考察多模态长上下文预训练课程、优化器/batch 选择与稳定性、大规模后训练数据/环境/rollout 系统、多教师 OPD 的改进方向。
- **资料类型：** 原始论文和模型团队技术报告；未用搜索摘要替代论文中对应方法和实验段落。
- **扫描结果及去向：**

| 候选 | 证据与边界 | 评估与去向 |
| --- | --- | --- |
| 多模态训练的分阶段对齐与长度课程 | Qwen3-VL 报告的 S0—S3 训练阶段与长 CoT SFT/蒸馏/RL 路线；属于团队自述的单系列方案 | 已有第 12 章和第 11 章落点；章级审阅时检查数据/长度课程和文本能力保持边界，不新增章节 |
| Muon 参数分组、batch scaling 与稳定性验证 | Qwen3.8-Next 报告包含参数分组、batch/LR scaling 和指定模型预算下的 ramp 对照 | 已有第 6 章优化器/批量策略和第 9 章稳定性入口；后续核对证据条件，不把单模型结论写成通则 |
| 合成任务/环境与异步 RL/OPD | DeepSeek-V4.1-Flash 报告披露合成数据/环境、policy lag 控制、stale token mask、checkpoint 切换中断/恢复与多教师 OPD | 相关主题已在第 11、17、18 章出现；定向检查发现第 11 章已有异步 rollout 系统案例，第 18 章已有教师加载和陈旧轨迹边界。章级扫描时比较可复用机制和版本差异，不新增章节 |
| 多教师能力整合 MOPD | 2026 年 MOPD 预印本以 Qwen3-30B-A3B 的多领域对照及 MiMo-V2-Flash 案例报告该路线 | 对应第 17/18 章，不形成独立主线；核对模型/任务/预算后决定示例取舍 |
| UP-MOPD、Nash-MOPD | 2026-10 两篇预印本分别探索 optimizer update 投影和按学习进度动态分配领域样本；实验规模与领域有限，作者也列出长期保持、扩展规模或领域等限制 | 暂列跟踪项；第 17 章章级扫描时判断是否作为“研究候选”短例，当前不升为通用方法或新章 |

- 详细来源、版本和支持范围见研究笔记 [`llm-training-and-post-training-systems-2026.md`](../../research/llm-training-and-post-training-systems-2026.md)。

## `architecture_decisions`

1. 保留 18 章骨架和“1—14 主线、15—18 选读”的结构；现有定位覆盖训练生命周期，不需要把数据治理、通用评估、serving 或 AI Infra 的整册内容重复搬入。
2. 本轮未发现需要新增独立主线章节的证据。新方法和报告案例进入既有训练、后训练、优化器或多模态章节，具体增补等相关章节扫描后决定。
3. 已在简介新增逐章摘要；目录路径不变。结构稳定后按第 1 章开始主线精修。

## `visual_map`

- 全册适合检查的视觉关系：目标到数据/模型/更新/验收的训练闭环；样本到 token 与 label mask 的协议流；一次 update 的状态变化；DP/TP/PP/EP 的切分和通信；checkpoint 恢复状态；RL/OPD 中 rollout、verifier/teacher 与参数更新；训练成本分账。
- 本轮只登记机会，没有逐图审查或修改 Mermaid；相应章节精修时根据规范模块 04 决定保留、修改或用表/流程图表达，并记录渲染门禁。

## `coverage_map`

- 训练全生命周期的主解释入口保持在第 1—14 章；第 01 章给全景，第 14 章做整合，不取代中间章节的机制解释。
- 后训练通用方法在第 11 章，领域任务/RLVR/OPD 分别在第 15—17 章展开，第 18 章作为模型披露案例；重复边界在逐章扫描中核实。
- 数据工程与治理全流程、通用评估、模型架构史、多模态专题、Reasoning/Agent、生产 serving 和平台均保留对应书册为主讲入口；第五册第 02 章只提供训练决策所需的输入、曝光与实验解释视角。

## `open_risks`

- 结构标题显示数处可能重复，须在逐章精读时区分职责：第 01/14 章总览与综合设计；第 06/09 章 optimizer 与稳定性；第 07/08 章并行与显存；第 08/09 章数值与诊断；第 10/13/14 章 checkpoint、成本与方案复盘；第 11/15—18 章后训练方法和案例。
- 第 18 章的 DeepSeek V4 来源、版本和可迁移结论需和 2026-09 的 V4.1-Flash 报告对照；先逐章阅读正文再决定更新，不凭章节标题替换案例。
- 章节代码、公式、数值例、引用、目录锚点、视觉表达和配套索引均未在册级阶段验收。
- 目标平台 Mermaid 渲染、外链终审、代码是否需要复验及配套资料同步，保留到对应章节和整册门禁。
- 第五册第 02 章已按简介/目录确定第五册与第九册的分工，并加入深讲路由；第九册正文尚未逐章审阅。轮到第九册时需复核数据清洗、去重、混合、专项数据和治理方法的完整主讲位置，避免重复保留两处深讲。

## 第 01 章：训练总览——精修记录

- **scope：** 完整阅读并精修 `book-05-llm-training/chapters/01-训练总览.md`，完成本章定向扫描；保留全景地图定位，不把中间章节的机制主讲搬进总览。章节标题共同构成训练闭环的连续论证，不是彼此独立的目录讲解单元。
- **chapter_scan：** 2026-10-30；问题范围为训练目标到交付证据的生命周期、checkpoint 恢复状态、训练阶段与测试时计算的边界、缩放经验的适用范围，以及总览章与第 13/14 章的重复。检索和阅读原始论文、ICLR/PMLR 论文页及 PyTorch 官方文档。候选去向：① checkpoint 是否可复现累积梯度边界——纳入 1.6.2，区分 optimizer-step 边界和梯度累积中途的恢复状态；② reasoning checkpoint training 与 test-time compute——纳入 1.8.3，分开训练阶段和推理期预算；③ generative pass@k scaling——纳入 1.7.2，作为带任务、样本预算和 verifier 条件的研究证据，不推广为通用模型定律；④ data-quality scaling——在 1.9.2 说明它会影响拟合，并将数据质量机制深讲交给第 02 章；⑤ Scheduled Sampling 的历史边界——仅作为自回归训练/暴露偏差讨论的原始研究入口，不将其写成现代 LLM 通用配方。来源事实和边界见 `research/scaling-laws-data-mixtures-2026.md`、`research/pretraining-data-packing-checkpoints-2026.md`、`research/test-time-compute-and-evaluation-2026.md` 及对应正文链接。
- **structure：** 章首说明本章是全流程地图和共享账本，并增加第 05、06、10、13、14 章主讲入口；将 optimizer-boundary checkpoint 与累积中途状态分开；将训练成本总账 `Cost_total` 与训练 FLOPs `C_train` 分开；缩窄 1.7 的缩放结论，避免与第 13 章成本规划重讲。保留第 14 章作为综合方案主讲。
- **length_status：** 1.1—1.17 构成一个连续的核心总览单元，解释性汉字静态计数 9,846，核心单元下限 1,800，通过。计数排除标题、公式/代码围栏、表格、1.18 来源清单和 1.19 练习；本项按连续主线整章统计，不把导航小节虚拆为独立教学单元。
- **example_status：** 1.5.3 给出带有效 label mask 的 loss 分母；1.7.2 有等 FLOPs 的手工计算示例并区分经验拟合与推论；1.17 用明确标注的教学假设完成训练预算账本：1.3B 参数、120B token、global batch 524,288 token/update、约 228,882 updates、14,400 token/s、理想运行约 96.45 天、约 18,519 GPU-hours、`6ND=9.36e20` FLOPs、单份权重约 2.42 GiB。1800 token/s/GPU 明确是演示假设，不是实测吞吐；保留进位和估算边界。
- **code_status：** 删除原 1.17 静态审阅 Python 规划器，因其与第 13 章成本例及第 14 章综合方案重复，改为可手算的预算表；本章现无可执行代码块，因此无需运行或声称验证。数学与 fenced-block 配对做静态检查。
- **formula_status：** 修正带 mask loss 的分母为 `sum_i m_i`，定义二值 mask 且要求有效 token 数大于零；在 6ND 中明确 `N>0` 及 active parameter 口径；将金额总账命名为 `Cost_total`，避免与 FLOPs `C_train` 混淆。关键算式和预算数值逐步复算。
- **visual_status：** 保留适合总览的阶段/资源账本表；无需额外 Mermaid，避免把单页地图扩成与后续章节重复的机制图。图表读法由相邻正文说明。
- **link_status：** 本地 Markdown 目标及简介入口静态检查通过；新增/修订的论文和官方文档引用贴近对应论断。整册外链最终可达性与证据适配仍留到第五册终审。
- **sync_status：** `简介.md` 第 01 章摘要已改为全流程地图及 token/FLOPs/throughput/update/checkpoint 账本；术语表新增 Pass@k 估计量及其条件；`PAPERS.md` 已含 ICLR 2026 缩放研究入口，练习集已有数据质量缩放题，本轮不重复添加。
- **open_risks：** 章内外链尚未做整册最终可达性核验；需在逐章精修后按目录路径复读，并终审所有外链、锚点和图示门禁。本章未运行测试、模型、benchmark、书中代码或 PDF。

## 第 02 章：数据工程——精修记录

- **scope：** 完整阅读并精修 `book-05-llm-training/chapters/02-数据工程.md`，完成本章定向扫描；保留训练工程闭环中的数据输入位置，解释数据来源、筛选、混合、序列化和版本如何影响有效 token、领域曝光、训练成本及实验结论。
- **chapter_scan：** 2026-10-30；扫描训练前数据契约与授权、质量分类器目标偏置、PII 过滤、去重/benchmark 污染、mixture 估计、合成与专项数据、tokenize/packing/lineage、训练消融。阅读 NeurIPS 2024 DataComp-LM 原文及官方仓库、FineWeb v2、Dolma v2、Removing Noise v4、Leverage Score Sampling v1 和多语言 CPT 原文。候选处置：①数据采集/清洗/质量/去重/污染/mixture/估值/专项数据/治理等完整工程方法由第九册对应章节主讲，本章保留训练侧因果与数值视角并加直接路由；②DCLM 分类器表格分开说明 1B-1x 过滤器比较和 7B-1x 正例来源消融，避免规模与问题混写；③FineWeb-Edu 按论文 v2 报告的 46 万标注样本，不混用后续数据卡快照；④质量过滤与表示多样性作为有版本、有限任务条件的新研究对照，不写成全局排名；⑤没有发现需要新增本册章节或改变本章标题/顺序的候选。
- **architecture/boundary：** 本章新增书系阅读边界说明和指向第九册第 2—11 章的路由表；第五册聚焦训练数据契约、有效 token、领域曝光、目标 mixture 与训练收益解释，第九册承担数据工程全流程、治理和数据价值深讲。简介第 02 章短摘要同步。路由依据第九册简介/目录；第九册正文尚未审阅，后续需在其逐章处理时复核主讲位置。
- **length_status：** 静态计数仅计算连续中文解释文字，排除标题、表格行、围栏代码/公式和来源清单：① 2.0—2.5 生命周期/契约/解析/统计，2,464 汉字，核心单元，下限 1,800，通过；② 2.6—2.7 质量与风险准入，2,373 汉字，核心单元，下限 1,800，通过；③ 2.8—2.10 重复与评估污染，2,106 汉字，核心单元，下限 1,800，通过；④ 2.11—2.20 从 token 预算、数据类型分支、packing/版本，到消融、审计和故障定位，4,391 汉字，核心单元，下限 1,800，通过。各 `## 2.x` 标题是上述连贯教学链的推进步骤或专项分支，不单独充作完整讲解单元。2.21 资料清单 523 字、2.22 练习 410 字、2.23 过渡小结 282 字，不计为独立知识讲解单元。
- **example_status：** 新增 2.1.1 教学管线样例：4,000 bytes HTML → 1,200 字符 → 1,170 字符，`r_change=-2.5%`；固定 tokenizer 下假设 520 token，许可未知时可训练量为 0，解除隔离后两次采样得到 1,040 个训练 token 实例但仍只有 520 个唯一 token。2.6.3 给出质量加权分数手算 `0.71`，并说明许可硬门禁不可被总分抵消；2.9.1/2.9.5 有 Jaccard/LSH 与 BFF/MinHash 对照；2.11.5 有 `100:10:1`、`alpha=0.5` 和 200 token 单位的曝光/重复倍数；2.18.2 新增通用与代码切片的加权总分例，`0.676→0.684` 同时暴露代码通过率 `0.50→0.46` 的回归。新增数值明确为教学假设；论文分数带来源、方向和设置。
- **code_status：** 2.19 标准库 Python 教学审计代码的可执行语句、依赖、环境、输入输出约定和预期没有修改，沿用正文 2026-10-28 的验证标记。标记记录了从正文提取 fenced demo 并以 Python 3.12.9 标准库执行；输入为代码内 7 条合成记录，质量/近重复/污染阈值分别为 0.56/0.45/0.10；逐项核对过滤、重复、污染、保留 ID、mixture 和断言，包含空 shingle 与非法 shingle size 边界，最终输出 `data audit toy: ok`。原始 shell 命令全文和 stdout 未保存在现有维护记录中；由于本轮代码未变化，未重跑，后续如修改执行语句再按规范补足可复核命令和实际输出。
- **formula_status：** 复核长度变化率、过滤/保留分母、Jaccard/LSH 概率、温度采样、加权分数和消融差值；新增手算 `(-30)/1200=-2.5%`、`s=0.71`、`0.8×0.72+0.2×0.50=0.676` 与 `0.8×0.74+0.2×0.46=0.684`。BFF 7B-2x 的 token yield 采用 NeurIPS 2024 正式表 19 的 3T；当前 arXiv HTML 对应单元格为 3.8T，作为源版本差异明确记录，正文不混用两值。
- **visual_status：** 保留 2.1.1 生命周期等宽流程块、比较表和训练账本；线性阶段已能清楚呈现顺序，本章未新增 Mermaid 或装饰图，分支/适用条件在邻近文字中说明。
- **link_status：** 全章及更新后的简介本地 Markdown 文件目标静态检查通过；新增的第九册第 2—11 章、第七册简介路由均存在。26 对 fenced block 配对，无围栏错误；外链最终可达性和全册锚点验收留到第五册终审。
- **sync_status：** 更新第五册简介第 02 章摘要；`PAPERS.md` 固定 CQF v4 与 Leverage Score Sampling v1 标题/版本；`GLOSSARY_EN_ZH.md` 补充 leverage score 的几何覆盖含义与非质量保证边界；扩展 `EXERCISES.md` 阶段 3 mixture ablation 题，要求分域指标、有效曝光、聚合回归审计和独立评估切片。新研究证据写入 `research/pretraining-data-selection-and-diversity-2026.md`。目录标题、章序和文件名未变化。
- **open_risks：** 第九册正文尚未逐章复核，跨册深讲唯一性留待第九册对应章节验收；新近 Leverage Score Sampling 只有作者预印本结果，等待独立复核；本章外链与旧锚点尚未完成第五册整册门禁。2.19 的执行状态沿用正文既有标记，但历史 shell 命令和完整 stdout 未保存在维护记录中；代码未改且边界标记仍在，后续改动该代码时须重新验证并保存完整运行记录。

## 第 03 章：Tokenizer 与数据格式——精修记录

- **scope：** 完整阅读并精修 `book-05-llm-training/chapters/03-tokenizer与数据格式.md`，处理本章范围、训练协议主线、算法重复、文本/模板/labels/packing、任务 schema、tokenizer 扩展、代码审计与回归门禁；未改变目录中的章名和章序。
- **chapter_scan：** 2026-10-30；问题范围为 tokenization 算法和内在评测与第二册的重复、训练侧 tokenizer 预算与模型输入接口、规范化对可逆和 span 的影响、chat template/EOS/PAD/label mask/packing 的版本契约、SFT/偏好/RL/工具/多模态记录边界，以及词表扩展对 checkpoint/输出层/服务回归的影响。阅读 ACL Anthology 的 2026 Findings/ACL/SRW/EMNLP 正式论文页面和相应实验段落、两篇 2026 arXiv 词表扩展版本，并复核目标接口文档。候选去向：① BPE、Unigram、byte-level 算法和 intrinsic tokenizer 评测路线由第二册第三章第 26—27 讲完整主讲，本章保留接口影响和训练预算示例；② 格式故障复现与排查顺序由第十九册第三章主讲，本章留协议检查信号；③ 数据采集、筛选、去重和治理见第九册；④ Hugging Face/偏好优化完整微调项目见第三册；⑤ SFT、偏好/RL、工具和多模态的训练记录 schema、embedding 扩展和版本回归留在第五册。没有发现需要新增第五册独立章节或改变本章目录位置的候选。
- **structure/boundary：** 章首增加书系主讲路由。3.2—3.5 调整为接口背景并回链第二册；3.6 保留词表/序列/输出层训练账本和研究边界；3.7—3.9 组织为控制事件与监督协议的连续讲解；3.10 扩展文档边界、packing、label 分母、截断与数据 split 顺序；3.11—3.14 组织为不同训练任务的记录 schema；3.15 扩展 checkpoint/embedding 迁移和实测边界；3.16 保留协议审计器；3.17 收为故障信号索引；3.18 建立版本回归流程。更新第二册、第四册和第十九册交叉入口。
- **length_status：** 按解释性中文计数，排除标题、表格、公式和代码围栏；核心单元最低 1,800，辅助讲解最低 1,000。3.1 输入/规范化/manifest 为 1,839（核心，通过）；3.6 词表及训练预算为 1,854（核心，通过）；3.7—3.9 控制符号、模板与监督为 2,038（连续核心单元，通过）；3.10 文档序列化与 packing 为 1,889（核心，通过）；3.11—3.14 SFT、偏好/RL、工具和多模态 schema 为 2,115（连续核心单元，通过）；3.15 词表扩展为 1,824（核心，通过）；3.16 审计器讲解为 1,004（辅助，通过）；3.18 协议版本评估/回归为 1,880（核心，通过）。3.2—3.5 是带第二册深讲入口的背景速览，不作为独立教学单元；3.17 为第十九册故障手册的索引，不承担完整排障教程。
- **example_status：** 3.1 用 CRLF `a\r\nb` 的 4→3 bytes 说明规范化与源偏移映射，并用 40 个输入 token、12 个监督目标、模板增 3 token 解释 input/target 分母；3.6 计算 `32K×4096=131,072,000`、`128K×4096=524,288,000`，并在 `B=2` 下比较旧输出层 262,144,000,000 MAC 与新输出层 629,145,600,000 MAC（2.4×），另以假设 token/byte 比率区分固定 bytes 与固定 token 预算；3.7—3.9 保留 assistant label shift 与 10/100 token 的 token-mean/sample-mean 例；3.10 用两篇 4-token 文档计算 50%→100% block 利用率、20/36 因果可见位置对、6 个有效预测目标，并计算 1,140 token 样本在 1,024 上限下截断 116 token 的后果；3.11—3.14 保留单轮/多轮、pairwise、组内候选/验证器、工具轨迹和媒体数量示例；3.15 手算新向量均值、2,000 行 BF16 参数量及 65K→128K 报告范围；3.16 说明 toy tokenizer 中 `tokenizer` 为 1 个片段、`🧪` 为 4 个 byte token（仅示例词表）；3.18 以 100→85 token 计算 `ΔL=-15`、`r=-15%`。
- **code_status：** 3.16 的可执行语句本轮未改动。原正文虽写有 2026-10-28 Python 3.12.9 执行标记，但对应进度材料没有可靠运行命令和完整 stdout 供复核；按代码规范改为“静态审阅，未运行（2026-10-30）”。本轮未运行章节代码、框架、模型或 benchmark。
- **formula_status：** 复核 `Vd` embedding 参数量、`B·L·d·V` 输出投影 MAC/FLOPs、固定 bytes/token 预算换算、BPB 分母条件、packing 利用率、causal visible-pair 求和、跨样本标签屏蔽和长度差 `ΔL/r`。示例值均按文中变量口径复算；FLOPs 局部账本明确未含反向、softmax、通信、padding 与其他层开销。
- **visual_status：** 保留数据流、序列/attention 示例、表格和代码审计器；本轮未新增 Mermaid，因现有状态/协议表和逐样本数据流已表达关键关系。全册读者平台渲染仍待终审。
- **link_status：** 第三章 36 对 fenced blocks 配对；章首、简介及第二/第四/十九册新增本地 Markdown 目标存在；全仓没有引用本章已变更的旧小节锚点。新引用直达 ACL Anthology/arXiv 原论文页；既有外链全量可达性与证据适配留到第五册最终门禁。
- **sync_status：** 更新第五册简介第 03 章摘要；第二册加回第五册训练协议路由，第四册改为算法与协议分工导航，第十九册链接第五册 schema 主讲；`PAPERS.md` 增补 Equity with Efficiency、To Each Language Its Tokenizer、From Bytes to Subwords；新研究证据写入 `research/tokenizer-design-and-training-evidence-2026.md`。目录标题/章序不变；未重复改动 `EXERCISES.md` 或术语表。
- **open_risks：** 第九册和第三册正文尚未逐章复核，对应深讲路由需在这些书册的精修过程中继续确认；新近 tokenizer/词表扩展结论均为来源证据，未在本地重训复现；3.16 代码保持静态未运行；全章旧外链和跨册锚点仍需第五册整册门禁。未处理 PDF。

## 第 04 章：模型架构与初始化——精修记录

- **scope：** 完整阅读并精修 `book-05-llm-training/chapters/04-模型架构与初始化.md`，围绕“如何把能力、训练数据、算力、显存、服务负载和稳定性转换为一份可复核的模型配置”完成章级扫描。保留章节名和位置，不改第五册骨架；架构机制与历史的深讲回路第二册、第二十一册，完整训练成本回路第五册第 13 章。
- **chapter_scan：** 2026-10-30；扫描 decoder-only、参数/token/算力预算、RoPE 长度扩展、Norm/残差与初始化、FFN 形状、MHA/MQA/GQA 缓存、长上下文计算/有效能力、MoE active 参数与路由负载、架构改动后的稳定性。阅读相应原始论文/技术报告，检查 Qwen3.8-Flash-Next §3.3、DeepSeek-V4.1-Flash 长序列训练披露、Chinchilla FLOPs computation 附录，以及书系中第二册 Transformer 机制、第二十一册相关章节和第五册第 05、06、08、09、13、14 章的主讲边界。候选去向：① PI、YaRN、LongRoPE 保留为位置策略对照，完整推导由第二十一册第 18 章主讲；② DeepSeek-V4.1-Flash 的 64K 起训、约 34T token 后扩至 1M 只作单一团队配方案例，不升格为通用课程；③ RMSNorm、Pre/Post-Norm、DeepNorm、μP、SwiGLU、GQA/MQA、MoE 的机制和论文谱系回链第二册/第二十一册，本章保留配置字段、预算影响、例子与训练验收；④ Qwen3.8-Flash-Next 的 gate stress test 作为“架构与 recipe 联合验证”的有限案例，指标范围和单模型/压力测试边界保留在正文，证据与阅读位置记入 `research/llm-training-and-post-training-systems-2026.md`；⑤ `research/transformer-architecture-research-2026.md` 中 SpanNorm、PowLU、MoA、log-linear attention、GatedDeltaNet2 等候选没有改变本章配置决策主线，机制留在架构专册，不新增章节或泛化成训练建议。
- **architecture/boundary：** 章首明确本章主线是训练前约束转换和配置验收。新增书系分工路由及 decoder-only 四 token 因果 mask 案例；将原先过长的未运行 Python 预算脚本收束为手算账本；合并初始化和稳定性讲解；把 RoPE/NorM/FFN/head 组织/长上下文/MoE 机制压到本章所需的训练决策深度并保留准确深讲入口。扩展 4.12 参数、训练 FLOPs、KV、attention 与 MoE 联合账本和 4.13 低成本排障顺序。修复误放在 RoPE 节首的稳定性段落归属；不改章序、目录或模型结构覆盖范围。
- **length_status：** 只计连贯中文解释文字，排除标题、表格行、公式/代码围栏与引用清单；按中心问题合并连续论证单元：① 4.0—4.4 目标约束、decoder-only 数据流、参数账本与 token 预算，2,402 汉字，核心单元，下限 1,800，通过；② 4.5—4.10 位置、归一化/残差、FFN、KV、上下文和 MoE 如何改变同一训练/运行账本，4,112 汉字，核心单元，下限 1,800，通过；③ 4.11 初始化与稳定性配置支持，1,178 汉字，辅助单元，下限 1,000，通过；④ 4.12 手算架构总账与门槛决策，1,913 汉字，核心单元，下限 1,800，通过；⑤ 4.13 故障诊断路径，1,096 汉字，辅助单元，下限 1,000，通过。4.14 资料清单、4.15 练习和 4.16 小结不计作独立知识讲解单元。
- **example_status：** 4.2 以 `[BOS,A,B,C]→[A,B,C,EOS]` 逐行演算因果可见位置，共 10 个下三角位置对、无 mask 时 16 个；shape 例为 `d=4096,Hq=32,Hkv=8,dh=128`。4.3 计算 GQA attention 参数 `41,943,040` 对 MHA `67,108,864`、cache 比例 `25%`，并对照普通 FFN `134,217,728`；4.4 解释 `6PD` 一阶预算口径。4.5 `d_rot=8,base=10,000` 给出四个频率，并以 `(1,0)` 在位置 1 旋转到约 `(0.5403,0.8415)`；4.6 RMSNorm `(3,4)` 得 RMS `3.536`、输出约 `(0.849,1.131)`，Pre-Norm residual 另给深度尺度假设；4.7 `d=4096` 下 dense FFN 与 `d_ff=10,923` 的 SwiGLU 分别为 `134,217,728` 与 `134,221,824` 参数/MAC，取整差约 `0.003%`；4.8 在 `L=32,B=4,C=4096,dh=128,BF16` 下比较 MHA 8 GiB 和 GQA-8 2 GiB cache；4.9 计算显式 score matrix 4 GiB/层，4K→32K 时 KV 为 2→16 GiB、score matrix 为 4→256 GiB/层（后者是假设完整物化）。4.10 计算 8 expert/top-2 的 routed FFN 总/active 参数 `1.074B/268.44M` 及 `T=1024,E=8,k=2,alpha=1.25` 的 320 slots/expert。4.11 Xavier `fan_in=fan_out=4` 得 `std=0.5`、输出方差约 1；32 层 toy residual scale 每条支路 `sigma/8`，总方差仍约 `sigma²`（独立同方差假设）。4.12 汇总约 `5.769B` 参数、`D/P≈20.8`、`6PD≈4.15×10^21` FLOPs；假设持续有效吞吐 1 PFLOP/s 得理想约 48.0 天；再用 24 GiB 独立 KV 预算比较 32K/4 并发的 GQA-8 16 GiB 与 MHA-32 64 GiB。所有自构造预算都明确是教学假设，不伪装成本地模型或硬件实测。
- **code_status：** 旧的长 Python 预算脚本已从正文移除，以逐步公式、配置表和手算账本代替；本章当前没有可执行代码块。保留的 `~~~text` 仅为训练目标接口的文字示意，不是程序代码，不运行示例。
- **formula_status：** 静态复算 attention/FFN/embedding 参数、tied/untied 差额、`6PD` FLOPs、`D/P`、RoPE 一维旋转、RMSNorm、残差方差、SwiGLU MAC 和宽度取整误差、GQA/MHA KV bytes、attention score bytes、MoE active/total 参数、capacity 和训练时长换算。确认 `6PD` 只作短序列 dense 一阶筛选；`4K→32K` 对全量 attention 的二次主项为 64 倍、KV 线性主项为 8 倍；稀疏 MoE FLOPs 使用 active 口径而权重驻留用 total 口径。公式保留相应正值/维度/独立性假设和省略项。
- **visual_status：** 保留配置约束表、架构取舍表和 text dataflow 示意；小型因果序列及逐步预算更适合直接列出状态/中间量，无需另加 Mermaid 图。已有表格在正文中解释维度和读法。
- **link_status：** 第四章 39 对 fenced block 配对；24 个本地 Markdown 链接目标静态存在、无本地 fragment。第二册第 13—25 讲、第二十一册机制章节、第五册成本/训练/调试章节与简介摘要路由可达。移除了 Chinchilla 链接上不正确的 `#S6` fragment，保留直达原论文页。外部链接可达性和全文旧锚点适配留到第五册整册最终门禁。
- **sync_status：** 已更新第五册简介第 04 章短摘要、第二册架构章节到第五册配置/账本的交接、第二十一册简介的训练预算导航。`PAPERS.md` 已有相关架构原始论文、Qwen3.8 与 DeepSeek-V4.1 来源入口；`EXERCISES.md` 已覆盖 FFN/MoE、参数-token 预算和 GQA KV 估算；`GLOSSARY_EN_ZH.md` 已含本章术语，因此本章不重复添加。Qwen3.8 §3.3 gate stress-test 新读事实及分母边界补入既有训练研究笔记。
- **open_risks：** 论文数字属于报告作者的特定压力测试；`1 PFLOP/s`、24 GiB cache 预算和 toy 模型配置均为示例假设，目标项目必须以真实 config、profile、吞吐和质量评估复核。本章未运行代码、模型、benchmark、程序测试或 PDF；后续做第五册整册外链/锚点门禁时复查来源可达性与跨册链接。

## 第 05 章：预训练目标与训练循环——精修记录

- **scope：** 完整审阅并精修 `book-05-llm-training/chapters/05-预训练目标与训练循环.md`，围绕“训练目标如何经过数据对齐、有效 loss 分母和数值更新，形成可复核的参数变化”完成章级扫描。保留第 05 章位置和 5.0—5.17 主体结构；目标谱系与训练循环留在本章，样本协议回路第 03 章，优化器状态详见第 06 章，speculative decoding 的候选验证回路第二十四册。
- **chapter_scan：** 2026-10-30；扫描 causal LM 与 BERT-style MLM 的监督差异、T5/UL2/FIM 的目标变换、MTP 未来目标数和推理复用边界、mask diffusion 的损失/采样差异，以及 label shift、mask、packing、梯度累积、低精度、验证、checkpoint 和排障间的缺口。按原论文定向核对 T5、UL2、FIM、Gloeckle et al. MTP、DeepSeek-V3 MTP 与 MDLM。候选去向：① causal LM 保持训练主线；② BERT、T5、UL2、FIM、MTP 和 MDLM 在 5.1.4 按上下文、目标 token、架构约束和证据边界比较，不另拆章；③ Gloeckle 的共享主干并行预测头与 DeepSeek-V3 的串联 MTP 模块作为两种不同实现，推理候选接受与加速留给第二十四册 MTP 主讲；④ MDLM 仅在此处解释噪声输入、加权 masked objective 与反向采样入口，ELBO、完整 mask 机制和采样数例由第二册第二章离散扩散节主讲；⑤ tokenizer、样本 schema、label mask 和 packing 的完整协议由第三章主讲。没有发现需要改变第五册骨架或增加独立讲次的候选。
- **architecture/boundary：** 保留 causal 最大似然到一次 optimizer update 的解释链。将 5.1.4 扩展为目标对照并纳入 MTP/MDLM，补齐 MTP 两预测距离的有效监督数（第一步 4 项、第二步 3 项，共 7 项）；把 label shift、三类 mask 与 packing 的协议主责回链第三章。扩展 update 顺序、裁剪与变长 token accumulation 的逐步算例，并说明 DDP 全局分母、loss-scale 还原和梯度裁剪顺序；延伸验证、日志、恢复、checkpoint 选择和排障；保留最小 token-to-logits 代码，但明确 attention/causal mask 只构造打印，不进入 logits/loss。
- **length_status：** 只计连贯中文解释性汉字，排除标题、表格、代码围栏、公式和状态标记；按解释链分组。① 5.1—5.5 目标、label 对齐、交叉熵、mask 与 packing，3,278 汉字，核心单元，下限 1,800，通过；② 5.6—5.8 optimizer update、梯度累积与数值精度，2,030 汉字，核心单元，下限 1,800，通过；③ 5.9—5.12 与 5.14 验证、观测、checkpoint 和故障定位，1,958 汉字，核心单元，下限 1,800，通过；④ 5.13 最小训练步代码讲解，1,138 汉字，辅助单元，下限 1,000，通过。5.0 为章首导读，5.15—5.17 为资料、练习和小结，不作为独立知识单元计数。
- **example_status：** 5.1.4 手算 `[BOS,A,B,C,D]` 上的两个未来距离，分别计 4 与 3 个有效目标；5.2 用 `[BOS,A,B,EOS]` 外部 shift 得 3 个目标；5.3 从 logits `[2,1,0]` 得中间类别概率约 0.2447、loss 1.4076 nat，并用两有效位算 masked mean 1.2531；5.4 逐位置说明 causal、padding-attention 与 label mask；5.5 对两篇 3-token 文档比较分块/packing 的 50%/66.7% 目标利用率与 EOS→下一文档跨界标签。5.6.1 以 `g=(6,8)`、范数 10、阈值 5 算裁剪梯度 `(3,4)` 和 SGD 更新；5.7 以 2/4 token micro-batch 手算 token 权重、二维梯度 `(2/3,8/3)`，并用两 rank 的 `P=2,N=6` 算 `P/N=1/3`；5.7 继续对比先整体裁剪与逐 micro-batch 裁剪结果；5.8 手算 FP16 loss scale 128 与 unscale/clipping 顺序。5.9—5.12 用有效 token 加权验证、step/token 日志、step 1200 恢复清单和阈值候选选择解释指标分母与状态生命周期；5.13 手工追踪 2 个 packed block 的 6 个有效 pair、跨文档 EOS→`the` 标签及 2/4 token 梯度加权。
- **code_status：** 5.13 可执行 Python 语句本轮未修改，正文状态仍为“静态审阅，未运行（2026-10-26）”；本轮重读代码并改写邻近说明，明确 attention 与 causal mask 未被模型计算使用，增加代码接口和跨文档目标的静态说明。没有运行章节程序、测试、模型、benchmark 或 PDF。
- **formula_status：** 静态复核 joint probability 分解、label shift、stable softmax/cross entropy、mask denominator、packing target 数、global token mean、micro-batch token weighting、DDP rank averaging、gradient clipping、loss scaling/unscale、validation loss 和 PPL。新增整窗向量裁剪例与 7 项 MTP 有效目标数；教材算例均为合成输入，未表述为模型实测。
- **visual_status：** 保留目标比较表、mask 对照表、token/梯度逐步文本示例与 toy 代码。关键顺序与数量可由表和逐步算例核对，本章未新增 Mermaid 图。
- **link_status：** 第五章 70 个围栏标记成 35 对；28 个 Markdown 链接中 25 个外链、3 个本地目标均存在，未使用本地 fragment；第五册简介的 20 个本地链接目标静态存在。`git diff --check` 通过。外链全量可达性和跨册旧锚点留到第五册整册门禁。
- **sync_status：** 更新第五册简介第 05 章短摘要；`PAPERS.md` 新增 MTP 与 MDLM 入口（第 23、24 项，后续条目顺延）；章节练习八纳入 MTP/扩散及每个 head/noise step 的有效分母。术语表已有 MTP、MDLM 词条，本章不重复添加。MTP/MDLM 的原论文版本、方法事实、阅读位置和适用边界写入 `research/pretraining-objectives-mtp-and-mdlm-2026.md`；既有 CE/PPL、mask、packing、精度和训练状态证据沿用相应研究记录。
- **open_risks：** MTP 各预测距离与权重、MTP 推理接受率、MDLM 噪声/目标和采样成本都受实现与评测设置约束；正文没有据论文结果推断本地性能。toy 代码无 attention、Transformer、真实优化器或多步训练，仍未运行。第五册其余章节、整册连续复读、外链可达性和跨册锚点门禁待完成；未更新 PDF。

## 第 06 章：优化器与学习率策略——精修记录

- **scope：** 完整审阅并精修 `book-05-llm-training/chapters/06-优化器与学习率策略.md`，沿“梯度经优化器状态、参数分组和学习率轨迹形成更新，如何以质量、token、显存、墙钟和恢复状态共同验收”组织。保留第六章位置和 6.0—6.24 结构；优化器机制留本章，显存规划回链第 08 章，分布式状态/裁剪实现回链第 07 章，checkpoint 与公平评测制度回链第 10 章。
- **chapter_scan：** 2026-10-30；扫描 Adam/AdamW 状态语义、Adafactor/Lion/Sophia/Muon/GaLore 的机制与实现差异、矩阵/低秩 optimizer、warmup/cosine/linear/WSD、decay/clipping、有效 token batch、调参和恢复公平性。定向阅读 AdamW 官方 2.14 文档、Adafactor v1、Lion v4、Sophia v4、GaLore v2、Muon 技术报告、ICLR 2026 优化器比较、SOAP、μP transfer、MiniCPM/WSD 与 River Valley 原文；来源版本、阅读位置及证据边界见 `research/optimizer-alternatives-and-scaling-2026.md`。候选处置：① 纳入各方法的更新方向、状态账本、超参/刷新/形状依赖与可复算例；② Shampoo/SOAP 保持支持性比较，不增独立讲次；③ 低层并行实现、完整显存 profiler、事故处置细节分别回链第 07、08、09、10 章；④ 论文加速值只作为限定条件的作者报告，按 tokens/FLOPs/wall-clock 拆分，不形成跨设置总排名；⑤ 未发现须增章或改变章节标题/顺序的候选。
- **structure/boundary：** 保留 6.0—6.24。6.2—6.5 建立 SGD/Adam/AdamW、状态内存/精度/分布的基线；6.6—6.10 比较 Adafactor、Lion、Sophia、Muon、GaLore 的不同状态与更新几何；6.11 明确参数对象的互斥分组；6.12—6.14 递进解释 warmup、cosine/linear 和 WSD/WSD-S；6.15—6.18 连接 decay、clipping、token batch 和公平对照；6.19—6.21 以静态教学代码、checkpoint 轨迹和 loss-spike 时间序列收束。新增原始 WSD 与 WSD-S checkpoint 角色对照表，不增 Mermaid。
- **length_status：** 按解释性中文计数，排除标题、表格、公式/代码围栏和状态标记；核心下限 1,800、辅助下限 1,000，低于门槛即未通过。

| 单元 | 类型 | 汉字数 | 下限/状态 |
| --- | --- | ---: | --- |
| 6.1 参数、梯度和状态账本 | 辅助 | 1,030 | 1,000，通过 |
| 6.2 SGD 与 Momentum | 核心 | 1,808 | 1,800，通过 |
| 6.3 Adam | 核心 | 1,875 | 1,800，通过 |
| 6.4 AdamW | 核心 | 1,852 | 1,800，通过 |
| 6.5 状态显存、精度和分布 | 辅助 | 1,077 | 1,000，通过 |
| 6.6 Adafactor | 核心 | 1,891 | 1,800，通过 |
| 6.7 Lion | 核心 | 1,908 | 1,800，通过 |
| 6.8 Sophia | 核心 | 1,960 | 1,800，通过 |
| 6.9 Muon | 核心 | 1,847 | 1,800，通过 |
| 6.10 GaLore | 核心 | 1,897 | 1,800，通过 |
| 6.11 参数分组 | 辅助 | 1,107 | 1,000，通过 |
| 6.12 Warmup | 核心 | 1,809 | 1,800，通过 |
| 6.13 Cosine/线性等调度 | 核心 | 1,890 | 1,800，通过 |
| 6.14 WSD/WSD-S | 核心 | 1,892 | 1,800，通过 |
| 6.15 Weight decay | 核心 | 1,828 | 1,800，通过 |
| 6.16 Gradient clipping | 核心 | 1,821 | 1,800，通过 |
| 6.17 Batch scaling | 核心 | 1,875 | 1,800，通过 |
| 6.18 公平实验 | 核心 | 1,862 | 1,800，通过 |
| 6.19 教学代码审计 | 辅助 | 1,070 | 1,000，通过 |
| 6.20 Checkpoint 恢复 | 辅助 | 1,139 | 1,000，通过 |
| 6.21 Loss-spike 诊断 | 辅助 | 1,076 | 1,000，通过 |

6.0（484 字）是章节导读与上位问题，不独立承担机制教学，故不按单元计；6.22 资料边界、6.23 练习、6.24 小结不作为知识讲解单元。
- **example_status：** 6.2 用 $L=\frac12(100x^2+y^2)$ 手算学习率递推、过大曲率下的误差增长和 momentum 方向滞后；6.3 手算两步 Adam 偏置修正、epsilon 位置及 beta2 状态半衰期；6.4 比较一/二坐标 AdamW 与耦合 L2，并以两步示例显示 L2 项写入历史状态。6.5 用 1B 参数账本估算 BF16 权重/梯度、FP32 状态与分片口径；6.6 对 $2\times3$ 梯度平方矩阵计算行列统计、$70/23$ 与 $117/23$ 的重构值及预条件更新；6.7 手算 Lion 两坐标动量、decoupled decay 与 sign 向量范数；6.8 手算 $2\times2$ Hutchinson 对角估计、gamma 裁剪和二维曲率例；6.9 用秩亏 polar 因子、$\operatorname{diag}(9,1)$ 更新及形状 RMS 比较矩阵方向；6.10 手算双侧 GaLore 投影/映回、状态/基显存、旋转基影响和奇异值能量保留。

6.11 用 tied embedding/head 的对象身份算独立参数 6,272，并手算参数组 AdamW 更新；6.12 用四步 warmup、二次目标稳定界和不同 group LR 解释峰值风险；6.13 在 10-step 预算中对照 cosine/linear 的中段 LR，并计算预算从 10 延至 20 时进度重映射；6.14 用 100-step WSD/WSD-S 和 100B 主干+20B 分支账本区分 checkpoint 语义；6.15 计算 $0.999^{100}$、$0.999^{50}$ 及 1B token 下 1M/2M batch 的纯 decay；6.16 计算 $(3,4)$ 全局裁剪、两 micro-batch 裁剪次序、跨 shard norm 与 Adam 首步缩放；6.17 比较 256K/512K/1M 有效目标 batch，并以两 rank 3:1 目标数得到 token mean 1.5、等权 rank mean 2；6.18 用假设 token-efficiency/吞吐算例拆分达标 token 与墙钟；6.19 手算代码第一个 AdamW 更新且把 `effective_tokens` 界定为名义 slots；6.20 对比恢复 $m,v$ 与 reset 后的下一步方向；6.21 用 $g=(60,80)$、阈值 1 和有效目标数变化说明应沿时间序列定位第一偏离量。所有数值均为教学输入或来源报告范围，不冒充本地训练实测。
- **code_status：** 6.19 Python 标准库代码块及可执行语句本轮未改，继续标记“静态审阅，未运行（2026-10-30）”。新增的运行入口和断言解释明确标成代码逻辑推得的预期，而非 stdout。本轮未执行章节代码、测试、模型、benchmark 或 PDF。
- **formula_status：** 静态复核 SGD/Momentum 两种参数化、Adam 偏置修正/epsilon、AdamW 与耦合 L2 状态差、Adafactor 行列均值重构与偏置修正、Lion 符号方向、Sophia 两估计器/刷新/初始 $h_{1-k}=0$/decay/clipping、Muon 部分等距 polar 与满秩 RMS、GaLore 双侧 $P^TGQ$、schedule 边界、累计 decay、global norm clipping、token weighted mean 与 checkpoint 下一步状态。概率、矩阵、步数、batch 和预算条件附在对应解释和算例；公式函数记号遵循规范，不使用 `\operatorname`。
- **visual_status：** 保留参数组/优化器取舍表、梯度裁剪顺序和 checkpoint 字段表；新增 WSD/WSD-S checkpoint 对照表，说明主干、候选分支及恢复点。未新增 Mermaid；本章的数据流和学习率轨迹用公式与带输入的数值对照已足以核验。
- **link_status：** 第六章内 44 个 Markdown 链接均为直接论文/官方文档入口；简介 20 个本地链接目标存在。新加入 `EXERCISES.md` 的第五册第六章入口目标存在；`git diff --check` 通过，代码/公式围栏和行内公式分隔做静态配对检查。对完整 `EXERCISES.md` 扫描另发现 24 个此前已有、位于其他模型专题的缺失本地研究代码/笔记目标，本章新增练习没有引用这些旧路径；需在后续配套资料专项审阅时处理。
- **sync_status：** 更新第五册简介第 06 章摘要；在 `EXERCISES.md` 新增四题，覆盖累积后裁剪、token/墙钟 break-even、WSD 分支预算和变长 rank 的 token weighting。`PAPERS.md` 已有 Adam/AdamW、Adafactor、Lion、Sophia、Shampoo/SOAP、GaLore、Muon、WSD 及系统比较条目；`GLOSSARY_EN_ZH.md` 已有相关方法、更新 RMS、WSD、weight decay、clipping 词条，本轮不重复添加。Sophia 原算法关于初始曲率状态与独立衰减的新核验记入 `research/optimizer-alternatives-and-scaling-2026.md`。
- **open_risks：** 所有 optimizer 性能数字仍受论文模型、数据、硬件、预算和调参影响，未本地复现；6.19 代码仍静态未运行。`EXERCISES.md` 的 24 个旧缺失目标需后续确认应补档、改链或撤销路径；第五册 07—18 章、整册沿主线复读及全书链接/外链门禁待完成。未处理 PDF。

## 第 07 章：分布式训练——精修记录

- **scope：** 完整审阅并精修 `book-05-llm-training/chapters/07-分布式训练.md`，从 rank/process group、有效目标分母和 collective 出发，串起 DDP、ZeRO/FSDP、TP/PP/SP/CP/EP、通信重叠、checkpoint 重分片、可观测性与故障诊断。保持第 07 章及 7.0—7.21 的编号、章节位置和第五册骨架。
- **chapter_scan：** 2026-10-30；扫描不等长 rank 的 token-mean、DP/FSDP 状态分片、TP/PP/SP/CP/EP 的张量布局与通信、MoE all-to-all、流水线 bubble、激活重计算、通信重叠、checkpoint world-size 迁移、hang 诊断和 MFU/HFU 口径。阅读 Zero Bubble Pipeline Parallelism v1 摘要与 §3、DeepSeek-V3 技术报告 §3.2.1–3.2.2/Table 2、DynaTrain arXiv v1 摘要与 §§3–5、PyTorch 2.15 发布说明及 DCP/FSDP2 文档；复核 Megatron Core hierarchical DP、Context Parallel 指南、Megatron-LM、GPipe、Ring Attention 与 PaLM 的对应段落。新来源事实及证据边界见 [`research/llm-training-and-post-training-systems-2026.md`](../../research/llm-training-and-post-training-systems-2026.md)。候选处置：① ZeroBubble 与 DualPipe 纳入 schedule 对照，限定其 backward 拆分、通信重叠、参数/激活账本及偶数配置条件，不泛化为任意设置的零 bubble/固定提速；② DynaTrain 作为在线并行切换的早期预印本分支，保留作者报告的 70B dense `<2 s`、235B MoE `4.36 s` 和缺少独立复现的边界，不写成 DCP 或生产 runtime 已具备的能力；③ hierarchical DP 只解释 Megatron Core 当前配置语义，不据此推断吞吐；④ PyTorch 2.15 flight recorder、FSDP2 精度/gradient accumulation 和 pipeline buffer 生命周期均明确绑定版本；⑤ 没有发现需新增章节或改变骨架的候选。
- **architecture/boundary：** 保留现有分布式训练主线；章首补齐物理 GPU 集群、网络/集合通信和训练平台工程到第二十三册的路由。将“正确形成一次全局更新”“并行维度如何切分工作”“系统如何运行与恢复”分成连续解释链；DP token 分母例显式区分 rank mean 与 token mean；TP/PP schedule、SP/CP、MoE 与 checkpoint 示例把张量区间、通信和状态所有权写清。SP 只描述特定逐 token 激活布局，CP 不宣称消除全局 attention 的二次 token-pair 关系；DCP 的 load-time resharding 与应用级 manifest/data cursor/RNG 责任分开。
- **length_status：** 2026-11-02 静态计数；只计解释性中文，排除标题、表格、公式/代码围栏、状态标记。核心下限 1,800，辅助下限 1,000。按连续论证链分组：7.1—7.5 讲全局更新、通信与常驻状态；7.6—7.10 讲并行切分和组合；7.11—7.12、7.13—7.14、7.15—7.18 分别作为激活/通信优化、分布式恢复、观测与运行诊断的辅助链，其深入主讲分别交给第八、九/十章及第二十三册。

| 单元 | 类型 | 汉字数 | 下限/状态 |
| --- | --- | ---: | --- |
| 7.1—7.5 全局更新、collective 与状态分片 | 核心 | 3,120 | 1,800，通过 |
| 7.6—7.10 TP/PP/SP/CP/EP 与并行组合 | 核心 | 3,147 | 1,800，通过 |
| 7.11—7.12 激活重计算与通信重叠 | 辅助 | 1,128 | 1,000，通过 |
| 7.13—7.14 checkpoint 与失败恢复语义 | 辅助 | 1,108 | 1,000，通过 |
| 7.15—7.18 观测、布局选择、账本与排障 | 辅助 | 1,409 | 1,000，通过 |

7.0（288 字）是章首路线图；7.19 资料、7.20 练习和 7.21 小结不按知识讲解单元计。
- **example_status：** 7.2.2 以三个 rank 的有效目标数 100/200/400 和 loss sum 50/110/220 得到全局 token-mean `380/700≈0.5429`，与 rank-mean `≈0.5333` 对照，并推导 DDP rank-average 下局部 loss 的 `N_dp/N` 缩放。7.5.8 用 7B 参数、BF16 参数/梯度各 2 bytes、AdamW 两个 FP32 moment 共 8 bytes/parameter、DP=8 算得 DDP 84 GB、ZeRO-1 35 GB、ZeRO-2 22.75 GB、ZeRO-3 10.5 GB 的稳态状态估算，并明确排除 master weight、activation、通信与临时 buffer。7.6 用两个 2×2 矩阵逐步算出列切分局部输出 `[5,2]`、`[2,-1]` 和拼接结果 `[5,2,2,-1]`，再算行切分局部结果 `[5,2]`、`[-1,3]` 并 reduce-sum 为 `[4,5]`。7.7.2 对 `p=8,m=16` 算 43.8% bubble/ideal overhead 和 30.4% idle/total fraction，区分分母；7.7.5 用 8 stage、20 micro-batch、`F=4 ms`、`B=7 ms`、`W=2 ms`、显式假设 `F&B=8 ms`，按 DeepSeek-V3 Table 2 口径得到 1F1B 77 ms、ZB1P 49 ms、DualPipe 27 ms 的 bubble 项。7.8 以 `B=1,L=4096,d=8192,TP=8` 得单 rank `[1,512,8192]` 的 SP 激活布局，并以 `L=16384,CP=4` 得 4096 个本地 query 位置，说明 K/V ring 和 buffer 仍有峰值成本。7.12 以 80 ms 计算、30 ms 通信、隐藏 15 ms，得串行 110 ms、理想 80 ms、简化暴露时间 95 ms。7.13.4 对 12 元素向量逐项比较 DP=4 与 DP=3 的 shard 区间，并以 global step 100、sample cursor 48 说明状态/数据游标边界。7.15.5 假设 1,600 token/s、`2×10^12 FLOPs/token` 和 `8×10^15 FLOP/s` 得 MFU 40%；另加 `1×10^12` 重算 FLOPs/token 得 HFU 60%。自构造值均标为教学输入；PaLM、DeepSeek-V3、DynaTrain 数字均保留来源配置边界。
- **code_status：** 7.17 的 Python 标准库资源账本可执行语句未改，仍标记“静态审阅，未运行（2026-10-30）”；只核对代码及邻近说明，没有运行章节代码、测试、模型、benchmark 或 PDF。
- **formula_status：** 静态复核有效 token 总分母及 DDP rank-average 缩放、ring all-reduce 的通信量级、ZeRO/FSDP 状态显存、TP 的 concat/reduce-sum、PP bubble 比例和表内 schedule、SP/CP shape 与 attention 关系、MoE capacity/all-to-all、overlap、DCP shard 映射、MFU/HFU 和资源账本。核心算式配有变量定义、单位、步骤、结果和适用限制；无 `\operatorname`、带下划线的 `\text{}` 代码变量或尖括号下标。
- **visual_status：** 保留并解释并行方式、collective、schedule、拓扑和诊断表；用 TP 矩阵、时间算例和 checkpoint 区间列出可复核轨迹。未新增 Mermaid，现有表格与手算轨迹足以表达关键关系。
- **link_status：** 2026-11-02 静态检查发现 122 个围栏标记（61 对）；全章 4 个本地链接目标均存在；公式/文本围栏约束检查通过；`git diff --check` 通过。未执行整册外链可达性检查；来源适配按本章定向扫描和研究笔记记录，最终外链及跨册锚点门禁留到第五册整册审阅。
- **sync_status：** 更新第五册简介第 07 章摘要；`EXERCISES.md` 新增 TP 手算、ZeroBubble/DualPipe bubble 与状态开销、DCP 重分片/DynaTrain、PyTorch 2.15 hang 诊断四题；`PAPERS.md` 补入 ZeroBubble、DualPipe、DynaTrain 和 PyTorch 2.15/Megatron Core 资料；`GLOSSARY_EN_ZH.md` 新增 DualPipe、Zero Bubble/ZB1P、HSDP；`INTERVIEW_BANK.md` 更新 56A/56C/56D/56F；`KNOWLEDGE_GRAPH.md` 增加分布式训练链和研究分支。研究事实只写入 `research/`，修改与验收状态只记本文件。
- **open_risks：** 7.17 toy 代码仍未运行；本地静态推导不能证明真实集群的吞吐、精度、hang 诊断或 checkpoint 可恢复。DynaTrain 是单篇预印本，PyTorch 2.15 和 Megatron Core 在线文档均有明确版本/持续更新边界。第五册其余章节、整册连续复读和外链/锚点门禁待做；`EXERCISES.md` 先前发现的 24 个无关旧缺失本地研究链接仍待配套资料专项审阅。未处理 PDF。

## 第 08 章：显存优化与数值精度——精修记录

- **scope：** 完整审阅并精修 book-05-llm-training/chapters/08-显存优化与数值精度.md，完成章级定向扫描与本轮所需配套同步。保留训练状态、激活、attention 和临时空间的统一账本，沿峰值测量、精度与重计算、状态/微调策略、OOM/NaN 诊断形成一条训练资源决策链；不改第五册骨架。
- **chapter_scan：** 2026-11-02；扫描 memory allocated/reserved 与设备进程读数的差别、PyTorch allocator snapshot 的可见范围、动态 shape 分配配置、长上下文激活内存机制、全量 FP4 与 NF4 的区别、AMP/compile 交互和 checkpoint 重算语义。阅读 OOMB arXiv v4 的摘要、方法、实验与限制；FP4 All the Way v2 的方法、实验与限制；Quartet v4 的格式、训练结果和系统边界；PyTorch 2.15 CUDA memory、AMP、AMP examples、checkpoint 与 type-info 文档。候选去向：① OOMB 作为长上下文系统级案例并入 8.8.3，不改变 activation checkpointing 的通用解释；绑定 Qwen2.5-7B、H200、Dense Attention 配置及作者报告的约 10MB/10K-token 增量，并保留串行时延、稀疏近似和互连限制；② FP4 All the Way 与 Quartet 纳入 8.6.6，对照格式、舍入、实验和硬件证据，不推广为通用训练配方；③ expandable_segments 作为 PyTorch 2.15 实验选项说明，不表述为释放活张量的通用 OOM 修复；④ pinned-host snapshot 用带下划线的版本相关内部 API 明确标边界；⑤ AMP API 弃用、torch.compile backward autocast 默认和 checkpoint 显式 use_reentrant 按 2.15 文档更新；⑥ 未发现需增章或改骨架的候选。新事实和来源证据写入 research/activation-memory-and-low-precision-training-2026.md。
- **architecture/boundary：** 本章仍承担训练资源预算与数值异常的工程决策；第九章继续主讲稳定性根因链，第十三章主讲资源规划，第二十三册承接 GPU 集群与互连系统设计，第六册的 FP4/NF4 serving 量化不与本章全量 FP4 训练混为一谈。新增 FP4 训练/QLoRA 对照和普通 checkpoint/OOMB 系统对照表。
- **length_status：** 静态计数只统计连贯中文解释字符，排除标题、表格、代码/公式围栏、内联代码、维护状态和链接目标；按连续论证链分组，核心下限 1,800、辅助下限 1,000，均通过。

| 单元 | 类型 | 汉字数 | 下限/状态 |
| --- | --- | ---: | --- |
| 8.0—8.5 显存对象、7B 账本、峰值、激活与 attention | 核心 | 2,873 | 1,800，通过 |
| 8.6—8.8 数值格式、AMP、重计算与长上下文系统 | 核心 | 2,780 | 1,800，通过 |
| 8.9—8.11 有效 batch、状态位置与参数高效微调 | 核心 | 2,050 | 1,800，通过 |
| 8.12—8.16 OOM/NaN 诊断、策略、预算 toy 与实验流程 | 辅助 | 1,450 | 1,000，通过 |

- **example_status：** 复核 7B BF16 权重为 14e9 bytes = 13.04 GiB；原稿“少算约 7%”方向错误，现修为误记 14 GiB 会相对正确值高估约 7.4%。同时复核全参 AdamW 16 bytes/parameter、hidden activation 和 attention score 的 T/T² 账本、online softmax 输出约 18.45、checkpoint 的 K+L/K 近似、8-bit state 节省量、LoRA 参数与 QLoRA NF4 metadata。OOMB 的 4M context 和 10MB/10K-token 为来源报告，不当作本地测量或普通 checkpoint 保证；FP4 质量/速度结果保持对应模型、语料、算法与硬件条件。
- **code_status：** 8.15 唯一 Python 教学代码块未改；静态提取对照确认其 Python fenced block 与原文件一致。保留正文原有“已验证（2026-10-26，Python 标准库；解释器小版本未记录）”标记，不重跑；维护风险中明确缺少解释器小版本信息。本轮未执行章节代码、测试、模型、benchmark 或 PDF。
- **formula_status：** 修正 GB/GiB 换算解释；将 bytes、clip、round 的公式函数记号改用规范允许的 \mathrm；静态复核显存、精度、量化、梯度累积、重计算、optimizer-state、LoRA/QLoRA 和 toy 预算算式。无 operatorname、尖括号下标、带下划线的 text 变量或未配对行内美元符号。
- **visual_status：** 保留状态账本、精度/量化表和 OOM 策略表；新增 FP4 recipe/证据边界表和普通 activation checkpointing/OOMB 系统设计对照表。现有数值关系可由公式和表格直接复核，无需新增 Mermaid。

- **link_status：** 第八章 98 个 fence 标记成 49 对；2 个本地 Markdown 目标存在；章节、简介和新增练习的行内/本地链接静态检查通过。公式记号、行内美元分隔检查通过；git diff --check 通过。未做全章外链可达性请求。完整 EXERCISES.md 扫描中可见先前已记录、属于其它专题的缺失目标，本轮新增第五册第八章链接不命中这些旧路径。
- **sync_status：** 更新第五册简介第 08 章摘要；EXERCISES.md 新增四道显存单位、allocator 诊断、OOMB 与 FP4 比较练习；PAPERS.md 补入 OOMB、两篇 FP4 论文和 PyTorch 2.15 文档；GLOSSARY_EN_ZH.md 增补全量 FP4 训练、OOMB 和 QAF；INTERVIEW_BANK.md 更新 allocated/reserved/OOM 表述并新增 FP4/QLoRA 与 OOMB 面试题；KNOWLEDGE_GRAPH.md 补充训练显存、低精度与长上下文系统依赖链。来源事实只记于 research，章节决定和验收状态只记于 progress。
- **open_risks：** OOMB 和 FP4 数据均为论文作者报告，未在本地硬件复现；Gaudi2 的 FP4 运行是模拟，Quartet 的 RTX 5090 速度是局部 kernel 测量。PyTorch 2.15 的 expandable_segments 为实验选项，pinned-host snapshot 使用版本相关内部 API；目标版本应重新核对。代码既有验证标记未记录 Python 小版本。第五册全册连续复读、全部外链终审和跨章锚点门禁待全册结束时完成；PDF 未处理。

## 第 09 章：训练稳定性与调试——精修记录

- **scope：** 完整审阅并精修 book-05-llm-training/chapters/09-训练稳定性与debug.md，保持 9.0—9.16 编号和第五册结构。沿“健康基线—首个异常—候选机制—固定现场反事实—修复回归”组织解释，覆盖有效 token/loss 分母、loss spike、NaN/Inf、梯度、数据与 mask、optimizer/scheduler、混合精度、分布式同步和恢复连续性。
- **chapter_scan：** 2026-11-02；扫描 loss-spike 的 Adam 更新方向/时间相关梯度假说、自适应预条件器与曲率代理、SPAM 的在线代理/周期重置、AdaGC 逐张量阈值、GradientStabilizer 幅值控制；核对 PyTorch 2.15 anomaly detection、非有限总梯度范数检查、TORCH_DIST_NAN_CHECK、DDP 同步点/参数顺序及 DCP 接口边界。候选处置：① Molybog、Bai、SPAM、AdaGC、GradientStabilizer 作为同一 spike 问题下机制不同的假说和对照方案保留在 9.4.5，不按未统一指标排胜负；② anomaly detection 只表述为 backward NaN/forward traceback 辅助，clip 检查总范数，TORCH_DIST_NAN_CHECK 限于 collective 发送 buffer 中的 NaN，均不替代 first-bad-tensor/rank 日志；③ 两 rank NaN/hang 和 data cursor 漂移补入分布式及恢复链；④ 无新增独立讲次或结构调整。新核验事实及来源范围见 [research/training-stability-and-debug-2026.md](../../research/training-stability-and-debug-2026.md)；AdaGC 详细证据沿用 [research/initialization-and-training-stability-2026.md](../../research/initialization-and-training-stability-2026.md)。
- **architecture/boundary：** 9.0—9.4 建立健康账本、首异常检测与 spike 证据链；9.5—9.9 按坏值、梯度、数据、优化器和精度边界缩小根因；9.10—9.11 分开 collective 控制流、rank 数据差异与 checkpoint/data cursor 连续性；9.12—9.13 用静态 toy 诊断和可复盘假设收束。保留框架接口、论文机制、应用层恢复协议的责任边界；全量 checkpoint 管理/实验治理仍由第十章主讲，硬件集群与互连交接第二十三册。
- **length_status：** 2026-11-02 静态计数，只计连贯解释性汉字，排除标题、表格、公式/代码围栏、内联代码、状态标记及链接目标。核心下限 1,800，辅助下限 1,000；各组通过。

| 单元 | 类型 | 汉字数 | 下限/状态 |
| --- | --- | ---: | --- |
| 9.0—9.4 健康基线、异常时间轴与 spike 假设 | 核心 | 2,623 | 1,800，通过 |
| 9.5—9.9 坏值、梯度、数据、更新与精度排查链 | 核心 | 1,889 | 1,800，通过 |
| 9.10—9.11 collective 与恢复连续性 | 辅助 | 1,048 | 1,000，通过 |
| 9.12—9.13 toy 轨迹与复盘记录 | 辅助 | 1,067 | 1,000，通过 |

9.14 来源边界、9.15 练习和 9.16 小结不作为独立知识讲解单元计数。
- **example_status：** 9.4.5 手算普通 GD 的一维例：eta=0.001、常数曲率 h=2400，eta*h=2.4、参数乘以 -1.4、单步 loss 约为原来的 1.96 倍，并明确这不是 Adam spike 预测。9.6.4 以 (T1,G1)=(100,200)、(T2,G2)=(300,300) 算出 token mean 1.25，对照等权局部 mean 1.5（高 20%）；9.7.2 算 10/1024≈0.98% 有效监督比例，并以 span decode 区分低比例与 mask 错误；9.11.2 算 480,000×500=240M，有效 token cursor 245M 时相差 5M。9.12 逐步追踪 step 4 的中位数 2.18、loss ratio 2.66、grad norm 7.5、length 8192 与 valid ratio 0.03；step 6 的 NaN/Inf 和 rank 非有限值；checkpoint 缺 scheduler/scaler/RNG/data position，故 step 5 健康日志不能冒充完整回滚点。以上均为教学构造输入。
- **code_status：** 9.12 唯一 Python 代码块的可执行语句本轮未改；静态提取与 HEAD 原块一致。未找到可追溯执行记录，故将原“已验证”状态修正为“静态审阅，未运行（2026-11-02）”。手工核对 step 4、step 6、rank spread、checkpoint 缺项和 rollback 判定的预期逻辑；未运行章节代码。
- **formula_status：** 静态检查有效 token loss、spike ratio/异常指示、median、global norm clipping、relative update、accumulation token-mean、label ratio、AdamW 单步 decay、GD 路径平均 Hessian 条件、HVP 曲率代理和 checkpoint token cursor 算术。未发现 \operatorname、尖括号下标、带下划线的 \text{} 代码变量或未配对行内美元分隔；GD 教学例明确带常数曲率条件。
- **visual_status：** 保留异常时间线、根因/证据/处理边界表、恢复状态账本及逐步诊断示例。机制、分支和状态已有表格与序列化流程表达，无需增加装饰性 Mermaid。
- **link_status：** 章节 62 个 fence 标记配成 31 对；16 个 Markdown 链接中 14 个外链、2 个本地目标均存在（第 04、10 章）；第五册简介的 20 个本地链接目标均存在。静态公式约束及行内美元配对通过；git diff --check 通过。本轮未执行全章外链可达性或 GitHub 渲染检查。
- **sync_status：** 更新第五册简介第 09 章摘要；EXERCISES.md 的第五册第九章追加第五题比较 global clipping、AdaGC、beta2、SPAM 与 GradientStabilizer；INTERVIEW_BANK.md 更新 Q35 纳入 AdaGC；KNOWLEDGE_GRAPH.md 增加训练稳定方法链。PAPERS.md 已有 Adam instability/Bai/SPAM/GradientStabilizer/AdaGC 来源入口，GLOSSARY_EN_ZH.md 已有 AdaGC、GSS、GradientStabilizer 词条，本轮不重复添加。研究事实只记 research，修改决定与验收只记本进度文件。
- **open_risks：** 论文数字和方法表现绑定各自的模型、数据、step/token 预算、并行布局、指标与硬件，未在本地复现；SPAM 的 60M token 数在论文主表与 Appendix D 之间为 1.1B/1.3B 不一致，正文显式保留冲突。HVP 监控不是免费常规指标，PyTorch 2.15 行为只对应该版官方文档。9.12 示例仍未运行；第五册第 10—18 章、整册连续复读和最终链接/锚点门禁待做。未运行测试、模型、benchmark 或 PDF。

## 第 10 章：Checkpoint、评估与实验管理——精修记录

- **scope：** 完成 `book-05-llm-training/chapters/10-checkpoint评估与实验管理.md` 的章级定向扫描、正文精修和知识配套同步。沿“状态能否被正确保存—故障后能否连续恢复—指标是否对应同一评测协议—实验差异能否解释—候选 checkpoint 是否满足项目门槛”组织 1—19 节。第七章主讲分布式并行与 DCP 重分片，第九章主讲训练异常和根因排查，第十三章负责训练资源与成本预算；本章保留训练状态、评测合同、实验血缘和版本选择的全生命周期闭环。
- **chapter_scan：** 2026-11-02；检索和核对 PyTorch 2.15 DCP/save/load 与 async staging/completion、随机性限制、FSDP 状态边界；Transformers Trainer 当前文档的恢复语义；固定长度 causal LM PPL 的窗口、stride、target mask 和分母；HELM 的场景/指标覆盖口径；lm-eval 的可复现实验协议、成对统计和污染审计；Oren 顺序似然、ConStat 表现差距信号及 NLP 显著性检验指南。候选处置：① 将恢复字段写成按精度/训练配置生效的状态集合，FP16 scaler 仅在实际使用时要求；② 把 DCP 的本地 staging 完成与存储写入完成分开，并把 manifest/版本发布留给应用协议；③ 将固定窗口 PPL、按有效 token 聚合的 domain loss、benchmark 协议和 regression 样本分母并入同一评估合同，避免不同 mask/stride/样本集的结果直接比较；④ 按语料可见性和黑盒接口条件选择污染审计，不把顺序似然或异常表现当作训练样本包含的直接证明；⑤ 用预注册硬门槛过滤候选，再做归一化打分或保留 Pareto 前沿，显式声明 toy 示例未含推理成本；⑥ 随机种子不作为跨版本、平台或硬件的逐位复现保证，公平对照需记录数据、token 预算、环境和评测版本；⑦ 未发现需增加讲次、迁移主题或调整章节骨架的候选。版本事实及来源边界见 [research/checkpoint-evaluation-and-experiment-management-2026.md](../../research/checkpoint-evaluation-and-experiment-management-2026.md)；HELM、统计检验和污染审计的研究证据按正文来源及既有来源笔记追溯。
- **architecture/boundary：** 保持 10.0—10.19 结构。10.1—10.4 形成状态完整性、保存频率/容量与恢复探针；10.5—10.9 形成 validation、domain、benchmark、回归及污染审计合同；10.10—10.13 形成 run manifest、命名/版本、复现边界和受控对照；10.14—10.15 形成门槛、归一化与最小可运行选择流程；10.16 为误判诊断辅助链。没有新增跨册主讲；训练稳定性根因回链第九章，DCP/并行重分片回链第七章，完整资源预算回链第十三章。
- **length_status：** 2026-11-02 静态计数，只计连贯中文解释性汉字，排除标题、表格、代码围栏、行内代码、公式内容及代码状态标记；按连续解释链分组。核心单元下限 1,800，辅助单元下限 1,000，均通过。

| 单元 | 类型 | 汉字数 | 下限/状态 |
| --- | --- | ---: | --- |
| 10.1—10.4 checkpoint 状态、频率、存储与恢复 | 核心 | 3,132 | 1,800，通过 |
| 10.5—10.9 validation、domain、benchmark、regression 与污染审计 | 核心 | 3,181 | 1,800，通过 |
| 10.10—10.13 实验追踪、版本、复现与公平对照 | 核心 | 1,854 | 1,800，通过 |
| 10.14—10.15 候选门槛、多目标选择与可运行 demo | 核心 | 2,015 | 1,800，通过 |
| 10.16 常见误判与检查路径 | 辅助 | 1,326 | 1,000，通过 |

10.0 为章首路线图；10.17 小结、10.18 练习和 10.19 资料不作为独立知识讲解单元计数。
- **example_status：** 10.2 用每 step 2M 有效 token、500-step 保存间隔得到最大回退 1B token；12,000 step 共 24 次保存；120 秒写入、4 秒/step 时串行开销为 6%。10.3 用 7B BF16 权重 14e9 bytes≈13.04 GiB 及优化器恢复状态估算容量，并以 latest/best 重叠 ID 集合验证只计唯一 artifact。10.4 逐字段核对恢复状态，并用 step 2000→2001、loss、LR 与 consumed-token 连续性组成一次恢复探针。10.6 手算两目标 PPL；10.6 另以 context 4、stride 2、7-token 序列说明重叠窗口只计 6 个唯一目标，非重叠示例漏计 1 个；10.7 以 80 个 loss=2 token 和 20 个 loss=3 token 得有效 token 加权 loss 2.2，与 domain macro mean 2.5 对照。10.9 以 4/80=5% 计算 regression rate；10.10 用 4/6 个有值字段得到 66.7% coverage，并说明它不是实验成功率；10.13 以固定数据和 104.9B token 只改 LR 形成公平对照，对比数据版本和训练量共同变化的混杂例。10.14—10.15 手算 `ckpt_2000` 四分量加权分数约 0.595，解释 `ckpt_3000` 的 stability/safety/regression/domain gate；新增归一化截断、macro-average、门槛预注册、成本缺失、Pareto 前沿和精度/tie-break 例子。所有数值均为明确的教学输入，不是训练或服务实测。
- **code_status：** 10.15 标准库 Python demo 的排序代码由舍入值改成全精度比较，展示分数只舍入至四位；增加全精度同分时取较早 step 的确定规则。本轮用 Python 3.12.9 从 10.15 Python 围栏提取并执行；运行输出与文中预期一致：完整性/恢复/对照诊断均为 True、选择 `ckpt_2000`、保留 `ckpt_2000` 与 `ckpt_3000`、存储 360 GiB、保存开销 0.0075、最大回退 52.5B tokens。输入校验和异常分支只作静态审阅，未逐项执行；未运行训练、DCP、多 GPU、对象存储或外部 benchmark。
- **formula_status：** 静态复核精度条件化 `C_resume`、保存频率的 RPO/串行开销、去重 artifact 存储、全局有效 token loss、PPL 的 `exp(loss)`、stride mask 分母、domain token-weighted mean、benchmark 统计及 paired comparison、regression rate、实验字段 coverage、归一化多目标 score 与 Pareto 支配关系。手算中间值和单位均已给出；无 `\operatorname`、尖括号下标、带下划线的 `\text{}` 代码变量或未配对行内美元。
- **visual_status：** 保留并补充恢复字段/快照生命周期、评测协议、实验变量差异和候选门槛表；PPL 窗口、step/token、artifact ID 和候选排序均有逐步文字轨迹。现有表格加手算足以展示状态和决策关系，未新增 Mermaid。
- **link_status：** 本章 86 个 fence 标记配为 43 对；21 个 Markdown 链接均为外部来源，本章没有本地文件/fragment 链接；第五册简介 20 个本地文件链接目标均存在，无 fragment。禁止公式记号扫描、行内美元配对及本地路径检查通过；`git diff --check` 在本记录写入前通过，最终会再复核。未执行外链全量可达性检查或 GitHub 渲染检查。
- **sync_status：** 更新第五册简介第 10 章摘要；`EXERCISES.md` 新增练习 181，覆盖最大回退量、串行保存开销、latest/best 唯一 artifact 和恢复探针；`KNOWLEDGE_GRAPH.md` 新增 checkpoint—评估—实验管理子链。`PAPERS.md`、`GLOSSARY_EN_ZH.md`、`INTERVIEW_BANK.md` 已有本章所需论文/术语/题目入口，本轮不重复添加。PyTorch DCP、随机性、Trainer 动态文档及 HELM 新核验事实写入 research；审计结论和修改过程仅存本 progress 文件。
- **open_risks：** Transformers `main` 文档为动态页面，访问日显示 v5.22.1；具体运行仍须用锁定版本复核。PyTorch DCP 2.15 保存状态不承诺跨版本兼容，应用 manifest 和对象存储发布事务不由 DCP 自动保证。HELM 覆盖率只对其论文定义场景集合有效；污染信号需按语料可见性和接口能力解释。demo 不含延迟/推理成本输入，完整部署决策应以实际 SLO/硬件条件加入成本或保留 Pareto 候选。代码未覆盖真实训练和异步存储故障演练；21 个外链未做本轮在线可达性复查。第五册第 11—18 章、整册连续复读和最终外链/锚点门禁待完成。未更新 PDF。

## 第 11 章：后训练与对齐训练——精修记录

- **scope：** 完成第五册第 11 章的章级扫描、正文精修、数值例核对和配套同步，保留原有结构。全章沿“反馈来源与行为目标 → 在线/离线更新方法 → reasoning RL 的更新粒度与长度处理 → 公开训练案例 → 方法选择、数据、风险和工程诊断”组织。第十五章主讲领域专家任务，第十六章主讲 RLVR/verifier，第十七章主讲 OPD，第十八章主讲 DeepSeek-V4 训练披露；本章保留通用方法地图，不迁移这些完整专题。
- **chapter_scan：** 2026-11-02；围绕 2025—2026 后训练路线复核 PPO/RLHF、偏好优化与 RLVR，定向阅读 DAPO v2、GSPO v2、DrGRPO v2、Dynamic Clipping DCPO v2、同名的 Distribution-Centric DCPO v1，以及 DHPO v1/GAPO v1 的原始论文方法与实验段；复用 InstructGPT、PPO、DPO、RLVR 噪声和异步 RL/OPD 来源记录，并核对 GLM-5.3 官方披露、gpt-oss 发布说明/模型卡及 Grok 4.7 官方发布页。候选处置：① DAPO 保留为组合 recipe，progressive 分数不解释为独立随机消融；② DrGRPO 作为 response-length 与组内标准差归一化的目标修正，边界限定在作者推导和训练条件；③ GSPO 解释序列几何平均 ratio 与外层 loss reduction 的分离，不把它当作普通整句重要性比；④ DCPO 明确全称、动态 clip/SAS/OTM，并区分同名 Distribution-Centric DCPO；RUR 仅作为论文定义的训练期非零 advantage 占比；⑤ DHPO/GAPO 仅列为早期预印本候选并固定 v1 链接，不升格为通用标准；⑥ 反馈标签/verifier 噪声纳入风险排查；⑦ 长周期环境、异步 rollout、stale policy 与公开模型案例保留在本章概览，具体 verifier 工程、领域流程、OPD 和厂商配方继续回链第十五至十八章。无新增或拆分章节。方法版本、来源支持和未验证边界见 research/rlhf-preference-and-policy-optimization-2026.md 与 research/llm-training-and-post-training-systems-2026.md。
- **architecture/boundary：** 保留原有二十个编号小节。第 1—2 节与章首问题组成后训练路线和方法层级入口；第 3—9 节分别主讲 PPO-RLHF、DPO、SimPO、GRPO、GSPO、DAPO、DCPO；第 10—12 节由 DeepSeek-R1 过渡到现代模型披露；第 13—16 节形成方法选择、数据合同、风险诊断和 pipeline 的应用决策链；第 17—18 节为分析 demo 与诊断辅助链。没有将机制不同的核心方法合并计数。
- **length_status：** 2026-11-02 静态计数，仅计连贯解释性汉字，排除标题、代码围栏及代码、表格、公式、行内代码、维护状态标记；链接文字计入。核心单元下限 1,800，辅助单元下限 1,000，均通过。

| 单元 | 类型 | 汉字数 | 下限/状态 |
| --- | --- | ---: | --- |
| 章首、为什么需要后训练、第 1—2 节路线与方法层级 | 核心 | 3,338 | 1,800，通过 |
| 第 3 节 PPO-RLHF | 核心 | 1,869 | 1,800，通过 |
| 第 4 节 DPO 与偏好数据形态 | 核心 | 1,820 | 1,800，通过 |
| 第 5 节 SimPO | 核心 | 1,829 | 1,800，通过 |
| 第 6 节 GRPO | 核心 | 1,910 | 1,800，通过 |
| 第 7 节 GSPO | 核心 | 1,893 | 1,800，通过 |
| 第 8 节 DAPO | 核心 | 1,879 | 1,800，通过 |
| 第 9 节 Dynamic Clipping DCPO | 核心 | 1,899 | 1,800，通过 |
| 第 10 节 DeepSeek-R1 / RLVR 案例 | 辅助 | 1,003 | 1,000，通过 |
| 第 11 节代表方法比较 | 辅助 | 1,102 | 1,000，通过 |
| 第 12 节领域训练、RLVR 与公开系统案例 | 核心 | 1,878 | 1,800，通过 |
| 第 13—16 节选择、数据、风险与 pipeline 应用链 | 核心 | 2,236 | 1,800，通过 |
| 第 17 节后训练分析 demo | 辅助 | 1,060 | 1,000，通过 |
| 第 18 节常见误判与诊断 | 辅助 | 1,326 | 1,000，通过 |

第 19 节小结和第 20 节资料列表不是独立知识讲解单元。
- **example_status：** 手算保留并复核了 SFT 两目标 token loss、Bradley–Terry RM、PPO ratio/clipping 正负 advantage 与 GAE 递推、DPO chosen/rejected 相对 reference margin、SimPO 平均 logp/margin 与梯度、GRPO 组内均值/标准差和二元正确率下无区分组概率、GSPO 几何平均 ratio/sequence clipping/外层 reduction、DAPO 非对称 clip 与 10/90-token global token mean 权重、DCPO q=0.5/0.1 动态边界和 RUR 分母、DeepSeek-R1 样本数 75%/25% 但不等于 token exposure、OPD 两 token KL、风险评估分母。新增例子已逐步解释输入、单位和局限：GRPO 跨 rank 分片前后组统计；GSPO 两 token ratio 1.2 下完整轨迹 ratio 1.44、1.2^10≈6.19 与长度归一化 ratio 1.2 的差别，以及 response 等权 0.5 对 token 加权 0.74；DAPO 100 条 rollout 中 30 条预算删失；DCPO 生成总量分母下 RUR 70% 与过滤后分母下 80% 的不可比，并说明非零 advantage 可被 clipped surrogate 平坦分支截住。
- **code_status：** 第 17 节 Python 分析 demo 的可执行语句、输入输出、依赖与正文承诺本轮均未改。沿用 2026-11-02、Python 3.12.9、标准库的既有主路径执行验证：SFT/preference/group/eval 汇总、math_001 advantage 与 ratio、拒答召回 1.0、过度拒答 0.3333 和诊断字典输出与正文一致；异常输入与边界分支静态审阅、未逐项执行。本轮未重跑代码，因代码与验证条件未变而沿用既有验证状态。
- **formula_status：** 静态检查无 operatorname 命令、尖括号下标、带下划线的 text 代码变量或未配对行内美元。手算复核包括 GRPO reward [1,0] 的总体标准差 0.5、advantage ±1；GSPO 外层两种 reduction 0.5 与 0.74、1.2^10≈6.19；DAPO 30/100=30% 删失率；DCPO 70/100=70%、48/60=80%。原有 PPO GAE、DPO/SimPO、DAPO token 权重和 DCPO clipping 数例保持一致。
- **visual_status：** 保留并使用方法层级表、更新方法对照表、方案证据比较表、数据要求表、评估风险指标公式和 pipeline 流程。新增正文数例解释组统计、长度归约和指标分母；现有表达足够说明关系，本轮未新增 Mermaid。
- **link_status：** 章节共 50 个 Markdown 链接（45 外链、5 本地目标）；5 个本地文件目标均存在且无 fragment。简介的 20 个本地链接目标均存在。DHPO/GAPO 正文入口固定为 arXiv v1；既有论文、产品和框架外链本轮未做在线全量可达性检查。
- **sync_status：** 第五册简介第 11 章摘要已覆盖 PPO/偏好、现代 RLVR 方法、DHPO/GAPO 候选、verifier/异步 rollout、安全与回归。练习 43 系新增 GSPO reduction 与跨 rank group 统计计算；INTERVIEW_BANK 20D 增加 DHPO/GAPO 的早期预印本定位；KNOWLEDGE_GRAPH 增加从反馈类型、算法层、ratio/clipping 到 loss reduction、漂移审计和准确率/安全回归的依赖链。PAPERS.md 已有 DHPO/GAPO v1 与本章主要论文入口；GLOSSARY_EN_ZH.md 已有 PPO/DPO/SimPO/GRPO、各 RLVR 变体、SAS/OTM/RUR 术语入口，本轮无需重复插入。结构未变，目录无需更新。
- **open_risks：** GLM-5.3 案例使用官方页面搜索索引返回的披露文本，原始页面直开未返回正文行；发布方吞吐和 log-prob 差缺少完整 workload 条件，不能外推。DHPO/GAPO 仍为早期预印本候选。45 个外链未做本轮在线可达性复查。demo 不是真实训练器，异常分支未逐项运行。未生成或更新 PDF。

## 第 12 章：多模态训练——精修记录

- **scope：** 完成第五册第 12 章章级扫描、正文精修、手算复核与配套同步，保留原 18 节结构。主线由图文对比目标进入 VLM 桥接与阶段化训练，再讲多模态 SFT、caption/VQA、OCR/文档、语音、视频、diffusion/Flow Matching，最后汇入数据、对齐、算力和评估。第十五册承担多模态架构与任务机制的完整专题，第七册承担一般评估方法；第五册保留面向训练决策所需的合同、预算与分任务回归。
- **chapter_scan：** 2026-11-02；时间窗 2022—2026，问题覆盖图文目标的归约和多正例标签、VLM connector/训练课程、assistant-only SFT 监督合同、语音生成路线、视频时间证据及连续生成目标。定向阅读 CLIP、SigLIP/SigLIP 2、Qwen3-VL Technical Report v2（摘要、引言、§3.1、视频时间戳和 §3.2 数据段）、Qwen3-Omni Technical Report（§2.2、§2.4—2.5、§3、§4.1—4.3）、Flow Matching 原论文及 Pyramidal Flow Matching 原论文，并复用 `research/multimodal-systems-and-evaluation-2026.md`、`research/speech-recognition-generation-and-realtime-2026.md`、`research/diffusion-generation-guidance-and-sampling-2026.md`、`research/flow-matching-and-video-generation-2026.md` 和 `research/llm-training-and-post-training-systems-2026.md` 中的来源记录。候选处置：① CLIP/SigLIP 的 pair loss reduction、双向目标和多正例聚合纳入第 2 节，强调标签协议不会由 loss 自动修复；② Qwen3-VL S0—S3 用作单一报告的训练课程案例，阶段、token 预算和上下文长度均标为作者配置，不升格为通用 recipe；③ Qwen3-Omni 的 Talker 多码本、MTP、Code2Wav 和后训练顺序纳入第 7 节语音生成小节，完整 Thinker-Talker 架构、服务和实时验收继续由专题承接；④ Flow Matching 与 Pyramidal Flow Matching 纳入第 9 节，讲清路径目标、向量场和 ODE 求解的区别，论文结果限定于各自配置；⑤ SFT 的 token/sample 归约、任务 loss 混合和媒体引用截断作为训练合同问题补入第 4 节。没有新增章节或迁移独立主讲。
- **architecture/boundary：** 18 节顺序不变。第 1 节及章首承担任务地图；第 2—4 节形成表示、桥接到对话监督的链；第 5—9 节按图像局部证据、文档、音频、视频与连续信号分开讲目标；第 10—13 节把样本合同、跨模态对齐、资源和评测接成验收闭环；第 14—15 节连接更新策略和故障定位。数据采集治理接第九册，媒体/对话 token 协议接第三册第 3 章，训练预算接本册第 13 章，多模态任务细节接第十五册，评估设计深讲接第七册。
- **length_status：** 2026-11-02 静态计数，仅计连贯解释性汉字，排除标题、代码与数学围栏、表格、行内代码、公式、URL 目标和代码状态标记，保留链接文字。核心单元下限 1,800，辅助单元下限 1,000，所有独立讲解单元通过。

| 单元 | 类型 | 汉字数 | 下限/状态 |
| --- | --- | ---: | --- |
| 章首、为什么多模态不同、第 1 节任务地图（连续导读链） | 辅助 | 1,159 | 1,000，通过 |
| 第 2 节 CLIP/SigLIP 图文对齐 | 核心 | 2,024 | 1,800，通过 |
| 第 3 节 VLM 架构、桥接与训练课程 | 核心 | 2,060 | 1,800，通过 |
| 第 4 节多模态 instruction tuning | 核心 | 1,921 | 1,800，通过 |
| 第 5 节 caption 与 VQA | 辅助 | 1,136 | 1,000，通过 |
| 第 6 节 OCR 与文档理解 | 核心 | 1,926 | 1,800，通过 |
| 第 7 节语音训练与生成 | 核心 | 2,044 | 1,800，通过 |
| 第 8 节视频训练 | 核心 | 1,998 | 1,800，通过 |
| 第 9 节 diffusion 与 Flow Matching | 核心 | 1,926 | 1,800，通过 |
| 第 10 节多模态数据组织 | 核心 | 2,003 | 1,800，通过 |
| 第 11 节多模态对齐 | 核心 | 1,932 | 1,800，通过 |
| 第 12 节显存和算力 | 辅助 | 1,150 | 1,000，通过 |
| 第 13 节多模态安全和评估 | 辅助 | 1,161 | 1,000，通过 |
| 第 14—15 节策略选择到误判诊断（连续链） | 辅助 | 1,714 | 1,000，通过 |

第 16 节练习、第 17 节小结和第 18 节参考资料不作为独立讲解单元。
- **example_status：** 手算复核包括：SigLIP `B=2` 时 `1/B` 与 `1/B²` 两种 pair reduction；VLM 的 `224×224/14×14` patch、Q-Former 32 query 和 `T+N_v` attention cells；SFT assistant mask、4-token/1-token 样本平均与 OCR/VQA 任务加权；caption/VQA 总准确率 `0.825`、宏平均 `0.60` 和计数切片 `11/20`；OCR `18S+4D+3I` 得 `5% CER`、字段 exact match `24/30`；2 秒、12.5 frame/s、4 码本得到 25 帧/100 codes；视频 8 秒、2 fps 采样及 `[5,6]` 对 `[5.5,6.5]` 的 `tIoU=1/3`；Flow Matching 线性路径 `x_noise=2`、`x_data=-1`、`t=0.4` 得 `x_t=0.8`、目标速度 `-3`；数据过滤 10,000→8,500、混合权重 `.90/.09/.10` 归一化和视频 patch 曝光；grounding 框 IoU `0.5` 与反事实 log-prob 差 `0.51`；全注意力 toy 序列长度 4,040、score-cell 约 996 倍；unsupported rate `9/150=6%`。票据分辨率策略比较中的数值明确为假设教学观测，不冒充模型实验。
- **code_status：** 第 12.1 节 Python 标准库 demo 的可执行语句、输入输出和依赖本轮均未改。仓库中没有找到可追溯的该 demo 执行记录，因此把正文原“已验证（2026-10-26）/五类异常输入检查”改为“静态审阅，未运行（2026-11-02）”，并删除无据的运行声称；本轮未运行代码或测试。
- **formula_status：** 手工核对了新增损失归约、patch/attention 预算、CER、tIoU、Flow Matching 目标、IoU 和比率。将 SFT 条件上下文的尖括号下标 `x_{<t}` 改写为明确范围 `x_{1:t-1}`。首次章级静态扫描后，册级公式门禁又发现 seq2seq 式残留 `y_{<u}`；已改为 `y_{1:u-1}`，并说明 `u=1` 时输出前缀为空。全册复扫后，108 个围栏标记配对；`operatorname`、尖括号下标、`\text` 中代码下划线命中均为 0，行内美元符号配对检查通过。未使用 GitHub 渲染器做视觉公式验收。
- **visual_status：** 保留任务类型、方法取舍、Qwen3-VL 阶段、文档监督和指标等对照表；简短 encoder→connector→LLM 数据流与正文手算预算可读。本轮未添加 Mermaid，避免把清楚的线性结构重复画图。
- **link_status：** 第 12 章共有 54 个 Markdown 链接，均为外链；没有新增本地文件链接。外链在线可达性未做全量复查。公式/围栏静态检查及 `git diff --check` 通过。
- **sync_status：** 第五册简介第 12 章摘要已纳入语音生成、阶段化训练、资源与证据评估；`EXERCISES.md` 第 12 章增加 Qwen3-VL、离散 speech codec 和 Flow Matching 三题；`INTERVIEW_BANK.md` 补 Qwen3-Omni Talker 训练与 Qwen3-VL S0—S3 问答；`PAPERS.md` 补入 Qwen3-VL v2 版本入口；`KNOWLEDGE_GRAPH.md` 增加 Flow Matching 路径/向量场/ODE 和 Pyramidal Flow Matching 关系。`GLOSSARY_EN_ZH.md` 已有 Flow Matching、Rectified Flow、Talker MTP、Residual Codebook 和 Code2Wav 词条，不重复添加。目录结构未变，目录文件不需同步。
- **open_risks：** 论文数字与配方均是作者来源陈述，未独立复现；Chapter 12 demo 保持未运行状态；54 个正文外链未做在线全量可达性复核。未运行模型、训练实验或 benchmark，未生成或更新 PDF。第五册后续章节、连续复读和整册门禁尚未完成。

## 第 13 章：训练成本与资源规划——精修记录

- **scope：** 完成章级扫描、正文精修、手算复核和配套同步，保留 15 个主节；在第 6 节的生命周期规模论证中增加一个固定训练预算与请求量的连续手算。讲解链从总成本账本进入 `6ND`、序列/MoE FLOPs、主循环与墙钟吞吐、GPU/MFU，再延伸到 token 配额、推理生命周期、checkpoint 存储、失败恢复、规划流程和诊断。未新增独立章节或改变第五册与第十四章的分工。
- **chapter_scan：** 2026-11-02；时间窗 2001—2024，重点核对固定训练预算分配、计算/注意力/MoE FLOPs、训练吞吐和可用率、MFU/HFU、推理请求生命周期成本、能耗边界与 checkpoint 故障成本。复用并核对 `research/training-cost-and-resource-planning-2026.md` 中的 PaLM、MegaScale、Llama 2、OLMo、Kaplan/Chinchilla 与 inference-aware scaling 来源事实和证据边界；主要来源阅读位置为 PaLM §4/Appendix B、MegaScale §6.1/Table 2/§§6.2—6.3、Llama 2 §2.2.1、OLMo Appendix B、Chinchilla 固定预算比较及 inference-aware scaling §§2—6/Appendix B—C。候选处置：① 将序列二次项、MoE 激活参数与总参数分开，纳入第 2 节的 2K/4K 长度比较和 top-2 MoE 手算；② 将 `R_active`、`R_wall` 和可用率 `a` 作为互斥测量口径，修正训练时长、GPU 倒推和 MFU 公式，避免重复计算暂停；③ 将 MegaScale 表格中的 `1.34×` 明确标成 MFU 比值，并单列按 tokens/s 算出的约 `1.35×` 吞吐比；④ 为 Chinchilla 与 inference-aware 规划补固定预算/请求量手算，限定其质量、延迟、硬件与数据假设；⑤ 将 checkpoint 容量与保存时长连接到 failure/retry 成本，补充带宽下界、保留策略与保存间隔手算；⑥ 修正完整重试期望成本公式，使失败前平均消耗 `H_fail` 显式出现。没有新增论文主线；具体来源事实只保存在 `research/`。
- **architecture/boundary：** 第 2 节为核心算力模型，第 3—5 节按“吞吐—规模—利用率”构成连续核心链，第 6 节为 scaling 与生命周期成本核心，第 7—8 节按“保存状态—故障回滚”作为连续 checkpoint/recovery 核心链，第 9—12 节按规划、例题、错误和诊断构成辅助闭环。能源碳排保留为成本边界案例；云价格不写死。未将平台架构、checkpoint API 或多模态成本迁入本章。
- **length_status：** 2026-11-02 静态计数；只计连贯解释性汉字，排除标题、代码/数学围栏、表格、行内代码、行内公式、URL 目标和代码状态标记，保留链接文字。核心下限 1,800，辅助下限 1,000。§§3—5 与 §§7—8 按相邻论证链合计，具体分组理由见上。

| 单元 | 类型 | 汉字数 | 下限/状态 |
| --- | --- | ---: | --- |
| 章首、Why、§1 总成本账本（连续导读） | 辅助 | 1,664 | 1,000，通过 |
| §2 FLOPs、序列项与 MoE 估算 | 核心 | 1,963 | 1,800，通过 |
| §§3—5 主循环吞吐、资源倒推与 MFU/HFU（连续链） | 核心 | 2,779 | 1,800，通过 |
| §6 训练数据、compute-optimal 与推理生命周期 | 核心 | 1,866 | 1,800，通过 |
| §§7—8 checkpoint 存储与失败恢复间隔（连续链） | 核心 | 2,257 | 1,800，通过 |
| §§9—12 规划流程、示例、误区与诊断（连续链） | 辅助 | 2,152 | 1,000，通过 |

第 13 节练习、第 14 节小结和第 15 节资料不作为独立讲解单元。
- **example_status：** 手算复核包括：相同 `BS=16,384` 下 2K/4K 序列的 full-square attention 复杂度代理为 33,554,432/67,108,864；1,024 与 3,072 两条序列按各自长度累计 10,485,760 个 score-matrix proxy cells，按平均长度估算少 20%；64 个 5B 专家加 10B shared/0.1B router 的 toy MoE 总/激活参数为 330.1B/20.1B，`D=1T` 时 `6ND` 分别为 `1.9806e24/1.206e23` FLOPs；7B、1T token 得 `4.2e22` FLOPs，示例有效算力与 `a=0.8` 得约 9.7 个主循环日/12.1 个墙钟日；`2T/200k` 得约 115.7 个主循环日，`a=0.8` 后约 144.6 个墙钟日；7B×1T 与 3.5B×2T 的简化训练 FLOPs 同为 `4.2e22`，按 1B 次、70 输入/215 输出 token 请求作 `2N` 参数项估算，推理 FLOPs 为 `3.99e21/1.995e21`；7B AdamW mixed-precision toy 恢复点为 112 GB/104.31 GiB，2 份恢复点加 10 份推理权重约 339.0 GiB，20 GB/s 下理想写入下界为 5.6 秒；14 天每 4 小时写一次约 84 个恢复点；128 GPU、每周 2 次故障、3 分钟同步保存下，4 小时与 1 小时间隔的一阶保存+回滚成本约为 780.8/1,203.2 GPU-hours，平方根估算候选约 2.9 小时。全部明确为教学假设或公式复算，不冒充集群测量。
- **code_status：** §10.1 唯一 Python 标准库 demo 的可执行语句未改；找不到旧“已验证（2026-10-26）/5 类非法输入”声称的可追溯执行记录，正文状态已改为“静态审阅，未运行（2026-11-02）”，并将预期输出说明为手工推算。未运行代码或异常输入检查。静态检查发现 `recovery_assumption_recorded` 仅检查期望失败次数非负和间隔为正，不验证 `latest_resume_count` 是否大于零，也不证明 checkpoint 可读/可恢复；正文已明确其范围，代码保留未运行状态。
- **formula_status：** 统一 `T_active=D/R_active`、`R_wall=R_active×a` 与 `T_wall=D/R_wall` 的边界；GPU 倒推用 `a=1-rho`，若吞吐已含暂停则不再调整；MFU 使用同一主循环时间窗口。checkpoint 例子统一 GB/GiB 与带宽单位；失败期望成本改为 `H_success+q/(1-q)×H_fail`，仅在 `H_fail=H_success` 时简化。静态扫描 31 个 display-math 块、80 对行内美元公式；`operatorname`、尖括号下标、`	ext` 中代码下划线命中均为 0；无 Mermaid。
- **visual_status：** 保留章节原有成本组成、能源边界、研究比较和资源预算表；手算链可直接复核。新增长度、MoE 和 checkpoint 预算用算式解释，不需另画流程图；本轮无 Mermaid 页面渲染项。
- **link_status：** 正文 18 个 Markdown 链接均为外链，本地链接数为 0；本轮未全量检查外链在线可达性。82 个 fence marker 配对；公式/链接静态检查及 `git diff --check` 通过。
- **sync_status：** `EXERCISES.md` 第五册第 13 章第 2/5 题加入 active/wall-clock 吞吐和数值 checkpoint 间隔练习；`INTERVIEW_BANK.md` 第 56H 补充 `R_active`/`R_wall`/`a` 的口径；`GLOSSARY_EN_ZH.md` 增加主循环吞吐、墙钟吞吐词条并限定 MFU 窗口；`KNOWLEDGE_GRAPH.md` 补入吞吐—可用率—time-to-train 依赖链。第五册简介摘要已覆盖成本主线，无需改写；`PAPERS.md` 已有 scaling、PaLM/MegaScale 等来源入口，不重复增加。
- **open_risks：** 论文与生产训练数字均是各作者的特定配置报告，未独立复现；demo 未运行，预期输出只作手工核算；18 个外链未做本轮在线可达性复查。未运行测试、benchmark 或模型，未更新 PDF。第五册第 14—18 章、连续复读和整册门禁仍待完成。

## 第十四章：训练系统综合设计与故障诊断——精修记录

- **scope：** 完成章级扫描、正文精修、数值/公式复核和配套同步，保留 19 节结构及主线位置。将章节讲解边界明确为：前半整合训练方案与预算；§§4—6 从训练异常推进到能力回归和评测污染；§§7—14 是带回章节入口的系统配置复核卡，不替代第 2—13 章的机制主讲。未新增独立章节或改变第五册骨架。
- **chapter_scan：** 2026-11-02；针对训练系统综合设计与事故排查，重点核对 2023—2026 的 loss spike 机制、长程数值精度风险、分布式 checkpoint/恢复语义、主循环与墙钟吞吐口径、污染检测分母。资料类型为原始论文/预印本及 PyTorch 官方固定版本文档，并复用 Molybog、Bai、PaCoST、PALOMA、DCP 2.15 等已记录来源。新来源阅读范围：Chen et al., [The Role of Quantization in Loss Spikes](https://arxiv.org/abs/2610.38277) §§1、2.3、3.1—3.4、结论与 Appendix C.1—C.3；Han et al., [TierCheck](https://arxiv.org/abs/2605.17821) §3、§5.1—5.2；[PyTorch Distributed Checkpoint 2.15](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html) 的 save/load、reshard 与版本兼容警告。候选及去向：① 将输出头均值漂移/有限精度解释纳入 §4.5，与 Molybog 的 update 方向机制、Bai 的二阶矩机制并列，配 BF16 数值例；保留其小模型、高学习率、长程过训练边界，不写成通用根因或修复方案；② 将 TierCheck 纳入 §2.7 作为分层 checkpoint 的研究案例，保留测试拓扑和故障类型边界，不冒充 PyTorch/DCP 默认能力；③ 将 DCP 引用从 2.14 更新到 2.15，并补保存状态不保证跨版本向后兼容的门槛；④ 将 §2.6、§12.2 的 `R_active`、`R_wall`、availability `a` 与第 13 章对齐，避免把暂停重复计入；⑤ 污染命中率改以实际扫描覆盖项为分母，添加覆盖不足时保留 unknown 的手算例；⑥ 为损失/成本/偏好复核卡补齐对应前章链接，阻止短摘要被误读成完整机制课。
- **architecture/boundary：** 本章仍为 1—14 章主线的收束，不取代前 13 章机制解释；§§2—3 和 §15 合成“设计—估算—方案账本”核心链；§§4—6 合成“首个异常—能力结果—评测可信度”诊断核心链；§§7—14 作为带章节入口的配置检查卡；专题算法和数据/架构/并行/精度/评估深讲继续路由到相应章节。第五册与其他书册分工未变。
- **length_status：** 2026-11-02 静态计数；只计连贯解释性汉字，排除标题、代码/数学围栏、表格、行内代码/公式、URL 目标和状态标记，链接标签计入。核心下限 1,800，辅助下限 1,000；不以整篇文件总字数代替逐单元计数。

| 单元 | 类型 | 汉字数 | 下限/状态 |
| --- | --- | ---: | --- |
| 章首、§§0—1 目标—分母—证据到账本（连续导读） | 辅助 | 1,256 | 1,000，通过 |
| §§2—3 与 §15 训练设计—公式估算—可审计账本（核心链） | 核心 | 3,370 | 1,800，通过 |
| §§4—6 首次异常—能力回归—污染证据（核心诊断链） | 核心 | 2,624 | 1,800，通过 |

§§7—14 为带前章链接的系统复核卡，不承担独立机制讲解；§16 常见误区、§17 练习、§18 小结和 §19 资料不作为独立知识讲解单元。
- **example_status：** 方案账本手算为 `2×8×16=256` 个样本、`256×4,096=1,048,576` 个有效 token/update、`6×7×10^9×10^12=4.2×10^22` FLOPs、`7×10^9×16=112 GB≈104.31 GiB` 全量 AdamW 状态；`1T/1,219,048≈9.49` 主循环日，按 `a=0.92` 得约 10.32 墙钟日。有限精度例中 `(0,1)` 与平移后的 `(-1025,-1024)` 在精确 softmax 下相同；BF16 在约 1024 的值间距为 8，二者可舍入成相同 logits，使 `(0.269,0.731)` 变为 `(0.5,0.5)`，正确标签交叉熵从约 0.313 变为 0.693。污染例中 1,000 个评测项有 900 个可扫描、36 个候选、10 个确认，覆盖率 90%、扫描项候选率 4%、扫描项确认率约 1.11%，另 100 项保留 unknown；`10/1,000` 只代表已确认命中在全量题目的观测占比，不代表总体污染率或分数膨胀。
- **code_status：** §15.1 Python 标准库账本 demo 的可执行语句本轮未改。正文原“已验证（2026-10-26）”未找到可追溯的执行记录，故改标为“静态审阅，未运行（2026-11-02，Python 标准库）”；输出按手算列为预期结果。本轮未运行章节代码或测试。静态审阅确认代码只检查必填字段、部分状态词/数值范围并按输入计算量级；`recovery_probe="passed"`、污染/许可状态、吞吐等都是调用者提供的字符串/数字，脚本不会检查数据集、评测隔离、真实吞吐或 checkpoint 是否可恢复；正文已补足此范围边界。
- **formula_status：** 将吞吐时长统一为 `T_wall=D/R_wall=D/(R_active a)`；若吞吐本身含暂停则直接用 `R_wall`。集群扩展式的 active 吞吐与 availability 分开。将语言模型条件上下文的 `x_{<t}` 改为 `x_{1:t-1}`，将 `\operatorname{clip}` 改为 `\mathrm{clip}`；静态扫描 25 个 display-math 块，56 个围栏标记配对，禁用的 `\operatorname`、尖括号下标和 `\text` 中代码下划线命中为 0。行内公式采用代码样式表示，本章没有美元行内公式；无新 Mermaid。
- **visual_status：** 保留数据流代码围栏、诊断表、方案比较表和账本；已有结构可支撑对照和排错，不新增装饰图。TierCheck 的状态关系用正文解释及进度/知识图谱记录，没有画成未经目标平台渲染验证的 Mermaid。
- **link_status：** 本章 50 个 Markdown 链接中，23 个为本地链接且目标均存在、无本地 fragment；27 个外链未做本轮全量在线可达性检查。`git diff --check` 通过。
- **sync_status：** 第五册简介第 14 章摘要仍覆盖训练设计与诊断承诺，目录结构未变；`EXERCISES.md` 第五册第 14 章题 3 纳入三类 spike 假说与适用边界；`INTERVIEW_BANK.md` 更新训练区 Q2 与 Q34；`PAPERS.md` 补入 TierCheck 和输出头量化研究；`GLOSSARY_EN_ZH.md` 新增 Output Logit Mean Drift、Logit Quantization Error；`KNOWLEDGE_GRAPH.md` 增加稳定性—恢复证据链。新论文与系统事实、版本、阅读位置和边界分别写入 `research/training-stability-and-debug-2026.md`、`research/checkpoint-evaluation-and-experiment-management-2026.md`。
- **open_risks：** 两项新研究的结果均为作者报告，未在本项目独立复现；TierCheck 受作者 A800/网络/存储配置约束，量化论文对自然大模型生产训练的迁移尚待验证。账本 demo 未执行；27 个外链未做全量在线复查。未运行训练、模型、benchmark、checkpoint 恢复、程序测试或 PDF 构建/更新。下一步处理第五册第 15 章《领域专家 SFT 与 RL》，第 16—18 章、第五册连续复读和最终门禁仍待完成。

## 第十五章：领域专家 SFT 与 RL——精修记录

- **scope：** 完成第十五章定向扫描、正文重组、算例复核和配套同步；不改变章节标题、第五册目录或与第十六至十八章的边界。把原有短节整理成三条连续教学链：§§15.0—15.9 由领域缺口推到训练信号、金融案例、工具权限和评估；§§15.10—15.18 由样本契约推到目标函数、verifier、采样分布和回归；§§15.19—15.28 由专家分歧推到硬门禁、数据更新、实验、成本和发布审计。保留编号作为链内小节，并在章首说明各链的阅读关系。
- **chapter_scan：** 2026-11-02；围绕领域后训练如何选择 CPT/SFT/偏好/RLVR、专家和程序监督如何对应到任务成功、训练与评估分布如何对齐、怎样避免把 verifier/reward 当成真实权限或安全证明，核对 2024—2026 年金融、纵向 EHR、ICD 编码与数学奖励研究。候选及去向：① 将 FINDAP 纳入§15.0.1，展示 FinCap/FinRec 对能力、CPT+IT 与结果/过程偏好信号的组织，同时保留单一 Llama 3 8B 配方和未验证跨家族迁移的边界；② 将生成式 ICD 编码的 arXiv v1 纳入同表，展示 SFT schema、样本级 F1-GRPO 和仅用于训练的漏码提示，明确 MIMIC/两个 Qwen 底模及稀有码召回限制；③ 复核 FinGPT v2 §3.3.1 与 §5.2.1 的 3.67M/8.3M 参数配置差异，避免单一 LoRA 参数数字被泛化；④ 复核 TIMER 的 +6.50% 表列 win% advantage，删除将其写成“百分点”的误读；⑤ 不引入没有原始资料核验的其他候选。来源、版本、阅读位置和边界见 `research/domain-specific-post-training-2026.md`。
- **architecture/boundary：** 第十五章继续承担领域任务拆解、训练信号选择、样本合同、专家和 verifier 组合及独立发布验收；第十六章主讲 RLVR/verifier 任务合同，第十七章主讲 OPD，第十八章作公开模型案例。§15.7 的生产授权状态与§15.20 的训练轨迹硬门禁明确分开；§15.3 的数据内容层次与§15.13 的错误位置分类明确为两套轴。章内评估维度表只保留在§15.8，§15.10.5 改为引导读者从分项指标走向工作流验收；§15.10.1 解释契约用途，§15.25 保留具体字段示例。第五册骨架不变。
- **length_status：** 2026-11-02 静态计数；只计连贯中文解释文字，排除标题、代码/数学围栏、表格、行内代码/公式、引用块和 URL 目标，保留链接标签。三个核心教学链均高于 1,800 汉字下限。§15.11 是练习，§15.29—15.30 是小结和资料入口；均不另计教学单元。

| 单元 | 类型 | 汉字数 | 下限/状态 |
| --- | --- | ---: | --- |
| 章首导读与 §§15.0—15.9：领域缺口、训练信号与任务评估 | 核心链 | 4,744 | 1,800，通过 |
| §§15.10—15.18：工作契约、训练目标、verifier 与回归 | 核心链 | 2,473 | 1,800，通过 |
| §§15.19—15.28：分歧、硬门禁、实验与发布审计 | 核心链 | 2,633 | 1,800，通过 |

- **example_status：** 已手算并解释三条主线中的例子。§15.6 使用明确标注的合成财报值：收入 100、COGS 64 得毛利 36、毛利率 36%；可比期间 COGS 69 得 31%，变化 −5 个百分点。§15.12 以普通/高风险桶成功率 0.95/0.50，分别复算 90/10 权重得 0.905、等权得 0.725，说明总分依赖业务权重。§15.18 用 4-token 样本平均损失 0.6 和 1-token 样本损失 1.0，得到全局 token 均值 0.68 与样本均值 0.8。§15.28 对合成发布记录复算毛利率 36%、成本 `(105+126+21+63)/42=7.5`，再按 0.90/0.05/10 阈值逐项判断；输出为手算预期而非程序运行结果。
- **code_status：** §15.28 Python 标准库 demo 的可执行代码未改；找不到正文原“已验证（2026-10-26）/五类异常输入”的可追溯执行记录，故改为“静态审阅，未运行（2026-11-02）”。本轮没有执行代码、异常输入或测试。静态检查确认程序仅处理调用者提供的字段并做部分类型/范围检查、简单毛利率和单位成本计算；revision 仅比较字符串，未验证 claim 的实际 source span；硬检查、允许动作、质量/风险/成本、协议和回滚状态均由调用者提供；不连接真实证据、policy、授权或恢复系统，阈值参数本身亦未完整校验。正文已限定其可证明范围。
- **formula_status：** 新增财报毛利率变化、任务桶加权指标、token-level 与 sample-level loss reduction 的可复算数例；核对§15.27 release 门槛和§15.28成本/毛利公式。将条件上下文 `y_{<t}` 改成 `y_{1:t-1}`。静态扫描 32 个围栏标记配为 16 对，包含 7 个数学围栏；`operatorname`、尖括号下标、`	ext` 中代码下划线命中为 0；行内美元公式为 0。
- **visual_status：** 保留研究路线比较、评估维度表、领域样本字段表、发布条件和状态/动作代码示例；通过三条链的上位标题组织阅读，不新增 Mermaid。
- **link_status：** 章节 28 个 Markdown 链接中 1 个本地目标存在且无本地 fragment；外链未做全量在线可达性检查。简介与目录入口未变。`git diff --check` 通过。
- **sync_status：** 第五册简介摘要和目录骨架无需改动；`PAPERS.md` 增加 FINDAP 与 ICD 编码预印本并标明边界，更新 FinGPT 配置说明；`EXERCISES.md` 增加 FINDAP/ICD 受控比较题；`INTERVIEW_BANK.md` 更新领域模型路线题 20E；`KNOWLEDGE_GRAPH.md` 增加任务缺口—样本契约—训练信号—回归—发布门禁链。`GLOSSARY_EN_ZH.md` 暂无必须新增的独立术语。
- **open_risks：** FINDAP、FinGPT、TIMER 的数值均是来源方特定实验报告，未在本项目复现；ICD 编码研究是 arXiv v1 预印本，尚需跨医院和底模验证。§15.28 demo 未运行。正文外链可达性未全量检查。未运行测试、训练、benchmark、模型或 PDF 构建/更新。

## 第十六章：RLVR 与可验证奖励——精修记录

- **scope：** 完成第十六章章级扫描、正文结构精修、证据核对和配套同步；保留原 §16.0—§16.22 编号、章节标题和书册目录。原有多处独立短节重组为五条连续教学链：①章首导读与 §§16.0—16.7，任务边界、成功契约、verifier 和沙箱；② §§16.8—16.9，GRPO 信号与公开路线；③ §§16.10—16.12，奖励投机、污染和迁移评估；④ §§16.13—16.17，成本、稳定性、代码 demo、训练系统和症状定位；⑤ §§16.18—16.20，RLVR/RLHF 分工、采用条件和结果审计。末链作为跨章辅助决策单元，不另立核心算法主讲。
- **chapter_scan：** 2026-11-02；围绕任务是否有稳定成功定义、verifier 的可靠性/覆盖、奖励与过程监督的边界、代码测试质量、测试漏洞利用、污染和跨环境迁移，以及 GRPO 估计与训练成本，定向核验 2023—2026 年原始论文及正式会议页面。候选及去向：① SAGA/TCGBench 纳入 §16.5，明确 VAcc@50 分母和测试集范围，不写成模型正确率或 RL 收益；② ImpossibleBench 纳入 §16.5/§16.10，记录 GPT-5 在 full-scaffold ONEOFF-SWEBENCH 76%、minimal-scaffold ONEOFF-LIVECODEBENCH 2.9% 的作者报告，并说明不同 harness 不可横比、研究不识别 RLVR 训练因果；③ IPT 纳入 §16.4.7/§16.10，突出 extensional 与同构规则检验，以及单一 SLR-Bench、7B 受控训练边界；④ Spurious Rewards 保留在 §16.8.4/§16.10/§16.12，复用 `research/rlhf-preference-and-policy-optimization-2026.md` 中的证据，不重复建研究条目；⑤ PRM800K 继续使用 `research/test-time-compute-and-evaluation-2026.md` 已有证据；未发现需要新增本册章节或改变第五册骨架的候选。新核验的来源事实、版本、阅读位置和边界写入 `research/verifier-quality-and-reward-hacking-2026.md`。
- **architecture/boundary：** 本章仍承担 RLVR 的任务契约、verifier、GRPO、沙箱和迁移审计主讲；不重复第 11 章的 GRPO 变体和第 15 章的领域任务配方。新增五个上位教学链标题，编号小节保持可定位。SAGA 的测试生成质量、ImpossibleBench 的 Agent 测试漏洞利用和 IPT 的逻辑规则捷径保持为不同证据类型，不合并为一种 RLVR 训练结论。
- **length_status：** 2026-11-02 静态计数；只计连贯中文解释文字，排除标题、代码/数学围栏、表格、引用块、行内代码/公式和 URL 目标，保留链接标签。核心下限 1,800，辅助下限 1,000。

| 单元 | 类型 | 汉字数 | 下限/状态 |
| --- | --- | ---: | --- |
| 章首导读与 §§16.0—16.7：任务契约、verifier 和工具环境 | 核心链 | 6,056 | 1,800，通过 |
| §§16.8—16.9：GRPO 估计、更新与公开路线 | 核心链 | 2,410 | 1,800，通过 |
| §§16.10—16.12：奖励漏洞、污染与迁移评估 | 核心链 | 1,931 | 1,800，通过 |
| §§16.13—16.17：成本、稳定、demo 与运行诊断 | 核心链 | 2,058 | 1,800，通过 |
| §§16.18—16.20：方法选择与能力结论审计 | 辅助链 | 1,185 | 1,000，通过 |

§16.21 小结和 §16.22 资料不另计讲解单元。
- **example_status：** 手算并标出合成/假设边界：§16.4.5 用 10 个有效、20 个无效挑战样例，9 个有效接受、2 个错误误收，得经验 completeness 90%、soundness 90%；正文明确有限样本 soundness 近似无效样本上的 specificity，不是 accepted-result precision。§16.8 组奖励 `[1,0,0]` 得均值 `1/3`、总体标准差 `sqrt(2/9)≈0.4714`、优势 `[1.414,-0.707,-0.707]`，并用 `0.20→0.28` 的概率得到 ratio 1.4、clip 1.2 及正优势项约 1.980/1.697；§16.8.2 用 1-token loss 3 与 3-token losses `[1,1,1]`，序列平均 2、token 平均 1.5；§16.10 用合成训练 verifier 60%→82%、独立正确率 70%→72%、长度 180→430 token，解释 +22 pp 与 +2 pp 不可混称；§16.11 对 500 题审计，12 个精确匹配加 8 个确认近重复为 4%，18 个候选留 unknown；§16.12 用 90% 对 66% 复算迁移差距 −24 pp，并用 2×2 检查拆分 verifier shift 与题目分布 shift；§16.13 用每次成本 2/成功概率 0.25 和成本 4/成功概率 0.6，得每次成功成本 8 与约 6.67 个合成单位。数值均为教学假设；SAGA、IPT、ImpossibleBench 和 PRM800K 的论文数字带来源及实验条件。
- **code_status：** §16.15 Python 标准库 demo 找不到原“已验证（2026-10-28，Python 3.12）”对应的可追溯执行记录，正文改标为“静态审阅，未运行（2026-11-02）”，输出表改为静态推演的预期判定。本轮未运行代码或测试。静态检查确认它拒绝 NaN/重复 key、检查 UTF-8 字节预算、schema、动作白名单和数值答案；正文说明动作白名单不等于真实授权，函数不建模 executor/verifier 的 unknown 故障状态。
- **formula_status：** 补齐 soundness/completeness 的分母和数例、GRPO group standardization 与 ratio clipping 手算、sequence/token reduction 对照、transfer gap 和成本例；将 `\operatorname{clip}` 改为 `\mathrm{clip}`。章节 62 个围栏标记配成 31 对，含 29 个 math fence、1 个 Python demo 和 1 个文本流程围栏；禁用 `\operatorname`、尖括号下标、`\text` 中代码下划线命中均为 0，行内美元公式为 0。
- **visual_status：** 保留任务边界表、reward/error 类型表、verifier 质量/代码修复表、泛化审计表、成本账本和症状定位表；开篇文本流程用于显示 verifier 到策略更新的因果链，无 Mermaid，无需额外图示。
- **link_status：** 章节 19 个 Markdown 链接均为外链，无本地 target/fragment；逐项核对新增 SAGA、ImpossibleBench、IPT 和 Spurious Rewards 来源，未对全部旧外链作在线可达性复查。静态围栏、数学记号及 `git diff --check` 通过。
- **sync_status：** 第五册简介摘要和目录骨架无需修改。更新 `PAPERS.md` 的 SAGA/ImpossibleBench/IPT 固定来源入口，`KNOWLEDGE_GRAPH.md` 加入任务契约—verifier—奖励—GRPO—沙箱—独立泛化链，`GLOSSARY_EN_ZH.md` 新增 verifier soundness/completeness 与 extensional correctness，`INTERVIEW_BANK.md` 20G 加入 ImpossibleBench 结果与脚手架边界；`EXERCISES.md` 43D/43F 已覆盖 advantage/归约和 verifier 红队评估，无需重复添加题目。
- **open_risks：** SAGA、ImpossibleBench 与 IPT 数字均是来源方特定数据、模型、任务和 harness 下的报告，未独立复现；IPT 跨领域迁移未证实。§16.15 demo 未运行；19 个外链没有全量在线可达性检查。未运行测试、训练、benchmark、模型或 PDF 构建/更新。

## 第十七章：On-Policy Distillation——精修记录

- **scope：** 完成第十七章章级扫描、正文精修及配套同步；保留章名、文件入口和 17.0—17.24 编号。原独立 H2 结构整理为四条连续教学链：①章首导读与 §§17.0—17.8，分布错位、状态契约、教师信号、调用成本与缓存；② §§17.9—17.12，sampled estimator、时间信用与更新稳定性；③ §§17.13—17.20.1，安全、评估、成本账本、调用 demo 与研究路线；④ §§17.21—17.22，迁移评估和方法边界。§17.23/17.24 继续作为资料和练习门面。小节编号与链接锚点保留；简介/目录只列章节入口，书册骨架未变。
- **chapter_scan：** 2026-11-02；围绕 OPD 的状态来源、监督目标、rollout 与教师成本、后段信号偏差、verifier 可靠性、任务迁移和答案收束能力，核验原始论文、固定版本预印本及当前框架文档。候选去向：① OPRD v4 纳入 §17.20，说明学生 rollout 上隐藏表示对齐及 OPRD-Bridge，保留数学实验和 code/agent/开放对话未证范围；② OPTD v1 纳入 §17.20，区别层间 log-prob 增量与隐藏状态；③ TF-OPD v2 纳入 §17.20，说明共享底模专家、rollout 二阶矩和无梯度最小二乘，标明主证据并非严格当前学生 on-policy 更新；④ `Solving Without Stopping` v1 加入 §17.21.5，要求分开测解题、答案正确/标记、正确后停止和过度生成；⑤ TRL 当前 DistillationTrainer/GKDTrainer 文档改到 `/en/` 入口并记录动态 API 语义，不断言稳定包版本；⑥ 明确成功/失败 peer 多 rollout 论文与 Multi-Teacher OPD 的 MOPD 缩写冲突。研究事实、阅读位置、版本与边界见 `research/llm-training-and-post-training-systems-2026.md`。未改变第五册结构或新增章节。
- **architecture/boundary：** 第十七章继续主讲学生分布上的教师监督和其优化/运行边界；区分输出分布、隐藏表示、层间预测变化与专家 checkpoint 合并四类目标。TF-OPD 标为邻接路线，避免把专家 rollout 的无梯度合并冒充严格学生 on-policy 更新。与第十八章 Multi-Teacher OPD/K2 披露语境不混用裸 MOPD。在线框架配置只作实现入口，论文算法、文档 API 和模型团队报告分层表述。
- **length_status：** 2026-11-02 静态计数；只计连续中文解释性汉字，排除标题、代码/数学围栏、表格、引用块、行内代码/公式和 URL 目标，保留 Markdown 链接标签。四条核心教学链均超过 1,800 字。

| 单元 | 类型 | 汉字数 | 下限/状态 |
| --- | --- | ---: | --- |
| 章首导读与 §§17.0—17.8：分布错位、教师契约、成本与缓存 | 核心链 | 4,084 | 1,800，通过 |
| §§17.9—17.12：采样估计、时间信用与稳定更新 | 核心链 | 2,115 | 1,800，通过 |
| §§17.13—17.20.1：安全、评估、账本、demo 与研究路线 | 核心链 | 2,968 | 1,800，通过 |
| §§17.21—17.22：迁移能力与停止/收束边界 | 核心链 | 3,000 | 1,800，通过 |

§17.23 资料和 §17.24 练习不另计教学单元。
- **example_status：** §17.7 以合成假设 `c_S=1,n=4,p_T=0.25,c_T=2` 得选择性调用成本 3；若成功率假设为 0.60，则为 5 单位/成功。全状态调用 `p_T=1` 得成本 9；另一个假设成功率 0.75 得 12 单位/成功。正文说明两组成功率是假设且不是同题比较。§17.15 用同一假设成本单位将训练 1,200、评估 80、运行 720 汇总为 2,000，除以 400 次独立成功得 5 单位/成功，并说明不重复累计 §17.7 教师费用。§17.21.5 配对玩具例 20 题（两者皆成功 9、仅 OPD 5、仅 offline KD 2、皆失败 4）得 OPD 70%、offline KD 55%、差 15 个百分点；明确 n=20 只演示分母，不作显著性结论。保留 §17.9 `p_S=(0.5,0.5),p_T=(0.8,0.2)` 的 sampled log-ratio/reverse-KL 手算。
- **code_status：** §17.16 Python 标准库 demo 的可执行语句本轮未改。正文旧“已验证（2026-10-26）”未找到可追溯运行记录，故改为“静态审阅，未运行（2026-11-02）”；本轮未执行代码。静态审阅了状态字段、概率/整数/schema 检查、风险/连续失败/低置信触发、cache payload、有限 JSON 和 SHA-256 标识路径；代码按设计推演可产生相应触发原因及状态变更后的不同 key。其 action schema 不是真实权限系统，hash 不是匿名化，尚未接入数据脱敏、executor、真实 verifier 或 teacher API；正文将这些范围列为系统边界。
- **formula_status：** 把 sampled policy-gradient 式中的 `\operatorname{sg}` 改为规范允许的 `\mathrm{sg}`。静态扫描 64 个 fence marker 配成 32 对，其中 28 个 math fence opener、4 个代码/文本 opener；禁用 `\operatorname`、尖括号下标、`\text` 代码下划线命中均为 0；本章没有行内美元公式。
- **visual_status：** 四条上位教学链连接原有状态/反馈、研究比较、成本、评估与故障表；保留已有数值流程、方法对照和指标定义表。关系可由表格及顺序讲解读清，本轮无须新增 Mermaid。
- **link_status：** 章节 41 个 Markdown 链接全为外链，无本地目标或 fragment；新增的四篇固定版本预印本与 TRL `/en/` 文档于 2026-11-02 复核；其余外链未作全量可达性检查。章节编号锚点不变。`git diff --check` 通过。
- **sync_status：** 第五册简介摘要和目录章级入口仍覆盖本章承诺，未列出节级标题，无需改动。更新 `PAPERS.md` 的方法分类与文档链接、`GLOSSARY_EN_ZH.md` 的 OPRD/OPTD/TF-OPD 及停止能力词条、`EXERCISES.md` 第五册 OPD 题和 `INTERVIEW_BANK.md` 20H；`KNOWLEDGE_GRAPH.md` 增加输出/隐藏表示/层间变化/共享底模合并及答案—正确—停止链。研究事实仅记入 `research/llm-training-and-post-training-systems-2026.md`；修改结论和状态留在本进度文件。
- **open_risks：** OPRD、OPTD、TF-OPD 和停止能力材料均为预印本作者证据，未在本项目独立复现；模型、任务、teacher routing、数据筛选、精度及评测协议限制见对应研究记录和正文。TRL 文档动态变化，部署时需对照固定安装版本源码。§17.16 demo 未运行；本轮不运行测试、训练、benchmark、模型或 PDF 构建/更新，未全量复查全部旧外链。

## 第十八章：DeepSeek-V4 领域专家与统一蒸馏——精修记录

- **scope：** 完成第五册第十八章章级扫描、教学链重组、正文精修和配套同步；保留章节入口和原有主问题，将原独立 H2 整理为四条连续教学链：①任务差异、领域 SFT/GRPO、reasoning effort、GRM 与权限边界；②学生状态协议、multi-teacher reverse KL、full-vocabulary 目标动机和教师冲突；③hidden-state/logit 调度、审计循环、运行成本和失败模式；④代表合并路线与采用条件。面试追问与练习仍为独立 H2 门面；简介/目录只承诺章级主题，目录骨架不变。
- **chapter_scan：** 2026-11-02；围绕 V4 专家培养是否继续使用 RL、困难任务的奖励信号、推理 effort 与 rollout 协议、学生状态上的多教师目标、全词表 KL 的方差/资源取舍、teacher head 调度与报告披露边界，阅读 DeepSeek-V4 arXiv v1 §5.1.1、§5.1.2 和 §§5.2.2–5.2.5。①纳入报告披露：领域专家先 SFT 再按领域 prompts/rewards 做 GRPO，区分这与后续 mixed-RL 阶段由 OPD 替换；reasoning effort 的不同 RL 配置、长度惩罚、上下文窗口及 Think Max prompt；GRM、`|DSML|` XML 工具协议和 interleaved-thought context policy。均标为发布方报告，未公开的 effort 超参数、GRM schema、独立效果比较和教师数归属保留未知。②纳入 >10 teacher reverse-KL 与 full-vocabulary OPD，并解释作者的 sampled-estimator 方差动机；报告没有同配置受控消融，因此不推广为 sampled OPD 普遍不稳定。③纳入参数按需分片、末层 hidden-state 缓存、按 teacher index 装载 prediction head、异步读写和 TileLang exact KL；将“effectively unbounded teacher count / potential trillion-parameter teacher”限定为系统能力，不说成 V4 实际配置。④报告 §5.2.3–5.2.5 的 token-level WAL、百万 token loader 与 DSec 作为相邻训练执行层候选，正文点到为止，并回链第五册第十四章的恢复/资源审计。⑤Quick Instruction 记录在 research 作为报告事实，正文暂不扩展：它偏向 chatbot 辅助任务与 serving prefill/KV 复用，不是本章专家训练/教师合并中心问题。⑥比较路线核验 GKD v3、MiniLLM v6 和 Model Soups PMLR 2022 原始来源；只比较目标对象、student-state 来源和证据条件，不横比不同模型任务的分数。来源事实及证据边界仅写入 `research/llm-training-and-post-training-systems-2026.md`。
- **architecture/boundary：** 第十七章仍讲通用 OPD 目标和学生分布，第十八章承担 V4 的公开案例；第十四章承担系统账本与训练故障排查。V4 是作者公开技术报告案例，不是配方复现。领域 checkpoint 与最终 MoE routed expert、effort 模式与教师 checkpoint 数、full-vocabulary 对前缀采样与对状态分布采样均分别表述。
- **length_status：** 2026-11-02 静态计数；只计连贯解释性中文，排除标题、代码/数学围栏、表格、引用块、行内代码/公式及 URL 目标，保留 Markdown 链接标签。前三条为核心链（下限 1,800），第四条为辅助比较链（下限 1,000）；面试/练习不另计教学单元。

| 单元 | 类型 | 汉字数 | 下限/状态 |
|---|---|---:|---|
| 一、从任务差异到领域专家训练 | 核心链 | 2,092 | 1,800，通过 |
| 二、从学生状态到多教师分布目标 | 核心链 | 2,063 | 1,800，通过 |
| 三、把全词表目标落进可调度的训练系统 | 核心链 | 1,979 | 1,800，通过 |
| 四、比较合并路线并核算采用条件 | 辅助链 | 1,155 | 1,000，通过 |

- **example_status：** 组内奖励 `[0.2,0.8,0.5,0.1]` 手算均值 `0.4`、总体方差 `0.075`、标准差约 `0.273861` 和优势 `[-0.730,1.461,0.365,-1.095]`；三 token `p_S=(0.60,0.30,0.10)`、`p_T=(0.70,0.20,0.10)` 得 full-vocabulary reverse KL 约 `0.02915 nat`，单 sampled log-ratio 标准差约 `0.25046 nat`；两位等权教师 `[0.9,0.1]` 与 `[0.2,0.8]` 得几何平均学生 `[0.6,0.4]`，区别于算术平均 `[0.55,0.45]`。logits 缓冲算例 `8,192×100,000×2` 为 `1,638,400,000 bytes≈1.53 GiB`/教师，十位同时物化约 `15.3 GiB`。成本账本用合成假设算出 full-route `2.80` 点/任务、成功率 `0.70` 时 `4.00` 点/成功；稀疏调用 `2.48` 点/任务、成功率 `0.55` 时约 `4.51` 点/成功。比较链的部署 break-even 假设 `700/(1.8−1.1)=1,000` 请求；所有成本输入均标为教学假设，不作为 V4 测量。
- **code_status：** 两段 Python 标准库代码原标“已验证（2026-10-26）”但没有找到可追溯执行记录；已分别改为“静态审阅，未运行（2026-11-02）”。本轮没有改可执行语句、没有执行章节代码或测试。静态检查了 reward 序列的有限实数约束、总体标准差路径、输入分布长度/归一化、教师支持集和 exact KL/log-ratio 方差计算；输出由手算/公式推演。运行环境版本、实际 stdout 和边界分支执行结果未声称。
- **formula_status：** 将 sampled-KL surrogate 中 `\operatorname{sg}` 改为规范允许的 `\mathrm{sg}`；保留符号定义、reverse-KL 权重方向、几何平均条件和 BF16 byte/GiB 换算。静态扫描 28 个 fence marker 配成 14 对（11 个 math、2 个 Python、1 个 text block）；`\operatorname`、尖括号下标和行内美元命中均为 0。
- **visual_status：** 四条 H2 教学链连接了领域/部署专家分类表、effort 模式表、采样误差轨迹表、teacher-routing 表、方案比较表、成本账本和训练流程文本图；数据流和取舍关系可由表及文字图读取，本章无需新增 Mermaid。
- **link_status：** 章节共 22 个 Markdown 链接，其中 20 个外链、2 个本地链接；第十七章和第十四章目标存在且没有 fragment。报告固定到 arXiv v1；GKD v3、MiniLLM v6、Model Soups PMLR 来源及新增 V4 报告位置已在线阅读；其余历史外链未全部在线复查。公式、围栏和链接静态检查通过，`git diff --check` 通过。
- **sync_status：** 更新第五册简介第十八章摘要和正式模型名；`PAPERS.md` 固定 V4 报告 v1 并补入后训练披露，同时固定 GKD v3/MiniLLM v6；`GLOSSARY_EN_ZH.md` 增加 reasoning effort、Interleaved Thinking 与 `|DSML|` 协议说明；`EXERCISES.md` 增加 effort/context-policy 评估题；`INTERVIEW_BANK.md` 增加对应面试问答；`KNOWLEDGE_GRAPH.md` 增加 V4 专家—学生—教师调度链。联网研究事实和来源版本只入 research，章节决定与修改状态留在本记录。
- **open_risks：** 报告只证明作者公开这些流程，不等于独立因果验证；教师路由/权重、完整 loss 和 GRM schema、effort mode 参数、10+ teacher 的组成、采样与全词表 OPD 受控比较、V4 峰值显存/端到端成本均未披露。GRM、工具协议和 interleaved-thinking 的作者收益主张无独立复现。代码未运行；旧外链未全面在线复核。本轮未运行测试、训练、benchmark、模型、PDF 更新或构建。

## 整册复读与最终门禁：2026-11-02

- **复读范围与路径：** 重新通读第五册简介和目录；沿 1—14 章主线及 15—18 章专题路径，逐章对照章节地图、当前 H1/H2 结构、开篇问题、收束和相邻章交接，并结合各章完整精修记录核对核心链、例子、篇幅和边界。主线依次从训练目标/证据链进入数据、tokenizer/协议、架构、训练循环、优化器、并行、显存/数值、稳定性、checkpoint/评估、后训练、多模态、资源规划和综合诊断；14 章收束后，15—17 章依次展开领域适配、verifier/RLVR 与学生分布蒸馏，18 章以 V4 披露作案例。章节顺序、简介地图、目录阶段和章节间交接相符；没有发现需要改变骨架或新增主讲入口的断层。第 01/14、06/09、07/08、10/13/14、11/15—18 的重叠由已完成的章节边界与主讲路由处理。
- **前置与目录：** 简介的读者前置、能力承诺、1—14 主线/15—18 选读建议和跨册分工与目录一致；18 个章节入口均有对应文件。第 02/03 章通往数据治理与 tokenizer 深讲，第 12 章通往多模态专题，第 11/15—18 章通往推理、Agent 及蒸馏专题，第 06/14 章保留 serving/平台成本交接。
- **静态门禁：** 对简介、目录和 18 个章节文件共 20 个 Markdown 文件做本地链接扫描：156 个本地链接目标全部存在，无本地 fragment；555 个外链未做全量在线可达性复查。所有围栏标记数为偶数；全册静态扫描 `\operatorname` 和尖括号数学下标均为 0。扫描发现第 12 章 seq2seq 公式残留 `y_{<u}`，已改为 `y_{1:u-1}`，并说明首 token 的前缀为空。视觉审视沿用逐章记录的表格、文本数据流和数值轨迹；全册没有 Mermaid 块，因此无 Mermaid 渲染项。`git diff --check` 通过。
- **代码与限制：** 本次未运行章节代码、测试、训练、benchmark、模型或 PDF。代码状态以逐章记录为准，静态未运行的示例没有记为已验证；整册复读不替代这些示例的实际运行验证。`EXERCISES.md` 中既有的 24 个非第五册失效本地目标不属于第五册正文链接，本轮未改动，留到配套资料审阅时逐项处置。
- **册级决定：** 简介/目录、主线衔接、章节定位与本地链接、围栏和公式记号的静态门禁通过；不调整第五册 18 章骨架。外链在线可达性和各章代码状态仍按已记录范围保留，不声称本轮已完成这两类运行/联网复核。

## `work_handoff`

- **已完成：** 第五册契约、简介/目录审视、18 章地图、册级研究扫描与骨架决定；简介含 18 条短摘要；第 01—18 章均已完成章级扫描、正文精修、相关配套同步和逐章静态门禁；新研究事实已写入 `research/`。2026-11-02 完成册级主线复读、本地链接/围栏/公式静态门禁，骨架维持不变。
- **当前决定：** 第五册骨架不变。第 01 章负责全景地图和共享账本，第 14 章负责训练系统综合诊断，第 17 章主讲通用学生分布蒸馏，第 18 章承担 DeepSeek-V4 报告案例；四者避免互相替代主讲。
- **遗留：** 555 个外链未全量在线核验；静态未运行的示例保持如实状态，后续如需验证则按代码改动和适用范围处理；`EXERCISES.md` 的 24 个非第五册失效目标留待配套资料审阅。按用户要求跳过 PDF 更新/构建。
- **下一步入口：** 按 `plan.md` 与 `progress/全套书精修-2026-10.md` 进入 `book-06`，先审书册契约、简介/目录、相邻册边界和册级扫描，再稳定骨架。
