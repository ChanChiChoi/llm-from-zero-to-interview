# 第八册第九章 Model Editing 与 Unlearning 章级扫描

- `date`：2026-10-27。
- `scope`：依据当前 `WRITING_SPEC.md` 通读 `09-model-editing与unlearning.md` 的 25 个二级段，检查章内主线、书册边界、研究证据、公式、数值例、代码和视觉表达；定向核对 2025—2026 年知识编辑过拟合、删除上下文效用及表示残留审计研究。本记录是结构/研究扫描，不代表正文精修完成。
- `chapter_role`：本章主讲参数或显式编辑记忆如何改变局部知识/行为，以及怎样评估数据、知识、行为和能力层面的 unlearning。第八章讲推理时 activation steering；本章讲较持久的参数修改或显式编辑记忆。第十章主讲隐私、记忆和水印风险，第十一章主讲治理流程；第六章负责危险能力/红队测量，第七册负责评测统计基础，第九册数据工程负责数据血缘。应避免把“模型拒答”“某基准分数下降”“训练样本影响消失”和法律删除完成当作同一结论。

## 结构和主题去向

- `structure_read`：章节从“模型不是数据库”起步，区分 RAG、训练、编辑、遗忘与 guardrail，随后解释编辑目标和 Transformer 内部假设，介绍 ROME、MEMIT、MEND、SERAC，转到评估；第二部分定义遗忘目标、方法、forget/retain/attack 评估、基准、公式、三类案例和回滚工程，最后用审计 demo 收束。整体顺序可用，不新增章节。建议重组为四个连续教学单元：概念与任务契约（§1–§4）、编辑方法和编辑评估（§5–§10）、遗忘目标/方法/评估（§11–§16）、综合案例与可回滚流程（§17–§20）；§21 demo 复用前述指标。§15 标题称“三个重要基准”，内容实际覆盖 CounterFact/zsRE、TOFU、WMDP 和 MUSE，需重命名或明确层级。§9、§10 的编辑评估与 §14、§16 的遗忘评估应共享评估矩阵，但保留各自目标差异。
- `cross_chapter_boundary`：与第八章区分临时激活干预；与第十章区分模型行为遗忘、成员/内容泄露测量和完整隐私生命周期治理；与第十一章区分技术证据与治理/法务决策。安全能力 unlearning 只讲受控评估和代理基准的证据边界，不复述第六章完整红队方法。

| 研究候选 | 建议落点 | 证据范围和处理 |
| --- | --- | --- |
| 直接目标答对可能是编辑过拟合 | §5–§6、§10、§17 | ICLR 2025 EVOKE 研究指出编辑后模型可能对目标答案过度增概率，并削弱多跳复杂场景中的新知识泛化；建议在 reliability/locality 外保留关系组合与多跳 ripple tests。结论绑定论文任务和 benchmark。 |
| 方法比较需要包含扎实的 fine-tuning baseline | §2、§7、§10 | ICLR 2026 LocFT-BF 论文将样本逐条训练到收敛与 mini-batch epoch 流程作对照，作者报告后者及局部化位置选择改善编辑结果，并称可扩展到其 100K 编辑/72B 模型实验。应作为论文中的方法比较，不写成所有 fine-tuning 都优于 editing 的结论。 |
| 编辑的局部性涉及多跳邻域 | §10、§15、§17 | NeurIPS 2025 UniEdit 用 25 个领域和知识图谱邻域多跳采样扩展 generality/locality 评估，适合补充“相邻事实 ripple effect”；其样本自然语言化使用专有 LLM，评价仍受数据构造和模型/编辑器覆盖范围约束。 |
| forget set、retain set 之外还需测上下文效用 | §12、§14、§16、§18 | ICML 2026 Context-Aware Unlearning 对六种方法测试“被遗忘内容重新由提示提供时模型能否使用”，报告现有方法会损害这类 contextual utility。可增加独立维度，区别“模型自行回忆”和“用户在合法上下文中重新提供”；不推断所有删除场景都应恢复敏感内容。 |
| 输出指标可能漏掉内部残留信息 | §14、§15、§16、§18 | EACL 2026 使用信息分解和表示探针，报告指定设置中遗忘内容仍可从内部表示线性解码，并与已知重构攻击敏感性相关。适合作为输出/成员推断之外的审计线索；可解码不等同于已发生外部泄露，结论绑定作者模型、数据与攻击协议。 |
| 忘记与保留具有非对称泛化要求 | §3、§14、§15 | 2026-07 arXiv 预印本 SUITE 把忘记集的表达覆盖与更广泛的保留边界分开，适合补充近邻集合和边界标注。需保留其预印本状态及数据/方法范围。 |

## 篇幅、公式、例子和代码

- `length_scan`：按 `##` 分段，去掉标题、`~~~`/``` 代码围栏、表格行和独立 display 公式后粗计汉字；仍含部分清单/引用，仅用于结构定位，不是正式门禁：§1–§25 依次约为 597、940、523、531、861、852、465、379、548、972、415、415、780、606、720、438、492、353、369、384、208、511、430、568、323。候选核心单元 §1–§4 合计约 2,591；§5–§10 合计 4,077；§11–§16 合计 3,374；§17–§20 合计 1,598。各组仍须人工检查是否达到解释深度，不能以合计长度自动通过。
- `formula_and_example_scan`：全文有大量 `$$` display 公式和 `\(...\)` 行内表达，精修时统一到 `$...$` 行内及 `math` 围栏，并复核维度、权重分母、空样本、概率分布比较和编辑前后配对。现有案例解释选择过程较具体，但没有把一组数值从输入逐步算到关键评估结果。可复用 §21 的虚构编辑输入手算 `S_target=(3×1+2×1+1×0+4×1)/10=0.9`；retain 加权均值由 `0.883` 变为 `0.851`，所以 `Δ_retain=-0.032`。解释权重和每一步分母，再对照代码结果。此处数字是教学合成值。
- `code_status`：§21 只处理合成记录，不加载模型或修改参数；未运行且未见“已验证/静态审阅，未运行/待验证”状态标记，也没有逐行预期输出。静态手算编辑信号：target success `0.900`、paraphrase `0.667`、locality `0.800`、retain delta `-0.032`、冲突编辑数 `2`，因此前四项达其 toy 阈值、冲突检查不达标。遗忘信号：exact leakage `0.417`、paraphrase `0.750`、multi-turn `0.417`、综合 robust leakage `0.528`、MIA delta `-0.378`、retain delta `-0.026`；MIA/retain toy 阈值通过，但 exact/robust leakage 不通过。该代码能展示分项 hold 决策，不构成 unlearning 实测。静态代码复读发现 `city_fact` 的 `paraphrase_ok` 字典键及 `unlearning_signals` 的 `retain_before` 键重复（当前后值相同）；空编辑集时 `conflict_count=0` 会被当作通过，虽其它指标为 N/A 会整体 hold，仍应将该项设为 N/A。总权重为零会抛 `ValueError`，属于当前输入契约。改可执行语句后按门禁复验。

## 图表和完成门禁

- `visual_map`：§18 有 `~~~text` 删除闭环；尚无 Mermaid。可将“请求范围→数据血缘/副本定位→选择源数据、检索、模型或行为处置→forget/retain/contextual/攻击审计→版本记录与回滚”画为责任边界流程；另以小表区分源数据删除、参数编辑、输出抑制和模型遗忘。若使用 Mermaid，采用标准 `mermaid` 围栏，中文/空格/标点节点标签优先加引号，带文字虚线箭头用 `-.->|标签|`，并核验 GitHub 渲染。
- `completion_gate`：本轮完成章级结构和定向研究扫描。仍需第十至十六章扫描、整册 coverage map、重组后讲解单元篇幅/手算门禁、正文精修、图表渲染和整册阅读路径复读；本章及第八册均未过完成门禁。
- `open_risks`：新增评估维度需要明确适用范围；编辑方法论文结论和未公开训练数据要分别标证据等级；上下文 utility、内部可解码信息和外部泄露不可混写；节标题与基准数量不一致；demo 的边界状态、重复键和逐步解释待精修；公式、链接及 GitHub 图示尚未验收。
- `validation`：完成 25 个二级段复读、定向 primary-source 阅读、解释性正文粗计和 toy demo 关键数值手算；未运行 Python、未做全章外链检查、未渲染公式/Mermaid、未构建 PDF。

## 参考来源

1. Zhang et al., [Uncovering Overfitting in Large Language Model Editing](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c592fc7e6207f82560ed45fece8d6937-Abstract-Conference.html), ICLR 2025。
2. Yang et al., [Fine-tuning Done Right in Model Editing](https://proceedings.iclr.cc/paper_files/paper/2026/hash/74b8ccb5ad21871fd436867df19c35ac-Abstract-Conference.html), ICLR 2026。
3. Chen et al., [UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6cc6c2be42c838827d2f1d12baa05539-Abstract-Datasets_and_Benchmarks_Track.html), NeurIPS 2025 Datasets and Benchmarks Track。
4. Peng et al., [Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models](https://proceedings.mlr.press/v306/peng26a.html), ICML 2026。
5. Goel et al., [Auditing Language Model Unlearning via Information Decomposition](https://aclanthology.org/2026.eacl-long.35/), EACL 2026。
6. Peleg et al., [Forget Narrowly, Retain Broadly: Unlearning as an Asymmetric Generalization Problem](https://arxiv.org/abs/2607.09236), arXiv, 2026-07。

- `status`：第九章全文结构和研究主题扫描完成，正文尚未修改，demo 未运行；下一步扫描第十章，并在 coverage map 中确认 unlearning、隐私记忆和数据血缘的主讲边界。
