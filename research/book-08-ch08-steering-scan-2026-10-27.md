# 第八册第八章 Steering 与 Representation Engineering 章级扫描

- `date`：2026-10-27。
- `scope`：依据当前 `WRITING_SPEC.md` 通读 `08-steering与representation-engineering.md` 的 19 个二级段，检查章节职责、讲解单元、研究证据、公式、手算例、代码和视觉表达；定向核对 2025—2026 年 steering 稳健性、行为权衡和推理时控制研究。本记录是结构/研究扫描，不代表正文精修完成。
- `chapter_role`：本章解释如何从行为目标构造可施加于推理时激活的表示方向，并判断目标收益、正常能力、鲁棒性和安全副作用。第七章主讲如何提出并检验内部机制假设；第八章主讲利用表示方向做短时行为控制；第九章主讲参数编辑和持久 unlearning。Activation patching/ablation 是机制研究中的因果干预，不能和线上持续 steering 混为一谈；steering 也不能替代训练、服务端授权或应用级安全控制。

## 结构和主题去向

- `structure_read`：现有主线从 prompt/decoding/fine-tuning 的控制层次出发，介绍 Representation Engineering、steering vector、CAA 和拒答方向，再讲应用、指标、方法比较、项目治理、风险与案例，最后接资料、练习和 toy demo。职责总体明确。建议按“控制层次与方向构造（§1–§6）—安全行为与评估（§7–§9）—方法边界、部署治理与风险（§10–§13）—综合案例和审计（§14、§18）”梳理单元；§8–§9 的应用/指标、§11–§12 的使用/风险和 §13 的未来方向可能重复，应在全册 coverage map 定稿时确定删并或交叉链接，不新增章节。
- `cross_chapter_boundary`：第七章解释 attribution、SAE、patching/ablation 等机制证据及其因果局限；本章讲基于表示方向的推理时控制和副作用；第九章承接需要改动参数的模型编辑、遗忘与持久性评估。第六章提供独立红队/危险能力评估协议，第十三章提供自动化对抗评估流程，第十五/十六章分别负责运行时访问和安全降级。可在本章说明 steering 的安全评估需求，但不重讲上述章节的完整方案。

| 研究候选 | 建议落点 | 证据范围和处理 |
| --- | --- | --- |
| 方向干预不一定能持续控制生成 | §5、§9、§12 | ICML 2026 的 Endogenous Steering Resistance 使用 SAE latent 做 steering，报告 Llama-3.3-70B 的抵抗/中途恢复较明显，较小 Llama-3 与 Gemma-2 模型较少见；应补逐 token/多轮持久性检查。结论绑定作者模型、latent 和任务，不概括所有模型。 |
| 强度、几何变化和控制回路 | §5、§9、§13 | ICML 2026 Spherical Steering 研究保范数旋转与置信度门控，并报告其多选 benchmark 和开放生成实验；ICLR 2026 Feedback Controller 将作者分析中的常见固定方向方法类比为比例控制并提出 PID steering。可作为固定向量/固定强度的研究进展，不能把论文的稳定性表述写成所有模型的部署保证。 |
| 多维安全权衡与细粒度目标 | §9、§14 | SteeringSafety（arXiv v3，已被 ICML 2026 接收）覆盖九种安全视角、18 个数据集，并报告方法、模型、行为目标之间的表现差异和行为纠缠；ACL 2026 SteerEval 按语言特征、情感和人格以及不同具体化粒度评估控制能力。适合把单一 target score 扩成副作用矩阵；数字只绑定其 2B–8B 级模型和任务集。 |
| 对抗输入会改变方向有效性和最优层位 | §9、§12、§14 | 2026-06 arXiv 预印本在 5 个模型（1.5B–30B）、4 种提取方法、3 类攻击和 6 种 persona 上评估，报告若干设置中的 steerability 下降及最优层位偏移。作为待独立复现的脆弱性线索，不能写成真实流量故障率或已确立的通用定律。 |

上述新主题优先补在现有评价、风险和案例段，不新增章节。Evaluation-awareness steering 暂不扩为本章主线；如全册评估边界需要，可与第六章的评测可信度交叉引用，并把其特定训练设置和提示线索写清。

## 篇幅、公式、例子和代码

- `length_scan`：按 `##` 标题分段，去掉标题、代码围栏内容、表格行和独立 display 公式后粗计汉字（仍含部分清单/引用，不是正式验收数字）：§1 556、§2 98、§3 98、§4 594、§5 705、§6 542、§7 632、§8 454、§9 1,472、§10 344、§11 577、§12 625、§13 569、§14 1,259、§15 447、§16 162、§17 125、§18 135、§19 226。章节本身小节较多，不能把每个 `##` 都算作独立单元；§4–§6 合并后约 1,841 字，可形成一个核心机制单元，但应人工复核连贯解释；§9 单独承担评估机制时低于 1,800 字核心门槛；§14 超过 1,000 字辅助线，但如作为独立核心案例仍需展开。正式修订时依 §3.3 对重组后的每个实际讲解单元重新实计。
- `formula_and_example_scan`：§9 有多处 `$$` display 公式及全章 `\(...\)` 行内数学，后续应按新版规范改为 `$...$` 行内和 `math` 围栏，并逐项复核变量、量纲、空样本与分母边界。§14 是定性场景，没有给出一组配对输入和逐步结果；§18 代码有预期输出但没有正文手算。可复用 demo 输入讲一个短例：正例均值 `[1.1, 0.8, 0.0]`，负例均值 `[0.1, -0.2, 0.8]`，差向量 `[1.0, 1.0, -0.8]`，范数 `sqrt(2.64)≈1.625`，单位方向约 `[0.615, 0.615, -0.492]`；对输入 `[0.4, 0.2, 0.4]` 取 `α=0.4`，激活变为约 `[0.646, 0.446, 0.203]`。再解释这只是 toy hidden-state 投影变化，不等于真实任务上的 factuality 或安全收益。
- `code_status`：§18 的 Python 代码未运行且没有“已验证/静态审阅，未运行/待验证”状态标记。静态复算样例均值、方向及强度为 0.4 时的聚合目标增益 `0.078`、安全增益 `0.048`、side loss `0.018`、over-refusal `0.013` 与目标函数约 `0.096`，和预期输出相符。需要在正文说明 `target_gain`、`safe_gain`、`side_loss`、`over_refusal` 是预置 toy 数，不是从真实模型输出测得；由于该目标函数对给定合成值随 α 单调上升，选择最大 α 是输入构造的结果，不是模型实验发现。空正负样本会传播 N/A；若单个 case 激活向量为零，`projection_score` 返回 `None`，随后 `after - before` 会抛 TypeError；`dot`/`add` 使用 `zip` 而没有维度一致性校验。精修时需明确输入契约或处理这些边界，改动可执行行后按代码门禁复验。

## 图表和完成门禁

- `visual_map`：章首控制路径和若干段落目前使用 `text` 围栏，没有 Mermaid。建议评估用一张流程图连接“行为目标/对照样本 → 激活方向 → 层位与强度扫描 → 多维行为/安全评估 → holdout/撤销”；用小表或配对轨迹解释干预前后激活与输出。若使用 Mermaid，遵守新增图表规范：标准 `mermaid` 代码围栏、带中文/空格/标点的节点标签优先加引号、带文字虚线箭头使用 `-.->|标签|`；核验围栏、语法和 GitHub 页面渲染。仅源码存在不能算渲染门禁通过。
- `completion_gate`：本轮完成章级结构和定向研究扫描。当前尚未完成整册第九至十六章扫描、coverage map、篇章篇幅验收、正文精修、图表渲染和整册阅读路径复读，因此第八册及本章均未过完成门禁。
- `open_risks`：讲解单元的真实边界和字数待全册映射后确认；§9 篇幅与 §8/§11/§12/§13 内容重叠待处理；demo 的指标为人造输入，不能暗示真实模型测量；零激活向量和维度错配未被安全处理；公式格式和图示/平台渲染未验收；近年论文的结论均绑定原模型、数据集、评估器和实验协议。
- `validation`：静态通读 19 个二级段，完成定向 primary-source 阅读、粗略长度诊断和 demo 关键值手算；未运行 Python、未请求章节外链扫描、未渲染公式/Mermaid、未构建 PDF。

## 参考来源

1. Mckenzie et al., [Endogenous Resistance to Activation Steering in Language Models](https://proceedings.mlr.press/v306/mckenzie26a.html), ICML 2026。
2. You et al., [Spherical Steering: Geometry-Aware Activation Rotation for Language Models](https://proceedings.mlr.press/v306/you26a.html), ICML 2026。
3. Nguyen et al., [Activation Steering with a Feedback Controller](https://proceedings.iclr.cc/paper_files/paper/2026/hash/fa5617c176e76fee83f3f9947fdf9f3f-Abstract-Conference.html), ICLR 2026。
4. Siu et al., [SteeringSafety: Benchmarking Representation Steering in LLMs Across Safety Perspectives](https://arxiv.org/abs/2509.13450), arXiv v3 (2026)，接收于 ICML 2026。
5. Le & Le, [Adversarial Robustness of Activation Steering in Large Language Models](https://arxiv.org/abs/2606.07696), arXiv, 2026-06。
6. Xu et al., [How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities](https://arxiv.org/abs/2603.02578), ACL 2026。

- `status`：第八章全文结构和研究主题扫描完成，正文尚未修改，demo 未运行；下一步扫描第九章，将 SAE 表示干预、参数编辑与 unlearning 的边界放入第八册 coverage map。
