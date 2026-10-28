# 第八册第八章 Steering 与 Representation Engineering 精修记录

- `date`：2026-10-28。
- `scope`：依据当前 `WRITING_SPEC.md` 从头复读第八章，在 2026-10-27 结构/研究扫描基础上完成正文精修；重核核心来源，检查讲解单元、手算、公式、图表、代码、参考链接及配套索引。
- `chapter_role`：从行为目标构造推理时表示方向，区分表示可读出性、局部控制效果、跨任务迁移和系统安全后果。第七章解释机制证据，第九章讲持久参数编辑和 unlearning；本章不把 activation patching 与线上 steering 混为一谈，也不把 steering 当作权限控制。

## 复核来源与证据边界

1. [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405) 用于支持从高层行为出发寻找可监控表示的研究路线。论文里的模型和任务结果不外推到任意内部方向。
2. [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248) 的 arXiv 当前标题与早期流传标题不同；本章按当前页面标题引用，并将 Activation Addition（ActAdd）作为方法名称说明。
3. [Steering Llama 2 via Contrastive Activation Addition](https://aclanthology.org/2024.acl-long.828/) 是 CAA 的 ACL 2024 正式论文，确认其使用配对正负样本的 residual-stream 激活差，并在生成时施加 steering。正文原来误链 `arXiv:2312.07569`，该编号对应光学论文；现已更正为 ACL 正式论文页。
4. [Endogenous Resistance to Activation Steering in Language Models](https://proceedings.mlr.press/v306/mckenzie26a.html) 的 ICML 2026 摘要报告了 SAE latent steering 下的生成中恢复现象：Llama-3.3-70B 更明显，较小 Llama-3 和 Gemma-2 模型较少出现。此结果支持检查 token 序列和多轮持续性，不表示所有 steering 都会被抵抗。
5. [Spherical Steering](https://proceedings.mlr.press/v306/you26a.html) 研究范数保持旋转与置信度门控；[Activation Steering with a Feedback Controller](https://proceedings.iclr.cc/paper_files/paper/2026/hash/fa5617c176e76fee83f3f9947fdf9f3f-Abstract-Conference.html) 将常见方法联系到比例控制并提出 PID steering。两篇摘要中的稳健性/稳定性主张仅绑定各自形式、模型和评测设置。
6. [SteeringSafety](https://arxiv.org/abs/2509.13450) 覆盖九种安全视角和 18 个数据集，在 Gemma-2-2B、Llama-3.1-8B、Qwen-2.5-7B 上比较 steering。论文报告目标视角之间的行为纠缠；相关数字不是其他模型或产品上的风险率。
7. [How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities](https://aclanthology.org/2026.acl-long.1443/) 的出版 PDF 确认 benchmark 名为 SteerEval，按 language features、sentiment、personality 三类和 L1–L3 目标粒度评估。ACL 元数据摘要末句出现 `SteerBench` 名称不一致；本记录和正文依据论文 PDF 的正文与图表称 SteerEval。
8. [Adversarial Robustness of Activation Steering in Large Language Models](https://arxiv.org/abs/2606.07696) 是 2026-06 arXiv 预印本。摘要范围为五种 1.5B–30B 模型、四种方向提取方法、三类文本扰动及六种 persona；摘要报告方向稳健性最高下降 64%、最优层位最多移动 17 层。正文将其限为作者设置下的风险线索，不称为生产攻击率。
9. [Contrastive Covariance Subspaces for Activation Steering](https://arxiv.org/abs/2610.14025) 是 2026-09 预印本，研究控制方向估计中的共变混杂；作者报告三种模型、四个 steering 域的实验及多轮样本。当前证据尚不足以说该估计器可跨模型替代均值差。
10. [AI Steerability 360](https://aclanthology.org/2026.acl-demo.43/) 的 ACL 2026 系统演示介绍 input、structural、state、output 四种控制面及组合式 pipeline；该工具降低实现/比较门槛，但不能作为某种 steering 的有效性证明。
11. [TransformerLens 文档](https://transformerlensorg.github.io/TransformerLens/) 当前文档导航显示 4.0 主版本入口，支持将其作为激活提取/干预工具的入口而非 steering 质量保证；[NIST AI RMF 页面](https://www.nist.gov/itl/ai-risk-management-framework)提示 AI RMF 1.0 正在修订，因此正文提醒记录治理框架版本。

## 正文与配套修改

- 扩展 §1.4–§1.5，把四种控制面放回输入、参数、激活、解码和系统权限的范围内；用同一类证据受限问答故障说明不同控制面的责任边界。
- 扩展 §4.5，分别解释方向“能读出”“能控制”“能迁移”需要什么证据；以匹配的证据引用对照说明风格捷径如何造成伪提升。
- 补足 §6.2、§6.4 的 CAA 配对流程、层位/token 读取与注入范围；更正错误参考链接。
- 在 §5.4 加入生成中抵抗/恢复和共变混杂方向估计；在 §7.5 加入多视角安全评估；在 §9.3 加入 SteerEval 粒度和语义、时间、对抗三轴检查，并明确各指标的 $M$ 是各自切片的分母；在 §12.2 加入输入扰动与层位漂移证据；在 §13.2 加入几何旋转和 PID 反馈方向。
- §9.6 增加可复算的 toy 数值：目标提升 `0.078`；安全收益在高风险切片上为 `0.12`；正常能力损失为 `0.0173`；误拒增量为 `0.0107`；教学加权值约 `0.171`。各指标使用不同的明确分母；toy 输入不可解释成模型实测。
- 将全章原有 49 对行内 `\(...\)` 改成 `$...$`，并新增 7 对 MathJax 行内公式（含 5 组数值代入和 2 处分母记号）；目前共 56 对，14 组 display 公式迁移为 `math` 围栏；无 `\operatorname`。章首 Mermaid 使用 `mermaid` 围栏、节点标签加引号，带文字虚线箭头写作 `-.->|重新构造并评估|`。
- §18 代码增加激活维度/有限数检查、零向量与零激活的 `None` 传播、切片分母隔离；强度候选先过每一项教学门槛，再按 toy objective 排序。代码和状态附近只写运行状态，不包含维护文件名或内部工作流引用。
- `PAPERS.md` 增加本章方法与评估来源；`EXERCISES.md` 增加 35EC 的分母、缺测与研究对照题；`INTERVIEW_BANK.md` 更新 steering / representation engineering 问题。

## 讲解单元篇幅

按连贯讲解职责分组。以下是脚本统计的中文解释性汉字数，排除标题、代码围栏、表格、公式本身、状态行和空白；用于维护记录，仍需人工判断论证深度。

| 单元 | 范围 | 类型/门槛 | 统计 | 状态 |
| --- | --- | --- | ---: | --- |
| 控制位置与表示工程基础 | §§1–4 | 核心 / 1,800 | 2,007 | 达标 |
| Steering vector 与 CAA | §§5–6 | 核心 / 1,800 | 2,041 | 达标 |
| 拒答、安全权衡与评估 | §§7–9 | 核心 / 1,800 | 3,208 | 达标 |
| 方法组合、运行与治理边界 | §§10–13 | 核心 / 1,800 | 2,346 | 达标 |
| 企业知识助手案例 | §14 | 辅助 / 1,000 | 1,222 | 达标 |
| 可运行审计演示 | §18 | 辅助 / 1,000 | 1,061 | 达标 |

§15 为来源与证据边界，§16 为误区索引，§17 为练习，§19 为总结；未把它们包装成独立讲解单元来满足篇幅门槛。

## 例子、代码、公式和开放门禁

- `example_status`：§5.3 手算正负均值差 `[1.0, 1.0, -0.8]`、范数 `sqrt(2.64)≈1.625`、单位向量 `[0.615, 0.615, -0.492]`，并把 `α=0.4` 加到 `[0.4, 0.2, 0.4]` 得 `[0.646, 0.446, 0.203]`；§9.6 复算目标、风险、正常能力和误拒各自的聚合。
- `code_status`：§18 当前为已验证（2026-10-28；Python 3.12.9，标准库）。抽取代码块执行，程序输出与正文预期逐行一致。另覆盖零方向、空高风险切片、零激活以及 `add`、`subtract`、`mean_vector` 三类维度不匹配；缺测返回 `None`，维度错配抛出 `ValueError`。这些是 toy 本地验证，不是模型效果验证。
- `formula_status`：行内公式使用单 `$`；14 组 display 数学使用 `math` 围栏；围栏已静态检查。未完成 GitHub 页面 MathJax 渲染。
- `visual_status`：Mermaid 语法、引号标签和带标记虚线箭头已静态检查；未在 GitHub 页面检查渲染，因此图示门禁仍开放。
- `link_status`：CAA 错链已更正；本轮新增论文页和官方会议页均作了 primary-source 阅读。未对全章所有已有外链执行 HTTP 扫描，未构建 PDF。
- `sync_status`：正文、研究记录、`PAPERS.md`、`EXERCISES.md`、`INTERVIEW_BANK.md` 已同步；目录入口原有，无需调整。
- `open_risks`：所有论文结论绑定其模型、任务、层位、数据和评估器。没有目标模型、运行时 hook、真实流量、工具 trace 或隐私授权数据；不声明产品效果或安全保证。GitHub Mermaid/MathJax 渲染、全章外链 HTTP 扫描、PDF、全书 coverage map 与沿主阅读路径的整体复读仍待完成。
- `status`：第八章正文已实质精修并完成声明范围的代码/手算/静态格式检查；不标记章节完整门禁通过。下一步按顺序精修第九章 Model Editing 与 Unlearning。
