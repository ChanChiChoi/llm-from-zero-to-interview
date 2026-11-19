# 第 6 章《Residual、LayerNorm、RMSNorm、Pre-Norm 与训练稳定性》审阅记录

审阅日期：2026-11-19。状态：章级来源扫描、正文重写、配套资料同步和本地静态检查完成；未运行代码，目标平台 MathJax 渲染、目标读者试读与第二十一册连续复读留待后续门禁。

## 读者契约与章节职责

- **书册目标贡献：** 让读者理解 Transformer 子层如何把更新写回 residual stream，区分 LayerNorm 与 RMSNorm，沿 block 方程比较 Pre-LN/Post-LN，并按证据范围理解训练稳定性设计。
- **读者前置：** 已理解 attention、FFN 和 hidden representation；能读向量、导数及基础 Transformer block 方程。第 5 章刚解释 attention 与 FFN 的子层计算。
- **离章结果：** 能复述 $x+F(x)$ 的恒等路径和局部 Jacobian；手算 LayerNorm/RMSNorm 的小例子；写出 Pre-LN/Post-LN 方程；说明 Xiong 等人的 warmup 结果适用边界；比较 Admin、NormFormer、DeepNorm 与 Peri-LN 的不同着力点；从绑定配置的真实 trace 中提出稳定性排查方向，而不套用虚构通用阈值。
- **读者路径：** residual 更新 → 恒等梯度分量 → 两种归一化的统计与数值结果 → norm placement → 梯度路径和 warmup 证据 → residual-branch dependence 与后续设计 → 真实训练 trace 的观测口径。
- **前后交接：** 承接第 5 章的 attention/FFN 输出，下一章转向位置编码。NormFormer、DeepNet 以及训练配方仅解释其架构问题和边界，不展开训练工程配方。

## 外部研究与主题去向

2026-11-19 通过中英文题名、术语与课程检索词完成章级研究和原文核验。检索范围、阅读位置、版本、证据等级和未覆盖边界见[Residual、Norm 与训练稳定性证据记录](../../research/transformer-residual-norm-stability-evidence-2026-11-19.md)。Stanford CS224N 讲义只用于核对一种教学组织方式，不用于宣称该顺序有学习效果证据。

| 外部主题 | 原文支持与读者需要 | 本章去向 |
| --- | --- | --- |
| Residual mapping | ResNet 将目标映射写为输入加分支更新；Transformer 将 residual add 用于 attention/FFN 子层。恒等分量提供直接传播路径，但不保证多层状态或梯度普遍有界 | 6.1—6.2 解释加法锚点、局部 Jacobian 和非保证边界 |
| LayerNorm / RMSNorm | LayerNorm 对单个样本的特征统计并有可学习 gain/bias；RMSNorm 不减均值、按 RMS 缩放。速度不能只从公式简洁度推断 | 6.3—6.4 用同一 $[1,3]$ 算例比较归一化过程和结果 |
| Pre-LN / Post-LN 与 warmup | 原始 Transformer 使用 Post-LN；Xiong 等人的分析和指定 IWSLT14/WMT14/BERT 实验支持其范围内的 Pre-LN 梯度/warmup 结论，不支持普遍删除 warmup | 6.5—6.6 由 block 方程推到局部 Jacobian，再限制 warmup 推论 |
| Admin | 训练稳定性还涉及 residual branch 对参数扰动的依赖；过强或过弱都有代价。论文证据来自其机器翻译设置 | 6.7 说明问题机制和实验边界 |
| NormFormer / DeepNorm | 前者处理 Pre-LN 层间梯度幅度；后者共同设计残差缩放与初始化，实验为多语机器翻译深层模型 | 6.8 用对齐表区分作用位置、作者报告和迁移边界 |
| Peri-LN 与后续理论 | Kim 等人 ICML 2025 论文把已有的子层输入/输出外围归一化做法归纳并命名为 Peri-LN，报告到 3.2B 的特定实验；Kan 等人 2026 v2 是后续控制理论分析，实验为 GPT-2 家族、OpenWebText 和 H200，不能和前一工作合并成同一证据 | 6.8 明确两篇来源、贡献和实验范围分别归因 |
| 训练稳定性观察 | activation、gradient、update、loss 及非有限值处于不同诊断层次；来源不支持一个跨模型的统一通过阈值 | 6.9 只定义真实 trace 的分层观察量与配置绑定，删去 toy gate |

## 正文取舍与配套同步

- 从头组织 6.1—6.11 教学路线；移除原作者工作计划、资料清单式开头、将 Pre-LN 或 warmup 描写成通用规则的句子，以及没有来源支撑的更新门槛和稳定性判定。
- 删除自拟 Python residual/norm 审计 demo。该章目标是解释机制和判断证据范围，没有要求读者编写程序复现；因此正文不需要代码。原练习和项目若依赖 demo 输出，也一并改成数值手算与真实训练记录分析。
- 用 $x=[1,3]$ 同时解释 LayerNorm 与 RMSNorm；公式与数值例子保持同一统计轴、仿射参数和 epsilon 假设。
- 把 Peri-LN 的原始命名/架构研究与 Kan 等人的后续稳定性预印本分开引用，避免把架构命名、理论框架及不同 GPT-2/DCLM 实验混为一项结果。
- 用 residual-branch 输出与输入流范数比观察分支尺度，并明确它不是 parameter update ratio；项目条目要求绑定实际模型 revision、初始化、optimizer、学习率、dtype 与训练记录，不使用任意阈值或单一 gate 宣告稳定。
- 同步 `EXERCISES.md` 31F—31G、`INTERVIEW_BANK.md` 31F、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、`PAPERS.md` 和 `GLOSSARY_EN_ZH.md`，移除旧 demo/gate 依赖并明确方法间的证据差异。

## 验收与遗留风险

- 静态验收：H2 标题 `6.1—6.11` 连续；16 个 Markdown 数学围栏标记成对；行内数学定界、全角标点间距及 2 个本地章节链接检查通过；正文无代码块和 `\operatorname`；相关练习/面试题编号唯一，旧 residual/norm toy demo 和稳定性 gate 引用已清理；本章相关已跟踪文件的定向 `git diff --check` 通过。
- 全仓 `git diff --check` 另报告一处与本章无关的既有改动：`book-05-llm-training/chapters/11-后训练与对齐训练.md` 第 32 行有尾随空格。本轮未改动该文件，以保留工作区中已有的第 5 册正文修改。
- 未运行代码、测试、模型、API、benchmark 或 PDF 构建/更新；目标平台 MathJax 实际渲染和目标读者试读未进行。
- 论文结果均为作者报告，未复现训练。尤其 Kan 等人的 2026 v2 仍是 arXiv 预印本，理论受建模条件限制；两篇 Peri-LN 研究的模型、数据与预算不可互换。外链全量可达性和全册连续复读留到后续门禁。
- 下一步为第 7 章位置编码；全套书精修 goal 继续 active，第二十一册尚有第 7—95 章待审。
