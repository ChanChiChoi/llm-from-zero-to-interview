# 第十章精修记录（本轮独立审阅）

## 范围与顺序

- 任务：按本轮修订后的写作规范精修第五册；允许保持章节中心不变的正文重写。
- 用户要求从规范模块 02a 开始、跳过模块 01，并忽略此前 `progress/` 记录。本轮没有读取或借用旧 progress 内容，按章节顺序从第 9 章继续到第 10 章。
- 第 10 章契约 `写作规范/全套书不同册的章节大纲结构/book-05-llm-training/10-checkpoint评估与实验管理.md` 共 145 行，已完整阅读；正文 `book-05-llm-training/chapters/10-checkpoint评估与实验管理.md` 共 1213 行，已从头到尾阅读。
- 本记录和对应 research 文件均为本轮新建；审阅判断留在此处，知识与来源证据留在 research 文件。

## 中心与初审判断

本章中心是回答“哪个 checkpoint 值得继续”：把可恢复训练状态、保存成本、恢复探针、validation/per-domain/benchmark/regression 证据、实验元数据和多目标选择连成一条可复核链路。正文 1–19 节与契约层级一致，保持标题结构，不扩展分布式保存实现、全生命周期预算或新的书册主题。

初审发现三类需收紧的内容：

1. `$C_{save}/(fT_{step})$` 表示保存时间相对于计算时间的开销比；正文把它近似描述为总时间占比，容易让读者混淆计算基线与总墙钟分母。
2. PyTorch 2.15 DCP 的异步 staging / upload 两个完成 future 应限定于采用对应 `AsyncStager` 接口的路径；DCP 支持 load-time resharding，而“改变 shard 数加载失败”只能描述为格式不支持重切分时的故障可能。
3. 第 14 节步骤清单后的教学例子误落在同一个 fenced code block 中，会被渲染为代码。第 15 节现有三百多行只处理预设 checkpoint 字典，没有训练循环，也没有实际保存、恢复和评估调用，与契约不符；应改成简短 CPU PyTorch 训练循环。旧 demo 的“已验证”状态不能挪用成本轮验证。

此外，正文存在相邻章节间重复解释“分数不是能力总排名”“阈值和权重受项目目标约束”等判断。本轮采取局部收束，保留对初学者有用的机制、计算、边界与例子，不重写契约要求的评测和选择步骤。

## 本轮研究

- 已对 PyTorch 2.15 DCP、异步保存、随机性说明，以及 Hugging Face 的 fixed-length PPL 文档进行一手资料核验。
- 已对 HELM、lm-eval 可复现评测、Dror NLP 显著性检验、Oren benchmark contamination 和 ConStat 原始论文核验其定义、数字与适用边界。
- 具体版本、来源类型、支持的主张和不能外推的边界见 `research/checkpoint-evaluation-experiment-management-evidence-2026-11-24.md`。

## 预定编辑与审阅边界

- 保持全章中心、契约标题和章节顺序不变。
- 已修正保存开销比的分母解释、DCP 2.15 异步返回状态的适用条件和 shard 数变化加载的条件；第 14 节教学说明已移出错误的 code fence。
- 已把第 15 节改为最小 CPU PyTorch 训练例，展示模型/optimizer/scheduler/RNG 与游标保存、恢复后继续训练、JSONL 追踪和验证集评估。代码未执行；范围限定在单进程 CPU，不涵盖分布式、对象存储和跨版本恢复。
- 第 14 节的三个候选现在以独立构造数据呈现，并明确固定恢复完整性、稳定性、safety、回归和领域门槛；正向 regression 得分与风险项公式的关系已说明。
- 完整静态复核了本章公式定义与手算例子；没有改动展示公式源码。未做 GitHub/PDF 目标平台渲染，渲染验收仍待完成。
- `git diff --check` 通过；28 个编号结构标题与契约匹配；41 组 fenced blocks 闭合；31 个 math fences 成对；23 个外部链接，无内部相对链接缺失；新建 progress/research 文件无行尾空白。
- 未运行 demo、未构建 PDF、未执行测试套件。下一章按顺序进入第 11 章，不查阅旧 progress 记录。
