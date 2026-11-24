# 第十八章《RoPE 长上下文扩展：插值、频率缩放与训练》精修审计

审阅日期：2026-11-20。正文重写、研究记录、配套同步和本轮静态检查已完成；GitHub MathJax 公式实际渲染、读者试读和外链逐条终审留在后续整册门禁。不更新 PDF 或 `plan.md`。全套书精修 goal 继续 active。

## 读者契约与章节边界

- **前置：** 已理解自回归注意力、RoPE 中位置对 Q/K 旋转的基本关系和上下文窗口；第七章负责位置表示基础。
- **离章结果：** 能解释位置插值、NTK-aware/Dynamic NTK、YaRN、非均匀位置映射和长上下文训练的差别；能区分配置长度、训练长度、位置映射范围和指定任务上的有效长度；能按 checkpoint、训练方案与评测任务限定长上下文主张。
- **章节职责：** 承接第十七章的模型配置比较，主讲预训练 RoPE 模型如何改变位置映射和适配长序列。
- **范围外：** 不重推 RoPE 的二维旋转和全族位置编码；长距离信息使用与位置敏感评测回链第十三章，KV/cache 回链第十一章，稀疏连接回链第十九章，attention kernel 回链第二十章；NoPE、iRoPE 等架构专题回链第六十四、六十七、七十二章。

## 旧稿诊断与结构决定

旧稿共 267 行、28 个 H2，交织 RoPE 公式、PI/NTK/YaRN、ALiBi 与其他位置方法、Python 实现、位置及性能评测流水线、KV cache/runtime/kernel 和多模态部署，章节主线和相邻章节职责不清。精修前已完整阅读旧稿，并结合册级架构审计、第十七章交接及第七、十一、十三、十九、二十章的分工重新研究和设计。

新增完整章节大纲[《RoPE 长上下文扩展》](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/18-RoPE与长上下文位置外推.md)，保留章节文件路径和编号，将正文重组为六节：

1. 区分位置映射范围、训练长度、配置上限与有效上下文。
2. 解释 Position Interpolation 如何把推理位置映回预训练范围及其相邻距离压缩代价。
3. 比较 NTK-aware、Dynamic NTK 与 YaRN 对频率变化和训练适配的处理。
4. 用 LongRoPE、MrRoPE、Jet-Long 说明非均匀或长度自适应映射，并限定模型、任务与 cache 证据。
5. 以 Llama 3、LongRoPE2、Shuffle the Context 对照位置映射之外的长序列训练适配。
6. 用方法对照表收束比较条件，不把最大窗口数字当成方法排名。

## 研究核验与正文决定

知识点、原文阅读位置、版本和证据边界见[RoPE 长上下文扩展研究记录](../../research/rope-context-extension-2026-11-20.md)。主要取舍如下：

- 保留一条 PI 位置映射公式；正文先定义训练长度、目标长度和位置，再用扩展例子解释映射后的坐标。PI 的 2K→32K、微调步数等数据限定为论文的 LLaMA 实验。
- 将 NTK-aware 标为社区启发式，将 Dynamic NTK 标为随当前序列长度变化的位置规则；不因名称含 NTK 而声称它由严格 NTK 定理推出。YaRN 依 ICLR 2024 正式论文解释为 NTK-by-parts 频率插值与 attention logit 温度缩放组合，并把效率数字限制在论文配置。
- 将 LongRoPE、MrRoPE、Jet-Long 的模型与 benchmark 结果绑定对应论文。Jet-Long 的 cache 位置校正不重算历史 hidden states/values，因此不能把缓存解码等同于每一步完整重算；kernel 性能留给第二十章。
- 将位置坐标改变与权重/数据适配分开。Llama 3、LongRoPE2 和 Shuffle the Context 说明不同训练路径，不把某个报告方案写成通用配方或因果结论。
- 删除 Python RoPE 实现、cache/kernel 操作教程、位置/性能流水线和多轮重复验收；NoPE、iRoPE 及多模态位置机制回链对应章节。

## 配套同步

- 更新第二十一册目录与核心主线速读中的第十八章标题。
- `EXERCISES.md` 新增 31W：计算 PI 映射并区分位置坐标与有效上下文；`INTERVIEW_BANK.md` 新增 31T：解释配置长度为何不是能力证明。
- `PAPERS.md` 增补 YaRN 正式 ICLR 论文、LongRoPE2、MrRoPE、Jet-Long、Shuffle the Context 和 Llama 3 长上下文训练来源及范围。
- `GLOSSARY_EN_ZH.md` 补充 NTK-aware/Dynamic NTK、NTK-by-parts、LongRoPE2、MrRoPE、Jet-Long、Shuffle the Context 条目；`KNOWLEDGE_GRAPH.md` 增加位置扩展链。RoPE、Position Interpolation、YaRN 与 LongRoPE 已有定义，不重复添加。

## 静态验收与遗留项

- 正文 H2 `18.1—18.6` 连续；两张正文表格分别说明长度概念和方法差异；PI 公式变量和扩展倍率在公式之前定义，位置映射结果在随后解释。
- 正文不含代码或 Mermaid；章节标题与目录、核心主线速读一致，旧标题全仓检索无残留。
- 新增练习 `31W`、面试题 `31T` 的编号在各自题库中唯一；研究记录、章节大纲和正文引用路径存在。
- `git diff --check` 通过。公式源码检查通过；**GitHub MathJax 视觉渲染未验收**，因为工作副本修改尚未进入目标页面。未运行测试、代码、模型、API、benchmark 或 PDF 构建。
- 后续门禁：确认 GitHub 页面上的 PI 公式渲染；目标读者试读、第二十一册连续复读和本章外链全量终审纳入后续全册门禁。

下一入口为[第十九章《Sliding Window Attention、Global Attention 与稀疏注意力》](../../book-21-transformer-architecture-evolution/chapters/19-Sliding-Window与稀疏注意力.md)。第 1—18 章已审阅与修订；第 19—95 章共 77 章待逐章精修。
