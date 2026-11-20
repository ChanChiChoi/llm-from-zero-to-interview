# 第十二章《In-Context Learning 与显式 token 检索能力》精修审计

审阅日期：2026-11-20。状态：章节职责核验、正文修订、配套同步和静态源码检查完成；公式的 GitHub 实际渲染尚未验收，因此本章最终呈现门禁未完成。未运行代码、测试、模型、API、benchmark 或 PDF。

## 读者契约与章节边界

- **前置：** 理解基础 Transformer、next-token prediction 和 self-attention；本章只唤回解释上下文信息流所需的 query/key/value，不重复 attention 推导。
- **离章结果：** 能区分固定参数下的 ICL 与参数更新；说明 demonstrations 可传递的多种信号；区分 prompt/ICL 与 RAG 外部检索；设计映射、顺序或证据位置的受控对照；把可见 attention 路径、模型实际使用证据的行为和内部机制解释分开。
- **章节边界：** 本章主讲上下文条件化及其行为证据。长上下文位置评测转交第 13 章；归纳偏置与状态记忆转交第 14 章；跨架构 workload 比较转交第 42 章。模型架构总体评价不能由单项检索或 ICL 结果决定。

## 外部研究与证据边界

来源、版本、实际阅读位置和适用限制记录在[ICL 与上下文内检索来源证据](../../research/in-context-learning-and-context-retrieval-2026-11-20.md)。本章综合 GPT-3 zero/one/few-shot 评测、Min 等人对 demonstrations 的分类/多选实验、ICL 的 induction-head/潜在任务/算法模拟解释、prompt 顺序敏感性、长上下文位置结果、RAG 原始架构，以及 attention 权重解释性争论。课程材料 Stanford CS224U 2023 和 CME295 Fall 2025 用于定义与教学路线，不当作普遍效果证据。

关键限定：few-shot 标签不证明任务对训练数据真正新；Min 等人的随机标签发现限于其分类和多选设置；ICL 机制研究各自依赖小模型、受控假设或合成任务；顺序和位置效应限于原研究模型与任务；attention 权重不自动构成忠实解释，但相关结论也不能概括所有 Transformer 电路。RAG 的检索和生成阶段应分开评价。

## 正文与配套修改

- 将第十二章正文重写为九节，以任意映射 `dax → vep` 说明输入上下文如何改变当前预测；依次解释 ICL 与 fine-tuning、示例信号、prompt/RAG 边界、attention 信息路径、机制解释、顺序/位置效应及架构评测用途。
- 增加 attention 加权汇总公式，并定义 query/key/value、可见位置、head 维度、归一化权重和加权输出；明确可见路径不保证证据被正确定位、绑定或使用。引用 Jain 与 Wallace、Wiegreffe 与 Pinter 时同时注明其任务范围及对解释定义/测试的争论。
- 将 induction heads、潜在任务推断和类梯度下降解释分别绑定到原研究的模型与实验条件；不把某个解释写成 ICL 的通用内部算法。将 Lost in the Middle 的位置细节回链第 13 章。
- 移除无来源的检索命中率、冲突率、位置风险阈值和 ICL 通用门禁；正文改为受控映射、顺序和证据位置对照，不承诺一个合成查找题代表长文档能力。
- 同步[面试题 31O](../../INTERVIEW_BANK.md)、[练习 31R 与 54C](../../EXERCISES.md)、[ICL Context Dependence Brief](../../PROJECTS.md)、[术语表](../../GLOSSARY_EN_ZH.md)和[知识图谱](../../KNOWLEDGE_GRAPH.md)，移除对应的 toy 相似度/伪指标表达。

## 静态检查与遗留项

- H2 `12.1—12.9` 连续；6 个 Markdown 围栏标记成对；逐行检查行内数学定界符，未发现未配对美元符；全角标点与数学定界符相邻扫描无命中；3 个本地章节链接目标均存在；未发现尾随空白或控制字符。
- 手工复核 attention 公式中的可见位置集合、head 维度和加权输出定义；本轮新加入的 cost/latency 与 attention-explanation 主张均有相邻来源链接。`git diff --check` 通过。
- **待验：** 修改仍在工作区，尚无对应的 GitHub 实际页面可逐式检查；需在目标页面可见本版后确认 12.1、12.5 两个块级公式的 MathJax 渲染及邻近段落布局。在完成该项前，不将本章公式渲染或最终呈现门禁标记为通过。PDF 按用户要求不处理。
- 下一入口为第十三章《长距离依赖与 Lost in the Middle》；第十二章 GitHub 渲染待验项留待最终呈现门禁补验。全套书精修 goal 继续 active。
