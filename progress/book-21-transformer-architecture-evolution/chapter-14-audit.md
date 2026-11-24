# 第十四章《Transformer 的归纳偏置不足：顺序、局部性和状态记忆》精修审计

审阅日期：2026-11-20。状态：章节职责复核、章级研究、正文修订、配套同步和静态源码检查完成。正文含一处块级状态更新公式；源码、符号与教学含义已核对，GitHub MathJax 实际渲染尚待目标页面可见后验收。未运行代码、模型、API、benchmark、测试或 PDF，也未更新 `plan.md`。

## 读者契约与章节边界

- **前置：** 已知 self-attention、因果生成、位置线索和序列模型的基本对象。本章只唤回理解顺序和历史访问所需的最小内容，不重讲位置编码、mask、KV 容量或 SSM 推导。
- **离章结果：** 能用局部连接/共享、递归状态更新和内容相关 attention 解释 CNN、RNN/SSM 与 Transformer 的结构先验；说明无位置输入、无位置偏置且无 mask 时 self-attention 的置换等变性质，以及 causal mask 如何改变该对称性；区分逐位置 KV 表示、递归状态和模型外部记录，并按未来查询需要判断它们各自保留/读取什么。
- **本章职责：** 作为第 13 章长距离证据依赖之后的概念桥接，解释序列顺序、局部性和历史状态如何由架构提供；以 bounded evidence 说明 exact recall 与状态大小为何需要按任务讨论。
- **范围外：** 第 7/8 章完整讲位置与可见性；第 11 章讲 KV cache 容量和 serving 状态；第 15 章讲流式与在线服务；第 29—33 章讲 SSM/RNN/CNN/Mamba 机制、数学和硬件；第 42 章负责横向架构比较；第 45/46 章负责检索增强和外部记忆架构。本章不维护一套比较 benchmark、项目实验或 serving 门槛。

## 外部主题、来源和内容去向

新增研究知识、来源版本、实际阅读位置和限制见[归纳偏置与状态记忆研究记录](../../research/transformer-inductive-bias-and-state-memory-2026-11-20.md)。

| 主题 | 来源支持范围 | 本章处理 |
| --- | --- | --- |
| CNN 的局部性和共享规则 | D2L 解释局部感受野与跨位置共享如何形成 locality/shift equivariance；等变不等于分类不变 | 保留在架构先验表和文字解释，避免把“位置不敏感”泛化成结论 |
| Transformer 的顺序 | Vaswani 原论文使用 attention 替换 recurrence/convolution 并加入位置信息；D2L 给出无位置/无 mask self-attention 的置换等变条件 | 区分 causal mask 的前后可见性和位置机制所提供的位置线索；不再声称 causal decoder 靠显式位置编码才拥有顺序 |
| 显式历史与递归状态 | D2L RNN 章节定义固定形状的递归状态；第 11 章覆盖逐层 K/V cache | 说明两者分别以逐位置表示与递归更新保留信息；缓存不是原文数据库，状态也不保证保留任意旧事实 |
| S4 与 Mamba 的训练/推理表示 | S4 教学资料展示递归与卷积表示；Mamba v1 说明输入依赖选择、固定卷积等价性的变化和并行 scan | 作为计算表示的交接例子，避免把递归架构误写成训练必定逐 token 串行 |
| Context recall 和有限状态 | Zoology v1 的 Pile 重复 n-gram 切片/MQAR；BASED v1 的 MQAR 状态规模实验及有界状态精确 recall 理论 | 在任务和模型范围内报告；明确不外推为所有语言任务或所有状态结构的普遍容量结论 |

## 正文修改与配套同步

- 将原 19 节收束为 6 节：先用“最新配置 vs 指定旧记录”建立问题，再解释结构先验、顺序信号、逐位置 K/V 与递归状态、按任务读取的 evidence，最后给出结构问题和跨章交接。
- 标题与目录的“Transformer 的归纳偏置不足：顺序、局部性和状态记忆”同步。正文区分卷积的局部感受野与权重共享；只在相应条件下称 shift equivariance，不把它说成 translation invariance。
- 修正“无位置编码则 Transformer 不知道顺序”的粗略说法：置换等变性质限定于无 position signal/bias/mask 的 self-attention；causal mask 本身已经引入顺序可见性，位置机制再提供设计相关的位置线索。
- 把 KV cache 定义为逐层已处理 K/V 表示，说明它不是原始文本数据库；将固定形状递归状态说明为有更新规则的历史表示，不再将状态元素数或自拟碰撞分数当作有效信息量定理。
- 用 S4/Mamba 交代递归规则与训练计算表示不同；将精确 recall 证据限定到 Zoology/BASED 的 Pile/MQAR 设置和论文假设。
- 删除旧稿的模拟 Python、顺序消融流程、自拟状态容量/碰撞公式、任意样本效率和架构得分、paired experiment、固定阈值、长 serving 指标及重复练习/误区段落。
- 更新配套资料：新增 `EXERCISES.md` 31S；修订 `INTERVIEW_BANK.md` 31A 对 SSM 训练串行性和架构胜负的过度概括；在 `GLOSSARY_EN_ZH.md` 添加 Inductive Bias；在 `PAPERS.md` 增加 Zoology 和 BASED 边界入口；将相关节点加入 `KNOWLEDGE_GRAPH.md`。目录题名已与正文标题一致，无需额外改动。

## 静态验收与遗留项

- H2 标题 `14.1—14.6` 连续；正文有 1 个 `math` 围栏（成对）和 1 张 4 列比较表；无代码或 Mermaid 块。
- 逐式核对状态变量 `s_t`、输入 `x_t`、输出 `y_t` 及递归关系；行内数学和引用文本对应。公式源码检查通过，尚未确认 GitHub MathJax 渲染。
- 本章 12 个本地章节链接均对应现存文件：第 7、8、11、15、18、29、30、31、33、42、45、46 章。原论文、D2L、Annotated S4、Zoology、BASED 与 Mamba 链接已针对正文主张复核。
- 已扫描相邻习题/题库/论文索引/术语/图谱：删除的章内练习由 31S 承接；面试题 31A 已修正架构比较边界；无须同步简介或目录结构。
- `git diff --check` 通过。未运行代码、测试、模型、API、kernel、benchmark 或 PDF。
- **未完成门禁：** GitHub MathJax 逐式视觉验收、目标读者试读、全章外链可达性巡检和第二十一册连续复读仍待后续完成；第 12 章两处块级公式的 GitHub MathJax 实际渲染也仍待验收。PDF 按用户要求不处理。
- 下一入口为第十五章《Transformer 在流式、在线和低延迟场景的局限》。全套书精修 goal 继续 active。
