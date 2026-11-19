# 第 7 章《位置表示：绝对位置、相对位置、RoPE 与 ALiBi》精修审计

审阅日期：2026-11-19。状态：章级研究、章节职责审视、正文重写和配套同步完成；静态验收完成。未运行代码、测试、模型、API、benchmark 或 PDF。目标平台 MathJax 页面渲染、读者试读与第二十一册连续复读留待后续门禁。

## 读者契约与章节职责

- **书册目标贡献：** 解释 Transformer 如何获得顺序线索，并让读者按位置线索进入计算的位置区分绝对位置向量、相对位置表示、query/key 旋转和 attention score 偏置；不把架构公式误当成长上下文能力证明。
- **读者前置：** 已理解 token 表示、self-attention 的 query/key/value 与 softmax，并能阅读向量、点积和基础矩阵记号。第 2—5 章已分别解释 attention、Q/K/V、head 和基本 block。
- **离章结果：** 能解释无位置线索和位置相关 mask 时 self-attention 的置换等变性；区分 Learned Absolute、Sinusoidal、Shaw、T5 relative bias、RoPE 和 ALiBi 的注入位置与计算作用；推导 RoPE 单个二维子空间的相对位移点积；说明位置公式可计算、相对距离可区分和训练范围外任务能力是不同问题。
- **本章边界：** 第 8 章主讲 mask 可见性和注意力模式；RoPE 长上下文扩展方法与评测由第 18 章主讲；NoPE 及因果 mask 的隐式顺序线索由第 64 章主讲。本章只作建立理解所需的短介绍和回链。

## 外部研究与主题去向

2026-11-19 按位置表示、相对位置、RoPE 扩展、NoPE 和课程/教材组织检索并阅读原始论文、官方课程/教材材料。检索词、来源版本、阅读位置、主张支持范围和限制见[位置表示与上下文扩展研究记录](../../research/embedding-and-position-representations-2026.md)。

| 外部主题 | 来源支持的结论与限制 | 本章处理 |
| --- | --- | --- |
| 无位置 self-attention 的顺序缺口 | 在无位置输入、位置偏置及 mask 的条件下，self-attention 对输入行置换呈等变性；因果 mask 可形成隐式顺序线索，特定 NoPE 实验不支持普遍移除显式位置线索 | 7.1 建立置换等变锚点；7.8 限定因果 mask/NoPE 主张，并回链第 8、64 章 |
| Learned absolute 与 Sinusoidal | Vaswani 等人的原论文在输入处加入位置表示，并在其 WMT 配置中比较 learned 与 sinusoidal；结果不代表普遍优劣。固定正弦频率包含固定偏移的线性变换关系，但不保证模型学会或外推 | 7.3 解释位置表、正弦多频率和固定偏移关系，明确公式可算与任务能力的差别 |
| Shaw 相对向量与 T5 bias | Shaw 将相对距离向量用于 key 和 value 路径，报告限于 WMT 2014 配置；T5 将分桶 offset 映射为标量 logit bias，32 bucket 和 128 边界属于 T5 设置 | 7.4 分开讲向量和标量机制；包含 Shaw value-side 项及实验边界 |
| Transformer-XL 相对位置 | 相对打分项与跨 segment 状态复用共同设计，长依赖结果不能单独归因于位置机制 | 7.4 作为归因边界，不展开其记忆机制 |
| RoPE | RoFormer 的旋转矩阵使单个 QK 点积中的位置旋转项依赖位置差；这是局部代数性质，不等于完整网络只依赖相对位置或能可靠处理未见长度 | 7.5 展开二维旋转与点积关系，长上下文细节回链第 18 章 |
| ALiBi | 每个 head 的距离线性惩罚可按任意距离计算；论文长序列结果受模型、语料和评测设置限制，距离可算不等于任务有效 | 7.6 解释 score bias、权重影响及 1,024→2,048 的特定实验边界 |
| PI、YaRN、LongRoPE | 展示针对既有 RoPE checkpoint 的不同位置映射与适配路径；名义窗口结果只适用于各自实验 | 7.7 只举 PI 与 YaRN 作机制入口并链接第 18 章，不横向比较窗口数字 |
| 教材与课程组织 | D2L §10.4 与 Stanford CS224N Winter 2025 Lecture 8 提供具体教学组织样本；不构成学习效果证据 | 用于检查前置与顺序，不声称该顺序普遍提升学习 |

## 正文修订与同步决定

- 将章节从作者工作计划、来源清单、位置方法盘点和工程审计混排，重组为“置换等变 → 位置注入分类 → 绝对位置 → 相对位置 → RoPE → ALiBi → 长度能力边界 → 因果 mask 隐式线索 → 对照与自测”。
- 更正“self-attention 本身没有顺序”这一过宽说法：结论明确要求无位置输入、无位置偏置且无位置相关 mask；并说明因果 mask 是另一种顺序线索。
- 区分 Shaw 的 key/value 相对向量与 T5 的标量 bias；补全 Shaw value-side 相对项。将 Transformer-XL、ALiBi、PI/YaRN 的结果限定于原论文方法和实验条件。
- 补齐 RoPE 的二维旋转推理和同相对距离小例子，并指出推理只覆盖位置旋转对单次 QK 点积的影响。区分可计算位置、位置距离编码和有效长上下文任务能力。
- 移除作者工作流、来源清单式正文、Python 位置审计 demo、任意 `Position Encoding Gate` 及缓存/packed sequence 的重复工程讲解。本章不承诺编码或程序复现，因此不保留代码。
- 同步[目录](../../book-21-transformer-architecture-evolution/目录.md)、[练习 31H—31I](../../EXERCISES.md)、[面试题 31G](../../INTERVIEW_BANK.md)、[论文路线](../../PAPERS.md)、[项目说明](../../PROJECTS.md)、[术语表](../../GLOSSARY_EN_ZH.md)和[知识图谱](../../KNOWLEDGE_GRAPH.md)。目录标题与正文一致；练习改为 RoPE/ALiBi 手算及 position ID 边界分析；清除位置编码 gate 依赖。

## 验收与遗留风险

- 静态验收：H2 `7.1—7.10` 连续；9 个 `math` 围栏成对；行内数学定界符数为偶数；第 8、18、64 章本地链接目标存在；练习题目编号和配套交叉引用已核对；相关改动文件的 `git diff --check` 通过。
- 正文不含代码块或 `Position Encoding Gate`；配套资料不再引用该 gate。公式的变量在使用处定义；RoPE head 维度偶数条件以通常配置口径陈述。
- 引用均直达论文或官方课程/教材页面；来源结果按模型、任务和参数配置限定。没有对论文实验做本地复现，也没有做目标平台实际渲染或目标读者试读。
- 后续仍须执行第二十一册整册连续复读、目标平台 MathJax/图表渲染和全套外链终审；本章静态检查不能代替这些门禁。
- 下一步进入第 8 章《Causal Mask、Prefix LM、Bidirectional Attention 与注意力模式》。第二十一册第 1—7 章完成首轮精修，第 8—95 章待审。
