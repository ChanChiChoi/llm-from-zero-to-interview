# 第一册第二份正文重构记录：Transformer 核心

日期：2026-10-27  
对象：book-01-core-30/chapters/02-transformer核心.md  
用途：记录本轮结构决策、旧内容去向、来源支持范围和未完成验证；不替代正文中的解释，也不代表代码运行或模型实验。

## 书册契约与章节任务

第一册是书系共同入口，沿“输入—表示—计算—学习—生成—验证”建立文本大模型的主心智模型。第二部分的任务收窄为：读者能从 token ID 出发，解释一次 decoder-only Transformer 前向如何形成下一 token logits，并区分训练目标、因果可见性和生成循环。第二册承接研究和架构深度；第五册承接训练工程；第六册和第二十四册承接 Serving、KV Cache 和推理实现。

本轮保留第 6—15 讲的讲次编号，以便沿用题库和其他导航中的定位；调整讲次分工、章节论证顺序和本册细节深度。第一册简介已经承诺本册建立主线而非百科，现有承诺与重构后章节一致。目录改为四段学习路径，章节文件继续作为第 6—15 讲的正文入口。

## 新的章节关系

| 阶段 | 讲次 | 上游输入 | 阶段产物 | 下一阶段为什么出现 |
| --- | --- | --- | --- | --- |
| 输入表示 | 6—7 | 文本 | token ID 和初始向量 | 向量仍需在位置间混合上下文 |
| 上下文计算 | 8—10 | 带位置的向量 | Attention 输出、多头布局、KV 头账本 | 需要规定哪些前缀可见并定义预测目标 |
| 因果训练与深层结构 | 11—13 | 带因果约束的表示 | 训练/生成协议和可堆叠 Block | 位置机制需在 Q/K 已定义后解释 |
| 位置与综合 | 14—15 | Q/K、完整 Block | RoPE 边界和端到端 miniGPT | 以可追踪的实现闭合输入到 logits 的路径 |

旧稿一篇内重复模板较多，讲次之间有四组结构重叠：7/14 都展开位置机制；9/10 都展开 KV 头变体；12/13 都重复解释 Norm、残差和 Pre-LN；8/9/11 将直觉、公式和因果限制拆散。新稿把每种知识留在一个完整主讲位置，另一处只做承接。

## 旧内容去向

| 旧稿知识 | 新稿位置 | 处理 |
| --- | --- | --- |
| token、词表、ID、word/char/subword、BPE、WordPiece、SentencePiece、字节和特殊符号 | 第 6 讲 | 收束为“离散化机制 + tokenizer 输入契约”；不声称所有 tokenizer 的训练算法相同 |
| embedding lookup、位置编码、shape、输出权重共享 | 第 7 讲 | 保留向量查表、位置机制比较与 shape；RoPE 推导移到 Q/K 已建立后的第 14 讲 |
| attention 直觉、Q/K/V、信息路由、权重解释边界、并行和代价 | 第 8 讲 | 只建立概念与动机，公式统一到第 9 讲 |
| scaled dot-product 公式、shape、sqrt 缩放、手算、mask、复杂度、FlashAttention 与注意力变体 | 第 9 讲 | 保留完整基础计算和因果矩阵；FlashAttention 与改变连接模式的方法只做准确定位和专题分流 |
| MHA、MQA、GQA、多头 shape、KV Cache 成本 | 第 10 讲 | 集中讲多头布局和 KV 头数对缓存载荷的影响，删除第 9 讲重复目录式介绍 |
| causal mask、训练目标右移、teacher forcing、自回归生成、Masked LM 对照 | 第 11 讲 | 把 mask 的数学表示和语言模型训练/推理协议连成一条解释链 |
| Block、Attention/MLP、encoder-decoder 与 decoder-only、Pre-LN 结构 | 第 12 讲 | 解释层的组合与形状守恒；把稳定性论证交给第 13 讲 |
| LayerNorm、RMSNorm、残差、Pre-LN/Post-LN 和手算对比 | 第 13 讲 | 作为单独的“深层可训练性”问题递进，不重讲 Block 结构 |
| RoPE、相对位置性质、长上下文方法、成本与评估 | 第 14 讲 | 保留 RoPE 推导，长上下文压缩成概念区分、方案地图和评估边界；专题研究移交第二册 |
| 字符级 miniGPT、训练数据错位、causal attention、Block、loss 和生成 | 第 15 讲 | 用一个完整脚本贯穿前述 shape；明确它是教学实现，不带 KV Cache 或生产 kernel |
| 多个讲次机械重复的坑、易混淆、边界、练习、回顾小节 | 各讲内融入必要解释，综合检查收敛到第 15 讲和本部分收束 | 删除模板化重复，保留会改变判断或调试顺序的具体知识 |

## 图表设计

正文用 Mermaid 画出全章数据流、MHA 组件、Pre-LN Block 和生成时序；表格承载 tokenization 取舍、位置机制、张量 shape、因果可见矩阵、MHA/GQA/MQA 结构、KV Cache 算例、长上下文方案和故障定位。图均为本章根据已解释机制自行绘制，没有搬用外部图片，因此不需要第三方图片许可标注；外部论文仍在结论附近引用。

## 来源与支持范围

| 一手来源 | 支持的正文判断 | 不支持的外推 |
| --- | --- | --- |
| Vaswani 等，[Attention Is All You Need](https://arxiv.org/abs/1706.03762) | scaled dot-product attention、多头、正弦位置编码、原始 Transformer 组件 | 不证明所有现代 decoder-only 配置与原论文相同 |
| Sennrich 等，[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) | BPE 子词与罕见词处理动机 | 不证明所有当前 tokenizer 使用相同预切分、字节和词表规则 |
| Devlin 等，[BERT](https://arxiv.org/abs/1810.04805) | WordPiece 的代表性模型实例 | 不据此声称 WordPiece 变体都使用同一个训练细节 |
| Kudo 与 Richardson，[SentencePiece](https://arxiv.org/abs/1808.06226) | 原始文本子词建模工具及支持模型类型 | 工具名不能当作单一 tokenizer 算法名 |
| Su 等，[RoFormer](https://arxiv.org/abs/2104.09864) | 对 Q/K 旋转与相对位置性质 | 不保证超出训练长度后任意外推 |
| Press 等，[Train Short, Test Long](https://arxiv.org/abs/2108.12409) | ALiBi 在 attention score 中加入距离相关的线性 bias | 不外推论文设置之外的长度泛化收益 |
| Dao 等，[FlashAttention](https://arxiv.org/abs/2205.14135) | IO-aware 分块精确 attention 与中间读写优化 | 不保证所有硬件、shape 和 dtype 下都有同一性能收益 |
| Ainslie 等，[GQA](https://arxiv.org/abs/2305.13245) | Query 头与 KV 头分组关系及论文评测配置中的取舍 | 不外推所有模型质量、吞吐与缓存的实际收益 |
| Zhang 与 Sennrich，[RMSNorm](https://arxiv.org/abs/1910.07467) | RMSNorm 公式和论文中的实验 | 不证明其对所有模型都优于 LayerNorm |
| Xiong 等，[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) | 研究设置下 Pre-LN/Post-LN 梯度路径分析 | 不将其外推为架构稳定性的单因素结论 |
| Liu 等，[Lost in the Middle](https://arxiv.org/abs/2307.03172) | 指定长上下文任务中证据位置效应的研究实例 | 不声称所有模型和任务都呈现相同位置曲线 |

上述来源详情和 2026-10-07 的链接核验记录沿用 research/transformer-core-review-2026-10-07.md。本轮重写没有再次下载论文全文或逐项重查全部外链，因此将论文结论限制在来源实际支持的机制与实验设置内。

## 公式与教学算例

- Attention 手算沿用定义直接计算：q=(1,0)，K=[(1,0),(0,1)]，V=[(2,0),(0,4)]，d_h=2；scaled scores 约为 [0.7071,0]，softmax 权重约为 [0.6698,0.3302]，输出约为 (1.3396,1.3208)。
- KV Cache 算例采用 L=32、T=8192、d_h=128、每元素 b=2 字节：MHA 32 KV 头为 4 GiB，GQA 8 KV 头为 1 GiB，MQA 1 KV 头为 128 MiB。结果只表示单序列原始载荷，不含框架和内存管理开销。
- LayerNorm/RMSNorm 算例使用 [1,2,3,10]。LayerNorm 均值为 4、标准差约 3.5355；RMSNorm 分母约 5.3385。均为按公式手算的数值例，不是外部运行结果。
- RoPE 算例使用二维单位向量与每步 0.25 弧度，点积项为 cos((n-m)×0.25)，只验证相对位移性质。

## 状态与未验证范围

- code_status：miniGPT 是完整教学脚本，正文说明 Python 版本参考仓库锁定的 Python 3.13+ 与 PyTorch 2.12.0。本轮没有运行脚本或新增测试，故没有 local-demo/verified 代码状态；代码输出只写可静态推导的 shape 和行为。
- formula_status：检查 math 围栏、行内公式定界符、符号定义与关键手算结果；未使用外部 LaTeX/MathJax 渲染器。
- link_status：本轮静态确认所有相对 Markdown 链接目标存在；外链保留原始论文链接，本轮未再次访问检查 HTTP 状态。
- sync_status：第一册目录已改为四阶段路径，简介承诺与本章任务一致。题库、练习、百科和术语表已有 Transformer 主题，本轮不复制章节摘要；进阶注意力、长上下文和缓存以准确跨册入口承接。
- open_risks：未执行 miniGPT 代码；未构建 PDF 或用目标 Markdown 渲染器复核图、表与公式；未完成第一册改写后的连续整册复读；未对本轮引用链接做外网连通检查。代码静态检查不能替代运行验证。
- status：本轮完成本章整体重写、目录同步、旧知识去向映射、来源边界记录和静态结构检查。状态只表示章节重构完成，不表示全册架构精修或教学代码运行验证完成。
