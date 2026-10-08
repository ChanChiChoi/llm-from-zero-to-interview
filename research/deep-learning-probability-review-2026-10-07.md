# 第二册第一份正文：深度学习与概率基础进阶资料核验

核验日期：2026-10-07。用途：精修 `book-02-advanced-100/chapters/01-深度学习与概率基础进阶.md`，并同步其中直接依赖的论文索引与百科条目。论文结论只用于其公开方法和实验范围，不外推成所有 LLM 或生产系统的普遍结论。

## 关键来源与证据范围

| 来源 | 等级 | 支持的内容 | 证据边界 |
| --- | --- | --- | --- |
| [Scheduled Sampling](https://arxiv.org/abs/1506.03099) | `paper` | 训练时混用真实 token 和模型生成 token 的早期序列预测方法 | RNN 任务中的方法方案，不证明其是通用的 exposure bias 修复 |
| [Huszár 对 Scheduled Sampling 的分析](https://arxiv.org/abs/1511.05101) | `paper` | 该方法训练目标不一致的理论批评 | 是对特定目标的分析，不证明所有多步训练方法都无效 |
| [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) | `paper` | 教师软目标和温度蒸馏的早期入口 | 不保证任意学生、教师和数据组合都有收益 |
| [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) | `paper` | 图像分类设置中的 label smoothing 例子 | 不能直接推导其适用于语言建模、精确复制或结构化输出 |
| [Sharp Minima Can Generalize For Deep Nets](https://arxiv.org/abs/1703.04933) | `paper` | 参数重缩放可能改变 sharpness 度量而保持网络函数的讨论 | 不据此否定所有平坦度分析；指标须绑定参数化和扰动定义 |
| [Reconciling Modern Machine-Learning Practice and the Classical Bias–Variance Trade-Off](https://arxiv.org/abs/1812.11118) 与 [Deep Double Descent](https://arxiv.org/abs/1912.02292) | `paper` | 双下降及复杂度、训练时间、数据规模变化的实验现象 | 经验曲线受模型族、噪声、优化与评估协议限制，不是参数量单变量定律 |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | `paper` | 指定模型、数据和计算范围内的语言模型交叉熵缩放关系 | 经验拟合，不为所有架构和下游能力提供外推保证 |
| [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | `paper` | Chinchilla 训练设置下参数量与训练 token 的预算权衡 | 最优比例不脱离数据质量、序列设置、训练效率和目标指标 |
| [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977) | `paper` | aleatoric 与 epistemic uncertainty 的区分 | 论文实验是计算机视觉；正文把该区分作为分析框架，不宣称完整 LLM 后验推断 |
| [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) | `paper` | 分类校准与 ECE 估计的使用 | ECE 依赖置信度定义、分桶和数据分布，不等同于开放式回答的真实性概率 |
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | `paper` | RAG 将检索到的非参数化记忆与参数化模型结合 | 论文方法不保证任意检索结果相关、正确或被生成忠实使用 |
| [Understanding the Difficulty of Training Deep Feedforward Neural Networks](https://proceedings.mlr.press/v9/glorot10a.html) | `paper` | 深层前馈网络初始化和信号尺度背景 | Xavier/Glorot 尺度依赖初始化分析中的假设；不等于现代 Transformer 完整初始化配方 |
| [On the Difficulty of Training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063) | `paper` | RNN 梯度消失/爆炸及范数裁剪背景 | 对 Transformer 只作为梯度问题的历史方法入口，不据此声称具体 Transformer kernel 行为 |
| [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913) | `paper` | 特定 Transformer 语言模型中的 FFN key–value memory 机制视角 | 是有用的机制解释，不表示某个神经元唯一存放一条事实 |

本轮通过 arXiv Atom API 获取 24 个章节参考文献 ID 的元数据；额外的 Huszár 条目以单独 API 查询核对，Dinh 与 Lewis 页面返回了对应标题和摘要。PMLR 页面、Shannon PDF 和 *Deep Learning* 教材站均可访问。arXiv 页面经本地代理的 HEAD 请求有间歇超时，因此不把该代理状态记作链接失效；链接存在性以对应 arXiv 元数据记录和页面内容为证。

## 公式和数值核验

- 偏差—方差分解限定为固定输入回归：`y=f*(x)+ε`、`E[ε|x]=0`，且噪声与训练集抽样独立；分解项分别为估计偏差平方、训练集抽样方差和条件噪声方差。
- Scaling law 只作为可分离项的教学示意，定义 `L`、`N`、`D` 和拟合常数，并说明计算预算约束 `(N,D)`；没有把 `N`、`D`、`C` 写成彼此独立的三个幂律加项。
- MAP 与 L2 项绑定零均值、各向同性高斯先验精度 `λ`，并明确负对数似然未按样本数取平均。
- 温度 `τ>0`、label smoothing `ε∈[0,1]`、ECE 的桶和样本定义、梯度裁剪阈值 `c≥0` 均已检查。
- 文件包含 87 个 `math` 围栏；花括号和 `begin/end` 环境配对检查通过。未使用外部 Markdown 数学渲染器。

## 本地教学代码验证

环境：系统 Python 3.12.9；全部示例只依赖标准库。10 个 Python 围栏逐块编译并独立运行，输出与正文一致：温度 softmax 与平均 NLL、熵/交叉熵/KL、二维优化轨迹、构造的双下降曲线、Beta–Binomial 后验、ECE、代理指标反转、梯度裁剪、标量门控和 weight-decay 分组。

另通过 24 项断言，覆盖非法温度和缩放溢出、目标越界及概率未归一化、极小正概率下 KL 仍保持有限、`P>0,Q=0`、分布长度不匹配、优化器非法步数/学习率/坐标、梯度溢出、Beta 参数与计数边界、ECE 置信度/标签边界、极大有限向量的稳定范数裁剪、零向量、非法阈值、sigmoid 极值与非有限输入，以及冻结参数分组。

这些结果只验证本地教学代码及输入，不代表论文复现、真实模型训练、GPU 数值行为或生产系统验证。人为构造的双下降曲线与指标反转示例不构成实验结论。

## 配套同步与遗留边界

- `PAPERS.md` 增补 Scheduled Sampling 与目标批评、蒸馏与 label smoothing、双下降、不确定性与校准、初始化与梯度问题论文入口，并将 FFN key–value memory 放到架构方向。
- `book-04-llm-encyclopedia/chapters/14-重要论文索引.md` 的 Scaling Laws 条目改为带定义域的 `(N,D)` 教学近似，补充计算预算约束并引用 Kaplan 与 Hoffmann；`book-04-llm-encyclopedia/chapters/01-数学与机器学习.md` 补充偏差—方差分解假设，并把 label smoothing 结论限定在任务和标签条件内。
- 术语表已有 Bias-Variance Trade-off、Scaling Law、Aleatoric/Epistemic Uncertainty、ECE、梯度消失/爆炸和 Knowledge Distillation 条目；`EXERCISES.md` 已有偏差—方差、MAP、Beta 更新、熵、ECE、不确定性与 scaling law 练习，本轮没有复制题目。
- 章节在第二册目录中列为第 1—10 讲，文件和相对链接存在；本章没有新增其他简介或目录条目。
- 代理对 arXiv HEAD 的间歇超时仍是访问环境限制。没有外部数学渲染和真实模型/GPU实验；这些限制不改变已核验的论文元数据与本地示例范围。
- `python3 scripts/build_pdfs.py --dry-run` 因当前环境缺少 pandoc 返回失败，因此本轮没有检查 PDF 构建命令或生成 PDF。
