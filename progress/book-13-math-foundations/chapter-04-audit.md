# 第四章精修审计：优化基础：从梯度估计到可恢复的训练动态

审计日期：2026-11-09  
状态：章级扫描、正文精修、配套同步、局部验收和最终主路径复读均完成；第十三册精修未完成。  
下一步：按目录进入第五章《矩阵分解与低秩》。

## scope

- 完整复读第四章，检查目标函数、导数/链式法则、梯度估计、有效 batch、Momentum、Adam/AdamW、schedule、clipping、Hessian/非凸边界、loss spike 诊断和恢复状态是否形成连续主线。
- 复算参数更新、batch/token reduction、优化器矩状态、权重衰减、离散 schedule、global norm clipping、恢复差异等数例；执行 §4.13 标准库代码并检查正常和边界输入。
- 核对第一至第三章的概率/信息论接口、第十四册 autograd 实现入口及第二册/第五册训练系统分工；不更新 PDF。

## chapter_scan

- **日期与问题：** 2026-11-09。检查初学者是否能从复合损失推到参数梯度；样本平均、token 平均与 rank 局部均值的分母是否明确；Momentum、Adam、AdamW 的状态和更新变化是否能手算；schedule 与 clipping 的位置、曲率解释和失效条件是否清楚；训练异常和 checkpoint 恢复是否能通过最小实验区分。
- **资料范围：** 原始 Adam、AdamW、AMSGrad、梯度裁剪、sharp minima 和大 batch 论文；PyTorch 2.15 autograd、AdamW、AMP clipping、optimizer state loading 与 reproducibility 官方文档。范围覆盖本章公式、API 语义和特定经验主张，不构成所有 optimizer 或训练配方的穷尽扫描。
- **候选主题及去向：**

| 候选 | 来源与适配 | 决定 |
| --- | --- | --- |
| 链式法则和梯度如何从 loss 到参数 | PyTorch 2.15 autograd 文档说明反向模式沿前向操作图应用链式法则；该册负责数学入口，第十四册负责计算图与 API | 纳入一层线性单元的手算，按参数坐标解释梯度形状；完整工程实现链接第十四册第二章 |
| AMSGrad 的理论和实现差异 | Reddi 等人的原始论文 Algorithm 2 维护未校正二阶矩运行最大值并在凸在线优化假设下给出 regret bound；PyTorch 2.15 AdamW 文档先取运行最大值、再执行 Adam 式偏差修正 | 纳入边界说明，区分论文算法、PyTorch 实现和各自理论范围；修正错误地指向 SAM 论文的 arXiv URL；证据写入[优化器来源记录](../../research/adam-adamw-and-learning-rate-schedules-2026.md) |
| 梯度裁剪及 AMP/累积顺序 | Pascanu 等人支持 RNN 梯度范数裁剪的研究出处；PyTorch 2.15 AMP 示例支持先完成累积、unscale 后 clip/step 的 API 用法 | 纳入全局范数可复算例子及 micro-batch 顺序差异；不写成通用阈值建议 |
| 大 batch 线性学习率缩放 | Goyal 等人的实验限于 ResNet-50/ImageNet、batch 8192、线性缩放和 warmup | 作为带边界的候选启发保留，明确不外推为 LLM token batch 的定律 |
| sharpness 与平坦性 | Dinh 等人构造函数保持不变但参数空间 sharpness 可变化的重参数化例子 | 纳入参数化边界，不把局部曲率直接当成泛化结论 |
| checkpoint 恢复语义 | PyTorch 2.15 optimizer state loading 文档说明参数组映射按顺序关联；reproducibility 文档限定跨平台/版本的确定性范围 | 纳入 optimizer 参数映射、状态连续性和容差边界；增加梯度累积中途保存的工程推论并明确为状态一致性推导 |
| 新 optimizer 排名或特定训练 recipe | 不同来源的模型、数据、token 预算和硬件不可直接合并；相关替代路线已有独立研究记录和 `PAPERS.md` 入口 | 不在数学基础章扩成排行榜；保留方法定义和读者入口，详细比较不改变本章边界 |

## 修改结果

- 在梯度入口补充线性单元链式法则手算：从 `h=wx+b`、平方损失推到 `dL/dw`、`dL/db`，执行一次参数更新并回算新损失；补充参数张量与对应梯度形状的关系和第十四册 autograd 链接。
- 为 SGD 增加总体梯度、抽样噪声、方差按 batch 缩放所需假设、序列均值与有效 token 均值、重要性加权的中间计算；区分独立抽样的玩具方差与整次训练误差。
- 增加 Momentum 三步状态轨迹；Adam 一步/两步 bias correction、epsilon 尺度、二阶矩与方差区别及状态存储估算；AdamW 手算耦合 L2 与解耦衰减，并说明 decoupled decay 仍受学习率影响。
- 固定 cosine 离散端点定义，增加 batch scaling 证据范围、token 进度例子、global norm clipping 分段式和 micro-batch clip 次序对照；明确 AMP unscale 顺序及 clipping 只能限制幅度。
- 增加 Hessian 稳定性和驻点例子；扩展 loss numerator/denominator 与 assistant-only mask 算例、rank 级排障、optimizer state 参数映射、恢复差异与中途 gradient accumulation 保存边界。
- 更正 AMSGrad 论文 URL，并区分原论文未偏差修正的最大二阶矩与 PyTorch 2.15 带偏差修正的 `amsgrad` 更新；定理边界收窄至论文凸在线优化假设。
- 清除旧资料日期和自动 HTTP 状态措辞。章节来源列至 2026-11-09；章节摘要已同步简介/目录。

## length_status

按连贯教学链分组，不把每个 H2 机械拆开。计数为解释性正文中的汉字数；排除标题、公式、代码围栏、表格、独立列表、链接文字/URL、内联代码/公式及练习、资料和小结。使用脚本按章节边界累计，教学链独立性另经人工判断。

| 教学单元 | 类型 | 实计汉字 | 门槛 | 状态与分组理由 |
| --- | --- | ---: | ---: | --- |
| §§4.1–4.3 | 核心 | 2,015 | 1,800 | 通过。由优化目标和链式求导推到 mini-batch 估计、token reduction 与分布式分母，概念和更新例子连续依赖。 |
| §§4.4–4.6 | 核心 | 1,821 | 1,800 | 通过。按历史方向、坐标自适应矩和参数衰减构成 SGD/Momentum 到 Adam/AdamW 的更新状态演进。 |
| §§4.7–4.10 | 核心 | 1,903 | 1,800 | 通过。以训练更新稳定性为共同问题，依次讨论时间尺度、幅度控制、曲率解释及凸/非凸结论边界。 |
| §§4.11–4.14 | 核心 | 1,893 | 1,800 | 通过。以 loss spike 为待解释现象，连接根因实验、恢复状态、审计代码和三类故障复盘。 |
| §§4.15–4.17 | 资料/练习/总结 | 不计 | 不适用 | 不承担新的独立机制教学。 |

## example_status

- **链式法则：** 构造 `x=2,w=1,b=0,y=3`，得 `h=2,L=0.5,dL/dw=-2,dL/db=-1`；按 `η=0.1` 更新后 `h=2.5,L=0.125`。
- **梯度下降/多维梯度：** 一维二次损失从 `w=0`、`η=0.1` 得到 `w_1=1`；二维例从 `(0,0)` 更新到 `(0.4,-0.4)`，loss 由 6 降至 3.28。过大学习率例对应增幅振荡。
- **SGD/batch：** `y=(0,2,4)` 的单样本梯度为 `0,-2,-4`、平均梯度 `-2`；总体方差 `8/3`，独立有放回抽 4 次后的均值方差 `2/3`。非均匀采样概率 `(0.6,0.3,0.1)` 下，重要性校正期望回到 `-2`。
- **reduction/并行：** 序列等权均值 `0.4` 对照 token 加权均值 `0.34`；64×128×0.9 约 7,373 有效 token；100 token 的均值 `0.2` 与 50 token 的均值 `0.8` 对照 rank 均值 `0.5` 和全局 token 均值 `0.4`。
- **Momentum/Adam：** Momentum 速度 `-2,-3.6,-1.88` 显示当前梯度反向时状态仍可能更新旧方向；Adam 的 `g=0.2` 第一步校正更新约 `0.0995`，第二步 `g=0.1` 更新约 `0.0926`；对照无 bias correction、epsilon 和历史半衰期。
- **AdamW/显存：** 大小参数的 L2 耦合与 AdamW 一步结果分别约 `9.999/9.998` 和 `0.099/0.09899`；7B BF16 权重约 14 GB，FP32 的 `m/v` 合计约 56 GB，明确不含其他训练状态。
- **schedule/clipping/曲率：** `10` 步、`3` 步 warmup 的余弦 schedule 从 peak `0.001` 到 min `0.0001`；`g=(3,4,12), C=5` 范数由 13 缩到 5；两个 micro-batch 先裁剪再平均得到 `(2.5,2.5)`，先平均再裁剪约 `(3.536,3.536)`；二次函数 `a=4` 的 `η=0.2/0.5/0.6` 展示收敛、等幅震荡和放大。
- **诊断/恢复：** loss numerator `30/100` 对照错误分母 `30/25`；两个 prompt token 与一个 answer token 的 mask 均值 `0.6` 对照全序列均值 `0.4`；Adam 恢复两步更新约 `0.00741` 对照只恢复权重约 `0.01`。这些均为构造输入，不是生产训练观察。

## code_status

- **代码块：** §4.13 标准库优化审计实验。执行语句改动涉及 `math.hypot` global norm、有限范数检查、精确 clipping 分支和至少两个 decay step 的 cosine schedule，因此已重跑完整代码和 stdout 对照。
- **环境与命令：** `.tmp/llm-book-cpu-check/bin/python`，Python 3.12.9 标准库；从 Markdown 提取 Python 围栏并在隔离命名空间执行，与紧邻 `text` 围栏逐字比较。stdout 完全一致，正文 7 个检查项均为 `True`。
- **边界检查：** 10 项全部通过：低于阈值不变、超过阈值按比例裁剪、`C=0` 的零/非零梯度、`[1e308,1e308]` 的有限稳定范数、非有限梯度与溢出范数拒绝、两步 schedule 拒绝、三步 schedule 接受、正常 schedule 两端点正确。
- **未覆盖范围：** 此实验是标准库教学实现，不验证 PyTorch fused/foreach、GPU、分布式通信或生产 optimizer 数值路径。

## formula_status

- 手算复核梯度方向、链式法则、mini-batch 方差、重要性采样、Momentum、Adam bias correction、AdamW decay、schedule 端点、global norm clipping、二次迭代和 checkpoint 更新差异。
- 复核 AMSGrad 论文算法与 PyTorch 2.15 文档写法的区别；将收敛结论限定于原论文声明的凸在线优化假设。
- 全章有 26 个 display-math 围栏块，60 个围栏标记成 30 对；数学块花括号计数平衡。未发现未配对围栏；未使用 Mermaid。由于本地未安装 MathJax 命令，本轮未宣称公式渲染通过。

## visual_status

- 已检查视觉机会：优化指标分类表、loss spike 根因表、公式和逐步数值例子分别表达状态类别与更新机制；本章无 Mermaid 图，现有问题链由公式/表格/手算足以核对，不额外插入装饰图。

## link_status

- 第四章共有 24 个 Markdown 链接，其中 1 个本地链接指向第十四册 autograd 章节且目标存在；其余为论文和官方文档入口。AMSGrad 论文链接从错误的 arXiv `2010.01412` 改为 `1904.09237`，并打开核对原始论文与 PyTorch 2.15 AdamW 公式。
- 本地链接检查没有缺失；本轮未逐个探测所有外链 HTTP 状态，也未验证公开 GitHub 页面中的 MathJax 呈现。

## sync_status

- 更新第十三册[简介](../../book-13-math-foundations/简介.md)第四章摘要和[目录](../../book-13-math-foundations/目录.md)章级问题/能力描述。
- `PAPERS.md` 的 Adam/AdamW 阅读入口增加 AMSGrad 原论文及适用边界；`GLOSSARY_EN_ZH.md` 增加 AMSGrad 术语。
- [优化器来源记录](../../research/adam-adamw-and-learning-rate-schedules-2026.md)新增 AMSGrad 原论文与 PyTorch 2.15 变体事实；[PyTorch API 边界记录](../../research/pytorch-training-api-boundaries-2026.md)新增 2.15 autograd 机制及证据边界。
- `INTERVIEW_BANK.md` 现有 optimizer、chain rule、clipping 和 checkpoint 题目均能覆盖这些基础概念，本轮无需另复制章节答案；未改 `plan.md`、README、PDF。

## final_path_reread

- 对照前三章的章节职责复读第四章入口：前三章分别建立表示、概率与信息论工具，第二、三章也讨论语言模型预测损失及其扩展目标。原开篇“前两章回答了模型在优化什么”容易把交接压缩成概率损失一项，现改为明确交代前三章的工具分工，并将本章的优化对象限定为负对数似然/交叉熵这类预测目标。
- 本轮只调整跨章引导句；公式、代码、数据例子和代码运行契约未变化，原验证记录继续适用。第四章与第八章策略更新、第十四册 autograd 工程入口及第二册/第五册训练流程分工未发现新增冲突。
- 第四章最终路径复读通过；下一步进入第五章，并继续核对其与第一章的低秩入口、第四章的优化取舍和第五册训练显存主讲之间的边界。

## open_risks

1. §4.13 是标准库数值演示，不测混合精度、GPU、分布式或具体生产 optimizer 内核；相关结论保留为官方文档语义或教学推演。
2. 外部来源未做全量 HTTP 探测；工作区未发布，公开 GitHub 页面 MathJax 渲染待形成可访问版本后复核。
3. 本章完成不代表第十三册完成；第五至第十章、全册主路径复读与最终门禁待处理。PDF 不更新。
