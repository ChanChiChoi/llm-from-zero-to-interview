# 第十章：Data Attribution 与 Valuation

前面几章一直在回答“数据怎么采、怎么洗、怎么配”。本章讨论一个更难的问题：一条数据、一个数据源、一个数据池到底贡献了多少？

这就是 Data Attribution 与 Data Valuation。

Data Attribution 关注归因：模型某个行为、某个错误、某个能力提升，可能来自哪些训练数据。Data Valuation 关注估值：某条数据、某类数据、某个来源对目标指标有多少价值，值不值得保留、上采样、购买、标注或继续扩充。

在大模型场景下，这两个问题都非常难。因为训练数据巨大、训练成本高、模型非凸、数据之间存在强交互，几乎不可能精确回答“这条数据贡献了多少”。但工程上仍然需要近似方法，否则数据决策只能靠直觉。

本章重点：数据贡献估计、influence functions、数据选择、主动学习、数据价值评估。

合规边界：本章讨论数据价值评估、错误分析、训练数据治理和审计，不提供训练数据反推、隐私抽取或绕过数据保护的方法。

## 0. 本讲资料边界与第二轮精修口径

按照 `WRITING_PLAN.md` 的要求，本讲精修前核对了 influence functions、Data Shapley、高效 Shapley 近似、TracIn、Dataset Cartography、DSIR、LESS 和 DataInf 等公开论文资料。

本讲聚焦大模型数据工程中可落地的数据归因和估值：源级 / 簇级 / 样本级近似、目标依赖效用函数、influence 近似、Shapley 近似、小模型 proxy、消融实验、数据选择、主动标注优先级、负价值数据识别和版本化审计。

```text
目标指标 -> 数据单元 -> 弱信号估值 -> 小规模验证 -> 风险成本修正 -> 数据选择 -> 版本审计
```

本讲不把任何估值方法写成“大模型训练数据价值的精确答案”。大模型数据价值通常只能通过多种弱证据近似：小模型实验、源级 ablation、梯度相似、相似检索、人工审计、下游评测和成本风险分析共同支撑。

---

## 1. 先建立直觉：为什么需要数据归因和估值？

假设你训练了一个模型，发现它在数学题上提升了 5 分，在法律问答上下降了 3 分，同时安全拒答变得过度保守。你会问：

1. 哪些数据让数学变好了？
2. 哪些数据导致法律能力下降？
3. 哪些安全数据让模型过度拒答？
4. 某个高价购买的数据源到底值不值？
5. 如果只能再标注 10 万条数据，应该标哪类？
6. 如果要删掉 20% 低价值数据，删哪里损失最小？

没有 attribution 和 valuation，团队只能靠经验拍脑袋。数据工程就会停留在“收更多、洗更干净、试试看”的阶段。

有了归因和估值，团队可以更科学地做数据选择、数据采购、质量修复、配比优化和问题定位。

---

## 2. 来龙去脉：从解释单点预测到评估数据资产

数据归因最早不是为大模型设计的。传统机器学习里，人们关心某个训练样本对某个预测有什么影响。Influence functions 是一个代表性方向。Koh 和 Liang 的工作把经典稳健统计中的 influence functions 用于解释黑盒模型预测，尝试追溯某个预测背后最有影响的训练点。

数据估值则和合作博弈里的 Shapley value 有关。Data Shapley 提出用 Shapley 思想衡量每个训练样本对模型性能的边际贡献。Jia 等工作进一步研究了基于 Shapley value 的高效数据估值近似。

这些方法在中小规模监督学习中更容易解释。但到了大模型时代，问题规模变了：

1. 数据从百万级变成万亿 token。
2. 训练一次模型成本极高。
3. 单条样本影响很小，数据源和数据分布影响更大。
4. 训练过程复杂，包含预训练、继续预训练、SFT、偏好训练和安全训练。
5. 模型能力来自数据交互，不是简单线性叠加。

因此，大模型数据估值更常用数据源级、数据簇级、任务级和配比级近似，而不是逐条精确估值。

---

## 3. Data Attribution 和 Data Valuation 的区别

Data Attribution 问的是“某个结果来自哪里”。

例如：

1. 模型为什么会输出某个错误事实？
2. 某个 benchmark 提升主要来自哪类训练数据？
3. 某个安全误拒是否来自某批安全数据？
4. 模型记住某段文本，训练集中哪些样本最相关？

Data Valuation 问的是“某个数据有多值钱”。

例如：

1. 这个数据源是否提升目标指标？
2. 这个领域数据是否值得继续采购？
3. 哪些样本最值得人工标注？
4. 哪些数据可以删掉或降权？
5. 哪些低质量数据对模型有负贡献？

简单说：attribution 更偏解释和诊断，valuation 更偏决策和资源分配。

---

## 4. 为什么大模型数据价值很难精确计算？

大模型数据价值难在几个方面。

第一，成本太高。最朴素的方法是删掉某个数据源重新训练，再看效果变化。但训练大模型成本巨大，不可能对每个数据源反复重训。

第二，数据有交互。代码数据可能增强结构化推理，数学数据可能增强验证，二者一起效果大于单独相加。单独估值会忽略交互。

第三，指标多目标。某个数据源可能提升代码能力，但降低多语言能力；提升安全性，但增加误拒。它到底“值钱”还是“不值钱”，取决于目标函数。

第四，价值随模型阶段变化。预训练阶段低价值的数据，在 SFT 阶段可能很有价值；小模型有效的数据，大模型未必同样有效。

第五，数据质量和配比耦合。一个数据源本身好，但比例过高会过拟合；一个数据源噪声大，但少量保留可增加多样性。

所以，大模型中的数据价值不是静态属性，而是相对于模型、阶段、目标、配比和评估集定义的。

### 4.1 关键公式与估值指标

把训练数据写成样本集合：

```math
D=\{z_i\}_{i=1}^{n}
```

其中 `z_i` 可以是一条预训练文档、一条 SFT 样本、一对 chosen / rejected 偏好样本，也可以是一个数据源 `C_k` 中的样本。大模型数据工程里更常见的估值单元不是单条样本，而是数据源、数据簇、任务池或数据版本：

```math
C_k=\{z_i: c_i=k\}
```

其中 `c_i` 表示样本所属来源、领域、任务或标注批次。

估值必须先定义目标效用函数。一个常见写法是：

```math
U(S)=\sum_{m=1}^{M} w_m M_m(S)-\lambda_R R(S)-\lambda_C C(S)
```

其中 `S` 是被选中的数据集合，`M_m(S)` 是第 `m` 个目标指标，例如数学、代码、安全、通用能力或人工质量，`w_m` 是业务权重，`R(S)` 是隐私、版权、安全和污染风险，`C(S)` 是采购、清洗、标注、训练和维护成本。

数据源级 ablation 的加入价值可以写成：

```math
\Delta_k^{\mathrm{add}}=U(S_{\mathrm{base}}\cup C_k)-U(S_{\mathrm{base}})
```

删除价值可以写成：

```math
\Delta_k^{\mathrm{drop}}=U(S_{\mathrm{base}})-U(S_{\mathrm{base}}\setminus C_k)
```

如果 `Delta_add` 为正，说明加入数据源有收益；如果 `Delta_drop` 为正，说明删掉它会损失收益。两者都要结合风险和成本解释。

Influence functions 关心训练点 `z_i` 对测试点 `z_*` 的 loss 影响。经典近似形式是：

```math
I_{\mathrm{up}}(z_i,z_*)=-\nabla_\theta \ell(z_*,\hat{\theta})^\top H_{\hat{\theta}}^{-1}\nabla_\theta \ell(z_i,\hat{\theta})
```

其中 `ell` 是 loss，`theta_hat` 是训练后的参数，`H_theta_hat` 是经验风险 Hessian。这个公式表达的是“如果稍微上调训练点 `z_i` 的权重，测试点 `z_*` 的 loss 如何变化”。在大模型中直接求 `H^{-1}` 通常不可行，因此常用梯度相似、TracIn 式训练轨迹或小模型 proxy 做近似。

梯度相似 proxy 可以写成：

```math
A_i=\frac{g_i^\top g_T}{\|g_i\|_2\|g_T\|_2}
```

其中 `g_i` 是训练样本或数据源的梯度特征，`g_T` 是目标任务、目标验证集或错误样本的梯度特征。`A_i` 越高，说明方向越接近，但它仍然不是严格因果证明。

Data Shapley 把每条数据看成参与者，价值为平均边际贡献：

```math
\phi_i=\sum_{S\subseteq D\setminus \{z_i\}}\frac{|S|!(n-|S|-1)!}{n!}\left[U(S\cup \{z_i\})-U(S)\right]
```

这个公式理论性质好，但精确计算需要枚举大量子集。大模型里通常只在小数据池上做近似，或者把 Shapley 思想用于源级、簇级和任务级估值。

最终数据选择可以写成一个带约束的预算问题：

```math
\max_{s_i\in\{0,1\}}\sum_i s_i v_i
```

```math
\sum_i s_i b_i\le B,\quad R(\{z_i:s_i=1\})\le \tau_R
```

其中 `v_i` 是估计数据价值，`b_i` 是 token、标注或训练成本，`B` 是预算，`tau_R` 是风险上限。工程上还会加语言、领域、任务、来源和多样性约束。

---

## 5. 最朴素的方法：数据消融

数据消融是最直接的数据价值评估方法。

做法是：

1. 设定 baseline 数据配比。
2. 删除或降低某个数据源。
3. 训练模型或小模型。
4. 比较 validation loss、benchmark、安全指标和人工评测。

例如，想知道代码数据是否有价值，可以训练两个小模型：一个包含代码，一个不包含代码。再比较代码 benchmark、通用问答、多语言和安全表现。

优点：

1. 直观。
2. 解释性强。
3. 和最终训练目标一致。

缺点：

1. 成本高。
2. 只能评估少量候选。
3. 小模型结论不一定外推到大模型。
4. 删除一个数据源会改变整体配比，混入其他变量。

数据消融是最可靠但最贵的方法，通常用于关键数据源和最终决策。

---

## 6. 数据源级估值

大模型中比逐条样本更常用的是数据源级估值。

数据源可以是：

1. 某个网站集合。
2. 某类书籍。
3. 某个代码数据池。
4. 某个数学题库。
5. 某个多语言语料。
6. 某个合成数据生成版本。
7. 某批偏好标注数据。

评估方法包括：

1. 源级 ablation。
2. 源级上采样/下采样实验。
3. 源级 validation loss。
4. 下游任务关联分析。
5. 质量分与收益的相关性分析。
6. 训练曲线和梯度信号观察。

数据源级估值适合回答：“这个数据池值不值得继续投入？”

---

## 7. 样本级估值

样本级估值关注单条数据的价值。它更细，但更难。

可能用途包括：

1. 找出有害样本。
2. 找出高价值标注样本。
3. 找出错误标签。
4. 选择主动学习样本。
5. 清理低质量合成数据。
6. 解释某个模型错误。

在大模型预训练中，逐条样本估值通常不可行，因为样本太多、影响太小、训练不可重复。但在 SFT、偏好训练、安全数据、领域数据等较小数据池中，样本级估值更有现实意义。

例如，DPO 数据里某些 chosen/rejected 标注可能反了；安全数据里某些拒答样本导致误拒；领域 QA 中某些专家答案过期。样本级估值可以帮助定位这些问题。

---

## 8. Influence functions：从预测追溯训练点

Influence functions 的直觉是：如果把某个训练样本权重稍微增加或删除，模型在某个测试点上的 loss 会怎样变化？

它可以用于回答：哪些训练样本最影响这个预测？

优点：

1. 可解释性强。
2. 适合错误诊断。
3. 可用于发现数据错误。
4. 不一定需要完整重训每个样本。

局限：

1. 依赖近似。
2. 对深度非凸模型不稳定。
3. 计算 Hessian-vector 相关量成本高。
4. 大模型规模下很难直接使用。
5. 训练过程中的优化路径影响难以完全捕捉。

所以在大模型里，influence 思想更常被用于小模型、embedding 近邻、数据簇分析、SFT 数据诊断，而不是直接对万亿 token 预训练做精确 influence 计算。

---

## 9. Shapley value 和 Data Shapley

Shapley value 来自合作博弈，思想是公平分配多个参与者对整体收益的贡献。用于数据估值时，每条数据被看作参与者，模型性能是整体收益。

Data Shapley 的直觉是：一条数据的价值等于它在不同数据子集里加入后带来的平均边际提升。

优点：

1. 理论性质好。
2. 能考虑数据之间的交互。
3. 可用于发现高价值、低价值和有害数据。
4. 可指导数据采购和数据清洗。

缺点：

1. 精确计算极其昂贵。
2. 需要大量训练或近似训练。
3. 在大模型预训练规模下不可直接使用。
4. 结果依赖目标评估集。
5. 样本价值不是通用常数。

因此，Shapley 更适合作为估值思想和小规模数据池工具，而不是大模型全量 token 估值的直接方案。

---

## 10. 近似方法：从精确估值到可用信号

工程上常见近似包括：

1. 小模型 proxy：用小模型或短训练评估数据价值。
2. 数据源 ablation：删除或降权整个数据源。
3. 子集采样：随机采样数据子集训练多个模型。
4. validation loss 分桶：看不同数据源对验证集 loss 的影响。
5. embedding 相似度：找和目标任务最相近的数据。
6. gradient similarity：看训练样本梯度是否与目标任务梯度方向一致。
7. nearest neighbor attribution：用检索找最相似训练样本。
8. LLM judge 质量评分：对 SFT、偏好、合成数据做辅助估值。
9. 人工审计：抽样检查高低分数据。

这些方法没有一个完美。成熟系统通常把多个弱信号组合起来，而不是迷信某一个分数。

---

## 11. 数据选择

数据估值最终要服务数据选择。

数据选择的目标包括：

1. 在固定 token budget 下选最有价值数据。
2. 删除低质量或负贡献数据。
3. 上采样目标能力相关数据。
4. 降低重复和污染。
5. 控制安全和隐私风险。
6. 平衡通用能力和专项能力。

常见数据选择策略：

1. 质量分阈值。
2. 来源信誉。
3. 与目标任务 embedding 相似。
4. 多样性采样。
5. 难例优先。
6. 低 loss 或中等 loss 筛选。
7. 人工审核优先级。
8. 小模型收益验证。

注意，数据选择不是只选“最像评测集”的数据。那样容易过拟合 benchmark，损害泛化和多样性。

---

## 12. 主动学习和标注优先级

主动学习关注：在标注预算有限时，哪些样本最值得标注？

在大模型数据工程中，主动学习常用于：

1. SFT 数据标注。
2. 偏好数据标注。
3. 安全边界样本标注。
4. 专业领域专家标注。
5. 多模态 caption 或 QA 标注。

常见选择信号包括：

1. 模型不确定性高。
2. 多个模型分歧大。
3. 用户频率高。
4. 风险等级高。
5. 目标能力薄弱。
6. 代表性强。
7. 与已有数据差异大。

主动学习的核心不是找最难样本，而是在价值、覆盖、风险和标注成本之间做权衡。

---

## 13. 负价值数据

不是所有数据都有正贡献。有些数据会伤害模型。

负价值数据包括：

1. 错误标注。
2. 低质量合成数据。
3. 重复模板。
4. 过时专业知识。
5. 含隐私或敏感信息的数据。
6. benchmark 泄漏数据。
7. 不安全或不合规回答。
8. 与目标产品风格冲突的数据。
9. 引导模型过度拒答的数据。

数据估值的一个重要用途就是发现负价值数据。删掉坏数据有时比增加好数据更有效。

---

## 14. Attribution 在错误分析中的用途

当模型出现错误时，data attribution 可以帮助定位原因。

例如：

1. 模型输出错误医学知识，追溯到过时网页或低质量论坛。
2. 模型在某语言上表现差，发现低资源语言数据被质量过滤误删。
3. 模型安全误拒，发现某类安全数据过度保守。
4. 模型代码生成过时 API，发现训练数据中旧版本文档占比过高。
5. 模型在某 benchmark 异常高，发现训练集中存在题目泄漏。

这里的 attribution 不一定是精确数学归因，更多是证据链：相似训练样本、数据源统计、版本变更、消融实验和人工审计共同支持结论。

---

## 15. 数据估值和 Data Mixture 的关系

Data valuation 直接服务 data mixture。

如果某个数据源对目标任务收益高，可以上采样或扩充。如果某个数据源对安全有副作用，可以隔离或降权。如果某个合成数据版本提升 benchmark 但降低人工质量，需要重新生成或调低比例。

但估值不能只看单项指标。

例如：

1. 代码数据提升 HumanEval，但可能影响自然语言风格。
2. 安全数据降低漏拒，但可能增加误拒。
3. 多语言数据提升目标语言，但可能占用英语和代码 token budget。
4. 合成数据提升格式遵循，但可能造成模板化。

所以数据估值要输出多维价值向量，而不是单一分数。

---

## 16. 数据估值的工程指标

一个数据源的工程价值可以从多个维度衡量：

1. 目标能力收益。
2. 通用能力副作用。
3. 安全风险。
4. 隐私风险。
5. 版权和授权成本。
6. 清洗成本。
7. 标注成本。
8. 去重后有效 token 数。
9. 多样性贡献。
10. 时效性。
11. 可追溯性。
12. 对产品场景的覆盖度。

真实项目里，数据价值不是学术意义上的准确率提升，而是收益、成本、风险和可治理性的组合。

---

## 17. 大模型中的 practical attribution

在大模型里，一个更实用的 attribution pipeline 可能是：

1. 对问题样本做 embedding 检索，找相似训练样本和数据源。
2. 检查数据版本和来源统计。
3. 分析相关数据源在训练前后的配比变化。
4. 做小规模 ablation 或 downweight 实验。
5. 观察目标评估集和人工样例变化。
6. 对可疑数据做人审。
7. 将结论转成清洗规则、配比调整或标注任务。

这比追求单条样本的精确因果归因更现实。

---

## 18. 面向专家：数据价值是目标依赖的边际贡献

从专家视角看，数据价值不是数据本身的固定属性，而是目标依赖的边际贡献。

同一条数据在不同情况下价值不同：

1. 在缺少数学数据时，一条高质量数学题很有价值。
2. 在数学数据已经过量时，它的边际价值下降。
3. 对通用助手有价值的数据，对法律模型未必有价值。
4. 对小模型有价值的数据，对大模型可能只是重复。
5. 对训练有价值的数据，对评估可信度可能有污染风险。

因此，讨论数据价值必须同时说明：目标模型、训练阶段、评估指标、已有数据分布和风险约束。

---

## 19. 一个可落地的数据归因和估值方案

如果面试官问：“如何评估一个数据源对大模型是否有价值？”可以按下面回答。

第一步，明确目标。是提升代码、数学、多语言、安全、事实性，还是降低幻觉和误拒。

第二步，建立数据元信息。记录来源、语言、领域、质量分、去重簇、license、token 数、时间和版本。

第三步，做离线质量评估。看噪声、重复、污染、隐私、版权和人工抽样质量。

第四步，做小规模训练实验。比较加入、删除、上采样、下采样该数据源后的多维指标变化。

第五步，做目标任务相关性分析。用 embedding、关键词、领域分类、validation loss 和人工样例分析它覆盖哪些能力。

第六步，做副作用评估。看通用能力、安全、误拒、幻觉、多语言和风格是否变差。

第七步，估算成本和风险。包括采购、清洗、标注、授权、隐私和维护成本。

第八步，形成决策。保留、扩充、上采样、降权、隔离、重洗或删除。

第九步，版本化记录。保存实验配置、数据版本、评估结果和决策理由。

### 19.1 最小可运行数据归因与估值 demo

下面这个 demo 不依赖外部库，也不读写文件。输入是一组 toy 数据源，输出包括源级价值排名、目标任务 attribution proxy、token budget 下的数据选择、小规模 Shapley 估值、负价值 / 阻断数据和污染阻断清单。

它演示的是数据估值工程闭环，不是真实 influence function、生产级 Shapley、完整小模型训练或大规模数据选择系统。真实系统需要接入训练日志、数据版本、评估矩阵、embedding / gradient 特征、消融实验、人工审计和合规风险系统。

```python
from itertools import permutations
from math import sqrt


weights = {"math": 0.35, "code": 0.25, "safety": 0.20, "general": 0.20}
target_grad = [0.70, 0.55, 0.20, 0.10]

sources = [
    {"id": "math_verified", "tokens": 900, "quality": 0.92, "coverage": 0.78, "risk": 0.03, "cost": 0.22, "license_ok": True, "privacy": False, "contam": False, "grad": [0.82, 0.28, 0.12, 0.06], "delta": {"math": 0.080, "code": 0.010, "safety": 0.000, "general": 0.015}},
    {"id": "code_tests", "tokens": 760, "quality": 0.88, "coverage": 0.71, "risk": 0.04, "cost": 0.18, "license_ok": True, "privacy": False, "contam": False, "grad": [0.34, 0.86, 0.10, 0.04], "delta": {"math": 0.010, "code": 0.070, "safety": 0.000, "general": 0.010}},
    {"id": "safety_boundary", "tokens": 640, "quality": 0.84, "coverage": 0.66, "risk": 0.05, "cost": 0.15, "license_ok": True, "privacy": False, "contam": False, "grad": [0.18, 0.12, 0.88, 0.18], "delta": {"math": -0.005, "code": 0.000, "safety": 0.060, "general": -0.005}},
    {"id": "zh_domain", "tokens": 820, "quality": 0.80, "coverage": 0.73, "risk": 0.06, "cost": 0.12, "license_ok": True, "privacy": False, "contam": False, "grad": [0.40, 0.20, 0.15, 0.80], "delta": {"math": 0.000, "code": 0.000, "safety": 0.005, "general": 0.050}},
    {"id": "synthetic_template", "tokens": 700, "quality": 0.50, "coverage": 0.42, "risk": 0.22, "cost": 0.05, "license_ok": True, "privacy": False, "contam": False, "grad": [0.20, 0.18, 0.10, 0.15], "delta": {"math": 0.020, "code": 0.010, "safety": -0.020, "general": -0.030}},
    {"id": "old_legal_forum", "tokens": 680, "quality": 0.46, "coverage": 0.35, "risk": 0.36, "cost": 0.08, "license_ok": True, "privacy": False, "contam": False, "grad": [0.08, 0.04, 0.20, 0.34], "delta": {"math": 0.000, "code": 0.000, "safety": -0.030, "general": -0.020}},
    {"id": "benchmark_leak", "tokens": 520, "quality": 0.90, "coverage": 0.40, "risk": 0.70, "cost": 0.04, "license_ok": True, "privacy": False, "contam": True, "grad": [0.75, 0.25, 0.05, 0.05], "delta": {"math": 0.090, "code": 0.000, "safety": 0.000, "general": 0.000}},
]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return sqrt(dot(a, a))


def cosine(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def weighted_delta(src):
    return sum(weights[k] * src["delta"].get(k, 0.0) for k in weights)


def source_value(src):
    if not src["license_ok"] or src["privacy"] or src["contam"]:
        return -1.0
    raw = (
        weighted_delta(src)
        + 0.08 * src["quality"]
        + 0.05 * src["coverage"]
        + 0.05 * cosine(src["grad"], target_grad)
        - 0.12 * src["risk"]
        - 0.04 * src["cost"]
    )
    if src["risk"] > 0.30 or src["quality"] < 0.48:
        return -abs(raw)
    return raw


rows = []
for src in sources:
    rows.append({
        "id": src["id"],
        "weighted_delta": round(weighted_delta(src), 4),
        "grad_sim": round(cosine(src["grad"], target_grad), 3),
        "value": round(source_value(src), 4),
    })

ranked = sorted(rows, key=lambda x: x["value"], reverse=True)
negative = [row["id"] for row in ranked if row["value"] < 0]

budget = 2600
selected, used_tokens = [], 0
for row in sorted(rows, key=lambda r: r["value"] / next(s["tokens"] for s in sources if s["id"] == r["id"]), reverse=True):
    src = next(s for s in sources if s["id"] == row["id"])
    if row["value"] <= 0 or used_tokens + src["tokens"] > budget:
        continue
    selected.append(row["id"])
    used_tokens += src["tokens"]

players = ["math_verified", "code_tests", "synthetic_template"]
base_gain = {"math_verified": 0.30, "code_tests": 0.22, "synthetic_template": -0.06}


def utility(subset):
    subset = set(subset)
    score = sum(base_gain[p] for p in subset)
    if {"math_verified", "code_tests"}.issubset(subset):
        score += 0.08
    if "synthetic_template" in subset and "math_verified" not in subset:
        score -= 0.05
    return score


shapley = {p: 0.0 for p in players}
orders = list(permutations(players))
for order in orders:
    prefix = []
    for p in order:
        shapley[p] += utility(prefix + [p]) - utility(prefix)
        prefix.append(p)
shapley = {k: round(v / len(orders), 4) for k, v in shapley.items()}

report = {
    "ranked_sources": [(row["id"], row["value"]) for row in ranked],
    "top_attribution": [(row["id"], row["grad_sim"]) for row in sorted(rows, key=lambda x: x["grad_sim"], reverse=True)[:3]],
    "selected_under_budget": selected,
    "used_tokens": used_tokens,
    "negative_or_blocked": negative,
    "shapley_demo": shapley,
    "total_selected_value": round(sum(row["value"] for row in rows if row["id"] in selected), 4),
    "blocked_contamination": [src["id"] for src in sources if src["contam"]],
}

for key, value in report.items():
    print(f"{key}=", value)

assert report["selected_under_budget"] == ["code_tests", "math_verified", "safety_boundary"]
assert report["used_tokens"] == 2300
assert report["negative_or_blocked"] == ["old_legal_forum", "benchmark_leak"]
assert report["shapley_demo"] == {"math_verified": 0.365, "code_tests": 0.26, "synthetic_template": -0.085}
```

运行后会看到类似输出：

```text
ranked_sources= [('math_verified', 0.1808), ('code_tests', 0.1599), ('zh_domain', 0.1288), ('safety_boundary', 0.1202), ('synthetic_template', 0.0782), ('old_legal_forum', -0.0184), ('benchmark_leak', -1.0)]
top_attribution= [('math_verified', 0.942), ('benchmark_leak', 0.93), ('synthetic_template', 0.922)]
selected_under_budget= ['code_tests', 'math_verified', 'safety_boundary']
used_tokens= 2300
negative_or_blocked= ['old_legal_forum', 'benchmark_leak']
shapley_demo= {'math_verified': 0.365, 'code_tests': 0.26, 'synthetic_template': -0.085}
total_selected_value= 0.4609
blocked_contamination= ['benchmark_leak']
```

这个 demo 的重点是：高 attribution 相似度不等于可训练价值。`benchmark_leak` 和目标梯度很相似，但因为评测污染必须被阻断；`old_legal_forum` 虽然有一点覆盖度，但质量低、风险高且带来负向指标，应进入重洗或降权候选。

---

## 20. 常见面试题

### 20.1 什么是 data attribution？

data attribution 是追溯模型行为、预测、错误或能力变化可能来自哪些训练数据、数据源或数据分布的过程。它主要用于解释、诊断和审计。

### 20.2 什么是 data valuation？

data valuation 是估计数据对目标模型性能、能力、风险或产品价值的贡献，用于数据选择、采购、标注优先级和配比决策。

### 20.3 Influence functions 的核心思想是什么？

它估计某个训练样本权重发生微小变化时，对某个测试点预测或 loss 的影响，从而追溯哪些训练点最影响该预测。大模型中直接使用成本高，通常作为思想或局部诊断工具。

### 20.4 Data Shapley 的核心思想是什么？

把每条数据看作参与者，用它在不同数据子集中的平均边际贡献衡量价值。它理论性质好，但精确计算昂贵，大模型中通常只能做近似或用于小数据池。

### 20.5 大模型中如何实际评估数据源价值？

更常用源级 ablation、小模型 proxy、上采样/下采样实验、validation loss、目标任务评估、人工审计和成本风险分析，而不是逐条精确估值。

### 20.6 数据价值是不是一个固定分数？

不是。数据价值依赖目标模型、训练阶段、已有数据分布、评估指标和风险约束。同一数据在不同任务和配比下价值不同。

### 20.7 如何发现负价值数据？

可以结合低质量分、异常 loss、人工审计、错误归因、数据消融、安全回归和污染检测，找出错误标注、过时知识、低质合成、隐私风险或导致误拒的数据。

---

## 21. 常见误区

误区一：数据价值可以精确算出来。

在大模型场景下，价值通常只能近似估计，并且依赖目标和评估集。

误区二：单条数据估值最重要。

预训练中源级、簇级和任务级估值往往更实用；样本级估值更适合 SFT、偏好和安全数据。

误区三：某数据源提升一个 benchmark 就一定有价值。

还要看污染、泛化、副作用、安全、成本和授权。

误区四：低质量分数据一定没用。

某些真实口语、低资源语言或边界样本可能质量分低但对目标能力重要。

误区五：数据归因能证明严格因果。

多数工程归因是近似证据链，不是严格因果证明。

误区六：数据选择只选高价值样本。

还要保留多样性、长尾覆盖和真实分布，否则模型容易过拟合窄目标。

---

## 22. 本章小结

Data Attribution 与 Valuation 是大模型数据工程从“经验驱动”走向“证据驱动”的关键。

本章要记住几句话：

1. attribution 关注模型行为来自哪些数据，valuation 关注数据有多少决策价值。
2. influence functions 和 Data Shapley 提供了重要思想，但大模型中通常需要近似。
3. 数据价值不是固定属性，而是目标依赖的边际贡献。
4. 大模型更常做数据源级、数据簇级和任务级估值。
5. 好的数据估值要同时看收益、成本、风险、副作用和可治理性。
6. 最实用的方案是质量评估、小模型实验、消融、相似检索、人工审计和版本记录组成的证据链。

如果面试中被问到数据价值评估，最好的回答不是直接说“用 Shapley 算”，而是说明：先定义目标和评估矩阵，再建立数据元信息，做源级和小规模实验，结合 attribution 诊断、副作用评估、成本风险分析，最后形成可版本化的数据决策。
