# 第十章：Reasoning 评估：从一个分数到可解释证据

Reasoning model 的评估不能只回答“最终答案对不对”。我们还需要知道：模型是否理解了题目，过程是否支持结论，换一种题面后是否仍然有效，是否见过评测数据，是否使用了更多推理时计算，选择器是否把正确候选选了出来，以及这个提升是否稳定、可复现、值得相应成本。

一个平均准确率可以很高，却掩盖证明题全错、简单题回归、变体崩溃或被污染样本抬分。一个模型也可能在候选池中生成了正确答案，却被 verifier 选错；另一个模型的分数提升可能只是用了更多采样、搜索和工具调用。若不把这些因素拆开，评估报告看起来精确，结论却无法解释。

本章从评估对象开始，逐步建立任务契约、答案指标、过程指标、变体和鲁棒性、污染审计、推理时计算公平比较、配对统计、人工与模型评审、成本账本和完整报告结构。最后的 demo 使用六个教学样本，故意制造污染、过程错误、变体失败、简单题回归和小样本统计不稳定，让读者看到为什么评估报告必须保留失败证据。

## 0.1 一个平均分提升但结论仍不稳的例子

假设一个 baseline 在六道题上答对两道，candidate 答对四道。平均准确率从 `0.333` 上升到 `0.667`，这看起来是明显提升。但进一步查看会发现：

- candidate 在一条已污染的数学题上答对，而对应变体仍然答错；
- proof hard slice 仍然是零准确率；
- 一道原本简单且答对的题被 candidate 改错；
- 过程步骤准确率只有 `0.722`；
- 第一处错误定位只有 `0.667`；
- 配对 bootstrap 区间跨过零，样本太少，不能把提升说成稳定结论。

这不是说 candidate 没有进步，而是说“平均分提升”只回答了一个问题。完整评估还要告诉读者提升来自哪些样本、是否迁移到变体、是否损伤其他切片、过程是否可靠、统计不确定性多大以及成本增加了多少。

## 0.2 初学者视角：评估究竟在比较什么

初学者可以把一次评估理解为四个问题：

1. 给模型同一批题，它答对了多少。
2. 把题目换个数字、顺序或条件，它还答对吗。
3. 把模型的解法拆开看，中间步骤是否成立。
4. 为了得到这个结果，模型用了多少次采样、工具、token 和时间。

如果只看第一个问题，模型可以通过记忆原题、套用模板或依赖不完整测试得到高分。后面三个问题用于判断高分是不是可迁移、可解释和可承担。

## 0.3 专家视角：评估至少有三层对象

专家需要区分：

- 模型能力：在固定预算、固定任务和独立数据上产生正确行为的能力；
- 策略能力：self-consistency、verifier、search、tool loop 或多候选选择带来的系统收益；
- 产品能力：在真实延迟、权限、成本、数据漂移和失败处置下完成任务的能力。

模型能力提升和策略计算增加可能同时发生。若 candidate 使用 64 个候选、三轮 verifier 和外部工具，而 baseline 只使用一次 greedy 生成，那么两者的准确率不能直接被解释为模型能力差异。评估必须把预算和系统组件写进结果。

## 0.4 资料与证据边界

HELM 提供了多维度、可复现的语言模型评估框架入口，强调不应只用一个分数描述模型。BIG-bench 收集了广泛任务，适合说明任务多样性和切片分析；MMLU 与 GPQA 可作为知识、推理和专家题评估的公开入口，但公开题、数据来源和污染风险必须单独审计。

LiveCodeBench 代表按时间更新的代码评估思路，适合减少一部分静态题库污染；HumanEval、SWE-bench 和前一章的代码执行指标分别覆盖函数级生成与真实仓库修复的不同层面。EleutherAI 的 lm-evaluation-harness 和 OpenAI Evals 仓库提供了评估实现与样例组织方式，但仓库代码能说明框架如何运行，不能自动证明某个模型在所有任务上的能力。

本章的公式是评估定义，代码和数字是教学构造。论文、数据集、评估框架、模型产品说明和目标系统实测属于不同证据等级；正文不把它们混写成同一种事实。

## 1. 评估对象：从题目到证据记录

### 1.1 评估样本的完整表示

一个 reasoning 评估样本不能只保存题面和一个答案。可以抽象为：

~~~math
e_i=(x_i,y_i^\star,V_i,P_i,z_i,q_i,g_i,r_i,b_i)
~~~

其中：

- `x_i` 是原题、输入或任务规格；
- `y_i^star` 是答案、答案等价类或任务 oracle；
- `V_i` 是由原题生成的变体和反事实集合；
- `P_i` 是模型生成的候选集合；
- `z_i` 是被评估的过程、工具轨迹或中间状态；
- `q_i` 是步骤正确性、相关性或证据支持标签；
- `g_i` 是题型、难度、语言、领域、长度等切片标签；
- `r_i` 是污染、泄漏、格式异常和评估争议等风险标记；
- `b_i` 是采样、token、搜索、verifier 和工具预算。

如果没有 `g_i`，平均分无法定位短板；没有 `r_i`，污染样本可能被当作普通正确样本；没有 `b_i`，不同推理策略的分数无法公平比较；没有 `V_i`，无法判断模型是否依赖原题表面形式。

### 1.2 三种“正确”

评估中常见三种正确性：答案正确、过程正确和任务完成正确。数学答案可以是数值或表达式，过程正确要求每一步满足前提和变换，任务完成正确还可能要求单位、格式、引用、权限和工具状态都符合契约。

代码任务中，程序通过测试不等于解释正确；文档问答中，答案文字正确不等于引用页码支持了该答案；工具 agent 中，最终状态正确不等于中间执行没有越权。评估对象要明确自己测的是哪一层。

### 1.3 评估声明的强度

“模型在某 benchmark 上得分高”是较弱声明；“模型在未见题型、变体和独立工具环境上稳定”是更强声明；“模型在真实工作流中安全、低成本且可回滚”则还需要系统和运营证据。一个评测集的好成绩只能支持它覆盖到的声明，不能自动外推到更强结论。

可以把报告里的每个结论写成 `任务范围 + 数据切分 + 预算 + 指标 + 不确定性 + 证据来源`，避免用一句“推理能力提升”覆盖多个不同命题。

### 1.4 评估 oracle

封闭答案题可以用精确匹配、数值容差、符号等价或程序验证；开放题可以用 rubric、多个标注者、参考证据和人工复核。oracle 不是天然真理：参考答案可能遗漏等价表达，程序检查器可能覆盖不全，LLM judge 可能偏好长文本，人工标注也可能存在分歧。

每个 oracle 都应记录适用范围、已知盲点和拒判策略。评估器无法判断时，宁可返回“不确定”并进入人工抽样，也不要把不确定样本强行标成错误或正确。

## 2. 评估实验的基本契约

### 2.1 固定数据切分

至少区分训练集、开发集、最终测试集、变体集、新任务集和污染审计集。开发集可以用于调 prompt、解析器和 verifier；最终测试集不能反复用于调参；新任务集用于检验时间、来源和题型外推。

按样本随机切分可能把同一模板、同一作者、同一仓库或同一参数族分到两边。更可靠的切分单位可能是题目族、教材、竞赛、仓库、时间窗口或生成器版本。

### 2.2 固定 prompt 和解码

评估记录至少包括系统提示、用户模板、few-shot 示例、是否要求过程、答案格式、temperature、top-p、top-k、最大 token、停止词、随机种子和重试规则。只改变一个 prompt 标点而没有记录，可能造成不可解释的分数波动。

如果评估使用自适应路由，路由规则本身也属于实验条件。模型看到的任务难度预测、历史失败和工具状态都应记录，否则别人无法重放同一策略。

### 2.3 工具和候选预算

应分别记录候选数、平均输出 token、verifier 调用、搜索节点、工具调用、并发数、平均延迟、P95 延迟和错误重试。工具返回的信息量也可能改变任务难度，不能只写“开启工具”。

### 2.4 Baseline 的选择

一个可信比较至少需要：

1. 简单 direct 或 greedy baseline；
2. 使用相同模型和更多推理预算的策略 baseline；
3. 使用相同预算但不同选择器的对照；
4. 对关键切片的已知强基线或人工上限。

没有 direct baseline，无法知道 reasoning 策略带来多少增益；没有预算匹配 baseline，无法判断增益来自计算量还是算法；没有简单题切片，可能看不到策略对低难度任务的伤害。

### 2.5 运行环境和版本

模型权重、推理框架、tokenizer、评估脚本、答案 parser、数据版本和硬件都要记录。代码任务还要记录编译器、依赖、测试命令和网络状态；长上下文任务要记录截断、检索、缓存和文档排序。

## 3. 最终答案评估

### 3.1 精确答案准确率

最基本的答案准确率是：

~~~math
A_{\mathrm{ans}}=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[\hat y_i\equiv y_i^\star]
~~~

这里要求 `N` 是正整数，即评估集不能为空。`\mathbb{1}[\cdot]` 是指示函数：条件成立时取 1，否则取 0；如果没有样本，准确率不是 0，而是“没有可计算结果”。正式报告应把空集合记为 `NA` 或 `None`，不能用 0 伪装成模型在某个任务上全部答错。

`equiv` 不一定是字符串相等，可以是数字容差、表达式等价、集合相等、代码通过独立测试或结构化字段一致。报告必须说明归一化和容差，否则同一个模型可能因 parser 不同得到不同分数。

### 3.2 部分得分和任务完成

有些任务不适合二元对错。例如多字段抽取、长答案引用和多步骤工具操作可以按字段、claim 或状态分解。若第 `i` 个样本有 `m_i` 个可检查字段，字段级支持率可以写成：

~~~math
A_{\mathrm{field}}=
\frac{\sum_i\sum_{j=1}^{m_i}\mathbb{1}[\hat y_{ij}\text{ satisfies }y_{ij}^\star]
}{\sum_i m_i}
~~~

这里 `m_i` 必须是非负整数，并且所有样本至少合计一个可检查字段，即 `\sum_i m_i>0`。如果字段总数为 0，字段级准确率应记为未定义；如果某个样本没有字段，它可以被保留，但不能贡献一个隐含的 0 分。

字段平均可能被大量容易字段抬高，因此还要报告任务级全对率、关键字段召回和高风险字段错误率。

### 3.3 选择性回答和拒答

当模型可以选择回答或拒答时，不能只统计所有请求上的准确率。设模型在置信度阈值 `t` 下选择回答的集合为 `S_t`，覆盖率和条件准确率分别是：

~~~math
\operatorname{coverage}(t)=\frac{|S_t|}{N},\qquad
A_{\mathrm{selective}}(t)=
\frac{\sum_{i\in S_t}\mathbb{1}[\hat y_i\equiv y_i^\star]}{|S_t|}
~~~

该式要求 `N>0`。当 `S_t` 为空时，覆盖率是 `0`，但选择性准确率没有分母，应记为未定义而不是 `0`；只有至少有一个被回答的样本时，`A_selective(t)` 才能计算。这样可以区分“模型拒答了所有问题”和“模型回答了问题但全部答错”。

高风险系统要观察“少答一些后是否真的更可靠”，以及拒答是否集中在简单题、少数语言或某类用户上。拒答本身也可能错误，例如模型在有充分证据时拒答，造成可用性下降。

### 3.4 代码最终行为

代码评估的最终答案通常是程序行为。需要区分 pass@1、候选池 pass@k、公开测试通过率、隐藏测试通过率、修复后准确率、超时率和安全事件。一个函数级候选通过小测试，不代表它满足完整规格；一个仓库补丁通过局部测试，不代表没有回归。

### 3.5 数学和符号等价

数学答案需要处理单位、数值精度、表达式等价、多解和无法确定。`0.5`、`1/2` 和 `50%` 只有在量纲和题目语境允许时才等价；`x=2` 与 `2=x` 通常等价，但 `x=2` 与 `x≈2` 的证据强度不同。答案 parser 的每条归一化规则都应有正例和反例。

## 4. 过程评估：答案之外的证据

### 4.1 步骤准确率

若第 `i` 个轨迹有 `L_i` 个步骤，步骤标签为 `q_ij`，模型或评估器预测为 `hat q_ij`，步骤准确率可以写成：

~~~math
A_{\mathrm{step}}=
\frac{\sum_i\sum_{j=1}^{L_i}\mathbb{1}[\hat q_{ij}=q_{ij}]}
{\sum_i L_i}
~~~

`L_i` 是非负整数，且整个评估集合必须至少包含一个步骤，即 `\sum_i L_i>0`。没有步骤的任务不应被强行计成步骤准确率 0；应从这个指标中排除并单独报告，或者将结果记为未定义。步骤标签的定义也要固定，例如这里的分子表示“候选轨迹的步骤正确性标签与 oracle 标签相同”，并不等于模型生成的解释在语言上更流畅。

这个指标容易被大量简单正确步骤抬高。应同时报告错误步骤召回、相关性、按步骤位置分组的结果，以及没有错误样本上的误报率。

### 4.2 第一处错误

如果一条轨迹的真实步骤标签中第一处错误位置是 `e_i^star`，预测位置是 `hat e_i`，在存在错误的集合 `E` 上可定义：

~~~math
A_{\mathrm{first}}=
\frac{1}{|E|}
\sum_{i\in E}\mathbb{1}[\hat e_i=e_i^\star]
~~~

这里要求 `|E|>0`；如果评估集没有任何真实错误轨迹，第一处错误准确率是未定义，而不是 0。正文公式采用 1-based 位置，即第一步为 1；工程实现若使用 Python 列表索引，通常采用 0-based 位置，必须在数据 schema 中明确转换规则。无错误轨迹用 `None` 或单独的 `no_error` 状态表示，不能让它和某一个整数位置混用。第一处错误比错误步骤总数更接近修复目标，因为后续步骤可能只是错误前提的连锁结果。

### 4.3 过程支持最终答案吗

答案正确、过程错误的样本会制造假成功；答案错误、过程大部分正确的样本可能是格式、单位或最后一步计算错误。可以把轨迹拆成“关键事实、关键变换、边界检查、结论”四类 claim，并检查每类是否有支持。

对于开放式解释，不能把模型生成的长过程直接视为真实内部思考。可见解释可能是事后合理化，评估应使用反事实干预、步骤删除、工具结果替换或独立程序检查测试它是否真的支撑答案。

### 4.4 过程评估器的来源

过程标签可以来自专家、程序 verifier、多个模型交叉评估、用户反馈或人工抽样。程序检查适合算术、代码、形式化证明和结构约束；人工适合判断问题理解、相关性和开放式证据；LLM judge 可以扩大覆盖，但必须校准和抽检。

评估器本身也应有独立数据、错误案例和置信度。不能因为 judge 给出了详细解释，就把它当作比简单二元判定更可靠。

### 4.5 过程与结果的交叉表

至少报告四类交叉情况：

- 答案对、过程对：可作为高质量成功；
- 答案对、过程错：需要识别碰巧正确或错误抵消；
- 答案错、过程对：可能是格式、单位或末步错误；
- 答案错、过程错：需要定位第一处错误。

这张交叉表通常比一个总分更能指导训练数据和 verifier 改进。

## 5. 切片、变体和鲁棒性

### 5.1 为什么平均分会隐藏问题

如果 easy 样本占 80%，模型在 easy 上提升 5 个百分点，在 proof hard 上下降 20 个百分点，整体平均可能仍然上升。切片准确率对每个组 `G_g` 定义为：

~~~math
A_g=
\frac{\sum_{i\in G_g}\mathbb{1}[\hat y_i\equiv y_i^\star]}
{|G_g|}
~~~

每个被报告的切片都必须满足 `|G_g|>0`。空切片只能报告为未定义，并应从汇总表中标出样本数为 0；把它写成 0 会让读者误以为该组有样本且模型全部失败。

至少应按题型、难度、语言、长度、工具需求、是否污染、是否需要多步依赖和风险等级切片。小切片要同时报告样本量和区间，避免把一个样本的波动当成趋势。

### 5.2 原题和变体

变体评估只改变一个有明确作用的条件，例如数字、实体顺序、单位、叙述风格、无关信息或问题目标。变体生成后必须重新求解和验证，不能假设机械替换一定保持题目可解。

变体集应与原题成对保存：原题答案、变体答案、改变字段、预期变化和生成器版本。若一个变体改变了题目难度或解空间，报告应说明这种变化，而不是把下降全部归因于鲁棒性。

### 5.3 鲁棒性下降

设原题准确率为 `A_orig`，变体准确率为 `A_var`，可定义下降量：

~~~math
D_{\mathrm{robust}}=A_{\mathrm{orig}}-A_{\mathrm{var}}
~~~

`A_orig` 和 `A_var` 应在同一批非空、逐样本配对的题目上计算；如果两边使用了不同样本集合，差值同时混入了模型变化和样本难度变化，不能称为纯粹的鲁棒性下降。若配对集合为空，下降量应记为未定义。

下降量大说明模型可能依赖表面形式、原题记忆、固定位置或无关线索。下降量为负并不自动表示模型更鲁棒，也可能说明变体更容易、两组数据难度不匹配或样本量太小。

### 5.4 反事实干预

更强的测试不仅改写输入，还检验模型是否改变了应该改变的结论。例如把利率从 5% 改成 6%，模型应更新金额；把不支持结论的页码替换成支持页，模型应重新判断证据；把代码边界从奇数改成偶数，函数行为应随规格变化。

反事实评估可以报告正确响应率、无关字段不变率和方向性错误率。一个只在最终答案上比较的分数，可能看不到模型对关键条件没有响应。

### 5.5 长上下文和干扰

长上下文评估要记录目标证据位置、干扰段数量、证据重复、顺序、截断和检索策略。模型找到了某个正确答案，不代表它使用了目标证据；可以通过替换、删除和位置移动测试证据依赖。

## 6. 数据污染和切分审计

### 6.1 污染的定义

污染是训练、提示示例、检索库或模型缓存中存在评测题目、答案、解析、测试或近似结构，使得测试不再独立。它不一定是模型逐字记住了题目；模板、变量重命名、同一仓库的相邻提交和标准解题措辞都可能产生泄漏。

### 6.2 污染率

设 `r_i^contam=1` 表示第 `i` 个样本存在污染风险，教学上的样本污染率是：

~~~math
R_{\mathrm{contam}}=
\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[r_i^{\mathrm{contam}}=1]
~~~

这里要求 `N>0`，并且每个 `r_i^contam` 都已经按照同一审计规则标注。空评估集没有污染率；检测器没有证据时也不能把“未发现”直接解释为“没有污染”。

这个比率是审计结果，不是模型真实记忆概率。检测应报告规则、阈值、时间范围、数据来源和人工复核样本。

### 6.3 多层去重

字符串 exact match 只能找到逐字重复；语义 embedding 可以找到改写，却可能把不同题误合并；公式和 AST 结构去重可以发现变量重命名，但需要处理等价变换；来源和时间切分可以减少同一出版物或仓库族泄漏。可靠审计通常组合多层信号，再对高风险近邻人工复核。

### 6.4 时间和来源切分

按公开时间切分可以减少未来题泄漏，但新题仍可能与旧模板相似。按作者、竞赛、教材、仓库或生成器切分更严格，却可能造成训练和评估分布差异。报告必须说明选择切分的原因以及它对外推结论的限制。

### 6.5 污染样本的处理

污染样本可以删除、降权、单独报告或作为污染敏感性分析的一部分。不能只删除能降低分数的样本，再把清洗后的最高分当作唯一结果。至少报告包含全部样本、排除高风险样本和只看新题三种结果。

## 7. 数学、代码和开放任务的不同评估

### 7.1 数学任务

数学评估可以利用数值、符号、程序和形式化证明检查，但要处理多解、单位、容差和证明跳步。GSM 类文字题适合观察读题和多步算术，MATH 类题覆盖更多竞赛结构，形式化证明则把自然语言结论转换为检查器可接受的项。

数学 benchmark 的高分不能单独证明通用 reasoning。公开题库、答案解析和模板污染会抬高分数；只看最终答案会忽略过程错误；只用一个题型会造成能力误判。至少要报告题型、难度、变体、过程和污染切片。

### 7.2 代码任务

代码评估的强项是可执行 oracle，弱项是测试覆盖、环境差异和安全边界。HumanEval 风格任务主要看函数级生成，LiveCodeBench 强调时间更新，SWE-bench 更接近真实仓库修复。它们不能用同一 pass@k 数字直接排序。

代码报告应拆分生成、候选池、选择、修复、隐藏测试、资源、回归和安全。一个模型 pass@k 高但选择器差，和一个模型候选少但单次稳定，是不同的工程方案。

### 7.3 逻辑、知识和开放式任务

逻辑题需要检查条件满足和反例；知识题需要核对事实时间、来源和引用；开放式任务需要 rubric、多个标注者和拒判。LLM judge 可以帮助扩大覆盖，但不能绕过参考证据、偏差审计和人工抽样。

### 7.4 Agent 和工具任务

Agent 评估要同时观察最终状态、工具选择、参数正确性、权限、失败恢复、轨迹成本和副作用。只看最后答案可能漏掉模型读取不该读的文件、调用不该调用的工具或修改了错误资源。

工具任务的 oracle 还要回答“是否完成了正确动作”和“是否以允许的方式完成”。在高风险任务中，安全拒绝、澄清请求和只读预览也应成为可评估结果，而不是简单算成失败。

## 8. 推理时计算的公平比较

### 8.1 预算向量

评估策略可以用预算向量记录：

~~~math
b_i=(K_i,T_i,D_i,J_i,U_i,L_i)
~~~

`K_i` 是候选或采样数，`T_i` 是生成 token，`D_i` 是搜索深度或节点数，`J_i` 是 verifier 调用，`U_i` 是工具调用，`L_i` 是延迟或时间预算。计数和时间必须是有限的非负数；通常 `K_i、D_i、J_i、U_i` 取非负整数，`T_i、L_i` 可以按 token 和毫秒记录为非负数。不同任务可以拥有不同预算，但比较时必须明确归一化方式，不能把“未调用工具”写成缺失值后再当作 0 参与平均。

### 8.2 成本账本

一个任务的教学成本模型可以写成：

~~~math
C_i=
c_{\mathrm{tok}}T_i
+c_{\mathrm{cand}}K_i
+c_{\mathrm{ver}}J_i
+c_{\mathrm{tool}}U_i
+c_{\mathrm{lat}}L_i
~~~

`c` 表示把各类资源换算成同一账本的单位价格或权重。实际比较可以用金钱、GPU 秒、延迟或能耗，但同一表格中不能混用未校准的单位。

若成本模型用于排序或单位成本报告，所有 `c` 和资源量都应是有限、非负数；允许某一项权重为 0，但负权重会让额外调用反而降低成本，通常不再具有成本含义。还要说明是按单任务计费、按批次摊销，还是包含人工复核和失败重试。

### 8.3 质量—成本曲线

设预算上限为 `B`，策略在该预算下的任务准确率为 `A(B)`。评估不应只报最高点，还要画出从低到高预算的曲线，观察边际收益：

~~~math
g(B_1,B_2)=
\frac{A(B_2)-A(B_1)}{C(B_2)-C(B_1)}
~~~

该式要求 `B_1`、`B_2` 都是有效预算，且成本增量 `C(B_2)-C(B_1)` 非零；通常比较 `B_2>B_1` 且成本严格增加的两个点。如果增加预算没有增加成本，边际收益每单位成本没有定义，应报告准确率变化和成本变化两项，而不是除以 0。若预算曲线包含失败或超时，`A(B)` 的分母也必须在各预算点保持一致。

如果高预算只增加很少准确率，可能不适合延迟敏感任务；如果提升集中在 hard slice，可能仍值得对高风险请求路由高预算。曲线还要包含失败、超时和安全事件，不应只画成功率。

### 8.4 候选覆盖和最终选择

候选池存在正确程序或答案，不等于最终选择正确。评估要分别记录候选存在率、verifier 选择准确率、人工或独立 oracle 复核准确率，以及选择器错误类型。否则多个候选的高 pass@k 可能掩盖选择器偏差。

### 8.5 延迟分布

并行采样可能降低平均墙钟时间，但增加峰值资源；串行修复可能提高成功率，却拉长尾延迟。报告至少包括 P50、P95、P99、超时率、排队时间和重试次数。平均延迟不能代表用户实际遇到的长尾。

### 8.6 预算匹配的比较例子

比较两个系统时可以建立三组实验：相同模型、不同预算；不同模型、相同预算；不同模型、不同预算但报告完整成本曲线。第一组测策略效果，第二组测模型差异，第三组更接近产品决策但解释难度最高。任何一组都不能只给一个最终分数。

## 9. 配对统计和不确定性

### 9.1 为什么要配对

如果 baseline 和 candidate 在同一批题上都运行，可以定义每个样本的差值：

~~~math
d_i=\mathbb{1}[\hat y_i^{\mathrm{new}}\equiv y_i^\star]
-\mathbb{1}[\hat y_i^{\mathrm{base}}\equiv y_i^\star]
~~~

配对提升为：

~~~math
\Delta_{\mathrm{pair}}=\frac{1}{N}\sum_{i=1}^{N}d_i
~~~

这里要求 `N>0`，且 baseline 与 candidate 必须在同一组任务、同一 oracle 下形成一一配对。`d_i` 只能在两个结果都可判定时计算；若某个系统超时或拒答，应先定义这些状态如何映射到任务契约，不能在两套规则下分别处理。

它利用同一道题的难度配对，通常比两个独立平均分相减更有诊断价值。还应列出 candidate 修复的样本、退化的样本和两者都失败的样本。

### 9.2 Bootstrap 区间

从 `d_i` 中按任务重采样 `R` 次，得到 `Delta*` 分布，百分位区间可以写成：

~~~math
\operatorname{CI}_{1-\alpha}=
\left[q_{\alpha/2}(\Delta^\ast),
q_{1-\alpha/2}(\Delta^\ast)\right]
~~~

这里要求 `d_i` 集合非空、`R` 是正整数，且 `0<\alpha<1`。`q` 是分位数，报告时要说明采用线性插值、最近秩还是其他约定；不同约定在小样本上可能给出不同端点。若样本按同一仓库、模板或题目族相关，应按族重采样，而不是把所有题当成独立样本。区间跨过零表示在当前样本和重采样假设下，提升方向仍不稳定。

### 9.3 McNemar 类配对检验

二元结果的配对表可以统计：baseline 错而 candidate 对的数量 `b`，以及 baseline 对而 candidate 错的数量 `c`。McNemar 类检验关注 `b` 和 `c` 的不对称，而不是把所有样本当成两组独立比例。小样本应使用精确版本或报告效应量和区间，不要把一个显著性数字当成能力大小。

### 9.4 多切片和多指标

同时测试很多题型、难度、语言和预算曲线，会增加偶然发现。应预先定义主要指标和关键切片，把探索性结果标出来，并报告样本量和多重比较处理。一个切片偶然提升不应自动变成总体能力结论。

### 9.5 随机种子和重跑

temperature、候选采样、搜索和 LLM judge 都可能产生随机波动。至少在固定任务上使用多个种子，报告均值、标准差、最差种子、候选重复率和失败案例。只保留表现最好的种子会产生选择偏差。

### 9.6 样本量和效应量

小幅提升需要足够样本才能区分真实效应和噪声。样本量不足时，应诚实报告“方向性结果”或扩大评估，而不是用复杂的统计术语掩盖不确定性。高风险任务还要单独估计错误上界，因为平均准确率可能无法覆盖罕见但严重的失败。

## 10. LLM judge 和人工评估

### 10.1 LLM judge 的适用场景

LLM judge 可以帮助评估开放式解释、风格、相关性、证据支持和多候选比较。它的优点是覆盖快、可以输出结构化理由、适合做初筛；它的结果必须通过 rubric、参考资料、交叉模型、人工抽样和校准验证。

### 10.2 Judge 的常见偏差

Judge 可能偏好更长、更有条理、更自信或格式更漂亮的答案；可能把错误但流畅的推理当成正确；可能与被评模型共享训练分布；也可能被题面中的提示注入影响。对过程评估，judge 看到的解释不一定是真实产生答案的因果过程。

### 10.3 评分 rubric

一个可复核 rubric 应明确每一项的定义、正反例、拒判条件和权重。例如：答案是否正确，关键条件是否覆盖，步骤是否有非法变换，引用是否支持 claim，工具动作是否符合权限，输出是否满足 schema。权重不能让“写得漂亮”抵消“结论错误”。

### 10.4 一致性和校准

多名标注者或多个 judge 对同一样本的分歧应被记录。可以报告一致率、类别混淆、置信度校准和按题型的分歧。分歧集中在 proof 或开放题时，说明任务本身的 oracle 需要更明确，而不是简单多数投票后假装没有问题。

### 10.5 人工抽样策略

人工不必查看所有样本，但应优先抽查：模型和 verifier 分歧、答案正确过程错误、污染近邻、变体失败、置信度极端、长轨迹、高风险工具动作和平均分改善却切片回归的样本。抽样规则和比例也要写入报告。

## 11. 评估失败模式

### 11.1 只报最高分

只报告最大候选数、最长推理或最高预算下的分数，会让读者误解为单次调用能力。应同时报告预算曲线、低预算基线、单位成功成本和尾延迟。

### 11.2 只看最终答案

这会把碰巧正确、错误抵消、格式修复和过程胡编混在一起。至少增加过程交叉表、第一处错误、证据支持和独立 verifier 结果。

### 11.3 原题高分、变体崩溃

这通常意味着记忆、模板匹配、位置偏置或条件理解不足。应报告原题和变体成对结果，并检查变体是否真的保持任务等价或只改变了明确条件。

### 11.4 污染未报告

公开 benchmark 题面、答案和解析容易进入训练数据。污染不一定能完全消除，但必须给出检测规则、风险样本数量和去除前后结果。

### 11.5 Judge 替代 oracle

LLM judge 不能替代代码测试、数学计算、引用核验或形式化证明。它适合补充开放维度，不适合把所有可判定任务都交给语言模型。

### 11.6 平均分掩盖回归

新的 reasoning 策略可能改善难题，却伤害简单题、低延迟请求或少数语言；也可能改善数学，伤害代码。切片和 paired regression 必须和平均分同时出现。

### 11.7 评估器被模型利用

模型可能学习 judge 喜欢的格式、固定措辞、冗长解释或测试漏洞。需要 hard negative、独立程序检查、隐藏变体和人工抽查，不能只优化 evaluator score。

### 11.8 忽略环境和协议

tokenizer、prompt、答案 parser、工具版本和上下文截断变化，都可能造成分数差异。评估脚本和环境版本应与模型结果一同保存。

## 12. 完整评估报告的结构

### 12.1 实验身份

报告开头应写模型版本、权重或 API 版本、评估脚本 commit、数据版本、时间、硬件、推理框架和随机种子。

### 12.2 任务和切分

说明任务来源、样本数、题型、难度、语言、训练/开发/测试关系、变体生成、污染检测、人工复核和排除规则。被排除的样本和原因也要保留。

### 12.3 推理配置

列出 prompt、few-shot、temperature、候选数、最大 token、search、verifier、工具、重试、超时、并发和预算。对自适应策略，还要展示路由分布和不同请求的平均成本。

### 12.4 结果表

至少包含总体答案、过程、第一处错误、变体、关键切片、候选存在率、选择准确率、延迟、成本和区间。不要只给一个总分；表格中要标注样本量和是否存在污染风险。

### 12.5 失败案例

选择能解释机制的案例：答案碰巧正确、过程第一步错误、variant 反应错误、verifier 选错、工具越权、简单题回归、超时和环境不一致。每个案例应有输入、模型输出、oracle、失败分类和修复建议。

### 12.6 结论强度

报告最后区分已被数据支持的结论、只在当前 benchmark 成立的结论、需要更多样本的方向性结果和无法由实验回答的问题。不要把“在六个教学样本上观察到”写成“模型具备通用 reasoning”。

## 13. 综合案例：合同证据助手的评估设计

### 13.1 任务对象

合同助手需要从指定版本的合同中找到金额、折扣、税率和付款条件，计算结果并引用页码。评估对象不只是最终金额，还包括引用是否支持数字、是否使用正确版本、是否遵守币种和审批状态。

### 13.2 样本和变体

原题可以给出清晰的合同页；变体替换页码顺序、增加相似旧版本、改变税率、加入无关附件或把金额写成不同格式。每个变体要记录预期答案变化和应保持不变的字段。

### 13.3 过程和工具指标

评估要记录模型检索了哪些页、是否读取了无权访问的附件、是否正确调用计算器、每个 claim 的证据支持率、引用页码准确率、金额计算准确率、拒答准确率和每任务 token/延迟。最终金额正确但引用错误，是“答案对、证据错”的失败，不应计作完整成功。

### 13.4 成本和风险分层

普通低金额请求可以使用较低预算，高金额或版本冲突请求可以增加检索、候选和人工复核。但比较时要按风险等级分层，不能把高风险请求的更多成本隐藏在总体平均中。涉及真实付款时，评估还要包含只读、审批和回滚行为。

## 14. 最小可运行实验：Reasoning 评估审计

下面的 demo 模拟六个 toy reasoning 样本，对比 baseline 和 candidate。它同时记录原题、变体、过程步骤、第一处错误、污染、切片、成本和 paired bootstrap 区间。candidate 的平均准确率确实更高，但其他证据不足，因此最后的 `all_review_checks_pass` 会是 `False`。

为了让示例可以审计，样本 schema 约定如下：`id` 必须唯一，三个正确性字段和 `contaminated` 必须是布尔值，`steps` 是非空布尔列表，`cost` 是有限非负数。正文公式用 1-based 步骤编号；本 demo 的 `first_error_gold` 和 `first_error_pred` 使用 Python 常见的 0-based 索引。无错误样本用两个 `None` 表示，并要求所有步骤都正确；有错误样本的 `first_error_gold` 必须指向步骤列表中的第一处 `False`。

~~~python
import math
from random import Random


SAMPLES = [
    {
        "id": "math_easy",
        "slice": "math",
        "contaminated": False,
        "baseline_correct": True,
        "candidate_correct": True,
        "variant_correct": True,
        "steps": [True, True, True],
        "first_error_gold": None,
        "first_error_pred": None,
        "cost": 120,
    },
    {
        "id": "math_contam",
        "slice": "math",
        "contaminated": True,
        "baseline_correct": False,
        "candidate_correct": True,
        "variant_correct": False,
        "steps": [True, True],
        "first_error_gold": None,
        "first_error_pred": None,
        "cost": 160,
    },
    {
        "id": "logic_distractor",
        "slice": "logic",
        "contaminated": False,
        "baseline_correct": False,
        "candidate_correct": True,
        "variant_correct": False,
        "steps": [True, False, True],
        "first_error_gold": 1,
        "first_error_pred": 1,
        "cost": 260,
    },
    {
        "id": "code_hidden",
        "slice": "code",
        "contaminated": False,
        "baseline_correct": False,
        "candidate_correct": True,
        "variant_correct": True,
        "steps": [True, True, True, True],
        "first_error_gold": None,
        "first_error_pred": None,
        "cost": 260,
    },
    {
        "id": "proof_hard",
        "slice": "proof",
        "contaminated": False,
        "baseline_correct": False,
        "candidate_correct": False,
        "variant_correct": False,
        "steps": [True, False, False, False],
        "first_error_gold": 1,
        "first_error_pred": 2,
        "cost": 420,
    },
    {
        "id": "simple_regression",
        "slice": "simple",
        "contaminated": False,
        "baseline_correct": True,
        "candidate_correct": False,
        "variant_correct": False,
        "steps": [True, False],
        "first_error_gold": 1,
        "first_error_pred": 1,
        "cost": 170,
    },
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ratio(numerator, denominator):
    require(
        isinstance(numerator, (int, float))
        and not isinstance(numerator, bool)
        and math.isfinite(numerator),
        "numerator must be a finite number",
    )
    require(
        isinstance(denominator, (int, float))
        and not isinstance(denominator, bool)
        and math.isfinite(denominator)
        and denominator >= 0,
        "denominator must be a finite non-negative number",
    )
    return round(numerator / denominator, 3) if denominator else None


def validate_samples(samples):
    require(isinstance(samples, list) and samples, "samples must be a non-empty list")
    ids = set()
    boolean_fields = (
        "contaminated",
        "baseline_correct",
        "candidate_correct",
        "variant_correct",
    )
    for sample in samples:
        require(isinstance(sample, dict), "each sample must be a dictionary")
        sample_id = sample.get("id")
        require(isinstance(sample_id, str) and sample_id, "id must be non-empty text")
        require(sample_id not in ids, "sample ids must be unique")
        ids.add(sample_id)
        require(isinstance(sample.get("slice"), str) and sample["slice"], "slice must be non-empty text")
        for field in boolean_fields:
            require(type(sample.get(field)) is bool, f"{field} must be boolean")
        steps = sample.get("steps")
        require(isinstance(steps, list) and steps, "steps must be a non-empty list")
        require(all(type(step) is bool for step in steps), "steps must contain booleans")
        gold = sample.get("first_error_gold")
        pred = sample.get("first_error_pred")
        for name, position in (("first_error_gold", gold), ("first_error_pred", pred)):
            require(
                position is None or (type(position) is int and position >= 0),
                f"{name} must be None or a non-negative integer",
            )
        require(
            isinstance(sample.get("cost"), (int, float))
            and not isinstance(sample["cost"], bool)
            and math.isfinite(sample["cost"])
            and sample["cost"] >= 0,
            "cost must be a finite non-negative number",
        )
        if gold is None:
            require(pred is None, "a no-error sample must not have a predicted position")
            require(all(steps), "a no-error sample must contain only correct steps")
        else:
            require(gold < len(steps), "gold error position is outside steps")
            require(not steps[gold], "gold error position must point to an incorrect step")
            require(all(steps[:gold]), "gold error position must be the first incorrect step")
            require(pred is not None and pred < len(steps), "predicted error position is outside steps")


validate_samples(SAMPLES)


def accuracy(key):
    require(
        key in {"baseline_correct", "candidate_correct", "variant_correct"},
        "unknown accuracy field",
    )
    return ratio(sum(sample[key] for sample in SAMPLES), len(SAMPLES))


def paired_bootstrap_ci(diffs, rounds=2000, seed=7, alpha=0.05):
    require(isinstance(diffs, list) and diffs, "diffs must be a non-empty list")
    require(
        all(isinstance(value, (int, float)) and math.isfinite(value) for value in diffs),
        "diffs must contain finite numbers",
    )
    require(type(rounds) is int and rounds > 0, "rounds must be a positive integer")
    require(type(seed) is int and seed >= 0, "seed must be a non-negative integer")
    require(isinstance(alpha, (int, float)) and 0 < alpha < 1, "alpha must be in (0, 1)")
    rng = Random(seed)
    n = len(diffs)
    values = []
    for _ in range(rounds):
        values.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    values.sort()

    # Use a fixed zero-based floor index so small teaching samples are reproducible.
    def quantile(probability):
        index = min(len(values) - 1, int(probability * len(values)))
        return values[index]

    return (
        round(quantile(alpha / 2), 3),
        round(quantile(1 - alpha / 2), 3),
    )


def nearest_rank(values, probability):
    require(values, "values must be non-empty")
    require(0 < probability <= 1, "probability must be in (0, 1]")
    rank = max(1, math.ceil(probability * len(values)))
    return sorted(values)[rank - 1]


def cost_per_success(total_cost, successes):
    require(total_cost >= 0 and math.isfinite(total_cost), "total cost must be non-negative and finite")
    require(type(successes) is int and successes >= 0, "successes must be a non-negative integer")
    return round(total_cost / successes, 3) if successes else None


baseline_accuracy = accuracy("baseline_correct")
candidate_accuracy = accuracy("candidate_correct")
variant_accuracy = accuracy("variant_correct")
diffs = [
    int(sample["candidate_correct"]) - int(sample["baseline_correct"])
    for sample in SAMPLES
]
paired_lift = round(sum(diffs) / len(diffs), 3)
bootstrap_ci = paired_bootstrap_ci(diffs)

all_steps = [step for sample in SAMPLES for step in sample["steps"]]
error_cases = [
    sample for sample in SAMPLES
    if sample["first_error_gold"] is not None
]
process_step_accuracy = ratio(sum(all_steps), len(all_steps))
first_error_accuracy = ratio(
    sum(sample["first_error_gold"] == sample["first_error_pred"] for sample in error_cases),
    len(error_cases),
)
contaminated = [sample["id"] for sample in SAMPLES if sample["contaminated"]]
robustness_drop = round(candidate_accuracy - variant_accuracy, 3)

slice_accuracy = {}
for slice_name in sorted({sample["slice"] for sample in SAMPLES}):
    group = [sample for sample in SAMPLES if sample["slice"] == slice_name]
    slice_accuracy[slice_name] = ratio(
        sum(sample["candidate_correct"] for sample in group),
        len(group),
    )

total_cost = sum(sample["cost"] for sample in SAMPLES)
candidate_correct = sum(sample["candidate_correct"] for sample in SAMPLES)
summary = {
    "baseline_accuracy": baseline_accuracy,
    "candidate_accuracy": candidate_accuracy,
    "paired_lift": paired_lift,
    "bootstrap_ci": bootstrap_ci,
    "variant_accuracy": variant_accuracy,
    "robustness_drop": robustness_drop,
    "process_step_accuracy": process_step_accuracy,
    "first_error_accuracy": first_error_accuracy,
    "contamination_rate": ratio(len(contaminated), len(SAMPLES)),
    "cost_per_correct": cost_per_success(total_cost, candidate_correct),
    "p95_cost_proxy": nearest_rank([sample["cost"] for sample in SAMPLES], 0.95),
}
review = {
    "accuracy_ok": candidate_accuracy is not None and candidate_accuracy >= 0.65,
    "lift_ci_nonnegative": bootstrap_ci[0] >= 0.0,
    "process_ok": process_step_accuracy is not None and process_step_accuracy >= 0.8,
    "first_error_ok": first_error_accuracy is not None and first_error_accuracy >= 0.8,
    "contamination_ok": len(contaminated) == 0,
    "robustness_ok": robustness_drop is not None and robustness_drop <= 0.2,
    "cost_ok": summary["cost_per_correct"] is not None and summary["cost_per_correct"] <= 400,
}

print(f"summary={summary}")
print(f"slice_accuracy={slice_accuracy}")
print(f"contaminated={contaminated}")
print(
    "regressions="
    f"{[sample['id'] for sample in SAMPLES if sample['baseline_correct'] and not sample['candidate_correct']]}"
)
print(
    "variant_failures="
    f"{[sample['id'] for sample in SAMPLES if sample['candidate_correct'] and not sample['variant_correct']]}"
)
print(f"review={review}")
print(f"all_review_checks_pass={all(review.values())}")
~~~

预期输出：

~~~text
summary={'baseline_accuracy': 0.333, 'candidate_accuracy': 0.667, 'paired_lift': 0.333, 'bootstrap_ci': (-0.333, 0.833), 'variant_accuracy': 0.333, 'robustness_drop': 0.334, 'process_step_accuracy': 0.722, 'first_error_accuracy': 0.667, 'contamination_rate': 0.167, 'cost_per_correct': 347.5, 'p95_cost_proxy': 420}
slice_accuracy={'code': 1.0, 'logic': 1.0, 'math': 1.0, 'proof': 0.0, 'simple': 0.0}
contaminated=['math_contam']
regressions=['simple_regression']
variant_failures=['math_contam', 'logic_distractor']
review={'accuracy_ok': True, 'lift_ci_nonnegative': False, 'process_ok': False, 'first_error_ok': False, 'contamination_ok': False, 'robustness_ok': False, 'cost_ok': True}
all_review_checks_pass=False
~~~

### 14.1 读取平均结果

candidate 的平均准确率从 `0.333` 提升到 `0.667`，paired lift 为 `0.333`。这说明在六个教学样本上 candidate 比 baseline 多答对两道题，但不说明提升已经稳定，更不说明所有题型都改善。

### 14.2 读取 bootstrap 区间

区间是 `(-0.333, 0.833)`，跨过零。原因不是 bootstrap 出错，而是样本太少，六个样本的配对差值不足以稳定估计总体提升。正式实验应增加独立任务，并按题目族或仓库重采样。

### 14.3 读取污染和变体

`math_contam` 被标为污染，candidate 在原题正确、变体错误；`logic_distractor` 也在变体失败。这两个样本说明原题准确率可能包含记忆或表面模板收益。报告可以给出包含污染样本和排除污染样本的敏感性分析，但不能只保留更高的那一个数字。

### 14.4 读取过程和切片

过程步骤准确率只有 `0.722`，第一处错误准确率是 `0.667`，说明 candidate 的答案提升伴随过程质量不足。proof 切片为 `0.0`，simple 切片也为 `0.0`，前者说明高难度证明没有改善，后者说明新策略造成了简单题回归。平均分必须和这些切片一起解释。

### 14.5 读取成本

候选正确的四个样本总成本为全部任务成本除以正确数，教学结果是 `347.5`。这个数字只是本 demo 的成本单位，不是现实价格；正式报告需要把 token、GPU 秒、工具费用、人工复核和延迟拆开，并按成功任务计算成本。

## 15. 把评估结果转成下一轮实验

### 15.1 先修 oracle 和切分

如果答案 parser、测试 oracle 或数据切分有问题，直接调模型没有意义。先修正等价类、单位、测试覆盖、变体配对、污染规则和版本记录，再重复 baseline 与 candidate 的配对实验。

### 15.2 根据失败类型分配工作

原题和变体都错，可能是生成或理解问题；原题对、变体错，可能是模板或污染问题；候选存在但最终选错，可能是 verifier 问题；答案对、过程错，可能是监督和解释问题；简单题回归，可能是训练混合或路由问题。失败分类应该直接对应下一轮数据、模型或系统改动。

### 15.3 控制变量

一次实验尽量只改变一个主要因素：模型版本、数据、prompt、候选数、verifier、工具或训练目标。若多个因素同时改变，应增加消融实验，否则无法知道收益来自哪一项。

### 15.4 记录负结果

变体下降、proof 不变、简单题回归、成本变高和 judge 分歧都是有价值的结果。只保存最好的 checkpoint、最高预算和最好切片，会让研究记录失去方向性信息。

## 16. 小练习

### 练习一：定义评估样本

为一个多解数学题设计 `e_i` 的字段，包含答案等价类、变体、过程标签、题型、污染和预算。说明为什么只保存一条参考答案会误伤合理解法。

### 练习二：解释 paired lift

在十道相同题目上，baseline 答对六道，candidate 答对七道；其中 candidate 修复了两道 baseline 错题，却让一道 baseline 正确题变错。计算配对提升，并解释它与独立平均分差值的关系。

### 练习三：构造变体集

为一道折扣和税率题设计四个变体：数字变化、单位变化、条件顺序变化和无关信息干扰。标注哪些结论应该变化，哪些应该保持不变。

### 练习四：污染敏感性分析

一个评估集 100 题，其中 8 题存在近重复风险。分别计算包含和排除高风险样本时的准确率，说明为什么应同时报告风险样本数量和去除规则。

### 练习五：预算公平比较

设计三组实验比较 greedy、self-consistency 和 verifier reranking。为每组记录候选数、token、verifier 调用、P95 延迟、准确率和单位成功成本。

### 练习六：评估 demo

把 `math_contam` 从样本中移除，重新运行 demo，观察平均准确率、bootstrap 区间和 `review` 的变化。解释清洗后分数变化为什么不等于模型能力变化。

## 17. 本章总结

Reasoning 评估要从最终答案扩展到过程质量、变体泛化、鲁棒性、污染独立性、推理预算、统计不确定性、安全和成本。每一个指标只回答一个局部问题，不能把答案准确率、pass@k、verifier 分数和单位成功成本混成一个总分。

可信评估需要明确数据切分、prompt、解码、候选、工具、verifier、环境和成本；需要同时看总体和切片、原题和变体、baseline 和 candidate、答案和过程、平均值和区间；还需要保留失败案例、污染样本、judge 分歧和负结果。

评估的目的不是制造一个漂亮数字，而是判断某个结论由哪些证据支持、在哪些范围内成立、哪些地方仍然未知。只有当实验能够区分模型能力、额外计算、选择器、数据污染、评估器偏差和统计噪声时，分数才真正具有解释价值。

下一章将讨论 reasoning model 的安全与局限，进一步分析幻觉、过度自信、伪推理、reward hacking、工具误用和高风险场景中的可靠性边界。

## 18. 资料索引

1. HELM：<https://arxiv.org/abs/2211.09110>
2. BIG-bench：<https://arxiv.org/abs/2206.04615>
3. MMLU：<https://arxiv.org/abs/2009.03300>
4. GPQA：<https://arxiv.org/abs/2311.12022>
5. LiveCodeBench：<https://arxiv.org/abs/2403.07974>
6. HumanEval：<https://arxiv.org/abs/2107.03374>
7. SWE-bench：<https://www.swebench.com/>
8. EleutherAI lm-evaluation-harness：<https://github.com/EleutherAI/lm-evaluation-harness>
9. OpenAI Evals：<https://github.com/openai/evals>

本章使用的链接分别代表论文、公开评估集、代码评估项目和评估框架。它们可以支持方法、数据和实现入口的说明，但不能替代当前模型版本、当前数据切分和目标系统的独立复测。
