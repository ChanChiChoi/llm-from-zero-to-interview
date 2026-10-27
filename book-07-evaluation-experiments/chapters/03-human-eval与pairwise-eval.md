# 第三章：Human Eval 与 Pairwise Eval

模型评估最容易被忽略的一件事，是“答案看起来不错”并不是一个可直接计算的
事实。对于分类题，可以把预测标签和标准标签逐个比较；对于开放式生成，答案
可能有很多种写法，真正重要的是它是否回答了问题、是否可靠、是否让用户能够
完成下一步动作，以及它有没有引入新的风险。

例如，用户问“这笔订单能否退款”。回答甲写得很流畅，却引用了已经失效的政策；
回答乙比较简短，却正确说明了退款期限、例外情况和需要补充的订单信息。只看
长度或词面相似度，可能会偏爱回答甲；让熟悉业务规则的人按照明确标准比较，
才有机会发现回答乙更有用。

这就是 human evaluation（人工评估）存在的原因。它不是“找几个人看一眼”，
也不是把主观印象包装成一个分数，而是一套有任务、有样本、有标注规则、有
质量控制和有统计解释的测量方法。pairwise evaluation（成对比较评估）是其中
一种重要形式：让评审在同一个输入下比较两个匿名回答，判断哪一个更好，或者
承认两者没有足够差异。

人工评估不能替代程序判定器。SQL 是否修改了数据、代码是否通过测试、工具是否
越权和 JSON 是否能解析，应该尽量由可执行检查完成。人工评估解决的是另一类
问题：多个可接受答案之间的质量差异、任务完整性、证据是否真正支持结论、表达
是否适合用户，以及自动指标没有捕捉到的失败。把两者放在同一套证据链里，才能
得到既可解释又可落地的结论。

## 1. 从答案匹配到任务完成

### 1.1 小白视角：为什么“看起来像正确答案”不够

假设题目是：

~~~text
请解释 Transformer 中的 attention，并举一个直观例子。
~~~

模型 A 用三句话解释了查询、键和值，但没有例子；模型 B 讲了一个很长的例子，
可是把 softmax 说成了归一化后的线性函数；模型 C 公式完全正确，却默认读者
已经知道矩阵乘法，初学者读完仍然不知道每个词为什么要关注另一个词。

这三个回答不适合用 exact match 比较，因为没有唯一的字符串答案。BLEU、ROUGE
或 embedding 相似度也只能回答“它和参考答案像不像”，不能可靠回答：

1. 核心概念是否正确；
2. 例子是否真的帮助理解；
3. 有没有遗漏题目要求的部分；
4. 语言是否适合目标读者；
5. 错误是否会让用户采取危险行动。

人工评估的第一步不是打分，而是把“好”拆成可以观察的行为。例如，解释题可以
要求“attention 的输入、权重计算和值聚合都正确，并且例子能对应到公式”；客服
题可以要求“结论符合当前政策，指出缺失信息，不能承诺系统没有授权的补偿”。

### 1.2 专家视角：测量的是构念，不是文风

评估术语中的 construct 指希望测量的抽象能力，例如事实正确性、任务完整性、
证据支撑、帮助程度或安全性。一个回答的语言风格只是表面变量，不能自动代表
这些构念。

对企业知识库问答，至少可以拆出下面几个层次：

1. **事实层**：回答中的原子声明是否与当前文档一致；
2. **证据层**：引用的段落是否真的支持对应声明；
3. **范围层**：文档没有说明时，模型是否承认不确定，而不是把推测说成政策；
4. **任务层**：是否回答了用户的全部子问题，并给出可执行的下一步；
5. **风险层**：是否泄露权限外信息，或提出未经授权的工具动作。

如果这些层次被揉成一个“整体印象分”，低分时无法知道是检索失败、证据拼接
失败、生成失败，还是标注员不喜欢回答风格。好的 human eval 不只是产生分数，
还要留下能指向故障位置的标签和理由。

### 1.3 什么时候需要人类判断

可以把评估任务按判定性质分成三类。

第一类是可以执行判定的任务，例如分类、代码单元测试、JSON schema、SQL 权限
和工具副作用。这些任务应优先使用程序判定器，人工只处理执行器无法覆盖的
语义问题。

第二类是存在参考证据但表述开放的任务，例如文档问答、摘要、改写和解释。程序
可以检查引用、数字和关键字段，人工再判断结论是否完整、是否误导。

第三类是没有单一标准答案的任务，例如研究方案、系统设计、复杂对话和用户偏好。
这类任务更依赖 rubric 和成对比较，但仍然需要明确评价对象，不能把“我喜欢”
直接当成“系统更可靠”。

## 2. 先定义一个可评判的样本

### 2.1 样本不只是 prompt 和 response

人工评估记录至少要包含：

~~~text
item_id
input
context_or_evidence
candidate_a
candidate_b
task_contract
rubric_revision
annotator_id
display_order
label
reason_codes
created_at
model_revision
prompt_revision
tool_revision
~~~

其中 `candidate_a` 和 `candidate_b` 不应该让评审看到真实模型名称；`display_order`
记录哪个候选实际出现在左侧或右侧；`reason_codes` 记录事实错误、遗漏、越权、
引用不支持等可分析的原因。没有这些字段，后来即使发现新旧版本分数不同，也
无法判断差异来自模型，还是来自 prompt、位置或标注协议变化。

一个样本可以抽象为：

~~~math
h_i=(x_i,c_i,a_i,b_i,y_i,z_i)
~~~

这里 `x_i` 是用户输入，`c_i` 是允许使用的上下文或证据，`a_i`、`b_i` 是两个
候选回答，`y_i` 是参考行为、参考证据或 rubric，`z_i` 是任务、语言、风险、
难度、来源、时间和版本等元数据。

### 2.2 写清被测系统的边界

同一个回答可以来自完全不同的系统：

~~~text
模型权重
  -> system prompt
  -> 检索器和上下文组装
  -> 工具调用与执行环境
  -> 对话记忆和重试策略
  -> 最终回答
~~~

如果只想比较模型权重，就必须固定 prompt、检索结果、工具和解码参数；如果想
比较完整产品，就应把这些组件一起记录，并在结论中说明收益属于哪个系统边界。
一个带检索的模型赢了无检索的模型，不能直接写成“模型本身更聪明”。

### 2.3 任务契约决定人类评什么

以退款客服为例，任务契约可以写成：

~~~text
输入：用户问题、当前退款政策和用户可见订单字段
输出：解释结论、所需补充信息、下一步操作
必须满足：金额和期限与政策一致；引用的政策版本正确
不可接受：捏造优惠、泄露其他用户信息、承诺没有授权的赔偿
不确定时：明确说明证据不足，并转人工或请求订单号
~~~

评审员据此判断回答是否完成任务，而不是判断哪一段文字更像参考答案。对于
多轮对话，还要给出历史消息、每轮目标、允许的工具和最终状态；对于工具 Agent，
还要记录工具请求、参数、结果以及外部副作用。最后一句话说得正确，却已经
错误删除了生产数据，不能因为文风好而算作成功。

### 2.4 选择评估对象

人工评估可以针对：

1. **回答质量**：只比较输出文本或结构化结果；
2. **证据质量**：比较检索段落、引用和声明之间的对应关系；
3. **交互质量**：评估澄清、记忆、拒答、打断和多轮状态；
4. **任务结果**：判断用户的业务目标是否完成；
5. **安全行为**：判断是否泄露、越权或产生不可逆副作用。

这些对象可以同时出现在同一个项目中，但标签必须分开保存。否则“回答更好”
和“任务结果更安全”会互相掩盖。

## 3. 绝对评分：建立质量画像

### 3.1 绝对评分适合什么问题

绝对评分让评审给一个回答或一个维度赋予等级，例如 0 到 5 分。它适合回答
“这一版本在事实性、完整性和帮助程度上分别表现如何”，也适合建立长期
回归基线。

它不一定适合非常细的模型差异。两个回答都接近满分时，评审可能无法稳定地区分
4.2 和 4.4；这时成对比较往往更容易。

### 3.2 Rubric 必须描述可观察行为

下面是一个客服回答的五级 rubric：

| 分数 | 可观察表现 |
| ---: | --- |
| 5 | 结论正确，覆盖全部关键条件，引用有效，下一步清楚且没有风险 |
| 4 | 主要结论正确，只有不影响行动的小遗漏或表达问题 |
| 3 | 部分正确，但遗漏关键条件，用户仍需二次确认 |
| 2 | 存在会误导用户的事实、范围或步骤错误 |
| 1 | 基本答非所问，或核心结论错误 |
| 0 | 严重幻觉、泄露信息、越权建议或完全无法使用 |

这里的关键不是数字本身，而是每个数字对应的行为。只写“5 分很好、1 分很差”
会让每个评审员自行发明尺度。

### 3.3 把维度拆开

一个回答可以在正确性上得 5 分，在帮助程度上得 3 分，在安全性上直接记为
严重问题。因此更实用的 rubric 是：

~~~text
correctness：0-5，结论和关键事实是否正确
completeness：0-5，是否覆盖任务要求的子问题
groundedness：0-5，是否能由给定证据支持
actionability：0-5，用户是否知道下一步怎么做
style：0-3，是否清楚、适合目标读者且不过度冗长
safety：safe / unsafe / needs_expert_review
~~~

安全通常不应与普通质量简单相加。一个流畅但泄露隐私的回答，不能靠 style
得分抵消风险。把安全单列，还能保留“需要专家复核”的不确定状态，而不是
强迫评审在安全和不安全之间猜测。

### 3.4 加权平均不是唯一真相

如果确实需要一个摘要分数，可以设每个维度的权重 `\alpha_d`：

~~~math
S_i
=
\sum_{d\in D}\alpha_d s_{i,d},
\qquad
\alpha_d\ge 0,
\qquad
\sum_{d\in D}\alpha_d=1
~~~

`s_{i,d}` 是样本 `i` 在维度 `d` 上的归一化分数。总体分数可以按样本权重聚合：

~~~math
S_{\mathrm{abs}}
=
\frac{\sum_{i=1}^{N}w_iS_i}
{\sum_{i=1}^{N}w_i}
~~~

权重的含义必须写清楚。它可以表示线上流量校正、风险优先级或稀有任务的
分析权重，但不能一会儿表示样本抽样概率，一会儿又表示严重度而不做区分。
高风险切片最好同时报告原始分数、样本量和独立的风险率，不要只留下加权总分。

### 3.5 序数分数的边界

0 到 5 分往往是序数尺度：5 分比 4 分好，但不一定比 4 分“多好同样的距离”。
把它们直接求均值是一种实用近似，不是测量理论上的必然结论。若评审主要在
几个等级之间做选择，可以报告每个等级的分布、中位数和 bootstrap 区间；若
团队需要均值来做回归，必须保持 rubric 和标注培训稳定。

问题不止是“序数值能不能求平均”。Ethayarajh 和 Jurafsky 从效用理论分析了
Likert 评分汇总所隐含的评审者假设，并证明在某些条件下，按平均 Likert 分给
系统排序会反转真实偏好；他们还提出 system-level probabilistic assessment
（SPA）来评估故事生成等开放任务。这不是说所有 5 分量表都会反转，也不能推出
pairwise 在所有任务都更可靠。它提醒我们先定义测量目标：若目标是目标用户的
选择倾向，应直接收集偏好；若目标是事实正确、证据有效或安全，则仍需独立的
行为维度和硬门槛，不能把“用户更喜欢”当作正确性的替代量。
[The Authenticity Gap in Human Evaluation](https://aclanthology.org/2022.emnlp-main.406/)
给出了这类汇总假设的理论分析；结论应绑定其效用模型和任务，不要推广为所有
人工评分都无效。

### 3.6 拒答和无效输出怎么记

拒答不是总是错误。面对证据不足、危险请求或权限外操作，合理拒答可能是正确
行为；面对普通且可回答的问题，错误拒答则是质量损失。记录中应至少区分：

1. 正确拒答；
2. 错误拒答；
3. 有条件回答但没有说明不确定；
4. 结构化输出无法解析；
5. 评审无法判断，需要专家复核。

如果把这些状态全部映射为 0 分，系统会看似容易统计，实际却丢掉了安全和
可用性之间的关键差异。

## 4. Pairwise Comparison：让评审比较两个答案

### 4.1 基本协议

pairwise eval 给评审同一个输入和两个候选回答：

~~~text
用户问题：同一份输入
回答 A：匿名候选
回答 B：匿名候选
判断：A 更好 / B 更好 / 两者相当 / 无法判断
原因：选择一个或多个预定义原因标签，并可补充短说明
~~~

系统应随机决定哪个候选在左边，随机化候选生成顺序，并把真实模型身份保存在
评估后台而不展示给评审。随机化不是装饰：如果新模型总在左侧，位置偏好会
和模型效果混在一起。

### 4.2 成对比较的优势与代价

人通常更容易回答“哪一个更适合这个用户”，而不是为单个回答寻找一个绝对
分数。成对比较减少了不同评审员对“4 分”的尺度差异，特别适合比较一个 prompt
或模型版本的局部变化。

但它只能提供相对信息。A 胜 B 不代表 A 达到了业务要求，也不说明 A 没有安全
问题；如果 A 和 B 都很差，评审仍然可能被迫选择一个。成对比较还会随着候选
数量增加而产生更多对局，因此需要抽样、对战设计和成本预算。

### 4.3 Win rate 的计算

假设新模型相对旧模型得到：

~~~text
new wins = 420
old wins = 350
ties     = 230
total    = 1000
~~~

把 tie 计作半个胜场时：

~~~math
\hat p_{\mathrm{new}}
=
\frac{420+0.5\times230}{1000}
=0.535
~~~

这表示在“随机抽取一个成对评审结果，并把平局折半”的定义下，新模型的
平均偏好得分是 53.5%。它不等于“53.5% 的用户会长期选择新模型”，因为用户
分布、重复使用、场景严重度和回答后的实际任务结果都可能不同。

如果把 tie 排除，则胜率是 `420/(420+350)=0.545`；如果把 tie 作为单独类别，
还可以同时报告 `tie_rate=0.230`。三种数字回答不同问题，报告不能只写一个
没有定义分母的 “win rate”。

### 4.4 不确定性和配对结构

把 tie 折半后，每条结果的得分是 \(z_i\in\{0,0.5,1\}\)，总体偏好率是这些得分的
均值。由于单条得分可能是 0.5，不能直接把它当作 Bernoulli 变量，再套
\(\hat p(1-\hat p)\) 的方差。若暂时假设 \(N\) 条结果相互独立，可以用样本方差估计：

~~~math
s_z^2
=
\frac{1}{N-1}\sum_{i=1}^{N}(z_i-\hat p)^2,
\qquad
\mathrm{SE}(\hat p)=\frac{s_z}{\sqrt{N}}
~~~

若胜、负、平数量分别为 `W`、`L`、`T`，且 `N=W+L+T`，折半平局均值为
`p̂=(W+0.5T)/N`。经验总体方差可以改写为
`p̂(1-p̂)-0.25(T/N)`；样本方差还要乘 `N/(N-1)` 的有限样本修正。因此直接套
Bernoulli 方差 `p̂(1-p̂)` 会把所有分数假装成 0 或 1，忽略平局的 0.5 得分。

回到前面的 420 胜、350 负、230 平例子，\(\sum z_i^2=420+0.25\times230=477.5\)，
所以 \(s_z^2\approx0.1915\)、标准误约为 \(0.0138\)。独立样本正态近似下，95% 区间约为
\([0.508, 0.562]\)。这个区间只是教学演算：若同一 prompt 被多位评审重复看、同一
用户贡献多条回答，独立性不成立，区间会偏窄。正式分析应先明确目标抽样单位，
再按 prompt、用户或会话聚类重采样；有分层或不等抽样概率时，还要保留设计权重。
对复杂的对战图排名，可以使用 Bradley--Terry 系数的 bootstrap 或稳健 sandwich
标准误，而不是把所有投票当作独立二元试验。

近似的 95% 区间可写成 \(\hat p\pm1.96\,\mathrm{SE}(\hat p)\)，但应同时报告胜、负、平数量、
样本单位和区间算法。上述独立假设下区间下界仅略高于 0.5；换成聚类区间后，
结论可能改变，因此不能只凭点估计 53.5% 宣布稳定提升。

### 4.5 让 tie 有清楚的语义

tie 不是“评审员懒得决定”，也不是“两个答案都没有错误”的同义词。规范可以
这样写：

1. 两个答案都正确、完整、风险相当，而且差异不足以影响用户行动时，选择 tie；
2. 一个答案更完整、更有证据或更容易执行时，即使两者都没有明显错误，也应
   选择更好的一方；
3. 一个答案更长但新增了无关内容时，不因为长度本身给予优势；
4. 评审缺少必要领域知识时，选择“无法判断”，不要把不确定性伪装成 tie；
5. 任一答案存在严重安全问题时，先记录安全原因，再按项目规则处理比较结果。

把“相当”和“无法判断”分开，可以区分真实质量接近与标注资源不足。

### 4.6 成对比较不能替代最低质量标准

如果新模型比一个很差的旧模型更好，win rate 很高，却仍然可能不适合发布。
因此每个候选答案都要单独检查硬性事实、安全、结构解析、权限和任务结果；
pairwise 只回答“在这个输入下谁相对更好”。相对偏好和绝对可用性必须并列
记录。

## 5. 绝对评分和成对比较如何配合

### 5.1 两种测量回答不同问题

绝对评分适合构建质量画像：新模型在正确性、完整性、引用和风格上各是多少。
成对比较适合回答局部选择问题：新 prompt 产生的回答是否比旧 prompt 更适合
用户。

可以把一次评估安排成两层：先对全体样本运行低成本的程序检查或自动筛选，再
对代表性样本做成对比较；对高风险、争议和模型分歧大的样本补做绝对 rubric
评分。这样既保留相对差异，又能知道两个候选是否都没有达到任务要求。

### 5.2 不要把不同量尺直接相减

5 分制从 3.1 提升到 3.4，与 pairwise win rate 从 50% 提升到 53% 并不是同一
个量纲。前者依赖 rubric 的等级间距，后者依赖候选对、tie 规则和样本分布。
可以在同一报告中并列展示，但不能不加说明地合成一个“综合提升”。

### 5.3 设计交叉验证

一种实用设计是从同一批 prompt 生成两个模型的回答，同时抽取三个结果：

1. 程序判定结果，例如 JSON、引用和安全规则；
2. 绝对 rubric 分数，例如正确性和完整性；
3. pairwise 偏好，例如用户更愿意采用哪一个。

如果程序正确率不变、绝对完整性提高、pairwise 也提高，证据比较一致；如果
pairwise 提高但引用准确率下降，说明偏好可能被文风或长度驱动，应回看样本，
而不是直接宣布新模型更好。

### 5.4 回答级偏好与系统级偏好：SPA 解决哪一层问题

开放式生成还有一个常被忽略的层级差异：评审是在比较两条具体输出，还是在评价
两个系统长期产生的输出分布？Ethayarajh 与 Jurafsky 把常见方案区分为 outcome-level
absolute assessment（OAA）、outcome-level relative assessment（ORA）和
system-level probabilistic assessment（SPA）。

| 协议 | 评审实际回答什么 | 聚合时依赖什么 | 主要风险 |
| --- | --- | --- | --- |
| OAA：回答级绝对评分 | 这条输出在指定维度上得几分？ | 不同标注者的分数可以放在同一量尺上；输出样本能代表系统表现 | Likert 刻度不同、缺少共同高低锚点时，平均分未必保留真实偏好，甚至可能反转系统次序 |
| ORA：回答级成对比较 | 对这条输入，两个输出哪一个更好？ | 根据每场局部判断，用 Bradley--Terry 等模型推回系统相对强度 | 50% 以上的局部胜率并不自动表示用户总体效用更高；少量灾难性失败可能被大量普通胜局淹没 |
| SPA：系统级概率判断 | 在看过双方一定数量的样本后，我选择 X 而不是 Y 的概率是多少？ | 让标注者直接把多个输出汇总为系统级偏好概率，再汇总不同标注者 | 评审必须能理解规定的曝光量并估计自己的长期选择；曝光样本如何抽取仍会影响结果 |

SPA 的关键变化不是换一种分数，而是把“多个输出怎样汇总成一个系统偏好”的责任
交给评审者明确表达。该研究在 GPT-3 故事生成上预先列出 5 个预期偏好关系，SPA
恢复了 5/5；标准 5 点量表只恢复 2/5，尽管每项检验功效约为 1 且显著性水平为
0.001。标准评分还没有显著区分 Curie 与 Davinci。
[论文](https://aclanthology.org/2022.emnlp-main.406/) 同时强调：这个结果来自
特定故事生成任务，不能直接推出 SPA 适合知识问答、安全决策或任何产品评估。
即便报告用户总体选择概率，正确性和严重风险仍要单独测量。

## 6. Elo、Bradley--Terry 与 Arena

### 6.1 Elo 的直觉

当候选模型只有两个时，win rate 已经足够表达相对差异；模型数量增加后，不
可能在每次报告中手工维护所有胜负组合。Elo 把每场对局的结果递归地更新到
两个 rating 中。

设模型 A、B 的 rating 为 `R_A` 和 `R_B`，A 的预期得分可以写成：

~~~math
E_A
=
\frac{1}{1+10^{(R_B-R_A)/400}}
~~~

真实结果 `S_A` 取胜为 1、平局为 0.5、失败为 0，则更新为：

~~~math
R_A'
=
R_A+K(S_A-E_A)
~~~

`K` 是更新步长。若低 rating 模型击败高 rating 模型，`S_A-E_A` 为正且较大，
它会获得更明显的上升；若强模型赢了符合预期的对局，更新就较小。

Elo 不是“模型真实能力”的绝对测量。初始分数、对局顺序、`K` 的大小、对战
样本和流量变化都会影响结果；它更像一种在线维护相对排序的工程算法。

### 6.2 一个 Elo 手算例子

若 `R_A=1500`、`R_B=1600`，则 A 的预期得分约为：

~~~math
E_A
=
\frac{1}{1+10^{100/400}}
\approx0.360
~~~

取 `K=32`，如果 A 爆冷获胜：

~~~math
R_A'
\approx1500+32(1-0.360)=1520.48
~~~

B 的分数则相应下降。这个例子只能说明更新机制，不能说明 20.48 分的变化
对应某个统一的能力单位。

### 6.3 Bradley--Terry：从全量对局估计强度

Bradley--Terry 模型直接把模型强度 `r_m` 转成成对胜率：

~~~math
P(m_i\succ m_j)
=
\frac{\exp(r_i)}{\exp(r_i)+\exp(r_j)}
=
\sigma(r_i-r_j)
~~~

其中 `\sigma` 是 logistic 函数。若一场对局的结果 `y_{ij}` 为 1 表示 `i` 胜，
则可以最大化：

~~~math
\mathcal L(r)
=
\sum_{(i,j)}
\left[
y_{ij}\log p_{ij}
+(1-y_{ij})\log(1-p_{ij})
\right]
~~~

因为只观察相对差异，所有 `r_m` 同时加上一个常数不会改变概率，实际拟合时
需要固定一个参考模型或令所有强度的均值为零。存在大量 tie 时，不能悄悄把
tie 任意归给一方；可以预先折半作近似，也可以使用显式建模平局的扩展模型。

BT 把每个模型压成一个标量强度，并假设同一对模型的胜率可由强度差的 logistic
函数解释。这便于在不完整对战图上汇总证据，却会压平任务和用户差异：模型 A
可能更受普通用户偏好，模型 B 却在专业问题上更可靠；若汇总成一个数，最终答案
取决于样本里各类用户和任务的比例。遇到这类异质性，应报告分层 BT 结果或加入
任务/用户效应，并检查不同切片的偏好是否反转。若存在循环偏好，单一标量排名
也无法完整表示对战关系。

### 6.4 对战图和可识别性

排名系统需要一张足够连通的比较图。若模型 A 只和 B 比，模型 C 只和 D 比，
即使每个局部排名都很稳定，也无法可靠比较 A 与 C。

至少要检查：

1. 每个模型的对局数量；
2. 每个模型覆盖的任务、语言和难度；
3. 是否存在连接不同模型群的桥接对局；
4. 某个模型是否只在自己擅长的 prompt 上被抽样；
5. 排名区间是否随着新增对局稳定收敛。

### 6.5 排名方法的代表方案与证据边界

几条常见路线并不是同一算法的不同名字。下表按它们主要解决的评估约束对照：

| 方案/研究 | 要解决的问题与机制 | 研究中观察到什么 | 限制与后续补强 |
| --- | --- | --- | --- |
| Elo | 新对局到来时快速更新相对 rating，适合在线维护排行榜 | 公式简单、单场结果即可更新 | 更新顺序、初始分和学习步长会影响路径；分数本身不是能力单位。报告对局量、时间窗和区间，并检查排序对参数/采样的敏感性 |
| Bradley--Terry 与 Arena 的稳健区间（[Arena 论文](https://arxiv.org/abs/2403.04132)） | 从全量 pairwise 结果估计相对强度，并为系数提供不确定性 | Arena 论文说明 BT 比早期 Elo 分数更适合其统计估计目标，并比较 bootstrap 与 sandwich 区间 | 标量强度、连接对战图和 tie 口径仍是建模假设；按任务、语言、用户分层，并同时给出原始胜负平数据 |
| MT-Bench / Chatbot Arena 人评对照（[研究](https://arxiv.org/abs/2306.05685)） | 检查开放式回答能否用多轮专家题或真实用户 pairwise 偏好区分系统 | 研究分别构建受控多轮题和众包对战，比较人工、模型 judge 的偏好一致性 | 结果绑定旧模型、特定问题集、标注者群体和一致率定义；需要在目标任务重建人工 gold 和混淆矩阵 |
| 偏好数据影响性分析（[ICLR 2026 论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/92d777fb721936613b6444b07c4ffc18-Abstract-Conference.html)） | 用 Approximate Maximum Influence Perturbation 找出少量最能改变 BT 排名的偏好记录，再对删点后的模型精确重拟合 | 在其 57,477 条人类偏好快照中，删除 2 条（0.00348%）即可翻转点估计榜首；若以 bootstrap 区间定义并列名次，翻转所需记录为 29 条（0.0510%） | 这是固定快照上的最坏情况删点，不是随机抽样波动，也不证明记录错误或遭到操纵；方法采用影响函数近似筛选，可能漏掉脆弱情形。应人工复核被识别样本，并把常规区间、最坏情况敏感性和数据质量审计分开报告 |

所以，“人评很多”不自动等于结论稳健。先确定估计对象是平均用户偏好、专家
有效性还是特定任务能力；再选择标注协议和聚合模型，报告样本分布、对战连通性、
平局规则、区间与敏感性。不同论文的榜单数字只有在模型、题目、用户、时间窗、
界面和聚合定义都相近时才能比较。

### 6.6 Arena 的数据含义

Arena 通常让真实用户或志愿评审在两个匿名回答中选择更喜欢的一方。它有机会
捕获题库难以预先设计的真实问题，也能快速积累大量偏好数据；代价是流量分布、
用户意图、评审耐心、语言比例和模型可用性都可能随时间变化。

Chatbot Arena 论文针对模型差距较小的对局采用非均匀采样，以减少估计排名所需的
投票；它还用 bootstrap 和稳健 sandwich 方法估计 BT 系数的不确定性。作者在其
213,576 条留出数据构造的重放实验中，估计胜率矩阵到 0.2 精度时，自适应采样约需
4,400 条、随机采样约需 6,800 条。这个节省量属于该数据、抽样规则和模拟设置，
不是所有评测的固定比例。[论文](https://arxiv.org/abs/2403.04132) 也明确指出，
其志愿用户偏向模型爱好者/研究者，网页聊天提示不必然代表生产业务提示，且该
研究聚焦 helpfulness、没有覆盖安全性。

该论文报告截至 2024 年 1 月约有 24 万次投票、9 万多名用户和 100 多种语言；
其中英语约占 77%，中文约占 5%，这说明题目来源广，但语言覆盖并不均衡。数字
属于论文当时的快照，不能写成当前平台规模；更不能从“有多种语言”推断每个语种
都有足够样本支持独立排名。

上述规模、语言比例和抽样细节均来自 [Arena 论文的 2024 年 1 月数据快照](https://arxiv.org/abs/2403.04132)，不是当前平台统计。

样本量和单票质量是两类不同问题。Zhao、Rush 与 Goyal 的 2024 年预印本将
Arena 的 55K 偏好快照暂时当作“真实排名”，再分别模拟冷漠用户随机投票和针对
目标模型的定向投票；在他们挑选的 3 个旧模型中，10% 的随机低质量票可让其中
2 个模型移动 5 位，10% 的定向票也造成了数位变化。这个结果是有明确假设的
压力测试：初始 55K 投票被当作 gold，目标模型只选了 3 个，而且模型对局频率
会影响脆弱性，所以它不表示当前平台真的有 10% 坏票。它指出的是一个质量控制
难题：对开放式问题，单票意见不同可能是偏好差异，不能只凭低一致率删票；可以
补充评审理由和维度标签、检查用户层行为、做过滤前后敏感性分析，并保留原始
投票供复查。[预印本](https://arxiv.org/abs/2412.04363)的攻击模拟和真实攻击
案例也应分开理解。

Arena 排名更接近“在这段时间、这批用户和这套界面中，人们更偏好哪些回答”，
而不是所有专业任务上的统一能力排序。对于安全、医学、法律、代码执行和权限
操作等高风险领域，仍需要领域样本、专家 rubric 和程序判定器。

## 7. 标注协议：把主观判断变成可重复流程

### 7.1 Rubric 的五个组成部分

一份能工作的 rubric 至少说明：

1. **任务目的**：用户想完成什么，而不是只描述文本形式；
2. **评分维度**：正确性、完整性、证据、帮助程度和安全等；
3. **锚点示例**：每个等级或每种偏好的正例、反例和边界例；
4. **优先级**：安全、事实、任务完成和风格冲突时如何处理；
5. **不确定性**：何时选择无法判断、请求专家复核或标记证据缺失。

如果规范只说“选择更有帮助的回答”，不同人会把“更长”“更礼貌”“更像自己
的表达”误认为帮助程度。可观察行为和反例比形容词重要。

rubric 和样本量规则还必须留下可复查的记录。Mei 等人对 2023—2025 年 *CL
长文本生成论文中的人工评估流程做了 284 篇人工复核，并对另外 1.8k 余篇做
辅助分析；他们发现，任务指南或评估维度选择理由等关键信息经常没有报告，样本量
功效分析和统计置信信息尤其少见。该研究测量的是论文报告规范，不能推成产品
评估团队的失败率；它说明若只留下总体胜率，读者往往无法判断评的是什么、由谁
评、以及数字应该怎样解释。[ACL 2026 研究](https://aclanthology.org/2026.acl-long.635/)
据此建议把这些设计项纳入报告，而不是只保存最终分数。

### 7.2 标注员看到什么

评审界面应只展示作出判断所需的信息：用户问题、允许的证据、候选回答和
必要的对话历史。它不应展示模型名称、版本、生成时间、价格、内部排名或
“预期答案是 A”一类暗示性信息。

对于需要外部事实的任务，要明确评审是否可以查资料。允许查资料时，必须记录
资料版本和查阅规则；不允许查资料时，题目应提供足够证据。不同规则混在一起，
会把评审员的搜索能力误当成模型质量。

### 7.3 领域专家和普通用户

普通用户更适合判断语言自然度、任务是否易用和下一步是否清楚；领域专家更
适合判断政策、医学、法律、代码、数学和安全边界。两类评价不应互相替代。

可以为同一任务保存两种标签：

~~~text
user_preference：目标用户更愿意采用哪个回答
expert_validity：领域标准下哪个回答更正确、更安全
~~~

二者不一致时，它本身就是分析结果。例如用户偏爱自信而详细的回答，但专家
发现其中存在未经证据支持的结论；产品需要解决的是信任和表达的冲突，而不是
删除其中一个标签。

### 7.4 校准样本与反馈闭环

开始正式标注前，先让所有评审独立完成一小批带参考解释的样本。讨论分歧后，
修订 rubric 和示例，再进入正式批次。校准样本不能只挑容易题，也要包含：

1. 两个答案都正确但风格不同；
2. 一个答案更长但事实错误；
3. 证据不足时应该拒答；
4. 轻微遗漏与严重安全问题；
5. 引用存在但没有支持声明。

规范修订后，要给 revision 编号，并说明哪些历史标签需要重标。否则前后批次
的分数变化可能只是标注口径改变。

### 7.5 多轮和工具任务的标注单位

单轮回答可以按一问一答评估；多轮任务不能只截取最后一句。评审需要看到：

~~~text
用户目标
历史状态
模型澄清
工具请求和参数
工具返回
中间副作用
最终状态
~~~

最终文字很礼貌，但错误地重复扣款；或者模型没有立即回答，却通过澄清问题
避免了错误操作，这两种情况都需要根据任务契约判断。把中间状态从样本中删掉，
会让人工评估无法解释工具型系统的失败。

### 7.6 争议和仲裁

当两个评审分歧时，不应只由系统随机取一个标签。低风险样本可以通过第三位
评审仲裁；高风险样本应交给领域专家或安全专家，并保存：

1. 原始标签；
2. 分歧原因；
3. 仲裁标签；
4. 使用的 rubric revision；
5. 是否修改了参考证据或任务契约。

仲裁不是为了让一致率看起来更高，而是为了识别规范中的歧义、gold 错误和
真正不存在唯一答案的任务。

Oortwijn 等人的六步流程是：形成研究问题、定义任务与标注指南、对关键概念做
评审间校准、独立标注、比较标签、再按规则解决分歧。这里“独立”很重要：校准时
可以讨论样例，但正式数据应先各自提交标签，再做仲裁；如果先让评审协商出共识，
就不能再把仲裁后的共识率当作独立评审的一致性估计。

分歧复盘可以按来源编码，而不是只记“意见不同”。Oortwijn 等人的仲裁流程将
任务/指南不清、评审者领域知识不一致、同一评审前后不一致、解释分歧和简单错误
分开处理：指南缺边界就修订并重标受影响数据；专业知识不足就更换或升级评审；
偶发误操作可纠正标签；真实价值判断不同则可能应保留多视角，而不是强行制造
唯一 gold。[其流程](https://aclanthology.org/2021.humeval-1.15/) 面向标注一致性
目标，不能机械套用到本来就要收集人群偏好的任务。

## 8. 标注一致性：分歧也是测量信号

### 8.1 简单一致率

对同一批由两位评审重复标注的样本，简单一致率是：

~~~math
A_{\mathrm{agree}}
=
\frac{N_{\mathrm{same}}}{N_{\mathrm{double}}}
~~~

它容易解释，但会受到类别分布影响。如果 95% 的样本都被标成“安全”，即使
评审员总是选择多数类，一致率也可能很高。

### 8.2 Cohen kappa 和其他指标

Cohen kappa 扣除按边际分布推算的随机一致：

~~~math
\kappa
=
\frac{p_o-p_e}{1-p_e}
~~~

`p_o` 是实际一致比例，`p_e` 是由两位评审各自标签分布计算的预期一致比例。
多于两位评审且每个样本由固定数量的人标注时，可以考虑 Fleiss' K；它常被叫作
Fleiss kappa，但 Artstein 与 Poesio 指出，该统计量更接近多标注者版 Scott pi，
并非把 Cohen kappa 直接推广到多人。标签有
不同数量的评审、存在缺失，或等级之间有可定义的距离时，Krippendorff alpha
更灵活。缺失结构或评审者/样本之间的聚类仍要单独说明，不能假设换一个系数就
消除了依赖。选择指标要随标签结构说明，不能只报一个数字。

例如，100 个安全审核样本中，评审甲标出 5 个 unsafe、95 个 safe，评审乙将
100 个都标为 safe。原始一致率是 95%，但机会一致率为
$p_e=0.95×1+0.05×0=0.95$，因此 Cohen $\kappa=0$。这是类别极不平衡时 κ 可能
显得很低的一种情形，不表示 5 个分歧样本可以忽略。报告应同时给混淆矩阵、正类
比例、原始一致率和机会校正系数；κ/α 衡量的是标注者按特定模型定义的可靠性，
不证明标签符合事实，也不等于模型答案有效。Artstein 与 Poesio 的[综述](https://aclanthology.org/J08-4004/)
系统讨论了这些系数的假设与选择边界。

### 8.3 成对比较的一致性

pairwise 标签包含 new、old、tie 和无法判断。可以先报告完全相同的比例，再
报告分歧的方向：一位认为 new 胜、另一位认为 old 胜，和一位选择 tie 的含义
不同。对于有序的“明显胜/略胜”标签，可以使用加权一致性，让明显分歧比轻微
分歧承担更大代价。

### 8.4 一致率低说明什么

低一致率可能来自四种原因：

1. rubric 没有定义关键边界；
2. 样本本身需要专业知识，但评审员不具备；
3. 候选回答质量非常接近，差异低于人的分辨能力；
4. 评审界面、上下文或证据展示不完整。

因此，不能看到低一致率就直接删掉争议样本。先按任务、语言、风险和评审员
切片，查看分歧集中在哪里；如果分歧恰好集中在政策冲突或证据不足问题，它
可能是在提醒团队补充任务契约。

同时也不能把“提高一致率”当作唯一目标。Amidei 等人在自动问题生成的人评研究
中指出，语言质量判断存在难以消除的解释差异；若为了让评审完全一致而过度收紧
规范，可能把生成目标推向更僵硬、反而不像自然语言的输出。正确动作是判断当前
评估要估计共识标签还是用户偏好分布：前者应追查可修复的协议问题，后者则可能
要保留不同人群的判断并报告分层结果。
[Rethinking the Agreement in Human Evaluation Tasks](https://aclanthology.org/C18-1281/)
提供了这一反例方向；它针对自动问题生成任务，不构成对所有低一致性的豁免。

### 8.5 标注员漂移

标注员可能因为疲劳、任务切换或规范变化而改变尺度。可以在每个批次插入
固定校准题，记录每位评审的通过率、反应时间、标签分布和与共识的差异。固定
题不能占太大比例，否则评审会记忆答案；也不能只用明显题，否则测不出边界
理解是否漂移。

## 9. 偏见控制：让比较尽量公平

人工评估不是天然客观的观察窗口。评审可能被答案的顺序、长度、格式、品牌、
语言或自信语气影响。偏差不会因为把结果称为“用户偏好”就自动消失。

### 9.1 位置偏见

如果新回答放在左侧时更容易获胜，就需要记录新回答所在位置，并计算左侧
胜出比例：

~~~math
B_{\mathrm{pos}}
=
\left|
\frac{N_{\mathrm{left\_win}}}
{N_{\mathrm{left\_win}}+N_{\mathrm{right\_win}}}
-0.5
\right|
~~~

这个指标只有在候选位置近似随机、样本量足够时才有解释力。发现偏移后，可以
重新随机展示顺序，或者对同一对候选做左右交换，比较交换前后的选择。还应把
它理解为“左右侧胜出是否不平衡”的描述信号，而不是位置偏差的因果估计：若强
候选恰好更多出现在左侧，整体左胜率会同时受候选强弱影响。较稳妥的设计是在
prompt 层随机分派顺序，并抽取一部分相同候选对做反向呈现；分别计算候选身份
在左/右时的胜率及差值，再按 prompt 聚类给区间。[LLM judge 的位置偏差研究](https://arxiv.org/abs/2406.07791)
也表明，应同时看重复稳定性、换位一致性和偏好公平性；该结论来自模型评审实验，
不能直接当作人类位置效应的量级。

### 9.2 长度和信息密度

长答案可能包含更多有用细节，也可能只是重复、跑题和未经证实的推断。若把
“完整”写成“越长越好”，评审就会形成系统性长度偏好。可以记录非平局中较长
答案获胜比例：

~~~math
B_{\mathrm{len}}
=
\frac{N_{\mathrm{longer\_winner}}}{N_{\mathrm{non\text{-}tie}}}
~~~

它不能单独证明长度偏见，因为复杂问题本来可能需要更长答案。更好的做法是
在相近长度的候选之间比较，或在回归模型中控制长度、任务难度和语言切片。

### 9.3 品牌、格式和语气

模型名称、版本、价格和已知能力会引入品牌偏见；列表、加粗和 Markdown 结构
会引入格式偏见；自信语气会让错误结论看起来可靠。匿名化只能减少一部分
问题，不能修复“事实正确性”没有进入 rubric 的根本缺陷。

### 9.4 语言和领域偏见

同一套中文 rubric 不能未经检查地用于英语、方言、代码和专业术语。评审员
熟悉某种表达方式，也可能偏爱它而不是更准确的另一种表达。应按语言和领域
分配合适评审，分别报告切片结果，并检查翻译、术语和文化语境是否改变了
任务难度。

### 9.5 安全与拒答偏见

评审可能把“敢于回答”误认为有帮助，也可能把所有拒答都认为安全。规范要给
出允许回答、应该澄清、应该拒绝和必须转人工的边界。安全评估还要分辨：

1. 保护用户免受危险建议；
2. 防止泄露内部信息；
3. 防止工具产生未授权副作用；
4. 维持良性请求的可用性。

高安全性不能通过让模型拒绝所有请求获得。安全标签和正常任务质量应分别
统计，然后共同参与发布讨论。

## 10. 样本量、抽样和成本

### 10.1 先确定要检测的差异

样本量要按要检测的最小差异、置信水平、检验功效、平局比例和抽样结构规划。
这里把每个独立 prompt 的偏好分数记作 `z_i\in\{0,0.5,1\}`，比较其总体均值与
无偏好基线 0.5。一个便于初步预算的正态近似是：

~~~math
N
\approx
\frac{(z_{1-\alpha/2}\sigma_0+z_{1-\beta}\sigma_1)^2}{\delta^2}
~~~

`\delta` 是相对 0.5 的最小重要差异，`\sigma_0` 和 `\sigma_1` 分别是原假设与
备择假设下 prompt 级分数的标准差；应尽量由 pilot 数据估计。比如要检测 5 个
百分点差异，双侧 `\alpha=0.05`、功效 80%，并用 `[0,1]` 变量的保守上界
`\sigma_0=\sigma_1=0.5`，粗算约需 784 个独立 prompt。这个数字不是通用配额：
若重复评审共享 prompt 或用户，需按聚类设计扩大；若抽样分层或样本权重不等，
应按实际设计重算。若终点是两个模型对同一题的二元正确/错误率，规划应使用
配对检验（例如依赖不一致对的 McNemar 设计），不能直接套用偏好均值公式。

### 10.2 成对样本优先于独立样本

比较两个模型时，让它们回答同一批 prompt，通常比让两个模型各自回答不同
问题更有效，因为题目难度被配对抵消。对每个 prompt 保存两份回答和同一个
任务契约，分析时以 prompt 为重采样单位。

若同一个用户贡献了许多相似请求，仍然要按用户或会话聚类。否则一个高活跃
用户的偏好可能被错误地当成很多独立用户的证据。

### 10.3 分层抽样和总体估计

设切片 `g` 在真实流量中的比例为 `p_g`，切片内观察到的新模型偏好率为
`\hat p_g`，按自然分布估计总体偏好可以写成：

~~~math
\hat p_{\mathrm{population}}
=
\sum_g p_g\hat p_g
~~~

如果为了发现高风险问题而过采样某个切片，样本中的原始比例就不是 `p_g`；
要么用已知流量比例加权，要么把结果明确称为风险切片表现，不能直接当线上
平均偏好。

### 10.4 人工评估的成本账本

设样本数为 `N`，每个样本需要 `r` 位评审，每次评审耗时 `t` 分钟，单位人力
成本为 `c`，还要加上培训、仲裁和平台成本 `C_{\mathrm{fixed}}`，则粗略预算为：

~~~math
C_{\mathrm{human}}
\approx
Nr\frac{t}{60}c + C_{\mathrm{fixed}}
~~~

这里把 `t` 定义为分钟、`c` 定义为每小时成本，所以公式中要除以 60；若把 `t`
改成小时，应同时去掉这个换算。公式看似简单，却能揭示一个常见误区：把每个样本多交给一名评审，成本几乎
线性增加，而一致性收益可能只在争议切片出现。

### 10.5 自适应复审

可以先用程序判定器和自动 judge 找出高分歧、边界和高风险样本，再把人工预算
集中到这些样本。但样本筛选器本身会改变评估分布，因此最终报告应保留：

1. 自然分布样本的人工结果；
2. 经过筛选的困难样本结果；
3. 筛选规则、阈值和版本；
4. 哪些样本从未进入人工复审。

自适应复审适合发现问题，不适合单独估计全体用户的平均质量。

## 11. 生产化评估链路

### 11.1 从数据到报告

一个可复现的人工评估链路通常包含：

~~~text
样本抽取
  -> 固定模型与运行协议生成候选
  -> 去除模型身份信息并随机展示顺序
  -> 分配评审员和校准样本
  -> 收集标签、理由和反应状态
  -> 处理重复标注与争议
  -> 聚合总体和切片结果
  -> 分析不确定性、偏差和成本
  -> 形成发布、回退或继续调查的判断
~~~

每一步都应产生有版本的 artifact，而不是最后只保存一张排行榜截图。

### 11.2 生成阶段的 manifest

在评审开始前固定：

~~~text
model_revision
tokenizer_revision
system_prompt_revision
decoding_revision
retriever_revision
tool_revision
context_snapshot
random_seed
generation_timestamp
~~~

模型输出也要保留原始文本、结构化解析结果、token usage、延迟和错误状态。若
评审员看到的是经过后处理的文本，需要同时记录后处理版本；否则文本清洗可能
被误认为模型能力提升。

### 11.3 评审阶段的 schema

一个成对比较标签可以写成：

~~~text
item_id
annotator_id
candidate_left_model
candidate_right_model
preference: left | right | tie | unclear
reason_codes
dimension_labels
confidence
review_duration_ms
rubric_revision
created_at
~~~

`candidate_left_model` 和 `candidate_right_model` 在分析数据库中可见，但在评审
界面中应被替换为匿名标识。`unclear` 不应被自动折算成 tie，它反映的是评审
信息不足或任务边界不清。

### 11.4 质量控制的三层

第一层是输入质量：样本脱敏、证据版本、问题重复和任务切片正确。第二层是
评审质量：校准样本、重复样本、反应状态和一致性。第三层是结果质量：分母、
权重、置信区间、切片和原始标签可追溯。

任何一层缺失，最终数字都可能看起来精确，却无法回答“这个差异从哪里来”。

可复现报告至少要能回答五件事：测量的构念及任务指南是什么；评审者怎样招募、
筛选和培训；样本如何抽取、回答如何匿名和随机展示；重复标注、低质量票和争议
如何处理；最后采用什么聚合、权重、区间或功效假设。Mei 等人的 2026 年研究
在其论文样本中发现，这些报告项有系统性缺漏；因此这五项既是团队内部复核清单，
也是读者解释研究结果所需的最小上下文。具体清单仍应依任务类型调整，不能把
单一论文模板当成所有人评的强制标准。

## 12. 企业客服系统的完整案例

### 12.1 问题背景

一家企业准备把旧版客服模型替换为新版模型。新版在普通问答中更简洁，但
团队担心它在退款政策、无答案问题和高价值用户投诉上过度自信。目标不是
寻找一个漂亮的总分，而是判断新版是否改善用户任务，同时没有增加危险行为。

### 12.2 样本构成

评估集按业务切片组成：

| 切片 | 样本来源 | 要观察的行为 |
| --- | --- | --- |
| 普通 FAQ | 脱敏线上请求 | 事实、清楚和下一步 |
| 历史 bad case | 事故和工单 | 是否复现旧失败 |
| 政策边界 | 专家构造和版本冲突 | 证据版本与条件 |
| 无答案 | 线上无命中请求 | 正确拒答和澄清 |
| 高价值投诉 | 客服升级记录 | 语气、权限和转人工 |
| 多语言 | 不同地区请求 | 术语和文化语境 |

样本元数据保存用户角色、文档 revision、语言、问题难度、风险等级和是否有
gold evidence。旧模型和新模型使用同一 prompt、同一检索结果和同一工具权限，
这样成对比较的差异更接近模型或解码变化。

### 12.3 评分优先级

评审先判断安全和权限，再判断事实与证据，最后判断完整性、行动性和表达：

~~~text
安全/权限
  > 事实正确
  > 证据支撑
  > 任务完整
  > 下一步可执行
  > 简洁和语气
~~~

若新模型回答更完整，却把旧政策当成当前政策，应在原因标签中记录“版本证据
错误”，而不是让“更完整”与“更错误”互相抵消。

### 12.4 读取总体结果

假设 1000 个成对判断得到：新模型 420 胜、旧模型 350 胜、230 平。折半平局
后的偏好率为 0.535。这个结果说明新模型在这套样本和协议下有轻微优势，不能
单独支持全面替换。

还要看切片。假设结果如下：

| 切片 | 新模型偏好率 | 样本量 | 解释 |
| --- | ---: | ---: | --- |
| 普通 FAQ | 0.61 | 500 | 主要收益来源 |
| 政策边界 | 0.48 | 150 | 检查检索与版本提示 |
| 无答案 | 0.45 | 120 | 检查拒答和澄清 |
| 历史 bad case | 0.50 | 130 | 没有证据证明旧问题已修复 |
| 高价值投诉 | 0.56 | 100 | 需结合专家安全标签 |

总体优势可能完全由普通 FAQ 贡献。若无答案切片的错误回答率上升，即使总体
偏好率提高，也应先调查拒答策略和知识库覆盖，而不是让大切片掩盖小切片风险。

### 12.5 从分歧找到 rubric 问题

如果普通用户偏爱新版的详细解释，而政策专家频繁标记“证据过期”，说明两类
评估测量的是不同构念。团队可以把“政策版本正确”提升为独立维度，同时给无
答案问题增加“明确说明不知道”的行为标签。下一次评估不必删掉原有用户偏好，
而要同时保留易用性和领域有效性。

### 12.6 发布判断的表达

更严谨的结论可以写成：

~~~text
新版在普通 FAQ 的用户偏好上有改善，但在政策边界和无答案切片没有形成稳定
优势；高价值投诉需要专家复核，当前证据不足以支持全量替换。可以继续收集
边界样本，先在普通 FAQ 做受控流量实验，并把政策版本错误和错误拒答作为独立
回归指标。
~~~

这种结论比“新版胜率超过 50%，所以发布”更接近实际工程决策，因为它同时说明
收益在哪里、证据缺什么、风险如何处理。

## 13. 一个可运行的人工评估诊断 demo

下面的 toy demo 不调用模型，只处理一组已经产生的人工标签。它计算绝对评分、
成对偏好及按 prompt 聚合的粗略标准误、任务切片、一致性、位置和长度信号，并把需要调查的事项逐条列出。
示例故意放入一个高风险 RAG 回退、一个位置偏移和一个分歧样本，读者可以看到
总体偏好较高时，为什么仍然需要继续调查。

代码中的 `0.70`、`0.53` 和 `0.20` 是为了让这组教学数据触发不同诊断分支而设的
演示阈值，不是跨任务通用的一致性标准、发布标准或统计显著性阈值。真实项目应
根据风险等级、样本设计、聚类结构和历史基线预先确定判定规则，并同时报告原始
样本量和不确定性区间；不能因为某个布尔信号为 `True` 就跳过人工复核。

~~~python
import math
from pprint import pprint


rubric = {
    "dimensions": ["safety", "correctness", "evidence", "helpfulness", "style"],
    "priority_order": ["safety", "correctness", "evidence", "helpfulness", "style"],
    "tie_rule": True,
    "anonymized": True,
    "position_randomized": True,
    "min_annotators": 2,
    "gold_samples": 2,
}

items = {
    "qa_001": {"task": "qa", "risk": "normal", "old_len": 90, "new_len": 96},
    "rag_001": {"task": "rag", "risk": "high", "old_len": 130, "new_len": 150},
    "code_001": {"task": "code", "risk": "normal", "old_len": 100, "new_len": 105},
    "safe_001": {"task": "safety", "risk": "high", "old_len": 130, "new_len": 125},
    "summary_001": {"task": "summary", "risk": "normal", "old_len": 80, "new_len": 220},
    "math_001": {"task": "math", "risk": "normal", "old_len": 90, "new_len": 115},
}

pairwise = [
    {"item": "qa_001", "annotator": "ann_a", "winner": "new", "new_pos": "left"},
    {"item": "qa_001", "annotator": "ann_b", "winner": "new", "new_pos": "right"},
    {"item": "rag_001", "annotator": "ann_a", "winner": "old", "new_pos": "left"},
    {"item": "rag_001", "annotator": "ann_b", "winner": "old", "new_pos": "right"},
    {"item": "code_001", "annotator": "ann_a", "winner": "new", "new_pos": "left"},
    {"item": "code_001", "annotator": "ann_b", "winner": "tie", "new_pos": "right"},
    {"item": "safe_001", "annotator": "ann_a", "winner": "new", "new_pos": "left"},
    {"item": "safe_001", "annotator": "ann_b", "winner": "new", "new_pos": "left"},
    {"item": "summary_001", "annotator": "ann_a", "winner": "new", "new_pos": "left"},
    {"item": "summary_001", "annotator": "ann_b", "winner": "new", "new_pos": "left"},
    {"item": "math_001", "annotator": "ann_a", "winner": "old", "new_pos": "right"},
    {"item": "math_001", "annotator": "ann_b", "winner": "new", "new_pos": "left"},
]

ratings = [
    ("qa_001", "old", 3), ("qa_001", "old", 3), ("qa_001", "new", 4), ("qa_001", "new", 4),
    ("rag_001", "old", 4), ("rag_001", "old", 4), ("rag_001", "new", 3), ("rag_001", "new", 3),
    ("code_001", "old", 3), ("code_001", "old", 3), ("code_001", "new", 4), ("code_001", "new", 3),
    ("safe_001", "old", 2), ("safe_001", "old", 3), ("safe_001", "new", 4), ("safe_001", "new", 4),
    ("summary_001", "old", 3), ("summary_001", "old", 3), ("summary_001", "new", 4), ("summary_001", "new", 4),
    ("math_001", "old", 4), ("math_001", "old", 3), ("math_001", "new", 4), ("math_001", "new", 4),
]


def mean(values):
    return sum(values) / len(values) if values else 0.0


rating_by_model = {"old": [], "new": []}
for _, model, score in ratings:
    rating_by_model[model].append(score)

rating_summary = {
    "old": round(mean(rating_by_model["old"]), 3),
    "new": round(mean(rating_by_model["new"]), 3),
}
rating_summary["delta"] = round(rating_summary["new"] - rating_summary["old"], 3)

new_wins = sum(row["winner"] == "new" for row in pairwise)
old_wins = sum(row["winner"] == "old" for row in pairwise)
ties = sum(row["winner"] == "tie" for row in pairwise)
total = len(pairwise)
win_rate = (new_wins + 0.5 * ties) / total

scores_by_item = {}
labels_by_item = {}
for row in pairwise:
    item_id = row["item"]
    score = {"new": 1.0, "tie": 0.5, "old": 0.0}[row["winner"]]
    scores_by_item.setdefault(item_id, []).append(score)
    labels_by_item.setdefault(item_id, []).append(row["winner"])

item_scores = {item_id: mean(scores) for item_id, scores in scores_by_item.items()}
item_level_values = list(item_scores.values())
item_level_mean = mean(item_level_values)
if len(item_level_values) > 1:
    item_level_variance = sum(
        (score - item_level_mean) ** 2 for score in item_level_values
    ) / (len(item_level_values) - 1)
    item_level_se = math.sqrt(item_level_variance / len(item_level_values))
else:
    item_level_se = None

slice_scores = {}
for item_id, score in item_scores.items():
    task = items[item_id]["task"]
    slice_scores.setdefault(task, []).append(score)
slice_win_rate = {
    task: round(mean(scores), 3)
    for task, scores in slice_scores.items()
}

agreements = [
    len(set(labels)) == 1
    for labels in labels_by_item.values()
    if len(labels) >= 2
]
agreement_rate = mean(agreements)

left_wins = 0
right_wins = 0
longer_wins = 0
non_tie = 0
for row in pairwise:
    if row["winner"] == "tie":
        continue
    item = items[row["item"]]
    non_tie += 1
    if row["winner"] == "new":
        winner_pos = row["new_pos"]
        winner_len = item["new_len"]
        loser_len = item["old_len"]
    else:
        winner_pos = "right" if row["new_pos"] == "left" else "left"
        winner_len = item["old_len"]
        loser_len = item["new_len"]
    left_wins += winner_pos == "left"
    right_wins += winner_pos == "right"
    longer_wins += winner_len > loser_len

left_win_rate = left_wins / non_tie
position_bias = abs(left_win_rate - 0.5)
longer_win_rate = longer_wins / non_tie

high_risk_failures = [
    item_id
    for item_id, item in items.items()
    if item["risk"] == "high" and item_scores[item_id] < 0.5
]

required_rubric_fields = {
    "dimensions",
    "priority_order",
    "tie_rule",
    "anonymized",
    "position_randomized",
    "min_annotators",
    "gold_samples",
}
rubric_missing = sorted(required_rubric_fields - set(rubric))

signals = {
    "rubric_defined": (
        not rubric_missing
        and bool(rubric["dimensions"])
        and len(rubric["dimensions"]) == len(set(rubric["dimensions"]))
        and len(rubric["priority_order"]) == len(rubric["dimensions"])
        and set(rubric["priority_order"]) == set(rubric["dimensions"])
        and rubric["tie_rule"] is True
        and rubric["anonymized"] is True
        and rubric["position_randomized"] is True
        and rubric["min_annotators"] >= 1
        and rubric["gold_samples"] >= 1
    ),
    "sample_count_ok": len(items) >= 5,
    "annotator_agreement_ok": agreement_rate >= 0.70,
    "overall_win_rate_ok": win_rate >= 0.53,
    "high_risk_slice_ok": not high_risk_failures,
    "position_bias_ok": position_bias <= 0.20,
    "length_bias_ok": longer_win_rate <= 0.70,
    "gold_samples_available": rubric["gold_samples"] >= 2,
}

actions = []
if not signals["rubric_defined"]:
    actions.append("complete_or_revise_rubric")
if not signals["sample_count_ok"]:
    actions.append("collect_more_demo_items")
if not signals["annotator_agreement_ok"]:
    actions.append("review_disagreement_cases")
if not signals["overall_win_rate_ok"]:
    actions.append("inspect_overall_estimate_and_uncertainty")
if not signals["high_risk_slice_ok"]:
    actions.append("investigate_high_risk_regressions")
if not signals["position_bias_ok"]:
    actions.append("rebalance_answer_positions")
if not signals["length_bias_ok"]:
    actions.append("audit_length_and_task_mix")
if not signals["gold_samples_available"]:
    actions.append("add_calibration_gold_samples")

decision = "revise_evaluation" if actions else "continue_to_report"
summary = {
    "rating_summary": rating_summary,
    "pairwise": {
        "new_wins": new_wins,
        "old_wins": old_wins,
        "ties": ties,
        "win_rate": round(win_rate, 3),
        "tie_rate": round(ties / total, 3),
        "item_level_count": len(item_level_values),
        "item_level_se_iid": round(item_level_se, 3) if item_level_se is not None else None,
    },
    "slice_win_rate": slice_win_rate,
    "item_agreement": round(agreement_rate, 3),
    "position_bias": {
        "left_win_rate": round(left_win_rate, 3),
        "bias": round(position_bias, 3),
    },
    "longer_win_rate": round(longer_win_rate, 3),
    "high_risk_failures": high_risk_failures,
    "rubric_missing": rubric_missing,
    "signals": signals,
    "actions": actions,
    "decision": decision,
}

pprint(summary, sort_dicts=False)
~~~

输出为：

~~~text
{'rating_summary': {'old': 3.167, 'new': 3.75, 'delta': 0.583},
 'pairwise': {'new_wins': 8,
              'old_wins': 3,
              'ties': 1,
              'win_rate': 0.708,
              'tie_rate': 0.083,
              'item_level_count': 6,
              'item_level_se_iid': 0.164},
 'slice_win_rate': {'qa': 1.0,
                    'rag': 0.0,
                    'code': 0.75,
                    'safety': 1.0,
                    'summary': 1.0,
                    'math': 0.5},
 'item_agreement': 0.667,
 'position_bias': {'left_win_rate': 0.818, 'bias': 0.318},
 'longer_win_rate': 0.545,
 'high_risk_failures': ['rag_001'],
 'rubric_missing': [],
 'signals': {'rubric_defined': True,
             'sample_count_ok': True,
             'annotator_agreement_ok': False,
             'overall_win_rate_ok': True,
             'high_risk_slice_ok': False,
             'position_bias_ok': False,
             'length_bias_ok': True,
             'gold_samples_available': True},
 'actions': ['review_disagreement_cases',
             'investigate_high_risk_regressions',
             'rebalance_answer_positions'],
 'decision': 'revise_evaluation'}
~~~

注意，demo 使用的是教学数据，不是生产质量证明。它的作用是让读者看到一条
具体的分析链：总体 win rate 为 0.708，但只有 6 个 prompt；先将每个 prompt 下的
两位评审得分取平均，再让六个 prompt 等权，iid 标准误已经是 0.164。当前示例
每题的评审人数相同，因此 prompt 均值与逐票总均值一致；若标注数不等或使用
分层权重，应先定义目标估计量并调整标准误。绝对评分均值虽提高，`rag_001` 仍在高风险切片中回退，
评审一致率只有 0.667，且左侧胜出比例过高。这个标准误还没有处理评审员复用、
用户聚类或切片权重；样本这么少也不适合靠正态近似作发布判断。后续动作应分别
复核分歧样本、高风险回退并重新平衡答案位置，而不是把点估计阈值写成上线门禁。

## 14. Human Eval 与 LLM-as-a-Judge

### 14.1 两者的分工

LLM-as-a-Judge 可以快速读取大量回答，按 rubric 给出分数、偏好和理由；人工
评审更昂贵，却能在高风险、争议、专业知识和新任务上发现 judge 没有学会的
错误。二者不应被理解为“人工是旧方法、judge 是自动替代品”。更准确的关系是：

~~~text
人工评审
  -> 提供高质量标签和边界案例
  -> 校准 judge 的维度、偏差和阈值
  -> 复核高风险和高分歧样本

LLM judge
  -> 扩大覆盖
  -> 运行快速回归
  -> 按不确定性挑选人工复审样本
~~~

### 14.2 先测 judge 与人工的关系

不要只计算 judge 和人工的总体相关系数。应按任务、语言、长度、风险、模型
版本和错误类型切片；同时查看 judge 是否偏爱更长回答、是否受候选展示顺序
影响、是否能识别引用不支持和安全违规。

对于 pairwise judge，可以保存人工标签 `y_h`、judge 标签 `y_j` 和不确定状态，
构造混淆矩阵。对于绝对评分，可以报告等级差、加权一致性和严重错误的漏报率。
平均相关很高，仍可能掩盖“普通 FAQ 正常、高风险样本全部漏报”的情况。

研究基线只能作为方法证据，不能当作自动 judge 的验收线。Zheng 等人在 MT-Bench
的受控实验中使用 80 个多轮问题、58 位专家级标注者并收集约 3,000 票；其特定
协议下 GPT-4 和人类非平局判断一致率为 85%，人类之间为 81%。这说明当时的强模型
judge 在该任务上接近人工间一致性，不说明 judge 与真实用户偏好一致，更不保证
目标领域、安全切片或新版本上仍达到同一水平。目标系统仍需独立人工校准样本和
按风险切片的误差分析。[论文中的设置与结果](https://arxiv.org/abs/2306.05685)
应连同问题集、模型和是否排除平局一起阅读。

### 14.3 什么时候必须保留人工

下列场景不宜只交给 LLM judge：

1. 可能造成财务、医疗、法律或人身风险的回答；
2. 需要判断新政策、专业规范或权限边界的回答；
3. 新任务没有稳定 rubric 和人工 gold set；
4. judge 与模型使用相似训练来源，可能共享错误或污染；
5. 评估结果会改变大规模用户流量、数据权限或不可逆工具动作。

人工预算有限时，可以只让人工覆盖这些场景，并用 judge 处理低风险、稳定且
有程序判定器辅助的回归样本。

### 14.4 与下一章的关系

下一章会单独讨论 LLM-as-a-Judge 的评分协议、位置和长度偏差、reference
使用、judge ensemble、校准和失效模式。本章先建立人工标签和成对比较的测量
基础，因为没有可靠的人类参照，就无法知道 judge 到底在复现什么。

## 15. 常见失败模式

### 15.1 用一个总分代替任务契约

总分上升可能来自低风险、简单或高流量任务；关键政策、拒答和工具安全却可能
回退。必须同时看切片、原始标签和风险等级。

### 15.2 把“更长”当成“更好”

长度能增加解释和证据，也能增加错误声明和阅读负担。rubric 应评估信息密度、
相关性和事实支持，而不是 token 数量本身。

### 15.3 强迫二选一

两个答案都正确时强迫选择会制造随机噪声；两个答案都危险时强迫选择又会掩盖
共同失败。允许 tie、无法判断和独立安全标签，才能表达实际情况。

### 15.4 只看单次标注结果

没有重复标注，无法估计一致性；没有保存评审理由，无法修订 rubric；没有保存
模型身份映射和展示顺序，无法分析位置或品牌偏见。

### 15.5 让公开 Arena 排名代表全部能力

真实用户偏好很有价值，但它受流量、界面、语言、时段和参与者影响。专业任务
仍需受控样本、专家评审和执行判定器，Arena 只能作为证据的一层。

### 15.6 评审协议随模型变化

如果新模型使用更长的输出预算、不同的工具、不同的上下文或不同的后处理，
pairwise 结果测量的是新系统，不是“同条件下的模型差异”。每次协议变更都要
生成新的 revision，并在报告中明确不能与旧结果直接拼接。

## 16. 练习：把偏好变成证据

### 练习一：设计一个 RAG 成对评估

为企业知识库问答写出样本 schema、证据字段、候选生成协议、rubric、tie 与
无法判断规则。要求至少包含一个无答案样本和一个版本冲突样本。

### 练习二：分析总体提升与切片回退

一个新模型在 800 个普通问题上偏好率为 0.58，在 100 个政策问题上偏好率为
0.44。假设真实流量比例分别为 90% 和 10%，计算自然分布加权偏好率，并说明
为什么这个数字仍不足以替代政策安全分析。

### 练习三：计算位置偏差

记录候选随机化后的左侧胜出 132 次、右侧胜出 98 次、平局 70 次。按照非平局
样本计算位置偏移，并讨论为什么还需要检查随机化是否真正均衡。

### 练习四：处理低一致性

两位评审在一个法律问答切片上的 kappa 很低，但在普通 FAQ 上很高。列出至少
三种可能原因，并设计一轮重标或专家仲裁方案。

### 练习五：人工与 judge 的组合

给出一个预算固定的评估计划：哪些样本由程序判定，哪些由 LLM judge 扩展，
哪些必须由领域专家审阅；说明如何用人工 gold set 检查 judge 的漏报。

## 17. 资料与证据边界

人工评估论文通常描述特定任务、样本、标注员和统计协议。它们能支持方法为何
有效、偏差如何出现以及在给定实验条件下的结果，不能自动证明某种 rubric 在
所有产品和语言中都可靠。

1. [InstructGPT](https://arxiv.org/abs/2203.02155)：展示人类示范、偏好数据和 RLHF 如何用于对齐评估目标。
2. [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)：说明摘要任务中的人类偏好收集与奖励建模。
3. [HELM](https://crfm.stanford.edu/helm/latest/) 与 [HELM paper](https://arxiv.org/abs/2211.09110)：提供多场景、多指标评估的框架和实验背景。
4. [MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)：讨论多轮开放式对话的人类 pairwise 标注、LLM judge 比较及二者的一致性；具体一致率绑定其问题和投票协议。
5. [Chatbot Arena](https://arxiv.org/abs/2403.04132)：介绍基于用户偏好的开放平台、BT 排名、区间估计和自适应采样；论文公开数据快照及用户群体有明确时间范围。
6. [Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge](https://arxiv.org/abs/2406.07791)：用重复与换位实验拆分 judge 的位置稳定性和偏好公平性；只支持所测 judge、任务和题集。
7. [Best Practices for the Human Evaluation of Automatically Generated Text](https://aclanthology.org/W19-8643/)：综述 NLG 人评中的任务定义、标注和报告实践；覆盖早期生成研究，需结合 LLM 开放任务调整。
8. [The Authenticity Gap in Human Evaluation](https://aclanthology.org/2022.emnlp-main.406/)：分析 Likert 汇总与偏好排序的隐含假设，并提出 SPA；理论结论绑定作者的效用设定。
9. [Rethinking the Agreement in Human Evaluation Tasks](https://aclanthology.org/C18-1281/) 与 [Interrater Disagreement Resolution](https://aclanthology.org/2021.humeval-1.15/)：分别讨论自然语言评价中不可消除的分歧，以及将可修复分歧按来源仲裁的流程。
10. [Inter-Coder Agreement for Computational Linguistics](https://aclanthology.org/J08-4004/)：系统比较 Cohen kappa、Krippendorff alpha 等一致性系数的数学假设和适用条件。
11. [Illusions of the Gold Standard: A Large-scale Analysis of Human Evaluation Protocols for Long-form Text Generation](https://aclanthology.org/2026.acl-long.635/)：人工审查 284 篇、辅助分析 1.8k 余篇 2023—2025 年 *CL 长文本生成人评论文，总结复现信息的系统性缺漏；结论绑定作者论文样本和报告标准。
12. [Challenges in Trustworthy Human Evaluation of Chatbots](https://arxiv.org/abs/2412.04363)：以 55K Arena 快照进行随机差票、定向投票和主观题分歧的压力分析；10% 票数模拟变化不能解释为实际坏票率，原快照被假设为真实排序。
13. [Dropping Just a Handful of Preferences Can Change Top Large Language Model Rankings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/92d777fb721936613b6444b07c4ffc18-Abstract-Conference.html)：ICLR 2026 论文用 AMIP 近似筛出高影响偏好并精确重拟合；结果绑定作者公开平台快照，最坏情况删点不等同于随机抽样置信区间。
14. [OpenAI Evals](https://github.com/openai/evals)：提供可扩展评估任务和运行框架入口。
15. [Hugging Face Evaluate](https://huggingface.co/docs/evaluate/index)：提供指标和评估脚本的实现入口。

这些资料的证据类型并不相同：论文给出方法和实验口径，平台文档说明实现入口，
真实业务的脱敏样本才说明某个产品遇到了什么问题。面对新模型的公开偏好榜单，
还应记录模型 revision、采样时间、用户分布、展示界面、对战抽样、是否允许工具
和不确定性区间；没有这些信息的排名适合作为线索，不适合作为跨任务能力承诺。

## 18. 结语：人工判断的价值在于可解释

Human eval 的价值不是给机器加上一层“人类感觉”，而是把开放式任务中无法用
单一字符串判定的质量，拆成可讨论、可复查、可统计的行为。绝对评分回答质量
画像，pairwise 回答相对选择，Elo 或 Bradley--Terry 聚合对战关系，专家与普通
用户标签则说明“谁认为更好”以及“为什么更好”。

真正可靠的评估报告不会只留下一个总胜率。它会保留任务契约、候选生成协议、
匿名化和随机化记录、rubric revision、原始标签、分歧、切片、成本和高风险样本。
这样，当新模型发布、prompt 改变或产品流量变化时，团队仍能判断过去的数字
是否可比，以及下一步应该收集什么证据。
