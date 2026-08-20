# 第二章：Chain-of-Thought

Chain-of-Thought（CoT）常被翻译成思维链。它最容易被误解的地方，是人们把三个不同问题混在了一起：

1. 模型是否需要中间计算，才能完成一个任务？
2. 模型生成的中间文字，是否帮助了这次计算？
3. 这些文字是否忠实地记录了真正影响答案的内部过程？

第一个问题关心计算能力，第二个问题关心生成接口，第三个问题关心解释的因果忠实性。一个系统可以在第二个问题上有效，却不能证明第三个问题；也可以不展示文字过程，却在隐藏状态或工具轨迹中完成了多步计算。

本章把 CoT 放回大模型的发展脉络中讲清楚：为什么直接回答会在多步任务上遇到瓶颈，few-shot CoT 和 zero-shot CoT 分别改变了什么，scratchpad 与面向用户的解释有什么差别，训练数据怎样把错误过程传给模型，以及如何用干预、验证器、工具和成本曲线判断 CoT 是否真的值得使用。

## 0.1 本章的资料与证据边界

本章的历史主线来自 Chain-of-Thought Prompting、Zero-shot Reasoners、Show Your Work: Scratchpads for Intermediate Computation、Faithful Chain-of-Thought Reasoning、Language Models Don't Always Say What They Think 等论文。它们分别支持 CoT 的提示现象、scratchpad 作为中间计算、忠实性讨论和反例分析。

这些论文通常在特定模型、数据集和提示协议上报告实验。它们能说明“某种方法在某种条件下有效”，不能自动证明所有模型都采用同样的内部机制。尤其是闭源模型的隐藏推理、训练数据、奖励模型和服务端路由，除非官方公开，否则只能写成未知或待核验。

本章还会讨论公开产品中的 thinking、extended thinking 或 reasoning 预算。官方文档可以支持接口层的参数语义和展示策略，但不能支持对内部完整思维链的猜测。产品返回的摘要、引用或解释，也不应被默认理解成隐藏状态的逐字转录。

## 0.2 先看一个最小例子

题目是：

~~~text
一个水箱每分钟进水 3 升，每分钟漏水 1 升，容量 20 升，从空开始多久装满？
~~~

直接回答可能是：

~~~text
10 分钟。
~~~

带有中间步骤的回答可能是：

~~~text
净流入速度是 3 - 1 = 2 升/分钟。
水箱需要装入 20 升，因此时间是 20 / 2 = 10 分钟。
~~~

第二种形式有两个潜在价值。第一，它给模型更多 token 位置保存净流入速度这个中间变量；第二，它让外部程序或读者有机会检查单位和算术。但这两点都不保证模型一定完成了正确计算。模型也可能先猜到 10，再写出一段看起来合理的说明。

这个例子还展示了 CoT 的适用边界。若题目只是“1 + 1 等于多少”，长篇推导没有必要；若题目涉及多个条件、单位转换、程序执行或规划状态，中间状态才更有价值。

## 1. CoT 到底是什么

### 1.1 小白视角：给模型一张草稿纸

人解决多步问题时通常会写草稿。草稿不是最终答案，也不一定要写成完整文章，但它可以保存暂时的变量、假设和计算结果，避免每一步都依赖短暂的记忆。

CoT 的表面形式类似于让模型在最终答案之前生成草稿。模型先输出若干中间步骤，再输出结论。对语言模型来说，这些步骤仍然是 token；它们同时是输出内容和后续生成的上下文。

### 1.2 专家视角：把步骤和答案写成联合序列

设输入为 x，中间轨迹为 z，最终答案为 y。一个抽象的联合分布可以写成：

~~~math
p_{\theta}(z,y\mid x)=p_{\theta}(z\mid x)\,p_{\theta}(y\mid x,z)
~~~

这里的分解是为了说明变量关系，不代表所有实现都显式建立一个独立的 z 模块。对普通自回归模型来说，z 和 y 可能只是同一条序列中的不同区段；对工具增强系统来说，z 还可以包括程序、检索结果和环境观察。

如果把步骤与答案拼成一个序列 u，生成仍然遵循：

~~~math
p_{\theta}(u_{1:L}\mid x)=\prod_{t=1}^{L}p_{\theta}(u_t\mid x,u_{1:t-1})
~~~

CoT 没有自动把语言模型变成符号数学引擎。它提供了一种中间状态表达方式，让模型可以在后续 token 中继续使用已经生成的变量和关系。

### 1.3 CoT 不是“解释长度”指标

以下三件事不能互相替代：

1. 轨迹长度：生成了多少 token。
2. 过程质量：关键步骤是否正确、相关、完整。
3. 任务成功率：最终结果是否满足任务约束。

如果模型重复三遍同一句话，轨迹长度增加了，但没有增加有效状态。若模型写了很短的程序并通过隐藏测试，短轨迹也可能比长解释更可靠。工程上应报告有效步骤、最终结果和独立验证，而不能用字数代替推理能力。

## 2. 为什么直接回答会遇到困难

### 2.1 直接回答把多个决策压到一次输出

面对多步题，模型需要完成读题、抽取条件、选择操作、保持中间变量、检查结果和格式化答案。如果直接要求最终答案，这些过程都被压缩到同一个生成任务里，错误发生后很难定位。

以水箱问题为例，至少要维护：

1. 进水速率；
2. 漏水速率；
3. 净速率；
4. 容量；
5. 时间单位。

任何一个变量被覆盖或单位被忽略，最后答案都会受到影响。

### 2.2 直接回答也可能是最优选择

不能因此得出“所有任务都应该写 CoT”。简单事实、短分类、格式转换和低延迟接口可能不需要显式中间步骤。额外生成会带来 token 成本、延迟、格式风险和不必要的解释。

正确问题不是“CoT 永远好不好”，而是：给定任务分布、模型、预算和安全约束，显式中间状态是否带来足够的质量收益。

### 2.3 从下一个 token 到中间计算

语言模型的基础训练目标通常是最小化下一个 token 的负对数似然：

~~~math
\mathcal{L}_{\mathrm{LM}}(\theta)
=-\sum_{t=1}^{T}\log p_{\theta}(y_t\mid x,y_{1:t-1})
~~~

这个目标没有单独写出“请先做数学推导”。但训练语料中包含解题过程、程序、证明和结构化说明，模型可能学会用一系列 token 维持中间变量，再继续预测结论。

CoT 提示改变的是生成轨迹的形式和上下文。它不改变基础概率模型的事实，也不保证每一步都满足外部世界的约束。

## 3. CoT 的历史演化

### 3.1 从直接答案到带步骤示例

早期的 prompt 通常给模型一个问题，让它直接输出答案。这个形式在事实问答和简单分类上有效，但在多步算术和逻辑问题上容易出现“最后一步错了，却看不出为什么”。

few-shot CoT 的关键变化，是在提示中给出若干带有中间步骤的示例，再让模型按照类似结构处理新问题。示例不只是展示答案格式，也展示了如何选择变量、如何分解子问题以及何时结束。

### 3.2 Zero-shot CoT 的变化

Zero-shot CoT 不提供具体解题示例，而是通过一句自然语言提示要求模型先进行分步思考。它的实验意义在于：部分大模型已经从预训练和指令数据中学到逐步解答模式，简单的提示可能把这种模式激活。

但激活效果依赖模型规模、任务类型、提示语言、答案抽取和解题难度。对一个本来不具备必要知识或算术能力的模型，要求它“逐步思考”通常只会得到更长的错误文本。

### 3.3 Scratchpad 的变化

scratchpad 研究进一步强调中间工作区，而不是面向用户的自然语言解释。草稿可以是加法结果、程序片段、状态表或离散标记。它的目标是帮助模型完成中间计算，最终输出可以只保留答案。

这条路线揭示了一个重要边界：中间状态有用，不等于中间状态必须公开；公开解释有用，也不等于它就是模型内部真正使用的全部状态。

### 3.4 从 CoT 到更复杂的 reasoning 系统

CoT 后来与多个方向连接起来：

- 多条 CoT 可以用于 self-consistency；
- CoT 节点可以组成 Tree-of-Thought 搜索；
- 每一步可以交给 process verifier 检查；
- 某一步可以调用代码、计算器或检索工具；
- 高质量轨迹可以用于监督微调、蒸馏或强化学习。

这些方向共享“中间状态”的思想，但优化目标和失败模式不同。不能因为一个系统使用了 CoT，就把它自动归类为搜索、过程监督或 RLVR 系统。

## 4. Few-shot CoT：用示例教会解题结构

### 4.1 一个完整示例

提示可以包含一个问题、推导和答案：

~~~text
问题：小明有 3 个苹果，又买了 2 个，一共有几个？
推导：原来有 3 个，又买了 2 个，所以 3 + 2 = 5。
答案：5。

问题：一本书 10 元，买 3 本多少钱？
推导：
~~~

模型可能继续生成：

~~~text
每本 10 元，买 3 本，所以 10 × 3 = 30。
答案：30 元。
~~~

示例实际上同时规定了三种协议：如何拆解问题，如何书写中间步骤，如何标记最终答案。若示例只展示冗长语言而没有展示关键变量，模型可能只学到形式，不一定学到解法。

### 4.2 示例选择比示例数量更重要

few-shot prompt 的上下文有限，示例要在覆盖和成本之间取舍。可以考虑：

1. 题型覆盖：加法、比例、单位转换、条件判断不要全是同一种结构。
2. 难度梯度：从简单组合到多步组合，避免突然跳跃。
3. 负例边界：展示如何处理缺失条件或不适用的公式。
4. 输出协议：统一答案标记和单位。
5. 语言和符号：尽量接近真实请求分布。

示例选择错误会带来两类问题。一类是模型把某个具体数字或表面模板错误套到新题上；另一类是上下文太长，真正的问题被大量示例挤压，导致延迟和上下文成本上升。

### 4.3 示例污染与记忆

如果测试题与 prompt 示例过于相似，提升可能来自模板匹配而不是更好的推理。评估时应使用未见过的数值、实体、顺序和表述，并报告原题和变式题的差距。

### 4.4 Few-shot CoT 的优点和限制

优点：

1. 不需要更新模型参数。
2. 能显式表达任务所需的步骤结构。
3. 可以控制答案格式和单位。
4. 适合快速验证某类任务是否需要中间状态。

限制：

1. 占用输入上下文和缓存空间。
2. 示例错误会被模型复制。
3. 不同模型对同一示例的响应可能不同。
4. 很难用有限示例覆盖所有异常情况。
5. 示例中的推理风格不等于任务真值。

## 5. Zero-shot CoT：一句提示能做什么，不能做什么

### 5.1 它改变的是行为诱因

“请一步一步思考”这类提示，可能让模型输出中间步骤；它改变了模型在当前上下文中的生成分布。但它并没有新增训练知识，也没有提供外部验证。

可以把提示前后的答案概率抽象成：

~~~math
p_{\theta}(y,z\mid x,\mathrm{prompt}_{\mathrm{cot}})
\ne
p_{\theta}(y\mid x,\mathrm{prompt}_{\mathrm{direct}})
~~~

这说明两种调用可能产生不同结果，但公式没有说哪一种一定正确。

### 5.2 什么时候它更容易有效

Zero-shot CoT 更可能帮助具有以下特点的任务：

- 需要两个或更多中间操作；
- 题目中明确给出足够条件；
- 中间结果可以用规则或工具检查；
- 模型已经具备相关知识和基本算术能力；
- 最终答案可以从过程或结构化字段中稳定抽取。

### 5.3 什么时候它容易失效

以下情况中，zero-shot CoT 可能只增加文本：

1. 题目缺少必要信息。
2. 模型没有相关领域知识。
3. 任务要求精确计算但没有工具。
4. 题目包含歧义或隐藏否定。
5. 模型对错误前提过度自信。
6. 目标只是快速输出一个短标签。

因此，提示词实验要同时记录直接回答和 CoT 回答，不能只展示 CoT 的成功案例。

### 5.4 语言和格式影响

中英文提示、符号格式和答案标记可能改变结果。对于多语言系统，不能把英文 CoT 的收益直接迁移到中文任务；对于数学系统，也要区分自然语言步骤和纯公式步骤。真实评估应固定语言、分隔符、采样参数和最大输出长度。

## 6. CoT、Scratchpad 与外部解释

### 6.1 三种中间状态

可以把中间状态分成三类：

1. **自然语言 CoT**：面向人类可读的步骤和说明。
2. **scratchpad**：为模型提供计算空间的中间符号、数字、程序或表格。
3. **外部解释**：给用户展示的、经过筛选的理由、证据和结论。

它们可能重叠，但不应混用。自然语言 CoT 的可读性高，却可能包含冗余和合理化；scratchpad 计算效率可能更高，却未必适合人类阅读；外部解释强调可验证和必要性，通常不追求完整记录。

### 6.2 为什么产品常不展示完整 CoT

不展示完整过程并不等于系统没有中间计算。常见原因包括：

1. 完整轨迹太长，用户难以阅读。
2. 轨迹可能包含尚未验证的草稿。
3. 可能泄露系统提示、个人数据或工具权限。
4. 用户真正需要的是结论、证据和可执行的下一步。
5. 某些危险任务中，详细过程会暴露规避安全控制的路径。

更可靠的外部解释通常包括：使用了哪些输入证据，执行了哪些可审计操作，结论依赖哪些假设，以及哪些部分仍然不确定。

### 6.3 简洁解释也可能不可靠

把长 CoT 压缩成两句话，并不会自动提高事实性。摘要可能遗漏关键前提，也可能把错误过程包装得更流畅。外部解释应由证据引用、程序检查或结构化事件支持，而不是只靠另一次语言生成。

### 6.4 面向专家：可见性是接口合同

如果产品对外承诺“解释”，需要明确解释的对象：

- 是输入证据的来源？
- 是工具调用的结果？
- 是约束和假设？
- 是模型生成的理由？
- 还是对隐藏轨迹的摘要？

不同对象需要不同的验证方法。把它们统称为 reasoning trace，会让用户误以为解释具有更高的因果可信度。

## 7. CoT 是否忠实：从相关性走向因果干预

### 7.1 正确答案与忠实解释是两个指标

最终答案正确，说明任务成功事件发生；解释忠实，要求解释中的关键步骤确实参与了答案形成。后者更难测，因为模型内部过程不可直接观察。

可以定义答案正确率：

~~~math
A_{\mathrm{final}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\hat y_i=y_i^*]
~~~

再定义解释与答案的一致率：

~~~math
C_{\mathrm{consistent}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[g(z_i)=\hat y_i]
~~~

g 是从轨迹中抽取结论的函数。C 高只表示轨迹说的结论和最终答案一致，不表示轨迹是答案的真实原因。

### 7.2 反事实替换实验

一种简单干预是替换中间步骤中的关键数字，再让模型继续生成。例如把“净流入速度为 2”改成“净流入速度为 4”，观察模型是否相应输出 5 分钟。如果答案完全不变，原步骤可能只是装饰性文字；如果答案随关键变量变化，说明它至少在行为上依赖该步骤。

干预差异可以写成：

~~~math
\Delta_{\mathrm{int}}=
\mathbb{1}[\hat y(z)=y^*]
-
\mathbb{1}[\hat y(\tilde z)=y^*]
~~~

z 是原轨迹，tilde z 是被替换的轨迹。这个量只是一次干预的效果，不是完整的内部因果证明，因为替换文本也可能改变语法、长度和模型的后续分布。

### 7.3 删除步骤实验

把一个中间步骤删除，观察后续是否恢复、改变或继续使用该信息。需要控制删除造成的 token 位置变化和格式破坏，最好用多个等价表述重复实验。

### 7.4 工具验证实验

如果 CoT 声称进行了算术，就把同一表达式交给独立计算器；如果声称读取了文档，就检查引用片段；如果声称运行了代码，就核对执行日志。工具结果比语言模型自评更能判断“步骤是否可执行”，但仍不能证明隐藏状态完全遵循这些步骤。

### 7.5 专家边界：忠实性没有单一分数

忠实性至少包括：

1. 事实忠实：步骤没有捏造输入或外部证据。
2. 计算忠实：关键算术、逻辑或程序操作真实成立。
3. 因果忠实：干预步骤会影响依赖它的结论。
4. 覆盖忠实：解释没有遗漏决定性前提。
5. 语义忠实：摘要没有改变原结论的条件。

一个解释可以在其中一项上合格，在另一项上失败。因此论文或产品不能只给一个“解释质量”总分就结束分析。

## 8. CoT 数据：模型会学到什么

### 8.1 数据的基本结构

一个最小的 CoT 样本可以包含：

~~~json
{
  "question": "小明有 3 个苹果，又买了 2 个，一共有几个？",
  "reasoning": "原来有 3 个，又买了 2 个，所以 3 + 2 = 5。",
  "answer": "5"
}
~~~

这个结构看似简单，但数据生产者必须明确 reasoning 是人工解答、教师模型生成、程序轨迹还是最终答案的解释性改写。来源不同，可信度和错误类型不同。

### 8.2 高质量数据的五个条件

高质量 CoT 数据通常需要：

1. 关键中间步骤正确。
2. 步骤与题目条件对应。
3. 最终答案可以从过程复算。
4. 过程长度与任务复杂度匹配。
5. 失败、缺失条件和拒答样本有明确标签。

“写得像教科书”不是充分条件。某些简洁解法是正确的，某些很长的解法包含未经验证的跳跃。

### 8.3 错误 CoT 的传播

如果把教师模型生成的错误轨迹直接作为监督数据，学生模型可能学到三种坏习惯：

- 先给出结论，再补一段貌似合理的理由；
- 遇到不确定条件时自行补全；
- 用更长的文字掩盖无法计算的部分。

因此，CoT 数据构造应尽量加入程序验证、符号检查、人工抽样、反事实测试和步骤级去重。

### 8.4 过程监督与答案监督

若目标只是在有限任务上得到正确答案，可以只对最终答案计算损失；若目标是让模型学会某类中间结构，则可以对步骤 token 也施加监督。对第 t 个 token 使用掩码 m_t，统一的监督损失可以写成：

~~~math
\mathcal{L}_{\mathrm{masked}}
=-\sum_{t=1}^{T}m_t\log p_{\theta}(y_t\mid x,y_{1:t-1})
~~~

m_t 为 1 表示该 token 参与损失，为 0 表示忽略。不同的 m_t 设计会产生不同学习目标：只监督答案、监督答案和步骤、或者对工具结果和关键状态单独加权。

如果采用 mean loss，还必须除以有效位置数 `\sum_t m_t`，并要求这个分母大于零；上面的求和式是未归一化写法。整条样本都被 mask，不能把 loss 默认为零，因为它表示没有监督信号。`m_t` 还应与 shift 后的目标 token 对齐，padding、问题文本和媒体占位符不能因为序列位置相邻而被误纳入答案监督。

### 8.5 步骤损失的风险

对每个文字 token 都同样加权，可能让模型偏好长解释而不是有效计算。更合理的做法可能是按步骤、关键变量、验证结果或任务成功事件加权，但权重本身也会引入偏差。

例如，一道题有两种同样正确的解法，逐 token 模仿其中一种可能压低另一种解法的概率。过程监督不应被误解为“唯一正确文本的文字复制”。

### 8.6 蒸馏中的 CoT

强教师模型可以生成较长轨迹，小模型学习这些轨迹。蒸馏的收益可能来自知识、分解方式、格式协议或错误模式；若教师轨迹未经验证，小模型也会继承错误。

蒸馏评估需要把四个对象分开：教师原始答案、教师轨迹、学生答案和学生轨迹。学生最终答对，并不代表学生学到了教师的真实中间算法；学生答错，也不一定说明轨迹格式没有价值。

## 9. CoT 的答案格式与解析

### 9.1 为什么答案边界很重要

自动评估通常只想比较最终答案，但 CoT 把推理文字和结果混在一起。若没有明确分隔符，解析器可能把步骤里的中间数字当成最终答案。

一个简单协议是：

~~~text
推理：
净流入速度为 2 升/分钟。
时间为 20 / 2 = 10 分钟。

最终答案：10 分钟
~~~

分隔符不是推理能力，但它是系统可用性的基础。解析失败、单位丢失和多答案并存都应单独记录，而不是直接算作模型推理失败。

### 9.2 解析器的精确率和召回率

设解析器从模型输出中抽取答案，抽取结果正确的样本数为 TP，抽取错误的样本数为 FP，漏掉可抽取答案的样本数为 FN，则：

~~~math
P_{\mathrm{parse}}=\frac{TP}{TP+FP},
\qquad
R_{\mathrm{parse}}=\frac{TP}{TP+FN}
~~~

解析器精确率高但召回率低，会把很多本来正确的回答判成格式失败；召回率高但精确率低，会把错误中间数字当成答案。应把 parser 指标和模型答案指标分开报告。

这两个比率分别要求 `TP+FP>0` 和 `TP+FN>0`。当评估集中没有任何成功抽取或没有任何可抽取答案时，对应指标是不可用的，不应写成零。解析失败、模型拒答和答案本身错误也应使用不同状态码，否则 parser 的错误会被混入模型能力指标。

### 9.3 等价答案和单位归一化

数学答案需要支持等价形式、分数、小数、单位和四舍五入规则。代码答案需要区分编译成功、测试通过和接口契约。自然语言答案则可能需要结构化字段和证据支持。

归一化器本身也可能有 bug。例如把 10%、0.1 和 10 混为一谈，会产生假提升。答案处理代码需要独立单元测试和边界样例。

## 10. CoT 与工具使用

### 10.1 CoT 负责规划，工具负责精确执行

大数计算、符号变换、代码运行和数据库查询不应只依赖自然语言。一个稳妥流程是：

1. 从问题中抽取目标和约束。
2. 选择适合的工具。
3. 生成结构化参数或程序。
4. 执行并读取 observation。
5. 检查工具结果是否与输入和权限一致。
6. 用外部证据组织最终说明。

这里的 CoT 不应替代工具，而应负责提出可执行动作和解释需要验证的假设。

### 10.2 工具轨迹的形式化

设 s_t 是当前状态，a_t 是模型提出的动作，o_{t+1} 是环境返回的观察：

~~~math
(s_t,a_t)\xrightarrow{\mathcal{E}}(o_{t+1},s_{t+1})
~~~

环境 E 可以是计算器、Python 沙箱、检索系统或数据库。工具失败、超时、权限拒绝和空结果都必须作为不同 observation 处理。模型不能把“工具没有返回”直接当成“事实不存在”。

### 10.3 工具调用不是免费的正确性

工具可能被错误选择、错误参数化或错误解释。模型也可能把一个成功运行的程序当成业务正确，而程序本身读取了错误字段。

所以工具系统要记录：动作、参数、权限、返回值、异常、重试和最终采用的证据。CoT 文字只能说明模型声称做了什么，日志才说明系统实际做了什么。

### 10.4 什么时候应该直接让模型调用工具

如果任务需要精确算术、外部最新信息、代码执行或状态变化，应优先定义工具契约，而不是要求模型写更长的自然语言。增加 CoT token 不能替代缺失数据，也不能保证外部事实没有变化。

## 11. 什么时候使用 CoT，什么时候缩短

### 11.1 任务特征

CoT 更可能有效的任务通常具有：

1. 两步或更多中间操作。
2. 明确的变量、约束或状态。
3. 可以进行答案或过程验证。
4. 结果错误的代价足以覆盖额外成本。

不一定需要 CoT 的任务包括简单事实、短分类、固定格式转换和只要求低延迟的接口。

### 11.2 用路由而不是全局开关

可以让系统先判断任务类型，再选择 direct、CoT、工具或拒答。路由不是“模型聪明程度”的标签，而是资源分配策略。

设策略 r 为 direct、cot、tool 等选项，可以把选择目标写成：

~~~math
\max_{r}\;
\mathbb{E}\left[Q(r)-\lambda C(r)-\mu L(r)-\nu R(r)\right]
~~~

Q 是任务质量，C 是计算成本，L 是延迟，R 是风险，lambda、mu、nu 是业务权重。不同应用的权重不同：付款审核会更重视正确性和证据，聊天产品可能更重视延迟，安全工具则需要严格限制风险。

这个目标函数要求 `Q,C,L,R` 在候选策略和同一任务分布上可比较，`lambda、mu、nu` 为有限的非负权重。若某个策略的风险超过硬性安全约束，不能只靠把 `nu` 调小来让它在加权平均中获选；高风险动作还应有独立的不可违反约束。

### 11.3 难度估计的陷阱

题目长度不等于推理难度。一个长的背景说明可能只需要抽取一个日期；一个短的问题可能隐藏否定、单位或多重条件。

可用的难度信号包括任务类型、历史错误率、工具验证失败、候选不一致和输入结构，但任何单一信号都可能失效。路由策略必须在独立切片上校准，而不是只在训练样本上选择阈值。

### 11.4 CoT 回归

如果直接回答正确而 CoT 回答错误，这就是一个重要回归。回归率可以定义为：

~~~math
R_{\mathrm{reg}}
=\frac{1}{N}\sum_{i=1}^{N}
\mathbb{1}\left[
\hat y_i^{\mathrm{direct}}=y_i^*
\land
\hat y_i^{\mathrm{cot}}\ne y_i^*
\right]
~~~

简单事实题上的 CoT 回归说明强制推理可能让模型偏离熟悉答案；它不应被“复杂题平均提升”掩盖。

回归率要求 `N>0`，并且 direct 与 CoT 使用同一批有效样本、同一答案归一化规则。若任一策略解析失败，需明确它是失败事件还是不可比较样本；不能一边把 direct 的解析失败排除，一边把 CoT 的解析失败计入分母。

## 12. 长 CoT：更多 token 的收益和反作用

### 12.1 质量曲线不一定单调

设 CoT token 预算为 B，任务质量为 A(B)。如果增加预算带来收益：

~~~math
\Delta A(B)=A(B+\Delta B)-A(B)
~~~

但真实系统中 Delta A 可能变小甚至变负。模型可能重复检查已经正确的步骤、引入新的算术错误、忘记最初约束，或在上下文中把无关信息权重提高。

### 12.2 长度和有效步骤

更有意义的指标不是总 token，而是有效步骤数、关键变量覆盖率、重复比例和错误传播长度。可以定义重复比例：

~~~math
\rho_{\mathrm{repeat}}=
\frac{\text{重复或无新状态的 token 数}}{\text{CoT token 总数}}
~~~

这个量需要先定义重复检测规则，不能把任何相似词语都当成无效。它的作用是提醒工程师：长输出不等于长计算。

重复比例要求 CoT token 总数大于零，且“重复或无新状态”的判定规则在实验前固定。空轨迹应记录为无可评估轨迹，而不是重复比例为零；不同 tokenizer 或标点归一化也可能改变表面重复计数。

### 12.3 长 CoT 的资源影响

在自回归生成中，输出 token 会增加 decode 时间和服务成本；如果推理状态保存在上下文中，还会影响后续工具调用、缓存和上下文上限。高并发系统不能只看单请求质量，还要看队列等待和吞吐。

### 12.4 早停和中间检查

如果中间状态已经满足目标，可以提前停止；如果关键步骤验证失败，可以修正或转交工具。早停条件应该基于可检查状态，而不是模型自己声称“已经完成”。

## 13. CoT 的失败模式

### 13.1 计算步骤看似合理但结果错误

模型可能在公式选择、符号变换或最后一步算术上出错。解决方式是将关键运算交给程序，或对中间变量设置独立断言。

### 13.2 过程和答案不一致

过程写出 5 分钟，最后答案写 10 分钟。若 parser 只读取最后一行，系统会漏掉这个矛盾；若只读取过程，也可能误判。应该把一致性检查作为独立指标。

### 13.3 自行补全不存在的条件

遇到缺失税率、未说明单位或模糊时间，模型可能选择一个常见假设并继续计算。高质量系统应显式列出假设，必要时要求用户补充。

### 13.4 先猜答案再编理由

当模型对答案有强先验时，它可能围绕候选答案构造解释。干预实验、工具执行和数字替换可以帮助发现这种现象，但一次失败不能证明所有解释都是事后编写。

### 13.5 格式漂移

不同题目中答案标记、单位、列表和代码块格式变化，会让自动评估和下游工具失败。格式协议需要在训练、推理和解析三处一致。

### 13.6 过度推理

简单任务被拆成很多无关步骤，会增加延迟并制造新错误。路由和回归切片是识别过度推理的基本工具。

## 14. CoT 的安全与隐私

### 14.1 完整过程不是默认应公开的数据

CoT 可能包含用户输入中的隐私、系统规则、工具权限和危险操作细节。即使某段文字看起来像普通推理，也可能把不应暴露的上下文重新组合出来。

### 14.2 解释最小化

面向用户的解释应提供完成任务所需的证据和假设，而不是默认展示全部草稿。对文档问答，可以展示原文片段和页码；对计算，可以展示公式、代入值和结果；对安全拒答，可以说明边界和安全替代方案，但不提供规避路径。

### 14.3 安全任务中的 CoT

安全模型可能通过更长的 CoT 发现危险操作的分解方式，也可能把拒答理由写得过于详细。评估应同时看拒答正确率、危险信息泄露、工具动作和多轮绕过。

### 14.4 日志与用户展示分离

系统日志可以保留结构化的工具调用、验证结果、风险标签和版本信息；用户界面可以展示简洁理由和证据。二者的权限、保留期限和脱敏策略应不同。

## 15. 一个综合案例：合同付款计算

### 15.1 任务定义

用户上传一份合同，询问本月应付金额。系统需要找到基础金额、折扣、税率和生效日期，再计算金额并给出证据。

这不是单纯的“让模型写 CoT”。它包含文档检索、OCR、字段抽取、条件判断、算术、证据对齐和权限检查。

### 15.2 一条可审计的轨迹

~~~text
检索：正文第 3 页，基础金额为 100000 元。
检索：正文第 4 页，折扣为 5%，且适用于当前月份。
检索：正文第 7 页，税率为 6%。
计算：100000 × (1 - 0.05) × (1 + 0.06) = 100700 元。
验证：三个字段都有来源位置，表达式可由独立计算器复算。
输出：应付金额为 100700 元，并附上三处证据。
~~~

这里的文字步骤有价值，是因为每个关键值都绑定了证据和可执行计算。若只是输出“我检查了合同并计算得到 100700”，可读性更短，但审计价值更低。

### 15.3 缺失条件的处理

如果正文只找到基础金额和折扣，没有税率，系统不能用常见税率填空。它可以输出：

~~~text
已找到基础金额和折扣，但合同中未找到适用于本次付款的税率。当前无法可靠计算含税金额；请补充税率或确认按未税金额付款。
~~~

这不是“推理不够长”，而是证据不足。增加 CoT token 不能替代缺失信息。

### 15.4 失败归因

合同案例中的错误至少可以分成：

1. 检索没有找到适用条款。
2. OCR 把 6% 读成 60%。
3. 模型把上一季度折扣用于本月。
4. 计算表达式错误。
5. 引用位置与字段错配。
6. 用户没有查看权限。

每种错误的修复不同。CoT 只能覆盖其中一部分，不能成为所有问题的总开关。

## 16. 最小可运行实验：CoT、直答与路由

下面的实验不调用真实模型，而是用六个 toy case 模拟三种策略：

1. direct：直接输出答案。
2. cot：先生成推理链再输出答案。
3. routed：只在复杂数学题和代码题上使用 CoT，简单题和安全边界题使用短答或拒答。

实验故意保留两个坏样本：capital_lookup 展示 CoT 回归，distractor_math 展示步骤引入题目外条件。它们不是模型统计结论，而是帮助读者理解为什么需要分层报告。

~~~python
import math
from collections import Counter


cases = [
    {
        "id": "water_tank",
        "route": "hard_math",
        "gold": "10",
        "direct": "12",
        "cot": "10",
        "direct_tokens": 5,
        "cot_tokens": 55,
        "steps": [True, True, True],
        "unsupported": False,
        "visible_cot_allowed": True,
    },
    {
        "id": "apple_simple",
        "route": "simple",
        "gold": "5",
        "direct": "5",
        "cot": "5",
        "direct_tokens": 4,
        "cot_tokens": 30,
        "steps": [True, True],
        "unsupported": False,
        "visible_cot_allowed": True,
    },
    {
        "id": "distractor_math",
        "route": "hard_math",
        "gold": "12",
        "direct": "99",
        "cot": "111",
        "direct_tokens": 5,
        "cot_tokens": 45,
        "steps": [False, False],
        "unsupported": True,
        "visible_cot_allowed": True,
    },
    {
        "id": "code_loop",
        "route": "code",
        "gold": "pass",
        "direct": "fail",
        "cot": "pass",
        "direct_tokens": 6,
        "cot_tokens": 60,
        "steps": [True, True, True],
        "unsupported": False,
        "visible_cot_allowed": True,
    },
    {
        "id": "capital_lookup",
        "route": "simple",
        "gold": "tokyo",
        "direct": "tokyo",
        "cot": "kyoto",
        "direct_tokens": 4,
        "cot_tokens": 48,
        "steps": [False, False],
        "unsupported": False,
        "visible_cot_allowed": True,
    },
    {
        "id": "safety_boundary",
        "route": "safety",
        "gold": "refuse",
        "direct": "refuse",
        "cot": "refuse",
        "direct_tokens": 8,
        "cot_tokens": 24,
        "steps": [True],
        "unsupported": False,
        "visible_cot_allowed": False,
    },
]


def validate_cases(items):
    if not isinstance(items, list) or not items:
        raise ValueError("cases must be a non-empty list")
    seen = set()
    allowed_routes = {"hard_math", "simple", "code", "safety"}
    for index, case in enumerate(items):
        required = {
            "id", "route", "gold", "direct", "cot", "direct_tokens", "cot_tokens",
            "steps", "unsupported", "visible_cot_allowed",
        }
        if not isinstance(case, dict) or set(case) != required:
            raise ValueError(f"case[{index}] has an invalid schema")
        if not isinstance(case["id"], str) or not case["id"] or case["id"] in seen:
            raise ValueError("case ids must be unique non-empty strings")
        seen.add(case["id"])
        if case["route"] not in allowed_routes:
            raise ValueError(f"case[{index}].route is unknown")
        for key in ("gold", "direct", "cot"):
            if not isinstance(case[key], str) or not case[key]:
                raise ValueError(f"case[{index}].{key} must be non-empty")
        for key in ("direct_tokens", "cot_tokens"):
            if isinstance(case[key], bool) or not isinstance(case[key], int) or case[key] < 0:
                raise ValueError(f"case[{index}].{key} must be a non-negative integer")
        if not isinstance(case["steps"], list) or not case["steps"] or any(not isinstance(step, bool) for step in case["steps"]):
            raise ValueError(f"case[{index}].steps must be a non-empty boolean list")
        if not isinstance(case["unsupported"], bool) or not isinstance(case["visible_cot_allowed"], bool):
            raise TypeError(f"case[{index}] boolean fields are invalid")


def accuracy(items, key):
    validate_cases(items)
    if key not in {"direct", "cot"}:
        raise ValueError("accuracy key must be direct or cot")
    return sum(case[key] == case["gold"] for case in items) / len(items)


def route_answer(case):
    if case["route"] not in {"hard_math", "code", "simple", "safety"}:
        raise ValueError("unknown route")
    if case["route"] in {"hard_math", "code"}:
        return case["cot"], case["cot_tokens"]
    return case["direct"], case["direct_tokens"]


validate_cases(cases)
direct_acc = accuracy(cases, "direct")
cot_acc = accuracy(cases, "cot")
routed_correct = 0
routed_tokens = 0
route_counts = Counter()

for case in cases:
    answer, tokens = route_answer(case)
    routed_correct += answer == case["gold"]
    routed_tokens += tokens
    route_counts[case["route"]] += 1

all_steps = [ok for case in cases for ok in case["steps"]]
step_accuracy = sum(all_steps) / len(all_steps)
regression_ids = [
    case["id"]
    for case in cases
    if case["direct"] == case["gold"] and case["cot"] != case["gold"]
]
unsupported_ids = [case["id"] for case in cases if case["unsupported"]]
visible_cot_blocked = [
    case["id"] for case in cases if not case["visible_cot_allowed"]
]
simple_cases = [case for case in cases if case["route"] == "simple"]
if not simple_cases:
    raise ValueError("simple_cases must not be empty")
simple_waste_or_regression = sum(
    case["direct"] == case["gold"]
    and case["cot_tokens"] > case["direct_tokens"]
    for case in simple_cases
) / len(simple_cases)

report = {
    "direct_accuracy": round(direct_acc, 3),
    "cot_accuracy": round(cot_acc, 3),
    "routed_accuracy": round(routed_correct / len(cases), 3),
    "step_accuracy": round(step_accuracy, 3),
    "avg_cot_tokens": round(
        sum(case["cot_tokens"] for case in cases) / len(cases), 3
    ),
    "cost_per_cot_correct": (
        round(
            sum(case["cot_tokens"] for case in cases)
            / sum(case["cot"] == case["gold"] for case in cases),
            3,
        )
        if any(case["cot"] == case["gold"] for case in cases)
        else None
    ),
    "cost_per_routed_correct": round(routed_tokens / routed_correct, 3) if routed_correct else None,
    "simple_waste_or_regression": round(simple_waste_or_regression, 3),
    "route_counts": dict(route_counts),
    "cot_regression_ids": regression_ids,
    "unsupported_step_ids": unsupported_ids,
    "visible_cot_blocked": visible_cot_blocked,
}

assert report["direct_accuracy"] == 0.5
assert report["cot_accuracy"] == 0.667
assert report["routed_accuracy"] == 0.833
assert report["step_accuracy"] == 0.692
assert report["cost_per_routed_correct"] == 35.2

for key, value in report.items():
    print(f"{key}={value}")
~~~

预期输出：

~~~text
direct_accuracy=0.5
cot_accuracy=0.667
routed_accuracy=0.833
step_accuracy=0.692
avg_cot_tokens=43.667
cost_per_cot_correct=65.5
cost_per_routed_correct=35.2
simple_waste_or_regression=1.0
route_counts={'hard_math': 2, 'simple': 2, 'code': 1, 'safety': 1}
cot_regression_ids=['capital_lookup']
unsupported_step_ids=['distractor_math']
visible_cot_blocked=['safety_boundary']
~~~

这个实验应这样解读：

1. CoT 在这组构造样本上比 direct 正确率高，但仍然有错误步骤。
2. routed 只把 CoT 用在 hard_math 和 code，因此避开了 capital_lookup 的回归。
3. distractor_math 的两步都被标成错误，说明最终答案和过程质量必须分开统计。
4. simple_waste_or_regression=1.0 表示两个 simple 样本都满足“CoT 比直答更费 token”，其中一个还发生了回归。
5. safety_boundary 被列入不可展示 CoT 的样本，说明内部工作区和用户界面策略可以不同。

## 17. 怎样设计 CoT 实验

### 17.1 固定比较条件

比较 direct 和 CoT 时，至少固定模型版本、输入题目、最大输出、采样温度、答案 parser 和停止条件。否则提升可能来自不同的输出上限或不同的解析规则。

### 17.2 分层报告

不要只报告总体准确率。至少按任务类型、难度、输入长度、是否需要工具、是否有干扰条件和是否属于高风险请求分层。

### 17.3 过程抽样

步骤级标注成本很高，可以先对错误样本和高不确定样本抽样，标记关键变量、错误位置、外部条件和是否需要工具。抽样协议要写清楚，避免只挑最容易解释的案例。

### 17.4 干预实验

对一部分样本做数字替换、步骤删除、顺序交换和无关信息注入。若答案、引用或工具调用没有按预期变化，需要进一步研究轨迹是否只是表面格式。

### 17.5 成本曲线

用不同的 CoT token 上限运行同一任务集，报告：

- 最终答案准确率；
- 过程错误率；
- parser 失败率；
- 平均和 P95 延迟；
- 输入和输出 token；
- 工具调用和验证成本；
- 单位成功成本。

只有在目标切片和成本约束下都值得，CoT 才适合进入生产策略。

## 18. 常见误区

### 误区一：让模型逐步思考就一定会变强

提示只改变生成条件，不会凭空补足知识、工具和验证器。应通过对照实验判断收益。

### 误区二：CoT 越长越可靠

长轨迹可能包含更多有效状态，也可能包含更多错误、重复和偏离。长度是资源变量，不是质量证明。

### 误区三：过程和最终答案一致就说明过程真实

一致性只说明两者表达相同结论。它不能证明步骤参与了答案形成，也不能证明外部事实正确。

### 误区四：隐藏 CoT 就等于不透明

系统可以通过证据、工具日志、结构化假设和最终检查结果提供可审计性，不必把未经验证的完整草稿暴露给用户。

### 误区五：所有错误都能靠更好 prompt 修复

缺失数据、OCR 错误、工具权限和测试覆盖不足，不是多写一段 prompt 就能解决的。失败归因必须定位到具体系统环节。

### 误区六：教师模型写出的 CoT 天然正确

教师轨迹也需要验证和抽样。错误教师会把“自信但错误的推理风格”传给学生。

## 19. 小练习

### 练习一：设计 direct 与 CoT 对照

选择十道包含单位、否定和多步计算的题目，固定模型和解码参数，比较 direct 与 CoT 的答案正确率、parser 失败率和 token 成本。

### 练习二：构造忠实性干预

写一个包含中间变量的题目，替换其中一个变量，再观察模型是否修改后续答案。说明如何控制替换造成的格式变化。

### 练习三：判断过程是否可验证

分别为数学、代码、开放式建议设计一个过程检查器，并说明哪些步骤不能被自动判定。

### 练习四：训练损失掩码

给出一条包含问题、推理和答案的序列，设计三种 m_t：只训练答案、训练推理与答案、只训练关键字段。比较每种设计可能带来的收益和副作用。

### 练习五：设计 CoT 路由

把请求分成简单事实、复杂数学、代码、文档问答和安全敏感五类，设计 direct、CoT、tool 和 refuse 的初始路由，并为每一类指定回归指标。

### 练习六：查阅原始论文

阅读 Chain-of-Thought Prompting、Zero-shot Reasoners、Scratchpads、Faithful Chain-of-Thought 和 Unfaithful Explanations 论文，记录每篇论文的实验对象、方法、主要结论和没有证明的内容。

## 20. 本章总结

CoT 是一种中间状态表达和推理时计算方法。它可能让模型把复杂任务拆成步骤，保留变量，匹配训练中学到的解题结构，并为验证器和工具提供接口。

但 CoT 不是自动正确的思维证明，也不是必须展示给用户的完整内部日志。可靠使用 CoT 需要同时处理：

1. 任务是否真的需要中间状态。
2. few-shot 或 zero-shot 提示是否改变了有效行为，而不只是增加文字。
3. scratchpad、自然语言轨迹和外部解释的边界。
4. 过程是否通过干预、工具和独立规则得到验证。
5. CoT 数据是否正确、可复算、覆盖失败和拒答情况。
6. 答案 parser、单位归一化和格式协议是否可靠。
7. CoT token、延迟、工具和验证成本是否值得质量收益。
8. 简单任务是否出现回归，安全任务是否出现信息泄露。

下一章进入 self-consistency 与采样，研究如何从多条 CoT 候选中利用多样性、答案标准化和投票降低一次生成的偶然错误。

## 21. 资料索引

以下链接优先保留原始论文。复现实验时要记录模型版本、提示文本、采样参数、最大输出长度、答案解析器和评估切分。

1. Chain-of-Thought Prompting：<https://arxiv.org/abs/2201.11903>
2. Zero-shot Reasoners：<https://arxiv.org/abs/2205.11916>
3. Show Your Work: Scratchpads for Intermediate Computation：<https://arxiv.org/abs/2112.00114>
4. Faithful Chain-of-Thought Reasoning：<https://arxiv.org/abs/2301.13379>
5. Language Models Don't Always Say What They Think：<https://arxiv.org/abs/2305.04388>
6. Chain-of-Thought Reasoning in Large Language Models：<https://arxiv.org/abs/2211.12588>
7. Let's Verify Step by Step：<https://arxiv.org/abs/2305.20050>
8. OpenAI reasoning guide：<https://platform.openai.com/docs/guides/reasoning>
9. Anthropic extended thinking：<https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking>
10. Gemini thinking：<https://ai.google.dev/gemini-api/docs/thinking>

本章写作时已对上述论文和官方文档入口进行联网访问。正文将论文实验、官方接口、教学构造和目标系统实测分开描述；没有把闭源模型的未公开内部推理写成确定事实。
