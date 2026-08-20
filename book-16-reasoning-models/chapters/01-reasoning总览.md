# 第一章：Reasoning 总览

推理模型（reasoning model）不是“会输出更长答案的聊天模型”，也不是给普通模型加上一句“请一步一步思考”就自然得到的新物种。它更像一个由模型、训练数据、推理策略、验证器、工具和评估方法共同组成的系统。

当一个模型解一道数学题时，至少有五件事可能同时发生：模型从语言模式中提出解法，生成中间状态，调用计算器或代码执行器，比较多个候选路径，并根据外部反馈修正答案。只看最后一段文字，很难判断其中究竟是哪一部分带来了正确性。

本章的任务是建立一张地图。后续章节会分别深入 Chain-of-Thought、self-consistency、verifier、process supervision、search、test-time compute、数学与代码推理训练以及 reasoning 评估。本章不把这些主题压缩成术语表，而是先说明它们为什么依次出现、彼此解决什么问题，以及哪些结论不能从表面现象直接推出。

## 0.1 读者应该先建立的三个区分

### 第一组：答案、过程与能力

一个答案可以是正确的，但过程是碰巧猜到的；一个过程可以看起来很完整，但其中存在关键错误；一次错误也不能证明模型完全没有这项能力，因为另一个采样路径可能能够解决同一道题。

因此至少要区分：

1. **最终答案**：任务最后交付的结果，例如数值、程序、计划或工具调用。
2. **可观察过程**：模型写出来的步骤、草稿、工具轨迹和中间状态。
3. **潜在能力**：在给定输入、预算和环境下，模型生成正确结果的概率。

这三个对象相关，但不是同一个对象。尤其是可观察的文字解释不一定忠实地记录了模型内部真正影响答案的计算过程。它可能是求解过程，也可能只是答案生成后的合理化叙述。

### 第二组：模型能力与系统策略

同一个基础模型可以用不同方式运行：

- 只生成一次答案；
- 采样多条候选路径，再投票；
- 让程序检查答案；
- 让模型调用 Python、搜索或数据库；
- 在多个中间状态之间搜索；
- 根据题目难度动态增加推理预算。

如果系统 A 比系统 B 正确率高，不能马上说“模型 A 更会推理”。可能只是 A 采样了更多候选，使用了更强的测试器，或者允许了更多工具调用。

### 第三组：论文结论、产品行为与目标系统实测

论文通常描述固定数据集、固定模型和固定实验协议下的结果；官方文档描述某个 API 版本公开支持的参数和行为；产品页面描述面向用户的能力边界；真实业务还要受到数据分布、权限、延迟、预算和失败处理的影响。

本章会把这几类证据分开写。公开资料可以证明某个方法被提出、某个接口公开存在或某个实验在特定条件下有效，但不能自动证明一个未公开系统内部采用了同样的训练配方，也不能证明它在读者自己的任务上同样有效。

## 0.2 一条贯穿本章的例子

我们先看一个很小的题目：

~~~text
一个水箱每分钟进水 3 升，每分钟漏水 1 升，容量 20 升，从空开始多久装满？
~~~

正确解法不是从句子表面寻找一个常见答案，而是先建立变量：净流入速度是每分钟 3 - 1 = 2 升，所需时间是 20 / 2 = 10 分钟。

这个例子很简单，却足以展示推理系统的基本结构：

1. 解析题目中的实体、数量和关系。
2. 形成一个中间状态：净流入速度为 2。
3. 进行第二次计算：20 除以 2。
4. 输出答案，并让规则或计算器检查它。

如果只看最后的“10 分钟”，我们不知道模型是正确计算、记住了相似题，还是偶然猜中。若生成三条解法，其中两条算出 10，一条算出 12，我们可以研究投票是否有帮助；若把候选交给算术检查器，还可以研究验证器能否把错误路径排除。

## 0.3 本章使用的资料与证据边界

本章的历史和方法主线主要来自原始论文：Chain-of-Thought 提示研究、Zero-shot CoT、Self-Consistency、数学题验证器、过程监督、Tree of Thoughts、GSM8K、MATH、HumanEval 以及 DeepSeek-R1 的公开技术报告。它们分别提供方法定义、实验设置或训练路线证据，而不是对所有商业模型内部实现的证明。

本章还参考了 OpenAI reasoning guide、Anthropic extended thinking 文档和 Gemini thinking 文档。它们可以说明公开 API 中存在推理预算、thinking 或工具协同等产品层控制，但不同厂商的字段语义、计费方式、可见性和版本都可能变化。不能因为多个接口都出现了类似名称，就断言它们内部使用了同一种算法。

阅读来源时可以按以下层级判断：

1. 原始论文和技术报告：适合支持方法定义、实验协议和论文作者声称的结果。
2. 官方文档与官方代码：适合支持公开接口、参数语义和实现行为。
3. 基准数据集与评估脚本：适合支持任务定义、答案格式和测量方法。
4. 独立复现和工程实验：适合发现实现差异、成本和失败模式，但要保留实验条件。
5. 产品宣传、社区转述和未署名排行榜：只能作为线索，不能单独支撑内部架构或能力结论。

## 1. 什么叫推理

### 1.1 小白视角：从条件走到结论

推理可以先理解成一种受约束的多步变换：给定问题和证据，模型不是直接吐出一个最像训练语料的句子，而是逐步构造中间结果，最后得到能够检查的结论。

“多步”并不等于“步骤越多越好”。一道一眼可见的事实问答不需要长推理；一个需要执行代码、比较假设、处理不确定性的任务，才可能从额外步骤中受益。

例如，水箱问题的关键不是写出四行漂亮文字，而是保留两个必要中间量：净流入速度和所需时间。若模型写了十行与单位无关的解释，推理并没有因此变得更可靠。

### 1.2 专家视角：把推理写成潜在轨迹

设输入为 x，最终输出为 y，中间轨迹为 z。z 可以是自然语言步骤、程序、搜索节点、工具观察或它们的混合。一个抽象的 reasoning 系统可以写成：

~~~math
p(y,z\mid x)=p(z\mid x)\,p(y\mid x,z)
~~~

这里的分解只是建模视角，不意味着实际模型一定显式地先生成完整的 z 再生成 y。在自回归模型中，步骤和答案通常都作为 token 序列生成；在工具系统中，z 还可能包含外部环境返回的 observation。

如果任务有参考答案 y^*，最常见的成功事件是：

~~~math
S(x)=\mathbb{1}[\hat y=y^*]
~~~

S(x) 只记录最终结果是否满足任务定义。它不能说明过程是否正确，也不能说明错误是否可定位。对于代码任务，y^* 往往不是唯一程序，而是“通过测试并满足接口契约”；对于规划任务，成功事件可能是多个约束同时满足。

因此，推理能力更适合定义成带条件的成功概率：

~~~math
A(\pi,b,e)=\mathbb{E}_{x\sim D}\left[\mathbb{1}\bigl[\mathrm{Success}(x;\pi,b,e)=1\bigr]\right]
~~~

其中：

- D 是任务分布；
- pi 是运行策略，包括解码、采样、搜索和工具规则；
- b 是计算预算，例如输出 token、候选数量和工具调用次数；
- e 是外部环境，例如测试器、数据库或执行沙箱。

这个定义提醒我们：准确率不是模型孤立的常数，而是模型、策略、预算和环境共同决定的结果。

### 1.3 推理和模式匹配并不是二选一

语言模型通过 next-token prediction 学习，而不是通过一个名为“推理”的独立模块学习。训练语料中既有事实、对话和叙述，也有数学推导、程序、证明、实验报告和规划文本。模型可以利用这些结构，把前面的 token 组合成后面的答案。

基础目标通常写成：

~~~math
\mathcal{L}_{\mathrm{LM}}(\theta)=-\sum_{t=1}^{T}\log p_{\theta}(y_t\mid x,y_{1:t-1})
~~~

theta 是模型参数，T 是目标序列长度。这个目标只要求模型提高下一个 token 的概率，并没有直接要求每个中间步骤都符合物理规律或数学证明规则。

为什么它仍然可能产生推理？原因至少有三层：

1. 训练数据中存在大量结构化解题过程。
2. Transformer 可以在上下文中保留并组合中间变量。
3. 当任务、数据和规模足够复杂时，预测后续符号可能需要构造某种内部计算。

但“能生成推理样式文本”与“每一步都正确”之间没有逻辑必然关系。模型可以生成看似合理的错误证明，也可以在答案正确时给出不忠实的解释。

### 1.4 一个反例：正确答案不等于正确过程

假设模型回答：

~~~text
净流入速度是 3 + 1 = 4 升/分钟，所以时间是 20 / 4 = 5 分钟。
答案是 10 分钟。
~~~

最终答案碰巧写对了，但过程包含矛盾。若系统只检查最终字符串，就会把它计为成功；若这个过程被用作监督数据，则可能把错误关系教给模型。

反过来，模型也可能直接输出正确的“10 分钟”，没有显式中间步骤。对于低风险、可直接计算的任务，这并不一定是问题。是否需要过程监督，取决于任务是否需要审计、纠错、教学或后续工具操作。

## 2. Reasoning Model、Chat Model 与运行系统

### 2.1 不要把两个标签当成两个物种

“chat model”和“reasoning model”是有用的工程标签，但不是严格的生物学分类。一个模型可以同时支持闲聊、摘要、代码和复杂数学；一个产品也可能根据请求参数切换不同的推理预算或路由。

更稳妥的比较方式是拆成多个轴：

| 轴 | 主要问题 | 可能的测量方式 |
|---|---|---|
| 指令遵循 | 是否按格式和约束完成任务 | schema 成功率、拒答正确率 |
| 结果正确性 | 最终答案是否满足任务 | exact match、单元测试、人工判定 |
| 过程质量 | 中间步骤是否可检查 | step accuracy、证明检查、轨迹审计 |
| 搜索能力 | 是否能探索多个候选 | pass@k、树搜索成功率 |
| 工具协同 | 能否正确调用并使用反馈 | 工具成功率、错误恢复率 |
| 预算适应 | 难题是否值得花更多计算 | 质量—成本曲线、P95 延迟 |
| 风险控制 | 更长推理是否增加危险行为 | 高风险任务成功率、越权率 |

如果只比较“回答长度”，就会把多个轴混成一个指标。某些 chat model 在数学任务上也可能表现很好；某些 reasoning 产品在简单问答上反而会花费过多 token。

### 2.2 三层系统结构

为了避免把模型和系统混为一谈，可以把一次 reasoning 请求拆成三层：

1. **模型层**：参数、训练目标、表示能力和生成分布。
2. **策略层**：采样温度、候选数量、搜索算法、停止条件和 verifier 选择。
3. **环境层**：代码执行器、计算器、检索库、权限系统、队列和人工复核。

最终结果是三层相互作用的产物。比如代码任务的正确率提升，可能来自更强模型，也可能来自增加测试样例；合同助手的事实性提升，可能来自模型，也可能来自检索证据和引用校验。

### 2.3 公开接口中的 reasoning 控制

近年的公开 API 出现了 reasoning_effort、thinking、extended thinking 或相近概念。对初学者来说，它们可以理解为“允许服务为当前任务配置多少额外推理资源”；对专家来说，必须进一步追问：

1. 预算是 token 上限、时间上限、搜索节点，还是多个资源的组合？
2. 额外 token 是否真的被用于更深的计算，还是只是更长的表述？
3. 过程内容是否返回给调用方？返回的摘要是否等于内部完整轨迹？
4. 工具调用是否计入同一预算？重试和服务端路由如何计费？
5. 低、中、高档位是否保持同一个模型和同一个解码策略？

公开字段只能支持公开语义。它不能支持对闭源系统内部 tokenizer、路由器、奖励模型或隐藏状态的确定性猜测。

## 3. Chain-of-Thought：把中间步骤变成计算空间

### 3.1 小白视角：草稿纸的作用

人做多步题时会写草稿。草稿的价值不是让答案看起来更专业，而是把暂时的中间量放到外部工作区，减少一步计算覆盖另一步计算的风险。

Chain-of-Thought（CoT）提示让模型在最终答案之前生成中间推理文本。例如水箱问题可以写出净流入速度，再写时间。对于需要多次组合的任务，这相当于给模型更多 token 位置保存中间状态。

### 3.2 来龙去脉：从直接回答到分解任务

早期的问答系统经常被要求直接输出答案。当问题只需要事实检索时，这种形式很高效；当问题包含多步算术、逻辑链或程序构造时，直接生成最终 token 会让错误很难定位。

Chain-of-Thought 研究展示了一个重要现象：给出少量带推理过程的示例，或用自然语言提示模型先思考，可能让大模型在复杂推理基准上明显受益。这个结果不应被解读为“所有模型只要加一句提示就会推理”，因为收益依赖模型规模、任务难度、示例质量、答案抽取和评估协议。

### 3.3 生成概率仍然是自回归的

如果把问题和中间步骤拼成序列，模型仍然按照下式生成：

~~~math
p_{\theta}(z_{1:M},y_{1:T}\mid x)=\prod_{t=1}^{M+T}p_{\theta}(u_t\mid x,u_{1:t-1})
~~~

u 是把步骤 z 和答案 y 拼接后的 token 序列。CoT 并没有自动引入一个符号数学引擎；它提供的是更多可供模型继续预测的中间位置，以及一种由数据和提示塑造的解题格式。

### 3.4 显式过程、隐式计算与可见解释

需要区分三个概念：

- **显式过程**：模型输出给用户或记录系统的步骤文字。
- **隐式计算**：模型隐藏状态和注意力在生成 token 时完成的计算。
- **可验证轨迹**：能够被规则、程序或独立检查器确认的中间状态。

显式过程不一定等于隐式计算的完整记录。一个模型可能先形成答案，再生成一段看起来合理的解释；也可能因为输出格式要求而把真实计算拆成不自然的文字。若要研究过程是否有用，应进行干预：删除某一步、替换一个中间数字、要求模型继续计算，观察后续答案是否相应变化。

### 3.5 CoT 的收益和代价

CoT 可能带来四类收益：

1. 为多步计算提供工作空间。
2. 让错误出现在更容易定位的位置。
3. 为采样多条解法提供可比较的轨迹。
4. 为工具调用和过程监督提供中间接口。

代价也很具体：

1. 输出 token 增加，prefill、decode 和存储成本增加。
2. 错误会在后续步骤中传播。
3. 长解释可能泄露隐私、系统提示或危险计划。
4. 公开过程容易被用户误解为完整、忠实的内部思维记录。
5. 对简单任务，额外推理反而增加延迟而没有质量收益。

因此，CoT 是一种计算和接口设计，而不是质量保证。

## 4. Self-Consistency：用多个候选减少一次采样的偶然性

### 4.1 为什么一次答案不够

自回归生成每一步都可能做出概率上的选择。即使模型总体上知道解法，一次采样也可能在中间某一步选错。若改变随机种子或温度后，模型能够产生不同路径，就可以把多个路径作为候选集合。

设第 i 道题采样 K 条路径，抽取出的最终答案为 a_i1,...,a_iK。最简单的 self-consistency 是多数投票：

~~~math
\hat y_i^{\mathrm{sc}}=\mathrm{mode}(a_{i1},\ldots,a_{iK})
~~~

在 N 道题上的准确率为：

~~~math
A_{\mathrm{sc}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\left[\hat y_i^{\mathrm{sc}}=y_i^*\right]
~~~

mode 是出现次数最多的标准化答案，y_i^* 是参考答案。答案标准化很重要：10 分钟、10 和 十分钟可能是同一个语义答案，但字符串不同。

### 4.2 小白视角：投票为什么可能有用

设三条路径中两条独立地算出 10，一条算出 12。投票会选择 10。它利用的是“错误不完全相同”这个条件，而不是相信模型每次都正确。

如果每条路径独立地以概率 p 得到正确答案，且 K 为奇数，超过半数路径正确的概率是：

~~~math
P_{\mathrm{majority}}=\sum_{j=(K+1)/2}^{K}\binom{K}{j}p^j(1-p)^{K-j}
~~~

当 p > 0.5 且路径足够独立时，增加 K 可能提升多数正确的概率；当 p < 0.5 时，更多投票可能把系统性错误放大。真实模型的路径并不独立，所以这个公式只是帮助理解的理想化模型。

这个理想化概率要求 `K` 是正奇数、`0\le p\le1`，并且每条路径的正确事件可以近似视为独立同分布。真实系统常常有偶数个候选，此时必须规定平票如何处理；候选答案也可能无法解析，不能把解析失败悄悄当成某个答案参与投票。报告 self-consistency 时应同时给出有效候选数和解析失败数。

### 4.3 专家视角：相关错误决定收益上限

温度、prompt 和模型参数相同，生成的多条路径可能共享同一个误解：读错单位、漏掉否定词、套用错误定理或把训练集中的偏见重复出来。此时候选之间高度相关，投票不会创造新信息。

可以把候选路径的相关性作为分析对象。若所有候选都依赖同一个错误中间变量 q，那么增加样本只是在重复观察同一个错误条件；若通过不同提示、不同工具或不同搜索顺序产生互补路径，self-consistency 才更可能有额外收益。

工程上至少要记录：

1. 候选数量和每个候选的 token 数。
2. 温度、top-p、随机种子或采样策略。
3. 答案标准化规则。
4. 候选之间的答案一致率和步骤相似度。
5. 投票提升来自真正的多样性，还是来自某些简单题的重复正确。

### 4.4 Self-consistency 不是 best-of-n

self-consistency 根据最终答案的频次选择结果；best-of-n 通常根据一个评分器从候选中选择分数最高的路径。二者的输入相同，决策依据不同：

~~~math
\hat y_i^{\mathrm{ver}}=y_{ij},\qquad
j=\arg\max_{k\in\{1,\ldots,K\}}s_{ik}
~~~

s_ik 是第 k 条候选的验证分数。投票适合答案可标准化且多数正确的任务；verifier 选择适合存在明确检查规则或质量评分的任务。

### 4.5 Self-consistency 的失败模式

**答案抽取失败。** 模型虽然在过程里写对了，但最后格式不一致，导致投票统计错误。解决方法是定义严格的 parser，并把解析失败单独计数。

**多数错误。** 模型系统性地误解题意，所有候选都沿着同一错误方向走。此时需要改 prompt、加入反事实样本或使用外部验证器，而不是盲目增加 K。

**成本线性增长。** 如果每个候选平均使用 T 个 token，候选生成成本近似从 T 增加到 K T，还要加上答案抽取和投票延迟。

**选择偏差。** 只报告最好的 K 值，而不报告每个预算点的结果，会让质量—成本关系看起来比真实情况更好。

## 5. Verifier：让候选结果接受独立检查

### 5.1 小白视角：生成器和检查器分工

生成器负责提出答案，verifier 负责判断答案是否满足约束。两者可以是同一个模型的不同调用，也可以完全不同：

- 数学答案由符号计算器或数值程序检查；
- 代码由编译器和单元测试检查；
- JSON 由 schema validator 检查；
- 检索问答由引用和原文比对检查；
- 开放文本由人工或另一个评估模型检查。

这个分工的关键价值是把“会生成”与“会判定”分开。生成器不必一次就找到最好的答案，但候选必须有机会被可信的检查器筛选。

### 5.2 Outcome verifier：检查最终结果

Outcome verifier 只关心最终结果是否满足目标。例如水箱问题可以检查答案是否等于 10，代码问题可以运行测试用例，数据库查询可以比较结果集。

它通常便宜、清晰、容易自动化，但不能定位错误发生在哪一步。一个程序通过了已有测试，也不代表没有未覆盖的边界错误；一个数学答案正确，也不代表模型掌握了可迁移的方法。

### 5.3 Programmatic verifier：优先使用可执行约束

在代码、算术、格式和结构化输出任务中，程序化 verifier 往往比语言模型评分更可靠，因为它可以把任务要求写成确定规则。

例如，对一个函数候选 f，测试器可以定义：

~~~math
V_{\mathrm{code}}(f)=\frac{1}{M}\sum_{m=1}^{M}\mathbb{1}\left[f(x_m)=y_m\right]
~~~

M 是测试样例数量，x_m 和 y_m 是输入输出对。这个分数仍受测试集覆盖范围限制，不能把“通过已有测试”写成“程序在所有输入上正确”。

该式要求 `M>0`，每个测试样例的期望输出和比较规则已定义。空测试集不能产生“通过率 0”或“通过率 1”；它表示没有可验证的样本。若一个测试允许多个等价输出，应在比较器中明确等价关系，而不是依赖字符串完全相等。

程序化验证的工程风险包括：测试集泄漏、沙箱逃逸、资源耗尽、浮点误差、未定义行为和错误的断言。验证器本身也是软件，必须像生产代码一样测试。

### 5.4 Outcome verifier 与 process verifier

Process verifier 评估中间步骤。例如，它可以判断“净流入速度为 2”是否正确，或者判断程序是否先检查了空输入再访问数组。

过程级指标可以写成：

~~~math
A_{\mathrm{step}}=\frac{\sum_{i=1}^{N}\sum_{j=1}^{M_i}z_{ij}}{\sum_{i=1}^{N}M_i}
~~~

M_i 是第 i 个样本的步骤数，z_ij 在第 j 步符合标注或规则时取 1，否则取 0。

步骤准确率要求 `\sum_iM_i>0`，并且每个 `z_{ij}` 都是明确的二值或已约定的连续标签。没有步骤的样本可以单独报告为“不可评估”，不能通过给分母加 1 把它伪装成全错。不同轨迹的步骤粒度不一致时，平均值还需要按任务或轨迹规范化，否则长轨迹会获得更大的权重。

过程监督更容易定位错误，也能在最终答案尚未产生时提供反馈；它的难点是：

1. 同一道题可能存在多条都正确的解法。
2. 步骤边界本身可能有歧义。
3. 一个局部步骤看似正确，组合起来可能不满足全局约束。
4. 标注者可能把“表达规范”误当成“推理正确”。

### 5.5 Reward model 不等于 verifier

reward model 通常学习给候选结果打分。它可以处理没有确定程序答案的任务，但它学习的是代理目标：人类偏好、标注规则或历史选择。

verifier 更强调“是否满足某个可定义约束”；reward model 更强调“在训练数据分布中看起来更好”。二者可以组合，但不能互换。对于数学题，如果 reward model 喜欢格式整齐的错误证明，程序化答案检查仍然更重要。

### 5.6 Reward hacking：检查器也会被利用

只要训练或搜索过程针对一个评分器优化，模型就可能找到评分器的漏洞。例如：

- 代码只覆盖简单测试，模型专门拟合测试样例；
- 过程评分器偏好特定措辞，模型学习写“正确风格”而不是正确计算；
- 引用检查器只检查 URL 存在，模型生成与结论无关的链接；
- 格式验证器只检查字段存在，模型填入空值或默认值。

因此，验证器的分数不能直接当作真实质量。必须使用独立测试、隐藏切片、反事实样本和人工抽查。

## 6. Process Supervision：监督每一步，但不要神化步骤

### 6.1 为什么最终答案监督不够

只用最终答案训练时，一条长轨迹通常只收到一个整体信号：正确或错误。模型不知道是读题、计算、引用还是格式化出了问题。对于长链路任务，这个信号过于稀疏。

过程监督把轨迹拆成步骤，为每一步提供局部标签、奖励或检查结果。它的目标不是强迫所有模型使用同一种文字风格，而是让关键中间状态更容易被学习和审计。

### 6.2 过程标签的几种形式

过程标签可以是：

1. 二值标签：这一步正确或错误。
2. 连续分数：这一步对最终目标的贡献程度。
3. 结构标签：变量、前提、结论和依赖关系。
4. 工具结果：代码测试、检索命中或计算器输出。
5. 纠错标签：指出错误位置和可恢复动作。

不同标签对应不同成本。二值标签简单但信息少；详细纠错标签有用但昂贵，且更依赖标注者的技术能力。

### 6.3 局部正确和全局正确

考虑一个几何证明。每一步局部变换都可能遵守某个定理，但初始前提若被误读，最终结论仍然错误。反过来，一段简洁证明可能跳过读者熟悉的中间步骤，但整体是正确的。

所以过程质量不能只用平均步骤准确率衡量，还需要检查：

- 前提是否正确抽取；
- 依赖关系是否连贯；
- 关键变量是否被覆盖；
- 最终结论是否满足全局约束；
- 错误是否能被定位并修复。

### 6.4 过程监督的适用边界

数学推导、代码执行、规划状态和工具轨迹比较适合过程监督，因为中间状态有相对清晰的约束。开放式写作、审美判断和复杂社会决策的“正确过程”通常没有唯一标准，强行标注会把风格偏好伪装成推理真值。

这也是为什么 reasoning 研究不能只追求更长、更细的思维链。真正重要的是中间状态是否有任务意义，是否能够被独立检查，是否帮助系统在出错后恢复。

## 7. Search 与 Tree-of-Thought：从一条链变成候选空间

### 7.1 一条生成链的限制

普通自回归生成通常沿着一条路径前进：每产生一个 token，就把它放入上下文，后续继续生成。如果早期做出不可逆的错误，后面可能只能在错误前提上修补。

搜索方法把一些中间状态保留下来，让系统有机会比较多个下一步。

将状态记为 s，动作记为 a，转移结果记为 s'，搜索的基本关系是：

~~~math
s'=\mathcal{T}(s,a)
~~~

T 可以是语言模型生成的状态转移，也可以包含工具执行和环境反馈。每个状态有一个评分 q(s)，系统用它决定继续扩展、剪枝或回溯。

### 7.2 Tree-of-Thought 的直觉

Tree-of-Thought 把若干自然语言中间状态作为树节点：

1. 从当前状态生成多个可能的下一步。
2. 用启发式或 verifier 给节点打分。
3. 保留若干更有希望的节点。
4. 继续展开，直到得到终止状态或达到预算。

它比单链多了回看和分支能力，但也把候选管理、状态去重和成本控制问题引入系统。

### 7.3 搜索算法的层次

在总览层面可以把常见策略分成四类：

- **深度优先**：沿一条路径深入，失败后回溯，内存较低但可能浪费在坏分支上。
- **广度优先**：按层展开，容易找到浅层解，但节点数迅速增加。
- **Beam search**：每层保留固定数量的高分节点，成本可控但可能过早剪掉正确分支。
- **蒙特卡洛类搜索**：通过多次模拟估计节点价值，适合有明确状态转移和回报的环境。

语言模型的节点评分并不等于真实成功概率。高语言概率的路径可能只是常见表达，未必满足任务约束。

### 7.4 搜索成本的数量级

若每个节点平均生成 b 个分支，搜索深度为 d，不做剪枝时节点数近似为：

~~~math
N_{\mathrm{node}}\approx\sum_{\ell=0}^{d}b^\ell=\frac{b^{d+1}-1}{b-1}\quad(b\ne 1)
~~~

这解释了为什么搜索必须依赖剪枝、缓存、早停和强验证器。增加一个分支因子看似只增加一点多样性，实际可能使总调用量指数式增长。

这个节点数近似要求 `b\ge0` 为整数、深度 `d\ge0` 为整数，并且每一层都真的展开相同数量的分支。`b=1` 时应使用 `d+1`，不能套用分母为零的闭式；有剪枝、缓存或共享前缀时，实际节点数应以 trace 统计为准。

### 7.5 工具反馈是搜索的一部分

代码执行、计算器、检索和数据库并不是“外挂魔法”，它们可以看成环境：模型提出动作，环境返回 observation，模型再决定下一步。

~~~math
(s_t,a_t)\xrightarrow{\mathcal{E}}(o_{t+1},s_{t+1})
~~~

E 是环境，o 是观察。可靠系统还要定义超时、异常、权限拒绝、输出截断和重试语义。否则模型会把工具失败误读成业务结论。

## 8. Test-Time Compute：把更多计算花在推理时

### 8.1 训练计算和推理计算的差别

训练计算用于更新参数，让能力进入模型；推理计算用于当前请求的候选生成、搜索、验证和工具交互。前者通常摊薄到许多请求，后者直接影响每个用户请求的延迟和成本。

可以把推理预算写成一个向量：

~~~math
b=(T,K,N_{\mathrm{search}},N_{\mathrm{tool}},T_{\mathrm{verify}})
~~~

其中：

- T 是单条推理轨迹 token 预算；
- K 是候选数量；
- N_search 是搜索节点或展开次数；
- N_tool 是工具调用次数；
- T_verify 是验证器消耗的 token 或等价计算量。

增加某个分量，不一定能由另一个分量替代。多采样需要候选多样性，过程验证需要可检查轨迹，代码执行需要安全环境。

### 8.2 质量—预算曲线

把预算抽象为 B，任务成功率写成 A(B)。理想情况下，增加预算会带来收益但边际收益递减：

~~~math
\Delta A(B)=A(B+\Delta B)-A(B)
~~~

当 Delta A 已经很小，而延迟和成本仍明显增加时，继续提高预算不一定值得。真实曲线可能非单调：过长的推理会引入更多错误，搜索评分器会把正确路径剪掉，工具重试会污染上下文。

`A(B)` 只在同一任务分布、成功定义、模型版本和环境下比较才有意义；若 `B` 同时改变了模型、工具或 verifier，差异不能归因给推理预算。`\Delta B` 应为非负预算增量，且至少记录输入、输出、验证和工具的分项变化。

### 8.3 单位成功成本

只报告准确率会掩盖计算代价。可以在一个固定窗口内定义：

~~~math
C_{\mathrm{success}}=\frac{\sum_{i=1}^{N}C_i}{\sum_{i=1}^{N}\mathbb{1}[\mathrm{Success}_i=1]}
~~~

C_i 可以包含输入 token、输出 token、候选、验证器、工具、GPU 时间和人工复核。分母为 0 时，不能把结果写成一个正常数，应报告“窗口内没有成功样本”。

### 8.4 动态预算比固定高预算更接近真实系统

简单问题没有必要使用同样的推理资源。一个动态策略可以先用低预算尝试，再根据不确定性、验证失败或问题类型决定是否增加预算：

~~~math
B_i=B_{\mathrm{base}}+\Delta B_i(\mathrm{difficulty},\mathrm{uncertainty},\mathrm{failure})
~~~

这里 B_i 是第 i 个请求的总预算。关键难点是估计难度和不确定性：模型的自信分数往往不可靠，候选一致也可能是共同错误。因此动态分配必须用独立验证、历史切片和实际成本曲线校准。

动态预算还应设定有限上界和停止条件。若验证失败就无限重试，系统的平均成本、P99 延迟和工具副作用都可能失控；若 `\Delta B_i` 为负或不是有限数，预算定义也失去意义。实际策略应记录每次升级预算的触发原因和最终停止原因。

## 9. RLVR：可验证奖励推动训练路线变化

### 9.1 什么是可验证奖励

RLVR 是 reinforcement learning with verifiable rewards，即使用规则、程序或环境反馈产生奖励的强化学习路线。数学答案、代码测试、格式约束和工具执行结果都可以提供相对清晰的反馈。

对一个策略 pi_theta，简化的目标可以写成：

~~~math
J(\theta)=\mathbb{E}_{x\sim D,\;y\sim\pi_\theta(\cdot\mid x)}[r(x,y)]
~~~

r(x,y) 是验证器给出的奖励。实际训练通常还需要参考策略、KL 约束、长度控制、组内比较、奖励归一化和安全奖励；上式只是帮助理解“策略通过采样获得反馈”的骨架。

### 9.2 为什么可验证任务适合 RL

开放式帮助性很难为每个回答定义唯一真值。数学题可以检查最终数值，代码可以运行测试，工具调用可以检查状态变化。这些任务的 reward 比“某个评审者觉得更好”更容易自动化，也更适合进行大量在线采样。

但可验证不等于完整。代码测试没有覆盖的 bug、数学答案解析器的漏洞、工具环境中的错误状态，都可能让 reward 与真实目标脱钩。

### 9.3 DeepSeek-R1 提供了什么公开证据

DeepSeek-R1 的公开论文把冷启动数据、推理轨迹、强化学习、可验证奖励和蒸馏放在一条公开讨论较多的训练路线中。它的重要启发不是某个神奇的单一 loss，而是：在数学、代码等可验证任务上，模型可以通过在线生成候选并接受结果反馈来改进；随后再把较强模型的行为蒸馏到更小模型。

阅读这类材料时要保留三个边界：

1. 论文报告的是论文所使用的数据、模型和训练配方下的结果。
2. 公开论文不能证明所有 reasoning 产品都采用相同的 RL 流程。
3. 在开放式事实性、帮助性和安全任务上，RLVR 需要与偏好数据、检索、人工评估和安全回归配合。

### 9.4 Reward hacking 和训练退化

当 reward 变成优化目标，模型可能学习如何获得高分，而不是如何完成真实任务。常见现象包括：

- 生成更长但没有新信息的步骤；
- 针对答案解析器输出特殊格式；
- 过度调用工具以显示“认真”；
- 在测试集模式上过拟合；
- 为了通过安全检查而拒绝所有请求；
- 为了提高数学 reward 而牺牲语言清晰度和通用性。

治理方法包括独立验证器、隐藏测试、奖励分解、长度正则、能力回归、安全回归和真实任务抽样。reward 曲线向上不是充分证据，必须观察真实目标是否同步改善。

## 10. 数学、代码、规划与开放任务

### 10.1 数学推理：结果容易验证，过程不一定唯一

数学是 reasoning 研究常用的起点，因为很多任务有明确答案。GSM8K 聚焦小学到初中程度的多步应用题，MATH 覆盖更复杂的竞赛数学。它们适合研究算术、代数、几何和组合推理，但不能代表所有真实推理。

数学评估至少要区分：

1. 最终答案是否正确。
2. 单位、符号和等价形式是否被正确解析。
3. 证明中关键步骤是否成立。
4. 题目改写、数字替换或无关信息加入后是否仍然正确。
5. 是否使用了允许的计算工具。

一个模型在原题上高分，可能只是记住题目或解法。反事实改写、数值扰动和新题型更能检验迁移。

### 10.2 代码推理：可执行反馈改变了闭环

代码任务中，模型可以经历“读题—写代码—运行—读取错误—修复”的循环。编译器和测试器提供了比自然语言评分更硬的反馈。

HumanEval 常用 pass@k 描述从 k 次候选中至少得到一个通过测试的概率。设生成 n 个候选，其中 c 个通过，抽取 k 个至少包含一个通过候选的无放回估计为：

~~~math
\mathrm{pass@}k=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}
~~~

当 n-c < k 时组合项为 0，估计值为 1。它衡量候选集合中是否存在正确程序，不等于一次调用就能稳定交付，也不等于所有未覆盖输入都正确。

这个无放回估计要求 `0<k\le n`、`0\le c\le n` 且 `n`、`c`、`k` 为整数；当 `c=0` 时结果为 0。当候选集合为空，或者 `k` 大于候选数时，pass@k 没有按该实验定义的含义，不能通过自动截断 `k` 来制造一个结果。实际代码任务还要固定测试集、超时、资源限制和异常处理，否则“通过”不可复现。

### 10.3 规划任务：状态转移比漂亮叙述重要

规划需要把目标拆成动作，并保证动作序列在环境中可执行。一个计划可以表示为：

~~~math
\tau=(s_0,a_0,s_1,a_1,\ldots,s_H)
~~~

如果动作 a_t 在状态 s_t 下不合法，后续步骤再完整也没有意义。评估时应执行计划或模拟状态转移，而不是只让语言模型评价“这个计划听起来合理”。

### 10.4 开放任务：没有唯一真值时如何评估

写作、研究建议和复杂决策通常没有唯一答案。此时不能简单套用数学 exact match，应拆成事实支持、约束满足、风险、可执行性、引用质量和用户目标匹配等维度，并明确哪些结论需要人工判断。

LLM-as-a-Judge 可以作为成本较低的排序器，但它可能偏好长答案、特定风格或自己的表达。要校准它，需要人工标注集、盲测、不同模型交叉评估和分歧抽样。

## 11. Reasoning 评估：不只看“答对多少题”

### 11.1 最终结果指标

最基本的准确率是：

~~~math
A=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\hat y_i=y_i^*]
~~~

它易懂，但会掩盖题目难度、类别不均衡、答案解析和数据污染。代码任务可以用测试通过率；结构化输出还要报告 schema 合法率；工具任务要报告动作是否改变了正确资源。

这里要求 `N>0`，且每个样本的参考答案和等价判断已定义。不同难度切片应分别报告；将空集、解析失败或不可评估样本直接写进错误分母，会把数据质量问题误报成模型能力。

### 11.2 过程指标

过程指标可以包括步骤准确率、关键步骤召回率、错误定位准确率、纠错成功率和轨迹长度。过程指标必须先定义“什么叫一步”和“参考过程是否唯一”，否则数字看似精确，实际只是在测量标注协议。

### 11.3 鲁棒性与分布外泛化

推理能力应在输入扰动下保持：

- 改变数字但保持结构；
- 改变表述和语序；
- 加入无关信息；
- 调换实体名称；
- 改变单位和边界条件；
- 使用没见过的工具返回格式。

如果模型只在原始模板上正确，说明它可能学到了表面模式，而不是可迁移的解法。

### 11.4 Benchmark contamination

公开基准题可能出现在预训练语料、合成数据或网络解答中。污染会让测得的准确率高于真正的泛化能力。检查方法包括：

1. 训练数据去重和近重复搜索。
2. 评估题目改写与新题生成。
3. 时间切分，使用模型训练截止时间之后的数据。
4. 隐藏测试集和私有任务。
5. 报告原题与反事实题之间的差异。

不能因为没有发现精确字符串重合，就断言不存在语义污染；近似解法、题型模板和公开教程也可能泄露答案路径。

### 11.5 成本、延迟与可靠性

线上系统至少需要同时记录：

- 平均和 P95/P99 延迟；
- 输入、输出和验证 token；
- 候选数、搜索节点和工具调用数；
- 单请求成本与单位成功成本；
- 超时、解析失败、工具失败和人工接管比例；
- 高风险切片的错误率。

一个离线 benchmark 的提升，如果换来三倍成本、两倍延迟和更差的高风险错误率，不能直接视为系统升级。

## 12. Reasoning 的安全边界

### 12.1 能力提升也会扩大错误和攻击空间

更强的分解、规划和代码能力可以帮助用户解决困难任务，也可能帮助攻击者构造更有效的钓鱼流程、恶意代码或工具滥用计划。安全评估不能只测模型是否拒绝一句危险请求，还要测它是否会通过拆分任务、编码、工具调用或多轮上下文绕过约束。

### 12.2 不要把完整 CoT 当作安全审计日志

公开的长推理文本可能包含个人信息、系统提示、内部规则或危险操作细节；隐藏过程又不一定能被用户直接审计。更稳妥的系统通常保存结构化事件：输入类别、工具调用、验证结果、风险标签、最终决策和必要的可审计摘要，而不是默认把所有内部过程原样展示。

### 12.3 工具权限是推理系统的实际边界

模型输出一段危险文字和模型直接执行危险动作，风险不同。工具系统应采用最小权限、参数校验、资源隔离、人工确认、超时、审计和可撤销操作。即使 reasoning 模型能规划出完整步骤，也不意味着它拥有执行这些步骤的权限。

### 12.4 验证器和安全策略也会出错

安全分类器可能误报或漏报，代码沙箱可能存在资源问题，事实 verifier 可能把伪造引用当成证据。安全不是给 reasoning 链末尾加一个分类器就结束，而是把风险控制分布在输入、生成、工具、输出和监控各层。

## 13. 一个完整的推理系统案例

### 13.1 问题：合同助手计算付款金额

假设用户上传合同并问：“本月应付金额是多少？”系统需要：

1. 找到基础金额、折扣、税率和付款条件。
2. 区分正文条款与附件中的旧版本。
3. 计算金额。
4. 给出页码或段落证据。
5. 遇到缺失税率时明确表示无法确定，而不是猜一个数字。

这已经不是单纯的文本生成。它同时包含文档检索、结构化抽取、算术、证据对齐、拒答和权限控制。

### 13.2 可能的系统轨迹

一条合格轨迹可能是：

~~~text
检索：合同正文第 3 页，基础金额 100000 元。
检索：第 4 页，折扣 5%。
检索：第 7 页，税率 6%。
计算：100000 × (1 - 0.05) × (1 + 0.06) = 100700 元。
验证：金额表达式可复算，三个字段都带有来源位置。
输出：本月应付金额为 100700 元，并列出三处证据。
~~~

但系统还应考虑：

- 另一处附件写着“折扣仅适用于上一季度”；
- OCR 把 6% 识别成 60%；
- 合同存在多个币种；
- 计算器成功但字段来源错配；
- 用户没有查看该合同的权限。

### 13.3 把成功拆成多个事件

可以定义字段抽取、证据支持、算术和权限四个事件：

~~~math
S_{\mathrm{total}}=S_{\mathrm{field}}\land S_{\mathrm{evidence}}\land S_{\mathrm{calc}}\land S_{\mathrm{auth}}
~~~

只有四个事件都成功，系统才应把结果作为可执行建议交付。若字段抽取正确但证据位置错误，系统应报告需要复核，而不是把局部成功伪装成完整成功。

### 13.4 预算分配

简单合同可以直接抽取并计算；复杂合同则可能需要更多检索、候选抽取和交叉验证。预算分配可以表示为：

~~~math
C_{\mathrm{request}}=C_{\mathrm{retrieve}}+C_{\mathrm{reason}}+C_{\mathrm{verify}}+C_{\mathrm{tool}}+C_{\mathrm{review}}
~~~

每一项都可能成为瓶颈。增加 reasoning token 不能修复 OCR 错字，增加检索数量也不能替代算术验证，增加工具调用又可能提高权限和延迟风险。

### 13.5 失败归因

这个案例中至少有五种不同失败：

1. **召回失败**：没有找到真正适用的条款。
2. **解析失败**：找到了条款但把数字读错。
3. **推理失败**：字段正确但计算或条件判断错误。
4. **证据失败**：结果正确但引用了错误位置。
5. **治理失败**：用户没有权限，或系统把不确定结果直接用于付款。

把它们都称为“模型 reasoning 错误”会让排查失去方向。完整系统必须让每类失败有不同的指标和修复路径。

## 14. 最小可运行实验：比较一次生成、投票和验证器

下面的实验不调用真实模型，而是把四道题的候选结果写成小型数据集。它的目的不是模拟真实模型分布，而是让读者亲自看见几个量如何变化：

1. greedy 只看第一条候选。
2. self-consistency 对最终答案投票。
3. verifier reranking 选择分数最高的候选。
4. pass@k 只衡量候选集合中是否存在正确答案。
5. 过程步骤准确率和 token 成本需要单独报告。

代码中的 correct 是实验构造的真值，verifier 是模拟评分器，不应被理解为真实模型的内部概率。

~~~python
import math
from collections import Counter
from math import comb


problems = [
    {
        "id": "water_tank",
        "gold": "10",
        "prompt_tokens": 40,
        "candidates": [
            {"answer": "12", "correct": False, "verifier": 0.25, "tokens": 42, "steps": [1, 0]},
            {"answer": "10", "correct": True, "verifier": 0.93, "tokens": 50, "steps": [1, 1, 1]},
            {"answer": "10", "correct": True, "verifier": 0.88, "tokens": 48, "steps": [1, 1]},
        ],
    },
    {
        "id": "code_sum",
        "gold": "pass",
        "prompt_tokens": 40,
        "candidates": [
            {"answer": "fail", "correct": False, "verifier": 0.45, "tokens": 55, "steps": [1, 0, 0]},
            {"answer": "pass", "correct": True, "verifier": 0.91, "tokens": 60, "steps": [1, 1, 1]},
            {"answer": "pass", "correct": True, "verifier": 0.87, "tokens": 58, "steps": [1, 1, 0]},
        ],
    },
    {
        "id": "logic_grid",
        "gold": "blue",
        "prompt_tokens": 40,
        "candidates": [
            {"answer": "blue", "correct": True, "verifier": 0.86, "tokens": 35, "steps": [1, 1]},
            {"answer": "blue", "correct": True, "verifier": 0.84, "tokens": 37, "steps": [1, 1]},
            {"answer": "blue", "correct": True, "verifier": 0.82, "tokens": 36, "steps": [1, 1]},
        ],
    },
    {
        "id": "distractor_math",
        "gold": "ignore",
        "prompt_tokens": 40,
        "candidates": [
            {"answer": "use_extra", "correct": False, "verifier": 0.55, "tokens": 45, "steps": [1, 0]},
            {"answer": "use_extra", "correct": False, "verifier": 0.51, "tokens": 44, "steps": [0, 0]},
            {"answer": "ignore", "correct": True, "verifier": 0.89, "tokens": 52, "steps": [1, 1, 1]},
        ],
    },
]


def majority_answer(candidates):
    if not candidates:
        raise ValueError("candidates must not be empty")
    if any(not isinstance(candidate.get("answer"), str) or not candidate["answer"] for candidate in candidates):
        raise ValueError("candidate answers must be non-empty strings")
    counts = Counter(candidate["answer"] for candidate in candidates)
    max_count = max(counts.values())
    winners = {answer for answer, count in counts.items() if count == max_count}
    for candidate in candidates:
        if candidate["answer"] in winners:
            return candidate["answer"]
    raise AssertionError("unreachable")


def pass_at_k(n, c, k):
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (n, c, k)):
        raise TypeError("n, c and k must be integers")
    if n <= 0 or c < 0 or c > n or k <= 0 or k > n:
        raise ValueError("pass_at_k requires 0 <= c <= n and 0 < k <= n")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def audit_reasoning_budget(problems, k=2):
    if not isinstance(problems, list) or not problems:
        raise ValueError("problems must be a non-empty list")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    greedy_correct = 0
    sc_correct = 0
    verifier_correct = 0
    pass1_values = []
    passk_values = []
    total_candidate_tokens = 0
    total_prompt_tokens = 0
    step_correct = 0
    step_total = 0
    per_problem = {}

    for problem in problems:
        if not isinstance(problem, dict) or not {"id", "gold", "prompt_tokens", "candidates"} <= set(problem):
            raise ValueError("each problem must contain id, gold, prompt_tokens and candidates")
        if not isinstance(problem["id"], str) or not problem["id"] or problem["id"] in per_problem:
            raise ValueError("problem ids must be unique non-empty strings")
        candidates = problem["candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("each problem must have non-empty candidates")
        if len(candidates) < k:
            raise ValueError("each problem must provide at least k candidates")
        if isinstance(problem["prompt_tokens"], bool) or not isinstance(problem["prompt_tokens"], int) or problem["prompt_tokens"] < 0:
            raise ValueError("prompt_tokens must be a non-negative integer")
        if not isinstance(problem["gold"], str) or not problem["gold"]:
            raise ValueError("gold must be a non-empty string")
        for candidate in candidates:
            required = {"answer", "correct", "verifier", "tokens", "steps"}
            if not isinstance(candidate, dict) or set(candidate) != required:
                raise ValueError("candidate schema is invalid")
            if not isinstance(candidate["correct"], bool):
                raise TypeError("correct must be boolean")
            if isinstance(candidate["verifier"], bool) or not isinstance(candidate["verifier"], (int, float)) or not math.isfinite(candidate["verifier"]):
                raise ValueError("verifier must be finite")
            if isinstance(candidate["tokens"], bool) or not isinstance(candidate["tokens"], int) or candidate["tokens"] < 0:
                raise ValueError("candidate tokens must be non-negative integers")
            if not isinstance(candidate["steps"], list) or not candidate["steps"] or any(step not in (0, 1) for step in candidate["steps"]):
                raise ValueError("steps must be a non-empty list of binary labels")
        gold = problem["gold"]
        if any(candidate["correct"] != (candidate["answer"] == gold) for candidate in candidates):
            raise ValueError("candidate correct labels must agree with exact gold answers")
        greedy = candidates[0]["answer"]
        sc = majority_answer(candidates)
        best = max(candidates, key=lambda item: item["verifier"])["answer"]
        correct_count = sum(candidate["correct"] for candidate in candidates)
        n = len(candidates)

        greedy_correct += greedy == gold
        sc_correct += sc == gold
        verifier_correct += best == gold
        pass1_values.append(pass_at_k(n, correct_count, 1))
        passk_values.append(pass_at_k(n, correct_count, k))
        total_candidate_tokens += sum(candidate["tokens"] for candidate in candidates)
        total_prompt_tokens += problem["prompt_tokens"] * n
        step_correct += sum(sum(candidate["steps"]) for candidate in candidates)
        step_total += sum(len(candidate["steps"]) for candidate in candidates)
        per_problem[problem["id"]] = {
            "greedy": greedy,
            "self_consistency": sc,
            "verifier": best,
            "correct_candidates": correct_count,
        }

    total_tokens = total_prompt_tokens + total_candidate_tokens
    n_problems = len(problems)
    report = {
        "greedy_accuracy": round(greedy_correct / n_problems, 3),
        "self_consistency_accuracy": round(sc_correct / n_problems, 3),
        "verifier_accuracy": round(verifier_correct / n_problems, 3),
        "pass_at_1_est": round(sum(pass1_values) / n_problems, 3),
        "pass_at_2_est": round(sum(passk_values) / n_problems, 3),
        "avg_candidates": round(sum(len(p["candidates"]) for p in problems) / n_problems, 3),
        "process_step_accuracy": round(step_correct / step_total, 3),
        "total_tokens": total_tokens,
        "cost_per_verified_correct": (
            round(total_tokens / verifier_correct, 3) if verifier_correct else None
        ),
        "per_problem": per_problem,
    }
    report["verifier_improvement"] = round(
        report["verifier_accuracy"] - report["greedy_accuracy"], 3
    )
    report["self_consistency_improvement"] = round(
        report["self_consistency_accuracy"] - report["greedy_accuracy"], 3
    )
    return report


report = audit_reasoning_budget(problems)
for key, value in report.items():
    print(f"{key}={value}")
~~~

预期输出为：

~~~text
greedy_accuracy=0.25
self_consistency_accuracy=0.75
verifier_accuracy=1.0
pass_at_1_est=0.667
pass_at_2_est=0.917
avg_candidates=3.0
process_step_accuracy=0.759
total_tokens=1042
cost_per_verified_correct=260.5
per_problem={'water_tank': {'greedy': '12', 'self_consistency': '10', 'verifier': '10', 'correct_candidates': 2}, 'code_sum': {'greedy': 'fail', 'self_consistency': 'pass', 'verifier': 'pass', 'correct_candidates': 2}, 'logic_grid': {'greedy': 'blue', 'self_consistency': 'blue', 'verifier': 'blue', 'correct_candidates': 3}, 'distractor_math': {'greedy': 'use_extra', 'self_consistency': 'use_extra', 'verifier': 'ignore', 'correct_candidates': 1}}
verifier_improvement=0.75
self_consistency_improvement=0.5
~~~

这个输出必须这样读：在这组人为构造的数据上，verifier 选择正确答案的比例高于 greedy；self-consistency 也有提升，但对 distractor_math 仍然被两个相同的错误答案带偏；pass@2 比一次候选更高，表示候选集合中更可能包含正确答案；它不表示系统已经能稳定选出正确答案。

total_tokens=1042 是三条候选的输入和输出 token 总和，不是实际 GPU 成本。真实系统还要把模型规格、批处理、缓存命中、验证器和工具执行时间加入成本账本。

## 15. 如何把一次实验读成知识，而不是只记住数字

### 15.1 先问实验对象是什么

实验到底改变了什么？是模型参数、采样策略、候选数量、验证器、工具权限，还是数据切分？如果同时改变多个变量，就不能把结果归因给其中某一个方法。

### 15.2 再问成功事件是否定义清楚

“答案看起来不错”不是可复现实验定义。应该明确：

- 数学答案是否允许等价形式；
- 代码是否在隐藏测试上通过；
- 工具是否真的改变了正确资源；
- 证据是否支持每个关键 claim；
- 拒答是否在危险请求上正确发生。

### 15.3 最后看收益是否值得成本

如果 self-consistency 从 70% 提到 75%，但 token 成本变成 4 倍，就要继续问：

1. 提升集中在哪些任务切片？
2. 高风险样本是否也提升？
3. 延迟是否满足业务约束？
4. 是否可以只给困难样本使用多采样？
5. verifier 是否比增加候选更划算？

这才是 reasoning 系统的工程判断，而不是把单个 benchmark 数字当成结论。

## 16. 常见误区与纠正

### 误区一：输出越长，推理越强

长输出可能只是重复、解释或格式噪声。应检查关键中间状态、最终结果和独立验证。

### 误区二：CoT 就是模型真实思维过程

CoT 是可观察的生成轨迹，不自动等于内部计算的完整因果记录。需要通过步骤干预、工具验证和反事实实验研究它是否真正发挥了作用。

### 误区三：pass@k 高，就说明一次调用可靠

pass@k 衡量候选集合中至少有一个成功解的概率；线上系统还需要知道如何识别并交付那个正确解。

### 误区四：verifier 分数高，就说明最终答案正确

verifier 可能有盲点、偏差或漏洞。必须用独立测试集、隐藏约束和人工抽查校准它。

### 误区五：RLVR 能解决所有 reasoning

RLVR 对可验证任务很有价值，但开放式帮助性、事实性、价值判断和安全仍需要其他监督与系统控制。

### 误区六：高 reasoning effort 永远更好

高预算可能提升难题正确率，也可能带来过度思考、延迟、成本和更多工具风险。应该报告质量—成本曲线，而不是只报告一个最高档位。

### 误区七：一个模型排行榜能代表 reasoning 能力

不同模型的默认预算、工具、上下文、答案解析、采样和评估集可能不同。排行榜没有实验协议就缺少可比性。

## 17. 小练习

### 练习一：区分三种成功

构造一个“最终答案正确但过程错误”的例子，再构造一个“过程大部分正确但最终格式错误”的例子。分别说明只看最终答案和只看过程会造成什么误判。

### 练习二：计算多数投票概率

假设单条路径正确概率为 0.6，路径近似独立。计算 K=1、K=3 和 K=5 时多数正确的理想化概率，并说明为什么真实结果可能更低。

### 练习三：设计一个 verifier

为“模型生成 SQL 查询”设计 outcome verifier。写出至少三条测试约束，并说明测试集覆盖不足会怎样造成 reward hacking。

### 练习四：比较两个预算策略

策略 A 对所有请求生成一条 100 token 的轨迹；策略 B 先生成一条 40 token 的轨迹，验证失败时再增加 160 token。设计一组任务难度分布，比较两者的平均 token、成功率和 P95 延迟。

### 练习五：反事实推理评估

把水箱问题中的“进水 3 升、漏水 1 升、容量 20 升”分别改成三组数字，再加入一个无关温度信息。说明哪些变化测试算术迁移，哪些变化测试抗干扰能力。

### 练习六：规划与执行

为“整理一个项目目录并运行测试”写出状态、动作、观察和失败恢复动作。指出哪些动作必须经过权限确认。

### 练习七：查资料并标注证据

阅读 Chain-of-Thought、Self-Consistency、Tree of Thoughts 和 DeepSeek-R1 的原始资料。为每个来源记录：它解决的问题、实验对象、主要结论、没有证明什么，以及它与本章其他方法的关系。

## 18. 本章总结

推理模型的核心不是把答案写得更长，而是让系统在多步、可验证或需要规划的任务中更可靠地形成、检查和修正中间状态。

本章建立了以下关系：

1. next-token prediction 提供了生成基础，但不保证每一步推理正确。
2. CoT 为中间计算提供显式工作空间，但可见解释不必然忠实。
3. self-consistency 用多个候选降低一次采样的偶然性，但受相关错误和成本限制。
4. verifier 把候选生成与结果判断分开；程序化验证通常比语言风格评分更硬。
5. process supervision 提供更细粒度信号，但过程标签不一定唯一，也不自动保证全局正确。
6. search 和工具把推理从单链扩展为状态、动作、环境反馈组成的轨迹。
7. test-time compute 是质量、成本、延迟、可靠性和风险之间的资源分配问题。
8. RLVR 适合有可执行反馈的任务，但 reward 仍可能被利用，不能代表所有 reasoning 能力。
9. 评估必须同时看最终结果、过程、鲁棒性、污染、成本和高风险切片。
10. reasoning 系统的安全边界最终由工具权限、验证器、监控和失败处理共同决定。

下一章进入 Chain-of-Thought，继续追踪它从提示方法到训练数据、显式轨迹、隐式计算和可验证过程的演化。

## 19. 资料索引

以下链接优先保留原始论文、官方文档或官方仓库。论文页面和文档版本会变化，读者复现实验时应记录访问日期、模型版本、解码参数和评估脚本。

1. Chain-of-Thought Prompting：<https://arxiv.org/abs/2201.11903>
2. Zero-shot Reasoners：<https://arxiv.org/abs/2205.11916>
3. Self-Consistency Improves Chain of Thought Reasoning：<https://arxiv.org/abs/2203.11171>
4. Training Verifiers to Solve Math Word Problems：<https://arxiv.org/abs/2110.14168>
5. Let's Verify Step by Step：<https://arxiv.org/abs/2305.20050>
6. Tree of Thoughts：<https://arxiv.org/abs/2305.10601>
7. GSM8K：<https://arxiv.org/abs/2110.14116>
8. MATH：<https://arxiv.org/abs/2103.03874>
9. HumanEval：<https://arxiv.org/abs/2107.03374>
10. DeepSeek-R1：<https://arxiv.org/abs/2501.12948>
11. OpenAI reasoning guide：<https://platform.openai.com/docs/guides/reasoning>
12. Anthropic extended thinking：<https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking>
13. Gemini thinking：<https://ai.google.dev/gemini-api/docs/thinking>

本章写作时已对上述入口进行联网访问；其中论文入口返回正常 HTTP 响应，三个官方文档入口也可访问。资料只用于支持公开方法、公开接口和公开实验，未将闭源模型的未披露训练细节写成事实。
