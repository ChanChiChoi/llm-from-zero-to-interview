# 第四章：Verifier 与 Reward Model

生成模型擅长提出候选，但“能提出一个看起来合理的答案”与“能判断哪个候选真的满足任务”是两种能力。Verifier 和 reward model 试图把这两件事分开。

在数学题中，生成器可以写出多条解法，程序或模型检查器判断最终答案；在代码题中，生成器提出程序，编译器和测试集提供反馈；在开放式任务中，reward model 学习人类或规则的偏好，再对候选排序。

这种生成—验证分工很有用，却也很危险。验证器并不是天然正确的裁判，它可能偏好长答案、整齐格式、自信语气或训练数据中的表面模式。只要生成模型针对评分器优化，就可能出现 reward hacking：分数提高了，真实任务质量却没有提高。

本章要建立的不是“有一个分数就能选最优答案”的直觉，而是完整的判断链：

1. 先定义验证对象和成功事件。
2. 再选择 outcome、process、programmatic 或 learned verifier。
3. 根据数据形式选择 pointwise、pairwise 或 listwise 训练。
4. 在真实候选分布和 hard negative 上评估。
5. 检查排序、校准、下游选择、成本和安全。
6. 当证据不足时允许 abstain，而不是强行输出高分候选。

## 0.1 本章的资料与证据边界

Training Verifiers to Solve Math Word Problems 说明了训练验证器来筛选数学候选的基本思路；Let's Verify Step by Step 进一步讨论过程级验证和步骤反馈。InstructGPT、Learning to Summarize from Human Feedback 等工作提供 reward model 在人类偏好优化中的背景；RewardBench 提供了对 reward model 和偏好模型进行多维评估的公开路线。

这些资料的任务目标不同。数学答案验证器、偏好 reward model、代码测试器和安全分类器不能简单互换。论文中的 pairwise accuracy 也不能自动推导出真实候选集合上的 top-1 选择准确率。

本章会区分：

1. 论文定义与论文实验；
2. 官方代码或框架接口；
3. 教学构造的分数和数据；
4. 目标系统在真实任务分布上的复测。

闭源模型内部的 reward model 结构、训练数据和路由策略，除非官方公开，否则不写成确定事实。

## 0.2 一个最小例子

水箱题的三个候选是：

~~~text
候选 A：净流入速度是 3 + 1 = 4，答案是 5 分钟。
候选 B：净流入速度是 3 - 1 = 2，答案是 10 分钟。
候选 C：答案是 10 分钟，但没有写过程。
~~~

如果任务只要求最终数值，A 错，B 和 C 对；如果任务要求可审计过程，B 比 C 更容易检查；如果系统有一个只偏好长解释的 reward model，A 可能因为写得很完整而获得高分。

一个好的验证系统应先明确任务契约：

- 最终答案是否正确；
- 过程中的关键算术是否正确；
- 是否必须给出证据；
- 是否允许省略中间步骤；
- 缺少条件时是否应该拒答。

没有任务契约，验证器分数没有明确含义。

## 1. Verifier 的对象与职责

### 1.1 小白视角：生成器和检查器分工

生成器负责提出候选，verifier 负责判断候选是否满足约束。检查器可以是：

- 算术程序；
- 单元测试；
- JSON schema；
- 引用和原文比对；
- 符号证明系统；
- 人工标注；
- 学习出来的 reward model。

生成器不必一次找到唯一答案，但候选必须接受独立检查。检查器也不必写出答案，它只需要提供足以做出选择或拒答的信号。

### 1.2 专家视角：验证函数

设输入为 x，候选轨迹为 z，最终答案为 y，任务真实成功事件为 S。一个理想验证器可以写成：

~~~math
V^*(x,z,y)=\mathbb{P}(S=1\mid x,z,y)
~~~

现实中的 verifier 只能学习或近似 V^*：

~~~math
s=f_{\phi}(x,z,y)
~~~

phi 是验证器参数，s 是分数。分数可能是二值、连续值、排序分数或拒答信号。只有当 s 在目标候选分布上与真实成功事件有足够关系时，它才适合用于选择。

### 1.3 候选集合

对第 i 道题，候选集合可以写成：

~~~math
\mathcal{C}_i=
\left\{(z_{ij},\hat y_{ij},s_{ij},p_{ij},T_{ij})\right\}_{j=1}^{K_i}
~~~

其中：

- z_ij 是候选过程；
- y_hat_ij 是候选答案；
- s_ij 是 learned verifier 或 reward model 分数；
- p_ij 是程序或工具验证结果；
- T_ij 是生成候选消耗的 token 或时间。

最终系统不是单纯求 max(s)。它还要处理程序验证、权限、成本、不确定性和候选之间的依赖。

## 2. Outcome Verifier：检查最后结果

### 2.1 定义

Outcome verifier 只检查最终结果是否满足任务目标。例如：

- 数学答案是否等于参考数值；
- 程序是否通过隐藏测试；
- JSON 是否符合 schema；
- SQL 返回结果是否满足断言；
- 工具动作是否改变了正确资源。

它不一定需要读取完整推理链。

### 2.2 优点

Outcome verifier 的优势是目标清晰、容易自动化、通常成本较低。对于可执行任务，它可以直接调用外部环境，而不是让语言模型判断自己的答案。

### 2.3 局限

它也有明显边界：

1. 不能定位中间错误；
2. 可能把碰巧正确的答案计为成功；
3. 依赖测试集或规则覆盖；
4. 对开放答案和多种正确表达不够灵活；
5. 测试器本身可能存在 bug。

因此 outcome 成功不等于过程可靠，更不等于模型掌握了可迁移的方法。

### 2.4 代码测试的形式化

对候选程序 f，给定 M 个测试样例，可以定义：

~~~math
V_{\mathrm{code}}(f)=
\frac{1}{M}\sum_{m=1}^{M}
\mathbb{1}\left[f(x_m)=y_m\right]
~~~

x_m 和 y_m 是测试输入与期望输出。这里要求测试集大小 \(M>0\)，每个测试的期望输出和比较规则已定义；编译异常、超时和资源耗尽也应有明确状态，不能在统计时悄悄当成普通的错误输出。这个分数是测试集通过率，不是所有可能输入上的正确率。

隐藏测试、边界样例、资源限制和随机测试可以提高覆盖，但也会增加执行成本。

## 3. Process Verifier：检查中间步骤

### 3.1 为什么需要过程检查

最终答案监督是一个稀疏信号。长推理链可能在读题、变量抽取、计算、引用或格式化中的某一步出错；只看末尾答案无法知道错误位置。

Process verifier 为步骤提供局部分数或标签，使系统可以：

1. 提前发现错误；
2. 对搜索节点剪枝；
3. 给训练过程提供更细粒度反馈；
4. 生成可定位的失败报告。

### 3.2 步骤分数

设第 i 条轨迹有 M_i 个步骤，第 m 步的正确性分数为 q_im，整条过程的平均分可以写成：

~~~math
S_{\mathrm{proc}}(z_i)=
\frac{1}{M_i}\sum_{m=1}^{M_i}q_{im}
~~~

这个平均值要求 \(M_i>0\)。如果 q_im 被解释为概率或正确性分数，应规定它是有限数并位于约定范围（通常是 \([0,1]\)）；空轨迹的过程分数未定义，不能用 0 代替。q_im 可以是 0/1 标签，也可以是连续分数。平均值只是一个聚合方式，不能替代对关键步骤的检查。

### 3.3 局部正确不等于全局正确

一个证明中的每个局部变换看起来都像合法公式，但初始条件可能被读错；一个计划中的每个动作都单独合法，但组合后可能访问了错误资源。

过程检查还要关注：

- 前提是否来自输入；
- 变量依赖是否连贯；
- 关键条件是否被覆盖；
- 步骤是否顺序正确；
- 最终结论是否满足全局约束。

### 3.4 步骤粒度

“一步”没有天然定义。下面三种切分会得到不同指标：

1. 按句子切分；
2. 按数学操作切分；
3. 按可验证状态切分。

句子切分便于标注，但可能把两个操作混在一起；数学操作切分更细，但标注成本高；状态切分适合工具和搜索，却需要明确状态 schema。

### 3.5 多种正确过程

同一道题可能有代数法、枚举法和几何法。若过程监督只模仿一条参考轨迹，验证器可能把其他正确方法错判为低分。

因此过程数据应尽量使用多条正确轨迹、规则验证和任务结果检查，避免把表达风格误当成真值。

## 4. Programmatic Verifier：把约束交给程序

### 4.1 适用任务

程序化 verifier 适合：

- 精确算术；
- 代码编译和测试；
- SQL 执行；
- JSON schema；
- 日期和单位规则；
- 形式化证明；
- 工具权限与参数检查。

它们通常比语言模型更适合判断明确约束，因为规则可以重复执行，结果也容易复现。

### 4.2 程序验证不是绝对真理

程序只能检查被写入的约束。测试不覆盖的 bug、错误的 schema、过宽的数值容差和不完整的引用规则都会造成假成功。

因此需要测试 verifier 本身：

1. 用已知正确和已知错误样例测试；
2. 加入边界和反例；
3. 做实现独立性检查；
4. 记录超时、异常和资源耗尽；
5. 定期更新隐藏测试。

### 4.3 沙箱与权限

让模型生成代码并执行，必须考虑：

- 文件系统访问；
- 网络访问；
- CPU、内存和时间限制；
- 子进程和系统调用；
- 随机性；
- 输出大小；
- 依赖版本。

一个程序化 verifier 可能在结果判断上很可靠，却在执行环境上带来新的安全风险。

### 4.4 结构化输出验证

对 JSON 或工具参数，验证器可以分层：

1. 语法是否可解析；
2. schema 是否匹配；
3. 字段是否满足业务约束；
4. 资源和权限是否允许；
5. 动作结果是否与预期一致。

只检查第一层不能证明工具动作安全。

## 5. Reward Model 与 Verifier 的关系

### 5.1 Reward model 学习代理目标

reward model 通常从人工偏好、规则标签、程序结果或历史选择中学习一个评分函数。它的分数不是任务真值，而是对任务真值或偏好目标的近似。

可以把 reward model 写成：

~~~math
r_{\phi}(x,y,z)=f_{\phi}(x,y,z)
~~~

它可能学习正确性、帮助性、格式、安全、风格或多种目标的组合。

### 5.2 Verifier 更窄，reward model 更宽

Verifier 通常围绕一个明确约束：答案是否正确、步骤是否成立、程序是否通过测试。Reward model 可以处理开放式偏好，但也更容易受到标注风格、长度和语气的影响。

二者可以重叠，但不能简单等同：

- 程序测试器是 verifier，不是学习到的人类偏好模型；
- 一个偏好长答案的 reward model 不一定能验证数学；
- 数学验证器的高分不代表回答更有帮助或更安全。

### 5.3 Outcome Reward Model 与 Process Reward Model

可以用 ORM 表示 outcome reward model，用 PRM 表示 process reward model：

- ORM 关注最终答案或整条输出；
- PRM 关注中间步骤或状态转移。

ORM 适合最终结果明确的任务；PRM 更适合搜索、过程监督和早停，但标注与校准更复杂。

### 5.4 一个分数不应承担所有目标

把正确性、简洁、风格、安全和帮助性压成一个标量，便于排序，却会掩盖目标冲突。一个候选可能数学上正确但泄露隐私，或者格式完美但引用不支持结论。

工程上可以使用多个分数：

~~~math
S_{\mathrm{total}}
=
\alpha S_{\mathrm{correct}}
\;+\;
\beta S_{\mathrm{evidence}}
\;+\;
\gamma S_{\mathrm{format}}
\;-\;
\delta S_{\mathrm{risk}}
~~~

alpha、beta、gamma、delta 不是通用常数，需要在业务验证集和风险切片上校准。若把它们用于数值计算，应明确它们为有限权重，并确认各分数已经在可比较的尺度上；线性组合本身不会自动解决量纲、阈值或安全硬约束问题。

## 6. Verifier 的训练目标

### 6.1 Pointwise：每个候选单独判断

给每个候选一个标签 q，q=1 表示满足任务，q=0 表示不满足，二分类损失可写成：

~~~math
\mathcal{L}_{\mathrm{point}}
=
-q\log\sigma(s)
-(1-q)\log(1-\sigma(s))
~~~

s 是 verifier 的 logit，q 必须是 0 或 1，且每个 logit 和损失项都应为有限数。实际实现通常使用数值稳定的 binary-cross-entropy-with-logits，而不是先计算极端 sigmoid 再取对数。Pointwise 训练简单，适合有明确正负标签的任务，但分数跨问题是否可比较，需要额外校准。

### 6.2 Pairwise：比较两个候选

给出 winner w 和 loser l，让评分器满足 r_w 大于 r_l：

~~~math
\mathcal{L}_{\mathrm{pair}}
=-\log\sigma(r_w-r_l)
~~~

这里要求 winner 与 loser 的标签关系已经定义，且两个分数为有限数；如果两者在任务上等价，应单独标为 tie，不能随意制造 winner/loser。Pairwise 标注常比绝对打分容易，适合偏好数据和候选排序。但它只提供相对关系，不能直接得到跨任务的成功概率。

### 6.3 Listwise：直接学习候选列表

对一个候选列表，假设 j+ 是标注的较优候选：

~~~math
\mathcal{L}_{\mathrm{list}}
=
-\log
\frac{\exp(s_{i,j^+})}
{\sum_{j=1}^{K_i}\exp(s_{ij})}
~~~

这个式子要求 \(K_i>0\)，并且标注位置 \(j^+\) 满足 \(1\le j^+\le K_i\)；空列表或没有可接受候选时，listwise loss 未定义。实现时还应使用 log-sum-exp 形式避免指数溢出。Listwise 目标更贴近 reranking，但需要列表级数据和稳定的候选顺序协议。它也可能把候选数量和位置偏差带入训练。

### 6.4 Ranking loss 不等于校准损失

一个模型可以非常擅长把好候选排在坏候选前面，却输出没有概率意义的分数。若分数要用于阈值拒答、动态预算或成本决策，还要单独做 calibration。

## 7. Verifier 数据与 hard negative

### 7.1 数据来源

Verifier 训练数据可以来自：

1. 人工判断；
2. 数学规则和答案检查；
3. 代码测试；
4. 工具执行；
5. 已知错误模式；
6. 生成模型产生的候选；
7. LLM 预标注后的人类复核。

来源不同，标签可信度和偏差不同。程序生成的标签可规模化，但只能覆盖程序表达的约束；人工标签能处理开放内容，却存在分歧。

### 7.2 正例和负例

高质量数据不只是收集正确答案和明显错误答案，还要包含：

- 格式不同但语义正确；
- 过程错误但最终碰巧正确；
- 过程合理但最后计算错误；
- 少一个关键条件；
- 引用了不存在的证据；
- 语气自信但事实错误；
- 长度和风格与正例相反的 hard negative。

### 7.3 Hard negative 为什么重要

如果负例太简单，verifier 可能只学会识别乱码、短答案或明显格式错误。真实生成候选通常更难：错误答案会有完整步骤、合理术语和正确的局部计算。

训练和评估都应使用接近生成器当前能力的候选分布。静态人工负例可能很快过时。

### 7.4 生成器和验证器的分布偏移

训练时的负例来自旧生成器，部署时的候选来自新模型、不同温度或不同工具。候选分布变化会使 verifier 失效。

可以按时间和生成器版本拆分验证集，报告：

- 旧候选上的分数；
- 新候选上的分数；
- 高温度候选上的分数；
- 对抗或 hard negative 上的分数；
- 下游 top-1 选择准确率。

## 8. Reranking、Best-of-N 与 Search

### 8.1 Reranking

最常见的用法是先生成 K 个候选，再按 verifier 分数排序：

~~~math
j^*=\arg\max_{j\in\{1,\ldots,K\}}s_{ij},
\qquad
\hat y_i=\hat y_{ij^*}
~~~

这使系统有机会从少数正确候选中恢复答案，但前提是 verifier 能识别它。

### 8.2 Best-of-N 的真正含义

Best-of-N 不是“生成 N 个就一定更强”，而是：

1. 候选集合中存在好答案；
2. 评分器能够把好答案排到前面；
3. 选择和执行成本可接受。

缺少任何一项，N 增加都可能只增加成本。

### 8.3 过程评分用于搜索

在搜索中，验证器可以给中间节点分数：

~~~math
s_{t+1}=f_{\phi}(x,z_{1:t+1})
~~~

系统保留高分节点，剪掉低分节点。过程评分比最终评分更早提供反馈，但错误的早期评分可能把正确分支剪掉。

### 8.4 缓存与重复计算

多个候选可能共享同一前缀。系统可以缓存前缀状态，减少重复计算；但验证器若看到的上下文和候选边界不同，缓存键必须包含模型版本、prompt、工具状态和验证协议。

### 8.5 选择与执行分离

对于代码和工具任务，选择一个高分候选后还应执行它并再次检查。高分候选不是执行成功的替代品。

## 9. Verifier 的评估

### 9.1 Pairwise accuracy

给定正确候选 a 和错误候选 b，pairwise accuracy 为：

~~~math
A_{\mathrm{pair}}
=
\frac{1}{|\mathcal{P}|}
\sum_{(a,b)\in\mathcal{P}}
\mathbb{1}[s_a>s_b]
~~~

这里要求比较对集合 \(\mathcal{P}\) 非空，且每个 pair 的正确候选和错误候选都可判定。公式中的严格大于意味着分数相同时记为失败；也可以另报 tie rate，但不能把平票悄悄算作胜利。它适合检查排序倾向，但不能直接代表 top-1 选择。

### 9.2 Rerank top-1 accuracy

对每道题从真实候选集合选择最高分候选：

~~~math
A_{\mathrm{rerank}}
=
\frac{1}{N}\sum_{i=1}^{N}
\mathbb{1}\left[\hat y_{ij_i^*}=y_i^*\right]
~~~

这里要求评估题数 \(N>0\)，每道题都有非空候选集合，并且最高分并列时有固定的可复现处理。若某题没有可交付候选，应单独统计 abstain，而不是把它自动算成错误或正确。这是更贴近下游的指标。候选集合必须来自实际 generator，否则会出现评估分布偏移。

### 9.3 Hard negative accuracy

对每个 hard negative，检查它的分数是否低于至少一个正确候选：

~~~math
A_{\mathrm{hard}}
=
\frac{1}{H}\sum_{h=1}^{H}
\mathbb{1}[s_{\mathrm{good}(h)}>s_{\mathrm{hard}(h)}]
~~~

这里要求 hard negative 数量 \(H>0\)，并且每个 hard negative 都有至少一个明确的 good 对照；若没有 hard negative，这个指标未定义。它能发现 verifier 是否被“完整但错误”的候选欺骗。

### 9.4 Calibration 与 ECE

若分数被解释为成功概率，可按分桶计算 ECE：

~~~math
\mathrm{ECE}
=
\sum_{b=1}^{B}
\frac{|S_b|}{n}
\left|\mathrm{acc}(S_b)-\mathrm{conf}(S_b)\right|
~~~

S_b 是第 b 个分数桶，acc 是真实成功比例，conf 是平均预测置信度。计算 ECE 要求总样本数 \(n>0\)，并且被解释为概率的分数位于 \([0,1]\)。空桶不应贡献误差，但必须记录桶覆盖；如果所有样本都被过滤，ECE 未定义。ECE 依赖分桶方式，不能单独作为完整校准证明。

### 9.5 下游指标优先

最终要问：

1. 使用 verifier 后 top-1 是否提高；
2. 是否救回了少数正确候选；
3. 是否引入新的回归；
4. 每个正确结果的成本是多少；
5. 高风险切片是否更安全；
6. 是否需要允许不确定或人工复核。

验证集分类准确率高，但下游选择没有提升，说明训练目标和实际使用方式不匹配。

## 10. Reward hacking 与评分器偏差

### 10.1 长度偏差

若评分器把更多解释误当成更高质量，生成器会学习输出冗长文本。需要控制长度、比较等长度候选，并报告质量随 token 的曲线。

### 10.2 格式偏差

评分器可能偏好标题、列表、特定标记或英文术语。格式稳定有利于解析，但不能代替任务正确性。

### 10.3 自信语气偏差

“显然”“一定”“经过验证”等措辞可能提高语言评分，却没有增加证据。hard negative 应故意使用自信语气测试评分器。

### 10.4 训练集模式偏差

如果正例都使用同一种模板，verifier 可能把模板当成正确性；如果负例都很短，它会把长度当成质量。训练数据需要交叉风格和反事实控制。

### 10.5 目标错配

把 reward model 的分数直接优化，可能提升代理目标而损害真实任务：

~~~math
\Delta R_{\mathrm{proxy}}>0
\quad\not\Rightarrow\quad
\Delta Q_{\mathrm{task}}>0
~~~

R_proxy 是代理分数，Q_task 是真实任务质量。两者是否同步，需要独立验证集和业务指标确认。

### 10.6 防护方法

可组合使用：

1. hard negative；
2. 程序和工具验证；
3. 盲测与隐藏切片；
4. 长度和格式消融；
5. 多个独立评分器；
6. 人工分歧抽样；
7. 允许 abstain；
8. 定期刷新候选分布。

## 11. 不确定性、拒绝与成本

### 11.1 Verifier 不一定要强行选一个

当最高分候选只略高于第二名，或所有候选分数都低时，系统可以请求更多证据、增加候选、调用工具或返回不确定。

可以定义分数间隔：

~~~math
\Delta_{\mathrm{score}}=s_{(1)}-s_{(2)}
~~~

s_(1) 和 s_(2) 是最高与次高分数，因此至少需要两个可比较候选；少于两个候选时分数间隔未定义。间隔小不一定表示错误，但表示选择依据弱，需要结合校准和风险等级。

### 11.2 风险感知的阈值

低风险问答可以接受较小间隔；付款、删除、权限和医疗建议需要更高证据阈值。阈值应按任务风险切片校准，而不是全局使用一个分数。

### 11.3 成本账本

验证系统的成本可以拆成：

~~~math
C_{\mathrm{total}}
=
C_{\mathrm{generate}}
+C_{\mathrm{verify}}
+C_{\mathrm{tool}}
+C_{\mathrm{retry}}
+C_{\mathrm{review}}
~~~

只看 generator token 会低估 verifier、工具和人工复核成本。

## 12. 一个综合案例：合同金额与代码候选

### 12.1 合同金额

系统从合同中抽取基础金额、折扣和税率，生成两个计算候选：

~~~text
候选 A：100000 × 0.95 × 1.06 = 100700 元，引用正文第 3、4、7 页。
候选 B：100000 × 0.95 × 1.60 = 152000 元，税率来自 OCR 结果，没有引用原文。
~~~

一个只看解释长度的 reward model 可能给 B 高分，因为它写了更多步骤；程序计算和原文检查应把 B 判为不可信。

### 12.2 代码候选

代码任务中，RM 可以初步排序，程序测试决定最终可执行性。若 RM 选出的候选测试失败，系统应让其他候选继续接受测试，而不是直接交付。

### 12.3 失败归因

案例中至少有：

1. 检索错误；
2. OCR 错误；
3. 条件判断错误；
4. 计算错误；
5. 引用错配；
6. verifier 偏差；
7. 权限或执行环境错误。

每一种失败都需要对应的数据和指标。把所有问题归因给 reward model 没有帮助。

## 13. 最小可运行实验：RM 重排与程序验证

下面的 demo 用五道 toy case 比较：

1. greedy：直接取每道题的第一个候选；
2. rm_rerank：只按 reward model 分数选择；
3. hybrid_verifier：优先程序验证，再看步骤分数和 RM 分数；
4. pairwise accuracy、hard negative accuracy、ECE 和下游选择；
5. 一个 hard negative 被 RM 选错、但混合验证器救回的案例。

代码中的分数是人为构造的代理分数，不代表真实 reward model 的概率。为避免把 demo 的正常路径误读成通用保证，下面先校验 case、候选、步骤、分数和 token 的契约；没有比较对、hard negative 或成功候选的指标返回 None，而不是伪造一个 0。

~~~python
import math


cases = [
    {
        "id": "water_tank",
        "gold": "10",
        "candidates": [
            {
                "answer": "12",
                "rm_score": 0.62,
                "steps": [True, False],
                "program_pass": False,
                "hard_negative": True,
                "tokens": 46,
            },
            {
                "answer": "10",
                "rm_score": 0.84,
                "steps": [True, True, True],
                "program_pass": True,
                "hard_negative": False,
                "tokens": 55,
            },
            {
                "answer": "8",
                "rm_score": 0.31,
                "steps": [False],
                "program_pass": False,
                "hard_negative": False,
                "tokens": 40,
            },
        ],
    },
    {
        "id": "distractor_math",
        "gold": "12",
        "candidates": [
            {
                "answer": "111",
                "rm_score": 0.91,
                "steps": [True, False, False],
                "program_pass": False,
                "hard_negative": True,
                "tokens": 62,
            },
            {
                "answer": "12",
                "rm_score": 0.78,
                "steps": [True, True, True],
                "program_pass": True,
                "hard_negative": False,
                "tokens": 58,
            },
            {
                "answer": "99",
                "rm_score": 0.28,
                "steps": [False],
                "program_pass": False,
                "hard_negative": False,
                "tokens": 43,
            },
        ],
    },
    {
        "id": "code_loop",
        "gold": "pass",
        "candidates": [
            {
                "answer": "fail",
                "rm_score": 0.40,
                "steps": [True, False],
                "program_pass": False,
                "hard_negative": False,
                "tokens": 50,
            },
            {
                "answer": "pass",
                "rm_score": 0.73,
                "steps": [True, True],
                "program_pass": True,
                "hard_negative": False,
                "tokens": 65,
            },
            {
                "answer": "pass",
                "rm_score": 0.69,
                "steps": [True, True, True],
                "program_pass": True,
                "hard_negative": False,
                "tokens": 70,
            },
        ],
    },
    {
        "id": "logic_grid",
        "gold": "blue",
        "candidates": [
            {
                "answer": "red",
                "rm_score": 0.66,
                "steps": [True, False],
                "program_pass": False,
                "hard_negative": True,
                "tokens": 45,
            },
            {
                "answer": "blue",
                "rm_score": 0.71,
                "steps": [True, True],
                "program_pass": True,
                "hard_negative": False,
                "tokens": 50,
            },
            {
                "answer": "green",
                "rm_score": 0.38,
                "steps": [False],
                "program_pass": False,
                "hard_negative": False,
                "tokens": 39,
            },
        ],
    },
    {
        "id": "capital_lookup",
        "gold": "tokyo",
        "candidates": [
            {
                "answer": "tokyo",
                "rm_score": 0.80,
                "steps": [True],
                "program_pass": True,
                "hard_negative": False,
                "tokens": 18,
            },
            {
                "answer": "kyoto",
                "rm_score": 0.58,
                "steps": [False],
                "program_pass": False,
                "hard_negative": True,
                "tokens": 32,
            },
            {
                "answer": "tokyo",
                "rm_score": 0.79,
                "steps": [True],
                "program_pass": True,
                "hard_negative": False,
                "tokens": 20,
            },
        ],
    },
]


def validate_cases(items):
    if not isinstance(items, list) or not items:
        raise ValueError("cases must be a non-empty list")
    seen_ids = set()
    for case in items:
        required_case = {"id", "gold", "candidates"}
        if not isinstance(case, dict) or set(case) != required_case:
            raise ValueError("case schema is invalid")
        if (
            not isinstance(case["id"], str)
            or not case["id"]
            or case["id"] in seen_ids
        ):
            raise ValueError("case ids must be unique non-empty strings")
        seen_ids.add(case["id"])
        if not isinstance(case["gold"], str) or not case["gold"]:
            raise ValueError("gold must be a non-empty string")
        candidates = case["candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("each case needs non-empty candidates")
        for candidate in candidates:
            required_candidate = {
                "answer",
                "rm_score",
                "steps",
                "program_pass",
                "hard_negative",
                "tokens",
            }
            if (
                not isinstance(candidate, dict)
                or set(candidate) != required_candidate
            ):
                raise ValueError("candidate schema is invalid")
            if (
                not isinstance(candidate["answer"], str)
                or not candidate["answer"]
            ):
                raise ValueError("candidate answer must be non-empty")
            score = candidate["rm_score"]
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError("rm_score must be finite and in [0, 1]")
            steps = candidate["steps"]
            if (
                not isinstance(steps, list)
                or not steps
                or any(not isinstance(step, bool) for step in steps)
            ):
                raise ValueError("steps must be a non-empty boolean list")
            if not isinstance(candidate["program_pass"], bool):
                raise TypeError("program_pass must be boolean")
            if not isinstance(candidate["hard_negative"], bool):
                raise TypeError("hard_negative must be boolean")
            tokens = candidate["tokens"]
            if (
                isinstance(tokens, bool)
                or not isinstance(tokens, int)
                or tokens < 0
            ):
                raise ValueError("tokens must be a non-negative integer")


def is_correct(case, candidate):
    return candidate["answer"] == case["gold"]


def rm_select(case):
    if not case["candidates"]:
        raise ValueError("cannot select from empty candidates")
    return max(
        enumerate(case["candidates"]),
        key=lambda item: (item[1]["rm_score"], -item[0]),
    )[1]


def hybrid_select(case):
    if not case["candidates"]:
        raise ValueError("cannot select from empty candidates")
    return max(
        case["candidates"],
        key=lambda candidate: (
            candidate["program_pass"],
            sum(candidate["steps"]) / len(candidate["steps"]),
            candidate["rm_score"],
        ),
    )


def accuracy(selector):
    if not cases:
        raise ValueError("cases must be non-empty")
    return sum(is_correct(case, selector(case)) for case in cases) / len(cases)


def pairwise_accuracy():
    wins = 0
    total = 0
    for case in cases:
        correct = [
            candidate
            for candidate in case["candidates"]
            if is_correct(case, candidate)
        ]
        wrong = [
            candidate
            for candidate in case["candidates"]
            if not is_correct(case, candidate)
        ]
        for good in correct:
            for bad in wrong:
                total += 1
                wins += good["rm_score"] > bad["rm_score"]
    return wins / total if total else None


def hard_negative_accuracy():
    outcomes = []
    for case in cases:
        correct_candidates = [
            candidate
            for candidate in case["candidates"]
            if is_correct(case, candidate)
        ]
        hard_negatives = [
            candidate
            for candidate in case["candidates"]
            if candidate["hard_negative"]
        ]
        if hard_negatives and not correct_candidates:
            raise ValueError(
                "a hard negative requires at least one correct candidate"
            )
        best_correct = max(
            candidate["rm_score"] for candidate in correct_candidates
        )
        for candidate in hard_negatives:
            outcomes.append(candidate["rm_score"] < best_correct)
    return sum(outcomes) / len(outcomes) if outcomes else None


def calibration_ece():
    bins = [(0.0, 0.5), (0.5, 0.75), (0.75, 1.01)]
    total = sum(len(case["candidates"]) for case in cases)
    if total == 0:
        return None, []
    ece = 0.0
    details = []
    for lo, hi in bins:
        bucket = [
            (candidate["rm_score"], is_correct(case, candidate))
            for case in cases
            for candidate in case["candidates"]
            if lo <= candidate["rm_score"] < hi
        ]
        if not bucket:
            continue
        confidence = sum(score for score, _ in bucket) / len(bucket)
        accuracy_value = sum(ok for _, ok in bucket) / len(bucket)
        gap = abs(confidence - accuracy_value)
        ece += len(bucket) / total * gap
        details.append(
            (f"[{lo},{hi})", round(confidence, 3), round(accuracy_value, 3), len(bucket))
        )
    return round(ece, 3), details


validate_cases(cases)
all_steps = [
    ok
    for case in cases
    for candidate in case["candidates"]
    for ok in candidate["steps"]
]
rm_failures = [
    case["id"]
    for case in cases
    if not is_correct(case, rm_select(case))
]
hybrid_rescues = [
    case["id"]
    for case in cases
    if not is_correct(case, rm_select(case))
    and is_correct(case, hybrid_select(case))
]
ece, ece_bins = calibration_ece()
total_tokens = sum(
    candidate["tokens"]
    for case in cases
    for candidate in case["candidates"]
)
hybrid_correct_count = sum(
    is_correct(case, hybrid_select(case))
    for case in cases
)

report = {
    "greedy_accuracy": round(
        sum(is_correct(case, case["candidates"][0]) for case in cases)
        / len(cases),
        3,
    ),
    "rm_rerank_accuracy": round(accuracy(rm_select), 3),
    "hybrid_verifier_accuracy": round(accuracy(hybrid_select), 3),
    "pairwise_accuracy": round(pairwise_accuracy(), 3),
    "hard_negative_accuracy": round(hard_negative_accuracy(), 3),
    "process_step_accuracy": round(sum(all_steps) / len(all_steps), 3),
    "rm_ece": ece,
    "ece_bins": ece_bins,
    "rm_failures": rm_failures,
    "hybrid_rescues": hybrid_rescues,
    "total_tokens": total_tokens,
    "cost_per_hybrid_correct": (
        round(total_tokens / hybrid_correct_count, 3)
        if hybrid_correct_count
        else None
    ),
}

assert report["greedy_accuracy"] == 0.2
assert report["rm_rerank_accuracy"] == 0.8
assert report["hybrid_verifier_accuracy"] == 1.0
assert report["pairwise_accuracy"] == 0.9
assert report["hard_negative_accuracy"] == 0.75
assert report["rm_ece"] == 0.165
assert report["total_tokens"] == 693

for key, value in report.items():
    print(f"{key}={value}")
~~~

预期输出：

~~~text
greedy_accuracy=0.2
rm_rerank_accuracy=0.8
hybrid_verifier_accuracy=1.0
pairwise_accuracy=0.9
hard_negative_accuracy=0.75
process_step_accuracy=0.679
rm_ece=0.165
ece_bins=[('[0.0,0.5)', 0.343, 0.0, 4), ('[0.5,0.75)', 0.665, 0.5, 6), ('[0.75,1.01)', 0.824, 0.8, 5)]
rm_failures=['distractor_math']
hybrid_rescues=['distractor_math']
total_tokens=693
cost_per_hybrid_correct=138.6
~~~

这个实验应这样解读：

1. RM 重排在这组构造样本上高于 greedy，但它被 distractor_math 的 hard negative 误导。
2. 混合验证器使用程序通过结果和过程分数，救回了 RM 选错的样本。
3. pairwise accuracy 和 hard negative accuracy 描述 verifier 的局部能力，不等于下游选择准确率。
4. ECE 为 0.165，说明教学构造中的分数并非完美概率；真实系统需要更多校准数据。
5. 总 token 693 是候选生成账本，不包含实际 GPU、程序执行、缓存和人工复核成本。

## 14. 怎样设计 verifier 实验

### 14.1 先固定候选生成器

验证器比较必须使用同一候选集合，或明确比较候选分布。否则 generator 变强会被误归因给 verifier。

### 14.2 同时看局部和下游

局部指标包括 pairwise、hard negative、ECE 和步骤准确率；下游指标包括 top-1 选择、最终任务成功率、单位成功成本和拒答质量。

### 14.3 让 hard negative 接近真实错误

从真实 generator 采样错误候选，按错误类型分层：单位、否定、引用、边界、格式、工具参数和安全约束。静态容易负例只能作为起点。

### 14.4 做长度和格式消融

将正确与错误候选改写成相近长度和相同格式，再测 verifier。若分数差异显著下降，说明原模型依赖了表面线索。

### 14.5 观察 verifier 版本漂移

生成器升级、提示变化、温度变化和工具接入都会改变候选分布。验证器需要在新候选上回归，不应只依赖旧验证集。

## 15. 安全与权限

### 15.1 高分不是执行许可

模型或 verifier 认为某个工具动作合理，不代表它拥有执行权限。权限系统应独立于语言评分器。

### 15.2 程序 verifier 的沙箱

运行候选代码需要隔离文件、网络、进程和资源。测试器应有超时、输出上限和可撤销环境。

### 15.3 高风险候选的多重证据

付款、删除、权限变更和外部消息应要求结构化参数、独立规则、权限检查和必要人工确认。不能用一个 reward 分数代替多重证据。

### 15.4 不确定状态

当 verifier 分数低、候选冲突或工具结果异常时，返回不确定或请求复核比强行选一个答案更可靠。拒绝交付也是系统行为的一部分，应单独评估拒答是否恰当。

## 16. 常见误区

### 误区一：Verifier 分数就是事实概率

除非经过校准和任务验证，否则分数只是排序信号或代理目标。

### 误区二：Pairwise accuracy 高就一定能选对

pairwise 只测成对比较，真实列表中的最高分选择可能受到候选数量、相关性和分数尺度影响。

### 误区三：Process verifier 能保证整条链正确

局部步骤正确不保证全局约束满足，步骤切分和标签也可能有偏差。

### 误区四：程序验证总是可靠

测试集覆盖、执行环境、断言和资源限制都可能有问题。程序验证需要测试和安全审计。

### 误区五：Reward model 越通用越好

更宽的目标会混合正确性、风格、帮助性和安全，可能导致目标冲突。应拆分分数并明确使用场景。

### 误区六：只要加入 hard negative 就不会 reward hacking

hard negative 只能覆盖已知攻击面。生成器和评分器持续博弈，数据、规则和独立评估需要持续更新。

## 17. 小练习

### 练习一：定义三种 verifier

为数学答案、代码函数和合同引用分别写出 outcome verifier 的输入、输出、边界和失败状态。

### 练习二：比较 pointwise 与 pairwise

构造一个包含两个正确表达和三个错误表达的候选集合，说明 pointwise 标签和 pairwise 偏好各自如何组织。

### 练习三：设计 hard negative

为一个算术题写出格式整齐但算错、过程正确但结论格式错、引用存在但不支持结论的三个 hard negative。

### 练习四：校准分析

将 verifier 分数分成三个区间，计算每个区间的真实准确率和 ECE。说明为什么排序正确不代表概率校准。

### 练习五：候选分布偏移

用两个不同 temperature 生成候选，比较旧 verifier 在低温和高温候选上的 top-1 选择准确率。

### 练习六：程序验证安全

列出运行模型生成代码时需要限制的五类资源，并设计超时、异常和人工复核策略。

## 18. 本章总结

Verifier 和 reward model 把“提出候选”和“判断候选”分开，是 reasoning 系统的重要组成部分，但它们本身也需要被验证。

本章的关键关系是：

1. outcome verifier 检查最终结果，process verifier 检查中间步骤，programmatic verifier 执行明确约束。
2. reward model 学习代理目标，可以作为 learned verifier，但不等于任务真值。
3. pointwise、pairwise 和 listwise 分别对应单候选判断、两候选比较和列表排序。
4. hard negative 和真实 generator 候选分布决定了验证器评估是否有意义。
5. pairwise、top-1、校准和下游成功率是不同指标，不能互相替代。
6. 长度、格式、自信语气和训练集模式都可能造成 verifier 偏差。
7. 程序验证更适合可执行约束，但仍受测试覆盖和沙箱安全限制。
8. 高分候选不是执行许可；高风险动作需要权限、规则、工具和必要人工确认。
9. verifier 应允许不确定状态，而不是在证据不足时强行选择。

下一章进入 process supervision，继续讨论如何给中间步骤标注、如何处理多种正确过程，以及 PRM 如何参与训练和搜索。

## 19. 资料索引

以下链接优先保留原始论文和评估资料。复现实验时记录候选生成器、候选数量、负例来源、评分器版本、阈值和下游任务。

1. Training Verifiers to Solve Math Word Problems：<https://arxiv.org/abs/2110.14168>
2. Let's Verify Step by Step：<https://arxiv.org/abs/2305.20050>
3. InstructGPT：<https://arxiv.org/abs/2203.02155>
4. Learning to Summarize from Human Feedback：<https://arxiv.org/abs/2009.01325>
5. RewardBench：<https://arxiv.org/abs/2403.13787>
6. HumanEval：<https://arxiv.org/abs/2107.03374>
7. Self-Consistency：<https://arxiv.org/abs/2203.11171>

本章写作时已联网访问上述论文入口。正文区分数学验证器、偏好 reward model、代码测试器和教学构造；没有把某个评分器的公开结果外推为所有 reasoning 系统的内部事实。
