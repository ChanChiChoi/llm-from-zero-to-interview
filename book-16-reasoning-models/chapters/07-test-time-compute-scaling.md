# 第七章：推理时计算扩展：质量曲线、预算分配与自适应路由

模型部署之后，能力并没有被固定在“一次前向计算”的结果上。系统可以让模型生成更长的推理，采样多个候选，调用验证器，执行工具，展开搜索，或者让多个 worker 独立完成子任务，再由程序和模型合并结果。这些额外计算发生在推理阶段，统称为 test-time compute，中文常译为推理时计算或测试时计算。

推理时计算扩展的关键问题不是“多花计算是否有用”。在许多可验证任务上，多花计算确实可能提高正确率；真正困难的问题是：预算应该花在哪里？哪些请求值得升级？如何分辨候选是独立的新证据，还是同一个错误的重复表达？当质量提高时，延迟、GPU 占用、工具调用和人工复核成本是否仍然可以接受？

本章把推理时计算当作一个资源分配问题来讲。初学者会先看到 direct、self-consistency、verifier、search 和 tool loop 的直观差别；专家则需要进一步区分候选覆盖、选择质量、计算相关性、曲线拟合、动态停止、p95 延迟和单位成功成本。章节最后用一个可运行的 toy 实验说明：自适应路由可以减少平均成本，但仍可能把高预算浪费在难以解决的对抗样本上。

## 0.1 一个请求为什么不该只有一个预算

考虑下面四类请求：

~~~text
请求 A：查询一个稳定的事实，答案很短，几乎没有歧义。
请求 B：解一道中等难度的代数题，有明确的数值答案。
请求 C：修复一个代码 bug，可以运行测试，但可能需要多次修改。
请求 D：审阅合同付款条款，金额错误的代价很高，证据还可能来自 OCR。
~~~

如果四类请求都使用同一个“最大思考”预算，会浪费 A 的资源；如果都只用一次生成，C 和 D 的错误风险可能过高。更合理的系统会先用低成本策略得到初步信号，再决定是否继续计算。

但“继续计算”也不是单一旋钮。B 可能需要多条数学路径和结果验证，C 可能需要代码执行和回归测试，D 可能需要检索、引用核对和人工复核。总 token 相同，放在不同环节，质量和延迟可能完全不同。

## 0.2 资料与证据边界

几项研究说明，推理时计算的收益取决于任务和分配方式。Snell 等在其数学任务与模型设置中发现，策略效果随题目难度变化，并报告按题分配计算的策略比 Best-of-N 基线有更高的预算效率；论文的“超过 4 倍”结果只适用于其比较设置，不能外推到任意模型或业务流量。[论文版本](https://arxiv.org/abs/2408.03314v1)

`Large Language Monkeys` 在代码、形式证明等任务上研究重复采样，报告候选覆盖随样本数增加；在缺少自动验证器的实验任务中，多数投票和 reward model 的选择效果会遇到瓶颈。这说明“池中至少有一个正确答案”和“系统能选对答案”必须分开测量。[论文 v3](https://arxiv.org/abs/2407.21787v3)

2026 年的分析工作按计算发生的位置区分单轨迹延长、完整候选生成后的终局聚合，以及对未完成前缀进行搜索；这些是描述推理结构的类别，实际系统可以组合使用。[Hariri 等，预印本 v2](https://arxiv.org/abs/2608.04001v2) 另一项预印本把逐题预算选择写成受平均计算约束的准确率优化，并在 MATH、GSM8K 和指定模型上评估 learned allocator；其结果是有限题集上的实验报告，不证明同一路由器可直接迁移到代码、合同或线上延迟目标。[Zhai 等，预印本 v1](https://arxiv.org/abs/2604.14853v1)

本章还会讨论产品接口中常见的 effort、thinking level 或类似控制面。字段名称只能证明产品暴露了某种控制方式，不能证明不同厂商的档位对应相同 token、相同搜索深度或相同内部算法。除非官方文档明确说明，正文不把产品词翻译成确定的内部结构。

本章的曲线数字、成本权重和 Python 数据均为教学构造。论文结果只在论文任务、模型、数据和预算口径内成立；本章 demo 只说明 toy 数据下的指标关系，不代表真实模型性能。

## 0.3 初学者视角：多花计算到底在做什么

推理时计算通常通过以下几种方式增加工作量：

- 生成一条更长的中间过程；
- 生成多条候选，再投票或重排；
- 每个步骤调用 verifier；
- 展开搜索树并剪枝；
- 运行计算器、测试器、浏览器或数据库；
- 让多个 worker 独立探索，再合并结果；
- 初次结果不确定时继续追问、反思或修正。

这些方法的共同点是，模型参数不变，但每个请求消耗的推理资源不同。它们也有共同的限制：额外计算只能从候选或工具反馈中寻找正确性，不能凭空创造训练数据中完全没有的能力；评分器错误时，更多计算可能更快地选择错误答案。

## 1. 推理时计算的形式化表示

### 1.1 预算不是一个数字

对第 `i` 个请求，把推理策略预算表示成向量：

~~~math
b_i=(K_i,L_i,D_i,V_i,U_i,H_i)
~~~

其中，下面的符号表示请求允许使用的预算上限，通常取非负整数：

- `K_i` 是候选数量或采样次数；
- `L_i` 是单条轨迹允许的最大长度；
- `D_i` 是搜索深度；
- `V_i` 是 verifier 调用上限；
- `U_i` 是工具调用上限；
- `H_i` 是反思、修正或控制循环的轮数。

把预算写成向量很重要。增加 `K` 和增加 `D` 都会让系统更忙，但一个主要增加横向候选，一个主要增加纵向搜索；增加 `V` 可能提高选择质量，也可能让一个有偏的评分器影响更多决策。

例如，合同核对请求可先设置 `b_i=(4,512,2,2,1,1)`：最多生成 4 条、每条最多 512 token，搜索深度不超过 2，调用 verifier 不超过 2 次、工具不超过 1 次、修正循环不超过 1 轮。它们是不同资源的上限，实际策略可能提前停止；预算口径和成本账本的细化见[第十三章](13-reasoning-effort与推理预算.md)。

### 1.2 成本函数

设 `T_i` 是请求实际消耗的生成和评分 token，`n_{\mathrm{ver},i}`、`n_{\mathrm{tool},i}` 分别是实际 verifier 和工具调用数量，`R_i` 是用户可感知的服务时间，可以写一个抽象成本：

~~~math
C_i(m)
=
c_{\mathrm{tok}}T_i
+c_{\mathrm{ver}}n_{\mathrm{ver},i}
+c_{\mathrm{tool}}n_{\mathrm{tool},i}
+c_{\mathrm{lat}}R_i
~~~

`m` 是推理策略，`c_tok`、`c_ver`、`c_tool` 和 `c_lat` 是有限的非负业务权重；`T_i`、`n_{\mathrm{ver},i}`、`n_{\mathrm{tool},i}` 和 `R_i` 也必须是有限且非负的实际消耗。批处理评测可能更重视总计算量，在线交互可能更重视延迟，高风险审阅则可能把人工复核和错误代价加入成本。预算上限与实际消耗要分开记录：没有达到上限，不代表系统一定使用了相同的计算量。

手算一笔教学成本：设 `T_i=500` token、`n_{\mathrm{ver},i}=2` 次 verifier 调用、`n_{\mathrm{tool},i}=1` 次工具调用、`R_i=0.4` 秒，并假设 `c_tok=0.002`、`c_ver=1`、`c_tool=3`、`c_lat=5`。则 `C_i=0.002×500+1×2+3×1+5×0.4=8` 个归一化成本单位。这里的权重是示例中的换算假设，不是市场价格；实际系统若没有可靠的共同换算率，就应分别报告 token、调用数和秒数，不能直接相加。

### 1.3 质量函数

令 `B` 表示一种有限的非负预算口径，例如每请求 token、总模型调用或金钱成本。当评估任务数 `N>0` 时，固定策略在预算 `B` 下的任务准确率可以写成。比较不同预算时要固定模型版本、提示、评估样本和选择规则；若资源仍是多维向量，就选定一个预算轴并固定其他轴，或先说明有依据的标量换算，不能把 token、调用数和秒数直接拼成 `B`：

~~~math
A(B)
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[\hat y_i(B)=y_i^\star]
~~~

当 `N=0` 时，`A(B)` 没有定义，不能把空评估集报告为 0。`A(B)` 是平均质量，不代表每个样本随预算单调变好。某个样本可能在低预算时碰巧答对，高预算时被错误 verifier 选走；因此还需要看按样本、按任务切片的变化。

### 1.4 边际收益

当 `B_2>B_1` 且对应成本满足 `C(B_2)>C(B_1)` 时，从预算 `B_1` 增加到 `B_2` 的质量增益除以成本增量可以写成：

~~~math
G(B_1,B_2)
=
\frac{A(B_2)-A(B_1)}{C(B_2)-C(B_1)}
~~~

这里 `C(B)` 表示固定评估集上采用预算 `B` 时的总成本，即对各请求的 `C_i` 求和；各成本必须使用同一口径。若两种预算的成本相同，`G` 没有定义；不能用 0 代替。`G` 不是能力的绝对度量，而是一个资源决策指标。不同阶段的 `G` 可能差别很大：从一次生成增加到少量采样，收益可能明显；从大量采样再翻倍，收益可能接近零。

例如，固定 4 道题评估时，低预算答对 2 道，故 `A(B_1)=2/4=0.50`；高预算答对 3 道，故 `A(B_2)=3/4=0.75`。若两档总成本分别为 12 和 20 个同口径成本单位，则 `G=(0.75−0.50)/(20−12)=0.03125`，即每增加一个成本单位，样本平均准确率增加 0.03125。小样本的这个差值很不稳定，正式比较还要报告不确定性和任务切片。

## 2. 为什么增加推理时计算可能有效

### 2.1 覆盖偶然的正确路径

如果单次候选独立命中正确答案的概率为 `p_i\in[0,1]`，生成非负整数个 `K` 候选且至少有一个正确候选的理想化概率是：

~~~math
P_i(K)=1-(1-p_i)^K
~~~

当 `K=0` 时，按空事件概率为 0 处理；当 `K>0` 时，公式依赖候选相互独立且每个候选命中概率相同的假设。它只说明“候选覆盖”会随着 `K` 增加而提高。它没有说明最终答案会正确，因为系统还要从候选中选出正确的那个。

例如，若单条候选答对概率 `p_i=0.2`，生成 `K=3` 条且暂时假设独立，则至少一条正确的概率为 `1−(1−0.2)^3=1−0.8^3=1−0.512=0.488`。这个 48.8% 是候选覆盖概率，不是最终交付准确率：选择器仍可能挑中错误候选。重复采样在代码任务上的覆盖结果与无自动验证器时的选择瓶颈，见[Large Language Monkeys v3](https://arxiv.org/abs/2407.21787v3)；其结论依论文中的模型、任务和 oracle。

真实候选往往相关。若所有候选都共享同一个错误的题意理解，独立假设就不成立，实际提升会比公式慢得多。可以用有效独立样本数、错误类型覆盖和答案去重率补充描述。

### 2.2 把判断延后交给验证器

单次生成必须同时负责提出答案和判断答案；Best-of-N 把候选生成和候选选择分开。只要生成器已经能偶尔提出正确答案，可靠的 outcome verifier 或程序测试器就可能把它挑出来。

相反，如果 verifier 偏好长文本、自信语气或错误模式，增加 `N` 会增加遇到高分错误候选的机会。Best-of-N 的收益取决于“候选池里有好答案”和“评分器能把好答案排在前面”两个条件。

### 2.3 让工具提供外部约束

计算器、编译器、测试套件、数据库、检索文档和符号系统可以提供模型自身没有的反馈。工具调用增加了推理时计算，却也可能把模糊的语言判断变成可检查状态。

工具不是自动真理。测试覆盖、文档版本、OCR 错误、数据库快照和沙箱限制都会影响反馈含义。工具结果应作为带来源和版本的观察记录，而不是一句无上下文的“通过”。

### 2.4 给困难请求更多机会

如果请求难度差异很大，统一预算会造成浪费。自适应路由让简单请求走 direct，低置信度请求增加采样，高风险可验证请求调用工具，剩下的预算留给需要搜索或人工复核的任务。

这类路由的核心不是预测一个抽象的“智能程度”，而是预测：继续计算能否降低当前任务的不确定性，降低错误是否有足够业务价值，以及当前任务是否存在可靠的验证信号。

## 3. 几种推理时扩展路线

### 3.1 Long CoT

Long CoT 允许模型生成更长的中间轨迹。它可能提供更多计算空间、变量记录和自检机会，但长度本身不保证每一步有用。长轨迹还会增加上下文、缓存、输出成本和可见信息风险。

评估长 CoT 时，需要控制任务答案、步骤质量和 token 数。若长轨迹只增加重复解释，却没有提高独立的最终验证结果，说明增加的是文本，不是有效计算。

### 3.2 Self-Consistency

Self-Consistency 生成多条推理链，将最终答案归一化后投票或聚合。它适合答案形式明确、错误路径不会高度相关的数学和逻辑任务；对开放写作和主观偏好任务，投票本身可能没有明确含义。

Self-consistency 的核心成本是候选生成；它通常不在中间步骤动态剪枝。若候选可以并行，墙钟延迟可能低于同 token 数的串行搜索，但 GPU 资源和总 token 仍会上升。原始论文在其任务与模型上报告了多路径采样和终局聚合的收益，不能把该结果当作所有开放式任务都适合多数投票的证明。[Wang 等，ICLR 2023](https://arxiv.org/abs/2203.11171v4)

### 3.3 Best-of-N

Best-of-N 的流程是生成 `N` 个候选、评分、选择最高分。它比自一致性更依赖 verifier，因为最终不一定采用多数答案，而是采用评分器最喜欢的答案。

如果候选存在多个等价正确表达，评分器应先做答案等价类归一化或使用程序验证；否则同一答案的不同措辞可能被错误当成多个竞争候选。Best-of-N 的生成—评分—选择路线已有数学题任务研究，但收益依赖候选分布和 verifier；Cobbe 等在 GSM8K 上报告其训练 verifier 对候选排序的实验结果。[论文 v2](https://arxiv.org/abs/2110.14168v2)

### 3.4 Verifier reranking

Verifier reranking 让 verifier 在候选生成之后参与排序。outcome verifier 主要看终局，process verifier 还看中间步骤，programmatic verifier 执行代码、公式或 schema；监督粒度、verifier 与 reward model 的差别及常见校准指标见[第四章](04-verifier与reward-model.md)。

reranking 的评估不能只看 verifier 的 pairwise accuracy。真正的下游问题是：在真实生成器产生的候选集合上，最高分候选是否正确，是否增加了高分 hard negative，单位成功成本是多少。候选覆盖与 verifier 选择也是不同的实验对象；重复采样研究已显示，无自动 oracle 的部分任务中选择方法会遇到上限。[Brown 等，arXiv v3](https://arxiv.org/abs/2407.21787v3)

### 3.5 Tree search

Tree-of-Thought、beam、best-first 和 MCTS 将候选扩展提前到中间步骤。它们可以在发现明显错误时节省后续计算，也可以回溯和尝试不同策略；代价是状态表示、去重、剪枝、缓存和日志更复杂。Tree of Thoughts 的原论文在 Game of 24、创意写作和迷你填字等特定任务上研究中间状态搜索；算法结构和细节见[第六章](06-search与tree-of-thought.md)及[论文 v2](https://arxiv.org/abs/2305.10601v2)。

### 3.6 Reflection 和修正循环

反思循环先生成答案，再要求模型找错误并修正。它能在没有独立 verifier 时提供低成本的再检查，但模型可能只是重复原来的错误，或者把正确答案改坏。

反思必须有停止条件和独立信号，例如答案差异、工具测试、证据支持或外部校验。单纯增加“再检查一次”的轮数不能证明质量提高。

### 3.7 Tool loop

工具循环把生成、执行、观察和下一次决策串起来。代码修复可以生成补丁、运行测试、分析失败、再次修改；文档问答可以检索条款、核对引用、比较版本、重新计算。代码修复 demo 中，search 用 `940` 个生成/评分 token、`5` 次 verifier 和 `2` 次工具调用；按示例权重，成本为 `940×1+5×45+2×100=1365` 个成本单位。这个手算把复合策略拆成独立资源项，也展示为何不能只报 token 数。

工具循环的计算不是单纯 token。每一次执行都有环境状态、时间、权限和失败成本，必须在预算账本中单独记录。

### 3.8 Multi-agent 和程序化工作流

多个 worker 可以并行提出计划、检索资料、写代码或做独立核验；主控组件再合并结果。它可以看作推理时计算的一种系统化形态，但不应把“多个 Agent”自动等同于“多个独立证据”。如果所有 worker 使用同一错误前提，结果仍然相关。

程序化 workflow 也属于推理时计算的一部分。批量检索、表格处理、格式转换和测试执行交给确定性程序，模型处理策略选择、异常和解释，往往比让多个 Agent 自由调用工具更容易复现。

下表把策略的计算位置和主要限制放在同一尺度上比较。单轨迹、终局聚合和前缀搜索是分析预算发生位置的视角；同一系统可以先搜索中间状态，再聚合完整候选。[Hariri 等的分类](https://arxiv.org/abs/2608.04001v2)提供了术语依据。表中适用场景是机制层面的判断，不是跨论文效果排名。

| 路线 | 计算发生在哪里 | 反馈或选择 | 适合的问题 | 主要限制 |
|---|---|---|---|---|
| Long CoT / reflection | 延长单条轨迹，或在轨迹末尾追加批评与修正 | 模型自身续写或自评 | 可在线性步骤中记录变量、补充推导的任务 | 错误前提可能沿同一路径重复，文本变长不等于有效计算 |
| Self-Consistency / Best-of-N | 生成多条完整候选后再聚合 | 多数投票、答案归一化或 verifier 重排 | 有明确答案等价关系，且采样能覆盖不同路径的任务 | 候选相关时覆盖增长变慢；选择器会成为瓶颈 |
| Tree Search | 候选完成前扩展和比较部分状态 | 中间状态评分、环境反馈、剪枝或回溯 | 早期选择会影响后续结果，且状态能表示和比较的任务 | 状态设计和估值有偏会造成误剪，控制逻辑更复杂；搜索细节见[第六章](06-search与tree-of-thought.md) |
| Tool loop / 程序化 workflow | 模型调用之间插入执行、检索或确定性计算 | 测试、数据库、文档或工具返回的观察 | 有可重复的外部检查器或权威数据源的任务 | 结果受工具版本、覆盖、权限和串行延迟约束 |
| Multi-agent / 多 worker | 多个轨迹并行或分工处理子问题 | 主控合并、冲突消解和独立验证 | 子任务相对独立、产物可验证且可并行的任务 | 多 worker 不自动提供独立证据；协调和合并成本见[第十六章](16-multi-agent与ultra-test-time-compute.md) |

## 4. 候选覆盖和选择质量

### 4.1 pass@k 不是最终选择准确率

代码任务中常见的 `pass@k` 用候选覆盖能力描述“抽出的 k 个程序里至少有一个通过测试”的概率。HumanEval 论文给出基于每题 `n` 个样本和其中 `c` 个通过样本的无偏估计，公式中的 `C(n,k)` 表示从 `n` 项中取 `k` 项的组合数：[Chen 等，论文 v2](https://arxiv.org/abs/2107.03374v2)

~~~math
\widehat{\mathrm{pass@}k}
=1-\frac{\binom{n-c}{k}}{\binom{n}{k}},
\qquad 0<k\le n
~~~

例如 `n=5,c=2,k=2` 时，`pass@2=1−C(3,2)/C(5,2)=1−3/10=0.7`。它估计从这 5 个样本中取 2 个时命中通过解的比例，依赖该题的测试 oracle，并不表示一个实际选择器必定能交付通过的候选。在候选数 `K>0` 且每个候选的正确性和分数定义完整时，还需把候选覆盖和实际选择分开：

~~~math
A_{\mathrm{exist}}(K)
=
\Pr(\exists j\le K:\ y_j=y^\star)
~~~

~~~math
A_{\mathrm{select}}(K)
=
\Pr(y_{j^*}=y^\star),
\qquad
j^*=\arg\max_j s_j
~~~

分数并列时，评估协议必须固定一个平局处理规则。`A_exist` 和 `A_select` 是固定评估分布上的概率或其样本估计；单个请求的 0/1 例子只展示两种事件的差别。当 `K=0` 时，这两个指标都没有定义；空候选不能被当作 0% 的选择准确率。`A_exist` 高而 `A_select` 低，说明生成器能提出正确候选，但评分器或选择策略没有识别出来。以单个请求为例，若三个候选的答案为 `42、42、40`，正确性依次为真、真、假，verifier 分数为 `0.70、0.74、0.92`，则 `A_exist=1`，但最高分选择的是错误的 `40`，所以 `A_select=0`。这个小例子表示一题的状态；总体指标要在固定评估集上取平均。增加 `K` 可能进一步扩大两者的差距。

### 4.2 候选相关性

假设独立命中的公式需要独立性。实际系统应测量：

- 最终答案去重率；
- 方法或状态去重率；
- 错误类型之间的重叠；
- 多次采样的 pairwise 相似度；
- 是否共享同一错误前提。

在候选数 `K>0`、且 `normalize` 规则固定时，答案去重率可以写成：

~~~math
D_{\mathrm{answer}}
=
\frac{|\mathrm{uniq}(\mathrm{norm}(y_1,\ldots,y_K))|}{K}
~~~

`norm` 是固定的答案规范化规则，`uniq` 统计规范化后不重复的答案。若三个候选依次是 `42、40、40`，则 `D_answer=2/3≈0.667`。它只测表面答案多样性，不证明正确性；对自一致性中的候选聚合和采样关系，可回看[第三章](03-self-consistency与采样.md)，搜索路径的多样性见[第六章](06-search与tree-of-thought.md)。当 `K=0` 时，去重率没有定义。去重率高不代表候选正确，只说明答案表面更分散；去重率低也不一定坏，多个独立推导得到同一正确答案可能正是自一致性的目标。它必须和真实正确率一起解释。

### 4.3 hard negative

随着 `K` 增加，候选池中的错误也会增加。评分器需要面对完整、流畅、自信而错误的 hard negative，而不是只面对短乱码。否则高预算模式可能比低预算模式更容易被评分器欺骗。

hard negative 应按错误类型分层：题意误读、单位错误、引用错配、边界遗漏、格式合法但业务非法、程序通过公开测试却失败隐藏测试。每类错误都可能有不同的选择和安全后果。

## 5. 质量曲线与边际收益

### 5.1 画预算曲线

至少选择几个非负且可比较的预算点，例如 `B=1, 2, 4, 8, 16`，对每个点测量准确率、覆盖率、选择准确率、平均 token、p95 延迟和单位成功成本。每个点都要使用非空、固定的评估集；曲线比一个最高预算数字更能说明系统。

### 5.2 边际收益递减

推理时计算常见边际收益递减：候选开始重复，评分器成为瓶颈，正确路径已经被发现，或任务缺少可验证信号。一个教学性的饱和曲线可以写成：

~~~math
A(B)=A_{\infty}-cB^{-\alpha},
\qquad
c>0,\ \alpha>0
~~~

`A_{\infty}` 是理想化的饱和值，`c` 和 `\alpha` 是拟合参数。若把 `A` 当准确率，通常还应检查 `0\le A_{\infty}\le 1`；拟合时需要正预算 `B>0` 和足够的观测点。例如，取教学参数 $A_{\infty}=0.9,c=0.2,\alpha=0.5$，在 `B=1` 时 `A(1)=0.9−0.2×1^{-0.5}=0.7`；在 `B=4` 时 `A(4)=0.9−0.2×4^{-0.5}=0.9−0.1=0.8`。这只展示该函数形状：预算增加时质量靠近假设的上限，不能当作实测曲线或普适 scaling law。拟合真实曲线还要报告数据、误差和模型选择。

### 5.3 质量提高但业务价值下降

准确率增加可能集中在低价值样本，而高价值 hard slice 没有改善。也可能平均质量提高，但 p95 延迟和人工复核成本超过业务承受范围。

当权重 `w_i` 有限、非负且至少有一个正权重时，可以报告加权质量：

~~~math
Q_{\mathrm{weighted}}
=
\frac{\sum_i w_i q_i}{\sum_i w_i}
~~~

当所有权重都为 0 时，分母为 0，加权质量没有定义；不能把它报告为 0。`q_i` 是第 `i` 个请求的质量值，需使用同一套定义和尺度（例如准确/错误分别为 1/0）；`w_i` 是有限、非负的任务权重，可以表示任务价值或风险权重，但不能借此隐藏普通样本回归。普通准确率、价值加权准确率和关键切片结果应同时报告。例如两个任务的 `q=(0.8,0.5)`、权重 `w=(1,3)` 时，`Q_weighted=(1×0.8+3×0.5)/(1+3)=2.3/4=0.575`。第二项权重大，所以它的较低质量拉低了总分；仍应同时公布两个单项结果，避免加权平均遮住回归。

## 6. 预算如何分配

### 6.1 固定预算

固定预算最容易实现：所有请求使用同一候选数、最大长度或搜索深度。它适合离线基线和可控实验，却会在请求难度差异较大时浪费资源。

### 6.2 按请求分配

自适应路由可以把请求 `i` 的策略写成：

~~~math
m_i=\pi(x_i,\hat d_i,v_i,e_i,r_i)
~~~

其中 $\hat d_i$ 是难度估计，`v_i` 是业务价值，`e_i` 表示是否有可用且适合该任务的外部验证信号，`r_i` 是风险。用 `e_i` 避免和前文预算向量中的 `U_i`（工具调用上限）混淆。策略 `m_i` 可以是 direct、self-consistency、verifier、search、tool loop 或人工复核。`π` 是待评估的路由策略，不是已训练或已经验证的模型。

例如，一个 code patch 请求若 $\hat d_i=0.90,v_i=0.95,e_i=1,r_i=0.70$，可被教学规则送往 search；一个稳定事实查询若 $\hat d_i=0.15,v_i=0.20,e_i=0,r_i=0.10$，可走 direct。这两个数值只演示特征如何影响路由，不能当成通用阈值。路由器也会犯错：它把简单题升级会浪费计算，把难题降级会增加错误。因此需要评估路由混淆矩阵、升级收益和降级损失。已有逐题分配方案在数学数据集和指定模型上与固定/启发式预算作过匹配预算比较，但跨任务迁移仍需重测。[Zhai 等，预印本 v1](https://arxiv.org/abs/2604.14853v1) 按观测反馈调节预算的控制环留给[第十八章](18-adaptive-thinking与动态预算.md)。

### 6.3 选择继续计算还是停止

设继续计算需要成本 `c`，预期质量提升为 `Delta A`，任务价值为 `v`，可以用一个简化的继续条件：

~~~math
v\,\mathbb{E}[\Delta A\mid x]-c>0
~~~

例如，若一次继续计算的预期准确率增量是 `0.05`，任务价值权重 `v=10` 个效用单位/准确率单位，增量成本 `c=0.4` 个效用单位，则 `10×0.05−0.4=0.1>0`，按这个简化准则值得继续。若预期增量降到 `0.03`，净值为 `10×0.03−0.4=−0.1`，应停止或转向其他检查。这不是可直接部署的概率模型，而是把业务权衡写出来。困难在于估计条件期望：模型可能不知道自己是否会被下一轮纠正，verifier 分数也可能没有校准。

### 6.4 多目标预算优化

对多个请求，可以把预算分配写成：

~~~math
\max_{b_1,\ldots,b_N}
\sum_{i=1}^{N}w_iA_i(b_i)
\quad\text{subject to}\quad
\sum_{i=1}^{N}C_i(b_i)\le C_{\mathrm{budget}}
~~~

`A_i(b_i)` 是第 `i` 个请求在预算 `b_i` 下的质量，`w_i` 是任务价值权重。现实系统还会加入 p95 延迟、并发 GPU、工具配额和安全约束。

用两个请求演示离散分配。设请求 A 的低/高档准确率为 `0.6/0.8`、成本为 `2/5`，权重 `w_A=1`；请求 B 的低/高档准确率为 `0.7/0.8`、成本为 `3/5`，权重 `w_B=2`；总预算是 7。两题都用低档花 5；升级 A 会多花 3，合计 8，超预算；升级 B 多花 2，合计 7，目标值从 `1×0.6+2×0.7=2.0` 增至 `1×0.6+2×0.8=2.2`。在这组离散选项下，应把额外预算给 B。真实问题通常需要在离线实验中估计每条路由的质量—成本曲线。

### 6.5 预算花在什么位置

候选数、单条长度、搜索深度、verifier 精度、工具调用和反思轮数是不同的预算轴。数学题可能从多路径采样中获益，代码题可能从测试和修复循环中获益，合同任务可能更需要检索和证据核对。

不能只用“token 数”作为预算，因为一百个生成 token 和一次数据库写操作的风险、延迟和成本不在同一尺度上。

## 7. 自适应计算与置信度

### 7.1 可用的难度信号

路由器可以使用：

- 输入长度和结构；
- 任务类型和历史切片；
- 初次答案的置信度；
- 多候选答案分歧；
- verifier 分数和分数间隔；
- 是否有可执行工具；
- 历史错误类型；
- 任务价值和风险。

这些信号只是预测器，不是事实。模型高置信度犯错的样本，正是需要 hard slice 的地方。

### 7.2 分歧

若候选数 `K>0`，且 `\hat p(y)` 是非负、总和为 1 的经验答案分布，可以用熵表示答案分歧：

~~~math
H(\hat p)
=
-\sum_y\hat p(y)\log\hat p(y)
~~~

约定 `0\log 0=0`；当 `K=0` 时，分布和熵都没有定义。例如，若候选答案的经验分布是 `\hat p(42)=0.75,\hat p(40)=0.25`，使用自然对数时，`H=−0.75\ln(0.75)−0.25\ln(0.25)≈0.562` nat。若所有候选都给同一答案，熵为 0；这只表示答案一致，不表示答案正确。分歧高时增加计算通常更有理由，但低分歧不一定安全：所有候选可能共享同一错误前提。分歧应和输入变化、证据覆盖和外部验证结合。

### 7.3 分数间隔

候选最高和次高分的差距为：

~~~math
\Delta_{\mathrm{score}}=s_{(1)}-s_{(2)}
~~~

候选少于两个时，`\Delta_{\mathrm{score}}` 没有定义；不能把单候选的间隔写成 0。若最高和次高分分别是 `0.81` 与 `0.78`，原始分差就是 `0.81−0.78=0.03`；它不是 3% 的正确概率，也不是自动可比较的置信度。小间隔可以触发更多验证或保留两个候选，大间隔可能允许提前停止。但只有经过校准的分数间隔才有较稳定的解释，未经校准的高间隔也可能是评分器自信地犯错。

### 7.4 动态停止

自适应计算的停止条件可以是：

- 程序测试和结构化检查已经通过；
- 多个独立路径得到等价答案，且证据覆盖足够；
- verifier 分数达到经过验证的阈值；
- 继续计算的预期收益低于成本；
- 风险或预算达到上限。

停止不是“模型觉得差不多了”。它应当写成可观察信号，支持离线复现和线上审计。

## 8. 延迟、并行和成本

### 8.1 并行候选的延迟

如果候选数 `K>0` 且 K 个候选可以完全并行，粗略的生成延迟接近最慢候选：

~~~math
R^{\mathrm{parallel}}
=
\max_{1\le k\le K}R_k^{\mathrm{gen}}
+R^{\mathrm{verify}}
~~~

例如，三条候选的生成时延为 `120、180、150 ms`，并行完成后 verifier 还需 `40 ms`，则近似请求时延为 `max(120,180,150)+40=220 ms`。现实中还要加队列等待、显存不足、批处理填充和 verifier 服务时间。并行降低墙钟延迟，却不降低总 token，也可能降低系统同时服务其他请求的吞吐。

### 8.2 串行搜索和工具循环

串行循环轮数 `H` 为正整数时，累计时延更接近：

~~~math
R^{\mathrm{serial}}
=
\sum_{t=1}^{H}
\left(
R_t^{\mathrm{gen}}
+R_t^{\mathrm{tool}}
+R_t^{\mathrm{verify}}
\right)
~~~

例如，两轮串行循环的三段时延分别是 `(100+80+20) ms` 和 `(120+50+30) ms`，总计 `200+200=400 ms`。工具调用、数据库锁和人工复核通常不能完全并行。相同 token 数的 self-consistency 和工具搜索，用户体验可能差几个数量级。

### 8.3 P50 和 P95

平均延迟掩盖不了高预算长尾。自适应系统需要在延迟样本非空时至少报告 P50、P95 和超时率，并按路由模式、任务类型和工具路径分组。P95 还要明确分位数插值规则；样本数太少时，不能把最大值无条件称为 P95。

本章 demo 使用最近秩（nearest-rank）约定：将 `n` 个延迟排序后取秩 `ceil(0.95n)`。若有 6 个观测值，秩为 `ceil(5.7)=6`，所以代码返回样本最大值。六个样本不足以稳定估计尾部延迟，报告时必须同时给样本量；生产比较还应采用预先固定的分位数定义和足够样本。高质量模式可以比普通模式慢，但必须有硬上限和降级行为。超时后的空答案、部分结果和人工转交都要计入任务质量，不能只统计成功完成的请求。

### 8.4 成本账本

总成本可以拆成：

这些成本项只有换算到同一单位后才能相加；没有可靠换算率时，应分别报告资源向量，不直接相加。

~~~math
C_{\mathrm{total}}
=
C_{\mathrm{model}}
+C_{\mathrm{verifier}}
+C_{\mathrm{tool}}
+C_{\mathrm{storage}}
+C_{\mathrm{review}}
~~~

如果要比较每个正确任务的成本，并且至少有一个任务通过独立成功检查：

~~~math
C_{\mathrm{per\ correct}}
=
\frac{C_{\mathrm{total}}}{N_{\mathrm{correct}}}
~~~

这里 `N_{\mathrm{correct}}` 是通过独立成功检查的任务数；当它为 0 时，单位正确任务成本没有定义，应报告 `None` 或“没有正确任务”。例如，某组请求的模型、verifier、工具、存储和人工复核成本分别为 `3、1、2、0.2、0` 个同口径单位，则 `C_total=3+1+2+0.2+0=6.2`。成本定义应包含失败重试、缓存未命中、日志、工具执行和人工复核，否则高预算模式会看起来比实际便宜。沿用后文 demo 的教学权重，direct 的总成本是 `488`、答对 2 题，故每正确题 `488/2=244` 个成本单位；adaptive 的总成本是 `3223`、答对 5 题，故 `3223/5=644.6` 个成本单位。单位成本更高不代表 adaptive 不值得：它在这组构造数据上把准确率从 `2/6` 提到 `5/6`。这些单位由 toy 权重构成，不是货币价格；生产账本口径见[第十三章](13-reasoning-effort与推理预算.md)。

## 9. Multi-agent、Ultra 和产品预算旋钮

### 9.1 从产品词回到可观测系统

产品可能把更高预算暴露为 effort、thinking level、deep research、ultra、coding agent 或类似名称。学习这些词时，先问它们在公开文档中改变了什么：最大输出、模型路由、工具调用、搜索深度、超时、并行 worker，还是只改变了服务等级。

如果文档没有公开内部实现，就只把它写成“一个更高预算的产品策略”，不要断言它一定使用了某种隐藏 CoT、MCTS 或多 Agent 架构。

### 9.2 Multi-agent 作为计算分解

多 Agent 系统可以把工作分给检索、规划、实现、测试和审查 worker。它增加的不只是模型 token，还有上下文传递、协调、冲突合并和权限隔离成本。

本章只把多 worker 当作一种计算分配形态；角色设计、协作协议和系统编排由[第十六章](16-multi-agent与ultra-test-time-compute.md)深入处理。一个可观察的 multi-agent 账本应包含：

- worker 数量和角色；
- 每个 worker 的输入输出 token；
- 工具和权限范围；
- 中间产物和证据；
- 冲突次数和重试次数；
- 主控合并耗时；
- 最终独立验证结果。

### 9.3 什么时候 multi-agent 有价值

当子任务可以相对独立、产物可验证、错误代价高且用户能接受延迟时，多 Agent 可能带来收益。简单问答、共享状态高度耦合或没有独立验证的任务，多个 worker 可能只是复制同一错误并增加协调噪声。

### 9.4 程序化 workflow 的边界

批量检索、表格运算、代码测试和格式转换通常应优先用确定性程序。模型适合做策略选择、异常解释和需要语义判断的部分。把所有环节都交给 Agent 并不等于拥有更多有效推理。

## 10. 失败模式与诊断

### 10.1 更多候选但没有新信息

候选表面不同、错误前提相同，说明生成分布没有提供独立探索。查看答案去重率、方法去重率和错误相关性，而不是只看 K。

### 10.2 Verifier 成为瓶颈

生成器已经提出正确候选，但 verifier 受长度、格式或自信语气影响，把错误候选排在前面。此时增加 K 可能降低选择准确率，应该改进 hard negative、程序检查和校准。

### 10.3 高预算掩盖了路由错误

平均准确率可能提高，但路由器把高预算分配给容易解决的样本，把低预算分配给真正的 hard slice。应按 difficulty、value、risk 和 route 分组报告。

### 10.4 反思重复错误

没有新证据时，反思循环往往只是对同一推理做同义改写。停止条件应要求新工具结果、不同路径、反事实检查或明确的错误定位。

### 10.5 搜索深度增加但质量下降

长搜索会积累错误、污染状态和放大评分偏差。深度与质量的关系不一定单调，必须测完整曲线和误剪率。

### 10.6 工具调用失败后没有降级

工具超时、版本冲突和权限不足会把串行循环卡住。系统应区分“无工具可验证”“工具异常”“工具返回矛盾”和“工具验证失败”，再决定重试、替代工具、人工复核或安全停止。

### 10.7 日志不能解释预算浪费

如果只记录最终答案，就不知道时间花在生成、评分、工具还是重试。每个节点和路由都要有 trace id、预算变化、停止原因和失败类型。

## 11. 一个推理时计算控制器

### 11.1 组件

一个可审计的控制器可以包含：

- Router：估计任务类型、难度、价值和风险；
- Generator：生成候选或下一步；
- Verifier：提供过程、结果、程序或证据检查；
- Search Controller：管理展开、剪枝、回溯和停止；
- Tool Executor：在隔离环境运行工具；
- Budget Manager：管理 token、时间、调用和并发；
- Aggregator：归一化、合并和选择结果；
- Logger：记录候选、评分、工具、预算和失败原因。

### 11.2 路由状态

路由器不仅输出模式，还应输出理由和预算：

~~~json
{
  "route": "verifier",
  "reason": "答案存在歧义且有可执行检查",
  "budget": {
    "max_tokens": 600,
    "max_verifier_calls": 4,
    "max_tool_calls": 1,
    "deadline_ms": 1200
  },
  "risk": "medium"
}
~~~

`route`、`reason`、`risk` 应是非空且受 schema 约束的字段；预算值应是非负整数，`deadline_ms` 还应允许为 0 但不能为负。生产系统应拒绝未知路由、负预算、空理由和未知风险等级，而不是静默使用默认值。这样才能在事后分析“为什么这个请求用了高预算”，并发现路由器是否把简单请求误升级。

### 11.3 停止原因

每个请求应记录明确的停止原因，例如 `verified_success`、`budget_exhausted`、`tool_timeout`、`low_expected_gain`、`human_review` 或 `unsafe_action_blocked`。停止原因比一个 `finished=true` 更能帮助定位质量和成本问题。

## 12. 综合案例：合同付款助手的动态预算

### 12.1 三种请求

合同助手收到三类请求：

~~~text
低风险：列出合同中定义的付款日期。
中风险：计算折扣、税率和含税金额，并标页码。
高风险：准备一笔付款草稿，并确认审批状态、收款账户和合同版本。
~~~

低风险请求可以先走直接抽取；中风险请求需要程序计算和引用核对；高风险请求即使算出正确金额，也不能跳过权限和人工确认。

### 12.2 初次路由

路由器可以使用文档类型、金额字段、风险标签和是否存在结构化工具。初次路由只负责选择预算，不负责宣称答案正确。

如果金额字段来自 OCR，且合同有多个版本，路由器应提高证据核对预算；如果是稳定的标准模板，可能降低搜索预算，但仍保留最终 schema 和金额校验。

### 12.3 继续还是停止

当初次结果的税率引用与正文一致、计算器通过、合同版本唯一时，系统可以停止检索；当两个税率候选冲突时，继续计算的价值更高；当工具超时且金额将用于真实付款时，应转人工，而不是用语言模型补猜。

### 12.4 评价路由而不是只评价答案

需要分别报告：

- 低风险任务是否被过度升级；
- 中风险任务是否获得正确引用；
- 高风险任务是否拦截了权限和版本问题；
- 每种路由的 p95 延迟和单位成功成本；
- 失败请求是否有可解释的停止原因。

## 13. 最小可运行实验：固定策略与自适应预算

下面的 demo 模拟 6 个 toy 请求，对比四种固定策略和一个自适应路由：

- `direct`：低预算直接回答；
- `self_consistency`：多样本采样；
- `verifier`：候选加 verifier；
- `search`：搜索加 verifier 或工具；
- `adaptive`：根据难度、价值和可验证性选择策略。

`TOKEN_COST`、`VERIFIER_COST` 和 `TOOL_COST` 是教学用的归一化成本权重；它们不是服务报价或跨系统通用换算率。`adversarial_math` 故意让高预算策略仍然失败。这个样本用来说明：把请求升级到更高预算不等于一定解决问题，高预算失败仍然是成本浪费和 hard slice 证据。

~~~python
from math import ceil, isfinite


CASES = [
    {
        "id": "easy_lookup",
        "difficulty": 0.15,
        "value": 0.20,
        "verifiable": False,
        "modes": {
            "direct": {"correct": True, "tokens": 32, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 120},
            "self_consistency": {"correct": True, "tokens": 180, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 260},
            "verifier": {"correct": True, "tokens": 240, "verifier_calls": 3, "tool_calls": 0, "latency_ms": 430},
            "search": {"correct": True, "tokens": 520, "verifier_calls": 5, "tool_calls": 0, "latency_ms": 900},
        },
    },
    {
        "id": "factual_short",
        "difficulty": 0.25,
        "value": 0.30,
        "verifiable": False,
        "modes": {
            "direct": {"correct": True, "tokens": 36, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 130},
            "self_consistency": {"correct": True, "tokens": 210, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 280},
            "verifier": {"correct": True, "tokens": 260, "verifier_calls": 3, "tool_calls": 0, "latency_ms": 460},
            "search": {"correct": True, "tokens": 540, "verifier_calls": 5, "tool_calls": 0, "latency_ms": 920},
        },
    },
    {
        "id": "algebra_hard",
        "difficulty": 0.62,
        "value": 0.80,
        "verifiable": True,
        "modes": {
            "direct": {"correct": False, "tokens": 90, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 220},
            "self_consistency": {"correct": True, "tokens": 520, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 680},
            "verifier": {"correct": True, "tokens": 430, "verifier_calls": 4, "tool_calls": 0, "latency_ms": 620},
            "search": {"correct": True, "tokens": 760, "verifier_calls": 6, "tool_calls": 0, "latency_ms": 1180},
        },
    },
    {
        "id": "code_patch",
        "difficulty": 0.90,
        "value": 0.95,
        "verifiable": True,
        "modes": {
            "direct": {"correct": False, "tokens": 120, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 260},
            "self_consistency": {"correct": False, "tokens": 650, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 760},
            "verifier": {"correct": False, "tokens": 610, "verifier_calls": 5, "tool_calls": 0, "latency_ms": 820},
            "search": {"correct": True, "tokens": 940, "verifier_calls": 5, "tool_calls": 2, "latency_ms": 1420},
        },
    },
    {
        "id": "logic_puzzle",
        "difficulty": 0.78,
        "value": 0.70,
        "verifiable": False,
        "modes": {
            "direct": {"correct": False, "tokens": 100, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 240},
            "self_consistency": {"correct": True, "tokens": 560, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 740},
            "verifier": {"correct": True, "tokens": 620, "verifier_calls": 5, "tool_calls": 0, "latency_ms": 860},
            "search": {"correct": True, "tokens": 980, "verifier_calls": 8, "tool_calls": 0, "latency_ms": 1500},
        },
    },
    {
        "id": "adversarial_math",
        "difficulty": 0.82,
        "value": 0.60,
        "verifiable": True,
        "modes": {
            "direct": {"correct": False, "tokens": 110, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 240},
            "self_consistency": {"correct": False, "tokens": 620, "verifier_calls": 0, "tool_calls": 0, "latency_ms": 760},
            "verifier": {"correct": False, "tokens": 590, "verifier_calls": 5, "tool_calls": 0, "latency_ms": 840},
            "search": {"correct": False, "tokens": 900, "verifier_calls": 7, "tool_calls": 0, "latency_ms": 1450},
        },
    },
]

TOKEN_COST = 1.0
VERIFIER_COST = 45.0
TOOL_COST = 100.0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def finite_non_negative_int(value):
    return type(value) is int and value >= 0 and finite_number(value)


def validate_cases(cases):
    require(isinstance(cases, list) and cases, "cases must be a non-empty list")
    case_ids = []
    required_modes = {"direct", "self_consistency", "verifier", "search"}
    for case in cases:
        require(isinstance(case, dict), "each case must be a dictionary")
        require(isinstance(case.get("id"), str) and case["id"], "case id must be non-empty")
        require(case["id"] not in case_ids, "case ids must be unique")
        case_ids.append(case["id"])
        require(finite_number(case.get("difficulty")) and 0 <= case["difficulty"] <= 1, "difficulty must be in [0, 1]")
        require(finite_number(case.get("value")) and 0 <= case["value"] <= 1, "value must be in [0, 1]")
        require(type(case.get("verifiable")) is bool, "verifiable must be boolean")
        modes = case.get("modes")
        require(isinstance(modes, dict) and set(modes) == required_modes, "modes must contain exactly the four fixed modes")
        for mode, stats in modes.items():
            require(isinstance(stats, dict), f"{mode} stats must be a dictionary")
            require(type(stats.get("correct")) is bool, "correct must be boolean")
            for field in ("tokens", "verifier_calls", "tool_calls", "latency_ms"):
                require(finite_non_negative_int(stats.get(field)), f"{field} must be a finite non-negative integer")


validate_cases(CASES)


def unit_cost(stats):
    require(isinstance(stats, dict), "stats must be a dictionary")
    for field in ("tokens", "verifier_calls", "tool_calls"):
        require(finite_non_negative_int(stats.get(field)), f"{field} must be a finite non-negative integer")
    try:
        cost = (
            TOKEN_COST * stats["tokens"]
            + VERIFIER_COST * stats["verifier_calls"]
            + TOOL_COST * stats["tool_calls"]
        )
    except OverflowError as error:
        raise ValueError("cost must be finite") from error
    require(finite_number(cost), "cost must be finite")
    return round(cost, 3)


def p95(values):
    require(isinstance(values, list) and values, "p95 requires a non-empty list")
    require(all(finite_non_negative_int(value) for value in values), "latencies must be finite non-negative integers")
    ordered = sorted(values)
    rank = max(1, ceil(0.95 * len(ordered)))
    return ordered[rank - 1]


def choose_route(case):
    if case["difficulty"] <= 0.35:
        return "direct"
    if case["verifiable"] and case["value"] >= 0.90:
        return "search"
    if case["verifiable"] and case["difficulty"] <= 0.70:
        return "verifier"
    if case["difficulty"] >= 0.60:
        return "self_consistency"
    return "direct"


def summarize(mode):
    require(mode in {"direct", "self_consistency", "verifier", "search"}, "unknown mode")
    chosen = [case["modes"][mode] for case in CASES]
    require(chosen, "cannot summarize an empty case list")
    correct = sum(item["correct"] for item in chosen)
    cost = sum(unit_cost(item) for item in chosen)
    latencies = sorted(item["latency_ms"] for item in chosen)
    return {
        "accuracy": round(correct / len(chosen), 3),
        "total_cost": round(cost, 3),
        "cost_per_correct": round(cost / correct, 3) if correct else None,
        "p95_latency_ms": p95(latencies),
    }


fixed_modes = ["direct", "self_consistency", "verifier", "search"]
fixed_summary = {mode: summarize(mode) for mode in fixed_modes}

adaptive_records = []
for case in CASES:
    route = choose_route(case)
    stats = case["modes"][route]
    adaptive_records.append(
        {
            "id": case["id"],
            "route": route,
            "correct": stats["correct"],
            "cost": unit_cost(stats),
            "latency_ms": stats["latency_ms"],
        }
    )

adaptive_correct = sum(item["correct"] for item in adaptive_records)
adaptive_cost = sum(item["cost"] for item in adaptive_records)
adaptive_latencies = sorted(item["latency_ms"] for item in adaptive_records)
adaptive_summary = {
    "accuracy": round(adaptive_correct / len(adaptive_records), 3),
    "total_cost": round(adaptive_cost, 3),
    "cost_per_correct": round(adaptive_cost / adaptive_correct, 3) if adaptive_correct else None,
    "p95_latency_ms": p95(adaptive_latencies),
}

marginal = {}
base_acc = fixed_summary["direct"]["accuracy"]
base_cost = fixed_summary["direct"]["total_cost"]
for mode in ["self_consistency", "verifier", "search"]:
    acc_gain = fixed_summary[mode]["accuracy"] - base_acc
    cost_gain = fixed_summary[mode]["total_cost"] - base_cost
    marginal[mode] = round(acc_gain / cost_gain, 6) if cost_gain else None

wasted_high_compute = [
    item["id"]
    for item in adaptive_records
    if item["route"] != "direct" and not item["correct"]
]

review = {
    "adaptive_accuracy_ok": adaptive_summary["accuracy"] >= 0.8,
    "adaptive_cheaper_than_search": adaptive_summary["total_cost"] < fixed_summary["search"]["total_cost"],
    "latency_ok": adaptive_summary["p95_latency_ms"] <= 1500,
    "wasted_high_compute": bool(wasted_high_compute),
}

print(f"fixed_summary={fixed_summary}")
print(f"adaptive_records={adaptive_records}")
print(f"adaptive_summary={adaptive_summary}")
print(f"marginal_accuracy_per_cost={marginal}")
print(f"wasted_high_compute={wasted_high_compute}")
print(f"review={review}")
~~~

**代码状态：已验证（2026-11-12；Python 3.12.9；仅使用标准库）。** 完整代码块已执行，实际输出与下方预期输出一致；另检查了有限数值、布尔值、NaN/无穷、超大整数计数、成本溢出和 nearest-rank 边界。

预期输出：

~~~text
fixed_summary={'direct': {'accuracy': 0.333, 'total_cost': 488.0, 'cost_per_correct': 244.0, 'p95_latency_ms': 260}, 'self_consistency': {'accuracy': 0.667, 'total_cost': 2740.0, 'cost_per_correct': 685.0, 'p95_latency_ms': 760}, 'verifier': {'accuracy': 0.667, 'total_cost': 3875.0, 'cost_per_correct': 968.75, 'p95_latency_ms': 860}, 'search': {'accuracy': 0.833, 'total_cost': 6460.0, 'cost_per_correct': 1292.0, 'p95_latency_ms': 1500}}
adaptive_records=[{'id': 'easy_lookup', 'route': 'direct', 'correct': True, 'cost': 32.0, 'latency_ms': 120}, {'id': 'factual_short', 'route': 'direct', 'correct': True, 'cost': 36.0, 'latency_ms': 130}, {'id': 'algebra_hard', 'route': 'verifier', 'correct': True, 'cost': 610.0, 'latency_ms': 620}, {'id': 'code_patch', 'route': 'search', 'correct': True, 'cost': 1365.0, 'latency_ms': 1420}, {'id': 'logic_puzzle', 'route': 'self_consistency', 'correct': True, 'cost': 560.0, 'latency_ms': 740}, {'id': 'adversarial_math', 'route': 'self_consistency', 'correct': False, 'cost': 620.0, 'latency_ms': 760}]
adaptive_summary={'accuracy': 0.833, 'total_cost': 3223.0, 'cost_per_correct': 644.6, 'p95_latency_ms': 1420}
marginal_accuracy_per_cost={'self_consistency': 0.000148, 'verifier': 9.9e-05, 'search': 8.4e-05}
wasted_high_compute=['adversarial_math']
review={'adaptive_accuracy_ok': True, 'adaptive_cheaper_than_search': True, 'latency_ok': True, 'wasted_high_compute': True}
~~~

这个结果说明四种固定策略的质量和成本并不相同。`search` 在这组构造数据上准确率最高，但总成本也最高；`adaptive` 把简单请求留在 direct，把代码修复升级到 search，因此以更低总成本得到相同的 0.833 准确率。

但 `adversarial_math` 仍然被路由到较高预算并且失败，所以 `wasted_high_compute` 为真。这不是代码出错，而是路由器和策略在 hard slice 上的可观察缺陷。真实系统应继续增加反事实样本、独立工具验证、人工复核或拒答策略，而不能只看自适应平均准确率。

## 14. 设计一次可复现的推理时计算实验

### 14.1 先固定任务和成功事件

数学题可以要求数值等价，代码题可以要求隐藏测试和资源限制，合同任务可以要求金额、引用、版本和状态全部满足。成功事件不清晰，预算曲线就没有稳定含义。

### 14.2 比较预算而不是只比较模式名称

`search`、`thinking`、`ultra` 这些名称不能直接比较。应记录每个策略的 token、模型调用、verifier、工具、搜索深度、并行度、缓存、p95 延迟和金钱成本。

### 14.3 保存候选全集和路由决策

只保存最终答案，无法知道失败来自候选生成、评分、剪枝、路由还是终局合并。实验日志应保留候选、分数、停止原因、工具观察、预算变化和失败类型。

### 14.4 画完整曲线

对固定预算和自适应路由分别画准确率—成本、准确率—延迟、成功率—token、hard slice 准确率和单位成功成本曲线。曲线上的每个点都应说明候选数量、最大长度和 verifier 版本。

### 14.5 做路由反事实

把同一个请求强制送到 direct、self-consistency、verifier 和 search，比较它在不同策略下的结果。这样可以评估路由器是否把真正需要高预算的请求正确升级。

### 14.6 检查收益是否来自独立信息

对多候选做语义去重、错误相关性和证据来源分析。若质量提升只来自更多重复候选，系统对 prompt、温度和生成器升级可能很脆弱。

## 15. 安全与权限

高预算不等于高权限。多次生成、多个 Agent 和更多工具调用会扩大提示注入、数据泄露、错误动作和日志暴露的表面积。

执行代码时使用沙箱；访问网页和文档时把外部内容当作不可信数据；对数据库和付款使用事务草稿；对删除、权限变更和外部通信保留人工确认。路由器只能选择预算，不能授予权限。

高风险任务还需要独立的规则和结果验证。一个高预算模式如果仍然无法确认金额、账户或合同版本，应返回不确定或转人工，而不是因为“已经想了很久”就交付。

推理轨迹和多 Agent trace 可能包含隐私、密钥、合同和源代码。日志采集要做最小化、脱敏、访问控制和保留期限管理。

## 16. 小练习

### 练习一：预算向量

为数学、代码、合同和普通事实查询分别设计 `(K,L,D,V,U,H)`，说明每个维度为什么不同。

### 练习二：候选覆盖和选择

构造一个候选集合，其中至少有一个正确答案，但最高 verifier 分数属于错误答案。计算 `A_exist` 和 `A_select`，解释两者差别。

### 练习三：边际收益

给出四个预算点的准确率和成本，计算相邻预算区间的边际收益，并判断在哪个点继续增加预算不再划算。

### 练习四：动态停止

为一个代码修复任务设计三个停止条件：测试通过、工具异常和预期收益低于成本。说明它们分别需要哪些可观察变量。

### 练习五：路由器评估

构造一个路由混淆矩阵，分别统计把难题降级和把简单题升级的代价。说明为什么不能只报告平均准确率。

### 练习六：multi-agent 账本

设计一份记录，包含 worker、工具、权限、输入输出 token、冲突、重试和最终验证结果。指出哪些字段对于复盘预算浪费最重要。

### 练习七：运行 demo

把 `adversarial_math` 的 `search` 策略改为正确，观察 adaptive 的准确率和单位成功成本；再把一个简单请求改为错误，观察高预算浪费如何变化。

## 17. 本章总结

推理时计算扩展是在模型参数固定后，通过更多生成、采样、验证、搜索、工具或协调计算提高任务质量。它不是“越多越好”，而是一个按请求和风险分配资源的问题。

候选数增加主要改善覆盖，verifier 和程序改善选择，搜索改善中间探索，工具提供外部约束，动态路由把预算集中给值得升级的请求。每条路线都有不同的相关错误、延迟、成本和安全边界。

质量曲线、边际收益、候选存在率、最终选择准确率、p95 延迟、单位成功成本和 hard slice 是理解推理时计算的基本指标。必须把生成失败、选择失败、路由失败和工具失败分开归因。

产品中的 effort、thinking level、deep research 或 multi-agent 等词，应当先还原为可观察的预算和系统行为；没有公开证据时，不要把产品档位写成确定的内部算法。

最终，推理时计算系统需要做到三点：知道什么时候继续计算，知道什么时候停止，知道什么时候证据不足应当拒绝交付或转人工。

下一章将进入数学推理训练，讨论数据构造、过程与结果监督、数学验证器和训练评估之间的关系。

## 18. 资料索引

1. Charlie Snell et al., *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*, ICLR 2025，[arXiv v1](https://arxiv.org/abs/2408.03314v1)。
2. Bradley Brown et al., *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling*, [arXiv v3](https://arxiv.org/abs/2407.21787v3)。
3. Mohsen Hariri et al., *Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and Reproducibility*, [arXiv 预印本 v2](https://arxiv.org/abs/2608.04001v2)。
4. Zhiyuan Zhai et al., *Adaptive Test-Time Compute Allocation for Reasoning LLMs via Constrained Policy Optimization*, [arXiv 预印本 v1](https://arxiv.org/abs/2604.14853v1)。
5. Shunyu Yao et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*, NeurIPS 2023，[arXiv v2](https://arxiv.org/abs/2305.10601v2)。
6. Xuezhi Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*, ICLR 2023，[arXiv v4](https://arxiv.org/abs/2203.11171v4)。
7. Karl Cobbe et al., *Training Verifiers to Solve Math Word Problems*, [arXiv v2](https://arxiv.org/abs/2110.14168v2)。
8. Mark Chen et al., *Evaluating Large Language Models Trained on Code*, [arXiv v2](https://arxiv.org/abs/2107.03374v2)。

来源按论文或预印本的固定版本标注。正文中的实验结论保留其任务、模型、预算和评估设置；本章曲线、成本和代码样本是教学构造，不能代替目标系统复测。
