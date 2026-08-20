# 第六章：搜索推理：从状态空间到可控探索

一条推理链只能沿着当前选择继续向前。如果模型在第二步把“至少”读成“恰好”，或者在代码修复时选择了错误的根因，后面每一步都可能在错误前提上变得越来越完整。文本看起来越连贯，错误反而越难被发现。

搜索推理提供了另一种组织方式：模型先提出多个中间候选，再让评分器、程序、工具或环境反馈帮助选择；当一条路走不通时，系统保留其他分支，必要时回溯或重新展开。Tree-of-Thought、beam search、best-first search、Monte Carlo Tree Search，以及带工具环境的 language agent tree search，都可以放在这个框架中理解。

搜索不是“让模型多想几秒”的同义词。它是一个有状态空间、动作、转移、评分、剪枝和预算的计算过程。搜索能否带来收益，取决于三个条件是否同时成立：候选分支之间确实有有价值的差异；中间状态能够被可靠地比较；增加的计算成本没有超过任务所能承受的范围。

本章先用初学者能理解的例子建立直觉，再给出状态空间和搜索算法的数学表达，最后讨论工具环境、错误剪枝、成本账本、评估、综合案例和可运行实验。读者不应把某个算法名称当成答案；真正需要掌握的是，为什么要搜索、搜索什么、凭什么保留一条路，以及如何证明它没有把正确路径过早丢掉。

## 0.1 一个具体问题：同一目标，几条不同的路

有一个水壶容量为 8 升和 5 升，初始都是空的，目标是得到恰好 3 升。可执行动作包括装满、倒空和互相倾倒。模型可以写出一条动作序列，也可以先枚举多种可能：

~~~text
路径 A：装满 5 升 -> 倒入 8 升 -> 再装满 5 升 -> 倒入 8 升直到 8 升满
路径 B：装满 8 升 -> 倒入 5 升 -> 倒空 5 升 -> 把剩余 3 升保留在 8 升壶中
路径 C：先把 5 升倒入 8 升，再重复相同动作
~~~

路径 A 和 B 可能都能到达目标，路径 C 可能陷入重复状态。若系统只沿着一条语言生成链前进，早期动作一旦不合适，就要从头重来；若系统维护多个状态，就可以比较剩余容量、动作长度和是否重复，并继续扩展更有希望的分支。

对合同审阅助手也一样。一个候选可能先检索付款条款，再计算金额；另一个候选先抽取所有数字，再回到条款确认条件。只要中间状态能表示“已经确认了什么、还缺什么、证据来自哪里”，搜索就不只是生成不同文字，而是在探索不同的证据获取路径。

## 0.2 资料与证据边界

本章的主要研究入口是 Tree of Thoughts、Language Agent Tree Search，以及经典的 Monte Carlo Tree Search 和 UCT 资料。

`Tree of Thoughts: Deliberate Problem Solving with Large Language Models` 讨论了让语言模型生成、评估和搜索中间 thought 的框架。它支持“把推理组织为多个候选状态”的研究背景，但论文中的任务、提示和实验设置不能直接推出所有模型和所有任务都能获得相同收益。

`Language Agent Tree Search` 把树搜索与语言代理、环境反馈和反思机制结合起来，适合说明当动作会改变外部环境时，搜索状态不再只是文本前缀。它仍然是特定研究方法，不应被写成闭源产品的内部实现事实。

UCT 和 MCTS 的理论与算法来自更早的树搜索和蒙特卡洛规划研究。把它们用于语言模型时，模型生成的 thought、工具结果和自然语言评分会带来新的噪声；经典公式可以说明算法结构，却不能保证语言评分满足游戏搜索中的理想假设。

本章的数字、候选路径和 Python demo 都是教学构造。它们用于解释节点数量、分数、剪枝和成本之间的关系，不代表任何公开模型的 benchmark 成绩。对于论文、公开仓库、框架 API 和本章 toy 代码，证据等级始终分开记录。

## 0.3 初学者视角：从“一条链”到“多条候选路径”

普通 Chain-of-Thought 可以画成一条直线：

~~~text
问题 -> 步骤 1 -> 步骤 2 -> 步骤 3 -> 答案
~~~

搜索则保留多个中间状态：

~~~text
问题
  -> thought A -> thought A1 -> 答案
  -> thought B -> thought B1 -> 答案
  -> thought C -> 发现矛盾，停止
~~~

最初接触搜索时，容易产生一个误解：分支越多，答案一定越好。实际上，分支只是候选。系统还需要一个能区分好坏的评分机制。如果所有分支都由同一个错误规则生成，搜索只会得到更多相似错误；如果评分器偏好长文本，搜索会把预算花在更长的错误解释上。

专家需要进一步区分三种计算：候选生成计算、节点评分计算和环境执行计算。增加 beam 宽度可能增加三者中的一项或全部；如果工具调用无法并行，搜索的墙钟延迟还可能远高于 token 数量看起来的增长。

## 1. 把推理写成状态空间

### 1.1 状态、动作和转移

要让搜索算法工作，至少要定义状态 `s`、动作 `a` 和转移函数 `T`。状态表示系统当前知道的内容，动作表示下一步可以做什么，转移表示动作执行后状态如何变化。

~~~math
s_{t+1}=T(s_t,a_t)
~~~

在数学题中，`s_t` 可以是题目和已经写出的推导前缀，`a_t` 可以是一个代数变换；在代码任务中，`s_t` 还应包括当前代码、测试结果和错误日志；在工具任务中，`s_t` 可能包括检索证据、权限状态、资源版本和未解决的约束。

如果只把自然语言前缀当成状态，系统可能遗漏工具结果和外部环境变化。两个文本相同的节点，若一个节点已经执行过付款模拟、另一个没有执行，它们不是同一个真实状态。

### 1.2 一个更完整的状态表示

可以把第 `t` 步的搜索状态写成：

~~~math
s_t=(x,z_{1:t},m_t,b_t,r_t)
~~~

其中：

- `x` 是原始问题或任务输入；
- `z_1,...,z_t` 是已经生成的 thought、步骤或动作；
- `m_t` 是工具结果、证据、测试和约束元信息；
- `b_t` 是已经消耗的预算；
- `r_t` 是风险、权限或需要人工确认的状态。

`b_t` 和 `r_t` 很容易被初学者忽略。一个分数很高但已经超出 token 预算的节点，不能继续扩展；一个数学上正确但准备删除生产数据的动作，也不能因为语言评分高就被执行。

### 1.3 动作集合

动作不是任意一句续写，而是当前状态下允许的下一步：

~~~math
a_t\in\mathcal{A}(s_t)
~~~

`A(s_t)` 由任务协议、模型生成器、工具 schema 和权限共同决定。对于代码修复，动作可以是“修改某一行”“新增测试”“运行测试”；对于文档问答，动作可以是“检索某一条款”“核对版本”“形成一个带引用的 claim”。

动作空间如果定义得太宽，模型会生成许多不可执行或重复的候选；如果定义得太窄，正确解法可能根本不在搜索空间中。搜索失败不一定是评分器失败，也可能是动作设计遗漏了必要操作。

### 1.4 终止状态

搜索需要明确什么时候停止。终止条件可能包括：

- 得到满足约束的最终答案；
- 所有必要测试通过；
- 证据已经覆盖所有 claim；
- 达到最大深度或 token 预算；
- 工具返回不可恢复的错误；
- 节点风险超过系统允许范围。

“模型说完成了”不是充分的终止条件。终止状态应由任务协议、工具结果或独立检查共同决定。

### 1.5 轨迹和搜索树

从初始状态 `s_0` 依次执行动作可以得到轨迹：

~~~math
\tau=(s_0,a_0,s_1,a_1,\ldots,s_T)
~~~

如果一个状态可以有多个动作，所有可能轨迹组成一棵树或更一般的有向图。不同动作可能到达同一个状态，所以实际系统有时应把搜索树合并成图，避免重复计算。

## 2. 分支数量、深度和预算

### 2.1 完整树为什么很快爆炸

假设每个节点平均有 `B` 个候选动作，最大搜索深度为 `D`，不做剪枝时节点总数近似为：这里的 `D` 是边的层数，根节点位于第 0 层；`B` 必须是非负数，`D` 必须是非负整数。只有每个节点的分支数都相同，这个等式才是精确计数；使用“平均分支数”时，它只是预算估计。

~~~math
N_{\mathrm{full}}
=
\sum_{d=0}^{D}B^d
=
\frac{B^{D+1}-1}{B-1}
\qquad (B\ne 1)
~~~

当 `B=1` 时，不能使用上面的除法形式，而应取极限 `N_{\mathrm{full}}=D+1`；当 `B=0` 时，按 `0^0=1` 的计数约定只保留根节点，节点数为 1。比如 `B=4`、`D=6` 时，节点数已经是 `1+4+16+64+256+1024+4096=5461`。这还没有计算每个节点的长上下文评分、工具调用和重试。

公式中的“平均分支数”只是粗略估算。真实生成器可能在简单状态生成 2 个候选，在不确定状态生成 20 个候选；工具动作还可能产生不同数量的观察结果。因此工程账本要记录每层实际节点数，而不是只用理论上限。

### 2.2 有效分支数

语言模型可能生成很多表面不同、语义相同的候选。可以用去重后的有效分支数描述搜索真正探索了多少选择：

~~~math
B_{\mathrm{eff}}
=
\left|\operatorname{unique}(\operatorname{normalize}(A(s)) )\right|
~~~

`normalize` 可以做格式归一化、变量重命名、答案单位统一或状态哈希。它不应过度合并语义不同的路径，否则会把真正的多样性误判为重复。

### 2.3 节点、token 和工具成本

对第 `i` 个任务，可以把节点数量、生成 token 和工具成本分别记为：

~~~math
C_{\mathrm{node}}
=
\sum_{n\in\mathcal{N}_i}1
~~~

~~~math
C_{\mathrm{tok}}
=
\sum_{n\in\mathcal{N}_i}T_n
~~~

~~~math
C_{\mathrm{tool}}
=
\sum_{n\in\mathcal{N}_i}c_{\mathrm{tool}}(n)
~~~

`N_i` 是搜索过程中访问的节点集合，`T_n` 是节点生成或评分产生的 token，`c_tool(n)` 可以是一次调用的金钱成本、执行时间或风险权重。三种账本不能互相替代：缓存可能减少 token，却不一定减少工具延迟；并行生成可能增加 token，却减少墙钟时间。

### 2.4 单位成功成本

对整个评估集，若至少有一个任务成功，搜索系统的单位成功成本可以写成：

~~~math
C_{\mathrm{success}}
=
\frac{
 C_{\mathrm{generate}}
 +C_{\mathrm{score}}
 +C_{\mathrm{tool}}
 +C_{\mathrm{retry}}
 +C_{\mathrm{review}}
}{
 N_{\mathrm{success}}
}
~~~

这里 `N_{\mathrm{success}}` 是通过独立终局检查的任务数，不是“模型返回了文本”的任务数。当 `N_{\mathrm{success}}=0` 时，单位成功成本没有定义，应报告 `None` 或“没有成功任务”，不能用分子本身伪装成单位成本。如果搜索把成功率从 70% 提高到 75%，但生成和评分成本增加十倍，系统是否值得采用要结合任务价值、错误损失和延迟要求判断。不能只看准确率的绝对提升。

## 3. Tree-of-Thought：把中间思路变成搜索节点

### 3.1 thought 的含义

Tree-of-Thought，简称 ToT，把推理过程组织成多个可评估的 thought。thought 可以是一个中间结论、子问题分解、候选计划、代码修改或数学变换，不必是单个 token，也不必是一整篇答案。

一个 thought 要成为好的搜索节点，通常需要满足三个条件：读者或工具能判断它是否成立；它能和已有状态合并；它能生成下一步动作。如果一段文本既不能验证，也不能改变状态，只是装饰性解释，就不适合作为搜索节点。

### 3.2 ToT 的基本循环

一个抽象的 ToT 循环是：

1. 从初始状态建立 frontier；
2. 对 frontier 中的每个状态提出多个候选 thought；
3. 将 thought 应用到状态，得到新节点；
4. 用过程评分、结果检查或工具反馈评估节点；
5. 选择、剪枝或合并节点；
6. 继续展开，直到得到终止状态或耗尽预算。

可以把候选状态集合写为：

~~~math
\mathcal{F}_{t+1}
=
\operatorname{Select}
\left(
\left\{T(s,a):s\in\mathcal{F}_t,\ a\in\mathcal{A}(s)\right\}
\right)
~~~

`Select` 不一定是简单的 top-k。它可以是 beam、阈值筛选、随机抽样、去重后保留、MCTS 选择，或者程序验证后只保留可执行状态。

`\mathcal{F}_t` 是有限 frontier；它可以为空。若 frontier 为空，或某个状态的 `\mathcal{A}(s)` 为空，说明当前搜索没有可扩展候选，系统应进入“找到终局”或“搜索失败”的终止状态，而不是凭空生成一个节点。`Select` 还必须明确保留数量和 tie-break；否则相同分数下依赖输入顺序的结果无法复现。

### 3.3 thought 粒度的取舍

thought 太短时，节点数和评分次数会暴涨。比如把每个 token 都当成一个节点，搜索会被语言表面细节淹没。

thought 太长时，一步内部可能已经包含多个不可逆错误；评分器只能对整段做判断，不能知道应该回滚到哪一部分。

合理粒度通常是“一个能改变问题状态的逻辑单元”。在数学题中可能是一个公式变换，在规划中可能是一个动作，在代码任务中可能是一次可测试修改。不同任务必须通过消融实验选择粒度，不能直接套用某个论文的分段方式。

### 3.4 ToT 与单链 CoT 的关系

单链 CoT 是搜索宽度为 1 的特殊情况，但实际实现还可能有温度、重采样和隐藏状态等差异。ToT 的核心变化不是简单增加输出长度，而是让系统在中间节点进行选择和回溯。

如果 ToT 的所有候选都在最后一次性生成、直到结束才投票，它更接近 self-consistency；如果每层都根据中间状态筛选，才体现了树搜索的特点。

## 4. Beam Search：按层保留前 K 个状态

### 4.1 基本更新

设第 `t` 层 frontier 为 `F_t`，每个状态生成动作集合 `A(s)`，节点评分为 `S`。beam search 的更新可以写成：

~~~math
F_{t+1}
=
\operatorname{TopK}
\left(
\left\{T(s,a):s\in F_t,\ a\in A(s)\right\},
S,
K
\right)
~~~

`K` 是正整数 beam 宽度。每层先扩展再排序，只保留分数最高的 `K` 个节点；如果候选数小于 `K`，就保留全部候选；如果候选集合为空，frontier 为空并终止。分数必须是有限值，并且相同分数时要使用固定的次级键，例如状态 ID 的字典序，不能依赖哈希表遍历顺序。

### 4.2 Beam 的优点

Beam 的优点是实现简单、预算容易估算、适合批量并行。对于每层候选数稳定、评分器可快速调用的任务，它是从单链生成走向搜索的自然第一步。

它还容易加入硬约束，例如 JSON 可解析、代码能编译、状态没有重复、工具参数符合 schema。硬约束先过滤，软评分再排序，通常比把所有约束压成一个语言模型分数更容易诊断。

### 4.3 Beam 的三个弱点

第一，beam 可能过早丢弃潜在正确路径。正确解法常常在中间步骤看起来不如错误捷径漂亮。

第二，候选容易同质化。模型从同一个前缀采样时，多个 thought 可能只是同义改写，beam 宽度增加却没有增加有效覆盖。

第三，分数尺度可能跨层漂移。长节点获得更多评分机会，或者评分器对短文本有系统偏好，都会使简单 top-k 失去可比性。

### 4.4 长度和路径分数

如果直接把每一步分数相加，长路径可能因为步骤多而得到更大或更小的数值，取决于分数符号。对包含至少一个步骤、且每个 `s_j` 都是有限数的路径，可以使用平均分：

~~~math
S_{\mathrm{mean}}(z_{1:t})
=
\frac{1}{t}\sum_{j=1}^{t}s_j
~~~

也可以使用折扣累计分数，其中折扣因子满足 `0\le\gamma\le 1`：

~~~math
S_{\mathrm{disc}}(z_{1:t})
=
\sum_{j=1}^{t}\gamma^{j-1}s_j
~~~

当 `t=0` 时，平均分没有定义；折扣分也应把空路径当作“尚未评分”，而不是默认得分为 0。平均分和折扣分都改变了搜索偏好。它们不是数学上“正确”的修正，只是不同的决策策略；需要在相同候选集合上比较长度偏差和最终任务效果。

## 5. Best-First、A* 和回溯

### 5.1 Best-first 的思想

Best-first search 维护一个全局候选池，每次取当前评分最高的节点扩展，而不是严格按层推进。它适合某些路径很快变得明显有希望、另一些路径需要暂时搁置的任务。

如果节点分数只表示当前状态质量，best-first 可能沉迷于一个局部看起来很好的分支。更合理的评分有时需要同时考虑已经付出的代价和对未来的估计：

~~~math
F(n)=g(n)+h(n)
~~~

`g(n)` 是从起点到节点 `n` 的累计代价，`h(n)` 是对剩余代价或未来价值的估计。经典 A* 对启发式函数有额外假设；语言模型的 PRM 分数通常不满足可采纳性和一致性，因此这里只能借用结构直觉，不能宣称得到经典 A* 的最优性保证。

### 5.2 回溯

当一个节点被证明违反硬约束，搜索应回溯到最近的可替代决策点，而不是继续在错误状态上生成文本。回溯需要保存父指针、已尝试动作、工具状态和缓存键。

有些动作不可逆，例如真实付款、删除数据和发送外部消息。对这类动作，搜索应在模拟环境或事务草稿中进行，只有最终决策通过独立权限检查后才允许提交。

### 5.3 状态合并

两条不同路径可能到达同一个状态。若状态的可观测约束完全相同，可以合并节点，保留较低成本或更高可信度的到达路径：

~~~math
n_a\sim n_b
\quad\Longleftrightarrow\quad
\operatorname{StateKey}(n_a)=\operatorname{StateKey}(n_b)
~~~

`StateKey` 不能只对自然语言文本做哈希。它还应包含工具版本、文档版本、权限状态、随机种子和未完成的约束，否则两个看似相同的文本可能对应不同环境。

## 6. Monte Carlo Tree Search：探索和利用

### 6.1 四个阶段

MCTS 通常重复四个阶段：

1. Selection：从根节点沿选择规则走到待扩展节点；
2. Expansion：生成一个或多个新动作和子节点；
3. Simulation：估计从子节点继续发展的结果；
4. Backpropagation：把结果沿父链回传，更新访问次数和价值。

在语言任务中，simulation 不一定是真正随机 rollout。它可以是语言模型继续生成、工具执行、快速 value model 预测或最终答案 verifier。

### 6.2 UCT 公式

为便于教学，可以把带平滑项的 UCT 选择分数写成：

~~~math
U(v)
=
Q(v)
+c\sqrt{\frac{\log(N_p+1)}{N_v+1}}
~~~

`Q(v)` 是节点当前平均价值，`N_v` 和 `N_p` 分别是节点与父节点的非负整数访问次数，`c` 是非负探索系数。第一项鼓励利用已知高价值节点，第二项鼓励访问次数少的节点。这里的 `+1` 是为了让教学 demo 在 `N_v=0` 或 `N_p=0` 时仍可计算；经典 UCT 实现也常把未访问子节点直接赋为 `+\infty`，优先完成首次访问，而不是使用这个平滑形式。两种约定不能混写，实验报告必须说明采用哪一种。

语言模型版本通常还要处理价值归一化、未完成节点、不同动作成本和工具失败。如果一个节点的价值来自不可校准的语言评分，UCT 公式仍然可以运行，但它不再自动拥有经典随机树搜索的统计保证。

### 6.3 访问次数不是质量

某个节点访问次数多，可能是因为它容易被选择，也可能是因为探索项推动它被尝试。不能把访问次数直接解释成“模型认为它最正确”。最终选择应查看平均价值、置信度、终局结果和成本。

### 6.4 MCTS 适合什么任务

MCTS 更适合有明确状态、可执行动作和可重复反馈的规划任务，例如游戏、代码修复、组合搜索和工具流程。对完全开放的写作任务，如果没有可靠状态和结果检查，MCTS 可能只是在语言空间中反复生成相似文本。

## 7. LLM 代理树搜索与环境反馈

### 7.1 从文本节点到环境节点

语言代理的动作可能是搜索网页、读取文件、运行代码、调用数据库或修改草稿。执行动作后得到的观察值会改变下一步可选动作，因此搜索节点必须包含环境状态。

~~~math
o_{t+1}=E(s_t,a_t)
~~~

`E` 是环境执行函数，`o_{t+1}` 是观察结果。语言模型可以根据 `o_{t+1}` 更新状态，但观察结果本身不应被当成无条件可信的自然语言指令。

### 7.2 Language Agent Tree Search 的启发

公开的 Language Agent Tree Search 工作说明了一种研究路线：让语言模型提出动作和反思，环境提供反馈，树搜索在多个候选轨迹之间分配计算。它有助于理解“代理搜索”的结构，但具体的反思提示、价值估计和工具接口仍属于论文实验设置。

工程实现还要增加事务、权限、重试、环境快照和资源清理。没有这些机制，搜索可能在不同分支之间污染文件、数据库或浏览器状态，导致分数无法比较。

### 7.3 可回放的环境状态

要公平比较两个搜索分支，环境应尽可能可回放。可以保存：

- 输入和工具版本；
- 当前文件或数据库快照；
- 权限和身份上下文；
- 外部响应和时间戳；
- 随机数状态；
- 超时、异常和资源使用。

如果每个分支看到的网页内容、数据库行或代码依赖都不同，搜索分数的差异可能来自环境变化，而不是动作质量。

## 8. 评分函数：凭什么保留一条路径

### 8.1 多种评分来源

搜索节点可以由不同评分器判断：

- 语言模型自评；
- process verifier；
- outcome verifier；
- 程序编译和测试；
- 规则约束；
- 检索证据支持；
- 工具返回的状态变化；
- 人工或专家复核。

语言自评便宜但容易自我确认；程序检查精确但覆盖有限；人工判断灵活但昂贵。更可靠的系统会按任务分解评分来源，而不是让一个标量承担所有职责。

### 8.2 组合节点分数

一个教学性的组合分数可以写成：

~~~math
S(n)
=
\lambda_p S_{\mathrm{process}}(n)
+\lambda_o S_{\mathrm{outcome}}(n)
+\lambda_e S_{\mathrm{evidence}}(n)
+\lambda_t S_{\mathrm{tool}}(n)
-\lambda_c C(n)
-\lambda_r R(n)
~~~

`S_process` 衡量中间步骤，`S_outcome` 衡量已知终局结果，`S_evidence` 衡量证据支持，`S_tool` 衡量工具反馈，`C(n)` 是计算成本，`R(n)` 是风险。所有权重都需要通过任务数据校准，不能从一个任务直接搬到另一个任务。

### 8.3 硬约束和软评分分开

JSON 解析失败、权限不足、代码无法编译、状态违反不变量等情况通常应先作为硬约束处理，而不是只扣一点分。软评分适合比较都满足硬约束的候选。

如果把硬约束和审美偏好压成一个分数，模型可能用很高的语言分数抵消权限违规或未定义字段。分层判断更容易审计：先问能不能执行，再问哪个可执行候选更好。

### 8.4 分数校准和不确定性

如果分数用于概率阈值或人工复核，应验证分数和真实成功率的关系。排名正确不等于校准正确。两个候选的分数差距很小，可能表示模型不确定，也可能表示两个分数都没有概率意义。

当至少有两个候选节点，且它们的分数都是有限数时，可以记录最高和次高节点的间隔：

~~~math
\Delta_{\mathrm{top}}
=S_{(1)}-S_{(2)}
~~~

当候选少于两个时，`\Delta_{\mathrm{top}}` 没有定义；不要把单个候选的间隔写成 0。间隔小的时候保留备选、增加验证或请求更多证据，通常比强行输出一个候选更稳妥；但保留备选会增加成本，必须纳入预算。

## 9. 剪枝：减少成本，也可能丢掉正确答案

### 9.1 常见剪枝方法

搜索常用的剪枝包括：

- 分数低于阈值；
- 每层保留 top-k；
- 合并重复状态；
- 超过深度或 token 预算；
- 硬约束失败；
- 工具执行失败；
- 已经证明无法达到目标；
- 与已知更低成本状态支配相同约束。

每种剪枝都应说明它依赖什么证据。硬约束失败通常可以安全剪掉；“语言模型觉得不太好”则只是软信号，误剪概率需要单独评估。

### 9.2 正确路径被剪掉

设第 `i` 个任务的搜索空间中存在至少一条正确路径，若该路径在终局前被剪掉，就记为一次 false negative。在分母非空、也就是评估集中至少有一个任务生成了正确路径时，剪枝错误率可以写成：

~~~math
R_{\mathrm{prune}}
=
\frac{
\sum_{i=1}^{N}
\mathbb{1}[\text{correct path pruned}_i]
}{
N_{\mathrm{tasks\ with\ a\ correct\ path}}
}
~~~

分母不能随意使用所有任务数。如果生成器根本没有产生正确路径，不能把它归因于剪枝；需要区分候选生成失败和搜索选择失败。如果没有任何任务生成正确路径，该指标返回 `None`，而不是 0；0 表示“已知有正确路径的任务中没有发生误剪”，两者含义不同。

### 9.3 早期和后期的不同策略

早期节点承载许多潜在未来，评分器对它们的判断通常更不稳定，因此可以保留更多多样性，使用较宽 beam 或较宽阈值。后期节点已经接近终局，工具测试和硬约束更有信息，可以加强筛选。

这不是固定算法，而是一个资源分配原则。实验应比较不同深度的保留率、正确路径覆盖率和最终成本。

### 9.4 支配关系和重复状态

如果两个节点达到相同约束状态，其中一个成本更高、证据不更多、风险也不更低，它可能被另一个节点支配。可以定义一个简单支配关系：

~~~math
n_a\succ n_b
\quad\text{if}\quad
S(n_a)\ge S(n_b),\ C(n_a)\le C(n_b),
\text{且约束覆盖不低于 }n_b
~~~

支配剪枝需要完整状态表示。只比较最终文本或语言分数，容易错误合并不同证据和权限状态。

## 10. 多样性、相关错误和搜索退化

### 10.1 候选不同不等于方法不同

三个候选只把“因此”改成“所以”，对搜索没有真正帮助。在候选集合非空、且归一化规则固定时，可以先做答案和状态归一化，再统计有效多样性：

~~~math
D_{\mathrm{answer}}
=
\frac{|\operatorname{unique}(\operatorname{answer}(\mathcal{C}))|}
{|\mathcal{C}|}
~~~

对规划任务，还要比较动作序列、访问的证据和中间状态，而不是只比较最终字符串。

### 10.2 重复率

在候选集合非空时，重复率可以写为：

~~~math
R_{\mathrm{dup}}
=
1-
\frac{|\operatorname{unique}(\operatorname{normalize}(\mathcal{C}))|}
{|\mathcal{C}|}
~~~

候选集合为空时，多样性和重复率都没有定义，应报告 `None`。重复率高，说明增加搜索宽度可能只是在重复计算；重复率低也不能证明候选有用，因为不同的错误方法仍然会导致低质量多样性。

### 10.3 相关错误

如果所有候选共享同一个错误的题意解释，投票、beam 和 MCTS 都可能失败。要测试搜索是否真的增加了独立证据，应按错误类型统计候选相关性，并构造反事实题目：只改变一个条件，看候选路径是否相应改变。

### 10.4 多样性和质量的冲突

强行要求候选不同，可能引入明显低质量分支；强行追求高分，可能让所有候选收敛到同一表面模式。实际系统可以分阶段：先用生成器提供多种策略，再用工具和 verifier 做质量筛选。

## 11. Search 和 Self-Consistency 的区别

Self-consistency 通常是独立生成多条完整推理链，最后抽取答案并聚合。它的中间过程一般不会影响其他候选的展开。

Search 是分步展开、分步评分和动态选择。某个分支的中间状态会决定它是否继续消耗预算，也可能回溯到父节点并尝试另一动作。

可以用下面的对照理解：

~~~text
self-consistency：多条完整链 -> 最终答案归一化 -> 投票或重排
search：          多步展开 -> 中间评分/工具反馈 -> 剪枝/回溯 -> 终局选择
~~~

Self-consistency 实现简单，适合候选独立且最终答案容易验证的任务；search 更适合中间状态有信息、错误可以提前发现、并且任务价值能覆盖额外成本的场景。

两者也可以组合：先采样多个策略，再对每个策略进行局部搜索；或在搜索叶子上做 self-consistency。组合后必须重新计算成本和相关错误，不能把两个方法的收益简单相加。

## 12. 工具反馈和搜索环境

### 12.1 代码修复

代码修复的状态可以包括源码、测试结果、静态分析和依赖版本。动作可以是修改、运行测试或增加断言。工具反馈比语言模型自评更接近任务结果，但测试覆盖和执行环境仍决定信号质量。

一个候选通过公开测试，不等于在隐藏输入、并发、资源限制和安全输入上正确。搜索记录中应保留失败测试、超时和环境异常，而不是把所有状态压成一个布尔值。

### 12.2 数学计算

计算器、符号系统和单位检查器可以验证局部步骤。它们不能独立证明模型正确抽取了自然语言条件，也不能检查所有证明中的语义跳跃。

因此数学搜索可采用“自然语言状态 + 结构化变量 + 工具结果”三层表示。工具结果只证明它实际检查的表达式，不能替代题意理解。

### 12.3 文档和网页检索

每一次检索都可能改变证据集合。搜索节点应保存查询、返回文档、文档版本、截取片段和支持的 claim。网页中的指令性文字要当作不可信数据处理，不能直接改写系统目标或权限。

### 12.4 工具调用安全

搜索可能探索删除、付款、发送消息等高风险动作。所有不可逆动作应先在模拟或事务草稿中执行；最终提交需要独立权限检查、业务规则和必要人工确认。搜索评分器只负责排序或发现异常，不负责授予权限。

## 13. 延迟、并行和缓存

### 13.1 节点并行不等于端到端并行

同一层的候选常常可以并行生成和评分，但工具调用可能有顺序依赖，某些环境还需要串行锁。应分别测模型计算时间、工具时间、队列等待和人工等待。

### 13.2 前缀缓存

多个节点共享同一个前缀时，可以缓存模型的前缀状态，减少重复计算。缓存键至少要考虑模型版本、输入、工具状态、系统提示、采样设置和权限上下文。

缓存错误会比没有缓存更危险：如果不同环境状态共享了同一缓存，搜索可能使用不存在的工具结果。缓存命中率和正确性要同时监控。

### 13.3 增量成本

一个新节点真正增加的成本不一定是完整上下文长度。若前缀已缓存，增量可能主要来自新 thought 和评分；若工具环境必须重新启动，增量成本又可能远高于 token 成本。

工程账本应记录平均节点成本、p95 延迟、缓存命中率、工具调用次数和每个成功任务的总耗时，而不是只记录平均输出 token。

## 14. Search 的评估

### 14.1 最终选择准确率

设搜索选出的答案为 `hat{y}_i^search`，参考答案为 `y_i^star`。当评估任务数 `N>0` 时，最终准确率为：

~~~math
A_{\mathrm{search}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[\hat y_i^{\mathrm{search}}=y_i^\star]
~~~

当 `N=0` 时，准确率没有定义；空评估集不能被报告为 0。它应和 greedy、self-consistency、beam、ToT 或 MCTS 在同一任务切分和相同预算口径下比较。

### 14.2 搜索增益

如果基线方法准确率为 `A_base`，搜索方法准确率为 `A_search`，绝对增益是：

~~~math
\Delta A=A_{\mathrm{search}}-A_{\mathrm{base}}
~~~

但绝对增益没有包含成本。还应报告每增加一个正确任务所消耗的 token、工具调用和延迟。

### 14.3 生成失败和选择失败要分开

如果生成器没有产生任何正确候选，搜索无法恢复；如果候选中有正确路径但评分器选错，则是选择失败。可以把任务分成三类：

- 没有正确候选；
- 有正确候选但被剪枝；
- 正确候选保留到终局但最终选择错误。

这种分解比单一最终准确率更能指导修复：第一类需要增强生成或动作空间，第二类需要调整剪枝和评分，第三类需要改进终局 verifier 或答案解析。

### 14.4 正确路径覆盖率

在知道候选中存在正确路径的任务上，保留到深度 `t` 的比例可以写成：

~~~math
C_{\mathrm{path}}(t)
=
\frac{
\#\{i:\text{至少一条正确路径在深度 }t\text{ 仍被保留}\}
}{
\#\{i:\text{生成了正确路径}\}
}
~~~

它能揭示“最终没选对”究竟发生在早期剪枝还是叶子重排。

### 14.5 候选多样性和搜索退化

报告答案去重率、状态去重率、不同错误类型覆盖和平均分支数。搜索宽度增加但有效多样性不变，说明系统可能只是重复采样；多样性增加但最终准确率不变，说明评分或终局检查没有利用这些候选。

### 14.6 成本和风险

高风险任务还要报告错误动作拦截率、误拒率、人工复核比例和权限违规尝试。普通数学 benchmark 上的搜索提升不能直接推出付款或删除任务安全。

## 15. 常见失败模式

### 15.1 分支爆炸

动作空间、深度和候选数同时增大时，节点数量呈指数增长。没有深度、节点、token、工具和延迟上限的搜索无法稳定部署。

### 15.2 评分器误导

评分器偏好长度、格式或自信语气时，搜索会把更多预算集中给错误但漂亮的路径。需要 hard negative、反事实改写和独立程序验证。

### 15.3 过早剪枝

正确路径在中间步骤可能暂时低分，尤其是需要先做看似无关的变量替换或检索。早期剪枝过强会使最终结果对 beam 宽度极其敏感。

### 15.4 候选高度相关

温度改变了表面采样，却没有改变错误前提。搜索树看起来很宽，实际只有一条错误思路。应按状态和方法去重，而不是只按字符串去重。

### 15.5 状态表示丢信息

如果状态只保存最后一句 thought，而不保存已确认条件、工具结果、文档版本和权限，评分器无法判断全局一致性，搜索也无法正确合并或回溯。

### 15.6 局部最优

best-first 或 beam 可能不断扩展短期高分节点，忽略需要更多步骤才能显现价值的路径。MCTS 的探索项、保留多样性和有限回溯可以缓解，但不能消除错误评分。

### 15.7 工具环境污染

一个分支修改了文件或数据库，另一个分支在污染后的状态上运行，所有分数都失去可比性。搜索环境必须隔离、可恢复或使用事务。

### 15.8 终局整合错误

多个分支分别找到了正确片段，最后合并时却混用了不同版本、单位或实体。终局答案仍要经过结构化检查和全局约束验证。

### 15.9 成本收益倒挂

搜索提高了低频难题的准确率，却让所有简单任务都付出同样的高成本。可以做风险路由：简单任务先用单链，只有不确定或高价值任务才触发搜索。

## 16. 综合案例：合同审阅中的证据路径搜索

### 16.1 状态定义

任务是确认合同中的基础金额、折扣、税率和付款状态。一个搜索状态可以包含：

~~~text
已确认字段：base_amount, discount
未确认字段：tax_rate, payment_status
证据：page_3, page_4
当前计算：95000 元
风险：税率来自 OCR，尚未与正文核对
剩余预算：检索 2 次、计算 1 次、人工复核 1 次
~~~

这比只保存“我已经算到 95000 元”更完整，因为下一步动作取决于未确认字段和证据风险。

### 16.2 候选动作

系统可能提出：

- 检索合同中所有“税率”条款；
- 打开第 7 页并核对 OCR 文本；
- 用原文数字重新计算；
- 检查付款状态字段是否来自最新合同版本；
- 请求人工确认，而不是继续猜测。

这些动作的价值不同。一个动作可能暂时不产生最终答案，却能降低关键不确定性。评分器不能只看“离答案还有多近”，还要看它是否覆盖了未解决约束。

### 16.3 终局选择

假设两个候选最终都给出 `100700` 元：

~~~text
候选 A：税率来自第 7 页正文，页码和合同版本匹配，状态为 needs_review。
候选 B：税率来自 OCR 页脚，金额相同，但没有确认版本，状态为 confirmed。
~~~

如果只比较字符串答案，两者一样；如果任务要求证据支持和正确状态，A 更可靠。搜索的最终 verifier 应检查字段、证据和状态，而不是只比较金额。

### 16.4 不可逆动作的边界

如果下一步是创建付款指令，搜索只能生成草稿和模拟结果。真实提交必须离开语言搜索，进入独立的权限和审批流程。搜索树可以帮助找到正确金额和证据路径，但不能把“高分节点”直接转换成生产动作。

## 17. 最小可运行实验：Beam 与 MCTS 的差异

下面的 demo 使用四个 toy 任务比较 greedy、按 verifier 分数选择的 beam 和带 UCT 探索项的选择。它不是 MCTS 的完整实现，而是把每个任务的候选分支看成已经展开的一层，用来展示：

- 搜索可能比 greedy 找到更多正确答案；
- 高分 hard negative 会让 beam 选错；
- 探索项可能选择价值更高但语言分数较低的候选；
- 需要同时报告正确路径被剪掉、候选多样性和 token 成本。

~~~python
from math import isfinite, log, sqrt


cases = [
    {
        "id": "water_jug",
        "gold": "4",
        "greedy": "4",
        "branches": [
            {"name": "wrong_volume", "answer": "5", "score": 0.45, "value": 0.20, "visits": 7, "tokens": 70, "correct": False},
            {"name": "state_search", "answer": "4", "score": 0.82, "value": 0.92, "visits": 6, "tokens": 110, "correct": True},
            {"name": "irrelevant_fact", "answer": "5", "score": 0.18, "value": 0.10, "visits": 3, "tokens": 40, "correct": False},
        ],
    },
    {
        "id": "logic_grid",
        "gold": "blue",
        "greedy": "red",
        "branches": [
            {"name": "row_elimination", "answer": "blue", "score": 0.79, "value": 0.86, "visits": 5, "tokens": 80, "correct": True},
            {"name": "column_guess", "answer": "blue", "score": 0.61, "value": 0.55, "visits": 4, "tokens": 95, "correct": True},
            {"name": "shortcut_guess", "answer": "red", "score": 0.35, "value": 0.22, "visits": 8, "tokens": 50, "correct": False},
        ],
    },
    {
        "id": "code_patch",
        "gold": "pass",
        "greedy": "fail",
        "branches": [
            {"name": "patch_edge_case", "answer": "pass", "score": 0.74, "value": 0.88, "visits": 7, "tokens": 90, "correct": True},
            {"name": "rewrite_all", "answer": "fail", "score": 0.41, "value": 0.35, "visits": 5, "tokens": 125, "correct": False},
            {"name": "add_sleep", "answer": "timeout", "score": 0.12, "value": 0.05, "visits": 2, "tokens": 60, "correct": False},
        ],
    },
    {
        "id": "hard_negative",
        "gold": "40",
        "greedy": "42",
        "branches": [
            {"name": "polished_wrong", "answer": "42", "score": 0.88, "value": 0.38, "visits": 10, "tokens": 100, "correct": False},
            {"name": "plain_correct", "answer": "40", "score": 0.62, "value": 0.78, "visits": 3, "tokens": 85, "correct": True},
            {"name": "unit_checked", "answer": "40", "score": 0.59, "value": 0.71, "visits": 2, "tokens": 75, "correct": True},
        ],
    },
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )


def validate_cases(items):
    require(isinstance(items, list) and items, "cases must be a non-empty list")
    case_ids = []
    for case in items:
        require(isinstance(case, dict), "each case must be a dictionary")
        require(isinstance(case.get("id"), str) and case["id"], "case id must be non-empty")
        require(case["id"] not in case_ids, "case ids must be unique")
        case_ids.append(case["id"])
        require(isinstance(case.get("gold"), str) and case["gold"], "gold must be non-empty")
        require(isinstance(case.get("greedy"), str) and case["greedy"], "greedy answer must be non-empty")
        branches = case.get("branches")
        require(isinstance(branches, list) and branches, "branches must be non-empty")
        branch_names = []
        for branch in branches:
            require(isinstance(branch, dict), "each branch must be a dictionary")
            require(isinstance(branch.get("name"), str) and branch["name"], "branch name must be non-empty")
            require(branch["name"] not in branch_names, "branch names must be unique per case")
            branch_names.append(branch["name"])
            require(isinstance(branch.get("answer"), str) and branch["answer"], "branch answer must be non-empty")
            require(finite_number(branch.get("score")) and 0 <= branch["score"] <= 1, "score must be finite and in [0, 1]")
            require(finite_number(branch.get("value")) and 0 <= branch["value"] <= 1, "value must be finite and in [0, 1]")
            require(type(branch.get("visits")) is int and branch["visits"] >= 0, "visits must be a non-negative integer")
            require(type(branch.get("tokens")) is int and branch["tokens"] >= 0, "tokens must be a non-negative integer")
            require(type(branch.get("correct")) is bool, "correct must be boolean")
            require(branch["correct"] == (branch["answer"] == case["gold"]), "correct must match gold")


validate_cases(cases)


def uct_score(branch, parent_visits, exploration=0.45):
    require(type(parent_visits) is int and parent_visits >= 0, "parent_visits must be a non-negative integer")
    require(finite_number(exploration) and exploration >= 0, "exploration must be finite and non-negative")
    bonus = exploration * sqrt(log(parent_visits + 1) / (branch["visits"] + 1))
    return round(branch["value"] + bonus, 3)


def accuracy(flags):
    require(isinstance(flags, list) and flags, "accuracy requires a non-empty list")
    require(all(type(flag) is bool for flag in flags), "accuracy flags must be boolean")
    return round(sum(flags) / len(flags), 3)


def unit_cost(total_cost, successes):
    require(finite_number(total_cost) and total_cost >= 0, "total cost must be finite and non-negative")
    require(type(successes) is int and successes >= 0, "successes must be a non-negative integer")
    if successes == 0:
        return None
    return round(total_cost / successes, 3)


beam_choices = []
mcts_choices = []
pruned_correct_paths = []
unique_ratios = []

for case in cases:
    parent_visits = sum(branch["visits"] for branch in case["branches"])
    beam_choice = sorted(
        case["branches"],
        key=lambda branch: (-branch["score"], branch["name"]),
    )[0]
    mcts_choice = sorted(
        case["branches"],
        key=lambda branch: (-uct_score(branch, parent_visits), branch["name"]),
    )[0]
    beam_choices.append(
        {"id": case["id"], "choice": beam_choice["name"], "correct": beam_choice["correct"]}
    )
    mcts_choices.append(
        {"id": case["id"], "choice": mcts_choice["name"], "correct": mcts_choice["correct"]}
    )
    if (not beam_choice["correct"]) and any(
        branch["correct"] for branch in case["branches"]
    ):
        pruned_correct_paths.append(case["id"])
    unique_ratios.append(
        len({branch["answer"] for branch in case["branches"]})
        / len(case["branches"])
    )

greedy_accuracy = accuracy(
    [case["greedy"] == case["gold"] for case in cases]
)
beam_accuracy = accuracy([choice["correct"] for choice in beam_choices])
mcts_accuracy = accuracy([choice["correct"] for choice in mcts_choices])
avg_unique_answer_ratio = round(sum(unique_ratios) / len(unique_ratios), 3)
total_nodes_expanded = sum(1 + len(case["branches"]) for case in cases)
total_tokens = sum(
    branch["tokens"] for case in cases for branch in case["branches"]
)
cost_per_beam_correct = unit_cost(
    total_tokens,
    sum(choice["correct"] for choice in beam_choices),
)
mcts_rescues = [
    beam["id"]
    for beam, mcts in zip(beam_choices, mcts_choices)
    if (not beam["correct"]) and mcts["correct"]
]

review = {
    "beam_accuracy_ok": beam_accuracy >= 0.75,
    "mcts_accuracy_ok": mcts_accuracy >= 0.9,
    "diversity_ok": avg_unique_answer_ratio >= 0.7,
    "budget_ok": total_tokens <= 1000,
    "pruned_correct_paths": bool(pruned_correct_paths),
}

print(f"greedy_accuracy={greedy_accuracy}")
print(f"beam_accuracy={beam_accuracy}")
print(f"mcts_accuracy={mcts_accuracy}")
print(f"avg_unique_answer_ratio={avg_unique_answer_ratio}")
print(f"total_nodes_expanded={total_nodes_expanded}")
print(f"total_tokens={total_tokens}")
print(f"cost_per_beam_correct={cost_per_beam_correct}")
print(f"beam_choices={beam_choices}")
print(f"mcts_choices={mcts_choices}")
print(f"pruned_correct_paths={pruned_correct_paths}")
print(f"mcts_rescues={mcts_rescues}")
print(f"review={review}")
~~~

预期输出：

~~~text
greedy_accuracy=0.25
beam_accuracy=0.75
mcts_accuracy=1.0
avg_unique_answer_ratio=0.75
total_nodes_expanded=16
total_tokens=980
cost_per_beam_correct=326.667
beam_choices=[{'id': 'water_jug', 'choice': 'state_search', 'correct': True}, {'id': 'logic_grid', 'choice': 'row_elimination', 'correct': True}, {'id': 'code_patch', 'choice': 'patch_edge_case', 'correct': True}, {'id': 'hard_negative', 'choice': 'polished_wrong', 'correct': False}]
mcts_choices=[{'id': 'water_jug', 'choice': 'state_search', 'correct': True}, {'id': 'logic_grid', 'choice': 'row_elimination', 'correct': True}, {'id': 'code_patch', 'choice': 'patch_edge_case', 'correct': True}, {'id': 'hard_negative', 'choice': 'plain_correct', 'correct': True}]
pruned_correct_paths=['hard_negative']
mcts_rescues=['hard_negative']
review={'beam_accuracy_ok': True, 'mcts_accuracy_ok': True, 'diversity_ok': True, 'budget_ok': True, 'pruned_correct_paths': True}
~~~

这个实验要这样读。Greedy 只看每题的第一个答案，在四题中只对一题；beam 使用语言分数后找回了三题，但在 `hard_negative` 上选择了高分错误候选。MCTS 风格的 UCT 分数使用了另一套价值和探索项，选择了 `plain_correct`，因此在这个构造中达到 1.0。

`pruned_correct_paths=['hard_negative']` 说明 beam 的错误不是“候选不存在”，而是正确候选存在却被更高的表面分数压过。`total_tokens=980` 是候选生成账本，不包括实际模型硬件、工具执行、缓存和人工等待。`review` 只是实验报告中的观察字段，用来把结果拆开，不是一个神奇的最终结论。

## 18. 设计一次可复现的搜索实验

### 18.1 固定候选生成器

比较 beam 和 MCTS 时，要记录同一模型、同一 prompt、相同动作候选或相同生成预算。否则一个方法获得更好的候选集合，另一个方法获得更差的候选集合，最终差异无法归因于搜索策略。

### 18.2 分离生成、评分和选择

一次搜索失败可能来自三个地方：生成器没有产生正确动作，评分器给错分，选择策略在分数正确时仍然做错决策。实验应保存候选全集、每个评分、每次剪枝和最终选择，不能只保存最后答案。

### 18.3 对照不同预算

至少比较单链、采样多个完整答案、beam、ToT 或 best-first，以及 MCTS 风格搜索。预算可以按 token、模型调用、工具调用、延迟或金钱计量。相同节点数量不一定代表相同成本，因为上下文长度和工具耗时不同。

### 18.4 做剪枝反事实

对已经被剪掉的节点继续离线扩展，检查是否存在正确终局；对高分节点做独立程序验证；把正确路径改写成不同风格，检查评分器是否仍能保留。这样才能估计误剪率和评分器的表面偏差。

### 18.5 分层评估

按题目难度、路径长度、错误类型、工具类型、风险等级和生成器版本分层。搜索往往对简单题没有收益，却对少数需要回溯的难题有明显帮助；只报告平均准确率会掩盖这种差异。

## 19. 安全、权限与可回滚性

搜索在模拟推理中可以探索任意动作，但真实工具执行需要独立权限。搜索节点的高分不应直接变成生产提交。

运行代码时使用沙箱，限制文件、网络、进程、CPU、内存、时间和输出；访问网页时隔离不可信内容和系统指令；修改数据库时使用事务或快照；发送外部消息时保留草稿和人工确认。

环境状态必须可回滚。没有回滚能力的搜索不能安全地尝试不可逆动作，因为失败分支会污染后续分支，最终也无法解释哪条路径造成了副作用。

日志应保存动作、观察、评分、权限检查和失败原因，但要遵守隐私和数据保留要求。搜索日志可能包含合同、代码和个人信息，不能因为“方便调试”就无限期保留。

## 20. 小练习

### 练习一：定义搜索状态

为一个需要检索合同条款并计算金额的任务设计 `state`、`action`、`transition`、`termination`。说明哪些信息必须进入状态键。

### 练习二：估算树大小

每层平均分支数为 3，最大深度为 5，计算不剪枝节点数。再假设每层保留 4 个节点，估算 beam 的节点上限。

### 练习三：Beam 的剪枝错误

构造一个三层树，让正确路径在第二层的语言分数低于错误路径，但最终结果更好。计算正确路径被剪掉后的选择准确率。

### 练习四：UCT

给定两个子节点的 `Q`、`N_v` 和父节点访问次数，分别计算不同探索系数下的 UCT 分数，说明探索项什么时候会改变选择。

### 练习五：状态合并

写出两个不同文本轨迹，它们到达相同业务状态；再写出两个文本相同但权限状态不同的节点，说明为什么不能简单去重。

### 练习六：工具环境

为代码修复、网页检索和数据库更新分别列出环境快照、回滚和安全限制。

### 练习七：Search 与 self-consistency

说明一个任务为什么适合独立生成多个完整答案，却不适合中间剪枝；再说明另一个任务为什么需要搜索和工具反馈。

### 练习八：运行 demo

把 `hard_negative` 的 `plain_correct` 分数提高到 0.90，观察 beam 准确率、剪枝路径和成本解释如何改变。再把所有分支答案改成同一个答案，解释有效多样性指标为何下降。

## 21. 本章总结

搜索推理把生成过程从单条链扩展为状态空间中的多条候选路径。它的核心不是树形图本身，而是五个可审计对象：状态、动作、转移、评分和停止条件。

Tree-of-Thought 提供了把 thought 作为中间节点的框架；beam search 按层保留有限候选，简单且易控制；best-first 和回溯可以优先探索高价值节点；MCTS 通过访问次数、价值和探索项平衡未知分支与已知好分支；带工具的代理搜索还必须管理环境状态、权限和回滚。

搜索的主要风险是分支爆炸、评分偏差、候选相关、状态丢信息和正确路径被过早剪掉。评价时要把“没有生成正确候选”“生成了但被剪掉”“保留了却最终选错”分开统计。

最终决策不能只看准确率。节点数、token、工具调用、p95 延迟、单位成功成本、候选多样性、剪枝误报和高风险动作拦截率都属于搜索系统的真实行为。

下一章将讨论 test-time compute scaling：当推理阶段可以继续增加采样、验证、搜索或思考预算时，质量曲线、成本曲线和动态分配应该如何分析。

## 22. 资料索引

1. Tree of Thoughts：<https://arxiv.org/abs/2305.10601>
2. Language Agent Tree Search：<https://arxiv.org/abs/2310.04406>
3. Mastering the Game of Go without Human Knowledge：<https://arxiv.org/abs/1712.01815>
4. Monte Carlo Tree Search: A Review：<https://arxiv.org/abs/2007.14554>
5. Self-Consistency Improves Chain of Thought Reasoning in Language Models：<https://arxiv.org/abs/2203.11171>
6. Let's Verify Step by Step：<https://arxiv.org/abs/2305.20050>

本章写作时联网访问了 ToT、Language Agent Tree Search、AlphaZero、MCTS review、Self-Consistency 和过程监督论文入口。正文区分经典搜索算法定义、论文实验、工具环境抽象、教学构造和目标系统实测；没有把论文中的搜索结果外推为所有模型或所有任务的通用保证。
