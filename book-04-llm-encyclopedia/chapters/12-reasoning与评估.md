# 第 12 章 Reasoning 与评估：从候选生成到可信结论

“模型会推理”不是一个可以只用回答长度或一个 benchmark 分数证明的属性。对简单问题，直接生成答案可能已经足够；对数学、代码、规划和工具任务，系统往往需要保留中间状态，生成多个候选，执行程序或检索证据，发现错误后回退，再用统计方法判断改进是否稳定。推理能力、验证器、搜索策略、测试时计算和评估方法因此构成一条系统链路，而不是几个可以孤立背诵的术语。

初学者可以先把 reasoning 理解为“把一个难题拆成若干可检查的步骤，并在得到反馈后继续或修正”。专家还需要追问：中间步骤是否真的影响了结论，验证器是否被长度和格式欺骗，额外计算是否换来了足够的正确率，评估集是否被污染，线上高风险切片是否退化。下面的公式会说明变量和分母，代码示例只验证局部机制，论文结果会保留其任务和模型范围。

~~~text
一个可验证的推理请求：

问题与约束
  -> 生成一个或多个候选
  -> 解析最终答案和中间状态
  -> 规则、程序、工具或模型验证
  -> 选择、回退、修复或继续搜索
  -> 统计正确率、成本、延迟和安全事件
~~~

## 12.1 推理对象：答案、步骤与状态

### 直接生成和多步推理

语言模型本质上按条件概率生成序列。给定输入 x，一个输出 y=(y_1,...,y_T) 的概率可以写成：

~~~math
P(y\mid x)
=\prod_{t=1}^{T}P(y_t\mid x,y_{<t}).
~~~

直接回答时，模型主要把计算压缩在一次生成轨迹中；多步推理则显式或隐式地保留中间状态 z：

~~~math
P(y\mid x)
=\sum_z P(y\mid x,z)P(z\mid x).
~~~

这个分解是帮助理解的概率视角，并不意味着真实模型一定按一个可读的 z 变量运行。它说明了为什么多条候选、工具反馈或外部搜索可能有帮助：系统不再只依赖一条生成路径，而是给候选和验证分配额外计算。

推理并不等于输出长解释。一个短答案可能经过可靠的程序验证，一个长答案也可能是先猜结论再补充貌似合理的步骤。产品应该根据任务选择输出“最终答案 + 关键依据 + 可复核结果”，而不是把所有草稿都展示给用户。

### 候选集合

对同一道问题，系统可以通过不同随机种子、温度、搜索分支或工具观察生成候选集合：

~~~math
\mathcal{Y}(x)=\{(z^{(1)},y^{(1)}),\ldots,(z^{(K)},y^{(K)})\}.
~~~

z^(k) 是第 k 条候选的中间状态，y^(k) 是最终答案。候选集合的价值取决于四个因素：至少有一条候选正确的概率、候选之间的有效差异、选择器识别正确候选的能力，以及生成和验证成本。只增加 K 而不改变这些因素，可能只是把同一个错误重复很多次。

### 推理状态

在工具或搜索环境中，状态不应只有自然语言历史。它至少应包含原问题、已满足约束、已执行动作、观察结果、未解决子目标、错误、剩余预算和当前候选。一个抽象状态转移为：

~~~math
s_{t+1}=T(s_t,a_t,o_{t+1}),
~~~

其中 s_t 是当前状态，a_t 是下一动作，o_(t+1) 是环境或工具观察，T 是状态更新函数。状态记录的作用是让系统知道动作是否已经发生、结果是否可信、下一步是否重复，以及最终结论是否满足原始约束。

## 12.2 Chain-of-Thought、Scratchpad 与可见解释

### CoT 的作用

Chain-of-Thought（CoT）提示通过示例或指令让模型生成中间推理步骤。Wei 等人的 [CoT prompting 论文](https://arxiv.org/abs/2201.11903) 在多步算术、常识和符号推理任务上展示了这种提示方式的实验收益。该结果支持“在某些任务和模型上，中间步骤能提供额外计算与结构”，不支持“所有可见推理都忠实呈现内部过程”。

少样本 CoT 将带步骤的示例放入上下文；零样本 CoT 只改变提示方式。两者都可能受到示例质量、题目格式、模型版本和上下文预算影响。示例中的跳步、错误或错误的答案格式会被模型复制，因此 CoT 示例也需要像训练数据一样审核。

### Scratchpad 与 Hidden CoT

Scratchpad 更强调中间工作区，可以包含草稿、表格、代码计划和工具调用计划；它不一定是面向用户的解释。某些系统把完整推理保留在内部，只向用户展示答案、证据和简短理由。这样做可以减少敏感策略、错误探索和不必要细节泄露，但也要求系统保留独立的审计字段和可验证结果。

不要用“解释很流畅”替代过程可信度。过程审计可以进行干预测试：修改某一步、遮蔽某个证据、替换工具结果，再观察最终答案是否按预期变化。即使干预结果看起来合理，也只能说明在该测试上的因果敏感性，不能证明所有内部状态都可解释。

### CoT 回归和伪推理
回归率的分母 N 必须是非空、同时运行了两种配置并有参考答案的样本数；缺失任一结果的样本应单独报告，不能混入分母。

有些问题在直接回答配置下正确，加入长推理后反而错误。设样本数为 N，直答为 y_direct，CoT 配置为 y_cot，参考答案为 y*，可以定义回归率：

~~~math
R_{\mathrm{cot}}
=\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[y_i^{\mathrm{direct}}=y_i^\star
\land
y_i^{\mathrm{cot}}\ne y_i^\star].
~~~

常见原因包括简单题过度推理、把题外条件引入推导、早期猜测被后续文字合理化，以及长度增加导致关键约束被忽略。评估 CoT 时要同时报告复杂任务增益和简单任务回归，而不是只报总体平均准确率。

## 12.3 采样与 Self-Consistency

### 温度、Top-p 与候选多样性
这里要求温度 tau 大于 0，Top-p 的 p 位于 (0,1]。温度和 p 都应是有限数；非法值应在请求层拒绝，而不是让 softmax 产生 NaN。

模型输出 logits 为 l_i 时，温度 tau 后的概率为：

~~~math
p_i(\tau)
=\frac{\exp(l_i/\tau)}
{\sum_j\exp(l_j/\tau)}.
~~~

tau 越小，分布越尖锐，候选更接近贪心解码；tau 越大，尾部 token 获得更多机会。Top-p sampling 再从累计概率达到 p 的最小候选集采样。推理候选需要有差异，但过高温度会产生大量不可验证或偏离题意的路径，因此应在目标任务上测量候选质量和有效多样性。

### 答案标准化

Self-consistency 常先采样多条推理路径，再抽取和聚合最终答案。答案标准化必须处理空格、大小写、单位、数值表示、选项标签和等价表达。例如 42 与 42.0 可能等价，但 42 minutes 是否等价取决于题目单位。标准化器的错误会把同一正确答案拆成多个桶，也可能错误合并不同答案。

### 多数投票

令第 i 个问题的标准化候选答案为 a_i^(1),...,a_i^(K)，多数投票选择出现次数最多的类别：

~~~math
\hat a_i
=\operatorname*{arg\,max}_{a}
\sum_{k=1}^{K}\mathbf{1}[a_i^{(k)}=a].
~~~

Self-consistency 论文 [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) 研究了通过多条采样路径聚合答案的思路。多数投票在候选错误相互独立、正确答案有较大概率出现时更有帮助；如果模型系统性误读题意，多个候选会一致错。因而应同时报告单次准确率、候选中至少一条正确的比例、聚合后准确率和平均候选成本。

~~~python
from collections import Counter


def majority_vote(candidates):
    if not candidates:
        raise ValueError("candidates must not be empty")
    counts = Counter(candidates)
    winner, votes = counts.most_common(1)[0]
    return winner, votes / len(candidates)


assert majority_vote(["42", "42.0", "42"])[0] == "42"
assert majority_vote(["a", "b", "b"])[1] == 2 / 3
print(majority_vote(["42", "42", "41"]))
~~~

这个示例把 42.0 当成不同字符串，说明标准化应在投票前完成；代码只展示聚合，不判断答案是否真实正确。

### 加权投票和失败边界

如果每条候选有 verifier 分数 s_k，加权聚合可以写成：

~~~math
\hat a
=\operatorname*{arg\,max}_{a}
\sum_{k:a_k=a}w(s_k).
~~~

权重函数 w 可以由规则验证、程序测试、log probability 或校准后的模型分数给出。评分器偏差会直接改变聚合结果；长度、格式和自信语气不应在没有校准时被当成正确性证据。

## 12.4 Verifier：把生成和判断拆开

### 三类验证器

程序验证器执行代码、数学计算、schema、单元测试或形式规则，优点是确定性和可复现；学习型验证器使用分类器、Reward Model 或 LLM judge，适合开放式质量判断；混合验证器先使用硬约束，再将不能自动判定的候选交给模型或人工。

验证器的输入必须和任务输出对齐。代码验证器应执行补丁后的完整测试并限制网络、文件和时间；数学验证器应检查答案等价而不是只匹配字符串；RAG 验证器应检查 claim 与 evidence span 的绑定；工具验证器还要检查权限和副作用状态。一个“通过”字符串不能替代这些语义检查。

### 候选重排

给定问题 i 的候选分数 s_i1,...,s_iK，验证器重排选择：

~~~math
\hat y_i=y_{ij^\star},
\qquad
j^\star=\operatorname*{arg\,max}_{j}s_{ij}.
~~~

重排效果由生成候选覆盖和验证器排序共同决定。若正确答案没有生成，验证器无法凭空创造它；若 hard negative 得分过高，增加候选反而可能增加错误完成。

### Pairwise Accuracy 与选择准确率
这里要求候选对集合 P 非空，并预先约定平局如何计分。相对排序指标不能在没有有效候选对时输出 0 来冒充评估结果。

Pairwise accuracy 测量验证器是否给更优候选更高分：

~~~math
A_{\mathrm{pair}}
=\frac{1}{|\mathcal{P}|}
\sum_{(a,b)\in\mathcal{P}}\mathbf{1}[s_a>s_b].
~~~

它衡量相对排序，不等于从一组候选中选中正确答案的 top-1 accuracy。评估器还应报告 hard negative、长度切片、格式切片和真正的下游任务成功率。

### 校准
ECE 和 Brier score 都要求 N>0；分桶应覆盖每个样本且互不重叠。没有校准样本时应报告 N/A，而不是把空集合当作完美校准。

如果验证器输出 p_i，系统可能用阈值决定自动通过、拒答或转人工。把预测分数分成 B 个桶，ECE 可写成：

~~~math
\operatorname{ECE}
=\sum_{b=1}^{B}\frac{|S_b|}{N}
\left|\operatorname{Acc}(S_b)-\operatorname{Conf}(S_b)\right|.
~~~

S_b 是第 b 个分桶，Acc 是实际正确率，Conf 是平均预测置信度。Brier score 为：

~~~math
\operatorname{BS}
=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2,
~~~

其中 y_i 是真实标签。Guo 等人的 [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) 讨论了现代神经网络的校准和温度缩放；在 Agent 或高风险推理中，校准集必须独立于验证器训练集，并按任务和风险切片检查。

### Reward Model Bias 与 Hard Negative

验证器可能偏好更长、更有条理、更自信或更像训练集的回答，而不是实际正确的回答。Hard negative 应保留正确的格式和风格，只改变关键事实、步骤或约束，以检验验证器是否关注真正信号。验证器训练和评估不能只使用容易区分的负例，否则线上候选分布一变化，分数就会失效。

## 12.5 过程监督：PRM、ORM 与第一处错误

### Outcome supervision 与 process supervision

Outcome Reward Model（ORM）只看最终答案，标注便宜，适合有明确终点的任务；Process Reward Model（PRM）对中间步骤打分，能提供更密集的训练和搜索信号，但需要定义步骤边界、标签协议和不同解法之间的可比性。

过程监督不能自动带来完整可解释性。一个步骤可能形式上正确但与目标无关，也可能最终答案正确但中间过程不可靠。评估必须同时保留最终答案标签、步骤标签、第一处错误、验证器结果和标注来源。

### 步骤标签与第一处错误
如果所有步骤都正确，第一处错误位置应记录为不存在；如果步骤标签缺失或步骤数为零，应从步骤准确率分母中排除并报告覆盖率。

设第 i 个样本被切成 M_i 个步骤，q_ij=1 表示第 j 步正确，第一处错误位置为：

~~~math
j_i^{\mathrm{err}}
=\min\{j:q_{ij}=0\}.
~~~

若所有步骤正确，可以将 j_i^err 标记为不存在。第一处错误比平均步骤分数更有诊断价值：后续步骤可能只是错误前提的连锁反应，修复或搜索回退应优先发生在第一处错误处。

步骤准确率为：

~~~math
A_{\mathrm{step}}
=\frac{\sum_i\sum_jq_{ij}}{\sum_iM_i}.
~~~

这个指标受切分粒度影响。把一个推导拆成五个小步骤会改变分母，因此报告时必须保留步骤切分规则和标注者一致性。

### 自动标注与人工成本

若 a_ij=1 表示某一步可由规则、计算器、测试或工具自动标注，自动覆盖率为：

~~~math
C_{\mathrm{auto}}
=\frac{\sum_i\sum_ja_{ij}}{\sum_iM_i}.
~~~

人工成本可以粗略写成：

~~~math
C_{\mathrm{label}}
=\sum_i\sum_j(1-a_{ij})c_{ij},
~~~

其中 c_ij 是该步骤人工切分、判断、复核和仲裁的成本。提高自动覆盖率不等于提高标签质量，自动规则仍需要抽样审计。

### PRM 剪枝

在逐步搜索中，系统可以根据累计过程分数保留前 K 条路径：

~~~math
B_{t+1}
=\operatorname{TopK}
\left(
\{b+a:b\in B_t,a\in\mathcal{A}(b)\},
S_{\mathrm{process}}
\right).
~~~

剪枝节省计算，但 PRM 低估正确路径或高估 polished hard negative 时，会把真正可行的答案提前删除。应记录被剪掉的路径、剪枝时分数和后来可验证的结果，估计 false-negative pruning rate。

## 12.6 Search Reasoning：Beam、Best-First、ToT 与 MCTS

### 状态空间搜索

搜索推理把中间状态视作节点，把下一步推导、代码修改、子问题选择或工具调用视作动作。若状态为 s_t，动作集合为 A(s_t)，则：

~~~math
a_t\in\mathcal{A}(s_t),
\qquad
s_{t+1}=T(s_t,a_t).
~~~

动作粒度过细会导致树迅速膨胀，粒度过粗又难以回退和定位错误。状态应包含搜索所需的约束、工具观察、测试结果和预算，而非只保存最终文本。

### Beam Search

Beam search 按层展开候选，每层保留分数最高的 K 个状态：

~~~math
F_{t+1}
=\operatorname{TopK}\left(
\{T(s,a):s\in F_t,a\in\mathcal{A}(s)\},
S,K
\right).
~~~

它实现简单、预算可控，适合结合 PRM 或 verifier；问题是早期剪枝。如果正确路径前几步的局部得分较低，它可能在还没有出现关键信息前就被删除。

### Best-First Search

Best-first search 在全局 frontier 中取当前分数最高的节点，而不是严格按深度推进。它可能快速深入一个高分分支，也可能因评分器偏差长期探索局部最优。需要设置最大深度、节点数、时间和回退条件。

### Tree of Thoughts 与 MCTS
对已访问节点要求访问次数和父节点访问次数均为正，探索常数非负；未访问节点必须走单独的扩展分支，不能直接计算 UCT。

Tree of Thoughts 论文 [arXiv:2305.10601](https://arxiv.org/abs/2305.10601) 将中间想法组织成可搜索的树，并通过评估与回溯探索多条路径。MCTS 则在选择、扩展、模拟和回传之间反复分配预算。UCT 常用的选择分数可写成：

~~~math
\operatorname{UCT}(v)
=\frac{W_v}{N_v}
+c\sqrt{\frac{\ln N_{\mathrm{parent}}}{N_v}},
~~~

W_v 是节点累计价值，N_v 是访问次数，N_parent 是父节点访问次数，c 控制探索强度。未访问节点需要单独处理，不能直接代入分母。MCTS 并不自动解决评分器偏差、动作生成质量和环境模拟不准确的问题。

### 搜索预算

搜索至少需要限制节点数、深度、候选宽度、verifier 调用次数、工具调用次数、token 和墙钟时间。完整 trace 要记录哪些路径被展开、哪些被剪掉、停止原因和最后的验证结果。否则出现错误时无法判断是生成、评分、剪枝还是预算不足。

## 12.7 Test-Time Compute：把额外计算投到正确位置

### 预算向量

测试时计算（TTC）包括多次采样、候选重排、搜索、代码执行、工具调用和迭代修复。可以用预算向量表示一次请求的资源：

~~~math
b=(K,L,D,V,U,T),
~~~

其中 K 是候选数，L 是单候选长度，D 是搜索深度，V 是 verifier 调用次数，U 是工具调用次数，T 是 token 或总计算预算。这样比笼统说“让模型多思考”更容易做成本归因。

### 自适应路由

不同请求的难度、业务价值和可验证性不同。预算路由器可以根据输入特征、初步置信度和任务类型选择 direct、self-consistency、verifier、search 或 tool loop。路由策略必须可记录、可回放，并在灰度中观察错误路由。

### 单位正确成本与边际收益
没有正确样本时，单位正确成本没有定义，应写成 N/A。边际收益率还要求两种预算的成本差严格为正；成本相同或更低时，应分别记录为不可比较或成本回归。

设总成本为 C_total，正确样本数为 N_correct，单位正确成本为：

~~~math
C_{\mathrm{correct}}
=\frac{C_{\mathrm{total}}}{N_{\mathrm{correct}}}.
~~~

从预算 B_1 增加到 B_2 的边际正确率收益与成本可以写成：

~~~math
G(B_1,B_2)
=\frac{A(B_2)-A(B_1)}{C(B_2)-C(B_1)}.
~~~

高预算策略可能提高准确率，却让 P95、token 成本或工具费用翻倍。应画出预算—准确率—成本曲线，并单独统计高预算仍失败、低价值请求误用高预算的浪费。

### 延迟和停止

多候选可以并行，搜索和工具链常有串行关键路径。线上至少记录 P50、P95、超时率、平均调用数和单位成功任务成本。停止条件可以来自答案已通过程序验证、候选差异已经足够小、剩余预算不足、达到截止时间或发现不可修复的权限/安全问题。

## 12.8 数学推理训练：答案监督、合成数据与污染

### 数据对象

一条数学训练样本不应只有题面和答案，还可以包含步骤、难度、题型、来源、验证程序、模板指纹和污染风险：

~~~math
m_i=(x_i,y_i^\star,z_i,q_i,d_i,t_i,s_i,r_i).
~~~

x_i 是题目，y_i* 是参考答案，z_i 是步骤，q_i 是步骤标签，d_i 是难度，t_i 是题型，s_i 是来源或模板指纹，r_i 是风险标记。答案可由程序检查时，自动验证应优先于语言模型自评。

### Answer、Process 与验证训练

只监督最终答案成本低，但不能告诉模型哪一步错；过程监督提供更密集信号，却需要可靠步骤标签。数学训练通常还会加入错误解法、反例、不同解法和验证器反馈，避免模型只学固定模板。

### 合成数据

合成题可以扩展难度和题型覆盖，但生成器错误、模板重复、解答污染和难度估计偏差会被放大。合成数据应记录生成模型、提示、验证器、拒弃率和去重指纹，并保留人工抽样。通过验证器的合成答案仍可能存在题意歧义或测试覆盖不足。

### 课程与模板多样性

课程学习可以按运算、方程、证明、组合、几何和多步混合逐渐增加难度；模板多样性用于防止模型把固定句式当成解题规则。难度不能只由生成器自报，应结合独立模型、程序验证和人工样本估计。

### 污染审计

Benchmark contamination 指评估题或近重复题泄漏到训练数据，导致分数高估。可以结合 exact match、n-gram、embedding 近重复、时间切分、来源核查和答案异常模式检查。公开题库上的提升，如果在干净、时间隔离和变体集合上消失，结论应降级为记忆或格式适配证据。

## 12.9 Code Reasoning：执行反馈和隐藏测试

### 代码推理闭环

代码 Agent 的推理不能只看生成文本。更完整的链路是：理解需求、定位接口、生成补丁、运行公开测试、读取错误、修复、运行隐藏测试并检查安全边界。执行反馈是环境观察，不等于自然语言提示；它必须被结构化为退出码、测试结果、日志摘要和资源状态。

### pass@k 与 pass@1
该估计要求 n 为正整数，k 为整数且 0≤k≤n；当没有候选或 k 超出候选数时，结果不应静默补默认值。通过测试的候选数 c 也必须满足 0≤c≤n。

HumanEval 论文 [arXiv:2107.03374](https://arxiv.org/abs/2107.03374) 讨论了代码生成评估中的 pass@k。若生成 n 个候选，其中 c 个通过测试，常用无偏估计为：

~~~math
\operatorname{pass@}k
=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}.
~~~

要求 0 <= k <= n。它回答“候选集合中至少找到一个通过测试解的潜力”，不是一次调用默认成功率。生产系统还应报告 pass@1、隐藏测试、执行时间、资源占用和单位正确成本。

### 公开测试、隐藏测试与修复

公开测试可以提供快速反馈，隐藏测试更接近未见样本，但隐藏测试弱或覆盖不足时，二者都不能证明完全正确。定义公开测试准确率 A_pub、隐藏测试准确率 A_hid，差距为：

~~~math
\Delta_{\mathrm{pub-hid}}=A_{\mathrm{pub}}-A_{\mathrm{hid}}.
~~~

较大的差距可能来自过拟合公开测试、测试难度不同或隐藏测试噪声。修复成功率应按“收到错误后最终通过独立测试”的任务数计算，不要把生成了第二版代码当成修复成功。

### 沙箱和代码完成条件

代码执行必须限制文件根目录、网络、进程、CPU、内存和时间，并脱敏环境变量与密钥。沙箱违规率、超时率和异常进程需要单独记录。代码任务只有在补丁范围符合要求、目标测试通过、差异可解释、没有高危副作用且结果可复现时，才算完成。

## 12.10 评估样本与统计比较

### 评估样本的完整对象

一条可审计评估样本可以写为：

~~~math
e_i=(x_i,y_i^\star,V_i,P_i,z_i,q_i,g_i,r_i,b_i),
~~~

其中 x_i 是问题，y_i* 是参考答案，V_i 是候选集合，P_i 是变体，z_i 和 q_i 是步骤及标签，g_i 是能力切片，r_i 是风险标记，b_i 是推理预算。还要绑定模型、harness、提示、工具、随机种子和评估集版本。

### 变体与鲁棒性下降

对原题和变体分别测得 A_orig、A_var，可以定义鲁棒性下降：

~~~math
D_{\mathrm{robust}}=A_{\mathrm{orig}}-A_{\mathrm{var}}.
~~~

变体可以改变数值、顺序、措辞、无关上下文、格式、语言或约束位置。变体生成器本身也需要验证，不能把变体改成了另一个难度完全不同的问题。

### Paired Lift

在同一批样本上比较新旧系统逐题正确性，配对提升为：

~~~math
\Delta_{\mathrm{pair}}
=\frac{1}{N}\sum_{i=1}^{N}
\left(
\mathbf{1}[\hat y_i^{\mathrm{new}}=y_i^\star]
-\mathbf{1}[\hat y_i^{\mathrm{base}}=y_i^\star]
\right).
~~~

配对比较控制了样本难度差异，比两个模型分别报告平均分更适合判断小幅变化。报告时还应给出新模型赢、旧模型赢和双方都对/都错的数量。

### Bootstrap 置信区间

令每个样本的正确性差异为 d_i，从 d_1,...,d_N 中有放回抽取第 b 个 bootstrap 样本：

~~~math
\Delta^{(b)}=\frac{1}{N}\sum_{r=1}^{N}d_{i_r^{(b)}},
\qquad
\operatorname{CI}_{95\%}
=\left[
q_{0.025}(\Delta^{(b)}),
q_{0.975}(\Delta^{(b)})
\right].
~~~

重采样对象应是同一道题的新旧结果对，而不是独立打散两套结果。区间跨过 0 时，不能把一个点估计写成确定提升；样本本身不代表目标流量时，窄区间也不能解决外推问题。

~~~python
from random import Random
from numbers import Integral


def _validate_pairs(base, candidate):
    if len(base) != len(candidate) or not base:
        raise ValueError("paired inputs must have equal non-zero length")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value not in (0, 1)
        for value in (*base, *candidate)
    ):
        raise ValueError("paired correctness values must be 0 or 1")


def paired_lift(base, candidate):
    _validate_pairs(base, candidate)
    return sum(new - old for old, new in zip(base, candidate)) / len(base)


def bootstrap_lift(base, candidate, rounds=1000, seed=7):
    _validate_pairs(base, candidate)
    if isinstance(rounds, bool) or not isinstance(rounds, Integral) or rounds < 1:
        raise ValueError("rounds must be positive")
    rng = Random(seed)
    n = len(base)
    values = []
    for _ in range(rounds):
        indices = [rng.randrange(n) for _ in range(n)]
        values.append(
            sum(candidate[i] - base[i] for i in indices) / n
        )
    values.sort()
    low = values[int(0.025 * rounds)]
    high = values[int(0.975 * rounds)]
    return paired_lift(base, candidate), (low, high)


lift, interval = bootstrap_lift(
    [1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1],
    rounds=200,
)
assert lift == 0.2
assert interval[0] <= lift <= interval[1]
try:
    bootstrap_lift([1, 0], [1], rounds=20)
except ValueError:
    print("unpaired bootstrap input rejected")
else:
    raise AssertionError("unpaired bootstrap input was accepted")
try:
    bootstrap_lift([1, 0], [1, 1], rounds=0.5)
except ValueError:
    print("invalid bootstrap rounds rejected")
else:
    raise AssertionError("invalid bootstrap rounds was accepted")
print("lift=", lift, "interval=", interval)
~~~

这是小样本教学实现，分位点插值、区间类型和随机种子处理都可以有不同选择；生产评估还要考虑分层、聚类和多重比较。

## 12.11 Benchmark、污染与多维评估

### Benchmark 能说明什么

Benchmark 提供相同数据和指标下的比较入口，但它只测量任务、提示、模型、采样预算和评分脚本定义的行为。HELM（[Stanford CRFM HELM](https://crfm.stanford.edu/helm/latest/)）强调从准确性、鲁棒性、偏差、毒性和效率等多维度评估；它的价值在于保留维度，而不是给出一个万能总分。

SWE-bench 论文 [arXiv:2310.06770](https://arxiv.org/abs/2310.06770) 用真实软件仓库 issue 评估代码 Agent；其结果仍受仓库、测试、环境、补丁和执行策略影响。公开 benchmark 适合建立共同语言，业务结论还需要私有、干净、时间隔离和线上任务集。

### Aggregate Score Trap

总体平均分可能被容易样本或大切片主导，掩盖中文、代码、长上下文、高风险工具或核心业务切片退化。对每个关键切片 s 记录：

~~~math
\Delta_s=A_s^{\mathrm{new}}-A_s^{\mathrm{base}}.
~~~

Delta_s < 0 只说明方向上退化；是否阻止发布还要结合样本量、置信区间、流量和严重度。高风险约束不应被其他低风险切片的提升抵消。

### Clean Eval Lift

评估集应区分开发集、公开 benchmark、污染可疑集、干净集、变体集和线上样本。只有在去除近重复、训练时间重叠和调参泄漏后仍然存在的提升，才更接近泛化证据。模型卡或产品页中的自报分数要和独立复现、目标版本和成本条件分开记录。

## 12.12 Human Evaluation 与 LLM-as-a-Judge

### 人工评估的设计

开放式回答、风格、可用性、事实支持和复杂安全判断很难完全交给自动指标。人工评估应先定义维度、标签、示例、冲突处理和抽样方案，再决定采用绝对打分、两两偏好、错误类型标注或专家审查。

标注者一致性、专业资格、盲法、答案顺序和样本呈现方式都会影响结果。人评成本高但能暴露自动指标没有覆盖的错误；它也不是绝对真值，仍需要指南、仲裁和抽查。

### LLM-as-a-Judge

LLM judge 可以规模化进行 rubric 打分、两两偏好和错误分类，但会受到答案长度、顺序、格式、模型同源性和提示设计影响。MT-Bench/Chatbot Arena 论文 [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) 讨论了 LLM judge 的使用和偏差，不能被解读为所有 judge 在所有领域都可靠。

降低偏差的措施包括随机化候选顺序、隐藏模型身份、控制答案长度、使用明确 rubric、独立人工校准、多个 judge 交叉和高风险样本人工复核。judge 自身必须被评估，不能把它的分数直接当作发布真值。

### Judge-Human Agreement
一致率要求存在非空且逐样本配对的人审和 judge 标签；类别不平衡时，单独的一致率不足以支持发布结论。

若 J_i 是 judge 标签、H_i 是人工或程序标签，简单一致率为：

~~~math
A_{\mathrm{jh}}
=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[J_i=H_i].
~~~

类别不平衡时，一致率可能被多数类别抬高，还应报告混淆矩阵、precision/recall、Cohen's kappa 或 pairwise win agreement，并按任务、长度和风险切片分析。

### Length Bias

如果评审器偏好更长、更结构化或免责声明更多的回答，生成模型会被激励输出冗长模板，而不是更正确的内容。长度控制、顺序随机化、截断对照、人工样本和程序验证可以帮助识别这种偏差。

## 12.13 Factuality、Grounding、Robustness 与 Calibration

### Factuality 和 Grounding

Factuality 关注答案是否符合真实事实；grounding 关注答案是否由给定证据支持。答案碰巧正确但没有证据，或引用存在但只支持部分 claim，都不能混为同一指标。评估时应把答案拆成原子断言，并记录 supported、unsupported、contradicted 和 not-enough-information。

### 幻觉

幻觉可能来自知识缺失、训练记忆冲突、提示歧义、过度迎合、错误检索、工具误读或验证器漏洞。缓解不是一个“更强 prompt”就能完成的：需要可靠数据、RAG、引用绑定、程序验证、拒答、工具权限和错误复盘共同作用。

### Robustness

鲁棒性测试可以改变措辞、数值、顺序、无关上下文、拼写、格式、语言和攻击内容，再比较原始与变体表现。变体失败常暴露模型依赖表面模板、位置偏差或训练记忆；要按长度、语言、任务、风险和用户群体切片。

### Calibration

置信度的意义是“说有 80% 把握的样本，长期约有 80% 正确”，不是输出一句“我很确定”。温度缩放、选择性回答、verifier 分数和多候选一致性都可以作为置信度来源，但每一种都要在独立校准集上检查。高风险系统应优先使用拒答或人工确认阈值，而不是强迫模型给出伪精确概率。

## 12.14 Reasoning Safety：伪推理、工具误用与人审

### 伪推理和过度自信

伪推理是用流畅步骤包装未经支持的结论；过度自信则是在错误答案上给出强确定性。两者会让用户更难发现错误。安全评估要同时看最终结果、证据、步骤干预、置信度、拒答和高风险切片。

### 工具误用

推理模型接入工具后，风险从“回答错误”扩大到“执行错误”。常见问题包括参数来源不明、权限不匹配、不可逆动作未确认、工具输出被当作系统指令、重试造成重复副作用和执行结果未验证。系统层应负责 schema、权限、沙箱、确认、幂等、回滚与 trace，不能只要求模型“谨慎”。

### 人工审核覆盖

高风险、不可逆、涉及个体权益或专业建议的任务需要人工确认或抽查。人审覆盖率应按风险等级定义分母，不要把低风险样本大量自动通过后，写成整体高覆盖率。人工并不是模型失败后的临时补丁，而是风险分级设计的一部分。

### 严重度加权
严重度分母必须大于零，所有权重应预先定义为有限且非负；禁止类行为应独立作为硬约束统计。

格式错误、轻微事实错误、越权读取和资金转移不能在平均准确率中等价。可按风险等级 r 赋予严重度 w_r，统计：

~~~math
R_{\mathrm{severity}}
=\frac{\sum_i w_{r_i}\mathbf{1}[\text{sample }i\text{ failed}]}
{\sum_i w_{r_i}}.
~~~

权重必须事先定义，不能看到结果后为了让数字好看而改变。安全约束和禁止类行为应作为不可违反条件单独报告。

## 12.15 Evaluation Pipeline 与发布决策

### 可复现评估流水线

一条持续评估流水线至少要固定测试集版本、模型版本、harness、prompt、采样参数、工具配置、随机种子、输出日志、评分器版本和报告生成代码。每次结果都应能回到具体样本、切片和 trace。

可把一次评估样本的元数据分为：输入与参考答案、候选与中间状态、版本与预算、评分与证据、风险与人工结果。敏感内容要脱敏，但不能删掉复盘所需的 ID、版本和决策字段。

### 发布证据向量

发布判断不应压成一个不透明总分。可以保留以下证据向量：

~~~math
\mathcal{D}
=\left(
\Delta_{\mathrm{pair}},
\operatorname{CI}_{\mathrm{pair}},
\min_{s\in\mathcal{S}}\Delta_s,
R_{\mathrm{contam}},
A_{\mathrm{jh}},
C_{\mathrm{correct}},
L_{95},
R_{\mathrm{safe}}
\right).
~~~

其中 S 是关键切片，R_contam 是污染风险，A_jh 是 judge 与人工一致性，C_correct 是单位正确成本，L_95 是 P95 延迟，R_safe 是安全风险。每个分量都有自己的分母和证据来源，不能未经归一化后简单相加。

### Harness-Aware Evaluation

推理结果取决于模型、harness、环境、预算和数据：

~~~math
R=F(M,H,E,B,D).
~~~

比较不同模型或 reasoning effort 时必须固定或明确这些变量。否则更高分可能来自更长预算、更强工具、不同 prompt、更宽松的测试或更好的上下文构造。长周期 Agent 还需评估 workspace、memory、checkpoint、恢复、权限和工具协议，不能只比较最终答案。

## 12.16 资料、证据与适用边界

本章主要资料按证据类型分层：

- CoT 与 self-consistency 分别参见 [CoT Prompting](https://arxiv.org/abs/2201.11903) 和 [Self-Consistency](https://arxiv.org/abs/2203.11171)。它们说明特定任务上的提示和采样现象，不证明可见推理忠实或所有模型都同样受益。
- 候选验证和过程监督可参见 [Training Verifiers](https://arxiv.org/abs/2110.14168) 与 [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)。论文结论受数据、步骤标签、模型和验证器设置限制。
- 搜索推理可参见 [Tree of Thoughts](https://arxiv.org/abs/2305.10601)；搜索收益取决于状态、动作、评分器和预算，不能由算法名称直接推出。
- 代码生成评估可参见 [HumanEval](https://arxiv.org/abs/2107.03374)，pass@k 不能替代隐藏测试、pass@1 和生产执行成本。
- 多维模型评估可参见 [HELM](https://crfm.stanford.edu/helm/latest/)；LLM judge 的偏差可参见 [MT-Bench/Chatbot Arena](https://arxiv.org/abs/2306.05685)。
- 校准基础可参见 [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)；代码评估工具链可参考 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)，实现版本和任务配置仍需固定。

论文、benchmark、官方 model card、产品页、模型输出和目标系统实测的可信度用途不同。论文适合说明方法和实验条件，benchmark 适合定义共同测量，官方材料适合说明某版本公开声明；业务结论必须由独立、干净、成对、分层、成本受控的评估支持。

## 12.17 Reasoning 与评估的统一理解

Reasoning 系统可以看成一个候选—验证—资源分配循环：

~~~text
问题与约束
  -> 生成候选步骤/答案
  -> 程序、工具、verifier 或人工检查
  -> 选择、回退、修复或继续搜索
  -> 记录预算、延迟、证据、风险和结果
  -> 在干净切片与线上任务上比较
~~~

CoT 提供中间表示，self-consistency 提供多路径候选，verifier 负责质量判断，process supervision 提供步骤级信号，search 组织分支与回退，TTC 决定额外计算投向哪里，评估流水线判断提升是否真实、可复现、可负担和安全。它们没有一个可以单独保证正确性。

对初学者，最重要的习惯是把“生成了答案”“存在正确候选”“验证器选中了候选”“任务完成”分开。对专家，最重要的是保留条件：模型、harness、数据、预算、验证器、评估集、成本和风险。只有这些条件同时明确，reasoning 的改进才有可比较的意义。

下一章将进入安全与治理，继续讨论模型能力如何转化成组织、权限和风险控制。
