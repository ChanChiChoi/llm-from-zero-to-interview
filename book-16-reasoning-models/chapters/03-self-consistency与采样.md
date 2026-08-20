# 第三章：Self-Consistency 与采样

上一章讨论了 CoT 如何提供中间状态。本章继续问一个更具体的问题：如果一次 CoT 可能在某一步出错，能不能对同一道题生成多条不同路径，再利用它们之间的共识降低偶然错误？

Self-consistency 的答案是：在一定条件下可以。它先用随机采样生成多个候选，再抽取并归一化每条候选的最终答案，最后通过多数投票、加权投票或外部 verifier 选择结果。

这个方法看起来像“多做几遍题”，实际上包含多个容易被忽略的系统问题：

1. 采样参数决定候选分布，而不是只决定文字风格。
2. 多条路径可能高度相关，表面不同但共享同一个错误。
3. 答案抽取和等价形式归一化会改变投票结果。
4. pass@k 只说明候选集合里可能存在正确答案，不说明系统能选中它。
5. 候选数量增加会增加 token、延迟、队列压力和验证成本。
6. 复杂任务上提升的同时，简单任务和高风险任务可能发生回归。

## 0.1 本章的资料与证据边界

Self-Consistency 的核心方法来自 Wang 等人的 Self-Consistency Improves Chain of Thought Reasoning in Language Models。pass@k 的常见评估口径来自 HumanEval 及其代码生成实验。temperature、top-k 和 top-p 的实现语义参考了采样研究与 Transformers 官方生成文档。

论文结果是在特定模型、数据集、采样参数和答案解析协议下获得的。不能把一个论文中的最佳 k、temperature 或准确率直接搬到另一个模型和业务上。官方框架文档可以说明参数如何截断或重排 token 分布，但不能说明某个模型在真实任务上一定更可靠。

本章中的代码是教学构造。它用预先写好的候选、分数和 token 数模拟采样结果，用于解释聚合和成本，不代表真实模型的概率校准或线上性能。

## 0.2 先看一个投票例子

同一道题生成四条候选：

~~~text
路径 1：步骤正确，答案 42
路径 2：中间计算错，答案 36
路径 3：步骤正确，答案 42
路径 4：读错题意，答案 40
~~~

如果只使用路径 2，答案会错；如果把四条路径的最终答案归一化后投票，42 可能胜出。这个结论依赖一个关键条件：错误不是完全相同的，而且正确路径在候选中占有足够比例。

如果四条路径都把题目中的“每小时”误读成“每分钟”，投票会非常稳定地选出同一个错误答案。稳定性不等于正确性，共识也不等于独立证据。

## 1. Self-Consistency 的完整对象

### 1.1 小白视角：多次尝试再归纳

人解决难题时可能先写出几种解法，再比较它们的结果。Self-consistency 把这种思路放到模型推理时：

1. 对同一个输入生成多个候选。
2. 从每个候选中抽取最终答案。
3. 把等价答案合并。
4. 统计频次或分数。
5. 交付聚合后的结果。

它不是重新训练模型，而是给一次请求更多计算预算。

### 1.2 专家视角：候选集合

对第 i 个问题，设生成 K_i 个候选：

~~~math
\mathcal{C}_i=
\left\{(z_{ij},\hat y_{ij},s_{ij},T_{ij})\right\}_{j=1}^{K_i}
~~~

其中：

- z_ij 是第 j 条推理轨迹；
- y_hat_ij 是从轨迹中抽取的最终答案；
- s_ij 是可选的 verifier、规则或模型分数；
- T_ij 是该候选使用的 token 或等价计算量。

聚合器接收的是候选集合，而不是只有答案字符串。即使最终采用多数投票，也应保留轨迹、抽取结果、归一化结果和失败原因，方便分析共识是否来自重复错误。

### 1.3 多路径不是多模型

同一个模型、同一个 prompt、不同随机数，能够产生不同 token 路径；但这些路径仍然共享参数、知识缺陷、系统提示和输入误读。增加随机种子能带来采样多样性，却不等于引入独立专家。

如果需要真正互补的候选，可以改变提示、解法顺序、工具、模型或搜索策略，但每种变化都会增加实验变量和系统成本。

## 2. 从 logits 到候选：随机性如何产生

### 2.1 Greedy decoding

给定当前位置的 logits l_v，greedy decoding 选择分数最高的 token：

~~~math
v^*=\arg\max_{v\in\mathcal{V}}l_v
~~~

如果模型、输入和处理流程完全相同，greedy 通常重复同一条路径。因此它适合稳定复现一次结果，不适合作为 self-consistency 的多样性来源。

### 2.2 Temperature

temperature 用 tau 调整 logits 的尖锐程度：

~~~math
p_v(\tau)=
\frac{\exp(l_v/\tau)}
{\sum_{u\in\mathcal{V}}\exp(l_u/\tau)}
~~~

这里要求词表 \(\mathcal{V}\) 非空、logits 为有限数，且 \(\tau>0\)。\(\tau=0\) 不是这个公式的合法输入；工程实现通常把它解释为接近 greedy 的特殊路径，而不是直接执行除以零。V 是词表，l_v 是 token v 的 logit，tau 是温度：

- tau 较小，概率集中在高分 token，候选更稳定；
- tau 较大，低分 token 获得更多机会，候选更分散；
- tau 趋近于 0 时，行为接近 greedy；
- tau 过大时，推理可能从合理路径发散。

temperature 并不是“思考深度”旋钮。它只改变采样分布，不能保证探索到正确解法。

### 2.3 Top-k

Top-k 只保留 logits 最高的 k 个 token，再在这个集合中重新归一化采样：

~~~math
\mathcal{V}_k=
\left\{v:\;v\text{ 属于 logits 最高的 }k\text{ 个 token}\right\}
~~~

这里的 k 必须是整数，并满足 \(1\le k\le |\mathcal{V}|\)；k 等于词表大小时等于不做 top-k 截断。k 太小会限制多样性，k 太大则可能把低质量长尾 token 引入推理。top-k 的 k 是每个生成位置的候选数量，不是整条回答的路径数量。

### 2.4 Top-p 或 nucleus sampling

先按概率从大到小排列 token，选择累计概率达到 p 的最小前缀：

~~~math
\mathcal{N}_p=
\left\{v_{(1)},\ldots,v_{(m)}\right\},
\qquad
\sum_{j=1}^{m}p_{(j)}\ge p
~~~

这里要求概率来自合法的非空分布，且 \(0<p\le 1\)。当分布很尖时，前缀可能很短；当分布平坦时，前缀会变长。因此 top-p 会根据当前位置的不确定性动态调整 token 集合，比固定 top-k 更能适应不同位置。

### 2.5 参数的组合关系

真实生成通常先调整 logits，再进行 top-k 或 top-p 截断，最后从剩余分布采样。不同框架对处理顺序、边界值和随机数实现可能存在差异。复现实验必须记录：

1. 模型版本和 tokenizer。
2. temperature。
3. top-k、top-p 和是否启用各自的截断。
4. 最大新 token 数和停止 token。
5. 随机种子、批处理方式和并发方式。

只记录“temperature=0.7”而不记录其他参数，通常不足以复现候选分布。

## 3. 多样性不是越高越好

### 3.1 答案多样性

把答案归一化函数记为 nu，答案层面的多样性可以粗略写成：

~~~math
D_{\mathrm{ans}}=
\frac{1}{N}\sum_{i=1}^{N}
\frac{\left|\left\{\nu(\hat y_{ij})\right\}_{j=1}^{K_i}\right|}{K_i}
~~~

这个式子要求评估题目数 \(N>0\)，并且每道题都有 \(K_i>0\) 条可解析候选；解析失败的候选不能悄悄当作一个普通答案塞进分子，应另报 parser failure rate。若某道题没有任何可解析候选，该题的多样性未定义，不能用 0 偷换。它表示每道题不同归一化答案的比例。D_ans 很低，说明候选可能只是同一路径的复制；D_ans 很高，也可能说明采样已经失控。

### 3.2 轨迹多样性和答案多样性不同

两条轨迹可能文字完全不同，但最后都得到 42；也可能文字高度相似，却在最后一步得到不同答案。答案多样性适合研究投票，轨迹多样性适合研究解法是否真正互补。

可以额外统计：

- 不同答案的比例；
- 中间关键变量的差异；
- 步骤顺序的差异；
- 工具调用的差异；
- 轨迹之间的 token 或 embedding 相似度。

没有一种相似度能完整代表“独立思考”。指标应与任务结构结合。

### 3.3 相关错误

如果候选都依赖一个错误前提 q，可以写成条件相关：

~~~math
p(\hat y_{i1},\ldots,\hat y_{iK}\mid q)
~~~

当 q 被所有候选共享时，增加 K 主要是在重复同一个条件下的采样。self-consistency 的收益上限由错误相关性决定，而不只是由 K 决定。

### 3.4 理想独立模型下的多数概率

假设每条路径独立地以概率 p 得到正确答案，K 为奇数，则多数正确的概率为：

~~~math
P_{\mathrm{maj}}=
\sum_{j=(K+1)/2}^{K}
\binom{K}{j}p^j(1-p)^{K-j}
~~~

当 p 大于 0.5 时，增加 K 在这个理想模型中可能提升多数正确概率；当 p 小于 0.5 时，更多采样可能使多数错误更稳定。真实候选通常不独立，所以不能直接用这个公式预测线上准确率。

### 3.5 候选质量的混合分布

采样可能同时产生容易题的高置信候选和难题的低质量候选。整体平均多样性会掩盖某个关键切片的候选崩溃。因此应按题型、难度、输入长度和是否需要工具分层统计。

## 4. 答案抽取与等价类归一化

### 4.1 为什么抽取会改变实验结论

Self-consistency 不直接对整段自然语言投票，而是对抽取后的答案投票。如果抽取器把中间数字当成最终答案，或者把等价答案分成不同字符串，投票结果就会偏离模型实际候选。

推荐使用明确的输出协议：

~~~text
推理：
净流入速度为 2 升/分钟。
时间为 20 / 2 = 10 分钟。

最终答案：10 分钟
~~~

协议的作用是降低解析歧义，不是提高模型知识。

### 4.2 抽取器的失败类型

常见失败包括：

1. 输出没有最终答案标记。
2. 过程里出现多个数字。
3. 同时给出条件答案和无条件答案。
4. 单位在正文，数字在末行。
5. 分数、百分比和小数混用。
6. 中文数字、阿拉伯数字和 LaTeX 混用。
7. 多选题同时写出选项字母和解释。

每种失败都应有独立计数。不能把所有解析失败都归因于采样策略。

### 4.3 数值和单位归一化

例如下面四种答案可能表示同一个数：

~~~text
42
42.0
答案是 42
四十二
~~~

但“42 秒”和“42 分钟”绝不应被简单合并。归一化必须把数值、单位、精度和题目约束一起考虑。

对于数值答案，可以先解析数值，再按任务规定的容差比较：

~~~math
\mathrm{Match}(a,b)=
\mathbb{1}\left[|a-b|\le \epsilon_{\mathrm{abs}}
\lor
\frac{|a-b|}{\max(1,|b|)}\le\epsilon_{\mathrm{rel}}\right]
~~~

epsilon_abs 是绝对误差阈值，epsilon_rel 是相对误差阈值。不同任务的阈值不能随意复用。

### 4.4 答案等价类

将所有语义等价的字符串归入同一个类别：

~~~math
[a]=\left\{b:\;b\text{ 与 }a\text{ 在任务语义下等价}\right\}
~~~

投票应该在等价类上进行，而不是在原始字符串上进行。等价关系必须由任务规则、符号系统或人工协议定义；不能把模型的相似度分数直接当成数学等价。

## 5. 聚合方法

### 5.1 Majority vote

多数投票选择出现次数最多的归一化答案：

~~~math
\hat y_i^{\mathrm{maj}}
=
\arg\max_{a}
\sum_{j=1}^{K_i}
\mathbb{1}\left[\nu(\hat y_{ij})=a\right]
~~~

这个定义要求候选集合非空，即 \(K_i>0\)。如果有多个答案取得同一个最大票数，argmax 是一个集合而不是唯一答案，系统必须使用下一节的平票策略；不能把实现恰好返回的第一个元素误解为统计上更可信。优点是简单、便宜、无需额外模型；缺点是它不理解路径质量，也无法处理少数正确、多数错误的情况。

### 5.2 平票处理

当两个答案票数相同，系统必须定义规则：

- 增加一个候选；
- 交给 verifier；
- 选择分数更高的答案；
- 返回不确定状态；
- 使用固定但可审计的 tie-break。

随机选择会使结果难以复现；默认选第一个答案会引入候选顺序偏差。

### 5.3 Weighted vote

如果每条候选有分数 s_ij，可以按分数加权：

~~~math
\hat y_i^{\mathrm{w}}
=
\arg\max_a
\sum_{j=1}^{K_i}
s_{ij}\,
\mathbb{1}\left[\nu(\hat y_{ij})=a\right]
~~~

s_ij 可以来自程序测试、规则检查、verifier 或经过校准的置信度。未经校准的 log probability 可能偏好短答案、常见措辞或高频格式，不应直接视为正确概率。
s_ij 至少应为有限数；若把它解释为置信度，通常还要约定非负或 \([0,1]\) 范围。分数范围不是加权公式自动提供的，而是评分器协议的一部分。候选集合为空或所有候选分数都无效时，加权结果应返回不可用，而不是人为补一个零分答案。

### 5.4 Verifier rerank

也可以先生成候选，再让 verifier 直接选择候选：

~~~math
j^*=\arg\max_{j\in\{1,\ldots,K_i\}}s_{ij},
\qquad
\hat y_i=\hat y_{ij^*}
~~~

多数投票只看答案频次；rerank 还看候选内容或外部测试。它可以救回少数正确候选，也可能被 verifier 偏差带偏。

### 5.5 聚合器也需要评估

不能只评估候选生成模型。应单独测：

1. 抽取器是否读对答案。
2. 归一化器是否合并等价形式。
3. 投票器是否处理平票。
4. weighted vote 是否使用了校准分数。
5. verifier 是否覆盖关键边界。

聚合器的错误可能让一个本来有正确候选的系统最终交付错误结果。

## 6. Pass@k 与最终选择准确率

### 6.1 Pass@k 测量什么

设生成 \(n\) 个候选，其中 \(c\) 个通过任务测试。在已知这组候选的通过标记时，随机抽取 k 个且不放回，至少包含一个正确候选的无放回估计为：

~~~math
\mathrm{pass@}k
=
1-\frac{\binom{n-c}{k}}{\binom{n}{k}}
~~~

这个式子要求 \(n\) 是正整数、\(0\le c\le n\)、k 是正整数且 \(k\le n\)。当 \(c=0\) 时结果为 0；当 n-c 小于 k 时，分子组合数为 0，估计值为 1。它测量的是候选集合的潜力：只要知道哪些候选正确，就能判断集合里是否包含解。线上选择阶段通常并不知道 c，因此不能在生成时用真实 c 代替 parser、测试器或 verifier。

### 6.2 Pass@k 不等于系统成功率

线上系统通常不知道 c，也不知道哪个候选正确。它还必须依赖答案 parser、投票器、verifier 或测试器选择候选。因此至少要区分：

- pass@k：候选集合中是否存在正确答案；
- majority accuracy：多数聚合后的答案是否正确；
- rerank accuracy：评分器是否选中正确候选；
- execution success：程序或工具动作是否真实成功。

如果 pass@10 很高但 rerank 很差，系统仍然不能可靠交付。

### 6.3 pass@1 与单次准确率

对每道题只生成一个候选时，pass@1 与单次候选正确率在任务定义一致的情况下接近；当 n 大于 1 时，pass@1 的估计可以表示候选集合中正确比例，但它仍不是最终聚合准确率。

### 6.4 评估脚本的边界

代码任务中，测试集覆盖不足会让 pass@k 高估真实正确率；数学任务中，答案解析器可能把格式错误掩盖；开放任务中，是否“正确”本身可能依赖人工协议。任何 pass@k 数字都必须与测试集、解析器和允许工具一起报告。

## 7. 成本、延迟与候选数量

### 7.1 Token 成本

如果第 i 道题有 K_i 条候选，每条候选使用 T_ij 个 token，则生成成本的简单近似为：

~~~math
C_{\mathrm{sample}}
=
\sum_{i=1}^{N}\sum_{j=1}^{K_i}T_{ij}
~~~

真实成本还包括输入重复、模型调用固定开销、验证器、工具执行、缓存、GPU 时间和失败重试。

### 7.2 延迟不一定线性

在串行调用中，K 个候选可能使首个结果等待时间近似增加 K 倍；在批处理或并行调用中，总 wall-clock 延迟可能接近最长候选，但 GPU 峰值显存、队列和吞吐压力会增加。

因此要同时记录：

- 单请求总 token；
- 首 token 延迟；
- 最终结果延迟；
- P95/P99；
- 并发下的队列等待；
- 验证器和工具的额外时间。

### 7.3 单位成功成本

对一个固定评估窗口，可以定义：

~~~math
C_{\mathrm{correct}}
=
\frac{\sum_{i=1}^{N}C_i}
{\sum_{i=1}^{N}\mathbb{1}[\mathrm{Success}_i=1]}
~~~

这里的 \(C_i\) 应是同一成本单位下的非负请求成本，且成功样本数必须大于 0；如果评估窗口一次都没有成功，单位成功成本是未定义的，不应记录成 0。若要比较不同方案，还要固定评估窗口和成功判定。 如果采样让准确率提高，但单位成功成本变成数倍，是否值得取决于业务中一次错误的代价。

### 7.4 候选数量的边际收益

把最终聚合准确率记作 A(K)，增加一个候选的边际收益为：

~~~math
\Delta_K=A(K+1)-A(K)
~~~

当 Delta_K 接近 0，而成本和延迟继续上升时，应停止增加候选。更重要的是在困难切片上单独计算 Delta_K，因为总体平均可能掩盖高风险样本没有改善。

## 8. 早停与自适应采样

### 8.1 固定 K 的问题

固定 K 对所有问题使用相同计算量，简单题浪费预算，难题可能仍然不够。它还会让服务成本随流量直接放大。

### 8.2 根据共识早停

如果前 m 个候选已经形成足够强的答案优势，可以提前停止。设当前领先答案票数为 n_1，第二名为 n_2，可以用差值或置信上界作为停止信号：

~~~math
\Delta_{\mathrm{vote}}=n_1-n_2
~~~

但早停不能只看票数。若候选高度相关，共识可能是共同错误；若题目风险高，还应要求独立 verifier 或工具检查。

### 8.3 验证失败触发增采样

一种常见策略是先生成少量候选，若 parser 失败、答案冲突或 verifier 不通过，再增加候选或改用工具：

~~~math
K_i=
\begin{cases}
K_{\mathrm{base}}, & \mathrm{valid}(C_i)=1\\
K_{\mathrm{base}}+\Delta K, & \mathrm{valid}(C_i)=0
\end{cases}
~~~

valid 需要由可观测规则定义，不能只使用模型自报的“我已经检查过”。

### 8.4 自适应采样的风险

如果系统只在看起来容易的请求上停止，难题样本可能被过早结束；如果模型的置信度不校准，动态策略会把预算分给错误的请求。自适应策略必须在留出的任务切片上评估，而不是根据离线平均分数直接上线。

## 9. 什么时候 self-consistency 有效

### 9.1 更适合的任务

self-consistency 更适合：

1. 最终答案可以稳定抽取和归一化。
2. 存在多条合理解法。
3. 单次生成有偶然错误。
4. 单条候选正确率已经高于随机水平。
5. 额外延迟和成本可以接受。

数学应用题、逻辑题、结构化选择题和可执行代码候选通常符合这些条件。

### 9.2 不适合的任务

以下任务中，盲目投票往往收益有限：

- 开放式创意写作，答案没有唯一等价类；
- 模型对题意有系统性误解；
- 事实需要外部最新数据；
- 候选答案无法可靠解析；
- 低延迟强约束服务；
- 高风险任务没有独立 verifier。

这类任务可能更需要检索、工具、人工评估或明确的拒答，而不是更多随机采样。

### 9.3 少数正确和多数错误

如果正确候选只有一个，错误候选有四个，多数投票很可能选错。此时应让候选接受程序测试、事实比对或过程检查。self-consistency 不是替代 verifier 的理由。

## 10. Self-Consistency 与 verifier 的组合

### 10.1 组合流程

一个更完整的流程是：

1. 生成多个候选。
2. 解析答案和关键中间字段。
3. 运行规则或工具检查。
4. 用 verifier 过滤明显错误。
5. 对剩余候选投票或重排。
6. 输出答案、证据和不确定状态。

### 10.2 先过滤还是先投票

数学题可以先用格式和算术规则过滤；代码题可以先运行测试；文档问答可以先检查引用是否命中原文。先过滤能减少错误候选对投票的影响，但 verifier 运行本身有成本，也可能误删少数正确候选。

### 10.3 分数校准

若 verifier 输出分数 s，不能只观察排序，还要检查分数与成功概率是否匹配。可以按分数区间统计真实正确比例，或报告可靠性图和 Brier score。

~~~math
\mathrm{Brier}
=
\frac{1}{N}\sum_{i=1}^{N}(s_i-o_i)^2
~~~

这里要求 \(N>0\)、\(s_i\in[0,1]\)，并且 \(o_i\) 是 0 或 1。s_i 是预测成功概率，o_i 是真实结果。一个评分器排序能力不错，但概率校准很差，仍然可能导致错误的动态预算和成本决策。

### 10.4 verifier 的反作用

verifier 偏好格式整齐的候选时，weighted vote 可能选择漂亮但错误的路径；测试集覆盖不足时，代码候选可能针对测试过拟合；答案解析器有漏洞时，采样器可能学会利用格式。验证器应使用隐藏切片、反事实输入和独立实现进行检查。

## 11. 失败模式与诊断

### 11.1 所有候选都错

原因可能是知识缺失、题意误读、数据污染或任务本身缺少信息。增加 K 只能重复错误，应该改用检索、工具、澄清问题或拒答。

### 11.2 多数投票错，少数候选对

这是投票器的问题，不一定是生成器的问题。应记录候选集合中是否有正确解，并比较 verifier 是否能够救回少数候选。

### 11.3 答案抽取错

正确轨迹可能被 parser 读成错误答案。应保存原始响应、抽取片段、归一化结果和解析错误类型。

### 11.4 标准化过度

把不同单位、不同条件或不同精度的答案合并，会制造假共识。归一化规则必须绑定任务 schema。

### 11.5 多样性过低

低 temperature、过小 top-k 或过于严格的 prompt 可能让所有候选相同。此时增加 K 主要增加重复成本。

### 11.6 多样性过高

temperature 过高或 top-p 过大可能引入大量语义无关、格式错误和不可验证候选。此时多数票会被噪声稀释。

### 11.7 verifier 选错

少数正确候选可能得分低，错误候选可能利用评分器偏差得分高。需要独立验证器、人工抽样和隐藏测试。

### 11.8 成本失控

候选数量、输出长度和验证调用互相放大。应把总成本拆成生成、解析、验证、工具和重试，而不是只记录模型 token。

## 12. 评估 self-consistency 实验

### 12.1 基线必须清楚

至少比较：

1. greedy direct；
2. greedy CoT；
3. sampling 单候选；
4. majority vote；
5. weighted vote 或 verifier rerank；
6. 工具增强版本。

否则无法知道提升来自采样、CoT、验证器还是工具。

### 12.2 同一预算下比较

如果一个方案使用 16 条候选，另一个只使用 1 条，直接比较准确率会高估多采样收益。应报告：

- 固定候选数的质量；
- 固定 token 预算的质量；
- 固定延迟预算的质量；
- 单位成功成本；
- 不同切片上的失败类型。

### 12.3 数据污染与题目重复

self-consistency 可能放大记忆。若所有候选都复述训练语料中的同一个答案，投票并不能证明模型重新推导了结果。应使用变式题、时间切分、隐藏测试和近重复检测。

### 12.4 统计不确定性

采样实验本身有随机性。应重复不同随机种子，报告均值、区间和每个预算点的结果。只挑表现最好的 seed 或 k 会造成选择偏差。

## 13. 一个综合案例：代码候选的选择

假设模型需要实现一个数组去重函数。系统可以生成四个候选：

~~~text
候选 A：代码简短，但没有处理空数组。
候选 B：代码较长，通过公开样例。
候选 C：使用集合，但输出顺序错误。
候选 D：通过公开和隐藏测试。
~~~

如果只做 majority vote，A、B、C 可能因为格式或常见写法得到较多支持；如果运行测试，D 可以被识别。这个例子说明代码任务中程序化 verifier 往往比语言风格投票更合适。

但测试器也要检查：

1. 空输入；
2. 重复元素；
3. 负数和大整数；
4. 顺序契约；
5. 时间和空间限制；
6. 隐藏边界。

pass@4 高，只说明 D 存在；交付成功还要求测试器能够找到 D。

## 14. 最小可运行实验：采样、投票与加权选择

下面的 demo 用五道 toy reasoning 题模拟：

1. greedy：单次答案；
2. majority：归一化后的多数投票；
3. weighted：按 verifier 或规则分数加权；
4. pass@2：候选集合至少包含一个正确答案的估计；
5. 候选多样性和 token 成本。

其中 distractor_math 专门构造“多数错误、少数正确”的情况，用于观察 weighted 选择是否能救回正确候选。

为了让定义域在代码中可见，下面的审计函数会拒绝空题集、重复题目 ID、空答案、非有限或负的 verifier 分数、负 token、非法 k 和空候选集合。答案归一化器只服务于这个受控 toy 数据集：它把示例中的单位文本约化为数值，不能替代真实任务中按题目 schema 处理单位、精度和条件的 parser。

~~~python
from collections import Counter, defaultdict
from math import comb
import math
import re


problems = [
    {
        "id": "water_tank",
        "gold": "10",
        "greedy": "12",
        "candidates": [
            {"answer": "10 minutes", "score": 0.86, "tokens": 58},
            {"answer": "10", "score": 0.78, "tokens": 54},
            {"answer": "12", "score": 0.42, "tokens": 46},
            {"answer": "10.0", "score": 0.81, "tokens": 61},
        ],
    },
    {
        "id": "distractor_math",
        "gold": "12",
        "greedy": "99",
        "candidates": [
            {"answer": "111", "score": 0.35, "tokens": 52},
            {"answer": "111", "score": 0.40, "tokens": 49},
            {"answer": "12", "score": 0.91, "tokens": 57},
            {"answer": "111", "score": 0.37, "tokens": 50},
            {"answer": "12 units", "score": 0.88, "tokens": 59},
        ],
    },
    {
        "id": "code_loop",
        "gold": "pass",
        "greedy": "fail",
        "candidates": [
            {"answer": "fail", "score": 0.20, "tokens": 42},
            {"answer": "pass", "score": 0.83, "tokens": 64},
            {"answer": "pass", "score": 0.79, "tokens": 68},
            {"answer": "pass", "score": 0.77, "tokens": 63},
        ],
    },
    {
        "id": "logic_grid",
        "gold": "blue",
        "greedy": "red",
        "candidates": [
            {"answer": "blue", "score": 0.72, "tokens": 47},
            {"answer": "green", "score": 0.41, "tokens": 43},
            {"answer": "blue", "score": 0.74, "tokens": 45},
            {"answer": "red", "score": 0.30, "tokens": 44},
        ],
    },
    {
        "id": "capital_lookup",
        "gold": "tokyo",
        "greedy": "Tokyo",
        "candidates": [
            {"answer": "tokyo", "score": 0.82, "tokens": 20},
            {"answer": "kyoto", "score": 0.25, "tokens": 35},
            {"answer": "Tokyo.", "score": 0.81, "tokens": 22},
            {"answer": "tokyo city", "score": 0.76, "tokens": 24},
        ],
    },
]

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_answer(answer):
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a non-empty string")
    text = answer.strip().lower()
    if text in {"pass", "fail", "blue", "green", "red", "tokyo", "kyoto"}:
        return text
    if text.endswith("."):
        text = text[:-1]
    if text == "tokyo city":
        return "tokyo"
    numbers = NUMBER_RE.findall(text)
    if numbers:
        value = float(numbers[-1])
        if not math.isfinite(value):
            raise ValueError("numeric answer must be finite")
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return text


def majority_vote(answers):
    if not isinstance(answers, list) or not answers:
        raise ValueError("answers must be a non-empty list")
    if any(not isinstance(answer, str) or not answer for answer in answers):
        raise ValueError("answers must contain non-empty strings")
    counts = Counter(answers)
    max_count = max(counts.values())
    winners = {answer for answer, count in counts.items() if count == max_count}
    # A fixed lexical tie-break keeps the toy result independent of input order.
    return min(winners)


def validate_candidate(candidate):
    required = {"answer", "score", "tokens"}
    if not isinstance(candidate, dict) or set(candidate) != required:
        raise ValueError("candidate schema must be answer, score and tokens")
    normalize_answer(candidate["answer"])
    score = candidate["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("score must be numeric")
    if not math.isfinite(score) or score < 0:
        raise ValueError("score must be a finite non-negative number")
    tokens = candidate["tokens"]
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise ValueError("tokens must be a non-negative integer")


def validate_problems(items):
    if not isinstance(items, list) or not items:
        raise ValueError("problems must be a non-empty list")
    seen_ids = set()
    for item in items:
        required = {"id", "gold", "greedy", "candidates"}
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("problem schema is invalid")
        if not isinstance(item["id"], str) or not item["id"] or item["id"] in seen_ids:
            raise ValueError("problem ids must be unique non-empty strings")
        seen_ids.add(item["id"])
        normalize_answer(item["gold"])
        normalize_answer(item["greedy"])
        candidates = item["candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("each problem needs non-empty candidates")
        for candidate in candidates:
            validate_candidate(candidate)


def weighted_vote(candidates):
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list")
    for candidate in candidates:
        validate_candidate(candidate)
    scores = defaultdict(float)
    for candidate in candidates:
        scores[normalize_answer(candidate["answer"])] += candidate["score"]
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def pass_at_k(n, c, k):
    values = (n, c, k)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("n, c and k must be integers")
    if n <= 0 or c < 0 or c > n or k <= 0 or k > n:
        raise ValueError("pass_at_k requires n > 0, 0 <= c <= n and 0 < k <= n")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


validate_problems(problems)
rows = []
for item in problems:
    gold = normalize_answer(item["gold"])
    normalized = [
        normalize_answer(candidate["answer"])
        for candidate in item["candidates"]
    ]
    n = len(normalized)
    c = sum(answer == gold for answer in normalized)
    rows.append(
        {
            "id": item["id"],
            "greedy": normalize_answer(item["greedy"]),
            "majority": majority_vote(normalized),
            "weighted": weighted_vote(item["candidates"]),
            "correct_candidates": c,
            "unique_answers": len(set(normalized)),
            "tokens": sum(candidate["tokens"] for candidate in item["candidates"]),
            "pass_at_2": round(pass_at_k(n, c, 2), 3),
        }
    )

greedy_accuracy = sum(
    row["greedy"] == normalize_answer(item["gold"])
    for row, item in zip(rows, problems)
) / len(problems)
majority_accuracy = sum(
    row["majority"] == normalize_answer(item["gold"])
    for row, item in zip(rows, problems)
) / len(problems)
weighted_accuracy = sum(
    row["weighted"] == normalize_answer(item["gold"])
    for row, item in zip(rows, problems)
) / len(problems)
pass_at_1_est = sum(
    row["correct_candidates"] / len(item["candidates"])
    for row, item in zip(rows, problems)
) / len(problems)
pass_at_2_est = sum(row["pass_at_2"] for row in rows) / len(rows)
total_tokens = sum(row["tokens"] for row in rows)
total_candidates = sum(len(item["candidates"]) for item in problems)
avg_unique_ratio = sum(
    row["unique_answers"] / len(item["candidates"])
    for row, item in zip(rows, problems)
) / len(rows)
majority_failures = [
    row["id"]
    for row, item in zip(rows, problems)
    if row["majority"] != normalize_answer(item["gold"])
]
weighted_rescues = [
    row["id"]
    for row, item in zip(rows, problems)
    if row["majority"] != normalize_answer(item["gold"])
    and row["weighted"] == normalize_answer(item["gold"])
]

majority_correct_count = sum(
    row["majority"] == normalize_answer(item["gold"])
    for row, item in zip(rows, problems)
)

report = {
    "greedy_accuracy": round(greedy_accuracy, 3),
    "majority_accuracy": round(majority_accuracy, 3),
    "weighted_accuracy": round(weighted_accuracy, 3),
    "pass_at_1_est": round(pass_at_1_est, 3),
    "pass_at_2_est": round(pass_at_2_est, 3),
    "avg_unique_answer_ratio": round(avg_unique_ratio, 3),
    "total_candidates": total_candidates,
    "total_tokens": total_tokens,
    "cost_per_majority_correct": (
        round(total_tokens / majority_correct_count, 3)
        if majority_correct_count
        else None
    ),
    "majority_failures": majority_failures,
    "weighted_rescues": weighted_rescues,
    "per_problem": rows,
}

assert report["greedy_accuracy"] == 0.2
assert report["majority_accuracy"] == 0.8
assert report["weighted_accuracy"] == 1.0
assert report["pass_at_2_est"] == 0.907
assert report["total_tokens"] == 1003

for key, value in report.items():
    print(f"{key}={value}")
~~~

预期输出：

~~~text
greedy_accuracy=0.2
majority_accuracy=0.8
weighted_accuracy=1.0
pass_at_1_est=0.63
pass_at_2_est=0.907
avg_unique_answer_ratio=0.53
total_candidates=21
total_tokens=1003
cost_per_majority_correct=250.75
majority_failures=['distractor_math']
weighted_rescues=['distractor_math']
per_problem=[{'id': 'water_tank', 'greedy': '12', 'majority': '10', 'weighted': '10', 'correct_candidates': 3, 'unique_answers': 2, 'tokens': 219, 'pass_at_2': 1.0}, {'id': 'distractor_math', 'greedy': '99', 'majority': '111', 'weighted': '12', 'correct_candidates': 2, 'unique_answers': 2, 'tokens': 267, 'pass_at_2': 0.7}, {'id': 'code_loop', 'greedy': 'fail', 'majority': 'pass', 'weighted': 'pass', 'correct_candidates': 3, 'unique_answers': 2, 'tokens': 237, 'pass_at_2': 1.0}, {'id': 'logic_grid', 'greedy': 'red', 'majority': 'blue', 'weighted': 'blue', 'correct_candidates': 2, 'unique_answers': 3, 'tokens': 179, 'pass_at_2': 0.833}, {'id': 'capital_lookup', 'greedy': 'tokyo', 'majority': 'tokyo', 'weighted': 'tokyo', 'correct_candidates': 3, 'unique_answers': 2, 'tokens': 101, 'pass_at_2': 1.0}]
~~~

这个实验应这样解读：

1. majority_accuracy 高于 greedy_accuracy，说明这组构造样本中的多路径聚合降低了一次生成的偶然错误。
2. distractor_math 的 majority vote 选错，但 weighted vote 选对，说明 pass@k 高不等于自动聚合后一定正确。
3. pass_at_2_est 高于 pass_at_1_est，说明增加候选数提高了候选集合包含正确答案的概率。
4. avg_unique_answer_ratio 不能太低，否则多次采样主要是在重复同一答案。
5. cost_per_majority_correct 必须和准确率一起看，否则 self-consistency 会退化成堆叠 token。

## 15. 怎样设计 self-consistency 实验

### 15.1 固定比较协议

比较 greedy、采样和聚合时，固定模型版本、tokenizer、输入题目、最大输出、答案 parser、采样参数记录格式和测试切分。否则提升可能来自答案处理，而不是采样。

### 15.2 逐步增加候选数量

至少测试 K=1、2、4、8、16 等预算点，绘制准确率、pass@k、候选多样性、延迟和成本曲线。只报告最佳 K 会隐藏边际收益递减和高成本。

### 15.3 分层观察

按题型、难度、输入长度、答案空间大小、是否需要工具和是否存在干扰条件分层。一个总体提升可能只来自容易题，而困难或高风险切片没有变化。

### 15.4 记录候选轨迹

保存每个候选的原文、抽取结果、归一化结果、分数、token、随机种子和失败原因。只保存最后投票结果，无法诊断多数错误、解析错误和 verifier 错误。

### 15.5 对照独立验证器

在代码任务中运行隐藏测试；在数学任务中使用独立计算器或符号检查；在文档问答中检查引用片段。比较 majority 和 verifier 的互相救援，才能知道采样和验证各自贡献。

## 16. 安全与可靠性

### 16.1 多采样可能放大危险候选

对普通问题，多采样增加的是正确和错误解法；对安全敏感问题，多采样也可能增加危险计划、越权动作或规避策略的覆盖率。不能把“多数投票”当作安全过滤器。

### 16.2 工具动作不能只投票

如果候选包含删除文件、转账或发送消息等工具动作，系统不能因为多数候选都建议该动作就自动执行。必须进行权限检查、参数校验、人工确认和可撤销控制。

### 16.3 解释和候选保留

候选轨迹可能包含用户隐私和未验证的危险细节。日志应按权限保存，用户界面应展示证据和必要说明，而不是默认把所有采样结果暴露出来。

### 16.4 失败状态要能表达

当候选分歧很大、parser 失败、verifier 互相矛盾或所有候选都不满足约束时，系统应该返回不确定或请求补充信息，而不是强行选一个答案。

## 17. 常见误区

### 误区一：采样次数越多越好

如果候选共享同一个错误，更多样本只会增加错误共识和成本。

### 误区二：temperature 越高越有创造力

高温度增加随机性，也增加格式错误和无关路径。推理任务需要合理的候选多样性，而不是无约束噪声。

### 误区三：不同答案字符串就是不同解法

42、42.0 和答案是 42 可能是同一个答案；不同文字也可能依赖同一个错误前提。必须同时分析答案等价类和关键步骤。

### 误区四：pass@k 就是线上成功率

pass@k 不包含候选识别能力。没有可靠 parser 或 verifier，正确候选可能永远不会被交付。

### 误区五：加权投票天然比多数投票强

加权投票依赖分数质量。未经校准的评分器可能把形式、长度或常见表达当成正确性。

### 误区六：自适应早停只会节省成本

错误的难度估计会让难题过早停止；过于依赖模型置信度会把共享错误误判成高确定性。

## 18. 小练习

### 练习一：计算多数概率

假设单条候选正确率为 0.6，路径近似独立，计算 K=1、3、5 时多数正确的理想化概率，并说明相关错误会如何改变结果。

### 练习二：实现答案归一化

写一个函数处理整数、小数、百分比、单位和中文数字，列出不能自动合并的边界样例。

### 练习三：比较 pass@k 与选择准确率

构造五道题，每题生成四个候选，记录正确候选数量、pass@2、majority 和 verifier 选择结果。

### 练习四：设计早停策略

要求系统先生成两个候选，只有答案冲突或验证失败时才继续采样。比较固定 K 和早停策略的平均 token、P95 延迟与高难度切片准确率。

### 练习五：构造 verifier rescue

构造一个多数答案错误、少数答案正确的代码题，设计隐藏测试，让 verifier 选中少数正确候选。

### 练习六：实验资料阅读

阅读 Self-Consistency 和 HumanEval 原始资料，记录它们如何定义候选、正确性、采样数量和评估指标，以及这些定义不能证明什么。

## 19. 本章总结

Self-consistency 是一种以多路径采样换取候选多样性的推理时计算方法。它的成功需要同时满足：单条候选有一定正确率，错误不完全相关，答案能稳定抽取和归一化，聚合器能够选出好候选，额外成本仍然可接受。

本章的关键关系是：

1. greedy 提供稳定性，但通常没有 self-consistency 所需的路径多样性。
2. temperature、top-k 和 top-p 改变 token 分布，不等于直接改变推理能力。
3. 答案多样性、轨迹多样性和错误独立性不是同一个指标。
4. 多数投票简单，但可能在多数错误时失败。
5. weighted vote 和 verifier rerank 可以利用候选质量分数，但受评分器偏差影响。
6. pass@k 衡量候选集合中是否存在正确答案，不等于最终交付成功率。
7. 候选数量增加会带来 token、延迟、验证和队列成本，必须观察边际收益。
8. 自适应采样和早停需要校准，并要保留不确定状态。
9. 高风险工具动作不能靠投票自动执行。

下一章进入 verifier 与 reward model，讨论如何判断候选是否满足任务约束，以及为什么一个评分器也可能被模型利用。

## 20. 资料索引

以下链接优先保留原始论文和官方实现文档。复现实验时应记录模型版本、tokenizer、采样参数、随机种子、候选数量、答案解析器和测试集。

1. Self-Consistency Improves Chain of Thought Reasoning in Language Models：<https://arxiv.org/abs/2203.11171>
2. Chain-of-Thought Prompting：<https://arxiv.org/abs/2201.11903>
3. HumanEval：<https://arxiv.org/abs/2107.03374>
4. The Curious Case of Neural Text Degeneration / nucleus sampling：<https://arxiv.org/abs/1904.09751>
5. Transformers text generation：<https://huggingface.co/docs/transformers/main/en/main_classes/text_generation>
6. Deep Learning for Text Generation with Nucleus Sampling：<https://arxiv.org/abs/1904.09751>
7. A Simple Method for Controlled Generation of Text：<https://arxiv.org/abs/1609.08144>

本章写作时已联网访问 Self-Consistency、HumanEval 和 Transformers 生成文档入口；部分论文入口受瞬时网络连接限制时，沿用此前已取得的原始页面状态，并没有把连接失败写成资料不存在。正文明确区分论文评估口径、框架参数语义、教学构造和目标系统实测。
