# 第 17 章 On-Policy Distillation：让学生在自己的分布上学会教师的行为

知识蒸馏常被描述成“让小模型模仿大模型”。这句话只说对了一半。教师在干净问题上生成的答案，通常属于教师自己的状态分布；学生上线时会使用自己的 token、自己的错误和自己的工具动作，走到一条不同的轨迹上。如果训练数据只包含教师走过的路径，学生一旦偏离示范，就可能不知道如何恢复。

On-policy distillation（OPD）把学生自己的 rollout 放回训练环路：学生先按照当前策略生成状态或轨迹，教师再在这些状态上提供概率分布、下一步动作、批注、候选答案或验证结果。学生学习的对象不再只是“教师从理想起点如何作答”，还包括“自己已经走错时，如何回到可验证的正确路径”。

这并不意味着教师永远正确，也不意味着 on-policy 天然优于离线蒸馏。学生可能走到教师不熟悉的状态，教师可能给出高置信度的错误建议，token 级监督可能在长轨迹后段变得不可靠，教师调用还会带来巨大的推理和数据安全成本。因此，OPD 的核心问题是状态分布、教师信号和独立验证三者如何对齐。

## 17.0 先理解离线蒸馏为什么会失效

设输入为 x，教师策略为 pi_T，学生策略为 pi_S。普通的离线蒸馏通常从教师或人工数据中得到前缀 s 和目标 a_T，然后最小化学生在这些固定状态上的损失：

~~~math
\mathcal{L}_{\mathrm{off}}
=\mathbb{E}_{s\sim d_T}
[-\log \pi_S(a_T\mid s)]
~~~

d_T 是教师示范产生的状态分布。这个训练目标可能很有效，因为教师示范往往质量高、格式稳定、容易批处理。但部署时学生从自己的历史开始生成，访问的是 d_S。一旦学生在某个位置选了教师不会选择的 token，后续前缀就不再属于 d_T。

这就是 exposure bias 的一个来源：训练时总看到理想前缀，推理时却必须面对自己生成的前缀。若每一步出现错误的概率都很小，长轨迹仍可能累积出明显的偏移。把每个位置的错误概率粗略记为 q_t，在相互独立的教学近似下，至少出现一次错误的概率为：

~~~math
P(\text{at least one error by }T)
=1-\prod_{t=1}^{T}(1-q_t)
~~~

真实语言生成中的错误并不独立，早期错误还会改变后续状态，所以这个公式不是能力定律。它的作用是说明：即便每个 token 的错误概率不大，轨迹足够长时，训练分布与部署分布之间的差异也不能忽略。

这个教学近似要求 `T` 为正整数，且每个 `q_t` 是同一条件定义下的概率，满足 `0\leq q_t\leq1`。若某一步来自工具超时、权限拒绝或不可判定任务，就不应把它硬塞进“模型出错”的 `q_t`；这些事件需要在状态分布与环境失败率中单独记录。

学生自生成的错误状态可能包括：选错文件、误解接口、重复调用工具、把空结果当成事实、在测试失败后继续重复同一 patch，或者在数学推导中使用了不成立的假设。教师只提供一条从干净输入开始的最终答案，不能直接教会学生处理这些状态。

## 17.1 OPD 解决什么，不能解决什么

on-policy 的“on”指的是监督状态由当前学生策略产生，而不是指所有教师数据都必须在线生成。一个实际系统可以混合三种来源：

| 数据来源 | 状态分布 | 适合的作用 |
| --- | --- | --- |
| 教师/人工示范 | 较干净的参考分布 | 建立基础能力和协议遵循 |
| 学生 rollout | 学生当前会访问的分布 | 纠正实际错误和恢复行为 |
| 反事实/对抗状态 | 人工构造的边界分布 | 扩展安全和罕见失败覆盖 |

OPD 主要减少 d_T 与 d_S 的状态分布差异。它不自动解决教师错误、verifier 漏洞、学生探索不足、隐私泄漏、工具权限和任务本身不可判定等问题。学生如果只生成最保守的短答案，虽然状态分布看起来稳定，实际可能是逃避了探索；学生如果反复生成无意义长轨迹，教师也会在错误分布上浪费预算。

可以把 OPD 的收益拆成三个问题：

1. 学生是否真的访问了需要纠正的状态；
2. 教师在这些状态上是否提供了可靠且适合学生能力的信号；
3. 学生是否把局部纠正迁移到了未见过的新状态和真实工作流。

只有第三个问题在独立评估中成立，前两个训练现象才有能力意义。

## 17.2 一条样本的状态契约

OPD 的样本不是孤立的 prompt 和 answer，而是一个带环境上下文的状态转移记录。可以把一个训练单元写成：

~~~math
u=(x,s_t,a_t,o_{t+1},y,u_T,v,\rho_e)
~~~

x 是任务输入，s_t 是学生在第 t 步看到的状态，a_t 是学生采取的 token 或工具动作，o_{t+1} 是环境返回的 observation，y 是最终 artifact，u_T 是教师信号，v 是 verifier 结果，rho_e 是环境和协议的版本信息。

记录 s_t 时不能只保存文本前缀。代码 Agent 还需要保存工作区 diff、当前分支、测试输出和工具权限；RAG Agent 需要保存检索候选、文档版本和过滤结果；数据库任务需要保存 schema、数据快照和事务状态。相同的文字在不同 workspace 中可能代表完全不同的状态。

一条可复查样本至少应该包含：

| 字段 | 说明 |
| --- | --- |
| task_id 与输入版本 | 能定位任务和题目修订 |
| student_revision | 学生 checkpoint、tokenizer 和采样参数 |
| state | 文本前缀、工具 observation 和外部状态摘要 |
| teacher_revision | 教师模型、系统 prompt 和工具版本 |
| teacher_signal | logits、动作、批注、候选或最终 artifact |
| verifier_revision | 判定器、测试集和数据快照 |
| outcome | 成功、失败、超时、拒绝或不可判定 |
| resource_usage | token、延迟、工具调用和存储 |

缺少版本信息时，旧建议可能被误用于新环境；缺少失败状态时，学生只能学习教师的最终答案；缺少资源信息时，训练中看似有效的策略可能在线上完全不可用。

## 17.3 蒸馏目标的层次

### 17.3.1 Token-level 目标

如果学生和教师使用同一词表，教师可以在学生访问的状态上输出完整的 token 分布 p_T，学生分布为 p_S。标准的温度蒸馏交叉熵可以写成：

~~~math
\mathcal{L}_{\mathrm{KD}}
=-\frac{1}{Z}\sum_t m_t T^2
\sum_{v\in\mathcal{V}}
p_T^{(T)}(v\mid s_t)\log p_S^{(T)}(v\mid s_t)
~~~

m_t 是有效监督 mask，Z=\sum_t m_t 是归一化分母，V 是词表，T 是蒸馏温度，p^(T) 表示使用温度后的 softmax 分布。T^2 是常见的梯度尺度补偿项；是否保留、如何与 hard label 混合，取决于具体实现。

这个 token-level 写法要求教师和学生在同一状态上使用可对应的词表与特殊 token 协议，温度 `T>0`，并且有效监督数 `Z=\sum_t m_t>0`。若样本经截断或 chat template 后没有剩余的 assistant token，它不是“零蒸馏损失的成功样本”，而是数据协议异常；应跳过并统计空监督率。完整 softmax 的每一项还必须是有限概率，和为一，不能把缺失的教师 logits 用零填充后继续训练。

温度较高时，教师低概率 token 的相对关系更容易被学生看到；温度较低时，监督更接近硬标签，教师的最高概率动作更突出。温度不是“越大越有知识”的开关，过高会把不确定性和错误一同扩散给学生。

### 17.3.2 Reverse-KL 与 sampled token 信号

OPD 文献中常出现 student-to-teacher 的 reverse KL：

~~~math
D_{\mathrm{KL}}(p_S\|p_T)
=\sum_{v}p_S(v\mid s)
\log\frac{p_S(v\mid s)}{p_T(v\mid s)}
~~~

学生从自己的策略采样 token 时，可以使用教师和学生在该 token 上的 log-probability 差异作为一个 token-level 信号：

~~~math
r_t^{\mathrm{KD}}
=\log p_T(a_t\mid s_t)-\log p_S(a_t\mid s_t)
~~~

对 a_t 按学生分布取期望时，这个信号与最小化 D_KL(p_S\|p_T) 的方向相关。它和“教师在教师分布上给学生做交叉熵”不是同一个估计器，不能把两种方向的 KL 混写。

reverse KL 要求在学生赋予正概率的动作上教师也有正概率；否则 `\log p_T(a_t\mid s_t)` 和 KL 都没有有限值。有限精度实现通常使用稳定的 `log_softmax`、记录截断 top-k 的剩余质量，并对教师拒答、服务失败或版本不匹配标为 `unknown`，而不是把缺失 log-prob 当成一个很大的负 reward。只有两端概率来自同一状态快照、同一动作定义时，差值才具有蒸馏语义。

reverse KL 往往倾向于追逐教师的高概率 mode；当教师分布本身有多个合理答案或高熵时，学生可能过早收缩到单一表达。forward KL 更关心覆盖教师支持的多种可能性，但可能让学生保留更多低质量模式。选择哪一种，应该结合任务、教师熵、verifier 和独立多样性评估。

### 17.3.3 序列、动作和 artifact 目标

教师不一定要输出每个 token 的 logits。可用的目标还包括：

| 教师信号 | 学到的行为 | 主要代价或风险 |
| --- | --- | --- |
| 完整 token 分布 | 细粒度语言和不确定性 | 推理、存储和词表通信成本高 |
| top-k 分布 | 近似局部选择 | 截断会丢失尾部概率质量 |
| 下一步动作 | 工具规划和状态转换 | 需要可靠的动作 schema |
| 错误类别或批注 | 诊断与恢复 | 批注可能主观或风格化 |
| 最终 artifact | 可交付结果 | 缺少如何恢复的过程信息 |
| verifier 结果 | 成功与失败边界 | 反馈稀疏，依赖检查器 |

代码 Agent 更需要“下一步查看哪个文件”“测试失败属于哪类”“是否撤销 patch”；数学任务可以蒸馏最终可验证答案和关键中间式；敏感任务不应为了提高 token loss，就无条件复制隐藏思维链、用户数据或工具凭证。

## 17.4 On-policy 训练循环

一个最小的循环是：

~~~text
student rollout -> collect states and observations
       -> select states for teacher query
       -> teacher labels, scores, or proposes actions
       -> verifier checks target and artifact
       -> student update on trusted signals
       -> independent evaluation on new rollouts
~~~

学生先生成状态并不等于所有状态都应该交给教师。教师调用可以由错误、置信度、工具风险、状态转折和任务价值触发。已经通过测试且只需要格式化输出的状态，没有必要支付一次完整教师调用；准备执行高风险动作或连续两次出现同类失败时，教师或人工介入的价值更高。

教师调用触发器最好基于可观察量，而不是模型自己写出的“我不确定”。可以结合 token entropy、候选分歧、工具返回异常、verifier 失败类别和历史重复率。模型声明“确定”只能是一个特征，不能成为权限依据。

学生 rollout 应保留完整失败上下文。只保存教师修复后的最终 patch，会让训练集看起来很干净，却无法训练恢复行为；只保存错误文本而不保存工具返回，又会让学生无法区分代码错误、环境错误和权限拒绝。

## 17.5 教师信号的可靠性

教师输出不是标签真理。一个强教师也可能在开放事实任务中产生幻觉，在陌生代码仓库中误判依赖，在学生偏离太远时给出低质量 token 分布。可靠性至少要从四个维度评价：

1. 教师是否看到了完成任务所需的证据；
2. 教师输出是否满足当前工具和协议；
3. verifier 或人工抽样是否支持该输出；
4. 教师建议是否适合学生的能力和当前状态。

可以用一个示意权重表示“某个教师信号是否进入高权重监督”：

~~~math
w_t=m_t\cdot q_T(s_t)\cdot q_V(y_t)
\cdot\exp\left(-\lambda\sum_{j<t}D_j\right)
~~~

m_t 表示基本有效 mask，q_T 是教师置信或校准质量，q_V 是 verifier 对当前目标的支持，D_j 是前面位置的分布差异，lambda 控制偏离累积的衰减。这个公式是教学性的组合，不是所有 OPD 方法的标准定义；它表达的工程直觉是：无效 token、低可信教师输出和远离教师支持区域的后段 token 不应拥有同样权重。

使用这类权重前，需要声明 `m_t\in\{0,1\}`、`q_T,q_V\in[0,1]`、`D_j\geq0`、`\lambda\geq0`，并保证参与求和的量有限。教师无法回答、verifier 超时或状态已被撤销时，`q_T` 或 `q_V` 应为未知而不是主观补成 `0` 或 `1`；这类轨迹应从当前更新中隔离，同时作为数据或系统健康信号保留。

对安全动作，权重不能替代执行器拒绝。教师说“可以写文件”不等于 student 获得写权限，教师生成的 SQL 不等于事务可以提交，教师建议访问某个 URL 也不等于网络策略允许访问。权限必须在 environment executor 层判定。

如果教师的隐藏 reasoning 含敏感信息或未验证假设，不应无条件蒸馏完整思维链。更可审计的目标是最终可验证 artifact、工具 schema、证据引用、错误类别、安全拒答和结构化修复动作。需要展示推理时，也应区分用于训练的内部信号和可对外输出的解释。

## 17.6 一个代码 Agent 的 on-policy 例子

设学生接到 issue 后经历以下状态：

~~~text
s_0: 读取 issue，猜测 bug 在 parser.py
s_1: 修改 parser.py，公开测试失败
s_2: 看到错误消息，仍重复同一 patch
s_3: 读取 tokenizer.py，发现真实边界条件
s_4: 生成修复，公开和隐藏测试通过
~~~

如果训练数据只有教师从 s_0 直接生成的正确 patch，学生主要学习的是结果模板。它不一定知道 s_1 失败后要查看什么，也不知道 s_2 的重复动作已经没有信息增益。

在 OPD 中，教师可以针对 s_1 给出“先分类测试失败，不要立即改代码”，针对 s_2 给出“比较 parser.py 与 tokenizer.py 的边界契约”，针对 s_3 给出一个可验证的测试添加建议。最终 reward 仍应来自独立测试、回归测试和副作用检查；教师批注只是更密的中间信号。

一个好的训练样本应同时保留：

| 内容 | 作用 |
| --- | --- |
| 学生错误动作 | 让模型看到需要纠正的状态 |
| 环境 observation | 区分知识错误和工具错误 |
| 教师局部建议 | 学习下一步恢复动作 |
| 修复前后 diff | 连接动作与 artifact |
| 测试结果 | 提供可执行证据 |
| 权限和资源结果 | 防止把副作用当能力 |

教师如果直接重写整个答案，可能降低短期 loss，却隐藏了错误定位过程。局部建议更接近恢复训练，但也可能不够完整；两种目标可以按状态阶段混合，而不是全程使用同一种粒度。

## 17.7 教师调用策略与成本

教师调用昂贵时，不能只比较学生参数量。设每个任务平均产生 n 个学生状态，每次教师调用成本为 c_T，学生状态采样和执行成本为 c_S，教师触发比例为 p_T，单位任务成本可粗略写成：

~~~math
C_{\mathrm{task}}=c_S+n\,p_T\,c_T
~~~

若把学生 rollout、工具和存储的固定成本分别记为 C_S，也可以写成：

~~~math
C_{\mathrm{total}}
=N_S C_S+N_T C_T+C_{\mathrm{storage}}+C_{\mathrm{verify}}
~~~

N_S 和 N_T 分别是学生和教师 token/调用的计数，C_T 往往远大于一个学生 token 的成本。缓存可以降低 N_T，但会引入一致性和隐私成本。

更有意义的指标是单位成功成本：

~~~math
C_{\mathrm{success}}
=\frac{C_{\mathrm{task}}}{P_{\mathrm{success}}},
\qquad P_{\mathrm{success}}>0
~~~

`P_success` 是独立任务上的真正成功率，要求 `0<P_success\leq1`；`C_task` 要在相同货币、时间窗口和分摊口径下计量。若独立评估没有任何成功任务，单位成功成本没有定义，不能用 `epsilon` 伪造一个极大但看似可比较的数字。应报告零成功、样本数和置信区间，再先解决成功定义或能力问题。若 OPD 让学生成功率从 0.45 提高到 0.55，但教师调用量增加三倍，单位成功成本可能上升；若训练后线上可以少调用教师、少重试且延迟下降，收益才更可信。

触发策略可以比较四个条件：所有状态调用教师、只在困难状态调用、复用离线缓存、完全不用教师。每个条件都按任务难度、错误类型、学生长度和工具状态分桶，否则平均值会掩盖教师只在简单题上有效的事实。

## 17.8 教师缓存、版本与隐私

教师结果可以缓存，但相同文本不代表相同状态。代码仓库、依赖、工具 schema、数据快照和权限域改变后，旧建议可能已经失效。

一个缓存键至少应绑定：

~~~text
student_state_hash
student_revision
teacher_revision
tokenizer_and_template_revision
environment_revision
verifier_revision
permission_domain
sampling_configuration
~~~

缓存命中只能减少重复计算，不能绕过权限检查、数据删除请求或最新 verifier。教师更新后，旧轨迹可以降权、重新验证或停止用于训练；不能静默把新旧教师的 logits 混在一个分布里。

教师看到的状态可能包含用户代码、个人资料、工具返回、未提交的 secret 和跨租户数据。蒸馏系统要限制输入范围，过滤凭证，记录数据用途，并规定教师结果是否允许进入持久化训练集。如果教师通过外部 API 调用，隐私和隔离不能由 prompt 口头保证，必须由传输、存储和访问控制实现。

恢复训练时还要保存建议的来源、适用状态和验证结果。学生重启后不能把旧 workspace 的建议当成新 workspace 的事实；否则 cache 会把环境漂移隐藏起来。

## 17.9 OPD 的数学视角：从 KL 到 token reward

在完整分布可计算的情况下，可以直接最小化学生与教师的 KL；在学生 rollout 上只观察了已采样 token 时，常见做法是使用 sampled log-probability 信号。

教师和学生在状态 s_t 上的 reverse KL 为：

~~~math
D_{\mathrm{KL}}(p_S\|p_T)
=\sum_{v\in\mathcal{V}}p_S(v\mid s_t)
\log\frac{p_S(v\mid s_t)}{p_T(v\mid s_t)}
~~~

学生采样 a_t 后，教师与学生的 log-probability 差可以写成：

~~~math
r_t=\log p_T(a_t\mid s_t)-\log p_S(a_t\mid s_t)
~~~

取 a_t~p_S 的期望时，r_t 与负的 reverse KL 相关，因此可以把它看作一种 token-level reward。它不是独立的任务正确性 reward：教师高概率不保证答案正确，教师低概率也不保证学生答案错误。

如果使用序列长度为 T 的轨迹，一个简化的 OPD 目标是：

~~~math
\mathcal{L}_{\mathrm{OPD}}
=-\frac{1}{Z}\sum_{t=1}^{T}m_t w_t r_t
~~~

m_t 是有效 token mask，w_t 是可信度、位置或 verifier 条件权重，Z 是归一化分母。r_t 的符号、KL 方向、是否 stop-gradient 以及是否与 hard label 混合，都必须根据实现明确记录。

这里同样要求 `Z=\sum_tm_t>0`，有效位置的 `w_t` 与 `r_t` 都有限，并且 `w_t` 的方向已在 schema 中固定。例如“verifier 支持度越高权重越大”和“风险越高权重越小”不能共用一个未说明方向的字段。若仅部分 token 有教师分数，mask、截断策略和未标注位置是否参与 hard-label loss 都要与 checkpoint 一起保存。

当 teacher/student tokenizer 不同，p_T(v) 与 p_S(v) 没有同一个词表索引，不能直接做 token-level KL。可选办法是使用共享 tokenizer、把教师输出转换为文本序列、在字符/词片段边界上对齐，或退回到动作、artifact 和 verifier 级别的目标。把不同 tokenizer 的 logits 按数组下标相乘，是数值上能运行但语义上错误的实现。

## 17.10 为什么长轨迹后段的教师信号更危险

学生越往后生成，前缀越可能偏离教师支持区域。教师仍然可以给出一个概率分布，但这个分布未必是在“教师会到达的状态”上校准过的。长轨迹后段的低质量信号可能造成负迁移：学生本来有一个正确但不同的路径，却被教师的高概率 mode 强行拉走。

可以用累计分布差异描述这种风险：

~~~math
D_t^{\mathrm{cum}}=\sum_{j=1}^{t-1}
D_{\mathrm{KL}}(p_S(\cdot\mid s_j)\|p_T(\cdot\mid s_j))
~~~

这不是线上必须计算的唯一量，但它表达了一个事实：第 t 个 token 的监督质量不仅取决于当前 token，还取决于前缀已经偏离了多少。有效监督权重可以采用示意形式：

~~~math
w_t=\mathbf{1}[D_t^{\mathrm{cum}}\leq\delta]
\cdot q_T(s_t)\cdot q_V(s_t)
~~~

delta 是允许的偏离范围，q_T 是教师置信或校准分数，q_V 是 verifier 对状态/结果的支持。这里的乘法表示只有可信区域才参与高权重学习；实际工作也可以使用连续衰减，而不是直接截断。

该示意要求 `\delta\geq0`，累计 KL 有限，且 `q_T,q_V` 来自同版本的、范围明确的检查。没有 verifier 结论不等于 verifier 不支持，也不等于可以安全纳入可信区域；应报告未判定状态比例，避免过滤器仅在容易样本上显得可靠。

2026 年的公开预印本中，位置偏差研究报告了前段 token 与后段 token 的监督质量差异，并提出用累计差异给 token 加权的 IW-OPD 思路。这个方向与工程直觉一致，但具体提升幅度依赖模型、任务、长度和教师配置，不能把一个预印本的 benchmark 增益当作普遍定律。

## 17.11 稳定 OPD 的几种思路

### 17.11.1 可信区域与信赖域

当 student 和 teacher 分布差距很大时，教师在 student-generated token 上的梯度估计可能成为异常值。TrOPD 一类工作把 OPD 限制在教师信号较可靠的区域，对 outlier 使用裁剪、mask 或 forward-KL 等替代，并使用教师前缀进行 off-policy 引导。它的核心思想不是“所有学生状态都强行模仿”，而是让探索逐渐回到教师能够可靠指导的区域。

这类方法的取舍是：过滤太少，训练容易发散或负迁移；过滤太多，学生只学习熟悉的简单区域，on-policy 的价值下降。需要同时报告被过滤的状态比例、过滤前后成功率和状态分布变化。

### 17.11.2 置信度与熵

教师熵高时，教师可能认为多个 token 都合理。只选择最高概率 token 会过早收缩多样性；在高熵位置使用更覆盖性的目标，可能比强行做 mode-seeking 模仿更稳。

教师 token 熵为：

~~~math
H_T(s_t)=-\sum_{v}p_T(v\mid s_t)\log p_T(v\mid s_t)
~~~

Entropy-Aware OPD 一类研究讨论了在教师高熵位置混合 forward KL、在低熵位置保持更精确模仿的思路。实践中应观察学生熵、pass@k、多样性和最终正确率；只看 top-1 对齐下降，可能错过学生已经丢失了合理候选的事实。

### 17.11.3 奖励裁剪、动态采样与基线

REOPOLD 把教师-学生 likelihood ratio 看作 token reward，并通过混合式 reward clipping、基于熵的动态采样和探索到精炼的训练策略稳定 OPD。vOPD 则从控制变量角度处理单样本估计的方差，尝试利用已经计算的 KL 信息构造 token baseline。

这些方法都说明一个共同问题：OPD 不只是一个交叉熵损失，而是带有 on-policy 采样、估计方差和策略更新性质的训练过程。稳定化方法要记录梯度范数、被裁剪 token 比例、教师熵、student/teacher KL 和独立任务结果，不能只报告蒸馏 loss。

## 17.12 多 rollout 与 verifier 的结合

如果同一个输入采样多个学生 rollout，只对每条轨迹独立蒸馏，可能浪费了组内成功和失败的对比信息。MOPD 一类研究让教师同时看到同一问题的成功和失败 peer rollout，再构造更有区分度的教师信号。

设同一任务的学生轨迹为 tau_1,...,tau_G，每条有 verifier 结果 v_i。教师信号可以条件化在成功集合和失败集合上：

~~~math
u_i=\mathrm{Teacher}(s_i,
\{\tau_j:v_j=1\},
\{\tau_j:v_j=0\})
~~~

这里的 u_i 不是一个固定的概率公式，而是表示教师在同组对照上下文中给出的局部建议。成功轨迹提供可模仿的证据，失败轨迹提供需要避免的反例。它的价值取决于 verifier 质量和 peer rollout 是否真的有差异；如果所有轨迹共享同一漏洞，组内对比会放大共同错误。

Reward-Gated OPD 则把 verifier 反馈用于决定哪些教师 logits 可以被信任：正确且被独立验证的轨迹可以提高教师监督权重，未验证或相互冲突的信号降低权重。这个思路把稀疏结果奖励和密集 token 监督连接起来，但也把 verifier 的偏差带进蒸馏系统，必须做独立评估。

## 17.13 数据和环境的安全边界

教师看到的是学生当前状态，可能包含用户代码、个人资料、未提交凭证、工具返回和跨租户信息。把整条 workspace 直接发送给外部教师 API，会产生数据出境、日志留存和权限扩散风险。

更安全的做法是先构造最小必要状态：

| 状态内容 | 是否默认发送给教师 | 需要的条件 |
| --- | --- | --- |
| 任务描述 | 可以 | 脱敏并确认用途 |
| 相关文件片段 | 视任务而定 | 路径和 secret 过滤 |
| 完整 workspace | 不应默认 | 明确租户和留存策略 |
| 工具错误摘要 | 通常可用 | 去除凭证和内部地址 |
| 隐藏测试与答案 | 不应发送 | 独立保密 |
| 用户个人数据 | 谨慎 | 合法性、最小化和访问控制 |

教师建议即使来自可信模型，也不能自动获得工具权限。执行器仍应检查动作 schema、账户权限、路径和资源。教师的建议、学生的动作和环境结果需要分开记录，才能在事故后判断是谁产生了越权。

对隐藏 reasoning 的处理要尤其谨慎。训练可以使用内部 token 信号，但不应把包含隐私、未验证假设或第三方机密的完整思维链无条件写入学生数据。可优先蒸馏最终 artifact、证据引用、错误类别、安全拒答和结构化动作。

## 17.14 评估：学生是否真的学会恢复

OPD 最容易被错误评估成“学生和教师越来越像”。但目标不是最大化风格相似度，而是让学生在自己的部署分布中完成更多任务、少走弯路并且少依赖教师。

至少比较四个基线：

1. SFT 或普通行为克隆；
2. 在固定教师轨迹上的 offline KD；
3. 在学生状态上的 on-policy KD；
4. 教师本身和不使用教师的学生。

测试集合要包括学生常见错误、教师未见的新状态、工具失败、长轨迹、分布外任务、权限撤销和安全边界。代码 Agent 还应包含“第一次 patch 错误但第二次恢复”的样本，因为这正是 OPD 与干净示范之间的区别。

建议报告：

| 指标 | 含义 |
| --- | --- |
| first-attempt success | 第一次尝试是否完成任务 |
| recovery success | 出现指定错误后能否恢复 |
| recovery steps | 从错误到成功需要多少步 |
| hidden-test pass | 未公开边界是否通过 |
| tool-call validity | 工具动作和参数是否有效 |
| unsafe-action rate | 越权、泄露和副作用比例 |
| teacher calls per task | 部署依赖和训练成本 |
| student tokens per success | 学生自身推理成本 |
| unit successful-task cost | 达成真实成功的综合成本 |

如果学生在教师提供建议的 replay 数据上变好，但在新工具版本和新错误类型上不变，说明它可能只记住了教师格式。平均 token KL 下降也不等于任务成功率提高，需要按错误类型、序列位置和环境状态切片。

## 17.15 成本收益的计算

学生 rollout、教师推理、验证和存储构成 OPD 的总成本：

~~~math
C_{\mathrm{total}}
=N_S C_S+N_T C_T+C_{\mathrm{storage}}+C_{\mathrm{verify}}+C_{\mathrm{retry}}
~~~

N_S 和 N_T 可以按 token、调用次数或 GPU 时间统计，C_T 往往远高于学生单 token 成本。缓存可以降低 N_T，但会引入一致性和隐私成本。

不论采用哪种计数单位，`N_S,N_T` 和各成本项都应非负，且所有项必须换算到相同的时间窗口、货币和资源口径。token、调用次数与 GPU 小时不能直接相加；前式只是将每一项已经换算成成本后的账本，而不是把不同单位的原始计数相加。

若独立评估真正成功的任务数为 N_success，单位成功任务成本为：

~~~math
C_{\mathrm{unit\ success}}
=\frac{C_{\mathrm{training}}+C_{\mathrm{evaluation}}+C_{\mathrm{operations}}}
       {N_{\mathrm{success}}}
~~~

该量要求 `N_success>0`，成功要由独立任务结果、权限、预算和必要人工复核共同定义。零成功时应报告“单位成功成本未定义”和失败类别，而不是零成本或任意平滑值。教师调用降低一次错误恢复的步数，可能提高一次任务的教师成本却降低总成本；反过来，学生在训练中依赖大量教师建议，线上没有教师时可能退化。成本分析必须在和部署相同的教师可用性、工具版本和重试预算下进行。

## 17.16 一个轻量的教师调用选择 demo

下面的代码只演示选择逻辑，不模拟模型 logits。它将“低置信度、连续测试失败、高风险动作”作为教师调用触发条件，并生成一个可复查的 cache key。真实系统还应加入数据脱敏、权限校验和环境 revision。

~~~python
from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class AgentState:
    task_id: str
    workspace_revision: str
    text: str
    confidence: float
    failed_tests: int
    proposed_action: str
    risk_level: str


class ContractError(ValueError):
    """A state or cache-input contract is not satisfied."""


def require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be non-empty text")
    return value


def require_probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a finite probability")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ContractError(f"{name} must be between zero and one")
    return value


def require_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return value


def validate_state(state: AgentState) -> None:
    if not isinstance(state, AgentState):
        raise ContractError("state must be an AgentState")
    require_text(state.task_id, "task_id")
    require_text(state.workspace_revision, "workspace_revision")
    require_text(state.text, "state text")
    require_probability(state.confidence, "confidence")
    require_nonnegative_int(state.failed_tests, "failed_tests")
    if state.proposed_action not in {"read", "write", "execute", "delete"}:
        raise ContractError("proposed_action is not in the action schema")
    if state.risk_level not in {"low", "medium", "high", "critical"}:
        raise ContractError("risk_level is not in the risk schema")


def canonical_json(value: object, name: str) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ContractError(f"{name} must be finite JSON data") from error


def should_query_teacher(state: AgentState) -> tuple[bool, str]:
    validate_state(state)
    risky_actions = {"write", "execute", "delete"}
    if state.proposed_action in risky_actions and state.risk_level in {"high", "critical"}:
        return True, "high_risk_action"
    if state.failed_tests >= 2:
        return True, "repeated_test_failure"
    if state.confidence < 0.45:
        return True, "low_confidence"
    return False, "ordinary_state"


def state_cache_key(
    state: AgentState,
    student_revision: str,
    teacher_revision: str,
    tokenizer_and_template_revision: str,
    environment_revision: str,
    verifier_revision: str,
    permission_domain: str,
    sampling_configuration: dict[str, object],
) -> str:
    validate_state(state)
    if not isinstance(sampling_configuration, dict) or not sampling_configuration:
        raise ContractError("sampling_configuration must be a non-empty mapping")
    payload = {
        "task_id": state.task_id,
        "workspace_revision": state.workspace_revision,
        "text": state.text,
        "student_revision": student_revision,
        "teacher_revision": teacher_revision,
        "tokenizer_and_template_revision": tokenizer_and_template_revision,
        "environment_revision": environment_revision,
        "verifier_revision": verifier_revision,
        "permission_domain": permission_domain,
        "sampling_configuration": sampling_configuration,
    }
    for name in (
        "student_revision",
        "teacher_revision",
        "tokenizer_and_template_revision",
        "environment_revision",
        "verifier_revision",
        "permission_domain",
    ):
        require_text(payload[name], name)
    encoded = canonical_json(payload, "cache payload")
    return hashlib.sha256(encoded).hexdigest()


states = [
    AgentState("bug-1", "ws-7", "read parser", 0.91, 0, "read", "low"),
    AgentState("bug-1", "ws-7", "retry patch", 0.72, 2, "write", "medium"),
    AgentState("bug-2", "ws-2", "execute migration", 0.88, 0, "write", "high"),
]

for state in states:
    query, reason = should_query_teacher(state)
    key = state_cache_key(
        state,
        "student-3",
        "teacher-9",
        "tokenizer-2/template-4",
        "sandbox-4",
        "verifier-5",
        "tenant-a-readonly",
        {"temperature": 0.7, "top_p": 0.95, "max_tokens": 2048},
    )
    print(state.task_id, query, reason, key[:12])


def expect_contract_error(label: str, action) -> None:
    try:
        action()
    except ContractError as error:
        print(f"{label}: {error}")
    else:
        raise AssertionError(f"{label} should have failed")


expect_contract_error(
    "confidence out of range",
    lambda: should_query_teacher(
        AgentState("bad", "ws-1", "state", 1.5, 0, "read", "low")
    ),
)
expect_contract_error(
    "negative failed tests",
    lambda: should_query_teacher(
        AgentState("bad", "ws-1", "state", 0.5, -1, "read", "low")
    ),
)
expect_contract_error(
    "non-finite sampling",
    lambda: state_cache_key(
        states[0], "student-3", "teacher-9", "tokenizer-2/template-4",
        "sandbox-4", "verifier-5", "tenant-a-readonly", {"temperature": float("nan")}
    ),
)
~~~

这个 demo 有三个边界。第一，`confidence` 已被限制在 `[0,1]`，但它只是触发特征，不是权限判断；真正的 write 动作仍要由执行器拒绝或要求独立授权。第二，cache key 同时绑定 workspace、学生/教师、tokenizer/template、环境、verifier、权限域和采样版本，任何字段缺失都可能把旧建议错用于新状态。第三，缓存 payload 必须是有限 JSON；`NaN`、缺失版本和非法状态会在进入教师调用前失败。缓存命中仍不能绕过最新的安全检查、数据删除请求或人工审批。

## 17.17 常见失败与排查

| 现象 | 可能原因 | 优先修复方向 |
| --- | --- | --- |
| 学生在 replay 上变好，真实状态不变 | 状态采样太窄或 teacher 只看干净前缀 | 增加学生失败、长轨迹和工具异常 |
| 学生输出越来越像教师但正确率下降 | teacher 误导、mode collapse 或目标方向错误 | 检查 verifier、teacher 熵和 KL 方向 |
| 后段 token 的 loss 降不下来 | 前缀偏离、长轨迹信号退化 | 位置加权、可信区域或截断消融 |
| 教师调用比例越来越高 | 学生没有学会恢复、触发器过宽 | 分析触发原因并按收益重新路由 |
| 学生变得过于保守 | 只训练失败拒答或过滤掉探索状态 | 加入正向多样性和反事实样本 |
| 工具成功率提高但安全事件增加 | teacher 建议没有经过执行器约束 | 将权限检查放在 executor 层 |
| cache 命中后结果漂移 | key 缺少环境、模板或 verifier 版本 | 扩充 key 并重新验证旧缓存 |
| 多 seed 结果差距很大 | 单样本梯度方差、教师分布不匹配 | 采用基线、裁剪、混合目标和更稳定采样 |

排查时不要只读蒸馏 loss。应抽取正常、低置信、失败、超时、拒绝和高成本轨迹，重放教师和 verifier，再按状态位置和错误类别分桶。OPD 的主要价值在错误恢复，恰恰最容易被平均 token 指标掩盖。

## 17.18 训练阶段如何组合

在实际项目中，OPD 通常不是第一步。一个相对清晰的阶段组合是：

1. SFT 建立语言能力、格式协议、工具 schema 和基本安全边界；
2. offline KD 或高质量人工轨迹建立初始任务能力；
3. on-policy correction 采样学生自己的失败状态，获取局部教师信号；
4. verifier 或人工检查教师目标和最终 artifact；
5. independent evaluation 检查新题面、新工具和真实成本；
6. regression 与安全评估确认通用能力没有被局部蒸馏破坏。

每一阶段都要保存学生 rollout 分布。若 on-policy 阶段 reward 上升，但状态越来越短、拒答越来越多、探索熵越来越低，可能是模型学会了避开困难状态，而不是学会恢复。若教师调用减少但独立成功率同步下降，也不能把“少调用”解释成蒸馏成功。

## 17.19 学习目标的冲突

教师 imitation、任务 outcome、工具安全、长度和部署成本可能指向不同方向。例如教师喜欢完整解释，线上要求短答案；教师建议尝试多个工具，线上权限只允许一个只读查询；学生的独特解法能通过 verifier，但与教师高概率 mode 不同。

可以先用诊断向量记录冲突，而不是马上把它们压成一个总分：

~~~math
g(\tau)=
(g_{\mathrm{teacher\_fit}},g_{\mathrm{outcome}},g_{\mathrm{recovery}},
 g_{\mathrm{safety}},g_{\mathrm{cost}},g_{\mathrm{diversity}})
~~~

教师拟合提高但 outcome 不提高，说明监督目标不对；outcome 提高但 diversity 下降，可能出现 mode collapse；恢复提高但安全下降，说明工具动作和策略能力之间缺少执行器约束。只有理解这些冲突，才有理由选择权重、mask 或分阶段训练。

对于高风险动作，g_safety 不是可以与其他分量平均的普通指标。执行器拒绝越权动作，训练日志记录它；教师信号只能帮助学生学会在允许范围内选择动作。

## 17.20 研究进展如何读

截至 2026 年 8 月，OPD 的公开研究正在从“学生自生成状态上做教师监督”扩展到稳定化、可信区域、位置加权、熵感知、多 rollout 对比和 reward-gated supervision。它们共同关注的不是一个新的蒸馏名词，而是同一个统计问题：学生访问的状态是否仍处在教师可可靠指导的支持区域。

从论文摘要可以确认的方法主张，与当前工程可以直接采用的结论，需要分开：

| 证据 | 可以说什么 | 不能直接说什么 |
| --- | --- | --- |
| GKD 论文 | 学生自生成序列和教师反馈能缓解序列分布错位 | 任何 student 都会稳定恢复 |
| REOPOLD 预印本 | 提出裁剪、动态采样和探索到精炼的稳定化方案 | 已成为通用工业 recipe |
| TrOPD 预印本 | 讨论可靠区域、outlier 处理和教师前缀引导 | 信赖域阈值可以跨任务复用 |
| 位置偏差研究 | 长 rollout 中后段信号可能退化 | 所有模型都必须只训练前 30% |
| Entropy-Aware OPD | 教师熵可能影响 reverse/forward KL 取舍 | 高熵位置一定需要同一种混合公式 |
| RG-OPD/MOPD | verifier 或 peer rollout 可帮助筛选教师信号 | verifier/peer 质量不重要 |

新方法应先做小规模复现和消融：固定学生、教师、任务、采样和 verifier，只改变一个稳定化组件；报告 teacher calls、过滤率、位置权重、KL、独立成功和单位成本。否则多个改动一起上线，无法知道收益来自目标、数据还是采样器。

## 17.21 常见误区与诊断路径

OPD 的损失下降很容易被误读成“学生已经学会了教师的能力”。真正需要追踪的是状态分布、教师信号、任务结果和线上约束之间是否形成了因果链。下面几个误区经常同时出现，诊断时应逐一拆开，而不是用一个平均蒸馏 loss 作结论。

### 17.21.1 学生生成状态不等于训练始终是 on-policy

“on-policy”描述的是状态或动作由当前行为策略产生，而不是数据集只要包含学生文本就永远是 on-policy。学生 checkpoint 更新后，旧 rollout 仍然来自旧策略；如果把它们长期重复训练，状态分布就会逐渐变成 replay 分布。于是，一个名义上的 OPD 管线实际上可能混合了当前策略、旧学生策略和教师示范三种来源。

设第 k 轮采样时的学生策略为 pi_{S,k}，第 k+1 轮更新使用的数据来自多个历史版本，则训练分布更接近：

~~~math
d_{\mathrm{train}}(s)
=\sum_{k=0}^{K}\alpha_k d_{\pi_{S,k}}(s),
\qquad
\sum_{k=0}^{K}\alpha_k=1.
~~~

alpha_k 是各版本 rollout 的混合权重。若当前策略与旧策略差距很大，教师信号和学生 log-probability 可能已经不再对应同一个行为分布。工程上要为每条轨迹记录 behavior policy revision、生成时间、采样配置和环境 revision，并决定旧数据是丢弃、降权、重新生成，还是用明确的 importance ratio 进行校正。不要因为样本中的前缀由学生写出，就跳过这一步。

这个混合写法要求轨迹集合非空，`\alpha_k\geq0`，并且每个正权重分量都能追溯到实际 policy revision。某个旧 rollout 因权限撤销、环境失效或日志不完整而无法重放时，不能继续带着权重进入当前梯度；它应标为不可用数据，而不是静默归入当前策略分布。

一个实用的诊断顺序是：先按 policy revision 绘制状态访问比例，再比较当前学生在这些状态上的 log-probability，最后观察旧样本和新样本对梯度范数、成功率及恢复率的贡献。如果旧 rollout 占比不断增加而独立恢复率下降，问题可能是 replay 过期，而不是教师能力不足。

### 17.21.2 OPD 与 RL 有联系，但优化对象不同

在学生采样到 a_t 后，教师与学生的 log-probability 差可以作为 token-level shaped signal。它的期望与负的 reverse KL 相关，因此可以放进策略梯度或其他在线更新中。但这个信号只说明“教师认为该动作相对可能”，不等于外部环境确认了任务成功。

若训练目标只包含教师信号，可以抽象为：

~~~math
\mathcal{L}_{\mathrm{teacher}}
=-\mathbb{E}_{\tau\sim\pi_S}
\left[\sum_t m_t w_t
\bigl(\log p_T(a_t\mid s_t)-\log p_S(a_t\mid s_t)\bigr)\right].
~~~

若同时存在任务结果 reward，则还需要明确它与教师项如何组合，以及哪个分量负责最终正确性：

~~~math
\mathcal{L}
=\mathcal{L}_{\mathrm{teacher}}
-\beta\,\mathbb{E}_{\tau\sim\pi_S}[R_{\mathrm{task}}(\tau)]
+\lambda\,\mathcal{L}_{\mathrm{constraint}}.
~~~

这里的 beta 和 lambda 不是越大越好。教师项过强，学生会追逐教师的表达 mode；任务项没有独立 verifier 时，结果可能被错误奖励放大；约束项如果只是可被抵消的软惩罚，就不能承担权限控制。是否称为“蒸馏”“策略优化”或“两者混合”，应由实际梯度、采样和奖励定义决定，而不是由训练器的名称决定。

### 17.21.3 更强的教师不一定在学生状态上更有用

教师在标准 benchmark 上更强，只说明它在那个任务分布、提示格式和资源预算下取得了更好的结果。学生可能把状态带到教师很少见的区域，例如损坏的代码工作区、缺少字段的工具响应、版本不匹配的依赖或已经执行过一次危险动作的环境。此时教师的总体能力排名不能替代对该状态分布的覆盖测试。

应把教师建议分成至少三类：被 verifier 或人工复核支持的建议、暂时无法判定的建议、明确错误或越权的建议。对前两类分别记录教师置信度和 abstain 行为；对第三类必须阻止其直接进入高权重训练集。一个教师在普通样本上准确率很高，但在学生失败状态上高置信度地产生错误建议，仍然可能使 OPD 退化。

教师信号的选择也要考虑学生的学习能力。完整 logits、长篇批注和复杂工具计划并不一定比一个明确的错误类别或下一步动作更好。可以做分层目标：先蒸馏结构化动作和安全协议，再加入局部 token 分布；对高不确定状态允许教师拒答或请求额外证据。这样做的依据是可验证性和可执行性，而不是教师回答的长度。

### 17.21.4 教师调用减少不等于蒸馏成功

训练后教师触发率下降可能有三种完全不同的解释：学生学会了在困难状态中恢复；触发器变得过窄，困难状态没有再被识别；学生通过早早拒答、缩短轨迹或避免工具动作来逃避风险。三者的教师调用曲线可能相同，但任务结果和安全后果相反。

因此要把调用次数和行为结果配对分析。对每个任务至少记录是否触发教师、触发原因、触发前的状态、教师建议、后续动作、最终结果和总成本。按“低置信度”“重复失败”“高风险动作”“工具异常”等原因分桶，比较触发与不触发的条件成功率。一个更有解释力的量是条件恢复率：

~~~math
P(\mathrm{recover}\mid \mathrm{failure\ state},\mathrm{teacher\ available}).
~~~

同时报告没有教师时的对应结果，才能判断学生是学会了恢复，还是只是在部署时继续依赖教师。若触发率下降伴随首次成功率、恢复率和探索熵一起下降，应优先检查逃避行为，而不是继续压低教师调用预算。

这个条件概率只在评测中存在至少一个被定义的 failure state，且“教师可用/不可用”的分组、预算、权限和任务版本可比时才有意义。若失败状态样本为零，或教师因系统故障而非策略实验不可用，应报告分母和未知状态，而不是把条件恢复率解释成学生能力差异。

### 17.21.5 如何证明 OPD 带来了能力，而不是风格相似度

教师 replay 上的 token loss、KL 和输出相似度只能证明学生更像教师。能力主张至少要比较四组对象：固定教师轨迹上的 offline KD、学生自生成状态上的 OPD、没有蒸馏的学生基线，以及教师本身。所有条件应尽量固定任务、提示、工具、模型预算和评估环境，只改变数据分布或蒸馏方法。

测试不能只使用训练中出现过的错误。应分别加入新的题面、从未见过的错误类别、工具失败、长轨迹后段、权限撤销、依赖版本变化和不允许的动作。对代码任务，尤其要构造“第一次 patch 失败、第二次可以恢复”的成对场景；对 RAG 任务，要更换文档版本并检查引用是否仍然支持结论；对数学任务，要检查不同等价表达而不是只比较字符串。

结果报告应同时包含首次成功率、错误后的恢复率、恢复步数、隐藏测试通过率、unsafe-action rate、教师调用数、学生 token、延迟和单位成功成本。可以用配对任务估计 OPD 与 offline KD 的成功率差异：

~~~math
\Delta_{\mathrm{success}}
=\hat P_{\mathrm{OPD}}(\mathrm{success})
-\hat P_{\mathrm{offline}}(\mathrm{success}).
~~~

这个差异还应按错误类型和轨迹位置给出置信区间或重采样范围。只报告一个总体均值，会把 OPD 真正改善的恢复样本和它造成负迁移的长尾样本混在一起。

### 17.21.6 tokenizer 不一致时，数值对齐可能掩盖语义错误

token-level KL 的前提是教师和学生在同一状态上使用可对应的动作空间。若两者 tokenizer 不同，教师词表中的第 1234 个 token 与学生词表中的第 1234 个 token 没有语义上的对应关系；按数组下标相乘虽然不会触发 shape error，却是在优化错误目标。

可行的替代方案有四类。第一，共享 tokenizer 和 chat template，使词表、特殊 token、停止条件与位置边界一致。第二，把教师输出还原成文本，在学生 tokenizer 下重新编码，进行序列级或片段级监督，但要接受重新编码带来的边界和概率损失。第三，把监督转到动作级，例如工具名、参数 schema、文件操作和结构化字段。第四，只使用 verifier、最终 artifact 或错误类别作为更粗粒度目标。

片段级对齐也需要记录边界。教师生成一个文本片段的概率，通常是该教师 tokenizer 下多个条件 token 概率的乘积；学生对同一片段的分解可能完全不同，不能把不同长度的 token loss 直接平均后比较。工程记录中应保存 tokenizer revision、normalization、special-token policy、chat template 和 stop rule。若这些协议不一致，先修复协议，再讨论蒸馏 loss 的变化。

最后，任何 tokenizer 迁移都要用语义任务验证：格式是否可解析、工具动作是否有效、引用是否落在正确 span、数学表达是否等价。数值训练正常只说明张量计算完成，不能证明教师和学生学习的是同一个行为。

## 17.22 小结

On-policy distillation 的价值在于让教师看到学生真正会访问的状态，并在这些状态上提供可以验证、可以约束、适合学生能力的指导。它针对的是离线示范与部署轨迹之间的分布差异，不是对教师正确性、探索质量、安全权限或任务泛化的自动保证。

一套可解释的 OPD 系统应分开记录学生状态分布、教师版本与信号粒度、tokenizer 和协议、verifier、过滤和权重、工具环境、缓存、教师成本、恢复成功率以及独立任务结果。长轨迹后段、学生与教师分布偏离、高熵 token 和教师高置信错误，是需要单独抽样的风险区域。

2023 年 GKD 奠定了从学生自生成错误中学习的公开基线；2026 年的 REOPOLD、TrOPD、位置偏差、熵感知、vOPD、RG-OPD、MOPD 和 TOP-D 继续处理稳定性、可信监督和多 rollout 信号。它们应当作为可复现的研究线索来阅读，而不是被压缩成“用了 OPD 就能把大模型能力传给小模型”的新闻结论。

## 17.23 资料与进一步阅读

资料按公开证据强弱分层。2023 年 GKD 已有 ICLR 2024 接收信息；后面的 2026 条目主要是 arXiv 预印本，摘要可以支持方法主张和研究方向，但不能替代对代码、数据、版本和独立 benchmark 的复现。

1. [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes / GKD](https://arxiv.org/abs/2306.13649)：OPD 的代表性早期工作，讨论学生自生成序列、教师反馈、分布错位以及与 RL fine-tuning 的结合。
2. [Scaling Reasoning Efficiently via Relaxed On-Policy Distillation / REOPOLD](https://arxiv.org/abs/2603.11137)：讨论 mixture-based reward clipping、熵驱动 token 采样和探索到精炼的训练安排；当前仍应按预印本处理。
3. [Trust Region On-Policy Distillation / TrOPD](https://arxiv.org/abs/2606.01249)：讨论可靠教师区域、异常 token、forward-KL 和教师前缀引导。
4. [On the Position Bias of On-Policy Distillation](https://arxiv.org/abs/2606.22600)：分析长 rollout 后段 token 的监督退化，并提出 IW-OPD 的位置加权思路。
5. [Entropy-Aware On-Policy Distillation of Language Models](https://arxiv.org/abs/2603.07079)：讨论教师熵、高熵位置的 forward-KL 混合和生成多样性。
6. [KL for a KL: On-Policy Distillation with Control Variate Baseline / vOPD](https://arxiv.org/abs/2605.07865)：把 OPD 的单样本估计与控制变量、token baseline 和梯度方差联系起来。
7. [Reward-Gated On-Policy Distillation / RG-OPD](https://arxiv.org/abs/2607.04037)：使用 verifier 反馈决定教师 logits 是否可信，适合讨论稀疏结果奖励与密集教师信号的组合边界。
8. [Multi-Rollout On-Policy Distillation via Peer Successes and Failures / MOPD](https://arxiv.org/abs/2605.12652)：利用同一问题的成功/失败 peer rollout 构造更有区分度的教师上下文。
9. [Trust Region Policy Distillation / TOP-D](https://arxiv.org/abs/2607.04751)：提出动态构造 proximal teacher 的稳定化路线；论文结论仍需独立复现。

阅读这些资料时，应分别记录论文明确报告的训练设置、方法假设、数据范围和结果指标。教师与学生的分布、verifier 的独立性、tokenizer 的兼容性、隐私边界和线上成本，任何一项没有落到当前系统的实测证据上，都不能被一句“蒸馏成功”掩盖。
