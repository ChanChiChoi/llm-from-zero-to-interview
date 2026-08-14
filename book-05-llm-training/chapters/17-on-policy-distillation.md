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
=\frac{C_{\mathrm{task}}}{\max(\epsilon,P_{\mathrm{success}})}
~~~

P_success 是独立任务上的真正成功率，epsilon 只是避免评估样本太少时出现除零。若 OPD 让学生成功率从 0.45 提高到 0.55，但教师调用量增加三倍，单位成功成本可能上升；若训练后线上可以少调用教师、少重试且延迟下降，收益才更可信。

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

若独立评估真正成功的任务数为 N_success，单位成功任务成本为：

~~~math
C_{\mathrm{unit\ success}}
=\frac{C_{\mathrm{training}}+C_{\mathrm{evaluation}}+C_{\mathrm{operations}}}
       {N_{\mathrm{success}}}
~~~

教师调用降低一次错误恢复的步数，可能提高一次任务的教师成本却降低总成本；反过来，学生在训练中依赖大量教师建议，线上没有教师时可能退化。成本分析必须在和部署相同的教师可用性、工具版本和重试预算下进行。

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


def should_query_teacher(state: AgentState) -> tuple[bool, str]:
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
    environment_revision: str,
) -> str:
    payload = {
        "task_id": state.task_id,
        "workspace_revision": state.workspace_revision,
        "text": state.text,
        "student_revision": student_revision,
        "teacher_revision": teacher_revision,
        "environment_revision": environment_revision,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


states = [
    AgentState("bug-1", "ws-7", "read parser", 0.91, 0, "read", "low"),
    AgentState("bug-1", "ws-7", "retry patch", 0.72, 2, "write", "medium"),
    AgentState("bug-2", "ws-2", "execute migration", 0.88, 0, "write", "high"),
]

for state in states:
    query, reason = should_query_teacher(state)
    key = state_cache_key(state, "student-3", "teacher-9", "sandbox-4")
    print(state.task_id, query, reason, key[:12])
~~~

这个 demo 有两个边界。第一，confidence 只是触发特征，不是权限判断；真正的 write 动作仍要由执行器拒绝或要求独立授权。第二，cache key 绑定了 workspace 和环境版本，但生产系统还应绑定 tokenizer、chat template、verifier、权限域和采样配置。缓存命中不能绕过最新的安全检查。

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

## 17.21 面试问题：如何解释 OPD

回答“on-policy distillation 和普通蒸馏有什么区别”时，应先说状态分布：普通蒸馏主要在教师示范分布 d_T 上训练，OPD 让学生在自己的分布 d_S 上生成状态，再请求教师指导。这样更能覆盖学生实际会遇到的错误，但教师成本、噪声和隐私风险更高。

回答“OPD 是否等于 RL”时，可以说二者存在联系但不等同。student-generated token 上的 teacher/student log-probability 比可以被解释为 token reward，OPD 也会遇到 on-policy 采样、credit assignment、方差和策略偏移问题；但具体优化器、KL 方向、是否有 verifier 和是否使用 value baseline，取决于实现。

回答“为什么教师越强不一定越好”时，应补充教师在学生错误状态上的支持覆盖、置信校准、任务正确性、调用成本和数据安全。教师如果对错误状态高置信地给出错误建议，蒸馏会把错误变成密集监督。

回答“如何证明 OPD 有效”时，应比较 SFT、offline KD、OPD 和教师，并在学生自生成错误、长轨迹、工具异常、新环境和安全边界上报告首次成功、恢复率、教师调用、长度、延迟和单位成功成本。教师 replay 上的 loss 下降不能替代独立任务结果。

回答“tokenizer 不一致怎么办”时，应指出 token logits 不能按下标直接对齐；应使用共享 tokenizer、文本级序列蒸馏、片段对齐、动作级目标或 verifier 结果。忽略 tokenizer 边界可能让训练数值正常、语义目标完全错误。

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
