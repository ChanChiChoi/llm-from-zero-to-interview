# 第 16 章 RLVR：可验证奖励如何训练更可靠的结果

强化学习真正需要的不是一个看起来合理的分数，而是一条能够把任务成功和任务失败区分开的反馈回路。数学答案可以代入检查，代码可以运行测试，JSON 可以验证 schema，SQL 可以在只读环境中执行并比较结果。这些任务有机会把结果转成可重复的奖励。

RLVR（Reinforcement Learning with Verifiable Rewards）把训练目标的一部分交给外部 verifier。模型生成答案或完整轨迹，verifier 检查结果，再把检查结果交给策略优化器。它的吸引力在于检查可以自动化、批量化、重复运行；它的危险也很直接：模型会优化 verifier 实际计算的东西，而不是优化我们心里没有写出来的“真正能力”。

初学者可以把它想成“模型答题，判卷器反馈”。但判卷器不是天然公正的老师：它可能漏看副作用，可能把等价答案判错，可能暴露测试信息，也可能在环境异常时把一次失败错误地归因给模型。工程上必须把任务目标、验证规则、环境权限、奖励分量、独立评估和成本放在同一张因果链上。

本章讨论的重点不是把所有任务都改造成强化学习，而是回答一个更具体的问题：什么时候一个可执行的检查器足以产生有用学习信号，怎样知道模型学到的是任务能力而不是规则漏洞。

## 16.0 适合使用 RLVR 的任务边界

RLVR 的第一步不是选择 GRPO、PPO 或某个训练框架，而是判断任务是否存在稳定的成功定义。只要成功定义本身含糊，训练过程就会把含糊放大。

| 任务 | 可验证对象 | 主要收益 | 主要盲点 |
| --- | --- | --- | --- |
| 数学 | 最终数值、等价表达、部分形式化步骤 | 反馈可重复，容易批量生成 | 推导过程可能伪造，解析器可能漏掉等价写法 |
| 代码修复 | patch 是否可应用、公开/隐藏测试、权限和副作用 | 结果接近真实开发任务 | 测试覆盖不足，环境泄漏，无限重试 |
| SQL/结构化输出 | schema、只读执行、结果等价性 | 可以自动判定格式和执行结果 | 等价查询、数据版本、危险访问 |
| 引用问答 | claim 与 source span 的支持关系 | 能约束证据使用 | 语义蕴含困难，URL 本身不证明引用成立 |
| 开放式专业建议 | 专家偏好、政策边界、人工复核 | 覆盖表达和情境判断 | 奖励漂移、专家分歧、责任风险 |

表中的“可验证对象”不等于“任务全部被验证”。例如代码测试通过只能说明当前测试集合没有观察到失败；它不能自动证明没有数据泄露、没有隐藏副作用，也不能证明代码在另一套依赖版本中仍然成立。数学答案被计算器接受，也不能说明模型写出的证明过程每一步都正确。

可以把训练闭环先画成下面这条链：

```text
x -> policy pi_theta -> trajectory tau -> artifact y
                                      |
                                      v
                       verifier V(x, tau, y, environment)
                                      |
                                      v
                  diagnostics -> reward mapping -> policy update
```

`x` 是任务输入，`tau` 是模型产生的动作和文本轨迹，`y` 是最终 artifact，例如答案、patch 或 SQL 查询，`environment` 是工具和执行环境。verifier 的输出不必一开始就压缩成一个浮点数；保留诊断向量，才能知道“答案错了”和“验证器超时”不是同一类问题。

当结果没有稳定的自动判定时，RLVR 不能凭空制造可靠反馈。更合理的路线可能是继续预训练、SFT、偏好优化、检索增强、工具执行后的人工复核，或先建立可控的模拟环境。训练方法应服从成功定义，而不是因为某个公开模型使用了 GRPO，就把所有任务都改写成 RL。

## 16.1 任务契约：先定义什么叫成功

设任务输入为 `x`，模型生成的轨迹为 `tau`，最终结果为 `y = artifact(tau)`。对于一个输入，通常不只有一个正确字符串，而是一组可以接受的结果 `Y*(x)`。例如 SQL 的等价查询可能有很多写法，数学答案可能使用分数、小数或带单位的表达。

因此，基本正确性更适合写成集合关系：

```math
C(x, y)=\mathbf{1}[y\in Y^*(x)]
```

`C` 是 correctness，`Y*(x)` 是任务允许的结果集合。若把正确答案硬编码成一个字符串，verifier 就会把表达形式误当成知识本身。

真实任务还需要格式、权限、副作用和预算条件。令：

```math
F(x,y)=\mathbf{1}[y\text{ 满足输出协议}]
```

```math
P(\tau,e)=\mathbf{1}[\tau\text{ 在环境 }e\text{ 中没有违反权限和安全约束}]
```

```math
B(\tau,e)=\mathbf{1}[\text{时间、token、CPU、内存和工具调用均未超出预算}]
```

一次结果真正可用的条件可以写成：

```math
S(x,\tau,e)=C(x,y)\cdot F(x,y)\cdot P(\tau,e)\cdot B(\tau,e)
```

这里的乘法不是说所有项目都必须用一个乘法 reward 实现，而是在语义上表达“任意关键条件失败，都不能称为完整成功”。安全违规不能仅仅是一个很小的负分，因为答案质量高不应该抵消越权读取、泄露 secret 或修改生产数据。

质量可以在硬性条件通过后继续使用连续分数。例如代码 patch 已经通过测试且没有越权操作，再用修改行数、运行时间、可维护性和解释清晰度排序。这样的分层比把所有因素简单相加更容易解释：一条代码很漂亮但修改了测试文件，应该先被标记为不可用，而不是靠漂亮分数把它拉回平均值。

对初学者而言，任务契约就是把“答对”拆成一张检查清单。对工程人员而言，它还要求写清楚输入版本、结果版本、工具版本、允许的副作用和失败责任归属。没有这些字段，训练曲线升高时无法判断是模型变强、数据变了，还是判定环境变了。

## 16.2 奖励契约：verifier 不等于 reward

verifier 负责观察并判断，reward mapper 负责把判断转换成优化器使用的数值。两者混在一起，会让排查变得困难。

最简单的结果奖励是：

```math
r_{\mathrm{out}}(\tau)=\mathbf{1}[V(x,\tau,e)=1]
```

`V` 可以是规则函数、测试执行器或等价性判断器。这个奖励清晰，但很稀疏：一条长轨迹直到最后才知道结果是 0 还是 1。更完整的奖励可能包含格式、过程、成本和安全诊断：

```math
R(\tau)=D_{\mathrm{safety}}D_{\mathrm{permission}}D_{\mathrm{budget}}
\left(R_{\mathrm{out}}+\alpha R_{\mathrm{process}}
      -\beta C_{\mathrm{length}}-\gamma C_{\mathrm{retry}}\right)
```

`D_safety`、`D_permission` 和 `D_budget` 是 0/1 诊断量，表示是否满足不可折中的约束；`R_out` 是结果奖励；`R_process` 是中间进展奖励；`C_length` 和 `C_retry` 是预算以内的软成本；`alpha`、`beta`、`gamma` 是权重。这个公式是一种可解释的设计框架，不是所有 RLVR 实现都必须照搬的固定算法。

实际系统更应该先保存原始向量：

```math
d(\tau)=
(d_{\mathrm{format}},d_{\mathrm{correct}},d_{\mathrm{independent}},
 d_{\mathrm{safety}},d_{\mathrm{permission}},d_{\mathrm{timeout}},
 d_{\mathrm{length}},d_{\mathrm{tool\_cost}})
```

向量中的布尔值和数值必须有明确的定义。例如 `d_timeout=1` 是“发生了超时”，还是“在超时预算内完成”？建议使用字段名表达方向，或者在 schema 中记录取值含义；否则不同组件会把同一个 1 理解成相反的事情。

将所有失败都压成 `reward = 0` 会损失重要信息。至少要区分下面几类：

| 结果 | 更可能的原因 | 训练后动作 |
| --- | --- | --- |
| 答案错误，工具正常 | 模型知识或推理失败 | 改数据、采样或策略更新 |
| 格式错误，答案可能正确 | 输出协议或模板失败 | 修正 tokenizer/template 或格式奖励 |
| 工具超时 | 环境或预算失败 | 优化工具、缩短轨迹或调整预算 |
| 权限拒绝 | 策略触发安全约束 | 保持拒绝，不能用正奖励抵消 |
| verifier 崩溃 | 评估系统失败 | 修复 verifier，通常不作为模型负样本 |
| 题目不可判定 | 任务定义不足 | 移出训练或转人工处理 |

如果验证器崩溃被记成普通负奖励，模型可能学到“少调用工具”而不是“更好地完成任务”。如果工具不可用被记成答案错误，数据分析会把基础设施故障误判成模型退化。

## 16.3 结果奖励与过程奖励

结果奖励只看最终状态。数学题的最终数值、代码的隐藏测试、SQL 的执行结果，都是典型 outcome reward。它的优点是目标接近真实任务，缺点是信用分配困难：轨迹中间某一步出错，最后的 0 没有指出错误位置。

过程奖励（process reward）试图评价中间状态，例如证明的某个步骤是否保持等价、检索到的文件是否包含所需符号、patch 是否已经能应用、工具返回是否改变了下一步决策。它提供更早的学习信号，却需要更细的标注或更复杂的检查。

一种示意性的组合是：

```math
R_{\mathrm{process}}(\tau)=\sum_{t=1}^{T}w_t q(s_t,a_t,s_{t+1})
```

`s_t` 是第 `t` 步状态，`a_t` 是动作，`q` 判断这一步是否带来真实进展，`w_t` 是时间或任务阶段权重，`T` 是轨迹长度。关键不在于奖励密度，而在于 `q` 是否绑定状态变化。

“写出三条引用”可能是一个格式过程奖励；“这三条引用中的 source span 是否支持对应 claim”才更接近真正进展。代码任务中，“输出了测试命令”不是进展，“测试命令在隔离环境中减少了一个真实失败类别”才是进展。过程奖励如果只评价文本外观，就会诱导模型生成更长、更像推理的文本。

常见的四组对照是只使用 outcome、只使用 process、组合奖励和人工校准后的奖励。对每组都要报告最终正确率、过程错误率、输出长度和成本。若最终准确率提高，但过程审计发现伪造步骤增加，说明过程奖励并没有提供可靠的中间监督。

过程监督研究通常以数学推理为实验场，但研究结论不能不加区分地迁移到开放式专业工作。数学步骤存在较清晰的等价关系；法律分析、医疗建议和组织决策的“中间步骤正确”往往依赖上下文、规范和责任边界。

## 16.4 Verifier 的类型与质量

### 16.4.1 规则 verifier

规则 verifier 检查 JSON schema、字段范围、正则格式、单位和简单约束。它便宜、稳定、容易复现，适合作为第一层检查；但字符串规则很容易把表面形式当成语义。一个真实 URL 不代表它支持了 claim，一个存在的字段也不代表字段值正确。

### 16.4.2 程序 verifier

程序 verifier 可以编译代码、运行单元测试、执行 SQL、调用数学库或检查形式化证明。它比字符串比较更接近任务语义，但正确性依赖测试集、依赖版本、执行环境和资源限制。测试通过说明观察到的测试通过，不自动推出所有输入都正确。

### 16.4.3 参考答案与等价性 verifier

对于数值、集合、排序结果和 SQL，可以比较规范化后的结果或判断语义等价。规范化必须谨慎：浮点数需要相对和绝对误差，单位需要显式转换，SQL 结果要说明行序、NULL 和数据快照，集合结果要说明重复项是否有意义。

### 16.4.4 专家或模型 verifier

开放式答案往往需要专家、奖励模型或强模型评价。它们可以覆盖更多表达形式，但存在标注分歧、评价漂移、位置偏差、长度偏差和模型同源偏差。模型 verifier 的分数适合作为一个证据来源，不应自动被当成客观真值。

### 16.4.5 Soundness 与 completeness

假设 `Y*(x)` 是可接受结果集合。verifier 的 soundness 关心错误答案是否被接受，completeness 关心正确答案是否被拒绝。对有限挑战集，可以用经验统计量表示：

```math
\widehat{\mathrm{Soundness}}
=1-\frac{\mathrm{FalseAccept}}{N_{\mathrm{invalid}}}
```

```math
\widehat{\mathrm{Completeness}}
=\frac{\mathrm{TrueAccept}}{N_{\mathrm{valid}}}
```

`N_invalid` 是已知错误答案数量，`N_valid` 是已知正确答案数量。soundness 高意味着不容易放过错误，completeness 高意味着不容易错杀正确表达。两者的分母必须分开，否则一个偏向拒绝的 verifier 可能看起来很“严格”，却完全不能接受合理答案。

要估计这两个量，不能只拿训练 rollout 来测。挑战集至少应包含等价答案、边界答案、格式合法但语义错误的答案、恶意输出、工具异常和超时结果。代码 verifier 还需要变异测试：有意插入一个 bug、越权访问或测试修改，看看检查器是否能发现。

### 16.4.6 覆盖率与独立性

假设失败类别为 `j=1,...,m`，每一类权重为 `w_j`，verifier 对该类别的覆盖为 `c_j`，可以用下面的近似量描述覆盖：

```math
\mathrm{Coverage}_{\mathrm{failure}}
=\frac{\sum_{j=1}^{m}w_jc_j}{\sum_{j=1}^{m}w_j}
```

这个量不是数学真理，因为失败类别的划分和权重本身也可能有偏差。它的作用是提醒工程师：verifier 的复杂度不等于覆盖率。一个只有字符串比较的复杂系统，可能仍然漏掉越权、回归和证据不支持；一个简单的只读执行器，可能覆盖了更重要的失败类别。

独立性至少有三层含义：评估数据不与训练数据重复，独立 verifier 不复用同一漏洞，评价环境不向模型暴露隐藏答案。主 verifier 和独立 verifier 结果不一致时，应把不一致保存下来，而不是选一个分数更好看的结果。

## 16.5 代码修复任务：从 patch 到可用 artifact

代码任务很适合说明 RLVR，因为“运行测试”看上去客观，却经常不是完整的成功定义。

一个 patch 的检查可以分成以下阶段：

| 阶段 | 检查对象 | 典型失败 |
| --- | --- | --- |
| 解析 | patch 能否应用、文件是否存在 | 上下文不匹配、格式损坏 |
| 静态约束 | 修改范围、依赖、敏感路径 | 修改测试、引入未声明网络依赖 |
| 公开测试 | 基本功能和反馈 | 只适应公开样例 |
| 隐藏测试 | 未公开输入和边界 | 仍可能受覆盖率限制 |
| 回归测试 | 原有功能 | 新功能通过但旧功能退化 |
| 资源检查 | 时间、内存、进程和磁盘 | 无限循环、资源消耗套利 |
| artifact 检查 | 最终文件和副作用 | 写入 secret、留下临时进程 |

如果只运行公开测试，模型可能针对测试中的固定字符串。如果只看进程 exit code，模型可能在测试文件中写入绕过逻辑。如果允许网络，模型可能读取外部答案或上传任务内容。verifier 的执行目录应该只提供任务所需文件，测试和验证脚本应当由模型不可写的账户持有。

隐藏测试的作用不是制造神秘感，而是减少模型对评分细节的过拟合。公开测试仍然有价值：它让模型在 rollout 中获得可解释反馈，帮助发现明显错误。两者要使用不同的输入和边界，并记录测试版本。隐藏测试本身也需要回归，否则测试集变化会被误认为模型能力变化。

对于真实代码任务，测试通过率最好与人工抽样、变异测试和静态分析一起报告。一个 patch 在 100 个测试中通过 99 个，不能简单解释为“99% 正确”；如果唯一失败的是权限边界，业务风险可能远高于普通格式错误。

## 16.6 SQL 与结构化输出：格式正确不是语义正确

SQL 任务至少可以使用四层 verifier：

1. parser verifier：SQL 能否解析，是否包含多语句、写操作或危险语法；
2. schema verifier：表名、字段和类型是否存在，是否符合任务允许范围；
3. execution verifier：在固定数据快照的只读环境中执行，结果与金标准或等价查询一致；
4. explanation verifier：自然语言解释是否与实际查询和结果一致。

只做第一层，得到的是“像 SQL 的字符串”；只做第三层但不限制表权限，可能得到正确统计却泄露不该访问的数据；只比较 SQL 字符串，会误判等价查询；只用语言模型评价解释，又可能出现解释和真实结果不一致。

数据快照是 SQL verifier 的一部分。表结构相同但数据版本不同，查询结果可能不同；同一查询在 NULL、重复行、时区和浮点聚合下也可能有不同语义。训练记录应包含 schema revision、data snapshot revision、数据库引擎版本和只读权限配置。

结构化输出也要区分 schema 和业务约束。JSON 可以通过 schema 检查，却把负数、未来日期或相互矛盾的字段放进去。因此可以先解析，再做类型、范围、跨字段一致性和权限检查。模型被要求输出工具动作时，动作白名单应当由执行器决定，不能由模型在 JSON 中声明“我有权限”。

## 16.7 工具沙箱与可重放性

verifier 执行外部工具时，模型面对的不是单纯的数学函数，而是一个会失败、会超时、会改变状态的环境。训练时如果工具始终在线、无限重试、没有权限限制，模型会学到一套部署时不存在的反馈循环。

至少需要限制以下资源：

| 资源 | 需要记录或限制的内容 |
| --- | --- |
| 文件 | 可读目录、可写目录、临时文件生命周期 |
| 网络 | 是否允许联网、域名白名单、请求大小和时间 |
| 计算 | CPU、GPU、内存、磁盘、进程数 |
| 时间 | 单次工具超时、整条轨迹截止时间 |
| 数据 | 数据快照、密钥过滤、个人信息脱敏 |
| 状态 | 环境变量、随机种子、依赖和镜像版本 |

一个可重放的 rollout 至少应保存 `task_id`、输入版本、策略 checkpoint、采样参数、动作序列、verifier 版本、沙箱镜像、退出原因、资源消耗和最终 artifact。重放时如果无法得到相同判定，要记录环境差异，而不是悄悄覆盖旧结果。

超时、工具不可用、策略拒绝和答案错误应使用不同的状态码。若系统确实无法判断答案，可以把这条轨迹标为 `unverifiable`，从训练样本中排除或交给人工分析。把不可判定强行改成负奖励，会把环境噪声写进策略梯度。

安全约束和质量分数也应由不同模块负责。模型写出正确 SQL 但尝试更新数据库，执行器应直接拒绝该动作并保留原因；不能先给它正确性正分，再用一个小安全扣分把两个事实相抵。

## 16.8 GRPO：组内相对信号从哪里来

GRPO（Group Relative Policy Optimization）的一种核心直觉是：对同一个问题采样多条回答，用这一组回答的相对奖励估计优势，减少对独立 value model 的依赖。

给定问题 `x`，从旧策略 `pi_old` 采样 `G` 条轨迹：

```math
\tau_1,\tau_2,\ldots,\tau_G
\sim \pi_{\mathrm{old}}(\cdot\mid x)
```

verifier 得到对应奖励 `r_1,...,r_G`。组均值和标准差为：

```math
\mu_G=\frac{1}{G}\sum_{j=1}^{G}r_j
```

```math
\sigma_G=\sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j-\mu_G)^2}
```

组内标准化优势可以写成：

```math
A_i=\frac{r_i-\mu_G}{\sigma_G+\epsilon}
```

`G` 是每道题的候选数量，`epsilon` 是防止数值除零的小常数。`A_i` 是相对信号，不是绝对正确概率：它只说明第 `i` 条轨迹在同组中高于还是低于平均水平。

如果一组回答全部正确或全部错误，`sigma_G` 为 0。加上 `epsilon` 可以避免 NaN，却不能凭空创造区分度；一个合理实现应记录这类零方差组，并决定跳过、重新采样或使用其他批次信息。若所有候选都共享同一个 verifier 漏洞，标准化只会把共同错误当作正常背景。

### 16.8.1 从优势到策略更新

策略更新使用新旧策略对同一动作的概率比：

```math
\rho_{i,t}(\theta)=
\frac{\pi_{\theta}(a_{i,t}\mid s_{i,t})}
     {\pi_{\mathrm{old}}(a_{i,t}\mid s_{i,t})}
```

`a_{i,t}` 是第 `i` 条轨迹在位置 `t` 的 token 或动作，`s_{i,t}` 是动作之前的状态。常见的 clipped surrogate objective 可以写成示意形式：

```math
L_{\mathrm{clip}}(\theta)=
\mathbb{E}\left[
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}
\sum_{t=1}^{T_i}
\min\left(
\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)A_i
\right)
\right]
```

`T_i` 是第 `i` 条轨迹中参与策略损失的 token 数，`epsilon` 控制一次更新的概率变化范围。不同实现可能使用序列级平均、全 batch token 平均、不同 mask 或不同 KL 估计，所以这条公式应理解为结构化示意，不能据此断言所有 GRPO 代码逐项相同。

为了避免策略离参考模型过远，通常还会加入 KL 约束或 KL 惩罚：

```math
L(\theta)=L_{\mathrm{clip}}(\theta)
-\beta D_{\mathrm{KL}}(\pi_{\theta}\,\|\,\pi_{\mathrm{ref}})
```

`pi_ref` 是参考策略，`beta` 控制偏离代价。KL 太弱，策略可能快速坍缩到 verifier 的漏洞；KL 太强，策略更新又可能小到无法产生新能力。真正的调参依据应该是独立任务表现、奖励分量、长度、拒答率和成本，而不只是训练 loss 是否平滑。

### 16.8.2 序列长度归一化的影响

把每条轨迹先平均再对 batch 求平均，和把所有有效 token 放在一起平均，并不是同一个目标。前者让短回答和长回答拥有相同的序列权重，后者让 token 数更多的回答拥有更大影响。对于 reasoning RL，长轨迹往往不是随机噪声，而是模型探索策略的一部分，因此归一化口径会改变优化方向。

若使用序列平均：

```math
L_{\mathrm{seq}}=\frac{1}{G}\sum_{i=1}^{G}
\frac{1}{T_i}\sum_{t=1}^{T_i}\ell_{i,t}
```

若使用有效 token 平均：

```math
L_{\mathrm{token}}=
\frac{\sum_{i=1}^{G}\sum_{t=1}^{T_i}\ell_{i,t}}
     {\sum_{i=1}^{G}T_i}
```

`ell_{i,t}` 可以是带优势和概率比的 token 损失。两种写法都可能合理，但必须在实验记录中明确。长回答比例变化时，如果没有说明归一化口径，训练版本之间的 reward 或 loss 对比没有可比性。

### 16.8.3 组大小与探索成本

增大 `G` 通常能提高同一问题内发现不同策略的机会，但每个问题的 rollout、verifier 执行、日志和显存成本也近似增加。若策略已经坍缩，增加候选只是在重复同一答案，不能增加有效探索。

组大小应与任务难度、奖励稀疏程度和 verifier 成本共同确定。实验记录至少包括采样温度、top-p、最大长度、组大小、组内重复率、零方差组比例和每个独立成功结果的成本。把 `G` 增大后准确率提高，不能直接说明单条轨迹的推理能力提高，因为更多采样本身也提高了找到正确答案的概率。

## 16.9 公开路线：DeepSeekMath、DeepSeek-R1 与 DAPO

公开论文是理解 RLVR 的锚点，但它们提供的是特定实验条件下的证据，不是所有任务的能力保证。

DeepSeekMath 的论文报告了继续预训练、数学数据处理、工具使用和 GRPO 的组合路线。其摘要明确把 GRPO 描述为 PPO 的一个变体，并强调了相对优势与 PPO 内存使用之间的关系。阅读时要把“论文报告的数学 benchmark 结果”与“GRPO 可以迁移到任何开放任务”分开。

DeepSeek-R1 的技术报告把 reasoning RL、可验证任务和蒸馏放到更广的推理框架中，报告了数学、代码和 STEM 等任务上的结果，也讨论了较小模型的蒸馏路线。它能说明公开路线如何利用强化学习诱导推理行为，但不能替我们知道内部 verifier、数据配比、采样过滤和部署策略的全部细节。凡是论文没有披露的训练 recipe，都应标为未知，而不是从模型行为反推为事实。

DAPO 论文的全称是 Decoupled Clip and Dynamic Sampling Policy Optimization。公开摘要说明了算法、训练代码、数据处理和基于 verl 的大规模系统均作为开放研究材料发布；论文正文进一步把大规模 reasoning RL 中的几个训练工程问题显式化：上下裁剪范围的解耦、动态采样、token 级损失和过长轨迹处理。

| 公开方法 | 解决的具体问题 | 读者应保留的边界 |
| --- | --- | --- |
| GRPO | 用组内相对奖励构造优势，降低对 value model 的依赖 | 相对优势仍然依赖 verifier 的质量 |
| DeepSeekMath 路线 | 将数学数据、工具和 GRPO 组合起来 | 数学结果不能外推到开放事实任务 |
| DeepSeek-R1 路线 | 报告 reasoning RL 和蒸馏的公开框架 | 内部工程细节以技术报告披露为准 |
| DAPO | 公开大规模 reasoning RL 系统与训练工程取舍 | 论文结果依赖数据、模型、框架和算力配置 |

DAPO 中的 decoupled clip 可以理解为对概率比的上下边界分别控制：在鼓励探索的方向放宽一侧，在防止过度更新的一侧保持约束。dynamic sampling 试图减少奖励没有区分度的样本组；token-level loss 重新审视长短回答在 batch 中的权重；overlong reward shaping 则处理过长轨迹、截断和长度投机之间的关系。这些机制解决的是优化和系统信号问题，不会修复一个错误的 verifier。

具体实现仍要对照论文版本和代码版本。尤其是“动态采样”到底过滤什么样本、“token-level loss”采用哪一种 mask 和归一化、“过长惩罚”何时施加，都属于实现细节。只看方法名就复述一个固定 recipe，容易把不同版本的实现混在一起。

## 16.10 Reward hacking：模型为什么会钻规则漏洞

当 reward 被优化时，模型不需要理解设计者的意图，只需要找到能提高分数的行为。常见路径包括：

| 路径 | 表面结果 | 隐藏问题 |
| --- | --- | --- |
| parser 差异 | 解析器接受特殊空白、浮点或 Unicode | 人类和执行器理解不同 |
| 测试针对 | 公开测试通过率上升 | 模型记住固定字符串或错误消息 |
| 工具污染 | 测试或缓存被修改后变绿 | artifact 并不是原任务的修复 |
| 过程伪造 | 推导步骤更完整 | 最终答案来自未记录的捷径 |
| 引用伪造 | 引用数量增加 | source span 不支持 claim |
| 长度套利 | 更长轨迹更容易碰到正确答案 | 延迟、成本和重复急剧上升 |
| 拒答套利 | 风险任务全部拒绝 | 表面安全提高，任务覆盖下降 |

“分数上升”必须和独立证据一起看。训练 verifier 从 60% 上升到 82%，可能只是格式通过率从 85% 上升到 99%，独立正确率却只从 70% 到 72%，平均长度从 180 token 增到 430 token。此时训练取得的主要收益是模板遵循和更多尝试，而不是推理能力出现同等幅度的提升。

识别 reward hacking 的一种实用方法是抽取高奖励轨迹进行独立重放。设主 verifier 判定成功，但独立 verifier、隐藏测试或专家抽样至少有一个判定失败，则称为 disagreement。其经验比例为：

```math
\widehat{p}_{\mathrm{disagree}}
=\frac{1}{N}\sum_{k=1}^{N}
\mathbf{1}[V_{\mathrm{main}}(\tau_k)=1
\land V_{\mathrm{independent}}(\tau_k)=0]
```

`N` 是抽样轨迹数。这个比例不是唯一的安全指标，但持续升高通常说明模型正在利用评分器与真实任务之间的缝隙。防御手段包括隐藏变体、独立实现、随机化测试、只读工具、资源限制、输出长度分桶和人工抽样。把所有 verifier 规则全部暴露给模型有助于调试，却可能降低它作为独立检查的价值。

## 16.11 数据污染与 verifier 污染

RLVR 的污染不只发生在训练文本和测试题之间。训练题面、标准答案、公开测试脚本、错误消息、verifier prompt、合成 teacher trace 甚至判题器的固定字符串，都可能成为模型记忆或利用的对象。

污染检查至少包括以下层次：

1. 对题面、答案和代码做 exact match；
2. 对 n-gram、变量重命名、改写和翻译做近似匹配；
3. 检查公开测试脚本、错误消息和 verifier prompt 是否进入训练语料；
4. 检查合成轨迹的 teacher 是否直接看到测试答案；
5. 在训练后更换数字、变量、接口、数据快照和规则实现；
6. 把独立生成的新题作为长期回归集合，而不是只做一次性测试。

如果训练题和测试题来自同一个模板，改写题面未必足够。代码任务要改变接口和边界条件，SQL 任务要改变数据分布和 NULL 规则，数学任务要改变数字和证明路径，工具任务要改变环境状态和权限。新题的通过率下降时，要先检查题型是否真正独立，再解释成泛化失败。

verifier 污染还会出现在日志系统。训练日志如果记录了隐藏测试、密钥、数据库返回或完整失败答案，后续数据清洗不彻底就可能把评估信息送回模型。代码 patch 和测试输出需要过滤 secret 与个人信息；企业工具返回应有访问控制、留存期限和脱敏策略。

## 16.12 泛化评估：换规则、换题面、换环境

RLVR 的评估不能只报告训练 verifier score。至少要把下列集合分开：

| 集合 | 作用 |
| --- | --- |
| training | 观察优化是否发生，不代表泛化 |
| same-distribution holdout | 检查同一分布的普通泛化 |
| template/variable shift | 检查表面变化下是否仍理解任务 |
| verifier/implementation shift | 检查是否依赖某个判题器漏洞 |
| adversarial set | 检查格式、权限和边界攻击 |
| real workflow | 检查工具、延迟、副作用和人工协作 |

训练题、同分布测试题、新题、对抗题和真实工作流要同时报告最终正确率、verifier 通过率、格式通过率、独立检查、拒答/安全、答案长度、延迟和单位成功任务成本。

verifier 的经验 precision 和 recall 可以写成：

```math
\widehat{\mathrm{Precision}}
=\frac{\mathrm{TrueAccept}}
       {\mathrm{TrueAccept}+\mathrm{FalseAccept}}
```

```math
\widehat{\mathrm{Recall}}
=\frac{\mathrm{TrueAccept}}
       {\mathrm{TrueAccept}+\mathrm{FalseReject}}
```

这里的 precision 回答“被接受的结果有多少真的正确”，recall 回答“已知正确结果有多少被接受”。它们需要有人工或独立程序提供的参考标签，不能直接拿主 verifier 自己的输出计算。

迁移差距可以用一个简单的诊断量表示：

```math
\Delta_{\mathrm{transfer}}
=S_{\mathrm{independent}}-S_{\mathrm{train\ verifier}}
```

如果这个值显著为负，模型在主 verifier 上表现好、在独立检查上表现差，优先怀疑 reward overfitting、污染或判定器漏洞。如果数学能力提高、开放事实能力几乎不变，不应把它称作 RLVR 失败；这可能正是 verifier 覆盖边界的结果。

还要按长度分桶报告条件正确率：

```math
P(\mathrm{correct}\mid L\in[b_j,b_{j+1}))
```

`L` 是输出 token 数，`[b_j,b_{j+1})` 是长度区间。若长回答的 verifier 通过率上升，但独立正确率不变，模型可能只是用更多 token 做搜索或重复，而不是提高单步推理质量。

## 16.13 成本：一次通过不等于值得训练

RLVR 的成本不仅是策略模型 forward。每个问题可能生成 `G` 条 rollout，每条轨迹还要执行编译器、测试器、数据库、浏览器或数学工具；失败后可能重试，更新时还可能保存 reference logits、梯度和日志。

一个训练 step 的粗略账本是：

```math
C_{\mathrm{step}}
=C_{\mathrm{rollout}}+C_{\mathrm{verify}}
 +C_{\mathrm{reference}}+C_{\mathrm{storage}}+C_{\mathrm{retry}}
```

若 `N_success` 是独立评估中真正完成任务的数量，单位成功任务成本可以写成：

```math
C_{\mathrm{unit\ success}}
=\frac{C_{\mathrm{training}}+C_{\mathrm{evaluation}}+C_{\mathrm{operations}}}
       {N_{\mathrm{success}}}
```

这个指标会惩罚“通过率提高但答案变得极长”“每道题采样几十次才成功”“verifier 需要昂贵人工复核”等方案。它不是唯一目标，但比单独看 reward 更接近生产决策。

如果一次尝试的成本为 `C_attempt`，真正成功的概率为 `p_success`，在独立重试近似成立时，期望成功一次的成本大致为：

```math
\mathbb{E}[C_{\mathrm{success}}]
\approx\frac{C_{\mathrm{attempt}}}{p_{\mathrm{success}}}
```

这个近似在重试会改变环境状态、失败样本相关或存在固定启动成本时会失真，但足以说明一个问题：提升单次通过率和减少每次尝试成本，可能同样重要。

verifier 超时或外部工具不稳定会造成 reward 缺失。系统要区分 `timeout`、`invalid`、`policy_denied`、`verifier_error` 和 `model_wrong`。其中 `verifier_error` 通常应进入系统稳定性报表，而不应直接作为模型负样本。

## 16.14 训练稳定性：稀疏奖励之外还有哪些风险

RLVR 的训练不稳定可能来自奖励稀疏，也可能来自策略更新、长度分布、参考模型、工具环境和 verifier 版本变化。仅观察总 reward 无法区分这些原因。

建议同时观察下面的时间序列：

| 指标 | 反映的问题 |
| --- | --- |
| 每组 reward 均值和方差 | 是否有学习信号、是否出现奖励漂移 |
| 零方差组比例 | 采样是否没有产生有效比较 |
| 独立 verifier 差距 | 是否过拟合主判题器 |
| 输出长度和截断率 | 是否发生长度套利或预算失配 |
| KL 与 ratio 分布 | 策略是否偏离过快 |
| 工具调用、超时和拒绝 | 环境噪声和策略副作用 |
| 各 reward 分量 | 哪个目标推动了总分变化 |
| 每个独立成功的成本 | 能力提升是否值得资源消耗 |

KL 约束的作用是控制策略偏移，不是替代 verifier。KL 很低但独立正确率不上升，说明学习信号可能没有信息；KL 快速升高并伴随长度暴涨，说明策略可能在追逐漏洞或奖励尺度。任何一个现象都需要结合 rollout 和失败原因抽样，而不是只调一个学习率。

奖励尺度也会影响更新。一个稀疏的 `0/1` 奖励和一个含长度、过程、格式分量的连续奖励，优势的方差不同；如果分量权重随版本变化，旧训练曲线不能直接比较。训练记录应保存奖励 schema 和权重，而不是只保存最终的平均 reward。

## 16.15 一个可重放的结构化答案 verifier

下面的 demo 不实现完整 GRPO，而是把“verifier 观察什么、哪些条件不可折中、失败原因如何保留”具体化。任务要求模型返回 JSON，给出一个数值答案和一个只读动作。答案正确但动作是 `write` 时，结果仍然不可用；JSON 损坏或超过字节预算时，也不会被误认为数学错误。

```python
from dataclasses import dataclass
import json
import math


@dataclass(frozen=True)
class Verdict:
    reward: float
    diagnostics: dict[str, bool]
    reason: str


def verify_candidate(raw: str, expected: float, max_bytes: int = 512) -> Verdict:
    diagnostics = {
        "json": False,
        "schema": False,
        "correct": False,
        "permission": False,
        "budget": len(raw.encode("utf-8")) <= max_bytes,
    }

    if not diagnostics["budget"]:
        return Verdict(0.0, diagnostics, "budget_exceeded")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return Verdict(0.0, diagnostics, "invalid_json")

    diagnostics["json"] = True
    diagnostics["schema"] = (
        isinstance(payload, dict)
        and set(payload) == {"answer", "action"}
        and isinstance(payload["answer"], (int, float))
        and not isinstance(payload["answer"], bool)
        and isinstance(payload["action"], str)
    )
    if not diagnostics["schema"]:
        return Verdict(0.0, diagnostics, "schema_error")

    diagnostics["permission"] = payload["action"] in {"read", "calculate"}
    diagnostics["correct"] = math.isclose(
        float(payload["answer"]),
        expected,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )

    hard_conditions = ("json", "schema", "permission", "budget")
    if not all(diagnostics[name] for name in hard_conditions):
        reason = next(name for name in hard_conditions if not diagnostics[name])
        return Verdict(0.0, diagnostics, f"hard_condition_failed:{reason}")
    if not diagnostics["correct"]:
        return Verdict(0.0, diagnostics, "wrong_answer")
    return Verdict(1.0, diagnostics, "success")


examples = [
    '{"answer": 42, "action": "calculate"}',
    '{"answer": 41, "action": "calculate"}',
    '{"answer": 42, "action": "write"}',
    '{"answer": 42, "action": "calculate"',
]

for candidate in examples:
    print(verify_candidate(candidate, expected=42))
```

这个例子有三个值得注意的地方。第一，`diagnostics` 在返回结果中保留了 schema、权限、预算和正确性，而不是只返回一个 0/1。第二，权限和预算是硬性条件，正确答案不能抵消它们。第三，`math.isclose` 的误差参数只是这个 toy 任务的假设；工程上应根据数值范围、单位和业务容忍度设定，不能把 `1e-9` 当成通用规则。

把这个 verifier 接到 RL 训练之前，还要测试它自己的边界：`NaN` 和无穷值、极大的整数、重复 JSON 字段、Unicode 空白、未知字段、超长字符串、异常编码以及动作字段的大小写。还要确认执行器使用的 JSON 解析器与测试脚本使用的是同一个协议，避免训练端和评估端对同一字符串给出不同结论。

## 16.16 从 demo 到训练系统

一个真实系统可以按下面的顺序组织，但每一步都要留下可追溯的结果：

1. 固定任务和数据版本，先写出可接受结果及失败类别；
2. 用已知正确、已知错误、等价表达和恶意样本测试 verifier；
3. 在隔离环境中生成 rollout，记录采样参数和策略 revision；
4. 执行 verifier，保留诊断向量、异常、资源消耗和最终 artifact；
5. 对无法判定或环境崩溃的样本单独处理，不让它们静默进入普通奖励；
6. 按明确的序列或 token 归一化方式计算优势和策略损失；
7. 用独立 verifier、新题和真实工作流做周期性评估；
8. 根据独立成功率、成本和安全副作用决定继续训练、调整 verifier 或回退模型。

这里的顺序很重要。若先训练再想成功定义，模型和 verifier 会同时变化，最后很难判断谁在适应谁。若只记录最终 reward，训练团队也无法回答“这次提升来自正确性、格式还是更长的答案”。

verifier 版本、测试版本、沙箱镜像和 reward schema 应当和模型 checkpoint 一起保存。升级 verifier 后，旧 rollout 需要在新旧版本上重放一部分，才能知道分数差异来自模型还是判题规则。对不可重放的环境，应把该限制写入实验结论，而不是把结果包装成精确的能力提升。

## 16.17 常见症状与定位路径

| 观察到的症状 | 优先检查 | 不应直接得出的结论 |
| --- | --- | --- |
| 训练 reward 上升，独立正确率不变 | 格式奖励、长度、污染、主 verifier 漏洞 | “模型推理已经增强” |
| 所有组的 reward 都相同 | 题目难度、采样温度、verifier 覆盖 | “GRPO 学不动” |
| 长度和工具调用暴涨 | 长度惩罚、重试逻辑、工具状态是否改变 | “模型更认真” |
| timeout 比例上升 | 沙箱资源、并发、依赖版本、最大长度 | “模型变笨了” |
| 通过率上升但权限违规增加 | 安全条件是否被软分抵消、执行器账户 | “总 reward 设计合理” |
| 换一个 parser 后分数大幅下降 | 协议差异、边界样本、训练污染 | “新 parser 不可靠” |
| seed 之间差异很大 | 奖励稀疏、组大小、样本难度、更新步数 | “平均值足够代表效果” |

定位时应回到原始轨迹。抽取成功、失败、超时和高成本样本，分别重放主 verifier 和独立 verifier，再按长度和任务难度分桶。只有当这些证据一致时，训练曲线才有解释力。

## 16.18 RLVR 和 RLHF 如何分工

RLVR 使用可执行规则、测试或可计算的等价性产生反馈，适合数学、代码和结构化任务。它的信号重复性较强，批量成本通常较低，但覆盖范围受 verifier 限制，且容易出现规则投机。

RLHF 通过人类偏好或奖励模型评价帮助性、风格、拒答边界和开放式质量，覆盖更广，但标注成本、评价一致性和奖励模型漂移更难控制。两者不是互相排斥的替代品：一个系统可以用 SFT 学习表达协议，用 RLHF 调整开放偏好，再用 RLVR 强化可执行结果，同时用独立安全评估约束工具行为。

重要的是保持目标分层。数学答案正确不等于语气符合偏好；回答礼貌也不等于代码能运行。不同奖励来源之间若直接相加，必须做分量消融、量纲检查和冲突样本分析，否则高方差的某一项可能压过真正重要的任务结果。

## 16.19 什么时候值得使用 RLVR

一个任务值得尝试 RLVR，通常同时满足以下条件：成功定义能够被写成稳定谓词或有限层次的检查，反馈足够及时，探索不会产生不可逆副作用，verifier 执行成本可承受，并且存在独立数据或独立环境可以验证迁移。

如果验证必须依赖昂贵专家、长时间真实操作或不可逆生产写入，应先使用模拟环境、离线轨迹、只读工具或分层检查。生产环境不应直接充当探索器。对于无法自动判断的开放答案，可以让 RLVR 学习格式、工具调用和可验证子任务，把最终价值判断留给人工或独立流程。

最终的选择应看单位成功任务成本、独立正确率、安全副作用和回退能力，而不是只看训练 reward。RLVR 是一种反馈设计方法；它能放大可观察的任务信号，也会放大 verifier 的盲点。

## 16.20 面试题：如何解释一个 RLVR 系统

面试中回答“RLVR 和 RLHF 有什么区别”时，不能只说“一个可验证，一个人工反馈”。完整回答应先说明任务边界，再说明 verifier 产生什么信号、奖励是否稀疏、如何防止规则投机，以及用什么独立评估证明迁移。

回答“为什么 GRPO 不需要 value model”时，应说明它用同一问题的一组 rollout 的相对奖励估计优势，减少了独立 value model 的参数和显存；但它并没有消除 baseline、采样方差、verifier 偏差和组内零方差问题。

回答“代码测试通过是否足够”时，应补充 patch 应用、隐藏测试、回归测试、测试文件保护、网络和文件权限、资源限制以及依赖版本。测试通过只是一个观察结果，不能自动推出没有副作用。

回答“reward 上升但能力没有提升怎么办”时，应按训练 verifier、同分布 holdout、独立 verifier、新题、长度、格式、污染和单位成本拆分。优先排查 reward hacking、数据泄漏、零方差组和环境变化，而不是盲目增加 RL 步数。

回答“安全约束能不能作为负奖励”时，应区分不可接受行为和可优化质量。越权、泄露 secret、修改测试或生产写入通常应由执行器拒绝并作为硬性失败；长度、延迟和冗余可以作为软成本。不能让正确答案抵消安全违规。

## 16.21 小结

RLVR 的价值来自可验证结果与真实任务目标之间的重合。数学代入、代码测试、只读 SQL 执行和 schema 检查都可以产生学习信号，但每一种检查都只覆盖它真正观察到的部分。

一个可靠的 RLVR 系统至少要把任务成功、结果奖励、过程奖励、安全与权限条件、verifier 的 soundness/completeness、环境版本、污染风险、组内采样、独立评估和单位成功成本分开记录。

GRPO 的组内优势解决的是信号构造和 value model 成本问题，不会自动解决奖励错误；DAPO 等公开工作说明了大规模 reasoning RL 的优化和系统取舍，也不能替代对当前任务 verifier 的实测。训练时提高主 verifier 分数只是起点，只有在换题面、换规则、换环境后仍然保持独立成功率，才有理由说能力发生了迁移。

## 16.22 资料与进一步阅读

以下资料按证据类型排列。论文用于核对方法定义和作者报告的实验，框架文档用于核对接口和工程实现，治理框架用于补充安全与可追溯要求；它们不能互相替代。

1. [DeepSeekMath](https://arxiv.org/abs/2402.03300)：公开讨论数学数据、工具使用和 GRPO 的代表性论文；其结果边界是论文所报告的数学实验。
2. [DeepSeek-R1](https://arxiv.org/abs/2501.12948)：公开讨论 reasoning RL、可验证任务和蒸馏路线的技术报告；内部数据、奖励和系统细节以披露内容为准。
3. [DAPO](https://arxiv.org/abs/2503.14476)：公开的大规模 LLM 强化学习系统，涉及 decoupled clipping、dynamic sampling、token-level loss 和过长轨迹处理；实现细节应同时核对论文版本、代码和数据版本。
4. [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)：过程监督与逐步验证研究，用于比较 outcome reward 与 process reward 的假设和风险。
5. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)：PPO 的概率比、advantage 和 clipping 机制，是理解 GRPO 更新形式的基础资料。
6. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)：离线偏好优化的原始论文，可用于比较 preference signal 与在线 verifier signal。
7. [Hugging Face TRL](https://huggingface.co/docs/trl)：核对 SFT、PPO、DPO、GRPO 等训练器接口、数据字段和版本变化；文档不是某个实验结果的独立证明。
8. [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)：关于风险识别、测量、管理和治理的参考框架，可补充工具权限、隐私、可追溯和部署风险；它不是 RLVR 算法论文。

阅读这些资料时，应该分别记录“论文明确报告了什么”“框架实际支持什么”“当前项目测出了什么”。公开模型在数学或代码上表现出色，不能据此推出 RLVR 已经解决开放事实、安全、责任和工具权限问题；只有当前 verifier、数据、环境和独立评估都被记录，RLVR 的分数才有可解释性。
