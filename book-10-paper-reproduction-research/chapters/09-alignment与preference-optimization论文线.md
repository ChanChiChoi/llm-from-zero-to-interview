# 第九章：Alignment 与 Preference Optimization 论文线

预训练语言模型的基本任务是预测下一个 token。经过足够多的数据和计算，它可以学会语言结构、事实关联、代码模式以及一部分推理规律。但“能够续写”不等于“能够作为助手可靠地完成任务”。助手还要理解指令、遵守格式、表达不确定性、避免危险帮助，并在 helpfulness、truthfulness 和 harmlessness 发生冲突时做出可接受的取舍。

Alignment 论文研究的就是这段从能力到行为的转换。Preference optimization 则是其中一条重要技术线：它把“哪个回答更好”的比较信号变成训练目标，让模型在候选行为之间改变概率分配。本章按一条完整证据链阅读这条论文线：先定义行为目标，再看示范和偏好如何成为数据，接着解释 reward model、RLHF、DPO、IPO、KTO、ORPO 和 RLAIF 的差异，最后讨论评估、风险、复现与部署边界。

这里的 alignment 不是一个已经被数学上唯一规定的“人类价值函数”。它是一个受任务、用户、政策、评估和系统权限共同影响的工程对象。论文能够证明的是某些数据和评估协议下的行为变化，而不是模型从此在所有环境中都与所有人的价值观一致。

## 1. 从 next-token prediction 到助手行为

### 1.1 预训练目标没有直接表达用户意图

设上下文为 $x$，目标序列为 $y=(y_1,\ldots,y_T)$。语言模型的自回归训练目标可以写为：

~~~math

L_{LM}(\theta)
=-\mathbb{E}_{(x,y)}
  \sum_{t=1}^{T}\log \pi_{\theta}(y_t\mid x,y_{<t}).

~~~

这个目标要求模型提高训练语料中下一个 token 的概率。它能奖励语法完整、风格相似和统计上常见的延续，却没有直接告诉模型：回答是否解决了用户真正的问题，是否应该承认不知道，是否会泄露隐私，是否会执行越权操作。

因此，一个 base model 可能生成语法流畅但答非所问的文本，也可能在训练语料里见过危险操作的描述后继续补全步骤。模型并不是“故意不听话”，而是优化目标和部署目标之间缺少直接联系。

### 1.2 Alignment 至少包含三个不同对象

讨论 alignment 时，首先要区分三个层次。

| 层次 | 主要问题 | 常见证据 |
| --- | --- | --- |
| 行为层 | 回答是否有用、真实、无害、遵循指令 | 人工偏好、任务成功率、事实性、安全评估 |
| 策略层 | 哪些请求应回答、拒绝、澄清或转交 | policy、边界样本、红队、误拒/漏拒 |
| 系统层 | 模型是否能调用不该调用的工具或访问不该访问的数据 | 权限、审计、沙箱、事故记录 |

Preference optimization 主要改变行为层的概率分布，也可能间接改善策略层的表达。但它不能替代系统层的权限控制。一个模型即使学会在文字上拒绝危险请求，如果执行器仍然允许任意文件删除，系统就不能被称为安全。

### 1.3 目标是向量，不是一个天然的总分

helpfulness、truthfulness、harmlessness、格式遵循和延迟通常不是同一个方向。可以把一次回答的结果写成向量：

~~~math

\mathbf{u}(x,y)
=
\bigl(u_{help},u_{truth},u_{safe},u_{format},u_{cost}\bigr).

~~~

训练时常把它们压成一个标量，例如：

~~~math

U(x,y)=\sum_{j=1}^{m}w_j u_j(x,y),
\qquad w_j\geq0.

~~~

标量化方便优化，却把权重和冲突藏起来。如果把安全惩罚的权重设得过大，模型可能大量拒答；如果只奖励 helpfulness，模型可能在不确定时自信编造；如果只奖励格式，模型可能学会漂亮地输出错误内容。论文必须说明 reward 或偏好标签实际覆盖了哪些维度。

### 1.4 从模型行为到系统行为的边界

模型回答是系统的一部分，而不是系统全部。一个企业知识助手至少还有检索器、文档权限、工具执行器、日志系统和人工升级路径。模型训练可以提高“基于证据回答”的概率，但不能凭语言行为保证检索文档没有越权，也不能保证工具参数没有副作用。

因此阅读对齐论文时，要问两个问题：论文测量的是模型在孤立对话中的偏好，还是包含检索、工具和状态变化的端到端行为？如果只测量前者，结论不能直接外推到后者。

## 2. SFT：先把可接受的行为写成示范

### 2.1 示范数据改变了条件分布

监督微调通常使用 $(x,y^*)$ 形式的指令和理想回答。它继续优化语言模型似然，但数据分布已经从自然网页文本转向指令、对话和任务示范：

~~~math

L_{SFT}(\theta)
=-\mathbb{E}_{(x,y^*)}
  \sum_{t=1}^{T}\log \pi_{\theta}(y_t^*\mid x,y_{<t}^*).

~~~

对于初学者来说，SFT 就是“给模型看一批优秀示范，让它模仿回答方式”。专家要注意，SFT 并没有显式告诉模型 rejected answer 为什么不好，它只提高 chosen demonstration 的概率。示范本身承担了任务定义、风格规范、拒答边界和事实表达的全部负担。

### 2.2 SFT 能解决什么

高质量 SFT 往往能改善：

1. 对话格式和角色边界。
2. 指令遵循和输出结构。
3. 工具调用的参数格式。
4. 不知道时的表达习惯。
5. 基本的安全拒答和替代帮助。

它还可以把预训练模型的知识转成更容易被用户调用的接口。一个 base model 可能知道某个事实，却不知道用户是在要求摘要、比较还是代码修复；示范将任务形式显式化。

### 2.3 SFT 的局限

示范数据通常只提供一条被认为可接受的回答，但现实中可能存在许多同样好的答案。强行模仿表面措辞可能造成模板化，也可能把标注员的偏好误当成任务本质。

此外，SFT 对错误示范很敏感。如果示范回答在事实、长度或安全边界上存在系统性偏差，模型会稳定地复制这种偏差。它也不天然解决“两个回答都看似合理，哪个更好”的相对判断问题。

这就是 preference data 出现的原因：让训练目标表达回答之间的差异，而不仅是一条正例。

## 3. 偏好数据与 reward model

### 3.1 Pairwise preference 的对象

给定同一个 prompt $x$，模型产生两个候选回答 $y_w$ 和 $y_l$，其中 $y_w$ 被标注为 preferred，$y_l$ 被标注为 less preferred。一个样本可以写成：

~~~math

\mathcal{D}_{pref}
=\{(x_i,y_{w,i},y_{l,i})\}_{i=1}^{n}.

~~~

这个数据形式表达的是相对顺序，不等于回答存在绝对质量分数。一个回答被选中，可能只是“两个候选中相对更好”，并不意味着它真实、完整或适合所有用户。

### 3.2 人类标注的实际工作

人类标注员通常需要先理解任务标准，再比较多个候选回答。标注协议可能要求关注正确性、相关性、风格、安全和是否遵循指令，也可能只要求选择整体更好的一项。

标注结果会受到几个因素影响：候选顺序、回答长度、格式、标注员专业知识、时间压力、任务难度和标准冲突。如果论文只报告 pair 数量，却没有说明标注协议、分歧率、重标注比例和 prompt 分布，读者无法判断偏好信号的可靠程度。

### 3.3 Bradley-Terry：把比较转成概率

一个常见的 pairwise preference 模型是 Bradley-Terry 形式。设 reward model 对回答给出标量 $r_{\phi}(x,y)$，则回答 $y_w$ 被偏好的概率可以写成：

~~~math

P(y_w\succ y_l\mid x)
=\sigma\bigl(r_{\phi}(x,y_w)-r_{\phi}(x,y_l)\bigr),

~~~

其中 $\sigma(z)=1/(1+e^{-z})$。对应的负对数似然为：

~~~math

L_{RM}(\phi)
=-\mathbb{E}_{\mathcal{D}_{pref}}
  \log\sigma\bigl(r_{\phi}(x,y_w)-r_{\phi}(x,y_l)\bigr).

~~~

初学者可以把 reward model 理解为一个学习出来的“比较器”：它不一定知道绝对真理，只是尝试复现标注员在给定数据上的排序。专家需要进一步追问 reward scale 是否可比、是否做了 prompt 分桶、是否存在长度泄漏，以及 reward model 是否会被策略模型带到训练数据覆盖之外。

### 3.4 reward model 不是人类价值函数

reward model 的输入通常是 prompt 和回答，输出是一个分数。它把复杂的偏好压缩为便于优化的标量，因而必然丢失信息。一个标注员可能同时考虑正确性、简洁性和风险，但 reward model 可能只学到“更长、更有礼貌、更像训练样本”的表面特征。

更危险的情况是 reward model 在离线候选上表现良好，策略模型更新后生成了分布外回答。策略可能找到 reward model 的漏洞，生成形式上满足评分器、实质上没有完成任务的答案。这是 reward hacking 的一种表现，不需要模型有恶意，只要优化器不断寻找高分区域就可能发生。

### 3.5 从排序数据到二元反馈

并不是所有场景都有成对比较。产品日志可能只有点赞/点踩，审核系统可能只有通过/拒绝，代码任务可能只有测试通过/失败。它们都能成为偏好信号，但数据语义不同：通过可能表示满足一个硬约束，点赞可能混合了速度、风格和用户情绪。

把这些信号放入同一个训练框架前，要记录标签来源、正负比例、缺失机制和目标范围。一个“没有点赞”可能是用户没有看到回答，也可能是不满意，不能默认等价于 undesirable。

## 4. RLHF：显式 reward 与策略优化

### 4.1 三阶段结构

经典 RLHF 路线通常包含三个阶段：

第一阶段用示范数据训练 SFT policy。它提供一个相对稳定的初始化，使模型先学会基本指令格式。

第二阶段用偏好比较训练 reward model。它尝试把人类对候选回答的相对判断映射成可优化的 reward。

第三阶段让 policy 生成回答，得到 reward，并通过 PPO 等强化学习算法更新 policy，同时用 reference model 或 KL 惩罚限制策略漂移。

用数据流表示为：

~~~text
prompt
  -> SFT policy 生成候选
  -> 人类比较候选
  -> reward model 学习排序
  -> policy 重新采样
  -> reward + KL 进入 RL 更新
~~~

这里最重要的变化是从离线模仿转向了策略优化。policy 的新输出可能不在原始 SFT 数据中，因此 RL 有机会探索新的行为，也同时增加了分布偏移和 reward hacking 的风险。

### 4.2 KL-regularized RL 的目标

一个常用的简化目标是：

~~~math

J(\theta)
=\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_{\theta}}
\left[r_{\phi}(x,y)
-\beta\,\mathrm{KL}\bigl(\pi_{\theta}(\cdot\mid x)\,\|\,\pi_{ref}(\cdot\mid x)\bigr)\right].

~~~

$\pi_{ref}$ 通常是冻结的 SFT 模型，$\beta$ 控制偏离参考分布的代价。$\beta$ 太大时，policy 被 reference 牢牢限制，reward 提升有限；$\beta$ 太小时，policy 可能快速离开已知的可用区域。

在实现中，KL 可能以 token 级近似、序列级估计或 reward shaping 的形式出现。读论文时不能只看到“使用 KL”就认为所有实现等价，要检查 KL 的位置、估计方式、是否裁剪以及它和 PPO clipping 的关系。

### 4.3 PPO 在这里做什么

PPO 比较新旧 policy 对同一动作或 token 序列的概率，并限制单次更新不要过大。一个教学化的 clipped objective 可以写成：

~~~math

L_{PPO}(\theta)
=\mathbb{E}_t\left[
\min\left(
r_t(\theta)A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right],

~~~

其中 $r_t(\theta)=\pi_{\theta}(a_t\mid s_t)/\pi_{old}(a_t\mid s_t)$ 是概率比，$A_t$ 是 advantage，$\epsilon$ 是裁剪范围。语言模型把 token 生成视为连续动作序列，还要处理 value model、长度、终止、批次构造和 reward 分配。

这解释了 RLHF 的工程复杂度：一次更新不仅依赖训练 batch，还依赖采样 policy、reward model、reference model、value estimate、KL 统计和序列长度。任何一个组件的版本或归一化变化，都可能改变结果。

### 4.4 RLHF 的优势

RLHF 的核心优势不是“用了 PPO”，而是它可以优化离线监督难以直接表达的相对行为目标，并允许 policy 在训练过程中生成新候选。对于开放式对话，很多质量标准无法写成单一 token-level label，比较候选回答更自然。

它还可以组合多种 reward：人类偏好、事实核查、格式验证、安全分类器或可执行测试。只要 reward 的来源和权重可解释，RL 就能把这些信号纳入策略更新。

### 4.5 RLHF 的代价与失败模式

RLHF 的主要成本包括：

1. 需要收集和维护偏好数据。
2. 需要训练、校准和监控 reward model。
3. 需要在线或准在线生成候选。
4. PPO 等算法对 batch、学习率、KL 和 value 估计敏感。
5. reward model 可能被策略过度优化。
6. 策略提升一个目标时可能损害其他目标。

因此论文中的“人类偏好提升”不能脱离 reward model 质量、KL 曲线、生成长度和能力回归单独解读。高 reward 但低真实任务成功率，是最需要展开的负结果之一。

## 5. InstructGPT：把 alignment 变成可测量的行为变化

### 5.1 论文的核心问题

InstructGPT 研究的不是“如何训练一个更大的语言模型”，而是如何让 GPT-3 类模型更好地遵循用户意图。它把用户请求、示范、人工排序和策略优化连接起来，展示了 alignment 可以产生与参数规模不同的能力增量。

论文报告了一个具有代表性的比较：在其人类偏好评估设置下，较小的 InstructGPT 模型可以得到比更大的原始 GPT-3 更高的偏好评价。这个结果的意义不是“小模型永远胜过大模型”，而是说明参数量和用户偏好不是同一个指标。

### 5.2 数据如何改变问题分布

InstructGPT 使用了标注员编写的 prompts、来自 API 使用分布的 prompts、示范回答和排序数据。与直接在互联网语料上继续训练相比，这些数据把目标从“预测自然文本”转向“完成真实用户请求”。

这个变化非常关键。模型能力不仅由参数决定，也由训练问题分布决定。如果训练 prompt 主要来自摘要和问答，模型可能改善这些任务，却没有覆盖长上下文、多轮工具调用或专业领域。论文阅读必须把用户 prompt 来源与评估 prompt 来源放在一起检查。

### 5.3 三阶段证据链

InstructGPT 的路线可以拆成：

1. 用示范训练 SFT 模型，让模型具备基本指令遵循能力。
2. 对候选回答进行人类排序，训练 reward model。
3. 用 PPO 优化 policy，并保留能力、truthfulness 和 toxicity 等回归评估。

每一步解决的问题不同。SFT 解决“不会按格式做”；reward model 解决“怎样比较多个回答”；PPO 解决“怎样让 policy 的生成分布朝高 reward 方向移动”。把三步合成一个“RLHF loss”会丢掉证据链。

### 5.4 论文结果应该怎样解释

人类偏好提升说明在指定评估者、prompt 和比较协议下，回答更常被选中。它不自动证明事实性提升，也不自动证明危险能力下降。论文需要同时看公开 NLP 任务、truthfulness、toxicity 和能力退化结果。

此外，标注员偏好不等于所有用户的偏好。标注员接受了任务训练，用户分布却可能不同；一个在短问答上偏好的回答，未必适合专业研究、代码执行或医疗咨询。结论应写成“在该协议下的行为变化”，而不是“模型完全对齐”。

### 5.5 InstructGPT 的可迁移边界

这项工作为后续 RLHF 建立了重要范式，但它不能回答所有对齐问题：

1. 人类排序如何覆盖罕见高风险请求。
2. reward model 如何识别长答案中的隐蔽错误。
3. policy 进入分布外区域后是否仍保持正确边界。
4. 多语言、专业领域和工具调用是否共享同一偏好标准。
5. 更强模型是否需要更强的监督和更严格的安全评估。

这正是后续 Constitutional AI、RLAIF、DPO 和可验证 reward 研究继续出现的原因。

## 6. Constitutional AI 与 RLAIF：把原则变成可扩展监督

### 6.1 人类逐条比较的扩展瓶颈

当模型输出变长、任务变复杂或内容变危险时，让人类逐条比较所有候选的成本会迅速增加。标注员可能缺少专业知识，也可能不愿直接阅读大量有害内容。Constitutional AI 试图把一部分监督从“人类逐样本打分”转成“人类先写原则，模型按原则批评和改写”。

这里的 constitution 可以是一组关于安全、诚实、尊重和帮助性的原则。原则不是自动产生的价值真理，它是人类选择的监督接口。它的优点是可复用、可审计、能作为生成 critique 的共同依据；它的缺点是原则可能冲突、遗漏和含糊。

### 6.2 critique 与 revision

一个简化的 Constitutional AI 流程是：

~~~text
初始回答
  -> 选择相关原则
  -> 模型生成 critique
  -> 模型根据 critique 生成 revised answer
  -> 用修订结果构造监督数据
~~~

初学者可以把它理解成让模型先问自己“这段回答违反了哪条规则”，再重写答案。专家要注意，critique 本身也是模型生成的，可能出现漏检、错误指责或只修改语气不修改事实的情况。因此 revised answer 不能因为包含了道德解释就被视为正确。

### 6.3 RLAIF 的反馈来源

RLAIF 通常用 AI feedback 替代部分人类 pairwise labeling。模型根据 constitution 或其他评价标准比较候选，再用这些偏好训练 reward/preference model，最后进入 RL 或直接偏好优化。

AI feedback 的优势是速度、规模和一致的格式；它的风险是评价者与被评价者共享模型偏差，可能形成 correlated error。若生成模型、judge 和训练 policy 之间存在相同的知识盲区，扩大数据规模不会自动修复错误，反而会把错误监督扩散得更广。

### 6.4 原则监督的证据边界

Constitutional AI 和 RLAIF 解决的是监督扩展问题，不是“AI 自己监督自己就安全”。至少还要检查：

1. 原则是否覆盖目标风险。
2. 原则之间冲突时如何排序。
3. AI judge 与人类 judge 的一致性和分歧样本。
4. revised answer 是否真实改善事实和安全，而非只改变措辞。
5. 是否有人工审计、红队、OOD 和系统级工具测试。

原则还可能造成过度拒答。如果规则只强调避免风险，而没有定义安全替代帮助，模型可能把正常教育、医疗信息和代码调试也拒绝掉。安全原则必须同时描述允许帮助的边界。

## 7. DPO：把 RLHF 的一部分变成直接偏好学习

### 7.1 DPO 的问题意识

RLHF 需要 reward model、在线采样、value estimate、PPO 和 KL 控制。DPO 的关键想法是：在特定 KL-regularized RL 假设下，可以把最优 policy 与 reference model 的概率比联系起来，从 pairwise preference 直接构造 policy loss。

DPO 的数据仍然是 $(x,y_w,y_l)$。它不要求先显式训练一个 reward model，也不需要在每个更新步都运行完整 PPO loop。简化的是优化链路，不是偏好目标本身。

### 7.2 DPO 的概率比

对给定 prompt $x$，定义完整回答的序列对数概率：

~~~math

\ell_{\theta}(y\mid x)
=\log \pi_{\theta}(y\mid x)
=\sum_{t=1}^{T_y}
\log \pi_{\theta}(y_t\mid x,y_{<t}).

~~~

DPO 关注 policy 相对于 reference 的对数概率比：

~~~math

\Delta_{\theta}(x,y)
=\ell_{\theta}(y\mid x)-\ell_{ref}(y\mid x).

~~~

如果 chosen 的相对概率高于 rejected，差值

~~~math

m_{\theta}
=\Delta_{\theta}(x,y_w)-\Delta_{\theta}(x,y_l)

~~~

就会变大。这个 margin 不是回答的事实质量分数，而是 policy 相对于 reference 对两条回答的偏好差。

### 7.3 DPO loss

常见的 DPO 目标可以写成：

~~~math

L_{DPO}(\theta)
=-\mathbb{E}_{(x,y_w,y_l)}
\log\sigma\left(
\beta\left[
\Delta_{\theta}(x,y_w)-
\Delta_{\theta}(x,y_l)
\right]
\right).

~~~

$\beta$ 控制偏好 margin 的温度和 reference 约束的相对强度。不同实现可能对序列 log-prob 做长度归一化、截断或采用不同的 padding mask，这些细节会改变实际目标。

初学者可以把它看成“提高 chosen 相对于 rejected 的相对概率”。专家还要问：这个提高是通过抬高 chosen、压低 rejected，还是两者同时发生？reference 是否冻结？回答长度是否被序列求和偏置？训练数据是否来自旧 policy？

### 7.4 DPO 与 KL-regularized RL 的联系

在一个理想化的 KL 正则 RL 问题中，给定 reward $r(x,y)$，最优 policy 具有如下形式：

~~~math

\pi^*(y\mid x)
=\frac{1}{Z(x)}\pi_{ref}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right),

~~~

其中 $Z(x)$ 是对同一 prompt 的归一化项。反过来可以得到：

~~~math

r(x,y)
=\beta\log\frac{\pi^*(y\mid x)}{\pi_{ref}(y\mid x)}
 +\beta\log Z(x).

~~~

对同一个 prompt 的两个回答做差时，$\log Z(x)$ 抵消，于是可以只用 policy/reference 的相对概率构造 pairwise loss。这是 DPO 理论直觉的来源。

这个推导依赖理想化假设：偏好数据与目标分布关系合适，policy 的概率能可靠估计，reference 约束和 reward 结构符合设定。它不是说所有 RLHF 训练都能无损替换成 DPO。

### 7.5 DPO 解决了什么

DPO 的工程吸引力包括：

1. 离线 pairwise 数据即可训练。
2. 不需要单独部署 reward model 参与每轮采样。
3. 不需要维护 PPO 的 rollout、value 和 advantage 链路。
4. 容易和开源 SFT 模型、训练框架结合。
5. 目标函数易于做 beta、数据量和 rejected 质量消融。

它降低了实验门槛，使更多团队能够研究偏好优化。但“更容易复现”不等于“更容易证明安全”，因为数据分布和评估偏差仍然存在。

### 7.6 DPO 的边界

DPO 常见的失败来源包括：

1. chosen 只是更长，而不是更正确。
2. rejected 太差，模型只学会区分明显坏答案。
3. reference 与训练 policy 不匹配。
4. 离线数据覆盖不到新策略生成的区域。
5. sequence log-prob 的长度处理导致隐性偏好。
6. beta 过大或过小造成更新不足或模型漂移。
7. 训练目标改善，但安全、事实或通用能力退化。

DPO 没有显式 reward model，因此没有 reward model 这一类错误；但偏好标签本身就是隐含 reward，数据偏差和目标错配仍可以被过度优化。

## 8. IPO：重新检查直接偏好目标的理论假设

### 8.1 为什么需要 IPO 这条线

DPO 把偏好优化变得简洁后，研究者继续追问：当 policy 过度拟合有限 pairwise 数据时，log-ratio margin 会不会无限增大？DPO 的 logistic loss 只要求 preferred margin 越大越好，可能出现过度自信、分布漂移或泛化变差。

IPO 所在的理论工作试图更仔细地分析 preference learning、KL 正则和估计误差之间的关系。它不是单纯换一个工程 API，而是提醒读者：一个看起来直接的 surrogate loss 仍然携带关于数据噪声、最优 policy 和正则化的假设。

### 8.2 一个教学化的 IPO 目标

在常见的简化写法中，IPO 不再只要求 margin 无限增大，而是把相对 log-ratio margin 拉向有限目标：

~~~math

L_{IPO}(\theta)
=\mathbb{E}\left[
\left(
m_{\theta}-\frac{1}{2\tau}
\right)^2
\right],

~~~

其中 $m_{\theta}$ 是 policy/reference 的 chosen-rejected margin，$\tau$ 是温度或正则相关参数。不同论文和实现的符号、系数和数据假设可能不同，这里使用的是帮助理解“有限目标 margin”思想的教学形式。

与 DPO 的 logistic loss 相比，这种目标体现了不同的 inductive bias：当 margin 已经达到目标后，继续扩大并不会带来无限收益。它可能缓解某些过度优化现象，也可能在数据本身不可靠时把错误目标拟合得更稳定。

### 8.3 读 IPO 论文要看什么

IPO 类工作应重点检查：

1. 理论中的 preference model 是否与实验数据一致。
2. 推导使用了什么 KL、采样和 realizability 假设。
3. 比较 DPO 时 beta、epoch、数据量和 reference 是否公平。
4. 训练 loss 的改善是否转成事实、任务和安全指标改善。
5. 对 noisy preference、重复 pair 和分布外 prompt 是否稳健。

理论上的更好性质不能替代实际评估。一个方法可能在合成偏好模型下有漂亮的上界，却在真实标注员分歧、长回答和多目标冲突下表现不稳定。

## 9. KTO：只有单样本好坏标签时的偏好优化

### 9.1 数据形态的变化

DPO 需要同一个 prompt 下的 chosen/rejected pair。现实系统经常只有单条反馈：用户点赞或点踩、审核通过或拒绝、测试通过或失败、专家标记为 desirable 或 undesirable。若为了凑成 pair 而错误配对，比较信号可能比原始二元标签更噪。

KTO 的出发点是直接使用单样本的 desirable/undesirable 标签，并借鉴 prospect theory 对正负反馈的非对称感知。它扩展的是数据入口，不是宣称二元反馈天然比 pairwise preference 更可靠。

### 9.2 二元信号的目标表达

可以用 $z\in\{1,0\}$ 表示样本是否 desirable，并用相对于 reference 的 log-ratio 表示 policy 改变程度：

~~~math

\rho_{\theta}(x,y)
=\log\frac{\pi_{\theta}(y\mid x)}{\pi_{ref}(y\mid x)}.

~~~

一个用于解释数据形态的抽象目标可以写成：

~~~math

J_{binary}(\theta)
=\mathbb{E}\left[
u(z)\,\rho_{\theta}(x,y)
-\lambda\,\mathcal{R}_{ref}(\theta;x,y)
\right],

~~~

其中 $u(z)$ 对 desirable 和 undesirable 样本赋予不同的效用，$\mathcal{R}_{ref}$ 表示相对 reference 的正则项。KTO 的具体实现还包含类别权重、KL 估计和非对称损失；上式只用来说明它不需要同 prompt 的成对回答。

### 9.3 KTO 的优点和风险

单样本反馈很适合从产品日志或审核流程构建数据，降低 pair 构造成本。但二元标签丢失了“为什么 A 比 B 好”的相对信息，也无法告诉模型一个 desirable 回答是否只是勉强合格。

尤其要检查正负样本不平衡。如果 99% 的回答没有明确负反馈，模型可能把“未被投诉”误当成高质量；如果审核标签只覆盖极端危险内容，模型学到的安全边界会非常稀疏。KTO 的数据优势必须以标签生成机制透明为前提。

### 9.4 KTO 与 DPO 的比较方式

公平比较时要保持目标 prompt 分布、训练 token、总样本计算和评估预算一致。不能让 KTO 使用海量日志、DPO 只使用小规模 pair，然后把差异归因于 loss；也不能把 KTO 的单样本标签与人工精排 pair 混在同一质量桶中。

应分别报告：标签覆盖率、正负比例、重复率、专家复核准确率、跨域泛化、长度变化、误拒和安全回归。KTO 解决的是“没有 pair 怎么用反馈”，不是“任何二元日志都能直接训练”。

## 10. ORPO：单阶段、无 reference 的偏好优化

### 10.1 ORPO 的工程动机

传统流程往往先 SFT，再执行 DPO 或 RLHF。每个阶段有不同数据、checkpoint 和训练配置，DPO 还需要冻结 reference model。ORPO 尝试把 SFT 和偏好优化合并到一个目标中，并去掉额外 reference model。

它保留 chosen response 的监督似然，同时加入 favored/disfavored response 的 odds-ratio 偏好项。这样模型在学习“如何回答”的同时，也学习“不要选择哪种回答风格”。

### 10.2 SFT 项与 odds-ratio 项

用 $P_{\theta}(y\mid x)$ 表示一个回答的序列概率，可以定义概念上的 odds：

~~~math

o_{\theta}(y\mid x)
=\frac{P_{\theta}(y\mid x)}{1-P_{\theta}(y\mid x)}.

~~~

一个教学化的组合目标是：

~~~math

L_{ORPO}(\theta)
=L_{SFT}(\theta)
+\lambda L_{OR}(\theta),

~~~

~~~math

L_{OR}(\theta)
=-\log\sigma\left(
\log o_{\theta}(y_w\mid x)
-\log o_{\theta}(y_l\mid x)
\right).

~~~

实际实现会对语言模型序列概率做长度归一化，并有具体的 odds-ratio 定义和权重。这里的关键思想是保留正例模仿，同时惩罚 rejected 方向，而不是把公式中的每个概率当成可以直接观测的独立类别概率。

### 10.3 没有 reference 不等于没有约束

ORPO 的 reference-free 只说明它没有显式维护一个冻结 reference policy。SFT 项本身提供了行为锚点，偏好项的权重提供了另一种约束。如果偏好项过强，模型仍可能改变原有知识、格式或安全边界。

所以比较 ORPO 和 DPO 时，要看总训练 token、训练 epoch、SFT 数据、偏好数据、模型初始化和推理模板是否一致。流程少一个模型，不代表实验变量只少一个。

### 10.4 ORPO 的适用问题

ORPO 更适合讨论训练链路简化和资源约束。如果项目已经有可靠 SFT 数据、偏好 pair 数量有限、希望减少 reference 显存，单阶段目标可能有吸引力。但若任务需要严格控制 policy 相对 base 的漂移，显式 reference 或其他 KL 约束可能更容易审计。

## 11. RLAIF 与 preference 方法的关系

RLAIF 不是一个与 DPO、KTO、ORPO 完全同层的 loss 名称。它首先描述反馈来源是 AI，而 DPO、KTO、ORPO 描述如何使用偏好或二元数据优化 policy。AI 产生的比较数据可以进入 reward model + RL，也可以进入 DPO 或其他直接偏好目标。

这一区分很重要。下面两个实验都可能被称为“RLAIF”：

1. 用 constitution 引导 AI judge 产生 pair，再训练 reward model 并做 PPO。
2. 用 AI judge 产生 chosen/rejected 数据，直接做 DPO。

它们的反馈来源相似，优化路径不同，计算成本、分布漂移和失败模式也不同。读论文时要分别记录 feedback generator、label policy、training objective 和 evaluation judge。

## 12. 方法谱系：用问题而不是缩写组织比较

| 方法 | 主要数据 | 主要优化对象 | 解决的工程问题 | 仍需警惕 |
| --- | --- | --- | --- | --- |
| SFT | 指令和示范 | 监督似然 | 让模型先会按任务回答 | 没有显式负例，示范偏差会复制 |
| RLHF/PPO | 排序、reward model、rollout | reward 加 KL 的 policy | 支持在线探索和复杂 reward | 链路复杂，reward hacking 和漂移 |
| DPO | chosen/rejected pair | policy/reference margin | 去掉显式 reward model 和 PPO | 离线分布、长度偏差、beta |
| IPO | pairwise preference | 有限目标 margin | 从理论上缓解过度优化问题 | 理论假设和真实噪声的差距 |
| KTO | desirable/undesirable | 二元反馈效用 | 使用单样本反馈 | 标签噪声、类别不平衡、弱相对信息 |
| ORPO | 示范加偏好 pair | SFT 加 odds-ratio | 合并阶段并去掉 reference | 漂移、训练公平性、权重敏感 |
| RLAIF | AI 生成的偏好或 reward | 可接 RL 或直接偏好目标 | 扩展监督规模并减少人工负担 | judge 偏差、原则遗漏、相关错误 |

这张表的作用是让读者先找到研究问题，再看算法。论文的贡献可能不是“发明了一个更好的 loss”，也可能是降低了标注成本、改进了可验证任务、提高了在线探索效率，或让安全原则更容易审计。

## 13. 偏好数据：算法效果的隐形上限

### 13.1 Prompt 分布

如果训练 prompt 主要是简单问答，模型很难从偏好数据中学到复杂工具调用、长上下文引用和多轮状态管理。需要记录 prompt 的来源：人工编写、真实用户、合成任务、红队请求、专家领域或线上日志。

训练与评估 prompt 的近重复会让结果虚高，过度清洗又可能删除真实困难样本。数据版本、时间切分和污染检测必须和偏好结果一起报告。

### 13.2 chosen 不一定代表正确

标注员选中的回答可能只是相对更好，也可能更短、更礼貌、更有结构。对于代码和数学，表面可读性不能代替执行测试；对于事实问答，自信语气不能代替证据；对于安全请求，拒绝措辞不能代替边界判断。

数据制作应尽量加入可验证信号：代码测试、数学 verifier、引用支持、事实核查、专家复审和安全 policy 标签。偏好排序可以作为综合信号，但不能承担所有事实证明责任。

### 13.3 长度偏差

设 chosen 和 rejected 的 token 长度分别为 $T_w,T_l$。如果 reward 或模型 log-prob 使用序列求和，长度会同时影响概率总量；如果 judge 喜欢更详细的回答，标签还会产生人类层面的长度偏好。至少要报告：

~~~math

R_{longer}
=\frac{1}{n}\sum_{i=1}^{n}
\mathbf{1}[T_{w,i}>T_{l,i}].

~~~

当 $R_{longer}$ 极高时，不能直接把 chosen 胜出解释为质量提升。应构造长度匹配 pair、截断对照、按长度分桶的 win rate，并检查更短回答是否因为省略关键事实而被错误奖励。

### 13.4 多目标与标注员分歧

两个标注员可能一个优先正确性，一个优先安全谨慎；这不一定意味着有人标错，而是目标未被明确排序。报告可以保留分歧率、重标注一致性和各维度标签，而不是把所有差异隐藏在一个 chosen 字段中。

如果项目最终只需要一个 policy，仍应在训练前决定硬约束和软目标：危险工具操作不能用 helpfulness 抵消；事实回答的引用缺失不能仅靠语气得分补偿。权重和硬约束属于任务定义，不是调参后才随意选择的结果。

### 13.5 合成和 AI 反馈数据

合成 preference 数据能扩大覆盖，但生成器、judge 和被训练模型可能共享错误。需要保存 prompt、候选、judge 解释、原则版本、生成模型、采样参数和人工抽查结果。

当 AI judge 给出的标签被再次用于训练同一个或相近模型时，必须留出由独立模型、专家和真实任务构成的 holdout。否则模型可能只学会模拟 judge 的风格，离线分数提高而真实用户体验没有改变。

## 14. 评估 preference optimization

### 14.1 Win rate 的含义

对于一组成对评估，最基本的 win rate 是：

~~~math

WinRate
=\frac{N_{win}}
       {N_{win}+N_{loss}+N_{tie}}.

~~~

分母必须明确是否包含 tie、无效回答和 judge 无法判断的样本。没有分母和样本构成的“提升 5%”没有可比性。

### 14.2 成对差异和不确定性

如果每个 prompt 同时评估 baseline 和新模型，可以先定义成对差异：

~~~math

\Delta_i=s_{new,i}-s_{base,i},
\qquad
\bar\Delta=\frac{1}{n}\sum_i\Delta_i.

~~~

成对设计能抵消一部分 prompt 难度差异，但仍需报告 bootstrap 区间、seed 差异和 prompt 分桶。平均 win rate 上升可能只来自少数长答案或某一类简单任务。

### 14.3 LLM judge 的独立性

LLM-as-a-Judge 便宜、可扩展，却容易受长度、位置、格式和模型家族影响。若 judge 与被评模型共享训练数据或风格，它可能高估熟悉的回答。实验至少要做回答顺序交换、长度控制、多个 judge、人工抽样和 disagreement audit。

judge 解释文字也不是独立证据。一个 judge 可以给出非常有说服力的理由却判断错误，尤其在数学、代码、事实和工具轨迹任务上。能执行的任务应优先使用 verifier 或真实环境结果。

### 14.4 多维评估矩阵

对齐结果至少可以按以下维度切片：

| 维度 | 典型问题 | 结果指标 |
| --- | --- | --- |
| 有用性 | 是否完成用户目标 | task success、人工偏好 |
| 真实性 | 是否有事实错误或无依据断言 | factuality、引用支持、专家复核 |
| 安全性 | 是否提供危险帮助 | attack success、policy violation |
| 误拒 | 正常请求是否被拒绝 | valid request refusal rate |
| 指令/格式 | 是否遵守 schema 和边界 | parse rate、constraint success |
| 泛化 | 是否在新领域、新语言、新长度上保持行为 | OOD 分桶 |
| 系统性 | 是否产生越权调用和状态副作用 | tool audit、trace replay |

一个总分无法替代这张矩阵。特别是安全场景，允许任务成功和危险任务拒绝通常是两个不同的条件分布，不能只报总体平均。

### 14.5 能力回归

偏好训练可能提高对话 win rate，却损害代码、数学、长上下文或知识覆盖。应保留 base、SFT、preference model 的同一套 regression suite，并记录提示模板、解码参数和工具权限。

如果新模型的 safety score 提高，同时正常请求拒绝率上升，结论应写成安全与可用性的 trade-off，而不是简单的安全提升。对齐决策需要解释接受了什么代价。

## 15. Preference optimization 与 safety 的边界

### 15.1 安全偏好不是完整安全系统

偏好训练可以让模型更倾向于拒绝某些危险请求、提供安全替代或在不确定时澄清。但安全还依赖 policy 定义、输入处理、检索权限、工具执行器、沙箱、监控、事故响应和人工接管。

模型行为的安全性可以粗略写成条件结果：

~~~math

Risk_{system}
=P(\text{harmful outcome}
\mid\text{model},\text{tools},\text{permissions},\text{environment}).

~~~

改变模型偏好只改变这个条件概率的一部分。若工具权限没有收紧，系统风险未必随拒答率下降。

### 15.2 过度拒答与安全替代

安全数据若只包含“拒绝”样本，模型可能学到看到敏感词就拒绝，不会区分教育、研究、新闻分析和真实攻击。更好的数据要包含允许、拒绝、澄清、降级和安全替代等动作，并标明为什么动作不同。

评估中要同时报告危险请求的漏拒率和正常请求的误拒率。两者的权重取决于风险，但不能用一个模糊的“安全性”掩盖可用性损失。

### 15.3 工具和 agent 场景

在工具场景中，模型输出的自然语言偏好不是唯一结果。一个模型可能在聊天评估中更 helpful，却更容易把未授权的参数传给工具。训练和评估应加入工具 schema、权限、环境状态、幂等性和回滚。

特别要区分“拒绝文字写得很好”和“危险动作没有发生”。后者需要执行器日志和状态差分，不能由 judge 读一段文本推断。

## 16. Reasoning 与可验证 reward：偏好优化的新压力

### 16.1 Outcome reward 与 process signal

开放式对话常依赖人类偏好；代码、数学和部分推理任务可以得到更客观的 outcome signal，例如测试是否通过、答案是否匹配、证明是否由 verifier 接受。可验证 reward 减少了语言风格偏差，但只覆盖 verifier 能检查的目标。

可以把一个任务的 reward 拆成：

~~~math

R
=R_{outcome}
+\lambda R_{process}
-\mu R_{unsafe}.

~~~

如果只奖励最终答案正确，模型可能生成不可解释、不可审计甚至投机的路径；如果过度奖励过程格式，模型可能输出漂亮的伪推理。process reward 也需要验证其与真正任务因果相关，而不是只奖励更长的轨迹。

### 16.2 Test-time compute 不是免费的 alignment

self-consistency、多候选生成、verifier reranking 和多次工具调用都可能提高任务成功率。但它们增加了每次请求的 token、延迟和失败面。应报告：

~~~math

C_{request}
=C_{prefill}
+kC_{decode}
+C_{verifier}
+C_{tool},

~~~

其中 $k$ 是候选数量。只比较 best-of-$k$ 的质量而不比较 $k=1$ 的基线，会把额外搜索预算误写成模型 alignment 能力。

### 16.3 可验证 reward 也会被投机

如果测试存在漏洞，模型可能利用 verifier 的盲点；如果 reward 只检查格式，模型可能输出空壳；如果训练数据包含测试模式，模型可能记忆答案。可验证不等于不可被 hacking，必须做隐藏测试、变体测试、环境隔离和人工抽样。

这条线与 DPO 的关系也不是“reasoning 只用 RL”。可以把 verifier 产生的通过/失败转成 binary data 做 KTO，把通过答案和失败答案组成 pair 做 DPO，也可以用 reward 做在线 RL。选择取决于探索需求、反馈延迟、数据覆盖和成本。

## 17. Worked case：一个 DPO 提升为何可能只是长度偏差

### 17.1 现象

某团队用 20,000 条偏好 pair 训练 DPO，离线 judge win rate 从 0.51 上升到 0.58。新模型的回答平均长度增加 42%，正常任务看起来更详细，但事实性没有独立提升。

如果只看 win rate，结论会是 DPO 有效。如果检查数据，发现 88% 的 chosen answer 比 rejected 更长，且 judge 对结构化长回答存在明显偏好，那么更合理的解释是模型学到了长度和语气特征。

### 17.2 反事实评估

可以做三个对照：

1. 从 chosen 和 rejected 中截取相近长度，再重新比较。
2. 将回答顺序随机交换，检查 position bias。
3. 用事实核查、代码执行和人工专家评估替代单一 judge。

假设长度匹配后的 win rate 只从 0.51 变为 0.52，事实性差异区间跨过 0，代码通过率不变，而 token 成本增加 1.42 倍。此时报告应写成“原始 judge 指标提升主要由长度相关信号解释，尚无足够证据证明任务质量提升”，而不是“DPO 提升了模型能力”。

### 17.3 修复数据和目标

修复可以包括长度匹配采样、加入事实和执行 verifier、按任务分桶、降低 judge 对格式的依赖、补充短而正确的 chosen answer，并重新训练多个 beta。修复后仍要在独立 prompt、正常/危险任务和系统成本上回归。

这个案例说明算法、数据和评估不能拆开看。DPO loss 下降只说明优化目标被拟合，不能单独证明回答更真实。

## 18. Worked case：拒答率提高并不等于安全改善

某安全偏好数据集把大量危险请求与拒答回答配成 chosen/rejected。训练后，危险测试集的直接帮助率下降，但正常的化学教育、漏洞修复和政策分析请求也有更多拒答。

如果只报告危险请求的 refusal rate，模型会显得更安全。进一步分桶后发现：

1. 真正高风险请求的漏拒下降很少。
2. 正常请求的误拒明显上升。
3. 安全替代回答变少，用户只能得到空泛拒绝。
4. 规则关键词附近的拒答概率异常高。

正确的修复不是简单降低安全 loss，而是补充边界样本，区分允许/拒绝/澄清/转向，加入专家审计和工具权限测试。安全目标必须包括“拒绝危险动作”和“继续提供允许帮助”两部分。

## 19. 小规模复现：如何公平比较偏好方法

### 19.1 先固定研究问题

不要一上来复现整篇 RLHF 论文。可以选择一条窄 claim，例如：

> 在相同 SFT 初始化、相同 prompt 和相同训练 token 下，DPO 是否比 SFT 提高长度控制后的任务成功率，同时不增加安全误拒？

这个 claim 已经指定了模型状态、数据、比较对象、主指标和副作用。它比“DPO 是否有效”更容易复查。

### 19.2 实验契约

至少固定：

1. base model 和 checkpoint。
2. tokenizer、chat template 和截断规则。
3. train/validation/test prompt 划分。
4. chosen/rejected 构造和数据版本。
5. beta、epoch、batch、学习率和随机种子。
6. 解码参数、最大输出长度和工具权限。
7. 评估 judge、人工抽样和 verifier 版本。

若比较 SFT、DPO、KTO 或 ORPO，还要决定总 token、总 GPU 时间或总 wall time 哪一个公平。不同方法的目标计算不同，不能只比较最终 checkpoint 的训练 loss。

### 19.3 评估矩阵

一份最小评估可以包含：

| 类别 | 最小测量 |
| --- | --- |
| 偏好 | 人工或独立 judge 的 pairwise win rate |
| 任务 | 任务成功率、代码测试或数学 verifier |
| 事实 | 引用支持、事实核查或专家复审 |
| 安全 | 漏拒、误拒、安全替代和攻击变体 |
| 形式 | schema/格式解析成功率 |
| 成本 | 输入输出 token、延迟、显存和单位成功成本 |

所有指标都要保存 prompt 分桶和失败样本。均值只适合概览，失败类型才帮助判断目标是否真的改善。

### 19.4 复现层级

资源不足时可以分层：先验证 DPO loss 和 reference log-prob 的实现，再运行小模型的行为 smoke test，然后做完整数据量和多 seed 实验，最后才尝试在线或工具环境。每一层都要说明它复现的是代码、数值、行为还是论文主张。

如果没有原始偏好数据，使用替代数据只能称为方法复现或机制复现，不能称为原始结果的直接复现。数据不同往往比 loss 不同更能改变结果。

## 20. 可运行的偏好审计 demo

下面的纯 Python demo 用合成的序列 log-prob 计算 DPO margin，并同时审计长度偏差和安全 holdout。它没有训练模型，只演示如何把一个平均 loss 拆成可解释信号。

~~~python
from math import exp, log


def sigmoid(value):
    return 1.0 / (1.0 + exp(-value))


def dpo_row(beta, row):
    policy_margin = row["policy_w"] - row["policy_l"]
    reference_margin = row["ref_w"] - row["ref_l"]
    margin = policy_margin - reference_margin
    probability = sigmoid(beta * margin)
    loss = -log(probability)
    return margin, loss


rows = [
    {
        "name": "fact_helpful",
        "policy_w": -1.05,
        "policy_l": -1.95,
        "ref_w": -1.20,
        "ref_l": -1.80,
        "chosen_len": 84,
        "rejected_len": 70,
        "safe_holdout": True,
    },
    {
        "name": "length_bias",
        "policy_w": -0.80,
        "policy_l": -1.50,
        "ref_w": -1.00,
        "ref_l": -1.30,
        "chosen_len": 160,
        "rejected_len": 72,
        "safe_holdout": True,
    },
    {
        "name": "unsafe_pair",
        "policy_w": -1.25,
        "policy_l": -1.65,
        "ref_w": -1.10,
        "ref_l": -1.40,
        "chosen_len": 96,
        "rejected_len": 91,
        "safe_holdout": False,
    },
    {
        "name": "format_success",
        "policy_w": -0.90,
        "policy_l": -1.35,
        "ref_w": -1.10,
        "ref_l": -1.20,
        "chosen_len": 62,
        "rejected_len": 58,
        "safe_holdout": True,
    },
]

beta = 0.2
audits = [dpo_row(beta, row) for row in rows]
mean_margin = sum(item[0] for item in audits) / len(audits)
mean_loss = sum(item[1] for item in audits) / len(audits)
longer_chosen = sum(
    row["chosen_len"] > row["rejected_len"] for row in rows
) / len(rows)
safe_holdout = sum(row["safe_holdout"] for row in rows) / len(rows)

signals = {
    "mean_margin": round(mean_margin, 4),
    "mean_loss": round(mean_loss, 4),
    "chosen_longer_rate": round(longer_chosen, 4),
    "safe_holdout_coverage": round(safe_holdout, 4),
}
actions = []
if longer_chosen >= 0.75:
    actions.append("length_match_and_rejudge")
if safe_holdout < 0.80:
    actions.append("expand_independent_safety_holdout")
decision = "continue_after_bias_audit" if actions else "continue_to_behavior_eval"

print(f"mean_margin={signals['mean_margin']}")
print(f"mean_loss={signals['mean_loss']}")
print(f"chosen_longer_rate={signals['chosen_longer_rate']}")
print(f"safe_holdout_coverage={signals['safe_holdout_coverage']}")
print(f"actions={actions}")
print(f"decision={decision}")
~~~

这个 demo 的 `mean_loss` 只说明合成 log-prob 满足了某种 pairwise 目标。`chosen_longer_rate` 和 `safe_holdout_coverage` 则提醒我们，偏好结果是否被长度或安全覆盖问题污染。真实训练还要加入独立模型、人工抽样、任务 verifier、多个 seed 和真实成本。

## 21. 常见失败模式与诊断

### 21.1 win rate 上升但任务成功率不变

优先检查长度、格式、judge 熟悉度和 prompt 泄漏。将回答做长度匹配，改用执行 verifier，并按任务分桶。如果提升只出现在一个 judge 或一个 prompt 模板，不能归因于 alignment。

### 21. reward 上升但人工评价下降

可能是 reward hacking、reward model 过拟合或策略离开训练分布。检查 reward model 在新策略样本上的人工校准、KL、长度、重复短语和错误类型，降低更新幅度并加入新鲜偏好数据。

### 21. 安全分数上升但正常请求大量拒绝

检查安全标签是否只有极端拒绝样本，补充允许/澄清/安全替代样本，按风险等级和领域分桶。不要只把拒答率作为安全指标。

### 21. DPO 训练不稳定或模型漂移

检查 reference checkpoint、beta、序列长度归一化、chosen/rejected 的质量差异、训练 epoch 和学习率。比较 policy/reference 的 log-ratio 分布，而不只看平均 loss。

### 21. KTO 的结果受日志数量影响

检查点赞/点踩是否存在曝光偏差、用户选择偏差和未反馈样本的含义。先做标签质量抽样和类别重权，再和同预算的 pairwise 数据比较。

### 21. ORPO 看似省资源但结果不可比

检查它是否使用了额外 SFT 数据、不同 epoch、不同初始化或更大的有效 token。reference-free 只减少一种模型状态，不代表总计算和总监督相同。

### 21. RLAIF 离线分数高但人工分歧大

检查 constitution 版本、AI judge 与人类 judge 的分歧样本、生成模型和被训练模型的相关性。增加独立 judge、专家审核和真实任务 holdout，不能只扩大同一个 AI judge 的数据。

## 22. 如何写一份可复查的 alignment 研究报告

一份成熟报告不应只写“某方法 win rate 提升”。至少要交代：

1. **行为目标**：希望改善 helpfulness、truthfulness、安全、格式还是可验证任务。
2. **反馈来源**：人类、AI、规则、用户日志、专家或执行测试。
3. **数据对象**：示范、pair、二元标签、scalar reward、轨迹和 verifier 输出。
4. **优化路径**：SFT、reward model + RL、DPO、IPO、KTO、ORPO 或混合流程。
5. **参考约束**：reference model、KL、SFT 锚点或其他漂移控制。
6. **主要结果**：win rate、任务成功、事实、安全、误拒、格式和成本。
7. **不确定性**：seed、prompt 分桶、标注分歧、judge 分歧和区间。
8. **失败样本**：长度偏差、reward hacking、事实错误、越权、过度拒答和分布外失败。
9. **证据边界**：哪些来自论文，哪些来自本次复现，哪些只是教学合成数据。
10. **下一步**：补数据、改评估、限制范围、回滚或进行在线 shadow。

写报告时要把“模型更喜欢 chosen”与“用户得到更好结果”分开。前者是训练目标的直接测量，后者需要任务、事实、安全和系统证据共同支持。

## 23. 资料与证据边界

预训练目标与人类偏好的早期路线可参考 [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) 和 [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)。这些论文支持人类比较、reward modeling 和策略优化的研究脉络；它们不证明单一偏好指标能代表所有用户价值。

InstructGPT 主要参考 [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)。论文支持其公开实验中的 SFT、reward model、PPO、人工偏好和能力评估；具体结果依赖 prompt、标注协议、模型版本和评估设置。

RLHF 中的 PPO 背景可参考 [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)。PPO 论文支持 clipped policy optimization 的算法背景；本章的语言模型序列和 KL 公式是教学化抽象，不能替代具体实现的 rollout、value 和 token masking 细节。

Constitutional AI 参考 [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)，RLAIF 参考 [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267)。这些资料支持原则监督、critique/revision 和 AI feedback 的研究方向；AI feedback 的质量、原则覆盖和人类一致性仍需独立验证。

DPO 参考 [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)。论文支持在特定 KL-regularized preference model 假设下使用直接 policy loss 的理论和实验路线；本章的 log-prob 公式省略了实现中的 padding、长度、数据混合和优化细节。

IPO 参考 [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)。论文支持从理论角度分析 DPO 类目标、偏好学习和正则化假设；本章的有限 margin 形式是教学写法，实际符号和参数化应以论文与代码版本为准。

KTO 参考 [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)，ORPO 参考 [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)。它们分别支持单样本 desirable/undesirable 反馈和无额外 reference 的单阶段偏好优化研究；具体数据要求、损失实现和收益不能脱离模型、标签、训练预算与评估集。

安全和事实性评估可参考 [TruthfulQA](https://arxiv.org/abs/2109.07958) 以及 [Helpful and Harmless Language Models](https://arxiv.org/abs/2204.05862)。它们支持把 truthfulness、helpfulness 和 harmlessness 分开评估的必要性；本章的系统权限、工具审计和事故分析属于工程扩展，不是这些论文单独证明的结果。

本章中的数值、worked case、DPO 审计 demo 和决策字符串均为教学构造。真实项目应保存数据 snapshot、标注协议、原则版本、reference checkpoint、tokenizer、训练配置、judge/validator 版本、失败 trace 和成本记录，并明确区分论文事实、项目实测与教学近似。

## 24. 结语：对齐是目标、数据、优化和系统的共同问题

Alignment 论文线的演进可以这样理解：SFT 把可接受行为写成示范，RLHF 用 reward 和策略优化扩展相对目标，Constitutional AI 与 RLAIF 试图扩大监督来源，DPO 把部分 RL 链路重参数化为直接偏好学习，IPO 重新审视理论假设，KTO 扩展到二元反馈，ORPO 简化训练阶段，而可验证 reward 把一部分任务从主观偏好拉向执行结果。

这些方法没有消除共同难题：反馈是否代表目标，数据是否覆盖真实分布，模型是否利用了表面偏差，评估是否独立，能力是否发生回归，安全行为是否能延伸到工具和环境。读一篇 alignment 论文时，最有价值的问题不是“它比 DPO 新在哪里”，而是：它改变了哪个监督瓶颈，依赖了什么假设，在哪些数据和评估上成立，失败时谁承担后果。

当一条结论同时交代行为目标、反馈来源、优化机制、评估分母、代价和边界时，alignment 才从算法缩写变成了可以复查的研究对象。
