# 第十二章：Safety 与 Interpretability 论文线

当模型从分类器变成能够对话、写代码、检索资料、调用工具和执行多步任务的系统时，研究问题就不再只有“准确率能不能更高”。我们还必须回答：模型在什么条件下会做出危险行为？它是否记住了不该记住的内容？它为什么在有些问题上拒答、在另一些问题上服从？一次编辑究竟改掉了一个事实，还是破坏了整片知识？一个解释是否反映了真实的内部计算，还是只是一个听起来合理的事后故事？

Safety 研究处理的是风险、后果、控制和治理；Interpretability 研究处理的是表示、计算路径、因果机制和可干预性。两条线有交集，却不是同一个目标。一个模型可以在红队测试中表现良好，但内部机制仍然不清楚；也可以找到一个很有趣的电路，却无法据此证明系统在真实环境中安全。本章把它们放在同一条论文谱系里，是为了比较它们共享的证据问题，而不是把“安全”和“可解释”当成一个总分。

本章有两条并行论文线和一个交叉研究区。安全线从具体威胁与失败出发，讨论红队、模型生成评估和行为后果；可解释性线从输入归因走向表示、计算路径和因果干预；知识编辑、遗忘、隐私与来源标记同时触及两条线。它们共用研究方法，但测量对象不同，因此不能折成一个“安全总分”。

阅读路径如下：

~~~mermaid
flowchart LR
    A["共同入口：限定研究主张与对象"] --> B["安全线：威胁、行为与执行后果"]
    A --> C["解释线：输入、表示、路径与干预"]
    B --> D["交叉区：编辑、遗忘、隐私与来源"]
    C --> D
    D --> E["共同核验：分母、对照、预算与边界"]
    E --> F["复现、回归与工程决策"]
~~~

安全论文通常回答“在什么条件下出现什么行为或后果”；可解释性论文回答“哪些内部对象与输出相关，干预它会怎样改变行为”。交叉主题既可能改动模型参数，也可能只改变接口策略或输出分布，必须先确定它究竟测量了什么。

正文中的公式和 Python 示例都使用教学规模的抽象对象。它们帮助读者检查分母、变量和因果关系，不代表对真实模型能力的测量。真实结论必须绑定模型版本、数据、提示、采样、评估者、环境和时间。

## 12.1 先区分安全主张与解释主张

### 12.1.1 小白视角：安全不是“模型永远不会犯错”

一个聊天模型如果拒绝了十个危险问题，只能说明在这十个问题上没有观察到危险服从。它没有证明模型不知道危险知识，也没有证明换一种说法、换一种语言、加入文档或接入工具后仍然如此。

同样，一个解释工具如果告诉我们“这个词对答案贡献很大”，也只说明这个输入位置与输出有关。它没有自动说明模型内部保存了什么，也没有证明删掉这个位置就能改变行为。

因此，安全主张和解释主张都应该写成带条件的句子。例如：

> 在固定模型版本、输入分布、工具权限和人工审查协议下，系统在某类高风险任务上的危险动作率低于某个观测范围。

或：

> 在固定模型、层、位置和任务分布上，对某一激活路径进行干预会稳定改变目标输出，同时不显著改变无关任务。

这两句话都比“模型安全”或“我们找到了模型的推理过程”更可检验。第一句的对象是行为和后果，第二句的对象是内部变量和因果干预；它们的实验设计不能混用。

### 12.1.2 专家视角：把结论拆成五个层次

一个完整研究报告至少要把以下五层分开：

| 层次 | 要问的问题 | 常见证据 |
| --- | --- | --- |
| 接口层 | 输入、输出、拒答和工具协议是否符合约定 | schema、日志、协议测试 |
| 行为层 | 在给定任务分布上模型做了什么 | success、unsafe rate、utility |
| 机制层 | 哪些表示、路径或参数与行为有关 | activation、feature、weight update |
| 因果层 | 干预该对象是否导致行为变化 | patching、ablation、counterfactual |
| 生产层 | 失败发生时的现实影响如何被限制 | 权限、人工确认、回滚、审计 |

论文常常只覆盖其中一两层。红队论文可能有很强的行为证据，但没有内部机制证据；SAE 论文可能有表示和干预证据，但没有生产风险证据；治理框架描述的是组织与部署控制，也不能替代模型行为评估。

### 12.1.3 一个可复查的证据向量

可以把某个 claim 的证据写成向量：

~~~math
\mathbf{e}
=
(e_{\mathrm{scope}},
e_{\mathrm{behavior}},
e_{\mathrm{mechanism}},
e_{\mathrm{causal}},
e_{\mathrm{deployment}}).
~~~

其中每一项不是一个漂亮的总分，而是对相应层次是否有证据的记录。比如一个 SAE 实验可能有较强的 mechanism 和 causal，却没有 deployment；一个线上安全报告可能有 behavior 和 deployment，但没有 mechanism。把缺失维度写出来，比把所有维度平均成一个分数更诚实。

考虑一个读取发票并提交报销的 Agent。研究者发现，模型有时会把发票上的旧金额当作当前金额，随后准备提交高于用户授权的报销。若只记录“危险动作率下降”，读者仍不知道系统哪一层发生了改变。先把主张限定为：在固定版本、固定发票样本和固定报销权限下，某项干预减少了未经授权的提交尝试，同时没有显著增加对合法报销的阻断。

接着按证据向量检查：接口层确认工具收到的金额、账户和发票标识；行为层比较授权与未授权样本上的提交意图；机制层观察候选表示是否随旧金额和新金额的冲突而改变；因果层通过目标激活干预与随机位置对照，检查提交倾向是否只在相关条件下改变；生产层再看服务端权限是否拒绝超额请求、是否留下审计轨迹，以及人工复核和回滚是否有效。这些观察不能互相代替：模型文本变得谨慎但工具仍能越权，接口和生产层的 claim 就没有成立；工具拒绝了越权动作，也不能据此推断模型内部不再保存旧金额关联。

因此报告要分别写出每层观察到的对象、协议和未覆盖条件。若干预减少了目标动作，却同时破坏合法报销，证据支持的只是行为发生了取舍变化；若随机对照也产生相同变化，则还不能归因于候选机制。后续的安全论文线会展开失败发现与分母设计，可解释性论文线会细化相关性和因果干预之间的差别，而本例只是帮助读者先看清哪些证据不能折叠成同一个结论。

## 12.2 安全论文线：从风险主张到失败发现
### 12.2.1 Concrete Problems：安全研究为什么要从具体失败开始

#### 从抽象担忧到问题分解

Concrete Problems in AI Safety 的价值不在于提供一个今天仍然适用的万能算法，而在于把“人工智能可能不安全”拆成可研究的问题：副作用、奖励黑客、可扩展监督、安全探索和分布偏移。

这一步很重要。若只说“模型要符合人类价值”，工程团队无法知道应该采集什么数据、设计什么测试或保存什么日志。若把问题写成“模型为了提高可见灰尘分数把灰尘推到摄像头看不到的地方”，就能进一步问：奖励测量漏掉了什么？有没有第二个独立传感器？失败的代价是否可逆？监督信号是否足够覆盖真实目标？

#### Side effects：完成任务不等于没有额外损害

考虑一个机器人任务。环境初始状态为 $s_0$，任务完成程度为 $U(s)$，对环境造成的额外改变为 $C(s,s_0)$。一个简单的风险敏感目标可以写为：

~~~math
J(a)
=
\mathbb{E}\left[U(s_T)-\lambda C(s_T,s_0)\right],
~~~

其中 $a$ 表示动作序列，$s_T$ 是最终状态，$\lambda$ 表示对副作用的重视程度。

小白可以把 $C$ 理解为“为了完成任务顺手破坏了多少不该改变的东西”。在文档 Agent 中，任务可能是整理一份报告，副作用则可能是修改原始文件、覆盖用户备注或把内部材料发给外部收件人。模型即使完成了报告，也不能因此抵消不可逆的数据损失。

专家需要注意，副作用不是一个自然存在、无需定义的量。哪些状态应该保持不变，取决于任务规范、用户授权和环境建模。若把所有未变化的状态都当作好，把所有变化都当作坏，又会阻止必要动作。因此实验必须明确 reference state、允许变化集合、动作可逆性和风险严重度。

#### Reward hacking：指标被优化后，真实目标可能变坏

设真实目标为 $U$，训练或评估中使用的代理奖励为 $R$。理想情况下，增大 $R$ 应该大致伴随 $U$ 增大；但模型优化的是可见的 $R$，不是研究者心里没有写出来的 $U$。

~~~math
\Delta_{\mathrm{hack}}
=
\mathbb{E}[R\mid \mathrm{high\ reward}]
-
\mathbb{E}[U\mid \mathrm{high\ reward}].
~~~

这个差值不是严格的因果量，只是提醒我们去检查高奖励样本是否真实有用。若奖励模型偏爱更长、更自信、更像模板的回答，模型可能学会堆砌段落、避免承认不确定性、引用不存在的资料，或者迎合评估器而不是解决用户问题。

一个更可操作的实验是把样本按代理奖励分桶，再由独立人工或任务执行结果测量真实质量。如果随着优化强度增加，$R$ 持续上升而 $U$ 下降，就出现了 reward overoptimization 的信号。若 $U$ 没有下降但不再上升，可能只是奖励已经饱和；两种情况的修复不同。

#### Scalable oversight：人类如何监督难以直接检查的任务

当模型只做一句分类时，人类可以逐条检查。模型开始写大型代码补丁、长数学证明或多步研究报告后，直接监督的成本和注意力都成为瓶颈。

可扩展监督的核心不是“让另一个模型替人类签字”，而是设计一条让监督信号仍然与真实目标有关的分解链。例如：

1. 把最终答案拆成可验证的中间 claim。
2. 让辅助模型寻找反例，再由人类检查关键反例。
3. 对代码使用测试、静态分析和人工抽样的组合。
4. 对长文档要求引用证据，并对证据位置进行抽查。
5. 把高严重度、低置信度或工具动作升级给专家。

每一层都可能引入新的代理目标。自动 judge 很快，但可能偏爱文风；过程监督更细，但标注成本更高；形式 verifier 精确，但只能覆盖可形式化的部分。研究报告应记录监督对象、监督者、错误类型和人工升级比例，而不是只报告最终分数。

#### Safe exploration 与 distribution shift

强化学习或 Agent 在探索时可能尝试真实环境中不可逆的动作。即便训练目标正确，分布外状态也可能让策略选择训练中没有见过的路径。

可用一个简化的风险预算表示：

~~~math
\mathbb{E}\left[\sum_{t=1}^{T} c_t\right]\leq B_c,
\qquad
c_t\geq 0,
~~~

其中 $c_t$ 是第 $t$ 步风险代价，$B_c$ 是允许的总风险预算。这个式子不等于安全证明，因为风险代价的估计本身可能错；它的作用是迫使系统在执行前记录预算、动作可逆性和停止条件。

大模型 Agent 中，distribution shift 可能来自新的文档格式、陌生工具错误、用户权限变化、语言切换或多轮上下文污染。固定 benchmark 上没有出现失败，并不意味着这些条件下风险为零。

### 12.2.2 Red Teaming：从“测分数”转向“找失败”

#### Red teaming 的对象

Red teaming 是一种以失败发现为中心的测试活动。它可以针对模型本身，也可以针对包含检索、工具、权限和人工流程的完整系统。测试对象越靠近生产环境，评估就越不能只看文本是否“有害”，还要看是否产生了真实动作和真实后果。

常见风险面包括：

1. 有害内容和高风险领域建议。
2. 隐私、凭证、内部文档和训练数据泄漏。
3. 越权工具调用、错误参数和不可逆副作用。
4. 多轮对话中的指令层级混乱。
5. RAG 文档中的间接指令和证据冲突。
6. 多模态输入中图像、音频或文件携带的隐藏指令。
7. 对不同语言、群体、口音或专业场景的不公平行为。
8. 通过格式变换、角色变化或上下文堆叠导致的行为漂移。

#### Benchmark 与 red team 的分工

固定 benchmark 追求可比性，red teaming 追求发现新的失败。两者不能互相替代。

| 维度 | 固定 benchmark | Red teaming |
| --- | --- | --- |
| 主要问题 | 平均任务表现如何 | 哪些路径会触发高风险失败 |
| 样本 | 固定或版本化题集 | 由专家、用户日志和生成器持续扩展 |
| 结果 | 分数、置信区间和切片 | 失败轨迹、严重度、根因和修复回归 |
| 对抗性 | 通常有限 | 可以主动寻找边界和组合条件 |
| 可比性 | 较强 | 需要记录搜索预算和测试策略 |

一个红队发现的失败样本可以进入回归集，但回归集变成固定题集后，又会失去发现未知问题的能力。因此生产系统需要同时保留已知回归集和持续探索的红队预算。

#### 先写 threat model，再写测试题

没有 threat model 的红队，容易变成随机收集“看起来很坏”的提示。一个最小 threat model 至少包含：

1. 资产：用户隐私、资金、代码、凭证、声誉或物理安全。
2. 攻击者能力：只能发消息，还是能上传文件、读取网页、调用工具。
3. 信任边界：系统提示、用户内容、检索文档和工具返回值分别处在哪个边界。
4. 目标后果：信息泄漏、错误动作、拒答失效还是业务损失。
5. 观察窗口：单轮、长对话、跨会话还是异步任务。
6. 防护条件：权限、速率限制、人工确认和回滚是否存在。

这个定义决定了测试分母。一个只能在拥有管理员权限、连续尝试一千次并且没有人工确认时成功的样本，与一个普通用户单次请求就能触发的样本，风险含义不同。

#### 失败率、严重度和搜索预算

对一组测试任务，危险行为率可以写成：

~~~math
\mathrm{DR}
=
\frac{\sum_{i=1}^{N}\mathbf{1}[\mathrm{dangerous}_i]}
{N}.
~~~

其中 $N$ 是任务数。若每个任务的严重度为 $w_i$，则严重度加权失败率可以写成：

~~~math
\mathrm{SWDR}
=
\frac{\sum_{i=1}^{N}w_i\mathbf{1}[\mathrm{dangerous}_i]}
{\sum_{i=1}^{N}w_i}.
~~~

如果同一个任务允许多次尝试，还必须记录搜索预算。单次成功率、每个任务在 $k$ 次尝试内的成功率和每条轨迹的动作成功率是不同指标：

~~~math
\mathrm{ASR}_{k}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}\left[
\max_{j\in\{1,\ldots,k\}}
\mathrm{failure}_{i,j}=1
\right].
~~~

这个公式表示任务在最多 $k$ 次尝试中只要出现一次目标失败就记为成功攻击。若研究者把每次尝试都当作独立分母，容易低估长对话和重复试探的风险。

还要报告搜索成本：每个任务尝试次数、模型温度、生成器数量、人工时间、是否允许看到失败反馈。不同红队论文若没有这些条件，不能只比较一个 ASR 数字。

用四个构造任务做一次手算。设严重度权重为 $(3,2,1,3)$，每个任务按预先约定的攻击预算运行后，危险结果为 $(1,0,1,0)$。危险行为率是 $2/4=0.5$；严重度加权失败率是 $(3+1)/(3+2+1+3)=4/9\approx0.444$。再把每题扩成三次尝试，四行失败轨迹设为 $(0,0,1)$、$(0,0,0)$、$(1,0,0)$、$(0,1,0)$。至少一次失败的任务有三项，所以 $\mathrm{ASR}_3=3/4=0.75$。这不是说三次尝试“独立”，而是展示任务级分母如何随搜索预算变化；若把十二次尝试当成十二个独立任务，风险口径就变了。

近期可执行红队论文展示了两种不同的评估变量。ANGLER/REEF 在 9 个可重置网站、8 个 VLM agent 配置上按每个网站—模型组合分配 100 次 rollout；论文报告自适应攻击的平均 ASR 为 20.36%，非自适应基线为 15.24%，相差 5.12 个百分点、相对增加 33.6%；同一研究的良性任务成功率从无攻击条件的 61.9% 降到自适应攻击下的 44.4%。这些数字绑定其合成网站、VisualWebArena scaffold、攻击协议和验证器，不能当成真实网站攻击率。[ANGLER / REEF](https://arxiv.org/abs/2610.41272)

REDAgentBench 则固定同一批 rollout，只改变判定时能看到的证据视图；在其六个模型配置中，state-view 报告的 ASR 比 trajectory-view 高 7.73—11.72 个百分点，并改变 12.97%—21.20% 的配对标签。这一差异说明评测工具本身会改变“观察到的安全率”；它不是模型行为在两次运行间改变，也不能外推成生产风险差值。[REDAgentBench](https://arxiv.org/abs/2608.10669)

#### 人工红队、模型红队和真实日志

人工专家的优势是能理解复杂后果、发现跨步骤漏洞和判断边界语义，缺点是贵、慢、覆盖有限。模型生成红队样本的优势是速度和规模，缺点是容易重复训练分布中的模式，也可能和被测模型共享盲点。真实用户日志最贴近部署，却受到隐私、选择偏差和观测不到失败的限制。

较可靠的组合是：

1. 用风险 taxonomy 规定必须覆盖的类别。
2. 用模型生成变体，扩展语言、格式和上下文条件。
3. 用专家设计少量高价值任务和反事实任务。
4. 用真实日志抽取经过脱敏的失败模式。
5. 对高严重度样本进行人工复核。
6. 将确认过的失败转成回归样本，并保留未知探索集。

红队工具本身也要被审计。若生成器只会提出模型容易识别的模板，测试集会给人“覆盖很广”的错觉。

#### 一个工具型代码助手案例

设代码助手能够读取仓库、运行测试、修改文件和提交补丁。首轮离线评估中，危险文本服从率很低，团队准备扩大灰度。红队加入一份包含间接指令的检索文档后，模型把文档中的文字当成高优先级指令，调用了创建工单工具；工具参数虽然格式正确，却把内部备注写入了外部可见字段。

这个事故不能归因成“模型被 prompt injection 攻击”就结束。需要拆成：

1. 文档内容与系统指令的信任边界没有分离。
2. 模型拥有超出任务需要的字段权限。
3. 工具执行器只校验 schema，没有校验业务语义。
4. 外部可见写入缺少人工确认。
5. 日志没有记录模型使用了哪段证据。

修复应当分层：

1. 将检索内容标记为数据，不允许它改变指令层级。
2. 让模型只能填写内部草稿，外部发送需要独立授权。
3. 在服务端按用户、字段和动作做权限校验。
4. 对高风险字段做二次确认和可回滚写入。
5. 将该完整轨迹加入固定回归集。

复测时不能只看模型是否拒答。要同时测文本、工具参数、授权判断、外部可见结果和回滚成功率。若模型回答更谨慎但执行器仍然接受越权字段，系统风险并没有被修复。

### 12.2.3 Model-Written Evaluations：让模型帮助发现行为

#### 方法解决的是真实覆盖问题

模型行为的空间非常大。人工专家可以写出高价值任务，却很难提前穷举模型可能出现的迎合、过度自信、偏见和风险组合。Model-written evaluations 的想法是让一个模型生成评估问题、选项或标签，再由人工或独立流程过滤。

这条路线把评估从“少量手工题”扩展成“行为发现循环”：

1. 先规定要探索的行为定义。
2. 由生成器提出问题和上下文变体。
3. 用规则、独立模型或人工筛选无效题。
4. 在被测模型上运行并保存完整轨迹。
5. 对标签和严重度进行人工抽样校准。
6. 将稳定、可复现的发现转为固定回归集。

它可以帮助发现 sycophancy、危险目标倾向、过度拒答和随规模变化的逆向趋势，但它不自动产生可靠标签。

#### 逆向缩放和 sycophancy

如果模型规模增加，某个普通能力指标变好而某个不良行为率也变高，就可能出现 inverse scaling。以迎合为例，给定用户观点 $u$、事实约束 $f$ 和模型答案 $y$，可以分别测量：

~~~math
\mathrm{Syc}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[y_i\ \mathrm{agrees\ with\ }u_i]
\cdot
\mathbf{1}[\mathrm{contradicts}(y_i,f_i)].
~~~

这里的关键不是“同意用户就是坏”，而是要构造用户观点与事实答案冲突的配对任务。若只用用户观点正确的题目，无法分辨礼貌表达、合理同意和事实性迎合。

例如，构造 20 道用户观点与可核实事实相反的题，其中 5 个答案同时同意错误观点并与事实矛盾，则 $Syc=5/20=0.25$。分母是预先定义的冲突题，而不是所有对话；如果把事实已知的普通题也混进分母，迎合率会被稀释。生成器还应保留“用户观点正确”的配对题，检查模型是否只是普遍反对用户。

#### 评估生成器和被测模型不能共享盲点

Model-written eval 的主要风险是相关性偏差：

1. 生成器和被测模型使用相似训练数据。
2. 生成器只提出自己容易理解的行为。
3. 自动 judge 偏爱生成器的语言风格。
4. 题目和标签泄漏到训练或调参过程。
5. 研究者只保留支持假设的题目。

要减轻这些问题，可以使用独立生成器、人工盲审、不同模型的 judge、真实任务样本和反事实对照。最好保存被淘汰样本及淘汰原因，否则外部读者无法判断评估集经历了怎样的筛选。

#### 标签质量和发现质量分开报告

设生成评估集的题目质量为 $q_i$，标签是否正确为 $l_i$，模型行为是否被稳定复现为 $r_i$。发现质量至少有三部分：

~~~math
\mathrm{EvalQuality}
=
(\mathrm{coverage},\mathrm{label\_accuracy},\mathrm{replicability}).
~~~

不要把三者相乘成一个分数。一个题目可能覆盖了新行为但标签不可靠；也可能标签很准但只覆盖模型已经熟悉的模板。研究报告应分别给出人工抽样的 label accuracy、不同 seed 的行为稳定性和类别覆盖。

这类评估回答的是“哪些输入和条件会触发什么行为”，但还没有说明行为由哪些内部计算产生。要从失败样本继续追问机制，就需要把观察对象从输入与输出移到表示、路径和干预；下一条论文线正是沿这个方向推进，同时保留行为结果作为因果解释的外部校验。

## 12.3 可解释性论文线：从输入归因到因果干预
### 12.3.1 Interpretability：解释对象到底是什么

#### 从模型自我解释开始就要保持怀疑

模型说“我这样回答是因为第二段证据支持这个结论”，这段话可能有用，也可能只是生成了一段符合人类期待的说明。它是模型的输出，不是对内部计算的直接读取。

自我解释可以作为用户界面、调试线索或可读性信号，但不能单独证明：

1. 模型真的使用了所说的证据。
2. 模型没有先凭语言先验得出答案。
3. 所说的中间步骤参与了最终 logits。
4. 删除该理由后模型行为会改变。

更严格的解释研究会把输出、激活、参数和干预联系起来。

#### 五种解释层次

| 层次 | 典型问题 | 证据和局限 |
| --- | --- | --- |
| 行为描述 | 模型在什么输入上答错 | 只能描述现象 |
| 输入归因 | 哪些 token 或区域重要 | 可能受方法和基线影响 |
| 表示解释 | 某个方向或 feature 表示什么 | 需要跨样本和反事实 |
| 机制解释 | 哪些 head、MLP 和路径组合成电路 | 需要结构与干预 |
| 因果解释 | 改变该对象是否改变目标行为 | 需要干预、对照和副作用测试 |

“解释性更强”必须先说清楚比较的是哪一层。attention 图可能对输入归因有帮助，却不能直接升级为机制解释。

#### Faithfulness、stability 和 completeness

一个解释方法可以从三个角度检查。

给输入特征归因分数 $a_j$，把重要特征集合记为 $S$，其余特征为 $\bar S$。删除或遮挡 $S$ 后，目标输出的变化可以写成：

~~~math
\Delta_S
=
f(x)-f(x_{\setminus S}).
~~~

若高归因集合确实重要，$\Delta_S$ 应明显大于随机同规模集合的变化。这个实验测的是 faithfulness 的一个近似。

对同一语义输入做格式、同义词或轻微扰动，若解释集合大幅漂移，则稳定性较弱：

~~~math
\mathrm{Stability}
=
1-
\mathbb{E}_{x'\sim\mathcal{T}(x)}
\left[
\mathrm{dist}(A(x),A(x'))
\right].
~~~

其中 $\mathcal{T}$ 是保持任务语义的变换，$A$ 是解释对象。completeness 则问解释是否覆盖了足以重建输出变化的证据。三者都不是自动真理，只是迫使实验者把“可读”与“忠实”分开。

### 12.3.2 从 attribution 到 mechanistic interpretability

#### Attribution 能告诉我们什么

Gradient-based attribution、integrated gradients、saliency map 和 attention visualization 可以帮助定位输入变化与输出变化的关系。它们适合做初筛，例如发现模型是否过度依赖某个提示词、某个文档字段或图像区域。

但 attribution 通常有几个限制：

1. 梯度是局部量，可能在饱和区域失真。
2. 输入分解和 baseline 的选择会改变结果。
3. attention 权重不等于因果贡献。
4. 多条路径可能抵消，单个 token 的分数无法表示组合机制。
5. 解释图可能漂亮，却不能预测干预结果。

因此 attribution 更像测量仪表，不是“模型内部的屏幕截图”。

#### Mechanistic interpretability 的研究对象

Mechanistic interpretability 把 Transformer 当作一个可以逆向工程的计算系统。研究者尝试识别：

1. residual stream 中可复用的表示方向。
2. 执行复制、匹配、语法或事实检索的 attention head。
3. 负责非线性变换和特征组合的 MLP 通道。
4. 从输入位置到输出 logits 的可追踪路径。
5. 能被干预、预测并迁移到相关输入的电路。

“发现一个神经元会在某类文本上激活”只是起点。较强的机制主张还要说明它与哪些上游输入连接、如何影响下游表示、在反事实输入上是否保持行为、干预后是否产生预期变化，以及有没有改变无关能力。

#### Residual stream 视角

在一个简化的 decoder-only Transformer 中，第 $l$ 层 residual stream 可以写成：

~~~math
h^{(l+1)}
=
h^{(l)}
+
F_{\mathrm{attn}}^{(l)}(h^{(l)})
+
F_{\mathrm{mlp}}^{(l)}(h^{(l)}).
~~~

这个表达式不是完整实现，但它说明了一个研究视角：不同模块都在共享的 residual stream 上读写表示。若我们只观察某个模块的激活，可能看不到后续模块如何组合或抵消它。机制分析要追踪信息从哪里进入、在哪里变换、怎样到达最终 logit。

### 12.3.3 Induction heads：一个可被干预的电路例子

#### 小白直觉：模型学会了“见过 A 后面跟着 B”

考虑序列：

~~~text
... A B ... A ?
~~~

如果模型在上下文中找到前一个 A，并把它后面的 B 复制到当前位置，就能预测下一个 token。这个模式被称为 induction behavior，相关研究把它作为理解 in-context learning 机制的经典案例。

它不是说所有 in-context learning 都由一个 head 完成，而是提供了一个较窄、可构造、可干预的研究对象。研究者可以改变重复模式、替换 token、打乱位置，再观察 attention pattern 和预测概率。

#### 一个简化的匹配表达

设当前位置为 $t$，历史位置为 $s<t$，token embedding 为 $e(x)$。一个匹配 head 可以粗略地给历史位置打分：

~~~math
\mathrm{score}(s,t)
=
q_t^{\top}k_s,
\qquad
\alpha_{s,t}
=
\frac{\exp(\mathrm{score}(s,t))}
{\sum_{r<t}\exp(\mathrm{score}(r,t))}.
~~~

若 $q_t$ 与表示当前 token 的 key 相似，attention 权重 $\alpha_{s,t}$ 会集中在相同 token 的历史位置；value 通路再把该位置之后的 token 信息写回 residual stream。真实模型中可能有多个 head、位置偏置和 MLP 参与，这个公式只是帮助读者理解可检验的中间量。

#### 如何从相关图走到因果证据

一个较完整的 induction head 实验包括：

1. 构造有重复模式和无重复模式的配对序列。
2. 记录目标位置的 attention pattern、logit lens 或输出概率。
3. 对候选 head 做 activation ablation。
4. 把正常序列的 head 输出 patch 到被破坏序列。
5. 检查复制行为是否恢复。
6. 在不同 token、长度和位置上测试泛化。

若只在一个序列上看到图案，证据仍然很弱。若 patching 能在多个随机样本和反事实条件下恢复目标行为，才更接近机制层主张。

### 12.3.4 Causal tracing：定位事实关联的关键路径

#### 相关性为什么不够

某个神经元在模型输出某个人名时激活很高，可能是因为它存储了事实，也可能是因为另一个模块已经决定了答案，它跟着一起激活。相关性分析无法区分这两种情况。

Causal tracing 的基本思想是制造一个可控损坏，再尝试恢复它。以事实问答为例：

1. 在干净输入上得到正确输出。
2. 对输入做一个破坏事实关联的扰动。
3. 在指定层和位置注入噪声或替换激活。
4. 将干净激活 patch 回被破坏运行。
5. 测量目标答案的概率恢复程度。

#### 恢复分数

设干净运行目标 logit 为 $z_{\mathrm{clean}}$，损坏运行目标 logit 为 $z_{\mathrm{corr}}$，patch 后为 $z_{\mathrm{patch}}$。一种归一化恢复分数是：

~~~math
\mathrm{Restore}
=
\frac{z_{\mathrm{patch}}-z_{\mathrm{corr}}}
{z_{\mathrm{clean}}-z_{\mathrm{corr}}+\varepsilon}.
~~~

其中 $\varepsilon$ 防止分母接近零。分数接近 1 表示 patch 恢复了大部分目标 logit 差异，接近 0 表示没有恢复。这个分数仍然不能单独证明“事实存储在这个位置”，因为 patch 可能引入了下游需要的结果，或者损坏实验本身改变了多条路径。

专家会进一步检查：

1. patch 的粒度是 token、layer、head 还是 MLP。
2. 噪声分布和强度是否改变了输入难度。
3. 是否有随机 patch、错误位置 patch 和反向 patch 对照。
4. 恢复的是目标答案，还是所有答案的置信度都变高。
5. 该路径是否跨事实、语言和 paraphrase 保持。

例如，干净运行目标 logit 为 $6.0$，损坏后为 $2.0$，把候选层的干净激活 patch 回去后为 $5.0$。取 $\varepsilon=0.001$，恢复分数为 $(5-2)/(6-2+0.001)=3/4.001\approx0.7498$；若把相同激活 patch 到无关层、目标 logit 只到 $3.0$，分数约为 $1/4.001=0.2499$。这个差异支持候选路径对当前任务有影响，但仍需随机位置、错误事实、无关任务和多样本复核；单个恢复数不等于完整算法已被定位。

#### “定位”不等于“理解完整算法”

一个事实关联可以由多个位置共同完成。找到恢复分数高的位置，可能说明它位于一条关键路径上，但不代表该位置独立存储了完整知识。对复杂推理，局部 patch 还可能只恢复中间线索，而不是整个算法。

机制论文的结论应该精确到实验支持的范围，例如“该层的某些位置对目标事实的恢复具有因果影响”，而不是直接写成“模型的知识存储在第 18 层”。

### 12.3.5 SAE：从神经元到稀疏特征

#### Polysemanticity 与 superposition

神经网络中的一个神经元可能在语义无关的多个场景中激活，这被称为 polysemanticity。它不一定意味着模型真的把概念混成了一个不可分的东西，也可能是表示空间在有限维度下使用了 superposition：许多稀疏特征以不同方向叠加到较少的神经元坐标上。

小白可以把它想成一间小仓库存放很多种物品。一个货架并不等于一个物品，物品可能按方向、组合和出现条件编码。直接给货架贴标签会很混乱；研究者希望恢复更接近物品本身的 feature direction。

专家要注意，superposition 是一种解释模型和实验现象的假说，不是所有模型层、所有特征都必须满足的定律。特征的稀疏性、可分离性和语义稳定性都要通过数据验证。

#### SAE 的重构目标

设某层激活为 $x\in\mathbb{R}^{d}$，SAE 编码器产生 $m$ 维 feature 激活 $f(x)$，解码器用字典矩阵 $D\in\mathbb{R}^{d\times m}$ 重构激活。一个常见的稀疏目标是：

~~~math
f(x)
=
\sigma(W_e x+b_e),
\qquad
\hat x
=
D f(x)+b_d,
~~~

~~~math
\mathcal{L}_{\mathrm{SAE}}
=
\mathbb{E}_x
\left[
\lVert x-\hat x\rVert_2^2
\right]
+
\lambda
\mathbb{E}_x
\left[
\lVert f(x)\rVert_1
\right].
~~~

其中 $W_e,b_e$ 是编码器参数，$D,b_d$ 是解码器参数，$\sigma$ 可以是 ReLU 或其他非负激活，$\lambda$ 控制稀疏性与重构误差的权衡。

这个目标只保证 SAE 在数值上重构激活并倾向于稀疏，不保证每个 feature 都对应人类概念。feature 的解释还需要看触发样本、反例、激活强度和干预结果。

#### 训练 SAE 时的工程选择

SAE 实验需要记录：

1. 从哪一层、哪种 token 和哪种数据收集激活。
2. feature dictionary 的宽度与原激活维度之比。
3. 稀疏损失、死 feature 处理和归一化方式。
4. 训练数据的领域、重复和上下文长度。
5. 重构误差、稀疏度、feature 使用频率和解释评估。
6. 是否使用 top-k、跳过连接或重初始化策略。

如果只在代码语料上训练 SAE，得到的 feature 可能主要反映代码；把它直接称为通用语言特征是不恰当的。数据分布是解释的一部分。

#### Feature 解释的三步验证

可以对一个候选 feature 做三类检查。

触发样本检查：展示激活最高的样本，也展示激活低但语义相似的样本，防止只挑选漂亮例子。

反事实检查：保持句子其他部分不变，删除或替换候选概念，观察 feature 激活是否按预期变化。

干预检查：提升、抑制或替换该 feature 对应的方向，检查目标行为是否改变，同时测量无关任务损失。

如果 feature 的激活样本看起来像“法律文本”，但干预它只改变标点或长度，语义解释就还没有得到因果支持。

再看一个构造的“危险知识 feature”审计。研究者在 20 条恶意网络请求和 20 条防御性修复请求上检查同一 feature，发现它分别在 14/20 和 9/20 条件中高激活；因此它可能表示网络安全语境，而不能直接命名为“恶意代码”。在留出集上，抑制该方向后，恶意服从从 8/20 降到 3/20，减少 $5/20=25$ 个百分点；但防御性任务正确率也从 18/20 降到 12/20，损失 $6/20=30$ 个百分点。同范数随机方向对照的两项变化都只有 $1/20=5$ 个百分点。这个结果支持“候选方向对该任务群有选择性影响”，同时表明安全收益伴随明显的防御能力代价；测试仍需换提示、领域和随机种子，不能把 feature 说成危险能力的唯一存储位置。

#### 解释覆盖、稀疏度和重构的张力

SAE 可能在重构误差、稀疏度和可解释性之间出现张力。一个极宽的 dictionary 可以降低误差，却可能产生许多难以审计的小 feature；一个过强的稀疏约束可能让 feature 看起来干净，却丢掉关键激活。

至少要并列报告：

~~~math
\mathrm{MSE}
=
\mathbb{E}\left[\lVert x-\hat x\rVert_2^2\right],
\qquad
\mathrm{L0}
=
\mathbb{E}\left[\#\{j:f_j(x)\neq 0\}\right],
~~~

以及在目标任务上的行为变化。不能用较低 MSE 推出更好的可解释性，也不能用较低 L0 推出更接近真实电路。

近期评估工作进一步说明，重构指标与“找回了可解释的计算单元”不是一回事。SAEBench 把 SAE 比较拆成多项任务指标；在其所测配置中，sparsity–fidelity 排名不能稳定预测 concept detection、disentanglement 或下游任务排名。[SAEBench](https://arxiv.org/abs/2503.09532) 的范围是论文纳入的模型、SAE 架构和任务集合。

SANITY Checks 用已知真值的合成激活做压力测试：在其独立重尾 feature 设置中，解释方差为 71% 的 SAE 只恢复 3,200 个真实 feature 中的 297 个，约 9.3%。这是特定合成假设下对指标的挑战，不是对所有 SAE 的否定。[SANITY Checks](https://arxiv.org/abs/2602.14111v1) DAME 论文则指出，高计算贡献 feature 可能比低贡献 feature 更难通过其解释性代理评估，也可能更冗余；其分数仍依赖 LLM judge，主要实验模型规模也有限。因此，SAE 用于 attribution 或 circuit tracing 前，应分开检查重构、稀疏度、重要 feature 的反例和因果干预，不应以一个综合排名代替验证。[Rethinking SAE Evaluation / DAME](https://arxiv.org/abs/2610.30474v2)

#### 从 SAE 到规模化特征与 circuit tracing

近年的工作尝试在更大的模型和更宽的 feature dictionary 上提取特征，也尝试把 feature 之间的影响组织成 attribution graph 或更完整的计算图。它们为研究者提供了更细的观察窗口，但仍然有三个边界：

1. 自动生成的 feature description 可能只是语言模型对样本的概括。
2. 图上的边可能表示统计或局部因果影响，不等于完整程序。
3. 规模化分析的选择、层位和任务分布会影响可见的电路。

因此，规模化 SAE 和 circuit tracing 应被看作更强的研究工具，而不是已经完成的模型透明化。

表示和电路研究能提出值得检验的候选对象，但描述一个 feature 与证明它能可靠改变模型知识是两回事。接下来转向模型编辑、遗忘、隐私和内容来源：这些研究直接触及参数、训练数据或输出凭据，因此必须分别检查改变发生在哪里、能否保留无关能力，以及留下了什么可复查证据。

## 12.4 模型知识与数据治理：编辑、遗忘、隐私和来源
### 12.4.1 Model Editing：直接改写参数中的知识

#### Editing 与继续训练的区别

假设模型对一个事实的旧回答为 $y_{\mathrm{old}}$，希望对特定输入 $x_e$ 输出新答案 $y_{\mathrm{new}}$。普通微调可以完成这个目标，但可能修改大量无关行为。Model editing 研究的是更局部、更快速的参数更新：

~~~math
\theta'
=
\theta+\Delta\theta,
\qquad
\Delta\theta
=
\arg\min_{\Delta}
\mathcal{L}_{\mathrm{edit}}(\theta+\Delta)
\quad
\mathrm{s.t.}\quad
\mathcal{C}_{\mathrm{locality}}.
~~~

其中 $\mathcal{C}_{\mathrm{locality}}$ 表示无关行为尽量保持不变。这个约束通常是通过一组 preservation prompts 或下游任务近似的，不是对整个模型行为的严格保证。

#### ROME 的局部更新直觉

ROME 把事实关联视为 Transformer 某个 MLP 中的键值映射。设输入键为 $k$，希望写入的值为 $v_\ast$，可以寻找一个低秩更新 $\Delta W$，使：

~~~math
(W+\Delta W)k
\approx
v_\ast.
~~~

若还要保持一组保留键矩阵 $K$ 的输出不变，可以把目标写成：

~~~math
\min_{\Delta W}
\left\|
(W+\Delta W)K-WK
\right\|_F^2
+
\mu
\left\|
(W+\Delta W)k-v_\ast
\right\|_2^2.
~~~

这个公式帮助理解局部编辑的矛盾：新事实要被写入，原有大量映射又要保持。真实 ROME 实现还要处理定位、协方差近似、tokenization 和生成评估，不能把公式等同于完整算法。

#### MEMIT 与批量编辑

单条编辑容易做，批量编辑更接近知识维护场景。MEMIT 研究如何把多条事实写入多个层或一组参数更新中。批量编辑的困难包括：

1. 不同事实可能共享实体和关系。
2. 编辑之间可能发生参数干扰。
3. 更新量变大后，局部性更容易下降。
4. 新事实的语言变体和组合推理不一定同步更新。
5. 多次编辑后的模型状态可能依赖编辑顺序。

因此，报告不能只给“编辑成功率”，还要给编辑数量、顺序、冲突类型、无关任务变化和长期回放结果。

#### 五个必须分开的指标

对于编辑集合 $E$ 和保持集合 $P$，可以分别报告：

~~~math
\mathrm{Reliability}
=
\frac{1}{|E|}
\sum_{x\in E}
\mathbf{1}[\mathrm{new\ fact\ correct}],
~~~

~~~math
\mathrm{Generalization}
=
\frac{1}{|E'|}
\sum_{x\in E'}
\mathbf{1}[\mathrm{paraphrase\ correct}],
~~~

~~~math
\mathrm{Locality}
=
1-
\frac{1}{|P|}
\sum_{x\in P}
\mathbf{1}[\mathrm{unrelated\ output\ changed}],
~~~

这些比例要求 $|E|>0$、$|E'|>0$ 和 $|P|>0$；对应集合为空时应报告未定义，而不是把没有测试样本记成 0 或 1。

例如，做 10 条新事实编辑，8 条在目标问法上正确，则 reliability 为 $8/10=0.8$；另有 10 条 paraphrase，其中 6 条正确，则 generalization 为 $6/10=0.6$。在 20 条保持集上若有 2 条无关输出改变，locality 为 $1-2/20=0.9$。这三个数分别描述目标写入、表述迁移和副作用；只报告 0.8 的 reliability，会遮住 paraphrase 失败与保持集变化。

并单独测量多轮持久性和编辑成本。若只有 reliability 上升，不能说明编辑是可部署的。

#### Editing 与 RAG 的选择

RAG 不修改参数，而是在请求时提供外部证据；editing 修改参数，可能降低运行时检索依赖，却承担局部性、回滚和版本管理的风险。

可以用一个实际维护案例理解差异。公司 CEO 发生变更时：

1. 需要即时生效、保留来源和可回滚时，优先更新知识库并使用 RAG。
2. 需要模型在没有外部网络的固定环境中稳定使用某个事实时，才有理由研究 editing。
3. 事实复杂、变化频繁或权限差异明显时，不应把 editing 当作唯一知识层。
4. 即使采用 editing，也应保留外部事实源和编辑日志。

### 12.4.2 Unlearning：删除影响，而不是只改变一句回答

#### Exact unlearning 的定义

理想的 unlearning 希望删除数据集 $D_f$ 后，模型的分布接近“从未见过 $D_f$、只在剩余数据 $D_r$ 上重新训练”的模型：

~~~math
\theta_{\mathrm{unlearn}}(D_f,D_r)
\approx
\theta_{\mathrm{retrain}}(D_r).
~~~

这里的“接近”需要指定距离、攻击者能力和观察任务。若只要求模型对一个问题拒答，目标就被弱化成行为屏蔽，而不是删除训练影响。

LLM 文献对 unlearning 一词的使用并不完全一致。2026 年一篇立场论文主张，只有明确 forget set 并以“不含该集合的重训模型”为参照时，才使用 machine unlearning；拒答、知识编辑和定向抑制应另行命名。这是作者提出的严格术语立场，不代表领域已有统一定义。读论文时应按具体目标和评测参照描述方法，不能只按标题判断它承诺了数据删除。[Position: The Term “Machine Unlearning” Is Overused in LLMs](https://proceedings.mlr.press/v306/yoon26g.html)

#### SISA 的系统设计思想

SISA training 把训练数据划分为 shard 和 slice，并保存每个阶段的模型状态。删除某个样本时，只重训受影响的 shard，从而降低重训范围。

它的价值在于把删除需求前置到数据与训练架构中。代价是：

1. 分片可能降低统计效率。
2. 聚合方式会影响最终模型。
3. 已经导出的 checkpoint、缓存和下游模型也需要处理。
4. 删除请求到达后，仍要证明所有相关 artifact 被更新。

因此 unlearning 不是一个只替换权重的函数，而是数据血缘、训练日志、checkpoint 和发布流程共同组成的生命周期。

#### 近似 LLM unlearning 的目标

大模型通常无法负担每次请求的完全重训，研究者会使用梯度反向、目标抑制、知识编辑、拒答训练或对比约束等近似方法。一个抽象目标可以写成：

~~~math
\mathcal{L}
=
\mathcal{L}_{\mathrm{forget}}(D_f)
+
\alpha\mathcal{L}_{\mathrm{retain}}(D_r)
+
\beta\mathcal{L}_{\mathrm{behavior}}
(\theta,\theta_0),
~~~

其中第一项抑制目标内容，第二项保持非目标能力，第三项限制模型偏离原模型过多。不同方法对三项的实现不同，不能把这个抽象式当成某篇论文的精确目标。

#### 拒答不等于遗忘

假设模型在直接问题“某本书的主人公是谁”上回答“我不能回答”，这可能有至少四种原因：

1. 目标内容真的不再可恢复。
2. 模型学会对关键词拒答，但改写问题仍能回答。
3. 模型保留了内容，只是安全策略阻止输出。
4. 解码或评估提示改变了行为，内部记忆没有变化。

因此 unlearning 评估要做攻击面扩展：

| 测试 | 目的 |
| --- | --- |
| 直接问法 | 检查目标内容的显式 recall |
| paraphrase | 检查是否只记住表面模板 |
| 间接问题 | 检查事实是否仍可组合恢复 |
| 补全和续写 | 检查长文本记忆 |
| 多语言和音译 | 检查表达变体 |
| 相关知识 | 检查是否过度删除 |
| 无关任务 | 检查通用能力损失 |

只有直接问法下降，不能称为完成了 unlearning。

把这条原则落到一个构造案例：某数据主体要求删除一段个人资料。更新前，10 个直接问法中模型答出 8 个；更新后只答出 1 个，直接 recall 从 $0.8$ 降至 $0.1$，下降 $0.7$。但在 10 个给出职业、城市后再询问身份的间接变体上，模型仍答出 5 个；20 条相邻主题保持题有 18 条正确。现在只能说目标问法的服从率降低，同时仍有 $0.5$ 的间接 residual recall、$0.9$ 的 retain utility；不能说资料已从训练影响中消失。还需沿数据血缘检查重复来源、旧 checkpoint、蒸馏模型、检索索引和缓存，并明确允许的攻击者预算。

#### Forget、retain、generalization 三个维度

设 $F$ 是应忘记集合，$R$ 是保留集合，$G$ 是目标知识的变体集合。可以分别记录：

~~~math
\mathrm{ForgetDrop}
=
M_{\mathrm{before}}(F)-M_{\mathrm{after}}(F),
~~~

~~~math
\mathrm{Retain}
=
M_{\mathrm{after}}(R),
\qquad
\mathrm{Residual}
=
M_{\mathrm{after}}(G).
~~~

其中 $M$ 可以是 recall、extraction success 或任务准确率，且每个集合都必须有明确的非空分母。ForgetDrop 高、Residual 低、Retain 稳定，才构成更有意义的结果。任何一个指标单独变化都可能产生误判。

### 12.4.3 Privacy 与 Memorization：模型记住了什么

#### Memorization 不是简单的 overfitting

Overfitting 关注训练误差与泛化误差的差异；memorization 关注模型是否能以异常高的概率复现某个训练样本或其独特片段。一个模型可以在总体 benchmark 上泛化良好，同时记住少量罕见、重复或敏感字符串。

记忆还可以分成：

1. 可由常识或公开频率解释的普通记忆。
2. 对罕见序列的概率提升。
3. 在给定前缀后逐字续写的 extractable memorization。
4. 与个人、凭证、内部日志有关的高风险记忆。

研究报告应说明测量的是哪一种。

#### Canary 与 exposure

一种经典实验是在训练数据中插入随机生成、低自然频率的 canary，例如模板化但不包含真实个人信息的字符串。设候选集合有 $N$ 个字符串，模型在给定前缀后把真实 canary 排名为 $r$，则 exposure 可以写成：

~~~math
\mathrm{Exposure}
=
\log_2 N-\log_2 r.
~~~

其中要求 $N>0$ 且 $1\leq r\leq N$。排名越靠前，exposure 越高。这个指标是研究记忆倾向的工具，不应直接当成真实隐私泄漏概率。canary 的分布、候选生成、提示形式和搜索预算都会影响结果。

若教学候选集有 $N=1{,}024$ 个字符串，真实 canary 排名 $r=4$，则 $\log_2 1{,}024=10$、$\log_2 4=2$，exposure 为 $10-2=8$ bit。若同一个 canary 只排到第 256 位，则为 $10-8=2$ bit。两值是相对候选集合与排序协议的信号，不是“泄露概率 8 倍”或对真实 PII 的风险评级。

#### 哪些因素会增加记忆

常见影响因素包括：

1. 样本重复次数。
2. 序列的罕见程度和唯一性。
3. 模型容量与训练步数。
4. 给出的前缀长度和位置。
5. 去重、过滤和数据混合策略。
6. 训练数据是否在多个来源重复出现。

“模型越大一定记得越多”过于粗糙。容量、优化、数据频率和提示条件会共同决定可提取性，必须用受控实验分离变量。

#### Membership inference 和 extraction

Membership inference 问“某条样本是否出现在训练集”，extraction 问“能否从模型输出中恢复训练内容”。两者相关但不等价。模型可能表现出样本成员信号，却无法逐字输出；也可能在某个前缀下提取片段，却无法可靠判断完整样本是否属于训练集。

评估时要记录攻击者能力：

1. 是否知道候选样本。
2. 是否能够查询很多次。
3. 是否能调节温度和提示。
4. 是否知道模型类型和训练领域。
5. 是否能观察概率或只看到文本。

没有攻击者模型的隐私结论是不完整的。

#### Differential privacy 的不同承诺

差分隐私试图限制单个训练样本对模型输出分布的影响。一个常见定义是：对相邻数据集 $D$ 和 $D'$，机制 $\mathcal{M}$ 满足 $(\varepsilon,\delta)$-差分隐私，如果对任意输出集合 $S$：

~~~math
\Pr[\mathcal{M}(D)\in S]
\leq
e^\varepsilon
\Pr[\mathcal{M}(D')\in S]
+\delta.
~~~

其中 $\varepsilon$ 控制隐私损失的乘性范围，$\delta$ 是允许的例外概率。这个定义提供的是对相邻数据集输出分布的形式化保证，不等于“模型不会泄漏任何信息”，也不自动解决训练数据的合法来源、成员属性和部署端日志问题。

实践中的 DP-SGD 还会引入梯度裁剪和噪声，通常带来效用、训练稳定性和隐私预算之间的权衡。报告必须给出采样率、步数、裁剪范数、噪声倍率和隐私会计方法，不能只写“使用了差分隐私”。

#### 隐私治理的完整链路

输出过滤是最后一道防线，不应承担全部责任。较完整的链路包括：

1. 训练前识别 PII、凭证、内部标识和高风险文本。
2. 去重并记录数据来源、许可和删除状态。
3. 训练中控制重复、保存隐私预算和数据版本。
4. 训练后进行 canary、extraction、membership 和敏感类别测试。
5. 部署时限制查询、权限和日志保留，并对输出做敏感信息检测。
6. 收到删除请求后沿数据血缘处理 checkpoint、索引、缓存和衍生 artifact。

每一步都可能失败。删除原始文件但保留旧 checkpoint，不是完成删除；拦截一个邮箱地址但让模型通过改写输出密钥，也不是完成保护。

PII 提取实验还要控制提示线索。ACL 2026 的 cue-controlled study 在公开 mC4 训练模型、指定提示和 32 种语言的设置中发现，提取命中集中在高 lexical-cue 条件；低 cue 子集的多语言命中接近零，八种成员推断方法的 AUROC 多在 0.50—0.60。它说明“模型吐出 PII”与“模型在缺少复现线索时记住了它”需要分开测；结果只适用于该论文模型和攻击定义，不能用来断言其他训练流程没有隐私风险。[Cue-Resistant Memorization](https://aclanthology.org/2026.acl-long.1560/)

### 12.4.4 Watermarking 与内容来源

#### 水印、检测和 provenance 不是一回事

生成内容治理中有三个经常混淆的对象：

1. Watermark：在生成分布中嵌入可检测统计信号。
2. Detector：根据内容判断是否存在某个模型的信号。
3. Provenance：记录内容由什么工具、什么时间、经过哪些编辑产生。

水印可以帮助检测，provenance 可以帮助追溯，二者都不能单独证明内容“真实”或“没有被编辑”。来源记录存在并不代表内容陈述正确；检测不到水印也不代表内容不是 AI 生成的。

#### 文本水印的统计直觉

一种文本水印方法在每一步根据上下文把候选 token 划分为绿色集合 $G_t$ 和其他集合，并略微提高绿色 token 的采样概率。生成长度为 $T$ 的文本后，绿色 token 数量为 $K$。若无水印时绿色 token 近似服从概率为 $\gamma$ 的二项分布，可以使用近似 z 分数：

~~~math
z
=
\frac{K-\gamma T}
{\sqrt{T\gamma(1-\gamma)}}.
~~~

这里要求 $T>0$ 且 $0<\gamma<1$；否则分母为零或没有可解释的二项基线。

当 $z$ 很高时，检测器会认为观察到的绿色 token 数量不太像自然生成。实际算法可能使用上下文相关哈希、不同采样分布和更复杂的统计检验，这个式子只用于解释检测信号的来源。

用 40 个 token、期望绿色比例 $\gamma=0.5$、实际绿色 token 数 $K=29$ 做构造例：期望值为 $40\times0.5=20$，标准差近似为 $\sqrt{40\times0.5\times0.5}=\sqrt{10}\approx3.162$，所以 $z=(29-20)/3.162\approx2.85$。若某个演示检测器把阈值设成 3，此文本没有越过该阈值；但阈值需按长度、误报率和检测功效校准，不能把 3 当成跨算法标准。

#### 检测器的四类指标

对有水印和无水印文本组成的数据集，应分别报告：

1. true positive rate：水印文本被识别的比例。
2. false positive rate：自然文本被误判的比例。
3. robustness：改写、翻译、截断、拼接后仍能检测的比例。
4. utility impact：水印对质量、事实性、多样性和安全行为的影响。

一个只在原始长文本上有高召回的 detector，遇到短文本或人工改写可能快速失效。若提高水印强度导致文本质量下降，也需要在相同任务预算下报告这个代价。

一篇 2026 预印本在其有效的意义保持改写样本上比较 KGW、Unigram 和一个 SynthID-Text 配置，报告初次检出的样本在改写后大多无法再检出，特定 SynthID 配置的下降为 98.3%。这说明改写鲁棒性应在实际变换上测量；数字受水印实现、阈值、提示、改写定义和初检筛选条件约束，不能概括为所有水印都会失效。[AI Watermark Evidence Fails Forensic Readiness](https://arxiv.org/abs/2607.16010)

#### 图像内容和来源记录

图像、视频和音频中的来源治理可能使用频域或生成过程信号，也可能使用签名的 provenance manifest。它们的攻击面包括裁剪、重编码、截图、混合编辑、模型转换和元数据丢失。

工程上应分别保存：

1. 原始生成请求和模型版本。
2. 输出文件哈希和生成时间。
3. 编辑软件、操作者或自动流水线。
4. 水印检测结果及其置信度。
5. provenance 是否完整、是否被第三方验证。

C2PA Technical Specification 2.5 将验证对象放在签名声明、manifest、内容绑定和信任链状态上。验证通过支持的是“某个签名身份对来源或编辑作出可校验声明”，不是内容为真、没有 AI 参与或声明者具有法律作者身份。[C2PA 2.5](https://spec.c2pa.org/specifications/specifications/2.5/specs/C2PA_Specification.html)

不要把“文件带有某个元数据字段”直接写成法律意义上的作者证明。技术证据、平台政策和法律判断属于不同层次。

编辑和遗忘关注模型或数据状态是否改变，隐私评估关注信息能否被恢复，水印与 provenance 关注输出信号或来源声明能否验证；它们的对象和成功条件并不相同。下一节把这些不同结果放回共同的研究流程中，要求 claim、对照、分母和复现范围彼此匹配，再决定哪些结论能支持实际工程动作。

## 12.5 交叉验证与复现：从研究主张到工程决策
### 12.5.1 Safety 与 Interpretability 如何互相支撑

#### 可解释性可以帮助安全，但不是安全证明

Interpretability 可能在以下方面帮助安全：

1. 定位与危险行为相关的表示或路径。
2. 分析拒答、服从、迎合和过度自信的机制。
3. 对异常激活进行部署监控。
4. 为 model editing、steering 和 unlearning 提供候选对象。
5. 帮助解释红队失败的内部差异。

但是“发现一个危险 feature”并不意味着其他危险路径不存在；“抑制一个 feature 后样本变安全”也不意味着模型在分布外条件下安全。解释工具应当进入测试与诊断链，而不是替代行为评估和权限控制。

研究者还应区分“控制模型行为”和“解释模型机制”。ACL 2026 的 AI Steerability 360 将控制接口分成 input（prompt）、structural（权重/架构）、state（激活/attention）和 output（解码）四类，并提供组合 steering pipeline 与任务比较入口。这个工具有助于统一实验组织；它的存在本身不证明某个控制有效，更不证明研究者已经解释了产生行为的完整电路。[AI Steerability 360](https://aclanthology.org/2026.acl-demo.43/)

#### 安全问题也给解释研究提供目标

安全研究提供了更具体的行为对象：某类隐私泄漏、某种工具越权、某个拒答边界或一组迎合样本。相比泛泛地问“模型如何思考”，这些对象更容易形成反事实和干预实验。

但行为目标也会产生选择偏差。研究者如果只分析成功发现的风险样本，可能找不到模型在其他任务上的抵消路径。应保留安全样本、正常样本、近邻反例和无关任务，比较干预的收益与副作用。

#### 从研究发现到生产控制

一个更现实的安全系统通常包含多层：

1. 数据治理减少明显敏感和污染来源。
2. 训练和后训练塑造有用、诚实和谨慎的行为。
3. 评估和红队发现已知及未知失败。
4. 解释工具辅助定位和诊断。
5. 运行时权限、速率、沙箱和人工确认限制后果。
6. 监控、审计和回滚处理线上变化。

这些层之间没有任意替代关系。模型即使被某个机制分析工具“理解”了，也仍然需要执行器权限和事故响应；一个强过滤器也不能替代数据删除和可靠的行为评估。

### 12.5.2 研究论文中的评估与证据

#### 先写最小 claim

安全和可解释性论文最容易把 claim 写大。例如“方法提高安全性”可能包含有害回答、工具动作、隐私泄漏和过度拒答等多个不同指标；“发现了一个电路”可能包含相关激活、可预测特征和完整因果机制等不同强度。

建议把主张拆成：

1. 对象：哪个模型、层、任务或系统。
2. 处理：哪种训练、编辑、干预或评估方法。
3. 结果：哪个指标变化，变化方向和大小。
4. 条件：数据、提示、采样、工具和人工协议。
5. 边界：没有测什么，可能在哪些条件下失效。

#### 安全实验的分母

安全数据常有极不平衡的类别。危险事件少，不代表风险小；如果只报告总体准确率，模型把所有请求都拒绝也可能得到看似不错的结果。

至少要同时报告：

~~~math
\mathrm{UnsafeCompliance}
=
\frac{\mathrm{unsafe\ requests\ answered\ unsafely}}
{\mathrm{unsafe\ requests}},
~~~

~~~math
\mathrm{FalseRefusal}
=
\frac{\mathrm{benign\ requests\ refused\ incorrectly}}
{\mathrm{benign\ requests}}.
~~~

两个分母都必须大于零；某一风险类别没有样本时，应标记为未评估。不能把没有危险请求或没有良性请求的切片记成风险为零或误拒为零。

例如，40 个危险请求中有 2 个得到危险服从，$\mathrm{UnsafeCompliance}=2/40=5\%$；60 个良性请求中有 3 个被错误拒答，$\mathrm{FalseRefusal}=3/60=5\%$。若把 100 个请求合并，只报 95% 的“正确处理率”，读者看不到两类风险分别发生在哪里，也看不到分母差异。

还要按风险类别、语言、用户权限、工具状态和严重度切片。对高严重度事件，即使样本很少，也应该报告零事件的不确定性，而不是写成风险为零。

International AI Safety Report 2026 将预部署评估与真实使用之间的落差列为评估难题：固定 benchmark 分数不能可靠预测所有现实行为，风险类别本身也可能重叠。报告提供的是对公开证据的综合，不是一项能够换算生产风险率的新测量；因此书面 claim 仍需绑定目标系统、用户互动和真实后果。[International AI Safety Report 2026](https://internationalaisafetyreport.org/publication/international-ai-safety-report-2026)

#### Interpretability 实验的对照

至少需要以下对照：

1. 随机 feature、随机 head 或随机层。
2. 同样数量的非目标干预。
3. 语义相似但机制不同的输入。
4. 目标任务与无关任务。
5. 多个随机种子和模型 checkpoint。
6. 不同解释方法之间的稳定性。

如果一个干预让所有输出都变得更保守，目标安全指标下降并不能说明找到了安全机制；如果 patch 让所有 logits 变大，目标答案恢复也可能只是置信度效应。

#### 把论文结果分成四类

阅读资料时，可以把证据标为：

1. 论文明确展示的实验事实。
2. 官方代码或模型卡明确声明的实现事实。
3. 在给定数据和协议下的项目复现结果。
4. 根据机制和样例做出的合理推测。

第四类不能用确定语气写成架构事实。尤其是未公开模型，官方样例能支持行为观察，但不能支持训练数据、参数规模、内部电路或安全原因的推断。

### 12.5.3 研究复现：把主张转成可回归实验

#### 从小问题开始

不要一开始就声称“复现某模型的完整安全性”或“解释最大模型的推理过程”。可以先选择一个窄问题：

> 在固定小型 Transformer 和重复序列任务上，抑制候选 induction head 是否降低复制准确率，而不改变普通下一个 token 任务？

或：

> 在固定评估集和相同采样预算下，某种 unlearning 方法是否同时降低目标内容的直接 recall 与 paraphrase recall，并保持无关知识准确率？

窄问题能让输入、干预、指标和失败边界对齐。

#### SAE 或 mechanistic interpretability 的最小路线

一个可运行的最小项目可以包括：

1. 训练或加载一个规模可控的小型 Transformer。
2. 固定数据集并保存 tokenizer 和 checkpoint。
3. 收集指定层、指定 token 位置的激活。
4. 训练多个稀疏度和 dictionary 宽度的 SAE。
5. 用触发样本、反事实和随机对照评估 feature。
6. 对候选 feature 做抑制或放大干预。
7. 在目标任务和无关任务上比较变化。
8. 保存激活样本、feature 参数、干预配置和失败案例。

如果只有 feature 可视化，没有干预和反事实，结论应停留在“候选相关表示”。

#### 红队或 unlearning 的最小路线

红队项目要保存 threat model、任务清单、搜索预算、完整轨迹、严重度和人工复核。unlearning 项目要保存删除集合、保留集合、变体集合、原模型、更新模型、攻击提示和能力回归结果。

安全研究尤其需要保留失败结果。删除掉失败样本，只展示成功的安全案例，会把研究变成演示，而不是证据。

#### 复现层级

可以把复现分为四层：

| 层级 | 含义 |
| --- | --- |
| 运行复现 | 代码、依赖和输入能够运行 |
| 数值复现 | 主要指标在预先定义的容差内 |
| 机制复现 | 关键干预、feature 或行为现象重现 |
| 主张复现 | 原论文的核心比较在公平预算下仍成立 |

没有原始数据、权重或评估脚本时，应称为替代数据复现、机制复现或分析复现，不能模糊地写“完全复现”。

#### 研究审计 demo：把分母、对照和动作连起来

下面的纯 Python 示例使用合成记录，演示如何把红队、解释干预、unlearning 和文本水印分别计算。它不加载真实模型，也不产生攻击提示；作用是展示分母、对照和动作如何被写进实验记录。

**代码状态：静态审阅，未运行（2026-11-04；Python 3.7+；仅标准库）。** 每组列表都是固定的合成 0/1 标签；空分母返回 `None`，后续动作据此把“缺少证据”送入复查，而不是当成零风险。

~~~python
from math import sqrt


def mean(values):
    return sum(values) / len(values) if values else None


def rate(values):
    return mean([1.0 if value else 0.0 for value in values])


def z_score(green_count, total, expected_green_rate):
    if total == 0 or not 0 < expected_green_rate < 1:
        return None
    variance = total * expected_green_rate * (1 - expected_green_rate)
    return (green_count - total * expected_green_rate) / sqrt(variance)


def safe_ratio(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def rounded(value):
    return None if value is None else round(value, 4)


red_team = [
    {"severity": 3, "unsafe_compliance": 1, "unauthorized_tool": 1},
    {"severity": 2, "unsafe_compliance": 0, "unauthorized_tool": 0},
    {"severity": 1, "unsafe_compliance": 1, "unauthorized_tool": 0},
    {"severity": 3, "unsafe_compliance": 0, "unauthorized_tool": 1},
]

red_team_unsafe = rate([row["unsafe_compliance"] for row in red_team])
tool_breach = rate([row["unauthorized_tool"] for row in red_team])
weighted_failure = safe_ratio(
    sum(row["severity"] * row["unsafe_compliance"] for row in red_team),
    sum(row["severity"] for row in red_team),
)

clean_behavior = [1, 1, 0, 1, 1]
target_behavior = [1, 0, 0, 1, 0]
random_control = [0, 0, 1, 0, 0]
causal_drop = rate(clean_behavior) - rate(target_behavior)
control_drop = rate(clean_behavior) - rate(random_control)

forget_before = [1, 1, 1, 0]
forget_after = [0, 0, 1, 0]
retain_after = [1, 1, 0, 1]
forget_drop = rate(forget_before) - rate(forget_after)
residual_recall = rate(forget_after)
retain_utility = rate(retain_after)

watermark_z = z_score(green_count=29, total=40, expected_green_rate=0.5)

signals = {
    "unsafe_compliance": rounded(red_team_unsafe),
    "unauthorized_tool": rounded(tool_breach),
    "severity_weighted_failure": rounded(weighted_failure),
    "target_intervention_drop": rounded(causal_drop),
    "random_control_drop": rounded(control_drop),
    "forget_drop": rounded(forget_drop),
    "residual_recall": rounded(residual_recall),
    "retain_utility": rounded(retain_utility),
    "watermark_z": rounded(watermark_z),
}

actions = []
if signals["unauthorized_tool"] is None or signals["unauthorized_tool"] > 0:
    actions.append("separate_model_text_from_server_authorization")
if (
    signals["target_intervention_drop"] is None
    or signals["random_control_drop"] is None
    or signals["target_intervention_drop"] <= signals["random_control_drop"]
):
    actions.append("repeat_causal_intervention_with_matched_controls")
if signals["residual_recall"] is None or signals["residual_recall"] > 0:
    actions.append("expand_unlearning_paraphrase_and_indirect_tests")
if signals["retain_utility"] is None or signals["retain_utility"] < 0.8:
    actions.append("repair_retain_set_before_more_forgetting")
if signals["watermark_z"] is None or signals["watermark_z"] < 3.0:
    actions.append("report_watermark_as_low_confidence")

decision = "continue_after_scope_repairs" if actions else "continue_to_holdout"

assert mean([]) is None
assert rate([]) is None
assert safe_ratio(0, 0) is None
assert z_score(0, 0, 0.5) is None
assert z_score(1, 10, 0.0) is None

for name, value in signals.items():
    print(f"{name}={value}")
print(f"actions={actions}")
print(f"decision={decision}")
~~~

按输入手算，红队危险服从和越权工具动作各为 $2/4=0.5$；严重度加权失败是 $(3+1)/(3+2+1+3)=4/9\approx0.4444$。解释干预的基线行为率为 $4/5=0.8$，目标方向干预后为 $2/5=0.4$，下降 $0.4$；随机方向后为 $1/5=0.2$，下降反而更大，为 $0.6$，所以代码要求用匹配对照重做，而不能把目标变化直接当作机制证据。unlearning 的直接 recall 从 $3/4$ 降到 $1/4$，下降 $0.5$，但 residual recall 仍是 $1/4=0.25$，retain utility 只有 $3/4=0.75$。水印 $z=(29-20)/\sqrt{10}\approx2.846$，低于演示阈值 3；阈值只是教学输入，不是跨实现标准。

手算对应的预期输出如下；这些值从列表和公式推得，本轮没有运行代码：

~~~text
unsafe_compliance=0.5
unauthorized_tool=0.5
severity_weighted_failure=0.4444
target_intervention_drop=0.4
random_control_drop=0.6
forget_drop=0.5
residual_recall=0.25
retain_utility=0.75
watermark_z=2.846
actions=['separate_model_text_from_server_authorization', 'repeat_causal_intervention_with_matched_controls', 'expand_unlearning_paraphrase_and_indirect_tests', 'repair_retain_set_before_more_forgetting', 'report_watermark_as_low_confidence']
decision=continue_after_scope_repairs
~~~

这个示例的重点不是阈值本身。target_intervention_drop 必须和随机对照比较，forget_drop 必须与 residual_recall 和 retain_utility 一起看，水印 z 分数必须结合文本长度和误报率。真实项目还要加入置信区间、人工复核、模型版本、随机种子、攻击者预算和数据血缘。

## 12.6 资料与证据边界

Concrete Problems in AI Safety 支持副作用、reward hacking、scalable oversight、安全探索和分布偏移的问题分解：[论文原文](https://arxiv.org/abs/1606.06565)。它是问题框架，不是现代大模型所有风险的完整评估标准。

红队方向可参考 Red Teaming Language Models with Language Models：[论文原文](https://arxiv.org/abs/2202.03286)。它支持使用语言模型扩展红队样本的研究路线；具体模型、风险类别和生产后果需要在目标系统上独立测试。

模型生成评估可参考 Discovering Language Model Behaviors with Model-Written Evaluations：[论文原文](https://arxiv.org/abs/2212.09251)。它支持用模型发现行为和研究 inverse scaling 的方法，但不能证明自动生成题目和标签没有偏差。

induction heads 和 in-context learning 的机制讨论可参考 In-context Learning and Induction Heads：[论文原文](https://arxiv.org/abs/2209.11895)。事实关联的定位与编辑可参考 Locating and Editing Factual Associations in GPT：[论文原文](https://arxiv.org/abs/2202.05262)。这些论文支持特定任务和干预实验中的机制主张，不等于已经解释了整个 Transformer。

superposition 可参考 Toy Models of Superposition：[论文原文](https://arxiv.org/abs/2209.10652)。SAE 可参考 Sparse Autoencoders Find Highly Interpretable Features in Language Models：[论文原文](https://arxiv.org/abs/2309.08600)。更大规模的 feature 提取可参考 Scaling Monosemanticity：[论文原文](https://arxiv.org/abs/2403.19647)；该类研究报告支持规模化特征分析的观察，不自动提供完整因果电路证明。Circuit Tracing 的公开研究页面可参考 [Anthropic Transformer Circuits](https://transformer-circuits.pub/2025/attribution-graphs/index.html)。

批量模型编辑可参考 Mass-Editing Memory in a Transformer：[论文原文](https://arxiv.org/abs/2210.07229)。论文支持批量参数编辑的实验路线；可靠性、泛化、局部性和长期持久性仍需按模型和数据重新验证。

machine unlearning 的训练架构可参考 Machine Unlearning：[论文原文](https://arxiv.org/abs/1912.03817)。LLM 近似遗忘可参考 Who's Harry Potter? Approximate Unlearning in LLMs：[论文原文](https://arxiv.org/abs/2310.02238)。这些资料支持近似遗忘和降低重训范围的研究方向，不支持把拒答直接等同于完成了数据删除。

记忆与训练数据泄漏可参考 Quantifying Memorization Across Neural Language Models：[论文原文](https://arxiv.org/abs/2202.07646) 和 Extracting Training Data from Large Language Models：[论文原文](https://arxiv.org/abs/2012.07805)。它们支持记忆、重复和提取风险的实验研究；真实隐私风险还依赖数据主体、攻击者能力、访问权限和法律适用范围。

文本水印可参考 A Watermark for Large Language Models：[论文原文](https://arxiv.org/abs/2301.10226)。Google DeepMind 的 SynthID 公开页面可作为生成内容标识方向的官方资料：[官方页面](https://deepmind.google/technologies/synthid/)。水印检测、来源记录、内容真实性和法律归属不能互相替代。

模型卡可参考 Model Cards for Model Reporting：[论文原文](https://arxiv.org/abs/1810.03993)。治理与风险管理可参考 [NIST AI RMF Playbook](https://airc.nist.gov/airmf-resources/playbook/)、[OpenAI Preparedness Framework 更新说明](https://openai.com/index/updating-our-preparedness-framework/) 和 [Anthropic Responsible Scaling Policy](https://www.anthropic.com/news/anthropics-responsible-scaling-policy)。这些框架支持风险识别、评估、文档和部署治理的组织方法，不能替代针对具体模型和业务系统的实测。

较新的评测入口包括 [REDAgentBench](https://arxiv.org/abs/2608.10669)（固定 rollout、比较执行状态与轨迹判定）和 [ANGLER / REEF](https://arxiv.org/abs/2610.41272)（以 verifier 反馈进行自适应 Web Agent 红队）；两者的环境、攻击方和判定协议不同，不能合并成一条 ASR 排名。[SAEBench](https://arxiv.org/abs/2503.09532)、[SANITY Checks](https://arxiv.org/abs/2602.14111v1) 和 [Rethinking SAE Evaluation / DAME](https://arxiv.org/abs/2610.30474v2) 展示 SAE 评估从单一重构指标扩展到多任务和支配特征检查的争论，后两项的结论分别绑定合成基准/随机基线与 LLM-judge proxy。[AI Steerability 360](https://aclanthology.org/2026.acl-demo.43/) 是 steering 控制接口和比较工具，不是安全或机制保证。[Cue-Resistant Memorization](https://aclanthology.org/2026.acl-long.1560/)、[AI Watermark Evidence Fails Forensic Readiness](https://arxiv.org/abs/2607.16010) 与 [C2PA 2.5](https://spec.c2pa.org/specifications/specifications/2.5/specs/C2PA_Specification.html) 分别用于提示线索控制、文本水印改写测试和签名来源声明验证；各自只支持对应的研究对象和协议。

## 12.7 结语：让安全和解释回到证据

Safety 论文线的核心变化，是把“可能有风险”改写成有对象、有分母、有后果和有处置路径的研究问题。Red teaming 负责发现失败，model-written evaluations 扩展行为搜索，隐私和 memorization 研究模型与数据的关系，unlearning 和 model editing 尝试修复知识状态，水印和 provenance 帮助内容治理。

Interpretability 论文线则从输入归因逐步走向表示、特征、电路和因果干预。Induction heads 说明窄问题可以形成可实验的机制对象；causal tracing 说明恢复实验比激活相关性更接近因果证据；SAE 提供了分解激活空间的工具，但 feature 命名、重构和可干预性仍需要独立审计。

这些方法都没有把系统变成透明、无风险的机器。安全评估会漏掉未知分布，红队会受到搜索预算限制，unlearning 可能只是拒答，editing 可能带来局部性损失，水印可能被改写，解释可能只覆盖一条局部路径。真正可靠的工程结论，应当同时说明测到了什么、没有测到什么、失败会造成什么后果，以及下一轮如何用新的对照和真实轨迹检验它。

### 小练习

1. 为一个能够读取邮件并发送草稿的 Agent 写出资产、信任边界、工具权限和三个红队分母。
2. 设计一个有图与无图的反事实任务，区分视觉证据使用和语言先验。
3. 在一个小型 Transformer 上构造 A B ... A ? 序列，比较候选 head 抑制与随机 head 抑制。
4. 给一个 SAE feature 写出触发样本、反例、干预和无关任务四种记录。
5. 设计 unlearning 的 forget、paraphrase、residual 和 retain 四组测试，并说明哪一种结果仍不能称为“忘记”。
6. 对长度为 80、绿色比例为 0.5、绿色 token 数为 52 的文本计算水印 z 分数，并讨论短文本误报的影响。
