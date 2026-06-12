# L. Reasoning 与评估

条目：Reasoning、Reasoning Model、Reasoning Candidate Set、Chain-of-Thought、Few-shot CoT、Zero-shot CoT、Scratchpad、Hidden CoT、Visible Explanation、CoT Faithfulness、CoT Regression、CoT Routing、CoT Audit、Self-Consistency、Sampling Temperature、Top-p Sampling、Answer Normalization、Majority Vote、Weighted Vote、Candidate Diversity、Majority Failure、Self-Consistency Cost、Self-Consistency Accuracy、Verifier、Programmatic Verifier、Hybrid Verifier、Verifier Reranking、Pairwise Accuracy、Hard Negative、Verifier Calibration、Reward Model Bias、Process Supervision、Step Label、First-Error Detection、Process Reward Model、Outcome Reward Model、Process Step Accuracy、Auto Label Coverage、Human Label Cost、Process Search Pruning、Process Supervision Audit、Search Reasoning、Search State、Search Action、Beam Search、Best-First Search、UCT、Prune False Negative、Search Budget、Search Audit、Test-Time Compute、Test-Time Compute Scaling、Compute Budget Vector、Adaptive Compute、Budget Router、Cost per Correct、Marginal Accuracy per Cost、P95 Latency、Wasted High Compute、TTC Audit、Math Reasoning Training、Math Training Sample、Answer Supervision、Synthetic Math Data、Math Curriculum、Math Contamination Audit、Template Diversity、Math Training Gate、Code Reasoning、Execution Feedback、Unit Test Verifier、Public Test、Hidden Test、Public-Hidden Gap、Self-Debug、Repair Success Rate、Sandbox Violation Rate、Code Execution Audit、Code Reasoning Gate、Reasoning Evaluation、Evaluation Sample、Variant Evaluation、Robustness Drop、Paired Lift、Bootstrap Confidence Interval、Reasoning Eval Gate、Test-Time Compute Cost、Pass@k、Reasoning Audit、Reasoning Safety、Pseudo Reasoning、Overconfident Error、Hidden CoT Exposure、Tool Misuse、Human Review Coverage、Severity-Weighted Risk、Reasoning Safety Gate、Reasoning Interview Readiness、Reasoning Interview Rubric、Reasoning Formula Coverage、Reasoning Demo Coverage、Weak Reasoning Question、Reasoning Revision Plan、Tree-of-Thought、MCTS、Benchmark、Human Evaluation、Elo Rating、Hallucination、Factuality、Robustness、Evaluation Metric Incident、Aggregate Score Trap、Slice Regression、Clean Eval Lift、Judge-Human Agreement、Judge Length Bias、Evaluation Gate。

## Reasoning

一句话定义：reasoning 是模型通过多步中间过程解决问题的能力，常见于数学、代码、逻辑推理、规划、复杂问答和工具使用任务。

为什么重要：很多任务不能靠模式匹配直接回答，需要拆解条件、维护中间状态、比较多个选项、验证假设并逐步推出结论。

典型表现：能把复杂问题拆成子问题，能根据约束排除错误答案，能在计算和逻辑链条中保持一致，能发现前一步推理的错误并修正。

核心难点：模型可能生成看似合理但实际错误的推理步骤；长推理中错误会累积；推理文本正确不代表最终答案正确，最终答案正确也不代表推理过程真实可靠。

面试表达：大模型 reasoning 不是传统符号推理的简单替代，而是概率生成、训练数据模式、搜索策略、验证器和测试时计算共同作用的结果。

## Reasoning Model

一句话定义：Reasoning Model 是更强调多步推理、数学、代码、规划、验证和复杂任务分解的大模型或系统。

和普通 chat model 的区别：chat model 更强调指令遵循、对话自然和通用问答；reasoning model 更强调可验证任务上的正确性、推理预算、候选生成、验证器和错误修正。

重要边界：reasoning model 不等于“输出很长解释”的模型。长 CoT 可能只是看似合理的文本，真正可靠性需要最终答案、过程步骤、工具执行、verifier 和污染风险一起评估。

面试表达：reasoning model 是训练数据、推理策略、test-time compute、verifier 和评估体系共同作用的结果，不应只看回答长度。

## Reasoning Candidate Set

一句话定义：Reasoning Candidate Set 是同一道题在不同采样、搜索或工具反馈下产生的一组候选推理链和最终答案。

候选集合常用于 self-consistency、best-of-n、verifier reranking、代码执行筛选和 tree search。

面试表达：单次 greedy 答案可能错，但候选集合中可能已经包含正确解。reasoning 系统要同时看候选质量、候选多样性、选择器可靠性和成本。

## Chain-of-Thought

一句话定义：Chain-of-Thought，简称 CoT，是让模型生成中间推理步骤，再给出最终答案的方法。

为什么提出：直接问答案时，模型容易跳步。让模型先写推理过程，相当于给复杂任务分配更多计算和中间状态，有助于数学、逻辑和多跳问答。

解决什么问题：CoT 主要解决复杂问题中“一步到位”难以得到正确答案的问题。它让模型显式展开条件、公式、子问题和推导过程。

优点：提升复杂推理准确率，过程更可检查，也便于发现模型错在哪里。

缺点：推理过程可能是事后编造；长 CoT 会增加 token 成本；错误推理可能让答案更自信；在简单任务上不一定有收益。

常见用法：few-shot CoT、zero-shot CoT、让模型先分析再回答、要求输出简洁推理摘要。

面试表达：CoT 的本质是通过中间步骤增加测试时计算和状态表达，但 CoT 文本本身不一定等于模型真实内部推理。

## Few-shot CoT

一句话定义：Few-shot CoT 是在 prompt 中提供少量带推理步骤的示例，让模型模仿逐步解题格式。

优点：不需要重新训练，能通过示例控制问题分解、答案格式和推理粒度。

局限：占用上下文；示例选择会影响稳定性；示例中的错误步骤、跳步或格式混乱会被模型继承。

面试表达：few-shot CoT 是最早被系统验证有效的 CoT 用法之一，核心是用示例把模型从“直接答题模式”切到“逐步解题模式”。

## Zero-shot CoT

一句话定义：zero-shot CoT 是不提供示例，只通过提示语诱导模型进行逐步思考的方法。

典型提示：例如“让我们一步一步思考”或“请先分析再给出答案”。

优点：使用简单，不需要构造 few-shot 示例。

局限：对模型能力依赖强；提示语可能引入冗余推理；对格式、任务类型和模型对齐策略敏感。

面试表达：zero-shot CoT 是低成本推理增强方法，但稳定性通常弱于针对任务设计的 few-shot CoT 或 verifier 方案。

## Scratchpad

一句话定义：Scratchpad 是模型或系统用于中间计算、草稿、表格、代码计划或工具调用规划的工作区。

和 CoT 的区别：CoT 更常指自然语言推理链；scratchpad 更强调中间工作区，不一定适合完整展示给用户。

面试表达：scratchpad 的价值是给复杂任务留下中间状态，但产品输出通常应展示简洁、可验证的解释，而不是完整草稿。

## Hidden CoT

一句话定义：Hidden CoT 是模型或系统内部使用但不直接展示给用户的推理过程。

为什么需要区分：完整 CoT 可能冗长、包含错误探索、暴露系统策略或在安全任务中给出不必要细节。

面试表达：隐藏推理不是为了逃避评估，而是为了把内部计算和用户可见解释分开；可靠性仍要通过最终答案、步骤审计、工具验证和人工评估来证明。

## Visible Explanation

一句话定义：Visible Explanation 是面向用户展示的简洁理由、关键依据或可验证摘要。

和 Hidden CoT 的区别：visible explanation 服务用户理解和复核，不承诺完整复现模型内部推理轨迹。

面试表达：好的产品解释应当短、可核查、和结论相关；不能把冗长 CoT 当成可信度本身。

## CoT Faithfulness

一句话定义：CoT Faithfulness 衡量推理文本是否真实影响了模型结论，而不只是事后合理化。

常见检查：修改中间步骤是否改变最终答案、遮蔽关键步骤是否降低正确率、步骤错误是否能被 verifier 捕捉、同一答案是否伴随互相矛盾的理由。

面试表达：CoT 可读不等于 faithful。评估 CoT 要看最终答案、过程步骤、干预实验和工具验证，而不是只看语言是否流畅。

## CoT Regression

一句话定义：CoT Regression 是直答本来正确，但加入 CoT 后答案变错或输出质量下降的样本。

典型来源：简单事实题过度推理、错误类比、题目外条件被纳入推理、推理链中早期错误被后续步骤放大。

简化口径：

```math
R_{\mathrm{reg}}=
\frac{
\sum_i
\mathbb{1}[
\hat y_i^{\mathrm{direct}}=y_i^\star
\land
\hat y_i^{\mathrm{cot}}\ne y_i^\star
]
}{N}
```

面试表达：如果只汇报 CoT 平均准确率，可能掩盖简单题回归。上线前要单独看回归样本和任务切片。

## CoT Routing

一句话定义：CoT Routing 是按任务难度、风险、延迟预算和可验证性决定是否启用 CoT 的策略。

常见规则：简单事实题短答，复杂数学 / 代码题启用 CoT 或工具，安全敏感题展示简洁拒答或安全替代解释，高价值任务再增加 verifier。

面试表达：工程上不是“所有请求都长思考”，而是按任务价值和风险动态分配推理预算。

## CoT Audit

一句话定义：CoT Audit 是对 CoT 策略的答案准确率、步骤准确率、回归样本、unsupported step、token 成本和可见解释安全边界进行统一检查。

面试表达：CoT audit 能回答“CoT 是否真的值得开”，而不是只展示几条看起来漂亮的推理链。

## Self-Consistency

一句话定义：self-consistency 是对同一问题采样多条推理路径，再通过投票或聚合选择最终答案的方法。

为什么有效：复杂推理任务中，单条推理链可能偶然走错。多次采样能探索不同解题路径，如果正确答案在多条路径中反复出现，可信度更高。

基本流程：提高采样温度生成多个 CoT；提取每条路径的最终答案；按多数投票、置信度或验证器打分选择答案。

优点：显著提升数学和逻辑推理稳定性。

缺点：成本成倍增加；如果模型系统性误解题意，多数投票也会一致错误；答案抽取和等价判断并不总是简单。

面试表达：self-consistency 是用测试时采样换准确率，本质是搜索多个候选推理路径再做聚合。

## Sampling Temperature

一句话定义：Sampling Temperature 是控制生成分布平滑程度的采样参数。

在 reasoning 中的作用：温度太低时候选几乎重复，self-consistency 没有多样性；温度太高时推理路径容易跑偏，错误答案变多。

面试表达：self-consistency 不是把温度越调越高越好，而是要在候选多样性和候选质量之间找平衡。

## Top-p Sampling

一句话定义：Top-p Sampling / Nucleus Sampling 是每一步只从累计概率达到阈值 `p` 的最小高概率 token 集合中采样。

在 reasoning 中的作用：top-p 能截断长尾低质量 token，同时保留比 greedy 更丰富的候选路径。

面试表达：temperature 控制分布平滑，top-p 控制候选集合尾部，两者共同决定 self-consistency 的多样性和跑偏风险。

## Answer Normalization

一句话定义：Answer Normalization 是把不同表述但等价的最终答案合并到同一标准形式。

例子：`42`、`42.0`、`答案是 42`、`42 minutes` 在某些数学题中应归一为同一个答案。

面试表达：self-consistency 的投票质量高度依赖答案标准化。标准化差会把同一个正确答案拆散，或者把不同答案错误合并。

## Majority Vote

一句话定义：Majority Vote 是对多个标准化候选答案计数，选择票数最多的答案。

优点：简单、无需额外模型，适合答案空间明确的数学、选择题和逻辑题。

局限：如果模型系统性误解题意，多数候选会一起错；如果正确答案是少数，没有 verifier 时会被多数错误淹没。

面试表达：majority vote 提升的是聚合稳定性，不保证真值发现能力。

## Weighted Vote

一句话定义：Weighted Vote 是用 verifier、规则、log probability、单元测试或工具结果给候选答案加权，再按答案聚合分数。

适用场景：当正确答案是少数但 verifier 能识别高质量候选时，加权投票可能优于多数投票。

面试表达：weighted vote 的核心风险是评分器偏差。如果 verifier 错，聚合也会被带偏。

## Candidate Diversity

一句话定义：Candidate Diversity 是候选推理路径或标准化答案之间的差异程度。

为什么重要：候选全都一样时，多采样只是浪费成本；候选过于发散时，错误路径和格式混乱会增加。

面试表达：self-consistency 需要“有约束的多样性”，不是随机性越大越好。

## Majority Failure

一句话定义：Majority Failure 是多数投票选择了错误答案，但候选集合中可能已经存在正确答案的失败模式。

常见原因：系统性误解题意、干扰条件被错误使用、正确候选数量少、答案标准化错误或候选分布偏斜。

面试表达：majority failure 说明 pass@k、majority accuracy 和 verifier accuracy 必须分开汇报。

## Self-Consistency Cost

一句话定义：Self-Consistency Cost 是多路径采样带来的 token、延迟、调用次数和工具验证成本。

面试表达：self-consistency 的准确率提升必须和成本曲线一起看，生产系统通常只对高价值或高难度请求启用。

## Self-Consistency Accuracy

一句话定义：Self-Consistency Accuracy 是对每道题采样多条推理链后，经答案投票得到的最终准确率。

简化口径：

```math
A_{\mathrm{sc}}=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[\hat y_i^{\mathrm{sc}}=y_i^\star]
```

其中 `hat y_i^sc` 是多数投票答案。

面试表达：self-consistency accuracy 通常高于单次 greedy accuracy，但它仍可能被系统性误解、答案抽取错误和投票等价类问题限制。

## Verifier

一句话定义：verifier 是用于检查模型答案、推理步骤或工具结果是否正确的评估器或模型。

为什么提出：生成模型擅长提出候选答案，但不总擅长判断答案是否可靠。verifier 把“生成”和“验证”拆开，有助于从多个候选中选出更优答案。

常见形式：规则验证器、单元测试、数学答案检查器、代码执行器、检索证据核查器、奖励模型和另一个 LLM judge。

用途：降低幻觉，筛选候选答案，辅助 test-time compute，评估推理步骤，做 rejection sampling。

优点：可以显著提高复杂任务可靠性，特别是答案可自动验证的任务。

缺点：verifier 本身也可能错；不可自动验证的开放问答仍然需要人工或强 judge；验证成本可能很高。

面试表达：verifier 的价值在于把生成问题转化为候选搜索加质量判断。对于代码、数学和工具任务，验证往往比单纯提示工程更可靠。

## Programmatic Verifier

一句话定义：Programmatic Verifier 是用程序执行、规则、单元测试、计算器、schema 或形式化检查器验证候选输出的验证器。

优点：客观、可复现，不容易被语言风格、长度和自信语气欺骗。

局限：只适合可执行或可规则化任务；测试覆盖不足时仍会漏掉错误；执行环境需要沙箱、超时和权限隔离。

面试表达：代码、数学表达式、SQL、JSON 和工具调用这类任务，程序验证通常比纯语言 reward model 更硬，但它不能覆盖所有开放问答。

## Hybrid Verifier

一句话定义：Hybrid Verifier 是把程序验证、process score、learned reward model 和规则检查组合起来选择候选的验证策略。

为什么需要：单一 RM 可能被 hard negative 骗过，程序验证又不总可用。混合验证器可以在可验证任务上优先使用硬信号，在开放任务上退回 learned score 或人工复核。

面试表达：实际 reasoning 系统里，verifier 往往不是一个模型，而是一组按任务类型路由的验证信号。

## Verifier Reranking

一句话定义：Verifier Reranking 是先生成多个候选，再用 verifier 分数选择最高分候选的 reasoning 策略。

简化口径：

```math
\hat y_i^{\mathrm{ver}}=
y_{ij},
\qquad
j=\mathrm{index\ of\ max}_{k}(s_{ik})
```

其中 `s_ik` 是第 `i` 个问题第 `k` 个候选的 verifier 分数。

面试表达：verifier reranking 的关键不是生成更多文本，而是有一个足够可靠的选择器。代码测试、数学检查器和工具执行通常比纯文本自评更硬。

## Pairwise Accuracy

一句话定义：Pairwise Accuracy 衡量 verifier 是否给正确候选比错误候选更高的分数。

简化口径：

```math
A_{\mathrm{pair}}
=
\frac{1}{|\mathcal{P}|}
\sum_{(a,b)\in\mathcal{P}}
\mathbb{1}[s_a>s_b]
```

其中 `a` 是正确或更优候选，`b` 是错误或更差候选。

面试表达：pairwise accuracy 能检查 reward model 的相对排序能力，但仍要结合 top-1 selection accuracy 和真实下游提升。

## Hard Negative

一句话定义：Hard Negative 是看起来很像正确答案、格式很好、推理很长或很自信，但实际错误的负样本。

为什么重要：如果训练集只有简单负样本，verifier 会在真实候选分布上高估错误答案，导致 reward hacking。

面试表达：训练和评估 verifier 时必须加入 hard negatives，否则模型容易学到表面特征。

## Verifier Calibration

一句话定义：Verifier Calibration 衡量 verifier 分数是否能当作可靠概率或阈值使用。

常见指标：ECE、Brier score、分桶准确率、阈值下 precision / recall 和高风险切片校准。

面试表达：verifier 分数高不代表概率校准。上线门禁和人工复核阈值必须单独校准。

## Reward Model Bias

一句话定义：Reward Model Bias 是 reward model 偏好长度、格式、自信语气、模板化表达或训练集常见风格，而不是偏好真实正确性。

风险：generator 可以优化这些偏差来骗过 reward model，形成 reward hacking 或 verifier hacking。

面试表达：评估 reward model 时要看 hard negative、长度切片、格式切片、校准和下游任务提升，不能只看验证集平均准确率。

## Process Supervision

一句话定义：Process Supervision 是对推理中间步骤进行监督，而不是只监督最终答案。

为什么提出：outcome supervision 只告诉我们终点对不对，却无法定位哪一步错，也可能把“过程错但答案碰巧对”的样本当成好样本。

适用场景：数学推理、代码推导、工具计划、复杂规划和其他可拆成可检查步骤的任务。

核心边界：过程监督提升的是可检查性和训练信号密度，不等于模型内部推理完全可解释。工程上仍要同时看最终答案、步骤正确性、搜索效果、成本和 hard negative。

面试表达：process supervision 的价值是把长推理从“只看结果”拆成“每一步都可审计”，但代价是步骤切分、标注一致性和人工成本。

## Step Label

一句话定义：Step Label 是对推理链中每个步骤的正确性、相关性或错误类型的标注。

常见字段：步骤文本、正确 / 错误标签、是否相关、错误类型、是否可自动验证、人工复核成本。

简化记号：

```math
q_{ij}\in\{0,1\},
\qquad
r_{ij}\in\{0,1\}
```

其中 `q_ij` 表示步骤正确性，`r_ij` 表示步骤相关性。

面试表达：step label 不能只判断语言是否流畅，还要看这一步是否真的推动解题，否则会奖励正确但无用的冗余步骤。

## First-Error Detection

一句话定义：First-Error Detection 是定位推理链中第一处错误步骤的能力。

为什么重要：长推理里后续错误常常只是第一处错误的连锁反应。训练和 debug 时优先定位第一处错误，比平均步骤分数更有诊断价值。

简化记号：

```math
j_i^{\mathrm{err}}=
\min\{j:q_{ij}=0\}
```

面试表达：如果系统能准确找到第一处错误，就能更早停止坏路径、提示模型回退或触发人工复核。

## Process Reward Model

一句话定义：Process Reward Model，简称 PRM，是对推理过程中的每一步进行评分的奖励模型。

为什么提出：只看最终答案无法知道哪一步推理出错。PRM 直接评价中间步骤，有助于引导模型走正确路径。

适用场景：数学推理、程序推导、复杂规划、需要逐步验证的任务。

优点：能提供更细粒度监督，适合结合搜索或逐步生成。

缺点：过程标注成本高；不同正确推理路径可能很多；评分标准难统一；模型可能学会迎合过程格式而非真正推理。

面试表达：PRM 强调“过程正确”，适合需要长链条推理和搜索的任务，但工程上标注和泛化都更难。

## Outcome Reward Model

一句话定义：Outcome Reward Model，简称 ORM，是只对最终答案或最终输出进行评分的奖励模型。

为什么常用：最终结果比中间过程更容易标注。例如一道数学题答案是否正确、一段代码是否通过测试、一个回答是否满足用户偏好。

优点：标注成本相对低，和用户最终体验更直接相关。

缺点：无法定位中间错误；可能奖励错误但碰巧得到正确答案的推理；对长推理任务的引导较弱。

面试表达：ORM 看终点，PRM 看过程。ORM 更容易落地，PRM 更适合引导复杂推理搜索。

## Process Step Accuracy

一句话定义：Process Step Accuracy 是推理过程中间步骤被判定为正确的比例。

```math
A_{\mathrm{step}}=
\frac{\sum_i\sum_j z_{ij}}
{\sum_i M_i}
```

其中 `z_ij=1` 表示第 `i` 个样本的第 `j` 个推理步骤正确。

面试表达：步骤准确率能定位推理链中哪里出错，但高步骤准确率不必然保证最终答案正确，低步骤准确率也可能偶然得到正确答案。

## Auto Label Coverage

一句话定义：Auto Label Coverage 是过程监督数据中可以由规则、程序、测试、计算器或工具自动标注的步骤比例。

简化口径：

```math
C_{\mathrm{auto}}=
\frac{
\sum_i\sum_j a_{ij}
}{
\sum_i M_i
}
```

其中 `a_ij=1` 表示该步骤可自动标注。

面试表达：自动覆盖率越高，process supervision 越容易规模化；但自动规则覆盖不到的开放推理仍需要人工复核和质量抽检。

## Human Label Cost

一句话定义：Human Label Cost 是过程监督中需要人工切分、判断、复核和仲裁步骤标签的成本。

简化口径：

```math
C_{\mathrm{label}}=
\sum_i\sum_j(1-a_{ij})c_{ij}
```

面试表达：PRM 的效果不能脱离标注成本讨论。若每个步骤都要专家复核，真实上线要考虑只标 hard cases、半自动初标和抽样审计。

## Process Search Pruning

一句话定义：Process Search Pruning 是在 tree search、beam search 或 step-by-step search 中用 PRM 分数提前剪掉低质量中间路径。

简化口径：

```math
B_{t+1}=
\mathrm{TopK}
(
\{b+a:b\in B_t,\ a\in\mathcal{A}(b)\},
S_{\mathrm{avg}}
)
```

面试表达：PRM 搜索剪枝能节省 test-time compute，但如果 PRM 低估正确路径或高估 polished hard negative，就会把好解提前剪掉。

## Process Supervision Audit

一句话定义：Process Supervision Audit 是把最终准确率、步骤准确率、第一处错误检测、自动标注覆盖、人工成本、搜索效果和盲区样本放在一起检查。

典型盲区：最终答案正确但过程错误；过程正确但最终格式错误；冗余步骤很多；PRM 在 hard negative 上给高分。

面试表达：过程监督上线前不能只看 step accuracy，要把 outcome、process、search、cost 和 failure slices 放进同一张审计表。

## Test-Time Compute

一句话定义：test-time compute 是推理阶段额外投入计算来提升答案质量的方法。

为什么提出：模型参数固定后，仍可以通过多次采样、搜索、验证、反思和工具调用提升复杂任务表现。

常见形式：self-consistency、多候选生成、best-of-n、verifier reranking、Tree-of-Thought、MCTS、代码执行验证和迭代修正。

优点：不需要重新训练模型，能按任务难度动态增加计算。

缺点：增加延迟和费用；效果依赖候选多样性和验证器质量；生产系统要控制预算和停止条件。

面试表达：test-time compute 是用推理时搜索和验证换效果，适合高价值、复杂、可验证任务，不适合所有低延迟场景。

## Test-Time Compute Cost

一句话定义：Test-Time Compute Cost 是推理阶段为采样、搜索、验证和工具执行额外消耗的 token、延迟或计算资源。

简化 token 成本：

```math
C_{\mathrm{ttc}}=
\sum_{i=1}^{N}
\sum_{k=1}^{K_i}
T_{ik}
```

其中 `K_i` 是第 `i` 个样本的候选数，`T_ik` 是第 `k` 个候选的 token 成本。

面试表达：reasoning 系统常用更多 test-time compute 换正确率，但上线时必须把准确率增益和 P95 延迟、token 成本、工具成本、安全审核成本放在一起看。

## Test-Time Compute Scaling

一句话定义：Test-Time Compute Scaling 是在模型参数固定时，通过增加推理阶段采样、候选、verifier、搜索、工具和反思预算提升输出质量的方法。

核心问题：哪些请求值得更多计算，预算应该投到采样、搜索、验证还是工具，什么时候停止，以及准确率提升是否抵得过成本和延迟。

面试表达：TTC scaling 不是“所有请求都更慢”，而是把额外计算投给低置信度、高价值、可验证的请求，并用成本和延迟门禁约束。

## Compute Budget Vector

一句话定义：Compute Budget Vector 是把单个请求的推理时预算拆成候选数、推理长度、搜索深度、verifier 调用、工具调用和 token 数的向量。

简化记号：

```math
b_i=(K_i,L_i,D_i,V_i,U_i,T_i)
```

面试表达：预算向量能帮助你解释“多花计算”到底花在哪里，而不是笼统说让模型多思考。

## Adaptive Compute

一句话定义：Adaptive Compute 是根据任务难度、价值、置信度和可验证性动态选择推理预算的策略。

典型路由：简单请求 direct，中等难度请求 self-consistency 或 verifier，高价值可验证难题启用 search / tools。

面试表达：adaptive compute 的价值是用接近高预算策略的准确率，接近低预算策略的成本；风险是难度估计错，把 hard case 分到错误策略。

## Budget Router

一句话定义：Budget Router 是把请求路由到 direct、self-consistency、verifier、search 或 tool loop 的控制模块。

简化记号：

```math
m_i=\pi(x_i,d_i,v_i,u_i)
```

其中 `d_i` 是难度估计，`v_i` 是任务价值，`u_i` 是可验证性。

面试表达：budget router 要可观测、可回放、可灰度，否则很难解释为什么某类请求被分配了高预算。

## Cost per Correct

一句话定义：Cost per Correct 是单位正确样本消耗的成本，用来衡量推理时计算是否划算。

简化口径：

```math
C_{\mathrm{correct}}=
\frac{C_{\mathrm{total}}}
{\sum_i\mathbb{1}[\hat y_i=y_i^\star]}
```

面试表达：如果高预算策略准确率略高但 cost per correct 翻数倍，生产系统未必应该默认开启。

## Marginal Accuracy per Cost

一句话定义：Marginal Accuracy per Cost 衡量从低预算切到高预算时，每增加一单位成本带来的准确率提升。

简化口径：

```math
g(B_1,B_2)=
\frac{A(B_2)-A(B_1)}
{C(B_2)-C(B_1)}
```

面试表达：TTC scaling 通常存在边际收益递减，必须画预算曲线，而不是只测一个最大预算点。

## P95 Latency

一句话定义：P95 Latency 是 95% 请求能完成的延迟阈值，是用户体验和 SLO 里的关键指标。

面试表达：多候选并行、串行搜索和工具 loop 对 P95 的影响不同。线上不能只看平均延迟。

## Wasted High Compute

一句话定义：Wasted High Compute 是高预算请求仍然失败，或低价值请求被错误分配高预算的现象。

面试表达：它是 adaptive compute 的核心坏味道，说明难度估计、路由策略或 verifier / tool 选择需要修正。

## TTC Audit

一句话定义：TTC Audit 是把固定预算策略和 adaptive routing 的准确率、总成本、单位正确成本、P95 延迟、边际收益和浪费样本放在一起审计。

面试表达：TTC audit 能回答“多花的推理计算是否真的值”，而不是只证明某个高预算模式在 benchmark 上分数更高。

## Math Reasoning Training

一句话定义：Math Reasoning Training 是围绕数学题训练模型的多步推理、答案验证、步骤质量、错误定位、verifier、curriculum 和搜索能力。

为什么重要：数学任务答案相对明确、过程可拆解、很多题能自动验证，因此常用于训练和评估 reasoning model。

关键边界：数学 benchmark 分数不等于通用推理能力。数学题可能被公开题库污染，也可能被合成模板、格式偏好或 verifier 偏差放大。

面试表达：数学推理训练不是只增加 CoT 数据，而是要把题目来源、答案验证、步骤标签、错误解法、难度分层、合成题质量和评估污染放在同一个数据闭环里审计。

## Math Training Sample

一句话定义：Math Training Sample 是一条可用于数学推理训练的样本，通常包含题目、标准答案、解题步骤、步骤标签、题型、难度、来源和风险标记。

简化记号：

```math
m_i=(x_i,y_i^\star,z_i,q_i,d_i,t_i,s_i,r_i)
```

面试表达：高质量数学样本不只看答案是否对，还要看步骤是否可复核、难度是否明确、来源是否干净、是否与评测集重叠。

## Answer Supervision

一句话定义：Answer Supervision 是只根据最终答案正确性训练或筛选数学推理样本的方法。

优点：标注成本低，适合答案明确、可自动检查的计算题、选择题和程序化验证题。

缺点：它看不到中间过程，可能奖励碰巧猜对的答案，也会惩罚推理基本正确但最后格式或符号出错的样本。

面试表达：answer supervision 是数学训练的基础信号，但长链条 reasoning 通常还需要 step label、first-error detection、PRM 或人工抽审。

## Synthetic Math Data

一句话定义：Synthetic Math Data 是通过程序模板、规则系统、模型生成、人机协作或自动证明系统构造的数学题、解法和步骤数据。

优点：规模大、难度可控、答案可验证，能覆盖稀有题型和指定技能。

风险：模板痕迹、题型分布偏差、答案或步骤错误、训练到评测污染、模型学会表面模式而不是真正推理。

面试表达：合成数学数据必须同时看正确性、模板多样性、题型覆盖、难度覆盖、真实评测迁移和污染风险。

## Math Curriculum

一句话定义：Math Curriculum 是把数学训练样本按难度、题型、步骤长度、验证难度或概念组合逐步安排的训练策略。

常见顺序：基础计算、代数变形、应用题、几何、组合计数、证明题、奥赛级难题。

面试表达：curriculum 的目的不是让训练集变简单，而是让模型先学稳定局部推理，再逐步学习长链条、抽象和难验证任务。

## Math Contamination Audit

一句话定义：Math Contamination Audit 是检查数学训练集和评测集在题面、答案、解析、模板或参数化变体上是否重叠的审计流程。

常见检查：exact / near dedup、题面改写相似度、答案和解析重叠、模板 ID、来源 URL、公开题库泄漏和参数化变体覆盖。

面试表达：数学题公开解析很多，只做题面去重不够，还要检查答案、步骤、模板和近似变体。

## Template Diversity

一句话定义：Template Diversity 衡量合成题或模板化数据中不同生成模板、参数模式和解题结构的多样性。

简化口径：

```math
D_{\mathrm{temp}}=
\frac{|\{s_i\}_{i=1}^{N}|}{N}
```

面试表达：模板多样性低时，模型可能记住题型外壳；必须结合真实题迁移、hard slice 和错误类型分析判断数据是否有价值。

## Math Training Gate

一句话定义：Math Training Gate 是在数学推理训练数据或模型上线前，对答案准确率、步骤准确率、第一处错误定位、污染率、模板多样性和难题切片设置的质量门禁。

简化口径：

```math
G_{\mathrm{math}}=
\mathbb{1}[
A_{\mathrm{ans}}\ge \alpha
\land
A_{\mathrm{step}}\ge \beta
\land
A_{\mathrm{first}}\ge \gamma
\land
R_{\mathrm{contam}}\le \rho
\land
D_{\mathrm{temp}}\ge \delta
\land
A_{\mathrm{hard}}\ge \eta
]
```

面试表达：数学训练 gate 的价值是防止平均准确率掩盖步骤错误、评测污染、模板化和 hard proof / geometry 切片失败。

## Code Reasoning

一句话定义：Code Reasoning 是模型围绕需求理解、程序生成、测试执行、错误定位、补丁修复和复杂度约束进行多步推理的能力。

为什么重要：代码可以被编译、运行和测试，执行反馈比纯文本评分更硬，因此代码任务非常适合训练 verifier、self-debug、search 和 tool-use reasoning。

关键边界：通过公开测试不等于功能完全正确；测试覆盖不足、环境差异、隐藏边界条件和沙箱安全都会影响代码 reasoning 的真实可靠性。

面试表达：代码 reasoning 的核心不是“写出像代码的文本”，而是把规格、候选代码、测试反馈、修复策略、隐藏泛化和执行安全组成闭环。

## Execution Feedback

一句话定义：Execution Feedback 是编译器、解释器、单元测试、类型检查、静态分析、超时和内存限制返回给模型或控制器的反馈信号。

常见用途：候选 rerank、self-debug、RL reward、搜索剪枝、错误分类和训练数据过滤。

面试表达：执行反馈能把代码任务从语言生成变成可验证任务，但反馈质量取决于测试覆盖、环境一致性和沙箱边界。

## Unit Test Verifier

一句话定义：Unit Test Verifier 是用单元测试判断候选代码是否满足给定输入输出行为的程序化 verifier。

优点：自动、客观、可复现，适合函数级代码生成、修复和 rerank。

局限：单测不完备会漏错；公开测试可能被过拟合；真实仓库任务还需要依赖、环境、集成测试和人工审查。

面试表达：代码任务里 unit test 是最常见的硬 verifier，但必须同时看 public / hidden test gap 和测试覆盖。

## Public Test

一句话定义：Public Test 是训练、调试或提交前可见的测试样例，常用于给模型提供执行反馈。

风险：模型或 reranker 可能只拟合公开样例，生成 hardcode、边界遗漏或样例投机代码。

面试表达：public test 适合用于 self-debug，但不能作为最终泛化证明。

## Hidden Test

一句话定义：Hidden Test 是提交前不可见、用于最终评估泛化能力的测试集合。

常见覆盖：空输入、极值、重复元素、类型边界、性能压力、异常路径和真实业务边界条件。

面试表达：hidden test 才能更好衡量 functional correctness，但也不是形式化证明；测试覆盖仍然需要审计。

## Public-Hidden Gap

一句话定义：Public-Hidden Gap 衡量公开测试表现和隐藏测试表现之间的差距。

简化口径：

```math
\Delta_{\mathrm{pub-hid}}=
A_{\mathrm{pub}}-A_{\mathrm{hid}}
```

面试表达：这个 gap 大说明公开测试覆盖不足、候选选择器过拟合样例，或模型没有真正理解规格。

## Self-Debug

一句话定义：Self-Debug 是模型根据编译错误、运行异常、测试失败或静态检查反馈，解释问题并生成修复补丁的过程。

简化轨迹：

```math
\tau_i=(p_i^{(0)},f_i^{(0)},p_i^{(1)},f_i^{(1)},\ldots,p_i^{(R_i)},f_i^{(R_i)})
```

面试表达：self-debug 的价值来自外部反馈闭环；没有测试或执行信号时，模型反思可能只是重复原错误。

## Repair Success Rate

一句话定义：Repair Success Rate 衡量初始代码失败后，经过若干轮执行反馈和补丁修复，最终通过评估测试的比例。

简化口径：

```math
A_{\mathrm{repair}}=
\frac{1}{N_{\mathrm{fail}}}
\sum_{i\in \mathcal{F}}
\mathbb{1}[
S(p_i^{(R_i)},T_i^{hid})=1
]
```

面试表达：它比 pass@k 更关注模型能否利用反馈修错，而不是候选池里是否碰巧已有正确解。

## Sandbox Violation Rate

一句话定义：Sandbox Violation Rate 衡量候选代码触发危险调用、导入、文件 / 网络访问、无限循环、子进程或资源超限等安全拦截的比例。

简化口径：

```math
R_{\mathrm{sandbox}}=
\frac{1}{M}
\sum_{i,j}
\mathbb{1}[u_{ij}=1]
```

面试表达：执行模型生成代码必须默认不可信。沙箱指标不是攻击指南，而是证明权限隔离和拦截策略有效的审计信号。

## Code Execution Audit

一句话定义：Code Execution Audit 是把 greedy、public rerank、pass@k、public-hidden gap、self-debug 修复、沙箱拦截和执行成本放在一起检查的代码 reasoning 审计。

面试表达：它能回答“执行反馈是否真的提升代码可靠性”，并揭示公开测试过拟合、隐藏边界失败和不安全候选。

## Code Reasoning Gate

一句话定义：Code Reasoning Gate 是代码 reasoning 系统上线前对隐藏测试准确率、公开 / 隐藏差距、修复成功率、沙箱违规率和执行成本设置的门禁。

简化口径：

```math
G_{\mathrm{code}}=
\mathbb{1}[
A_{\mathrm{hid}}\ge \alpha
\land
\Delta_{\mathrm{pub-hid}}\le \delta
\land
A_{\mathrm{repair}}\ge \beta
\land
R_{\mathrm{sandbox}}\le \rho
\land
C_{\mathrm{exec}}\le C_{\max}
]
```

面试表达：代码 reasoning gate 防止系统只靠公开测试或 pass@k 看起来很强，却在隐藏用例、沙箱安全或执行成本上不达标。

## Reasoning Evaluation

一句话定义：Reasoning Evaluation 是对模型多步推理能力的最终答案、过程质量、变体泛化、鲁棒性、污染风险、推理成本和统计不确定性进行统一评估。

为什么重要：reasoning 模型常使用 CoT、self-consistency、verifier、search、工具和 test-time compute。如果不记录预算和评估配置，分数不可复现，也不公平。

面试表达：reasoning eval 不是只跑一个 benchmark，而是要把 benchmark、切片、过程、污染、变体、成本和 paired significance 放进同一份报告。

## Evaluation Sample

一句话定义：Evaluation Sample 是评估集中一条可审计样本，包含原题、标准答案、候选输出、变体、步骤标签、切片标签、风险标记和推理预算。

简化记号：

```math
e_i=(x_i,y_i^\star,V_i,P_i,z_i,q_i,g_i,r_i,b_i)
```

面试表达：评估样本 schema 越完整，越容易追踪分数提升来自模型能力、提示词、预算、污染还是评估器偏差。

## Variant Evaluation

一句话定义：Variant Evaluation 是在原题基础上构造题面改写、数值替换、条件扰动、干扰信息、格式变化或跨题型组合，测试模型是否真正泛化。

面试表达：如果模型原题正确、变体错误，说明它可能依赖记忆、模板或表面线索，不能把原题分数当成稳健 reasoning。

## Robustness Drop

一句话定义：Robustness Drop 衡量模型从原始样本到扰动 / 变体样本的准确率下降。

简化口径：

```math
D_{\mathrm{robust}}=
A_{\mathrm{orig}}-A_{\mathrm{var}}
```

面试表达：robustness drop 大时，要回看扰动类型、任务切片和模型推理过程，而不是只报告原始 benchmark 分数。

## Paired Lift

一句话定义：Paired Lift 是在同一批样本上比较新旧模型逐题正确性差异后的平均提升。

简化口径：

```math
\Delta_{\mathrm{pair}}=
\frac{1}{N}
\sum_{i=1}^{N}
(\mathbb{1}[\hat y_i^{new}=y_i^\star]
-
\mathbb{1}[\hat y_i^{base}=y_i^\star])
```

面试表达：paired evaluation 比两个模型分别报平均分更适合判断小幅改进，因为它控制了样本难度差异。

## Bootstrap Confidence Interval

一句话定义：Bootstrap Confidence Interval 是通过对评估样本重复重采样，估计指标不确定性的置信区间。

用途：小样本或提升很小时，单个平均分可能误导；需要看区间是否跨过 0、是否足以支撑上线判断。

面试表达：如果 paired lift 的 bootstrap interval 跨过 0，应该说“证据不足”，而不是宣称模型显著更强。

## Reasoning Eval Gate

一句话定义：Reasoning Eval Gate 是 reasoning 评估上线前对最终答案、过程质量、第一处错误、污染率、鲁棒性和成本设置的可信度门禁。

简化口径：

```math
G_{\mathrm{eval}}=
\mathbb{1}[
A_{\mathrm{ans}}\ge \alpha
\land
A_{\mathrm{step}}\ge \beta
\land
A_{\mathrm{first}}\ge \gamma
\land
R_{\mathrm{contam}}\le \rho
\land
D_{\mathrm{robust}}\le \delta
\land
E_{\mathrm{cost}}\ge \eta
]
```

面试表达：reasoning eval gate 的目标不是制造漂亮分数，而是证明评估结论没有被污染、统计噪声、过程错误、变体失败或成本失控掩盖。

## Best-of-N

一句话定义：best-of-n 是生成 N 个候选答案，再用评分器或规则选择最优答案的方法。

与 self-consistency 的区别：self-consistency 常用答案投票，best-of-n 更强调用 reward model、verifier 或规则打分选最优。

优点：简单直接，常用于对齐、代码、数学和开放生成质量提升。

缺点：N 越大成本越高；评分器偏差会决定最终输出；如果候选都差，选择器也无能为力。

面试表达：best-of-n 的核心是候选生成加选择器，关键在于候选多样性和选择器可靠性。

## Pass@k

一句话定义：Pass@k 是可执行验证任务中常用指标，表示生成 `k` 个候选时至少有一个正确候选的概率。

给定 `n` 个候选，其中 `c` 个正确，不放回估计为：

```math
\mathrm{pass@}k=
1-
\frac{\binom{n-c}{k}}
{\binom{n}{k}}
```

面试表达：pass@k 衡量候选集合里是否存在正确解，不等于线上一次调用的可靠性。k 越大成本越高，也越依赖测试覆盖率。

## Reasoning Audit

一句话定义：Reasoning Audit 是对 reasoning 系统的准确率、候选质量、verifier、过程步骤、test-time compute 成本、安全风险和污染风险进行统一审计。

一个简化上线门禁：

```math
G_{\mathrm{reason}}=
\mathbb{1}[
A_{\mathrm{ver}}\ge \alpha
\land
A_{\mathrm{step}}\ge \beta
\land
C_{\mathrm{ttc}}\le C_{\max}
\land
R_{\mathrm{unsafe}}\le \rho
]
```

面试表达：reasoning audit 能把“模型会思考”拆成可验证的候选、选择器、步骤质量、成本和安全边界，而不是只看回答是否很长。

## Reasoning Safety

一句话定义：Reasoning Safety 是对推理模型在长链条推理、工具调用、高风险场景、隐藏 CoT、过度自信和滥用边界上的安全性进行系统治理。

核心问题：reasoning 能力越强，模型越可能完成复杂规划和外部动作；如果没有权限、验证、审核和日志，错误或误用的影响会被放大。

面试表达：reasoning safety 不是单个拒答分类器，而是模型行为、CoT 展示策略、verifier、工具权限、人工审核、红队回归和治理门禁组成的系统工程。

## Pseudo Reasoning

一句话定义：Pseudo Reasoning 是模型生成看似合理但不能真正支持最终答案的推理过程。

典型表现：先猜答案再补过程、步骤逻辑跳跃、引用不存在证据、把相关性说成因果性，或最终答案与中间步骤不一致。

面试表达：伪推理的危险在于它让错误更可信。评估时要同时看最终答案、步骤标签、证据支持和反事实扰动，而不是只看解释是否流畅。

## Overconfident Error

一句话定义：Overconfident Error 是模型在错误答案上仍表达高置信度的失败模式。

为什么重要：医疗、法律、金融、安全和招聘等场景中，错而自信会误导用户做高影响决策。

面试表达：过度自信要用 calibration、abstention、verifier、人审和高风险门禁约束，而不是只要求模型输出更长解释。

## Hidden CoT Exposure

一句话定义：Hidden CoT Exposure 是本应隐藏的内部推理、策略边界、隐私推断或工具权限细节进入用户可见输出。

边界：内部推理、安全监控和用户解释应区分开；用户更需要可验证证据、关键假设和最终理由，而不是完整内部轨迹。

面试表达：不展示完整 CoT 不等于不透明；可以用引用、证据摘要、工具结果、置信边界和审计日志建立可复核信任。

## Tool Misuse

一句话定义：Tool Misuse 是 reasoning model 在工具环境中发生权限不匹配、参数污染、不可逆动作未确认或工具输出被误当成上级指令的失败模式。

防护方式：最小权限、schema 校验、沙箱、二次确认、工具输出不可信标记、审计日志和回滚策略。

面试表达：Agent 工具安全的关键是让模型做受约束的计划者，系统层负责权限、确认、执行、审计和回滚。

## Human Review Coverage

一句话定义：Human Review Coverage 是高风险样本中被人工审核或人工确认覆盖的比例。

为什么重要：高风险任务不能只靠模型自动判断，尤其是个体权益、不可逆操作、专业建议和安全相关动作。

面试表达：人审覆盖率不是越高越好，而是要匹配风险分级：低风险自动化，高风险确认，禁止类拒绝。

## Severity-Weighted Risk

一句话定义：Severity-Weighted Risk 是按严重度权重聚合安全失败的风险指标。

为什么需要：低风险格式错误和高风险工具越权不能在平均准确率里同等对待。

面试表达：安全评估要按严重度、风险域和可逆性加权，否则平均分会掩盖少数高影响事故。

## Reasoning Safety Gate

一句话定义：Reasoning Safety Gate 是 reasoning 系统上线前对伪推理、过度自信、高风险不当服从、工具误用、隐藏 CoT 暴露、人审覆盖和过度拒答设置的综合门禁。

简化口径：

```math
G_{\mathrm{safe}}=
\mathbb{1}[
R_{\mathrm{pseudo}}\le \alpha
\land
R_{\mathrm{conf}}\le \beta
\land
R_{\mathrm{unsafe}}\le \gamma
\land
R_{\mathrm{tool}}\le \delta
\land
R_{\mathrm{cot}}\le \epsilon
\land
C_{\mathrm{review}}\ge \eta
]
```

面试表达：如果工具越权、隐藏 CoT 暴露或高风险人工审核不过线，即使 reasoning benchmark 提升，也不能直接上线。

## Reasoning Interview Readiness

一句话定义：Reasoning Interview Readiness 是判断候选人是否能把 reasoning 概念、公式、demo、评估、安全和工程 trade-off 讲成完整面试答案的准备度。

核心维度：topic coverage、formula coverage、demo coverage、risk coverage、trade-off coverage 和 weak question revision。

面试表达：准备 reasoning 面试不能只背 CoT、PRM、MCTS 等术语，而要能用公式、项目审计表和失败案例证明自己理解机制和边界。

## Reasoning Interview Rubric

一句话定义：Reasoning Interview Rubric 是对 reasoning 面试回答进行结构化评分的标准。

常见字段：目标、机制、公式、工程实现、评估指标、风险边界、项目证据和表达清晰度。

面试表达：好的 rubric 能把“讲得流畅”拆成可训练项，避免复盘时只说“还要多看看”。

## Reasoning Formula Coverage

一句话定义：Reasoning Formula Coverage 是 reasoning 面试回答中覆盖关键公式或指标的比例。

典型公式：最终答案准确率、步骤准确率、self-consistency 投票、pass@k、pairwise verifier loss、test-time compute cost 和 safety gate。

面试表达：公式覆盖不是为了炫技，而是证明你知道方法在优化什么、评估什么、约束什么。

## Reasoning Demo Coverage

一句话定义：Reasoning Demo Coverage 是 reasoning 面试准备中能用最小代码或项目审计 demo 支撑的主题比例。

典型 demo：CoT 质量审计、self-consistency 投票、verifier rerank、process supervision、search / MCTS、TTC routing、reasoning eval report 和 safety gate。

面试表达：如果一个知识点能配一个 toy demo，面试回答会从“概念复述”升级成“我知道怎么落地和验证”。

## Weak Reasoning Question

一句话定义：Weak Reasoning Question 是 mock interview 中得分低、公式缺失、demo 缺失或风险边界讲不清的 reasoning 题目。

处理方式：为每个薄弱题绑定一个公式、一个 demo、一个失败案例和一个 3 分钟回答模板。

面试表达：高效复盘不是泛泛复习，而是把弱题拆成可补齐的公式、代码、指标和表达。

## Reasoning Revision Plan

一句话定义：Reasoning Revision Plan 是针对 reasoning 面试薄弱项制定的下一轮修正计划。

常见内容：补数学训练题、补代码执行反馈题、补 process supervision 公式、补 safety gate demo、补系统设计模块图。

面试表达：revision plan 的目标是让下一轮 mock interview 有可验证改善，而不是只增加阅读时间。

## Search Reasoning

一句话定义：Search Reasoning 是把 LLM 推理看成状态空间搜索，在多个中间状态之间展开、评分、剪枝和选择。

为什么提出：单条 CoT 一旦早期走错，后续通常会沿着错误路径继续。搜索保留多个候选路径，可以用 verifier、PRM、工具反馈或规则提前剪掉低质量分支。

核心边界：搜索不是无成本增强。它依赖状态表示、动作粒度、评分器质量和预算门禁；评分器偏差会把错误路径放大。

面试表达：search reasoning 的核心不是“多生成几条答案”，而是把生成、评估、剪枝、回溯和预算控制组成一个闭环。

## Search State

一句话定义：Search State 是搜索过程中当前保留的中间推理状态。

常见内容：原始问题、已生成步骤、工具结果、测试结果、约束满足情况、已消耗 token / 时间 / 调用预算。

简化记号：

```math
s_t=(x,z_{1:t},m_t,b_t)
```

面试表达：状态表示不完整时，verifier 看到的信息不足，搜索评分会失真。

## Search Action

一句话定义：Search Action 是从当前搜索状态可以选择的下一步操作。

例子：写下一步数学推导、选择一个子问题、修改一段代码、调用工具、回退到某个分支。

简化记号：

```math
a_t\in\mathcal{A}(s_t),
\qquad
s_{t+1}=T(s_t,a_t)
```

面试表达：动作粒度太细会让树爆炸，动作粒度太粗又会失去纠错和剪枝能力。

## Beam Search

一句话定义：Beam Search 是按层展开候选，每层只保留分数最高的 `K` 个状态的搜索策略。

简化口径：

```math
F_{t+1}=
\mathrm{TopK}
(
\{T(s,a):s\in F_t,\ a\in\mathcal{A}(s)\},
S,
K
)
```

面试表达：beam search 简单、成本可控，适合接 verifier；风险是 early pruning，正确路径可能因为早期分数低被剪掉。

## Best-First Search

一句话定义：Best-First Search 是每次从全局 frontier 中优先展开当前分数最高的节点。

和 beam search 的区别：beam search 常按层推进，best-first search 更像维护一个全局优先队列。

面试表达：best-first search 在评分器可靠时能快速深入高质量路径；评分器有偏时会更快陷入局部最优。

## UCT

一句话定义：UCT 是 MCTS 中常用的节点选择公式，用“当前价值 + 探索奖励”平衡 exploitation 和 exploration。

简化口径：

```math
U(v)=
Q(v)
+c
\sqrt{
\frac{\log(N_p+1)}{N_v+1}
}
```

面试表达：UCT 避免只盯着当前高分节点，也避免完全随机探索；但在语言任务里 `Q(v)` 的可靠性仍取决于 verifier、工具或最终反馈。

## Prune False Negative

一句话定义：Prune False Negative 是搜索剪枝时把潜在正确路径错误剪掉的失败模式。

典型来源：PRM 低估朴素但正确的分支，偏好格式漂亮的 hard negative，或者早期步骤证据不足。

面试表达：搜索系统上线前要统计被剪掉的正确路径和 hard negative 切片，不能只看最终平均准确率。

## Search Budget

一句话定义：Search Budget 是搜索推理允许消耗的节点数、token、verifier 调用、工具调用、延迟和费用上限。

面试表达：search 更适合高价值复杂任务；简单问答和低延迟场景要用路由策略限制搜索预算。

## Search Audit

一句话定义：Search Audit 是对搜索推理的准确率、候选多样性、剪枝错误、MCTS rescue、节点 / token 成本和 hard negative 进行统一审计。

面试表达：search audit 能判断“搜索是否值得开”，也能定位到底是候选生成不够多样、评分器偏差，还是预算策略过紧。

## Tree-of-Thought

一句话定义：Tree-of-Thought，简称 ToT，是把推理过程看成一棵搜索树，在多个中间思路之间探索和选择。

为什么提出：线性 CoT 一旦早期走错，后续很难恢复。ToT 保留多个中间状态，可以回溯和比较不同推理分支。

基本流程：生成若干中间想法，评估每个想法的潜力，保留较优分支继续扩展，直到得到答案。

优点：适合组合搜索、规划、谜题和复杂推理。

缺点：计算成本高，评估函数难设计，分支数增长快，不适合低延迟任务。

面试表达：ToT 是把语言模型推理从单路径生成扩展为多路径搜索，核心代价是测试时计算增加。

## MCTS

一句话定义：MCTS 是 Monte Carlo Tree Search，蒙特卡洛树搜索，通过选择、扩展、模拟和回传在搜索树中寻找高价值路径。

为什么相关：在复杂推理、代码生成、博弈和规划任务中，模型可以作为策略生成候选动作，verifier 或 reward model 作为价值评估，MCTS 用于系统性搜索。

四个阶段：选择已有高潜力节点，扩展新动作，模拟后续结果，回传更新节点价值。

优点：比简单采样更有结构，能在大搜索空间中平衡探索和利用。

缺点：实现复杂，计算开销大，依赖状态表示和奖励信号；自然语言任务中状态转移和价值评估不如棋类清晰。

面试表达：MCTS 在大模型推理中本质上是把模型生成能力接入搜索框架，用外部评价信号指导多步决策。

## Reflection

一句话定义：reflection 是让模型检查自己前一轮输出，发现问题并修正的机制。

为什么有效：很多错误在模型重新审视答案时可以被发现，尤其是格式遗漏、约束未满足、代码错误和逻辑跳步。

常见流程：生成初稿，模型自评，指出问题，生成修正版；或结合工具测试结果进行修正。

局限：模型可能无法发现自己不知道的错误；没有外部证据或验证器时，反思可能只是重复或改写原错误。

面试表达：reflection 是低成本质量提升手段，但最好和工具验证、测试或外部反馈结合。

## Benchmark

一句话定义：benchmark 是用于衡量模型某类能力的标准化测试集或任务集合。

为什么需要：没有统一评估，就无法比较模型、训练策略、prompt、检索系统和部署版本的效果。

常见类型：知识问答、数学推理、代码生成、阅读理解、指令遵循、安全评估、多模态理解和真实用户偏好评估。

优点：可复现、可横向比较、便于跟踪迭代效果。

风险：污染、过拟合、格式 gaming、不能覆盖真实用户分布、单一分数掩盖能力结构差异。

面试表达：benchmark 是必要但不充分的评估工具。面试中要强调公开榜单、内部集、人工评测和线上指标需要结合。

## Benchmark Contamination

一句话定义：benchmark contamination 是评估集样本泄漏到训练数据中，导致模型分数虚高的现象。

为什么危险：如果模型训练时见过测试题，它的分数就不能代表泛化能力，而更像记忆能力。

检测方法：exact match、n-gram overlap、embedding similarity、near-duplicate search、时间切分、人工抽查和答案异常模式分析。

缓解方法：使用新构造测试集、私有评测集、动态生成题目、时间隔离数据和严格数据去重。

面试表达：看到模型 benchmark 分数时，要追问是否有污染、是否有 prompt tuning、是否使用公开题库，以及是否在真实业务集上验证。

## Human Evaluation

一句话定义：human evaluation 是由人类标注者或专家对模型输出质量进行评价。

为什么需要：开放式生成、对话体验、创意写作、复杂事实核查和业务满意度很难完全用自动指标衡量。

常见方式：单答案打分、两两偏好比较、排序、错误类型标注、专家审查和红队测试。

优势：更接近真实用户体验，可以覆盖细腻的风格、可用性和安全判断。

缺点：成本高，速度慢，主观性强，一致性难保证，标注说明和标注者质量会强烈影响结果。

面试表达：人工评估的关键是设计清晰维度、标注指南、一致性校验和抽样策略，而不是简单让人“觉得哪个好”。

## LLM-as-a-Judge

一句话定义：LLM-as-a-Judge 是用强语言模型作为评审器，对其他模型输出打分、排序或解释错误。

为什么流行：人工评估昂贵且慢，LLM judge 可以低成本、快速、规模化地评估大量样本。

常见用法：pairwise preference、rubric scoring、错误分类、事实一致性初筛和评测报告生成。

风险：长度偏好、位置偏好、模型偏见、格式敏感、专业事实判断不可靠、和被评模型同源导致偏差。

改进方式：使用明确 rubric，打乱答案顺序，隐藏模型名称，引入少量人工校准，使用多 judge 集成，并对 judge 本身做一致性评估。

面试表达：LLM judge 适合做大规模辅助评估，但不能无条件当真值。关键任务仍需要人工或可执行验证闭环。

## Elo Rating

一句话定义：Elo rating 是根据两两对战或偏好比较结果给模型估计相对能力分数的方法。

为什么适用：开放生成任务中直接给绝对分数很难，但人或 judge 更容易判断两个回答哪个更好。

基本思想：强模型战胜弱模型增分少，弱模型战胜强模型增分多。大量 pairwise 比较后，可以得到相对排名。

优点：适合竞技场式评测和对话偏好比较。

局限：结果依赖题目分布、投票人群、对战采样和统计假设；Elo 分不能直接解释为某项能力的绝对水平。

面试表达：Elo 适合衡量相对偏好，但要结合任务切片和置信区间看，不能只看一个总排名。

## Hallucination

一句话定义：hallucination 是模型生成看似合理但不真实、不可靠或缺少依据的内容。

常见类型：事实幻觉、引用幻觉、推理幻觉、上下文幻觉、工具幻觉和身份能力幻觉。

为什么出现：语言模型优化的是下一个 token 概率和人类偏好，不天然等价于事实数据库；当信息缺失、问题模糊、训练记忆冲突或模型过度迎合用户时，容易编造。

影响：在医疗、法律、金融、科研和企业知识库场景中，幻觉会造成严重错误决策。

缓解方法：RAG grounding、引用核查、拒答机制、工具验证、答案不确定性表达、事实性微调、后处理校验和人工审查。

面试表达：幻觉不是单一 bug，而是生成式模型的系统性风险。要从数据、检索、解码、对齐、工具和评估多层治理。

## Factuality

一句话定义：factuality 是模型输出与真实世界事实或给定证据一致的程度。

评估方式：事实问答、引用检查、RAG grounding、人工核查、知识库比对、claim decomposition 和自动事实核查。

关键区别：事实正确不等于有引用支持；有引用也不等于引用真的支持结论。因此需要同时检查答案断言和证据对应关系。

提升方式：使用可靠数据源、检索增强、让模型基于证据回答、要求不知道就说不知道、对关键断言做验证。

面试表达：factuality 关注“说得对不对”，grounding 关注“是否由给定证据支持”。两者相关但不完全相同。

## Citation Grounding

一句话定义：citation grounding 是要求模型输出结论能被引用来源或检索证据支持。

为什么重要：在 RAG 和知识问答中，用户不仅需要答案，还需要知道答案来自哪里、是否可核查。

常见问题：引用不存在、引用位置错误、引用只支持部分结论、多个证据被错误拼接、模型加入证据外知识。

评估方法：把答案拆成原子断言，检查每个断言是否被引用片段支持，并统计 supported、unsupported、contradicted 和 not enough information。

面试表达：引用不是装饰。好的 citation grounding 要验证“答案中的每个关键断言是否被对应证据支持”。

## Robustness

一句话定义：robustness 是模型面对扰动、改写、噪声、分布外输入或攻击时保持稳定表现的能力。

常见扰动：同义改写、顺序变化、无关上下文插入、拼写错误、格式变化、对抗提示、多语言切换和长上下文噪声。

为什么重要：真实用户输入不会像 benchmark 一样干净。模型如果对轻微改写非常敏感，线上体验会不稳定。

评估方法：构造扰动集，比较原始样本和扰动样本表现差异；按任务类型、长度、语言、用户群体和风险等级切片分析。

面试表达：robustness 评估关注模型在分布变化下是否稳定，而不是只看标准测试集平均分。

## Calibration

一句话定义：calibration 是模型表达的置信度与实际正确率一致的程度。

为什么重要：如果模型错得很自信，用户更容易被误导。理想情况下，模型说“高置信”的答案应更可能正确，说“不确定”的答案应确实更难。

常见方法：置信度打分、拒答阈值、选择性回答、温度缩放、基于 verifier 的置信估计和答案一致性估计。

难点：语言模型没有天然可靠的概率置信度；生成文本中的“我确定”不等于真实置信。

面试表达：calibration 关注模型是否知道自己知道什么、不知道什么，是高风险场景可靠性的关键指标。

## Evaluation Metric

一句话定义：evaluation metric 是把模型输出质量转化为可比较数值或标签的评估标准。

常见指标：accuracy、exact match、F1、pass@k、BLEU、ROUGE、BERTScore、胜率、拒答率、幻觉率、事实支持率、人工偏好率。

选择原则：指标必须和任务目标一致。代码任务看 pass@k，问答任务看正确性和引用支持，客服任务看解决率和满意度，安全任务看违规率和过拒率。

常见误区：用摘要指标评估事实问答，用平均分掩盖高风险错误，用自动指标替代人工体验，用单一 benchmark 代表所有能力。

面试表达：评估指标不是越多越好，而是要覆盖核心业务目标、失败风险和用户体验。

## Pass@k

一句话定义：pass@k 是代码生成和可验证任务中常用指标，表示生成 k 个候选中至少一个通过测试的概率。

为什么重要：代码任务常允许多次生成和筛选。pass@1 衡量一次生成能力，pass@k 衡量采样多个候选后的潜在解题能力。

优点：适合有单元测试或自动判题器的任务。

局限：测试用例不完备时，pass 不代表代码完全正确；k 增大带来计算成本；可能鼓励生成大量候选而非单次可靠输出。

面试表达：pass@k 衡量的是候选搜索空间中是否存在正确解，和生产中的延迟、成本、测试覆盖率要一起看。

## Evaluation Slicing

一句话定义：evaluation slicing 是按任务类型、难度、领域、语言、长度、用户群体或风险等级拆分评估结果。

为什么需要：总平均分可能掩盖严重问题。例如总体准确率提升，但医疗样本、长尾语言或长上下文样本下降。

常见切片：数学、代码、知识、指令遵循、安全、长上下文、多轮对话、RAG、工具调用、中文、英文、低资源语言。

面试表达：模型评估不能只看 aggregate score，要看能力切片和失败模式，否则很容易误判模型改进方向。

## Online Evaluation

一句话定义：online evaluation 是在真实线上流量中评估模型表现的方法。

常见形式：A/B test、灰度发布、用户反馈、留存指标、点击率、解决率、人工抽检、投诉率和安全告警。

优点：最接近真实用户分布和业务目标。

风险：线上实验有用户影响，需要严格流量控制、回滚机制、安全监控和统计显著性分析。

面试表达：离线评估决定能不能上线试，线上评估决定真实业务是否变好。两者不能互相替代。

## Evaluation Pipeline

一句话定义：evaluation pipeline 是把测试集、推理配置、打分器、统计分析和报告生成串起来的自动化评估流程。

为什么需要：模型迭代频繁，如果评估不可复现，就无法判断变化来自模型、prompt、数据、解码参数还是评估脚本。

关键组件：固定测试集版本，记录模型版本、prompt、采样参数、工具配置、随机种子、输出日志、打分结果和错误样本。

工程要求：可复现、可追踪、可切片、可回归比较、支持人工复核和失败案例沉淀。

面试表达：成熟团队不会只跑一次 benchmark，而是建设持续评估 pipeline，用回归集防止能力退化。

## Evaluation Metric Incident

一句话定义：evaluation metric incident 是指标看起来提升，但干净集、关键切片、人工评估、线上反馈、成本延迟或安全门禁反而变差的评估事故。

典型表现：公开 benchmark 分数提升但真实用户任务下降，LLM judge 分数升高但人工偏好下降，平均分上升但中文、代码、安全或长上下文切片退化。

面试表达：指标事故排查要同时看样本、切片、污染、judge 校准、统计置信、成本延迟和线上反馈，不能只解释总分。

## Aggregate Score Trap

一句话定义：aggregate score trap 是总体平均分掩盖关键任务、用户群体、语言、难度或风险切片退化的问题。

为什么重要：总分常被简单样本或大切片主导。高风险业务里，一个安全边界或核心任务切片下降，比总体提升更重要。

面试表达：报告总分时必须配切片分、失败样本和上线门禁，否则平均分容易误导决策。

## Slice Regression

一句话定义：slice regression 是 candidate 在某个关键评估切片上相对 baseline 退化。

常见切片：数学、代码、中文、长上下文、RAG 引用、工具调用、安全边界、高价值用户任务和线上高频请求。

面试表达：模型发布前要设置 hard-slice gate。只要关键切片显著退化，就不能只用总体分提升掩盖。

## Clean Eval Lift

一句话定义：clean eval lift 是在排除污染、公开题近重复、调参开发集和可疑样本后，candidate 相对 baseline 的真实提升。

为什么重要：公开 benchmark 上升可能来自污染、记忆或格式适配。干净集提升更接近泛化证据。

面试表达：如果污染切片提升而干净集不提升，应先降级结论，不要把分数解释为能力进步。

## Judge-Human Agreement

一句话定义：judge-human agreement 是 LLM judge 的偏好或评分与人工标注、专家复核或程序验证器结论一致的比例。

用途：校准 judge 是否能作为大规模辅助评估器，尤其要按任务、长度、位置、模型来源和风险等级切片看。

面试表达：LLM-as-a-judge 必须先被评估。judge 自身没有校准，就不能当作上线真值。

## Judge Length Bias

一句话定义：judge length bias 是 LLM judge 倾向选择更长、更结构化或免责声明更多的回答，而不是选择更正确、更有用的回答。

风险：会鼓励模型输出变长、变空、模板化，导致成本升高和用户体验下降。

面试表达：排查 judge 偏差时要控制答案长度、随机化顺序、隐藏模型名，并用人工样本校准。

## Evaluation Gate

一句话定义：evaluation gate 是把成对提升、bootstrap 置信区间、切片退化、污染率、judge-human 一致率、成本、延迟、安全和线上反馈合并成的评估上线门禁。

用途：防止团队只看一个 benchmark、一个 judge 分数或一个平均分就发布模型。

面试表达：evaluation gate 的价值是把“分数好看”转成“证据足够、风险可控、能上线”的决策标准。

## 本章小结

本章围绕 reasoning 和评估建立了核心词汇表。

核心结论如下：

1. reasoning 能力体现为多步拆解、状态维护、验证和修正，而不只是输出更长解释。
2. CoT 可以提升复杂推理，但推理文本不一定真实可靠。
3. self-consistency、best-of-n、ToT 和 MCTS 都是在用测试时计算换更高准确率。
4. verifier 是复杂任务可靠性的关键，尤其适合数学、代码和工具任务。
5. PRM 关注过程正确，ORM 关注最终结果正确。
6. benchmark 必须结合污染风险、任务覆盖和真实业务分布理解。
7. human evaluation 和 LLM-as-a-Judge 都有价值，但都需要明确 rubric 和一致性控制。
8. hallucination、factuality、citation grounding 和 calibration 是模型可靠性评估的核心概念。
9. robustness 关注模型在扰动和分布变化下是否稳定。
10. 成熟评估体系应包含离线 benchmark、内部回归集、人工评测、线上 A/B 和错误分析闭环。

下一章，我们进入安全与治理。
