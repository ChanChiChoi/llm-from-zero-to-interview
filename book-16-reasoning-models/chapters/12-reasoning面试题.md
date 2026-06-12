# 第十二章：Reasoning 面试题

本章是第十六册的综合面试题。前面章节已经覆盖 reasoning model 的概念、CoT、self-consistency、verifier、process supervision、search、test-time compute、数学和代码训练、评估与安全。本章的目标是把这些内容整理成面试中可以直接表达的答案。

面试回答要避免两个问题：一是只背术语，比如 CoT、PRM、MCTS、verifier；二是只说现象，不讲机制和边界。一个好的 reasoning 回答通常包含五层：问题是什么，为什么有效，公式或指标是什么，工程上怎么做，有什么局限和风险。

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修参考公开资料，包括 [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)、[Self-Consistency](https://arxiv.org/abs/2203.11171)、[Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)、[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)、[Tree of Thoughts](https://arxiv.org/abs/2305.10601)、[HumanEval / pass@k](https://arxiv.org/abs/2107.03374)、[OpenAI o1 System Card](https://openai.com/index/openai-o1-system-card/) 和 test-time compute scaling 相关公开论文。它们共同提示：reasoning 面试不是背模型名，而是要能把“生成候选、过程质量、验证器、搜索、推理预算、评估统计、安全门禁”连成一套可落地系统。

本章定位是面试复盘和表达训练，不展开每篇论文的完整证明，也不预测闭源 reasoning 模型的内部实现。面试中遇到未公开细节时，应明确区分官方披露、论文结论、工程常识和个人推断。高风险安全问题保持审计与防御口径，不提供可复用的攻击流程或绕过技巧。

## 12.1 Reasoning 面试的评分框架

面试官通常不是在听你背“CoT、self-consistency、PRM、MCTS”这些词，而是在判断你是否能把一个复杂模型能力拆成可验证系统。

一个高分回答通常包含：

1. 目标：这个方法解决什么问题。
2. 机制：它为什么可能有效。
3. 公式：核心概率、loss、指标或门禁怎么写。
4. 工程：数据、推理流程、工具、日志和预算如何实现。
5. 评估：看哪些指标、哪些切片、哪些失败样本。
6. 边界：成本、延迟、偏差、安全、污染和过度自信。
7. 项目：能否用一个可运行 demo 或项目审计表证明你真的做过。

把第 `i` 道 reasoning 面试题的回答写成：

```math
a_i=(t_i,f_i,d_i,e_i,r_i,u_i)
```

其中 `t_i` 是覆盖的主题，`f_i` 是公式或指标，`d_i` 是 demo 或项目证据，`e_i` 是评估口径，`r_i` 是风险边界，`u_i` 是工程 trade-off。

回答覆盖率：

```math
C_{\mathrm{ans}}=
\frac{
|T_{\mathrm{covered}}|
}{
|T_{\mathrm{required}}|
}
```

公式覆盖率：

```math
C_{\mathrm{formula}}=
\frac{
|F_{\mathrm{covered}}|
}{
|F_{\mathrm{required}}|
}
```

demo 覆盖率：

```math
C_{\mathrm{demo}}=
\frac{
|D_{\mathrm{covered}}|
}{
|D_{\mathrm{required}}|
}
```

一个简化 reasoning 面试准备门禁：

```math
G_{\mathrm{interview}}=
\mathbb{1}[
C_{\mathrm{ans}}\ge \alpha
\land
C_{\mathrm{formula}}\ge \beta
\land
C_{\mathrm{demo}}\ge \gamma
\land
C_{\mathrm{risk}}\ge \rho
\land
C_{\mathrm{trade}}\ge \eta
]
```

这个门禁的直觉是：会讲概念只是最低要求；能写公式、能讲 demo、能说评估和风险，才更接近真实面试要求。

## 12.2 关键公式速查

自回归 reasoning 生成：

```math
P(y,z\mid x)=
\prod_{t=1}^{T}
P(o_t\mid x,o_{1:t-1})
```

其中 `z` 是中间推理过程，`y` 是最终答案，`o_t` 可以是推理步骤、答案 token 或工具调用摘要。

最终答案准确率：

```math
A_{\mathrm{ans}}=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[\hat y_i=y_i^\star]
```

步骤准确率：

```math
A_{\mathrm{step}}=
\frac{
\sum_{i,j}
\mathbb{1}[\hat q_{ij}=q_{ij}]
}{
\sum_i L_i
}
```

self-consistency 投票：

```math
\hat y_i=
\arg\max_y
\sum_{j=1}^{K}
\mathbb{1}[\mathrm{norm}(\hat y_{ij})=y]
```

pass@k 不放回估计：

```math
\mathrm{pass@}k=
1-
\frac{\binom{n-c}{k}}{\binom{n}{k}}
```

pairwise verifier loss：

```math
L_{\mathrm{pair}}=
-\log \sigma(r_w-r_l)
```

process supervision loss：

```math
L_{\mathrm{proc}}=
-\frac{1}{M}
\sum_{i,j}
\left[
q_{ij}\log p_{ij}
+
(1-q_{ij})\log(1-p_{ij})
\right]
```

test-time compute 成本：

```math
C_i=
c_{\mathrm{tok}}T_i
+c_{\mathrm{cand}}K_i
+c_{\mathrm{ver}}V_i
+c_{\mathrm{tool}}U_i
+c_{\mathrm{lat}}R_i
```

reasoning 安全门禁：

```math
G_{\mathrm{safe}}=
\mathbb{1}[
R_{\mathrm{pseudo}}\le \alpha
\land
R_{\mathrm{conf}}\le \beta
\land
R_{\mathrm{tool}}\le \delta
\land
C_{\mathrm{review}}\ge \eta
]
```

这些公式不需要在每道题里全部写出，但面试中至少要能根据题目拿出对应公式。例如 CoT 题讲 `A_step`，self-consistency 题讲投票和 pass@k，verifier 题讲 pairwise loss 和校准，TTC 题讲成本，安全题讲 `G_safe`。

## 12.3 什么是 Reasoning Model

回答要点：

```text
Reasoning model 是强调多步推理、规划、验证和自我修正能力的模型或系统。它不只是直接生成答案，而是能把复杂任务拆成多个中间步骤，通过 CoT、verifier、搜索、工具调用或 test-time compute 来提升复杂问题的解决能力。典型场景包括数学、代码、规划、科学问答和复杂决策。
```

补充说明：

```text
需要强调 reasoning model 不等于简单“输出更长解释”。真正的 reasoning 关注中间步骤是否正确、是否能泛化、是否能用验证器或工具检查，而不是答案看起来更像推理。
```

专家追问：

```text
我会把 reasoning model 拆成训练侧和推理侧。训练侧包括数学、代码、过程标注、偏好数据和 verifier 数据；推理侧包括候选生成、投票、verifier rerank、搜索、工具反馈和预算路由。评估时必须同时看最终答案、过程、成本、鲁棒性和安全。
```

## 12.4 CoT 为什么有效

回答要点：

```text
CoT 有效的原因是它把复杂问题拆成多个中间步骤，降低一次性映射的难度，让模型显式生成中间状态。对于数学、逻辑和规划任务，中间步骤可以帮助模型保持上下文、分解子问题并减少跳步。但 CoT 不是万能的，它可能产生伪推理、错误传播和冗长无效解释，所以通常需要 verifier、过程监督或工具验证配合。
```

面试展开：

```text
我会区分“生成 CoT”和“CoT 真的正确”。CoT 能提高表现，但不能自动保证推理忠实性。因此评估时要看步骤正确率、第一处错误和最终答案，而不是只看解释长度。
```

常见误区：

1. 把 CoT 长度当成推理能力。
2. 认为 CoT 一定忠实反映内部计算。
3. 忘记简单题可能出现 CoT regression。
4. 在安全场景默认展示完整内部推理。

## 12.5 Zero-shot CoT 和 Few-shot CoT 的区别

回答要点：

```text
Zero-shot CoT 是通过类似“请一步步思考”的提示让模型生成推理步骤，不需要示例。Few-shot CoT 会在 prompt 中给出带推理过程的示例，引导模型模仿解题格式和推理粒度。Few-shot 通常更稳定，但依赖示例质量，也可能带来格式过拟合。Zero-shot 更简单，但对模型基础能力要求更高。
```

工程补充：

```text
实际系统里我会把 CoT prompt 当成可评估配置，而不是拍脑袋写一句提示。要记录 prompt 版本、示例来源、采样参数、token 成本、正确率、步骤准确率和安全展示策略。
```

## 12.6 Self-Consistency 是什么

回答要点：

```text
Self-consistency 是对同一个问题采样多条推理路径，然后对最终答案进行投票或聚合。它利用了正确答案更可能在多条独立路径中重复出现的现象，可以降低单次采样偶然错误。它适合答案明确的数学和逻辑题，但对开放式任务或错误答案高度一致的场景效果有限。
```

公式表达：

```math
\hat y=
\arg\max_y
\sum_{j=1}^{K}
\mathbb{1}[\mathrm{norm}(\hat y_j)=y]
```

追问回答：

```text
self-consistency 和 pass@k 不一样。self-consistency 看投票后最终选择是否正确，pass@k 看候选集合里是否至少存在一个正确解。如果多数候选一起错，self-consistency 会失败；如果 verifier 能挑出少数正确候选，pass@k 高仍然有价值。
```

## 12.7 Verifier 和 Reward Model 有什么区别

回答要点：

```text
Verifier 更强调判断一个答案或步骤是否正确，常用于数学、代码和可验证任务。Reward model 更一般，用来给模型输出打偏好分，可以包含有用性、安全性、风格和正确性等维度。在 reasoning 中，verifier 常用于候选筛选、reranking、搜索剪枝和 RL reward。如果 verifier 不可靠，search 和 best-of-N 反而可能放大错误。
```

公式表达：

```math
L_{\mathrm{pair}}=
-\log \sigma(r_w-r_l)
```

工程补充：

```text
我会优先区分 programmatic verifier、learned outcome verifier 和 process verifier。数学表达式、代码测试这类任务可以用程序化信号；开放任务需要 learned verifier 或 LLM judge，但必须做 calibration、hard negative 和切片评估。
```

## 12.8 Outcome Supervision 和 Process Supervision 的区别

回答要点：

```text
Outcome supervision 只监督最终答案是否正确，信号简单但稀疏，不知道中间哪一步错。Process supervision 会监督推理过程中的每一步是否正确，能提供更密集的训练信号，也适合训练 process reward model。它的主要问题是步骤标注成本高、粒度难统一，而且过程评分器本身也可能有偏差。
```

面试展开：

```text
我会同时看 outcome accuracy 和 step accuracy。最终答案正确但过程错误，说明模型可能 lucky；过程正确但最终格式错误，说明需要结果 verifier 或格式约束。两者不应互相替代。
```

## 12.9 PRM 是什么

回答要点：

```text
PRM 是 process reward model，用来给推理过程中的中间步骤打分。它可以帮助搜索时剪枝、帮助 reranking 选择更可靠的推理链，也可以作为训练中的过程奖励。PRM 的价值在于它不只看最终答案，而是能判断当前推理路径是否还值得继续。但训练 PRM 需要高质量步骤标注或可靠的自动过程监督。
```

公式表达：

```math
S_{\mathrm{proc}}(z)=
\frac{1}{L}
\sum_{j=1}^{L}
p_j
```

追问回答：

```text
PRM 的风险是偏好“看起来规范”的步骤，低估朴素但正确的路径，或者在 hard negative 上被欺骗。所以 PRM 要和 outcome verifier、工具验证、人工抽检和 search failure 分析一起用。
```

## 12.10 Tree-of-Thought 相比 CoT 的优势是什么

回答要点：

```text
CoT 通常是一条推理链，早期步骤错了后面容易全部偏。Tree-of-Thought 把推理组织成树，让模型在每一步生成多个候选 thought，再通过 verifier、规则或工具反馈选择更好的分支继续展开。它能探索多种解法，适合复杂规划和数学任务。代价是计算成本高，并且非常依赖评分函数质量。
```

工程补充：

```text
我会把 ToT 看成 search controller，而不是单纯 prompt trick。系统里要定义 state、action、transition、score、stop condition 和 budget。没有这些，ToT 很容易变成昂贵的多次生成。
```

## 12.11 Beam Search 和 MCTS 如何比较

回答要点：

```text
Beam search 通常按层展开，每层保留 top-k 候选，简单、稳定、成本可控，但容易过早剪掉潜在正确路径。MCTS 通过 selection、expansion、simulation 和 backpropagation 估计节点价值，更强调探索和利用的平衡，适合规划和博弈类任务，但实现复杂、成本高，并且依赖可靠 reward。
```

公式表达：

```math
U(v)=
Q(v)
+c
\sqrt{
\frac{\log(N_p+1)}{N_v+1}
}
```

面试边界：

```text
在语言 reasoning 中，MCTS 的难点不是公式本身，而是状态如何表示、模拟结果如何评估、reward 是否可靠，以及成本是否可控。
```

## 12.12 Test-Time Compute Scaling 是什么

回答要点：

```text
Test-time compute scaling 是指模型参数固定时，通过增加推理阶段计算提升答案质量。典型方式包括长 CoT、多样本采样、self-consistency、best-of-N、verifier reranking、tree search、工具调用和反思修正。核心问题是如何分配预算，以及如何在准确率、延迟和成本之间做权衡。
```

公式表达：

```math
C_i=
c_{\mathrm{tok}}T_i
+c_{\mathrm{cand}}K_i
+c_{\mathrm{ver}}V_i
+c_{\mathrm{tool}}U_i
+c_{\mathrm{lat}}R_i
```

追问回答：

```text
更多 test-time compute 不一定更好。它会带来边际收益递减、P95 延迟上升、成本增加和高预算仍失败样本。上线时要用 adaptive compute，而不是所有请求默认最高预算。
```

## 12.13 Adaptive Compute 怎么设计

回答要点：

```text
我会先用路由器判断任务类型、难度、价值、风险和可验证性。简单问题直接回答；中等问题使用多样本采样或 self-consistency；困难且可验证的问题启动 verifier、search 或工具调用；高风险问题进入人工审核或更严格门禁。系统需要限制 token、采样数、搜索深度、工具调用次数和总延迟。
```

简化路由：

```math
m_i=\pi(x_i,d_i,v_i,u_i,r_i)
```

其中 `d_i` 是难度，`v_i` 是业务价值，`u_i` 是可验证性，`r_i` 是风险等级。高价值、可验证、低安全风险的复杂题更适合高预算；低价值或高风险任务不能无脑增加搜索。

## 12.14 如何训练数学推理模型

回答要点：

```text
数学推理训练通常需要题目答案数据、步骤解法数据、多解法数据、错误纠错数据和合成题数据。训练流程可以先用 SFT 学习基本解题格式和推理步骤，再用 verifier 或自动检查过滤数据，进一步通过 reranking 或 RL 强化正确解法。还要做难度 curriculum，从基础计算到复杂证明逐步提升，并注意 benchmark 污染和合成题模板化。
```

工程指标：

```text
我会同时看 answer accuracy、step accuracy、first-error accuracy、template diversity、contamination rate、topic coverage 和 hard slice accuracy。数学 benchmark 分数不能单独代表通用 reasoning。
```

## 12.15 数学 Verifier 有什么用

回答要点：

```text
数学 verifier 可以用于数据清洗、候选 reranking、过程评分、搜索剪枝和 RL reward。程序化 verifier 在表达式、数值计算、方程检查等任务上可靠，但覆盖范围有限；LLM verifier 覆盖范围更广，但可能被表面合理的错误推导欺骗。因此实际系统常把规则、符号计算、工具验证和模型评分结合起来。
```

追问回答：

```text
Verifier 的评估要看 pairwise accuracy、top-1 rerank accuracy、hard negative accuracy、calibration 和下游提升。只看 verifier loss 不够，因为它可能学到格式偏好，而不是正确性。
```

## 12.16 如何训练代码推理模型

回答要点：

```text
代码推理模型可以利用“生成代码 -> 执行测试 -> 读取反馈 -> 修复代码”的闭环训练。数据可以包括需求到代码、错误代码到修复、测试失败信息到修复解释、多候选代码和执行结果。执行反馈可以作为 verifier 和 reward，例如编译通过、单元测试通过率、隐藏测试通过率、无超时等。工程上必须有安全沙箱、资源限制和环境记录。
```

重点指标：

```text
我会区分 public tests 和 hidden tests。公开测试通过不等于泛化正确，所以要看 public-hidden gap、pass@k、repair success rate、首次通过轮数、资源限制和 sandbox violation rate。
```

## 12.17 Self-Debug 系统怎么做

回答要点：

```text
系统先生成初始代码，在沙箱中运行测试，收集编译错误、运行异常和失败用例。模型根据反馈分析失败原因并生成补丁，再运行测试验证。控制器需要限制最大轮数、超时和资源，并记录历史修改，避免反复震荡。最终根据公开测试、隐藏测试、静态检查和代码质量选择候选。
```

工程边界：

```text
self-debug 的风险是过拟合公开测试或在反馈噪声下反复改坏代码。必须有隐藏测试、静态扫描、沙箱和最大迭代预算。
```

## 12.18 如何评估 Reasoning Model

回答要点：

```text
评估要覆盖最终答案、过程质量、泛化能力、鲁棒性和推理成本。数学可以看 GSM、MATH 类 benchmark 和变体题；代码可以看 Pass@1、Pass@k、隐藏测试通过率和 self-debug 提升。还要做污染检测，报告采样数、token、verifier 和工具调用预算。过程评估要看步骤正确率、第一处错误位置和推理链是否支持最终答案。
```

可信报告应该包含：

1. 评测集版本和污染检查。
2. prompt、采样参数、候选数和随机种子。
3. final answer、process、variant 和 slice。
4. paired lift 和 confidence interval。
5. token、latency、verifier、tool 和 cost per correct。
6. failure cases 和安全门禁。

## 12.19 为什么不能只看 Benchmark 分数

回答要点：

```text
Benchmark 分数可能受到数据污染、prompt、推理预算、采样次数、verifier 和评估器偏差影响。一个模型分数高，可能是因为见过题，或者用了更多 test-time compute。Reasoning 还要看过程是否正确、变体题是否泛化、成本是否可接受、失败案例是什么，以及是否存在伪推理和过度自信。
```

面试高分补充：

```text
我会要求同预算比较，并报告 cost-normalized accuracy。否则一个模型用 1 次 greedy，另一个模型用 64 次采样加 verifier，直接比较准确率是不公平的。
```

## 12.20 Reasoning Model 有哪些安全风险

回答要点：

```text
主要风险包括伪推理、过度自信、长链条错误传播、reward hacking、工具误用、hidden CoT 暴露、高风险场景误导、过度拒答，以及更强规划能力带来的滥用风险。缓解上需要 verifier、工具权限控制、沙箱、人工审核、日志审计、安全评估、不确定性表达和红队回归，不能只依赖模型自己解释。
```

指标补充：

```text
我会同时看 unsafe compliance rate、over-refusal rate、tool misuse rate、hidden CoT exposure rate、high-risk review coverage 和 severity-weighted risk。安全不是拒答率越高越好，而是正确拦截高风险，同时保持良性任务可用。
```

## 12.21 为什么完整 CoT 不一定应该展示给用户

回答要点：

```text
完整 CoT 可能泄露系统策略、隐私推断、工具权限或安全边界，也可能把不忠实的中间想法包装成解释。产品中更合理的做法通常是展示简洁、可验证的解释、关键依据或证据摘要，而不是原始内部推理轨迹。需要区分模型内部 reasoning、安全监控和对用户展示的可审计解释。
```

追问回答：

```text
不展示完整 CoT 不等于不可解释。可以展示引用、证据、关键假设、工具结果、置信边界和最终检查清单，让用户能复核结论，而不是暴露完整隐藏推理。
```

## 12.22 设计一个 Reasoning 系统

回答框架：

```text
我会把系统拆成 router、generator、verifier、search controller、tool executor、budget manager、aggregator、safety gate 和 logger。Router 判断任务类型、难度、价值、风险和可验证性；generator 生成候选或步骤；verifier 评分；search controller 控制展开和剪枝；tool executor 提供外部验证；budget manager 控制 token、延迟和调用次数；aggregator 输出最终答案；safety gate 处理高风险和工具权限；logger 用于审计和失败分析。
```

关键补充：

```text
系统要有降级策略。简单问题不需要 search；高风险问题需要人工审核或更强验证；工具调用必须做权限控制和沙箱隔离；评估时要报告质量、成本、延迟和安全。
```

## 12.23 面试中的高分表达模板

通用模板：

```text
这个问题我会从五层回答：第一，它解决什么问题；第二，为什么这种方法有效；第三，核心公式或指标是什么；第四，工程上如何实现；第五，它的局限、评估和安全边界是什么。
```

用于 CoT：

```text
CoT 解决的是复杂问题一次性生成困难的问题。它通过显式中间步骤降低推理难度。工程上可以用 zero-shot、few-shot、SFT 或 reasoning 数据训练。但它可能产生伪推理，所以需要过程评估、verifier 或工具验证。
```

用于 verifier：

```text
Verifier 解决的是候选选择和错误识别问题。它可以用于 reranking、search 剪枝和 reward。工程上要区分 outcome verifier、process verifier 和 programmatic verifier。局限是 verifier 本身可能有偏差，可能被格式或流畅性欺骗。
```

用于 test-time compute：

```text
Test-time compute 解决的是单次采样不稳定的问题。它通过多采样、搜索、验证和工具反馈提升质量。工程上要做 adaptive compute 和预算控制。局限是延迟、成本和边际收益递减。
```

用于安全：

```text
Reasoning safety 解决的是更强规划、工具调用和高风险建议带来的边界问题。工程上要把 verifier、权限、沙箱、人工审核、日志和红队回归串起来。评估时不能只看拒答率，也要看 over-refusal 和良性任务可用性。
```

## 12.24 最小可运行 reasoning 面试复盘 demo

下面的 0 依赖 demo 用一轮 toy mock interview 记录检查：是否覆盖了 reasoning 主题、公式、demo、风险和 trade-off。它适合在面试前自查“我是不是只会背概念，而不能讲公式和项目”。

```python
REQUIRED_TOPICS = {
    "reasoning_model", "cot", "self_consistency", "verifier",
    "process_supervision", "search", "test_time_compute",
    "math_training", "code_feedback", "evaluation", "safety", "system_design"
}
REQUIRED_FORMULAS = {
    "A_ans", "A_step", "pass_at_k", "G_ver", "G_proc", "C_ttc", "G_safe"
}
REQUIRED_DEMOS = {
    "cot_audit", "sc_vote", "verifier_rerank", "process_audit",
    "search_audit", "ttc_router", "eval_report", "safety_gate"
}
REQUIRED_RISKS = {
    "pseudo_reasoning", "overconfidence", "contamination",
    "tool_misuse", "over_refusal"
}
REQUIRED_TRADEOFFS = {
    "accuracy_cost", "latency_quality", "verifier_bias", "safety_helpfulness"
}

ANSWERS = [
    dict(
        q="q1_reasoning_model",
        topics={"reasoning_model", "test_time_compute"},
        formulas={"C_ttc"},
        demos={"ttc_router"},
        risks={"overconfidence"},
        tradeoffs={"accuracy_cost"},
    ),
    dict(
        q="q2_cot",
        topics={"cot"},
        formulas={"A_step"},
        demos={"cot_audit"},
        risks={"pseudo_reasoning"},
        tradeoffs={"accuracy_cost"},
    ),
    dict(
        q="q3_self_consistency",
        topics={"self_consistency"},
        formulas={"pass_at_k"},
        demos={"sc_vote"},
        risks=set(),
        tradeoffs={"accuracy_cost"},
    ),
    dict(
        q="q4_verifier",
        topics={"verifier"},
        formulas={"G_ver"},
        demos={"verifier_rerank"},
        risks={"verifier_bias"},
        tradeoffs={"verifier_bias"},
    ),
    dict(
        q="q5_process",
        topics={"process_supervision"},
        formulas=set(),
        demos={"process_audit"},
        risks=set(),
        tradeoffs={"verifier_bias"},
    ),
    dict(
        q="q6_search_ttc",
        topics={"search", "test_time_compute"},
        formulas={"C_ttc"},
        demos={"search_audit", "ttc_router"},
        risks=set(),
        tradeoffs={"latency_quality"},
    ),
    dict(
        q="q7_eval",
        topics={"evaluation"},
        formulas={"A_ans"},
        demos={"eval_report"},
        risks={"contamination"},
        tradeoffs={"accuracy_cost"},
    ),
    dict(
        q="q8_safety",
        topics={"safety"},
        formulas={"G_safe"},
        demos=set(),
        risks={"tool_misuse", "over_refusal"},
        tradeoffs={"safety_helpfulness"},
    ),
]


def union(field):
    out = set()
    for item in ANSWERS:
        out |= item[field]
    return out


def coverage(observed, required):
    return round(len(observed & required) / len(required), 3)


observed = {
    "topics": union("topics"),
    "formulas": union("formulas"),
    "demos": union("demos"),
    "risks": union("risks"),
    "tradeoffs": union("tradeoffs"),
}
summary = {
    "topic_coverage": coverage(observed["topics"], REQUIRED_TOPICS),
    "formula_coverage": coverage(observed["formulas"], REQUIRED_FORMULAS),
    "demo_coverage": coverage(observed["demos"], REQUIRED_DEMOS),
    "risk_coverage": coverage(observed["risks"], REQUIRED_RISKS),
    "tradeoff_coverage": coverage(observed["tradeoffs"], REQUIRED_TRADEOFFS),
}
missing = {
    "topics": sorted(REQUIRED_TOPICS - observed["topics"]),
    "formulas": sorted(REQUIRED_FORMULAS - observed["formulas"]),
    "demos": sorted(REQUIRED_DEMOS - observed["demos"]),
    "risks": sorted(REQUIRED_RISKS - observed["risks"]),
    "tradeoffs": sorted(REQUIRED_TRADEOFFS - observed["tradeoffs"]),
}
question_scores = {}
for item in ANSWERS:
    score = 0
    score += 2 if item["topics"] else 0
    score += 1 if item["formulas"] else 0
    score += 1 if item["demos"] else 0
    score += 1 if item["risks"] else 0
    score += 1 if item["tradeoffs"] else 0
    question_scores[item["q"]] = round(score / 6, 3)

weak_questions = [q for q, score in question_scores.items() if score < 0.75]
revision_plan = {
    "math_training": "补一道数学训练题，讲清数据 schema、污染检测、课程学习和 verifier 过滤。",
    "code_feedback": "补一道代码执行反馈题，讲清 public-hidden gap、sandbox 和 repair success。",
    "process_formula": "把 process supervision 回答补上 A_step 或 G_proc。",
    "safety_demo": "给 safety 回答绑定一个 toy safety gate demo。",
}
gates = {
    "topic_ok": summary["topic_coverage"] >= 0.95,
    "formula_ok": summary["formula_coverage"] >= 0.85,
    "demo_ok": summary["demo_coverage"] >= 0.85,
    "risk_ok": summary["risk_coverage"] >= 0.80,
    "tradeoff_ok": summary["tradeoff_coverage"] >= 0.90,
    "weak_question_ok": len(weak_questions) == 0,
}

print(f"summary={summary}")
print(f"missing={missing}")
print(f"question_scores={question_scores}")
print(f"weak_questions={weak_questions}")
print(f"revision_plan={revision_plan}")
print(f"gates={gates}")
print(f"interview_ready={all(gates.values())}")
```

预期输出：

```text
summary={'topic_coverage': 0.75, 'formula_coverage': 0.857, 'demo_coverage': 0.875, 'risk_coverage': 1.0, 'tradeoff_coverage': 1.0}
missing={'topics': ['code_feedback', 'math_training', 'system_design'], 'formulas': ['G_proc'], 'demos': ['safety_gate'], 'risks': [], 'tradeoffs': []}
question_scores={'q1_reasoning_model': 1.0, 'q2_cot': 1.0, 'q3_self_consistency': 0.833, 'q4_verifier': 1.0, 'q5_process': 0.667, 'q6_search_ttc': 0.833, 'q7_eval': 1.0, 'q8_safety': 0.833}
weak_questions=['q5_process']
revision_plan={'math_training': '补一道数学训练题，讲清数据 schema、污染检测、课程学习和 verifier 过滤。', 'code_feedback': '补一道代码执行反馈题，讲清 public-hidden gap、sandbox 和 repair success。', 'process_formula': '把 process supervision 回答补上 A_step 或 G_proc。', 'safety_demo': '给 safety 回答绑定一个 toy safety gate demo。'}
gates={'topic_ok': False, 'formula_ok': True, 'demo_ok': True, 'risk_ok': True, 'tradeoff_ok': True, 'weak_question_ok': False}
interview_ready=False
```

这里 `interview_ready=False` 是预期结果：demo 故意留下数学训练、代码反馈、系统设计和 process supervision 公式缺口，用来说明复盘脚本应该暴露短板，而不是给自己一个漂亮分数。

## 12.25 小练习

1. 从本章任选 10 个问题，每题用“目标 -> 机制 -> 公式 -> 工程 -> 评估 -> 边界”六段式回答。
2. 写出 `A_ans`、`A_step`、`pass@k`、`L_pair`、`C_i` 和 `G_safe` 的公式，并说明每个符号含义。
3. 给一轮 reasoning mock interview 设计评分表，字段包含 topic coverage、formula coverage、demo coverage、risk coverage、trade-off coverage 和 weak questions。
4. 选择你最薄弱的一题，补一个 0 依赖 Python toy demo 或一个项目审计表。
5. 用 5 分钟讲清楚“如何设计一个可上线的 reasoning 系统”，要求同时覆盖 router、generator、verifier、search、tool、budget、safety 和 logging。

## 12.26 本章小结

Reasoning 面试的核心不是背术语，而是能把方法、机制、公式、工程实现、评估和局限连起来。CoT 解决分步推理，self-consistency 解决单次采样不稳定，verifier 解决候选选择和错误识别，process supervision 解决中间步骤监督，search 和 test-time compute 解决复杂问题探索，数学和代码提供可验证训练场景，评估和安全决定这些能力能否可靠落地。

到这里，第十六册《Reasoning Model、长思维链与可验证推理》完成第二轮阶段性精修。本册主线是：从“模型如何一步步思考”到“如何训练、验证、搜索、评估、安全使用 reasoning 能力”，再到“如何在面试中把这些能力讲成可落地系统”。后续进入 Agent 与工具调用专题时，要把本册的 verifier、search、tool、safety gate 和 audit 思路继续迁移到 Agent 系统中。
