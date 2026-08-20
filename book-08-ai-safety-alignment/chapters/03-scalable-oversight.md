# 第三章：Scalable Oversight

一个人可以检查简单的答案，却很难在有限时间内检查一段包含几十个工具调用的
Agent 轨迹、一份长合同的每个风险点，或一个复杂证明的每一步。若仍然要求专家
逐条直接判定，监督会同时受到成本、知识、注意力和覆盖范围的限制；若把判断全部
交给另一个模型，错误又可能被批量放大。

Scalable Oversight 研究的就是这个中间地带：如何把人类的目标和价值判断保留下来，
再借助分解、AI critique、debate、verifier、原则和人工升级，让监督信号覆盖更
复杂的任务。本章不仅介绍方法名称，也说明每种方法依赖什么假设、会在哪些地方
失效，以及如何在真实系统中留下可审计证据。

资料依据包括 AI Safety via Debate、Iterated Amplification、Scalable agent alignment
via reward modeling、Constitutional AI、Learning to Summarize from Human Feedback、
Let's Verify Step by Step、weak-to-strong generalization、OpenAI Evals / Model Spec、
NIST AI RMF / Generative AI Profile 等公开论文、规范和框架。论文支持特定方法与
实验条件，治理框架支持风险组织；它们都不能单独证明某个监督系统在生产环境中
可靠。

```text
复杂任务 -> 监督瓶颈 -> 分解 / AI critique / debate / verifier / 人工审计 -> 监督证据与治理动作
```

本章不把 Debate、Iterated Amplification、Recursive Reward Modeling、Constitutional
AI 或 AI feedback 写成已经解决 alignment 的银弹。它们都是监督增强方案，需要
人工 gold set、工具验证、高风险人工复核、分布外评估和回归测试共同约束。初学者
可以把 scalable oversight 理解为“让有限的专家注意力用在最需要的地方”；工程
读者还要继续追问监督信号的来源、相关错误、升级分母和单位成功成本。

## 1. 来龙去脉：为什么需要 Scalable Oversight

### 1.1 最早的监督假设

传统监督学习有一个隐含假设：人类知道正确答案。

例如：

1. 图片里是猫还是狗。
2. 句子情感是正面还是负面。
3. 翻译是否基本正确。
4. 数学题最终答案是否对。

只要人类能判断，标注数据就能提供训练信号。

### 1.2 问题变复杂后，人类判断开始吃力

大模型任务变复杂后，这个假设不再稳。

例如：

1. 一个复杂代码补丁是否引入安全漏洞？
2. 一个长法律合同分析是否遗漏关键风险？
3. 一个医学建议是否符合最新指南？
4. 一个多步数学证明是否每一步都正确？
5. 一个 Agent 的 30 步工具调用计划是否安全？
6. 一个 RAG 回答是否忠实整合了 20 篇文档？

这些任务中，普通标注员很难直接判断。

甚至专家也需要大量时间、工具和协作。

### 1.3 RLHF 的监督瓶颈

RLHF 用人类偏好来训练模型，解决了“手写 reward function 很难”的问题。

但它仍然依赖人类判断。

如果人类看不懂任务，偏好数据就可能错误。

例如：

1. 模型写出一个看似合理但有 bug 的程序。
2. 模型给出一个流畅但错误的医学解释。
3. 模型引用了文档，但引用并不支持结论。
4. 模型提出一个复杂计划，其中第 17 步有权限风险。

人类如果只看表面，会把错误答案标成好答案。

这会让模型学会“骗过监督者”，而不是真正解决问题。

### 1.4 Scalable Oversight 的问题定义

Scalable Oversight 要解决的问题是：

```text
当任务太复杂，单个人类无法直接可靠判断时，如何仍然构造可靠监督信号？
```

核心不是“减少标注成本”这么简单。

更本质的是：

1. 如何监督超过人类直接判断能力的模型？
2. 如何把复杂任务拆成可判断的小问题？
3. 如何借助 AI 帮助人类监督 AI？
4. 如何避免 AI feedback 放大模型自身偏差？
5. 如何让监督过程可验证、可追溯、可扩展？

## 2. 小白例子：老师批改超过自己能力的作业

假设一个老师要批改学生的超复杂数学证明。

如果老师自己看不懂证明，就有几个选择。

第一，直接看最终答案。

问题：学生可能写对答案但证明错。

第二，请专家批改。

问题：专家贵，而且不可能批改所有作业。

第三，让学生把证明拆成很多小步骤，每一步都解释。

问题：老师可以逐步检查，但仍然可能漏掉细节。

第四，让两个学生辩论，一个指出另一个证明的问题。

问题：如果辩论规则设计好，老师只需要判断谁指出的关键点更可信。

第五，让一个助手帮老师查资料、验证步骤、运行计算。

问题：助手本身也可能错，需要校验。

Scalable oversight 研究的就是这些路线在 AI 监督中的对应形式。

## 3. 人类监督瓶颈

### 3.1 成本瓶颈

人工标注很贵。

尤其是：

1. 专家标注。
2. 多轮对话标注。
3. 长上下文标注。
4. 安全红队标注。
5. 代码和数学验证。

如果每个样本都需要专家花 30 分钟，训练数据规模很难扩大。

### 3.2 能力瓶颈

人类可能没有足够专业知识。

例如普通标注员无法判断：

1. 生物安全风险。
2. 复杂网络攻击链。
3. 大规模分布式训练 bug。
4. 金融合规建议。
5. 数学证明细节。

### 3.3 注意力瓶颈

即使人类有能力，也可能没有足够时间和注意力。

长文档、长代码、多工具 trace 都容易让人漏看关键细节。

### 3.4 激励和一致性瓶颈

不同标注员会有不同偏好。

例如：

1. 有人偏好长回答。
2. 有人偏好简洁回答。
3. 有人过度强调安全。
4. 有人过度强调 helpfulness。
5. 有人更容易被自信语气说服。

监督信号本身会有噪声和偏差。

### 3.5 分布外瓶颈

人类标注的数据只能覆盖有限场景。

部署后用户会提出：

1. 新问题。
2. 新攻击。
3. 新工具组合。
4. 新领域。
5. 新语言和新格式。

所以监督不仅要覆盖训练样本，还要考虑泛化。

## 4. Scalable Oversight 方法谱系

可以把主要思路分成五类。

### 4.1 分解复杂任务

把复杂任务拆成很多人类能判断的小任务。

例如：

```text
复杂问题 -> 子问题 1 + 子问题 2 + 子问题 3 -> 汇总答案
```

代表方向：Iterated Amplification、Recursive Reward Modeling。

### 4.2 让 AI 辅助人类判断

用模型帮助人类：

1. 找证据。
2. 总结长文档。
3. 标出潜在错误。
4. 生成反例。
5. 运行工具验证。

代表方向：AI-assisted evaluation、model-written critique。

### 4.3 让 AI 互相博弈或辩论

让两个模型围绕答案进行辩论，人类判断哪一方更可信。

代表方向：AI Safety via Debate。

### 4.4 用原则替代大量标签

用一组原则指导模型自我批评、修正和偏好判断。

代表方向：Constitutional AI、RLAIF。

### 4.5 训练模型表达不确定性

让模型知道自己知道什么、不知道什么。

代表方向：self-evaluation、calibration、abstention。

例如 “Language Models (Mostly) Know What They Know” 探索了模型评估自己答案正确概率和是否知道答案的能力。

### 4.6 把监督可靠性写成可测指标

Scalable oversight 可以抽象成“复杂任务的监督信号是否可靠”的度量问题。指标
必须同时记录覆盖、错误、人工升级、证据支持和成本；单独提高覆盖率，可能只是
把未经校准的错误扩散得更快。

设第 `i` 个监督样本为：

```math
o_i=(x_i,y_i,g_i,h_i,q_i,a_i,v_i,c_i,w_i)
```

其中 `x_i` 是任务输入，`y_i` 是模型输出，`g_i` 是高质量 gold label 或专家复核结果，
`h_i` 是人类对该样本的标签，`q_i` 是人类对自己判断的置信度或“是否适合直接审查”的
分数，`a_i` 是 AI feedback 或 judge 判断，`v_i` 是工具 / verifier 判断，`c_i` 是复杂度
或风险类别，`w_i` 是样本权重。`h_i` 和 `q_i` 必须分开：一个人可以给出“错误”这个
标签，同时把自己的把握标为 0.45；如果把标签当成置信度，覆盖率和错误率就会混为一谈。

下面的公式默认 `w_i>0`。令 `\mathcal{G}` 表示有独立 gold label 或专家复核结果的校准集，
`1[condition]` 表示条件成立时取 1，否则取 0；若分母为 0，指标应记为“未定义”或
`N/A`，不能用 0 或 `max(1, denominator)` 把“没有被测量”伪装成“没有错误”。gold set
也不能只挑容易样本：它应按任务类型、风险等级、长度和模型版本分层抽取，并尽量保留
不参与提示词调优的 holdout 部分。

人类直接监督覆盖率：

```math
C_{\mathrm{direct}}=\frac{\sum_i w_i 1[q_i\ge \tau_{\mathrm{human}}]}{\sum_i w_i}
```

这里的 `q_i` 可以来自标注员的置信度，也可以来自预先定义的“信息是否足够、时间是否
足够、是否需要专家”的审查资格量表；`\tau_{human}` 是项目设定的最低分数。复杂代码、
长文档、专业领域和长工具 trace 会让覆盖率下降。这个指标只回答“多少样本被认为可以
直接审查”，不回答这些判断是否正确；如果没有任何达到阈值的样本，应报告未覆盖。

在人类直接覆盖的样本中，人类标签相对于 gold set 的错误率为：

```math
E_{\mathrm{direct|covered}}=\frac{\sum_{i\in\mathcal{G}} w_i 1[q_i\ge\tau_{\mathrm{human}}]1[h_i\ne g_i]}{\sum_{i\in\mathcal{G}} w_i 1[q_i\ge\tau_{\mathrm{human}}]}
```

它衡量的是“在声称自己能够直接审查的样本上，人类判断是否跟得上任务复杂度”。如果
要诊断低置信度标签本身的风险，还可以另报全量已产出人工标签错误率；但不能把未标注、
弃答或未进入 gold set 的样本当作正确。

AI feedback 校准错误率：

```math
E_{\mathrm{ai}}=\frac{\sum_{i\in\mathcal{G}} w_i 1[a_i\ne g_i]}{\sum_{i\in\mathcal{G}} w_i}
```

如果 `a_i` 是概率分数而不是离散标签，还要先声明分类阈值，并另外检查校准误差；不能
把“判断错误率”直接称为概率校准。AI feedback 成本低、覆盖广，但必须用独立人工 gold
set 或专家复核校准，否则模型可能把自身偏差扩展成更大规模的伪监督。

工具 / verifier 覆盖率：

```math
C_{\mathrm{ver}}=\frac{\sum_i w_i 1[v_i\ne \varnothing]}{\sum_i w_i}
```

代码单元测试、数学答案检查、检索证据核验、policy checker 和工具权限校验都可以看作
verifier。`v_i\ne\varnothing` 只表示 verifier 产出了结果，不表示结果正确；因此还应在
`\mathcal{G}` 上报告 verifier 的错误率或漏检率。verifier 通常比纯自然语言 judge 更可
审计，但覆盖范围有限，无法验证它没有定义的性质。

过程监督准确率：

```math
A_{\mathrm{proc}}=\frac{\sum_{i\in\mathcal{G}}w_i\sum_{j=1}^{n_i}1[s_{ij}=g_{ij}]}{\sum_{i\in\mathcal{G}}w_i n_i}
```

其中 `s_ij` 是第 `i` 个样本第 `j` 个中间步骤的监督判断，`g_ij` 是该步骤的 gold label，
`n_i` 是步骤数。上式采用“按步骤计数”的 micro 平均，长 trace 会贡献更多步骤；如果
希望每个样本权重相同，应另报 macro 平均并明确公式。过程监督适合数学、代码、规划和
Agent trace，但标注成本更高；若 `\sum_{i\in\mathcal{G}}w_i n_i=0`，该指标为 `N/A`。

证据支持率：

```math
S_{\mathrm{evidence}}=\frac{\sum_i m_i^{\mathrm{supported}}}{\sum_i m_i^{\mathrm{claim}}}
```

其中 `m_i^{claim}` 是被纳入评估的 claim 数量，`m_i^{supported}` 是被证据充分支持的
数量；只有在 claim 抽取规则已经固定、且 `\sum_i m_i^{claim}>0` 时才解释这个比率。
它适合 RAG、长文档 QA 和专业建议场景，但“证据支持”不等于“结论完整、因果关系成立”
或“建议适合用户”。监督系统仍需检查引用范围、时间条件、反例和 unsupported claim。

人工升级覆盖率：

```math
C_{\mathrm{audit}}=\frac{\sum_i w_i 1[r_i^{\mathrm{high}}=1]1[b_i^{\mathrm{audit}}=1]}{\sum_i w_i 1[r_i^{\mathrm{high}}=1]}
```

其中 `r_i^{high}=1` 表示在审查前按规则识别为高风险样本，`b_i^{audit}=1` 表示已经
进入人工或专家复核。分母为 0 时应报告“本批没有高风险样本”或检查风险分类是否失效，
不能报告 100%。AI feedback 可以扩展规模，但高风险样本不能完全无人审计；还要防止把
人工升级集中在容易识别的高风险样本上，而漏掉低显著度但高后果的样本。

监督成本节省率：

```math
R_{\mathrm{cost}}=1-\frac{\sum_i k_i^{\mathrm{mixed}}}{\sum_i k_i^{\mathrm{human}}}
```

其中 `k_i^{human}` 是在同一批样本、同一质量要求下全人工专家审查的基线成本，
`k_i^{mixed}` 是 AI 辅助、工具验证、必要人审、返工和基础设施共同构成的混合成本。
两者都必须大于 0，并且覆盖相同任务范围；否则比率没有可比性。成本节省可能为负，
这意味着混合流程更贵，但如果它显著降低高严重度错误，仍可能是合理选择。成本节省
必须和错误率、漏检率及事故预期损失一起看，不能只追求便宜。

对于需要“通过后才算有效”的流程，还可以报告单位成功任务成本：

```math
K_{\mathrm{success}}=\frac{K_{\mathrm{total}}}{N_{\mathrm{accepted}}}
```

其中 `N_{accepted}` 是通过人工、verifier 和风险规则后真正被接受的样本数。若没有
任何成功样本，`K_success` 为 `N/A`，而不是把分母替换成 1；否则一个“零成功”的方案
会看起来拥有一个虚假的有限成本。

一次监督方案比较可以把关键结果写成一组独立约束：

```math
\mathcal{C}_{\mathrm{over}}=\{
E_{\mathrm{ai}}\leq t_a,
A_{\mathrm{proc}}\geq t_p,
S_{\mathrm{evidence}}\geq t_e,
C_{\mathrm{audit}}\geq t_c,
R_{\mathrm{cost}}\geq t_k
\}
```

其中阈值 `t_a`、`t_p`、`t_e`、`t_c` 和 `t_k` 应按风险和业务约束设定，且只有在相应
分母有效时才比较。这个集合不是不可诊断的总开关：每个条件都必须能回指到 gold set、
工具 trace 或人工复核记录。Scalable oversight 的目标也不是让 AI 自己给自己打分，
而是把人类原则、AI 辅助、工具验证和人工复核组织成可量化、可追溯的监督闭环。

## 5. Iterated Amplification

### 5.1 来龙去脉

Iterated Amplification 的提出背景是：很多真实任务目标很复杂，人类直接写 reward 很难，人类直接判断完整答案也很难。

它的思路是：让一个弱专家借助多个模型副本或助手，把复杂问题拆成简单子问题，然后组合答案，形成更强监督信号。

### 5.2 核心直觉

一个人直接解决大问题很难。

但如果他能把问题拆成很多小问题，再调用助手分别回答，最后自己整合，就可能监督更复杂的任务。

简化流程：

```text
Human H + model copies -> amplified overseer Amp(H)
Amp(H) 生成训练信号
训练模型 M 模仿 Amp(H)
新的 M 再帮助 H 形成更强 Amp(H)
循环迭代
```

### 5.3 它解决哪一类监督瓶颈

前人路线：人类直接标注或直接判断模型输出。

问题：任务太复杂时，人类判断不可靠。

Iterated Amplification 试图通过“分解 + 递归辅助”提升人类监督能力。

### 5.4 优点

1. 不依赖手写外部 reward function。
2. 适合可分解任务。
3. 强调逐步构造监督信号。
4. 和复杂推理、长任务监督有天然关系。

### 5.5 缺点

1. 任务不一定容易正确分解。
2. 子问题答案错误会累积。
3. 人类整合仍可能失败。
4. 真实 LLM 任务中的落地成本高。
5. 如果模型助手有系统性偏差，可能放大偏差。

### 5.6 机制与边界

Iterated Amplification 可以看成一种构造 stronger overseer 的方法。

它假设复杂任务存在某种可分解结构，且人类在模型辅助下可以验证或组合子结果。

关键问题包括：

1. Decomposition 是否保真？
2. 子问题之间是否独立？
3. 错误如何传播？
4. 模型辅助是否引入 correlated error？
5. 训练出的模型是否会继承 overseer 的盲点？

如果这些假设不成立，amplification 可能只是把监督偏差放大。

## 6. Debate

### 6.1 来龙去脉

Debate 的背景同样是人类无法直接判断复杂答案。

如果一个复杂答案错了，人类可能看不出来。

但如果另一个智能体指出关键错误，人类可能更容易判断。

### 6.2 核心直觉

让两个模型进行辩论：

1. 一个支持答案 A。
2. 一个指出 A 的问题或支持答案 B。
3. 双方轮流给出论点。
4. 人类评委判断谁更可信。

简化形式：

```text
Question -> Agent A answer / Agent B answer
Agent A and B debate
Human judge chooses winner
Train agents to win by being truthful and exposing flaws
```

### 6.3 它解决哪一类监督瓶颈

直接监督时，人类可能看不出复杂错误。

Debate 希望把“找错误”的工作交给另一个模型，让人类只判断辩论质量。

### 6.4 优点

1. 适合复杂判断。
2. 能主动暴露隐藏错误。
3. 可以用于事实核查、代码审查、长推理评估。
4. 有助于减少单模型自说自话。

### 6.5 缺点

1. 辩论模型可能学会说服而不是真实。
2. 人类可能被修辞而不是证据影响。
3. 辩论成本高。
4. 多轮辩论规则难设计。
5. 两个模型可能共享同样盲点。

### 6.6 机制与边界

Debate 的关键假设是：对复杂问题，错误答案存在某种短而可理解的反驳，人类在看到反驳后能判断。

这个假设并不总成立。

例如：

1. 专业领域中反驳本身也需要专家知识。
2. 错误可能分散在很多细节里。
3. 参与辩论的模型可能选择攻击对方弱点而非追求真相。
4. 人类 judge 的偏好可能被优化和操纵。

因此 Debate 更适合作为监督增强工具，而不是单独解决 alignment 的银弹。

## 7. Recursive Reward Modeling

### 7.1 核心思想

Recursive Reward Modeling 的思路是：复杂任务的 reward 很难直接建模，可以先训练模型完成子任务，再用这些模型帮助构造更复杂任务的 reward。

简化理解：

```text
先学会评估小问题
再用小问题评估器帮助评估大问题
递归构造复杂监督信号
```

### 7.2 和 Iterated Amplification 的关系

二者都强调递归分解。

区别可以这样理解：

1. Iterated Amplification 更强调人类加模型助手形成更强 overseer。
2. Recursive Reward Modeling 更强调递归构造 reward model 或评估器。

概念边界上不必死记形式定义，重点是理解：它们都试图让监督信号随任务复杂度扩展，
但依赖的分解、评估器和人工校准方式不同。

### 7.3 风险

1. 子 reward model 错误会累积。
2. 复杂目标拆分后可能丢失整体约束。
3. Reward model 本身可能被 hack。
4. 人类很难验证最终递归系统是否忠实。

## 8. Constitutional AI 和 RLAIF

### 8.1 来龙去脉

RLHF 依赖大量人类偏好标签。

问题是：

1. 人类标签成本高。
2. 标注员偏好不一致。
3. 安全边界样本难覆盖。
4. 有害内容标注对人类有心理负担。

Constitutional AI 的思路是：用一组人类写下的原则或规则，让模型根据这些原则进行自我批评、自我修正和偏好判断。

### 8.2 核心流程

公开论文中的高层流程可以理解为两阶段。

第一阶段：监督学习式自我修正。

```text
模型生成初始回答
根据 constitution 生成 critique
根据 critique 生成 revised answer
用 revised answer 做监督微调
```

第二阶段：AI feedback 偏好优化。

```text
模型生成两个回答
另一个模型根据 constitution 判断哪个更好
用 AI preference 训练 preference model
再用 RL 优化模型
```

这也常被称为 RLAIF：Reinforcement Learning from AI Feedback。

### 8.3 它解决哪一类监督瓶颈

相比 RLHF，Constitutional AI 试图减少对大量人工有害样本标注的依赖。

它把一部分监督从“人直接逐条判断”变成“人制定原则，AI 根据原则扩展监督”。

### 8.4 优点

1. 降低人类标注成本。
2. 减少人类接触有害内容。
3. 原则更透明。
4. 可以更一致地生成 critique 和 revision。
5. 有助于训练 non-evasive harmless assistant，即安全但不机械回避。

### 8.5 缺点

1. Constitution 本身可能不完整或冲突。
2. AI feedback 可能继承模型偏差。
3. 模型可能学会迎合原则表述。
4. 难处理复杂价值冲突。
5. 对原则解释能力和 judge 能力依赖很强。

### 8.6 机制与边界

Constitutional AI 的关键不是“AI 自己管自己”这么简单。

人类仍然提供了：

1. 原则集合。
2. 训练流程设计。
3. 模型选择。
4. 评估和红队。
5. 最终上线边界。

所以更准确的说法是：它把逐样本监督扩展为原则驱动的监督生成。

风险在于，原则到具体判断之间仍然需要解释，而解释过程由模型完成。

这就要求评估 AI feedback 的偏差、稳定性和可审计性。

## 9. AI Feedback 和 Human Feedback

### 9.1 Human Feedback 的优势

1. 直接来自人类偏好。
2. 能反映真实用户体验。
3. 对价值判断更有合法性。
4. 可用于校准 AI judge。

### 9.2 Human Feedback 的局限

1. 成本高。
2. 速度慢。
3. 一致性不足。
4. 专业能力有限。
5. 难覆盖长尾场景。

### 9.3 AI Feedback 的优势

1. 成本低。
2. 规模大。
3. 速度快。
4. 可用于生成 critique、revision、preference。
5. 能辅助人类处理长上下文和复杂任务。

### 9.4 AI Feedback 的风险

1. 放大模型偏差。
2. 自我强化错误。
3. 被模型输出操纵。
4. 对新任务校准差。
5. 失去人类价值锚点。

### 9.5 实战取舍

更稳妥的路线通常是混合监督：

```text
human principles + human gold labels + AI critique + AI preference + human audit
```

也就是说，AI feedback 用于扩展规模，人类反馈用于定义方向和校准边界。

## 10. Self-Evaluation 和不确定性表达

Scalable oversight 还包括让模型帮助判断自己的输出是否可靠。

例如让模型回答：

1. 这个答案是否正确？
2. 我是否知道这个问题？
3. 这个引用是否支持结论？
4. 哪一步推理最可能出错？

模型自我评估可以用于：

1. 拒答或澄清。
2. 触发检索。
3. 触发工具验证。
4. 触发人工审核。
5. 生成 error analysis。

但不能无条件相信。

模型可能：

1. 对错误答案过度自信。
2. 在新任务上校准差。
3. 给出看似合理的自我解释。
4. 受 prompt 格式影响。

## 11. 真实项目如何落地 Scalable Oversight

### 11.1 RAG 场景

可以让模型辅助检查：

1. 检索文档是否相关。
2. 回答中的 claim 是否有证据支持。
3. 引用是否精确。
4. 是否存在 unsupported claim。
5. 是否需要拒答。

但高风险样本要人工复核。

### 11.2 Code 场景

可以结合：

1. 单元测试。
2. 静态分析。
3. LLM code review。
4. 多模型辩论。
5. 人工审核关键补丁。

模型负责扩大覆盖，人类负责关键决策。

### 11.3 Safety 场景

可以使用：

1. AI 生成红队样本。
2. AI 生成 critique。
3. AI 根据 policy 初筛风险。
4. 人工审核 P0/P1 高风险样本。
5. 将确认问题加入 regression suite。

### 11.4 Agent 场景

对 Agent，监督不只看最终答案。

还要看：

1. 计划是否合理。
2. 工具选择是否正确。
3. 参数是否安全。
4. Observation 是否被正确理解。
5. 是否越权。
6. 是否需要用户确认。

Scalable oversight 可以把长 trace 拆成可审核片段。

### 11.5 最小可运行监督覆盖审计 demo

下面这个 demo 不依赖外部库，也不读写文件。输入是一组抽象 toy oversight case，只有任务类别、gold label、人类直接判断、AI feedback、verifier / tool 判断、debate 判断、过程步骤、证据支持、高风险标记、审计标记和成本，不包含任何可复用攻击提示或危险操作细节。

它演示的是监督闭环审计：人类直接判断覆盖是否不足，AI feedback 是否被 gold set 校准，工具验证覆盖了多少复杂样本，高风险样本是否进入人审，以及混合监督是否真的降低成本但不放大错误。真实系统还需要专家标注规范、完整 trace、评估平台、权限日志、模型版本管理和上线后监控。

```python
from collections import Counter, defaultdict


cases = [
    {"id": "rag_long_context", "slice": "rag", "gold": True, "human_label": True, "human_conf": 0.55, "ai_label": True, "verifier_label": True, "debate_label": None, "process_ok": 4, "process_total": 5, "claims_supported": 4, "claims_total": 5, "high_risk": False, "human_audit": False, "severity": 2, "human_cost": 40, "mixed_cost": 9},
    {"id": "code_patch_security", "slice": "code", "gold": False, "human_label": True, "human_conf": 0.42, "ai_label": False, "verifier_label": False, "debate_label": False, "process_ok": 5, "process_total": 6, "claims_supported": 0, "claims_total": 0, "high_risk": True, "human_audit": True, "severity": 5, "human_cost": 60, "mixed_cost": 18},
    {"id": "math_proof", "slice": "math", "gold": False, "human_label": True, "human_conf": 0.60, "ai_label": True, "verifier_label": False, "debate_label": False, "process_ok": 3, "process_total": 5, "claims_supported": 0, "claims_total": 0, "high_risk": False, "human_audit": False, "severity": 3, "human_cost": 45, "mixed_cost": 10},
    {"id": "medical_summary", "slice": "high_risk_domain", "gold": False, "human_label": True, "human_conf": 0.70, "ai_label": True, "verifier_label": None, "debate_label": None, "process_ok": 2, "process_total": 4, "claims_supported": 2, "claims_total": 4, "high_risk": True, "human_audit": False, "severity": 5, "human_cost": 80, "mixed_cost": 14},
    {"id": "simple_faq", "slice": "normal_help", "gold": True, "human_label": True, "human_conf": 0.92, "ai_label": True, "verifier_label": None, "debate_label": None, "process_ok": 1, "process_total": 1, "claims_supported": 2, "claims_total": 2, "high_risk": False, "human_audit": False, "severity": 1, "human_cost": 8, "mixed_cost": 3},
    {"id": "agent_tool_trace", "slice": "agent", "gold": False, "human_label": False, "human_conf": 0.50, "ai_label": False, "verifier_label": False, "debate_label": None, "process_ok": 4, "process_total": 6, "claims_supported": 0, "claims_total": 0, "high_risk": True, "human_audit": True, "severity": 5, "human_cost": 70, "mixed_cost": 20},
    {"id": "legal_contract", "slice": "high_risk_domain", "gold": False, "human_label": True, "human_conf": 0.45, "ai_label": False, "verifier_label": None, "debate_label": False, "process_ok": 3, "process_total": 4, "claims_supported": 3, "claims_total": 4, "high_risk": True, "human_audit": True, "severity": 4, "human_cost": 90, "mixed_cost": 25},
    {"id": "summary_grounded", "slice": "summarization", "gold": True, "human_label": True, "human_conf": 0.82, "ai_label": True, "verifier_label": True, "debate_label": None, "process_ok": 2, "process_total": 2, "claims_supported": 3, "claims_total": 3, "high_risk": False, "human_audit": False, "severity": 2, "human_cost": 25, "mixed_cost": 8},
    {"id": "policy_boundary", "slice": "safety_boundary", "gold": True, "human_label": False, "human_conf": 0.68, "ai_label": False, "verifier_label": True, "debate_label": True, "process_ok": 2, "process_total": 3, "claims_supported": 1, "claims_total": 1, "high_risk": True, "human_audit": True, "severity": 3, "human_cost": 35, "mixed_cost": 13},
    {"id": "unsupported_research_claim", "slice": "research", "gold": False, "human_label": True, "human_conf": 0.73, "ai_label": False, "verifier_label": False, "debate_label": None, "process_ok": 2, "process_total": 3, "claims_supported": 2, "claims_total": 5, "high_risk": False, "human_audit": False, "severity": 3, "human_cost": 50, "mixed_cost": 12},
]


def majority_label(case):
    votes = [case["ai_label"]]
    for key in ("verifier_label", "debate_label"):
        if case[key] is not None:
            votes.append(case[key])
    positives = sum(1 for vote in votes if vote is True)
    negatives = sum(1 for vote in votes if vote is False)
    if positives == negatives:
        return case["ai_label"]
    return positives > negatives


human_threshold = 0.75
direct_covered = [case for case in cases if case["human_conf"] >= human_threshold]
verifier_cases = [case for case in cases if case["verifier_label"] is not None]
high_risk_cases = [case for case in cases if case["high_risk"]]

oversight_errors = []
slice_errors = defaultdict(list)
severity_error = 0
total_severity = sum(case["severity"] for case in cases)
total_process = sum(case["process_total"] for case in cases)
total_claims = sum(case["claims_total"] for case in cases)
total_human_cost = sum(case["human_cost"] for case in cases)
total_mixed_cost = sum(case["mixed_cost"] for case in cases)

oversight_labels = {}
for case in cases:
    label = majority_label(case)
    oversight_labels[case["id"]] = label
    if label != case["gold"]:
        oversight_errors.append(case["id"])
        slice_errors[case["slice"]].append(case["id"])
        severity_error += case["severity"]

high_risk_missing_audit = [case["id"] for case in high_risk_cases if not case["human_audit"]]

metrics = {
    "direct_coverage": round(len(direct_covered) / len(cases), 3),
    "direct_accuracy_on_covered": round(
        sum(case["human_label"] == case["gold"] for case in direct_covered) / max(1, len(direct_covered)), 3
    ),
    "human_direct_error_on_covered": round(
        sum(case["human_label"] != case["gold"] for case in direct_covered) / max(1, len(direct_covered)), 3
    ),
    "ai_feedback_error": round(sum(case["ai_label"] != case["gold"] for case in cases) / len(cases), 3),
    "verifier_coverage": round(len(verifier_cases) / len(cases), 3),
    "oversight_accuracy": round(sum(oversight_labels[case["id"]] == case["gold"] for case in cases) / len(cases), 3),
    "process_step_accuracy": round(sum(case["process_ok"] for case in cases) / total_process, 3),
    "evidence_support": round(sum(case["claims_supported"] for case in cases) / max(1, total_claims), 3),
    "high_risk_audit_coverage": round(
        sum(case["human_audit"] for case in high_risk_cases) / max(1, len(high_risk_cases)), 3
    ),
    "cost_saving": round(1 - total_mixed_cost / total_human_cost, 3),
    "severity_weighted_error": round(severity_error / total_severity, 3),
}

thresholds = {
    "ai_feedback_error": 0.25,
    "process_step_accuracy": 0.80,
    "evidence_support": 0.75,
    "high_risk_audit_coverage": 1.00,
    "severity_weighted_error": 0.10,
    "cost_saving": 0.50,
}

actions = []
if metrics["ai_feedback_error"] > thresholds["ai_feedback_error"]:
    actions.append("扩大人工 gold set，校准 AI feedback 的错误切片")
if metrics["process_step_accuracy"] < thresholds["process_step_accuracy"]:
    actions.append("补充中间步骤标注和可执行 verifier")
if metrics["evidence_support"] < thresholds["evidence_support"]:
    actions.append("复核 claim 与证据的支持关系，禁止只看流畅度")
if metrics["high_risk_audit_coverage"] < thresholds["high_risk_audit_coverage"]:
    actions.append("把未审计的高风险样本转人工或专家复核")
if metrics["severity_weighted_error"] > thresholds["severity_weighted_error"]:
    actions.append("按严重度重排监督预算，优先修复高影响错误")
if metrics["cost_saving"] < thresholds["cost_saving"]:
    actions.append("比较全人工、混合监督和单位成功成本")

if high_risk_missing_audit or metrics["severity_weighted_error"] > thresholds["severity_weighted_error"]:
    decision = "hold_for_high_risk_audit"
elif metrics["ai_feedback_error"] > thresholds["ai_feedback_error"]:
    decision = "calibrate_feedback_before_expansion"
elif metrics["process_step_accuracy"] < thresholds["process_step_accuracy"]:
    decision = "improve_process_supervision_before_expansion"
else:
    decision = "continue_bounded_oversight_trial"

report = {
    "slice_counts": dict(sorted(Counter(case["slice"] for case in cases).items())),
    "metrics": metrics,
    "oversight_errors": oversight_errors,
    "high_risk_missing_audit": high_risk_missing_audit,
    "slice_errors": dict(sorted(slice_errors.items())),
    "thresholds": thresholds,
    "actions": actions,
    "decision": decision,
}

for key, value in report.items():
    print(f"{key}=", value)

assert report["metrics"] == {
    "direct_coverage": 0.2,
    "direct_accuracy_on_covered": 1.0,
    "human_direct_error_on_covered": 0.0,
    "ai_feedback_error": 0.3,
    "verifier_coverage": 0.7,
    "oversight_accuracy": 0.9,
    "process_step_accuracy": 0.718,
    "evidence_support": 0.708,
    "high_risk_audit_coverage": 0.8,
    "cost_saving": 0.738,
    "severity_weighted_error": 0.152,
}
assert report["oversight_errors"] == ["medical_summary"]
assert report["high_risk_missing_audit"] == ["medical_summary"]
assert report["decision"] == "hold_for_high_risk_audit"
```

运行后会看到类似输出：

```text
slice_counts= {'agent': 1, 'code': 1, 'high_risk_domain': 2, 'math': 1, 'normal_help': 1, 'rag': 1, 'research': 1, 'safety_boundary': 1, 'summarization': 1}
metrics= {'direct_coverage': 0.2, 'direct_accuracy_on_covered': 1.0, 'human_direct_error_on_covered': 0.0, 'ai_feedback_error': 0.3, 'verifier_coverage': 0.7, 'oversight_accuracy': 0.9, 'process_step_accuracy': 0.718, 'evidence_support': 0.708, 'high_risk_audit_coverage': 0.8, 'cost_saving': 0.738, 'severity_weighted_error': 0.152}
oversight_errors= ['medical_summary']
high_risk_missing_audit= ['medical_summary']
slice_errors= {'high_risk_domain': ['medical_summary']}
thresholds= {'ai_feedback_error': 0.25, 'process_step_accuracy': 0.8, 'evidence_support': 0.75, 'high_risk_audit_coverage': 1.0, 'severity_weighted_error': 0.1, 'cost_saving': 0.5}
actions= ['扩大人工 gold set，校准 AI feedback 的错误切片', '补充中间步骤标注和可执行 verifier', '复核 claim 与证据的支持关系，禁止只看流畅度', '把未审计的高风险样本转人工或专家复核', '按严重度重排监督预算，优先修复高影响错误']
decision= hold_for_high_risk_audit
```

这个 demo 的重点是：混合监督可以显著省成本，`oversight_accuracy` 也可能看起来不错，
但只要高风险样本没有人审、证据支持不足或 AI feedback 未校准，就不能把监督结果
写成普遍可靠。程序保留每个信号、动作和决定，读者可以进一步追查 `medical_summary`
为什么既造成监督错误，又没有进入人工审计。

## 12. Scalable Oversight 的优缺点

### 12.1 优点

1. 缓解人工标注成本。
2. 帮助监督复杂任务。
3. 提高长上下文和多步任务可审查性。
4. 可以结合工具验证和模型 critique。
5. 适合生成 safety eval 和 regression case。

### 12.2 缺点

1. AI feedback 可能放大错误。
2. 复杂任务分解可能丢失整体目标。
3. 人类仍需要校准和审计。
4. 多模型系统成本和复杂度更高。
5. 可能产生“监督看起来更强，但实际更难验证”的错觉。

### 12.3 适用场景

适合：

1. 长文档 QA。
2. RAG 忠实性评估。
3. 代码审查。
4. 数学和推理步骤检查。
5. 安全红队。
6. Agent 工具 trace 审计。

不适合作为唯一手段：

1. 高风险医疗法律结论。
2. 需要现实世界责任判断的决策。
3. 模型和 judge 同源且没有外部验证的场景。

## 13. 和其他概念的关系

### 13.1 和 RLHF 的关系

RLHF 是用人类反馈提供监督。

Scalable oversight 研究的是当人类反馈本身不够强时，如何增强监督。

可以说：

```text
RLHF 是基础监督路线，scalable oversight 是监督能力扩展路线。
```

### 13.2 和 LLM-as-a-Judge 的关系

LLM-as-a-Judge 是 AI feedback 的一种工程形式。

但 scalable oversight 更广，还包括分解、辩论、递归监督、原则驱动监督和人机协作。

### 13.3 和 interpretability 的关系

Interpretability 希望直接理解模型内部。

Scalable oversight 更多关注如何构造外部监督信号。

二者互补。

### 13.4 和 red teaming 的关系

Red teaming 可以发现失败样本。

Scalable oversight 可以帮助生成、筛选、归因和复核这些失败样本。

## 14. 案例：长文档研究助手的混合监督

设想一个研究助手需要阅读几十篇论文，回答“某种方法在什么条件下有效”，并给出
逐条引用。最终答案很长，且判断依赖实验设置、样本范围和统计方法。让普通标注员
直接比较两份答案，会遇到四个问题：他们可能看不完所有论文，可能无法复核实验，
可能被流畅表达影响，也可能没有足够时间判断每条引用是否支持结论。

### 14.1 把任务分成不同难度的监督单元

可以把最终任务拆成：

1. 文档是否与问题相关。
2. 每条 claim 是否能在指定段落中找到证据。
3. 引用是否真的支持 claim，而不是只与主题相关。
4. 计算或实验条件是否被正确转述。
5. 多条证据合并后的结论是否超出了资料支持范围。

前四项可以分别交给检索器、规则、verifier 或 AI critique 辅助处理，第五项通常
需要专家抽样复核。分解降低了单次审查负担，但不能假设子任务完全独立；最终结论
仍要保留全局约束和原始引用。

### 14.2 让 AI 做扩展，让人类校准

第一步用少量专家 gold set 校准 AI judge：专家不仅标答案对错，还标证据覆盖、
因果过度推断和引用错配。第二步让 AI 对大批样本提取 claims、标记可疑段落并
生成反例。第三步使用程序检查链接、页码和数值，最后把高风险、低置信度和多评估
器分歧的样本送回专家。

关键不是把人类从流程中删除，而是把专家时间集中到 AI 最不可靠、后果最严重的
切片。人工升级覆盖率、AI 与 gold set 的错误率、证据支持率和单位成功任务成本
要一起记录。

### 14.3 观察监督系统是否在自我强化错误

如果 AI judge 主要依据写作流畅度打分，它可能把同一套表面偏差扩展到数万样本。
因此 holdout gold set 不能参与提示词调优，且应定期加入新的领域和时间切片。若
AI judge 与专家在同一类 claim 上持续分歧，就应暂停自动扩展，检查 rubric、引用
解析器和任务分解，而不是继续增加样本量。

类似地，debate 只有在辩手能提出可验证证据、人类能理解关键反驳时才有帮助；如果
双方共享同一错误，或评委只偏好更自信的表达，辩论轮数越多不一定越可靠。

### 14.4 从结果映射到动作

一个可审计的监督报告可以保存：

~~~text
signals: AI 与 gold set 的一致性、过程准确率、证据支持率、人工升级覆盖、成本
actions: 校准 judge、增加 verifier、补充专家样本、限制自动扩展、复核高风险轨迹
decision: 继续受限扩展、保持混合监督、暂停自动标注或回退旧流程
~~~

这种结构把 scalable oversight 从一句口号变成了可以观察的系统。它也保留了一个
重要事实：监督扩展的成功不是“AI 标了多少数据”，而是单位成本下增加了多少可信
监督，并且没有把高严重度错误藏在平均数里。

## 15. 常见误区

### 15.1 误区：Scalable oversight 只是降低标注成本

成本只是表层问题，核心是复杂任务中人类监督能力不足。若只看每条标注的价格，
可能忽略错误监督带来的训练和部署损失。

### 15.2 误区：AI feedback 可以完全替代人类

AI feedback 需要人类原则、人工 gold set 和高风险审核校准，且要定期检查新任务
和新模型上的误差变化。

### 15.3 误区：Debate 一定能得到真相

Debate 依赖人类 judge 能判断论点，也可能变成说服力竞赛。辩论轮数增加并不自动
增加事实性。

### 15.4 误区：任务总能无损分解

很多任务有全局约束，拆分后可能丢失整体目标，因此需要保留最终状态和全局约束的
独立验证。

### 15.5 误区：模型会自我评估就可信

自我评估也需要校准、验证和分布外测试，不能因为模型说“我有把握”就把它当成事实。

## 16. 资料与证据边界

本章的资料可以分为监督方法论文、训练与评估论文、官方规范和治理框架。方法论文
支持某种监督路线的设计与实验条件，不能直接证明方法在其他模型、任务或风险等级
上有效；规范和框架支持原则、审计和责任组织，也不能替代具体任务中的 gold set、
工具验证和人工复核。

### 16.1 监督方法

- [AI Safety via Debate](https://arxiv.org/abs/1805.00899)：讨论通过竞争性论证帮助人类监督复杂答案的研究方向；它依赖评委能理解关键反驳。
- [Iterated Amplification](https://arxiv.org/abs/1810.08575)：支持通过人类与模型助手的递归分解构造更强监督者的研究入口。
- [Scalable agent alignment via reward modeling](https://arxiv.org/abs/2210.06794)：讨论用递归或分层 reward modeling 监督更复杂 Agent 行为的研究方向；具体实现仍要校准子评估器。
- [Constitutional AI](https://arxiv.org/abs/2212.08073)：支持原则驱动的 critique、revision 和 AI feedback 流程，不等于 AI feedback 无偏。
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)：说明复杂文本质量可以用人类反馈训练代理评估器，但代理奖励需要独立验证。

### 16.2 过程监督、评估与治理

- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)：支持过程监督和逐步验证在数学推理中的研究入口，不能自动外推到所有 Agent 轨迹。
- [Weak-to-strong generalization](https://arxiv.org/abs/2312.09390)：讨论弱监督者如何监督更强模型的实验框架；结果依赖任务、监督者和强模型的具体设置。
- [OpenAI Model Spec](https://model-spec.openai.com/)：官方行为规范入口，支持分析监督目标和行为边界；规范文本不是独立实测结果。
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)：支持 Govern、Map、Measure、Manage 的治理结构。
- [NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)：支持生成式系统的风险、测量和人工监督活动组织。

引用这些资料时，应区分论文实验、厂商规范、治理建议和本项目实际测量。尤其是
“监督更强”“AI feedback 可靠”或“人类成本下降”等结论，必须同时报告任务切片、
gold set、判定器、人工升级、错误严重度和成本分母；没有这些条件时，只能写成
方法假设或待验证结果。

## 17. 小练习

### 练习 1

用自己的话解释为什么 RLHF 会遇到 scalable oversight 问题。

说明成本、专业能力、长上下文、多步任务和标注偏差如何造成监督瓶颈。

### 练习 2

比较 Iterated Amplification、Debate 和 Constitutional AI。

分别说明它们解决什么问题、依赖的核心假设、优点、风险和可验证证据。

### 练习 3

为一个 RAG 系统设计 AI-assisted evaluation 流程。

覆盖 claim extraction、evidence checking、citation verification、human audit、gold set 和升级规则。

### 练习 4

为一个 coding agent 设计 scalable oversight 流程。

覆盖单元测试、静态分析、LLM review、多模型辩论、权限审计和人工审核。

### 练习 5

讨论 AI feedback 的一个优势和一个危险。

给出一个具体例子，并说明如何用独立 gold set 检查这个风险。

## 18. 本章总结

Scalable Oversight 要解决的是复杂任务中的监督瓶颈：当人类无法直接可靠判断模型输出时，如何仍然构造可靠训练和评估信号。

RLHF 是重要基础，但人类偏好标注存在成本、能力、注意力、一致性和分布外瓶颈。

Iterated Amplification 和 Recursive Reward Modeling 强调递归分解复杂任务。

Debate 强调用模型之间的竞争暴露错误，让人类更容易判断。

Constitutional AI 和 RLAIF 用人类原则驱动 AI feedback，减少逐样本人类标注成本。

AI feedback 可以扩展监督规模，但不能完全替代 human feedback；更可靠的路线是人类原则、人工 gold set、AI critique、工具验证、人审和 regression suite 的混合监督闭环。

Scalable oversight 不是一个单一算法，而是一组让监督能力跟上模型能力增长的方法
谱系；判断它是否有效，必须回到覆盖、错误、升级和成本证据。
