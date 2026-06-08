# 第三章：Scalable Oversight

重点：人类监督瓶颈、AI feedback、Constitutional AI、Debate、Iterated Amplification、Recursive Reward Modeling。

面试重点：Scalable Oversight 的核心问题是：当模型能力超过单个人类直接判断能力时，我们如何仍然给出可靠监督信号？

## 0. 本讲资料边界与第二轮精修口径

按照 `WRITING_PLAN.md` 的要求，本讲精修前核对了 AI Safety via Debate、Iterated Amplification、Scalable agent alignment via reward modeling、Constitutional AI、Learning to Summarize from Human Feedback、Let's Verify Step by Step、weak-to-strong generalization、OpenAI Evals / Model Spec、NIST AI RMF / Generative AI Profile 等公开资料。

本讲定位为 scalable oversight 的方法谱系和工程落地章，重点是回答一个面试高频问题：当任务复杂到单个人类难以直接判断时，怎样仍然构造可校准、可审计、可扩展的监督信号。

```text
复杂任务 -> 监督瓶颈 -> 分解 / AI critique / debate / verifier / 人工审计 -> 监督门禁
```

本讲不把 Debate、Iterated Amplification、Recursive Reward Modeling、Constitutional AI 或 AI feedback 写成已经解决 alignment 的银弹。它们都是监督增强方案，需要人工 gold set、工具验证、高风险人工复核、分布外评估和上线回归共同约束。

## 本章目标

学完本章，你要能回答：

1. 什么是 Scalable Oversight？
2. 为什么 RLHF 和人工标注会遇到监督瓶颈？
3. Debate、Iterated Amplification、Recursive Reward Modeling、Constitutional AI 分别想解决什么问题？
4. AI feedback 和 human feedback 有什么区别？
5. Scalable oversight 和 reward hacking、deceptive alignment、hallucination 有什么关系？
6. 面试中如何评价这些方法的优缺点和适用边界？
7. 如何在真实 LLM 系统中落地“可扩展监督”的思想？

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

### 4.6 关键公式与监督覆盖指标速查

Scalable oversight 可以抽象成“复杂任务监督信号是否可靠”的度量问题。

设第 `i` 个监督样本为：

```math
o_i=(x_i,y_i,g_i,h_i,a_i,v_i,c_i,w_i)
```

其中 `x_i` 是任务输入，`y_i` 是模型输出，`g_i` 是高质量 gold label 或专家复核结果，`h_i` 是单个人类直接判断，`a_i` 是 AI feedback 或 judge 判断，`v_i` 是工具 / verifier 判断，`c_i` 是复杂度或风险类别，`w_i` 是样本权重。

人类直接监督覆盖率：

```math
C_{\mathrm{direct}}=\frac{\sum_i w_i 1[q_i^{\mathrm{human}}\ge \tau_{\mathrm{human}}]}{\sum_i w_i}
```

其中 `q_i_human` 是人类直接判断的置信度或可审查性分数。复杂代码、长文档、专业领域和长工具 trace 会让这个覆盖率下降。

人类直接监督错误率：

```math
E_{\mathrm{direct}}=\frac{\sum_i w_i 1[h_i\ne g_i]}{\sum_i w_i}
```

这个指标衡量“普通标注是否能跟上任务复杂度”。如果人类直接错误率很高，单纯扩大人工标注规模会把错误监督信号也一起放大。

AI feedback 校准错误率：

```math
E_{\mathrm{ai}}=\frac{\sum_i w_i 1[a_i\ne g_i]}{\sum_i w_i}
```

AI feedback 成本低、覆盖广，但必须用人工 gold set 或专家复核校准。否则模型可能把自身偏差扩展成更大规模的伪监督。

工具 / verifier 覆盖率：

```math
C_{\mathrm{ver}}=\frac{\sum_i w_i 1[v_i\ne \varnothing]}{\sum_i w_i}
```

代码单元测试、数学答案检查、检索证据核验、policy checker 和工具权限校验都可以看作 verifier。它们通常比纯自然语言 judge 更可审计，但覆盖范围有限。

过程监督准确率：

```math
A_{\mathrm{proc}}=\frac{\sum_i \sum_j 1[s_{ij}=g_{ij}]}{\sum_i n_i}
```

其中 `s_ij` 是第 `i` 个样本第 `j` 个中间步骤的监督判断，`g_ij` 是该步骤的 gold label，`n_i` 是步骤数。过程监督适合数学、代码、规划和 Agent trace，但标注成本更高。

证据支持率：

```math
S_{\mathrm{evidence}}=\frac{\sum_i m_i^{\mathrm{supported}}}{\sum_i m_i^{\mathrm{claim}}}
```

这个指标适合 RAG、长文档 QA 和专业建议场景。它要求监督系统检查回答中的 claim 是否真的被证据支持，而不是只看回答是否流畅。

人工升级覆盖率：

```math
C_{\mathrm{audit}}=\frac{\sum_i w_i 1[r_i^{\mathrm{high}}=1]1[b_i^{\mathrm{audit}}=1]}{\sum_i w_i 1[r_i^{\mathrm{high}}=1]}
```

其中 `r_i_high=1` 表示高风险样本，`b_i_audit=1` 表示进入人工或专家复核。AI feedback 可以扩展规模，但高风险样本不能完全无人审计。

监督成本节省率：

```math
R_{\mathrm{cost}}=1-\frac{\sum_i k_i^{\mathrm{mixed}}}{\sum_i k_i^{\mathrm{human}}}
```

其中 `k_i_human` 是全人工专家审查成本，`k_i_mixed` 是 AI 辅助 + 工具验证 + 必要人审的混合成本。成本节省必须和监督错误率一起看，不能只追求便宜。

一个简化的 scalable oversight 门禁可以写成：

```math
G_{\mathrm{over}}=G_{\mathrm{direct}}\land G_{\mathrm{ai}}\land G_{\mathrm{ver}}\land G_{\mathrm{proc}}\land G_{\mathrm{audit}}\land G_{\mathrm{cost}}
```

面试中可以强调：scalable oversight 的目标不是让 AI 自己给自己打分，而是把人类原则、AI 辅助、工具验证和人工复核组织成一个可量化的监督闭环。

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

### 5.3 它解决前人什么问题

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

### 5.6 面向专家

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

### 6.3 它解决什么问题

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

### 6.6 面向专家

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

面试中不必死记形式定义，重点讲清：它们都试图让监督信号随任务复杂度扩展。

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

### 8.3 它解决前人什么问题

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

### 8.6 面向专家

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
    "human_direct_error": round(sum(case["human_label"] != case["gold"] for case in cases) / len(cases), 3),
    "ai_feedback_accuracy": round(sum(case["ai_label"] == case["gold"] for case in cases) / len(cases), 3),
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

gates = {
    "ai_feedback": metrics["ai_feedback_accuracy"] >= 0.75,
    "oversight_accuracy": metrics["oversight_accuracy"] >= 0.90,
    "process": metrics["process_step_accuracy"] >= 0.80,
    "evidence": metrics["evidence_support"] >= 0.75,
    "high_risk_audit": metrics["high_risk_audit_coverage"] >= 1.00,
    "severity_error": metrics["severity_weighted_error"] <= 0.10,
    "cost": metrics["cost_saving"] >= 0.50,
}

report = {
    "slice_counts": dict(sorted(Counter(case["slice"] for case in cases).items())),
    "metrics": metrics,
    "oversight_errors": oversight_errors,
    "high_risk_missing_audit": high_risk_missing_audit,
    "slice_errors": dict(sorted(slice_errors.items())),
    "gates": gates,
    "oversight_ready": all(gates.values()),
}

for key, value in report.items():
    print(f"{key}=", value)

assert report["metrics"] == {
    "direct_coverage": 0.2,
    "direct_accuracy_on_covered": 1.0,
    "human_direct_error": 0.6,
    "ai_feedback_accuracy": 0.7,
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
assert report["oversight_ready"] is False
```

运行后会看到类似输出：

```text
slice_counts= {'agent': 1, 'code': 1, 'high_risk_domain': 2, 'math': 1, 'normal_help': 1, 'rag': 1, 'research': 1, 'safety_boundary': 1, 'summarization': 1}
metrics= {'direct_coverage': 0.2, 'direct_accuracy_on_covered': 1.0, 'human_direct_error': 0.6, 'ai_feedback_accuracy': 0.7, 'verifier_coverage': 0.7, 'oversight_accuracy': 0.9, 'process_step_accuracy': 0.718, 'evidence_support': 0.708, 'high_risk_audit_coverage': 0.8, 'cost_saving': 0.738, 'severity_weighted_error': 0.152}
oversight_errors= ['medical_summary']
high_risk_missing_audit= ['medical_summary']
slice_errors= {'high_risk_domain': ['medical_summary']}
gates= {'ai_feedback': False, 'oversight_accuracy': True, 'process': False, 'evidence': False, 'high_risk_audit': False, 'severity_error': False, 'cost': True}
oversight_ready= False
```

这个 demo 的重点是：混合监督可以显著省成本，`oversight_accuracy` 也可能看起来不错，但只要高风险样本没有人审、证据支持不足或 AI feedback 未校准，就不能说 scalable oversight 已经可靠上线。

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

## 14. 面试官会怎么问

### 问题 1：什么是 Scalable Oversight？

回答要点：

1. 当任务复杂到人类难以直接判断时，仍然构造可靠监督信号。
2. 方法包括任务分解、AI 辅助评估、debate、amplification、Constitutional AI。
3. 目标是让监督能力随模型能力增长。

标准回答：

```text
Scalable Oversight 关注的是如何监督越来越强、越来越复杂的 AI 系统。当模型输出的代码、推理、长文档分析或工具调用计划超出单个人类直接判断能力时，我们需要通过任务分解、AI 辅助、辩论、递归监督、原则驱动反馈和人工审计来构造更可靠的训练和评估信号。
```

### 问题 2：为什么 RLHF 不够？

回答要点：

1. RLHF 依赖人类偏好判断。
2. 人类可能看不懂复杂任务。
3. 偏好标注有成本、偏差和一致性问题。
4. 模型可能学会迎合标注员而不是真实解决问题。
5. 需要 AI-assisted oversight 和更强评估。

### 问题 3：Constitutional AI 解决什么问题？

回答要点：

1. 用人类写下的原则指导模型自我批评、修正和偏好判断。
2. 减少对逐样本人类有害内容标注的依赖。
3. 可以训练 harmless but non-evasive assistant。
4. 风险是原则不完整、AI feedback 继承偏差、需要人工校准。

### 问题 4：Debate 有什么优缺点？

回答要点：

1. 优点是让模型互相指出复杂错误，降低人类直接判断难度。
2. 缺点是模型可能优化说服力而不是真实性。
3. 人类 judge 仍可能被误导。
4. 辩论规则和评估成本很关键。

### 问题 5：AI feedback 能否替代 human feedback？

回答要点：

1. 不能完全替代。
2. AI feedback 可扩规模、降成本、做 critique 和初筛。
3. Human feedback 提供价值锚点和校准。
4. 高风险场景需要人工或专家复核。

## 15. 标准回答模板

面试中可以这样回答：

```text
Scalable oversight 的核心是监督瓶颈。RLHF 假设人类可以判断模型输出好坏，但当任务变成长代码审查、复杂法律分析、多步数学证明、RAG 长文档综合或 Agent 工具调用时，单个人类很难直接判断。

解决思路包括几类：第一，把复杂任务拆成子任务，例如 iterated amplification 和 recursive reward modeling；第二，让 AI 辅助人类找证据、做 critique、生成反例；第三，让模型之间辩论，让人类判断谁指出了关键问题；第四，用 constitution 这类人类原则生成 AI feedback，减少逐样本人类标注成本。

这些方法的共同目标是让监督能力随模型能力扩展。但它们不是银弹，因为 AI feedback 可能放大模型偏差，任务分解可能丢失整体目标，debate 可能优化说服力而不是真实性。因此真实系统中应该采用 human gold labels、AI-assisted eval、工具验证、红队、人工审核和 regression suite 的混合监督闭环。
```

## 16. 常见误区

### 16.1 误区：Scalable oversight 只是降低标注成本

纠正：成本只是表层问题，核心是复杂任务中人类监督能力不足。

### 16.2 误区：AI feedback 可以完全替代人类

纠正：AI feedback 需要人类原则、人工 gold set 和高风险审核校准。

### 16.3 误区：Debate 一定能得到真相

纠正：Debate 依赖人类 judge 能判断论点，也可能变成说服力竞赛。

### 16.4 误区：任务总能无损分解

纠正：很多任务有全局约束，拆分后可能丢失整体目标。

### 16.5 误区：模型会自我评估就可信

纠正：自我评估也需要校准、验证和分布外测试。

## 17. 小练习

### 练习 1

用自己的话解释为什么 RLHF 会遇到 scalable oversight 问题。

要求包含：成本、专业能力、长上下文、多步任务和标注偏差。

### 练习 2

比较 Iterated Amplification、Debate 和 Constitutional AI。

要求说明：它们分别解决什么问题、核心假设、优点和风险。

### 练习 3

为一个 RAG 系统设计 AI-assisted evaluation 流程。

要求覆盖：claim extraction、evidence checking、citation verification、human audit。

### 练习 4

为一个 coding agent 设计 scalable oversight 流程。

要求覆盖：单元测试、静态分析、LLM review、多模型辩论和人工审核。

### 练习 5

讨论 AI feedback 的一个优势和一个危险。

要求给出具体例子。

## 18. 本章总结

Scalable Oversight 要解决的是复杂任务中的监督瓶颈：当人类无法直接可靠判断模型输出时，如何仍然构造可靠训练和评估信号。

RLHF 是重要基础，但人类偏好标注存在成本、能力、注意力、一致性和分布外瓶颈。

Iterated Amplification 和 Recursive Reward Modeling 强调递归分解复杂任务。

Debate 强调用模型之间的竞争暴露错误，让人类更容易判断。

Constitutional AI 和 RLAIF 用人类原则驱动 AI feedback，减少逐样本人类标注成本。

AI feedback 可以扩展监督规模，但不能完全替代 human feedback；更可靠的路线是人类原则、人工 gold set、AI critique、工具验证、人审和 regression suite 的混合监督闭环。

面试中要强调：Scalable oversight 不是一个单一算法，而是一组让监督能力跟上模型能力增长的方法谱系。
