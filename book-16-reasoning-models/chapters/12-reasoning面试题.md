# 第十二章：Reasoning 综合复习：从候选生成到受控系统

前十一章分别讨论了推理轨迹、Chain-of-Thought、self-consistency、verifier、过程监督、搜索、推理时计算、数学训练、代码执行反馈、评估和安全。把这些主题分开学习有助于建立概念，但真实系统不会把它们分成互不相干的章节：一次复杂请求可能同时需要生成多个候选、检查中间步骤、调用工具、分配预算、处理权限、保存轨迹，并在最后用独立评估判断结果是否值得信任。

本章不再按题目和固定回答组织内容，而是沿着一个完整系统复习这些知识。读者会先看到一个合同证据助手如何从需求进入状态空间，再分别理解 generator、verifier、process supervision、search、tool executor、budget manager、评估和安全策略的职责。每个模块都有自己的目标、公式、输入输出、失败模式和边界，最后再把它们放进一个综合审计 demo。

综合复习的重点不是记住更多术语，而是能够回答三个工程问题：一个能力由什么机制产生，一个结果由什么证据支持，一个失败应该由哪一层修复。只有把这三个问题连起来，才能判断一个 reasoning 系统是真的变强，还是只是输出更长、花费更多或更会迎合评估器。

## 0.1 贯穿案例：合同证据助手

系统接收一份合同和一个问题：“请根据当前版本计算含税金额，并给出支持金额、税率和付款条件的页码；如果审批状态不足，不要发起付款。”

这个任务看似只是算术，实际上包含多个子任务：

1. 判断当前版本和有效页面；
2. 找到基础金额、折扣、税率和币种；
3. 计算中间金额和最终金额；
4. 检查引用是否真正支持每个 claim；
5. 判断是否需要澄清或人工审核；
6. 只允许执行与授权相符的工具动作；
7. 在预算和延迟范围内返回结果。

若模型直接输出一个数字，可能算术正确却引用旧合同；若只展示长推理，可能解释流畅却没有真正读取审批状态；若让模型直接调用付款接口，系统就把“生成建议”和“改变外部状态”混在了一起。这个案例足以连接本册的主要概念。

## 0.2 初学者视角：一条受控的推理流水线

初学者可以先把系统想成下面的过程：

~~~text
读取任务和约束
-> 生成一个或多个候选
-> 检查答案、步骤和证据
-> 必要时搜索、调用工具或修正
-> 按风险和预算决定是否继续
-> 输出可核对结果或请求人工确认
~~~

每一箭头都可能失败。生成器可能误读需求，verifier 可能漏掉错误，搜索可能过早剪枝，工具可能返回不可信文本，预算路由可能把难题分到低计算，安全层可能没有阻止不可逆动作。因此系统不能只看最后一行文字。

## 0.3 专家视角：三条账本

专家可以同时维护三条账本：

- 能力账本：候选是否正确、过程是否成立、变体是否泛化；
- 资源账本：token、候选数、搜索节点、工具调用、延迟和单位成功成本；
- 风险账本：证据缺失、权限不符、隐私、过度自信、评估污染和外部副作用。

三条账本不能互相抵消。成本低不代表安全，答案正确不代表权限正确，安全拒绝也不代表用户任务完成。综合系统的报告需要把这些结果放在同一任务上对齐，而不是只汇报一个总分。

## 0.4 资料与证据边界

本章引用的 CoT、self-consistency、Training Verifiers、过程监督、Tree-of-Thought、HumanEval 和 test-time compute scaling 论文分别支撑对应方法的研究背景；它们不自动证明闭源产品的内部实现。数学和代码案例是教学构造，合同助手是工程抽象，不是某个真实部署的性能报告。

回看入口按核心路径排列为：[Chain-of-Thought 第二章](02-chain-of-thought.md)、[self-consistency 与采样第三章](03-self-consistency与采样.md)、[verifier 第四章](04-verifier与reward-model.md)、[过程监督第五章](05-process-supervision.md)、[搜索第六章](06-search与tree-of-thought.md)、[推理时计算第七章](07-test-time-compute-scaling.md)、[数学训练第八章](08-数学推理训练.md)、[代码反馈第九章](09-代码推理与执行反馈.md)、[评估第十章](10-reasoning评估.md)和[安全第十一章](11-reasoning安全与局限.md)。

安全部分沿用 NIST 生成式 AI 风险管理、[OWASP GenAI LLM Top 10 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/)、[OWASP Agentic Applications Top 10 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)和[第十一章的防御性边界](11-reasoning安全与局限.md)。这些框架用于组织检查，不提供目标系统的发生率或控制有效性证明。公开系统卡、官方产品说明和论文的证据等级不同，涉及未公开模型细节时只写可核对的公开行为，不把推测写成内部事实。

## 1. 统一状态：Reasoning 系统到底在处理什么

### 1.1 任务状态

令 `x` 表示原始请求和不可变任务约束，`m_t` 表示当前证据、候选和中间计算结果，`e_t` 表示已确认的外部环境状态，`p_t` 表示当前身份和工具权限，`b_t` 表示剩余资源预算，`ell_t` 表示审计日志、错误和回滚信息。把合同助手在时刻 `t` 的完整状态写成：

~~~math
s_t=(x,m_t,e_t,p_t,b_t,\ell_t)
~~~

这里的 `b_t` 应由有限的非负资源余额组成，例如剩余 token、调用次数、毫秒和费用；余额耗尽时不能继续假设还有预算。`x` 中的授权、禁止事项和成功条件应是不可变字段，模型只能提出对 `m_t` 的候选更新，不能自行改写 `x`、`e_t` 或 `p_t`。`ell_t` 可以包含脱敏后的事件引用，但不能为了方便重放而无期限保存所有敏感原文。

例如第 7 步读取付款前合同：`x` 规定“只读，审批不足时不得付款”；`m_7` 已有待核对的金额 claim；`e_7` 记着合同版本 `v4`、审批状态为 pending；`p_7` 只有只读权限；`b_7` 还剩 1,200 个生成 token、3 次工具调用、2,000 毫秒和 0 元额外费用；`ell_7` 只保存脱敏事件 ID。这个状态让系统能检查版本、权限和预算，而不把它们藏在一段自然语言里。

如果状态只保存模型最近一段文字，就会丢掉版本、权限和预算。下一步动作可能因此基于旧合同、旧测试或已撤销权限。

### 1.2 动作与观察

动作可以是生成候选、读取页面、调用计算器、运行测试、请求澄清、提交补丁或请求人工确认。工具观察可能是文档片段、计算结果、编译错误、测试输出或权限拒绝。

状态转移可以抽象为：

令 `F` 是由宿主系统实施的状态转移函数，`a_t` 是第 `t` 步提出的动作，`o_t` 是工具或环境返回的观察，则：

~~~math
s_{t+1}=F(s_t,a_t,o_t)
~~~

`F` 不是模型自己决定的函数。权限、日志、资源限制和回滚都应由系统实现；模型只能在允许的动作空间中提出下一步。

具体地，若 `a_7=read_page(3)`，而 `o_7` 返回当前 `v4` 合同第 3 页的金额 `1,000 元`，系统验证返回版本后，得到 `m_8=m_7∪{金额 claim→第 3 页}`，`e_8` 仍是 `v4/pending`，工具余额从 3 次减为 2 次。若观察来自旧版本，状态转移应拒绝把它加入当前证据，而不是让模型自行判断版本可以忽略。

### 1.3 成功谓词

令 `AnswerOK`、`EvidenceOK`、`VersionOK`、`PermissionOK`、`ReviewOK` 分别表示答案正确、证据支持、版本有效、权限允许和要求的审核已完成；每项都取布尔值。合同助手的成功不是单一数字，而是这些检查的合取：

~~~math
\mathrm{Success}=
\mathrm{AnswerOK}
\land\mathrm{EvidenceOK}
\land\mathrm{VersionOK}
\land\mathrm{PermissionOK}
\land\mathrm{ReviewOK}
~~~

如果任务只是低风险只读计算，`ReviewOK` 可以由系统策略设为不需要人工；如果任务会改变付款状态，则需要显式确认。成功谓词必须根据任务风险定义，不能用一个通用布尔值覆盖所有场景。

例如只读报价场景的五项值为 `(1,1,1,1,1)`，合取结果为真；付款动作的审批仍为 pending 时，`ReviewOK=0`，即使答案、证据、版本和当前权限检查全为 1，成功仍是 `1∧1∧1∧1∧0=0`。

## 2. Generator：候选从哪里来

### 2.1 一次生成和多候选生成

Generator 把任务状态转换成文本答案、结构化动作、代码、数学步骤或工具计划。一次生成成本低、延迟短，但容易受单次采样错误影响；多候选生成可以提高正确候选出现的机会，却增加 token、选择和执行成本。

令 `p_{ij}` 表示任务 `i` 的第 `j` 个候选，`K_i` 是该任务生成的候选数。候选池可以表示为：

~~~math
P_i=\{p_{i1},p_{i2},\ldots,p_{iK_i}\}
~~~

候选不仅是不同答案文本，也可以是不同证据路径、不同代码实现、不同搜索分支或不同工具计划。

对需要聚合的任务，`K_i` 应是正整数；候选集合为空时，不能把“没有生成候选”和“生成了候选但全部错误”混成一个准确率。候选还应有稳定 ID、生成配置和所属任务 ID，便于区分重复采样与真正不同的路径。

例如任务 `i=2` 生成 `P_2={p_{21},p_{22},p_{23}}`，因此 `K_2=3`；三个候选可分别是不同算式、不同证据页和不同工具计划，候选 ID 应继续关联到同一个任务，而不能因为文本相同就合并掉不同的证据来源。

### 2.2 候选多样性

表面不同的候选可能共享同一个错误。评估多样性时可以比较答案等价类、失败测试集合、证据页集合、AST 结构或动作序列，而不是只比较字符串。

如果候选失败集合高度重叠，继续增加 `K` 的收益会下降；如果失败集合互补，verifier 或搜索更可能找到正确路径。多样性是选择质量的前提，不是越高越好。

### 2.3 Self-consistency

当任务有明确答案等价类时，令 `\hat y_j` 表示第 `j` 个候选的答案，`y` 表示一个规范化答案类，`norm` 把候选映射到可比较的答案类，`\mathbb{1}[condition]` 在条件成立时取 1、否则取 0，`K>0` 为候选数；可以对多个候选做规范化投票：

~~~math
\hat y=
\arg\max_y\sum_{j=1}^{K}
\mathbb{1}[\mathrm{norm}(\hat y_j)=y]
~~~

`norm` 必须把输出映射到有限且可比较的答案等价类；如果存在并列最高票，应预先规定固定的 tie-break 规则。`K` 越大只说明观察到的候选更多，不自动说明候选相互独立，也不说明多数答案是真实答案。

`norm` 把等价表达映射到同一类。投票依赖候选错误不完全相关；如果模型在同一错误模板上重复采样，多数票只会更稳定地选择错误。

例如四个候选经 `norm` 后为 `{1,040, 1,040, 1,040, 1,080}`，票数分别是 3 和 1，self-consistency 选择 `1,040`；它只是候选池的多数答案。若两类各 2 票，系统必须使用预先声明的 tie-break 规则，不能临时挑更符合期待的结果。

### 2.4 pass@k 和最终选择

如果 `n` 个候选中有 `c` 个通过独立 oracle，且 `k` 表示从该池中无放回抽取的候选数，候选池的 pass@k 估计为：

~~~math
\mathrm{pass@k}=
1-\frac{\binom{n-c}{k}}{\binom{n}{k}}
~~~

该估计要求 `n` 是正整数、`0\le c\le n`，并且 `1\le k\le n`；`c` 是同一候选池中通过独立 oracle 的候选数。`c=0` 时结果为 0；当 `n-c<k` 时组合数分子按 0 处理，结果为 1。候选池为空或 `k` 超出 `n` 时，pass@k 未定义，不能返回一个看似正常的 0。

它表示候选池中找到正确候选的潜力，不表示最终选择一定正确。最终系统还要报告 verifier 选择后的准确率、选择失败案例和候选存在但未被选中的比例。

例如 `n=5` 个候选中 `c=2` 个通过 oracle，抽看 `k=2` 个时：`pass@2=1-\binom{3}{2}/\binom{5}{2}=1-3/10=0.70`。这表示两次抽取至少遇到一个正确候选的池内概率估计，不表示最终 verifier 一定选中正确解。

### 2.5 生成失败的诊断

若 pass@k 很低，候选池中没有正确解，应先检查需求理解、训练数据、采样分布和工具接口；若 pass@k 高而最终选择低，应检查 verifier、聚合和测试覆盖；若原题高而变体低，应检查污染和模板泛化。三种失败不能用同一个“模型不够强”解释。

## 3. Verifier：如何判断候选

### 3.1 三类 verifier

程序化 verifier 使用代码、符号计算、单元测试、schema 或权限服务，适合可明确判定的约束；outcome verifier 判断最终答案或终态；process verifier 判断中间步骤、证据和状态是否成立。

合同助手中，金额公式可以用程序检查，页码支持需要文档证据检查，付款动作需要权限服务，模型生成的解释可以由人工或 judge 抽样。一个 verifier 不应承担自己无法观察的全部事实。

### 3.2 Pairwise reward model

如果 `r_w` 是偏好候选的分数，`r_l` 是较差候选的分数，且 `\sigma(z)=1/(1+e^{-z})` 是 logistic sigmoid，pairwise 训练损失可以写成：

~~~math
L_{\mathrm{pair}}=-\log\sigma(r_w-r_l)
~~~

取 `r_w=2`、`r_l=1`，则 `\sigma(1)≈0.731`，损失为 `-\log(0.731)≈0.313`；若两者同分，损失是 `-\log(0.5)≈0.693`。它训练排序关系，不保证分数具有概率含义。排序器可能偏好长度、格式和自信语气，因此需要 hard negative、独立 oracle 和校准评估。

### 3.3 过程监督

若 `q_{ij}\in\{0,1\}` 是第 `i` 条轨迹第 `j` 步的正确标签，`p_{ij}\in(0,1)` 是过程模型预测，`M>0` 是所有有效步骤的数量，可以使用：

~~~math
L_{\mathrm{proc}}=-\frac{1}{M}\sum_{i,j}
\left[q_{ij}\log p_{ij}+(1-q_{ij})\log(1-p_{ij})\right]
~~~

后续步骤可能只是第一处错误的连锁结果，标注时应区分第一处错误、受影响步骤和真正独立的新错误。

这里要求 `M>0`，`q_{ij}\in\{0,1\}`，而 `p_{ij}` 是严格位于 `(0,1)` 的有限概率；实际实现通常需要对模型输出做裁剪以避免 `\log 0`。没有有效步骤时，过程损失未定义，不能把空轨迹当成全错或全对。

例取三步标签 `q=(1,1,0)`、预测概率 `p=(0.9,0.8,0.2)`，则 `L_{proc}=[-\log0.9-\log0.8-\log(1-0.2)]/3≈(0.105+0.223+0.223)/3≈0.184`。第三步标签为 0，所以代入的是 `-\log(1-p_3)`，不能误算成 `-\log p_3`。

### 3.4 Verifier 的校准

候选选择器不仅要看 top-1 是否正确，还要看分数是否能表示不确定性。可以报告 ECE、Brier、pairwise accuracy、hard negative accuracy 和下游任务准确率。如果 verifier 自身高分但下游选择没有改善，说明它学习的是代理特征而非任务真值。

### 3.5 Verifier 失效的后果

错误 verifier 会把搜索方向引向更坏的区域；在 Best-of-N 中，它可能稳定选择格式漂亮的错误答案；在 RL 中，它可能变成可被利用的 reward。安全相关 verifier 还可能因为漏检一次越权动作造成比普通答案错误更大的影响，因此应把安全约束和质量得分分开。

## 4. Process Supervision：哪里错比是否错更有用

### 4.1 Outcome 和 process

Outcome supervision 只告诉系统最终答案是否正确，信号简单但稀疏；process supervision 记录步骤、状态、证据和错误类型，能支持训练、搜索和修复，但标注成本更高、步骤边界更主观。

两者的交叉结果有四种：答案对且过程对，答案对但过程错，答案错但过程大致对，答案和过程都错。综合训练不能把第一类和第二类混在一起。

### 4.2 第一处错误

令 `q_{ij}` 是以 1 开始编号的步骤正确性标签；若存在错误，轨迹 `i` 的第一处错误位置定义为：

~~~math
e_i^\star=\min\{j:q_{ij}=0\}
~~~

这个最小值只有在错误位置集合非空时才存在。若整条轨迹没有错误，应使用 `None` 或 `no_error` 状态；正文公式的步骤编号从 1 开始，代码实现若使用 0-based 索引，必须显式注明转换。

第一处错误用于修复和搜索剪枝，但它不是所有任务都能唯一标注。证明题可能有多条路径，代码中的局部错误可能来自更早的规格误读，因此需要保存标注分歧和证据。

若四步标签为 `q=(1,1,0,0)`，错误位置集合是 `{3,4}`，所以 `e^\star=\min\{3,4\}=3`；第 4 步可能只是第 3 步的后果。若标签为 `(1,1,1,1)`，集合为空，应记为 `None/no_error`，不能把不存在的最小值写成 0。

### 4.3 步骤粒度

步骤太大，verifier 无法定位错误；步骤太小，标注成本高，后续步骤高度相关。数学可按关键变换、代码可按补丁或测试状态、合同任务可按 claim 和证据对切分。步骤 schema 要与下游用途匹配，而不是追求一种全局统一粒度。

### 4.4 自动监督和人工监督

算术、代码编译、表达式等价和权限检查适合自动监督；问题理解、证据相关性和开放式证明需要人工或多模型辅助。自动标签要记录覆盖率和误判样本，人工标签要记录一致性、成本和争议。

## 5. Search：什么时候值得探索多个路径

### 5.1 状态、动作和终止

搜索系统需要定义状态、动作、转移、评分和终止条件。合同助手的状态可以是已确认页面和未解决 claim；动作可以是检索、计算、请求澄清或生成证据链；终止条件是所有关键字段得到支持，或预算耗尽。

### 5.2 Tree-of-Thought

ToT 将中间 thought 作为搜索节点，在每层产生多个候选并使用 verifier 或规则继续展开。它比单链 CoT 有更多回溯机会，但需要明确节点粒度和状态去重，否则“多个 thought”只是重复调用模型。

### 5.3 Beam 和 best-first

Beam search 按层保留有限候选，成本容易控制，却可能过早删除暂时低分的正确路径；best-first 按当前评分选择节点，适合不同深度的分支，但会受启发式偏差影响。评分应保留功能、证据、安全和成本分量。

### 5.4 MCTS 和探索利用

MCTS 使用访问次数和价值估计平衡探索与利用。令 `Q(v)` 为节点价值，`N_p` 和 `N_v` 分别为父节点和当前节点的访问次数，`c` 为非负探索系数，常见 UCT 形式为：

~~~math
U(v)=Q(v)+c\sqrt{\frac{\log(N_p+1)}{N_v+1}}
~~~

语言任务的难点不是记住公式，而是如何定义状态、模拟、回传价值和可靠的终止奖励。

取 `Q(v)=0.6`、`N_p=10`、`N_v=3`、`c=1`，则 `U(v)=0.6+\sqrt{\ln(11)/4}≈0.6+0.774=1.374`。探索项会提高访问较少节点的分数；它不是正确性的概率，也不能替代独立终止验证。

### 5.5 正确路径误剪

如果 verifier 把中间状态误判为坏，搜索会删除最终正确路径；如果只按当前测试分数剪枝，暂时低分但能修复的代码也可能消失。保留少量多样分支、设置回溯、使用硬约束和记录剪枝原因，可以减少这种错误。

### 5.6 搜索成本

搜索节点、模型 token、工具调用和 verifier 都会增加成本。搜索不应默认用于每个请求，应根据任务难度、错误代价、可验证性和剩余预算选择。高风险任务还要考虑搜索扩大外部动作机会的风险。

## 6. 推理时计算和动态预算

### 6.1 预算向量

令 `K_i`、`T_i`、`D_i`、`J_i`、`U_i` 和 `L_i` 分别表示候选数、生成 token 数、搜索节点数、verifier 调用数、工具调用数和延迟。可以用下面的向量记录一次请求的推理时资源：

~~~math
b_i=(K_i,T_i,D_i,J_i,U_i,L_i)
~~~

`D_i` 在本章按搜索节点计数；延迟的单位固定为秒，不能在同一账本里混用毫秒和秒。这里使用 `J_i`，避免与前文的变体集合 `V_i` 混淆。

在预算比较中，计数通常取非负整数，token 和延迟可以取有限非负数；如果系统用“未知”或“未记录”，应保留缺失状态，不能悄悄当作 0。不同策略必须采用同一计量口径，例如是否包含重试、排队、人工复核和工具返回 token。

例如一个请求使用 `K=3` 个候选、生成 `T=1,200` 个 token、搜索 `D=5` 个节点、调用 verifier `J=4` 次、工具 `U=1` 次，端到端延迟 `L=2` 秒，则账本记为 `b=(3,1200,5,4,1,2)`。若未记录工具耗时，不能把 `L` 误读成工具调用耗时之和。

### 6.2 成本账本

令 `c_{tok}`、`c_{cand}`、`c_{node}`、`c_{ver}`、`c_{tool}`、`c_{lat}` 是把各资源项换算成抽象成本的非负系数。教学成本模型为：

~~~math
C_i=c_{\mathrm{tok}}T_i
+c_{\mathrm{cand}}K_i
+c_{\mathrm{node}}D_i
+c_{\mathrm{ver}}J_i
+c_{\mathrm{tool}}U_i
+c_{\mathrm{lat}}L_i
~~~

这些 `c` 是换算系数，不是跨系统通用价格。产品报告应分别保存 token、GPU 秒、工具费用、人工成本和延迟，必要时再给出统一的单位成功成本。教学时令 `c_tok=0.001`、`c_cand=0.10`、`c_node=0.03`、`c_ver=0.02`、`c_tool=0.25`、`c_lat=0.05`，并使用上一节的 `T=1200,K=3,D=5,J=4,U=1,L=2`，则 `C=1.20+0.30+0.15+0.08+0.25+0.10=2.08` 个抽象成本单位。这些权重只为展示记账；实际标定时不能把已计入 token 的成本再次重复算进候选项。

若要把 `C_i` 用于排序或单位成功成本，资源量和系数都应是有限非负数；允许某一项权重为 0，但负权重会让增加资源反而降低账面成本。统一成本只在换算系数经过说明和校准后才有比较意义。

### 6.3 边际收益

如果预算从 `B_1` 增加到 `B_2`，令 `A(B)` 为预算 `B` 下的任务准确率，`C(B)` 为相同任务集上的成本，则质量—成本边际收益可写为：

~~~math
g(B_1,B_2)=
\frac{A(B_2)-A(B_1)}{C(B_2)-C(B_1)}
~~~

该式要求两个预算点有效，且 `C(B_2)-C(B_1)\ne0`；通常还要说明是否要求 `B_2>B_1`。成本没有增加时，不能除以 0，应分别报告质量变化和成本变化。若 `A(B)` 的评估样本、超时处理或安全失败定义在两个预算点不同，边际收益同时混入了评估规则变化。

边际收益递减不表示高预算没有价值。高风险、难题或可验证任务可能值得高预算；简单请求则可能只需要短路径。路由器需要同时考虑价值、风险和延迟。

例如预算点 1 的准确率 `A_1=0.60`、成本 `C_1=2`，预算点 2 的准确率 `A_2=0.75`、成本 `C_2=3`，则 `g=(0.75-0.60)/(3-2)=0.15`，即每增加 1 个抽象成本单位，准确率增加 0.15。只有在两点使用同一任务集、成功定义和成本口径时，这个差值才可比较。

### 6.4 动态路由

路由器可以根据任务类型、难度、可验证性、业务价值和风险选择 direct、采样、verifier、search、tool 或人工确认。路由器本身也要评估公平性和错误分配：难题被低预算处理、高风险动作被自动执行，都是路由失败。

## 7. 数学和代码：两个可验证训练场

### 7.1 数学训练闭环

数学数据可以包含题目、答案等价类、步骤、第一处错误、难度、题型、来源、验证和污染标签。程序生成适合控制数字和结构，专家解法适合提供证明和多解法，模型初标适合扩展规模但需要抽查。

训练路线可以是：SFT 学习题型和可读步骤，过程监督学习局部正确性，verifier 过滤和重排候选，RL 或拒绝采样利用程序反馈，再用独立新题和变体检查迁移。每个环节都有自己的失败模式，不能用数学 benchmark 的最终分数替代数据审计。

### 7.2 代码执行反馈

代码训练可以保存需求、接口、公开/隐藏测试、执行环境、初始程序、错误反馈、补丁和回归结果。编译器、解释器、测试器和静态扫描提供不同信号；公开测试全过不等于隐藏泛化，pass@k 高不等于选择器可靠。

执行代码必须有独立沙箱、文件和网络限制、资源上限、依赖版本和日志。模型生成代码可以成为训练样本，但不能直接获得生产权限。

### 7.3 数学和代码的共同点

两者都能提供自动验证、过程反馈、错误纠正和候选搜索；两者也都容易被公开测试、模板和 verifier 漏洞利用。共同指标包括答案/行为正确率、第一处错误、候选池覆盖、隐藏或变体泛化、成本和安全。

### 7.4 两者的差异

数学证明可能存在多条合理路径，步骤标签更主观；代码可以运行，但环境、依赖、权限和资源会引入额外状态。不能简单把数学 PRM 的标签方式复制到代码，也不能把代码测试通过当作开放证明的完整正确性。

## 8. 评估：如何区分真实提升

### 8.1 答案、过程和任务完成

最终答案准确率是必要指标，但应与步骤准确率、第一处错误、证据支持、工具状态和安全权限分开。合同助手的金额正确、引用正确和付款权限正确，分别属于不同的检查。

### 8.2 原题、变体和污染

原题高而变体低，可能是记忆或模板；污染样本高而新题低，评估独立性不足；同一题族跨 train/test，平均分会过于乐观。报告至少包括原题、变体、新题和高风险污染切片。

### 8.3 配对提升

在同一批 `N>0` 个题上比较 baseline 和 candidate；令 `y_i^{new}`、`y_i^{base}` 分别为两系统按同一 oracle 得到的 0/1 正确标签，`\mathbb{1}[condition]` 在条件成立时为 1、否则为 0，单题差值为：

~~~math
d_i=\mathbb{1}[y_i^{\mathrm{new}}\text{ correct}]
-\mathbb{1}[y_i^{\mathrm{base}}\text{ correct}]
~~~

配对提升是 `sum(d_i)/N`。还要列出 candidate 修复和退化的样本，不能只报两个平均分的差。

这里要求 `N>0`，且两个系统在同一批任务、同一 oracle 和同一成功判定下形成一一配对。超时、拒答和执行错误应先按任务契约编码，不能对 baseline 和 candidate 使用不同的缺失值处理。

例如四题的新旧系统正确性依次为 `(1,0)`、`(0,1)`、`(1,1)`、`(0,0)`，则 `d=(1,-1,0,0)`，平均配对提升为 `(1-1+0+0)/4=0`。平均分没有变化，但第一题是修复、第二题是回退；只报总分会漏掉这两种变化。

### 8.4 Bootstrap 和样本量

从任务或题目族重采样配对差值，得到 `R>0` 个 bootstrap 估计 `\Delta^*`；令 `0<\alpha<1` 为尾部概率、`q_p` 为已声明分位数规则下的 `p` 分位数，可以得到提升区间：

~~~math
\mathrm{CI}_{1-\alpha}=
\left[q_{\alpha/2}(\Delta^\ast),q_{1-\alpha/2}(\Delta^\ast)\right]
~~~

差值集合必须非空，重采样轮数 `R` 是正整数，且 `0<\alpha<1`。`q` 的分位数实现要固定，例如使用最近秩或线性插值；小样本下不同约定可能产生不同端点。若题目按仓库、模板或题目族相关，应按组重采样。

例如有 `R=100` 次重采样，排序后的第 5 和第 95 个提升值分别为 `0.01` 和 `0.04`；取 `\alpha=0.10`、最近秩规则，则两端秩是 `\lceil0.05\times100\rceil=5` 与 `\lceil0.95\times100\rceil=95`，所以 90% 区间为 `[0.01,0.04]`。这是排序端点的教学算例，不是本书系统的实验结果。

小样本区间很宽是正常现象。推理时计算曲线和安全稀有事件尤其需要足够任务数、多个随机种子和按族重采样。

### 8.5 质量和成本曲线

比较 greedy、self-consistency、verifier 和 search 时，要把候选数、token、verifier、工具、延迟和准确率一起画出来。一个低预算稳定的系统，可能比高预算偶尔正确的系统更适合实时产品；高预算策略也可能在高风险难题上值得保留。

## 9. 安全：为什么能力链必须有权限边界

### 9.1 伪推理和过度自信

长解释可能只是事后合理化，正确答案可能掩盖错误过程；确定语气可能掩盖证据不足。需要过程标注、反事实、独立 verifier、置信度校准和澄清机制。

### 9.2 工具和外部状态

模型可以提出工具动作，但权限、参数、确认和回滚由系统决定。工具输出是观察，不是上级指令；网页、文件、邮件和代码注释中的文本都可能改变模型目标，必须和系统策略隔离。

### 9.3 高风险任务

医疗、法律、金融、身份、工业和不可逆外部动作需要分层处理。系统可以提供一般性辅助、证据整理和只读计算，但个体决策、付款、权限修改和高影响建议需要人工审核或拒绝不适当请求。

### 9.4 防御纵深

安全不能只靠模型拒答。需要模型训练、策略服务、工具白名单、沙箱、人工确认、审计日志、红队回归、版本监控和事故响应。安全指标和质量指标不能互相抵消。

## 10. 综合系统设计：一个可重放的闭环

### 10.1 组件契约

一个受控 reasoning 系统可以包含：

1. `router`：预测任务难度、价值、风险和可验证性；
2. `generator`：生成答案、步骤、代码或工具计划；
3. `verifier`：检查答案、过程、证据、代码和权限；
4. `search_controller`：管理候选、分支、回溯和终止；
5. `tool_executor`：在最小权限和沙箱中执行允许动作；
6. `budget_manager`：控制 token、候选、搜索、工具和延迟；
7. `aggregator`：根据独立证据选择或合并结果；
8. `safety_policy`：判断拒答、确认、人工审核和回滚；
9. `logger`：保存可重放的状态、动作、观察和结果。

每个组件都要定义输入、输出、错误和权限。组件名称不是系统设计，契约和可重放行为才是。

### 10.2 一次合同请求的轨迹

~~~text
任务解析 -> 版本检索 -> 证据候选 -> 金额计算
-> 程序/规则验证 -> 证据与 claim 对齐
-> 风险和权限判断 -> 输出预览或请求人工确认
~~~

如果版本检索失败，不能直接进入金额计算；如果 claim 没有证据，不能用模型自信语气填空；如果付款状态需要人工确认，不能让 aggregator 直接调用付款接口。

### 10.3 可恢复性

每个检查点保存状态摘要、证据引用、工具结果和版本。失败后可以回到最近可信状态，重新生成或请求澄清。不可逆动作前必须有明确的预览和确认，不能把回滚责任交给模型生成一段道歉文本。

### 10.4 观察与审计

评估和线上监控要把最终答案、过程摘要、候选分数、verifier、工具动作、权限、成本和外部影响关联到同一任务 ID。这样才能回答“为什么选了这个候选”“哪个 verifier 放过了错误”“哪个动作改变了状态”。

## 11. 失败归因：哪一层应该修

### 11.1 生成失败

正确候选没有出现，说明需求理解、训练数据、采样分布或模型能力需要改进。增加 verifier 和搜索无法凭空创造候选，只能从现有候选中选择。

### 11.2 选择失败

候选池存在正确解，但最终选错，说明 verifier、投票、排序或测试覆盖有问题。需要 hard negative、独立 oracle、校准和选择器消融。

### 11.3 过程失败

答案正确但过程错误，说明结果监督不足、过程标签有噪声或模型学会了猜答案。需要第一处错误、反事实和过程/结果交叉数据。

### 11.4 搜索失败

正确路径被剪掉、分支重复、状态合并错误或预算过早耗尽，说明搜索控制器、评分、节点表示或预算路由有问题。

### 11.5 工具失败

工具结果不可信、权限过宽、参数错误或环境不稳定时，模型可能在错误观察上继续推理。先修工具契约和执行环境，不要只增加模型提示。

### 11.6 评估失败

污染、oracle 错误、变体难度不匹配、样本量过小或 judge 偏差会让实验结论失真。评估失败不是模型失败，应单独记录并修正实验设计。

### 11.7 安全失败

工具越权、隐私暴露、过度代理和高风险不当服从属于系统安全失败，即使最终答案正确也不能被质量分抵消。应冻结风险动作、保留日志、回滚并进入安全回归。

## 12. 综合审计 demo：把能力、成本和安全放在一起

下面的 0 依赖 demo 使用六个教学任务，分别记录候选池是否存在正确解、最终选择、过程、变体、安全、成本和风险。它不调用模型、工具或外部数据，目的是演示综合报告如何同时暴露能力和系统边界。

为了让这个综合表可以复核，任务 schema 约定如下：`id` 必须唯一，能力和安全字段必须是布尔值，`cost` 是有限非负数，`risk` 为 `None` 或与记录问题相符的主要风险类别。若候选池没有正确解，最终选择不可能被标记为正确；若动作不安全，必须记为 `permission_failure`。这些只是教学数据的一致性检查，真实系统还需要独立 oracle 和权限服务重新验证。

**代码状态：已验证（2026-11-13，Python 3.12.9；仅标准库）**

~~~python
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import math


TASKS = [
    {
        "id": "math_multi",
        "candidate_pool_has_correct": True,
        "selected_correct": True,
        "process_correct": True,
        "variant_correct": False,
        "safe_action": True,
        "cost": 120,
        "risk": None,
    },
    {
        "id": "code_hidden",
        "candidate_pool_has_correct": True,
        "selected_correct": True,
        "process_correct": True,
        "variant_correct": True,
        "safe_action": True,
        "cost": 260,
        "risk": None,
    },
    {
        "id": "proof_wrong",
        "candidate_pool_has_correct": False,
        "selected_correct": False,
        "process_correct": False,
        "variant_correct": False,
        "safe_action": True,
        "cost": 420,
        "risk": "process_failure",
    },
    {
        "id": "contract_evidence",
        "candidate_pool_has_correct": True,
        "selected_correct": True,
        "process_correct": False,
        "variant_correct": False,
        "safe_action": False,
        "cost": 500,
        "risk": "permission_failure",
    },
    {
        "id": "simple_regression",
        "candidate_pool_has_correct": True,
        "selected_correct": False,
        "process_correct": True,
        "variant_correct": True,
        "safe_action": True,
        "cost": 80,
        "risk": "selection_failure",
    },
    {
        "id": "tool_plan",
        "candidate_pool_has_correct": True,
        "selected_correct": True,
        "process_correct": True,
        "variant_correct": True,
        "safe_action": True,
        "cost": 300,
        "risk": None,
    },
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_nonnegative_number(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, float):
        return math.isfinite(value) and value >= 0
    if isinstance(value, Decimal):
        return value.is_finite() and value >= 0
    return False


def validate_tasks(tasks):
    require(isinstance(tasks, list) and tasks, "tasks must be a non-empty list")
    ids = set()
    boolean_fields = (
        "candidate_pool_has_correct",
        "selected_correct",
        "process_correct",
        "variant_correct",
        "safe_action",
    )
    allowed_risks = {None, "process_failure", "permission_failure", "selection_failure"}
    for task in tasks:
        require(isinstance(task, dict), "each task must be a dictionary")
        task_id = task.get("id")
        require(isinstance(task_id, str) and task_id, "task id must be non-empty text")
        require(task_id not in ids, "task ids must be unique")
        ids.add(task_id)
        for field in boolean_fields:
            require(type(task.get(field)) is bool, f"{field} must be boolean")
        risk = task.get("risk")
        require(
            risk is None or isinstance(risk, str) and risk in allowed_risks,
            "unknown risk category",
        )
        cost = task.get("cost")
        require(
            isinstance(cost, (int, float)) and finite_nonnegative_number(cost),
            "cost must be a finite non-negative number",
        )
        require(
            not task["selected_correct"] or task["candidate_pool_has_correct"],
            "a correct selection requires a correct candidate in the pool",
        )
        require(
            task["safe_action"] or risk == "permission_failure",
            "an unsafe action requires a permission_failure label",
        )
        if risk == "process_failure":
            require(not task["process_correct"], "process_failure conflicts with process_correct")
        elif risk == "permission_failure":
            require(not task["safe_action"], "permission_failure conflicts with safe_action")
        elif risk == "selection_failure":
            require(
                task["candidate_pool_has_correct"] and not task["selected_correct"],
                "selection_failure requires a correct candidate that was not selected",
            )


validate_tasks(TASKS)


def ratio(numerator, denominator):
    require(
        type(numerator) is int and numerator >= 0,
        "numerator must be a non-negative integer count",
    )
    require(
        type(denominator) is int and denominator >= 0,
        "denominator must be a non-negative integer count",
    )
    require(numerator <= denominator, "numerator cannot exceed denominator")
    return round(float(Fraction(numerator, denominator)), 3) if denominator else None


def accuracy(field):
    require(
        isinstance(field, str)
        and field in {
            "candidate_pool_has_correct",
            "selected_correct",
            "process_correct",
            "variant_correct",
            "safe_action",
        },
        "unknown task field",
    )
    return ratio(sum(row[field] for row in TASKS), len(TASKS))


def cost_per_success(total_cost, successes):
    require(
        finite_nonnegative_number(total_cost),
        "total cost must be finite and non-negative",
    )
    require(type(successes) is int and successes >= 0, "successes must be a non-negative integer")
    if not successes:
        return None
    try:
        result = float(Decimal(str(total_cost)) / Decimal(successes))
    except (InvalidOperation, OverflowError) as error:
        raise ValueError("cost per success could not be represented") from error
    require(math.isfinite(result), "cost per success must fit in a finite float")
    return round(result, 3)


candidate_pool_coverage = accuracy("candidate_pool_has_correct")
selected_accuracy = accuracy("selected_correct")
process_accuracy = accuracy("process_correct")
variant_accuracy = accuracy("variant_correct")
safe_action_rate = accuracy("safe_action")
correct_count = sum(row["selected_correct"] for row in TASKS)
total_cost = sum((Decimal(str(row["cost"])) for row in TASKS), Decimal("0"))
cost_per_correct = cost_per_success(total_cost, correct_count)
risk_ids = [row["id"] for row in TASKS if row["risk"] is not None]
rescue_candidates = [
    row["id"]
    for row in TASKS
    if row["candidate_pool_has_correct"] and not row["selected_correct"]
]
review = {
    "candidate_pool_ok": candidate_pool_coverage is not None and candidate_pool_coverage >= 0.8,
    "selected_accuracy_ok": selected_accuracy is not None and selected_accuracy >= 0.75,
    "process_ok": process_accuracy is not None and process_accuracy >= 0.75,
    "variant_ok": variant_accuracy is not None and variant_accuracy >= 0.7,
    "safe_action_ok": safe_action_rate is not None and safe_action_rate == 1.0,
    "cost_ok": cost_per_correct is not None and cost_per_correct <= 400,
}
summary = {
    "candidate_pool_coverage": candidate_pool_coverage,
    "selected_accuracy": selected_accuracy,
    "process_accuracy": process_accuracy,
    "variant_accuracy": variant_accuracy,
    "safe_action_rate": safe_action_rate,
    "cost_per_correct": cost_per_correct,
}

print(f"summary={summary}")
print(f"risk_ids={risk_ids}")
print(f"rescue_candidates={rescue_candidates}")
print(f"review={review}")
print(f"all_review_checks_pass={all(review.values())}")
~~~

预期输出：

~~~text
summary={'candidate_pool_coverage': 0.833, 'selected_accuracy': 0.667, 'process_accuracy': 0.667, 'variant_accuracy': 0.5, 'safe_action_rate': 0.833, 'cost_per_correct': 420.0}
risk_ids=['proof_wrong', 'contract_evidence', 'simple_regression']
rescue_candidates=['simple_regression']
review={'candidate_pool_ok': True, 'selected_accuracy_ok': False, 'process_ok': False, 'variant_ok': False, 'safe_action_ok': False, 'cost_ok': False}
all_review_checks_pass=False
~~~

### 12.1 先看候选池

`candidate_pool_coverage=0.833` 表示六个任务中五个任务的候选池包含正确解。这个数字说明生成器并非完全失败，但不能说明系统能选对。`simple_regression` 是候选池有正确解、最终选择却错误的任务，因此它是选择器问题，而不是生成器问题。

### 12.2 再看过程和变体

最终选择准确率只有 `0.667`，过程准确率也是 `0.667`，变体准确率更低，为 `0.5`。`math_multi` 和 `contract_evidence` 说明原题或任务结果正确，并不代表过程、证据和变体都正确；综合系统需要为这些字段分别设置数据和验证。

### 12.3 再看安全和成本

`contract_evidence` 的金额结果正确，但 `safe_action=False`，代表权限或外部动作失败；它不能被最终答案分数抵消。总成本除以四个选择正确任务得到 `420`，超过 demo 的教学检查阈值，说明增加候选和工具并没有带来足够单位收益。

### 12.4 为什么最终结果是 false

`all_review_checks_pass=False` 不是程序错误，而是审计对象本身暴露了选择、过程、变体、安全和成本问题。综合报告的价值就在于指出下一步：改进 verifier 选择 `simple_regression` 的正确候选，补过程数据，增加数学和合同变体，收紧权限，并降低无效调用成本。

### 12.5 demo 的边界

这些布尔字段是人工构造的标签，不代表真实模型的统计结果；成本是抽象单位，不是价格；`safe_action` 也不是实际权限服务。现实系统需要真正执行器、独立 oracle、日志和安全审查。

## 13. 综合系统的实验顺序

### 13.1 先做 direct baseline

先用单次生成测量基础能力，再加入多候选、verifier、search、tool 和人工审核。这样可以知道每个模块增加了什么，也能发现新模块是否伤害简单题。

### 13.2 再做单模块消融

固定模型和任务，只改变候选数、verifier、过程评分、搜索深度或工具权限。每次记录答案、过程、变体、成本和安全，避免多个变量同时变化造成错误归因。

### 13.3 再做联合策略

组合 generator、verifier、search 和 tool 后，报告每个组件的调用次数、错误转移和预算分布。联合策略可能出现模块交互问题，例如 verifier 偏差被 search 放大，工具噪声使路由器分错，高预算使简单题回归。

### 13.4 最后做新题和高风险回归

在公开开发集优化后，用新来源、变体、污染敏感样本、工具异常和高风险场景测试。综合系统只有在独立回归和可重放日志下，才适合进入小范围灰度；灰度也应保留人工和回滚。

## 14. 小练习

### 练习一：写出状态契约

为合同证据助手定义 `s_t`、动作、观察、权限和终止状态。说明哪些字段不可由模型修改，哪些动作必须先获得确认。

### 练习二：区分生成和选择

构造四个任务：两个候选池没有正确解，一个候选池有正确解但选择器选错，一个候选池和选择器都正确。分别标注应修复 generator 还是 verifier。

### 练习三：过程监督

写出一条最终答案正确但第一步错误的数学轨迹，标注第一处错误、受影响步骤和答案结果。说明为什么 outcome-only 数据会把它误收为正例。

### 练习四：搜索预算

为简单算术、中等代码修复、复杂证明和合同付款预览分配 `K,T,D,J,U,L`，说明每个预算的风险和成本理由。

### 练习五：评估报告

设计一张表，同时展示 selected accuracy、process accuracy、variant accuracy、candidate pool coverage、safe action rate、P95 和 cost per correct。给出一个平均分提升但安全退化的反例。

### 练习六：运行综合 demo

把 `simple_regression` 的 `selected_correct` 改为 `True`，并把其主要风险标签改为 `None`，重新运行 demo：选择准确率变为 `0.833`，单位正确成本降至 `336.0`，`risk_ids` 不再包含该任务；过程准确率仍为 `0.667`、变体准确率仍为 `0.5`、安全动作率仍为 `0.833`，所以整体验收仍失败。解释为什么修复选择器不代表过程、变体和安全能力自动提高。

## 15. 本章总结

Reasoning 系统不是一个会输出长答案的模型，而是一组围绕状态、候选、验证、搜索、工具、预算、评估和权限组成的可审计闭环。Generator 负责提出可能解，verifier 负责检查，process supervision 负责定位路径质量，search 负责探索，tool executor 负责受控观察，budget manager 负责成本，safety policy 负责边界，logger 负责重放和复盘。

综合评估必须区分候选池覆盖、最终选择、过程正确、变体泛化、安全动作和单位成功成本。一个模块的成功不能抵消另一个模块的失败：正确金额不能抵消越权付款，pass@k 不能抵消选择错误，长 CoT 不能抵消伪推理，高平均分不能抵消污染和严重度高的事故。

真正可迁移的 reasoning 能力，应在新题、变体、工具反馈、错误纠正、不同预算和高风险边界上保持可解释表现。系统还要知道什么时候继续推理，什么时候调用验证，什么时候请求澄清，什么时候交给人工，什么时候停止并回滚。

## 16. 资料索引

1. Chain-of-Thought Prompting：<https://arxiv.org/abs/2201.11903>
2. Self-Consistency：<https://arxiv.org/abs/2203.11171>
3. Training Verifiers to Solve Math Word Problems：<https://arxiv.org/abs/2110.14168>
4. Let's Verify Step by Step：<https://arxiv.org/abs/2305.20050>
5. Tree of Thoughts：<https://arxiv.org/abs/2305.10601>
6. HumanEval：<https://arxiv.org/abs/2107.03374>
7. Scaling LLM Test-Time Compute Optimally：<https://arxiv.org/abs/2408.03314>
8. NIST AI 600-1 Generative AI Profile：<https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence>
9. OWASP GenAI LLM Top 10 2026：<https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/>
10. OWASP Top 10 for Agentic Applications 2026：<https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/>

本章将论文机制、公开评估、官方治理资料、教学案例和目标系统实测区分开来。综合复习只能帮助建立系统推理框架，不能替代针对具体模型、数据、工具权限和部署环境的独立验证。
