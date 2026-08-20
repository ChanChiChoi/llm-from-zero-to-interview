# 第五章：过程监督：把“哪里错了”变成可学习信号

很多推理系统只在最后检查答案。答案对，就把整条轨迹当作成功；答案错，就把整条轨迹当作失败。这种做法简单，却把一个很长的因果链压缩成了一个比特：`成功` 或 `失败`。

设想一个学生解数学题。学生先正确读出了题目，正确写出关系式，随后在一次减法中把 `3 - 1` 算成 `1`，最后得到错误答案。如果老师只看最后一行，只能说“这题错了”；如果老师检查每一步，就能知道错误发生在计算，而不是读题或建模。过程监督要做的，就是把这种更细的反馈变成数据、模型和搜索算法可以使用的信号。

但过程监督不是把每一句话都贴上“正确”或“错误”这么简单。它必须回答一连串更困难的问题：什么算一步？一步是事实正确，还是必须对当前任务有用？后面的连锁错误是否应该重复处罚？两条不同的解法能否同时被接受？模型写出漂亮的解释，是否真的代表它使用了这些步骤？一个过程奖励模型如果错了，搜索会不会把真正的正确路径剪掉？

本章从这些问题出发，建立一套完整的理解框架。读完本章，初学者应当能够解释 outcome supervision、process supervision、PRM 和 ORM 的区别；能够读懂逐步标注数据；能够用简单公式计算步骤质量、第一处错误和搜索选择指标。工程人员则需要进一步掌握数据分布、标注协议、校准、分布偏移、自动验证、安全沙箱和单位成功成本，因为过程监督的价值最终不在于分数好看，而在于它是否让真实任务更可靠。

## 0.1 先看一个具体问题：同一个答案，不同的过程

有一个水箱，进水速度是每分钟 3 升，出水速度是每分钟 1 升，水箱需要增加 20 升。模型给出了两条候选解法。

~~~text
候选 A：
净速度 = 3 - 1 = 2 升/分钟。
时间 = 20 / 2 = 10 分钟。
答案：10 分钟。

候选 B：
净速度 = 3 + 1 = 4 升/分钟。
时间 = 20 / 4 = 5 分钟。
不过我检查了另一种计算，最后答案写成 10 分钟。
~~~

如果只比较最终答案，A 和 B 看起来一样：答案都是 10 分钟。如果只把答案当监督标签，两条轨迹都可能被当作正例。可是 B 的过程包含一个错误的净速度，最终答案只是碰巧写对。将 B 当作优质示范，会把错误的中间规则灌入训练数据。

反过来，再看第三条轨迹：

~~~text
候选 C：
净速度 = 3 - 1 = 2 升/分钟。
时间 = 20 / 2 = 10 分钟。
输出格式不符合任务要求，缺少单位字段。
~~~

C 的数学过程是对的，但如果任务契约要求结构化输出，它仍然不能直接交付。这里至少有三个彼此不同的判断对象：数学过程是否正确，最终数值是否正确，输出协议是否满足。过程监督只能解决其中一部分，不能替代最终答案检查和格式验证。

这也是本章最重要的起点：过程标签不是“真实质量”的同义词，而是针对某个任务契约设计的一组观察。任务契约改变，标签含义也会改变。

## 0.2 本章的资料与证据边界

过程监督的经典公开资料主要来自数学推理和验证器研究。本章使用以下几类来源，并在叙述中区分它们的证据等级。

第一类是原始论文。`Training Verifiers to Solve Math Word Problems` 研究了用验证器判断数学候选答案的路线，适合说明 outcome-level verifier 和候选筛选的基础思路；`Let's Verify Step by Step` 讨论了对数学推理步骤进行监督，并公开了 PRM800K 数据集和相关实验，适合说明过程标签、步骤反馈和过程奖励模型的研究背景。

第二类是公开数据和代码仓库。`openai/prm800k` 提供数据说明、标注格式和相关使用入口，可以帮助读者理解真实过程数据并不只是“每行一个 0/1 标签”，还涉及步骤边界、标注选择和数据组织。公开仓库能证明仓库中公开了什么，不能证明闭源系统内部采用了完全相同的训练配方。

第三类是本章的教学构造。后面的水箱题、合同金额案例和 Python 程序是为了演示指标关系而人为设计的。它们可以运行、可以复核，但不代表 PRM800K 的真实统计，也不代表任何商业模型的效果。

因此，本章不会把以下说法混在一起：论文报告过某个实验结果；公开仓库包含某种字段；本章 demo 在 toy 数据上得到某个数字；某个线上系统一定使用了过程奖励模型。每一种说法都需要自己的证据。

## 0.3 初学者视角：结果监督和过程监督

结果监督只问“最后对不对”。如果题目是 `2 + 3`，标签可能是 `5`；如果是代码任务，标签可能是“通过所有测试”或“没有通过”。

过程监督还会问“中间每个可检查状态是否成立”。在数学题里，这可能是每个代数变换；在代码任务里，可能是每次修改后的测试结果、变量不变量或编译状态；在工具任务里，可能是参数是否来自证据、资源是否经过权限检查。

可以用一个简单对照理解两者：

~~~text
结果监督：
问题 -> 最终答案 -> 对 / 错

过程监督：
问题 -> 步骤 1 -> 步骤 2 -> 步骤 3 -> 最终答案
             对        对        错          错
~~~

结果监督的优点是标签相对便宜、目标清晰。过程监督的优点是反馈更密集，可以定位错误，也可以给搜索中的中间节点打分。它的代价是步骤划分、标注一致性和错误定义都更复杂。

不能把“反馈更密集”理解为“必然更好”。如果每一步的标签不可靠，模型会得到更多错误反馈；如果过程评分器偏好某种表达风格，生成器可能学会模仿参考写法，却没有提高真实任务能力。

## 1. 监督对象：答案、步骤和状态

### 1.1 把一条轨迹写成数据对象

设第 `i` 个样本的输入是 `x_i`，模型生成的过程由 `M_i` 个步骤组成：

~~~math
z_i=(z_{i1},z_{i2},\ldots,z_{iM_i})
~~~

`z_ij` 表示第 `i` 个样本的第 `j` 个步骤。最终答案记为 `y_hat_i`，任务期望答案记为 `y_i^star`。如果只做结果监督，训练数据通常只需要 `(x_i, y_i^star)` 或 `(x_i, y_hat_i, outcome_label_i)`。

过程监督还需要描述每一步。最简单的正确性标签是：

~~~math
q_{ij}\in\{0,1\}
~~~

`q_ij = 1` 表示该步骤在当前任务契约下成立，`q_ij = 0` 表示不成立。这里的“成立”要有明确参照：它可以是数学等价性、代码测试、证据支持或人工判断，不能脱离任务单独解释。

### 1.2 正确性不等于相关性

一个句子可能在事实层面正确，却对任务没有帮助。题目说“仓库有 3 个苹果”，模型补充“苹果是一种水果”，这句话没有事实错误，但没有推进计算。

因此可以额外记录相关性：

~~~math
r_{ij}\in\{0,1\}
~~~

`r_ij = 1` 表示该步骤与完成当前任务有关。对开放式解释任务，相关性可能还有“必要”“可选”“重复”几个等级；在数学演示中，二值标签已经足够说明问题。

如果只最大化 `q_ij`，模型可能生成大量正确但无关的句子。如果只最大化 `r_ij`，模型可能追求简短，却遗漏关键证明。因此正确性和任务相关性通常应分开记录，再决定如何组合。

### 1.3 状态转移比句子列表更接近工具任务

在工具调用或代码执行任务中，一步不一定是自然语言句子。更自然的抽象是状态和动作：

~~~math
s_{t+1}=T(s_t,a_t)
~~~

`s_t` 是第 `t` 步之前的系统状态，`a_t` 是模型提出的动作，`T` 是环境转移函数，`s_{t+1}` 是动作执行后的状态。过程监督可以检查动作参数、权限、执行结果和状态变化，而不是评价一段解释是否“听起来合理”。

例如合同付款助手的状态可能包括已确认的合同编号、收款方、币种和审批状态。模型提出“付款 100,700 元”只是一个动作候选；真正的过程标签还要确认金额是否来自原文、税率是否匹配、收款账户是否有权限、模拟执行后状态是否符合预期。

### 1.4 第一个错误比错误总数更有诊断价值

一旦某一步出错，后面的步骤常常只是沿着错误前提继续计算。把后面每一步都标成独立错误，会夸大问题数量，也会让训练目标混淆“根因”和“后果”。

如果第 `i` 条轨迹中存在错误，第一处错误位置可以写成：

~~~math
j_i^{\mathrm{first}}
=
\min\{j:q_{ij}=0\}
~~~

如果整条轨迹没有错误，可以用特殊值 `None` 或一个独立的“无错误”标签表示。这个标签不能简单当作普通的整数位置，否则“无错误”和“第 0 步错误”可能被错误合并。

第一处错误有三个用途。它可以帮助人类定位根因，可以构造更有针对性的 hard negative，也可以用于评估过程模型是否真的会找到错误，而不是只给整条长轨迹一个模糊低分。

## 2. 过程步骤究竟是什么

### 2.1 三种常见的步骤切分

“一步”不是自然界中固定存在的单位。不同切分方式会产生不同训练样本和不同指标。

第一种是按句子切分。它实现简单，标注员容易操作，但一条长句可能同时包含读题、建模和计算三个动作，导致一个标签无法表达内部差异。

第二种是按运算切分。例如把“先求净速度，再用容量除以净速度”拆成两个步骤。它更适合代数题，但需要标注员理解运算边界。

第三种是按可验证状态切分。每一步都应当让系统状态发生一个可检查的变化，例如“已经确认税率为 6%”“测试集中的边界样例通过”。这种切分适合工具和搜索任务，却需要先设计状态 schema。

### 2.2 粒度太粗和太细的两种失败

粒度太粗时，一步内部可能同时有正确和错误内容。标签只能选择“整步正确”或“整步错误”，结果既不能精确训练，也不能可靠定位。

粒度太细时，标注成本会急剧上升。把一段自然语言拆成每个 token，再要求人类判断每个 token 是否有贡献，并不会自然地产生更好的监督；许多 token 只有在上下文中才有意义。

实用的粒度通常是“一个可独立检查的逻辑单元”。这不是一条普适规则，而是一个设计目标。数据集应当记录步骤切分规范和例外情况，否则不同标注员会用不同的标准理解“正确的一步”。

### 2.3 步骤必须带上下文

同一句话在不同题目中可能真假不同。`x = 5` 不能脱离前置条件判断；“把两个数量相加”也不能脱离单位和题目关系判断。

因此过程评分器通常看到：

~~~text
题目 + 已确认的步骤前缀 + 当前步骤
~~~

只给当前句子而不提供题目和前缀，模型可能学到表面语法，而不是判断这一步是否承接了前面的状态。对长上下文任务，还要明确哪些历史状态已经被压缩，压缩是否丢失了约束。

### 2.4 一步需要满足哪些属性

对于一个需要审计的推理步骤，至少可以分别问四件事。

第一，它是否正确。公式是否成立，事实是否有证据，动作参数是否符合 schema。

第二，它是否相关。它是否推进了目标，而不是添加无关常识。

第三，它是否足够。它是否包含当前结论所需的关键前提，例如单位转换和边界条件。

第四，它是否可追溯。读者能否从输入、工具结果或前置状态复核这一步。

不要把这四个问题压成一个含义不明的“质量分”。如果确实需要一个综合分数，应保留各个子分数，后续才能知道模型是算错了、缺证据，还是只写得不够简洁。

## 3. 过程数据的结构与标注协议

### 3.1 一个可扩展的数据记录

可以把一条过程监督样本抽象成：

~~~math
p_i=(x_i,z_i,y_i^\star,\ell_i,e_i,c_i)
~~~

其中 `x_i` 是输入，`z_i` 是步骤序列，`y_i^star` 是期望答案，`ell_i` 是步骤标签集合，`e_i` 是错误类型和第一处错误信息，`c_i` 是标注成本与置信度等元数据。

一个教学 JSON 可以这样组织：

~~~json
{
  "id": "water_tank_001",
  "question": "进水 3 升/分钟，出水 1 升/分钟，增加 20 升需要多久？",
  "steps": [
    {
      "id": "s1",
      "text": "净速度 = 3 - 1 = 2 升/分钟",
      "correct": true,
      "relevant": true,
      "error_type": null,
      "confidence": 0.99
    },
    {
      "id": "s2",
      "text": "时间 = 20 / 2 = 10 分钟",
      "correct": true,
      "relevant": true,
      "error_type": null,
      "confidence": 0.98
    }
  ],
  "answer": "10 分钟",
  "answer_correct": true,
  "first_error": null,
  "source": "expert_review"
}
~~~

这个 schema 不是唯一标准。它的价值在于把不同层次的信息分开：步骤文本、步骤标签、最终答案、错误位置和来源不应混在一列自由文本里。

### 3.2 错误类型要服务于改进

错误类型不是越多越好。一个可用的起始分类可以包括：

- 读题错误：忽略否定、比较关系或时间范围；
- 条件抽取错误：把输入中的数字、单位或实体记错；
- 建模错误：选择了不适用的公式或算法；
- 推导错误：代数变换或逻辑蕴含不成立；
- 计算错误：算术、索引或边界实现出错；
- 单位错误：米和千米、含税和未税、秒和毫秒混用；
- 证据错误：引用不能支持结论，或把缺失证据当成事实；
- 协议错误：字段缺失、类型不符或工具参数非法；
- 安全错误：越过权限、泄露隐私或执行未经授权的动作。

分类的目的不是给报告增加漂亮的标签，而是让下一轮数据收集和修复有方向。如果所有错误最终都被标成“其他”，分类就没有产生工程价值。

### 3.3 标注员应该先看什么

可靠的标注协议需要规定顺序。一个常见顺序是先读输入和任务契约，再读完整轨迹，最后从前往后判断每一步。标注员不应先看最终答案再倒推过程，否则容易产生确认偏差：看到正确答案，就倾向于原谅过程中的错误。

对于每一步，标注员可以先判断“这一步是否依赖了一个已经错误的前提”。如果前提已错，后续步骤可能被标为“受前置错误影响”，而不是重新标成新的根因错误。不同数据集可以采用不同标签，但必须在说明中固定含义。

### 3.4 多个标注员和分歧记录

过程标签往往比最终答案标签更容易分歧。两个专家可能都同意最终答案，却对“是否需要写出中间恒等式”有不同看法；一个人认为某步是冗余，另一个人认为它是可审计证据。

不要只保留多数投票后的最终标签。还应记录标注员数量、原始标签、置信度、争议原因和最终裁决。分歧本身能告诉我们：步骤定义是否含糊，任务契约是否不完整，还是需要领域专家。

如果两个标注员对同一二分类标签的同意比例为 `p_agree`，可以用简单的观测一致率描述分歧；对于更严谨的研究，还可以使用 Cohen's kappa 或 Krippendorff's alpha。但任何一致性指标都不能替代阅读争议样本，因为一个统一地误标的协议也可能拥有很高一致率。

## 4. PRM：Process Reward Model

### 4.1 PRM 在判断什么

Process Reward Model，简称 PRM，是对步骤、步骤前缀或中间状态进行评分的模型。一个典型输入是：

~~~text
题目 x + 前缀 z_1,...,z_{j-1} + 当前步骤 z_j
~~~

输出可以是当前步骤正确的概率、一个排序分数，或多个维度的预测，例如正确性、相关性和风险。它不一定生成下一步，也不一定知道最终答案；它的角色是提供判断信号。

可以写成：

~~~math
s_{ij}=f_\phi(x_i,z_{i,1:j})
~~~

`f_phi` 是参数为 `phi` 的过程模型，`s_ij` 是截至第 `j` 步的分数。这里的分数不自动具有概率意义。只有经过校准并在目标候选分布上验证后，才可以把它解释成近似成功概率。

### 4.2 PRM 和 ORM 的边界

Outcome Reward Model，简称 ORM，评价最终答案或完整轨迹；PRM 评价中间步骤或前缀。ORM 的反馈更稀疏，但目标通常更接近最终任务；PRM 的反馈更密集，但更容易受到步骤切分、标签噪声和局部偏差影响。

如果一条轨迹的最终答案正确，ORM 可能给高分；PRM 仍然可以发现其中间过程存在不可靠步骤。反过来，PRM 判断每一步看似合理，ORM 仍可能因为全局约束、格式或隐藏测试失败而拒绝整条轨迹。

过程监督不是让 PRM 取代 ORM，而是让两者在不同层次提供互补证据。

### 4.3 如何聚合步骤分数

假设每一步有分数 `s_ij`，整条轨迹可以用平均分表示：

~~~math
S_{\mathrm{avg}}(z_i)
=
\frac{1}{M_i}
\sum_{j=1}^{M_i}s_{ij}
~~~

这里要求 $M_i>0$，且每个 $s_{ij}$ 都是有限数；空轨迹的平均分未定义，不能用 0 代替。平均分适合描述整体趋势，但会稀释一个关键错误。另一种选择是最弱步骤：

~~~math
S_{\mathrm{min}}(z_i)
=
\min_{1\le j\le M_i}s_{ij}
~~~

同样要求 $M_i>0$。最小分适合“任何关键步骤出错都不能接受”的任务，却对偶然的低分和噪声很敏感。还可以使用加权和：

~~~math
S_{\mathrm{weighted}}(z_i)
=
\sum_{j=1}^{M_i}w_{ij}s_{ij},
\qquad
\sum_{j=1}^{M_i}w_{ij}=1
~~~

这里要求权重是有限的非负数，且步骤数非零、权重和为正，再将权重归一化到和为 1。`w_ij` 可以提高关键步骤、证据引用或安全检查的权重。但权重不是免费选择：它把业务判断写进了聚合函数，应当在验证集和风险切片上说明依据。

如果把分数当作每一步正确概率，还可以考虑对数聚合：

~~~math
S_{\mathrm{log}}(z_i)
=
\sum_{j=1}^{M_i}\log\max(s_{ij},\epsilon)
~~~

这里要求 $M_i>0$、$s_{ij}$ 为有限非负数，且 $\epsilon>0$；如果 `s_ij` 被解释为概率，还应满足 $s_{ij}\le 1$。它会惩罚低概率步骤，但也会明显偏好短轨迹。比较不同长度轨迹时，必须说明是否做了长度归一化，否则模型可能通过少写步骤获得更高分。

### 4.4 过程分数与最终答案不能脱钩

一个简单的混合评分可以写成：

~~~math
S_{\mathrm{final}}
=
\alpha S_{\mathrm{process}}
+\beta S_{\mathrm{outcome}}
+\gamma S_{\mathrm{evidence}}
-\delta S_{\mathrm{risk}}
~~~

`S_process` 是过程质量，`S_outcome` 是最终结果，`S_evidence` 是证据支持，`S_risk` 是风险惩罚。`alpha`、`beta`、`gamma`、`delta` 不是论文中的通用常数，而是特定业务的决策参数。

在可执行代码任务中，`S_outcome` 可以是测试结果；在合同任务中，`S_evidence` 可能来自页码和原文片段；在高风险工具动作中，`S_risk` 不能只由语言模型自己判断，而应由独立权限系统提供。

## 5. 训练过程模型

### 5.1 Pointwise：每一步单独分类

如果每一步都有二值标签 `q_ij`，可以使用二元交叉熵：

~~~math
\mathcal{L}_{\mathrm{point}}
=
-q_{ij}\log\sigma(s_{ij})
-(1-q_{ij})\log(1-\sigma(s_{ij}))
~~~

`s_ij` 是模型输出的 logit，`sigma` 是 sigmoid 函数。该损失要求 `q_ij` 是 0 或 1，logit 和损失项为有限数，并且统计时至少有一个有效步骤。工程实现应使用数值稳定的 binary-cross-entropy-with-logits，避免极端 logit 先经过 sigmoid 再取对数造成下溢。Pointwise 目标容易实现，适合标签含义稳定的任务；问题是正负标签的边界可能很主观，而且不同问题的分数尺度不一定可比较。

如果某一步的标签只是“受前一步错误影响”，而不是明确的独立错误，就不应把它无条件当作负类。数据协议可以增加第三类 `downstream_affected`，或者在损失中降低这类样本的权重。

### 5.2 Pairwise：让正确前缀排在错误前缀前

在许多偏好数据中，人类更容易回答“哪一个步骤更好”，而不是为每一步打一个绝对分数。设较优步骤分数为 `s_plus`，较差步骤分数为 `s_minus`，常见的 logistic ranking loss 是：

~~~math
\mathcal{L}_{\mathrm{pair}}
=
-\log\sigma(s_{+}-s_{-})
~~~

这个目标只保证相对顺序，不保证 `s_plus = 0.9` 真的是 90% 成功率。它适合排序和搜索，却不能直接用于概率阈值、风险拒答或成本决策，除非额外校准。

### 5.3 关键步骤加权

如果一条轨迹有很多普通步骤和少数关键步骤，平均损失会让关键步骤的影响被稀释。可以设步骤权重 `w_ij`：

~~~math
\mathcal{L}_{\mathrm{weighted}}
=
\frac{1}{\sum_{i,j}w_{ij}}
\sum_i\sum_{j=1}^{M_i}
w_{ij}\,\mathcal{L}_{ij}
~~~

这个式子要求权重有限、非负，且总权重大于 0；否则分母为零，损失未定义。权重可以来自标注员对关键性的判断，也可以来自任务结构，例如最终付款金额、权限字段和安全确认的权重高于普通说明文字。权重过大则会让模型忽略其他步骤，必须通过消融实验验证。

### 5.4 负例不是随机乱码

一个只包含明显乱码的训练集，会让 PRM 学到“格式像不像”，而不是“推理是否正确”。高价值负例包括：

- 读题完全正确，但在最后一步算错；
- 每个局部计算都像对的，但漏掉一个边界条件；
- 引用了真实文档，却把文档结论解释反了；
- 使用正确公式，但把单位混用；
- 最终答案正确，过程通过错误抵消得到；
- 过程很长、格式整齐、语气自信，但关键结论没有证据；
- 只改变表达风格，保持任务语义不变，用来测试格式偏差。

这些负例比“答案是乱码”更接近生成模型在真实系统中产生的错误。

## 6. 数据从哪里来

### 6.1 人工步骤标注

人工标注适合开放式数学推导、证据支持、规划和安全判断，因为这些任务往往不能由一个简单程序完全决定。它的缺点是成本高、速度慢，而且领域专家之间也会有分歧。

标注成本至少包括阅读题目、理解步骤、确认外部事实、判断第一处错误和记录理由。如果任务需要核对引用，标注员还要打开原文或工具结果。只统计“点击一个标签”的数量，会严重低估真实成本。

### 6.2 规则和程序产生的标签

代码编译器、单元测试、符号计算器、JSON Schema、单位换算器和数据库约束都可以提供过程或结果信号。程序标签的优势是可重复、便宜、能处理大量候选；局限是它只能检查被写进规则的约束。

例如，代码通过公开测试不等于对所有输入正确；JSON 解析成功不等于付款账户合法；符号等式成立不等于题目条件抽取正确。程序监督仍然需要测试集设计和验证器本身的测试。

### 6.3 模型初标和人工复核

可以让一个模型先标注候选步骤，再由人类重点复核低置信度、模型间分歧和高风险样本。这样的流程能降低成本，但初标模型会把自己的偏差带入数据。

复核样本不能只随机抽取。还应按错误类型、轨迹长度、生成器版本、温度、领域和风险等级分层，否则数据集可能在平均指标上很好，却覆盖不到最危险的错误。

### 6.4 PRM800K 能说明什么

PRM800K 是公开过程监督研究背景中的重要数据资源。读者可以通过公开仓库了解其数据组织和使用入口，但不应把仓库名称理解为“所有任务都有 800,000 个高质量、完全无争议的步骤标签”。真实数据仍然有任务范围、标注策略和研究设置。

更稳妥的阅读方式是：论文支持某项研究在其数据和实验设置下的结论；仓库支持其中公开文件和代码的存在；至于换成另一种模型、另一种题目或另一种候选分布后是否同样有效，需要重新实验。

## 7. 第一处错误与错误传播

### 7.1 为什么不能把后果当成根因

看一条错误轨迹：

~~~text
步骤 1：正确读出速度是 3 和 1。
步骤 2：把净速度写成 3 + 1 = 4。
步骤 3：用 20 / 4 得到 5 分钟。
步骤 4：把 5 分钟换算成 300 秒。
~~~

步骤 2 是第一处错误。步骤 3 的计算在它自己的前提下是正确的，步骤 4 的单位换算也可能正确。若训练数据把步骤 2、3、4 都当作独立错误，模型学到的不是根因定位，而是“只要后面结果不对，整段都要低分”。

### 7.2 三种错误标签策略

第一种策略是根因标签：只把第一处违反任务约束的步骤标为错误，后续步骤标为受影响。它适合错误诊断和数据修复。

第二种策略是语义正确性标签：每一步都独立判断其局部陈述是否成立。它适合细粒度分类，但需要标注员接受“基于错误前提的局部计算可能仍然算术正确”这一约定。

第三种策略是可执行状态标签：只判断执行该步骤后状态是否仍然满足全局约束。它适合工具和程序任务，却要求有明确的环境状态。

三种策略没有谁天然正确。最危险的是数据集没有说明自己采用了哪一种，却把不同策略的标签混在一起训练。

### 7.3 第一处错误检测指标

设真实第一处错误为 `j_i^first`，模型预测为 `hat{j}_i^first`。最严格的准确率是：

~~~math
A_{\mathrm{first}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}
\left[\hat{j}_i^{\mathrm{first}}=j_i^{\mathrm{first}}\right]
~~~

这里要求样本数 N>0，并且“无错误”作为独立类别参与完全匹配；不能把 `None` 随意转换为 0。位置差一格和差十步的错误严重程度不同，可以报告平均绝对位置误差：

~~~math
E_{\mathrm{pos}}
=
\frac{1}{N'}
\sum_{i\in\mathcal{E}}
\left|\hat{j}_i^{\mathrm{first}}-j_i^{\mathrm{first}}\right|
~~~

`E` 是存在真实错误且模型给出合法位置的样本集合；只有 $N'>0$ 时，$E_{pos}$ 才有定义。位置必须是从 1 开始的整数，模型漏报或误报应另报。对于“无错误”的轨迹，最好单独报告误报率，不要把特殊值硬塞进位置距离。

## 8. 自动过程监督

### 8.1 代码任务

代码任务提供了较强的自动信号。编译结果可以指出语法和类型问题；单元测试可以检查输入输出；静态分析可以检查未定义变量、越界和危险调用；不变量检查可以验证排序、唯一性或资源释放。

不过自动测试是抽样约束。一个候选程序可能通过公开测试，却在隐藏边界、并发、资源限制或恶意输入上失败。因此过程监督应同时记录测试覆盖和测试环境，而不是只记录一个 `pass`。

### 8.2 数学与符号任务

数学表达式可以经过解析、化简、数值抽样或形式化证明。符号系统能够验证某些等式和不等式，但它不一定知道模型是否正确理解了题目中的自然语言条件。

例如，模型从“至少 3 个”读成“恰好 3 个”，后面所有代数步骤可能都能被符号工具验证。自动工具证明了推导在错误前提下成立，却没有证明前提来自题目。

### 8.3 引用和文档任务

文档问答的过程监督可以检查：实体是否来自检索片段，数字是否与原文一致，引用页码是否覆盖结论，是否把两个版本的合同混在一起。它不能只检查“引用字符串存在”，因为存在的引用可能不支持当前句子。

可以为每条 claim 记录支持状态：

~~~math
u_k\in\{\mathrm{supported},\mathrm{contradicted},\mathrm{unknown}\}
~~~

`unknown` 不是失败的同义词，而是证据不足。把 unknown 强行转成 false 会鼓励系统在证据缺失时随便猜一个相反结论。

### 8.4 混合自动和人工信号

实际系统通常采用混合方法：能由程序确定的约束交给程序，开放判断交给模型和人类，冲突样本进入复核。自动标注覆盖率可以写成：

~~~math
C_{\mathrm{auto}}
=
\frac{\sum_i\sum_j a_{ij}}
{\sum_i M_i}
~~~

这里要求总步骤数大于 0，且 `a_ij` 是 0/1 标签。`a_ij = 1` 表示第 `i` 个样本第 `j` 步有可靠的自动标签。覆盖率高不等于数据质量高；如果自动规则只覆盖简单步骤，剩下的难例仍可能决定线上效果。

人工成本可以用：

~~~math
C_{\mathrm{human}}
=
\sum_i\sum_j(1-a_{ij})c_{ij}
~~~

`c_ij` 应为有限的非负成本，并包含阅读、工具核验、复核和争议裁决的成本，而不是只计算一次点击。

## 9. 过程监督和搜索

### 9.1 为什么搜索需要中间评分

如果只在完整答案生成后检查，搜索树中的大量错误分支要走到叶子才会被发现。过程评分可以在树的中间节点做初步筛选。

设当前保留的前缀集合为 `B_t`，每个前缀 `b` 可以扩展出动作集合 `A(b)`。一个简化的 beam 更新是：

~~~math
B_{t+1}
=
\operatorname{TopK}
\left(
\left\{b\mathbin{+}a:\ b\in B_t,\ a\in A(b)\right\},
S_{\mathrm{process}}
\right)
~~~

这个更新要求当前 beam 和至少一个可扩展动作非空，`K` 是正整数；若候选集合为空，下一层不是“准确率为 0”，而是搜索失败或状态不可扩展。它表示对所有候选下一步评分，只保留分数最高的 `K` 个前缀。这里的 `K` 是 beam 宽度，不是候选总数。

### 9.2 过早剪枝的代价

PRM 对一个正确但表达不熟悉的步骤打低分，搜索就会把正确分支剪掉；对一个措辞漂亮但错误的步骤打高分，错误分支就会占据 beam。过程监督把反馈提前了，也把评分器的错误提前放大了。

因此不能只报告“PRM 预测很准”。还要测：

- 正确分支被错误剪掉的比例；
- 搜索后最终答案准确率；
- beam 宽度增加时的收益和成本；
- 不同模型、温度和任务类型的候选分布；
- hard negative 是否会占据前几名。

### 9.3 保留多样性

如果所有 beam 都共享同一个错误前缀，增加 beam 宽度也没有用。可以按不同方法、实体选择或结构状态分组，限制相同前缀的数量，或者在分数接近时保留多样候选。

多样性和质量之间需要单独测量。一个简单的候选重复率是：

~~~math
R_{\mathrm{dup}}
=
1-
\frac{|\mathrm{unique}(B)|}{|B|}
~~~

这里要求当前候选集合 $B$ 非空；空集合的重复率未定义。重复率高说明搜索预算可能被相同路径消耗；重复率低也不自动代表候选质量高，因为大量不同的错误路径仍然可能没有价值。

### 9.4 PRM 的不确定性

当两个候选分数非常接近时，强行只保留一个可能不划算。可以记录 top-1 和 top-2 的分数间隔：

~~~math
\Delta_{\mathrm{prm}}
=s_{(1)}-s_{(2)}
~~~

这里至少需要两个可比较且有限的分数；少于两个候选时，间隔未定义。间隔小不等于第二个候选正确，但它说明评分依据弱。系统可以保留两个候选，调用程序检查，或继续收集证据。这个设计要配合成本预算，不能无限保留所有分支。

## 10. 过程奖励与强化学习

### 10.1 稀疏奖励和密集奖励

结果奖励通常只在轨迹结束时给出：答案正确为 1，错误为 0。过程奖励允许每一步得到反馈 `r_t`。在折扣因子 `gamma` 下，回报是：

~~~math
G_t
=
\sum_{k=0}^{T-t-1}
\gamma^k r_{t+k+1}
~~~

这里要求 $0\le\gamma\le1$、$T>t$，并且参与回报的奖励是有限数；若没有后续奖励，回报不是一个可比较的普通数值。密集反馈可能改善 credit assignment，让系统更容易知道哪一步贡献了结果；但它也让奖励模型的偏差进入每一个时间点。一个错误的过程奖励比一个错误的最终标签更频繁地影响训练。

### 10.2 过程奖励不能替代结果奖励

可以把两者组合：

~~~math
R
=
\lambda R_{\mathrm{process}}
+(1-\lambda)R_{\mathrm{outcome}}
~~~

这里要求 $0\le\lambda\le1$，两个奖励在可比较尺度上且为有限数。`lambda` 越大，训练越重视局部过程；越小，越重视最终任务。这个组合只是一种教学表达，实际系统还要考虑奖励尺度、长度归一化、终止条件和安全约束。

过程奖励的常见风险是模型学会了“让每一步看起来像高分步骤”，却没有提高最终能力。例如它可能在每一步都重复“因此结论可靠”，使 PRM 感到自信，却没有增加证据。独立 outcome verifier、隐藏测试和人工抽查仍然必要。

### 10.3 奖励塑形和目标错配

如果真实任务质量为 `Q_task`，过程模型分数为代理目标 `R_proxy`，训练优化 `R_proxy` 后，不能直接推出任务质量提高：

~~~math
\Delta R_{\mathrm{proxy}}>0
\quad\not\Rightarrow\quad
\Delta Q_{\mathrm{task}}>0
~~~

这不是说过程奖励没有用，而是说它需要独立验证。尤其在长轨迹和开放任务中，代理目标和真实目标之间可能出现越来越大的偏差。

## 11. 过程监督的评估

### 11.1 步骤分类准确率

如果每个步骤都有真实标签 `q_ij` 和预测标签 `hat{q}_ij`，步骤准确率是：

~~~math
A_{\mathrm{step}}
=
\frac{
\sum_i\sum_{j=1}^{M_i}
\mathbb{1}[\hat q_{ij}=q_{ij}]
}{
\sum_i M_i
}
~~~

它容易理解，却可能被大量简单步骤主导。一个数据集里 95% 的步骤都是正确标签，模型全部预测“正确”也会获得很高准确率。因此应同时报告错误类召回率、混淆矩阵和按任务分组的结果。

### 11.2 第一处错误检测

第一处错误准确率直接衡量根因定位。还可以报告错误样本上的召回率和误报率：模型是否会把无错误轨迹误判为错误，是否会在长轨迹中漏掉早期错误。

如果模型把错误定位在第 5 步，而真实错误在第 4 步，是否完全失败取决于使用场景。用于自动修复时，一步误差可能仍然有帮助；用于精确数据分析时，则可能不够。指标必须和下游动作对应。

### 11.3 排序和 top-1 选择

可以在正确步骤和错误步骤之间计算 pairwise accuracy：

~~~math
A_{\mathrm{pair}}
=
\frac{1}{|\mathcal{P}|}
\sum_{(p^+,p^- )\in\mathcal{P}}
\mathbb{1}[s(p^+)>s(p^-)]
~~~

这里要求比较对集合非空，且每一对的两个分数为有限数；并列分数应另报 tie，而不是默认为成功。真正部署时通常是从一组候选中选择一个。设第 `i` 题的候选集合为 `C_i`，PRM 选出的候选为：

~~~math
j_i^*=\arg\max_{j\in C_i}s_{ij}
~~~

下游选择准确率是：

~~~math
A_{\mathrm{select}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[y_{i,j_i^*}=y_i^\star]
~~~

这里要求题目数 N>0，每个 `C_i` 非空，并且最高分并列时有固定规则；若没有可交付候选，应单独统计 abstain。pairwise 高而 `A_select` 低是可能的，因为候选数量、相关错误和分数尺度会改变列表选择行为。

### 11.4 校准

如果 PRM 输出被解释为成功概率，就要检查预测置信度和真实频率是否匹配。分桶 ECE 可以写成：

~~~math
\operatorname{ECE}
=
\sum_{b=1}^{B}
\frac{|S_b|}{n}
\left|
\operatorname{acc}(S_b)-\operatorname{conf}(S_b)
\right|
~~~

`S_b` 是第 `b` 个分数桶，`acc` 是桶内真实成功率，`conf` 是平均预测分数。计算 ECE 要求 n>0，且被解释为概率的分数位于 [0,1]；空桶可以跳过，但要记录桶覆盖。若没有任何有效样本，ECE 未定义。ECE 会受分桶方式影响，不能单独证明校准正确。还应查看可靠性图、Brier score、不同长度和不同风险切片的校准。

### 11.5 最终任务和成本

过程指标只是中间指标。最终还要报告使用过程监督前后的最终任务成功率、搜索延迟、候选 token、程序执行次数、人工复核次数和单位成功成本：

~~~math
C_{\mathrm{success}}
=
\frac{
 C_{\mathrm{generate}}
 +C_{\mathrm{prm}}
 +C_{\mathrm{tool}}
 +C_{\mathrm{review}}
}{
 N_{\mathrm{successful}}
}
~~~

这里要求各项成本为同一单位下的有限非负数，且 N_successful>0。如果评估窗口一次都没有成功，单位成功成本未定义，应报告为 `None` 或“无成功”，不能用 `max(1, ...)` 把它伪装成一个成本。如果 PRM 让准确率提高 1 个百分点，却让每次成功任务的成本提高 20 倍，是否值得采用取决于任务价值和风险，不能只看准确率曲线。

## 12. 常见失败模式

### 12.1 过程越长，分数越高

如果训练正例普遍比负例长，PRM 可能把长度当成质量。生成器就会添加无关解释，甚至重复同一个结论。评估时应做等长度对比、长度分桶和质量—token 曲线。

### 12.2 格式比内容重要

如果所有正例都使用编号列表和固定标题，所有负例都使用自然段，评分器可能只识别格式。需要交叉改写正确和错误候选，并让同一种格式同时出现在正负样本中。

### 12.3 自信语气偏差

“显然”“一定正确”“已经验证”等词语不等于证据。应当故意构造语气自信但结论错误的 hard negative，检查分数是否受到措辞影响。

### 12.4 多种正确解法被拒绝

数学题可能有代数法、枚举法和几何法。只用一条参考路径训练，会让过程模型把“没有按照参考顺序写”当成错误。数据应包含多种可证明正确的轨迹，或者用结果和规则验证辅助判断。

### 12.5 步骤边界漂移

同一个模型升级后，可能从短句改成段落；同一条逻辑被切成两步或合成一步，旧 PRM 的输入分布就变了。训练和评估都要记录生成器版本、步骤切分器和 prompt 版本。

### 12.6 过程看起来忠实，不代表内部机制透明

模型可以生成一条合理的解释，却没有真正依赖它完成答案。过程监督提升的是可检查性和可训练性，不自动等于机制解释。对解释忠实性的判断需要干预实验、输入替换、工具轨迹和独立答案验证。

### 12.7 局部正确但全局错误

每个局部动作都合法，组合后可能访问错误账户、漏掉时间条件或违反资源约束。过程评估必须保留全局状态和终局 outcome 检查。

### 12.8 过程模型和生成模型互相适应

生成器会逐渐学会评分器喜欢的表面模式。旧 hard negative 很快过时，新的生成器分布也可能包含新错误。应按生成器版本持续采样、复核和更新评估集。

## 13. 综合案例：合同金额审阅助手

### 13.1 任务契约

助手需要从合同中抽取基础金额、折扣、税率，计算含税金额，并给出支持每个数字的页码。最终输出要求是：

~~~json
{
  "base_amount": 100000,
  "discount": 0.05,
  "tax_rate": 0.06,
  "total": 100700,
  "citations": ["page_3", "page_4", "page_7"],
  "status": "needs_review"
}
~~~

这里至少有四类可监督过程：检索到正确条款，抽取正确字段，执行正确计算，生成支持结论的引用。最终 JSON 可解析只是最浅的一层。

### 13.2 两条候选过程

~~~text
候选 A：
1. 从第 3 页抽取基础金额 100000 元。
2. 从第 4 页抽取折扣 5%，因此折后金额为 100000 × 0.95 = 95000 元。
3. 从第 7 页抽取税率 6%，含税金额为 95000 × 1.06 = 100700 元。
4. 引用第 3、4、7 页，并把结果标记为需要复核。

候选 B：
1. 抽取基础金额 100000 元。
2. 折扣 5% 计算正确，得到 95000 元。
3. OCR 结果把税率识别为 60%，计算得到 152000 元。
4. 引用了一段没有税率的页脚，并把结果标记为已确认。
~~~

候选 B 的前两步正确，第三步是 OCR 和单位比例错误，第四步又出现证据和状态错误。只用最终答案监督时，B 可能被整体判为错，却无法指出应修复 OCR、引用还是状态协议。过程监督可以把这些问题拆开，但仍需要文档原文和业务规则作为独立证据。

### 13.3 过程标签和证据标签

对于候选 B，可以记录：

~~~text
步骤 1：correct, supported
步骤 2：correct, supported
步骤 3：wrong, unsupported, first_error
步骤 4：wrong, unsupported, safety_or_status_error
~~~

如果第三步的税率来自 OCR 而不是原文，标签还应保留“来源类型”。这样才能分析问题究竟来自视觉识别、检索、计算还是决策状态，而不是把所有事故都归结为“模型推理失败”。

### 13.4 风险分层

普通合同摘要可以允许助手返回低置信度结果；付款金额、收款账户和审批状态则需要更高证据强度。过程分数只能作为辅助信号，不能授予权限。权限判断、金额上限、人工确认和可撤销执行应由独立系统控制。

## 14. 最小可运行实验：过程标签、第一处错误和搜索选择

下面的 demo 不训练真实 PRM，而是构造三条教学轨迹和三个搜索状态。它同时展示几个容易混淆的指标：最终答案准确率、步骤准确率、第一处错误定位、相关步骤比例、自动标注覆盖率、人工标注成本和搜索 top-1 选择。

代码中特意保留两个反例：`lucky_answer` 的最终答案正确但过程有错误；`format_error` 的过程正确但最终答案不符合数值格式要求。搜索部分还保留一个分数最高却错误的 `polished_wrong` 候选，用来说明高分不等于正确。

~~~python
import math


def first_error(labels):
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels must be a non-empty list")
    if any(
        isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1)
        for label in labels
    ):
        raise ValueError("labels must contain only 0 or 1")
    for index, label in enumerate(labels, start=1):
        if label == 0:
            return index
    return None


cases = [
    {
        "id": "clean_solve",
        "gold": "10",
        "answer": "10",
        "steps": [
            {"id": "extract_rate", "correct": 1, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "net_rate", "correct": 1, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "volume", "correct": 1, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "time_formula", "correct": 1, "pred": 1, "relevant": 1, "auto": 0, "cost": 2},
            {"id": "compute_time", "correct": 1, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "final_answer", "correct": 1, "pred": 1, "relevant": 1, "auto": 0, "cost": 2},
        ],
    },
    {
        "id": "lucky_answer",
        "gold": "42",
        "answer": "42",
        "steps": [
            {"id": "read_question", "correct": 1, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "wrong_formula", "correct": 0, "pred": 0, "relevant": 1, "auto": 0, "cost": 3},
            {"id": "bad_substitution", "correct": 0, "pred": 1, "relevant": 1, "auto": 0, "cost": 0},
            {"id": "redundant_plan", "correct": 1, "pred": 1, "relevant": 0, "auto": 0, "cost": 2},
            {"id": "lucky_arithmetic", "correct": 0, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "unsupported_finish", "correct": 0, "pred": 1, "relevant": 1, "auto": 0, "cost": 2},
        ],
    },
    {
        "id": "format_error",
        "gold": "6",
        "answer": "six hours",
        "steps": [
            {"id": "extract_distance", "correct": 1, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "extract_speed", "correct": 1, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "divide", "correct": 1, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "unit_check", "correct": 1, "pred": 1, "relevant": 1, "auto": 0, "cost": 3},
            {"id": "numeric_result", "correct": 1, "pred": 1, "relevant": 1, "auto": 1, "cost": 0},
            {"id": "format_instruction", "correct": 1, "pred": 1, "relevant": 1, "auto": 0, "cost": 2},
        ],
    },
]


search_states = [
    {
        "id": "distractor_state",
        "choices": [
            {"name": "follow distractor", "score": 0.22, "good": False},
            {"name": "ignore distractor", "score": 0.81, "good": True},
        ],
    },
    {
        "id": "logic_state",
        "choices": [
            {"name": "branch_red", "score": 0.45, "good": False},
            {"name": "branch_blue", "score": 0.76, "good": True},
        ],
    },
    {
        "id": "hard_negative_state",
        "choices": [
            {"name": "plain_correct", "score": 0.62, "good": True},
            {"name": "polished_wrong", "score": 0.84, "good": False},
        ],
    },
]


def validate_cases(items):
    if not isinstance(items, list) or not items:
        raise ValueError("cases must be a non-empty list")
    seen_case_ids = set()
    for case in items:
        required_case = {"id", "gold", "answer", "steps"}
        if not isinstance(case, dict) or set(case) != required_case:
            raise ValueError("case schema is invalid")
        case_id = case["id"]
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in seen_case_ids
        ):
            raise ValueError("case ids must be unique non-empty strings")
        seen_case_ids.add(case_id)
        for key in ("gold", "answer"):
            if not isinstance(case[key], str) or not case[key]:
                raise ValueError(f"{key} must be a non-empty string")
        steps = case["steps"]
        if not isinstance(steps, list) or not steps:
            raise ValueError("each case needs non-empty steps")
        seen_step_ids = set()
        for step in steps:
            required_step = {
                "id",
                "correct",
                "pred",
                "relevant",
                "auto",
                "cost",
            }
            if not isinstance(step, dict) or set(step) != required_step:
                raise ValueError("step schema is invalid")
            step_id = step["id"]
            if (
                not isinstance(step_id, str)
                or not step_id
                or step_id in seen_step_ids
            ):
                raise ValueError("step ids must be unique non-empty strings")
            seen_step_ids.add(step_id)
            for key in ("correct", "pred", "relevant", "auto"):
                value = step[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value not in (0, 1)
                ):
                    raise ValueError(f"{key} must be 0 or 1")
            cost = step["cost"]
            if (
                isinstance(cost, bool)
                or not isinstance(cost, (int, float))
                or not math.isfinite(cost)
                or cost < 0
            ):
                raise ValueError("cost must be a finite non-negative number")


def validate_search_states(states):
    if not isinstance(states, list) or not states:
        raise ValueError("search_states must be a non-empty list")
    seen_ids = set()
    for state in states:
        required_state = {"id", "choices"}
        if not isinstance(state, dict) or set(state) != required_state:
            raise ValueError("search state schema is invalid")
        state_id = state["id"]
        if (
            not isinstance(state_id, str)
            or not state_id
            or state_id in seen_ids
        ):
            raise ValueError("search state ids must be unique non-empty strings")
        seen_ids.add(state_id)
        choices = state["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("each search state needs non-empty choices")
        for choice in choices:
            required_choice = {"name", "score", "good"}
            if not isinstance(choice, dict) or set(choice) != required_choice:
                raise ValueError("search choice schema is invalid")
            if not isinstance(choice["name"], str) or not choice["name"]:
                raise ValueError("choice name must be non-empty")
            score = choice["score"]
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise ValueError("choice score must be finite")
            if not isinstance(choice["good"], bool):
                raise TypeError("choice good must be boolean")


validate_cases(cases)
validate_search_states(search_states)
outcome_correct = [case["answer"] == case["gold"] for case in cases]
all_steps = [step for case in cases for step in case["steps"]]
step_matches = [step["correct"] == step["pred"] for step in all_steps]

outcome_accuracy = round(sum(outcome_correct) / len(cases), 3)
if not all_steps:
    raise ValueError("all_steps must be non-empty")
step_accuracy = round(sum(step_matches) / len(step_matches), 3)

true_first_errors = [
    first_error([step["correct"] for step in case["steps"]])
    for case in cases
]
pred_first_errors = [
    first_error([step["pred"] for step in case["steps"]])
    for case in cases
]
first_error_accuracy = round(
    sum(true == predicted for true, predicted in zip(true_first_errors, pred_first_errors))
    / len(cases),
    3,
)

relevant_step_ratio = round(
    sum(step["relevant"] for step in all_steps) / len(all_steps),
    3,
)
auto_label_coverage = round(
    sum(step["auto"] for step in all_steps) / len(all_steps),
    3,
)
human_label_cost = sum(
    0 if step["auto"] else step["cost"]
    for step in all_steps
)

final_correct_bad_process = [
    case["id"]
    for case in cases
    if case["answer"] == case["gold"]
    and any(step["correct"] == 0 for step in case["steps"])
]
process_correct_bad_final = [
    case["id"]
    for case in cases
    if case["answer"] != case["gold"]
    and all(step["correct"] == 1 for step in case["steps"])
]
redundant_step_ids = [
    step["id"] for step in all_steps if step["relevant"] == 0
]

search_choices = []
for state in search_states:
    choice = max(
        enumerate(state["choices"]),
        key=lambda item: (item[1]["score"], -item[0]),
    )[1]
    search_choices.append(
        {"id": state["id"], "choice": choice["name"], "good": choice["good"]}
    )

search_top1_accuracy = round(
    sum(choice["good"] for choice in search_choices) / len(search_choices),
    3,
)
search_failures = [choice["id"] for choice in search_choices if not choice["good"]]

review = {
    "step_quality_ok": step_accuracy >= 0.8,
    "first_error_ok": first_error_accuracy >= 0.9,
    "auto_coverage_ok": auto_label_coverage >= 0.5,
    "search_selection_ok": search_top1_accuracy >= 0.8,
    "cost_ok": human_label_cost <= 20,
    "outcome_blind_spots": bool(final_correct_bad_process or process_correct_bad_final),
}

print(f"outcome_accuracy={outcome_accuracy}")
print(f"step_accuracy={step_accuracy}")
print(f"first_error_accuracy={first_error_accuracy}")
print(f"relevant_step_ratio={relevant_step_ratio}")
print(f"auto_label_coverage={auto_label_coverage}")
print(f"human_label_cost={human_label_cost}")
print(f"final_correct_bad_process={final_correct_bad_process}")
print(f"process_correct_bad_final={process_correct_bad_final}")
print(f"redundant_step_ids={redundant_step_ids}")
print(f"search_top1_accuracy={search_top1_accuracy}")
print(f"search_choices={search_choices}")
print(f"search_failures={search_failures}")
print(f"review={review}")
~~~

预期输出如下：

~~~text
outcome_accuracy=0.667
step_accuracy=0.833
first_error_accuracy=1.0
relevant_step_ratio=0.944
auto_label_coverage=0.556
human_label_cost=16
final_correct_bad_process=['lucky_answer']
process_correct_bad_final=['format_error']
redundant_step_ids=['redundant_plan']
search_top1_accuracy=0.667
search_choices=[{'id': 'distractor_state', 'choice': 'ignore distractor', 'good': True}, {'id': 'logic_state', 'choice': 'branch_blue', 'good': True}, {'id': 'hard_negative_state', 'choice': 'polished_wrong', 'good': False}]
search_failures=['hard_negative_state']
review={'step_quality_ok': True, 'first_error_ok': True, 'auto_coverage_ok': True, 'search_selection_ok': False, 'cost_ok': True, 'outcome_blind_spots': True}
~~~

逐项解释这个结果。`outcome_accuracy=0.667` 是三个样本中有两个最终答案符合字符串标签；它没有发现 `lucky_answer` 的过程错误。`step_accuracy=0.833` 说明这个 toy 预测器大多数步骤判断正确，但并不代表它能保证完整任务成功。`first_error_accuracy=1.0` 是因为构造数据让它准确找到了每条轨迹的第一处错误；真实系统中这个指标通常会明显下降。

`relevant_step_ratio=0.944` 暗示存在一个正确但冗余的步骤。`auto_label_coverage=0.556` 说明接近一半步骤仍需要人工或其他非自动判断。搜索 top-1 只有 `0.667`，因为 `polished_wrong` 的分数高于 `plain_correct`。这正是过程评分器需要 hard negative 和独立结果检查的原因。

注意 `review` 只是一个教学字典，表示各项观察结果，不是模型内部的发布命令。真实系统应当把这些指标放入实验报告，并根据任务风险、预算和人工复核能力做决定。

## 15. 如何设计一次可靠实验

### 15.1 先定义成功事件

在采集数据之前，先写清楚最终成功是什么。数学题可能要求数值等价；代码题可能要求通过隐藏测试并满足资源限制；合同任务可能要求每个金额都有支持引用，并且输出状态不能超过权限。

如果成功事件含糊，过程标签也会含糊。比如“解释充分”没有定义，就无法判断一个步骤是必要证据还是冗余文本。

### 15.2 固定候选生成分布

比较两个 PRM 时，应尽量使用同一批候选，或者明确记录生成器、模型版本、temperature、top-p、工具状态和步骤切分方式。否则生成器变强或候选变简单，可能被误认为 PRM 变强。

评估集要按时间和生成器版本拆分。旧模型生成的 hard negative 不能代表新模型会产生的错误，新工具接入也会改变步骤状态。

### 15.3 做反事实和消融

把正确候选改成不同格式，把错误候选改成相同长度；交换引用页码但不改变句子；替换单位或否定词；删掉一个关键步骤；把步骤顺序打乱。每次只改变一个因素，观察 PRM 分数变化。

如果只改变标题格式就导致分数大幅变化，说明评分器依赖表面模式。如果只删掉无关解释就分数下降，说明它可能把长度当成质量。

### 15.4 报告局部指标和下游指标

一份完整报告至少应包含步骤标签质量、第一处错误、排序、校准、搜索 top-1、最终任务成功率、拒答或人工复核质量、延迟和成本。只报一个 `step accuracy` 无法回答系统是否真的更好。

### 15.5 记录不确定性

每个硬数字都应说明样本量、置信区间、任务切片和是否来自 toy 构造。对于人类标签，报告标注员数量和分歧；对于程序标签，报告测试覆盖、超时和异常；对于线上结果，报告版本、流量和时间窗口。

## 16. 安全与权限

### 16.1 高分不是执行权限

PRM 认为“删除文件”这一步很合理，不代表模型获得了删除权限。语言评分、工具参数检查和真实执行权限必须分离。

### 16.2 运行候选代码需要沙箱

自动过程监督可能需要执行模型生成的代码。沙箱至少要限制文件系统、网络、子进程、CPU、内存、执行时间和输出大小，并固定依赖版本。超时、异常和资源耗尽应作为独立状态记录，而不是简单计为“错误答案”。

### 16.3 过程文本可能含有敏感信息

推理步骤、检索片段和工具返回值可能包含个人数据、合同金额或访问令牌。数据集构造应做脱敏、访问控制和保留期限管理；日志中不能因为需要“审计过程”就无限保存所有原始内容。

### 16.4 外部内容可以攻击过程评分器

网页、PDF 和代码注释可能包含提示注入，诱导模型把恶意文本当成高优先级步骤。过程监督需要区分“输入中的指令”和“任务要求”，工具权限系统也不能由文档内容直接改写。

### 16.5 高风险任务需要多重证据

付款、删除、权限变更和外部消息发送应同时满足结构化参数、独立业务规则、权限检查、工具结果和必要人工确认。过程分数可以帮助发现异常步骤，但不能单独承担安全责任。

## 17. 小练习

### 练习一：定义步骤

为一道含单位换算的数学题分别设计句子级、运算级和状态级步骤切分。比较三种切分下的标注成本和第一处错误定位能力。

### 练习二：区分错误根因和错误后果

构造一条四步轨迹，其中第 2 步是第一处错误，第 3、4 步只是沿着错误前提继续计算。分别给出根因标签、局部正确性标签和状态标签。

### 练习三：多种正确解法

用两种不同方法解同一道题，并说明一个只模仿参考路径的 PRM 可能在哪一步误判。

### 练习四：设计 hard negative

为一个合同金额任务写出四个 hard negative：格式漂亮但税率错、引用存在但不支持、过程正确但单位缺失、最终数值碰巧正确但中间公式错误。

### 练习五：训练目标

用三个步骤分数计算平均聚合、最小聚合和加权聚合。说明哪一种聚合更容易受到长轨迹和单步异常值影响。

### 练习六：搜索剪枝

构造一个 beam 宽度为 2 的三层搜索树，让 PRM 在第二层错误地低估正确分支。计算正确分支被剪掉后，最终 top-1 的变化。

### 练习七：自动监督边界

分别为代码编译、单元测试、JSON schema 和合同引用写出它们能检查的约束，以及不能检查的约束。

### 练习八：运行 demo

运行本章 Python demo，把 `lucky_answer` 的第一步改成错误，再观察 `outcome_accuracy`、`first_error_accuracy` 和 `review` 是否变化。解释为什么最终答案指标可能没有变化。

## 18. 本章总结

过程监督的核心不是让模型写出更长的解释，而是把一条任务轨迹拆成可定义、可标注、可验证的中间对象。

第一，结果监督关注最终答案，过程监督关注步骤或状态；两者分别解决稀疏反馈和终局正确性问题，不能互相替代。

第二，一步必须有明确边界和上下文。正确性、相关性、充分性、证据支持和安全性最好分开记录，再按任务契约组合。

第三，第一处错误比错误总数更能帮助诊断。后续步骤可能只是错误前提的连锁结果，标签协议应明确如何处理。

第四，PRM 可以用于步骤评分、候选排序和搜索剪枝，但它的分数是代理信号，不自动是事实概率。平均分、最小分和加权分都有适用边界。

第五，程序化标签能降低成本，却只能覆盖写入规则的约束；人工标签能处理开放判断，却需要一致性、争议记录和成本核算。

第六，过程模型最容易被长度、格式、自信语气、参考路径和生成器分布欺骗。hard negative、反事实消融、校准和独立 outcome 检查是基本防线。

第七，真正的效果要看下游任务成功率、错误定位、搜索选择、延迟、人工复核和单位成功成本，而不是只看步骤分类准确率。

第八，高分不等于权限。涉及代码执行、付款、删除、隐私和外部通信时，过程监督必须和沙箱、权限、规则、工具结果及人工确认配合。

下一章将进入搜索与 Tree-of-Thought，进一步讨论如何表示搜索节点、展开候选、控制分支、使用 outcome 或 process verifier，并分析搜索预算与错误剪枝之间的关系。

## 19. 资料索引

以下入口优先使用原始论文和公开仓库。复现实验时，应同时记录候选生成器、候选数量、步骤切分、负例来源、评分器版本、阈值、搜索宽度和下游任务指标。

1. Training Verifiers to Solve Math Word Problems：<https://arxiv.org/abs/2110.14168>
2. Let's Verify Step by Step：<https://arxiv.org/abs/2305.20050>
3. OpenAI PRM800K repository：<https://github.com/openai/prm800k>
4. RewardBench：<https://arxiv.org/abs/2403.13787>
5. Faithful Chain-of-Thought Reasoning：<https://arxiv.org/abs/2301.13379>
6. HumanEval：<https://arxiv.org/abs/2107.03374>

本章写作时对前两篇论文和 PRM800K 公开仓库入口进行了联网访问；OpenAI 介绍页本轮返回 403，因此没有把该页面的未读取内容作为事实依据。正文区分原始论文结论、公开仓库内容、程序化约束、教学构造和目标系统复测，没有把 toy demo 的输出外推为真实 PRM 的公开性能。
