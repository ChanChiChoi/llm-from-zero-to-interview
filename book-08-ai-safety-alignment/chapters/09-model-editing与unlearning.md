# 第九章：Model Editing 与 Unlearning

大模型的知识更新有一个容易被忽略的事实：模型不是数据库。数据库中的一行记录可以被定位、修改和删除；神经网络中的一个回答倾向，却可能是许多训练样本、参数方向、上下文模式和解码策略共同作用的结果。

因此，“让模型知道新的事实”和“让模型不再泄露旧数据”看起来都像更新知识，实际上是两个不同的问题。前者通常需要一个新的目标答案，后者通常需要目标影响减少，同时尽量保留其余能力。前者可以是 model editing，后者属于 machine unlearning 的问题范围；RAG、继续训练、输出策略和数据治理又处在不同的层。

本章从一个具体的事实修改问题开始，逐层建立这几个概念的边界。文中的公式是为了说明变量、约束和评估逻辑；ROME、MEMIT、MEND、SERAC 等论文中的具体实现还依赖模型结构、训练数据和超参数，不能把教学抽象当成对所有模型的定理。

本章只讨论防御性、治理性和评估性的技术。涉及版权、隐私或危险能力时，示例使用虚构数据或抽象能力标签，不提供提取隐私、植入错误事实、绕过安全策略或恢复危险操作的步骤。

## 1. 一个事实为什么不能像数据库记录一样修改

### 1.1 模型保存的不是一句话

自回归语言模型接收提示 \(x\)，然后对下一个 token 给出概率分布。一个答案 \(y=(y_1,\ldots,y_T)\) 的概率可以写成：

$$
p_\theta(y\mid x)=\prod_{t=1}^{T}p_\theta(y_t\mid x,y_{<t})
$$

这里的 \(\theta\) 是模型参数，\(y_{<t}\) 表示第 \(t\) 个 token 之前已经生成的内容。模型并没有一个名为“某公司 CEO”的字段，而是通过多层表示和概率分布，逐步生成一串 token。

如果模型把某公司的 CEO 回答成 A，至少可能有几种原因：

1. 参数中存在过时的事实关联。
2. 提示中的实体和关系没有被正确解析。
3. 上下文里出现了与 A 相关的强线索。
4. 模型知道 B，但在当前模板下选择了 A。
5. 解码、系统提示或外部检索改变了最终文本。

只观察最终答案，无法直接区分这五种情况。model editing 研究的是在尽量小的范围内改变模型行为；它不是把自然语言句子写入一个隐形表格。

### 1.2 训练样本的影响是分布式的

一篇文章可能在预训练中影响很多参数。某个事实还可能在新闻、百科、论坛和重复转载中出现。模型在不同层使用这些信息时，可能分别完成实体识别、关系抽取、语言组织和答案校验。

这带来两个后果。

第一，同一个知识可能有多个来源。删除一份来源，并不自动消除其他来源提供的相同知识。

第二，一组参数可能服务于多个任务。试图让它不再输出某个知识，可能同时损伤阅读理解、实体识别或一般推理。

小白可以把模型想成一张相互交叉的知识网，而不是一本逐句编号的书。专家则会进一步问：目标行为的因果路径是否可识别？编辑位置是否对目标样本具有足够的特异性？修改后是否存在一个可以重复验证的保留集？

### 1.3 四种不同的“删除”

工程讨论中经常把下面四件事都叫作删除，但它们的责任对象不同。

| 名称 | 实际要改变的对象 | 成功的直观含义 | 不能由它单独证明的事情 |
| --- | --- | --- | --- |
| 源数据删除 | 训练集、数据仓库、索引、日志或缓存 | 指定副本不再被系统保存或使用 | 模型参数已经没有相关影响 |
| 知识编辑 | 参数或编辑记忆中的一个事实关联 | 新事实在指定问题族上生效 | 所有相关推论都自动一致 |
| 行为抑制 | 输出策略、拒答策略或分类器 | 模型在指定场景不输出目标内容 | 内部表征已经消失 |
| 模型遗忘 | 训练后模型的目标数据、知识或能力影响 | 行为接近没有见过目标数据的参考模型 | 对无限输入空间的完全遗忘 |

这一张表是本章的主线。只要把“源数据删除”“行为不泄露”和“参数不再包含影响”混为一谈，后面的评估就会失去意义。

## 2. RAG、继续训练、Model Editing、Unlearning 与 Guardrail

### 2.1 RAG：把变化留在模型外

RAG 在推理时检索文档，将文档片段放进上下文。企业产品的组织架构、库存和政策文件经常采用这种方式，因为文档可以带版本号、权限和引用。

RAG 的优势是更新快、来源清楚、可以按用户权限过滤。它的代价是检索质量会影响回答，模型参数里的旧知识可能仍然干扰，文档冲突也需要额外的时间排序和可信度规则。

如果每周都要更新某个业务事实，先考虑 RAG 通常更稳妥。把每次变化都写进参数，会产生连续编辑、回滚和一致性维护的负担。

### 2.2 继续训练和微调：用更多梯度改变整体分布

继续预训练适合让模型吸收大规模的新语料；SFT 适合学习一组新的输入输出行为。二者都通过梯度更新参数，因此可能让目标变化更持久，也可能影响较宽的能力范围。

微调并不一定是“全局破坏”。通过小学习率、参数高效适配器、保留集约束和版本化实验，可以把漂移控制在可接受范围。但如果需求只有一两个稳定事实，完整微调通常难以解释为什么改变了相邻知识。

### 2.3 Model editing：把变化集中到少量知识或行为

Model editing 的典型形式是给出一个编辑请求：

$$
e=(x_e,y_e^{old},y_e^{new})
$$

其中 \(x_e\) 是目标提示，\(y_e^{old}\) 是旧答案，\(y_e^{new}\) 是希望出现的新答案。编辑器产生新参数 \(\theta'\)，希望模型在目标问题上回答 \(y_e^{new}\)，同时在不相关问题上保持原行为。

它最适合研究和处理少量、边界较清楚的改变。它不天然适合动态知识库，也不天然等价于版权或隐私意义上的彻底删除。

### 2.4 Unlearning：减少目标影响，而不只是改变一个答案

Unlearning 的目标可能是删除训练样本影响、降低某类知识记忆、减少某种行为或抑制一项危险能力。它通常没有一个新的事实答案可供模型学习；目标更多是“目标影响消失，其他能力保留”。

所以 unlearning 的评估必须同时包含 forget set 和 retain set。只看忘记集上的准确率下降，可能只是在破坏模型或让它统一拒答。

### 2.5 Guardrail：在模型之外控制可见行为

输出过滤、策略分类器、权限检查和工具沙箱可以快速降低风险。它们对产品很有价值，因为它们容易开关、容易审计，也不必等待参数编辑完成。

但 guardrail 只改变系统允许呈现或执行的结果。一个模型在拒答后仍可能保留目标知识；一个过滤器误判，也可能造成过度拒答。因此 guardrail 不能被写成参数级 unlearning 的证据。

### 2.6 同一个需求的不同工程答案

假设“员工手册中的报销上限从 500 元改为 800 元”：

1. 文档频繁变化并要求引用，优先更新知识库和 RAG。
2. 模型在无上下文时也必须知道一个长期稳定的事实，可以研究 model editing。
3. 某用户要求删除个人身份证号，需要同时处理源数据、索引、训练记录、模型版本和泄露评估。
4. 高风险问题暂时不能回答，可以先用策略和权限系统抑制，再研究更长期的训练或遗忘方案。

技术选择取决于变化的频率、目标的边界、可接受的副作用、回滚需求和法律责任，而不是取决于某种方法的论文名字。

## 3. 先把编辑问题写清楚

### 3.1 目标、改写和局部性集合

一个完整的编辑请求不应只有一个问答对。至少要准备四类样本：

1. 目标样本：直接询问旧事实的位置。
2. 改写样本：使用不同句法、同义表达或不同语言表达同一事实。
3. 局部性样本：与目标实体相关，但按要求不应该改变的事实。
4. 保留样本：用于检查通用能力、语言质量和安全行为。

可以把第 \(i\) 个编辑写成：

$$
e_i=(x_i,y_i^{old},y_i^{new},P_i,L_i,w_i)
$$

\(P_i\) 是改写集合，\(L_i\) 是局部性集合，\(w_i\) 是风险或重要性权重。没有 \(P_i\) 时，编辑可能只是记住了一个模板；没有 \(L_i\) 时，无法知道副作用的范围。

### 3.2 编辑的多个目标

用 \(M_\theta\) 表示编辑前模型，\(M_{\theta'}\) 表示编辑后模型。一个教学层面的综合目标可以写成：

$$
\mathcal L_{edit}(\theta')=
\mathcal L_{target}+
\lambda_p\mathcal L_{paraphrase}+
\lambda_l\mathcal L_{locality}+
\lambda_r\mathcal L_{retain}+
\lambda_c\mathcal L_{conflict}+
\lambda_\Delta\lVert\theta'-\theta\rVert^2
$$

最后一项约束参数变化幅度；\(\lambda\) 决定不同目标的相对重要性。这个公式不是所有编辑器共用的训练损失，而是帮助读者看清：目标正确率只是其中一项。

如果编辑的是高风险行为，\(\mathcal L_{conflict}\) 还要覆盖安全边界、正常帮助和工具调用。只把“拒答率”设得越高，并不代表系统越安全，因为误拒答和绕过风险也可能同时上升。

### 3.3 事实编辑和行为编辑不是同一难度

“某城市的首都是 X”是相对窄的事实关联。模型可能在特定中间层通过实体和关系触发一个答案。

“遇到隐私问题要谨慎回答”则是跨模板、跨领域、跨风险等级的行为规则。它涉及分类、解释、拒答、替代帮助和多轮对话，通常不能依靠一次局部权重更新完整表达。

编辑请求越接近一个可枚举的事实，内部编辑方法越容易定义目标。请求越接近政策、价值判断或复杂能力，越需要数据、策略、评估和系统控制共同承担责任。

## 4. 从 Transformer 的内部计算理解“为什么可能编辑”

### 4.1 MLP 的 key-value 视角

Transformer 的一个前馈模块可以抽象成：

$$
h_{l+1}=h_l+W_{out}\,\sigma(W_{in}h_l)
$$

\(h_l\) 是第 \(l\) 层的隐藏状态，\(W_{in}\) 把状态投影到中间维度，\(\sigma\) 是非线性函数，\(W_{out}\) 再把结果投影回模型维度。

为了便于理解，可以把：

$$
k=W_{in}h_l,\qquad v=W_{out}\sigma(k)
$$

看成一个“由当前上下文产生 key，再贡献 value”的计算。某些研究把中间层 MLP 解释为储存或调用事实关联的候选位置。

这不是说每个事实都只存放在一个神经元里，也不是说 MLP 真的是一个可直接查询的数据库。它是一种可检验的机制假设：如果对某个位置的表示做因果干预，目标答案的概率应该出现可重复变化。

### 4.2 因果定位和相关性观察

普通相关性只能告诉我们某个激活与答案一起出现。因果定位会把一个输入的激活替换到另一个输入的前向计算中，再观察答案概率变化。

设干预前目标答案概率为 \(p_{base}\)，把候选层的隐藏状态替换为参考运行中的 \(h_l^*\) 后，概率为 \(p_{patch}\)，则可以记录：

$$
\Delta_{causal}=p_{patch}-p_{base}
$$

如果 \(\Delta_{causal}\) 在多个模板、随机种子和相邻层上都稳定，候选位置就更值得研究。它仍然不是“事实只存在那里”的证明，因为干预可能改变了信息流，而不是准确识别了唯一存储位置。

### 4.3 位置假设的边界

ROME 论文对 GPT 类模型的实验发现，中间层 feed-forward 模块对一类 factual association 的回忆有重要作用。这个结论支持一种有用的编辑路线，但不应外推为：

1. 所有模型都在相同层存储事实。
2. 所有知识都可以低秩修改。
3. 改变一个 MLP 就能同步修改所有多跳推理。
4. 编辑后模型已经删除了训练数据来源。

模型架构、训练配方、指令微调、分词器、语言和上下文都会改变编辑问题。白盒编辑前应先做定位和小规模复现；黑盒 API 通常不能直接使用这些参数级方法。

## 5. ROME：一次局部事实编辑

### 5.1 ROME 要解决什么

早期的直接微调可以让一个目标问题答对，但往往会对单个模板过拟合，或者改变太多相邻行为。ROME，即 Rank-One Model Editing，尝试把事实修改集中到与目标关联相关的 feed-forward 权重。

论文的核心实验链路包括：

1. 用因果干预定位对事实预测有影响的中间层步骤。
2. 以 subject token 的隐藏状态构造 key。
3. 为目标关系构造期望 value。
4. 用一次低秩更新修改相关权重。
5. 在 CounterFact 和 zsRE 等任务上检查成功率、泛化和特异性。

“低秩”描述的是更新的结构，不是说事实本身只有一个数字。rank-one 更新只改变一个很受约束的方向，但这个方向经过网络的非线性传播，仍可能影响多个输出。

### 5.2 从约束优化得到 rank-one 更新

设待修改矩阵为 \(W\in\mathbb R^{d_v\times d_k}\)，目标 key 为 \(k_e\in\mathbb R^{d_k}\)，期望 value 为 \(v^*\in\mathbb R^{d_v}\)。最直接的要求是：

$$
W'k_e=v^*
$$

如果只要求这一条约束，有无数个 \(W'\) 可选。为了尽量不影响其他常见 key，可以用一个协方差矩阵 \(C\) 表示正常 key 的分布，并求：

$$
\min_{W'}\operatorname{Tr}\left((W'-W)C(W'-W)^\top\right)
\quad
\text{s.t.}\quad W'k_e=v^*
$$

当 \(C\) 可逆时，一个教学层面的解是：

$$
W'=W+
\frac{(v^*-Wk_e)k_e^\top C^{-1}}
{k_e^\top C^{-1}k_e}
$$

分子中的 \(v^*-Wk_e\) 是需要补上的 value 差值；\(k_e^\top C^{-1}\) 决定把修改投影到哪些参数方向；分母是归一化项，避免同一尺度下更新过大。

如果把 \(C\) 换成单位矩阵，公式会退化为沿 \(k_e\) 方向的简单 rank-one 更新。ROME 使用的协方差思想更接近“在常见输入方向上少动一些”，但实际实现还要处理层选择、目标 token、正则化和数值稳定性。

### 5.3 为什么同一个编辑可能在改写问法上生效

目标问法和改写问法虽然 token 不同，但可能在中间层产生相近的 subject-relation 表示。如果编辑改变的是关系计算所依赖的方向，而不是只改变一个完整字符串，改写泛化就有机会出现。

这也解释了泛化不稳定的原因：如果改写改变了语言、上下文位置或推理路径，key 可能已经离开编辑所覆盖的局部区域。编辑成功不是“字符串替换成功”，而是要验证一族输入的行为变化。

### 5.4 ROME 的能力边界

ROME 的结果支持以下有限结论：对一类 GPT 类模型和事实关联任务，直接修改中间 feed-forward 计算可以有效改变指定关联，并在某些基准上同时取得一定的泛化与特异性。

它不支持以下更强结论：

1. ROME 对所有开源或闭源模型都有效。
2. 一次编辑足以保持实体图中的全部多跳关系。
3. 目标内容从参数中已经物理消失。
4. 连续执行很多次后仍然没有累积副作用。

因此工程上应记录模型版本、编辑位置、编辑前后 checkpoint、测试模板和回滚信息。只保存一份“修改后的权重”，将来很难追究某次变化由哪一条编辑请求造成。

### 5.5 一个简化的事实例子

假设模型在无检索条件下把“某产品的设计者是谁”回答成旧答案 A。编辑请求把答案改为 B。

编辑前后至少要检查：

| 样本 | 预期 |
| --- | --- |
| 直接问设计者 | 从 A 变成 B |
| “谁负责设计该产品” | 也应倾向 B |
| 该设计者的出生地 | 不应被无关地改掉 |
| 同名产品的设计者 | 不应被连带替换 |
| 一般阅读理解 | 与编辑前差异应在容许范围内 |

如果只有第一行变化，说明编辑可能过拟合模板；如果表中三、四行也发生大幅变化，说明 locality 不足。

## 6. MEMIT：从单个事实到批量记忆

### 6.1 为什么需要批量编辑

生产系统很少只遇到一条过时信息。新闻事实、产品目录、组织关系和领域知识可能同时有成百上千条变化。逐条执行 ROME 会重复定位和更新，也可能使前一次更新改变后一次更新看到的 key/value 分布。

MEMIT，即 Mass-Editing Memory in a Transformer，把问题扩展到大规模的 memory edits。其论文摘要报告了在 GPT-J 6B 和 GPT-NeoX 20B 上对数千条关联进行编辑的实验，这个结果应理解为特定模型、数据和实验设置下的可扩展性证据，而不是所有模型的容量保证。

### 6.2 批量更新的教学抽象

令 \(K=[k_1,\ldots,k_m]\in\mathbb R^{d_k\times m}\) 是多个编辑 key，\(V^*=[v_1^*,\ldots,v_m^*]\in\mathbb R^{d_v\times m}\) 是期望 value。希望：

$$
W'K\approx V^*
$$

一个带正则的批量更新可抽象为：

$$
\Delta W=(V^*-WK)K^\top
(KK^\top+\lambda C)^{-1}
$$

这里写法中的矩阵结合顺序应理解为：

$$
\Delta W=(V^*-WK)K^\top(KK^\top+\lambda C)^{-1}
$$

\(m\) 是编辑数量，\(\lambda\) 控制正则强度，\(C\) 表示正常 key 的统计结构。正则项让更新不要为了满足少数 key 而过度改变常见方向。

MEMIT 的实际方法不是把一条公式机械地套到一个矩阵上，而是把更新分配到多个层，并处理不同层的表示传播。上式的价值在于说明批量编辑的三个难点：

1. key 之间可能相似，导致矩阵病态。
2. 新 value 之间可能冲突。
3. 让许多约束同时满足，必然增加局部性和能力保留的压力。

### 6.3 编辑数量与矩阵条件数

当两个编辑的 key 很接近，却要求不同的 value，系统会出现冲突。可以用条件数观察数值问题：

$$
\kappa(KK^\top+\lambda C)
=\frac{\sigma_{\max}}{\sigma_{\min}}
$$

\(\sigma_{\max}\) 和 \(\sigma_{\min}\) 是该矩阵的最大、最小奇异值。条件数很大意味着某些方向几乎无法稳定区分；小的数值误差可能产生很大的权重变化。

这不是唯一的质量指标。即使矩阵条件数良好，语义上的冲突仍然可能存在。例如“公司 X 的总部在哪里”和“公司 X 的历史总部在哪里”共享实体，却对应不同时间关系。编辑器必须把时间、关系和上下文纳入测试集，而不能只按实体字符串去重。

### 6.4 顺序编辑的累积效应

如果第 \(t\) 次更新为 \(\Delta W_t\)，连续编辑后的权重是：

$$
W_T=W_0+\sum_{t=1}^{T}\Delta W_t
$$

这个表达式很简单，却揭示了工程问题：每次更新都很小，并不代表总变化很小。更麻烦的是第 \(t+1\) 次编辑是在已经改变的模型上计算，误差和冲突会逐步累积。

实际系统应定期做：

1. 编辑数量分桶评估。
2. 不同顺序的重放实验。
3. 原始 checkpoint 与编辑后 checkpoint 的差分审计。
4. 失败编辑的单独回滚，而不是只保留最终权重。
5. 大规模编辑和新训练之间的对比。

### 6.5 MEMIT 适合什么，不适合什么

MEMIT 适合研究或处理一批边界相对清楚、可以生成改写和 locality 样本的事实更新。它不适合被当成通用知识库，也不适合在缺少数据溯源和回滚机制的情况下直接修改生产主模型。

当需求具有强时效性、权限差异或引用要求时，外部记忆通常更容易审计。批量参数编辑只有在“模型无上下文时也必须改变”的价值足够大时，才值得承担内部副作用。

## 7. MEND：学习一个快速编辑器

### 7.1 从一次梯度更新出发

给定单个输入输出对，普通微调会产生梯度：

$$
g=\nabla_W\mathcal L(x_e,y_e^{new};W)
$$

直接使用 \(g\) 可能对目标样本过拟合，也可能修改过多参数。MEND，即 Model Editor Networks with Gradient Decomposition，学习一组小型辅助网络，把普通梯度变换成更适合编辑的更新。

MEND 的关键不是手工找某个事实神经元，而是从许多训练时的编辑任务中学习“什么样的梯度应该怎样变换”。论文摘要描述了利用梯度的低秩分解，使变换参数可控，并在多种模型上研究快速局部编辑。

### 7.2 为什么需要元训练分布

编辑器网络本身要先训练。它学到的是训练期间编辑请求的统计规律，所以它的泛化依赖于：

1. 训练编辑与部署编辑是否同分布。
2. 输入输出格式是否变化。
3. 目标是事实还是行为。
4. 基础模型结构是否变化。
5. 编辑请求之间是否存在长期冲突。

这使 MEND 与 ROME 有一个重要差异：ROME 更强调对当前模型的机制定位，MEND 更强调一个可重复调用的学习型编辑器。MEND 速度快并不意味着不需要部署前验证；编辑器的元训练分布本身就是一种隐藏依赖。

### 7.3 MEND 的工程取舍

MEND 适合大量同类型编辑请求，尤其当编辑器已经针对目标模型和任务训练好时。它的主要代价是编辑器训练、模型迁移和分布外请求判断。

如果基础模型从 T5 换成另一类架构，或者从事实问答换成复杂工具策略，旧的编辑器不能因为输入形状相同就被默认复用。必须重新做目标、改写、locality 和能力保留实验。

## 8. SERAC：把编辑放进显式记忆

### 8.1 参数编辑与外部编辑记忆

SERAC，即 Semi-Parametric Editing with a Retrieval-Augmented Counterfactual Model，不把每条编辑都永久写入基础模型参数，而是保存一个显式 edit memory。

一次请求到来时，系统大致经历：

1. 将输入与已保存编辑进行相关性计算。
2. 判断这个输入是否落在某条编辑的作用范围内。
3. 如果相关，交给 counterfactual model 产生新行为。
4. 如果不相关，继续使用基础模型。

可以用下面的教学式表达描述混合输出：

$$
p(y\mid x,E)=
(1-s(x,E))p_{\theta}(y\mid x)
+s(x,E)p_{\phi}(y\mid x,E)
$$

\(E\) 是编辑记忆，\(s(x,E)\in[0,1]\) 是相关性分数，\(p_\phi\) 是处理编辑后的模型。论文中的实现包含显式记忆、范围判断和反事实模型；公式只是帮助理解“何时使用编辑”的系统结构。

### 8.2 SERAC 的优势

显式记忆具有三个工程优点：

1. 记录中可以保存来源、时间和编辑者。
2. 单条编辑可以删除或替换，不必重写整个 checkpoint。
3. 可以把基础模型和编辑模型分开回滚。

它的代价也同样明确：

1. 推理时增加检索和路由开销。
2. 编辑记忆不断增长，需要压缩和冲突解决。
3. 相关性判断错了，会漏用编辑或误用编辑。
4. 删除编辑记忆不等于删除基础模型原有知识。

### 8.3 四种方法放在同一张地图上

| 方法 | 变化存放位置 | 适合的请求 | 主要失败模式 |
| --- | --- | --- | --- |
| ROME | 少量内部权重 | 单个或少量事实 | 泛化不足、局部副作用、连续编辑退化 |
| MEMIT | 多层权重更新 | 批量事实关联 | key 冲突、容量和累积副作用 |
| MEND | 学习到的编辑器加参数更新 | 大量相似类型请求 | 元训练分布外失效 |
| SERAC | 显式编辑记忆和反事实模型 | 可追踪、可撤销的外部编辑 | 检索范围误判、记忆增长和延迟 |

没有哪一种方法在所有指标上都占优。方法选择应该由目标的持久性、规模、可撤销性和可解释性决定。

## 9. Model Editing 不只修改事实

### 9.1 事实修正

事实修正最容易定义，因为通常存在一个新的目标答案。但即使是事实，也要处理时间关系、条件关系和多跳推理。

例如“产品 A 的负责人是 B”至少可能派生出：

1. B 负责产品 A。
2. B 所属部门负责产品 A。
3. 产品 A 在 B 任职期间发布。
4. 产品 A 与另一个产品共享负责人。

只改第一句，其他推论可能仍然围绕旧事实展开。编辑集需要显式列出应改变和不应改变的关系。

### 9.2 风格、偏好和政策行为

行为编辑通常没有唯一正确的答案。一个新的政策行为可能需要：

1. 在危险请求上拒绝。
2. 在安全相邻请求上继续帮助。
3. 给出原因但不泄露内部规则。
4. 推荐安全替代方案。
5. 在工具调用前要求权限。

这类目标更像一个条件策略，而不是一个 token 替换。单个编辑样本无法覆盖边界，应该使用成组的正例、负例、相邻安全例和多轮例。

### 9.3 多跳编辑

如果目标事实被另一个问题间接引用，直接编辑可能在单跳测试上成功，却在多跳问题上失败。多跳评估可以把中间实体、关系和最终答案拆开，观察失败发生在：

1. 新事实没有被召回。
2. 旧事实仍在中间推理中占优势。
3. 关系组合错误。
4. 最终语言生成覆盖了正确中间结果。

这类诊断比一个总准确率更有用，因为不同原因需要不同修复。

### 9.4 编辑安全行为的双用风险

改变安全相关行为可能被用于防御性修复，例如减少过度拒答；也可能被用于削弱模型的防护。研究环境应该使用受控模型、合成标签和离线评估，生产系统则要保留权限、审计、回滚和外部策略层。

“能够编辑”不等于“应该允许任意人编辑”。编辑接口本身需要身份、审批、变更原因、影响范围和可恢复版本。

## 10. 编辑评估：目标答对只是第一步

### 10.1 Edit success

令 \(a_i^{target}=1\) 表示第 \(i\) 个编辑的直接目标按新答案回答，反之为 0。加权目标成功率为：

$$
S_{target}=
\frac{\sum_i w_i a_i^{target}}{\sum_i w_i}
$$

分母是所有编辑的总权重，不是样本条数。使用权重时，重要或高风险编辑对总体分数贡献更大；如果所有 \(w_i=1\)，公式退化为普通平均。

目标成功率适合回答“直接目标有没有改到”，但无法回答“改写是否泛化”和“相邻知识有没有被破坏”。

### 10.2 Paraphrase generalization

设第 \(i\) 个编辑有 \(m_i\) 个改写，\(a_{ij}^{para}=1\) 表示第 \(j\) 个改写也得到新答案，则：

$$
S_{para}=
\frac{\sum_i w_i
\left(\frac{1}{m_i}\sum_{j=1}^{m_i}a_{ij}^{para}\right)}
{\sum_i w_i}
$$

先在每个编辑内部平均，再按编辑权重平均，避免改写数量多的某个编辑支配总分。\(m_i\) 不能为 0；没有改写样本的编辑应该单独标为“未测量”，而不是当作满分。

改写不应只做词语替换。至少要覆盖句法变化、上下文变化、代词引用、语言变化和间接询问。对事实编辑，还要区分“同一事实的改写”和“相关但不同的事实”。

### 10.3 Locality 与 specificity

令 \(b_{ik}^{loc}=1\) 表示第 \(i\) 个编辑的第 \(k\) 个局部性样本在编辑前后保持预期行为，\(n_i\) 是局部性样本数：

$$
S_{local}=
\frac{\sum_i w_i
\left(\frac{1}{n_i}\sum_{k=1}^{n_i}b_{ik}^{loc}\right)}
{\sum_i w_i}
$$

局部性不是“参数变化很小”的同义词。一个很小的参数变化，经过网络放大后仍可能改变许多输出；一个较大的变化，也可能在目标子空间之外保持稳定。要把参数差分、行为差分和因果干预分开记录。

局部性样本可以分三层：

1. 同一实体的不同关系。
2. 语义相关但不应改变的事实。
3. 完全无关的通用任务。

这样才能看到副作用是局部扩散，还是全局漂移。

### 10.4 Consistency、portability 和 fluency

一个新事实可能要在多个关系和多个上下文中自洽。Consistency 检查的是同一编辑的不同表述是否彼此矛盾；portability 检查的是新事实在相关推理任务中能否被使用；fluency 检查生成是否出现异常重复、语法破碎或模板化。

这些指标没有一个统一的“正确答案”。应把自动指标与人工抽样结合，并保存原始输出。只保存分数会让后续人员无法判断评估器是否把格式变化误报成知识变化。

### 10.5 通用能力保留

设保留任务集上的能力为 \(A_{ret}^{before}\) 和 \(A_{ret}^{after}\)，可以记录：

$$
\Delta_{ret}=A_{ret}^{after}-A_{ret}^{before}
$$

\(\Delta_{ret}<0\) 表示能力下降。不同任务的分数不可直接相加，应先按任务定义归一化，再分别报告数学、代码、事实问答、语言质量和安全边界。

一个综合分数可能掩盖某一项关键退化。例如平均能力只下降 1%，但隐私相关的保留任务下降 20%，对产品来说仍可能不可接受。

### 10.6 连续编辑和顺序敏感性

对于 \(T\) 次编辑：

$$
\theta_T=\theta_0+\sum_{t=1}^{T}\Delta\theta_t
$$

这只是参数加法的记号，实际模型还会通过非线性改变后续编辑的输入表示。评估时要比较：

1. 一次性批量编辑与逐条编辑。
2. 不同编辑顺序。
3. 目标集合相同但随机种子不同的结果。
4. 编辑数量从少到多时的曲线。
5. 回滚某一条编辑后的局部恢复情况。

如果只在最终 checkpoint 上测一次，无法知道什么时候开始退化。

### 10.7 小样本分数的统计边界

对 \(n\) 个等权二元样本，正确率估计为 \(\hat p\)。一个教学层面的标准误差近似是：

$$
SE(\hat p)=\sqrt{\frac{\hat p(1-\hat p)}{n}}
$$

当样本很少、\(\hat p\) 接近 0 或 1、或样本之间存在同一事实的聚类时，这个近似会失真。改写样本共享同一编辑，不能被当作完全独立的 \(n\) 个证据。更可靠的报告应按编辑分组做 bootstrap 或报告置信区间，而不是只比较两个小数点。

## 11. Unlearning 的理想定义

### 11.1 以重新训练模型作为参照

令完整训练数据为 \(D\)，目标删除集合为 \(F\)，保留集合为 \(R=D\setminus F\)。最清楚的参照模型是：

$$
\theta_{-F}=Train(R;\xi)
$$

\(\xi\) 表示训练配方、随机种子、数据顺序和优化器设置。unlearning 模型记为 \(\theta_u\)，理想状态不是参数逐元素等于 \(\theta_{-F}\)，而是在明确的审计分布 \(T\) 上表现足够接近：

$$
\mathbb E_{x\sim T}
\left[d\left(f_{\theta_u}(x),f_{\theta_{-F}}(x)\right)\right]
\le \varepsilon
$$

\(d\) 可以是输出分布的 KL 散度、答案差异、成员推断攻击优势或任务损失差异。不同 \(d\) 测量不同性质，不能把一个小的 \(d\) 写成对所有输入的完全等价。

### 11.2 为什么完全遗忘难以证明

输入空间几乎无限，用户可以使用新的语言、上下文、拼写、间接线索和多轮对话。即使一组测试集上不再复现目标文本，也可能在另一组提示中泄露相关信息。

此外，模型可能从多个来源学习相同事实。删除某个来源后，模型仍然能回答该事实，并不一定说明 unlearning 失败；如果法律或隐私要求的对象是特定个人数据，则“还能回答公共事实”和“还能复现个人独有信息”必须区分。

因此“遗忘”必须先定义对象：

1. 是删除某条训练样本，还是删除一个人的全部数据？
2. 是禁止逐字复现，还是禁止回答其内容？
3. 是降低危险能力，还是只禁止具体操作建议？
4. 是从当前模型删除，还是从所有 adapter、缓存和下游副本删除？

没有范围定义，就没有可验证的遗忘声明。

## 12. 四种 unlearning 目标

### 12.1 Data unlearning

Data unlearning 关注指定训练数据的影响。目标可能是一个样本、一个用户、一批文档或一个来源集合。

它要求数据溯源足够好：知道目标数据出现在哪些训练切片、重复版本、微调集和评估集。没有 lineage，算法即使改变了模型，也无法说明改变对应了哪个删除请求。

### 12.2 Knowledge unlearning

Knowledge unlearning 关注某个知识或内容是否仍能被模型召回。版权内容可以作为例子，但“作品本身”“作品中的事实”“逐字记忆”和“主题常识”是不同目标。

如果把整个主题都当成 forget set，模型可能丢失不应删除的公共知识；如果只测几段原文，又可能漏掉改写和间接问法。

### 12.3 Behavior unlearning

Behavior unlearning 关注某类输出行为，例如不再泄露某种格式的个人信息，或不再执行某个不被允许的动作。它通常更接近策略学习和安全调优，而不是数据删除。

行为目标应包含安全相邻的正常请求。例如“拒绝泄露隐私”不能变成“拒绝所有包含姓名的文本处理请求”。

### 12.4 Capability unlearning

Capability unlearning 试图降低一类能力，例如某个危险知识簇的可用性。能力通常和正常的科学、代码或语言能力共享表示，因此必须同时测量目标能力和邻近保留能力。

WMDP 论文把它作为危险知识评估和 unlearning 研究的公开基准之一，但 WMDP 分数下降只说明该基准上的能力变化，不能自动推出真实世界的全面能力消失。

## 13. Unlearning 方法谱系

### 13.1 从头重新训练：昂贵但清晰的参考

从 \(D\setminus F\) 重新训练最接近“没有看到 F”的定义。如果资源允许，它是最有解释力的 reference model。

它仍有实际限制：

1. 重新训练可能使用不同的随机顺序和优化轨迹。
2. 数据重复和近似副本会影响参照。
3. 模型发布后还有 adapter、检索库和缓存副本。
4. 逐次删除请求会导致重复昂贵训练。

所以从头训练常被用作小模型或小数据实验的上界参照，而不是每次生产请求的默认操作。

### 13.2 Gradient ascent 与 retain 约束

对 forget set \(F\)，普通训练损失为：

$$
\mathcal L_F(\theta)=
-\frac{1}{|F|}\sum_{(x,y)\in F}\log p_\theta(y\mid x)
$$

如果最小化 \(-\mathcal L_F\)，就会倾向于提高 forget set 上的损失，也就是降低模型对目标答案的拟合。为了保留一般能力，可以加入保留集损失：

$$
\mathcal L(\theta)=
\mathcal L_R(\theta)
-\lambda_F\mathcal L_F(\theta)
$$

\(\lambda_F\) 越大，遗忘压力越强，但破坏保留能力的风险也可能越高。梯度方向并不知道“目标知识只存在于哪里”，所以在参数共享严重的模型上尤其容易产生能力退化。

### 13.3 用 KL 约束保留行为

一种常见思路是在保留集上让新模型接近原模型：

$$
\mathcal L(\theta_u)=
\mathcal L_R(\theta_u)
-\lambda_F\mathcal L_F(\theta_u)
+\lambda_{KL}
\mathbb E_{x\sim R}
D_{KL}\left(
p_{\theta_0}(\cdot\mid x)
\;\|\;
p_{\theta_u}(\cdot\mid x)
\right)
$$

\(\theta_0\) 是遗忘前模型。KL 项约束的是保留集上的输出分布，而不是所有输入；它能缓解漂移，却不能让忘记集自动消失。

### 13.4 Preference optimization 与“我不知道”

把 forget set 的目标答案替换成“我不知道”或安全拒答，再用偏好优化训练，通常能降低目标内容的直接输出概率。这在行为抑制上可能很实用。

但它有一个清楚的语义边界：模型可能只是学会在这些模板上拒答。如果用户换一种表达，或者把目标内容放进上下文，内部表示仍可能影响回答。因此这类方法应称为“输出抑制”或“拒答调优”，除非有更强的遗忘证据，不要直接称为完整 unlearning。

### 13.5 表示空间方法

WMDP 论文中的 RMU，即 Representation Misdirection for Unlearning，尝试改变目标危险知识相关的表示，同时保持一般能力。它展示了“表示变化 + 专门能力评估”的路线有研究价值。

需要注意两个边界：

1. RMU 的结果依赖 WMDP、模型、训练设置和评估任务。
2. 降低一个公开代理基准的分数，不等于消除所有危险能力或所有迁移路径。

能力抑制实验必须在受控环境下进行，并由外部策略和权限系统承担现实风险控制。

### 13.6 分片训练和可逆记录

在部分机器遗忘研究中，SISA 一类思路把训练拆成多个 shard 和 slice，使删除一个数据子集时只需重训受影响的部分。对超大语言模型，分片训练会带来模型融合、性能和存储成本，不能直接照搬分类模型的结论。

它提醒我们：最便宜的遗忘往往不是训练后魔法，而是训练前就设计好数据分区、版本记录和可重建路径。

## 14. Unlearning 评估：忘记集、保留集与攻击集

### 14.1 Forget set 上的直接泄露

最初可以检查目标问答的回答准确率、目标文本的复现率或目标答案的 token likelihood。但直接问法很容易被输出策略针对，不能作为唯一证据。

更安全的评估做法是使用合成或已授权数据，并记录：

1. exact prompt 的泄露。
2. paraphrase prompt 的泄露。
3. 多轮对话中的泄露。
4. 给出部分线索后的泄露。
5. 语言和格式变化后的泄露。

不需要把敏感原文放进公开日志。日志可以保存样本 ID、风险标签、哈希、输出分类和受控访问地址。

### 14.2 Retain set 上的保留能力

保留集不能只取与 forget set 完全无关的随机数据。至少要有三层：

1. 近邻集：语义相关但不在删除范围内。
2. 领域集：同一领域的正常任务。
3. 通用集：数学、代码、语言和安全边界等基础能力。

近邻集最能揭示过度遗忘。随机通用集分数不变，不代表与目标相邻的公共知识没有被误伤。

### 14.3 改写和多轮攻击

攻击集的目的不是教人绕过系统，而是检验“直接不回答”是否只是表面现象。测试模板可以抽象为：

1. 用代词替代实体。
2. 把问题拆成多个无害子问题。
3. 在上下文中提供部分信息。
4. 请求摘要、翻译或事实核对。
5. 在多轮中逐步累积线索。

每个模板都要有安全审查，避免把真实敏感材料重新暴露给评估参与者。

### 14.4 Membership inference 的位置

Membership inference attack 试图判断某个样本是否出现在训练数据中。可以用攻击准确率、AUC 或 advantage 记录变化。若攻击成功率下降，说明某种可检测的成员信号减弱。

但 MIA 不是完整遗忘证明：

1. 攻击器的能力和训练数据影响结果。
2. 校准、温度和输出接口会改变攻击表现。
3. 样本可能因为公共重复而难以区分。
4. 隐私风险不只来自成员身份，也来自内容复现。

所以 MIA 应与复现、改写、保留能力和参考模型比较共同使用。

### 14.5 和 retain-only reference 比较

如果能训练一个只看 \(R\) 的 reference model，就可以比较 unlearned model 与 reference model 的输出分布：

$$
D_{ref}=
\frac{1}{|T|}\sum_{x\in T}
D_{KL}\left(
p_{\theta_u}(\cdot\mid x)
\;\|\;
p_{\theta_{-F}}(\cdot\mid x)
\right)
$$

\(T\) 是审计提示集合。低 \(D_{ref}\) 说明在这个集合上更接近 retain-only reference，但它对未覆盖输入仍然没有保证。

## 15. 三个重要基准告诉了我们什么

### 15.1 CounterFact 与 zsRE：编辑并非字符串替换

ROME 工作使用 CounterFact 反事实事实数据集，并在 zsRE 关系抽取任务上评估编辑。它们共同推动了目标成功、改写泛化和特异性这几个维度。

CounterFact 适合检验事实关联的局部修改，但它的反事实样本是人为构造的，不能完全代表真实生产中的时间冲突、权限冲突和长尾关系。zsRE 强调零样本关系抽取，也有其任务分布和模板限制。

因此基准分数应回答“在该基准定义的任务上表现如何”，不能回答“模型已经具备通用知识更新能力”。

### 15.2 TOFU：用虚构人物降低现实隐私风险

TOFU，即 Task of Fictitious Unlearning for LLMs，构造了 200 个虚构作者档案，每个档案包含 20 个问答，并把其中一部分作为 forget set。它的价值在于提供了可重复、不会直接暴露真实个人资料的研究环境。

TOFU 同时提供忘记质量、模型效用等一组指标。论文摘要明确指出，所研究的基线没有达到“表现得像从未在 forget data 上训练过”的有效遗忘。这是一个重要的负面结果：方法可运行，不等于目标已实现。

TOFU 的边界也要写清楚。虚构档案不具有真实世界的重复转载、隐私攻击和法律流程复杂度；在 TOFU 上有效的策略仍需在授权数据和更广攻击集上复核。

### 15.3 WMDP：危险知识的公开代理评估

WMDP，即 Weapons of Mass Destruction Proxy，包含生物、网络和化学安全领域的 3,668 道选择题，并在公开前对敏感信息做了筛除。它有两个用途：

1. 测量一类危险知识的代理能力。
2. 研究降低这类能力的 unlearning 方法。

WMDP 的“Proxy”很关键。选择题分数是某种风险相关信号，不是对真实危险行为能力的完整测量。评估结果还要和一般生物、计算机科学、语言能力以及外部安全策略一起解释。

### 15.4 MUSE：把部署者和数据主体的要求放到一起

MUSE，即 Machine Unlearning Six-Way Evaluation for Language Models，提出了六个 desiderata：

1. 不再逐字记忆。
2. 不再保留目标知识记忆。
3. 降低隐私泄露。
4. 保留非删除数据上的效用。
5. 随删除请求规模扩大仍可工作。
6. 连续多次删除请求后仍可持续工作。

MUSE 在 7B 语言模型上研究书籍和新闻数据。其摘要报告，多数算法能在不同程度上降低逐字和知识记忆，但在隐私泄露、通用效用、连续请求和大规模删除方面仍有明显困难。

这六项正好说明，unlearning 不是一个单指标任务。尤其是“连续请求”很容易被忽略：第一次删除有效，不代表第十次、第百次删除仍然稳定。

## 16. 公式化的遗忘评估

### 16.1 Forget leakage

设 \(a_i^{leak}=1\) 表示第 \(i\) 个 forget case 仍泄露目标内容，\(w_i\) 是风险权重。直接泄露率为：

$$
R_{forget}=
\frac{\sum_iw_i a_i^{leak}}{\sum_iw_i}
$$

数值越低越好。分母说明了评估覆盖的风险总量；如果高风险样本权重更大，少量高风险泄露可能比大量低风险样本更显著。

exact、paraphrase 和 multi-turn 应分别计算，而不是先混在一个标签里。否则一个方法可能只降低 exact leak，却在多轮中保留大量泄露。

### 16.2 Robust leakage

设第 \(i\) 个目标有 \(q_i\) 个攻击变体，\(r_{ij}=1\) 表示第 \(j\) 个变体仍然泄露：

$$
R_{robust}=
\frac{\sum_iw_i
\left(\frac{1}{q_i}\sum_{j=1}^{q_i}r_{ij}\right)}
{\sum_iw_i}
$$

先在每个目标内部平均，可以避免某个目标因为攻击模板多而支配总体分数。\(q_i=0\) 的目标不应进入这个分母。

### 16.3 Retain utility

保留集任务 \(R\) 的能力可以表示为：

$$
A_{retain}=
\frac{1}{|R|}\sum_{j\in R}s_j
$$

\(s_j\) 可以是准确率、F1、归一化 log-likelihood 或人工评分，但所有任务必须明确评分尺度。与遗忘前相比的变化为：

$$
\Delta_{retain}=A_{retain}^{after}-A_{retain}^{before}
$$

如果 \(\Delta_{retain}\) 大幅为负，说明模型可能是通过普遍破坏能力来降低 forget set 分数。

### 16.4 参考模型差异

将 \(D_{ref}\) 与泄露率、保留能力并列，可以把“像没见过目标数据”变成一个有限分布上的可检验命题。它仍需要声明：

1. reference model 的训练种子和配方。
2. \(T\) 是否包含近邻和多轮提示。
3. 是否控制数据重复和污染。
4. 输出分布的 tokenization 是否一致。

### 16.5 把指标变成可追踪的上线决定

工程系统不应把所有信号压成一个名为“通过”的总布尔值。更清楚的记录方式是：

| 证据项 | 信号 | 目标阈值 | 当前状态 | 下一动作 |
| --- | --- | --- | --- | --- |
| 目标编辑 | \(S_{target}\) | 不低于任务要求 | 达标或未达标 | 补充目标样本 |
| 改写泛化 | \(S_{para}\) | 不低于任务要求 | 达标或未达标 | 扩展表达族 |
| 局部性 | \(S_{local}\) | 不低于任务要求 | 达标或未达标 | 检查冲突邻域 |
| exact 泄露 | \(R_{forget}\) | 不高于风险上限 | 达标或未达标 | 扩大攻击集 |
| 稳健泄露 | \(R_{robust}\) | 不高于风险上限 | 达标或未达标 | 暂停外部暴露 |
| 保留能力 | \(\Delta_{retain}\) | 不低于退化上限 | 达标或未达标 | 降低更新强度或重训 |

“当前状态”只说明已测样本和已知攻击下的证据；“下一动作”说明不确定性如何被减少。它不是把复杂证据伪装成一个总开关。

## 17. 案例一：企业事实过期，应该编辑还是 RAG

### 17.1 需求

一个企业助手经常回答组织信息。负责人变更后，产品团队发现旧回答仍然出现。需求是：

1. 新文档立即生效。
2. 回答带来源和生效日期。
3. 旧文档不能覆盖新文档。
4. 无权限员工不能看到内部字段。
5. 将来可以恢复错误更新。

这不是单纯的 model editing 问题，而是时效、来源、权限和回滚问题。

### 17.2 第一层方案：带时间和权限的 RAG

知识库中保存：

| 字段 | 示例 |
| --- | --- |
| subject | 产品 A |
| relation | current_owner |
| object | 员工 B |
| valid_from | 2026-07-01 |
| source_id | policy-2026-07-01 |
| access_scope | department-x |

检索器先按权限过滤，再按有效时间和来源可信度排序。生成器必须引用检索证据，不能在没有证据时直接使用参数记忆。

这样解决了更新快、可追溯和权限控制，但仍要测试模型在检索冲突、空结果和恶意文档中的行为。

### 17.3 第二层方案：少量参数编辑

如果离线产品明确要求模型在无检索条件下也修正一个长期稳定事实，可以在冻结的模型副本上做 ROME 或其他编辑实验。在线服务同时保留 RAG，以处理时间变化和权限差异。

编辑的验收数据应包括：

1. 新事实的直接和改写问法。
2. 历史负责人问题，确认时间条件仍然可表达。
3. 其他产品的负责人。
4. 员工 B 的部门和角色等邻近事实。
5. 无权限请求。
6. 来源引用和日期表达。

如果参数编辑提高了新事实成功率，却让历史事实全部被改写成当前事实，就应该撤回编辑，保留 RAG 的时间建模。

### 17.4 这个案例的结论

RAG 解决的是外部知识的可更新性；editing 解决的是参数行为的局部修正。二者可以组合，但不能互相冒充。企业事实系统的责任链最终仍然要落在数据源、权限、引用、审计和回滚上。

## 18. 案例二：版权或隐私删除请求

### 18.1 先定义删除范围

收到删除请求时，第一步不是立即对权重做梯度更新，而是确定请求覆盖：

1. 原始上传文件。
2. 清洗后的副本。
3. 去重索引和特征缓存。
4. 预训练数据快照。
5. SFT、偏好和评估数据。
6. 已发布模型、adapter 和编辑记忆。
7. 推理日志、prompt cache 和备份。

源数据删除、模型 unlearning 和日志保留有不同的生命周期。技术团队不能只处理其中一个对象，然后宣称整项请求已经完成。

### 18.2 小型闭环

一个可审计的闭环可以是：

~~~text
识别请求
  -> 定位数据谱系和副本
  -> 停止继续使用目标数据
  -> 从索引、缓存和待训练队列移除
  -> 选择重训、局部遗忘或暂时抑制
  -> 在授权数据上做 forget/retain/攻击评估
  -> 记录版本、范围、残余风险和后续复核
~~~

其中“选择局部遗忘”需要充分说明它是近似技术。若责任要求高、目标规模大或数据影响无法定位，重新训练或不再部署受影响模型可能比参数补丁更容易解释。

### 18.3 版权内容和公共知识的边界

删除一部作品的逐字复现能力，不等于删除所有关于该主题的公共知识。反过来，模型不再复现原文，也不代表没有保留人物、情节或风格的抽象信息。

评估集应把：

1. 逐字记忆。
2. 事实记忆。
3. 改写复述。
4. 风格相似性。
5. 独立公共知识。

分开报告。法律结论应由适用法域和专业人员判断，技术指标只能说明系统行为证据。

## 19. 案例三：降低危险能力而不损害正常能力

### 19.1 能力和拒答的区别

一个模型可能在内部具备某种知识，但在系统策略下拒绝回答。也可能模型确实缺乏某个知识簇，却仍能通过工具或上下文完成相关任务。

因此安全团队要分别测：

1. 受控危险能力代理集。
2. 安全领域的正常教育和分析任务。
3. 相邻但不危险的代码、数学或生物基础任务。
4. 直接输出、工具调用和多轮任务。
5. 外部策略、权限和审计是否正常工作。

### 19.2 WMDP 类评估的正确用法

WMDP 公开了不包含敏感操作细节的代理题目，适合比较模型在特定能力维度上的变化。研究人员应使用其官方数据和安全协议，不应把教材中的分数转写成现实危险能力的绝对量化。

如果一个 unlearning 方法让 WMDP 分数下降 20%，同时一般计算机科学任务下降 18%，这更像整体能力损伤；如果 WMDP 下降而保留任务稳定，证据更有价值，但仍需看迁移、工具和后续微调。

### 19.3 外部控制仍不可省略

危险能力治理不能只依赖模型内部变化。权限、沙箱、网络隔离、工具白名单、人工复核和审计日志是最后的现实责任层。模型编辑或 unlearning 失败时，系统控制应仍能阻止不可逆副作用。

## 20. 工程实现：把一次修改变成可回滚的变更

### 20.1 变更清单

每条编辑或遗忘请求至少保存：

~~~yaml
change_id: edit-2026-0007
base_model: model-revision-id
request_type: factual_edit
target_scope: product-owner-relation
source_reference: authorized-policy-id
created_at: 2026-08-10T10:00:00+08:00
method: serac-or-rome
evaluation_revision: eval-2026-08-10-a
rollback_artifact: checkpoint-before-edit
owner: team-name
~~~

遗忘请求还要保存 forget scope 的哈希或受控引用，不要把真实隐私内容直接写入普通日志。

### 20.2 离线、只读影子和有限发布

一个新编辑先在离线 checkpoint 上评估。通过后可以让只读影子流量同时运行新旧模型，比较目标、改写、局部性、引用和延迟。

影子阶段不能产生真实工具副作用。若模型回答涉及外部动作，必须把工具调用替换为审计事件或模拟结果。

有限发布时要绑定模型 revision、编辑清单和评估 revision。出现回归时，能够按变更清单回退单条编辑或整个版本。

### 20.3 冲突处理

两条编辑可能满足以下关系：

1. 同一 subject、同一 relation、不同时间。
2. 同一 subject、不同 relation。
3. 不同 subject、共享别名。
4. 一条编辑依赖另一条编辑。

系统需要明确优先级：时间、来源可信度、权限范围和人工确认都可能比字符串相似度更重要。无法判定时应保留冲突状态，让应用返回需要确认，而不是选择一个看似最相近的答案。

### 20.4 回滚不是重新编辑

如果把错误的编辑 B 改成 C，不一定能恢复 A。第二次编辑可能已经改变了内部表示和相邻行为。可靠回滚应使用原始 checkpoint、可逆 adapter 或显式 edit memory 的删除操作，并重新跑原始评估集。

## 21. 一个最小的编辑与遗忘审计 demo

下面的代码只模拟审计数据，不加载模型，也不修改权重。它把编辑和遗忘分成两套证据，分别记录 thresholds、signals、evidence_status、actions 和 decision。这样可以看见哪个指标未达要求，而不是用一个总布尔值掩盖失败原因。

~~~python
def weighted_mean(rows, key):
    if not rows:
        return None
    total_weight = sum(row["weight"] for row in rows)
    if total_weight <= 0:
        raise ValueError("total weight must be positive")
    if any(row[key] is None for row in rows):
        return None
    return round(
        sum(row[key] * row["weight"] for row in rows) / total_weight,
        3,
    )


def weighted_flag_rate(rows, key):
    return weighted_mean(
        [
            {
                "weight": row["weight"],
                "value": 1.0 if row[key] else 0.0,
            }
            for row in rows
        ],
        "value",
    )


def weighted_nested_mean(rows, key):
    values = []
    for row in rows:
        nested = row[key]
        if not nested:
            return None
        values.append(
            {
                "weight": row["weight"],
                "value": sum(nested) / len(nested),
            }
        )
    return weighted_mean(values, "value")


def compare(signal, rule):
    if signal is None:
        return False
    operator = rule["operator"]
    target = rule["value"]
    if operator == ">=":
        return signal >= target
    if operator == "<=":
        return signal <= target
    raise ValueError(f"unsupported operator: {operator}")


edit_cases = [
    {
        "id": "owner_fact",
        "weight": 3,
        "target_ok": True,
        "paraphrase_ok": [1, 1, 0],
        "locality_ok": [1, 1, 1],
        "retain_before": 0.91,
        "retain_after": 0.90,
    },
    {
        "id": "product_fact",
        "weight": 2,
        "target_ok": True,
        "paraphrase_ok": [1, 1, 1],
        "locality_ok": [1, 0, 1],
        "retain_before": 0.88,
        "retain_after": 0.86,
    },
    {
        "id": "city_fact",
        "weight": 1,
        "target_ok": False,
        "paraphrase_ok": [0, 0, 0],
        "locality_ok": [1, 1],
        "retain_before": 0.90,
        "retain_after": 0.89,
    },
    {
        "id": "policy_behavior",
        "weight": 4,
        "target_ok": True,
        "paraphrase_ok": [1, 0, 1],
        "locality_ok": [1, 1, 0],
        "retain_before": 0.86,
        "retain_after": 0.80,
    },
]

editing_signals = {
    "target_success": weighted_flag_rate(edit_cases, "target_ok"),
    "paraphrase_generalization": weighted_nested_mean(
        edit_cases,
        "paraphrase_ok",
    ),
    "locality": weighted_nested_mean(edit_cases, "locality_ok"),
    "retain_before": weighted_mean(edit_cases, "retain_before"),
    "retain_after": weighted_mean(edit_cases, "retain_after"),
}
editing_signals["retain_delta"] = (
    round(
        editing_signals["retain_after"]
        - editing_signals["retain_before"],
        3,
    )
    if editing_signals["retain_after"] is not None
    and editing_signals["retain_before"] is not None
    else None
)
editing_signals["conflict_cases"] = [
    row["id"]
    for row in edit_cases
    if row["target_ok"] and not all(row["locality_ok"])
]

editing_thresholds = {
    "target_success": {"operator": ">=", "value": 0.80},
    "paraphrase_generalization": {"operator": ">=", "value": 0.65},
    "locality": {"operator": ">=", "value": 0.75},
    "retain_delta": {"operator": ">=", "value": -0.05},
    "conflict_count": {"operator": "<=", "value": 1},
}

editing_evidence = {
    "target_success": compare(
        editing_signals["target_success"],
        editing_thresholds["target_success"],
    ),
    "paraphrase_generalization": compare(
        editing_signals["paraphrase_generalization"],
        editing_thresholds["paraphrase_generalization"],
    ),
    "locality": compare(
        editing_signals["locality"],
        editing_thresholds["locality"],
    ),
    "retain_delta": compare(
        editing_signals["retain_delta"],
        editing_thresholds["retain_delta"],
    ),
    "conflict_count": compare(
        len(editing_signals["conflict_cases"]),
        editing_thresholds["conflict_count"],
    ),
}

editing_actions = {
    "target_success": "add_direct_target_cases",
    "paraphrase_generalization": "expand_language_and_template_families",
    "locality": "inspect_neighborhood_conflicts",
    "retain_delta": "reduce_edit_strength_or_compare_retraining",
    "conflict_count": "review_conflicting_edits_before_release",
}

forget_cases = [
    {
        "id": "synthetic_profile",
        "weight": 3,
        "exact_leak": False,
        "paraphrase_leak": False,
        "multi_turn_leak": False,
        "mia_before": 0.82,
        "mia_after": 0.31,
    },
    {
        "id": "private_note_proxy",
        "weight": 4,
        "exact_leak": False,
        "paraphrase_leak": True,
        "multi_turn_leak": False,
        "mia_before": 0.91,
        "mia_after": 0.42,
    },
    {
        "id": "capability_proxy",
        "weight": 5,
        "exact_leak": True,
        "paraphrase_leak": True,
        "multi_turn_leak": True,
        "mia_before": 0.88,
        "mia_after": 0.67,
    },
]

retain_tasks = [
    {"id": "general_language", "weight": 3, "before": 0.88, "after": 0.86},
    {"id": "safe_qa", "weight": 4, "before": 0.90, "after": 0.87},
    {"id": "math", "weight": 2, "before": 0.78, "after": 0.77},
    {"id": "code", "weight": 2, "before": 0.74, "after": 0.70},
]

unlearning_signals = {
    "exact_leakage": weighted_flag_rate(forget_cases, "exact_leak"),
    "paraphrase_leakage": weighted_flag_rate(
        forget_cases,
        "paraphrase_leak",
    ),
    "multi_turn_leakage": weighted_flag_rate(
        forget_cases,
        "multi_turn_leak",
    ),
    "mia_before": weighted_mean(forget_cases, "mia_before"),
    "mia_after": weighted_mean(forget_cases, "mia_after"),
    "retain_before": weighted_mean(retain_tasks, "before"),
    "retain_after": weighted_mean(retain_tasks, "after"),
}
unlearning_signals["robust_leakage"] = weighted_nested_mean(
    [
        {
            "weight": row["weight"],
            "variants": [
                1.0 if row[key] else 0.0
                for key in (
                    "exact_leak",
                    "paraphrase_leak",
                    "multi_turn_leak",
                )
            ],
        }
        for row in forget_cases
    ],
    "variants",
)
unlearning_signals["mia_delta"] = (
    round(
        unlearning_signals["mia_after"]
        - unlearning_signals["mia_before"],
        3,
    )
    if unlearning_signals["mia_after"] is not None
    and unlearning_signals["mia_before"] is not None
    else None
)
unlearning_signals["retain_delta"] = (
    round(
        unlearning_signals["retain_after"]
        - unlearning_signals["retain_before"],
        3,
    )
    if unlearning_signals["retain_after"] is not None
    and unlearning_signals["retain_before"] is not None
    else None
)

unlearning_thresholds = {
    "exact_leakage": {"operator": "<=", "value": 0.10},
    "robust_leakage": {"operator": "<=", "value": 0.20},
    "mia_delta": {"operator": "<=", "value": -0.30},
    "retain_delta": {"operator": ">=", "value": -0.05},
}

unlearning_evidence = {
    name: compare(unlearning_signals[name], rule)
    for name, rule in unlearning_thresholds.items()
}

editing_undefined_metrics = [
    name
    for name, value in editing_signals.items()
    if name != "conflict_cases" and value is None
]
unlearning_undefined_metrics = [
    name for name, value in unlearning_signals.items() if value is None
]

unlearning_actions = {
    "exact_leakage": "expand_forget_set_and_check_output_variants",
    "robust_leakage": "keep_external_exposure_restricted_and_add_attack_families",
    "mia_delta": "repeat_with_calibrated_membership_attack",
    "retain_delta": "compare_retain_only_reference_and_reduce_update_strength",
}

decision = {
    "editing": {
        "scope": "offline_checkpoint",
        "status": "hold_for_conflict_review",
        "evidence": editing_evidence,
        "undefined_metrics": editing_undefined_metrics,
        "next_actions": editing_actions,
    },
    "unlearning": {
        "scope": "synthetic_audit_only",
        "status": "hold_for_robust_leakage",
        "evidence": unlearning_evidence,
        "undefined_metrics": unlearning_undefined_metrics,
        "next_actions": unlearning_actions,
    },
}

print("editing_signals=", editing_signals)
print("editing_evidence=", editing_evidence)
print("editing_undefined_metrics=", editing_undefined_metrics)
print("unlearning_signals=", unlearning_signals)
print("unlearning_evidence=", unlearning_evidence)
print("unlearning_undefined_metrics=", unlearning_undefined_metrics)
print("decision=", decision)
~~~

这段程序的结果故意不会全部满足阈值。它应该暴露两个独立事实：

1. 编辑目标大多成功，但产品事实和政策行为仍有局部冲突。
2. exact 泄露和成员推断风险有所下降，但 capability proxy 在改写和多轮中仍然泄露。

程序中的布尔值只描述单个证据项是否达到事先写明的阈值。真正的 decision 仍然保留范围、状态、下一动作和证据明细，避免把实验结果伪装成普遍保证。

## 22. 常见的错误理解

### 22.1 “Model editing 就是给模型做一次微调”

两者都可能用梯度，但研究对象不同。微调通常以一组数据改变整体行为分布；model editing 强调目标范围和局部保持。实际方法也可能使用梯度，因此不能仅按优化器名称区分。

### 22.2 “RAG 已经能解决所有知识更新”

RAG 很适合时效性、权限和引用，但检索失败、上下文冲突和参数旧知识仍然存在。对于某些无上下文场景，局部 editing 可能有价值；对于数据删除，RAG 也不能替代模型和源数据治理。

### 22.3 “模型不回答就已经忘了”

不回答可能来自拒答策略、关键词过滤、采样变化或上下文缺失。要证明更强的遗忘，需要改写、多轮、间接提示、保留集、MIA 和参考模型等多维证据。

### 22.4 “参数变化越小，副作用越小”

参数范数只是一个代理量。神经网络的局部参数变化可能影响关键方向；参数变化较大也可能集中在低影响子空间。行为差分和邻域评估不能省略。

### 22.5 “一个公开基准分数就是通用结论”

CounterFact、TOFU、WMDP 和 MUSE 各自定义了不同的数据、任务和指标。基准越专业，越需要说明它覆盖了什么、没有覆盖什么，以及是否存在模板过拟合或数据重复。

### 22.6 “第二次编辑可以自动纠正第一次编辑”

连续参数编辑可能留下不可逆的交互效应。可靠的修正应从可回滚的原始版本或显式编辑记忆开始，并重新执行完整评估。

### 22.7 “技术上能删，所以法律上就完成了”

法律中的删除对象、保留义务、证明责任和适用法域不是一个模型指标能够决定的。技术团队应提供数据范围、版本、处理记录、残余风险和评估证据，不能代替法律判断。

## 23. 练习：把概念变成可检查的设计

### 练习一：选择知识更新层

某企业的费用政策每月变化，并要求员工看到当前版本和来源。请比较 RAG、继续训练和 model editing，给出选择和两个反例。

要求说明：

1. 变化频率。
2. 是否需要权限。
3. 是否需要引用。
4. 哪些旧参数行为仍可能干扰。
5. 如何验证空检索和冲突文档。

### 练习二：设计一次事实编辑

为一个虚构公司的负责人变更建立编辑集。至少写出五个直接问题、五个改写问题、五个同实体不同关系的问题和五个无关保留任务。

然后分别计算 \(S_{target}\)、\(S_{para}\) 和 \(S_{local}\)，说明每个分母代表什么。

### 练习三：比较 ROME 和 SERAC

假设需求是每天产生少量、可撤销、需要来源的产品事实变更。分别说明 ROME 和 SERAC 的存储位置、推理成本、回滚方式和残余旧知识风险。

### 练习四：设计虚构数据的 unlearning 实验

构造一组不含真实隐私的虚构人物资料，将其中一部分设为 forget set。建立 retain-only reference，分别设计 exact、改写、多轮、近邻和通用能力测试。

实验报告必须写出：

1. 目标数据和训练谱系。
2. reference model 如何训练。
3. forget、retain、attack 三类数据如何隔离。
4. 输出日志如何避免再次泄露目标内容。
5. 何种结果会触发回滚或重新训练。

### 练习五：安全能力的证据边界

使用公开的能力代理评估，不接触危险操作细节，分析“目标分数下降、正常能力保持、外部策略仍然有效”三种证据之间的关系。写出至少两个仍然无法得出的结论。

## 24. 资料、证据等级与可复核范围

### 24.1 论文和项目页

以下资料直接支持本章的机制描述和基准定义：

1. [ROME: Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)：论文与项目页支持因果定位、middle-layer feed-forward 关联、rank-one editing、CounterFact 和 zsRE 实验叙述。
2. [MEMIT: Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229)：支持从单个关联扩展到批量 memory edits，以及在 GPT-J 6B、GPT-NeoX 20B 上的论文实验范围。
3. [MEND: Fast Model Editing at Scale](https://arxiv.org/abs/2110.11309)：支持用梯度分解和学习型编辑器进行快速局部编辑的机制概述。
4. [SERAC: Memory-Based Model Editing at Scale](https://arxiv.org/abs/2206.06520)：支持显式 edit memory、相关性范围判断和 counterfactual model 的半参数路线。
5. [TOFU: A Task of Fictitious Unlearning for LLMs](https://arxiv.org/abs/2401.06121)：支持虚构作者资料、forget/retain 设计和基线遗忘效果的限制。
6. [WMDP: Measuring and Reducing Malicious Use With Unlearning](https://arxiv.org/abs/2403.03218)：支持危险知识代理评估、3,668 道公开选择题和 RMU 研究路线。
7. [MUSE: Machine Unlearning Six-Way Evaluation for Language Models](https://arxiv.org/abs/2407.06460)：支持逐字记忆、知识记忆、隐私、效用、规模和连续请求六个评估维度。

论文摘要、论文正文和项目页的证据强度不同。摘要适合核对论文主张和数据规模，正文适合核对公式与实验条件，项目页适合核对代码和数据入口。它们都不能自动推出商业模型的通用效果。

### 24.2 治理资料

[NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) 可以支持风险识别、测量、管理和治理的组织化讨论。它不规定某个 editing 或 unlearning 算法一定有效，也不替代适用法域的隐私和版权判断。

工程记录中的“来源可信”至少要区分：

| 证据来源 | 能支持的结论 | 不能支持的结论 |
| --- | --- | --- |
| 论文 | 方法定义、实验条件、论文报告结果 | 对未测模型和生产流量的保证 |
| 基准项目页 | 数据集结构、代码和任务说明 | 法律上的删除完成 |
| 模型卡或官方文档 | 公开模型能力和接口范围 | 未披露的内部参数编辑机制 |
| 企业内部审计 | 该版本、该数据和该流量上的行为 | 对未来版本和所有输入的保证 |
| 法规或政策文件 | 义务、权利和治理要求的文本背景 | 技术方案自动合规 |
| 社区文章 | 线索和复现实验入口 | 权威事实和跨版本定律 |

### 24.3 本章的证据边界

本章可以严谨地说：

1. 某些研究在特定模型和基准上展示了局部事实编辑。
2. 批量编辑、学习型编辑器和显式编辑记忆分别代表不同的工程路线。
3. 现有 LLM unlearning 通常是近似、有限评估分布上的证据。
4. TOFU 和 MUSE 等工作揭示了遗忘与保留、隐私、规模和持续性的张力。
5. WMDP 等公开代理基准可以帮助研究能力变化，但不能代替现实风险评估。

本章不能严谨地说：

1. 某个方法对所有大模型都能安全编辑。
2. 某次输出拒答已经证明参数中没有目标知识。
3. 某个 benchmark 通过就代表法律删除完成。
4. 某个公开模型的未披露内部结构采用了某种编辑机制。
5. 一次 unlearning 成功就能保证后续微调、工具调用和多轮上下文中仍然遗忘。

## 25. 本章小结

Model editing 解决的是局部改变模型知识或行为的问题，核心要求不是只让一个目标问题答对，而是同时考察改写泛化、局部性、关系一致性和通用能力保留。

ROME 用因果定位和 rank-one 权重更新研究一类 factual association；MEMIT 把相似思路扩展到批量关联；MEND 通过学习型编辑器变换梯度；SERAC 把编辑保存在显式记忆中，用相关性判断决定何时使用反事实模型。

Unlearning 的参照对象是“只在保留数据上训练的模型”，而不是一个可以直接从参数中擦掉的字符串。数据删除、知识删除、行为抑制和能力降低必须分别定义、分别评估。

高质量 unlearning 评估至少要同时覆盖 forget set、改写和多轮攻击、retain set、成员推断、近邻能力、连续删除和参考模型差异。TOFU、WMDP 和 MUSE 说明，直接降低目标输出远远不够。

真实系统中，RAG、数据治理、参数编辑、显式编辑记忆、输出策略、权限、审计和回滚承担不同责任。技术方案的可信度来自清楚的范围、可复核的证据和可恢复的版本，而不是来自一个听起来确定的“已经忘记”。
