# 第七章：Interpretability 与 Mechanistic Interpretability

重点：特征可解释性、activation patching、circuits、SAE、模型内部机制分析。

面试重点：Interpretability 不是“画几张 attention heatmap”，Mechanistic Interpretability 的目标是像逆向工程程序一样理解模型内部实际实现了什么算法。

## 本章目标

学完本章，你要能回答：

1. Interpretability 和 Mechanistic Interpretability 有什么区别？
2. 为什么普通可解释性方法不足以支撑模型安全？
3. Circuits、features、neurons、superposition、polysemanticity 分别是什么意思？
4. Activation patching、ablation、causal tracing 解决什么问题？
5. Sparse Autoencoder 为什么成为 LLM 可解释性的重要方向？
6. Mechanistic interpretability 对 alignment 和 safety 有什么价值？
7. 这个方向当前有哪些局限和未来可能演化？

## 1. 来龙去脉：为什么需要可解释性

### 1.1 传统机器学习时代的问题

在传统机器学习中，很多模型本身就比较可解释。

例如：

1. 线性回归可以看权重。
2. 决策树可以看分裂路径。
3. 逻辑回归可以看特征系数。
4. 朴素贝叶斯可以看条件概率。

这些模型虽然能力有限，但人能大致理解它为什么给出某个结果。

### 1.2 深度学习带来的黑箱问题

深度神经网络能力强很多，但内部表示变得难懂。

一个 Transformer 可能有：

1. 数十亿参数。
2. 多层 attention。
3. MLP 中间表示。
4. 残差流。
5. 多头注意力。
6. 非线性特征组合。

模型回答一个问题时，我们很难直接知道：

1. 它用了哪些信息？
2. 它是否真的推理了？
3. 它是否只是在模式匹配？
4. 它为什么幻觉？
5. 它内部是否有危险特征？
6. 它什么时候会拒答？

### 1.3 早期可解释性方法

早期 interpretability 常用方法包括：

1. Feature importance。
2. Saliency map。
3. Attention visualization。
4. Gradient-based attribution。
5. Probing classifier。
6. Example-based explanation。

这些方法有价值，但往往只能回答：

```text
哪些输入或激活和输出相关？
```

它们不一定能回答：

```text
模型内部到底实现了什么机制？
```

### 1.4 Mechanistic Interpretability 的出现

Mechanistic Interpretability 试图更进一步。

它不满足于“这个 token 重要”或“这个 attention head 看这里”。

它希望逆向工程模型内部计算。

目标类似：

```text
把神经网络当成一个被训练出来的程序，理解它内部用哪些特征、哪些线路、哪些子算法完成任务。
```

Distill 的 Circuits 系列在视觉模型上展示了一个重要思路：认真分析单个神经元、特征和它们之间的连接，可能可以逆向出模型学到的局部算法。

后来 Transformer Circuits、activation patching、causal tracing、SAE 等方法进一步把这种思路推向语言模型。

## 2. 小白例子：看答案、看草稿、看脑回路

假设一个学生做数学题。

你有三种理解方式。

第一，只看最终答案。

你知道对错，但不知道为什么。

第二，看草稿。

你能看到他用了哪些步骤。

第三，理解他的思维习惯。

你知道他遇到哪类题会想到什么方法，哪里容易错。

对应到模型：

1. 普通评估：看最终答案。
2. Attribution / attention：看部分中间痕迹。
3. Mechanistic interpretability：试图理解内部算法和因果机制。

这就是机制可解释性比普通可解释性更进一步的地方。

## 3. Interpretability 的层次

可解释性不是单一概念。

可以分成四层。

### 3.1 输入归因层

问题：哪些输入影响输出？

方法：

1. Saliency。
2. Gradient attribution。
3. Integrated gradients。
4. Token importance。

优点：容易实现。

缺点：相关性强，因果性弱。

### 3.2 表示分析层

问题：模型激活中是否编码了某些信息？

方法：

1. Probing。
2. Linear classifier。
3. Representation similarity。
4. PCA / UMAP。

优点：能看模型是否包含某类信息。

缺点：包含信息不代表模型实际使用该信息。

### 3.3 因果干预层

问题：某个激活、head、feature 是否因果影响输出？

方法：

1. Ablation。
2. Activation patching。
3. Causal tracing。
4. Path patching。

优点：更接近机制验证。

缺点：实验设计复杂，干预可能产生分布外激活。

### 3.4 机制逆向层

问题：模型内部实现了什么可理解算法？

方法：

1. Circuits analysis。
2. Feature decomposition。
3. SAE。
4. Mechanistic case study。
5. Weight and activation analysis。

优点：能形成较强机制解释。

缺点：成本高、扩展困难、解释可能不完整。

## 4. Features、Neurons 和 Circuits

### 4.1 Feature 是什么

Feature 可以理解为模型内部表示的某种概念或方向。

例如：

1. “这是 Python 代码”。
2. “这句话在表达拒绝”。
3. “这个 token 是人名”。
4. “这里需要引用前文实体”。
5. “这段内容涉及高风险安全策略”。

在理想情况下，一个 feature 对应一个清晰概念。

但真实模型更复杂。

### 4.2 Neuron 是什么

Neuron 是网络中的单个维度或单个激活单元。

早期可解释性常希望一个 neuron 对应一个 feature。

例如视觉模型里某些 neuron 可能检测曲线、纹理、颜色或物体部件。

### 4.3 Polysemanticity

Polysemanticity 指一个 neuron 在多个语义上不同的场景中都激活。

例如一个 neuron 可能同时对：

1. 某类代码语法。
2. 某类英文短语。
3. 某种文档格式。

都激活。

这会让“看单个 neuron”变得困难。

### 4.4 Superposition

Superposition 是解释 polysemanticity 的一个重要假设。

直觉是：模型需要表示的 feature 数量可能多于激活空间维度。

于是模型把多个 feature 叠加在同一个空间里，用不同方向表示。

小白可以这样理解：

```text
神经网络没有给每个概念分配一个独立抽屉，而是把很多概念压缩放进同一个房间，用不同方向区分。
```

这会导致单个 neuron 看起来混合多个意义。

### 4.5 Circuit 是什么

Circuit 是由多个 feature、neurons、attention heads、MLP 和连接组成的子网络，用来实现某个功能。

例如一个模型完成代词消解，可能需要：

1. 识别候选实体。
2. 记录语法位置。
3. 判断性别或数。
4. 把信息写入残差流。
5. 在输出位置读取相关信息。

这些组件共同形成 circuit。

## 5. Attention Head 可解释性

### 5.1 Attention 可视化的诱惑

Transformer 中 attention head 会给不同 token 分配权重。

很容易把 attention heatmap 当解释。

例如：

```text
模型看了哪个 token，所以它为什么这么回答。
```

但这很危险。

Attention 权重不一定等于因果解释。

### 5.2 Attention head 的功能类型

一些研究中，人们发现 attention head 可能形成某些功能模式。

例如：

1. Induction head：复制或延续之前出现过的模式。
2. Name mover head：把实体名称信息传到预测位置。
3. Previous token head：关注前一个 token。
4. Syntax-related head：关注语法相关位置。

这些名字帮助我们建立直觉，但不能把每个 head 都简单贴标签。

### 5.3 面向专家

Attention head 的解释要看它对残差流写入了什么信息，以及后续层如何读取这些信息。

只看 attention pattern 不够。

更完整分析需要：

1. Attention pattern。
2. Value vector 写入内容。
3. Output projection。
4. 残差流中的信息流。
5. 下游 head 或 MLP 的读取。
6. 因果干预验证。

## 6. Activation Patching

### 6.1 核心思想

Activation patching 用来判断某个中间激活是否对输出有因果作用。

基本思路：

1. 准备一个 clean input，模型输出正确。
2. 准备一个 corrupted input，模型输出错误或不同。
3. 把 clean run 中某个位置的激活替换到 corrupted run。
4. 看输出是否恢复。

如果恢复，说明这个激活可能携带关键因果信息。

### 6.2 小白例子

假设一个机器坏了。

你不知道哪个零件坏。

你从正常机器上拆一个零件，替换到坏机器上。

如果机器恢复，说明这个零件很关键。

Activation patching 就是对模型内部激活做类似替换。

### 6.3 它解决什么问题

普通 probing 只能说某个激活里“有信息”。

Activation patching 更进一步问：

```text
这个信息是否真的影响模型输出？
```

### 6.4 局限

1. 需要构造 clean/corrupted 对。
2. 干预可能产生模型训练时没见过的激活组合。
3. 恢复输出不等于完整理解机制。
4. 大模型中搜索空间很大。
5. 多组件共同作用时，单点 patch 可能误导。

## 7. Ablation 和 Causal Tracing

### 7.1 Ablation

Ablation 是把某个组件移除、置零或替换，看模型行为如何变化。

例如：

1. 去掉某个 attention head。
2. 置零某层 MLP 输出。
3. 禁用某个 feature。
4. 删除某条路径。

如果性能显著下降，说明组件重要。

### 7.2 Causal Tracing

Causal tracing 更关注信息在模型中的传播路径。

例如模型回答一个事实问题时：

1. 哪一层读取实体？
2. 哪一层储存事实关联？
3. 哪一层把信息传到输出？

它试图把模型行为拆成因果链。

### 7.3 和 Activation Patching 的关系

Activation patching 是因果干预工具。

Causal tracing 是用这些干预工具追踪信息流的一类分析。

Ablation 则更像移除组件看影响。

三者常一起使用。

## 8. Sparse Autoencoder

### 8.1 为什么需要 SAE

如果 neuron 是 polysemantic 的，单看 neuron 就不够。

Superposition 假设说，feature 可能藏在激活空间的方向里，而不是单个 neuron 里。

Sparse Autoencoder 的思路是：从模型激活中学习一组稀疏 feature，把原始激活重构出来。

目标是找到更可解释、更接近 monosemantic 的 feature。

### 8.2 小白例子

想象一段音乐里混着很多乐器。

单个麦克风通道可能同时包含钢琴、小提琴和鼓声。

SAE 想做的是把混合信号分解成更独立的乐器轨道。

在 LLM 中：

```text
混合激活 -> SAE -> 稀疏 feature 激活 -> 重构原激活
```

### 8.3 基本结构

一个简化 SAE：

```text
activation x -> encoder -> sparse feature z -> decoder -> reconstructed activation x_hat
```

训练目标包括：

1. 重构好。
2. Feature 稀疏。

简化形式：

```text
loss = reconstruction_loss + sparsity_penalty
```

### 8.4 它解决前人什么问题

前人问题：单个 neuron 常常 polysemantic。

SAE 试图在更高维 feature 空间中找到更干净的概念方向。

这让研究者可以：

1. 找到更可解释 feature。
2. 定位某些行为相关 feature。
3. 做 feature-level ablation。
4. 做 steering 或 safety 相关分析。

### 8.5 Gated SAE 等后来者

普通 SAE 中，稀疏惩罚可能导致 feature activation 被系统性低估，也就是 shrinkage。

Gated SAE 的思路是把“是否使用某个方向”和“这个方向的幅度是多少”分开处理，从而减少稀疏惩罚带来的副作用。

这体现了后来者的改进方向：

1. 更好的重构质量。
2. 更少 feature 同时激活。
3. 更高可解释性。
4. 更少训练偏差。
5. 更适合大模型规模。

### 8.6 面向专家

SAE 的核心假设是 activation 中存在稀疏、可线性组合的 feature basis。

关键 trade-off：

1. Reconstruction fidelity vs sparsity。
2. Feature interpretability vs feature splitting。
3. Overcomplete dictionary size vs compute。
4. Dead features vs too many active features。
5. Automated interpretability metric vs human understanding。

SAE 给了 mechanistic interpretability 可扩展工具，但不自动等于完整机制解释。

找到 feature 之后，还需要验证它是否因果影响模型行为。

## 9. Mechanistic Interpretability 和 Safety

### 9.1 为什么对安全有价值

Safety 和 alignment 不只关心模型输出。

还关心模型内部是否有：

1. 欺骗相关表示。
2. 拒答相关机制。
3. 危险知识调用路径。
4. 不确定性表达机制。
5. 目标错配特征。
6. Prompt injection 服从特征。

Mechanistic interpretability 试图让我们不只看外部行为，还能理解内部机制。

### 9.2 可能应用

1. 解释模型为什么幻觉。
2. 找到拒答机制。
3. 定位某些安全策略相关 feature。
4. 分析 jailbreak 为什么成功。
5. 做 model editing 或 steering。
6. 监控危险 feature 激活。
7. 辅助 red teaming 和 safety eval。

### 9.3 不能夸大

当前 interpretability 还不能保证：

1. 完整读懂大模型。
2. 证明模型绝对安全。
3. 发现所有隐藏目标。
4. 替代红队和评估。
5. 自动解决 alignment。

面试中要避免夸张。

更稳妥的表述是：

```text
Mechanistic interpretability 是理解和治理模型的重要工具，但目前更适合作为安全评估、调试和研究的一部分，而不是单独的安全保证。
```

## 10. Grokking 和机制分析

Grokking 指模型在训练过程中先记忆训练集，后来突然泛化变好。

从外部看，它像突然涌现。

Mechanistic interpretability 的一个价值是：它可能揭示看似突然的行为背后，其实有连续发展的内部机制。

Progress measures for grokking 这类工作通过分析小型 Transformer 在 modular addition 上学到的机制，展示了如何把“突然泛化”拆成更连续的内部过程。

这给大模型研究一个启发：

1. 外部指标可能突然变化。
2. 内部机制可能逐步形成。
3. 如果能找到 progress measure，就能更早预测能力变化。
4. 对安全来说，这可能帮助提前发现危险能力或行为倾向。

## 11. 真实项目中的使用方式

### 11.1 Debug 模型行为

当模型出现稳定失败时，可以问：

1. 哪些 token 或上下文影响最大？
2. 哪些 layer/head/MLP 参与？
3. 替换激活能否恢复正确行为？
4. 是否存在可解释 feature？
5. 是否能通过 patch 或 ablation 验证？

### 11.2 分析安全策略

例如模型拒答异常。

可以分析：

1. 拒答相关 feature 是否过度激活？
2. 正常请求和危险请求在激活上有什么差异？
3. Jailbreak 是否绕过了某些安全 circuit？
4. Safety tuning 是否改变了关键表示？

### 11.3 支持模型编辑和 steering

如果找到某个 feature 和行为相关，可以尝试：

1. 增强安全相关特征。
2. 抑制不希望的风格或行为。
3. 做 activation steering。
4. 指导 model editing。

但必须验证副作用。

## 12. 方法局限

### 12.1 扩展性问题

大模型非常大。

逐个 circuit 分析成本高。

### 12.2 解释不完整

解释一个 circuit 不等于解释整个模型。

### 12.3 人类解释偏差

研究者可能给 feature 起一个看似合理但不完整的名字。

### 12.4 因果性困难

相关性不等于因果。

需要 patching、ablation 等验证。

### 12.5 分布外干预

修改激活可能产生模型训练中没见过的状态。

### 12.6 安全保证不足

即使理解了一部分机制，也不能证明没有其他危险机制。

## 13. 未来可能演化

### 13.1 从手工分析到自动化分析

未来需要更多自动化工具帮助发现 feature、命名 feature、验证 circuit。

### 13.2 从小模型到 frontier model

很多机制研究先在小模型上做。

挑战是迁移到大模型。

### 13.3 从解释到控制

可解释性最终可能服务于：

1. Steering。
2. Model editing。
3. Safety monitor。
4. Training-time diagnostics。
5. Release gate。

### 13.4 从局部解释到系统保证

长期目标可能是把局部机制解释、行为评估、红队和形式化约束结合，形成更强的安全证据。

但这仍是开放研究方向。

## 14. 面试官会怎么问

### 问题 1：Interpretability 和 Mechanistic Interpretability 有什么区别？

回答要点：

1. Interpretability 泛指理解模型行为和表示。
2. Mechanistic interpretability 更强调逆向工程内部机制。
3. 它希望找到 features、circuits 和因果路径。
4. 目标是解释模型如何实际计算，而不只是哪些输入相关。

标准回答：

```text
Interpretability 是广义可解释性，包括输入归因、attention 可视化、probing 等。Mechanistic interpretability 更进一步，把模型当作一个被训练出来的程序，试图逆向工程它内部的 features、circuits 和信息流，并通过 ablation、activation patching 等因果干预验证这些机制是否真的影响输出。
```

### 问题 2：为什么 attention heatmap 不等于解释？

回答要点：

1. Attention 权重只说明信息读取模式。
2. 不说明 value 写入了什么。
3. 不说明后续层如何使用。
4. 不一定有因果性。
5. 需要结合 patching、ablation 和路径分析。

### 问题 3：什么是 superposition？

回答要点：

1. 模型要表示的 feature 数可能多于维度。
2. 多个 feature 可能叠加在激活空间中。
3. 这会导致 neuron polysemantic。
4. SAE 试图从激活中分解出更可解释的 feature 方向。

### 问题 4：Activation patching 解决什么问题？

回答要点：

1. 判断某个激活是否因果影响输出。
2. 用 clean/corrupted 输入对。
3. 把 clean 激活 patch 到 corrupted run。
4. 看输出是否恢复。
5. 它比普通 probing 更接近因果验证。

### 问题 5：SAE 为什么重要？

回答要点：

1. 单个 neuron 常常 polysemantic。
2. Superposition 说明 feature 可能是空间方向。
3. SAE 用稀疏重构学习可解释 feature。
4. 可用于 feature-level analysis、ablation 和 steering。
5. 但仍需因果验证，不能自动等于完整机制解释。

## 15. 标准回答模板

面试中可以这样回答：

```text
我会把 interpretability 分成几个层次。最浅层是输入归因和 attention 可视化，它们能提示哪些输入相关，但因果性有限。再往上是 probing 和表示分析，能说明模型激活中是否编码某类信息，但不代表模型实际使用这些信息。更强的是 activation patching、ablation 和 causal tracing，它们通过干预中间激活判断因果作用。

Mechanistic interpretability 的目标是逆向工程模型内部机制，找到 features、circuits 和信息流。这里的难点包括 polysemanticity 和 superposition，即单个 neuron 可能混合多个概念，feature 可能不是 neuron 而是激活空间里的方向。SAE 是当前重要方向之一，它试图从激活中学习稀疏、可解释的 feature。

对 safety 来说，这些方法可能帮助我们理解幻觉、拒答、jailbreak、危险能力和 steering。但当前它还不能单独证明模型安全，更适合作为 red teaming、safety eval、模型调试和治理的一部分。
```

## 16. 常见误区

### 16.1 误区：Attention 可视化就是解释

纠正：attention pattern 只是线索，不是完整因果解释。

### 16.2 误区：Probing 证明模型使用了某信息

纠正：probing 证明信息可被读出，不等于模型实际用它决策。

### 16.3 误区：找到一个 feature 就理解了模型

纠正：模型行为通常由多个 feature 和 circuit 共同决定。

### 16.4 误区：SAE 自动解决可解释性

纠正：SAE 提供 feature decomposition，但还需要命名、验证和因果分析。

### 16.5 误区：Mechanistic interpretability 已经能保证安全

纠正：它很有潜力，但目前不能替代红队、评估、权限控制和治理。

## 17. 小练习

### 练习 1

用自己的话解释 interpretability 和 mechanistic interpretability 的区别。

要求包含：输入归因、probing、activation patching 和 circuits。

### 练习 2

解释为什么 attention heatmap 不能直接当作模型解释。

要求说明 value、output projection、后续层和因果验证。

### 练习 3

设计一个 activation patching 实验，分析模型在某个 factual QA 上为什么答错。

要求包含 clean input、corrupted input、patch 位置和恢复指标。

### 练习 4

用小白能懂的话解释 superposition 和 polysemanticity。

### 练习 5

讨论 SAE 在 safety 中的一个可能应用和一个风险。

## 18. 本章总结

Interpretability 是广义可解释性，Mechanistic Interpretability 更强调逆向工程模型内部机制。

传统归因、attention 可视化和 probing 有价值，但容易停留在相关性层面。

Activation patching、ablation 和 causal tracing 更关注因果作用。

Circuits 试图解释多个组件如何共同实现功能。

Polysemanticity 和 superposition 解释了为什么单个 neuron 往往难以解释。

Sparse Autoencoder 试图从激活空间中分解出更稀疏、更可解释的 feature，是当前 LLM 机制可解释性的重要方向。

Mechanistic interpretability 对 safety 有潜在价值，可以帮助理解幻觉、拒答、jailbreak、steering 和危险能力，但目前仍不能单独提供安全保证。

面试中要把它讲成一个方法谱系：从输入归因，到表示分析，到因果干预，再到机制逆向和安全治理。
