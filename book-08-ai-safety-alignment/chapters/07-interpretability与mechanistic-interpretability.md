# 第七章：Interpretability 与 Mechanistic Interpretability

一个模型给出正确答案，并不意味着我们知道它为什么正确；一个模型在某类输入上拒答，
也不意味着拒答是由稳定的安全机制产生。可解释性研究的价值，正在于把“输出发生了
什么”进一步追问为“哪些内部状态参与了计算”“改变哪个状态会改变结果”，以及“这个
解释在新输入和新模型版本上是否仍然成立”。

本章从四种证据逐层展开：输入归因提供线索，表示分析检查信息是否可被读出，因果干预
检验某个组件是否真正影响行为，机制逆向则尝试把多个组件组织成可检验的 feature、
circuit 和算法假设。这个顺序很重要：一张 attention 图可以帮助提出假设，却不能直接
承担因果结论；一个漂亮的 SAE feature 可以帮助定位方向，却不能自动证明模型已经被
理解。

```text
行为现象 -> 激活 / 表示 -> 因果干预 -> feature / circuit -> 安全调试或监控
```

后文的数学和代码都是教学级 toy example。它们用于说明实验对象、指标和因果解释的
边界，不代表某个真实模型的测量结果，也不把局部机制证据写成整体安全保证。读者可以
先用“小白视角”理解每种方法回答的问题，再用“专家视角”检查对照组、干预位置、分母、
分布外状态和证据可迁移性。

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

### 1.5 从“解释结果”到“解释计算”

可以把解释证据分成四个强度不同的层次。第一层只描述模型看起来关注了什么；第二层
说明某类信息能否从内部状态读出来；第三层通过干预观察行为是否随之改变；第四层还要
说明多个组件如何按特定顺序协作，并在新任务上通过预测性检验。

| 层次 | 主要问题 | 典型证据 | 仍然没有证明什么 |
| --- | --- | --- | --- |
| 描述 | 哪些输入或组件与输出同时出现 | saliency、attention pattern | 输入是因果原因 |
| 可读出 | 激活中是否包含某类信息 | probing、线性解码 | 模型决策使用了该信息 |
| 干预 | 改变该状态是否改变结果 | ablation、activation patching | 已经找到完整计算链 |
| 机制 | 哪些组件通过什么路径共同完成任务 | circuit 假设、路径干预、跨样本预测 | 模型没有其他等价或隐藏路径 |

专家在报告中应把这四类证据分栏，而不是把它们平均成一个“可解释性分数”。例如一个
probe 准确率很高，只能支持“信息可被某个线性读出器提取”；如果 patch 后目标 logit
没有变化，就不能把该信息写成模型实际使用的机制。反过来，单次 patch 成功也可能来自
非自然激活组合或捷径，仍需负对照、重复样本和组件组合实验。

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

### 4.6 Feature 是方向假设，不是现成的语义标签

在机制分析中，“feature”经常被口语化为“某个概念”，但更准确的说法是：它是一个可被
描述、定位并通过实验检验的表示方向或激活模式。若 \(h\) 是某层残差流，\(d_k\) 是第
\(k\) 个候选方向，一个简化的 superposition 表示可以写成：

$$
h\approx \sum_{k\in S}a_k d_k+\epsilon
$$

其中 \(S\) 是当前输入激活的 feature 子集，\(a_k\) 是各方向的幅度，\(\epsilon\) 是无法
由当前字典解释的残差。这个式子不是说真实模型一定使用线性稀疏字典，而是说明为什么
“一个 neuron 对应一个概念”的假设可能过于简单：多个方向可以共同投影到同一组 neuron，
同一概念也可能被多个方向分裂表示。

小白可以把 \(d_k\) 想成混合录音中的一条乐器轨道；专家则要继续问：方向是否在不同
上下文中稳定，激活是否只是词形或位置的代理，字典是否遗漏了重要残差，feature 之间
是否存在互相抑制或组合关系。给 feature 起名只是研究记录，只有跨样本预测和干预结果
才能提高这个名字的可信度。

### 4.7 Circuit 假设怎样变成可检验命题

一个可用的 circuit 描述至少包含四部分：输入条件、候选组件、信息传递路径和预测的
行为变化。比如“某个 name mover head 把实体信息写入输出位置”不是结论，而是一个假设，
它应当预测：移除该 head 后目标 logit 下降，替换其 value 路径后下降更明显，保留无关
路径的负对照不会产生同样变化，并且这个效应在未参与建模的同类句式中复现。

机制分析还要面对两种常见现象。其一是冗余：多个组件可以完成相似功能，单独 ablation
不一定使行为消失；其二是退化或补偿：模型可能在干预后调用另一条路径。因此“组件被
移除后性能下降”支持必要性，“注入该组件状态后行为恢复”支持一定的充分性，但两者
都不能单独证明整个 circuit 是唯一实现。报告应把必要性、充分性、路径特异性和跨样本
泛化分别记录。

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

### 5.3 机制与边界

Attention head 的解释要看它对残差流写入了什么信息，以及后续层如何读取这些信息。

只看 attention pattern 不够。

更完整分析需要：

1. Attention pattern。
2. Value vector 写入内容。
3. Output projection。
4. 残差流中的信息流。
5. 下游 head 或 MLP 的读取。
6. 因果干预验证。

### 5.4 QK 电路与 OV 电路

对一个 attention head，可以把计算先写成：

$$
A=\operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M\right),
\qquad
o=A V W_O
$$

这里 \(QK^{\mathsf T}\) 决定“哪些位置互相匹配”，通常被称为 QK 侧的路由或匹配
计算；\(V W_O\) 决定“从被读取位置写回什么内容”，通常被称为 OV 侧的写入计算。
因此，即使一张 heatmap 显示某个 head 关注了实体 token，也还要检查 value 向量是否
真的携带了目标实体、输出投影是否把它写到正确的残差方向，以及后续 unembedding 是否
把该方向转成目标 token 的 logit。

小白可以把它比作邮局：QK 像分拣规则，决定包裹从哪个地址取件；OV 像包裹内容和投递
方向，决定取回的东西如何写入当前位置。只看分拣记录，不能证明包裹内容就是最终答案。
专家还要控制位置、词形、频率和上下文长度等混杂因素，因为一个 head 可能同时利用
多个线索。

### 5.5 从“命名 head”到验证 head

Induction head、name mover head 和 previous-token head 是有用的研究命名，但命名不是
实验结论。一个 head 是否承担某功能，至少要通过三种检查：在目标任务上的行为效应，
对输入结构变化的选择性，以及对无关任务的副作用。只在一个 prompt 上看到 pattern
相似，最多得到一个候选标签。

例如，要研究一个可能的复制头，可以改变前缀中重复片段的位置、间隔和词汇，同时保留
目标结构；如果它仍然把信息写向预测位置，且对打乱重复关系的负对照不再产生同样效应，
这个功能假设才更有说服力。不同随机种子、不同样本模板和不同模型 checkpoint 的复现
也很重要，因为 head 的编号和功能可能随训练阶段或模型规模改变。

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

### 6.5 一个合格的 patching 实验

首先要定义什么叫“恢复”。对于分类任务，可以使用正确类别的 logit margin；对于生成
任务，可以使用目标序列的对数概率、答案 token 的 rank 或任务级成功率。只观察最终文本
是否改变会丢失细微效应，也容易受采样噪声影响。设 clean、corrupted 和 patched 的
目标差值分别为 \(\Delta^+\)、\(\Delta^-\) 和 \(\Delta^{patch}\)，常见的归一化恢复分数是：

$$
S_{patch}=\frac{\Delta^{patch}-\Delta^-}{\Delta^+-\Delta^-}
$$

分母接近零时，这个分数没有稳定含义；恢复分数大于 1 也不自动表示实验更好，可能是
干预带来了 clean run 中没有的额外效应。报告应同时给出原始 logit、绝对变化、样本数、
置信区间或重复运行范围，并保留未 patch 的 corrupted 对照。

其次要设计负对照。可以把激活 patch 到无关位置、使用同层不同样本的激活、打乱对应关系，
或用均值激活作为替代。若所有替代都提高目标分数，说明实验可能利用了通用偏置，而不是
定位了特定信息。对 layer、position、head 和 feature 大量搜索时，还要记录搜索网格和
未参与挑选的 holdout；只报告最好的一个 patch 会产生选择偏差。

### 6.6 Patching 能证明到哪一步

如果把 clean 激活放入 corrupted 输入后行为恢复，可以支持“该状态足以在这个上下文中
补回某部分信息”。它不一定支持“模型在正常运行时就是通过这条路径完成计算”，因为
干预可能绕过了上游编码，也可能把多个本来不相容的状态拼在一起。要接近机制结论，需
继续做 path patching、上游/下游联合干预和结构化负对照。

对专家而言，最有价值的报告不是一张 patch score 热图，而是一组可复核命题：哪个输入
差异被隔离，哪个组件被替换，预期哪些输出会变，哪些无关输出不应变，干预后的激活是否
落在训练分布附近，以及该效应能否迁移到新模板。这样才可能把相关性线索逐步提升为
有限作用域内的因果证据。

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

### 7.4 必要性、充分性与替代路径

这三个词经常被混在一起。若移除组件 \(c\) 后目标行为明显下降，得到的是“在当前
上下文和干预强度下，\(c\) 对行为具有必要性证据”。若把某状态注入到 corrupted run
后行为恢复，得到的是“该状态在当前背景下具有一定充分性证据”。二者都不是逻辑上的
绝对必要或绝对充分，因为模型可能存在冗余电路、条件触发器和可替代路径。

可以用一个简单的二维实验理解：一组实验只移除 \(c\)，另一组只补入候选状态 \(s_c\)，
再加上移除 \(c\) 后补入 \(s_c\) 的组合实验。若单独移除有效、单独补入有效、组合后仍
能恢复，机制解释更完整；若组合实验失败，说明上游编码、下游读取或状态兼容性仍有
遗漏。所有结论都要带着任务、层、位置、模型版本和干预实现方式。

在大模型上，ablation 还可能触发补偿。模型原本依赖两个冗余 head，移除一个后另一个
head 加强作用，最终输出变化很小；这不说明被移除的 head 不重要，只说明行为级指标被
补偿掩盖。实践中可以同时记录组件激活、路径贡献、目标 logit 和任务成功率，避免只
依据一个最终指标下结论。

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

### 8.6 机制与边界

SAE 的核心假设是 activation 中存在稀疏、可线性组合的 feature basis。

关键 trade-off：

1. Reconstruction fidelity vs sparsity。
2. Feature interpretability vs feature splitting。
3. Overcomplete dictionary size vs compute。
4. Dead features vs too many active features。
5. Automated interpretability metric vs human understanding。

SAE 给了 mechanistic interpretability 可扩展工具，但不自动等于完整机制解释。

找到 feature 之后，还需要验证它是否因果影响模型行为。

### 8.7 SAE 训练中的字典问题

SAE 的训练并不是把神经元逐一翻译成自然语言。它是在一批激活样本上学习一个字典，
因此数据分布会直接影响最终 feature。只用代码数据训练，得到的 feature 可能对代码结构
很清晰，却无法解释聊天、工具返回或多模态 token；只取模型已经成功的样本，又会漏掉
失败轨迹中的安全相关状态。训练 manifest 至少应记录模型 checkpoint、层和位置、激活
采样策略、上下文长度、词元分布、字典宽度、归一化方式、稀疏惩罚和评估集版本。

字典宽度增加时，模型可能把一个混合 feature 拆成多个更细方向，这叫 feature splitting；
如果某些字典项几乎从不激活，则形成 dead feature；若很多输入都触发一个 feature，它
可能只是常量、位置或频率的代理。于是“feature 数越多”“平均激活越少”都不是独立的
质量结论。需要同时观察重构、激活分布、死特征比例、跨切片稳定性、人工解释一致性和
干预副作用。

### 8.8 自动命名与因果验证的分工

自动解释器可以根据 feature 的 top activating examples 生成标签，帮助研究者浏览大规模
字典。但标签是压缩后的假设，可能只描述表面词形，或者把多个触发条件误合并。更稳妥
的流程是：先用自动方法提出候选描述，再用反例搜索检查边界，最后用 feature ablation
或 steering 验证它是否改变预期行为。

例如一个 feature 被命名为“拒答”，不能只因为 top examples 中出现拒答句式。还要检查
它对安全请求、正常否定句、引用他人拒答和工具错误消息的激活差异；若增加该 feature
只让文本更像拒答，却没有降低高风险任务的 unsafe compliance，标签就不能直接用于安全
监控。feature 的语义解释、行为相关性和安全因果性应当是三个独立字段。

### 8.9 关键公式与机制可解释性指标

设某层某位置的残差流激活为：

$$
r_{\ell,p}\in \mathbb{R}^{d}
$$

其中 \(\ell\) 表示层，\(p\) 表示 token 位置，\(d\) 是 hidden size。

**1. Clean / Corrupted 运行**

设 clean 输入为 \(x^+\)，corrupted 输入为 \(x^-\)。目标 logit 差可以写成：

$$
\Delta(x)=logit_y(x)-logit_{y'}(x)
$$

其中 \(y\) 是期望答案，\(y'\) 是干扰答案。

**2. Activation patching 恢复分**

把 clean run 的激活 \(r^+_{\ell,p}\) patch 到 corrupted run：

$$
x^-_{\ell,p \leftarrow r^+}
$$

恢复分：

$$
S_{patch}(\ell,p)=
\frac{\Delta(x^-_{\ell,p \leftarrow r^+})-\Delta(x^-)}
{\Delta(x^+)-\Delta(x^-)}
$$

如果 \(S_{patch}\) 接近 1，说明该位置激活能显著恢复目标行为。

**3. Ablation 影响分**

把组件 \(c\) 的输出置零或替换为均值，得到 \(x_{abl(c)}\)：

$$
E_{abl}(c)=\Delta(x)-\Delta(x_{abl(c)})
$$

它衡量移除组件后目标行为下降多少。需要注意，ablation 可能产生分布外状态。

**4. Path patching 贡献**

如果只 patch 从组件 \(a\) 到组件 \(b\) 的路径，可以得到路径贡献：

$$
S_{path}(a\to b)=
\frac{\Delta(x^-_{a\to b \leftarrow clean})-\Delta(x^-)}
{\Delta(x^+)-\Delta(x^-)}
$$

这用于区分“单个组件重要”和“某条信息流重要”。

**5. SAE 编码与重构**

一个简化 Sparse Autoencoder：

$$
z=\mathrm{ReLU}(W_{enc}r+b_{enc})
$$

$$
\hat r=W_{dec}z+b_{dec}
$$

训练目标：

$$
L_{SAE}=\|r-\hat r\|_2^2+\lambda \|z\|_1
$$

第一项要求重构好，第二项要求 feature 激活稀疏。

**6. 重构保真度**

$$
F_{rec}=1-\frac{\sum_i \|r_i-\hat r_i\|_2^2}{\sum_i \|r_i-\bar r\|_2^2}
$$

它类似解释方差比例。保真度太低时，feature 解释可能只是漂亮标签，不能可靠代表原模型激活。

**7. 稀疏度**

设 \(z_{ik}\) 是第 \(i\) 个样本第 \(k\) 个 SAE feature 激活：

$$
K_{active}=\frac{1}{N}\sum_i \sum_k \mathbb{1}[z_{ik}>\epsilon]
$$

平均激活 feature 太多，解释会变得难以理解；太少则可能重构质量差。

**8. Feature purity**

设 feature \(k\) 激活最高的 top-m 样本中，人工或自动标签为同一概念的比例为：

$$
P_k=\max_c \frac{1}{m}\sum_{i\in Top_m(k)} \mathbb{1}[label_i=c]
$$

这个指标衡量 feature 是否接近 monosemantic。它仍然依赖标签质量，不能替代因果验证。

**9. Feature-level intervention**

如果将 feature \(k\) 的激活增加 \(\alpha\)：

$$
r' = r+\alpha d_k
$$

其中 \(d_k\) 是 SAE decoder 中对应方向。行为变化：

$$
I_k(\alpha)=\Delta(r')-\Delta(r)
$$

这可以用于 feature-level steering 或 safety monitor 的候选验证，但必须评估副作用。

**10. 把局部结果记录成机制证据向量**

这些指标回答的问题不同，不宜合并成一个总布尔量。可以把一项局部机制研究记录为：

$$
\mathcal{E}_{mech}=
(F_{rec},K_{active},P_k,S_{patch},E_{abl})
$$

其中：

1. \(F_{rec}\) 说明 SAE 是否保留了足够的激活信息。
2. \(K_{active}\) 说明平均每个样本有多少 feature 被激活。
3. \(P_k\) 说明某个 feature 的 top 样本是否集中于一个标签。
4. \(S_{patch}\) 说明指定位置的干预能恢复多少目标行为。
5. \(E_{abl}\) 说明移除组件后目标行为下降多少。

团队可以为每个信号设定研究用途，例如低 \(F_{rec}\) 触发 SAE 重训，低 \(P_k\) 触发
反例分析，低 \(S_{patch}\) 触发路径重画，高 \(E_{abl}\) 触发跨切片复验。它们是证据
和后续动作的记录，不是“解释通过”或“模型安全”的总开关。即使五项都达到研究者
预先设定的范围，也只能支持某个局部机制假设，仍需检查替代路径、分布外输入和系统
层面的红队结果。

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

在安全工程记录中应避免夸张，把方法能支持的结论和暂时不能支持的结论分开：

```text
Mechanistic interpretability 是理解和治理模型的重要工具，但目前更适合作为安全评估、调试和研究的一部分，而不是单独的安全保证。
```

### 9.4 机制信号怎样进入安全系统

机制分析进入安全工程时，通常有三种用途。第一种是离线调查：在红队失败和正常样本
上比较激活，提出可能的危险 feature 或绕过路径。第二种是训练诊断：观察安全微调前后
某类表示是否改变，并检查这种改变是否伴随帮助性退化。第三种是线上辅助监控：把稳定、
低成本的内部信号作为升级线索，与输出分类器、工具权限和人工复核结合。

第三种用途尤其容易被误解。假设一个 feature monitor 把“是否需要人工复核”标为正类，
它的漏报率和误报率可以分别写成：

$$
FNR=\frac{N(\text{危险且未触发})}{N(\text{危险})},\qquad
FPR=\frac{N(\text{正常但触发})}{N(\text{正常})}
$$

监控信号的阈值改变会同时影响两者，不能只展示 feature 激活与某个标签的相关系数。高
风险动作还需要服务端授权和可回滚执行，即使内部 monitor 漏报，系统也不应因此直接执行
不可逆操作；即使误报增加，也应通过分层升级和抽样复核评估帮助性成本。

此外，机制信号可能被模型版本变化、量化、batching、KV cache、提示模板和多模态编码
改变。上线前在一种 checkpoint 上找到的方向，不能未经复验直接绑定到另一种 artifact。
安全工程应把模型版本、层位、tokenizer、精度、推理引擎和 hook 位置写进 manifest，并
把机制信号与行为结果配对保存。这样当内部信号失效时，仍可由外部策略和权限层承担
最后一道控制责任。

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

### 10.1 一个可操作的 progress measure

在 modular addition toy task 中，输入可以表示为两个整数 \(a,b\)，目标是预测：

$$
y=(a+b)\bmod p
$$

研究者可以跟踪某个候选 Fourier 方向、特定 attention pattern 或输出读出方向在训练
过程中的变化。如果训练集准确率还没有明显提升，但候选方向的能量、相位一致性或目标
logit margin 已经连续变化，这些内部量就可能成为比最终准确率更早的 progress measure。

小白需要注意：它不是“提前看到了未来答案”，而是测量模型内部是否正在形成一种可泛化
的表示。专家则要检查该 measure 是否真的预测后续泛化，还是只追踪训练集记忆；应使用
不同随机种子、不同训练/测试划分和未参与选择的 checkpoint 验证，并报告 measure 与
最终能力之间的时间关系，而不是只展示一条漂亮曲线。

从 toy task 推向 frontier model 时，最大风险是把可观察的内部变量误当成通用预警器。
模型规模、数据、优化器和架构变化可能产生完全不同的表示坐标；因此 progress measure
更适合作为研究假设和早期诊断信号，不能替代目标能力评估与安全测试。

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

### 11.4 从发现到复验的实验记录

真实项目可以把机制研究分成发现集、验证集和回归集。发现集允许研究者浏览激活、调整
层位和提出 feature 名称；验证集冻结分析选择，只检验候选机制是否预测新的样本；回归集
则在模型、提示模板或推理引擎变化后重复测试。三者混用会让研究者不知不觉在同一批
样本上反复调参，最后把记住样本误认为解释泛化。

每条机制结论至少应附带以下字段：模型 checkpoint、tokenizer 和精度，hook 的层/位置/张量
语义，clean/corrupted 构造规则，目标指标，干预方式，负对照，样本切片，随机种子，
搜索范围，失败样本和结论作用域。对 SAE 还要记录字典版本、训练激活来源、稀疏系数、
死 feature 处理和自动命名器版本。

一个安全调试任务可以这样组织：先从一次可复现的拒答或越权边界失败开始；再比较正常
请求、风险请求、语义改写和无关主题四个切片；随后对候选 feature 做 patch、ablation
和反向干预；最后回到行为评估，检查拒答质量、正常帮助性、引用忠实度和工具权限是否
一起变化。内部激活图只是中间证据，最终修复仍要在产品 harness 中复测。

### 11.5 机制研究的成本边界

Hook 全部层和位置会产生巨大的存储与带宽成本，SAE 训练也会消耗独立计算资源。工程上
可以先用行为切片缩小范围，再用低频采样、激活缓存、候选层筛选和分层 hook 降低成本；
但任何筛选都要记录，因为“没有观察到 feature”可能只是没有采到相应位置。若机制研究
的成本高于一次安全事故的预防收益，团队可以把它定位为离线调查工具，而不是每个请求
都运行的在线组件。

## 12. 方法局限

### 12.1 扩展性问题

大模型非常大。

逐个 circuit 分析成本高。

成本不仅来自参数数量，还来自需要保存的激活数量、候选层与位置组合、干预重复次数和
人工解释时间。若模型有 \(L\) 层、序列长度为 \(T\)，对每层每个位置做一次候选 patch，
最朴素的搜索规模就是 \(O(LT)\)；若再乘上 head、feature、样本和负对照，实验量会快速
增长。实际系统通常先用行为切片、梯度或粗粒度激活统计缩小候选范围，再进行精细干预，
但这种筛选会改变发现分布，必须保留筛选规则和未筛选的抽样复核。

小模型中的完整 circuit case study 因为组件少、任务简单，容易得到清晰图景；frontier
模型则可能有并行、冗余和条件化路径。把小模型的“一个 head 负责一个功能”直接外推到
大模型，会把研究示范误写成架构定律。

### 12.2 解释不完整

解释一个 circuit 不等于解释整个模型。

机制解释通常选择一个任务、一个层段和一组输入。即使这个局部解释在目标样本上能够
重建行为，模型仍可能在其他语言、长度、模态、工具上下文或采样温度下使用不同路径。
研究者应报告解释覆盖的输入分布、输出指标和失败样本，并明确哪些行为没有被解释。对
安全用途尤其不能把“已解释拒答路径”写成“不存在未解释的危险路径”。

### 12.3 人类解释偏差

研究者可能给 feature 起一个看似合理但不完整的名字。

标签会反过来影响实验选择：一旦把 feature 命名为“欺骗”，研究者可能只寻找支持该名字
的样本，忽略它在普通规划、引用或格式控制中的作用。更好的做法是同时保存正例、反例、
边界例和自动命名的原始证据，使用盲测或多名标注者比较解释一致性，并让行为预测先于
故事化描述。自然语言名称是索引，不是机制本体。

### 12.4 因果性困难

相关性不等于因果。

需要 patching、ablation 等验证。

即便做了干预，也要问干预是否只改变了目标变量。置零一个 head 可能破坏归一化统计、
残差尺度或后续层输入分布；把一个样本的激活复制到另一个样本，可能同时携带位置、
词频和上下文信息。因果分析需要尽可能小的干预、匹配的对照和干预后状态检查，并把
“组件必要”“状态充分”“路径特异”分开陈述。

### 12.5 分布外干预

修改激活可能产生模型训练中没见过的状态。

一个 feature 方向在正常激活范围内有效，不代表把它放大十倍仍然有可解释意义。steering
强度、层位置和多个方向叠加都可能把状态推到训练分布之外，产生语法退化、过度拒答或
隐藏副作用。实验应扫描有限强度，记录激活范数、与自然样本的距离和多个帮助性/安全
指标，而不是只挑一个最漂亮的强度展示。

### 12.6 安全保证不足

即使理解了一部分机制，也不能证明没有其他危险机制。

机制分析还可能制造新的攻击面：公开 feature 方向、内部 hook 或干预接口可能让有权限
的使用者更容易改变模型行为。因此安全资料要区分公开方法、内部 artifact 和高风险
细节；对外只披露足以复核结论的聚合信息。机制解释应与 red teaming、行为评估、权限
隔离、日志审计和事故响应并行，不能替代其中任何一层。

### 12.7 工具与版本的混杂

解释结果不仅依赖模型，也依赖 tokenizer、框架、权重精度、hook 实现和推理引擎。一个
在原始 PyTorch forward 上得到的激活位置，迁移到量化 kernel、张量并行或融合算子后，
可能已经没有相同的张量语义。复制实验时应先验证“抓到的张量确实对应同一计算点”，再
比较 feature 或 patch 分数；否则看似模型机制变化，实际上可能是工具链变化。

## 13. 未来可能演化

### 13.1 从手工分析到自动化分析

未来需要更多自动化工具帮助发现 feature、命名 feature、验证 circuit。

自动化最先适合做候选生成和重复劳动：批量提取 top activating examples、聚类激活、
搜索反例、执行固定 patch 网格和生成实验报告。它不应直接把自动标签当作机制结论，
因为自动解释器同样可能受数据偏差、语言模式和研究者预设影响。更可靠的流水线会把
候选假设、反例、干预结果和人工审阅分开保存，使人可以回到原始激活检查推断是否越界。

### 13.2 从小模型到 frontier model

很多机制研究先在小模型上做。

挑战是迁移到大模型。

迁移至少有三种含义：相同功能是否在相同层位出现，相同功能是否由同一种电路实现，
以及在不同模型上发现的指标是否仍然能预测行为。前两种属于机制相似性，后一种属于
测量可迁移性，不能用一张相似热图代替。模型规模、训练数据、指令微调、工具使用和
多模态适配都可能改变内部坐标，因而需要跨 checkpoint、跨任务和跨实现做验证。

### 13.3 从解释到控制

可解释性最终可能服务于：

1. Steering。
2. Model editing。
3. Safety monitor。
4. Training-time diagnostics。
5. 反事实实验与安全调试。

控制比解释要求更高。若通过 feature steering 改变了拒答率，必须同时观察正常帮助性、
事实准确率、工具调用、不同语言和不同用户群体的变化；若通过 model editing 改变了某
个事实关联，还要检查相邻事实、时间版本和多跳推理是否被意外破坏。内部方向可以成为
控制输入，但不能绕过服务端授权、输出策略和回滚机制。

一个适合工程决策的记录方式是把“发现了什么”“改变了什么”“副作用是什么”“哪些条件
仍未测”分开。这样可解释性成果既能进入训练诊断，也能在风险较高时停留在离线研究，
不会因为研究结果看起来漂亮就被自动接入线上。

### 13.4 从局部解释到系统保证

长期目标可能是把局部机制解释、行为评估、红队和形式化约束结合，形成更强的安全证据。

这条路径可以理解为证据组合，而不是某种单一的“完全解释”：行为评估告诉我们哪里
失败，机制分析帮助定位可能原因，红队寻找未覆盖变体，权限与策略层限制现实副作用，
回归测试检查修复是否保持。任一层的证据都带有作用域和盲区，组合后仍要保留这些边界。

对安全团队而言，最实际的长期目标不是给每个参数贴上人类标签，而是让高影响行为有
更短的定位路径、更清楚的反事实证据和更可复核的修复。完整的形式化保证仍是开放研究
问题。

## 14. 案例：从拒答异常到可验证的机制假设

一家企业把语言模型接入内部知识助手。系统需要拒绝高风险请求，但也要回答普通的
合规说明、故障排查和数据字典问题。上线后，团队发现同一类正常问题在不同表达方式下
表现不一致：直接提问时经常过度拒答，换成包含相同业务含义的表格或引用片段后，回答
又变得过于宽松。仅看最终文本，团队无法判断是策略分类器、提示模板、模型内部表示，
还是工具上下文造成了差异。

这个案例不研究任何真实危险内容，而是用抽象的 `safe_request`、`high_risk_request`、
`quoted_untrusted_text` 和 `normal_boundary` 标签表示四类任务。目标是示范机制研究如何
形成证据链，而不是展示一套可以绕过安全策略的输入。

### 14.1 先把行为现象固定下来

团队先冻结模型版本、system prompt、tokenizer、解码参数、策略服务版本和工具权限，
再为四类任务各准备训练外的模板。每个任务至少记录：模型是否给出安全替代、是否错误
拒绝正常任务、是否泄露上下文、是否调用工具以及是否需要人工升级。

单次回答不够作为现象定义。团队对每个模板重复解码，并把结果聚合到任务级；同时保留
原始轨迹，以便区分模型文本、策略服务和工具服务的责任。若只选一个最显眼的拒答案例，
后续机制解释很容易变成对个别 prompt 的故事。

### 14.2 提出竞争性假设

观察到“正常问题被拒答”后，团队提出三个互相竞争的假设：

1. 输入中某些词形触发了过宽的拒答表示，但模型仍然知道如何完成正常任务。
2. 模型内部的拒答方向没有过宽，真正的问题在外部策略分类器或 system prompt。
3. 引用片段改变了上下文结构，模型在不同位置使用了不同的工具/证据路径。

这三种假设都能解释部分现象，所以不能先选中一个 feature 再寻找支持它的例子。机制
研究的第一步是为每个假设写出可观察预测：哪些 layer/position 应该变化，patch 哪一侧
应该恢复行为，哪些无关任务不应受到影响，以及外部策略服务在关闭或替换后会发生什么。

### 14.3 干预设计

研究者在 white-box 的离线副本上抓取残差流、attention 输出和 SAE feature。对同一任务
构造 clean/corrupted 对：clean 使用正常说明，corrupted 只改变表达格式或无害上下文
位置，不改变任务语义。然后依次做：

1. 在 layer/position 网格上进行 activation patching，寻找行为恢复位置。
2. 对候选 head、MLP 或 feature 做局部 ablation，测试必要性。
3. 对候选路径做 path patching，检查恢复是否依赖预期的信息流。
4. 使用无关位置、均值激活和打乱配对作为负对照。
5. 在未参与候选选择的模板和语言切片上重复。

如果外部策略服务可以独立运行，团队还要做一个服务层消融：保留模型输入和模型版本，
只替换策略判断。这样可以避免把系统层面的拒答误归因于模型内部。所有干预都在隔离副本
中执行，不能把内部 hook 或写入接口暴露给生产请求。

### 14.4 一组 toy 结果应该怎样读

假设实验观察到：某个候选拒答 feature 在 top 样本中的标签纯度很高；把它从风险请求
patch 到正常请求后，正常回答概率下降；但把它从正常请求 patch 到风险请求时，风险
边界并没有完全恢复；对无关位置做同样 patch 也有小幅变化。

初学者可能会说“找到了拒答 feature”。更准确的说法是：该 feature 与拒答行为有关，
在一个方向上有局部干预证据，但仍可能携带位置或上下文信息，且不是拒答机制的充分
解释。无关位置也产生变化，说明需要继续检查残差尺度、层归一化和 patch 的分布外效应。

专家报告还会给出不同切片的结果：正常问题的误拒率是否下降，真正高风险任务的漏拒率
是否上升，安全替代质量是否变化，工具调用是否被错误抑制，以及 feature monitor 的
FNR/FPR 是否在语言和长度切片上稳定。机制解释只有回到这些行为指标，才具有工程价值。

### 14.5 修复和回归

假设根因最终被定位为外部策略阈值过宽，而模型内部 feature 只是对输入风险的正常响应。
团队应修策略服务和任务切片，不应直接把 feature 从模型里抹掉。相反，如果模型在策略
服务关闭时仍出现同样的错误拒答，且 patch/ablation 在 holdout 上复现，才可以把模型训练
或表示层作为修复候选。

修复后至少复测五类任务：原始失败、语义改写、正常边界、真正高风险抽象任务和无关任务。
还要比较机制信号变化与外部行为变化是否一致。若正常帮助性改善但高风险漏拒增加，
发布范围应保持受限并继续调查；若行为恢复而机制信号不再稳定，则不能把旧 feature
monitor 当作线上依据。

这个案例体现了一个重要原则：可解释性不是给模型贴上“安全”或“不安全”的标签，而是
帮助团队在多个可能根因之间做可复核的区分，并把修复送回行为、权限和回归系统。

## 15. 资料与证据边界

机制可解释性资料的证据层次差异很大。原始论文和研究机构的技术文章可以支持某种
方法、toy model 或特定 checkpoint 上的实验结果；教程和开源工具可以支持复现实验路径；
它们都不能直接证明任意 frontier model 的内部机制，更不能替代生产安全评估。

### 15.1 机制与 circuits

- [Circuits](https://distill.pub/2020/circuits/)：Distill 的视觉模型 circuits 系列，说明从神经元、特征到局部算法的逆向工程思路；结论作用域是所分析的模型和任务。
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)：给出 Transformer 电路的数学分析框架和 QK/OV 视角；它是分析工具，不是对所有 Transformer 实现的完整证明。
- [Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small](https://arxiv.org/abs/2211.00593)：展示在 GPT-2 Small 上分析 IOI circuit 的案例；规模、任务和组件结论不能直接移植到其他模型。

### 15.2 Superposition 与 SAE

- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)：研究 feature 数量超过表示维度时的 superposition toy model；它支持机制假设和可视化直觉，不是大模型 feature 的直接测量。
- [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html)：讨论用 dictionary learning 分解语言模型激活的研究结果；应同时关注训练层、激活数据和解释指标。
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)：展示更大规模 feature 提取和解释工作；“可提取 feature”仍不等于完整 circuit 或因果安全证明。

### 15.3 因果干预、事实关联与工具

- [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)：提出 ROME 并使用 causal tracing 分析事实关联；编辑效果、泛化和副作用必须在具体模型上重新测量。
- [Progress Measures for Grokking via Mechanistic Interpretability](https://arxiv.org/abs/2301.05217)：在 modular addition 等 toy setting 中研究内部进展指标；不能直接当作 frontier 能力预警器。
- [TransformerLens](https://transformerlensorg.github.io/TransformerLens/)：提供 Transformer 内部激活分析和干预的开源工具文档；工具接口和模型支持范围需要按当前版本核对。

引用这些资料时，应记录论文版本或网页访问时间、模型 checkpoint、层/位置、激活采样、
干预强度、负对照、指标分母和 holdout 结果。机构博客或产品页面可以说明研究方向和
公开声明，但涉及“发现了某个机制”“提高了安全性”的句子，仍需回到可复现实验和作用域。

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

## 18. 最小可运行 Mechanistic Interpretability 审计 demo

下面的 demo 不依赖真实模型，只用 toy logit delta、toy 激活和 toy SAE feature 来演示本章核心审计口径。真实项目中要把这些字段替换成模型实际 forward hook、patching 实验和 SAE 训练结果。

```python
from collections import Counter


def ratio(num, den):
    return round(num / den, 3) if den else 0.0


def squared_error(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def mean_vector(rows):
    dims = len(rows[0])
    return [sum(row[j] for row in rows) / len(rows) for j in range(dims)]


def recovery_score(clean_delta, corrupt_delta, patched_delta):
    return ratio(patched_delta - corrupt_delta, clean_delta - corrupt_delta)


clean_delta = 2.0
corrupt_delta = -1.0
patched_deltas = {
    "layer1_pos3": 0.2,
    "layer2_pos1": -0.4,
    "layer2_pos3": 1.4,
    "layer3_pos3": 1.1,
}

patch_scores = {
    name: recovery_score(clean_delta, corrupt_delta, patched)
    for name, patched in patched_deltas.items()
}
best_patch = max(patch_scores, key=patch_scores.get)

ablated_deltas = {
    "previous_token_head": 1.5,
    "name_mover_head": 1.2,
    "refusal_feature": 0.4,
}
ablation_effects = {
    name: round(clean_delta - delta, 3)
    for name, delta in ablated_deltas.items()
}

path_patched_deltas = {
    "refusal_feature_to_output": 1.25,
    "name_head_to_output": 0.75,
}
path_scores = {
    name: recovery_score(clean_delta, corrupt_delta, patched)
    for name, patched in path_patched_deltas.items()
}

labels = ["code", "code", "refusal", "refusal", "name", "name"]
residuals = [
    [2.0, 0.1, 0.0],
    [1.8, 0.2, 0.1],
    [0.1, 2.1, 0.2],
    [0.0, 1.7, 0.2],
    [0.2, 0.1, 2.0],
    [0.1, 0.0, 1.8],
]
reconstructions = [
    [1.9, 0.2, 0.0],
    [1.7, 0.2, 0.1],
    [0.1, 2.0, 0.2],
    [0.1, 1.6, 0.2],
    [0.2, 0.1, 1.9],
    [0.1, 0.1, 1.7],
]
feature_activations = [
    [2.5, 0.0, 0.1],
    [2.0, 0.1, 0.0],
    [0.1, 2.4, 0.2],
    [0.0, 2.2, 0.1],
    [0.1, 0.2, 2.3],
    [0.0, 0.1, 2.1],
]
feature_names = ["code_feature", "refusal_feature", "name_feature"]

mean_residual = mean_vector(residuals)
sse = sum(squared_error(row, rec) for row, rec in zip(residuals, reconstructions))
sst = sum(squared_error(row, mean_residual) for row in residuals)
reconstruction_fidelity = ratio(sst - sse, sst)

active_threshold = 0.5
active_counts = [
    sum(value > active_threshold for value in row)
    for row in feature_activations
]
avg_active_features = round(sum(active_counts) / len(active_counts), 3)

purity = {}
top_k = 2
for feature_idx, feature_name in enumerate(feature_names):
    ranked = sorted(
        range(len(feature_activations)),
        key=lambda i: feature_activations[i][feature_idx],
        reverse=True,
    )
    top_labels = [labels[i] for i in ranked[:top_k]]
    most_common = Counter(top_labels).most_common(1)[0][1]
    purity[feature_name] = round(most_common / top_k, 3)

feature_interventions = {
    "boost_refusal_feature": {"before": 0.30, "after": 1.10},
    "boost_name_feature": {"before": 0.30, "after": 0.45},
}
intervention_effects = {
    name: round(values["after"] - values["before"], 3)
    for name, values in feature_interventions.items()
}

metrics = {
    "best_patch": best_patch,
    "best_patch_score": patch_scores[best_patch],
    "max_ablation_effect": max(ablation_effects.values()),
    "best_path_score": max(path_scores.values()),
    "reconstruction_fidelity": reconstruction_fidelity,
    "avg_active_features": avg_active_features,
    "min_feature_purity": min(purity.values()),
    "refusal_intervention_effect": intervention_effects["boost_refusal_feature"],
}

thresholds = {
    "patch_causal": {"operator": ">=", "value": 0.75},
    "ablation_effect": {"operator": ">=", "value": 1.0},
    "path_patch": {"operator": ">=", "value": 0.7},
    "reconstruction": {"operator": ">=", "value": 0.95},
    "sparsity": {"operator": "<=", "value": 1.5},
    "purity": {"operator": ">=", "value": 0.9},
    "feature_intervention": {"operator": ">=", "value": 0.5},
}

signals = {
    "patch_causal": metrics["best_patch_score"],
    "ablation_effect": metrics["max_ablation_effect"],
    "path_patch": metrics["best_path_score"],
    "reconstruction": metrics["reconstruction_fidelity"],
    "sparsity": metrics["avg_active_features"],
    "purity": metrics["min_feature_purity"],
    "feature_intervention": metrics["refusal_intervention_effect"],
}


def meets_threshold(signal, threshold):
    if threshold["operator"] == ">=":
        return signal >= threshold["value"]
    if threshold["operator"] == "<=":
        return signal <= threshold["value"]
    raise ValueError(f"unsupported operator: {threshold['operator']}")


evidence_status = {
    name: meets_threshold(signals[name], threshold)
    for name, threshold in thresholds.items()
}

actions = {
    "patch_causal": "replicate_patch_on_holdout",
    "ablation_effect": "test_component_redundancy",
    "path_patch": "check_upstream_and_downstream_paths",
    "reconstruction": "review_sae_dictionary_fidelity",
    "sparsity": "inspect_feature_activation_distribution",
    "purity": "search_feature_counterexamples",
    "feature_intervention": "run_behavioral_side_effect_regression",
}

decision = {
    "scope": "local_mechanism_hypothesis",
    "status": "collect_holdout_and_behavioral_evidence",
    "evidence_status": evidence_status,
    "next_actions": list(actions.values()),
}

print("patch_scores=", patch_scores)
print("ablation_effects=", ablation_effects)
print("path_scores=", path_scores)
print("purity=", purity)
print("metrics=", metrics)
print("thresholds=", thresholds)
print("signals=", signals)
print("evidence_status=", evidence_status)
print("actions=", actions)
print("decision=", decision)
```

预期输出：

```text
patch_scores= {'layer1_pos3': 0.4, 'layer2_pos1': 0.2, 'layer2_pos3': 0.8, 'layer3_pos3': 0.7}
ablation_effects= {'previous_token_head': 0.5, 'name_mover_head': 0.8, 'refusal_feature': 1.6}
path_scores= {'refusal_feature_to_output': 0.75, 'name_head_to_output': 0.583}
purity= {'code_feature': 1.0, 'refusal_feature': 1.0, 'name_feature': 1.0}
metrics= {'best_patch': 'layer2_pos3', 'best_patch_score': 0.8, 'max_ablation_effect': 1.6, 'best_path_score': 0.75, 'reconstruction_fidelity': 0.993, 'avg_active_features': 1.0, 'min_feature_purity': 1.0, 'refusal_intervention_effect': 0.8}
thresholds= {'patch_causal': {'operator': '>=', 'value': 0.75}, 'ablation_effect': {'operator': '>=', 'value': 1.0}, 'path_patch': {'operator': '>=', 'value': 0.7}, 'reconstruction': {'operator': '>=', 'value': 0.95}, 'sparsity': {'operator': '<=', 'value': 1.5}, 'purity': {'operator': '>=', 'value': 0.9}, 'feature_intervention': {'operator': '>=', 'value': 0.5}}
signals= {'patch_causal': 0.8, 'ablation_effect': 1.6, 'path_patch': 0.75, 'reconstruction': 0.993, 'sparsity': 1.0, 'purity': 1.0, 'feature_intervention': 0.8}
evidence_status= {'patch_causal': True, 'ablation_effect': True, 'path_patch': True, 'reconstruction': True, 'sparsity': True, 'purity': True, 'feature_intervention': True}
actions= {'patch_causal': 'replicate_patch_on_holdout', 'ablation_effect': 'test_component_redundancy', 'path_patch': 'check_upstream_and_downstream_paths', 'reconstruction': 'review_sae_dictionary_fidelity', 'sparsity': 'inspect_feature_activation_distribution', 'purity': 'search_feature_counterexamples', 'feature_intervention': 'run_behavioral_side_effect_regression'}
decision= {'scope': 'local_mechanism_hypothesis', 'status': 'collect_holdout_and_behavioral_evidence', 'evidence_status': {'patch_causal': True, 'ablation_effect': True, 'path_patch': True, 'reconstruction': True, 'sparsity': True, 'purity': True, 'feature_intervention': True}, 'next_actions': ['replicate_patch_on_holdout', 'test_component_redundancy', 'check_upstream_and_downstream_paths', 'review_sae_dictionary_fidelity', 'inspect_feature_activation_distribution', 'search_feature_counterexamples', 'run_behavioral_side_effect_regression']}
```

这个 demo 的重点不是数值本身，而是机制可解释性报告应该同时包含：

1. 行为指标：clean / corrupted 的 logit delta。
2. 因果证据：patching、ablation、path patching。
3. 表示分解质量：SAE reconstruction fidelity 和 sparsity。
4. 可解释性质量：feature purity。
5. 安全使用边界：feature intervention 有效果，但仍然只说明局部机制有证据，不能证明模型整体安全。
6. 决策范围：当前结果只支持在 holdout 和行为回归中继续验证，不能直接改变生产策略或工具权限。

## 19. 本章总结

Interpretability 是广义可解释性，Mechanistic Interpretability 更强调逆向工程模型内部机制。

传统归因、attention 可视化和 probing 有价值，但容易停留在相关性层面。

Activation patching、ablation 和 causal tracing 更关注因果作用。

Circuits 试图解释多个组件如何共同实现功能。

Polysemanticity 和 superposition 解释了为什么单个 neuron 往往难以解释。

Sparse Autoencoder 试图从激活空间中分解出更稀疏、更可解释的 feature，是当前 LLM 机制可解释性的重要方向。

Mechanistic interpretability 对 safety 有潜在价值，可以帮助理解幻觉、拒答、jailbreak、steering 和危险能力，但目前仍不能单独提供安全保证。

理解这章可以抓住一条方法谱系：从输入归因，到表示分析，到因果干预，再到机制逆向和安全治理。
