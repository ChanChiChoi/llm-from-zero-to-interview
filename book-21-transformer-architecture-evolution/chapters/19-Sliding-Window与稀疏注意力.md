# 第十九章：Sliding Window Attention、Global Attention 与稀疏注意力

## 19.1 从全连接到稀疏连接图

在标准因果 attention 中，位置 $i$ 的 query 可以读取它之前的 key/value。把每个 token 作为一个节点，把一次允许的 query-key 交互作为一条有方向的边，就得到一张逐层的信息连接图。每个位置都能直接访问完整历史时，长度为 $n$ 的序列包含约 $n^2$ 个可见位置对；稀疏 attention 则通过规则或输入相关的 pattern，只保留其中一部分。

因此，稀疏 attention 改变的是“这一层允许哪些位置直接交换信息”。它不是一种单独算法：局部窗口、扩张窗口、全局 token、随机边和内容选择式 pattern 会形成不同的连接图。某个 token 不在当前位置的直接可见集合中，不代表它的信息绝无机会进入后续层；反过来，两点间存在多层路径，也不代表模型一定保留并使用了路径上的细节。

稀疏连接数也不自动等于线性复杂度。固定宽度的局部窗口会随 $n$ 近似线性增长，但不同结构得到不同的渐近阶。例如 Sparse Transformer 的论文用特定结构化 factorization 将 attention 复杂度降为 $O(n\sqrt{n})$；稀疏一词本身不能告诉我们复杂度，更不能说明 GPU 上实际更快。[Sparse Transformers 原论文](https://arxiv.org/abs/1904.10509)

还要把稀疏 pattern 与精确 attention 的高效执行分开。FlashAttention 分块重排标准 attention 的计算，仍对 mask 允许的完整位置集合求精确 attention；稀疏 pattern 会排除原本可见的连接。前者的算法和 IO 机制见[第二十章](20-FlashAttention与PagedAttention.md)，attention pair 的总体成本见[第十章](10-o-n2-attention的计算与显存瓶颈.md)。

## 19.2 Causal Sliding Window：直接读取范围与跨层传播

causal sliding window 让每个位置读取最近的一段历史。设序列位置从 $0$ 编号，窗口宽度 $W$ 表示单层可见 token 的总数，**包含当前位置**。位置 $i$ 直接可见的 key 位置是：

```math
\mathcal{V}_i=\{j\mid \max(0,i-W+1)\le j\le i\}.
```

当 $W=3$ 时，各位置的窗口如下：

| 查询位置 $i$ | 本层可直接读取的位置 |
| --- | --- |
| $0$ | $0$ |
| $1$ | $0,1$ |
| $2$ | $0,1,2$ |
| $3$ | $1,2,3$ |
| $4$ | $2,3,4$ |
| $5$ | $3,4,5$ |

所以位置 $5$ 这一层不能直接读取位置 $0$ 至 $2$。有的实现把窗口参数写成“可见 token 总数”，有的把它写成“当前 token 之外的历史数”或半径；参数名字相似，边界可能相差一个位置。读配置或比较模型时，需要确认窗口的实际包含规则。

每个 query 最多读取约 $W$ 个 key，因此注意力位置对数量约为 $nW$。当 $W$ 固定时，这一部分随序列长度近似线性增长；若窗口随上下文长度一起扩大，成本也会随之增加。这个计数只描述 attention 的连接数量，不包含投影、其他网络层和实现开销。

“单层看不到”与“所有层都无法传递”是两回事。以 $W=3$ 为例，第一层中位置 $2$ 可以读到位置 $0$；第二层中位置 $4$ 可以读到位置 $2$ 的上一层表示。于是位置 $0$ 的信息存在一条通向位置 $4$ 的两层路径。窗口以外的信息可以沿这样的路径逐层传播，但中间表示可能混合或丢失原始细节，路径存在并不保证精确回忆。长距离依赖的任务结果由[第十三章](13-长距离依赖与lost-in-the-middle.md)讨论。

[Mistral 7B v0.1 论文](https://arxiv.org/abs/2310.06825)报告使用 sliding-window attention，是 causal LLM 采用局部窗口的一个具体案例。它说明模型层内的直接窗口可以小于跨层信息传播的理论范围；这不能单独证明任意长序列都能被模型有效使用，窗口和模型配置也应绑定具体版本。架构配置比较见[第十七章](17-LLaMA-GPT-Mistral架构对比.md)。

## 19.3 Dilated 与结构化稀疏：扩大跨度会改变哪些连接

扩张窗口（dilated window）在候选位置之间留出间隔。例如某个 query 可以直接读取当前位置以及每隔两个位置的历史 token。与连续扩大窗口相比，它能在相近的单层连接预算下触及更远位置；代价是中间位置没有直接连接。若任务依赖相邻语境或某个恰好落在间隔中的证据，只扩大跨度并没有解决它。

Longformer 论文把局部窗口、dilated window 和 task-motivated global attention 作为不同 pattern 展示；其自回归实验也讨论了如何在不同层和 head 中安排 dilation。[Longformer 原论文](https://arxiv.org/abs/2004.05150) 这类设计说明，多层和多头可以组合不同范围的连接，但具体受益仍绑定训练模型和任务。

结构化稀疏还可以用不同分解构造，而不只是对角线附近的窗口。Sparse Transformer 论文的 $O(n\sqrt{n})$ 结果就是一个不同于固定窗口的稀疏方案。比较方法时，应先问清每个 query 实际有哪些 key，再计算该 pattern 的位置对数量；只看“稀疏率”或论文标题，不足以判断远程可达性和复杂度。

块级 pattern 还要把 token 级连接映射到硬件计算块，块内可能会计算逻辑上不需要的 pair。其 mask 表示、padding、block size 与 kernel 路径属于[第二十六章](26-稀疏与块稀疏Attention.md)；attention kernel 的 IO 和 softmax 机制见[第二十章](20-FlashAttention与PagedAttention.md)。

## 19.4 Global Attention：为少量位置增加全序列路径

Longformer 用少量 task-global 位置补充局部窗口。论文中的分类任务可以让 `[CLS]` 汇聚文档表示；问答任务则可将问题 token 标为 global，让它们与文档其他位置交互。在原论文的双向设置中，global 位置可以读取全序列，普通位置也可以读取 global 位置。因此全局位置既能汇总分散的信息，也能把汇总表示提供给其他 token。

如果有 $g$ 个 global 位置，局部窗口宽度为 $W$，这类连接的 pair 数量约为 $O(nW+ng)$。当 $g$ 固定且远小于 $n$ 时，它仍随序列长度近似线性增长。这个结论依赖连接规则和固定 global 数的假设；如果全局位置数随序列增大，数量级也会改变。

Longformer 的 global 连接是面向双向长文档任务的设计，不能原样搬到因果 decoder。decoder 的位置不能读取未来 token；若某个位置有特殊全局规则，仍须明确它何时可读、谁能读它，以及该规则怎样与 causal mask 组合。第八章介绍的 mask 可见性约定在这里仍然有效。

Global token 能缩短连接图中的传播路径，但不是一条无损总线。它输出的是经过模型层变换后的表示，具体能携带多少与模型宽度、层、训练目标和输入内容有关。所有远处位置都能连到一个 global token，不等于每个原始事实都能从中无损恢复。

## 19.5 BigBird：把局部、随机与全局连接组合起来

BigBird 将三类连接合在同一 attention pattern 中：每个 token 读取附近位置，读取一组随机选中的位置，少量 global token 再连接全序列。local 边保留连续局部上下文；random 边提供跨片段连接；global 边提供指定的汇聚和广播路径。不同实现的具体 mask 细节可能不同，讨论时应以对应论文或模型配置为准。

用 $W$ 表示每个位置的局部连接数，用 $R$ 表示随机连接数，用 $G$ 表示 global token 数，则这类 pattern 的连接量可概括为 $O(nW+nR+nG)$。当 $W$、$R$ 和 $G$ 固定时，attention pair 数随 $n$ 线性增长。NeurIPS 2020 论文进一步在其理论设定下论证 universal approximation 和 Turing completeness，并给出长序列任务实验。[BigBird 原论文](https://papers.neurips.cc/paper_files/paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html)

表达能力的理论结论回答的是模型在特定条件下能够表示什么，不保证一个给定深度、宽度和训练方案的模型会在某项任务上达到 dense attention 的质量。随机边能缩短某些图路径，但也不代表该稀疏布局在目标 GPU 上更快。需要分别讨论 pattern 的可见关系、训练后模型的任务表现和 kernel 的实际执行。

## 19.6 从固定 mask 到内容选择：动态稀疏与适用边界

局部、扩张、global 和 BigBird 的随机边都可以由 pattern 预先指定。另一类方法会根据当前 query 选择历史位置：DeepSeek Sparse Attention（DSA）使用 query-conditioned indexer 给历史位置评分，再将选出的 top-$k$ KV entries 交给主 attention 读取。indexer 的分数负责挑候选，主 attention 才在候选上计算 attention 输出；两者不能混成一个“softmax 只看 top-$k$”的说法。[DeepSeek-V3.2 技术报告](https://arxiv.org/abs/2512.02556)；算法讲解与 indexer 扫描成本见[Stanford MS&E 319 Lecture 5](https://web.stanford.edu/class/msande319/lectures/lecture-5-notes.pdf)。

内容选择让不同 query 可以读取不同位置，但筛选器也要为历史打分，候选选择、KV 状态和主 attention 读取各有成本。主 attention 的候选数量不能代表整套方法的总复杂度，也不能证明关键证据总能进入候选集。DSA 的模型字段和实现细节由[第八十九章](89-glm-5-dsa-slime与agentic-engineering.md)等模型案例承接；动态选择与块级执行的通用问题由第二十六章展开。

读到一个稀疏 attention 声明时，可以依次分清三件事：mask 或选择器允许谁读取谁；多个 attention 层是否提供从证据到 query 的路径；目标任务是否实际保留并使用了证据。最后一项需要任务级评测，不能从连接图推断。[StreamingLLM](https://arxiv.org/abs/2309.17453) 还提示，流式生成中保留初始 attention-sink KV 与近期窗口是一种推理时 cache 策略，不等于训练时采用滑动窗口，也不能找回已从缓存淘汰的中间内容；相关状态管理见[第十一章](11-kv-cache长上下文推理和显存增长问题.md)与[第十五章](15-Transformer流式在线与低延迟局限.md)。

| Pattern | 每个 query 的直接连接 | 主要作用 | 不能据此推出 |
| --- | --- | --- | --- |
| Causal sliding window | 最近 $W$ 个位置 | 限制单层读取范围；固定 $W$ 时约有 $O(nW)$ 对 | 窗口外证据不可传播，或任意窗口长度都具备有效能力 |
| Dilated window | 跨间隔选取位置 | 用相近连接数触及更远距离 | 被跳过的邻近位置仍可直接读取 |
| Longformer-style global | 局部窗口加少量双向 global 位置 | 为指定任务位置提供全序列路径 | 任意 causal decoder 都能使用相同双向规则 |
| BigBird-style | local + random + global | 组合局部关系、跨区边和全局路径 | 理论表达性保证给定模型的任务质量或 GPU 加速 |
| Query-conditioned selection | indexer 为当前 query 选候选，再由主 attention 读取 | 让不同 query 选择不同历史位置 | indexer、候选 cache、gather 和主 attention 的总成本等于候选数 |

attention pattern 改变模型内部可见关系；RAG 则在模型外改变送入上下文的候选文档。长上下文位置映射属于[第十八章](18-RoPE与长上下文位置外推.md)，任务级远程证据使用属于[第十三章](13-长距离依赖与lost-in-the-middle.md)，kernel 与块稀疏执行分别属于[第二十章](20-FlashAttention与PagedAttention.md)和[第二十六章](26-稀疏与块稀疏Attention.md)。这些边界帮助读者把“连接较少”理解成一种具体架构选择，而不是有效上下文或端到端速度的保证。
