# 第三章：Tokenizer 与数据格式：从字符到训练协议

## 3.0 模型看到的不是句子，而是一串带协议的整数

人看到的是中文句子、英文段落、Python 函数和一组对话消息；模型在 embedding 层之前看到的是 token id。Tokenizer 决定字符串如何映射成整数，数据格式决定这些整数哪些是内容、哪些是角色边界、哪些是工具事件，label mask 决定哪些位置真正参与训练。

初学者可以先记住这条链路：

~~~text
字符串或消息
  -> 规范化与模板序列化
  -> tokenizer 编码
  -> token ids、attention mask、labels
  -> embedding
  -> Transformer
  -> logits over vocabulary
  -> token ids
  -> detokenizer
  -> 字符串或结构化事件
~~~

只要其中一个环节的协议不一致，模型就可能出现看似奇怪的行为：同一个词被映射成不同 id，`<eos>` 没有被当作停止条件，assistant 标签错位，工具 JSON 的括号被切断，图片占位符和视觉 embedding 数量不匹配。

专家还要把 tokenizer 放进训练账本：一段数据产生多少有效 token？中文、代码和多语言之间是否公平？词表增大节省的序列长度是否抵消了 embedding 和输出层成本？训练时的 chat template、推理时的 template 和评估脚本是否来自同一版本？

本章先建立编码链路，再解释 BPE、Unigram LM 和 byte-level 方法，随后讨论词表大小、压缩率、special token、chat template、label mask、预训练 packing、SFT、偏好、工具调用、多模态占位符和词表扩展，最后用一个无依赖示例把这些协议跑通。

## 3.1 Tokenizer 的输入、输出和不变量

### 3.1.1 编码和解码

给定文本 `x`、tokenizer `T` 和词表大小 `V`，编码结果是：

~~~math
z=T(x)=(z_1,z_2,\ldots,z_L),
\qquad
z_t\in\{0,1,\ldots,V-1\}
~~~

`L` 是这段文本的 token 数。解码器 `D` 将 token id 转回字符串：

~~~math
\hat{x}=D(z)
~~~

对可逆 tokenizer，理想情况下 `D(T(x))=x`，或至少在规定的规范化等价关系下相同。实际系统可能会规范化空白、Unicode、特殊字符或字节表示，因此复现时要区分原文相等和规范化后相等。

### 3.1.2 Token id 不是 token 字符串

字符串 `<assistant>`、字节片段和整数 id 是不同层次的对象。模型参数只知道整数 id 对应的 embedding 行；这个对应关系由 tokenizer 文件和 special token 配置定义。两个模型都叫 `<eos>`，其字符串、id、出现位置和停止语义也可能不同。

因此，查看模型时要同时记录：

| 项目 | 要确认什么 |
| --- | --- |
| 词表 | token 字符串、id、大小和版本 |
| 规范化 | Unicode、空格、大小写和字节规则 |
| special token | 字符串、id、是否跳过解码和停止语义 |
| 编码 | truncation、padding、offset 和返回字段 |
| 解码 | 是否跳过 special token、空格如何恢复 |
| 模板 | 消息如何序列化，生成从哪里开始 |

这份记录最好成为模型版本 manifest，而不是散落在预处理代码里的默认值。至少固定 tokenizer 文件及其 hash、词表和 added-token 映射、normalizer、pre-tokenizer、BPE merge ranks 或 Unigram 模型、special-token id、chat template、processor 配置、padding / truncation 规则，以及训练样本的 label mask 和 packing 规则。只记一个仓库名或模型名不够：同一个模型目录里的 tokenizer 文件也可能被更新，服务端还可能自行覆盖 EOS 或模板。

### 3.1.3 Tokenizer 的不变量

一个训练 tokenizer 至少要满足：

1. 训练和推理使用相同的词表与规则。
2. 编码结果 id 都在 embedding 和输出层支持的范围内。
3. special token 的角色在数据、模型和服务端一致。
4. 截断和 padding 不破坏目标字段与文档边界。
5. 解码后的文本和结构化事件符合预期。

这些是不变量，不是某个框架的固定实现。换 tokenizer 往往不是换预处理器，而是改变模型输入空间；已经训练完成的模型不能随意接入另一个词表。

### 3.1.4 规范化决定“可逆”究竟指什么

完整编码链通常不是直接计算 `T(x)`，而是先经过规范化函数 `N`：

~~~math
z=T(N(x))
~~~

例如，Unicode 中看起来相同的 `é` 可以由一个预组字符表示，也可以由 `e` 加组合重音符表示；全角拉丁字母 `Ａ` 也可能被 NFKC 变为 ASCII `A`。如果 `N` 折叠了这些形式，解码器最多恢复规范化后的文本 `N(x)`，不能再恢复原始字节串 `x`。所以“可逆”需要说明是对原始字节、Unicode 字符序列，还是规范化等价类可逆。

规范化能减少等价写法带来的词表碎片，却也可能改动代码标识符、数学符号、空格或用户原文。训练前应固定并保存 normalizer 版本；需要精确引用、代码执行或文本对齐的数据，要同时保留原始文本和参与 tokenization 的规范化文本，并让 offset、标签和 span 对齐同一版本。SentencePiece 的论文描述了空格保留、NFKC 规范化和可配置规则；它是一个具体工具设计，不能据此假定所有 tokenizer 都做相同规范化。

规范化是否有利还取决于目标任务。van der Goot 在 2026 年对 2,307 种语言、每种 1,000 条样本做语言识别实验，以 1—5 字符 n-gram Naive Bayes 比较 NFC 和原文；十个数据划分中不规范化的准确率都更高，均值为 91.75% 对 91.70%。差距很小，但说明组合字符可能携带语言或输入来源信号。这个实验不是 LLM 预训练或通用文本质量测试，不能推出“不要规范化”；应把它理解为一项反例，提醒你依据任务比较 raw / normalized 两种输入，并明确可逆性和对齐成本。

## 3.2 为什么不用字符或整词直接建模

### 3.2.1 字符级方法的序列成本

字符级词表小、覆盖范围广，但一段文本会变成很长的序列。序列变长后，模型需要更多位置预测相同语义，attention、激活和 KV cache 成本都会增加。

### 3.2.2 整词方法的数据稀疏

整词级词表可以让常见词变成一个 token，但新词、拼写变化、代码标识符和低资源语言会产生大量未登录词。把所有词都塞进词表又会造成词表极大、低频项难以学习和输出层成本升高。

### 3.2.3 子词和字节是折中

子词方法让常见片段保持较短表示，罕见词可以退化为更小片段；byte-level 方法进一步提供任意输入的覆盖。这个折中把“词表成本”和“序列长度”放在同一个优化问题里。

对数据域来说，好的 tokenizer 不一定产生人类直觉上的词根边界。它更重要的性质是：覆盖目标字符和字节，控制 token 数，保持可逆，支持特殊格式，并且在代表性语料上给出稳定的训练和推理成本。

## 3.3 BPE：从高频相邻片段开始合并

### 3.3.1 核心机制

BPE 从较小的初始单元开始，统计相邻片段的频率，每轮合并最常见的一对。第 `r` 轮可以写成：

~~~math
(a^*,b^*)
=
\arg\max_{(a,b)}C_r(a,b)
~~~

`C_r(a,b)` 是当前分词结果中相邻片段 `(a,b)` 的计数。合并后：

~~~math
u^*=a^*b^*,
\qquad
V_{r+1}=V_r\cup\{u^*\}
~~~

编码时应用学习到的 merge rule，通常优先使用更高优先级的合并。真实实现还会处理预分词、字节编码、空格标记、特殊 token 和正则规则。

把计数过程摊开看，能发现 BPE 学到的是“当前表示中经常相邻的片段”，而不是词根或语义。假设预分词后得到下面的小语料，重复项按出现次数计数：

| 预分词单元 | 出现次数 | 初始相邻对 |
| --- | ---: | --- |
| `aba` | 2 | `a-b`, `b-a` 各出现 2 次 |
| `abac` | 1 | `a-b`, `b-a`, `a-c` 各出现 1 次 |
| `ac` | 1 | `a-c` 出现 1 次 |

第一轮的总计数是 `a-b: 3`、`b-a: 3`、`a-c: 2`。假设并列时选择 `a-b`，合并后语料变为 `ab-a` 两次、`ab-a-c` 一次、`a-c` 一次。重新统计，`ab-a` 出现 3 次，`a-c` 出现 2 次，于是下一轮合并 `ab a` 得到 `aba`。继续合并会让常见串变成更大的 token。实际实现还要定义并列频率的 tie-break、词内边界、空格预分词和 merge-rank 的应用顺序；这些细节一变，最终 token id 序列就可能变。

这个手算例子也揭示一个代价：某片段频繁共现，不代表它在新领域依然有用。例如把某个代码库的标识符学成大 token，可能压缩该仓库，却无法泛化到新项目；词表空间就此被训练语料的频率分布占用。BPE 是有确定学习规则的压缩启发式，不是“找出最正确的语言学切分”。

### 3.3.2 BPE 的优点和边界

优点是机制直观、工程实现成熟、未见词可以退化为更小片段。边界是合并规则受训练语料分布影响：中文、代码、数学、低资源语言和混合文本可能出现很不同的切分效率；高频不等于语义上合理；一个 token 也不等于一个词。

### 3.3.3 训练语料决定 merge

如果 tokenizer 主要用英文网页训练，代码标识符、中文短语和数学符号可能被切得更碎。把词表直接用于另一种语言或领域，模型不仅序列变长，还会在训练预算上给予该语言更多预测位置，却未必给予更多有效语义容量。

因此 tokenizer 训练语料应代表目标语言、代码、数字、公式、对话和特殊格式，并保存采样版本。只报告词表大小，不足以判断 tokenizer 是否适合目标模型。

## 3.4 SentencePiece 与 Unigram LM

### 3.4.1 SentencePiece 是工具体系

SentencePiece 可以直接从原始文本训练子词模型，不强依赖语言特定的空格预分词，适合多语言和没有明显词边界的语言。它不是一种唯一的算法，常见路线包括 BPE 和 Unigram LM。

### 3.4.2 Unigram LM 的概率视角

对字符串 `x`，设一种可行切分为 `s=(u_1,\ldots,u_m)`，片段概率的乘积可以写成：

~~~math
p(s)=\prod_{j=1}^{m}p(u_j)
~~~

同一字符串可能有多种切分，完整概率要对可行集合求和：

~~~math
p(x)=\sum_{s\in\mathcal{S}(x)}p(s)
~~~

训练过程通常从候选片段集合开始，删除对语料似然贡献较小的片段，逐步得到目标词表。编码时可以使用最大概率路径；某些实现还支持 subword regularization，在训练中对切分进行采样。

### 3.4.3 与 BPE 的差异

BPE 是基于合并历史构造词表，Unigram LM 是保留一组片段并用概率选择切分。两者都能产生子词，但学习过程、切分稳定性和采样能力不同。工程选型应比较目标语料的 token/byte、训练速度、可逆性、多语言切分和下游任务，而不是因为工具名字熟悉就默认某种算法更好。

### 3.4.4 子词采样：把一种切分变成训练期扰动

Unigram 模型可为同一个字符串保留多种合法切分，并按其概率采样；SentencePiece 也实现了 BPE-dropout。若某次训练只用最优切分，字符串每次都以同一串 id 出现；采样后，同一拼写会以不同片段组合进入模型，作用类似输入表示层的数据增强。它可能让低资源或分布外文本更稳，也增加了训练随机性，并不保证任何 LLM 的质量都会上升。

Kudo 的原始实验主要是神经机器翻译：多种 IWSLT 低资源语言对上报告约 1—2 BLEU 的改善，WMT14 英→捷是文中未观察到同样增益的例外；作者也发现训练时未见过多种切分，推理时做 n-best 切分会使一些方向退化。这个证据说明切分采样要与训练和解码方式共同设计，数字不能直接迁移成现代大模型的收益承诺。

## 3.5 字节级 BPE 与 byte fallback：两种覆盖策略

“字节级”常被用来指两种不同设计，不能因为都出现 byte token 就当成同一种 tokenizer：

| 设计 | 基础原子与常见路径 | 未见字符如何处理 | 主要边界 |
| --- | --- | --- | --- |
| Byte-level BPE | 以 256 种原始字节为基础单位，再按训练得到的 merge ranks 合并字节串 | 任意输入字节都能拆成基础 byte token，因此没有字符级 OOV | 一个 Unicode 字符的 UTF-8 字节可以被拆到多个 token；流式解码要缓存不完整字节片段 |
| Unicode 子词 + byte fallback | 先把常见 Unicode 字符/子词纳入普通词表 | 词表未覆盖的字符才拆成 UTF-8 byte token | 常见字符可较短表示，但 byte fallback 触发时序列变长；byte fallback 不是 byte-level BPE |

Byte-level BPE 的 token id 仍然是有限词表中的整数，只是最底层覆盖单位是字节；它不是让模型直接处理“任意 id”。Byte fallback 通常也只针对未覆盖字符，不表示整个输入都走 byte-level BPE。

对任意非空输入，byte-level 基础字节可以保证退化为至少一个 token：

~~~math
x\ne\varnothing
\Longrightarrow
|T_{\mathrm{byte}}(x)|\ge 1
~~~

在可逆实现中，解码应恢复原始字节或规范化后的字符串。它几乎消除了词表层面的 OOV，但没有保证序列短、语义边界合理或模型已经学会某个字符的使用方式。流式输出时，一个 token 可能只含某个 UTF-8 字符的前几个字节；解码器应缓冲到字节序列完整再转成 Unicode，不能把中间碎片当成模型输出乱码。

byte-level 设计可覆盖脏输入、代码、emoji、混合语言和特殊符号；代价是某些语言或字符可能产生更多 token。覆盖率和效率是两个不同指标：它解决“能不能编码”，不自动解决“编码是否经济”。GPT-2 的 byte-level BPE 与 SentencePiece 的 byte-fallback 配置可以分别作为两种实现入口；实际模型必须看自己保存的 tokenizer 文件和配置。

## 3.6 词表大小：序列长度和输出层成本的交换

### 3.6.1 参数账本

词表大小为 `V`，hidden size 为 `d`。输入 embedding 参数量约为：

~~~math
P_{\mathrm{emb}}=Vd
~~~

如果输出层不与 embedding 共享权重，输出投影也约为 `Vd`。给定 batch `B` 和序列长度 `L`，logits 形状为：

~~~math
\ell\in\mathbb{R}^{B\times L\times V}
~~~

词表越大，常驻参数、输出计算和概率归一化越重；词表越小，同一文本可能需要更多 token。共享输入输出权重可以减少参数，但不会消除词表对输出分类规模的影响。

### 3.6.2 词表大的好处

常见短语、代码片段和多语言单位可能被更紧凑地表示；上下文能容纳更多字符；自回归生成的 token 步数可能减少。前提是这些 token 在训练语料中出现足够多，并且不是把大量低频字符串硬塞进词表。

### 3.6.3 词表大的代价

embedding 和输出层增大，低频 token 训练不足，softmax 和分片通信更重。词表过度偏向高资源语言或某个领域，也会让其他语言和代码的序列成本变差。

### 3.6.4 一个词表扩大的账本

设 hidden size `d=4096`，把词表从 `V=32,000` 扩到 `128,000`：输入 embedding 从约 `1.31 亿` 个参数增到 `5.24 亿`。若输出层不共享权重，还要再增加同样数量。全序列训练时，输出投影约需 `B·L·d·V` 次乘加；以 FLOPs 计通常再乘 2，尚未计入 softmax、通信和 padding。

假设新 tokenizer 把一段文本从 `L=1,000` 个位置压到 `600` 个位置，比较 logits 的位置-词表格点数：

| 配置 | `L × V` | 相对旧配置 |
| --- | ---: | ---: |
| 旧词表：`1,000 × 32,000` | 3,200 万 | 1.0× |
| 新词表：`600 × 128,000` | 7,680 万 | 2.4× |

这个教学账本只计算输出位置数乘词表大小；它显示序列缩短 40% 不代表输出层工作量也下降。实际选择还要算注意力和 MLP、embedding 参数、词表并行通信、输入 token 预算以及长上下文收益，不能从“每句话少几个 token”单独推出总训练更快。

### 3.6.5 不能只看平均压缩率

定义一个语料桶 `D` 的每字节 token 比例：

~~~math
\rho_{\mathrm{byte}}(D)
=
\frac{\sum_{x\in D}|T(x)|}
{\sum_{x\in D}|x|_{\mathrm{byte}}}
~~~

这个比例只在语料桶含有至少一个非空字节串时有定义；全空或解析失败的桶应单独报告覆盖问题，不能填写为零压缩率。还要按语言、代码、数学、JSON、长文档和短消息分别报告分位数。平均值可能掩盖低资源语言极端膨胀，或者掩盖 JSON 括号和缩进造成的额外 token。

如果上下文窗口为 `C_tok`，某语料桶的平均每字符 token 比例为 `rho_char`，可粗略估计字符容量：

~~~math
C_{\mathrm{char}}
\approx
\frac{C_{\mathrm{tok}}}{\rho_{\mathrm{char}}}
~~~

这里 `C_{\mathrm{tok}}>0` 且 `\rho_{\mathrm{char}}>0`；空文本或未定义压缩率不能推出“无限字符容量”。这是成本和容量的直觉，不是精确的文档长度保证；special token、模板、padding、截断和语言混合都会改变实际结果。

### 3.6.6 压缩率重要，但它不是 tokenizer 质量的全部

代表性实验对“token 越少，模型一定越好”给出了不同侧面的证据：

| 研究 | 它改变/研究什么 | 作者观察 | 结论边界 |
| --- | --- | --- | --- |
| [Unpacking Tokenization](https://arxiv.org/abs/2403.06265) | 通过 tokenizer 训练语料规模改变 BPE 压缩能力，并在英语和土耳其语上训练 10M、128M、1B 模型 | 压缩率与下游结果在其任务和设置下相关；相关性在生成任务和较小模型上更明显 | 资源受限下的受控实验；不能由相关性推出任意 tokenizer 的压缩提升都会转成质量收益 |
| [Tokenization Is More Than Compression](https://arxiv.org/abs/2402.18376) | 比较预分词、词表构造和切分方法；用能最小化 token 数的 PathPiece 检查“只追求更短” | 350M 模型的 10 项任务、三种词表大小上，没有统计显著地全面胜出算法；最短 token 序列本身不是最佳性能保证，预分词边界和词表初始化也影响结果 | 结论绑于英语数据、论文模型、任务和候选算法；不同模型规模的相对排序也会变化 |
| [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425) | 用 17 种 tokenizer 比较平行语义内容在不同语言中的长度 | 论文报告部分语言对 token 长度差距最高约 15 倍；即使字符级或字节级表示，部分语言对仍有超过 4 倍差距 | 这是论文所测语料和 tokenizer 的结果快照；它说明要审计语言负担，不是当前所有产品的固定比例 |
| [Tokenizer Choice for LLM Training](https://aclanthology.org/2024.findings-naacl.247/) | 在 2.6B 模型规模训练 24 个单语/多语模型，消融 tokenizer 算法和参数 | tokenizer 选择会改变下游表现与训练成本；论文报告其英语中心词表在所测多语设置中带来最高 68% 额外训练成本，但 fertility / parity 不总能预测任务结果 | 结果对应论文的模型、语言和训练方案；不能把 68% 当成所有 tokenizer 替换的固定成本 |
| [Objective vs. Search](https://aclanthology.org/2026.emnlp-main.2307/) | 把压缩/似然目标与 bottom-up/top-down 搜索拆成 2×2 对照，新增 BottomUpLL 与 TopDownComp | 在最多 1B、英语及五种语言实验中，bottom-up 方法在多数设置有更低 BPB；BLiMP 语法判断没有稳定排序 | 研究只到 1B，跨语种覆盖有限，较大模型和部分多语配置只有单随机种子；不能推出 bottom-up 在所有任务更优 |

论文对压缩和模型质量的观察看起来不同，先要检查它们改变了什么。`Unpacking Tokenization` 改变 tokenizer 的训练数据量，`Tokenization Is More Than Compression` 对照切分、预分词与词表构造，Ali 等人则训练一组 2.6B 单语和多语模型；它们的模型、语料、目标和计算预算并不相同。压缩可以是有用的成本信号，却不是通用的质量代理：token 变少通常减少给定文本的序列位置，但词表会扩大 embedding / 输出层，且 fertility、parity 与目标任务成绩的相关性会随实验变化。

2026 年两项模型群研究进一步说明 intrinsic metric 的预测力依赖评测范围。[No Single Tokenizer Feature Reliably Predicts Downstream Language Model Performance](https://aclanthology.org/2026.findings-emnlp.96/) 分析约 200 个、覆盖九种语言的模型，没有发现某一个 tokenizer intrinsic 指标能跨模型家族、任务和语言稳定预测下游成绩；[TokLens](https://aclanthology.org/2026.acl-srw.18/) 对 24 个 tokenizer、15 种语言计算六类指标，在控制模型规模后未发现这些指标预测英语 benchmark 的稳定关系，却在其多语 MMLU-ProX 分析和 Qwen2.5 家族实验中发现与单语表现/扩展斜率相关的信号。TokLens 的多语结果仍可能受各语言预训练数据组成混杂，而且其任务覆盖只有七个有可比成绩的模型、十一种语言。两者不是“指标完全无用”和“指标可以选出好 tokenizer”的二选一：前者强调跨家族泛化失败，后者给出特定多语评测中的相关性；相关性不能替代受控重训。

从工程方案看，tokenizer 设计也不只是选 BPE 还是 Unigram。[MUTANT](https://aclanthology.org/2026.acl-long.2146/) 将多语语料配比、语言/脚本感知的预分词、子词训练和第二阶段多词单元训练组合成一套 recipe。作者在英语、22 种印度语言和代码上训练 200K 词表，并报告平均 fertility 相对 LLaMA-4 低 39.5%、相对 Sutra 低 18%；1B 模型的作者自测中，OTPT 从 117.99 增至 169.42 tokens/s，TTFT 为 19.17ms 与 18.98ms。该吞吐对照使用 200 条输入、5 次 warmup、10 轮结果、8 张 H100、Triton 和最多 256 个新 token；下游训练模型最高 1B，作者也明确说其他非印度语系、更大模型和形态分析部署仍未验证。更换 tokenizer 会改变一个“token”包含的文本量，所以 tokens/s 不能独立代表用户可见文本吞吐；部署对照还应固定任务质量，并报告 bytes/s 或 characters/s、请求完成时间和端到端成功率。

因此，选型时应同时报告 token/byte、按语言和领域的长度分布、编码可逆性、fallback 比例、embedding/输出层账本、统一分母的 BPB，以及模型在相同数据和受控训练预算下的任务结果。快速 tokenizer 统计适合淘汰覆盖差或明显膨胀的候选；要声称它提高了模型能力或实际服务效率，还需要目标任务和目标运行栈上的验证。

tokenizer 变更还会改变“预算相同”的含义：固定训练 token 数时，模型可能看到不同数量的原始字节；固定原文字节时，训练 token 数和计算量又不同。研究比较至少要说明自己固定了原文样本、token 数还是 FLOPs。跨 tokenizer 的语言建模损失也不要只比每 token perplexity；在编码可逆、规范化一致、字节分母相同的条件下，可再报告每原文字节的平均负对数似然：

~~~math
\mathrm{BPB}
=
\frac{-\sum_{t=1}^{L}\log_2 p_\theta(z_t\mid z_{1:t-1})}
{|x|_{\mathrm{byte}}}
~~~

`BPB` 是 bits per byte，分母是同一规范化文本的 UTF-8 字节数。序列是否包含 BOS/EOS 必须在对照模型间保持一致；若 tokenizer 丢失空格、改写原文或 `N(x)` 不一致，这个比较就不再只隔离 tokenizer 的差异，必须先修复数据协议。

## 3.7 Special token：把普通序列变成结构化协议

### 3.7.1 三类控制符号

| 类型 | 可能的例子 | 作用 |
| --- | --- | --- |
| 序列控制 | BOS、EOS、PAD、UNK | 开始、结束、占位和未知输入 |
| 角色/事件标记 | system、user、assistant、tool | 对话角色、工具调用和结果边界 |
| 模态占位 | image、audio、video | 在语言序列中表示外部模态位置 |

这些名字不是跨模型标准。一个模型可能把 `<assistant>` 注册为单独 vocab id，另一个模型可能把一段字符串模板展开成多个普通 token；一个 `<image>` 可能只占一个文本位置，也可能对应 vision encoder 产生的多段连续 embedding。

查看具体模型时，应确认 token 字符串、id、`special` 属性、解码行为、模板插入方式和停止条件。

还要区分“模板插入的控制 token”和“用户正文里长得像控制 token 的普通文本”。有的 tokenizer 会把输入文本中完全匹配的标记直接编码成专用 id；若外部文档包含类似 `<|assistant|>` 的字面串，它可能被解析成结构，而不是普通引用文本。预处理器应按目标协议转义、清洗或作为不可信内容标注，并在固定样例中确认控制标记只由模板产生。Hugging Face 的 `apply_chat_template` 提供 `sanitize_control_tokens` 作为一个实现例子；该选项和行为依赖 Transformers 版本及模板，不是所有框架的通用默认保护。

### 3.7.2 EOS 的训练与推理语义

“一条消息结束”“assistant 回合结束”和“把控制权交给工具”是不同事件，不能都叫 EOS：

| 语义事件 | 运行时要做什么 | 若标记混用会怎样 |
| --- | --- | --- |
| 消息结束 | 读完当前 role 的消息，继续解析序列 | assistant 可能还有后续 channel 或消息 |
| assistant 回合完成 | 停止本轮采样并把回答交给调用方 | 如果漏停会继续生成；过早停会截断答案 |
| 工具交接 | 停止模型采样，校验并派发 tool call，等待工具结果后再继续 | 如果当普通答案展示，工具不会执行；如果把普通文本当交接，会误触发调用 |

OpenAI 的 gpt-oss Harmony 文档给出一个具体协议例子：`<|end|>` 结束消息，`<|return|>` 表示模型完成当前回复，`<|call|>` 表示向工具交接。这三个字符串是该协议的 token，不是通用词表；别的模型可能用不同标记或把消息终止和回合终止映射到不同 id。训练标签、生成停止 id、tool dispatcher 与对话历史拼接方式必须共用同一份版本化协议定义。

预训练文档常在结尾追加 EOS；chat 模型则可能每条消息、每个 assistant 回合和工具交接都有不同边界。训练里漏掉结束标记，会让模型不容易学会何时停；服务端把停止 id 配错，可能不停生成或过早结束。停止条件优先按 token id/结构事件处理，不要只靠解码后的字符串匹配，因为 byte-level tokenizer、特殊 token 解码和流式输出可能把同一可见字符串拆开或延后输出。

### 3.7.3 PAD、attention mask 和 loss mask

padding 是为了把不同长度样本放进规则张量。定义：

~~~math
a_t=\mathbf{1}[t<\ell]
~~~

这里按右 padding 书写，`\ell` 是该样本在 padding 前的真实长度；若使用左 padding，应从真实起始位置或 collator 输出计算有效位置。不能总用 `z_t\ne z_{\mathrm{pad}}` 推断真实位置：有些训练配置会让 PAD 与 EOS 共用同一个 id；此时按 id 比较会把真实结束符也误判成 padding。模型的 causal mask 还要阻止当前位置看未来 token；packed 独立样本可能还需要 document/block mask。三类作用不同，不能混为一个“attention mask”。

loss 也要忽略 PAD。很多 PyTorch 训练代码用 `ignore_index=-100` 表示不计算该位置，但 `-100` 是实现约定，不是模型词表中的真实 token：

~~~math
y_t=
\begin{cases}
z_t,&a_t=1\\
-100,&a_t=0
\end{cases}
~~~

在 batch 内，PAD 对应的 label 应设成 ignore index；token 与 label 是否相同必须逐位置核对。若 PAD key 没有被 attention mask 屏蔽，真实 token 可能读取填充位置的隐藏状态；若 EOS 和 PAD 共用 id，则按 id 构造 mask 又可能挡住真实 EOS。保存 padding 前长度或显式 `attention_mask` 才能把这两个目的区分开。

上式只负责把 padding 位置排除在 token loss 外；若任务还要求 assistant-only loss，还要再与 role/generation mask 合并，不能把“非 PAD”当成“应监督”。

## 3.8 Chat template：消息如何变成生成提示

### 3.8.1 两层转换

消息对象可以写成：

~~~math
M=((r_1,c_1),(r_2,c_2),\ldots,(r_K,c_K))
~~~

其中 `r_k` 是 role，`c_k` 是内容。chat template `T_chat` 先把消息序列化为字符串或事件序列：

~~~math
s=T_{\mathrm{chat}}(M)
~~~

再由 tokenizer 编码：

~~~math
z=T(s)
~~~

因此训练和推理真正使用的是 `T(T_chat(M))`，而不是原始 JSON。模板可能决定 system、user、assistant、tool 的边界、换行、生成起点、EOS 和工具参数格式。

### 3.8.2 模板不一致的症状

训练模板和推理模板不同，常见症状包括：

1. 模型把用户问题当成待续写文本。
2. assistant 回答起点偏移，出现重复角色标签。
3. EOS 和停止条件失效。
4. 多轮历史顺序或 system 约束丢失。
5. 工具调用 JSON 外面出现解释文字或格式错误。

因此模板应像模型权重一样版本化。评估和线上服务必须使用与训练数据一致的序列化规则，除非实验明确研究模板迁移。

### 3.8.3 生成提示和训练标签

推理时通常把 assistant 起始标记放在输入末尾，要求模型从这里开始生成。SFT 时则把完整 assistant 内容放入序列，同时生成 label mask。二者使用同一模板但 mask 不同：推理没有真实 assistant label，训练有。

### 3.8.4 模板是可执行的数据协议

模板不只是把 role 名称替换成几个字符串。它可能按消息是否为最后一条、是否有工具、content 是文本还是图像块、assistant 属于哪个 channel 等条件分支。推理端的两个常见选项也不能互换：`add_generation_prompt=True` 通常在对话尾部追加一个新的 assistant 起始头；`continue_final_message=True` 则去掉末尾结束标记，让模型接着已有 assistant 前缀生成。Hugging Face Transformers 的文档规定二者不能同时启用，并明确模板是否支持 generation prompt 取决于模型实现。

模板本身已经可能写入 BOS、EOS 和角色边界。若先 `apply_chat_template(..., tokenize=False)` 得到字符串，再由 tokenizer 编码，通常需要 `add_special_tokens=False`，否则第二次编码可能再插一套 BOS/EOS。能直接在模板函数里 `tokenize=True` 时，框架会把生成后的控制标记作为实际 special token 编码，减少重复插入风险。若改模板、模型 revision 或库版本，应保存渲染结果和 token ids 做回归，而不是只检查模板字符串是否“看起来一样”。

## 3.9 Label mask：输入可以完整，监督必须有边界

### 3.9.1 Assistant-only SFT

一个对话样本可能是：

~~~text
<system> 你是严谨助手。
<user> 解释 attention。
<assistant> attention 是信息聚合机制。 <eos>
~~~

模型输入包含全部 token，但可以只让 assistant 输出位置参与 loss：

~~~text
system tokens   -> ignore
user tokens     -> ignore
assistant tokens -> compute loss
~~~

令 `m_t=1` 表示第 `t` 个 token 是监督目标，`m_t=0` 表示只作为条件，则 causal LM 的 masked loss 可以写成：

~~~math
L_{\mathrm{SFT}}
=
-\frac{1}{\max(\sum_{t=1}^{L-1}m_{t+1},1)}
\sum_{t=1}^{L-1}
m_{t+1}
\log p_\theta(z_{t+1}\mid z_{1:t})
~~~

教学代码常用 `max(\cdot,1)` 避免除零，但它不能把没有监督的位置变成有效样本。真实训练应要求 `\sum_{t=1}^{L-1}m_{t+1}>0`；若模板、截断或 role 标注让分子和监督分母都为空，应跳过并记录该样本，而不是以零 loss 混入 batch。分母使用有效 assistant token 数，避免不同回答长度让 loss 口径变化。system、user、tool result 和 padding 是否被 mask，要由训练目标决定，而不是机械按角色名称处理。

### 3.9.2 多轮和工具数据的 mask

如果目标是生成 assistant 回复，历史 user 和 tool result 通常是条件；如果目标是学习生成 tool call，tool call 的函数名和参数应成为监督区域；如果目标是预测工具结果，则需要明确结果是否来自可信环境，不能把未验证的字符串当作标准答案。

对偏好训练，chosen/rejected 都应在相同 prompt 和模板下计算回答 token 的对数概率。对 GRPO、RLVR 等轨迹数据，还可能需要组内候选、奖励、验证器和环境状态，不能用单一 assistant mask 表示全部监督。

### 3.9.3 Label 在模型里还要右移一位

许多数据集把 `labels[t]` 与 `input_ids[t]` 按相同位置存储，但 causal LM 训练时会将 logits 和 labels 错开一位：位置 `t` 的 logits 预测 `labels[t+1]`。例如：

| 位置 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `input_ids` | BOS | user | 问题 | 消息结束 | assistant 头 | 答案 1 | 回合结束 |
| 存储的 `labels` | `-100` | `-100` | `-100` | `-100` | `-100` | 答案 1 | 回合结束 |
| 参与 loss 的 logits 目标 | 忽略 | 忽略 | 忽略 | 忽略 | 答案 1 | 回合结束 | — |

这样答案第一个 token 由 assistant 头作为上下文来预测，结束 token 也能被训练出来。若 label mask 误向左或右移一格，loss 可能照常下降，却会变成预测用户末尾、跳过答案第一个 token，或把不该监督的边界计入损失。检查时应打印 `input_ids[t]`、`logits[t]` 对应的 `labels[t+1]`，不要只看一条未 shift 的 role/mask 列表。某些模型/训练器会内部 shift，外部数据 collator 则可能显式 shift；只能选一处做，不能重复右移。

batch reduction 也是数据语义的一部分。按有效 token 总数求平均，与先算每个样本平均再对样本求平均，会给长答案不同权重：

~~~math
L_{\mathrm{token}}
=
\frac{\sum_i\sum_t m_{i,t}\ell_{i,t}}
{\sum_i\sum_t m_{i,t}},
\qquad
L_{\mathrm{sample}}
=
\frac{1}{N}\sum_i
\frac{\sum_t m_{i,t}\ell_{i,t}}
{\sum_t m_{i,t}}
~~~

假设样本 A 有 10 个监督 token、平均 loss 为 1；样本 B 有 100 个监督 token、平均 loss 为 2。token 平均是 `(10×1+100×2)/110≈1.91`，样本平均则是 `(1+2)/2=1.5`。前者让每个目标 token 权重相同，后者让每条样本权重相同；分布式训练还要保证跨 rank 聚合时的有效 token 分母口径一致。两种都可以是有意设计，但不能默默换口径。

## 3.10 预训练文本、EOS 与 packing

### 3.10.1 文档序列化

预训练常把文档编码后追加 EOS：

~~~math
s_i=T(x_i)\oplus[z_{\mathrm{eos}}]
~~~

所有文档拼接为：

~~~math
S=s_1\oplus s_2\oplus\cdots\oplus s_n
~~~

再按上下文长度 `C` 切成 block：

~~~math
b_j=S_{jC:(j+1)C-1}
~~~

EOS 告诉模型文档边界，但在普通 causal mask 下，后一个文档仍可看到前一个文档的历史。若希望样本完全独立，要使用 document-level attention mask 或分别计算样本；如果允许跨文档上下文，则必须在评估中接受这个训练假设。

SFT packing 和预训练连续语料尤其不能混为一谈。假设把两个样本接成 `[BOS_A,A_1,A_2,END_A | BOS_B,B_1,B_2,END_B]`：普通 causal attention 会允许 `B_1` 读取 `A` 的全部 token；若两个样本应独立，必须在注意力上隔开两段，而 EOS 本身不会清除 KV 或改写 attention mask。小例子中的允许注意力矩阵是：

~~~text
         BOS_A A_1 A_2 | BOS_B B_1 B_2
BOS_A      1   0   0  |  0    0   0
A_1        1   1   0  |  0    0   0
A_2        1   1   1  |  0    0   0
BOS_B      0   0   0  |  1    0   0
B_1        0   0   0  |  1    1   0
B_2        0   0   0  |  1    1   1
~~~

实现可以显式构造 block-causal mask，也可以把每个独立序列的 cumulative lengths 传给支持变长输入的 attention kernel。后者跳过跨样本的注意力计算；理论 attention 工作量可保持为 `\sum_i s_i^2`，而不是把总长度 `\sum_i s_i` 当成一个样本计算平方。具体格式、position id 是否在边界重置、padding 如何对齐，都由模型和训练 kernel 的 packing contract 决定。

边界还会影响 next-token label。若 `BOS_B` 是 `B_1` 的上下文，`B_1` 可以照常监督；若样本 B 没有 BOS，而它的首 token 在扁平流里紧接着 A，那么不能让 A 的末 token 来预测 B 的首 token，除非训练目标本来就允许跨文档条件。应在 boundary 处插入模型规定的起始符，或屏蔽那一个跨样本预测，并检查 position id、label 和 attention 的边界采用同一序列切分。

### 3.10.2 Packing 利用率

固定 block 总位置数为 `M`，真实有效 token 数为 `V`，可以定义利用率：

~~~math
\eta_{\mathrm{pack}}=\frac{V}{M}
~~~

这里 `M>0` 且 `0\le V\le M`；没有形成 block 的空输入不属于“利用率为 100%”的样本。利用率高只说明 padding 少，不说明边界正确。NeMo 的一个 SFT 实现把 pack 的 cumulative sequence lengths 传给变长 attention，避免样本间 attention；这是该训练栈的实现方式，不是所有框架都采用相同数据布局。应同时记录每个文档的起止位置、EOS 数、attention/block mask、position ids、padding 比例和被截断样本数。

### 3.10.3 截断策略

长样本超出上下文窗口时，截断开头可能丢失 system 或问题，截断结尾可能丢失答案或工具结果，随机窗口可能破坏文档结构。对训练数据，策略要与目标任务匹配；对评估数据，必须报告截断和滑窗规则。

## 3.11 SFT 数据格式：JSON 只是容器

### 3.11.1 单轮与多轮

单轮数据可以有 instruction、input、output 字段，多轮数据通常使用 messages：

~~~json
{
  "messages": [
    {"role": "system", "content": "你是严谨助手。"},
    {"role": "user", "content": "解释 RoPE。"},
    {"role": "assistant", "content": "RoPE 使用旋转变换编码相对位置信息。"}
  ],
  "target_roles": ["assistant"]
}
~~~

真正进入训练的是：

~~~math
d_i=(M_i,z_i,m_i),
\qquad
z_i=T(T_{\mathrm{chat}}(M_i))
~~~

`M_i` 是消息，`z_i` 是 token ids，`m_i` 是 label mask。JSON 字段名字可以不同，但这三个对象必须能被重建和审计。

### 3.11.2 数据质量边界

SFT 数据需要检查回答事实性、格式、角色、拒答、安全、长度、语言和重复。assistant 内容里的工具结果、引用和代码必须有来源或验证状态；否则模型会把错误输出学成标准答案。

### 3.11.3 训练框架字段不等于模型协议

不同框架可能使用 `prompt/completion`、`messages`、`text` 或自定义字段。字段名只属于数据加载器，真正的模型协议由模板、tokenizer 和 labels 共同决定。迁移数据时不能只改字段名，还要重新检查序列化、special token、mask 和停止条件。

把数据保存成 token ids 后，还应记录原始样本 id、tokenizer hash、template hash、规范化版本、token ids、labels、mask、截断方向和截断原因。仅保存 `input_ids` 而不保存原文和协议版本，无法判断新模型里同一个 id 还表示不表示同一片段，也无法重新生成正确的 assistant mask。

## 3.12 偏好、强化和可验证数据格式

### 3.12.1 Pairwise preference

最常见的偏好结构是同一个 prompt 下有两个回答：

~~~json
{
  "prompt": "用户问题。",
  "chosen": "更符合目标的回答。",
  "rejected": "存在问题的回答。",
  "criteria": ["correctness", "helpfulness"],
  "source": "human_reviewed"
}
~~~

可以写成：

~~~math
d_i=(x_i,y_i^+,y_i^-)
~~~

两条回答必须共享 prompt、tokenizer、模板和截断规则。否则模型可能学到长度、格式或截断位置的差异，而不是回答质量。

用于偏好优化时，最好把样本进一步保存为“共同上下文 + 两个回答”，并附上原文/序列化消息、chosen/rejected token ids、回答 mask、标注者或 judge 来源、比较标准、评分时间、tokenizer/template revision 和截断信息。不要分别 tokenize `prompt` 与回答后直接拼 ids：空格、byte-level BPE 合并和模板边界可能让分开编码的结果不同于完整串行化后编码；应在同一个协议边界上编码并验证公共前缀。

### 3.12.2 组内候选和验证器

GRPO、RLVR 或其他强化训练可能保存一个 prompt 的多个候选、组内 reward、验证器输出、环境状态和工具轨迹。可把一组数据抽象为：

~~~math
G_i=\left(x_i,\{y_{i,j},r_{i,j},v_{i,j}\}_{j=1}^{n_i}\right)
~~~

`y` 是候选，`r` 是奖励或相对分数，`v` 是验证器证据。奖励来自字符串匹配、代码执行、数学检查还是人工偏好，会影响训练信号的含义，必须记录来源。

一次 rollout 的审计记录通常还需要行为策略的模型 revision、采样参数、生成 token 的旧 log-probability、`group_id`、response/loss mask、终止原因、reward 分项、verifier 版本和环境事件。离线偏好对没有这些在线采样字段；来自在线 GRPO/RLVR 的轨迹则不能只压扁成 chosen/rejected 文本，否则很难复核样本由哪个策略产生、工具结果是否可信、奖励是否过期。不同算法的训练器 schema 不统一，这里列的是复现与追踪所需的账本字段，不是某个库的固定 API。

### 3.12.3 长度和偏好偏差

如果 chosen 几乎总比 rejected 长，偏好优化可能学习冗长；如果安全回答总是拒绝，模型可能过度拒答；如果 judge 对格式有偏好，模型可能优化表面结构。偏好目标常用整段回答的 log-probability 或其比值，回答 token 数和 tokenizer 切分都会影响聚合尺度，所以不能在更换 tokenizer 后沿用旧 token-level 长度统计或未经校准的 loss。偏好数据要做长度、语言、领域、风险等级和错误类型的分层统计，并固定是否按 token、按回答或按 pair 平均。

## 3.13 Tool calling：数据格式连接模型和外部副作用

### 3.13.1 一条完整轨迹

一个工具轨迹可能是：

~~~text
<user> 查询北京天气
<assistant_tool_call> {"name":"weather","arguments":{"city":"北京"}}
<tool_result> {"temperature":"20C","condition":"晴"}
<assistant> 北京今天晴，约 20C。
~~~

训练数据要明确哪个阶段由模型生成，哪个阶段由真实工具返回，哪些字段被验证。只把 tool result 当作普通文本，会让模型学习伪造工具结果或混淆角色。

结构化轨迹至少要把三个事件分开：assistant 产生带 recipient 的调用与参数；运行时校验后由工具执行并以工具名作为来源返回结果；assistant 读取结果后再产生最终消息。OpenAI gpt-oss 的 Harmony 示例把 recipient 放在 assistant header、JSON 约束放在格式元数据中，并以专用 call 标记交接；这展示了“调用内容”和“调度控制”如何同时编码。它只是一个已发布协议实例，训练别的模型必须按该模型 template 重建边界。

### 3.13.2 Schema 和边界检查

工具样本至少要检查：

~~~math
\mathbf{c}_{\mathrm{tool}}
=
\left(
c_{\mathrm{name}},
c_{\mathrm{json}},
c_{\mathrm{schema}},
c_{\mathrm{result}},
c_{\mathrm{boundary}}
\right)
~~~

分别表示工具名、JSON 语法、参数 schema、结果对应关系和角色边界。即使这些字段全部正确，也不能推出运行时安全；权限、幂等、超时、重试、确认和未知状态属于 Agent runtime 的职责。

### 3.13.3 训练与运行时的边界

训练可以教模型生成合法参数，不能保证工具授权正确，也不能让一个有副作用的 API 自动变成安全动作。生产系统仍需在执行前校验 schema、权限和用户确认，并记录调用 trace。

对每条轨迹，至少保存调用样本的原始输入、模型生成的 recipient / arguments、handoff 位置、tool schema revision、执行结果来源、调用状态（成功/错误/超时/取消）、环境时间和 assistant 后续回答。通常模型生成的工具选择、参数和交接标记参与 assistant loss；工具观察结果是否参与 loss 由训练目标决定。若训练数据伪造一个成功结果、却没有环境 trace，模型会学到的是一段“看起来像工具轨迹”的文本，而不是经验证的工具行为。

## 3.14 多模态 token 与占位符

### 3.14.1 `<image>` 不一定是一个视觉 token

多模态输入常见结构为：

~~~text
<user> <image> 描述图片
<assistant> 图片中有一只猫。
~~~

`<image>` 可能是一个文本侧占位符，真实视觉信息由 vision encoder、projector、resampler 或 cross-attention 提供。它也可能展开成多个 patch embedding。不能只数字符串中有几个 `<image>` 就推断视觉序列长度。

### 3.14.2 粗略 token 账本

图片高宽为 `H,W`，patch 边长为 `P`，若每个 patch 保留一个位置，视觉位置数粗略为：

~~~math
N_{\mathrm{image}}
=
\left\lceil\frac{H}{P}\right\rceil
\left\lceil\frac{W}{P}\right\rceil
~~~

文本、图像、音频和 special 位置合计为：

~~~math
N_{\mathrm{seq}}
=
N_{\mathrm{text}}
+N_{\mathrm{image}}
+N_{\mathrm{audio}}
+N_{\mathrm{special}}
~~~

该估算要求 `H,W,P>0`，并把 padding、裁剪和多图顺序固定下来；零尺寸媒体或零 patch size 是预处理错误，不是零视觉 token。真实模型可能先压缩视觉表示，也可能使用连续 embedding 而不是词表 token。因此还要记录视觉 encoder 输出长度、时间采样、图像裁剪、projector 维度和训练时的对齐方式。

用一个纯教学参数看尺寸的影响：`448×448` 图像经 `14×14` patch 切分得到 `32×32=1,024` 个视觉位置。如果后续模块每 `2×2` 个 patch 合并成一个位置，则留下 `16×16=256` 个位置；若 16 帧都按此处理，合计是 `4,096` 个视觉位置，尚未计文本和时间戳。这个例子只演示“分辨率/帧数→模型序列长度”的路径，不是某个模型的 processor 配置。高分辨率 crop、动态分块和视频采帧策略会同时改变视觉信息、序列成本与显存。

音频也要把波形字节数、codec 帧数、量化码本数和语言模型的序列步数分开。若某 codec 每秒输出 `f` 帧、每帧有 `K` 个活跃 codebook，则原始离散 code 数可估为 `duration×f×K`；但模型可能把 codebooks 并行预测、交错排列或先压缩，所以不能把 code 总数直接当成自回归步数。多模态数据建议保存媒体文件 hash、采样/裁剪时间戳、processor revision、文本片段与媒体片段的对齐区间，以及每个处理阶段实际产生的序列长度。

### 3.14.3 视觉、音频与文本的边界

多模态数据必须同时检查模态内质量和模态间对齐：图片是否真的对应 caption，音频转写是否与时间戳一致，视频标题描述的是画面还是频道，多个图片占位符是否和输入对象顺序一致。数据格式能表示模态，不代表模型已经学会模态关系。

## 3.15 Tokenizer 扩展：改变词表就是改变模型接口

### 3.15.1 需要同步的对象

给已有模型增加工具、领域或多模态 special token 时，需要同步：

1. tokenizer vocabulary 和 special token map。
2. 输入 embedding 和输出层的行数。
3. 初始化方式和继续训练数据。
4. chat template、工具模板和多模态处理器。
5. checkpoint 分片、量化文件和服务端配置。

原词表大小为 `V`，新增 `K` 个 token 后：

~~~math
V'=V+K,
\qquad
E'\in\mathbb{R}^{V'\times d}
~~~

若输入输出权重不共享，输出矩阵也需要扩展；若权重共享，新增行仍必须在同一个参数和词表版本中保持一致。

### 3.15.2 新 token 如何获得语义

随机初始化或相关 token 均值初始化只能提供起点，不能自动赋予新 token 语义。继续训练数据必须真正使用这些 token，且 label mask、模板和生成停止规则要覆盖它们。只改 tokenizer 而不训练，模型很可能把新 id 当成未学过的随机向量。

初始化可以从“随机向量”“旧子词向量的均值/加权组合”“外部对齐后映射的向量”几类做对照。均值初始化便宜，但若许多新 token 都由相似子词取平均，可能把初始表示压到很窄的空间；外部向量映射需要额外对齐数据；子词组合继承旧模型表示，却仍要让新 token 在目标文本和输出任务中通过继续训练学会自己的输入/输出作用。

近年的词表扩展研究可帮助提出实验假设，但结果不能直接推广到任意语言或模型。FOCUS 在 XLM-R 单语种适配设置中，用辅助 fastText 空间寻找语义相近的共享 token，再组合原模型已有 embedding 来初始化新词；它解决“新 tokenizer 的 id 没有旧 embedding”问题，限制是依赖多语模型、共享 token 和辅助词向量，且论文模型并非现代生成式大模型。2026 年一篇 Hindi 扩词表预印本在 Nemotron-3-Nano-30B-A3B 上加入 25,600 个 Hindi token，使其样本中 fertility 从 1.95 降到 1.25；该文比较 20 多种初始化，报告子词组合与输入/输出不对称方案比 mean-all 更省继续预训练步数，50-step CPT probe 也比 step-zero 初始化 loss 更能预测其设置中的收敛。最佳方案对输入 embedding 与输出头使用不同组合。语种、模型、继续训练数据和评价范围都有限；应在目标模型上比较短程 CPT 收敛速度、目标语言任务和旧语言回归，而不是把某一种初始化写成通用标准。

### 3.15.3 旧数据和新词表

旧数据如果保存的是 token id，扩词表或重排词表后不能直接复用；如果保存的是原始文本，可以在新 tokenizer 下重新编码，但要重新计算 token 数、packing、训练步数和评估基线。数据版本应记录 tokenizer hash，避免同一个字段名下混入不同编码空间。

## 3.16 一个最小的数据格式审计器

下面的 toy tokenizer 不复刻真实 BPE，只用有限片段加固定的 256 个 UTF-8 byte fallback token，帮助读者观察不同文本的 token/byte、可逆性、chat template 的角色边界、assistant-only label mask、预训练 packing 和扩词表后的 embedding 行数。它按最长片段匹配，完全没有训练 merge ranks；packing 只返回有效长度 mask，没有构造 causal/document attention mask。真实系统还要处理完整 Unicode 规范化、offset mapping、批量 padding、processor 和模型特定模板。

~~~python
IGNORE_INDEX = -100
SPECIAL_TOKENS = [
    "<bos>",
    "<eos>",
    "<system>",
    "<user>",
    "<assistant>",
    "<pad>",
    "<image>",
]
BASE_PIECES = [
    "tokenizer",
    "improves",
    "attention",
    "数据工程",
    "需要",
    "保留",
    "来源",
    "def",
    "get_user_id",
    "return",
    "user_id",
    "解释",
    "你",
    "严谨",
    "助手",
    "是",
    "文本",
    "到",
    "token",
    "id",
    "的",
    "协议",
    "。",
    "(",
    ")",
    ":",
    ".",
    " ",
    "x",
]
BYTE_TOKENS = [f"<0x{value:02X}>" for value in range(256)]

vocab = {}
id_to_token = {}


def add_token(token):
    if token not in vocab:
        idx = len(vocab)
        vocab[token] = idx
        id_to_token[idx] = token
    return vocab[token]


for token in SPECIAL_TOKENS + BYTE_TOKENS + BASE_PIECES:
    add_token(token)
special_token_ids = {vocab[token] for token in SPECIAL_TOKENS}

match_pieces = sorted(BASE_PIECES, key=len, reverse=True)


def fallback_pieces(char):
    return [f"<0x{value:02X}>" for value in char.encode("utf-8")]


def tokenize_text(text):
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    pieces = []
    index = 0
    while index < len(text):
        matched = None
        for piece in match_pieces:
            if text.startswith(piece, index):
                matched = piece
                break
        if matched is None:
            char = text[index]
            pieces.extend(fallback_pieces(char))
            index += 1
            continue
        pieces.append(matched)
        index += len(matched)
    return pieces


def encode_pieces(pieces):
    if not isinstance(pieces, list) or any(
        not isinstance(piece, str) or not piece for piece in pieces
    ):
        raise ValueError("pieces must be non-empty strings")
    if any(piece not in vocab for piece in pieces):
        raise ValueError("piece is not registered in the fixed vocabulary")
    return [vocab[piece] for piece in pieces]


def decode_ids(ids):
    if not isinstance(ids, list) or any(
        not isinstance(index, int) or isinstance(index, bool) or index not in id_to_token
        for index in ids
    ):
        raise ValueError("ids must be known integer token ids")
    return [id_to_token[index] for index in ids]


def decode_text(pieces):
    if not isinstance(pieces, list) or any(
        not isinstance(piece, str) or not piece for piece in pieces
    ):
        raise ValueError("pieces must be non-empty strings")
    raw = bytearray()
    for piece in pieces:
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            raw.append(int(piece[3:5], 16))
        else:
            raw.extend(piece.encode("utf-8"))
    return raw.decode("utf-8")


def serialize_chat(messages):
    allowed_roles = {"system", "user", "assistant"}
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    pieces = ["<bos>"]
    assistant_mask = [0]
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be a mapping")
        role = message.get("role")
        if not isinstance(role, str) or role not in allowed_roles:
            raise ValueError("message role is not registered in this template")
        if not isinstance(message.get("content"), str):
            raise ValueError("message content must be a string")
        role_piece = f"<{role}>"
        pieces.append(role_piece)
        assistant_mask.append(0)

        content_pieces = tokenize_text(message["content"])
        pieces.extend(content_pieces)
        assistant_mask.extend(
            [1 if role == "assistant" else 0]
            * len(content_pieces)
        )

        if role == "assistant":
            pieces.append("<eos>")
            assistant_mask.append(1)

    if not any(assistant_mask):
        raise ValueError("SFT sample has no assistant supervision")
    input_ids = encode_pieces(pieces)
    labels = [
        token_id if mask else IGNORE_INDEX
        for token_id, mask in zip(input_ids, assistant_mask)
    ]
    return input_ids, labels, assistant_mask


def pack_documents(documents, block_size):
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 1
    ):
        raise ValueError("block_size must be an integer greater than one")
    if not isinstance(documents, list) or not documents:
        raise ValueError("documents must be a non-empty list")
    blocks = []
    valid_lengths = []
    current = []
    for document in documents:
        if not isinstance(document, str) or not document:
            raise ValueError("documents must be non-empty strings")
        doc_ids = encode_pieces(tokenize_text(document) + ["<eos>"])
        if len(doc_ids) > block_size:
            raise ValueError("document exceeds block_size; apply an explicit truncation policy")
        if len(current) + len(doc_ids) > block_size and current:
            pad_len = block_size - len(current)
            blocks.append(current + [vocab["<pad>"]] * pad_len)
            valid_lengths.append(len(current))
            current = []
        current.extend(doc_ids)
    if current:
        pad_len = block_size - len(current)
        blocks.append(current + [vocab["<pad>"]] * pad_len)
        valid_lengths.append(len(current))
    return blocks, valid_lengths


texts = {
    "en": "tokenizer improves attention",
    "zh": "数据工程需要保留来源",
    "code": "def get_user_id(x): return x.user_id",
}
fallback_probe = "🧪"
compression = {
    name: {
        "chars": len(text),
        "bytes": len(text.encode("utf-8")),
        "tokens": len(tokenize_text(text)),
        "tokens_per_char": round(len(tokenize_text(text)) / len(text), 3),
        "tokens_per_byte": round(
            len(tokenize_text(text)) / len(text.encode("utf-8")), 3
        ),
    }
    for name, text in texts.items()
}

messages = [
    {"role": "system", "content": "你是严谨助手。"},
    {"role": "user", "content": "解释 tokenizer"},
    {"role": "assistant", "content": "tokenizer 是文本到 token id 的协议。"},
]
input_ids, labels, assistant_mask = serialize_chat(messages)
label_tokens = [
    id_to_token[token_id] if token_id != IGNORE_INDEX else "IGN"
    for token_id in labels
]
first_assistant_pos = assistant_mask.index(1)
prompt_label_count = sum(
    1 for value in labels[:first_assistant_pos] if value != IGNORE_INDEX
)
assistant_label_count = sum(value != IGNORE_INDEX for value in labels)

blocks, packed_lengths = pack_documents(list(texts.values()), block_size=16)
valid_masks = [
    [int(position < valid_length) for position in range(16)]
    for valid_length in packed_lengths
]

old_vocab_size = len(vocab)
for token in ["<tool_call>", "<tool_result>"]:
    special_token_ids.add(add_token(token))
new_vocab_size = len(vocab)

packed_eos_count = sum(block.count(vocab["<eos>"]) for block in blocks)
checks = {
    "specials": all(token in vocab for token in SPECIAL_TOKENS),
    "tool_controls_registered": all(
        vocab[token] in special_token_ids
        for token in ["<tool_call>", "<tool_result>"]
    ),
    "byte_fallback_roundtrip": all(
        decode_text(tokenize_text(text)) == text for text in texts.values()
    ) and decode_text(tokenize_text(fallback_probe)) == fallback_probe,
    "assistant_only_labels": prompt_label_count == 0 and assistant_label_count > 0,
    "length_based_padding_mask": all(
        len(block) == 16
        and sum(mask_row) == valid_length
        and block[valid_length:] == [vocab["<pad>"]] * (16 - valid_length)
        for block, mask_row, valid_length in zip(
            blocks, valid_masks, packed_lengths
        )
    ),
    "packing_shape": all(len(block) == 16 for block in blocks),
    "eos_boundary": packed_eos_count == len(texts),
    "vocab_resize": new_vocab_size == old_vocab_size + 2,
}

print("compression=", compression)
print("byte_fallback_tokens=", tokenize_text(fallback_probe))
print("chat_tokens=", decode_ids(input_ids))
print("label_tokens=", label_tokens)
print("assistant_label_count=", assistant_label_count)
print("prompt_label_count=", prompt_label_count)
print("packed_valid_tokens=", packed_lengths)
print("packed_pad_count=", [16 - length for length in packed_lengths])
print("packed_eos_count=", packed_eos_count)
print("old_vocab_size=", old_vocab_size)
print("new_vocab_size=", new_vocab_size)
print("special_token_count=", len(special_token_ids))
print("checks=", checks)
assert all(checks.values())
try:
    serialize_chat([{"role": "user", "content": "only a prompt"}])
except ValueError:
    pass
else:
    raise AssertionError("empty SFT supervision must be rejected")
try:
    pack_documents(["a"], block_size=1)
except ValueError:
    pass
else:
    raise AssertionError("invalid block size must be rejected")
print("data format toy: ok")
~~~

对这三个固定字符串，toy 词表会给出英文 `5/28≈0.179` token/字符，中文 `4/10=0.4` token/字符，代码 `13/36≈0.361` token/字符；中文 UTF-8 占 30 bytes，因此中文是 `4/30≈0.133` token/byte。它们只反映手工加入 `BASE_PIECES` 的结果，换一组片段就会改变，不是语言间优劣结论。`byte_fallback_roundtrip` 还用未加入词表的 🧪 字符检查字节回退并恢复原文；`assistant_label_count` 与 `prompt_label_count` 分开统计，说明输入可以包含完整对话而监督只落在 assistant 区域；`packed_eos_count` 说明文档结束符被保留；padding mask 来自原始有效长度而非 `token_id != pad_id`；`vocab_resize` 与 `tool_controls_registered` 分别说明新增工具 token 需要扩展词表/embedding，并登记为控制 token。toy 明确拒绝未知角色、空 SFT 监督、非法 token id、空文档和超长文档，因为这些情形需要显式模板或截断策略，不能由默认行为悄悄吞掉。该代码仍不是真实 BPE、生产 tokenizer、完整 attention mask 或模型训练器，读者应把同样的审计思路迁移到目标模型的官方 tokenizer 和数据处理器上。

## 3.17 常见协议故障

### 3.17.1 训练和推理 tokenizer 不一致

这是最严重的输入错误之一。相同 id 在两个词表中可能对应不同 token，模型会收到语义错位的 embedding。应比较 tokenizer 文件 hash、词表大小、special token map 和一组固定文本的 ids。

### 3.17.2 EOS、PAD 和停止条件不一致

训练中的 EOS、padding id、generation config 和服务端 stop condition 需要联动。EOS 漏加会让模型不停生成，PAD 进入 loss 会污染训练，停止字符串只覆盖部分 token 序列则会在流式解码中失效。

### 3.17.3 Label mask 错位

loss 可能下降，但模型学会复述用户、输出角色标签或忽略工具结果。应打印 token、role、mask、label 和 shift 后的目标，至少检查第一轮、最后一轮、空回答、工具调用和截断样本。

### 3.17.4 Packing 边界错误

EOS 只能提示边界，不能自动切断 attention。若任务假设文档独立，必须检查 document mask；若任务允许跨文档上下文，应在训练和评估中保持一致。

### 3.17.5 截断丢失监督

长对话可能把 assistant 回答截掉，得到一条只有 prompt、没有有效 label 的样本。数据审计应记录有效监督 token 数，并拒绝或单独处理 `sum(mask)=0` 的样本。

### 3.17.6 新 token 未被学习

增加 special token 后只更新 tokenizer 配置，模型 embedding 仍是随机或未训练状态。扩词表必须有继续训练数据、初始化记录、checkpoint 结构检查和生成回归。

## 3.18 评估 tokenizer 与数据格式

### 3.18.1 Tokenizer 评估

至少按语言、代码、数学、JSON、长文本和多模态占位符报告：

| 指标 | 作用 |
| --- | --- |
| token/character 或 token/byte | 序列压缩效率 |
| fertility（tokens/word） | 比较词形、词边界与低频词拆分；必须说明分词器如何定义 word |
| 长度分位数 | 长尾和极端输入 |
| 可逆率 | 编码-解码是否恢复内容 |
| unknown/byte fallback | 覆盖和异常输入 |
| special token 命中 | 协议边界是否稳定 |
| 截断比例 | 上下文预算是否足够 |

跨 tokenizer 比语言建模质量时，除每 token perplexity 外，还要给相同规范化原文的 BPB；比较下游能力时，使用相同 held-out 样本，并明确对照的是固定文本字节数、训练 token 数还是 FLOPs。逐语言和逐领域报中位数、p95/p99 与最差桶，避免总体均值掩盖某一语言或代码格式被切得特别碎。token 数是成本和模型输入长度的证据，不是能力公平或语言理解质量本身。

### 3.18.2 数据格式评估

格式评估不仅看 JSON 能否解析，还要检查：

1. role 顺序和允许集合。
2. chat template 是否可重放。
3. assistant label 覆盖率。
4. EOS、PAD 和 document boundary。
5. tool schema、参数和结果对应关系。
6. image/audio/video 占位符与实际输入数量。
7. 截断后是否仍有有效目标。

这些检查应在小样本上打印可读 token 序列，在大规模数据上输出分布和失败样本索引。

### 3.18.3 跨版本回归

改 tokenizer 或 template 后，应固定一组文本和消息，比较：

~~~math
\Delta L_i
=
|T_{\mathrm{new}}(x_i)|-|T_{\mathrm{old}}(x_i)|
~~~

比较前必须保证同一非空原始输入通过两套明确版本的规范化和模板；任一侧编码失败时，`\Delta L_i` 应记录为不可比较，而不是以零替代。还要比较 token ids、special token、assistant 起点、有效 label 数、pack 利用率和模型生成。token 数变少并不自动代表新版本更好；如果角色边界错了，压缩收益没有意义。

### 3.18.4 一套可复查的协议回归样例

升级 tokenizer、chat template、processor 或训练框架时，固定一个小型 gold set，至少包含：

1. Unicode 组合字符、全角字符、连续空格、换行、emoji、中文、英语、代码标识符、数字、JSON 和模型支持的低资源语言。
2. system/user/assistant 多轮消息，assistant prefill，工具调用/工具结果/最终答复，以及用户正文里出现字面 special-token 字符串的情况。
3. padding 与 EOS 同 id 的 batch、空监督样本、回答在边界处截断的样本、跨样本 packing 边界和带媒体时间戳的多模态消息。

逐样本保存原文和规范化结果、token ids 与可见 token、消息/事件边界、attention 与 label mask、shift 后的目标、截断/packing 位置、有效 token 数和解码结果。再抽少数样本用目标模型生成，核对 stop id、工具 handoff 和最终结束标记。测试集里应故意保留负例：未知 role、未闭合 JSON、空回答、错误 tool recipient 和无效媒体尺寸。这样版本差异能定位到 tokenizer、模板、collator、模型或 runtime 的具体环节，而不是只看模型“回答好像变了”。

## 3.19 资料边界与延伸阅读

本章的算法和接口事实来自原始论文、官方 tokenizer 实现和框架文档；具体 token 字符串、id、模板、截断策略和 loss mask 必须以目标模型版本为准。`<system>`、`<user>`、`<assistant>`、`<tool>` 和 `<image>` 不是跨模型统一标准，名字相同也不表示协议相同。

推荐阅读：

- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)：BPE 子词单元的原始论文。
- [SentencePiece](https://arxiv.org/abs/1808.06226)：语言无关的子词 tokenizer 与 Unigram LM。
- [From Bytes to Subwords](https://aclanthology.org/2026.findings-acl.530/)：多语输入表示与规范化边界；语言分类结果不应外推为 LLM 预训练收益。
- [Subword Regularization](https://arxiv.org/abs/1804.10959)：训练期对子词切分采样的动机和神经机器翻译证据。
- [Unpacking Tokenization](https://arxiv.org/abs/2403.06265) 与 [Tokenization Is More Than Compression](https://arxiv.org/abs/2402.18376)：对 token 压缩与下游效果关系给出不同侧面的实验；需同时读实验条件和限制。
- [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425)：多语言切分长度差异及其对成本、时延和上下文预算的影响。
- [Tokenizer Choice for LLM Training](https://aclanthology.org/2024.findings-naacl.247/)：24 个 2.6B 单语/多语模型的 tokenizer 消融；fertility、parity 和训练成本的结果仅适用于其语种与训练设置。
- [Objective vs. Search](https://aclanthology.org/2026.emnlp-main.2307/)、[No Single Tokenizer Feature Reliably Predicts Downstream Language Model Performance](https://aclanthology.org/2026.findings-emnlp.96/) 与 [TokLens](https://aclanthology.org/2026.acl-srw.18/)：分别拆开算法目标/搜索过程、跨模型 intrinsic 指标预测力及多语指标相关性；阅读时要分清受控训练和跨模型观察。
- [MUTANT: A Recipe for Multilingual Tokenizer Design](https://aclanthology.org/2026.acl-long.2146/) 与 [MUTANT 实现仓库](https://github.com/ola-krutrim/MUTANT)：多语语料、预分词和子词/多词训练的工程路线；吞吐数字仅绑定论文 1B 模型和 H100/Triton 测量条件。
- [OpenAI tiktoken](https://github.com/openai/tiktoken)：具体 byte-level BPE 实现。
- [SentencePiece byte fallback options](https://github.com/google/sentencepiece/blob/master/doc/options.md)：Unicode 子词词表缺少字符时的 byte fallback 配置。
- [OpenAI gpt-oss Harmony response format](https://developers.openai.com/cookbook/articles/openai-harmony)：一个将消息结束、工具交接和 assistant 回合完成分别编码的公开协议实例。
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index)：训练、编码、special token 和 offset 接口。
- [Transformers Chat Templates](https://huggingface.co/docs/transformers/main/en/chat_templating)：消息序列化和生成提示模板。
- [TRL SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer)：SFT 数据格式和 assistant/completion-only loss。
- [PyTorch CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)：`ignore_index` 和逐位置损失。
- [NVIDIA NeMo Sequence Packing](https://docs.nvidia.com/nemo-framework/user-guide/25.02/sft_peft/packed_sequence.html)：以累计序列长度传递 packing 边界的一个变长 attention 实现。
- [FOCUS](https://arxiv.org/abs/2305.14481) 与 [2026 vocabulary initialization study](https://arxiv.org/abs/2608.03494)：新词表 embedding 初始化研究；后者是特定 Hindi / 模型设置下的预印本。

这些资料能支持公开算法和框架接口，不能替代当前模型的 tokenizer hash、模板测试、数据授权和目标硬件上的 token 统计。生产结论必须保留配置、版本、固定样本和回归结果。

## 3.20 章节练习：从协议打印到协议回归

### 练习一：比较多语言和代码压缩率

准备中文、英文、Python、JSON、数学公式和 emoji 样本，使用目标 tokenizer 报告 token/byte 的平均值和分位数。解释为什么平均值掩盖了低资源语言和长尾输入。

### 练习二：实现 BPE toy

从几个短句开始统计相邻片段，执行三轮最高频合并，打印每轮词表和切分结果。说明为什么合并规则依赖训练语料，为什么未见词仍需 fallback。

### 练习三：检查 chat template

构造 system、user、assistant、tool call 和 tool result 的多轮消息。打印模板字符串、token id、special token 和生成起点，比较训练模板与推理模板差异。

### 练习四：定位 label mask 错位

对一个多轮 SFT 样本打印 role、token、shift 后 label 和 mask。分别制造 assistant 起点偏移、PAD 未忽略、tool result 被错误监督和回答被截断四种故障，观察 loss 统计如何变化。

### 练习五：比较 packing 边界

把三篇短文 pack 到固定 block，分别使用普通 causal mask 和 document mask。比较第二篇文本能否看到第一篇，说明 EOS 与 attention mask 的差异。

### 练习六：扩展工具 token

给一个 toy vocab 添加 `<tool_call>` 和 `<tool_result>`，同步扩展 embedding 行数，构造一条包含工具调用的 SFT 样本，检查模板、mask、JSON 和停止条件。

### 练习七：设计 tokenizer 版本升级回归

固定一组多语言、代码和多模态消息，比较旧、新 tokenizer 的 token 数、ids、special token、assistant mask、截断和生成结果。列出哪些变化是预期的，哪些变化意味着协议不兼容。

### 练习八：比较“更短”是否真的更好

选两种 tokenizer，在同一份多语言、代码和 JSON 原文上统计 token/byte、fertility、p95/p99、fallback 和 BPB。然后分别设计“固定原文字节数”和“固定训练 FLOPs”的小规模训练对照，说明两种预算下模型看到的原始文本量、序列长度与输出层成本怎样变化。写出你会用什么 held-out 任务判断收益，并解释为什么不能只拿每 token perplexity 或总 token 数下结论。

## 3.21 本章小结

Tokenizer 把字符串映射到模型的离散输入空间，数据格式则定义 token 在训练中的结构和监督含义。BPE、Unigram LM 和 byte-level 方法分别在合并策略、概率切分、覆盖率和序列效率上做取舍；词表大小则在 embedding/输出层成本与序列长度之间交换。

EOS、PAD、角色标记、工具事件和多模态占位符构成模型协议。chat template 把消息变成 token 序列，label mask 决定哪些位置产生监督，packing 和 document mask 决定样本边界，偏好和可验证轨迹需要比普通 SFT 更丰富的 schema。

扩展词表必须同步 tokenizer、embedding、模板、数据和服务端；改动任何协议都要用固定样本做 token、mask、截断、生成和回归检查。最终，Tokenizer 不是字符串切分工具，而是贯穿数据、训练、评估和推理的接口层。
