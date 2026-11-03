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

本章的阅读边界如下。第五册在这里讲清训练样本怎样变成稳定、可回放的模型输入和监督信号；算法选择、数据治理、完整框架项目和故障手册分别接到书系中的主讲位置。

| 读者问题 | 本章负责的内容 | 深入入口 |
| --- | --- | --- |
| BPE、Unigram、字节覆盖如何工作，怎样比较 tokenizer 指标 | 本章保留接口影响和训练预算所需的背景 | [第二册第三章](../../book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md) |
| 数据采集、筛选、去重、配比与治理 | 本章约定进入训练前后的序列和版本字段 | [第九册数据工程](../../book-09-data-engineering/简介.md) |
| SFT、偏好优化、RL 或多模态任务怎样训练 | 本章说明记录结构、标签和输入边界 | [本册第十一章](11-后训练与对齐训练.md)、[第十二章](12-多模态训练.md)、[第十五章](15-领域专家sft与rl.md)、[第十六章](16-rlvr与可验证奖励.md)、[第十七章](17-on-policy-distillation.md)、[第十八章](18-deepseek-v4领域专家与统一蒸馏.md) |
| 如何实现完整微调项目 | 本章给出协议清单和回归样例 | [第三册 Hugging Face 微调实战](../../book-03-practical-handbook/chapters/04-huggingface微调实战.md)、[偏好优化实战](../../book-03-practical-handbook/chapters/05-偏好优化实战.md) |
| 格式错位如何定位和复发预防 | 本章给出契约检查项 | [第十九册 Tokenizer 与格式故障手册](../../book-19-practitioner-playbook/chapters/03-tokenizer与格式坑.md) |
| 图像、音频、视频的任务与表示方法 | 本章只明确训练记录里的媒体边界、占位符和数量账本 | [第十五册多模态专题](../../book-15-multimodal-generative-models/简介.md) |

本章沿着一条训练输入链展开：先固定原文、规范化、模板与 tokenizer 的版本，再把编码结果接到控制符号、监督标签和 packing 边界，最后设计可复查的协议回归。BPE、Unigram 和字节表示只作为理解接口差异的背景；算法推导与 tokenizer 评测指标由第二册第三章完整主讲。末尾的无依赖示例展示一套小型审计器的结构，代码当前只作静态审阅。

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

规范化是否有利还取决于目标任务。[van der Goot 的 ACL 2026 研究](https://aclanthology.org/2026.findings-acl.530/)在 2,307 种语言、每种 1,000 条样本上，以 1—5 字符 n-gram Naive Bayes 比较 NFC 和原文；十个数据划分中不规范化的准确率都更高，均值为 91.75% 对 91.70%。差距很小，但说明组合字符可能携带语言或输入来源信号。这个实验不是 LLM 预训练或通用文本质量测试，不能推出“不要规范化”；应把它理解为一项反例，提醒你依据任务比较 raw / normalized 两种输入，并明确可逆性和对齐成本。

### 3.1.5 把编码过程变成可回放记录

这个原则也适用于行尾。ASCII 文件 `a\r\nb` 使用 CRLF 时占 4 bytes；若预处理把 CRLF 统一为 LF，模型输入变成 `a\nb`，占 3 bytes。规范化后的 tokenizer ids 可以一致，但字符/行偏移已不再直接指向原文件位置。处理代码 span、补丁和文档引用时，要同时保存原始偏移与进入模型的规范化偏移，或保存能在两者之间转换的映射。

对自然语言问答，折叠这类形式可能无害；对代码、引文、签名、文档坐标或字符级标注，原始字节与 span 可能有意义。此时同时保存 raw 样本标识、原文或受控引用、规范化文本、normalizer 版本和两者间的 offset 映射。后续模板、tokenizer 和标签所指的 span 必须引用同一规范化版本；否则一个标在原文第 2 个 code point 的答案片段，可能在 NFC 文本中变成第 1 个 code point，产生错位但不一定报错。

一个可复查样本应按顺序保存：结构化消息、模板渲染结果、规范化文本、token ids、可见 token、特殊事件位置、attention mask、labels、padding/truncation 和 packing 元数据。比如 user 消息经过模板产生 role 头，再由 tokenizer 编码为整数；user token 可以参与 assistant 的上下文 attention，却在 assistant-only SFT 中被设为 ignore label。这个区别让读者看到“模型看见什么”和“模型被要求预测什么”是两份不同的记录。

manifest 的版本字段应能回答输入最终如何产生：原始文本编码、normalizer 和 pre-tokenizer 配置、词表和 merge/model 文件 hash、added tokens、special-token id、chat template、processor、collator、最大长度、padding/truncation 与 loss mask 规则。任何一个会影响 ids 或目标位置的配置变化，都应生成新的 manifest revision；只存模型名称或 JSON 字段名，无法重建相同的训练输入。

再看一个更小的协议变更：若当前 tokenizer 把字符串 `cat` 编成 `[120]`，某个新词表把相同表面 token 编成 `[312]`，这两个 id 只是各自词表中的索引，模型会分别读取 embedding 的第 120 行或第 312 行。即使碰巧都是 id 120，只要两个 tokenizer 的第 120 行代表不同片段，旧 checkpoint 对新输入的解释也会错位。因此 manifest 不只是为排障准备的文档，而是说明权重行与输入符号如何对应的模型资产。

转换链还要区分三个“看起来长度相同”的概念：原文字符/字节长度、tokenizer 输出长度、参与 loss 的有效目标长度。假设一条对话经过模板得到 40 个 token，其中 system/user/context 占 28 个、assistant 答案占 12 个；若是 assistant-only SFT，输入仍有 40 个位置，但有效监督分母是 12（若答案末尾 EOS 也参与训练则按实际目标再计入）。若模板多插入 3 个控制 token，输入变成 43 个位置，答案目标数未必变化。把这三个数量分开记录，才能判断 token 成本变化来自模板、切分还是监督范围。

因此，对每个固定样本至少保留 `input_length`、`valid_target_count` 和 `source_bytes`，并记录它们采用的分母定义。跨版本统计时，若 input token 增加而 valid target 不变，可能只是模板头变长；若 valid target 下降，则可能是 mask、截断或回答字段变化。仅比较 tokenizer 的 `encode(text)` 结果，无法覆盖整条训练数据接口。

数据 manifest 还应区分配置文件的名称和内容身份。仓库名、branch 或模型 alias 可能指向可变文件；hash 可以识别实际 tokenizer、template 和 processor 字节，但不能代替说明其字段意义。因此同时保存稳定 revision/hash、可读参数、预处理软件版本和样本 schema version。训练数据缓存的键也应绑定这些 revision；若只按原始样本 ID 命中缓存，更新模板后可能误用旧 token ids 与旧 labels。

若训练期启用随机子词切分，固定原文和 tokenizer 文件也可能因采样种子产生不同 ids。此时重现记录还要含采样开关、温度/概率参数、epoch 或随机种子；验证、推理和数据审计则应说明采用确定性最优切分还是同一采样策略。可复现不是要求每一处都确定，而是让任何随机变换都有可重建的输入条件。

## 3.2 切分方法在训练协议里的位置

字符、整词、子词和字节表示决定同一段原文需要多少个离散位置。字符或字节粒度覆盖面广，但序列较长；整词表示常见词紧凑，却容易遇到新词和领域词；子词把两者折中，让常见片段可合并、罕见内容仍能拆解。它们的差异最终都会进入训练账本：token 数改变上下文占用、预测目标数、packing 形状和输出层规模。

| 表示路线 | 编码时的基本思路 | 本章关心的训练影响 |
| --- | --- | --- |
| 字符或字节 | 保留较小的输入原子，必要时逐字节回退 | 覆盖稳定；同一文本可能占用较多位置 |
| 整词 | 以词表中的完整词项为主 | 常见词较短；新词、代码标识符和语言变体容易稀疏 |
| 子词 | 把常见邻接片段组合成更大的词项 | 长度与词表成本共同变化，必须按语料桶测量 |

若要推导这些路线的历史动机、语言模型公式和训练算法，见[第二册第三章第 26 讲](../../book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md)。本章后续使用 tokenizer 的实际配置和输出，不把算法名称当作可复现的 tokenizer 定义。

## 3.3 BPE：训练语料会决定合并表

BPE 从较小的符号开始，按训练时的统计结果逐步合并相邻片段；编码阶段再依照保存的合并优先级处理输入。训练侧首先要知道：合并表依赖语料、预切分、规范化、字节映射和并列处理规则。两个模型都声称使用 BPE，不代表它们生成相同 token id。

复现实验时应保存 tokenizer 文件及其 hash、normalizer、pre-tokenizer、词表、merge ranks、added tokens 和 special-token 映射。仅在配置里写“BPE、32K 词表”无法重建编码结果。逐轮 pair 计数和手算合并见[第二册第三章](../../book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md)的 BPE 教学实现。

## 3.4 SentencePiece 与 Unigram LM：工具、模型和采样要分开

SentencePiece 是可承载不同子词算法的训练与编码工具；Unigram LM 则为候选片段建模概率，再从可能切分中选择一条路径。工具名说明了软件体系，算法名说明了切分规则，实际 token 序列仍由词表、归一化和配置共同决定。训练数据与模型 checkpoint 需要引用具体文件和版本，而不是只登记 SentencePiece 或 Unigram。

部分实现允许对子词切分采样。若训练时启用采样，同一原文可能在不同 epoch 生成不同 id 序列；它改变了训练输入分布，需记录采样概率、配置和随机种子，并在验证数据上采用明确的确定性编码策略。Kudo 的子词正则化证据主要来自神经机器翻译，不能直接当作 LLM 预训练的质量保证。Unigram 的概率机制、采样实验和 BPE 对照见[第二册第三章](../../book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md)。

## 3.5 Byte-level BPE 与 byte fallback：覆盖不等于效率

两类设计都能处理常见词表以外的输入，但回退路径不同：

| 设计 | 常见路径 | 罕见字符的处理 | 对训练账本的影响 |
| --- | --- | --- | --- |
| Byte-level BPE | 从 UTF-8 字节原子开始，再按 merge ranks 合并 | 任何输入字节都可拆成 byte token | 覆盖面稳定，但某些字符会占多个预测位置 |
| Unicode 子词加 byte fallback | 常见字符直接走 Unicode 子词词表 | 词表未覆盖时才按 UTF-8 字节回退 | 常见文本可能较短；回退桶需要单独统计 |

因此，“没有 OOV”只说明编码路径存在，不说明模型训练过这个 token、不说明序列便宜，也不保证每个片段都对应完整 Unicode 字符。流式解码时还要按配置缓冲不完整 UTF-8 字节。训练数据统计至少应按语言、代码、JSON 和异常输入分桶，报告 fallback 比例和长度分位数；字节级实现与 byte fallback 的完整算法和解码细节见[第二册第三章](../../book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md)。

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


把 batch 也代入可以看到数量级。设 `B=2`、`d=4,096`。旧配置的输出投影约有 `2×1,000×4,096×32,000=262,144,000,000` 次乘加；新配置约有 `2×600×4,096×128,000=629,145,600,000` 次乘加，比例仍为 `2.4`。若把一次乘和一次加各计一个 FLOP，两者约为 `524.3 GFLOPs` 与 `1.258 TFLOPs`。这个局部账本尚未包含前向之外的反向传播、softmax、并行通信和 padding；共享 embedding 能省一份权重矩阵，却不能把输出分类从 `V=128,000` 个类别变小。比较 tokenizer 时应注明固定的是原文 bytes、训练 token 还是总 FLOPs，否则“预算相同”并没有统一含义。

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

同一组长度数字对不同计算项的影响也不同。`L=1,000` 降到 `600` 时，依赖每个 token 线性处理的投影工作量在理想条件下约剩 `0.6`；若注意力按所有位置对计算，`L^2` 项约剩 `0.6^2=0.36`。但输出分类同时按词表大小 `V` 扩展，前面的教学账本得到输出投影变为 `2.4` 倍。模型在短序列或不同 kernel 下还可能由 MLP、通信、softmax 或 padding 主导，所以三个账本不能合并成一个“token 越少越快”的结论。

assistant-only label mask 也不一定自动缩小 logits 矩阵。许多实现先对完整 `B×L` 序列计算输出，再用 `ignore_index` 去掉不计 loss 的位置；只有训练器确实跳过某些位置的输出计算，才可以把它们从输出投影成本里扣除。查看 shape、logits 生成位置和 loss denominator，能区分“减少监督目标数”和“减少实际计算量”。

做项目规划时至少分开记录：输入 embedding 的参数存储、有效 token/更新数、最大和平均序列长度、注意力/激活账本、输出层 logits 形状、实际 batch padding、词表并行通信及硬件吞吐。这些是不同瓶颈的可观测量。词表增大带来的上下文容量收益可能有价值，但要明确它是在相同 bytes 下扩大覆盖，还是在相同 FLOPs 下改变模型看到的数据量；再用相应的 held-out 结果判断收益是否抵得过新增成本。

### 3.6.6 压缩率重要，但它不是 tokenizer 质量的全部

完整的算法谱系与指标定义见[第二册第三章第 27 讲](../../book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md)。本章需要建立的判断是：压缩率适合描述某种文本在 tokenizer 下占多少位置，却不能单独预测换 tokenizer 后模型能力或训练效率。

| 研究 | 实验覆盖 | 可以支持的判断 | 不能推出的结论 |
| --- | --- | --- | --- |
| [No Single Tokenizer Feature Reliably Predicts Downstream Language Model Performance](https://aclanthology.org/2026.findings-emnlp.96/) | 约 200 个模型、九种语言和多类 benchmark | 没有单一 intrinsic 指标能跨模型家族、任务和语言稳定预测下游结果 | 不能由一个 token/word 或压缩分数直接选出最好的 tokenizer |
| [TokLens](https://aclanthology.org/2026.acl-srw.18/) | 24 个 tokenizer、15 种语言、六类 intrinsic 指标 | 控制模型规模后，英语 benchmark 未见稳定预测关系；其多语 MMLU-ProX 分析发现与逐语言成绩有关联 | 多语关联仍是特定模型/任务的相关性，不能当成因果或跨家族排名 |
| [Equity with Efficiency](https://aclanthology.org/2026.emnlp-main.1783/) | 同一数据上的 1.5B 模型训练，比较 11 种东南亚语言 tokenizer | 在作者设置中，parity-aware BPE 位于效率与公平性的 Pareto 前沿；形态驱动方案在部分语义任务更好但计算成本更高 | 不能外推到其他模型规模、语言集合或生产系统 |
| [To Each Language Its Tokenizer](https://aclanthology.org/2026.findings-emnlp.1019/) | 提出模块化 BPE/Unigram 词表及按 batch 选择相关子词表的训练框架 | 模块化词表是可继续评估的设计候选 | 论文方案不等于可直接部署的通用配方，仍需目标服务栈验证 |

这些结果回答的是不同问题。跨模型分析检验“某个统计量能否先验预测模型得分”；受控训练检验“固定模型和数据时，哪些方案带来目标任务变化”。因此，第一步可以用代表性原文统计 token/byte、p50/p95、fallback 和上下文占用，筛掉覆盖不足或极端膨胀的候选；第二步再在相同原文、同一模型规模和明确预算下评估目标任务、训练吞吐和输出层代价。若 tokenizer 变化后固定 token 数，模型看到的原文字节数可能改变；若固定原文字节，token 数和训练 FLOPs 可能改变。两种对照都可用，但应分别标出问题和资源口径。

跨 tokenizer 比较语言建模结果时，可补充每字节位数（bits per byte，BPB）：

~~~math
\mathrm{BPB}
=
\frac{-\sum_{t=1}^{L}\log_2 p_\theta(z_t\mid z_{1:t-1})}
{|x|_{\mathrm{byte}}}
~~~

分母必须是同一规范化原文的 UTF-8 字节数且大于零，序列是否含 BOS/EOS 也要统一；空文本的 BPB 应记为 N/A。若一方规范化后丢失空格、另一方保留，分子对应的就不是同一文本事件，BPB 不能消除这个协议差异。早期 tokenizer 压缩、词表公平性和算法目标对照的实验条件，可从第二册第三章及本节引用的原论文追溯。

### 3.6.7 从 token 统计走到训练预算

假设一份原文包含 `1,000,000` bytes。旧 tokenizer 的统计值为 `0.5 token/byte`，则得到 `500,000` 个 token；候选 tokenizer 为 `0.4 token/byte`，同一份原文得到 `400,000` 个 token，固定原文字节时 token 数减少 20%。若改为固定 `500,000` token 的训练预算，候选 tokenizer 理论上可表示 `500,000/0.4=1,250,000` bytes，比旧配置多读 25% 的原始字节。两个账本回答不同问题，训练计划应写明采用哪一个。

这项换算还没有包括新词表的 embedding 和输出层参数、不同长度带来的 attention/activation 成本、batch padding 和分布式词表通信。一个方向性的判断可以来自静态成本账本，正式能力结论则要使用匹配的 held-out 原文、记录训练 FLOPs 和更新数，并报告每个重要语言/领域的结果。若候选压缩只来自重复模板或高频单一语域，整体 token/byte 可能下降而目标能力切片退步，因此分桶统计和目标任务回归必须成对查看。

对 tokenizer 的比较可先固定原文和训练配方，报告 `token/byte`、长度分位数、fallback 和输出层成本；再根据项目约束选择固定 bytes 或固定 FLOPs 的受控训练。若要估算“上线每字符更快”，还必须在目标推理设备上测量 decode，并把模型质量和批量条件一起写入结果。训练 token/s 会随 token 含义改变，不能单独充当用户可见吞吐量。

## 3.7 序列化与监督协议：特殊符号、模板和标签

这一组小节回答同一个训练问题：token 序列里哪些位置承载消息结构，哪些位置参与上下文计算，哪些位置成为预测目标。把角色、结束事件、padding 和监督标签分别编码，后续的模板渲染、collator 和服务端停止规则才有一致的输入契约。标签中看似相同的可见字符串也可能映射到不同的 id，因此所有例子都要同时核对文本、token id、mask 与配置版本。

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

### 3.8 Chat template：消息如何变成生成提示

#### 3.8.1 两层转换

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

#### 3.8.2 模板不一致的症状

训练模板和推理模板不同，常见症状包括：

1. 模型把用户问题当成待续写文本。
2. assistant 回答起点偏移，出现重复角色标签。
3. EOS 和停止条件失效。
4. 多轮历史顺序或 system 约束丢失。
5. 工具调用 JSON 外面出现解释文字或格式错误。

因此模板应像模型权重一样版本化。评估和线上服务必须使用与训练数据一致的序列化规则，除非实验明确研究模板迁移。

#### 3.8.3 生成提示和训练标签

推理时通常把 assistant 起始标记放在输入末尾，要求模型从这里开始生成。SFT 时则把完整 assistant 内容放入序列，同时生成 label mask。二者使用同一模板但 mask 不同：推理没有真实 assistant label，训练有。

#### 3.8.4 模板是可执行的数据协议

模板不只是把 role 名称替换成几个字符串。它可能按消息是否为最后一条、是否有工具、content 是文本还是图像块、assistant 属于哪个 channel 等条件分支。推理端的两个常见选项也不能互换：`add_generation_prompt=True` 通常在对话尾部追加一个新的 assistant 起始头；`continue_final_message=True` 则去掉末尾结束标记，让模型接着已有 assistant 前缀生成。Hugging Face Transformers 的文档规定二者不能同时启用，并明确模板是否支持 generation prompt 取决于模型实现。

模板本身已经可能写入 BOS、EOS 和角色边界。若先 `apply_chat_template(..., tokenize=False)` 得到字符串，再由 tokenizer 编码，通常需要 `add_special_tokens=False`，否则第二次编码可能再插一套 BOS/EOS。能直接在模板函数里 `tokenize=True` 时，框架会把生成后的控制标记作为实际 special token 编码，减少重复插入风险。若改模板、模型 revision 或库版本，应保存渲染结果和 token ids 做回归，而不是只检查模板字符串是否“看起来一样”。

### 3.9 Label mask：输入可以完整，监督必须有边界

#### 3.9.1 Assistant-only SFT

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

#### 3.9.2 多轮和工具数据的 mask

如果目标是生成 assistant 回复，历史 user 和 tool result 通常是条件；如果目标是学习生成 tool call，tool call 的函数名和参数应成为监督区域；如果目标是预测工具结果，则需要明确结果是否来自可信环境，不能把未验证的字符串当作标准答案。

对偏好训练，chosen/rejected 都应在相同 prompt 和模板下计算回答 token 的对数概率。对 GRPO、RLVR 等轨迹数据，还可能需要组内候选、奖励、验证器和环境状态，不能用单一 assistant mask 表示全部监督。

#### 3.9.3 Label 在模型里还要右移一位

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

### 3.10.4 用一组文档复算 packing 与边界成本

设文档 A 的序列为 `[BOS_A,a_1,a_2,EOS_A]`，文档 B 为 `[BOS_B,b_1,b_2,EOS_B]`，各有 4 个 token。若训练器按长度 8 为每篇文档单独建立一个 block，会为两篇共分配 16 个位置，其中 8 个是 padding，packing 利用率是 `8/16=50%`。把两篇放入同一个长度 8 的 block 后，8 个位置都有效，利用率是 `8/8=100%`。这是固定 block 的教学账本；真实批次还要把样本长度、attention 实现和负载均衡一起算入。

利用率提升没有自动决定文档之间能否互相读取。普通 causal attention 下，`b_1` 的隐藏状态可以读取 A 的全部前缀，即使 `EOS_A` 和 `BOS_B` 都存在。若目标要求两篇独立，attention 还要使用 block-causal 边界；若 kernel 能跳过跨文档计算，两个长度 4 的因果块各处理 `4×5/2=10` 个可见位置对，共 20 对，而把它们当成一条长度 8 的流则有 `8×9/2=36` 对。实际内核可能使用不同的计数和填充策略，这个手算只展示独立注意力的理论关系。

同一边界还必须出现在三个位置：attention mask 决定能读取哪些 token，position id 决定序列位置从哪里计起，label mask 决定跨文档的 next-token 目标是否参与损失。若 `BOS_B` 是 B 的合法起点，应由它预测 `b_1`；若 B 没有 BOS 且训练目标要求文档独立，就要屏蔽由 `a_2` 预测 `b_1` 的跨界目标。三个实现有一个漏掉边界，loss 仍可能正常下降，但学习到的条件分布已改变。

截断也要按任务处理。长 SFT 样本如果从末尾切掉回答，可能只剩条件文本；若从开头切掉 system 或问题，回答就失去正确上下文。可以统计截断前后的有效监督 token、保留下来的 role、被切断字段和样本数量，并拒绝 `sum(mask)=0` 的样本。因而一个训练 block 的审计至少同时记录文档起止、EOS/BOS、position ids、有效 label、padding 和截断计数；单一的高利用率数值不能证明样本协议正确。

这些边界还可以用同一批样本做一次数量复算。设每个独立文档有 `BOS`、两个内容 token 和 `EOS`，每篇共 4 个 token。若分别装入长度 8 的固定 block，两个 block 共 16 个位置，其中 8 个是 padding；合并到一个长度 8 的 block 后，padding 数为 0。若每篇不训练 `BOS`，两篇共有 6 个有效预测目标；合并前后 loss 的分母都应是 6，而不是分别按 16 或 8 个物理位置平均。这样 padding 优化改变矩阵形状，却不会悄悄改变目标 token 的权重。

文档独立时，稀疏实现只需计算两个长度 4 的因果块，各自有 `1+2+3+4=10` 个可见位置对，合计 20；当作一条长度 8 的普通序列，则是 `1+2+...+8=36` 个位置对，而且 B 能读取 A。只有支持变长或 block mask 的 kernel 才可能跳过多余的跨文档计算；将 mask 写成稀疏矩阵但内核仍做稠密乘法，不一定省下这部分时间。这个手算从样本长度推到 attention 计算边界，也提醒维护者把数据 layout 与 kernel 能力一起记录。

不同训练目标可以采用不同边界政策：预训练语料可能有意让后续文本承接前文；SFT packing 通常要保持每个对话样本的 role 和回答 mask；工具/RL 轨迹还要保留一次行动和观察的因果次序。数据是否跨样本可见，应是可说明的训练假设，而不是由 dataloader 的拼接结果偶然决定。

数据集划分也要发生在 packing 之前。若先把同一长文切成多个 block，再随机分配 block，前后片段可能分别进入训练和验证，后一个片段通过训练样本中的上下文泄漏文档内容。应先按文档或来源实体完成 train/validation/test 隔离，再在每个 split 内做去重、截断和 packing；记录 document id、切分规则和 pack 版本，后续重建才能避免跨集合串接。

packing 的优化目标不只是提高 `η_pack`。长度分布和 padding 方向会影响每步 token 数，变长 kernel 需要正确的 cumulative sequence lengths，分布式训练还要确保各 rank 的有效目标分母一致。可以同时报告有效 token/物理位置、每 block 文档数、跨文档 attention 规则、截断数和每步 target count；若利用率提高但长回答被截断、边界标签变了，吞吐收益并没有保住原训练目标。

例如，一个 1,140-token 的对话在上下文上限 `C=1,024` 时超长 116 个 token。若策略从尾部截断，假设前缀占 240、回答占 900，剩余容量只能保留 784 个回答 token，回答末尾和 EOS 一起丢失；若从头部截断，可能先失去系统指令或问题。真正要检查的是截断后保留的条件字段、监督 token 数和终止目标，而不是只统计“减少了 116 token”。对长文预训练也要记录 chunk overlap：重叠窗口会重复训练窗口边缘的 token，改变它们的有效曝光。

每次生成训练/验证切分后，应先冻结文档集合与其版本，再分别编码和 packing。对评估集，若采用滑窗，应记录窗口长、stride 和哪些 token 计入 loss，避免相邻窗口在聚合时重复计分却按唯一 token 数作分母。训练侧与评估侧可以使用不同的 attention 策略，但差异必须是显式配置，并在比较结果时按同一协议解释。

## 3.11 训练任务的数据格式：JSON 只是容器

本节把常见训练任务看成不同的数据协议，而不是同一种 JSON 记录换字段名。每条记录至少要能追溯输入来源、消息/事件顺序、目标字段、参与 loss 的 token 范围、结束条件、可选媒体载荷和处理版本。这样同一组数据才能分别送入训练器、评估脚本和服务端时保持含义一致。

后面的 SFT、偏好/RL、工具轨迹和多模态样本有不同监督对象：SFT 监督回答 token，偏好数据保存候选与比较关系，RL 轨迹还需要 rollout 状态、奖励和验证器来源，工具记录要区分模型发出的调用和环境返回，多模态记录则要关联文本占位符和真实载荷。这里主讲字段、边界与可检查状态；数据采集和治理方法见[第九册](../../book-09-data-engineering/简介.md)，后训练算法分别见[本册第十一章](11-后训练与对齐训练.md)及[第十五至十八章](15-领域专家sft与rl.md)，实现项目见[第三册](../../book-03-practical-handbook/简介.md)。


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

### 3.12 偏好、强化和可验证数据格式

#### 3.12.1 Pairwise preference

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

#### 3.12.2 组内候选和验证器

GRPO、RLVR 或其他强化训练可能保存一个 prompt 的多个候选、组内 reward、验证器输出、环境状态和工具轨迹。可把一组数据抽象为：

~~~math
G_i=\left(x_i,\{y_{i,j},r_{i,j},v_{i,j}\}_{j=1}^{n_i}\right)
~~~

`y` 是候选，`r` 是奖励或相对分数，`v` 是验证器证据。奖励来自字符串匹配、代码执行、数学检查还是人工偏好，会影响训练信号的含义，必须记录来源。

一次 rollout 的审计记录通常还需要行为策略的模型 revision、采样参数、生成 token 的旧 log-probability、`group_id`、response/loss mask、终止原因、reward 分项、verifier 版本和环境事件。离线偏好对没有这些在线采样字段；来自在线 GRPO/RLVR 的轨迹则不能只压扁成 chosen/rejected 文本，否则很难复核样本由哪个策略产生、工具结果是否可信、奖励是否过期。不同算法的训练器 schema 不统一，这里列的是复现与追踪所需的账本字段，不是某个库的固定 API。

#### 3.12.3 长度和偏好偏差

如果 chosen 几乎总比 rejected 长，偏好优化可能学习冗长；如果安全回答总是拒绝，模型可能过度拒答；如果 judge 对格式有偏好，模型可能优化表面结构。偏好目标常用整段回答的 log-probability 或其比值，回答 token 数和 tokenizer 切分都会影响聚合尺度，所以不能在更换 tokenizer 后沿用旧 token-level 长度统计或未经校准的 loss。偏好数据要做长度、语言、领域、风险等级和错误类型的分层统计，并固定是否按 token、按回答或按 pair 平均。

### 3.13 Tool calling：数据格式连接模型和外部副作用

#### 3.13.1 一条完整轨迹

一个工具轨迹可能是：

~~~text
<user> 查询北京天气
<assistant_tool_call> {"name":"weather","arguments":{"city":"北京"}}
<tool_result> {"temperature":"20C","condition":"晴"}
<assistant> 北京今天晴，约 20C。
~~~

训练数据要明确哪个阶段由模型生成，哪个阶段由真实工具返回，哪些字段被验证。只把 tool result 当作普通文本，会让模型学习伪造工具结果或混淆角色。

结构化轨迹至少要把三个事件分开：assistant 产生带 recipient 的调用与参数；运行时校验后由工具执行并以工具名作为来源返回结果；assistant 读取结果后再产生最终消息。OpenAI gpt-oss 的 Harmony 示例把 recipient 放在 assistant header、JSON 约束放在格式元数据中，并以专用 call 标记交接；这展示了“调用内容”和“调度控制”如何同时编码。它只是一个已发布协议实例，训练别的模型必须按该模型 template 重建边界。

#### 3.13.2 Schema 和边界检查

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

#### 3.13.3 训练与运行时的边界

训练可以教模型生成合法参数，不能保证工具授权正确，也不能让一个有副作用的 API 自动变成安全动作。生产系统仍需在执行前校验 schema、权限和用户确认，并记录调用 trace。

对每条轨迹，至少保存调用样本的原始输入、模型生成的 recipient / arguments、handoff 位置、tool schema revision、执行结果来源、调用状态（成功/错误/超时/取消）、环境时间和 assistant 后续回答。通常模型生成的工具选择、参数和交接标记参与 assistant loss；工具观察结果是否参与 loss 由训练目标决定。若训练数据伪造一个成功结果、却没有环境 trace，模型会学到的是一段“看起来像工具轨迹”的文本，而不是经验证的工具行为。

### 3.14 多模态 token 与占位符

#### 3.14.1 `<image>` 不一定是一个视觉 token

多模态输入常见结构为：

~~~text
<user> <image> 描述图片
<assistant> 图片中有一只猫。
~~~

`<image>` 可能是一个文本侧占位符，真实视觉信息由 vision encoder、projector、resampler 或 cross-attention 提供。它也可能展开成多个 patch embedding。不能只数字符串中有几个 `<image>` 就推断视觉序列长度。

#### 3.14.2 粗略 token 账本

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

#### 3.14.3 视觉、音频与文本的边界

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

随机初始化、旧子词均值或外部向量映射都只是新 embedding 行的起点。假设新词由两个旧子词表示，旧向量分别为 `e_1=(2,0)`、`e_2=(0,4)`，简单均值初始化得到 `e_new=(1,2)`；这个计算可复现，但它只说明如何生成初值，不能证明新 token 已经具有正确语义。只有继续训练数据实际出现该 token，并且输入 embedding、输出头、模板、labels 和 optimizer 都按新词表版本连接，模型才有机会学习它的读写作用。

扩词表的内存变化也应先算清。设新增 `K=2,000` 个 token，hidden size `d=4,096`，每张矩阵要增加 `K×d=8,192,000` 个参数。若参数用 BF16 存储，单个新增矩阵约占 `16,384,000` bytes，即 `15.625 MiB`；输入和不共享的输出矩阵合计约 `31.25 MiB`，尚未计入梯度、优化器状态、分片填充和 checkpoint 元数据。这个例子帮助判断词表扩张的直接参数账本；真实训练峰值还取决于 optimizer、并行布局和权重共享方式。

初始化方法需要按目标模型做短程对照。FOCUS 在 XLM-R 单语种适配中用辅助 fastText 空间，从源/目标共享词汇中寻找语义相近向量，再组合原模型 embedding；它不是面向现代生成式 LLM 的普遍配方。[Beyond Initialization Loss](https://arxiv.org/html/2608.03494v1) 在 Nemotron-3-Nano-30B-A3B 的 Hindi 扩展中加入 25,600 个 token，并在 100,000 篇 Sangraha Hindi 文档上报告 fertility 从 1.95 降至 1.25。该研究比较 20 多种初始化后发现，在其 Hindi 设置中，50-step CPT probe 比 step-zero 初始化 loss 更能作为选择信号；最佳方案的输入和输出矩阵使用了不同子词组合。模型、语种、数据和预算均有限，结果应转成待验证的实验设计，而非默认初始化规则。

若仍能控制源 tokenizer 的 BPE 合并表，可评估沿用旧 merge ranks 并追加新合并的路线。arXiv v1 的 LFM2 案例在保留旧 token id 映射后，为新 token 做子词均值初始化，先进行 embedding-only 适配，再对全模型做多语言继续预训练；作者报告 65K 扩至 128K 后，Hindi、Vietnamese、Thai 的 token 数分别约减少 2.4、2.6、4.0 倍。其 2.2–3.7 倍每字符解码速度是压缩比例与参考设备吞吐合成的估计；同一报告还测得更大词表在部分设备上的每 token 解码下降，并指出英语、代码等压缩较少的切片可能出现每字符回退。报告只覆盖 LFM2 家族、指定设备和继续训练方案，不能证明独立替换第三方 tokenizer 同样有效。[方法、实验条件与限制](https://arxiv.org/html/2607.15232)

因此，扩词表应走完一条可回滚的状态链：冻结源 tokenizer 与旧 id 映射；统计新 token 的语料频次和目标域覆盖；生成新矩阵并记录初始化版本；在 embedding-only 和全模型阶段分别记录训练数据、token 预算和 checkpoint；重编码代表样本并重算 packing；最后比较目标语言质量、原支持语言回归、总训练成本和目标设备的每字符延迟。若训练数据根本未生成新增 token，或输出头缺少对应行，配置表里的新 vocab size 并不代表模型已经学会扩展后的协议。

### 3.15.3 旧数据和新词表

旧数据如果保存的是 token id，扩词表或重排词表后不能直接复用；如果保存的是原始文本，可以在新 tokenizer 下重新编码，但要重新计算 token 数、packing、训练步数和评估基线。数据版本应记录 tokenizer hash，避免同一个字段名下混入不同编码空间。

### 3.15.4 词表扩展需要模型状态迁移和回归

以报告中的约数 `65K→128K`、`d=4,096` 作为教学账本，约增加 `63,000×4,096=258,048,000` 个 embedding 参数。若按 BF16 的 2 bytes/parameter 计，输入矩阵新增约 `516,096,000` bytes，即 `492.2 MiB`；不共享的输出矩阵还会增加约同样大小。实际 checkpoint/训练峰值还需计入量化格式、梯度、优化器状态、分片和对齐开销。这个例子说明词表新增行在大模型整体中可能不占多数，却足以影响设备内存、文件布局和词表并行。

扩词表会建立一套新的模型状态：token id 到 embedding 行的映射，输入/输出矩阵形状，可能的 tied-weight 配置，optimizer state，模型配置、tokenizer 文件和分片 checkpoint 必须互相吻合。若直接用新矩阵加载旧 optimizer state，行数和状态槽位可能不匹配；若只更新 tokenizer，新增 id 会指向未训练的向量。迁移说明中应写清哪些旧行逐项复制、哪些新行重建、优化器哪些状态清零或扩展，以及恢复 checkpoint 的边界。

验收时选两组输入：旧模型已支持的控制样本和新目标语言/领域样本。逐样本比较旧 id 映射是否按设计保留、新 token 是否在预期语料出现、解码是否恢复目标文本、packing 与 label 数是否变化；再比较新领域效果、旧语言能力和目标硬件上的端到端成本。只看 fertility 下降可能漏掉旧语种退化，只看初始化 loss 又不能代表短程训练后的收敛；扩词表要同时通过结构检查和目标任务回归。

若模型方掌握原 tokenizer，可以选择沿原 merge 规则扩展；若只有第三方 tokenizer 文件，id、规范化、special token 和 embedding 对应关系可能无法无损恢复，应把它作为模型适配项目单独评估。无论哪条路线，都要保留可恢复的旧 checkpoint、可定位的源样本和版本 manifest，使失败后可以回到明确状态。

扩大词表也要重新审视控制 token 的注册方式。若新增的 `<tool_call>` 只是普通字符串，它可能被拆成多个常规子词；只有目标 tokenizer 将其登记为独立 special token，且模板、collator、训练 labels 和运行时 parser 都接受该事件，模型才能按控制符解释它。应保存每个新增 token 的字符串、id、属性、匹配规则、decode 行为和停止语义，并检查普通用户文本里出现相同字面串时是否需要转义或隔离。

检查矩阵维度时也不能只看 `vocab_size` 字段。对每个有效 token id，都要确认输入 embedding 与输出头有对应行；若权重共享，检查共享参数的行数和配置；若有非连续 added-token id，则以最大有效 id 和模型具体实现验证索引范围。恢复 checkpoint 时还要确认 tokenizer/config revision 与参数分片共同来自同一扩展阶段，避免新 tokenizer 搭配旧矩阵或旧 optimizer 状态。

新增 special token 后应做一条端到端样本回放：模板产生可识别的控制事件，tokenizer 输出预期 id，输入 embedding 与输出头均能索引该 id，目标 labels 在需要模型生成的位置监督它，模型生成后 parser 能还原同一事件。比如训练了 `<tool_call>` 却把它从 assistant labels 中屏蔽，模型可能只会识别而不会发出调用；若模型能生成但服务端没有注册对应 parser，调用仍无法执行。这个检查把新符号从“词表条目”连到“可用模型行为”。

## 3.16 一个最小的数据格式审计器

下面的 toy tokenizer 不复刻真实 BPE，只用有限片段加固定的 256 个 UTF-8 byte fallback token，帮助读者观察不同文本的 token/byte、可逆性、chat template 的角色边界、assistant-only label mask、预训练 packing 和扩词表后的 embedding 行数。它按最长片段匹配，完全没有训练 merge ranks；packing 只返回有效长度 mask，没有构造 causal/document attention mask。真实系统还要处理完整 Unicode 规范化、offset mapping、批量 padding、processor 和模型特定模板。

> **代码状态：静态审阅，未运行（2026-10-30）。** 本轮检查代码结构和样例边界，没有可复核的历史运行命令与完整输出记录；示例不模拟真实 tokenizer、attention 或模型训练。

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

代码的阅读顺序可以沿着同一条数据流进行。先看 `SPECIAL_TOKENS` 与 `BASE_PIECES` 如何建立词表和反向映射，再看 `tokenize_text` 如何执行最长片段匹配、未命中时如何把单个字符转换成 UTF-8 byte token；之后追踪 `serialize_chat` 如何加入 role 边界并生成 assistant label mask，最后对照 `pack_documents` 的有效长度、EOS 数和 padding 结果。这样读者可以从一个字符串一路跟到 id、训练目标和 block，而不是把几百行代码当成互不相干的工具函数。

手算时先挑三个输入：词表中已有的 `get_user_id`、未在词表中的 emoji、以及一条只含 user prompt 的消息。第一项应按最长已有片段编码；第二项会按 UTF-8 字节展开并在 byte fallback 解码时恢复；第三项没有 assistant 目标，serializer 应拒绝空监督样本。代码中的 `ValueError` 和断言分别对应这些协议门禁。它们检查的是这个固定教学词表和输入，不验证真实模型的 tokenizer、训练框架或线上停止行为。

这个实现用 Python 循环逐个位置扫描所有基础片段，目的是让匹配顺序可以直接阅读，不追求生产吞吐。真实 BPE 有 merge ranks、预切分和高效查表；Unicode 规范化和 offset mapping、processor 多模态载荷、block-causal attention 与优化器状态也不在本例中。迁移时应保留“固定输入—可见 token—id—mask—pack—预期错误”的审计顺序，再调用目标模型和框架的真实实现，不能把 toy 的输出比率当作某种语言固有的 token 效率。

从表面 token 追踪到数字时，可以先用 `tokenizer` 这个已登记片段做参照：toy 词表中它被最长匹配为一个片段，因此 token 数为 1；未加入 `BASE_PIECES` 的 `🧪` 在 UTF-8 中由 `F0 9F A7 AA` 四个字节组成，byte fallback 会产生四个 byte token。对于这两个示例，toy 的计数分别是 1 和 4；四个 byte id 解码后可以拼回同一 Unicode 字符。这个结果只由人为加入的片段表决定，不能解释为英文天然比 emoji 更高效，也不能代表真实模型的 tokenizer。

读者可复制一个已有词片段到 `BASE_PIECES` 并改变其先后位置，手推最长匹配如何改变 token 数；随后把一个未登记符号换成两字节、三字节或四字节字符，核对回退序列长度。最后分别检查 `tokenize_text` 的文本片段列表与 `encode_pieces` 的整数列表，避免把可见字符串、byte token 名和模型 id 混作同一个对象。

## 3.17 常见协议故障

第十九册第三章提供按症状、日志和最小复现推进的格式故障手册；本节把常见信号映射到本章的训练协议对象，方便先确定检查点。

### 3.17.1 训练和推理 tokenizer 不一致

比较 tokenizer 与 processor 文件 hash、词表大小、special-token map，并重放固定文本；同一个数字 id 在两种词表里可能代表不同 embedding 行。

### 3.17.2 EOS、PAD 和停止条件不一致

并排核对模板产生的结束 id、参与训练的 labels、padding mask 和生成配置；输出过早停止或一直延伸都要回到这些字段检查。

### 3.17.3 Label mask 错位

按 token、role、mask、shift 后目标逐位置打印；空监督、回答首 token 漏训和错误训练用户内容都应被固定负例捕获。

### 3.17.4 Packing 边界错误

EOS 表示序列里写入了边界符号，不能替代 document attention mask；核对 attention、position id 和跨界 label 是否共享同一切分点。

### 3.17.5 截断丢失监督

记录截断前后有效 assistant label 数，拒绝没有有效目标的样本，并按任务保留必要的 system、问题、工具结果和回答字段。

### 3.17.6 新 token 未被学习

同时核对新增 id、embedding/输出矩阵行数、实际出现频次和继续训练数据；tokenizer 能编码新 token 不等于 checkpoint 已经学会它。

## 3.18 评估 tokenizer 与数据格式

协议评估的对象是整条转换链，而不是最后的 JSON 字符串。对一个固定样本，维护者要能从原文重建规范化结果、消息模板、token id、mask、labels、packing 位置和实际停止事件；配置更新后还要说明哪些差异是预期行为、哪些会改变模型看到的目标。对 tokenizer 算法指标的体系评估见[第二册第三章](../../book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md)，这里重点验收训练数据能否按同一版本重放。

### 3.18.1 Tokenizer 评估

快速统计用于发现异常输入桶，不用于直接排名模型质量。至少按语言、代码、数学、JSON、长文本和多模态占位符分组，计算 token/byte、长度中位数与 P95/P99、fallback 比例、空输出/编码失败率、规范化前后可逆性和截断率。若用 fertility（tokens/word），必须说明分词器怎样定义 word；跨语言时应另外保留 bytes 或字符口径，因为中文、日文和泰语不能直接沿用英文空格词数。

当一种 tokenizer 让目标语料缩短时，继续检查词表参数与输出投影、padding 和 batch 形状、实际训练吞吐以及 held-out 任务结果。模型组的多语言指标相关性不能替代目标数据的受控训练；BPB 可以补充跨 tokenization 的字节级比较，但要求规范化文本、计分目标和字节分母一致。更完整的 tokenizer 质量指标、相关研究对照和例子由第二册承担。

### 3.18.2 数据格式评估

数据字段检查要从解析一路覆盖到损失函数和推理回放。role 只决定消息身份，template 决定如何序列化，tokenizer 决定 id，collator 决定 batch 中的 padding/packing，目标函数再决定哪些 logits 与 labels 配对。JSON 能解析只说明容器语法成立，不证明角色顺序、工具调用或监督范围正确。

固定样本应检查允许的 role 与 channel、生成起点、BOS/EOS 是否重复、工具 recipient 和参数是否符合 schema、assistant 有效 label 数、媒体载荷与占位符是否一一对应、截断后必需字段是否保留。批量报告还要包含每类错误数量和可定位的样本 ID，使维护者能区分字段缺失、模板版本变化与 tokenizer 编码差异。

### 3.18.3 跨版本回归

升级 tokenizer、chat template、processor、collator 或训练框架时，先固定原始样本和旧版本输入，再分别运行新旧转换链。token 数差异可以写成：

~~~math
\Delta L_i=|T_{\mathrm{new}}(x_i)|-|T_{\mathrm{old}}(x_i)|,
\qquad
r_i=\frac{\Delta L_i}{|T_{\mathrm{old}}(x_i)|}
~~~

其中 `r_i` 仅在旧版本 token 长度大于零时定义；空输出应单独报告为 N/A 或编码失败。比如某条相同文本由 100 个 token 变成 85 个，`ΔL=-15` 且相对长度变化 `r=-15%`。这是长度差异，不代表质量提升；只有两边都成功编码同一非空原文且规范化口径一致时才能计算。若 tokenizer 版本改变了 Unicode 处理，应同时保存 raw 与 normalized 文本，并把不可直接比较的项标为失败或 N/A，不能用零填补。

除 token 长度外，还要逐条比较可见 token 与 id、special-token 位置、assistant 生成起点、labels 右移后的目标、有效监督 token 数、EOS/PAD、position id、pack 边界和解码文本。对工具和多模态样本还要核对调用事件、载荷尺寸/数量和 dispatcher 收到的结构。差异必须按字段给出允许范围；不能因为模型最后一段回答“看起来正常”就跳过前面的接口偏差。

### 3.18.4 一套可复查的协议回归样例

Gold set 需要覆盖输入表示、消息控制和 batch 边界。文本样本包括 NFC/NFD、全角字符、连续空格、换行、emoji、中英混排、代码标识符、数字、JSON 和低资源语言；消息样本包括 system/user/assistant 多轮、assistant prefill、工具调用/工具结果/最终答复，以及正文中出现字面 special-token 字符串的情况；batch 样本包括 PAD 与 EOS 共用 id、空监督、回答截断、跨样本 packing 和带时间戳的多模态消息。

为每条样本保存来源或合成标识、raw text、normalized text、模板版本、tokenizer/processor hash、token ids、attention mask、labels、shift 后目标、截断/packing 位置、媒体载荷引用和解码结果。敏感或授权受限的原始数据不应复制进回归集；可保存脱敏样本与稳定的内部样本键。选少量公开或合成记录通过目标模型生成，复核 eos stop、工具 handoff 和最终结束标记；模型输出是端到端信号，不能代替静态字段断言。

### 3.18.5 版本通过条件

发布前把每个变更分为三类：预期编码变化、需适配的格式变化、阻断的监督或接口变化。预期变化要列出样本范围和新的 token/byte 分布；格式变化须同时修改训练和推理使用的模板/processor 并保留旧版本回放；角色丢失、有效 label 变零、EOS 停止不一致、旧 id 错映射、媒体载荷断链或未授权数据混入都应阻止切换。若变化目标仅是压缩，评估必须同时报告目标任务、原有语言/代码回归和成本账本；若目标是修复协议，则至少证明固定正例的转换结果稳定、负例被拒绝或正确隔离。这样回归集既能给新协议留出变化空间，也能防止静默改写训练目标。

回归结果还应按语料桶和字段维度解释。比如一轮模板更新让平均输入长度只增加 1%，但 assistant label 的 P95 长度增加 18%，可能会改变长回答在 batch 中的权重和截断概率；若只比较全局 token 总数，这种变化会被大量短样本稀释。版本报告应同时给样本数、分母、p50/p95/p99、空监督比例、截断率和 token/byte，而不是仅给一个均值。

对每项差异指定负责人和处理决定：与目标功能一致且通过质量回归的编码差异可以接受；需要同步服务端或训练器的 schema 变化应先做灰度和回滚准备；角色丢失、有效 labels 归零、旧 token 错映射、工具调用内容被误解析或媒体载荷错配属于阻断问题。只有把“预期变化”和“训练目标改变”分开记录，评审者才知道一项 token 变短是否值得承担新 checkpoint 与数据重编码成本。

回归流程还应把版本依赖的产物分清。改 normalizer 时，重点比较 raw/normalized text、offset 和覆盖的 Unicode 样本；改 tokenizer 文件时，检查 ids、词表边界、embedding 行和 token/byte 分布；改 chat template 时，检查角色事件、assistant 起点、labels 和 stop id；改 processor 时，检查媒体载荷与占位符；改 collator 时，检查 padding、packing、position id 和分母。一个总版本号可以绑定这些组件，但每个子组件都应能单独识别，才容易定位差异来源。

差异记录至少要包含固定样本键、旧/新组件 revision、预期变化、观测结果、影响桶、处理结论和回滚指针。exact-match 适用于声明不应变化的 id/格式；统计容差适用于允许分布波动的长度指标；模型任务质量则需要成对样本与明确的计算预算。选择哪种门槛取决于组件契约，不能用全局 token 总量替代所有检查。

若多个组件同时更新，最好先单独记录每项期望的输入输出差异，避免把 tokenizer、模板和 collator 的效应混在同一批模型输出里。无法分批发布时，至少在回归清单中标出变更组合，并保留组件级快照，让旧版链路可以独立重放。这样出现回归时，维护者可以先比较规范化文本，再比较模板事件、ids、labels 和 runtime 结果，而不是从最终回答猜测故障发生在哪一层。

## 3.19 资料边界与延伸阅读

本章固定的是训练数据接口和版本边界。具体 token 字符串、id、规范化、chat template、截断策略、processor 和 loss mask 必须以目标模型及软件版本为准；`<system>`、`<user>`、`<assistant>`、`<tool>` 和 `<image>` 不是跨模型统一标准。

算法和 tokenizer intrinsic evaluation 的系统讲解见[第二册第三章](../../book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md)。本章使用的关键研究入口包括：[No Single Tokenizer Feature Reliably Predicts Downstream Language Model Performance](https://aclanthology.org/2026.findings-emnlp.96/)、[TokLens](https://aclanthology.org/2026.acl-srw.18/)、[Equity with Efficiency](https://aclanthology.org/2026.emnlp-main.1783/) 与 [To Each Language Its Tokenizer](https://aclanthology.org/2026.findings-emnlp.1019/)，分别提供跨模型预测、特定多语相关性、受控效率/公平比较和模块化词表候选；结论须按各论文的模型、语种和任务边界使用。

词表扩展的研究入口包括 [FOCUS](https://arxiv.org/abs/2305.14481)、[Beyond Initialization Loss](https://arxiv.org/html/2608.03494v1) 和 [In-Place Tokenizer Expansion](https://arxiv.org/html/2607.15232)。后两项是指定模型与语种上的 2026 研究；每字符延迟合成、初始化排名和继续预训练步数需保留来源配置，不能写成通用收益。Unicode 与 byte 表示的边界可读 [From Bytes to Subwords](https://aclanthology.org/2026.findings-acl.530/)，其语言识别观察不等价于 LLM 预训练效果。

实现与格式资料包括 [OpenAI Harmony 格式说明](https://developers.openai.com/cookbook/articles/openai-harmony)、[Transformers Chat Templates](https://huggingface.co/docs/transformers/main/en/chat_templating)、[TRL SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer)、[PyTorch CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) 和 [NeMo Sequence Packing](https://docs.nvidia.com/nemo-framework/user-guide/25.02/sft_peft/packed_sequence.html)。它们分别支持公开协议或特定版本的接口说明；目标模型行为仍需用固定样本复核。需要故障定位顺序时转到[第十九册第三章](../../book-19-practitioner-playbook/chapters/03-tokenizer与格式坑.md)，需要完整项目实现时转到[第三册微调实战](../../book-03-practical-handbook/简介.md)。

## 3.20 章节练习：从协议打印到协议回归

### 练习一：为训练建立 token 预算

准备中文、英文、Python、JSON、数学公式、emoji 和一种低资源语言样本，固定 tokenizer 文件和原文版本，分别统计 token/byte、中位长度、P95/P99、fallback 和截断率。然后按有效监督 token 重新估算训练 block 数、输出投影账本和目标上下文容量，并说明为什么总体均值不能代表所有语料桶。

### 练习二：做一份 tokenizer manifest

选一组含 NFC/NFD、空格、代码和 special-token 字面文本的固定输入，记录 tokenizer 文件 hash、normalizer、pre-tokenizer、词表与 merge 表版本、special-token id、模板版本和编码输出。再改变其中一个配置，指出哪些 token id、offset 或解码结果会变，以及怎样让旧数据可以重建。BPE 逐轮合并的手算练习见[第二册第三章](../../book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md)。

### 练习三：检查 chat template

构造 system、user、assistant、tool call 和 tool result 多轮消息，打印模板字符串、token id、special token 和生成起点。分别启用 generation prompt 和 assistant prefill，解释训练和推理的序列差异及各自的有效 labels。

### 练习四：定位 label mask 错位

对一个多轮 SFT 样本逐位置打印 role、token、shift 后 label 和 mask。分别制造 assistant 起点偏移、PAD 未忽略、tool result 被错误监督和回答被截断四种故障，计算有效监督 token 分母并解释 loss 为什么仍可能下降。

### 练习五：比较 packing 边界

把三篇短文 pack 到固定 block，分别使用普通 causal mask 和 document mask。手算 padding 利用率及可见上下文，检查第二篇是否能读取第一篇、边界处的 next-token label 是否有效，并说明 EOS 本身承担不了哪一类隔离。

### 练习六：扩展工具 token

给 toy vocab 添加 `<tool_call>` 和 `<tool_result>`，同步扩展 embedding 行数，构造一条包含工具调用的 SFT 样本。检查工具名、参数、结果和最终回答的事件顺序、label mask、JSON 结构及停止条件。

### 练习七：设计 tokenizer 版本升级回归

固定一组多语言、代码、工具和多模态消息，比较旧、新版本的 token 数、ids、special token、assistant mask、截断/packing 结果和目标模型生成。把预期差异列成 allowlist；为 tokenizer 错映射、空监督、错误 stop id 和媒体数量不匹配各保留一个负例。

### 练习八：区分压缩收益与训练收益

选两种 tokenizer，先对同一份原文比较 token 数和输出层成本，再分别设计固定原文字节数、固定训练 token 数和固定 FLOPs 的对照。列出每种方案下模型实际看到的原文量、更新数或输出层工作量；选择 held-out 任务和原支持语言回归指标，解释为什么单独比较 token 数或每 token perplexity 不足以证明能力提升。

### 练习九：评估受控的词表扩展

假设你能控制一个已训练模型的源 tokenizer，沿用旧 merge table 扩词表。画出旧 id 映射、新 embedding 初始化、embedding-only 适配和全模型继续训练的状态链；估算新增矩阵的参数与存储量。再设计目标语种压缩、旧语言回归、held-out 任务、训练成本和目标设备每字符延迟的对照，把论文合成估计与设备实测分开。

## 3.21 本章小结

Tokenizer 是训练输入空间的一部分：模型处理的是带版本的 token id，数据格式则为这些 id 指定消息角色、边界事件、媒体内容和监督目标。本章的主线从原文、规范化和模板开始，接到 tokenizer 编码、attention/label mask、文档 packing 和任务专属 schema，最后通过固定样本把训练转换链与推理配置逐项回归。

词表大小和 token 压缩要放在 embedding/输出层、训练预算和目标任务中一起判断；单一 intrinsic 指标不能普遍预测下游模型质量。扩词表会改变 checkpoint 形状、embedding 语义和旧数据编码，必须记录 id 映射、初始化、继续训练及旧能力回归。算法与 tokenizer 评估深讲见第二册，数据治理见第九册，格式故障定位见第十九册；本章负责让训练协议本身可以重放、比较和审计。
