# C. NLP 与 Tokenization：从字符串到模型输入协议

## 阅读边界

语言模型不会直接读取“字”或“词”。它先把字符串变成一个离散符号序列，再把每个符号映射成整数 id，最后通过 embedding 变成向量。这个过程通常被叫作 tokenization，但它不是一个无关紧要的预处理步骤。词表大小、切分规则、特殊 token、对话模板和解码方式共同定义了模型的输入输出协议，也决定了同一段文本要消耗多少上下文、训练多少步和多少显存。

本章不把 tokenizer 当作词典速查表，而是沿着一条完整链路展开：

~~~text
原始字符串
 -> 规范化与边界处理
 -> token 序列
 -> token id
 -> embedding
 -> Transformer
 -> 生成 token id
 -> detokenization
 -> 可读字符串
~~~

全章分为十二个独立主题。每个主题都同时说明初学者容易混淆的直觉、工程上需要记录的变量、算法或公式、最小实验、失败边界和资料来源。关于 tokenizer 数据工程的完整实践，可以继续阅读[第五册第三章：Tokenizer 与数据格式](../../book-05-llm-training/chapters/03-tokenizer与数据格式.md)；关于 tokenization 在预训练和长上下文中的影响，可以阅读[第二册第三章：Tokenization 与预训练进阶](../../book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md)。

本章中的 token 文本是教学示例，不代表某个具体模型的真实词表。不同 tokenizer 即使使用同一个算法名称，也可能因为规范化、字节映射、merge 顺序和特殊 token 配置不同而产生完全不同的结果。

## 3.1 Token 是模型与字符串之间的接口

### token 不等于词

token 是模型处理的离散单位，但它不必对应自然语言学意义上的词。一个 token 可以是：

- 一个完整的常见词。
- 一个词的一部分。
- 一个汉字或多个汉字。
- 一个标点、空格片段或换行。
- 一段 UTF-8 字节映射。
- 一个表示角色、边界或工具调用的控制符号。

例如，英文中的 uncommon 词可能被切成多个子词，中文短句可能按字、词或子词切分，代码中的缩进和括号也可能分别成为 token。token 的边界由 tokenizer 学到的词表和规则决定，不由模型看到字符串后临时猜测。

### 从 token 到 id

tokenization 至少包含两个概念：

1. 把字符串切成 token 文本。
2. 把 token 文本查表得到 token id。

token id 只是离散索引。id 为 100 的 token 不比 id 为 99 的 token 更“大”、更接近或更有意义。真正的语义表示来自 embedding 矩阵中对应的行：

~~~math
e_i=E[i],\qquad E\in\mathbb{R}^{V\times d}.
~~~

V 是词表大小，d 是 hidden size，i 是 token id。模型接收的是 id 序列，embedding 查表后才得到连续向量。

### token 是协议的一部分

在训练和推理中，以下信息都属于输入协议：

~~~text
tokenizer 文件和版本
词表到 id 的映射
规范化规则
special token 的 id
是否自动添加 BOS / EOS
padding 方向和 pad id
chat template
截断和拼接规则
~~~

只保存原始字符串而不保存这些配置，无法保证以后得到相同的模型输入。对一个已训练模型来说，tokenizer 不是可以随手替换的 UI 组件，而是模型参数语义的一部分。

### token 数量与计算账本

一段文本 x 被编码后得到长度 T(x)。一个 batch 的有效 token 数可以写成：

~~~math
N_{\mathrm{tokens}}
=\sum_{i=1}^{B}T(x_i).
~~~

若使用定长 padding，实际送入矩阵的 token 数还包括 padding：

~~~math
N_{\mathrm{padded}}
=B\cdot T_{\max}.
~~~

padding waste 可以用下面的比例表示：

~~~math
\operatorname{waste}
=1-\frac{\sum_i T(x_i)}{B T_{\max}}.
~~~

这个比例只描述 batch 内的填充浪费，不是模型质量指标。它会影响显存、吞吐和每个有效 token 的成本。按长度分桶、动态 padding 或 packing 可以减少浪费，但会增加数据管线复杂度。

### token fertility

在中文、代码或多语言比较中，经常需要定义 token fertility。一个简单版本是：

~~~math
\operatorname{fertility}
=\frac{N_{\mathrm{tokens}}}{N_{\mathrm{reference}}}.
~~~

参考单位可以是 Unicode 字符、字节、词或代码字符，但必须明确写出来。不同论文使用的分母可能不同，因此“某 tokenizer 更省 token”只有在相同文本集和相同计数口径下才有意义。

### 一个不依赖 tokenizer 的成本估算

下面的标准库代码演示如何把不同样本的 token 长度转成 padding waste 和平均长度。它不模拟切分算法，只验证成本账本。

~~~python
def padding_report(lengths):
    if not isinstance(lengths, (list, tuple)) or not lengths:
        raise ValueError("lengths must be a non-empty list or tuple")
    if any(isinstance(length, bool) or not isinstance(length, int)
           or length <= 0 for length in lengths):
        raise ValueError("lengths must contain positive integers")
    batch = len(lengths)
    max_length = max(lengths)
    valid = sum(lengths)
    padded = batch * max_length
    return {
        "samples": batch,
        "valid_tokens": valid,
        "padded_tokens": padded,
        "waste_ratio": 1.0 - valid / padded,
        "mean_length": valid / batch,
    }


print(padding_report([4, 5, 9]))
~~~

输出中的 waste_ratio 约为 0.333333。这个数字只反映三个长度的 batch 组织方式，不代表某个 tokenizer 的效率；要比较 tokenizer，必须先在同一语料和同一 normalization 规则下得到 lengths。

### 初学者最容易犯的错

把 token 数当成字符数，会错误估算上下文和费用；把 token id 当成连续数值，会错误理解 embedding；只记录模型名称而不记录 tokenizer 版本，会让评估和线上推理不可复现。遇到 token 相关问题，先问“切分规则是什么、id 映射是什么、特殊符号如何处理”，再讨论模型能力。

## 3.2 Vocabulary、Embedding 与词表大小

### 词表是什么

Vocabulary 是 tokenizer 支持的 token 集合以及 token 到整数 id 的映射。模型通常有一个输入 embedding 表，输出端有一个投影到词表的 LM head：

~~~math
E\in\mathbb{R}^{V\times d},
\qquad
W_{\mathrm{lm}}\in\mathbb{R}^{V\times d}.
~~~

如果输入和输出权重共享，W_lm 可以直接使用 E；如果不共享，则两者分别占用参数。一个词表扩展可能同时影响输入 embedding、输出 logits、权重绑定、checkpoint 和 tokenizer 配置。

### 词表大小的两面

增大 V 通常会让常见片段拥有更完整的 token，从而减少序列长度；代价是 embedding 参数和输出投影增大。若 hidden size 为 d，输入 embedding 的参数量为：

~~~math
P_{\mathrm{embed}}=Vd.
~~~

输出 logits 的一次矩阵乘法大致涉及 [B,T,d] 与 [V,d] 的投影，V 变大也会增加输出计算和 softmax 成本。词表太小会让字符串切得很碎，词表太大则可能让长尾 token 统计稀疏、参数成本变高。

因此不存在脱离语料、语言、硬件和目标任务的“最佳词表大小”。比较词表时至少要同时看：

~~~text
平均 token 数
长尾文本的 token 数
每语言或每领域的 fertility
embedding / lm head 参数
训练和推理吞吐
special token 预算
下游任务的错误类型
~~~

### token id 的稳定性

一个 tokenizer 的 id 映射只在自己的词表文件和版本中有意义。不能把 A 模型 tokenizer 输出的 id 序列直接交给 B 模型，即使两个模型都说自己使用 BPE。更不能把“同名 token”当成“同 id token”；真正需要比较的是 token 文本、id、字节映射和 special token 配置。

### Weight tying 的影响

输入输出权重共享时，参数量大致少掉一份 Vd，但这不意味着输入和输出功能完全相同。共享矩阵通过两条路径接收梯度，训练数据和损失仍然决定它如何变化。词表 resize 时要确认：

1. embedding 行数是否变为新的 V。
2. LM head 行数是否同步。
3. 两者是否仍然引用同一组参数。
4. 新增行的初始化方式是什么。
5. checkpoint 加载后是否保持绑定。

### 词表边界测试

一个最小 tokenizer 评估集不应只有普通英文句子。至少要包含：

~~~text
空字符串和只含空格的输入
中英混排
emoji 和组合字符
罕见 Unicode 符号
换行、制表符和多余空格
代码括号、缩进和长标识符
数字、小数、日期和单位
special token 的字面字符串
超长连续字符
~~~

对每条样本记录 token 文本、id、decode 结果和是否保留边界。词表大小的统计平均值无法替代这些边界样例。

## 3.3 BPE：从基础符号逐步合并

### BPE 的基本思想

Byte Pair Encoding 最初是一种压缩思想，在语言模型 tokenizer 中通常从字符或字节基础单元开始，通过统计相邻 pair 的频率，反复合并最常出现的 pair。合并过程产生词表和 merge 顺序。

如果初始符号序列是：

~~~text
l o w
l o w e r
~~~

某个 pair 如 l、o 出现频率较高，就可以合并为 lo；之后 lo、w 可能继续合并为 low。真正的实现还要处理词边界、字节映射、空格和跨样本统计，不能把这个玩具例子当成完整 GPT tokenizer。

### 一个抽象算法

设语料被表示成基础符号序列，当前词表为 V，pair 计数为 C(a,b)。一次合并可以写成：

~~~math
(a^\*,b^\*)
=\underset{(a,b)}{\operatorname{argmax}}\;C(a,b),
~~~

然后把所有相邻的 a*、b* 替换成新符号 ab，并把新符号加入词表。重复到达到目标词表大小或没有值得合并的 pair。

解码时，tokenizer 依据学习到的 merge 规则把输入拆成基础符号和合并片段。训练阶段的 merge 顺序必须被保存；只保存最终词表而丢失规则，无法保证 encode 一致。

### BPE 为什么能处理未见词

词级词表遇到新词时容易把整个词映射成 UNK。BPE 则允许新词退化为更小的已知片段，最差可以退化到字节级。因此开放词汇不再依赖“训练时见过完整词”，但这不代表新词一定被高质量理解。一个专有名词被拆成十几个碎片时，序列成本和学习难度都可能上升。

### Byte-level BPE

Byte-level BPE 先把文本转换到字节层面，再对字节片段进行合并。它的覆盖优势来自任意文本最终都能由字节表示，适合混合语言、代码和脏输入。它不表示每个 token 永远只有一个字节：高频字节片段仍会被 merge 成更长 token。

字节层方案的代价包括：

- 某些语言的常见字符可能对应多个字节。
- byte fallback 或不可见映射会让 token 文本不直观。
- 空格和换行边界需要特殊处理。
- 用户看到的 decode 文本与内部字节片段不一定一一对应。

### BPE 的统计偏差

BPE 依据训练语料的共现频率，频繁出现的语言、格式和领域会获得更高效的片段。低资源语言、少见代码库、医学缩写或新产品名可能被切得更碎。一个整体平均 token 数很低的词表，仍可能对少数语言极不公平。

### 一个教学版 BPE merge

下面的代码只实现“统计 pair 并合并”的核心，不包含 Unicode 规范化、byte mapping、边界标记或高效增量计数。它的价值是让合并规则可手算。

~~~python
from collections import Counter


def pair_counts(sequences):
    if not isinstance(sequences, (list, tuple)) or not sequences:
        raise ValueError("sequences must be non-empty")
    counts = Counter()
    for seq in sequences:
        if not isinstance(seq, (list, tuple)) or not seq:
            raise ValueError("each sequence must be non-empty")
        counts.update(zip(seq, seq[1:]))
    return counts


def merge_pair(sequence, pair, merged):
    result = []
    i = 0
    while i < len(sequence):
        if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
            result.append(merged)
            i += 2
        else:
            result.append(sequence[i])
            i += 1
    return result


data = [list("low"), list("lower"), list("low")]
counts = pair_counts(data)
if not counts:
    raise ValueError("at least one adjacent pair is required")
pair, frequency = counts.most_common(1)[0]
data = [merge_pair(seq, pair, "".join(pair)) for seq in data]
print(pair, frequency)
print(data)
~~~

输出会显示最高频 pair 及合并后的序列。真实 tokenizer 还要定义同频 pair 的 tie-break、跨词边界规则和重复合并的高效实现；教学代码没有这些保证。

## 3.4 WordPiece、Unigram LM 与 SentencePiece

### WordPiece 的选择目标

WordPiece 也使用子词词表，但它的候选选择通常与语言模型似然或特定 score 相关，而不是简单按 pair 原始频率逐次合并。BERT 体系中常见的 ## 标记用于表示某个片段处在词内部，但这只是具体实现的可见标记，不是所有 WordPiece 系统都必须使用同一表示。

在推理时，WordPiece 往往在候选词表中寻找能够覆盖输入的片段组合。词表学习目标、最长匹配细节和未知词处理必须以具体 tokenizer 实现为准。把 WordPiece 简化成“最长词典匹配”会漏掉其训练过程和 score 定义。

### Unigram LM 的概率视角

Unigram LM 把一个字符串的切分看成若干子词片段的组合，并给每个片段一个概率。若一种切分为 z=(z_1,\ldots,z_m)，可用：

~~~math
P(x,z)=\prod_{j=1}^{m}P(z_j).
~~~

给定字符串 x，tokenizer 可能选择概率较高的切分：

~~~math
z^\*=\underset{z\in\mathcal{Z}(x)}{\operatorname{argmax}}\;P(x,z).
~~~

实际算法会处理候选集合、似然、删词和动态规划，以上公式只是核心直觉。Unigram 的一个重要特点是可以保留多种候选切分，从而支持 subword regularization 或采样式数据增强。

### SentencePiece 是工具与方法集合

SentencePiece 可以直接从原始文本训练子词模型，不要求先调用语言相关的空格分词器。它支持 BPE 和 Unigram 等不同算法，常用可见符号表示词边界。因为它不依赖英语式空格，中文、日文和多语言语料可以直接进入同一训练管线。

“SentencePiece tokenizer”本身并没有唯一的切分行为。必须同时记录：

~~~text
model_type：BPE 或 Unigram
normalizer 配置
词表文件
special token
unk / byte fallback 行为
是否采样切分
decode 规则
~~~

### 算法名称不是复现条件

两个模型都写着 BPE，仍可能拥有不同结果；两个模型都使用 SentencePiece，仍可能在空格、规范化和 unknown 行为上不同。复现实验需要保存实际 tokenizer 文件和配置，而不是只在 README 写一个算法名。

### 训练时的随机切分

subword regularization 让同一文本在训练时有机会采用不同合法切分，从而给模型输入增加扰动。这可能提高鲁棒性，但也会改变 token 数、padding、loss 对齐和可复现性。评估和线上推理通常要关闭随机切分，否则同一请求可能得到不同输入长度。

## 3.5 字节、字符、子词与多语言公平性

### 三个层次

字符是 Unicode 层面的抽象符号，字节是编码后的存储单位，子词是从语料统计中学习出的片段。一个 Unicode 字符可能由多个 UTF-8 字节表示；一个子词又可能包含多个字符或多个字节。因此“字符数”“字节数”和“token 数”不能互换。

### Character-level

字符级 tokenizer 的词表较小，规则直观，几乎不需要完整词级词表。代价是序列变长，模型需要自己学习字符组合成词、短语、代码标识符和语义单位。对于上下文长度和 attention 计算，长序列会带来直接成本。

### Byte-level

字节级表示覆盖更鲁棒，能够处理罕见符号、emoji、混合文本和任意文件内容。代价是某些字符需要多个字节，若没有足够有效的 merge，token fertility 会升高。byte-level 解决的是覆盖问题，不自动解决语言公平和语义效率问题。

### Subword-level

子词在词级覆盖和字符级长度之间折中。常见短语可以压缩成一个 token，新词可以退化为多个已知片段。它仍然会继承训练语料偏差：高频语言和领域更容易获得短片段，低资源语言或特殊格式可能被切碎。

### 中文

中文没有类似英语空格的天然词边界，但“按字”也不是唯一或始终最佳方案。按字具有鲁棒性，却可能造成更长序列；按词依赖分词器，遇到新词、专有名词和中英混排可能出错；子词模型可以从原始语料学习折中。

中文评估至少应按以下分桶报告：

~~~text
现代汉语连续文本
人名、地名和产品名
数字、日期、单位和金额
中英混排与缩写
古文、方言或低资源变体
带标点和换行的长文档
~~~

只报告整个语料的平均 token 数，会掩盖某些分桶的严重退化。

### 代码

代码的符号和空白可能具有语义。括号、逗号、点号、下划线、驼峰标识符、缩进、换行、字符串和注释都可能影响语法。对 Python 来说，删除或合并换行并不等同于普通文本规范化。代码 tokenizer 需要在 token 效率、语法边界和长尾标识符覆盖之间做平衡。

代码评估不应只看平均 token 数，还要看：

~~~text
标识符是否被过度切碎
缩进和换行是否可逆
括号与运算符是否稳定
长路径和 URL 如何处理
非 ASCII 标识符是否保留
生成代码的 stop token 是否正确
~~~

### 多语言公平的测量方式

对语言集合 L，可以按语言统计平均 token 数：

~~~math
\bar T_\ell
=\frac{1}{N_\ell}\sum_{i=1}^{N_\ell}T(x_i^\ell).
~~~

但平均长度不等于能力公平。还要在相同语义任务上比较质量、上下文截断率、每个有效字符成本和错误类型。tokenizer 的效率差异会通过训练 token 分配和上下文窗口放大，但它只是影响因素之一，不应把所有多语言差异都归因于切分。

## 3.6 Special token：边界、角色和控制信号

### special token 的作用

Special token 是词表中承担控制语义的 token。常见用途包括：

~~~text
BOS：序列开始
EOS：序列结束
PAD：批处理填充
UNK：未知片段
角色标记：system / user / assistant / tool
多模态占位符：image / audio / video
结构边界：turn start / turn end
~~~

它们不是普通文本字符串的别名。通常有独立 id，并且模型训练时会学习它们与生成行为之间的关系。

### BOS、EOS 和 PAD 不是可以随意互换的名字

BOS 是否自动添加、EOS 是否参与 loss、PAD 是否参与 attention，都由训练格式和模型实现决定。对于 decoder-only 模型，padding token 可能在原始预训练中不存在，但批量微调和生成服务仍需要一个 pad id。把 EOS 复用为 PAD 有时可行，但必须重新检查：

1. padding 位置是否被 attention mask 屏蔽。
2. padding 位置是否被 loss mask 忽略。
3. 生成器是否把 pad 当作停止条件。
4. 左右 padding 的 position id 是否正确。

### EOS 是训练和生成之间的连接点

训练数据若没有一致的结束标记，模型可能不知道回答什么时候结束；推理侧若配置了错误的 eos_token_id，模型可能过早停止或持续生成。max_new_tokens 只是上限，不是语义上的正常结束。

### 新增 special token 的结构变化

新增 token 通常需要：

~~~text
更新 tokenizer 词表
扩展 embedding 行数
扩展 lm head 行数
设置 special token 属性
更新 chat template 或数据格式
用训练数据让模型学习新 token
保存并验证 checkpoint
~~~

只更新 tokenizer 而不 resize embedding，会造成 id 越界；只 resize 而没有训练，新增向量通常没有目标语义。若输入输出权重绑定，还要确认扩展后绑定关系仍然存在。

### special token 的字面冲突

若一个特殊字符串既可作为普通用户文本出现，又被 tokenizer 当作控制 token，系统必须定义转义或禁止规则。否则用户输入可能意外改变角色边界或工具调用结构。安全系统尤其要把“可见文本”和“不可由用户伪造的控制 token”分开处理。

### 特殊符号的最小测试

测试每个 special token 时，至少比较三条路径：

~~~text
encode(token 的字面形式)
encode 已注册的 special token
decode 后是否保留、跳过或变换该 token
~~~

很多线上 bug 并不发生在模型，而发生在 decode 时把控制 token 过滤掉、流式输出时把部分字节提前返回，或数据清洗时把角色标记当普通文本拼接。

## 3.7 Chat template：把消息变成训练和推理格式

### 消息对象不是模型输入

应用层常用结构化消息：

~~~text
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."}
]
~~~

模型本身通常不直接接收这个列表。chat template 把它序列化成带角色标记、边界标记和 assistant 起始位置的 token 序列。不同模型对相同 messages 可能产生不同字符串和不同 token ids。

### 模板定义的内容

一个模板通常决定：

- system 消息是否存在及其位置。
- user、assistant、tool 的边界。
- 每轮消息是否添加换行或特殊标记。
- 是否在最后追加 assistant generation prompt。
- 工具调用的 JSON 或文本结构。
- 哪些边界 token 参与 loss。

所以“prompt 文本一样”不能证明输入一样。评估时应保存模板渲染后的字符串、token ids、special token 位置和 generation prompt。

### SFT 中的 prompt/completion 边界

对于只训练 assistant 回答的 SFT，通常把 prompt 部分的 label 设为 ignore_index，只对 completion token 计算损失：

~~~math
L
=-\frac{1}{|\mathcal{I}|}
\sum_{t\in\mathcal{I}}
\log p_\theta(y_t\mid y_{<t},x),
~~~

其中 I 是 assistant completion 的有效位置集合。若把 system、user 或 padding 误加入 I，模型会学到复述指令、生成角色标记或预测 padding 的行为。

### Tool calling 和多模态占位符

工具调用和图像/音频输入会引入结构化占位符。占位符是否对应真实视觉 token、一个压缩后的 embedding，还是只是一段文本标记，要以模型架构为准。模板负责序列化协议，但不能凭空提供模态编码器或工具执行能力。

### 模板版本迁移

模板改变可能让同一个 checkpoint 的输入分布发生变化。迁移时应保留旧模板和新模板的对照样本，比较：

~~~text
渲染字符串
token ids
角色边界
assistant 起始位置
label mask
生成停止位置
工具调用解析结果
~~~

不能只看最终回答是否“看起来差不多”，因为少量模板变化可能在长对话、拒答和工具调用场景中积累成明显差异。

## 3.8 Prompt、Instruction、Completion 与上下文窗口

### 四个概念的边界

Prompt 是模型接收的完整条件上下文；instruction 是其中要求模型完成的任务；completion 是模型在条件下生成的结果；system prompt 是具有特定角色和优先级的全局指令。四者可能重叠，但不应当作同义词。

一个训练样本可以抽象为：

~~~text
context = system + history + user instruction + optional evidence
target  = assistant completion
~~~

训练时需要明确 context 哪些 token进入模型输入，target 哪些 token参与 loss，工具结果和引用证据放在哪个边界内。

### 上下文预算

若模型最大上下文长度为 C，输入 token 数为 T_in，最大生成预算为 T_out，隐藏思考、工具结果或多模态 token 也占用同一预算时，总约束可以写成：

~~~math
T_{\mathrm{in}}+T_{\mathrm{out}}+T_{\mathrm{hidden}}
\le C.
~~~

有些服务把 T_hidden 或工具 token 单独计费，有些把它们统一计入窗口；不能只依据产品界面上的一个数字推断具体 API 语义。

### 截断策略

超出窗口时，系统可能从左侧截断历史、压缩摘要、裁剪检索证据、减少生成预算或直接拒绝请求。每种策略都会改变任务语义：

~~~text
左侧截断：丢失早期指令或对话约束
右侧截断：丢失用户问题或证据末尾
证据裁剪：丢失支持结论的关键段落
生成预算减少：答案可能不完整
摘要替换：引入摘要错误或信息丢失
~~~

工程上要记录实际送入模型的 token 数，而不是只记录原始字符数。

### 长上下文不等于有效使用

标称上下文窗口只表示接口或 runtime 在某种条件下能接受的最大 token 数。有效利用还受训练长度、位置表示、注意力稀释、KV cache、显存、检索排序和任务结构影响。评估长上下文时，应使用已知位置检索、多跳证据、冲突信息和不同位置分桶，而不是只发送一段很长的文本看请求是否成功。

## 3.9 Detokenization：从 id 回到可读文本

### decode 不是简单字符串拼接

Detokenization 把 token ids 按 tokenizer 的映射、空格规则、字节恢复和 special token 过滤策略还原为文本。对于 byte-level tokenizer，单个 token 的可见表示可能只是内部字节片段；对流式生成，前一个 token 和当前 token 合并后才可能形成合法 Unicode 或完整词片段。

因此：

~~~text
decode([id_1, id_2])
不一定等于
decode([id_1]) + decode([id_2])
~~~

尤其在空格、组合字符、byte fallback 和特殊 token 边界处，逐 token decode 再拼接可能出现多余空格、乱码或控制符泄漏。

### 流式输出的边界

流式服务通常每次收到一个或多个新 id，但可以等到安全边界再向用户发送文本。需要定义：

1. 不完整 UTF-8 字节是否暂存。
2. special token 是否过滤。
3. EOS 是否在客户端可见。
4. 工具调用 JSON 是否先缓存到完整结构再解析。
5. markdown 或代码块是否允许分片发送。

把“模型生成 token”与“用户看到文本”当成同一事件，会让流式协议、审计和工具调用出现边界错误。

### encode/decode 的回环测试

一个 tokenizer 的基本回环不是无条件要求原文逐字相等，因为规范化和 special token 可能改变文本。应先写清回环预期：

~~~text
规范化后文本是否应相同
空格和换行是否保留
special token 是否跳过
未知字符是否可逆
非法 Unicode 如何处理
~~~

然后分别测试普通文本、代码、emoji、控制字符和混合语言。

## 3.10 Tokenizer 训练、领域扩展与迁移验证

### tokenizer 训练用什么数据

Tokenizer training 学习词表、merge 规则或子词概率模型。训练语料应代表模型的目标分布，至少考虑：

~~~text
语言比例
代码比例
领域术语
数字和单位
标点与格式
特殊符号
长尾 Unicode
文档和对话结构
~~~

若 tokenizer 训练语料几乎没有代码，真实代码中的标识符和符号可能被切得很碎；若中文和其他语言比例失衡，平均 token 效率会出现系统性差异。tokenizer 不是独立于数据的压缩器，它会把语料分布编码到词表中。

### 规范化规则

大小写、Unicode normalization、空格、换行、全角半角、控制字符和 byte fallback 都会改变 token 序列。规范化可能提高统计一致性，也可能丢失代码或用户输入中的重要信息。必须把 normalizer 配置视为模型协议的一部分，不能只保存 merges 文件。

### tokenizer 与 embedding 扩展

如果新增 ΔV 个 token，embedding 至少要从 [V,d] 变成 [V+ΔV,d]。新增行的初始化可以来自随机分布、已有 token 平均值或专门的初始化策略，但初始化不等于模型已经理解新 token。需要通过包含新 token 的训练数据建立它与上下文之间的关系。

若使用 weight tying，扩展后的 LM head 也必须有对应行。若使用分片、量化或 adapter，扩展还会涉及 checkpoint 格式和量化参数的兼容。

### 何时不应扩展词表

新增领域术语不必然需要新增 token。子词或字节片段已经能够表示它时，扩展词表可能只增加参数和迁移风险。是否扩展应比较：

~~~text
术语出现频率
现有 token 长度
上下文和推理成本
是否需要不可拆分的控制符号
新 token 的训练数据量
旧 checkpoint 兼容性
~~~

对低频词随意加 token，通常无法凭一次离线样例证明收益。

### Tokenizer mismatch

Tokenizer mismatch 指训练、微调、评估、服务或数据预处理使用了不一致的 tokenizer、词表、special token 或 chat template。它可能导致：

- id 与 embedding 行不对应。
- 角色边界被错误解释。
- padding 和 loss mask 错位。
- EOS 不再触发停止。
- 评估 token 数无法比较。
- 训练和线上分布发生隐性漂移。

排查时应打印原始文本、规范化文本、token 文本、token ids、special token 位置、attention mask、label mask 和 decode 结果，而不是只看最终回答。

### 一个可执行的 tokenizer 审计结构

~~~python
def audit_record(
    raw_text,
    normalized_text,
    tokens,
    token_ids,
    special_ids,
    attention_mask,
    labels,
):
    if not isinstance(raw_text, str) or not isinstance(normalized_text, str):
        raise TypeError("raw_text and normalized_text must be strings")
    if not all(isinstance(value, (list, tuple)) for value in (
            tokens, token_ids, special_ids, attention_mask, labels)):
        raise TypeError("token fields must be lists or tuples")
    if len(tokens) != len(token_ids):
        raise ValueError("token and id length mismatch")
    if len(token_ids) != len(attention_mask):
        raise ValueError("attention mask length mismatch")
    if len(token_ids) != len(labels):
        raise ValueError("label length mismatch")
    if any(not isinstance(token, str) for token in tokens):
        raise TypeError("tokens must contain strings")
    if any(isinstance(token_id, bool) or not isinstance(token_id, int)
           or token_id < 0 for token_id in token_ids):
        raise ValueError("token_ids must contain non-negative integers")
    if any(mask not in (0, 1) for mask in attention_mask):
        raise ValueError("attention_mask must contain only 0 or 1")
    if any(isinstance(label, bool) or not isinstance(label, int)
           or label < -100 for label in labels):
        raise ValueError("labels must contain integers >= -100")
    if any(isinstance(token_id, bool) or not isinstance(token_id, int)
           or token_id < 0 for token_id in special_ids):
        raise ValueError("special_ids must contain non-negative integers")
    special_set = set(special_ids)
    return {
        "raw_chars": len(raw_text),
        "normalized_chars": len(normalized_text),
        "tokens": list(tokens),
        "token_ids": list(token_ids),
        "special_positions": [
            i for i, token_id in enumerate(token_ids)
            if token_id in special_set
        ],
        "valid_labels": sum(label != -100 for label in labels),
    }


print(
    audit_record(
        "hi",
        "hi",
        ["h", "i"],
        [4, 5],
        [0, 1],
        [1, 1],
        [-100, 5],
    )
)
~~~

这个审计结构不判断模型回答质量，只确保编码、mask 和 label 的长度契约一致。真实项目还应保存 tokenizer 文件哈希、版本、模板版本和 decode 回环结果。

## 3.11 Tokenization 对训练目标和评估的影响

### next-token loss 的单位

语言模型通常以 token 为单位计算交叉熵：

~~~math
L_{\mathrm{token}}
=-\frac{1}{N}
\sum_{i=1}^{N}
\log p_\theta(y_i\mid y_{<i}),
~~~

N 是有效 label token 数。相同的字符数据，如果 tokenizer 切成更多 token，就会产生更多预测位置和更长的依赖链。训练 token 数不是“文本量”的唯一表达，比较训练预算时必须说明 tokenizer。

### Perplexity 的 tokenizer 依赖

若平均 token loss 为 L，perplexity 可以写为：

~~~math
\operatorname{PPL}=\exp(L)
~~~

但不同 tokenizer 的 token 单位不同，因此不能直接比较两个 tokenizer 下的 PPL，除非明确接受这种差异或换算到共同的字符、字节或词级单位。对跨语言比较尤其要谨慎。

### 评估样本的污染

tokenizer 训练本身通常不需要标签，但它仍可能从目标评估文本中学习到切分统计。更严重的污染发生在模型预训练语料包含评估答案或原文。评估时应记录 tokenizer 版本和模型训练数据边界，避免把“词表看过某种字符串”与“模型学会任务答案”混为一谈。

### 以 token 为单位的公平预算

若给不同语言相同字符预算，它们可能获得不同 token 数和不同有效计算量；若给相同 token 预算，它们又可能覆盖不同字符长度。实验应明确采用哪一种预算，并报告另一种单位的结果。对于生成任务，还要报告输出 token 数、截断率、EOS 命中率和单位成功成本。

## 3.12 资料与证据边界

本章关于 tokenizer 算法的机制，优先依据原始论文；关于 Hugging Face 或 PyTorch 接口的行为，优先依据官方文档；代码中的 merge、长度和 mask 数字是教学构造。资料不能互相替代：论文不保证某个库的当前参数名，官方文档也不证明某种 tokenizer 在所有语言上都更好。

建议优先阅读：

- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)：BPE 子词分割。
- [Subword Regularization](https://arxiv.org/abs/1804.10959)：Unigram 与子词采样。
- [SentencePiece](https://arxiv.org/abs/1808.06226)：从原始文本训练子词模型。
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)：WordPiece 在 BERT 中的应用背景。
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)：GPT-2 byte-level BPE 背景。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：token 序列进入 Transformer 的基础架构。
- [Hugging Face Tokenizers 文档](https://huggingface.co/docs/tokenizers/index)：训练、编码、解码和 tokenizer 组件。
- [Transformers Chat Templates](https://huggingface.co/docs/transformers/main/en/chat_templating)：消息到模型输入的模板接口。
- [Transformers tokenizer 基础](https://huggingface.co/docs/transformers/main/en/tokenizer_summary)：不同 tokenizer 家族的说明。
- [Unicode Standard Annex #29](https://unicode.org/reports/tr29/)：Unicode 文本边界和组合字符。

阅读这些资料时要区分三种结论：

1. 算法论文说明原始方法和特定数据上的实验。
2. 官方文档说明当前实现的接口、默认值和版本边界。
3. 目标语料和目标模型上的实测才说明 token 效率、质量、延迟和成本。

## 本章回顾

tokenization 的核心不是记住 BPE、WordPiece 和 SentencePiece 的名词差异，而是能把一段原始字符串的完整生命周期说清楚：

~~~text
字符与字节
 -> normalization
 -> token 边界
 -> token id
 -> embedding 行
 -> chat template / label mask
 -> context budget
 -> Transformer loss
 -> 生成 id
 -> 流式 detokenization
 -> 用户可见文本
~~~

当模型输出异常时，问题可能不在 Transformer 参数，而在 tokenizer 版本、special token、模板、padding、label mask 或 decode 边界。只有把这些中间产物保存下来，才能区分模型能力问题、数据协议问题和服务实现问题。下一章将进入 Transformer 架构，继续追踪 token embedding 之后的张量如何在 attention 和 MLP 中流动。
