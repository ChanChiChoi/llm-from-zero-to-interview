# 第三部分：Tokenization、数据与预训练进阶

大模型的训练不是把文本丢进 Transformer 就结束了。原始字符串首先要经过 tokenizer，数据要经过筛选、去重、配比和版本管理，训练过程要在有限计算预算下决定模型规模与 token 数，训练结果还要通过分桶验证、下游任务和安全测试才能被解释。

这一部分把预训练链路拆成十五个相互独立、又能连成一条工程路径的主题：tokenizer 的表示选择、tokenizer 评估、数据清洗、重复与污染、数据混合、合成数据、Scaling Law、loss 与能力、训练稳定性、混合精度、checkpoint、评估设计、多语言训练、代码模型，以及一个端到端的预训练方案。每个主题都同时给出入门直觉和进阶分析。入门者先抓住“它解决什么问题”，再看公式；有经验的读者则应继续追问统计口径、反事实、系统成本和证据边界。

一个贯穿全部分的判断是：token 数、参数量、validation loss、benchmark 分数和产品成功率分别描述不同层面的事实。它们可能相关，却不能相互替代。真正可靠的预训练结论必须把数据分布、训练 recipe、评估协议和硬件条件一起写清楚。

整部分按一次预训练决策的因果顺序阅读：

```mermaid
flowchart LR
    A["原始文本与协议"] --> B["Tokenizer 与 token 成本"]
    B --> C["来源、清洗、去重与数据版本"]
    C --> D["配比、合成数据与有效曝光"]
    D --> E["模型规模、token 预算与计算"]
    E --> F["优化轨迹、精度与 checkpoint"]
    F --> G["分桶验证、下游任务与污染检查"]
    G --> H["停止、发布或回到数据改进"]
    H -.->|新错误重新定义训练目标| C
```

图中每条箭头都代表一个需要留证据的转换，而不是一组可以独立调参的按钮：tokenizer 变化会改 token 预算，去重和采样会改有效数据量，recipe 会改 scaling 曲线，评估协议则决定我们能否判断能力变化。后面各讲依次解释这些接口。

## 26. Tokenizer：模型与文本之间的表示接口

### 26.1 模型为什么不能直接读取字符串

神经网络接收的是有限维数值张量，而不是 Unicode 字符串。最基本的处理链路是：

~~~text
原始字符串 -> 规范化 -> token 序列 -> token id -> embedding -> Transformer
~~~


tokenizer 决定了字符串被切成多少个基本单元，以及每个单元如何映射到词表中的整数。这个决定会传导到至少六个地方：

1. 同一段文本占用多少上下文位置；
2. 训练一个 batch 需要多少矩阵乘法；
3. 语言模型的交叉熵如何被平均；
4. 罕见词、专名和代码标识符能否复用已有片段；
5. 生成时一个字符需要多少次解码循环；
6. 训练数据和推理输入是否使用完全相同的边界协议。

因此 tokenizer 不是一个可以随时替换的前处理脚本。它更接近模型结构的一部分。换一个词表，通常意味着 embedding 表、输出头、特殊 token、checkpoint 和评估口径都需要重新处理。

入门时可以把 token 理解成“模型一次读写的积木”。积木太小，句子会变得很长，计算变贵；积木太大，罕见内容会被硬拆，词表也会膨胀。进阶时要注意，这个比喻不能把 token 等同于词或语义单位：一个 token 可能是半个词、一个汉字、几个字节、一个常见代码片段，甚至只是一个标点组合。

### 26.2 字符级、词级与子词级

字符级 tokenizer 直接把每个字符作为基本单元。它几乎没有未登录词问题，但序列通常很长，模型需要经过更多位置才能表达同样的语义。

词级 tokenizer 先按空格或词法规则切分，再把常见词放进词表。它在英文等有明显空格的语言上直观，却有三个结构性问题：词表会随领域膨胀；新词、拼写变化和复合词很难处理；中文、日文、泰语等语言不能自然套用英文空格边界。

子词 tokenizer 试图在两者之间取得平衡。常见词或片段可以用一个 token 表示，罕见词则拆成多个已知片段。于是它既保持开放词表能力，又避免每个字符都占一个位置。

设一段文本的字符数为 C，token 数为 T，常用的粗略效率指标可以写成：

~~~math
\rho=\frac{T}{C}
~~~


`rho` 越小，表示这段文本在当前 tokenizer 下越紧凑。但它不是越小越好：过度把整词、整句甚至训练语料中的长片段记入词表，可能降低跨领域组合能力，增加词表参数，并让罕见片段的 embedding 学得不充分。真正的比较要同时看压缩率、泛化、生成质量和词表成本。

### 26.3 BPE：从频繁相邻片段开始合并

Byte Pair Encoding（BPE）最初是压缩领域的方法，后来被用于子词建模；用于机器翻译的子词做法见 [Sennrich 等人的原始论文](https://aclanthology.org/P16-1162/)。它的训练过程可以抽象为：从字符或 byte 作为初始符号，统计相邻 pair 的频率，每次把最高频 pair 合并为一个新符号，直到达到目标词表大小。

假设训练语料中有如下片段：

~~~text
low lower lowest
~~~


初始符号可能是 `l o w`。如果 `l` 和 `o` 经常相邻，可以先学习 `lo`；如果 `lo` 和 `w` 又经常相邻，可以继续学习 `low`。对于没见过的 `lowest`，tokenizer 仍可用 `low`、`est` 或更小的片段表示，而不必输出一个未知词。

用 `f(a,b)` 表示相邻 pair 的计数，第 r 次合并选择：

~~~math
(a_r,b_r)=\arg\max_{(a,b)\in\mathcal P} f(a,b)
~~~


然后把所有符合规则的相邻 `a_r,b_r` 替换成新符号 `a_rb_r`。训练阶段保存合并序列；编码阶段按既定顺序应用合并规则。

BPE 的优点是算法简单、编码速度快、容易实现。它的局限也来自这种频率驱动的贪心合并：合并规则依赖训练语料，英文高频片段、代码模板或某个网站的特殊格式可能占据词表；局部最频繁的 pair 不一定对应跨领域最有用的单元；不同的预分词和空格处理会产生不同结果。

Byte-level BPE 把 byte 而不是 Unicode 字符作为起点。它几乎可以表示任意输入，降低了未知字符问题，适合开放互联网数据；代价是某些非英文字符会拆成多个 byte 相关 token，中文、阿拉伯语和低资源语言的序列效率可能较差。讨论 byte-level BPE 时，不能只说“不会出现 OOV”，还要报告不同语言的 token fertility。

### 26.4 Unigram：把切分当作概率选择

Unigram tokenizer 采用另一条路径：先构造较大的候选子词集合，为每个候选片段学习一个概率或代价，再从可能的切分中选择总体概率较高的方案；子词采样的提出可参见 [Kudo 的论文](https://aclanthology.org/P18-1007/)。给定字符串 s 的一个切分 z=(z_1,...,z_m)，其近似概率可以写成：

~~~math
P(z\mid s)\propto\prod_{i=1}^{m}P(z_i)
~~~


训练会逐步删除贡献较小的候选片段，使词表收敛到目标大小。与 BPE 的“按 pair 合并”相比，Unigram 更自然地表达“一句话可以有多种切法”。在 subword regularization 中，还可以从多个高概率切分中采样，让模型对边界变化更鲁棒。

Unigram 并不意味着一定优于 BPE。它带来概率建模和采样的灵活性，也带来更复杂的训练和实现。实际选择要看语言覆盖、代码数据、编码速度、词表训练工具和下游效果。

### 26.5 SentencePiece 解决的是边界假设

SentencePiece 是 tokenizer 训练与编码框架，不等同于某一个单独算法。它可以承载 BPE 或 Unigram，并把原始句子视为字符流处理，而不是强制先按空格切词；这与 [SentencePiece 论文](https://aclanthology.org/D18-2012/)描述的 raw-sentence 接口一致。这个设计对中文、日文等没有天然空格边界的语言尤其重要。

SentencePiece 常把空格作为一种显式边界符号，例如 `▁`。这样编码器能区分词首和词内位置，也能在解码时恢复原始空格。这里需要区分三件事：算法决定如何学习片段，框架决定如何训练和编码，词表与归一化规则决定最终协议。只写“用了 SentencePiece”还不足以复现 tokenizer。

### 26.6 词表大小的数量账

词表大小 V 会影响 embedding 和输出头的参数量。若模型宽度为 d，输入 embedding 与 untied 输出头大致增加：

~~~math
P_{\mathrm{vocab}}\approx 2Vd
~~~


若输入输出权重共享，则大致只有 Vd。V 增大通常降低序列长度，但也会增加 softmax 计算和词表参数；V 太小则让序列变长，增加 attention、KV Cache 和每次生成的循环次数。

可以用一个简单的成本近似帮助建立直觉。假设同一批原始字符数为 C，token fertility 为 rho，则序列长度约为 T=rho C。对 dense attention，长度相关的交互量近似为：

~~~math
\mathrm{work}_{\mathrm{attn}}\propto (\rho C)^2
~~~


所以把 fertility 从 1.0 降到 0.8，不只是少 20% token；在 attention 主导的场景中，长度平方项可能让交互量降到原来的 64%。但在短序列或宽模型上，线性投影、FFN、通信和输出 softmax 可能成为主导，不能只靠这一项推断端到端速度。

### 26.7 tokenizer 对中文、多语言和代码的不同影响

多语言 tokenizer 不能只用英文数据训练后再期待所有语言公平。应至少分别统计：

1. 每语言的字符到 token 比率；
2. 每词平均 token 数；
3. 数字、标点和 Unicode 特殊字符的切分；
4. 同一语义内容在不同语言表达下的长度；
5. 低资源语言中未知或 byte fallback 的比例。

代码场景还要关注换行、缩进、下划线、camelCase、括号、运算符和字符串边界。对 Python，空格和换行可能直接决定语法；对 C++，模板符号、作用域和预处理指令会影响补全；对 JavaScript，标识符与符号密集度较高。把所有空白压缩掉再训练，可能让自然语言更紧凑，却破坏代码可执行性。

跨语言 tokenizer 的成本差异也不只是“某种语言在训练数据里够不够多”。NeurIPS 2025 的研究比较了约 7,000 个覆盖 97 种语言的单语 tokenizer。由于不同语言的“词”边界并不总能对齐，作者在平行文本上比较同一内容被编码出的 token 数，而不是把 token/word 当作跨语言通用公平分数。实验显示词表大小和预分词策略都会改变 token 成本；扩大词表本身不能保证消除差异，按语言选择词表规模和 SuperBPE 在其单语实验中降低了 token 数及语言间波动。这些结果支持把表示成本纳入测量，但没有验证共享多语言词表的普遍方案，也没有直接测下游语言模型质量。多语言采样和迁移另在第 38 讲讨论。[Explaining and Mitigating Crosslingual Tokenizer Inequities](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5b91cefbfa52340af2f16f249572dc76-Abstract-Conference.html)。

### 26.8 一个可检查 BPE 合并过程的最小实现

下面的实现刻意保持简单：输入是“空格分隔的符号串 -> 正整数频次”映射，输出是合并后的新映射；它用字符作为初始符号，不处理 byte、Unicode 规范化和跨词边界，只用于观察频率统计与合并顺序。使用 Python 3.12 标准库即可运行。示例每轮选择最高频 pair；并列时按语料插入顺序打破平局，这只是确定性的教学约定。

~~~python
from collections import Counter
from collections.abc import Mapping


def validate_corpus(words):
    if not isinstance(words, Mapping) or not words:
        raise ValueError("words must be a non-empty mapping")
    for word, frequency in words.items():
        if not isinstance(word, str) or not word.split():
            raise ValueError("each word must contain space-separated symbols")
        if (not isinstance(frequency, int) or isinstance(frequency, bool)
                or frequency <= 0):
            raise ValueError("frequencies must be positive integers")


def pair_counts(words):
    validate_corpus(words)
    counts = Counter()
    for word, frequency in words.items():
        symbols = word.split()
        for left, right in zip(symbols, symbols[1:]):
            counts[(left, right)] += frequency
    return counts


def merge_pair(words, pair):
    validate_corpus(words)
    if (not isinstance(pair, tuple) or len(pair) != 2
            or any(not isinstance(symbol, str) or not symbol for symbol in pair)):
        raise ValueError("pair must contain two non-empty symbols")
    if pair_counts(words)[pair] == 0:
        raise ValueError("pair must occur in the corpus")
    merged = "".join(pair)
    result = {}
    for word, frequency in words.items():
        symbols = word.split()
        output = []
        index = 0
        while index < len(symbols):
            if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
                output.append(merged)
                index += 2
            else:
                output.append(symbols[index])
                index += 1
        key = " ".join(output)
        result[key] = result.get(key, 0) + frequency
    return result


corpus = {
    "l o w </w>": 5,
    "l o w e r </w>": 2,
    "l o w e s t </w>": 1,
}
for _ in range(4):
    candidates = pair_counts(corpus).most_common(1)
    if not candidates:
        break
    pair, frequency = candidates[0]
    print("merge:", pair, "count:", frequency)
    corpus = merge_pair(corpus, pair)
print(corpus)
~~~


> **代码状态：静态审阅，未运行（2026-10-29）**

该语料共有 8 个词实例；第一轮中 pair `("l", "o")` 的加权计数是 8。读者可以改动频次或加入新词，观察合并顺序如何变化。工业 tokenizer 还要处理词边界、特殊 token、规范化和高效编码，但这个例子能说明一个容易被忽略的事实：词表不是人工列出的词典，而是训练语料统计和预设边界共同产生的结果。

### 26.9 证据边界

BPE 的子词开放词表思想可以由原始论文支持；SentencePiece 的 raw-sentence 训练、BPE/Unigram 实现和 subword regularization 也有公开论文与工具文档支持。至于某个具体模型的词表大小、byte fallback、特殊 token 名称和归一化规则，应以该版本 tokenizer 文件或模型卡为准，不能从模型系列名称推断。

## 27. Tokenizer 工程：效率、可逆性与协议兼容

### 27.1 训练好 tokenizer 只是起点

一个 tokenizer 能把文本编码成整数，并不表示它适合训练大模型。真正的工程验收至少要回答四个问题：

1. 它是否能稳定地表示目标语言、代码和符号；
2. 它是否在相同字符量下产生合理长度；
3. 编码再解码能否恢复应当恢复的文本；
4. 特殊 token、chat template 和 checkpoint 是否可以长期兼容。

最危险的 tokenizer 问题往往不是“程序直接报错”，而是协议悄悄变化。例如训练时把换行规范化为一个空格，部署时却保留换行；训练时使用一个 `<eos>` id，推理服务把另一个 id 当停止符；新词表的 token id 与旧 embedding 不一致，却被误加载到同一个 checkpoint。这些问题可能只表现为生成质量下降或输出提前截断。

### 27.2 三类效率指标

第一类是序列效率。对每个语言或领域，记录字符数、词数、token 数和 byte 数，计算平均与分位数 fertility。不要只报告均值，因为长尾专名、代码和混合文本可能让 P95 远高于平均值。

第二类是计算效率。用同一模型和 batch 配置比较 token 数变化对吞吐、显存和延迟的影响。tokenizer 使序列变短，可能降低 attention 成本；但更大的词表会增加输出投影，且不同 kernel 对词表大小的敏感性不同。

第三类是信息效率。可以把固定上下文窗口中的有效字符、词或任务信息量作为近似指标。例如在 4096 token 窗口中，中文平均每字一个 token 与中文平均每三字一个 token，实际能承载的文本量不同；这不是语言能力差异，而是表示协议差异。

把这类比较写成数字：同一段中文有 120 个字符，tokenizer A 编成 30 个 token，tokenizer B 编成 60 个，则两者 fertility 分别为 `30/120=0.25` 和 `60/120=0.5 token/字符`。固定 4096 token 窗口时，按这两个均值粗估可容纳约 16,384 和 8,192 个字符。这个结果只说明长度成本；词表参数、代码恢复和跨领域泛化仍须另测。

### 27.3 可逆性不是简单的字符串相等

对普通文本，常见测试是：

~~~math
\mathrm{decode}(\mathrm{encode}(x))=x
~~~


但真实系统要先明确“等价”的定义。Unicode 组合字符、换行符、末尾空格、制表符、大小写和规范化形式可能有多个视觉相同的表示。若 tokenizer 设计了 NFC 规范化，那么原始字节级相等不一定是目标；若处理代码，保留空白和换行通常比视觉相同更重要。

因此应分别测试：

1. 普通 Unicode 文本是否按预期恢复；
2. 代码缩进和换行是否恢复；
3. 特殊 token 字符串是否会被意外解释为控制符；
4. 无效 Unicode、emoji、组合字符和 RTL 文本是否稳定；
5. streaming 编码时分块边界是否改变结果。

流式测试可把一段含 emoji 和中文的文本一次性编码，再按 UTF-8 字节边界附近切成小块送入，比较最终 token 序列与逐块 detokenize 的输出。若整体编码一致、逐块显示却出现乱码或延迟补字，问题可能在增量解码缓存，而非词表本身；代码场景还要比较换行和缩进是否跨块保留。

### 27.4 特殊 token 是控制协议，不是普通词

`BOS`、`EOS`、padding、角色标记、工具调用标记和 FIM 标记都会影响模型的信息流。特殊 token 需要固定 id、明确插入位置和明确 loss mask。例如对 chat 数据，用户消息、助手消息和工具结果是否计算 loss，会改变 SFT 的训练目标；对代码 FIM，prefix、suffix、middle 的顺序必须和训练时一致。

特殊 token 的字符串形式也需要防碰撞。如果用户输入中可以自然出现同名字符串，预处理器必须能够区分“文本中的字面量”和“协议控制符”。一个更安全的做法是使用 tokenizer 内部保留 id，并在模板层明确转义，而不是依赖字符串替换。

### 27.5 词表迁移的数学障碍

旧模型 embedding 矩阵为 E_old∈R^(V_old×d)，新 tokenizer 的词表为 V_new。即使两个 tokenizer 都能编码相同文本，token id 的语义也不一致。除非新旧词表有明确映射，否则不能直接把 E_old 的第 i 行当成 E_new 的第 i 行。

如果只增加少量新 token，可以保留旧行并随机初始化新增行，再用继续训练让新 embedding 对齐；如果改变了大量切分规则，则输出头、数据分布、位置长度和学习率计划都可能需要重新校准。所谓“只换 tokenizer、不动模型”通常只适用于非常受限的扩展，不是通用迁移方案。

版本清单至少要把词表哈希、特殊 token id、规范化规则、template 版本和预处理代码 revision 绑定到 checkpoint。否则权重文件加载成功，只能说明张量形状兼容，不能证明每个 id 仍代表训练时的字符串片段。

### 27.6 评估面板应该长什么样

一个实用 tokenizer 评估面板可以包含：

| 维度 | 指标 | 解释 |
| --- | --- | --- |
| 长度 | 平均 token/字符、P50/P95 | 上下文与吞吐成本 |
| 覆盖 | byte fallback、未知符号、非法序列 | 输入鲁棒性 |
| 可逆 | 编码—解码恢复率 | 协议正确性 |
| 语言 | 每语言 fertility、长尾差异 | 多语言公平性 |
| 代码 | 缩进/换行恢复、标识符切分 | 可执行性 |
| 特殊符号 | 控制 token 碰撞率 | 对话与工具协议安全 |
| 系统 | encode/decode 吞吐、内存 | 服务成本 |

不要只选一个“最小 token 数”的 tokenizer。一个对网页压缩很好的词表，可能在代码、低资源语言或安全控制符上表现糟糕。

测试集应包含组合字符、emoji、RTL、代码缩进、末尾空格、特殊 token 字面量和多轮工具消息。给每条样本保存输入文本、期望 token id、解码结果及停止条件；升级后先比较差异，再判断哪些是预期的规范化变化。这样能把“模板错误”“token 切分改变”和“模型行为退化”分成不同故障。

### 27.7 一个 tokenizer 统计脚本

下面的代码使用 Python 3.12 标准库和一个可替换的 `encode` 函数，输入是按领域标注的文本样本，输出每个领域的样本数、逐样本 token 数和语料加权 fertility。样例编码器会把 ASCII 字母数字串作为一个片段、其余非空白字符逐个计数；它不是实际模型 tokenizer。实际测量时替换 `toy_encode`，并报告真实样本量，不能把这里每桶两条构造样本的结果解释为语言总体统计。

~~~python
from collections import defaultdict
from statistics import median


def toy_encode(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    # 教学用：把 ASCII 单词视作片段，其他字符按字符处理。
    tokens = []
    current = []
    for char in text:
        if char.isascii() and char.isalnum():
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []
            if not char.isspace():
                tokens.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


samples = [
    ("zh", "模型会读取上下文。"),
    ("zh", "词表影响序列长度。"),
    ("en", "The model reads context."),
    ("en", "A tokenizer maps text to ids."),
    ("code", "def add_numbers(left, right):\n    return left + right"),
    ("code", "value = model(input_ids)"),
    ("empty", ""),
]
lengths = defaultdict(list)
for domain, text in samples:
    if not isinstance(domain, str) or not domain:
        raise ValueError("domain must be a non-empty string")
    tokens = toy_encode(text)
    lengths[domain].append((len(tokens), len(text)))


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return None
    return round(numerator / denominator, 3)


for domain, values in lengths.items():
    token_counts = [item[0] for item in values]
    total_chars = sum(item[1] for item in values)
    pooled_fertility = safe_ratio(sum(token_counts), total_chars)
    print("domain=", domain, "n=", len(values))
    print("  tokens/sample=", token_counts)
    print("  median tokens=", median(token_counts))
    print("  corpus fertility=", pooled_fertility)
~~~


> **代码状态：静态审阅，未运行（2026-10-29）**

这个统计器的价值不在于 toy tokenizer 的数值，而在于把“词表好不好”改写成按语言、领域、长度和协议逐项测量的问题。`corpus fertility` 是总 token 数除以总字符数；`empty` 桶展示了全空文本时分母为零，结果保留 `None`，不会把分母替换成 1 后制造一个看似正常的比例。生产报告还应给出 P50/P95 长度、置信区间或分层抽样方案；七条演示记录只用来验证统计口径。

### 27.8 chat template 与 tokenizer 的边界

chat template 常和 tokenizer 一起发布，但两者职责不同。tokenizer 负责字符串与 id 的变换，template 负责把角色、消息、工具调用和结束条件组织成输入字符串或特殊 token 序列。模板改变了训练和推理的上下文，可能影响 loss、停止位置和工具协议。

审计一个模型时，应保存 tokenizer 版本、特殊 token 表、template 版本和一组 golden examples。只验证普通文本编码而不验证完整对话模板，无法排除角色串位、重复 BOS、错误 EOS 或工具结果被当作用户文本等问题。

近期研究也在检验模型是否必须始终接收 tokenizer 的规范切分。NeurIPS 2025 的 Broken Tokens? 在其 20 个基准上构造未见过的非规范 token 序列；指令微调模型在抽样切分下最高保留 93.4% 的原聚合表现，在字符级切分下最高保留 90.8%。字符切分还让特定字符串操作和代码理解任务最高提升 15%，右对齐数字分组让特定大整数运算提升超过 33%。这些是论文模型和任务上的结果，作者观察到鲁棒性依赖指令微调阶段；它们说明部分模式可跨切分保留，却不证明生产端能安全改动 tokenizer、特殊 id、模板、解码或缓存协议。[Broken Tokens?](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2b5c5689fae6fa9a4883e73e511d52c8-Abstract-Conference.html)。

zip2zip 把压缩从固定词表延伸到推理时动态合并：LZW 根据当前上下文形成 hypertoken，并配套动态 embedding/unembedding；论文用 PEFT 对已有模型做约 10 GPU 小时的 uptraining，在其设置中把输入和输出序列缩短 15%—40%。论文还在特定模型、上下文、生成长度和硬件上报告最高约 40% 的端到端延迟改善；token 变少本身不保证同样的端到端提速，因为词表操作、输出投影、内核和请求形态都会改变成本。[zip2zip](https://proceedings.neurips.cc/paper_files/paper/2025/hash/def682d57535b9e778508b4b1a911034-Abstract-Conference.html)。

2026 年的 zip2zip++ 预印本继续改善动态词表的 embedding 几何和合并过程，但并未在所有能力上追回未压缩模型。作者在四个指令模型上用 1B 微调数据做 8,000 步 LoRA、最大合并长度设为 4；以 Phi-3.5-mini 为例，GSM8K 从 zip2zip 的 0.36 提升到 0.68，未压缩模型为 0.75；HumanEval pass@1 从 0.26 提升到 0.32，未压缩模型为 0.58；IFEval 仍为 0.32，低于未压缩模型的 0.38。14B 设置中的代码和指令结果也仍有回退。它目前是小范围预印本证据，应理解为压缩与质量折中的研究进展，而非可以直接替换现有 tokenizer 的生产方案。[zip2zip++](https://arxiv.org/abs/2610.13605)。

最终验收要把文本、token id、模型输入、流式输出和停止条件放在同一次回归中。若 encode/decode 正常但角色串位，优先查 template；若只有 chunk 边界异常，查增量 tokenizer/detokenizer；若 id 和输入一致而答案变化，再检查模型或解码配置。这样的分层检查让协议问题能定位到具体接口。

## 28. 数据来源与清洗：把网页集合变成可追溯训练语料

### 28.1 数据质量不是一个分数

“高质量数据”至少包含四层含义：文本本身可读，内容对目标能力有用，来源和许可可追溯，样本不会因为重复、污染或隐私问题带来系统风险。一个语法通顺的网页可能是 SEO 垃圾；一个技术论坛回答可能很有价值，却包含过时 API；一段代码能运行，却带有不允许进入训练集的密钥。

因此，数据质量更适合被看成一个带 metadata 的对象，而不是单一标签。对样本 x，可以记录：

~~~text
source, timestamp, language, domain, license, quality_score,
dedup_cluster, pii_flag, safety_flag, benchmark_overlap, version
~~~


训练时不仅使用文本内容，还要保留这些字段，以便做采样、分桶验证、回溯和删除。

### 28.2 数据管线的分层结构

一个可复现的数据管线可以按以下顺序组织：

1. **采集**：保存原始来源、抓取时间、URL 或仓库 revision；
2. **解析**：提取正文、代码、标题、表格和文档结构；
3. **语言与领域识别**：记录 document-level 和必要的 paragraph-level 标签；
4. **安全与合规过滤**：处理个人信息、凭证、恶意内容和许可约束；
5. **质量过滤**：去模板、去乱码、去极短样本和异常重复；
6. **规范化**：只做不会破坏目标信息的 Unicode、空白和格式处理；
7. **去重与污染分析**：产生 cluster id 和评估集重叠报告；
8. **切分与混合**：形成训练、验证和隐藏评估清单；
9. **打包**：写入不可变 manifest、版本号和校验信息。

每一阶段都应可独立重跑。否则当评估集污染或隐私问题出现时，团队只能重新抓取全部数据，无法定位是哪条规则造成了问题。

### 28.3 解析与规范化的边界

网页中常见导航、广告、cookie 弹窗、重复页脚和脚本内容。过滤它们能提高正文比例，但过度清洗也会删除代码块、表格、标题层级和问答上下文。不同领域要使用不同规则：新闻的日期与标题可能重要，API 文档中的代码块和版本号更重要，论坛中的问题—答案关系比单段文本更重要。

规范化同样要谨慎。Unicode NFC 可以合并部分等价表示，统一换行符可以改善跨平台处理；但删除所有空格会破坏代码，折叠多个换行可能消除 Markdown 结构，修改大小写会影响标识符和专名。每个规范化规则都应有反事实测试：它是否改变了一个可执行样例、数学公式或引用关系？

### 28.4 质量过滤的三种证据

启发式过滤速度快、容易解释，例如长度范围、乱码比例、重复行比例、链接密度和语言一致性。它适合第一道粗筛，但容易把特殊领域误判为低质。

模型分类器能识别可读性、教育价值、代码质量或毒性，但分类器本身有训练分布偏差。它的输出应该作为采样权重或人工抽检依据，而不是无条件真值。

任务可执行性提供更强证据。代码可以编译或运行测试，数学题可以通过符号或数值验证，结构化文档可以检查 schema。它的成本更高，也不能覆盖所有质量维度，但在目标能力数据上往往比通用质量分更有区分度。

可以把样本质量记成一个可解释的组合分数：

~~~math
q(x)=w_r r(x)+w_d d(x)+w_v v(x)-w_n n(x)
~~~


其中 `r` 是可读性，`d` 是领域相关性，`v` 是可验证性，`n` 是内容噪声；实现时应把各项映射到一致尺度，并令权重非负。这个式子只是对已获准进入候选池的样本做质量排序，不是自然定律，权重要由目标任务和验证结果校准。来源可追溯性、许可是否符合项目策略、隐私与安全审查、评估集污染等应作为独立准入门槛或隔离条件，不能并入加权分数；否则一个高质量分可能错误地抵消许可未知或隐私风险。

### 28.5 语言、代码和多模态数据的特殊清洗

多语言数据要关注语言识别错误、机器翻译痕迹、乱码、重复翻译和低资源语言的样本稀少。不能直接把英文 stopword 比例、空格比例或字符范围规则套在中文、阿拉伯语、泰语和混合文本上。

代码数据要保留仓库结构、测试和文档关系，过滤二进制、minified 文件、vendor 依赖、自动生成物和明显的复制模板。对 Python 不能随意重排缩进；对 Markdown 不能把代码围栏当作普通标点删除。

图像、音频或视频数据还要记录媒体 hash、分辨率、时长、字幕来源、OCR 质量和版权信息。虽然本章重点是文本预训练，但同样的原则适用：样本内容、元数据、许可和评估污染必须一起管理。

### 28.6 隐私、密钥和许可

训练数据中可能出现姓名、电话、地址、邮箱、身份证号、API key、云凭证、私钥和数据库连接串。检测规则可以结合正则、secret scanner 和上下文分类器；脱敏后要检查是否仍保留了不应暴露的组合信息。

代码仓库还涉及 license。不同项目可能只接受 MIT、Apache-2.0、BSD 等许可，也可能依据司法辖区和产品用途制定不同策略。一个训练数据集的 manifest 应记录 license 识别结果和排除原因，而不是只在项目说明中笼统写“使用开源代码”。

### 28.7 数据版本和可删除性

每个样本最好有稳定的内容 hash、来源 id 和数据版本。这样当某个来源要求删除、发现 benchmark 泄漏或更新许可规则时，可以从 cluster 和 manifest 中定位受影响样本。

训练数据的不可变版本并不意味着永不更新。正确的做法是保留版本谱系：v1 由哪些来源组成，v2 删除或新增了什么，质量规则发生了什么变化，训练 run 使用的是哪个版本。没有版本谱系，loss 变化和能力变化就很难归因。

### 28.8 一个带来源字段的清洗示例

下面的代码展示数据记录如何携带来源和许可 metadata，并把许可策略作为硬门槛。输入是包含文本与来源字段的字典，以及由项目合规策略提供的许可白名单；输出包含内容哈希、拒绝原因和准入布尔值。它只检查两类凭证模式和若干结构规则，不是完整的隐私检测器，也不能代替法务审查。

~~~python
import hashlib
import re


SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA|OPENSSH) PRIVATE KEY-----"),
]


def content_hash(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inspect_record(record, allowed_licenses):
    if not isinstance(record, dict):
        raise TypeError("record must be a dictionary")
    if not isinstance(allowed_licenses, (set, frozenset)) or not allowed_licenses:
        raise ValueError("allowed_licenses must be a non-empty set")
    if any(not isinstance(item, str) or not item.strip()
           or item.strip().casefold() in {"unknown", "unspecified", "none"}
           for item in allowed_licenses):
        raise ValueError("allowed licenses must be named policy entries")
    text = record["text"]
    if not isinstance(text, str):
        raise TypeError("record text must be a string")
    stripped = text.strip()
    flags = []
    if len(stripped) < 80:
        flags.append("too_short")
    if stripped.count("<script") > 2:
        flags.append("template_or_script_heavy")
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        flags.append("possible_secret")
    source = record.get("source")
    if (not isinstance(source, str) or not source.strip()
            or source.strip().casefold() == "unknown"):
        flags.append("source_untraceable")
    license_name = record.get("license")
    if (not isinstance(license_name, str) or not license_name.strip()
            or license_name.strip().casefold() in {"unknown", "unspecified", "none"}
            or license_name not in allowed_licenses):
        flags.append("license_not_approved")
    return {
        "id": content_hash(text),
        "source": source if isinstance(source, str) else "unknown",
        "language": record.get("language", "unknown"),
        "license": license_name if isinstance(license_name, str) else "unknown",
        "flags": flags,
        "keep": not flags,
    }


records = [
    {"text": "A short note.", "source": "web", "language": "en", "license": "unknown"},
    {"text": "A" * 120, "source": "docs", "language": "en", "license": "CC-BY"},
]
for record in records:
    print(inspect_record(record, allowed_licenses={"CC-BY"}))
~~~


> **代码状态：静态审阅，未运行（2026-10-29）**

运行方式为 Python 3.12 标准库直接执行；两条构造记录中，短文本且许可未知的第一条会因多个硬门槛被拒绝，第二条仅在该示例白名单策略下通过。生产系统中应把 `keep=False` 拆成具体原因，不能让一个综合布尔值掩盖“短文本”“疑似密钥”“来源不可追溯”和“许可未获准”之间完全不同的处理路径。白名单是项目输入，不表示这些许可在所有地区、用途或产品中都自动适用。

### 28.9 证据边界

数据清洗规则的具体阈值通常是项目经验，不应写成普适定律。[C4](https://arxiv.org/abs/1910.10683)、The Pile、[FineWeb](https://arxiv.org/abs/2406.17557) 和 [BigCode](https://arxiv.org/abs/2308.07124) 等数据工作可以支持某些数据构造与过滤经验；但某个模型到底使用了多少网页、代码或合成数据，只有其正式技术报告或模型卡披露后才能作为事实。

清洗规则的误差应有抽样估计。比如质量分类器每 1000 条误删 30 条高质量样本，误删率为 3%；如果误删集中在低资源语言或代码片段，整体平均值会掩盖该领域损失。可以按来源、语言和内容类型分层抽查误删与漏放，再调整阈值或增加规则。过滤门槛越多，越需要记录各门槛的单独命中数量，否则无法知道最终语料变化来自何处。

把清洗门槛变成可复算计数。假设一批 4 条记录中，2 条文本长度合格、来源可追溯且许可在白名单内；1 条只有 20 个字符，低于项目设定的最小长度；另 1 条内容完整但许可未知。若本轮硬门槛要求文本至少 50 字且许可通过，则保留 2/4=50%。被拒的两条原因不同：短文本可考虑并入文档或单独处理，许可未知则需要法务/授权核实；不能把它们都归为“低质量”。

## 29. 去重与数据污染：避免把记忆误判为泛化

### 29.1 为什么重复会改变训练结果

相同或近似样本被重复采样，会同时影响优化、记忆和评估。[Lee 等人](https://arxiv.org/abs/2107.06499)在公开实验中研究了训练数据去重对语言模型记忆和质量的影响，但这不构成所有数据集上的统一收益结论。重复可能降低训练 loss，却不增加等量的新信息；热门仓库或模板被大量 fork，会让某一领域在混合数据中被隐性放大；验证集或 benchmark 若出现在训练集，分数就不能再解释为未见泛化。

重复并非总是坏事。高质量样本在目标能力稀缺时可以有意过采样，低资源语言也可能需要重复暴露。关键是区分“有计划的重复采样”和“数据管道不知道自己重复了”。前者应记录 exposure，后者会让预算和评估失真。

### 29.2 三层去重

**Exact dedup** 对原始字节或规范化文本计算 hash。它便宜、确定性强，适合先删除完全相同的网页、文件和样本。

**Near dedup** 处理空白变化、模板变化、轻微改写或代码格式变化。常用 shingle、MinHash、SimHash 或 token n-gram 指纹。设两个文档的 k-shingle 集合为 A 和 B，Jaccard 相似度为：

~~~math
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
~~~


当 J 超过某阈值时，可以把它们视作同一重复簇，但阈值必须按领域校准。新闻文章的共同模板不等于全文复制，代码中的标准库导入也不等于两个函数重复。

**Semantic dedup** 使用 embedding 或分类器发现语义高度相似的内容。它能识别改写和翻译，却更贵、更依赖模型，并可能误删同一主题下真正互补的解释。它适合做候选聚类和采样降权，不宜在没有人工抽检的情况下大规模硬删除。

### 29.3 文档级、行级和仓库级

网页去重通常以页面或段落为单位；代码去重还需要文件、函数和仓库三个粒度。fork 仓库可能只有少量业务改动，整仓库 hash 不同，但核心实现高度相似。对代码，可以先根据 repo fork、依赖目录和文件指纹建立候选簇，再对函数或 token n-gram 做细粒度比较。

去重顺序也会影响结果。先做 exact dedup 可以减少 near-dedup 成本；先按来源划分训练和验证，再跨集合做重复检测，可以避免同簇样本被分到两边；对有时间信息的数据，按时间切分后再做 overlap 报告，有助于判断模型是否可能见过未来内容。

### 29.4 训练污染的几种形态

1. **原文污染**：评估题目或完整文档直接进入训练集；
2. **答案污染**：题目不在训练集，但答案、解析或代码实现出现；
3. **模板污染**：训练数据包含与 benchmark 高度相同的题型和格式；
4. **改写污染**：题目经过轻微改写、翻译或重排后仍可匹配；
5. **时间污染**：评估集发布时间早于训练数据截止时间，模型可能见过后续转载；
6. **合成污染**：根据公开 benchmark 生成大量相似题目，模型学到评估分布而非一般能力。

污染分析不是判断模型“是否记住了某题”的充分条件，但它能告诉我们 benchmark 分数的证据强度。发现 overlap 后，通常应更换 fresh eval、报告污染比例，或把结果标成受污染的观察值。

### 29.5 去重的反作用

过度去重会删除互补解释、不同语言表达和真实复述。比如同一个 API 在官方文档、教程、代码和 issue 中出现，表面相似却提供不同监督信号；数学定理的多种证明也不应因为共享公式就被全部删除。

因此去重系统需要输出 cluster，而不只是一个删除列表。可以按 cluster 选择最高质量样本、保留不同语言和不同任务形式，或对同簇样本降低采样概率。去重结果也要进入训练 manifest，便于之后分析某个能力是否因过滤而下降。

### 29.6 评估污染的检测组合

一个可操作的污染检测组合包括：

1. exact hash 匹配；
2. normalized text hash；
3. n-gram overlap；
4. MinHash/SimHash 候选；
5. embedding 相似度复核；
6. 题目、答案和解析分别匹配；
7. 时间截止与来源追踪；
8. fresh、hidden 和人工构造评估。

任何一种检测都可能漏报或误报。报告时应说明检测算法、阈值、参与比较的数据版本和被标记样本的处理方式。

### 29.7 一个可复现的 shingle 相似度示例

代码输入两段字符串和正整数窗口宽度，输出字符 shingle 集合及其 Jaccard 相似度；若文本短于窗口，无法构造 shingle，函数返回 `None`，调用方不能把它解释为“完全相同”。它使用 Python 3.12 标准库运行。

~~~python
def shingles(text, width=5):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(width, int) or isinstance(width, bool):
        raise TypeError("width must be an integer")
    if width <= 0:
        raise ValueError("width must be positive")
    normalized = " ".join(text.lower().split())
    return {
        normalized[index:index + width]
        for index in range(max(0, len(normalized) - width + 1))
    }


def jaccard(left, right):
    if not isinstance(left, set) or not isinstance(right, set):
        raise TypeError("left and right must be sets")
    union = left | right
    return len(left & right) / len(union) if union else None


documents = [
    "Transformer uses attention to mix token representations.",
    "The Transformer uses attention for mixing token representations.",
    "A database index speeds up selective queries.",
]
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        score = jaccard(shingles(documents[i]), shingles(documents[j]))
        print(i, j, round(score, 3))
~~~


> **代码状态：静态审阅，未运行（2026-10-29）**

三个样本两两比较会输出相似度；若把 `width` 改为大于所有样本长度，分数为 `None`，因为空集合之间的相似度在这里没有可用的判定意义。这个实现只是候选生成器，不足以承担生产去重。它没有处理词序变化、代码语法、跨语言改写或模板公共部分；它的意义是让 Jaccard 阈值、样本长度和误删风险可以被实际观察。

### 29.8 记忆与泛化的实验边界

重复率高不自动证明模型只是在记忆，去重后能力下降也不自动证明原始数据更有价值。可以设计三组对照：原始数据、exact dedup、near dedup；固定 token budget、模型规模和训练 recipe；在训练域、fresh 数据和受控复述集上分别评估。这样才能区分记忆收益、数据多样性收益和过滤误伤。

以五字符 shingles 为例，若文档 A 的集合是 `{abcde, bcdef, cdefg, defgh}`，文档 B 为 `{abcde, bcdef, cdefg, xyz12}`，交集有 3 项、并集有 5 项，Jaccard=3/5=0.6。阈值若设为 0.6，这两篇会进入同一候选簇；阈值调到 0.7 则不再匹配。这个数字说明阈值是决策边界，不是内容相同的概率。文档模板可能抬高相似度，短文本又容易被单一公共片段主导，所以网页、代码、公式和问答要分别抽样校准。

去重结果应保留可追溯的簇 ID、代表样本、相似度和处理动作。若簇内有 100 个近重复页面，直接删除 99 条会降低重复曝光，却也可能删掉不同时间版本、纠错记录或多语言互补解释。可以选择最高质量代表、按来源/语言保留少量变体，或保留全部但把采样权重除以簇大小。训练后再看 fresh eval 和记忆测试，判断过滤是在减少记忆风险还是误删有效监督。

相似度阈值要按数据用途校准。对于新闻转载，文本相似度高可能意味着重复；对于 API 文档，多个版本共享相同参数说明却可能各自提供有效版本信息；对于代码，公共函数签名相同但实现和测试不同也不应一律删除。抽取一批高相似候选，由人工标注“应合并、应保留、需版本区分”，再计算阈值的精确率与召回率。删除策略应可回滚，并把被删内容、原因与代表样本留在数据谱系中。

Benchmark 污染检测也需要分别看题干、答案、解析和模板。题目文本不重合，答案代码或解析仍可能进入训练；通过公开数据生成的改写题，也可能让模型学习到测试分布。训练截止日期、来源链接、近重复检测、人工抽查与 fresh 题共同构成证据。未找到 exact match 只说明这一种检测没有命中，不能写成“确定没有污染”。

去重的结果会改变各领域剩余的有效样本量；下一讲转向预算分配，讨论在已知来源、质量和可用量后，如何选择采样比例并测出重复暴露。

## 30. 数据配比、采样与课程：有限预算如何分配

### 30.1 token 数量不是数据价值

假设两个数据集都包含 1T token：A 是高质量、去重充分、覆盖代码和数学的混合语料；B 是重复网页、模板文本和低质量自动生成内容。它们的 token 数相同，却不意味着提供相同训练信号。

可以把总训练效用粗略写成：

~~~math
U(D)=\sum_{x\in D}v(x)-\lambda\,r(x)-\mu\,c(x)
~~~


其中 `v(x)` 表示样本对目标能力的价值，`r(x)` 表示重复和污染风险，`c(x)` 表示处理或验证成本。这个式子不是可直接求解的自然规律，而是提醒我们：配比决策同时受能力、风险和预算约束。

### 30.2 原始比例、均匀比例与温度采样

对领域或语言 i，原始 token 比例为 p_i。最简单的原始比例采样令 q_i=p_i；均匀采样令所有 q_i 接近 1/K。多语言和多领域训练经常使用温度采样：

~~~math
q_i=\frac{p_i^\alpha}{\sum_j p_j^\alpha},\qquad 0<\alpha\le 1
~~~


当 α=1 时保持原始比例；α<1 时分布变平，低资源领域概率上升。用温度 T 表示时常写作 α=1/T。

温度采样并不是自动公平。若低资源数据可用量为 A_i，而计划训练量为 B_i，则平均暴露倍数为：

~~~math
e_i=\frac{B_i}{A_i}
~~~


当 e_i 很大时，模型可能反复看到同一批样本，低资源语言的 validation loss 可能下降，但 fresh eval 不提升。采样表必须同时记录 q_i、可用 token、预期 exposure 和每领域验证趋势。

### 30.3 目标驱动配比

真实产品通常不是“每个领域都一样重要”。如果目标是代码助手，代码、测试、文档、issue 和自然语言解释的比例应围绕代码任务设计；如果目标是中文技术助手，应保证中文技术语料、英文论文和中英转换任务都有足够锚点。

一个可解释的配比表应包含：

| 数据桶 | 原始量 | 训练比例 | 预期重复倍数 | 主要目标 | 主要风险 |
| --- | ---: | ---: | ---: | --- | --- |
| 通用网页 | 900B | 45% | 0.50x | 常识与语言流畅 | 噪声、模板 |
| 代码与测试 | 300B | 25% | 0.83x | 代码生成与修复 | 许可、污染 |
| 数学 | 80B | 10% | 1.25x | 符号推理 | 答案错误 |
| 中英文技术文档 | 200B | 15% | 0.75x | 跨语言技术问答 | 翻译腔 |
| 对话与工具 | 50B | 5% | 1.00x | 指令与协议 | 风格过拟合 |

表中假设总训练预算为 1T token。例如数学桶分到 `1T×10%=100B`，可用量是 80B，所以平均曝光为 `100/80=1.25` 次；其余各行同样按计划 token 数除以原始可用量计算。数字仅为教学输入，不是配方推荐。

目标权重也应显式写出。假设代码助手把代码通过率权重设为 0.7、通用能力设为 0.3；基线分数分别为 0.40 和 0.75，综合分为 `0.7×0.40+0.3×0.75=0.505`。候选配比得到 0.46 和 0.72，综合分为 0.538，增加 0.033。这个加权数只是预先声明的决策规则；若候选触发安全、延迟或许可硬约束，即使综合分更高也不能直接选用。

一次配比试验的增益还可能来自随机种子或样本顺序。关键候选应使用相同种子集合、相同 token 预算，并报告平均差和 run 间波动；若目标提升小于波动范围，应扩大实验或标为证据不足，不能据一个最优 run 调整整套数据。

### 30.4 课程学习与阶段配比

课程学习不是简单地“先容易后困难”。它可以表示为训练阶段 t 的数据分布 q_t(x)：

~~~math
q_t(x)=\mathrm{Normalize}\left(q_0(x)\,w_t(x)\right)
~~~


早期可能提高干净、短、结构稳定的样本比例，帮助模型建立基本语言和代码模式；中后期逐步加入长上下文、困难数学题、真实 bug 和多轮工具轨迹。问题在于课程切换会造成 loss 分布突变，也可能让模型遗忘早期领域，因此每次切换都需要保留混合回放和分桶验证。

### 30.5 动态采样的反馈环

可以依据每个领域的验证损失、目标任务收益或数据重复度调整采样。一个简化控制器是：

~~~math
q_i^{(t+1)}\propto q_i^{(t)}\exp\left(\eta\,g_i^{(t)}\right)
~~~


其中 g_i 可以是该领域的收益、欠训练程度或目标缺口，η 是调整步长。若反馈噪声很大，q_i 会来回震荡；若只追逐短期 benchmark，系统可能过拟合评估集。因此动态采样应设置上下界、平滑窗口和人工审计点。

### 30.6 一个采样与暴露量计算器

下面的 Python 3.12 标准库示例输入各数据桶可用 token 数、总训练预算和温度，输出采样比例、计划 token 数及每桶 exposure。实现使用 log-space 归一化，避免直接计算极端温度的幂时让所有权重同时下溢；它假设各桶数据量为正，不能表达“没有可用数据”的桶。

~~~python
import math
from collections.abc import Mapping


def normalize(values):
    if not isinstance(values, Mapping) or not values:
        raise ValueError("values must be a non-empty mapping")
    numbers = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("weights must be real numbers, not booleans")
        try:
            numeric = float(value)
        except OverflowError as error:
            raise ValueError("weights must be representable as finite floats") from error
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError("weights must be finite and non-negative")
        numbers[key] = numeric
    maximum = max(numbers.values())
    if maximum <= 0:
        raise ValueError("values must have a positive total")
    scaled = {key: value / maximum for key, value in numbers.items()}
    total = math.fsum(scaled.values())
    return {key: value / total for key, value in scaled.items()}


def temperature_ratio(available, temperature):
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a real number")
    try:
        temperature = float(temperature)
    except OverflowError as error:
        raise ValueError("temperature must be finite") from error
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not isinstance(available, Mapping) or not available:
        raise ValueError("available must be a non-empty mapping")
    amounts = {}
    for key, amount in available.items():
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise TypeError("available amounts must be real numbers")
        try:
            numeric = float(amount)
        except OverflowError as error:
            raise ValueError("available amounts must be finite floats") from error
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError("available amounts must be finite and positive")
        amounts[key] = numeric
    log_amounts = {key: math.log(amount) for key, amount in amounts.items()}
    largest_log = max(log_amounts.values())
    weights = {
        key: math.exp((log_amount - largest_log) / temperature)
        for key, log_amount in log_amounts.items()
    }
    return normalize(weights)


def sampling_plan(available, budget, temperature):
    if isinstance(budget, bool) or not isinstance(budget, (int, float)):
        raise TypeError("budget must be a real number")
    try:
        budget = float(budget)
    except OverflowError as error:
        raise ValueError("budget must be finite") from error
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and non-negative")
    ratio = temperature_ratio(available, temperature)
    planned = {key: budget * value for key, value in ratio.items()}
    exposure = {key: planned[key] / float(available[key]) for key in available}
    if any(not math.isfinite(value) for value in exposure.values()):
        raise OverflowError("exposure exceeds the Python float range")
    return ratio, planned, exposure


available = {"web": 900, "code": 300, "math": 80, "bilingual": 200}
budget = 400
for temperature in [1.0, 2.0, 4.0]:
    ratio, planned, exposure = sampling_plan(available, budget, temperature)
    print("T=", temperature)
    print("ratio=", {key: round(value, 3) for key, value in ratio.items()})
    print("exposure=", {key: round(value, 2) for key, value in exposure.items()})
~~~


> **代码状态：静态审阅，未运行（2026-10-29）**

当温度升高时，低资源桶的比例会上升，但 exposure 也可能迅速变大。采样策略的正确评价不是“低资源语言占比是否上升”，而是“它是否在目标任务上带来收益，且没有用重复和高资源退化换来这个收益”。

### 30.7 配比实验的控制变量

比较两种数据配比时，固定模型结构、总训练 token、学习率计划、batch、tokenizer 和评估协议。至少保留原始比例、目标驱动比例和温度采样三个候选；若要判断课程顺序，再加一组阶段混合，但让总 token 和学习率随训练进度的曲线相同。否则比例变化会和计算量、优化进度或 tokenizer 长度混在一起。

手算温度采样能看清“比例变平”为什么会增加重复。三个桶原始占比为 `[0.80,0.15,0.05]`，取温度指数 `α=0.5`，未归一化权重为 `[√0.80,√0.15,√0.05]≈[0.894,0.387,0.224]`，归一化后约为 `[0.594,0.257,0.149]`。若可用量分别是 800、150、50 单位，总预算 1000，则计划曝光约为 `594/800=0.74`、`257/150=1.71`、`149/50=2.98` 次。低资源桶从 5% 升到约 15%，同时被重复近三遍；是否值得要看 fresh 验证和其他桶的退化。

2025 年 NeurIPS 的 Scaling Laws for Optimal Data Mixtures 用若干小规模实验拟合目标域 loss 与模型规模 $N$、训练 token $D$ 和领域权重向量 $h$ 的关系；$h_j$ 表示第 $j$ 个领域的训练占比，非负且总和为 1。随后为指定目标和预算优化 $h$。它在 LLM、神经机器翻译和视觉模型设置中做验证，LLM 实验最大约 7B 参数、150B token；这是一种把目标与预算写进 mixture 选择的方法，不是可脱离语料和预训练目标使用的通用比例。其拟合假设不含数据重复，也不直接预测下游任务能力。

另一项 NeurIPS 2025 研究发现，在作者的训练 recipe 中，经过激进过滤的 DCLM-baseline（RefinedWeb 中按 FastText 选出的前 10%）重复训练最多 10 个 epoch，可以在 1B、3B、7B 和 12.6B 模型设置中胜过只训练一轮、数据集大约大十倍的 RefinedWeb。这个结果依赖重复次数与 recipe 的配合，例如作者随重复次数增加权重衰减，近似按重复次数平方根调整；论文也明确指出不能据此宣布 DCLM 在所有目标上都优于 RefinedWeb。两项工作共同说明应把目标域、有效文档覆盖和重复曝光纳入小规模校准，而不是照抄一个数据比例。[Scaling Laws for Optimal Data Mixtures](https://proceedings.neurips.cc/paper_files/paper/2025/hash/bc1d640f841f752c689aae20b31198c1-Abstract-Conference.html)、[Datasets, Documents, and Repetitions](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2aba6ec20299931d46ddeefd5ddcb442-Abstract-Conference.html)。

若采用课程学习，可比较“前 70% 基础网页与代码、后 30% 困难数学”与全程固定混合。前者数学提高时，先检查同一 learning-rate schedule 下的分桶 loss、fresh 数学题和基础能力回放；不能把后段低学习率的作用误认成课程效果。预先写好停止条件：例如连续三个评估周期目标域收益低于 0.5 个百分点、重复曝光超过 3 次且通用域退化超过容忍值时，暂停加权并检查样本质量。阈值应由项目目标设定，这组数只作演示。

### 30.8 采样表如何对应真实训练 token

抽样概率、文档数和最终 token 占比不是一回事。设网页桶有 100 篇、每篇 900 token，数学桶有 100 道短题、每题 100 token；各随机抽 50 条，看起来样本数各占一半，实际却是网页 45,000 token、数学 5,000 token，即 90% 对 10%。截断、packing、去重后长度和 padding 还会继续改变有效更新量。报告至少保留样本数、抽样概率、实际输入 token、有效目标 token、截断量和重复曝光，并按最终 token 重新计算 mixture。

平均曝光也不是每个样本都恰好出现同样次数。若把 80B 个可用 token 理想化为等概率独立抽取，并训练 100B token，平均曝光是 1.25 次，但至少出现过一次的比例约为 `1−e^(−1.25)≈71.3%`；其余仍未被采到，部分已见 token 则出现多次。真实采样以文档和序列为单位，偏差可能更大，因此曝光倍数应结合簇大小和样本级分布一起看。

### 30.9 质量与合规门槛不能折算成配比权重

预算优化只在候选样本通过硬门槛后进行。网页来源、代码许可、个人信息、凭证和合成答案错误需要各自的审查状态；把这些风险折进一个平均质量分，可能让高任务分抵消未处理的许可或隐私问题。若某桶的质量尚未核验，先补证据和抽查，再决定是否提高它的采样权重。

## 31. 合成数据：从生成样本到可验证训练信号

### 31.1 为什么需要合成数据

高质量人类数据在数学、代码修复、工具调用、低资源语言和安全拒答等领域往往稀缺。强模型可以生成问题、答案、解释、错误样例、测试和多轮轨迹，让数据生产从被动收集变成主动构造。

合成数据有三个独特价值：可以定向补足能力缺口；可以控制格式和难度；在代码和数学场景中可以配合执行器或验证器筛选。它也有三个结构性风险：教师模型的错误会被复制，数据风格会收窄，生成模型和学生模型之间的分布差异会被掩盖。

### 31.2 合成数据的四种形态

1. **重写与扩展**：把一条高质量样本改写成多种表述，增加语言和格式多样性；
2. **教师示范**：由更强模型生成指令、回答、推理过程或工具轨迹；
3. **可验证任务**：生成题目、代码和测试，由执行器筛选；
4. **对抗与修复**：生成错误回答、攻击提示、失败轨迹，再构造修复或拒答数据。

不同形态需要不同验证器。可执行代码适合测试与静态分析；代数题可用符号或数值校验；事实问答可能需要检索来源或人工核验；偏好数据则不能只用教师模型自评。

### 31.3 生成—验证—过滤—去重—训练闭环

一个稳健的合成数据管线不是“调用大模型生成 N 条”。它更像一个闭环：

~~~text
定义缺口 -> 生成候选 -> 外部验证 -> 质量与安全过滤
    -> 去重与难度分层 -> 混入训练 -> 评估错误
    -> 根据错误重新设计生成条件
~~~


生成记录应包括 teacher 版本、提示模板、随机种子、工具版本、验证结果和失败原因。只有保留这些字段，后续才能判断某一批合成数据是提升了能力，还是只让模型适应了一个模板。

### 31.4 为什么“推理过程”不能直接当真值

模型生成的解释可能与最终答案一致，也可能是事后编造的合理化文本。即便答案正确，过程也可能包含错误步骤；即便过程看起来严谨，最终结论也可能错误。因此过程文本的价值要和可验证结果分开记录。

在训练 reasoning 数据时，可区分：最终答案是否可验证、每一步是否可验证、过程是否与答案一致、过程是否包含不必要的模板话术。对于不可验证的开放问题，应该降低自动生成标签的权重，增加人工抽样与独立模型复核。

### 31.5 分布坍缩与错误放大

若学生模型反复训练模型生成的数据，数据分布可能逐渐丢失长尾和多样性，错误模式也会被循环放大。[Shumailov 等人的研究](https://www.nature.com/articles/s41586-024-07566-y)展示了递归使用生成数据可能造成的分布退化；具体程度仍取决于数据混合比例、真实数据保留量、生成质量和任务，不能据此推断所有合成数据都会导致坍缩。

工程上可采取：保留足量人类数据；限制同一 teacher 风格的占比；按 semantic cluster 限制重复；引入不同 teacher 或不同采样策略；在 fresh 与人工数据上验证；为每个合成批次保留回滚能力。合成数据应该提高单位 token 的有效学习信号，而不是替代所有真实分布。

### 31.6 一个教学用的合成数据筛选器

下面的 Python 3.12 标准库 demo 输入一条带验证、来源、许可、隐私和安全标志的候选记录，输出是否接受；硬门槛必须全部通过，质量分再做软阈值筛选。代码只展示策略组合，不验证这些标志本身是否由可靠流程产生。

~~~python
import math


def accept(candidate):
    required_flags = (
        "format_ok", "answer_verified", "safety_ok", "provenance_ok",
        "license_ok", "privacy_ok",
    )
    if not isinstance(candidate, dict):
        raise TypeError("candidate must be a dictionary")
    if any(not isinstance(candidate.get(key), bool) for key in required_flags):
        raise ValueError("all hard-gate fields must be present booleans")
    quality = candidate.get("quality")
    if isinstance(quality, bool) or not isinstance(quality, (int, float)):
        raise TypeError("quality must be a real number")
    try:
        quality = float(quality)
    except OverflowError as error:
        raise ValueError("quality must be a finite score") from error
    if not math.isfinite(quality) or not 0 <= quality <= 1:
        raise ValueError("quality must be finite and within [0, 1]")
    return (
        all(candidate[key] for key in required_flags)
        and quality >= 0.8
    )


candidates = [
    {"id": "a", "format_ok": True, "answer_verified": True, "safety_ok": True, "provenance_ok": True, "license_ok": True, "privacy_ok": True, "quality": 0.91},
    {"id": "b", "format_ok": True, "answer_verified": False, "safety_ok": True, "provenance_ok": True, "license_ok": True, "privacy_ok": True, "quality": 0.97},
    {"id": "c", "format_ok": True, "answer_verified": True, "safety_ok": False, "provenance_ok": True, "license_ok": True, "privacy_ok": True, "quality": 0.95},
    {"id": "d", "format_ok": True, "answer_verified": True, "safety_ok": True, "provenance_ok": True, "license_ok": False, "privacy_ok": True, "quality": 0.99},
]
accepted = [item for item in candidates if accept(item)]
print("accepted:", [item["id"] for item in accepted])
~~~


> **代码状态：静态审阅，未运行（2026-10-29）**

真实系统的 `answer_verified` 不能由同一个生成模型自报。它应来自执行器、独立证明器、检索来源、规则检查或人工标注；对不同任务，应记录验证器的覆盖率和误判率。

### 31.7 合成数据的配比与消融

对合成数据最有信息量的实验通常不是比较“生成一百万条”和“生成两百万条”，而是比较同一训练预算下的真实数据比例、合成数据比例、验证器强弱和多样性控制。至少保留：真实数据 baseline、低比例合成、较高比例合成、未经严格验证的合成和严格验证的合成。

评估要包含训练分布内任务、fresh 任务、长尾任务和错误鲁棒性。若只在合成题上提升，不能说明真实能力提升；若 overall loss 降而长尾下降，可能是分布坍缩；若代码测试通过率升而安全扫描变差，需要把安全指标作为独立约束。

### 31.8 证据边界

筛选器只执行预设门槛，不证明输入标志真实：候选 `a` 即使质量分为 0.91，只要 `license_ok=False` 就会被拒绝。门禁本身的漏检率还要通过分层抽查估计；答案验证应来自相对独立的执行器、证明器、检索证据或人工审核。

### 31.9 把筛选结果变成可审计的误差率

从 100 条候选开始，执行检查通过 80 条、安全过滤留下 72 条、来源与许可检查留下 68 条，最后保留 54 条；每个阶段都应记录分母和拒绝原因。若从这 54 条随机抽查发现 6 条答案错误，表面精度约为 `(54−6)/54=88.9%`，但不能代表所有任务切片。估计误收率时也要报告不确定性：例如从 5,000 条通过样本抽 200 条、发现 10 条错误，点估计是 5%，Wilson 近似 95% 区间约 2.7%—9.0%；即使 100 条抽查均无错，95% 上界仍约 3%。应按来源、任务、teacher 和风险分层抽样，并在生成器或过滤器升级后复查。

### 31.10 生成难度要落在学生可学习的范围

EMNLP 2025 的正式论文在其预训练设置中比较自然网页、改写合成文本、教材式合成及其混合；研究覆盖 1,000 多个模型变体和超过 100,000 GPU 小时。较大数据预算下，约三分之一改写文本加三分之二自然网页文本，在特定对照中可快约 5—10 倍达到相同验证 loss；单独使用改写文本并未普遍更快，教材式文本单独训练在若干下游域也有退化。结果来自论文特定的改写方式、单轮训练、模型和预算，不能把 1/3 当成通用比例，也不能外推到所有合成数据类型。[Demystifying Synthetic Data in LLM Pre-training](https://aclanthology.org/2025.emnlp-main.544/)。

样本正确也不代表目标学生能从中学到东西。teacher 生成的证明或工具轨迹若远超学生已有能力，可能只增加难以拟合的 token；过于简单、模板化的样本又可能只教会格式。可按难度分层做小规模训练，比较各层在 fresh 任务上的增益，并保留基础数据回放，观察收益和遗忘。生成量、正确率和学习价值是三种不同统计，不能互相替代。

一个直接的难度实验可以把同一批任务分成基础、可学习和超出当前能力三档，例如分别让 teacher 提供 1000 条样本。用相同 token 预算训练后，记录学生在各档 fresh 变体上的准确率，并比较加入合成样本前后的基础任务回归。如果可学习档提升 8 个百分点、基础档下降 1 个点，而超难档只提升 1 个点却消耗一半训练 token，就说明样本筛选应按边际收益调整，而不是按 teacher 输出的自信度排序。数字是教学假设，不是普遍的难度配方；真实实验还要控制样本长度与重复率，避免“更难”其实只是更长。

还要区分教师覆盖和学生覆盖：teacher 能解题只说明候选值得生成，不能说明学生能从轨迹中学习。可以在保留相同任务结构的前提下缩短证明、拆分工具步骤或增加中间检查点，再比较单位训练 token 的迁移收益。若较短轨迹效果相同，冗余解释只增加训练成本；若移除某步后错误率升高，那一步可能承载关键监督。这样才把合成数据从“量产内容”变成可以检验的信息设计。


## 32. Scaling Law：用实验曲线规划参数、数据与计算

### 32.1 Scaling Law 研究的对象

Scaling Law 讨论的是模型规模、训练数据、计算量与某个可测指标之间的经验关系。[Kaplan 等人](https://arxiv.org/abs/2001.08361)和 [Hoffmann 等人](https://arxiv.org/abs/2203.15556)分别在各自公开的模型、数据和训练设置下研究了这类关系。最经典的指标是验证集上的 next-token cross-entropy loss。它不是“模型越大越好”的口号，而是一个预算规划工具：在可用算力下，参数量和训练 token 应如何分配，何时继续扩大模型，何时应该改善数据或 recipe。

常用变量包括模型参数量 N、训练 token 数 D 和训练计算量 C。对于 dense Transformer，可用非常粗略的近似表达：

~~~math
C\approx kND
~~~


其中 k 取决于架构、前向反向实现、序列长度、精度和是否统计通信。这个式子只用于数量关系，不可直接当作某硬件上的实际 FLOPs 计数。

作一个数量级估算：若 dense 模型有 8B 参数，训练 200B token，并暂取 `k≈6`，则 `C≈6×8×10^9×2×10^11=9.6×10^21 FLOPs`。这个数字不等于实际墙钟时间；长序列注意力、padding、通信和硬件利用率会让真实训练成本偏离该粗估，MoE 还要分清总参数与激活参数。

这里的训练 token 数 D 也不等于去重后的唯一 token 数 U。两个 run 都处理 200B token，一个来自 200B 个不同 token，另一个只覆盖 100B 后又重复一遍，虽然账面 D 相同，信息覆盖和记忆风险不同；记录 D/U、文档去重簇和各域 exposure，才能解释曲线中的数据项。

### 32.2 幂律形式与边际收益

经典工作常用如下经验形式描述损失随规模下降：

~~~math
L(N,D,C)\approx L_\infty+aN^{-\alpha}+bD^{-\beta}+cC^{-\gamma}
~~~


`L_infty` 是不可约或近似地板，α、β、γ 是由数据、架构和实验范围拟合出的指数。规模扩大带来收益，但边际收益会下降：从 N 增长到 2N 的改善，不会永远与再增长相同。

初学者应把幂律看成一条向下弯的经验曲线；进阶者要记住它只在拟合范围和 recipe 稳定时有效。更换 tokenizer、数据质量、上下文长度、优化器、架构或后训练目标，都可能改变曲线参数。

### 32.3 Chinchilla 的预算启示

固定训练计算量时，只增大参数而不增加 token，会得到 undertrained model：参数容量没有被足够数据训练。Chinchilla 研究强调模型大小和训练 token 应共同增长，计算最优点通常不同于“把最大模型训练很少步”。

把固定预算写成 C=kND，若把 N 增大，D 就必须相应减少；但 loss 中参数项与数据项的下降速度不同，所以存在一个使两类误差平衡的近似选择。实际比例取决于硬件、数据质量、训练目标和是否要长期复用模型，不能把某篇论文的具体比例机械复制到任何项目。

### 32.4 数据价值改变曲线

两个训练 run 即使 N、D、C 完全相同，若一个使用高质量、去重和目标匹配的数据，另一个使用低质量重复数据，它们的有效损失和下游能力仍会不同。更合理的预算账本应把 token 数拆成有效 token 与重复、污染、低质量 token。

可以用一个教学近似表示有效数据量：

~~~math
D_{\mathrm{eff}}=\sum_{x\in D}w(x),\qquad 0\le w(x)\le 1
~~~


高质量、目标相关且未重复的样本 w 接近 1；低质量或高度重复样本 w 较小。`D_eff` 不是公开论文中的统一标准，而是工程上帮助解释“相同 token 数为何不同”的记账概念。

### 32.5 外推为什么会失败

从 100M、300M、1B 模型外推到 70B 需要假设：数据分布不变，优化器和学习率缩放有效，训练稳定性不发生突变，评估任务与 loss 有稳定关系，且高质量数据不会耗尽。这些假设中的任意一个被破坏，曲线都可能系统性偏移。

MoE 还引入 total parameters 与 activated parameters 的区别；长上下文会改变 token 的计算和位置分布；多模态把图像、音频、视频 token 加入预算；后训练和推理时计算则不在经典预训练曲线中。因而公开 scaling curve 更适合作为初始方向，真正预算决策要用内部小规模实验校准。

### 32.6 loss、能力和系统成本的三条曲线

训练中至少要画三类曲线：验证 loss 随计算量变化，目标任务成功率随计算量变化，单位成功任务成本随计算量变化。第三条很重要：模型更大或 reasoning 更长可能提高准确率，却让延迟和成本超过产品承受范围。

如果模型 B 的 loss 更低，但代码任务需要更多采样才能通过；模型 A 的 loss 稍高，却一次生成就能通过测试，那么面向交互式代码助手时 A 可能更划算。Scaling Law 不应只优化一个离线标量。

### 32.7 test-time compute 与系统级扩展

现代系统还可以在推理阶段增加采样、搜索、验证、代码执行或工具调用。可以把最终成功率抽象为：

~~~math
S=f(N,D,C_{\mathrm{train}},C_{\mathrm{test}},Q,H)
~~~


其中 Q 表示数据质量与后训练，H 表示工具、检索、验证器和 harness。这个函数没有统一闭式形式，但它提醒我们：更大预训练模型只是系统能力的一项来源。

### 32.8 一个外推失效的教学实验

下面的 Python 3.12 标准库示例输入正的规模因子和有限的曲线参数，输出构造的 loss 曲线；它演示 recipe 或数据分布发生变化时，外推曲线可能整体偏移。

~~~python
import math


def finite_number(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def predicted_loss(scale, floor=1.0, amplitude=3.0, exponent=0.22):
    scale = finite_number("scale", scale)
    floor = finite_number("floor", floor)
    amplitude = finite_number("amplitude", amplitude)
    exponent = finite_number("exponent", exponent)
    if scale <= 0 or floor < 0 or amplitude < 0 or exponent <= 0:
        raise ValueError("require scale > 0, floor/amplitude >= 0 and exponent > 0")
    try:
        result = floor + amplitude / (scale ** exponent)
    except OverflowError as error:
        raise ValueError("scale and exponent exceed the demo's numeric range") from error
    if not math.isfinite(result):
        raise ValueError("predicted loss is not finite")
    return result


def shifted_loss(scale, penalty=0.18):
    penalty = finite_number("penalty", penalty)
    if penalty < 0:
        raise ValueError("penalty must be non-negative")
    result = predicted_loss(scale) + penalty
    if not math.isfinite(result):
        raise ValueError("shifted loss is not finite")
    return result


small = [1, 2, 4, 8]
large = [16, 32, 64]
print("small-scale observations")
for scale in small:
    print(scale, round(predicted_loss(scale), 4))
print("extrapolation")
for scale in large:
    print(scale, round(predicted_loss(scale), 4))
print("after a data or recipe shift")
for scale in large:
    print(scale, round(shifted_loss(scale), 4))
~~~


> **代码状态：静态审阅，未运行（2026-10-29）**

使用 Python 3.12 标准库直接运行，代码输入正的 `scale` 及有限的曲线参数，输出构造曲线上的 loss 数值；例如 `predicted_loss(1)=4.0`，`predicted_loss(64)` 约为 `2.202`。负尺度、非有限参数和无法表示的幂运算会被拒绝。这个实验没有拟合真实模型，它只说明：小规模曲线可以非常平滑，但大规模阶段若数据、架构或训练 recipe 发生系统变化，真实结果会整体偏离。正确做法是把中等规模结果回填模型，而不是把外推误差归因于“Scaling Law 完全无效”。

### 32.9 预算决策的顺序

预算分析从目标约束开始：先写清目标能力、最大延迟和总计算预算，再用多个小模型估计 loss 与任务曲线，同时记录数据质量、重复曝光和训练稳定性。接着比较增加参数、增加有效 token、改善数据、后训练或推理时计算的边际收益；最后把通信、存储、服务成本和上线周期纳入同一决策。清洗投入也应计价：更严格筛选若减少低价值 token，就要与其人力、许可核验和数据覆盖成本一起比较。

近期研究把数据质量直接放进 scaling 关系。ICLR 2026 的 *Scaling Laws Revisited* 用可控噪声和覆盖变化研究合成的翻译与自回归建模设置；另一项 ICLR 2026 实证工作比较去重、启发式过滤和 LLM 改写，在其 100M—8B 模型、100M—200B token 的实验范围内，发现数据干预会改变拟合系数和指数，并影响计算最优的 token/参数关系。这些结果支持“数据清洗可能改变预算曲线”，不提供任意真实语料的现成质量分或通用配方。新项目仍需在自己的数据、tokenizer 和目标任务上做小规模校准。[Scaling Laws Revisited](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d0c80a0c294a16190c8904b9809c5fba-Abstract-Conference.html)、[How Text Quality Interventions Reshape Neural Scaling Laws](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6544c0b53cc1ea51d0c7086b536830a8-Abstract-Conference.html)。

把一条构造曲线手算一次：设 $L(N)=1+3N^{-0.22}$，当 $N=1$ 时 $L=4$；当 $N=16$ 时，$16^{0.22}\approx1.84$，所以 $L\approx1+3/1.84\approx2.63$。如果数据或 recipe 改变带来 0.18 的系统偏移，同一规模的 loss 约为 2.81。拟合曲线可以描述原条件，却不会自动预知偏移；应留出一个中型 run 检查外推，并记录有效 token、训练时长、硬件和验证口径。若偏差持续同向，先查数据耗尽或 recipe 改变，再调整拟合，而不是把一次预测失准解释成 scaling law 毫无价值。

## 33. Training loss、validation loss 与下游能力

### 33.1 三个指标回答三个问题

训练 loss 回答“模型在参与更新的 batch 上是否在优化”；validation loss 回答“模型在相似但未参与更新的数据上是否泛化”；下游任务回答“模型在一个具体任务协议中是否成功”。三者有相关性，却不等价。

自回归语言模型的 token-level 交叉熵为：

~~~math
L=-\frac{1}{M}\sum_{t=1}^{M}\log p_\theta(x_t\mid x_{1:t-1})
~~~


训练时通常用 teacher forcing，把真实前缀提供给模型。真实生成时，前一步可能使用模型自己的输出，因此错误会沿序列传播；解码温度、top-p、停止条件和上下文模板也会改变下游结果。

### 33.2 Perplexity 的含义与边界

若 loss 使用自然对数，perplexity 定义为：

~~~math
\mathrm{PPL}=\exp(L)
~~~


它可以给出每个位置平均不确定性的直觉，但不能把它解释成模型真的在固定数量的候选词中选择。PPL 依赖 tokenizer、验证集、上下文切分和平均口径。不同 tokenizer 的 token 粒度不同，跨 tokenizer 直接比较 PPL 通常没有可靠意义。

### 33.3 loss 下降而任务不涨的原因

第一，训练和任务分布不一致。大量网页 token 的改善可能降低 overall loss，却不改变数学、代码或工具调用。

第二，token-level 平均和 task-level 成败不同。一道代码题只要一个关键 token 错就可能无法编译；一道数学题中间一步错，最终答案就错。平均 loss 的小变化不一定越过任务成功的阈值。

第三，后训练和解码协议不同。base model 的语言建模能力可能很好，却没有学会遵循指令；同一个 checkpoint 在 greedy、temperature sampling 和 verifier rerank 下也会得到不同的结果。

第四，评估存在污染、噪声或模板过拟合。模型在 benchmark 上提升并不自动证明真实能力提升。

### 33.4 overall loss 掩盖局部变化

如果验证集有 80% 网页、10% 代码、2% 数学，数学能力提升会被总平均稀释。应同时记录 web、code、math、multilingual、long-context 等分桶 loss，并与对应下游任务配对。

一个更可解释的评估表如下：

| 现象 | 可能解释 | 下一步检查 |
| --- | --- | --- |
| 总 loss 降，所有任务涨 | 训练整体健康 | 检查是否有污染 |
| 总 loss 降，代码降 | 数据配比或遗忘 | code loss、代码数据版本 |
| 总 loss 不变，数学涨 | 局部收益被平均 | math bucket、样本量 |
| loss 好，聊天差 | 缺少后训练或模板错 | SFT mask、对话评估 |
| benchmark 涨，fresh 不涨 | 污染或过拟合 | 时间切分、hidden eval |

### 33.5 训练曲线的典型形态

健康训练通常表现为训练和验证 loss 的滑动平均总体下降；欠拟合表现为两者都高且下降慢；过拟合常见于训练 loss 继续降而验证 loss 持平或上升；不稳定则表现为尖峰、Inf、NaN 或多个 rank 的 loss 分叉。

但 web-scale 预训练不应机械使用“小数据过拟合图”。高质量数据被重复、领域配比变化、验证集污染、后期引入困难数据，都可能让曲线产生类似过拟合的形状。要结合数据版本和分桶指标解释。

### 33.6 统计口径必须一致

loss 可能按 token 平均、按序列平均，可能包含 padding，也可能只计算回答部分。预训练通常计算所有有效目标 token；SFT 可能把 prompt label 设为 -100，只计算 answer。两种 loss 的数值不能直接比较。

继续预训练、SFT、偏好优化和 RL 的 loss 也不是同一个对象。模型选择时应保存 mask 规则、tokenizer、数据分布和 reduction 方式。

### 33.7 交叉熵与 PPL 的可运行示例

代码依赖 PyTorch 2.x，输入 logits 的 shape 为 `[batch, length, vocab]`、token id 的 shape 为 `[batch, length]`，输出 token 平均交叉熵及其指数 PPL。标签右移后才与预测位置对齐；若传入 `ignore_index`，相应目标不计入 loss；类别索引和忽略规则见 [PyTorch `CrossEntropyLoss` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。

~~~python
import math
import torch
import torch.nn.functional as F


def lm_loss_and_ppl(logits, input_ids, ignore_index=None):
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("logits must be [batch, length, vocab] and ids must be [batch, length]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids must have matching batch and length")
    if input_ids.size(1) < 2 or logits.size(-1) <= 1:
        raise ValueError("sequence length must be at least two and vocab must have at least two classes")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must use a floating-point dtype")
    if input_ids.dtype != torch.long:
        raise TypeError("input_ids must use torch.long class indices")
    if logits.device != input_ids.device:
        raise ValueError("logits and input_ids must be on the same device")
    if (ignore_index is not None
            and (not isinstance(ignore_index, int) or isinstance(ignore_index, bool))):
        raise TypeError("ignore_index must be an integer or None")
    if not torch.isfinite(logits).all().item():
        raise ValueError("logits must be finite")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    valid = torch.ones_like(shift_labels, dtype=torch.bool)
    if ignore_index is not None:
        valid = shift_labels != ignore_index
    if not valid.any().item():
        raise ValueError("at least one target token must remain after masking")
    if ((shift_labels[valid] < 0) | (shift_labels[valid] >= logits.size(-1))).any().item():
        raise ValueError("target token ids are outside the vocabulary")
    kwargs = {"reduction": "mean"}
    if ignore_index is not None:
        kwargs["ignore_index"] = ignore_index
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        **kwargs,
    )
    if not torch.isfinite(loss).item():
        raise ValueError("loss must be finite")
    try:
        ppl = math.exp(loss.item())
    except OverflowError as error:
        raise OverflowError("perplexity exceeds the Python float range") from error
    return loss, ppl


torch.manual_seed(0)
logits = torch.randn(2, 5, 8)
input_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 2, -100, 6]])
loss, ppl = lm_loss_and_ppl(logits, input_ids, ignore_index=-100)
print("loss=", round(loss.item(), 4), "ppl=", round(ppl, 4))
~~~


> **代码状态：静态审阅，未运行（2026-10-29）**

在固定随机种子的示例中，函数返回有限 loss 和 `exp(loss)`；具体浮点末位依赖 PyTorch 版本。`shift_logits` 与 `shift_labels` 的错位是语言模型 loss 中最常见的实现细节之一。若把同一位置的 logits 与同一位置的 input 当作目标，就会改变任务定义；若 padding 没有设为 ignore index，短样本会被无效位置污染。该示例没有实现序列级加权、分布式 reduction 或模型训练评估。

### 33.8 checkpoint 选择不能只看最低 loss

base model 通常重视 validation loss 和分桶 loss；代码模型要看可执行测试；聊天模型要看指令遵循、偏好和安全；推理模型要看验证器通过率、采样成本和延迟。最低 overall loss 只是候选条件之一，不能替代目标任务选择。

用一个小例子观察总体平均如何遮住局部收益。若验证桶权重为网页 80%、代码 10%、数学 10%，旧模型的分桶 loss 是 `[1.00,1.20,1.50]`，总 loss 为 0.8×1.00+0.1×1.20+0.1×1.50=1.07。新模型变为 `[0.98,1.18,1.50]`，总体是 1.054，只改善约 1.5%；代码变化几乎看不出来，数学则完全没变。若模型只在数学上从 1.50 降到 1.30，总 loss 也只下降 0.02，约 1.9%，但目标数学任务可能有明显价值。反过来，总 loss 小幅下降不保证目标任务变好，需把同域 loss 与下游任务成对观察。

### 33.9 统计口径决定一条 loss 曲线代表谁

按 token 平均和先按序列平均，会给长短样本不同权重。设短序列有 2 个有效 token、loss sum=2，长序列有 8 个有效 token、loss sum=4；按 token 汇总是 `(2+4)/(2+8)=0.60`，先算每条序列再平均则是 `(2/2+4/8)/2=0.75`。前者让长序列贡献更多，后者让两条样本等权。padding、response-only mask 和 packed boundary 也会改变分母，比较 run 时必须固定并报告 reduction 规则。

跨 tokenizer 比较 PPL 也有同类问题。同一句话若在 A 中是 20 token、在 B 中是 30 token，逐 token 平均的预测位置和分母不同，数值不能直接当作能力排名。可以固定 tokenizer 比较 checkpoint，或另报按字符/byte 定义的辅助指标；任何重标度都必须写出分母和单位，不能把不同训练目标伪装成同一量。

预测位置也必须右移一格。输入 token 是 `[A,B,C]` 时，第一个位置的 logits 预测 B，第二个位置预测 C，而不是让 A 预测自身。若模型给出 $P(B\mid A)=0.8$、$P(C\mid A,B)=0.25$，两项 NLL 分别为 $-\ln0.8\approx0.223$ 与 $-\ln0.25\approx1.386$；总 NLL 约 1.609，平均 loss 约 0.805 nat/token，PPL 为 $e^{0.805}\approx2.24$。若把 logits 和标签错位，分数可能仍是有限数，却对应了另一个任务。

分桶总分还要固定每桶权重。假设网页 80%、代码 10%、数学 10%，若将数学权重临时改为 20% 并按比例缩减网页，overall loss 会变，即使每桶 loss 原封不动。这个变化来自评估分布而非模型更新；报告 checkpoint 前要固定验证清单和聚合权重，同时保留分桶原始值，才能区别模型变化与统计口径变化。

loss 的小数差也要回到同一分布上解释。若固定 tokenizer 和验证集后平均 NLL 从 1.0 降到 0.9 nat/token，几何平均的目标 token 概率比值约为 $e^{0.1}=1.105$，即概率质量提升约 10.5%。这仍不等于任务成功率提高 10.5%；它只描述被评估 token 的平均条件概率，最终能力还需相应任务指标验证。

### 33.10 loss 与任务表现相反时做最小反事实

当 loss 与下游指标方向相反，先按语言、长度、代码符号、数学难度和有效 token 数切片，检查整体均值是否掩盖了困难尾部。再固定 checkpoint 只改解码，或固定数据只改训练目标，每次改变一项，观察预先指定的指标。若训练 loss 与 benchmark 同时变好，还要排除重复曝光或测试污染造成的共同变化；污染检测和 fresh eval 的设计见第 37 讲。

任务成功率还受采样、verifier、工具和评测器影响。比较模型时应将 base 单次输出与多次采样筛选分开，并记录成本；这类推理预算口径由第 32、87—96 讲展开。发布报告则绑定 checkpoint、评估代码、数据版本和模板，避免把评估器或运行配置变化误认为模型能力变化。

response-only loss 也必须只除以有效回答 token。若一条样本有 10 个目标 token、loss 总和为 12 nat，正确平均是 `12/10=1.2`；若把 22 个 prompt 或 padding 位置也塞进分母，会得到 `12/32=0.375`，看起来 loss 大幅下降，模型却没有变。mask 分母应按当前 batch 的有效目标数计算，并在训练和验证中采用同一规则。

## 34. 训练稳定性：loss spike、NaN 与梯度爆炸

### 34.1 先区分现象

loss spike 是某些 step 的 loss 突然升高；loss divergence 是升高后持续失控；NaN 表示数值已经不是有限实数；Inf 表示溢出；gradient explosion 表示梯度范数或某层梯度突然变得极大。它们可能相互关联，但诊断路径不同。

短暂 spike 可能来自困难 batch、数据 shard 切换或正常波动；持续 spike 可能来自学习率、数据或 optimizer 状态；NaN 通常会污染参数和状态，不能把它当作普通噪声继续训练。

### 34.2 学习率、warmup 与梯度裁剪

参数更新可粗略写成：

~~~math
\theta_{t+1}=\theta_t-\eta_t\,u_t
~~~


其中 η_t 是学习率，u_t 是优化器根据梯度和历史状态得到的更新。如果初始梯度统计不稳定而 η 一开始就很大，模型可能被推离正常区域。warmup 让学习率从较小值逐渐升到峰值，给参数、激活和 Adam moments 建立稳定统计。

全局梯度裁剪按范数缩放：

~~~math
g' = g\cdot\min\left(1,\frac{\tau}{\lVert g\rVert_2+\epsilon}\right)
~~~


它限制单次更新的极端幅度，但不能修复 label shift、attention mask、损失计算、数据乱码或错误的 checkpoint。裁剪阈值过小会长期压制有效梯度，过大则几乎不起作用。

### 34.3 数据路径导致的 spike

应把每个 step 的 loss 与 data shard、sample id、序列长度、领域、token 分布和特殊 token 比例关联起来。常见异常包括：空样本、超长截断错误、label 与 input 错位、全是 padding、乱码、极端重复字符串和错误的 attention mask。

若固定 seed 回放同一 batch 仍然复现 spike，优先检查数据和前向路径；若单卡回放正常而多卡异常，检查 reduce、batch 切分、通信和 rank-specific 数据；若只有恢复后异常，检查 optimizer、scheduler、scaler 和数据游标。

### 34.4 数值路径与模型结构

FP16 动态范围有限，attention logits、softmax、norm、loss 和梯度都可能溢出。Pre-LN 通常提供更直接的残差梯度路径，深层 Post-LN 则可能需要额外初始化或 residual scaling；MoE router 还可能出现专家负载极不均衡和局部激活异常。

数值检查应定位“第一个非有限值”出现的位置，而不是只在最终 loss 处报错。可以在关键层检查 activation max、均值、标准差和 finite 状态，在反向检查 per-layer grad norm 和参数范数。

### 34.5 分布式训练中的异常传播

一个 rank 先出现 NaN 后，all-reduce 可能把异常传播到所有 rank。只看 rank 0 日志会丢失最早的证据。应记录每个 rank 的 loss、grad norm、数据 shard、step time、通信等待和 finite 检查。

如果所有 rank 同时出现异常，优先检查全局学习率、模型结构和同步后的梯度；如果一个 rank 先异常，优先检查它的数据、硬件和分片状态。通信超时和 silent hardware error 也可能表现为 loss 断裂，不能把所有异常都归因于优化器。

### 34.6 NaN 排查顺序

1. 停止继续写入可能被污染的 checkpoint，保存 step、LR、batch、rank 和监控数据；
2. 从最近一个健康 checkpoint 恢复，而不是从 NaN 状态继续；
3. 回放异常前若干 step，确认是否可复现；
4. 检查 loss、logits、activation、gradient、parameter 和 optimizer state 的 finite 状态；
5. 检查数据、mask、label、序列长度和 tokenizer 输出；
6. 检查学习率、warmup、梯度裁剪、weight decay、Adam epsilon；
7. 检查 AMP、loss scaling、softmax、norm 和自定义 kernel；
8. 检查 rank、通信、分片和 checkpoint 恢复；
9. 在小规模环境重现修复，再恢复大规模训练。

### 34.7 稳定性监控面板

训练 loss 之外，建议监控 global grad norm、clipping ratio、parameter norm、update-to-weight ratio、activation max、attention logits max、loss scale、overflow count、skipped step、每 rank loss、序列长度分布和数据 shard。每项指标都要有历史基线，否则“异常”没有参照。

update-to-weight ratio 是更新尺度相对参数尺度的粗略读数。某层参数范数为 100、更新范数为 0.1，比例是 `0.1/100=0.001`；另一层参数范数仅 0.01、更新范数 0.001，比例则为 0.1。全局范数可能看起来正常，后一层却相对改变了十分之一。接近零的参数组会使这个比值不稳定，所以应按层或参数组观察，并结合绝对更新量和任务表现判断。

### 34.8 finite 检查与裁剪代码

输入是 PyTorch 2.x `model`、可供 `model(**batch)` 使用的 batch、optimizer，以及正的有限 `max_grad_norm`；输出为标量 loss 和裁剪前梯度范数。代码末尾用一个线性模型的平方误差演示一次更新，不是语言模型训练。AMP 下还必须先 unscale 再裁剪。

~~~python
import math
import torch
from types import SimpleNamespace


def train_step(model, batch, optimizer, max_grad_norm=1.0):
    if isinstance(max_grad_norm, bool) or not isinstance(max_grad_norm, (int, float)):
        raise TypeError("max_grad_norm must be a real number")
    try:
        max_grad_norm = float(max_grad_norm)
    except OverflowError as error:
        raise ValueError("max_grad_norm must be finite") from error
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    optimizer.zero_grad(set_to_none=True)
    output = model(**batch)
    loss = output.loss
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise ValueError("model output must contain a scalar loss tensor")
    if not torch.is_floating_point(loss):
        raise TypeError("loss must use a floating-point dtype")
    if not torch.isfinite(loss).item():
        raise RuntimeError("non-finite loss")

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    if not torch.isfinite(grad_norm).item():
        raise RuntimeError("non-finite gradient norm")
    optimizer.step()
    if any(not torch.isfinite(parameter).all().item()
           for parameter in model.parameters()):
        raise RuntimeError("optimizer produced non-finite parameters")
    return float(loss.detach()), float(grad_norm)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 1)

    def forward(self, features):
        prediction = self.projection(features)
        return SimpleNamespace(loss=prediction.square().mean())


torch.manual_seed(0)
model = ToyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss, grad_norm = train_step(
    model, {"features": torch.tensor([[1.0, 2.0], [2.0, 1.0]])}, optimizer
)
print("loss=", round(loss, 4), "preclip_grad_norm=", round(grad_norm, 4))
~~~


> **代码状态：静态审阅，未运行（2026-10-29）**

这里使用单卡 eager 模式、无 AMP 的 PyTorch 标量 loss；调用方须提供有限 batch，并在该函数报错后保留异常现场。AMP 中必须先把 scaled gradient unscale，再裁剪。生产训练还要避免在异常后无条件执行 optimizer.step，并保证最后一个健康 checkpoint 不被覆盖。

### 34.9 用手算看裁剪如何限制更新

设梯度 $g=[3,4]$，L2 范数为 $\sqrt{3^2+4^2}=5$。阈值 $\tau=1$ 时缩放系数是 $1/5$，所以 $g'=[0.6,0.8]$，新范数恰为 1；若范数突然升到 50，系数会变成 $1/50=0.02$。裁剪保留方向、限制单次幅度，但持续高触发率说明原始更新长期越界，可能有学习率、数据或梯度尺度问题。监控应同时保存裁剪前范数、裁剪后范数、触发比例和逐层范数。

全局裁剪把同一系数应用到所有参数；若异常集中在一层，其余层也会被缩小。分组裁剪能隔离极端张量，却会增加阈值和状态选择。应先看异常是全局尺度还是单层尖峰，再比较两种方案的触发率、任务质量和训练成本。裁剪不是清理坏样本、修复 mask 或恢复损坏 optimizer state 的替代品。

### 34.10 一次异常如何从现场走到恢复

发现 spike 后，先冻结最近健康 checkpoint，并保存异常 step 的样本 ID、有效 token 数、rank、学习率、scaler、optimizer step 和数据游标。用同一 checkpoint 重放同一 batch，记录 logits、loss、activation、分层 gradient norm 和参数更新第一次偏离的位置。若重复出现，沿首次异常张量追查；若只在一个 rank 触发，优先核对该 rank 的数据和设备状态；所有 rank 同时变化则检查共同 batch、调度和精度路径。不能只凭“降低学习率后恢复”断定根因就是学习率。

恢复动作要与故障状态匹配。非法输入先隔离样本并保留来源；仅 AMP scale 溢出时，可依既定策略跳过该次更新并观察频率；参数或 optimizer state 已有 NaN/Inf 时应停止，从最近健康状态恢复。多卡场景必须让所有 rank 对是否跳过更新作同一决定，并对齐 optimizer、scheduler、scaler 与 consumed-token 计数；本地梯度范数、聚合后的全局范数和分片范数的语义不同，通信顺序要与实现一致，详细梯度同步见[第四部分](04-大规模训练系统进阶.md)。

裁剪与聚合的次序会改变更新。设两个 rank 的梯度分别为 $g_1=[3,0]$、$g_2=[0,4]$，先求平均得到 `[1.5,2]`，范数为 2.5；全局按阈值 1 裁剪为 `[0.6,0.8]`。若各 rank 先按本地阈值裁剪，再平均，则得到 `[0.5,0.5]`，范数约 0.707。两种流程都“把梯度裁到 1 以下”，更新却不同；分布式实现必须按目标语义明确裁剪发生在归约前还是归约后。

还要区分难例和不稳定：孤立的一次高 loss 可能是合法难题，连续多个 step 上升且多个层的激活或梯度同时偏移，才更支持系统性故障假设。对均值和滑动平均保留原始点，避免平滑把尖峰抹掉；日志按实际 token 计量，而不只记 step，因为变长 batch 的每步工作量不同。修复后用固定小 batch 做回放，再跑短程目标域评估；只有异常消失且目标质量未回退，才能说明这次修复达到预期。

例如 100 次更新中有 90 次触发范数阈值，训练虽然没有 NaN，更新长期却被压缩；若只有 1 次触发，则可能只是一个难 batch。90% 不是通用报警线，但这个比例说明应继续检查学习率、异常样本和梯度尺度。任何阈值都要结合连续窗口、各层更新量及验证任务解释。


## 35. 混合精度：FP16、BF16 与 FP8 的数值取舍

### 35.1 浮点格式的两个维度

浮点数的指数位主要决定动态范围，尾数位主要决定相对精度。数值范围不足会 overflow 或 underflow；精度不足会让相近数被舍入到相同表示。训练中两种问题都可能影响收敛，但大模型首先需要防止 Inf、NaN 和梯度归零。

FP32 使用 32 bit，范围和精度都较好；FP16 使用 16 bit，节省显存但指数位少；BF16 也是 16 bit，却保留接近 FP32 的指数范围，尾数更短；FP8 进一步压缩到 8 bit，需要显式缩放和高精度累加。FP16 loss scaling 的训练动机和实验设置见 [Micikevicius 等人的论文](https://arxiv.org/abs/1710.03740)；框架中的 autocast、GradScaler 和更新次序见 [PyTorch AMP 指南](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)。

### 35.2 FP16 与 loss scaling

若某个真实梯度 g 太小，转换到 FP16 后可能下溢为 0。loss scaling 先把 loss 乘以 s：

~~~math
\widetilde L=sL,\qquad \nabla_\theta\widetilde L=s\nabla_\theta L
~~~


反向传播后再把梯度除以 s，恢复真实尺度。如果放大后出现 Inf，则跳过本次更新并降低 s；若连续稳定，可以逐步提高 s。动态 loss scaling 的目的不是提高模型目标，而是保护低精度梯度中的小数值。

AMP 的正确顺序是：autocast 前向，scale loss，反向，unscale 梯度，gradient clipping，optimizer step，更新 scaler。若在 unscale 之前裁剪，阈值对应的是放大后的梯度，监控和裁剪都失去原有含义。

### 35.3 为什么 BF16 常用于大模型

BF16 的指数位与 FP32 接近，因此大激活和梯度更不容易溢出，通常不需要 FP16 那样的动态 loss scaling。[Kalamkar 等人的 BFLOAT16 研究](https://arxiv.org/abs/1905.12322)比较了多类训练任务中的该数值格式；实际支持和操作语义还取决于设备与框架。它的尾数精度较低，所以并非“完全等同 FP32”；loss、归一化统计、reduction、optimizer state 和关键累加仍常保留 FP32。

在支持 BF16 的硬件上，BF16 往往是效率与稳定性的折中。旧硬件、特定 kernel 或精细数值任务可能不支持或不适合 BF16，因此 dtype 选择必须与设备和实现绑定。

### 35.4 Master weights 与 optimizer state

低精度参数更新可能丢掉很小的增量。例如参数约为 1.0，而更新量只有 1e-7，低精度表示可能无法保留。常见做法是用低精度权重做矩阵计算，同时维护 FP32 master weights 做 optimizer update。

AdamW 还需要一阶矩 m 和二阶矩 v。若它们用过低精度保存，更新统计可能产生更大误差。因此“模型用 BF16”不等于“所有状态都用 BF16”，显存规划必须把参数、梯度、master weights、optimizer states 和 activation 分开计算。

看一个刻意简化的低精度状态例子：Adam 的 $\beta_2=0.999$，某次梯度 $g=10^{-4}$，新二阶矩增量约为 $(1-0.999)g^2=10^{-11}$。它远小于 FP16 最小正 subnormal 约 $5.96\times10^{-8}$，若二阶状态直接以 FP16 累积就可能变成 0；偏差修正后分母也会与保留该状态时不同。真实 optimizer 通常会在更高精度保存或累积统计，这个构造例说明 optimizer state 有独立精度需求，不能从权重 dtype 推断。

### 35.5 FP8 需要一整套缩放协议

FP8 常见格式包括 E4M3 与 E5M2，分别在范围和精度之间取不同权衡。FP8 训练通常需要 per-tensor 或 per-channel scaling、amax 统计、delayed scaling 和 FP16/BF16/FP32 accumulation。

设一个张量为 x，缩放因子为 s，量化可以抽象为：

~~~math
\hat x=Q(sx),\qquad x\approx s^{-1}\hat x
~~~


s 太大可能溢出，太小会让大量值量化为 0；如果 x 的分布随训练变化，静态 s 很快失效。因此 FP8 不是简单把 dtype 字段从 FP16 改成 FP8，而是硬件、kernel、统计和训练 recipe 的组合。

### 35.6 哪些操作更谨慎

矩阵乘和线性层通常最能受益于低精度；softmax、norm、loss、reduction、gradient norm、optimizer update 和 all-reduce 累加更需要关注高精度。稳定 softmax 会先减去最大值：

~~~math
\mathrm{softmax}(x)_i
=\frac{\exp(x_i-\max_jx_j)}{\sum_k\exp(x_k-\max_jx_j)}
~~~


即使 dtype 合理，错误的 mask、过大的 attention score 或自定义 kernel 中的 accumulation dtype 也能造成 NaN。

例如 logits 为 `[1000,1001]` 时，直接计算 `exp(1000)` 会溢出；稳定写法先减最大值，得到 `[-1,0]`，再算 softmax，概率约为 `[0.269,0.731]`。减最大值不改变 softmax，因为分子、分母同乘一个常数；这个例子说明稳定公式处理的是动态范围，而不只是换更大的浮点类型。实现仍要核对 mask 和 accumulation dtype。

### 35.7 CPU 上观察范围和精度

依赖 PyTorch 2.x CPU；输入四个构造数值，输出每种 dtype 的最大有限值、机器精度、转换后的张量和有限值标志。该代码观察浮点表示，不测量设备吞吐或训练收敛。

~~~python
import torch


for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    info = torch.finfo(dtype)
    print(dtype, "max=", info.max, "eps=", info.eps)

values = torch.tensor([1.0, 1_000.0, 70_000.0, 1_000_000.0])
for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    casted = values.to(dtype)
    print(dtype, casted, torch.isfinite(casted).tolist())
~~~


> **代码状态：静态审阅，未运行（2026-10-29）**

这个实验能直观看到 FP16 更早达到有限值上限，而 BF16 的范围更接近 FP32；同时 BF16 的 `eps` 更大，说明它的有效精度更粗。具体硬件上的吞吐和 kernel 行为仍要用真实设备测量。

### 35.8 精度迁移的验证顺序

从 FP32 切换到 BF16、FP16 或 FP8 时，先在单卡小模型上对比 loss、grad norm 和 finite 状态，再做多卡短跑；之后比较 validation loss、下游任务、overflow、skipped steps、吞吐和显存；最后才在目标规模上长跑。只验证“程序能启动”不足以证明低精度没有伤害模型。

### 35.9 梯度范围与参数更新精度是两种不同问题

设真实梯度为 $g=10^{-8}$，小于 FP16 常见最小正 subnormal 约 $5.96\times10^{-8}$，直接转为 FP16 可能变成 0。若 loss scale $s=1024$，反向得到 $sg=1.024\times10^{-5}$，unscale 后再除以 1024 可回到 $10^{-8}$；若缩放产生 Inf，则该次更新应跳过并降低 scale。BF16 能表示这个量级，但尾数较短，不能因此消除舍入误差。

更新量太小也会在参数加法时消失。以参数值 1 为例，FP16 在该数量级的相邻可表示值间隔约为 $9.77\times10^{-4}$，BF16 约为 $7.81\times10^{-3}$；直接加 $10^{-5}$ 往往仍会舍入回 1。FP32 master weights 可保留小更新的累计结果，随后再生成低精度计算副本。这说明 loss scaling 处理梯度下溢，master weights 处理参数更新精度，两者解决不同环节。

### 35.10 显存账本要包括完整训练状态

以 7B 参数为例，若权重和梯度各占 2 字节，FP32 master weights 占 4 字节，Adam 的一阶、二阶状态各占 4 字节，则合计约 16 字节/参数：`7×10^9×16≈112 GB`（约 104.3 GiB）。其中 BF16 权重本身约 14 GB（约 13.0 GiB）。该估算不含激活、临时 buffer、通信副本和 padding；框架可能省略 master weights、分片状态或使用低精度 optimizer，因此要按实际配置重算，不能用权重占用代替完整训练显存。

FP8 的位宽降低可进一步省矩阵输入和权重存储，但 scale、累加精度、kernel 覆盖率及未压缩的 optimizer state 都会影响总节省。若理想权重从 BF16 的 2 字节降到 1 字节，只能说权重项减半，不能推出训练或服务显存也减半。应逐项列出参数、梯度、状态、激活和通信 buffer 的 dtype 与峰值。

### 35.11 用固定输入验证数值变化传到哪里

先取固定激活和权重，比较 FP32、BF16、FP16、FP8 的矩阵输出最大绝对差、相对差和非有限值；再沿真实模型的 attention、norm、MLP、logits 检查误差是否放大。FP8 还要记录 scale 更新规则、历史统计与张量格式。单个 kernel 的差异小，不等于多层训练轨迹不变，因此短程训练仍需比较 loss、梯度范数和目标任务。

训练 checkpoint 可能同时包含 FP32 master weights、低精度参数和 scale；发布时导出推理权重后，要在同一组输入上比较 logits、生成 token 和非有限值，并补测长序列及极端输入。报告绑定设备型号、软件栈、kernel 和各状态精度，再比较吞吐、峰值显存、跳过更新及目标任务；仅凭权重能加载或理论位宽降低，都不足以证明迁移成功。

比较容量时要确认工具显示的是 GB 还是 GiB，并为激活与临时 buffer 另留空间。数值范围也取决于输入、累加和输出 dtype：kernel 若以 FP32 累加再转回 FP16，中间累加和最终 cast 的溢出位置可能不同，所以报告不能只写模型权重格式。

FP8 的缩放粒度也会改变误差分布。设同一张量块中多数值落在 $[-1,1]$，却有一个离群值为 64；若用整个张量的单一 scale 覆盖离群值，常见值可能只占可表示区间很小的一段，量化误差会集中在小量级元素。分块或按通道 scale 可以改善局部动态范围，但会增加 scale 元数据、统计更新和 kernel 复杂度。可在固定输入上比较 global、per-channel 与 block scale 的最大误差、均方误差和 outlier 误差，再将差异沿层传播到 logits 与任务指标。这个实验不证明哪一种协议最好，只帮助定位误差来自格式本身还是缩放策略。


## 36. Checkpoint：保存的不只是模型权重

### 36.1 两种 checkpoint

用于推理或发布的 model checkpoint 通常包含模型权重、config、tokenizer 和生成配置。用于故障恢复的 training checkpoint 还必须包含优化器、调度器、随机状态、数据进度、混合精度状态和分布式元数据。

二者的目标不同：前者希望体积小、格式稳定、易部署；后者希望恢复后训练轨迹尽量接近中断前。把二者混为一个文件，往往导致要么部署包过大，要么恢复信息缺失。

### 36.2 完整训练状态

至少需要保存：

1. 模型参数与 buffer；
2. tied embedding、模型配置和词表版本；
3. AdamW 等优化器的一阶矩、二阶矩和 step；
4. learning-rate scheduler 的进度与当前学习率；
5. FP16 GradScaler 或 FP8 scaling metadata；
6. global step、consumed tokens、gradient accumulation 进度；
7. Python、NumPy、CPU、CUDA 和各 rank 的随机状态；
8. dataloader 的 shard、sample offset 和 shuffle 状态；
9. ZeRO/FSDP、tensor parallel、pipeline parallel 的分片信息；
10. 代码 commit、依赖、硬件拓扑、数据版本和 schema 版本。

如果只恢复模型参数而丢失 Adam moments，参数初值虽然相同，后续更新却不同；如果 scheduler 从错误的 step 恢复，学习率可能突然回到 warmup 或跳到过高位置；如果 dataloader 游标丢失，训练会重复或跳过数据。

### 36.3 保存频率与保留策略

按 step 保存容易受 batch 和序列长度影响；按 consumed tokens 或时间更适合比较不同配置。保存过于频繁会带来 I/O 和存储成本，保存过少则故障回滚损失大。可以采用分层保留：最近一段时间密集保存，历史 checkpoint 稀疏保存，关键评估点长期保存，异常附近额外保留 debug 状态。

checkpoint 名称和 manifest 应包含 run id、step、consumed tokens、数据版本、模型版本和类型。不可变目录、状态标记、大小与 checksum 能避免半写入文件被当作可用状态。

### 36.4 sharded 与 consolidated

在 FSDP、ZeRO 或 tensor parallel 中，参数和 optimizer state 分散在多个 rank。sharded checkpoint 让各 rank 并行写入，适合超大模型；consolidated checkpoint 方便单机推理和发布，却可能需要额外聚合内存与 I/O。PyTorch 的 [distributed checkpoint 文档](https://pytorch.org/docs/stable/distributed.checkpoint.html)提供了对应的分布式状态保存接口入口，具体语义仍要按所用版本核对。

恢复时不一定保持相同 GPU 数。若要从 128 卡保存的状态恢复到 64 卡，需要 resharding 和可靠的元数据。只把 shard 文件拼接起来通常不够，因为参数布局、optimizer 分片和并行组都可能变化。

### 36.5 异步保存的一致性

大 checkpoint 写入可能需要很长时间。异步保存可以让训练更快继续，但必须保证“写入完成”与“状态可用”之间有明确协议。常见做法是写入新的不可变前缀、完成所有 shard、写 manifest 并校验 checksum，最后提交一个 `complete` 标记或更新版本指针；同一文件系统上的目录改名可以使用原子 rename，而对象存储通常应使用 manifest/条件写入等提交协议，不能假定 rename 原子。加载器只接受 complete 且校验通过的状态。

### 36.6 恢复后 loss 跳变的定位

先检查 checkpoint 是否完整，再检查模型 key、shape 和 dtype；接着核对 optimizer、scheduler、scaler、RNG、dataloader、consumed tokens、训练 config 和 rank mapping。若可以复现，比较中断前后同一 batch 的 forward loss、grad norm 和参数更新；若只在多卡发生，重点检查 shard 与通信。

严格 bitwise reproducibility 需要相同硬件、kernel、随机数、数据顺序和并行策略。许多大规模训练只要求 statistical reproducibility，即恢复后的趋势和最终能力接近；但无论目标是哪一种，都应避免未解释的巨大 loss 跳变。

### 36.7 一个简化的保存与恢复示例

以下示例依赖 PyTorch 2.x 与 NumPy，输入是小型线性模型、AdamW optimizer、学习率 scheduler、step 和数据游标；输出展示保存后修改参数，再从 checkpoint 恢复模型、优化器、scheduler、CPU 随机状态和游标。它是 CPU 教学演示，不覆盖 CUDA RNG、分布式 shard 或故障原子性。PyTorch 的通用 checkpoint 指南也建议按恢复目标保存必要状态，参见[官方示例](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)。

~~~python
import random
import tempfile

import numpy as np
import torch


def save_state(path, model, optimizer, scheduler, step, data_state=None):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
        "data_state": data_state,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    torch.save(state, path)


def load_state(path, model, optimizer, scheduler=None):
    # weights_only=False 会反序列化 Python 对象；仅对可信 checkpoint 使用。
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    return state["step"], state["data_state"]


torch.manual_seed(7)
model = torch.nn.Linear(2, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
batch = torch.tensor([[1.0, 2.0]])
optimizer.zero_grad(set_to_none=True)
model(batch).square().mean().backward()
optimizer.step()
scheduler.step()
expected = {key: value.detach().clone() for key, value in model.state_dict().items()}

with tempfile.TemporaryDirectory() as directory:
    path = f"{directory}/training-state.pt"
    save_state(path, model, optimizer, scheduler, step=1,
               data_state={"shard": 2, "offset": 16})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100.0)
    step, data_state = load_state(path, model, optimizer, scheduler)
    restored = all(torch.equal(model.state_dict()[key], value)
                   for key, value in expected.items())
    print("restored:", step, data_state, "parameters=", restored)
~~~


> **代码状态：静态审阅，未运行（2026-10-29）**

预期输出为 `restored: 1 {'shard': 2, 'offset': 16} parameters= True`。这里的 `weights_only=False` 会反序列化 checkpoint 中的 Python 对象，只能用于可信文件；不可信权重应优先采用受限加载或张量专用格式。生产系统还要保存 CUDA RNG、GradScaler、分布式分片、版本和 manifest；训练状态文件不能因为扩展名是 `.pt` 就被当作可信输入。

### 36.8 保存频率要按重算和 I/O 一起估算

文件大小由恢复目标决定。1B 参数模型的 BF16 权重约 2 GB；若完整训练状态按 16 字节/参数估算，则约 16 GB，是权重包的 8 倍。这个差额来自梯度、FP32 master weights 和 Adam 状态，具体配置可能因分片或低精度 optimizer 改变。

保存间隔则在故障重算与写入开销之间折中。假设同步保存耗时 10 分钟、每 2 小时训练保存一次，每个周期约有 `10/(120+10)=7.7%` 墙钟时间用于写盘；若故障时间均匀分布在两次保存之间，平均会损失约 1 小时训练工作。缩短间隔可降低重算量，却增加 I/O 和存储；异步保存能减少停顿，但会占用额外带宽与临时空间。应结合最大可接受重算步数、保存失败率和恢复耗时设定间隔。

### 36.9 用恢复演练验证保存的状态

定期在独立目录恢复模型、optimizer、scheduler、scaler、RNG 和数据游标，再以固定 batch 比较输出、loss、梯度范数和一次更新。若要求严格复现，还要恢复 sampler epoch、各 rank 随机状态、分片位置和 gradient accumulation 进度；若只要求统计复现，也要说明允许的差异，不能仅凭“loss 大致正常”声称恢复完全一致。

若一次 optimizer 更新由 8 个 micro-batch 组成、每个含 2,048 个有效 token，则 2,400 次更新对应 `8×2,048×2,400=39,321,600` consumed tokens。变长序列、padding 或丢弃不完整 batch 会改变实际数目；因此 checkpoint 要同时记录 optimizer step 和真实有效 token 计数，不能只用文件名中的 step 推断训练进度。

支持不同 GPU 数量时，要做真实的 resharding 演练。例如从 8 rank 保存、在 4 rank 加载，先核对重组后的参数值和 optimizer step，再跑固定 batch。只加载权重属于新的训练轨迹，不是从中断点续训。恢复失败时应保留上一份健康状态，不要让演练覆盖唯一 checkpoint。

### 36.10 分布式保存需要明确提交协议

安全提交按以下顺序进行：各 rank 写入不可变 shard；等待所有写入完成；校验文件大小和 checksum；写出包含 rank、参数范围、dtype、配置哈希和并行组版本的 manifest；最后才发布 `complete` 标记或更新可见指针。任一分片缺失、重复或校验失败时，加载器应拒绝该版本并继续使用上一份健康状态。对象存储和本地文件系统的 rename/条件写入语义不同，不能假设它们共享同一种原子性。

异步复制还要保证所有文件来自同一个训练时刻。比较安全的边界是一次 optimizer 更新完成后记录模型、optimizer、scheduler、scaler 和数据游标，再对这个不可变快照启动后台写入；如果一边复制、一边修改同一批状态，权重可能来自 step 2400，而 optimizer moments 已来自 step 2401。manifest 应分别保存 global step、optimizer step、scheduler 进度和 consumed tokens；使用 token 驱动 scheduler 时它们不必数值相同，但关系必须明确。

历史保留也要分层：近期 checkpoint 保留完整训练状态，较旧版本可只保留推理权重和配置，关键里程碑再连同评估和数据谱系保存。删除旧目录前先验证仍有至少一份可恢复版本。发布包与训练状态应分开命名和授权；后者包含可反序列化对象、随机状态和数据游标，只能从可信来源加载并限制访问。

例如 64-rank 作业的 manifest 应明确列出 64 个预期 shard。若 rank 17 的文件缺失或 checksum 不符，加载器要拒绝该 checkpoint，并指出缺少的分片；用零张量填补或跳过该 rank 会制造形状正确、语义损坏的状态。恢复工具的失败路径和错误信息也是 checkpoint 协议的一部分。


## 37. 评估集设计与停止决策

### 37.1 四类数据集的职责

训练过程至少要区分：用于 next-token loss 的 held-out validation，用于局部分析的 domain validation，用于具体能力的 downstream benchmark，以及尽量少使用的 hidden/fresh evaluation。它们的职责不同：

| 集合 | 主要用途 | 适合频率 | 主要风险 |
| --- | --- | --- | --- |
| core validation | 训练趋势与泛化 | 高频 | 分布过窄 |
| domain validation | 定位领域退化 | 高频/中频 | 桶定义偏差 |
| downstream benchmark | 任务能力 | 中频 | 污染、模板敏感 |
| hidden/fresh eval | 最终可信确认 | 低频 | 样本量与成本 |

固定核心集合便于跨 run 比较，同时保留新近数据和困难样本作为 fresh、stress 或 canary。一个评估集合一旦被反复用于调参，就逐渐失去独立测试的意义。

### 37.2 污染与时间切分

评估集污染会让模型在题目、答案、解析或模板上获得额外优势。训练前应做 exact/near dedup，训练后还可以用时间截止后的新题、新代码 issue、新文档和人工构造任务检查泛化。

时间切分不是万能方法：网页会被转载，题目会被改写，模型可能见过同一知识的其他表达。因此报告中要说明训练数据截止时间、来源范围、污染检测方法和 fresh eval 的构造方式。

算法生成任务是另一种补充。ICLR 2026 的 BeyondBench 用程序生成数学和组合问题，并以确定性解答校验；论文列出 44 类任务和 117 种变化形式，生成空间超过 $10^{15}$ 个实例，并在三折设置中评测 101 个模型。它能降低固定题面被记住的风险，但测量对象仍限于这些算法和数学题族；论文还使用单一 prompt，不覆盖百科知识、开放写作或真实产品流程。生成规则、提示与 verifier 共同定义被测能力，因此它只能作为污染风险较低的一类能力切片，不能证明一个模型“总体无污染”。[BeyondBench](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22b4e30a7a660b17f1fb58ab49671e77-Abstract-Conference.html)。

### 37.3 生成式评估的协议

生成式 benchmark 受 prompt、system instruction、temperature、top-p、max tokens、工具权限和停止字符串影响。比较模型时，这些条件要固定或进行多次运行；若使用 judge model，要报告 judge 版本、评分规则、位置偏差和人工抽样。

代码任务应优先使用编译和测试结果；数学任务应明确最终答案解析规则；工具调用任务应检查参数 schema、执行结果和错误恢复，而不只是文本相似度。

### 37.4 置信区间与小差异

对于 n 道独立近似题目，答对 k 道的准确率估计为：

~~~math
\hat p=\frac{k}{n}
~~~


二项近似下的标准误差为：

~~~math
SE\approx\sqrt{\frac{\hat p(1-\hat p)}{n}}
~~~


粗略 95% 区间可以写成：

~~~math
\hat p\pm1.96SE
~~~


这不是所有 benchmark 的最终统计方法。题目之间可能相关，生成可能有随机性，多个 checkpoint 的最高分还会引入选择偏差。重要结论应增加样本、固定协议、做 paired comparison、报告多次运行，并进行人工复核。

### 37.5 传统 early stopping 与大规模预训练

小数据监督学习中，验证 loss 先降后升时，可以在最低点停止并恢复最佳权重。大模型预训练更复杂：数据量极大，loss 可能长期下降；能力指标可能滞后；课程切换可能暂时抬高 loss；训练预算和数据配比通常预先规划。

因此，大规模预训练中的停止决策应同时考虑：关键分桶 loss、目标任务边际收益、训练成本、数据是否耗尽、稳定性、推理成本和安全指标。它更像多指标的预算管理，而不是一个 patience 参数。

### 37.6 checkpoint 的多指标选择

可以先设置不可妥协的条件，例如安全评估不能退化、目标领域 loss 不超过阈值、训练没有未解释的异常；在满足条件的候选中，再按目标任务排序。这个顺序比把所有指标简单加权更容易解释，也能避免一个高分 benchmark 掩盖严重安全或稳定性问题。

### 37.7 一个候选选择器

代码输入三个带验证 loss、代码 loss、数学分数、安全分数和稳定性标志的候选 checkpoint，先执行硬性筛选，再在剩余候选中按预先声明的目标分排序；`code` 越低越好，`math` 越高越好。它使用 Python 3.12 标准库运行，分数是构造数据，不代表真实模型质量。

~~~python
import math


checkpoints = [
    {"name": "a", "val": 1.90, "code": 1.48, "math": 0.41, "safety": 0.97, "stable": True},
    {"name": "b", "val": 1.87, "code": 1.46, "math": 0.45, "safety": 0.95, "stable": True},
    {"name": "c", "val": 1.85, "code": 1.59, "math": 0.47, "safety": 0.88, "stable": False},
]
if not checkpoints:
    raise ValueError("at least one checkpoint is required")
required_numeric = ("val", "code", "math", "safety")
for item in checkpoints:
    if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"]:
        raise ValueError("each checkpoint must have a name")
    for key in required_numeric:
        value = item.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("checkpoint metrics must be finite numbers")
        try:
            finite = math.isfinite(float(value))
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("checkpoint metrics must be finite numbers")
    if not isinstance(item.get("stable"), bool):
        raise ValueError("stable must be a boolean")

best_val = min(item["val"] for item in checkpoints)
eligible = [
    item for item in checkpoints
    if item["val"] <= best_val + 0.06
    and item["code"] <= 1.50
    and item["safety"] >= 0.93
    and item["stable"]
]
if not eligible:
    raise RuntimeError("no checkpoint satisfies the release constraints")
selected = max(eligible, key=lambda item: 0.6 * item["math"] - 0.4 * item["code"])
print("eligible:", [item["name"] for item in eligible])
print("selected:", selected["name"])
~~~


> **代码状态：静态审阅，未运行（2026-10-29）**

预期输出为 `eligible: ['a', 'b']` 和 `selected: b`。这段代码展示的是决策顺序，不是通用权重。不同产品可以把代码通过率、语言一致性、延迟或偏好指标放在目标函数中，但规则应在看结果之前确定，避免事后只挑最有利的 checkpoint。

### 37.8 评估频率的分层

高频层计算 loss、分桶 loss 和少量 canary；中频层运行 benchmark 子集；低频层运行完整能力、安全和人工评估；最终候选才使用 hidden eval。评估成本也要进入训练预算，70B 模型的完整代码和安全评估可能比一次小模型训练更贵。

用一个教学预算比较频率：若完整评估每次耗时 4 小时，对 20 个 checkpoint 全部运行一次就要 80 小时；若先用每个 5 分钟的 canary 筛选，再对 4 个候选做完整评估，总计约 `20×5/60+4×4=17.7` 小时。筛选阈值必须预先固定，且未入选 checkpoint 的风险切片仍需监控，不能让省下的时间变成漏掉退化的盲区。

再看样本量对差异解释的影响。若 n=100 道近似独立题答对 60 道，$\hat p=0.60$，标准误约为 `√(0.6×0.4/100)=0.049`，粗略 95% 区间约为 50.4%—69.6%；只差 3 个百分点很难据此下结论。逐题配对能提供更多信息：若 B 相对 A 修复 8 题、又丢失 5 题，净增 3 题，应继续看这 13 题的任务类型和难度，而不是只报两个总分。

配对题的显著性由不一致答案决定，不能直接把两个独立比例的标准误套上去；小样本可用 McNemar 或精确检验。若同一道题重复采样多次，样本还会在题目内相关，应按题目聚类处理，不能把每次生成都当成一条独立题目。

### 37.9 把评估错误拆到可行动的层级

两个模型都答错 20 题，一个可能是知识过时，另一个可能是输出格式解析失败；准确率相同，修复方案不同。按主题、语言、长度、证据位置、错误严重度和失败阶段切片，并保留人工复核样本，才能把模型能力、harness、评测器与数据疑点区分开。自动 judge 也应报告版本、评分规则、位置偏差和抽检结果。

污染检测没有一个能作最终裁决的分数。Exact match 会漏掉改写，语义相似会把共同知识或标准题型误报，成员推断依赖可区分性等假设。NAACL 2025 Findings 的综述整理了 50 篇研究和 8 类检测假设，并在三个直接的 instance-level 成员推断假设上做实证检验；这些方法在论文使用的若干预训练语料上接近随机猜测，较高分也可能反映训练集与测试集的领域差异。这个结果只挑战所测假设和数据条件，不是对所有污染检测方法的否定；未找到重叠仍只能记为“当前检测未命中”，不能写成“已证明无污染”。[Does Data Contamination Detection Work (Well) for LLMs?](https://aclanthology.org/2025.findings-naacl.291/)。

### 37.10 用独立评估控制 checkpoint 选择偏差

若每隔固定步数检查 100 个 checkpoint，偶然波动中总会有一个特别高分；再用同一集合选择并报告，就会对验证集过拟合。先用高频 canary 和分桶 loss 查回归，中频子集检查能力变化，validation 按预先规则筛选候选，最终用低频 hidden/fresh eval 确认。每次评估是否参与过调参都要记录，已经反复查看的集合不应再被称为独立测试。

### 37.11 停止训练要比较边际收益与评估成本

对最近几个 checkpoint 估计每增加一单位计算带来的目标能力、分桶 loss 和成功任务成本变化。若核心指标进入平台期、目标收益低于预设门槛，或安全/稳定性切片持续退化，就暂停扩训并定位原因；若指标仍变好，也要确认新增收益值得训练和评估成本。停止条件应在看结果前设定，避免为了追求最低 validation loss 无限延长训练。

例如两个 checkpoint 在同一组 200 道代码题上分别通过 80 道和 82 道，点估计从 40% 到 41%，只增加 1 个百分点。应列出逐题变化：B 修复了几道、又破坏了几道，变化集中在哪类任务，运行预算是否一致；这组配对结果比重复套用独立比例区间更有信息。只有配对差异、重复抽样和成本账本共同支持改进时，额外训练才有充分理由；若新增 compute 没带来可复核收益，就应把预算转去定位数据或评测缺口。数字为教学构造，不是模型实测。

## 38. 多语言预训练：数据不均衡、迁移与干扰

### 38.1 多语言不是简单拼接

多语言模型要同时解决数据数量不均衡、质量差异、tokenizer 效率、共享容量竞争、跨语言迁移和评估覆盖不足。按原始网页量训练通常让英文等高资源语言占主导；完全均匀采样又可能让低资源语言重复过多。

产品目标决定配比：全球助手、中英技术助手、翻译模型和低资源语言模型需要不同的数据、tokenizer 和评估，不能用一个“多语言比例”概括。

XLM-R 与 NLLB 分别研究了多语言预训练表示和多语言机器翻译的数据扩展；它们的经验必须限定在各自的语料、模型与任务范围内（[XLM-R](https://arxiv.org/abs/1911.02116)，[NLLB](https://arxiv.org/abs/2207.04672)）。

### 38.2 温度采样与重复暴露

对语言 i 的原始比例 p_i，可以使用：

~~~math
q_i=\frac{p_i^\alpha}{\sum_jp_j^\alpha}
~~~


α<1 时低资源语言占比提高。若语言 i 的可用 token 量为 A_i，训练计划为 B_i，则重复暴露 e_i=B_i/A_i。过高 e_i 可能让模型记忆有限语料，验证 loss 好看但 fresh 任务变差。

因此要同时看每语言的样本质量、可用量、重复簇大小、per-language loss、原生 benchmark 和人工母语评估。语言之间的 loss 数值也不能简单横比，因为 tokenizer 粒度和语言结构不同。

手算温度采样：三种语言原始比例为 90%、8%、2%，取 $\alpha=0.5$ 后未归一化权重约为 0.949、0.283、0.141，总和 1.373，低资源语言的新比例约为 `0.141/1.373=10.3%`。若总预算 100M token，它会分到 10.3M；可用语料只有 2M 时，平均重复约 5.15 次。提升占比同时放大重复风险，所以还要看原生 fresh 任务、其他语言是否退化，以及单语数据能否补充。

### 38.3 语言识别与质量过滤

文档级语言标签不够时，应增加段落或句子级识别，记录混合语言比例。中文技术文档可能包含大量英文代码，阿拉伯语和希伯来语还涉及方向性文本；机器翻译数据可能语法正确却有明显翻译腔。

英文的空格比例、stopword 和 Unicode 规则不能直接套用中文、泰语或印地语。质量阈值最好按语言校准，并用各语言人工抽样检查误删和漏放。

### 38.4 跨语言迁移的来源

多语言模型可以共享世界知识、数学结构、代码语义和文档模式，因此英文技术资料可能帮助中文技术问答。但迁移需要目标语言的锚点数据：中文问题的表达、术语、答案风格和安全边界都需要被模型看见。

一个英文数学能力很强的模型中文数学仍可能较弱，原因可能是中文题意解析、术语映射、中文推理样本或后训练不足，而不一定是数学知识缺失。改进方案可以组合中文题目、双语解释、可验证答案和中文指令数据。

### 38.5 容量竞争与语言干扰

共享参数会带来迁移，也会带来竞争。低资源语言过采样可能损害高资源语言；相似文字系统可能互相混淆；后训练中某种语言的强风格可能导致模型在其他语言中输出翻译腔或混杂语言。

缓解手段包括合理采样、扩大容量、语言标签、语言一致性数据、目标语言 SFT、多语言安全数据和按语言监控。是否使用独立 adapter 或路由模块，要用质量收益与部署复杂度共同评估。

### 38.6 tokenizer 的语言公平性

多语言评估要先分清两个量。Token fertility 通常是每个参考词对应的 token 数，受“词”的分词规则影响，适合在分母定义固定的语料或任务里比较；跨语言时，词边界未必能公平对齐。同一内容的 token 成本则在平行文本或其他对齐样本上直接比较 token 数，避免把某一种语言的词切分当作通用分母，但仍受翻译长度和对齐质量影响，也只测压缩效率，不测模型能力。NeurIPS 2025 的 tokenizer 研究采用同内容的 token-count 比较；The Token Tax 则在其 AfriMMLU 评估中使用 fertility，并报告它与准确率相关。两者回答的问题不同，不能把一个指标换算成另一个。

这不意味着必须让所有语言的 token 数完全相等，而是要识别表示差异对训练暴露、推理成本、长上下文和 benchmark 结果的影响，并把每个指标的分母和对齐方式一并报告。

### 38.7 多语言评估组合

一个完整的评估组合应包括：每语言 validation loss 趋势、原生语言 benchmark、翻译与跨语言问答、语言一致性、代码与数学任务、真实用户任务、人类母语评估和多语言安全测试。翻译 benchmark 有价值，却不能替代原生任务，因为翻译可能改变文化背景、术语和题意。

安全也不能只在英文测试。拒答边界、提示注入、隐私泄漏和有害内容在低资源语言中可能更弱，应把语言作为安全评估的分桶维度。

### 38.8 采样表的教学实现

下面的 Python 3.12 标准库代码输入语言可用 token 数、总预算和温度，输出各语言计划量与 exposure。它在 log-space 中计算温度权重，并拒绝布尔值、非正可用量、负预算和非有限参数；这个离线计划不包含有放回抽样方差或动态数据耗尽策略。

~~~python
import math
from collections.abc import Mapping


def language_plan(available, budget, temperature):
    if not isinstance(available, Mapping) or not available:
        raise ValueError("available data must be a non-empty mapping")
    amounts = {}
    for language, amount in available.items():
        if not isinstance(language, str) or not language:
            raise ValueError("language keys must be non-empty strings")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise TypeError("available amounts must be real numbers")
        try:
            amount = float(amount)
        except OverflowError as error:
            raise ValueError("available amounts must be finite") from error
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("available amounts must be finite and positive")
        amounts[language] = amount
    if isinstance(budget, bool) or not isinstance(budget, (int, float)):
        raise TypeError("budget must be a real number")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a real number")
    try:
        budget = float(budget)
        temperature = float(temperature)
    except OverflowError as error:
        raise ValueError("budget and temperature must be finite") from error
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and non-negative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    log_amounts = {lang: math.log(amount) for lang, amount in amounts.items()}
    largest_log = max(log_amounts.values())
    weights = {
        lang: math.exp((log_amount - largest_log) / temperature)
        for lang, log_amount in log_amounts.items()
    }
    normalizer = math.fsum(weights.values())
    ratios = {lang: weight / normalizer for lang, weight in weights.items()}
    plan = {lang: budget * ratio for lang, ratio in ratios.items()}
    exposure = {lang: plan[lang] / amounts[lang] for lang in amounts}
    if any(not math.isfinite(value) for value in exposure.values()):
        raise OverflowError("exposure exceeds the Python float range")
    return plan, exposure


available = {"en": 900, "zh": 80, "ar": 20, "sw": 2}
for temperature in [1.0, 2.0, 5.0]:
    plan, exposure = language_plan(available, 100, temperature)
    print("T=", temperature)
    print("plan=", {k: round(v, 2) for k, v in plan.items()})
    print("exposure=", {k: round(v, 2) for k, v in exposure.items()})
~~~


> **代码状态：静态审阅，未运行（2026-10-29）**

温度 `T=1` 对应原始数据比例；温度增大时，斯瓦希里语等低资源语言获得更高比例，但曝光倍数也可能变大。计划表可以用来提前发现重复风险，模型最终是否受益仍需由原生 fresh 评估和人工质量验证回答。

### 38.9 Token 预算对应多少语言覆盖

设两个语言桶各有 100 万原生字符，A 平均每 4 个字符编码成 1 token，B 平均每个字符占 1 token，则相同字符量分别需要 25 万和 100 万 token。若训练预算按 token 相等，B 获得的字符覆盖只有 A 的四分之一；若固定字符量，B 则消耗四倍 token。这个账本说明 token 级采样公平不等于内容覆盖公平，但它本身不证明哪种语言能力更强。

AfricaNLP 2026 的正式论文 *The Token Tax* 在 AfriMMLU 上评估 10 个 LLM、5 个学科和 16 种非洲语言，发现 token fertility 与准确率呈负相关。作者指出 fertility 也与训练数据可用量和质量等因素共同变化，因此不能把相关关系解释为 tokenizer 导致的因果效应。论文的成本推算还假设 attention 计算近似随序列长度平方增长；它适用于说明长序列成本机制，不代表每种模型或工作负载的端到端延迟都按同一比例变化。[The Token Tax](https://aclanthology.org/2026.africanlp-main.10/)。

长上下文评估也受 fertility 影响。假设两种语言各有 3,000 个字符，A 每字符平均 0.25 token，B 每字符 1 token，则分别约需 750 和 3,000 token。在 2,048-token 窗口里，A 的文本能全部进入，B 只能容纳约 `2048/3000=68.3%`；如果答案依赖末尾证据，差距可能来自截断而非推理能力。评估应同时报告按字符对齐的内容量和按 token 约束的计算量。

区分数据和 tokenizer 影响时可以做两套匹配：一套固定 token 预算，比较实际文本覆盖与任务收益；另一套固定字符或文档量，比较训练计算成本。两套实验都要固定模型规模、优化器和验证题意，并报告各语言 exposure。若只做其中一种，结论就只能回答“同等 token”或“同等内容量”下的差异，不能把结果推广到另一个预算条件。

因此多语言报告应同时列出原生字符或词数、token fertility、训练曝光、有效上下文长度和原生任务表现。若优化 tokenizer 让同样预算覆盖更多文本，还要检查形态变化、专名、代码和特殊字符是否仍能正确恢复。最后由母语评估确认自然度、术语、语言一致性和安全拒答；长度变短是成本改善，不自动等同于语义能力提升。


## 39. 代码模型预训练：结构、执行反馈与软件工程任务

### 39.1 代码与普通文本的差异

代码具有语法约束、长程依赖、可执行反馈和许可风险。一个字符或缩进错误可能使程序无法运行；函数可能依赖其他文件、配置、测试和版本环境；仓库中还包含 README、issue、PR 和 commit message 等自然语言上下文。

因此代码模型数据不应只是一堆文件内容。更完整的样本单位可以包含：文件路径、仓库结构、依赖、相关测试、错误日志、变更 patch、文档和任务描述。

### 39.2 数据来源与治理

常见来源包括开源仓库、官方 API 文档、README、issue、PR、commit、技术问答、单元测试、合成任务和获授权的内部代码。采集时需要保存仓库 revision、license、语言、路径和时间，过滤二进制、minified 文件、vendor 依赖、自动生成代码、密钥和敏感配置。

代码去重要处理 fork、模板、复制粘贴、不同版本和 benchmark 污染。仅对完整文件做 hash 不够，还要在函数、token n-gram 和仓库关系上建立重复簇；代码语料与许可治理的公开案例可参见 [The Stack v2 / StarCoder 2](https://arxiv.org/abs/2402.19173) 和 [BigCode Responsible AI](https://arxiv.org/abs/2308.07124)。

若样本来自可执行仓库，还应绑定 commit hash、语言版本、依赖锁文件、测试命令和许可结论。相同 API 在不同版本中可能改变参数；没有 revision 的代码样本会把不同时期的接口混成一个看似矛盾的训练目标。

### 39.3 代码 tokenizer

代码 tokenizer 应稳定表示缩进、换行、标识符、操作符、括号、字符串和注释。`get_user_profile`、`HTTPRequestHandler` 和 `maxSequenceLength` 等标识符的切分会影响模型学习命名模式与 API 组合；Python 的空白不能被普通文本清洗规则破坏。

代码与自然语言混合模型需要在代码片段、文档和用户指令之间共享词表，但不必让所有 token 都对两类文本同样高效。评估应分别报告代码 fertility、自然语言 fertility、补全长度和可执行成功率。

### 39.4 file-level、repo-level 与 task-level

file-level 样本训练简单，适合基础语法和局部补全；repo-level 样本让模型看到 import、配置、接口和测试之间的关系；task-level 样本则直接表达“根据 issue 修复 bug”“根据测试补全实现”“阅读错误日志并修改多个文件”。

coding agent 需要后两种能力，因为真实软件工程不是从空白文件生成一个孤立函数，而是定位文件、理解上下文、修改 patch、运行测试、读取错误并迭代。

### 39.5 FIM：从文件末尾续写到中间补全

Fill-in-the-Middle 将代码拆成 prefix、middle、suffix，让模型根据左右上下文生成中间片段。一个抽象样本为：

~~~text
<fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle
~~~


不同 tokenizer 的特殊 token 名称可能不同，但核心是把 suffix 放入条件上下文。FIM 比例过低会让模型只擅长 left-to-right；比例过高又可能影响普通续写，因此应在补全 benchmark 和一般代码生成上做消融。

### 39.6 测试、执行反馈与 verifier

代码能够运行，是相对强的外部反馈。可以生成多个候选，执行单元测试，选择通过者；也可以把错误日志反馈给模型，构造 bug-fix 轨迹；合成代码可先执行过滤，再进入训练。

但测试通过不等于完全正确。测试覆盖可能不足，代码可能有性能、风格、资源泄漏和安全问题。执行器还可能受到环境、依赖、网络和随机性的影响，因此评估要结合测试、静态分析、安全扫描和人工 review。

通过 5 条可见测试只证明覆盖到的 5 个条件通过。保留一组开发时不可见的边界测试，并分别记录编译失败、可见测试、隐藏测试和安全扫描，才能估计 verifier 的漏检风险；把同一组测试反复用于生成和筛选，会让模型学会针对测试模板而不是任务本身。

### 39.7 pass@k 的定义

若为一道题生成 n 个候选，其中 c 个通过测试，常用的 pass@k 无偏估计形式为 [HumanEval 论文](https://arxiv.org/abs/2107.03374)所用的组合估计：

~~~math
\widehat{\mathrm{pass@}k}
=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}
~~~


它估计从 n 个候选中不放回抽取 k 个时至少有一个正确候选的概率。pass@1 更接近一次生成可靠性；pass@k 还包含多次采样和筛选成本，不能把高 pass@k 直接解释为交互式代码助手的一次成功率。

### 39.8 一个 FIM 与 pass@k 示例

下面的 Python 3.12 标准库示例输入原始代码和 middle span 的半开区间 `[start, end)`，构造 FIM 协议字符串，并根据候选数 `n`、通过数 `c` 与抽样数 `k` 计算 pass@k。FIM 训练目标见 [Bavarian 等人的论文](https://arxiv.org/abs/2301.03988)；示例特殊 token 字面值仅作说明，实际名称由 tokenizer 协议决定。

~~~python
def build_fim(code, start, end):
    if not isinstance(code, str):
        raise TypeError("code must be a string")
    if any(not isinstance(value, int) or isinstance(value, bool)
           for value in (start, end)):
        raise TypeError("start and end must be integers")
    if start < 0 or end < start or end > len(code):
        raise ValueError("require 0 <= start <= end <= len(code)")
    return (
        "<fim_prefix>" + code[:start]
        + "<fim_suffix>" + code[end:]
        + "<fim_middle>" + code[start:end]
    )


def pass_at_k(n, c, k):
    if any(not isinstance(value, int) or isinstance(value, bool)
           for value in (n, c, k)):
        raise TypeError("n, c and k must be integers")
    if n <= 0 or c < 0 or c > n or k <= 0 or k > n:
        raise ValueError("require n > 0, 0 <= c <= n and 1 <= k <= n")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    miss_probability = 1.0
    for index in range(k):
        miss_probability *= (n - c - index) / (n - index)
    return 1.0 - miss_probability


code = "def area(r):\n    return 3.14 * r * r\n"
start = code.index("    return")
sample = build_fim(code, start, len(code) - 1)
print(sample)
for k in [1, 5, 10]:
    print("pass@", k, "=", round(pass_at_k(10, 3, k), 3))
~~~


> **代码状态：静态审阅，未运行（2026-10-29）**

这个实现只用于展示公式。生产评估需要隔离执行环境、限制资源、固定依赖，并防止候选代码访问不应访问的文件或网络。

手算 `n=10,c=3,k=5`：5 个候选全失败的概率是 `C(7,5)/C(10,5)=21/252≈0.0833`，所以 `pass@5=1−0.0833≈0.9167`；而 `pass@1=3/10=0.3`。前者表示多次采样后至少找到一个通过候选的估计，不是助手一次回答有 91.7% 成功率，也没有扣除生成 5 次和运行测试的成本。

### 39.9 评估不应止于 HumanEval

函数级 benchmark 可以测试局部生成，却覆盖不了多文件依赖、真实 repo、环境配置、bug 修复、代码解释、测试生成和安全。更完整的面板应包括：

1. 函数生成与补全；
2. FIM 与长文件补全；
3. 多语言编译和测试通过率；
4. 单元测试生成；
5. bug 修复与 patch 最小性；
6. repo-level issue；
7. 真实开发任务；
8. 安全漏洞与不安全 API 识别；
9. 延迟、采样次数和执行成本。

执行型评估还必须固定分母。假设 100 个代码请求中有 10 个结果无法解析或补丁无法应用，剩下 90 个进入测试，其中 81 个通过。只在可运行候选中计算，测试通过率是 $81/90=90\%$；端到端任务成功率则是 $81/100=81\%$。前者衡量可执行代码的质量，后者还包含协议、格式和补丁应用失败，更接近用户实际得到正确修复的概率。报告时应同时保留生成有效率、有效候选测试通过率和端到端成功率，不能因为某类失败“没有进入测试”就从分母里消失。

SWE-bench 等仓库任务把模型放进更接近真实的软件环境；[SWE-bench 论文](https://arxiv.org/abs/2310.06770)定义了其 issue 解决任务，但结果仍依赖 harness、补丁应用、测试命令和环境版本。评估报告必须把这些条件写清楚。

### 39.10 代码安全与工具权限

代码能力提升也可能提升生成 SQL injection、XSS、命令注入、不安全反序列化、路径遍历、硬编码密钥和恶意脚本的能力。训练中应加入安全修复与安全解释数据，评估中要做静态扫描和红队任务，agent 运行时则要限制文件、网络、凭证和命令权限。

最小权限、沙箱、可审计 trace、人工确认和可回滚 patch 是系统设计的一部分，不是代码模型训练结束后再补的装饰。


### 39.11 从候选生成走到仓库任务验收

以一个固定 issue 为例：锁定仓库 commit、依赖镜像、测试命令、可读写目录和运行预算。模型返回 patch 后依次检查能否应用、能否编译、公开测试、隐藏边界测试和安全扫描。即使 5 条公开测试全通过，2 条隐藏测试只通过 1 条，任务仍未完成；可见测试的分数不能替代完整任务的成功定义。

如果允许 5 次采样并用 verifier 选候选，还要记录候选间相关性、筛选器误判、测试耗时、工具调用和人工介入。pass@k 估计的是候选中存在正确解的机会；面向用户的指标应是固定权限和成本预算内，issue 是否被可靠解决。不同 harness、依赖版本或测试集上的数字不可直接横比，训练样本也应保留对应 revision 与许可信息。

复现实验还需保存 prompt、temperature、候选数、随机种子、verifier 版本和每条测试结果；否则无法判断 pass@k 的差异来自模型、采样策略还是评测环境。


## 40. 端到端预训练方案：把数据、训练与评估连成一条证据链

### 40.1 一个具体目标

假设要训练一个中英双语、具备数学和代码基础能力的 decoder-only base model，后续还要把它用于代码助手和工具调用。这里不直接给出“最优配方”，而是展示如何把目标转化成可审计的实验计划。

先定义成功条件：

1. 中英文普通文本的 validation loss 稳定下降；
2. 代码补全、FIM 和测试通过率不因双语配比而退化；
3. 数学任务在 fresh 评估上有可重复提升；
4. 训练过程可从任意健康 checkpoint 恢复；
5. 最终模型的推理成本、上下文长度和安全表现符合部署边界。

这些条件同时包含质量、可靠性和系统成本，避免只用一个总 loss 代表整个项目。

### 40.2 数据 manifest

每条训练样本写入不可变 manifest，至少包括：

~~~text
sample_id, source_id, revision, language, domain, license,
quality_score, dedup_cluster, contamination_flags,
tokenizer_version, data_version, inclusion_reason
~~~


训练前固定 core validation、domain validation、fresh evaluation 和 hidden test。训练集与验证集做 exact/near dedup；代码仓库保存 revision 与 license；合成样本保存 teacher、生成提示、验证器和失败原因。

### 40.3 tokenizer 选择实验

准备至少两个候选 tokenizer：一个基线词表，一个针对中英和代码重新平衡的词表。对中文、英文、数学、Python、TypeScript、混合技术文档分别统计 fertility、P95 长度、特殊符号恢复和 encode/decode 吞吐。然后在小模型上比较同等字符预算和同等 token 预算两种口径，避免把长度优势和模型能力混在一起。

若新 tokenizer 只在中文长度上改善，却让代码标识符和特殊 token 变差，不能只按平均 fertility 选择。需要把部署上下文、KV Cache、输出 softmax 和 FIM 协议一起计入。

### 40.4 数据配比实验

固定总 token 预算，至少比较：原始比例、温度采样、目标驱动比例。记录每个领域的可用量与 exposure，训练中按固定 interval 计算分桶 loss，周期性运行数学、代码和多语言子集。

一个教学性的预算可以是 1B token：45% 通用网页与书籍对应 450M，25% 代码和测试对应 250M，10% 数学对应 100M，15% 中英文技术资料对应 150M，5% 对话和工具格式对应 50M，合计 1B。若代码桶只有 100M 可用 token，则 250M 训练计划意味着平均曝光 2.5 倍；变长文档、截断和 packing 还会让实际有效 token 偏离这个平均。它不是推荐配方，只展示如何把比例换成可核对的预算。

### 40.5 Scaling 试验矩阵

选择 3–4 个模型规模和 2–3 个数据配比，在总计算预算内做短跑。每个 run 保存：参数量、有效 token、序列长度、batch、学习率、warmup、精度、吞吐、验证 loss、分桶 loss、目标任务和稳定性事件。

用小规模曲线预测中等规模，再用中等规模校准大规模。若外推偏差来自数据质量、重复或 recipe 切换，应更新内部曲线，而不是继续套用公开指数。

### 40.6 训练监控与恢复

高频监控 loss、学习率、grad norm、tokens/s、显存和通信；中频运行 domain validation；出现 spike 时记录 batch、rank 和 shard。训练 checkpoint 保存模型、optimizer、scheduler、scaler、RNG、dataloader、数据版本和分布式 metadata。

训练作业还要有“健康状态”与“异常状态”的明确转换：发现非有限 loss 时停止更新，保留异常现场，从最近健康状态恢复；修复后先在小规模回放，再继续主 run。不能让自动重试覆盖唯一健康状态。

### 40.7 多指标发布判断

候选模型先经过硬性条件筛选：没有未解释的 NaN 或 checkpoint 损坏，core 与目标领域 loss 不严重退化，代码与数学 fresh eval 达到目标，安全与许可审查通过。剩余候选再按产品目标选择：代码助手重视测试通过与延迟，研究模型重视综合能力，双语产品重视语言一致性和中英任务。

最终报告应把模型版本、数据截止时间、tokenizer、训练 token、上下文、解码、工具和评估 harness 写明。只给一个 benchmark 分数，无法复现也无法判断能力来自数据污染、采样预算还是模型本身。

### 40.8 失败案例复盘表

| 现象 | 优先检查 | 可能修复 | 不能直接下的结论 |
| --- | --- | --- | --- |
| 中文 loss 降、代码通过率降 | 数据比例、tokenizer、代码 bucket | 增加高质量代码与测试，复查切分 | 不能说模型整体变差 |
| 训练 loss 降、fresh 数学不涨 | 任务覆盖、污染、后训练需求 | 增加可验证任务与独立评估 | 不能说 scaling 无效 |
| 恢复后 loss 跳变 | optimizer、scheduler、数据游标、RNG | 完整恢复或回滚 | 不能只归因于权重 |
| FP16 频繁 skip | overflow、loss scale、attention logits | BF16、稳定 kernel、调 LR | 不能说数据一定有错 |
| pass@k 高、pass@1 低 | 采样与 verifier 成本 | 改善一次生成或筛选 | 不能把候选能力当一次可靠性 |
| 多语言 benchmark 高、母语评估差 | 翻译模板、语言一致性、文化语境 | 原生数据与人工评估 | 不能只看翻译集 |

### 40.9 证据等级与写作边界

预训练书稿中的结论应区分：数学定义和代码行为是可直接复查的；论文中的实验结论要绑定数据、模型和评估条件；模型卡和产品页的数字要绑定版本、日期、硬件和提示；内部训练配方若未公开，只能写成假设或教学抽象。

“某模型使用了某方法”需要一手模型卡、技术报告、论文、代码或官方文档支持；“某方法在教学实验中展示了某现象”应明确是构造例子；“某方法通常更稳定”应说明硬件、实现和对照条件。证据边界本身就是技术写作的一部分。

### 40.10 全章的因果链

本部分可以压缩成一条但不能再简化成一句口号的因果链：

~~~text
文本边界
  -> token 长度与表示偏置
  -> 数据质量、重复和污染
  -> 领域/语言/难度配比
  -> 有效训练 token 与计算预算
  -> loss、稳定性和 checkpoint 轨迹
  -> 分桶评估、下游能力与单位成功成本
  -> 发布、回滚和下一轮数据改进
~~~


任何一环的统计口径变化，都会改变对下一环的解释。例如 tokenizer 变了，PPL 和上下文长度的比较就要重新校准；数据去重变了，Scaling 曲线和验证污染风险就要重新估计；后训练或推理预算变了，base model loss 就不能直接代表产品能力。

实际项目可通过变更记录保持因果线索。一次 run 应固定不可变的数据版本、tokenizer、模型配置和训练代码；若改变采样温度、去重阈值或学习率，就建立新 run id 并记录差异。出现结果变化时，先比较数据和 recipe 的变更，再看模型和硬件。每个结果都关联到 manifest、checkpoint 和 evaluation harness，才能复现这条链，而不是只留一个最终权重文件。

最终结论可以按“观察—解释—验证—决定”来写。观察是某分桶 loss 或任务成功率变化；解释列出数据、优化和评估等竞争原因；验证设计最小对照；决定说明是否继续训练、调整数据或发布。若缺少足够证据，应写出不确定性和下一项测量。这样的记录让预训练成为可迭代的实验过程，也为后训练与 serving 阶段留下清楚交接。

交接时把模型适用边界与复现实验入口一起交给后训练和部署团队：训练数据截止、tokenizer/template、上下文训练范围、已验证任务与失败切片、污染风险、代码 revision、配置、数据 manifest、验证集版本、软硬件环境、checkpoint、评估输出和异常记录。后训练可据此选择监督数据与拒答目标，Serving 可据此设置长度、采样和缓存；若数据不能随书发布，保留脱敏摘要、过滤统计和访问申请流程。这样后续团队能复核结论，也能知道哪些能力需要重测。

### 40.11 自测与实践题

1. 设计一个同时覆盖中文、英文、Python 和数学公式的 tokenizer 评估集，并说明至少五个指标。
2. 给定四个领域的可用 token 数和目标训练预算，分别计算原始比例与温度采样下的 exposure。
3. 设计 exact、near 和 semantic dedup 的三级管线，并说明一个可能的误删案例。
4. 构造一个“loss 下降但目标能力下降”的数据配比实验，列出控制变量。
5. 为数学或代码合成数据设计一个独立验证器，并说明验证器覆盖不到的错误。
6. 从一个健康 checkpoint 恢复训练时，列出至少十类需要恢复的状态。
7. 解释为什么 FP16 需要 loss scaling，而 BF16 通常不需要同样的机制。
8. 为一个多语言代码模型设计 core、domain、fresh 和 hidden 四类评估集。
9. 用 pass@k 解释多次采样能力与一次生成可靠性之间的差异。
10. 写一页预训练实验记录，明确哪些结论是论文事实、哪些是本次实测、哪些只是教学假设。

到这里，训练输入、目标、token 预算、稳定性与停止条件已经形成一份可检查的预训练计划；接下来的问题是怎样把同一个目标拆到多张加速卡上执行，并保证梯度、通信和恢复状态一致。第四部分因此从 DDP 与状态分片开始：[大规模训练系统进阶](04-大规模训练系统进阶.md)。

## 参考资料与证据边界

本章优先使用原始论文、正式技术报告和官方框架文档。论文中的数值结果只对其公开实验条件负责；教学代码用于解释机制，不等同于生产实现；闭源模型的内部数据配方和训练细节不能由产品表现反推。

1. Sennrich, Haddow and Birch, *Neural Machine Translation of Rare Words with Subword Units*：BPE 子词分割与开放词表。
   https://aclanthology.org/P16-1162/
2. Kudo and Richardson, *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*：SentencePiece 框架。
   https://aclanthology.org/D18-2012/
3. Kudo, *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*：Unigram 与子词采样。
   https://aclanthology.org/P18-1007/
4. Radford et al., *Language Models are Unsupervised Multitask Learners*：GPT 类自回归语言模型的公开早期说明。
   https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
5. Lee et al., *Deduplicating Training Data Makes Language Models Better*：训练数据去重与记忆风险。
   https://arxiv.org/abs/2107.06499
6. Gao et al., *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*：多来源语料与数据治理讨论。
   https://arxiv.org/abs/2101.00027
7. Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*：大规模网页数据过滤与复现经验。
   https://arxiv.org/abs/2406.17557
8. Lozhkov et al., *StarCoder 2 and The Stack v2*：代码数据与代码模型数据治理路线。
   https://arxiv.org/abs/2402.19173
9. Li et al., *Textbooks Are All You Need*：高质量合成/筛选数据的教学型实验。
   https://arxiv.org/abs/2306.11644
10. Wang et al., *Self-Instruct: Aligning Language Models with Self-Generated Instructions*：自生成指令数据。
    https://arxiv.org/abs/2212.10560
11. Shumailov et al., *AI models collapse when trained on recursively generated data*：递归合成数据的分布风险。
    https://www.nature.com/articles/s41586-024-07566-y
12. Kaplan et al., *Scaling Laws for Neural Language Models*：语言模型规模经验曲线。
    https://arxiv.org/abs/2001.08361
13. Hoffmann et al., *Training Compute-Optimal Large Language Models*：Chinchilla 计算最优训练。
    https://arxiv.org/abs/2203.15556
14. Brown et al., *Language Models are Few-Shot Learners*：大规模语言模型训练与评估中的污染讨论背景。
    https://arxiv.org/abs/2005.14165
15. Micikevicius et al., *Mixed Precision Training*：FP16、loss scaling 与混合精度。
    https://arxiv.org/abs/1710.03740
16. Kalamkar et al., *A Study of BFLOAT16 for Deep Learning Training*：BF16 数值范围及多类训练任务实验；不代表所有当前硬件的吞吐或 kernel 表现。
    https://arxiv.org/abs/1905.12322
17. Micikevicius et al., *FP8 Formats for Deep Learning*：FP8 格式与训练数值问题。
    https://arxiv.org/abs/2209.05433
18. Wang et al., *DeepNet: Scaling Transformers to 1,000 Layers*：深层 Transformer 稳定性与 DeepNorm。
    https://arxiv.org/abs/2203.00555
19. PyTorch, *Automatic Mixed Precision package*：autocast、GradScaler 与 AMP 工程接口。
    https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
20. PyTorch, *Saving and Loading a General Checkpoint in PyTorch*：模型、优化器和训练状态保存。
    https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
21. PyTorch, *torch.distributed.checkpoint*：分布式 checkpoint 接口。
    https://pytorch.org/docs/stable/distributed.checkpoint.html
22. Conneau et al., *Unsupervised Cross-lingual Representation Learning at Scale*：XLM-R 多语言预训练。
    https://arxiv.org/abs/1911.02116
23. NLLB Team et al., *No Language Left Behind: Scaling Human-Centered Machine Translation*：低资源语言与多语言数据。
    https://arxiv.org/abs/2207.04672
24. Chen et al., *Evaluating Large Language Models Trained on Code*：HumanEval 与 pass@k。
    https://arxiv.org/abs/2107.03374
25. Bavarian et al., *Efficient Training of Language Models to Fill in the Middle*：FIM 训练目标。
    https://arxiv.org/abs/2301.03988
26. Li et al., *CodeXGLUE: A Machine Learning Benchmark Dataset for Code Intelligence*：代码任务评估集合。
    https://arxiv.org/abs/2102.04664
27. Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*：仓库级软件工程任务。
    https://arxiv.org/abs/2310.06770
28. Li et al., *StarCoder: may the source be with you!*：代码模型、代码数据和治理背景。
    https://arxiv.org/abs/2305.06161
29. *BigCode: Open Code LLMs Meet Responsible AI*：代码模型数据治理、许可和责任边界。
    https://arxiv.org/abs/2308.07124
30. Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*：T5 与 C4 语料的构造和使用背景。
    https://arxiv.org/abs/1910.10683
31. PyTorch, *Automatic Mixed Precision examples*：autocast、GradScaler 与 AMP 更新次序的官方教程。
    https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
32. PyTorch, *CrossEntropyLoss*：类别索引、`ignore_index` 和 reduction 的公开接口语义。
    https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
