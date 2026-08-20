# 第三部分：Tokenization、数据与预训练进阶

大模型的训练不是把文本丢进 Transformer 就结束了。原始字符串首先要经过 tokenizer，数据要经过筛选、去重、配比和版本管理，训练过程要在有限计算预算下决定模型规模与 token 数，训练结果还要通过分桶验证、下游任务和安全测试才能被解释。

这一部分把预训练链路拆成十五个相互独立、又能连成一条工程路径的主题：tokenizer 的表示选择、tokenizer 评估、数据清洗、重复与污染、数据混合、合成数据、Scaling Law、loss 与能力、训练稳定性、混合精度、checkpoint、评估设计、多语言训练、代码模型，以及一个端到端的预训练方案。每个主题都同时给出入门直觉和进阶分析。入门者先抓住“它解决什么问题”，再看公式；有经验的读者则应继续追问统计口径、反事实、系统成本和证据边界。

一个贯穿全部分的判断是：token 数、参数量、validation loss、benchmark 分数和产品成功率分别描述不同层面的事实。它们可能相关，却不能相互替代。真正可靠的预训练结论必须把数据分布、训练 recipe、评估协议和硬件条件一起写清楚。

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

Byte Pair Encoding（BPE）最初是压缩领域的方法，后来被用于子词建模。它的训练过程可以抽象为：从字符或 byte 作为初始符号，统计相邻 pair 的频率，每次把最高频 pair 合并为一个新符号，直到达到目标词表大小。

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

Unigram tokenizer 采用另一条路径：先构造较大的候选子词集合，为每个候选片段学习一个概率或代价，再从可能的切分中选择总体概率较高的方案。给定字符串 s 的一个切分 z=(z_1,...,z_m)，其近似概率可以写成：

~~~math
P(z\mid s)\propto\prod_{i=1}^{m}P(z_i)
~~~

训练会逐步删除贡献较小的候选片段，使词表收敛到目标大小。与 BPE 的“按 pair 合并”相比，Unigram 更自然地表达“一句话可以有多种切法”。在 subword regularization 中，还可以从多个高概率切分中采样，让模型对边界变化更鲁棒。

Unigram 并不意味着一定优于 BPE。它带来概率建模和采样的灵活性，也带来更复杂的训练和实现。实际选择要看语言覆盖、代码数据、编码速度、词表训练工具和下游效果。

### 26.5 SentencePiece 解决的是边界假设

SentencePiece 是 tokenizer 训练与编码框架，不等同于某一个单独算法。它可以承载 BPE 或 Unigram，并把原始句子视为字符流处理，而不是强制先按空格切词。这个设计对中文、日文等没有天然空格边界的语言尤其重要。

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

### 26.8 一个可检查 BPE 合并过程的最小实现

下面的实现刻意保持简单：它用字符作为初始符号，不处理 byte、Unicode 规范化和跨词边界，只用于观察频率统计与合并顺序。

~~~python
from collections import Counter


def pair_counts(words):
    if not words:
        raise ValueError("words must not be empty")
    counts = Counter()
    for word, frequency in words.items():
        symbols = word.split()
        for left, right in zip(symbols, symbols[1:]):
            counts[(left, right)] += frequency
    return counts


def merge_pair(words, pair):
    if len(pair) != 2 or not all(pair):
        raise ValueError("pair must contain two non-empty symbols")
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

工业 tokenizer 还要处理词边界、特殊 token、规范化和高效编码，但这个例子能说明一个容易被忽略的事实：词表不是人工列出的词典，而是训练语料统计和预设边界共同产生的结果。

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

### 27.4 特殊 token 是控制协议，不是普通词

`BOS`、`EOS`、padding、角色标记、工具调用标记和 FIM 标记都会影响模型的信息流。特殊 token 需要固定 id、明确插入位置和明确 loss mask。例如对 chat 数据，用户消息、助手消息和工具结果是否计算 loss，会改变 SFT 的训练目标；对代码 FIM，prefix、suffix、middle 的顺序必须和训练时一致。

特殊 token 的字符串形式也需要防碰撞。如果用户输入中可以自然出现同名字符串，预处理器必须能够区分“文本中的字面量”和“协议控制符”。一个更安全的做法是使用 tokenizer 内部保留 id，并在模板层明确转义，而不是依赖字符串替换。

### 27.5 词表迁移的数学障碍

旧模型 embedding 矩阵为 E_old∈R^(V_old×d)，新 tokenizer 的词表为 V_new。即使两个 tokenizer 都能编码相同文本，token id 的语义也不一致。除非新旧词表有明确映射，否则不能直接把 E_old 的第 i 行当成 E_new 的第 i 行。

如果只增加少量新 token，可以保留旧行并随机初始化新增行，再用继续训练让新 embedding 对齐；如果改变了大量切分规则，则输出头、数据分布、位置长度和学习率计划都可能需要重新校准。所谓“只换 tokenizer、不动模型”通常只适用于非常受限的扩展，不是通用迁移方案。

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

### 27.7 一个 tokenizer 统计脚本

下面的代码不依赖第三方 tokenizer 库，用一个可替换的 `encode` 函数演示如何按领域统计长度分布。实际使用时，只需把示例编码器替换为目标 tokenizer。

~~~python
from collections import defaultdict
from statistics import mean, median


def toy_encode(text):
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
    ("en", "The model reads context."),
    ("code", "def add_numbers(left, right):\n    return left + right"),
]
lengths = defaultdict(list)
for domain, text in samples:
    tokens = toy_encode(text)
    lengths[domain].append((len(tokens), len(text)))


def safe_ratio(numerator, denominator):
    return round(numerator / denominator, 3) if denominator else None


for domain, values in lengths.items():
    ratios = [
        safe_ratio(token_count, char_count)
        for token_count, char_count in values
    ]
    measured = [value for value in ratios if value is not None]
    print(domain, "tokens=", [item[0] for item in values])
    print(
        "  mean fertility=",
        round(mean(measured), 3) if measured else None,
    )
    print("  median tokens=", median(item[0] for item in values))
~~~

这个统计器的价值不在于 toy tokenizer 的数值，而在于把“词表好不好”改写成按语言、领域、长度和协议逐项测量的问题。字符数为零时 fertility 没有定义，代码保留 `None`，不会把分母替换成 1 后制造一个看似正常的比例。

### 27.8 chat template 与 tokenizer 的边界

chat template 常和 tokenizer 一起发布，但两者职责不同。tokenizer 负责字符串与 id 的变换，template 负责把角色、消息、工具调用和结束条件组织成输入字符串或特殊 token 序列。模板改变了训练和推理的上下文，可能影响 loss、停止位置和工具协议。

审计一个模型时，应保存 tokenizer 版本、特殊 token 表、template 版本和一组 golden examples。只验证普通文本编码而不验证完整对话模板，无法排除角色串位、重复 BOS、错误 EOS 或工具结果被当作用户文本等问题。

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
q(x)=w_r r(x)+w_l l(x)+w_d d(x)+w_v v(x)-w_n n(x)
~~~

其中 `r` 是可读性，`l` 是许可与来源完整度，`d` 是领域相关性，`v` 是可验证性，`n` 是噪声或风险。这个式子是工程记账，不是自然定律；权重应由目标任务和验证结果校准。

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

下面的代码展示数据记录应该如何携带 metadata。它不是完整的隐私检测器，也不应被直接用于生产；它的重点是让“过滤结果可解释、可回溯”。

~~~python
import hashlib
import re


SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA|OPENSSH) PRIVATE KEY-----"),
]


def content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inspect_record(record):
    text = record["text"]
    stripped = text.strip()
    flags = []
    if len(stripped) < 80:
        flags.append("too_short")
    if stripped.count("<script") > 2:
        flags.append("template_or_script_heavy")
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        flags.append("possible_secret")
    return {
        "id": content_hash(text),
        "source": record.get("source", "unknown"),
        "language": record.get("language", "unknown"),
        "license": record.get("license", "unknown"),
        "flags": flags,
        "keep": not flags,
    }


records = [
    {"text": "A short note.", "source": "web", "language": "en", "license": "unknown"},
    {"text": "A" * 120, "source": "docs", "language": "en", "license": "CC-BY"},
]
for record in records:
    print(inspect_record(record))
~~~

生产系统中应把 `keep=False` 拆成具体原因，不能让一个综合布尔值掩盖“短文本”“疑似密钥”和“许可未知”之间完全不同的处理路径。

### 28.9 证据边界

数据清洗规则的具体阈值通常是项目经验，不应写成普适定律。C4、The Pile、FineWeb、BigCode 等数据工作可以支持某些数据构造与过滤经验；但某个模型到底使用了多少网页、代码或合成数据，只有其正式技术报告或模型卡披露后才能作为事实。

## 29. 去重与数据污染：避免把记忆误判为泛化

### 29.1 为什么重复会改变训练结果

相同或近似样本被重复采样，会同时影响优化、记忆和评估。它可能降低训练 loss，却不增加等量的新信息；热门仓库或模板被大量 fork，会让某一领域在混合数据中被隐性放大；验证集或 benchmark 若出现在训练集，分数就不能再解释为未见泛化。

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

~~~python
def shingles(text, width=5):
    normalized = " ".join(text.lower().split())
    return {
        normalized[index:index + width]
        for index in range(max(0, len(normalized) - width + 1))
    }


def jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 1.0


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

这个实现只是候选生成器，不足以承担生产去重。它没有处理词序变化、代码语法、跨语言改写或模板公共部分；它的意义是让 Jaccard 阈值、样本长度和误删风险可以被实际观察。

### 29.8 记忆与泛化的实验边界

重复率高不自动证明模型只是在记忆，去重后能力下降也不自动证明原始数据更有价值。可以设计三组对照：原始数据、exact dedup、near dedup；固定 token budget、模型规模和训练 recipe；在训练域、fresh 数据和受控复述集上分别评估。这样才能区分记忆收益、数据多样性收益和过滤误伤。

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
| 通用网页 | 900B | 45% | 0.5x | 常识与语言流畅 | 噪声、模板 |
| 代码与测试 | 300B | 25% | 1.7x | 代码生成与修复 | 许可、污染 |
| 数学 | 80B | 10% | 2.5x | 符号推理 | 答案错误 |
| 中英文技术文档 | 200B | 15% | 1.5x | 跨语言技术问答 | 翻译腔 |
| 对话与工具 | 50B | 5% | 2.0x | 指令与协议 | 风格过拟合 |

数字只是教学构造，不能当作通用推荐。重点是每个比例都应能说明目标、重复和风险。

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

~~~python
import math


def normalize(values):
    if not values or any(
        not math.isfinite(value) or value < 0 for value in values.values()
    ):
        raise ValueError("values must be non-empty and finite non-negative")
    total = sum(values.values())
    if total <= 0:
        raise ValueError("values must have a positive total")
    return {key: value / total for key, value in values.items()}


def temperature_ratio(available, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    raw = normalize(available)
    alpha = 1.0 / temperature
    adjusted = {key: value ** alpha for key, value in raw.items()}
    return normalize(adjusted)


available = {"web": 900, "code": 300, "math": 80, "bilingual": 200}
budget = 400
for temperature in [1.0, 2.0, 4.0]:
    ratio = temperature_ratio(available, temperature)
    planned = {key: budget * value for key, value in ratio.items()}
    exposure = {key: planned[key] / available[key] for key in available}
    print("T=", temperature)
    print("ratio=", {key: round(value, 3) for key, value in ratio.items()})
    print("exposure=", {key: round(value, 2) for key, value in exposure.items()})
~~~

当温度升高时，低资源桶的比例会上升，但 exposure 也可能迅速变大。采样策略的正确评价不是“低资源语言占比是否上升”，而是“它是否在目标任务上带来收益，且没有用重复和高资源退化换来这个收益”。

### 30.7 配比实验的控制变量

比较两种数据配比时，至少固定模型结构、总训练 token、学习率计划、batch、tokenizer、随机种子范围和评估协议。否则观察到的差异可能来自更多计算、更长序列或不同优化器状态，而不是比例本身。

推荐保留一个原始比例 baseline、一个目标驱动比例、一个温度采样比例，并在总量相同的前提下比较 overall loss、per-domain loss、目标任务、fresh eval、重复暴露和单位成功成本。

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

若学生模型反复训练模型生成的数据，数据分布可能逐渐丢失长尾和多样性，错误模式也会被循环放大。研究者将这种风险称为 model collapse，但具体程度取决于数据混合比例、真实数据保留量、生成质量和任务。

工程上可采取：保留足量人类数据；限制同一 teacher 风格的占比；按 semantic cluster 限制重复；引入不同 teacher 或不同采样策略；在 fresh 与人工数据上验证；为每个合成批次保留回滚能力。合成数据应该提高单位 token 的有效学习信号，而不是替代所有真实分布。

### 31.6 一个教学用的合成数据筛选器

~~~python
def accept(candidate):
    return (
        candidate["format_ok"]
        and candidate["answer_verified"]
        and candidate["safety_ok"]
        and candidate["quality"] >= 0.8
    )


candidates = [
    {"id": "a", "format_ok": True, "answer_verified": True, "safety_ok": True, "quality": 0.91},
    {"id": "b", "format_ok": True, "answer_verified": False, "safety_ok": True, "quality": 0.97},
    {"id": "c", "format_ok": True, "answer_verified": True, "safety_ok": False, "quality": 0.95},
]
accepted = [item for item in candidates if accept(item)]
print("accepted:", [item["id"] for item in accepted])
~~~

真实系统的 `answer_verified` 不能由同一个生成模型自报。它应来自执行器、独立证明器、检索来源、规则检查或人工标注；对不同任务，应记录验证器的覆盖率和误判率。

### 31.7 合成数据的配比与消融

对合成数据最有信息量的实验通常不是比较“生成一百万条”和“生成两百万条”，而是比较同一训练预算下的真实数据比例、合成数据比例、验证器强弱和多样性控制。至少保留：真实数据 baseline、低比例合成、较高比例合成、未经严格验证的合成和严格验证的合成。

评估要包含训练分布内任务、fresh 任务、长尾任务和错误鲁棒性。若只在合成题上提升，不能说明真实能力提升；若 overall loss 降而长尾下降，可能是分布坍缩；若代码测试通过率升而安全扫描变差，需要把安全指标作为独立约束。

### 31.8 证据边界

Self-Instruct、Textbooks Are All You Need、Phi 系列和 model collapse 研究可以支持合成数据的路线与风险讨论；具体模型采用多少合成数据、由哪个 teacher 生成、过滤器是什么，必须以对应技术报告为准。教学中构造的 verifier 或评分器只能说明方法，不等于生产系统已经具备同等可靠性。

## 32. Scaling Law：用实验曲线规划参数、数据与计算

### 32.1 Scaling Law 研究的对象

Scaling Law 讨论的是模型规模、训练数据、计算量与某个可测指标之间的经验关系。最经典的指标是验证集上的 next-token cross-entropy loss。它不是“模型越大越好”的口号，而是一个预算规划工具：在可用算力下，参数量和训练 token 应如何分配，何时继续扩大模型，何时应该改善数据或 recipe。

常用变量包括模型参数量 N、训练 token 数 D 和训练计算量 C。对于 dense Transformer，可用非常粗略的近似表达：

~~~math
C\approx kND
~~~

其中 k 取决于架构、前向反向实现、序列长度、精度和是否统计通信。这个式子只用于数量关系，不可直接当作某硬件上的实际 FLOPs 计数。

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

~~~python
def predicted_loss(scale, floor=1.0, amplitude=3.0, exponent=0.22):
    return floor + amplitude / (scale ** exponent)


def shifted_loss(scale, penalty=0.18):
    return predicted_loss(scale) + penalty


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

这个实验没有拟合真实模型，它只说明：小规模曲线可以非常平滑，但大规模阶段若数据、架构或训练 recipe 发生系统变化，真实结果会整体偏离。正确做法是把中等规模结果回填模型，而不是把外推误差归因于“Scaling Law 完全无效”。

### 32.9 预算决策的顺序

一个可复用的预算分析顺序是：先定义目标能力和最大延迟，再用多个小模型估计 loss 与能力曲线；同时测数据质量、重复暴露和训练稳定性；然后比较增加参数、增加有效 token、改善数据、加强后训练和增加推理计算的边际收益；最后把硬件、通信、存储和发布周期加入决策。

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

~~~python
import math
import torch
import torch.nn.functional as F


def lm_loss_and_ppl(logits, input_ids, ignore_index=None):
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("logits must be [batch, length, vocab] and ids must be [batch, length]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids must have matching batch and length")
    if input_ids.size(1) < 2 or logits.size(-1) <= 0:
        raise ValueError("sequence length must be at least two and vocab must be non-empty")
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
    return loss, math.exp(loss.item())


torch.manual_seed(0)
logits = torch.randn(2, 5, 8)
input_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 2, 3, 6]])
loss, ppl = lm_loss_and_ppl(logits, input_ids)
print("loss=", round(loss.item(), 4), "ppl=", round(ppl, 4))
~~~

`shift_logits` 与 `shift_labels` 的错位是语言模型 loss 中最常见的实现细节之一。若把同一位置的 logits 与同一位置的 input 当作目标，就会改变任务定义；若 padding 没有设为 ignore index，短样本会被无效位置污染。

### 33.8 checkpoint 选择不能只看最低 loss

base model 通常重视 validation loss 和分桶 loss；代码模型要看可执行测试；聊天模型要看指令遵循、偏好和安全；推理模型要看验证器通过率、采样成本和延迟。最低 overall loss 只是候选条件之一，不能替代目标任务选择。

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

### 34.8 finite 检查与裁剪代码

~~~python
import torch


def train_step(model, batch, optimizer, max_grad_norm=1.0):
    if not torch.isfinite(torch.tensor(max_grad_norm)).item() or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    optimizer.zero_grad(set_to_none=True)
    output = model(**batch)
    loss = output.loss
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite loss")

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    if not torch.isfinite(grad_norm):
        raise RuntimeError("non-finite gradient norm")
    optimizer.step()
    return float(loss.detach()), float(grad_norm)
~~~

在 AMP 中必须先把 scaled gradient unscale，再裁剪；这一点放到下一节详细解释。生产训练还要避免在异常后无条件执行 optimizer.step，并保证最后一个健康 checkpoint 不被覆盖。

## 35. 混合精度：FP16、BF16 与 FP8 的数值取舍

### 35.1 浮点格式的两个维度

浮点数的指数位主要决定动态范围，尾数位主要决定相对精度。数值范围不足会 overflow 或 underflow；精度不足会让相近数被舍入到相同表示。训练中两种问题都可能影响收敛，但大模型首先需要防止 Inf、NaN 和梯度归零。

FP32 使用 32 bit，范围和精度都较好；FP16 使用 16 bit，节省显存但指数位少；BF16 也是 16 bit，却保留接近 FP32 的指数范围，尾数更短；FP8 进一步压缩到 8 bit，需要显式缩放和高精度累加。

### 35.2 FP16 与 loss scaling

若某个真实梯度 g 太小，转换到 FP16 后可能下溢为 0。loss scaling 先把 loss 乘以 s：

~~~math
\widetilde L=sL,\qquad \nabla_\theta\widetilde L=s\nabla_\theta L
~~~

反向传播后再把梯度除以 s，恢复真实尺度。如果放大后出现 Inf，则跳过本次更新并降低 s；若连续稳定，可以逐步提高 s。动态 loss scaling 的目的不是提高模型目标，而是保护低精度梯度中的小数值。

AMP 的正确顺序是：autocast 前向，scale loss，反向，unscale 梯度，gradient clipping，optimizer step，更新 scaler。若在 unscale 之前裁剪，阈值对应的是放大后的梯度，监控和裁剪都失去原有含义。

### 35.3 为什么 BF16 常用于大模型

BF16 的指数位与 FP32 接近，因此大激活和梯度更不容易溢出，通常不需要 FP16 那样的动态 loss scaling。它的尾数精度较低，所以并非“完全等同 FP32”；loss、归一化统计、reduction、optimizer state 和关键累加仍常保留 FP32。

在支持 BF16 的硬件上，BF16 往往是效率与稳定性的折中。旧硬件、特定 kernel 或精细数值任务可能不支持或不适合 BF16，因此 dtype 选择必须与设备和实现绑定。

### 35.4 Master weights 与 optimizer state

低精度参数更新可能丢掉很小的增量。例如参数约为 1.0，而更新量只有 1e-7，低精度表示可能无法保留。常见做法是用低精度权重做矩阵计算，同时维护 FP32 master weights 做 optimizer update。

AdamW 还需要一阶矩 m 和二阶矩 v。若它们用过低精度保存，更新统计可能产生更大误差。因此“模型用 BF16”不等于“所有状态都用 BF16”，显存规划必须把参数、梯度、master weights、optimizer states 和 activation 分开计算。

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

### 35.7 CPU 上观察范围和精度

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

这个实验能直观看到 FP16 更早达到有限值上限，而 BF16 的范围更接近 FP32；同时 BF16 的 `eps` 更大，说明它的有效精度更粗。具体硬件上的吞吐和 kernel 行为仍要用真实设备测量。

### 35.8 精度迁移的验证顺序

从 FP32 切换到 BF16、FP16 或 FP8 时，先在单卡小模型上对比 loss、grad norm 和 finite 状态，再做多卡短跑；之后比较 validation loss、下游任务、overflow、skipped steps、吞吐和显存；最后才在目标规模上长跑。只验证“程序能启动”不足以证明低精度没有伤害模型。

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

在 FSDP、ZeRO 或 tensor parallel 中，参数和 optimizer state 分散在多个 rank。sharded checkpoint 让各 rank 并行写入，适合超大模型；consolidated checkpoint 方便单机推理和发布，却可能需要额外聚合内存与 I/O。

恢复时不一定保持相同 GPU 数。若要从 128 卡保存的状态恢复到 64 卡，需要 resharding 和可靠的元数据。只把 shard 文件拼接起来通常不够，因为参数布局、optimizer 分片和并行组都可能变化。

### 36.5 异步保存的一致性

大 checkpoint 写入可能需要很长时间。异步保存可以让训练更快继续，但必须保证“写入完成”与“状态可用”之间有明确协议。常见流程是写临时目录、完成所有 shard、写 manifest、校验 checksum、原子切换目录名，再把 status 标成 complete。加载器只接受 complete 且校验通过的状态。

### 36.6 恢复后 loss 跳变的定位

先检查 checkpoint 是否完整，再检查模型 key、shape 和 dtype；接着核对 optimizer、scheduler、scaler、RNG、dataloader、consumed tokens、训练 config 和 rank mapping。若可以复现，比较中断前后同一 batch 的 forward loss、grad norm 和参数更新；若只在多卡发生，重点检查 shard 与通信。

严格 bitwise reproducibility 需要相同硬件、kernel、随机数、数据顺序和并行策略。许多大规模训练只要求 statistical reproducibility，即恢复后的趋势和最终能力接近；但无论目标是哪一种，都应避免未解释的巨大 loss 跳变。

### 36.7 一个简化的保存与恢复示例

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
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    return state["step"], state["data_state"]
~~~

生产系统还要保存 CUDA RNG、GradScaler、分布式分片、版本和 manifest。加载不可信 checkpoint 时必须遵守框架的安全加载约束；训练状态文件不能因为扩展名是 `.pt` 就被当作可信输入。

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

~~~python
checkpoints = [
    {"name": "a", "val": 1.90, "code": 1.48, "math": 0.41, "safety": 0.97, "stable": True},
    {"name": "b", "val": 1.87, "code": 1.46, "math": 0.45, "safety": 0.95, "stable": True},
    {"name": "c", "val": 1.85, "code": 1.59, "math": 0.47, "safety": 0.88, "stable": False},
]
if not checkpoints:
    raise ValueError("at least one checkpoint is required")

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

这段代码展示的是决策顺序，不是通用权重。不同产品可以把代码通过率、语言一致性、延迟或偏好指标放在目标函数中，但规则应在看结果之前确定，避免事后只挑最有利的 checkpoint。

### 37.8 评估频率的分层

高频层计算 loss、分桶 loss 和少量 canary；中频层运行 benchmark 子集；低频层运行完整能力、安全和人工评估；最终候选才使用 hidden eval。评估成本也要进入训练预算，70B 模型的完整代码和安全评估可能比一次小模型训练更贵。

## 38. 多语言预训练：数据不均衡、迁移与干扰

### 38.1 多语言不是简单拼接

多语言模型要同时解决数据数量不均衡、质量差异、tokenizer 效率、共享容量竞争、跨语言迁移和评估覆盖不足。按原始网页量训练通常让英文等高资源语言占主导；完全均匀采样又可能让低资源语言重复过多。

产品目标决定配比：全球助手、中英技术助手、翻译模型和低资源语言模型需要不同的数据、tokenizer 和评估，不能用一个“多语言比例”概括。

### 38.2 温度采样与重复暴露

对语言 i 的原始比例 p_i，可以使用：

~~~math
q_i=\frac{p_i^\alpha}{\sum_jp_j^\alpha}
~~~

α<1 时低资源语言占比提高。若语言 i 的可用 token 量为 A_i，训练计划为 B_i，则重复暴露 e_i=B_i/A_i。过高 e_i 可能让模型记忆有限语料，验证 loss 好看但 fresh 任务变差。

因此要同时看每语言的样本质量、可用量、重复簇大小、per-language loss、原生 benchmark 和人工母语评估。语言之间的 loss 数值也不能简单横比，因为 tokenizer 粒度和语言结构不同。

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

多语言评估应把 tokenizer fertility 作为成本维度。若中文每字符一个 token，而英文每数个字符一个 token，那么相同 context window 中两种语言可表达的内容不同；在训练 token 预算相同的情况下，不同语言获得的字符量也不同。

这不意味着必须让所有语言的 token 数完全相等，而是要识别这种差异对训练暴露、推理成本、长上下文和 benchmark 结果的影响，并在采样和系统容量中体现。

### 38.7 多语言评估组合

一个完整的评估组合应包括：每语言 validation loss 趋势、原生语言 benchmark、翻译与跨语言问答、语言一致性、代码与数学任务、真实用户任务、人类母语评估和多语言安全测试。翻译 benchmark 有价值，却不能替代原生任务，因为翻译可能改变文化背景、术语和题意。

安全也不能只在英文测试。拒答边界、提示注入、隐私泄漏和有害内容在低资源语言中可能更弱，应把语言作为安全评估的分桶维度。

### 38.8 采样表的教学实现

~~~python
import math


def language_plan(available, budget, temperature):
    if not available or any(
        not math.isfinite(amount) or amount <= 0
        for amount in available.values()
    ):
        raise ValueError("available data must be non-empty and positive")
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and non-negative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    total = sum(available.values())
    raw = {lang: amount / total for lang, amount in available.items()}
    alpha = 1.0 / temperature
    weights = {lang: ratio ** alpha for lang, ratio in raw.items()}
    normalizer = sum(weights.values())
    plan = {lang: budget * weight / normalizer for lang, weight in weights.items()}
    exposure = {lang: plan[lang] / available[lang] for lang in available}
    return plan, exposure


available = {"en": 900, "zh": 80, "ar": 20, "sw": 2}
for temperature in [1.0, 2.0, 5.0]:
    plan, exposure = language_plan(available, 100, temperature)
    print("T=", temperature)
    print("plan=", {k: round(v, 2) for k, v in plan.items()})
    print("exposure=", {k: round(v, 2) for k, v in exposure.items()})
~~~

斯瓦希里语等低资源语言在高温度下会获得更多训练比例，但曝光倍数也可能非常高。模型最终是否受益，需要由原生 fresh 评估和人工质量验证回答。

## 39. 代码模型预训练：结构、执行反馈与软件工程任务

### 39.1 代码与普通文本的差异

代码具有语法约束、长程依赖、可执行反馈和许可风险。一个字符或缩进错误可能使程序无法运行；函数可能依赖其他文件、配置、测试和版本环境；仓库中还包含 README、issue、PR 和 commit message 等自然语言上下文。

因此代码模型数据不应只是一堆文件内容。更完整的样本单位可以包含：文件路径、仓库结构、依赖、相关测试、错误日志、变更 patch、文档和任务描述。

### 39.2 数据来源与治理

常见来源包括开源仓库、官方 API 文档、README、issue、PR、commit、技术问答、单元测试、合成任务和获授权的内部代码。采集时需要保存仓库 revision、license、语言、路径和时间，过滤二进制、minified 文件、vendor 依赖、自动生成代码、密钥和敏感配置。

代码去重要处理 fork、模板、复制粘贴、不同版本和 benchmark 污染。仅对完整文件做 hash 不够，还要在函数、token n-gram 和仓库关系上建立重复簇。

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

### 39.7 pass@k 的定义

若为一道题生成 n 个候选，其中 c 个通过测试，常用的 pass@k 无偏估计形式为：

~~~math
\widehat{\mathrm{pass@}k}
=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}
~~~

它估计从 n 个候选中不放回抽取 k 个时至少有一个正确候选的概率。pass@1 更接近一次生成可靠性；pass@k 还包含多次采样和筛选成本，不能把高 pass@k 直接解释为交互式代码助手的一次成功率。

### 39.8 一个 FIM 与 pass@k 示例

~~~python
from math import comb


def build_fim(code, start, end):
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
    return 1.0 - comb(n - c, k) / comb(n, k)


code = "def area(r):\n    return 3.14 * r * r\n"
start = code.index("    return")
sample = build_fim(code, start, len(code) - 1)
print(sample)
for k in [1, 5, 10]:
    print("pass@", k, "=", round(pass_at_k(10, 3, k), 3))
~~~

这个实现只用于展示公式。生产评估需要隔离执行环境、限制资源、固定依赖，并防止候选代码访问不应访问的文件或网络。

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

SWE-bench 等仓库任务把模型放进更接近真实的软件环境，但其结果仍依赖 harness、补丁应用、测试命令和环境版本。评估报告必须把这些条件写清楚。

### 39.10 代码安全与工具权限

代码能力提升也可能提升生成 SQL injection、XSS、命令注入、不安全反序列化、路径遍历、硬编码密钥和恶意脚本的能力。训练中应加入安全修复与安全解释数据，评估中要做静态扫描和红队任务，agent 运行时则要限制文件、网络、凭证和命令权限。

最小权限、沙箱、可审计 trace、人工确认和可回滚 patch 是系统设计的一部分，不是代码模型训练结束后再补的装饰。

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

一个教学性的预算可以是 1B token：45% 通用网页与书籍，25% 代码和测试，10% 数学，15% 中英文技术资料，5% 对话和工具格式。这个数字不是推荐配方；它只是要求每个桶都有明确用途、风险和评估，且可以在消融实验中被改动。

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
16. Wang and Kanwar, *BFloat16: The secret to high performance on Cloud TPUs*：BF16 的范围与硬件背景。
    https://arxiv.org/abs/1905.12322
17. Micikevicius et al., *FP8 Formats for Deep Learning*：FP8 格式与训练数值问题。
    https://arxiv.org/abs/2209.05433
18. Wang et al., *DeepNet: Scaling Transformers to 1,000 Layers*：深层 Transformer 稳定性与 DeepNorm。
    https://arxiv.org/abs/2203.00555
19. PyTorch, *Automatic Mixed Precision package*：autocast、GradScaler 与 AMP 工程接口。
    https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
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
