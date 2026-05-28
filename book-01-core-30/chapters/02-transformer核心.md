# 第二部分：Transformer 核心

## 第 6 讲：Tokenization

### 本讲目标

前面我们一直说大模型处理的是 token，而不是直接处理字符串。本讲就回答：token 到底是什么？文本如何变成 token？tokenization 为什么会影响模型能力、成本和上下文长度？

你需要掌握：

1. token、token id、vocabulary 是什么。
2. 为什么模型不能直接处理原始字符串。
3. 常见 tokenization 方法：word-level、character-level、subword。
4. BPE、WordPiece、SentencePiece 的核心直觉。
5. 中文、英文、代码、多语言场景下 tokenizer 的差异。
6. tokenizer 如何影响上下文长度、训练效率、推理成本和模型能力。

### 问题背景

神经网络处理的是数字，不是文字。

原始文本：

```text
I love machine learning.
```

模型不能直接吃这个字符串。通常要先经过 tokenizer，把文本变成 token，再把 token 映射成整数 id：

```text
text -> tokens -> token ids
```

例如：

```text
text:      I love machine learning.
tokens:    ["I", " love", " machine", " learning", "."]
token ids: [40, 3021, 5780, 6975, 13]
```

不同 tokenizer 切出来的结果可能完全不同。

### 这个问题为什么重要

Tokenizer 不是简单预处理，而是模型输入空间的定义。

它会影响：

1. 同样一段文本占多少 token。
2. 上下文窗口能装多少有效内容。
3. 训练和推理成本。
4. 多语言表现。
5. 代码能力。
6. 数字、符号、公式处理能力。
7. 模型是否容易复制、拼写、处理罕见词。

举例：

如果一个中文句子被切成很多碎片，那么同样 8K context window 能装的中文内容会更少，训练和推理成本也更高。

### Token、Vocabulary 和 Token ID

Token 是模型处理文本的基本离散单位。它可以是：

1. 一个词。
2. 一个子词。
3. 一个字符。
4. 一个字节片段。
5. 一个特殊符号。

Vocabulary 是所有 token 的集合。

Token id 是 token 在 vocabulary 中对应的整数编号。

例如：

```text
vocab = {
  "I": 40,
  " love": 3021,
  " machine": 5780,
  " learning": 6975,
  ".": 13
}
```

模型真正接收的是 token id 序列。

### 为什么不用 word-level tokenization

最直观的方法是按词切分。

例如：

```text
I love machine learning
-> ["I", "love", "machine", "learning"]
```

问题是：

1. 词表会非常大。
2. 罕见词和新词无法处理。
3. 拼写变化会造成大量 OOV。
4. 中文、日文等语言没有天然空格分词。
5. 代码、URL、数字、公式很难稳定处理。

OOV 指 out-of-vocabulary，也就是词表外 token。

### 为什么不用纯 character-level tokenization

另一种方法是按字符切分。

优点：

1. 词表很小。
2. 几乎不会 OOV。
3. 可以处理任意新词。

缺点：

1. 序列会变得很长。
2. 同样 context window 能装的信息变少。
3. 模型需要更多步才能组合出词义。
4. 训练和推理更慢。

例如：

```text
machine
-> ["m", "a", "c", "h", "i", "n", "e"]
```

这对长文本建模不够高效。

### Subword Tokenization

现代 LLM 常用 subword tokenization。

核心思想：

常见词可以作为完整 token，罕见词可以拆成更小的子词。

例如：

```text
unbelievable
-> ["un", "believ", "able"]
```

这样兼顾了：

1. 词表规模可控。
2. 序列长度不会像字符级那么长。
3. 罕见词仍然可以被拆分表示。
4. 对多语言和代码更灵活。

### BPE 的核心直觉

BPE 是 Byte Pair Encoding。

它的基本思路是：从小单位开始，不断合并最常见的相邻片段。

简化过程：

1. 初始时把文本拆成字符或字节。
2. 统计最常见的相邻 pair。
3. 把这个 pair 合并成新 token。
4. 重复直到达到目标词表大小。

例如语料中经常出现：

```text
l o
```

就合并为：

```text
lo
```

再经常出现：

```text
lo ve
```

就合并为：

```text
love
```

BPE 的优点：

1. 简单有效。
2. 常见片段会被压缩成更短 token 序列。
3. 罕见词可以拆成子词。

局限：

1. 合并基于频率，不一定符合语言学边界。
2. 对数字、空格、大小写、代码符号的处理依赖具体实现。
3. 不同语料训练出的 tokenizer 差异很大。

### WordPiece 和 SentencePiece

WordPiece 和 BPE 类似，也是一种子词方法。它通常根据 likelihood 改善选择合并，而不只是简单频率合并。BERT 系列模型常见 WordPiece。

SentencePiece 更像一个 tokenizer 工具和方法集合。它的特点是可以直接从原始文本训练，不依赖预先分词，常用于多语言模型。

SentencePiece 常把空格也当成普通符号处理，例如用特殊符号表示词边界。这对没有空格分词的语言更方便。

### Byte-Level Tokenization

Byte-level tokenizer 从字节层面保证所有输入都可表示。

优点：

1. 基本不会 OOV。
2. 可以处理任意 Unicode 字符。
3. 对代码、emoji、特殊符号更鲁棒。

缺点：

1. 某些语言可能 token 数更多。
2. 可读性差。
3. 对 tokenizer 实现细节依赖更强。

### 特殊 Token

大模型 tokenizer 里通常有特殊 token。

常见包括：

```text
BOS: beginning of sequence
EOS: end of sequence
PAD: padding
UNK: unknown
system/user/assistant: chat role token
tool: 工具调用相关 token
image/audio: 多模态占位 token
```

特殊 token 非常重要。它们不仅是文本片段，还定义了模型如何理解对话结构、何时停止、如何识别工具调用或多模态输入。

### 一个最小 BPE 代码例子

下面用纯 Python 写一个非常简化的 BPE 合并过程。它不是生产 tokenizer，只用于理解 BPE 的核心思想。

```python
from collections import Counter


def get_pairs(tokens):
    pairs = Counter()
    for token_seq in tokens:
        for a, b in zip(token_seq, token_seq[1:]):
            pairs[(a, b)] += 1
    return pairs


def merge_pair(tokens, pair):
    merged = []
    target = "".join(pair)
    for token_seq in tokens:
        new_seq = []
        i = 0
        while i < len(token_seq):
            if i < len(token_seq) - 1 and (token_seq[i], token_seq[i + 1]) == pair:
                new_seq.append(target)
                i += 2
            else:
                new_seq.append(token_seq[i])
                i += 1
        merged.append(new_seq)
    return merged


words = ["low", "lower", "newest", "widest"]
tokens = [list(word) + ["</w>"] for word in words]

for step in range(5):
    pairs = get_pairs(tokens)
    if not pairs:
        break
    best_pair, count = pairs.most_common(1)[0]
    tokens = merge_pair(tokens, best_pair)
    print(f"step={step}, merge={best_pair}, count={count}, tokens={tokens}")
```

这个例子展示：BPE 会不断把高频相邻片段合并成更大的 token。

### Tokenizer 如何影响上下文长度

模型的 context window 通常按 token 数计算，而不是按字符数或词数计算。

如果一个模型支持 8K token：

```text
不是 8K 个汉字
不是 8K 个英文单词
而是 8K 个 tokenizer 切出来的 token
```

同样一句话，不同 tokenizer 可能产生不同 token 数。

这会影响：

1. 能放进上下文的信息量。
2. attention 计算成本。
3. KV Cache 显存。
4. 推理延迟。
5. 训练 token 数统计。

### 中文、英文和代码的差异

英文中空格提供了天然词边界，很多 tokenizer 会把空格和后面的词一起编码，例如 `" learning"`。

中文没有空格，tokenizer 可能按字、词或子词切分。不同 tokenizer 对中文压缩率差异很大。

代码中有大量符号、缩进、变量名、数字和特殊字符。一个好的代码 tokenizer 需要高效处理：

1. 缩进。
2. 下划线。
3. 驼峰命名。
4. 括号和运算符。
5. 长变量名。
6. 数字和字符串。

### 真实项目中的坑

#### 坑 1：训练和推理 tokenizer 不一致

这会导致 token id 完全错位，模型行为不可预测。

#### 坑 2：特殊 token 没配置好

EOS 错误会导致模型停不下来。PAD 处理错误会污染 loss。chat role token 错误会导致对话格式混乱。

#### 坑 3：tokenizer 对目标语言压缩率差

同样内容占更多 token，会增加成本并减少有效上下文。

#### 坑 4：新增 token 后没有正确 resize embedding

如果给 tokenizer 新增特殊 token，需要同步扩展模型 embedding 矩阵，否则会报错或行为异常。

#### 坑 5：评估 token 成本时用错 tokenizer

不同模型 tokenizer 不同，不能用 A 模型 tokenizer 估算 B 模型成本。

### 优点、缺点和适用场景

Subword tokenizer 的优点：

1. 词表大小可控。
2. 罕见词可拆分。
3. 序列长度比字符级更短。
4. 能处理多语言和代码。

缺点：

1. 切分不一定符合语义。
2. 对不同语言不一定公平。
3. 数字和符号处理可能不稳定。
4. tokenizer 一旦选定，后续更换成本很高。

适用场景：

1. 大语言模型预训练。
2. 多语言模型。
3. 代码模型。
4. 对话模型。
5. 多模态模型中的文本部分。

### 面试官会怎么问

#### 问题 1：什么是 tokenization？

标准回答：

```text
Tokenization 是把原始文本转换成 token 序列，再映射成 token id 的过程。因为神经网络只能处理数字，tokenizer 定义了模型的离散输入空间。现代 LLM 通常使用 subword 或 byte-level tokenizer，以平衡词表大小、OOV 问题和序列长度。
```

#### 问题 2：BPE 的核心思想是什么？

回答框架：

1. 从字符或字节开始。
2. 统计高频相邻 pair。
3. 不断合并高频 pair。
4. 最终得到常见词更短、罕见词可拆的子词词表。

#### 问题 3：Tokenizer 如何影响大模型成本？

回答框架：

1. context window 按 token 计算。
2. token 数越多，attention 和 KV Cache 成本越高。
3. 压缩率差会导致训练 token 数和推理成本上升。
4. 多语言和代码场景尤其明显。

#### 问题 4：新增特殊 token 要注意什么？

回答框架：

1. 更新 tokenizer vocabulary。
2. resize model embedding。
3. 确保训练和推理模板一致。
4. 检查特殊 token 是否参与 loss。

### 常见误区

1. 误区：token 就是英文单词。
   纠正：token 可以是词、子词、字符、字节片段或特殊符号。

2. 误区：tokenizer 只是无关紧要的预处理。
   纠正：tokenizer 决定输入空间，影响成本、上下文长度、多语言和代码能力。

3. 误区：词表越大越好。
   纠正：词表大可能缩短序列，但会增加 embedding 和输出层参数，也可能影响罕见组合泛化。

4. 误区：训练后可以随便换 tokenizer。
   纠正：tokenizer 和模型 embedding 强绑定，随便更换会破坏模型输入含义。

### 小练习

1. 用自己的话解释 token、vocabulary、token id。
2. 比较 word-level、character-level、subword tokenization 的优缺点。
3. 手动模拟一次 BPE 合并过程。
4. 解释为什么中文 tokenizer 压缩率会影响上下文长度。
5. 思考：如果新增 `<tool>` token，需要改模型哪些部分？

### 本讲总结

本讲最重要的结论：

1. 神经网络不能直接处理字符串，文本必须先变成 token id。
2. Tokenizer 定义了模型的离散输入空间。
3. 现代 LLM 通常使用 subword 或 byte-level tokenizer。
4. BPE 的核心是不断合并高频相邻片段。
5. Tokenizer 会影响上下文长度、训练成本、推理成本、多语言能力和代码能力。
6. 特殊 token、chat template、EOS、PAD 等工程细节非常重要。
7. 训练和推理 tokenizer 必须一致。

关键问题：什么是 token、BPE、WordPiece、SentencePiece，以及 tokenizer 如何影响上下文长度、训练效率和模型能力。

## 第 7 讲：Embedding 与位置编码

### 本讲目标

第 6 讲我们知道了文本会被 tokenizer 转成 token id。但 token id 只是整数编号，模型不能直接从编号大小中理解语义。

本讲回答：token id 如何变成模型可以处理的向量？Transformer 为什么还需要位置信息？

你需要掌握：

1. token embedding 是什么。
2. embedding matrix 的形状和训练方式。
3. 为什么 token id 不能直接当作数值输入。
4. 为什么 Transformer 需要位置编码。
5. 绝对位置编码、相对位置编码、RoPE 的基本区别。
6. embedding 和输出层权重共享的直觉。

### 问题背景

Tokenizer 输出的是 token id：

```text
[40, 3021, 5780, 6975]
```

这些 id 只是词表中的编号。

编号大小本身没有语义。例如：

```text
token id 100 不一定比 token id 50 更“大”或更重要
```

神经网络需要连续向量作为输入。因此第一步是把每个 token id 查表变成向量。

这张表就是 embedding matrix。

### Token Embedding 是什么

Token embedding 是把离散 token id 映射成连续向量的过程。

假设：

$$
V=|\mathcal{V}|,\qquad d=d_{\mathrm{model}}
$$

那么 embedding matrix 的形状是：

$$
[V,d]
$$

每一行对应一个 token 的向量表示。

如果 token id 是 `i`，那么它的 embedding 就是：

$$
E_i
$$

例如：

```text
input_ids: [B, T]
```

```text
embedding weight: [V, d]
```

```text
output embeddings: [B, T, d]
```

其中：

```text
B = batch size
T = sequence length
d = hidden size
```

### 为什么不能直接用 token id

token id 是类别编号，不是连续数值。

如果直接把 id 当数字输入，模型会误以为：

```text
id 100 比 id 50 大
id 200 和 id 201 更接近
```

但这些关系通常没有意义。

Embedding 的作用是让模型自己学习每个 token 的向量表示。训练过程中，如果两个 token 在类似上下文中出现，它们的 embedding 可能会逐渐变得相似。

例如：

```text
king, queen, prince
```

这些词可能在 embedding 空间中更接近。

注意：不要过度神化 embedding。它不是直接存储“完整语义”，而是模型训练过程中学到的可用表示。

### Embedding 如何训练

Embedding matrix 是模型参数的一部分。

训练时：

1. token id 查表得到 embedding。
2. embedding 进入 Transformer 层。
3. 模型输出 logits。
4. 计算 loss。
5. 反向传播更新 embedding matrix。

也就是说，embedding 不是人工设计的，而是通过语言模型训练目标学出来的。

### 一个最小 PyTorch 例子

```python
import torch
import torch.nn as nn

vocab_size = 10000
hidden_size = 768
batch_size = 2
seq_len = 4

embedding = nn.Embedding(vocab_size, hidden_size)

input_ids = torch.tensor([
    [40, 3021, 5780, 6975],
    [10, 20, 30, 40],
])

outputs = embedding(input_ids)

print(outputs.shape)  # [2, 4, 768]
```

这个例子说明：embedding 层做的是查表操作，不是复杂神经网络计算。

### 为什么需要位置编码

Self-attention 本身对输入顺序不敏感。

如果没有位置信息，Transformer 看到：

```text
I love you
```

和：

```text
you love I
```

可能很难区分顺序差异。

因为 attention 主要根据 token 内容计算相似度，它本身不会天然知道哪个 token 在第 1 个位置，哪个在第 3 个位置。

所以 Transformer 需要额外注入位置信息。

### 绝对位置编码

最直观的方法是给每个位置一个向量。

假设最大长度是 `max_seq_len`，hidden size 是 `d`，那么 position embedding matrix 形状是：

$$
[T_{\max},d]
$$

第 `t` 个位置对应：

$$
P_t
$$

输入表示通常是：

$$
h_t=E_{x_t}+P_t
$$

优点：

1. 简单。
2. 容易实现。
3. 可以学习任务相关的位置模式。

缺点：

1. 对超过训练长度的位置泛化较差。
2. 位置是绝对编号，不直接表达相对距离。

### 正弦位置编码

Transformer 原论文使用 sinusoidal positional encoding。

它不是学习出来的参数，而是用固定函数生成：

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

直觉：不同维度用不同频率的正弦和余弦表示位置。

优点：

1. 不增加可学习参数。
2. 有一定长度外推能力。

缺点：

1. 不一定最适合现代 LLM。
2. 实践中很多模型改用 RoPE、ALiBi 或其他方法。

### 相对位置思想

语言中很多关系依赖相对距离，而不是绝对位置。

例如：

```text
the cat sat on the mat
```

模型关心的是 `cat` 和 `sat` 相距多远，而不只是它们分别在第几个位置。

相对位置编码试图让模型更好地感知 token 之间的相对距离。

后面第 14 讲会详细讲 RoPE 和长上下文，这里先建立直觉。

### RoPE 的基本直觉

RoPE 是现代 LLM 常见的位置编码方法。

它不是简单把 position embedding 加到 token embedding 上，而是把位置信息注入到 attention 的 Q 和 K 中。

直觉：

1. 每个位置对应一个旋转角度。
2. Q 和 K 根据位置进行旋转。
3. 两个 token 的 attention score 会自然包含相对位置信息。

本讲只需要知道 RoPE 是一种相对位置友好的方法，详细数学后面再讲。

### 输出层和 Embedding 权重共享

语言模型输入时需要把 token id 变成向量。

输出时需要把 hidden state 变回词表 logits。

输出层通常是：

$$
\ell=hW_{\mathrm{out}}^T
$$

其中 `output_weight` 形状也是：

$$
[V,d]
$$

很多模型会让输入 embedding matrix 和输出层 weight 共享参数，叫 weight tying。

优点：

1. 减少参数量。
2. 输入 token 表示和输出 token 分类共享语义空间。
3. 在语言模型中通常效果不错。

### 真实项目中的坑

#### 坑 1：新增 token 后忘记 resize embedding

如果 tokenizer 新增 token，embedding matrix 的行数也要扩展。

#### 坑 2：position embedding 长度不够

如果模型使用绝对位置 embedding，输入超过最大长度会报错或行为异常。

#### 坑 3：错误地认为 embedding 本身已经包含全部上下文语义

初始 token embedding 是静态查表，真正的上下文语义来自 Transformer 层后的 contextual representation。

#### 坑 4：训练和推理位置处理不一致

长上下文扩展、RoPE scaling、position id 处理不一致，会导致推理质量下降。

### 优点、缺点和适用场景

Token embedding 的优点：

1. 把离散 token 转成连续向量。
2. 可训练。
3. 能学习 token 间统计关系。

缺点：

1. 依赖 tokenizer。
2. 词表大时参数量大。
3. 静态 embedding 不包含上下文信息。

位置编码的优点：

1. 让 Transformer 感知顺序。
2. 支持序列建模。
3. 相对位置方法有助于长上下文。

缺点：

1. 绝对位置编码外推能力有限。
2. 不同位置编码方案对长上下文影响很大。
3. 工程实现细节容易出错。

### 面试官会怎么问

#### 问题 1：Embedding 层做了什么？

标准回答：

```text
Embedding 层把离散 token id 映射成连续向量。它本质上是一个可训练查表操作，embedding matrix 的形状是 [vocab_size, hidden_size]。输入 token ids 的 shape 是 [B, T]，输出 embedding 的 shape 是 [B, T, hidden_size]。
```

#### 问题 2：为什么 Transformer 需要位置编码？

回答框架：

1. Self-attention 本身对顺序不敏感。
2. 语言序列顺序很重要。
3. 位置编码向模型注入 token 的位置信息。
4. 常见方法包括绝对位置编码、正弦位置编码、RoPE、ALiBi。

#### 问题 3：Token embedding 和 contextual representation 有什么区别？

回答框架：

1. Token embedding 是静态查表结果。
2. Contextual representation 是经过 Transformer 层、结合上下文后的动态表示。
3. 同一个 token 在不同句子中的 contextual representation 可以不同。

#### 问题 4：新增 token 后要注意什么？

回答框架：

1. 更新 tokenizer。
2. resize embedding matrix。
3. 初始化新 token embedding。
4. 检查输出层是否 weight tying。
5. 确保训练和推理都使用同一 tokenizer。

### 常见误区

1. 误区：token id 大小有语义。
   纠正：token id 只是类别编号，语义来自 embedding 和后续训练。

2. 误区：embedding 是固定词向量，不会训练。
   纠正：LLM 的 embedding matrix 通常是可训练参数。

3. 误区：Transformer 天然知道顺序。
   纠正：self-attention 本身不包含顺序，需要位置编码。

4. 误区：同一个 token 在任何上下文里表示都一样。
   纠正：输入 embedding 一样，但经过 Transformer 后的上下文表示不同。

### 小练习

1. 假设 `vocab_size=50000`，`hidden_size=4096`，计算 token embedding 参数量。
2. 写一个 PyTorch `nn.Embedding` 示例，输入 `[B, T]`，输出 `[B, T, d]`。
3. 解释为什么 token id 不能直接当连续数值输入。
4. 举例说明为什么顺序对语言理解重要。
5. 用自己的话解释 token embedding 和 position embedding 的区别。

### 本讲总结

本讲最重要的结论：

1. Token id 是离散编号，不能直接表示语义。
2. Token embedding 是可训练查表，形状是 `[vocab_size, hidden_size]`。
3. Embedding 把 token id 变成连续向量。
4. Self-attention 本身不感知顺序，Transformer 需要位置编码。
5. 绝对位置编码简单但长度外推有限。
6. RoPE 等方法把相对位置信息更自然地注入 attention。
7. 新增 token、扩展上下文和位置处理都是真实工程中的高频坑。

关键问题：为什么 token 需要变成向量，embedding 矩阵如何训练，Transformer 为什么需要位置编码。

## 第 8 讲：Self-Attention 直觉

### 本讲目标

前面我们知道 token 会先变成 embedding，并加入位置信息。但一个 token 的初始 embedding 仍然是不含上下文的。

例如 `apple` 在下面两句话中含义不同：

```text
I ate an apple.
Apple released a new product.
```

同一个 token 需要根据上下文获得不同表示。Self-attention 就是解决这个问题的核心机制。

本讲先不推公式，目标是建立直觉：

1. self-attention 解决了什么问题。
2. 为什么每个 token 需要看其他 token。
3. Query、Key、Value 的直觉是什么。
4. attention 为什么适合并行。
5. attention 和 RNN、CNN 相比有什么优势。

### 问题背景

语言理解依赖上下文。

看这个例子：

```text
The animal didn't cross the street because it was too tired.
```

这里 `it` 更可能指 `animal`。

再看：

```text
The animal didn't cross the street because it was too wide.
```

这里 `it` 更可能指 `street`。

同样是 `it`，含义由上下文决定。

模型要理解一句话，就不能把每个 token 孤立处理。它需要让每个 token 根据其他 token 的信息更新自己。

### RNN 和 CNN 的局限

在 Transformer 之前，序列建模常用 RNN 或 CNN。

RNN 从左到右处理序列：

```text
x_1 -> x_2 -> x_3 -> ... -> x_T
```

优点：天然处理顺序。

缺点：

1. 难以并行。
2. 长距离依赖容易衰减。
3. 后面的 token 要等前面的 token 处理完。

CNN 用局部窗口处理序列。

优点：可以并行，局部模式建模强。

缺点：

1. 感受野有限。
2. 长距离交互需要堆很多层。
3. 对任意两个 token 的直接交互不够灵活。

Self-attention 的核心优势是：每个 token 可以直接和序列中其他 token 建立联系。

### 核心直觉

一句话：

Self-attention 让每个 token 根据当前上下文，决定应该重点关注哪些 token，然后汇总它们的信息来更新自己。

例如句子：

```text
The cat sat on the mat.
```

当模型更新 `sat` 的表示时，它可能关注：

1. `cat`：谁坐？
2. `mat`：坐在哪里？
3. `on`：动作和位置关系。

不同 token 关注的信息不同。

`cat` 可能更关注 `sat`。

`mat` 可能更关注 `on` 和 `sat`。

这就是 attention 的直觉。

### 为什么叫 Self-Attention

因为 Query、Key、Value 都来自同一个序列本身。

如果输入序列是：

```text
x_1, x_2, ..., x_T
```

self-attention 会让序列中的每个位置去关注同一个序列中的其他位置。

它和 encoder-decoder attention 不同。encoder-decoder attention 中，decoder 的 token 会关注 encoder 输出。self-attention 则是“自己看自己”。

### Query、Key、Value 的直觉

可以把 attention 想象成一次信息检索。

每个 token 会产生三种向量：

1. Query：我想找什么信息。
2. Key：我能提供什么索引。
3. Value：我真正携带的信息内容。

类比搜索：

```text
Query = 搜索请求
Key = 文档索引
Value = 文档内容
```

一个 token 的 Query 会和所有 token 的 Key 做匹配，得到注意力权重。然后用这些权重对 Value 做加权求和。

直觉上：

1. Query 和某个 Key 越匹配，就越关注那个 token。
2. 被关注 token 的 Value 会更多进入当前 token 的新表示。

### 一个例子

句子：

```text
The bank raised interest rates.
```

`bank` 可能有多个含义：河岸、银行。

当 `bank` 看上下文时，它会关注：

1. `interest`
2. `rates`
3. `raised`

这些 token 提供金融语境，所以 `bank` 的上下文表示会偏向“银行”。

再看：

```text
The boat reached the bank.
```

这里 `bank` 会关注：

1. `boat`
2. `reached`

上下文更像“河岸”。

这说明 self-attention 产生的是上下文相关表示，而不是固定词义。

### Attention 权重是什么

Attention 权重表示：当前 token 在更新自己时，从其他 token 中拿多少信息。

例如对 `sat`：

```text
The   0.05
cat   0.40
sat   0.20
on    0.10
the   0.05
mat   0.20
```

这不是模型真实一定学到的权重，只是直觉例子。

加权求和后，`sat` 的新表示就融合了 `cat`、`mat` 等上下文信息。

### 为什么 Attention 可以并行

RNN 的问题是第 `t` 个位置依赖第 `t-1` 个位置的隐藏状态，所以难以完全并行。

Self-attention 中，每个 token 的 Q、K、V 都可以一次性算出来。

然后所有 token 两两计算匹配分数。

这使得 Transformer 在训练时可以高效利用 GPU 并行矩阵乘法。

这也是 Transformer 能够 scale 到大模型的重要原因之一。

### Self-Attention 解决了什么问题

它主要解决三个问题：

1. 上下文相关表示：同一个 token 在不同上下文中有不同表示。
2. 长距离依赖：任意两个 token 可以直接交互。
3. 并行训练：相比 RNN 更适合 GPU 并行。

例如：

```text
The book that I borrowed from the library yesterday was fascinating.
```

`was` 需要和较远的 `book` 对应。self-attention 可以让它们直接交互。

### Self-Attention 的代价

Self-attention 的主要代价是复杂度。

如果序列长度是 `T`，每个 token 都要和其他 token 计算关系，所以 attention score 矩阵大小是：

$$
[T,T]
$$

计算和显存复杂度大致是：

$$
O(T^2)
$$

这就是长上下文困难的重要原因之一。

后面的 FlashAttention、稀疏 attention、长上下文方法，都在尝试缓解这个问题。

### Causal Self-Attention 的直觉

GPT 是自回归模型，生成第 `t` 个 token 时不能看到未来 token。

所以 GPT 使用 causal self-attention。

也就是说，第 `t` 个位置只能关注：

$$
1,2,\ldots,t
$$

不能关注：

$$
t+1,t+2,\ldots
$$

这通过 causal mask 实现。

本讲先建立直觉，第 11 讲会详细讲 causal mask。

### 真实项目中的坑

#### 坑 1：把 attention 权重当成完整解释

attention 权重能提供一些线索，但不等于模型推理的完整因果解释。

#### 坑 2：忽略 $O(T^2)$ 成本

序列长度翻倍，attention score 矩阵规模约变成 4 倍。长上下文成本会迅速上升。

#### 坑 3：mask 写错

causal mask 写错会导致模型看到未来 token，训练 loss 虚低，推理崩坏。

#### 坑 4：把 self-attention 和 cross-attention 混淆

self-attention 的 Q/K/V 来自同一序列，cross-attention 的 Q 和 K/V 来自不同来源。

### 优点、缺点和适用场景

Self-attention 的优点：

1. 能建模长距离依赖。
2. 每个 token 可以动态选择关注对象。
3. 训练时并行性强。
4. 能产生上下文相关表示。
5. 适合大规模预训练。

缺点：

1. $O(T^2)$ 复杂度高。
2. 长上下文显存压力大。
3. attention 权重不等于完全可解释性。
4. 实现中 mask、shape、数值稳定性容易出错。

适用场景：

1. 语言模型。
2. 机器翻译。
3. 代码模型。
4. 多模态模型。
5. 长文理解和生成。

### 面试官会怎么问

#### 问题 1：Self-attention 解决了什么问题？

标准回答：

```text
Self-attention 让序列中每个 token 根据上下文动态关注其他 token，从而得到上下文相关表示。相比 RNN，它更容易并行，也能让任意两个 token 直接交互，因此更适合建模长距离依赖和大规模训练。
```

#### 问题 2：Query、Key、Value 的直觉是什么？

回答框架：

1. Query 表示当前 token 想找什么。
2. Key 表示每个 token 可被匹配的索引。
3. Value 表示真正被汇总的信息。
4. Query 和 Key 匹配产生权重，权重加权 Value 得到输出。

#### 问题 3：Attention 和 RNN 相比有什么优势？

回答框架：

1. Attention 训练时更并行。
2. 任意两个 token 可以直接交互。
3. 长距离依赖路径更短。
4. 代价是 $O(T^2)$ 复杂度。

#### 问题 4：Attention 权重能解释模型吗？

回答框架：

1. Attention 权重可以提供部分可视化线索。
2. 但它不等价于完整因果解释。
3. 模型内部还有 MLP、残差、多层表示等复杂机制。

### 常见误区

1. 误区：attention 就是简单找关键词。
   纠正：attention 是动态的信息路由机制，不只是关键词匹配。

2. 误区：attention 权重越高说明越重要。
   纠正：权重有参考价值，但不能直接等同于因果重要性。

3. 误区：self-attention 没有顺序概念。
   纠正：attention 本身不含顺序，但 Transformer 会通过位置编码注入顺序。

4. 误区：attention 可以免费解决长上下文。
   纠正：标准 attention 的 $O(T^2)$ 成本是长上下文的核心瓶颈。

### 小练习

1. 用自己的话解释 self-attention。
2. 举一个例子说明同一个 token 在不同上下文中含义不同。
3. 用搜索引擎类比解释 Q、K、V。
4. 比较 RNN 和 self-attention 在长距离依赖上的差异。
5. 解释为什么标准 attention 的复杂度是 $O(T^2)$。

### 本讲总结

本讲最重要的结论：

1. Self-attention 让每个 token 根据上下文更新自己的表示。
2. Q、K、V 可以理解为查询、索引和内容。
3. Attention 权重决定从其他 token 聚合多少信息。
4. Self-attention 比 RNN 更适合并行训练。
5. 任意两个 token 可以直接交互，有利于长距离依赖。
6. 标准 attention 的代价是 $O(T^2)$。
7. Attention 权重有解释线索，但不是完整因果解释。

关键问题：attention 解决了 RNN 的什么问题，Query、Key、Value 的直觉是什么，attention 为什么可以并行。

## 第 9 讲：Self-Attention 公式推导

### 本讲目标

第 8 讲我们用直觉理解了 self-attention：每个 token 根据上下文动态关注其他 token。本讲进入公式和张量形状。

你需要掌握：

1. Q、K、V 如何由输入得到。
2. scaled dot-product attention 的完整公式。
3. 为什么使用 $QK^T$。
4. 为什么要除以 $\sqrt{d_k}$。
5. softmax 后的 attention weights 表示什么。
6. attention 的计算复杂度和显存复杂度。
7. full attention、causal attention、local attention、sparse attention、linear attention、FlashAttention 等变体解决什么问题。

### 从输入到 Q、K、V

假设输入 hidden states 是：

$$
X: [B,T,d_{\text{model}}]
$$

其中：

$$
B=\text{batch size},\qquad T=\text{sequence length},\qquad d_{\text{model}}=\text{hidden size}
$$

Self-attention 会通过三个线性变换得到 Q、K、V：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

如果单头 attention 的 head dimension 是 `d_k`，那么：

$$
W_Q:[d_{\text{model}},d_k],\qquad W_K:[d_{\text{model}},d_k],\qquad W_V:[d_{\text{model}},d_v]
$$

$$
Q:[B,T,d_k],\qquad K:[B,T,d_k],\qquad V:[B,T,d_v]
$$

通常 `d_k = d_v`。

### Scaled Dot-Product Attention 公式

标准公式：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

分步骤看：

1. $QK^T$ 计算每个 query 和每个 key 的相似度。
2. 除以 $\sqrt{d_k}$ 做缩放，避免分数过大。
3. softmax 把分数变成概率分布。
4. 用 attention weights 对 V 做加权求和。

### Shape 推导

对单头 attention：

$$
Q:[B,T,d_k],\qquad K:[B,T,d_k],\qquad V:[B,T,d_v]
$$

计算 attention scores：

$$
\text{scores}=QK^T
$$

shape：

$$
[B,T,d_k]\times[B,d_k,T]\to[B,T,T]
$$

`scores[b, i, j]` 表示第 `i` 个 token 对第 `j` 个 token 的关注分数。

softmax 后：

$$
\text{weights}:[B,T,T]
$$

再乘 V：

$$
[B,T,T]\times[B,T,d_v]\to[B,T,d_v]
$$

输出 shape：

$$
[B,T,d_v]
$$

### 为什么使用 QK^T

$QK^T$ 是在计算 query 和 key 的匹配程度。

点积越大，说明两个向量方向越相似，当前 token 越应该关注对应 token。

直觉：

```text
Query: 我想找什么
Key:   我有什么索引
Q · K: 我想找的东西和你的索引匹配程度
```

### 为什么除以 sqrt(d_k)

如果 $d_k$ 很大，点积 $Q\cdot K$ 的方差会变大。

分数太大时，softmax 会变得非常尖锐：

```text
一个位置概率接近 1，其他位置接近 0
```

这会导致梯度变小，训练不稳定。

除以 $\sqrt{d_k}$ 可以让 scores 的尺度更稳定。

这就是 scaled dot-product attention 中 “scaled” 的含义。

### Mask 如何加入

对于 causal LM，不能看到未来 token。通常会在 softmax 前对未来位置加一个很大的负数：

$$
\text{scores}=\text{scores}+\text{mask}
$$

其中未来位置 mask 为：

```text
-inf
```

softmax 后这些位置概率接近 0。

这保证第 `t` 个位置只能关注 `<= t` 的位置。

### PyTorch 最小实现

```python
import math
import torch


def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k: [B, T, d_k]
    # v:    [B, T, d_v]
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # [B, T, T]

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = torch.softmax(scores, dim=-1)             # [B, T, T]
    output = weights @ v                                # [B, T, d_v]
    return output, weights


B, T, d = 2, 4, 8
q = torch.randn(B, T, d)
k = torch.randn(B, T, d)
v = torch.randn(B, T, d)

output, weights = scaled_dot_product_attention(q, k, v)
print(output.shape)   # [2, 4, 8]
print(weights.shape)  # [2, 4, 4]
```

### Causal Mask 代码

```python
T = 4
causal_mask = torch.tril(torch.ones(T, T)).unsqueeze(0)  # [1, T, T]

output, weights = scaled_dot_product_attention(q, k, v, mask=causal_mask)
print(causal_mask)
```

mask 形状可以 broadcast 到 `[B, T, T]`。

### 复杂度分析

标准 full attention 需要计算 `[T, T]` attention matrix。

计算复杂度大致是：

$$
O(T^2d)
$$

显存复杂度主要来自 attention weights：

$$
O(T^2)
$$

这就是为什么长上下文很贵。

如果序列长度从 4K 增加到 8K，attention matrix 大小约变成 4 倍。

### Attention 变体总览

不同 attention 变体本质上都在回答一个问题：

```text
如何在质量、计算成本、显存、长上下文能力和推理效率之间做 trade-off？
```

下面先给总览，后续相关章节会展开。

### Full Attention

Full attention 指每个 token 都可以关注所有 token。

在非 causal 场景中：

```text
第 i 个 token 可以看 1..T 所有位置
```

优点：

1. 信息交互最充分。
2. 任意 token 可以直接建立联系。
3. 表达能力强。

缺点：

1. $O(T^2)$ 成本高。
2. 长上下文显存压力大。

适用场景：

1. BERT 类 encoder。
2. 非自回归编码任务。
3. 序列长度不太长的场景。

### Causal Attention

Causal attention 是 GPT 类模型使用的 attention。

第 `i` 个 token 只能关注：

$$
1,\ldots,i
$$

优点：

1. 符合自回归生成。
2. 防止看到未来 token。
3. 训练目标和推理一致。

缺点：

1. 仍然是 $O(T^2)$。
2. 生成时逐 token decode，推理延迟高。

适用场景：

1. GPT。
2. Causal LM。
3. 文本生成和代码生成。

### Local / Sliding Window Attention

Local attention 让每个 token 只关注附近窗口内的 token。

例如窗口大小是 `w`：

第 $i$ 个 token 只关注 $i-w$ 到 $i+w$。

优点：

1. 复杂度从 $O(T^2)$ 降到约 $O(Tw)$。
2. 适合很长序列。
3. 局部依赖建模高效。

缺点：

1. 远距离信息交互受限。
2. 需要额外机制传递全局信息。

适用场景：

1. 长文建模。
2. 局部相关性强的任务。
3. 部分长上下文 LLM。

### Sparse Attention

Sparse attention 只计算部分 token pair 的 attention。

常见稀疏模式：

1. local window。
2. global token。
3. block sparse。
4. strided pattern。
5. random pattern。

优点：

1. 降低计算和显存。
2. 可以支持更长上下文。
3. 设计灵活。

缺点：

1. 稀疏模式设计复杂。
2. 可能损失全局依赖。
3. 硬件效率不一定好。

适用场景：

1. 长上下文模型。
2. 文档建模。
3. 特定结构化序列任务。

### Linear Attention

Linear attention 试图把 attention 的复杂度从 $O(T^2)$ 降到 $O(T)$。

标准 attention 的瓶颈是：

$$
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

linear attention 通常通过 kernel trick 或特征映射，把计算顺序改写，避免显式构造 `[T, T]` attention matrix。

非常粗略地说，它尝试把：

$$
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

近似或改造成：

$$
\phi(Q)\left(\phi(K)^T V\right)
$$

这样可以先聚合 $K$ 和 $V$，再和 $Q$ 结合。

优点：

1. 理论复杂度可接近 $O(T)$。
2. 更适合长序列。
3. 某些形式适合流式处理。

缺点：

1. 通常是近似或改变 attention 形式。
2. 质量可能不如 full attention。
3. 工程实现和数值稳定性更复杂。
4. 在现代 GPU 上理论复杂度低不一定实际更快。

适用场景：

1. 超长序列研究。
2. 流式序列建模。
3. 对全局精确 attention 要求不高的任务。

### FlashAttention

FlashAttention 不是改变 attention 数学结果，而是改变计算方式。

它仍然计算精确 full attention，但通过 IO-aware 分块计算减少显存读写和中间矩阵存储。

优点：

1. 结果和标准 attention 等价或非常接近。
2. 显存更省。
3. 速度更快。
4. 已广泛用于现代 LLM 训练和推理。

缺点：

1. 依赖高效 kernel 实现。
2. 对硬件和 shape 有要求。
3. 不改变 $O(T^2)$ 的理论交互规模。

适用场景：

1. 大模型训练。
2. 长上下文训练。
3. 高性能推理。

### MQA 和 GQA

MQA 和 GQA 主要优化推理时 KV Cache。

MHA 中每个 query head 都有自己的 K/V head。

MQA 让多个 query head 共享同一组 K/V。

GQA 介于两者之间，让一组 query heads 共享一组 K/V。

优点：

1. 减少 KV Cache 显存。
2. 提升 decode 阶段吞吐。
3. 对大模型推理很重要。

缺点：

1. 可能影响模型质量。
2. 需要在训练时设计好结构。

适用场景：

1. 大模型推理。
2. 长上下文生成。
3. 对 KV Cache 显存敏感的服务。

第 10 讲会进一步讲 MHA、MQA、GQA。

### 变体对比表

| 类型 | 核心思想 | 复杂度直觉 | 优点 | 缺点 | 常见场景 |
| --- | --- | --- | --- | --- | --- |
| Full Attention | 所有 token 两两交互 | $O(T^2)$ | 表达力强 | 长序列贵 | BERT、encoder |
| Causal Attention | 只能看过去 | $O(T^2)$ | 适合生成 | decode 慢 | GPT |
| Local Attention | 只看局部窗口 | $O(Tw)$ | 长序列省 | 全局依赖弱 | 长文 |
| Sparse Attention | 只算部分连接 | 低于 $O(T^2)$ | 可扩展 | 模式复杂 | 长上下文 |
| Linear Attention | 避免显式 $T\times T$ 矩阵 | 近似 $O(T)$ | 理论高效 | 质量和稳定性挑战 | 超长序列研究 |
| FlashAttention | 优化精确 attention 计算 | 交互仍 $O(T^2)$ | 快、省显存 | 依赖 kernel | 现代 LLM |
| MQA/GQA | 共享 K/V heads | 降 KV cache | 推理省显存 | 可能损质量 | LLM serving |

### 真实项目中的坑

#### 坑 1：只看理论复杂度，不看硬件效率

Linear attention 理论上更低复杂度，但在 GPU 上不一定比优化良好的 FlashAttention 更快。

#### 坑 2：把 FlashAttention 误解成近似 attention

FlashAttention 的核心是 IO 优化，通常不是通过牺牲 attention 数学结果换速度。

#### 坑 3：长上下文只靠 attention 变体不够

长上下文还涉及位置编码、数据构造、训练长度、KV Cache、评估和检索策略。

#### 坑 4：稀疏模式破坏关键依赖

如果任务需要远距离精确信息交互，local 或 sparse attention 可能漏掉关键 token。

### 面试官会怎么问

#### 问题 1：写出 scaled dot-product attention 公式。

标准回答：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中 $QK^T$ 计算 query 和 key 的匹配分数，除以 $\sqrt{d_k}$ 稳定尺度，softmax 得到 attention weights，最后对 $V$ 加权求和。

#### 问题 2：为什么 attention 要除以 $\sqrt{d_k}$？

回答框架：

1. $d_k$ 大时点积分数方差变大。
2. softmax 容易饱和。
3. 梯度变小，训练不稳定。
4. 缩放可以稳定分数尺度。

#### 问题 3：Full attention 和 linear attention 区别是什么？

回答框架：

1. full attention 显式计算所有 token pair，表达力强但 $O(T^2)$。
2. linear attention 通过特征映射或改写避免显式 $T\times T$ 矩阵，理论上更适合长序列。
3. linear attention 可能带来近似误差、质量下降或工程复杂性。

#### 问题 4：FlashAttention 和 sparse attention 的区别是什么？

回答框架：

1. FlashAttention 通常计算精确 attention，只优化内存读写和 kernel。
2. Sparse attention 改变 attention 连接模式，只计算部分 token pair。
3. 前者主要是系统优化，后者是建模结构改变。

### 常见误区

1. 误区：linear attention 一定比 full attention 好。
   纠正：linear attention 有理论复杂度优势，但质量、稳定性和硬件效率不一定更好。

2. 误区：FlashAttention 改变了 attention 公式。
   纠正：它主要改变计算方式，不是改变数学定义。

3. 误区：所有 attention 都适合生成模型。
   纠正：生成模型通常需要 causal 约束，不能看未来。

4. 误区：局部 attention 一定能处理长文。
   纠正：它省成本，但可能牺牲远距离依赖。

### 小练习

1. 推导 $Q:[B,T,d]$ 和 $K:[B,T,d]$ 相乘后 scores 的 shape。
2. 用 PyTorch 实现 causal scaled dot-product attention。
3. 解释为什么 attention scores 需要 mask 后再 softmax。
4. 比较 full attention、local attention、linear attention 的优缺点。
5. 用自己的话解释 FlashAttention 为什么是系统优化而不是建模近似。

### 本讲总结

本讲最重要的结论：

1. 标准 attention 公式是 $\mathrm{softmax}(QK^T/\sqrt{d_k})V$。
2. $QK^T$ 计算 token 间匹配分数。
3. 除以 $\sqrt{d_k}$ 是为了稳定 softmax。
4. attention weights shape 是 $[B,T,T]$。
5. 标准 attention 的核心瓶颈是 $O(T^2)$。
6. Full、causal、local、sparse、linear、FlashAttention、MQA/GQA 都是在不同约束下做 trade-off。
7. 面试时要能区分“改变 attention 数学结构”和“优化 attention 计算实现”。

关键问题：Q、K、V 如何由输入得到，为什么使用 $QK^T$，为什么除以 $\sqrt{d_k}$。

## 第 10 讲：Multi-Head Attention

### 本讲目标

第 9 讲我们推导了单头 scaled dot-product attention。本讲继续讲现代 Transformer 中真正使用的 Multi-Head Attention。

你需要掌握：

1. 为什么需要多个 attention head。
2. Multi-Head Attention 的完整数据流。
3. head split、concat、output projection 的 shape。
4. MHA 的参数量如何估算。
5. MHA、MQA、GQA 的区别。
6. 为什么 MQA/GQA 对大模型推理和 KV Cache 很重要。
7. 多头注意力真实工程中的常见坑。

### 问题背景

单头 attention 可以让每个 token 关注其他 token，但它只有一套 Q/K/V 投影和一套注意力模式。

语言中的关系是多样的：

1. 语法关系。
2. 指代关系。
3. 局部搭配。
4. 长距离依赖。
5. 位置关系。
6. 任务格式关系。

单个 attention head 很难同时建模所有关系。

Multi-Head Attention 的核心想法是：让多个 head 在不同子空间里并行做 attention。

### 核心直觉

一句话：

Multi-Head Attention 让模型用多组不同的注意力视角观察同一段文本。

比如句子：

```text
The animal didn't cross the street because it was too tired.
```

某个 head 可能关注指代关系：

```text
it -> animal
```

另一个 head 可能关注局部短语：

```text
too -> tired
```

另一个 head 可能关注句法结构。

当然，真实模型中的 head 不一定像人类这样清晰分工，但多头结构提供了这种建模能力。

### Multi-Head Attention 数据流

假设输入：

$$
X:[B,T,d_{\text{model}}]
$$

有：

$$
n_{\mathrm{heads}}=h,\qquad d_{\mathrm{head}}=d_h,\qquad d_{\mathrm{model}}=h\cdot d_h
$$

通常先用线性层得到 Q、K、V：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

如果投影后仍是 `d_model` 维：

$$
Q,K,V:[B,T,d_{\text{model}}]
$$

然后 reshape 成多头：

$$
[B,T,d_{\text{model}}]\to[B,h,T,d_{\text{head}}]
$$

每个 head 独立做 attention：

$$
\text{head}_i=\mathrm{Attention}(Q_i,K_i,V_i)
$$

得到：

$$
\text{heads}:[B,h,T,d_{\text{head}}]
$$

再 concat：

$$
[B,h,T,d_{\text{head}}]\to[B,T,h\cdot d_{\text{head}}]=[B,T,d_{\text{model}}]
$$

最后经过 output projection：

$$
O=H_{\mathrm{concat}}W_O
$$

### 公式

Multi-Head Attention 可以写成：

$$
\text{head}_i=\mathrm{Attention}(XW_Q^i,XW_K^i,XW_V^i)
$$

$$
\mathrm{MHA}(X)=\mathrm{Concat}(\text{head}_1,\ldots,\text{head}_h)W_O
$$

其中每个 head 有自己的 `W_Q^i, W_K^i, W_V^i`。

### PyTorch 最小实现

下面是一个简化版 MHA，只用于理解 shape。

```python
import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: [B, T, d_model]
        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, T, d_model] -> [B, h, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # [B, h, T, T]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        out = weights @ v  # [B, h, T, head_dim]

        # [B, h, T, head_dim] -> [B, T, d_model]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


x = torch.randn(2, 4, 32)
mha = MultiHeadAttention(d_model=32, num_heads=4)
y = mha(x)
print(y.shape)  # [2, 4, 32]
```

### 参数量估算

如果忽略 bias，MHA 的主要参数是：

$$
W_Q:d_{\text{model}}\times d_{\text{model}}
$$

$$
W_K:d_{\text{model}}\times d_{\text{model}}
$$

$$
W_V:d_{\text{model}}\times d_{\text{model}}
$$

$$
W_O:d_{\text{model}}\times d_{\text{model}}
$$

总参数量约：

$$
4d_{\text{model}}^2
$$

注意：增加 head 数不一定增加总参数量。如果 $d_{\text{model}}$ 固定，`num_heads` 增加时 `head_dim` 通常变小，总投影矩阵仍然是 $[d_{\text{model}},d_{\text{model}}]$。

head 数影响的是注意力子空间划分方式，而不是简单线性增加参数量。

### head 数越多越好吗

不一定。

更多 head 的潜在好处：

1. 更多注意力子空间。
2. 更灵活的信息路由。
3. 可能捕捉更多关系模式。

但也有问题：

1. 如果 $d_{\text{model}}$ 固定，head 越多，`head_dim` 越小。
2. 单个 head 表达能力可能下降。
3. 过多 head 可能冗余。
4. 推理和 kernel 实现会受 shape 影响。

现代模型通常会在模型规模、硬件效率、训练稳定性之间选择合适 head 数。

### MHA、MQA、GQA 的区别

#### MHA：Multi-Head Attention

标准 MHA 中，每个 query head 都有自己的 K/V head。

如果有 32 个 query heads：

```text
Q heads: 32
K heads: 32
V heads: 32
```

优点：表达能力强。

缺点：推理时 KV Cache 大。

#### MQA：Multi-Query Attention

MQA 中，多个 query heads 共享同一组 K/V。

例如：

```text
Q heads: 32
K heads: 1
V heads: 1
```

优点：

1. KV Cache 显著减少。
2. decode 阶段更省显存。
3. 推理吞吐更好。

缺点：

1. K/V 表达能力下降。
2. 可能影响模型质量。

#### GQA：Grouped-Query Attention

GQA 是 MHA 和 MQA 的折中。

多个 query heads 分组共享 K/V。

例如：

```text
Q heads: 32
K/V heads: 8
```

每 4 个 query heads 共享一组 K/V。

优点：

1. 比 MHA 更省 KV Cache。
2. 比 MQA 保留更多 K/V 表达能力。
3. 现代 LLM 常用。

### 为什么 MQA/GQA 对推理重要

自回归生成时，每生成一个 token，都需要保存历史 token 的 K 和 V，这就是 KV Cache。

KV Cache 大小和以下因素相关：

1. 层数。
2. batch size。
3. sequence length。
4. K/V head 数。
5. head_dim。
6. 数据类型。

MHA 中 K/V head 多，KV Cache 大。

MQA/GQA 减少 K/V head 数，因此能显著降低 KV Cache 显存。

这对长上下文和高并发 serving 很关键。

### MHA/MQA/GQA 对比表

| 类型 | Q heads | K/V heads | 优点 | 缺点 | 常见场景 |
| --- | --- | --- | --- | --- | --- |
| MHA | 多个 | 和 Q 一样多 | 表达能力强 | KV Cache 大 | 标准 Transformer |
| MQA | 多个 | 1 组 | 推理省显存 | 可能损质量 | 极致推理优化 |
| GQA | 多个 | 少数组 | 质量和效率折中 | 设计更复杂 | 现代 LLM |

### 真实项目中的坑

#### 坑 1：reshape / transpose 写错

MHA 代码最常见 bug 是 shape 变换错。尤其是 `[B,T,h,d]` 和 `[B,h,T,d]` 混淆。

#### 坑 2：忘记 contiguous

transpose 后 tensor 可能不连续，直接 view 可能报错或行为异常。常见写法是 `.transpose(...).contiguous().view(...)`。

#### 坑 3：mask shape 不匹配

MHA scores shape 是 `[B,h,T,T]`，mask 需要能 broadcast 到这个形状。

#### 坑 4：只看参数量，不看 KV Cache

MHA、MQA、GQA 参数量差异不是唯一重点。推理服务中 KV Cache 显存更关键。

#### 坑 5：把 head 数增加理解为参数必然增加

在 `d_model` 固定时，head 数变化通常不改变 Q/K/V/O 总参数量，只改变每个 head 的维度。

### 优点、缺点和适用场景

MHA 优点：

1. 多个注意力子空间。
2. 表达能力强。
3. 适合复杂关系建模。

MHA 缺点：

1. KV Cache 大。
2. attention 计算仍是 $O(T^2)$。
3. shape 实现复杂。

MQA/GQA 优点：

1. 降低 KV Cache 显存。
2. 提升推理吞吐。
3. 更适合大规模 serving。

MQA/GQA 缺点：

1. 可能损失模型质量。
2. 需要在模型结构中提前设计。
3. 不同模型配置差异大。

### 面试官会怎么问

#### 问题 1：为什么需要 Multi-Head Attention？

标准回答：

```text
Multi-Head Attention 让模型在多个子空间中并行计算 attention。不同 head 可以关注不同类型的关系，比如局部依赖、长距离依赖、指代关系或句法结构。相比单头注意力，它提供了更灵活的信息路由能力。
```

#### 问题 2：MHA 的 shape 怎么变化？

回答框架：

1. 输入 $[B,T,d_{\text{model}}]$。
2. Q/K/V 投影后仍是 $[B,T,d_{\text{model}}]$。
3. reshape 成 $[B,h,T,d_{\text{head}}]$。
4. attention scores 是 $[B,h,T,T]$。
5. 输出 $[B,h,T,d_{\text{head}}]$。
6. concat 回 $[B,T,d_{\text{model}}]$。

#### 问题 3：MHA、MQA、GQA 有什么区别？

回答框架：

1. MHA：每个 query head 有自己的 K/V。
2. MQA：所有 query heads 共享一组 K/V。
3. GQA：query heads 分组共享 K/V。
4. MQA/GQA 主要是为了减少 KV Cache，提高推理效率。

#### 问题 4：增加 head 数会增加参数量吗？

标准回答：

```text
如果 $d_{\text{model}}$ 固定，增加 head 数通常不会显著增加 Q/K/V/O 的总参数量，因为 head_dim 会相应变小，总投影矩阵仍然是 $d_{\text{model}}$ 到 $d_{\text{model}}$。head 数主要改变的是注意力子空间划分，而不是简单增加参数量。
```

### 常见误区

1. 误区：每个 head 一定学到人类可解释的不同功能。
   纠正：多头提供不同子空间，但具体 head 功能不一定稳定可解释。

2. 误区：head 越多越好。
   纠正：head 多会降低单 head 维度，可能带来冗余和效率问题。

3. 误区：MQA/GQA 只是训练优化。
   纠正：它们对推理 KV Cache 和 serving 成本非常关键。

4. 误区：MHA 参数量随 head 数线性增加。
   纠正：在 $d_{\text{model}}$ 固定时通常不是这样。

### 小练习

1. 假设 $d_{\text{model}}=4096$，`num_heads=32`，计算 `head_dim`。
2. 推导 MHA 中 scores 的 shape。
3. 估算忽略 bias 时 MHA 的参数量。
4. 比较 MHA、MQA、GQA 对 KV Cache 的影响。
5. 修改本讲 PyTorch 代码，加入 causal mask。

### 本讲总结

本讲最重要的结论：

1. Multi-Head Attention 让模型在多个子空间并行做 attention。
2. 输入 $[B,T,d_{\text{model}}]$ 会被拆成 $[B,h,T,d_{\text{head}}]$。
3. attention scores shape 是 $[B,h,T,T]$。
4. 在 $d_{\text{model}}$ 固定时，head 数变化不一定改变总参数量。
5. MHA 表达能力强，但 KV Cache 大。
6. MQA 共享 K/V，最省 KV Cache，但可能损质量。
7. GQA 是 MHA 和 MQA 的折中，是现代 LLM 常见设计。

关键问题：为什么需要多个 head，multi-head attention 的参数量如何计算，MHA、MQA、GQA 有什么区别。

## 第 11 讲：Causal Mask 与自回归生成

### 本讲目标

前面我们已经理解了 attention 和 multi-head attention。本讲聚焦 GPT 类模型中非常关键的约束：causal mask。

你需要掌握：

1. 为什么自回归语言模型不能看到未来 token。
2. causal mask 是什么。
3. causal mask 如何作用在 attention scores 上。
4. 训练时为什么也要使用 causal mask。
5. teacher forcing 和 causal mask 的关系。
6. causal LM 和 masked LM 的区别。
7. 自回归生成的完整过程。

### 问题背景

GPT 类模型的训练目标是：

$$
P(x_t\mid x_{1:t-1})
$$

也就是第 $t$ 个 token 只能依赖前面的 token。

如果训练时第 $t$ 个位置可以看到未来 token，比如 $x_{t+1}$、$x_{t+2}$，那模型就可以作弊。

例如：

```text
I love machine learning
```

如果模型在预测 `machine` 时已经能看到后面的 `machine`，loss 会很低，但这不是有效学习。

推理时模型不可能看到未来 token，所以训练时也必须禁止看未来。

### Causal Mask 是什么

Causal mask 是一个下三角 mask。

对于长度为 4 的序列：

```text
位置: 1 2 3 4
```

第 1 个 token 只能看第 1 个：

```text
1 0 0 0
```

第 2 个 token 可以看第 1、2 个：

```text
1 1 0 0
```

第 3 个 token 可以看第 1、2、3 个：

```text
1 1 1 0
```

第 4 个 token 可以看第 1、2、3、4 个：

```text
1 1 1 1
```

合起来：

$$
\begin{bmatrix}
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1
\end{bmatrix}
$$

这就是 causal mask。

### Mask 如何作用在 Attention 上

attention scores shape 是：

$$
[B,h,T,T]
$$

其中 $\text{scores}[\ldots,i,j]$ 表示第 $i$ 个位置关注第 $j$ 个位置的分数。

如果 $j>i$，说明第 $i$ 个位置在看未来，需要 mask 掉。

通常做法是在 softmax 前把未来位置加上 `-inf`：

$$
S_{\mathrm{masked}}=S+M
$$

未来位置：

```text
-inf
```

softmax 后概率变成 0。

为什么要在 softmax 前 mask？

因为 softmax 会把分数归一化成概率。如果 softmax 后再 mask，概率和不再为 1，还需要重新归一化。

### PyTorch 代码

```python
import torch

T = 4
mask = torch.tril(torch.ones(T, T))
print(mask)
```

输出：

```text
tensor([[1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]])
```

加入 attention：

```python
import math
import torch


def causal_attention(q, k, v):
    # q, k, v: [B, h, T, d]
    B, h, T, d = q.shape
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)  # [B, h, T, T]

    mask = torch.tril(torch.ones(T, T, device=q.device)).view(1, 1, T, T)
    scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    return weights @ v
```

### 训练时为什么也要 Mask

训练时我们一次输入完整序列：

$$
[x_1,x_2,x_3,x_4]
$$

模型并行预测：

$$
x_2,x_3,x_4
$$

因为整个序列都在输入里，如果不加 causal mask，第 1 个位置可能看到第 2、3、4 个位置。

这会破坏 next-token prediction 的条件概率定义。

训练时使用 causal mask，可以同时做到：

1. 并行计算所有位置。
2. 保证每个位置不能看未来。

这也是 Transformer 训练高效的关键。

### Teacher Forcing 和 Causal Mask 的关系

训练时使用 teacher forcing：每个位置的历史上下文来自真实 token，而不是模型自己生成的 token。

但 teacher forcing 并不意味着可以看未来。

训练时：

第 $t$ 个位置可以看到真实的 $x_{1:t-1}$。

第 $t$ 个位置不能看到 $x_{t+1:T}$。

Causal mask 保证第二点。

所以：

```text
teacher forcing 解决训练输入稳定性
causal mask 解决不能看未来的问题
```

### 自回归生成过程

推理时没有完整答案，只能一步步生成。

给定 prompt：

```text
I love
```

第 1 步：

$$
P(\text{next}\mid \text{I love})
$$

假设生成：

```text
machine
```

上下文变成：

```text
I love machine
```

第 2 步：

$$
P(\text{next}\mid \text{I love machine})
$$

继续生成：

```text
learning
```

直到生成 EOS 或达到 max tokens。

### Causal LM vs Masked LM

Causal LM：

1. 从左到右建模。
2. 预测下一个 token。
3. 不能看未来。
4. 适合生成。
5. 典型模型：GPT。

Masked LM：

1. 随机 mask 输入中的某些 token。
2. 根据双向上下文预测被 mask 的 token。
3. 可以看左右两边上下文。
4. 适合理解和表示学习。
5. 典型模型：BERT。

例子：

```text
The capital of France is [MASK].
```

BERT 可以同时看 `[MASK]` 左右上下文。

GPT 生成时只能看左边上下文。

### 为什么 GPT 用 Causal LM

因为 GPT 的目标是生成。

生成天然是从左到右的过程：

```text
先生成第一个 token，再生成第二个 token，再生成第三个 token
```

Causal LM 的训练目标和推理过程一致。

BERT 的 masked LM 更适合理解任务，但不天然适合从左到右生成长文本。

### 真实项目中的坑

#### 坑 1：mask 方向写反

如果把下三角写成上三角，模型可能只能看未来或无法看历史。

#### 坑 2：mask shape broadcast 错误

attention scores 是 `[B,h,T,T]`，mask 需要能正确 broadcast。

#### 坑 3：padding mask 和 causal mask 混淆

causal mask 防止看未来，padding mask 防止看 padding token。两者解决的问题不同，但经常要合并使用。

#### 坑 4：训练时漏掉 causal mask

loss 可能异常低，但推理效果会很差，因为模型训练时作弊了。

#### 坑 5：KV Cache 时 position 和 mask 处理错误

增量推理时只输入新 token，但它要能关注历史 cache。position id 和 mask 处理错会导致生成质量异常。

### 优点、缺点和适用场景

Causal mask 的优点：

1. 保证自回归训练目标正确。
2. 防止信息泄漏。
3. 允许训练时并行计算所有位置。
4. 与生成过程一致。

缺点：

1. 只能使用单向上下文。
2. 不如双向模型适合某些理解任务。
3. 生成时容易错误累积。

适用场景：

1. GPT 类语言模型。
2. 文本生成。
3. 代码生成。
4. 对话模型。
5. 自回归多模态生成。

### 面试官会怎么问

#### 问题 1：什么是 causal mask？

标准回答：

```text
Causal mask 是自回归模型中的下三角注意力 mask，用来保证第 t 个位置只能关注自己和之前的位置，不能看到未来 token。它通常在 attention scores softmax 之前把未来位置置为 -inf，使 softmax 后这些位置概率为 0。
```

#### 问题 2：训练时为什么也要 causal mask？

回答框架：

1. 训练时完整序列同时输入。
2. 如果不 mask，模型能看到未来 token。
3. 这会导致信息泄漏和 loss 虚低。
4. causal mask 允许并行训练同时保持自回归约束。

#### 问题 3：Causal LM 和 Masked LM 区别是什么？

回答框架：

1. Causal LM 从左到右预测下一个 token。
2. Masked LM 用双向上下文预测被 mask 的 token。
3. GPT 是 causal LM，更适合生成。
4. BERT 是 masked LM，更适合理解表示。

#### 问题 4：teacher forcing 和 causal mask 是一回事吗？

标准回答：

```text
不是。Teacher forcing 指训练时使用真实历史 token 作为上下文，而 causal mask 指限制当前位置不能看到未来 token。二者经常同时用于 causal LM 训练，但解决的问题不同。
```

### 常见误区

1. 误区：训练时有完整序列，所以不需要 causal mask。
   纠正：正因为完整序列都在输入里，所以必须 mask 未来 token。

2. 误区：causal mask 和 padding mask 是一回事。
   纠正：causal mask 防未来，padding mask 防无效 padding。

3. 误区：BERT 和 GPT 只是模型大小不同。
   纠正：它们训练目标和 attention 可见性不同。

4. 误区：causal mask 只在推理时需要。
   纠正：训练时也必须使用，否则会信息泄漏。

### 小练习

1. 写出长度为 5 的 causal mask。
2. 用 PyTorch 生成 `[1,1,T,T]` 形状的 causal mask。
3. 解释为什么 mask 要在 softmax 前加入。
4. 比较 causal mask 和 padding mask。
5. 用自己的话解释 GPT 和 BERT 的训练目标区别。

### 本讲总结

本讲最重要的结论：

1. Causal mask 保证当前位置不能看到未来 token。
2. 它通常是下三角 mask。
3. mask 在 softmax 前作用于 attention scores。
4. 训练时也必须使用 causal mask，防止信息泄漏。
5. Teacher forcing 和 causal mask 不是一回事。
6. GPT 是 causal LM，BERT 是 masked LM。
7. causal mask 是 GPT 类自回归生成模型成立的关键机制。

关键问题：什么是 causal mask，为什么训练时也要 mask 未来 token，causal LM 和 masked LM 有什么区别。

## 第 12 讲：Transformer Block

### 本讲目标

前面我们分别讲了 tokenizer、embedding、attention、multi-head attention 和 causal mask。本讲把这些组件组合起来，形成一个完整的 Transformer block。

你需要掌握：

1. Transformer block 包含哪些模块。
2. Attention 和 MLP 分别起什么作用。
3. 残差连接为什么重要。
4. LayerNorm / RMSNorm 在哪里使用。
5. Pre-LN 和 Post-LN 有什么区别。
6. decoder-only Transformer block 的完整数据流。

### 问题背景

单独一个 attention 层还不是完整 Transformer。

现代 GPT 类模型通常由很多层相同结构的 block 堆叠而成：

```text
token ids
-> embedding
-> block 1
-> block 2
-> ...
-> block N
-> final norm
-> lm head
-> logits
```

每个 block 都会进一步加工 token 表示。

### Transformer Block 的核心组成

一个 decoder-only Transformer block 通常包含：

1. Normalization
2. Causal self-attention
3. Residual connection
4. Normalization
5. MLP / FFN
6. Residual connection

现代 LLM 多使用 Pre-LN 结构，简化写法：

$$
x=x+\mathrm{Attention}(\mathrm{Norm}(x))
$$

$$
x=x+\mathrm{MLP}(\mathrm{Norm}(x))
$$

这两行就是一个 Transformer block 的核心。

### 数据流直觉

输入：

$$
x:[B,T,d_{\text{model}}]
$$

第一步，self-attention 让每个 token 和上下文交互：

$$
a=\mathrm{Attention}(\mathrm{Norm}(x))
$$

$$
x=x+a
$$

第二步，MLP 对每个位置独立做非线性变换：

$$
m=\mathrm{MLP}(\mathrm{Norm}(x))
$$

$$
x=x+m
$$

输出仍然是：

$$
[B,T,d_{\text{model}}]
$$

所以多个 block 可以连续堆叠。

### Attention 的作用

Attention 负责 token 间信息交互。

它回答的问题是：

```text
当前位置应该从其他位置拿什么信息？
```

例如在一句话中：

1. 代词关注它指代的名词。
2. 动词关注主语和宾语。
3. 代码中的变量使用关注变量定义。
4. 数学推理中后续步骤关注前提。

Attention 是跨 token 的信息混合。

### MLP 的作用

MLP 也叫 FFN，Feed-Forward Network。

它通常对每个 token 位置独立作用，不直接混合不同 token。

典型结构：

$$
\mathrm{MLP}(x)=\mathrm{Linear}_2(\mathrm{Activation}(\mathrm{Linear}_1(x)))
$$

形状通常是：

$$
d_{\text{model}}\to d_{\text{ff}}\to d_{\text{model}}
$$

其中 `d_ff` 通常比 `d_model` 大，例如 4 倍左右，现代 LLM 也常用 SwiGLU 等门控结构。

MLP 的作用可以理解为：

1. 对每个 token 的表示做非线性变换。
2. 增加模型容量。
3. 存储和激活某些特征模式。
4. 在 attention 混合上下文后进一步加工信息。

Attention 负责“token 之间交流”，MLP 负责“每个 token 内部思考”。

### 残差连接为什么重要

残差连接形式：

$$
x=x+\mathrm{sublayer}(x)
$$

它的作用：

1. 保留原始信息。
2. 改善梯度传播。
3. 让深层网络更容易训练。
4. 让每层只需要学习对表示的增量修改。

如果没有残差连接，深层 Transformer 训练会困难很多。

直觉：

每一层不是完全重写表示，而是在已有表示上做修正。

### Normalization 的作用

Normalization 用来稳定激活分布和梯度传播。

Transformer 中常见：

1. LayerNorm
2. RMSNorm

它们通常作用在 hidden dimension 上。

现代 decoder-only LLM 多使用 Pre-LN 或 RMSNorm。

第 13 讲会详细讲 LayerNorm、RMSNorm 和残差连接，本讲先理解它在 block 中的位置。

### Pre-LN 和 Post-LN

Post-LN 是原始 Transformer 常见形式：

$$
x=\mathrm{Norm}(x+\mathrm{Attention}(x))
$$

$$
x=\mathrm{Norm}(x+\mathrm{MLP}(x))
$$

Pre-LN 是现代 LLM 常见形式：

$$
x=x+\mathrm{Attention}(\mathrm{Norm}(x))
$$

$$
x=x+\mathrm{MLP}(\mathrm{Norm}(x))
$$

Pre-LN 的优点：

1. 深层训练更稳定。
2. 梯度更容易沿残差路径传播。
3. 大模型中更常见。

Post-LN 的问题：

1. 深层模型训练更容易不稳定。
2. 需要更小心的初始化或训练技巧。

### PyTorch 最小实现

下面是一个简化版 decoder-only Transformer block。

```python
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = SimpleMLP(d_model, d_ff)

    def forward(self, x, attn_mask=None):
        # Pre-LN attention block.
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + attn_out

        # Pre-LN MLP block.
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x


x = torch.randn(2, 4, 32)
block = TransformerBlock(d_model=32, num_heads=4, d_ff=128)
y = block(x)
print(y.shape)  # [2, 4, 32]
```

注意：这个代码用于理解结构。真实 LLM 会使用更高效的 attention 实现、RoPE、RMSNorm、SwiGLU、KV Cache 等。

### Decoder-Only Block 和 Encoder Block 的区别

Decoder-only GPT block 使用 causal self-attention。

Encoder-only BERT block 使用 bidirectional self-attention。

Encoder-decoder Transformer 的 decoder block 通常还多一个 cross-attention，用来关注 encoder 输出。

GPT 类模型的 block 通常是：

```text
causal self-attention + MLP
```

BERT 类模型的 block 通常是：

```text
bidirectional self-attention + MLP
```

Seq2seq decoder block 通常是：

```text
causal self-attention + cross-attention + MLP
```

### 真实项目中的坑

#### 坑 1：Pre-LN 和 Post-LN 混淆

读论文或代码时要确认 norm 在 residual 之前还是之后，这会影响训练稳定性。

#### 坑 2：残差维度不一致

残差相加要求 shape 一致，所以 attention 和 MLP 输出必须回到 $d_{\text{model}}$。

#### 坑 3：MLP 扩展维度配置错误

`d_ff` 过小会限制容量，过大会增加参数和计算成本。

#### 坑 4：忽略 dropout、bias、activation 差异

不同模型实现细节不同，例如 GeLU、SwiGLU、bias-free Linear、RMSNorm 等。

#### 坑 5：用 PyTorch 内置 MultiheadAttention 直接类比生产 LLM

内置模块适合学习，但生产 LLM 通常有定制 attention、RoPE、FlashAttention 和 KV Cache。

### 优点、缺点和适用场景

Transformer block 的优点：

1. attention 和 MLP 分工清晰。
2. 残差连接支持深层堆叠。
3. normalization 提升训练稳定性。
4. 结构统一，容易 scale。
5. 适合语言、多模态、代码等任务。

缺点：

1. attention 成本随序列长度平方增长。
2. MLP 参数和计算量大。
3. 深层训练需要很多稳定性技巧。
4. 推理时 KV Cache 和显存管理复杂。

适用场景：

1. GPT 类语言模型。
2. BERT 类理解模型。
3. 编码器-解码器模型。
4. 多模态大模型。
5. 代码模型和 reasoning model。

### 面试官会怎么问

#### 问题 1：一个 Transformer block 包含什么？

标准回答：

```text
以 decoder-only Transformer 为例，一个 block 通常包含 normalization、causal self-attention、residual connection、normalization、MLP 和 residual connection。现代 LLM 常用 Pre-LN 结构，即 $x=x+\mathrm{Attention}(\mathrm{Norm}(x))$，然后 $x=x+\mathrm{MLP}(\mathrm{Norm}(x))$。
```

#### 问题 2：Attention 和 MLP 分别起什么作用？

回答框架：

1. Attention 负责 token 间信息交互。
2. MLP 对每个 token 独立做非线性特征变换。
3. Attention 更像通信，MLP 更像局部计算和特征加工。

#### 问题 3：Pre-LN 和 Post-LN 有什么区别？

回答框架：

1. Post-LN 是 $\mathrm{Norm}(x+\mathrm{sublayer}(x))$。
2. Pre-LN 是 $x+\mathrm{sublayer}(\mathrm{Norm}(x))$。
3. Pre-LN 更利于深层模型训练稳定，因此现代 LLM 常用。

#### 问题 4：为什么残差连接重要？

回答框架：

1. 保留原始信息。
2. 改善梯度传播。
3. 让每层学习增量修改。
4. 支持深层网络训练。

### 常见误区

1. 误区：Transformer block 只有 attention。
   纠正：MLP、norm、residual 同样重要。

2. 误区：MLP 会混合不同 token 信息。
   纠正：标准 MLP 通常对每个位置独立作用，不直接跨 token 混合。

3. 误区：Pre-LN 和 Post-LN 只是写法不同。
   纠正：它们对训练稳定性和梯度传播有重要影响。

4. 误区：每一层都完全重写表示。
   纠正：残差结构让每层更像在原表示上做增量更新。

### 小练习

1. 画出 decoder-only Transformer block 的数据流。
2. 写出 Pre-LN block 的两行核心公式。
3. 解释 attention 和 MLP 的分工。
4. 比较 GPT block、BERT block、seq2seq decoder block。
5. 修改本讲代码，把 GeLU 换成 SiLU，并观察输出 shape 是否变化。

### 本讲总结

本讲最重要的结论：

1. Transformer block 由 attention、MLP、normalization 和 residual connection 组成。
2. Attention 负责 token 间信息交互。
3. MLP 负责每个 token 内部的非线性特征加工。
4. Residual connection 支持深层训练和信息保留。
5. Modern LLM 通常使用 Pre-LN 结构。
6. 多个 block 堆叠形成完整 Transformer。
7. 真实 LLM block 会加入 RMSNorm、SwiGLU、RoPE、FlashAttention、KV Cache 等工程优化。

关键问题：Attention 和 MLP 分别起什么作用，残差连接为什么重要，Pre-LN 和 Post-LN 有什么区别。

## 第 13 讲：LayerNorm、RMSNorm 与残差连接

### 本讲目标

第 12 讲我们看到 Transformer block 中有 normalization 和 residual connection。本讲专门展开它们：为什么深层 Transformer 离不开归一化和残差连接？为什么现代 LLM 常用 RMSNorm 和 Pre-LN？

你需要掌握：

1. LayerNorm 和 BatchNorm 的区别。
2. 为什么 Transformer 常用 LayerNorm 而不是 BatchNorm。
3. RMSNorm 相比 LayerNorm 简化了什么。
4. 残差连接如何改善梯度传播。
5. Pre-LN 为什么对深层 LLM 更稳定。
6. 归一化和残差连接的真实工程坑。

### 问题背景

深层神经网络训练的一个核心问题是：每一层的输入分布会不断变化，梯度在很多层之间传播也容易不稳定。

Transformer 通常堆叠几十层甚至上百层。如果没有归一化和残差连接，训练会非常困难。

两个关键工具：

1. Normalization：稳定每层输入的数值尺度。
2. Residual connection：给信息和梯度提供更直接的路径。

### BatchNorm 回顾

BatchNorm 常用于 CNN。

它通常在 batch 维度上统计均值和方差。

简化理解：

```text
对一个 feature，在一个 batch 内计算 mean 和 variance
```

然后归一化：

$$
x_{\text{norm}}=\frac{x-\text{mean}}{\sqrt{\text{var}+\varepsilon}}
$$

BatchNorm 的问题：

1. 依赖 batch 统计。
2. 小 batch 时统计不稳定。
3. 自回归生成时 batch 和 sequence 形态变化复杂。
4. 训练和推理统计不一致。

因此 Transformer 语言模型通常不使用 BatchNorm。

### LayerNorm 是什么

LayerNorm 在每个样本内部，对 hidden dimension 做归一化。

对于 hidden vector：

$$
x:[d_{\text{model}}]
$$

计算：

$$
\mu=\frac{1}{d_{\text{model}}}\sum_{i=1}^{d_{\text{model}}}x_i
$$

$$
\sigma^2=\frac{1}{d_{\text{model}}}\sum_{i=1}^{d_{\text{model}}}(x_i-\mu)^2
$$

$$
x_{\text{norm},i}=\frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}}
$$

然后加上可学习缩放和平移：

$$
y_i=\gamma_i x_{\text{norm},i}+\beta_i
$$

在 Transformer 中，如果输入是：

$$
[B,T,d_{\text{model}}]
$$

LayerNorm 通常对最后一维 `d_model` 做归一化。

### 为什么 Transformer 用 LayerNorm

LayerNorm 的优点：

1. 不依赖 batch size。
2. 训练和推理行为一致。
3. 适合变长序列。
4. 对自回归生成更友好。
5. 能稳定每个 token 的 hidden state 尺度。

这使它比 BatchNorm 更适合 NLP 和 Transformer。

### RMSNorm 是什么

RMSNorm 是 LayerNorm 的简化版本。

LayerNorm 做两件事：

1. 减均值。
2. 除以标准差。

RMSNorm 不减均值，只用 root mean square 归一化：

$$
\mathrm{rms}=\sqrt{\mathrm{mean}(x^2)+\varepsilon}
$$

$$
y=\frac{\gamma x}{\mathrm{rms}}
$$

RMSNorm 省略了 mean centering。

优点：

1. 计算更简单。
2. 参数更少或实现更轻。
3. 在很多 LLM 中效果很好。
4. 常用于 LLaMA 系列等现代架构。

### LayerNorm 和 RMSNorm 对比

| 方法 | 是否减均值 | 是否除尺度 | 是否有 beta | 常见场景 |
| --- | --- | --- | --- | --- |
| LayerNorm | 是 | 是 | 通常有 | Transformer、BERT、GPT 早期模型 |
| RMSNorm | 否 | 是 | 通常无 beta | 现代 LLM，如 LLaMA 风格模型 |

核心区别：

```text
LayerNorm 标准化均值和方差
RMSNorm 只标准化向量尺度
```

### 残差连接再理解

残差连接：

$$
x=x+F(x)
$$

它让每层学习的是增量 `F(x)`，而不是从零生成新表示。

为什么重要？

1. 如果某层暂时没学好，至少可以保留原始输入。
2. 梯度可以沿残差路径更顺畅传播。
3. 深层模型更容易训练。
4. 多层逐步 refinement，而不是每层完全重写。

### Pre-LN 为什么稳定

Pre-LN：

$$
x=x+\mathrm{Attention}(\mathrm{Norm}(x))
$$

$$
x=x+\mathrm{MLP}(\mathrm{Norm}(x))
$$

因为 Norm 在子层前，子层收到的输入尺度更稳定。

同时残差路径上有一条相对直接的梯度通路。

这让很深的 Transformer 更容易训练。

Post-LN：

$$
x=\mathrm{Norm}(x+\mathrm{Attention}(x))
$$

在深层模型中更容易出现梯度传播不稳定。

### PyTorch 实现 LayerNorm 和 RMSNorm

LayerNorm：

```python
import torch
import torch.nn as nn

x = torch.randn(2, 4, 8)  # [B, T, d_model]
ln = nn.LayerNorm(8)
y = ln(x)
print(y.shape)  # [2, 4, 8]
```

手写 RMSNorm：

```python
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [..., dim]
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


x = torch.randn(2, 4, 8)
norm = RMSNorm(8)
y = norm(x)
print(y.shape)  # [2, 4, 8]
```

### 在 Transformer Block 中的位置

现代 Pre-LN block：

```python
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

如果使用 RMSNorm：

```python
x = x + attention(rmsnorm1(x))
x = x + mlp(rmsnorm2(x))
```

结构不变，只是 norm 类型不同。

### 真实项目中的坑

#### 坑 1：BatchNorm 和 LayerNorm 混淆

BatchNorm 依赖 batch 统计，LayerNorm 对单个样本 hidden 维度归一化。Transformer 通常用 LayerNorm/RMSNorm。

#### 坑 2：归一化维度写错

LayerNorm 应该对 hidden dimension 做归一化，而不是对 sequence dimension。

#### 坑 3：eps 太小导致数值不稳定

归一化里 `eps` 用于防止除零。混合精度训练时 eps 设置也会影响稳定性。

#### 坑 4：weight decay 作用到 norm 参数

LayerNorm/RMSNorm 参数通常不做 weight decay，否则可能影响训练稳定性。

#### 坑 5：Pre-LN/Post-LN 改动后直接加载旧 checkpoint

结构变了，checkpoint 行为不能直接等价。

### 优点、缺点和适用场景

LayerNorm 优点：

1. 不依赖 batch size。
2. 训练和推理一致。
3. 适合序列模型。
4. 稳定 hidden state 尺度。

LayerNorm 缺点：

1. 计算比 RMSNorm 略复杂。
2. 对具体实现和数值精度敏感。

RMSNorm 优点：

1. 更简单。
2. 计算更轻。
3. 现代 LLM 中表现好。

RMSNorm 缺点：

1. 不减均值，行为和 LayerNorm 不完全相同。
2. 替换时需要重新训练或充分验证。

残差连接优点：

1. 改善梯度传播。
2. 支持深层网络。
3. 保留输入信息。

### 面试官会怎么问

#### 问题 1：LayerNorm 和 BatchNorm 有什么区别？

标准回答：

```text
BatchNorm 通常在 batch 维度上统计均值和方差，依赖 batch 统计；LayerNorm 在单个样本的 hidden dimension 上归一化，不依赖 batch size。Transformer 和语言模型常用 LayerNorm，因为它适合变长序列、自回归生成，并且训练和推理行为一致。
```

#### 问题 2：RMSNorm 和 LayerNorm 有什么区别？

回答框架：

1. LayerNorm 减均值并除以标准差。
2. RMSNorm 不减均值，只除以 RMS。
3. RMSNorm 更简单，现代 LLM 常用。
4. 二者不完全等价，替换需要验证。

#### 问题 3：为什么残差连接重要？

回答框架：

1. 保留输入信息。
2. 让每层学习增量。
3. 改善梯度传播。
4. 支持深层 Transformer 训练。

#### 问题 4：为什么现代 LLM 常用 Pre-LN？

回答框架：

1. 子层输入被归一化，尺度更稳定。
2. 残差路径提供更直接梯度通路。
3. 深层训练比 Post-LN 更稳定。

### 常见误区

1. 误区：LayerNorm 和 BatchNorm 只是名字不同。
   纠正：它们统计维度和训练/推理行为不同。

2. 误区：RMSNorm 是 LayerNorm 的完全替代。
   纠正：RMSNorm 简化了均值中心化，行为不完全相同。

3. 误区：残差连接只是为了让 shape 对齐。
   纠正：它的核心价值是信息保留和梯度传播。

4. 误区：Pre-LN 和 Post-LN 只影响代码顺序。
   纠正：它们对深层模型训练稳定性影响很大。

### 小练习

1. 对一个长度为 4 的向量手算 LayerNorm。
2. 对同一个向量手算 RMSNorm。
3. 比较 LayerNorm 和 BatchNorm 的统计维度。
4. 解释为什么 norm 参数通常不做 weight decay。
5. 写出 Pre-LN Transformer block 的公式。

### 本讲总结

本讲最重要的结论：

1. Transformer 常用 LayerNorm/RMSNorm，而不是 BatchNorm。
2. LayerNorm 对 hidden dimension 做归一化，不依赖 batch 统计。
3. RMSNorm 不减均值，只归一化向量尺度。
4. 残差连接改善信息和梯度传播。
5. Pre-LN 对深层 LLM 训练更稳定。
6. 归一化维度、eps、weight decay、checkpoint 兼容性都是工程高频坑。

关键问题：LayerNorm 和 BatchNorm 有什么区别，为什么 Transformer 常用 LayerNorm 或 RMSNorm。

## 第 14 讲：RoPE 与长上下文

### 本讲目标

前面我们知道 Transformer 需要位置编码，否则 attention 不知道 token 顺序。本讲重点讲现代 LLM 常见的位置编码 RoPE，以及长上下文为什么困难。

你需要掌握：

1. 为什么绝对位置编码外推困难。
2. RoPE 的核心直觉是什么。
3. RoPE 如何把位置信息注入 Q 和 K。
4. RoPE 为什么天然包含相对位置性质。
5. 长上下文训练和推理分别难在哪里。
6. 常见长上下文扩展方法和风险。

### 问题背景

语言模型不仅要知道 token 内容，还要知道 token 的位置。

例如：

```text
Alice gave Bob a book.
Bob gave Alice a book.
```

两句话 token 类似，但位置不同，语义不同。

早期 Transformer 使用绝对位置编码，但现代 LLM 很多使用 RoPE。

RoPE 全称 Rotary Position Embedding，即旋转位置编码。

### 绝对位置编码的问题

绝对位置编码通常给每个位置一个向量：

```text
position 0 -> p_0
position 1 -> p_1
...
```

然后：

$$
x=E_{\mathrm{token}}+E_{\mathrm{pos}}
$$

问题：

1. 训练时只见过某个最大长度内的位置。
2. 推理时超过训练长度，未见过的位置可能表现差。
3. 绝对编号不直接表达 token 间相对距离。
4. 长上下文外推能力有限。

例如训练最大长度是 4K，推理直接拉到 32K，模型不一定知道位置 20000 应该如何处理。

### RoPE 的核心直觉

RoPE 的核心思想：

不要简单把位置向量加到 token embedding 上，而是在 attention 中对 Q 和 K 按位置做旋转。

直觉上：

1. 每个位置对应一个旋转角度。
2. token 在不同位置时，Q/K 会被旋转到不同方向。
3. 两个 token 的 attention score 会自然携带它们的相对位置关系。

也就是说，RoPE 把位置编码放进了 attention 匹配过程。

### 二维旋转直觉

先看二维向量。

一个向量 `(x_1, x_2)` 旋转角度 `θ` 后：

$$
x_1'=x_1\cos\theta-x_2\sin\theta
$$

$$
x_2'=x_1\sin\theta+x_2\cos\theta
$$

RoPE 会把 hidden dimension 两两分组，对每组做类似旋转。

不同位置使用不同角度。

位置越靠后，旋转角度越大。

### RoPE 如何作用到 Q 和 K

标准 attention：

$$
\mathrm{score}(i,j)=q_i\cdot k_j
$$

RoPE 中，先对 `q_i` 和 `k_j` 注入位置旋转：

$$
q_i'=\mathrm{rotate}(q_i,\text{position}=i)
$$

$$
k_j'=\mathrm{rotate}(k_j,\text{position}=j)
$$

然后计算：

$$
\mathrm{score}(i,j)=q_i'\cdot k_j'
$$

关键性质是：这个点积会依赖 $i-j$，也就是相对位置。

所以 RoPE 不是显式加一个相对位置 bias，而是通过旋转让 attention score 自然感知相对距离。

### 为什么 RoPE 适合相对位置

在二维旋转中，两个旋转向量的内积与它们旋转角度差有关。

如果 $q$ 在位置 $i$ 旋转，$k$ 在位置 $j$ 旋转，那么它们的匹配关系会受到 $i-j$ 影响。

这让模型更容易学习类似：

```text
关注前一个 token
关注距离我 10 个 token 的 token
关注句子开头附近的 token
```

这类相对位置模式。

### 最小 PyTorch 代码直觉

下面是简化版 RoPE，对最后一维两两旋转。真实实现会更注意缓存、频率、精度和 shape。

```python
import torch


def apply_rope(x, cos, sin):
    # x: [B, T, d], d must be even
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    out = torch.empty_like(x)
    out[..., 0::2] = rotated_x1
    out[..., 1::2] = rotated_x2
    return out


B, T, d = 2, 4, 8
x = torch.randn(B, T, d)

# 简化构造 cos/sin: [T, d/2]
positions = torch.arange(T).float().unsqueeze(1)
freqs = torch.arange(d // 2).float().unsqueeze(0)
theta = positions / (10000 ** (2 * freqs / d))
cos = torch.cos(theta).unsqueeze(0)  # [1, T, d/2]
sin = torch.sin(theta).unsqueeze(0)  # [1, T, d/2]

y = apply_rope(x, cos, sin)
print(y.shape)  # [2, 4, 8]
```

### RoPE 和绝对位置编码的区别

绝对位置编码：

$$
x=E_{\mathrm{token}}+E_{\mathrm{pos}}
$$

RoPE：

$$
q=\mathrm{rotate}(q,\text{position})
$$

$$
k=\mathrm{rotate}(k,\text{position})
$$

核心区别：

1. 绝对位置编码直接加到 hidden states。
2. RoPE 作用在 attention 的 Q/K 上。
3. RoPE 更自然地影响 token 间匹配分数。
4. RoPE 具有相对位置性质。

### 长上下文为什么困难

长上下文不是只把 `max_position_embeddings` 改大。

主要困难包括：

#### 困难 1：Attention 成本

标准 attention 是 $O(T^2)$。

上下文长度从 8K 到 32K，attention matrix 大小约变成 16 倍。

#### 困难 2：KV Cache 显存

自回归推理要缓存历史 K/V。

上下文越长，KV Cache 越大。

长上下文 serving 常常被 KV Cache 显存限制。

#### 困难 3：位置外推

模型训练时没见过特别长的位置，推理时直接外推可能退化。

RoPE 虽然有相对位置性质，但不代表可以无限外推。

#### 困难 4：长程依赖学习

即使上下文能放进去，模型也不一定会有效利用远处信息。

常见问题：

1. lost in the middle。
2. 只关注开头和结尾。
3. 无法稳定检索远处细节。
4. 多跳信息整合困难。

#### 困难 5：训练数据

如果训练数据大多是短序列，模型不会自然学会长上下文行为。

长上下文需要专门数据和训练策略。

### 常见长上下文方法

#### 方法 1：RoPE Scaling

通过调整 RoPE 的频率或位置映射，让模型适配更长上下文。

优点：

1. 工程上相对简单。
2. 可以扩展已有模型上下文。

缺点：

1. 可能影响短上下文能力。
2. 外推质量需要评估。
3. 不解决 attention 和 KV Cache 成本。

#### 方法 2：长上下文继续训练

用更长序列数据继续训练模型。

优点：

1. 模型真正见过长序列。
2. 长上下文能力更可靠。

缺点：

1. 训练成本高。
2. 数据构造困难。
3. 可能影响原有能力。

#### 方法 3：高效 Attention

例如 FlashAttention、sparse attention、local attention 等。

优点：降低显存或计算压力。

缺点：不同方法有不同质量和工程 trade-off。

#### 方法 4：RAG

不把所有信息塞进上下文，而是先检索相关片段。

优点：

1. 成本更可控。
2. 可处理超大知识库。
3. 知识更新方便。

缺点：

1. 依赖检索质量。
2. 检索不到就答不好。
3. 上下文构造和引用评估复杂。

### 长上下文评估

长上下文不能只看模型支持多少 token。

需要评估：

1. 能否找到远处信息。
2. 能否整合多个位置的信息。
3. 中间位置是否容易丢失。
4. 长文摘要是否忠实。
5. 长代码仓库理解是否准确。
6. 推理延迟和显存成本是否可接受。

常见 needle-in-a-haystack 测试有参考价值，但不足以代表真实长上下文能力。

### 真实项目中的坑

#### 坑 1：直接改最大长度

只改 config 里的 max length，不代表模型真的会长上下文。

#### 坑 2：只测 needle，不测真实任务

模型能找到一根针，不代表能做长文推理、代码理解或多文档分析。

#### 坑 3：忽略 KV Cache 成本

长上下文推理时，显存可能主要被 KV Cache 占用。

#### 坑 4：RoPE scaling 后不做短上下文回归测试

扩长上下文可能影响原有短上下文任务。

#### 坑 5：训练长度和推理长度差距过大

外推过远时，位置编码和 attention 行为都可能不稳定。

### 优点、缺点和适用场景

RoPE 优点：

1. 把位置注入 Q/K，适合 attention。
2. 具有相对位置性质。
3. 现代 LLM 实践广泛。
4. 比简单绝对位置编码更适合外推。

RoPE 缺点：

1. 不是无限长上下文解决方案。
2. scaling 需要小心评估。
3. 实现中频率、位置 id、精度容易出错。

长上下文适用场景：

1. 长文问答。
2. 多文档分析。
3. 代码仓库理解。
4. 法律、金融、医学文档处理。
5. Agent 长任务记忆。

### 面试官会怎么问

#### 问题 1：RoPE 的核心思想是什么？

标准回答：

```text
RoPE 是旋转位置编码。它不是把位置向量加到 hidden state 上，而是根据位置对 attention 中的 Q 和 K 做旋转。这样 Q 和 K 的点积会自然包含相对位置信息，因此更适合自注意力中的位置建模。
```

#### 问题 2：RoPE 为什么有相对位置性质？

回答框架：

1. Q 和 K 分别按各自位置旋转。
2. 旋转后点积与旋转角度差有关。
3. 角度差对应位置差 `i-j`。
4. 所以 attention score 能感知相对位置。

#### 问题 3：长上下文为什么困难？

回答框架：

1. attention `O(T^2)` 成本。
2. KV Cache 显存随长度增长。
3. 位置外推不稳定。
4. 模型不一定利用远处信息。
5. 长上下文训练数据和评估都困难。

#### 问题 4：如何评估长上下文模型？

回答框架：

1. needle-in-a-haystack 只是基础测试。
2. 还要测多文档问答、长文摘要、代码理解、多跳推理。
3. 要看不同位置、不同长度、不同任务。
4. 同时评估延迟、显存和成本。

### 常见误区

1. 误区：RoPE 可以无限外推。
   纠正：RoPE 有相对位置性质，但超出训练长度太多仍可能退化。

2. 误区：长上下文等于把 max length 改大。
   纠正：还需要位置编码、训练数据、attention 成本、KV Cache 和评估配套。

3. 误区：needle 测试好就代表长上下文好。
   纠正：真实任务还需要信息整合、推理和抗干扰能力。

4. 误区：RAG 和长上下文互相替代。
   纠正：它们是互补方案。长上下文处理上下文内信息，RAG 处理外部知识检索。

### 小练习

1. 用自己的话解释 RoPE 为什么作用在 Q/K 上。
2. 写一个简化函数，对二维向量做旋转。
3. 解释为什么标准 attention 的长上下文成本高。
4. 比较 RoPE scaling 和长上下文继续训练的优缺点。
5. 设计一个长上下文评估集，至少包含 3 类任务。

### 本讲总结

本讲最重要的结论：

1. RoPE 是旋转位置编码，常用于现代 LLM。
2. RoPE 通过旋转 Q/K 注入位置信息。
3. RoPE 的点积结构天然包含相对位置性质。
4. 长上下文困难不只是位置编码问题。
5. attention 成本、KV Cache、位置外推、训练数据和评估都很关键。
6. RoPE scaling、继续训练、高效 attention、RAG 都是常见方向。
7. 长上下文能力必须用真实任务和成本指标共同评估。

关键问题：RoPE 如何把位置信息注入 Q 和 K，长上下文会带来哪些训练和推理问题。

## 第 15 讲：从零实现一个小 GPT

### 本讲目标

前面我们已经学习了 GPT 的关键组件：tokenization、embedding、位置编码、self-attention、multi-head attention、causal mask、Transformer block、loss 和优化器。

本讲把它们串起来，实现一个最小可训练的 decoder-only GPT。

你需要掌握：

1. GPT 模型的核心模块如何组织。
2. 输入和 label 如何构造。
3. causal attention 如何实现。
4. Transformer block 如何堆叠。
5. logits 和 loss 如何计算。
6. generate 函数如何逐 token 生成。
7. 最小训练 loop 包含哪些步骤。

### 最小 GPT 的整体结构

一个简化版 GPT 包含：

```text
token embedding
position embedding
N 个 Transformer block
final LayerNorm
lm head
```

数据流：

```text
input_ids [B,T]
-> token embedding [B,T,d]
-> position embedding [B,T,d]
-> transformer blocks [B,T,d]
-> final norm [B,T,d]
-> lm head [B,T,V]
```

其中：

```text
B = batch size
T = sequence length
d = hidden size
V = vocab size
```

### 为什么从字符级模型开始

为了让代码最小化，我们先用字符级 tokenizer。

也就是把每个字符当作 token。

优点：

1. 不依赖外部 tokenizer。
2. 代码简单。
3. 适合理解完整训练流程。

缺点：

1. 序列更长。
2. 不适合真实大模型。
3. 语义粒度较细。

本讲目标是理解 GPT 结构，不追求训练出强模型。

### 数据构造

给定一段文本：

```text
hello world
```

字符级 vocab：

```text
[' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w']
```

每个字符映射成 id。

训练时取长度为 `block_size` 的片段：

$$
\text{input}:x_1,x_2,\ldots,x_T
$$

$$
\text{label}:x_2,x_3,\ldots,x_{T+1}
$$

这就是 next-token prediction。

### 完整最小代码

下面代码是一个教学版 miniGPT。它可以独立理解，但真实训练需要更多工程处理。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)  # [B, T, 3C]
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)

        out = weights @ v  # [B, h, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=2, n_head=4, n_embd=128):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size

        pos = torch.arange(0, T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
```

### 训练 loop 示例

```python
text = "hello world\n" * 1000
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

block_size = 16
batch_size = 32


def get_batch():
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


model = MiniGPT(vocab_size=len(chars), block_size=block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(200):
    x, y = get_batch()
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(step, loss.item())

start = torch.tensor([[stoi["h"]]], dtype=torch.long)
out = model.generate(start, max_new_tokens=50)[0].tolist()
print("".join(itos[i] for i in out))
```

### 代码结构解释

#### CausalSelfAttention

负责：

1. 生成 Q/K/V。
2. 拆成多头。
3. 计算 causal attention。
4. 合并多头输出。
5. 做 output projection。

#### Block

负责：

1. Pre-LN attention。
2. 残差连接。
3. Pre-LN MLP。
4. 残差连接。

#### MiniGPT

负责：

1. token embedding。
2. position embedding。
3. 堆叠多个 block。
4. final norm。
5. lm head 输出 logits。
6. 计算 cross entropy loss。
7. generate 自回归生成。

### 训练和生成的关系

训练时：

```text
输入 x，目标 y = x 右移一位
一次 forward 预测所有位置
用 cross entropy 训练
```

生成时：

```text
输入已有上下文
取最后一个位置 logits
采样下一个 token
拼回上下文
重复
```

### 真实项目中的坑

#### 坑 1：input 和 target 没有错位

如果 `x` 和 `y` 一样，模型会学复制当前 token。

#### 坑 2：causal mask 没有裁剪到当前 T

训练或生成时输入长度可能小于 block_size，mask 要使用 `:T, :T`。

#### 坑 3：view 前忘记 contiguous

transpose 后直接 view 可能出错。

#### 坑 4：generate 时忘记截断上下文

如果输入长度超过 block_size，position embedding 会越界。

#### 坑 5：把教学版 miniGPT 当成生产实现

真实 LLM 还需要 RoPE、RMSNorm、SwiGLU、FlashAttention、KV Cache、混合精度、分布式训练等。

### 面试官会怎么问

#### 问题 1：从零实现 GPT 的核心模块有哪些？

回答框架：

1. token embedding。
2. position embedding 或 RoPE。
3. causal self-attention。
4. MLP。
5. LayerNorm/RMSNorm。
6. residual connection。
7. lm head。
8. cross entropy loss。

#### 问题 2：input 和 label 如何构造？

标准回答：

```text
对于 causal LM，输入是一段 token 序列，label 是同一序列右移一位。比如 input 是 $x_1$ 到 $x_T$，target 是 $x_2$ 到 $x_{T+1}$。模型在每个位置预测下一个 token。
```

#### 问题 3：generate 函数做了什么？

回答框架：

1. 截取最近 block_size 个 token。
2. forward 得到 logits。
3. 取最后位置 logits。
4. softmax 后采样或取最大值。
5. 拼接新 token。
6. 重复直到结束。

#### 问题 4：教学 miniGPT 和真实 LLM 差距在哪里？

回答框架：

1. tokenizer 更复杂。
2. 位置编码通常用 RoPE。
3. norm 常用 RMSNorm。
4. MLP 常用 SwiGLU。
5. attention 用 FlashAttention、GQA、KV Cache。
6. 训练需要混合精度和分布式。
7. 数据、评估和对齐流程复杂得多。

### 常见误区

1. 误区：GPT 只是 attention 堆叠。
   纠正：还包括 embedding、position、MLP、norm、residual、loss 和 generation。

2. 误区：训练和生成用完全相同输入。
   纠正：训练有 target，生成没有 target，需要逐步采样。

3. 误区：小 GPT 训练出来乱码就是代码错。
   纠正：也可能是数据太少、训练步数太少、模型太小或字符级任务本身能力有限。

4. 误区：教学代码可以直接扩展成生产大模型。
   纠正：生产训练需要大量系统、数值稳定性和性能优化。

### 小练习

1. 给定 token 序列 `[1,2,3,4,5]`，构造 block_size=4 的 input 和 target。
2. 修改 `generate`，实现 greedy decoding。
3. 给 miniGPT 加 dropout。
4. 把 position embedding 替换成 RoPE 的接口设计。
5. 打印每一层输出 shape，确认数据流。

### 本讲总结

本讲最重要的结论：

1. GPT 是 decoder-only Transformer。
2. miniGPT 的核心包括 embedding、causal attention、MLP、norm、residual、lm head。
3. 训练 label 是 input 右移一位。
4. loss 是所有 token 位置的 cross entropy。
5. 生成时逐 token 采样，并把新 token 拼回上下文。
6. 教学版 miniGPT 能帮助理解结构，但真实 LLM 还需要大量工程优化。

关键问题：GPT 模型核心模块如何组织，输入和 label 如何构造，causal attention 如何实现。
