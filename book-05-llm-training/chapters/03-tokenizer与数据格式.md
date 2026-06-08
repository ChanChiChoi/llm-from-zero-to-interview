# 第三章：Tokenizer 与数据格式

## 本章目标

理解 tokenizer 如何影响训练效率、上下文长度、多语言能力、代码能力和特殊任务格式。

## 核心议题

1. BPE、SentencePiece、byte-level tokenizer。
2. 词表大小和压缩率。
3. 特殊 token：BOS、EOS、PAD、UNK、system、user、assistant、tool。
4. Chat template 和 label mask。
5. 多模态 token、图像 patch token、语音 token。

## 面试重点

Tokenizer 不是简单预处理，而是模型输入空间的定义。它会影响训练 token 数、推理成本、上下文利用率和跨语言表现。

## 为什么 tokenizer 是训练协议

大模型不能直接处理字符串。所有文本都要先经过 tokenizer，变成 token id，再进入 embedding 层。

```text
原始文本
  -> tokenizer
  -> token ids
  -> embedding
  -> Transformer
  -> logits over vocabulary
```

所以 tokenizer 不是一个可随便替换的小工具，而是模型输入输出协议的一部分。

如果 tokenizer 变了，token id 的含义、词表大小、特殊 token、chat template 和 embedding 矩阵都会受影响。对于已经训练好的模型，随意更换 tokenizer 通常会导致模型无法正常理解输入。

面试表达：tokenizer 定义了自然语言进入模型的离散接口，也定义了模型输出如何还原为文本。

### Tokenizer 与数据格式的资料边界

按照 `WRITING_PLAN.md` 的要求，本章第二轮精修前核对了 BPE 原论文、SentencePiece 论文、OpenAI `tiktoken`、Hugging Face Tokenizers / Transformers tokenizer 文档、Transformers chat template 文档、TRL `SFTTrainer` 的 assistant-only / completion-only loss 说明，以及 PyTorch `CrossEntropyLoss(ignore_index)` 文档。

本章聚焦 tokenizer 和训练数据格式的工程协议：分词映射、词表大小、压缩率、special token、chat template、label mask、packing、SFT / 偏好 / tool calling / 多模态格式和扩词表风险。它不展开生产级 tokenizer 训练作业、完整 BPE / Unigram LM 训练器源码、所有模型家族模板差异，也不替代框架官方接口说明。

### 关键公式口径

给定 tokenizer `T` 和文本 `x`，编码结果可以写成：

```math
z=T(x)=(z_1,\ldots,z_L)
```

其中每个 token id 满足：

```math
z_t\in \{0,\ldots,V-1\}
```

`V` 是词表大小，`L` 是这段文本被切成的 token 数。tokenizer 效率可以用每字符 token 数衡量：

```math
\rho_{\mathrm{char}}(x)=\frac{L}{|x|_{\mathrm{char}}}
```

全量训练 token 数是：

```math
N_{\mathrm{tok}}=\sum_{i=1}^{n}|T(x_i)|
```

所以 tokenizer 会直接改变训练步数、注意力计算量、KV cache 长度和推理生成步数。面试中说“tokenizer 影响成本”，最好能落到 `L` 和 `N_tok` 这两个量上。

## 1. Tokenizer 影响哪些训练问题

Tokenizer 至少影响五类问题。

| 影响维度 | 说明 |
| --- | --- |
| 训练 token 数 | 同一文本切得越碎，训练步数和计算越多 |
| 上下文利用率 | 8K context 是 8K token，不是 8K 字符 |
| 多语言能力 | 低资源语言如果切得太碎，学习效率会降低 |
| 代码能力 | 空格、缩进、符号和标识符切分会影响代码建模 |
| 特殊任务格式 | chat、tool、多模态都依赖 special token 和模板 |

一个简单例子：

```text
英文：The model is good.
中文：这个模型很好。
代码：def get_user_id(x): return x.user_id
```

不同 tokenizer 对这三类文本的切分效率可能差异很大。如果中文或代码被切成过多 token，同样的上下文窗口能容纳的信息就更少，训练和推理成本也更高。

## 2. 常见 tokenizer 方法

现代大模型主流使用子词或字节级方案，而不是传统词级分词。

### 2.1 BPE

BPE 的核心思想是从小单位开始，不断合并高频相邻片段。

直觉上，它会把常见词、词根、代码片段或标点组合合并成更长 token，从而减少序列长度。

第 `r` 轮合并时，可以把最高频相邻片段写成：

```math
(a^\*,b^\*)=\arg\max_{(a,b)} C_r(a,b)
```

其中 `C_r(a,b)` 是当前语料分词结果中相邻片段 `(a,b)` 的出现次数。合并后的新片段是：

```math
u^\*=a^\*b^\*
```

词表更新为：

```math
V_{r+1}=V_r\cup \{u^\*\}
```

这个公式只表达 BPE 的核心机制：每轮把高频相邻片段合成一个更长 token。真实工程里还要处理预分词、字节级编码、特殊 token、正则规则和训练语料采样。

优点：

1. 简单有效。
2. 工程成熟。
3. 可以处理未见词，因为未见词能退化为更小片段。

缺点：

1. 合并规则来自训练语料，容易继承语料偏向。
2. 对低资源语言可能切得更碎。
3. 不保证切分符合语义边界。

### 2.2 SentencePiece

SentencePiece 常用于多语言模型，因为它可以直接从原始文本训练 tokenizer，不强依赖语言特定的预分词。

它常把空格也纳入模型处理，例如用特殊符号表示词边界。

优点：

1. 适合没有天然空格分词的语言。
2. 适合多语言混合语料。
3. 支持 BPE 和 Unigram LM 等算法。

面试中要注意：SentencePiece 不是单一切分算法，而是一个 tokenizer 训练和处理工具体系。

### 2.3 Byte-level tokenizer

Byte-level tokenizer 从字节层面保证覆盖率。任意 Unicode 字符最终都能被表示，因此几乎没有 OOV。

覆盖率可以写成：

```math
\forall x,\quad T_{\mathrm{byte}}(x)\ne \varnothing
```

意思是任何输入字符串都至少能退化成字节序列再编码。这个性质对代码、脏数据、混合语言和特殊符号很重要。

优点：

1. 鲁棒性强。
2. 能处理特殊字符、emoji、代码和脏输入。
3. 不容易因为未登录字符崩掉。

代价：

1. 某些语言可能 token 数变多。
2. 部分文本的语义单位被切得很碎。
3. 上下文窗口利用率可能降低。

面试表达：byte-level 的核心价值是覆盖和鲁棒，代价是部分文本上的 token 效率。

## 3. 词表大小如何权衡

词表大小不是越大越好。

假设 vocabulary size 是 `V`，hidden size 是 `d`，那么输入 embedding 参数量大约是：

```math
P_{\mathrm{emb}}=Vd
```

如果输出层没有和 embedding 共享权重，还会多出一个同规模矩阵：

```math
P_{\mathrm{out}}=Vd
```

每个位置都要预测 `V` 类 token，输出 logits 的形状是：

```math
\ell\in \mathbb{R}^{B\times L\times V}
```

其中 `B` 是 batch size，`L` 是序列长度。如果词表很大，embedding、输出层和 softmax 都会更重。

### 3.1 大词表的好处

1. 常见文本可以切得更短。
2. 多语言、代码和专业词可能有更好覆盖。
3. 推理时生成同样文本可能需要更少 steps。

### 3.2 大词表的代价

1. embedding 和输出层参数增加。
2. softmax 计算成本增加。
3. 低频 token 学得不充分。
4. 词表训练更依赖语料分布。

### 3.3 小词表的代价

小词表参数少，但序列更长。

序列变长会带来两个问题：

1. Attention 计算和显存更高。
2. 同样上下文窗口能放下的信息更少。

所以词表大小要结合语言覆盖、训练语料、上下文长度、推理成本和模型规模综合选择。

面试表达：词表大小是在 embedding/softmax 成本和序列长度之间做 trade-off。

## 4. 压缩率和 tokenizer 效率

评估 tokenizer 时，一个重要指标是压缩率，或者说每段文本会被切成多少 token。

例如可以比较：

```text
每 1000 个字符对应多少 token
每 1000 个英文单词对应多少 token
每 100 行代码对应多少 token
```

如果某种语言被切得特别碎，它在训练中等价于“更贵”。

比如同样 2048 token 的上下文：

```text
高效 tokenizer：能放入较长文章
低效 tokenizer：只能放入较短文章
```

这会影响长上下文任务、RAG、代码文件输入和多轮对话。

可以把 token 效率写成：

```math
R_{\mathrm{tok/byte}}(D)=
\frac{\sum_{x\in D}|T(x)|}{\sum_{x\in D}|x|_{\mathrm{byte}}}
```

如果上下文窗口是 `C_tok`，某个语料桶平均每字符 token 数是 `rho_char`，那么可容纳的字符数近似是：

```math
C_{\mathrm{char}}\approx \frac{C_{\mathrm{tok}}}{\rho_{\mathrm{char}}}
```

这解释了为什么同样是 8K token，上下文里能放下多少中文、英文、代码或 JSON 可能差异很大。

面试中可以说：tokenizer 的效率最终会体现在有效上下文长度、训练 token 预算和推理费用上。

## 5. Special token

Special token 是具有控制语义的 token，不是普通字符串。

常见 special token：

| token | 作用 |
| --- | --- |
| BOS | 序列开始 |
| EOS | 序列结束 |
| PAD | batch padding |
| UNK | 未知 token |
| system | 系统指令边界 |
| user | 用户消息边界 |
| assistant | 助手消息边界 |
| tool | 工具调用或工具结果边界 |
| image | 图像占位符 |
| audio | 音频占位符 |

这些 token 的处理必须在训练、微调、评估和推理中一致。

如果训练时使用一种格式，推理时使用另一种格式，模型可能无法正确理解角色、边界和停止条件。

### 5.1 EOS 很重要

EOS 告诉模型何时结束。

如果训练数据里 EOS 处理不一致，模型可能出现：

1. 生成停不下来。
2. 过早停止。
3. 多轮对话边界混乱。

### 5.2 PAD 不能参与 loss

batch 训练时常需要 padding 到相同长度。

PAD token 只是占位，不应该让模型学习预测 PAD。

所以训练 loss 通常要把 PAD 位置 mask 掉。

attention mask 可以写成：

```math
a_t=\mathbf{1}[z_t\ne z_{\mathrm{pad}}]
```

计算 loss 时也要忽略 PAD 对应位置：

```math
y_t=
\begin{cases}
z_t,& a_t=1\\
-100,& a_t=0
\end{cases}
```

这里 `-100` 是很多 PyTorch / Hugging Face 训练脚本默认使用的 ignore index。核心思想不是数字本身，而是 PAD 位置不能贡献 loss。

## 6. Chat template

Chat template 定义多轮对话如何序列化成模型输入。

例如一段对话：

```text
system: 你是一个有帮助的助手。
user: 什么是 tokenizer？
assistant: tokenizer 是把文本转成 token 的组件。
```

序列化后可能变成：

```text
<bos><system>
你是一个有帮助的助手。
<user>
什么是 tokenizer？
<assistant>
tokenizer 是把文本转成 token 的组件。<eos>
```

不同模型的 chat template 可能完全不同。即使内容一样，模板不同也会影响模型行为。

如果多轮消息是：

```math
M=((r_1,c_1),\ldots,(r_K,c_K))
```

其中 `r_k` 是 role，`c_k` 是内容，chat template 可以看成一个序列化函数：

```math
s=T_{\mathrm{chat}}(M)
```

再经过 tokenizer 得到：

```math
z=T(s)
```

因此 SFT 的真实输入不是原始 JSON，而是 `T(T_chat(M))`。如果训练和推理的 `T_chat` 不一致，模型看到的角色边界、停止 token 和回答起点都会错位。

面试表达：chat template 是 chat model 的输入协议，它把角色、轮次和输出位置显式编码进 token 序列。

## 7. Label mask

SFT 训练中经常不是对所有 token 都算 loss。

典型做法是只让模型学习 assistant 的回答，而不学习 user prompt。

例如：

```text
system: 你是助手。
user: 解释 attention。
assistant: attention 是一种信息聚合机制。
```

训练输入包含全部 token，但 loss 只算 assistant 部分：

```text
system tokens      -> ignore
user tokens        -> ignore
assistant tokens   -> compute loss
```

这就是 assistant-only loss mask。

### 7.1 为什么不对 user 部分算 loss

如果对 user prompt 也算 loss，模型会被训练去复现用户输入，而不是专注学习如何回答。

对 SFT 来说，用户输入是条件，助手回答才是监督目标。

### 7.2 常见错误

1. 把 system/user/assistant 全部算 loss。
2. assistant 起始 token 没处理好。
3. 多轮对话只 mask 了最后一轮。
4. padding 位置没有忽略。
5. template 和 label mask 不匹配。

这些错误会让 loss 看起来正常，但模型行为变差。

对 causal LM 来说，assistant-only SFT loss 可以写成：

```math
L_{\mathrm{sft}}=
\frac{
\sum_{t=1}^{L-1}m_{t+1}
\left[-\log p_\theta(z_{t+1}\mid z_{\le t})\right]
}{
\max\left(\sum_{t=1}^{L-1}m_{t+1},1\right)
}
```

其中 `m_t=1` 表示第 `t` 个 token 属于 assistant 监督区域，`m_t=0` 表示 system、user、tool result 或 padding 等只作为条件。分母用有效监督 token 数归一化，避免不同样本回答长度差异过大时 loss 口径混乱。

面试表达：SFT 的关键不是只把对话拼起来，还要确保 label mask 和 chat template 对齐。

## 8. 预训练数据格式

预训练通常不是 chat 格式，而是大量连续文本。

常见格式是：

```text
document_1 <eos> document_2 <eos> document_3 <eos>
```

训练目标是 next token prediction。

### 8.1 文档边界

文档之间通常需要边界 token，避免模型把两个无关文档当成连续上下文。

但是否插入 EOS、如何拼接、是否允许跨文档 attention，会影响训练行为。

### 8.2 Packing

为了提高训练效率，短文档通常会 pack 到固定长度 block 中。

例如把多个短样本拼成 2048 token 或 4096 token 的训练块。

packing 的好处是减少 padding，提高 GPU 利用率。

风险是如果边界处理不好，模型会学习到不自然的跨样本连接。

假设每个文档编码后追加 EOS：

```math
s_i=T(x_i)\oplus [z_{\mathrm{eos}}]
```

把所有文档拼成总序列：

```math
S=s_1\oplus s_2\oplus \cdots \oplus s_n
```

固定 block size 为 `C` 时，第 `j` 个训练块可以写成：

```math
b_j=S_{jC:(j+1)C-1}
```

如果最后一个 block 不足 `C`，再用 PAD 补齐并设置 attention mask。关键是保留 EOS 或显式边界，否则模型会把两个无关文档误当作自然连续文本。

## 9. SFT 数据格式

SFT 数据通常是 instruction-response 或多轮 messages。

单轮格式：

```json
{
  "instruction": "解释什么是 RoPE",
  "input": "",
  "output": "RoPE 是一种旋转位置编码..."
}
```

多轮格式：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个严谨的助手。"},
    {"role": "user", "content": "解释什么是 RoPE"},
    {"role": "assistant", "content": "RoPE 是一种旋转位置编码..."}
  ]
}
```

训练前要通过 chat template 转成 token ids，并生成 label mask。

可以把一条 SFT 样本写成：

```math
d_i=(M_i,z_i,m_i)
```

其中 `M_i` 是 messages，`z_i=T(T_chat(M_i))` 是 token ids，`m_i` 是 assistant-only loss mask。真正喂给模型的是 `z_i` 和 `m_i`，不是 JSON 文件本身。

面试中要强调：SFT 数据格式不只是 JSON 长什么样，更重要的是最后进入模型的 token 序列和 loss mask 是否正确。

## 10. 偏好数据格式

DPO、RLHF、GRPO 等偏好或强化学习阶段的数据格式和 SFT 不同。

最常见的偏好数据包含：

```json
{
  "prompt": "用户问题...",
  "chosen": "更好的回答...",
  "rejected": "更差的回答..."
}
```

关键点：

1. chosen 和 rejected 必须基于同一个 prompt。
2. 两个回答要用同一个 chat template 编码。
3. loss 通常只关注回答部分。
4. 要避免 chosen 只是更长，而不是真的更好。

如果数据格式处理错，偏好优化可能会学到错误偏好，例如偏向长回答、模板化回答或过度拒答。

偏好样本可以写成：

```math
d_i=(x_i,y_i^+,y_i^-)
```

其中 `x_i` 是同一个 prompt，`y_i^+` 是 chosen，`y_i^-` 是 rejected。两条回答必须使用同一个 tokenizer 和同一个 chat template 编码：

```math
z_i^+=T(T_{\mathrm{chat}}(x_i,y_i^+))
```

```math
z_i^-=T(T_{\mathrm{chat}}(x_i,y_i^-))
```

否则比较的不是“回答质量”，而可能是模板、截断或 tokenization 差异。

## 11. Tool calling 数据格式

工具调用模型需要额外表达工具 schema、工具调用和工具返回结果。

一个简化格式：

```text
<user>
查一下今天北京天气
<assistant_tool_call>
{"name": "weather", "arguments": {"city": "北京"}}
<tool_result>
{"temperature": "20C", "condition": "晴"}
<assistant>
今天北京天气晴，约 20C。
```

工具调用数据的难点：

1. JSON 格式必须稳定。
2. 工具参数要合法。
3. 工具结果和最终回答要对应。
4. tool token 边界要清晰。

如果格式不稳定，模型可能生成非法 JSON，或者把工具结果当成用户输入。

工具调用数据至少要保证函数名和参数 schema 可校验。简化门禁可以写成：

```math
G_{\mathrm{tool}}=
g_{\mathrm{name}}\,
g_{\mathrm{json}}\,
g_{\mathrm{schema}}\,
g_{\mathrm{result}}\,
g_{\mathrm{boundary}}
```

这些检查分别表示工具名合法、JSON 可解析、参数符合 schema、工具结果和最终回答一致、tool call / tool result / assistant 边界清楚。

## 12. 多模态 token

多模态模型通常需要把图片、音频或视频接入语言模型。

常见做法不是直接把图片像素当成普通文本 token，而是使用占位符或经过 encoder 的视觉表示。

例如：

```text
<user>
<image>
这张图里有什么？
<assistant>
图中有一只猫。
```

这里的 `<image>` 可能表示：

1. 一个图像占位 token。
2. 多个视觉 patch embedding 插入位置。
3. 连接到 vision encoder 输出的桥接位置。

语音模型也可能使用 audio token、codec token 或音频 encoder 输出。

面试表达：多模态 token 的关键是对齐。文本 token、图像 patch、音频片段和位置关系必须在模型输入中有一致协议。

## 13. Tokenizer 扩展

有时需要给已有模型增加特殊 token，例如工具 token、图像 token、领域控制 token。

这时通常需要：

1. 修改 tokenizer vocabulary。
2. resize embedding。
3. 初始化新增 token embedding。
4. 用包含新 token 的数据继续训练或微调。
5. 确保推理侧模板同步更新。

常见风险：

1. 只改 tokenizer，不改 embedding。
2. 新 token 没训练够，模型不会用。
3. 训练和推理模板不一致。
4. 新 token 和普通文本字符串混淆。

如果原词表大小是 `V`，新增 `K` 个 special token 后：

```math
V'=V+K
```

embedding 矩阵也必须从：

```math
E\in \mathbb{R}^{V\times d}
```

扩展为：

```math
E'\in \mathbb{R}^{V'\times d}
```

新增 token embedding 可以随机初始化，也可以用相关 token embedding 的均值初始化，但无论哪种方式，都需要后续训练让模型真正学会这些 token 的语义。

面试表达：新增 special token 是模型协议变更，不是只改配置文件。

## 14. 一个最小 label mask 示例

下面这个 demo 用标准库实现一个 toy tokenizer 和数据格式审计器。它不追求复刻真实 BPE，而是把最容易出错的训练协议跑一遍：压缩率统计、chat template 序列化、assistant-only label mask、预训练 packing、PAD attention mask、EOS 边界和新增 special token 后的 embedding 行数。

```python
IGNORE_INDEX = -100
SPECIAL_TOKENS = ["<bos>", "<eos>", "<system>", "<user>", "<assistant>", "<pad>", "<image>"]
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

vocab = {}
id_to_token = {}


def add_token(token):
    if token not in vocab:
        idx = len(vocab)
        vocab[token] = idx
        id_to_token[idx] = token
    return vocab[token]


for token in SPECIAL_TOKENS + BASE_PIECES:
    add_token(token)

match_pieces = sorted(BASE_PIECES, key=len, reverse=True)


def fallback_piece(ch):
    return f"<U+{ord(ch):04X}>"


def tokenize_text(text):
    pieces = []
    i = 0
    while i < len(text):
        matched = None
        for piece in match_pieces:
            if text.startswith(piece, i):
                matched = piece
                break
        if matched is None:
            matched = fallback_piece(text[i])
        pieces.append(matched)
        i += len(matched) if not matched.startswith("<U+") else 1
    return pieces


def encode_pieces(pieces):
    return [add_token(piece) for piece in pieces]


def decode_ids(ids):
    return [id_to_token[i] for i in ids]


def serialize_chat(messages):
    pieces = ["<bos>"]
    assistant_mask = [0]
    for msg in messages:
        role_piece = f"<{msg['role']}>"
        pieces.append(role_piece)
        assistant_mask.append(0)

        content_pieces = tokenize_text(msg["content"])
        pieces.extend(content_pieces)
        assistant_mask.extend([1 if msg["role"] == "assistant" else 0] * len(content_pieces))

        if msg["role"] == "assistant":
            pieces.append("<eos>")
            assistant_mask.append(1)

    input_ids = encode_pieces(pieces)
    labels = [
        token_id if mask else IGNORE_INDEX
        for token_id, mask in zip(input_ids, assistant_mask)
    ]
    return input_ids, labels, assistant_mask


def pack_documents(docs, block_size):
    blocks = []
    masks = []
    current = []
    for doc in docs:
        doc_ids = encode_pieces(tokenize_text(doc) + ["<eos>"])
        if len(current) + len(doc_ids) > block_size and current:
            pad_len = block_size - len(current)
            blocks.append(current + [vocab["<pad>"]] * pad_len)
            masks.append([1] * len(current) + [0] * pad_len)
            current = []
        current.extend(doc_ids)
    if current:
        pad_len = block_size - len(current)
        blocks.append(current + [vocab["<pad>"]] * pad_len)
        masks.append([1] * len(current) + [0] * pad_len)
    return blocks, masks


texts = {
    "en": "tokenizer improves attention",
    "zh": "数据工程需要保留来源",
    "code": "def get_user_id(x): return x.user_id",
}
compression = {
    name: {
        "chars": len(text),
        "tokens": len(tokenize_text(text)),
        "tokens_per_char": round(len(tokenize_text(text)) / len(text), 3),
    }
    for name, text in texts.items()
}

messages = [
    {"role": "system", "content": "你是严谨助手。"},
    {"role": "user", "content": "解释 tokenizer"},
    {"role": "assistant", "content": "tokenizer 是文本到 token id 的协议。"},
]
input_ids, labels, assistant_mask = serialize_chat(messages)
label_tokens = [id_to_token[x] if x != IGNORE_INDEX else "IGN" for x in labels]
first_assistant_pos = assistant_mask.index(1)
prompt_label_count = sum(1 for x in labels[:first_assistant_pos] if x != IGNORE_INDEX)
assistant_label_count = sum(1 for x in labels if x != IGNORE_INDEX)

blocks, attention_masks = pack_documents(list(texts.values()), block_size=16)

old_vocab_size = len(vocab)
added = 0
for token in ["<tool_call>", "<tool_result>"]:
    before = len(vocab)
    add_token(token)
    added += len(vocab) - before
new_vocab_size = len(vocab)
embedding_rows = new_vocab_size

packed_eos_count = sum(block.count(vocab["<eos>"]) for block in blocks)
gates = {
    "specials": all(token in vocab for token in SPECIAL_TOKENS),
    "assistant_only_labels": prompt_label_count == 0 and assistant_label_count > 0,
    "pad_mask": all(
        (tok != vocab["<pad>"]) == bool(mask)
        for block, mask_row in zip(blocks, attention_masks)
        for tok, mask in zip(block, mask_row)
    ),
    "packing_shape": all(len(block) == 16 for block in blocks),
    "eos_boundary": packed_eos_count == len(texts),
    "vocab_resize": new_vocab_size == old_vocab_size + added == embedding_rows,
}

print("compression=", compression)
print("chat_tokens=", decode_ids(input_ids))
print("label_tokens=", label_tokens)
print("assistant_label_count=", assistant_label_count)
print("prompt_label_count=", prompt_label_count)
print("packed_attention=", [sum(mask) for mask in attention_masks])
print("packed_pad_count=", [mask.count(0) for mask in attention_masks])
print("packed_eos_count=", packed_eos_count)
print("old_vocab_size=", old_vocab_size)
print("new_vocab_size=", new_vocab_size)
print("gates=", gates)
print("gate_pass=", all(gates.values()))
```

运行后应看到：

```text
compression= {'en': {'chars': 28, 'tokens': 5, 'tokens_per_char': 0.179}, 'zh': {'chars': 10, 'tokens': 4, 'tokens_per_char': 0.4}, 'code': {'chars': 36, 'tokens': 13, 'tokens_per_char': 0.361}}
chat_tokens= ['<bos>', '<system>', '你', '是', '严谨', '助手', '。', '<user>', '解释', ' ', 'tokenizer', '<assistant>', 'tokenizer', ' ', '是', '文本', '到', ' ', 'token', ' ', 'id', ' ', '的', '协议', '。', '<eos>']
label_tokens= ['IGN', 'IGN', 'IGN', 'IGN', 'IGN', 'IGN', 'IGN', 'IGN', 'IGN', 'IGN', 'IGN', 'IGN', 'tokenizer', ' ', '是', '文本', '到', ' ', 'token', ' ', 'id', ' ', '的', '协议', '。', '<eos>']
assistant_label_count= 14
prompt_label_count= 0
packed_attention= [11, 14]
packed_pad_count= [5, 2]
packed_eos_count= 3
old_vocab_size= 36
new_vocab_size= 38
gates= {'specials': True, 'assistant_only_labels': True, 'pad_mask': True, 'packing_shape': True, 'eos_boundary': True, 'vocab_resize': True}
gate_pass= True
```

这个 demo 里有几个检查点：

1. `<system>`、`<user>`、`<assistant>`、`<eos>`、`<pad>` 都是 special token，不是普通字符串。
2. `prompt_label_count=0` 说明 system 和 user 部分没有参与 loss。
3. `packed_pad_count` 和 `pad_mask` 说明 PAD 只占位，不参与 attention。
4. `packed_eos_count=3` 说明三个预训练文档边界都被保留。
5. `new_vocab_size=38` 说明新增 tool token 后 embedding 行数也要同步变大。

但核心原则不变：输入可以包含完整上下文，loss 只在需要模型学习生成的位置计算。

## 15. 常见坑

### 15.1 训练和推理 tokenizer 不一致

这是严重错误。token id 含义不同会让模型输入完全错位。

### 15.2 忘记 EOS

模型可能不知道何时停止，或多轮回答边界混乱。

### 15.3 label mask 错误

loss 可能正常下降，但模型学到复述 prompt 或生成格式异常。

### 15.4 截断策略错误

如果总是截断结尾，可能把答案截掉；如果总是截断开头，可能丢失系统指令。

### 15.5 多语言切分不均衡

某些语言 token 数过多，会导致训练成本更高、上下文更短。

### 15.6 特殊 token 被当普通字符串

如果 `<assistant>` 只是普通字符序列，而不是 tokenizer 中的 special token，模型可能无法稳定学习角色边界。

## 16. 面试官会怎么问

### 问法 1：为什么 tokenizer 会影响训练成本？

可以这样答：

```text
因为训练和推理都是按 token 计算的。同一段文本如果被切成更多 token，就会增加序列长度、attention 计算、KV cache 和生成步数。对于中文、代码、多语言和长文档，tokenizer 的压缩率会直接影响有效上下文长度和训练成本。
```

### 问法 2：词表越大越好吗？

可以这样答：

```text
不一定。大词表可以减少序列长度，但会增加 embedding 和输出 softmax 的参数与计算成本，也可能导致低频 token 学不好。小词表参数少，但文本会切得更碎，attention 成本上升。词表大小本质是在词表成本和序列长度之间权衡。
```

### 问法 3：SFT 中为什么要做 label mask？

可以这样答：

```text
因为用户输入和系统提示是条件，不是模型要学习生成的目标。SFT 通常只在 assistant 回答部分计算 loss。如果把用户问题也算进 loss，模型会被训练去复现用户输入，影响指令跟随效果。
```

### 问法 4：新增 special token 要注意什么？

可以这样答：

```text
新增 special token 后要同步修改 tokenizer、resize embedding，并用包含这些 token 的数据训练新增 embedding。同时训练、评估和推理侧的 chat template 必须一致，否则模型可能无法理解角色边界或工具调用格式。
```

### 问法 5：多模态里的 image token 是什么？

可以这样答：

```text
image token 通常不是图片像素本身，而是图像输入在语言模型序列中的占位或桥接位置。真实视觉信息可能来自 vision encoder 的 patch embedding，再通过 projector 接入 LLM。关键是文本 token 和视觉 embedding 在输入序列中要有一致的对齐协议。
```

## 17. 本章小结

本章核心结论：

1. Tokenizer 是模型输入输出协议，不是简单预处理。
2. Tokenizer 会影响训练 token 数、上下文利用率、多语言能力、代码能力和推理成本。
3. BPE、SentencePiece、byte-level tokenizer 各有覆盖率、效率和工程取舍。
4. 词表大小要在序列长度和 embedding/softmax 成本之间权衡。
5. Special token 和 chat template 定义了对话模型的结构协议。
6. SFT 训练必须正确处理 label mask。
7. 预训练、SFT、偏好优化、tool calling 和多模态训练的数据格式不同。
8. 新增 special token 需要同步 tokenizer、embedding、训练数据和推理模板。
9. 面试中要把 tokenizer 讲成训练系统的一部分，而不是字符串切分工具。
