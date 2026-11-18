# 第 3 章（第五篇）：Tokenizer 迁移与公平比较

字符级模型适合作为容易观察的教学基线，但它的 token 数、词表规模和上下文覆盖都与子词 tokenizer 不同。本篇先说明 BPE 怎样合并 token，再比较序列长度和词表参数带来的取舍，最后对齐训练预算、重复实验和证据要求，说明要凭什么判断一种 tokenizer 是否更合适。

## 3.6 从字符级迁移到 BPE：tokenizer 与 checkpoint 的兼容性

### 3.6.1 BPE 改变的是 token 化，不是自回归目标

字符级 tokenizer 把 transformer 拆成许多字符；子词 tokenizer 可能把它表示为一个或多个片段。BPE 的核心操作是从初始符号开始，反复把高频相邻 pair 合并成新符号；这是 Sennrich 等人在机器翻译子词方案中采用的基本过程，具体 tokenizer 还会增加预切分和规范化等步骤：[原论文](https://arxiv.org/abs/1508.07909v5)。设当前 pair 集合为 $P$，pair 频次为 $c(a,b)$，一个简化的合并规则是：

```math
(a^*,b^*)=\arg\max_{(a,b)\in P}c(a,b)
```

把所有相邻的 $a^*,b^*$ 替换成 $a^*b^*$，直到达到词表目标或没有继续合并的收益。真实 byte-level BPE 还涉及字节映射、pre-tokenization、normalization、merge rank 和特殊 token；SentencePiece 是 tokenizer 工具，可以承载 BPE 或 Unigram，因此“使用 SentencePiece”本身并不唯一确定内部模型：[原论文](https://arxiv.org/abs/1808.06226)。

换 tokenizer 后，next-token 目标仍是：

```math
x_j=d_{s+j},\qquad y_j=d_{s+j+1}
```

变化的是 $d_j$ 的单位、序列长度、词表大小和 decode 规则。

### 3.6.2 序列节省与词表开销需要同时计算

若同一段文本在字符级和 BPE 级的长度分别为 $T_c$ 和 $T_b$，标准 full attention 的 score 矩阵规模比约为：

```math
R_{\mathrm{score}}=\frac{T_b^2}{T_c^2}
```

当 $T_b<T_c$ 时，序列相关的 attention 计算和中间显存会下降。但词表从 $V_c$ 变成 $V_b$ 后，embedding 参数量从 $V_cD$ 变成 $V_bD$，未绑权重的 lm head 还大约增加 $DV_b+V_b$。因此应分开记录序列计算量和词表参数量，而不是把单位不同的 FLOPs 与参数个数直接相加：

```math
(\Delta\mathrm{FLOPs}_{\mathrm{attention}},
\Delta N_{\mathrm{embedding+head}})
```

这里两个 `\Delta` 都表示 BPE 相对字符级的差值。attention FLOPs 还要结合序列长度、kernel、dtype 和硬件才能换算成运行成本；embedding/head 的参数量则影响权重与优化器状态内存。小语料、小模型和超大词表组合在一起时，embedding 与 lm head 可能吞掉相当比例的参数预算。

### 3.6.3 使用现成 tokenizer 的接口

下面是 Hugging Face Transformers v5.17.0 tokenizer API 的接口示例。它需要安装 `transformers`、网络或本地缓存，不能在没有依赖和模型文件的环境中当作零依赖 demo 执行；代码只展示 tokenizer 接口，不加载模型：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import torch
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "hello transformer, hello language model"
ids = tokenizer.encode(text, add_special_tokens=False)
recovered = tokenizer.decode(ids)

vocab_size = len(tokenizer)
input_ids = torch.tensor([ids], dtype=torch.long)

print("token_count=", len(ids))
print("vocab_size=", vocab_size)
print("roundtrip=", recovered)
print("input_shape=", tuple(input_ids.shape))
```
整数 id 的具体值由 tokenizer artifact 决定，不应把某一次运行的 id 列表写成跨版本固定答案。`len(tokenizer)` 通常比 `tokenizer.vocab_size` 更适合表示当前可用的完整词表，尤其是在添加 special token 之后；Transformers 文档也要求扩容模型时传入完整 tokenizer 长度：[v5.17.0 Tokenizer API](https://huggingface.co/docs/transformers/v5.17.0/main_classes/tokenizer)。本段只演示接口，未固定 Hub 仓库的 `revision`；要复现实验，应锁定 commit hash 并保存 tokenizer artifact。

### 3.6.4 special token、padding 与 loss mask

连续 token 流切成固定长度 block 时不需要 padding。不同长度序列组成 batch 时才需要 pad token。下面的两段代码属于同一个 tokenizer 上下文，必须先完成 3.6.3 的加载。若把 EOS 复用为 PAD：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```
这只解决了“使用哪个 id 填充”的接口问题，不会自动告诉 attention 和 loss 哪些位置有效。带 padding 的 batch 还需要：

```math
\mathrm{attention\_mask}_{b,t}=0
\quad (t\in\mathcal{P}_b)
```

为了避免在数学围栏里依赖文本命令，也可以把它写成集合约束：

```math
t\in\mathcal{P}_b\Longrightarrow
\mathrm{attention\_mask}_{b,t}=0,
\qquad
\mathrm{label}_{b,t}=-100
```

其中 $\mathcal{P}_b$ 是第 $b$ 条样本的 padding 位置集合，-100 是常见的 `ignore_index`；此处采用“1 表示可见、0 表示 padding”的 attention mask 约定，其他 API 的布尔语义要按目标版本核对。若 PAD id 与 EOS id 相同，不能只按 token id 把所有同 id 的 label 设为 -100，否则真实 EOS 目标也可能被屏蔽。应从样本长度或 tokenizer 生成的有效位置 mask 标出真正的 padding；Transformers v5.17.0 的 `DataCollatorForLanguageModeling(mlm=False)` 会按 `pad_token_id` 屏蔽标签，因此复用 EOS 时应检查它与所用数据整理器的组合行为：[固定版本源码](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/data/data_collator.py#L3414-L3422)。EOS 兼作 PAD 时还要确认生成终止逻辑不会把批处理填充误当成真实结束信号。

### 3.6.5 模型和 checkpoint 的迁移点

从字符级模型迁移到 BPE，Transformer block 的 hidden shape 可以保持不变，但下面几项必须一起变化：

| 对象 | 字符级版本 | BPE 版本 |
| --- | --- | --- |
| 编码 | stoi 或字符 codec | tokenizer artifact |
| 词表大小 | len(chars) | len(tokenizer) |
| 输入 id | 字符 id | 子词/字节片段 id |
| 输出头 | Linear(D, V_char) | Linear(D, V_bpe) |
| 解码 | itos 拼接 | tokenizer.decode |
| checkpoint 元数据 | stoi/itos | tokenizer 文件、版本、配置、special ids |

如果加载一个已有模型后添加 special token，需要同步调整 embedding 和输出头；在 Hugging Face 模型中通常使用 `model.resize_token_embeddings(len(tokenizer))`，其接口要求给出完整词表长度：[Transformers v5.17.0 文档](https://huggingface.co/docs/transformers/v5.17.0/main_classes/tokenizer)。从零训练时直接以新词表大小初始化，不能把旧字符级权重的行号当作新 BPE 权重的语义。

checkpoint 应至少保存以下 tokenizer 信息：

```text
tokenizer.json 或等价完整 artifact
词表和 merge 规则
normalization / pre-tokenization 配置
special token 到 id 的映射
tokenizer 库和版本
词表大小
训练数据切分与编码配置
```

只保存一个本地路径不够，因为路径可能在另一台机器不存在，文件内容也可能被替换。tokenizer 是模型输入输出协议的一部分，不是训练脚本的可选装饰。

### 3.6.6 一个零依赖的 BPE 合并演示

下面手动给出 merge 规则，不是完整 BPE trainer；它只用来观察 token 数、右移标签和词表相关参数量。真实 BPE 的 merge 规则应由训练语料统计得到。

> 代码状态：静态审阅，未运行（2026-11-16）

```python
TEXT = "low lower lowest low lower transformer transformer low"
MERGES = [
    ("l", "o"),
    ("lo", "w"),
    ("low", "e"),
    ("lowe", "r"),
    ("lowe", "s"),
    ("lowes", "t"),
    ("t", "r"),
    ("tr", "a"),
    ("tra", "n"),
    ("tran", "s"),
    ("trans", "f"),
    ("transf", "o"),
    ("transfo", "r"),
    ("transfor", "m"),
    ("transform", "e"),
    ("transforme", "r"),
]
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def initial_symbols(text):
    return [character for character in text]


def apply_merges(tokens, merges):
    tokens = list(tokens)
    for left, right in merges:
        merged_tokens = []
        index = 0
        while index < len(tokens):
            pair = tokens[index:index + 2]
            if pair == [left, right]:
                merged_tokens.append(left + right)
                index += 2
            else:
                merged_tokens.append(tokens[index])
                index += 1
        tokens = merged_tokens
    return tokens


def make_vocab(tokens):
    vocab = {token: index for index, token in enumerate(SPECIAL)}
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(tokens, vocab):
    unknown = vocab["<unk>"]
    return [vocab.get(token, unknown) for token in tokens]


def decode(ids, inverse_vocab):
    return "".join(
        inverse_vocab[index]
        for index in ids
        if inverse_vocab[index] not in {"<pad>", "<bos>", "<eos>"}
    )


def vocab_dependent_params(vocab_size, d_model, tied=False, bias=True):
    embedding = vocab_size * d_model
    head = 0 if tied else vocab_size * d_model
    if bias:
        head += vocab_size
    return embedding + head


characters = initial_symbols(TEXT)
tokens = ["<bos>"] + apply_merges(characters, MERGES) + ["<eos>"]
vocab = make_vocab(tokens)
inverse_vocab = {index: token for token, index in vocab.items()}
ids = encode(tokens, vocab)
decoded = decode(ids, inverse_vocab)

block_size = 8
starts = [0, len(ids) - block_size - 1]
x = [ids[start:start + block_size] for start in starts]
y = [ids[start + 1:start + block_size + 1] for start in starts]

char_vocab_size = len(set(characters)) + len(SPECIAL)
bpe_content_tokens = apply_merges(characters, MERGES)
bpe_vocab_size = len(vocab)
d_model = 32

assert decoded == TEXT
assert all(row_x[1:] == row_y[:-1] for row_x, row_y in zip(x, y))
print("char_token_count=", len(characters))
print("bpe_content_token_count=", len(bpe_content_tokens))
print("compression_ratio=", round(len(bpe_content_tokens) / len(characters), 3))
print("roundtrip_ok=", decoded == TEXT)
print("x_shape=", (len(x), block_size))
print("shift_ok=", all(row_x[1:] == row_y[:-1] for row_x, row_y in zip(x, y)))
print("char_vocab_params=", vocab_dependent_params(char_vocab_size, d_model))
print("toy_bpe_vocab_params=", vocab_dependent_params(bpe_vocab_size, d_model))
```
这个 toy 实验没有证明手动 merge 规则优于字符级 tokenizer。它只把三个数量关系显式化：合并可以缩短序列，词表大小会改变 embedding/head 参数，next-token 的右移关系并不会因为 tokenizer 类型改变。

### 3.6.7 BPE 迁移时的验证顺序

迁移不应只替换 encode 函数然后立即训练。更稳妥的顺序是：

1. 对一小段固定文本执行 encode/decode，记录 normalization 之后的结果；
2. 检查所有 id 小于 len(tokenizer)；
3. 检查 x/y 的 shift 和最后一个窗口边界；
4. 检查 vocab_size、embedding 行数和 lm head 输出类别一致；
5. 检查 special token、padding side、attention mask 和 loss mask；
6. 用相同 token 预算，而不是相同字符数，比较训练曲线；
7. 从 checkpoint 恢复并验证 tokenizer 复原；
8. 用固定 prompt 和固定解码配置比较输出。

如果第 3 步失败，问题在数据窗口；如果第 4 步失败，问题在模型接口；如果第 8 步出现差异，才有资格讨论 tokenizer 对生成质量的影响。

做个简单估算，看看两方面的代价：若字符序列长 8、BPE 序列长 3，full attention 的 score 元素数会从 $8^2=64$ 降到 $3^2=9$；但若词表从 4 项扩大到 20 项、$D=16$，embedding 参数会从 64 增至 320，未共享的输出头还要再算一份投影。序列变短不一定让总成本下降，实际还要测吞吐、推理延迟和内存。

旧 checkpoint 的 embedding 行绑定旧 token id 的含义。新 BPE 的 id 0 不一定代表旧字符级的 id 0；只换 tokenizer 文件、让张量 shape 对齐，并不能让旧行获得新语义。迁移通常要重新初始化 embedding 和输出头，或设计并评估显式映射。保存模型时，应把 tokenizer artifact 与权重作为一个版本对，记录 revision/hash、special token、padding side、normalization、pre-tokenization 和 merge 配置；只记模型名称或本地路径不足以复原输入输出协议。

最后用代表性数据验证切分收益，而不是只看平均 token/字符比。按语言、代码、空白、标点和稀有字符串分桶，记录压缩比、未知/特殊 token 行为、原文覆盖量、最大长度截断率，以及 embedding/head 参数。roundtrip 样例还应覆盖大小写、空格、换行和 Unicode 组合字符；若 normalization 会合并文本形式，去重和评估也应采用同一约定。这样才能判断较短序列是否换来了真实吞吐收益，同时没有让某类样本更容易被截断。

## 3.7 公平比较与可复现实验：把“小 GPT”变成可复查的实验

### 3.7.1 训练步数不是完整预算

若每个 optimizer update 使用 $B$ 个样本、每个样本包含 $T$ 个 token，训练 $S$ 步，名义处理 token 位置数为：

```math
N_{\mathrm{seen}}=BTS
```

梯度累积 $G$ 个 micro-batch、数据并行设备数为 $R$ 时，可近似写为：

```math
N_{\mathrm{seen}}=B_{\mathrm{micro}}TGRS
```

比较字符级和 BPE 级时，如果都训练 1000 个 step，但每 step 的 token 数不同，loss 差异同时包含 tokenizer 和预算差异。至少要准备两种比较：

| 比较方式 | 固定什么 | 能回答什么 | 局限 |
| --- | --- | --- | --- |
| 固定 optimizer steps | 更新次数 | 相同更新次数下的优化轨迹 | token 预算可能不同 |
| 固定 token budget | 看到的 token 数 | 相同数据暴露量下的效率 | step 数、评估频率不同 |
| 固定 wall-clock | 时间和硬件 | 实际吞吐与成本 | 受系统噪声影响 |
| 固定参数量 | 模型容量 | tokenizer 对预算的取舍 | 词表和 hidden size 需联动 |

报告中应说明使用哪一种公平性定义，而不是把“同样训练 40 步”直接称为公平比较。比如 A 使用 $B=8,T=32,S=100$，处理 $25{,}600$ 个 token 位置；B 使用 $B=16,T=64,S=100$，处理 $102{,}400$ 个，是 A 的 4 倍。随机窗口可能重复，因此这是处理量，不是唯一 token 数。若 B 的验证 loss 更低，它看到的训练位置也更多，不能把差异简单说成 tokenizer 带来的。

### 3.7.2 一个最小消融矩阵

不同对照回答不同问题，最好分开看：

```text
A：字符级，D=32，T=64，固定 optimizer steps
B：BPE，D=32，T=64，和 A 使用相同 steps
C：字符级，D=32，token budget 与 B 相同
D：BPE，调小 D，使 embedding + head 参数接近 A，并使用相同 token budget
```

每个条件至少记录：参数量、词表大小、平均 token 长度、每秒 token 数、训练/验证 loss、PPL、固定 prompt 输出、重复率和 checkpoint 恢复结果。这样才能区分“序列更短带来的 attention 收益”“词表参数增加带来的代价”和“训练预算不同造成的差异”。

### 3.7.3 证据等级

本章实验的结论可以分成三层：

1. **接口事实**：embedding 接收整数 id，交叉熵接收 logits 与类别标签，checkpoint 可以保存 state dict；这些应以目标版本官方文档为准。
2. **局部实现事实**：本章代码的 shift、shape、mask、top-p 候选和 checkpoint roundtrip 断言通过；这只说明教学实现符合这些局部要求。
3. **模型能力事实**：loss、PPL、生成质量和 tokenizer 选择在目标数据上的效果；这需要固定实验、重复运行、分桶评估和适当基线。

把第一层或第二层证据写成第三层结论，是小模型教学中最常见的逻辑跳跃。一个 assert logits.shape == ... 不能证明模型理解语言；一段生成文本像训练语料，也不能证明它具有可迁移的知识。

### 3.7.4 读者可以复现所需的实验材料

一个完整的 mini-GPT 实验目录可以包含：

```text
data_manifest.json       原始数据、hash、切分和 token 统计
tokenizer/               tokenizer artifact 与版本
config.json              模型、优化器、预算和设备配置
train.py                 训练入口
evaluate.py              固定评估集和生成配置
checkpoints/last.pt      最近训练状态
checkpoints/best.pt      验证指标最优状态
logs/train.csv           loss、token budget、耗时
samples/                 带 step 和解码配置的生成样例
report.md                结果、消融、失败原因和剩余不确定性
```

目录本身不是工程质量的证明；它的价值在于让一次训练可以被别人重新定位。尤其是 tokenizer、数据切分和 checkpoint 元数据缺失时，单独保留模型权重往往无法解释输出。

### 3.7.5 一次完整的复盘问题

完成一次训练后，可以围绕以下因果问题复盘：

- 如果 y=x，loss 是否仍然下降？下降说明了什么，不能说明什么？
- 如果关闭 causal mask，训练 loss 会如何变化？这是否构成信息泄漏？
- 如果只加载模型权重而不加载 AdamW 状态，恢复后的曲线是否相同？
- 如果把 tokenizer 换掉但保留旧 checkpoint，哪一个接口首先失配？
- 如果验证 loss 最低的 checkpoint 生成更差，是否可能是 token loss 与任务指标不一致？
- 如果 top-p 输出更丰富，是否同时检查了格式、事实性和重复率？

这些问题把代码操作变成可以检查的判断。小 GPT 的价值不是提供一个缩小版的生产模型，而是让读者能在可控规模上观察数据、目标、优化和解码之间的关系。

### 3.7.6 把观察写成有边界的结论

固定 token budget 与固定时间回答不同问题。假设 A 每秒处理 1,000 个 token，B 每秒处理 700 个，处理 100 万 token 的纯训练时间分别约为 16.7 分钟和 23.8 分钟，还没有计入评估与 checkpoint I/O。若关心固定数据暴露，比较相同 token 数后的质量；若关心成本，比较相同时间或费用内达到的效果，并记录实际处理量。

复现清单应固定代码 revision、数据与切分 hash、tokenizer artifact、模型配置、token budget、batch/梯度累积、优化器与 scheduler、评估窗口、采样参数、软件和硬件。记录 seed 不等于保证逐位相同：非确定算子和并行归约仍可能造成差异。对小差异要重复运行；例如 A 三个 seed 的验证 loss 为 1.02、0.98、1.00，B 为 0.97、1.03、0.99，不能只挑 B 的最低值宣称胜出。三次运行也不一定足够，只能让随机波动更可见。复现研究同样指出 prompt、实现细节和报告方式会影响结果复核：[Biderman 等](https://arxiv.org/abs/2405.14782v3)。

结论可按“条件—观察—解释—边界”组织：在固定 tokenizer revision、相同 100 万训练 token 和同一小模型下，BPE 每步耗时更低；验证 loss 差异仍落在三个 seed 的波动范围内，因此目前只能报告效率观察，不能断言任务质量提高。随后按任务类别列出质量、吞吐、内存和失败样例；若要推广结论，还需要更多数据或独立复跑。这样读者能看出这项实验支持什么，也能看出它还没有证明什么。

交付时保存可定位条件的命令、依赖版本、下载数据/模型的 revision 或 hash、启动参数、工作目录和预期产物。若语料不能公开，应给出获取条件、切分脚本和脱敏统计；失败运行与超预算记录也应保留。第三方在独立环境复跑，才构成独立复核；在同一个进程里重复调用不能替代它。

## 延伸阅读：原始论文与官方接口

本章的实现与结论应结合目标版本文档复核。论文描述的是方法或实验条件，官方文档描述的是接口语义，本文的 toy 数字和生成样例只属于教学构造。

1. Vaswani 等，Attention Is All You Need：<https://arxiv.org/abs/1706.03762>。Transformer、scaled dot-product attention、multi-head attention 和位置表示。
2. PyTorch 2.12 `nn.Embedding`：<https://docs.pytorch.org/docs/2.12/generated/torch.nn.Embedding.html>。整数索引、padding 行和 embedding 查表。
3. PyTorch 2.12 `F.cross_entropy`：<https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.cross_entropy.html>。未归一化 logits、类别索引、形状和 ignore_index。
4. PyTorch 2.12 AdamW：<https://docs.pytorch.org/docs/2.12/generated/torch.optim.AdamW.html>。优化器状态与解耦 weight decay。
5. PyTorch 2.12 `Module.train`/`Module.eval` 与 `torch.no_grad`：<https://docs.pytorch.org/docs/2.12/generated/torch.nn.Module.html>、<https://docs.pytorch.org/docs/2.12/generated/torch.no_grad.html>。训练/评估模式与自动求导是两个独立控制。
6. PyTorch 2.12 `torch.save` 与 `torch.load`：<https://docs.pytorch.org/docs/2.12/generated/torch.save.html>、<https://docs.pytorch.org/docs/2.12/generated/torch.load.html>。状态保存、加载及 `weights_only` 接口。
7. PyTorch 2.12 数据加载：<https://docs.pytorch.org/docs/2.12/data.html>。Dataset、DataLoader 和批处理接口。
8. PyTorch 2.12 `multinomial`：<https://docs.pytorch.org/docs/2.12/generated/torch.multinomial.html>。按权重抽样。
9. Hugging Face Transformers v5.17.0 tokenizer API：<https://huggingface.co/docs/transformers/v5.17.0/main_classes/tokenizer>。预训练 tokenizer 的编码、解码、词表长度和 special token。
10. Hugging Face Transformers v5.17.0 tokenization algorithms：<https://huggingface.co/docs/transformers/v5.17.0/tokenizer_summary>。BPE、Unigram 和 WordPiece 的区别与机制概述。
11. Sennrich 等，Neural Machine Translation of Rare Words with Subword Units：<https://arxiv.org/abs/1508.07909v5>。特定机器翻译设置中的 BPE 子词切分方法。
12. Holtzman 等，The Curious Case of Neural Text Degeneration：<https://arxiv.org/abs/1904.09751v2>。nucleus sampling 与开放式生成退化。
13. Kudo 与 Richardson，SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing：<https://arxiv.org/abs/1808.06226>。SentencePiece 工具支持的子词建模方式及原始文本处理。
