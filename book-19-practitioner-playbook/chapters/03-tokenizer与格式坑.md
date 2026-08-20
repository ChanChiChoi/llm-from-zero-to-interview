# 第三章：Tokenizer、模板与训练格式

很多“模型不听话”的故障，根因不在模型结构，而在输入和监督边界。模型不停生成，可能是 EOS 没有进入训练目标；模型复述用户问题，可能是 user token 没有被 mask；模型忽略 system 指令，可能是训练和推理使用了不同的 chat template；工具调用无法解析，可能是训练数据里的 schema、工具名和线上协议不一致。

这些问题很隐蔽，因为训练过程通常仍然能够运行，loss 也可能正常下降。格式错不是“字符串长得不漂亮”，而是改变了模型看到的序列、角色和目标。一个 token 级别的偏差会在数百万条样本和多轮生成中累积，最后表现为能力退化、成本上升或外部动作错误。

本章把格式处理看成一条有契约的数据流：消息经过模板渲染成为序列，tokenizer 把序列映射为 token ID，collator 生成 attention mask 和 labels，损失函数只在规定位置监督，推理侧再用同一协议解析角色、工具和停止边界。读者将看到 special token、BOS/EOS/PAD、assistant-only loss、chat template、截断、中文/代码/数学 tokenization、多模态占位符和工具调用如何在同一条链上相互影响。

## 3.1 先建立序列协议

一个聊天模型请求至少包含四种不同对象：

1. **语义消息**：`system`、`user`、`assistant`、`tool` 等角色及其内容。
2. **模板序列**：把角色和内容组织成模型训练时使用的文本格式。
3. **token 序列**：tokenizer 将模板序列转换成整数 ID。
4. **监督序列**：labels 和 attention mask 决定哪些 token 参与计算、哪些 token 参与损失。

可以写成：

```math

M\xrightarrow{\phi_v}X\xrightarrow{\tau_v}Z\xrightarrow{\mu_v}(A,Y)
```

其中 `M` 是消息列表，`\phi_v` 是版本为 `v` 的 chat template，`X` 是渲染后的序列，`\tau_v` 是 tokenizer，`Z` 是 token ID，`A` 是 attention mask，`Y` 是 labels，`\mu_v` 是监督 mask 规则。`v` 不能只指模型权重，还可能包括 tokenizer、模板、special token 映射和 collator 版本。

训练和推理只要有一个版本不一致，系统就可能出现协议漂移。比如训练使用 `<|assistant|>` 作为角色边界，推理却使用 `### Response:`；训练让 EOS 参与损失，推理却配置了另一个 `eos_token_id`；训练把 tool result 当作 user 文本，推理则使用独立的 `tool` role。这些变化都会改变模型的条件分布。

一个最小聊天样本可以写成：

```text
messages = [
  {role: system,    content: "你是一个严谨的助手"},
  {role: user,      content: "解释注意力机制"},
  {role: assistant, content: "注意力机制根据查询与键的关系聚合值"}
]
```

对于这个样本，训练时通常希望模型学习 assistant 内容和它的结束边界，而不是学习把 system 和 user 内容重新生成一遍。模型需要“看见”整个上下文，但不代表每个位置都应该进入 loss。

## 3.2 Tokenizer 改变的是计算单位

Tokenizer 把字符、字节、词片段或符号映射为模型词表中的离散 ID。BPE、Unigram、SentencePiece 和 byte-level 方法的切分粒度不同。模型并不直接以“汉字”“单词”或“代码行”为计算单位，而是以 token 序列长度、位置和嵌入为计算单位。

对文本 `x`，可以定义 token 长度：

```math

L_{\tau}(x)=|\tau(x)|
```

同样的字符数在不同 tokenizer 下可能产生不同的 `L_tau`。这会影响上下文容量、attention 计算、KV cache、批处理形状、延迟和价格。一个中文产品如果只用英文样本估算上下文和成本，可能在真实长文场景中产生系统性误判。

Tokenizer 统计不能只看全局平均。至少要按语言、领域和格式切片：

| 切片 | 需要观察的量 | 常见问题 |
| --- | --- | --- |
| 中文 | token/字符、术语切分、混合标点 | 长文预算不足、专业词被切碎 |
| 英文 | token/词、大小写和空格 | 代码混入、缩写处理差 |
| 代码 | token/字符、缩进、符号 | 函数和运算符切分过细 |
| 数学 | token/符号、LaTeX 控制序列 | 公式结构被打散 |
| 多语言 | 每语言 token/字符和保留率 | 高资源语言主导平均值 |
| 工具 JSON | 键名、引号、括号和枚举值 | schema 变长、格式错误 |

Tokenizer 不是越省 token 越好。合并过度可能损失边界和组合能力，切分过细会增加序列长度；最终要在目标任务的准确率、可解释性、上下文长度和成本之间比较。

## 3.3 Special token 是协议字段

`BOS`、`EOS`、`PAD`、`UNK`、role token、tool token 和多模态 placeholder 都是协议的一部分。它们不能仅按“词表中几个保留 ID”处理，因为每个 ID 在训练、生成、padding、停止和跨模态对齐中承担不同语义。

常见字段包括：

- `BOS`：序列开始，是否需要由模板或 tokenizer 添加必须明确；
- `EOS`：回答或序列结束，训练目标和生成停止条件必须一致；
- `PAD`：批处理对齐用的填充，不应作为真实内容参与 loss；
- `UNK`：无法映射的未知符号，出现比例过高说明词表或输入处理有问题；
- role token：标记 system、user、assistant、tool 的边界；
- tool token：标记调用名称、参数、结果或调用结束；
- image/audio/video token：承载跨模态特征插入位置。

检查 special token 时至少要验证：ID 是否唯一，训练和推理是否使用同一个映射，decode 是否能恢复边界，padding 和停止配置是否与模板一致，模型词表大小和 embedding 行数是否匹配。

如果 tokenizer 的 `pad_token_id` 与 `eos_token_id` 被错误地设置为同一个 ID，短序列 padding 可能被解释成结束，生成可能提前停止；如果 `PAD` 参与 loss，模型会学到批处理长度模式；如果 EOS 只出现在文本而没有出现在 labels，模型看到了结束符却没有被监督生成它。

## 3.4 BOS、EOS 和 PAD 的不同责任

### BOS

有些模型由模板插入 BOS，有些模型由 tokenizer 的 `add_special_tokens` 插入。两者同时开启会产生重复 BOS，全部关闭又可能造成训练和推理边界不同。必须在渲染前后分别打印 token 序列，不要只看 decode 后的文本，因为某些 special token 在 decode 时会被隐藏。

### EOS

EOS 负责告诉模型当前回答可以结束，也影响生成器何时停止。下面几种情况都会让模型不停生成：

1. 训练样本没有 assistant EOS。
2. EOS 存在于 input，却在 labels 中被设为 `-100`。
3. 推理配置的 `eos_token_id` 与训练使用的 ID 不同。
4. 多轮模板把下一轮 user 直接拼在 assistant 后面，没有结束边界。
5. stop string 与 token 级 EOS 不一致，文本层停止没有覆盖所有分词形式。

### PAD

PAD 只用于把不同长度的样本放入同一个 batch。常见约定是 `attention_mask=0` 且 `labels=-100`，但实际框架可能采用不同的数据结构。关键不是背某个数字，而是验证 padding 位置不会参与 attention 的有效上下文，也不会参与训练损失。

对某个 batch，若监督 mask 为 `m_t`，单个 token 损失为 `ell_t`，有效监督损失为：

```math

L=\frac{\sum_t m_t\ell_t}{\sum_t m_t}
```

要求 `\sum_t m_t>0`。如果一个 batch 全部是 padding、过滤后没有 assistant token 或角色解析失败，损失没有定义。不能把分母替换成一个人为的小数后继续训练，否则会把数据错误变成一个看似正常的数值。

## 3.5 Chat template 是模型协议，不是普通 prompt

Chat template 把消息列表变成模型训练和推理时看到的具体序列。它决定角色标签、换行、special token、assistant 起始位置、工具调用语法和结束边界。两个模板都能读起来“像对话”，并不代表它们对模型等价。

例如训练序列可能是：

```text
<bos><system>你是助手<user>解释 Transformer<assistant>Transformer 是...<eos>
```

推理序列却是：

```text
### System:
你是助手
### User:
解释 Transformer
### Assistant:
```

人类能够理解它们表达了相同语义，模型却依赖 token、角色和位置的统计模式。训练时使用一种格式、推理时使用另一种格式，会导致不听 system、生成模板残留、多轮角色错乱或工具调用字段漂移。

模板兼容性不一定要求逐字符相同，但以下语义必须对齐：

1. system、user、assistant、tool 的角色边界；
2. assistant 生成起点；
3. EOS 和每轮结束位置；
4. tool call 名称、参数和 observation 的结构；
5. 多模态 placeholder 与消息位置；
6. 是否添加 BOS、是否自动截断和是否保留空消息。

模板变更应和模型版本一起管理。一个旧 checkpoint 加载成功，不代表它能理解新模板；模型权重 shape 不变也不意味着协议兼容。模板版本应进入评估、trace 和 release manifest。

## 3.6 Assistant-only loss：看见不等于被监督

监督微调通常希望模型根据 system 和 user 上下文生成 assistant 答案，但不希望它把 user 问题也当作需要模仿的目标。于是需要构造 labels：assistant 内容和必要的 EOS 保留真实 token ID，system、user、控制 token 和 PAD 设为忽略值，常见实现是 `-100`。

设序列中第 `t` 个位置的角色为 `r_t`，labels 为 `y_t`。assistant 覆盖率可以写成：

```math

C_{\mathrm{assist}}=
\frac{\sum_t\mathbf{1}[r_t=\mathrm{assistant}\land y_t\ne -100]}
{\sum_t\mathbf{1}[r_t=\mathrm{assistant}]}
```

只有分母大于 0 时这个比例才有定义。没有 assistant 目标的样本可能是纯 user 输入、工具结果或解析失败，不能直接当成覆盖率 0；应该根据任务标记为 `not_applicable` 或 `unknown`。

prompt 泄漏率为：

```math

R_{\mathrm{prompt}}=
\frac{\sum_t\mathbf{1}[r_t\in P\land y_t\ne -100]}
{\sum_t\mathbf{1}[r_t\in P]}
```

其中 `P` 是 system、user 和控制角色集合。对于标准 assistant-only SFT，`R_prompt` 应为 0；如果任务有意训练完整序列，例如某些语言建模或格式续写任务，则应单独定义目标，不能把所有训练都套用 assistant-only 规则。

多轮对话还要决定是否监督每一轮 assistant。只监督最后一轮可以节省目标 token，但会让早期 assistant 行为没有训练信号；监督所有 assistant 轮次则需要确保每一轮 role、EOS 和上下文都正确。选择哪一种，应与训练目标和数据构成一致。

## 3.7 多轮、工具和结构化输出的边界

多轮对话的格式错误经常发生在单轮样本无法暴露的地方：上一轮 assistant 没有 EOS，tool result 被当成 user，下一轮 assistant 起始 token 丢失，历史被截断后 role 不闭合。排查时要把一条完整多轮样本 decode 出来，并在每个 role 边界标出 labels 和 attention mask。

工具调用再增加三类协议字段：工具名、参数 schema 和结果 observation。训练数据应该明确模型是在“请求工具”，还是在“看到工具结果后回答”；二者不能只靠自然语言中的“调用一下”区分。

一个工具调用生命周期可以写成：

```text
assistant -> tool_call(name, arguments)
executor -> tool_result(call_id, content, status)
assistant -> final_answer 或 next_tool_call
```

必须验证：

- `name` 是否存在于当前工具注册表；
- 参数是否符合当前 schema，而不是旧版本 schema；
- `call_id` 是否唯一且与 observation 对应；
- tool result 是否被当作不可信数据处理；
- 工具调用和自然语言是否有清晰边界；
- 生成失败时是否允许重试以及如何避免重复副作用。

结构化输出也不能只检查 JSON 是否可解析。JSON 可以语法正确，却缺字段、类型错误、枚举值非法或违反业务约束。协议检查应分成语法、schema、语义和权限四层。

## 3.8 截断是任务策略，不是简单切掉尾部

上下文超过最大长度时，系统必须决定删掉什么。简单保留前 `L_max` 个 token 可能截掉最新问题、assistant 起始位置、EOS、工具 schema 或图像 placeholder；简单保留后缀又可能丢掉 system 约束和必要的历史证据。

截断策略应先写优先级，再实现：

1. 保留不可替代的系统约束和安全策略。
2. 保留当前用户问题及其必要附件。
3. 保留工具 schema 和当前动作所需状态。
4. 保留能够支撑回答的检索证据。
5. 对较旧对话做摘要或按轮次删除。
6. 保留 assistant 生成起点和合法结束边界。

多模态输入还要把文本 token 与视觉、音频或视频特征的占位关系作为不可拆分单元。截掉 placeholder 而保留特征，或保留 placeholder 而截掉对应特征，都会让模型输入错位。

截断前后应记录：原始长度、保留长度、删除区间、被删除的角色、是否删除了证据、是否保留了 EOS、是否改变了 labels。不能只记录最终 token 数，否则无法解释某类长输入为什么突然失败。

## 3.9 中文、代码和数学的 token 预算

中文字符、代码符号和数学控制序列的 token 化效率差异很大。中文专业词可能被切成多个单字，代码的缩进、括号和运算符可能占用大量 token，LaTeX 的反斜杠命令和 Unicode 数学符号可能形成复杂边界。

对切片 `b`，可以报告 token/字符压缩比：

```math

\rho_b=\frac{\sum_{x\in b}|\tau(x)|}{\sum_{x\in b}|x|}
```

要求字符或字节分母大于 0，并固定字符定义。`rho_b` 越小不一定越好：过度合并可能影响泛化，过大则减少有效上下文容量和增加成本。更重要的是比较同一 tokenizer 的切片差异，并将差异连接到任务准确率和截断率。

建议建立小型领域词表：中文产品术语、代码 API、数学符号、工具字段、法律条款和多语言实体。逐项查看 token ID、decode round trip 和长度。一个 tokenizer 更新后，即使词表大小只增加少量，也可能改变大量领域序列的切分和历史 checkpoint 行为。

## 3.10 多模态 placeholder 与特征对齐

多模态模型通常在文本序列中插入 `<image>`、音频或视频 placeholder，再由视觉或音频编码器产生特征。placeholder 数量、顺序、位置和特征数量必须满足模型协议。

若第 `i` 条样本的 placeholder 数为 `n_p(i)`，对应特征组数为 `n_f(i)`，在确实包含媒体的样本集合 `M` 非空时，一致率为：

```math

C_{\mathrm{media}}=\frac{1}{|M|}\sum_{i\in M}\mathbf{1}[n_p(i)=n_f(i)]
```

如果评估集没有任何多模态样本，`C_media` 是 `not_applicable`，不能当作 1；如果样本应该有媒体但特征提取失败，应是 `unknown` 或失败状态。多图任务还要检查顺序，例如第一张图的特征不能被放到第二个 placeholder。

排查时同时打印：

- 原始消息中的媒体引用；
- 渲染后的 placeholder 位置；
- token ID 和媒体索引；
- 编码器输出的特征组数和形状；
- 截断前后的媒体映射；
- labels 是否错误地监督了视觉占位符。

“模型像没看到图”可能是图像质量、编码器、占位符、特征拼接、attention mask 或任务标注中的任意一层，不能直接归因于视觉能力。

## 3.11 版本和 checkpoint 兼容性

格式协议的版本需要和模型权重一起记录。一个可复现的训练 manifest 至少应包括：

```text
model_revision
tokenizer_revision
vocab_size
special_token_map
chat_template_revision
collator_revision
label_mask_policy
padding_side
truncation_policy
tool_schema_revision
multimodal_placeholder_revision
```

改变 tokenizer 可能改变 embedding 行数和输出层；改变 special token ID 可能让旧 labels 指向错误语义；改变模板可能让旧 checkpoint 接收不同的 role 序列；改变 padding side 可能影响位置编码和生成对齐。加载 checkpoint 成功只说明张量形状兼容，不说明输入协议兼容。

新旧版本比较时，要固定一批原始消息，分别记录渲染文本、token ID、special token 位置、labels 和生成结果。不能只比较最终回答，因为中间序列的变化可能正是问题来源。

## 3.12 格式故障的最小复现

格式问题应先用一条短样本复现，不要一开始在大训练和长对话中猜测。推荐使用包含 system、user、assistant、EOS、PAD、工具或媒体的最小样本，逐层保存：

1. 原始 messages。
2. 模板渲染文本。
3. token ID 和 decode 结果。
4. 每个 token 的 role 标签。
5. attention mask。
6. labels 和 `-100` 位置。
7. 截断前后的区间。
8. 推理侧的 prompt 和 stop 配置。

若这个样本在训练 collator 和线上推理路径中分别渲染，应该逐字段对比，而不是只比较字符串长度。最小复现还要保留 tokenizer 和模板版本，否则后续重新运行时可能已经无法复现。

一个常见的“复述 user”事件可以这样定位：先看生成；再 decode 训练样本；发现 role 正常；继续看 labels；发现 user 位置有真实 token；修复 collator；在一个小数据集上重新训练；最后用多轮、长输入和工具样本回归。每一步都应有一个能支持或否定假设的观察结果。

## 3.13 格式指标的定义域

### 3.13.1 模板匹配率

在有 `N>0` 条固定消息样本时，若训练和推理协议的语义比较结果为 `I_i`：

```math

C_{\mathrm{template}}=\frac{\sum_{i=1}^{N}I_i}{N}
```

`I_i` 的判定不能只做字符串相等，应包括 role、assistant 起点、EOS、工具和媒体边界。没有样本时结果未定义。

### 3.13.2 Assistant 覆盖率和 prompt 泄漏率

```math

C_{\mathrm{assist}}=\frac{N_{\mathrm{assistant\ target}}}{N_{\mathrm{assistant\ positions}}}
```

只有 assistant position 非空时有定义。prompt 泄漏率的分母是 system/user/control 的位置数；纯 assistant continuation 任务不应套用普通对话的分母。

### 3.13.3 PAD 泄漏率

```math

R_{\mathrm{pad}}=\frac{N_{\mathrm{pad\ positions\ with\ loss}}}{N_{\mathrm{pad\ positions}}}
```

没有 PAD 位置时是 `not_applicable`，不是 0。若存在 PAD 但 labels 没有忽略，才是实际泄漏。

### 3.13.4 EOS 覆盖率

对确实要求 EOS 的样本集合 `E`，且 `|E|>0`：

```math

C_{\mathrm{eos}}=\frac{N_{\mathrm{eos\ present\ and\ supervised}}}{|E|}
```

工具中间消息或不需要终止符的任务不应强行计入 `E`。

### 3.13.5 有效损失分母

```math

N_{\mathrm{loss}}=\sum_t\mathbf{1}[y_t\ne -100]
```

若 `N_loss=0`，该样本或 batch 的平均损失未定义，应在数据处理阶段阻断或隔离。

这些指标不能简单相乘成一个格式总分。模板不匹配、PAD 泄漏和媒体错位的后果不同，应该保留具体失败项和触发样本。

## 3.14 四个典型格式事故

### 事故一：SFT 后复述用户

训练样本 decode 看起来正常，生成却先复述 user 问题。检查 labels 发现 system 和 user 位置没有设为 `-100`。模型被训练去预测整个对话序列，因此复述是训练目标的合理结果。修复后要比较 prompt loss 泄漏率、assistant 覆盖率和固定生成样例。

### 事故二：模型生成到最大长度

输出没有 EOS，或者推理使用的 `eos_token_id` 与训练不一致。检查训练样本的 EOS 是否存在、是否参与 loss，确认模板是否在 assistant 末尾加入结束边界，再检查 generation 配置。只增加 `max_new_tokens` 会掩盖问题并增加成本。

### 事故三：工具 JSON 语法正确但动作错误

模型生成了合法 JSON，却使用了旧字段名、非法枚举值或错误工具名。解析通过不代表 schema 和语义通过。修复需要绑定工具 schema 版本、增加参数验证、保存 call ID 和拒绝原因，并将自然语言、工具请求和工具结果分开训练和评估。

### 事故四：模型像没有看到图

检查发现消息里有两个 `<image>`，视觉编码器只返回一组特征；或者截断保留了 placeholder，却删除了对应图片。此时模型回答错误不是纯粹的视觉识别失败，而是输入对齐失败。修复后要用多图顺序、缺图、空图和截断边界回归。

## 3.15 一个可运行的格式审计示例

下面的 Python 示例使用 toy tokenizer，不代表任何具体模型的内部实现。它故意构造模板不一致、错误 labels、截断丢 EOS 和媒体数量不匹配，同时验证正确的 assistant-only mask 和空分母边界。

```python
import re
from math import isfinite
from typing import Dict, List, Optional, Tuple


SPECIAL_IDS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<system>": 3,
    "<user>": 4,
    "<assistant>": 5,
    "<image>": 6,
    "<tool>": 7,
}
VOCAB = dict(SPECIAL_IDS)


def strict_int(value: object) -> bool:
    return type(value) is int


def safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    if not strict_int(numerator) or not strict_int(denominator):
        raise ValueError("ratio arguments must be integers")
    if numerator < 0 or denominator < 0:
        raise ValueError("ratio arguments must be non-negative")
    if numerator > denominator:
        raise ValueError("numerator cannot exceed denominator")
    if denominator == 0:
        return None
    return round(numerator / denominator, 3)


def validate_special_ids(mapping: Dict[str, int]) -> None:
    if not mapping:
        raise ValueError("special token map must not be empty")
    values = list(mapping.values())
    if any(not strict_int(value) or value < 0 for value in values):
        raise ValueError("special token ids must be non-negative integers")
    if "<pad>" in mapping and "<eos>" in mapping and mapping["<pad>"] == mapping["<eos>"]:
        raise ValueError("PAD and EOS must have different ids")
    if len(values) != len(set(values)):
        raise ValueError("special token ids must be unique")


def pieces(text: str) -> List[str]:
    return re.findall(r"<[^>]+>|[A-Za-z_]+|\d+|[\u4e00-\u9fff]|[^\s]", text.lower())


def token_id(piece: str) -> int:
    if piece not in VOCAB:
        VOCAB[piece] = 100 + len(VOCAB) - len(SPECIAL_IDS)
    return VOCAB[piece]


def encode(text: str) -> List[int]:
    return [token_id(piece) for piece in pieces(text)]


def render_train(messages: List[Dict[str, str]], add_eos: bool = True) -> List[Tuple[str, str]]:
    sequence: List[Tuple[str, str]] = [("<bos>", "control")]
    for message in messages:
        role = message["role"]
        tag = f"<{role}>"
        if tag not in SPECIAL_IDS:
            raise ValueError(f"unknown role: {role}")
        sequence.append((tag, "control"))
        for piece in pieces(message["content"]):
            sequence.append((piece, role))
        if role == "assistant" and add_eos:
            sequence.append(("<eos>", "assistant_eos"))
    return sequence


def render_infer_mismatch(messages: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    sequence: List[Tuple[str, str]] = []
    for message in messages:
        sequence.append(("###", "control"))
        sequence.append((message["role"] + ":", "control"))
        for piece in pieces(message["content"]):
            sequence.append((piece, message["role"]))
    return sequence


def labels_assistant_only(sequence: List[Tuple[str, str]], padded_len: int) -> List[int]:
    if padded_len < len(sequence):
        raise ValueError("padded_len cannot be shorter than sequence")
    labels = [
        token_id(piece) if role in {"assistant", "assistant_eos"} else -100
        for piece, role in sequence
    ]
    labels.extend([-100] * (padded_len - len(labels)))
    return labels


def labels_all_tokens(sequence: List[Tuple[str, str]], padded_len: int) -> List[int]:
    if padded_len < len(sequence):
        raise ValueError("padded_len cannot be shorter than sequence")
    labels = [token_id(piece) for piece, _ in sequence]
    labels.extend([SPECIAL_IDS["<pad>"]] * (padded_len - len(labels)))
    return labels


def ratio_from_positions(labels: List[int], positions: List[int]) -> Optional[float]:
    numerator = sum(labels[index] != -100 for index in positions)
    return safe_ratio(numerator, len(positions))


validate_special_ids(SPECIAL_IDS)
messages = [
    {"role": "system", "content": "只回答助手内容"},
    {"role": "user", "content": "解释 Transformer"},
    {"role": "assistant", "content": "Transformer 是 注意力 和 MLP 组成"},
]

train_sequence = render_train(messages, add_eos=True)
infer_sequence = render_infer_mismatch(messages)
input_ids = [token_id(piece) for piece, _ in train_sequence]
padded_len = len(input_ids) + 4
correct_labels = labels_assistant_only(train_sequence, padded_len)
bad_labels = labels_all_tokens(train_sequence, padded_len)

prompt_positions = [
    index
    for index, (_, role) in enumerate(train_sequence)
    if role in {"system", "user", "control"}
]
assistant_positions = [
    index
    for index, (_, role) in enumerate(train_sequence)
    if role in {"assistant", "assistant_eos"}
]
pad_positions = list(range(len(train_sequence), padded_len))

samples = {
    "en": "the quick brown fox explains attention",
    "zh": "中文客服需要覆盖错别字",
    "code": "def add(a, b): return a+b",
    "math": "L = - log p(y | x)",
}
token_counts = {name: len(encode(text)) for name, text in samples.items()}
compression = {
    name: round(token_counts[name] / len(text), 3)
    for name, text in samples.items()
    if text
}

truncated = train_sequence[:18]
critical_missing = (
    not any(piece == "<assistant>" for piece, _ in truncated)
    or not any(piece == "<eos>" for piece, _ in truncated)
)
placeholder_count = pieces("<image> <image> 请比较两张图").count("<image>")
image_feature_count = 1

label_summary = {
    "seq_len": len(train_sequence),
    "padded_len": padded_len,
    "assistant_coverage": ratio_from_positions(correct_labels, assistant_positions),
    "prompt_loss_rate": ratio_from_positions(correct_labels, prompt_positions),
    "bad_prompt_loss_rate": ratio_from_positions(bad_labels, prompt_positions),
    "bad_pad_loss_rate": ratio_from_positions(bad_labels, pad_positions),
    "eos_label_ok": correct_labels[
        [piece for piece, _ in train_sequence].index("<eos>")
    ] == SPECIAL_IDS["<eos>"],
}

template_match = [piece for piece, _ in train_sequence] == [
    piece for piece, _ in infer_sequence
]
checks = {
    "special_ids_ok": len(set(SPECIAL_IDS.values())) == len(SPECIAL_IDS),
    "template_match": template_match,
    "assistant_mask_ok": (
        label_summary["assistant_coverage"] == 1.0
        and label_summary["prompt_loss_rate"] == 0.0
    ),
    "bad_mask_detected": (
        label_summary["bad_prompt_loss_rate"] is not None
        and label_summary["bad_prompt_loss_rate"] > 0.0
        and label_summary["bad_pad_loss_rate"] is not None
        and label_summary["bad_pad_loss_rate"] > 0.0
    ),
    "eos_supervised": label_summary["eos_label_ok"],
    "truncation_preserves_boundaries": not critical_missing,
    "media_alignment": placeholder_count == image_feature_count,
    "token_budget_observed": compression["zh"] <= 1.5 and compression["code"] <= 0.8,
}

print("special_ids=", SPECIAL_IDS)
print("token_counts=", token_counts)
print("compression=", compression)
print("label_summary=", label_summary)
print("template_match=", template_match)
print("train_prefix=", " ".join(piece for piece, _ in train_sequence[:12]))
print("infer_prefix=", " ".join(piece for piece, _ in infer_sequence[:12]))
print("truncation=", {"max_len": 18, "critical_missing": critical_missing})
print(
    "multimodal=",
    {"placeholders": placeholder_count, "image_features": image_feature_count},
)
print("checks=", checks)
print("all_checks_pass=", all(checks.values()))

assert checks["special_ids_ok"]
assert checks["assistant_mask_ok"]
assert checks["bad_mask_detected"]
assert checks["eos_supervised"]
assert checks["template_match"] is False
assert checks["truncation_preserves_boundaries"] is False
assert checks["media_alignment"] is False
assert safe_ratio(0, 0) is None

try:
    safe_ratio(1, 0)
except ValueError:
    pass
else:
    raise AssertionError("positive numerator with zero denominator is invalid")

try:
    validate_special_ids({"<pad>": 0, "<eos>": 0})
except ValueError as exc:
    assert "different ids" in str(exc)
else:
    raise AssertionError("PAD and EOS collision must be rejected")

empty_assistant = [("<bos>", "control"), ("<user>", "control"), ("hello", "user")]
empty_labels = labels_assistant_only(empty_assistant, len(empty_assistant))
assert ratio_from_positions(empty_labels, []) is None
```

示例中的 `all_checks_pass=False` 是有意的：它证明审计器能够发现模板不一致、截断边界丢失和媒体数量错位；同时 `assistant_mask_ok=True` 说明正确的 collator 已经避免 prompt 和 PAD 参与损失。真实项目不应因为某个局部检查通过就把整个格式链路标成可用，而应保留失败项、样本 ID、版本和修复动作。

## 3.16 复盘和回归

格式事故的复盘需要保存“模型当时实际看到了什么”，而不是只写“模板错了”。一份可复现记录应包含：

```text
原始 messages
模板版本和渲染文本
tokenizer 版本、词表大小和 special token map
token ids 与 decode 结果
attention mask、labels 和 role spans
截断前后区间
模型和 checkpoint 版本
推理 stop 配置
工具或多模态 schema 版本
失败样本和修复后的对照输出
```

回归集至少覆盖：单轮、多轮、空消息、长上下文、只读回答、工具调用、工具结果、中文、代码、数学、多图、缺图、PAD 较多的 batch 和需要 EOS 的生成任务。每个样本都要声明哪些检查适用，避免把没有媒体的文本任务错误地计入媒体一致率，也避免把没有 assistant target 的输入误判为 mask 失败。

格式回归通过率只有在测试集非空且每条测试有明确判定时才有定义。没有测试记录不是全通过；没有实际媒体样本不是媒体一致率为 1；没有 assistant 监督位置不是 assistant 覆盖率为 0。保持 `unknown` 和 `not_applicable` 的区别，才能知道下一步是修代码、补测试还是调整任务定义。

## 3.17 资料与证据边界

本章的机制和工程方法参考以下资料，公开接口和论文结论不能替代目标 checkpoint 的实测：

- BPE 和 SentencePiece 论文用于说明子词和无监督分词的基本机制；论文中的词表和语言结果不自动适用于当前 tokenizer。
- Hugging Face Transformers 的 tokenizer、special token、generation 和 chat template 文档用于说明公开 API 的字段和模板调用方式；具体模型仍需查看自己的 tokenizer 配置和模型卡。
- TRL 的 SFT 文档用于说明 assistant-only loss、数据整理和训练字段边界；框架默认行为可能随版本变化，必须记录 revision 并用样本回放验证。
- OpenAI Function Calling / Structured Outputs 文档和公开工具协议资料用于说明结构化调用的接口边界；JSON 可解析不等于业务动作安全。
- LLaVA、BLIP-2 等多模态论文用于说明文本 token 与视觉特征连接的常见架构；不同实现的 placeholder 和 projector 协议不能互相假定兼容。

资料只能支持公开的 tokenization、模板、字段和方法说明。它不能证明训练数据的 labels 一定正确、当前 serving 配置一定使用了同一模板，也不能从 API 成功响应推断模型内部如何处理隐藏状态。真正的结论必须来自固定样本、版本 manifest、token 序列、labels 和目标系统的回归结果。

## 3.18 本章小结

Tokenizer 和格式不是训练前的琐碎准备，而是模型协议。special token 决定边界和停止，chat template 决定角色语义，labels 决定模型被要求学习什么，attention mask 决定模型能看到什么，截断策略决定哪些信息留下，多模态 placeholder 决定文本和特征如何对应，工具 schema 决定结构化动作如何落地。

排查这类问题时，最有效的方法是从一条原始消息开始，逐层保存渲染文本、token ID、decode、role、mask、labels、截断区间和推理配置。先证明协议一致，再讨论模型能力；先用最小样本和边界断言复现，再扩展到真实数据。只要训练和推理看到的是同一个协议，格式问题就能从“模型很怪”变成可定位、可修复、可回归的工程问题。
