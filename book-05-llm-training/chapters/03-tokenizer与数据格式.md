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

### 3.1.3 Tokenizer 的不变量

一个训练 tokenizer 至少要满足：

1. 训练和推理使用相同的词表与规则。
2. 编码结果 id 都在 embedding 和输出层支持的范围内。
3. special token 的角色在数据、模型和服务端一致。
4. 截断和 padding 不破坏目标字段与文档边界。
5. 解码后的文本和结构化事件符合预期。

这些是不变量，不是某个框架的固定实现。换 tokenizer 往往不是换预处理器，而是改变模型输入空间；已经训练完成的模型不能随意接入另一个词表。

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

## 3.5 Byte-level tokenizer：覆盖率优先

Byte-level tokenizer 从字节层面表示文本，任意非空输入都可以退化为字节序列：

~~~math
x\ne\varnothing
\Longrightarrow
|T_{\mathrm{byte}}(x)|\ge 1
~~~

在可逆实现中，解码还应恢复原始字节或规范化后的字符串。它几乎消除了词表层面的 OOV，但没有保证序列短、语义边界合理或模型已经学会某个字符的使用方式。

它特别适合脏输入、代码、emoji、混合语言和特殊符号；代价是某些语言或字符可能产生更多 token。覆盖率和效率是两个不同指标：byte-level 解决“能不能编码”，不自动解决“编码是否经济”。

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

### 3.6.4 不能只看平均压缩率

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

## 3.7 Special token：把普通序列变成结构化协议

### 3.7.1 三类控制符号

| 类型 | 可能的例子 | 作用 |
| --- | --- | --- |
| 序列控制 | BOS、EOS、PAD、UNK | 开始、结束、占位和未知输入 |
| 角色/事件标记 | system、user、assistant、tool | 对话角色、工具调用和结果边界 |
| 模态占位 | image、audio、video | 在语言序列中表示外部模态位置 |

这些名字不是跨模型标准。一个模型可能把 `<assistant>` 注册为单独 vocab id，另一个模型可能把一段字符串模板展开成多个普通 token；一个 `<image>` 可能只占一个文本位置，也可能对应 vision encoder 产生的多段连续 embedding。

查看具体模型时，应确认 token 字符串、id、`special` 属性、解码行为、模板插入方式和停止条件。

### 3.7.2 EOS 的训练与推理语义

EOS 既是训练文本的边界，也是生成停止协议的一部分。训练中如果某些回答有 EOS、某些回答没有，模型可能学不到稳定的终止行为；推理端如果把 EOS id 配错，模型可能不停生成或过早结束。

多轮对话还要区分 assistant 回答的 EOS、整段对话的结束和工具事件的结束。停止条件不应只靠字符串匹配，因为字节级 token、模板和流式解码可能把同一字符串拆成不同序列。

### 3.7.3 PAD、attention mask 和 loss mask

padding 是为了把不同长度样本放进规则张量。定义：

~~~math
a_t=\mathbf{1}[z_t\ne z_{\mathrm{pad}}]
~~~

`a_t` 只说明位置是否为真实 token。causal LM 还需要因果 mask，保证当前位置不能看未来位置；如果是 packed 多文档训练，还可能需要 document mask。三者不能混为一个“attention mask”概念。

loss 也要忽略 PAD。很多 PyTorch 训练代码用 `ignore_index=-100` 表示不计算该位置，但 `-100` 是实现约定，不是模型词表中的真实 token：

~~~math
y_t=
\begin{cases}
z_t,&a_t=1\\
-100,&a_t=0
\end{cases}
~~~

如果 PAD 进入 loss，模型会浪费容量学习填充；如果 PAD 进入 attention，样本之间可能出现虚假上下文。

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

### 3.10.2 Packing 利用率

固定 block 总位置数为 `M`，真实有效 token 数为 `V`，可以定义利用率：

~~~math
\eta_{\mathrm{pack}}=\frac{V}{M}
~~~

这里 `M>0` 且 `0\le V\le M`；没有形成 block 的空输入不属于“利用率为 100%”的样本。利用率高只说明 padding 少，不说明边界正确。应同时记录每个文档的起止位置、EOS 数、attention mask、padding 比例和被截断样本数。

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

### 3.12.2 组内候选和验证器

GRPO、RLVR 或其他强化训练可能保存一个 prompt 的多个候选、组内 reward、验证器输出、环境状态和工具轨迹。可把一组数据抽象为：

~~~math
G_i=\left(x_i,\{y_{i,j},r_{i,j},v_{i,j}\}_{j=1}^{n_i}\right)
~~~

`y` 是候选，`r` 是奖励或相对分数，`v` 是验证器证据。奖励来自字符串匹配、代码执行、数学检查还是人工偏好，会影响训练信号的含义，必须记录来源。

### 3.12.3 长度和偏好偏差

如果 chosen 几乎总比 rejected 长，偏好优化可能学习冗长；如果安全回答总是拒绝，模型可能过度拒答；如果 judge 对格式有偏好，模型可能优化表面结构。偏好数据要做长度、语言、领域、风险等级和错误类型的分层统计。

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

### 3.15.3 旧数据和新词表

旧数据如果保存的是 token id，扩词表或重排词表后不能直接复用；如果保存的是原始文本，可以在新 tokenizer 下重新编码，但要重新计算 token 数、packing、训练步数和评估基线。数据版本应记录 tokenizer hash，避免同一个字段名下混入不同编码空间。

## 3.16 一个最小的数据格式审计器

下面的 toy tokenizer 不复刻真实 BPE，只用有限片段帮助读者观察五件事：不同文本的压缩率、chat template 的角色边界、assistant-only label mask、预训练 packing 和扩词表后的 embedding 行数。真实系统还要处理 Unicode、byte fallback、offset mapping、批量 padding 和模型特定模板。

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


def fallback_piece(char):
    return f"<U+{ord(char):04X}>"


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
            matched = fallback_piece(text[index])
        pieces.append(matched)
        index += len(matched) if not matched.startswith("<U+") else 1
    return pieces


def encode_pieces(pieces):
    if not isinstance(pieces, list) or any(
        not isinstance(piece, str) or not piece for piece in pieces
    ):
        raise ValueError("pieces must be non-empty strings")
    return [add_token(piece) for piece in pieces]


def decode_ids(ids):
    if not isinstance(ids, list) or any(
        not isinstance(index, int) or isinstance(index, bool) or index not in id_to_token
        for index in ids
    ):
        raise ValueError("ids must be known integer token ids")
    return [id_to_token[index] for index in ids]


def serialize_chat(messages):
    allowed_roles = {"system", "user", "assistant"}
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    pieces = ["<bos>"]
    assistant_mask = [0]
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be a mapping")
        if message.get("role") not in allowed_roles:
            raise ValueError("message role is not registered in this template")
        if not isinstance(message.get("content"), str):
            raise ValueError("message content must be a string")
        role_piece = f"<{message['role']}>"
        pieces.append(role_piece)
        assistant_mask.append(0)

        content_pieces = tokenize_text(message["content"])
        pieces.extend(content_pieces)
        assistant_mask.extend(
            [1 if message["role"] == "assistant" else 0]
            * len(content_pieces)
        )

        if message["role"] == "assistant":
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
    masks = []
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
label_tokens = [
    id_to_token[token_id] if token_id != IGNORE_INDEX else "IGN"
    for token_id in labels
]
first_assistant_pos = assistant_mask.index(1)
prompt_label_count = sum(
    1 for value in labels[:first_assistant_pos] if value != IGNORE_INDEX
)
assistant_label_count = sum(value != IGNORE_INDEX for value in labels)

blocks, attention_masks = pack_documents(list(texts.values()), block_size=16)

old_vocab_size = len(vocab)
for token in ["<tool_call>", "<tool_result>"]:
    add_token(token)
new_vocab_size = len(vocab)

packed_eos_count = sum(block.count(vocab["<eos>"]) for block in blocks)
checks = {
    "specials": all(token in vocab for token in SPECIAL_TOKENS),
    "assistant_only_labels": prompt_label_count == 0 and assistant_label_count > 0,
    "pad_mask": all(
        (token != vocab["<pad>"]) == bool(mask)
        for block, mask_row in zip(blocks, attention_masks)
        for token, mask in zip(block, mask_row)
    ),
    "packing_shape": all(len(block) == 16 for block in blocks),
    "eos_boundary": packed_eos_count == len(texts),
    "vocab_resize": new_vocab_size == old_vocab_size + 2,
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

示例中 `assistant_label_count` 与 `prompt_label_count` 分开统计，说明输入可以包含完整对话而监督只落在 assistant 区域；`packed_eos_count` 说明文档边界被保留；`vocab_resize` 说明增加 tool token 后 embedding 行数必须同步增加。toy 明确拒绝未知角色、空 SFT 监督、非法 token id、空文档和超长文档，因为这些情形需要显式模板或截断策略，不能由默认行为悄悄吞掉。它不代表真实模型的模板或 token 数，读者应把同样的检查思路迁移到目标模型的官方 tokenizer 和数据处理器上。

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
| 长度分位数 | 长尾和极端输入 |
| 可逆率 | 编码-解码是否恢复内容 |
| unknown/byte fallback | 覆盖和异常输入 |
| special token 命中 | 协议边界是否稳定 |
| 截断比例 | 上下文预算是否足够 |

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

## 3.19 资料边界与延伸阅读

本章的算法和接口事实来自原始论文、官方 tokenizer 实现和框架文档；具体 token 字符串、id、模板、截断策略和 loss mask 必须以目标模型版本为准。`<system>`、`<user>`、`<assistant>`、`<tool>` 和 `<image>` 不是跨模型统一标准，名字相同也不表示协议相同。

推荐阅读：

- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)：BPE 子词单元的原始论文。
- [SentencePiece](https://arxiv.org/abs/1808.06226)：语言无关的子词 tokenizer 与 Unigram LM。
- [OpenAI tiktoken](https://github.com/openai/tiktoken)：具体 byte-level BPE 实现。
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index)：训练、编码、special token 和 offset 接口。
- [Transformers Chat Templates](https://huggingface.co/docs/transformers/main/en/chat_templating)：消息序列化和生成提示模板。
- [TRL SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer)：SFT 数据格式和 assistant/completion-only loss。
- [PyTorch CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)：`ignore_index` 和逐位置损失。

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

## 3.21 本章小结

Tokenizer 把字符串映射到模型的离散输入空间，数据格式则定义 token 在训练中的结构和监督含义。BPE、Unigram LM 和 byte-level 方法分别在合并策略、概率切分、覆盖率和序列效率上做取舍；词表大小则在 embedding/输出层成本与序列长度之间交换。

EOS、PAD、角色标记、工具事件和多模态占位符构成模型协议。chat template 把消息变成 token 序列，label mask 决定哪些位置产生监督，packing 和 document mask 决定样本边界，偏好和可验证轨迹需要比普通 SFT 更丰富的 schema。

扩展词表必须同步 tokenizer、embedding、模板、数据和服务端；改动任何协议都要用固定样本做 token、mask、截断、生成和回归检查。最终，Tokenizer 不是字符串切分工具，而是贯穿数据、训练、评估和推理的接口层。
