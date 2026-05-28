# C. NLP 与 Tokenization

条目：Token、Vocabulary、Token ID、Tokenization、Tokenizer、BPE、WordPiece、Unigram LM、SentencePiece、Byte-Level Tokenization、Character-Level Tokenization、Subword Tokenization、OOV、Special Token、BOS、EOS、PAD、UNK、Chat Template、Prompt、Completion、Instruction、System Prompt、Context Window、Detokenization、Tokenizer 训练、中文分词、代码 Tokenization、Tokenizer 与 Embedding 扩展。

## Token

一句话定义：token 是模型处理文本的基本离散单位。

核心直觉：token 不一定是自然语言中的“词”，也可能是子词、字符、字节片段、标点、空格片段或特殊控制符号。

在语言模型中：原始字符串会先被 tokenizer 转成 token 序列，再映射为 token id，最后进入 embedding 层。

为什么重要：token 决定模型看到文本的方式，也影响上下文长度、训练成本、推理成本、多语言能力和代码能力。

例子：英文单词 `unbelievable` 可能被切成 `un`、`believ`、`able`；中文句子可能按字、词或子词切分。

常见误区：token 不是字符数，也不是单词数。面试中讨论上下文长度时，要明确是 token 维度而不是字数维度。

面试表达：LLM 不是直接处理字符串，而是处理 token id 序列；tokenization 是自然语言进入神经网络之前的离散化接口。

## Vocabulary

一句话定义：vocabulary 是 tokenizer 支持的 token 集合及其到整数 id 的映射表。

核心内容：每个 token 对应一个唯一 token id，模型的输入 embedding 和输出 logits 通常都以 vocabulary size 为维度。

在大模型中：如果词表大小为 `V`，hidden size 为 `d`，输入 embedding 参数量约为 `V * d`。

词表大小影响：词表越大，单个文本可能被切得更短，但 embedding 和输出层参数更多；词表越小，参数更少，但序列可能更长。

常见误区：大词表不一定更好。过大词表会增加参数和 softmax 成本，也可能让低频 token 学得不好。

面试追问：为什么 tokenizer 会影响训练和推理成本？因为相同文本被切成的 token 数不同，直接影响 attention 计算、KV cache 长度和生成步数。

## Token ID

一句话定义：token id 是 token 在 vocabulary 中对应的整数编号。

在模型中：模型不能直接输入字符串或 token 文本，而是输入 token id 序列。

常见流程：`text -> tokens -> token ids -> embeddings -> transformer`。

工程注意：不同模型即使 token 文本看起来相同，token id 也可能不同，不能混用 tokenizer。

常见误区：token id 本身没有数值大小语义。id 为 100 的 token 不比 id 为 99 的 token “更大”或“更接近”。

面试表达：token id 是离散索引，真正的语义表示来自 embedding 矩阵中对应的向量。

## Tokenization

一句话定义：tokenization 是把原始文本切分成 token 并映射成 token id 的过程。

为什么需要它：神经网络处理的是张量，不能直接处理可变长度字符串；tokenization 把文本转成离散序列。

影响范围：上下文窗口利用率、训练 token 数、推理延迟、多语言公平性、代码表示、特殊符号处理和安全边界都受 tokenization 影响。

典型步骤：文本规范化、预分词、子词切分、special token 拼接、id 映射。

常见误区：tokenization 是模型能力的一部分。它不是无关紧要的预处理，尤其在中文、代码、数学公式和多语言任务中影响明显。

面试表达：同样的模型架构，如果 tokenizer 不同，实际输入长度、词表覆盖和生成行为都可能显著不同。

## Tokenizer

一句话定义：tokenizer 是执行文本和 token id 序列互相转换的组件。

核心能力：encode 把文本转成 token ids，decode 把 token ids 转回文本。

在训练中：tokenizer 决定语料如何统计为训练 token，也决定模型学习的最小文本单位。

在推理中：tokenizer 决定 prompt 如何进入模型，也决定生成 token 如何还原为可读文本。

工程注意：训练、微调、评估和线上推理必须使用一致 tokenizer，否则输入格式和 embedding 对齐会出错。

常见误区：换 tokenizer 通常不是轻量改动。除非重新训练或正确扩展 embedding，否则模型无法理解新 token id 的含义。

面试表达：tokenizer 是模型输入输出协议的一部分，不能只把它看成字符串切分工具。

## Subword Tokenization

一句话定义：subword tokenization 使用子词作为基本单位，在词级和字符级之间折中。

为什么需要它：纯词级词表容易遇到 OOV，纯字符级序列太长；子词方法兼顾覆盖率和序列长度。

常见方法：BPE、WordPiece、Unigram LM、SentencePiece。

优点：能表示未见词、多语言词形变化、专有名词和拼写变化。

局限：切分结果不一定符合语言学边界，可能把语义相关部分切散。

面试表达：子词切分的核心价值是用有限词表覆盖开放词汇，同时控制序列长度。

## BPE Byte Pair Encoding

一句话定义：BPE 通过不断合并高频相邻片段来构建子词词表。

核心过程：从字符或字节等基础单位开始，统计相邻 pair 频率，每次合并最高频 pair，直到达到目标词表大小。

优点：简单、有效、工程成熟，能缓解 OOV 问题。

局限：合并依据主要是统计频率，不保证符合语义边界或词法规则。

在 LLM 中：GPT 系列使用过 BPE 或 byte-level BPE 变体。

常见误区：BPE 的 merge 规则来自训练语料分布，因此 tokenizer 会继承语料偏向；低资源语言可能切得更碎。

面试追问：BPE 为什么能处理未见词？因为未见词可以退化为更小的子词、字符或字节片段组合。

## WordPiece

一句话定义：WordPiece 是一种子词 tokenization 方法，常见于 BERT 系列模型。

核心直觉：它学习一组能较好表示语料的子词单元，并倾向于选择能提升语言模型似然的合并。

常见标记：BERT 中常用 `##` 表示某个子词不是词首，例如 `play ##ing`。

与 BPE 的区别：BPE 更强调高频 pair 合并，WordPiece 更接近基于似然或得分选择子词。

优点：能在词表大小和未登录词处理之间取得平衡。

常见误区：WordPiece 不是简单按词典最大匹配，它背后有子词词表学习和切分策略。

面试表达：WordPiece 是 BERT 时代很典型的子词方案，适合解释为什么预训练模型不需要传统意义上的完整词表。

## Unigram Language Model Tokenization

一句话定义：Unigram LM tokenizer 假设文本由子词片段概率生成，并通过概率模型选择切分。

核心思想：先准备较大的候选子词集合，再逐步删减对语料似然贡献较低的子词。

切分方式：同一句文本可能有多种切分，算法选择概率更高的切分。

在实践中：SentencePiece 支持 Unigram LM，是很多多语言模型 tokenizer 的选择之一。

优点：概率建模更明确，也支持 subword regularization 等训练增强方式。

面试追问：Unigram 和 BPE 的差异是什么？BPE 是从小到大合并，Unigram 通常从大候选集合中做概率筛选。

## SentencePiece

一句话定义：SentencePiece 是一个可直接从原始文本训练 tokenizer 的工具和方法集合。

核心特点：把空格也视为普通符号的一部分，不强依赖预先分词。

为什么适合多语言：中文、日文等语言没有天然空格分词，SentencePiece 可以直接处理原始文本。

常见符号：SentencePiece 常用 `▁` 表示空格或词边界。

支持算法：BPE 和 Unigram LM。

在大模型中：LLaMA 等模型使用 SentencePiece tokenizer。

常见误区：SentencePiece 不是某一种单独切分算法，它既是工具，也支持不同训练算法。

面试表达：SentencePiece 的优势是端到端处理原始文本，减少对语言特定预分词器的依赖，因此很适合多语言大模型。

## Byte-Level Tokenization

一句话定义：byte-level tokenization 从字节层面表示文本，保证几乎所有输入都可以被编码。

核心优势：任意 Unicode 字符最终都能转成字节序列，因此几乎没有 OOV。

适用场景：特殊字符、emoji、代码、混合语言、脏数据和用户输入鲁棒性要求高的场景。

局限：部分语言或特殊文本会被切成更多 token，降低上下文利用率并增加计算成本。

在 LLM 中：byte-level BPE 是常见方案，先保证字节级覆盖，再通过 BPE merge 缩短常见片段。

常见误区：byte-level 不代表每个 token 都是一个字节。很多 byte-level BPE tokenizer 会把高频字节片段合并成更长 token。

面试表达：byte-level tokenization 的核心价值是鲁棒覆盖，代价是某些文本上的 token 效率可能下降。

## Character-Level Tokenization

一句话定义：character-level tokenization 以字符为基本单位切分文本。

优点：词表小，OOV 少，直观简单。

局限：序列长度长，模型需要自己学习从字符到词、短语和语义单位的组合，训练和推理成本更高。

与 byte-level 区别：字符是 Unicode 层面的符号，字节是编码后的底层表示；一个字符可能对应多个字节。

在大模型中：纯字符级方案不是主流 LLM 默认选择，但其思想常用于讨论鲁棒性和开放词汇问题。

面试表达：字符级方案牺牲序列长度换取覆盖率，子词方案则是在覆盖率和效率之间折中。

## OOV Out-of-Vocabulary

一句话定义：OOV 指输入中出现 tokenizer 词表无法直接表示的词或符号。

传统词级问题：如果词表只包含完整词，未见词可能只能映射为 `UNK`。

子词方案缓解：BPE、WordPiece、SentencePiece 可以把未见词拆成更小片段，减少 OOV。

byte-level 方案：几乎可以消除 OOV，因为任意字符最终可分解为字节。

常见误区：没有 OOV 不代表表示质量好。一个罕见词如果被切成很多碎片，模型理解和生成仍可能变差。

面试表达：现代 LLM tokenizer 的目标不是只消除 OOV，还要让常见文本被高效、稳定、语义相对合理地表示。

## Special Token

一句话定义：special token 是具有特殊语义或控制作用的 token。

常见例子：`BOS`、`EOS`、`PAD`、`UNK`、`system`、`user`、`assistant`、`tool`、`image`、`audio`。

在大模型中：special token 用于标记序列边界、角色边界、填充位置、多模态占位符和工具调用结构。

工程注意：新增 special token 后通常需要 resize embedding，并确保训练、微调、评估和推理模板一致。

风险：如果 special token 处理不一致，模型可能无法正确区分用户、助手、系统指令或工具结果。

常见误区：special token 不是普通文本字符串。它们通常在 tokenizer 中有独立 id，并可能被模型学到特殊行为。

面试表达：chat model 的行为很大程度依赖 special token 和 chat template，它们定义了对话数据的结构协议。

## BOS Beginning of Sequence

一句话定义：BOS 是表示序列开始的特殊 token。

作用：告诉模型一个新序列或新样本开始，有助于统一训练样本格式。

在推理中：部分模型需要显式 BOS，部分 tokenizer 会自动添加。

工程注意：是否添加 BOS 必须与模型训练时的格式一致。

常见误区：所有模型都必须手动加 BOS。不同模型约定不同，应以 tokenizer 和 chat template 为准。

## EOS End of Sequence

一句话定义：EOS 是表示序列结束的特殊 token。

作用：训练时告诉模型何时结束生成，推理时常作为停止条件。

在对话模型中：EOS 可能表示 assistant 回答结束，也可能与 turn 边界 token 共同使用。

工程注意：如果训练数据缺少 EOS，模型可能学不会正常停止；如果停止符配置错误，可能截断或无限生成。

常见误区：max_new_tokens 不是正常停止机制。好的模型应能在合适位置生成 EOS 或停止标记。

面试表达：EOS 影响生成终止行为，是训练格式和推理 stop condition 的连接点。

## PAD Padding Token

一句话定义：PAD 是用于把不同长度序列补齐到同一长度的特殊 token。

为什么需要它：batch 训练时张量通常需要统一形状，短序列需要 padding。

配套机制：attention mask 用于告诉模型哪些位置是有效 token，哪些位置是 padding。

在 decoder-only LLM 中：很多模型原始预训练不需要 PAD，但微调或批量推理时可能需要设置 pad token。

工程注意：不要让 loss 计算在 PAD 位置生效，否则模型会学习无意义目标。

常见误区：把 PAD 简单设成 EOS 虽然常见，但要确认 attention mask、loss mask 和生成停止逻辑不会冲突。

## UNK Unknown Token

一句话定义：UNK 用于表示词表中无法识别的未知 token。

在现代 tokenizer 中：由于子词和 byte-level 方法普及，UNK 的使用频率通常显著降低。

风险：大量 UNK 会导致信息丢失，因为不同未知文本都被压成同一个 id。

工程注意：如果数据中 UNK 比例异常高，通常说明 tokenizer 与数据语言或文本格式不匹配。

面试表达：现代 LLM 尽量避免把未知词直接映射为 UNK，而是拆成更小单位保留信息。

## Chat Template

一句话定义：chat template 是把多轮对话消息转换为模型输入 token 序列的格式规则。

核心内容：角色标记、消息边界、system prompt 位置、assistant 起始标记、工具调用格式、结束标记。

例子：同样的 messages 列表，不同模型可能格式化为完全不同的字符串和 special token 序列。

为什么重要：chat model 训练时学习的是某种模板格式，推理时模板不一致会导致角色混乱、拒答异常或输出格式错误。

工程注意：不要手写猜测模板，优先使用 tokenizer 自带的 `apply_chat_template` 或官方格式。

常见误区：prompt 内容相同不代表模型输入相同。chat template 不同，实际 token 序列可能完全不同。

面试表达：chat template 是 instruct/chat 模型的输入协议，决定 system/user/assistant 等角色如何被模型识别。

## Prompt

一句话定义：prompt 是提供给模型的输入上下文，用来约束或引导模型生成。

组成：任务说明、背景信息、示例、约束条件、输出格式、用户问题和必要的系统指令。

在 base model 中：prompt 通常就是一段普通文本上下文。

在 chat model 中：prompt 常由 system、user、assistant 历史消息经过 chat template 组装而成。

常见误区：prompt 不是越长越好。冗余、冲突或低质量上下文会增加成本并干扰模型。

面试表达：prompt engineering 的核心不是堆关键词，而是清晰定义任务、约束、上下文和输出协议。

## Completion

一句话定义：completion 是模型在给定 prompt 后继续生成的文本。

在预训练中：语言模型学习根据前文预测后续 token，本质上就是 completion 式目标。

在指令模型中：completion 通常对应 assistant 的回答。

与 prompt 的关系：prompt 是条件，completion 是模型生成的结果；训练数据中二者边界必须清楚。

工程注意：SFT 数据要正确 mask prompt 部分，通常只对 assistant completion 计算 loss。

常见误区：把 prompt 和 completion 边界处理错，会让模型学习复读用户问题或错误角色文本。

## Instruction

一句话定义：instruction 是用户或系统给模型的任务指令。

例子：翻译、总结、分类、改写、写代码、按 JSON 输出、扮演某种角色。

在 SFT 中：instruction-response 数据用于让模型学会遵循任务说明生成回答。

与 prompt 的区别：prompt 是完整输入上下文，instruction 是其中表达任务要求的部分。

常见误区：instruction 越复杂不一定越好。冲突指令、多目标指令和含糊约束会降低生成稳定性。

面试表达：指令微调的关键是让模型从纯续写转向理解并执行显式任务要求。

## System Prompt

一句话定义：system prompt 是对模型行为、身份、边界和全局规则的高优先级指令。

常见作用：设定助手角色、安全边界、回答风格、工具使用规则和输出格式约束。

在 chat template 中：system prompt 通常位于对话最前面，并通过特殊角色标记与用户消息区分。

工程风险：如果训练和推理中 system prompt 格式不一致，模型可能忽视规则或把系统指令当普通文本。

常见误区：system prompt 不是绝对安全机制。模型仍可能受越狱提示、上下文冲突或能力限制影响。

面试表达：system prompt 是控制模型行为的重要接口，但真正可靠的系统还需要数据、对齐、安全策略和推理侧防护共同支撑。

## Context Window

一句话定义：context window 是模型一次前向可接收的最大 token 长度。

核心单位：上下文窗口以 token 计，不以字符、汉字或英文单词计。

影响因素：位置编码、训练长度、attention 实现、KV cache、显存和推理框架限制。

与 tokenizer 的关系：同一段文本在不同 tokenizer 下 token 数可能不同，因此能放入上下文的实际字符量也不同。

常见误区：标称 128K context 不代表所有 128K token 信息都能被同等有效利用。长上下文还涉及注意力稀释、位置外推和检索能力问题。

面试表达：上下文窗口是 token 级限制，长上下文能力还要看训练数据、位置编码、注意力机制和评估结果。

## Detokenization

一句话定义：detokenization 是把 token id 序列还原为文本的过程。

核心流程：`token ids -> tokens -> text`。

工程难点：空格、换行、Unicode、byte fallback、特殊 token 跳过和流式输出都会影响还原结果。

在生成中：模型每生成一个 token，推理系统通常会逐步 decode 并返回可读文本。

常见误区：decode 单个 token 再拼接，不一定等价于 decode 完整 token 序列，尤其在 byte-level 或特殊空格编码中要小心。

面试表达：tokenization 和 detokenization 共同定义模型的文本 I/O 协议，线上问题经常出在边界符、空格和特殊 token 处理上。

## Tokenizer Training

一句话定义：tokenizer training 是从语料中学习词表、merge 规则或子词概率模型的过程。

核心输入：代表目标应用分布的文本语料，包括语言比例、代码比例、领域文本、特殊符号和格式化数据。

关键超参数：vocab size、算法类型、normalization、special tokens、character coverage、byte fallback。

为什么重要：tokenizer 一旦确定，通常会和模型训练绑定，后续随意更换成本很高。

数据影响：如果训练 tokenizer 的语料中中文、代码或数学公式不足，相关文本可能被切得很碎。

常见误区：只用通用语料训练 tokenizer 就能覆盖所有场景。面向代码、金融、医疗、多语言的模型需要考虑目标数据分布。

面试表达：tokenizer 训练是模型数据工程的一部分，它决定哪些文本模式被压缩为高效 token，哪些模式会被拆碎。

## 中文 Tokenization

一句话定义：中文 tokenization 需要处理没有天然空格分词边界的问题。

常见方式：按字、按词、按子词或使用 SentencePiece 直接从原始文本学习。

难点：中文词边界不显式，专有名词、新词、数字单位和中英混排很常见。

对 LLM 的影响：如果中文被切得过碎，相同语义会占用更多 token，降低上下文利用率并增加推理成本。

工程观察：中文模型或多语言模型通常需要关注中文 token fertility，即每个汉字或每句话平均被切成多少 token。

常见误区：中文按字切就一定好。按字鲁棒但序列长，按词可能语义好但 OOV 和分词错误更多，子词方法是常见折中。

面试表达：中文 tokenizer 的重点是覆盖率、切分效率和语义边界之间的平衡。

## Code Tokenization

一句话定义：code tokenization 是面向代码文本的切分方式，需要处理缩进、符号、标识符和多语言语法。

代码特点：大量括号、点号、下划线、驼峰命名、缩进、换行、字符串和特殊运算符。

关键影响：tokenizer 如果不适合代码，常见标识符和语法符号会被切得很碎，影响代码生成效率。

重要细节：空格和换行在 Python 等语言中有语义，不能随意 normalize 掉。

byte-level 优势：对特殊符号和罕见字符更鲁棒，适合处理真实代码仓库中的混杂文本。

常见误区：代码只是英文文本。代码有强格式和符号结构，tokenizer 需要保留足够结构信息。

面试表达：代码模型的 tokenizer 要重视标识符、缩进、符号和长尾字符，否则会直接影响上下文长度和生成质量。

## Tokenizer 与 Embedding 扩展

一句话定义：扩展 tokenizer 通常意味着新增 token，并同步扩展模型 embedding 和输出层。

典型场景：新增 special token、领域术语、多模态占位符、工具调用标记或结构化输出标记。

工程步骤：更新 tokenizer，resize token embeddings，初始化新增 token 向量，并通过训练让模型学会使用这些 token。

风险：只改 tokenizer 不改模型会导致 token id 越界；只 resize 不训练，新 token 语义也不会自动学会。

在 tied embedding 中：输入 embedding 和输出 lm head 可能共享权重，扩展时要确认两者同步。

常见误区：新增 token 后模型马上理解它。新 token 的 embedding 初始通常是随机或均值初始化，需要微调学习。

面试表达：tokenizer 扩展是模型结构和数据协议变更，必须配合 embedding resize、初始化和训练数据设计。

## Tokenizer Mismatch

一句话定义：tokenizer mismatch 指训练、微调、评估或推理阶段使用了不一致的 tokenizer 或模板。

常见后果：输入 id 错误、special token 错位、角色边界混乱、生成异常、评估结果不可比。

典型场景：用 A 模型 tokenizer 编码 B 模型 prompt，或者 SFT 使用一种 chat template，线上推理使用另一种模板。

工程排查：打印 token ids、decode 后文本、special token 位置和 attention mask，确认与训练格式一致。

常见误区：只要文本看起来一样，模型输入就一样。实际 token id 序列可能完全不同。

面试表达：tokenizer mismatch 是大模型工程中非常常见且隐蔽的问题，排查时必须看 token 级输入，而不只是看原始字符串。

## 本章小结

本章覆盖 NLP 与 tokenization 中最常见的面试概念和工程问题。

核心结论如下：

1. LLM 处理的是 token id，不是原始字符串。
2. token 不等于词、字符或字节，它由 tokenizer 的训练方式和词表决定。
3. BPE、WordPiece、Unigram LM 和 SentencePiece 都是在词表大小、覆盖率和序列长度之间做权衡。
4. byte-level tokenization 几乎消除 OOV，但可能牺牲部分语言的 token 效率。
5. special token 和 chat template 定义了对话模型的输入输出协议。
6. prompt、completion、instruction、system prompt 的边界处理会直接影响 SFT 和推理行为。
7. tokenizer mismatch、PAD/EOS 配置错误、模板不一致是大模型工程中的高频坑。
8. 面试中要能从 tokenization 连接到上下文长度、KV cache、训练成本、多语言能力和线上稳定性。

下一章，我们进入 Transformer 架构。
