# 第三章：Tokenizer、模板与训练格式

很多“模型不听话”的故障，根因不在模型结构，而在输入和监督边界。模型不停生成，可能是协议要求的结束标记没有进入训练目标或停止配置；模型复述用户问题，可能是 user token 没有被 mask；模型忽略 system 指令，可能是训练和推理使用了不兼容的 chat template；工具调用无法解析，可能是训练数据里的 schema、工具名和线上协议不一致。

这些问题很隐蔽，因为训练过程通常仍然能够运行，loss 也可能正常下降。格式错不是“字符串长得不漂亮”，而是改变了模型看到的序列、角色和目标。一个 token 级别的偏差会在数百万条样本和多轮生成中累积，最后表现为能力退化、成本上升或外部动作错误。

本章把格式处理看成一条有契约的数据流：消息经过模板渲染成为序列，tokenizer 把序列映射为 token ID，collator 生成 attention mask 和 labels，损失函数只在规定位置监督，推理侧再按兼容的协议解析角色、工具和停止边界。读者将看到 special token、BOS/EOS/PAD、assistant-only loss、chat template、截断、中文/代码/数学 tokenization、多模态占位符和工具调用如何在同一条链上相互影响。

本章以故障复现、观测与修复为主；训练侧字段契约、SFT/偏好/工具/多模态样本结构及 tokenizer 扩展流程见[第五册第三章](../../book-05-llm-training/chapters/03-tokenizer与数据格式.md)。Tokenizer 算法和指标比较见[第二册第三章](../../book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md)。

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

训练和推理的协议字段若不兼容，系统就可能发生协议漂移。比如训练使用 `<|assistant|>` 作为角色边界，推理却使用 `### Response:`；训练让结束标记参与损失，推理却配置了另一个 `eos_token_id`；训练把 tool result 当作 user 文本，推理则使用独立的 `tool` role。这些变化都会改变模型的条件分布。

一个最小聊天样本可以写成：

```json
[
  {"role": "system", "content": "你是一个严谨的助手"},
  {"role": "user", "content": "解释注意力机制"},
  {"role": "assistant", "content": "注意力机制根据查询与键的关系聚合值"}
]
```

对于这个样本，训练时通常希望模型学习 assistant 内容和任务规定的结束边界，而不是学习把 system 和 user 内容重新生成一遍。模型需要“看见”整个上下文，但不代表每个位置都应该进入 loss。

## 3.2 Tokenizer 改变的是计算单位

Tokenizer 把文本映射为模型词表中的离散 ID。BPE（Byte Pair Encoding）和 Unigram 是子词模型算法；SentencePiece 是一种分词工具包，可以训练和应用 BPE 或 Unigram 等模型；byte-level 描述输入单位或算法变体，常与 BPE 组合使用，不是与 SentencePiece 同层级的算法名称。它们的实际行为还取决于规范化、预分词、词表和后处理配置。[BPE 论文](https://arxiv.org/abs/1508.07909)、[SentencePiece 论文](https://arxiv.org/abs/1808.06226)与 [Transformers 分词算法说明](https://huggingface.co/docs/transformers/v5.17.0/tokenizer_summary)分别展示了算法与工具的区别。模型并不直接以“汉字”“单词”或“代码行”为计算单位，而是以 token 序列长度、位置和嵌入为计算单位。

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

Token 数适合估算序列长度、上下文占用和成本，但不是 tokenizer 质量或下游能力的通用代理。合并过度可能损失边界和组合能力，切分过细会增加序列长度；实际选择仍要在指定任务、语言、模型规模和训练条件下比较。对约 200 个、覆盖 9 种语言的模型所做的一项 2026 年研究发现，没有单一内在 tokenizer 指标能在其测试的模型与任务中稳定预测下游表现；这一结论也受其模型和评估范围限制。[研究论文](https://aclanthology.org/2026.findings-emnlp.96/)

## 3.3 Special token 是协议字段

`BOS`、`EOS`、`PAD`、`UNK`、role token、tool token 和多模态 placeholder 都是协议的一部分。它们不能仅按“词表中几个保留 ID”处理，因为每个 ID 在训练、生成、padding、停止和跨模态对齐中承担不同语义。

常见字段包括：

- `BOS`：序列开始，是否需要由模板或 tokenizer 添加必须明确；
- `EOS`：某个 tokenizer 或模型协议定义的结束标记；它是否表示消息结束、assistant 轮次结束或整段序列结束，须看协议；
- `PAD`：批处理对齐用的填充，不应作为真实内容参与 loss；
- `UNK`：无法映射的未知符号，出现比例过高说明词表或输入处理有问题；
- role token：标记 system、user、assistant、tool 的边界；
- tool token：标记调用名称、参数、结果或调用结束；
- image/audio/video token：承载跨模态特征插入位置。

检查 special token 时至少要验证：每个 ID 是否符合该模型协议，若多个 token 有意共用 ID 是否已明确记录，训练和推理是否使用兼容映射，decode 是否能恢复边界，padding 和停止配置是否与模板一致，模型词表大小和 embedding 行数是否匹配。

若 `pad_token_id` 与 `eos_token_id` 共用一个 ID，影响取决于数据整理和推理实现。按 ID 相等来屏蔽 padding 标签的 collator 可能连真实 EOS 目标一起屏蔽；根据真实 padding 位置构造 mask 的实现则可以区分二者。推理停止还取决于生成器的停止 ID、停止字符串和协议边界，不能仅凭 ID 共享就断言会提前停止。检查时要同时核对 tokenizer 配置、attention mask、collator 逻辑和 generation 配置；例如 [Transformers v5.17 的语言模型 collator](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/data/data_collator.py#L3393-L3400)按与 `pad_token_id` 的相等关系屏蔽 labels，这个例子只说明该实现路径的行为。

结束标记也不总是同一种边界。协议可能分别表示“当前消息结束”“assistant 轮次结束”和“交出控制权给工具”；某个格式可以用不同 token 表示，也可能由外层字段表达。不要把所有结束标记都直接叫作 EOS。先从模型或 API 的协议文档确认标记含义，再分别核对训练目标、解析器和停止配置。

## 3.4 BOS、EOS 和 PAD 的不同责任

### BOS

有些模型由模板插入 BOS，有些模型由 tokenizer 的 `add_special_tokens` 插入。两者同时开启会产生重复 BOS，全部关闭又可能造成训练和推理边界不同。必须在渲染前后分别打印 token 序列，不要只看 decode 后的文本，因为某些 special token 在 decode 时会被隐藏。

### EOS 与轮次边界

有的协议用 EOS 表示序列结束；有的聊天协议还会区分消息结束、assistant 轮次结束和工具交接。模型什么时候该停，取决于其协议如何映射这些边界，以及生成器监听哪些停止条件。下面几种情况可能让模型不停生成或过早停止：

1. 任务要求 assistant 生成结束标记，但样本中缺少该标记。
2. 任务要求模型生成结束标记，但它存在于 input，却在实际 loss 目标中被设为 `-100`。
3. 推理配置的 `eos_token_id` 与训练使用的 ID 不同。
4. 多轮模板没有提供协议要求的消息结束边界，或把消息结束与 assistant 轮次结束混为一谈。
5. stop string 与 token 级 EOS 不一致，文本层停止没有覆盖所有分词形式。

### PAD

PAD 用于把不同长度的样本放入同一个 batch。常见做法是用 attention mask 标记 padding，并在损失目标中忽略对应位置；但 attention mask 与 loss mask 是不同字段，框架也可能用显式长度或位置元数据代替某个字段。关键是分别验证 padding 是否按预期影响可见上下文，以及对应目标是否实际进入损失。

还要区分**因果标签位移**。语言模型常用位置 `t` 的输出预测下一个 token；有些模型头部会在内部把 logits 与 labels 错开一位，另一些训练管线则预先移位。若模型已在内部移位，再由 collator 移一次，目标和预测就会错开两位。查看目标模型和固定版本的 loss 实现，确认唯一的一次 shift 落在哪里；例如 [Transformers v5.17 的 GPT-2 头](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/gpt2/modeling_gpt2.py#L3227-L3232)说明 labels 在模型内部移位，这不是所有模型的共同保证。

对某个 batch，令 `m_t` 表示经过目标位置对齐、attention/label 规则和忽略值处理后，真正进入交叉熵的目标位，单个 token 的损失为 `ell_t`，则平均损失可写为：

```math

L=\frac{\sum_t m_t\ell_t}{\sum_t m_t}
```

要求 `\sum_t m_t>0`。如果一个 batch 全部是 padding、过滤后没有 assistant token 或角色解析失败，损失没有定义。不能把分母替换成一个人为的小数后继续训练，否则会把数据错误变成一个看似正常的数值。

用一个六 token 的 toy 序列看清 shift 与 mask 的关系：

| 位置 | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| 输入 token | BOS | user | 问题 | assistant 头 | 答案 | EOS |
| attention mask | 1 | 1 | 1 | 1 | 1 | 1 |
| labels | `-100` | `-100` | `-100` | `-100` | 答案 ID | EOS ID |

假设这项任务只监督答案和 EOS，且模型头部负责 next-token shift。模型位置 3 的 logits 预测 labels 位置 4 的答案 ID，位置 4 的 logits 预测位置 5 的 EOS ID，所以有效目标数为 2；user 问题和 assistant 头仍作为上下文输入，却不进入 loss。若训练代码已经预先移位，表中的 labels 与 logits 对齐方式必须按该代码重新核对，不能再由模型头重复移位。

## 3.5 Chat template 是模型协议，不是普通 prompt

Chat template 把消息列表变成模型训练和推理时看到的具体序列。它决定角色标签、换行、special token、assistant 起始位置、工具调用语法和结束边界。不同模型可以为相同消息采用不同 token 和控制格式；使用框架时，应读取目标模型随附的模板并按其文档调用，而非把示例模板当成标准格式。[Transformers v5.17 的模板说明](https://huggingface.co/docs/transformers/v5.17.0/chat_templating)列出了这种模型间差异。

下面两个片段只是**抽象伪格式，用于说明格式可能不同，不对应任何具体模型**：

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

一种常见的监督微调目标是让模型根据 system 和 user 上下文生成 assistant 答案，而不把 user 问题本身当作模仿目标。于是需要构造 labels：assistant 内容以及任务规定要生成的结束标记保留目标 ID；哪些角色头、分隔符或其他控制 token 也作为目标，要由训练目标和模板约定明确决定。system、user 和明确的上下文控制位置通常设为忽略值，常见实现是 `-100`。因此，“assistant-only loss”描述的是一个训练目标，不等于“所有控制 token 都必须忽略”。[TRL v0.29 的 SFT 文档](https://huggingface.co/docs/trl/v0.29.0/en/sft_trainer)提供了一个具体框架入口；实际位置仍须回放模板与 labels。

先把两类 mask 和位移分开：attention mask 规定输入位置如何参与模型计算；loss mask 规定哪些目标位置计入损失；causal shift 则把每个模型输出与它应预测的下一个 token 对齐。三者不能互相替代。以下指标中的“实际计入损失”指完成目标模型所需的 shift 后、真实送入损失函数的位置。令 `T_assistant` 是由任务合同声明的 assistant 目标位置集合，可包括内容 token、结束标记，以及目标明确要求生成的控制 token；角色头是否属于目标也要在合同中说明。实际计入损失的位置集合记为 `T_loss`。

assistant 覆盖率可以写成：

```math

C_{\mathrm{assist}}=
\frac{|T_{\mathrm{assistant}}\cap T_{\mathrm{loss}}|}
{|T_{\mathrm{assistant}}|}
```

只有 `T_assistant` 非空时这个比例才有定义。没有 assistant 目标的样本可能是纯 user 输入、工具结果或解析失败，不能直接当成覆盖率 0；应该根据任务标记为 `not_applicable` 或 `unknown`。

反过来，定义 `P` 为本任务合同明确规定只作为上下文、不作为目标的位置集合，例如 system/user 内容和某些控制边界。prompt 泄漏率为：

```math

R_{\mathrm{prompt}}=
\frac{|P\cap T_{\mathrm{loss}}|}{|P|}
```

对于约定这些位置只作为上下文的 assistant-only SFT，`R_prompt` 应为 0；如果任务有意训练完整序列，例如某些语言建模或格式续写任务，就要重新定义目标集合，不能把所有训练都套用同一个 assistant-only 规则。

只有 `P` 非空时该比率才有定义；若当前任务没有只作上下文的位置，应记为 `not_applicable`，而不是把空分母当作零泄漏。

多轮对话还要决定是否监督每一轮 assistant。只监督最后一轮可以节省目标 token，但会让早期 assistant 行为没有训练信号；监督所有 assistant 轮次则需要确保每一轮 role、EOS 和上下文都正确。选择哪一种，应与训练目标和数据构成一致。

## 3.7 多轮、工具和结构化输出的边界

多轮对话的格式错误经常发生在单轮样本无法暴露的地方：上一轮 assistant 缺少协议要求的轮次结束标记，tool result 被当成 user，下一轮 assistant 起始 token 丢失，或历史被截断后 role 不闭合。排查时要把一条完整多轮样本 decode 出来，并在每个 role 边界标出 labels 和 attention mask。

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

结构化输出也不能只检查 JSON 是否可解析。JSON 可以语法正确，却缺字段、类型错误、枚举值非法或违反业务约束。协议检查应分成语法、schema、语义和权限四层。以 OpenAI API 为例，Function Calling 使用 JSON Schema 定义工具参数，Structured Outputs 可约束符合支持范围的 schema；官方文档仍要求应用处理拒绝、不完整响应和 schema 未表达的业务规则。[Function Calling 文档](https://developers.openai.com/api/docs/guides/function-calling)和[Structured Outputs 文档](https://developers.openai.com/api/docs/guides/structured-outputs)说明的是该 API 的公开行为，不能直接推断其他服务或本地模型的协议。

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

截断前后应记录：原始长度、保留长度、删除区间、被删除的角色、是否删除了证据、是否保留了协议要求的结束边界、是否改变了 labels。不能只记录最终 token 数，否则无法解释某类长输入为什么突然失败。

## 3.9 中文、代码和数学的 token 预算

中文字符、代码符号和数学控制序列的 token 化效率差异很大。中文专业词可能被切成多个单字，代码的缩进、括号和运算符可能占用大量 token，LaTeX 的反斜杠命令和 Unicode 数学符号可能形成复杂边界。

对切片 `b`，先明确分母的含义。下面的 token/字符比把字符定义为 Unicode 码点数；不同语言比较时，也可以另报 token/UTF-8 字节比。UTF-8 字符使用 1 到 4 个字节编码，不能假设每个汉字都恰好占 3 字节。[RFC 3629](https://www.rfc-editor.org/rfc/rfc3629)规定了 UTF-8 的字节结构。

```math

\rho_b=\frac{\sum_{x\in b}|\tau(x)|}{\sum_{x\in b}|x|_{\mathrm{cp}}}
```

要求码点分母大于 0，并在比较中固定计数口径；若报告 UTF-8 字节比，就把分母明确替换为字节数。`rho_b` 越小不一定越好：过度合并可能影响泛化，过大则减少有效上下文容量和增加成本。更重要的是比较同一 tokenizer 的切片差异，并将差异连接到任务准确率和截断率。

建议建立小型领域词表：中文产品术语、代码 API、数学符号、工具字段、法律条款和多语言实体。逐项查看 token ID、decode round trip 和长度。一个 tokenizer 更新后，即使词表大小只增加少量，也可能改变大量领域序列的切分和历史 checkpoint 行为。

## 3.10 多模态 placeholder 与特征对齐

多模态模型通常在文本序列中插入 `<image>`、音频或视频 placeholder，再由媒体编码器产生特征。placeholder 到特征的映射方式由具体模型协议决定：一个图像标记可能对应多个 crop 或 patch 特征，一个媒体输入也可能经过查询模块形成不同数量的特征。因此，要核对的是模型要求的映射关系、顺序、位置和特征形状，不能普遍要求 placeholder 个数等于特征组数。[LLaVA](https://arxiv.org/abs/2304.08485)和 [BLIP-2](https://arxiv.org/abs/2301.12597)展示了不同的视觉编码与语言模型连接设计，只能说明这些架构的做法，不能推导出统一的一对一计数规则。

对包含媒体的样本集合 `M`，令 `\pi_i^{\mathrm{obs}}` 表示第 `i` 条样本实际构造的 placeholder、媒体和特征映射，`\pi_i^{\mathrm{exp}}` 表示该模型版本规定的映射，则映射符合率为：

```math

C_{\mathrm{media}}=\frac{1}{|M|}\sum_{i\in M}\mathbf{1}[\pi_i^{\mathrm{obs}}=\pi_i^{\mathrm{exp}}]
```

如果评估集没有任何多模态样本，`C_media` 是 `not_applicable`，不能当作 1；如果样本应该有媒体但特征提取失败，应是 `unknown` 或失败状态。只有在某个实现明确声明“一标记对应一组特征”时，原始数量相等才是映射符合的必要检查之一。多图任务还要检查顺序，例如第一张图的特征不能被放到第二个 placeholder。

排查时同时打印：

- 原始消息中的媒体引用；
- 渲染后的 placeholder 位置；
- token ID 和媒体索引；
- 编码器输出的特征组数和形状；
- 截断前后的媒体映射；
- labels 是否把协议规定只作媒体边界的占位符错误地计入文本目标；若任务明确要求模型生成该标记，则按该目标合同检查。

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

格式问题应先用一条短样本复现，不要一开始在大训练和长对话中猜测。推荐使用包含 system、user、assistant、协议要求的结束边界、PAD、工具或媒体的最小样本，逐层保存：

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

C_{\mathrm{assist}}=\frac{|T_{\mathrm{assistant}}\cap T_{\mathrm{loss}}|}{|T_{\mathrm{assistant}}|}
```

符号沿用 3.6：分母是任务合同声明应监督的 assistant 目标位置，不是所有 assistant 角色 token；实际计数在模型所需的 causal shift 和 mask 规则应用后进行。分母为空时为 `not_applicable` 或 `unknown`。prompt 泄漏率的分母是合同定义为上下文的位置数；纯 assistant continuation 任务不应套用普通对话的分母。

### 3.13.3 PAD 泄漏率

```math

R_{\mathrm{pad}}=\frac{N_{\mathrm{pad\ positions\ with\ loss}}}{N_{\mathrm{pad\ positions}}}
```

PAD 位置应由 attention mask 或显式长度元数据定位，不能只通过 token ID 反推，特别是 PAD 与 EOS 共用 ID 时。没有 PAD 位置时是 `not_applicable`，不是 0；若存在 PAD 但实际目标仍计入 loss，才是泄漏。

### 3.13.4 EOS 覆盖率

对确实要求模型生成 EOS 的样本集合 `E`，且 `|E|>0`：

```math

C_{\mathrm{eos}}=\frac{|T_{\mathrm{eos}}\cap T_{\mathrm{loss}}|}{|T_{\mathrm{eos}}|}
```

其中 `T_eos` 是 `E` 中按协议预期应由模型生成 EOS 的位置，即使实际样本缺少该 token 也要计入分母。消息结束、轮次结束或工具交接使用其他边界时，应定义相应指标，不要混入 `E`。

### 3.13.5 有效损失分母

```math

N_{\mathrm{loss}}=|T_{\mathrm{loss}}|
```

`T_loss` 是模型完成 causal shift、attention/label 规则与忽略值处理后，实际送入损失计算的目标位置集合。原始 `labels != -100` 的数量不一定等于它：如果模型头在内部移位并舍弃边界位置，分母会随具体实现改变。若 `N_loss=0`，该样本或 batch 的平均损失未定义，应在数据处理阶段阻断或隔离；还要确认 shift 只发生一次。

这些指标不能简单相乘成一个格式总分。模板不匹配、PAD 泄漏和媒体错位的后果不同，应该保留具体失败项和触发样本。

## 3.14 四个典型格式事故

### 事故一：SFT 后复述用户

训练样本 decode 看起来正常，生成却先复述 user 问题。检查 labels 发现 system 和 user 位置没有设为 `-100`。模型被训练去预测整个对话序列，因此复述是训练目标的合理结果。修复后要比较 prompt loss 泄漏率、assistant 覆盖率和固定生成样例。

### 事故二：模型生成到最大长度

输出没有任务要求的结束边界，或者生成器没有监听对应的停止条件。检查训练样本中预期边界是否存在并进入实际 loss，确认模板的轮次结束语义，再核对生成侧的 stop ID、stop string 和解析规则。只增加 `max_new_tokens` 会掩盖问题并增加成本。

### 事故三：工具 JSON 语法正确但动作错误

模型生成了合法 JSON，却使用了旧字段名、非法枚举值或错误工具名。解析通过不代表 schema 和语义通过。修复需要绑定工具 schema 版本、增加参数验证、保存 call ID 和拒绝原因，并将自然语言、工具请求和工具结果分开训练和评估。

### 事故四：模型像没有看到图

检查发现消息中的媒体占位映射与该模型协议要求的不一致，或者截断保留了 placeholder，却删除了对应媒体。此时模型回答错误不是纯粹的视觉识别失败，而是输入对齐失败。修复后要用多图顺序、缺图、空图和截断边界回归。

## 3.15 一个 toy 格式审计示例

下面的 Python 示例使用 toy tokenizer 和明确声明的一对一媒体映射教学协议，不代表任何具体模型的内部实现。它故意构造模板不一致、错误 labels、截断丢结束标记和映射不一致，同时演示 assistant 目标位置、padding 位置和空分母边界。脚本检查同一序列位置上的 labels 与 mask，不模拟模型头部的 causal shift；实际系统还要按目标模型确认 logits 和 labels 如何对齐。压缩比只报告观测值，不使用统一的 token/字符阈值判断合格与否。

代码状态：**静态审阅，未运行**。

```python
import re
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
    if denominator == 0:
        if numerator == 0:
            return None
        raise ValueError("a positive numerator requires a non-zero denominator")
    if numerator > denominator:
        raise ValueError("numerator cannot exceed denominator")
    return round(numerator / denominator, 3)


def validate_special_ids(mapping: Dict[str, int]) -> None:
    if not mapping:
        raise ValueError("special token map must not be empty")
    values = list(mapping.values())
    if any(not strict_int(value) or value < 0 for value in values):
        raise ValueError("special token ids must be non-negative integers")
    # Aliased IDs such as PAD=EOS may be valid under a declared model contract.


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
attention_mask = [1] * len(input_ids) + [0] * (padded_len - len(input_ids))
input_ids.extend([SPECIAL_IDS["<pad>"]] * (padded_len - len(input_ids)))
correct_labels = labels_assistant_only(train_sequence, padded_len)
bad_labels = labels_all_tokens(train_sequence, padded_len)

# Demonstrate one collator behavior when PAD and EOS share an ID.
shared_pad_eos_id = 2
shared_ids = [shared_pad_eos_id, shared_pad_eos_id]  # real EOS, then padding
shared_attention_mask = [1, 0]
id_based_labels = [
    -100 if value == shared_pad_eos_id else value for value in shared_ids
]
position_based_labels = [
    value if visible else -100
    for value, visible in zip(shared_ids, shared_attention_mask)
]

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
pad_positions = [index for index, visible in enumerate(attention_mask) if visible == 0]

samples = {
    "en": "the quick brown fox explains attention",
    "zh": "中文客服需要覆盖错别字",
    "code": "def add(a, b): return a+b",
    "math": "L = - log p(y | x)",
}
token_counts = {name: len(encode(text)) for name, text in samples.items()}
code_point_counts = {name: len(text) for name, text in samples.items()}
utf8_byte_counts = {name: len(text.encode("utf-8")) for name, text in samples.items()}
tokens_per_code_point = {
    name: round(token_counts[name] / code_point_counts[name], 3)
    for name in samples
    if code_point_counts[name]
}
tokens_per_utf8_byte = {
    name: round(token_counts[name] / utf8_byte_counts[name], 3)
    for name in samples
    if utf8_byte_counts[name]
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
    "special_ids_valid": all(
        strict_int(value) and value >= 0 for value in SPECIAL_IDS.values()
    ),
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
    "media_alignment_under_toy_protocol": placeholder_count == image_feature_count,
    "shared_id_case_shows_id_mask_bug": id_based_labels[0] == -100,
    "position_mask_preserves_real_eos": position_based_labels[0] == shared_pad_eos_id,
}

print("special_ids=", SPECIAL_IDS)
print("token_counts=", token_counts)
print("tokens_per_code_point=", tokens_per_code_point)
print("tokens_per_utf8_byte=", tokens_per_utf8_byte)
print("input_ids_head=", input_ids[:12])
print("attention_mask=", attention_mask)
print("label_summary=", label_summary)
print(
    "shared_pad_eos=",
    {
        "id_based_labels": id_based_labels,
        "position_based_labels": position_based_labels,
    },
)
print("template_match=", template_match)
print("train_prefix=", " ".join(piece for piece, _ in train_sequence[:12]))
print("infer_prefix=", " ".join(piece for piece, _ in infer_sequence[:12]))
print("truncation=", {"max_len": 18, "critical_missing": critical_missing})
print(
    "multimodal=",
    {
        "declared_protocol": "one_placeholder_per_feature_group",
        "placeholders": placeholder_count,
        "image_features": image_feature_count,
    },
)
print("checks=", checks)
print("all_checks_pass=", all(checks.values()))

assert checks["special_ids_valid"]
assert checks["assistant_mask_ok"]
assert checks["bad_mask_detected"]
assert checks["eos_supervised"]
assert checks["template_match"] is False
assert checks["truncation_preserves_boundaries"] is False
assert checks["media_alignment_under_toy_protocol"] is False
assert safe_ratio(0, 0) is None

try:
    safe_ratio(1, 0)
except ValueError:
    pass
else:
    raise AssertionError("positive numerator with zero denominator is invalid")

empty_assistant = [("<bos>", "control"), ("<user>", "control"), ("hello", "user")]
empty_labels = labels_assistant_only(empty_assistant, len(empty_assistant))
assert ratio_from_positions(empty_labels, []) is None
```

示例中的 `all_checks_pass=False` 是有意的：它表达这个构造样本里模板不同、截断丢失 toy 协议规定的 EOS，且两处 placeholder 对应一组特征、不符合本例声明的一对一映射；这不是所有模型都应采用的媒体数量规则。同时，`assistant_mask_ok=True` 表示本例的 assistant 目标位置被保留，prompt 位置被忽略；共享 PAD/EOS ID 的小片段还展示按 token ID 相等处理可能屏蔽真实 EOS，而按 attention 位置处理可以保留它。这些断言只检查 toy 例子的预期状态，代码尚未运行；真实项目应按自己的协议保留失败样本、版本和修复动作，不因局部检查通过就把整条格式链路判为可用。

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

- 子词算法与工具边界：Sennrich 等人的 [BPE 论文](https://arxiv.org/abs/1508.07909)、Kudo 与 Richardson 的 [SentencePiece 论文](https://arxiv.org/abs/1808.06226)，以及 [Transformers v5.17 tokenizer 文档](https://huggingface.co/docs/transformers/v5.17.0/tokenizer_summary)。论文中的词表和语言结果不自动适用于当前 tokenizer。
- Tokenizer 指标证据：[ACL 2026 研究](https://aclanthology.org/2026.findings-emnlp.96/)分析约 200 个、覆盖 9 种语言的模型；它不支持把某个内在 tokenizer 指标当作跨任务性能保证。
- 模板与训练字段：[Transformers v5.17 chat template 文档](https://huggingface.co/docs/transformers/v5.17.0/chat_templating)和 [TRL v0.29 SFT 文档](https://huggingface.co/docs/trl/v0.29.0/en/sft_trainer)说明各自公开的模板和训练入口；行为仍须结合具体模型配置与样本回放。
- Mask 和 shift 实现边界：[Transformers v5.17 collator](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/data/data_collator.py#L3393-L3400)展示了按 `pad_token_id` 屏蔽标签的具体路径；同版 [GPT-2 模型头](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/gpt2/modeling_gpt2.py#L3227-L3232)说明该模型在内部 shift labels。它们是固定 revision 的实现例子，不代表所有 collator 或模型头。
- 工具调用与结构化输出：[OpenAI Function Calling](https://developers.openai.com/api/docs/guides/function-calling)和 [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)用于说明该 API 的公开 schema 边界；符合 schema 仍不等于业务动作安全。
- 多模态接口例子：[LLaVA](https://arxiv.org/abs/2304.08485)和 [BLIP-2](https://arxiv.org/abs/2301.12597)展示不同的视觉特征与语言模型连接设计，不能据此假定不同模型的 placeholder、feature 和 projector 兼容。
- Checkpoint 保存和加载：[PyTorch 教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)说明 state dict 与 optimizer state 等保存对象；张量能够加载不等于 tokenizer、模板和数据协议兼容。

资料只能支持公开的 tokenization、模板、字段和方法说明。它不能证明训练数据的 labels 一定正确、当前 serving 配置一定使用了同一模板，也不能从 API 成功响应推断模型内部如何处理隐藏状态。真正的结论必须来自固定样本、版本 manifest、token 序列、labels 和目标系统的回归结果。

## 3.18 本章小结

Tokenizer 和格式不是训练前的琐碎准备，而是模型协议。special token 定义协议边界，chat template 决定角色序列，labels 和 loss mask 决定哪些位置构成学习目标，attention mask 决定输入可见性，causal shift 决定预测与目标怎样对齐，截断策略决定哪些信息留下，多模态映射和工具 schema 则约束媒体与结构化动作如何进入系统。

排查这类问题时，最有效的方法是从一条原始消息开始，逐层保存渲染文本、token ID、decode、role、attention mask、loss mask、labels、截断区间和推理配置。先证明训练目标与推理协议兼容，再讨论模型能力；用最小样本和边界断言定位问题，再扩展到真实数据。只有协议映射、标签目标和停止规则都符合任务合同，格式问题才会从“模型很怪”变成可定位、可修复、可回归的工程问题。
