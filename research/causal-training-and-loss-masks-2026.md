# 因果训练目标、监督遮罩与生成前缀：事实和证据

检索日期：2026-10-28。

## 因果可见性与标签错位

- Vaswani 等人的 Transformer 原论文 v7（2023-08-02）§3.1 说明 decoder 的输出嵌入错开一位，并与自注意力遮罩共同保证位置 i 的预测只能依赖先前已知输出；§3.2.3 说明对非法的后续位置在 softmax 前设为负无穷。来源：[论文 HTML §3.1](https://arxiv.org/html/1706.03762v7#S3.SS1)、[§3.2.3](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS3)。证据等级：原始论文中的架构定义。边界：支持原论文 decoder 的设计和机制，不表示所有框架公开的 mask API 使用相同张量形状或布尔语义。
- 本记录中以 C_{ij}=0（j≤i）、C_{ij}=-∞（j>i）表示的加性因果遮罩，是对上述机制的数学写法；softmax 后被屏蔽位置的权重为零。它不规定 padding 位置或目标标签是否计入 loss。

## Causal LM 与 Masked LM

- Devlin 等人的 BERT 论文 v2（2019-05-24）摘要与 §3.1 说明，BERT 通过 masked language modeling 选择输入中的部分 token 作为预测目标，并使用左右上下文恢复其原词；论文将该目标用于双向表示预训练。来源：[论文 HTML §3.1](https://arxiv.org/html/1810.04805v2#S3.SS1)。证据等级：原始论文对 BERT 目标及其设计的定义。边界：说明 BERT 的训练目标，不等同于自回归下一个 token 生成；具体遮盖替换比例属于该论文的输入构造方案，不应据此概括所有 Masked LM。

## Teacher forcing 与训练/生成前缀

- Bengio 等人的 Scheduled Sampling 论文 v3（2015-09-23）§1、§2.1–2.4 描述序列预测中的训练/推理差异：训练条件使用真实的前一目标 token，推理时改用模型选择或采样出的 token；论文讨论偏离后的生成状态可能导致后续错误，并在 RNN 序列任务上提出 scheduled sampling。来源：[论文 HTML §1](https://arxiv.org/html/1506.03099v3#S1)、[§2.4](https://arxiv.org/html/1506.03099v3#S2.SS4)。证据等级：原始论文的研究动机与方法描述。边界：论文研究的是其 RNN 序列预测任务；它支持“真实前缀与模型生成前缀不同”这一概念来源，不证明现代 LLM 的每次长文本失败都由 exposure bias 导致，也不证明该论文方法可普遍修复此问题。

## 对话模板与 assistant-only loss 的版本行为

- Hugging Face Transformers v5.17.0 官方聊天模板文档说明，不同模型可把同一组角色/内容消息格式化为不同控制 token 序列；模板可能处理 BOS/EOS 特殊 token，且添加 generation prompt 会改变 assistant 响应的起始边界。来源：[Chat templates](https://huggingface.co/docs/transformers/v5.17.0/chat_templating)，阅读位置：“Chat templates”概述、特殊 token 提示及 add_generation_prompt。证据等级：固定版本官方文档。边界：适用于该文档所描述的 Transformers 接口，不定义任意模型服务的停止条件或所有聊天协议。
- Hugging Face TRL v0.29.0 SFTTrainer 文档说明，assistant_only_loss=True 可用于仅对 assistant 响应计算 loss、忽略 system/user 内容；该功能要求聊天模板能借助 generation 标记返回 assistant token mask。该版本指标定义将 loss 描述为对未被 mask 的 token 计算的交叉熵平均。来源：[SFTTrainer 文档](https://huggingface.co/docs/trl/v0.29.0/en/sft_trainer)，阅读位置：“Train on assistant messages only”及训练指标字段。证据等级：固定版本官方文档。边界：这是 TRL v0.29.0 的配置与模板要求示例，不构成其他训练库的统一参数、标签移位约定或 reduction 规则。

## 未验证边界

- 未运行任何训练代码、框架 API 或模型服务；未验证特定 tokenizer 的结束 token、实际停止配置、padding 约定、位置 ID、label shift 所在层级或分布式 loss reduction。
- 本记录中的手算概率、token ID 和变长 batch 是构造的教学输入，不是论文实验、模型测量或生产行为证据。

## 2026-11-19 补充：Prefix LM、packed attention 与框架遮罩语义

- **本次范围与检索：** 目标是解释因果、自回归、双向编码、Prefix/Seq2Seq 可见性模式，以及 padding、loss 和 packed-sequence 边界的不同职责。检索日期 2026-11-19；使用 Web 搜索查询 `Attention Is All You Need masked multi-head attention decoder section 3.2 original paper`、`UniLM unified language model prefix language modeling attention mask original paper`、`FlashAttention varlen attention cu_seqlens packed sequences block diagonal causal mask official documentation`、`D2L transformer causal mask prefix language model attention mask Stanford CS224N lecture transformer mask teaching`、PyTorch 官方 Transformer/SDPA boolean mask 语义；英语来源为主。直接核对论文、官方框架文档及课程讲义，未以搜索摘要支持定义。
- **Transformer causal mask 与 label shift：** Vaswani 等，*Attention Is All You Need*，[NeurIPS 2017 论文 PDF](https://papers.neurips.cc/paper/7181-attention-isall-you-need.pdf)，阅读 §3.1、§3.2.3。Decoder 的 self-attention 可读取当前位置及之前位置；目标嵌入右移一位后，位置 $i$ 的预测只依赖已知的前序目标。注意力遮罩在 softmax 前把非法连接分数设为 $-\infty$。因此“遮罩是否屏蔽对角线”必须连同输入/标签对齐约定说明：在以当前输入 token 为 query 的常见写法中可见集合是 $j\le i$，该位置的 logit 预测下一个 token；若示意图用目标词位置对齐，矩阵可能画成 $j<i$。证据是原论文架构定义，不定义各框架张量 API。
- **双向与 Prefix/Seq2Seq 可见性：** Dong 等，*Unified Language Model Pre-training for Natural Language Understanding and Generation*，[NeurIPS 2019 论文 PDF](https://proceedings.neurips.cc/paper/2019/file/c20bb2d9a50d5ac1f713f8b34d9aac5a-Paper.pdf)，阅读 §2 与 Figure 1。UniLM 共享 Transformer 参数，并通过不同 self-attention mask 表达 unidirectional、bidirectional 和 sequence-to-sequence 上下文；Seq2Seq 模式中 source segment 双向互见，target segment 可以读完整 source 和 target 中当前位置及之前位置。其训练采用 cloze token replacement 和跨任务目标；因此这个可见性矩阵不能单独等同于“训练标签总是普通 shifted next-token loss”。证据边界：代表 UniLM 的目标设计，不表示所有命名为 Prefix LM 的方法都有相同监督方式。
- **Attention mask 与 padding/loss 的职责及 API 语义：** PyTorch 2.15 [Transformer API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Transformer.html) 区分结构性 `src/tgt/memory_mask` 与 key padding mask；其中布尔遮罩的 `True` 表示不允许参与。PyTorch 2.15 [scaled_dot_product_attention API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html) 则规定布尔 `True` 表示参与 attention，语义与 `nn.Transformer` 相反。它们是固定版本 API 行为，不可概括为通用 mask 编码。注意力可见性控制信息读取；loss/label mask 控制哪些目标计入目标函数；不能用前者代替后者。
- **Packed variable-length attention：** PyTorch [Variable Length Attention 教程](https://docs.pytorch.org/tutorials/intermediate/variable_length_attention_tutorial.html)，标注版本 PyTorch 2.15.0+cu132，阅读 packing 图及 `varlen_attn` 定义/示例。累积序列边界张量（如 `cu_seq_q`、`cu_seq_k`）表达扁平化后的逻辑文档边界，kernel 可以在不填充的 packed tensor 上按片段处理；教程用 `window_size=(-1,0)` 表达 causal、`(-1,-1)` 表达 full attention。此教程支持所述 API 版本和示例，不说明所有 packed training 实现都自动处理 loss、位置 ID 或 sample-specific position reset。
- **独立边界的推论：** 若目标是把 packed 样本当作互不相干的训练序列，block-diagonal causal visibility 可以防止样本间的 hidden-state 读取；按 shifted next-token loss 构造标签时，还需检查跨边界转移是否计入 loss。后一点是由 label shift 与样本独立性推出的审计问题，并非上述 PyTorch 教程的显式保证；实际目标应按数据集/训练目标确认。
- **教学组织样本：** Stanford CS224N 2021 Lecture 9，[官方讲义 PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/slides/cs224n-2021-lecture09-transformers.pdf)，阅读 pp. 20–22：先提出 decoder 不能查看未来，再将非法 score 设为 $-\infty$，说明该做法避免逐 timestep 重建 key/query 集合并保留并行化。其 mask 图采用的对角线约定须与原 Transformer 的右移输入一并解释。该讲义是教学组织实例，不是学习效果对照研究。
- **Encoder-only 与双向 MLM：** Devlin 等，*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*，[arXiv v2](https://arxiv.org/html/1810.04805v2)，阅读摘要和 §3.1。论文用随机 mask 的 token prediction 预训练深层双向表示；证据支持该模型的目标和上下文可见性，不支持“所有双向模型更擅长所有理解任务”这类普遍性能结论。
- **Encoder-decoder 去噪：** Lewis 等，*BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*，[ACL Anthology PDF](https://aclanthology.org/2020.acl-main.703.pdf)，阅读摘要、§1、Figure 1。BART 把损坏文本送入双向 encoder，再用 left-to-right decoder 重建原文本；架构形态和去噪目标分别说明，不能只凭 mask 图决定训练目标。
- **UL2 与目标族：** Tay 等，*Unifying Language Learning Paradigms*，[arXiv v1](https://arxiv.org/html/2205.05131v1)，阅读摘要、§§2–3。作者明确把架构类别与预训练目标分开，并提出 mixture-of-denoisers 与模式切换；该研究可支持“可见性、被预测内容和架构是关联但不同的设计维度”，不要求本章复述所有 R/S/X 目标或把它们归为单一 attention mask。
- **局部窗口案例：** Jiang 等，*Mistral 7B*，[arXiv v1](https://arxiv.org/html/2310.06825v1)，阅读 §2。Mistral 7B v0.1 报告使用 GQA 与 sliding-window attention，表列 `window_size=4096`、`context_len=8192`，并说明层间信息传播可能越过单层窗口。此证据绑定该论文版本和配置，不用于概括所有 Mistral 版本或宣称窗口大小等于有效上下文能力。
- **尚未验证：** 没有运行 PyTorch API 或 kernel；未比较框架版本、硬件 backend、tokenizer EOS 分段策略和生产 packed-data label policy；也未测量不同 mask 排布的性能。

## 2026-11-19 补充：Decoder-only 的原始预训练实例

- **GPT-1 原始报告：** Radford 等，*Improving Language Understanding by Generative Pre-Training*，[OpenAI 托管论文 PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)，阅读摘要、§3.1 和公式 (1)。§3.1 将预训练目标写成每个 token 在此前上下文条件下的语言模型似然，并说明实验使用多层 Transformer decoder。该来源支持将 GPT-1 作为 decoder-only 语言模型的代表实例；将其因果读取解释为前序条件分解的结构含义，是对目标公式的直接推论。来源不规定当前框架的 mask API，也不代表所有 GPT 版本采用相同的数据或训练配方。
