# 第五册第三章：Tokenizer 与数据格式深度精修研究记录

## 研究范围与问题图谱

- `date`：2026-10-26。
- `chapter`：`book-05-llm-training/chapters/03-tokenizer与数据格式.md`，全文 3.0—3.21。
- `goal`：保留原章所有主要问题路径，补足算法推演、研究对照、训练协议和工程边界；不把 tokenizer 质量缩减为单一压缩率，也不把 JSON 字段名当作训练语义。
- `research_questions`：
  1. BPE、Unigram、byte-level BPE 和 byte fallback 分别如何训练、编码和处理未见输入？
  2. 词表大小如何同时改变序列长度、embedding/输出层、训练 FLOPs、上下文预算和多语公平？
  3. 为什么研究对“压缩率是否预测模型效果”的结论不同；如何区分因果消融、跨模型相关性和具体下游任务？
  4. special token、chat template、EOS/PAD、assistant-only label、label shift 和 packing 如何共同定义实际训练协议？
  5. 词表扩展、多模态序列化、tokenizer 版本迁移和代码审计器有哪些同步边界？

## 旧知识点覆盖图

原有 3.0—3.21 章节结构、主题和练习均保留。原有短期知识点仍在对应主题下深化，没有删除原有读者问题；新增讨论作为研究和工程证据补充。

| 原有主题 | 本轮去向 | 说明 |
| --- | --- | --- |
| Tokenizer 输入/输出、不变量、id、规范化 | 3.1.1—3.1.4 | 增加 tokenizer manifest、NFKC 可逆边界、原文/规范化文本双存和任务特定 normalization 证据。 |
| 字符/整词/子词权衡、BPE | 3.2—3.3 | 增加 BPE 手算与 tie-break，解释 merge 拟合的是预分词后局部共现，不是词根语义。 |
| SentencePiece、Unigram、子词采样 | 3.4 | 保留算法差异；补 NMT 原论文的收益和 n-best 解码限制，不将 BLEU 直接推广到现代 LLM。 |
| Byte-level BPE、byte fallback | 3.5 | 拆分两种覆盖设计，说明任意字节覆盖、Unicode/UTF-8 边界和流式解码。 |
| 词表规模、fertility、压缩率、多语言公平 | 3.6 | 增加参数/logit 账本、研究结果差异、受控与观察性证据对照、MUTANT recipe 和指标边界。 |
| special token、EOS、PAD、mask、Harmony 事件 | 3.7 | 保留旧协议主题；细分 message end、turn end、handoff，并使用 gpt-oss Harmony 的明确协议示例。 |
| Chat template、SFT label mask、多轮与工具数据 | 3.8—3.9 | 增加模板作为可执行协议、框架选项差异、标签右移表和 token/sample loss reduction 算例。 |
| 预训练 EOS、packing、截断 | 3.10 | 加独立 SFT 样本的 block-causal mask、边界 label 和 cumulative sequence length 机制。 |
| SFT、偏好、RL/可验证数据与工具轨迹 | 3.11—3.13 | 保留原 schema 主题；补字段口径、共享 prompt tokenization、rollout provenance 和运行时协议边界。 |
| 多模态占位与 Tokenizer 扩展 | 3.14—3.15 | 增加分辨率/patch/frame 手算、音频码数与 AR 步数区别、词表扩展研究和同步对象。 |
| 数据格式审计代码、故障、评估和练习 | 3.16—3.21 | 保留并增强 toy auditor、回归用例、来源清单和第八项固定预算比较练习；代码未执行。 |

## 代表性研究与证据

### 子词算法、规范化与采样

- [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/) 用 BPE 子词组合处理开放词汇 NMT 的罕见词问题。它支持 BPE 对未登录整词的退化表示机制；实验是 NMT，不等同于现代 decoder-only LLM 的最佳 tokenizer 证明。
- [SentencePiece](https://aclanthology.org/D18-2012/) 是从原始文本训练子词模型的工具体系，支持 BPE 和 Unigram LM；论文具体说明空白处理、可配置 normalization（默认描述为 NFKC）和文本还原约束。工具名不代表唯一算法，也不能把其默认行为迁移给其他 tokenizer。
- [Subword Regularization](https://aclanthology.org/P18-1007/) 在 NMT 训练中为同一字符串采样多种合法切分。论文报告多种 IWSLT 低资源语言对约 1—2 BLEU 的提升，WMT14 English→Czech 是未观察到相同增益的例外；若训练未见多切分而解码执行 n-best segmentation，部分方向会退化。正文把它限定为 NMT 证据。
- [From Bytes to Subwords: Challenges of Input Representations in NLP](https://aclanthology.org/2026.findings-acl.530/) 汇总当时 Hugging Face 热门模型中 tokenizer pipeline 的重复度，并用 2,307 种语言、每种 1,000 个样本和 1—5 字符 n-gram Naive Bayes 做 normalization case study。十个 split 下原文模型均优于 NFC 版本，平均准确率 91.75% vs. 91.70%；这是幅度很小的语言识别结果，不是 LLM 预训练质量结论。它支持“按任务验证 normalization”，不能证明任意规范化都有害。

### 压缩、tokenizer 选择和模型任务表现

- [Unpacking Tokenization](https://arxiv.org/abs/2403.06265) 改变 BPE tokenizer 训练语料规模，在英语和土耳其语上训练 10M、128M、1B 模型。作者在其评测中观察到 compression 与下游性能相关，较小模型和生成任务中的相关性更强；模型、语言和任务有限，不能由相关性推出所有 tokenizer 的压缩增益都会转成质量增益。
- [Tokenization Is More Than Compression](https://arxiv.org/abs/2402.18376) 比较预分词、词表构造和算法，并用 PathPiece 检查最短 tokenization 是否足够。350M 模型、10 项任务、三种 vocab size 下没有算法统计显著地全面胜出；不同模型规模的排名会变，初始化和 pre-tokenization 也影响结果。结论支持多维评估，不代表压缩没有价值。
- [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425) 比较 17 种 tokenizer 的平行语义文本，报告部分语言的 token 长度差异最高约 15 倍；字符/字节表示也未消除所有差距。该比例是论文数据和 tokenizer 快照，用于说明公平审计问题，不是当前任意产品的常量。
- [Tokenizer Choice for LLM Training: Negligible or Crucial?](https://aclanthology.org/2024.findings-naacl.247/) 在 70B-word 训练语料（单语英语、多语英语/德语/法语/意大利语/西班牙语）上训练 24 个 2.6B decoder-only LMs，每模型最多约 52.6B tokens；其 tokenizer 设计、库、词表和算法做受控消融。英语中心 tokenizer 处理论文多语文档时，作者报告最高 68% 额外训练计算（对照为 Multi-UNI-SP-50 与 EN-UNI-SP-100 的 German documents），而 fertility/parity 不能稳定预测全部任务。限制包括只测试零样本、未为每个 tokenizer 分别调参、无多随机种子、模型最多 2.6B、语言主要是拉丁字母的印欧语；不能将 68% 外推到所有语言、模型或数据预算。
- [Objective vs. Search: Decomposing What Makes a Good Tokeniser](https://aclanthology.org/2026.emnlp-main.2307/) 把算法的两个易混变量拆开：compression vs. unigram log-likelihood objective，以及 bottom-up merge vs. top-down pruning search。作者新增 BottomUpLL 和 TopDownComp，和 BPE/Unigram 形成 2×2。论文训练 100M—1B 英语模型及英语、德语、西语、土耳其语、中文 1B 多语模型；bottom-up 在多数条件下 BPB 更低，但 BLiMP / 多语最小对语法任务没有稳定排序；小词表下 objective 也仍有作用。Top-down deletion score 是近似而 bottom-up merge gain 为精确；300M、500M 和多语 1B 部分只有单 seed，最大 1B，语言五种且大多高资源。它解决了“BPE 优于 Unigram 就一定是 compression objective 更好”的混杂比较问题，不建立所有任务下 bottom-up 恒优的规则。

### 评测指标、观察性证据和多语方案

- [No Single Tokenizer Feature Reliably Predicts Downstream Language Model Performance](https://aclanthology.org/2026.findings-emnlp.96/) 分析约 200 个、九种语言的模型和多类 benchmark；作者未发现一个 intrinsic tokenizer metric 能跨模型家族、任务和语言稳定预测结果。新提出的语言学指标也需要按模型家族和语言继续验证。它是跨模型观察，不等于 tokenizer 选择无因果影响。
- [TokLens](https://aclanthology.org/2026.acl-srw.18/) 提供 fertility、characters/token、bytes/token、normalized sequence length、single-token retention rate 和 parity 六类指标；测 24 个 tokenizer、15 种语言。控制模型规模后未见 intrinsic metrics 与英语 benchmark 的稳定关系，但在七个模型、十一种 MMLU-ProX 语言的混合效应分析中，STRR 等指标与单语表现相关；Qwen2.5 家族的每语言 scaling slope 与 STRR 相关。作者明确将多语结果视作相关证据，并承认语言预训练数据组成仍然混杂。它与“没有一个全局稳定预测器”并不矛盾：跨家族外推、控制变量和具体多语 benchmark 不同。
- [MUTANT: A Recipe for Multilingual Tokenizer Design](https://aclanthology.org/2026.acl-long.2146/) 解决的问题不是简单替换 BPE，而是同时设计语料、脚本/语言预分词、subword stage 和 multiword stage。作者以约 10GB 多语文本训练 200K shared vocabulary；在 22 个印度语言、英语和代码中，平均 fertility 相对 LLaMA-4 与 Sutra 分别低 39.5% 和 18%。其两种 1B 模型按 50B tokens 和匹配超参训练；inference 对照用 200 个多语请求、5 个 warmup、10 轮、8×H100、Triton、最多 256 个生成 token，报告 LLaMA-4 OTPT 117.99 tokens/s、MUTANT-Indic 169.42 tokens/s，TTFT 19.17ms 与 18.98ms。局限：下游模型最高 1B、限定印度语言/英语/代码，不能外推到其他语系和大模型；token 定义也随 tokenizer 改变，因此 tokens/s 不能单独代表用户可见文本吞吐，比较应另看 bytes/s、completion time 和任务质量。

上述研究的结论属于不同证据类型：同预算重训有更强的 tokenizer 因果证据，但模型/任务覆盖较窄；跨模型观察覆盖较广，却受模型规模、数据配比和训练 recipe 混杂。内在压缩指标用于发现覆盖、token 长度和预算问题；外在 task evaluation 才能判断特定模型/数据/目标下的效果。章节不把不同论文裸分排序。

## 官方文档、代码和框架实现

- [OpenAI tiktoken](https://github.com/openai/tiktoken) 与 [GPT-2 byte encoder](https://github.com/openai/gpt-2/blob/master/src/encoder.py)：作为 byte-level BPE 的具体代码入口；实现细节属于对应仓库版本。
- [SentencePiece options](https://github.com/google/sentencepiece/blob/master/doc/options.md)：byte fallback 配置与 SentencePiece API 设计；byte fallback 不是全输入 byte-level BPE。
- [Hugging Face Chat Templates](https://huggingface.co/docs/transformers/main/chat_templating)：`add_generation_prompt` 与 `continue_final_message` 的区别、两者互斥、二次 tokenize 时避免重复 BOS/EOS 的框架接口说明。`main` 文档会变化，必须按实际 Transformers version 复核。
- [Hugging Face Tokenizer docs](https://huggingface.co/docs/transformers/main/main_classes/tokenizer) 与 [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)：special token sanitization、assistant-only loss/generation tags、pretokenized labels / `-100` 约定；这些行为依版本和模板。
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)：`ignore_index` 对逐位置目标的作用，不能据此把 `-100` 当模型 vocab token。
- [NVIDIA NeMo 25.02 sequence packing](https://docs.nvidia.com/nemo-framework/user-guide/25.02/sft_peft/packed_sequence.html)：一个用 cumulative sequence lengths / packed attention 避免样本间注意力的 SFT 实现；并非通用数据布局标准。
- [OpenAI gpt-oss Harmony response format](https://developers.openai.com/cookbook/articles/openai-harmony)：gpt-oss 特定协议把 `<|end|>`、`<|return|>` 和 `<|call|>` 用于 message end、assistant stop 和工具 handoff；正文明确不把 token 名称视为通用标准。
- [MUTANT GitHub](https://github.com/ola-krutrim/MUTANT)：训练 recipe 和评估代码入口。当前正文只引用作者论文数字，没有宣称本项目运行过该仓库。

## 正文变化

1. 3.1.2 新增 tokenizer fingerprint/manifest 字段，涵盖 normalizer、pre-tokenizer、词表/merge ranks、special token、template、processor、padding、truncation、mask 和 packing。
2. 3.1.4 把 `N(x)` 规范化放进 tokenizer 链路，说明 NFKC 对“恢复原文”的影响并保留 raw/normalized 双版本；补 2026 multilingual language classification 结果，明确不外推到 LLM 预训练。
3. 3.3.1 用 `aba×2, abac×1, ac×1` 手算 BPE：初始 `a-b=3,b-a=3,a-c=2`，并列时选 `a-b`，得到 `ab-a=3,a-c=2`，第二轮合并为 `aba`；修正原例第二轮错误计数并写明 tie-break 假设。
4. 3.4.4 介绍 Unigram/BPE-dropout 切分采样和 NMT 实验范围，强调训练和 n-best inference 要匹配。
5. 3.5 拆开 byte-level BPE 与 Unicode subword + byte fallback 的覆盖路径，增加 UTF-8 partial chunk 的 streaming 解码边界。
6. 3.6.4 用 `d=4096`、`32K→128K` 计算 embedding `131.072M→524.288M` 参数；以 `L=1000→600` 说明 `L×V` 输出位置-词表代理量从 32M 到 76.8M（2.4×）。明确不是完整 FLOPs。
7. 3.6.6 对照 Unpacking、PathPiece、Petrov、Tokenizer Choice、Objective vs. Search；另解释 No Single 与 TokLens 相关性结果范围，并用 MUTANT 连接数据、预分词、multiword recipe、推理基准和 tokens/s 的分母陷阱。
8. 3.7 增加字面 special string 与注册控制 token 的输入风险；以 gpt-oss Harmony 为特定实例区分消息完成、回合 stop 和工具 handoff。
9. 3.7.3 将有效 token mask 改为基于右 padding 长度；区分 padding、causal、document 和 assistant-role/loss masks，处理 PAD=EOS。
10. 3.8.4 增加 generation prompt/continue-final-message 和二次 tokenize 造成重复 special tokens 的框架 contract。
11. 3.9.3 以 token 表展示 `logits[t] -> labels[t+1]`，补 assistant answer 首 token、EOS 监督、shift ownership，以及 token-average vs. sample-average 的 10-token/100-token 推演。
12. 3.10 增加 SFT 独立样本 packing 的 block-causal 矩阵、EOS 无法重置 attention 的原因、首 token label 边界和 `sum_i s_i^2` 的变长计算口径。
13. 3.11—3.15 保留 JSON/SFT/偏好/RL/工具/图像音频/扩词主题，补 pairwise prompt/token 对齐、rollout provenance、视觉 patch 和 audio code/AR 步数的条件账本，以及 FOCUS/2026 Hindi 扩词表研究边界。
14. 3.16 在 toy auditor 中加入固定 byte vocabulary、roundtrip、按原始长度构造 padding mask、packing EOS、角色/工具控制 token 注册和词表扩行；明确不是完整 BPE、生产处理器或 document attention。
15. 3.18—3.20 加入固定协议回归样例和第八项受控 tokenizer comparison 练习；增加 BPB 分母与 BOS/EOS 对齐要求。

## 数值与静态核对

- BPE 手算：初始 pair counts `a-b=3`、`b-a=3`、`a-c=2`；假定平局选 `a-b` 后，`aba×2` 贡献 `ab-a=2`，`abac×1` 贡献 `ab-a=1`、`a-c=1`，`ac×1` 贡献 `a-c=1`，因此第二轮 `ab-a=3`、`a-c=2`，可合并为 `aba`。
- Parameter arithmetic：`32,000×4,096=131,072,000`，`128,000×4,096=524,288,000`。
- Logit proxy：`1,000×32,000=32,000,000`，`600×128,000=76,800,000`，ratio `2.4×`；只计位置×类别数，不是端到端训练 FLOPs。
- 固定 toy strings：英文 5 tokens / 28 characters；中文 4 / 10 characters、30 UTF-8 bytes；代码 13 / 36 characters。对应 `0.179, 0.4, 0.133, 0.361` 均为该教学 vocab 下的精确示例，不是一般语言排名。
- 图像示例：`448/14=32`，`32²=1,024` patch；2×2 merge 后 `16²=256`；16 帧 ×256=`4,096` 个视觉位置。参数仅作教学假设。
- Loss reduction：样本 A 有 10 tokens/loss 1，样本 B 有 100 tokens/loss 2；token average `(10+200)/110≈1.91`，sample average `(1+2)/2=1.5`。
- BPB 需要同一规范化原文 UTF-8 byte denominator、相同 special boundary（特别是 BOS/EOS）和可逆/对齐一致性；不能拿不同分母的每-token PPL 直接比较。

## 代码、验证与未完成项

- 修改并静态审阅 3.16 toy Python 审计器：固定 256 个 byte-fallback tokens、编码/解码、token id 检查、assistant-only labels、长度 mask、packing 形状、EOS 数和新增工具 token 均有对应断言；修复非字符串 role 在 set membership 时报 `TypeError` 的输入边界。
- 依本轮执行限制，未运行章节代码、Python 测试、论文实验、目标 tokenizer 统计、外部公式渲染或 PDF 构建；代码的运行状态不是已验证。后续进度应保留这个未完成项，不能写成 execution verified。
- 静态核对公式变量、BPE 手算、标题序号、代码 fence、公式 fence、章节目录映射和新增外链。`git diff --check` 通过；未运行自动化测试。
- 框架 `main` 文档和 GitHub `master` 链接会变化；正文只说明其版本相关接口，实际使用时应记录真实库版本和 tokenizer hash。

## 本轮来源入口

- Algorithms：[BPE rare words](https://aclanthology.org/P16-1162/)、[SentencePiece](https://aclanthology.org/D18-2012/)、[Subword Regularization](https://aclanthology.org/P18-1007/)
- Tokenizer performance：[Unpacking Tokenization](https://arxiv.org/abs/2403.06265)、[Tokenization Is More Than Compression](https://arxiv.org/abs/2402.18376)、[Tokenizer Choice for LLM Training](https://aclanthology.org/2024.findings-naacl.247/)、[Objective vs. Search](https://aclanthology.org/2026.emnlp-main.2307/)
- Multilingual evaluation/design：[Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425)、[From Bytes to Subwords](https://aclanthology.org/2026.findings-acl.530/)、[No Single Tokenizer Feature](https://aclanthology.org/2026.findings-emnlp.96/)、[TokLens](https://aclanthology.org/2026.acl-srw.18/)、[MUTANT](https://aclanthology.org/2026.acl-long.2146/)
- Vocabulary extension：[FOCUS](https://arxiv.org/abs/2305.14481)、[Beyond Initialization Loss](https://arxiv.org/abs/2608.03494)
- Protocol / implementation：[OpenAI Harmony](https://developers.openai.com/cookbook/articles/openai-harmony)、[Hugging Face Chat Templates](https://huggingface.co/docs/transformers/main/chat_templating)、[TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)、[NeMo packing](https://docs.nvidia.com/nemo-framework/user-guide/25.02/sft_peft/packed_sequence.html)、[SentencePiece options](https://github.com/google/sentencepiece/blob/master/doc/options.md)
