# 解码采样、重复控制与多路径推理：来源证据

记录日期：2026-10-29

本记录保存联网发现的新事实及来源边界。章节范围、纳入决定、正文修改和后续状态只记录在 `progress/`。推理时计算资源分配的 Snell 等人证据另见 `research/test-time-compute-and-evaluation-2026.md`。

## 1. 解码策略与开放式文本质量

- Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi, “The Curious Case of Neural Text Degeneration,” arXiv:1904.09751v2（2020-02-14；ICLR 2020）；[固定版本](https://arxiv.org/abs/1904.09751v2)。
  - 实际阅读范围：摘要、引言、§4.1 实验设置、§5 的重复与分布分析、§6 人工评估和结论。
  - 来源支持：作者报告解码策略可改变同一语言模型生成文本的质量，并提出 nucleus sampling，在动态概率质量集合内采样；论文展示其测试中高似然最大化解码出现平淡或重复文本、未截尾纯采样会从低置信长尾抽取 token。
  - 实验范围：核心 likelihood 实验用 GPT-2 Large（762M 参数）和 WebText；在 held-out WebText 文档前 1–40 token 条件下生成 5,000 段，单段到 end-of-document 或最多 200 token。论文也报告人工/统计 HUSE 设置。
  - 证据等级：source-backed（同行评审论文在指定模型、语料和开放式生成条件下的实验）。
  - 适用边界：不能外推为所有现代聊天模型、任务或服务端解码器的行为；论文测的是文本生成特性，不是事实正确率或当前模型的幻觉风险。beam search、temperature、top-k 与 top-p 的比较应限于作者实验口径。

## 2. Self-consistency 多路径聚合

- Xuezhi Wang et al., “Self-Consistency Improves Chain of Thought Reasoning in Language Models,” arXiv:2203.11171v4（2023-03-07；ICLR 2023）；[固定版本](https://arxiv.org/abs/2203.11171v4)。
  - 实际阅读范围：摘要、引言、§3.2 主结果、采样参数与运行设置。
  - 来源支持：方法先从 chain-of-thought 解码器采样不同推理路径，再对最终答案做 marginalization/一致性聚合。作者在若干算术及常识推理 benchmark 上报告相对单路径 CoT 的提升。
  - 实验口径：主结果跨 10 次运行平均，每次独立采样 40 个输出；模型包括 UL2-20B、LaMDA-137B、PaLM-540B 和 GPT-3，具体任务与采样配置并不相同。
  - 证据等级：source-backed（论文报告的模型、任务和提示条件下的 benchmark 结果）。
  - 适用边界：不能据此保证所有任务、多数票设置、模型或预算都提升；路径错误可能相关，一致答案不等于正确答案。增加采样和答案聚合会增加推理计算和成本。

## 3. Transformers generation 接口和重复惩罚实现

- Hugging Face Transformers 官方 [Generation 文档](https://huggingface.co/docs/transformers/main_classes/text_generation) 与 [logits_process.py](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py)，访问日期 2026-10-29。
  - 实际阅读范围：GenerationConfig 的 stop strings、EOS、max_new_tokens、temperature、top-k/top-p、min-p、top-h、typical-p、epsilon/eta cutoff 和 repetition_penalty 描述；当前源码中 RepetitionPenaltyLogitsProcessor 的文档和实现。
  - 来源支持：文档分别暴露模型输出 token ID、生成长度和 stop strings 等参数；当前接口还包括按最大 token 概率相对阈值、熵预算、局部典型性、绝对概率阈值与熵相关阈值筛选的参数。当前实现对每个已出现 token 至多应用一次惩罚。decoder-only 模型默认将提示中的 token 纳入惩罚历史。惩罚系数大于 1 时，正分数除以系数、负分数乘以系数；可配置忽略 prompt 长度。
  - 证据等级：source-backed（官方接口文档及代码）。
  - 适用边界：`main` 分支与当前文档会更新，不能代表所有 Transformers 版本或其他 runtime；实现公式、prompt 范围、默认值和 stop 语义应按实际依赖 revision 检查。该机制改变 logits，不保证整段无重复或整体质量提升。

## 4. Presence 与 frequency penalty 的产品定义

- OpenAI API 官方 [Advanced usage：Frequency and presence penalties](https://developers.openai.com/api/docs/guides/advanced-usage#frequency-and-presence-penalties) 及 [Chat Completions create reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)，访问日期 2026-10-29。
  - 实际阅读范围：惩罚项的 logits 公式与变量说明；Chat Completions stop 参数类型和支持说明。
- 来源支持：该 API 说明先前采样计数为 $c_j$ 时，将 logits 调整为 $\mu_j-c_j\alpha_f-\mathbf{1}(c_j>0)\alpha_p$。presence 项是出现过即一次扣减，frequency 项按先前计数线性扣减。stop 参数是字符串或字符串数组，但接口注明部分模型不支持。
  - 证据等级：source-backed（单一供应商的当前产品文档）。
  - 适用边界：这是指定 OpenAI API 的公开参数合同，不代表其他 provider 或推理框架同名参数具有相同公式、单位、取值范围或兼容模型；接口支持可能随模型与 API 版本变化。

## 5. 固定版本 assisted/speculative decoding 接口

- **Transformers v5.17.0**：[assisted decoding 文档](https://huggingface.co/docs/transformers/v5.17.0/assisted_decoding) 与 [Generation API](https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/text_generation)，访问日期 2026-10-29。默认 assistant-model 路线要求 target 与 assistant 使用相同 tokenizer，文档说明此路线支持 greedy 和 sampling、但不支持 batched inputs；Universal Assisted Decoding (UAD) 允许不同 tokenizer，通过 candidate re-encoding 和最长公共子序列对齐；self-speculation 与 MTP 仅能用于相应训练/配置支持的模型。以上为接口与方法说明，不表示目标模型 pair 可以自动兼容或获得加速。
- **vLLM v0.32.0**：[speculative decoding 文档](https://docs.vllm.ai/en/v0.32.0/features/speculative_decoding/)，访问日期 2026-10-29。文档列出 EAGLE、MTP、draft model、PARD、MLP、n-gram、suffix、hidden-state extraction 等方法，并指出收益随目标模型、QPS、硬件和请求分布而变；异词表 draft 路线 `use_heterogeneous_vocab` 仅适用于指定 draft-model 方法，且当前只支持 greedy draft sampling。版本文档的建议是候选选择入口，不是任何项目的性能保证。

## 6. 示例模型快照

- **来源与版本**：[EleutherAI/pythia-160m-deduped](https://huggingface.co/EleutherAI/pythia-160m-deduped/tree/582159a2dfe3e712a8d47ae83dec95ae3bde8e7e)，Hub commit `582159a2dfe3e712a8d47ae83dec95ae3bde8e7e`；[EleutherAI/pythia-70m-deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped/tree/e93a9faa9c77e5d09219f6c868bfc7a1bd65593c)，Hub commit `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`；Hub model API 的 `sha` 字段分别核对到上述快照，访问日期 2026-10-29。
- **官方说明**：[EleutherAI Pythia 项目](https://github.com/EleutherAI/pythia)说明该系列使用 GPT-NeoX tokenizer；两个 checkpoint 可作为同词表的 draft/target API 教学示例。
- **证据等级与边界**：source-backed（固定 Hub 模型快照与作者项目说明）。模型尺寸仅用于演示身份和 assisted-generation 调用，不是加速测试，也不支持性能或质量比较结论。

## 生命周期

解码策略、重复控制与多路径聚合：active。论文机制可用于教学；具体 API 和源码行为须绑定所用版本，效果须在目标任务、明确预算和评估口径下验证。Transformers v5.17.0 与 vLLM v0.32.0 的 assisted/speculative 接口事实绑定各自固定文档。

## 固定版本来源复核（2026-10-30）

### Transformers v5.20.0

- [Generation API](https://huggingface.co/docs/transformers/v5.20.0/en/main_classes/text_generation)，访问日期 2026-10-30。实际阅读 GenerationConfig 的 generation length、stop strings、temperature、top-k/top-p、min-p、top-h、typical-p、epsilon/eta cutoff、repetition penalty、length penalty 与 logits renormalization 参数说明。`top_h` 按概率排序，保留使归一化候选熵不高于原分布熵一定比例的最短前缀；epsilon 是绝对概率阈值，eta 在绝对阈值与熵相关阈值间取满足条件的候选。证据等级：source-backed（固定版官方 API 文档）。边界：只描述 v5.20.0 所公开的接口，不支持这些过滤器在不同任务上更优的质量结论，也不能外推到其他 runtime。
- [固定版源码 `logits_process.py`](https://github.com/huggingface/transformers/blob/v5.20.0/src/transformers/generation/logits_process.py)，访问日期 2026-10-30。`RepetitionPenaltyLogitsProcessor` 对每个已出现 token 至多处理一次；decoder-only 默认把 prompt token 纳入惩罚历史；系数大于 1 时，正 logit 除以系数、负 logit 乘以系数；`prompt_ignore_length` 可排除输入前缀。证据等级：source-backed（固定 tag 实现）。边界：不代表 Transformers 其他版本或其他框架。
- [Assisted decoding 文档](https://huggingface.co/docs/transformers/v5.20.0/assisted_decoding)，访问日期 2026-10-30。阅读 speculative assistant model、prompt lookup、self-speculative、MTP、UAD 和 static ensemble verification 段落。文档说明同 tokenizer assistant 路线、prompt lookup 的前缀匹配提案、MTP 对 checkpoint 专用预测层的依赖、UAD 的重编码对齐，以及 static ensemble verification 会把验证分布改成目标与草稿的混合分布。证据等级：source-backed（固定版官方指南）。边界：接口可用性不等于模型 pair 可兼容、批处理支持、目标 workload 加速或质量收益；混合验证的输出分布不等于原 target 分布。

### 停止、惩罚与延迟口径

- [OpenAI API Advanced Usage](https://developers.openai.com/api/docs/guides/advanced-usage#frequency-and-presence-penalties)，访问日期 2026-10-30。当前文档把 Chat Completions 与 Legacy Completions 的 presence/frequency penalty 公开为 logits 加法：`mu[j] - c[j] * alpha_frequency - 1(c[j] > 0) * alpha_presence`。证据等级：source-backed（单一供应商公开接口说明）。边界：不能推广为相同命名参数的跨供应商共识；取值、模型支持和 API 生命周期需另查。
- [OpenAI Completions create reference](https://developers.openai.com/api/reference/resources/completions/methods/create)，访问日期 2026-10-30。接口说明返回文本不包含匹配 stop sequence，并注明 stop 支持与模型有关。证据等级：source-backed（指定 API 参考页）。边界：不能外推至其他 API 的 stop 返回与终止语义。
- [vLLM v0.32.0 benchmark CLI](https://docs.vllm.ai/en/v0.32.0/benchmarking/cli/)，访问日期 2026-10-30。`vllm bench serve` 的 TTFT 为请求发送至首个流式输出的客户端侧时间；ITL 是连续流式输出事件间隔；请求级 TPOT 为 `(E2E latency - TTFT)/(output token count - 1)`。标准逐 token 流中 ITL 通常接近 TPOT；推测解码若一个流事件含多个 token，两者的采样单位和聚合权重不同。证据等级：source-backed（版本化 benchmark 测量定义）。边界：这是 vLLM benchmark 的客户端口径，非全行业统一名称或某模型性能保证。

### GNMT beam length normalization

- Wu et al., [Google's Neural Machine Translation System, arXiv:1609.08144v2 §7 Eq. (14)](https://arxiv.org/html/1609.08144v2#S7)，访问日期 2026-10-30。论文公开 `lp(Y)=((5+|Y|)^alpha)/(6^alpha)`，并将其用于 beam hypothesis 分数；同一分数还包含 coverage penalty。证据等级：source-backed（固定版本原论文）。边界：该公式是 GNMT 的翻译实现，不是通用 beam-search 标准；原实验的长度惩罚与 coverage 参数在特定机器翻译验证集上选择。

## 2026-11-02 固定接口与新增 speculative decoding 证据

### vLLM v0.32.0 采样和结构化输出接口

- 来源：[vLLM v0.32.0 SamplingParams](https://docs.vllm.ai/en/v0.32.0/api/vllm/sampling_params/) 与[结构化输出指南](https://docs.vllm.ai/en/v0.32.0/features/structured_outputs/)，访问日期 2026-11-02。实际阅读 SamplingParams 和 online serving 结构化输出段落。
- 来源支持的事实：固定版参数对象列有 temperature、top-k/top-p/min-p、presence/frequency/repetition penalty、seed、stop/stop token IDs、beam search 和 structured outputs。结构化输出指南公开 `structured_outputs` 接口，列出 choice、regex、JSON Schema、CFG grammar、structural tag，并说明该版本可使用 XGrammar 或 Guidance backend；不同 backend 的 regex 语法有差异，`auto` 会按请求和当前支持情况选择 backend。旧 `guided_*` 字段自 v0.12.0 起移除。
- 证据等级：source-backed（固定版本官方 API 文档）。边界：这是 v0.32.0 的接口合同，不能替代目标模型、tokenizer、schema 和 backend 的编译及输出回归；参数存在不等于效果、采样次序或跨版本兼容保证。

### 托管 API 的结构化输出与 reasoning effort

- 来源：OpenAI [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs) 和 [Reasoning guide](https://developers.openai.com/api/docs/guides/reasoning)，Google [Gemini thinking guide](https://ai.google.dev/gemini-api/docs/thinking)，访问日期 2026-11-02。实际阅读 JSON mode/strict schema、refusal/incomplete、reasoning effort 和 thought token limits/pricing 段落。
- 来源支持的事实：OpenAI 当前文档将 JSON mode 的合法 JSON 与 Structured Outputs 的受支持 JSON Schema 子集遵循区分；工具调用使用 function calling，Responses 可在省略 `strict` 时尝试规范化 schema，不兼容时回退到 best-effort 非 strict 行为，调用方仍须检查实际 strict 状态。Structured Outputs 还需处理 refusal、content-filter 和 incomplete response。reasoning effort 取值、默认值和可用字段依模型而定，降低 effort 可能偏向更快响应和较少 reasoning token。Gemini 文档说明 `max_output_tokens` 包含 thought token，达到硬上限可返回 incomplete，思考 token 也计入公开计费口径。
- 证据等级：source-backed（供应商当前公开产品合同）。边界：动态文档且不代表跨 provider 的统一参数含义或计费规则；预算和输出状态须按调用的具体 API、model 与日期核对。

### 论文版本与范围复核（访问日期 2026-11-02）

- **Locally typical sampling**：[Meister et al., TACL 2023](https://aclanthology.org/2023.tacl-1.7/)，实际阅读 ACL Anthology 摘要和论文元数据。作者以 token 信息量接近模型条件熵为动机；自动和人工评测报告在 abstractive summarization 与 story generation 上相对 nucleus/top-k 质量有竞争力，并减少所测设置中的退化重复。证据等级：同行评审论文结果（source-backed）。边界：这些任务结果不能外推为代码、工具参数、结构化输出或任意对话的默认优势。
- **Min-p 原始论文与复核**：[Turning Up the Heat: Min-p Sampling](https://arxiv.org/abs/2407.01082v8)，当前引用版本 v8（2025-11-20，ICLR 2025 oral），以及 [Schaeffer et al. critical analysis](https://arxiv.org/abs/2506.13681v2)，v2（2025-06-19）；实际阅读两篇摘要。原论文在 GPQA、GSM8K、AlpacaEval creative writing、Mistral/Llama 3 多种模型规模的设置中报告高温下的质量/多样性收益及人评偏好；批判性复核重新检查人评数据、统计、LLM-as-a-judge 报告和超参数预算公平，认为原证据未能支持其普遍优于 baseline 的主张。证据等级：原始论文作者实验与独立作者复核。边界：复核针对原研究证据，不证明 min-p 对所有模型/任务都无效；正文只把它当作另一种候选截断规则，具体质量需在同一任务/预算下验证。
- **XGrammar**：[XGrammar v3](https://arxiv.org/abs/2411.15100v3)，2025-05-12，MLSys ’25；实际阅读 §1、§3、§4.1–4.4 和硬件/模型配置。方法以 byte-level PDA 表示 CFG，按 stack state 预计算 context-independent token mask、运行时检查 context-dependent token，并用 persistent stack 支持分支/回滚和与 GPU 执行重叠。mask microbenchmark 在 Llama-3.1-8B-Instruct、AMD Ryzen 9 7950X + RTX 4090 上对比 Outlines v1.0、llama.cpp b3998、LM Format Enforcer v0.10.9；作者报告 JSON Schema 最多约 3×、unconstrained CFG JSON 超过 100×相对各自最佳 baseline 的 mask latency。论文另报告 SGLang/MLC-LLM/H100 等端到端设置。证据等级：论文指定 engine、grammar、模型和硬件下的实现及测量（source-backed）。边界：microbenchmark 的倍率不是统一端到端收益；模型、batch、grammar、CPU/GPU overlap 和 baseline release 会改变结果。
- **Tokenizer/grammar alignment**：[Flexible and Efficient Grammar-Constrained Decoding v2](https://arxiv.org/abs/2502.05111v2)，2025-07-15；实际阅读摘要。作者指出 subword tokenizer 与 CFG terminal 的 sound alignment 会增加离线预处理成本，报告其方法相对既有方案将常见 grammar 的 offline preprocessing 加速 17.71×，同时在线 mask computation 保持有竞争力。证据等级：预印本作者算法/实验（source-backed）。边界：17.71×指该实验的离线预处理，不是每 token mask、端到端延迟或生产 serving speedup。
- **独立 draft speculative sampling**：[Leviathan et al., arXiv:2211.17192v2](https://arxiv.org/abs/2211.17192v2)，ICML 2023 Oral；实际阅读 §2.1–2.3、§3.5、§4.1。方法先由 draft 分布 $q$ 提议，再按 $\min(1,p/q)$ 校正；拒绝时从 $\mathrm{norm}(\max(0,p-q))$ 采样，以保留调整后 target 分布。T5-XXL 11B 实测使用 batch 1、单 TPU-v4，translation 为 standard sampling 2.6×、greedy 3.4×，summarization 为 2.3×、3.1×，draft 包含 T5-small 77M 等。证据等级：原论文算法与作者所测 T5X/TPU 配置（source-backed）。边界：这些 speedup 假设和测量绑定 batch、设备、task、temperature、实现及 draft；不是任意 decoder-only service 的保证。
- **Medusa 与 EAGLE**：[Medusa v3](https://arxiv.org/abs/2401.10774v3)，2024-06-14；[EAGLE v3](https://arxiv.org/abs/2401.15077v3)，2025-03-04；实际阅读摘要。Medusa 增加多个 decoding heads 并用 tree attention 并行验证；Medusa-1 冻结 backbone，Medusa-2 联合微调 backbone，作者将 typical acceptance 描述为维持生成质量而非严格分布保持。EAGLE 自回归预测第二高层 feature，并前移 token 输入以处理 feature 不确定性；作者在 LLaMA2-Chat 70B 的实验报告 2.7–3.5× latency speedup、吞吐约翻倍且保持所称输出分布。证据等级：原论文方法和作者模型/任务实验。边界：训练要求、模型接口、硬件、任务和 verifier 均影响结果，不能泛化到任意 target。

### 精确采样、贪心接受与 batch correctness

- Sharma, [When Is a Draft Accepted? A Theory of Acceptance in Speculative Decoding](https://arxiv.org/abs/2606.30265)，arXiv 预印本，2026-06-29；访问日期 2026-11-02。实际阅读摘要、引言和接受条件章节。论文区分精确 target-distribution sampling 与 strict greedy、relaxed acceptance、top-m/tree 等确定性局部判据，给出在其分布差异假设下的接受证书，并以 Qwen3 输出分布评估证书。证据等级：source-backed（预印本理论结果与作者分析）。边界：接受条件的理论证书不等同于端到端加速、跨系统正确性或普遍质量收益；不能把 relaxed/typical acceptance 写成严格分布保持。
- Zhang et al., [Batch Speculative Decoding Done Right](https://arxiv.org/abs/2510.22876)，arXiv 首版 2025-10-26，访问日期 2026-11-02。阅读当前论文摘要与 introduction；当前版本题为 “Correctness Forensics for Batch Speculative Decoding: Diagnosing the Ragged Tensor Problem”，标注被 Findings of EMNLP 2026 接收。作者报告，在所审查的 batch speculative decoding 实现中，不同请求接受不同数量候选会形成 ragged tensor，若位置、mask 和 KV 状态未正确对齐会破坏输出等价性。证据等级：source-backed（论文的实现审查和作者实验）。边界：这不是对所有 runtime 的结论；greedy 的 token 级等价和 sampling 的分布等价须区分，具体同步与状态实现留给 runtime/调度主讲。
- Liu et al., [Speculative Decoding: Performance or Illusion?](https://arxiv.org/abs/2601.11580)，arXiv 首版 2025-12-31，访问日期 2026-11-02。实际阅读摘要、实验设置、§3、§5 和设备/模型配置表。作者在其 vLLM 配置中比较 n-gram、EAGLE/EAGLE-3、draft model 和 MTP，覆盖 CNN/DailyMail、ShareGPT、InstructCoder、GSM8K、AIME 22–24、GPQA-Main 六个数据集；目标模型为 Llama3.1-8B、Llama3-70B、Qwen3-8B/Thinking、GLM-4.5-Air，使用 H100。大部分非 reasoning 配置 batch 为 1–128；reasoning 配置因模型/数据集而为 1–16 或 1–4。作者在所测配置中报告 target verification 占执行时间约 42%–95%，相对加速通常随 batch 增大而收窄。证据等级：source-backed（论文指定 vLLM revision/config、模型、H100、请求和 batch 的测量）。边界：这不是跨硬件或所有 workload 的性能定律；文中输出长度也会随 batch 和实现改变，应同时检查正确性口径及目标环境 SLO。
