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
