# Decoder-only 与通用 LLM：研究记录

检索日期：2026-11-20。研究问题：哪些结构和训练/使用接口特点解释 decoder-only 在通用生成型 LLM 中的主流地位；哪些证据限制“它普遍更好”的结论。研究对象是文本模型的架构、预训练目标与任务接口，不试图制作 2026 年全部模型的架构清单或性能排行榜。

## 外部主题与证据边界

| 主题 | 来源发现 | 边界与未支持的推断 |
| --- | --- | --- |
| 架构与目标可分离 | T5 对 encoder-decoder、causal LM、prefix LM 结构及 denoising/LM 目标做组合比较；Wang 等分别比较三种架构、两种目标及 multitask prompted finetuning；UL2 明确提出分开讨论 architecture archetypes 与 objective，并混合多个 denoising paradigm。 | 常见搭配是历史经验，不是定义；不能从“decoder-only”一词直接推出损失只落在全部 token、也不能从结构单独推出任务能力。 |
| 三类信息流 | 原始 Transformer 是 seq2seq encoder-decoder；T5 延续双向编码、因果解码和 cross-attention；causal decoder 可用单一序列主干把给定前缀与输出接起来。 | 架构只说明内部连接和可见范围，不能替代 tokenizer、task serialization、loss mask 或推理策略的说明。 |
| 生成型通用接口 | GPT-3 论文报告 175B autoregressive LM 在不做梯度更新的情况下，使用文本交互和 few-shot demonstrations 完成多种任务，包括翻译、问答和 cloze；作者也报告失败任务及 web-corpus 方法风险。 | 单篇模型论文不能证明纯架构造成了能力、所有任务都适合生成标签，或 zero-/few-shot 表现普遍可靠。 |
| 架构比较的条件性 | T5 论文在其 text-to-text 方案及比较配置中发现 encoder-decoder + denoising 表现最好；论文说明比较时参数量与 FLOPs 不能同时简单对齐。Wang 等的实验中 causal decoder-only + autoregressive LM 在纯无监督 zero-shot 最强，而 non-causal input + masked LM 经 multitask finetuning 后在其比较中最好。 | 结果绑定各自训练 recipe、参数/计算对齐方式、任务集、调优和评估；不能把某一组结果外推为永久排序。 |
| 近期尺度比较 | Zhang 等 2025-10-30 的 arXiv 预印本比较约 150M—8B 的 RedLLM encoder-decoder prefix LM 与 decoder-only causal LM；使用 RedPajama V1 约 1.6T tokens 预训练并用 FLAN 指令微调。作者报告 decoder-only 预训练的 compute-optimal 前沿较强；encoder-decoder 微调后多任务可比或部分更好，并在其测量设置下有推理效率优势。 | 这是单个团队、特定数据/架构 recipe、约 8B 上限的预印本；作者承认不同任务结果有差异。它不能确定全行业选择原因，也不能代表现有最大模型、硬件或全部 serving 负载。 |

## 已查阅来源

### 论文、报告与官方资料

1. Ashish Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，NeurIPS 2017；arXiv 页面显示 2017-06-12 v1、2023-08-02 v7。查阅 arXiv HTML 页面摘要及序列转换背景（访问日期 2026-11-20）。支持原始 Transformer 的 encoder-decoder 定位和论文的机器翻译设置；不支持 decoder-only 优越性结论。
2. Colin Raffel et al., [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)，JMLR 2020；使用 arXiv v4 页面（最近修订 2023-09-19）。查阅摘要、§3.2 架构实验/比较条件及结果（HTML §3.2，特别是结构和表 2，页面行约 221—300；访问日期 2026-11-20）。论文说明在其设置下 encoder-decoder denoising 的任务分数最好，且参数量/计算量需分别对齐；这不是对现代所有 LLM 的全局比较。
3. Tom B. Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)，2020；使用 arXiv v4 页面（2020-07-22）。查阅摘要和 GPT-3 few-shot 设置（访问日期 2026-11-20）。支持 GPT-3 175B 在论文所测任务上的纯文本 few-shot 实验与失败边界；不能单独归因于 decoder-only 架构。
4. Thomas Wang et al., [What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?](https://arxiv.org/abs/2204.05832)，发表于 PMLR 162 / ICML 2022；使用 arXiv v1（2022-04-12）摘要及 PMLR 论文比较设置。实验覆盖 causal/non-causal decoder-only 与 encoder-decoder、autoregressive/MLM 目标，超 5B 参数、170B tokens，及无/有多任务提示微调条件。支持结论随 objective 与 finetuning 改变；不证明某类架构在当前全部任务最优。
5. Yi Tay et al., [UL2: Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131)，2022；使用 arXiv v3 页面（最近修订 2023-02-28），查阅摘要。支持把架构类型和预训练目标分开讨论、组合多种 denoising paradigm；本文仅用它说明设计轴可组合，不把摘要中的 benchmark 比较当作现代架构排名。
6. Biao Zhang et al., [Encoder-Decoder or Decoder-Only? Revisiting Encoder-Decoder Large Language Model](https://arxiv.org/abs/2510.26622)，arXiv 预印本，2025-10-30。查阅摘要、§1、§3、§5—6（HTML 页面约 0—195 行；访问日期 2026-11-20）。用于记录近期受控比较和作者报告的 trade-off，保留预印本、数据、模型尺寸、训练与评测条件。

### 教学与解释组织来源

7. Stanford CS224N 2024 archived lecture 09, [Pretraining encoder-decoders: what pretraining objective to use?](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/slides/cs224n-2024-lecture09-pretraining.pdf)，课程讲义。查阅 slides 41—44 和 T5 结果/任务示例（PDF 页内约 810—859、860—930、1011—1025 行；访问日期 2026-11-20）。其教学顺序先比较架构可见范围，再讲预训练目标和 span corruption，随后呈现实验；用于组织解释，不把讲义简化的典型搭配当成定义。
8. Stanford CS25 V4, [Hyung Won Chung of OpenAI: Transformers architecture and scaling perspective](https://www.youtube.com/watch?v=orDKvo8h71o)，课程记录标记为 2024-04-11；课程归档页见 [CS25 archive](https://web.stanford.edu/class/archive/cs/cs25/cs25.1246/)。查阅时段 15:18—23:29 的课程描述及可访问转录片段（[转录文本](https://lawwu.github.io/transcripts/transcript_orDKvo8h71o.html)，访问日期 2026-11-20）。讲者明确把若干解释标为个人、强观点；这里只借鉴从三类结构、信息路径到 scaling 假设的教学路线，不把讲者对历史趋势的观点当成因果证据。
9. Yee Seng Chan, [Pretraining objectives: why decoder-only won](https://yeesengchan.com/posts/series/research-foundations-of-modern-llms/01-pretraining-objectives/index.html)，2026-03-18。查阅“unification”部分及编码器边界（页面约 114—153 行；访问日期 2026-11-20）。作者提出“预训练—提示—生成—部署接口对齐”的解释，并强调 encoder 仍适合表示类任务。此为作者综合解释和教学视角，不是独立因果评估；正文关键事实以论文实验为准。
10. Hugging Face, [How Transformers solve tasks](https://huggingface.co/learn/llm-course/en/chapter1/5?fw=pt)，官方在线课程（未显示固定修订号，访问日期 2026-11-20）。查阅 encoder-only、decoder-only、encoder-decoder 三类任务映射段（页面约 127—129 行）。用于核对面向初学者的模型家族入口；其概览式“适合任务”表述不作为硬性能力边界。

## 交叉来源得出的边界

- “通用 LLM”指一个主干可用统一接口覆盖多种开放式文本生成任务，不等于它在所有检索、分类、输入密集抽取、翻译或短输出任务上成本最低。
- 统一 next-token 训练接口可以减少对每种任务单独设计输出结构的依赖，但实际数据仍需要合理序列化、任务信息、监督位置和评估；目标统一不意味着能力、成功标准或可靠性统一。
- GPT-3 的规模化 few-shot 结果与 decoder-only 的流行时间线一致，但现有这些来源没有把架构、数据、算力、模型规模、提示格式、后训练和生态影响完全因果隔离。
- 对比架构时须分别报告可见范围、目标函数、参数量、训练计算/token、后训练、任务格式、输出长度与评估口径。不要用单一参数量、单个 benchmark 或理论 FLOPs 判定普遍优劣。
