# 解码策略与生成控制：资料驱动精修研究记录

日期：2026-10-26  
范围：第六册第四章《解码策略与生成控制》§4.1—§4.19  
状态：章节精修与声明范围内验收完成；结果及未验证边界见 `progress_v2.md` 对应记录

## 本轮研究问题

本章原稿已经覆盖 logits、greedy、beam、sampling、temperature、top-k/top-p/min-p、penalty、stop、JSON/schema、reasoning effort、speculative decoding、复现、调参、教学 demo 和工具任务。此次不是删去这些入口或另写一章，而是围绕原知识点把关键因果链和方案差异讲清：

1. top-p、locally typical、min-p 各按什么统计量挑 token？哪些结论来自方法定义，哪些只是作者在特定模型/任务上的质量主张？
2. Grammar-constrained decoding 如何从 JSON schema/CFG 状态得到下一步合法 token？子词 token、grammar 初始化和每步 mask 为什么会形成系统开销？
3. speculative sampling 如何通过接受/拒绝与 residual correction 保持 target 的下一个 token 分布？“同分布”与“同 seed 输出相同”有什么区别？
4. 独立 draft、prompt lookup、Medusa、EAGLE 和 checkpoint MTP 分别解决什么提案成本问题？它们的训练、显存、正确性和 workload 约束是什么？
5. 研究原型中的单请求加速能否直接迁移到高 batch serving？需要看哪些执行阶段和分母？
6. 公开 API 的 JSON、schema、reasoning budget、stop 和采样参数到底承诺了什么，哪些仍须由宿主应用检查？

## 代表来源与证据边界

| 来源 | 阅读/核对后的支持内容 | 不能据此推出 |
| --- | --- | --- |
| [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) | nucleus sampling 的动机：按当前分布累计质量自适应保留候选，回应概率最大化生成的重复/枯燥问题。 | 不能保证某个 top-p 值适合所有模型、温度或任务。 |
| [Locally Typical Sampling](https://aclanthology.org/2023.tacl-1.7/) | token 的 surprisal 接近本步条件熵时被视为局部 typical；作者在摘要/故事生成设置比较 nucleus/top-k，并报告质量有竞争力和较少退化重复。 | 不证明聊天、代码、工具调用等任务普遍更好；质量结论绑论文数据和评价。 |
| [Turning Up the Heat: Min-p Sampling](https://arxiv.org/abs/2407.01082) | min-p 将概率阈值按最大 token 概率缩放；原论文在 GPQA、GSM8K、创意写作等设置报告高温度下优势。 | “高温下更好”不是跨模型/任务/调参预算的定律。 |
| [Min-p, Max Exaggeration: A Critical Analysis of Min-p Sampling in Language Models](https://arxiv.org/abs/2506.13681) | 作者重新检查 min-p 原论文的人评数据、显著性检验、NLP benchmark 超参数 sweep 和 judge 结果，认为现有证据不足以支持 min-p 普遍优于基本采样或 top-p。 | 这是批判性复核论文的结论，不是 min-p 在所有应用中无效的证明；正文并列报告双方及其争议。 |
| [Hugging Face GenerationConfig](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation) | 当前公开 generation 参数包括 top-k/top-p/min-p/typical-p/eta/epsilon、penalty、length penalty、beam 和 logit renormalization；各参数由具体 generation config 与版本决定。 | main 文档和模型仓库 generation_config 是动态对象，不能当作每个模型的统一默认值。 |
| [vLLM SamplingParams](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/) | 当前服务接口暴露 temperature、top-k/top-p/min-p、penalty、seed、stop 和结构化输出等参数。 | 参数存在不代表任意组合均兼容；例如当前代码对 speculative decoding 与 min-p、logit bias 等组合有约束，须绑定 release 验证。 |
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) | 给出 draft q、target p、接受概率 min(1,p/q) 和拒绝后的正残差分布；一次 target verification 可提交一段 draft prefix 和一个 correction/bonus token。作者在 T5-XXL 11B 翻译/摘要实验报告约 2.3–3.4 倍加速。 | 分布保持依赖算法和分布定义；速度数字绑定论文模型、batch、任务、框架及硬件，不能外推目标 serving。 |
| [Decoding Speculative Decoding](https://arxiv.org/abs/2402.01528) | 350 余项 LLaMA-65B / OPT-66B 实验讨论 draft selection；作者发现 draft latency 对收益关键，draft 的语言建模能力与 speculative 加速并非简单正相关。 | 论文结论不能推广到所有新模型、硬件、batch 和运行时。 |
| [Medusa](https://arxiv.org/abs/2401.10774) | 在 target 上增加多个 prediction head、用 tree attention 并行验证分支；Medusa-1 冻结 backbone，Medusa-2 联合训练。论文区分严格 rejection sampling 与速度更高但只主张相近生成质量的 typical acceptance。 | “无损”/“相近质量”不能混用；加速数字受模型、采样、硬件和候选树影响。 |
| [EAGLE](https://arxiv.org/abs/2401.15077) 和 [项目代码](https://github.com/SafeAILab/EAGLE) | 以 target 倒数第二层 feature 为 draft 对象，并通过前移 token 处理 feature 不确定性；论文报告 LLaMA2-Chat 70B 的 2.7–3.5 倍 latency speedup 和约两倍吞吐。 | 数字不能跨模型、任务、temperature、batch、实现和硬件比较；feature head 依赖目标模型内部接口。 |
| [Transformers assisted decoding](https://huggingface.co/docs/transformers/assisted_decoding) | 官方文档区分独立 assistant、prompt lookup、self-speculative、MTP 和 universal assisted decoding；prompt lookup 从输入中的重叠 n-gram 提案，不需独立模型。 | 文档支持矩阵属于其版本和模型接口，不代表其它引擎可用。 |
| [Speculative Decoding: Performance or Illusion?](https://arxiv.org/abs/2601.11580) 与[作者维护的 benchmark 页面](https://specdecode-bench.github.io/) | 在 vLLM 上比较 n-gram、EAGLE/EAGLE-3、draft model 和 MTP；模型覆盖 Llama 3.1-8B、Llama 3-70B、Qwen3-8B、GLM-4.5-Air-106B，六类 workload，H100，batch 1—128（部分 profiling 至 512）。作者报告相对加速随 batch 上升普遍变小；target verification 在所测配置约占 42%—95% 执行时间，且方法与任务存在交互。 | 不可把其结果当成当前 vLLM 其它版本、不同 GPU、target/workload 或 SLO 下的通用倍率。 |
| [XGrammar](https://arxiv.org/abs/2411.15100) 与[官方实现文档](https://github.com/mlc-ai/xgrammar/tree/main/docs) | CFG/PDA 解析、context-independent token mask 预处理、context-dependent token 运行时检查、persistent stack 分支/回滚以及 grammar 与 GPU 推理重叠。论文在 Llama-3.1-8B-Instruct、Ryzen 9 7950X + RTX 4090 的约束引擎测量中报告 JSON Schema 最多约 3 倍、CFG JSON 超过 100 倍于各自最佳对照的 token-mask 速度。 | 约束引擎延迟与端到端服务延迟不是同一指标；论文最高倍率不能外推到所有 grammar/backend/任务。 |
| [Flexible and Efficient Grammar-Constrained Decoding](https://arxiv.org/abs/2502.05111) | 重点解决 CFG 和 subword tokenizer 边界的 sound alignment；报告 common grammar offline preprocessing 比先前方法快 17.71 倍、在线 mask 仍具竞争力。 | 17.71 倍是离线 preprocessing，不是在线 TPOT 或 serving throughput。 |
| [vLLM structured outputs](https://docs.vllm.ai/en/stable/api/vllm/config/structured_outputs/) | 核对 backend 选择以及不同 JSON Schema 支持边界。 | “支持 JSON Schema”不等同于支持 JSON Schema 全标准，也不说明所有 release 后端相同。 |
| [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) 与 [function calling API reference](https://platform.openai.com/docs/api-reference/chat/object) | 当前公开 API 区分 JSON mode（有效 JSON）与 strict Structured Outputs（支持的 schema 子集）；拒答、长度截断/不完整响应仍需处理；function arguments 仍须宿主验证。 | 是 OpenAI API 合同，不代表其他产品或开源引擎。 |
| [OpenAI reasoning_effort API reference](https://platform.openai.com/docs/api-reference/chat/object) 与 [Gemini thinking guide](https://ai.google.dev/gemini-api/docs/thinking) | 两家公开 API 将 effort/thinking 作为控制面；Gemini 文档说明 max_output_tokens 含 thought tokens，触顶可返回 incomplete。 | 参数名、默认档位、支持模型、计费与内部机制不可跨厂商等同。 |

## 代表方案对照

### 随机截断策略

| 方法 | 候选集合规则 | 解决的问题 | 证据/限制 |
| --- | --- | --- | --- |
| top-p | 按 p_i 降序，取累计质量达到 p 的最小前缀 | 候选数量随当前分布变化 | 原论文提供方法动机；具体质量取决于温度、p、任务和模型。 |
| locally typical | 按 |−log(p_i)−H(p)| 从小到大，累计到 typical_p | 关注 token 信息量是否接近条件熵 | TACL 论文在摘要/故事生成报告结果，不能当通用对话结论。 |
| min-p | 保留 p_i ≥ alpha × max(p) | 用当前最可能 token 的概率缩放阈值 | 原论文优越性主张受到 2025 复核质疑；作为可比较规则，不预设更优。 |
| eta/epsilon cutoff | 绝对概率底线，eta 还结合熵尺度 | 提供不同于相对排名/累计质量的截断基线 | 本章保留术语入口和官方参数文档，不展开算法推导。 |

教学分布 p=(0.40,0.30,0.20,0.10) 的 entropy 约为 1.280 nat。Locally typical 的 surprise 距离顺序为 0.30、0.20、0.40、0.10；typical_p=0.40 因而保留 0.30 和 0.20。top-p=0.40 只保留 0.40；min-p=0.30 的绝对门限为 0.12，保留 0.40、0.30、0.20。例子只说明集合差异，不说明哪种输出质量更高。

### 语法约束路线

XGrammar 的核心加速是把大部分可由局部 grammar 位置决定的 token mask 提前缓存，把需要 parser stack 的少数 token 留到运行时判定，再用 persistent stack 支持分支/回滚，并尝试将 CPU grammar work 与 GPU generation 重叠。Flexible and Efficient Grammar-Constrained Decoding 更直接地针对 tokenizer/subword 与 CFG terminal 边界做安全对齐、压缩 preprocessing。两篇工作分别优化 token mask执行路径和 token/grammar 预处理；17.71x preprocessing 与 100x token-mask headline 不可横向排名。

### Speculative decoding 路线

| 方法 | proposer | 主要优势 | 主要代价/边界 |
| --- | --- | --- | --- |
| Independent draft | 单独的小自回归模型 | 不要求改 target 架构；标准 rejection correction 可保持 target 分布 | draft 权重、KV、算力和 draft/target 接近程度影响收益。 |
| Prompt lookup / n-gram | 输入 prompt 中重叠子串 | 不需要独立 draft 模型，输入重复度高时很便宜 | 依赖 workload 重复和可复制片段，无匹配时退化到常规 decode。 |
| Medusa | target 上额外 token heads + tree candidates | 免维护独立 draft，多个分支同时 target verify | 需要额外 heads 和训练；严格拒绝采样与 typical acceptance 的分布口径不同。 |
| EAGLE | target hidden feature 上的 drafter | 借 feature-level 规律提高 proposal acceptance，并保留 target verifier | 配套训练、hidden-state 接口、候选/KV 状态及架构依赖。 |
| MTP | checkpoint 自带 future-token heads | 复用已有 checkpoint 附加参数 | 仅适用带兼容 MTP 权重的模型；引擎、batch、verifier 仍决定收益。 |

用 p=(0.50,0.30,0.20)、q=(0.20,0.50,0.30) 可核验单步 rejection correction：B 的接受概率为 0.30/0.50=0.60；被拒后的正残差只在 A 上有质量。最终 A、B、C 的质量分别回到 0.50、0.30、0.20。期望提交 token 数要把每个位置联合接受前缀的概率相加，不能直接把条件接受率相加；即 E[N_committed]=1+Σ_j P(前 j 个草稿 token 均接受)，假设该轮没有 EOS/stop 边界。

## 原正文知识点覆盖图

| 原位置 | 已有知识及本轮处理 |
| --- | --- |
| §4.1 | logits/softmax/候选集层次和定义域保留；新增 generation 参数/processor 顺序应绑定实现版本的来源提醒。 |
| §4.2 | greedy 局部 argmax、稳定性和事实正确率边界保留。 |
| §4.3 | beam 累计分数、length penalty 定义域、KV 和开放式任务边界保留。 |
| §4.4 | sampling 随机性和任务范围保留。 |
| §4.5 | temperature 公式、熵、任务边界保留。 |
| §4.6 | top-k/top-p/min-p 原公式保留；加 typical 机制与手算、min-p 正反研究、策略对照；eta/epsilon 仍留根并提供原论文/接口入口。 |
| §4.7 | presence/frequency/repetition 公式、符号敏感实现与过度惩罚失败保留。 |
| §4.8 | EOS/stop/max token 与 finish reason 表保留。 |
| §4.9 | JSON/schema/semantic 三层和工具状态机保留；新增从 grammar state 到合法 token mask 的轨迹、tokenizer 边界、XGrammar 与 sound alignment 路线比较、OpenAI schema/refusal/incomplete 边界。 |
| §4.10 | reasoning effort 与随机性区分、成本账本保留；新增当前 OpenAI/Gemini 官方参数语义并标明不可跨厂商等同。 |
| §4.11 | draft/target 流程、接受长度、成本直觉和边界保留；修正接受长度公式中 a_j 的定义；新增精确采样数值轨迹、同分布≠同 seed、draft source 方法比较及真实 batch 评测边界。 |
| §4.12 | seed、batch、统计 pass rate 和回归记录保留。 |
| §4.13 | workload 默认起点表保留，仍明确为实验起点而非处方。 |
| §4.14 | tuning 流程、目标向量和失败样本分类保留。 |
| §4.15 | 零依赖 logits demo 未删除；将按当前代码完整运行并对照输出块。 |
| §4.16 | 常见误解与边界保留。 |
| §4.17 | 报销工具的完整协议/权限/业务校验案例保留。 |
| §4.18—§4.19 | 来源清单改正 typical 论文正式标题，补 min-p 争议、约束引擎、speculative 方法与官方接口；总结扩充本轮方法差异。 |

## 已完成的声明范围内验收

- §4.15 零依赖 Python demo 已运行，确定性输出与正文 fenced text 一致；不代表真实模型或 serving runtime。
- 手算 locally typical 熵和 distance 排序、top-p/min-p/typical 候选集合，以及 speculative p/q 接受和 residual correction 的最终边际分布；检查期望提交长度使用联合接受前缀概率。
- 检查 §4.1—§4.19 章节顺序、代码/数学围栏、公式定义和表格列数；第六册目录已有本章入口。
- 打开核对本章新增来源；eta-sampling 论文题名和方法支持已确认。来源记录区分论文、官方接口和工程评测，并标明动态文档及实测迁移边界。
- 同步 `PAPERS.md`、`GLOSSARY_EN_ZH.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`；验收细节见 `progress_v2.md`。
- `git diff --check` 通过。未运行真实模型/GPU/线上 serving、人评复现、外部公式渲染或 PDF 构建。

## 未验证边界

没有真实模型、GPU、目标 serving runtime、production trace 或受控人评复现。Min-p、typical、grammar engine 和 speculative 数字只属于已列论文/文档条件；纸上概率轨迹验证算法，不验证 runtime 数值实现。动态 Hugging Face/vLLM/OpenAI/Gemini 文档代表访问日所见页面，部署时须按实际模型、endpoint、SDK 和服务版本复查。没有外部公式渲染或 PDF 构建。
