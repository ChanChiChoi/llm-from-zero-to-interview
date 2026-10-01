# Qwen3.6-35B-A3B：榜单锚点、Thinking Preservation 与 Agent RL 环境报告

核验日期：2026-09-28。模型锚点来自 Artificial Analysis；DataCurve 只用于检查是否存在精确 Agent 评测行。技术事实以 Qwen 官方仓库、ModelScope 模型卡/配置/chat template 和 Qwen 发布博客为准。没有把同家族其他 checkpoint 的结构、分数或技术迁移到本模型。

## 1. 榜单身份与评测边界

Artificial Analysis 收录同一模型的两个配置：[`qwen3-6-35b-a3b`](https://artificialanalysis.ai/models/qwen3-6-35b-a3b)（Reasoning）和 [`qwen3-6-35b-a3b-non-reasoning`](https://artificialanalysis.ai/models/qwen3-6-35b-a3b-non-reasoning），日期均为 2026-04-16。它们按一个 Qwen3.6-35B-A3B 锚点归并，不按两种推理配置重复计模。

- AA 中文榜单快照：1,781,428 bytes，SHA-256 `15e9763f0d516791e7b7c2c5679e87e5878eaffed4bef31e485c9043b758ba59`。
- AA Reasoning 详情：4,056,617 bytes，SHA-256 `120de76d2630e0fe777fa10bb82ec7b484f4c33ef2a7487085c489f4158e6391`。页面列出的 262K context、约 18 的 Intelligence Index 及 provider 参数、速度和价格属于 AA/provider 口径；官方卡写总参数 35B、激活参数 3B。AA 的舍入数字不替代模型卡规格。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 快照：268,036 bytes，SHA-256 `14436c31be1e50a0b62171e4ee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`。本快照未检出 Qwen3.6 精确行；不把 Qwen3.5、Qwen3.8 或其他 Qwen 的 Agent 分数迁移过来。

## 2. 权威来源与固定版本

1. [QwenLM/Qwen3.8 README](https://github.com/QwenLM/Qwen3.8/blob/2ea10dc725823bf7c3e21ce8557cbe15245132ae/README.md)：Qwen 官方仓库固定 commit `2ea10dc725823bf7c3e21ce8557cbe15245132ae` 的快照，14,334 bytes / SHA-256 `a71ec46607f81d6056336fb0a8431a26a1c7d8db6ac568a0c021c36f1ed3c92e`。README 说明 Qwen3.6-35B-A3B 于 2026-04-16 发布，并链接官方发布页。
2. [Qwen 官方 ModelScope 模型卡](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B)：README revision `913c459c5c83fa016a0e54a52e5b95f6c894e0fe`，64,550 bytes / SHA-256 `c4ddaa065649ff6352648f64747a16eda31726f3e34add94ce04abb461c77b75`。模型卡将其定位为带 vision encoder 的 causal language model，并给出参数、层布局、MTP 与上下文信息。
3. 固定 ModelScope artifact revision `1a5ae24e867f8d82388070d3f61590158a01d15c`：[config.json](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B/resolve/1a5ae24e867f8d82388070d3f61590158a01d15c/config.json)，3,686 bytes / SHA-256 `93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`；[chat_template.jinja](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B/resolve/1a5ae24e867f8d82388070d3f61590158a01d15c/chat_template.jinja)，7,764 bytes / SHA-256 `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`。chat template 是解释历史 thinking 保留条件的直接实现证据。
4. [Qwen3.6-35B-A3B 官方博客](https://qwen.ai/blog?id=qwen3.6-35b-a3b)：直开博客路由是客户端 shell；通过页面公开调用的 [Qwen 文章 API](https://qwen.ai/api/v2/article/?language=zh-CN&path=qwen3.6-35b-a3b&type=qwen_ai) 于 2026-09-28 经 `10.24.27.134:7890` 取得 HTTP 200，JSON 94,678 bytes / SHA-256 `d287402f27a6ffa3226466aa57407310f670eb73a6d72bcfef03597a94e35d49`（含动态 `request_id`）。嵌入 HTML 正文 91,941 bytes / SHA-256 `706889d145ea17b8c8234c4cda35b00fdecc0b6bcb9e1f5f20d2ed3ff9e15ed1`。元数据标题《Qwen3.6-35B-A3B：智能体编程利器，现已开源》，作者 Qwen Team，发布日期/修改时间 `2026-04-15T10:00:00+08:00`。AA 与 QwenLM/Qwen3.8 README 标注 2026-04-16；保留各来源日期，不擅自调和。
5. [arXiv 精确检索](https://arxiv.org/search/?query=%22Qwen3.6-35B-A3B%22&searchtype=all)：较早的一次 HTTP 200 检索快照为 56,117 bytes / SHA-256 `5a9a97e4a2e559991a76a6c607cefc650bb83f058d818370d1137691a5222466`，当时未列出专属报告；该历史快照已被下述 2026-09-23 v1 论文取代，不能再据此写“未检出报告”。
6. [arXiv:2609.27321v1《Verifiable Hidden Dynamics Play: Generating Agentic RL Environments from Solved Mechanisms》](https://arxiv.org/abs/2609.27321v1)：HTML 正文为 `Qwen Technical Report` 评论，发布日期 2026-09-23。它公开的是 Agentic RL 环境生成与 Qwen3.6-35B-A3B 后训练实验，不是该基础模型的完整预训练配方。正文全量 HTML 快照 461,978 bytes / SHA-256 `da56349817e450cde05eeed3be1cf2274217cec17a849f4111c697e270cee28a`；7890 代理取得的 arXiv 摘要页 43,529 bytes / SHA-256 `34bfd8e7f8b4931a008b3f1b35771f748ac00da837681be3861abaedad7a6e2b`。全量正文经 1234 保存；本轮 7890 对摘要页返回 HTTP 200，但请求 /html 时代理连接失败。未使用不完整 PDF。

## 3. 公开架构：混合线性注意力、显式注意力与 MoE

官方模型卡列出 35B 总参数、3B 激活参数、hidden size 2,048、40 层、256 experts（每 token 8 routed + 1 shared）、expert intermediate size 512，以及带 vision encoder 的模型形态。40 层布局为：

```text
10 × [3 × (Gated DeltaNet → MoE) + 1 × (Gated Attention → MoE)]
```

Gated DeltaNet 配置为 V 方向 32 heads、QK 方向 16 heads、head dimension 128；Gated Attention 为 16 Q / 2 KV heads、head dimension 256、RoPE dimension 64。模型卡还称 MTP 采用 multi-step training。公开信息足以说明“递归/线性状态层与周期性显式 attention 混合，再接稀疏专家”的结构；不足以推导 vision encoder 内部结构、完整训练配方或生产 kernel。

要避免跨版本误归因：Qwen3.8 Flash-Next 的 QSA、Gated Residual、N-gram embedding、Muon 等细节不因同属 Qwen 家族而自动适用于 Qwen3.6。

## 4. Thinking Preservation：历史推理块的序列化策略

模型卡称 Qwen3.6 经额外训练以保留并利用历史消息中的 thinking traces。固定 chat template 将行为实现为一个历史消息渲染开关：

- 默认 `preserve_thinking` 未启用时，模板只在最近一条 user 消息之后保留 assistant reasoning block；更早轮次的 reasoning 内容不渲染进本次提示词。
- `preserve_thinking=True` 时，模板把历史 assistant 的 `<think>…</think>` block 一并保留。模型卡称这有利于 Agent 场景下维持决策一致性，并可能减少重复推理、改善 KV cache 利用率；这是发布方能力/效率说明，不是本项目独立测出的收益。
- `enable_thinking` 是另一个开关：它决定生成提示词是否以空的 `<think>` block 结束，控制当前生成模式；`preserve_thinking` 处理输入历史里已有 reasoning block 的保留。不能把“历史思考进入 prompt”说成“本轮启用思考”，也不能把二者合并成一个 memory 开关。
- API 参数路径依 surface 不同：本地 vLLM/SGLang 示例使用 `chat_template_kwargs`；Alibaba Cloud Model Studio 示例使用顶层 `enable_thinking` / `preserve_thinking`。迁移时应按 endpoint/schema 做 capability check，不能机械复用请求体。

最重要的概念边界：`preserve_thinking` 只影响调用方提供的对话历史如何渲染、哪些 reasoning 文本进入当前 prompt。它不创建跨会话存储，不是数据库/workspace，也不会在 transcript 被丢弃后自行记住内容。较长历史会增加输入 token、占用 context/KV，并带来前缀变化和缓存失配等工程权衡；是否更省 token，取决于减少的重复推理是否抵消被保留历史的成本。

## 5. 长上下文、YaRN 与 MTP serving

官方卡给出的 native context 为 262,144 tokens，并以 YaRN `factor=4.0` 示例扩展至约 1,010,000；README 建议只在任务长度超出 native window 时使用 RoPE scaling/YaRN。卡片特别提醒主流框架采用 static YaRN 时 scaling factor 不随输入长度变化，可能影响较短文本，因此不应默认把 1M 外推配置当作短文本无代价的设置。外推窗口也不等于该长度下的可靠召回或服务商 API 上限。

README 分别给出 vLLM 与 SGLang 的 MTP serving 示例。不同引擎的参数合同不是可直接比较的性能数据；本轮未下载完整权重、安装 serving runtime、跑 speculative decoding 或做 GPU profiling。

## 6. 发布博客的 Agent benchmark：结果绑定任务、执行器与 judge

Qwen 官方博客报告的若干 coding/Agent 结果可与表中被测模型直接对照；它们都是发布方数字，不是本项目复现：

| Benchmark | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B | 博客披露的关键条件 |
|---|---:|---:|---|
| SWE-bench Verified | 70.0 | 73.4 | 内部 bash + file-edit agent scaffold；temperature 1.0、top-p 0.95、200K context |
| SWE-bench Pro | 44.6 | 49.5 | 部分公开任务经修订，所有基线在 refined set 重跑 |
| Terminal-Bench 2.0 | 40.5 | 51.5 | Harbor/Terminus-2；3 h timeout、32 CPU、48 GB RAM、256K context、最多 80K 输出、5 次均值 |
| SkillsBench Avg5 | 4.4 | 28.7 | OpenCode；78 个 self-contained 任务，排除 API-dependent tasks；5 次均值 |
| QwenClawBench | 47.7 | 52.6 | 发布方称其为 internal real-user-distribution Claw benchmark；博客当时称将开源，非可独立重建数据 |
| NL2Repo | 20.5 | 29.4 | 对比模型用 Claude Code，最高 900 turns |

表格没有证明参数稀疏度或单一架构改动造成分数变化；任务集修订、Agent scaffold、模型执行轮次和重复次数都是结果定义的一部分。尤其一些 benchmark 的 evaluator/user model 也属于系统：博客称 TAU3-Bench 使用 GPT-5.2 low-reasoning 作为 user model 并采用默认 BM25 retrieval；VITA-Bench 用 Claude 4 Sonnet 作 judge，因为当时官方指定的 Claude 3.7 Sonnet 已不可用；MCPMark 固定 GitHub MCP v0.30.3 且把 Playwright 响应截断到 32K tokens；MCP-Atlas 使用公开集与 Gemini 2.5 Pro judge。这里的 GPT/Claude/Gemini 仅是 Qwen 论文式评测里的依赖项，不是从排行榜发现的新锚点或本轮扩展的厂商范围。

QwenWebBench 又是另一类量：双语前端任务先自动渲染，再由多模态 judge 评估代码与视觉正确性，并以 Bradley–Terry/Elo 汇总；Elo 不等于 verifier pass rate。面试应追问 judge 版本、任务修订、工具/模拟器版本、输出截断和 run 数是否锁定，以及这些依赖变化时基线是否重跑。博客列出的评测脚注适用于其发布结果，不等同公开数据/完整 verifier 已开放。

## 7. 发布别名与客户端预算边界

官方博客称该开源 Qwen3.6-35B-A3B checkpoint 可通过阿里云百炼的 `qwen3.6-flash` 名称调用，并示例 OpenAI-compatible Chat Completions/Responses 与 Anthropic-compatible API。它是已由 Artificial Analysis 发现的 35B-A3B checkpoint 的 hosted/API 名称证据，不因字符串 `Flash` 新增一个模型锚点；未调用真实 endpoint，不能据此确认当前账户、地区或完整 API contract 可用。

博客给 OpenClaw 的例子使用 `contextWindow=131072`、`maxTokens=16384`。这是客户端/harness 配置，低于模型卡的 262,144 native context；文档没有说明其取值原因，也没有证明百炼服务端实际限制。它和 Qwen3.6-27B 博客的客户端示例相同，均应作为预算层级示例而非模型规格。`preserve_thinking` 仍是调用方 transcript 的历史 thinking 序列化接口，不是持久记忆；此博客没有披露新的内部架构或完整训练 recipe。百炼“可调用”与文章其他上线表述不完全一致，本项目没有做真实 API probe。

## 8. 当前结论与待补证

- 状态：**AA 单榜身份核验 + 官方模型卡/config/chat-template/发布博客 + Qwen Technical Report 的 Agent RL 环境方法专题闭环**；DataCurve 当前快照无精确 Qwen3.6 Agent 行。
- arXiv v1 报告 VHD-Play 环境生成、GRPO 配置和 Qwen3.6-35B-A3B 的实验，但不是完整基础模型预训练 recipe，也没有本项目独立复现。此前博客 benchmark 数值仍是发布方自报；此论文结果同样保留为论文作者报告。
- 未验证完整 checkpoint、视觉路径、MTP acceptance rate、目标硬件吞吐/显存、真实 Model Studio endpoint、独立 benchmark 或生产 SLO。

正式教学映射：第二十一册第 83 章 83.15（混合架构与评测协议）、第二十册第 19 章 19.41（VHD-Play 与可验证 Agentic RL 环境）及第 21 章 21.30（thinking trace 与持久状态边界）；客户端预算对照见第二十四册第 61 章 61.32。

## 9. VHD-Play：先求解机制，再生成可交互环境

### 9.1 来源、归属与模型边界

arXiv 页面将论文评论标为 “Qwen Technical Report”；作者栏显示 Georgia Institute of Technology 与 Alibaba Token Foundry, Alibaba Group。论文日期为 2026-09-23，当前可见版本为 v1。arXiv 摘要页的引用作者元数据与完整 HTML 作者区块并非完全一致，因此这里以题名、编号、版本和评论标识来源，不据此断言作者总数。该预印本不是同行评审或独立复现证据。

VHD-Play 训练的基础策略是 **Qwen3.6-35B-A3B**；论文将 **Qwen3.7-Max** 用作额外的环境 setter / 外部基准对照。不能把 VHD-Play 写成 Qwen3.7-Max 的内部训练 recipe，也不能把其 GRPO checkpoint 的收益记到 Qwen3.7-Max。该报告与 Qwen3.7 Max 博客提到的 environment scaling 主题有关联，但论文没有明确说它实现了博客的 `Task × Harness × Verifier` 训练组合或完整说明 Max 的内部训练法。

### 9.2 技术核心：共同来源、不同视图

环境优先的生成通常先得到执行环境，再另行定义 reward/verifier，因而需要事后证明环境行为与评分规则一致。VHD-Play 反转顺序：

```text
sample θ → solve mechanism M(θ) → freeze reference zθ / reward rule
         → realize executable dynamics D(θ) → wrap as stateful tools E(θ)
policy interacts with E(θ) → score trajectory against the frozen reference
```

每个机制族注册 sampler、reference solver 和可复用 realization/replay adapter。求解可用闭式解、枚举、动态规划或数值优化，不用语言模型估算最优值或当裁判。setter 接收完整参数和从 28 个主题域抽取的真实文本 passage，生成场景、关系数据库、至少 10 个工具、玩家说明和默认策略；policy 只能看接口返回的观察，不直接读取隐藏参数、最优值或默认值。由此把“求解器负责可验证语义”和“语言模型负责语境化环境/接口合成”拆开。

论文定义 episode-level normalized reward：

```text
r(π; θ) = clip_[0,1]((u(π; θ) - u₀(θ)) / (u*(θ) - u₀(θ)))
```

`u*(θ)` 是求解器预先算出的 optimum，`u₀(θ)` 是固定默认策略结果；因此默认策略映射到 0，最优参考映射到 1，越界结果裁剪到 [0,1]。部分可观测在线策略未必能达到 full-information optimum，所以 1 是 reference ceiling，不保证可达。这个 reward provenance 是本方法的重点：环境动力学和评分都从同一已求解机制派生，不依赖 learned judge；但 solver 正确仍不自动证明生成代码的每种行为都正确，仍须 admission/replay。

### 9.3 数据、训练与作者报告结果

- 共报告 3,300 个 admitted environments：2,200 训练、300 个三种训练机制的 held-out（每族 100）、800 个八种未见机制族的评测（每族 100）。三种训练族是 inventory DP、routing、negotiation；完整的 11 个数学/OR 家族见论文 Appendix G。论文估算每个 admitted record 的成本约 $0.01–$0.03，属于 token 量与公开单价重建。
- GRPO 用 64 prompts/step，每 prompt 从 18 次尝试留 16 条 rollout；优势为 return 减组均值，不做标准差归一化、不加 KL penalty、不用 value model；单 epoch、learning rate `2e-6`、clip ratio `4e-3`。报告 checkpoint 在 34 optimizer steps，使用 2,176 个训练 environments、共 34,816 条 scored rollouts。
- 五族 presentation diagnostic 的 agentic mean 从 Base `0.204` 到 step-34 `0.815`；written-out mean 为 `0.962 → 0.992`，显示提升主要落在状态化执行而不是已能完成的文字题。隐藏参数的 agentic 与全量参数已知的 written-out gap 从 `0.758` 降到 `0.177`；这些是作者报告的单次训练/评测配置。
- 三个外部 benchmark：BFCL V4 十个交互类 cell 的不加权均值 `61.25 → 64.08`；TravelBench 20 个英文任务的 plan score `0.700 → 0.794`；365-day E-Commerce Bench 五次 storefront run 的平均期末余额 `54,294 → 182,844`，完赛由 `4/5` 到 `5/5`。论文给出的 Qwen3.7-Max E-Commerce 对照均值为 `165,224`，但五次 run 不构成稳健的普遍胜出结论；TravelBench 中训练 checkpoint 的 `0.794` 仍低于 Max 对照 `0.891`。

### 9.4 验收覆盖与证据限制

论文 admission 检查 executability、默认策略结果是否在族定义区间、以及可 replay 时 reference policy 是否复现预计算 optimum。8 个 replay-supported 家族各抽 10 个 held-out environments，oracle/default/idle 轨迹各运行两次，共 480 次 execution；报告无异常、240 组重复 terminal outcome 一致，且 raw utility 未超过 optimum。其余 3 个家族虽有可计算 reference，但没有上述 shared replay driver 覆盖。因此应说“8/11 家族完成这类 replay audit”，而不是“所有生成环境均被穷尽证明正确”。

其他限制：作者报告 one training run / one evaluation seed；环境/机制合成与规模外推仍依赖已注册数学族；matched EnvScaler 对照是同一 35B 起点、2,200 条环境和 34 步预算，但仍是单一报告配置；论文未提供本项目独立执行所需的完整数据/环境 artifact 与多 seed 复现。论文的性能、环境成本和可扩展性结论应标为 author-reported。
