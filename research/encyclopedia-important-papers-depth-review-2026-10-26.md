# 第四册第十四章《重要论文索引》深度复核记录（2026-10-26）

对象：`book-04-llm-encyclopedia/chapters/14-重要论文索引.md`。本轮依据新版 `WRITING_SPEC.md` 重新研究，不把 2026-10-23 的旧链接/题名审计当作本轮精修证据；旧记录仅作待核线索。

## 复核问题与资料范围

本章跨模型架构、预训练、对齐、推理、检索、推理服务、多模态、安全和 Agent benchmark，不能用单篇综述代替各主题原始研究。本轮重点查清几组常被并列、实际解决约束不同的代表方案：

1. BERT 的预训练目标是否只有 MLM；RoBERTa 改动了目标，还是也改了数据量与训练配方。
2. Transformer、Mamba、FlashAttention、GQA、PagedAttention 各自改变哪一层成本，能否相互替代。
3. RAG、REALM、RETRO 分别在推理、预训练还是语言模型内部接入检索；各自需要承担什么索引/训练代价。
4. InstructGPT、Constitutional AI、DPO、GRPO/R1 解决的是哪类反馈或优化问题；后续方法针对哪个实际瓶颈。
5. CoT、Self-Consistency、Verifier、Tree of Thoughts 分别增加何种测试时计算，以及准确率、预算和验证器覆盖之间的关系。
6. Agent benchmark 的环境、动作空间和验收器差异，为什么论文成功率不能被拼成一个排行榜。

本轮在线阅读了原论文的摘要、方法或实验页中支持上述判断的具体部分，并查看了两篇领域综述的分类框架。没有把所有论文重新做独立复现，也没有把作者报告的分数当作外部复核结果。

## 来源实际支持的判断

### 综述与领域地图

- Gao et al., [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)，阅读 v5（2024-03-27）HTML 的 RAG 范式、检索/生成/增强和评估目录及引言。它把工作组织成 Naive、Advanced、Modular RAG，并按 indexing、retrieval、generation、augmentation、evaluation 展开；支持本章把早期架构论文与现代系统优化分层的结构。它是 2024 年的综述地图，不代表 2026 年所有实现，也不证明单个组件的线上收益。
- Zhang et al., [A Survey of Reinforcement Learning for Large Reasoning Models](https://arxiv.org/abs/2509.08827)，阅读 v3（2025-10-09）摘要、引言和目录。其分类把奖励设计、策略优化、采样、训练资源和应用分开，并将 verifier、过程/结果奖励、长度管理、基础模型先验列为独立问题；支持本章避免把 R1 后训练压成“多做 RL”一句话。该文是领域综述/预印本，本章仅用其组织问题，不据此断言所有争议已有定论。

### 架构与系统

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，阅读原论文 HTML 的摘要、动机和架构章节。原工作替换序列位置间的递归依赖，改善训练并行性；论文结果来自 WMT 机器翻译设置，不能推出 attention 在所有硬件和任务都更优。
- Gu et al., [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)，核对 v2 摘要。论文把传统次二次模型的内容寻址能力不足作为问题，使用输入依赖状态参数和硬件感知的递归并行算法；吞吐和长序列结果是作者报告，绑定论文配置。线性序列成本不等于低延迟、低状态量和检索质量在任意实现中都占优。
- Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180)，核对原论文摘要。针对随请求动态增长的 KV cache 分配、碎片与重复副本，以 block table 分页管理；2–4 倍吞吐是在论文所列系统、模型与延迟比较配置下的结果，不能直接移植为新硬件或当前 vLLM 版本的保证。
- FlashAttention、FlashAttention-2、GQA 的核心定位与上一轮记录交叉核对：前者改 IO/分块算法，GQA 改 Q/KV head 数及缓存结构，PagedAttention 改缓存物理分配；它们是不同层次的优化，能组合而非互为替代。

### 双向预训练与检索

- Devlin et al., [BERT](https://arxiv.org/abs/1810.04805)，阅读原论文 HTML 中 MLM 与 NSP 方法段。原始 BERT 同时采用 masked language modeling 和 next sentence prediction；正文此前只讲 MLM，遗漏了对句间关系预训练目标的交代。
- Liu et al., [RoBERTa](https://arxiv.org/abs/1907.11692)，阅读 v1 HTML 的方法和 4.1–4.3 实验。其改动包含更长训练、更大 batch、更多数据、动态 masking、较长输入，并在其具体设置中移除 NSP；多项变量一同改变，因此不能把性能差异单独归因于“去掉 NSP”。
- Lewis et al., [RAG](https://arxiv.org/abs/2005.11401)，阅读 v4 方法中 RAG-Sequence 和 RAG-Token 公式/解码说明。RAG-Sequence 对整个输出序列的文档潜变量求和；RAG-Token 对各 token 条件分布求和。原文现有公式是 Sequence 级简化，须标明类型，不能称为两种 RAG 的通用公式。
- Guu et al., [REALM](https://arxiv.org/abs/2002.08909)，阅读原论文方法和检索训练部分。REALM 把隐式知识存储问题转化为基于 MLM 的潜在文档检索训练，检索进入预训练、微调和推理；其在线索引刷新和 corpus 级 retrieval 有训练/工程成本。
- Borgeaud et al., [RETRO](https://arxiv.org/abs/2112.04426)，阅读 v3 方法、chunked cross-attention 和实验概要。RETRO 用冻结检索器从大规模 token 数据库取邻居 chunk，以 chunked cross-attention 注入自回归预测；论文报告 2T-token 数据库、特定 Pile 配置下用更少参数达到与 GPT-3/Jurassic-1 可比的语言建模表现。它是作者实验，不可直接外推为生产 RAG 成本或模型质量优势。

### 工具接口与协议层

- OpenAI [Function calling 官方文档](https://developers.openai.com/api/docs/guides/function-calling)，2026-10-26 页面读取。文档把调用描述为模型请求、应用执行、结果回传的多步流程，并以 JSON schema 约束函数参数；它证明该 API 的公开接口约定，不证明其他 provider 或生产应用的执行/授权方式。
- [Model Context Protocol 规范](https://modelcontextprotocol.io/specification/2026-07-28)，2026-07-28 版本。规范定义 host/client/server 互操作、resources/prompts/tools 等能力，并明确协议本身不能强制执行所有安全原则，授权/隐私控制仍由实现者承担。正文据此区分“工具 schema/协议互操作”与“业务侧权限、幂等、审计”。

### 偏好对齐与推理强化学习

- Ouyang et al., [InstructGPT](https://arxiv.org/abs/2203.02155) 与 Bai et al., [Constitutional AI](https://arxiv.org/abs/2212.08073)：保留上轮原论文结论。本轮用于区分人类示范/成对偏好反馈与原则引导的模型批评/RLAIF；两者改变主要是反馈生成与监督流程，不是同一种优化算法。
- Rafailov et al., [DPO](https://arxiv.org/abs/2305.18290)，核对 v3 摘要。其闭式重参数化允许从离线偏好对以分类损失训练，不需显式在线 rollout；实验范围是作者选择的任务。它减少 RLHF 流程部件，不解决偏好覆盖、长度偏差、离线分布外行为或长期环境奖励。
- Shao et al., [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)，核对 v3 摘要；用组内样本相对奖励估优势，省去独立 critic/value model 的显存负担。论文报告 DeepSeekMath 7B 在 MATH 上 pass@1 51.7%，64 路 self-consistency 60.9%；不得把二者当作同一推理预算或直接比较，也不把摘要分数外推到其他模型/数据。
- Guo et al., [DeepSeek-R1](https://arxiv.org/abs/2501.12948)，使用 2026-01-04 的 arXiv v2 阅读 R1-Zero、R1 pipeline、奖励和数据 recipe。R1-Zero 直接对 base model 做 RL、没有预先 SFT，使用可验证奖励并报告出现更长推理/反思行为；作者同时指出可读性差、语言混杂和非推理任务表现不足。R1 则加入 cold-start SFT、第一阶段 RL、拒绝采样生成数据、混合推理/非推理 SFT 和第二阶段 RL；R1-Zero 和 R1 不应被写成同一 recipe。
- Yu et al., [DAPO](https://arxiv.org/abs/2503.14476)，阅读 v2 的方法、实验章节：提出 clip-higher、dynamic sampling、token-level policy loss、overlong reward shaping，并开放 verl 训练代码和数据。作者在 Qwen2.5-32B/AIME 2024 报告 50 分及相较指定基线的训练步数比较；这是整套系统与数据配方的结果，不能单独归因于其中一条算法技巧。
- Liu et al., [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)，阅读 v2 的目标分析、Dr.GRPO 与实验段。作者指出 GRPO 的序列长度/标准差归一化可能造成优化偏置，并以移除这两项提出 Dr.GRPO，在特定 Qwen/MATH 训练设置中观察到更好的 token 效率；这是后续论文提出且实验支持的解释，模型族、任务和实现外的普遍性仍待验证。
- Zhang et al. 的推理 RL 综述将基础模型先验、SFT 与 RL 关系、奖励、采样和长度预算分开讨论。正文应把“更长 CoT”作为需要答案质量、成本和错误率共同验证的观测，不把长度本身当作推理能力的证明。

## 跨论文方案比较与修改决策

| 问题 | 代表方案各自解决什么 | 机制与证据 | 代价、限制与后续关系 |
| --- | --- | --- | --- |
| 双向表示预训练 | BERT 用 MLM 学双向上下文，并用 NSP 训练文本对关系；RoBERTa 研究 BERT 是否因训练不足而未达上限 | RoBERTa 同时改训练步数、batch、数据、mask、输入长度和 NSP；论文做了分项控制/消融，但最终配方含多项联合变化 | 不能把综合提升归结为单一改动。保留 BERT 两个目标，并说明 RoBERTa 的变量与适用条件 |
| 序列建模与推理成本 | Transformer 解决递归计算不能并行的问题；Mamba 处理长序列成本与选择性；FlashAttention 降低 attention 的 IO；GQA 减少 KV head；PagedAttention 缓解动态缓存分配浪费 | 分别改变网络架构、attention kernel、Q/KV 结构、服务缓存分配；结构可以组合，但每种收益的指标和层级不同 | Transformer attention 对任意位置做内容交互但序列矩阵成本高；SSM 有压缩状态及随机访问信息局限；kernel 与分页不降低模型的理论 attention 计算复杂度；正文新增分层比较，禁止把这些方法放在同一“更快模型”榜上 |
| 参数外知识 | RAG 把知识在推理时检索到 prompt；REALM 训练检索器，让 latent document 参与 LM 预训练；RETRO 在生成过程中按 chunk 接入检索 | RAG 论文区分 sequence/token latent document；REALM 通过 MLM 学检索；RETRO 使用邻居 chunk 和跨注意力 | 分别承担在线检索/上下文预算、预训练与索引更新、外部大语料和专用架构成本；现代产品 RAG 系统不是这些论文模型的逐字复现 |
| 对齐反馈 | InstructGPT 使用人工示范、偏好排序和 RLHF；Constitutional AI 将原则批评与 AI 偏好纳入流程；DPO 直接优化离线偏好对 | 反馈来源与优化算法属于不同维度；DPO 的主要优势是用偏好分类目标替代 reward model 加 online PPO 的典型链路 | 偏好数据/原则仍有覆盖盲区；DPO 无法自动获得在线环境探索；要按任务反馈成本、可判定性和分布漂移选方法 |
| 推理 RL | GRPO 以同题样本组估计相对优势；R1-Zero 探索无 SFT 冷启动 RL；R1 加冷启动和混合式多阶段训练；DAPO 改长 CoT 训练系统的采样/裁剪/长度处理；Dr.GRPO 针对 GRPO 的长度归一化偏置提出修正 | R1 和 DAPO/Dr.GRPO 都含多变量配方；分别保留“模型/recipe 结果”和“算法局部消融”的证据粒度 | 验证器覆盖、探索能力、语言一致性、可读性、错误响应长度和训练 token 成本均需报告；单一 AIME 分数不足以证明普适推理能力 |
| 测试时推理 | CoT 增加单条生成轨迹；Self-Consistency 对多条独立采样投票；Verifier 对候选作外部判别；ToT 显式扩展/筛选中间状态 | 这些方法调用次数、候选生成、验收器和搜索控制方式不同 | 错误相关性、答案解析、verifier 误差、调用成本和延迟决定收益；需要报告每题 token/模型调用预算，不能把它们都称为“多想一会儿” |
| Agent 评估 | SWE-bench 以仓库 issue/测试为核心；WebArena 是多站点网页操作；OSWorld 在真实桌面应用中执行任务 | 任务状态、工具环境与验收方式各异，论文数字必须与对应 harness 和数据版本绑定 | 三者不可直接横向比较；本章保留各基准并补上任务/环境差别，不得用统一成功率排榜 |

## coverage_map

| 旧正文主题 | 旧稿问题/重要性判断 | 精修后位置与解释层级 | 处理 |
| --- | --- | --- | --- |
| 14.1 论文问题、控制变量、成本与证据账本 | 本章读法的基础，需完整保留 | 14.1，核心；保留实验账本和边界规则 | 保留并补充与后续对比表的读法衔接 |
| 14.2–14.14 Transformer、BERT/GPT/T5、Scaling/Chinchilla、位置、Mamba、FlashAttention、GQA | 架构与训练主线，不可因加横向比较而删掉单篇论文贡献 | 原章节保留；新增 14.42.1 比较架构/系统层次 | BERT 加 NSP 与 RoBERTa；KV 容量公式补齐全量维度；其余主题入口保留 |
| 14.15–14.24 InstructGPT、Constitutional AI、DPO、GRPO/R1、LoRA/QLoRA、ZeRO、MoE | 对齐、推理训练和效率的重要谱系 | 原章节保留；新增 14.42.4–14.42.5 对齐/推理训练对照 | 修复 GRPO cases 公式；拆分 R1-Zero/R1；加 DAPO、Dr.GRPO 的证据和范围 |
| 14.25–14.29 RAG、REALM/RETRO、ReAct、Toolformer、PagedAttention | 参数外知识、工具和服务状态的不同层次 | 原章节保留；新增 14.42.3 检索范式比较 | 标注 RAG-Sequence 简化公式；区分模型内检索和在线 RAG；保留 Agent/缓存主题 |
| 14.30–14.33 CoT、Self-Consistency、Verifier、ToT | 测试时计算的主要比较族 | 原章节保留；新增 14.42.6 | 增加预算、相关错误、判别器和显式搜索区别 |
| 14.34–14.40 CLIP、ViT、LDM、Whisper、安全、ROME/MEMIT、circuits | 多模态、安全和解释性主题仍需保留，横向增加内容不应挤掉 | 原章节保留，当前作为代表性论文索引与延伸入口 | 主要结论添加近邻原论文链接；未声称本轮对这些主题逐篇全文复现 |
| 14.41 Agent benchmark | 环境闭环评估是 Agent 关键前置，不可缩成“Agent benchmark”一行 | 原章节保留并增加基准比较表 | 写明 task/environment/evaluator 不同，分数不可直接排序 |
| 14.42 谱系；14.43 阅读记录；14.44 阅读路径；14.45 来源入口 | 是读者导航和方法，不是低价值尾部，必须保留 | 新增比较章 14.42 后，旧内容改为 14.43–14.46 | 内容保留，仅顺延编号；补充综述与后续论文入口 |

## 公式与证据边界

- GRPO 相对优势式的分段条件需以 `\\` 分行；`s=0` 定义优势为 0 是零方差处理，不声称该分段式就是论文全部实现。完整算法还包括 clipped policy objective、KL/参考模型选项及 token mask。
- KV cache 的逻辑容量按 `2 × N_layers × B × T × H_kv × d_head × bytes_per_element` 表示；假设 K 与 V 都缓存、所有层同构、无量化元数据和分页碎片。实际 serving 占用还要计 block 对齐、分配器开销和共享前缀。
- RAG 原式标成 RAG-Sequence 的 top-k 边缘化简化；正文并列说明 RAG-Token 的逐 token 边缘化差异。
- 新增论文收益均写“论文报告/在该设置下”，不做跨模型和跨 benchmark 的数值排名；建议/后续改进若不是来源已实验的结论，标为作者推论或仍需测量。

## 尚未完成的边界

- 本章是大范围论文索引，不等同于每一篇论文的系统综述。多模态、安全、解释性和部分架构论文保留其原入口与有限结论，不据本次聚焦调研声称已逐篇阅读全文。
- 本轮联网只核对了研究记录列出的版本和具体章节；没有在本环境重跑论文实验、没有对所有引用重新做完整链接爬取，也未构建 PDF。
- R1/DAPO/Dr.GRPO 的公开论文结论受训练模型、数据、verifier、采样和实现版本限制；不同论文结果不能直接比较。
