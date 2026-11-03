# 大模型训练与后训练系统研究资料（2026）

检索与阅读日期：2026-10-30。本文只记录公开论文/技术报告中的训练机制和证据边界，不记录书册审阅进度或正文修改决定。

## 多模态预训练的阶段化课程

**来源**：[Qwen3-VL Technical Report](https://arxiv.org/html/2511.21631v1)，Qwen Team，arXiv v1，2025-11-26。阅读摘要、引言和 §3.1。

- 报告把预训练分为四段：S0 只训练 vision-language merger，约 67B token、8K 序列；S1 全模型约 1T token、8K；S2 全模型约 1T token、32K；S3 全模型约 100B token、262K。
- 报告将后训练描述为长 CoT SFT、较强教师蒸馏和 RL 三段；正文还报告区分 thinking 与 non-thinking 变体。
- 这是 Qwen3-VL 团队对其单一模型系列的自述训练方案。阶段长度、数据配比和效果不能直接当作其他多模态模型的通用最优配方，也不是独立复现实验。

### Qwen3-VL v2 的图文训练、数据混合与文本能力保持

- **来源：** [Qwen3-VL Technical Report v2](https://arxiv.org/html/2511.21631v2)，Qwen Team，arXiv v2（2025-11-27）。复核日期：2026-11-02。实际阅读摘要、引言、§3.1 训练阶段和视频时间戳描述；另阅读 §3.2 的图文、OCR、文档、grounding 与 video 数据小节。
- **来源支持：** 报告以继续训练的 SigLIP-2 vision encoder、MLP merger 与 Qwen3 LLM 组成 VLM；S0 仅训练 merger（67B tokens、8K），S1 全参数多模态训练（约 1T、8K），S2 全参数长上下文训练（约 1T、32K），S3 全参数超长上下文适配（100B、262K）。作者描述 S1 同时混合文本和 VL 数据，S2 增加文本占比及长视频/agent instruction，S3 聚焦长视频与长文档；其摘要还称对 text-only 与多模态目标使用平方根重加权和 per-token loss normalization。报告把秒与时分秒格式的文字时间戳作为视频帧组前缀。
- **证据等级：** source-backed（模型团队技术报告中的方法、数据策略与作者实验）。数字、训练阶段和收益是团队披露，不是独立重训结果。
- **边界：** 不将 S0—S3 当作通用最优课程；平方根重加权的报告描述不单独证明采用该方法必然保持文本能力。报告中的 context length 不等于长序列有效召回，且不同多模态任务需使用各自分母与回归集。论文报告的 per-modality/budget 数字按其定义理解，不跨章节或其他模型作简单总量比较。

### Qwen3-Omni 的统一多模态预训练和 Talker 后训练

- **来源：** [Qwen3-Omni Technical Report](https://arxiv.org/html/2509.17765)，Qwen Team，arXiv v1（2025-09-22）。复核日期：2026-11-02。实际阅读 §2.2 AuT、§2.4—2.5 语音生成/流式架构、§3 预训练和 §4.1—4.3 后训练。
- **来源支持：** 报告称 AuT 从头用 20 million hours 监督音频训练，数据包含中英 pseudo-label ASR、其他语种 ASR 和 audio understanding；预训练从 encoder alignment 进入全参数的多模态 general stage，再用 32,768 token 序列提升长音视频能力。报告所列模态包括纯文本、图文、视频文本、音频文本、视频音频及其组合。Talker 先以带多模态上下文的语音数据训练映射，后续进行高质量 continual pretraining、长上下文训练、多语言偏好 DPO 和 speaker fine-tuning；语音生成采用多码本 autoregressive token、MTP 残差码本和因果 Code2Wav。
- **证据等级：** source-backed（作者技术报告描述与其评估设置）。模块职责、数据阶段和训练顺序属于发布方披露。
- **边界：** 训练时数、配比、架构与作者报告延迟/质量绑定 Qwen3-Omni 的配置和实验协议，未独立复现。codec frame rate 不等于客户端首包时延；论文对比结果不外推为所有语音生成模型、硬件或并发负载的保证。

## 优化器分组、batch scaling 与稳定性评估

**来源**：[On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability](https://arxiv.org/html/2608.30320v1)，arXiv:2608.30320，2026-08。阅读 §3.1–§3.3 中优化器、超参数缩放、稳定性压力测试及生产配置检查。

- 报告把 Muon 用在若干矩阵形状的投影和专家权重上，而将 embedding、输出头、MoE router 及特定低秩投影交给 AdamW；报告说明其 router 的早期波动是采用这一分组的原因之一。
- 在其 10.8B-A0.89B MoE、4T token 对照中，报告称按 scaling-law 预测从 12.6M token batch 调到 25.2M 优于旧设置；37.7M 的变化较小。该报告中的 batch-size ramp 没有优于固定目标 batch，并多用了 18.8% optimizer steps。
- 在另一个 156B-A7B、419B-token 对照中，作者报告预测值附近的若干 batch/LR 配置结果接近，训练曲线稳定。以上结论绑定该报告的模型、预算、优化器、数据顺序和实现，不构成 Muon、batch ramp 或 LR 的普遍规则；报告也提醒窄幅 benchmark 排序可能处于运行/测量噪声内。
- 在 §3.3 的 28 层 MoE 压力测试中，作者把学习率固定在最优值的 3 倍，固定 AdamW、结构和数据顺序，只切换门控。作者报告 loss spike 率由每 10,000 步 32.0 降至 3.2，gradient-norm threshold crossings 由 256 降至 20；该来源将 spike 定义为超过 201 步滚动中位数至少 0.1 的训练步。threshold-crossing 计数在该对照句中没有写成每 10,000 步归一化数，因此不可把它与 spike rate 使用同一分母。该结果是作者在特定压力测试中的单变量证据，不等同于目标模型上的复现，也不能外推为启用门控后的通用收益。

## 多教师 on-policy distillation

**来源**：[MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training](https://arxiv.org/html/2606.30406v1)，arXiv:2606.30406，2026-06-29。阅读 §§1、3–5 和附录训练细节。

- 方法先从一个通用 SFT 初始化分别训练领域 RL teachers，再让 student 生成自己的 rollout，由对应冻结 teacher 在 student 访问到的状态提供逐 token 监督；论文重点是多领域能力整合，不是提出新的通用 RL objective。
- 论文在 Qwen3-30B-A3B 的数学、指令遵循和软件工程设置，以及作者报告的 MiMo-V2-Flash 案例中比较集成路线。归一化分数、模型、任务、数据与训练预算均属于论文设置；这些结果不能外推为任意 teacher 数量或任务组合的保证。

## 大规模 RL/OPD 的异步采样与状态恢复

**来源**：[DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://arxiv.org/html/2609.19969v1)，DeepSeek-AI，arXiv v1，2026-09-17。阅读 §§4.1–4.2、§5.1–§5.2。

- 报告称该模型使用 45T 多模态 token 预训练；稀疏注意力从头在 64K 序列长度训练，并在约 34T token 时把上下文扩到 1M。作者报告的这些训练设置不代表对其他架构的推荐配置。
- 后训练使用 SFT、RL 和 OPD 的既有组合。报告强调其变化集中在合成数据与交互环境：任务构造、参考解/奖励、过滤、去重和难度校准。
- 异步后训练设施以并发生成缓解 rollout 长尾；报告还描述了控制在途样本和等待条件以限制 policy lag、对过度陈旧 token 使用 loss mask，以及在 checkpoint 切换时逐 token 中断 rollout、保存 KV/routing 状态并从新 checkpoint 恢复。
- 最终全词表 OPD 阶段据报告使用 40 余个 teacher，并支持不同训练阶段或不同架构的教师。相关吞吐与能力结论是报告作者的训练系统披露，不等同于独立复现或通用生产保证。

## 近期 MOPD 改进方向：仍属早期预印本

**来源一**：[UP-MOPD: Update Projection in Multi-Teacher On-Policy Distillation](https://arxiv.org/html/2610.08398v1)，arXiv v1，2026-10-06。阅读摘要、§§1–3 和附录 A。

- 该工作指出 AdamW 的动量、坐标自适应缩放和 weight decay 可能使“梯度层面不伤害各领域”的修正，在实际 optimizer displacement 上仍伤害某个领域；其方法对 optimizer 生成的候选位移作最小投影修正。
- 论文报告了医疗/通用两教师和数学、代码、指令遵循三教师设置；作者明确指出教师数量增加时的梯度计算成本和长期能力保持仍需研究。因此这些结果不支持把方法写成成熟通用标准。

**来源二**：[Nash-MOPD: Optimizing Task Trade-offs in Multi-Teacher On-Policy Distillation](https://arxiv.org/html/2610.26124v1)，arXiv v1，2026-10-19。阅读摘要、§§1–5、§6 及 limitations。

- 该工作根据各领域 teacher–student reverse-KL 差距及其近期变化估计剩余训练需求，再用局部 minimax 目标动态分配 prompt 采样比例。
- 论文报告的实验限于 DeepSeek-R1-Distill-Qwen-1.5B（三个领域）和 Qwen3-8B-Base（数学/代码）两种 student；作者把跨领域迁移、rate 不确定性、延迟反馈、更多模型规模和训练预算列为后续问题。
- 两篇均为近期预印本，适合作为 MOPD 主题的跟踪方向；单次论文对照不能替代跨实现、跨任务的复现或形成行业默认结论。

## 训练期并行布局切换：DynaTrain 预印本

**来源：** [DynaTrain: Fast Online Parallelism Switching for Elastic LLM Training](https://arxiv.org/abs/2605.18815)，arXiv:2605.18815 v1，2026-05-12。阅读摘要及 §§3–5，重点核对状态抽象、数据游标迁移、重配置时延和实验模型。

- DynaTrain 用 Virtual Parameter Space 表示并行状态的逻辑张量区间，并以 rank-local routing/state transition 计划将源布局中的参数及 optimizer-state 切片映射到目标布局；其 Elastic Device Manager 管理并行组切换。论文也明确把各 DP rank 已消费样本状态纳入切换，以避免新布局下数据跳过或重复。
- 作者报告，在其测试系统中，70B dense 模型重配置低于 2 秒，235B MoE 重配置为 4.36 秒，并称相对所比较的 checkpoint-based/elastic systems 有数量级优势。该数字属于论文作者的模型、系统、拓扑和 workload 结果，不代表一般 checkpoint 读取延迟，也不是 PyTorch DCP 默认提供的在线重配置能力。
- 这是单篇系统预印本证据；当前资料未形成独立、多集群复现或跨框架基准，不能据此把在线并行切换写成生产训练默认功能。

## 跨节点 MoE 下的 DualPipe

**来源：** [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)，DeepSeek-AI，arXiv v1，2024-12-27。阅读 §3.2.1–3.2.2 与 Table 2，核对 pipeline schedule、通信重叠和配置条件。

- 报告给出的 DeepSeek-V3 训练布局为 16-way PP、跨 8 节点的 64-way EP 和 ZeRO-1 DP，不使用 TP；作者称跨节点 EP 通信在其配置下造成约 1:1 的通信/计算比例。
- DualPipe 将 attention、all-to-all dispatch、MLP、all-to-all combine 与 pipeline 通信放入 forward/backward chunk 重排；反向又拆分 input-gradient 与 weight-gradient 计算，并从 pipeline 两端双向送入 micro-batch，以尝试覆盖 all-to-all 和 PP 通信。
- 作者的 Table 2 用特定 `F/B/W/F&B` 时长表达比较 bubble，记参数副本为 2×、激活占用为 PP+1；正文称其 schedule 要求 pipeline stage 数和 micro-batch 数可被 2 整除。该系统报告只能支持这一方案的设计意图和作者实验口径，不能推导任意 MoE、网络或 scheduler 都会获得相同速度、显存占用或零通信开销。
- 相关原始方法来源为 [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)，arXiv v1，2023-11-30，阅读摘要与 §3。该论文将 backward 拆成 input-gradient 和 parameter-gradient 工作，并提出相应同步 schedule；作者报告相对 1F1B 的吞吐差异依其配置与显存约束，不能把方法名理解为所有 pipeline 设置都实现字面上的零 bubble。

## PyTorch 2.15 的分布式训练与 checkpoint 接口

**来源一：** [PyTorch 2.15 Release Blog](https://pytorch.org/blog/pytorch-2-15-release-blog/)，PyTorch Foundation，2026-10-28。阅读 release 说明中 Distributed Training 的 NCCL backend、flight recorder、FSDP2 和 pipeline 段落。

- 发布说明称，重写的 NCCL backend 在 PyTorch 2.15 中成为 `nccl` process group 默认实现，并提供 collective hang 后的 flight-recorder 诊断记录；说明中列出用于回退的配置路径。该信息限定于发布说明中的版本和后端，不能推及旧版 PyTorch、非 NCCL backend 或其他 vendor runtime。
- 同一发布说明列出 FSDP2 对精度策略和 CUDA graph gradient accumulation 的细化，以及 pipeline receive buffer 延迟分配、send 完成后释放 forward output 等改动。这些是 2.15 工程变更，不构成目标模型、硬件和拓扑上的性能保证。

**来源二：** [PyTorch Distributed Checkpoint 2.15 文档](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html)，核对日期 2026-10-30；并对照 [FSDP2 `fully_shard` 2.15 文档](https://docs.pytorch.org/docs/2.15/distributed.fsdp.fully_shard.html)。

- DCP 文档说明其多 rank 并行存取与 load-time resharding 语义；`async_save` 返回的 `staging_completion` 和 `upload_completion` 分别标记本地 staging 与 checkpoint 上传完成。
- 文档不保证不同 PyTorch 版本之间 state_dict/checkpoint 的向后兼容。框架可以重分配张量 shard，不替应用定义训练数据游标、全局 manifest 原子提交或恢复后浮点轨迹完全一致的语义。

## Megatron Core 的层级数据并行

**来源：** [Megatron Core Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)，持续更新的官方工程文档，核对日期 2026-10-30；阅读 Megatron-FSDP 与 hierarchical DP 配置段落。

- 当前指南列出由 `num-distributed-optimizer-instances` 启用的 hierarchical data parallelism，以及外层 DP 的 `no_shard`（复制模型状态）和 `optim`（外层再分片 optimizer state）选项；其中 `optim` 要求内层使用 `optim_grads_params`。
- 这些是 Megatron Core 当前文档定义的配置语义，可帮助读者理解如何把不同分片/复制边界映射到层级拓扑；接口动态演进，页面没有提供所有任务的公平对照，不能由选项名称推出固定性能收益。

## GLM-5.3 长周期 RL 环境与训练/rollout 系统披露

**来源：** [GLM-5.3: Frontier Coding with Emergent Cyber Capabilities](https://z.ai/blog/glm-5.3)，Z.ai 官方博客，发布日期 2026-08-14。实际阅读 environment scaling、verifier pipeline 与 “slime: Built for Long-Horizon RL Scaling” 段落；直接打开页面未返回正文行，所读相关段落来自该官方页面的搜索索引全文。

- 博客称其从真实工作中收集任务模式并生成带多步依赖和隐藏状态的可运行长周期环境；judge agent 会尝试任务以检查可解性，verifier 在看不到参考解时合成，solver trajectory 用于发现奖励捷径。发布方称其 verifier 会经过 oracle、no-op 与 unsolved-state 检查，并指出环境生成与校验仍需要人类参与。
- 系统部分将 slime 描述为训练侧 Megatron、rollout 侧 SGLang 与数据 buffer 的统一 dataflow；公开能力包括 top-p mask、top-k/full-vocabulary OPD 和更完整的训练/rollout 数值对齐。博客报告其训练—rollout 一致性评估的平均 log-prob 差控制在约 1e-7，并称较旧设置降低超过 99.99%。发布方还称，本地分层缓存、teacher 动态切换与 prefetch 等系统优化使其长周期 coding RL 端到端训练吞吐提高超过 2.3×。
- 证据等级：source-backed（发布方技术/产品博客中的自述）。
- 适用边界：页面没有充分给出上述一致性与吞吐对照的全部 workload、硬件、并发、模型、轨迹长度和测量协议，数字不可复现为通用加速保证；任务生成和 verifier 方案也是发布方描述，不代表独立验证其奖励可靠性。

## 其他模型团队公开的后训练方向

- OpenAI, [Introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/) 与 [gpt-oss model card](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf)，发布说明及模型卡，2025-08。实际阅读训练过程、目标和 chain-of-thought 监督说明。
  - 来源支持：OpenAI 披露 gpt-oss 经 SFT 阶段和高算力 RL 阶段训练，目标包括遵循 Model Spec、CoT reasoning 和 tool use；模型卡说明训练并未直接对 CoT 施加优化压力。
  - 证据等级：source-backed（发布方说明和模型卡）。
  - 适用边界：未披露足以复现的具体优化器、reward、数据、训练预算和阶段超参数；上述表述只能证明公开的阶段与目标，不能推断完整内部配方。

- xAI, [Introducing Grok 4.7](https://x.ai/news/grok-4-7)，2026-09-21 官方发布页。实际阅读训练投入与长程任务描述。
  - 来源支持：xAI 称 Grok 4.7 使用更大的 base model 和更长的 RL run，训练任务混合提高了困难任务权重，其中部分任务需要数小时；发布页还提到自验证与长上下文管理。
  - 证据等级：source-backed（模型团队的发布披露）。
  - 适用边界：未公开足够的算法、reward、数据、预算或消融细节；这些陈述属于方向性自述，不构成独立因果验证，也不能直接转成其他模型的训练配方。

## On-Policy Distillation 的表示目标、层间转变与停止边界

### On-Policy Representation Distillation（OPRD）

- **来源：** [OPRD: On-Policy Representation Distillation](https://arxiv.org/abs/2606.06021v4)，arXiv v4，2026-06-20。2026-11-02 复核；阅读摘要、§4.1 的模型/数据设置及 limitations/future-work 对迁移范围的说明。
- **来源支持：** 方法在学生自生成的同一 rollout 上对齐选定层的 teacher/student hidden representations，以逐样本、逐层的确定性梯度信号补充输出空间 token 匹配。OPRD-Bridge 通过冻结投影桥处理深度/宽度不一致及跨 tokenizer 情形。论文在长 CoT 数学推理实验中报告 AIME24、AIME25、AIMO 的学生能力提升，并称其配置相对 top-k 输出蒸馏训练快 1.44 倍、显存最多降低 54%。
- **证据等级：** source-backed（预印本 v4 的方法与作者实验）。性能和资源收益为作者在其模型、实现、基准和比较设置下的结果。
- **边界：** 主要证据来自数学推理；作者将 code、agent、open dialogue 等场景列为后续研究方向。隐藏表示对齐不自动解决语义/tokenizer/架构对应问题，也不证明跨任务迁移或普遍效率收益。

### On-Policy Transition Distillation（OPTD）

- **来源：** [On-Policy Transition Distillation](https://arxiv.org/abs/2610.28048v1)，arXiv v1，2026-10-20。2026-11-02 复核；阅读摘要、层间 log-prob increment 定义、三领域实验结果表及限制说明。
- **来源支持：** OPTD 在学生生成 rollout 上对齐相邻网络层预测分布变化的 log-prob increments，监督对象是深度方向的预测转变，与 OPRD 的隐藏表示目标不同。论文在三种 backbone、八项数学/代码/指令基准中报告结果；一个 1.5B 三领域设置中总体分数由 OPD 的 37.89 到 OPTD 的 39.57，教师—学生能力差距恢复比例由 74.9% 到 89.2%。摘要中的中间层转变与答题正确性关联属于诊断观察，不能单独识别因果。
- **证据等级：** source-backed（单篇预印本 v1 的方法和作者实验）。
- **边界：** 所有结果依赖论文的数据、教师路由、训练预算与评估协议；版本仍是 v1 预印本，不能据此断言层间目标普遍优于输出目标。

### Training-Free On-Policy Distillation（TF-OPD）

- **来源：** [Training-Free On-Policy Distillation](https://arxiv.org/abs/2610.22236v2)，arXiv v2，2026-10-20。2026-11-02 复核；阅读摘要、层级最小二乘方法、实验表和 rollout-filtering/precision limitations。
- **来源支持：** 对共享同一底模的专家模型，逐层用专家读取自身回答时的 rollout 输入二阶矩构造最小二乘解，不做梯度更新。主实验以 Olmo-3.1-7B 数学/代码专家为例，使用约 256 个数学与 256 个代码题目，筛选出 175 个数学和 92 个代码样本；作者报告七个数学/代码基准平均 61.1，高于数学专家 59.8 和代码专家 53.6。论文的 student-rollout 变体在剔除重复循环后没有超过专家 rollout 的收益。
- **证据等级：** source-backed（预印本 v2 的算法及作者报告结果）。
- **边界：** 该路线要求共享底模，主结果主要来自领域专家 rollout，因此是邻接的 checkpoint 合并方法，不应和严格的当前学生 on-policy 训练等同。筛选样本量有限；竞赛题样本较少、精度敏感，不能外推为通用合并表现。

### 小模型 OPD 中的答案与停止

- **来源：** [Solving Without Stopping: On-Policy Distillation at Small Scale](https://arxiv.org/abs/2609.37326v1)，arXiv v1，2026-09-29。2026-11-02 复核；阅读摘要、Qwen3-8B→4B/1.7B/0.6B thinking/non-thinking 设置、answer commitment 与 stopping 结果讨论。
- **来源支持：** 论文将解题、答案标记/提交和停止/结束行为分别测量。在其 thinking-mode 设置中，学生解题能力提升，但停止能力没有同样迁移；较小学生有时已生成正确答案仍继续写。non-thinking 设置的结果不同。
- **证据等级：** source-backed（单篇预印本的模型、模式和评估观察）。
- **边界：** 结果限于所用 Qwen3 尺寸、thinking protocol 与任务指标，不证明 OPD 普遍无法迁移停止能力。评估应分别报告答案是否出现、答案是否正确、正确答案后是否恰当结束和过度生成率。

### TRL 在线蒸馏文档快照

- **来源：** [TRL Distillation Trainer 文档](https://huggingface.co/docs/trl/en/distillation_trainer) 与 [TRL Generalized Knowledge Distillation Trainer 文档](https://huggingface.co/docs/trl/en/gkd_trainer)，Hugging Face 持续更新文档，2026-11-02 核对。
- **来源支持：** DistillationTrainer 文档描述学生自生成 completions（可选 vLLM），通过分块 JSD 匹配完整 next-token teacher distribution；`AsyncDistillationTrainer` 将生成与训练解耦，并支持经 HTTP 访问的 teacher 服务。GKD 文档当前把 `lmbda` 定义为学生自生成数据比例，`seq_kd` 控制教师生成序列数据选项，`beta` 插值 generalized JSD 的 forward/reverse（文档亦称 inverse）KL 近似，公开 API 位于 `trl.experimental.gkd`。
- **证据等级：** source-backed（文档当日公开接口说明）。
- **边界：** 文档为动态页面，字段与代码可随版本改变；任何实现审计仍须固定已安装 revision 并查看对应源码，文档本身不支持论文性能或配置可复现性主张。

## DeepSeek-V4 v1 的后训练披露与上下文协议

- **来源：** [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/html/2606.19348v1)，DeepSeek-AI，arXiv v1（2026-04-26）。2026-11-02 复核；实际阅读 §5.1.1、§5.1.2 与 §§5.2.2–5.2.5。
- **专家培养与 reasoning effort：** 报告称各领域专家沿 V3.2 后训练路线先 fine-tune、再按领域 prompt 和 reward 做 RL，RL 使用 GRPO。Pro 和 Flash 均支持 Non-think、Think High、Think Max 三种模式；不同模式用不同 RL 配置、长度惩罚和上下文窗口。Think Max 还在 system prompt 开头加入专门指令，思考/回答格式使用 `<think>` 与 `</think>`。
- **GRM 与工具协议：** 对 hard-to-verify 任务，报告称使用 rubric-guided RL 数据和 Generative Reward Model（GRM）评价轨迹，并直接 RL 优化 GRM；作者称 actor 网络兼任 GRM，内部推理能力因此参与评价。工具调用协议使用专用 `|DSML|` token 和 XML 结构；作者称其能减少转义失败和 tool-call errors，但报告未给独立复现或可横比的错误率消融。
- **思考轨迹管理与 Quick Instruction：** 报告称 tool-call 场景会跨工具结果和新用户消息保留思考轨迹；普通对话仍在新用户消息后丢弃上一轮思考内容。把工具调用模拟成 user message 的框架可能不进入 tool-call 路径。Quick Instruction 则为网页检索判断、query、authority/domain 分类、URL 读取和标题生成等任务追加专用 token，并复用已计算的 KV cache；作者称可降低重复 prefill、用户感知 TTFT 和小模型维护负担。以上是该报告披露的协议与作者解释，不是任意聊天/Agent 系统的默认设计结论。
- **Multi-Teacher OPD：** 报告写出加权 `D_KL(π_student || π_teacher)`，由 student 自己生成轨迹，再按任务上下文向相关 teacher 学习，第二阶段使用 10 个以上教师。报告将 sampled-token log-ratio 复用作 policy loss advantage 的方案描述为资源效率较高但梯度估计方差较高、训练常不稳定，并称 V4 因而选 full-vocabulary logit distillation。报告没有提供其配置下 sampled 与 full-vocabulary OPD 的受控消融，无法据此推出所有 sampled OPD 都不稳定。
- **全词表教师调度：** 教师权重放在集中式分布存储并按需以 ZeRO-like 参数分片加载；教师前向只缓存最后一层 hidden states，学生训练时读取对应 prediction head 重建 full logits；样本按 teacher index 排序，使每个 teacher head 每 mini-batch 只加载一次、设备上最多驻留一个 head；加载/卸载异步运行，TileLang kernel 计算 exact KL。报告的“effectively unbounded teacher count”和单教师可能有 trillion parameters 属于框架能力陈述，不等于 V4 实际使用的教师数量/尺寸；未披露可通用容量规划的峰值显存、I/O 或端到端加速数字。
- **相邻 rollout 与执行基础设施：** §5.2.3 描述 token-granular WAL、保存未完成请求 KV cache 并在恢复时续接；报告指出从头重采样可能产生 length bias。§5.2.4 描述百万 token rollout 的轻量 metadata 与 per-token 字段分离、共享内存 loader 和按 mini-batch 释放字段。§5.2.5 描述 DeepSeek Elastic Compute（DSec）沙箱平台，并称单集群管理数十万并发 sandbox。属于发布方系统披露，规模与行为未独立复现。
- **证据等级与边界：** source-backed（单一模型团队技术报告的方案描述和作者主张）。该文支持“报告披露了这些设计”，不单独证明其一般因果收益或其他模型可复用性；内部路由权重、完整 GRM schema、各 effort 超参数、全量消融、线上服务成本及独立复现实验未公开。

## OPD 与 checkpoint soup 的比较证据

- **GKD 来源：** [On-policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/html/2306.13649v3)，Google DeepMind 等，arXiv v3（2024-01-17）。阅读摘要、§§3–4。论文让 student 在自生成序列上接受 teacher token-probability supervision，并允许按任务选择 forward/reverse KL 或 JSD；实验覆盖 T5 的摘要、WMT 翻译、GSM8K 算术推理及 instruction tuning。指标和收益属于作者的 T5、teacher/student、数据与训练设置，不证明大规模多教师系统可扩展。
- **MiniLLM 来源：** [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/html/2306.08543v6)，arXiv v6（2026-01-31）。阅读摘要、§§2–3 与主要分析。方法以学生完整序列分布的 reverse KL 为目标，推导 on-policy policy-gradient，并用单步分解、teacher-mixed sampling 和 length normalization 稳定/处理长序列。作者实验面向 120M–13B instruction-following 模型；对响应质量、长文本、校准和 exposure bias 的结论均绑定论文中的模型和评估。
- **Model Soups 来源：** [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://proceedings.mlr.press/v162/wortsman22a.html)，ICML 2022，PMLR 162。阅读摘要和结果概述。论文研究共享预训练 checkpoint 上以不同超参数 fine-tune 的模型权重平均，作者把其可行性联系到模型处于单一低误差区域；结果覆盖 CLIP、ALIGN、ViT-G、ImageNet 及部分 NLP/迁移任务，并称 soup 不增加推理时间/内存。
- **证据边界：** GKD、MiniLLM 与 Model Soups 的合并对象、student state 生成方式、目标函数、模型家族和 benchmark 都不同。它们可支持机制层面的比较，不能按原论文分数直接排名，也不能把视觉模型的参数平均结果当成多领域 RL 教师能无条件平均的证据。
