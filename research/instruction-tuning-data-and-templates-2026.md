# 指令微调数据、模板与后训练方法关系：来源证据

记录日期：2026-10-28

本文件保存联网核验得到的知识点、来源版本和证据范围；不记录书内修改过程或逐讲进度。

## 指令微调、示范数据和偏好信号

- Ouyang et al., “Training language models to follow instructions with human feedback,” arXiv:2203.02155（2022），[论文](https://arxiv.org/abs/2203.02155)。
  - 实际阅读范围：摘要、引言、§3 方法、示范监督与 PPO 训练说明、评估和局限相关章节。
  - 来源支持：论文从 GPT-3 开始，先用标注员示范做监督微调，再收集回答比较训练奖励模型，最后以 PPO 优化；论文报告的偏好结果对应其 API 提示分布和标注设置，并同时讨论公开 NLP 基准。
  - 证据等级：source-backed（论文方法、实验与作者讨论）。
  - 适用边界：这是具体 GPT-3/InstructGPT 配方，实验人群、提示和任务不代表普遍用户偏好；论文没有证明所有后训练都必须按该三阶段顺序。

- Chung et al., “Scaling Instruction-Finetuned Language Models,” arXiv:2210.11416（2022），[论文](https://arxiv.org/abs/2210.11416)。
  - 实际阅读范围：摘要、引言和实验概要。
  - 来源支持：论文研究任务数量、模型规模及 CoT 数据对 PaLM、T5、U-PaLM 等模型在指定未见任务评估中的影响；作者报告了多个 benchmark 的改善。
  - 证据等级：source-backed（论文实验）。
  - 适用边界：结果随模型、任务集合、提示和 benchmark 而定；不证明任务或样本越多就会在任何目标上单调变好。

## 数据量、质量、多样性和合成数据

- Zhou et al., “LIMA: Less Is More for Alignment,” arXiv:2305.11206（2023），[论文](https://arxiv.org/abs/2305.11206)。
  - 实际阅读范围：摘要、§4 人类评估、§5 数据量/质量/多样性消融、§6 多轮对话。
  - 来源支持：论文在 65B LLaMA 上用 1,000 条精选单轮问答进行 SFT；研究还报告在其 7B 设定中至少 2,000 条样本有助于训练稳定性。单轮模型的 10 段多轮测试中，6 段在三轮内未能跟随提示；加 30 段对话样本后，同一小型评估中表现改善。
  - 证据等级：source-backed（特定模型、数据构造和小型人工/模型评估）。
  - 适用边界：不能据此推断普遍最小训练集规模；测试集较小，模型和数据集有限，论文自身也讨论评估与分布外限制。

- Wang et al., “Self-Instruct: Aligning Language Models with Self-Generated Instructions,” ACL 2023，[论文](https://arxiv.org/abs/2212.10560)。
  - 实际阅读范围：摘要、§2 方法、§3 GPT-3 生成数据统计与质量检查、§4 评估。
  - 来源支持：流程从少量人工种子任务出发，让 GPT-3 生成任务、输入和目标输出，再过滤无效或相似样本；该实验过滤后包含 52,445 条指令和 82,439 个实例。作者手工抽查 200 个实例，发现生成样本仍有噪声；论文在其 GPT-3 与 Super-NaturalInstructions 评测设置中报告了相对基线的提升。
  - 证据等级：source-backed（论文算法、单一生成模型实验和人工抽查）。
  - 适用边界：不能外推为合成数据总是成本更低或比人工数据更好；生成调用、过滤、评估和人工检查仍需资源，质量结果限于该生成器与采样流程。

- Bukharin et al., “Data Diversity Matters for Robust Instruction Tuning,” Findings of EMNLP 2024，[论文页面](https://aclanthology.org/2024.findings-emnlp.195/)。
  - 实际阅读范围：摘要、QDIT 的实验与结论简介。
  - 来源支持：作者提出 Quality-Diversity Instruction Tuning，在其多个指令数据集实验中观察到质量与多样性的取舍，并报告提高多样性改善了最差情形的指令跟随表现。
  - 证据等级：source-backed（同行评审论文的受控实验）。
  - 适用边界：质量和多样性的操作化定义、模型、数据集与指标会影响结果；不构成通用数据筛选排序或配方。

## Chat template 与 assistant-only loss

- Hugging Face Transformers, “Multimodal chat templates,” v5.21.0，[固定版本文档](https://huggingface.co/docs/transformers/v5.21.0/chat_templating_multimodal)，访问日期 2026-11-11。实际阅读范围：多模态消息 `content` 列表、Processor 处理职责、`apply_chat_template()` 输出及视频输入部分。文档说明多模态内容由不同类型项目组成，Processor 负责预处理、分词和模板应用。证据等级：source-backed（固定版本官方文档）。适用边界：支持 Transformers v5.21.0 的文档接口；页面提示另有 v5.22.2，但本记录未据此核验该版本行为。
- Hugging Face Transformers, “Chat templates,” v5.17.0，[官方文档](https://huggingface.co/docs/transformers/v5.17.0/chat_templating)；固定源 revision 见[文档源码](https://github.com/huggingface/transformers/blob/e598fbad926d80bc356c0c4d96532030e3dfdb62/docs/source/en/chat_templating.md)。
  - 实际阅读范围：消息序列化、模型模板差异、`add_generation_prompt`、`continue_final_message` 和训练数据格式说明。
  - 来源支持：chat template 将消息对象转成模型格式化序列；不同模型的控制 token 可不同；generation prompt 常添加 assistant 起始标记，但并非所有模型都需要；训练已完成的对话通常不添加空生成前缀。
  - 证据等级：source-backed（固定版本官方文档）。
  - 适用边界：这是 Transformers v5.17.0 文档示例；模型行为和性能差异仍需对目标 checkpoint、模板与任务实测。

- Hugging Face TRL, “SFT Trainer” and “Chat Templates,” v1.16.0，[SFT Trainer](https://huggingface.co/docs/trl/v1.16.0/sft_trainer)，[Chat Templates](https://huggingface.co/docs/trl/v1.16.0/chat_templates)，访问日期 2026-10-28。
  - 实际阅读范围：数据格式、completion-only/assistant-only loss 参数和模板训练标记说明。
  - 来源支持：该版本将 `assistant_only_loss` 定义为只对 assistant 响应计算 loss；相应功能要求模板正确返回 assistant mask，文档用 `{% generation %}` 与 `{% endgeneration %}` 标记说明模板需求，并介绍部分受支持模板的修补。
  - 证据等级：source-backed（TRL v1.16.0 官方接口文档）。
  - 适用边界：这是框架的可选实现，不代表所有 SFT 必须使用 assistant-only loss；模板是否产生预期 mask 仍需检查目标数据和实际输出。
  - 补充核验（2026-11-11）：同一版 SFT Trainer 文档还注明，vision dataset 使用 `assistant_only_loss` 时需要 `transformers>=5.18.0`，并要求 conversational language-modeling dataset。该依赖门槛绑定 TRL v1.16.0 的文档实现，不能外推为其他训练器或更高版本的通用约束。

## 后训练方法关系的 2026 年视角

- Zhao et al., “Large Language Model Post-Training: A Unified View of Off-Policy and On-Policy Learning,” arXiv:2604.07941 v2（2026-04-16），[论文](https://arxiv.org/abs/2604.07941)。
  - 实际阅读范围：v2 摘要与方法组织说明。
  - 来源支持：该综述预印本按轨迹来源讨论 off-policy 与 on-policy，并将后训练方法视为对行为分布的不同干预角色。
  - 证据等级：source-backed（综述作者提出的概念框架）。
  - 适用边界：这是研究综述框架，不是经过统一实验验证的唯一方法分类或生产配方。

- Khatri et al., “The Art of Scaling Reinforcement Learning Compute for LLMs,” ICLR 2026，[会议论文页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/75ea88aca02640cca54c1231db8cacdd-Abstract-Conference.html)。
  - 实际阅读范围：ICLR 摘要。
  - 来源支持：摘要报告超过 400,000 GPU-hours 的研究，并分析损失聚合、归一化、课程和 off-policy 等配方选择对其 RL compute scaling 实验的影响。
  - 证据等级：source-backed（会议论文摘要）。
  - 适用边界：摘要所述性能曲线、配方和效率结论限于论文任务、模型和评估口径；不构成所有 LLM 强化学习的统一排名。

## 多模态指令微调和任务论文的固定版本

2026-11-11 核对 arXiv 记录页的版本历史与摘要，固定正文引用入口：

- Liu et al., “Visual Instruction Tuning” (LLaVA)，arXiv v2，2023-12-11 修订：[v2](https://arxiv.org/abs/2304.08485v2)。该版本记录用于定位论文描述的数据生成、模型连接和训练阶段；不能支持任意部署或业务质量结论。
- Dai et al., “InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning”，arXiv v2，2023-06-15 修订：[v2](https://arxiv.org/abs/2305.06500v2)。摘要支持其多任务数据整理和 instruction-aware Query Transformer 的研究定位；实验结论仍绑定模型和评估集。
- Liu et al., “Improved Baselines with Visual Instruction Tuning” (LLaVA-1.5)，arXiv v2，2024-05-15 修订：[v2](https://arxiv.org/abs/2310.03744v2)。引用用于核对其公开基线与训练设置，不能替代目标业务复测。
- Zhu et al., “MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models”，arXiv v2，2023-10-02 修订：[v2](https://arxiv.org/abs/2304.10592v2)。摘要支持其视觉编码器、冻结语言模型与单投影层对齐及两阶段数据路线的研究背景；不证明这些配置在其他模型上最优。
- Singh et al., “Towards VQA Models That Can Read” (TextVQA)，arXiv v2，2019-05-13 修订：[v2](https://arxiv.org/abs/1904.08920v2)。摘要定义依赖图中文字作答的任务和数据集；不等于通用 OCR 质量保证。
- Masry et al., “ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning”，arXiv v1，2022-03-19：[v1](https://arxiv.org/abs/2203.10244v1)。摘要支持图表视觉读取与逻辑推理任务定位；分数绑定发布数据与评分口径。
- Liu et al., “MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models”，arXiv v5，2024-06-19 修订：[v5](https://arxiv.org/abs/2311.17600v5)。摘要支持图像相关操纵风险的多模态安全评估动机；不替代目标系统工具权限或副作用检查。
- Gebru et al., “Datasheets for Datasets”，arXiv v8，2021-12-01 修订：[v8](https://arxiv.org/abs/1803.09010v8)。摘要提出记录数据集动机、组成、采集过程与推荐用途的文档框架；框架不是法规，也不核验具体填报内容。

本节只固定引用修订版本和摘要范围；若正文需要更细的论文机制、实验数字或局限，应回读相应版本的正文与附录。

## 生命周期

指令微调、指令数据构造、chat template 与 assistant-only mask 的机制：active。论文数值、模型行为、框架 API 和模板修补均依赖来源中的模型、任务、版本和评估条件。
