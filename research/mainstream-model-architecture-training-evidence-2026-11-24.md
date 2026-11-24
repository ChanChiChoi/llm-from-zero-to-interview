# 主流模型架构、训练与公开证据横向比较

## 研究问题与范围

本次研究为第二十一册第 59 章服务，将第 51—58 章已经修订的固定版本案例放进一致的比较坐标：模型身份/版本、架构、训练与任务适配、模态/上下文接口、评测和披露边界。目标是识别可并列的变化和不可归因的部分，不建立综合排名，不评价 2026 年当前产品或部署性能。研究日期：2026-11-24。

本轮网页搜索实际查询：`site:arxiv.org GPT-3 Chinchilla LLaMA Mistral Mixtral technical report architecture training original paper`；`site:arxiv.org Qwen2.5 Gemma 2 DeepSeek V2 V3 R1 technical report`；`site:openai.com GPT-4 technical report architecture details not disclosed GPT-4V system card`；`Stanford CS336 2026 Language Modeling from Scratch course lectures architecture scaling training evaluation`。另外复读了本书相关章节及其研究记录中的原始论文、模型报告、模型卡与课程材料；事实只按固定来源支持范围采用。

## 主要发现

1. **模型名称和变化层次需要拆开。** GPT-3 论文说明 zero/one/few-shot 评测不进行梯度更新；InstructGPT 报告的 SFT、偏好建模和 RLHF 是后训练阶段；R1 论文把 R1/R1-Zero 建在 V3-Base 上。这些材料分别涉及上下文使用方式、训练阶段和 checkpoint 关系，不能被合称为 backbone 架构创新。
2. **架构组件的组合和作用位置不同。** LLaMA 报告列出 RMSNorm、SwiGLU、RoPE 等 Transformer 组件的组合；Mistral 7B 使用 GQA 与 sliding-window attention；Mixtral 将 Mistral FFN 路径替换为每 token 路由到两个专家的 MoE；DeepSeek-V2 报告 MLA 与 DeepSeekMoE。总参数、每 token 激活参数、直接注意力窗口、历史状态表示和系统成本是不同字段，不能相互代替。
3. **规模和计算预算结论各有目标。** GPT-3 的 few-shot 研究是固定模型在提示中呈现任务示例的评估；Chinchilla 研究固定计算预算下模型参数与训练 token 的配置。两者不组成一项受控架构比较。PaLM 的多 TPU 系统报告也不能仅凭参数量与其他报告推导相同硬件利用率。
4. **训练报告的披露维度不等价。** Qwen2.5、Llama 3.1 和 Gemma 2 分别公开不同程度的数据规模/混合、训练阶段、蒸馏、基础设施与许可信息。token 数不能单独说明数据质量或训练效率；开放权重不等于完整语料和训练流程可复现。
5. **多模态接口不必然披露内部拓扑。** Gemini 报告公开了部分高层架构信息；GPT-4V 与 Claude 3 材料可支持图像输入和发布方评估范围，但没有公开完整视觉编码/融合结构。GPT-4 技术报告明确不公开模型规模、硬件、训练计算等细节。披露量不齐时，结论也必须不齐。
6. **上下文窗口和长文本任务结果不是同一指标。** Gemini 1.5 与 Claude 3 发布期材料中的窗口字段、needle-in-a-haystack 结果和不同长文本任务须保留各自版本与协议。它们不能直接合并成统一排名，也不能证明窗口内所有信息都能可靠利用。

## 主要来源及支持范围

| 来源 | 类型/版本与阅读入口 | 支持什么 | 不支持什么 |
| --- | --- | --- | --- |
| [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | GPT-3 原始论文；摘要、§2—5、contamination analysis；详细读法见[第 51 章研究记录](gpt-lineage-scaling-icl-evidence-2026-11-23.md) | GPT-3 的 zero/one/few-shot 评测、免梯度更新条件、作者报告的强弱项与污染分析 | 任意任务上的普遍迁移，或 ICL 的单一内部机制 |
| [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | Chinchilla 原始论文；固定计算预算与参数/数据配置分析；并参见[Chinchilla/PaLM 研究记录](chinchilla-palm-compute-optimal-evidence-2026-11-23.md) | 指定训练与评估条件下的 compute-optimal 配置主张 | 跨数据质量、目标、训练系统和推理约束的永久比例规则 |
| [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) | PaLM 原始论文；模型/系统设置与评估；详见上方 Chinchilla/PaLM 记录 | PaLM 报告的模型、数据、TPU v4 系统配置及对应结果 | 仅按参数量比较不同系统的训练吞吐、利用率或质量 |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | Ouyang 等，原始论文；摘要、训练方法与评估部分；详见[第 53 章研究记录](instructgpt-chatgpt-rlhf-evidence-2026-11-24.md) | 指定 InstructGPT 版本的监督微调、偏好数据、奖励模型/RLHF 路线和作者评估 | 由此推断模型骨干发生改变，或将发布期 ChatGPT 当作同一实验对象 |
| [Introducing ChatGPT](https://openai.com/index/chatgpt/) | OpenAI 官方发布说明，2022-11-30；方法与限制部分 | 发布期 ChatGPT 的多轮对话接口、基于 GPT-3.5 的模型描述及其披露的 RLHF 训练概述 | 独立隔离某个后训练阶段的因果效果，或推断未公开的 backbone 细节 |
| [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | 原始论文；§2.2；详见[LLaMA 家族研究记录](llama-family-architecture-evidence-2026-11-24.md) | LLaMA 报告中 RMSNorm、SwiGLU、RoPE 等组件及训练模型设置 | 声称 LLaMA 发明了这些组件，或把模型结果归因于其中单项 |
| [Mistral 7B](https://arxiv.org/abs/2310.06825) 与 [Mixtral of Experts](https://arxiv.org/abs/2401.04088) | 原始论文及固定模型配置；详见[Mistral/Mixtral 研究记录](mistral-mixtral-architecture-evidence-2026-11-24.md) | Mistral 7B 的 GQA/SWA；Mixtral 8x7B 与 Mistral block 的关系、8 专家/top-2 路由及报告参数口径 | 将 active parameters 直接等同于总权重、通信、显存或端到端延迟 |
| [Gemini 1.0 技术报告](https://arxiv.org/abs/2312.11805)、[Gemini 1.5 技术报告 v1](https://arxiv.org/abs/2403.05530v1)、[GPT-4 技术报告](https://arxiv.org/abs/2303.08774) | 发布期原始报告；版本与 NIAH 口径详见[Gemini/GPT-4V/Claude 3 研究记录](gemini-gpt4v-claude3-disclosure-evidence-2026-11-24.md) | 各自实际公开的结构描述、模型类别、接口/实验和未披露项 | 由发布 API、窗口或单项基准补足报告没有披露的内部结构，或将不同协议结果排名 |
| [GPT-4V System Card](https://cdn.openai.com/papers/GPTV_System_Card.pdf) 与 [Claude 3 Model Card](https://assets.anthropic.com/m/61e7d27f8c8f5919/original/Claude-3-Model-Card.pdf) | 发布期 system/model card；见 Gemini/GPT-4V/Claude 3 研究记录中的阅读位置 | 各卡明确的图像输入、公开的高层训练信息和作者评测范围 | 未公开的视觉 encoder、融合点、token 路径或跨供应商直接排名 |
| [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)、[The Llama 3 Herd of Models v3](https://arxiv.org/abs/2407.21783v3)、[Gemma 2](https://arxiv.org/abs/2408.00118) | 固定报告版本与披露差异详见[Qwen/Llama/Gemma 研究记录](open-weight-training-engineering-qwen25-llama31-gemma2-2026-11-24.md) | 对应报告披露的预训练数据规模、训练阶段、蒸馏和模型设置 | 单靠 token 数推导数据质量、通用效率或跨报告因果；权重开放不等于语料全公开 |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434v5)、[DeepSeek-V3](https://arxiv.org/abs/2412.19437v2)、[DeepSeek-R1](https://arxiv.org/abs/2501.12948v2) | 固定报告修订版及阅读位置见[DeepSeek 研究记录](deepseek-v2-v3-r1-architecture-training-2026-11-24.md) | V2/V3 架构关系、V3 MTP、R1 基于 V3-Base 的后训练与蒸馏模型基座 | 将 R1 当作独立 backbone 创新，或将不同报告的参数/评估结果拼成受控比较 |

## 组织与教学来源

[Stanford CS336 Spring 2026 官方课程页](https://cs336.stanford.edu/)把语言模型学习组织为模型/架构、训练、缩放、评估与部署等相连环节，并在课程安排中单列 scaling laws。这里仅借鉴它把结构与训练/评估分层又相互承接的课程组织；课程页面不是上述模型的事实来源，也不证明某种编排对所有读者效果更好。

## 采用边界与待核项

- 每条历史比较回到该章所用的固定版本来源，不把旧报告扩展为家族当前实现。
- 参数、激活量、上下文窗口、训练 token、benchmark 和授权条款是不同维度；只有数据与评估设置对齐时才讨论数值比较。
- 若某来源没有披露结构、硬件或训练细节，继续标记为未知。没有作者消融或匹配的独立实验时，不把整体结果归因于单个模块。
- 读者可据矩阵定位证据和边界；尚未完成同硬件复测、统一 harness 评估或当前产品行为核验，因此本研究不提供运行性能、质量排名或部署建议。
