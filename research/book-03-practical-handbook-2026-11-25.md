# 第三册正文修订的研究证据

研究日期：2026-11-25。以下只记录可用于第三册技术内容的知识点、来源和适用边界，不记录章节审阅状态或修改过程。

## 推理缓存和辅助解码

- Transformers 的当前主线生成文档列出 DynamicCache、StaticCache、offloaded、offloaded_static 和 QuantizedCache 等实现；`max_cache_len` 仅适用于静态缓存，并影响预分配和重复生成调用的重编译行为。[Generation API](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
- Transformers 辅助解码文档把 speculative decoding 描述为 draft 提议、target 一次验证；当前指南说明其适用条件、批处理限制和可用的 prompt lookup、自 speculative、MTP 等路径。这些是不断演进的框架行为，章节中的 pinned 版本示例不能据此外推到所有版本或 runtime。[Assisted decoding](https://huggingface.co/docs/transformers/main/assisted_decoding)
- Transformers 当前 assisted decoding 文档将 `assistant_ensemble_weight` 对应的 static ensemble verification 描述为用 target/draft 混合分布验证候选，从而放宽标准 speculative decoding 的无损分布保证；文档还说明它需要 assistant logits，不兼容 prompt lookup decoding。Transformers v5.23.0 Generation API 有该参数说明。此路线是接受分布偏差来换取接受率变化的可选质量/性能权衡，接受率提升本身不能证明端到端延迟或业务质量改善。[Assisted decoding](https://huggingface.co/docs/transformers/main/assisted_decoding) · [v5.23.0 Generation API](https://huggingface.co/docs/transformers/v5.23.0/en/main_classes/text_generation)
- vLLM 当前 cache 配置文档包含与缓存类型、block 对齐及不同后端相关的实现参数。此类 API 细节属于版本和后端特定信息；稳定的教学核心应保持在缓存容量、调度和目标负载实测上。[vLLM cache configuration](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)

## 训练循环与项目判断

- PyTorch 的入门训练教程把训练集和未参与更新的数据分别评估，并建议记录两者的 loss；它提供的是分类教学示例，支持“训练目标下降不等于泛化已经改善”的检查路径，不直接替代语言模型的任务级评估。[Training with PyTorch](https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html)
- PyTorch 的模型保存教程区分只用于推理的模型权重与用于续训的 checkpoint；续训需要恢复 optimizer state 等训练状态，并切回 `train()` 模式。项目还应按其训练流程保存 scheduler、GradScaler、随机数和数据进度等确实影响后续更新的状态。[Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

## Agent 工具和授权边界

- MCP 2025-06-18 规范明确提醒工具可以代表任意代码执行，工具描述不一定可信；规范本身不能在协议层强制所有安全原则，集成方应负责同意流程、授权、访问控制和数据保护。[MCP 2025-06-18 specification: Tool Safety](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2025-06-18/index.mdx)
- MCP 2026-07-28 授权教程描述了基于 OAuth 的受保护资源发现、`resource` 参数和 PKCE 流程。它是当前教程版本的协议实现细节，不意味着 schema 校验或连接授权本身就能取代每次动作的业务权限检查。[MCP authorization tutorial](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/docs/2026-07-28/tutorials/security/authorization.mdx)
- NIST AI RMF 官方页面将 Generative AI Profile 标为 NIST-AI-600-1，并说明 AI RMF 1.0 正在修订；引用时应标出版本和自愿风险管理性质，不能把框架建议写成通用法律义务。[NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework)

## RAG 评估和引用支持

- RAGAS 原始论文将 faithfulness（回答主张是否能从给定上下文推得）、answer relevance（是否回应问题）与 context relevance（上下文是否聚焦）视为不同评估维度；论文中的 faithfulness 由 LLM 抽取回答主张并判断上下文支持。它说明指标的操作定义，不证明自动 judge 与人工事实核验等价。[RAGAS paper](https://aclanthology.org/2024.eacl-demo.16.pdf)
- 因此，评估代码中引用编号合法性、引用 chunk 是否命中 gold ID 属于结构代理；claim 是否真正受引用语义支持仍需逐 claim 标注或校准评审。相似度、引用存在和总体 answer score 不应合并解释为真实性。

## Diffusion 和语音接口边界

- Diffusers 的 DDPMScheduler 当前 API 接受不同 variance 类型和 `epsilon`、`sample`、`v_prediction` 等预测目标。手算推导必须写清所选目标、噪声日程和 scheduler 假设，不能把 epsilon 参数化当成全部模型的默认。[DDPMScheduler API](https://huggingface.co/docs/diffusers/api/schedulers/ddpm)
- OpenAI Whisper 模型卡描述其 ASR 与语音翻译能力、不同模型规模和 turbo 版本；模型卡介绍不能替代目标语言、口音、噪声、时长、时间戳和关键字段上的本地测试。[Whisper model card](https://github.com/openai/whisper/blob/main/model-card.md)

## 平台生命周期信息

- OpenAI 官方迁移页及 deprecations 页面显示，托管 Evals 在 2026-10-31 进入只读，计划于 2026-11-30 关闭；这是 2026-11-25 核对的服务生命周期事实，需随官方时间表更新。它不改变独立评测集、固定 rubric 和逐样本比较这些通用实践。[Evals migration guidance](https://developers.openai.com/api/docs/guides/evals) · [Deprecations](https://developers.openai.com/api/docs/deprecations)
