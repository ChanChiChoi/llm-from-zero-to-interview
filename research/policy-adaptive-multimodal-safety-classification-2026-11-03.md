# 策略自适应多模态安全分类：来源证据

核验日期：2026-11-03。本文只记录新查资料支持的事实、版本和证据边界，不记录章节审阅进度或编辑决定。

## Shieldstral

- **官方模型卡：** [mistralai/Shieldstral-1.0-3B](https://huggingface.co/mistralai/Shieldstral-1.0-3B)，核对 Key Features、How It Works、benchmark 表及两种推理示例。模型卡称其为 3B policy-adaptive 多模态分类器，基于 Ministral-3-3B-Base-2512 和原生 Pixtral 视觉编码器，单次前向输出 yes/no；输入可为文本、图像或图文。连续分数由 yes/no token logprob 重新归一化。示例 helper 以 `score > threshold` 判违规，恰好等于阈值时不标记；参考代码将缺失 yes/no token 的 logprob 初始化为 `-10.0`，这只是示例实现的哨兵值，不能视为通用缺测处理。
- **官方产品目录：** [Shieldstral 1.0](https://docs.mistral.ai/models/shieldstral-1-0)，页面标记日期为 2026-08-04，显示托管产品处于 Public Preview、权重许可证 Apache 2.0、参数量 3.8B、上下文 32k。该产品页的参数规格与模型卡/论文所称“3B”不是同一精度口径。
- **官方生命周期说明：** [Model lifecycle policy](https://docs.mistral.ai/inference/model-lifecycle)，范围明确为 Mistral Serverless API。该页称 Public Preview 可以静默更新，没有保证进入 General Availability 的路径，也可能在 GA 前退役。此条款描述托管 API 生命周期，不等同于已下载权重的开源许可证。
- **技术报告 v2：** [Shieldstral, arXiv:2607.25857v2](https://arxiv.org/abs/2607.25857v2)，2026-08-04 修订。摘要、任务定义、数据构造及实验章节支持：作者把审核表达成 `<Instruct>`、`<Query>`、`<Document>` 条件下的 yes/no 判断；报告约 5,410 万个样本（4,520 万开源文本、440 万合成对比文本、450 万多模态）；作者报告文本平均 F1 84.9%、多模态平均 F1 83.8%、细粒度政策评测 F1 91.3%。这些是作者在其任务、数据、阈值和配置下报告的实验结果，不是独立复现或业务系统漏报率。
- **模型卡 benchmark 限定：** Shieldstral 表格使用 0.5 阈值；多模态 LlavaGuard 一项注明部分测试图像不可用；不同比较模型的 reasoning 配置也不同。多项数字不可并成一个总准确率或脱离原设定排名。

## 策略翻转与图像 guardrail

- **SafeEditBench：** [Towards Policy-Adaptive Image Guardrail: Benchmark and Method, arXiv:2603.01228v2](https://arxiv.org/abs/2603.01228v2)，页面显示 v2 于 2026-03-31 修订。论文摘要说明，它使用视觉语义相近的安全/不安全图像编辑对，并由人工在五种政策下标注，用来检验跨政策泛化。论文同时提出 SafeGuard-VL；其方法和结果不等于 Shieldstral 的结果。
- **PolicyShiftBench：** [PolicyShiftGuard: Benchmarking and Improving Policy-Adaptive Image Guardrails, arXiv:2607.05910v1](https://arxiv.org/abs/2607.05910v1)，2026-07-07 提交的预印本。论文摘要及基准构造章节报告 2,000 个评测实例、265 张静态图像、28 种政策变体、每图平均 7.55 个 policy-conditioned prompt。作者定义 Policy Shift Score（PSS）：先按同一图像和风险类别分组，只对金标相反的政策翻转样本配对；一对只有两边预测都正确才得 1；先在组内平均，再对有效组宏平均。该定义旨在度量政策翻转敏感度，而非一般 safe/unsafe 分类能力。
- **PolicyShiftBench 发布元数据：** [dataset_stats_sanitized.json](https://huggingface.co/datasets/PolicyShiftBench/PolicyShiftBench/blob/main/dataset_stats_sanitized.json)，evaluation summary 显示 265 张图中 262 张同时具有 pass 与 block 标签。该数值来自数据发布元数据，不是论文正文的模型结果。
- **PolicyShiftBench 划分和限制：** Shift track 留出政策定义；论文称训练图像和评测图像不重叠，但 Adaptive 与 Shift 两条评测轨道共享 17 张图像。基准限定为静态图像与显式政策文本，不含视频、音频、多轮对话或动态网页；政策主要为英语结构化审核风格，政策目录有限。因此“未见政策”不等于“所有评测图像彼此独立”或生产中的任意政策泛化。
- **SafeVision：** [SafeVision, arXiv:2510.23960v1](https://arxiv.org/abs/2510.23960v1)，2025-10-28 提交的预印本。摘要把该方法描述为推理时适应政策、提供解释的图像 guardrail，并提出 VisionHarm 数据集；其 benchmark 性能陈述属于作者自己的模型和评测设置。

## 动态多模态规则

- **SingGuard v3：** [SingGuard: A Policy-Adaptive Multimodal LLM Guardrail with Dynamic Reasoning, arXiv:2606.22873v3](https://arxiv.org/abs/2606.22873v3)，2026-06-25 修订。摘要称活动策略作为运行时输入，模型逐条检查规则并预测安全标签与触发规则；论文提出 fast、hybrid、slow 三种推理模式。SingGuard-Bench 报告 56,340 个例子和 80 多种细分风险，包含图像与文字分别无害、跨模态组合后才形成风险的样本。作者在自己的 dynamic-rule 评测中报告 Qwen3-VL-8B policy-following accuracy 0.6465、SingGuard-slow 0.7415；该对比绑定其评测和模型设置，不可与 Shieldstral 或 PolicyShiftBench 的 F1/PSS 横比。
- **SingGuard 局限：** 论文第 7 节称判断依赖清楚且无歧义的活动规则；benchmark 不能覆盖现实政策长尾与快速出现的滥用模式；合成数据和模型辅助标注仍可能引入 teacher bias、标注伪影和分布差异，hybrid early-exit 的 token 置信度在分布变化下也可能校准不足。

## 证据等级与未验证范围

模型卡和官方产品/生命周期文档支持公开字段及托管 API 生命周期说明；论文支持作者提出的方法、实验设计、作者报告结果和自述局限；数据集元数据支持其记录的发布统计。以上材料不证明特定企业系统的误报成本、跨租户隔离、工具权限或端到端安全，也没有在本地复现上述模型或 benchmark。
