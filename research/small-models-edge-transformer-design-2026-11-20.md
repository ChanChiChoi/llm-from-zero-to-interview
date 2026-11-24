# 小模型、边缘 Transformer 与设备相关效率：研究证据记录

研究日期：2026-11-20。研究问题是小模型架构、边缘设备内存/执行约束和端到端效率证据如何区分。范围对应第二十一册第二十三章；不做本地设备测试，不发布通用模型排名，不提供目标设备部署操作手册。

## 检索范围

- 知识线索：`small language model edge deployment architecture depth width weight sharing`、`on-device transformer memory hierarchy model architecture latency`、`mobile mixture of experts real smartphone profiling`、`Gemma 3n PLE MatFormer parameter memory`。
- 教学线索：`MobileLLM architecture explanation on-device profiling`、`Gemma 3n developer guide PLE MatFormer`、`small language model edge capability runtime benchmark`。
- 检索与筛选：PMLR、arXiv HTML、ACL Anthology、Apple Machine Learning Research、Google DeepMind/Gemma 官方文档与 Google Developers Blog；优先阅读原始论文/报告和官方资料，配合研究/工程团队材料比较组织方式。版本与页面状态按 2026-11-20 实际访问记录。
- 未研究：未在本地手机、NPU、GPU 或边缘板上测量；未建立当前 SLM 榜单、能源成本系数或跨设备性能倍率；未把单一公开设备测试外推到所有手机或嵌入式设备。

## 来源卡与证据边界

| 来源 | 版本与阅读位置 | 支持 | 不支持 |
| --- | --- | --- | --- |
| Liu et al., [MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases](https://proceedings.mlr.press/v235/liu24ce.html) | ICML 2024 / PMLR 235；§§2、3.4–3.6、结论与附录 C | 对 125M/350M 模型比较深而窄、SwiGLU、嵌入共享、GQA 与相邻 block 权重共享；报告指定任务下能力结果，并以 ExecuTorch、iPhone 13、iOS 17.2.1、MPS 做 125M FP16 推理 profile | 不证明深而窄或权重共享对所有参数规模/芯片都更好；profile 只对应论文模型、固定设备和软件路径；API calling 与指定评测不能代表通用智能 |
| Mehta et al., [OpenELM: An Efficient Language Model Family with Open Training and Inference Framework](https://arxiv.org/html/2404.14619v2) | arXiv v2，2024-05-02；ICML 2024 Efficient Systems for Foundation Models workshop；§§2.1、4 | layer-wise scaling 可在层间非均匀分配 attention/FFN 维度；报告公开训练/评估框架和 Apple Silicon 上的 MLX 测量设置 | 结果绑定 OpenELM 模型与其预训练/评测；MacBook Pro M2 Max、64 GiB、MLX 0.10.0 的结果不代表手机/NPU；架构和数据差异未必被单一对照完全隔离 |
| Allal et al., [SmolLM2: When Smol Goes Big—Data-Centric Training of a Small Language Model](https://arxiv.org/abs/2502.02737v1) | arXiv v1，2025-02-04；§§3–5、Appendix E–G | 记录小模型数据集评估、分阶段混合/上采样和相应 benchmark 变化；可说明训练数据设计是能力结果的一部分 | 不证明某种数据混合能普遍提升能力；该工作的大规模 H100 训练结果不是端侧运行速度证据 |
| Lu et al., [Demystifying Small Language Models for Edge Deployment](https://aclanthology.org/2025.acl-long.718.pdf) | ACL 2025 Long Papers，pp. 14747–14764；§§1–3 | 研究样本为 68 个、约 100M–5B 的公开 decoder-only SLM，样本时间截止 Llama 3.2（2024-09）；同时测任务能力及两个开发边缘板的 prefill/decode 与内存，讨论 task/architecture/runtime 的差异 | 样本不覆盖 2025–2026 新模型，两个开发板不代表所有终端；论文结论和 benchmark 分布不能写成当前通用排行或普遍能力规律 |
| Google, [Gemma 3n model overview](https://gemma.ai/models/gemma-3n/) | 动态官方文档，访问于 2026-11-20；“PLE caching”“MatFormer architecture”“Conditional parameter loading” | 官方说明 Gemma 3n 的 Per-Layer Embeddings 缓存、嵌套 MatFormer 子模型和按需跳过/载入部分参数等设计 | 文档中的机制说明属于特定 Gemma 3n；effective memory load 不是所有模型可直接换算的 active-parameter 指标，也不替代独立目标设备测量 |
| Google Developers, [Introducing Gemma 3n: The developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/) | 2025-06-26；全文与开发入口 | 作为官方工程团队如何把移动架构、PLE、MatFormer 和开发路径组织给开发者的教学材料 | 发布方介绍不构成独立性能评估；宣传性性能主张需回到论文或固定设备测量 |
| Chen et al., [MobileMoE: Scaling On-Device Mixture of Experts](https://arxiv.org/html/2605.27358v1) | arXiv v1，2026-05-26；§§3.2–3.4、4.1–4.3、7 | 作者报告针对手机约束设计的 sub-billion-active MoE，并用自定义 fused MoE operator 在 Samsung Galaxy S25 与 iPhone 16 Pro 上、配合 ExecuTorch/XNNPACK 或 MLX 测量；说明稀疏结构的设备收益依赖执行 kernel、量化、负载和模型配置 | 单篇预印本的模型/基线/设备/软件结果不是独立复现，不证明 MoE 一般优于 Dense；不能把 active 参数、INT4 权重或作者报告的速度直接换算成任意设备内存/延迟 |
| Google, [Announcing Gemma 3n preview: powerful, efficient, mobile-first AI](https://developers.googleblog.com/en/introducing-gemma-3n/) | 2025-05-20；官方开发者文章 | 用作面向开发者解释 on-device 模型取舍与可组合子模型的组织样本 | 预览发布文章不是独立 benchmark，也不支持脱离模型、设备和 runtime 的性能结论 |

## 可复用的研究关系

1. “小模型”描述参数规模，“边缘”描述目标执行环境，“高效”必须相对任务目标和资源指标定义。三者没有可替换的一一对应关系。
2. 模型质量取决于容量、层间设计、训练数据与后训练等共同因素。MobileLLM 的结构实验、OpenELM 的 layer-wise scaling 与 SmolLM2 的数据消融回答的问题不同，不可把一个模型结果归因给全部小模型。
3. 结构、压缩、训练和输入处理改动作用于不同账本：GQA 减少 KV head 状态；局部或线性注意力改变信息访问/状态表示；MoE 改变 token 选择的 FFN 计算，但仍涉及总权重和路由；量化改变数值表示；蒸馏改变 student 所学函数；token 压缩减少序列同时可能丢失任务信息。参数或文件变小不自动意味着延迟、能耗或任务成本更低。
4. 端到端收益取决于目标设备的内存层级、权重/状态数据移动、算子与 kernel、runtime、上下文负载和量化支持。MobileLLM、OpenELM、Gemma 3n 与 MobileMoE 的报告分别显示不同实现与设备路径，结果必须绑定原实验。
5. 小模型评估需要把能力和系统测量放在一起。ACL 2025 的 68-model/two-board 研究同时覆盖多个任务与设备指标，但样本时间和设备范围有限；它适合作为评估维度参考，不是 2026 模型目录或跨设备排行榜。

## 结构组织参考与待核边界

- MobileLLM 以小参数预算问题开篇，沿结构选择、下游能力和端侧 profile 推进；其论证适合辨认“参数分配”与“设备执行”之间的中间证据。
- OpenELM 把非均匀层级参数分配与训练数据、评测和硬件 profile 放在同一报告中；引用结果时仍需分开架构、数据和设备因素。
- ACL 2025 SLM 研究同时组织模型能力、训练数据和 runtime 成本，支持将能力曲线与边缘设备执行分开分析。
- Gemma 3n 官方文档将 PLE、MatFormer 与条件加载按内存/计算效果解释；这提供读者可跟随的机制地图，证据层级仍是官方产品技术说明。
- MobileMoE 预印本的设备结果需要和其自定义 kernel、INT4/QAT、手机型号、后端与负载一起阅读。版本或软件更新可能改变比例；本记录不代表本地复现。
