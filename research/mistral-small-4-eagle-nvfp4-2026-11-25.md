# Mistral Small 4、EAGLE 与 NVFP4 研究记录

研究日期：2026-11-25  
研究范围：第二十一册第 79 章 Mistral Small 4 案例的公开架构、请求级 reasoning effort、MLA KV 表示、EAGLE 草稿/验证机制、NVFP4 通用格式与该模型量化制品。  
读者任务：从固定配置和实现估算参数/cache 资源，解释 EAGLE 的目标分布校正，并区分格式资料、模型制品和部署实测。  
证据状态：公开模型配置和指定源码实现为 source-backed；论文机制为 source-backed；资源数字为基于其字段的手工推导；未下载或运行模型、代码、kernel、API、benchmark。

## 检索方向与筛选

- 主题与关键词：Mistral Small 4、Mistral4、MoE、MLA、reasoning_effort、active parameters、KV cache、EAGLE speculative decoding、tree attention、acceptance/rejection sampling、EAGLE-3、NVFP4、E2M1、E4M3 scaling。
- 来源路径：并行检查 Mistral 官方发布/产品资料、模型卡和固定配置；查 Transformers 指定代码实现；用 EAGLE 原论文和后续论文核验方法；用 NVIDIA 固定版本文档核对量化格式；另读课程讲义、技术博客和架构讲解作为教学路线/实践入口。
- 版本策略：模型配置、EAGLE 与 NVFP4 制品分别固定到查阅时可见的 revision；实现说明绑定 Mistral4 在 Transformers 中引入时的 commit。官方发布性能只按发布方模型级主张记录。
- 教学来源用于发现讲解路径，不代替原始事实；未发现于所查页面的信息只表述为“截至该次查阅未公开”，不推断为不存在。

## 核心来源及支持范围

### Mistral Small 4 发布与配置

1. **Mistral AI, “Mistral Small 4”**
   - [官方发布文章](https://mistral.ai/news/mistral-small-4/)，2026-03-16。
   - 支持：发布方称模型合并 Instruct、Magistral Reasoning、Devstral 能力；文章披露 119B total、6B active，以及包含 embeddings/output layers 时 8B；说明 40% latency 与 3x throughput 两类发布方比较。
   - 边界：这些是发布方主张；两个性能数字来自不同优化设置，不能归因给 EAGLE、NVFP4 或单个结构字段。文章没有给出可重现全部对照条件的完整协议。
2. **Mistral Docs, Mistral Small 4**
   - [产品模型页](https://docs.mistral.ai/models/mistral-small-4)与[2026-03-16 发布说明](https://docs.mistral.ai/resources/release-note/2026-03-16-mistral-small-4-release)。
   - 支持：公开模型 ID/能力、256K 上下文、推理设置和资源说明。
   - 边界：产品描述不揭示全部内部机制，也不证明任意后端对所有设置的支持。
3. **Mistral AI, Mistral-Small-4-119B-2603 model card**
   - [基础模型卡](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603)；配置核对使用[revision 7ecb947 的 config.json](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603/blob/7ecb947/config.json)。
   - 支持：119B total、约 6.5B activated/token、128 routed experts、每 token 4 个路由专家、模型卡标称 256K 上下文、文本/图像输入与文本输出；配置树含 Mistral3 条件生成外层与 Mistral4 文本子配置。
   - 所查配置字段：36 层、32 query heads、32 KV heads、q_lora_rank 1024、kv_lora_rank 256、qk_nope_head_dim 64、qk_rope_head_dim 64、v_head_dim 128、n_routed_experts 128、num_experts_per_tok 4、n_shared_experts 1、max_position_embeddings 1,048,576。
   - 边界：config 字段说明指定制品的配置，不等于训练过程或产品支持长度的完整证明。配置位置字段与模型卡 256K 标称值不同，不能据 1,048,576 单独宣称有效 1M 上下文。模型卡约 6.5B 与发布文 6B/8B 的口径未由所查资料精确对账。
4. **Hugging Face Transformers Mistral4 implementation**
   - [引入 Mistral4 的固定实现 commit 3b50327](https://github.com/huggingface/transformers/blob/3b50327/src/transformers/models/mistral4/modeling_mistral4.py)。
   - 支持：Mistral4Attention 的 MLA 路径；KV latent 与位置 key 的构造/缓存；实现先缓存 k_pass/k_rot，再扩展成逐头 attention 表示；MoE 中路由专家与共享专家输出的代码关系。
   - 边界：只代表所查 Transformers commit；不代表其他框架、Mistral 服务端或后续实现的张量布局。

### EAGLE 与推测解码

5. **Li et al., “EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty”**
   - [ICML 2024 / PMLR 页面](https://proceedings.mlr.press/v235/li24bt.html)；[全文 HTML](https://arxiv.org/html/2401.15077)，本次读取版本页面标注 v3。
   - 支持：原始 EAGLE 的第二顶部隐藏特征预测、one-step-ahead token 作为离散条件、共享目标 LM head、Tree Attention 候选验证与推测采样校正机制。
   - 边界：论文实验是其自身模型、任务和 serving 设置；不能外推为 Mistral Small 4 性能。
6. **EAGLE-3 paper**
   - [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)。
   - 用途：说明 EAGLE 家族不同代际在特征使用与训练方法上有演进。
   - 边界：不能据方法论文断定 Mistral 专属草稿制品的代际。
7. **Mistral Small 4 EAGLE model card**
   - [固定 revision 9a8ea22 的仓库](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603-eagle/tree/9a8ea22)。
   - 支持：Mistral 发布了与 Small 4 关联的 EAGLE 草稿制品。
   - 边界：所查模型卡未标明 EAGLE 代际，也未提供完整性能表或 EAGLE+NVFP4 配对验证。不同 checkpoint / precision / runtime 的组合需单独确认。
8. **Stanford MS&E 319 lecture 10**
   - [Speculative Decoding 讲义 PDF](https://web.stanford.edu/class/msande319/lectures/lecture-10-slides-animated.pdf)。
   - 用途：教学组织和接受/拒绝校正流程的第二来源。
   - 边界：用于解释算法概念，不作为 Mistral artifact 或 serving 性能证据。
9. **NVIDIA Developer Blog, “An Introduction to Speculative Decoding for Reducing Latency in AI Inference”**
   - [文章](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)。
   - 用途：候选生成、并行验证、拒绝校正和成本的工程解释入口。
   - 边界：行业文章中的其他模型与硬件测试不迁移为 Small 4 数据。
10. **Mistral Small 4 architecture gallery**
    - [Sebastian Raschka 的架构讲解](https://sebastianraschka.com/llm-architecture-gallery/models/mistral-small-4/)。
    - 用途：检查 MLA、MoE、36 层和 cache 讲解组织；cache 形状与数值另以固定配置和 Transformers 源码复核。
    - 边界：二手讲解不替代固定实现或模型卡。

### NVFP4

11. **NVIDIA Transformer Engine 2.19, NVFP4**
    - [固定版本格式文档](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.19/features/low_precision_training/nvfp4/nvfp4.html)。
    - 支持：通用 E2M1 4-bit payload、每 16 values 的 E4M3 block scaling、每 tensor FP32 global scale，以及文档讨论的权重 scale 方案。
    - 边界：通用格式定义不等于 Mistral checkpoint 的完整逐层 quantization recipe。
12. **Mistral Small 4 NVFP4 model card**
    - [固定 revision 084d625](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603-NVFP4/tree/084d625)。
    - 支持：制品被描述为由 llm-compressor 生成的 post-training quantized 变体，发布页提及 vLLM、Red Hat 与 NVIDIA 支持。
    - 边界：所查模型卡和文件说明没有完整逐层量化/排除表、校准数据、所有张量精度/scale layout 及按任务质量对照；不据通用 NVFP4 文档替它补齐这些细节。

## 推导与冲突记录

- **参数/cache 数字：** 每层 latent 256 + RoPE key 64 = 320 values/token。按指定实现 36 层计算，BF16 cache 为 36 × 320 × 2 = 23,040 bytes/token = 22.5 KiB/token；乘 262,144 token 得约 5.625 GiB。该值是结构与 dtype 假设下的静态推导，未计 allocator、并发、临时激活、图像 token、draft 状态及 workspace。
- **显式 K/V 对照：** 32 heads × (128 key + 128 value)=8,192 values/layer/token；与指定实现压缩缓存 320 values/layer/token 的元素比约 25.6。仅作表示级账本，不是延迟或吞吐倍率。
- **active parameter 口径：** 发布文报告 6B active，另称包括 embedding/output 时 8B；模型卡与产品页约 6.5B。保留差异，不猜测精确 reconcile 方法。
- **上下文口径：** 模型卡/产品页的 256K 与固定 config 的 1,048,576 位置字段分别报告，不把后者解释成已验证窗口。
- **EAGLE 兼容：** 校正机制目标是当前 verifier 的分布。量化改变 verifier 分布时，正确采样只相对于量化 verifier；不能推出与 BF16 target 完全相同。Mistral EAGLE card 未确认代际或 NVFP4 组合成绩。
- **NVFP4 存储：** 119B × 4 bits 的 59.5 GB 是忽略尺度、对齐、非量化张量和制品结构后的 payload 算术下界，不是文件体积或运行显存。
- **发布性能：** 40% lower end-to-end time 与 3x throughput 为 Mistral 发布方在两类优化设置下相对 Small 3 的模型级比较。未把它们拆给某一组件。

## 未验证事项

- 未下载制品完整权重，未运行模型、API、代码、kernel 或 benchmark；未验证 Mistral Small 4 的真实端到端延迟、量化质量或 EAGLE 接受率。
- 未确认 Mistral EAGLE artifact 的代际和完整训练/树配置，也未确认 EAGLE 与 NVFP4 联合部署的后端支持。
- 未逐条验证所有外链在未来仍可访问；本记录描述截至 2026-11-25 的阅读版本和范围。

