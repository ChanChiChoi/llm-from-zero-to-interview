# Weight Quantization Formats and PTQ Evidence (2026-10)

检索日期：2026-10-29；补充核验日期：2026-11-02。本文记录低位宽权重表示、FP8/FP4 格式和代表性 LLM 权重量化方法的来源事实及证据边界。本文不是书籍审阅或修改日志。

## 已核验事实

### FP8 表示与缩放

- [NVIDIA Transformer Engine 2.20.2 FP8 Current Scaling 文档](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html)，访问于 2026-10-29，阅读“FP8 data type”“Scaling factors”“FP8 Current Scaling quantization” §§，支持 E4M3 由 1 符号位、4 指数位、3 尾数位构成，最大有限幅值为 448；E5M2 由 1/5/2 位构成，最大有限幅值为 57,344，动态范围更大而尾数精度更低。该 recipe 每个 tensor 使用一个 FP32 scale，并以绝对最大值计算 scale、再舍入到 FP8 表示。
- 证据类型：官方库文档；范围只覆盖 Transformer Engine 2.20.2 描述的格式和 recipe。其他设备上的 dtype variant、scale 粒度、舍入和饱和行为仍应按其官方格式/运行时资料检查。

### NVFP4 recipe

- [NVIDIA Transformer Engine 2.20.2 NVFP4 文档](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)，访问于 2026-10-29，阅读“Data Format”“Scaling Factor Computation”“2D weight scaling” §§：该 NVFP4 payload 为 E2M1（1 符号位、2 指数位、1 尾数位），最大绝对值为 6；一维方案用 E4M3 block scale 覆盖连续 16 个值，并用一个 FP32 global scale 覆盖 tensor。权重还可采用 16×16 的二维 scaling。
- 证据类型：官方库文档。E2M1 是标量码本；block 轴、scale 存储、打包、硬件和 kernel 属于具体 recipe，不可据此推断所有 FP4 格式相同或所有部署均采用 NVFP4。

### GPTQ

- [GPTQ arXiv v2](https://arxiv.org/abs/2210.17323v2)，2023-03-22 版本，访问于 2026-10-29；阅读摘要、方法中 layer-wise reconstruction、Hessian/inverse-Hessian 更新和 calibration data 相关段落，以及实验设置说明。论文将 GPTQ 定义为利用近似二阶信息的一次性权重量化方法；按层重建输出，并在顺序量化权重时用 Hessian 相关信息将误差传播到剩余权重。论文实验的模型、数据、位宽和 GPU 结果绑定原文设置。
- 证据类型：同行评审论文（ICLR 2023）；方法机制可由论文支持，速度和质量数字不代表其他模型、任务、group size、kernel 或硬件的通用结果。

### AWQ

- [AWQ MLSys 2024 论文](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf)及 [arXiv HTML](https://arxiv.org/html/2306.00978)，访问于 2026-10-29，阅读摘要及 HTML 中关于 salient channels、activation statistics、equivalent scaling 和 scaling 搜索目标的 §§。论文依据校准激活分布识别敏感通道，通过权重通道与输入通道的等价缩放降低敏感权重的量化误差；其方法属于低比特 weight-only quantization。
- 证据类型：同行评审论文；“更快/更准”结论应限定于论文模型、任务、量化设置和 kernel。激活感知不表示该方法也必须把 activation 量化，也不等于将敏感权重持久保留成 FP16。

### QAT 的 fake quantization

- [PyTorch Quantization-Aware Training for Large Language Models](https://pytorch.org/blog/quantization-aware-training/)，官方技术博客，访问于 2026-10-29，阅读 QAT 定义段及相邻示例。该文将 fake quantization 描述为在浮点权重和/或 activation 上模拟量化数值，而不是在训练前向时直接把张量永久转换成目标低位宽类型。
- 证据类型：官方框架说明。博客不等于特定 LLM 的完整 QAT recipe；训练成本、可导出的 artifact、目标 backend 和 kernel 仍需单独核验。

### INT4 runtime 与量化接口的版本化说明

- [TensorRT-LLM 1.0.0 Numerical Precision](https://nvidia.github.io/TensorRT-LLM/1.0.0/reference/precision.html)，访问于 2026-11-02，阅读 W4A16/W8A16 与 GPTQ/AWQ 说明：该版本将 W4A16/W8A16 描述为在线性层中按需反量化权重，并说明 GPTQ/AWQ 使用逐组缩放和 zero offset。页面的模型支持矩阵绑定 1.0.0，不能推出其他 release、模型 shape 或部署均支持相同路径。
- [vLLM v0.32.0 Quantization 文档](https://docs.vllm.ai/en/v0.32.0/features/quantization/)，访问于 2026-11-02：文档列出量化格式入口、可选 linear backend 与硬件兼容表，并提醒支持矩阵会变化。它支持“该版本公开了这些配置入口”的结论，不证明任一模型、shape 或 batch 实际命中快速 kernel。
- [torchao 0.17 Quantized Inference](https://docs.pytorch.org/ao/stable/workflows/inference.html)，访问于 2026-11-02：页面将 INT4 weight-only 配置描述为 group-wise，并列出 group size 256、128、64、32 等选项。页面性能基准显式绑定 torch 2.9.0/vLLM 0.13.0 等条件，不能作为其他版本的部署性能证据。
- [PyTorch 2.15 FakeQuantize API](https://docs.pytorch.org/docs/2.15/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)，访问于 2026-11-02：API 定义说明训练时 fake quant 会模拟量化与反量化并使用 observer 状态；它不构成某一 LLM 的完整 QAT recipe 或 serving artifact 保证。
- **证据等级与边界**：source-backed（固定版本官方框架文档）。这些资料分别支持版本化的接口和公开实现说明；真实兼容性仍需核对目标 artifact、硬件、shape、engine/kernel 与任务质量，不将接口列表当作实测结果。

### GPTQ、AWQ、LLM-QAT 与 MARLIN 的实验条件

- [GPTQ arXiv v2](https://arxiv.org/html/2210.17323v2)，版本日期 2023-03-22，访问于 2026-11-02；阅读方法、实验设置、OPT 实验表和量化时间表：校准使用 C4 的 128 个随机 2,048-token 片段，量化实现用单张 80 GB A100；OPT-175B 在 WikiText-2 上的 16-bit baseline PPL 为 8.34，4-bit RTN 为 10.54，4-bit GPTQ 为 8.37，Table 2 报告全模型量化耗时 4.2 小时。PPL 与耗时均绑定论文模型、数据和实现，不是目标生产质量或当前 GPU 性能保证。
- [AWQ arXiv HTML v6](https://arxiv.org/html/2306.00978v6)，版本日期 2026-04-25，访问于 2026-11-02；阅读算法、OPT-6.7B 对照表和 PubMed/Enron 分布互换实验：OPT-6.7B、INT3、group size 128、WikiText-2 上 FP16/RTN/AWQ PPL 分别为 10.86/23.54/11.39。论文另报告在其 OPT-6.7B INT3-g128、PubMed 与 Enron 互换校准/评测分布设置中，AWQ PPL 增量为 0.5–0.6、GPTQ 为 2.3–4.9；测试使用 1,000 个评估样本。它只支持该实验中相对分布敏感性的比较，不能推广为无需校准或普遍更稳。
- [LLM-QAT Findings of ACL 2024](https://aclanthology.org/2024.findings-acl.26/)，访问于 2026-11-02；阅读摘要和论文表 1：该方案用教师模型生成文本及 logits 软标签，并共同处理权重、激活和 KV 量化；LLaMA-7B 的 W4A8KV4 设置在七项常识任务平均分上，RTN/LLM-QAT/16-16-16 分别为 41.2/63.0/66.2。结果绑定该论文的模型、生成数据、训练和任务配置，不表示任意 runtime 支持同一 serving path。
- [MARLIN arXiv](https://arxiv.org/abs/2408.11743)，访问于 2026-11-02；阅读论文的 per-layer 与端到端评估：作者在 A10 大矩阵层级报告 batch 16–32 约 3.9×、batch 128 约 1.5× 相对 FP16；论文的 vLLM 端到端结果约 3×，具体实验另绑定 Llama 2 7B/A10、输入/输出长度及实现。层级 kernel 数字与端到端 serving 数字不可互换，也不表示目标硬件或 batch 的通用收益。
- **证据等级与边界**：source-backed（指定论文版本、原始表格和实验段落）。以上结果用于解释算法、训练和 kernel 各自证据的不同作用；未在本地复现，不能外推为本书目标系统的实测质量、延迟或吞吐。

## 新近候选资料的证据状态

- [AAAC arXiv v2](https://arxiv.org/html/2605.08692v2) 与 [OpenReview 论文 PDF](https://openreview.net/pdf?id=ddOlFn6d0B)，2026-06-05 版本，访问于 2026-11-02；阅读摘要、§3.2、granularity/storage 说明和结论。论文提出每层学习两个标量 codebook，基于校准激活以 activation-weighted k-means 形成 codebook；每组按重建误差选择其一。scale group size 为 g、选择 group size 为 S；当 S=g 时，可把选择位放在正 scale 未用的符号位中，无额外选择位存储；当 S<g 时，选择位本身有额外成本。OpenReview 页面将其列为 ICML 2026 Workshop AdaptFM 论文；ICML 2026 官方下载页也列有该 workshop。
- **证据等级与边界**：source-backed（作者论文和会议 workshop 页面）。论文性能、格式兼容及选择位复用都依赖其 artifact/分组约定；尚未核验第三方复现或通用 runtime/kernel 支持，不能把“无额外选择位”推广到 S≠g 的设置。
- [D²Quant arXiv v2](https://arxiv.org/abs/2602.02546v2)，2026-02-06 版本，访问于 2026-10-29；只阅读 arXiv 摘要。作者摘要称其在低于 4-bit 的 weight-only PTQ 中结合面向 down-projection 的双 scale 设计与 LayerNorm 均值偏移校正；指标提升和机制有效性尚未通过正文实验表、代码或独立复现核验。
- [SliderQuant ICLR 2026 官方 proceedings 页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b08ac1692309611b6f1adce96e96816f-Abstract-Conference.html)，访问于 2026-11-02；阅读论文页摘要。作者报告浅层/深层通常比中间层更敏感，首层/末层尤为敏感；SliderQuant 用 learnable parameters 支持层间滑窗策略，并以增量方式量化层内窗口。摘要称其在多模型、文本生成、常识、数学、代码及 weight-only/weight-activation 设置中优于所比较方法。
- **证据等级与边界**：source-backed（ICLR 2026 官方论文页摘要）。本次未阅读完整 27 MB proceedings PDF、未检查其代码或独立复现；摘要中的质量结论仍绑定作者模型、位宽和任务，不支持目标 runtime/kernel 兼容或部署性能结论。
- [A Comprehensive Evaluation on Quantization Techniques for Large Language Models arXiv v4](https://arxiv.org/abs/2507.17417v4)，v4 于 2026-10-21 修订，arXiv 页面标注已被 Pattern Recognition 接收；访问于 2026-11-02，阅读 v4 摘要。作者把 PTQ 拆成量化前变换与误差补偿两步，报告更细粒度通常改善质量但增加存储、scale 格式/精度影响结果，并指出有益于 INT4 的 rotation 对 MXFP4/NVFP4 改善较少。
- **证据等级与边界**：source-backed（指定 arXiv v4 摘要及版本元数据）。摘要结果属于作者的 text-only 模型与对照配置；具体算法归因、表格条件及跨硬件适用性仍需正文和固定目标实验支持。

### OAS / MBS：MXFP4 scale 与 outlier 补偿

- [Unveiling the Potential of Quantization with MXFP4: Strategies for Quantization Error Reduction（arXiv v1）](https://arxiv.org/html/2603.08713)，v1 提交于 2026-01-30；访问于 2026-11-02。阅读 §3.1—3.5、§4.1—4.4、§5.1—5.3、Appendix A—E。作者分析 E8M0 的幂次尺度和 32→16 block 粒度，提出 Overflow-Aware Scaling（OAS）及 Macro Block Scaling（MBS）：OAS 调整归一化最大值并允许少量饱和，以扩大尾部分布的有效动态范围；MBS 把 8 个 1×16 局部块组成 1×128 macro block，并在局部 E8M0 指数之外保存 8-bit mantissa 因子。作者报告 MBS-H 在 Llama 3.1-8B-Instruct 和 Qwen3-8B 的六项任务平均分分别为 66.50/70.84，NVFP4 为 67.02/71.48；报告的 prefill GEMM 平均开销为 6.2%。
- **评估条件与证据边界**：论文在 vLLM 与 LM Evaluation Harness 上，对 Llama 3.1-8B、Qwen3-8B、Llama 4-Maverick 和 DeepSeek-R1 的线性权重与激活做 direct-cast 量化，未使用校准、重训或微调；性能路径基于作者的 CUTLASS 4.3.0、NVIDIA Blackwell SM100 custom kernel。MBS-H 使用动态 MBS 量化权重、静态 MBS 处理激活；作者报告动态实现比静态实现慢约 2.5—3 倍。上述质量和性能均绑定论文版本、模型、任务和 kernel；不能证明标准 OCP MXFP4 的一般支持、目标 runtime 兼容、目标 GPU 收益或目标任务质量，未进行本地复现。
- **证据等级与边界**：source-backed（arXiv v1 全文及附录）。该方法是近期候选，适合作为 E8M0 scale 误差和 macro-scale 元数据的机制案例；它不是 OCP MX v1.0 格式定义，也不是生产质量/性能保证。arXiv 页面当前列示 v1，后续出现新版本时应复核结论范围。

### Four Over Six（4/6）：NVFP4 的自适应 block scale

- [Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling（arXiv v1）](https://arxiv.org/abs/2512.02010v1)，2025-12-01 提交；访问于 2026-11-02。阅读 §3 的 4/6 机制与 scale selection、Table 2 算例、§4.2 PTQ 评估及结论。方法对每个 NVFP4 block 分别按最大值映射到 6 和映射到 4 量化，再按配置误差比较反量化候选。选择 4 会舍弃 ±6 端点，将码值 3 放到 block 最大值的 75% 处；按论文示例 `[10,20,30,40]`，映射到 6 的重构为 `[9.75,19.5,26,39]`、MSE 约 4.33，映射到 4 的重构无误差。论文报告 MSE scale selection 整体较好，并报告其在线 CUDA quantization 操作开销低于 15%；开销数字只对应论文实现和量化操作，不是端到端 inference latency。
- [Transformer Engine 2.20.2 Common API](https://docs.nvidia.com/deeplearning/transformer-engine/api/common.html) 与 [Environment Variables](https://docs.nvidia.com/deeplearning/transformer-engine/envvars.html)，访问于 2026-11-02：公开 `NVFP4BlockScaling` 的 `nvfp4_4over6`、`nvfp4_4over6_err_mode` 与 `nvfp4_4over6_e4m3_use_256` 配置；API 默认关闭 4/6、误差指标为 MAE，并对启用的 4/6 张量默认采用 E4M3 global scale 上界 256。文档称该版本当前支持面向 RL 和 PTQ，带 RHT 的预训练组合尚未实现。
- **证据等级与边界**：source-backed（arXiv v1 正文/表格及 Transformer Engine 2.20.2 官方 API/动态文档）。论文实验和 API 默认值不是同一设置：论文将 MSE 选为主要规则，TE API 默认 MAE；不能把论文质量结论直接归到 TE 默认配置。4/6 是 NVFP4 quantization recipe 的 scale 决策，不是新的 OCP MX 格式，也不证明其他 runtime、checkpoint 或目标模型兼容/有收益；需固定版本、tensor role、误差模式、scale bound、硬件和任务再验证。

### OCP MX E2M1 转换语义与 Model Optimizer 当前支持声明

- [OCP MX v1.0 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)，2023-09，访问于 2026-11-02；阅读 §5.3.3 E2M1 与 §5.4.1 E8M0。该规范定义 E2M1 的最大有限幅值 6、subnormal 支持、无 Inf/NaN 编码，并要求转换支持 roundTiesToEven 和饱和到最大 FP4 幅值；它定义格式语义，不定义某个 runtime 的物理 checkpoint layout 或 kernel。
- [NVIDIA Model Optimizer 当前 Linux 支持矩阵](https://nvidia.github.io/Model-Optimizer/guides/0_support_matrix.html)，访问于 2026-11-02，文档标题标识 `0.0.1.dev1+gd48a5450d`。页面列出 per-block FP4 weights/activations、Blackwell and later、PyTorch model format，以及 TensorRT/TensorRT-LLM deployment。此项是版本化文档声明；不能证明任意模型/shape、KV、算子或目标服务都命中原生 kernel，也不是模型质量或性能测量。
- **证据等级与边界**：source-backed（OCP 正式规范；NVIDIA 动态官方支持矩阵快照）。实际适用仍需按选定的硬件、软件 release、模型 shape 和构建出的 engine 检查。

## 跨来源边界

论文支持算法定义和其报告的实验；官方框架文档支持该版本公开的 dtype、scale recipe 与接口描述；官方 QAT 博客支持 fake quantization 的说明。三类来源都不能单独证明某模型在目标请求分布上的任务质量、目标 GPU 上的部署速度或生产可用性。低位 payload、模型权重误差、模型输出质量和 serving 性能是不同测量对象，需分别核对。
