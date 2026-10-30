# Weight Quantization Formats and PTQ Evidence (2026-10)

检索日期：2026-10-29。本文记录低位宽权重表示、FP8/FP4 格式和代表性 LLM 权重量化方法的来源事实及证据边界。本文不是书籍审阅或修改日志。

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

## 新近候选资料的证据状态

- [AAAC arXiv v2](https://arxiv.org/abs/2605.08692v2)，2026-06-05 版本，访问于 2026-10-29；只阅读 arXiv 摘要。作者摘要称该方法为 4-bit weight-only PTQ，使用两个 learned scalar codebooks，并通过组级选择编码降低额外存储；摘要也报告其校准时间和与若干方法的比较。方法细节、复现实验、目标 kernel 和跨模型外推均未在本次阅读中核验，证据限于作者摘要。
- [D²Quant arXiv v2](https://arxiv.org/abs/2602.02546v2)，2026-02-06 版本，访问于 2026-10-29；只阅读 arXiv 摘要。作者摘要称其在低于 4-bit 的 weight-only PTQ 中结合面向 down-projection 的双 scale 设计与 LayerNorm 均值偏移校正；指标提升和机制有效性尚未通过正文实验表、代码或独立复现核验。
- [SliderQuant arXiv v1](https://arxiv.org/abs/2603.25284v1)，2026-03-26 版本，访问于 2026-10-29；只阅读 arXiv 摘要。摘要称该方法按层敏感性使用不同 sliding quantization 策略，并标注 work accepted to ICLR 2026。论文正文中的算法、完整评测口径、开源实现与 runtime 覆盖未在本次阅读中核验。

## 跨来源边界

论文支持算法定义和其报告的实验；官方框架文档支持该版本公开的 dtype、scale recipe 与接口描述；官方 QAT 博客支持 fake quantization 的说明。三类来源都不能单独证明某模型在目标请求分布上的任务质量、目标 GPU 上的部署速度或生产可用性。低位 payload、模型权重误差、模型输出质量和 serving 性能是不同测量对象，需分别核对。
