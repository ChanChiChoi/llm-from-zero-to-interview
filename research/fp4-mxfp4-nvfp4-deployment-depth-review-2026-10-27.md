# 第六册第十五章：FP4、MXFP4 与 NVFP4 深度精修记录

日期：2026-10-27  
正文：[第十五章：FP4、MXFP4 与 NVFP4](../book-06-llm-deployment/chapters/15-fp4-mxfp4-nvfp4.md)

## 本轮要解决的问题

原稿已包含量化对象、outlier、kernel、PTQ/QAT、评估切片、artifact、fallback 与部署矩阵，主题主线完整，不适合整章推倒重写。主要缺口是读者仍无法从“4 bit”复算实际表示与存储；MXFP4/NVFP4 的 scale 精度、粒度和二维权重 recipe 没有讲实；代表研究也没有讲清楚它们解决哪一步误差、为何不一定能迁移到另一个格式。本轮沿用原 §1—§14 结构，在相应小节补数值、研究机制和证据范围。

## 一手规范、官方文档与可支持事实

### E2M1 与 MXFP4

- [OCP Microscaling Formats (MX) v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) 定义 MXFP4 为 E2M1 元素、block size 32、8-bit E8M0 scale。OCP v1.0 的向量 block 总量为 `32×4+8=136 bit`，scale 摊销后是 `4.25 bit/value`，未计 padding、对齐和实现可选的重复 scale 压缩。
- OCP FP4 E2M1 使用 1 个 sign、2 个 exponent、1 个 mantissa bit；可表示的非负幅值为 `0, 0.5, 1, 1.5, 2, 3, 4, 6`，没有 Inf/NaN 编码。E8M0 不含 sign 与 mantissa；有限 scale 是 2 的整数次幂。OCP 规定逻辑 block 参数，但不指定 scale 与 payload 必须如何物理排布。
- MXFP4 的受控比较应使用 `4 + 8/32`，但这不是每个文件都恰好 `4.25 bit/value` 的保证：padding、行布局、重复尺度压缩和模型中保留高精度的层都会改变物理大小。

### NVFP4 profile 与接口层次

- [NVIDIA Transformer Engine NVFP4 文档](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)，访问时页面为 2.20.2，定义 E2M1 × 16-element E4M3 block scale × per-tensor FP32 global scale；同页另说明当前 training recipe 默认权重用 16×16 二维 scaling、激活和梯度用一维 16-element scaling。后者不是所有 NVFP4 checkpoint 的共同约定。
- 若只算一维 profile，`N` 个值的名义平均位数为 `4 + 8/16 + 32/N = 4.5 + 32/N`。该值不包括 padding、对齐、rowwise/columnwise layout、副本或未量化模块。不能把它直接套到采用二维权重 scale 的 artifact。
- [TensorRT 10.x quantized types](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html) 的显式 Q/DQ 量化说明把 NVFP4 block scale 参数描述为 FP16/FP32/BF16。该接口层级不同于 Transformer Engine 对紧凑 NVFP4 recipe 的 E4M3+global FP32 描述。文档不足以单独确定某个导出器/engine 如何转换，必须检查实际 artifact 和 engine。
- [NVIDIA Model Optimizer support matrix](https://nvidia.github.io/Model-Optimizer/guides/0_support_matrix.html) 于 2026-10-27 访问时，Linux FP4 行列出 per-block weights/activations、Blackwell and later、PyTorch 输入格式及 TensorRT/TensorRT-LLM 部署。页面为 rolling release documentation，只证明该快照公开声明支持路径；不证明目标 model shape、KV、kernel 或性能。

## 代表研究各自解决了什么

| 工作 | 原问题 | 方法如何作用 | 证据与限制 | 可以沿什么方向继续改进 |
| --- | --- | --- | --- | --- |
| [A Comprehensive Evaluation on Quantization Techniques for LLMs](https://arxiv.org/abs/2507.17417) | 不同量化论文把变换、量化器和误差补偿混在一起；比较难归因 | 把 pre-quantization transformation 与 quantization error mitigation 分开，并比较量化粒度、对称性及 MXFP4/NVFP4 | 报告更细粒度往往提升效果但增加 scale 开销；scale 格式与精度显著影响 FP4，INT4 中有效的 rotations 在该设置中对 MXFP4/NVFP4 收益有限。绑定作者评测配置 | 对目标格式单独消融 block、scale dtype、rounding、transform 与误差补偿，不把不同论文配置拼成统一排行榜 |
| [Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization / MR-GPTQ](https://arxiv.org/abs/2509.23202) | 通用 PTQ/RTN 和 outlier rotation 未必适合 micro-scale FP4；rotation 可能改变小 block 的分布并放大 scale 误差 | 将 block-level Hadamard rotation、GPTQ 二阶误差回填和 format-specific scale search 合成 Micro-Rotated-GPTQ；以 QuTLASS 融合 kernel 处理 rotation overhead | 论文摘要把 Llama 3/Qwen 3 zero-shot 主实验概括为 MXFP4 recovery 距 NVFP4 “within 1–2%”；附录 I 的另一组 PlatinumBench 变体比较中，最佳 NVFP4/MXFP4 recovery 为 96.6%/92.3%，平均相差超过 4 个百分点。摘要结论不适用于每个评测切片。kernel 最高 B200 3.6× layer/2.2× end-to-end、RTX 5090 6×/4×，均绑定作者实现、shape、baseline 和测试设置 | 目标环境先确定质量瓶颈来自 rotation、scale search、RTN 还是 kernel；再以逐项消融验证，不能把“rotation 有效”泛化到 MXFP4/NVFP4 |
| [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149) | 从头或长周期训练时，4-bit scale/outlier/梯度舍入会影响训练稳定性，单独格式不足以解释收敛 | 12B hybrid Mamba-Transformer、10T tokens 训练 recipe 结合 16×16 weight scaling、部分 Wgrad Hadamard transform、gradient stochastic rounding 和 selective BF16 layers | 作者报告 stable phase validation loss 相对误差低于 1%、decay 后略高于 1.5%，多数下游切片接近 FP8、MBPP+ 有差异；是发布方训练报告，不是任意模型的 PTQ/inference 证据 | 将 training recipe 与 inference PTQ 分开评估；为敏感层、梯度、目标精度切换和任务回归设计自己的消融 |
| [Scale Learning (Model Optimizer, 2026-09-16)](https://nvidia.github.io/Model-Optimizer/announcements/20260916-scale-learning.html) | PTQ 固定尺度可能与任务损失不匹配；仅搜索初始化 scale 仍可能留有量化误差 | fake quantization + STE 将 block scale 变成可训练参数；scale-only 冻结权重，full-parameter 同时训练权重与 scale，导出时 scale 合入 artifact | 在 Nemotron 3.5 Lightning 指定 W4A4 routed experts / W8A8 attention+Mamba cache 配置中，GPQA 73.30→75.00、SciCode 33.42→35.42、Tau3 Banking 7.216→7.216（BF16 为 9.897）。这是团队单模型报告；当前实现限定 NVFP4 per-block weight scales，仍需训练/QAD 算力 | 在目标任务比较 MSE/scale search、scale-only QAD、full parameter QAD，并报告每个关键 slice 和额外训练成本；不要把某两个分数提升写成通用质量收益 |

两篇 PTQ 评测看起来对 rotation 的普遍收益不同，关键是它们研究的算法条件不等价：标准 rotation + RTN 与 block-aware rotation + GPTQ/scale search 不是同一方法；模型、zero-shot task 与 kernel 也不同。MR-GPTQ 论文内部的汇总也依赖评测切片：摘要所说的主实验概括与附录 I PlatinumBench 的最佳 recovery 差距不是同一统计结果。正文把这些数字分开呈现，避免从摘要句子推成所有 benchmark 的稳定差距。

## 原稿内容承接与位置

| 原稿内容 | 本轮处理 |
| --- | --- |
| §1 FP16/BF16、INT4、payload 与 scale metadata | 保留动机与通用存储账本；补充均匀 INT4 与非均匀 E2M1 格点的差别 |
| §2 FP4 / MXFP4 / NVFP4 命名和硬件边界 | 保留问题清单与支持矩阵读法；补 E2M1 codebook、MXFP4 E8M0/32、NVFP4 E4M3/16/global FP32、二维权重 scaling 与 Q/DQ 文档层次 |
| §3 block scale 与 outlier | 保留异常值、误差与缩小 block/高精度回退/calibration；加入 E2M1 `0.9/20→0` 的具体推演和 MR-GPTQ 的机制/证据/边界 |
| §4 权重、activation、KV、累加、logits 分开量化 | 保留，避免从 FP4 weights 外推到 KV 或整条服务 |
| §5 kernel / engine / task 分层 | 保留，补充格式字节收益和 endpoint 速度证据不在同一层 |
| §6 PTQ/QAT/校准集 | 保留 calibration 与 QAT 的因果解释；加 NVFP4 training recipe 与 scale learning 方案对比 |
| §7—8 任务切片、推测解码、异常值、内存算术 | 保留评测范围；新增精确 MXFP4/NVFP4 一维 bits 与 4096² 权重账本 |
| §9 Python demo | 将原非对称 toy codebook 更新为 E2M1 可表示值；scale 仍为 Python 浮点数，明确不是 E8M0/E4M3/NVFP4 完整实现；补空输入、非有限数、尺寸输入检查 |
| §10—12 manifest、fallback、测试矩阵、常见误区 | 保留，无删除 |
| §13—14 资料边界与总结 | 删除过时的“OCP PDF 站点验证阻止核验”说明，改为已读取 OCP 正式规范；加官方版本日期、研究边界和关键格式结论 |

## 公式与代码核验

- MXFP4：`(32×4+8)/32 = 4.25 bit/value`。
- NVFP4 一维：`4 + 8/16 + 32/N bit/value`；FP32 per-tensor scale 摊销为 `32/N`。
- `N=4096×4096=16,777,216`：FP16 为 32 MiB，E2M1 payload 为 8 MiB；MXFP4 scale 为 0.5 MiB、合计 8.5 MiB、理论压缩比 `32/8.5≈3.76×`；NVFP4 1D scales 为 `N/16+4=1,048,580 bytes`，约 9 MiB 总量，理论压缩约 3.56×。未将这些数写成实际 checkpoint 大小。
- E2M1 outlier 推演：`0.9/20=0.045` 最近码值为 0，重构绝对误差 0.9；若局部 scale=0.9，则 `0.9/0.9=1` 且可还原到 0.9。示例用于说明 scale 粒度，不模拟 profile scale 编码。
- §9 demo 使用标准库 `math` 与 `pprint`。实际 stdout 与文中 `~~~text` 样例逐字对照；异常输入检查覆盖非正 block size、空值、非有限值、MSE 长度不匹配和负元素数。该验证不依赖 CUDA，不运行外部框架。

## 配套资料同步

- `EXERCISES.md` 新增第六册第十五章练习，覆盖 E2M1 outlier 手算、不同格式 scale/storage 复算、PTQ/MR-GPTQ/scale-learning 对照与目标 workload 验收。
- `INTERVIEW_BANK.md` 新增五道 FP4/MXFP4/NVFP4 问答，包含有效 bit、E8M0/E4M3、MR-GPTQ、scale learning 与训练结果不可外推到 PTQ。
- `PAPERS.md` 补 MR-GPTQ、NVFP4 预训练、NVFP4 scale learning 和 TensorRT Q/DQ 来源；保留既有 OCP、Transformer Engine 与统一量化评测条目。
- `GLOSSARY_EN_ZH.md` 修订 FP4/MXFP4/NVFP4 条目并加入 E2M1、E8M0、MR-GPTQ 和 NVFP4 Scale Learning。

## 验收与遗留范围

- `structure_status`：§1—§14 主体完整；保留原有量化对象、校准、评估、deployment manifest、fallback 和常见误区，没有因扩展格式讨论而删除已有主题。
- `formula_status`：手核 codebook、尺度位数、有效 bit 数、outlier 重构和 4096² 存储示例；一维 NVFP4 公式与二维 weight scaling 明确分开。
- `code_status`：§9 纯 Python demo 运行并与正文输出逐字对照；边界输入按文中说明拒绝。未运行 GPU、Transformer Engine、Model Optimizer、TensorRT 或真实 checkpoint。
- `source_status`：浏览并读取 OCP MX v1.0 PDF、Transformer Engine 2.20.2 NVFP4 文档、TensorRT 10.x quantized types、Model Optimizer 支持矩阵、两篇 MXFP4/NVFP4 量化研究、NVFP4 预训练报告和 Model Optimizer scale-learning 技术说明。rolling docs 已加快照/版本限制。
- `link_status`：正文使用官方规格、官方文档与论文直接链接；配套索引和本地相对链接应在最终仓库扫描中复核。
- `validation`：执行 demo 与边界检查；检查章节标题、fence、表格和本地链接；运行 `git diff --check`。
- `open_risks`：没有目标模型/GPU、quantization toolkit 或 inference engine 实测；MR-GPTQ、训练结果和 Model Optimizer scale-learning 数字均绑定作者/发布方配置。TensorRT Q/DQ scale dtype 与 Transformer Engine 紧凑 profile 属接口层次差异，尚未验证特定导出 artifact 的转换过程。未渲染外部公式或 PDF。
