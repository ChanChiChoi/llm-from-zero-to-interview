# 第六册第七章：量化与模型压缩精修研究记录

日期：2026-10-26  
正文：[第七章：量化与模型压缩](../book-06-llm-deployment/chapters/07-量化与模型压缩.md)

## 研究问题

旧稿已覆盖格式、公式、PTQ/QAT、GPTQ、AWQ、SmoothQuant、KV Cache、蒸馏、剪枝、评估和长上下文案例。主要缺口不是目录少，而是读者难以看出代表方案各自修改了量化链路哪一步；论文中的质量、kernel microbenchmark、推理框架性能和当前 runtime 支持容易被读成同一种“量化收益”。本轮重点核对：

1. 权重、激活和 KV Cache 分别有什么量化对象与生命周期？
2. GPTQ、AWQ、LLM.int8()、ZeroQuant、SmoothQuant 的误差机制哪里不同？
3. KIVI 为什么让 K/V 走不同量化轴，逐 token 生成时如何实现？
4. TurboQuant 的理论失真、作者报告和 serving 运行结果之间是什么关系？
5. MXFP4/NVFP4 的编码和 scale 元数据到底是什么，active cache 与冷 Page 压缩是否等价？
6. 剪枝、蒸馏的原知识点如何保留，同时避免重复后训练专章？

## 阅读过的资料与具体证据

### 分类与方法总览

- [A Survey on Model Compression for Large Language Models，TACL 2024](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00704/125482/A-Survey-on-Model-Compression-for-Large-Language)：将模型压缩分为量化、剪枝、蒸馏等主路线，并把 PTQ 按 weight-only、weight-activation、KV cache 分类。该分类支持章节先按资源对象组织，而不是把所有 4-bit 方法放进同一个排行榜。
- [A Comprehensive Evaluation on Quantization Techniques for LLMs，arXiv:2507.17417](https://arxiv.org/abs/2507.17417)：将方法拆成量化前变换与量化误差补偿，研究旋转/缩放、granularity、对称/非对称和 MXFP4/NVFP4。论文报告更细粒度提升量化表现但增加 scale metadata；对 INT4 有效的 rotation 在 MXFP4/NVFP4 上收益有限；scale 格式和精度会改变 FP4 结果。该 arXiv 页面显示 2026-10-21 修订版；结论绑定作者的模型、设置和评测，不是全部量化方法的普遍因果规律。

### Weight-only 与 activation outlier

| 研究 | 具体约束与机制 | 作者报告的证据 | 局限与可检验的后续点 |
| --- | --- | --- | --- |
| [GPTQ](https://arxiv.org/abs/2210.17323)，§4–5 | 以校准激活近似 Hessian/曲率；列块量化后将误差按二阶耦合回填给未量化权重 | OPT-175B WikiText-2 上，4-bit GPTQ 困惑度增加 0.03，RTN 增加 2.2；175B 量化约 4 GPU 小时。A100/A6000 端到端倍率是各自论文 kernel 配置下的数值 | 离线校准成本与校准分布偏移；压缩比/误差与 kernel 吞吐是分离问题。可测试 GPTQ 加旋转/缩放或 low-rank error compensation，但必须作为新消融，不能说是 GPTQ 已验证的普遍改进。 |
| [AWQ](https://arxiv.org/abs/2306.00978)，v6 | 从激活统计识别 salient channels，通过保持浮点乘积的等价通道变换调 scale；避免把少数权重单独留 FP16 | 论文称约 1% salient weights 对误差影响明显；TinyChat 将 AWQ 与 kernel fusion、platform-aware packing 结合，报告桌面和移动 GPU 上相对 HF FP16 超过 3× | `1% salient` 不是 1% 权重原样留高精度。论文速度属于 TinyChat 路线，不能归因于算法本身；线上 activation 分布和分片 layout 仍需核对。 |
| [LLM.int8()](https://arxiv.org/abs/2208.07339) | vector-wise INT8 计算主体；把少量 outlier feature dimension 拆到 FP16 路径 | 论文称超过 99.9% 值走 INT8，并称最大 175B 模型保持表现 | 显式混合路径保质量，却增加额外乘法和融合需求；不代表端到端一定快。 |
| [ZeroQuant](https://arxiv.org/abs/2206.01861) | activation 按 token 动态 scale，权重按组量化 | 让不同 token 范围可以分开估 scale | 动态 scale 有运行成本；SmoothQuant 论文引用的 OPT-175B baseline 显示旧配置未保持准确率，不能把小模型经验外推。 |
| [SmoothQuant](https://arxiv.org/abs/2211.10438)，§2、§5 | 通过 `Y=XW=(XD^{-1})(DW)` 把激活极值的一部分离线迁移到更容易量化的权重，做 W8A8 | 论文报告特定模型/实现上最高约 1.56× 延迟改善和约 2× 显存收益；OPT-175B 的 alpha 消融中，约 0.4–0.6 更平衡 | alpha、校准、scale 更新和 fused INT8 GEMM 决定结果。论文 table 10 在 batch=4 的 OPT-13B seq=256 一项为 FP16 152.6ms、LLM.int8 237.1ms、SmoothQuant-O3 112.1ms；只说明论文路径中质量保持与延迟收益并非同一指标。 |

这里的技术关系是：LLM.int8 保留 outlier 的高精度旁路；SmoothQuant 改写数值尺度，让所有矩阵乘更容易走 INT8；ZeroQuant 用动态 per-token scale。它们解决相同表象，却给 runtime 留下不同执行结构。正文明确把方法选择写成待同硬件回放的假设，不把论文倍率合并成横向排名。

### KV Cache 量化

- [KIVI](https://arxiv.org/abs/2402.02750)，§3.2、§3.3 与 Algorithm 1：作者观察 K cache 的固定 channel outlier，K-per-channel 限制每个 channel 内的误差扩散；V 没有相同固定 outlier，且在稀疏 attention 加权和中，V-per-token 将误差限定到独立 token。Llama-2-13B 表中的 K attention-score 相对误差统计为 47.00（K/token）和 9.60（K/channel）；V 输出相对误差为 3.55（V/token）和 49.89（V/channel）。这些不是百分比，也不是所有模型的基准。
- K 的 scale 跨 token 分组，不能在每个新 token 到达时立即完成；KIVI 留下未成组高精度 residual，达到 residual 长度后再量化。V 按 token 可逐项量化，同时保留最近一段 full-precision window。该窗口对难数学生成质量重要，不是可以从内存账本中漏掉的细节。
- KIVI 论文报告 2-bit 路线在 Llama/Falcon/Mistral 上将“包含模型权重在内”的峰值内存降为约 1/2.6，最多 4× batch 和 2.35–3.47× throughput。这些是论文 runtime、GPU、模型与 workload 的作者结果，吞吐改善可由更大 batch 解释，不等于 2-bit attention 运算普遍更快。
- [TurboQuant ICLR 2026 / arXiv:2504.19874](https://arxiv.org/abs/2504.19874)：随机旋转后逐坐标 scalar quantization；完整论文加入 QJL residual 校正以使 inner-product estimate 无偏。论文报告约 3.5 bit/channel 在自身 KV task set 上 quality-neutral，2.5 bit/channel 有轻微退化；这不能推出任意复杂推理任务无损。
- [Google Research 技术说明](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)所述“4-bit 相对 32-bit key 最多 8×”指 H100 attention-logit 计算，不是端到端请求 TTFT/TPOT。
- [vLLM TurboQuant 评测](https://vllm.ai/blog/2026-05-11-turboquant)：固定 vLLM 0.20.2 commit `6ec9bbec3`，性能结果主要基于 2×H100 的 Qwen3-30B-A3B 与 4×H100 的 Llama-3.3-70B。质量包括 MRCR 长上下文与 AIME25、GPQA Diamond、MATH500、LiveCodeBench-v6；Qwen3 retrieval AUC 在该评测中 BF16 45.8，aggressive 3-bit preset 31.2。性能中 TQ latency over BF16 约 +10%–68%，离线吞吐约 66%–80%。在 Llama 70B burst、BF16 KV 内存饱和时，BF16 P99 TTFT 约 17 秒、压缩 TQ <3.5 秒、FP8 约 1.3 秒。这组证据显示缓存压缩可缓解排队，但仍以吞吐/TPOT 换容量；它不是论文原始算法的独立复现，也不能外推到其他 preset、runtime 或模型。

2026 年 TensorRT-LLM 文档还区分 active KV dtype 与 cold-page compression：前者把 GPU active cache 的表示改低精度，影响 attention 的读写路径；后者只压缩 offload 到 Host/Disk 的冷 Page，重新进入 GPU 前恢复 runtime dtype，主要减少冷层占用和传输量。当前页面列出 PyTorch backend、KVCM V2 与 SM100/SM103/SM107；NVFP4 cold-page 还要求 native C++ KVCM V2 backend 和 nonzero Host/Disk cache。[Quantization 文档](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html)；[KV Cache Compression 文档](https://nvidia.github.io/TensorRT-LLM/features/kv-cache-compression.html)，页面 2026-10-19 更新，源码 commit `0372799`。这些框架能力不等于本项目已在硬件上运行。

### FP4 格式的精确区别

- [OCP MX v1.0 specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)：MXFP4 为 E2M1 value，每 32 个数共享 8-bit E8M0 scale。忽略 padding 后，scale 导致名义 4 bit 变为约 4.25 bit/value。
- [NVIDIA Transformer Engine NVFP4 docs](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)：E2M1 value，每 16 个数共享 E4M3 scale，tensor 另有 FP32 global scale；忽略全局 scale 后约 4.5 bit/value，再加 `32/N`。
- FP4 不是单一的存储/算术规格；E2M1、scale encoding/granularity、tensor global scale、packing、kernel 指令和模型 recipe 要一起给出。单看 4-bit 文件大小会低估实际 scale metadata 与 padding。

### 蒸馏与剪枝

- 保留教师生成数据、logit/分布、轨迹/结果蒸馏三条路线；on-policy 状态分布与 reverse-KL 机制由[第五册第十七章](../book-05-llm-training/chapters/17-on-policy-distillation.md)主讲，本章给出压缩视角入口，避免复制训练机制全文。
- [SparseGPT](https://arxiv.org/abs/2301.00774)：在 OPT-175B/BLOOM-176B 研究 60% unstructured sparsity、2:4/4:8 semi-structured patterns；作者报告无需重训练和较小 perplexity 增量。显式指出 perplexity 不等于工具/代码任务效果，也指出没有稀疏 kernel 的 dense GEMM 不会因数值为零自动减少算时。

## 原稿知识点覆盖图

| 原稿位置 | 原知识内容 | 本轮处理与正文位置 |
| --- | --- | --- |
| 导言、资料边界 | 数值精度、带宽、kernel、任务质量的总问题；论文/官方资料边界 | 保留；新增综述、论文数值与框架资料各自证明范围。 |
| §1–2 | 显存四本账、精度格式、对称/非对称公式、MSE、粒度/metadata | 保留；FP4 增 scale 计算和 effective bit 例子。 |
| §3–5 | W-only、activation、PTQ/QAT、校准问题 | 保留；增加 scale/命名 contract、校准 manifest 与 activation target 约束。 |
| §6 | GPTQ | 保留并补列块/二阶统计/误差回填过程、OPT-175B 质量与速度证据及限制。 |
| §7 | AWQ | 保留并澄清“保护 salient weights”是通道等价缩放，不是留 1% FP16；拆开算法和 TinyChat kernel speed。 |
| §8 | SmoothQuant | 保留变换公式和 alpha；补 α 研究范围、延迟样例，并对照 LLM.int8/ZeroQuant。 |
| §9–9.1 | KV Cache 账本、K/V、FP8/FP4/native INT4 | 保留；单列 KIVI 生命周期/状态轨迹，重写 FP4 编码/scale 合约，增加 TurboQuant 与 active/cold cache 边界。 |
| §10–11 | 校准数据和量化评估 | 保留；区分 weight vs KV 校准路径，补统一比较中的 transformation/granularity/scale precision 证据。 |
| §12 | logit、hard-label、轨迹蒸馏 | 保留；加学生分布路线差异和第五册 OPD 主讲链接。 |
| §13 | 结构化/非结构化剪枝和 hardware dependency | 保留；补 SparseGPT 的目标、论文报告与 dense kernel 失败模式。 |
| §14、§17 | 选型表与长上下文部署案例 | 保留；决策按 bottleneck 和 runtime kernel；修正 KV budget 只按输入长度计数与“必定不够”的断言。 |
| §15–16、§18–19 | demo、常见误区、来源和总结 | 保留；修复非对称常量分支，补正输入/bit 验证、研究来源与章结论。 |

本章保留旧稿 19 节和量化、蒸馏、剪枝、评估、demo、部署案例的知识路径；新增的 14.1 将已识别的研究局限转成可检验的下一步实验，并明确这些是机制推导的假设，不冒充论文结论。蒸馏深入机制和 FP4/FP8/native INT4 产品化细节已有书系主讲文件，因此本章保留概览与入口，不重复搬运专章，也不删除读者原入口。

## 修改与同步清单

- 正文新增 GPTQ/AWQ/LLM.int8/ZeroQuant/SmoothQuant 的方法区别、KIVI 数值误差和生成状态轨迹、TurboQuant 论文/服务评测差异、active/cold-page cache 分层、MXFP4/NVFP4 的 effective bits、SparseGPT 研究结果。
- demo 修复非对称量化常量张量分支的量化/反量化不一致，加入 finite/bool/bit/group 输入约束和正/负常量精确映射检查。
- 更新 `PAPERS.md`、`GLOSSARY_EN_ZH.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`；第六册目录已有第七章条目，无需改目录。

## 验收范围与遗留风险

- `code_status`：从 Markdown 提取 §15 Python demo 后以 Python 3 执行成功，退出码 0、stderr 为空，七行 stdout 与正文“可能输出”逐字一致。另验证正/负常量、零常量、全正和全负范围的重建误差不超过半个量化步长；空输入、NaN、bool、非法 bit/group 参数六种情况均抛出预期 `ValueError`。这些只验证教学函数的输入合同和数值行为，不等于框架量化或模型质量复现。
- `formula_status`：手核 KV 单 token 账本 `131072 bytes`；16×8704 tokens 为 `17 GiB`；7B FP16≈13.04 GiB；INT4 raw≈3.26 GiB，128-group 16-bit scale≈0.10 GiB，组合预算约20.36 GiB。这里只推导理论账本，不代表 runtime 峰值。
- `link_status`：核对 GPTQ/AWQ/LLM.int8/ZeroQuant/SmoothQuant、KIVI、TurboQuant、SparseGPT 原论文，TACL 综述、Google Research 说明、vLLM 实测、OCP 格式规范和 NVIDIA 文档；章节与研究记录中的本地相对链接待最终脚本检查。
- `validation`：第1—19节一级标题连续；46 个代码/数学 fenced marker 成对，21 个 math block opener 均闭合；6 张表格列数一致；章节和研究笔记的本地 Markdown 链接目标均存在；demo 输出逐字一致且 11 个边界检查通过；`git diff --check` 通过。未做外部公式渲染或 PDF 构建。
- `open_risks`：未在 GPU、真实模型或目标服务环境复现 GPTQ/AWQ/SmoothQuant/KIVI/TurboQuant/SparseGPT；论文数据不代表生产 quality 或 throughput。vLLM TurboQuant 结果来自其实现和 workload，不等价于论文算法复现。TensorRT/vLLM/HF 文档和量化支持矩阵会变化，部署前需固定 release、model revision、backend、kernel 与 GPU。未做外部公式渲染或 PDF 构建。
