# 第五册第八章：显存优化与数值精度深度精修研究记录

日期：2026-10-26  
正文：[`book-05-llm-training/chapters/08-显存优化与数值精度.md`](../book-05-llm-training/chapters/08-显存优化与数值精度.md)  
依据：当前 [`WRITING_SPEC.md`](../WRITING_SPEC.md)；按 `progress_v2.md` 接第五册第八章，不回头重做已完成章节。

## 研究目标与覆盖原则

本章原有的参数/梯度/optimizer/activation/allocator 账本、7B 示例、峰值显存、长上下文、attention/FlashAttention、FP32/FP16/BF16/FP8、AMP/loss scaling、activation checkpoint、gradient accumulation、ZeRO/FSDP/offload、低比特 optimizer、LoRA/QLoRA、OOM、NaN/Inf、资源 toy 和七项练习全部保留。本轮把宏观列举改成可推演的生命周期、公式、算例、代表研究结果和证据边界；没有把显存减少误写成质量/吞吐保证。

## `coverage_map`

| 原章节主题 | 本轮级别 | 精修后的去向与解释变化 |
| --- | --- | --- |
| 训练前的显存问题和五类账本（8.0—8.1） | 核心保留并纠正口径 | 区分逻辑状态、仍活着的 tensor 分配与 allocator reserved；峰值改为对同时存活分配在时间上取最大，避免把不同事件峰值机械相加。 |
| 7B BF16/AdamW 状态账本（8.2） | 核心补深 | 保留 16 bytes/parameter 假设；补无 FP32 master copy 时 12 bytes/parameter 的对照，并明确实际参数/梯度/optimizer dtype 随 framework/recipe 变化。 |
| 峰值、transient gather、评估/checkpoint（8.3） | 核心补生命周期和来源 | 区分 allocated/reserved/设备外部分配；补 `empty_cache()` 边界、memory snapshot 可观察范围以及 PyTorch 2.14 Adam 第一次 step 懒初始化 m/v 的版本化源码证据。 |
| activation 的 B/T/d/layer 关系（8.4） | 核心补手算 | 对 1×4096×4096 BF16 的 hidden tensor 手算 32 MiB，32 层单份约 1 GiB；增加 4K/32K hidden 与 score tensor 对照。 |
| 标准 attention score 与 FlashAttention（8.5） | 核心机制展开 | 说明 online softmax 保存 `(m,l,o)` 并逐 tile 更新，手算 score `[1,2,0]` 的输出；明确节省的是中间量物化/HBM IO，精确 full-attention 的 query-key 乘加仍为平方量级。 |
| FP32/FP16/BF16/FP8 数值边界（8.6） | 核心定量对比 | 增加位布局、最大值、最小 normal、1 附近 spacing 表；对照 E4M3/E5M2，说明 delayed scaling、amax history、MXFP8 block scale 与 NVFP4 的版本/硬件限制，并与 QLoRA NF4 存储格式分开。 |
| AMP、autocast、loss scaling、clipping（8.7） | 核心补真实 API 轨迹 | 区分 parameter dtype 与 autocast 算子 dtype；修正 `optimizer.step()` 直接绕过 GradScaler 检查的问题，改为 unscale→clip→`scaler.step`→`scaler.update`；明确 scale 跨整个 accumulation window 保持恒定且可低于 1。 |
| activation checkpoint/recomputation（8.8） | 核心研究对照 | 将误导性的 `$M/K$` 直觉改为分段边界+单段重建模型 `(K+L/K)A`；比较 Chen 子线性 memory、Korthikanti selective recompute/sequence parallel 结果，并补 PyTorch 2.14 non-reentrant/RNG/determinism 边界。 |
| gradient accumulation、DDP `no_sync` 和 reduction（8.9） | 核心补推导 | 增变长 target 的 2-token/6-token 加权 loss 算例，以及 DDP rank averaging 下的 `R/N` 缩放；补 scaler effective-batch 行为、`no_sync` forward+backward 范围和 partial window 分母。 |
| sharding/offload/8-bit optimizer state（8.10） | 核心方案对照 | 7B 两个 Adam moment 从 8 bytes/parameter 到理想 2 bytes/parameter 的节省账本；展开 block-wise/dynamic-tree update 流程、outlier 隔离、稳定 embedding 和作者消融结果。 |
| LoRA 与 QLoRA（8.11） | 核心机制与案例加深 | 手算 4096×11008、rank-16 的 adapter 参数/显存；区分 LoRA 的 trainable state、NF4 base storage、Double Quant scale metadata、paged optimizer 与 backward activation。加入作者 7B 内存拆分和 65B/48GB 结果的限定条件。 |
| 分阶段 OOM、NaN/Inf、overflow/underflow、clipping（8.12—8.13） | 核心保留并诊断化 | 增加现象→证据→机制的矩阵；区分首个 optimizer step 的懒初始化与后续持久增长；说明 PyTorch snapshot 覆盖边界及 FP16 subnormal 不能仅按 min-normal 判为 zero。 |
| 策略比较表（8.14） | 核心重组 | 用“瓶颈/首要证据/手段/剩余代价”帮助读者从 OOM 痕迹选择方法，保留原有 BF16/FP16/FP8/checkpoint/accumulation/sharding/offload/FlashAttention/LoRA/QLoRA/8-bit 方法。 |
| Python 显存与数值审计账本（8.15） | 核心保留并修正 | 继续是标准库 toy；激活字节与权重字节拆开；checkpoint proxy 使用 `(K+L/K)/L`；加入显式 transient gather 假设和输出解释。未把账本当作 CUDA/FSDP/FlashAttention 模拟器。 |
| 故障实验流程、资料边界、七项练习和小结（8.16—8.19） | 核心保留并拓展 | 原流程/七题保留；练习扩为十二项，增加 online softmax、变长 accumulation、NF4/DQ、首步 OOM 和 FP8 scale policy。 |

本轮没有拆分文件；第五册目录已映射第八章。练习仍位于章节正文，不重复复制到 `EXERCISES.md`。

## 研究来源与证据边界

### 论文

1. [Mixed Precision Training](https://arxiv.org/abs/1710.03740)，重点读 §3 的 FP32 master weights、loss scaling、FP32 accumulation 和 §4 任务实验。论文说明它的 FP16 配方如何避免窄动态范围导致的精度损失；正文用作历史机制和计算路径来源，不把论文旧 GPU 性能外推到当前硬件。
2. [A Study of BFLOAT16 for Deep Learning Training](https://arxiv.org/abs/1905.12322)，核对 BF16 与 FP32 指数范围相同、fraction 更短，以及跨图像/语音/语言/推荐任务的作者结果。它支持格式动机与论文实验，不说明任意现代训练 recipe 都无需稳定性评测。
3. [FlashAttention](https://arxiv.org/abs/2205.14135)，重点读 §3 tiling/online statistics/backward recomputation 和 IO complexity。对每行在线维护最大值 `m`、指数归一化和 `l`、value 加权和 `o`，每个新 tile 都把旧统计换到新最大值基准后合并。该机制避免完整 score/probability 矩阵写回 HBM，但不减少 exact full attention 的所有 query-key 对乘加。
4. [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)，核对链式网络分段、保存边界、反传时局部重算、`O(sqrt(L))` feature-map memory 与额外 forward 计算。论文还报告 1,000 层 residual 网络从 48G 降至 7G、运行时间约增加 30%，该旧式图像任务仅用于说明论文历史实验，不作为 Transformer 预测。
5. [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)，重点读 §4—6、Table 2/4/5。作者分析 TP/SP 与 selective recompute；GPT-3 与 MT-NLG 论文配置中 selective recompute 分别报告 activation memory 约降 70%/65%、FLOPs overhead 约 2.7%/1.6%。22B 单层 forward+backward 测量表中 baseline 19.6ms、full recompute 27.2ms（+39%）、selective 20.9ms（+7%）、selective+SP 20.3ms（+4%）；SP+selective 在论文模型配置中将 activation memory 相对 TP baseline 降至约 20% 以下。所有数据绑定作者实现、TP=8/模型并行、论文设备和 token length，不可解释成一个普适 checkpoint 比例。
6. [LoRA](https://arxiv.org/abs/2106.09685)，核对冻结底座与低秩增量、参数公式和原文 GPT-3 175B memory/可训练参数报告。LoRA 省 trainable parameters/state，但不消除经过冻结层的梯度和激活。
7. [QLoRA](https://arxiv.org/abs/2305.14314)，重点读 §3 的 NF4、Double Quantization、paged optimizers 与 Eq. 5/6 计算路径，以及 §4 实验。一级 NF4 block size 64；FP32 scale metadata 平均 0.5 bits/parameter。作者把 scale 用 8-bit 量化、第二级常量 block size 256 后报告约 0.127 bits/parameter，节省约 0.373 bits/parameter，65B 模型约 3GB。作者报告 65B 模型可以在单张 48GB GPU 上 QLoRA 微调；7B LLaMA/FLAN v2/batch 1 案例约 26MB adapter、567MB LoRA input-gradient、5,048MB 4-bit base，activation checkpointing 后 input-gradient 平均 18MB/sequence。该论文自己的架构、语料、评估和实现不能自动迁移成目标用户训练合同。
8. [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861)，重点读 §2.1—2.3 和 §4 消融。论文以 2,048 元素分块、block absmax 归一化、dynamic-tree quantization 保存 optimizer stats；update 时 register 内反量化至 FP32，更新后再量化；stable embedding 与更高精度 embedding state 是该 NLP recipe 的一部分。209M 参数消融中完整 dynamic+blockwise+stable embedding 8-bit Adam 报告 0% unstable、PPL 16.4；有 stable embedding 的 32-bit Adam 报告 0%、PPL 16.3。分量消融说明“8-bit”标签单独不足以归因结果。

### 官方框架/实现资料

1. [PyTorch AMP 2.14](https://docs.pytorch.org/docs/2.14/amp.html) 与 [AMP examples 2.14](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html)：核对 op-specific autocast、FP16 gradient scaling、`unscale_` 只能在 effective batch 梯度累计后调用、scale 不在 batch 中途变化、clipping 顺序及 GradScaler scale 不保证大于 1。
2. [PyTorch checkpoint 2.14](https://docs.pytorch.org/docs/2.14/checkpoint.html)：文档推荐显式设置 `use_reentrant=False`；核对 RNG state、dropout 可重复性、early-stop recompute 与 checkpoint 函数前/重算时状态不一致可能导致梯度错误。
3. [PyTorch CUDA memory notes 2.14](https://docs.pytorch.org/docs/2.14/notes/cuda.html) 与 [Understanding CUDA Memory Usage 2.14](https://docs.pytorch.org/docs/2.14/torch_cuda_memory)：`allocated` 是 tensor 当前占用，`reserved` 是 caching allocator 管理的 segment；`empty_cache()` 释放未占用 cache 给其它应用，不会释放 live tensor；memory snapshot 无法显示不经过 PyTorch allocator 的直接 CUDA allocation。
4. [PyTorch Adam `v2.14.0` source](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/optim/adam.py)：源码 `_init_group` 在首次出现 grad 时惰性创建 `exp_avg`/`exp_avg_sq`，支持把“首步 OOM”作为一个独立检查点。结论只针对该固定 revision 和 Adam 实现。
5. [Transformer Engine FP8 Primer 2.20.2](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html)：核对 E4M3/E5M2、delayed scaling 的 amax history、MXFP8 每 32 个连续元素一组的 scaling，以及 NVFP4 每 16 个元素的 scale 组织；MXFP8/NVFP4 的训练路径依赖目标 GPU 与内核，不代表其它厂商的 FP8 recipe。

## 手工公式与数值校验

- 7B 参数：16 bytes/parameter 为 112 GB = 104.31 GiB；无 master copy 时 12 bytes/parameter 为 84 GB = 78.23 GiB。84 GB 是十进制字节，不能写成 84 GiB。
- BF16 hidden tensor：`1×4096×4096×2 bytes=32 MiB`；32 层一份 1 GiB；T=32768 时分别为 256 MiB/层与 8 GiB/32 层。
- FP16 attention score：`B=1,H=32,T=4096,b=2` 得 1 GiB/单层单份；T=32768 得 64 GiB/单层单份。causal kernel 的实际物化形状可能不同。
- online softmax：score `[1,2]`、value `[10,20]` 得 `m=2,l≈1.3679,o≈23.6788`；加入 score `0`/value `30` 得 `l≈1.5032,o≈27.7389,output≈18.45`，与一次性 softmax 一致。
- FP16/BF16：1 附近 spacing 分别为 `2^-10≈9.77e-4`、`2^-7≈7.81e-3`；FP16 最小 normal `2^-14≈6.10e-5`，BF16 `2^-126≈1.18e-38`；`2e-8×4096=8.192e-5`。低于 min-normal 不等同于标准 subnormal 必然成为 0。
- checkpoint segment toy：32 层、K=4/8/16 分别约 `(K+32/K)A=12A/12A/18A`，不是通用显存折扣。实际需把每层不同 activation shapes、边界和生命周期纳入 profile。
- accumulation：loss mean 0.60/2 targets 与 0.90/6 targets；简单 micro-batch mean 为 0.75，token mean 为 `(2×0.60+6×0.90)/8=0.825`。DDP 默认 rank average 时局部 loss sum 乘 `world_size/global_valid_targets`。
- 7B 8-bit Adam m/v 理论 payload：FP32 state `56 GB=52.15 GiB`；8-bit state `14 GB=13.04 GiB`；理想 payload 省 42 GB=39.11 GiB，不含 scales/其它显存。
- rank-16 LoRA：`16×(4096+11008)=241,664` 个参数；相对 `4096×11008=45,088,768` 为 0.536%；按 16 bytes/parameter 的 adapter 状态约 3.69 MiB。
- NF4 7B payload：4-bit 为 3.5 GB=3.26 GiB；Double Quant 0.127 bits/parameter 的二级 scale metadata 约 0.103 GiB；没有据此推断完整 CUDA peak。
- Python toy 当前正常输出：状态 104.31 GiB、activation proxy 6.00 GiB、单层 score proxy 1.00 GiB、合成 full peak proxy 115.31 GiB；8-way state share 13.04 GiB、transient gather 假设 8.00 GiB、4-segment checkpoint proxy 2.25 GiB、合成 sharded peak 27.54 GiB；LoRA 参数 241,664、state 0.003601 GiB。上述是题面参数下的教学账本，不是 GPU allocation/profile。

## 验证与配套同步

- `code_status`：标准库账本从 Markdown 中逐段提取并直接执行，输出与正文列明值一致，断言通过。尚未验证 CUDA、真实 FlashAttention/FP8 kernel、FSDP/ZeRO、目标模型或 GPU peak；账本仅为 `local-demo`。
- `formula_status`：复核 7B byte/GiB 转换、score/hidden scaling、online softmax 递推、FP16/BF16 量级、segment toy、token-weighted accumulation、8-bit Adam payload、LoRA 低秩参数与 NF4 元数据公式。没有外部 MathJax/PDF 渲染器验收。
- `link_status`：章节新增论文、PyTorch 2.14 固定文档、Adam 固定 revision 与 Transformer Engine 2.20.2 官方页；动态/latest 源只作当前页面入口，实测部署前需依目标版本复核。
- `sync_status`：`PAPERS.md` 增训练内存与精度来源；`GLOSSARY_EN_ZH.md` 补 NF4、Double Quantization、amax/delayed scaling、MXFP8/NVFP4、低比特 optimizer state 等；`INTERVIEW_BANK.md` 增 memory ledger、FlashAttention recurrence、GradScaler/accumulation、checkpoint、QLoRA/8-bit optimizer、FP8 scale 与 OOM 诊断追问。章节目录原有链接正确，练习保留在本章。
- `validation`：最终核对章节主标题 8.0—8.19 连续、49 个 fenced blocks 成对、1 个 Python 块、33 个 math blocks 花括号计数平衡、12 项练习及全部相对链接目标；标准库账本输出与正文列出的 GiB/参数数值一致，另对非有限数、布尔值、缺失配置、负字节数、零 micro-batch、负 optimizer state、超出层数的 checkpoint segments 和超矩阵维度的 LoRA rank 做了 9 项边界拒绝检查；`git diff --check` 后续执行并记录于 `progress_v2.md`。公式未通过外部渲染器。

## 未解决项

- 无目标 GPU、模型 checkpoint、数据或训练集群的本机实验；`M_peak` 的设备特定结果仍待 profiling。
- FP8/MXFP8/NVFP4 和 8-bit optimizer 的效果依硬件、kernel、layout、dtype、scale recipe、数据/模型和质量评测；章节只解释公开方案，不声称目标模型可直接使用。
- QLoRA/8-bit/activation recomputation 论文数字是作者设置，未在统一 model/tokenizer/data/hardware/compute budget 下横向重跑。
- 代码 toy 的“fit”只表示输入假设算术低于 80 GiB；它没有 allocator fragmentation、peak event、设备 context、真实 all-gather 和 checkpoint storage 生命周期。
- 外部数学渲染、PDF 构建与跨 PyTorch/Transformer Engine 版本兼容性没有本地验收。
