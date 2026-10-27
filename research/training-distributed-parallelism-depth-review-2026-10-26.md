# 第五册第七章：分布式训练深度精修研究记录

日期：2026-10-26  
正文：[`book-05-llm-training/chapters/07-分布式训练.md`](../book-05-llm-training/chapters/07-分布式训练.md)  
依据：当前 [`WRITING_SPEC.md`](../WRITING_SPEC.md)；按 `progress_v2.md` 接第五册第七章，不回头重做前序章节。

## 研究目标与覆盖原则

本轮目标是把“并行方式名称和优缺点”落实为数据/张量怎样切、collective 在哪一刻发生、显存为何降低、代价转移到哪里，以及怎样用可观测指标判断。原章节已覆盖大量主题，因此没有整体推倒或删除。对分布式训练中绕不开的主题补机制和数值；低优先级细节仍放在原小节、来源表或练习中保留入口。

## `coverage_map`

| 原章节主题 | 本轮级别 | 精修后的去向与解释变化 |
| --- | --- | --- |
| world/rank/local rank、process group、更新时间线 | 核心保留 | 仍在 7.1；作为并行拓扑和 collective 顺序的基础，不另拆专题。 |
| global batch、有效 token、rank loss reduction、gradient accumulation | 核心加深 | 7.2 区分 consumed input tokens 与经过 shift/mask 后的 prediction targets；增加 3-rank 数值反例和 PyTorch DDP 默认 rank averaging 下的 `P/N` 缩放条件。 |
| DDP 数据副本、梯度同步、sampler、`no_sync` | 核心保留并补边界 | 7.3 保留 DDP 主线；增加不等长 rank 的 `join()` 行为、initial/effective world-size 除数差异及额外 collective 的限制。 |
| all-reduce、reduce-scatter、all-gather、all-to-all、broadcast/send/recv、ring 和 alpha-beta 模型 | 核心保留 | 7.4 强调每种操作的输入/输出布局；解释 ring 量是每 rank 发送/接收量，不是系统总流量，并限定算法/链路假设。 |
| DDP 与 ZeRO-1/2/3 状态内存 | 核心保留并定量 | 7.5 保留各阶段公式；新增 7B、BF16、AdamW、DP=8 的 GiB 账本，明确省略 master weights、activation、workspace 和参数聚齐峰值。 |
| FSDP 生命周期、wrap 粒度、CPU/NVMe offload | 核心保留并按版本拆清 | 7.5 区分 FSDP1 wrapper/FlatParameter 与 PyTorch 2.14 FSDP2 `fully_shard`/DTensor、自底向上通信组；offload 继续与状态 sharding 分开。 |
| Tensor Parallel 行/列切分、attention、embedding/head/loss | 核心保留 | 7.6 补齐不整除、局部 shape 和输出 collective 前提；未删原来的矩阵布局问题。 |
| Pipeline stage、bubble、micro-batch、1F1B、stage imbalance | 核心加深 | 7.7 对照 GPipe flush、non-interleaved 1F1B、interleaved 1F1B；解释 activation 驻留、bubble 与每样本通信量的共同变化，并限定 Megatron 论文的收益条件。 |
| Sequence Parallel、Context Parallel、长上下文 attention | 核心加深 | 7.8 明确 SP 是 TP 配套的逐 token activation 分片；比较 Megatron CP ring P2P 路径与 Ring Attention blockwise online attention，不把概念名当作同一协议。 |
| DP/TP/PP 组合、rank mapping、节点内外拓扑 | 核心保留 | 7.9 保留正交维度和 64-GPU 对照；强调 group membership 与网络位置比 world size 更能解释性能。 |
| MoE EP、capacity、all-to-all、负载均衡/overflow | 核心加深 | 7.10 保留机制和指标；手算 4096 tokens/top-2/16 experts/capacity factor 1.25，展示余量与单 expert 偏斜下的 overflow；对照 Switch top-1 与 GShard top-2 的不同账本。 |
| activation checkpointing、activation recomputation | 核心加深 | 7.11 删除把一般激活量写成 `M_act/K` 的直觉式暗示，区分整层重算和选择性重算；加入论文在 GPT-3/MT-NLG 特定设置下的显存/额外 FLOPs 报告值与边界。 |
| 通信/计算 overlap、DDP bucket、FSDP prefetch | 核心保留 | 7.12 保留串行与理想重叠上界；强调 overlap 需由时间线和真实 step time 证明。 |
| checkpoint、恢复、RNG/data cursor、拓扑变化 | 核心加深 | 7.13 区分 PyTorch DCP 的并行 I/O、load-time resharding、异步 staging 与应用层 manifest/原子提交；记录格式兼容告警和必须验证的 optimizer/data 状态。 |
| collective 故障、hang、样本重复 | 核心保留 | 7.14 保留故障模型、日志字段和恢复点语义；不把故障类型压缩掉。 |
| rank/PP/MoE 指标、定位瓶颈和选型 | 核心补充 | 7.15 新增 MFU/HFU 定义和 PaLM 的作者报告示例；7.16 原决策顺序保留。 |
| Python 资源账本 | 核心保留并修正 | 7.17 修正 Gbps/GB/s 命名；输入 token 与预测 target 分开；用明确的 24 层分段教学模型替换任意 `activation/K` 内存代理。代码只作静态审读，未执行。 |
| 常见故障、资料边界、八项练习和总结 | 核心保留并拓展 | 7.18—7.21 原故障主题和练习全部保留；增三项 schedule/MFU/SP-CP 对照练习，更新资料版本边界和总结指标。 |

本轮没有拆分文件；现有第五册目录已链接本章，因此没有变更导航结构。术语、论文索引和面试题做了按需同步。

## 研究来源与证据边界

### 原论文

1. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)，重点看 §3 的 optimizer/gradient/parameter partitioning、Table 1 的特定精度状态账本。它支撑阶段性分片机制，不支撑本文目标集群的吞吐预测。原论文示例的 optimizer/master-weight 字节口径与正文 7B 示例不同，二者没有混成同一数字比较。
2. [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)，重点看 §2.2 pipeline schedules、§3 并行方式、§5.3 性能结果。论文说明 interleaved schedule 可缩短 pipeline flush；在更大 batch 下 non-interleaved bubble 下降、interleaved 每样本通信更多，且结果受 scatter/gather 优化影响。论文的架构、GPU 和实现设置绑定其作者结果。
3. [GPipe](https://arxiv.org/abs/1811.06965)，重点看 §2 的分区、micro-batch、bubble 与重算，§3 的吞吐和分区均衡结果。用作 flush schedule 的基线，不将论文中的模型速度外推到现代 LLM。
4. [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)，重点看 §4.2—§6 的 Sequence Parallel/Selective Activation Recomputation、activation memory 公式、Table 2 与实验。作者在 TP=8、序列长度 2048 的 GPT-3 与 MT-NLG 配置下，选择性重算相对 TP baseline 报告 activation memory 降幅约 70%/65%、FLOPs overhead 约 2.7%/1.6%；另报告 SP+selective recomputation 联合内存相对 TP baseline 约 5× 降低。正文区分了这两项对照；数字均绑定论文配置，不代表所有模型的常数收益。
5. [GShard](https://arxiv.org/abs/2006.16668)，重点看 §2 的 group-level top-2 gating、capacity 和辅助负载目标；用于说明 top-2 dispatch 如何受容量限制。
6. [Switch Transformers](https://arxiv.org/abs/2101.03961)，重点看 §2 的 top-1 routing 和 expert capacity。论文明确说明提高 capacity factor 可减少 overflow，但空 slot 增加计算/通信和内存；超容量 token 在其实现中可不执行该 MoE 层 expert 计算。结果仅绑定其模型和训练配方。
7. [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)，重点看 §3 的 blockwise ring attention 算法和 §5.2 的 MFU 评测。作为按序列块传 K/V、在线累积局部 attention 的具体算法，不等于任意 Context Parallel backend。
8. [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)，重点看 §4.1 与 Appendix B。论文定义 MFU 为观测 token throughput 相对模型前向/反向必需 FLOPs 在峰值硬件上的理论吞吐；HFU 的实际 FLOPs 可计入 rematerialization。PaLM 540B 作者报告 46.2% MFU、57.8% HFU，绑定其 6144 TPU v4 训练配置和 FLOPs 计数口径。

### 框架和通信官方资料

1. [PyTorch DistributedDataParallel 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)：核对 gradients are averaged 的行为、sum loss 的 world-size 缩放风险、bucket 配置，以及 uneven-input `join()` 的 initial/effective world-size 除数和额外 collective 警告。版本固定为文档 2.14；实际 trainer/hook 可额外缩放，须按目标代码复核。
2. [PyTorch FSDP2 `fully_shard` 2.14](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html)：核对 FSDP2 per-parameter DTensor、与 FSDP1 flat parameter 的差异、前/后向 hook all-gather/reshard、底向上模块分组及 reduce-scatter 梯度语义。本文不把某一 wrap 策略称为通用最优。
3. [PyTorch Distributed Checkpoint 2.14](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html)：核对多 rank I/O、load-time resharding、异步 staging 与上传完成 future、state_dict rank 对齐要求，以及没有跨 PyTorch 版本向后兼容保证。DCP 文档未承诺应用级跨 shard 原子提交；manifest/checksum 方案在正文标为应用层设计。
4. [NVIDIA NCCL collective overview](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)：核对 AllReduce、Broadcast、AllGather、ReduceScatter、AlltoAll 和 point-to-point 等操作。它不能给出本集群可用带宽和选择算法的实测值。
5. [Megatron Core Context Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)：核对文档列出的 p2p ring、all-gather、a2a 等 CP communication type，以及 p2p path 中 K/V 交换、反向梯度路径和 CP ranks 的权重复制语义。`latest` 是动态文档，核对日期 2026-10-26；部署时需按已安装 revision 重查。

## 关键推演与手工复核

### 变长目标与 DDP 平均

样例 ranks 的 `(有效目标数, loss sum)` 为 `(100,50)、(200,110)、(400,220)`：

- 全局 loss：`(50+110+220)/(100+200+400)=380/700≈0.542857`。
- 各 rank mean 的等权平均：`(50/100+110/200+220/400)/3≈0.533333`。
- PyTorch DDP 默认对 P 个 DP ranks 的梯度取平均；局部 loss sum `S_r` 乘 `P/N`，其中 `N=Σn_r`，则 `1/P Σ∇(S_r P/N)=ΣG_r/N`。如果 API 先将 loss mean、框架又除 accumulation steps，或使用自定义 communication hook，要按真实缩放重推。
- 因果 LM 的输入位置数不必等于监督 target 数；例中 4096 input positions、内部 shift、无遮挡时 4095 targets/sample。若 mask 排除位置，使用 mask 后 target count。

### 参数状态字节账本

条件：7B 参数；BF16 parameter 2 bytes + BF16 gradient 2 bytes + Adam 两个 FP32 moment 8 bytes；DP=8；不计 master weights/activation/workspace/all-gather transient。

| 模式 | 计算 | GB（十进制） | GiB |
| --- | --- | ---: | ---: |
| DDP | `7B×(2+2+8)` | 84.0 | 78.2 |
| ZeRO-1 | `7B×(2+2+8/8)` | 35.0 | 32.6 |
| ZeRO-2 | `7B×(2+(2+8)/8)` | 22.75 | 21.2 |
| ZeRO-3/full shard | `7B×(2+2+8)/8` | 10.5 | 9.8 |

这只是常驻状态估算。full shard 的参数仍在计算前临时聚齐，不能按此表直接判 OOM 安全。

### MoE capacity

`T=4096, k=2, E=16, alpha=1.25`：共有 8192 routed assignments；均匀期望 512 assignments/expert；容量 `ceil(1.25×512)=640`；总 buffer 为 10240 slots。若一个 expert 收到全体 assignments 的 30%，约 2458 个，超过静态 capacity 约 1818。drop、padding、动态容量和 dropless dispatch 改变的不是同一个实现合同；正文没有把容量式当作所有框架的精确 API。

### Pipeline schedule 与 bubble

正文分别写 bubble/ideal=`(p-1)/m` 与 idle/total=`(p-1)/(m+p-1)`，前者对应 Megatron 的 bubble 相对理想计算时间，后者对应 GPipe 的 idle fraction/总 schedule 时间；不是两种 schedule 固有不变量。例：`p=8,m=16` 时分别为 `7/16≈43.8%` 与 `7/23≈30.4%`。代码返回第二种 idle/total fraction。两者都假设 stage 时间相近，不含具体通信、interleaving、virtual-chunk、activation 峰值等项；增加 m 还会改变 batch 组成、激活在途量和 kernel shape，不能用一个比例选 schedule。

### 静态教学账本的输入和预期关系

- `DP=8, TP=4, PP=2`，world size 为 64；只有 DP 进入 batch multiplier。
- micro-batch 2、4096 input positions/sample、4095 valid targets/sample、累积 4：input tokens 为 262144，valid target tokens 为 262080。
- ring all-reduce 的 DP payload 4 GiB、DP size 8：每 rank 理想发送量 `2×7/8×4=7 GiB`。在 200 GB/s 十进制 bytes/s 下传输下界约 0.0376 s，代码名中明确 GB/s；标称 Gbps 要先除 8。
- Pipeline PP=2、m=8 的 idle/total 简式 `1/9≈0.1111`；相对理想计算时间则为 `1/8=0.125`。
- MoE 账本设 routing tokens=8192、top-k=2、E=8、alpha=1.25：capacity=2560；40% 偏斜 assignment peak=6554，overflow=3994。
- activation 教学模型假设 24 层、每层激活 1 GiB、每 4 层一个段；完整保存约 24 GiB，保存 6 段输入加一段重算约 10 GiB。完整重算计为 `6N+2N` 对比 `6N`，因子 `8/6=1.333...`。不是 Transformer 通用显存预测。

以上代码仅做静态检查和手工算术核验，没有执行脚本或实际 collective；研究记录不标 `local-demo`/runtime verified。

## 本轮新增/改写的来源边界

- `source-backed`：论文和固定版官方文档支持对应机制、实验数字或 API 行为；论文数字只代表作者设置。
- `manual calculation`：loss、内存、capacity、ring 下界、bubble 和 toy code expected values 为手算，未由脚本运行。
- `engineering proposal`：manifest 最后提交、checksum、恢复演练是应用层恢复设计建议，不声称 PyTorch DCP 已自动提供事务语义。
- `needs-verification`：目标训练框架、硬件拓扑、网络和真实模型上的 MFU/HFU、吞吐、OOM 峰值、失效率、恢复成功率均未在本机或生产系统实测。

## 配套同步

- `PAPERS.md`：补充 GShard、PaLM 与机制导读，并固定 PyTorch 2.14 资料入口。
- `GLOSSARY_EN_ZH.md`：解释 FSDP1/FSDP2、MFU/HFU，并沿用已有 Valid Prediction Target Count 定义。
- `INTERVIEW_BANK.md`：新增 DDP 变长 token weighting、ZeRO/FSDP1/FSDP2、pipeline schedule、MFU/HFU 与 DCP resharding 追问。
- `book-05-llm-training/目录.md`：既有第七章链接正确，无需调整。
- 本章练习：在原八题后加 schedule、MFU/HFU 与 SP/CP 对照题；不重复复制到 `EXERCISES.md`。

## 未解决项

- 本章 Python 账本未执行；`progress_v2.md` 中第五册第三至第七章代码运行门禁继续开放。
- 没有在 PyTorch 2.14 CUDA 多卡、FSDP1/FSDP2、DeepSpeed、Megatron Core 或 NCCL 实例上复现实测。
- `Megatron Core latest`、DeepSpeed 教程等在线文档会变；固定环境时应检查实际依赖版本。
- Pipeline bubble 与 MFU 依赖口径/实现；记录论文、版本、token 定义、dtype 和 peak capability 后才可横向比较。
- DCP checkpoint 的实际原子提交和跨 topology 恢复能力取决于 storage writer、模型/optimizer state contract 和应用 manifest；正文没有把协议建议描述为库保证。
