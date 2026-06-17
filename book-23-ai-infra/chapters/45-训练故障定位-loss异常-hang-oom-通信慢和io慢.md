# 第 45 章 训练故障定位：loss 异常、hang、OOM、通信慢和 I/O 慢

上一章讲了 AI Infra 可观测性总览。本章进入具体故障定位：训练任务出问题时，如何系统排查。

大模型训练任务通常运行时间长、成本高、节点多、依赖复杂。一次训练故障可能浪费数百张 GPU 数小时。因此，训练故障定位是 AI Infra 工程师必须掌握的能力。

先记住一句话：

> 训练故障定位不要先猜模型或框架，而要按数据、代码、配置、资源、通信、存储和平台链路逐层排查。

## 45.0 本讲资料边界与第二轮精修口径

本章按通用大模型训练平台故障定位抽象来写，不绑定 PyTorch DDP、FSDP、DeepSpeed、Megatron、Kubernetes、Slurm、NCCL 具体版本、GPU 型号或云厂商平台实现。资料校准时，主要参考 PyTorch autograd anomaly detection、CUDA memory management、PyTorch profiler 对算子 / CPU / CUDA 耗时定位的口径，参考 NVIDIA NCCL troubleshooting 对通信、环境变量、网络和异步错误排查的工程边界，也结合前文训练可观测性、checkpoint、数据供给和实验追踪章节中的版本、manifest、rank 日志和事件时间线要求。

第二轮精修只做三件事：

1. 把 loss 异常、NaN、hang、OOM、通信慢、I/O 慢、checkpoint 和 resume 问题统一成故障样本。
2. 补齐 step time、loss spike、NaN / Inf、gradient health、显存预算、通信占比、data wait、checkpoint 完整性和最小复现公式。
3. 增加一个 0 依赖 Python demo，用 toy training incident cases 检查训练故障定位是否有足够证据链。

## 45.1 训练故障的常见类型

大模型训练常见故障包括：

1. loss 异常。
2. loss 不下降。
3. loss 突然 NaN。
4. 训练 hang 住。
5. OOM。
6. 节点失败。
7. NCCL 通信错误。
8. step time 变慢。
9. 数据读取慢。
10. checkpoint 保存失败。
11. 任务反复重试。
12. GPU 利用率低。

这些问题可能互相影响。

例如数据读取慢会导致 GPU 利用率低，通信慢会导致 step time 变长，OOM 可能来自 batch、sequence length 或显存碎片。

一个训练故障样本可以写成：

$$
F_i=(j_i,e_i,r_i,l_i,g_i,m_i,c_i,d_i,k_i,\Delta_i,p_i,z_i)
$$

其中 `j_i` 是 TrainingJob，`e_i` 是生命周期事件，`r_i` 是 rank 日志，`l_i` 是 loss / eval 曲线，`g_i` 是梯度和 optimizer 证据，`m_i` 是显存与 OOM 证据，`c_i` 是通信证据，`d_i` 是数据和 I/O 证据，`k_i` 是 checkpoint / resume 证据，`\Delta_i` 是代码 / 配置 / 数据版本 diff，`p_i` 是最小复现证据，`z_i` 是最终根因和修复记录。

故障定位证据覆盖率：

$$
C_{\mathrm{fault}}=\frac{1}{N}\sum_{i=1}^{N}I(e_i,r_i,l_i,g_i,m_i,c_i,d_i,k_i,\Delta_i,p_i,z_i\ \mathrm{present})
$$

## 45.2 排查总原则

训练故障排查可以遵循这个顺序：

1. 看任务状态。
2. 看最近事件。
3. 看错误日志。
4. 看训练指标。
5. 看系统资源。
6. 看数据读取。
7. 看分布式通信。
8. 看 checkpoint。
9. 看代码和配置变更。
10. 看数据版本变更。

不要只盯着最后一行报错。

分布式训练中，真正原因可能出现在另一个 rank、另一个节点或更早的事件里。

排查优先级可以写成：

$$
\mathrm{priority}(s)=w_tT_s+w_iI_s+w_rR_s+w_cC_s
$$

其中 `T_s` 是时间上是否更早，`I_s` 是影响面，`R_s` 是是否可复现，`C_s` 是证据可信度。训练事故不要只看最后一个报错，而要优先看最早、影响最大、可复现且证据最完整的线索。

## 45.3 先看任务生命周期事件

训练平台应该记录事件：

1. Job submitted。
2. Job scheduled。
3. Pods created。
4. Image pulled。
5. Dataset mounted。
6. Training started。
7. Checkpoint saved。
8. Node lost。
9. Pod restarted。
10. Job failed。

这些事件能快速判断问题发生在哪个阶段。

如果任务连训练都没开始，就不要先看模型代码，应该看调度、镜像、权限和数据挂载。

## 45.4 Loss 异常怎么排查

Loss 异常包括：

1. loss 不下降。
2. loss 波动异常。
3. loss 突然升高。
4. loss 变 NaN。
5. eval loss 和 train loss 背离。

排查维度：

1. 数据是否变化。
2. 学习率是否过大。
3. batch size 是否变化。
4. mixed precision 是否稳定。
5. loss scale 是否异常。
6. 标签或 mask 是否错误。
7. tokenizer 是否匹配。
8. padding 和 attention mask 是否正确。
9. 梯度是否爆炸。
10. checkpoint resume 是否正确。

Loss 是模型、数据和训练配置共同作用的结果。

loss spike 可以用相对变化衡量：

$$
S_t=\frac{L_t-\mathrm{median}(L_{t-w:t-1})}{\max(\epsilon,\mathrm{median}(L_{t-w:t-1}))}
$$

当 `S_t` 超过阈值时，再结合数据 batch、学习率、梯度、precision 和 resume 证据判断根因。

## 45.5 Loss NaN

Loss NaN 是常见严重问题。

可能原因：

1. learning rate 过大。
2. 梯度爆炸。
3. FP16 溢出。
4. loss scale 不合适。
5. 输入数据包含异常值。
6. labels 全是 ignore index 或异常。
7. logits 出现 inf。
8. 自定义 loss 有除零。
9. resume checkpoint 状态损坏。

排查方法：

1. 查看 NaN 出现的 step。
2. 查看该 step 数据 batch。
3. 查看 gradient norm。
4. 降低学习率。
5. 开启 gradient clipping。
6. 切换 BF16 或调整 precision。
7. 检查 loss scale 日志。
8. 单卡复现最小 batch。

NaN 不要只靠跳过 batch 掩盖，必须找到触发条件。

NaN / Inf 命中率：

$$
R_{\mathrm{nan}}=\frac{N_{\mathrm{nan\ loss}}+N_{\mathrm{inf\ logit}}+N_{\mathrm{nan\ grad}}}{N_{\mathrm{checked}}}
$$

`R_nan>0` 时，至少要保存触发 batch、loss scale、precision、gradient norm、logits finite check 和最小复现配置。

## 45.6 Loss 不下降

Loss 不下降可能原因：

1. 学习率太小。
2. 数据标签错误。
3. 模型参数没有更新。
4. optimizer 没有正确加载。
5. 训练样本被 mask 掉。
6. tokenizer 不匹配。
7. 数据重复或质量差。
8. 冻结了不该冻结的层。
9. LoRA target modules 配置错误。
10. 梯度为 0。

排查方法：

1. 看参数梯度是否非零。
2. 看 optimizer step 是否执行。
3. 看 learning rate 曲线。
4. 看有效 token 数。
5. 看样本和标签是否合理。
6. 小数据集 overfit 测试。

如果模型连小数据集都无法 overfit，通常是训练代码或配置问题。

梯度更新健康度：

$$
G_{\mathrm{update}}=I(R_{\mathrm{nonzero\_grad}}\ge\tau_g)\cdot I(N_{\mathrm{optimizer\_step}}>0)\cdot I(\mathrm{lr}_t>0)
$$

其中 `R_nonzero_grad` 是非零梯度参数占比。loss 不下降时，先确认模型真的在更新。

## 45.7 Hang 住怎么排查

训练 hang 指任务没有退出，但 step 不再推进。

常见原因：

1. 某个 rank 卡在 dataloader。
2. 某个 rank OOM 或崩溃但其他 rank 等待。
3. NCCL collective 不一致。
4. 数据读取阻塞。
5. checkpoint 保存阻塞。
6. 节点网络异常。
7. barrier 使用不当。
8. 死锁。

排查方法：

1. 看所有 rank 最后一条日志。
2. 看 step 是否停止增长。
3. 看 GPU 利用率是否为 0 或部分为 0。
4. 看 dataloader worker 状态。
5. 看 NCCL 日志。
6. 看节点网络和存储指标。
7. 看是否有一个 rank 先失败。

分布式训练 hang 往往不是所有 rank 同时出错，而是一个 rank 出问题拖住所有 rank。

hang 判定可以写成：

$$
G_{\mathrm{hang}}=I(\Delta \mathrm{step}=0)\cdot I(\Delta t>\tau)\cdot I(\mathrm{job\ status}=\mathrm{running})
$$

如果某些 rank GPU 利用率为 0、某些 rank 还在等待 collective，优先找最早停止推进的 rank。

## 45.8 OOM 怎么排查

OOM 可以发生在：

1. GPU 显存。
2. CPU 内存。
3. shared memory。
4. dataloader worker。
5. checkpoint 保存阶段。

GPU OOM 常见原因：

1. batch size 太大。
2. sequence length 太长。
3. activation 占用过高。
4. optimizer state 过大。
5. ZeRO 配置不当。
6. gradient checkpointing 未开启。
7. flash attention 未生效。
8. 显存碎片。
9. eval 阶段未关闭缓存或梯度。

OOM 排查要看发生阶段：forward、backward、optimizer step、eval、checkpoint。

不同阶段对应不同优化。

训练显存可以粗略拆成：

$$
M_{\mathrm{train}}=M_{\mathrm{param}}+M_{\mathrm{grad}}+M_{\mathrm{optim}}+M_{\mathrm{act}}+M_{\mathrm{temp}}+M_{\mathrm{fragment}}
$$

OOM 阶段门禁：

$$
G_{\mathrm{oom\_phase}}=I(\mathrm{phase}\in\{\mathrm{forward},\mathrm{backward},\mathrm{optimizer},\mathrm{eval},\mathrm{checkpoint}\})\cdot I(M_{\mathrm{peak}}\ \mathrm{recorded})
$$

不知道 OOM 发生阶段，就很难选择正确修复策略。

## 45.9 GPU OOM 的常见解决方案

解决方案包括：

1. 降低 micro batch size。
2. 使用 gradient accumulation 保持 global batch。
3. 降低 sequence length。
4. 开启 activation checkpointing。
5. 使用 ZeRO/FSDP。
6. 使用 BF16/FP16。
7. 使用 flash attention。
8. 优化 optimizer state。
9. eval 时使用 no_grad。
10. 清理不必要缓存。

不要盲目降低 global batch size，否则训练动态也会变化。

优先区分 micro batch 和 global batch。

## 45.10 通信慢怎么排查

分布式训练中，通信慢会导致 step time 变长。

常见原因：

1. 网络带宽不足。
2. IB/RoCE 配置错误。
3. NCCL 参数不合适。
4. 拓扑不佳。
5. 跨机通信过多。
6. 某个节点慢。
7. 网卡错误。
8. all-reduce 数据量过大。

排查指标：

1. step time。
2. compute time。
3. communication time。
4. NCCL all-reduce latency。
5. 网络吞吐。
6. 重传和错误包。
7. GPU 等待时间。

通信慢通常需要结合 profiler、NCCL 日志和网络指标。

通信占比：

$$
R_{\mathrm{comm}}=\frac{T_{\mathrm{comm}}}{T_{\mathrm{step}}}
$$

straggler 比例：

$$
R_{\mathrm{straggle}}=\frac{\max_r T_r-\mathrm{median}_r(T_r)}{\max(\epsilon,\mathrm{median}_r(T_r))}
$$

通信慢要同时看 `R_comm`、rank 间 step time 分布、NCCL 日志和网络错误。

## 45.11 NCCL 错误

NCCL 错误常见表现：

1. timeout。
2. unhandled system error。
3. connection refused。
4. rank mismatch。
5. collective hang。

可能原因：

1. 某个 rank 崩溃。
2. 节点网络不通。
3. 端口冲突。
4. 环境变量错误。
5. world size 配置错误。
6. GPU 拓扑问题。
7. IB/RoCE 驱动问题。

排查时要找到最早报错的 rank。

最后报 NCCL timeout 的 rank 未必是根因。

## 45.12 I/O 慢怎么排查

训练 I/O 慢会导致 GPU 等数据。

常见原因：

1. 对象存储吞吐不足。
2. 小文件过多。
3. 网络带宽不足。
4. 解压缩 CPU 瓶颈。
5. tokenizer 在线计算慢。
6. shuffle buffer 太小或太大。
7. dataloader worker 不足。
8. 本地缓存未命中。
9. 共享文件系统抖动。

排查指标：

1. dataloader time。
2. batch ready time。
3. GPU idle time。
4. storage read throughput。
5. cache hit rate。
6. CPU utilization。
7. worker queue length。

I/O 慢往往表现为 GPU 利用率周期性下降。

data wait 比例：

$$
R_{\mathrm{data}}=\frac{T_{\mathrm{data}}}{T_{\mathrm{step}}}
$$

缓存命中率：

$$
H_{\mathrm{cache}}=\frac{N_{\mathrm{cache\ hit}}}{N_{\mathrm{read}}}
$$

`R_data` 高且 `H_cache` 低时，优先排查对象存储、小文件、预处理、worker 和本地缓存。

## 45.13 GPU 利用率低

GPU 利用率低不一定是 GPU 问题。

可能原因：

1. 数据读取慢。
2. CPU preprocessing 慢。
3. batch 太小。
4. 通信等待。
5. checkpoint 太频繁。
6. evaluation 太频繁。
7. 负载不均衡。
8. kernel 效率低。
9. dataloader hang。

排查要拆分 step time：

```text
step time = data time + forward + backward + optimizer + communication + checkpoint overhead
```

只有知道哪一段慢，才能优化。

更正式地写：

$$
T_{\mathrm{step}}=T_{\mathrm{data}}+T_{\mathrm{fwd}}+T_{\mathrm{bwd}}+T_{\mathrm{optim}}+T_{\mathrm{comm}}+T_{\mathrm{ckpt}}+T_{\mathrm{other}}
$$

GPU 利用率低的根因应该落到上面某一项，而不是直接归因成“GPU 有问题”。

## 45.14 Checkpoint 故障

Checkpoint 相关故障包括：

1. 保存失败。
2. 保存太慢。
3. 文件不完整。
4. 恢复失败。
5. 状态不一致。
6. 存储空间不足。
7. 多 rank 写入冲突。

排查维度：

1. checkpoint 大小。
2. 保存频率。
3. 存储吞吐。
4. rank 写入策略。
5. manifest 是否完整。
6. checksum 是否通过。
7. optimizer state 是否保存。
8. resume 配置是否匹配。

Checkpoint 是训练容错基础，不能只靠“目录里看起来有文件”。

checkpoint 完整性门禁：

$$
G_{\mathrm{ckpt}}=I(M_{\mathrm{manifest}}=1)\cdot I(C_{\mathrm{checksum}}=1)\cdot I(S_{\mathrm{model}}=1)\cdot I(S_{\mathrm{optim}}=1)\cdot I(S_{\mathrm{sched}}=1)\cdot I(S_{\mathrm{rng}}=1)
$$

## 45.15 Resume 后异常

从 checkpoint 恢复后可能出现：

1. loss 突然跳变。
2. learning rate 不连续。
3. optimizer 状态丢失。
4. dataloader 重复或漏读。
5. random seed 不一致。
6. global step 错误。
7. 分布式状态不一致。

恢复时要校验：

1. model state。
2. optimizer state。
3. scheduler state。
4. global step。
5. random state。
6. dataloader state。
7. config 是否一致。

只加载模型权重，不等于完整恢复训练。

resume 连续性：

$$
G_{\mathrm{resume}}=I(\Delta \mathrm{step}=1)\cdot I(|\mathrm{lr}_{t+1}-\mathrm{lr}^{\mathrm{expected}}_{t+1}|\le\epsilon)\cdot I(|L_{t+1}-L_t|\le\rho)
$$

## 45.16 数据问题导致的训练异常

数据问题包括：

1. 空样本。
2. 超长样本。
3. 非法 JSON。
4. 标签错位。
5. tokenizer 结果异常。
6. 特殊 token 错误。
7. 重复样本过多。
8. 数据分布变化。
9. train/eval 泄漏。
10. 敏感数据混入。

排查方法：

1. 保存触发异常的 batch。
2. 打印样本 token 长度分布。
3. 校验 schema。
4. 对比 dataset diff report。
5. 回滚数据版本验证。

数据问题是训练异常中最容易被低估的一类。

数据异常率：

$$
R_{\mathrm{data\_fault}}=\frac{N_{\mathrm{empty}}+N_{\mathrm{too\_long}}+N_{\mathrm{schema\_bad}}+N_{\mathrm{label\_bad}}+N_{\mathrm{tokenizer\_bad}}}{N_{\mathrm{sample}}}
$$

## 45.17 配置变更排查

训练异常常来自配置变更。

需要对比：

1. learning rate。
2. batch size。
3. sequence length。
4. precision。
5. ZeRO/FSDP config。
6. tokenizer。
7. data config。
8. model config。
9. checkpoint config。
10. distributed config。

实验追踪系统应该能显示两个 run 的 config diff。

没有 config diff，排查会非常低效。

配置差异集合：

$$
\Delta_{\mathrm{cfg}}=\{k: \mathrm{cfg}^{(a)}_k\ne \mathrm{cfg}^{(b)}_k\}
$$

排查时要把 `Delta_cfg` 和 dataset diff、code commit diff、environment diff 放在同一张证据表里。

## 45.18 最小复现

复杂故障要尽量最小复现。

方法：

1. 从多机多卡缩到单机多卡。
2. 从大数据缩到小数据。
3. 固定 seed。
4. 固定某个异常 batch。
5. 关闭不必要优化。
6. 简化配置。

如果单卡小数据能复现，多半是代码、数据或配置问题。

如果只有多机才复现，更可能是通信、调度或环境问题。

最小复现就绪门禁：

$$
G_{\mathrm{repro}}=I(\mathrm{seed\ fixed})\cdot I(\mathrm{batch\ saved})\cdot I(\mathrm{config\ frozen})\cdot I(\mathrm{data\ version\ frozen})\cdot I(\mathrm{single\ node\ trial\ recorded})
$$

## 45.19 故障定位需要的工具

常用工具和能力：

1. 训练指标 dashboard。
2. GPU metrics。
3. NCCL logs。
4. profiler。
5. dataloader timing。
6. config diff。
7. dataset diff。
8. checkpoint manifest。
9. per-rank logs。
10. event timeline。

平台应把这些整合到 TrainingJob 页面，而不是让工程师到处找。

## 45.20 故障复盘

训练故障修复后要复盘。

复盘应记录：

1. 故障现象。
2. 影响范围。
3. 根因。
4. 发现方式。
5. 修复方式。
6. 是否可提前告警。
7. 是否需要平台自动防护。
8. 是否需要文档或 runbook。

成熟平台会把常见故障变成自动检测和 runbook。

## 45.21 训练故障定位审计指标和最小 demo

把本章落到平台验收时，可以用 16 个门禁：

1. Training Fault Evidence Coverage：事件、rank 日志、loss、梯度、显存、通信、数据 I/O、checkpoint、配置 diff、最小复现和诊断结论是否齐全。
2. Lifecycle Event Timeline：任务提交、调度、启动、最后推进、异常检测等事件是否按时间线记录。
3. Rank Log Completeness：所有 rank 是否都有 node、last step、last event、status 和 heartbeat。
4. Loss Anomaly Diagnosis：loss spike 是否绑定触发 step、异常 batch、学习率、precision、梯度和 tokenizer 证据。
5. NaN Inf Guard：loss、logits、gradient 的非有限值检查是否覆盖，并且能在触发时停止或回滚。
6. Update Health Check：非零梯度比例、optimizer step、learning rate 和参数更新是否证明训练真的在更新。
7. Hang Straggler Detection：step 停止推进、rank heartbeat、collective wait、dataloader 状态和可疑 rank 是否定位清楚。
8. OOM Phase Memory Evidence：OOM 阶段、显存峰值、显存上限、显存拆分和修复建议是否完整。
9. Communication Bottleneck Diagnosis：通信占比、rank step time 分布、NCCL 日志、网络指标和首个错误 rank 是否齐全。
10. IO Data Bottleneck Diagnosis：data wait、cache hit、worker queue、存储吞吐和坏样本隔离是否齐全。
11. Checkpoint Integrity：manifest、checksum、model、optimizer、scheduler、RNG 和 committed 状态是否完整。
12. Resume Continuity：step、learning rate、loss、optimizer、scheduler 和 dataloader cursor 是否连续。
13. Data Fault Isolation：空样本、超长样本、schema、label、tokenizer 错误是否统计，并且触发 batch 可隔离。
14. Config Code Diff Readiness：配置 diff、代码 commit diff、数据 diff 和环境 diff 是否能一起对比。
15. Minimal Reproduction Readiness：seed、异常 batch、配置、数据版本、单机试验和复现命令是否固定。
16. Training Fault Diagnosis Gate：最终是否有 owner、runbook、postmortem、P0 风险阻断和训练故障定位门禁。

综合门禁可以写成：

$$
G_{\mathrm{training\_fault}}=\prod_{j=1}^{16}G_j
$$

其中每个 `G_j` 对应上面一个子门禁。训练故障定位题不要只说“看日志”，而要证明证据链能从现象走到根因、修复和复盘。

下面是一个 0 依赖 demo。它不模拟真实分布式训练，而是用 toy incident cases 检查一个训练平台是否能把 loss 异常、NaN、hang、OOM、通信慢、I/O 慢、checkpoint、resume、数据问题、配置变更和最小复现放进同一套诊断门禁。

```python
from copy import deepcopy


class MiniTrainingFaultDiagnosisAudit:
    GATES = [
        "training_fault_evidence_coverage",
        "lifecycle_event_timeline",
        "rank_log_completeness",
        "loss_anomaly_diagnosis",
        "nan_inf_guard",
        "update_health_check",
        "hang_straggler_detection",
        "oom_phase_memory_evidence",
        "communication_bottleneck_diagnosis",
        "io_data_bottleneck_diagnosis",
        "checkpoint_integrity",
        "resume_continuity",
        "data_fault_isolation",
        "config_code_diff_readiness",
        "minimal_reproduction_readiness",
        "training_fault_diagnosis_gate",
    ]

    EVIDENCE_FIELDS = [
        "events",
        "rank_logs",
        "loss",
        "gradient",
        "memory",
        "communication",
        "data_io",
        "checkpoint",
        "config_diff",
        "reproduction",
        "diagnosis",
    ]

    RANK_LOG_FIELDS = ["rank", "node", "last_step", "last_event", "status", "heartbeat_age_s"]
    VALID_OOM_PHASES = {"forward", "backward", "optimizer", "eval", "checkpoint"}

    @staticmethod
    def present(record, key):
        return key in record and record[key] is not None and record[key] != ""

    def coverage(self, record, fields):
        if not record:
            return 0.0
        return sum(1 for field in fields if self.present(record, field)) / len(fields)

    @staticmethod
    def median(values):
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2

    def loss_spike(self, case):
        loss = case["loss"]
        values = loss["values"]
        window = values[-loss["window"] - 1:-1]
        baseline = self.median(window)
        return (values[-1] - baseline) / max(1e-12, baseline)

    def nan_inf_rate(self, case):
        item = case["nan_inf"]
        total = item["checked_tensors"]
        bad = item["nan_loss"] + item["inf_logit"] + item["nan_grad"]
        return bad / total if total else 1.0

    def comm_ratio(self, case):
        comm = case["communication"]
        return comm["comm_time_s"] / comm["step_time_s"]

    def straggler_ratio(self, case):
        times = case["communication"]["rank_step_times_s"]
        med = self.median(times)
        return (max(times) - med) / max(1e-12, med)

    def data_wait_ratio(self, case):
        data = case["data_io"]
        return data["data_time_s"] / data["step_time_s"]

    def cache_hit_rate(self, case):
        data = case["data_io"]
        return data["cache_hits"] / data["reads"]

    def data_fault_rate(self, case):
        data = case["data_fault"]
        bad = data["empty"] + data["too_long"] + data["schema_bad"] + data["label_bad"] + data["tokenizer_bad"]
        return bad / data["samples"]

    def training_fault_evidence_coverage(self, case):
        evidence = case.get("evidence", {})
        return self.coverage(evidence, self.EVIDENCE_FIELDS) == 1.0 and all(
            evidence[field] is True for field in self.EVIDENCE_FIELDS
        )

    def lifecycle_event_timeline(self, case):
        events = case.get("lifecycle", [])
        names = [event.get("event") for event in events]
        times = [event.get("ts") for event in events]
        required = ["job_submitted", "job_scheduled", "training_started", "last_progress", "incident_detected"]
        return all(name in names for name in required) and times == sorted(times)

    def rank_log_completeness(self, case):
        ranks = case.get("ranks", {})
        logs = ranks.get("logs", [])
        logged = {log.get("rank") for log in logs}
        expected = set(range(ranks.get("world_size", 0)))
        fields_ok = all(self.coverage(log, self.RANK_LOG_FIELDS) == 1.0 for log in logs)
        return bool(expected) and logged == expected and fields_ok

    def loss_anomaly_diagnosis(self, case):
        loss = case.get("loss", {})
        spike = self.loss_spike(case)
        required = [
            "trigger_step",
            "batch_saved",
            "lr_checked",
            "precision_checked",
            "gradient_checked",
            "tokenizer_checked",
        ]
        if self.coverage(loss, required) < 1.0:
            return False
        return spike <= loss["spike_threshold"] or all(loss[field] is True for field in required[1:])

    def nan_inf_guard(self, case):
        item = case.get("nan_inf", {})
        if item.get("checked_tensors", 0) <= 0:
            return False
        nonfinite = self.nan_inf_rate(case)
        checks_ready = item.get("finite_logits_check") is True and item.get("loss_scale_logged") is True
        return checks_ready and (nonfinite == 0.0 or item.get("stop_on_nonfinite") is True)

    def update_health_check(self, case):
        update = case.get("update", {})
        return (
            update.get("nonzero_grad_ratio", 0.0) >= 0.9
            and update.get("optimizer_steps", 0) > 0
            and update.get("lr", 0.0) > 0
            and update.get("parameters_changed") is True
        )

    def hang_straggler_detection(self, case):
        hang = case.get("hang", {})
        detected = (
            hang.get("step_delta") == 0
            and hang.get("seconds_since_progress", 0) > hang.get("threshold_seconds", 10**9)
            and hang.get("job_status") == "running"
        )
        heartbeats = hang.get("rank_heartbeats", {})
        return (
            detected
            and bool(heartbeats)
            and hang.get("culprit_rank") in heartbeats
            and hang.get("collective_wait_seen") is True
            and hang.get("dataloader_state_seen") is True
        )

    def oom_phase_memory_evidence(self, case):
        memory = case.get("memory", {})
        components = memory.get("components", {})
        peak = memory.get("peak_gib", 0.0)
        limit = memory.get("limit_gib", 0.0)
        return (
            memory.get("oom_phase") in self.VALID_OOM_PHASES
            and peak > 0
            and limit > 0
            and peak <= limit
            and all(value >= 0 for value in components.values())
            and sum(components.values()) <= peak
            and bool(memory.get("remediation"))
        )

    def communication_bottleneck_diagnosis(self, case):
        comm = case.get("communication", {})
        return (
            self.comm_ratio(case) >= 0.2
            and len(comm.get("rank_step_times_s", [])) == case["ranks"]["world_size"]
            and comm.get("nccl_logs_collected") is True
            and comm.get("network_metrics") is True
            and comm.get("first_error_rank") in range(case["ranks"]["world_size"])
        )

    def io_data_bottleneck_diagnosis(self, case):
        data = case.get("data_io", {})
        return (
            self.data_wait_ratio(case) >= 0.1
            and self.cache_hit_rate(case) >= 0.8
            and data.get("worker_queue_observed") is True
            and data.get("storage_throughput_mibs", 0) > 0
            and data.get("bad_sample_quarantined") is True
        )

    def checkpoint_integrity(self, case):
        checkpoint = case.get("checkpoint", {})
        required = ["manifest", "checksum", "model", "optimizer", "scheduler", "rng", "committed"]
        return all(checkpoint.get(field) is True for field in required)

    def resume_continuity(self, case):
        resume = case.get("resume", {})
        lr_ok = abs(resume.get("lr_actual", 0.0) - resume.get("lr_expected", 1.0)) <= 1e-6
        loss_ok = abs(resume.get("loss_after", 0.0) - resume.get("loss_before", 0.0)) <= resume.get("rho", 0.0)
        return (
            resume.get("step_delta") == 1
            and lr_ok
            and loss_ok
            and resume.get("optimizer") is True
            and resume.get("scheduler") is True
            and resume.get("dataloader") is True
        )

    def data_fault_isolation(self, case):
        data = case.get("data_fault", {})
        return (
            data.get("samples", 0) > 0
            and self.data_fault_rate(case) <= 0.01
            and data.get("trigger_batch_saved") is True
            and data.get("dataset_diff_report") is True
            and data.get("quarantine") is True
        )

    def config_code_diff_readiness(self, case):
        diff = case.get("diff", {})
        return (
            bool(diff.get("config_diff_keys"))
            and diff.get("code_commit_diff") is True
            and diff.get("dataset_diff") is True
            and diff.get("environment_diff") is True
            and bool(diff.get("owner"))
        )

    def minimal_reproduction_readiness(self, case):
        repro = case.get("repro", {})
        required = [
            "seed_fixed",
            "batch_saved",
            "config_frozen",
            "data_version_frozen",
            "single_node_trial_recorded",
            "command_captured",
        ]
        return all(repro.get(field) is True for field in required)

    def training_fault_diagnosis_gate(self, case):
        gate = case.get("platform_gate", {})
        return (
            gate.get("enabled") is True
            and bool(gate.get("owner"))
            and bool(gate.get("runbook"))
            and gate.get("postmortem_required") is True
            and gate.get("p0_open") is False
        )

    def audit_case(self, case):
        return {gate: getattr(self, gate)(case) for gate in self.GATES}

    def run_all(self, cases):
        results = {case["case_id"]: self.audit_case(case) for case in cases}
        metrics = {}
        for gate in self.GATES:
            passed = sum(1 for result in results.values() if result[gate])
            metrics[gate] = round(passed / len(cases), 3)
        failed_cases = [
            case_id
            for case_id, result in results.items()
            if not all(result.values())
        ]
        failed_gates = [
            gate
            for gate in self.GATES
            if any(not result[gate] for result in results.values())
        ]
        return {
            "metrics": metrics,
            "hard_blocker_count": len(failed_cases),
            "failed_cases": failed_cases,
            "failed_gates": failed_gates,
            "training_fault_diagnosis_gate_pass": metrics["training_fault_diagnosis_gate"] == 1.0,
        }

    def example_outputs(self, case):
        ranks = case["ranks"]
        return {
            "evidence_coverage": round(self.coverage(case["evidence"], self.EVIDENCE_FIELDS), 3),
            "timeline_events": len(case["lifecycle"]),
            "rank_log_coverage": round(len(ranks["logs"]) / ranks["world_size"], 3),
            "loss_spike": round(self.loss_spike(case), 3),
            "nan_inf_rate": round(self.nan_inf_rate(case), 3),
            "nonzero_grad_ratio": case["update"]["nonzero_grad_ratio"],
            "hang_detected": self.hang_straggler_detection(case),
            "oom_phase": case["memory"]["oom_phase"],
            "memory_peak_gib": case["memory"]["peak_gib"],
            "comm_ratio": round(self.comm_ratio(case), 3),
            "straggler_ratio": round(self.straggler_ratio(case), 3),
            "data_wait_ratio": round(self.data_wait_ratio(case), 3),
            "cache_hit_rate": round(self.cache_hit_rate(case), 3),
            "checkpoint_integrity": self.checkpoint_integrity(case),
            "resume_ready": self.resume_continuity(case),
            "data_fault_rate": round(self.data_fault_rate(case), 3),
            "config_diff_keys": case["diff"]["config_diff_keys"],
            "minimal_repro_ready": self.minimal_reproduction_readiness(case),
        }


def build_good_case():
    return {
        "case_id": "full_training_fault_diagnosis",
        "evidence": {
            "events": True,
            "rank_logs": True,
            "loss": True,
            "gradient": True,
            "memory": True,
            "communication": True,
            "data_io": True,
            "checkpoint": True,
            "config_diff": True,
            "reproduction": True,
            "diagnosis": True,
        },
        "lifecycle": [
            {"ts": 1, "event": "job_submitted"},
            {"ts": 2, "event": "job_scheduled"},
            {"ts": 3, "event": "training_started"},
            {"ts": 205, "event": "last_progress"},
            {"ts": 236, "event": "incident_detected"},
        ],
        "ranks": {
            "world_size": 4,
            "logs": [
                {"rank": 0, "node": "node-a", "last_step": 205, "last_event": "all_reduce_wait", "status": "running", "heartbeat_age_s": 20},
                {"rank": 1, "node": "node-a", "last_step": 205, "last_event": "all_reduce_wait", "status": "running", "heartbeat_age_s": 22},
                {"rank": 2, "node": "node-b", "last_step": 205, "last_event": "all_reduce_wait", "status": "running", "heartbeat_age_s": 18},
                {"rank": 3, "node": "node-b", "last_step": 204, "last_event": "dataloader_wait", "status": "running", "heartbeat_age_s": 840},
            ],
        },
        "loss": {
            "values": [2.4, 2.1, 1.9, 1.8, 3.2],
            "window": 4,
            "spike_threshold": 0.3,
            "trigger_step": 205,
            "batch_saved": True,
            "lr_checked": True,
            "precision_checked": True,
            "gradient_checked": True,
            "tokenizer_checked": True,
        },
        "nan_inf": {
            "checked_tensors": 1000,
            "nan_loss": 0,
            "inf_logit": 0,
            "nan_grad": 0,
            "finite_logits_check": True,
            "loss_scale_logged": True,
            "stop_on_nonfinite": True,
        },
        "update": {
            "nonzero_grad_ratio": 0.94,
            "optimizer_steps": 205,
            "lr": 0.0002,
            "parameters_changed": True,
        },
        "hang": {
            "step_delta": 0,
            "seconds_since_progress": 900,
            "threshold_seconds": 600,
            "job_status": "running",
            "rank_heartbeats": {0: 20, 1: 22, 2: 18, 3: 840},
            "culprit_rank": 3,
            "collective_wait_seen": True,
            "dataloader_state_seen": True,
        },
        "memory": {
            "oom_phase": "backward",
            "peak_gib": 75.0,
            "limit_gib": 80.0,
            "components": {"param": 14.0, "grad": 14.0, "optim": 28.0, "activation": 16.0, "temp": 1.5},
            "remediation": ["activation_checkpointing", "micro_batch_down"],
        },
        "communication": {
            "step_time_s": 8.0,
            "comm_time_s": 2.2,
            "rank_step_times_s": [7.8, 7.9, 8.0, 11.2],
            "nccl_logs_collected": True,
            "network_metrics": True,
            "first_error_rank": 3,
        },
        "data_io": {
            "data_time_s": 1.4,
            "step_time_s": 8.0,
            "cache_hits": 930,
            "reads": 1000,
            "worker_queue_observed": True,
            "storage_throughput_mibs": 780,
            "bad_sample_quarantined": True,
        },
        "checkpoint": {
            "manifest": True,
            "checksum": True,
            "model": True,
            "optimizer": True,
            "scheduler": True,
            "rng": True,
            "committed": True,
        },
        "resume": {
            "step_delta": 1,
            "lr_expected": 0.0002,
            "lr_actual": 0.00020001,
            "loss_before": 1.88,
            "loss_after": 1.91,
            "rho": 0.1,
            "optimizer": True,
            "scheduler": True,
            "dataloader": True,
        },
        "data_fault": {
            "samples": 10000,
            "empty": 2,
            "too_long": 5,
            "schema_bad": 8,
            "label_bad": 4,
            "tokenizer_bad": 1,
            "trigger_batch_saved": True,
            "dataset_diff_report": True,
            "quarantine": True,
        },
        "diff": {
            "config_diff_keys": ["learning_rate", "precision"],
            "code_commit_diff": True,
            "dataset_diff": True,
            "environment_diff": True,
            "owner": "training-platform",
        },
        "repro": {
            "seed_fixed": True,
            "batch_saved": True,
            "config_frozen": True,
            "data_version_frozen": True,
            "single_node_trial_recorded": True,
            "command_captured": True,
        },
        "platform_gate": {
            "enabled": True,
            "owner": "train-debug-oncall",
            "runbook": "runbook://training-fault",
            "postmortem_required": True,
            "p0_open": False,
        },
    }


def build_bad_cases(good_case):
    cases = []

    case = deepcopy(good_case)
    case["case_id"] = "fault_evidence_missing_bad"
    case["evidence"]["diagnosis"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "lifecycle_timeline_bad"
    case["lifecycle"][3]["ts"] = 260
    case["lifecycle"][4]["ts"] = 236
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "rank_log_missing_bad"
    case["ranks"]["logs"].pop()
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "loss_anomaly_diagnosis_bad"
    case["loss"]["batch_saved"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "nan_inf_guard_bad"
    case["nan_inf"]["finite_logits_check"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "update_health_bad"
    case["update"]["nonzero_grad_ratio"] = 0.4
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "hang_straggler_detection_bad"
    case["hang"]["culprit_rank"] = None
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "oom_phase_memory_bad"
    case["memory"]["oom_phase"] = "unknown"
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "communication_bottleneck_bad"
    case["communication"]["nccl_logs_collected"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "io_data_bottleneck_bad"
    case["data_io"]["worker_queue_observed"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "checkpoint_integrity_bad"
    case["checkpoint"]["checksum"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "resume_continuity_bad"
    case["resume"]["step_delta"] = 5
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "data_fault_isolation_bad"
    case["data_fault"]["dataset_diff_report"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "config_code_diff_bad"
    case["diff"]["config_diff_keys"] = []
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "minimal_reproduction_bad"
    case["repro"]["single_node_trial_recorded"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "training_fault_gate_missing_bad"
    case["platform_gate"]["enabled"] = False
    cases.append(case)

    return cases


audit = MiniTrainingFaultDiagnosisAudit()
good = build_good_case()
cases = [good] + build_bad_cases(good)
summary = audit.run_all(cases)

print("training_fault_examples=" + repr(audit.example_outputs(good)))
print("metrics=" + repr(summary["metrics"]))
print("hard_blocker_count=" + repr(summary["hard_blocker_count"]))
print("failed_cases=" + repr(summary["failed_cases"]))
print("failed_gates=" + repr(summary["failed_gates"]))
print("training_fault_diagnosis_gate_pass=" + repr(summary["training_fault_diagnosis_gate_pass"]))
```

参考输出应类似：

```text
training_fault_examples={'evidence_coverage': 1.0, 'timeline_events': 5, 'rank_log_coverage': 1.0, 'loss_spike': 0.6, 'nan_inf_rate': 0.0, 'nonzero_grad_ratio': 0.94, 'hang_detected': True, 'oom_phase': 'backward', 'memory_peak_gib': 75.0, 'comm_ratio': 0.275, 'straggler_ratio': 0.409, 'data_wait_ratio': 0.175, 'cache_hit_rate': 0.93, 'checkpoint_integrity': True, 'resume_ready': True, 'data_fault_rate': 0.002, 'config_diff_keys': ['learning_rate', 'precision'], 'minimal_repro_ready': True}
metrics={'training_fault_evidence_coverage': 0.941, 'lifecycle_event_timeline': 0.941, 'rank_log_completeness': 0.941, 'loss_anomaly_diagnosis': 0.941, 'nan_inf_guard': 0.941, 'update_health_check': 0.941, 'hang_straggler_detection': 0.941, 'oom_phase_memory_evidence': 0.941, 'communication_bottleneck_diagnosis': 0.941, 'io_data_bottleneck_diagnosis': 0.941, 'checkpoint_integrity': 0.941, 'resume_continuity': 0.941, 'data_fault_isolation': 0.941, 'config_code_diff_readiness': 0.941, 'minimal_reproduction_readiness': 0.941, 'training_fault_diagnosis_gate': 0.941}
hard_blocker_count=16
failed_cases=['fault_evidence_missing_bad', 'lifecycle_timeline_bad', 'rank_log_missing_bad', 'loss_anomaly_diagnosis_bad', 'nan_inf_guard_bad', 'update_health_bad', 'hang_straggler_detection_bad', 'oom_phase_memory_bad', 'communication_bottleneck_bad', 'io_data_bottleneck_bad', 'checkpoint_integrity_bad', 'resume_continuity_bad', 'data_fault_isolation_bad', 'config_code_diff_bad', 'minimal_reproduction_bad', 'training_fault_gate_missing_bad']
failed_gates=['training_fault_evidence_coverage', 'lifecycle_event_timeline', 'rank_log_completeness', 'loss_anomaly_diagnosis', 'nan_inf_guard', 'update_health_check', 'hang_straggler_detection', 'oom_phase_memory_evidence', 'communication_bottleneck_diagnosis', 'io_data_bottleneck_diagnosis', 'checkpoint_integrity', 'resume_continuity', 'data_fault_isolation', 'config_code_diff_readiness', 'minimal_reproduction_readiness', 'training_fault_diagnosis_gate']
training_fault_diagnosis_gate_pass=False
```

这个 demo 的核心是把“训练故障排查”从经验性 checklist 变成可验收证据链：现象要有指标，指标要能关联 rank / step / batch / version / checkpoint，修复要能绑定最小复现和复盘，平台门禁要能阻断没有证据的结论。

## 45.22 常见误区

误区一：loss 异常就是模型结构问题。

也可能是数据、标签、tokenizer、学习率、precision 或 resume 状态问题。

误区二：NCCL timeout 的 rank 就是根因。

根因常常是另一个 rank 先崩溃或卡住。

误区三：GPU 利用率低就加 batch。

如果瓶颈是数据、通信或 checkpoint，加 batch 可能无效甚至 OOM。

误区四：OOM 只靠减 batch。

还可以优化 sequence length、activation checkpointing、ZeRO/FSDP、precision 和 eval 行为。

误区五：checkpoint 能加载就说明恢复正确。

完整恢复还要 optimizer、scheduler、random state、dataloader state 和 global step。

## 45.23 面试常见追问

问题一：训练 loss NaN 怎么排查？

可以回答：先定位 NaN 出现 step，检查数据 batch、学习率、gradient norm、precision、loss scale、logits/grad 是否 inf，再尝试降学习率、gradient clipping、BF16、保存异常 batch 做最小复现。

问题二：分布式训练 hang 怎么排查？

可以回答：看所有 rank 最后一条日志、step 是否推进、GPU 利用率、NCCL 日志、dataloader 状态、节点事件，重点找最早异常 rank，而不是最后 timeout 的 rank。

问题三：GPU 利用率低可能是什么原因？

可以回答：可能是数据读取慢、CPU preprocessing、batch 太小、通信等待、checkpoint/eval 太频繁、负载不均衡或 kernel 效率低，需要拆分 step time。

问题四：checkpoint resume 后 loss 跳变怎么办？

可以回答：检查是否恢复了 optimizer、scheduler、global step、random state 和 dataloader state，确认 config、数据版本和学习率曲线一致。

## 45.24 小练习

1. Loss NaN 常见原因有哪些？
2. 为什么小数据 overfit 测试能帮助排查训练问题？
3. 分布式训练 hang 时为什么要看所有 rank 日志？
4. GPU OOM 发生在 forward 和 optimizer step，排查方向有什么不同？
5. 通信慢需要看哪些指标？
6. I/O 慢为什么会导致 GPU 利用率低？
7. 完整 checkpoint resume 需要恢复哪些状态？
8. 如何设计 TrainingJob 故障排查页面？

## 45.25 本章小结

本章讲了训练故障定位。

你需要记住：

1. 训练故障要按任务事件、日志、指标、资源、数据、通信、checkpoint 和配置逐层排查。
2. Loss 异常可能来自数据、标签、tokenizer、学习率、precision、梯度、代码或 resume 状态。
3. Hang 通常需要看所有 rank 日志和分布式通信状态，不能只看最后报错。
4. OOM 要区分发生阶段，优化手段不只有减 batch。
5. GPU 利用率低常常是数据、通信、checkpoint 或调度问题。
6. 完整可观测性和实验追踪能显著降低训练故障定位成本。

下一章我们会讲推理故障定位：TTFT、TPOT、吞吐下降和错误率上升。
