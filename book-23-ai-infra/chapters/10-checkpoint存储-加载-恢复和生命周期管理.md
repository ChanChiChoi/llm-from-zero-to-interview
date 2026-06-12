# 第 10 章 Checkpoint 存储、加载、恢复和生命周期管理

上一章讲了存储体系。本章专门讲 checkpoint。

Checkpoint 是大模型训练平台里最关键、也最容易被低估的能力之一。它看起来只是“保存一下模型”，但在真实大规模训练中，checkpoint 决定了训练失败后能不能恢复、恢复到哪里、恢复是否正确、存储成本是否可控、实验是否可复现。

先记住一句话：

> Checkpoint 不是一个模型文件，而是训练状态的可恢复快照；它的价值不在于保存成功，而在于能可靠、快速、正确地恢复。

## 10.0 本讲资料边界与第二轮精修口径

本讲第二轮精修时，资料口径按“大模型训练平台 checkpoint 生命周期的稳定抽象”处理，而不是绑定某个框架、某个对象存储厂商或某套内部平台实现。分布式 checkpoint 部分参考 PyTorch Distributed Checkpoint 对分布式 state_dict 保存、加载和异步保存的抽象；工程接口部分参考 DeepSpeed 对训练 checkpoint 保存、加载和 ZeRO checkpoint 的接口边界；对象存储部分参考 Amazon S3 对强一致读取、对象生命周期和保留策略的通用口径；平台治理部分沿用上一章存储体系的 manifest、checksum、commit、权限、审计和成本分层。

需要注意三点：

1. 本章讲训练状态 checkpoint 的保存、加载、恢复、一致性和生命周期，不把下一章容器、镜像、驱动和 CUDA 依赖管理提前展开。
2. 模型权重 checkpoint、训练状态 checkpoint、发布 artifact 和归档对象不是同一个概念，面试中要先说明用途、状态内容、恢复要求和权限边界。
3. checkpoint 系统不能只证明“写入成功”，还要证明分片完整、manifest 可解释、commit 语义清楚、恢复演练通过、保留策略可控、成本和权限可审计。

## 10.1 为什么 checkpoint 这么重要

大模型训练时间长、成本高、故障概率高。

一个预训练任务可能运行数天甚至数周，使用成百上千张 GPU。期间可能遇到：

1. GPU 故障。
2. 节点宕机。
3. 网络 timeout。
4. 存储异常。
5. 训练进程 crash。
6. 驱动或容器问题。
7. 调度抢占。
8. 人为停止任务。

如果没有 checkpoint，任务失败就要从头开始，成本巨大。

Checkpoint 的作用：

1. 故障后恢复训练。
2. 支持抢占式调度。
3. 保存重要训练阶段。
4. 支持模型评估和发布。
5. 支持实验复现。
6. 支持回滚。
7. 支持训练过程分析。

所以 checkpoint 是训练可靠性的基础。

## 10.2 Checkpoint 里到底有什么

很多人以为 checkpoint 只包含模型参数。

真实训练 checkpoint 通常包含：

1. 模型参数。
2. 优化器状态。
3. 学习率 scheduler 状态。
4. 当前 global step。
5. epoch 或样本位置。
6. random seed。
7. dataloader 状态。
8. gradient scaler 状态。
9. 分布式训练状态。
10. 并行策略 metadata。
11. tokenizer 和 config 引用。
12. 训练代码版本。
13. 数据版本。
14. 训练参数。

如果只保存模型参数，通常只能用于推理或继续微调，不一定能精确恢复训练。

精确恢复训练需要保存完整训练状态。

## 10.3 模型权重 checkpoint 和训练状态 checkpoint

要区分两类 checkpoint。

### 10.3.1 模型权重 checkpoint

它主要包含模型参数。

用途：

1. 推理部署。
2. 模型评估。
3. 下游微调。
4. 模型发布。

它通常不包含完整优化器状态。

### 10.3.2 训练状态 checkpoint

它包含恢复训练需要的完整状态。

用途：

1. 断点续训。
2. 故障恢复。
3. 抢占恢复。
4. 精确复现实验。

它通常比模型权重 checkpoint 大很多，因为优化器状态可能非常大。

面试中要强调：用于部署的 checkpoint 和用于恢复训练的 checkpoint 不是一回事。

## 10.4 为什么 checkpoint 会很大

大模型 checkpoint 大，主要因为：

1. 参数量大。
2. 优化器状态大。
3. 分布式分片多。
4. 多精度状态并存。
5. 需要保存 metadata。

以 Adam 为例，每个参数通常有：

1. 参数本身。
2. 一阶动量。
3. 二阶动量。
4. 可能还有 FP32 master weight。

所以训练状态 checkpoint 可能是模型权重大小的数倍。

这也是为什么 checkpoint 会给存储带来巨大压力。

## 10.5 Checkpoint 保存频率

保存频率是一个 trade-off。

保存太频繁：

1. 训练被 I/O 打断。
2. 存储吞吐压力大。
3. checkpoint 数量爆炸。
4. 成本高。

保存太少：

1. 故障后回退太多 step。
2. 浪费训练算力。
3. 抢占恢复代价高。
4. 丢失重要中间版本。

选择保存频率时要考虑：

1. 训练任务规模。
2. 平均故障间隔。
3. checkpoint 保存耗时。
4. 存储成本。
5. 可接受的恢复点目标。
6. 评估和发布需求。

可以用一个简单问题判断：如果现在任务失败，最多能接受损失多少训练时间？

## 10.6 同步 checkpoint 和异步 checkpoint

### 10.6.1 同步 checkpoint

同步 checkpoint 是训练暂停，等待 checkpoint 写完，再继续训练。

优点：

1. 实现简单。
2. 一致性容易保证。
3. 恢复逻辑清楚。

缺点：

1. 训练会被阻塞。
2. 大模型保存耗时长。
3. GPU 可能等待 I/O。
4. 周期性 step time 尖刺明显。

### 10.6.2 异步 checkpoint

异步 checkpoint 是训练继续跑，后台线程或独立进程负责写 checkpoint。

优点：

1. 减少训练阻塞。
2. 提高 GPU 利用率。
3. 降低 step time 尖刺。

缺点：

1. 实现复杂。
2. 要处理内存占用。
3. 要保证写入一致性。
4. 失败状态更复杂。
5. 可能和训练争抢 I/O 或内存带宽。

异步 checkpoint 不是简单开线程写文件，而是一个一致性和资源隔离问题。

## 10.7 分片 checkpoint

大模型通常不能让一个 rank 写一个巨大文件。

分片 checkpoint 会把状态拆成多个 shard。

常见分片方式：

1. 按 rank 分片。
2. 按 layer 分片。
3. 按参数 shard 分片。
4. 按 optimizer state 分片。
5. 按 tensor parallel / data parallel 维度分片。

分片 checkpoint 的好处：

1. 单文件更小。
2. 多 rank 并行写入。
3. 恢复时可以并行读取。
4. 更适合分布式训练状态。

挑战：

1. metadata 更复杂。
2. 需要保证所有 shard 完整。
3. rank 数变化时恢复更复杂。
4. 文件数量多可能造成 metadata 压力。

因此分片 checkpoint 必须配套 manifest。

## 10.8 Manifest 和 metadata

Manifest 是 checkpoint 的目录和说明书。

它应该记录：

1. checkpoint_id。
2. global_step。
3. created_at。
4. model_version。
5. training_config。
6. data_version。
7. parallelism_config。
8. shard 列表。
9. 每个 shard 的大小。
10. checksum。
11. 保存状态。
12. 是否可恢复。

示例：

```json
{
  "checkpoint_id": "ckpt_00012000",
  "global_step": 12000,
  "status": "committed",
  "parallelism": {
    "tensor_parallel": 8,
    "pipeline_parallel": 4,
    "data_parallel": 16
  },
  "shards": [
    {
      "name": "rank_000_state.pt",
      "size_bytes": 10737418240,
      "checksum": "sha256:..."
    }
  ]
}
```

没有 manifest，恢复系统就不知道哪些文件属于同一个一致版本。

## 10.9 Commit 语义

Checkpoint 写入必须避免“半成品被当成可用版本”。

常见做法：

1. 写入临时目录。
2. 所有 shard 写完。
3. 校验 checksum。
4. 写 manifest。
5. 原子标记 committed。
6. 清理旧临时文件。

简化流程：

```text
writing -> validating -> committed
                  -> failed
```

恢复时只允许读取 committed checkpoint。

如果读取 writing 状态的 checkpoint，很容易恢复失败或训练状态不一致。

## 10.10 Checkpoint 加载

加载 checkpoint 不是简单读文件。

需要做：

1. 找到目标 checkpoint。
2. 检查 manifest。
3. 校验 shard 完整性。
4. 加载模型参数。
5. 加载优化器状态。
6. 加载 scheduler 状态。
7. 恢复 random seed。
8. 恢复 dataloader 位置。
9. 恢复分布式状态。
10. 验证配置兼容性。

常见加载失败原因：

1. shard 缺失。
2. checksum 不匹配。
3. 模型结构变化。
4. 并行策略变化。
5. 优化器配置变化。
6. 代码版本不兼容。
7. tokenizer 或 config 不匹配。

所以 checkpoint 加载需要严格校验，不应该静默忽略错误。

## 10.11 恢复训练的一致性

恢复训练要回答一个问题：恢复后是否和没中断时尽可能一致？

影响一致性的因素：

1. 模型参数。
2. 优化器状态。
3. scheduler 状态。
4. random seed。
5. dataloader 顺序。
6. gradient scaler。
7. 分布式 rank 映射。
8. dropout 随机性。
9. 数据 shuffle 状态。
10. 代码和依赖版本。

如果只恢复模型参数，不恢复 optimizer 和 scheduler，训练曲线可能发生变化。

如果不恢复 dataloader 状态，可能重复或跳过数据。

如果不恢复 random seed，严格复现会失败。

## 10.12 Rank 数变化和弹性恢复

有时恢复训练时 GPU 数量可能变化。

例如原来 128 卡训练，恢复时只有 64 卡。

这会带来问题：

1. checkpoint shard 和 rank 数绑定。
2. tensor parallel 维度可能不兼容。
3. pipeline stage 划分可能变化。
4. optimizer state 需要重新分片。
5. global batch size 可能变化。

弹性恢复需要支持 checkpoint resharding。

Resharding 是把原有 checkpoint 分片重新组织成新的并行布局。

这很有价值，但实现复杂。

面试里可以说：基础版本先要求相同并行配置恢复；高级版本支持 resharding 和弹性训练。

## 10.13 Checkpoint 与抢占式调度

抢占式调度可以提高集群利用率。

低优先级训练任务使用空闲 GPU，高优先级任务来了以后被抢占。

要支持抢占，必须有可靠 checkpoint。

抢占前最好：

1. 给任务发送 preemption notice。
2. 任务保存 checkpoint。
3. 标记任务可恢复。
4. 释放资源。
5. 后续重新排队恢复。

如果没有 checkpoint，抢占会浪费大量训练进度。

所以 checkpoint 是弹性调度和资源利用率优化的基础能力。

## 10.14 Checkpoint 生命周期管理

Checkpoint 不能无限保留。

生命周期管理要回答：

1. 保存哪些 checkpoint？
2. 保留多久？
3. 哪些是关键 checkpoint？
4. 哪些可以删除？
5. 哪些要归档到低成本存储？
6. 删除前是否需要审批？

常见策略：

1. 保留最近 N 个 checkpoint。
2. 每隔固定步数保留一个里程碑版本。
3. 最优 eval 分数 checkpoint 永久保留或长期保留。
4. 最终发布模型保留。
5. 失败或临时 checkpoint 定期清理。
6. 冷 checkpoint 迁移到低成本存储。

一个例子：

```text
最近 5 个 checkpoint：保留在高性能存储。
每 10k step checkpoint：保留 30 天。
最佳 eval checkpoint：长期保留。
发布版本：长期保留并加权限保护。
临时失败 checkpoint：7 天后删除。
```

## 10.15 Checkpoint 成本治理

Checkpoint 成本容易失控。

原因：

1. 单个 checkpoint 很大。
2. 保存频率高。
3. 多个实验并行。
4. 优化器状态巨大。
5. 没有删除策略。
6. 多份重复保存。
7. 日志和 artifact 一起膨胀。

成本治理方法：

1. 设置默认保留策略。
2. 区分训练恢复 checkpoint 和发布权重。
3. 冷热分层存储。
4. 压缩或去重。
5. 定期生成成本报告。
6. 按团队或项目归因。
7. 对超预算实验告警。

成本治理不是只删文件，而是建立可解释的保留规则。

## 10.16 Checkpoint 监控指标

训练平台应该监控 checkpoint。

关键指标：

1. save_duration。
2. load_duration。
3. checkpoint_size。
4. checkpoint_frequency。
5. save_failure_rate。
6. load_failure_rate。
7. committed_checkpoint_count。
8. temporary_checkpoint_count。
9. storage_write_throughput。
10. storage_read_throughput。
11. restore_success_rate。
12. time_since_last_checkpoint。

告警场景：

1. 长时间没有成功 checkpoint。
2. checkpoint 保存时间突然变长。
3. checkpoint 失败率上升。
4. 存储空间接近上限。
5. 恢复失败。
6. 临时 checkpoint 堆积。

如果训练任务已经跑了 12 小时但没有成功 checkpoint，这是严重风险。

## 10.17 常见故障和排查

### 10.17.1 Checkpoint 保存很慢

可能原因：

1. checkpoint 太大。
2. 所有 rank 同时写。
3. 存储吞吐不足。
4. 小文件太多。
5. metadata 服务瓶颈。
6. 网络到存储路径拥塞。
7. 同一时间多个任务保存。

排查：

1. 看保存耗时分布。
2. 看每个 shard 大小。
3. 看存储吞吐。
4. 看 metadata 延迟。
5. 看网络吞吐。
6. 看是否周期性和其他任务冲突。

### 10.17.2 Checkpoint 恢复失败

可能原因：

1. shard 缺失。
2. manifest 不完整。
3. checksum 不匹配。
4. 代码版本变化。
5. 并行配置变化。
6. optimizer 状态不兼容。
7. tokenizer / config 不匹配。

排查：

1. 只读取 committed checkpoint。
2. 校验 manifest。
3. 校验 checksum。
4. 检查版本兼容性。
5. 在小规模环境做恢复测试。

### 10.17.3 恢复后 loss 异常

可能原因：

1. optimizer 状态没恢复。
2. scheduler 状态没恢复。
3. random seed 没恢复。
4. dataloader 状态错。
5. mixed precision scaler 状态丢失。
6. 数据重复或跳过。
7. 代码逻辑变化。

排查：

1. 比较恢复前后 loss。
2. 检查 global step。
3. 检查学习率。
4. 检查数据位置。
5. 检查 optimizer state。
6. 用短任务做中断恢复测试。

## 10.18 Checkpoint 恢复测试

很多团队只测试保存，不测试恢复。

这是危险的。

平台应该定期做恢复测试：

1. 启动一个训练任务。
2. 训练若干 step。
3. 保存 checkpoint。
4. 模拟中断。
5. 从 checkpoint 恢复。
6. 对比 loss、step、学习率和数据位置。
7. 验证继续训练正常。

高级测试还可以包括：

1. rank 故障恢复。
2. 节点故障恢复。
3. checkpoint shard 缺失测试。
4. 并行策略变化恢复。
5. 存储短暂失败恢复。

Checkpoint 能不能用，必须通过恢复测试证明。

## 10.19 Checkpoint 生命周期审计指标与最小 demo

可以把 checkpoint 生命周期从“保存文件”写成一组可审计对象。

一个训练状态 checkpoint 样本可以抽象为：

```math
c_i=(m_i,o_i,p_i,s_i,w_i,d_i,r_i,a_i,v_i,k_i,l_i,g_i,z_i)
```

其中，`m_i` 是 manifest，`o_i` 是训练状态对象，`p_i` 是并行配置，`s_i` 是 shard 布局，`w_i` 是写入路径，`d_i` 是校验摘要，`r_i` 是恢复测试，`a_i` 是异步保存策略，`v_i` 是版本和血缘，`k_i` 是保留与成本策略，`l_i` 是权限和审计日志，`g_i` 是监控与 SLO，`z_i` 是最终门禁结果。

分片大小可以先用：

```math
S_{\mathrm{shard}}=\frac{S_{\mathrm{ckpt}}}{N_{\mathrm{shard}}}
```

其中，`S_ckpt` 是总 checkpoint 大小，`N_shard` 是 shard 数。真实系统还要考虑 tensor 大小、rank 映射、对象存储请求数、metadata 压力和恢复并发度。

保存耗时可以拆成：

```math
T_{\mathrm{save}}=\frac{S_{\mathrm{ckpt}}}{\eta_w B_{\mathrm{write}}}+T_{\mathrm{sync}}+T_{\mathrm{meta}}+T_{\mathrm{commit}}
```

其中，`B_write` 是写入带宽，`\eta_w` 是有效带宽系数，`T_sync` 是冻结或拷贝一致快照的时间，`T_meta` 是 manifest / metadata 写入开销，`T_commit` 是 checksum 校验和 committed 标记开销。

如果使用异步 checkpoint，训练可见阻塞不是总写入时间，而更接近：

```math
T_{\mathrm{visible}}=\max(0,T_{\mathrm{save}}(1-\gamma_{\mathrm{overlap}}))
```

其中，`\gamma_overlap` 是写入与训练重叠比例。这个比例不能只靠理论估计，必须从 step time 尖刺、I/O 争抢和后台队列积压中实测。

恢复耗时可以写成：

```math
T_{\mathrm{restore}}=\frac{S_{\mathrm{ckpt}}}{\eta_r B_{\mathrm{read}}}+T_{\mathrm{validate}}+T_{\mathrm{rebuild}}+T_{\mathrm{requeue}}
```

其中，`B_read` 是读取带宽，`\eta_r` 是有效读取系数，`T_validate` 是 checksum / manifest 校验时间，`T_rebuild` 是重建并行状态或 reshard 的时间，`T_requeue` 是重新排队、拉起容器和恢复训练进程的时间。

保存间隔决定最大丢失进度：

```math
W_{\mathrm{lost}}\le I_{\mathrm{ckpt}}
```

其中，`I_ckpt` 是两次成功 checkpoint 之间的时间间隔。抢占式训练、长周期预训练和不稳定集群要把 `I_ckpt` 作为平台 SLO，而不是只看“平均每天保存几次”。

生命周期成本可以写成：

```math
K_{\mathrm{ckpt}}=\sum_{a=1}^{A} S_a P_a T_a+K_{\mathrm{request}}+K_{\mathrm{egress}}+K_{\mathrm{ops}}
```

其中，`S_a` 是第 `a` 类 checkpoint 的容量，`P_a` 是对应存储层级单价，`T_a` 是保留时长，`K_request` 是对象请求成本，`K_egress` 是跨区或出站成本，`K_ops` 是校验、复制、扫描、人工审批和运维成本。

最后，可以把 checkpoint 生命周期门禁写成：

```math
G_{\mathrm{ckpt}}=\mathbf{1}\left[\min_j C_j\ge \tau_j \land T_{\mathrm{restore}}\le \tau_{\mathrm{restore}} \land W_{\mathrm{lost}}\le \omega \land P_0=0\right]
```

其中，`C_j` 是第 `j` 个 checkpoint 审计指标覆盖率，`\tau_j` 是覆盖率阈值，`\tau_restore` 是恢复 SLO，`\omega` 是最大可接受丢失进度，`P_0` 是 P0 级风险数量。

下面这个 0 依赖 demo 演示如何把 checkpoint 生命周期写成审计规则。它故意构造 1 个完整样本和 16 个坏样本，让每个关键维度各失败一次。

```python
import copy


METRICS = [
    "checkpoint_object_completeness",
    "shard_layout_fit",
    "async_save_overlap",
    "write_bandwidth_fit",
    "metadata_commit_integrity",
    "checksum_manifest_validation",
    "restore_replay_readiness",
    "dataloader_rng_state_capture",
    "distributed_rank_state_capture",
    "retention_policy_fit",
    "lifecycle_cost_governance",
    "cross_region_replication_fit",
    "security_access_control",
    "resume_slo_tracking",
    "failure_drill_coverage",
    "checkpoint_lifecycle_gate",
]


def shard_size_gib(total_gib, shard_count):
    return total_gib / shard_count


def transfer_time_s(gib, bandwidth_gib_s, efficiency):
    return gib / (bandwidth_gib_s * efficiency)


def save_time_s(size_gib, bandwidth_gib_s, efficiency, sync_s, metadata_s, commit_s):
    return transfer_time_s(size_gib, bandwidth_gib_s, efficiency) + sync_s + metadata_s + commit_s


def visible_save_overhead_s(total_save_s, overlap_ratio):
    return max(0.0, total_save_s * (1.0 - overlap_ratio))


def restore_time_s(size_gib, bandwidth_gib_s, efficiency, validate_s, rebuild_s, requeue_s):
    return transfer_time_s(size_gib, bandwidth_gib_s, efficiency) + validate_s + rebuild_s + requeue_s


def monthly_checkpoint_cost_usd(hot_tib, cold_tib, hot_per_tib, cold_per_tib, request_cost):
    return hot_tib * hot_per_tib + cold_tib * cold_per_tib + request_cost


def build_checkpoint_cases():
    complete = {
        "name": "complete",
        "object": {
            "type": "training_state",
            "model_state": True,
            "optimizer_state": True,
            "scheduler_state": True,
            "global_step": True,
            "scaler_state": True,
            "code_data_version": True,
        },
        "shard_layout": {
            "total_gib": 3200,
            "expected_shards": 64,
            "actual_shards": 64,
            "manifest_shards": 64,
            "max_shard_gib": 64,
            "parallelism_match": True,
        },
        "async_save": {
            "mode": "async",
            "snapshot_frozen": True,
            "overlap_ratio": 0.82,
            "io_isolated": True,
            "pending_limit": 2,
        },
        "write_path": {
            "write_gib_s": 80,
            "efficiency": 0.75,
            "sync_s": 5,
            "metadata_s": 20,
            "commit_s": 5,
            "save_slo_s": 120,
            "measured": True,
        },
        "metadata": {
            "manifest": True,
            "atomic_commit": True,
            "status": "committed",
            "temp_prefix": True,
        },
        "checksum": {
            "all_shards": True,
            "algorithm": "sha256",
            "validate_on_load": True,
        },
        "restore": {
            "restore_tested": True,
            "read_gib_s": 100,
            "efficiency": 0.75,
            "validate_s": 20,
            "rebuild_s": 10,
            "requeue_s": 20,
            "restore_slo_s": 120,
        },
        "rng_state": {
            "python": True,
            "numpy": True,
            "torch": True,
            "cuda": True,
            "sampler": True,
        },
        "rank_state": {
            "tensor_parallel": True,
            "pipeline_parallel": True,
            "data_parallel": True,
            "zero_partition": True,
            "grad_accum_step": True,
        },
        "retention": {
            "recent_n": 5,
            "milestone_days": 30,
            "best_eval_keep": True,
            "release_keep": True,
            "temp_failed_days": 7,
            "keep_all": False,
        },
        "cost": {
            "hot_tib": 120,
            "cold_tib": 480,
            "hot_per_tib": 23,
            "cold_per_tib": 4,
            "request_cost": 320,
            "monthly_budget_usd": 6000,
            "archive_cold": True,
            "budget_owner": True,
        },
        "replication": {
            "object_store": True,
            "cross_region_release": True,
            "restore_copy_tested": True,
        },
        "security": {
            "iam": True,
            "encryption": True,
            "audit_log": True,
            "tenant_isolation": True,
        },
        "slo": {
            "tracked": True,
            "save_p95_s": 90,
            "save_slo_s": 120,
            "restore_p95_s": 100,
            "restore_slo_s": 120,
            "time_since_last_ckpt_min": 12,
            "max_interval_min": 15,
        },
        "drill": {
            "automated": True,
            "restore_test_age_days": 7,
            "failure_modes": 4,
        },
        "gate": {"enabled": True},
    }

    def bad_case(name, mutator):
        case = copy.deepcopy(complete)
        case["name"] = name
        mutator(case)
        return case

    bad_cases = [
        bad_case("weights_only_checkpoint_bad", lambda c: c["object"].update({"type": "model_weights", "optimizer_state": False})),
        bad_case("single_huge_file_bad", lambda c: c["shard_layout"].update({"actual_shards": 1, "manifest_shards": 1})),
        bad_case("blocking_save_bad", lambda c: c["async_save"].update({"mode": "sync", "overlap_ratio": 0.0})),
        bad_case("write_bandwidth_unknown_bad", lambda c: c["write_path"].update({"measured": False})),
        bad_case("metadata_marked_before_commit_bad", lambda c: c["metadata"].update({"atomic_commit": False})),
        bad_case("checksum_missing_bad", lambda c: c["checksum"].update({"all_shards": False})),
        bad_case("restore_never_tested_bad", lambda c: c["restore"].update({"restore_tested": False})),
        bad_case("rng_state_missing_bad", lambda c: c["rng_state"].update({"torch": False})),
        bad_case("rank_state_missing_bad", lambda c: c["rank_state"].update({"zero_partition": False})),
        bad_case("keep_everything_forever_bad", lambda c: c["retention"].update({"keep_all": True, "temp_failed_days": 365})),
        bad_case("cold_archive_never_used_bad", lambda c: c["cost"].update({"archive_cold": False})),
        bad_case("no_replication_bad", lambda c: c["replication"].update({"cross_region_release": False})),
        bad_case("open_bucket_bad", lambda c: c["security"].update({"iam": False})),
        bad_case("resume_slo_missing_bad", lambda c: c["slo"].update({"tracked": False})),
        bad_case("failure_drill_missing_bad", lambda c: c["drill"].update({"automated": False})),
        bad_case("checkpoint_gate_missing_bad", lambda c: c["gate"].update({"enabled": False})),
    ]
    return [complete] + bad_cases


def check_object(case):
    obj = case["object"]
    required = ["model_state", "optimizer_state", "scheduler_state", "global_step", "scaler_state", "code_data_version"]
    return obj["type"] == "training_state" and all(obj[k] for k in required)


def check_shard_layout(case):
    layout = case["shard_layout"]
    expected = layout["expected_shards"]
    shard_gib = shard_size_gib(layout["total_gib"], max(layout["actual_shards"], 1))
    return (
        layout["actual_shards"] == expected
        and layout["manifest_shards"] == expected
        and shard_gib <= layout["max_shard_gib"]
        and layout["parallelism_match"]
    )


def check_async(case):
    async_save = case["async_save"]
    return (
        async_save["mode"] == "async"
        and async_save["snapshot_frozen"]
        and async_save["overlap_ratio"] >= 0.7
        and async_save["io_isolated"]
        and async_save["pending_limit"] <= 2
    )


def check_write(case):
    path = case["write_path"]
    total_s = save_time_s(
        case["shard_layout"]["total_gib"],
        path["write_gib_s"],
        path["efficiency"],
        path["sync_s"],
        path["metadata_s"],
        path["commit_s"],
    )
    return path["measured"] and total_s <= path["save_slo_s"]


def check_metadata(case):
    meta = case["metadata"]
    return meta["manifest"] and meta["atomic_commit"] and meta["status"] == "committed" and meta["temp_prefix"]


def check_checksum(case):
    checksum = case["checksum"]
    return checksum["all_shards"] and checksum["algorithm"] == "sha256" and checksum["validate_on_load"]


def check_restore(case):
    restore = case["restore"]
    total_s = restore_time_s(
        case["shard_layout"]["total_gib"],
        restore["read_gib_s"],
        restore["efficiency"],
        restore["validate_s"],
        restore["rebuild_s"],
        restore["requeue_s"],
    )
    return restore["restore_tested"] and total_s <= restore["restore_slo_s"]


def check_rng(case):
    return all(case["rng_state"].values())


def check_rank_state(case):
    return all(case["rank_state"].values())


def check_retention(case):
    retention = case["retention"]
    return (
        retention["recent_n"] >= 3
        and retention["milestone_days"] >= 7
        and retention["best_eval_keep"]
        and retention["release_keep"]
        and 0 < retention["temp_failed_days"] <= 14
        and not retention["keep_all"]
    )


def check_cost(case):
    cost = case["cost"]
    monthly = monthly_checkpoint_cost_usd(
        cost["hot_tib"],
        cost["cold_tib"],
        cost["hot_per_tib"],
        cost["cold_per_tib"],
        cost["request_cost"],
    )
    return cost["archive_cold"] and cost["budget_owner"] and monthly <= cost["monthly_budget_usd"]


def check_replication(case):
    repl = case["replication"]
    return repl["object_store"] and repl["cross_region_release"] and repl["restore_copy_tested"]


def check_security(case):
    return all(case["security"].values())


def check_slo(case):
    slo = case["slo"]
    return (
        slo["tracked"]
        and slo["save_p95_s"] <= slo["save_slo_s"]
        and slo["restore_p95_s"] <= slo["restore_slo_s"]
        and slo["time_since_last_ckpt_min"] <= slo["max_interval_min"]
    )


def check_drill(case):
    drill = case["drill"]
    return drill["automated"] and drill["restore_test_age_days"] <= 14 and drill["failure_modes"] >= 3


def check_gate(case):
    return case["gate"]["enabled"]


CHECKS = {
    "checkpoint_object_completeness": check_object,
    "shard_layout_fit": check_shard_layout,
    "async_save_overlap": check_async,
    "write_bandwidth_fit": check_write,
    "metadata_commit_integrity": check_metadata,
    "checksum_manifest_validation": check_checksum,
    "restore_replay_readiness": check_restore,
    "dataloader_rng_state_capture": check_rng,
    "distributed_rank_state_capture": check_rank_state,
    "retention_policy_fit": check_retention,
    "lifecycle_cost_governance": check_cost,
    "cross_region_replication_fit": check_replication,
    "security_access_control": check_security,
    "resume_slo_tracking": check_slo,
    "failure_drill_coverage": check_drill,
    "checkpoint_lifecycle_gate": check_gate,
}


def audit_checkpoint_lifecycle(cases):
    case_failures = {}
    for case in cases:
        failures = [name for name, check in CHECKS.items() if not check(case)]
        case_failures[case["name"]] = failures

    metrics = {}
    for name, check in CHECKS.items():
        metrics[name] = round(sum(int(check(case)) for case in cases) / len(cases), 3)

    failed_cases = [name for name, failures in case_failures.items() if failures]
    return {
        "metrics": metrics,
        "hard_blocker_count": len(failed_cases),
        "failed_cases": failed_cases,
        "checkpoint_gate_pass": not failed_cases and min(metrics.values()) >= 0.95,
    }


cases = build_checkpoint_cases()
case_by_name = {case["name"]: case for case in cases}
complete = case_by_name["complete"]
write_path = complete["write_path"]
restore_path = complete["restore"]
total_save_s = save_time_s(
    complete["shard_layout"]["total_gib"],
    write_path["write_gib_s"],
    write_path["efficiency"],
    write_path["sync_s"],
    write_path["metadata_s"],
    write_path["commit_s"],
)
checkpoint_examples = {
    "shard_size_gib": round(shard_size_gib(3200, 64), 1),
    "save_3200gib_s": round(total_save_s, 1),
    "async_visible_overhead_s": round(visible_save_overhead_s(total_save_s, complete["async_save"]["overlap_ratio"]), 1),
    "restore_3200gib_s": round(restore_time_s(3200, restore_path["read_gib_s"], restore_path["efficiency"], 20, 10, 20), 1),
    "max_lost_work_min": complete["slo"]["max_interval_min"],
    "monthly_retention_cost_usd": monthly_checkpoint_cost_usd(120, 480, 23, 4, 320),
}

smoke = {
    "complete_case_passes": all(check(complete) for check in CHECKS.values()),
    "caught_weights_only": not check_object(case_by_name["weights_only_checkpoint_bad"]),
    "caught_single_shard": not check_shard_layout(case_by_name["single_huge_file_bad"]),
    "caught_commit_gap": not check_metadata(case_by_name["metadata_marked_before_commit_bad"]),
    "caught_restore_gap": not check_restore(case_by_name["restore_never_tested_bad"]),
    "caught_cost_gap": not check_cost(case_by_name["cold_archive_never_used_bad"]),
}

audit = audit_checkpoint_lifecycle(cases)
print(f"checkpoint_examples={checkpoint_examples}")
print(f"smoke={smoke}")
print(f"metrics={audit['metrics']}")
print(f"hard_blocker_count={audit['hard_blocker_count']}")
print(f"failed_cases={audit['failed_cases']}")
print(f"checkpoint_gate_pass={audit['checkpoint_gate_pass']}")
```

一组典型输出是：

```text
checkpoint_examples={'shard_size_gib': 50.0, 'save_3200gib_s': 83.3, 'async_visible_overhead_s': 15.0, 'restore_3200gib_s': 92.7, 'max_lost_work_min': 15, 'monthly_retention_cost_usd': 5000}
smoke={'complete_case_passes': True, 'caught_weights_only': True, 'caught_single_shard': True, 'caught_commit_gap': True, 'caught_restore_gap': True, 'caught_cost_gap': True}
metrics={'checkpoint_object_completeness': 0.941, 'shard_layout_fit': 0.941, 'async_save_overlap': 0.941, 'write_bandwidth_fit': 0.941, 'metadata_commit_integrity': 0.941, 'checksum_manifest_validation': 0.941, 'restore_replay_readiness': 0.941, 'dataloader_rng_state_capture': 0.941, 'distributed_rank_state_capture': 0.941, 'retention_policy_fit': 0.941, 'lifecycle_cost_governance': 0.941, 'cross_region_replication_fit': 0.941, 'security_access_control': 0.941, 'resume_slo_tracking': 0.941, 'failure_drill_coverage': 0.941, 'checkpoint_lifecycle_gate': 0.941}
hard_blocker_count=16
failed_cases=['weights_only_checkpoint_bad', 'single_huge_file_bad', 'blocking_save_bad', 'write_bandwidth_unknown_bad', 'metadata_marked_before_commit_bad', 'checksum_missing_bad', 'restore_never_tested_bad', 'rng_state_missing_bad', 'rank_state_missing_bad', 'keep_everything_forever_bad', 'cold_archive_never_used_bad', 'no_replication_bad', 'open_bucket_bad', 'resume_slo_missing_bad', 'failure_drill_missing_bad', 'checkpoint_gate_missing_bad']
checkpoint_gate_pass=False
```

这个 demo 的重点是把 checkpoint 生命周期拆成可验证证据链：状态对象不能只存模型权重，分片布局要和 manifest 一致，异步保存要证明快照一致且训练可见阻塞下降，写入带宽和恢复耗时要实测，commit 不能先于 shard 校验，RNG / dataloader / rank state 要能恢复，保留策略不能无限膨胀，冷归档、跨区复制、权限、SLO 和失败演练都要能被审计。

## 10.20 面试中如何回答 checkpoint 系统设计

如果面试官问：

```text
如何设计大模型训练平台的 checkpoint 系统？
```

可以这样回答：

```text
我会先区分两类 checkpoint：用于推理和发布的模型权重 checkpoint，以及用于断点续训的训练状态 checkpoint。后者需要保存模型参数、优化器状态、scheduler、global step、随机种子、dataloader 状态和分布式并行 metadata。

存储上采用分片 checkpoint，每个 rank 或参数 shard 并行写入，并用 manifest 记录 shard、checksum、并行配置、数据版本和训练配置。写入流程使用临时目录和 commit 语义，只有完整校验后的 checkpoint 才标记为 committed，恢复时只读取 committed 版本。

为了降低训练阻塞，可以支持异步 checkpoint，但要处理一致性、内存占用和 I/O 干扰。平台需要监控 save/load duration、失败率、checkpoint size、time since last checkpoint，并定期做恢复测试。

生命周期上保留最近 N 个 checkpoint、关键 milestone、最佳 eval 和发布版本，临时和失败 checkpoint 自动清理，冷版本迁移到低成本存储，并按项目做成本归因。
```

## 10.21 常见误区

误区一：checkpoint 就是模型权重。

训练恢复需要完整训练状态，不只是模型参数。

误区二：保存成功就等于可恢复。

必须验证 manifest、checksum、配置兼容性，并做恢复测试。

误区三：checkpoint 越频繁越好。

频繁保存会增加 I/O、阻塞训练和存储成本，需要权衡恢复点目标和性能。

误区四：异步 checkpoint 一定更好。

异步保存能减少阻塞，但会带来一致性、内存占用和失败状态复杂度。

误区五：checkpoint 可以无限保留。

大模型 checkpoint 成本极高，必须做生命周期和成本治理。

## 10.22 面试题

### 题 1：模型权重 checkpoint 和训练状态 checkpoint 有什么区别？

答：模型权重 checkpoint 主要保存模型参数，适合推理、评估和下游微调。训练状态 checkpoint 保存恢复训练所需的完整状态，包括模型参数、优化器状态、scheduler、global step、随机种子、dataloader 状态和分布式 metadata，适合断点续训和故障恢复。

### 题 2：为什么大模型 checkpoint 很大？

答：因为除了模型参数，还可能包含梯度、优化器状态、一阶动量、二阶动量、FP32 master weight、分布式分片和 metadata。使用 Adam 时训练状态通常远大于纯模型权重。

### 题 3：为什么 checkpoint 需要 manifest？

答：分布式 checkpoint 通常由多个 shard 组成。Manifest 记录 checkpoint id、global step、并行配置、shard 列表、大小、checksum、数据版本和状态。没有 manifest，系统很难判断哪些文件组成同一个一致可恢复版本。

### 题 4：异步 checkpoint 有什么风险？

答：异步保存可能增加内存占用，和训练争抢 I/O，出现写入失败或状态不一致。必须有清晰的 snapshot、校验、commit 语义和失败处理，不能让半成品 checkpoint 被恢复使用。

### 题 5：恢复后 loss 异常可能是什么原因？

答：可能是 optimizer 状态、scheduler、random seed、dataloader 状态、mixed precision scaler 或数据位置没有正确恢复，也可能是代码、配置、tokenizer 或并行策略变化导致不兼容。

## 10.23 小练习

练习一：设计一个 checkpoint manifest。

要求：包含 checkpoint_id、global_step、model_config、parallelism_config、data_version、shards、checksum、status 和 created_at。

练习二：设计一个 checkpoint 生命周期策略。

要求：说明最近 checkpoint、milestone checkpoint、最佳 eval checkpoint、发布 checkpoint 和失败临时 checkpoint 分别保留多久。

练习三：分析 checkpoint 保存慢。

假设一个 256 卡训练任务每次保存 checkpoint 要 20 分钟，请从 shard、存储吞吐、metadata、网络、同步阻塞和并发任务角度排查。

练习四：设计一个恢复测试流程。

要求：模拟训练中断，从 checkpoint 恢复，并验证 step、loss、学习率、数据位置和 optimizer state 是否正确。

## 10.24 本章小结

本章讲了 checkpoint 存储、加载、恢复和生命周期管理。

你需要掌握：

1. Checkpoint 是训练状态的可恢复快照，不只是模型文件。
2. 模型权重 checkpoint 和训练状态 checkpoint 用途不同。
3. 训练状态需要保存参数、优化器、scheduler、step、随机状态、dataloader 和分布式 metadata。
4. 保存频率需要在训练阻塞、存储成本和故障恢复损失之间权衡。
5. 分片 checkpoint 需要 manifest、checksum 和 commit 语义。
6. 恢复训练要关注配置兼容性和状态一致性。
7. 弹性恢复需要 resharding，但实现复杂。
8. Checkpoint 是抢占式调度和容错训练的基础。
9. 生命周期管理和成本治理非常重要。
10. Checkpoint 必须定期做恢复测试，不能只测试保存。

下一章我们会讲容器化基础：Docker、镜像、驱动、CUDA 和依赖环境。
