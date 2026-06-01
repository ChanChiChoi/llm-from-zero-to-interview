# 第 22 章 训练容错：节点失败、通信失败、OOM 和自动重试

上一章讲了 checkpoint 策略。本章继续讲训练平台的可靠性：训练任务失败时，平台应该怎么办？

大模型训练任务运行时间长、资源多、组件多。节点、GPU、网络、存储、代码、数据、配置任意一处出问题，都可能导致任务失败。训练平台不能只把状态标成 failed，它应该尽量识别故障类型、判断是否可恢复、自动重试或从 checkpoint 恢复，并给用户可行动的诊断信息。

先记住一句话：

> 训练容错的核心不是无限重试，而是故障分类、损失控制、自动恢复和清晰诊断。

## 22.1 为什么训练容错重要

预训练任务可能使用几百张 GPU 跑几天。

如果任务因为一个节点故障中断，并且无法恢复，损失巨大。

训练容错的价值：

1. 降低失败重跑成本。
2. 提高 GPU 利用率。
3. 支持长周期训练。
4. 支持抢占式调度。
5. 降低运维压力。
6. 改善用户体验。
7. 让故障可复盘。

容错不是只靠训练脚本，也需要平台、调度、checkpoint、监控和存储共同配合。

## 22.2 故障分类

训练失败可以分为几大类：

1. 基础设施故障。
2. 资源不足。
3. 通信故障。
4. 存储和数据故障。
5. 用户代码故障。
6. 配置故障。
7. 数值稳定性故障。
8. 平台控制面故障。

不同故障处理方式不同。

例如：

1. 节点故障适合重调度后从 checkpoint 恢复。
2. OOM 可能需要调整 batch 或显存策略，盲目重试没用。
3. 数据权限错误应直接失败，不该自动重试。
4. 临时网络抖动可以重试。
5. loss NaN 需要诊断数值问题。

故障分类是自动恢复的前提。

## 22.3 可恢复故障和不可恢复故障

可以先粗分为两类。

可恢复故障：

1. 节点宕机。
2. GPU 短暂故障。
3. 网络短暂抖动。
4. 存储短暂不可用。
5. Pod 被驱逐。
6. 可抢占任务被抢占。
7. 镜像拉取临时失败。

不可恢复或不宜自动重试故障：

1. 代码语法错误。
2. 配置字段错误。
3. 数据权限错误。
4. checkpoint 格式不兼容。
5. 持续 OOM。
6. 模型结构和权重不匹配。
7. loss 持续 NaN。

自动重试要谨慎。错误分类不准，会导致 GPU 空烧。

## 22.4 节点失败

节点失败包括：

1. 机器宕机。
2. kubelet 异常。
3. GPU 掉卡。
4. 驱动异常。
5. 本地盘损坏。
6. 节点网络断开。

表现：

1. Pod 变成 failed 或 unknown。
2. 某个 rank 失联。
3. NCCL timeout。
4. 训练 hang。
5. GPU 指标消失。

处理策略：

1. 标记节点不可调度。
2. 收集节点诊断信息。
3. 释放任务资源。
4. 选择新节点。
5. 从最近 checkpoint 恢复。
6. 记录恢复事件。

如果任务不支持 checkpoint，只能失败或从头重跑。

## 22.5 GPU 故障

GPU 故障可能包括：

1. ECC error。
2. Xid error。
3. GPU reset。
4. 显存错误。
5. 温度过高。
6. 功耗异常。
7. GPU 不可见。

平台需要采集 GPU 健康指标。

如果某张 GPU 多次导致任务失败，应自动隔离该 GPU 或节点。

处理策略：

1. 记录 GPU 错误码。
2. 标记节点 unhealthy。
3. 禁止新任务调度。
4. 触发硬件维修流程。
5. 任务从 checkpoint 恢复到其他节点。

不要让坏 GPU 反复坑不同任务。

## 22.6 通信失败

通信失败是多机训练高频问题。

常见表现：

1. NCCL timeout。
2. NCCL unhandled error。
3. distributed init 卡住。
4. AllReduce 时间暴涨。
5. 某个 rank hang。
6. 网络重传增加。

可能原因：

1. 网络抖动。
2. 网卡故障。
3. RDMA 配置错误。
4. 防火墙或 NetworkPolicy 阻断。
5. 端口冲突。
6. rank 配置错误。
7. 某个 worker OOM 后退出，其他 worker 等通信。

处理策略：

1. 区分初始化失败还是运行中失败。
2. 检查第一个失败 rank。
3. 收集 NCCL 日志。
4. 检查网络指标。
5. 对临时故障自动重试。
6. 对配置错误快速失败。

通信失败不能只看最后报错，因为最后报错的 rank 不一定是根因。

## 22.7 OOM

OOM 是训练常见失败。

包括：

1. GPU OOM。
2. CPU memory OOM。
3. container memory limit OOM。
4. shared memory 不足。

GPU OOM 常见原因：

1. batch size 太大。
2. sequence length 太长。
3. activation 太多。
4. ZeRO/FSDP 配置不合理。
5. gradient accumulation 配错。
6. 没开启 activation checkpointing。
7. KV cache 或 eval 过程占用过高。

OOM 是否适合自动重试？

通常不适合直接原样重试。

更好的策略：

1. 失败并给出建议。
2. 自动降低 micro batch 后重试，但必须用户允许。
3. 建议开启 activation checkpointing。
4. 建议使用 ZeRO/FSDP。
5. 提示显存峰值和失败 step。

盲目重试 OOM 只会重复失败。

## 22.8 数据故障

数据故障包括：

1. 数据路径不存在。
2. 数据权限不足。
3. 数据文件损坏。
4. shard 缺失。
5. 读取超时。
6. tokenizer 不匹配。
7. 样本格式错误。
8. 数据中有异常值。

处理策略：

1. 提交前校验数据版本。
2. 启动前做小样本读取测试。
3. 训练中记录坏样本位置。
4. 对临时读取失败重试。
5. 对格式错误快速失败。
6. 生成数据诊断报告。

数据错误如果被吞掉，可能导致模型质量问题，而不只是训练失败。

## 22.9 Checkpoint 故障

Checkpoint 故障包括：

1. 保存失败。
2. 加载失败。
3. shard 缺失。
4. checksum 不匹配。
5. manifest 不完整。
6. 存储空间不足。
7. 权限不足。
8. 恢复后 loss 异常。

处理策略：

1. 保存失败要告警。
2. 不允许恢复 uncommitted checkpoint。
3. 加载前校验 manifest。
4. 失败后尝试上一个健康 checkpoint。
5. 定期做恢复测试。
6. 记录 checkpoint 失败事件。

Checkpoint 是容错基础，它自己也需要容错。

## 22.10 数值稳定性故障

数值故障包括：

1. loss NaN。
2. loss inf。
3. gradient norm 爆炸。
4. 参数变成 NaN。
5. mixed precision overflow。
6. reward 异常。

可能原因：

1. learning rate 太大。
2. 数据异常。
3. loss scaling 配置错误。
4. FP16 不稳定。
5. 梯度裁剪缺失。
6. 初始化或模型结构问题。

处理策略：

1. 立即停止或暂停任务。
2. 保存诊断状态。
3. 标记异常 step。
4. 从上一个健康 checkpoint 回滚。
5. 提示可能配置原因。

数值故障不应该简单自动重试，因为继续训练可能浪费资源。

## 22.11 自动重试策略

自动重试要有边界。

重试策略可以包含：

1. max_retry_count。
2. retry_backoff。
3. retryable_error_types。
4. non_retryable_error_types。
5. resume_from_checkpoint。
6. node_blacklist。
7. timeout。

示例：

```yaml
retry:
  max_attempts: 3
  retryable_errors:
    - node_failure
    - transient_network_error
    - image_pull_timeout
  non_retryable_errors:
    - config_error
    - permission_denied
    - repeated_oom
  backoff_seconds: 300
```

自动重试必须记录 attempt_id，避免日志和指标混乱。

## 22.12 从 checkpoint 恢复

从 checkpoint 恢复流程：

1. 找到最新 committed checkpoint。
2. 校验 manifest 和 checksum。
3. 检查配置兼容性。
4. 重新申请资源。
5. 启动新 attempt。
6. 加载 checkpoint。
7. 恢复 step、optimizer、scheduler、random state。
8. 记录恢复事件。

恢复后要观察：

1. loss 是否连续。
2. learning rate 是否正确。
3. 数据位置是否正确。
4. step 是否正确。
5. GPU 利用率是否正常。

恢复成功不只是进程跑起来，还要训练状态正确。

## 22.13 故障诊断报告

训练失败后，平台应该生成诊断报告。

报告包括：

1. 失败时间。
2. 失败阶段。
3. 首个失败 rank。
4. 错误类型。
5. 错误摘要。
6. 相关日志链接。
7. 相关指标截图。
8. 最近 checkpoint。
9. 是否已重试。
10. 建议动作。

示例建议：

```text
错误类型：GPU OOM
失败 rank：rank 3
失败 step：420
显存峰值：79.4GB / 80GB
建议：降低 micro_batch_size，开启 activation checkpointing，或使用 ZeRO stage 3。
```

诊断报告能显著降低用户和平台团队沟通成本。

## 22.14 故障隔离和黑名单

如果某个节点或 GPU 多次导致失败，平台应该隔离。

机制包括：

1. node blacklist。
2. GPU blacklist。
3. 自动 cordon 节点。
4. 标记 unhealthy。
5. 禁止新任务调度。
6. 通知运维维修。

同时要避免误伤。

例如一次用户代码 OOM 不应该把节点拉黑。

只有硬件错误、系统错误或多任务重复失败才适合隔离。

## 22.15 容错和用户体验

容错系统要让用户知道发生了什么。

用户应该能看到：

1. 任务失败原因。
2. 是否自动重试。
3. 第几次 attempt。
4. 从哪个 checkpoint 恢复。
5. 损失了多少 step。
6. 当前是否需要用户操作。

不要只显示：

```text
Job failed
```

这对用户没有帮助。

## 22.16 面试中如何回答训练容错

如果面试官问：

```text
大模型训练平台如何做容错和自动重试？
```

可以这样回答：

```text
我会先做故障分类，而不是所有失败都自动重试。故障可以分为节点/GPU 故障、通信故障、存储和数据故障、OOM、用户代码错误、配置错误和数值稳定性问题。

对节点故障、临时网络问题、Pod 驱逐、镜像拉取超时这类可恢复故障，可以自动重试，并从最新 committed checkpoint 恢复。对配置错误、权限错误、代码 bug、checkpoint 不兼容和持续 OOM，应该快速失败并给出诊断，而不是盲目重试。

平台需要 checkpoint manager 支持 manifest、checksum、committed 状态和恢复校验；调度层重新分配健康节点；观测系统记录 attempt_id、失败 rank、错误类型、相关日志和指标。

自动重试要有 max_attempts、backoff、retryable error list 和 node blacklist。任务恢复后要检查 loss、step、learning rate 和数据位置是否连续。最终给用户生成诊断报告，说明失败原因、是否恢复、损失进度和建议动作。
```

## 22.17 常见误区

误区一：自动重试越多越可靠。

盲目重试会浪费 GPU。必须区分可恢复和不可恢复错误。

误区二：有 checkpoint 就一定能恢复。

Checkpoint 可能不完整、不兼容或恢复后状态异常，必须校验和测试。

误区三：OOM 可以直接重试。

原样重试通常还会 OOM，需要调整 batch、显存策略或配置。

误区四：NCCL timeout 一定是网络问题。

也可能是某个 rank OOM、节点故障、配置不一致或端口问题。

误区五：失败日志足够定位问题。

还需要指标、事件、rank 信息、节点健康和 checkpoint 状态。

## 22.18 面试题

### 题 1：哪些训练故障适合自动重试？

答：节点故障、Pod 驱逐、临时网络抖动、存储短暂不可用、镜像拉取超时、可抢占任务中断等适合自动重试。配置错误、权限错误、代码 bug、持续 OOM、checkpoint 不兼容和数值发散不适合盲目重试。

### 题 2：OOM 应该如何处理？

答：先区分 GPU OOM、CPU OOM 和容器内存 OOM。GPU OOM 通常需要降低 micro batch、减少 sequence length、开启 activation checkpointing、使用 ZeRO/FSDP 或更大显存 GPU。平台可以给建议，不能简单原样重试。

### 题 3：NCCL timeout 如何排查？

答：先看第一个失败 rank 和 NCCL 日志，再查是否有 worker OOM 或节点失联。然后看网络、RDMA、端口、rank/world size 配置、NetworkPolicy、网卡错误和通信指标。NCCL timeout 可能是通信问题，也可能是其他 rank 先失败导致。

### 题 4：从 checkpoint 恢复后要验证什么？

答：要验证 checkpoint manifest 和 checksum、配置兼容性、global step、optimizer、scheduler、random state、dataloader 位置、loss 连续性和 learning rate 是否正确。

### 题 5：故障诊断报告应该包含什么？

答：包括失败时间、失败阶段、首个失败 rank、错误类型、错误摘要、相关日志和指标、最近 checkpoint、重试次数、恢复状态和建议动作。

## 22.19 小练习

练习一：设计一个错误分类表。

要求：把 node_failure、gpu_error、nccl_timeout、oom、permission_denied、config_error、loss_nan 分成 retryable 和 non_retryable。

练习二：设计自动重试策略。

要求：包含 max_attempts、backoff、retryable_errors、non_retryable_errors、checkpoint 恢复和 node blacklist。

练习三：分析一次 NCCL timeout。

要求：从 rank、OOM、节点、网络、端口、NCCL 环境变量和调度拓扑角度排查。

练习四：设计故障诊断报告模板。

要求：包含错误摘要、根因假设、证据、相关日志、相关指标和建议动作。

## 22.20 本章小结

本章讲了训练容错。

你需要掌握：

1. 训练容错的核心是故障分类、损失控制、自动恢复和清晰诊断。
2. 可恢复故障和不可恢复故障要区别处理。
3. 节点和 GPU 故障适合重调度并从 checkpoint 恢复。
4. 通信失败要结合 rank、NCCL、网络和其他 worker 状态排查。
5. OOM 通常不适合原样重试，需要调整资源或配置。
6. 数据和 checkpoint 故障要有校验、重试和诊断。
7. 数值稳定性故障需要停止、诊断和可能回滚。
8. 自动重试要有边界、backoff、attempt_id 和错误类型策略。
9. 从 checkpoint 恢复后要验证训练状态连续性。
10. 故障诊断报告能显著提升平台用户体验。

下一章我们会讲训练数据供给：流式读取、缓存、shuffle 和数据局部性。
