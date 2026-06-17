# 第 22 章 训练容错：节点失败、通信失败、OOM 和自动重试

上一章讲了 checkpoint 策略。本章继续讲训练平台的可靠性：训练任务失败时，平台应该怎么办？

大模型训练任务运行时间长、资源多、组件多。节点、GPU、网络、存储、代码、数据、配置任意一处出问题，都可能导致任务失败。训练平台不能只把状态标成 failed，它应该尽量识别故障类型、判断是否可恢复、自动重试或从 checkpoint 恢复，并给用户可行动的诊断信息。

先记住一句话：

> 训练容错的核心不是无限重试，而是故障分类、损失控制、自动恢复和清晰诊断。

## 22.0 本讲资料边界与第二轮精修口径

本讲第二轮精修时，资料口径主要对齐几类公开工程资料：PyTorch Elastic / `torchrun` 对 `max_restarts`、rendezvous、worker failure、agent failure、node failure 和 membership change 的分布式恢复口径；Kubernetes Job 对 `restartPolicy`、`backoffLimit`、`podFailurePolicy`、Pod 失败计数和指数退避的控制面语义；NVIDIA NCCL troubleshooting 对通信环境、网络接口、共享内存、GPU Direct RDMA、debug 日志和异步错误处理的排障口径；PyTorch CUDA memory management 对 caching allocator、`memory_allocated`、`memory_reserved`、`max_memory_allocated`、memory snapshot 和 OOM 调参的口径。

这里不把某个训练平台、某个云厂商、某个 GPU 型号、某个调度器插件或某个 NCCL 环境变量组合写成通用答案。正文只抽象训练容错里的稳定问题：故障分类、retryable / non-retryable 判断、attempt 预算、backoff、checkpoint 恢复、节点替换、GPU 健康隔离、通信 hang 检测、OOM 处理、数据和 checkpoint 故障、数值稳定性、诊断报告、黑名单、用户可见动作和审计 trace。

第二轮精修重点放在三个方面：

1. 把“自动重试”改写为有边界的故障分类和恢复策略，明确哪些错误可以重试，哪些错误应该快速失败。
2. 补充可计算指标：最大丢失 GPU-hours、重试成功率、无效重试率、OOM 显存余量、rank hang 心跳间隔和恢复连续性。
3. 增加一个 0 依赖 Python demo，把训练容错从经验 checklist 变成可运行的门禁检查。

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

## 22.16 训练容错审计指标与最小 demo

这一节把前面的经验规则收敛成一个可检查的训练容错审计表。真实平台可以接入 Kubernetes Job 事件、PyTorch Elastic attempt、NCCL 日志、GPU 健康事件、PyTorch CUDA OOM 统计、checkpoint manifest 和训练状态机；教学 demo 里先用 toy 字段模拟这些证据。

一个训练容错样本可以抽象成：

```math
f_i=(c_i,r_i,a_i,b_i,k_i,n_i,g_i,h_i,o_i,d_i,p_i,s_i,u_i,t_i,z_i)
```

其中，`c_i` 是故障分类，`r_i` 是 retryable / non-retryable 判断，`a_i` 是 attempt 预算，`b_i` 是 backoff，`k_i` 是 checkpoint 恢复证据，`n_i` 是节点替换证据，`g_i` 是 GPU 健康隔离证据，`h_i` 是 rank heartbeat / hang 证据，`o_i` 是 OOM 处理策略，`d_i` 是数据权限和数据读取证据，`p_i` 是 checkpoint committed / manifest / checksum 证据，`s_i` 是数值稳定性证据，`u_i` 是用户可见动作，`t_i` 是诊断 trace，`z_i` 是最终门禁。

统一覆盖率仍然可以写成：

```math
C_j=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[g_j(f_i)=1]
```

其中，`N` 是审计样本数，`g_j(f_i)=1` 表示第 `i` 个样本通过第 `j` 个训练容错检查。

失败后的最大丢失 GPU-hours 可以写成：

```math
L_{\mathrm{gpu}}=G\frac{t_{\mathrm{fail}}-t_{\mathrm{ckpt}}}{60}
```

其中，`G` 是训练使用的 GPU 数，`t_fail` 和 `t_ckpt` 用分钟表示，`L_gpu` 越大，说明 checkpoint 间隔、失败检测或恢复路径越贵。

重试成功率和无效重试率可以写成：

```math
R_{\mathrm{success}}=\frac{N_{\mathrm{recovered}}}{N_{\mathrm{retry}}}
```

```math
R_{\mathrm{waste}}=\frac{N_{\mathrm{bad\_retry}}}{N_{\mathrm{retry}}}
```

前者看自动恢复是否真的有效，后者看平台是否在对不可恢复故障浪费 GPU。

OOM 显存余量可以写成：

```math
H_{\mathrm{mem}}=1-\frac{M_{\mathrm{peak}}}{M_{\mathrm{limit}}}
```

如果 `H_mem` 接近 0，说明任务已经贴着显存上限运行；原样重试通常没有意义。

rank hang 的心跳年龄可以写成：

```math
A_{\mathrm{hb}}=t_{\mathrm{now}}-t_{\mathrm{last\_heartbeat}}
```

如果 `A_hb` 超过阈值，要结合首个失败 rank、NCCL 日志、worker 退出状态、节点健康和 OOM 事件判断根因，不能只把所有 hang 都归因于网络。

从 checkpoint 恢复后的连续性可以写成：

```math
V_{\mathrm{resume}}=\mathbf{1}\left[|\ell_{\mathrm{after}}-\ell_{\mathrm{before}}|\le \epsilon_{\ell}\land s_{\mathrm{after}}=s_{\mathrm{ckpt}}+\Delta s\right]
```

其中，`\ell` 是 loss，`s` 是 step。恢复成功不是进程重新启动，而是 loss、step、learning rate、optimizer、scheduler、RNG 和 dataloader cursor 都合理连续。

最终训练容错门禁可以写成：

```math
G_{\mathrm{fault}}=\mathbf{1}\left[\min_j C_j\ge \tau_j \land L_{\mathrm{gpu}}\le B_{\mathrm{loss}} \land R_{\mathrm{waste}}\le \rho_{\mathrm{waste}} \land A_{\mathrm{hb}}\le \tau_{\mathrm{hb}} \land V_{\mathrm{resume}}=1 \land P_0=0\right]
```

这个门禁背后的含义是：训练容错不是“失败了再拉起来”，而是故障分类、重试预算、checkpoint 恢复、节点和 GPU 隔离、通信 hang 诊断、OOM 快速失败、数值异常回滚、诊断报告和用户动作都可审计。

下面的 0 依赖 Python demo 演示一个最小训练容错审计器。它不是生产平台实现，而是把面试中最容易说散的规则变成可运行检查。

```python
from copy import deepcopy


GATES = [
    "fault_taxonomy_defined",
    "retryability_classification",
    "retry_budget_bounded",
    "backoff_policy_defined",
    "checkpoint_resume_readiness",
    "node_failure_reschedule",
    "gpu_health_isolation",
    "communication_hang_detection",
    "oom_handling_policy",
    "data_permission_fast_fail",
    "checkpoint_commit_safety",
    "numeric_instability_stop_rollback",
    "diagnosis_report_evidence",
    "blacklist_precision",
    "user_action_clarity",
    "training_fault_tolerance_gate",
]


def base_case():
    return {
        "name": "complete_fault_tolerance_case",
        "fault_type": "node_failure",
        "stage": "runtime",
        "taxonomy": {
            "node_failure": ["runtime", "infrastructure"],
            "gpu_xid": ["runtime", "hardware"],
            "transient_network_error": ["distributed_init", "runtime"],
            "oom": ["runtime", "resource"],
            "permission_denied": ["preflight", "data"],
            "checkpoint_uncommitted": ["resume", "checkpoint"],
            "loss_nan": ["runtime", "numeric"],
        },
        "retryable": True,
        "retryable_errors": ["node_failure", "gpu_xid", "transient_network_error", "image_pull_timeout"],
        "non_retryable_errors": ["config_error", "permission_denied", "repeated_oom", "checkpoint_uncommitted", "loss_nan"],
        "attempt": 2,
        "max_attempts": 3,
        "backoff_seconds": 300,
        "min_backoff_seconds": 60,
        "max_backoff_seconds": 1800,
        "jitter": True,
        "checkpoint": {
            "latest_committed": True,
            "manifest_valid": True,
            "checksum_valid": True,
            "config_compatible": True,
            "restore_uncommitted": False,
            "resume_fields": ["model", "optimizer", "scheduler", "random_state", "dataloader_cursor"],
        },
        "node": {
            "failed_node": "node-a",
            "rescheduled_node": "node-b",
            "cordoned_nodes": ["node-a"],
            "worker_group_restarted": True,
        },
        "gpu": {
            "error_codes": ["Xid 79"],
            "isolation_policy": True,
            "node_marked_unhealthy": True,
            "repeated_hardware_failures": 2,
            "false_positive_guard": True,
            "blacklist_user_code_oom": False,
        },
        "communication": {
            "heartbeat_age_s": 12,
            "heartbeat_timeout_s": 60,
            "first_failed_rank": 3,
            "nccl_logs_collected": True,
            "worker_exit_status_collected": True,
        },
        "oom": {
            "kind": "none",
            "blind_retry": False,
            "peak_gib": 78.0,
            "limit_gib": 80.0,
            "auto_batch_decrease_requires_opt_in": True,
            "recommendations": ["lower micro_batch_size", "enable activation checkpointing", "use FSDP or ZeRO"],
        },
        "data": {
            "permission_checked": True,
            "retry_permission_denied": False,
            "sample_read_test": True,
        },
        "numeric": {
            "non_finite_detected": False,
            "blind_retry_on_nan": False,
            "rollback_checkpoint": True,
            "diagnostics_saved": True,
        },
        "diagnosis_report": {
            "fields": [
                "failed_at",
                "stage",
                "first_failed_rank",
                "error_type",
                "checkpoint_id",
                "attempt_id",
                "log_links",
                "metric_links",
                "recommended_action",
            ]
        },
        "user_action": {
            "visible_status": "retrying",
            "attempt_visible": True,
            "checkpoint_visible": True,
            "lost_steps_visible": True,
            "recommendation": "Platform retried on a healthy node from committed checkpoint ckpt-1020.",
        },
        "history": {"retry_total": 5, "recovered": 4, "bad_retries": 1},
        "gpu_count": 64,
        "fail_time_min": 186,
        "checkpoint_time_min": 180,
        "loss_before": 1.920,
        "loss_after": 1.921,
        "loss_epsilon": 0.01,
        "checkpoint_step": 1000,
        "step_after": 1020,
        "delta_step": 20,
        "training_fault_tolerance_gate": True,
    }


def contains_all(values, required):
    return set(required).issubset(set(values))


def evaluate(case):
    required_resume = ["model", "optimizer", "scheduler", "random_state", "dataloader_cursor"]
    required_report = [
        "failed_at",
        "stage",
        "first_failed_rank",
        "error_type",
        "checkpoint_id",
        "attempt_id",
        "log_links",
        "metric_links",
        "recommended_action",
    ]

    retryable_expected = case["fault_type"] in case["retryable_errors"]
    communication = case["communication"]
    checkpoint = case["checkpoint"]
    node = case["node"]
    gpu = case["gpu"]
    oom = case["oom"]
    numeric = case["numeric"]

    checks = {}
    checks["fault_taxonomy_defined"] = (
        case["fault_type"] in case["taxonomy"]
        and case["stage"] in case["taxonomy"][case["fault_type"]]
    )
    checks["retryability_classification"] = (
        case["retryable"] == retryable_expected
        and not (case["retryable"] and case["fault_type"] in case["non_retryable_errors"])
    )
    checks["retry_budget_bounded"] = (
        isinstance(case["max_attempts"], int)
        and 1 <= case["max_attempts"] <= 5
        and 1 <= case["attempt"] <= case["max_attempts"]
    )
    checks["backoff_policy_defined"] = (
        case["min_backoff_seconds"] <= case["backoff_seconds"] <= case["max_backoff_seconds"]
        and case["jitter"]
    )
    checks["checkpoint_resume_readiness"] = (
        checkpoint["latest_committed"]
        and checkpoint["manifest_valid"]
        and checkpoint["checksum_valid"]
        and checkpoint["config_compatible"]
        and contains_all(checkpoint["resume_fields"], required_resume)
    )
    checks["node_failure_reschedule"] = (
        case["fault_type"] != "node_failure"
        or (
            node["worker_group_restarted"]
            and node["failed_node"] != node["rescheduled_node"]
            and node["failed_node"] in node["cordoned_nodes"]
        )
    )
    checks["gpu_health_isolation"] = (
        not gpu["error_codes"]
        or (
            gpu["isolation_policy"]
            and gpu["node_marked_unhealthy"]
            and gpu["repeated_hardware_failures"] >= 2
        )
    )
    checks["communication_hang_detection"] = (
        communication["heartbeat_age_s"] <= communication["heartbeat_timeout_s"]
        or (
            communication["first_failed_rank"] is not None
            and communication["nccl_logs_collected"]
            and communication["worker_exit_status_collected"]
        )
    )
    checks["oom_handling_policy"] = (
        oom["kind"] == "none"
        or (
            oom["kind"] in ["gpu", "cpu", "container"]
            and not oom["blind_retry"]
            and (oom["recommendations"] or oom["auto_batch_decrease_requires_opt_in"])
        )
    )
    checks["data_permission_fast_fail"] = (
        case["data"]["permission_checked"]
        and case["data"]["sample_read_test"]
        and not case["data"]["retry_permission_denied"]
    )
    checks["checkpoint_commit_safety"] = (
        checkpoint["latest_committed"]
        and checkpoint["manifest_valid"]
        and checkpoint["checksum_valid"]
        and not checkpoint["restore_uncommitted"]
    )
    checks["numeric_instability_stop_rollback"] = (
        not numeric["blind_retry_on_nan"]
        and (not numeric["non_finite_detected"] or (numeric["rollback_checkpoint"] and numeric["diagnostics_saved"]))
    )
    checks["diagnosis_report_evidence"] = contains_all(case["diagnosis_report"]["fields"], required_report)
    checks["blacklist_precision"] = (
        gpu["false_positive_guard"]
        and not gpu["blacklist_user_code_oom"]
        and (not gpu["error_codes"] or gpu["repeated_hardware_failures"] >= 2)
    )
    checks["user_action_clarity"] = (
        bool(case["user_action"]["visible_status"])
        and case["user_action"]["attempt_visible"]
        and case["user_action"]["checkpoint_visible"]
        and case["user_action"]["lost_steps_visible"]
        and bool(case["user_action"]["recommendation"])
    )
    checks["training_fault_tolerance_gate"] = bool(case["training_fault_tolerance_gate"])
    return checks


def summarize_examples(case):
    history = case["history"]
    retry_total = max(history["retry_total"], 1)
    lost_gpu_hours = case["gpu_count"] * (case["fail_time_min"] - case["checkpoint_time_min"]) / 60
    oom_headroom = 1 - oom_ratio(case)
    resume_continuity = (
        abs(case["loss_after"] - case["loss_before"]) <= case["loss_epsilon"]
        and case["step_after"] == case["checkpoint_step"] + case["delta_step"]
    )
    return {
        "lost_gpu_hours": round(lost_gpu_hours, 3),
        "retry_success_rate": round(history["recovered"] / retry_total, 3),
        "waste_retry_rate": round(history["bad_retries"] / retry_total, 3),
        "oom_headroom": round(oom_headroom, 3),
        "heartbeat_age_s": case["communication"]["heartbeat_age_s"],
        "resume_continuity": resume_continuity,
    }


def oom_ratio(case):
    limit = max(case["oom"]["limit_gib"], 1e-9)
    return case["oom"]["peak_gib"] / limit


def mutate(name, editor):
    case = deepcopy(base_case())
    case["name"] = name
    editor(case)
    return case


def build_cases():
    cases = [base_case()]
    cases.append(mutate("fault_taxonomy_missing_bad", lambda c: c.update({"fault_type": "mystery_exit", "retryable": False})))
    cases.append(mutate("retryability_wrong_bad", lambda c: c.update({"fault_type": "permission_denied", "stage": "preflight", "retryable": True})))
    cases.append(mutate("retry_budget_missing_bad", lambda c: c.update({"max_attempts": None})))
    cases.append(mutate("backoff_missing_bad", lambda c: c.update({"backoff_seconds": 0, "jitter": False})))
    cases.append(mutate("checkpoint_resume_missing_bad", lambda c: c["checkpoint"].update({"resume_fields": ["model"]})))
    cases.append(mutate("node_failure_not_rescheduled_bad", lambda c: c["node"].update({"rescheduled_node": "node-a", "cordoned_nodes": []})))
    cases.append(mutate("gpu_health_not_isolated_bad", lambda c: c["gpu"].update({"isolation_policy": False, "node_marked_unhealthy": False})))
    cases.append(mutate("nccl_hang_unclassified_bad", lambda c: c["communication"].update({"heartbeat_age_s": 180, "first_failed_rank": None, "nccl_logs_collected": False})))
    cases.append(mutate("oom_blind_retry_bad", lambda c: c["oom"].update({"kind": "gpu", "blind_retry": True, "recommendations": []})))
    cases.append(mutate("data_permission_retried_bad", lambda c: c["data"].update({"retry_permission_denied": True})))
    cases.append(mutate("checkpoint_uncommitted_recovered_bad", lambda c: c["checkpoint"].update({"restore_uncommitted": True})))
    cases.append(mutate("numeric_nan_retried_bad", lambda c: c["numeric"].update({"non_finite_detected": True, "blind_retry_on_nan": True, "diagnostics_saved": False})))
    cases.append(mutate("diagnosis_report_missing_bad", lambda c: c["diagnosis_report"].update({"fields": ["failed_at", "error_type"]})))
    cases.append(mutate("blacklist_false_positive_bad", lambda c: c["gpu"].update({"blacklist_user_code_oom": True, "false_positive_guard": False})))
    cases.append(mutate("user_action_unclear_bad", lambda c: c["user_action"].update({"visible_status": "", "recommendation": ""})))
    cases.append(mutate("fault_tolerance_gate_missing_bad", lambda c: c.update({"training_fault_tolerance_gate": False})))
    return cases


def main():
    cases = build_cases()
    evaluations = {case["name"]: evaluate(case) for case in cases}
    metrics = {
        gate: round(sum(1 for case in cases if evaluations[case["name"]][gate]) / len(cases), 3)
        for gate in GATES
    }
    failed_cases = [
        case["name"]
        for case in cases
        if not all(evaluations[case["name"]].values())
    ]
    failed_gates = [
        gate
        for gate in GATES
        if any(not evaluations[case["name"]][gate] for case in cases)
    ]
    complete = cases[0]
    smoke = {
        "complete_case_passes": all(evaluations[complete["name"]].values()),
        "caught_oom_blind_retry": not evaluations["oom_blind_retry_bad"]["oom_handling_policy"],
        "caught_uncommitted_checkpoint": not evaluations["checkpoint_uncommitted_recovered_bad"]["checkpoint_commit_safety"],
        "caught_nan_retry": not evaluations["numeric_nan_retried_bad"]["numeric_instability_stop_rollback"],
        "caught_blacklist_false_positive": not evaluations["blacklist_false_positive_bad"]["blacklist_precision"],
        "caught_gate_gap": not evaluations["fault_tolerance_gate_missing_bad"]["training_fault_tolerance_gate"],
    }
    training_fault_tolerance_gate_pass = not failed_cases and all(v >= 1.0 for v in metrics.values())

    print("training_fault_tolerance_examples=", summarize_examples(complete))
    print("smoke=", smoke)
    print("metrics=", metrics)
    print("hard_blocker_count=", len(failed_cases))
    print("failed_cases=", failed_cases)
    print("failed_gates=", failed_gates)
    print("training_fault_tolerance_gate_pass=", training_fault_tolerance_gate_pass)


if __name__ == "__main__":
    main()
```

预期输出类似：

```text
training_fault_tolerance_examples= {'lost_gpu_hours': 6.4, 'retry_success_rate': 0.8, 'waste_retry_rate': 0.2, 'oom_headroom': 0.025, 'heartbeat_age_s': 12, 'resume_continuity': True}
smoke= {'complete_case_passes': True, 'caught_oom_blind_retry': True, 'caught_uncommitted_checkpoint': True, 'caught_nan_retry': True, 'caught_blacklist_false_positive': True, 'caught_gate_gap': True}
metrics= {'fault_taxonomy_defined': 0.941, 'retryability_classification': 0.941, 'retry_budget_bounded': 0.941, 'backoff_policy_defined': 0.941, 'checkpoint_resume_readiness': 0.941, 'node_failure_reschedule': 0.941, 'gpu_health_isolation': 0.941, 'communication_hang_detection': 0.941, 'oom_handling_policy': 0.941, 'data_permission_fast_fail': 0.941, 'checkpoint_commit_safety': 0.941, 'numeric_instability_stop_rollback': 0.941, 'diagnosis_report_evidence': 0.941, 'blacklist_precision': 0.941, 'user_action_clarity': 0.941, 'training_fault_tolerance_gate': 0.941}
hard_blocker_count= 16
failed_cases= ['fault_taxonomy_missing_bad', 'retryability_wrong_bad', 'retry_budget_missing_bad', 'backoff_missing_bad', 'checkpoint_resume_missing_bad', 'node_failure_not_rescheduled_bad', 'gpu_health_not_isolated_bad', 'nccl_hang_unclassified_bad', 'oom_blind_retry_bad', 'data_permission_retried_bad', 'checkpoint_uncommitted_recovered_bad', 'numeric_nan_retried_bad', 'diagnosis_report_missing_bad', 'blacklist_false_positive_bad', 'user_action_unclear_bad', 'fault_tolerance_gate_missing_bad']
failed_gates= ['fault_taxonomy_defined', 'retryability_classification', 'retry_budget_bounded', 'backoff_policy_defined', 'checkpoint_resume_readiness', 'node_failure_reschedule', 'gpu_health_isolation', 'communication_hang_detection', 'oom_handling_policy', 'data_permission_fast_fail', 'checkpoint_commit_safety', 'numeric_instability_stop_rollback', 'diagnosis_report_evidence', 'blacklist_precision', 'user_action_clarity', 'training_fault_tolerance_gate']
training_fault_tolerance_gate_pass= False
```

面试里可以把这个 demo 压缩成一句话：训练容错的工程质量，要用“可恢复故障恢复成功、不可恢复故障快速失败、恢复后状态连续、坏节点和坏 GPU 不反复伤害任务、用户能看到明确动作”来证明，而不是用“平台会自动重试”来证明。

## 22.17 面试中如何回答训练容错

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

## 22.18 常见误区

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

## 22.19 面试题

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

## 22.20 小练习

练习一：设计一个错误分类表。

要求：把 node_failure、gpu_error、nccl_timeout、oom、permission_denied、config_error、loss_nan 分成 retryable 和 non_retryable。

练习二：设计自动重试策略。

要求：包含 max_attempts、backoff、retryable_errors、non_retryable_errors、checkpoint 恢复和 node blacklist。

练习三：分析一次 NCCL timeout。

要求：从 rank、OOM、节点、网络、端口、NCCL 环境变量和调度拓扑角度排查。

练习四：设计故障诊断报告模板。

要求：包含错误摘要、根因假设、证据、相关日志、相关指标和建议动作。

## 22.21 本章小结

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
