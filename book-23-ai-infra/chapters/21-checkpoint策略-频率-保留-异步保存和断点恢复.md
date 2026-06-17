# 第 21 章 Checkpoint 策略：频率、保留、异步保存和断点恢复

第 10 章我们讲过 checkpoint 的存储、加载、恢复和生命周期管理。本章回到训练平台工程视角，重点讲 checkpoint 策略：训练时应该多久保存一次？保留哪些？异步保存怎么做？抢占和失败恢复如何联动？

Checkpoint 策略不是越频繁越好，也不是越省越好。它是在训练成本、存储成本、恢复损失、评估需求和平台复杂度之间做平衡。

先记住一句话：

> Checkpoint 策略的目标，是用可接受的 I/O 和存储成本，把训练失败后的损失控制在可接受范围内，并支持评估、回滚、抢占和发布。

## 21.0 本讲资料边界与第二轮精修口径

本讲第二轮精修时，资料口径主要对齐几类公开工程资料：PyTorch Distributed Checkpoint 对分布式 `state_dict`、planner、storage writer、异步保存和加载的抽象；DeepSpeed checkpointing 对模型、优化器、scheduler、ZeRO 分片状态保存和恢复的接口边界；Kubernetes priority / preemption、Pod 终止宽限期和任务生命周期对可抢占训练的控制面影响；对象存储生命周期策略对 checkpoint 保留、归档和删除的治理口径。

这里不把某个框架 API、某种对象存储、某个调度器或某家云厂商的参数写成通用标准。正文只抽象 checkpoint 策略的稳定问题：RPO/RTO、保存触发、保存内容分层、同步 / 异步保存、一致 snapshot、临时目录、checksum、manifest commit、分片元数据、恢复选择、抢占联动、评估联动、发布联动、监控告警、保留成本和恢复演练。

第二轮精修重点放在三个方面：

1. 把“多久保存一次”从经验参数升级为 RPO、RTO、保存阻塞比例和最大丢失进度的可计算策略。
2. 把 resume、eval、release checkpoint 分清，避免所有 checkpoint 都保存完整训练状态或只保留 latest。
3. 增加一个 0 依赖 Python demo，把 checkpoint 策略从口头 checklist 变成能发现坏策略的门禁。

## 21.1 Checkpoint 策略要解决什么问题

Checkpoint 策略要回答八个问题：

1. 什么时候保存？
2. 保存什么内容？
3. 保存到哪里？
4. 同步保存还是异步保存？
5. 保留多少个？
6. 哪些 checkpoint 是关键版本？
7. 失败后从哪个 checkpoint 恢复？
8. 如何控制存储成本？

这些问题如果没有平台策略，就会变成每个训练脚本各自实现，最后无法治理。

## 21.2 保存频率的 trade-off

保存频率是最核心的策略。

保存太频繁：

1. 训练被 I/O 打断。
2. checkpoint 存储膨胀。
3. metadata 压力大。
4. 多任务同时保存导致存储拥塞。
5. 成本上升。

保存太少：

1. 故障后损失大量训练进度。
2. 抢占恢复代价高。
3. 找不到中间版本评估。
4. 训练异常后可回滚点少。

一个合理问题是：

```text
如果任务现在失败，我最多能接受损失多少 GPU-hours？
```

保存频率应该围绕这个目标设计。

## 21.3 RPO 和 RTO

可以借用两个可靠性概念。

RPO：Recovery Point Objective，恢复点目标。

它表示最多能接受丢失多少训练进度。

例如 RPO = 30 minutes，表示任务失败后最多回退 30 分钟训练。

RTO：Recovery Time Objective，恢复时间目标。

它表示从失败到恢复训练需要多久。

例如 RTO = 15 minutes，表示故障后 15 分钟内恢复运行。

Checkpoint 策略同时影响 RPO 和 RTO。

1. 保存越频繁，RPO 越小。
2. checkpoint 加载越快，RTO 越小。
3. checkpoint 越大，RTO 可能越大。
4. 存储越慢，保存和恢复都更慢。

面试中讲 checkpoint 策略时，用 RPO/RTO 会显得很工程化。

## 21.4 按 step 保存还是按时间保存

常见保存方式有两种。

按 step 保存：

```text
每 1000 step 保存一次
```

优点：

1. 和训练进度对齐。
2. 便于评估和比较。
3. 实现简单。

缺点：

1. step time 变化时，真实时间间隔不稳定。
2. 如果 step 很慢，RPO 可能变大。

按时间保存：

```text
每 30 分钟保存一次
```

优点：

1. RPO 更容易控制。
2. 适合长时间任务。

缺点：

1. checkpoint step 不规整。
2. 和 eval milestone 对齐较差。

实际平台可以同时支持：按 step、按时间、按里程碑保存。

## 21.5 保存哪些内容

不同 checkpoint 用途不同。

可以分三类：

### 21.5.1 Resume checkpoint

用于断点续训。

包含：

1. 模型参数。
2. 优化器状态。
3. scheduler。
4. random state。
5. dataloader state。
6. 分布式状态。

体积大，但恢复能力强。

### 21.5.2 Eval checkpoint

用于评估。

可以只包含模型权重和必要 config。

体积相对小，方便评估和对比。

### 21.5.3 Release checkpoint

用于发布或部署。

需要包含：

1. 模型权重。
2. tokenizer。
3. config。
4. generation config。
5. eval report。
6. metadata。

发布 checkpoint 要更严格校验。

平台应该区分这三类，不要所有 checkpoint 都保存完整训练状态。

## 21.6 保留策略

保留策略决定哪些 checkpoint 留下。

常见策略：

1. 保留最近 N 个。
2. 每隔 M step 保留一个 milestone。
3. 保留最优 eval checkpoint。
4. 保留最终 checkpoint。
5. 保留发布版本。
6. 删除失败或未提交 checkpoint。
7. 冷版本归档到低成本存储。

一个实用策略：

```text
最近 5 个 resume checkpoint：高性能存储保留。
每 10k step milestone：保留 30 天。
best eval checkpoint：长期保留。
release checkpoint：长期保留并加权限保护。
failed temporary checkpoint：7 天后清理。
```

保留策略要按任务类型区分。

预训练和 SFT 的策略不一定相同。

## 21.7 同步保存

同步保存是训练暂停，等待 checkpoint 写完。

适合：

1. 小模型。
2. checkpoint 不大。
3. 一致性要求高。
4. 平台实现早期。

优点：

1. 逻辑简单。
2. 一致性容易保证。
3. 恢复点明确。

缺点：

1. GPU 等 I/O。
2. step time 有尖刺。
3. 大模型保存时间长。
4. 多任务同时保存会造成拥塞。

同步保存适合作为基础能力，但大规模训练通常需要进一步优化。

## 21.8 异步保存

异步保存是把 checkpoint 写入从训练主循环中解耦。

常见做法：

1. 训练进程生成快照。
2. 后台线程或进程写入存储。
3. 训练继续往前跑。
4. 后台完成后提交 manifest。

优点：

1. 减少训练阻塞。
2. 降低 GPU 空等。
3. 提高训练吞吐。

缺点：

1. 实现复杂。
2. 需要额外内存或磁盘缓冲。
3. 可能和训练争抢 I/O。
4. 要处理失败和一致性。
5. 不能让半成品被恢复。

异步保存的关键是 snapshot 和 commit 语义。

## 21.9 异步保存的一致性问题

异步保存最大风险是保存内容不一致。

例如模型参数已经进入 step 1001，但 optimizer state 还是 step 1000。

防护方法：

1. 在确定边界生成一致 snapshot。
2. snapshot 完成后训练才能继续修改对应状态。
3. 写入临时目录。
4. 所有 shard 写完后校验 checksum。
5. 最后提交 manifest。
6. 恢复时只读取 committed checkpoint。

异步 checkpoint 不是简单把 `save()` 放到线程里。

## 21.10 分片保存策略

大模型 checkpoint 通常分片保存。

分片策略包括：

1. 按 rank 保存。
2. 按 tensor shard 保存。
3. 按 layer 保存。
4. 参数和 optimizer state 分开保存。
5. 模型权重和训练状态分开保存。

分片策略影响：

1. 保存速度。
2. 加载速度。
3. metadata 压力。
4. rank 数变化恢复。
5. 跨框架兼容性。

如果未来需要 resharding，分片格式要尽量带完整 metadata。

## 21.11 断点恢复策略

断点恢复要决定从哪个 checkpoint 恢复。

常见策略：

1. 从最新 committed checkpoint 恢复。
2. 从用户指定 checkpoint 恢复。
3. 从最近健康 checkpoint 恢复。
4. 从 best checkpoint 继续微调。
5. 从 milestone checkpoint 回滚。

恢复前要校验：

1. checkpoint 是否完整。
2. 配置是否兼容。
3. 数据版本是否可访问。
4. 并行策略是否匹配。
5. 权限是否允许。
6. 存储路径是否可读。

恢复后要记录事件：

```text
JobResumedFromCheckpoint(step=12000, checkpoint_id=ckpt_12000)
```

## 21.12 Checkpoint 与抢占联动

可抢占任务必须有 checkpoint 策略。

抢占前最好：

1. 发送 preemption notice。
2. 触发一次 checkpoint。
3. 等待 checkpoint committed。
4. 标记任务为 preempted。
5. 释放资源。

如果通知时间很短，可能来不及保存完整 checkpoint。

因此可抢占队列应该要求：

1. 周期性 checkpoint。
2. 最近 checkpoint 不超过某个时间阈值。
3. checkpoint 保存时间可控。

调度系统抢占时应该优先选择最近有 checkpoint 的任务，减少损失。

## 21.13 Checkpoint 与评估联动

很多训练平台会在 checkpoint 后触发评估。

流程：

```text
save checkpoint -> run eval -> record metrics -> mark best checkpoint
```

这样可以自动找到 best checkpoint。

注意：

1. 不一定每个 checkpoint 都要评估。
2. 评估会消耗推理资源。
3. 评估配置要版本化。
4. best checkpoint 需要保护。
5. eval 失败不一定代表训练失败。

训练平台要把 checkpoint 和 eval report 关联起来。

## 21.14 Checkpoint 与模型发布联动

发布模型通常来自某个 checkpoint。

发布前要做：

1. 权重转换。
2. tokenizer 和 config 校验。
3. 推理加载测试。
4. 评估报告检查。
5. 安全审核。
6. metadata 注册。
7. 权限标记。

发布 checkpoint 和训练 resume checkpoint 不同。

发布版本应该更稳定、更小、更可审计。

## 21.15 Checkpoint 监控和告警

Checkpoint 策略必须可观测。

监控指标：

1. save_duration。
2. load_duration。
3. checkpoint_size。
4. time_since_last_checkpoint。
5. save_failure_rate。
6. committed_count。
7. temporary_count。
8. storage_usage。
9. restore_success_rate。
10. preemption_without_recent_checkpoint_count。

告警：

1. 长时间未成功 checkpoint。
2. checkpoint 保存失败。
3. 保存耗时突然上升。
4. 存储容量接近上限。
5. 临时 checkpoint 堆积。
6. 恢复失败。

Checkpoint 没有监控，就不是真正可靠。

## 21.16 常见策略错误

错误一：所有 checkpoint 都保存完整训练状态。

会导致存储成本过高。应该区分 resume、eval、release。

错误二：只保留最新一个 checkpoint。

如果最新 checkpoint 损坏或训练已经发散，就没有回滚点。

错误三：异步保存没有 commit 语义。

可能恢复到半成品 checkpoint。

错误四：可抢占任务 checkpoint 太少。

抢占后损失大，用户体验差。

错误五：没有恢复测试。

保存成功不代表能恢复。

## 21.17 Checkpoint 策略审计指标与最小 demo

Checkpoint 策略不能只写成“每 1000 step 保存一次”。平台至少要能证明：保存频率符合 RPO，恢复耗时符合 RTO，保存阻塞比例可接受，resume / eval / release 三类 checkpoint 分清，异步保存有一致 snapshot 和 commit 语义，保留策略能控制成本，抢占、评估、发布、告警和恢复演练都能和 checkpoint 串起来。

可以把一次 checkpoint 策略审计样本写成：

```math
p_i=(r_i,f_i,s_i,a_i,m_i,h_i,u_i,q_i,e_i,v_i,l_i,c_i,o_i,d_i,z_i)
```

其中，`r_i` 是 RPO / RTO 目标，`f_i` 是保存触发策略，`s_i` 是保存内容分层，`a_i` 是异步保存策略，`m_i` 是 manifest / commit 语义，`h_i` 是分片元数据，`u_i` 是恢复选择策略，`q_i` 是抢占联动，`e_i` 是评估联动，`v_i` 是发布联动，`l_i` 是保留生命周期，`c_i` 是存储成本，`o_i` 是监控告警，`d_i` 是恢复演练，`z_i` 是最终策略门禁。

统一覆盖率可以写成：

```math
C_j=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[g_j(p_i)=1]
```

按 step 保存的最大丢失时间可以粗略写成：

```math
A_{\mathrm{step}}=\frac{I_{\mathrm{step}}T_{\mathrm{step}}}{60}
```

其中，`I_step` 是 step 间隔，`T_step` 是单 step 秒数。按时间保存的最大丢失时间就是 `A_time=I_time`。混合策略下，可以取更保守的触发上界：

```math
A_{\mathrm{lost}}=\min(A_{\mathrm{step}}, A_{\mathrm{time}})
```

如果有抢占通知，还要比较通知窗口和 checkpoint 完成时间：

```math
G_{\mathrm{preempt}}=\mathbf{1}\left[T_{\mathrm{notice}}\ge T_{\mathrm{snapshot}}+T_{\mathrm{commit}}\right]
```

同步保存的阻塞比例可以写成：

```math
R_{\mathrm{block}}=\frac{T_{\mathrm{block}}}{I_{\mathrm{step}}T_{\mathrm{step}}+T_{\mathrm{block}}}
```

`T_block` 是训练主循环真正停住等待 checkpoint 的时间。异步保存不是把 `T_block` 变成 0，而是把主循环阻塞从完整写入时间降低到 snapshot、buffer、metadata 或必要同步边界。

保留成本可以写成：

```math
K_{\mathrm{retain}}=\sum_{k=1}^{K}N_kS_kD_kP_k+K_{\mathrm{request}}+K_{\mathrm{ops}}
```

其中，`N_k` 是第 `k` 类 checkpoint 保留个数，`S_k` 是单个大小，`D_k` 是保留天数，`P_k` 是对应存储层的单位成本。

Checkpoint 策略门禁可以写成：

```math
G_{\mathrm{ckpt\_strategy}}=\mathbf{1}\left[A_{\mathrm{lost}}\le \mathrm{RPO} \land T_{\mathrm{restore}}\le \mathrm{RTO} \land R_{\mathrm{block}}\le \rho_{\mathrm{block}} \land K_{\mathrm{retain}}\le B_{\mathrm{storage}} \land \min_j C_j\ge \tau_j \land P_0=0\right]
```

下面这个 demo 不依赖第三方库。它用一条完整策略和 16 条坏样本，模拟平台如何审计 checkpoint 频率、保留、异步保存、抢占、评估、发布和恢复演练。

```python
from collections import OrderedDict


GATE_ORDER = [
    "rpo_rto_budget_defined",
    "save_trigger_policy",
    "checkpoint_scope_separation",
    "async_snapshot_consistency",
    "manifest_commit_integrity",
    "shard_metadata_resharding",
    "restore_selection_policy",
    "preemption_checkpoint_coupling",
    "evaluation_best_linkage",
    "release_checkpoint_readiness",
    "retention_tier_lifecycle",
    "storage_cost_budget",
    "checkpoint_monitoring_alerts",
    "restore_drill_coverage",
    "permission_release_protection",
    "checkpoint_strategy_gate",
]

REQUIRED_SCOPES = {
    "resume": {"model", "optimizer", "scheduler", "rng", "dataloader", "distributed", "config"},
    "eval": {"model", "config", "tokenizer", "metadata"},
    "release": {"model", "tokenizer", "config", "generation_config", "eval_report", "metadata"},
}
REQUIRED_TRIGGERS = {"step", "time", "milestone", "preemption"}
REQUIRED_ASYNC_CONTROLS = {
    "snapshot_boundary",
    "buffer_budget",
    "temp_path",
    "checksum",
    "manifest_commit",
    "committed_only",
    "failure_cleanup",
}
REQUIRED_MANIFEST_FIELDS = {"checkpoint_id", "global_step", "shards", "checksums", "state_kind", "status"}
REQUIRED_SHARD_METADATA = {"rank_map", "tensor_shards", "global_step", "format_version", "reshard_policy"}
REQUIRED_RESTORE_POLICY = {
    "latest_committed",
    "health_check",
    "config_compat",
    "data_access",
    "permission_check",
    "event_record",
}
REQUIRED_PREEMPTION_POLICY = {
    "notice_handler",
    "on_notice_checkpoint",
    "freshness_gate",
    "scheduler_uses_freshness",
}
REQUIRED_EVAL_LINKS = {"eval_after_milestone", "eval_config_version", "best_protection", "report_link"}
REQUIRED_RELEASE_LINKS = {
    "conversion_test",
    "tokenizer_config_check",
    "load_smoke",
    "eval_report_required",
    "security_review",
    "registry_metadata",
}
REQUIRED_MONITORING = {
    "save_duration",
    "load_duration",
    "checkpoint_age",
    "failure_rate",
    "temporary_count",
    "storage_usage",
    "restore_success",
    "preempt_without_recent",
}
REQUIRED_PERMISSIONS = {"release_protected", "rbac", "audit_event", "archive_policy"}
REQUIRED_RETENTION_KINDS = {"resume", "milestone", "best_eval", "release", "failed_temporary"}

TIER_PRICE_PER_GIB_DAY = {
    "hot": 0.004,
    "warm": 0.0015,
    "archive": 0.0004,
    "protected": 0.0025,
}


def has_all(values, required):
    return set(values) >= set(required)


def ratio(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def rounded(value):
    return round(value, 3)


def max_lost_minutes(case):
    step_minutes = case["step_interval"] * case["step_seconds"] / 60.0
    return min(step_minutes, case["time_interval_minutes"])


def blocking_ratio(case):
    interval_seconds = case["step_interval"] * case["step_seconds"]
    return ratio(case["main_loop_block_seconds"], interval_seconds + case["main_loop_block_seconds"])


def retention_cost(case):
    total = case.get("request_ops_cost", 0.0)
    for item in case["retention_items"]:
        price = TIER_PRICE_PER_GIB_DAY[item["tier"]]
        total += item["count"] * item["size_gib"] * item["days"] * price
    return total


def retention_kinds(case):
    return {item["kind"] for item in case["retention_items"]}


def scopes_are_separated(case):
    scopes = case["checkpoint_scopes"]
    return all(has_all(scopes.get(kind, set()), fields) for kind, fields in REQUIRED_SCOPES.items())


def audit_one(case):
    lost = max_lost_minutes(case)
    cost = retention_cost(case)
    gates = OrderedDict()
    gates["rpo_rto_budget_defined"] = (
        case["rpo_minutes"] > 0
        and case["rto_minutes"] > 0
        and lost <= case["rpo_minutes"]
        and case["restore_minutes"] <= case["rto_minutes"]
    )
    gates["save_trigger_policy"] = has_all(case["save_triggers"], REQUIRED_TRIGGERS)
    gates["checkpoint_scope_separation"] = scopes_are_separated(case)
    gates["async_snapshot_consistency"] = (
        not case["async_enabled"] or has_all(case["async_controls"], REQUIRED_ASYNC_CONTROLS)
    )
    gates["manifest_commit_integrity"] = has_all(case["manifest_fields"], REQUIRED_MANIFEST_FIELDS)
    gates["shard_metadata_resharding"] = has_all(case["shard_metadata"], REQUIRED_SHARD_METADATA)
    gates["restore_selection_policy"] = has_all(case["restore_policy"], REQUIRED_RESTORE_POLICY)
    gates["preemption_checkpoint_coupling"] = (
        has_all(case["preemption_policy"], REQUIRED_PREEMPTION_POLICY)
        and case["checkpoint_age_minutes"] <= case["preemption_freshness_minutes"]
    )
    gates["evaluation_best_linkage"] = has_all(case["eval_links"], REQUIRED_EVAL_LINKS)
    gates["release_checkpoint_readiness"] = has_all(case["release_links"], REQUIRED_RELEASE_LINKS)
    gates["retention_tier_lifecycle"] = has_all(retention_kinds(case), REQUIRED_RETENTION_KINDS)
    gates["storage_cost_budget"] = cost <= case["storage_budget"]
    gates["checkpoint_monitoring_alerts"] = has_all(case["monitoring"], REQUIRED_MONITORING)
    gates["restore_drill_coverage"] = (
        case["restore_drill"]["scheduled"]
        and case["restore_drill"]["pass_rate"] >= 0.99
        and case["restore_drill"]["last_days_ago"] <= 30
    )
    gates["permission_release_protection"] = has_all(case["permissions"], REQUIRED_PERMISSIONS)
    gates["checkpoint_strategy_gate"] = bool(case["strategy_gate"])

    return {
        "name": case["name"],
        "gates": gates,
        "max_lost_minutes": rounded(lost),
        "blocking_ratio": rounded(blocking_ratio(case)),
        "retention_cost": round(cost, 1),
        "restore_minutes": case["restore_minutes"],
        "retention_kinds": sorted(retention_kinds(case)),
    }


def clone(base, name, **changes):
    copied = {}
    for key, value in base.items():
        if isinstance(value, dict):
            copied[key] = {
                inner_key: set(inner_value) if isinstance(inner_value, set) else dict(inner_value)
                if isinstance(inner_value, dict)
                else inner_value
                for inner_key, inner_value in value.items()
            }
        elif isinstance(value, list):
            copied[key] = [dict(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, set):
            copied[key] = set(value)
        else:
            copied[key] = value
    copied["name"] = name
    copied.update(changes)
    return copied


complete_case = {
    "name": "complete_case",
    "rpo_minutes": 30,
    "rto_minutes": 20,
    "step_interval": 1000,
    "step_seconds": 1.5,
    "time_interval_minutes": 30,
    "restore_minutes": 14,
    "main_loop_block_seconds": 18,
    "save_triggers": REQUIRED_TRIGGERS,
    "checkpoint_scopes": REQUIRED_SCOPES,
    "async_enabled": True,
    "async_controls": REQUIRED_ASYNC_CONTROLS,
    "manifest_fields": REQUIRED_MANIFEST_FIELDS,
    "shard_metadata": REQUIRED_SHARD_METADATA,
    "restore_policy": REQUIRED_RESTORE_POLICY,
    "preemption_policy": REQUIRED_PREEMPTION_POLICY,
    "checkpoint_age_minutes": 18,
    "preemption_freshness_minutes": 30,
    "eval_links": REQUIRED_EVAL_LINKS,
    "release_links": REQUIRED_RELEASE_LINKS,
    "retention_items": [
        {"kind": "resume", "count": 5, "size_gib": 3200, "days": 7, "tier": "hot"},
        {"kind": "milestone", "count": 8, "size_gib": 800, "days": 30, "tier": "warm"},
        {"kind": "best_eval", "count": 2, "size_gib": 250, "days": 180, "tier": "warm"},
        {"kind": "release", "count": 1, "size_gib": 240, "days": 365, "tier": "protected"},
        {"kind": "failed_temporary", "count": 3, "size_gib": 100, "days": 7, "tier": "archive"},
    ],
    "request_ops_cost": 25.0,
    "storage_budget": 1500.0,
    "monitoring": REQUIRED_MONITORING,
    "restore_drill": {"scheduled": True, "pass_rate": 1.0, "last_days_ago": 14},
    "permissions": REQUIRED_PERMISSIONS,
    "strategy_gate": True,
}

cases = [
    complete_case,
    clone(complete_case, "rpo_rto_missing_bad", rpo_minutes=0, rto_minutes=0),
    clone(complete_case, "save_trigger_missing_bad", save_triggers={"step"}),
    clone(
        complete_case,
        "scope_mixed_bad",
        checkpoint_scopes={
            "resume": {"model"},
            "eval": {"model"},
            "release": {"model"},
        },
    ),
    clone(complete_case, "async_snapshot_missing_bad", async_controls={"temp_path", "checksum"}),
    clone(complete_case, "manifest_commit_missing_bad", manifest_fields={"checkpoint_id", "status"}),
    clone(complete_case, "shard_metadata_missing_bad", shard_metadata={"rank_map"}),
    clone(complete_case, "restore_policy_latest_bad", restore_policy={"latest_committed"}),
    clone(
        complete_case,
        "preemption_uncoupled_bad",
        preemption_policy={"notice_handler"},
        checkpoint_age_minutes=120,
    ),
    clone(complete_case, "eval_unlinked_bad", eval_links={"eval_after_milestone"}),
    clone(complete_case, "release_unready_bad", release_links={"registry_metadata"}),
    clone(
        complete_case,
        "retention_latest_only_bad",
        retention_items=[{"kind": "resume", "count": 1, "size_gib": 3200, "days": 7, "tier": "hot"}],
    ),
    clone(complete_case, "storage_cost_unbounded_bad", storage_budget=500.0),
    clone(complete_case, "monitoring_missing_bad", monitoring={"save_duration"}),
    clone(
        complete_case,
        "restore_drill_missing_bad",
        restore_drill={"scheduled": False, "pass_rate": 0.0, "last_days_ago": 999},
    ),
    clone(complete_case, "permission_release_open_bad", permissions={"archive_policy"}),
    clone(complete_case, "strategy_gate_missing_bad", strategy_gate=False),
]

audits = [audit_one(case) for case in cases]
metrics = OrderedDict()
for gate_name in GATE_ORDER:
    passed = sum(1 for audit in audits if audit["gates"][gate_name])
    metrics[gate_name] = rounded(ratio(passed, len(audits)))

failed_cases = [
    audit["name"]
    for audit in audits
    if not all(audit["gates"].values())
]
failed_gates = [name for name, value in metrics.items() if value < 1.0]
complete_audit = audits[0]

examples = {
    "max_lost_minutes": complete_audit["max_lost_minutes"],
    "restore_minutes": complete_audit["restore_minutes"],
    "blocking_ratio": complete_audit["blocking_ratio"],
    "retention_cost": complete_audit["retention_cost"],
    "retention_kinds": complete_audit["retention_kinds"],
}
smoke = {
    "complete_case_passes": all(complete_audit["gates"].values()),
    "caught_rpo_gap": "rpo_rto_missing_bad" in failed_cases,
    "caught_scope_gap": "scope_mixed_bad" in failed_cases,
    "caught_async_gap": "async_snapshot_missing_bad" in failed_cases,
    "caught_preemption_gap": "preemption_uncoupled_bad" in failed_cases,
    "caught_restore_drill_gap": "restore_drill_missing_bad" in failed_cases,
}

print("checkpoint_strategy_examples=", examples, sep="")
print("smoke=", smoke, sep="")
print("metrics=", dict(metrics), sep="")
print("hard_blocker_count=", len(failed_cases), sep="")
print("failed_cases=", failed_cases, sep="")
print("failed_gates=", failed_gates, sep="")
print("checkpoint_strategy_gate_pass=", all(value >= 0.99 for value in metrics.values()), sep="")
```

一组期望输出类似：

```text
checkpoint_strategy_examples={'max_lost_minutes': 25.0, 'restore_minutes': 14, 'blocking_ratio': 0.012, 'retention_cost': 1115.8, 'retention_kinds': ['best_eval', 'failed_temporary', 'milestone', 'release', 'resume']}
smoke={'complete_case_passes': True, 'caught_rpo_gap': True, 'caught_scope_gap': True, 'caught_async_gap': True, 'caught_preemption_gap': True, 'caught_restore_drill_gap': True}
metrics={'rpo_rto_budget_defined': 0.941, 'save_trigger_policy': 0.941, 'checkpoint_scope_separation': 0.941, 'async_snapshot_consistency': 0.941, 'manifest_commit_integrity': 0.941, 'shard_metadata_resharding': 0.941, 'restore_selection_policy': 0.941, 'preemption_checkpoint_coupling': 0.941, 'evaluation_best_linkage': 0.941, 'release_checkpoint_readiness': 0.941, 'retention_tier_lifecycle': 0.941, 'storage_cost_budget': 0.941, 'checkpoint_monitoring_alerts': 0.941, 'restore_drill_coverage': 0.941, 'permission_release_protection': 0.941, 'checkpoint_strategy_gate': 0.941}
hard_blocker_count=16
failed_cases=['rpo_rto_missing_bad', 'save_trigger_missing_bad', 'scope_mixed_bad', 'async_snapshot_missing_bad', 'manifest_commit_missing_bad', 'shard_metadata_missing_bad', 'restore_policy_latest_bad', 'preemption_uncoupled_bad', 'eval_unlinked_bad', 'release_unready_bad', 'retention_latest_only_bad', 'storage_cost_unbounded_bad', 'monitoring_missing_bad', 'restore_drill_missing_bad', 'permission_release_open_bad', 'strategy_gate_missing_bad']
failed_gates=['rpo_rto_budget_defined', 'save_trigger_policy', 'checkpoint_scope_separation', 'async_snapshot_consistency', 'manifest_commit_integrity', 'shard_metadata_resharding', 'restore_selection_policy', 'preemption_checkpoint_coupling', 'evaluation_best_linkage', 'release_checkpoint_readiness', 'retention_tier_lifecycle', 'storage_cost_budget', 'checkpoint_monitoring_alerts', 'restore_drill_coverage', 'permission_release_protection', 'checkpoint_strategy_gate']
checkpoint_strategy_gate_pass=False
```

这个 demo 想说明：checkpoint 策略的关键不是“保存成功”四个字，而是把 RPO/RTO、保存阻塞、异步一致性、分片元数据、恢复选择、抢占、评估、发布、留存、成本、权限、监控和恢复演练放进同一个策略门禁。大模型训练平台如果没有这张表，checkpoint 很容易从可靠性能力变成存储成本和恢复风险的来源。

## 21.18 面试中如何回答 checkpoint 策略

如果面试官问：

```text
大模型训练平台如何设计 checkpoint 策略？
```

可以这样回答：

```text
我会先按用途区分 checkpoint：resume checkpoint 用于断点续训，包含模型、优化器、scheduler、随机状态和 dataloader 状态；eval checkpoint 主要用于评估；release checkpoint 用于发布和推理，包含权重、tokenizer、config 和评估报告。

保存频率要根据 RPO/RTO 设计，可以支持按 step、按时间和 milestone 保存。保存太频繁会阻塞训练和增加存储成本，太少会导致故障后回退太多。

大模型 checkpoint 应采用分片保存，并用 manifest、checksum 和 commit 语义保证一致性。异步 checkpoint 可以减少训练阻塞，但必须先生成一致 snapshot，写入临时目录，校验后再标记 committed。

保留策略上，保留最近 N 个 resume checkpoint、周期性 milestone、best eval checkpoint 和 release checkpoint，临时失败 checkpoint 自动清理，冷版本归档到低成本存储。

调度层要和 checkpoint 联动：可抢占任务必须有周期 checkpoint，抢占前尽量触发保存，恢复时从最新 committed checkpoint 加载。平台还要监控 save/load duration、time since last checkpoint、失败率、存储用量和恢复成功率，并定期做恢复测试。
```

## 21.19 常见误区

误区一：checkpoint 越频繁越好。

频繁保存会拖慢训练并增加存储成本，要按 RPO/RTO 权衡。

误区二：checkpoint 只为故障恢复服务。

它还服务评估、回滚、抢占、发布和实验分析。

误区三：异步保存一定无损。

异步保存会带来一致性和资源竞争问题，需要 snapshot 和 commit 语义。

误区四：保留 latest 就够。

latest 可能损坏或对应坏训练状态，需要保留 milestone 和 best checkpoint。

误区五：能保存就能恢复。

必须做恢复测试和配置兼容校验。

## 21.20 面试题

### 题 1：Checkpoint 保存频率如何选择？

答：要根据可接受的恢复进度损失和恢复时间来选择，也就是 RPO/RTO。保存太频繁会增加 I/O 和存储成本，保存太少会导致故障后回退太多。可以支持按 step、按时间和 milestone 混合策略。

### 题 2：Resume checkpoint、eval checkpoint、release checkpoint 有什么区别？

答：Resume checkpoint 用于断点续训，包含完整训练状态；eval checkpoint 用于评估，通常只需要模型权重和必要配置；release checkpoint 用于发布和推理，需要权重、tokenizer、config、评估报告和 metadata。

### 题 3：异步 checkpoint 的关键风险是什么？

答：关键风险是不一致和半成品恢复。必须在一致边界生成 snapshot，写入临时目录，校验所有 shard 后提交 manifest，并且恢复时只读取 committed checkpoint。

### 题 4：可抢占训练任务为什么必须配 checkpoint？

答：抢占会中断任务，如果没有 checkpoint，训练进度会大量丢失。可抢占任务应周期性保存 checkpoint，抢占前尽量触发保存，恢复时从最新 committed checkpoint 继续。

### 题 5：为什么不能只保留最新 checkpoint？

答：最新 checkpoint 可能损坏，也可能对应训练发散后的状态。保留 milestone、best eval 和 release checkpoint 可以支持回滚、评估和发布。

## 21.21 小练习

练习一：设计一个 checkpoint 策略。

要求：给一个 70B SFT 任务设计保存频率、保留策略、best checkpoint 和 release checkpoint 规则。

练习二：设计异步 checkpoint 流程。

要求：画出 snapshot、写临时目录、checksum、manifest、commit 和失败清理流程。

练习三：设计抢占联动。

要求：说明可抢占任务收到 preemption notice 后如何保存 checkpoint，调度系统如何选择被抢占任务。

练习四：分析 checkpoint 成本失控。

要求：从保存频率、完整训练状态、保留策略、实验数量和冷存储归档角度分析。

## 21.22 本章小结

本章讲了 checkpoint 策略。

你需要掌握：

1. Checkpoint 策略要平衡训练成本、存储成本、恢复损失、评估需求和发布需求。
2. RPO/RTO 是设计保存频率的重要框架。
3. 可以按 step、按时间和 milestone 保存。
4. 要区分 resume、eval、release checkpoint。
5. 保留策略应包含最近 N 个、milestone、best eval、release 和临时失败清理。
6. 异步 checkpoint 需要一致 snapshot、临时目录、checksum、manifest 和 commit 语义。
7. 分片保存影响保存速度、加载速度、metadata 和 resharding。
8. 可抢占任务必须和 checkpoint 策略联动。
9. Checkpoint 应和评估、发布、监控和恢复测试打通。

下一章我们会讲训练容错：节点失败、通信失败、OOM 和自动重试。
