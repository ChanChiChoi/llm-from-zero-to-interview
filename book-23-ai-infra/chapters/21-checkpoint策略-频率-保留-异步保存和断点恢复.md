# 第 21 章 Checkpoint 策略：频率、保留、异步保存和断点恢复

第 10 章我们讲过 checkpoint 的存储、加载、恢复和生命周期管理。本章回到训练平台工程视角，重点讲 checkpoint 策略：训练时应该多久保存一次？保留哪些？异步保存怎么做？抢占和失败恢复如何联动？

Checkpoint 策略不是越频繁越好，也不是越省越好。它是在训练成本、存储成本、恢复损失、评估需求和平台复杂度之间做平衡。

先记住一句话：

> Checkpoint 策略的目标，是用可接受的 I/O 和存储成本，把训练失败后的损失控制在可接受范围内，并支持评估、回滚、抢占和发布。

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

## 21.17 面试中如何回答 checkpoint 策略

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

## 21.18 常见误区

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

## 21.19 面试题

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

## 21.20 小练习

练习一：设计一个 checkpoint 策略。

要求：给一个 70B SFT 任务设计保存频率、保留策略、best checkpoint 和 release checkpoint 规则。

练习二：设计异步 checkpoint 流程。

要求：画出 snapshot、写临时目录、checksum、manifest、commit 和失败清理流程。

练习三：设计抢占联动。

要求：说明可抢占任务收到 preemption notice 后如何保存 checkpoint，调度系统如何选择被抢占任务。

练习四：分析 checkpoint 成本失控。

要求：从保存频率、完整训练状态、保留策略、实验数量和冷存储归档角度分析。

## 21.21 本章小结

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
