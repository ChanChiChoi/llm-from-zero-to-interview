# 第 45 章 训练故障定位：loss 异常、hang、OOM、通信慢和 I/O 慢

上一章讲了 AI Infra 可观测性总览。本章进入具体故障定位：训练任务出问题时，如何系统排查。

大模型训练任务通常运行时间长、成本高、节点多、依赖复杂。一次训练故障可能浪费数百张 GPU 数小时。因此，训练故障定位是 AI Infra 工程师必须掌握的能力。

先记住一句话：

> 训练故障定位不要先猜模型或框架，而要按数据、代码、配置、资源、通信、存储和平台链路逐层排查。

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

## 45.21 常见误区

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

## 45.22 面试常见追问

问题一：训练 loss NaN 怎么排查？

可以回答：先定位 NaN 出现 step，检查数据 batch、学习率、gradient norm、precision、loss scale、logits/grad 是否 inf，再尝试降学习率、gradient clipping、BF16、保存异常 batch 做最小复现。

问题二：分布式训练 hang 怎么排查？

可以回答：看所有 rank 最后一条日志、step 是否推进、GPU 利用率、NCCL 日志、dataloader 状态、节点事件，重点找最早异常 rank，而不是最后 timeout 的 rank。

问题三：GPU 利用率低可能是什么原因？

可以回答：可能是数据读取慢、CPU preprocessing、batch 太小、通信等待、checkpoint/eval 太频繁、负载不均衡或 kernel 效率低，需要拆分 step time。

问题四：checkpoint resume 后 loss 跳变怎么办？

可以回答：检查是否恢复了 optimizer、scheduler、global step、random state 和 dataloader state，确认 config、数据版本和学习率曲线一致。

## 45.23 小练习

1. Loss NaN 常见原因有哪些？
2. 为什么小数据 overfit 测试能帮助排查训练问题？
3. 分布式训练 hang 时为什么要看所有 rank 日志？
4. GPU OOM 发生在 forward 和 optimizer step，排查方向有什么不同？
5. 通信慢需要看哪些指标？
6. I/O 慢为什么会导致 GPU 利用率低？
7. 完整 checkpoint resume 需要恢复哪些状态？
8. 如何设计 TrainingJob 故障排查页面？

## 45.24 本章小结

本章讲了训练故障定位。

你需要记住：

1. 训练故障要按任务事件、日志、指标、资源、数据、通信、checkpoint 和配置逐层排查。
2. Loss 异常可能来自数据、标签、tokenizer、学习率、precision、梯度、代码或 resume 状态。
3. Hang 通常需要看所有 rank 日志和分布式通信状态，不能只看最后报错。
4. OOM 要区分发生阶段，优化手段不只有减 batch。
5. GPU 利用率低常常是数据、通信、checkpoint 或调度问题。
6. 完整可观测性和实验追踪能显著降低训练故障定位成本。

下一章我们会讲推理故障定位：TTFT、TPOT、吞吐下降和错误率上升。
