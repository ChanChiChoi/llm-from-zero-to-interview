# 第三章：大模型训练平台工程

## 3.1 训练平台解决的不是“替用户跑脚本”

一个训练脚本可以在个人机器上启动，但团队要同时运行几十到几千个任务时，平台必须回答更多问题：任务使用了哪份代码和数据？申请了什么资源？失败后能否恢复？结果能否复现？谁可以读取 checkpoint？一个任务是否长期占用资源却没有产出？

训练平台因此包含三个相互配合的平面：

```text
控制面：任务 API、配置校验、队列、权限、状态和策略
数据面：代码、数据、checkpoint、日志和 artifact 的传输与存储
执行面：容器、GPU、分布式 launcher、worker、监控和故障处理
```

控制面决定“应该运行什么”，执行面负责“在哪里运行”，数据面保证“运行时拿到什么以及产出什么”。把三者混在一个提交脚本里，短期看起来简单，长期会造成不可审计、难恢复和难扩展。

## 3.2 任务对象和状态机

平台首先需要一个稳定的任务对象。最小字段可以包括：

```text
job_id, owner, tenant, code_ref, data_ref, image_ref,
resource_spec, command, env, checkpoint_policy,
created_at, status, attempt, output_refs
```

任务状态不能只用一个 `running` 布尔值。一个可操作的状态机至少包括：

```text
SUBMITTED -> VALIDATING -> QUEUED -> ALLOCATING -> RUNNING
RUNNING -> SUCCEEDED
RUNNING -> FAILED -> RETRYING -> QUEUED
RUNNING -> PREEMPTED -> RESTORING -> QUEUED
SUBMITTED/QUEUED/RUNNING -> CANCELED
```

状态转移要有 owner 和事件记录。例如，`RUNNING -> FAILED` 应记录 exit code、节点、rank、最近 checkpoint 和平台判定原因；`FAILED -> RETRYING` 只能在重试策略允许且任务输入仍然可用时发生。

状态机的好处是把“用户命令失败”和“平台分配失败”区分开。前者可能需要修复代码或数据，后者可能需要换节点、重试网络或释放资源；如果两者都显示成 `failed`，自动化恢复会做出错误决策。

## 3.3 配置和 artifact 的不可变引用

可复现训练不应只保存一段 YAML。一次 run 的有效输入可以抽象成：

```math
R=(C,D,M,E,S,H)
```

`C` 是代码 revision，`D` 是数据版本，`M` 是镜像和依赖，`E` 是训练配置，`S` 是资源与启动方式，`H` 是硬件和软件环境。若任意一项没有版本或 checksum，两个“相同配置”的 run 可能实际并不相同。

例如，数据路径仍叫 `latest`，但目录内容已被覆盖；镜像 tag 仍叫 `train:stable`，实际 digest 已经改变；命令相同但 tokenizer 文件来自不同 revision。这些都会让 loss 曲线和最终能力无法复现。

平台应在提交时解析并锁定引用：代码 commit、容器 digest、数据 manifest、tokenizer checksum、基础模型版本和依赖锁文件。展示给用户的 friendly name 可以保留，但执行面应使用不可变 ID。

## 3.4 资源申请和准入

资源规格至少要描述 GPU 类型、数量、显存、节点数、拓扑要求、CPU、内存、临时盘、网络和最长运行时间。准入逻辑可以写成：

```math
Admit(j)=I(resource_j\subseteq available)
\land I(usage_{tenant}+cost_j\le quota_{tenant})
\land I(policy(owner,resource_j)=allow)
```

这里的 `cost_j` 不只代表 GPU 数，也可以是预计 GPU 小时或优先级权重。若任务申请 8 张卡但实际只使用 2 张，平台需要用 profiling 或历史数据识别浪费，而不是因为申请成功就认为资源利用合理。

调度器应在分配前检查数据位置和并行拓扑。一个需要 tensor parallel 的任务被拆到跨机慢链路上，可能比排队等待更糟；一个需要高速本地数据的任务被分配到没有 cache 的节点，也会在运行后表现为“GPU 训练很慢”。

## 3.5 启动分布式训练

launcher 的职责不是简单执行用户命令，而是为每个 worker 建立一致的 world size、rank、master address、端口、环境变量和失败传播机制。启动后平台要确认：

1. 所有 rank 都拿到了相同的代码、配置和 tokenizer。
2. `WORLD_SIZE`、`RANK`、`LOCAL_RANK` 与实际进程数一致。
3. NCCL/CUDA/驱动版本和 GPU 能力满足模型要求。
4. rank 0 的 checkpoint 保存路径不会被多个任务冲突使用。
5. 任意 rank 失败都能让整个任务进入明确的失败或恢复状态。

如果只监控 rank 0，其他 rank hang 时平台可能误以为任务仍在训练。执行面应把 worker heartbeat、step counter、通信异常和节点事件汇聚到 job 状态。

## 3.6 日志、指标和 checkpoint

训练指标至少分为四类：

| 类别 | 示例 | 用途 |
| --- | --- | --- |
| 优化指标 | loss、grad norm、learning rate、throughput | 判断训练是否收敛和稳定 |
| 系统指标 | GPU memory、utilization、通信时间、I/O wait | 定位资源瓶颈 |
| 状态指标 | step、epoch、data position、attempt | 支持恢复和复现 |
| 质量指标 | validation loss、任务评测、回归样例 | 判断 checkpoint 是否可用 |

checkpoint 不能只保存权重。若要从 step `s` 继续训练，通常还需要 optimizer state、scheduler state、AMP scaler、随机数状态、数据游标、模型配置和 tokenizer。恢复正确性可以用一个小实验验证：从同一 checkpoint 直接继续和保存后重新加载继续，比较若干步的 loss、参数 checksum 和随机样本。

保存频率的粗略成本是：若 checkpoint 大小为 `S_ckpt`，保存频率为每 `K` step，step 时间为 `T_step`，则平均 I/O 占用比例近似为：

```math
\rho_{io}\approx\frac{S_{ckpt}/B_{storage}}{K\cdot T_{step}}
```

异步保存可以降低训练线程阻塞，但会增加一致性和失败恢复复杂度。只有当 checkpoint 内容、写入完成事件和恢复点语义明确时，异步保存才是可靠优化。

## 3.7 容错和幂等重试

平台需要区分可重试故障与不可重试故障。节点掉电、网络瞬断、临时对象存储错误可能适合重试；数据 schema 错误、显存需求错误、loss 已经 NaN 且配置不变，盲目重试只会浪费资源。

每次 attempt 要有独立的事件和输出目录，checkpoint 写入要使用临时路径加原子完成标记，避免恢复到半文件。一个简单的 artifact 提交协议是：

```text
write chunks -> fsync/complete upload -> write manifest -> publish commit marker
```

恢复时只接受 manifest 和 commit marker 都存在且 checksum 通过的版本。重试任务还要保证外部副作用幂等，例如指标上报不能重复计入正式实验，模型注册不能把失败 attempt 错误标成 production candidate。

## 3.8 权限与租户边界

训练平台的权限至少涉及任务提交、资源使用、数据读取、日志查看、checkpoint 读取、模型注册和取消他人任务。最小权限原则要求这些动作分开授权，不能因为用户能提交任务就默认能够读取整个对象存储桶。

日志尤其容易泄露密钥、用户数据和 prompt。平台应在采集端做字段分级、脱敏和 retention policy；访问审计要记录 actor、resource、action、decision、reason 和 request ID。管理员查看敏感日志也应留下审计事件。

## 3.9 训练平台的验收表

一个最小可用平台至少应通过以下验收：

1. 给定固定 revision，任务能在新节点恢复并得到一致的输入 manifest。
2. 任意 worker 失败后，平台能识别故障、释放资源并按策略恢复或终止。
3. 任务状态、日志、指标和 checkpoint 可以用 job ID 关联查询。
4. 两个租户不能读取彼此的数据、日志和 artifact。
5. quota、优先级和抢占策略在压力测试下不出现永久饥饿。
6. 失败 attempt 不会覆盖成功 attempt 的正式产物。

面试中回答“如何设计训练平台”时，重点应放在状态机、不可变 artifact、资源准入、分布式启动、checkpoint 一致性、容错、权限和可观测性，而不是只罗列 Kubernetes、Ray 或 DeepSpeed 名称。

## 3.10 训练平台的可复现状态机

任务状态、artifact 状态和资源状态要分开。训练任务可能已经提交但权重 artifact 尚未完整，worker 可能失败但 checkpoint 已经成功；把它们压成一个 running/failed 字段会造成错误重试。

每次重试都要绑定 run id、数据版本、代码 revision、资源配置和 checkpoint。平台验收不仅是“任务能跑”，还包括暂停、恢复、取消、失败重试和审计。

## 3.11 训练平台的控制面与数据面

控制面负责提交、排队、配额、版本、权限、状态和重试；数据面负责 GPU、网络、存储、数据加载、通信和实际训练。把两者混在一个脚本里，短实验可以运行，长期训练却难以审计和恢复。

一个训练 job 至少要绑定代码 revision、数据 manifest、模型 config、环境镜像、资源请求、随机种子、checkpoint 策略和评估配置。控制面状态变更要能回放，数据面指标要能定位到具体 job 和 rank。

## 3.12 checkpoint 和失败恢复

checkpoint 不只是模型权重，还可能包含 optimizer、scheduler、scaler、随机数、数据迭代位置和并行拓扑。恢复一致性取决于这些对象是否一起保存，或是否明确允许重新计算。

训练平台要区分可重试失败、数据坏 shard、节点故障、通信超时和代码错误。盲目重试会重复副作用、浪费 GPU 或掩盖 deterministic bug；重试策略应带上限、退避和失败原因。

## 3.13 训练平台的可观测验收条件

训练吞吐不能只看 tokens/s。还要监控 loss、梯度、学习率、GPU 利用率、通信等待、数据加载、checkpoint 时间、恢复成功率和验证集回归。长训练中一个短暂的 NaN 或数据分布变化可能在数小时后才显现。

评估、checkpoint 和发布最好使用同一条 lineage。这样能从线上模型回到训练 job、数据版本、代码和环境，而不是只知道一个权重文件名。

## 3.14 一个训练任务的全生命周期

任务提交时，控制面先冻结代码、数据 manifest、模型 config、镜像、资源请求和策略版本，生成不可变 `run_id`。调度器分配资源后，执行面创建 `attempt_id`，启动 launcher，并把每个 rank 的 heartbeat、step、数据位置和错误汇聚到任务状态。checkpoint、评估和模型注册是独立 artifact 状态，不应由一个 `job_status` 字段代替。

例如训练已经达到 step 20K，但 step 18K 的 checkpoint 才完整写入；此时 worker 失效，平台应把任务标记为 `RESTORABLE`，恢复到 18K，并明确重复计算窗口，而不是显示“从 20K 继续”。如果 checkpoint manifest 损坏，任务应进入 `FAILED_ARTIFACT`，而不是无限重试同一个坏文件。

## 3.15 训练平台的控制面/数据面验收

控制面负责声明和决策，数据面负责实际资源和文件。两者之间要有版本化事件：`allocation_granted`、`worker_started`、`checkpoint_committed`、`evaluation_finished`、`attempt_failed`。事件中包含 job、attempt、revision、resource allocation 和 policy，恢复时由事件与实际环境交叉验证。

可以把一次训练 run 的可复现验收条件写成：

```math
G_{\mathrm{run}}
=G_{\mathrm{code}}
 G_{\mathrm{data}}
 G_{\mathrm{environment}}
 G_{\mathrm{checkpoint}}
 G_{\mathrm{evaluation}}
```

任一项为零，结果最多作为探索性 artifact，不能直接注册为生产模型。任务状态为成功也不代表评估成功、权限审计完成或产物可恢复。

故障演练要覆盖 rank hang、节点被抢占、对象存储返回部分内容、日志系统不可用、quota 改变、用户取消和模型注册失败。每个演练记录检测延迟、资源释放时间、恢复点、重复计算量和最终 artifact 是否仍可追溯。

## 3.16 训练平台的恢复语义

任务状态、attempt 状态和模型 artifact 状态必须分开。任务可能因为节点故障失败，但 checkpoint 已经完整；也可能训练进程退出码为零，但评估、manifest 或模型注册失败。恢复逻辑不能把“进程重新启动”当成“实验从正确状态继续”。

一个可审计的 checkpoint manifest 至少包含：model revision、optimizer、scheduler、梯度累积状态、随机数状态、数据 shard 和 offset、global step、token count、代码/容器版本、校验和、保存完成标记和权限域。缺失数据游标时，恢复可能重复或跳过样本；缺失 RNG 时，实验虽能继续，随机轨迹却不可复现。

## 3.17 worked example：训练进程成功不等于实验成功

假设训练进程在 step `10000` 正常退出，checkpoint 文件都存在，但评估任务因数据版本找不到而失败。平台若只看训练退出码，会把产物注册为可发布；正确状态应该是 `trained`，而不是 `evaluated` 或 `registered`。

可以把发布条件表达为：

```math
G_{\mathrm{release}}
=G_{\mathrm{checkpoint}}
 G_{\mathrm{data\ lineage}}
 G_{\mathrm{evaluation}}
 G_{\mathrm{security}}
 G_{\mathrm{registry}}.
```

任一验收条件未通过，artifact 可以保留为探索结果，但不能进入默认路由。控制面应让用户看到具体缺失项，而不是返回一个含义不清的“任务完成”。

## 3.18 平台故障演练与指标

训练平台至少要演练 rank hang、节点被抢占、对象存储部分写入、数据 shard 过期、日志系统不可用、配额变更和用户取消。每次演练记录发现时间、停止时间、资源释放时间、最近可恢复 step、重复计算量、恢复后验证结果和人工介入次数。

训练平台的核心指标应按控制面、数据面和执行面分开：控制面看排队和准入，数据面看读取吞吐、cache hit 和数据错误，执行面看 step time、通信占比、GPU 利用率和 checkpoint。把所有指标汇成一个“成功率”，会掩盖最需要修复的边界。

## 3.19 训练平台的取舍条件

训练平台把一次实验变成可提交、可调度、可恢复、可审计的资源对象。它的核心不是界面，而是控制面、数据面和执行面之间的契约；任务、attempt、checkpoint、评估和注册状态必须能够分别恢复和审计。

复习问题：

1. 为什么 `latest` 数据路径会破坏复现？
2. checkpoint 为什么需要保存 optimizer 和数据游标？
3. 哪些失败适合自动重试，哪些失败不适合？
4. 如何证明多租户平台没有把日志权限当成任务权限？

本章的接口边界参考 [Kubernetes Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/)、[PyTorch Distributed](https://docs.pytorch.org/docs/stable/distributed.html)、容器运行时、DeepSpeed/Megatron 启动约定、对象存储一致性和主流实验追踪系统的公开资料；具体字段和实现依赖平台版本，不应当作唯一标准。

## 3.20 训练任务 API 的输入与输出

训练平台的提交 API 不应只接收一个脚本路径。它需要冻结代码 revision、数据 manifest、模型 config、tokenizer、镜像、资源请求、并行拓扑、随机种子、评估配置、checkpoint 策略和权限域。平台返回的也不应只有一个 job id，而应包括不可变 `run_id`、`attempt_id`、状态查询地址和 artifact 预期。

任务对象可以分成声明与执行两层：声明描述用户想要的实验，执行描述某次资源分配和启动尝试。一次 job 可能有多个 attempt，某个 attempt 失败并不意味着已写入的 checkpoint 无效；某次训练成功也不意味着评估和注册完成。API 这样设计，恢复和审计才有稳定的实体。

## 3.21 数据面启动的顺序

分布式训练启动时，控制面先锁定资源和配置，数据面准备镜像、权重、数据 shard 和通信地址，执行面再启动 rank。任何一步失败都要回写具体状态；不能让 launcher 一直等待，把镜像拉取慢、网络不可达和 rank hang 混成同一种超时。

启动后的 heartbeat 应包含 global step、token count、data offset、GPU/通信状态和最近 checkpoint。单纯的进程存活只能说明 PID 还在，不能说明训练仍然有效。平台要在 loss/梯度/吞吐异常时标记可疑，而不是等进程退出。

## 3.22 Checkpoint 提交是事务

checkpoint 可以分为 `writing`、`verified`、`committed` 和 `expired`。写入分片完成后先校验 checksum 和 shape，再生成 manifest，最后用原子标记提交。恢复只接受 `committed` 状态，避免读到半写入的 optimizer 或缺失的随机状态。

设训练进度为 `s`，最近完整 checkpoint 为 `c`，故障后的重复计算窗口是：

```math
W_{\mathrm{recompute}}=s-c.
```

更频繁 checkpoint 减少 `W_recompute`，却增加 I/O 和训练停顿；异步写盘减少停顿，却扩大可恢复状态滞后。平台需要根据任务价值、存储带宽和故障率选择策略，并在成本报告中计入重算 token。

## 3.23 多租户训练平台的隔离边界

租户隔离不只是 GPU quota。还包括数据和 checkpoint 读权限、镜像/缓存共享、日志与 trace、网络出口、secret、评估样本和模型注册。共享缓存可以降低启动时间，却必须按内容 hash、权限域和加密策略设计，不能因为文件名相同就让不同租户复用。

故障恢复也不能越过租户边界。worker 迁移、checkpoint 读取、人工排障和日志回看都应携带主体身份和用途；平台管理员可以获得运维权限，但不应默认获得业务原文。审计记录要区分用户动作、平台自动动作和人工 break-glass 动作。

## 3.24 训练平台的最小验收闭环

验收不应只跑一个成功任务。至少要覆盖：数据 manifest 不存在、镜像拉取失败、rank hang、节点抢占、checkpoint 半写入、对象存储短暂失败、用户取消、评估失败、模型注册失败和权限不足。每个场景都要验证状态、告警、资源释放、可重试性、恢复点和最终 artifact。

只有当训练任务能够被明确提交、准确观察、从正确 checkpoint 恢复、在失败时停止副作用，并把代码/数据/环境/评估串入血缘，它才是平台化训练，而不是把脚本放进一个更大的队列。

## 3.25 小结

训练平台的核心是可提交、可观察、可恢复和可审计的任务状态机。配置、数据、镜像、checkpoint、评估和权限必须有不可变引用；故障时既要释放资源，也要保护租户和 artifact 一致性。平台化不是增加一个 launcher，而是把实验生命周期变成可验证的生产流程。
