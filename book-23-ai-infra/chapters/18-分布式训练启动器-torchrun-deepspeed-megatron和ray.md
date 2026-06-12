# 第 18 章 分布式训练启动器：torchrun、DeepSpeed、Megatron 和 Ray

上一章讲了训练任务提交系统。本章讲任务提交后真正启动分布式训练时会遇到的问题：如何把一个训练脚本在多张 GPU、多台机器上正确启动起来？

很多同学以为分布式训练启动就是“多开几个进程”。真实情况复杂得多：每个进程需要知道自己的 rank、world size、master 地址、通信端口、GPU 绑定、并行策略、环境变量和失败处理方式。

先记住一句话：

> 分布式训练启动器的核心作用，是把平台分配的资源转换成训练框架能理解的进程拓扑和通信环境。

## 18.0 本讲资料边界与第二轮精修口径

本讲第二轮精修时，按官方资料校准以下边界：

1. PyTorch `torchrun` 的稳定口径是启动分布式 PyTorch 训练进程，并通过 `--nnodes`、`--nproc-per-node`、rank、world size 和 rendezvous 信息组织 worker。
2. DeepSpeed launcher 的稳定口径是结合 hostfile、节点数、每节点 GPU 数和 DeepSpeed 配置启动分布式训练，同时训练优化能力主要来自 ZeRO、offload、混合精度等配置。
3. Megatron 体系的重点不是“另一个启动命令”，而是 tensor parallel、pipeline parallel、data parallel、sequence parallel 等并行维度和训练脚本参数的一致性。
4. Ray Train 的稳定口径是把训练函数运行在一组 worker 上，并通过 scaling config、资源声明和运行时编排管理分布式任务。
5. NCCL 相关环境变量只作为通信 runtime 配置入口，例如 `NCCL_SOCKET_IFNAME`、`NCCL_IB_DISABLE` 等；是否真正走到期望网卡、RDMA 或拓扑路径，还要靠运行时日志、网络指标和连通性检查验证。

因此，本章不把某个云平台、某个训练 operator、某个内部 launcher 或某个版本的私有参数写成通用标准。这里讨论的是训练平台最稳定的抽象：资源分配、rank 拓扑、rendezvous、GPU 绑定、launcher adapter、网络、日志、失败分类、弹性恢复和审计门禁。

## 18.1 为什么需要启动器

单机单卡训练可以直接运行：

```bash
python train.py
```

单机多卡可能需要：

```bash
torchrun --nproc_per_node=8 train.py
```

多机多卡就更复杂：

1. 哪台机器是 master？
2. master_addr 是什么？
3. master_port 是什么？
4. 每台机器有多少进程？
5. 每个进程对应哪张 GPU？
6. 当前进程 rank 是多少？
7. world_size 是多少？
8. 通信 backend 用什么？
9. 如果某个 worker 失败怎么办？

启动器就是为了解决这些问题。

## 18.2 分布式训练的基本概念

### 18.2.1 Rank

Rank 是分布式训练中每个进程的编号。

例如 world size 是 8，就有 rank 0 到 rank 7。

通常 rank 0 会负责：

1. 打印主日志。
2. 保存部分 metadata。
3. 初始化一些全局状态。
4. 汇总指标。

但不要假设所有逻辑都只能 rank 0 做，不同框架会有不同约定。

### 18.2.2 Local Rank

Local rank 是当前节点内的进程编号。

如果一台机器有 8 张 GPU，通常 local rank 0 到 7 分别绑定 GPU 0 到 7。

训练脚本常用 local rank 设置当前进程使用哪张 GPU。

### 18.2.3 World Size

World size 是全局进程数量。

例如 4 台机器，每台 8 个进程：

```text
world_size = 4 * 8 = 32
```

写成统一公式就是：

```math
W_{\mathrm{world}}=N_{\mathrm{node}}P_{\mathrm{node}}
```

其中，`N_node` 是节点数，`P_node` 是每个节点启动的训练进程数。很多初始化卡住的问题，本质上都是平台分配出来的 `W_world`、launcher 参数里的 `world_size` 和训练脚本读到的环境变量不一致。

### 18.2.4 Master Addr 和 Master Port

分布式进程需要一个 rendezvous 地址来互相发现。

通常由 rank 0 所在节点提供 master_addr 和 master_port。

如果地址或端口配置错误，训练会卡在初始化阶段。

## 18.3 启动器要设置哪些环境变量

常见环境变量包括：

1. `RANK`。
2. `LOCAL_RANK`。
3. `WORLD_SIZE`。
4. `MASTER_ADDR`。
5. `MASTER_PORT`。
6. `CUDA_VISIBLE_DEVICES`。
7. `NCCL_DEBUG`。
8. `NCCL_SOCKET_IFNAME`。
9. `NCCL_IB_DISABLE`。
10. `OMP_NUM_THREADS`。

不同框架还可能需要自己的环境变量。

训练平台的 launcher 要负责把资源分配结果转换成这些变量。

## 18.4 torchrun

`torchrun` 是 PyTorch 官方推荐的分布式训练启动工具。

它常用于启动 PyTorch Distributed Data Parallel 或 FSDP 任务。

单机多卡示例：

```bash
torchrun \
  --nproc_per_node=8 \
  train.py --config configs/sft.yaml
```

多机多卡示例：

```bash
torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=10.0.0.1 \
  --master_port=29500 \
  train.py --config configs/pretrain.yaml
```

torchrun 的优点：

1. PyTorch 原生支持。
2. 简单直接。
3. 适合 DDP / FSDP。
4. 社区使用广泛。

挑战：

1. 多机参数容易配置错。
2. 需要平台分配 node_rank。
3. 容错能力需要额外设计。
4. 复杂并行策略需要训练代码配合。

## 18.5 DeepSpeed Launcher

DeepSpeed 提供训练优化和分布式启动能力。

它常用于 ZeRO、offload、大模型训练和微调。

DeepSpeed 启动示例：

```bash
deepspeed \
  --num_nodes 4 \
  --num_gpus 8 \
  train.py \
  --deepspeed \
  --deepspeed_config ds_config.json
```

DeepSpeed 配置通常包含：

1. ZeRO stage。
2. optimizer offload。
3. parameter offload。
4. fp16 / bf16。
5. gradient accumulation。
6. batch size。
7. communication options。

DeepSpeed 的优势：

1. ZeRO 生态成熟。
2. 降低显存压力。
3. 支持大模型训练。
4. 配置化程度高。

挑战：

1. 配置复杂。
2. 版本兼容要注意。
3. 和 HuggingFace / Megatron 等组合时排查难。
4. offload 会引入 I/O 或 PCIe 瓶颈。

## 18.6 Megatron 启动方式

Megatron 体系常用于大模型预训练，尤其强调模型并行。

Megatron 常见参数：

1. tensor model parallel size。
2. pipeline model parallel size。
3. data parallel size。
4. sequence parallel。
5. micro batch size。
6. global batch size。
7. distributed backend。

一个简化启动示例：

```bash
torchrun \
  --nnodes=8 \
  --nproc_per_node=8 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  pretrain_gpt.py \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 4 \
  --micro-batch-size 1 \
  --global-batch-size 512
```

Megatron 的重点不是一个单独 launcher，而是并行配置和训练脚本体系。

平台需要校验：

```math
W_{\mathrm{world}}=T_{\mathrm{tp}}P_{\mathrm{pp}}D_{\mathrm{dp}}
```

其中，`T_tp` 是 tensor parallel size，`P_pp` 是 pipeline parallel size，`D_dp` 是 data parallel size。这里故意不用同一个 `P` 表示 pipeline 和 process，避免面试时变量混乱。

如果使用 gradient accumulation，还要校验 global batch size：

```math
B_{\mathrm{global}}=B_{\mathrm{micro}}A_{\mathrm{grad}}D_{\mathrm{dp}}
```

其中，`B_micro` 是每个 data-parallel rank 上的 micro batch size，`A_grad` 是 gradient accumulation steps。并行维度或 batch 公式不一致，轻则启动失败，重则训练吞吐、loss 曲线和学习率 scaling 全部失真。

## 18.7 Ray

Ray 是分布式计算框架，可以用于训练、数据处理、强化学习、推理服务和 Agent 系统。

在训练平台中，Ray 常用于：

1. 分布式任务编排。
2. RLHF rollout。
3. 数据处理。
4. 分布式训练封装。
5. 多阶段 pipeline。
6. 弹性任务。

Ray 的优势：

1. 任务和 actor 模型灵活。
2. 适合复杂 pipeline。
3. 适合训练和推理混合任务。
4. 与 RLHF / Agent 场景结合较多。

挑战：

1. 资源调度和 K8s 调度需要协调。
2. Ray 集群自身有控制面。
3. 调试复杂。
4. 和传统 torchrun 思维不同。

Ray 更像分布式应用运行框架，不只是训练 launcher。

## 18.8 MPI 和其他启动方式

除了 torchrun、DeepSpeed、Megatron、Ray，还有 MPI、Slurm srun、自研 launcher 等方式。

MPI 常见于 HPC 体系。

Slurm 常见于传统 HPC 集群。

自研 launcher 常见于大型 AI 团队，用于统一环境变量、拓扑、日志、容错和平台集成。

平台设计时不要把自己绑死在一个 launcher 上。

更好的方式是抽象 launcher interface。

## 18.9 平台如何统一封装启动器

训练平台可以定义统一字段：

```yaml
distributed:
  launcher: torchrun
  nnodes: 4
  nproc_per_node: 8
  env:
    NCCL_DEBUG: INFO
```

平台根据 launcher 类型生成实际启动命令。

例如：

1. torchrun adapter。
2. deepspeed adapter。
3. megatron adapter。
4. ray adapter。
5. mpi adapter。

这样用户面对统一 TrainingJob schema，平台内部适配不同启动器。

这叫 launcher abstraction。

## 18.10 启动器和调度系统的关系

调度系统负责分配资源。

启动器负责把资源转成进程。

调度系统输出：

1. 节点列表。
2. 每个节点 GPU 数。
3. master 节点。
4. node_rank。
5. 网络信息。
6. 挂载信息。

启动器使用这些信息生成：

1. `MASTER_ADDR`。
2. `MASTER_PORT`。
3. `RANK`。
4. `LOCAL_RANK`。
5. `WORLD_SIZE`。
6. `CUDA_VISIBLE_DEVICES`。
7. 启动命令。

如果调度和启动器信息不一致，训练会失败。

## 18.11 启动器和网络配置

分布式训练启动时，网络配置很关键。

常见问题：

1. 选错网卡。
2. 没走 RDMA。
3. NCCL 找不到接口。
4. master_addr 不可达。
5. 端口冲突。
6. DNS 解析慢。
7. NetworkPolicy 阻断通信。

平台可以提供默认网络配置：

1. 自动选择训练网卡。
2. 设置 NCCL_SOCKET_IFNAME。
3. 检查 RDMA 设备。
4. 分配 master_port。
5. 做连通性检测。

不要把这些细节全部交给用户手写。

## 18.12 启动器和日志聚合

分布式训练有多个 rank，日志很多。

平台要决定：

1. 是否收集所有 rank 日志。
2. rank 0 日志是否单独突出显示。
3. worker 失败日志如何聚合。
4. 日志是否按节点和 rank 查询。
5. 日志中敏感信息是否脱敏。

如果只看 rank 0 日志，可能漏掉某个 worker 的真实错误。

如果收集所有日志但没有索引，用户也很难排查。

## 18.13 启动器和失败处理

启动阶段常见失败：

1. 镜像拉取失败。
2. 代码不存在。
3. 依赖缺失。
4. master_addr 不可达。
5. 端口冲突。
6. rank 数不一致。
7. NCCL 初始化失败。
8. 某个 worker OOM。
9. 数据挂载失败。

平台应该区分：

1. 启动前校验失败。
2. 容器启动失败。
3. 分布式初始化失败。
4. 训练运行失败。

不同阶段给用户不同诊断信息。

## 18.14 Elastic Training

Elastic training 是指训练任务可以适应节点数量变化或故障恢复。

PyTorch Elastic 等机制可以支持一定程度的弹性。

价值：

1. 节点故障后不用完全失败。
2. 支持抢占和恢复。
3. 支持资源动态变化。

挑战：

1. 训练代码要支持。
2. checkpoint 要兼容。
3. global batch size 可能变化。
4. 数据 shard 要重新分配。
5. 收敛行为可能变化。

弹性训练很有吸引力，但不是所有任务都能无痛支持。

## 18.15 启动器选择建议

可以按任务选择：

1. 普通 PyTorch DDP / FSDP：torchrun。
2. HuggingFace + ZeRO：DeepSpeed。
3. 大规模 GPT 预训练：Megatron 体系。
4. RLHF / rollout / 多阶段 pipeline：Ray。
5. HPC 集群：MPI / Slurm。
6. 企业统一平台：自研 launcher abstraction。

这不是绝对规则。

关键要看训练代码生态、并行策略、团队经验和平台集成成本。

## 18.16 分布式启动器审计指标与最小 demo

把启动器做成平台能力时，不能只问“命令能不能跑”。更合理的验收口径是：资源、rank、rendezvous、GPU 绑定、launcher adapter、框架配置、网络、日志、失败处理、弹性恢复、checkpoint 和调度交接是否都能被审计。

一个启动器审计样本可以写成：

```math
l_i=(r_i,n_i,p_i,w_i,m_i,b_i,e_i,g_i,c_i,o_i,f_i,z_i)
```

其中，`r_i` 是资源分配，`n_i` 是节点列表，`p_i` 是每节点进程数，`w_i` 是 world size，`m_i` 是 rendezvous，`b_i` 是 GPU 绑定，`e_i` 是环境变量，`g_i` 是 launcher adapter，`c_i` 是框架配置，`o_i` 是日志和观测，`f_i` 是失败和恢复策略，`z_i` 是审计 trace。

第 `j` 个检查维度的覆盖率可以写成：

```math
C_j=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[g_j(l_i)=1]
```

训练平台的启动器门禁可以写成：

```math
G_{\mathrm{launcher}}=\mathbf{1}\left[\min_j C_j\ge \tau_j \land W_{\mathrm{cfg}}=W_{\mathrm{env}} \land P_0=0\right]
```

其中，`W_cfg` 是配置层声明的 world size，`W_env` 是实际注入到 worker 环境里的 world size，`P_0` 是 P0 级启动风险数量。直觉上，launcher gate 不是替代 torchrun、DeepSpeed、Megatron 或 Ray，而是证明平台生成的启动拓扑和每个框架实际看到的拓扑一致。

下面这个 0 依赖 demo 演示如何把分布式启动器检查做成可运行审计。它不启动真实训练进程，只检查 toy 配置里最容易导致分布式训练 hang、错绑 GPU、并行维度不一致、日志缺失或弹性恢复不安全的字段。

```python
from collections import OrderedDict
from copy import deepcopy


def complete_case():
    return OrderedDict(
        name="complete_launcher",
        launcher="torchrun",
        adapter="torchrun",
        nodes=["node-a", "node-b"],
        gpus_per_node=4,
        nproc_per_node=4,
        allocated_world_size=8,
        world_size=8,
        ranks=list(range(8)),
        local_ranks=[[0, 1, 2, 3], [0, 1, 2, 3]],
        cuda_visible_devices=[[0, 1, 2, 3], [0, 1, 2, 3]],
        master_addr="10.0.0.10",
        master_port=29500,
        torchrun=OrderedDict(
            nnodes=2,
            nproc_per_node=4,
            node_rank_assigned=True,
            rdzv_backend="c10d",
        ),
        deepspeed=OrderedDict(
            config_path="ds_config.json",
            zero_stage=2,
            hostfile=True,
        ),
        megatron=OrderedDict(
            tensor_parallel=2,
            pipeline_parallel=2,
            data_parallel=2,
            micro_batch=2,
            grad_accum=8,
            global_batch=32,
        ),
        ray=OrderedDict(
            workers=8,
            worker_gpus=1,
            scaling_config=True,
        ),
        network=OrderedDict(
            nccl_socket_ifname="ib0",
            rdma_ready=True,
            master_reachable=True,
        ),
        logs=OrderedDict(
            rank_logs="all",
            rank_index=True,
            failed_worker_collected=True,
        ),
        failure=OrderedDict(
            stage="distributed_init",
            classified=True,
            retry_policy="checkpoint_resume",
        ),
        elastic=OrderedDict(
            enabled=True,
            checkpoint_compatible=True,
            data_reshard=True,
        ),
        checkpoint=OrderedDict(
            output_uri="s3://train-ckpt/job-18",
            resume_tested=True,
            launcher_state=True,
        ),
        scheduler=OrderedDict(
            node_list_bound=True,
            node_rank_bound=True,
            gpu_binding_bound=True,
        ),
        audit=OrderedDict(
            trace_id="trace-launcher-001",
            final_command_hash="sha256:cmd-v1",
            env_snapshot=True,
        ),
        gate=True,
    )


def make_bad(name, mutate):
    case = deepcopy(complete_case())
    case["name"] = name
    mutate(case)
    return case


def resource_world_size_consistency(case):
    expected = len(case["nodes"]) * case["nproc_per_node"]
    return (
        case["world_size"] == expected
        and case["allocated_world_size"] == expected
        and case["nproc_per_node"] <= case["gpus_per_node"]
    )


def rank_local_rank_mapping(case):
    expected_ranks = list(range(case["world_size"]))
    expected_local = list(range(case["nproc_per_node"]))
    return (
        case["ranks"] == expected_ranks
        and all(ranks == expected_local for ranks in case["local_ranks"])
    )


def rendezvous_endpoint_readiness(case):
    port = case["master_port"]
    return (
        bool(case["master_addr"])
        and isinstance(port, int)
        and 1024 <= port <= 65535
        and case["network"]["master_reachable"]
    )


def gpu_binding_visibility(case):
    need = case["nproc_per_node"]
    for devices in case["cuda_visible_devices"]:
        if len(devices) < need or len(set(devices)) != len(devices):
            return False
    return True


def launcher_adapter_fit(case):
    supported = {"torchrun", "deepspeed", "megatron", "ray", "mpi"}
    return case["launcher"] in supported and case["adapter"] == case["launcher"]


def torchrun_argument_fit(case):
    args = case["torchrun"]
    return (
        args["nnodes"] == len(case["nodes"])
        and args["nproc_per_node"] == case["nproc_per_node"]
        and args["node_rank_assigned"]
        and bool(args["rdzv_backend"])
    )


def deepspeed_config_fit(case):
    cfg = case["deepspeed"]
    return bool(cfg["config_path"]) and cfg["zero_stage"] in {0, 1, 2, 3} and cfg["hostfile"]


def megatron_parallel_consistency(case):
    cfg = case["megatron"]
    model_parallel_world = (
        cfg["tensor_parallel"] * cfg["pipeline_parallel"] * cfg["data_parallel"]
    )
    global_batch = cfg["micro_batch"] * cfg["grad_accum"] * cfg["data_parallel"]
    return model_parallel_world == case["world_size"] and global_batch == cfg["global_batch"]


def ray_runtime_fit(case):
    cfg = case["ray"]
    return (
        cfg["scaling_config"]
        and cfg["workers"] == case["world_size"]
        and cfg["worker_gpus"] == 1
    )


def network_nccl_readiness(case):
    net = case["network"]
    return bool(net["nccl_socket_ifname"]) and net["rdma_ready"] and net["master_reachable"]


def log_rank_aggregation(case):
    logs = case["logs"]
    return logs["rank_logs"] == "all" and logs["rank_index"] and logs["failed_worker_collected"]


def failure_stage_classification(case):
    known = {"preflight", "container_start", "distributed_init", "training_runtime"}
    failure = case["failure"]
    return failure["classified"] and failure["stage"] in known and bool(failure["retry_policy"])


def elastic_training_safety(case):
    elastic = case["elastic"]
    if not elastic["enabled"]:
        return True
    return elastic["checkpoint_compatible"] and elastic["data_reshard"]


def checkpoint_launcher_coupling(case):
    ckpt = case["checkpoint"]
    return bool(ckpt["output_uri"]) and ckpt["resume_tested"] and ckpt["launcher_state"]


def scheduler_launcher_handoff(case):
    handoff = case["scheduler"]
    return (
        handoff["node_list_bound"]
        and handoff["node_rank_bound"]
        and handoff["gpu_binding_bound"]
    )


def launcher_audit_trace(case):
    audit = case["audit"]
    return bool(audit["trace_id"]) and bool(audit["final_command_hash"]) and audit["env_snapshot"]


def distributed_launcher_gate(case):
    return case["gate"] is True


GATES = OrderedDict(
    resource_world_size_consistency=resource_world_size_consistency,
    rank_local_rank_mapping=rank_local_rank_mapping,
    rendezvous_endpoint_readiness=rendezvous_endpoint_readiness,
    gpu_binding_visibility=gpu_binding_visibility,
    launcher_adapter_fit=launcher_adapter_fit,
    torchrun_argument_fit=torchrun_argument_fit,
    deepspeed_config_fit=deepspeed_config_fit,
    megatron_parallel_consistency=megatron_parallel_consistency,
    ray_runtime_fit=ray_runtime_fit,
    network_nccl_readiness=network_nccl_readiness,
    log_rank_aggregation=log_rank_aggregation,
    failure_stage_classification=failure_stage_classification,
    elastic_training_safety=elastic_training_safety,
    checkpoint_launcher_coupling=checkpoint_launcher_coupling,
    scheduler_launcher_handoff=scheduler_launcher_handoff,
    launcher_audit_trace=launcher_audit_trace,
    distributed_launcher_gate=distributed_launcher_gate,
)


CASES = [
    complete_case(),
    make_bad("world_size_mismatch_bad", lambda c: c.__setitem__("allocated_world_size", 7)),
    make_bad("duplicate_rank_bad", lambda c: c.__setitem__("ranks", [0, 1, 2, 3, 4, 5, 6, 6])),
    make_bad("missing_rendezvous_bad", lambda c: c.__setitem__("master_addr", "")),
    make_bad("gpu_binding_mismatch_bad", lambda c: c.__setitem__("cuda_visible_devices", [[0, 1, 2], [0, 1, 2, 3]])),
    make_bad("wrong_launcher_adapter_bad", lambda c: c.__setitem__("adapter", "deepspeed")),
    make_bad("torchrun_arg_missing_bad", lambda c: c["torchrun"].__setitem__("node_rank_assigned", False)),
    make_bad("deepspeed_config_bad", lambda c: c["deepspeed"].__setitem__("zero_stage", 5)),
    make_bad("megatron_parallel_mismatch_bad", lambda c: c["megatron"].__setitem__("data_parallel", 1)),
    make_bad("ray_runtime_missing_bad", lambda c: c["ray"].__setitem__("scaling_config", False)),
    make_bad("nccl_network_missing_bad", lambda c: c["network"].__setitem__("nccl_socket_ifname", "")),
    make_bad("logs_rank0_only_bad", lambda c: c["logs"].__setitem__("rank_logs", "rank0")),
    make_bad("failure_unclassified_bad", lambda c: c["failure"].__setitem__("classified", False)),
    make_bad("elastic_without_safety_bad", lambda c: c["elastic"].__setitem__("checkpoint_compatible", False)),
    make_bad("checkpoint_uncoupled_bad", lambda c: c["checkpoint"].__setitem__("launcher_state", False)),
    make_bad("scheduler_handoff_missing_bad", lambda c: c["scheduler"].__setitem__("node_rank_bound", False)),
    make_bad("audit_trace_missing_bad", lambda c: c["audit"].__setitem__("env_snapshot", False)),
    make_bad("launcher_gate_missing_bad", lambda c: c.__setitem__("gate", False)),
]


def audit_one(case):
    return OrderedDict((name, bool(fn(case))) for name, fn in GATES.items())


def metric_coverage(audits):
    total = len(audits)
    return OrderedDict(
        (name, round(sum(result[name] for result in audits.values()) / total, 3))
        for name in GATES
    )


AUDITS = OrderedDict((case["name"], audit_one(case)) for case in CASES)
METRICS = metric_coverage(AUDITS)
FAILED_CASES = [
    name for name, result in AUDITS.items() if not all(result.values())
]
FAILED_GATES = [name for name, value in METRICS.items() if value < 1.0]

examples = OrderedDict(
    world_size=complete_case()["world_size"],
    torchrun_nnodes=complete_case()["torchrun"]["nnodes"],
    torchrun_nproc_per_node=complete_case()["torchrun"]["nproc_per_node"],
    megatron_world_size=(
        complete_case()["megatron"]["tensor_parallel"]
        * complete_case()["megatron"]["pipeline_parallel"]
        * complete_case()["megatron"]["data_parallel"]
    ),
    global_batch=complete_case()["megatron"]["global_batch"],
)
smoke = OrderedDict(
    complete_case_passes=all(AUDITS["complete_launcher"].values()),
    caught_world_size_mismatch=not AUDITS["world_size_mismatch_bad"]["resource_world_size_consistency"],
    caught_megatron_mismatch=not AUDITS["megatron_parallel_mismatch_bad"]["megatron_parallel_consistency"],
    caught_network_gap=not AUDITS["nccl_network_missing_bad"]["network_nccl_readiness"],
    caught_elastic_gap=not AUDITS["elastic_without_safety_bad"]["elastic_training_safety"],
)
launcher_gate_pass = not FAILED_CASES and all(value >= 1.0 for value in METRICS.values())

print("distributed_launcher_examples=", dict(examples))
print("smoke=", dict(smoke))
print("metrics=", dict(METRICS))
print("hard_blocker_count=", len(FAILED_CASES))
print("failed_cases=", FAILED_CASES)
print("failed_gates=", FAILED_GATES)
print("distributed_launcher_gate_pass=", launcher_gate_pass)
```

这段 demo 的重点不是把真实平台实现完，而是建立面试和工程审计的语言：启动器问题要能落到 `world_size` 一致性、rank 映射、rendezvous、GPU 绑定、launcher adapter、DeepSpeed / Megatron / Ray 配置、NCCL 网络、rank 日志、失败阶段、elastic checkpoint、调度交接和审计 trace。只要这些字段不可见，平台就很难解释“任务为什么卡住、为什么只某个 rank OOM、为什么重试后又重复失败”。

## 18.17 面试中如何回答分布式启动器

如果面试官问：

```text
训练平台如何支持 torchrun、DeepSpeed、Megatron 和 Ray？
```

可以这样回答：

```text
我会在训练平台里抽象统一的 launcher interface。用户在 TrainingJob 里声明 launcher 类型、nnodes、nproc_per_node、并行策略和环境变量。调度系统先分配节点和 GPU，平台确定 master 节点、node_rank、world_size、GPU 绑定和网络配置，然后由对应 launcher adapter 生成实际启动命令。

torchrun 适合 PyTorch DDP/FSDP，DeepSpeed 适合 ZeRO 和大模型微调，Megatron 适合大规模模型并行预训练，Ray 更适合 RLHF、rollout 和复杂分布式 pipeline。

平台要统一处理 master_addr、master_port、rank、local_rank、world_size、NCCL 配置、日志聚合、失败诊断和 checkpoint 恢复。这样用户不用手工管理复杂分布式启动细节。
```

## 18.18 常见误区

误区一：分布式启动就是多开几个 Python 进程。

还要正确设置 rank、local rank、world size、master 地址、GPU 绑定和通信环境。

误区二：torchrun、DeepSpeed、Megatron、Ray 是互相替代的同类工具。

它们边界不同。torchrun 是 PyTorch 分布式启动器，DeepSpeed 是训练优化框架和启动工具，Megatron 是大模型并行训练体系，Ray 是分布式应用框架。

误区三：启动失败都是代码 bug。

也可能是资源分配、网络、端口、NCCL、环境变量、数据挂载或镜像问题。

误区四：平台只要支持一种 launcher 就够。

不同团队和任务可能依赖不同生态，平台最好通过 adapter 支持多种 launcher。

误区五：弹性训练可以自动解决所有故障。

弹性训练需要代码、checkpoint、数据分片和收敛策略配合，不是免费能力。

## 18.19 面试题

### 题 1：rank、local rank 和 world size 分别是什么？

答：rank 是全局进程编号，local rank 是节点内进程编号，world size 是全局进程总数。local rank 常用于绑定当前节点内的 GPU，world size 用于初始化分布式通信。

### 题 2：torchrun 主要解决什么问题？

答：torchrun 用于启动 PyTorch 分布式训练进程，设置多进程、多节点相关环境，帮助训练脚本初始化 distributed process group。它常用于 DDP 和 FSDP。

### 题 3：DeepSpeed 和 torchrun 有什么区别？

答：torchrun 主要是 PyTorch 原生分布式启动工具。DeepSpeed 不只是启动器，还提供 ZeRO、offload、混合精度等训练优化能力，适合降低显存压力和训练大模型。

### 题 4：Megatron 为什么强调 tensor parallel 和 pipeline parallel？

答：大模型参数和计算可能无法放在单张 GPU 上，Megatron 通过 tensor parallel 和 pipeline parallel 把模型切分到多张 GPU 上训练，提高可训练模型规模和吞吐。

### 题 5：Ray 在训练平台里适合什么场景？

答：Ray 适合复杂分布式任务编排，例如 RLHF rollout、多阶段训练 pipeline、数据处理、训练和推理混合任务。它更像分布式应用框架，不只是训练启动器。

## 18.20 小练习

练习一：写出 2 节点、每节点 8 GPU 的 torchrun 启动参数。

要求：说明 nnodes、nproc_per_node、node_rank、master_addr、master_port 和 world_size。

练习二：设计一个 launcher adapter 接口。

要求：输入 TrainingJob、节点列表和资源分配，输出启动命令和环境变量。

练习三：排查分布式初始化卡住。

要求：从 master_addr、端口、rank、world_size、网络、NCCL、DNS 和 NetworkPolicy 角度排查。

练习四：比较 torchrun、DeepSpeed、Megatron 和 Ray。

要求：分别说明适用任务、优势、限制和平台集成注意点。

## 18.21 本章小结

本章讲了分布式训练启动器。

你需要掌握：

1. 分布式训练启动器把资源分配结果转换成进程拓扑和通信环境。
2. rank、local rank、world size、master_addr、master_port 是基础概念。
3. torchrun 适合 PyTorch DDP / FSDP。
4. DeepSpeed 适合 ZeRO、offload 和大模型微调。
5. Megatron 适合大规模模型并行预训练。
6. Ray 适合复杂分布式 pipeline、RLHF、rollout 和 Agent 场景。
7. 平台最好用 launcher abstraction 统一封装不同启动器。
8. 启动器要和调度、网络、日志、失败处理和 checkpoint 集成。
9. 分布式启动失败需要按资源、环境、网络、rank 和框架分层排查。

下一章我们会讲训练配置管理：超参、版本、环境和可复现性。
