# 第 18 章 分布式训练启动器：torchrun、DeepSpeed、Megatron 和 Ray

上一章讲了训练任务提交系统。本章讲任务提交后真正启动分布式训练时会遇到的问题：如何把一个训练脚本在多张 GPU、多台机器上正确启动起来？

很多同学以为分布式训练启动就是“多开几个进程”。真实情况复杂得多：每个进程需要知道自己的 rank、world size、master 地址、通信端口、GPU 绑定、并行策略、环境变量和失败处理方式。

先记住一句话：

> 分布式训练启动器的核心作用，是把平台分配的资源转换成训练框架能理解的进程拓扑和通信环境。

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

```text
world_size = tensor_parallel * pipeline_parallel * data_parallel
```

如果并行维度配置不一致，任务可能启动失败或训练错误。

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

## 18.16 面试中如何回答分布式启动器

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

## 18.17 常见误区

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

## 18.18 面试题

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

## 18.19 小练习

练习一：写出 2 节点、每节点 8 GPU 的 torchrun 启动参数。

要求：说明 nnodes、nproc_per_node、node_rank、master_addr、master_port 和 world_size。

练习二：设计一个 launcher adapter 接口。

要求：输入 TrainingJob、节点列表和资源分配，输出启动命令和环境变量。

练习三：排查分布式初始化卡住。

要求：从 master_addr、端口、rank、world_size、网络、NCCL、DNS 和 NetworkPolicy 角度排查。

练习四：比较 torchrun、DeepSpeed、Megatron 和 Ray。

要求：分别说明适用任务、优势、限制和平台集成注意点。

## 18.20 本章小结

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
