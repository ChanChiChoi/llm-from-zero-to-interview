# 第 22 章 vLLM worker、executor 和 engine 架构

上一章讲了 vLLM memory management：KV Cache 如何被切成 blocks，block pool、free queue、ref count、prefix cache、preemption 和 chunked prefill 如何共同影响并发和显存。

本章把视角从“内存资源”切到“执行架构”：scheduler 选好了本轮要跑的请求之后，这个执行计划如何被派发到 GPU？worker、executor、engine core、model runner 分别负责什么？单机多卡和多进程 serving 为什么要这么拆？

一句话概括：

> vLLM 的 worker、executor 和 engine 架构，本质是在入口层、调度层和 GPU 执行层之间建立清晰边界：engine core 做调度和状态管理，executor 负责派发执行，worker 控制具体设备，model runner 把调度结果转换成模型 forward。

## 22.1 本章目标

读完本章，你应该能讲清：

1. API server、LLMEngine、engine core、executor、worker、model runner、model 的职责边界。
2. 为什么 vLLM-like 系统通常采用多进程架构。
3. scheduler output 如何变成 GPU worker 上的模型执行。
4. worker 为什么通常是一进程控制一张 GPU。
5. model runner 在输入准备、KV cache、CUDA graph、attention backend 中扮演什么角色。
6. tensor parallel、pipeline parallel、data parallel 会如何影响 worker 数量和通信。
7. 面试中如何画出 vLLM 执行架构图。

## 22.2 先看整体进程图

vLLM V1 架构文档把核心进程分成几类：

1. API server process。
2. Engine core process。
3. GPU worker processes。
4. DP coordinator process，如果启用 data parallel。

简化图：

```text
Client
  |
  v
API Server Process
  |
  v
Engine Core Process
  |-- Scheduler
  |-- KV Cache Manager
  |-- Request State
  |
  v
Executor
  |
  +--> GPU Worker 0 -> Model Runner -> Model shard/device 0
  +--> GPU Worker 1 -> Model Runner -> Model shard/device 1
  +--> GPU Worker 2 -> Model Runner -> Model shard/device 2
  +--> GPU Worker 3 -> Model Runner -> Model shard/device 3
```

如果是单 GPU，可以简化成：

```text
API Server -> Engine Core -> Executor -> Worker -> Model Runner -> Model
```

如果是多 GPU，executor 和 worker 层就变得非常重要，因为 engine core 不能自己直接在每张 GPU 上做所有事情，它需要一个执行抽象来协调多个 worker。

## 22.3 各模块职责一句话版

先给出最重要的边界。

API server：接收 HTTP/OpenAI-compatible 请求，做输入处理、tokenization、多模态数据加载和 streaming 返回。

LLMEngine 或 AsyncLLMEngine：对外暴露 Python 或异步接口，连接 input processor、engine core 和 output processor。

Engine core：运行 scheduler，管理请求状态和 KV cache，协调模型执行，是 serving engine 的调度中枢。

Executor：接收 engine core 的执行命令，把它派发给一个或多个 worker，并收集 worker 输出。

Worker：绑定具体 device，负责加载模型权重、管理本地 GPU 资源、执行 forward。

Model runner：在 worker 内部准备模型输入、attention metadata、KV cache 引用、CUDA graph 捕获和模型 forward。

Model：真正的 `torch.nn.Module` 或等价模型实现，包含 transformer layers、attention、MLP、norm、embedding、lm head 等。

Output processor：把 engine core 输出转换成用户可见的 RequestOutput，处理 detokenization、stop、finish reason 和 streaming。

## 22.4 为什么要拆成这些层

如果只是教学版 MiniEngine，可以写成一个类：

```python
class MiniEngine:
    def step(self):
        batch = self.scheduler.schedule()
        outputs = self.model(batch)
        return self.decode(outputs)
```

但生产级 serving engine 不能这么简单。

原因一：CPU 和 GPU 职责不同。

输入处理、HTTP、tokenization、scheduler、output processor 很多是 CPU 工作；模型 forward 是 GPU 工作。

原因二：多卡执行需要协调。

Tensor parallel、pipeline parallel、expert parallel 都需要多个 worker 同步执行。

原因三：故障和资源隔离。

让每个 GPU worker 在独立进程中控制一张 GPU，可以降低 Python GIL、CUDA context、内存管理和崩溃影响的复杂度。

原因四：部署形态多样。

同一套 engine 可能支持 offline Python API、online server、单机多卡、多机多卡、data parallel、Ray 或 multiprocessing。

原因五：性能调优需要边界。

如果 API server 慢，可以扩 API server；如果 scheduler 慢，优化 engine core；如果 GPU 不满，调 worker/model runner；如果通信慢，调并行策略。

清晰分层让问题定位更容易。

## 22.5 Engine Core：调度中枢

Engine core 的职责可以概括为：

1. 接收新请求和 abort 请求。
2. 维护 waiting/running/finished request state。
3. 调用 scheduler 生成本轮执行计划。
4. 管理 KV Cache block 分配、释放和 prefix cache。
5. 把 scheduler output 转成 executor 能执行的输入。
6. 调用 executor 执行模型。
7. 接收模型输出并更新请求状态。
8. 输出本轮生成 token、finish reason 和统计信息。

它不是直接执行 PyTorch forward 的地方，而是调度和状态管理中心。

一个简化 engine core loop：

```python
class EngineCore:
    def step(self):
        self.handle_new_requests()
        self.handle_aborts()

        scheduler_output = self.scheduler.schedule(
            waiting=self.waiting,
            running=self.running,
            kv_cache_manager=self.kv_cache_manager,
        )

        if scheduler_output.is_empty():
            return []

        model_outputs = self.executor.execute_model(scheduler_output)
        engine_outputs = self.process_model_outputs(model_outputs)
        self.cleanup_finished_requests()
        return engine_outputs
```

真实系统还会加入 metrics、profiling、sleep/wakeup、multi-modal cache、structured output、speculative decoding、distributed execution 等能力。

## 22.6 Executor：执行派发层

Executor 是 engine core 和 workers 之间的桥。

Engine core 不应该关心底层是：

1. 单进程单 GPU。
2. 多进程多 GPU。
3. Ray workers。
4. multiprocessing workers。
5. 分布式多节点。

它只想说：

```text
这是本轮要执行的 batch，请执行模型 forward，并把结果给我。
```

Executor 负责把这个高层命令翻译成 worker 调用。

简化接口：

```python
class Executor:
    def execute_model(self, scheduler_output):
        worker_inputs = self.prepare_worker_inputs(scheduler_output)
        results = []
        for worker, worker_input in zip(self.workers, worker_inputs):
            results.append(worker.execute_model(worker_input))
        return self.merge_worker_outputs(results)
```

在 tensor parallel 场景中，所有 TP workers 可能必须共同执行同一个 forward，每个 worker 持有一部分权重，过程中需要 collective communication。

在 pipeline parallel 场景中，不同 worker 可能持有不同层，输入要沿 pipeline 传递。

在 data parallel 场景中，不同 engine core 或 worker group 可能处理不同请求 batch。

Executor 的价值是把这些复杂并行细节隐藏在统一执行接口后面。

## 22.7 Worker：一张 GPU 的控制者

vLLM 架构文档中，worker 通常是运行模型推理的进程，并且常见实践是一进程控制一个 accelerator device。

worker 的职责包括：

1. 设置当前 device。
2. 初始化分布式通信环境。
3. 加载或构造模型。
4. 分配本地 KV Cache memory。
5. 持有 model runner。
6. 执行 model forward。
7. 返回 logits、hidden states 或采样需要的信息。
8. 执行 profile、sleep、wake up 等控制命令。

为什么一进程一 GPU 常见？

1. CUDA context 更清晰。
2. GPU memory 管理边界更清楚。
3. 多进程能绕开部分 Python GIL 影响。
4. worker 崩溃或 hang 的问题更容易隔离。
5. 分布式 rank/local rank 映射更自然。

worker 通常有两个重要标识：

1. `rank`：全局 rank，用于分布式通信。
2. `local_rank`：本机 rank，用于选择本地 GPU。

例如 4 卡 tensor parallel：

```text
worker 0: rank=0, local_rank=0, cuda:0
worker 1: rank=1, local_rank=1, cuda:1
worker 2: rank=2, local_rank=2, cuda:2
worker 3: rank=3, local_rank=3, cuda:3
```

## 22.8 Model Runner：worker 内部的执行大脑

worker 控制设备，但真正把 request batch 变成模型 forward 的，通常是 model runner。

Model runner 负责：

1. 根据 scheduler output 准备 input tensors。
2. 准备 positions。
3. 准备 attention metadata。
4. 准备 block tables 和 slot mapping。
5. 管理 KV Cache 读写接口。
6. 选择 attention backend。
7. 调用模型 forward。
8. 处理 CUDA graph capture/replay。
9. 返回采样所需 logits。

上一章我们说过，scheduler 输出的是 request-level 执行计划，但 GPU kernel 需要 tensor-level metadata。

转换过程大致是：

```text
scheduler output
  -> input_ids tensor
  -> positions tensor
  -> block tables
  -> slot mapping
  -> attention metadata
  -> sampling metadata
  -> model forward
```

这就是 model runner 的核心价值。

## 22.9 Model Runner 为什么复杂

初学者容易以为 model runner 就是：

```python
logits = model(input_ids)
```

真实情况复杂得多。

复杂点一：prefill 和 decode 形态不同。

Prefill 一次处理多个 prompt tokens，decode 通常每个 sequence 处理一个 token。二者 attention metadata、KV 写入位置和 kernel 选择可能不同。

复杂点二：batch 是动态的。

continuous batching 每轮 batch 都可能变化，model runner 需要根据本轮 scheduler output 重新构造 tensor。

复杂点三：KV Cache 是 paged 的。

attention kernel 不是读连续 past key values，而是根据 block table 找 physical blocks。

复杂点四：并行策略影响模型执行。

TP 需要 shard 权重和 all-reduce，PP 需要跨 stage 传 hidden states，EP 需要路由 experts。

复杂点五：性能优化很多。

CUDA graph、torch.compile、fused kernels、attention backend、quantization、LoRA、speculative decoding 都会影响 model runner 的执行路径。

因此 model runner 是推理框架里最接近“模型系统优化”的地方。

## 22.10 从 Scheduler Output 到 Worker Input

假设 scheduler 本轮输出：

```text
prefill: request A, request B
decode: request C, request D, request E
```

Engine core 或 model runner 需要构造本轮输入。

可能包括：

```python
class WorkerInput:
    def __init__(self):
        self.input_ids = None
        self.positions = None
        self.block_tables = None
        self.slot_mapping = None
        self.seq_lens = None
        self.query_start_loc = None
        self.attention_metadata = None
        self.sampling_metadata = None
```

其中：

1. `input_ids`：本轮实际送入模型的 tokens。
2. `positions`：每个 token 的位置。
3. `block_tables`：每个请求 logical block 到 physical block 的映射。
4. `slot_mapping`：本轮新 KV 应该写到哪些 physical slots。
5. `seq_lens`：每个 sequence 当前长度。
6. `query_start_loc`：变长 batch 中每个请求 query 的起始位置。
7. `attention_metadata`：attention backend 所需的综合信息。
8. `sampling_metadata`：每个请求的采样参数和 logits 选择位置。

这一步如果错了，模型可能不会立刻报错，但会生成错误内容或读写错误 KV 位置。

## 22.11 多 GPU 下的执行方式

多 GPU serving 不是简单把 batch 平均分到多张卡。

具体取决于并行策略。

Tensor Parallel：

同一层的权重被切到多张 GPU 上，每张 GPU 算一部分矩阵乘，之后通过 all-reduce 或 all-gather 合并。

```text
same request batch -> worker 0/1/2/3 同时执行同一层的不同 shard
```

Pipeline Parallel：

模型层被切成多个 stage，不同 GPU 负责不同层。

```text
worker 0: layers 0-9
worker 1: layers 10-19
worker 2: layers 20-29
worker 3: layers 30-39
```

Data Parallel：

多个 replica 各自处理不同请求 batch，提高总体吞吐。

```text
engine core group 0 -> request batch A
engine core group 1 -> request batch B
```

Expert Parallel：

MoE expert 分布在不同 GPU 上，token 根据 router 被发送到对应 expert。

这些策略可以组合，所以 executor/worker 必须能处理不同通信和执行拓扑。

## 22.12 Engine Core 与 Worker 的通信内容

Engine core 发给 worker 的不是原始 HTTP 请求。

它通常已经是压缩后的执行信息：

1. 本轮输入 token ids。
2. positions。
3. KV block tables。
4. slot mapping。
5. sampling 相关 metadata。
6. LoRA 或 adapter id。
7. 多模态 embedding 或 encoder cache 引用。
8. 控制命令，如 profile、sleep、reset cache。

worker 返回的也不是最终文本，而是模型执行结果：

1. logits。
2. sampled token ids。
3. hidden states，某些任务需要。
4. finished flags 或辅助信息。
5. timing stats。

最终 detokenize、stop string、streaming response 仍然通常在 output processor 或前端路径完成。

## 22.13 为什么 API Server 不直接控制 GPU

有些初学者会问：既然 API server 收到请求，为什么不直接调用 worker？

原因是 API server 处理的是外部协议，而 GPU worker 处理的是模型执行，两者中间必须有 scheduler 和 engine core。

如果 API server 直接调用 GPU，会出现：

1. 无法全局 continuous batching。
2. 无法统一管理 KV Cache。
3. 多个 HTTP 请求会争抢 GPU。
4. abort、timeout、priority 难以统一处理。
5. 多 API server 或 data parallel 下路由混乱。

正确边界是：

```text
API server 负责请求接入和输出
Engine core 负责调度和状态
Worker 负责执行
```

这个边界可以让 API server 扩容、engine core 调度、worker 执行三者相对解耦。

## 22.14 CPU 资源为什么也重要

vLLM 优化文档强调，V1 多进程架构会让 API server、engine core 和每个 GPU worker 都需要 CPU 资源。

最小单机 N GPU 部署通常至少有：

```text
1 API server + 1 engine core + N GPU workers
```

也就是至少 `2 + N` 个进程。

如果 CPU 核不足，会影响：

1. tokenization。
2. chat template 渲染。
3. multi-modal data loading。
4. scheduler loop。
5. worker 侧 kernel launch 和控制逻辑。
6. detokenization 和 streaming。

GPU 利用率低不一定是 GPU 算不满，也可能是 CPU 调度、输入处理或输出处理跟不上。

尤其 engine core 是调度忙循环，如果 CPU 被抢占，会直接影响每轮请求派发速度，表现为 GPU 有空洞、TPOT 抖动或吞吐下降。

## 22.15 CUDA context、初始化和 warmup

worker 启动时通常要做很多初始化：

1. 设置 device。
2. 初始化 distributed process group。
3. 加载模型权重。
4. 初始化 tokenizer 或模型配置引用。
5. 分配 KV Cache。
6. 选择 attention backend。
7. 准备 CUDA graph 或 compile 路径。
8. 做 warmup。

这些工作解释了为什么大模型 serving 启动很慢。

模型越大，权重加载越慢；并行度越高，分布式初始化越复杂；CUDA graph 或 compile 优化越 aggressive，启动时间也可能越长。

工程上常见做法是：

1. 启动后先 warmup。
2. 健康检查通过后再接流量。
3. 滚动升级时避免所有 replica 同时冷启动。
4. 把模型权重放在本地高速盘或缓存层。

## 22.16 故障路径：worker 挂了怎么办

生产系统必须考虑 worker failure。

可能的故障包括：

1. CUDA OOM。
2. NCCL hang。
3. kernel crash。
4. worker process crash。
5. GPU reset。
6. 网络或 IPC 通信失败。

如果 worker 挂了，engine core 不能假装请求还在正常运行。

它需要：

1. 标记相关请求 failed。
2. 清理 request state。
3. 释放或重建 KV Cache 状态。
4. 通知 API server 返回错误。
5. 触发 worker restart 或实例摘流。

多 worker 并行时，一个 worker 挂掉通常会影响整个 parallel group。比如 TP=4 的一个 rank 挂了，这个模型 replica 基本不能继续正常执行。

## 22.17 和前几章的关系

可以把第 18 到 22 章串起来：

第 18 章：Block Manager 管 KV Cache blocks。

第 19 章：Continuous batching 决定每轮谁跑。

第 20 章：请求生命周期说明请求如何进入、执行、输出、结束。

第 21 章：Memory management 说明显存、prefix cache、preemption、offload 如何影响调度。

第 22 章：Worker/executor/engine 架构说明调度结果如何落到 GPU 执行。

合起来就是：

```text
request state + scheduler + KV cache manager + executor + worker/model runner
```

这才是 vLLM-like serving engine 的主干。

## 22.18 常见工程坑

坑一：把 executor 当成 scheduler。

Executor 负责执行派发，不应该决定业务调度策略。调度策略应该在 scheduler/engine core 中。

坑二：worker input 和 scheduler output 不一致。

block table、slot mapping、positions 任何一个错，都会导致 KV 读写错误。

坑三：多进程启动方式不正确。

CUDA、多进程、fork/spawn、NCCL 初始化顺序处理不好，可能导致死锁或奇怪的 CUDA 错误。

坑四：CPU 配置不足。

GPU worker 数上去了，但 CPU 核数不够，engine core 或 API server 被抢占，整体吞吐反而下降。

坑五：把所有逻辑都塞进 worker。

worker 应专注模型执行。请求队列、HTTP、stop string、复杂输出处理如果都塞进 worker，会让架构边界混乱。

坑六：忽略 rank/local_rank 映射。

多机多卡时 rank、local rank、device id、通信 group 配错，会导致模型 shard 错位或 collective hang。

## 22.19 面试官会怎么问

问题一：vLLM 的 engine core、executor、worker 分别负责什么？

回答要点：engine core 负责调度、请求状态和 KV cache 管理；executor 负责把执行计划派发给 worker 并收集结果；worker 绑定 GPU，持有 model runner 和模型，执行 forward。

问题二：为什么 worker 通常一进程一 GPU？

回答要点：CUDA context、显存管理、rank/local_rank、故障隔离和多进程并行更清晰，也减少 Python GIL 和共享状态复杂度。

问题三：model runner 做什么？

回答要点：把 scheduler output 转换成模型需要的 tensors 和 metadata，包括 input ids、positions、block tables、slot mapping、attention metadata、sampling metadata，并调用模型 forward。

问题四：executor 和 scheduler 有什么区别？

回答要点：scheduler 决定本轮谁应该执行，executor 决定如何把这个执行计划分发到具体 worker 上执行。

问题五：多 GPU 下 worker 数量怎么理解？

回答要点：通常每张 GPU 一个 worker。TP/PP 共同决定一个 engine core 下的 worker group 大小；DP 会复制 engine core/worker group 来处理不同 batch。

问题六：为什么 CPU 资源会影响 GPU serving？

回答要点：API server、engine core、worker control path、tokenization、scheduler、detokenization、streaming 都需要 CPU；CPU 饥饿会导致 GPU 等待调度和 kernel launch。

## 22.20 标准回答模板

如果面试官问“请讲一下 vLLM 的 worker、executor 和 engine 架构”，可以这样回答：

```text
vLLM-like 系统通常把请求入口、调度和 GPU 执行拆成多层。API server 负责 HTTP/OpenAI-compatible 请求、输入处理和 streaming 输出。LLMEngine 或 AsyncLLMEngine 作为对外 engine 接口，连接 input processor、engine core 和 output processor。

Engine core 是调度中枢，维护 waiting/running 请求状态，运行 scheduler，管理 KV cache，并把本轮 scheduler output 交给 executor。Scheduler 决定谁跑，executor 负责怎么把执行计划派发到 worker。

Worker 通常是一进程控制一张 GPU，负责设置 device、加载模型、持有本地 KV cache 和 model runner，并执行模型 forward。Model runner 在 worker 内部把 request-level 的调度结果转换成 tensor-level 的执行输入，比如 input ids、positions、block tables、slot mapping、attention metadata 和 sampling metadata，然后调用模型。

在多 GPU 下，tensor parallel、pipeline parallel、data parallel 会改变 worker group 和通信方式。TP 是多个 worker 共同执行同一 batch 的不同权重 shard，PP 是不同 worker 持有不同层，DP 是复制 engine/worker group 处理不同请求 batch。这样的分层让 vLLM 可以同时支持 continuous batching、PagedAttention、多进程和多卡 serving。
```

## 22.21 小练习

1. 画出单 GPU vLLM-like 架构图：API server、engine core、executor、worker、model runner、model。
2. 画出 TP=4 的架构图，标出 4 个 worker 的 rank 和 local rank。
3. 写一个简化 `Executor.execute_model()` 伪代码，把 scheduler output 派发给多个 workers。
4. 列出 model runner 构造一次 forward input 需要的 6 类 metadata。
5. 解释 scheduler 和 executor 的区别。
6. 分析为什么 CPU 核数不足会导致 GPU utilization 下降。

## 22.22 本章总结

vLLM worker、executor 和 engine 架构的核心是职责分层。

Engine core 负责调度和状态，executor 负责执行派发，worker 负责设备和模型执行，model runner 负责把调度结果转换成模型 forward 所需的 tensors 和 metadata。API server 和 output processor 则处理外部协议、输入输出和 streaming。

这种架构的价值在于：请求生命周期、continuous batching、PagedAttention、KV Cache management、多 GPU 并行和 streaming 输出都能在清晰边界下协作。理解这层执行架构后，再看 TP/PP/DP、prefix cache、PD 分离和推理性能调优都会更自然。

下一章会进入 Prefix Caching 与 Prompt Cache，重点讲 prefix cache 和传统 prompt cache 的区别、hash-based block reuse、cache isolation、命中率、淘汰策略，以及它们如何影响 TTFT 和显存。
