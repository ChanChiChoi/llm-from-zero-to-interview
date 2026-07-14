# 第 19 章 Continuous Batching 与 iteration-level scheduling

上一章讲了 KV Cache Block Manager：请求进入系统后，KV Cache 不再是一整块连续张量，而是由 block manager 分配、追踪和释放的一组 physical blocks。

有了 block manager，下一步才是调度问题：既然每个请求的 cache 可以动态增长和释放，那 serving engine 能不能不要等一个 batch 全部结束，而是在每一轮生成时动态加入新请求、移除完成请求？这就是 continuous batching，也常被称为 iteration-level scheduling。

一句话概括：

> Continuous batching 把调度粒度从“整批请求”下沉到“每一轮模型迭代”，让系统可以在每个 decode step 重新选择 running batch，从而提升吞吐、降低排队等待，并更充分利用 KV Cache block。

## 19.0 本讲资料边界与第二轮精修口径

本讲按第二轮精修要求做过资料校准，主要参考三类公开资料：

1. Orca OSDI 2022 论文对 Transformer 生成 workload 的 multi-iteration 特征、request-level scheduling 的局限，以及 iteration-level scheduling 的定义。
2. vLLM optimization / tuning 文档对 KV cache 不足、preemption、`max_num_seqs`、`max_num_batched_tokens`、chunked prefill、decode 优先和 TTFT / ITL 取舍的公开口径。
3. vLLM metrics 文档对 running / waiting 请求数、KV cache usage、TTFT、inter-token latency、queue time、prefill time 和 decode time 的观测口径。

本章只讲教学版 continuous batching 和 iteration-level scheduling，不展开 vLLM 真实 scheduler 源码、CUDA graph、worker / executor、multi-step scheduling、prefix cache policy、preemption / swap / recompute 实现、多卡流水、生产级优先级队列或 OpenAI-compatible server 参数全集。

本章给出的公式和 demo 只用于建立可面试、可验证的最小模型：每轮 scheduler 先处理 dynamic arrival 和 cleanup，再在 token budget、active sequence budget 与 KV block budget 下选择 decode 和 prefill。真实系统还会叠加 chunked prefill、prefix caching、LoRA、多模态输入、speculative decoding、抢占、租户公平和可观测性治理。

参考资料：

1. Orca OSDI 2022：<https://www.usenix.org/conference/osdi22/presentation/yu>
2. vLLM optimization and tuning：<https://docs.vllm.ai/en/latest/configuration/optimization.html>
3. vLLM metrics：<https://docs.vllm.ai/en/latest/design/metrics.html>

## 19.1 本章目标

读完本章，你应该能讲清：

1. 为什么传统 static batching 不适合自回归 LLM serving。
2. iteration-level scheduling 为什么出现，它和 request-level scheduling 的区别是什么。
3. continuous batching 每一轮到底调度什么。
4. prefill、decode、token budget 和 KV block budget 如何共同约束调度。
5. 为什么 continuous batching 需要 KV Cache block manager 配合。
6. 面试中如何用时间线解释 vLLM-like scheduler 的核心思想。

## 19.2 先看传统 batch 的问题

传统深度学习推理里，batch 通常是一次性组织好的。

例如图像分类：

```text
收集 32 张图片 -> 拼成 batch -> forward 一次 -> 返回 32 个结果
```

这种任务的特点是：一个请求通常只需要一次模型执行。

LLM 自回归生成不一样。一个请求要生成 100 个 token，就要经历多轮模型执行：

```text
prefill prompt -> decode token 1 -> decode token 2 -> ... -> decode token 100
```

如果把多个请求固定成一个 batch，然后等整个 batch 全部完成，就会遇到几个问题。

第一，短请求被长请求拖住。

```text
request A: 生成 10 个 token
request B: 生成 200 个 token
request C: 生成 20 个 token
```

如果 A、B、C 被固定在同一个 batch 里，A 和 C 很快结束，但 batch 还要陪 B 跑很久。A、C 的计算位置空出来后，如果系统不能动态补入新请求，GPU 利用率会下降。

第二，新请求 TTFT 变差。

假设 batch B 正在生成，新的 request D 到达。如果调度粒度是整批请求，D 必须等当前 batch 全部结束才能 prefill。只要 batch 里有一个长输出请求，D 的排队等待就会变长。

第三，batch size 会越来越小。

固定 batch 开始时可能有 32 个请求，但随着短请求结束，后面可能只剩几个长请求在跑。decode 阶段每轮只生成一个 token，batch 越小，GPU 越难吃满。

这就是为什么 LLM serving 不能只套用普通推理服务的 static batching。

## 19.3 Orca 提出的 iteration-level scheduling

Orca 论文指出，Transformer 生成模型的 serving workload 具有 multi-iteration 特征：一个请求不是执行一次模型就结束，而是需要多次迭代，每次迭代通常生成一个 token。

传统 request-level scheduling 的粒度是：

```text
选择一批请求 -> 运行到这些请求完成 -> 再选择下一批请求
```

Iteration-level scheduling 的粒度是：

```text
选择一批请求 -> 只执行一轮模型迭代 -> 回到 scheduler -> 重新选择下一轮请求
```

也就是说，scheduler 不再把 batch 当成一个长期固定对象，而是每轮都可以重新组织 batch。

一个简单时间线：

```text
t0: batch = [A, B]
t1: A 生成 1 token, B 生成 1 token
t2: A 完成，C 到达，batch = [B, C]
t3: B 生成 1 token, C prefill 或 decode
t4: D 到达，batch = [B, C, D]
```

这里的关键不是“batch 很大”，而是“batch 可以持续变化”。

所以 continuous batching 可以理解为 iteration-level scheduling 在 LLM serving engine 中的工程化形态。

## 19.4 Continuous Batching 的核心直觉

Continuous batching 的直觉是：

> 不要把请求固定成一批跑到结束，而是在每个 engine step 重新决定哪些请求执行。

一个 engine step 通常会做这些事：

1. 接收新到达请求，放入 waiting queue。
2. 清理已经完成、取消或失败的请求。
3. 释放 finished 请求占用的 KV blocks。
4. 从 running 请求中选择本轮继续 decode 的序列。
5. 从 waiting queue 中选择部分请求做 prefill。
6. 检查 token budget、sequence budget 和 KV block budget。
7. 执行本轮模型 forward。
8. 采样新 token，更新请求状态。
9. 把 token 通过 streaming 返回给客户端。

如果画成循环：

```text
while engine is running:
    schedule one step
    run model forward
    sample tokens
    update states
    stream outputs
    free finished resources
```

这就是 serving engine 的心跳。

## 19.5 prefill 和 decode 为什么要一起调度

在 LLM serving 中，waiting 请求通常需要先做 prefill，running 请求通常需要继续 decode。

prefill 的特点：

1. 一次处理 prompt 中的多个 token。
2. 计算量大，容易占用较长 GPU 时间。
3. 目标是尽快产出 first token，影响 TTFT。
4. 会一次性写入较多 KV Cache。

decode 的特点：

1. 每个 running 请求通常每轮只生成一个 token。
2. 单轮 token 数少，但需要频繁执行。
3. 直接影响 TPOT 和流式输出稳定性。
4. KV Cache 会逐 token 增长。

如果只偏向 prefill，新请求 TTFT 会变好，但已有请求的流式输出会卡顿。

如果只偏向 decode，已有请求输出平稳，但 waiting queue 中的新请求可能长时间拿不到 first token。

所以 scheduler 的核心难点是：在同一个 engine step 里，如何平衡 prefill 和 decode。

最小策略可以是：

```text
先保证 running decode 有基本进度，再用剩余 token budget 接纳部分 waiting prefill。
```

更激进的策略可能会在负载高时限制长 prompt prefill，或者把长 prompt 切成多个 chunk，这会在后续 chunked prefill 章节展开。

## 19.6 token budget：调度不是只数请求数

最小教学版 scheduler 可能只限制 `max_running_requests`。

但真实 serving engine 不能只看请求数，因为不同请求的 token 开销差异很大。

两个请求数相同的 batch，代价可能完全不同：

```text
batch A: 8 个请求，每个 prompt 64 tokens
batch B: 8 个请求，每个 prompt 8192 tokens
```

它们都是 8 个请求，但 batch B 的 prefill 计算和 KV Cache 需求远大于 batch A。

因此 continuous batching 通常需要 token budget。

简化定义：

```python
max_num_batched_tokens = 4096
```

每轮调度时，scheduler 统计本轮要处理的 token 数：

```text
本轮 token 数 = prefill token 数 + decode token 数
```

decode token 数通常近似等于 running sequence 数，因为每个 sequence 生成一个 token。

prefill token 数则取决于新请求 prompt 长度，或者 chunked prefill 中本轮处理的 prompt chunk 长度。

一个简化调度例子：

```python
def schedule_with_token_budget(waiting_queue, running_requests, max_tokens):
    scheduled_decode = []
    scheduled_prefill = []
    used_tokens = 0

    for req in running_requests:
        if used_tokens + 1 > max_tokens:
            break
        scheduled_decode.append(req)
        used_tokens += 1

    while waiting_queue:
        req = waiting_queue.peek()
        prompt_len = len(req.input_ids)
        if used_tokens + prompt_len > max_tokens:
            break
        scheduled_prefill.append(waiting_queue.pop())
        used_tokens += prompt_len

    return scheduled_prefill, scheduled_decode
```

这个代码还很粗糙，但它说明了关键点：scheduler 真正在调度的是 token workload，而不是单纯的 request count。

把它写成公式，本轮 iteration 记为 `\tau`，本轮 decode 集合记为 `D_{\tau}`，本轮 prefill 集合记为 `P_{\tau}`。

本轮 decode token 数可以近似为：

$$
N_{\mathrm{dec},\tau}=|D_{\tau}|
$$

如果请求 `i` 在本轮 prefill 了 `p_{i,\tau}` 个 prompt tokens，则本轮 prefill token 数是：

$$
N_{\mathrm{pre},\tau}=\sum_{i\in P_{\tau}}p_{i,\tau}
$$

本轮 token budget 门禁是：

$$
G_{\mathrm{tok},\tau}=\mathbf{1}[N_{\mathrm{dec},\tau}+N_{\mathrm{pre},\tau}\le C_{\mathrm{tok}}]
$$

这里的 `C_{\mathrm{tok}}` 可以理解成教学版 `max_num_batched_tokens`。它不是总吞吐目标，而是控制单轮 workload 和延迟尖刺的上限。

## 19.7 KV block budget：能算不代表能放下

token budget 控制计算量，但还不够。

LLM serving 的另一个硬约束是 KV Cache 显存。

一个请求能不能进入 prefill，不仅取决于本轮 token budget，还取决于 block manager 是否能给它分配足够的 KV blocks。

例如：

```text
block_size = 16
prompt_len = 4096
需要 blocks = ceil(4096 / 16) = 256
```

如果当前 free blocks 只有 200 个，即使 token budget 允许，也不能调度这个请求，否则 prefill 写 KV Cache 时会 OOM。

decode 阶段也一样。每个 running 请求生成新 token 时，如果当前 logical block 满了，需要再分配一个 physical block。

所以每轮 scheduler 至少要问 block manager 两类问题：

1. waiting 请求做 prefill 是否能分配足够 blocks。
2. running 请求继续 decode 是否可能需要追加新 block。

简化伪代码：

```python
def can_schedule_prefill(req, block_manager):
    return block_manager.can_allocate(req.prompt_len)


def can_schedule_decode(req, block_manager):
    if req.current_block_has_slot():
        return True
    return block_manager.has_free_block()
```

这就是上一章为什么说 scheduler 和 block manager 强耦合。

没有 block budget 的 continuous batching，只是“看起来动态”，实际上很容易把系统推向 OOM。

把 block 约束写成公式，block size 记为 `S`，请求 `i` 当前需要缓存的 token 数记为 `T_i`，则请求所需 block 数是：

$$
B_i=\left\lceil\frac{T_i}{S}\right\rceil
$$

如果请求 `i` 当前已经持有 `H_i` 个 block，那么本轮新增 block 需求是：

$$
\Delta B_{i,\tau}=\max(0,B_i-H_i)
$$

本轮所有被调度请求的新增 block 数不能超过当前 free blocks：

$$
G_{\mathrm{kv},\tau}=\mathbf{1}\left[\sum_{i\in D_{\tau}\cup P_{\tau}}\Delta B_{i,\tau}\le F_{\tau}\right]
$$

因此 continuous batching 的准入条件至少是 token budget 与 KV block budget 同时成立：

$$
G_{\mathrm{admit},\tau}=G_{\mathrm{tok},\tau}G_{\mathrm{kv},\tau}
$$

## 19.8 一轮 continuous batching 的完整流程

现在把 token budget 和 KV block budget 合起来看。

一个简化 engine step 可以写成：

```python
def engine_step(waiting_queue, running, block_manager, max_tokens):
    scheduled_decode = []
    scheduled_prefill = []
    used_tokens = 0

    # 1. 清理完成请求，释放 KV blocks。
    still_running = []
    for req in running:
        if req.finished or req.aborted:
            block_manager.free(req)
        else:
            still_running.append(req)
    running = still_running

    # 2. 优先给已有 running 请求安排 decode，保证流式输出。
    for req in running:
        if used_tokens + 1 > max_tokens:
            break
        if not block_manager.can_append_slot(req):
            continue
        scheduled_decode.append(req)
        used_tokens += 1

    # 3. 用剩余 budget 接纳 waiting 请求做 prefill。
    while waiting_queue:
        req = waiting_queue.peek()
        prompt_len = len(req.input_ids)
        if used_tokens + prompt_len > max_tokens:
            break
        if not block_manager.can_allocate(req, prompt_len):
            break

        req = waiting_queue.pop()
        block_manager.allocate(req, prompt_len)
        scheduled_prefill.append(req)
        used_tokens += prompt_len

    # 4. 执行模型。
    outputs = run_model_forward(scheduled_prefill, scheduled_decode)

    # 5. 更新状态、采样 token、加入 running。
    update_requests(outputs)
    running.extend(req for req in scheduled_prefill if not req.finished)

    return running
```

真实系统会复杂得多，例如支持多 beam、LoRA、多模态、prefix cache、chunked prefill、speculative decoding、优先级和抢占，但主干逻辑就是这几步。

## 19.9 为什么 continuous batching 能提高吞吐

Continuous batching 提高吞吐的原因不是魔法，而是减少了几个浪费。

第一，减少 batch 空洞。

短请求完成后，可以在后续 iteration 中移出，新的 waiting 请求可以补入，不必让已完成请求占着 batch 位置。

第二，减少排队等待。

新请求不必等当前整批请求全部结束，只要下一轮调度有 token budget 和 KV block budget，就可以进入 prefill。

第三，提高 GPU 利用率。

decode 单步计算粒度小，如果 batch 太小，GPU 利用率低。continuous batching 尽量让每轮都有足够请求参与。

第四，配合 PagedAttention 提高可调度并发。

PagedAttention 降低 KV Cache 碎片和浪费，block manager 让 cache 动态分配释放，scheduler 才能安全地把更多请求放进 running set。

也就是说：

```text
PagedAttention 解决 cache 怎么放
Block Manager 解决 cache 怎么分配和释放
Continuous Batching 解决每一轮谁来跑
```

三者合起来，才构成 vLLM-like serving engine 的核心。

可以用一个很小的式子量化 fixed batch 的空洞。假设固定 batch 中请求集合为 `S`，请求 `i` 实际输出 `y_i` 个 token，则静态 batch decode 行数近似是：

$$
W_{\mathrm{static}}=|S|\max_{i\in S}y_i
$$

真正有效输出行数是：

$$
W_{\mathrm{valid}}=\sum_{i\in S}y_i
$$

静态 batch 的空洞行数是：

$$
W_{\mathrm{hole}}=W_{\mathrm{static}}-W_{\mathrm{valid}}
$$

continuous batching 的目标不是让 `W_{\mathrm{valid}}` 消失，而是让已经完成的请求尽快退出 execution batch，让后续 iteration 的有效行比例更高。

## 19.10 它的代价和副作用

Continuous batching 不是只有好处。

代价一：scheduler 更复杂。

系统需要维护 waiting、running、finished、aborted 等状态，还要在每轮处理资源分配、释放和异常清理。

代价二：延迟和吞吐有 trade-off。

如果每轮都尽量塞满 token budget，吞吐可能更高，但单个请求的 TTFT 或 TPOT 可能变差。

代价三：长 prompt 可能影响 decode 稳定性。

一个很长的 prefill 加入本轮，会占用大量计算时间，导致已有 streaming 请求下一 token 返回变慢。

代价四：调度策略会影响公平性。

如果总是优先短请求，长请求可能饥饿。如果总是 FIFO，短请求可能被长 prompt 阻塞。

代价五：实现难度高。

要正确处理请求取消、客户端断连、OOM、超时、prefix sharing、block ref count 和 streaming 顺序。

所以线上系统通常要通过参数和策略调优，而不是简单地“开大 batch”。

## 19.11 常见调度策略

Continuous batching 只是机制，具体策略可以有很多种。

策略一：FIFO。

先来的 waiting 请求先 prefill，简单公平，但可能被长 prompt 阻塞。

策略二：decode-first。

优先保证 running 请求每轮 decode，减少流式输出卡顿，再用剩余 budget 做 prefill。

策略三：prefill-first。

优先降低新请求 TTFT，但高负载下可能让已有请求 TPOT 抖动。

策略四：token-budget based。

每轮严格控制总 token 数，避免某一轮过大导致延迟尖刺。

策略五：length-aware。

根据 prompt 长度、已生成长度或预估输出长度做分桶，减少长短请求互相影响。

策略六：priority-aware。

按租户等级、业务优先级、deadline 或 SLA 调度，但要防止低优先级请求长期饥饿。

真实 vLLM-like 系统一般不是单一策略，而是多种约束叠加：

```text
FIFO fairness + token budget + KV block budget + max running seqs + prefill/decode policy
```

## 19.12 TTFT、TPOT 和吞吐如何被影响

Continuous batching 会同时影响三个指标。

TTFT 取决于：

1. 请求在 waiting queue 中等多久。
2. scheduler 多快给它安排 prefill。
3. prefill 是否被长队列或 token budget 阻塞。
4. prefill 执行时间和 first token sampling 时间。

TPOT 取决于：

1. running 请求是否每轮都能 decode。
2. 每轮 batch 中是否混入过大的 prefill。
3. decode batch size 是否稳定。
4. GPU kernel、KV Cache 读取和采样开销。

吞吐取决于：

1. 每轮有效 token 数。
2. GPU 利用率。
3. KV Cache 显存是否限制并发。
4. scheduler 是否能持续填充 batch。

这三个指标经常互相拉扯。

例如，提高 `max_num_batched_tokens` 可能提升吞吐，但也可能让单轮执行时间变长，导致 TPOT 抖动。

降低 `max_num_batched_tokens` 可能让流式输出更平滑，但 GPU 利用率下降，总吞吐降低。

面试里不要只说“continuous batching 提升吞吐”，更好的回答是：它通过动态填充 batch 提升吞吐，但需要用 token budget 和 cache budget 控制 TTFT、TPOT 和显存风险。

如果请求 `i` 的到达时间是 `a_i`，客户端看到 first token 的时间是 `f_i`，第 `j` 个输出 token 可见时间是 `s_{i,j}`，输出 token 数是 `y_i`，则：

$$
T_{\mathrm{ttft},i}=f_i-a_i
$$

当 `y_i>1` 时，平均 TPOT 可以写成：

$$
T_{\mathrm{tpot},i}=\frac{s_{i,y_i}-s_{i,1}}{y_i-1}
$$

单轮 iteration 的吞吐可以看成：

$$
Q_{\mathrm{tok},\tau}=\frac{N_{\mathrm{dec},\tau}+N_{\mathrm{pre},\tau}}{\Delta t_{\tau}}
$$

但线上调优不能只最大化 `Q_{\mathrm{tok},\tau}`。如果 `N_{\mathrm{pre},\tau}` 太大，单轮 `\Delta t_{\tau}` 会变长，已有 streaming 请求的 TPOT 可能变差。

## 19.13 和 dynamic batching 的区别

很多人会把 dynamic batching 和 continuous batching 混在一起。

普通 dynamic batching 常见于传统在线推理：

```text
在短时间窗口内收集请求 -> 拼成 batch -> forward 一次 -> 返回
```

它解决的是“如何把同时到达或近似同时到达的请求拼起来”。

Continuous batching 面向自回归生成：

```text
每一轮生成都重新组织 batch，请求可以在 decode 过程中加入或退出
```

关键差异：

| 对比点 | Dynamic Batching | Continuous Batching |
| --- | --- | --- |
| 典型任务 | 单次 forward 推理 | 多轮自回归生成 |
| 调度粒度 | 请求到达窗口 | 模型迭代 step |
| batch 是否长期变化 | 一般 forward 前确定 | 每轮都可能变化 |
| 是否处理 KV Cache 增长 | 通常不需要 | 必须处理 |
| 核心指标 | 单次延迟和吞吐 | TTFT、TPOT、吞吐、显存 |

所以 continuous batching 可以看作是为 LLM 自回归 workload 定制的动态调度机制。

## 19.14 面向专家：iteration-level scheduling 的本质

从系统角度看，iteration-level scheduling 做了一个重要拆分：

```text
request lifetime != execution batch lifetime
```

请求的生命周期可能持续几秒甚至几十秒，但 execution batch 只存在一个 engine iteration。

这带来一个好处：scheduler 每轮都有机会重新优化资源使用。

但也带来一个约束：请求状态必须跨 iteration 持久保存。

需要持久保存的状态包括：

1. 已输入 token 和已生成 token。
2. sampling 参数。
3. block table。
4. 当前长度和已生成长度。
5. 是否遇到 EOS 或 stop string。
6. streaming offset。
7. 请求优先级和超时信息。

因此，continuous batching 的本质不是“把 requests 拼 batch”这么简单，而是：

```text
用持久化 request state + 短生命周期 execution batch，解耦请求生命周期和 GPU 执行批次。
```

这个视角非常适合系统设计面试。

## 19.15 最小实现时容易踩的坑

坑一：finished 请求没有及时释放 KV blocks。

表现是请求数不高，但 free blocks 越来越少，最后 OOM。

坑二：只处理正常结束，忘了取消和异常路径。

客户端断开、超时、采样异常、模型执行异常，都要进入 cleanup。

坑三：prefill 长请求把 decode 卡住。

表现是 tokens/s 看起来不低，但用户看到流式输出一顿一顿。

坑四：只限制请求数，不限制 token 数。

表现是同样并发数下，短 prompt 正常，长 prompt 直接延迟尖刺或 OOM。

坑五：decode 追加 block 失败后状态不一致。

如果 append slot 失败，不能半更新 request length 或 block table，否则后续 attention 会读错位置。

坑六：streaming 顺序和调度顺序混乱。

调度可以乱序，但每个请求内部输出 token 必须按生成顺序返回。

## 19.16 面试官会怎么问

问题一：什么是 continuous batching？

回答要点：它是在每个模型迭代 step 重新组织 batch 的调度机制，允许请求动态加入和退出，适合 LLM 多轮自回归生成。

问题二：continuous batching 和普通 dynamic batching 有什么区别？

回答要点：dynamic batching 多数是在 forward 前收集请求；continuous batching 是每个 decode iteration 都调度，必须维护跨 iteration 的 KV Cache 和 request state。

问题三：为什么它能提升吞吐？

回答要点：它减少固定 batch 中 finished 请求造成的空洞，让新请求可以补入 running workload，提升 GPU 利用率；配合 PagedAttention 可以承载更多并发。

问题四：为什么需要 token budget？

回答要点：请求数不能反映实际计算量，长 prompt prefill 可能远大于短请求 decode，所以要按本轮 token 数控制调度规模。

问题五：为什么需要 KV block budget？

回答要点：能调度 token 不代表 KV Cache 放得下。prefill 需要分配 prompt blocks，decode 可能追加新 block，必须在执行前判断 free blocks。

问题六：如何平衡 TTFT 和 TPOT？

回答要点：prefill 影响新请求 TTFT，decode 影响已有请求 TPOT。常见策略是 decode-first 加受限 prefill，或通过 chunked prefill 控制长 prompt 对 decode 的阻塞。

## 19.17 标准回答模板

如果面试官问“vLLM 的 continuous batching 是什么”，可以这样回答：

```text
Continuous batching 是面向自回归 LLM serving 的动态调度机制。

传统 batching 往往把一组请求固定在一起，等整批请求完成后再处理下一批。但 LLM 生成是多轮迭代过程，不同请求输出长度不同，短请求会先结束，新请求也会不断到达。如果 batch 固定不变，就会出现 batch 空洞、GPU 利用率下降和新请求 TTFT 变差。

Continuous batching 把调度粒度降低到每个模型 iteration。每一轮 scheduler 都可以移除 finished 请求，接纳 waiting 请求，选择 running 请求继续 decode，并根据 token budget 和 KV cache block budget 控制本轮 workload。它通常需要和 PagedAttention、KV Cache Block Manager 配合，因为请求的 KV Cache 要能动态增长、释放和映射。

它的收益是提高吞吐和显存利用率，代价是 scheduler 更复杂，并且需要在 TTFT、TPOT、吞吐、公平性和 OOM 风险之间做 trade-off。
```

## 19.18 Continuous Batching 公式、调度 trace 和可运行 demo

把前面的约束合在一起，一个教学版 continuous batching gate 可以写成：

$$
G_{\mathrm{cb}}=G_{\mathrm{iter}}G_{\mathrm{tok}}G_{\mathrm{kv}}G_{\mathrm{decode}}G_{\mathrm{cleanup}}G_{\mathrm{metric}}
$$

其中：

1. `G_{\mathrm{iter}}`：请求可以在不同 iteration 动态到达、进入、退出。
2. `G_{\mathrm{tok}}`：每轮 decode + prefill 不超过 token budget。
3. `G_{\mathrm{kv}}`：prefill 和 decode 追加 KV 时不超过 free block budget。
4. `G_{\mathrm{decode}}`：已有 running 请求能优先获得 decode 进度，避免 streaming 卡顿。
5. `G_{\mathrm{cleanup}}`：finished / cancelled 请求会释放 KV blocks。
6. `G_{\mathrm{metric}}`：trace 能复盘 waiting、running、deferred reason、TTFT 和 block 使用。

下面这个 0 依赖 demo 模拟 5 个请求：A、B 同时到达，C、D、E 后续到达；scheduler 采用 decode-first；prefill 受 token budget 约束；KV block 不足时请求继续等待；E 在等待中取消；最终输出 trace、TTFT、静态 batch 空洞和门禁。

```python
from dataclasses import dataclass, field
from math import ceil


@dataclass
class Request:
    request_id: str
    arrival: int
    prompt_tokens: int
    planned_output: list
    cancel_at: int = None
    state: str = "new"
    prefill_start: int = None
    first_token_step: int = None
    finish_step: int = None
    finish_reason: str = None
    emitted: list = field(default_factory=list)


class ToyBlockManager:
    def __init__(self, total_blocks, block_size):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_blocks = list(range(total_blocks))
        self.tables = {}
        self.lengths = {}
        self.max_used = 0

    def required_blocks(self, tokens):
        return ceil(tokens / self.block_size)

    def used_count(self):
        return self.total_blocks - len(self.free_blocks)

    def _remember_peak(self):
        self.max_used = max(self.max_used, self.used_count())

    def can_allocate(self, tokens):
        return self.required_blocks(tokens) <= len(self.free_blocks)

    def allocate(self, request_id, tokens):
        need = self.required_blocks(tokens)
        if need > len(self.free_blocks):
            return False
        self.tables[request_id] = [self.free_blocks.pop(0) for _ in range(need)]
        self.lengths[request_id] = tokens
        self._remember_peak()
        return True

    def can_append(self, request_id):
        length = self.lengths[request_id]
        return length % self.block_size != 0 or bool(self.free_blocks)

    def append_one(self, request_id):
        length = self.lengths[request_id]
        if length % self.block_size == 0:
            if not self.free_blocks:
                return False
            self.tables[request_id].append(self.free_blocks.pop(0))
        self.lengths[request_id] += 1
        self._remember_peak()
        return True

    def release(self, request_id):
        released = self.tables.pop(request_id, [])
        self.lengths.pop(request_id, None)
        self.free_blocks.extend(released)
        return released


class ToyContinuousBatcher:
    def __init__(self, requests, max_tokens, max_running, block_manager):
        self.requests = sorted(requests, key=lambda req: (req.arrival, req.request_id))
        self.future = list(self.requests)
        self.max_tokens = max_tokens
        self.max_running = max_running
        self.block_manager = block_manager
        self.time = 0
        self.waiting = []
        self.running = []
        self.finished = []
        self.cancelled = []
        self.trace = []
        self.deferred = []

    def add_arrivals(self):
        arrived = []
        while self.future and self.future[0].arrival <= self.time:
            req = self.future.pop(0)
            req.state = "waiting"
            self.waiting.append(req)
            arrived.append(req.request_id)
        return arrived

    def cancel_ready_requests(self):
        cancelled = []
        for req in list(self.waiting) + list(self.running):
            if req.cancel_at is not None and req.cancel_at <= self.time:
                if req in self.waiting:
                    self.waiting.remove(req)
                if req in self.running:
                    self.running.remove(req)
                    self.block_manager.release(req.request_id)
                req.state = "cancelled"
                req.finish_reason = "client_cancelled"
                req.finish_step = self.time
                self.cancelled.append(req)
                cancelled.append(req.request_id)
        return cancelled

    def finish(self, req, reason="eos"):
        if req in self.running:
            self.running.remove(req)
        req.state = "finished"
        req.finish_reason = reason
        req.finish_step = self.time
        self.block_manager.release(req.request_id)
        self.finished.append(req)

    def schedule_step(self):
        arrived = self.add_arrivals()
        cancelled = self.cancel_ready_requests()
        scheduled_decode = []
        scheduled_prefill = []
        deferred = []
        used_tokens = 0

        for req in list(self.running):
            if used_tokens + 1 > self.max_tokens:
                deferred.append((req.request_id, "token_budget_decode"))
                continue
            if not self.block_manager.can_append(req.request_id):
                deferred.append((req.request_id, "kv_decode"))
                continue
            self.block_manager.append_one(req.request_id)
            scheduled_decode.append(req)
            used_tokens += 1

        while self.waiting:
            req = self.waiting[0]
            if len(self.running) + len(scheduled_prefill) >= self.max_running:
                deferred.append((req.request_id, "active_sequence_limit"))
                break
            if used_tokens + req.prompt_tokens > self.max_tokens:
                deferred.append((req.request_id, "token_budget_prefill"))
                break
            if not self.block_manager.can_allocate(req.prompt_tokens):
                deferred.append((req.request_id, "kv_prefill"))
                break
            self.waiting.pop(0)
            self.block_manager.allocate(req.request_id, req.prompt_tokens)
            scheduled_prefill.append(req)
            used_tokens += req.prompt_tokens

        for req in scheduled_decode:
            next_token = req.planned_output[len(req.emitted)]
            req.emitted.append(next_token)
            if len(req.emitted) == len(req.planned_output):
                self.finish(req)

        for req in scheduled_prefill:
            req.state = "running"
            req.prefill_start = self.time
            req.first_token_step = self.time
            req.emitted.append(req.planned_output[0])
            if len(req.emitted) == len(req.planned_output):
                self.finish(req)
            else:
                self.running.append(req)

        self.deferred.extend((self.time, req_id, reason) for req_id, reason in deferred)
        self.trace.append(
            {
                "step": self.time,
                "arrived": arrived,
                "cancelled": cancelled,
                "decode": [req.request_id for req in scheduled_decode],
                "prefill": [req.request_id for req in scheduled_prefill],
                "used_tokens": used_tokens,
                "waiting": [req.request_id for req in self.waiting],
                "running": [req.request_id for req in self.running],
                "free_blocks": len(self.block_manager.free_blocks),
                "deferred": deferred,
            }
        )
        self.time += 1

    def run(self):
        while self.future or self.waiting or self.running:
            self.schedule_step()
            if self.time > 20:
                raise RuntimeError("scheduler did not converge")

    def report(self):
        output_rows = sum(len(req.emitted) for req in self.requests)
        static_rows = len(self.requests) * max(len(req.planned_output) for req in self.requests)
        queue_wait = {
            req.request_id: req.prefill_start - req.arrival
            for req in self.requests
            if req.prefill_start is not None
        }
        ttft = {
            req.request_id: req.first_token_step - req.arrival
            for req in self.requests
            if req.first_token_step is not None
        }
        reasons = sorted({reason for _, _, reason in self.deferred})
        terminal = all(req.finish_reason is not None for req in self.requests)
        gates = {
            "dynamic_arrivals_seen": any(item["arrived"] for item in self.trace[1:]),
            "decode_first_with_prefill": any(item["decode"] and item["prefill"] for item in self.trace),
            "token_budget_respected": all(item["used_tokens"] <= self.max_tokens for item in self.trace),
            "kv_budget_deferred": "kv_prefill" in reasons or "kv_decode" in reasons,
            "cleanup_released_blocks": terminal and self.block_manager.used_count() == 0,
            "metrics_trace_ready": bool(queue_wait) and bool(ttft) and self.block_manager.max_used > 0,
        }
        gates["continuous_batching_gate"] = all(gates.values())
        summary = {
            "finished_order": [req.request_id for req in sorted(self.finished, key=lambda r: (r.finish_step, r.request_id))],
            "cancelled": [req.request_id for req in self.cancelled],
            "queue_wait_steps": queue_wait,
            "ttft_steps": ttft,
            "static_decode_rows": static_rows,
            "continuous_output_rows": output_rows,
            "saved_rows": static_rows - output_rows,
            "max_kv_blocks_used": self.block_manager.max_used,
            "deferred_reasons": reasons,
            "trace_tail": self.trace[-4:],
        }
        return summary, gates


requests = [
    Request("A", arrival=0, prompt_tokens=3, planned_output=["a1", "a2", "a3"]),
    Request("B", arrival=0, prompt_tokens=6, planned_output=["b1", "b2", "b3", "b4"]),
    Request("C", arrival=1, prompt_tokens=2, planned_output=["c1"]),
    Request("D", arrival=2, prompt_tokens=5, planned_output=["d1", "d2"]),
    Request("E", arrival=3, prompt_tokens=3, planned_output=["e1", "e2"], cancel_at=5),
]

scheduler = ToyContinuousBatcher(
    requests=requests,
    max_tokens=8,
    max_running=4,
    block_manager=ToyBlockManager(total_blocks=5, block_size=4),
)
scheduler.run()
summary, gates = scheduler.report()

print("continuous_batching_summary=", summary)
print("continuous_batching_gates=", gates)
```

一份合理输出应当能看出这些事实：

1. `finished_order` 不是到达顺序，说明请求在不同 iteration 动态退出。
2. `deferred_reasons` 同时包含 `token_budget_prefill` 和 `kv_prefill`，说明调度不只看请求数。
3. `static_decode_rows` 大于 `continuous_output_rows`，能量化固定 batch 的空洞。
4. `cleanup_released_blocks=True`，说明完成和取消路径没有泄漏 KV blocks。
5. `continuous_batching_gate=True`，说明动态到达、decode-first、token budget、KV budget、cleanup 和 metrics 都能被 trace 复盘。

## 19.19 小练习

1. 画一个时间线，包含 4 个请求 A、B、C、D，其中 A 短输出，B 长输出，C 和 D 中途到达，分别画出 static batching 和 continuous batching 的执行方式。
2. 写一个简化 scheduler，输入 waiting queue、running requests、`max_num_batched_tokens` 和 free block 数，输出本轮 prefill 和 decode 请求。
3. 解释为什么只设置 `max_running_requests=32` 仍然可能 OOM。
4. 设计一个实验，观察 `max_num_batched_tokens` 增大后 TTFT、TPOT 和 tokens/s 的变化。
5. 思考长 prompt prefill 为什么会让 streaming decode 卡顿，并提出一个缓解方案。

## 19.20 本章总结

Continuous batching 是 vLLM-like serving engine 的核心能力之一。

它解决的问题是：自回归 LLM 请求长度不同、到达时间不同、结束时间不同，如果 batch 固定不变，会浪费 GPU、拉高 TTFT，并降低系统吞吐。

它的关键思想是：把调度粒度从 request-level 降到 iteration-level，每一轮模型执行前都重新选择要 prefill 和 decode 的请求。

真正可用的 continuous batching 不能只看请求数，还必须结合 token budget 和 KV block budget。token budget 控制本轮计算量，KV block budget 控制显存安全。PagedAttention、Block Manager 和 Continuous Batching 共同构成 vLLM 高吞吐 serving 的主干。

下一章会继续进入 vLLM 请求调度流程，具体拆解一个请求从进入 vLLM engine 到完成输出，中间会经历哪些状态迁移、调度决策和执行步骤。
