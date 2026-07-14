# 第 20 章 vLLM 请求调度流程

上一章讲了 continuous batching 与 iteration-level scheduling：LLM serving engine 不应该把一组请求固定成 batch 跑到结束，而应该在每个 engine step 重新决定哪些请求 prefill、哪些请求 decode、哪些请求退出。

本章继续往下走：一个请求进入 vLLM-like engine 后，到底经历哪些模块、哪些状态、哪些调度决策，最后如何把 token 流式返回给用户？

一句话概括：

> vLLM 的请求调度流程可以理解为：入口层接收和预处理请求，engine core 维护请求状态并执行 scheduler，KV cache manager 分配和释放 blocks，worker/model runner 执行模型 forward，output processor 把模型输出转换成用户可见的 token 或文本。

## 20.0 本讲资料边界与第二轮精修口径

本讲按第二轮精修要求做过资料校准，主要参考四类公开资料：

1. vLLM Architecture Overview 对 API server process、engine core process、GPU worker process、LLMEngine、AsyncLLMEngine、worker 和 model runner 的模块边界说明。
2. vLLM V1 guide 对 V1 core engine、chunked prefill、统一 token budget 调度和功能支持边界的说明。
3. vLLM optimization / tuning 文档对 preemption、KV cache capacity、`max_num_seqs`、`max_num_batched_tokens`、chunked prefill 和 decode-first 策略的说明。
4. vLLM metrics 文档对 TTFT、inter-token latency / TPOT、queue time、prefill time、decode time、running / waiting 请求数、KV cache usage 和 engine core iteration 统计的观测口径。

本章只讲 vLLM-like 请求调度流程的教学抽象，不绑定某个 vLLM 版本的真实类名、进程拓扑、ZMQ 通信细节、CUDA graph、scheduler 源码、model runner 内部张量格式、OpenAI server 参数全集、LoRA、多模态、speculative decoding、structured output 或分布式 data parallel / tensor parallel 实现。

本章给出的公式和 demo 只用于验证一条最小请求链路：API 输入处理、engine request、waiting queue、scheduler output、KV block 分配、model runner metadata、sampler / output processor、streaming / finish reason、abort cleanup 和 metrics trace。真实系统会把这些逻辑拆到更多进程、线程、队列和异步事件里。

参考资料：

1. vLLM Architecture Overview：<https://docs.vllm.ai/en/latest/design/arch_overview/>
2. vLLM V1 guide：<https://docs.vllm.ai/en/latest/usage/v1_guide/>
3. vLLM optimization and tuning：<https://docs.vllm.ai/en/latest/configuration/optimization/>
4. vLLM metrics：<https://docs.vllm.ai/en/latest/design/metrics/>

## 20.1 本章目标

读完本章，你应该能讲清：

1. vLLM-like 系统中 API server、LLMEngine、engine core、scheduler、worker、model runner、output processor 各自负责什么。
2. 一个请求从 HTTP 请求到 first token 再到 finished 的完整路径。
3. waiting、running、scheduled、finished、aborted 等状态如何变化。
4. scheduler 每一轮如何结合 token budget、KV block budget 和请求状态做选择。
5. 为什么输出处理和请求终止也会影响调度。
6. 面试中如何画出 vLLM 请求生命周期图。

## 20.2 先建立模块图

不同 vLLM 版本的内部类名会变化，但核心模块边界比较稳定。

可以先用这张简化图理解：

```text
Client
  |
  v
API Server / Python LLM API
  |
  v
Input Processor
  |
  v
Engine Core
  |-- Scheduler
  |-- KV Cache Manager / Block Manager
  |-- Request State Store
  |
  v
Executor / GPU Workers
  |
  v
Model Runner / Model Forward
  |
  v
Sampler
  |
  v
Output Processor / Detokenizer / Streamer
  |
  v
Client
```

每个模块解决一个问题。

API server 负责接入请求、校验参数、组织 OpenAI-compatible API、处理 streaming 连接。

Input processor 负责把原始 prompt、chat messages、多模态输入和 sampling 参数整理成 engine 能理解的请求对象。

Engine core 负责真正的调度主循环：接收请求、维护状态、调用 scheduler、管理 KV cache、派发模型执行。

Scheduler 负责每一轮选择要执行的请求。

KV cache manager 负责 blocks 的分配、追加、释放和可能的 prefix 复用。

Worker 和 model runner 负责在 GPU 上执行模型 forward。

Output processor 负责把 token id、logprob、finish reason、stop string 等转换成最终输出，并处理请求结束和中止。

## 20.3 请求进入系统：从原始输入到 Engine Request

用户发来的请求可能长这样：

```json
{
  "model": "some-model",
  "messages": [
    {"role": "user", "content": "Explain KV cache in simple words."}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 256,
  "stream": true
}
```

在进入 scheduler 之前，它通常要经历几步处理。

第一步，API 层校验请求。

检查内容包括：

1. model 是否存在。
2. `max_tokens`、temperature、top-p 等参数是否合法。
3. prompt 或 messages 是否为空。
4. 输入长度是否超过限制。
5. 是否启用 streaming。
6. 是否包含不支持的多模态字段或工具字段。

第二步，渲染 prompt。

Chat messages 需要根据 chat template 转成模型实际看到的文本或 token 序列。

例如：

```text
<|user|>
Explain KV cache in simple words.
<|assistant|>
```

第三步，tokenization。

把文本转成 token ids：

```text
prompt_text -> input_ids
```

第四步，构造 engine request。

一个最小 request state 可以包含：

```python
class RequestState:
    def __init__(self, request_id, input_ids, sampling_params, arrival_time):
        self.request_id = request_id
        self.input_ids = input_ids
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time

        self.status = "WAITING"
        self.num_computed_tokens = 0
        self.output_token_ids = []
        self.block_table = []
        self.first_token_time = None
        self.finished_time = None
        self.finish_reason = None
```

真实 vLLM 的数据结构更复杂，但主干字段离不开这些：输入 tokens、采样参数、状态、KV block 映射、输出 tokens 和时间统计。

抽象成公式，一个 engine request 可以写成：

$$
R_i=(u_i,a_i,x_i,\theta_i,z_i,P_i,Y_i,b_i,t_i,m_i)
$$

其中 `u_i` 是 request id，`a_i` 是到达时间，`x_i` 是输入 token 序列，`\theta_i` 是采样参数，`z_i` 是状态，`P_i` 是 block table，`Y_i` 是已经输出的 tokens，`b_i` 是 streaming buffer，`t_i` 是时间戳集合，`m_i` 是 metrics / trace 字段。

请求对象完整率可以写成：

$$
C_{\mathrm{req}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[u_i,a_i,x_i,\theta_i,z_i,P_i,Y_i,t_i,m_i\ \mathrm{ready}]
$$

如果缺 arrival time、状态或 block table，后面的 scheduler、metrics 和 cleanup 都会失真。

## 20.4 请求进入 waiting queue

构造好 engine request 后，请求不会马上执行模型。

它会先进入 engine core 的 waiting queue：

```text
new request -> WAITING -> waiting queue
```

waiting queue 的作用是把“请求到达”和“GPU 执行”解耦。

请求到达是外部事件，可能非常不均匀。

GPU 执行是 engine step 驱动的，需要按 token budget、KV block budget、优先级和并发限制来选择。

如果没有 waiting queue，请求一来就尝试执行，系统很容易出现：

1. 多个请求争抢 GPU。
2. KV Cache 分配没有统一入口。
3. prefill 插入时机混乱。
4. streaming decode 被长 prompt 打断。

所以 waiting queue 是 scheduler 的入口缓冲区。

## 20.5 Engine Core 的主循环

vLLM 架构文档中，engine core 是运行 scheduler、管理 KV cache、协调 GPU workers 的核心进程或核心组件。

可以把 engine core 理解成一个不断执行的循环：

```python
while True:
    new_requests = recv_new_requests()
    add_to_waiting_queue(new_requests)

    scheduler_output = scheduler.schedule(
        waiting_queue=waiting_queue,
        running_requests=running_requests,
        block_manager=block_manager,
    )

    model_outputs = executor.execute_model(scheduler_output)
    engine_outputs = process_model_outputs(model_outputs)
    send_outputs_to_frontend(engine_outputs)
```

这个循环看起来简单，但每一步都很关键。

`recv_new_requests()` 处理新请求和 abort 请求。

`scheduler.schedule()` 决定本轮谁跑。

`executor.execute_model()` 把本轮 workload 发给 GPU workers。

`process_model_outputs()` 处理采样结果、结束条件和状态更新。

`send_outputs_to_frontend()` 把输出交给 API server 或 output processor。

## 20.6 Scheduler 的输入

Scheduler 每轮需要的信息很多。

最小输入包括：

1. waiting requests：还没有完成 prefill 的请求。
2. running requests：已经进入系统、需要继续 decode 的请求。
3. KV block 状态：free blocks、已分配 blocks、block table、ref count。
4. token budget：本轮最多处理多少 token。
5. sequence budget：本轮最多处理多少序列。
6. priority 或 arrival time：用于公平性和优先级。
7. 当前 engine 状态：是否有 pending abort、是否处于 sleep/profile 等特殊状态。

用伪代码表示：

```python
class SchedulerInput:
    def __init__(self, waiting, running, block_manager, config):
        self.waiting = waiting
        self.running = running
        self.block_manager = block_manager
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_num_seqs = config.max_num_seqs
```

这里的 `max_num_batched_tokens` 控制计算规模，`max_num_seqs` 控制本轮 sequence 数，block manager 控制 KV Cache 安全。

本轮 scheduler 的基础预算可以写成：

$$
G_{\mathrm{sched},\tau}=\mathbf{1}[N_{\mathrm{tok},\tau}\le C_{\mathrm{tok}}]\mathbf{1}[N_{\mathrm{seq},\tau}\le C_{\mathrm{seq}}]\mathbf{1}[N_{\mathrm{block},\tau}\le F_{\tau}]
$$

这里 `N_{\mathrm{tok},\tau}` 是本轮 prefill + decode token 数，`N_{\mathrm{seq},\tau}` 是本轮执行序列数，`N_{\mathrm{block},\tau}` 是本轮新增 KV blocks，`F_{\tau}` 是调度前可用 free blocks。

## 20.7 Scheduler 的输出

Scheduler 输出的不是普通文本，而是“本轮执行计划”。

一个简化输出可以包含：

```python
class SchedulerOutput:
    def __init__(self):
        self.prefill_requests = []
        self.decode_requests = []
        self.blocks_to_allocate = {}
        self.blocks_to_append = {}
        self.requests_to_free = []
```

实际系统中还会包含更多信息，例如：

1. 每个请求本轮要处理多少 tokens。
2. attention metadata。
3. block tables。
4. sampling metadata。
5. LoRA 或 adapter 信息。
6. 多模态 encoder cache 信息。
7. speculative decoding 相关字段。

但从面试角度，最重要的是说明：scheduler 的输出要同时喂给模型执行和 KV Cache 管理。

如果只输出“请求列表”，model runner 不知道每个 token 对应哪些 block，也不知道哪些请求是 prefill、哪些是 decode。

因此 scheduler output 至少要满足 execution metadata 完整性。设本轮被调度请求集合为 `S_{\tau}`，每个请求要有 input ids、positions、block table、slot mapping 和 phase：

$$
G_{\mathrm{meta},\tau}=\prod_{i\in S_{\tau}}G_{\mathrm{ids},i}G_{\mathrm{pos},i}G_{\mathrm{block},i}G_{\mathrm{slot},i}G_{\mathrm{phase},i}
$$

如果 `G_{\mathrm{meta},\tau}=0`，模型 runner 即使拿到请求列表，也无法安全执行 attention 和 KV 写入。

## 20.8 请求状态机

为了理解调度流程，必须画出请求状态机。

简化状态：

```text
WAITING -> PREFILL -> RUNNING/DECODING -> FINISHED
   |          |              |               ^
   |          |              v               |
   +----------+----------> ABORTED / FAILED --+
```

更细一点：

WAITING：请求已经进入 engine，但还没有被 scheduler 选中 prefill。

PREFILL：本轮正在处理 prompt tokens，并写入 prompt KV Cache。

RUNNING 或 DECODING：prompt 已处理，后续每轮生成一个或多个 token。

FINISHED：遇到 EOS、stop string、达到 max tokens 或其他正常结束条件。

ABORTED：客户端断开、用户取消、上层超时或 output processor 判定需要中止。

FAILED：模型执行、KV 分配或内部逻辑出现错误。

关键原则是：

```text
任何离开 WAITING/RUNNING 的终止状态，都必须释放资源。
```

资源包括 KV blocks、streaming buffer、请求状态、统计对象和可能的多模态缓存引用。

对于请求 `i`，如果终止状态集合记为 `Z_{\mathrm{term}}=\{\mathrm{FINISHED},\mathrm{ABORTED},\mathrm{FAILED}\}`，则 cleanup 门禁可以写成：

$$
G_{\mathrm{cleanup},i}=\mathbf{1}[z_i\in Z_{\mathrm{term}}\Rightarrow |P_i|=0]
$$

这句话的含义是：只要请求已经进入终态，它的 KV block table 就不应该再持有可用资源。

## 20.9 从 WAITING 到 PREFILL

一个 waiting 请求能不能进入 prefill，至少要过几道门。

第一道门：token budget。

如果 prompt 太长，本轮剩余 token budget 不够，就不能完整 prefill。

如果支持 chunked prefill，则可以只处理 prompt 的一部分。

第二道门：KV block budget。

block manager 要判断 prompt tokens 需要多少 blocks。

```text
required_blocks = ceil(prompt_len / block_size)
```

如果 free blocks 不够，请求继续留在 waiting queue。

第三道门：sequence budget。

如果本轮 scheduled sequences 已经达到上限，新请求不能加入。

第四道门：策略门。

如果当前系统 decode 压力很大，scheduler 可能暂时不接纳新的长 prompt，避免 TPOT 抖动。

通过这些检查后，scheduler 会把请求放入 prefill set，并让 block manager 分配 blocks。

```python
if can_fit_tokens(req) and can_allocate_blocks(req) and can_fit_seq(req):
    block_manager.allocate(req)
    schedule_prefill(req)
```

## 20.10 从 PREFILL 到 RUNNING

Prefill 执行完后，模型已经为 prompt 中的 token 写好了 KV Cache。

如果这是普通生成请求，prefill 之后通常会产生第一个可采样位置的 logits，然后 sampler 采样出 first token。

这时会发生几件事：

1. 记录 first token time，用于计算 TTFT。
2. 把 first token 加入 output token ids。
3. 检查 first token 是否触发 EOS 或 stop 条件。
4. 如果未结束，请求进入 running set。
5. 如果已结束，释放 KV blocks 并输出 final response。

状态变化：

```text
WAITING -> PREFILL -> RUNNING
```

或者极短生成时：

```text
WAITING -> PREFILL -> FINISHED
```

一个请求可能在 first token 就结束。例如 `max_tokens=1`，或者采样出的 token 正好是 EOS。

## 20.11 RUNNING 阶段：每轮 decode

进入 running set 后，请求会在后续 engine steps 中被反复调度。

每一轮 decode 通常做：

1. 判断本轮是否选择该请求。
2. 检查当前 KV block 是否还有 slot。
3. 如果 block 满了，追加新 physical block。
4. 准备 decode input token，一般是上一步生成的 token。
5. 构造 attention metadata 和 block table。
6. 执行模型 forward。
7. sampler 采样下一个 token。
8. 更新 request state。
9. streaming 输出增量文本。
10. 检查结束条件。

伪代码：

```python
for req in scheduled_decode:
    if block_manager.need_new_block(req):
        block_manager.append_block(req)

model_outputs = model_runner.forward(scheduled_decode)
next_tokens = sampler.sample(model_outputs.logits)

for req, token in zip(scheduled_decode, next_tokens):
    req.output_token_ids.append(token)
    req.num_computed_tokens += 1
    if should_stop(req, token):
        req.status = "FINISHED"
```

这里要注意：decode 阶段不是重新处理整个上下文，而是依赖 KV Cache 读取历史 K/V，只处理新 token。

## 20.12 Model Runner 看到的不是请求，而是执行 batch

在上层看，系统里有 request。

但在 model runner 看，最终要执行的是一组张量和 metadata。

例如：

```text
input_ids
positions
slot_mapping
block_tables
attention_metadata
sampling_metadata
```

这说明一件事：scheduler 选中请求后，还要把 request-level state 转换成 model-level batch。

这个转换非常关键。

请求状态里记录的是：

1. 这个请求已经生成多少 token。
2. 它的 block table 是什么。
3. 这轮要 prefill 还是 decode。
4. 采样参数是什么。

模型执行需要的是：

1. 本轮实际输入 token ids。
2. 每个 token 的 position。
3. 每个 token 的 KV 写入位置。
4. attention kernel 如何根据 block table 读取历史 KV。
5. 哪些 logits 需要采样。

这就是为什么 vLLM-like 系统里 scheduler 和 model runner 之间会有一层 execution metadata。

## 20.13 Output Processor 为什么也影响调度

很多人只关注 scheduler，却忽略 output processor。

但 output processor 会影响请求什么时候结束、什么时候 abort、什么时候释放资源。

它通常负责：

1. 把 token ids detokenize 成文本。
2. 处理 streaming 增量输出。
3. 检查 stop strings。
4. 检查 EOS。
5. 检查 max tokens。
6. 组织 logprobs。
7. 生成 finish reason。
8. 通知 engine core abort 某些请求。

例如 stop string 检查常常需要 detokenized text，而不是单个 token id。

如果 output processor 发现某个请求已经匹配 stop string，它需要告诉 engine core：这个请求后续不要再调度了。

流程类似：

```text
model output -> output processor -> stop matched -> abort/finish request -> engine core cleanup
```

所以请求结束不是 scheduler 单方面决定的，它也依赖输出处理逻辑。

## 20.14 Streaming 输出路径

在线 serving 中，用户通常不想等完整回答生成完，而是希望 token-by-token 或 chunk-by-chunk 返回。

Streaming 路径可以理解为：

```text
model logits -> sampled token id -> detokenize delta -> output queue -> HTTP SSE chunk -> client
```

每个请求需要维护 streaming offset，避免重复返回文本。

例如 tokenizer 可能存在这种情况：几个 token 合起来才形成一个完整中文字符、英文单词片段或特殊符号。detokenizer 需要保证输出给客户端的是合法的增量文本。

Streaming 对调度有两个影响。

第一，TPOT 变成用户可感知指标。

如果 running 请求没有稳定 decode，用户会看到输出卡顿。

第二，客户端断开会触发 abort。

如果用户关闭连接，API server 应通知 engine core 取消请求，释放 KV blocks。

否则 GPU 还在为一个没人接收的请求生成 token。

从 metrics 角度，request flow 至少要能还原三类时间：

$$
W_i=t^{\mathrm{prefill}}_i-a_i
$$

$$
T_{\mathrm{ttft},i}=t^{\mathrm{first}}_i-a_i
$$

$$
T_{\mathrm{e2e},i}=t^{\mathrm{done}}_i-a_i
$$

其中 `W_i` 是 queue wait，`T_{\mathrm{ttft},i}` 是首 token 可见时间，`T_{\mathrm{e2e},i}` 是请求结束时间。没有这些时间戳，scheduler 调优就只能凭感觉。

## 20.15 Abort 和 cleanup 流程

Abort 是线上系统非常重要但容易被忽视的路径。

Abort 可能来自：

1. 用户主动取消。
2. HTTP 连接断开。
3. 上层 gateway 超时。
4. output processor 匹配 stop string 后要求终止。
5. 管理接口要求取消某个 request id。
6. 系统内部错误。

Abort 后必须做几件事：

1. 从 waiting queue 移除请求。
2. 从 running set 移除请求。
3. 释放 KV blocks。
4. 清理 output buffer。
5. 清理统计和 tracing 状态。
6. 通知前端请求已结束或已取消。

伪代码：

```python
def abort_request(request_id):
    req = request_store.get(request_id)
    if req is None:
        return

    req.status = "ABORTED"
    waiting_queue.remove(request_id)
    running_set.remove(request_id)
    block_manager.free(req)
    output_processor.abort(request_id)
    request_store.delete(request_id)
```

工程上最怕的是“只改状态，不释放资源”。

这会造成 silent KV leak：请求看起来结束了，但 block 还被占着。

## 20.16 一个完整请求生命周期例子

假设请求 R 到达：

```text
prompt_len = 1000
max_tokens = 4
block_size = 16
```

它的生命周期可能是：

```text
t0: API server 收到请求 R
t1: input processor 渲染 chat template，tokenize 得到 1000 tokens
t2: engine core 把 R 放入 waiting queue
t3: scheduler 发现 token budget 和 KV blocks 都足够
t4: block manager 为 R 分配 63 个 blocks
t5: model runner 执行 prefill
t6: sampler 采样 first token，R 进入 running set，stream 第一个 token
t7: 下一轮 scheduler 选择 R decode，追加或复用当前 block slot，stream 第二个 token
t8: 再下一轮 decode，stream 第三个 token
t9: 再下一轮 decode，达到 max_tokens=4，R 标记 FINISHED
t10: block manager 释放 R 的 blocks
t11: output processor 返回 final finish reason
```

如果中途用户断开连接，则可能变成：

```text
t7: client disconnected
t8: API server 发送 abort
t9: engine core 标记 R 为 ABORTED
t10: block manager 释放 R 的 blocks
```

这两个路径都必须正确。

## 20.17 和第 19 章的关系

第 19 章讲的是调度机制：每个 iteration 动态组织 batch。

本章讲的是请求路径：请求如何进入、如何被调度、如何执行、如何输出、如何结束。

两者可以合成一句话：

```text
Continuous batching 是调度机制，请求生命周期管理是让这个机制能在线上正确运行的状态机。
```

只有调度机制，没有请求状态机，系统会不知道请求何时进入、何时结束、何时释放资源。

只有状态机，没有 continuous batching，系统又会退化成低效的固定 batch 或单请求生成。

## 20.18 常见工程坑

坑一：输入处理和 engine request 不一致。

例如 chat template 在 API 层和离线测试层不一致，导致同一个 prompt 在不同入口下 token ids 不同。

坑二：arrival time 没有传入 scheduler。

缺少 arrival time 后，很难计算 queue time 和 TTFT，也难以做公平调度。

坑三：output processor 判定 stop 后，没有通知 engine core。

表现是用户侧已经看到结束，但 engine 仍继续 decode。

坑四：abort 只清理 output queue，不清理 KV blocks。

表现是显存慢慢下降，最终 OOM。

坑五：model runner metadata 和 request state 不一致。

例如 block table 更新了，但 slot mapping 没同步，attention kernel 可能读写错误位置。

坑六：streaming delta 处理不当。

表现是返回重复文本、乱码、截断特殊 token，或者 stop string 被提前/延后触发。

## 20.19 面试官会怎么问

问题一：一个请求进入 vLLM 后经历哪些模块？

回答要点：API server 接收请求，input processor 渲染和 tokenization，engine core 维护请求状态，scheduler 选择 prefill/decode，KV cache manager 分配 blocks，worker/model runner 执行 forward，sampler 采样 token，output processor detokenize、streaming 和判断结束。

问题二：engine core 的职责是什么？

回答要点：engine core 是调度和执行协调中心，负责接收 engine requests、运行 scheduler、管理 KV cache、协调 GPU workers、处理 abort 和输出状态。

问题三：scheduler 输出什么？

回答要点：不是只输出请求列表，而是输出本轮执行计划，包括 prefill/decode 请求、token 数、block table、slot mapping、sampling metadata 和需要分配/释放的资源信息。

问题四：请求什么时候释放 KV Cache？

回答要点：正常结束、abort、failed、超时、客户端断连等终止路径都要释放。不能只在 EOS 时释放。

问题五：为什么 output processor 会影响调度？

回答要点：stop string、EOS、max tokens、streaming 状态和 abort 可能在 output processor 中判断，它会决定请求是否还需要继续进入下一轮 scheduler。

## 20.20 标准回答模板

如果面试官问“请讲一下 vLLM 的请求调度流程”，可以这样回答：

```text
一个请求进入 vLLM-like 系统后，首先由 API server 或 Python LLM 接口接收。输入会经过 input processor，完成参数校验、chat template 渲染、tokenization，并构造成 engine request。

这个 request 会进入 engine core 的 waiting queue。Engine core 持续运行调度循环，每一轮调用 scheduler，根据 waiting requests、running requests、token budget、sequence budget 和 KV cache block budget，选择本轮要 prefill 和 decode 的请求。

被选中的请求会由 KV cache manager 分配或追加 blocks，然后 scheduler 输出会被转换成 model runner 需要的 execution metadata，例如 input ids、positions、block tables、slot mapping 和 sampling metadata。GPU worker 执行模型 forward 后，sampler 采样新 token。

模型输出再经过 output processor，完成 detokenization、streaming、EOS 和 stop string 检查、finish reason 生成。如果请求结束或被 abort，engine core 必须释放对应 KV blocks 并清理请求状态。这个流程每个 engine step 重复一次，因此 vLLM 可以实现 continuous batching，让请求在不同 iteration 动态加入和退出。
```

## 20.21 请求调度流程公式、状态 trace 和可运行 demo

把本章的请求链路合成一个教学版总门禁，可以写成：

$$
G_{\mathrm{flow}}=G_{\mathrm{input}}G_{\mathrm{state}}G_{\mathrm{sched}}G_{\mathrm{meta}}G_{\mathrm{output}}G_{\mathrm{cleanup}}G_{\mathrm{metric}}
$$

其中：

1. `G_{\mathrm{input}}`：API / input processor 产出 engine request 和 input ids。
2. `G_{\mathrm{state}}`：请求状态按 WAITING、PREFILL、RUNNING、DECODING、FINISHED / ABORTED 迁移。
3. `G_{\mathrm{sched}}`：scheduler 每轮不超 token、sequence 和 KV block budget。
4. `G_{\mathrm{meta}}`：model runner metadata 含 input ids、positions、block table、slot mapping 和 phase。
5. `G_{\mathrm{output}}`：output processor 能处理 EOS、stop token / string、max tokens 和 streaming 增量。
6. `G_{\mathrm{cleanup}}`：FINISHED / ABORTED / FAILED 都释放 KV blocks。
7. `G_{\mathrm{metric}}`：queue wait、TTFT、E2E、trace 和 finish reason 可复盘。

下面这个 0 依赖 demo 模拟一个最小 vLLM-like 请求调度流程：A 正常 EOS，B 被 output processor 的 stop token 结束，C 在 first token 后因 `max_tokens=1` 结束，D 进入 running 后被客户端 abort。demo 同时输出 scheduler metadata 和 cleanup 结果。

```python
from dataclasses import dataclass, field
from math import ceil


@dataclass
class RequestSpec:
    request_id: str
    arrival: int
    prompt_tokens: int
    planned_tokens: list
    max_tokens: int
    stop_token: str = None
    abort_at: int = None


@dataclass
class RequestState:
    spec: RequestSpec
    status: str = "NEW"
    input_ids: list = field(default_factory=list)
    output_tokens: list = field(default_factory=list)
    block_table: list = field(default_factory=list)
    arrival_step: int = None
    prefill_step: int = None
    first_token_step: int = None
    finish_step: int = None
    finish_reason: str = None
    transitions: list = field(default_factory=list)

    def transition(self, status, step):
        self.status = status
        self.transitions.append(f"{status}@{step}")


class ToyKVBlockManager:
    def __init__(self, total_blocks, block_size):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_blocks = list(range(total_blocks))
        self.lengths = {}
        self.max_used = 0

    def used_count(self):
        return self.total_blocks - len(self.free_blocks)

    def remember_peak(self):
        self.max_used = max(self.max_used, self.used_count())

    def required_blocks(self, tokens):
        return ceil(tokens / self.block_size)

    def can_allocate(self, tokens):
        return self.required_blocks(tokens) <= len(self.free_blocks)

    def allocate_prompt(self, state):
        need = self.required_blocks(len(state.input_ids))
        if need > len(self.free_blocks):
            return False
        state.block_table = [self.free_blocks.pop(0) for _ in range(need)]
        self.lengths[state.spec.request_id] = len(state.input_ids)
        self.remember_peak()
        return True

    def can_append(self, state):
        length = self.lengths[state.spec.request_id]
        return length % self.block_size != 0 or bool(self.free_blocks)

    def append_slot(self, state):
        request_id = state.spec.request_id
        length = self.lengths[request_id]
        if length % self.block_size == 0:
            if not self.free_blocks:
                return False
            state.block_table.append(self.free_blocks.pop(0))
        self.lengths[request_id] += 1
        self.remember_peak()
        return True

    def release(self, state):
        self.free_blocks.extend(state.block_table)
        state.block_table = []
        self.lengths.pop(state.spec.request_id, None)

    def slot_mapping(self, state, positions):
        mapping = []
        for position in positions:
            logical_block = position // self.block_size
            offset = position % self.block_size
            physical = state.block_table[logical_block] if logical_block < len(state.block_table) else None
            mapping.append({"position": position, "block": physical, "offset": offset})
        return mapping


class ToyRequestFlowEngine:
    def __init__(self, specs, max_tokens, max_seqs, block_manager):
        self.future_specs = sorted(specs, key=lambda spec: (spec.arrival, spec.request_id))
        self.states = {}
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.block_manager = block_manager
        self.time = 0
        self.waiting = []
        self.running = []
        self.finished = []
        self.aborted = []
        self.trace = []
        self.metadata_rows = []

    def input_processor(self, spec):
        state = RequestState(spec=spec)
        state.arrival_step = self.time
        state.input_ids = [1000 + idx for idx in range(spec.prompt_tokens)]
        state.transition("WAITING", self.time)
        return state

    def receive_arrivals(self):
        arrived = []
        while self.future_specs and self.future_specs[0].arrival <= self.time:
            spec = self.future_specs.pop(0)
            state = self.input_processor(spec)
            self.states[spec.request_id] = state
            self.waiting.append(state)
            arrived.append(spec.request_id)
        return arrived

    def abort_ready(self):
        aborted = []
        for state in list(self.waiting) + list(self.running):
            spec = state.spec
            if spec.abort_at is not None and spec.abort_at <= self.time:
                if state in self.waiting:
                    self.waiting.remove(state)
                if state in self.running:
                    self.running.remove(state)
                self.block_manager.release(state)
                state.finish_reason = "client_abort"
                state.finish_step = self.time
                state.transition("ABORTED", self.time)
                self.aborted.append(state)
                aborted.append(spec.request_id)
        return aborted

    def build_metadata(self, state, phase):
        if phase == "prefill":
            input_ids = list(state.input_ids)
            positions = list(range(len(input_ids)))
        else:
            input_ids = [state.output_tokens[-1]]
            positions = [len(state.input_ids) + len(state.output_tokens) - 1]
        row = {
            "request_id": state.spec.request_id,
            "phase": phase,
            "input_ids": input_ids,
            "positions": positions,
            "block_table": list(state.block_table),
            "slot_mapping": self.block_manager.slot_mapping(state, positions),
        }
        self.metadata_rows.append(row)
        return row

    def next_model_token(self, state):
        index = len(state.output_tokens)
        return state.spec.planned_tokens[index]

    def finish(self, state, reason):
        if state in self.running:
            self.running.remove(state)
        state.finish_reason = reason
        state.finish_step = self.time
        state.transition("FINISHED", self.time)
        self.block_manager.release(state)
        self.finished.append(state)

    def output_processor(self, state, token, phase):
        state.output_tokens.append(token)
        if state.first_token_step is None:
            state.first_token_step = self.time
        if token == "<eos>":
            self.finish(state, "eos")
        elif state.spec.stop_token is not None and token == state.spec.stop_token:
            self.finish(state, "stop")
        elif len(state.output_tokens) >= state.spec.max_tokens:
            self.finish(state, "max_tokens")
        elif len(state.output_tokens) >= len(state.spec.planned_tokens):
            self.finish(state, "eos")
        elif phase == "prefill":
            state.transition("RUNNING", self.time)
            self.running.append(state)

    def schedule_step(self):
        arrived = self.receive_arrivals()
        aborted = self.abort_ready()
        used_tokens = 0
        scheduled_decode = []
        scheduled_prefill = []
        metadata = []
        deferred = []

        for state in list(self.running):
            if used_tokens + 1 > self.max_tokens:
                deferred.append((state.spec.request_id, "token_budget_decode"))
                continue
            if not self.block_manager.can_append(state):
                deferred.append((state.spec.request_id, "kv_decode"))
                continue
            self.block_manager.append_slot(state)
            state.transition("DECODING", self.time)
            scheduled_decode.append(state)
            metadata.append(self.build_metadata(state, "decode"))
            used_tokens += 1

        while self.waiting:
            state = self.waiting[0]
            if len(scheduled_decode) + len(scheduled_prefill) >= self.max_seqs:
                deferred.append((state.spec.request_id, "sequence_budget"))
                break
            if used_tokens + len(state.input_ids) > self.max_tokens:
                deferred.append((state.spec.request_id, "token_budget_prefill"))
                break
            if not self.block_manager.can_allocate(len(state.input_ids)):
                deferred.append((state.spec.request_id, "kv_prefill"))
                break
            self.waiting.pop(0)
            self.block_manager.allocate_prompt(state)
            state.prefill_step = self.time
            state.transition("PREFILL", self.time)
            scheduled_prefill.append(state)
            metadata.append(self.build_metadata(state, "prefill"))
            used_tokens += len(state.input_ids)

        for state in scheduled_decode:
            self.output_processor(state, self.next_model_token(state), "decode")

        for state in scheduled_prefill:
            self.output_processor(state, self.next_model_token(state), "prefill")

        self.trace.append(
            {
                "step": self.time,
                "arrived": arrived,
                "aborted": aborted,
                "decode": [state.spec.request_id for state in scheduled_decode],
                "prefill": [state.spec.request_id for state in scheduled_prefill],
                "metadata_rows": len(metadata),
                "used_tokens": used_tokens,
                "waiting": [state.spec.request_id for state in self.waiting],
                "running": [state.spec.request_id for state in self.running],
                "free_blocks": len(self.block_manager.free_blocks),
                "deferred": deferred,
            }
        )
        self.time += 1

    def run(self):
        while self.future_specs or self.waiting or self.running:
            self.schedule_step()
            if self.time > 20:
                raise RuntimeError("request flow did not converge")

    def report(self):
        ordered_states = [self.states[key] for key in sorted(self.states)]
        queue_wait = {
            state.spec.request_id: state.prefill_step - state.arrival_step
            for state in ordered_states
            if state.prefill_step is not None
        }
        ttft = {
            state.spec.request_id: state.first_token_step - state.arrival_step
            for state in ordered_states
            if state.first_token_step is not None
        }
        e2e = {
            state.spec.request_id: state.finish_step - state.arrival_step
            for state in ordered_states
            if state.finish_step is not None
        }
        metadata_complete = all(
            row["input_ids"]
            and row["positions"]
            and row["block_table"]
            and all(slot["block"] is not None for slot in row["slot_mapping"])
            and row["phase"] in {"prefill", "decode"}
            for row in self.metadata_rows
        )
        finish_reasons = {state.spec.request_id: state.finish_reason for state in ordered_states}
        gates = {
            "input_processed": all(state.input_ids and state.arrival_step is not None for state in ordered_states),
            "state_machine_terminal": all(state.finish_reason is not None for state in ordered_states),
            "scheduler_budget_respected": all(
                item["used_tokens"] <= self.max_tokens and item["metadata_rows"] <= self.max_seqs
                for item in self.trace
            ),
            "execution_metadata_complete": metadata_complete,
            "output_processor_finished": finish_reasons == {
                "A": "eos",
                "B": "stop",
                "C": "max_tokens",
                "D": "client_abort",
            },
            "abort_cleanup": finish_reasons["D"] == "client_abort" and not self.states["D"].block_table,
            "metrics_ready": bool(queue_wait) and bool(ttft) and bool(e2e),
        }
        gates["request_flow_gate"] = all(gates.values())
        summary = {
            "finished_order": [state.spec.request_id for state in sorted(self.finished, key=lambda s: (s.finish_step, s.spec.request_id))],
            "aborted": [state.spec.request_id for state in self.aborted],
            "finish_reasons": finish_reasons,
            "queue_wait_steps": queue_wait,
            "ttft_steps": ttft,
            "e2e_steps": e2e,
            "metadata_rows": len(self.metadata_rows),
            "max_kv_blocks_used": self.block_manager.max_used,
            "kv_blocks_after_cleanup": self.block_manager.used_count(),
            "transitions": {state.spec.request_id: state.transitions for state in ordered_states},
            "trace_tail": self.trace[-4:],
        }
        return summary, gates


specs = [
    RequestSpec("A", arrival=0, prompt_tokens=3, planned_tokens=["a", "b", "<eos>"], max_tokens=3),
    RequestSpec("B", arrival=0, prompt_tokens=5, planned_tokens=["x", "STOP", "ignored"], max_tokens=3, stop_token="STOP"),
    RequestSpec("C", arrival=1, prompt_tokens=2, planned_tokens=["c"], max_tokens=1),
    RequestSpec("D", arrival=2, prompt_tokens=3, planned_tokens=["d", "e"], max_tokens=2, abort_at=3),
]

engine = ToyRequestFlowEngine(
    specs=specs,
    max_tokens=8,
    max_seqs=3,
    block_manager=ToyKVBlockManager(total_blocks=5, block_size=4),
)
engine.run()
summary, gates = engine.report()

print("request_flow_summary=", summary)
print("request_flow_gates=", gates)
```

这个 demo 的输出应当能看出：

1. A、B、C、D 都经过 `WAITING`，其中 A/B/D 进入 `RUNNING`，C 在 prefill 后立即结束。
2. B 的终止来自 output processor 的 `stop`，不是 scheduler 直接决定。
3. D 进入 running 后被 abort，并且 `kv_blocks_after_cleanup=0`。
4. `metadata_rows` 覆盖 prefill 和 decode，说明 scheduler output 已经转换成 model runner 可执行的 metadata。
5. `request_flow_gate=True`，说明输入处理、状态机、scheduler、metadata、output、cleanup 和 metrics 都能闭环。

## 20.22 小练习

1. 画出一个 vLLM-like 请求生命周期图，至少包含 API server、input processor、engine core、scheduler、block manager、worker、output processor。
2. 写一个简化 `RequestState`，包含状态、input ids、output ids、block table、arrival time、first token time 和 finish reason。
3. 写一个伪代码函数，模拟 `add_request -> waiting queue -> schedule -> prefill -> running -> decode -> finished -> free blocks`。
4. 解释为什么 stop string 检查可能不能只看最后一个 token id。
5. 列出请求 abort 时必须清理的 5 类资源。

## 20.23 本章总结

vLLM 请求调度流程的核心不是某一个类名，而是一条稳定的系统链路：入口层接收和预处理请求，engine core 维护请求状态和调度循环，scheduler 决定每轮执行计划，KV cache manager 保障显存资源，worker/model runner 执行模型，output processor 处理输出和终止条件。

理解这条链路后，continuous batching、PagedAttention 和 block manager 就能连起来：scheduler 每轮选择请求，block manager 给这些请求安排 KV blocks，model runner 根据 block table 执行 attention，output processor 决定请求是否继续进入下一轮。

下一章会继续讲 vLLM memory management，重点从更系统的角度拆解 KV Cache、GPU memory、CPU memory、prefix cache、swap/offload 和显存预算之间的关系。
