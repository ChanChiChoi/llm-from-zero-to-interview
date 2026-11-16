# 第 6 章（第三篇）：Serving runtime——多个请求怎样共享显存

测量若显示排队、KV 容量或请求长度差异正在限制服务，就要看 runtime 如何安排内存和计算。静态 batching 等整批请求一起完成，长短请求混在一起时容易互相拖慢；按需分页可以减少预留浪费，却会引入 block table、引用计数和调度决策。本篇用 vLLM 的术语解释这些机制，同时把公开 API 兼容与真实语义兼容分开。

我们会从一个序列的逻辑 token 位置追到物理 KV block，再看请求怎样进入和离开活动 batch。读完后，读者应能根据负载选择要记录的吞吐、尾延迟、显存和协议结果，而不是把“开启 PagedAttention”当作优化结论。

## 6.3 vLLM：PagedAttention、连续批处理和协议边界

### 6.3.1 KV Cache 为什么可以分块存放

最早的推理服务可以为每条请求预留一块连续 KV Cache。问题是，服务不知道请求最终会生成多少 token。若按最大长度预留，短请求浪费大量空间；若按当前长度不断搬家，扩容和复制成本又很高。多个请求交错结束时，还会产生空洞和碎片。

PagedAttention 的核心思路是把逻辑上的连续序列切成固定大小的 block，再把这些 block 映射到不必连续的物理内存。它和操作系统的分页有相似之处，但具体 kernel、block table 和调度实现属于 serving runtime 的设计。

```text
逻辑序列：token 0 1 2 3 | 4 5 6 7 | 8 9 ...
                 block 0 |  block 1 | block 2

物理显存：      page 12 | page  3 | page 27

block table：
逻辑 block 0 → 物理 page 12
逻辑 block 1 → 物理 page  3
逻辑 block 2 → 物理 page 27
```

这样可以按需分配，序列结束时释放自己的 block；共享相同前缀的请求还可以通过引用计数复用前缀 block。

### 6.3.2 KV block 容量估算

设 token block 大小为 $b$，每个物理 block 对一层的 K/V 元素数为：

```math
E_{\mathrm{block,layer}}
=
2\cdot b\cdot n_{\mathrm{kv}}\cdot d_h
```

所有 $L$ 层、每元素 $s$ 字节时：

```math
B_{\mathrm{block}}
=
2\cdot L\cdot b\cdot n_{\mathrm{kv}}\cdot d_h\cdot s
```

如果为 KV Cache 预留的显存池大小为 $M_{\mathrm{pool}}$，忽略元数据和碎片时，物理 block 数近似为：

```math
N_{\mathrm{blocks}}
=
\left\lfloor
\frac{M_{\mathrm{pool}}}{B_{\mathrm{block}}}
\right\rfloor
```

这些容量式默认 $L,b,n_{\mathrm{kv}},d_h$ 是正整数，$s$ 是正的数据类型字节数，$M_{\mathrm{pool}}>0$ 且与上述单位一致。否则 $B_{\mathrm{block}}$ 可能为零或单位错位，分块数也就没有可解释的物理意义。

可容纳的逻辑 token 总量则近似为：

```math
T_{\mathrm{capacity}}
\approx
b\cdot N_{\mathrm{blocks}}
```

举例，取 $L=32$、$b=16$、$n_{\mathrm{kv}}=8$、$d_h=128$、$s=2$：

```math
B_{\mathrm{block}}
=
2\cdot32\cdot16\cdot8\cdot128\cdot2
=2{,}097{,}152\ \mathrm{bytes}
\approx2\ \mathrm{MiB}
```

若 KV 池恰好有 1 GiB，粗略可放约 512 个 block，也就是约 8192 个 token。真实服务还要扣除权重、CUDA workspace、activation、通信缓冲区、block table、量化 scale 和 allocator 预留，因此这只是容量估算，不是启动参数的保证值。

### 6.3.3 分页为什么能减少浪费，但不能消灭开销

最后一个 block 通常没有填满。若一条序列长度为 $T$，block size 为 $b$，内部碎片上界小于 $b$ 个 token：

```math
N_{\mathrm{logical}}
=
\left\lceil\frac{T}{b}\right\rceil,
\qquad
0\leq
bN_{\mathrm{logical}}-T
<b
```

block 越小，尾部浪费越少，但 block table 更大、地址计算更频繁、kernel 访问粒度更碎；block 越大，元数据和寻址更简单，却可能浪费更多最后一页，还可能降低共享前缀的精细程度。

前缀共享也有代价：

1. 需要为每个物理 block 维护引用计数或等价生命周期信息；
2. 不同请求只能共享完全一致且只读的前缀；
3. 一旦某个序列要修改共享 block，必须 copy-on-write；
4. cache key 必须包括 token ids、模型 revision、adapter 和相关配置；
5. 共享内容的隐私和租户隔离需要由上层策略保证。

例如 block size 取 16、序列已有 1000 个 token，需要 $\lceil1000/16\rceil=63$ 个 block，分配容量为 1008 个位置，最后一块空 8 个位置。这个 8/1008 是单条序列尾块的内部浪费，不是整张 GPU 的碎片率。按最大上下文 4096 预留连续区域会留出更多空位；分页按需分配则用 block table 和调度管理换取更细的分配粒度。

### 6.3.4 Continuous Batching：按 token 调度，而不是等整批结束

静态 batching 需要等一批请求都准备好，再一起运行；其中一个长请求可能拖住短请求。Continuous batching 把调度单位变成正在运行的序列和下一 token step，完成的序列可以退出，新序列可以补入。

```text
时间 t0：A、B 运行
时间 t1：A 完成，B 继续，C 加入
时间 t2：B、C 继续，D 加入
时间 t3：C 完成，D、E 继续
```

调度器需要同时回答：

1. 哪些序列处于 waiting、running、finished、cancelled；
2. 本轮给每条序列分配多少 decode token；
3. 长 prompt 是否切成 chunk；
4. 新请求是否会挤压已经运行的请求；
5. 显存 block 是否足够；
6. 达到 max batch tokens、max sequences 或超时后如何选择；
7. 取消请求时如何释放 block 和回收流式连接。

因此 continuous batching 不是一个只改启动参数的优化。它把运行时变成了一个资源调度器；调度策略会影响平均吞吐、短请求公平性、长请求等待和 p99。

### 6.3.5 一个无依赖的 block allocator demo

下面的代码模拟物理 block 分配、释放和前缀共享引用计数。它没有实现 attention kernel，但可以帮助检查生命周期。

> 代码状态：静态审阅，未运行（2026-10-29）
```python
class BlockPool:
    def __init__(self, size):
        if type(size) is not int or size <= 0:
            raise ValueError("size must be a positive integer")
        self.all_blocks = set(range(size))
        self.free = list(range(size))
        self.refs = {}

    def _validate(self, blocks, require_allocated):
        blocks = list(blocks)
        if not blocks:
            raise ValueError("blocks must not be empty")
        if len(set(blocks)) != len(blocks):
            raise ValueError("a block may appear only once per operation")
        for block in blocks:
            if type(block) is not int or block not in self.all_blocks:
                raise ValueError("unknown physical block")
            if require_allocated and block not in self.refs:
                raise ValueError("block is not currently allocated")
        return blocks

    def allocate(self, count):
        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        if count > len(self.free):
            raise RuntimeError("not enough physical blocks")
        blocks = [self.free.pop() for _ in range(count)]
        for block in blocks:
            self.refs[block] = 1
        return blocks

    def share(self, blocks):
        blocks = self._validate(blocks, require_allocated=True)
        for block in blocks:
            self.refs[block] += 1

    def release(self, blocks):
        blocks = self._validate(blocks, require_allocated=True)
        for block in blocks:
            self.refs[block] -= 1
            if self.refs[block] == 0:
                del self.refs[block]
                self.free.append(block)


pool = BlockPool(size=4)
prefix = pool.allocate(2)
pool.share(prefix)
suffix = pool.allocate(1)
pool.release(suffix)
pool.release(prefix)
assert len(pool.free) == 2
pool.release(prefix)
assert len(pool.free) == 4
print("free_blocks=", len(pool.free))
```
这里最后两次 release 代表两个引用分别结束。真实实现必须处理异常、取消、重复释放、跨请求隔离和并发原子性，不能把这个列表实现直接用于多线程服务。

### 6.3.6 vLLM 参数如何映射到资源问题

不同版本的 vLLM 参数会变化，常见配置意图可以按资源问题理解：

```text
model：
  权重和 tokenizer 的来源；revision、trust_remote_code 等会影响可复现性。

dtype：
  权重/计算的默认数据类型；不等于 KV Cache 一定使用同一种 dtype。

max-model-len：
  允许的最大序列长度；过大可能提高预留和调度压力。

gpu-memory-utilization：
  runtime 用于估计可用显存池的比例；不是把所有显存安全地变成 KV。

tensor-parallel-size：
  把模型计算与状态分布到多张 GPU；通信和拓扑会改变收益。

enable-prefix-caching：
  尝试复用相同 token 前缀；必须验证 key、租户和版本边界。

kv-cache-dtype：
  指定 KV Cache 的表示；要确认硬件 kernel、scale 和质量影响。
```

一个典型的启动命令可以写成：

```text
vllm serve MODEL_ID
  --dtype auto
  --max-model-len 8192
  --gpu-memory-utilization 0.90
  --enable-prefix-caching
```

这里的命令只是配置形状示例，`MODEL_ID` 是要替换为实际模型仓库名的占位符。某个版本是否支持某个参数、参数名称是否变化、模型是否需要 custom code，都应在目标环境执行 help、查看版本文档并记录 revision。不能根据另一台机器的启动命令推断本地 kernel、显存和模型格式一定兼容。

### 6.3.7 OpenAI 兼容服务的边界

OpenAI 兼容通常表示 HTTP 路径、请求字段和响应外形接近某种 API。客户端能发出请求，不代表所有语义完全一致。至少要核对：

1. chat template 如何把 messages 转成 token；
2. system、tool、developer、assistant 等角色是否被保留；
3. stop、max_tokens、max_completion_tokens、temperature、top_p 的处理；
4. stream chunk 的 data 格式、结束标志和 usage 字段；
5. tool calling 的 JSON schema、并行调用和错误返回；
6. logprobs、seed、reasoning 字段是否真的支持；
7. prompt token、completion token 和缓存 token 的统计口径；
8. 客户端取消后 GPU 计算和 KV block 是否及时释放。

服务可以用一个简单的 HTTP 请求做协议冒烟：

```text
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "MODEL_ID",
  "messages": [
    {"role": "user", "content": "用一句话解释 KV Cache。"}
  ],
  "max_tokens": 32,
  "temperature": 0
}
```

请求里的 `MODEL_ID` 要替换为服务端公开的模型名。冒烟测试只证明路径能够返回某种响应。生产验证还要覆盖流式断开、超时、空 messages、超长输入、工具 schema 错误、并发取消、错误码和 token usage。

### 6.3.8 vLLM 与其他 runtime 的比较方法

PagedAttention、continuous batching、prefix cache 和 OpenAI adapter 并不是 vLLM 独占的抽象。SGLang、TensorRT-LLM、TGI、llama.cpp 等 runtime 可能采用不同的 cache layout、调度器、编译路径和量化 kernel。

比较 runtime 时，应该固定：

```text
同一模型 revision 和 tokenizer
同一 prompt/输出分布
同一硬件、驱动和并行拓扑
同一 dtype、量化格式和最大长度
同一请求到达率与并发模型
同一停止条件、采样参数和流式统计
同一正确性回归集
```

如果只比较一个“最高 tokens/s”，可能测到的是不同 batch、不同输出长度或不同的 warmup 状态，而不是 runtime 的普遍优势。

### 6.3.9 资料与证据边界

- vLLM 官方文档：<https://docs.vllm.ai/>
- vLLM PagedAttention 论文：<https://arxiv.org/abs/2309.06180>
- vLLM EngineArgs（v0.32.0）：<https://docs.vllm.ai/en/v0.32.0/configuration/engine_args/>
- vLLM OpenAI 兼容服务（v0.32.0）：<https://docs.vllm.ai/en/v0.32.0/serving/online_serving/>
- vLLM Paged Attention 设计页（v0.32.0 历史说明，不代表当前实现）：<https://docs.vllm.ai/en/v0.32.0/design/paged_attention/>
- SGLang 官方文档：<https://docs.sglang.ai/>
- TensorRT-LLM 官方文档：<https://nvidia.github.io/TensorRT-LLM/>
- Hugging Face Text Generation Inference：<https://huggingface.co/docs/text-generation-inference/index>

PagedAttention 论文支持分页 KV 的机制动机和实验背景；官方 runtime 文档支持参数和服务接口；具体吞吐、并发和显存容量属于目标版本与目标硬件上的实测结果。

需要区分论文给出的分页机制与当前 runtime 实现：vLLM v0.32.0 的 Paged Attention 设计页已说明它是基于论文的历史文档，不再描述当前代码。分页能减少按最大长度预留造成的浪费，但活跃请求、权重、workspace 和运行时仍受总显存限制；continuous batching 能补入已完成请求留下的计算空隙，却需要同时观察短请求尾延迟与长请求饥饿。下一篇转向另一类显存优化：改变权重或 KV 数值的表示。
