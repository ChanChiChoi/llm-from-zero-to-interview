# 第七章：大模型部署：从 checkpoint 到可观测服务

模型权重放进 GPU，能够对一条 prompt 产生文本，只能说明推理路径在某个最小条件下可用。真正的部署要回答更多问题：输入协议是否和训练一致，长短请求能否共享设备，首 token 和持续输出的延迟是否满足目标，KV Cache 是否会在高并发下耗尽，量化是否改变了格式和工具行为，RAG 与工具调用的失败能否归因，用户取消后资源是否释放，模型升级是否可以灰度和回滚，以及每一个成功任务究竟花费多少计算资源。

本章把部署看作一条有状态的服务链：

~~~text
checkpoint 与 tokenizer
    -> 请求契约与输入规范化
    -> 路由、排队与资源预算
    -> prefill 建立 KV Cache
    -> decode 逐 token 生成
    -> 流式协议、工具/RAG 与审计
    -> 质量、延迟、吞吐、成本与安全评估
    -> 灰度、回滚、恢复与容量迭代
~~~

初学者可以先理解“输入怎样变成一次生成、为什么生成越长越贵、为什么一个请求会占住一块 KV Cache”。有经验的工程师还要把模型、引擎、调度器、网关和业务任务分开记账：一个引擎吞吐更高，不等于任务成功率更高；一次响应更快，不等于 P99 更好；权重显存下降，不等于 KV Cache 或端到端成本下降。

第 6 章讨论训练系统如何产生可恢复 checkpoint，本章讨论这个产物怎样在真实请求下运行。第 8 章将处理 alignment 与 safety 的训练和评估问题，因此本章只覆盖部署运行时需要的安全边界，不把安全模型压缩成一个分类器分数。

## 7.1 先写服务契约，再选择推理引擎

### 7.1.1 部署的交付物不只是一个 endpoint

一个可交付的大模型服务至少包含四种契约：

1. 输入契约：模型版本、tokenizer、chat template、上下文长度、特殊 token 和输入大小限制。
2. 运行契约：允许的并发、优先级、超时、取消、流式事件、重试和资源预算。
3. 输出契约：文本、结构化 JSON、工具调用、引用、停止原因和错误语义。
4. 证据契约：质量、延迟、吞吐、成本、安全、审计和版本变更能够被复核。

如果只交付“模型服务地址”，调用方不知道 system prompt 是否会被拼接两次，不知道客户端断开是否会取消 GPU 计算，也不知道模型升级后输出格式变化是否能追踪。部署工程的边界因此比 HTTP handler 更大。

### 7.1.2 离线、内部和生产是三个不同目标

离线评测更关心固定样本的正确性、复现和吞吐；内部 demo 更关心接入速度和调试能力；生产 API 还要满足尾延迟、多租户、故障恢复、权限、数据删除和成本约束；端侧部署则增加模型大小、功耗、内存和断网运行限制。

同一个 checkpoint 可能适合离线批处理，却不适合交互式流式服务。部署目标必须先写清：请求量和 token 分布、TTFT/TPOT 或 E2E SLA、最大上下文、质量基线、允许的量化误差、故障恢复时间和每次成功任务的预算。

### 7.1.3 请求从网关到 GPU 的边界

一次生成请求通常经过：

~~~text
鉴权与配额
    -> 请求解析和大小限制
    -> tokenizer 与模板渲染
    -> 模型/租户/优先级路由
    -> prefill/decode 调度
    -> 流式事件和取消处理
    -> 结果校验、审计与计费
~~~

每一步都可能失败。网关接受请求不等于 GPU 已经开始计算；模型返回 token 不等于工具动作已经成功；流式连接关闭不等于后台生成停止。状态必须通过 request id、generation id 和事件序号关联，才能在日志中还原一次请求的完整路径。

### 7.1.4 请求状态机与流式事件

一次生成请求最好被表示为有边界的状态机，而不是一个从开始持续到结束的函数调用。一个最小状态集合可以是：`CREATED`、`QUEUED`、`PREFILLING`、`DECODING`、`WAITING_TOOL`、`COMPLETED`、`CANCELLED`、`FAILED` 和 `UNKNOWN`。前几个状态描述系统正在做什么，后几个状态描述系统已经知道或暂时不知道什么结果。

状态转移应当单向且可审计。例如，`QUEUED` 可以转到 `PREFILLING`、`CANCELLED` 或 `FAILED`；`DECODING` 可以转到 `WAITING_TOOL`、`COMPLETED`、`CANCELLED` 或 `FAILED`；工具调用超时后，如果执行器无法确认动作是否已经发生，应进入 `UNKNOWN`，而不是直接写成 `FAILED`。`UNKNOWN` 不是失败的同义词，它表示系统缺少足够证据，后续动作应当是查询、人工确认或补偿。

流式响应还需要自己的事件序号。一个事件至少可以包含：

~~~json
{
  "request_id": "r-1842",
  "generation_id": "g-7",
  "sequence": 18,
  "type": "delta",
  "text": "合同",
  "created_at": "2026-08-12T10:00:03Z"
}
~~~

`request_id` 关联一次用户请求，`generation_id` 区分同一请求的重试或重新生成，`sequence` 用来发现重复、丢失和乱序。客户端重连时，服务端可以从最近确认的序号之后重放事件；如果服务端没有保存事件或生成状态，就不能把“重新发送 prompt”当作可靠恢复，因为它可能重复工具调用、改变随机采样结果或再次消耗大量 GPU。

流式协议应区分至少四类事件：增量内容、结构化工具调用、用量/状态更新和终止事件。终止事件要携带停止原因，例如自然结束、长度上限、用户取消、下游错误或策略拒绝。把错误文本拼在普通回答末尾，会让客户端误把部分结果当作完整结果；把连接断开当作唯一错误信号，则无法区分用户取消、网络故障和 worker 崩溃。

服务端还要决定“已发送”代表什么。一个 token 写入 worker 缓冲区、写入网关 socket、被客户端收到和被客户端持久化，是四个不同的时间点。计费、重试和审计应选定明确边界，并在事件中保持一致。对于非幂等工具，只有拿到执行器的业务确认，才能把动作标记为成功；对于文本生成，已经发送的增量通常无法撤回，因此后续错误必须以显式终止事件表达。

### 7.1.5 准入前的请求规范化

资源判断必须发生在昂贵计算之前。网关先解析模型、租户、优先级、输入消息和最大输出，再由 tokenizer 得到输入 token 数，检查上下文上限、工具 schema 大小、预计 KV 占用和租户配额。请求通过这些检查，只表示它可以进入候选队列，不表示系统必须立即为它分配 GPU。

输入 token 数不能用字符数简单替代。中文、英文、代码、JSON 和图片占位符的 token 密度不同；同样的字符上限可能造成完全不同的显存和延迟。对于多模态请求，还需要把图像 patch、音频帧或视频 token 转换成模型实际使用的序列预算，并将预处理显存纳入总账本。

规范化阶段还应固定解码参数的默认值和上限。温度、top-p、最大输出、工具数量和重试次数如果由不同服务层分别设置，就会出现日志里“请求参数”和 GPU 实际参数不一致的情况。最终生效配置应生成一个可比较的摘要，和模型版本、模板版本一起写入 trace。

## 7.2 从 checkpoint 到运行时状态

### 7.2.1 权重、配置和 tokenizer 必须成套

服务启动时至少要加载：模型参数、结构配置、tokenizer 文件、special token 映射、位置编码配置、量化配置和 generation 默认值。若使用 LoRA 或 adapter，还要记录基础权重、adapter 版本和合并/动态加载方式。

模型看到的是 token id 序列，不是抽象的 messages 对象。tokenizer 决定字符串如何切分，chat template 决定 system、user、assistant、tool call 和结束 token 如何排列。训练模板与推理模板不一致，会造成分布偏移：轻则角色边界混乱，重则输出永不停止、JSON 失配或工具调用参数不完整。

### 7.2.2 输入规范化要可追踪

不要在多个服务层分别拼接 system prompt、历史消息和工具定义。应由一个版本化的模板组件生成最终序列，并记录：

- 原始消息数量和角色；
- 模板版本；
- 最终 token 数和截断位置；
- special token 与 stop token；
- 工具 schema 的哈希或版本；
- 是否命中 prefix cache。

生产日志不必保存所有敏感原文，但需要保存足以重放协议和 token 统计的脱敏信息。否则线上“模型质量下降”可能只是模板被网关和 worker 各渲染了一次。

### 7.2.3 启动检查与运行检查不同

启动检查确认权重 shape、词表大小、dtype、设备、量化 kernel 和配置可以组合；运行检查确认每条请求的上下文预算、stop 条件、KV block、租户权限和取消状态正确。服务能启动只能证明静态组合成立，不能证明长上下文、批量生成和流式取消语义正确。

## 7.3 Prefill 与 Decode：一次生成的两个时钟

### 7.3.1 Prefill 建立上下文状态

Prefill 处理完整输入 prompt。对长度为 `T_in` 的输入，模型可以在一次或少数几次前向中并行计算各位置表示，并为每层生成历史 K/V。输入越长，矩阵计算和 attention 相关访问越重；这一阶段通常决定用户等待第一个 token 的时间。

### 7.3.2 Decode 逐 token 延伸状态

Decode 每次只把新生成的 token 输入模型，读取历史 KV Cache，计算下一个 token。它不能像 prefill 那样把未来输出并行展开，因为下一个 token 依赖前一个采样结果。因此输出长度直接增加 GPU 计算、KV 读取和流式传输时间。

| 阶段 | 主要工作 | 典型指标 | 关键资源 |
| --- | --- | --- | --- |
| Prefill | 处理输入并建立 KV | TTFT | 计算、长 prompt、排队、模板和检索 |
| Decode | 逐 token 读取 KV 并生成 | TPOT、output tokens/sec | 权重带宽、KV 带宽、batch、调度 |

“输入 token 更长”和“输出 token 更长”不是同一种负载。输入通常更容易批量并行，输出需要串行迭代；优化 TTFT 与优化 TPOT 可能互相影响。

### 7.3.3 延迟指标要先定义时间边界

设请求进入网关的时间为 `t_arrive`，开始进入 GPU prefill 的时间为 `t_prefill_start`，首个有效输出事件发出的时间为 `t_first`，最后一个输出事件完成的时间为 `t_done`，则可以定义：

~~~math
\mathrm{TTFT}
=
t_{\mathrm{first}}-t_{\mathrm{arrive}}
~~~

~~~math
\mathrm{E2E}
=
t_{\mathrm{done}}-t_{\mathrm{arrive}}
~~~

若只从 GPU kernel 开始计时，会漏掉排队、tokenizer、RAG、工具和网络时间。若首个事件只是空的心跳，也不能把它当作首 token。指标名称相同，时间边界不同，数值就不可直接比较。

如果需要单独观察排队和前处理，可以再记录：

~~~math
t_{\mathrm{queue+preprocess}}
=
t_{\mathrm{prefill\_start}}-t_{\mathrm{arrive}}
~~~

这里的 `t_prefill_start` 只是一个时间戳，不应和完整 TTFT 混用。它能帮助判断首 token 变慢究竟发生在网关、检索、排队，还是 GPU 计算本身。

### 7.3.4 TPOT 与流式稳定性

对第 j 个输出 token，定义相邻 token 事件的时间差 `delta_j`。平均 TPOT 可以写成：

~~~math
\mathrm{TPOT}_{\mathrm{mean}}
=
\frac{1}{n-1}
\sum_{j=2}^{n}
\left(t_j-t_{j-1}\right)
~~~

但平均值会掩盖长时间停顿。交互服务还应报告 token 间隔的 P95/P99、首 token 后的最大间隔、客户端重连和流式事件丢失。一个平均 20 ms/token 的服务，如果每隔 30 个 token 停顿 2 秒，用户仍会觉得卡顿。

## 7.4 KV Cache：每个请求都在占用的运行时内存

### 7.4.1 为什么缓存 K 和 V

在 self-attention 中，历史 token 的 Key 和 Value 在后续 decode 步骤仍会被读取。如果每一步都重新计算历史 K/V，生成第 n 个 token 时会重复处理前面 n-1 个 token。有了 KV Cache，历史 K/V 只在它们首次出现时计算并保存，新一步主要计算当前 token 的投影和 attention。

KV Cache 是请求级状态，不是模型参数。它有生命周期：prefill 分配，decode 追加，完成/取消/超时后释放或回收到 block pool。忘记释放会造成显存泄漏；在错误租户间复用则可能造成上下文泄露。

### 7.4.2 KV Cache 容量公式

设模型层数为 `L`，并发请求数为 `B`，每个请求平均缓存长度为 `T`，K/V 头数为 `H_kv`，每个 head 维度为 `d`，每个元素占 `b` 字节。若不考虑分页元数据和对齐，KV Cache 字节数近似为：

~~~math
M_{\mathrm{KV}}
\approx
2LBT H_{\mathrm{kv}}d b
~~~

前面的 2 表示 K 和 V 两份缓存。这个公式解释了几个工程事实：并发、上下文和输出长度都会增加占用；GQA/MQA 减少 `H_kv`；BF16/FP16 使用 2 字节，FP8 或量化缓存可以减少字节数，但需要验证精度和 kernel；实际系统还要加 block 对齐、临时 buffer、共享前缀和 allocator 元数据。

### 7.4.3 一个 KV Cache 数值例子

设 `L=32`、`B=8`、`T=4096`、`H_kv=8`、`d=128`，使用 BF16，即 `b=2` 字节，则：

~~~math
M_{\mathrm{KV}}
\approx
2\times32\times8\times4096\times8\times128\times2
=
4{,}294{,}967{,}296\ \mathrm{bytes}
\approx4.0\ \mathrm{GiB}
~~~

如果改成 MQA，`H_kv=1`，理想化占用约为 0.5 GiB；如果把并发和长度同时扩大四倍，容量会扩大十六倍。这个计算只是所有请求合计的 KV 状态容量估算，不包括模型权重、激活、临时 workspace、页表和其他请求的实际长度差异。

### 7.4.4 MHA、GQA 和 MQA 的部署取舍

MHA 为每个 query head 保存独立 K/V，表达自由度高但 cache 大；MQA 让所有 query head 共享 K/V，cache 和带宽压力最低；GQA 处于中间位置。它们不是部署时可以任意切换的开关：改变 K/V head 数会改变权重 shape 和模型计算，通常需要训练或专门转换配方。

因此要在同一模型族、相同请求分布和相同质量协议下比较：KV 容量、decode 吞吐、长上下文任务、事实性、格式和工具调用不能只看一个数字。

## 7.5 KV Cache 管理：从连续数组到分页块

### 7.5.1 连续预留为什么浪费

请求到达和结束时间不同，生成长度也不同。如果每个请求按最大上下文一次分配连续显存，短请求会留下未使用空间，长请求可能因为找不到足够大的连续区域而无法接入，即使总空闲显存仍然够用。频繁移动和合并也会阻塞调度。

### 7.5.2 PagedAttention 的职责边界

PagedAttention 借鉴虚拟内存的分页思想：逻辑上的 token 序列分成固定大小的 KV blocks，物理显存中的 block 可以不连续，由 block table 映射。请求增长时追加 block，完成时回收 block，调度器可以在不同请求之间重新分配物理块。

它不是新的语言建模算法，也不改变 attention 的数学定义。它解决的是 KV Cache 的分配、碎片和共享管理问题；它不能修复错误的模型权重、提高事实性，也不能绕过总显存上限。

### 7.5.3 Prefix Cache 的共享条件

多个请求如果有完全相同的前缀，可以共享前缀对应的 KV blocks，减少重复 prefill。共享必须以精确的 token 序列、模型版本、adapter、位置语义、模板、租户权限和缓存生命周期为条件。相似字符串不能直接当成相同 prefix；不同权限的文档和 system prompt 也不能因为文本相似就复用。

prefix cache 命中率提高可能降低 TTFT，但会占用显存并增加失效管理。应同时测命中率、节省的 prefill 时间、占用的 block、版本切换后的失效和跨租户隔离。

### 7.5.4 Block pool、引用计数与抢占

Paged KV 的核心不是“把数组切小”，而是维护一组可以分配、共享、回收和失效的物理资源。每个请求有一张逻辑到物理 block 的映射表；block pool 记录空闲块、所属模型版本、引用计数、最后使用时间和是否允许共享。请求结束时，系统不能只删除请求对象，还必须沿着映射表减少每个 block 的引用计数，引用归零后才能回收到空闲池。

前缀共享会让一个物理 block 同时被多个请求引用。此时后续请求如果要在共享前缀之后写入新 token，不能直接覆盖原 block，而要为新内容申请独立 block，这类似写时复制。若引用计数错误，轻则造成显存泄漏，重则一个请求生成的内容会改变另一个请求仍在使用的 cache。

block 大小也有取舍。block 太小，尾部浪费较少，但映射表更大、调度元数据更多、kernel 处理的间接寻址更多；block 太大，映射简单，却可能为一个只增加少量 token 的请求保留更多空闲位置。应在真实长度分布下测内部碎片率、映射元数据、decode 速度和回收延迟，而不是只凭“页越小越省显存”做决定。

当物理 block 不够时，系统通常有三种选择：

1. 延迟接纳请求，保留已有 decode 的稳定性；
2. 把某些请求的 KV 暂存到 CPU 或其他层级，恢复时付出传输和调度成本；
3. 丢弃可重建的 KV，之后从保留的 token 重新 prefill。

换出不是免费的扩容。CPU 传输受 PCIe、NUMA 和内存带宽限制，重算则消耗 GPU prefill；如果一个请求马上又需要输出，频繁换入换出会形成抖动。调度器应根据请求优先级、已完成 token 数、剩余输出、恢复成本和取消概率选择对象，并记录每次抢占造成的额外时间和 token 计算。

一个实用的观察指标是“可用 block 数”和“可用但无法满足请求的 block 数”同时统计。前者低说明总容量紧张，后者高则可能说明碎片、保留策略或共享引用存在问题。只看 GPU 总显存占用，无法区分这几种原因。

### 7.5.5 vLLM、SGLang 与自研引擎的选择

推理引擎通常提供模型加载、kernel、KV 管理、continuous batching、流式输出、量化和分布式支持。vLLM 的 PagedAttention 是其中一类重要实现；SGLang 等系统还可能提供更强的结构化程序、前缀复用或缓存调度能力。具体功能随版本和模型后端变化，不能只根据项目名称推断性能。

选择引擎时，应对目标 checkpoint 和请求分布做 profiling：启动时间、首 token、持续 token、并发、长短请求混合、取消释放、结构化输出、量化质量、GPU 利用率和 P99 都应纳入比较。

## 7.6 Batching 与调度：吞吐和公平性的冲突

### 7.6.1 静态 batch、动态 batch 和 continuous batching

静态 batch 预先固定请求数量和形状，简单但容易被最长请求拖住。普通动态 batch 在短暂等待窗口内把新请求凑成一批，适合一次性推理，却不完全适合自回归生成。

Continuous batching 在每个 decode iteration 更新活跃集合：完成的请求退出，新请求在资源允许时加入，未完成请求继续生成。它提高了设备利用率，却让调度器必须同时管理请求状态、KV blocks、优先级、取消、token budget 和公平性。

### 7.6.2 Token budget 比 request count 更接近资源

两个请求的成本可能相差几十倍。设请求 i 的输入 token 数为 `T_in,i`，计划输出上限为 `T_out,i`，调度权重为 `w_i`，一个粗略的工作量预算可以写成：

~~~math
W_i
\approx
w_i\left(T_{\mathrm{in},i}+\alpha T_{\mathrm{out},i}\right)
~~~

`alpha` 表示输出 token 相对输入 token 的资源权重，实际值依赖 prefill/decode、模型和硬件。这个式子不是精确运行时间模型，却比单纯按请求数限流更接近资源消耗。调度还要区分 TTFT 敏感请求、长任务、批处理和高风险工具任务。

### 7.6.3 Continuous batching 的取舍

把新请求立即加入活跃 batch 可以提高吞吐和降低排队，但可能增加已有请求的 TPOT；保留资源给已运行请求可以降低输出抖动，却使新请求 TTFT 变高。实际策略可按 token budget、deadline、优先级、KV 余量和请求年龄组合，而不是只设置一个最大 batch size。

### 7.6.4 Chunked prefill 的作用

一个超长 prompt 的 prefill 可能长时间占用 GPU，阻塞已有请求的 decode。Chunked prefill 把输入切成多个块，穿插到 decode 调度中，减少长 prefill 对流式输出的冲击。

代价是长请求本身的 TTFT 可能变长，调度和 position/cache 管理更复杂。是否启用取决于产品更重视新请求首 token，还是在高并发下保持已有流的稳定 TPOT。应分别测两类请求，而不是只看平均吞吐。

### 7.6.5 取消、超时与资源回收

用户关闭浏览器、网关超时或上游取消时，取消信号必须穿过 API、队列、scheduler、GPU worker、streamer 和工具执行器。至少要定义：

- 取消是否阻止后续 token 计算；
- 已分配 KV blocks 何时释放；
- 正在执行的 kernel 是否只能延迟停止；
- 已完成的工具副作用是否需要补偿；
- 重试是否会重复计费或重复动作；
- 最终状态如何写入审计日志。

只在客户端停止读取流，不代表服务端停止生成。若取消路径不完整，流量越大，隐藏的 GPU 浪费和 KV 泄漏越严重。

### 7.6.6 资源接纳、抢占与公平

请求进入队列前，可以先估计它在最大输出下的 KV 上限。沿用前文记号，模型层数为 `L`，每层 K/V 头数为 `H_kv`，head 维度为 `d`，每个元素占 `b` 字节。对请求 i，输入 token 数为 `T_in,i`，允许的新 token 上限为 `T_out,max,i`，则忽略分页和共享时：

~~~math
m_{i,\mathrm{reserve}}
\approx
2L H_{\mathrm{kv}}d b
\left(T_{\mathrm{in},i}+T_{\mathrm{out,max},i}\right)
~~~

`m_i,reserve` 是保守预留，不是请求当前已经使用的字节数。用上限预留可以减少 decode 到一半才 OOM 的风险，但会降低并发；只按当前长度接纳可以提高利用率，却需要动态申请、等待或抢占。生产系统通常把硬上限、软上限和排队策略分开：硬上限保护设备，软上限允许短时借用，排队策略决定借用失败后谁先等待。

例如，一张卡经过权重和 runtime workspace 预留后有 24 GiB 可用于 KV。若请求平均最大上下文对应 1 GiB 的理想 KV，但 block 对齐和波动再预留 25%，安全估计约为 1.25 GiB/请求。理论上可以同时保持 19 个这样的请求，但还应为临时激活、通信、长请求尾部和故障恢复保留空间，因此实际并发上限可能低于 19。这个数字只是容量推演，不是任何特定引擎的承诺。

公平性也不能等同于 FIFO。若队列中有一个 64K 输入、4K 输出的请求和许多 200 token 的短请求，严格 FIFO 会让短请求长时间等待；完全优先短请求又可能让长任务饥饿。可以为租户维护加权 token 预算，并为等待时间设置老化项：

~~~math
P_i
=
\frac{T_{\mathrm{served},i}}{w_i}
\;-
\beta\,T_{\mathrm{wait},i}
~~~

其中 `T_served,i` 是已经消耗的 token 工作量，`w_i` 是租户或优先级权重，`T_wait,i` 是等待时间，`beta` 控制等待多久后优先级上升。这个式子只是说明调度信号的构成；真实系统还要加入 deadline、KV 余量、工具风险和请求取消概率。

任何抢占策略都应做成可观测事件：记录被抢占请求、释放的 block、换出或重算字节、恢复时间、额外 token 计算和最终任务成功率。否则“吞吐提高”可能只是把代价转成了更高的 TPOT 和更多失败。

## 7.7 量化：压缩表示而不是免费性能

### 7.7.1 权重、激活和 KV 的量化对象

权重量化减少模型参数存储和读取带宽；激活量化减少中间张量成本；KV Cache 量化减少请求状态容量。三者的误差传播、kernel 要求和质量风险不同，不能都叫“INT4 优化”。

PTQ 在训练后使用校准数据估计尺度，成本较低；QAT 在训练中模拟量化误差，可能获得更稳质量但训练成本更高。Weight-only 量化常在加载或矩阵乘时反量化；如果反量化和矩阵乘 kernel 不高效，显存下降不必然转成速度提升。

### 7.7.2 GPTQ、AWQ 与工程证据

GPTQ 类方法用近似二阶信息逐层补偿权重量化误差；AWQ 类方法利用激活统计保护更重要的权重通道。论文或项目报告中的质量和速度结果依赖校准数据、模型架构、硬件、kernel、batch、序列长度和评估任务。

因此量化候选至少要和 FP16/BF16 baseline 比较：

| 维度 | 必须检查的对象 |
| --- | --- |
| 质量 | 通用、领域、数学、代码、长上下文、格式、工具参数 |
| 安全 | 正常请求误拒、有害请求拒答、注入和越权工具 |
| 延迟 | TTFT、TPOT、P95/P99、流式间隔 |
| 资源 | 权重、KV、临时 buffer、OOM、并发上限 |
| 协议 | JSON、EOS、stop、函数 schema、错误恢复 |

PPL 变化很小，也不能保证 JSON、代码边界、工具调用或安全判断没有退化。平均 token 指标需要任务级失败样例补充。

### 7.7.3 一个权重显存例子

若模型有 `N` 个参数，FP16 每个参数占 2 字节，INT4 理想存储为每个参数 0.5 字节。忽略 scale、zero point、group metadata 和对齐时，权重存储约为：

~~~math
M_{\mathrm{FP16}}
\approx2N,
\qquad
M_{\mathrm{INT4}}
\approx0.5N
~~~

理想比例是四分之一，但实际 INT4 文件还要存 group scale、零点、索引和可能的高精度 outlier。更重要的是，权重节省出来的显存可能被更多 KV blocks 占用，最终吞吐取决于调度和请求分布。

### 7.7.4 校准数据决定哪些误差会被看见

后训练量化通常要用一批校准样本估计尺度、离群值和分组统计。校准集不是越大越好，也不是随便抽几段网页就足够。它应覆盖线上真正会出现的输入形态：自然语言、代码、表格、JSON、长上下文、多轮对话、工具 schema、不同语言以及领域术语。若线上主要是合同问答，而校准集只有通用英文新闻，权重可能在平均 token 误差上表现良好，却在日期、金额、否定词和结构化输出上失真。

以对称的均匀量化为例，设某个权重分组中的真实值为 `x`，整数表示范围为 `[-Q_max, Q_max]`，尺度为 `s`，量化和反量化可以简化为：

~~~math
q
=
\operatorname{clip}
\left(
\operatorname{round}\left(\frac{x}{s}\right),
-Q_{\max},
Q_{\max}
\right)
~~~

~~~math
\hat{x}
=
s q
~~~

`q` 是存储或计算使用的整数值，反量化后的近似值记为 `x_hat`。尺度 `s` 可以由分组最大绝对值、百分位值或更复杂的误差目标估计。最大值对离群点敏感，百分位值可能截断少量极端值；选择哪一种，应由目标任务和 kernel 支持共同决定。

校准时至少要记录三个层次的误差：

1. 数值误差：权重、激活或 KV 的绝对误差、相对误差和离群分布；
2. token 行为误差：logit 差异、top-1 改变、EOS/stop 概率和重复模式；
3. 任务误差：JSON 解析、代码测试、引用支持、工具参数、拒答和业务成功率。

前两层适合快速筛选，第三层才接近产品风险。一个量化版本可能只让少数 token 的 logit 改变，却恰好改变了 `}`、函数名或日期数字，使整个任务失败。因此不能用一个全局平均误差替代格式和动作级测试。

如果测试集被划分为若干业务桶 `b`，每个桶的样本数为 `n_b`，该桶的任务失败率为 `e_b`，权重为 `w_b`，可以用加权摘要描述已知流量：

~~~math
E_{\mathrm{task}}
=
\frac{\sum_b w_b e_b}{\sum_b w_b}
~~~

这里的 `w_b` 可以代表线上流量占比，也可以代表安全和合规风险权重。两种权重回答不同问题：流量权重描述平均用户体验，风险权重描述高影响失败。报告时必须说明使用哪一个，否则“整体没有下降”可能掩盖一个小流量但高后果的回归。

校准数据还要遵守数据治理约束。不能为了提高量化效果，把客户原文、密钥或个人信息未经授权复制到离线 artifact；也不能让校准集与最终评估集大量重复，导致尺度调得很好但泛化证据虚高。实际工程中应给校准集记录来源、时间、语言、任务桶、脱敏方式和版本 digest。

### 7.7.5 KV Cache 量化与长上下文退化

KV Cache 量化与权重量化不同。权重误差会在每次前向传播中影响表示，KV 误差则随着上下文长度积累，并且会反复参与后续 token 的 attention。对短问答几乎无影响的 KV 量化，可能在长文档定位、跨段引用和多轮对话中暴露问题。

工程上常见的取舍包括：按 token、按 head 或按 group 估计尺度；保留最近一段 token 的高精度 residual cache；对 K 和 V 使用不同的表示；为长上下文和高注意力敏感层保留更高精度。每一种设计都会改变显存、kernel 和误差结构，不能只看“KV 从 16 bit 变成 8 bit”这一行配置。

测试 KV 量化时应固定权重、tokenizer、模板和调度，只改变 KV 表示，并按上下文长度分桶：短对话、长检索、跨段问答、长代码和多轮工具轨迹分别报告。若只在短 prompt 上测平均准确率，无法证明长上下文缓存安全。量化版本的缓存命名空间也必须独立；不同 dtype 或 scale 规则的旧 KV 不能被新 worker 直接读取。

## 7.8 Speculative Decoding 与多 token 预测

### 7.8.1 提案与验证

Speculative decoding 使用较小的 draft model 先提出一段候选，target model 再验证。若候选与 target 分布兼容，可以在一次 target 前向中接受多个 token，减少昂贵的逐 token 调用。正确实现需要处理接受、拒绝、替代 token、随机采样和 KV 状态回滚。

它的收益不是由 draft 模型大小单独决定，而取决于接受率、候选长度、draft 延迟、target 验证是否高效、batch 混合和输出分布。温度、任务类型、语言和代码结构都会改变接受率。

### 7.8.2 Medusa、EAGLE 与验证边界

Medusa 使用额外预测头提出未来 token 候选；EAGLE 类方法更多利用 hidden/feature 层预测未来状态。它们可以放在“proposal -> verification”框架下理解，但不能把多个候选直接拼到上下文而跳过 target 语义验证。

上线评估应同时报告：

- 平均接受 token 数和接受率；
- draft/extra head 的额外计算；
- target 验证的 batch 形状和 kernel；
- KV Cache 额外占用与回滚成本；
- 不同温度、长度、任务和并发下的 P50/P99；
- 与不使用 speculative decoding 时的质量与输出分布一致性。

### 7.8.3 接受率如何影响吞吐

可以把一次 speculative round 看成“draft 提案、target 批量验证、提交一段最终 token”。设每轮 draft 提出 `k` 个候选，最终被提交到输出的 token 数为随机变量 `A`，其中已经包含被接受的候选和拒绝后的替代 token。设 draft 和验证一轮的平均耗时分别为 `t_draft(k)` 与 `t_verify(k)`，则一个简化的输出速率估计为：

~~~math
R_{\mathrm{spec}}
\approx
\frac{\mathbb{E}[A]}{t_{\mathrm{draft}}(k)+t_{\mathrm{verify}}(k)}
~~~

普通 target decode 若平均每个 token 用时 `t_target`，则对应速率约为：

~~~math
R_{\mathrm{base}}
\approx
\frac{1}{t_{\mathrm{target}}}
~~~

假设 `k=4`，四轮实验中最终提交 token 数分别为 4、3、1、4，因此 `E[A]=3`。如果 draft 生成四个候选需要 24 ms，target 一次验证需要 36 ms，则：

~~~math
R_{\mathrm{spec}}
\approx
\frac{3}{24+36}\ \mathrm{tokens/ms}
=50\ \mathrm{tokens/s}
~~~

如果普通 target decode 每个 token 需要 42 ms，则：

~~~math
R_{\mathrm{base}}
\approx
\frac{1}{42}\ \mathrm{tokens/ms}
\approx23.8\ \mathrm{tokens/s}
~~~

这个教学估算给出约 2.1 倍的速率比，但它没有计入 batch 形状变化、额外 KV、调度等待、网络和不同输出分布。更重要的是，`E[A]` 不是固定模型属性：摘要、代码补全和结构化 JSON 往往有不同的候选一致性；提高温度通常会降低 draft 与 target 的一致程度；并发增加后，验证 kernel 可能更高效，也可能因不同长度请求混合而变差。

正确性边界也必须单独验证。若系统改变了 target 的采样分布，得到的更高 token 速率不能直接称为等价加速；若拒绝分支没有正确回滚 draft 的 KV，后续 token 会建立在错误历史上。上线前应比较相同随机种子或相同采样协议下的输出分布、停止原因、工具 JSON 和任务成功率，并把接受率按任务桶和长度桶报告。

## 7.9 RAG 部署：证据链而不是向量库加模型

### 7.9.1 离线索引链路

企业 RAG 的离线链路通常包括 artifact 接入、版本登记、解析/OCR、结构保留、清洗去重、chunk、metadata、权限标签、embedding、关键词索引和向量索引。每个 chunk 应能回到文档版本、页码、段落或区域，而不是只有一串不可解释的向量。

稳定文档和实时数据的职责不同：前者适合版本化索引和缓存，后者可能需要数据库查询或工具调用。把实时库存、权限和合同状态永久写进 embedding，会产生新鲜度和撤销困难。

### 7.9.2 在线检索链路

在线流程可以是：

~~~text
用户问题
    -> query normalization/rewrite
    -> 权限过滤
    -> lexical + vector recall
    -> rerank
    -> context selection
    -> 生成与引用
    -> claim 检查、拒答与反馈回流
~~~

检索失败、解析失败、权限误杀、rerank 错误、上下文截断和生成幻觉是不同故障。回答错了不能只换 embedding model。

### 7.9.3 从问题到证据包

一个可追溯的在线检索请求，不应只返回一组相似度分数，而应形成“证据包”。证据包至少包含原始问题、规范化问题、检索器版本、过滤条件、候选 chunk 的文档版本与权限标签、rerank 分数、最终选入的片段、上下文截断位置以及生成时使用的引用标识。这样当用户指出答案错误时，工程师可以判断错误发生在资料不存在、检索没有取到、权限过滤误删、重排排序错误、上下文被截断，还是模型没有遵守证据。

以合同问答为例。用户问：“乙方在验收后多少天内收到尾款？”索引中可能存在三份看起来相似的条款：旧合同写 30 天，续签合同写 45 天，模板文档写 60 天。正确流程不是把三段文字都拼进 prompt 再让模型“自己判断”，而是先用租户、合同 id、有效期和文档版本过滤，再用关键词和向量召回，重排时优先包含“尾款”“验收”“支付期限”的完整条款，最后把冲突版本以明确 metadata 提供给生成器。如果当前合同没有可确认条款，答案应说明缺少证据，而不是选择一个看起来最常见的数字。

检索返回的片段应当和权限检查绑定。先全库召回、再在最后一步删除无权片段，会浪费资源并可能让 reranker 或 query rewrite 看到不该暴露的标题和摘要；但过早使用粗糙的权限过滤，也可能误删用户有权看到的继承权限。权限服务需要定义资源、主体、动作、时间和版本，并把最终过滤决策记录到 trace。缓存命中时同样要重新验证当前权限，不能因为 prefix 或 response cache 命中就跳过授权。

上下文选择是一个独立的优化问题。候选片段越多，召回遗漏的概率可能下降，但 prefill、噪声、冲突和引用错配会增加。可以按片段的相关性、来源可信度、新鲜度、覆盖的 claim、token 成本和冲突关系选择子集。一个片段只有在“被选入上下文”之后才有机会支持答案；出现在 top-k 列表里不等于真正参与生成。

### 7.9.4 RAG 的分层指标

设测试问题数为 `N`，其中能在索引中找到标注证据的问题数为 `N_evidence`；Top-k 召回命中的问题数为 `N_hit`。最终回答中的可验证 claim 总数为 `N_claims`，其中被证据支持的 claim 数为 `N_supported`，可分别报告：

~~~math
\mathrm{Recall@k}
=
\frac{N_{\mathrm{hit}}}{N_{\mathrm{evidence}}}
~~~

~~~math
\mathrm{CitationSupport}
=
\frac{N_{\mathrm{supported}}}{N_{\mathrm{claims}}}
~~~

这些分母必须按项目定义处理：一个问题有多个证据、一个 claim 需要多个来源、无证据问题是否计入拒答评估，都会改变数值。还要报告 rerank 延迟、权限过滤召回损失、新鲜度、上下文 token 和 unsupported claim，而不是只看最终答案分数。

### 7.9.5 引用不等于证据支持

模型在句末加一个文档链接，不代表该文档支持句子中的每个 claim。更可靠的评估要把答案拆成可验证 claim，判断引用是否存在、来源版本是否正确、证据是否蕴含或足以支持结论、是否遗漏冲突信息。

当检索结果不足时，系统应允许“不足以判断”或请求补充信息。把拒答率调到很低，不会自动提高事实性；把引用数量调高，也可能产生装饰性引用。

### 7.9.6 RAG 资源预算

RAG 的总延迟和成本包含 query rewrite、embedding、召回、rerank、context prefill、生成和引用检查。若一次请求触发 `n_call` 次模型调用，每次调用成本为 `c_i`，检索和存储成本为 `c_retrieval`，则：

~~~math
C_{\mathrm{request}}
=
\sum_{i=1}^{n_{\mathrm{call}}}c_i
+c_{\mathrm{retrieval}}
~~~

增加一个 verifier 可能提高引用质量，也会增加 prefill、延迟和失败路径。RAG 降本不能只缩小主模型，还要减少无效召回、重复 query rewrite、过长 context 和不必要的二次生成。

## 7.10 Agent 与工具调用：部署系统必须拥有动作边界

### 7.10.1 Function calling 只是协议起点

模型输出一个符合 JSON schema 的函数调用，不等于动作可以执行。完整工具链应包括：工具注册、版本化 schema、参数校验、身份和资源授权、风险判断、执行隔离、超时、幂等、审计、结果校验和错误恢复。

工具调用至少要区分：

- 只读查询；
- 可逆写操作；
- 不可逆或高影响操作；
- 外部通信、付款、发布、删除等动作。

不同动作应有不同确认和授权要求。高风险写操作不能只依赖 system prompt，也不能把模型的“我已完成”当作执行成功。

### 7.10.2 工具成功与任务成功不是一回事

设任务包含 `N_tool` 次工具调用，其中成功完成并返回有效结果的次数为 `N_tool_success`；任务最终达到业务目标的数量为 `N_task_success`，总任务数为 `N_task`。可以分层报告：

~~~math
\mathrm{ToolSuccessRate}
=
\frac{N_{\mathrm{tool\_success}}}{N_{\mathrm{tool}}}
~~~

~~~math
\mathrm{TaskSuccessRate}
=
\frac{N_{\mathrm{task\_success}}}{N_{\mathrm{task}}}
~~~

工具成功但参数语义错误，任务仍可能失败；工具失败但重试后任务成功，也应记录重试和副作用。需要把“调用发出”“服务返回 2xx”“结果通过业务校验”“目标状态改变”分成事件，而不是只看 HTTP status。

### 7.10.3 Prompt Injection 与工具注入

RAG 文档、网页和工具结果都可能包含模型可读的指令。部署系统要把数据、指令、权限和动作分开：检索内容是证据，不自动升级为 system instruction；工具返回的文本不能改变授权策略；服务端每次执行都重新检查用户、资源和动作权限。

日志要记录模型看到的工具结果摘要、最终参数、授权决策、执行结果和人工确认。防御目标不是让模型永远识别所有注入，而是即使识别失败，也不能直接越权或执行不可逆动作。

### 7.10.4 幂等、重试和 unknown

网络超时可能发生在服务已经执行动作之后。若系统无幂等键，自动重试可能重复退款、重复发送邮件或重复发布。执行器应返回明确成功、明确失败或 unknown；unknown 时先查询动作状态或进入人工/补偿流程，不能让模型猜测结果。

### 7.10.5 工具执行的四个边界

把一次工具调用拆成四个边界，有助于定位“模型说做了但系统没有做”的问题：

1. 语法边界：输出能否解析，字段类型、枚举和必填字段是否符合 schema；
2. 语义边界：参数是否指向真实存在的资源，时间、金额、租户和对象关系是否合理；
3. 授权边界：当前主体是否有权对这个资源执行这个动作，授权是否仍然有效；
4. 执行边界：下游是否实际接受并完成动作，结果是否能被独立查询确认。

模型生成的 JSON 只覆盖第一层的一部分。即使 `contract_id` 是合法字符串，也不说明它属于当前租户；即使数据库返回 200，也不说明查询结果满足业务条件。工具适配器应在模型和下游服务之间建立一个窄接口，模型只提供候选参数，服务端补充用户身份、租户、授权版本、幂等键和审计上下文。

### 7.10.6 一个带超时的工具调用案例

“砚桥合同助手”有一个只读的 `get_contract_status` 工具。模型可以提出合同编号，但不能自行指定租户。服务端收到调用后，先解析 schema，再把合同编号规范化，向权限服务询问当前用户是否能读取该合同，最后才访问合同数据库。数据库结果中的备注字段仍然是数据，不能改变下一次工具调用的授权规则。

假设数据库连接在 800 ms 后超时。此时有三种不同情况：

- trace 明确显示请求未到达数据库：可以记为失败并安全重试；
- trace 显示数据库已接受请求，但响应丢失：结果是 `UNKNOWN`，应查询请求状态或使用同一个幂等键重放；
- trace 显示读取已完成，只是响应在网关丢失：可以通过查询日志或结果缓存确认成功。

如果三种情况都返回“工具失败”，上层 Agent 可能重复调用或向用户报告错误；如果三种情况都返回“工具成功”，则可能把未确认的状态写入合同摘要。执行器应把状态、证据和下一步动作一起返回，例如 `status=unknown`、`operation_id=op-52`、`retryable=false`、`reconcile_after=30s`。模型可以据此告诉用户“暂时无法确认”，但不能凭语言流畅度猜测合同状态。

对于可写工具，还要让幂等键贯穿请求重试。可以用 `idempotency_key = hash(user_id, tool_name, canonical_args, generation_id)` 的形式生成候选键，但是否允许重试仍由工具的业务语义决定。一个“发送邮件”工具即使参数相同，也可能是用户明确要求发送两封邮件，因此键不能简单由文本相似度决定；应由业务动作和调用意图定义。

### 7.10.7 工具结果不能直接变成模型事实

工具返回值要有来源、时间和有效期。库存、合同状态、余额等易变数据应带 `observed_at` 和 `expires_at`；模型在较长推理链中再次使用时，系统应判断数据是否过期。对关键动作，可以要求执行器返回结构化结果和可查询凭证，而不是只返回一段自然语言。

这种设计同时改善可观测性和提示注入防御：结果中的文本可以被模型阅读，但授权服务只信任服务端传入的身份和资源字段；模型生成的“已确认”不能覆盖执行器的 `UNKNOWN`。第 8 章会继续讨论更系统的安全与对齐问题，本章只强调部署边界必须在模型之外存在。

## 7.11 延迟、吞吐、可靠性和成本的统一账本

### 7.11.1 请求时间线

一次生产请求的端到端时间可以拆成：

~~~math
t_{\mathrm{e2e}}
=
t_{\mathrm{queue}}
+t_{\mathrm{preprocess}}
+t_{\mathrm{retrieve}}
+t_{\mathrm{prefill}}
+t_{\mathrm{decode}}
+t_{\mathrm{postprocess}}
+t_{\mathrm{network}}
~~~

不同阶段可能并行或重叠，因此这不是所有实现的严格加法；它是排查和预算的时间轴。工具调用、审核、rerank 和重试会把额外阶段插入其中。平均值之外要看 P95/P99 和失败请求，因为尾部通常由长 prompt、KV 紧张、排队或下游超时主导。

### 7.11.2 吞吐要说明单位

可以报告 request/sec、input tokens/sec、output tokens/sec、total tokens/sec 或 successful tasks/sec。它们回答不同问题：

- request/sec 受请求长度影响很大；
- input tokens/sec 更接近 prefill 处理能力；
- output tokens/sec 更接近 decode 产能；
- successful tasks/sec 把重试、工具失败和质量纳入业务结果。

服务比较必须使用同一请求长度分布、同一 max output、同一解码参数、同一成功判定和同一硬件。单卡实验的 tokens/sec 不能直接外推到多租户 P99。

### 7.11.3 单位成功任务成本

设一段时间内总服务成本为 `C_total`，任务数为 `N_task`，其中成功任务数为 `N_success`。单位成功任务成本为：

~~~math
C_{\mathrm{per\ success}}
=
\frac{C_{\mathrm{total}}}{N_{\mathrm{success}}}
~~~

`C_total` 应包含 GPU、存储、网络、检索、工具、重试、冗余和人工处理。成功定义必须提前约定：格式正确、事实正确、工具状态改变和用户满意可能是不同层级。一个小模型单次便宜，但如果重试多、任务失败和人工介入多，单位成功任务成本可能更高。

## 7.12 线上排障：从症状到最小对照

### 7.12.1 TTFT 变高

先按时间轴拆分 queue、tokenizer、RAG/tool 前处理、prefill 和网络。重点检查输入 token 分布是否变化，长 prompt 是否抢占 prefill，prefix cache 是否失效，路由是否把流量打到热点 worker，模型或量化版本是否更换。

最小对照是固定 checkpoint、固定 prompt 和固定解码参数，只绕过 RAG/tool 重新测；再固定输入，比较 cache 命中和未命中；最后在同一 worker 上比较不同 batch。这样才能区分业务前处理、排队和 GPU prefill。

### 7.12.2 TPOT 变高或输出抖动

检查 active batch、平均和 P99 上下文、KV block 使用率、显存带宽、权重 kernel、量化反量化、decode 被 prefill 打断的比例，以及 speculative decoding 的接受率。流式间隔异常还要看网关缓冲和客户端消费速度。

只提高 batch 可能提升平均吞吐，却恶化单请求 TPOT；只降低 max output 可能改善 E2E，却改变任务成功率。优化必须同时报告质量和尾延迟。

### 7.12.3 OOM 与资源泄漏

OOM 增加时，先区分启动时权重 OOM、prefill 激活 OOM、decode KV OOM、临时 workspace OOM 和碎片/泄漏。检查上下文与并发分布、max output、KV dtype、block pool、prefix cache、LoRA adapter、取消路径和请求异常结束。

固定负载运行创建/完成/取消循环，观察 block 使用是否回到基线；如果不回收，再比较 worker 内存和 allocator 统计。单次 OOM 日志无法证明是容量不足还是释放错误。

### 7.12.4 质量突然下降

固定一组离线 prompt，对比模型权重、tokenizer、chat template、system prompt、stop token、generation 参数、量化版本、RAG index、embedding/reranker 和安全策略。输出要按格式、事实、引用、工具参数、拒答和长度分类。

模型版本切换后质量变差，不一定是模型能力下降；模板重复、旧 tokenizer、缓存复用、RAG 权限过滤和输出截断都可能制造相同症状。先做协议和依赖对照，再讨论模型本身。

### 7.12.5 Trace 要记录因果，而不只是耗时

一条端到端 trace 应能回答三个问题：请求经过了哪些阶段，阶段之间用了哪些版本和资源，最后的业务结果是什么。至少可以记录以下字段：

| 层次 | 关键字段 | 作用 |
| --- | --- | --- |
| 请求 | request id、generation id、租户、优先级、取消原因 | 关联重试、权限和用户生命周期 |
| 输入 | tokenizer/template digest、输入 token、截断位置、工具 schema digest | 重放输入协议 |
| 调度 | queue time、worker、batch id、active sequences、KV blocks | 解释排队和资源竞争 |
| 生成 | 首 token 时间、输出 token、stop reason、采样配置 | 区分模型行为和网络表现 |
| 外部链路 | index 版本、候选数、选入 token、工具 operation id | 复原 RAG 和动作结果 |
| 结果 | 格式校验、引用支持、任务状态、成本归因 | 把 token 指标连接到业务结果 |

日志、指标和 trace 的职责不同。指标适合发现整体趋势，例如 P99 TTFT 上升；trace 适合解释一条异常请求；结构化日志适合查询某个版本、租户或错误类型。把所有 prompt 原文都塞进日志会扩大隐私风险，把所有字段都删掉又会让事故无法复盘。应根据敏感等级保存脱敏摘要、哈希、版本和必要的重放 artifact，并为访问审计设置独立权限。

采样策略也会改变证据质量。只采样成功请求，无法解释失败尾部；只保存错误请求，无法计算基线和回归比例。可以对所有请求保留低成本计数和延迟，对错误、长上下文、权限决策、工具 unknown 和随机抽样请求保留更完整 trace。采样规则必须版本化，否则不同版本的“错误率”可能只是观测覆盖不同。

### 7.12.6 先构造最小对照，再扩大压力

部署排障不应一开始就同时改模型、引擎、量化、batch 和 prompt。更稳妥的实验顺序是先固定 checkpoint、tokenizer、模板、采样参数和硬件，只改变一个运行时因素；再固定运行时，只改变输入长度、并发或 RAG/工具链；最后才在代表性流量上测试组合方案。每次实验应保存 workload digest，避免“同一个测试集”在不同轮次被悄悄改写。

如果结果指标是越低越好，例如 P95 TTFT，变体和基线的差值可以写成：

~~~math
\Delta M
=
M_{\mathrm{variant}}-M_{\mathrm{baseline}}
~~~

如果指标是越高越好，例如任务成功率，则同样的正负号含义相反。报告时应同时给绝对值、相对变化、置信区间或重复实验的离散程度，不能只写“提升 20%”。对于具有随机采样的生成任务，还要固定随机种子或使用足够多的配对样本，并报告输出分布变化。

## 7.13 灰度、回滚和版本治理

### 7.13.1 一个模型版本包含多个依赖

版本标识至少要绑定：checkpoint digest、tokenizer、chat template、推理引擎版本、量化配置、generation 默认值、RAG index/embedding、工具 schema、安全策略和硬件/编译选项。只给权重版本号，无法解释为什么同一模型在不同 worker 行为不同。

### 7.13.2 灰度指标要按请求和任务看

灰度比较要同时看：TTFT/TPOT/P99、错误率、OOM、GPU 利用率、KV blocks、输入/输出 token、任务成功率、引用支持、工具成功、拒答和单位成功成本。总体平均可能掩盖长上下文、特定租户或高风险动作的退化。

灰度流量要固定或记录路由规则，避免新旧版本接收完全不同的请求分布。对同一输入做配对 shadow 或离线重放，能减少请求分布差异造成的误判。

### 7.13.3 回滚不能只换镜像

回滚时要恢复与模型一起变化的 tokenizer/template、RAG index、工具 schema、缓存命名空间和安全策略。旧版本不能读取新版本格式的 KV/prefix cache，也不能继续使用已撤销权限的旧缓存。回滚完成后要验证固定 prompt、健康请求、取消、流式和关键工具动作。

## 7.14 容量规划：按 token 和状态，而不是只按 QPS

### 7.14.1 输入输出分布

容量规划要采集输入 token、输出 token、上下文长度、并发、请求到达率、取消率、工具/RAG 调用放大和优先级。两个 QPS 相同的业务，如果一个平均输入 200 token、输出 50 token，另一个输入 32K、输出 4K，GPU 和 KV 需求完全不同。

### 7.14.2 KV 容量上限

设可分配给 KV 的显存为 `M_budget`，单请求实际 KV 占用为 `m_i`，并发请求集合为 `A`，则容量约束为：

~~~math
\sum_{i\in A}m_i
\le
M_{\mathrm{budget}}
~~~

由于 `m_i` 随历史长度增长，调度器需要预留输出上限或采用动态 block 申请和抢占策略。只用平均长度做容量规划会低估 P99 长请求；只用最大长度又可能浪费资源。应使用长度分桶和压力测试估计可接受的 tail。

### 7.14.3 Little 定律的边界

在稳定系统中，平均在途请求数 `L`、到达率 `lambda` 和平均时间 `W` 满足：

~~~math
L
\approx
\lambda W
~~~

它可以帮助估算队列和并发，但不能直接给出 GPU 数量。LLM 服务的 `W` 受 token 长度、调度优先级、KV 状态和阶段重叠影响，系统不稳定、请求被拒绝或存在大量取消时，简单套用会误导容量结论。

### 7.14.4 单卡 profiling 到集群排期

先在目标硬件和引擎上对代表性请求做 prefill/decode profiling，记录不同 batch、长度和量化的曲线，再根据流量分布和冗余系数估算 GPU。集群还要留出灰度、故障、维护、冷热切换和突发流量空间。

## 7.15 一个完整故障案例：高并发知识助手的延迟与越权

### 7.15.1 初始设计

“砚桥合同助手”使用一个 32B instruct checkpoint，接入企业合同 RAG 和只读合同查询工具。服务通过 API gateway 接收多轮消息，router 按租户选择 worker；worker 使用 GQA、Paged KV blocks、continuous batching 和 BF16；RAG 召回合同条款，工具查询当前合同状态。

产品目标是 P95 TTFT 小于 1.5 秒，P95 TPOT 小于 80 ms，引用支持率不低于 0.90，合同查询任务成功率不低于 0.95。高风险写操作不在本服务范围内，但只读工具也必须按租户和合同权限过滤。

### 7.15.2 线上症状

一次版本发布后，P95 TTFT 从 1.2 秒升到 3.8 秒，P99 TPOT 变差，部分请求 OOM。与此同时，少量答案引用了用户无权访问的旧合同条款。

### 7.15.3 证据分层

延迟日志显示新版本启用了更长的默认 max output，且 RAG query rewrite 从一次调用变成了两次；长 prompt 的 prefill 与 decode 争用同一 token budget。KV block 使用在取消请求后没有完全回落，压力测试中 block pool 单调增加。

越权案例的 trace 显示 prefix cache key 只包含 token 前缀，没有包含租户和权限版本；两个租户使用相同 system/template 前缀时，带有合同内容的缓存块被错误复用。这里既是缓存隔离错误，也是版本和权限字段缺失，不能把问题归咎于模型“记住了别人的合同”。

### 7.15.4 修复路径

第一步暂停长上下文和受影响租户的 prefix cache，回滚 query rewrite 和 max output 默认值；第二步修复取消路径，确保 scheduler、worker 和 block pool 都收到终止状态；第三步把 checkpoint、template、租户、权限版本、工具 schema 和 RAG index 纳入缓存命名空间；第四步用固定 prompt、跨租户同前缀、取消循环、长短请求混合和权限变更测试回归。

### 7.15.5 结果与边界

修复后 P95 TTFT 回到 1.4 秒，P99 TPOT 回到 76 ms，取消压力测试的 KV blocks 能回收，权限切换后的缓存不会命中旧条款。引用支持率恢复到 0.92，合同查询成功率为 0.96。这个案例不能证明系统永远没有越权；它只证明在注册的租户、权限、缓存、取消和请求切片下，已复现的故障得到修复。还需持续做跨租户、权限撤销、旧版本回滚和新 RAG index 的回归。

## 7.16 评估部署改动：质量和系统指标必须配对

### 7.16.1 质量评估

模型或引擎改动至少要覆盖通用能力、业务任务、格式、工具参数、RAG 召回和引用、安全拒答、长上下文、取消和错误恢复。量化和 speculative decoding 还要检查输出分布、重复率、停止原因和非确定性。

### 7.16.2 系统评估

同一请求集合下报告 TTFT、TPOT、E2E、P50/P95/P99、prefill/decode 吞吐、KV 使用、显存峰值、OOM、错误率、取消回收、GPU 利用率、成本和成功任务数。测试负载要包含短请求、长 prompt、长输出、突发、混合优先级和 RAG/工具放大。

### 7.16.3 任务成功成本

设一批任务的总成本为 `C_total`，成功完成的任务数为 `N_success`，则：

~~~math
C_{\mathrm{success}}
=
\frac{C_{\mathrm{total}}}{N_{\mathrm{success}}}
~~~

同一个模型在纯文本问答和工具任务上的成功定义不同，成本也不能只按 output token 计。重试、RAG、rerank、人工审核和失败回滚都应进入总成本。

### 7.16.4 部署实验矩阵

一个可复现的部署实验至少要把下面几类变量组合起来：

| 维度 | 最小分桶 |
| --- | --- |
| 输入长度 | 128、2K、16K、最大上下文附近 |
| 输出长度 | 32、256、1K、达到上限 |
| 并发 | 1、稳态目标、突发峰值 |
| 请求混合 | 短/长、交互/批处理、同租户/多租户 |
| 运行时 | baseline、continuous batching、chunked prefill |
| KV | 无共享、prefix 命中、不同 block size、不同 dtype |
| 权重 | BF16/FP16、INT8、INT4 候选 |
| 外部链路 | 无 RAG、召回、rerank、工具成功、工具超时 |
| 解码 | greedy、采样、speculative decoding |

不需要穷举所有笛卡尔积，但要保留能暴露不同瓶颈的代表性切片。例如短输入长输出主要观察 decode 和 KV 带宽，长输入短输出主要观察 prefill 和 TTFT，长输入长输出同时考验上下文容量和调度；RAG/工具请求还要测内部调用放大、取消和 unknown。每个切片都应有相同的成功判定、错误分类和资源上限。

实验结果不应只给一张 tokens/sec 表。至少同时保存质量、延迟、吞吐、显存、KV、错误、取消回收、任务成功和成本；对高风险动作，还要保留授权拒绝、越权尝试、重复副作用和人工处理结果。这样才能判断一个优化是否只是把成本或失败转移到另一层。

### 7.16.5 最小请求状态转换实现

下面的纯 Python 代码只实现状态约束，不执行 GPU、网络或工具动作。它的价值在于把“完成后不能再取消”“未知结果不能自动重试”等协议规则变成可测试的程序契约：

~~~python
from dataclasses import dataclass


TERMINAL = {"COMPLETED", "CANCELLED", "FAILED"}

ALLOWED = {
    "CREATED": {"QUEUED", "CANCELLED", "FAILED"},
    "QUEUED": {"PREFILLING", "CANCELLED", "FAILED"},
    "PREFILLING": {"DECODING", "CANCELLED", "FAILED"},
    "DECODING": {"WAITING_TOOL", "COMPLETED", "CANCELLED", "FAILED"},
    "WAITING_TOOL": {"DECODING", "COMPLETED", "FAILED", "UNKNOWN"},
    "UNKNOWN": {"WAITING_TOOL", "COMPLETED", "FAILED"},
    "COMPLETED": set(),
    "CANCELLED": set(),
    "FAILED": set(),
}


@dataclass
class RequestState:
    request_id: str
    status: str = "CREATED"
    sequence: int = 0

    def transition(self, next_status: str) -> None:
        if next_status not in ALLOWED[self.status]:
            raise ValueError(
                f"invalid transition: {self.status} -> {next_status}"
            )
        if self.status in TERMINAL:
            raise ValueError("terminal request cannot transition")
        self.status = next_status
        self.sequence += 1


request = RequestState("r-1842")
for status in ("QUEUED", "PREFILLING", "DECODING", "WAITING_TOOL", "UNKNOWN"):
    request.transition(status)
assert request.status == "UNKNOWN"
request.transition("WAITING_TOOL")
request.transition("COMPLETED")
assert request.sequence == 7
~~~

代码没有实现分布式一致性。真实服务还要处理重复事件、worker 崩溃、持久化失败、跨进程并发和客户端重连；这些问题应由带版本号或条件更新的状态存储、事件日志和恢复协议解决。教学代码只把状态机中的合法转移作为一个可运行的最小边界。

## 7.17 资料、版本与证据边界

本章使用不同类型的公开资料支撑不同层次的论述：

1. Attention Is All You Need：[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. FlashAttention：[https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
3. FlashAttention-2：[https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)
4. Efficient Memory Management for Large Language Model Serving with PagedAttention：[https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
5. vLLM Documentation：[https://docs.vllm.ai/](https://docs.vllm.ai/)
6. SGLang Documentation：[https://docs.sglang.ai/](https://docs.sglang.ai/)
7. Speculative Decoding：[https://arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192)
8. Medusa：[https://arxiv.org/abs/2401.10774](https://arxiv.org/abs/2401.10774)
9. EAGLE：[https://arxiv.org/abs/2401.15077](https://arxiv.org/abs/2401.15077)
10. GPTQ：[https://arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
11. AWQ：[https://arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)
12. REALM：[https://arxiv.org/abs/2002.08909](https://arxiv.org/abs/2002.08909)
13. DPR：[https://arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)
14. RAG：[https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
15. ReAct：[https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
16. Toolformer：[https://arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761)
17. OWASP Top 10 for LLM Applications：[https://owasp.org/www-project-top-10-for-large-language-model-applications/](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
18. NIST AI Risk Management Framework：[https://www.nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework)

原始论文用于支持 attention、PagedAttention、speculative decoding、量化和 RAG/Agent 方法；vLLM/SGLang 文档用于支持具体引擎的功能边界；OWASP 和 NIST 用于支持治理与风险管理框架。KV 容量、延迟分解、MFU、单位成功任务成本和故障案例中的数字是教学估算或合成案例，不是某个公开服务的实测承诺。

公开论文中的吞吐、接受率和量化质量依赖模型、硬件、请求分布、kernel 和版本，不能直接外推到目标集群。当前模型卡和服务能力还需要回到对应厂商或项目的版本文档、实际 profiling、任务评估和安全审计。本轮写入前尝试联网复核资料入口，但当前环境 DNS 暂时无法解析外部域名；保留这些权威入口，正式发布前应在网络恢复后重新记录 HTTP 状态和版本。

## 7.18 结语：部署是对模型能力的重新定价

部署把一个静态 checkpoint 变成带状态、带预算、带权限和带失败语义的服务。tokenizer 与模板决定模型看到什么，prefill/decode 决定延迟结构，KV Cache 决定并发容量，调度决定不同请求如何共享资源，量化和 speculative decoding 改变计算路径，RAG 和工具把外部证据与副作用引入系统，监控和版本治理决定问题能否被定位和回滚。

初学者应先能解释一次请求为什么慢、一次请求为什么占用显存、为什么 PagedAttention 不是新的 attention 算法，以及为什么模型返回 JSON 不等于工具动作成功。有经验的工程师还要能把 TTFT、TPOT、P99、KV blocks、权限版本、任务成功率和单位成功成本放在同一张证据账本中。

当部署系统能够说明每个 token 从哪里来、占用了什么资源、经历了哪些权限和工具边界、失败后如何释放和恢复，以及一次质量提升是否值得它增加的成本，模型才真正成为可以被稳定交付的服务。下一章将继续讨论 alignment 与 safety，关注模型行为如何被塑造、测量和约束。
