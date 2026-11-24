# 第二十章：FlashAttention、PagedAttention 与 Attention Kernel 优化

## 章节契约

- **前置：** 读者已理解标准 attention 的 query-key 位置对和在线 softmax 基本思路（第十章），以及自回归生成中 K/V 状态逐步增长的方式（第十一章）。本章按需说明 HBM、片上存储和 GPU kernel 的职责，不假设读者有 CUDA 编程经验。
- **离章结果：** 能解释 dense FlashAttention 如何在保留完整 attention 语义时减少中间矩阵数据搬运；能区分 FlashAttention 不同版本对并行划分和目标 GPU 的适配；能解释 decode 阶段为何出现跨 K/V 维度并行；能说明 PagedAttention 的逻辑位置、block table 与物理 KV block 的关系，以及共享 prefix 对分配和正确性的要求；能将 kernel latency、KV 利用率与端到端服务结果分开阅读。
- **本章职责：** 连接 attention 数学算子、GPU 内存层级与 serving 中的 KV 物理布局，解释 FlashAttention、Flash-Decoding 和 PagedAttention 各自在计算和状态访问路径中改变了什么。
- **边界：** 第十章负责 pair 计数、FLOPs、显式 score/probability 存储和在线 softmax 推导；第十一章负责 KV payload 容量与请求状态；第十九章负责 attention pattern；第二十六章负责块稀疏 mask、稀疏 kernel 和动态稀疏。训练/服务调度与完整端到端性能调优分别回链第六册及第二十四册的相关章节。本章不交付 CUDA/PyTorch 编程教程，也不复制 serving benchmark gate。
- **结构研究依据：** 旧稿 23 个 H2 存在重复主题和章节职责漂移；具体诊断与章节调整决定记录于[第二十章精修审计](../../../progress/book-21-transformer-architecture-evolution/chapter-20-audit.md)。事实及来源边界见[Attention kernel 与 Paged KV 研究记录](../../../research/attention-kernel-and-paged-kv-evidence-2026-11-20.md)。

## 主题分配

| 主题 | 本章所需认识 | 完整主讲位置 |
| --- | --- | --- |
| Dense attention 的 IO 行为 | full attention 的 pair 计算与把中间张量写回 HBM 是不同问题；FlashAttention 主要改变 tile 调度和数据移动 | 20.2；pair 数和完整在线 softmax 推导见第十章 |
| FlashAttention 版本演进 | 同一数学语义下，瓶颈从 HBM 往返逐步转向并行划分、硬件异步流水和特定设备的功能单元 | 20.3；论文 benchmark 仅支持对应设备和 workload |
| Decode kernel 并行 | 单步 query 数较少时，仅沿 query/batch 并行可能无法充分利用 GPU；沿 K/V 序列拆分需要合并分块归一化结果 | 20.3；历史状态与 decode 计算账本见第十、十一章 |
| PagedAttention | KV 逻辑顺序不要求物理连续；kernel 通过 block table 按逻辑位置访问分散物理块 | 20.4；KV 数学 payload 见第十一章 |
| Prefix sharing | 多个请求可以引用兼容的共享前缀状态；分支写入、释放和淘汰必须保留映射及所有权正确性 | 20.5；具体缓存 key 和策略依 serving engine 版本 |
| 技术组合与证据 | exact dense kernel、decode 拆分和 paged KV 可以互补，但是否在同一后端/shape 中组合须由版本化实现确认；kernel 结果不能直接当端到端收益 | 20.6；block-sparse 与改变连接图的方法见第二十六章 |

## 正文编号大纲

### 20.1 先分清四类优化对象

本节建立统一的区分：FlashAttention 对完整 attention 的计算与内存访问进行分块，Flash-Decoding 把 decode 时的长 K/V 维度拆开以增加并行度，PagedAttention 则改变 KV 状态的物理分配和地址查找。它们的目标、并行轴和主要状态不同；只要使用相同可见集合且计算正确，前两者和分页读取都不改变 attention 的语义。block-sparse 直接改变可见连接，属于另一类结构选择并回链第十九、二十六章。

### 20.2 FlashAttention：保留完整 attention，减少中间数据往返

本节围绕“算子结果相同，数据在不同存储层之间怎样流动”说明 FlashAttention。普通物化路径会将 score 和 probability 中间张量写入并读回 HBM；FlashAttention 将 Q、K、V 按 tile 送入片上存储，在完整 key 集合上持续更新每行的稳定 softmax 统计与 value 累积，输出结果后丢弃临时 score tile。前向保留紧凑行统计，反向可重算 tile，以计算换取较少中间存储。该方法不删除 dense attention 的 query-key pairs，也不保证逐位浮点相同；更完整的在线 softmax 代数由第十章负责。

### 20.3 从 work partition 到硬件专用流水，再到 decode 并行

本节说明 FlashAttention 版本不是数学 attention 定义的变化，而是对不同计算和数据移动瓶颈的再划分。FlashAttention-2 处理较低 occupancy、非矩阵乘 FLOPs 和 warp 间共享内存通信；FlashAttention-3 面向 Hopper 的异步矩阵计算/数据传输和低精度路径；FlashAttention-4 面向 Blackwell 的硬件吞吐比例变化重新安排异步计算、softmax 与共享内存访问。各论文的速度数字只绑定其 GPU、shape、dtype 和对照实现。另说明 Flash-Decoding 将较长的 K/V 序列切成并行分段并保留可合并的归一化统计，以适应单步 query 数少的 decode 形状；它仍计算全部允许的历史连接，适用收益依赖上下文长度和并行粒度。

### 20.4 PagedAttention：逻辑连续的 KV 与物理分散的 block

本节聚焦逐 token 生成时的 KV 状态分配。每条请求的逻辑序列由固定 token 数的 KV blocks 组成，block table 将逻辑 block 映射到物理 block；新 token 按需占用空闲物理块，attention kernel 按逻辑次序读取分散的 K/V 并跨 block 完成同一 attention 归一化。分页减少最大长度连续预留造成的浪费，并将尾部未填空间限制在最后 block 内，但不压缩活跃 token 的 K/V payload。block table 读取、非连续访问和变长处理本身有开销；PagedAttention 的价值要结合请求分布和系统可接纳的并发判断。

### 20.5 共享前缀：复用读取状态并保护分支写入

本节解释并行采样、beam 分支或重复前缀请求如何共享已计算的完整 KV blocks，减少重复占用或 prefill。共享前缀按只读状态处理；分支要写入共享 block 时需使用写时复制或等价隔离机制，释放时只在不再被引用后回收。PagedAttention 提供 block 共享的结构条件，prefix-cache 的匹配、hash key、淘汰和跨适配器行为则由具体 engine 版本决定；不可把共享 KV 与量化、历史淘汰或任意不同模板的复用混为一谈。

### 20.6 组合、后端选择与证据边界

本节把前述方法放回 prefill、decode 和动态请求三个工作阶段：长 query 计算、少 query/长历史计算和 KV 状态管理的主要并行轴各不相同。不同 backend 可能在输入 shape、dtype、mask、设备和分页布局上有不同限制；高层 API 的名字不证明本次请求实际运行了特定 fused kernel，fallback 与数值次序也会影响观察结果。比较时区分 kernel latency、端到端吞吐/延迟、峰值 KV 占用与请求容量，固定硬件、软件版本、shape、dtype 和 workload；PagedAttention 可能以少量 kernel 管理开销换取更大的可用请求集合，不能把 kernel 变慢直接等同于系统变差，也不能把论文中的系统吞吐外推到其他环境。其他 KV 分配路线可作为比较入口，但不在此展开实现。
