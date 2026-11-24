# 第 70 章 Latent Cache：压缩历史表示

本章聚焦按 token 保存低维 latent 的注意力缓存：每个历史位置仍保留一个可供 query 寻址的表示，但模型不再直接保存各 head 的完整 key 和 value。核心任务是让读者能说明 latent 如何形成、怎样参与 attention、为什么仍随上下文线性增长，并能区分这种表示压缩与 head sharing、量化、token selection 和递归状态。第 63 章主讲 Gated MLA 与 Kimi K3 配置，第 69 章交代混合模型逐层的表示类型，第 71 章承担递归状态及生命周期；本章只复用理解机制所需的基础公式，深入讨论任务保真和公开实验边界，不写 serving 管理流程。

## 70.1 从显式 K/V 到逐 token latent

本节从已学过的显式 KV cache 出发，说明 latent cache 改变的是每个历史位置所保存的特征宽度。DeepSeek-V2 MLA 的联合低秩投影作为主要已公开案例；描述重点是历史状态如何从每 token 多头 K/V 变成共享 latent，而不是把任意模型外置压缩器泛化成 MLA。

### 70.1.1 显式 K/V cache 的逐 token 账本

回顾每层、每个 token 需保留 key 与 value，MHA/GQA/MQA 的差别主要体现为 KV head/group 数。给出普通 cache 的逻辑 payload 变量，为后续表示压缩比较提供同口径基线；分页元数据、工作区和服务端分配仍由第 20 章及第二十四册处理。

### 70.1.2 一个共享 latent 联合承载 key 与 value 信息

围绕 $h_t\mapsto c_t^{KV}\mapsto(k_t,v_t)$ 说明下投影、缓存对象和 K/V 上投影的关系。指出 MLA 中一个 latent 同时服务 key 与 value 路径，压缩空间由模型学习；latent 并不是删掉一部分 head，也不是把整段文本写成一个自然语言摘要。完整推导和 Kimi Gated MLA 变体回链第 63 章。

### 70.1.3 每个位置仍可寻址，缓存仍随长度增长

说明每个历史 token 仍对应自己的 latent，当前 query 仍可对前缀位置计算权重；表示宽度下降并不会把所有历史折叠成固定大小的递归 state。对固定层数与维度的 latent attention，cache 仍随历史 token 数 $T$ 线性增长；递归状态的时间更新与生命周期交给第 71 章。

## 70.2 Attention 怎样直接读取 latent

本节说明缓存变窄后，query-key 打分与 value 聚合如何仍能得到 attention 输出。关键研究抓手是低秩因子、矩阵结合律、query/key 与 value/output 投影吸收；代数上可避免展开所有历史 K/V，但实际 kernel 是否采用何种融合仍须依具体实现核验。

### 70.2.1 把 key 上投影折入 query 路径

说明普通计算会将 latent 上投影为各 head 的 key，再与 query 做点积；通过结合矩阵乘法次序，可先变换 query，再直接与缓存 latent 打分。该等式解释了吸收投影的机制与节省历史展开的条件，不应被写成所有 MLA 推理实现都自动更快。

### 70.2.2 把 value 上投影与输出投影合并

注意力权重作用于 latent 后，可将加权 latent 汇总再映射到 value/head 输出空间，并与输出投影代数合并。解释这一变换保留逐 head 的权重作用，同时避免把整段历史显式展开成完整 value 张量；查询仍须处理其可见的位置集合。

### 70.2.3 RoPE 位置路径影响 latent 定义与字节数

说明位置旋转可能与低秩投影不能任意交换；DeepSeek-V2 将内容 latent 与单独的 RoPE key 分开，逐位置额外缓存位置分量，因此容量式不能只计 $d_c$。KV-Latent 则研究低维 attention head 下的频率感知 RoPE 和继续训练，作为另一种模型改造路线；具体配置与 Gated MLA 的 NoPE 方案回链第 63/64 章。

### 70.2.4 预训练联合学习与 checkpoint 缩维是两种路径

区分 DeepSeek-V2 式架构在模型训练中学习 latent 投影，与 KV-Latent 对既有注意力维度作缩减并通过两阶段继续训练恢复性能。后者的维数、RoPE 适配和蒸馏/训练结果不能直接作为前者的公式或质量保证；两者共同说明 latent 表示需要与模型的注意力计算和训练信号匹配。

## 70.3 不同压缩方法改变不同的轴

本节用统一坐标比较 cache 的表示宽度、每值精度、保留/读取的位置集合和时间状态形式。它们可以组合，但不能把不同机制的压缩倍率相乘后当作同一质量或性能结论；第 69 章的“可见范围”与“历史表示”两轴是本节的前置。

### 70.3.1 MQA/GQA 共享 head 与 latent 压缩维度

说明 MQA/GQA 减少 query heads 所对应的独立 K/V 表示数，而 MLA 类 latent 将跨 K/V 投影压入模型学习的低秩表示；二者改动的张量轴不同，某些设计和评测可组合。比较须记录 KV groups、head width、latent width 和位置附加量，不用“都压 cache”掩盖机制差异。

### 70.3.2 量化减少每个数值的位宽

区分相同张量维度的低位宽存储与降维 latent：前者改变标量表示精度并引入量化误差/scale，后者改变每个 token 的坐标数与投影结构。两者可串接，但内存计算、误差来源和解码成本分别核算；具体 runtime 量化接口由第二十四册维护。

### 70.3.3 Token selection 与 eviction 改变位置集合

区分只跳过部分位置的稀疏读取、实际删除/淘汰历史 token，以及仍为每个保留位置保存 latent 的维度压缩。SALS 展示低秩 key latent 可用于候选选择并与 value 量化、token sparsity 联合；该方法的流程是具体方案，不代表 latent cache 天生不保存被跳过位置。稀疏 attention 的结构基础留在第 26、74/77 章。

### 70.3.4 递归状态把逐 token 历史改成时间更新

说明递归状态依次将新 token 纳入固定结构的 state，不再给所有历史位置保留独立地址；它改变时间维度的表示和查询方式，和“每个位置仍有 latent”根本不同。状态容量、reset、snapshot、事务和迁移均交给第 71 章。

## 70.4 表示容量的瓶颈必须用任务保真来判断

本节区分 latent 的几何/维数压缩、attention 中间计算保持情况和最终任务能力。不能从 latent 维数推断哪些语义必然被保存；也不能将信息瓶颈术语写成所有 MLA 方法都优化的互信息目标。

### 70.4.1 信息瓶颈是容量直觉，不是 MLA 的通用训练目标

用有限维 latent 表示“可供未来 attention 使用的信息预算”这一事实来解释压缩边界；说明经典 Information Bottleneck 的互信息形式与 MLA/KV-Latent 公开的投影和训练目标不是同一项方法定义。表示不要求重构原始 hidden state 或原 K/V 的每一维，真正目标应按相应模型训练和评测设置判断。

### 70.4.2 Key 与 Value 可以采用不同的维度预算

解释 key 参与 query-key 匹配并影响读取权重，value 承载按权重聚合的内容；因此维度分配可能产生不同影响。KV-Latent 报告其特定实验中 $d_{vo}$ 比 $d_{qk}$ 更敏感，这只能作为待检验的设计假设，需在目标模型、层、任务和训练预算上分开消融。

### 70.4.3 按信息类型、距离和干扰测量保真

建立与模型用途一致的评测：精确实体/数字/版本、否定与冲突证据、跨段组合、同义改写问答和一般语言任务，并按证据位置与上下文长度分桶。区分 attention score/输出相似度等中间代理与最终正确性、引用定位等任务指标；长窗口支持或整体平均分不能单独证明所有位置的信息都能可靠使用。

## 70.5 Cache 字节数、读取成本与任务质量需要一起比较

本节把表示压缩转为可复核的资源账本，并解释内存下降不自动等于时延下降。公式统计逐 token 逻辑 payload；实际设备占用还受 dtype、辅助位置向量、scale、alignment、工作区和实现影响。质量与性能数字均绑定来源条件，不拼接异构论文来造通用压缩曲线。

### 70.5.1 以 latent 和附加位置项计算逻辑 payload

给出普通显式 cache 与 latent cache 的每 token/每层元素数及字节数对比，并定义压缩比仅在 dtype、层数、KV group 和位置分量对齐时成立。扩展至序列长度和 batch 的账本，明确 latent 仍有 $O(T)$ 项；不把元素数等同于真实显存占用。

### 70.5.2 分开节省的带宽与新增的投影计算

沿 decode 的读取路径解释更窄 cache 减少哪些历史数据搬运，又可能新增哪些 latent projection、RoPE、top-k、重构或融合计算。全历史 attention 的位置数仍随 $T$ 增长；只有额外改变读取集合的设计才会改变该项。分别比较 prefill、TTFT、单 token decode 延迟、吞吐和峰值显存，避免以理论 payload 代表真实速度。

### 70.5.3 固定预算消融才能解释表示方案的收益

说明有解释力的比较需固定模型/checkpoint、训练 token、数据、上下文、精度、batch、硬件、kernel 与 baseline，分开改变 latent width、RoPE 路径、key/value 维度和 token selection。报质量、cache payload、实际 memory traffic 与时延；等质量比较和等显存比较回答不同问题。

## 70.6 公开结果能证明什么

本节用少量研究案例示范如何读压缩收益，而不把论文摘要的压缩率当作算法常数。DeepSeek-V2 支撑 MLA 的架构与整模型报告，KV-Latent 显示低维配置和有限继续训练之间的质量边界，SALS 展示表示压缩与 token sparsity 可组合且受 reconstruction overhead 影响；它们的模型、目标与指标不能直接横向排名。

### 70.6.1 DeepSeek-V2 的整模型数字不等于 MLA 单因素吞吐

依据论文 §2.1.2–2.1.4 解释 joint latent、吸收投影和 decoupled RoPE cache 账本。若引用 93.3% cache 减少或 5.76× 最大 generation throughput，须紧邻说明其基线为 DeepSeek 67B、结果来自完整 DeepSeek-V2 设计，不能归因于 MLA 一项。

### 70.6.2 KV-Latent 显示维度与任务、训练预算相关

按 ACL 2025 的 LLaMA-3-8B/GQA 与 LLaMA-2-7B/MHA 配置介绍 64/64 与 16/16 的 cache/benchmark/NIAH 结果，指出模型继续训练、数据量、context/task 和硬件条件。该案例可说明容量缩减存在任务与维数依赖，不支持普遍 rank 下限或“低维一定恢复”的结论。

### 70.6.3 SALS 的结果属于低秩、选择与量化的组合

说明 SALS 同时压缩 pre-RoPE key 表示、挑选 token、重构所选 key，并对 value 采用量化；其单卡实验中短序列的重构开销与长序列吞吐结果不同。引用时标明组合机制和模型/基准/硬件，不能将所有加速归因于 latent 维数，也不替第 74/77 章展开稀疏机制。
