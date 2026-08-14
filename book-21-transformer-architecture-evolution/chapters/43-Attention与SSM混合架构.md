# 第四十三章：Attention + SSM 混合架构为什么现实

## 43.1 混合不是折中口号

Attention 和 SSM 关注不同问题：attention 保留显式历史并支持内容寻址，SSM 用固定状态高效扫描长序列。混合架构的价值是让两种信息流各承担擅长的工作，而不是宣称某一方已经被完全替代。

如果一个任务同时包含局部连续信号、超长历史和少量精确证据，纯 attention 可能贵，纯 SSM 可能丢证据，混合路径更有可能通过质量/成本约束。组合后也会增加层 schedule、cache、kernel 和训练复杂度。

## 43.2 两条路径的数学抽象

SSM 路径：

~~~math
s_t=F(s_{t-1},x_t),\qquad y_t^{s}=G(s_t,x_t)
~~~

Attention 路径：

~~~math
y_t^{a}=\operatorname{Attn}(q_t,K_{\le t},V_{\le t})
~~~

最简单的融合是：

~~~math
y_t=g_t\odot y_t^{a}+(1-g_t)\odot y_t^{s}
~~~

也可以把两条路径放在不同层、head 或 token 类型。gate 可以由输入、问题 token、长度、路由器或固定 schedule 决定。

## 43.3 小白直觉：高速公路和档案柜

SSM 像一条连续高速公路，适合快速扫过大量相似内容并维护状态；attention 像一个档案柜，当前问题可以拿着关键词去找具体文件。混合系统先用高速公路处理流，再在需要时打开档案柜。

如果所有请求都打开档案柜，成本上升；如果从不打开，关键证据可能找不到。混合设计的核心是定义“何时需要精确路径”。

## 43.4 层级混合

设模型有 L 层，global attention 层集合为 G，其余为 state 层：

~~~math
\mathcal{L}_{\mathrm{path}}=
\{p_l\mid p_l\in\{\mathrm{state},\mathrm{local},\mathrm{global}\}\}
~~~

粗略计算为：

~~~math
C\approx |G|C_{\mathrm{global}}+(L-|G|)C_{\mathrm{state}}
~~~

少量 global 层可以提供远程跳转，但信息必须先被 state 层处理到 global 层能读取的表示；放置位置影响质量。前层 global 有利于把证据广播，后层 global 有利于最终组合，具体要做消融。

## 43.5 Token/Head 级路由

更细粒度的设计让不同 token 或 head 选择路径：

~~~math
p_t=\operatorname{Router}(x_t)
~~~

token 级路由更灵活，却会产生不规则 batch 和动态 kernel；head 级路由较容易编译，但粒度较粗；固定周期 schedule 最容易 serving，却不能响应任务。

路由器还可能把安全、工具调用和用户数据分到不同路径。高风险任务不应只依赖不透明 gate，应有显式 policy gate 和 fallback。

## 43.6 一个 toy hybrid block

~~~python
import torch


def toy_hybrid(x):
    # x: [batch, seq, dim]
    state = torch.zeros_like(x[:, 0])
    state_outputs = []
    for step in range(x.size(1)):
        state = 0.95 * state + 0.05 * x[:, step]
        state_outputs.append(state)
    state_path = torch.stack(state_outputs, dim=1)

    scores = x @ x.transpose(-1, -2) / x.size(-1) ** 0.5
    causal = torch.tril(torch.ones(x.size(1), x.size(1), dtype=torch.bool))
    scores = scores.masked_fill(~causal, float("-inf"))
    attention_path = scores.softmax(-1) @ x
    gate = torch.sigmoid(x.mean(-1, keepdim=True))
    return gate * attention_path + (1.0 - gate) * state_path


print(toy_hybrid(torch.randn(2, 6, 8)).shape)
~~~

这个 demo 用指数平均代替 SSM、用 dense attention 代替 global path，只用于展示融合 shape。生产模型还要管理每层 state、KV、mask 和 kernel。

## 43.7 训练目标和路由塌缩

如果 gate 没有额外约束，模型可能始终依赖更容易优化的 attention 路径，导致 state 路径没有训练；也可能过度使用 state，精确任务下降。路径消融、gate entropy、分支梯度和 task-specific loss 可以诊断。

可以在训练中加入路径 dropout，让模型学会在一条路径缺失时工作；但 dropout 过强会降低整体质量。也可以进行 distillation，让 state 路径学习 attention 的重要摘要，代价是训练复杂。

## 43.8 Serving 状态和缓存

混合层同时带来显式 KV 和递归 state。请求状态要记录：

~~~text
model_revision
position_contract
attention_KV_blocks
state_tensors
committed_step
temporary_step
~~~

推测解码、抢占和恢复时，KV 和 state 必须一起提交或回滚。只回滚 KV 不回滚 state 会产生隐性历史污染。

## 43.9 适用场景和边界

长流日志、实时多模态、长对话和超长文档可以考虑 hybrid。对需要精确引用的文档，把 retrieval/global 层作为证据出口；对音频/视频，把 SSM/conv 处理密集时间轴，把 attention 用在跨模态对齐。

混合不一定适合短 prompt、单卡低并发或没有对应 kernel 的环境。若额外状态管理复杂度超过 KV 节省，dense Transformer 可能更稳。

## 43.10 常见失败模式

包括路径 schedule 与 checkpoint 不匹配、gate 饱和、state 和 KV position 不同、chunk 边界错误、global layer 过少、state 泄露、动态路由导致 batch 碎片和 kernel fallback。

排查用路径全开/全关、固定 gate、短序列 logits 对齐、长距 evidence probe、batch reorder、preemption/resume 和 profiler。每条路径都要有独立 reference。

## 43.11 机制与边界：混合的真正优化目标

混合架构要最大化任务质量并最小化两类状态和路径成本：

~~~math
\min_{\pi}\;C_{\mathrm{global}}(\pi)+C_{\mathrm{state}}(\pi)
\quad\text{s.t.}\quad Q_{\mathrm{evidence}}\ge Q_0
~~~

最优 global 层比例与 query 分布、证据稀疏度和上下文长度有关。研究时应报告 layer placement、gate 统计、状态 bytes、kernel 和证据任务，而不是只说“hybrid 更高效”。

## 43.12 面试追问、误区与练习

**问：为什么 attention+SSM 比纯 SSM 更适合通用 LLM？**

标准回答：SSM 负责低成本长流状态，attention 保留少量任意内容寻址和多证据组合能力；但混合增加 KV/state、路由和 serving 复杂度，需要真实 workload 验证。

**问：如何证明两条路径有分工？**

标准回答：做路径消融和 gate 分布，分别测局部、精确复制、冲突版本、长流、显存、kernel、p99 和恢复；如果关掉一条路径没有变化，说明它可能没有学到有效职责。

常见误区包括把 hybrid 当自动兼顾、忽略回滚双状态，以及只报告总吞吐。

练习：设计三种层 schedule，并用长距证据和流式分类任务比较 quality/cost Pareto。

## 43.13 混合层的梯度和信息泄漏

如果 attention 和 SSM 路径使用不同 mask，残差融合前必须确认它们看到的时间范围一致。流式任务中，SSM 只能看到已到达的 token，而 global attention 如果误用完整序列就会产生未来泄漏。

训练时做未来反事实：只改变未来 token，检查当前输出 logits；再分别关闭 attention 和 SSM，观察路径贡献。mask 正确性应在模块级和端到端都验证。

## 43.14 混合架构的路由验收条件

路由选择要兼顾质量和资源。可以设定 local/recursive 路径的最低 throughput 和 global 路径的最低 evidence recall；任一不达标就回退，而不是用总体平均分掩盖极端失败。

## 43.15 混合层的层间信息传递

SSM 层压缩历史，attention 层重新访问显式 token；两者交替时，attention 看到的是原始 KV、经过 SSM 的 hidden，还是两者并列，会决定实际能力。架构图若没有标出 cache 和 state，无法判断信息是否真的被保留。

可以对单层输出做路径遮挡和梯度归因，再测试多层 schedule。只看最终 attention map 不能证明 SSM 没有贡献，因为信息可能已经在前层写入 hidden。

## 43.16 训练和部署的共同验收条件

混合模型训练时要保证路径 dropout、mask、state reset 和 serving path 一致；部署时要保证 local/global cache、recursive state、prefix sharing 和 speculative rollback 一致。任何一个路径没有事务语义，都可能污染整个请求。

## 43.17 混合层的计算图和状态图必须同时画

计算图只能说明 hidden 如何流动，状态图还要说明 KV、recursive state、window 和 retrieval evidence 如何保存。一个 attention 层读取 SSM 输出，并不等于它能重新访问被 SSM 压缩掉的原始 token；反过来，SSM 也可能在 attention 之前把长期趋势写入 residual。

设计文档至少要画出 full prefill、chunk continuation、decode、snapshot restore 和 rollback 五条路径。每条路径都标明输入、可见范围、状态 owner、提交点和失败回退。

## 43.18 global attention 的最小配置问题

混合模型不一定需要很多 global 层，但 global 层的位置很重要。靠近输入的 global 层可以把远程证据写入后续 state，靠近输出的 global 层可以直接对齐问题和证据；两者的代价和错误模式不同。

可做 layer placement ablation：保持 global 层数量不变，只改变位置，测精确复制、冲突版本、引用支持、state bytes 和 p99。如果不同位置差异很大，说明“global 层比例”这个单一数字不足以描述架构。

## 43.19 路由失败时的可信回退

若混合路由器把需要精确引用的请求送入纯 state 路径，模型可能给出流畅但无证据的回答。系统可以在 query 分类不确定、verifier 失败或关键字段缺失时回读原文，或切换到显式 attention；高风险任务不应静默接受压缩路径。

回退策略要计入成本和 p99，并在评估中故意制造误路由。只有测到误路由率、回退成功率和单位成功成本，才能判断混合架构是否真的优于简单 dense baseline。

## 43.20 混合层的分工需要可观察

“一部分层用 attention，一部分层用 SSM”只是结构描述，不说明信息如何流动。需要知道哪些层负责局部模式、哪些层负责精确远程证据、哪些层负责状态压缩，以及它们的输入输出是否共享同一个 position、mask 和 residual scale。

可以通过层遮挡和路径替换实验回答分工：关闭 global attention 测引用和数字复制，关闭 state path 测长流和局部预测，交换层顺序测训练稳定，改变比例测质量—成本曲线。没有这些实验，混合结构的“互补”只是推测。

## 43.21 混合架构的训练与推理一致性

训练时全序列 attention 和并行 scan 可能并列计算，推理时却需要局部 KV、递归 state 和 chunk boundary。必须定义 state/cache 的同步点，处理 padding、reset、batch reorder、preemption 和 speculative rollback。层间 residual 不能把过期 state 当作当前 hidden。

部署时的 manifest 应包含层类型、层 schedule、cache/state schema、position 版本、dtype、kernel 和 fallback。只回滚权重而不回滚 schedule 或 cache schema，可能出现启动成功但输出语义错误。

## 43.22 路由预算与消融矩阵

混合架构可以按层固定比例，也可以按输入、任务或状态触发 global path。动态路由必须把额外 attention、检索或状态迁移成本放进预算，否则“只在难题上启用”可能在大量边界请求上造成 p99 峰值。

最小消融矩阵应包含：只 attention、只 SSM、固定混合、动态混合、混合加 retrieval。每条路径固定训练/解码预算，分别测局部语言、远距复制、精确引用、流式 reset、代码、工具 JSON、吞吐、峰值 state 和 fallback。路由器还要报告触发率与误触发率，不能只看混合模型最终平均分。

## 43.23 状态交接是混合架构的核心接口

attention 和 SSM 混合时，最容易被忽略的是状态交接。attention 可能持有按 token 分页的 KV，SSM 可能持有固定维度的递归 state；二者都要绑定 position、mask、dtype、model revision 和 request owner。切换层、chunk 或 worker 时，不能只传 hidden，而要说明两种历史表示是否已经同步到同一个逻辑位置。

一个实用的回归是：同一 prefix 分别经过完整路径、chunked 路径、单步 decode、preemption 后恢复和 batch reorder。比较每个路径的 logits、KV/state checksum、最终引用和工具参数。若 hidden 相同但 state schema 不同，下一轮 decode 仍可能分叉；若只有恢复路径失败，优先查 snapshot 边界，而不是重新训练模型。

混合架构的 fallback 也要从已提交状态开始。未提交的 speculative token、未验证的 global attention 输出和待执行工具参数不能写回长期 state。状态交接契约越明确，模型模块越容易独立演进。

## 43.24 混合架构的选择条件

SSM 和 attention 的基础分别见 S4（https://arxiv.org/abs/2111.00396）、Mamba（https://arxiv.org/abs/2312.00752）和 Transformer（https://arxiv.org/abs/1706.03762）。具体混合模型的层表和 gate 细节必须以公开模型资料为准。

混合架构现实，是因为不同历史信息需要不同访问方式。它的成功条件不是“两个模块都放进去”，而是分工清晰、状态一致、kernel 可用、质量检查通过。

## 43.25 混合层的最小状态图

对每个请求，显式 KV 和递归 state 应作为两个不同对象画出来。attention 路径保存 token/page 到 KV 的映射，SSM 路径保存当前递归 state；二者共享逻辑位置，却不共享同一种历史表示。chunk 结束时要记录 `position_end`、KV page、state checksum 和 owner worker。

如果只把两个模块的输出 hidden 相加，服务仍不知道下一次 decode 需要恢复什么。一个可审计的状态对象至少包括：

```text
request_id, model_revision, logical_position
kv_schema, state_schema, dtype
mask/position_revision, owner, checksum
```

这份 manifest 让 preemption、batch reorder、迁移和模型升级有明确边界。

## 43.26 混合比例的消融不能只看平均分

改变 attention/SSM 层比例会同时改变计算、状态和可访问路径。应至少比较全 attention、全 SSM、前段 attention/后段 SSM、交替层和动态路由，并记录每个方案在局部语法、远距数字、冲突版本、流式 reset、代码和工具任务上的结果。

若平均 loss 变化很小，但精确引用下降，说明被替换的 attention 层承担了寻址功能；若短上下文质量不变、长流显存下降，可能是合理的 Pareto 改善；若动态路由只在简单任务触发，却在难题上频繁误路由，应优先改路由器或 admission，而不是继续堆层。

## 43.27 混合架构的回退与成本

当状态路径无法满足当前证据精度时，回退到 attention、RAG 或更大模型。回退需要额外 prefill、KV、队列和网络成本，不能当成免费的异常分支。容量规划要按预计回退率计算：

```math
C_{\mathrm{expected}}
=C_{\mathrm{hybrid}}
 +p_{\mathrm{fallback}}C_{\mathrm{fallback}}.
```

`p_fallback` 应按任务和风险切片，而不是全局平均。高风险请求可以预留 dense capacity；低风险长流可以接受 state-only。所有回退都写入 trace，便于观察模型升级后是否出现系统性误路由。

混合架构值得采用的条件，是它在关键任务上保留证据能力、在目标硬件上减少真实成本，并且状态交接和回退可复现。没有这三点，结构图中的“互补”仍只是概念组合。
