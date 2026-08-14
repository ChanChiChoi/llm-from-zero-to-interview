# 第四十七章：MoE + SSM + Attention 的未来组合可能

## 47.1 三种技术优化不同预算

MoE 扩大参数容量并让每个 token 只激活部分专家；SSM 降低长流历史状态成本；attention 提供精确的内容寻址和全局组合。三者的组合不是简单相加，而是把容量、历史和交互三个预算分开。

一个可能的 block 是：共享局部/SSM 路径处理所有 token，MoE FFN 提供条件容量，少量 global attention 负责证据对齐。具体模型是否采用这种组合，必须以公开模型卡/技术报告为准；本章讨论架构设计空间。

## 47.2 三类成本的抽象

可以把每层计算写成：

~~~math
C_l=C_{\mathrm{state}}+
C_{\mathrm{MoE,active}}+
C_{\mathrm{global}}
~~~

其中 global 层不一定每层存在，MoE active 由 router 决定，state 更新可以固定大小。内存则包括：

~~~math
M=M_{\mathrm{weights,total}}+
M_{\mathrm{KV,global}}+
M_{\mathrm{state}}+
M_{\mathrm{expert/comm}}
~~~

MoE 可能减少 active compute，却不自动减少总权重；SSM 可能减少 state 随长度增长，却不自动消除少量 global KV。

## 47.3 小白直觉：工厂、流水线和档案柜

MoE 像很多专业工厂，当前零件只送到少数工厂；SSM 像流水线上的状态记录，持续处理大量输入；attention 像档案柜，遇到关键问题时精确查资料。

组合成功的前提是工厂不要拥堵、流水线不要丢关键状态、档案柜不要被所有 token 同时打开。每一类资源都需要自己的监控和预算。

## 47.4 路由顺序的选择

可以先 state 再 MoE：

~~~math
h'_t=\operatorname{SSM}(x_t),\qquad
y_t=\operatorname{MoE}(h'_t)
~~~

也可以先 MoE 再 global attention：

~~~math
h'_t=\operatorname{MoE}(x_t),\qquad
y_t=\operatorname{Attn}(h'_t)
~~~

顺序影响 router 输入、专家 specialization 和 global 证据形态。若 state 已压缩历史，MoE 看到的是摘要；若 attention 在 MoE 后，能对齐专家处理后的表示。需要做 layer order ablation。

## 47.5 共享专家和状态路径

MoE 可以保留 shared FFN 处理公共知识，再让 routed experts 处理 token/domain 特化；SSM 可以作为所有 token 的低成本公共路径。这样可减少路由完全失效时的质量崩溃。

代价是 shared path 可能让 active compute 增加，state 和 expert 输出的尺度也要校准。共享专家不是免费 fallback，应记录其 token 量和梯度。

## 47.6 一个资源估算器

~~~python
def combined_cost(tokens, global_layers, total_layers,
                  active_experts, expert_cost,
                  state_cost, global_cost, network_cost):
    state = tokens * total_layers * state_cost
    moe = tokens * active_experts * expert_cost
    global_part = tokens * global_layers * global_cost
    network = tokens * active_experts * network_cost
    return {
        "state": round(state, 2),
        "moe_active": round(moe, 2),
        "global": round(global_part, 2),
        "expert_network": round(network, 2),
        "total": round(state + moe + global_part + network, 2),
    }


print(combined_cost(
    tokens=100_000, global_layers=4, total_layers=32,
    active_experts=2, expert_cost=0.8, state_cost=0.2,
    global_cost=2.0, network_cost=0.15,
))
~~~

这个模型只用于容量评审，不能预测真实 latency；它把被忽略的 expert network 单独列出，避免只看 active expert GEMM。

## 47.7 训练难点

训练要同时稳定三种选择：state gate、MoE router 和 global attention。state 可能过度压缩，router 可能专家塌缩，global 层可能成为所有任务的 shortcut。辅助损失、路径 dropout、专家均衡和长程 probe 都要纳入训练监控。

数据 mixture 还会影响专家分工。若代码、多语言和推理数据比例变化，expert routing 可能重新组织；checkpoint 续训必须记录数据和 router 版本。

## 47.8 Serving 难点

一个请求可能同时需要：global KV block、state tensor、专家权重和 all-to-all buffer。scheduler 要决定请求在哪些 worker 上执行、状态如何迁移、热门专家是否复制、global 层是否分池。

推测解码/抢占时，要对 KV、state、router 临时选择和 expert output 一起提交或回滚。任何一个状态不同步都会导致不可复现的生成。

## 47.9 适用场景

大规模多领域服务、长周期 Agent、长日志流、多模态视频和高并发长上下文是组合路线的候选。短问答、低并发边缘设备和严格确定性服务可能更偏 dense 或较简单 hybrid。

组合架构的主要价值可能是单位成功成本，而不是所有 benchmark 都领先。要先明确业务失败的代价。

## 47.10 常见失败模式

包括 total/active 参数混淆、global 层 KV 没计入、state 与 expert router 分布不一致、热门专家拖慢 p99、跨节点 all-to-all、量化后 gate/route 漂移、请求恢复只保存一类状态和复杂度叙事超过公开证据。

排查要按层记录路径、state bytes、expert counts、KV blocks、通信、kernel 和恢复 checksum；使用 dense/state-only/MoE-only/global-only ablation。

## 47.11 机制与边界：组合的真正风险是状态耦合

三种技术的内部状态会相互影响：state 改变 router 输入，router 改变 token 表示和 global 需求，global attention 又可能改变后续 state。耦合让表达力更强，也让故障定位更难。

可以引入显式 contract：每个路径声明输入/输出 shape、提交点、版本、dtype、位置和回滚语义。没有 contract 的组合只能在单次 forward demo 中成立。

## 47.12 面试追问、误区与练习

**问：MoE、SSM、attention 组合时谁解决什么问题？**

标准回答：MoE 主要扩展条件容量，SSM 主要降低长流状态成本，attention 主要提供精确全局内容寻址；组合收益取决于路由、通信、KV/state 生命周期和任务分工。

**问：为什么 active FLOPs 低仍可能服务很慢？**

标准回答：专家通信、路由、热门专家排队、global KV、state 更新和 kernel fallback 可能主导端到端 latency。

常见误区包括把三个名字写成一条宣传句、只算 active expert、忽略状态提交和把未公开模型结构当事实。

练习：为多租户长文档服务设计 MoE+SSM+attention 的状态 schema、worker 拓扑、故障回滚和 quality/cost gate。

## 47.13 三种稀疏的不同含义

MoE 稀疏的是 FFN 专家计算，SSM/递归路径稀疏的是历史访问，attention 稀疏可能是可见边或 token/block。它们的资源收益和错误模式不同，不能把三个模块相加后称为“整体稀疏”。

评估要分别记录专家负载、state bytes、attention block、通信和精确证据 recall。组合模型的总成本大致为：

~~~math
C_{\mathrm{total}}
=C_{\mathrm{router}}
+C_{\mathrm{state}}
+C_{\mathrm{attention}}
+C_{\mathrm{communication}}
~~~

## 47.14 组合路线的回退和调度

当专家热点、state 不稳定或 global evidence 缺失时，系统需要可观测回退：减少并发、改用 dense layer、增加显式 attention、检索原文或转人工。调度器必须知道每条路径的资源，而不是只按 token 数排队。

## 47.15 路由器要感知任务风险

MoE router 选择专家，路径 router 选择 attention/SSM，产品 router 选择模型和工具。三层路由叠加时，低概率的组合路径可能成为质量或安全风险。高风险请求要限制可选路径，并保留显式 verifier。

## 47.16 组合模型的容量规划

规划时必须把静态和动态资源分开。静态部分包括 dense 权重、专家权重和 kernel workspace；动态部分包括专家 dispatch、global KV、recursive state、工具结果和 fallback 的峰值。可以用一个请求级账本表示：

```math
M_{\mathrm{peak}}
=M_{\mathrm{dense}}
+M_{\mathrm{expert\_resident}}
+M_{\mathrm{KV,global}}
+M_{\mathrm{state}}
+M_{\mathrm{dispatch}}
+M_{\mathrm{workspace}}
```

假设平均每个 token 激活两个专家，平均并发 1,000 个请求，但其中 5% 的请求属于长文档任务，并且 30% 的 token 被路由到同一个热门专家。平均 active FLOPs 可能看起来很低，热门专家的队列和 all-to-all buffer 却决定 P99。容量规划至少要用短/长请求混合、专家负载分位数和失败回退场景，而不能只用平均 token。

实际部署可以先按三种工作负载做上界：普通短问答、长上下文回读、需要工具的长周期任务。分别记录 `expert_load_p99`、`global_kv_bytes`、`state_bytes`、跨设备通信、fallback rate 和单位成功成本。若某条路径超预算，回退到 dense 或显式检索时也要把它的权重和延迟算进账本，否则系统会在压力下出现“模型本身没超，但回退把 GPU 吃满”的二次故障。

## 47.17 三种路由叠加后的尾延迟

MoE router、attention/SSM path router 和服务层 model router 可能同时作决定。每层路由都带来候选选择、状态准备和失败回退；串联之后，平均延迟看起来正常，p99 却可能被少数热门专家、global evidence 或跨设备 state 迁移拉高。

可以把一次请求的尾延迟近似分成：

~~~math
T_{\mathrm{tail}}
=T_{\mathrm{queue}}+T_{\mathrm{expert}}
+T_{\mathrm{state}}+T_{\mathrm{global}}
+T_{\mathrm{communication}}+T_{\mathrm{fallback}}
~~~

压测时要记录 expert load skew、state update time、attention KV bytes 和 fallback rate。只报告 active parameters 不能解释这些尾部成本。

## 47.18 组合训练中的目标冲突

MoE 负载均衡希望专家分布更均匀，SSM gate 可能希望不同 token 使用不同更新强度，attention 路径又可能只在少量证据 token 上激活。若把所有 gate 都加同一种均匀正则，可能压平真正有意义的稀疏性；完全不约束，又可能出现专家热点或路径塌缩。

训练报告应分别列出 load-balance loss、路径熵、state norm、global attention 使用率和任务质量。正则项的目标不是让每种路由看起来均匀，而是在容量、稳定性和任务所需的信息访问之间找到平衡。

## 47.19 组合模型的故障回退

当某个专家不可用时，系统可以选择备用专家、dense fallback 或拒绝请求；当 state 校验失败时，应重算 prefix 或走显式 attention；当 global evidence 不足时，应检索原文或降低自动化等级。回退策略必须在训练和评估中出现，否则线上回退会变成未测试的新模型。

高风险任务还要限制路由自由度，保留可解释的路径和 verifier。模型结构越复杂，越不能把安全验收条件只放在最终文本上。

## 47.20 三类模块组合时 router 不是唯一控制器

MoE router 决定 token 进入哪些 expert，attention/SSM schedule 决定信息如何跨 token 流动。二者叠加后，系统还要控制 expert capacity、state ownership、global evidence path、通信和 fallback。一个 token 被路由到 SSM expert，不代表它可以访问同一请求的所有历史 state；不同 expert 的 state 不能无条件混用。

设计时先明确路由粒度：token、sequence、layer 还是 request；再定义 capacity overflow、drop、reroute 和恢复。对高风险任务，global attention 或 retrieval 可能是硬性条件，不能因为 router 认为某个 expert 便宜就跳过。

## 47.21 组合架构的容量模型

总成本至少包括 dense 主干、expert 激活、expert 间通信、attention KV、SSM state 和临时 workspace：

```math
C_{\mathrm{total}}
=C_{\mathrm{dense}}+C_{\mathrm{expert}}
 +C_{\mathrm{comm}}+C_{\mathrm{KV}}
 +C_{\mathrm{state}}+C_{\mathrm{workspace}}
```

active parameters 降低单 token 的部分计算，不会自动降低所有显存；expert 权重可能仍驻留，KV/state 还随请求增长。容量规划要按 expert load、长短请求混合、通信拓扑和 p99 测量。

## 47.22 组合架构的训练与回退

训练中要检查 router load balance、expert collapse、state reset、attention mask 和长上下文任务；服务中要检查动态 batch、preemption、跨节点通信、cache key、量化和 unsupported fallback。新组合结构上线前保存完整 manifest：expert 配置、layer schedule、position、KV/state schema、router revision 和 kernel。

当某个 expert、kernel 或 global path 失败时，回退应保持输出协议、权限和状态语义。可以降低并发、切换 dense expert 或拒绝任务，但不能静默把未验证的 state 交给另一个路径。

## 47.23 路由粒度决定组合成本

MoE、SSM 和 attention 的组合首先要决定路由粒度。token-level 路由最灵活，却可能产生大量 dispatch 和 state owner；sequence-level 路由更容易管理，但可能把一个混合任务送进不合适的路径；layer-level 路由简单稳定，却无法按难度动态分配精确证据预算。

可以做一个三档消融：固定 dense/SSM/attention，按 token 动态路由，按 request 风险路由。每档固定质量、token budget 和硬件，记录 expert histogram、通信、KV/state bytes、路由延迟、误路由、fallback、引用支持和 p99。若 token-level 的质量增益被通信和状态迁移抵消，sequence-level 可能是更好的工程折中。

容量规划还必须同时考虑 total parameters、active parameters、专家权重驻留、KV、state 和 workspace。一个“每 token 只激活两个 expert”的说法不能推导整机显存或单位成功成本。

## 47.24 组合模型的选择条件

MoE、SSM 和 attention 的基础资料可参考 GShard（https://arxiv.org/abs/2006.16668）、Mamba（https://arxiv.org/abs/2312.00752）和 Transformer（https://arxiv.org/abs/1706.03762）。三者组合的具体模型必须以公开实现为准。

未来组合的关键不是把所有新模块放进模型，而是让容量、历史状态和精确交互各有明确职责，并能被训练、服务和审计。

## 47.25 组合模型的状态所有权

MoE expert、SSM state 和 attention KV 的所有权可能不同。token 路由到 expert 后，状态是按 token、sequence 还是 request 保留；expert 被迁移时 state 是否随之迁移；attention 分支能否读取其他 expert 产生的状态；这些都必须明确，否则 batch reorder 和 preemption 时会出现隐性串线。

每个 state handle 至少绑定 request、model revision、expert/layer、logical position、dtype、schema 和 owner。未提交的候选状态不能写入正式 state；expert overflow、kernel error 和 worker 故障应从最近 committed point 回退。

## 47.26 组合路由的错误归因

组合模型的一次错误可能来自 expert 选择、attention mask、SSM state、通信顺序、量化或最终 decoder。要在 trace 中记录 router decision、expert load、state checksum、KV page、通信耗时、fallback 和证据任务结果。

如果关键 token 被送到不具备所需历史的 expert，属于路由/状态路径问题；如果证据可达但输出错，属于表示/训练/读取问题；如果只在跨节点发生，属于通信或布局问题。把这些错误都归因于“模型能力”会浪费调试预算。

## 47.27 组合架构的匹配实验

至少比较 dense attention+FFN、attention+MoE、SSM+MoE、attention+SSM、三者联合和带 retrieval 的回退路径。固定 token、参数或 active compute 口径，记录局部任务、远距复制、多证据、专家热点、结构化工具、通信、state/KV、p99 和单位成功成本。

如果联合模型只在平均语言 loss 上提升，却在高风险引用或工具任务下降，不能用更高 active efficiency 放行。组合复杂度只有在其负责的失败类型和资源目标上形成可解释收益时才值得维护。
