# 第 64 章 DSpark：附加解码模块，不是新的基础模型

## 64.1 先把对象放对位置

在阅读某些模型生态的推理资料时，DSpark 可能以附加 speculative decoding module 的形式出现。理解它的第一步，是把基础模型、候选模块和 serving engine 分开：基础 checkpoint 负责最终分布和 target verify，附加模块负责提出或组织候选，engine 负责调度、cache、协议和 fallback。

如果把 DSpark 当成另一个基础模型，部署人员会误以为可以单独替换 target；如果把它当成一个只改配置的开关，又会忽略版本、量化、tokenizer 和 grammar 的契约。

## 64.2 一次请求的路径

```text
base target
  -> DSpark proposes candidates
  -> target verifies
  -> accepted prefix commits
  -> rejected branch rolls back
```

DSpark 的候选不能直接写入正式 KV。候选被拒绝时，临时 state、parser、streaming 和 usage 都必须回到 accepted prefix。它与 MTP、EAGLE 或独立 draft 的关系，要以具体实现说明为准。

## 64.3 artifact 契约

部署 manifest 至少要包含：

```text
base model revision
DSpark revision
tokenizer and chat template
quantization and KV dtype
parallel layout
runtime and kernel versions
grammar/tool support
sampling mode
```

启动时做文件 hash、config、hidden layout 和 golden logits 检查；运行时做 acceptance、p95、峰值显存和输出回放。能加载模块只能证明张量形状对得上，不代表候选分布和 target 兼容。

## 64.4 一个速度模型

设 baseline 需要 `L_target` 次 target step，DSpark 候选生成和验证分别有成本 `C_d`、`C_v`，平均每轮提交 `A+1` 个 token，调度固定成本为 `C_s`，可以用教学化公式估算：

```math
S_{\mathrm{speed}}
\approx\frac{C_{\mathrm{baseline}}L_{\mathrm{target}}}
{C_d L_d+C_v L_v+C_s}
```

`A` 通过接受长度影响 `L_v`，但它还受 workload、temperature、grammar、batch 和 target revision 影响。模块名称本身不提供速度保证。

## 64.5 worked example：模块失败时如何回退

一个请求已经由 DSpark 提出候选，target 在第二 token 拒绝。runtime 先提交 accepted prefix，清理 rejected branch，再按 target 普通路径生成后续 token。如果 DSpark kernel 在下一轮报错，新的请求可以关闭模块并走 baseline；正在执行的请求必须先完成 commit/rollback，不能直接把半个候选状态交给普通 decode。

如果模块发生 OOM，服务应按请求、batch 或模型池粒度关闭，并保留错误原因。静默切换可能改变 output event、usage 和延迟，客户端需要从 trace 知道实际走了哪条路径。

## 64.6 版本升级和回滚

升级前先在 shadow traffic 运行 DSpark，比较 acceptance length、target calls、TPOT、p95、峰值显存、输出质量和工具协议。manifest 中的 base revision 和 DSpark revision 必须成对发布。

回滚时不能只卸载模块。已有请求可能持有临时 candidate state，应等它们完成或取消清理；新请求才路由 baseline。回滚 runbook 还要包含 tokenizer/template、grammar parser 和 usage 的检查。

## 64.7 评估维度

不要只看平均 tokens/s。至少分桶记录：

| 维度 | 观察项 |
| --- | --- |
| 质量 | 输出等价性、工具正确率、结构化合法率 |
| 推理 | acceptance、target calls、draft latency |
| 系统 | TTFT、TPOT、p99、batch throughput、显存 |
| 协议 | stream event、usage、call id、fallback |
| 运维 | 启动失败、回滚时间、错误可定位性 |

低接受率、长输入和高并发往往比短文本 happy path 更能暴露模块不划算的地方。

## 64.8 常见失败

把附加模块当基础模型；base 和 module revision 错配；量化或 parallel layout 不匹配；被拒绝 token 污染 cache；工具 JSON 在候选阶段提交；fallback 没有清理状态；只看 target call 减少不看端到端；升级没有 shadow 和回滚。

## 64.9 面试回答与练习

回答“如何部署一个附加 speculative module”时，应把 base target、候选模块和 engine 分层，建立 manifest，启动做结构和 golden 检查，运行做 acceptance、协议和 SLO 监控，低收益或错误时显式 fallback，升级采用 shadow 和 revision 级回滚。

练习一：写出 DSpark 的 manifest 字段。

练习二：设计第二 token rejection 的 cache 和 stream 回放。

练习三：列出模块升级时必须同时回滚的三个对象。

### 64.9.1 附加模块的边界

DSpark 这类附加模块应被当作 target engine 的一个候选生成组件，而不是独立模型服务。它需要知道 target 的 tokenizer、position、hidden/cache 接口、sampling、grammar 和 stream 状态。模块加载成功只说明文件可以读取，不说明候选与 target 的概率语义一致。

### 64.9.2 manifest 和启动检查条件

启动 manifest 应保存 `target_revision`、`module_revision`、hidden shape、dtype、tokenizer/template、position scheme、max draft length、grammar support、hardware capability 和 engine version。启动时做 shape/checksum/版本检查，再用固定 golden prompt 比较 baseline 和 speculative 的 token、accepted prefix、usage 和 stream event。

### 64.9.3 线上动态开关

模块收益随任务变化。代码和结构化输出可能有较高 acceptance，开放写作和高温采样可能较低。runtime 可以按请求记录滚动 acceptance、head latency、target call reduction、p99 和错误率，在收益低或协议异常时关闭。动态关闭必须保留 baseline 的 cache 和状态语义，不能在半个 speculative step 中切换。

### 64.9.4 升级和回滚

模块、target、tokenizer、engine kernel 和 grammar backend 应作为 release bundle 管理。shadow 只执行候选和验证，不提交外部副作用；通过后再灰度。回滚时同时恢复所有绑定对象，并清理旧模块产生的临时 KV，避免旧 handle 被新 engine 误用。

## 64.10 附加模块的上线条件

DSpark 类附加模块应先检查 model revision、hidden shape、tokenizer、dtype、position、sampling 和 stream protocol。启动时做 capability probe，运行时记录接受率、回退、额外显存和 kernel time。

故障时必须保证 target-only 路径的输出正确，不能因为附加模块异常而丢失已经提交的 token 或重复工具动作。

## 64.11 附加解码模块的边界

DSpark 这类附加模块应被视为特定模型/引擎的解码组件，而不是新的基础模型架构。它可能改变候选生成、验证、采样或 kernel，但不能仅凭模块名称推断训练目标、层结构和通用能力。

集成时要绑定 target model、tokenizer、模板、position、dtype、权重格式和 runtime 版本。模块加载成功只证明接口匹配，不证明候选在任务上有收益。

## 64.12 组件化 speculative 的状态事务

附加模块产生候选后，target engine 负责验证和提交。draft page、target page、候选 logits、position 和 state snapshot 应有明确 owner；拒绝候选时，所有未提交对象都要释放或回滚。

如果组件无法提供回滚或错误状态，系统应关闭 speculative 而不是继续输出。故障时普通 decode 是功能回退，不应变成未审计的另一种模型行为。

## 64.13 版本和性能回归

回归集包含 acceptance、token 一致性、stop token、结构化输出、长 prompt、短 prompt、低并发、高并发、取消、preemption 和恢复。性能记录候选生成时间、验证时间、page 分配、回滚和 p99。

发布资料如果没有公开完整实现，只能把可观察接口和实验结果写入正文，未披露的内部算法保持条件化表述。

## 64.14 附加模块与基础模型的证据分层

看到一个名为 DSpark 的模块时，先把证据分成四层。第一层是可观察接口，例如是否需要 target hidden、如何加载 artifact、输出哪些候选；第二层是 runtime 行为，例如接受长度、验证时间和回退；第三层是训练方法，例如是否有额外 loss、蒸馏或 feature 预测；第四层才是基础模型架构。前两层可由接口和实验确认，后两层必须有论文、技术报告或源码支持。

这种分层很重要，因为“附加解码模块”通常只改变推理路径。它可能减少 target 调用，却不改变模型的参数总量、预训练知识或基础 Transformer block。若把模块宣传写成模型能力升级，会混淆性能加速、候选质量和模型本身的能力。

## 64.15 模块加载、健康检查与动态关闭

启动检查不应只验证文件存在。一个完整的 probe 至少检查 target revision、hidden shape、dtype、tokenizer/template、position scheme、sampling、grammar、设备能力和 engine ABI。然后用三个 golden request：普通文本、结构化输出和工具调用，比较 baseline 与附加模块的 token、stop reason、usage、stream event 和最终状态。

运行时要有明确的关闭条件：

~~~math
\mathrm{disable}
=
 (A_t<A_{\min})
\lor(p99_t>p99_{\max})
\lor(\mathrm{protocol\_error}_t>0)
\lor(\mathrm{memory\_headroom}_t<h_{\min}).
~~~

关闭动作应在一个 speculative round 提交或回滚之后发生，不能把正在使用的临时 page 半途交给普通 decode。关闭后要保持 target-only 的 cache 和 stream 语义，便于用户无感回退。

## 64.16 模块失败的故障注入

最小故障矩阵包括：模块文件损坏、checksum 不匹配、hidden shape 错、candidate kernel 超时、target verify 超时、临时 KV 不足、grammar backend 失败、请求取消和 worker 重启。每个故障都要回答四件事：已经提交了什么、哪些状态必须回滚、客户端能看到什么、是否可以安全重试。

工具调用要单独测试。若候选中已经出现 tool call，但 target 尚未确认，不得执行外部副作用；若 target 已确认而网络响应丢失，重试必须依靠 idempotency key 查询执行状态，而不是再次调用。附加模块的失败不应成为重复扣款、重复写文件或重复发送消息的原因。

## 64.17 性能报告的正确分母

DSpark 类模块的性能报告至少有三种分母：每个 target verify round 的平均推进 token、每个请求的端到端 latency、每个成功任务的成本。前者适合诊断算法，后两者才适合产品决策。

可以把一个请求的粗略速度写成：

~~~math
T_{\mathrm{request}}
=T_{\mathrm{prefill}}
 +R\,(T_{\mathrm{candidate}}+T_{\mathrm{verify}}+T_{\mathrm{commit}})
 +T_{\mathrm{network}},
~~~

其中 R 是实际轮数。若模块减少 R，但提高了每轮显存压力并触发更多 preemption，端到端时间反而可能变长。报告必须同时给出 workload、batch、硬件、并发、采样、p50/p99、峰值显存和 fallback 比例。

## 64.18 附加模块的状态机与热关闭

模块生命周期可以拆成 `discovered`、`loaded`、`probed`、`enabled`、`draining`、`disabled` 和 `failed`。`loaded` 只表示文件和 ABI 可读，`probed` 还要通过 target hidden、tokenizer、grammar、stream 和 golden request；只有 `enabled` 才允许候选进入正式请求。

热关闭时，正在进行的 speculative round 要先完成提交或回滚，之后释放临时 state，再让后续 iteration 使用 target-only。若模块在候选已生成、target 尚未验证时崩溃，必须丢弃候选并从 committed state 恢复。这个顺序比“把一个布尔开关改成 false”更重要，因为状态已经分配在 GPU 和 stream 协议中。

## 64.19 artifact manifest 是上线前的身份契约

附加解码模块至少要绑定 base model revision、head/module revision、hidden shape、dtype、tokenizer/template、grammar backend、engine ABI、硬件能力和 license。可以用 manifest 表示：

```text
module_id, base_model_revision, module_revision
hidden_schema, dtype, tokenizer_revision
engine_abi, supported_sampling, grammar_support
checksum, owner, created_at, rollback_target
```

加载文件成功只说明字节可读，不说明它和当前模型语义兼容。preflight 应运行 shape、token、stream、grammar 和 golden request 检查，任何一项失败都应保持 target-only。

## 64.20 DSpark 的启停是状态机

模块生命周期可以拆成 `discovered`、`loaded`、`probed`、`enabled`、`draining`、`disabled` 和 `failed`。只有 `probed` 通过目标模型、tokenizer、grammar、stream、显存和版本检查后，才能进入 `enabled`。热关闭时，正在进行的 speculative round 先提交或回滚，释放临时 state，再让后续 iteration 使用 target-only。

状态机的价值在于定义崩溃后的责任：候选已生成但未验证时丢弃候选；target 已确认但响应丢失时查询 action/state；已提交的 token 和外部副作用不因模块关闭而重做。一个布尔开关无法表达这些状态。

## 64.21 模块健康检查不能只测加载

健康检查要分层：文件 checksum 和 ABI、hidden shape、tokenizer/template、单轮候选、接受/拒绝回滚、结构化输出、流式事件、取消、工具调用和资源回收。每层都应有失败码和回退动作。

特别要把工具动作放在 target-confirmed 之后。候选中出现 tool call 不代表可以执行；如果模块失败造成响应未知，使用 idempotency key 查询工具状态，而不是再次调用。模块优化不能改变 executor 的安全边界。

## 64.22 性能和成本的分母

DSpark 类模块的报告至少有三种分母：每个 verify round 的推进 token、每个请求的端到端 latency、每个成功任务的成本。设候选轮数为 `R`，每轮候选、验证、提交和网络成本分别为 `T_c`、`T_v`、`T_m`、`T_n`：

~~~math
T_{\mathrm{request}}
=T_{\mathrm{prefill}}
 +R(T_c+T_v+T_m)+T_n.
~~~

减少 `R` 不一定降低总时间；如果候选显存导致 preemption 或验证队列变长，p99 可能变差。报告必须绑定 workload、batch、硬件、并发、采样、p50/p99、峰值显存和 fallback 比例。

## 64.23 灰度、热关闭和回滚

先在 shadow 中计算候选，但不发送正式 stream、不执行工具，再用低流量 canary 比较 target-only 和模块路径。灰度分桶要包括普通文本、代码、JSON、工具、长上下文和不同量化。

关闭条件包括 acceptance 过低、p99 超预算、protocol error、memory headroom 不足、cache rollback 失败和安全状态未知。关闭动作发生在一个 round 提交或回滚后，正式 KV 和已提交 stream 保留，临时候选释放。回滚要恢复 manifest、路由、cache schema 和观测标签。

## 64.24 资料范围与兼容性观察

DSpark 这类名称的稳定学习对象是附加推测模块的 artifact、生命周期、状态事务、性能分母和安全回退，而不是一个可以从名称推断的基础模型架构。未公开的候选头、feature 层、训练目标和 kernel 细节不得写成事实。

最终评估要把 target-only、附加模块和失败回退放在同一套 golden replay、压力测试和故障演练中比较。只有在协议正确、质量不退化、SLO 达标和单位成功成本下降时，附加模块才有上线价值；具体部署细节以对应官方模型卡、引擎文档和可复现实验为准。

## 64.25 附加模块的 ABI 与语义兼容

DSpark 类附加模块的兼容性至少有两层。ABI 兼容表示文件、shape、dtype、kernel 和 engine 可以加载；语义兼容表示它与当前 base model、tokenizer、position、sampling、grammar 和输出协议产生相同的候选语义。前者通过加载检查，后者必须用 golden request 和 reject/accept replay 验证。

manifest 应明确 base revision、module revision、hidden schema、支持的采样和 grammar、engine ABI、checksum、owner、license 和 rollback target。只检查文件能否读入，不能证明上线安全。

## 64.26 热关闭中的状态事务

热关闭不能直接把 `enabled=false` 写进配置。若请求正处于 candidate、verify、commit 或 tool proposal 阶段，系统要先确定哪部分已提交、哪部分可丢弃，再释放临时 state。候选已生成但未验证时丢弃；已提交 token 保留；工具状态未知时使用幂等查询。

这个顺序还要和 scheduler、stream、KV allocator、trace 一起实现。关闭期间的新请求可以直接 target-only，正在进行的请求可以 drain 或在安全边界回退。灰度与回滚测试应检查客户端是否收到重复/缺失 event、正式 KV 是否污染和模块是否释放全部 page。

## 64.27 失败归因和资源预算

附加模块失败可能来自 artifact、候选质量、target verify、grammar、临时显存、调度、网络或工具协议。trace 应为每轮记录 candidate length、accepted length、verify time、temporary bytes、rollback、fallback reason 和最终质量。

容量规划同时算正式和临时状态：

```math
M_{\mathrm{total}}
=M_{\mathrm{weights}}+M_{\mathrm{committed}}
 +M_{\mathrm{temporary}}+M_{\mathrm{workspace}}+M_{\mathrm{slack}}.
```

如果只按平均 acceptance 预估，会在长请求或低 acceptance bucket 上 OOM；如果只按平均显存，会掩盖模块冷启动、worker 迁移和回退重算的尾延迟。

## 64.28 附加推测模块的接口边界

附加模块至少要说明输入是 hidden、logits 还是 token，输出是候选 token、feature 还是验证辅助信息，以及它是否需要独立权重、额外显存和特定 kernel。若文档没有公开内部结构，只能按接口行为描述，不能把名称扩写成确定架构。

## 64.29 附加模块的资源账

一次 speculative round 的成本包括候选生成、target verify、临时 KV、回滚、调度和额外事件。可用粗略模型：

```math
C_{\mathrm{round}}
=C_{\mathrm{draft}}+C_{\mathrm{verify}}
+C_{\mathrm{temp\ cache}}+C_{\mathrm{rollback}}.
```

只有平均提交 token 足以摊薄这些成本，TPOT 才会下降。对低 acceptance 或短输出，请求级关闭可能比全局启用更好。

## 64.30 模块升级和故障关闭

模块版本、base model、tokenizer、template、grammar、量化和 engine ABI 必须一起做兼容检查。热关闭应等当前事务提交或回滚后进行，清理临时 page，并在 trace 中记录 disable reason；不能中途把候选当成正式 token。

## 64.31 附加模块的最小适配器

适配器应明确 draft 输入、候选格式、target 验证接口、临时状态、提交回调、回滚回调和 metrics。业务层只消费 committed event，不能直接读模块内部的候选 tensor。这样模块可以替换，且失败时能回到 target-only。

## 64.32 模块化推测的评测分层

先测候选质量和接受长度，再测 verify kernel 和 cache 成本，最后做端到端任务、协议和安全评测。任何一层失败都要能定位：候选差、验证慢、回滚错、grammar 不兼容还是 scheduler 不公平。

## 64.33 何时不启用附加模块

短输出、低 acceptance、高优先级低延迟、工具 JSON 严格约束、临时显存不足或 target/模块版本不匹配时，附加模块可能不划算。服务应有明确关闭条件，并把关闭比例纳入容量和成本报告。

## 64.34 附加模块的产品边界

DSpark 这类名称的稳定学习点是如何把一个附加推测模块变成可发现、可验证、可灰度、可关闭和可回滚的 artifact。未公开的 head 结构、训练目标和 kernel 不能从名称补齐。它的产品价值必须在 target-only baseline、模块路径和失败回退三者的同条件报告中成立。

尤其是工具 Agent，候选加速不能缩短 policy 和 executor 的确认链。任何平均 speedup 都不能抵消未授权动作、重复副作用、stream 回滚失败或数据隔离错误。模块只有在质量、协议、安全、SLO 和单位成功成本都通过验收后，才值得进入默认路由。
