# 第 63 章 NEXTN：多 token 候选路线如何进入 Serving 契约

## 63.1 名字背后的问题

在推理引擎和模型生态中，NEXTN 可能被用来描述“下一批 token 候选”或某种多 token 预测接口。它所面对的基本瓶颈与 MTP 相同：如果 target 每次只能提交一个 token，decode 的串行开销就很难降低；如果 runtime 能够批量验证候选，就有机会减少 target step。

但 NEXTN 不是一个脱离模型和引擎的统一算法名。它可能代表模型 head、附加模块、候选接口或某个 runtime 的配置。阅读资料时首先要确认：这个名称指的是训练组件、权重 artifact，还是 serving API。

## 63.2 从候选到提交

给定上下文 `X`，候选序列为 `D_{1:K}`，target 根据 position、mask、grammar 和 sampling 规则验证：

```math
Y^{\star}=\mathrm{Verify}_{\mathrm{target}}(X,D_{1:K},\mathrm{mask},\mathrm{grammar})
```

验证结果不是简单的布尔值。runtime 需要得到 accepted prefix、拒绝位置、target fallback token 和新的 cache/state。只有 accepted prefix 才能推进正式输出。

## 63.3 NEXTN 与 MTP 的边界

MTP 更强调训练目标或模型内部的未来 token 预测头；NEXTN 可能更强调 serving 侧如何取得并消费多个候选。一个 MTP head 可以成为 NEXTN 的候选来源，但两者不是同义词；某个名为 NEXTN 的接口也可能兼容独立 draft 或其他附加 head。

集成时不要根据名称猜测兼容性，而要在 manifest 中记录：checkpoint 是否包含 head、head 数和窗口、tokenizer、position 契约、grammar、sampling、streaming、batch 支持和 fallback。

## 63.4 一个 JSON 工具调用例子

模型要输出：

```json
{"name":"search","arguments":{"query":"..."}}
```

NEXTN 可能提出 `name` 和部分参数 token，target 接受函数名却在参数值中拒绝。此时 runtime 不能把半个 JSON 当成可执行工具调用，也不能让 parser 状态永久包含被拒绝 token。它应回滚到拒绝位置，按 target 和 grammar 重新生成，直到整个 JSON 合法，再发出一个原子的 tool call event。

这个例子说明推测解码的难点不止是 logits。parser、streaming、usage 和 executor 都必须理解“候选”和“已提交”的区别。

## 63.5 capability negotiation

请求进入 runtime 时先检查：

```text
model head -> tokenizer/template -> position/cache
    -> sampling -> grammar -> streaming -> batch -> fallback
```

任何关键能力不满足，都应显式关闭 NEXTN 或路由普通 decode。能力结果写进 trace，例如 `speculative=native`、`grammar=adapted`、`streaming=unsupported`。用户不能只看到一个模糊的 `speculative=true`。

## 63.6 调度器的候选空间

候选 token 需要临时内存，验证结果需要临时 logits 或状态，已提交 token 才进入正式 KV。scheduler 必须为三者分别记账。batch 中低接受率请求如果继续申请大窗口，可能浪费显存并拖慢其他请求；可以按历史接受长度自适应窗口。

对于 grammar 和 tool call，调度器还要保证一个请求的 parser state 不被另一个请求复用，拒绝分支释放临时 block，取消时删除未提交 artifact。

## 63.7 端到端收益

NEXTN 的速度收益可粗略写为：

```math
S\approx\frac{C_{\mathrm{baseline}}N}{C_{\mathrm{draft}}N_d+C_{\mathrm{verify}}N_v+C_{\mathrm{sched}}}
```

其中 `N` 是普通 target step 数，`N_v` 是验证轮数。接受长度、候选头成本、target kernel 和 batch 共同决定结果。减少 target calls 只是中间指标，不等于降低 p99 或单位成功成本。

## 63.8 正确性回放

准备一组 golden requests，覆盖普通文本、代码、JSON、工具成功、工具失败、高 temperature、短输出、长输出、取消和 batch。重点构造拒绝发生在第一个、中间和最后一个候选位置的样本。

回放时比较正式 token、KV 长度、grammar 状态、stream event、usage 和最终工具动作。happy path 全部接受最容易通过，真正能发现 bug 的是部分接受和回滚。

## 63.9 常见误区

把 NEXTN 当作基础模型；把候选 token 直接计入输出；把 MTP、EAGLE 和 NEXTN 名称混为一个算法；忽略 grammar 和工具原子性；低接受率仍强行打开；模型能加载就声称协议兼容；静默降级而不记录。

## 63.10 面试回答与练习

回答“NEXTN 是什么”时，应先说明公开资料中的具体语境，再把它放入“候选生成—target 验证—accepted prefix 提交”的 speculative pipeline。关键工程问题是 position/cache、grammar、streaming、usage、scheduler 和 fallback，而不是名称本身。

练习一：设计一张 NEXTN capability matrix，区分 native、adapted 和 unsupported。

练习二：画出 JSON tool call 在第二个参数 token 被拒绝时的状态变化。

练习三：解释为什么仅减少 target calls 不能证明 tokens/s 一定提升。

### 63.10.1 候选生成的接口

多 token 候选模块应返回 token、位置、候选概率或校验所需的状态，而不是只返回一段字符串：

```text
candidates = {
  base_position,
  tokens: [[t1, t2, ...], ...],
  scores,
  grammar_state,
  model_revision,
  cache_handle
}
```

target 验证后只提交连续 accepted prefix。候选内部的分支、被拒绝 token 和临时 cache 都不能直接暴露给下一个请求。

### 63.10.2 NEXTN 的任务边界

名称相同的模块可能在不同生态中指多 token head、候选插件或某个 engine 的 next-token 优化。分析 NEXTN 时应先确认来源和版本，再讨论共通 pipeline。没有公开定义时，不能把 NEXTN 展开成一个统一算法或声称所有模型都支持。

### 63.10.3 grammar 与流式输出

结构化输出的 grammar state 必须随 accepted prefix 更新；draft 候选被拒绝时恢复到上一个 grammar state。若服务端先流出候选，再发现 JSON 语法不合法，客户端会收到不可修复的半结构化结果。正确做法是把候选视为内部临时结果，确认后再发出 commit event。

### 63.10.4 scheduler 的取舍

候选模块适合 decode 受限、输出分布相对可预测的 workload。长 prefill、低接受率或高并发下，候选计算可能和短请求争用 GPU。调度器应按 acceptance、draft overhead 和队列状态动态启停，而不是对所有请求强制开启。

## 63.11 候选生成和结构化输出

多 token 候选在自由文本上可能收益明显，在 JSON、grammar 和工具调用中却容易被约束打断。scheduler 需要让 parser、候选、target verify 和 stream 使用同一个 accepted prefix。

评估要分别报告自然语言、代码、JSON、函数调用和长输出的 acceptance length 与 p99。

## 63.12 NEXTN 的候选协议

多 token 候选路线需要定义候选数量、分支结构、position、logits、sampling 参数、target 验证结果和提交规则。协议不清时，模型输出可能看似正常，却在 stop token、temperature、top-p 或结构化 schema 上与普通 decode 不一致。

候选生成和验证要共享 tokenizer、chat template、position scheme 和模型 revision。不同协议不能只通过字符串拼接适配。

## 63.13 候选长度的动态控制

候选长度可以根据最近 acceptance、任务类型、剩余预算和 verifier 结果动态调整。设最近窗口 acceptance 为 a，可以用简单策略：

~~~math
k_{t+1}
=\mathrm{clip}(k_t+\eta(a_t-a^\star),1,k_{\max})
~~~

实际实现要防止在 acceptance 波动时频繁切换。代码、数字、工具调用和 reasoning 任务应使用不同的安全上限。

## 63.14 fallback 和监控

监控记录候选长度、accepted prefix、reject position、target extra compute、rollback bytes、TTFT、TPOT 和最终质量。若候选路径频繁拒绝或状态回滚成本超过收益，应自动关闭或降级。

灰度发布要对比普通 decode 的 token 一致性、停止条件、结构化输出和错误率。推测路径的异常不能通过重试掩盖。

## 63.15 NEXTN 不是一个统一标准名

同一个“多 token 候选”概念，在不同模型和引擎中可能叫 MTP、N-token prediction、draft head、lookahead 或 NEXTN。名称相似不等于接口相同。阅读资料时先问三个问题：候选由独立模型、附加头还是搜索器产生；target 是否一次验证多个位置；候选拒绝后由谁负责采样和回滚。

如果公开资料只说明“支持多 token 生成”，不能直接推断候选树宽度、训练损失、采样等价性和显存布局。正文应把 NEXTN 当作候选路线的工程抽象，具体字段以对应模型卡、代码和 runtime 文档为准。

## 63.16 候选对象的最小契约

一个候选对象至少要携带以下信息：

~~~text
Candidate {
  request_id
  model_revision
  tokenizer_revision
  parent_position
  token_ids
  branch_id
  grammar_state
  sampling_revision
  temporary_state_handle
}
~~~

token_ids 不能脱离 tokenizer 解释，parent_position 决定 target 的 causal 位置，grammar_state 决定哪些节点可提交，temporary_state_handle 则决定拒绝时如何回收。缺少其中任何一类信息，系统可能仍能生成看似正常的字符串，却无法在取消、重试和跨 worker 迁移时证明状态正确。

## 63.17 候选生命周期和状态机

候选应经历 created、verified、accepted、rejected、committed 或 discarded 等状态。状态转移必须单向，尤其不能把 rejected 节点重新标记为 committed。一个请求的正式状态可以写成：

~~~math
s_{t+1}
=\mathrm{commit}(s_t,\mathrm{accepted\_prefix}),
\qquad
s_{t+1}\ne\mathrm{commit}(s_t,\mathrm{rejected\_suffix}).
~~~

日志要记录每次转移的原因、target logits 版本、grammar 版本和 page 变化。这样出现“输出少一个 token”“工具调用重复”时，才能区分候选错误、验证错误、提交错误和客户端重试，而不是只看最终文本。

## 63.18 NEXTN 与 chunked prefill 的调度冲突

多 token 候选主要优化 decode 串行性，但长输入的 prefill 仍会占用 batch token budget。若候选验证和长 prefill 争抢同一轮执行资源，短请求可能得到更高的 token throughput，却出现更差的 TTFT。

调度器可以把每轮预算拆成：

~~~math
B_{\mathrm{round}}
=B_{\mathrm{decode}}
+B_{\mathrm{verify}}
+B_{\mathrm{prefill}},
\qquad
B_{\mathrm{verify}}\le B_{\mathrm{candidate}}.
~~~

候选长度不能只由模型决定，还要受剩余 KV blocks、队列年龄、优先级和 p99 预算限制。一个实用控制器会在 acceptance 下降或队列等待上升时缩短候选窗口，而不是继续追求平均加速。

## 63.19 从协议测试到线上灰度

发布顺序应为：离线 token replay、单请求状态测试、混合长度压测、结构化输出回归、shadow、低比例 canary、再逐步扩大。shadow 阶段可以计算候选和 target 结果，但不得执行工具副作用或向客户端发 stream event。

线上灰度至少分开统计普通文本、代码、JSON、工具调用、长上下文和 reasoning。每一类记录候选长度、接受长度、target verify 时间、回滚 bytes、TPOT、p99、错误率、usage 差异和最终质量。任一高风险协议指标失败，应关闭 NEXTN 并保留 baseline 路径。

## 63.20 候选路线的资源和协议账本

NEXTN 不是“多生成几个 token”这么简单。候选对象至少要绑定 request、parent position、tokenizer/template、sampling、grammar、temporary state 和 target revision。调度器还要为候选保留临时 page、验证 workspace 和回滚时间；否则长请求加入时，候选会挤掉已有流式请求。

候选路径的状态可以写成：

```text
created -> schema_checked -> verified
       -> accepted -> committed
       -> rejected -> discarded
```

`verified` 表示 target 已检查，不表示业务动作已提交；`committed` 只允许包含 accepted prefix。工具 call、文件写入和发布等副作用必须留在候选协议之外，等待正式 item 和 policy gate。

## 63.21 路线选择与灰度控制

对于不同 workload，NEXTN 可以选择不同候选窗口。代码和规则性 JSON 可能接受较长窗口，开放写作和高温采样可能接近 baseline。控制器应使用滑动窗口统计 acceptance、p99、显存余量和结构化错误，并设置迟滞，避免请求级别频繁开关。

灰度报告不能只写“平均 TPOT 提升”。至少要比较 target call reduction、候选时间、verify 时间、临时 state、fallback、stream event、usage、任务成功率和单位成本；对工具任务还要验证没有重复执行。shadow 只计算候选，不得把未经 target 验证的 token 发给客户端。

## 63.22 候选对象的生命周期

NEXTN 的候选不是已经生成的正式输出，而是带 parent position、tokenizer/template、sampling、grammar、target revision 和 temporary state 的暂存对象。它应经历：

```text
created -> schema_checked -> target_verified
        -> accepted -> committed
        -> rejected -> discarded
```

`target_verified` 只表示目标模型检查过候选，`committed` 才表示客户端可以看到并进入正式 cache。工具 call、文件写入和发布必须在 committed item 之后再经过 policy gate，不能因为候选中出现了合法 JSON 就提前执行副作用。

## 63.23 NEXTN、MTP 和独立 draft 不能混为一谈

MTP 通常在训练阶段增加多 token 预测目标，NEXTN 更像 serving 中的候选路线名称；独立 draft 使用另一模型提出 token；EAGLE 使用附加 head 或 feature-level 信息。它们都可能减少 target decode 调用，但训练对象、候选来源、版本兼容和验证成本不同。

比较时固定同一 target、模板、采样、硬件、候选窗口和质量验收条件，分别记录候选成本、target verify、接受长度、临时 state、p99 和回退。一个方法在模型卡中叫多 token prediction，不等于它可以直接替换另一个引擎的 NEXTN 接口。

## 63.24 grammar 和结构化输出是压力测试

开放文本的局部 token 相关性较强时，候选可能容易接受；JSON schema、代码语法和工具参数有许多边界，候选在一个逗号或引号处失败就可能缩短整轮收益。grammar backend 还需要维护候选分支各自的状态，不能共享一个已经前进的 parser。

结构化回归要比较完整 event sequence、schema validity、tool call id、参数 round-trip、stop reason 和 rejected token 是否泄漏。grammar 错误应直接关闭候选路径或改走 baseline，不应把一个被 target 拒绝的半结构化对象发给客户端。

## 63.25 scheduler 如何处理候选资源

候选会增加临时 KV、验证算力和回滚时间，因此 scheduler 需要把候选预算纳入 admission。长请求的候选可能挤占短请求的正式 decode；验证批次也可能延迟新请求的 TTFT。调度器可以限制每轮候选数、按队列年龄公平、在高压时缩短窗口或只对高收益 bucket 开启。

候选 reservation 与正式 KV reservation 必须可区分。请求取消、target verify 失败和 worker 重启时，临时状态要被回收；正式 committed prefix 不能被误删。监控要能解释候选导致的 preemption 和 p99 变化。

## 63.26 端到端收益的正确分母

候选路径的收益可以用 target call reduction 诊断，但产品决策要看单位成功成本：

~~~math
S_{\mathrm{task}}
=\frac{C_{\mathrm{baseline}}-C_{\mathrm{candidate}}}
       {\max(1,N_{\mathrm{successful\ tasks}})}.
~~~

候选成本、验证成本、临时显存、回退和协议错误都要进入 `C_candidate`。如果 NEXTN 让普通文本 TPOT 变好，却使工具任务重试更多，按 token 计算的 speedup 可能与单位成功成本相反。

## 63.27 灰度和故障回放

上线顺序应是 token replay、单请求状态测试、混合长度压测、结构化输出回归、shadow、低比例 canary、逐步放量。shadow 可以计算 candidate 和 target 结果，但不能执行工具副作用或发送正式 stream event。

故障回放覆盖候选 schema 不兼容、grammar backend 失败、target 超时、临时 KV 不足、取消、cache rollback 失败和结果响应丢失。每个 case 检查 committed token、event sequence、usage、最终 state、工具副作用和用户可见状态。

## 63.28 资料范围与 serving 观察

NEXTN 的稳定知识是“多 token 候选进入 serving 后，必须具备候选对象、版本、grammar、状态事务和回退契约”。它不是一个可以脱离实现自行定义的行业标准。MTP、EAGLE、独立 draft 和 lookahead 的训练及采样语义不同，不能用一个平均 acceptance 覆盖。

参考 speculative decoding、MTP 和相关推理引擎公开资料时，应把论文结论、模型卡声明、代码行为和本地压测分开记录。最终是否启用，取决于目标 workload 下的端到端质量、延迟、显存、协议正确性和故障恢复。

## 63.29 NEXTN 作为 serving 接口而不是算法定理

NEXTN 更适合作为某个引擎或模型的多 token 候选路径名称，而不是脱离实现的统一算法定义。阅读资料时先问候选由什么产生、target 如何验证、grammar 如何跟随、正式 KV 如何提交、失败怎样回退。若这些字段未公开，不能从名称推断它等同于 MTP、EAGLE 或独立 draft。

一个 serving adapter 至少应暴露 candidate window、draft revision、verify mode、grammar support、sampling compatibility、temporary state bytes 和 fallback reason。这样上层 scheduler 能把候选资源纳入预算，下层 engine 能区分 target-only 与 NEXTN 的状态。

## 63.30 候选版本与 cache key

候选路径会改变 hidden、tokenizer/template、position、grammar 和临时 KV 的语义。prefix cache 若只按文本 hash 命中，可能把不同 NEXTN artifact 产生的状态混用。cache key 至少要包含 target revision、candidate revision、tokenizer/template、position scheme、dtype/quantization、grammar 和策略版本。

版本切换时可以排空、重算或只复用已确认的 target prefix，但不能静默复用未验证候选。发布 manifest 应记录 checksum、兼容模型、engine ABI、支持的 sampling 和 rollback target。

## 63.31 NEXTN 的调度公平性

候选生成和验证会占用额外 GPU、显存和 iteration。若只按普通 decode 的 token 数调度，长候选请求可能抢占短交互请求，导致 TTFT 和 p99 恶化。调度器可以按候选预算、队列年龄、租户配额、历史 acceptance 和风险等级做 admission，并在高压时关闭低收益 bucket。

评估要分别测短输入短输出、长输入短输出、代码/JSON、工具和 reasoning。记录 candidate/verify 时间、临时 page、正式 page、queue wait、preemption、fallback 和单位成功成本，才能解释 NEXTN 是降低了 target 调用，还是把成本转移给了 scheduler。

## 63.32 NEXTN 的候选路径不是正式输出路径

NEXTN 或类似多 token 候选模块可以把未来位置一次提出，但 target、grammar 和 scheduler 仍然是最终裁判。候选 token 只能进入临时状态；accepted prefix 才能推进 KV、usage、stream 和工具协议。任何“先发给客户端再验证”的实现都无法保证可撤销。

## 63.33 动态窗口的公平性

如果 scheduler 只按 acceptance 给高收益请求更大窗口，低接受率请求可能长期得到更少 GPU 服务，形成新的队列不公平。窗口策略应同时考虑 deadline、队列等待、显存、优先级和最近收益，并设置上限。评测要按请求类型报告 p50/p95，而不是只看总吞吐。

## 63.34 NEXTN 与 grammar/工具调用

候选生成要知道结构化输出的合法 token 集合，验证后才推进 parser state。工具 JSON、代码字符串转义和 stop token 是高风险切片；需要测试中途拒绝、半个字段、非法 unicode、并行工具和客户端断流。

## 63.35 NEXTN 的验收结论

NEXTN 上线前至少要通过 golden token replay、grammar/JSON、工具无副作用、取消恢复、worker 重启、cache 版本和灰度回退。它的价值是为 target 提供更长候选，不是免除 target、policy 和 executor 的确认。

当公开资料只给出一个模型接口字段时，正文应只确认该字段的产品语义；候选 head、训练数据和内部 kernel 的细节必须等待一手资料或本地验证。工程判断则以同条件端到端质量、延迟、显存、成本和回滚结果为准。
