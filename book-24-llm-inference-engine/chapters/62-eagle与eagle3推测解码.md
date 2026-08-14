# 第 62 章 EAGLE 与 EAGLE-3：从独立 draft 到附加候选头

## 62.1 speculative decoding 的基本结构

普通自回归推理每次让 target model 生成一个 token。speculative decoding 引入一个更快的 draft source，先提出多个候选，再让 target 一次验证。接受的前缀可以一起提交，拒绝位置由 target 继续生成。

最初常见的 draft source 是一个较小的独立模型。它有自己的权重、tokenizer、template 和 KV cache。EAGLE 系列探索另一条路线：使用 target 的 hidden features 或附加 draft head 生成候选，从而减少维护完整小模型的成本。

## 62.2 为什么不直接训练一个小模型

独立 draft model 的优点是可以单独缩放、升级和部署，target 甚至可以不改。缺点是它要占用额外权重和显存，还要保证 tokenizer、position、sampling 和模板对齐。两个模型的分布差异越大，候选接受率越低。

附加 head 可以复用 target 的主干表示，通常减少模型加载和跨模型通信；但它更强地绑定 target revision、hidden layout、position、量化方式和 runtime kernel。一旦 target 升级，head 可能必须重新训练或重新验证。

## 62.3 EAGLE 类路线的抽象

设 target 在当前上下文输出 hidden features `h_t`，draft head 根据这些特征生成候选树或候选序列 `D`：

```math
D=\mathrm{DraftHead}(h_{\le t},x;\phi)
```

target 再计算候选路径上的 logits，执行验证：

```math
Y=\mathrm{Verify}_{\mathrm{target}}(x,D,\mathrm{mask})
```

只有通过验证的前缀进入 committed output。附加 head 的关键不是“猜得像”，而是在尽量低的额外成本下产生高接受率候选。

## 62.4 EAGLE 与 EAGLE-3 应该怎样读

阅读版本演进时，不要只背 EAGLE、EAGLE-2、EAGLE-3 的名字。要比较五个问题：draft 使用哪些 hidden feature；候选是链式还是树式；训练目标和数据是什么；target 兼容范围是什么；在不同 workload 上的 acceptance、额外显存和 kernel 成本如何。

论文中的 speedup 往往绑定特定 target、GPU、batch、输出长度和 sampling。换成量化 target、工具 JSON 或高并发 batch 后，收益可能不同。

## 62.5 一个代码与开放写作对照

代码生成有大量固定语法、函数模板和重复模式，draft head 更容易预测未来 token。开放写作的候选分布更分散，temperature 一升高，target 和 draft 更容易分歧。工具调用和 JSON 又受到 grammar 约束，合法候选比例可能提高，但一次拒绝就需要完整回滚 parser 状态。

因此 EAGLE benchmark 至少要按代码、JSON、数学、开放文本和工具调用分桶，而不是只给一个平均 speedup。

## 62.6 一轮 EAGLE 的状态变化

```text
target hidden state
  -> EAGLE head proposes candidates
  -> target batched verify
  -> accepted prefix
  -> commit output/cache
  -> reject branch discarded
```

候选 state 和 committed state 必须分开。假设候选为 `A B C`，target 在 `B` 处拒绝，系统不能把 `C` 的 hidden 或 KV 留在正式 cache；grammar parser、tool call builder 和 stream event 也只能处理 accepted prefix。

## 62.7 正确性和采样

在 greedy 场景中，target 可以直接判断候选是否等于 target 的选择；在随机采样场景中，需要根据目标分布处理拒绝采样或 residual distribution。若简单地“接受匹配 token，否则取 target argmax”，结果就不再等价于原始采样分布。

工程上常为了性能采用近似模式，但报告必须说明是保持分布的 speculative decoding，还是为了吞吐使用了有质量偏差的启发式。二者不能用同一个“无损加速”描述。

## 62.8 兼容性 manifest

部署 EAGLE 类 head 时，manifest 至少应包含：

```text
target model revision
draft head revision
tokenizer/template hash
hidden dimension and layer layout
position and attention mode
quantization
parallel layout
runtime/kernel version
grammar and streaming support
```

启动时先做结构校验和 golden logits；线上再做 acceptance、p95、显存和输出等价性 shadow test。能加载权重不等于 head 与 target 语义兼容。

## 62.9 评估指标

至少记录 acceptance rate、平均 acceptance length、target call reduction、draft latency、verify latency、TTFT、TPOT、峰值显存、batch throughput 和质量差异。对低接受率请求，观察 runtime 是否自动关闭 speculative。

一个有用的收益定义是：

```math
G_{\mathrm{e2e}}=\frac{T_{\mathrm{baseline}}-T_{\mathrm{spec}}}{T_{\mathrm{baseline}}}
```

同时报告单位成功成本和 p99，因为高峰期 draft state 占用显存可能让并发下降，平均 TPOT 变好却使总体服务变慢。

## 62.10 常见失败

target 与 head revision 错配；量化改变 hidden 分布导致 acceptance 崩溃；拒绝分支污染 KV；grammar parser 提前提交半个工具调用；流式客户端看到被拒绝 token；高温度任务仍强制启用；fallback 时没有清理临时状态。

## 62.11 面试回答与练习

回答“EAGLE 和普通 draft model 有什么区别”时，应说 EAGLE 类路线使用 target 的 hidden/state 和附加候选头，减少独立 draft 的权重与通信，但与 target checkpoint 和 runtime 绑定更紧；target 仍然验证并决定提交。比较时要看 acceptance、draft/verify 成本、显存、采样正确性和 fallback。

练习一：为独立 draft 和 EAGLE head 设计 artifact manifest。

练习二：构造第二 token 被拒绝的回放，检查输出、KV、grammar 和 usage。

练习三：解释为什么开放写作的 speedup 可能低于代码。

### 62.11.1 EAGLE 的候选路径

独立 draft model 使用自己的参数和 hidden；EAGLE 类方法通常利用 target 的 hidden/state 与附加 head 生成候选，再交给 target 验证。抽象地说：

```text
target hidden/state -> candidate head -> draft tokens
                              -> target verify -> commit/reject
```

它减少独立 draft 的权重和通信，但增加了与 target checkpoint、hidden layout、position、sampling 和 engine 的绑定。

### 62.11.2 接受率不是唯一指标

EAGLE 的候选可能有更高 acceptance，但 head 计算、hidden 读取和候选树验证也有成本。要报告 `accepted_tokens_per_step`、head latency、target latency、临时 KV bytes、p50/p99、fallback rate 和目标任务质量。若 acceptance 提高但 head 代价更大，端到端可能没有收益。

### 62.11.3 artifact manifest 与版本

manifest 至少包含 base model revision、EAGLE head revision、hidden layer/shape、dtype、tokenizer/template、grammar support、sampling compatibility、target GPU 和 engine version。base model 升级后，即使名字相同，hidden 分布也可能改变，不能静默继续加载旧 head。

### 62.11.4 回退和观察

低接受率、grammar 冲突、模型不匹配、OOM 或 p99 超标时，runtime 可以回退普通 decode。回退必须保持输出、usage、cache 和 stream 语义一致，并在 trace 标记原因。用户看到的是同一个答案协议，系统内部却要能区分 speculative 和 baseline 路径。

## 62.12 附加候选头的训练和部署边界

EAGLE 类附加头依赖基座 hidden、tokenizer、position、sampling 和 target verify 语义。权重能加载只是第一关，还要用 golden prefix 比较候选、接受/拒绝、stream、grammar 和 cache。

如果候选头版本不匹配，系统应关闭 speculative 回到 target-only，而不是返回未经验证的 token。

## 62.13 feature-level draft 的直觉

独立 draft model 需要运行另一套主干；EAGLE 类方法利用 target model 的 hidden 或 feature 关系生成候选，目标是减少草稿成本。具体实现、训练数据和 head 形式随版本变化，不能把 EAGLE、EAGLE-2 和 EAGLE-3 的细节混成一个公式。

服务层真正关心三个接口：候选如何产生、候选如何组织成 tree、target 如何一次验证并提交最长 accepted prefix。

## 62.14 tree verification 和回滚

候选树可能包含多个分支，target model 验证后选择一条可接受路径。未被接受的分支不能污染主 KV/state；接受长度、分支索引、position 和 page 提交必须是原子的。

测试要故意制造第一 token 接受、第二 token 拒绝，以及多个分支都部分匹配的情况。比较普通 decode 和 speculative decode 的最终 token、每层 cache、随机种子语义和取消恢复。

## 62.15 EAGLE 类方法的适用条件

候选模型或附加头在目标领域、tokenizer、模板和 sampling 条件变化后，acceptance 可能下降。代码、数字、工具 JSON 和长 reasoning 的分布与普通聊天不同，应分别校准。

当 acceptance 过低时，系统应降低候选长度、切换独立 draft、关闭 speculative 或回到普通 decode。速度优化不能以输出正确性和状态一致性为代价。

## 62.16 EAGLE 类方法到底减少了哪一部分成本

普通 speculative decoding 通常要运行一个独立 draft model。draft model 有自己的 embedding、层、KV 和调度请求；它的好处是概念清晰，代价是额外的主干计算和显存。EAGLE 类路线的直觉是复用 target 的中间表示，让附加头或轻量候选模块在更低成本下提出 token。

可以把每轮成本拆成：

~~~math
C_{\mathrm{round}}
=C_{\mathrm{feature}}
+C_{\mathrm{candidate}}
+C_{\mathrm{verify}}
+C_{\mathrm{state}}
+C_{\mathrm{schedule}}.
~~~

独立 draft 主要增加 C_feature 和 draft state；附加头降低了这部分，却可能增加 feature 读取、树组织和 target hidden 适配的成本。所谓“比独立 draft 快”必须在同一硬件、相同 batch、相同输出质量和相同 verify kernel 下测量，不能只比较模型参数量。

## 62.17 训练数据和服务分布的错位

附加候选头常在 teacher-forcing 的正确前缀上训练，而服务时会遇到自己的候选、拒绝、采样和 grammar 约束。第一步候选即使准确，连续多步也可能进入训练分布之外。训练时应考虑 scheduled sampling、不同长度的未来目标和真实任务的 tokenizer 分布；部署时则应把 acceptance 按位置、任务和 sampling 分桶。

特别要关注以下错位：

1. 训练使用 base chat template，服务使用工具模板。
2. 训练只看普通文本，服务主要生成代码或 JSON。
3. 训练使用 greedy，服务使用 temperature 和 top-p。
4. 训练上下文较短，服务使用长上下文和多模态 placeholder。
5. 训练版本的 target hidden 与运行时版本不一致。

这些情况不会一定导致启动失败，却会让候选质量悄悄下降。兼容性测试必须包含 token 序列和 hidden shape，而不仅是最终字符串。

## 62.18 Tree verification 的提交规则

候选树不是“生成多个答案后挑一个”这么简单。每个节点都带有父节点、token、position、grammar state 和临时 page。target 验证后，系统需要选择一条满足目标分布或 greedy 规则的路径，并把该路径的最长 accepted prefix 原子提交。

可以用如下状态不变量描述：

~~~math
\mathrm{CommittedKV}
=\mathrm{KV}(\mathrm{context}\mathbin\Vert\mathrm{accepted\_prefix}),
\qquad
\mathrm{VisibleStream}
=\mathrm{accepted\_prefix}.
~~~

未选分支即使共享了父节点，也不能继续持有会被错误复用的 child page。引用计数、取消、抢占和回滚要在同一测试中组合，否则单独测试树验证通过后仍可能在高并发下泄漏临时状态。

## 62.19 采样正确性不能被 speedup 取代

greedy 场景可以直接比较最终 token；随机采样则需要说明 speculative 的接受/拒绝规则是否保持 target 分布。候选被接受的比例高，不代表输出分布没有改变。评估至少包含固定随机种子的可重复回放、经验 token 分布、temperature/top-p、重复惩罚、stop token 和 grammar。

一个实用的回归表应同时记录：

| 维度 | baseline | speculative | 验收 |
| --- | --- | --- | --- |
| 最终 token 序列 | 目标 | 候选路径 | 规则一致 |
| stop reason | 目标 | 候选路径 | 完全一致 |
| usage | 正式 token | 正式 token | 不计临时 token |
| stream event | committed | committed | 不出现撤回 |
| cache state | 单一路径 | 树后单一路径 | 无 rejected 污染 |

如果某实现只报告 tokens/s，却没有这些协议结果，应把它看成性能实验，不是可上线的推测解码结论。

## 62.20 一个树验证的资源账本

假设候选树深度为 4，第一层有 2 个分支，第二层每个分支再有 2 个分支。逻辑上这是 6 个候选节点，但节点可能共享父状态，物理 page、grammar state 和 feature buffer 的占用不一定等于 6 倍。资源账本需要分别记录：

```text
tree_id, parent_id, token, position
feature_ref, grammar_ref, temporary_page
verification_status, accepted_path, refcount
```

target 选择路径后，只把 accepted path 的最长前缀提交到正式 KV；其他分支的临时 page 释放，公共父 page 的引用计数按仍存活的请求更新。取消发生在验证中间时，所有未提交节点都必须可回收。把候选树序列化成一段 token 列表，会丢失这些状态关系。

## 62.21 版本切换与动态回退

EAGLE head、base model、量化、hidden layer 和 engine kernel 任何一个发生变化，都可能改变 acceptance。灰度时应为同一请求同时生成 baseline 和 speculative 的 shadow 结果，但 shadow 不得执行工具副作用；比较 token、stop reason、grammar、usage、p99 和显存后再放量。

运行中若 acceptance 持续低于阈值，或临时 state 将长请求推入 OOM，runtime 可以降低树深、减少分支、切换独立 draft 或关闭 speculative。回退必须保留正式 KV 和已提交 stream，不得把 speculative 的 rejected token 混进 target-only 路径。高风险工具和结构化输出可以设置更严格的独立验收条件。

## 62.22 feature draft 与 token draft 的边界

独立 draft model 在 token 空间预测候选，EAGLE 类方法则利用 target model 的隐藏表示或附加 head 产生候选。两者的候选来源不同，但都必须经过 target verification 才能提交。不能因为候选来自 target 的 feature 就跳过 target 检查，也不能把某个实现的 feature 层位置当成所有 EAGLE 版本的标准。

工程接口要明确 candidate 的来源、对应的 target revision、hidden schema、tokenizer、采样参数和训练 artifact。只加载一个名字相似的 head，不能证明它和当前 base model 对齐；候选质量下降可能来自版本、量化、层选择、模板或 workload，而不是算法本身失效。

## 62.23 训练分布和服务分布必须对齐

EAGLE head 在训练时看到的 hidden、上下文长度、语言、温度、工具格式和拒绝模式，决定了它在服务时能否提出容易被接受的候选。若训练主要覆盖普通自然语言，线上却以代码、JSON、reasoning 或长上下文为主，平均 acceptance 可能很低。

训练/服务对齐清单应包含：模型和 tokenizer revision、上下文位置、temperature/top-p、输出 stop、grammar、量化、batch、prefix cache 和工具任务比例。发布前用真实 workload shadow 测候选长度、接受位置、验证时间、显存和结构化错误，不能只引用论文中的离线 speedup。

## 62.24 树验证的资源事务

候选树不是一串 token，而是一组带 parent、position、feature、grammar 和临时 KV 引用的节点。目标模型验证后只提交 accepted path，其余分支回收；共享父节点的引用计数要等所有仍存活的分支释放后才能减少。

可以把一次提交抽象为：

~~~math
\mathrm{commit\_tree}(T)
=\mathrm{validate}(T)
\rightarrow\mathrm{select\_path}(T)
\rightarrow\mathrm{publish}(P^*)
\rightarrow\mathrm{reclaim}(T\setminus P^*).
~~~

中间任何一步失败，都应保留原来的 committed KV，并释放临时状态。否则 rejected branch 可能污染正式 cache，取消和抢占还会造成 page 泄漏。

## 62.25 动态关闭要考虑 workload 分桶

EAGLE 的收益不是一个全局常数。普通文本可能接受较长候选，代码和 grammar JSON 可能只接受很短前缀，temperature 较高时拒绝会更多。控制器应按任务 bucket 维护 acceptance histogram、p99、临时显存和最终质量，并对低收益 bucket 自动退回 target-only。

关闭条件可以包括：连续窗口的接受长度低于下限、验证时间超过预算、临时 state 触发 preemption、结构化事件不一致、cache rollback 失败或工具动作状态未知。安全和协议 hard gate 触发时立即关闭，不由普通文本的平均 speedup 抵消。

## 62.26 版本灰度和回滚

灰度时要同时运行 target-only 和 speculative shadow。shadow 只比较 token、stop reason、usage、事件、p99、显存和质量，不能执行工具或写入外部系统。逐步放量要按模型 revision、量化、语言、任务类型和租户分片。

回滚时保留已提交的 target KV，丢弃未验证候选；如果 head、base model、tokenizer 或 engine 版本任一不兼容，直接排空或重算。切换一个 URL 不能替代状态回滚，尤其不能把候选 token 发给客户端后再试图撤回。

## 62.27 端到端实验的最小表格

固定模型、模板、采样、硬件和 workload，比较 target-only、独立 draft、EAGLE 和 EAGLE-3（若目标实现支持）。每个请求保存 target call、候选长度、接受长度、验证时间、临时 bytes、回退、TTFT/TPOT、p99、质量、事件序列和单位成功成本。

再做候选树宽度、深度、grammar、长短上下文、取消、worker 重启和 prefix cache 的消融。实验若只报告接受率，无法回答实际省了多少 target call，也无法判断额外显存和状态复杂度是否值得。

## 62.28 最小回归的当前边界

建议构造一个四步 golden prefix：第一步接受、第二步拒绝、第三步有两个部分匹配分支、第四步遇到 stop token。分别执行普通 decode、EAGLE 路径、取消后恢复和重新排队，比较 token、position、KV page、grammar state、usage 和 trace。

EAGLE/EAGLE-3 的论文和模型实现可以支持方法的公开目标与实验边界，但不同版本的 head 结构、训练数据、树宽度、feature 层和引擎接口不能互相移植。未公开的内部结构不应从“feature-level draft”或产品宣传语推断。EAGLE 类方法的核心是降低候选成本，同时把更多版本和状态契约引入 serving；能否获益，必须由目标 workload 的端到端测量决定。

## 62.29 EAGLE 的候选来源边界

EAGLE 类方法通常利用 target 的中间 hidden/feature 训练附加预测模块，候选来源因此不同于独立 draft model。它可能减少候选模型的参数和调用成本，却增加了 base model hidden 接口、层选择、head artifact 和 engine 版本耦合。公开论文能说明方法目标和实验设置，不能自动说明某个产品实现的具体 feature 层。

公平比较要固定 target、tokenizer、模板、采样、候选窗口、硬件和 batch，再比较 independent draft、EAGLE、EAGLE-3 和 target-only。候选生成时间、target verify 时间、临时 KV、接受位置和最终输出协议都要记录。只比较接受率，会遗漏“候选很便宜但验证/回滚很贵”的情况。

## 62.30 树候选的状态回滚

树验证比线性候选多了 parent、branch、position、grammar state 和临时 page。验证结果可以只提交一条 accepted path，其他分支需要按引用计数释放。若 branch 的 grammar 或 KV 状态误共享，拒绝分支可能污染正式路径。

可用一个简单的事务序列描述：

```text
allocate temporary tree
verify target logits and grammar
select one accepted path
publish committed prefix
reclaim rejected branches
```

任一步失败，都从上一个 committed prefix 继续。stream 只发送 publish 后的事件；工具 executor 只接受已验证参数。取消和 worker 重启测试应检查 page 泄漏、重复 token、usage 和工具副作用。

## 62.31 EAGLE-3 等变体的证据边界

名称中的版本号可能对应训练目标、feature 使用方式、候选结构或实现优化，不能据此断言所有细节。正文应把“论文明确写出的机制”“模型卡声明的兼容性”“引擎代码实际支持的配置”和“本地压测观察”分层。未公开的训练数据、层选择和 kernel 只能写成未知或待核验。

当不同资料的接受率或 speedup 不一致时，先检查模型 revision、temperature、grammar、batch、上下文、硬件和统计分母。候选接受率高不一定端到端快，论文 speedup 也不一定适用于生产工具任务。

## 62.32 feature draft 与 token draft 的验证差异

有的 speculative 路径在 hidden feature 上预测，再由 target head 产生 token；有的路径直接生成 token 候选。前者可能共享 target 的表示并降低 draft 成本，后者接口更直观，但两者的训练目标、错误传播和 cache 事务不同。不要看到“draft model”就假设它一定是一个独立语言模型。

## 62.33 树候选的收益和代价

树状候选可以覆盖多个分支，但验证量、临时节点、grammar 状态和回滚复杂度增加。若树宽为 `w`、深度为 `d`，候选节点数可能接近几何级数；实际 runtime 必须限制总节点和每请求预算。候选越多不一定越快，关键是 target 验证能否批量利用硬件。

## 62.34 EAGLE 的兼容验收条件

draft artifact 必须绑定 target revision、hidden shape、tokenizer/template、position、dtype、sampling 和 engine ABI。启动时用固定 prefix 检查 feature/token 序列，运行时监控 acceptance、verify time、rollback error、p99 和质量；任一 hard gate 失败，安全关闭 speculative 回到 target-only。

## 62.35 EAGLE 的上线决策

EAGLE 路径适合在 shadow 中先比较 token 等价、acceptance、验证开销、p99 和显存，再用低比例 canary 验证流式、取消、工具和长上下文。低 acceptance bucket 可以回到 target-only；协议错误、回滚失败、临时 state 泄漏和工具状态未知是 hard gate。

最终报告按成功任务成本而不是单一 tokens/s 决策。只有当候选成本降低、质量和协议不退化、SLO 达标、故障可恢复且回退足够快时，附加 head 才是可维护的 serving 资产。
