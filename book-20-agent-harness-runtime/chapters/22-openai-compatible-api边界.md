# 第 22 章 OpenAI-Compatible API：字段相同，不代表语义相同

## 22.1 为什么“兼容”很容易被误读

很多模型服务提供 OpenAI-compatible endpoint。客户端把 `model`、`messages`、`temperature` 和 `tools` 发送过去，服务返回看起来相似的 JSON，于是团队以为可以无成本切换模型。

兼容通常只覆盖 HTTP 路径、基础字段和部分响应形状。真正影响结果的 tokenizer、chat template、reasoning、tool call、stream event、usage、错误、上下文限制和安全策略可能完全不同。接口能返回 200，只能证明请求被处理，不代表语义和能力等价。

## 22.2 五层兼容性

可以把兼容性拆成五层：

1. **transport**：URL、HTTP、认证、超时和重试；
2. **schema**：请求字段、响应字段、错误 JSON；
3. **serialization**：role、特殊 token、tool、thinking 和多模态内容如何编码；
4. **runtime**：context、KV/state、量化、采样和并行；
5. **behavior**：拒答、工具选择、结构化输出、质量、延迟和计费。

很多“兼容服务器”只承诺前两层。应用若依赖后三层，必须建立 capability matrix，而不能根据 SDK 能否连接判断。

## 22.3 Chat template 是隐藏协议

不同模型可能使用不同的 system、user、assistant、tool 和 reasoning 标记。客户端直接拼字符串，或者把另一个模型的 template 传给后端，模型可能把工具结果当用户文本，把 assistant 前缀重复生成，或无法识别结束位置。

tokenizer 可以加载成功，仍然不代表 template 正确。golden request 应保存最终 token ids、特殊 token、工具 schema 序列化和 stop 条件，切换后做结构比较。

## 22.4 Tool calling 的兼容边界

一个服务可能接受 `tools` 字段，但模型并不可靠地产生结构化调用；另一个服务会返回 tool call，却要求客户端使用不同的 `tool_call_id` 或回传角色。并行工具、工具失败、参数增量和 JSON Schema 支持也各有差异。

适配层应区分：

```text
native: 服务原生理解并验证
adapted: 客户端转换后可用，已通过 golden tests
unsupported: 应拒绝或路由其他后端
```

最危险的是把 unsupported 静默降级成普通文本。模型可能在回答中写出“调用工具”的文字，客户端却把它当完成答案，造成业务状态错误。

## 22.5 Reasoning 和 thinking 的差异

一个 provider 的 `reasoning_effort` 可能影响内部 token，另一个服务可能没有该字段，或者把它当成输出上限。一个 API 返回 reasoning summary，另一个只返回最终文本；一个 runtime 可以保留 item，另一个只能重新发送历史。

迁移时要记录 effective budget、可见内容、stream event、usage 和 fallback。不能仅把 `high` 字符串转发过去，然后在产品页面继续声称用户获得了同等的“深度思考”。

## 22.6 Context 和 tokenizer 的差异

两个服务的 token 数可能不同。相同中文、代码和特殊 token 在不同 tokenizer 下会产生不同长度；一个服务允许 1M，另一个可能在 128K 截断。应用需要在最终后端 tokenizer 上计数，并把服务端实际接收长度写入 trace。

长上下文还涉及模板和工具 token。不能用客户端字符数估算 KV、价格和上下文剩余量。

## 22.7 Usage、计费和限流

兼容响应中的 usage 字段可能只包含输入和输出 token，不包含 reasoning、缓存命中、工具时间或多模态 token。不同后端的价格、速率限制和重试语义也不同。

路由系统应保存 provider、model revision、输入输出 token、reasoning usage（若有）、cache hit、工具成本、重试和价格版本。否则迁移后看似 token 下降，实际账单却上升。

## 22.8 错误与重试

同样的 HTTP 429 可能表示速率限制、并发限制或账户额度；同样的 500 可能发生在模型执行前或外部工具已提交后。适配层应将 transport error、provider error、model refusal、tool error、policy denial 和 unknown execution 分开。

只有明确可重试且没有副作用的错误才适合自动 retry。未知工具状态应先查询 call id 或幂等状态，不能因为客户端没有收到响应就重新发送。

## 22.9 Capability matrix 示例

| 能力 | 后端 A | 后端 B | 迁移动作 |
| --- | --- | --- | --- |
| chat template | native | adapted | 保存 token golden |
| streaming tool | native | unsupported | 禁止流式工具 |
| reasoning level | native | approximate | 记录实际预算 |
| JSON schema | native | adapted | 本地验证并重试 |
| 1M context | declared | 128K | 先路由/切分 |
| usage detail | full | partial | 成本单独校准 |

表格的价值不是给后端贴好坏标签，而是让应用知道在哪些能力上可以迁移，在哪些能力上必须拒绝或降级。

## 22.10 Golden request 套件

每个后端都应回放：短文本、中文、代码、多轮、工具成功、工具失败、并行工具、JSON Schema、reasoning level、长输入、图片或文件、流式取消和限流。保存请求 hash、token ids、item/event 序列、usage、错误和最终 artifact。

先做协议结构比较，再做质量和性能。一个后端输出文本略有差异可能是模型质量变化；如果 tool call id 消失、event 顺序错了，则是适配 bug。

## 22.11 常见误区

把 HTTP 200 当语义成功；把兼容字段当标准；忽略 chat template；把 unsupported 静默转普通文本；用一个 tokenizer 计费所有后端；忽略 reasoning 和多模态 token；把所有错误都重试；只用平均质量判断迁移。

## 22.12 面试回答与练习

回答“OpenAI-compatible API 是否真的兼容”时，应说它通常只保证部分 transport/schema 兼容，仍要核对 template、tokenizer、tool、reasoning、stream、usage、错误、context 和行为。建立 capability matrix，使用 golden requests，明确 native/adapted/unsupported，并为降级和回滚设计协议。

练习一：为两个后端设计十个 golden requests。

练习二：为什么工具调用超时不能直接重试？

练习三：一个后端只支持 128K，但客户端以为是 1M，如何在路由和协议层阻止静默截断？

### 22.12.1 capability matrix 应该写到什么程度

兼容性矩阵至少要包含：

| 能力 | native | adapted | unsupported | 证据 |
| --- | --- | --- | --- | --- |
| chat/template |  |  |  | model card/config |
| tokenizer/context |  |  |  | tokenizer/runtime |
| streaming events |  |  |  | API test |
| tool call/parallel |  |  |  | golden trace |
| reasoning state |  |  |  | provider docs |
| multimodal |  |  |  | processor test |
| cancellation/usage |  |  |  | integration test |

`native` 表示后端原生支持并能通过契约测试，`adapted` 表示 adapter 做了转换，`unsupported` 表示必须在路由前阻止。矩阵还要标 model revision 和 runtime revision，因为同一个 endpoint 的能力可能随版本变化。

### 22.12.2 兼容层的三种错误

第一种是显式错误：后端返回不支持，客户端可以降级或提示用户；第二种是适配错误：字段能发送但语义被改变，例如 reasoning 参数变成普通 max tokens；第三种是静默错误：网关截断 context、丢失 tool result 或把 unknown 当 failed。第三种最危险，应通过 preflight、usage 对账和 golden trace 阻断。

### 22.12.3 worked example：128K 后端面对 1M 请求

路由器在转发前要读取后端 capability，计算模板展开后的真实 token 和输出预留。如果 `T_input+T_output` 超过 128K，应进入检索/分批/异步/其他后端路径；禁止依赖后端默认截断。响应里记录实际模型、输入 token、截断状态和降级原因，客户端才能知道结果没有覆盖全部材料。

### 22.12.4 conformance test

一套最小 conformance test 要包含 tokenizer 计数、纯文本、长输入、system/developer 消息、结构化输出、并行工具、工具拒绝、流式断线、取消、usage、错误、媒体和未知副作用。对每条测试保存 request、event sequence、final state 和资源指标。

通过 conformance test 只说明接口契约满足，不说明模型质量相同。还要用同一任务集比较答案、引用、工具成功、成本和 p99；所谓兼容不能被理解成行为等价。

### 22.12.5 兼容性的五层验收条件

迁移请求能否安全地路由到某个后端，可以用五层验收条件表达：

```math
G_{\mathrm{compat}}
=G_{\mathrm{transport}}G_{\mathrm{schema}}G_{\mathrm{serialization}}G_{\mathrm{runtime}}G_{\mathrm{behavior}}
```

`G_transport` 只回答连接、认证和超时是否成立；`G_schema` 回答字段和错误对象能否解析；`G_serialization` 回答模板、tokenizer、工具参数和多模态内容是否按后端语义编码；`G_runtime` 回答上下文、采样、状态和资源约束是否满足；`G_behavior` 则回答工具选择、结构化输出、拒答和延迟是否达到业务要求。前两层通过而后三层失败时，接口仍可能返回 200，但结果已经不能视为兼容。

例如，后端 B 能解析 `tools` 字段，却无法稳定产生合法参数，那么 `G_schema=1` 而 `G_behavior=0`。路由器应把它标记为 `adapted` 或 `unsupported`，而不是把一次偶然成功当作能力承诺。验收条件记录还要绑定 model revision 和 runtime revision，否则后端升级后，旧的兼容结论会悄悄失效。

## 22.13 Preflight：在请求发出前发现不兼容

兼容层最便宜的错误，是在发送请求前拒绝不满足约束的请求。preflight 至少要拿到目标后端的 model revision、tokenizer、chat template、上下文上限、输出预留、工具能力、结构化输出能力和速率限制，然后对展开后的真实输入计数。

```math
T_{\mathrm{request}}
=T_{\mathrm{template(input)}}+T_{\mathrm{tools}}
 +T_{\mathrm{reserved\ output}}
```

若 `T_request` 超出目标后端上限，路由器应选择切分、检索、异步任务或其他后端；不能依赖服务端截断。中文、代码、图片和工具 schema 都可能让 token 数远大于字符数，客户端用自己的 tokenizer 估算只能作为早期提示，最终验收条件应使用目标后端实际 tokenizer 或已校准的 tokenizer。

preflight 的结果要进 trace：

```text
target_provider, model_revision, tokenizer_hash
template_hash, input_tokens, reserved_output
capability_snapshot, route_decision, downgrade_reason
```

这样一次请求被切分或拒绝时，运维人员能知道是 context 不够、工具不支持、权限不允许还是预算不足，而不是只看到一个泛化的 400。

## 22.14 Adapter 的语义转换与本地验证

adapter 不只是字段重命名。它可能需要转换 role、模板、工具参数、结构化输出、流式事件、reasoning 配置、图片输入和 usage。每次转换都要有输入契约、输出契约和验证器。

例如后端 B 不原生支持 JSON Schema，可以让模型输出普通文本再由本地 parser 验证，但这属于 `adapted`，不等于原生结构化输出。若 parser 失败，系统可以在无副作用条件下重试；如果该输出会触发外部动作，则必须停在 `needs_review`，不能把一次解析失败的文本直接执行。

工具调用适配尤其要保留 call id 和状态。把 `function_call` 改成自然语言字符串会丢失参数边界、并行关系和重试依据；把 provider A 的 reasoning level 映射成 provider B 的 `max_tokens` 也只能称为近似预算，trace 必须记录实际使用的 token 和可见内容。

## 22.15 Fallback、降级与回滚

后端不可用时，fallback 不是“换一个 URL 再发同一个请求”。路由器要根据能力矩阵选择降级路径：只读问答可以切换较短 context 的模型；带工具的任务可能需要关闭并行或改为人工确认；不可逆副作用不能在未知执行状态下换后端重发。

降级决策可以表示为：

```math
G_{\mathrm{route}}
=G_{\mathrm{context}}
 G_{\mathrm{template}}
 G_{\mathrm{tool}}
 G_{\mathrm{policy}}
 G_{\mathrm{cost}}
```

某个检查失败时，系统要返回明确的 `route_decision` 和 `downgrade_reason`。不能只在 UI 上继续显示原模型名称，也不能把 128K 后端的截断结果伪装成覆盖 1M 文档的答案。

回滚除了切回旧模型，还要回滚 adapter、template、tokenizer、capability snapshot 和 verifier。新 adapter 可能已经改变了工具 call id 或错误映射，即使模型本身不变也会造成行为回归。发布前为每个后端保存 golden trace，发生异常时按相同输入进行 shadow replay。

## 22.16 worked example：把 1M 请求安全路由到 128K 后端

假设应用收到一个约 900K token 的文档分析请求，主后端声明支持 1M，备用后端只有 128K。应用不能只根据请求字段 `context_length=1000000` 决定路由，还要检查模板展开、工具 schema、输出预留和后端实际能力证据。

有三种安全结果：

| 条件 | 路由动作 | 客户端可见信息 |
| --- | --- | --- |
| 主后端通过 preflight | 直接请求，保留真实 token 计数 | 模型 revision、输入覆盖范围 |
| 主后端不可用，备用后端不够长 | 分块检索或异步 map/reduce | 分块范围、合并策略、遗漏风险 |
| 主后端未知，备用后端不够长 | 拒绝或请求用户缩小范围 | 明确说明未执行，不生成伪完整答案 |

如果采用 map/reduce，还要把每个块的来源、版本和局部结论保存下来，合并阶段不能声称看过没有进入上下文的文档。若任务包含工具调用，块级结果只能作为证据，外部提交仍需在完整权限和业务状态下单独验收。

## 22.17 Conformance test：从接口相似到行为可解释

每个兼容后端都要跑同一套 conformance test：中文和代码 token 计数、system/developer role、特殊 token、结构化输出、单工具、并行工具、工具拒绝、流式断线、取消、错误码、usage、图片或文件、超长输入和 unknown side effect。测试结果包含 request hash、event sequence、tool call id、最终状态和资源指标。

测试通过只说明契约在给定 revision 上成立，不能证明答案质量相同。后续还要比较引用覆盖、工具选择、拒答、p95、单位成功成本和安全失败。后端升级后重新跑 conformance；若只依赖接口名称，模板或 tokenizer 的变化会在生产中才暴露。

## 22.18 兼容层的最小内部模型

一个可靠的兼容层至少要有四个对象：规范化请求、模型能力快照、事件状态机和外部动作记录。规范化请求保存真实 token、模板和媒体占位；能力快照记录上下文、工具、结构化输出、reasoning 和取消能力；事件状态机负责流式顺序；外部动作记录负责幂等和审计。

如果只把请求 JSON 转发给后端，无法处理不同 tokenizer、工具事件或错误状态。兼容层应在发送前做 preflight，在返回后做 conformance 检查；未知字段默认不应被假设为等价。

## 22.19 兼容性的三种结果

后端适配可以是严格兼容、受限兼容或明确不兼容。严格兼容意味着事件、usage、错误、工具和状态契约都通过；受限兼容意味着某些能力被关闭并向调用方声明，例如不支持并行工具或多模态；不兼容则在执行前拒绝或转入人工选择。

最危险的是“看起来成功”的部分兼容：文本返回 200，但特殊 token 被错误编码，或者工具调用被当成普通文本。此类问题必须由 token-level replay、schema 校验和最终状态断言捕获。

## 22.20 兼容性回归的版本矩阵

每次升级同时记录 provider、model revision、tokenizer、chat template、adapter、engine 和客户端版本。测试矩阵应覆盖短文本、长上下文、中文、代码、结构化输出、单/并行工具、流式断开、取消、错误重试和多模态占位。通过一组输入不代表所有模型版本都通过，结果必须绑定 revision。

## 22.21 兼容适配器的内部数据流

请求进入兼容层后，先解析调用方语义，再根据目标后端的能力快照进行编译。编译结果不只是一个 JSON，还应包含真实 token 计数、模板版本、媒体占位、工具 schema、超时、权限、缓存 key 和允许的 fallback。响应回来后，再把后端事件翻译成内部状态，而不是让业务代码直接依赖某一家 provider 的字段。

这样可以把三个问题分开：请求能否被接受、模型能否正确生成、外部动作是否已经提交。HTTP 200 只回答第一个问题；第二个要看 token/模板和质量回归，第三个要看 tool event、幂等状态和业务 verifier。

## 22.22 兼容层的错误分类

错误至少分为输入不兼容、能力不支持、资源不足、模型生成失败、工具失败、动作状态未知和客户端取消。输入不兼容应在执行前拒绝；资源不足可以排队或降级；动作未知必须先查询状态，不能把整个请求重新发送当成默认恢复。

错误对象应携带可重试性、已提交副作用、request id、backend revision 和用户可见范围。没有这些字段，调用方往往会在不安全的时机自动重试。

## 22.23 兼容性的验收案例

准备一组最小 golden cases：普通文本、中文和代码 token、system/developer role、长输入、结构化输出、单工具、并行工具、工具拒绝、流式断开、取消、超时、图片占位和未知错误。每个 case 保存请求 hash、事件序列、usage、finish reason、工具状态和最终 artifact。

验收不是要求不同后端输出同样的 token，而是要求在声明支持的能力范围内，协议不丢事件、权限不扩大、输出可解析、外部动作不重复，并能明确告诉用户哪里发生了降级。

## 22.24 兼容后端的能力快照

能力快照应包含上下文上限、真实 tokenizer、chat template、special token、结构化输出、工具并发、多模态、reasoning、流式、取消、usage、错误和 cache 语义。快照带 revision 和生效日期，路由时只使用已验证字段；未知能力默认不开放高风险路径。

## 22.25 兼容性不能隐藏质量差异

接口兼容后，仍要比较事实性、引用、拒答、工具选择、长上下文有效长度和单位成功成本。adapter 只解决调用契约，不保证模型输出等价。若备用后端能力不足，应把范围、质量损失和未执行动作告知用户。

## 22.26 OpenAI-compatible API 的阶段性判断与资料边界

OpenAI-compatible 描述的是接口外形，不是模型、协议和运行时的完整等价。真正的迁移工作在 tokenizer/template、工具事件、推理状态、错误、计费和能力边界。preflight、adapter 验证、显式 fallback 和 conformance test 让不兼容在执行前可见；能连上服务只是第一道验收条件。

具体兼容字段以各服务的官方 API、模型卡、tokenizer 和 runtime 文档为准；没有公开的内部映射不能靠兼容名称推断。
