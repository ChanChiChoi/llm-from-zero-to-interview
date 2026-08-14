# 第 66 章 Model Protocol Fit：权重能加载，不等于模型能正确服务

## 66.1 兼容问题往往发生在模型之外

一个模型能被 runtime 加载，HTTP endpoint 能返回文本，看起来就像部署成功了。但前沿模型经常带有 custom encoding、特殊 chat template、reasoning channel、thinking level、tool call item、多模态 placeholder 或 preserve state。只检查权重和 HTTP 200，会把协议错误推迟到线上。

Model protocol fit 关注的是：模型发布的 artifact、输入序列化、runtime state、输出事件和 harness 是否共同形成一个正确的服务契约。它比“支持某模型”更具体，也更容易被测试。

## 66.2 五层契约

第一层是 artifact：权重、config、tokenizer、特殊 token、revision 和量化信息。第二层是 template：role、tool、reasoning 和多模态内容如何编码。第三层是 runtime：position、KV/state、parallel、kernel 和 batch。第四层是 protocol：stream event、usage、tool call id、reasoning item 和错误。第五层是 harness：memory、context folding、权限、trace 和 replay。

可以把适配验收条件写成：

```math
G_{\mathrm{fit}}=G_{\mathrm{artifact}}G_{\mathrm{template}}G_{\mathrm{runtime}}G_{\mathrm{protocol}}G_{\mathrm{harness}}
```

任意一层为 0，整体就不应被宣称为 native fit。通过 adapter 可以得到 adapted fit，但降级和缺失能力必须明确。

## 66.3 custom encoding 会改变什么

custom encoding 可能改变 role token、thinking item、tool call、图片占位、特殊结束符和多 token head 的位置。tokenizer 能加载但 template 不对，模型会把工具结果当普通文本；template 正确但 parser 不认识 reasoning event，可能丢失状态；token 统计不一致，context limit 和价格估算也会错。

多模态输入还要处理图片、音频和文件的 token 或 embedding 占位。客户端不能只计算文字 token，再把媒体当作免费输入。不同 processor 的 resize、patch 数和特殊 token 都会影响长度与显存。

## 66.4 一个 golden request 套件

为每个模型保存：非流式文本、流式文本、多轮、单工具、并行工具、工具失败、JSON Schema、reasoning level、图片输入、超长输入和取消请求。每个用例记录序列化 token ids、request hash、response item/event、usage、错误和最终 artifact。

升级 runtime 时先做结构回放：tokenizer/template hash、特殊 token、call id、事件顺序和 usage；再做质量、吞吐、长上下文和成本压测。这样能把“协议错”与“模型质量变”区分开。

## 66.5 native、adapted 和 unsupported

`native` 表示 runtime 原生理解模型的 template、state、事件和 cache；`adapted` 表示经过明确 adapter，能力差异和回退已经测试；`unsupported` 表示应拒绝或路由其他后端。

最危险的是静默降级。例如模型支持 reasoning item，但兼容层只保留最终文本；模型要求特殊 tool token，但服务把它转换成普通自然语言；模型可以输出多模态 item，但客户端只读 `content` 字符串。接口成功，语义已经错误。

## 66.6 协议风险的分解

可以将风险拆成：

```math
R_{\mathrm{protocol}}
=R_{\mathrm{template}}+R_{\mathrm{state}}+R_{\mathrm{event}}
+R_{\mathrm{tool}}+R_{\mathrm{accounting}}
```

其中 template 风险会让模型读错输入，state 风险会让多轮或 KV 错位，event 风险会让客户端重复或丢失输出，tool 风险会造成错误副作用，accounting 风险会让成本和限流错误。

## 66.7 一个迁移故障

假设模型服务启动正常，普通文本也能回答，但工具调用经常不执行。排查发现兼容层将模型的特殊 tool call item 转成了 assistant 文本，客户端没有得到结构化 `call_id`，因此不会调用 executor。若只查看日志中的 HTTP 200，很难发现问题；golden request 的 item 序列会直接暴露它。

另一个常见故障是思考状态没有在多轮中保留。模型第一次返回 reasoning item，客户端只保存可见文本，第二轮把 tool result 当新 user message，模型失去上下文并重复调用工具。

## 66.8 升级和回滚顺序

先锁定模型、tokenizer、template 和 runtime revision，跑协议 golden tests；再验证非流式和流式输出；再验证 tool call id、reasoning item、usage、错误和取消；最后才测质量、吞吐、长上下文和多模态。任何关键协议回归都应阻断质量压测后的发布。

回滚要同时恢复 adapter、parser、template 和 manifest，不能只回退权重。已有请求需要先完成状态清理，新请求才使用旧协议。

## 66.9 常见失败

只测权重加载；只测单轮文本；把兼容字段当能力证明；忽略特殊 token 和 template；usage 重复统计候选 token；工具 call id 丢失；流式取消后继续提交；多模态 processor 与模型版本错配；unsupported 静默降级。

## 66.10 面试回答与练习

回答“为什么兼容 server 能启动却不能服务某模型”时，应从 artifact、template、tokenizer、position/cache、量化、parser、stream event、tool call、reasoning state、usage 和 harness 逐层排查，并说明 unsupported 能力必须拒绝或显式降级。

练习一：为三个 provider 建立 capability matrix，标出 native、adapted 和 unsupported。

练习二：设计一个能发现 chat template 漂移的 token-level golden test。

练习三：列出 tool call 被拒绝后需要检查的五个状态。

### 66.10.1 五层适配契约

模型接入可以拆成五层：artifact/weight、tokenizer/template、position/cache、generation/grammar、tool/multimodal protocol。权重加载通过只说明第一层；一个服务能返回文本，也不代表其他四层都正确。

每层都要有 native、adapted、unsupported 三种状态。`adapted` 必须有转换和回归测试，`unsupported` 必须在路由前拒绝，不能让后端静默降级。模型 ID、runtime、tokenizer 和模板 revision 应进入 capability key。

### 66.10.2 custom encoding 的风险

自定义 token 或媒体编码改变序列长度、special token、position、stop token 和 tool placeholder。若客户端使用通用 tokenizer 计数，服务端使用 custom encoding，预算可能错误；若 prefix cache key 不包含 encoding revision，可能命中不兼容 cache。

适配层应提供 encode/decode golden、token count、round-trip、special token、empty input、Unicode、媒体 placeholder 和长输入测试。不能用“能生成中文”替代 token-level 契约测试。

### 66.10.3 worked example：工具调用被拒绝

模型生成了合法 JSON，但 policy 拒绝工具调用。正确状态应保留 assistant proposal、policy reason、未执行的 call ID 和后续用户确认；不能把拒绝转成普通文本后让下一个模型再次提交同一个 action。若模型输出 schema 错，属于 generation/adapter 问题；若权限拒绝，属于 policy 状态；两者的回归责任不同。

### 66.10.4 端到端适配验收

用固定 golden requests 测纯文本、长 context、工具、结构化输出、流式、取消、媒体、reasoning item、usage 和 fallback，再进行真实 workload 压测。每个结果保存 request hash、event sequence、final state、token 数、cache dtype、错误和 p99。兼容性是一个持续的 conformance 过程，不是一次启动成功。

## 66.11 适配不是一次性转换

模型协议适配要随着权重、tokenizer、模板、runtime、sampling、工具 schema 和 streaming 版本共同回归。custom encoding 可能改变 hidden shape、position 或特殊 token 语义，不能只检查启动成功。

发布前使用 golden request、长上下文、工具调用、grammar、取消、preemption 和 speculative 测试，并保存原始与适配后的 token/metadata 对照。

## 66.12 protocol fit 的四层检查

模型能正确服务至少要通过四层：权重和 config 能加载，tokenizer 和模板生成正确序列，runtime 的 position/cache/量化契约一致，输出协议和 stop 行为符合客户端预期。任何一层失败，都可能表现为“模型能力变差”。

检查项包括 vocab、special token、chat template、assistant prefix、tool schema、EOS、position ids、quantization scale、KV layout、dtype 和 sampling。

## 66.13 custom encoding 的风险

自定义 encoding 可以为多模态、工具、压缩或特定模型提供额外 token，但它改变 token id、长度、mask、position 或训练/服务边界。若训练使用一种编码、推理使用另一种编码，模型可能仍生成文本，却失去格式和证据能力。

兼容层应有 golden input/output、token 序列快照、decode round-trip、结构化输出、长上下文和错误输入测试。未知 special token 不应静默映射到普通词。

## 66.14 发布和回滚验收条件

模型 manifest 要绑定 encoding 版本、tokenizer、模板、权重、runtime、adapter、量化和评估报告。发布先做 canary，再比较普通请求、工具请求、多模态请求、streaming、prefix cache 和 speculative。

回滚不只切换权重，还要切换协议和 cache schema。旧请求状态若不能兼容，应排空或重算，而不是跨版本继续复用。

## 66.15 tokenizer 与 chat template 的逐 token 契约

适配模型时，第一道风险不是模型不会回答，而是输入序列已经不同。tokenizer revision、normalization、special token、BOS/EOS、role marker、assistant prefix、tool call marker 和 media placeholder 都会影响训练时模型看到的 token。

应保存一组 golden messages 及其 token id 序列，检查：

~~~text
messages -> rendered_text -> token_ids -> model_positions
model_output -> token_ids -> decoded_text -> protocol_events
~~~

空消息、Unicode、换行、长文本、工具参数、嵌套 JSON、图像占位符和多轮对话都要覆盖。只测试“能生成中文”会漏掉角色错位、stop token 丢失和 assistant-only mask 错误。

## 66.16 custom encoding 改变的不只是长度

自定义编码可能引入新的 token id、压缩媒体 token、reasoning channel、工具边界或特殊 mask。它会同时影响词表、position、attention mask、loss mask、KV cache、usage 计数、grammar 和 decode round-trip。

设原始序列为 x，编码器为 E，服务端解码器为 D，最小契约不是 D(E(x)) 看起来相似，而是：

~~~math
D(E(x))=x
\quad\text{on the supported domain},
\qquad
\mathrm{mask}(E(x))=\mathrm{mask}_{\mathrm{training}}(x).
~~~

对于不支持的输入，要返回明确错误；未知 special token 不能静默映射到普通词，否则错误会被推迟到模型生成阶段。

## 66.17 五层兼容性矩阵

可以把适配验收分成五层：

| 层 | 关键对象 | 典型失败 |
| --- | --- | --- |
| 权重 | config、dtype、quantization、adapter | shape 或 scale 不匹配 |
| 编码 | tokenizer、template、special token | role/stop/长度错位 |
| 运行时 | position、mask、KV schema、grammar | cache 或结构化输出错误 |
| 协议 | stream event、usage、error、finish reason | 客户端状态错乱 |
| 治理 | 权限、审计、版本、回滚 | 不可追踪或跨租户复用 |

一层通过不能推出下一层通过。模型权重加载成功只证明第一层，能生成文本通常只证明了部分第二到第四层。

## 66.18 cache schema 与跨版本状态

prefix cache、KV cache、latent state 和 speculative artifact 要绑定 tokenizer/template、model revision、position scheme、dtype、adapter、grammar 和 tenant scope。即使两个版本产生相同的 token prefix，hidden 或 cache layout 也可能不同。

发布时可以选择排空旧 cache、按 schema 双读双写、迁移并校验或直接禁用共享。不能为了提高 hit rate 让新 runtime 解释未知的旧 state。缓存失效是性能损失，错误复用则是正确性和安全事故。

## 66.19 golden replay 和回滚

golden replay 应覆盖普通文本、空输入、Unicode、长上下文、结构化输出、工具调用、流式、取消、多模态 placeholder、reasoning item、prefix cache、speculative 和 fallback。每次回放保存输入 hash、token 序列、event sequence、usage、finish reason、错误码和最终 state。

回滚需要同时切换权重、tokenizer、template、encoding、runtime、cache schema、adapter 和路由配置。已有请求若不能安全迁移，应排空或在旧版本完成；不能只把模型 URL 换回去就宣称完成回滚。

## 66.20 Preflight 与适配回退

请求进入 engine 前，可以做一个 protocol-fit preflight：读取 model/encoding/template revision，展开消息和工具 schema，计算 token 与媒体 placeholder，检查 position、grammar、上下文、cache schema、输出格式和 capability matrix。失败时返回明确原因，例如 `template_mismatch`、`encoding_unsupported` 或 `state_schema_incompatible`，而不是让模型继续生成一段错误文本。

adapter 的每个转换都要有可验证的输入和输出。若 provider 不支持某个 reasoning item，可以降级到只读最终文本，但必须在产品和 trace 中标注不可见状态；若 provider 不支持工具并行，可以改为串行，前提是幂等、顺序和成本已经测试；若 custom encoding 无法保证 round-trip，则只能拒绝该请求或路由到 native 后端。

## 66.21 tokenizer 和 chat template 是逐 token 契约

协议适配最容易被低估的对象是 tokenizer 和 chat template。相同的角色文本，如果 special token、tool call 标记、thinking 字段或 stop token 不同，模型看到的序列就不同。适配不能只比较解码后的字符串，还要比较 token id、位置、attention mask、assistant loss mask 和停止条件。

一个最小 golden case 应保存：输入消息、渲染后的模板文本、token id、特殊 token、生成起点、stop ids、工具 schema hash 和最终事件。任何 revision 改变都要重新生成并审批，不能因为普通问答仍有输出就忽略结构化和多轮状态。

## 66.22 custom encoding 可能改变语义

custom encoding 不只是压缩长度。它可能改变 token 边界、Unicode round-trip、特殊标记、位置编号、mask、grammar 状态、缓存 key 和工具参数解析。若编码后再解码不能恢复原始字节，或者特殊 token 被当成普通文本，错误会延迟到模型生成和 executor 阶段。

可以把适配要求写成三类性质：

~~~math
\mathrm{roundtrip}(x)=x,
\qquad
\mathrm{mask}(E(x))=\mathrm{mask}_{\mathrm{training}}(x),
\qquad
\mathrm{stop}(E(x))=\mathrm{stop}_{\mathrm{runtime}}(x).
~~~

这只是必要条件，不足以证明模型质量。还要检查长上下文、工具 JSON、Unicode、媒体 placeholder、reasoning item 和 cache schema。

## 66.23 适配矩阵的决策规则

把能力分成 `native`、`mapped`、`degraded` 和 `unsupported`。`native` 使用 provider 原生语义；`mapped` 经过明确转换并有回归；`degraded` 能完成部分任务但必须暴露损失；`unsupported` 在 preflight 阶段拒绝或换路由。

例如后端不支持 reasoning item，可以只返回最终文本，但要标记内部状态不可见；不支持并行工具，可以改为串行，但要验证顺序、幂等和成本；不支持某种 custom encoding 的 round-trip，则不能静默替换普通 token。

## 66.24 cache 和状态是适配的一部分

prefix cache、KV、latent state 和 speculative artifact 都要绑定 tokenizer/template、model revision、position scheme、dtype、adapter、grammar、tenant scope 和权限 revision。两个版本即使 token prefix 相同，hidden 或 cache layout 也可能不同。

发布时选择排空、双读双写、迁移校验或禁用共享。错误复用比 cache miss 严重得多。已有长请求若无法安全迁移，应在旧版本完成或从 committed prefix 重算；不能只把模型 URL 切回旧版本。

## 66.25 preflight、灰度和回滚

请求进入 engine 前先展开消息、工具 schema 和媒体 placeholder，计算 token、position、grammar、上下文、cache schema 和 capability matrix。失败应返回 `template_mismatch`、`encoding_unsupported` 或 `state_schema_incompatible` 等明确错误。

发布顺序是离线 token replay、单请求协议测试、长上下文/工具回归、shadow、低比例 canary、逐步放量。shadow 不执行外部副作用；回滚同时切换权重、tokenizer、template、encoding、runtime、cache schema、adapter、路由和观测标签。

## 66.26 适配失败的工程追问

看到“权重能加载”时，继续问：tokenizer 是否一致，模板是否一致，特殊 token 是否一致，mask 和 position 是否一致，KV schema 是否一致，流式事件是否一致，工具状态是否一致，权限和审计是否一致。看到“接口兼容”时，继续问：usage、finish reason、错误码、取消、重试、reasoning 和多模态 placeholder 是否真的兼容。

这些问题能把“能跑 demo”与“能正确服务”区分开。适配成本也要进入容量和单位成本账本；一个看似便宜的 adapter 可能增加 token、解析、重试和人工排查成本。

## 66.27 custom encoding 的正确性测试

自定义编码不是把字符串换成另一组整数那么简单。要验证 tokenizer、special token、chat template、role、图片/文件占位、stop token、position 和 usage 是否保持一致。最小测试包含中文、代码、空消息、连续换行、unicode、长文本、工具 JSON 和多模态占位，并比较 token-level golden output。

## 66.28 协议适配的三层回归

第一层是 schema：请求和响应字段、错误码、usage、finish reason；第二层是事件：流式顺序、tool call id、取消、重试和提交状态；第三层是任务：引用、结构化输出、工具副作用、TTFT、TPOT 和单位成功成本。低层通过不代表高层等价，三层应分别设置验收条件。

## 66.29 不兼容时如何降级

如果后端不支持某能力，应明确关闭、分块、转成异步任务或选择另一后端，并把损失告诉调用方。不能把 1M 输入静默截成 128K，也不能把 reasoning item 当普通文本、把工具失败当回答完成。fallback 必须保留原始范围、遗漏和错误状态。

## 66.30 适配维护的证据链

每次模型、tokenizer、模板、adapter、engine 或 provider 变更都重新跑 conformance。记录 revision、request hash、事件序列、最终状态和资源指标，出现差异时按层定位。兼容接口的价值在于降低迁移成本，不是承诺模型行为相同。

## 66.31 适配的 worked example

假设后端只支持 128K，而调用方按 1M 模型模板发送 900K token；即使 HTTP 接口字段完全相同，适配器也必须在 preflight 阶段拒绝、分块或转异步。若选择分块，结果必须携带块范围、局部证据和合并遗漏；不能返回一个没有范围说明的“完整总结”。

## 66.32 编码差异的质量后果

同一中文字符串在不同 tokenizer 下 token 数、截断位置和输出预算可能不同；特殊 token 不一致还会改变 role、工具和 stop 行为。适配器应把 tokenizer revision 和 template hash 写入请求 trace，并为中文、代码、emoji、长数字和 JSON 做回归。

## 66.33 协议兼容和模型质量的分界

conformance 通过只说明接口行为在给定版本成立，不能说明模型回答质量、事实性、拒答和工具选择相同。迁移验收要分协议 gate、能力 gate、质量 gate、成本 gate 和安全 gate，任一层都不能用另一层的结果替代。

## 66.34 资料边界与小结

Model Protocol Fit 的核心是证明模型在真实协议、编码和状态契约下仍然正确服务。custom encoding、reasoning、工具、媒体和流式事件都可能改变服务语义，必须以 token-level golden replay、协议验收条件、质量回归和端到端 SLO 验证。preflight 让不兼容在执行前可见，显式 adapter 让降级行为可审计。

公开模型卡和 vLLM/SGLang 文档可以支持具体 tokenizer、template、MTP 或量化接口；未公开的内部编码和 state 表示不能从兼容接口名称推断。适配不是一次性转换，而是随着权重、协议、runtime 和治理版本一起持续回归。
