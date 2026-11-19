# 第二十三章精修审计

审计日期：2026-11-19。状态：外部扫描与范围决定完成，正文重写进行中。

## 章节职责与读者结果

- **范围：**Gemini Developer API Interactions surface 的多轮状态、thought/tool 回放、流恢复、长解码续接、后台执行和 provider Interaction 资源保留。
- **前置：**读者会发送模型 API 请求、理解 JSON 对象，并知道自定义工具由宿主接收和执行。
- **中心问题：**一次 Gemini Interaction 中断后，哪些 API 历史可以续接，哪些协议字段必须原样保留，哪些工具事实仍须由宿主重新查询和验收？
- **预期结果：**读者能区分对话历史、事件流游标、长解码 token、工具调用 ID 与宿主任务状态；在 stateful 与 stateless 路径中正确处理配置和 opaque state；根据 Interaction status、保留和删除合同设计有边界的恢复。
- **书内交接：**第 20 章主讲 OpenAI Responses API；第 22 章主讲兼容 API 的调用方迁移；本章主讲 Gemini 专属 Interaction/status/step/signature 合同；第 24 章主讲 Claude thinking state。通用工具治理、权限、任务 checkpoint 与 workspace 分别回链第 4、8、17、21 章。

## 外部主题与来源对照

事实、版本和来源边界见[Gemini Interactions 状态证据记录](../../research/gemini-interactions-state-evidence-2026-11-19.md)。检索范围是 2026-11-19 当时的 Google AI Studio 官方 overview、API v1 reference、Thinking/Streaming/Background 文档及 Google Gen AI SDK release history；未运行 SDK 或调用 endpoint。

| 外部主题 | 证据所支持的范围 | 读者结果与修订去向 |
|---|---|---|
| Interactions surface 与 API version | 当前官方 overview 称 API 已 GA 并推荐新项目；API reference 提供 `/v1`，SDK release notes 记录 v3.0 默认值从 `v1beta` 转为 `v1` | 开篇说明需同时固定 SDK、API version 与 model；不把新默认版本说成模型能力变化 |
| Interaction history 与逐轮配置 | `previous_interaction_id` 让服务端取回历史；tools、system instruction、generation config 只作用于当次 Interaction，需逐轮重发 | 23.1—23.3 说明服务端会话续接不是应用任务快照，并与 host ledger 分开 |
| steps、thought summary 与 signature | API v1 reference 定义有序 typed steps；Thinking guide 要求无状态重发 thought blocks 和 built-in tool signatures；signature 不能当作可读推理 | 23.3—23.4 解释原样保留及官方必需性表述差异，不写 SDK 内部类实现 |
| 工具请求与结果关联 | `FunctionCallStep.id` 与 `FunctionResultStep.call_id` 配对；授权、执行回执和外部事实仍由宿主维护 | 23.5 用一条有副作用的调用轨迹连接 API step 与第 4、8、17 章 |
| stream 恢复与 long decode | SSE 事件含 `event_id`；API 有 `requires_continuation` 和 opaque `continuation_token`；SDK 增加相应 continuation 支持 | 23.2、23.6 分开事件流恢复、生成续接和 `previous_interaction_id`，不把任一 token 当作外部副作用恢复凭据 |
| 状态、预算与不完整结果 | API status 有 `incomplete` 和 `requires_continuation`；Thinking guide 说 `max_output_tokens` 包含 thoughts，可能硬截断并产出 incomplete | 23.6 解释状态判定；删掉特定型号 token 上限和成本/质量评测处方 |
| Interaction store、保留和删除 | 文档记录默认 store、付费/免费保留期、可配付费期限、按 ID 删除以及 `store=false` 的限制 | 23.7 绑定查阅日期，区分 provider 资源和应用 memory/workspace；部署前要求复核项目合同 |
| provider-managed media steps | API v1 reference 有 `processing_call` / `processing_result` 类型 | 仅在工具 step 边界中说明其不等于客户端 function；删除旧稿视频 token/质量百分比和模型性能段落 |
| 通用任务 checkpoint 与评测 | 相关状态所有权、幂等和 verifier 机制已有本册第 17、21 章及第七册覆盖 | 本章只说明 provider API 的交接边界，不另造统一 probe 套件或评分门禁 |

### 章节组织样本

Diátaxis 用来辨认教程、操作指南、参考和解释任务；它不是教材章节模板。CMU 的学习目标资料建议把读者结果写成可观察动作，但不证明固定顺序有普遍教学效果。采用一次中断的工具任务贯穿解释，再用短表区分 ID/token；不把 API reference 全量字段复制进正文。

## 原稿覆盖与取舍

| 原稿部分 | 处理决定 |
|---|---|
| 23.1 以 Artificial Analysis/DataCurve 中 Gemini 3.8 Flash 为锚点 | 删除榜单锚点。API state 应从接口契约讲起；模型资格和 benchmark 口径不是读者理解 Interactions 的前置。 |
| 23.2—23.3 Interaction steps、stateful/stateless | 保留主线，按同一请求明确区分 Interaction ID、previous ID、配置重发和 host task ledger；移除未证实的 schema/持有关系概括。 |
| 23.4 多文档签名差异与 Python `_gaos` 类型内部细节 | 保留“signature 是协议状态、不是用户可读思维”和官方必需性说明不一致；按 2026-11-19 的 API v1 与 Thinking guide 复核。删除旧 revision SHA、`extra`/open-union、`UnknownStep.raw` 和静态 toy 门禁，因为 SDK v3 系列与 API v1 已变化，且这些生成类实现不构成初学者主线。 |
| 23.5 SSE 恢复和外部工具副作用 | 保留工具调用与外部副作用不构成原子事务的系统推理；补充 event ID 与 continuation token 的不同用途；具体 ledger、幂等与 verifier 回链第 17、21 章。 |
| 23.6 工具上下文、host 权限与 provider-managed 视频 | 保留调用方和宿主责任边界；把 `processing_call/result` 收入 provider-managed step 示例。通用工具/权限回链第 4、8 章，删除独立长视频成本/质量对比段。 |
| 23.7 Gemini 3.8 thinking 档位、1M/65,536 token 数值 | 保留 `max_output_tokens` 硬截止可导致 `incomplete` 的一般合同；删除特定模型限制与路由/benchmark建议，避免动态模型配置取代 Interaction 状态主线。 |
| 23.8 存储、删除、缓存、GPU KV 和长期记忆 | 保留带日期的默认存储、期限、删除及 `store=false` 限制；区分 provider 资源与宿主数据。删除 GPU KV cache 展开。 |
| 23.9 七项 capability probe | 改为按应用真实依赖选择验证点的记录卡；不要求每个接入方运行固定测试清单。 |
| 23.10—23.12 面试模板、总结和参考 | 收束为恢复问题的答案结构与证据边界；替换失效的 Gemini 3.8 source-note 链接为当前 research 记录。 |

## 新正文路线

1. 用长任务中断切入，区分一次 provider Interaction 与应用 task。
2. 用简表说明 `id`/`previous_interaction_id`、function `id`/`call_id`、SSE `event_id` 和 `continuation_token` 各自恢复什么。
3. 沿同一 function call 例子讲 stateful history 和 stateless replay；说明配置重发、thought/signature 原样保存。
4. 讲清 summary 与 opaque signature 的差别，并如实保留两处官方文档对 required 的表达差异。
5. 追踪宿主授权、外部副作用、call result 回灌、断线后查询和 verifier 验收；把 built-in/provider-managed steps 与自定义 function 分开。
6. 解释 background/status/incomplete/continuation 与存储/删除合同，并将动态值限定为查阅日快照。
7. 回到“恢复后哪些事实仍成立”，列出当前接入需记录的 SDK、API version、model 与未知项。

## 配套和未决风险

- [2026-11-16 交叉链接记录](chapter-23-crosslink-note.md)指出原研究文件缺失；本轮补建 research 证据笔记并将替换正文与 `BOOK_SERIES.md` 中的失效引用。
- `thought.signature` 的必需性在 Thinking guide 与 API v1 reference 的标记粒度不同；本轮不调用真实 endpoint，按 endpoint/SDK 版本将该边界留给接入方验证。
- 保留期、API version、SDK continuation 行为均可能变化，正文会注明文档日期和适用产品范围。
- 当前状态：正文、目录、`BOOK_SERIES.md`、旧 2026-09 练习和相关面试题尚待同步；静态验收未执行。
