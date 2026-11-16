# 大模型产品体验与延迟资料来源记录（2026-11）

记录日期：2026-11-16。本文只存联网发现的来源事实、版本/状态和证据边界；章节修改与取舍记录在 progress/。

## OpenAI 延迟优化指南

- 来源：[Latency optimization](https://developers.openai.com/api/docs/guides/latency-optimization)。阅读日期：2026-11-16。
- 来源支持：time to first token（TTFT）描述 API 请求开始返回内容所需时间；端到端延迟从用户动作开始，包含检索、模型与工具调用、重试及渲染。指南建议在具有代表性的流量上比较 p50 和 p95，并按模型、服务层和任务分组。流式传输可让用户更早看到首段文字，但首段时间仍包含此前的请求、推理或工具过程，也不等于完整结果已生成。
- 证据等级：供应商官方工程指南；支持该 API 及其优化建议，不是独立的用户体验实验。
- 边界：文档没有为任意产品规定合适的延迟阈值；目标应依据用户任务、目标流量和质量要求设定。流式结果仍可能尚未通过完整的引用、格式或安全检查。

## OpenAI Structured Outputs 指南

- 来源：[Structured model outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)。阅读日期：2026-11-16。
- 来源支持：指定 JSON Schema 可约束受支持结构中的输出；可用字段与 Schema 子集受产品/模型能力限制。该格式与调用工具的 function calling 用途不同。官方文档说明拒绝回答可能不遵守预期 Schema，达到输出上限时也可能得到不完整响应，应用需要处理这些边界。
- 证据等级：供应商官方 API 文档，只支持其公开接口合同。
- 边界：Schema 合规不等于内容事实正确、来源适用或整个用户任务成功；产品仍需业务验证、版本检查和失败处理。具体支持范围会随模型和 API 文档变化。

## Google SRE 服务级目标与监控

- 来源：[Service Level Objectives](https://sre.google/sre-book/service-level-objectives/)；[Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)。阅读日期：2026-11-16。
- 来源支持：SLI 是对服务某项水平的明确定量测量，常见指标包括延迟、错误率、吞吐和可用性；SLO 是由 SLI 测量的目标值或范围。SRE 资料指出只看平均延迟会掩盖慢请求长尾；应结合用户关注的服务特征选择少数有代表性的指标。
- 证据等级：Google SRE 工程手册，记录生产可靠性实践与定义，不是所有交互产品必须使用的固定合同。
- 边界：服务 SLI/SLO 不能替代任务成功、内容质量、用户满意或错误恢复等体验指标；具体目标值由服务和用户需求决定，书中示例数字不能直接作为阈值。

## 延伸来源和状态衔接

- Google HEART / Goals–Signals–Metrics 的 2010 论文事实及边界已记录于[产品发现与评估来源记录](product-discovery-evaluation-sources-2026-11.md)。
- OpenAI Evals 的官方产品状态（截至 2026-11-16 已只读，计划 2026-11-30 关闭）及评估指南边界亦记于同一研究记录；Evals 的平台退役不等于任务样本、验收标准和评估流程这些通用方法失效。

