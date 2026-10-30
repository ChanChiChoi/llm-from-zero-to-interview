# 项目工具版本与生成式 AI 可观测性：来源证据

检索与阅读日期：2026-10-29。本记录仅保存公开来源支持的版本、接口和数据边界；书籍审阅、纳入决定与修改状态记录在 `progress/`。

## PyTorch 发布状态

- **来源与版本**：[PyTorch Developer Mailing List 的 Release Announcements](https://dev-discuss.pytorch.org/c/release-announcements/27)，访问日期 2026-10-29。
- **阅读位置**：发布列表中的 PyTorch 2.15.0 General Availability 条目。
- **来源支持的事实**：官方发布公告列表将 PyTorch 2.15.0 General Availability 的日期列为 2026-10-28。
- **证据等级与边界**：source-backed（PyTorch 官方开发者公告列表）。列表支持发布状态和日期，不说明某一项目代码、硬件后端或第三方依赖与该版本兼容；版本选择仍须按具体接口和受支持环境核对。

## PyTorch 2.15 API 语义复核

- **来源与版本**：PyTorch `stable` API 文档在访问时重定向到 [2.15 CrossEntropyLoss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html)、[2.15 inference_mode](https://docs.pytorch.org/docs/2.15/generated/torch.autograd.grad_mode.inference_mode.html)、[2.15 optimizer/scheduler](https://docs.pytorch.org/docs/2.15/optim.html) 与 [2.15 AdamW](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html)，访问日期 2026-10-29。
- **阅读位置**：CrossEntropyLoss 的 logits/target/reduction 说明；inference_mode 的限制和 model.eval 提醒；scheduler 调用顺序；AdamW 说明及更新伪代码。
- **来源支持的事实**：CrossEntropyLoss 接收未归一化 logits，类别索引目标要求 long 类型，概率目标与输入同形且目标分布需由调用者保证；`ignore_index` 适用于类别索引目标。inference_mode 比 no_grad 更严格，生成的 tensor 不能用于 autograd 记录的计算，且不会自动切换 `model.eval()`。学习率调度一般在优化器更新后推进；在更新前调用会跳过首个调度值。AdamW 的 weight decay 不累积进一阶矩或二阶矩。
- **证据等级与边界**：source-backed（PyTorch 2.15 官方 API 文档）。这些文档描述 2.15 接口；本教材中旧代码的已验证状态仍只绑定其记录的 PyTorch 2.12.0+cpu 环境，文档复核不等于在 2.15 上运行代码。

## Hugging Face TRL 文档版本

- **来源与版本**：[TRL DPO Trainer 文档](https://huggingface.co/docs/trl/main/dpo_trainer)，访问日期 2026-10-29；页面标注为 `main` 动态文档。
- **阅读位置**：文档版本提示和 DPO Trainer 概览。
- **来源支持的事实**：该页面说明 `main` 版本需要从源码安装，并将常规 pip 安装用户引导至页面标示的稳定版文档；同一页面描述当前 TRL 对 DPOTrainer 的支持。
- **证据等级与边界**：source-backed（Hugging Face 官方文档）。`main` 与稳定版文档可能对应不同 API，动态页面不构成固定版本接口承诺；教学项目应把代码版本和文档版本绑定。

## vLLM 请求与服务指标

- **来源与版本**：[vLLM Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/)，`latest` 动态文档，访问日期 2026-10-30。
- **阅读位置**：指标导言、通用指标表与直方图配置说明。
- **来源支持的事实**：当前文档称 OpenAI-compatible API server 通过 `/metrics` 暴露服务指标；列举端到端请求时延、排队时间、Prefill、Decode、TTFT、inter-token latency、每输出 token 时间、请求队列、运行数和 KV-cache 指标。文档还说明可自定义部分 histogram bucket，但 bucket 会增加每种标签组合的时间序列数量。
- **证据等级与边界**：source-backed（vLLM 动态官方文档，2026-10-30 复核）。`latest` 不固定具体 release；指标名、标签、计算端点和启用条件应以部署版本文档为准，不能仅凭相同名称推定不同 benchmark 的结果可比。

## NIST AI Risk Management Framework 的适用性质

- **来源与版本**：[NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)，动态官方页面，访问日期 2026-10-30。
- **阅读位置**：页面 Overview、AI RMF 1.0 状态说明及 2026-04-07 Critical Infrastructure Profile concept note。
- **来源支持的事实**：NIST 将 AI RMF 定位为自愿使用，用于把可信赖性考虑纳入 AI 产品、服务与系统的设计、开发、使用和评估；当前页面说明 AI RMF 1.0 正在修订，并列出 2026-04-07 发布的关键基础设施可信 AI profile 概念说明。
- **证据等级与边界**：source-backed（NIST 官方页面）。这是访问日期时的框架性质与维护状态，不是法律义务或对任何产品合规性的证明；修订状态和 profile 进度可能变化，实际风险分级、控制要求及适用制度须按业务与适用辖区另行确认。

## OpenTelemetry GenAI 属性和敏感内容

- **来源与版本**：[OpenTelemetry GenAI semantic-convention attributes](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/)，访问日期 2026-10-29。
- **阅读位置**：GenAI attributes、input/output messages、system instructions、tool arguments/results 的说明和警告。
- **来源支持的事实**：属性页面说明 GenAI semantic conventions 已迁至独立仓库；输入/输出消息、系统指令、检索查询、工具调用参数和结果都可能包含敏感信息，消息内容可由 instrumentation 过滤或截断。
- **证据等级与边界**：source-backed（OpenTelemetry 官方语义约定页面）。语义约定描述遥测字段及记录建议，不自动提供脱敏、保留期限、访问控制或合规保证；采用某个 trace SDK 也不代表其中内容已安全处理。
- **当前维护入口复核（2026-10-30）**：[OpenTelemetry GenAI semantic conventions: spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)，`main` 分支页面，访问日期 2026-10-30。页面明确把完整指令、输入和输出内容视为敏感且可能很大，默认建议不记录；生产内容追溯可考虑受独立访问控制的外部存储，并只在 trace 中留引用。该页面为动态且标注 Development 的语义约定，实施要求应再固定所用规范/SDK 版本。

## OpenTelemetry GenAI registry update（2026-10-30）

- **来源与版本**：[OpenTelemetry GenAI semantic-convention registry](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/)，页面标识 Semantic Conventions 1.44.0；访问日期 2026-10-30。
- **阅读位置**：GenAI attributes 中的输入/输出消息、属性弃用说明及敏感内容警告。
- **来源支持的事实**：注册表说明输入和输出消息可能包含敏感信息（包括个人数据），instrumentation 可以提供过滤或截断；若干旧属性已弃用并迁往独立 GenAI semantic-conventions repository。
- **证据等级与边界**：source-backed（OpenTelemetry 官方规范注册表）。字段迁移与状态说明不同页面和 SDK 版本可能不同；此页描述语义约定，不自动实施脱敏、访问控制或保存期限，引用时应固定所用规范和 SDK 版本。

## 模型卡与实验跟踪文档

- **来源与版本**：[Hugging Face Model Cards](https://huggingface.co/docs/hub/en/model-cards)，官方动态文档，及 [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)，latest 文档；访问日期 2026-10-30。
- **阅读位置**：Model Cards 对模型卡内容的说明；MLflow Tracking 对参数、代码版本、指标、输出文件、run metadata 和 artifacts 的说明。
- **来源支持的事实**：Hugging Face 建议模型卡说明模型、用途与潜在限制、训练参数和实验信息、训练数据集与评估结果，并以仓库 README/Markdown 和元数据呈现。MLflow Tracking 提供 API/UI，用于记录运行参数、代码版本、指标与输出文件；其 run 对应一次数据科学代码执行并带有运行 metadata 和 artifacts。
- **证据等级与边界**：source-backed（各自项目的官方文档）。这些工具文档描述记录能力与建议，不会验证填写内容是否完整、样本切分是否无泄漏或结论能否外推；latest 页面可能随版本变化，也不证明某个项目已联调或已复现。

## 直接标识符哈希与去标识化风险

- **来源与版本**：[NIST SP 800-188, *De-Identifying Government Datasets: Techniques and Governance*](https://doi.org/10.6028/NIST.SP.800-188)，2023 年 9 月。
- **阅读位置**：§4.3.1–4.3.2（PDF 第 50–52 页）；关于直接标识符变换、带密钥哈希和穷举风险的说明。
- **来源支持的事实**：NIST 将带密钥哈希列为假名化技术之一，要求密钥具备足够随机性并受保护；无密钥哈希通常不提供安全性，攻击者可能枚举原始值。NIST 还指出，即使使用密钥，密钥泄漏或被发现仍可能导致身份恢复，并在直接标识符处理语境中不建议依靠加密或哈希来保护这类标识符。
- **证据等级与边界**：source-backed（NIST 政府数据去标识化指南）。正文应保留“假名化不等于匿名化”的区别，并避免把哈希当作隐私保护或访问控制；该指南是去标识化治理参考，不等于对所有组织和司法辖区的完整合规意见。

## 生命周期

PyTorch 2.15.0 的发布日期是截至检索日的状态；Hugging Face Model Cards、MLflow `latest`、TRL `main`、vLLM `latest` 与 OpenTelemetry 属性页面属于动态资料。后续引用必须固定实际安装/部署版本，并复核字段语义、默认值和敏感数据处理配置。
