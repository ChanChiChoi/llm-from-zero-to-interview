# 第二十二章精修审计

审计日期：2026-11-19。状态：正文重写与静态验收完成。

## 章节职责与读者结果

- **研究范围：**调用方把已有 OpenAI SDK/API 工作流迁移到另一家 OpenAI-compatible 服务时，哪些调用表面和请求形状可复用，哪些字段、状态和运行行为需要重新核对。
- **前置：**读者会使用 SDK 或 HTTP 发送模型请求，并能读懂 JSON 请求/响应。
- **中心问题：**服务端标注 OpenAI-compatible 后，怎样判断一个具体 endpoint 是否保留了应用实际依赖的行为？
- **预期结果：**读者能绑定具体 SDK、endpoint、provider、model 与文档日期；把应用依赖逐项标为明确支持、有条件或转换、明确不支持、未说明；根据证据决定复用、适配、收缩功能或改用原生接口。
- **书内交接：**第 20 章解释 OpenAI Responses 原生对象和流式合同；第 23、24 章解释 Gemini 与 Claude 的 provider 专属状态。第 22 章只比较调用方迁移合同。第二十四册第 57 章负责服务端实现 OpenAI-compatible serving API 的协议、流、错误、鉴权和限流，本章链接过去，不重复讲实现。
- **组织样本：**Diátaxis 用来区分教程、操作指南、参考和解释需求，不当作教材目录模板；CMU Eberly Center 用可观察学习动作与评价对齐的教学设计建议，不能证明某一固定章节顺序对所有读者有效。

## 外部主题与来源对照

完整来源、版本/日期和限制见[兼容 API 研究记录](../../research/openai-compatible-api-compatibility-evidence-2026-11-19.md)。

| 外部主题 | 外部证据与支持内容 | 读者作用/前置 | 修订后的去向 |
|---|---|---|---|
| API surface 与端点 | OpenAI API Overview、Chat Completions 和 Responses 参考把 endpoint、schema 与流式对象分开描述；Hugging Face 声明其兼容端点仅面向 chat completion | 读者先确认自己在迁移哪个 endpoint，不能按“OpenAI-compatible”推断全 API 支持 | 22.1 区分客户端/传输、请求/响应形状、服务行为；22.2 要求先固定 endpoint |
| 字段、角色和错误方式 | Anthropic 文档列出 system/developer 移位合并、strict 忽略、部分字段静默忽略或 400 | 区分字段出现、被接受与语义保留 | 22.2 说明支持、有条件/转换、不支持、未说明四种记录状态 |
| 工具和结构化输出 | Anthropic strict 限制；vLLM parallel_tool_calls 与模型行为的条件 | 说明 schema 和工具调用不能仅按字段名对齐；宿主仍负责执行与授权 | 22.3 对齐选择、参数、call ID、并行、结果格式和拒绝，深层工具责任回链第 4、20 章 |
| 模型与模板 | vLLM Chat Completions 依赖模型 chat template，且 user 会忽略 | 识别同一 server endpoint 下模型/模板仍可改变请求解释 | 22.2 提供固定版本实例；不再泛称所有 provider 都用同样的序列化差异 |
| reasoning / provider 扩展 | Anthropic 忽略 reasoning_effort；Gemini 对该字段有模型相关行为；Alibaba 列出额外字段及模型差异 | 同名参数不等于同一控制量 | 22.3 按具体文档核对，不给字段做跨 provider 的统一换算 |
| state、stream、usage | OpenAI Chat Completions chunk 与 Responses typed events 不同；DeepSeek Responses 明确无状态并列出固定/忽略字段；供应商对 usage 细分也不同 | 决定当前客户端是否依赖续接、事件和用量语义 | 22.4 解释差异并交接至第 20、23 章；不复刻完整事件/状态教学 |
| 路由、地区和多模态范围 | Hugging Face 可路由到不同 provider；Alibaba base URL 按地区/工作空间区分；Gemini 和 Anthropic 文档列明不同媒体能力边界 | 能区分 API 形状与部署/模型资格 | 22.5、22.6 提醒记录路由、地区、模型与媒体范围；精确能力以所选端点文档为准 |
| 转换的语义保留 | Liu 等 arXiv:2611.15543v1 对合成转换套件、translator 与 SDK 做有限范围研究 | 理解 schema 接受不自动证明转换前后对话信息相同 | 22.4 只用作窄范围证据并明说不是现实流量发生率 |

## 原稿覆盖与取舍

| 原稿位置 | 原有重点 | 决定与理由 |
|---|---|---|
| `22.1` | 兼容字段不等于 tokenizer/template/tool/stream/usage 行为相同 | 保留中心判断，改为说明具体 endpoint 与服务合同范围 |
| `22.2` | transport/schema/serialization/runtime/behavior 五层分类 | 删除看似标准化的五层架构；改为三组读文档问题，并明确这是作者的核对角度 |
| `22.3` | chat template、特殊 token、token ids、golden request | 保留模型/服务配置可改变请求解释的观点，以 vLLM 固定版本文档说明；删除普遍要求保存 token IDs、hash 和 stop 条件的规定 |
| `22.4` | 工具调用 native/adapted/unsupported、静默降级风险 | 保留工具 schema、调用 ID、并行和结果处理需核对；删除未经来源支持的通用 adapter 标签和统一实现规定，深层宿主执行回链第 4、20 章 |
| `22.5` | reasoning/thinking 与会话状态差异 | 保留字段按 provider/model 核对；删除“reasoning 参数可映射成输出上限”等没有来源或证据的类比 |
| `22.6` | tokenizer/context 上限与 KV 估算 | 删除 1M 与 128K 固定示例、token-ID 检验和引擎级容量估算；仍需按目标模型查上下文限制，推理服务实现交第二十四册 |
| `22.7` | usage、成本、计费和限流 | 保留 usage 与价格合同不能跨服务直接相加；删除普遍要求保存完整 router cost manifest 的规定，服务端限流归 Book24 第 57 章 |
| `22.8` | 错误分类与重试/副作用恢复 | 保留“忽略、报错、未知”行为差异；副作用 unknown、幂等与安全重试交第 17 章，鉴权/错误码实现交 Book24 第 57 章 |
| `22.9` | capability matrix 示例 | 改成由应用实际依赖驱动的记录卡，不设全功能字段清单或统一 gate |
| `22.10` | 固定十项 golden request 套件 | 改为仅对高影响、实际依赖行为设计 app-specific 集成用例；说明验证仅覆盖已测条件 |
| `22.11` | 七项常见误区 | 合并到中心解释和迁移案例，不另留重复清单 |
| `22.12` | 面试回答与三项练习 | 改成一段准确的 endpoint/合同核对回答；练习同步为基于官方文档的定向迁移判断 |
| `22.12.1–22.12.5、22.13–22.26` | capability matrix 细则、三种错误、兼容 gate、preflight、request-token 公式、adapter、fallback、回滚、conformance test、内部数据模型、重复数据流、错误分类、验收、能力快照和质量差异 | 去重并并入 22.6 有范围的兼容记录卡；删除自拟 G_compat、G_route、T_request 公式及统一 gate。必须深度交由相邻章节或服务端实现书册的内容已明确链接 |

## 正文路线与写作决策

1. 从调用方改 base URL 后请求“能发出”但语义未必保留的现象切入，声明读者已有 SDK/HTTP 前置。
2. 先界定具体 API surface，再区分客户端/传输、请求/响应形状、服务行为；这些是本章读文档的检查角度，不是新协议标准。
3. 以 Anthropic system/developer 移位、strict 被忽略；DeepSeek Responses 无状态；vLLM template 依赖；Gemini/Alibaba 参数条件差异，带读者区分支持、转换、忽略、报错与未说明。
4. 分别解释工具/schema/reasoning，以及 stream/state/error/usage 对调用方的影响。
5. 用 OpenAI SDK 迁移到 Anthropic 兼容层作完整例子，再用一张简表记录 app-specific 依赖。
6. 明确未做真实账户/API probe。动态产品结论绑定文档访问日期；新预印本绑定其合成套件和 SDK 版本，不外推为全行业失败率。

## 配套和静态验收

- 更新[第二十册目录第 22 章摘要](../../book-20-agent-harness-runtime/目录.md)。
- 新增[练习 35EQ](../../EXERCISES.md)；新增[面试题 40AF](../../INTERVIEW_BANK.md)，并修正既有补充题中“compatible 通常只保证基础请求形状”的过宽表述。
- 本审计不记录外部知识结论的替代版本；事实、来源和证据边界保存在 research 记录中。
- 静态检查状态：已通过。第二十二章 H2 顺序为 22.1—22.7，章节旧公式/占位符扫描为 0，未发现围栏代码；研究记录、正文、审计、目录、练习、面试题、架构地图和总交接中的本地链接目标缺失为 0；35EQ、40AF 条目各唯一；`git diff --check` 通过。
- 未调用真实 API / SDK，未运行代码、自动化测试或模型；未构建或更新 PDF、未修改 plan.md。

## 未决事项

- 产品文档是动态页面；本章的能力样例仅绑定 2026-11-19 访问时的文档状态。实际接入前仍须确认具体账户、endpoint、model、区域和版本。
- 新近 API 转换研究目前是 arXiv v1 预印本，所报合成套件结果不可当作实际业务发生率。
- 第 23 章原有 Gemini source-note 目标仍有已登记的缺失链接，留到该章逐项审阅修复。
- 下一章为第 23 章 Gemini Interactions 与 thought-state 回放；第二十册第 23—24 章和全套书 goal 继续 active。
