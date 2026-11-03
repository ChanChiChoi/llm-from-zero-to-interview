# 第六册第 4 章《解码策略与生成控制》章级审计

审计日期：2026-11-02。状态：完成本章定向扫描、正文精修、配套同步和静态门禁。未运行章节代码、测试、模型、benchmark 或 PDF；没有逐一在线验证全部外链。研究事实与来源阅读范围记在 `research/decoding-sampling-and-test-time-compute-2026.md`，本文件只记书籍修改、验收状态和交接。

## Scope 与章节职责

- 通读并精修 `book-06-llm-deployment/chapters/04-解码策略与生成控制.md`，保留 4.1—4.19 编号，不增删或迁移本章。
- 中心问题：从 logits 到一个可交付响应，采样、搜索、历史惩罚、结构约束、输出上限、推理投入和 speculative proposal 分别改变什么；它们怎样影响质量、协议、状态、延迟和成本，哪些性质必须靠独立业务校验或目标环境测量。
- 与前后章的职责：承接第 2 章的 prefill/decode 和测量边界、第 3 章的 KV 生命周期；将请求参数合同交第 5 章 runtime 兼容性，将 batch 接受长度与调度交第 6 章，将应用工具状态交第 9 章，将运行质量/成本闭环交第 11/12 章；框架内部执行路径留给第二十四册。

## 章级扫描与候选去向

- **日期与范围**：2026-11-02。围绕 beam 长度/停止语义、top-k/top-p/min-p/typical sampling 的规则和证据、重复惩罚历史范围、JSON Schema/grammar 与业务验证、provider reasoning budget，以及 speculative decoding 的正确性和 batch 收益开展定向检索。资料包括原始论文和复核、固定版本 runtime 文档、OpenAI/Gemini 当前 API 文档和近期实现评估；此扫描用于当前章节取舍，不是穷尽全部解码研究。
- **Beam 与长度归一化**：保留 beam 的候选状态、累计对数概率、早停和排序教学；新增 GNMT 固定偏移长度因子与 coverage penalty 的边界，并以两条合成候选手算长度归一化导致的名次翻转。说明它不是所有 runtime 的通用公式。来源固定到 GNMT arXiv v2 §7 式 (14)。
- **Top-k/top-p/typical/min-p 及参数顺序**：扩写同一分布上的候选集合、过滤后重归一化、离散阈值边界、temperature 与 top-p 顺序、并列 top-k tie-break 和 grammar 空交集。Locally Typical 的质量结论限于论文的摘要/故事生成；min-p 的原论文 v8 与批判性复核 v2 并列，正文仅将其作为规则候选，不声称普遍优越。eta/epsilon 作为支持知识留接口入口，不另作主讲。
- **Penalty、stop 与 finish reason**：区分 presence、frequency 与符号敏感的 repetition 变换，明确 token ID 与 prompt/history scope；加入 logit 和概率手算。将 EOS、stop string、max output、schema complete、tool call、cancel/timeout 拆为不同终态，并补充跨 chunk stop 缓冲、token 上限与结构化对象完整性的请求轨迹。
- **结构化输出与 API 合同**：按 parse、schema、语义、授权、执行拆开成功门槛；加入 grammar state/tokenizer 字节边界、XGrammar mask 与端到端指标差别、首次编译/缓存摊销的假设算例、schema revision 和执行前重新授权。OpenAI strict schema/function calling、Gemini thinking 和 vLLM backend 行为均限定为当前文档/固定版本，不泛化到其他 provider。
- **Reasoning effort**：与 temperature 分开，加入可见/推理 token、工具/验证器、输出 cap、incomplete、P95 和单位成功成本的请求账本。明确 provider 未公开的内部 token 不能由可见文本长度反推，并分别演算整体单位成功成本与新增成功的边际成本。
- **Speculative decoding**：从原始 rejection sampling、Medusa/EAGLE、prompt lookup/MTP 接口到 2026 接受判据预印本、batch ragged-state correctness 审查和多 workload H100 评估，拆分 exact target sampling、greedy token equality、relaxed/typical/tree acceptance。正文补联合接受概率、bonus token、状态回滚和不同 batch 下收益收窄等限制；引擎修复细节留第 6 章/第二十四册。
- 未发现需要新增、拆分或重排的独立章级主线。新资料均能增强既有“选择 token—约束协议—结束请求—评估成本”链；章节结构保留。

## 主要正文修改

- 增补可复算例子：GNMT 长度项排名翻转；temperature 与 top-p 候选数变化；同一四 token 分布下 top-k、top-p、locally typical、min-p 的原质量和重归一化；presence/frequency 与 repetition penalty 的 logit/softmax 变化；stop 跨 chunk 缓冲与 8/9-token 上限轨迹；grammar 的 schema/业务失败漏斗和首编译摊销；reasoning effort 的假设成本、截断余量和边际收益；draft 的联合接受与 speculative 性能/正确性账本；按 workload 调参、切片和 holdout 的合成比较。
- §4.17 用明确标记的 100 请求教学漏斗计算格式修复对总任务成功率的理论上限，指出 grammar 不能消除剩余业务、授权与执行失败。
- §4.15 Python demo 的代码状态标为“静态审阅，未运行（2026-11-02）”。预期输出按固定 seed 和 toy logits 手工推演；代码不调用模型，未用执行结果冒充验证。§4.16 FAQ 仅作带链接的误解速查，不视为独立主讲。
- 动态产品资料在 §4.18 注明访问日期；易变的 Transformers、vLLM 和论文事实按固定文档/revision 描述。论文数字始终绑定作者模型、任务、硬件、batch 或预处理/端到端口径。

## 篇幅与数例门禁

按解释性汉字静态计数，排除标题、表格、公式围栏、代码/文本围栏、行内代码/公式和链接文本；按连续教学链计数，不把章节内每个编号标题机械视作独立单元。

| 连续教学单元 | 类型 | 汉字数 | 门槛 | 判断与例子 |
| --- | --- | ---: | ---: | --- |
| 章首导读 + §§4.1—4.5：从 logits 到采样/搜索选择 | 核心 | 2,348 | 1,800 | 通过；含 softmax、greedy/beam、GNMT 排名翻转和 temperature 数值例 |
| §4.6：top-k、top-p、typical、min-p 与组合顺序 | 核心 | 1,922 | 1,800 | 通过；共享分布手算、重归一化、阈值和顺序对照 |
| §§4.7—4.8：历史惩罚与请求停止协议 | 核心 | 2,017 | 1,800 | 通过；penalty logits/softmax、stop buffer 和 max-token 状态轨迹 |
| §4.9：结构化输出、grammar 和工具状态 | 核心 | 2,013 | 1,800 | 通过；token mask、100 请求阶段漏斗、compile amortization 和授权版本边界 |
| §4.10：reasoning effort | 辅助 | 1,220 | 1,000 | 通过；API 上限、假设成本和每新增成功成本算例，链接到请求测量与第 12 章成本主线 |
| §4.11：speculative decoding | 核心 | 1,971 | 1,800 | 通过；累计 log probability、逐位置接受、bonus token 与成本边界 |
| §§4.12—4.14：复现、workload 策略与调参 | 核心 | 1,935 | 1,800 | 通过；多候选、配对种子、质量/成本/流量切片与 holdout 例子 |
| §4.17：结构化工具任务综合案例 | 辅助案例 | 1,242 | 1,000 | 通过；同一假设漏斗推演 grammar 对总任务成功率的上界 |
| §§4.15—4.16、4.18—4.19 | demo、FAQ、资料、收束 | — | — | 不作为独立讲解单元；demo 和 FAQ 均回链主讲段 |

## 公式、代码、链接与配套状态

- **formula_status**：28 个 math 围栏逐一配对，花括号深度闭合；行内美元逐行成对。禁用的 `\operatorname`、尖括号下标、`\text` 内代码下划线均无命中。新增公式使用 `\mathrm` 与明确索引范围；变量和适用条件在邻近正文解释。
- **code_status**：§4.15 唯一 Python demo 静态审阅、未运行；softmax 改为先减最大 logit 再做温度缩放，避免有限大正 logit 在小温度下先溢出；按代码逐步推演 seed=7 的 EOS 路径及异常输入标记，没有执行章节代码。没有做 benchmark、模型实验或 PDF 构建。
- **link_status**：本章 61 个 Markdown 链接，包括 51 个外链和 10 个章节内锚点；锚点逐个对照 heading slug，本地文件/fragment 缺失为 0，重复 heading slug 为 0。全部外链未逐一在线确认。
- **visual_status**：保留/补充过滤策略比较表、finish reason 状态表和 grammar parser-state 轨迹表；内容以可读状态和数值解释为主，无新增 Mermaid。
- **sync_status**：第六册 `简介.md`、`目录.md` 的章摘要已覆盖本章主线，无须重复改写。`PAPERS.md` 固定论文版本并补入接受理论、batch correctness 入口；`EXERCISES.md` 新增本章 6 题；`INTERVIEW_BANK.md` 新增 36B—36F；`KNOWLEDGE_GRAPH.md` 补入解码采样、beam/speculative 与 reasoning budget 子链。`GLOSSARY_EN_ZH.md` 已有 temperature、各过滤策略、penalty、finish reason、reasoning effort 等定义，本轮无需重复造词条。
- **diff_status**：`git diff --check` 通过（2026-11-02）。

## 未解决边界与交接

- §4.15 代码仍未执行；若未来决定验证，须单独记录运行环境、依赖版本、命令、实际输出及边界覆盖，并更新正文状态。本轮精修不调用代码或测试。
- OpenAI/Gemini 动态 API 文档的结论仅按 2026-11-02 访问快照理解；上线前需以目标 provider、model revision 和请求返回字段复核。预印本的理论证书和论文运行结果都不替代本地 serving 评测。
- 2026-11-02 第六册连续复读时发现 `§4.14.4` 排在 `§4.14.3` 之前；已交换两个小节编号，正文内容、篇幅和 Python 代码均未改变。原章级正文和配套门禁结论仍适用；整册连续复读与最终门禁继续记录在 `architecture-audit.md`，本次未复验或运行代码。
