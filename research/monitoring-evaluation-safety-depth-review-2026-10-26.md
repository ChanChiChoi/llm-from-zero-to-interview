# 第六册第十一章：监控、评估与安全深度精修研究记录

- 日期：2026-10-26
- 正文：[第六册第十一章](../book-06-llm-deployment/chapters/11-监控评估与安全.md)
- 依据：当前 [WRITING_SPEC.md](../WRITING_SPEC.md)、[progress_v2.md](../progress_v2.md) 和仓库 AGENTS.md。
- 结论：保留原章监控、成本、质量评估、幻觉、内容安全、提示注入、隐私、灰度、事故响应和质量闭环；本轮优先纠正指标口径，并把代表研究的机制、解决的问题、作者证据和迁移限制补进正文。没有将章节缩减为工具清单或论文目录。

## 原知识点覆盖

| 原章范围 | 保留内容 | 本轮校正或补深 |
| --- | --- | --- |
| §1 可观测性 | 四种线上正确性、基础设施/推理/业务/风险四层、trace/log/metrics/profile 分工、低基数标签和隐私 | 增补 OpenTelemetry GenAI 语义约定的用途与当前仍在演进的状态；提醒输入/输出内容字段可能包含敏感数据。 |
| §2 延迟、吞吐与 SLO | TTFT、TPOT、ITL、分位数、SLI/SLO、错误预算和 token 工作量 | 说明并行 span 的持续时间不能相加得 E2E；区分集群输出吞吐、逐请求 decode rate 和 pooled decode rate。 |
| §3 GPU、KV 与资源健康 | GPU 利用率、排队风险、显存组成、OOM 归因 | 保留原诊断维度，无删除。 |
| §4 成本 | token/GPU/辅助模型/存储/重试、责任归属、单位成功成本和异常 | 保留原业务成功定义；本章 demo 仍报告每请求成本，不把合成 quality 分数伪装成 task success。 |
| §5 质量评估 | 样本契约、离线切片、线上抽样、自动/人工评估、judge、置信度、用户反馈 | 增加 paired delta、抽样单元、评分覆盖率和缺失输出的选择偏差；加入 HELM、MT-Bench、RAGAS、ARES 方法比较及 PPI 纠偏原理。 |
| §6 幻觉与证据 | 解析/检索/权限/上下文/生成/后处理归因，claim 支持率、引用和拒答 | 保留原证据链；评估器结果不能替代业务事实校验。 |
| §7 内容安全 | 策略契约、FNR/FPR/ASR、流式检查位置与延迟取舍 | 明确各指标的真值集合和分母，零分母记不可计算；补零成功事件的单侧置信上界及其独立同分布前提。 |
| §8 提示注入与 Agent | 直接/间接注入、信任边界、工具权限、幂等、副作用和攻击回放 | 比较 HarmBench 的行为级红队/稳健拒答与 AgentDojo 的状态化工具环境；分别解释效用和安全测量、限制及本地扩展方法。 |
| §9 隐私 | 数据分类、最小化、脱敏、保留、删除和派生数据 | 原知识全部保留，并与 GenAI span 内容采集风险对应。 |
| §10 发布与 §11 事故 | artifact bundle、shadow/canary、统计停止规则、回滚、告警和事故响应 | 保留原流程；章节已提示小流量不能替代统计设计。 |
| §12 质量闭环 | 企业问答、trace、评估回流、shadow/canary 和多模态扩展 | 保留原链路。 |
| §13 demo / §14 资料 / §15 小结 | 标准库监控示例、证据入口与章节总结 | 修正延迟、吞吐、质量覆盖和安全分母；新增验证输出和研究对照。 |

## 代表性研究和官方资料

| 来源 | 解决的问题 / 机制 | 证据与边界 | 本章落点 |
| --- | --- | --- | --- |
| [HELM](https://arxiv.org/abs/2211.09110) | 以共享场景组织语言模型评测，并尽可能在每个场景同时测准确率、校准、鲁棒性、公平性、偏见、毒性和效率。 | 原论文采用 16 个核心场景和 7 类指标，对 30 个模型进行共同覆盖；覆盖集合受可行性取舍约束，不是某个产品的真实流量全集。 | 把 HELM 当评估覆盖地图，再补目标业务、租户、风险和时间切片。 |
| [MT-Bench / Chatbot Arena judge 研究](https://arxiv.org/abs/2306.05685) | 用多轮题和人类偏好处理开放式回答难以写唯一答案的问题，并评估单答打分、reference-guided 与 pairwise judge。 | 作者报告 GPT-4 judge 在论文控制和众包设置下与人类偏好超过 80% 一致；论文也具体分析位置、冗长、自偏好和推理能力局限。这个结果不代表任意 judge、领域或 rubric 都有相同一致率。 | 盲化模型名、交换答案位置、固定 rubric、抽样人工校准，并把事实、数字、权限交给程序或人工验证。 |
| [RAGAS](https://arxiv.org/abs/2309.15217) | 用 reference-free 指标分别评 context relevance、answer faithfulness 和 answer relevance，缩短 RAG 迭代周期。 | 不要求每项都有人工标准答案，但自动 judge、prompt、语言和语料仍会影响分数；reference-free 不等于无测量误差。 | 适合定位组件并筛查；重要结论保留人工标签和独立抽样。 |
| [ARES](https://aclanthology.org/2024.naacl-long.20/) | 从语料合成训练样本、训练轻量 judge，再以少量人工标签和 prediction-powered inference 估计 RAG 组件表现及区间。 | NAACL 论文在 KILT、SuperGLUE 和 AIS 的八个知识密集任务上报告结果。该证据属于论文设定；迁移仍取决于校准标签与目标 query/document 分布相符。 | 正文解释了 PPI 的预测均值加人工残差校正，不把 judge 的均值当人工真值。 |
| [Prediction-Powered Inference](https://arxiv.org/abs/2301.09633) | 用大规模未标注预测覆盖目标分布，以少量人工标签估计预测误差并构造置信区间。 | 抽样、数据切分、预测器训练独立性和目标分布等条件必须满足；代表性不足不能由模型预测自动修复。 | 为 ARES 的“少量人工标签校正自动 judge”提供统计机制入口。 |
| [HarmBench](https://arxiv.org/abs/2402.04249) | 建立有害行为集合与自动红队/稳健拒答评估流程，让攻击方法、模型和拒答防御可比较。 | 原论文含 510 种行为，并比较 18 种红队方法和 33 个目标模型/防御；论文讨论生成 token budget 和 judge 稳健性会改变 ASR。测试覆盖和分类器质量仍有限。 | 攻击预算、输出长度、独立分类器、隐藏样本、人审争议样本都进入评测合同。 |
| [AgentDojo](https://proceedings.neurips.cc/paper_files/paper/2024/file/97091a5177d8dc64b1da8bf3e1f6fb54-Paper-Datasets_and_Benchmarks_Track.pdf) | 在有状态工具环境中同时定义用户任务和攻击目标，并通过环境状态检查 utility 与 security。 | 首版包括 97 个正常任务、629 个安全用例；可测工具链副作用，覆盖仍受其工具、任务和攻击面限制，论文建议充分安全评估包含自适应攻击。 | 用目标工具、租户授权、不可逆副作用和服务端提交状态扩展本地测试。 |
| [OpenTelemetry GenAI span conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md) | 为 model/chat、retrieval、memory 和 tool 操作提供共享的 span operation 与属性名。 | OpenTelemetry 语义约定总页将该体系标为 Development；输入/输出消息字段可含敏感内容，标准字段不等于业务质量指标。动态约定需按目标版本复核。 | 在 §1 连接观测互操作与隐私最小化。 |
| [Trustworthy Online Controlled Experiments](https://ai.stanford.edu/~ronnyk/puzzlingOutcomesInControlledExperiments.pdf) | 总结在线受控实验中的随机分配、指标和实际部署陷阱。 | 作者基于大规模线上实验总结实验实践；稳定实验单元有助于保持一致体验，但稳定分桶自身不证明 treatment/control 可比。 | Canary 若要估计版本影响，应在可比 cohort 内随机分配，或明确匹配、分层与同期对照条件。 |

本轮还保留章节原有的 Prometheus、Google SRE、vLLM metrics、OpenAI Evals、TruthfulQA、OWASP 与 NIST 来源。这些分别用于直方图/可靠性/运行时接口/评估入口/事实性任务/威胁分类/风险治理；它们不能互相替代，也不构成目标业务验收证据。

## 指标与公式审计

- E2E 若阶段严格串行可写为阶段之和；并行子 span 应以 root span 的 start/end 得到用户 E2E，阶段归因按关键路径和重叠关系解释。
- 服务 output tokens/s 为窗口内实际输出 token 数 / 墙钟窗口时长。代码的 pooled decode rate 是“首 token 后 token 数 / 请求 decode 时长之和”，不等于集群吞吐；逐请求平均 TPOT 也不与 pooled rate 互为倒数。
- TTFT 只对实际送达首 token 的请求有观测值。错误或安全阻断且未向用户发出 token 的请求不应被记成零延迟或伪造 TTFT。
- quality_coverage=已评分请求数/全部请求数。已评分回答均值是条件均值；失败、拒答或缺少 judge 输出须按契约作为失败、单列或明确排除，不能静默忽略。
- FNR 分母为真值攻击/危险样本，FPR 分母为真值安全样本，ASR 分母为攻击尝试，且攻击成功按目标副作用判定。分母为 0 时返回不可计算；block rate 不可替代三者。
- 零次攻击成功时，单侧置信度 1−α 的二项上界为 1−α^(1/n)。本 demo 只有 1 条攻击样本，95% 上界是 0.95；该数学界限假设独立同分布试验，不能修复攻击自适应、样本相关和覆盖不足。
- paired candidate/baseline 使用相同样本，计算逐样本差值；请求级 bootstrap 应按用户/会话等实际随机化单元重采样。M=0、无可观测 token 或 zero denominator 不应转成分数 0。

## 教学代码执行与边界检查

- 从 Markdown 提取唯一 Python block，compile 后执行成功；stdout 与相邻 text 输出逐字相同。
- 结果：8 个合成请求；P50/P95/P99 为 1371.0/3122.5/3500.5 ms；6 条有已送达 TTFT，均值 411.3 ms；7 条请求有 TPOT 观测，均值 10.84 ms；pooled decode rate 83.49 token/s；错误率 0.125；全请求 block rate 0.125。
- 安全子集：1 条攻击、7 条安全。攻击 block miss=0、FPR=0、ASR=0；但零 ASR 的 95% 上界为 0.95，安全证据门禁为 false，输出 hold_for_more_safety_evidence。
- 质量只给 6/8 请求打分，覆盖率 0.75，已评分均值 0.88；bad case 为 r2_long_rag、r3_tool_error、r6_policy_edge。
- 边界检查：分位数空样本/非法 q/NaN/Inf/布尔样本拒绝；q=0/1、单样本分位数正常；空均值和零分母返回 N/A，均值拒绝 NaN/Inf/布尔值；零事件 n=0 不计算、n=1 返回 0.95，负 n 和非法 alpha 拒绝；无质量/无攻击/无安全样本的变体不崩溃；0/1 token 请求无 TPOT；负/非有限/布尔阶段时长拒绝。
- 此示例只验证合成分阶段记录和计量逻辑，不连接模型、GPU、OpenTelemetry exporter、实时攻击环境或生产流量。示例门槛只是演示策略结构。

## 配套同步与验收

- PAPERS.md：补实 ARES/AgentDojo 方法边界、HarmBench 的标准化行为与 ASR 口径、Prediction-Powered Inference 入口；已有 HELM、MT-Bench、RAGAS 收录。
- GLOSSARY_EN_ZH.md：新增 False Negative Rate / FNR 与 False Positive Rate / FPR，并说明真值分母和空分母处理。
- EXERCISES.md：新增第六册第十一章练习，覆盖重叠 span、吞吐分母、安全混淆矩阵、零事件上界、paired bootstrap、缺失评分和代表评测框架选择。
- INTERVIEW_BANK.md：未新增重复题；已有 LLM judge 校准、发布报告门禁、RAGAS/ARES 与 AgentDojo 问题可复用。
- 第六册目录已登记第十一章；无目录修改。
- 已浏览并核对本轮新增的 HELM、MT-Bench、RAGAS、ARES/PPI、HarmBench、AgentDojo、OpenTelemetry 和 Trustworthy Online Controlled Experiments 源页；全章原有 URL 未做全量自动状态扫描。
- 未做外部公式渲染或 PDF 构建。

## 遗留边界

没有目标集群或线上 trace，也没有真实任务数据、目标 judge 人工校准、生产 SLO、授权 red-team 环境或隐私流程演练。因此，本轮不能声称目标系统具备某种质量、成本、延迟或安全率。所有论文结果只对作者的模型、数据、任务、攻击预算和评测器有效；GenAI semantic conventions 与 vLLM metrics 动态页面需按部署版本复核。零事件区间不适用于将自适应、相关攻击伪装成独立随机样本的场景。
