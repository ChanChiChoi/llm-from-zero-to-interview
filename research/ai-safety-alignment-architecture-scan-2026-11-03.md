# AI Safety 与 Alignment 架构扫描：来源证据

检索与阅读日期：2026-11-03。本文仅记录册级扫描中新发现的研究事实及证据范围；第八册架构决定和章节去向见 `progress/book-08-ai-safety-alignment/architecture-audit.md`。扫描结果不构成安全控制效果或部署风险的普遍证明。

## 自动化对抗评估

- **来源**：[Pavlova et al., “Automated Red Teaming with GOAT: the Generative Offensive Agent Tester,” ICML 2025 / PMLR 267](https://proceedings.mlr.press/v267/pavlova25a.html)。
- **阅读位置**：PMLR 官方论文页摘要和方法/结果概述。
- **支持的事实**：GOAT 是一个自动化多轮对抗系统，使用通用模型根据目标回复在七种攻击策略间作选择；作者报告在 JailbreakBench 上的特定模型与配置中，ASR@10 为 96%（Llama 3.1 8B）、91%（Llama 3.1 70B）和 94%（GPT-4o）。
- **证据等级与边界**：source-backed。结果限于论文指定的目标模型、JailbreakBench、攻击策略和判分口径；它证明一种自动化红队设计的可行性，不是通用自动红队标准，也不等于 Agent 环境中的副作用评测。

## Prompt Injection 与 Agent 对抗评估

- **来源**：[Evtimov et al., “WASP: Benchmarking Web Agent Security Against Prompt Injection Attacks,” arXiv:2504.18575](https://arxiv.org/abs/2504.18575)。
- **阅读位置**：arXiv 官方论文页摘要。
- **支持的事实**：WASP 提出 web-agent 劫持目标和隔离环境；论文报告被测 Agent 16%—86% 的情况下开始执行注入指令，但端到端完成攻击目标的比例为 0%—17%。作者据此区分“开始服从攻击”和“攻击者完成目标”，并指出其设置中的 Agent 能力与攻击控制持续性限制了端到端结果。
- **证据等级与边界**：source-backed。范围限于论文中的攻击者权限、网页环境、任务和被测 Agent；不能将比例解释为生产发生率或所有浏览器 Agent 的安全率。

## AI Control 与针对监督协议的自适应攻击

- **来源**：[Kutasov et al., “Evaluating Control Protocols for Untrusted AI Agents,” arXiv:2511.02997](https://arxiv.org/abs/2511.02997)，2025-11-04 预印本；[Terekhov et al., “Adaptive Attacks on Trusted Monitors Subvert AI Control Protocols,” ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/54b153ad8a138f4c186f21a8b7341d5e-Abstract-Conference.html)。
- **阅读位置**：前者摘要、引言、§2.2 AI Control 与 SHADE-Arena 设定、§§3—5 蓝队/红队协议结果和§6讨论；后者 ICLR 官方论文页摘要。
- **支持的事实**：Kutasov 等人在 SHADE-Arena 上比较可信模型委派、重采样、关键动作升级等协议，并迭代加强红队。其定义的安全性是独立任务对序列无限重复部署时，在未捕获副任务成功之前先被审核捕获的概率，计算为 `p_audit/(p_audit+p_attack)`；论文报告基础红队下部分协议安全性约 50%—96%，模拟 monitor 的自适应攻击可将 `resample-then-defer` 安全性降至 17%。`defer-on-critical` 在表 2 的强化红队变体下安全性为 100%，效用为 0.53—0.66。Terekhov 等人的 ICLR 2026 论文研究知道协议与 monitor 的自适应攻击，报告在两个控制 benchmark 上通过向输出嵌入 prompt injection 可绕过所测 monitor 型协议，并指出重采样可能放大攻击。
- **证据等级与边界**：source-backed。Kutasov 公式还依赖其攻击轨迹、捕获规则、独立重复任务对和终止事件定义；被审核捕获发生在整条轨迹后仍计为安全，未必保护已发生的不可逆副作用。100% 是论文有限样本和协议设定下的观测结果，不是零风险证明。所有结果依赖模型、威胁模型、攻击能力、监督协议和安全/效用定义；“安全分数”不是现实部署事故概率。对论文中的控制框架还须区分红队 elicitation 是否充分与现实发生率。

## Monitorability 作为测量对象

- **来源**：[Guan et al., “Monitoring Monitorability,” ICML 2026 / PMLR 306](https://proceedings.mlr.press/v306/guan26h.html)。
- **阅读位置**：PMLR 官方论文页摘要、评估定义和结论概述。
- **支持的事实**：论文把 monitorability 定义为监控器从指定观察预测某项属性的能力，强调模型轨迹是否含有可识别信号、以及监控器是否能提取该信号是两个相关条件；提出 intervention、process、outcome-property 三类评估方式，并对非干预设置给出 `g-mean²=TPR×TNR` 度量。作者报告在所测设置中 CoT monitoring 通常优于仅观察动作的监控，但并非完美；同时报告推理轨迹长度、推理计算量及 monitor 计算量与可监控性之间的实验关系。
- **证据等级与边界**：source-backed。结论只适用于论文选择的任务、模型、trace 暴露和监控协议；CoT 可监控不表示它是模型内部真实机制的完整或忠实解释，也不能据此省略权限、执行器和外部状态审计。

## 策略自适应多模态安全分类

- **来源**：[Calvi et al., “Shieldstral,” arXiv:2607.25857v2](https://arxiv.org/abs/2607.25857)，v2 发布于 2026-08-04；[Mistral 官方发布说明](https://mistral.ai/news/shieldstral/)，2026-08-04。
- **阅读位置**：arXiv 官方 HTML 的摘要、§§1–2、§3 数据构造摘要和 §7 评估结构；官方发布说明的模型输入/分数定义及评估范围。
- **支持的事实**：Shieldstral 将内容审核表达为带自然语言 `Instruct`、`Query` 和 `Document` 的二元问答，从 yes/no logits 得到连续分数；技术报告称模型为 3B 参数，训练语料构造约 54.1M 样本，并在其报告的 text、policy-adaptability 与 multimodal 数据集上进行评估。公开输入范围包括文本、图像及图文组合。
- **证据等级与边界**：source-backed。技术报告是作者团队发布的预印本，性能主张应归于其具体 benchmark、数据、基线和协议；Mistral 发布说明属于供应商材料。报告结果不证明该分类器能覆盖任意政策、所有模态或真实工作流，也不赋予授权或替代执行器检查。

## 既有证据记录

册级扫描还复用了 `research/ai-safety-governance-standards-2026.md` 中的 NIST、OWASP、C2PA 与 SSDF 来源，以及 `research/agent-tool-calling-security-2026.md` 中的工具授权、MCP、AgentDojo、ToolSandbox 和敏感日志处理来源。本文件不重复记录其已核验事实。
