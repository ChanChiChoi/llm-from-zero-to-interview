# 安全与 Prompt Injection 评测：来源证据

检索与阅读日期：2026-11-03。本记录仅保存新核验的研究事实、来源版本、阅读范围和证据边界。

## SafeAudit：Agent 工具安全 benchmark 元审计

- **来源：** Xuan Chen 等，*Who Tests the Testers? Systematic Enumeration and Coverage Audit of LLM Agent Tool Call Safety*，[ACL Anthology 页面](https://aclanthology.org/2026.emnlp-main.73/)和[论文 PDF](https://aclanthology.org/2026.emnlp-main.73.pdf)，EMNLP 2026。阅读摘要、引言、方法、实验与限制段落，以及人类验证附录。
- **来源支持：** SafeAudit 用 LLM enumerator 在给定环境和工具集内生成合法工具工作流与用户场景，再以 rule-resistance 检查从既有 benchmark 提取的安全规则仍未覆盖哪些生成样本。论文在 3 个 benchmark、12 个环境中报告超过 20% 的残余不安全样本，并观察到测试预算增加时覆盖增长、随后边际增益趋缓。
- **证据等级：** source-backed（EMNLP 2026 同行评审论文）。
- **边界：** “超过 20%”是论文特定枚举池、规则与环境下的 uncovered 结果，不是线上不安全率，也不是任意 benchmark 的统一漏测率。论文把 rule-resistance 作为完整性的操作性近似；enumerator 主要为 open-loop，论文用 system-prompt 追加规则作为 guardrail 实例，未证明对其他执行层控制同样成立。人类审核的一致性数据只来自论文抽样类别。

## Rethinking Assessments of Prompt Injection Attacks：真实应用任务和攻击条件

- **来源：** Chi Cui 等，*Rethinking Assessments of Prompt Injection Attacks*，[ACL Anthology 页面](https://aclanthology.org/2026.findings-acl.1191/)和[论文 PDF](https://aclanthology.org/2026.findings-acl.1191.pdf)，Findings of ACL 2026，2026-07。阅读摘要、引言、设置定义、实验结论和限制段。
- **来源支持：** 论文将 target task、injected task 和 attack instruction 分别细分，报告 8 种设置、37 个真实应用任务、185 个注入任务、21 条攻击指令和 143,745 次查询。其三个代表模型实验报告：所测真实应用任务比实验室任务更易受攻击，简单攻击指令在所测条件下往往比复杂指令有效；论文也比较 prompt-level 与 model-level 防御。
- **证据等级：** source-backed（Findings of ACL 2026 同行评审论文）。
- **边界：** 这些数量和差异来自论文选择的任务、三种代表模型、提示及攻击集合，不代表应用总体发生率。论文明确排除 tool-augmented setting，以隔离 prompt–model 交互；不可用它替代工具调用、权限、外部状态或多步 Agent 的安全评估。

## LongPIBench：长文档上下文中的注入评测

- **来源：** Yupei Liu 等，*LongPIBench: A Long-Context Benchmark for Prompt Injection*，[ACL Anthology 页面](https://aclanthology.org/2026.findings-emnlp.1510/)和[论文 PDF](https://aclanthology.org/2026.findings-emnlp.1510.pdf)，Findings of EMNLP 2026，2026-10。阅读摘要、引言、数据集与攻击/防御实验、限制段。
- **来源支持：** benchmark 覆盖论文审查、简历筛选、代码审查和邮件摘要四类文档任务；每类均有合成和真实数据，输入上下文从数千到数万 token。论文在八个模型及多种攻击、防御组合上报告，若干在短上下文 benchmark 上表现良好的防御在其长文档设置中明显退化。
- **证据等级：** source-backed（Findings of EMNLP 2026 同行评审论文）。
- **边界：** 文中“真实数据”是论文数据集，不等于在线生产应用审计。作者将范围限定为静态长文档和单次推理，未覆盖多步 Agent、工具调用、跨步骤传播或动态交互；结果只支持论文模型、任务、数据和防御配置。

## 间接注入防火墙与 benchmark 审计：arXiv 预印本 v2

- **来源：** Rishika Bhagwatkar 等，*Indirect Prompt Injections: Are Firewalls All You Need, or Stronger Benchmarks?*，[arXiv 2510.05244](https://arxiv.org/abs/2510.05244)，v1 于 2025-10-06 提交，页面标示 v2 于 2026-03-23 修订。阅读当前 v2 摘要、版本信息和作者对 AgentDojo、Agent Security Bench、InjecAgent、tau-Bench 的讨论。
- **来源支持：** 作者报告一种在 agent–tool 接口运行的输入/输出 firewall 在四个公开 benchmark 上达到其文中所称的低/零攻击成功率和较高任务效用，同时指出所审 benchmark 中存在成功指标缺陷、实现问题和较弱攻击，并提出分阶段攻击策略。
- **证据等级：** source-backed（作者当前 v2 预印本的声明；尚未作为同行评审结论使用）。
- **边界：** firewall 表现和 benchmark 缺陷均绑定作者测试的四个 benchmark、实现、任务和攻击流程；不能据此推广“所有现有 Agent benchmark 都已饱和”或把作者的审计作为领域共识。版本为预印本，仍需独立复核。

## InfoRMIA：token 级成员推断与记忆定位

- **来源：** Jiashu Tao、Reza Shokri，*Information-Theoretic Membership Inference for Granular Quantification of Memorization*，[ICLR 2026 会议页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ddf43af256e11499149cf06040c62caa-Abstract-Conference.html)和[论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/ddf43af256e11499149cf06040c62caa-Paper-Conference.pdf)。阅读摘要、引言、InfoRMIA 方法说明及 token-level 框架描述。
- **来源支持：** 作者提出 InfoRMIA 作为信息论形式的成员推断方法，并提出以 token-level signals 定位生成输出中的记忆信号；论文报告它在其基准上优于 RMIA，并减少对 population samples 的需求。
- **证据等级：** source-backed（ICLR 2026 同行评审论文）。
- **边界：** 方法结果绑定论文的成员推断游戏、模型访问、候选数据和实验基准；token 定位不等同于用户侧已发生可利用的隐私泄露，也不替代检索/工具权限审计、合法数据使用或生产隐私治理。
