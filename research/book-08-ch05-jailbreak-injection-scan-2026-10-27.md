# 第八册第五章 Jailbreak 与 Prompt Injection 章级扫描

- `date`：2026-10-27。
- `scope`：全文复读 `book-08-ai-safety-alignment/chapters/05-jailbreak与prompt-injection.md` 的 17 个二级段，并针对多步骤间接注入、Agent 工具链、跨会话记忆污染及当前评测方法做定向研究。本记录属于架构审视；不代表正文已精修或论文结果已独立复现。
- `chapter_role`：承接第二章目标错配、第三章可扩展监督和第四章 reward hacking，讲清模型行为层 Jailbreak 与应用信任边界层 Prompt Injection 的不同失效路径。正文覆盖 RAG/Agent 来源信任、攻击谱系、多层控制、应用级指标和邮件助手案例；本章应主讲威胁模型、来源/权限隔离、服务端强制控制和端到端风险测量。

## 结构与内容去向

- `structure_read`：章节按“信任边界 → Jailbreak/Prompt Injection 区分 → RAG/Agent 攻击面 → 攻击谱系 → 多层防御 → 自然语言边界机制 → 评估指标 → 防御闭环 → 误区/案例/资料/练习/demo/总结”推进，整体顺序合理。第 7.4 的多轮诱导主要针对一个会话；第 7.6 将间接注入描述为外部数据源中的恶意指令；第 6.2 虽列出 memory attack surface，却没有跨会话生命周期。长记忆把一次输入问题变成持久状态问题，应补上从写入、持久化、召回、采纳到副作用和修复的 threat model。
- `candidate_placement`：不新增章节。候选在 §6.2/§7.6 加入跨阶段和跨会话间接注入，在 §10.2 纳入一组生命周期检查点；写入恶意记录只是中间状态，完整攻击需观察后续召回、行为采纳及外部后果，修复也应单独评估。先在第五章讲威胁边界与最低限度的生命周期模型，再链接相邻主讲章节。
- `scope_map`：第六章主讲红队流程、危险能力评估、样本偏差和发布阈值；第十三章主讲自动化对抗生成和安全评估闭环；第十四章主讲安全分类器及其检测边界；第十五章主讲 risk-calibrated access 与运行时授权；第十六章主讲失败时权限不扩大的 fallback；第十册主讲数据保护、保留和隐私影响；第二十册第七章主讲上下文和 Agent memory 的工程管理。第五章只保留威胁模型、执行边界及评估目标，不重复这些章节的完整机制。

## 证据扫描

| 来源 | 研究观察 | 适用范围和限制 |
| --- | --- | --- |
| [OWASP GenAI LLM Top 10 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/)；[OWASP Agentic Top 10](https://genai.owasp.org/download/52117/) | LLM01:2026 保留 Prompt Injection 风险；Agentic Top 10 的 ASI06 将 Memory & Context Poisoning 单列为存储/重取状态的跨任务威胁。它适合作为术语和业界风险分类入口。 | 社区风险框架，不是发生率测量，也不证明任一产品当前脆弱；引用正文时应标具体版本，避免旧版编号继续充当现行编号。 |
| [WASP, NeurIPS 2025 Datasets and Benchmarks Track](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1c9818387f5dd0a0bc151214660f059d-Abstract-Datasets_and_Benchmarks_Track.html) | 端到端 web-agent 评测指出，注入可在最多 86% 的案例中部分成功，而 agent 完成攻击者最终目标的比例往往更低；文本/中间行为、任务被部分劫持和已验证副作用应分开测量。 | 数字属于论文 benchmark 的任务、harness 和判分定义；“最多 86%”是上界案例，不是总体生产发生率。 |
| [StepJack, arXiv:2608.06477](https://arxiv.org/abs/2608.06477)（2026-08-06 预印本） | 480 个 CUA 测试例覆盖将一个攻击目标拆散到多页、多步骤的间接注入。固定分解深度下，6 个 CUA 中有 3 个的 ASR 上升，单模型最大差 31.2 个百分点；能可靠跟随参考链的五个 CUA 均值由单步 31.3% 升至三步 36.9%。 | 论文限制包括 12 个攻击目标、分解深度不超过 3，受 API 成本限制仅一轮完整测量。结果说明多步骤值得进入测试矩阵，不说明所有 CUA 都随步骤数单调变脆弱。 |
| [INFUSE, arXiv:2610.20776](https://arxiv.org/abs/2610.20776)（2026-10-14 预印本） | 40 个长流程任务、56 个手工攻击场景、19 个模型，共 41,824 条对抗轨迹。全 benchmark 平均 ASR-L 为 19.6%、heuristic ASR-H 为 15.4%；两项观察口径不同，应配对报告行为 judge 与启发式/制品验证结果。 | 完整网格以 Mini-SWE-Agent 为主；其他 harness 只测 Mini-SWE-Agent 中已被攻破的子集。任务和 payload 仅英语、攻击集静态；沙箱中的 sinkholed 出口不等于现实副作用。ASR-L 由单一 judge 给全量轨迹打分，人工抽样验证只能限定、不能消除偏差。 |
| [MemSecBench, arXiv:2607.27080](https://arxiv.org/abs/2607.27080)（2026-07-29 预印本） | 310 个案例来自 48 个情境，覆盖 2 个 Agent harness、4 个 memory backend、3 个 LLM backend 的 24 种配置；按 Write–Execute–Forget 跨七个检查点审计。作者报告恶意记忆持久化 84.2%、完整 Write–Execute 链 50.3%；成功投毒案例中完整 Execute 链为 59.6%，选择性修复为 56.1%。 | 指定栈、隔离 runtime 和基准任务内的描述性结果，不是生产风险率；记忆持久化不能直接等同于真实动作或用户受害。该 benchmark 更适合作为“评估必须跟踪什么生命周期状态”的证据。 |

## 精修时的验收候选

- `length_scan`：按每个 `##` 粗计汉字、排除标题、代码围栏和表格行（仍会残留公式说明或清单，不能视为正式验收）：§1—§4 约 2,331 字，§5—§6 约 840 字，§7—§9 约 2,469 字，§10 约 1,902 字，§13 案例约 1,016 字；其余若干段约 150—500 字。§5—§6 的独立讲次深度风险最大；后续须先确定教学单元，再按 `WRITING_SPEC.md` §3.3 和附录 B 复数解释性正文。短段如果只是对比、练习或案例中的辅助结构，不应靠扩写硬凑。
- `formula_and_code_scan`：§10 覆盖层级违规、Jailbreak、Prompt Injection、间接注入、泄露、未授权工具、攻击下安全任务、正常误拒、边界标记和严重度加权失败；空分母写明 `N/A`，但现有数学公式多用 `$$` display dollar fences，需按新规范迁移到行内 `$...$` 或复杂公式 `math` 围栏。§17 的 `ratio` 对空分母返回 `None`，但当前 12 条 toy case 没有走到空集；每个 case 有 `expected` 字段，当前聚合代码未读取它，类别/结果依赖硬编码映射，精修时应使期望动作参与检查或移除未用字段。
- `static_calculation`：按当前固定样例手算层级违规 `5/8=0.625`、Jailbreak `1/2=0.5`、Prompt Injection `4/6≈0.667`、间接注入 `3/5=0.6`、泄露 `1/6≈0.167`、未授权工具 `1/2=0.5`、攻击下安全任务 `2/8=0.25`、正常任务成功 `2/3≈0.667`、误拒 `1/3≈0.333`、边界标记覆盖 `2/5=0.4`、严重度加权失败 `23/33≈0.697`；当前教学决策是暂停高风险动作并复测。尚未执行 Python；手算不替代代码运行或独立验证。
- `visual_map`：章首 `text` 信任流可在正文精修时考虑转为闭环 Mermaid：来源/身份 → 可信指令与不可信观察 → 模型候选回答/动作 → 服务端 policy、schema 和确认 → 审计副作用 → 修复与回归。按最新规范使用标准 `mermaid` 围栏，中文/空格/标点标签优先加引号；带文字虚线边统一写 `-.->|标签|`。GitHub 页面渲染仍要核验。
- `open_questions`：最终是否将 memory poisoning 扩成独立小节，取决于第六、十、十三、十五章和第二十册的 coverage map；若章节中引用当前 OWASP Top 10，核对 LLM01 与 ASI06 版本语义；是否沿用 benchmark 数值须明确 model、harness、任务、指标口径和时间。

## 参考来源

1. OWASP GenAI Security Project, [OWASP GenAI LLM Top 10 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/), 2026-08-03；以及[OWASP Top 10 for Agentic Applications](https://genai.owasp.org/download/52117/)，ASI06 Memory & Context Poisoning。
2. Evtimov et al., [WASP: Benchmarking Web Agent Security Against Prompt Injection Attacks](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1c9818387f5dd0a0bc151214660f059d-Abstract-Datasets_and_Benchmarks_Track.html), NeurIPS 2025 Datasets and Benchmarks Track。
3. Zhan et al., [StepJack: Benchmarking Computer-Use Agent Safety Against Multi-Step Indirect Prompt Injection](https://arxiv.org/abs/2608.06477), arXiv v1, 2026-08-06。
4. Nguyen et al., [INFUSE: A Unified Surface Benchmark for Indirect Prompt Injection in Agentic Systems](https://arxiv.org/abs/2610.20776), arXiv v1, 2026-10-14。
5. Chen et al., [MemSecBench: Tracking Agent Memory Poisoning from Persistence to Consequence and Repair](https://arxiv.org/abs/2607.27080), arXiv v1, 2026-07-29。

- `status`：第五章结构、跨章边界候选和外部主题扫描完成；尚未改正文、运行 demo 或确认 Mermaid 渲染。后续待整册关系图和 coverage map 稳定后实施精修。
