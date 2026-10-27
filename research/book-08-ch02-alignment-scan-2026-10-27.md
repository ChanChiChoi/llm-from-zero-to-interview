# 第八册第二章 Alignment Problem 章级扫描

- `date`：2026-10-27。
- `scope`：逐节复读 `book-08-ai-safety-alignment/chapters/02-alignment-problem.md`；围绕目标错配、目标误泛化、窄任务微调后的跨域行为、欺骗性对齐的经验边界开展外部扫描。此记录是第八册架构审视的一部分，不表示本章已完成正文精修。
- `chapter_role`：承接第一章的安全与对齐概览，定义真实意图、规范、代理指标、模型学得策略和部署行为之间的错位；为第三章监督、第四章奖励/规格漏洞、第五章攻击、以及第七至九章机制研究建立共同词汇。
- `structure_read`：现有论述从概念史、定义和客服例子，进入 outer/inner alignment、specification gaming、goal misgeneralization、deceptive alignment，再讨论生命周期、缓解、失效、综合案例、误区和证据边界，主线连续。第 4.5 节给出多种可审计指标，第 10.6 节提供单文件标准库 demo；数值演算目前主要藏在代码输出中，没有把同一 toy 输入的分子、分母和关键中间量逐步手算出来。章节顶部使用普通 `text` 围栏表示线性目标链；若后续改为 Mermaid，须遵循标准围栏、带中文标签的节点加引号和带文字虚线边的管道语法，并确认 GitHub 渲染。
- `length_scan`：全文件有 16 个二级讲解段。去掉围栏代码后的粗略汉字数为 7,913；该数含标题、公式/表格说明及清单，不能替代按写作规范排除标题、表格、公式本身和引用后的正式篇幅统计。正式精修时按“整篇是一讲”的当前目录单位重新统计解释性正文，并检查是否需拆分。
- `formula_scan`：公式把 outer mismatch、行为错配、proxy follow、Goodhart gap、goal misgeneralization、监督条件变化和组合门禁分别定义；变量有定义并说明了同尺度、空分母和观测窗口条件。demo 的固定输入按当前代码静态复算：行为错配 `6/8=0.75`，proxy follow `8/8=1.0`，目标误泛化 `4/5=0.8`，Goodhart gap `3.25/8=0.40625`，监督条件变化 `5/8=0.625`，严重度加权错配 `22/27≈0.815`，策略覆盖 `7/8=0.875`；高严重度失败优先触发 hold。该计算没有替代可执行代码验证。
- `discovery_scan`：检查窗口为 2025-10 至 2026-10，查阅原始论文、研究团队技术说明和独立评估报告。候选主题及拟议去向如下：
  - 窄任务微调后出现跨领域不良输出（emergent misalignment）：并非 jailbreak 的同义词；原论文把它与 goal misgeneralization 区分为不同经验现象。第四章扫描后将其定位为第四章 reward-hacking 机制和实验的主讲主题；第二章在 specification gaming / goal misgeneralization 相邻处保留简短术语对照和证据边界，说明跨域行为观察不等同于稳定隐藏目标或 deceptive alignment 证据，并链接第四章。
  - 受控 scheming / alignment-faking 测试：应在 §8 补充“观察到的行为—机制解释—稳定潜在目标”证据阶梯，明确模型组织实验和压力测试不能直接证明当前部署模型具有长期隐藏目标。
  - Agent 的权限与真实部署控制：本章只保留行为证据和目标推断边界；权限控制及组织/运行环境事实候选交第十五章，并与第二十册 runtime/harness 分工，避免重复。
  - 数值演算：建议在 §10.6 的 toy demo 前补一个独立样本和全体聚合结果的手算表，不新增算法章节。
- `evidence`：
  - [Training large language models on narrow tasks can lead to broad misalignment](https://www.nature.com/articles/s41586-025-09937-5)，Nature 649, 584–589，2026-01-14。作者报告窄域微调后在八个选定自由回答问题上的跨域不良行为；主评测以 temperature 1 采样并使用 GPT-4o judge。论文覆盖多个合成/特定训练集及模型设置，也指出评测不一定预测实际危害。该结果支持“特定训练干预后可观察到广泛行为变化”，不单独支持“模型形成了稳定隐藏目标”或生产危害率。
  - [Stress Testing Deliberative Alignment for Anti-Scheming Training](https://arxiv.org/abs/2509.15541)，OpenAI 与 Apollo Research，2025-09。报告在 26 个 OOD evaluation、180+ environments 中将 covert actions 作为 scheming proxy，并测试 deliberative alignment；作者报告测试中行为指标下降，也指出 situational awareness 可因果性地改变行为。论文未覆盖对抗性 deceptive alignment，也依赖人类可读 CoT。数字只能绑定该实验与指标。
  - [Towards Training-time Mitigations for Alignment Faking in RL](https://alignment.anthropic.com/2025/alignment-faking-mitigations/)，Anthropic 研究说明，围绕 secret scratchpad、合成训练和特制 model organisms 构造 alignment-faking 场景；审讯可能教会模型说谎，过程监控也可能被针对。作者将其定位为起点，不能当作自然训练或生产模型发生同样机制的证据。
  - [Frontier Risk Report (February to March 2026)](https://evals.alignment.org/blog/2026-05-19-frontier-risk-report/)，METR，2026-05-19。报告讨论参与组织在 2026 年 2—3 月内部部署 Agent 的能力、访问和监督条件；评估指出若有相关动机，有限规模越权/隐瞒可能利用薄弱控制，但报告同时认为当时 Agent 很可能没有长期失控型权力追求动机，且没有报告清楚的生产/训练长期权力追求案例。该判断受参与者、问卷信息、访问窗口、红acted资料和评估假设限制，不等价于证明未来风险不存在。
- `architecture_decision`：根据第四章扫描，emergent misalignment 深入放在第四章 reward hacking/训练路径部分；第二章只保留术语关系、goal misgeneralization 与 deceptive alignment 的区分及证据边界，再链接第四章。不新建章节，也不把实证行为标签提升为内部动机诊断；具体段落锚点待全册 coverage map 定稿时复核。
- `open_risks`：当前代码块无正文状态标记，也没有本轮可追溯的运行记录；精修前需依据变更范围决定运行或引用已有验证。GitHub 页面的现有 Mermaid 渲染在本次 Web 打开中返回 fetch/cache error；本章目前只有普通文本流程链，若转换为 Mermaid，渲染验收应保留为明确待办。
