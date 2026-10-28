# 第九册第八章 Preference Data 与安全数据：精修研究记录（2026-10-28）

## 范围与章节决策

按当前 `WRITING_SPEC.md` 从头复读并实际修改第九册第八章。复用第八册第五章维护条目的复核字段组织本记录。第七章负责 synthetic 来源和 distillation 关系，本章负责偏好比较、目标动作、安全数据质量和行为评估；第八册安全总览、红队评估章提供策略与威胁模型背景，本章不重复完整的安全 taxonomy、攻击评估或运行时防护。

保留单篇章节和现有编号。章首补充标准 Mermaid 数据审核闭环，所有节点标签加引号，带文字的虚线箭头使用 `-.->|标签|`。偏好数据 schema 保留原始标注协议与分歧状态；安全标签拆为意图/可回答性、后果严重度、领域、语境和目标动作等独立轴。教学 demo 中的 `slice` 只是非穷尽测试切片，不代表生产风险模型。

## 研究扫描与证据边界

- [KTO](https://arxiv.org/abs/2402.01306) 的公开方法使用单条回答的 desirable/undesirable 反馈；据此修正“偏好数据必然成对”的表述。DPO 和传统 reward modeling 的成对比较要求另行说明。
- ICML 2025 [Diverging Preferences](https://proceedings.mlr.press/v267/zhang25bx.html) 将标注者分歧归纳为高层类别和细分情形，指出任务欠定、风格等因素会造成合理差异。正文据此要求保留平局、不可比和需要上下文的原因，不把低一致度自动等同于脏标签。
- NeurIPS 2025 [PrefCleanBench](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b457a310c34efcaf6322d1bc6503ce16-Abstract-Datasets_and_Benchmarks_Track.html) 比较了 13 种清洗方法及多种数据、模型和目标设置。多评审过滤在论文评测条件下表现较好，不能据此宣布普适清洗规则。
- AACL/IJCNLP 2025 [Judging the Judges](https://aclanthology.org/2025.ijcnlp-long.18/) 涵盖 15 个 judge、22 项任务和超过 150,000 个评估实例。正文加入候选顺序交换、人工校准切片和高风险升级复核；论文结果不外推至所有 judge。
- ICLR 2025 [Trust or Escalate](https://proceedings.iclr.cc/paper_files/paper/2025/hash/08dabd5345b37fffcbe335bd578b15a0-Abstract-Conference.html) 的 selective escalation 保证依赖人工校准与特定假设，正文明确其条件性。
- [Human Preferences for Constructive Interactions](https://arxiv.org/abs/2503.16480) 的预印本样本覆盖 74 个国家、7,500 余段对话及 21 个模型；正文用它说明偏好表达具有群体和语境差异，不把抽样结果当成普适偏好。
- MRL 2026 [Cross-Lingual (Non-)Transfer of the Refusal Direction](https://aclanthology.org/2026.mrl-1.19/) 研究两个开源指令模型、十种语言和指定安全 benchmark。正文据此强调按目标语言验收，未声称跨模型普遍机制。
- InstructGPT、DPO、HH RLHF、Constitutional AI、Summarization from Human Feedback、Learning from Human Preferences 和 LLM red-teaming 来源继续作为历史与机制入口；具体来源和适用范围见章节 §27。

## 正文与教学材料改动

- 明确 synthetic/distillation 是来源血缘，preference/safety 是行为信号；把已批准安全策略映射到字段、标注规则、分母和版本证据。
- 将偏好分歧从默认过滤噪声改为可解释数据：保留 `tie`、`not_comparable`、`needs_context` 等状态及原因；区分 rubric 不清、数据质量缺陷和真实价值/风格差异。
- 重写安全标签分层，拆开请求意图、后果严重度、专业领域、语境与目标动作；“回答、澄清、拒绝、安全替代”用独立切片核对。医疗、法律和金融数据的建议落到来源、时效、法域/任务范围和专业复核。
- 区分样本级 target-action retention 与模型行为正确率；目标切片覆盖只在安全样本上计算。PII、污染、分歧和长度异常被隔离待复核，不假设永久删除。
- 补充数据来源授权、隐私及红队样本最小化；偏好质量、模型行为和系统副作用分别评估。LLM judge 使用随机顺序/交换、人工校准及高风险升级。
- 把偏好定义为给定任务、rubric、群体和时间条件下的分布；安全策略由负责方定义，不从用户偏好直接推出。
- 修正 LaTeX 公式围栏和变量/分母解释。新增 reward-model、DPO、pairwise agreement、误拒/漏拒、长度偏置提示量的逐步手算；未定义分母表示 N/A。
- Python 教学审计 demo 增加澄清动作样本；修正动作覆盖和 retention 的口径，按未舍入 margin 判门槛，校准汇总 margin、报告输出和断言。固定输入输出与手算一致；结论仅表示可进入 preference ablation，不表示模型安全通过。

## 篇幅与数值核对

本 Markdown 文件承担一篇核心讲解。排除标题、代码/公式/图示围栏、表格、引用清单、行内代码/公式和状态文字，解释性正文 **10,936 个汉字**，高于 1,800 字下限。篇幅统计不能代替连续阅读。

手算例：reward margin 为 `1.3-0.7=0.6`，`-ln σ(0.6)≈0.437`；DPO 示例中参考差与当前差之差为 `-0.1`，`β=0.5` 时损失约 `0.718`；动作一致度 `4/5=0.8`，不含机会校正；目标行为示例中 over-refusal 为 `3/4=0.75`，under-refusal 为 `1/5=0.20`；长答案占比 `3/8=0.375` 只作为长度偏置提示，不作为因果量。均为教学数字。

demo 固定输入得到总 token retention `4,840/8,010≈0.604`，preference `1,580/4,190≈0.377`，safety `3,260/3,820≈0.853`；目标动作样本保留比例分别为 answer `0.5`、clarify/refuse/safe_alt `1.0`；五个 safety 切片覆盖 `1.0`。逐项输出和正文预期输出相同，九项检查均为真。

## 代码、视觉、链接与配套检查

- 唯一 Python 可执行块于 2026-10-28 在 Python 3.12.9 标准库下运行；固定合成输入输出与正文块逐行匹配，内嵌断言通过。已检查空比率、空均值和关键一致性断言；未运行测试套件或连接训练系统。
- 唯一 Mermaid 围栏使用标准语法、引号节点及管道文字虚线箭头。源码静态检查通过；GitHub 页面实际 Mermaid/MathJax 渲染仍开放。
- 两条跨册链接修正为从本章目录到第八册的正确相对路径；本地相对链接静态检查无缺失。未对所有历史外链执行 HTTP 全量扫描。
- 已同步 `PAPERS.md`、`EXERCISES.md`、`INTERVIEW_BANK.md`、`GLOSSARY_EN_ZH.md`，并在本章目录更新问题与产出。
- `git diff --check` 通过；本章 22 个 fenced blocks 成对，数学语法静态检查未发现 `$$`、`\(...\)` 或 `\operatorname`。

## 未完成门禁与风险

尚未核验 GitHub 上的 Mermaid/MathJax 实际渲染、全章外链 HTTP 状态和第九册连续复读；没有目标语料、目标模型、训练配方、标注平台或生产流量验证。论文结论、抽样发现和固定 toy demo 均不证明项目生产收益、安全性或最优筛选阈值。章节级本轮修改完成，书册完整门禁保持开放。
