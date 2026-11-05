# 可扩展监督与人类—AI监督：来源证据

检索与阅读日期：2026-11-03。

## 互补标签与分区式人类监督

- **来源与版本：** Ren Yin、Takashi Ishida、Masashi Sugiyama，*Towards Scalable Oversight via Partitioned Human Supervision*，ICLR 2026 正式会议论文。[摘要页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b6f03ce953cf7f07dafa20891d2539e3-Abstract-Conference.html)；[论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/b6f03ce953cf7f07dafa20891d2539e3-Paper-Conference.pdf)。
- **实际阅读位置：** 摘要、方法 §2、互补标签估计式 (2)、选项/专家路由协议及附录数据标注协议。
- **来源支持的事实：** 任务是有 `K≥3` 个选项的多选评估。对每个样本，从 `K−1` 个错误类别中均匀随机抽取一个互补标签；模型预测避开该标签的比例为 `q`。论文给出无偏校正 `A=(K−1)q−(K−2)`。其数据收集协议先均匀打乱选项，再由随机路由到所选类别的专家判断该类别是否适用，从而生成 ordinary label 或 complementary label。
- **证据等级：** verified，正式会议论文，理论推导与论文实验。
- **适用边界：** 无偏关系依赖错误类别均匀抽样、同一评测分布和分类任务设定。专家按主观难度挑选 hard negative、错标签来源系统偏差或标签类别分配不均时，不满足抽样条件。公式说明该协议下怎样估计多选 top-1 accuracy，不等于“排除一项”的比例本身就是准确率，也不能直接套到开放文本、长文档或 Agent 轨迹。
- **数值例边界：** `K=4`、观察到 `q=0.8` 时，代入为 `A=3×0.8−2=0.4`。这是按论文公式构造的手算示例，不是论文实测值。

## Debate 对人类与 LLM 评审的差异

- **来源与版本：** Maria Victoria Carro 等，*Humans Correct Their Judgments Through Debate, but Weaker AI Models Do Not*，arXiv `2610.28539v1`（2026-10-20）；论文 HTML 标注为扩展版已录用 NeurIPS 2026。[arXiv 摘要](https://arxiv.org/abs/2610.28539)；[HTML 正文](https://arxiv.org/html/2610.28539)。
- **实际阅读位置：** HTML 摘要、§3.1 数据集、§3.2—3.4 实验协议、§4 人类结果、§5 LLM 结果、§6 讨论、§7 限制和附录的人类样本保留说明。
- **来源支持的事实：** 研究包括 5 个人类实验、40 多组 LLM 实验，覆盖 10 项科学/技术议题及 100 条作者编写的子主题主张。约 690 人受邀参与，排除后保留 507 条人类判断。指定的同主题人类 debate 提高判断准确率；弱 LLM 评审的结果混合，多裁判协议没有稳定提高 LLM 小组准确率。论文还比较 consultancy、跨子主题迁移、多裁判投票/讨论和人机混合协议。
- **证据等级：** verified，公开会议录用论文的扩展预印本 v1；报告为论文规定任务、模型和协议下的实验结果。
- **适用边界：** 论文明确说明任务结构简单，难度主要来自领域内容而非多步推理或长时程求解；主张由作者撰写且未由领域专家核验。专有模型裁判与 debater 的能力差距较窄，先验信念诱导不完美，能力不对称没有被直接随机操纵。多裁判结果被作者标为初步，模型多样性与 persona 等因素混杂，且只测有限面板大小和协议。结果不估计 Agent 长轨迹监督、最坏情形对抗鲁棒性或真实部署发生率。
- **样本口径：** 正文用 `N=507` 表示分析样本；附录说明约 690 人被招募、排除后保留 507 条响应。写作时应表述为保留的判断/响应，避免误读为 507 条独立实验或 507 条题目。

## LLM-as-Judge 评分几何

- **来源与版本：** Sourabrata Mukherjee、Hamna Abid、Kalika Bali、Sunayana Sitaram，*The Geometry of LLM-as-Judge: Why Inter-LLM Consensus Is Not Human Alignment*，Findings of EMNLP 2026，pp. 36237–36266。[ACL Anthology 正式会议页](https://aclanthology.org/2026.findings-emnlp.1696/)；[arXiv v2](https://arxiv.org/abs/2606.03043v2) 于 2026-09-02 修订。
- **实际阅读位置：** 摘要；PDF §4 实验设置；§6 结论及 Limitation 段；附录数据集、人类参考和 judge roster 说明。
- **来源支持的事实：** 研究使用两个社区构建的 Indic benchmark，覆盖四个领域、八种语言及 42 个 distinct LLM judges（15 个基础模型、27 个微调/偏好优化/few-shot/校准变体）。主观 rubric 上，judge-judge agreement 与 human-human agreement 相近，但 judge 仅达到 58%—66% 的人类一致程度；一个可验证 rubric 的差距大部分缩小。
- **证据等级：** source-backed（Findings of EMNLP 2026 正式会议摘要页和 arXiv v2 正文/附录）。
- **适用边界：** “几何”是对观测评分协方差、得分分布和人类参照关系的诊断抽象，不是模型内部机制解释，也非因果证明。客观区间只依赖一个可验证 rubric；community-health 人类标注合并 15 位标注者且无个体 ID，无法构建人际一致基线；人类参考覆盖不均、语言切片样本规模有限，结果绑定论文数据、rubric、语言和 judge roster。

## 前沿评估 judge 去偏的标签上限

- **来源与版本：** Florian Eddie Dorner、Vivian Nastl、Moritz Hardt，*Limits to Scalable Evaluation at the Frontier: LLM as Judge Won't Beat Twice the Data*，ICLR 2025。[会议摘要页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4264ee4376776907c0b87ed70b959585-Abstract-Conference.html)。
- **实际阅读位置：** 会议论文摘要。
- **来源支持的事实：** 在 judge 不比被评模型更准确的条件下，论文理论结果称任何去偏方法将所需 ground-truth labels 减少的幅度不可能超过一半；作者的经验结果显示实际节省更少。
- **证据等级：** source-backed，ICLR 2025 正式论文摘要。本记录依据摘要支持的定理范围，不扩展到具体 estimator 的实现条件。
- **适用边界：** 这是关于论文所分析的去偏框架和准确度条件的理论上限，不表示任何项目都能节省 50% 标签，也不是校准集规划公式。

## 自我评估和基础监督路线

- **Language Models (Mostly) Know What They Know：** Anthropic 等作者，arXiv `2207.05221v4`，2022-11-21。[v4 HTML](https://arxiv.org/html/2207.05221)。实际阅读位置：§§4.1、5.1—5.2、附录 A.2—A.3。研究针对候选答案的 `P(True)` 与不依赖某个具体候选答案的 `P(IK)`；论文以多个 `T=1` 样本的正确比例操作化题目层面的 `P(IK)`，并在 TriviaQA 训练后测试 LAMBADA、算术、GSM8K、Codex HumanEval 与 Python 函数合成任务。作者报告跨任务区分能力有一定泛化，但 `P(IK)` 的 OOD calibration 往往较差，若干切片欠自信，扩充任务数据后校准亦会变化。证据等级：verified。边界：结果依赖论文的模型、任务、采样格式和校准协议；模型自报置信度不是独立 verifier。
- **Iterated Amplification：** Christiano、Shlegeris、Amodei，arXiv `1810.08575v1`。[HTML 正文](https://arxiv.org/html/1810.08575)。实际阅读位置：摘要、§1.1、§§2.1—2.5、实验 §§4.1—4.3、§5。来源支持：人类专家可调用当前模型的多个副本组成复合系统，模型学习模仿该放大监督者；论文讨论 human predictor、训练动态和问题分解。实验使用算法化合成任务与简化实现；作者指出已训练模型未必照着人类分解执行，可能学整体启发式。证据等级：verified。边界：作者将现实复杂任务上的有效分解留作未来工作，文中现实场景为论证示例，不是生产实证。
- **AI Safety via Debate：** Irving、Christiano、Amodei，arXiv `1805.00899v2`。[HTML 正文](https://arxiv.org/html/1805.00899)。实际阅读位置：摘要、§2 辩论游戏、§3.1 MNIST 实验、§5 主要理论与实践顾虑。来源支持：两方交替给出受长度限制的短陈述，由人类裁判选更真实/有用信息的一方；稀疏 MNIST 分类器实验在 6 像素设置从 59.4% 到 88.9%，4 像素设置从 48.2% 到 85.2%。证据等级：verified。边界：是特定数据和预commit协议下的初步分类实验；裁判能否理解复杂论证、诚实是否最优、训练是否稳定等仍是论文明确提出的问题，不等于真实复杂监督效果。
- **Recursive Reward Modeling：** Leike 等，*Scalable agent alignment via reward modeling: a research direction*，arXiv `1811.07871v1`。[HTML 正文](https://arxiv.org/html/1811.07871)。实际阅读位置：§3.2（setup、examples、discussion）、§4.3—4.5。来源支持：较窄任务训练出的 `A_{k-1}` 可辅助用户评价更一般、更复杂的 `A_k`；辅助可包括补充信息、摘要、内部状态解释和用户拆出的子任务。评价分解需要穷尽重要维度，否则未评估方面可能被任意优化；层级间错误会否累积或可自我纠正仍是开放问题。作者将 sparse/poorly shaped/wrong-scale reward、训练停滞、探索不足、非预期行为、分布变化和对抗输入列为 reward-result gap 的可能来源。证据等级：verified，方向性研究论文。边界：不是已验证的生产配方，不能由层级数量推断奖励或策略已对齐。
- **Constitutional AI：** Bai 等，arXiv `2212.08073v1`。[HTML 正文](https://arxiv.org/html/2212.08073)。实际阅读位置：§1.2、§2、§§3.1—3.3、§§4.1—4.4。来源支持：两阶段流程分别是 critique→revision→监督学习，以及 AI comparison→preference model→强化学习；论文还专门评估 harmlessness 与 evasiveness。红队提示数据包含 42,496 条人工编写与 140,335 条 few-shot 模型生成提示，共 182,831 条，并为提示抽样四组 critique-revision 对。证据等级：verified。边界：数据和结论绑定论文模型、原则、提示集与训练流程；AI 生成提示不表示逐条人工审定，也不证明任意原则列表或 RLAIF judge 无偏。
- **Learning to Summarize from Human Feedback：** Stiennon 等，arXiv `2009.01325v3`。[摘要页](https://arxiv.org/abs/2009.01325)。在 Reddit TL;DR 数据上收集人类比较、训练偏好模型作为 reward，再对摘要策略做 RL。结果支持该数据/模型设置中的路线可行性，不证明 reward proxy 在其他任务中可靠。证据等级：source-backed（摘要）。
- **Let's Verify Step by Step：** Lightman 等，arXiv `2305.20050`。[摘要页](https://arxiv.org/abs/2305.20050)。摘要报告在 MATH 数学题上的过程监督与结果监督实验；78% 的 test subset 结果属于论文指定训练和评估协议，不自动迁移至任意 Agent 行为。证据等级：source-backed（摘要）。
- **Weak-to-Strong Generalization：** Burns 等，arXiv `2312.09390`。[摘要页](https://arxiv.org/abs/2312.09390)。摘要描述 GPT-4 系列预训练模型在 NLP、棋类与 reward modeling 中的弱监督实验，强模型在论文设置内超过弱监督者。结论不能外推为任意监督信号、任务或模型都会如此。证据等级：source-backed（摘要）。

## 监督模拟中的能力关系

- **来源与版本：** Joshua Engels 等，*Scaling Laws For Scalable Oversight*，NeurIPS 2025。[会议论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/8c3caae2f725c8e2a55ecd600563d172-Paper-Conference.pdf)。
- **实际阅读位置：** PDF 摘要、§1 概览和实验游戏说明。
- **来源支持的事实：** 使用修改版 Nim 验证框架，研究 Mafia、Debate、Backdoor Code、Wargames 四类监督博弈，并对 nested scalable oversight 进行理论分析。论文摘要中的 Elo 差 400 时成功率（Mafia 13.5%、Debate 51.7%、Backdoor Code 10.0%、Wargames 9.4%）来自这些游戏与作者模型化配置；作者报告监督更强系统时成功率进一步下降。
- **证据等级：** verified，NeurIPS 2025 正式会议论文。
- **适用边界：** 这些是模拟博弈、Elo 映射和理论建模结果，不是生产系统的 oversight 成功率，也不能用来为开放任务预测概率。
