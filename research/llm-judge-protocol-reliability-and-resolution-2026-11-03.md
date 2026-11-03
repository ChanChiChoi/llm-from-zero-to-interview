# LLM Judge 协议、重复性与判别力研究

研究日期：2026-11-03

## Pairwise 与逐条评分对干扰的敏感性

- **来源：** Tripathi, Wadhwa, Durrett, and Niekum, *Pairwise or Pointwise? Evaluating Feedback Protocols for Bias in LLM-Based Evaluation*, arXiv:2504.14716v2，2025-08-21。阅读原文第 3–6 节及附录局限说明。https://arxiv.org/abs/2504.14716
- **支持事实：** 论文比较 pairwise 偏好和 1–7 分 absolute scoring，并向候选回答施加自信语气、冗长或迎合等 distractor。MT-Bench 的受控改写把原先未获偏好的回答加入扰动后重新判分；四个可读 logits 的开源 judge 在三种扰动上的平均偏好翻转约为 35.2%，absolute-score 变化约为 9.3%。IFEval-TweakSet 则构造可验证的同质量回答，并比较两种协议是否保留相同评分或平局。
- **证据边界：** 翻转率衡量相对同一评审器基线对受控扰动的敏感性，不是对人工 gold 的准确率。协议、模型、任务与生成干扰均受论文设置限制；文字输出 API judge 的结果另列，不能和四个开源 judge 的汇总混算。不能据此说绝对评分普遍优于成对偏好。

## Agent 长输出中的逐 rubric 核验

- **来源：** Peng et al., *Can LLM-as-a-Judge Reliably Verify Rubrics in Agentic Scenarios?*, EMNLP 2026。阅读 ACL Anthology 正式论文第 1、3、4 节及附录实验设置。https://aclanthology.org/2026.emnlp-main.2192/
- **支持事实：** RuVerBench 包含 2,458 条输出—单条 rubric 判断，分属 Deep Research（1,615）和 Agentic Coding（843）；每条标签表示输出是否满足该条 rubric。专业标注团队与独立内部团队双重标注，总体一致率 90.4%，Cohen κ 为 0.808，分歧再基于相应输出和 rubric 仲裁。Deep Research 输出平均约 7.1K tokens，Agentic Coding 轨迹平均约 49.4K tokens。18 个 judge 的结果在两个领域和类别间不同；Deep Research 的最佳 Avg BAcc 为 94.7，Agentic Coding 的最佳为 89.4。作者将典型错误归纳为把片段证据当作完整满足，以及向 rubric 添加未要求的条件。多 rubric batching 和重复多数投票的影响依赖领域与模型。
- **证据边界：** 数据仅覆盖两个具代表性的 Agent 领域和其 rubric 来源。类别结果、最佳模型分数和 prompt/batching 效果绑定论文模型快照、分组、标签和实验子集；不代表自然用户任务或其他 Agent 系统的总体 judge 失误率。

## 去偏提示、平局与测量分辨率

- **来源：** Zhao, Wang, and Chen, *Debiasing as a Measurement Intervention: Calibrated Ties and Resolution Loss in LLM-as-a-Judge Evaluation*, Findings of EMNLP 2026。阅读论文摘要、问题设定、指标、结果与局限。https://aclanthology.org/2026.findings-emnlp.1880/
- **支持事实：** TraceJudgeBench 是面向 RAG 与 agent-workflow 中 citation-like 展示线索的诊断基准。论文把目标关系区分为内容等价、软边界和人工验证的中等质量差。GPT-5.5 在某个经人工验证的软质量差设置下，普通提示对较差引用式回答的胜出率为 50.5%；更强的 anti-citation 提示可降至 0，但一些提示强度也让真实中等差异变成 Tie。清晰正确性冲突上的报告准确率至少为 93%。论文把去偏视为测量干预，建议联合检查呈现线索抑制、正确性保留、质量判别力、按目标关系拆分的 Tie 与协议成本。
- **证据边界：** TraceJudgeBench 是受控诊断集，包含模板构造及经人工校验的数据，不是通用 RAG/Agent 排行榜。完整矩阵主要使用三个商业 judge；开源 judge 只覆盖部分实验。结果不说明所有 RAG 系统中的 citation bias 或分辨率损失比例。

## 重复调用稳定性与双向顺序偏差

- **来源：** Yagubyan, *The Coin Flip Judge? Reliability and Bias in LLM-as-a-Judge Evaluation*, Proceedings of GroundLM 2026，2026-10-29。阅读 ACL Anthology PDF 的方法、结果、建议与威胁有效性部分。https://aclanthology.org/2026.groundlm-1.64/
- **支持事实：** 主实验在 29 个刻意选为近似质量匹配的问题上，对六个 judge 重复进行 pairwise 和 pointwise 评审；pairwise 标签的平均重复翻转率约 11%–14%。双向换序实验覆盖五个 judge，共 8,700 次判断；按相同展示位置胜出的双向 order-driven 比例约 21.1%–46.0%。例如 GPT-5.4-mini 的首位胜出比例为 33.5%，但按同一展示位置选择的比例为 33.9%；只报告首位偏好方向会掩盖其第二位偏好。论文从该样本内的可靠性曲线估计，约 11 次调用平均可复现 50 次调用的多数标签达到 95%，困难项可能需要更多调用。
- **证据边界：** 重复性不等于对人工 gold 正确；研究样本刻意选择近似质量回答，翻转率不能外推到明显胜负的样本。问题数为 29，跨类别样本数少，类别差异仅属探索性。11 次调用是该实验内估计，不是通用采样规则；跨代结果不足以证明模型版本等价或所有 prompt 都无法改进。

## Pairwise judge 指标与比较难度

- **来源：** Brocai and Becker, *Pair Difficulty Matters: Rethinking Pairwise LLM-as-a-Judge Evaluation and Consistency*, arXiv:2609.37577v3，2026-10-27。阅读全文理论、模拟、两组实验及局限。https://arxiv.org/abs/2609.37577
- **支持事实：** 作者基于 Bradley–Terry 几何讨论位置偏差、传递性与成对一致性为何可能被近似等强度的回答对主导，而这些比较对总体排序影响相对小；更大 rank gap 的错判更直接影响排名。模拟和两份 human-rated 英语教育文本语料（CLEAR 可读性、ASAP 2.0 学生作文，各抽取 200 项）中，全局一致性等 proxy 与 gold ranking 的关系较弱，按远 gap 分层的一致性与排序相关较强。论文没有提出适用所有数据的单一 gap 指标。
- **证据边界：** 预印本实验只覆盖 pairwise 判断、两个教育语料和有限 judge；其中 ASAP 2.0 使用 pointwise 人评分数近似潜在强度，CLEAR 使用既有 Bradley–Terry 分数。结论受人类 gold 构念、数据域、抽样和 gap 估计影响，不能直接推广到多选/listwise judge、代码、问答或生产 leaderboard。作者也明确没有推荐一种通用 gap 估计方法。

## Rubric 评分档位与维度顺序偏差

- **来源：** Xu et al., *Am I More Pointwise or Pairwise? Revealing Position Bias in Rubric-Based LLM-as-a-Judge*, Findings of EMNLP 2026。阅读 ACL Anthology 正式论文的问题定义、实验设置、结果与附录。https://aclanthology.org/2026.findings-emnlp.1444/
- **支持事实：** 研究区分两种 rubric 顺序效应：评分档位在选项表中的位置，以及多维 rubric 中评价维度的排列位置。六个开源权重 judge、四个数据集上的实验发现，档位位置偏好方向因模型而异，既有偏向第一项也有偏向末项；多维度提示的维度顺序也会改变分数。作者报告仅改变 rubric 顺序会使其配置中 15%–41% 的 prompt top-1 候选发生变化。按预算匹配后，随机置换聚合与精确平衡置换在多数比较中难以区分；与固定顺序相比，人类相关性提升主要出现在原先偏差较强的 judge 上。
- **证据边界：** 六个模型是 GPT-OSS、Qwen 与 Gemma 的指定快照；数据为 MT-Bench、Vicuna-Bench、HANNA 和 SummEval。top-1 翻转比例、偏差显著性及人类相关变化绑定这些候选、rubric、指标和评测集；不能外推至所有模型或 rubric，也不表示置换必然提高正确率。作者建议的小量随机聚合是论文设置下的成本权衡，不是固定通用调用次数。

## Rubric 意图迁移与人工分数量尺校准

- **来源：** Hong et al., *From Rubrics to Reliable Scores: Evidence-Grounded Text Evaluation with LLM Judges*, EMNLP 2026。阅读 ACL Anthology 正式论文的问题设定、方法、四组实验、校准消融和局限。https://aclanthology.org/2026.emnlp-main.566/
- **支持事实：** 论文把人类评分标准到冻结 LLM 判分协议的转化称为 criteria transfer。RULERS 分三阶段处理：把原 rubric 整理并锁定为任务级 traits、checklist、分数边界和证据规则；让 judge 输出带有输入文本依据的结构化逐项判断，并按证据类型作可机械检查；只用人工标注 calibration split 学习分数对齐，再在 held-out split 评估。论文在 ASAP 2.0、SummHF、DREsS 与 WebNLG、以及 GPT-4o/mini 和 Llama-3.1-8B/70B 上测试；多数但不是全部设置中与人工分数的 QWK 更高，且 semantically equivalent rubric 改写下较稳定。主实验使用 200 条人工校准样本，数据效率分析还比较 50、100、150、200 条；将同一校准方法用于若干 baseline 并不能稳定复现 RULERS 的全部收益，组件消融也显示锁定、证据和校准共同起作用。
- **证据边界：** 研究对象是四个 rubric-governed 文本评分 benchmark 和指定冻结 judge，不是 Agent 轨迹评测或生产质量保证。有效校准仍依赖源 rubric 质量、人工校准标签和对应任务的 score distribution；迁移到新 rubric、领域或分数约定时通常要重新规格化并校准。论文结果不表示 200 是通用的最小样本量，也不证明自动评分可替代高风险场景中的专家判断。
