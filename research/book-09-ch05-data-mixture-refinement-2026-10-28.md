# 第九册第五章：Data Mixture 与配比精修研究记录（2026-10-28）

## 审查范围与章节决定

按当前 `WRITING_SPEC.md` 从头复读 `book-09-data-engineering/chapters/05-data-mixture与配比.md`，包括公式、手算、Mermaid、教学代码、参考资料和跨章交接。章节契约是把第二至第四章得到的清洗/去重数据池转成可审计的采样分布与预算，并向第六至第九章专项数据分支交接。保留单篇结构：职责连续，现有主题可通过区分“池级配比、样本级选择、样本顺序和实际曝光”而收束，无须拆章或迁移核心内容。

候选方法已纳入方法谱系和本章实验设计。多语言规模关系进入 §9；有限数据重复暴露进入 §17；样本级全局选择、代理优化、回归拟合和在线更新进入 §15；特定配比下知识获取的非线性进入 §20。研究结果只支持其对应实验条件，不转写成通用配方或风险率。

## 定向研究与证据边界

来源优先采用正式会议论文页、ACL Anthology 和 arXiv 预印本页。论文和预印本中的规模、性能与因果解释均保留作者实验范围。

| 来源 | 支持的正文判断 | 适用范围与限制 |
| --- | --- | --- |
| [Data Mixing Laws（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cc84bfabe6389d8883fc2071c848f62a-Abstract-Conference.html) | 以小规模配比实验拟合数据混合与语言模型表现关系，是“函数拟合”方法的代表。 | 结果绑定论文数据、proxy 模型、token 预算和评估目标，不给出通用固定比例。 |
| [Scaling Laws for Optimal Data Mixtures（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/bc1d640f841f752c689aae20b31198c1-Abstract-Conference.html) | 研究联合建模配比、训练规模与验证表现；LLM 设置包含 SlimPajama 的 7 个文本域，模型规模约 186M–7B，并报告指定条件下的跨规模/未见配比预测。 | 使用固定 mixture 和可持续供数等假设，不建模稀缺池重复曝光，目标主要是通用验证 loss；不能直接外推到动态排序或具体下游能力。 |
| [Scaling Laws for Multilingual Language Models（Findings of ACL 2025）](https://aclanthology.org/2025.findings-acl.221/) | 超过 100 个模型、23 种语言和 5 个语系的研究，在其设置中发现语系 loss 与本语系采样比例关系较强，并报告从 85M proxy 到 1.2B 的配比迁移。 | 语系内比例的拟合不等于语言之间互相独立，也不构成通用语言预算表。 |
| [ATLAS（arXiv 预印本）](https://arxiv.org/abs/2510.22037) | 汇总 774 次、10M–8B 模型实验，覆盖 400 多种训练语言、48 种评估语言及 1,444 个语言对，提供迁移/干扰需分语言测量的案例。 | 规模和语言覆盖不消除不同数据质量、tokenizer、训练配方及评估设计带来的边界；属于预印本证据。 |
| [SampleMix（Findings of EMNLP 2025）](https://aclanthology.org/2025.findings-emnlp.741/) | 文档级质量/多样性评分后的全局样本选择，是不先固定领域池比例的代表方法。 | 论文在 SlimPajama、1B LLaMA、100B token 的设置中排除 GitHub 域；作者报告其任务集上达到 baseline 平均准确率所需步数约为基线的 `1/1.9`，并说明超参数依赖数据集。不能外推为新语料的通用步数收益。 |
| [Rethinking Data Mixture（Findings of EACL 2026）](https://aclanthology.org/2026.findings-eacl.15/) | 为固定启发式、代理/算法优化、函数拟合及在线方法建立分类地图。 | 分类不代表各方法在同一模型、数据和预算下做过直接对照。 |
| [Scaling Laws for Mixture Pretraining Under Data Constraints（2026 arXiv 预印本）](https://arxiv.org/abs/2605.12715) | 超过 2,000 次训练、101M–805M 模型和德语/法语/斯瓦希里语、数学、科学论文、Wikipedia 等设置中，报告若干混合条件下目标池最优重复因子达到 15–20。 | 这是有限池与持续新数据共同训练的特定结果，仍为预印本；不是项目重复上限或通用 epoch 建议。 |
| [Data Mixing Can Induce Phase Transitions in Knowledge Acquisition（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/eee7ae5cf0c4356c2aeca400771791aa-Abstract-Conference.html) | 合成传记知识与网页的受控实验展示特定容量/配比下可能出现阈值式记忆行为，支持 mixture 结果不必线性。 | 合成任务现象不能推广成真实语料普遍存在同一相变。 |
| [DoReMi](https://arxiv.org/abs/2305.10429)、[RegMix](https://arxiv.org/abs/2407.01492)、[GRAPE（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e7bba023351f8262c22e3328e42645ec-Abstract-Conference.html)、[FastMix（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/01d845baaa31b1043e5f280592d57dc4-Abstract-Conference.html) | 作为代理损失、目标权重、配比回归拟合和在线更新等方法族的代表。 | 每种方法的目标、proxy、反馈集和计算成本不同，名称相近或最终权重相似不等于证据可比。 |

候选去向：温度/自然采样保留为透明基线；proxy 与回归方法保留用于代理实验；FastMix 作为在线方法且强调独立留出评估；SampleMix 归入样本级选择，不与池级配比混称；排序/训练阶段单独记录为 schedule；多语言迁移和重复暴露作为条件化章节补充。没有证据支持统一最佳比例，因此未加入固定推荐表。

## 主要修订和手算复核

- 章首由 ASCII 箭头替换为 Mermaid 反馈图，区分数据池、配置权重、计划预算、采样日志和多维评估。
- §3 区分池级 `p_k`、样本级选择、训练顺序和实际日志曝光；§12 增加预训练、继续预训练、SFT、偏好和安全训练的单位/分母/观察项矩阵。
- §4 给出 natural share、平滑权重、质量/目标/风险启发式、预算分配、effective epoch、风险/漂移指标及其定义域，并明确能力线性覆盖只是 proxy。加入 NeurIPS 2025 scaling-law 研究的适用假设。
- 三池例：`n=(800,000,100,000,100,000)`、`alpha=0.5`，权重平方根约为 `894.4/316.2/316.2`，归一化为 `0.586/0.207/0.207`；`B=1,000,000` 时计划约 `585,800/207,100/207,100` token，对应 `0.73/2.07/2.07` 个有效暴露轮次。数值说明平滑，不代表最优比例。
- §14 两池语言例：`900/100` token、`tau=2`，平方根权重 `30/10`，采样率 `0.75/0.25`，保留不同系统 temperature 定义可能相反的说明。
- §19 去重例：网页 `600,000`、代码 `400,000` token 原始自然比例 `60%/40%`；网页去重后 `300,000`，新快照比例约 `42.9%/57.1%`；百万 token 预算下约曝光 `428,600/571,400`。说明自然比例变化即使无需人工调权，也必须刷新重复率和实际采样审计。
- §20 假设 web/code 方向梯度为 `1/-2`：权重 `0.8/0.2` 得 `0.4`，改为 `0.6/0.4` 得 `-0.2`；强调这是单方向教学假设，不预测 benchmark。

## 教学代码验证

- 位置：§21.1 唯一 Python 教学代码块。
- 变化：补足池字段和有限区间校验；拒绝重复名称、非有限/负 mixture 权重；加入确定性的最大余数 token 分配，使整数化后的计划预算精确守恒；上采样倍数由未舍入权重计算；增加总预算断言及失败输入断言。代码旁状态为“已验证（2026-10-28；Python 3.12.9，标准库）”。
- 运行：从 Markdown 提取唯一 `~~~python` 区块并用 Python 3.12.9 标准库执行；没有增加测试文件或运行测试套件。
- 输入：8 个合成池、5 个 toy 能力维度、总预算 200,000 token。预期与实际均输出 `planned_total=200000`、目标能力 `{general: 0.591, code: 0.218, math: 0.285, multilingual: 0.256, safety: 0.212}`、平均质量 `0.823`、风险 `0.045`、空 watchlist 和 `continue_to_ablation`。
- 边界：空池、零 token、重复池名和 `NaN` 权重均按预期拒绝；计划配额之和断言等于预算；正文预期输出与执行结果一致。

## 配套同步与开放门禁

- `PAPERS.md` 增加第九册第五章方法来源入口；`EXERCISES.md` 的数据池设计、采样比较和 ablation 题加入层级/分母、整数预算、独立验证切片和重复曝光要求；`INTERVIEW_BANK.md` Q30 扩展 DoReMi/GRAPE、RegMix/Data Mixing Laws、FastMix、SampleMix 方法分工；`GLOSSARY_EN_ZH.md` 新增 SampleMix 与 Mixture Scaling Law。
- `progress_v2.md` 按第八册第五章字段登记本轮范围、决定、研究扫描、篇幅、手算、代码、图表、链接、配套与未解决风险。
- 本地章节链接静态检查均存在；章节没有维护文件名残留；Mermaid 节点标签均加引号，带文字虚线为 `-.->|根据证据调整|`。`git diff --check` 通过。
- 尚未确认 GitHub 页面中的 Mermaid/MathJax 实际渲染，也未对整章全部旧外链执行 HTTP 扫描；没有目标语料、训练系统或真实模型实验。因此本章精修记录不等同于完整章节/书册门禁通过。
