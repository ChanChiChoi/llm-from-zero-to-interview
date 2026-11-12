# Statistical inference for LLM evaluation: judge noise, repeated runs, and benchmark resolution

首次检索日期：2026-11-03；复核日期：2026-11-09。记录联网核对的来源事实和适用边界；未复现实验或论文计算。

## Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges

- **来源与版本：** Chen Feng、Minghe Shen、Ananth Balashankar、Carsten Gerner-Beuerle、Miguel Rodrigues，*Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges*。arXiv:2601.20913 v1 于 2026-01-28 提交，arXiv 页面标注 accepted to ICLR 2026；同时核读 [ICLR 2026 proceedings abstract](https://proceedings.iclr.cc/paper_files/paper/2026/hash/08a67eb74bfb9ded9e949dd52973e997-Abstract-Conference.html) 与 [arXiv HTML](https://arxiv.org/html/2601.20913v1)。阅读摘要、认证假设、第 3—6 节的测试设置/程序/保证/实验、结论限制和附录数据卫生说明。
- **支持事实：** 论文将认证问题表述为二元失败率阈值检验：原假设为真实失败率不低于容忍阈值；一组小规模人工标签估计 judge 的 TPR/FPR，另一组更大的 judge 标签用于计算 noisy failure rate，临界值将校准参数不确定性计入。Theorem 5.1 对给定校准集给出 $P_e^{(I)}\leq\zeta+O(n_J^{-1/2}+n_{M_1}^{-1/2}+n_{M_0}^{-1/2})$ 的界；论文的精度依赖 Berry–Esseen/正态近似项，因此不能把它压缩成所有有限样本都精确满足 $P_e^{(I)}\leq\zeta$。作者在 Jigsaw Comment、Hate Speech 和 SafeRLHF 的实验中比较 noisy hypothesis testing、仅人工标签的 direct testing、PPI 变体和参数已知的 oracle。方法在 judge 质量高于推导条件的部分设置中可提高功效；实际估计参数仍与理想 oracle 有差距。
- **证据边界：** 这是特定二元 pass/fail 认证方法，不等于通用多指标 p 值修正。理论依赖其 i.i.d. 抽样和 judge 误差建模假设；人工标签本身仍决定 ground truth 的含义。作者指出主观任务中的人工分歧、极小校准集和稀有失败事件可能削弱正态近似精度，并要求单独验证/选择 judge 以避免校准集 peeking。统计有效性不等于道德或构念效度。

## Towards Reproducible LLM Evaluation: Quantifying Uncertainty in LLM Benchmark Scores

- **来源与版本：** Robert E. Blackwell、Jon Barry、Anthony G. Cohn，arXiv:2410.03492 v2；Alan Turing Institute 另托管 PDF。核读摘要、第 3 节方法、第 4 节结果、第 5 节讨论与限制。
- **支持事实：** 研究在两个方位推理 benchmark（100 与 5,760 个问题）上，对六组当时的模型/API 设置重复运行，重复数最多 30 次；比较默认采样设置与 temperature=0、fixed seed。作者报告后者在多数但非全部测试条件下降低分数波动；仍有模型/API 设置不能被温度和 seed 完全确定。其区间估计面向未来重复运行的均值（prediction interval），与抽样题目总体的置信区间估计对象不同。论文还在其一个小型基准和 `n=90` 的配置比较中报告 Azure 与 OpenAI API 得分差异，但接口和消息格式同时不同，不能仅归因于底层模型。
- **证据边界：** benchmark 仅是两个特定空间推理任务；模型、版本、provider、部署和解码设置均来自论文时期。重复次数和方差不是当前 LLM 的通用量级，也不能外推到其他任务、prompt 或 top-p 设置。固定 seed 是需要记录的条件，不保证跨硬件/API 完全复现。

## The Sample Complexity of LLM Evaluation: Confidence Intervals and the Minimum Detectable Effect of a Benchmark

- **来源与版本：** TMLS AI Research，网页技术报告，2026-07-08。阅读网页摘要、引言、配对与小样本区间部分、聚类与 MDE 总览及限制说明；来源页面称其数值为既有资料与指定公式的分析/计算，不是新模型实验。
- **支持事实：** 报告把独立、同题配对和题目聚类作为 benchmark 设计，讨论小样本比例区间、McNemar 不一致率、设计效应、生成运行噪声和 MDE 样本规划。其 MDE atlas 给出的 HumanEval、GPQA Diamond、SWE-bench Verified、GSM8K、MMLU 差异分辨率，绑定报告采用的项目规模、错误率、显著性/功效和配对不一致假设。
- **证据边界：** 这是机构技术报告，未见同行评审出版信息；benchmark 特定 MDE 是分析结果，不能当作 benchmark 的永久常数。使用时必须确认题集版本、独立/配对结构、错误率和目标效应与报告设定相符。相关机制在教材中以公式和可复算的合成例讲解，不移植该 atlas 的单一数值表。

## Wilson score intervals for binomial proportions

- **来源与版本：** NIST/SEMATECH *e-Handbook of Statistical Methods*, §7.2.4.1, “Confidence intervals,” accessed 2026-11-05. 核读 Wilson 区间公式、由比例检验反演区间的说明、二项精确区间段落及术语说明：https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm。
- **支持事实：** NIST 将 Wilson 区间说明为通过反演二项比例的检验得到，并给出由样本比例、样本数和标准正态临界值计算上下界的表达式；页面另行描述基于二项分布的精确区间。书中 `n=20,k=14` 与 `n=10,000,k=7,000` 的 95% 区间是按公式自行复算的教学数据，不是 NIST 的实测结果。
- **证据边界：** 推断对象是独立二项抽样下的比例；相关题目、重复生成、题目聚类或同题模型比较需要保留其依赖/配对结构，不能把独立二项区间直接当成模型差异检验。NIST 给出的比例区间不替代对具体 benchmark 抽样设计和评估协议的审查。
