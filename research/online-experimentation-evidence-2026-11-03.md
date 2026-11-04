# Online experiment design and generative AI field evidence

检索日期：2026-11-03。记录本次核对的在线实验方法来源与生成式 AI 现场研究。网页检索结果仅作发现入口；下列事实按来源论文或研究机构页面核对。未复现论文分析或平台结果。

## 生成式 AI 现场研究

### Generative AI at Work

- **来源与版本：** Brynjolfsson、Li、Raymond，*The Quarterly Journal of Economics* 140(2), 889–942，正式发表于 2025-02-04；DOI: [10.1093/qje/qjae044](https://doi.org/10.1093/qje/qjae044)。核读期刊页面摘要、引言、部署设计与 RCT 附录说明。
- **支持事实：** 研究使用一家企业 5,172 名客服坐席的数据，分析基于 GPT-3 的实时对话建议；坐席能忽略或编辑建议。作者报告平均每小时解决问题数提高 15%，且不同技能与经验组的效果存在异质性。
- **设计边界：** 主分析研究分阶段部署，不是完整的并行 50:50 随机 A/B。论文还报告约 50 名员工的先导随机试验、约半数分配到处理组；作者没有拿到该试验控制组员工的信息，因而将先导处理组与其余未处理坐席比较。此附录分析不能直接替代主样本的分阶段部署设计。

### The Effects of Generative AI on High-Skilled Work

- **来源与版本：** Cui 等，*Management Science*，2026-02-27 online publication，DOI: [10.1287/mnsc.2025.00535](https://doi.org/10.1287/mnsc.2025.00535)。核读 INFORMS 期刊页面摘要和文章信息。
- **支持事实：** 三家企业（Microsoft、Accenture 和一家未具名 Fortune 100 企业）进行了软件开发者现场随机对照试验；合并样本为 4,867 名开发者。论文报告完成任务数增加估计为 26.08%，标准误 10.3%；单项实验结果噪声较大且存在差异，经验较少的开发者采用率和生产率收益较高。
- **证据边界：** 估计针对论文中的代码补全工具、参与开发者、任务与合并分析；不能直接推广为所有开发者、工作流或 LLM 产品的固定收益率。期刊页面显示 online publication，卷期页码可能后续补全。

### Generative AI-enabled clinical decision support in primary care

- **来源与版本：** Agweyu 等，*Nature Medicine* 32, 3032–3039 (2026)，正式网页版本发表于 2026-06-26；DOI: [10.1038/s41591-026-04503-6](https://doi.org/10.1038/s41591-026-04503-6)。核读期刊公开摘要、结果和方法摘要；网页正文读取受期刊授权页限制。
- **支持事实：** 肯尼亚 16 家基层医疗机构开展临床人员集群随机试验，103 名临床人员管理 9,691 名患者。14 天治疗失败率为干预组 102/4,693（2.2%）、对照组 94/4,654（2.0%）；调整后 OR 0.77，95% CI 0.55–1.08，P=0.13。临床文书质量等次要结果改善。
- **证据边界：** 主要患者结局没有显示显著组间差异。作者指出试验对观察到的低事件率下的较小效应及罕见严重伤害估计不精确；没有观察到相关安全问题不能证明普遍安全或安全等价。研究对象是特定医疗网络和 14 天窗口。

## 在线实验方法

### CUPED

- **来源与版本：** Deng 等，*Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data*，WSDM 2013，DOI: [10.1145/2433396.2433413](https://doi.org/10.1145/2433396.2433413)；[作者提供的论文页面](https://exp-platform.com/cuped/)。核读摘要、协变量选择、缺失前期数据处理和 Bing 实验结果部分。
- **支持事实：** 方法使用 treatment 之前的协变量调整结果指标，以减少指标方差。作者在 Bing 实验中报告约 50% 方差下降，并指出效果与指标、前期数据覆盖和协变量与结果的相关程度有关。
- **证据边界：** 约 50% 是特定平台和指标的实测经验，不是任意产品实验的预期值。实验前数据缺失、协变量预测力弱或 treatment 后变量会改变可用性或引入偏差；该方法不修复分配、曝光或结局日志问题。

### Always-valid inference 与连续查看

- **来源与版本：** Johari、Pekelis、Walsh，*Always Valid Inference: Bringing Sequential Analysis to A/B Testing*，arXiv:1512.04922，后发表于 *Operations Research* 70(3) (2022)，DOI: [10.1287/opre.2021.2135](https://doi.org/10.1287/opre.2021.2135)。核读论文对 always-valid p 值、序贯测试和多重检验的定义与方法概述。
- **支持事实：** 固定终点的常规 p 值和区间不能在随意的数据依赖停止规则下沿用原保证；作者提出了允许在数据不断到达时做有效推断的序贯方法，并讨论序贯多重检验及自适应分配（如多臂 bandit）的扩展。
- **证据边界：** 有效性依赖于指定的推断过程、数据假设、停止规则和检验家族；always-valid inference 不是让任意指标、任意查看和事后选择都自动有效。

### SRM 诊断

- **来源与版本：** Fabijan 等，*Diagnosing Sample Ratio Mismatch in Online Controlled Experiments*，KDD 2019；[Microsoft Research 对该研究的实践说明](https://www.microsoft.com/en-us/research/articles/diagnosing-sample-ratio-mismatch-in-a-b-testing/)。核读研究机构页面中的 taxonomy 描述和引用信息。
- **支持事实：** SRM 是分配、执行、日志处理或分析阶段中多种数据质量问题的共同症状；在解释处理效果前应定位根因，而不是简单删样本或重加权。
- **证据边界：** SRM 本身不识别唯一根因；具体比例检查需要明确预期流量、随机分配单位和取数阶段。

### Network interference

- **来源与版本：** Saveski 等，*Detecting Network Effects: Randomizing Over Randomized Experiments*，KDD 2017，DOI: [10.1145/3097983.3098192](https://doi.org/10.1145/3097983.3098192)；[作者公开论文](https://faculty.washington.edu/msaveski/assets/publications/2017_detecting_network_effects/paper.pdf)。核读摘要、SUTVA 假设说明与实验设计概述。
- **支持事实：** 论文提出并行运行完全随机与集群随机实验、比较两种估计，以检测网络溢出是否使个体随机设计的 SUTVA 假设不可信；作者在 LinkedIn 平台的两个实验中展示了网络效应信号。
- **证据边界：** 该设计是检测干扰的特定方法，不保证集群随机能够消除跨集群传播，也不直接给出所有产品适用的随机单位。

### Switchback design 与 carryover

- **来源与版本：** Bojinov、Simchi-Levi、Zhao，*Design and Analysis of Switchback Experiments*，*Management Science* 69(7), 3759–3777 (2023 issue)，online publication 2022-11-01，DOI: [10.1287/mnsc.2022.4583](https://doi.org/10.1287/mnsc.2022.4583)。核读 INFORMS 摘要与文章信息。
- **支持事实：** Switchback 按时间段对共享实验单位依次随机指定处理并观测结果。论文针对处理影响持续阶数的不同假设推导设计与推断方法，并讨论影响阶数设错时的性质及其估计。
- **证据边界：** 设计取决于如何刻画 carryover；时间块轮换本身不保证有效因果估计。结论受模型、周期划分、干扰和推断假设约束。

- **来源与版本：** Missault、Masoero，*Carryover Detection in Switchback Experimentation*，NeurIPS 2025 Workshop: CauScien (Uncovering Causality in Science)；[OpenReview PDF](https://openreview.net/pdf?id=fyfkP1TOKW) 与 [Amazon Science 页面](https://www.amazon.science/publications/carryover-detection-in-switchback-experimentation)。核读摘要和方法边界。
- **支持事实：** 论文提出比较不同滞后影响假设下的估计结果，检测预设 influence period 可能低估 carryover 的情形；论文报告其假设下的理论性质和基于合成效应/真实协变量的仿真。
- **证据边界：** 作者所述保证依赖加性线性效应等条件；workshop 论文与模拟证据不能证明方法在所有生产系统、干扰结构或时间趋势下都可靠。

### Let Them Peek: Group Sequential Designs for A/B Testing

- **来源与版本：** Perini 等，arXiv:2610.27899，v1 于 2026-10-20 提交。核读论文摘要、引言和方法概述。
- **支持事实：** 预印本提出固定最大期限下的 predictive probability of success (PPoS) group-sequential 方案，目标是保留平台原固定终点决策规则、用历史 A/A 结果校正日间相关并优化预期日历时长。作者报告其在线 A/A 与 A/B 评估中预期时长较 14 天基线减少 43.1%（7.97 天），并报告其设定下 FPR 与 power 保持。
- **证据边界：** 这是 2026 年 10 月的预印本和作者平台结果；结果依赖其设计、平台数据、历史 A/A 估计、模拟与执行条件。它不是随机产品的预期收益，也不替代按项目条件设计和验证序贯边界。
