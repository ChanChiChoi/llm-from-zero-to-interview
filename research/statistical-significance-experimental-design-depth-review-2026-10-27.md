# 统计显著性与实验设计深度精修研究记录

- 日期：2026-10-27
- 正文：[第七册第十章](../book-07-evaluation-experiments/chapters/10-统计显著性与实验设计.md)
- 依据：当前 [WRITING_SPEC.md](../WRITING_SPEC.md) 与 [progress_v2.md](../progress_v2.md)
- 范围：统计估计对象、成对评估、二分类配对检验、重采样、样本量、序贯分析、多重检验、CUPED、消融和可复核实验记录。
- 执行边界：阅读 ACL Anthology、arXiv、ExP 与论文 DOI 资料；未运行本章 Python、测试套件、外部 benchmark 或目标系统/API。

## 1. 原章节知识承接

原章已有 17 个一级主题，涵盖 estimand、分析单位、paired comparison、McNemar、均值/比例/长尾指标、p 值、区间、bootstrap、MDE/功效、聚类、多重比较、winner's curse、CUPED、消融、可复现性、统计审计 demo 和证据链。本轮保留整套结构，集中深化“怎样按数据结构选择方法”，补齐方法适用条件，并修复来源与样本量演示的对应关系。

| 原知识点 | 本轮深化位置 | 具体处理 |
| --- | --- | --- |
| 配对二分类结果 | §2.3 | 解释 exact McNemar 条件二项思路，手算 `b=1,c=8` 的双侧精确 p 值，并限定其解释 |
| 分析单位与检验选择 | §2.7 | 对比 paired t、McNemar、paired bootstrap、近似随机化检验和 cluster bootstrap；逐项说明数据结构、目的与前提 |
| 区间和外部有效性 | §5.2 | 加入 NLP 显著性研究对测试域与目标域迁移的证据边界，避免把 benchmark 内区间解释为部署保证 |
| MDE 与配对样本量 | §7.4 | 增加以 `sigma_d` 为规划量的配对样本量公式与完整数值例子 |
| 多重检验 | §8.3 | 补充 BH 的依赖条件、BY 任意依赖修正、Holm step-down 与 FWER/FDR 的目标差异 |
| CUPED | §9 | 修复错误来源链接，保留预实验协变量和不能修复日志/随机化偏差的边界 |
| toy audit demo | §12 | 将原先独立两组 `n_each` 公式改为与配对数据结构相符的 `n_paired` 规划函数，并同步输出字段、动作和解释 |
| 论文索引 | §16 | 修正 CUPED / sequential 的错配链接，加入 NLP 检验选择和跨域有效性研究 |

## 2. 代表性研究分别解决什么问题

| 来源 | 方法贡献 / 回答的问题 | 正文用法 | 限制 |
| --- | --- | --- | --- |
| Dror et al., [The Hitchhiker’s Guide to Testing Statistical Significance in NLP](https://aclanthology.org/P18-1128/), ACL 2018 | 根据 NLP 任务结构、指标和数据特性提出检验选择流程，并讨论参数与非参数检验；作者也审查了当时 ACL/TACL 实践中的误用 | §2.7 将检验方法按问题、单位、假设和限制对照，提醒不要只按“准确率/分数”套公式 | 这是方法指南与 2017 论文样本审查，不是所有现代 LLM 工作流的自动决策器；作者明确将跨大量数据集的复现分析留在范围外 |
| Berg-Kirkpatrick et al., [An Empirical Investigation of Statistical Significance in NLP](https://aclanthology.org/D12-1091/), EMNLP-CoNLL 2012 | 实证考察 bootstrap 显著性，并讨论测试集同分布校准与跨域系统胜负预测能力的区别 | §5.2 用作“统计显著性只在所设抽样/目标总体条件下解释”的例子 | 是早期 NLP、特定数据和系统比较；不能直接当作当前大模型线上效果的估计 |
| Deng et al., [CUPED](https://exp-platform.com/cuped/), WSDM 2013 | 用处理前协变量解释结果波动，降低在线实验估计方差；Bing 实验报告约 50% 方差降低 | §9 说明实验前信息如何提高精度，强调不同指标/历史覆盖的收益不同 | 数字绑定 Bing 指标、协变量和平台；不是所有 LLM 实验的先验收益，也不能修复曝光/分组偏差 |
| Johari et al., [Always Valid Inference](https://arxiv.org/abs/1512.04922) | 为持续监测下的 A/B 实验构造始终有效的 p 值与区间，并讨论序贯多重检验 | §4.6、§7.8 区分固定样本量分析与预先设计的序贯方案 | 不授权任意停止；要按设计的统计量、监测边界和决策规则解释 |
| Nie et al., [Ensure A/B Test Quality at Scale](https://arxiv.org/abs/2208.07766), CIKM 2022 | eBay 平台将 population stability index 用于随机化验证，并用序贯分析进行 SRM 监控，以减少大规模实验监控误报 | §16 将其作为 SRM/随机化质量监控的工程来源 | 是 eBay 平台实现与其数据/监控流程的案例；不能直接把报警参数移植到其他平台 |
| Holm, [A Simple Sequentially Rejective Multiple Test Procedure](https://doi.org/10.2307/4615733), 1979 | 逐步下降程序控制 family-wise error rate，提供 Bonferroni 之外的确认性多重检验方案 | §8.3 对比简单 Bonferroni 与 Holm | 需定义检验家族；控制 FWER 不等于控制 FDR，也不能修复事后选切片 |
| Benjamini & Yekutieli, [FDR under Dependency](https://doi.org/10.1214/aos/1013699998), 2001 | 用调和因子修正阈值，在任意依赖下控制 FDR | §8.3 解释共享题目/用户造成的检验相关性与保守修正 | 代价是较低发现力；不表示安全硬门禁适合按 FDR 放宽 |

这几种方法各自作用于不同层面：检验选择指南处理“当前任务该用什么检验”；paired / cluster 设计确定抽样单位；bootstrap 和随机化检验构造区间或零假设分布；CUPED 降低精度噪声；Holm、BH、BY 处理多重决策。没有一种方法能替代合理的数据生成过程、目标总体、分析计划和独立复验。

## 3. 来源身份与链接修正

本轮逐项打开相关 primary source 页面核对身份：

- `arXiv:1512.04922` 的真实标题为 **Always Valid Inference: Bringing Sequential Analysis to A/B Testing**，不能作为 CUPED 论文链接；CUPED 改为 ExP 原论文页面，并同时给出 WSDM 2013 DOI `10.1145/2433396.2433413`。
- `arXiv:2102.12352` 的真实标题为 **A convex analysis approach to tight expectation inequalities**，不是序贯检验；从序贯来源列表删除，改链到 `1512.04922`。
- `arXiv:2208.07766` 的真实标题为 **Ensure A/B Test Quality at Scale with Automated Randomization Validation and Sample Ratio Mismatch Detection**；本章和第九章的短标题更新为与实际论文身份一致的标题及贡献。
- 加入 ACL Anthology 的 Dror 2018 和 Berg-Kirkpatrick 2012，分别作为检验选择指南与测试域/目标域边界研究；结论仅按其原研究任务范围转述。

## 4. 数学与 demo 复核

### McNemar 精确检验

在零假设下，给定 `b+c` 个 discordant pairs，每一方向的条件概率为 `0.5`。`b=1,c=8` 时双侧精确二项值为 `2 × (1+9)/2^9 = 0.0390625`；连续性校正统计量为 `(|1-8|-1)^2/9 = 4.0`。精确 p 值针对不一致配对的方向，不是新模型更好的后验概率；二项检验的单位仍须是独立配对题目/随机单位。

### 配对样本量

独立两组样本量公式和配对设计的公式不能混用。对于连续差值 `d_i`，计划量应使用差值标准差 `sigma_d`：

`n_paired ≈ sigma_d² × (z_(1-alpha/2) + z_(1-beta))² / delta_min²`

在 `sigma_d=0.5`、双侧 `alpha=0.05`、power `0.80`、MDE `0.03` 下，向上取整为 2,178 对。原 demo 把独立两组 `n_each=4,356` 与配对样本数直接比较，口径不一致；现已新增配对规划函数并将输出改为 `sample_needed_pairs` / `sample_observed_pairs`。独立两组公式仍留在 §7.2，服务于其原有情形。

### 确定性部分的手工核算

- 20 对二分类 toy 数据中，旧模型正确 `9/20=0.45`，新模型正确 `16/20=0.80`，差值 `0.35`；McNemar 不一致数 `b=1,c=8`，连续性校正统计量为 `4.0`。
- p 值组为 `0.004, 0.014, 0.030, 0.070, 0.200`：简单 Bonferroni 阈值 `0.01` 仅保留首项；BH 在独立/适当依赖条件的 toy 计算中保留前三项。BY 不在 demo 中执行，正文仅解释公式和依赖边界。
- paired bootstrap 是随机重采样；本轮未执行 Python，也未声称逐次模拟区间已运行核验。章节代码静态复读其配对索引保留、样本量命名和中间结果输出。

## 5. 配套同步与静态检查

- `PAPERS.md`：增加本章入口、ACL 方法指南、NLP 跨域边界、CUPED、Always Valid Inference、Holm/BH/BY，并同步 SRM 论文准确标题。
- `EXERCISES.md`：增加 87D 精确 McNemar 和 87E 多重检验依赖/决策题。
- `INTERVIEW_BANK.md`：增加 5H 检验选择和 5I 多重检验方法比较。
- `GLOSSARY_EN_ZH.md`：补充 estimand、ATE、配对/聚类 bootstrap、近似随机化检验、Holm 与 BY 定义。
- 第七册目录已收录本章，无需改目录。
- 检查一级章节 §1—§17 连续、代码围栏成对、研究记录与索引链接目标存在，并运行 `git diff --check`。未运行 Python、测试套件、公式渲染或 PDF 构建；本轮只核验所引用关键来源，不代表全章外链 HTTP 可用性扫描。

## 6. 未验证事项

没有目标业务样本、真实 benchmark 抽样框、用户/租户随机实验、人工 judge 校准、生产 SRM trace 或目标系统分布。所有示例和阈值用于教学；研究来源中的方法收益绑定论文任务、样本、指标与平台。配对或聚类方法仍需按目标随机化单位选择；少数 cluster、域偏移、稀有安全事件、反复自适应使用 holdout 和外部状态干扰需要单独设计，不能由本章的 toy demo 代替。
