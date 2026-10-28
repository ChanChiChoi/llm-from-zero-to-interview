# 第八册第九章 Model Editing 与 Unlearning 精修记录

日期：2026-10-28

## 范围与结构决定

- 按当前写作规范复读第九章，沿用原有 25 节主结构，不新增章节。保留四个教学单元：任务与机制（§1–§4）、编辑方法和编辑评估（§5–§10）、遗忘定义/方法/评估（§11–§16）、综合案例和可回滚工程（§17–§20）；§21 demo 作为辅助单元。
- 将 §15 标题改为“四组重要基准和评测数据”，对应 CounterFact/zsRE、TOFU、WMDP、MUSE 四个小节组。
- 将 §18 的纯文本删除链改成数据处置与审计闭环图。源数据清理、模型参数处理、授权审计、回滚和副本检查作为不同责任节点呈现。
- 公式标记统一为单美元行内式和 `math` 围栏；移除旧式 `$$`、`\(...\)` 与 `\operatorname`。

## 定向研究与采用决定

| 来源 | 核对内容 | 纳入正文的用法与边界 |
| --- | --- | --- |
| [EVOKE，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c592fc7e6207f82560ed45fece8d6937-Abstract-Conference.html) | 官方会议页面摘要说明 editing overfit、目标概率过度集中和复杂/多跳泛化问题，并介绍 EVOKE benchmark。 | 纳入 §10.2，强调直接目标正确不代表组合推理稳定；结论绑定论文任务与测试方法。 |
| [LocFT-BF，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/74b8ccb5ad21871fd436867df19c35ac-Abstract-Conference.html) | 官方会议页面摘要对比逐样本训练到收敛与 breadth-first mini-batch/epoch 更新；作者报告测试设置中的编辑提升，并披露 100K 编辑和 72B 参数模型结果。 | 纳入 §2.2，指出应加入合理 fine-tuning baseline；不写成所有微调优于所有编辑方法。规模结论绑定作者模型、数据和方法。 |
| [UniEdit，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6cc6c2be42c838827d2f1d12baa05539-Abstract-Datasets_and_Benchmarks_Track.html) | 官方摘要说明覆盖 25 个常见领域、用知识图谱邻域多跳采样评估 ripple effects，并用专有 LLM 转写自然语言。 | 纳入 §10.2，支持新增 relation-combination、多跳和近邻 locality 切片；将专有 LLM 数据生成标为构造边界。 |
| [Context-Aware Unlearning，ICML 2026](https://proceedings.mlr.press/v306/peng26a.html) | PMLR 官方摘要报告对六种 unlearning 方法检查把被遗忘内容重新提供到上下文后的 contextual utility，并指出原有方法在该维度有退化。 | 纳入 §14.2 与 §16.5，单列授权上下文效用；它不等同于模型参数自行回忆目标内容。 |
| [Information Decomposition Audit，EACL 2026](https://aclanthology.org/2026.eacl-long.35/) | ACL Anthology 官方条目摘要称，在论文设置中遗忘数据的部分信息仍可从内部表征线性解码，并与已知重构攻击敏感性相关。 | 纳入 §14.4 与 §16.5，作为可选白盒审计维度；可解码不等于发生外部泄露或任意攻击者可提取。 |
| [SUITE，arXiv 预印本，2026-07](https://arxiv.org/abs/2607.09236) | arXiv 摘要提出 forget/retain 评测的非对称泛化，并讨论相邻保留边界标注。 | 纳入 §14.2，标注为预印本和数据设计线索，不视作已定型通用基准。 |

研究页面核对以正式会议、PMLR、ACL Anthology 和 arXiv 摘要/元数据为主；本轮未在目标模型上复现实验，也未将论文 headline 结果外推到生产模型。

## 正文和数值例修改

- §2.2 增加 LocFT-BF 对 fine-tuning baseline 的限定比较。
- §10.2 增加 EVOKE、UniEdit 对编辑过拟合、多跳泛化和 locality/ripple testing 的启示。
- §10.7 用 demo 四条编辑逐步手算目标成功率 `9/10=0.900`、改写泛化 `0.667`、locality `0.800`、retain 均值 `0.883→0.851`、变化 `-0.032`，并说明 2 条冲突超过 toy 门槛。
- §14.2 增加 contextual utility 和 forget/retain 边界非对称覆盖；§14.4 增加表征残留审计及其外部泄露边界。
- §16.5 扩展 evidence matrix，并手算 exact leakage `5/12=0.417`、robust leakage `19/36≈0.528`、MIA change `-0.378`、retain change `-0.025`。leakage gate 仍失败，因此不能被其他通过项抵消。
- §17 增加按预先阈值逐项判断企业事实编辑的数值案例；§18 增数据血缘文字与 Mermaid 删除审计流程；§19 补能力代理分数的解释边界。

## 教学 demo 修改和验证

- §21 代码使用标准库，只处理合成记录，不加载模型或修改参数。
- `weighted_mean_raw` 保留未舍入值，防止先分别舍入 before/after 改变 delta；显示均值和变化值再统一舍入到三位。
- `weighted_nested_mean` 跳过无子样本的记录；全无子样本返回 `None`。空编辑集合的冲突数标记为未测量，`compare(None, ...)` 不放行。非正总权重抛出输入错误。
- 固定输入输出：edit target/paraphrase/locality `0.900/0.667/0.800`，retain delta `-0.032`，冲突数 2；unlearning exact/paraphrase/multi-turn/robust leakage `0.417/0.750/0.417/0.528`，MIA delta `-0.378`，retain delta `-0.025`。编辑因冲突 hold，遗忘因 exact/robust leakage hold。
- Python 3.12.9 标准库运行固定输入，并单独检查空均值、全空嵌套集、单条空嵌套样本、未定义阈值、零总权重、缺字段；空值不会被解释为通过，错误输入会显式失败。

## 后续开放项

- 需要在 GitHub 实际页面核验 Mermaid 和 MathJax 渲染；源码围栏配对和 Mermaid 语法只构成静态证据。
- 需要做本章所有外链 HTTP 扫描和沿第八册主阅读路径连续复读；本章尚未通过完整正文门禁。
- 无真实 checkpoint、目标训练谱系、授权数据、生产 harness、人工 gold labels 或真实请求流量验证。论文基准和合成 demo 仅支持其声明范围内的教学/研究判断。
