# LLM 不确定性估计与 conformal abstention：来源证据

检索与阅读日期：2026-11-09。

## 多模型不确定性估计比较

- **来源与版本：** Linwei Tao 等，[Revisiting Uncertainty Estimation and Calibration of Large Language Models](https://arxiv.org/abs/2505.23854)，arXiv v1，2025-05-29。
- **阅读范围：** arXiv HTML 摘要、方法中的不确定性类型和评价定义、模型/数据/提示说明及结果概述；本轮未复现实验，也未逐项核读全部附录。
- **来源支持：** 作者在 MMLU-Pro 上比较 80 个开源与闭源模型（0.6B–671B），评估三类黑盒单次生成信号：token 概率不确定性（TPU）、数值口头置信度（NVU）和语言表达不确定性（LVU）。论文分别用 ECE 评估概率校准、用 AUROC 评估对正确/错误回答的排序；作者报告 LVU 在该设置下整体优于 TPU/NVU，并指出高准确率不保证不确定性可靠、校准较好也不保证错误排序有效。其提示要求模型按固定格式给出答案/信号，不符合格式的样本不纳入该项分析。
- **证据边界：** 这是 arXiv 预印本中的跨模型比较，结论限于 MMLU-Pro、多选任务、所测模型快照、三种信号和作者提示/解析器；不是开放文本、单条事实主张或高风险产品的通用校准保证。LVU 还依赖用另一个 LLM 判断措辞，信号来源与评估器都需要在目标数据上审查。

## 面向语言模型的 conformal abstention

- **来源与版本：** Rui Xu、Yi Chen、Sihong Xie、Hui Xiong，[Geometry-Calibrated Conformal Abstention for Language Models](https://arxiv.org/abs/2604.27914)，arXiv v1，2026-04-30；按预印本处理。
- **阅读范围：** arXiv HTML 摘要、§§2–3 的交换性及有限样本保证、§4 表征几何信号、§5 实验数据/模型和主结果；未核读全部附录证明与消融。
- **来源支持：** 论文提出 Conformal Abstention（CA），将回答/弃答决策与有限样本 conformal 过程结合，讨论参与率和“模型参与时回答正确”的条件保证；评分信号是回答置信度而非为开放文本构造完整非一致性预测集。置信度校准器使用模型表征几何特征。实验覆盖 NQ、TruthfulQA、Simple Questions Wiki、SciQ、GSM8K、CommonsenseQA 六个数据集；使用 Gemma-3-4B-Instruct、Llama-3.2-3B-Instruct 和 Llama-3-8B-Instruct，并用 HHEM-2.1-Open、GPT-4o、GPT-5.1 共同定义正确性评价函数。作者报告该实验中平均 conditional correctness 为 75.0%，并对校准集/测试集进行重复随机划分。
- **证据边界：** 理论保证依赖交换性、划分/校准协议和论文的参与/正确性事件定义；实验“正确”标签由论文指定的自动评价函数确定，不等同于独立人工真值。75.0% 绑定所列模型、六个数据集、评价器、校准器与参与阈值；不能外推为任意生成文本或生产系统准确率。论文是预印本，不能据此替代经典 split conformal 的基本假设和手算定义。
