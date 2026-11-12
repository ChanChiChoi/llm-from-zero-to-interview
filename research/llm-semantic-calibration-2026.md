# LLM 语义校准：来源证据

检索日期：2026-11-06。

本记录只保存联网核验得到的知识点及其证据范围，不记录书籍审阅进度或修改历史。

## 语义答案校准与 next-token 校准

- Preetum Nakkiran, Arwen Bradley, Adam Golinski, Eugene Ndiaye, Michael Kirchhof, and Sinead Williamson, “Trained on Tokens, Calibrated on Concepts: The Emergence of Semantic Calibration in LLMs,” ICLR 2026，[官方会议论文 PDF](https://openreview.net/pdf/6cd63883b6f1271c277f7b5d1d7ca64d47709f8d.pdf)。
  - 实际阅读范围：摘要、§§1–6、Appendix B.1 Limitations、Appendix D 的模型、数据、prompt、语义归并和校准测量说明，以及 Appendix E 中关于 B-confidence calibration 与 full B-calibration 的区分。
  - 来源支持：论文将生成回答通过语义归并函数 B 映射到有限答案类别，再把模型诱导的类别概率视为分类器输出。其主要实验测量 top-class 的 B-confidence calibration：每个问题采用 5-shot prompt，在 temperature=1 下生成 50 个回答，经 B 归并后用最高类别的经验频率作为语义置信度，再与 gold answer 比较准确性。作者在 Qwen、Gemini、Mistral 和 Llama 系列、0.5B–72B 模型，以及 GSM8K、OpenMathInstruct-2、MATH500、TriviaQA、SimpleQA 和 TruthfulQA 上报告超过 650 组实验。基础模型的非 CoT concise/sentence 设置在多数纳入条件下表现良好；TruthfulQA 是不符合论文分布假设且校准较差的例外。指令微调和 CoT 设置经常失准，方向可表现为过度或不足置信。
  - 实际测量中的 B 对不同任务采用不同归并办法：数学答案通过规范化字符串比较；非数学句式回答先由 Qwen3-14B-Instruct 抽取简答，再由同一模型判断语义等价并聚类。该评测依赖归并器、prompt、采样数量和温度。
  - 证据等级：source-backed（ICLR 2026 同行评审论文的定义、理论和作者实验）。
  - 适用边界：该结论针对特定的采样式 B-confidence calibration，不等于完整的多类别 B-calibration，也不等于口头自报置信度或单次回答的事实正确概率。结果依赖题目分布、模型后训练、回答形式、temperature 和语义归并函数；TruthfulQA 的失败表明其分布假设并非普遍成立。论文没有证明所有基础模型都校准，亦没有建立从 token-level calibration 到任意长答案事实校准的普遍保证。

## 生命周期

该论文对区分 token 级、语义答案级和 claim 级置信度仍有用；数值结论保持绑定论文所列模型、问答数据、五样本提示、temperature=1 及其语义归并流程。
