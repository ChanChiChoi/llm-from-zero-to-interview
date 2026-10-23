# 第三册第八章评估与 Debug 精修证据

- 日期：2026-10-22；范围：`book-03-practical-handbook/chapters/08-评估与debug实战.md`，8.1—8.7 全文。
- `evidence_level`：paper、official、local-demo；教学指标和排查顺序另按教学抽象处理。
- `lifecycle`：active；未整体复现论文、真实模型或线上系统。
- 使用维护者指定代理；链接可达、书目身份、内容核验和本地运行分别记录。

## 论文身份与来源

arXiv Atom API 本日返回以下十项题名；十个 abs 页面最终均 HTTP 200。题名核验不等于全文审阅。

| URL | 题名 | 支持范围 |
| --- | --- | --- |
| https://arxiv.org/abs/2211.09110 | Holistic Evaluation of Language Models | 多维评测与透明报告 |
| https://arxiv.org/abs/2009.03300 | Measuring Massive Multitask Language Understanding | MMLU 任务来源 |
| https://arxiv.org/abs/2206.04615 | Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models | BIG-bench 任务来源 |
| https://arxiv.org/abs/2109.07958 | TruthfulQA: Measuring How Models Mimic Human Falsehoods | 常见错误观念评测 |
| https://arxiv.org/abs/2305.14251 | FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation | 原子事实与外部可靠知识源支持率 |
| https://arxiv.org/abs/2303.08896 | SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models | 无外部知识时的采样诊断 |
| https://arxiv.org/abs/1412.6980 | Adam: A Method for Stochastic Optimization | 优化器原始定义入口 |
| https://arxiv.org/abs/1612.00796 | Overcoming catastrophic forgetting in neural networks | 参数重要性保护；不保证现代 LLM 全能力保留 |
| https://arxiv.org/abs/2106.09685 | LoRA: Low-Rank Adaptation of Large Language Models | 低秩更新定义；不保证无遗忘 |
| https://arxiv.org/abs/2306.05685 | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | position、verbosity、self-preference 等 judge 偏差入口 |

另实际读取 [FActScore HTML](https://arxiv.org/html/2305.14251) 的摘要、指标定义与支持关系相关段落：生成文本拆成原子事实，由可靠知识源核验，而非把生成文本作为自身证据。旧链接 `2305.14250` 已修正为 `2305.14251`。未复现论文的 claim 抽取器、检索系统和全套实验。

## 统计与工程来源

| 来源 | 本日检查结果 | 内容和证据边界 |
| --- | --- | --- |
| https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm | HTTP 200，读取 Wilson、边界与精确二项区间段落 | 支持区间选择；不证明业务安全非劣效 |
| https://www.microsoft.com/en-us/research/articles/diagnosing-sample-ratio-mismatch-in-a-b-testing/ | HTTP 200，读取页面 | SRM 是异常诊断入口；未检出不证明全部随机化条件成立 |
| https://doi.org/10.1177/001316446002000104 | DOI 页 403；Crossref API 200 | A Coefficient of Agreement for Nominal Scales |
| https://doi.org/10.1111/j.1468-2958.2004.tb00738.x | DOI 页 403；Crossref API 200 | Reliability in Content Analysis，副题 Some Common Misconceptions and Recommendations |
| https://doi.org/10.1145/2339530.2339653 | DOI 页 403；Crossref API 200 | Trustworthy online controlled experiments，副题 five puzzling outcomes explained |
| https://doi.org/10.1145/2433396.2433413 | DOI 页 403；Crossref API 200 | Improving the sensitivity of online controlled experiments by utilizing pre-experiment data |
| https://docs.pytorch.org/docs/stable/notes/autograd.html | HTTP 200 | 计算图官方入口；本章代码不依赖 PyTorch |
| https://github.com/EleutherAI/lm-evaluation-harness | HTTP 200 | 评测工程入口；未固定 revision 或运行 harness |
| https://github.com/openai/evals | HTTP 200 | 评测工程入口；未运行真实 API |
| https://www.deeplearningbook.org/ | HTTP 200 | 教材入口；未逐章核验 |

Crossref 核验地址为 `https://api.crossref.org/works/<DOI>`。原 Cohen 引用 `10.1037/h0048435` 实为 S-R stimulus selection and strength of R-S association.，与 Kappa 无关。四项 DOI 保留可追溯来源，但出版商全文访问是警告项，不记为全文通过。最终二十个唯一外链：十六个 HTTP 200，四个 DOI 页面 HTTP 403。

## 修订与本地验证

1. 支持率按回答 claim 统计，完整性另按问题所需事实统计；两者分母不同，不冒充 FActScore 原始指标。
2. claim 示例分别输入答案和政策证据；r1=1.0、r2=0.0。否定句仍会骗过字符串匹配，边界断言保留此反例。
3. 训练 toy 使用稳定 logits 二元交叉熵；说明 loss 和参数的更新时点，区分裁剪触发比例与梯度缩放系数。
4. 修正新任务比例与回放保护的方向；声明 KL 方向、有效 token、词表与支持集条件。
5. 人工评测拒绝重复 `(case, model, annotator)`，说明标注记录不等于独立问题。
6. 样本量公式明确正态分位数定义，近似每组 6735.15，向上取整 6736。
7. A/B 示例不再用 6/7 次安全事件的点估计放行；安全复核为 False，输出保持观察。未实现真实安全非劣效分析。

Python 3.12.9 下七个 Python 围栏分别 compile、独立执行成功，无额外依赖。边界检查覆盖空 benchmark、重复 ID、空关键词、Wilson 零分母和非法计数、NaN 临界值、JSON 非对象、shingle 非法大小、Jaccard 空并集、claim 空证据/空集合/NaN 权重、非法训练特征和标签、NaN 成本、重复标注、全平局、空 A/B 组、SRM 非法比例和稳定分桶。

- runner 全正确率 1.0，Wilson 95% 区间 `[0.5101, 1.0]`；近重复 a1/a2=0.5、b1/b2=0.55。
- 训练 loss 从 0.6931 降至第 299 步更新前 0.0565，最终预测 `[0, 0, 0, 1]`。
- mixed_replay 效用 0.9088；人工评分 A/B 均值 0.6725/0.94，B 非平局胜率 0.8。
- A/B 解决率差 0.0239、95% 近似区间 `[0.0134, 0.0343]`，SRM p=0.606176；安全证据不足，输出 `hold_and_investigate`。
- 四十六个 math 围栏花括号/方括号计数平衡，围栏成对；无 `$$`、`operatorname`、尖括号下标或 `text` 下划线。
- 第三册目录与相邻章节链接目标存在；配套论文索引和练习按需同步。
- PDF dry-run 因缺少 Pandoc 失败；未安装构建依赖，未验证外部数学渲染器。

教学代码不代表真实模型、污染检测召回、语义事实判断、标注者可靠性、线上因果效应或生产发布安全性。
