# 第二册第八份正文：Reasoning 与 Test-Time Compute 复核

核验日期：2026-10-08。对象：`book-02-advanced-100/chapters/08-reasoning与test-time-compute.md`，第 87—96 讲。本记录区分 arXiv 元数据/摘要、链接状态、本地代码证据和仍未核验的全文实验；它不代表模型或生产系统评测。

## 本轮发现与修订

- CoT 的生成式公式原先把待生成的轨迹写成输入。现改为联合条件分布分解 `p(r,y|x)=p(r|x)p(y|x,r)`，并说明解码策略和轨迹生成顺序。
- 干预敏感度补 `N>0` 与零样本时的 `N/A`；PRM 第一错步补步骤分、阈值和索引范围。
- CoT 审计器改为只识别整数加、减、乘；未识别等式显式记为不可检查。最终答案只能从显式 final/answer 字段提取，避免把解释文本末尾的数字误当答案；输出另列不可检查样本数。
- 加权投票公式和代码统一为正温度指数权重。实现通过减去最大 verifier 分数避免指数上溢；该常数平移不改变候选排序。投票 demo 另检查小数、分数和百分数等价，以及“not 4”不能被解析成 `4`。
- Test-time compute 的候选命中公式限定在独立同分布采样；`A_final ≤ P_hit` 限定在最终选择器只能从候选池中选择、不能另行生成或修复答案的情况。自适应路由实际使用 `verifiable` 字段，高风险且无法独立验证的请求转人工复核并计入独立状态。
- 污染率补全非空评估集和训练候选集合定义；数据过滤器改为整串数值解析，不再从“not 4”之类文本中抽取末尾数字，并处理等价分数/小数/百分数。HumanEval pass@k 补足整数及 `n,c,k` 的定义域。
- `PAPERS.md` 新增 Reasoning 与 Test-Time Compute 研究入口，并补上 InstructGPT、CodeRL 和 Self-Debug 的直接来源。`EXERCISES.md`、`INTERVIEW_BANK.md` 和 `GLOSSARY_EN_ZH.md` 已有覆盖这些主题的练习、追问和术语，本轮未重复堆叠。

## 来源与证据范围

通过用户指定的代理 `http://10.24.27.134:7890` 查询 arXiv Atom API。正文当前 23 个 arXiv 引用的 ID 和题名均已核对；同时读取摘要作为主题和范围的初筛。下面记录元数据标题，摘要支持范围仍受原论文数据、任务和设置限制。

| arXiv ID / 版本 | API 返回标题 |
| --- | --- |
| [1712.01815v1](https://arxiv.org/abs/1712.01815) | Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm |
| [2103.03874v2](https://arxiv.org/abs/2103.03874) | Measuring Mathematical Problem Solving With the MATH Dataset |
| [2107.03374v2](https://arxiv.org/abs/2107.03374) | Evaluating Large Language Models Trained on Code |
| [2110.14168v2](https://arxiv.org/abs/2110.14168) | Training Verifiers to Solve Math Word Problems |
| [2201.11903v6](https://arxiv.org/abs/2201.11903) | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models |
| [2203.02155v1](https://arxiv.org/abs/2203.02155) | Training language models to follow instructions with human feedback |
| [2203.11171v4](https://arxiv.org/abs/2203.11171) | Self-Consistency Improves Chain of Thought Reasoning in Language Models |
| [2203.14465v2](https://arxiv.org/abs/2203.14465) | STaR: Bootstrapping Reasoning With Reasoning |
| [2205.11916v4](https://arxiv.org/abs/2205.11916) | Large Language Models are Zero-Shot Reasoners |
| [2206.14858v2](https://arxiv.org/abs/2206.14858) | Solving Quantitative Reasoning Problems with Language Models |
| [2207.01780v3](https://arxiv.org/abs/2207.01780) | CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning |
| [2212.10560v2](https://arxiv.org/abs/2212.10560) | Self-Instruct: Aligning Language Models with Self-Generated Instructions |
| [2304.05128v2](https://arxiv.org/abs/2304.05128) | Teaching Large Language Models to Self-Debug |
| [2304.12244v3](https://arxiv.org/abs/2304.12244) | WizardLM: Empowering large pre-trained language models to follow complex instructions |
| [2305.04388v2](https://arxiv.org/abs/2305.04388) | Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting |
| [2305.10601v2](https://arxiv.org/abs/2305.10601) | Tree of Thoughts: Deliberate Problem Solving with Large Language Models |
| [2305.20050v1](https://arxiv.org/abs/2305.20050) | Let's Verify Step by Step |
| [2306.02707v1](https://arxiv.org/abs/2306.02707) | Orca: Progressive Learning from Complex Explanation Traces of GPT-4 |
| [2306.11644v2](https://arxiv.org/abs/2306.11644) | Textbooks Are All You Need |
| [2310.04406v3](https://arxiv.org/abs/2310.04406) | Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models |
| [2407.21787v3](https://arxiv.org/abs/2407.21787) | Large Language Monkeys: Scaling Inference Compute with Repeated Sampling |
| [2408.03314v1](https://arxiv.org/abs/2408.03314) | Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters |
| [2501.12948v2](https://arxiv.org/abs/2501.12948) | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning |

核验过程中发现章节原先把 CodeRL 链接到 `2302.00003`。API 返回该 ID 的真实标题为 *The Power of External Memory in Increasing Predictive Model Capacity*；已改用正确的 [CodeRL `2207.01780`](https://arxiv.org/abs/2207.01780)，其 API 标题、作者和摘要与代码生成及深度强化学习主题相符。`2203.02155` InstructGPT 及 PRM800K、GSM8K 两个 GitHub 仓库也已核验入口。

25 个唯一外链（23 个 arXiv 页面及两个 OpenAI GitHub 仓库）经该代理逐一请求均返回 HTTP 200。状态码证明目标可访问，不证明论文全文或仓库每条数据都与正文主张等价。论文摘要复核只用于核对来源身份、主题和结论边界；未逐篇复核全文、附录、全部实验数据或评分 harness。此前既有记录对 `2110.14168`、`2107.03374`、`2408.03314` 的范围说明可参照 [`post-training-alignment-review-2026-10-07.md`](post-training-alignment-review-2026-10-07.md) 和 [`training-alignment-inference-review-2026-10-07.md`](training-alignment-inference-review-2026-10-07.md)。

## 本地验证

- 环境：仓库 `.tmp/llm-book-cpu-check`，Python 3.12.9、PyTorch 2.12.0+cpu、NumPy 2.4.6；章节代码仅用标准库，没有 CUDA 依赖。
- 10 个 Python 块均逐块编译、独立执行，输出与正文展示结果逐字匹配。
- 19 项边界检查覆盖 final 字段缺失、未支持和混合算式、错误算式、分数/小数/百分数归一化、否定文本、缺失答案、温度零值/负值/非有限值/布尔值、高风险未验证路由、待人工复核计数、非文本答案和 pass@k 数值边界。
- 89 个 `math` 围栏的花括号和 `begin/end` 环境配对；Markdown 围栏成对；本地相对链接目标存在。25 个唯一外链全部 HTTP 200。
- 结果只证明这些教学脚本、断言和输出在记录的本地 CPU 环境下成立，不代表论文复现、真实模型推理质量、GPU 行为或生产路由收益。

## 遗留项

1. 本轮只看 arXiv 元数据和摘要，尚未逐篇复核全文、附录、实验表和代码；正文结论保持在摘要和可访问来源支持的范围。
2. toy verifier、路由阈值、费用和延迟数字均为构造值；不能外推成实际模型或产品保证。
3. 未调用外部 Markdown/LaTeX 渲染器；数学结构检查不等于目标 PDF 渲染验收。
