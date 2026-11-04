# 多模态与长上下文评测协议：来源证据

检索日期：2026-11-03。本记录仅保存新核验的评测事实、来源和证据边界。

## VQA 多标注员软分数

- **来源：** GT-Vision-Lab, [VQA 官方评测实现 `vqaEval.py`](https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py)，访问日期 2026-11-03；阅读 `evaluate()`、`processPunctuation()` 和 `processDigitArticle()`。
- **来源支持：** 官方实现逐个将一名标注员答案暂时排除，对余下答案计算 `min(1, matching_count / 3)`，再对所有留一情况取平均。答案不唯一时，评测器运行标点处理、数字/冠词归一化和 contraction 映射；它是固定实现中的字符串协议，不执行一般语义同义词判断。
- **证据等级：** source-backed（benchmark 仓库的评测代码）。
- **边界：** URL 指向可变 `master` 分支，引用具体实现时应同时固定 commit；其他 VQA 数据集可能采用不同答案数、规范化器或聚合方式。

## DocVQA ANLS 对提示后缀与答案格式的敏感性

- **来源：** Srinivas Harish, “Most of a Document-VQA Score Is Answer Formatting,” [ACL Anthology 页面](https://aclanthology.org/2026.docinsights-1.6/)及[论文 PDF](https://aclanthology.org/2026.docinsights-1.6.pdf)，DocInsights 2026，2026-10-24。实际阅读摘要、§1–3、实验结果及限制讨论。
- **来源支持：** 论文对 `Qwen2.5-VL-3B` 的实验称，仅移除所测试 harness 的简短作答 suffix，DocVQA ANLS 从 94.2 变为 7.5；同一论文对提示 suffix、输出后处理和 ANLS 进行对照，发现同一模型分数会随这些协议因素大幅变化。论文还报告 317 题 hosted-model 子集上的结果，及一种 length-gated span scorer 对“整页 OCR 直接作为答案”攻击的对照。
- **实验边界：** 数据包括作者使用的 1,200 题公开 DocVQA 样本及其它特定子集；托管模型比较使用 317 题，模型、日期、API、harness、提示后缀与解析规则绑定结果。该研究说明所测协议下答案格式会显著影响 ANLS，不证明所有历史 DocVQA 结果无效，也不说明替代 scorer 已适用于所有文档任务。
- **证据等级：** source-backed（2026 年同行评审 workshop 论文及指定实验）。

## LongShOTBench：rubric 级长视频音视频评估

- **来源：** Kurpath et al., “A Benchmark for Omni-Modal Reasoning in Long Videos,” [arXiv:2512.16978 v2](https://arxiv.org/abs/2512.16978v2)，2026-06-16 修订。实际阅读 v2 摘要、§1、§3.2–3.5、§5.1–5.3、附录 B.1、C 和 G。
- **来源支持：** v2 描述 274 段 7.8–89.7 分钟视频、3,401 个样本和 4,893 个问答轮次；问题覆盖视觉、语音、非语音音频、时间和跨模态推理，既有单轮也有多轮。作者使用 Qwen3 系列辅助生成问题、答案和 rubric，并进行人工审核。答案配有加权 criterion-level rubric，三个 LLM verifier 分别按标准项对照经人工验证的参考答案打分；作者在分层抽取的 400 个 rubric criterion 上报告三 verifier ensemble 的 Cohen’s κ 为 0.753，而 leave-one-out 人际 κ 均值为 0.672、范围 0.61–0.76。
- **证据等级：** source-backed（arXiv v2 预印本的 benchmark 构造与作者验证研究）。
- **边界：** 三 verifier 与人工对齐的数值只来自该论文的 400 项审阅子集；残余 verifier 偏差和提示敏感性仍可能影响绝对分数。模型每项仅运行一次，未估计模型采样波动；原生视频评测委托每个模型自己的默认帧采样，不能保证模型间输入等价；语料来自 Video-MME 与 YouTube，英语占多数，构题又使用单一 Qwen3 系列。多轮协议用理想上一轮回答更新后续上下文，因此不直接测真实对话漂移。
