# 推理时计算、实时交互与评估激励：来源证据

记录日期：2026-10-28

本记录只保存联网发现的知识点及其证据。来源内容的具体适用范围、与书册的映射、正文修改状态和后续安排记录在 progress/。

## 1. 推理时计算的分配

### 来源

- Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar, “Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters,” ICLR 2025；[arXiv 版本](https://arxiv.org/abs/2408.03314)，[ICLR 2025 论文页与 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/1b623663fd9b874366f3ce019fdfdd44-Paper-Conference.pdf)。
  - 实际阅读范围：arXiv 摘要、引言、测试时计算分配方法和实验结果部分；重点核对搜索/verifier、迭代修订和 compute-matched 比较。
  - 来源支持：在论文使用的数学推理任务与模型设置中，测试时计算的收益会随问题难度、策略和基础模型变化；作者提出按 prompt 难度调整预算，并报告相对 Best-of-N 基线的效率收益。论文也报告，在特定 FLOPs 匹配设置下，较小模型加额外推理计算可与约 14 倍参数规模的预训练模型比较。
  - 证据等级：source-backed（同行评审论文的特定实验结果）。
  - 适用边界：结果依赖该论文的 PaLM 2 系模型、数学问题、verifier/修订方法、预算和评价设置；不得外推为所有模型、任务或生产场景中推理计算都优于扩大预训练。
- DeepSeek-AI, “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning,” arXiv v2 revised 2026-01-04; journal reference: Nature 645, 633–638 (2025); [固定 arXiv v2](https://arxiv.org/abs/2501.12948v2)。
  - 实际阅读范围：摘要、引言、R1-Zero 训练方法和作者报告的实验结果。
  - 来源支持：作者报告 DeepSeek-R1-Zero 在没有先做监督微调的设定下直接进行大规模 RL，并观察到回答长度和某些推理行为随训练变化；论文也报告可读性差和语言混杂等问题。
  - 证据等级：source-backed（作者技术报告中的模型特定描述和实验）。
  - 适用边界：这是特定模型与训练方案的作者报告，不构成对所有 RL 训练的结论；论文中的 benchmark 分数、训练过程和推理行为不能证明通用推理能力或自然语言推理过程的忠实性。

### 综合证据边界

推理时计算是可以被分配和评估的推理资源。额外采样、搜索、验证或修订可能改善指定任务的成功率，但收益受问题难度、基础模型、预算和选择器能力影响；额外 token 和计算也带来延迟与成本。该研究方向不应被简化成“让模型多想总会更好”。

## 2. 实时多模态会话与工具事件

### 来源

- OpenAI API 文档，[Realtime conversations](https://developers.openai.com/api/docs/guides/realtime-conversations)，访问日期 2026-10-28。
  - 实际阅读范围：Realtime session 组件、会话状态和 function calling 流程。
  - 来源支持：文档描述的 Realtime Session 是客户端与模型间的有状态交互，包含 Session、Conversation 和 Responses；接口事件覆盖音频/文本生成、图像输入和 function calling。文档所述 function calling 流程中，客户端检测调用参数、执行自有代码并回传结果。
  - 证据等级：source-backed（单一供应商的当前产品/API 文档）。
  - 适用边界：只说明该 API 的公开接口合同和组件，不证明其他模型的内部架构或所有实时交互系统均采用同一实现；具体字段和行为会随 API 版本变化。

## 3. 准确率式评估与不确定回答

### 来源

- Adam Tauman Kalai, Ofir Nachum, Santosh S. Vempala, Edwin Zhang, “Evaluating large language models for accuracy incentivizes hallucinations,” Nature 653, 1047–1051 (2026), [DOI 页面](https://doi.org/10.1038/s41586-026-10549-w)，在线发表于 2026-04-22。
  - 实际阅读范围：摘要、正文中关于预训练误差、封闭式准确率激励、open-rubric 方案、SimpleQA 实验与 Discussion 的限制说明。
  - 来源支持：论文从学习理论出发分析 next-word 预训练如何在其理想化设定下对不可重复事实产生统计性幻觉压力；又指出把 abstention 计为错误的准确率评分会奖励猜测，并提出把评分规则写进问题的 open-rubric 评估。其示例实验使用四个推理模型和 SimpleQA，并按不同错误惩罚测试模型是否调整 abstention。
  - 证据等级：source-backed（同行评审论文的理论分析与有限实验）。
  - 适用边界：作者明确说明该比较不是跨模型的受控基准，采用默认设置且没有调参或成本归一化；closed-rubric 仍可能具有生态效度，correct/incorrect/abstain 三分法也不能覆盖自然语言中的完整校准表达。结果不能简化为“准确率指标必然让所有模型幻觉”，也不意味着开放 rubric 可消除幻觉。

## 生命周期

推理时计算：active；具体策略效果仍依赖任务与预算。

实时多模态会话接口：active；此处描述的供应商 API 细节需按版本复核。

评估激励分析：active；需结合具体任务效用、错误代价和 abstention 规则设计指标。
