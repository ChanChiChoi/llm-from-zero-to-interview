# 方法比较中的研究证据与固定版本

> 记录类型：联网核验得到的研究事实与来源边界。书籍章节修改过程和验收状态只写入 `progress/`。

## Long Context 与 RAG

### Retrieval Augmented Generation or Long-Context LLMs? v2

- **来源与版本**：[arXiv:2407.16833v2](https://arxiv.org/html/2407.16833v2)，v1 提交于 2024-07-23，v2 修订于 2024-10-17；论文标注发表于 EMNLP 2024 Industry Track。
- **阅读范围**：摘要、引言、§3.1–§3.3。核对了实验模型、数据集筛选、检索器和结果摘要。
- **支持的事实**：作者在 LongBench 与 ∞Bench 的英语、真实、query-based 子集上比较 LC 与 RAG；选用 9 个数据集，模型为 Gemini-1.5-Pro、GPT-4O、GPT-3.5-Turbo，主比较使用 Contriever，并另分析 Dragon。其报告称 LC 在三个模型的总体平均表现优于 RAG，而 RAG 的输入成本较低；GPT-3.5-Turbo 在两个超过其 16K 窗口的 ∞Bench 数据集上由 RAG 胜出。实验的 chunk 默认长度为 300 词、top-k 为 5。
- **证据等级与边界**：原始论文的作者 benchmark 结果。结论限于上述模型快照、英文数据集、retriever、提示和预算；作者指出黑盒模型的预训练数据泄漏仍是开放问题。文中的 token/API 成本不是当前价格表或任意目标系统的成本结论。

### Long Context vs. RAG for LLMs: An Evaluation and Revisits v1

- **来源与版本**：[arXiv:2501.01880v1](https://arxiv.org/html/2501.01880v1)，提交于 2024-12-27；检索时 arXiv 页面显示该版本为 v1。
- **阅读范围**：摘要、引言、问题过滤与数据扩充说明、实验结论及讨论。
- **支持的事实**：作者移除无需外部上下文即可回答的问题，使用 12 个 QA 数据集并扩展源数据；报告 LC 在 Wikipedia 类 QA 上通常胜过 RAG，摘要式检索可与 LC 相当，chunk-based retrieval 较弱，而 RAG 在对话式和一般问题上有优势。作者强调上下文相关性以及模型、检索方式、问题类型会改变结论。
- **证据等级与边界**：原始论文的作者评测。属于 2024 年数据和实验协议下的研究报告，不是所有语料、生成模型和生产成本的通用排序；与 v2 的数据、模型和过滤方式不完全相同，结果不能只按一句“LC 胜”或“RAG 胜”合并。

## Agent 与 Workflow 的操作性区分

- **来源**：[Anthropic, “Building effective agents”](https://www.anthropic.com/research/building-effective-agents)，发布于 2024-12-19；2026-10-30 阅读当前页面。
- **阅读范围**：页面的发布日期和时效提示、 “What are agents?” 与 “When (and when not) to use agents” 部分。
- **支持的事实**：Anthropic 将 workflow 描述为由预设代码路径编排 LLM 与工具的系统，将 agent 描述为由 LLM 动态决定过程和工具使用的系统，并建议从尽量简单的方案开始。页面明确提醒文中描述的工具生态自 2024-12 起已有变化。
- **证据等级与边界**：公司工程团队的实践经验和操作性术语划分，不是学界统一定义；可引用其控制流区别，不能用它作为当前 Agent 框架或产品功能的最新目录。

## 评测信号的范围

### CoT 解释忠实性

- **来源**：[Language Models Do Not Always Say What They Think v2](https://arxiv.org/abs/2305.04388v2)，arXiv 修订版 v2（2023-12-09）。
- **支持的事实**：该研究在受控输入干预中观察到，某些影响答案的偏置信号没有出现在模型生成的 CoT 中，模型还可能对受偏置信号影响的错误答案给出流畅理由。这支持“可见 CoT 不能直接当作忠实的内部决策记录”。
- **证据等级与边界**：论文中的特定模型和实验任务；不证明所有模型、任务或每条 CoT 都不忠实，也不把生成解释等同于模型内部计算的完整观察。

### LLM-as-a-Judge 偏差

- **来源**：[Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena v4](https://arxiv.org/abs/2306.05685v4)，arXiv 修订版 v4（2023-12-24）。
- **支持的事实**：论文分析位置、冗长和 self-enhancement 等 judge 偏差，并讨论 judge 与人类偏好的相关性及其限制。
- **证据等级与边界**：特定开放式对话题目、judge 模型和提示下的结果；不能据此给当前 judge 或专业领域任务指定统一偏差幅度。

### FActScore

- **来源**：[FActScore v2](https://arxiv.org/html/2305.14251v2)，v2 修订于 2023-10-11；论文发表于 EMNLP 2023。
- **阅读范围**：摘要，核对 atomic-fact 统计定义与评估对象。
- **支持的事实**：该方法将长篇回答拆成 atomic facts，并计算其中得到可靠知识来源支持的事实比例，针对长文本 factual precision 设计。
- **证据等级与边界**：原论文指标与作者实验；该分数不自动等同于“回答被本轮 RAG 上下文蕴含”，也不能单独代表完整事实性或 grounding。
