# 幻觉、评估与安全：研究证据

- 检索与核验日期：2026-10-29
- 覆盖问题：幻觉与拒答的评估激励、RAG 幻觉检测标签噪声、benchmark contamination 信号及污染缓解保真度、LLM-as-a-Judge 的校准边界、prompt injection / jailbreak 的术语和应用侧缓解。
- 证据状态：主要结论为 `source-backed`。以下论文结果限于其论文模型、任务、数据和评估协议；OWASP 页面是风险分类及工程建议，不是控制有效性的独立实验。

## 来源及其支持范围

### 准确率目标与拒答激励

- **来源**：[Kalai et al., “Evaluating large language models for accuracy incentivizes hallucinations,” Nature 653, 1047–1051 (2026)](https://www.nature.com/articles/s41586-026-10549-w)，2026-04-22，DOI: 10.1038/s41586-026-10549-w。
- **阅读范围**：论文题名、摘要及可访问的官方文章预览。2026-10-29 直接打开 Nature 正文时被重定向到身份验证页，未能重新读取完整 HTML。
- **支持的事实**：论文分析 next-token 预训练下的一次性事实回忆压力，以及在特定评分设定中，二元准确率把拒答计错时可能奖励猜测；作者讨论开放式 rubric 和显式错误代价。
- **证据边界**：是论文中的形式化论证和有限实验，不证明所有模型训练、奖励模型、数据集或产品评估都会产生同样激励。不要把它写成“SFT/RLHF 必然导致幻觉”。

### RAG 幻觉检测与标签噪声

- **来源**：[Chen et al., “Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights,” ACL 2026, pp. 14912–14931](https://aclanthology.org/2026.acl-long.680/)，[PDF](https://aclanthology.org/2026.acl-long.680.pdf)。
- **阅读范围**：ACL Anthology 论文记录和摘要；相关正文、实验及标注协议见论文 PDF。论文描述对长上下文 RAG 幻觉检测的 TRIVIA+ 基准、人工标注和三组样本相关标签噪声。
- **支持的事实**：作者报告，在其测试的 RAG 幻觉检测设置中，标签噪声降低检测表现；一个基础 LLM-as-a-Judge baseline 在该基准上有竞争力。
- **证据边界**：结果受 TRIVIA+ 样本、长上下文 RAG 任务、标注定义和所测 detector 限制，不能外推为所有评估集均有同等标签噪声，也不能说明 judge 普遍优于其他评估方式。

### Benchmark contamination 的黑盒信号

- **来源**：[Oren et al., “Proving Test Set Contamination in Black-Box Language Models,” ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/46e624c244cff669223d488defd4e835-Abstract-Conference.html)。
- **阅读范围**：ICLR 官方论文摘要。
- **支持的事实**：其方法比较 benchmark 的规范题目顺序与打乱顺序的 likelihood；在无污染时题集样本可交换等前提下，规范顺序异常更可能可作为污染信号。方法意在无需训练语料或模型权重的条件下识别信号，但仍需模型 likelihood 访问。
- **证据边界**：统计保证依赖论文的 exchangeability 等前提和评分可访问性。没有触发信号不能证明训练数据没有题目、答案或语义近邻。

### 污染缓解与评测保真度

- **来源**：[Sun et al., “The Emperor’s New Clothes in Benchmarking? A Rigorous Examination of Mitigation Strategies for LLM Benchmark Data Contamination,” ICML 2025, PMLR 267, pp. 57728–57753](https://proceedings.mlr.press/v267/sun25t.html)。
- **阅读范围**：PMLR 官方论文摘要。
- **支持的事实**：论文使用 10 个 LLM、5 个 benchmark、20 种缓解策略和 2 种污染情境，按样本级结果检查 fidelity 与 contamination resistance；作者报告在所测设置中未找到同时有效平衡两者的已有策略。
- **证据边界**：该结论是针对论文实验范围的比较，不代表之后提出的策略都无效，也不能据此断言任意题目改写必然改变难度。

### LLM-as-a-Judge 的偏差与校准

- **来源**：[Zheng et al., “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena,” arXiv:2306.05685v4](https://arxiv.org/html/2306.05685)，2023-12-24。
- **阅读范围**：§§3.3–3.4 与 §4.2。
- **支持的事实**：作者讨论 position、verbosity、self-enhancement bias 和 judge 推理能力限制，并在 MT-Bench / Chatbot Arena 特定实验中报告 GPT-4 judge 与人类偏好的高一致性。
- **证据边界**：论文中的一致率不能作为其他模型、任务、rubric 或人评样本的通用准确率；候选顺序、回答长度和 judge 配置仍须在目标数据校准。

- **来源**：[Shi et al., “Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge,” arXiv:2406.07791v9](https://arxiv.org/abs/2406.07791)，2025-11-11 修订版；论文发表于 AACL-IJCNLP 2025。
- **阅读范围**：v9 摘要。
- **支持的事实**：其研究在 MT-Bench 和 DevBench 的 22 类任务中测试 15 个 judge、约 40 个答案生成模型并报告超过 150,000 个评估样本；位置偏差并非随机噪声，且随 judge 与任务变化。
- **证据边界**：研究对象限于论文所选 judge、benchmark 与 pairwise/listwise 协议；顺序交换提高稳定性不等于判分正确或目标任务已经校准。

### Prompt injection 与 jailbreak 的术语和缓解

- **来源**：[OWASP LLM01:2025 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)，当前页面明确标记为 LLM01:2025。
- **阅读范围**：定义、jailbreak 关系、direct / indirect injection 分类以及 Prevention and Mitigation Strategies 页面部分。
- **支持的事实**：页面将 jailbreak 视作 prompt injection 的一种；direct / indirect 是按输入来自用户还是外部网站/文件作的工程分类；页面指出 RAG 和微调不能完全缓解，并建议限制模型行为、校验输出格式、最小权限等措施。
- **证据边界**：OWASP 页面是威胁分类和防御建议，不证明某种单一缓解策略可消除风险，也不是目标应用通过安全验收的证据。有效测试应在应用实际信任边界、权限与副作用条件下完成。
