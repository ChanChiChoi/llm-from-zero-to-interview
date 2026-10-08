# 第二册第九份正文：论文精读与开放研究题复核

核验日期：2026-10-08。对象：`book-02-advanced-100/chapters/09-论文精读与开放研究题.md`，第 97—104 讲。证据等级包括 `paper`、`official` 和 `local-demo`；正文讲授通用机制，研究结论仍受各自论文实验设置约束。

## 本轮复核与修订

- Attention toy demo 明确直接复用手工向量作为 Q/K/V，省略可学习投影和训练，限定了它能演示的机制。
- Scaling demo 把 `relative_inference_cost` 改为 `relative_dense_decode_compute`，并说明固定 dense 架构和生成长度下以参数量近似逐 token 解码计算；该量不代表账单或延迟，也不包含上下文、KV Cache、批处理、硬件与 kernel 差异。
- DPO demo 用稳定的负 log-sigmoid 计算 loss，避免偏好 logit 很负时先把 sigmoid 舍入为零、再计算 `log(0)`。
- LoRA demo 显式检查 A/B adapter 与 base weight 的输入维度、输出维度和 rank，避免 `zip` 在输出维度不一致时静默截断。
- FlashAttention demo 将 `peak_reduction_x` 改为 `score_cell_reduction_x`，并说明 1024 倍只比较完整 score 矩阵与单个 score tile 的元素数，不是完整 attention 峰值显存比。
- 研究计划评分说明先按项目参考范围归一化不同单位；成功成本公式要求请求成本采用统一口径。
- 将公式中的 `\\operatorname` 换为 `\\mathrm`，并把尖括号下标改为 `x_{1:i-1}` / `y_{1:t-1}` 等 GitHub Markdown 兼容形式。
- `PAPERS.md` 将 LoRA 和 DPO 条目补成直达原论文的链接。练习、面试题和术语表已覆盖这两项及 Scaling Laws、FlashAttention、MoE 等相关主题，本轮不重复复制题目或定义。

第二册目录把本章映射为第 97—104 讲，下一章从第 105 讲开始；简介的章节顺序一致。

## 来源与证据范围

使用维护者指定的代理 `http://10.24.27.134:7890` 查询 arXiv Atom API，核对 11 篇论文的当前元数据标题、版本和摘要主题：

| arXiv ID / 版本 | API 返回标题 |
| --- | --- |
| [1706.03762v7](https://arxiv.org/abs/1706.03762) | Attention Is All You Need |
| [2001.08361v1](https://arxiv.org/abs/2001.08361) | Scaling Laws for Neural Language Models |
| [2005.14165v4](https://arxiv.org/abs/2005.14165) | Language Models are Few-Shot Learners |
| [2101.03961v3](https://arxiv.org/abs/2101.03961) | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity |
| [2106.09685v2](https://arxiv.org/abs/2106.09685) | LoRA: Low-Rank Adaptation of Large Language Models |
| [2203.02155v1](https://arxiv.org/abs/2203.02155) | Training language models to follow instructions with human feedback |
| [2203.15556v1](https://arxiv.org/abs/2203.15556) | Training Compute-Optimal Large Language Models |
| [2205.14135v2](https://arxiv.org/abs/2205.14135) | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness |
| [2305.10601v2](https://arxiv.org/abs/2305.10601) | Tree of Thoughts: Deliberate Problem Solving with Large Language Models |
| [2305.18290v3](https://arxiv.org/abs/2305.18290) | Direct Preference Optimization: Your Language Model is Secretly a Reward Model |
| [2408.03314v1](https://arxiv.org/abs/2408.03314) | Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters |

本章 15 个唯一外链中，11 个 arXiv 页面、GPT-2 技术报告 PDF、GPT-4 System Card PDF 和 OpenAI Model Spec 共 14 个最终返回 HTTP 200。早期 GPT 官方研究页面 `https://openai.com/research/language-unsupervised` 返回 HTTP 403，正文已标明该访问限制；另两个首次请求遇到的 TLS EOF 在重试后恢复为 HTTP 200。arXiv HTML 页面在本轮后续请求中出现代理连接拒绝或 TLS EOF，因此本章的论文证据复核停留在元数据和摘要，不声称逐篇核对全文、附录、实验表或 harness。

## 本地验证

- 环境为仓库内 `.tmp/llm-book-cpu-check` 的 Python 3.12.9；本章示例只导入标准库，不需要 PyTorch 或 CUDA。
- 8 个 Python 代码块逐块编译、独立运行，输出与正文中的 text 输出逐字一致；31 项边界和不变量断言通过，覆盖非法维度、空输入、非有限值、DPO 极端 logit、mask/分块 attention 一致性、路由和研究计划硬门槛。
- 77 个 `math` 围栏及全部 104 个 fenced block 配对；公式花括号和 `begin/end` 环境检查通过。章节的 Markdown 表格列数一致，没有相对 Markdown 链接。
- `git diff --check` 通过。`python3 scripts/build_pdfs.py --dry-run` 因环境未安装 Pandoc 而停止；本轮没有安装构建依赖或 CUDA 库，也没有构建 PDF。

## 遗留边界

1. arXiv 元数据与摘要核验不等于完整论文阅读；论文实验结论只按摘要、正文引用及其可支持范围表述。后续如扩展 Kaplan 指数、Chinchilla 配比或各论文实验比较，应先读对应全文和附录。
2. toy attention、DPO、Scaling、LoRA、FlashAttention、MoE 和研究评分器均只验证所写代码及合成输入，不代表论文复现、模型行为、GPU 性能或生产决策。
3. OpenAI 早期 GPT 页面本轮返回 403，外部数学渲染和 PDF 构建未完成；页面可访问性或渲染差异发生变化时应复查。

本章正文精修、来源元数据/摘要核验、外链、代码输出、边界、公式结构和配套论文索引同步已完成；结论为 `source-backed`、`verified` 和 `local-demo`，证据范围以本记录为准。全书精修仍在继续。
