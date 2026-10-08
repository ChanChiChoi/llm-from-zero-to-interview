# 长上下文、RAG 与 Agent 精修研究记录（2026-10-08）

## 范围与结论

- `chapter`：`book-02-advanced-100/chapters/07-长上下文rag与agent.md`。
- `scope`：第 76—86 节；当前 2,596 行。第二册目录映射第 76—86 讲正确。
- `evidence_level`：`paper`、`official`、`local-demo`。
- `status`：从头复读、修订、资料/链接检查、公式结构检查及声明范围内的代码验证完成。没有把 toy 检索、Agent 或记忆示例写成生产系统保证。

## 主要修订

- 修正 `Recall@k`：分子是 top-k 命中的 gold 文档数，分母是该 query 的全部 gold 文档数；“是否至少命中一个”单列为 `Hit@k`。多相关文档样本和空 gold 样本的定义域已写明，toy 例子的召回结果与新公式一致。
- 长上下文混合训练目标明确采样权重非负且归一化；attention 成本说明按完整 $S\times S$ 矩阵估算，并说明利用因果三角区域只改变常数项。质量评分中的惩罚系数限定为非负；PQ payload 说明排除了 codebook 与索引元数据。
- 精排阶段明确 $0\le M\le |C_K|$；期望故障损失将每类失败概率乘以影响与条件恢复成本，避免把未加权的恢复成本直接加到期望值上。
- Function-calling 示例增加参数类型检查，并把 `allow` 明确作为策略结果，不声称工具已经实际执行。策略网关移除未参与决策的 `reversible` 字段。
- Memory 的几类记录明确为工程分类而非唯一学术本体；检索评分约束为非负权重；敏感度类型验证返回清晰错误，并让 tombstone 状态在教学输出中可见。
- 退款贯通案例加入订单地区、退款金额和政策例外条件；最终资格判断引用业务快照及对应政策，而不是只依赖期限条款。

## 来源与证据

章节有 41 个唯一外链。此次状态检查中 40 个返回 HTTP 200；HNSW 论文页 `1603.09320` 在重复检查时超时，但之前一次检查返回 200，且本轮单独下载其摘要页成功。arXiv Atom 批量查询服务返回 503，改为逐个读取摘要页；34 个被引 arXiv 页面均成功读取，页面 `citation_title` 元数据与正文引用的论文/方法名称相符。标题和摘要核对只确认来源身份及大致主题，不等于全文、附录、实验或基准 harness 的完整复核。

### arXiv 页面题名

| ID | 页面题名 |
| --- | --- |
| [1603.09320](https://arxiv.org/abs/1603.09320) | Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs |
| [1702.08734](https://arxiv.org/abs/1702.08734) | Billion-scale similarity search with GPUs |
| [1908.10084](https://arxiv.org/abs/1908.10084) | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks |
| [2002.08909](https://arxiv.org/abs/2002.08909) | REALM: Retrieval-Augmented Language Model Pre-Training |
| [2003.06713](https://arxiv.org/abs/2003.06713) | Document Ranking with a Pretrained Sequence-to-Sequence Model |
| [2004.04906](https://arxiv.org/abs/2004.04906) | Dense Passage Retrieval for Open-Domain Question Answering |
| [2004.12832](https://arxiv.org/abs/2004.12832) | ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT |
| [2005.11401](https://arxiv.org/abs/2005.11401) | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks |
| [2104.08663](https://arxiv.org/abs/2104.08663) | BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models |
| [2104.09864](https://arxiv.org/abs/2104.09864) | RoFormer: Enhanced Transformer with Rotary Position Embedding |
| [2210.03629](https://arxiv.org/abs/2210.03629) | ReAct: Synergizing Reasoning and Acting in Language Models |
| [2210.07316](https://arxiv.org/abs/2210.07316) | MTEB: Massive Text Embedding Benchmark |
| [2211.09110](https://arxiv.org/abs/2211.09110) | Holistic Evaluation of Language Models |
| [2302.04761](https://arxiv.org/abs/2302.04761) | Toolformer: Language Models Can Teach Themselves to Use Tools |
| [2302.12173](https://arxiv.org/abs/2302.12173) | Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection |
| [2303.11366](https://arxiv.org/abs/2303.11366) | Reflexion: Language Agents with Verbal Reinforcement Learning |
| [2304.03442](https://arxiv.org/abs/2304.03442) | Generative Agents: Interactive Simulacra of Human Behavior |
| [2304.08244](https://arxiv.org/abs/2304.08244) | API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs |
| [2305.04091](https://arxiv.org/abs/2305.04091) | Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models |
| [2305.14627](https://arxiv.org/abs/2305.14627) | Enabling Large Language Models to Generate Text with Citations |
| [2305.15334](https://arxiv.org/abs/2305.15334) | Gorilla: Large Language Model Connected with Massive APIs |
| [2305.16291](https://arxiv.org/abs/2305.16291) | Voyager: An Open-Ended Embodied Agent with Large Language Models |
| [2306.07174](https://arxiv.org/abs/2306.07174) | Augmenting Language Models with Long-Term Memory |
| [2306.15595](https://arxiv.org/abs/2306.15595) | Extending Context Window of Large Language Models via Positional Interpolation |
| [2307.03172](https://arxiv.org/abs/2307.03172) | Lost in the Middle: How Language Models Use Long Contexts |
| [2307.16789](https://arxiv.org/abs/2307.16789) | ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs |
| [2308.03688](https://arxiv.org/abs/2308.03688) | AgentBench: Evaluating LLMs as Agents |
| [2308.14508](https://arxiv.org/abs/2308.14508) | LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding |
| [2309.00071](https://arxiv.org/abs/2309.00071) | YaRN: Efficient Context Window Extension of Large Language Models |
| [2309.12307](https://arxiv.org/abs/2309.12307) | LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models |
| [2310.08560](https://arxiv.org/abs/2310.08560) | MemGPT: Towards LLMs as Operating Systems |
| [2311.09476](https://arxiv.org/abs/2311.09476) | ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems |
| [2401.00396](https://arxiv.org/abs/2401.00396) | RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models |
| [2404.06654](https://arxiv.org/abs/2404.06654) | RULER: What's the Real Context Size of Your Long-Context Language Models? |

被检查的关键摘要与正文主题一致：PI/YaRN/LongLoRA 处理窗口扩展或长序列适配；Lost in the Middle 与 RULER 关注长上下文有效利用和评估；RAG、DPR、REALM 对应检索增强路线；Toolformer/Gorilla 与间接提示注入论文对应工具和安全讨论；MemGPT/LongMem 对应运行时或长期记忆方向。公开 benchmark 的摘要和任务名称不能证明目标业务质量。

非 arXiv 来源包括 RAGAS 官方文档、Stanford IR book、OWASP LLM 应用风险页面、NIST AI RMF/GenAI Profile 和 JSON Schema 指南；链接检查均返回 HTTP 200。资料说明通用方法或接口，不能替代目标语料、权限、产品、工具和设备上的验收。

## 代码与公式检查

- 10 个 Python 代码块逐块运行成功，使用 Python 3.12.9；示例均为标准库代码。
- `.tmp/verify_chapter7.py` 额外运行 10 组有代表性的边界断言，覆盖上下文长度和布尔输入、空评估/标签类型、ACL 过滤、零向量与维度、reranker 预算/多样性、空引用指标、工具参数类型、依赖环、策略确认/资源范围、memory 隔离/敏感字段/删除。
- 执行环境为 Python 3.12.9、PyTorch 2.12.0+cpu，`torch.cuda.is_available()` 为 False；本章代码不依赖 PyTorch，没有安装 CUDA 库。环境位于被 `.gitignore` 忽略的仓库 `.tmp/llm-book-cpu-check/`，未使用系统 `/tmp`。
- 62 个 `math` 围栏；总计 136 组成对 Markdown 围栏。公式花括号和 `begin/end` 环境平衡，Markdown 表格列数一致。检查并复核 RoPE 插值、长度/缓存账本、长上下文分桶、检索和排序指标、claim/citation 支持率、工具动作条件、故障期望损失、memory 评分和贯通案例成本式。未运行外部 MathJax/Markdown 渲染器。

## 配套同步和未验证项

- `PAPERS.md` 增加长上下文、检索/RAG、RAG 证据评估、工具/API 评估、规划、安全和记忆论文入口；既有 RoPE/位置插值、Lost in the Middle、RAG 与工具基础来源没有再复制成独立清单。
- `GLOSSARY_EN_ZH.md` 统一 Recall@k 与 Hit@k 的差别；`EXERCISES.md` 的向量检索评估题要求有多个 gold 文档时比较两个指标；`INTERVIEW_BANK.md` 的检索评估答案补充分母和空 gold 处理。
- 目录已将本章映射为第 76—86 讲，下一章为第二册第八份正文；章节无相对 Markdown 链接。
- 未运行真实长上下文模型、企业文档 parser、embedding 模型、ANN 索引、reranker、RAG/Agent benchmark、外部工具或持久化 Memory。未验证真实系统的召回、事实性、权限、删除传播、生产安全、硬件成本和端到端延迟；来源链接/元数据不构成全文复核或论文结果复现。
