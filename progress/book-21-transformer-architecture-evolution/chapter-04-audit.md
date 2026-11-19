# 第 4 章《MHA、MQA、GQA、MLA 的演进与 trade-off》审阅记录

审阅日期：2026-11-19。状态：章级来源扫描、正文精修、配套资料同步和本地静态检查完成；目标平台公式渲染与本册连续复读留待后续门禁。

## 读者契约与章节职责

- **书册目标贡献：** 解释 Transformer 注意力中 K/V 状态如何从逐头保存发展到共享或低秩压缩，使读者能按一致单位估算缓存元素，并识别论文质量、训练和延迟结果的适用范围。
- **读者前置：** 理解第 2 章单头 Q/K/V 读取机制和第 3 章多头投影；可阅读向量/矩阵记号及常见模型配置。
- **离章结果：** 能区分 MHA、MQA、GQA 的 K/V head 共享方式和 MLA 的 latent cache；推导每 token、每层 cache 元素；解释 MLA 的位置 Key；不把元素比例直接外推成参数、质量或端到端延迟比例。
- **读者路径：** 多头基线与缓存动因 → MQA 共享 → GQA 分组和 checkpoint uptraining → MLA 低秩 latent 与解耦 RoPE Key → 统一 cache 元素口径 → 论文结果边界 → 架构、kernel、serving 管理与 FFN 路由分层。
- **前后交接：** 第 3 章解释 head 投影与固定总宽度下增加 head 数并不必然增加 K/V 总缓存；第 5 章主讲 FFN/SwiGLU/MoE；第 10 章主讲 attention 计算与 FlashAttention；第 11 章主讲请求长度、并发、数据类型和 serving 条件下的 KV cache 容量及管理；第 24 章主讲推理系统，29/42 章承接 SSM 与替代架构，63 章承接 gated MLA。

## 来源扫描与证据边界

- **检索与原文核验日期：** 2026-11-19。扫描主题覆盖 MQA 增量解码带宽动机、GQA 分组与既有 checkpoint 转换/继续训练、MLA 的 latent 与 RoPE cache 表示、MHA2MLA 的模型适配，以及公开课程中 attention-head sharing 与 cache 主题的组织方式。
- **标准 MHA 基线：** [Attention Is All You Need](https://arxiv.org/html/1706.03762v7)，阅读 §§3.2.2、4。支持每头独立可学习投影及并行计算；不支持各头投影正交或具有固定语义。
- **MQA：** [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/html/1911.02150)，阅读 §§2.2–2.4、§§3–4。支持共享一组 K/V 的结构和增量解码带宽动机；速度与质量观察受论文模型、翻译/语言建模任务和 TPU 设置约束。
- **GQA：** [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://aclanthology.org/2023.emnlp-main.298/)，阅读 §§2–4、表 1、§5。支持从 MHA checkpoint 组内 mean-pool K/V 并 uptrain 的方法；约 5% 计算量来自特定 T5 encoder-decoder 方案，论文没有评估 decoder-only 模型。
- **MLA：** [DeepSeek-V2](https://arxiv.org/html/2405.04434v5)，阅读 §§2.1.1–2.1.4、附录 C、D.1–D.2。缓存单位含 K/V latent 和位置相关 RoPE Key；作者报告的缓存/质量结果绑定模型规模、激活参数和 benchmark。摘要中的完整模型吞吐不作为 MLA 单因素因果证据。
- **MHA 转 MLA：** [MHA2MLA](https://aclanthology.org/2025.acl-long.1597/)，阅读 §§1–4、Llama2-7B LongBench 表和附录。论文使用 partial-RoPE、联合 SVD 和微调；结果不是零样本替换或通用转换保证。
- **公开课程组织样本：** Stanford CME295 2025 Lecture 2 slides/syllabus 以 query-group 数展示 MQA、GQA、MHA；University of Edinburgh 2026-10 Model Inference Systems 讲义将 cache 容量、GQA、MLA、kernel 和 serving 系统分层。课程资料仅证明其教学组织，不证明学习效果或唯一最佳顺序。
- 事实摘要、原文阅读位置与详细边界见[本章 MQA/GQA/MLA 研究记录](../../research/mha-mqa-gqa-mla-evidence-2026-11-19.md)。

## 主题去向与修改决定

| 主题 | 正文处理 | 理由或交接 |
| --- | --- | --- |
| MHA/MQA/GQA | 保留共享 head 数和缓存元素推导 | 区分固定每头宽度与固定总投影宽度，避免把 head 数本身说成缓存增长原因 |
| MLA | 解释联合低秩 K/V latent、另存 RoPE Key 和缓存宽度公式 | 不能把 MLA 简化成一个 KV head；实现可吸收上投影，不必显式恢复完整 K/V |
| 论文质量与性能 | 按来源列举结果并限定模型、训练、任务、硬件 | 不从单个论文推出“GQA 总接近 MHA”或“MLA 总优于 MHA” |
| FlashAttention、PagedAttention、MoE | 只做作用层面区分并链接各自主讲章节 | 分别属于 kernel、serving cache 管理、FFN 路由，不与 attention 架构混为一种缓存格式 |
| KV 容量与服务成本 | 本章只给架构元素公式；详细容量、请求分布和分配交给第 11 章 | 避免重复章节和将简式误读为真实 serving 预算 |
| SSM、线性注意力、门控 MLA | 从本章删除宽泛前沿架构盘点，链接第 24、29、42、63 章 | 本章集中解释 MHA/MQA/GQA/MLA，不提前替代后续专题 |

正文从旧有作者工作流、臆造评分/demo 和超宽泛模型盘点重写为 4.1—4.11 的读者教学路径。已将“缓存元素节省比例”限定为匹配宽度条件下的算术关系；未新增代码，现章没有需运行的代码块。

## 配套同步与静态验收

- 同步 `EXERCISES.md` 31C：去掉已删除的成本审计 demo，改为比较原始论文和 checkpoint/latent 转换证据。
- 同步 `INTERVIEW_BANK.md` 31D：删除无本章证据支撑的 Llama 2 选型陈述，改用来源条件说明共享 head、latent cache、uptraining 和独立 kernel/serving 层面。
- 同步 `PAPERS.md`、`GLOSSARY_EN_ZH.md` 和 `KNOWLEDGE_GRAPH.md`：补齐 MQA、GQA、DeepSeek-V2、MHA2MLA 的研究入口及边界；移除由旧 demo 引入的 KV Cache Saving Rate / Attention Architecture Gate 和 head 审计节点，保留其他项目使用的 KV Cache Gate 术语。
- 静态检查：H2 4.1—4.11 连续；12 个 Markdown fence 标记成对；行内 `$` 定界符逐行配对；全角标点与数学定界符之间的空格已检查；17 个 Markdown 链接中的本地目标均存在；章节内部公式、表格和章节回链已核对；`git diff --check` 通过。
- 未运行代码、测试、模型、API、benchmark 或 PDF 构建/更新。GitHub/目标平台 MathJax 渲染、外链终审和第二十一册整册连续复读留待后续门禁；质量与吞吐仍需具体模型和硬件上的独立验证。
