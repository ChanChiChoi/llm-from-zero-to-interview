# 数据工程回归诊断与来源追踪证据

核验日期：2026-11-03

本记录只收录联网核验得到的来源事实、版本和证据边界，不记录书籍审阅过程或章节修改决定。

## DataComp-LM / DCLM

- **来源与版本：** Jeffrey Li 等，*DataComp-LM: In search of the next generation of training sets for language models*，[arXiv:2406.11794v4](https://arxiv.org/abs/2406.11794v4)，v4 于 2025-04-21 更新。
- **实际阅读范围：** 摘要、DCLM benchmark 与数据集构造章节、数据混合/过滤实验、结论和 limitations。
- **来源支持的事实：** DCLM 提供从 Common Crawl 抽取的标准化 240T-token 数据池、OpenLM 训练配方和 53 项下游评测；benchmark 允许比较去重、过滤和混合策略，模型规模覆盖 412M 至 7B。论文报告其逐阶段数据构造实验，并使用多个性能口径评估候选数据。§4.5 在 1B-1x 设置下按 67% Common Crawl、33% 外部精选来源混合；相对纯基底数据，Core 指标在 C4 上增加 2.2 点，在 DCLM-baseline 上下降 1.2 点。
- **证据边界：** 混合结果绑定论文的 1B-1x 规模、具体基底语料、来源组合、Core 指标和训练协议，不说明相同结果会迁移到其他任务或预算。作者还说明受算力限制，许多设计维度只逐项消融，未训练超过 7B 的模型，run-to-run variation 也未得到充分探索；大多数实验使用 GPT-NeoX tokenizer。因此它能支持“固定数据池、训练配方和评估设计有助于比较数据变化”的方法论示例，不能支持普遍最优过滤器、数据配比或生产效果。

## C4 语料来源与过滤影响

- **来源与版本：** Dodge et al., [Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus](https://arxiv.org/abs/2104.08758v2)，EMNLP 2021 camera-ready，arXiv v2（2021-09-30）。
- **实际阅读范围：** 摘要；确认 C4 来自单一 Common Crawl 快照，作者检查来源、文本内容及过滤删除结果。
- **来源支持的事实：** 作者发现 C4 中存在专利和美国军事网站等非预期来源、机器生成文本（包括机器翻译文本）及其他 benchmark 的评估样本；对被过滤数据的分析显示，blocklist 筛选对少数群体本人及其相关文本有不成比例的移除影响。
- **证据等级与边界：** source-backed（EMNLP 2021 论文）；观察只针对 C4 的快照和原过滤流程，不证明其他网页语料或 blocklist 必然产生相同结果。该案例支持过滤规则必须按语言、群体和下游任务检查，不能把“被过滤器删除”当作质量真值。

## OLMo 训练数据顺序与运行工件

- **来源与版本：** Dirk Groeneveld 等，*OLMo: Accelerating the Science of Language Models*，[arXiv:2402.00838v4](https://arxiv.org/abs/2402.00838v4)，v4 于 2024-06-07 更新。
- **实际阅读范围：** §3.3 Data 与 §5 Artifacts Released。
- **来源支持的事实：** OLMo 训练数据由 Dolma 的 2T-token 样本构成；论文称训练数据顺序和每个 batch 的精确组成可从发布工件重建，并提供重建完整数据顺序及检查各训练步骤所见数据的工具。
- **证据边界：** 这是该开放项目所发布工件的能力示例，不意味着其他模型或闭源系统也保存或开放相同粒度的数据顺序、训练 batch 或日志。

## DebugLM 学习式来源标签

- **来源与版本：** Wenjie Jacky Mo 等，*DebugLM: Learning Traceable Training Data Provenance for LLMs*，[ACL Anthology 正式论文页](https://aclanthology.org/2026.emnlp-main.182/)及[arXiv:2603.17884v2](https://arxiv.org/abs/2603.17884v2)。EMNLP 2026 正式论文，页 4168—4182；正式论文页于 2026-10 发布。
- **实际阅读范围：** 摘要、§§3.1–3.8 和一般效用结果附录。
- **来源支持的事实：** 方法在回答中学习输出训练来源标签，并用专门 debug 接口返回标签。实验采用 Llama-3-8B 与 Qwen-3-8B 两个基础模型，以及 TOFU、ChatDoctor、TruthfulQA、Beavertails、WMDP 五个数据集；分两阶段进行指令微调和基于 WMDP 的 GRPO。来源追踪指标按保留测试集上的来源标签 exact-match 计算；多来源黑盒设置对每条测试实例采样 10 次生成。
- **证据边界：** 这是特定训练、数据集和评测协议下的模型行为标签研究，不等价于对象、转换事件、训练快照或 batch 的工程 lineage。标签 exact-match 结果本身不能证明某个实际训练来源对模型行为具有因果贡献；该点是依据方法定义和测量协议作出的范围判断。

## LLM 数据来源综述候选

- **来源与版本：** Richard Hohensinner 等，*Tracing the Data Trail: A Survey of Data Provenance, Transparency and Traceability in LLMs*，[arXiv:2601.14311v1](https://arxiv.org/abs/2601.14311v1)，2026-01-19 提交。
- **实际阅读范围：** 摘要和引言；未完整阅读 35 页综述。
- **来源支持的事实：** 作者摘要称其分析 95 篇文献，并按 provenance、transparency、traceability 及 bias/uncertainty、privacy、tools 等主题提出分类。
- **证据边界：** 当前仅作为综述分类入口；未据此采纳技术实现结论，也不将其分类框架当作工程系统已具备来源追踪能力的证据。
