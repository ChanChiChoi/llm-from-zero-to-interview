# 第九册第二章：Web-scale 数据采集与解析扫描及精修记录

## 扫描边界

- 日期：2026-10-28。
- 问题范围：面向预训练数据的来源准入、robots 与用途边界、Web 快照格式、请求/重试口径、解析产物可回溯、多语言网页语料工程。
- 时间范围：重点查看 2024–2026 的网页语料/数据处理工作与当前 Common Crawl 文档；协议语义以仍适用的 IETF RFC 为准。
- 阅读方式：核对 IETF RFC 9309、Common Crawl 官方 Get Started 页面、FineWeb2 和 Data Compliance Gap 论文全文/摘要与 COLM 2025 官方录用列表，并阅读 robots.txt 爬虫遵从性研究摘要。检索不是领域穷尽证明；范围限于本章主题和需要核对的主张。

## 证据与候选去向

| 资料 | 支持的判断 | 证据边界 | 正文决定 |
| --- | --- | --- | --- |
| [Common Crawl Get Started](https://www.commoncrawl.org/get-started)，检索日 2026-10-28 | 官方入口当日列出 `CC-MAIN-2026-43`；WARC 保存原始抓取/响应信息，WAT 保存派生元数据，WET 保存提取纯文本 | 页面会持续变化；快照字段和实际记录应以具体 crawl 版本为准 | 纳入 §2.5，说明具体快照 ID 要入账；不把 WET 当作 WARC 的无损替代 |
| [RFC 9309: Robots Exclusion Protocol](https://www.rfc-editor.org/rfc/rfc9309.html)，2022-09 | IETF 标准化 robots 规则语法及 crawler 处理行为，并明确 robots 规则不是访问授权 | 不能代替许可、合同、隐私或特定用途的法务判断 | 纳入 §2.4 和文献表；将用途审查、robots 偏好与技术访问拆成独立状态 |
| [FineWeb2](https://arxiv.org/abs/2506.20920)，COLM 2025 | 使用 96 个 Common Crawl 快照构造约 20 TB、覆盖 1,868 个语言—文字系统组合的多语言数据；方法关键消融集中于 9 种语言，比较实验覆盖 14 种 | 作者指出低资源组合常很小，长尾覆盖不等于质量均匀或能力验证；任务、模型、token 预算绑定结论 | 纳入 §2.3.7 作为多语言采集和快照管理实例；语言过滤细节交第三章，去重交第四章，采样/再平衡交第五章，专项语料交第六章 |
| [Can Performant LLMs Be Ethical? Quantifying the Impact of Web Crawling Opt-Outs](https://arxiv.org/abs/2504.06219)，COLM 2025 camera-ready v2 | 论文在 FineWeb-Edu 上按 2025 年 1 月可见的 robots 规则回溯过滤；1.5B 模型、100B-token 从头训练的通用知识平均差异不显著，但结构化/对抗性事实任务有差异；continual pre-training 的医学结果依赖作者特定数据和总 token 条件 | 不是法律授权意见；“合规”是论文操作性定义；结果不跨模型、语料、时间点或领域外推 | 纳入 §2.4.1，作为可以量化合规过滤代价的研究例，不把性能结果写成许可判断 |
| [Scrapers selectively respect robots.txt directives](https://arxiv.org/abs/2505.21733v2)，2025-10 修订 | 对某机构 40 天匿名 web log 和受控 robots 实验分析 130 个自称 bot 及匿名 bot，报告更严格规则下遵从更弱、部分 AI search crawler 很少读取 robots | 特定机构、观测窗口和 bot 样本；研究问题是爬虫行为，不是版权/合同解释或训练集构造效果 | 作为候选记录，但不纳入正文主线：RFC 已足以支持协议语义；该实验不改变 source registry/解析接口设计，也不应被概括成所有采集器行为 |

## 结构判断与实施位置

- 保持本章为“来源登记 → 请求与快照 → 解析 → 证据交接”的连续单元；不新增小节主题或章节。
- 保留来源类型介绍、Common Crawl 形态和 PDF/HTML 解析，因为它们说明载体与解析风险；把完整质量评分、PII/秘密过滤、benchmark contamination、exact/near dedup、温度采样、mixture 和模型侧删除方法交给第三至第五、十一章。
- 用同一组合成输入逐步区分 source-gate 候选、逻辑 URL、HTTP 尝试、最终 2xx URL、支持的 HTML 响应和可用 artifact。正文手算重试影响与各层分母，Python demo仅核对采集/解析机制。
- Mermaid 展示 source registry、准入、collector、raw snapshot、parser、人工抽样、隔离和后续章交接；使用标准围栏、引用节点标签和带管道标签的虚线箭头。

## 未解决边界

- 没有在真实网站发请求；无法验证生产 rate-limit、robots 版本追踪、重定向、压缩传输或组织内许可流程。
- Demo 的 HTML 抽取和登录模板识别是教学逻辑，不是通用 parser 或安全检测器；PDF/OCR、代码仓库解析和多模态格式仍需独立流程。
- 目标平台 Mermaid 渲染、GitHub MathJax 页面渲染和全章外部 URL HTTP 扫描尚未完成；章节完成门禁因此保持开放。
