# 第三十六章《SSM 会不会替代 Transformer：证据、争议与面试回答》精修审计

## 范围与诊断

按全套书精修的主题重写要求完成本章，保留章名、路径和编号。旧稿有 28 个 H2，重复讨论证据分级、实验控制、Pareto 选择、通用架构决策、hybrid 接口、serving 恢复和生产报告；其中多部分由第四十二章、第四十三章及服务相关书册承担。旧稿也把可复现实验清单和局部系统案例堆成多个相近小节，主问题“现有证据能支持哪一种替代主张”被稀释。

先按 02a 建立六节结构，再主题重写正文。新主线由“替代主张范围 → 复杂度与能力的边界 → 匹配规模实验 → 复制/检索与 ICL 的不同任务 → 新版本和 hybrid 结果 → 有边界的面试回答”组成。结构见[第三十六章大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/36-SSM替代Transformer的证据与争议.md)。

## 研究与内容

证据和来源阅读位置记录在[SSM 与 Transformer 比较研究记录](../../research/ssm-transformer-replacement-evidence-2026-11-23.md)。正文保留约 8B Mamba/Mamba-2/Transformer 对照中 1.1T 与 3.5T 的任务差异，并明确 3.5T 纯模型比较没有训练 Mamba；指出 MMLU 格式变化会改变结果。将复制/电话簿困难与函数/Markov ICL 正面结果并置，避免把 ICL、复制和 retrieval 合成一个能力标签。对 Mamba-3 限定到论文的 1.5B、2K cloze retrieval 实验；把 Mamba-2-Hybrid 结果标为 hybrid，并将生成速度说明为特定条件下的单 token 时间预测。

章节与第 34 章的任务级 Mamba 能力分析衔接，但不复写其完整 MQAR/ICL 研究；不重讲第四十二章的普遍架构选择，不写第四十三章的 hybrid 组件安排或 serving 接口。

## 配套同步

已为第二十一册目录添加大纲链接和 36.1—36.6 标题；新增练习 31AO、面试题 31AL；在 PAPERS.md 的 Mamba ICL 入口补充本章研究记录；在 KNOWLEDGE_GRAPH.md 增加跨架构替代证据路径。现有 SSM、ICL、Mamba-3、hybrid 等术语足够，未增重复词条。

## 验收与遗留

正文 H2 36.1—36.6 与大纲标题、顺序一致；正文无公式、代码围栏或图表。新增章节、结构大纲、研究记录、审计、目录、练习和面试题的本地目标均存在；练习 31AO 与面试题 31AL 各唯一。全文相对链接静态扫描未发现本章及本轮新增链接缺失；PAPERS.md 全文件扫描另发现 23 个既有缺失本地目标，均非本轮新增链接。`git diff --check` 通过。

本章无公式、代码或图表，不新增 MathJax、程序或 Mermaid 验收项。全册外链最终逐条复核、目标读者试读、第二十一册连续复读及既有章节公式/Mermaid 页面渲染门禁仍待全册总审。未运行代码、测试、模型、API、kernel、benchmark 或 PDF 构建；不更新 PDF 或 plan.md。全套书精修 goal 继续 active。
