# 第九册第一章：数据总览定向扫描与精修记录

## 范围与问题

- 检索日期：2026-10-28；主要检索窗口：2024-10 至 2026-10。
- 章级问题：第一章如何解释“数据对象经处理后怎样影响模型能力、行为、评估和治理”，同时给第二至第十二章留下清楚的主讲边界？近期研究是否改变了对合成数据、数据配比/顺序或来源血缘的总览性描述？
- 检索类型：近年综述、同行评审原始论文、官方论文集页面和工程文档。定向查询训练数据组织/排序、合成数据与递归生成、mixture 估计、provenance/traceability 和 dataset cards。检索用于发现并核读候选，不声称穷尽。

## 证据与范围

1. [Demystifying Synthetic Data in LLM Pre-training](https://aclanthology.org/2025.emnlp-main.544/)，EMNLP 2025。作者报告超过 1,000 个模型变体和 100,000 GPU 小时；实验覆盖 100M—3B 参数及最高 200B token。论文区分改写网页和教材式生成数据，报告混合收益依数据类型、模型尺度和预算而变。作者对更大模型/数据尺度的结果包含拟合外推，不能当作 frontier 训练复现。第一章只用它说明“合成数据不是同质类别、不能有固定通用比例”；分类型生成与质量门禁由第七章主讲。
2. [Collapse or Thrive: Perils and Promises of Synthetic Data in a Self-Generating World](https://proceedings.mlr.press/v267/kazdan25a.html)，ICML 2025。作者比较递归替换为合成数据、累积历史数据和每代固定大小子集等三种工作流，并在高斯估计、核密度估计和语言模型微调等任务设置中观察不同退化模式。此证据说明结果受数据维护和训练工作流影响；它不能证明所有语料递归生成均安全或均会坍塌。
3. [Data Mixing Laws](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cc84bfabe6389d8883fc2071c848f62a-Abstract-Conference.html)，ICLR 2025。论文拟合数据来源比例与语言模型表现关系，并在作者特定 RedPajama、模型和训练预算中用小规模 mixture 推测未测比例。第一章只据此说明配比需要与目标评估联合测量，不把其拟合结果外推成通用最优配方；详细实验方法交第五章。
4. [Demystifying Data Organization for Enhanced LLM Training](https://arxiv.org/abs/2605.30334)，ACL 2026 主会论文。作者从预训练与 SFT 实验讨论 sample-level data ordering、循环调度、课程连续性和局部多样性。该方向扩展了“数据设计只有来源占比”的简化表述。第一章在配比小节提示顺序也会改变暴露节奏，方法对比及可迁移边界交第五章。
5. [Tracing the Data Trail: A Survey of Data Provenance, Transparency and Traceability in LLMs](https://arxiv.org/abs/2601.14311)，2026 年 arXiv 综述预印本。基于作者整理的 95 篇工作归纳 provenance、透明度、traceability 及隐私、偏差等方向。适合作为概念地图，不证明模型生成内容可普遍逆向到逐条训练样本。第一章仅保留内部血缘与输出归因的边界；具体 manifest 和治理交第十一章。
6. [Hugging Face Dataset Cards 文档](https://huggingface.co/docs/hub/main/datasets-cards)，官方平台文档。dataset card 可说明数据内容、构造方式、元数据、限制和偏差；它是公开文档入口，不替代内部数据血缘、授权审查和删除证据。

## 候选主题去向

| 候选 | 章内决策 | 去向与理由 |
| --- | --- | --- |
| 合成数据类型与递归风险 | 合并到 §§1.2、1.18，压缩为证据边界提醒 | 是总览理解数据变化的必要例子；分类型 generation、teacher 验证和比例门禁交第七章，避免在概览重复主讲 |
| mixture 的定量拟合与样本排序 | §1.9 增加一段近期研究导航 | 能纠正“比例表就是完整策略”的误解；具体公式、proxy、排序算法和消融交第五章 |
| 数据 provenance、透明度与 dataset card | §§1.12、1.14 保留内外部文档责任区别 | 读者需要知道公开说明与内部逐样本 lineage 不等价；schema、权限、版本和删除沿第十一章主讲 |
| 新开一章或新增完整讲次 | 不新增 | 候选均能由现有第 5、7、11 章完整承接，不改变书册契约或主线 |

## 结构决定与改动

- 第一章继续作为一篇核心总览，中心解释链为：数据对象与样本分布 → 生命周期变换 → 能力/风险/评估表现 → 观测、治理和后续章节的工程交接。
- 收束 §1.7 过滤、§1.8 去重/污染、§1.9 mixture、§1.12 治理和 §1.14 版本管理的重复细节；保留决策地图、核心变量、可复算指标和章节链接，详细算法交相应主讲章。
- 删除 §1.1.3 与总览图重复的纯文本反馈箭头，改为解释“反馈先是线索、经治理后才可能进入下一版”；将 §1.5 主 pipeline 改为 Mermaid 生命周期图。虚线反馈只表示评估后定位到上游检查，不画成线上日志自动回灌训练。
- 修正 `D_effective` 介绍中的变量标注，明确评分代理不可跨来源直接比较；补一组与代码相同输入的 token 保留率、质量加权均值、领域 token 配比及独立风险率手算。
- 将 demo 的 mixture 统计由样本数改为 token 数；为 `None` 风险率增加安全格式化；固定样例输出与空输入路径均运行复核。
- 目录已建立的第 1—12 章交接路径在正文显式链接。练习增加样本顺序/调度和统计分母；面试题补充配比与排序需分开验证。

## 例子与代码验证

- 固定输入：原始 token `6,190`；保留 token `3,320`；保留率 `0.536`；加权质量 `0.845`；web/code/math/safety token 为 `1,200/900/700/520`；五个风险率（duplicate、contamination、low quality、PII、license gap）各 `1/9≈0.111`。污染、PII、许可缺口导致 `hold_for_repair`。
- Python demo 从 Markdown 提取至 `/tmp/codex-book09-ch01-demo.py`，命令 `python3 /tmp/codex-book09-ch01-demo.py`，Python 3.12.9 标准库；固定输出与正文块逐行一致。
- 空输入变体返回 `retention=None`、`weighted_quality=None`、空 mixture、各风险率为 `None`，动作包括补充样本/恢复非空集，最终 `hold_for_repair`；单条全过滤变体返回 `retention=0.0`、`weighted_quality=None`、空 mixture，并保持 `hold_for_repair`。未将缺测判为零风险。
- 没有新增仓库测试文件或运行测试套件；只运行该教学 demo 及指定边界输入。

## 章节门禁状态

- 解释正文是一篇核心讲解单元；按排除标题、围栏、表格、引用条目、练习和状态行后的机械口径统计为 8,875 个汉字，高于 1,800 字下限。计数不替代结构与解释链复读。
- 公式、图表、代码、局部链接与代码区域维护文件名扫描通过静态核对；`git diff --check` 结果写入进度条目。
- 新增来源逐项打开核对；未做本章所有既有链接的 HTTP 全量扫描。当前修改尚未发布到 GitHub，Mermaid/MathJax 页面渲染未确认；无目标训练栈、目标模型、数据平台或生产数据验证。
- 完整正文门禁保持开放，直到目标平台图表/公式渲染、全章外链扫描和第九册连续复读完成。
