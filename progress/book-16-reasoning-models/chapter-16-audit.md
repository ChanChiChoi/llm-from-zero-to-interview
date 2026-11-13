# 第十六章审计：Multi-Agent 与 Ultra Test-Time Compute

审阅日期：2026-11-13

## 范围与章级扫描

- **范围：** 精修多路径推理中的任务分解、并行效率、协作/搜索拓扑、候选与验证、预算和公平评估；避免把本章扩成通用 Agent Runtime 手册。
- **章级扫描：** 2026-11-13 阅读 ACL Anthology 的 PaCoRe、ParaThinker 论文摘要和元数据，以及 arXiv 的 Language Agent Tree Search、Multiagent Debate 摘要；对照已有 Self-Consistency、Tree of Thoughts、test-time compute 研究记录。扫描问题为：并行工作具体增加什么计算/信息；相同解码 token 预算与 wall-clock 收益如何区分；协调、验证和错误相关性怎样进入成本与质量判断。
- **候选主题去向：**
  - Self-consistency、debate、LATS、PaCoRe、ParaThinker：纳入 §16.2 作机制层次对照，区分最终答案采样、候选交互、状态空间搜索、跨轮并行消息协调和单架构并行路径。
  - PaCoRe 与 ParaThinker 的 2026 结果：正文只引用论文摘要明确给出的作者结果并注明 benchmark、模型、预算和实现边界；摘要缺少完整对齐条件时不做跨系统排名。证据记录见[推理时计算与评估研究记录 §9](../../research/test-time-compute-and-evaluation-2026.md)。
  - Worker artifact、revision 和冲突：保留与并行候选归因、去重、验证直接相关的最小 contract。
  - 取消、外部副作用、长程状态持久化、通用工具权限/runtime：收窄到边界提示；详尽工作流交第十七册与第二十册，工具事件状态机交第十九章。
  - Amdahl、预算账本和边际扩张条件：保留为并行设计的计算分析工具；明确不是具体服务性能预测。

## 正文结构、篇幅与公式

- 保留任务图、关键路径、证据独立性、拓扑、worker 产物、预算、成本、停止条件、评估和安全主线；改写方法演进段加入近期并行推理研究。
- 将 worker lifecycle 与 workspace/权限从通用 Runtime 说明压缩为并行候选状态和权限分离要点；把跨仓库故事改为并行取证划分，并链接第十七、二十册相应入口。
- 核心讲解单元解释性汉字实计 6,363 字，超过 1,800 字下限。
- 删除仅定义任务图的孤立 G=(V,E) 与均衡工作量冗余近似式；保留 8 个 display-math 块，全部补足变量、条件与可复算例：
  - 关键路径公式：4/6/5 秒 worker、效率 0.8、4 秒后处理，得到 11.5 秒；串行比较为 19 秒。
  - Amdahl 上界：20 秒串行、80 秒并行、4 个 worker 得理想上限 2.5 倍，无限 worker 上限 5 倍。
  - 同单位预算：85/100 credits 用量，余额 15 credits；预算向量给出 token、调用、deadline 与副作用的独立单位。
  - 六项成本求和为 120 credits；3 个成功任务下单位成功成本为 40 credits。
  - 边际效用教学例得 2 credits；扩张门槛例按质量、成本和风险三个条件逐项判定。
  - 示例数值均为合成教学输入，不表示模型或生产数据。operatorname 为 0。

## 代码、证据、链接与同步

- §16.17 代码状态标记：已验证，日期 2026-11-13，CPython 3.12.9，仅标准库。唯一 Python 块从 Markdown 提取执行；实际 stdout 与预期一致：

~~~text
artifact_valid True
budget {'total': 100.0, 'used': 82.0, 'remaining': 18.0}
conflicts ['retry-policy']
unit_success_cost None
~~~

- 边界检查 12 项：重复 artifact ID、空证据 hash、不可哈希状态、未知状态、verified/status 矛盾、预算超额、NaN/负数/布尔预算、超大整数、零成功数、布尔及负成功数。异常输入按契约拒绝，零成功数返回 None。
- 本章链接共 13 个，其中 8 个本地章节/书册链接均有效；引文直达原论文或正式出版页面。代码和数学/文本围栏配对，git diff --check 通过。
- 简介、目录已有的第十六章摘要和阅读路径仍准确，无需修改。复核 INTERVIEW_BANK、EXERCISES、GLOSSARY 与 KNOWLEDGE_GRAPH，已有多 Agent、Agent Debate、Multi-Agent TTC、失败模式、预算与审计材料，无需重复添加。
- 任务依赖文本图、方法拓扑表和预算对比已覆盖主要视觉机会；未增加 Mermaid。GitHub 页面 MathJax 实际渲染留在平台验收阶段。

## 遗留风险与交接

- PaCoRe 与 ParaThinker 的 2026 数值只按 ACL Anthology 摘要核对；没有据此推断完整 prompt、harness、硬件或生产延迟条件，也未执行本地复现实验。
- 多 worker 的收益、共因错误和尾延迟必须在目标 workload、固定预算和真实 verifier 下评估，教学公式不是线上性能保证。
- 第十六册第 1—16 章完成逐章审阅和精修；下一入口为第十七章《DeepSeek-R1 的 RLVR 与蒸馏路线》，第 17—20 章和整册连续复读/最终门禁待完成。按用户要求不更新 PDF 或 plan.md。
