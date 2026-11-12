# 第一章《Reasoning 总览》精修记录

审阅日期：2026-11-12  
状态：章节正文已精修；代码、公式、链接和简介摘要已同步。按用户要求不更新 PDF。

## 范围与章级扫描

- 通读全章 §§0.1–19、练习和资料索引；聚焦推理能力与可见轨迹的区别、模型/策略/环境边界、CoT/采样/verifier/search/RLVR 的方法关系、预算单位、reasoning API 合同、评估污染与轨迹安全。
- 本次定向检查使用原始论文、同行评审会议论文和 provider 官方文档。主要来源包括 CoT 与 Zero-shot CoT、Self-Consistency、GSM8K/verifier、过程监督、Tree of Thoughts、MATH、HumanEval、DeepSeek-R1 v2、Snell 等人的 test-time compute 研究、FACT-E、Hidden Thoughts Are Not Secret、TRACE，以及 OpenAI 和 Anthropic 当前 API 文档。
- 新主题去向：

| 候选内容 | 证据与去向 |
|---|---|
| CoT、self-consistency、verifier、process supervision、ToT | 保留为全章的机制地图；分别交第 2—6 章完整讲解。论文效果只按其任务、模型与评测协议表述。 |
| test-time compute 与按难度分配预算 | 保留预算向量、质量—成本关系和停止条件；第 7 章讲策略族，第 13 章讲预算计量，第 18 章讲动态策略。 |
| RLVR 与 DeepSeek-R1 | 本章只介绍可验证奖励及论文公开路线；具体训练阶段交第 8、17 章，不外推为通用 recipe。 |
| 数学/代码 benchmark 与 contamination | 保留任务适用边界、pass@k 条件和语义近重复风险；详细评估交第 9、10 章。原 GSM8K 链接误指向 arXiv:2110.14116（脑动态论文），已改为数学验证器原论文 arXiv:2110.14168。 |
| 轨迹忠实性、暴露和安全评估 | 纳入 FACT-E、Hidden Thoughts Are Not Secret、TRACE 的限定性结论；不声称可见文本等于内部状态，也不外推闭源服务风险率；深入内容交第 2、10、11 章。 |
| provider effort、thinking levels 和跨轮状态 | 只描述访问日期对应的公开接口合同；第 13—15、18—19 章分别承担预算、档位、状态、动态控制和工具事件的主讲。 |

没有发现需要拆章、并章或新增主线的证据。本章是全册地图；各方法小节承担导读，不替代链接章节的完整主讲。

## 正文修改

- 为章节导读补齐第 2—20 章的相关入口；更新第十六册简介中的第一章短摘要。
- 将 CoT、Zero-shot CoT、self-consistency、verifier、process supervision、ToT、test-time compute、DeepSeek-R1、benchmark 和安全轨迹相关事实的原始来源链接放到对应论断附近，并在资料索引中使用可读标题链接。
- 收窄 API 说明：provider 的 effort/thinking/budget 字段按模型、接口和版本解释；不当作统一计算单位或内部机制。更新 OpenAI/Anthropic 文档入口和访问日期，去除未在正文使用的 Gemini 资料入口。
- 将“动态预算比固定高预算更接近真实系统”改为“动态预算的动机与边界”；预算和成本公式改为区分多维资源向量与统一计价标量，避免把 token、GPU 秒、工具次数和人工分钟直接相加。
- 将 §13.4 从“预算分配”改名为“成本与资源账本”，使标题与向量记录一致；将演示指标改为 tokens_per_verifier_success，明确它只是合成数据的 token 账本，不是货币或 GPU 成本。
- 删除资料索引末尾的作者联网访问日志，改成面向读者的文档访问日期与版本边界说明。
- 代码修复：平票时要求显式策略；只接受严格的整数 0/1 步骤标签；对 Python 任意精度整数不调用浮点有限性检查，仍拒绝 NaN/无穷浮点分数。

## 验收字段

- **length_status：** 全章作为一个连贯的核心总览单元，解释性汉字实计 13,042（剔除代码、公式围栏、表格、标题、状态标记、维护性索引和空白），高于 1,800 汉字下限。CoT、采样、验证、搜索等小节作为导读并有明确主讲章节链接，不作为独立完整教程单元重复计算。
- **example_status：** 水箱例子逐步计算净流入 3−1=2 升/分钟及 20÷2=10 分钟；合同例子核算 100000×(1−0.05)×(1+0.06)=100700 元；§14 用明确标记的合成候选数据说明 greedy、投票、verifier、pass@k 与 token 账本。示例结果不声称来自真实模型或生产测量。
- **evidence：** 原始论文和官方 API 文档直接链接到相关结论。动态接口页面核对日期为 2026-11-12；DeepSeek-R1 固定为 v2。相关来源证据已在 research/test-time-compute-and-evaluation-2026.md、research/generalization-and-benchmark-contamination-evidence-2026-11-09.md 等研究记录中保存；章节审阅和修改历史仅记于 progress/。
- **formula_status：** 核对预算向量、A(B) 的可比条件、单位成功成本、动态预算向量、pass@k、步骤准确率、搜索节点数和合同付款式。成本只有在统一计价口径下才合并；各公式的关键变量、定义域和边界已在正文说明。
- **code_status：** §14 独立 Python 示例在 2026-11-12 用 Python 3.12.9、标准库运行；从 Markdown 提取完整代码块执行，实际 stdout 与正文预期输出一致。验证范围是本章合成数据，不代表真实模型、API、GPU 或生产服务结果。边界检查确认：投票平票拒绝、bool 步骤拒绝、任意精度整数 verifier 分数接受、NaN 拒绝、bool 形式的 k 拒绝。
- **link_status：** 新增章节内链指向对应的第 2—20 章；全章 70 个 Markdown 链接中，本地相对链接逐一解析，无断链。核心论文、ACL Anthology 页面及 OpenAI/Anthropic 官方文档链接已核对。
- **visual_status：** 复核已有方法对照表、层级表、步骤轨迹、预算向量和计算式；当前总览用这些现有表达已能读清依赖和数据流，不新增 Mermaid 图，不产生 GitHub 渲染待核项。
- **sync_status：** 已更新 book-16-reasoning-models/简介.md 的第一章摘要；目录路径与本章结构一致，无需调整 TOC。未更新 PDF、plan.md、题库或其他索引。
- **open_risks：** API 文档会按模型和时间变化，进入第 13—15、18—19 章时须再次核对目标接口；本文合成代码和论文结果均不代表生产表现。第 2—20 章仍需逐章审阅。

## 下一步

进入第二章《Chain-of-Thought》，围绕可见轨迹的机制、效果、忠实性和训练用途做独立章级扫描；继续保持 PDF 不更新。
