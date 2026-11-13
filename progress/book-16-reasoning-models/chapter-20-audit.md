# 第二十章审计：GLM-5.3 长任务环境与验证器

审阅日期：2026-11-13
整册连续复读：2026-11-13

## 范围与章级扫描

- **范围：** 以 GLM-5.3 发布方披露的长周期环境和 verifier 为案例，解释任务契约、环境反馈、奖励捷径、公开接口、第三方测量和 RL 轨迹的证据边界。通用工具执行、Agent Runtime 和 Harness 分别回链第十九章、第十七册和第二十册。
- **章级扫描：** 2026-11-13 定向复核 Z.ai GLM-5.3 文档及迁移指南、公开 Hugging Face 模型仓库与固定配置、Artificial Analysis 当前 release/index 页面、DataCurve/DeepSWE 当前主页与留存配置、SAO 与 IndexCache 原论文，以及固定 slime 源码和提交。研究事实、URL/版本、阅读范围及不能外推的事项记入[GLM-5.3 来源记录](../../research/model-update-2026-09/glm-5.3-source-notes.md)；本文件只记录审阅和修改结果。
- **候选主题去向：**
  - 长周期环境生成、judge agent、oracle/no-op/unsolved-state 和 reference-free verifier：纳入 §§20.1—7、20.16；官方披露不等于已独立验证 verifier 的完备性或真实任务成功率。
  - 公开接口、模型配置和工具参数流：纳入 §§20.2、20.9、20.18；分别限定 API 文档、固定 HF revision 与迁移指南，未从配置推导托管实现。
  - 第三方排行榜和 coding-agent 结果：纳入 §§20.15、20.20；保留 AA 当前 v5.0.1 与旧版快照的口径差异，DataCurve 汇总与历史精确明细绑定不同展示字段、harness 和时间。
  - SAO、IndexCache 与异步 RL 机制：纳入 §§20.17、20.19 作关联背景及合成教学推演；SAO 论文所述部署对象是 GLM-5.2，不能作为 GLM-5.3 专属实现证明，IndexCache 也不能证明生产 kernel 或检索表现。
  - Compaction、轨迹分叉、checkpoint 和 archive：纳入 §§20.21—23；绑定 slime 固定 revision/提交，明确其是通用训练系统案例。1,024 token 阈值只属于固定 trajectory 实现，不是 GLM API 的压缩阈值。
  - 未发现需要新增章节或改变第十六册边界的独立候选；通用 Agent 和生产 Runtime 仍由既有跨册入口承接。

## 正文结构、篇幅与例子

- 正文分为三段连续解释单元：§§20.1—14 从长任务问题推进到任务环境、verifier 门禁、奖励捷径与证据识别，作为核心单元；§§20.15—20 用第三方测量、训练系统、工具协议和关联算法限定“哪些结果能支持什么”，作为辅助证据单元；§§20.21—23 从未知的 GLM compaction 实现出发，依次讨论通用协议审计、训练轨迹分叉和恢复基础设施，作为辅助技术案例单元。§20.14 是第一段的局部收束，章末另有贯通结论 §20.24。
- 按连贯解释性中文计数，排除标题、表格、公式、代码、状态标记和清单：连续复读后核心 §§20.1—14 **3,356 汉字**（下限 1,800）；辅助 §§20.15—20 **1,710 汉字**、辅助 §§20.21—23 **1,606 汉字**（各自下限 1,000）。字数达标不替代机制、证据与边界审查。
- §20.6 的合成吞吐例：基线 80 samples/s、目标 100、完成态 120；增量为 120−80=20 samples/s，相对基线提升 20/80=0.5，即 50%。正确性测试、输出顺序和 verifier 完整性仍是独立门槛，不能被吞吐数字代替。
- §20.19 的 SAO toy 将三条构造轨迹 `[1, 3, 2]` 秒的同步等待算为 `(3−1)+(3−3)+(3−2)=3` 秒；新旧 log-prob 差 `[0.02, −0.30, 0.25, 0]` 对应 ratio `[1.0202, 0.7408, 1.2840, 1.0000]`，在 ±0.2 clipping 下得到 `[1.0202, 0.8, 1.2, 1.0000]`；标量 critic proxy 的两步 MSE 从 1.00 降至 0.64。输入为合成值，不是论文或模型复现。
- §20.22 以固定 slime revision 的 1,024 token fork threshold 做合成推演：700 token 新响应中 684 token 前缀相同、尾部 16 token 漂移，属于短漂移重对齐示意；1,100 token 或更早历史发生漂移时分叉。两个例子只解释固定实现的训练样本边界，不外推成 GLM-5.3 compaction 规则。

## 公式、代码、证据与链接

- **公式与呈现：** 5 个 math 围栏逐项检查变量/前提及数值或状态例；连续复读将状态转移、二值 verifier 奖励/成功事件、压缩器输入/保留键/输出，以及结果归因因素的定义移至公式前。`\operatorname` 为 0。全章 22 个围栏标记成 11 对，其中 1 个 Python 块、5 个 math 块和 5 个 text 块。对照表承载任务契约、benchmark 口径、工具事件、状态恢复和训练轨迹差异；没有 Mermaid 图，也没有需要依赖图色才能理解的关系。
- **§20.6 代码：** Python 3.12.9、标准库；按 Markdown 原块提取执行，stdout 为 `incomplete: throughput target not met | success | invalid: protected verifier changed`。覆盖目标/吞吐的 NaN、Infinity、负值、布尔和字符串，非布尔 verifier flags，以及测试失败、顺序变化和受保护 verifier 修改。代码仅验证合成 artifact 合同。
- **§20.21 代码：** `python3 research/model-update-2026-09/code/glm53_compaction_contract_audit.py`，CPython 3.12.9、标准库，退出码 0；三种截断状态通过审计，丢失 pending effect 和孤立 effect 两种坏状态被拒绝。脚本不调用模型或外部工具，不是 GLM-5.3 实现。
- **§20.19 代码：** `python3 research/model-update-2026-09/code/sao_async_rl_toy.py`，CPython 3.12.9、标准库，退出码 0；观察到等待 3.0 秒、ratio `[1.0202, 0.7408, 1.284, 1.0]`、clip mask `[true, false, false, true]`、clipped ratio `[1.0202, 0.8, 1.2, 1.0]`、GAE `[0.592, 0.6]` 和 critic loss `[1.0, 0.64]`。仅为确定性教学 toy。
- 连续复读只修改公式变量和证据口径的解释；§20.6、20.19、20.21 的代码、输入输出、依赖和环境均未改变，沿用各自验证记录，不重新执行。
- **上游源码边界：** 固定 slime branching test 的模型、GPU 数和覆盖范围作为源码测试设计引用；未运行该 4-GPU 集成测试，也不把它记为本地执行或 GLM benchmark。正文逐处区分源码行为、文档说法、外部测量和本地 toy。
- **链接：** 35 个 Markdown 行内链接（29 个外部、6 个本地）；全部本地目标存在。简介、目录中的第二十章摘要与本章范围、跨册边界和证据分类一致。研究来源记录仅收纳新知识及证据，不记修改流水。

## 配套同步与遗留风险

- 已更新第十六册简介与目录的第二十章摘要、问题描述和交接。`EXERCISES.md`、`PROJECTS.md` 已有两个教学脚本的入口；本轮补回了缺失脚本文件，没有重复粘贴练习内容。`PAPERS.md` 已覆盖相关论文入口，无需重复增加条目。
- `git diff --check` 通过；章节公式与代码围栏配对，局部链接静态检查通过。GitHub 页面 MathJax 实际渲染未从未提交工作区验证；本章无 Mermaid 图。
- 动态 API、榜单和 DataCurve 页面会变化，引用须继续绑定核验日期、模型配置和 harness。尚未公开的 GLM-5.3 compaction schema、完整训练配方、托管服务实现、真实 verifier 假阳性率和独立复现仍属未知，不影响本章以证据边界为中心的教学结论。
- 本章单篇门禁和连续复读完成；第十六册 20/20 章连续复读已完成，整册链接/交接终验及 GitHub MathJax 页面渲染尚待最终收口。全套书修书 goal 继续 active；按要求不更新 PDF 或 `plan.md`。
