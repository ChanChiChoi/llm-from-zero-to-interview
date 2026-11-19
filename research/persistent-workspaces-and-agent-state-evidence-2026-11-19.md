# Persistent Workspace 与 Agent 状态边界：外部证据

检索日期：2026-11-19。检索范围：版本控制中已提交快照、暂存区与工作树的区别；Agent 会话、任务、运行环境和文件状态的生命周期；文件原件、派生检索视图、修改差异与交付物的版本关系；用户在任务进行中修改共享工作区的研究设置；workspace 级任务的评估范围与教学组织样本。使用英文关键词检索并直接打开官方文档、课程材料和论文原文；关键词包括 Git HEAD index working tree、Git staging area course、agent workspace session state、versioned workspace content hash agent、workspace benchmark file dependencies、coding agents users edit shared workspace。

## 版本控制与文件状态

### Git 官方资料

- [Git User Manual：Making a commit](https://git-scm.com/docs/user-manual.html)：在线用户手册，访问日期 2026-11-19。阅读创建提交、HEAD 与 index/working tree 对比、工作树定义部分（“How to make a commit”及术语表）。
  - 支持：index 保存准备提交的树快照；git diff --cached 比较 HEAD 与 index，git diff 比较 index 与 working tree；git add 记录当时的文件内容，之后继续编辑不会自动改变 index 副本。手册说明 working tree 通常是 HEAD 中内容加上尚未提交的本地修改。
  - 边界：Git 的索引/提交合同不能代表所有 Agent workspace 的存储或恢复方式；Git diff 和内容一致性也不证明语义正确或任务已验收。
- [Pro Git, 2nd Edition：Recording Changes to the Repository](https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository)：第二版“Git Basics”章节，访问日期 2026-11-19。阅读 staging、commit snapshot 和未暂存文件说明。
  - 支持：提交记录的是暂存区准备的快照；没有暂存的修改仍可留在磁盘工作树中。
  - 边界：这是 Git 教程对其版本控制流程的解释，不是 Agent 状态持久性的保证。
- [git-worktree reference](https://git-scm.com/docs/git-worktree)：在线命令参考，访问日期 2026-11-19。阅读 DESCRIPTION 与 add 命令部分。
  - 支持：一个仓库可挂接多个 working tree；linked worktree 有各自的工作目录及 HEAD、index 等 per-worktree 文件，同时共享仓库对象数据。
  - 边界：多个 checkout 的存在不意味着模型会话、任务历史或外部执行状态自动共享或恢复。

### 教学组织样本

- [MIT Missing Semester 2026：Version Control and Git](https://missing.csail.mit.edu/2026/version-control/)：课程讲义“Staging area”及基础 Git 命令。
  - 支持：教学先解释为何 Git 允许用户选定下一次快照包含哪些修改，再引入 staging area、git add 和比较快照的命令，提供一个把概念接到操作的组织样本。
  - 边界：课程讲义说明教学选择，不构成该讲法改善学习效果的因果证据。

## Agent 会话与运行环境

- [OpenAI Agents API：Pre-warm sandboxes](https://developers.openai.com/api/docs/guides/agents-api/environments/prewarmed)：官方 API 文档，访问日期 2026-11-19。阅读 “How it works” 及多个 session 附着同一 environment 的说明。
  - 支持：文档将 environment 定义为 Agent 工具运行的 workspace，将 session 定义为 Agent 配置和 conversation；新建的 standalone environment 可由多个 session 共用文件、进程、已安装包和环境配置，而 conversation history 各自独立；共享环境中的修改对其他附着 session 可见。
  - 边界：官方文档明确旧环境仍可能是单 session lifecycle；这是该 API 对特定新建环境的合同，不能据此推导其他产品或旧环境的 session 隔离、workspace 生命周期和保留方式。
- [OpenAI Help Center：Using Codex Cloud](https://help.openai.com/en/articles/20001545-using-codex-cloud)：官方产品帮助，页面在 2026-11-19 显示 18 天前更新。阅读 task workspace 与恢复说明。
  - 支持：每个 Cloud task 使用独立隔离 workspace；继续同一 task 可以恢复未完成工作，新 task 不会取回另一个 task 的未提交改动；文档说明 VM 保存状态从最近一次启动或恢复起最多可恢复 7 天，并明确这不是 conversation history 的保留期限。
  - 边界：恢复窗口只适用于该产品文档描述的保存 VM 状态，不能概括为通用 workspace 或对话保留政策；产品合同可能变化。

## Agent 文件视图、并发编辑与评估

- Hua et al., [StagedWorkspace: A Versioned Workspace for Knowledge-Work Agents](https://arxiv.org/abs/2608.18050), arXiv:2608.18050v1, 2026-08-18。
  - 阅读 §3.1、§3.4、§4.2、§5 Limitations 及附录相关部分。论文把 native workspace、由文件解析出的检索记录、review diff 和最终提交物绑定到演进中的文件状态；解析记录标注源路径与内容 hash，源文件 hash 变化时记录标为 stale，刷新后才重新成为当前视图。
  - 固定模型/提示/解析器/检索器/评估器/预算的配对消融中，dual parsed/native view 相对 artifact-only 在 OfficeQA Pro Pass@1 报告 +8.3 至 +12.1 个百分点，相对 parsed-only 在 APEX-Agents 平均 rubric 分报告 +4.7 至 +9.2；APEX 的 57 个文件编辑任务中，显示 workspace diff 的测试组相对隐藏 diff 报告 +2.5 至 +8.5 个百分点。
  - 这些结果来自论文所选模型、单一 agent stack 和两个 benchmark 的特定消融。论文指出最终评分没有覆盖每个中间检索/同步/编辑状态，未将 workspace-state contract 与 parser/retriever 影响完全分离，并存在许多 read-axis 变更无法修复的 all-arm-zero 任务。因此它支持“文件原件、派生视图和 review 结果应说明各自对应的版本”这一设计问题，不证明该框架是唯一规范，也不支持普遍效果结论。
- Zhou et al., [Workspace-Bench 1.0: Benchmarking AI Agents on Workspace Tasks with Large-Scale File Dependencies](https://arxiv.org/abs/2605.03596), arXiv:2605.03596v4, 2026-05-14。
  - 阅读摘要、数据集构造和统计部分。基准包含 5 个 worker profile、74 类文件、20,476 个文件、388 个任务及每任务依赖图，并以 7,399 个 rubric 检查跨文件检索和推理。作者报告所测模型与 harness 的平均分为 43.3%，最佳约 60%，人加工具结果为 80.7%。
  - 这些数字绑定该基准的工作区、任务、rubric、模型和 harness；不能推出所有 Agent 的普遍可靠度，也不等同于 workspace 持久化正确性。
- He et al., [Benchmarking Coding Agents When Users Touch the Code](https://arxiv.org/abs/2608.02499), arXiv:2608.02499v1, 2026-08-03。
  - 阅读问题定义、主结果、Co-Edit 对照及限制。SWE-Touch 在抽样的 SWE-bench Verified 任务中，于 Agent 运行轨迹内注入经过验证的、与目标冲突的代码修改及上下文消息；论文对 9 个模型的受控 Counter-Edit 条件报告平均 resolve rate 比 autonomous 条件低 7.7 个百分点，单模型变化为 -1.3 至 -16.5。较小的 task-aligned Co-Edit 对照平均变化为 -0.1 点。
  - 研究检验的是受控模拟的任务冲突修改，不代表真实协作编辑的完整分布；模型、任务、注入触发和评分均有限定。结果说明“有人改了工作区”不够确定处理方式，必须判断变更是否与任务目标冲突并检查最终状态。
- Mraz et al., [Tracking State Footprints: How Agents Can Transact](https://arxiv.org/abs/2610.03140), arXiv:2610.03140v1, 2026-10-02。
  - 阅读摘要、§§3—6。作者提出以 state footprint 描述 Agent 在自身上下文/状态、orchestrator 和外部系统中的读写集合，并将并发下的 stale reads 与 lost updates 作为事务化 Agent 系统的设计问题。
  - 该文是问题框架和未来系统方向，不是已实现的通用事务保证；其 PVLDB 引文页仍显示待补的卷页/DOI 占位符。
- Wang et al., [OpenHands: An Open Platform for AI Software Developers as Generalist Agents](https://proceedings.iclr.cc/paper_files/paper/2025/file/a4b6ad6b48850c0c331d1259fc66a69c-Paper-Conference.pdf), ICLR 2025。
  - 阅读摘要和 §1。论文把 Agent framework 分解为交互接口、运行环境和人/Agent 间交互机制，并介绍 Docker sandbox runtime 作为一种实现。
  - 这是平台架构实例，不定义所有 Agent 的持久化、恢复或保留期限。

## 未验证边界

产品 workspace 生命周期、跨 session 共享和数据删除语义需按具体产品/功能的当前文档核验。revision 或 content hash 能标识状态差异与派生视图新鲜度，但不能单独证明行为正确、权限合规或外部副作用成功。benchmark 的分数不能外推为一般能力；受控共享编辑实验不能替代真实用户协作研究。当前教学组织样本只说明公开课程如何讲 Git 暂存快照，不声称某种固定讲解顺序对所有读者更有效。
