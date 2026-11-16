# Code Agent 评测：Terminal-Bench 版本与长程任务来源记录

资料核对日期：2026-11-16。本文只记录评测任务集的公开版本与范围，不记录书籍审阅过程或模型成绩。

## Terminal-Bench v4.0.0

- **来源与版本：** Harbor 官方 [Terminal-Bench GitHub 仓库](https://github.com/harbor-framework/terminal-bench)及 [v4.0.0 release](https://github.com/harbor-framework/terminal-bench/releases/tag/v4.0.0)，release 日期 2026-08-26；访问日期 2026-11-16。
- **实际阅读范围：** README 对 benchmark 类型与持续版本发布的说明；v4.0.0 release 的任务新增、移除和修改记录。
- **来源支持的事实：** Terminal-Bench 将自身描述为面向困难、广泛 Agent 工作任务的持续 benchmark，以 tagged release 发布。v4.0.0 release 明确列出被移除和修改的任务，并将 4.0 作为版本发布。因此跨 release 的分数需要检查任务集版本，不能假设题目集合保持不变。
- **证据等级与边界：** `source-backed`（benchmark 官方仓库与版本记录）。这支持版本管理和任务范围描述，不说明任一模型在生产任务中的成功率，也不单独支持跨版本分数可比。

## Terminal-Bench Challenges

- **来源与版本：** Terminal-Bench 官方 [Terminal-Bench Challenges 公告](https://www.tbench.ai/news/terminal-bench-challenges)，页面日期 2026-06-18；访问日期 2026-11-16。
- **实际阅读范围：** 公告对任务长度、资源投入、目标项目规模和与原版 Terminal-Bench 关系的描述。
- **来源支持的事实：** 该公告将 Challenges 描述为长时程、token 密集型的单任务评测，目标是衡量 Agent 进行大规模自主项目工作；官方称完成单项 challenge 相当于专家团队数月工作。此处“数月”是发布方对任务规模的描述，不是独立工时测量。
- **证据等级与边界：** `source-backed`（benchmark 发布方说明）。该描述支持与常规终端任务集的任务形态区分；不支持将 Challenge 结果与 issue 修复榜或其他 harness 的数字直接比较。

## 复核边界

比较 Agent 结果时，除模型 ID 与 effort 外，还要固定 benchmark release、agent/harness、容器/仓库 revision、硬件、资源与超时、重试、任务分母和 verifier。页面动态更新时，记录带日期或固定 tag 的快照；本文件没有收录或推导模型排名。
