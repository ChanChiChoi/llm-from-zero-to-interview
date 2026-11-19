# 第二十一章《Persistent Workspace 与状态边界》审计

审阅日期：2026-11-19  
状态：章节扫描、正文重组与配套同步完成；书册最终连续复读尚未完成。

## 章节契约与读者路径

- **中心问题：** 长任务恢复时，对话、模型记忆、任务检查点、workspace 文件、CI 结果和外部系统的当前事实分别由谁证明？
- **读者结果：** 读者能区分状态来源；识别提交快照、暂存区、当前工作树和派生索引的版本差异；在旧任务基线与当前文件不一致时保留候选、检查冲突并在新 revision 上验证；将交付物、来源版本、验证结果和未解决事项一起交接。
- **真实前置：** Git、文件系统和工具调用基础；第 7 章的上下文/记忆，第 9 章的 trace 证据，第 17 章的检查点与外部副作用。
- **阅读主线：** 从 task-42 的 r17/r18 冲突引出事实归属 → 区分各状态的证明范围 → 用 Git 的 HEAD/index/working tree 解释文件版本 → 说明派生检索视图如何过期 → 复走安全恢复 → 设计按版本交付的 artifact → 检查权限和保留边界。
- **新锚点：** task-42、patch-41、r17/r18/r19 均为明确标注的教学构造，不是生产运行或 benchmark 轨迹。

## 章级扫描与结构决定

2026-11-19 通过 web search/open 定向查阅 Git 官方手册、Pro Git、MIT Missing Semester 2026 Git 课程、OpenAI Agents API 预热环境文档、OpenAI Codex Cloud 文档，以及 StagedWorkspace、Workspace-Bench、SWE-Touch、Tracking State Footprints 和 OpenHands 原文。详细来源版本、实际阅读范围和证据边界见[研究记录](../../research/persistent-workspaces-and-agent-state-evidence-2026-11-19.md)。

外部主题对照：

| 候选主题 | 来源支持与适用范围 | 处理决定 |
| --- | --- | --- |
| HEAD、index、working tree 与提交快照 | Git 官方手册定义其文件版本关系；MIT 课程提供教学组织样本 | 第 21.3 节用最小表格解释；编辑操作细节仍在第 5 章 |
| session 与 workspace 生命周期 | OpenAI Agents API 和 Codex Cloud 文档展示两种具体合同 | 第 21.2 节用简短产品例子说明实现并不统一，不外推到其他产品 |
| parsed view、源文件、diff 与 artifact 版本关系 | StagedWorkspace 预印本提出 source-hash 绑定；消融范围固定于特定 harness/benchmark | 第 21.4 节讲清派生视图的新鲜度；不把论文设计说成通用标准或普遍收益 |
| 用户在 Agent 运行中修改文件 | SWE-Touch 研究使用受控任务冲突编辑并设 Co-Edit 对照 | 第 21.5 节用来说明应检查变更内容，不以研究数字推断所有协作场景 |
| workspace 级跨文件检索评估 | Workspace-Bench 的任务、工作区和评分合同 | 评测设计由第 19 章承接；第 21 章不复制 benchmark 统计 |
| 读写 footprint 与并发事务 | Tracking State Footprints 提出事务化 Agent 的研究方向，尚非实现保证 | 第 17 章负责 checkpoint、effect、重试与 fencing；本章只解释 workspace revision 边界 |
| Agent runtime 的 interface/environment/interaction | OpenHands ICLR 2025 系统实例 | 可帮助理解 runtime 层次，但不声称其提供通用持久化合同；正文未展开 |
| Provider thinking-state 协议细节 | 原稿的 Claude Fable 5.1 状态合同属于模型/API 专题 | 从本章删除，列为第 24 章复核候选；须核验第 24 章职责与官方资料后再决定是否承接 |
| Qwen3.6 历史 thinking 序列化 | 原稿的模型卡/template/API 细节属于模型专题；第二十一册第 83 章已有 §83.15 | 从本章删除，交由第 83 章主讲；该章的 Qwen source-note 失效链接仍待第二十一册审阅时处理 |

## 原稿覆盖去向

- 原 §§21.1—2、21.11.1、21.12—14、21.27 对模型上下文/记忆、workspace、checkpoint 和外部状态的区别多次复述；合并到新 §21.1—2，并把 checkpoint 和 context folding 细节分别链接第 17、18 章。
- 原 §§21.3、21.8、21.16、21.18—19、21.21、21.23、21.26 关于 workspace 身份、版本、条件写入、锁、合并和恢复反复展开；收束为 Git 版本模型与 task-42 r17→r18→r19 案例。文件 patch/diff 机制回链第 5 章；checkpoint、租约、fencing 和外部 effect 回链第 17 章。
- 原 §§21.4、21.25 的 artifact 交付字段合并为“交接信息”表。字段作为教学示例，不再宣称固定 manifest 是普遍必需 schema。
- 原 §§21.6—7、21.15、21.20、21.24 的权限、租户、删除和保留内容合并到新 §21.7，并收窄为边界问题；权限和沙箱方案交给第 8 章，删除语义依具体产品及数据类别核对。
- 原 §§21.9、21.17、21.22 中的 evaluation 清单和自拟 R_workspace 公式移除；结果归因与基准评测由第 10、19 章负责。本章仅要求验证证据绑定其目标 revision。
- 原 §21.10 的误区表和 §21.11 的面试/练习重复条目改由连续讲解、单一恢复路径、练习 35EP 与面试题 40AE 承接。
- 原 §21.28 的总结并入新 §21.8；未保留强制特定 manifest、综合指标或发布 gate 的表述。

## 修改与同步

- 重写[第二十一章正文](../../book-20-agent-harness-runtime/chapters/21-persistent-workspace与状态边界.md)，从 30 个重复/混合主题重组为 8 节主线。
- 更新[第二十册目录](../../book-20-agent-harness-runtime/目录.md)和[架构审计](architecture-audit.md)：第 21 章完成，下一章为第 22 章。
- 新增[练习 35EP](../../EXERCISES.md)与[面试题 40AE](../../INTERVIEW_BANK.md)，避免复制旧稿中的多项重复问答。

## 公式、代码、图表与验收

- 新正文没有公式、可执行代码或 Mermaid；无需代码运行或 MathJax/图形渲染检查。
- 使用状态归属表、Git 版本表和交付信息表呈现并列关系；时间变化由同一 r17/r18/r19 例子解释。
- 本次静态验收需确认 H2 21.1—21.8 连续、正文/研究/目录/进度/练习/题库本地链接均可解析、无旧的 Fable/Qwen provider 专题和自拟 R_workspace 公式残留，并运行 git diff --check。
- 未做目标读者试读或外链长期可达性核查；没有调用模型、API、SDK、benchmark、自动化测试或 PDF 构建。

## 交接

下一步从[第二十二章 OpenAI-Compatible API 边界](../../book-20-agent-harness-runtime/chapters/22-openai-compatible-api边界.md)开始，检查兼容语义的章级职责和官方文档版本。随后第 23—24 章分别复核 Gemini 与 Claude 的 provider-specific 状态合同。第二十一册第 83 章的 Qwen3.6 source-note 缺失链接需在该册逐章审阅时解决。全套书精修 goal 保持 active；不更新 PDF 或 plan.md。
