# 第八册第三章精修记录：Scalable Oversight（2026-10-27）

## 范围、职责与结构决定

- 按新版 `WRITING_SPEC.md` 从第一章顺序复读到第三章；完整复读第三章后，把原扫描记录作为发现线索，不视为正文精修完成。
- 本章仍是一篇核心教材，中心问题是：当人工直接判断跟不上任务复杂度时，如何把监督扩展成可校准、可复核的信号。保留现有方法谱系和章节边界，不新增 Markdown 篇目；Iterated Amplification / Recursive Reward Modeling 讲分解与递归，Debate 讲对抗性暴露错误，Constitutional AI / RLAIF 讲原则驱动反馈，人机比较与自评讲校准，项目案例讲多信号闭环。
- 把章首的纯文本流程改成监督闭环 Mermaid；补充互补标签、Human Debate 与模型 Debate 的差异、judge 共识限制和 judge 校准边界。§4.6 指标与 §11.5 demo 通过同一十条样例连起来。

## 章级扫描候选及去向

- **互补标签监督**：并入人类能力瓶颈和监督信号谱系。只支持论文所研究的多选估计问题；排除判断不是完整 gold label，且 hard-negative 选择会破坏原估计假设。
- **人类 Debate 与 AI Debate 的差异**：并入 Debate 的证据边界；报告两者在特定任务上的不同结果，不推广到其他模型或评测协议。
- **judge 校准与 judge 共识**：并入人类/AI feedback 的实战取舍；提醒多个 judge 一致可能源于相关错误，不等于专家校准。
- **Nested scalable oversight**：作为递归监督量化研究的边界说明，留在本章方法关系中；数值只绑定论文监督游戏和 Elo 配置。
- **章节分工**：与第七册 LLM-as-a-Judge 评测协议交叉但不重复主讲；Agent 执行权限交第十五章与第二十册 runtime/harness。

## 来源证据与限制

- [Towards Scalable Oversight via Partitioned Human Supervision](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b6f03ce953cf7f07dafa20891d2539e3-Abstract-Conference.html)，ICLR 2026：研究互补标签与多选任务监督估计。论文的错误选项抽样和同分布假设限制了将估计器迁移到自由文本、Agent trace 或有偏 hard negative 的能力。
- [Humans Correct Their Judgments Through Debate, but Weaker AI Models Do Not](https://arxiv.org/abs/2610.28539)，NeurIPS 2026 接收论文的扩展预印本：5 个人类实验共 507 名参与者、40 多组 LLM 实验；指定任务上人类裁判能从 debate 纠正部分判断，弱模型裁判增益不稳定。结论绑定论文任务、模型、提示和样本。
- [Limits to scalable evaluation at the frontier](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4264ee4376776907c0b87ed70b959585-Abstract-Conference.html)，ICLR 2025：分析评估去偏可减少 gold label 需求的理论上限与实践结果；本文只据此说明 gold set 不能任意缩减。
- [The Geometry of LLM-as-Judge](https://arxiv.org/abs/2606.03043)，2026 预印本：在四个 Indic 社群数据集、八种语言和 41 个 judge 上报告 judge 间一致性高于 judge—human 一致；在事实可验证 rubric 下差距缩小。结果受语料、语言、rubric 和模型限制。
- [Scaling Laws For Scalable Oversight](https://proceedings.neurips.cc/paper_files/paper/2025/file/8c3caae2f725c8e2a55ecd600563d172-Paper-Conference.pdf)，NeurIPS 2025：在 Mafia、Debate、Backdoor Code、Wargames 等监督游戏及 Elo 配置下分析 nested oversight；不是生产监督成功概率。
- 以上新来源及此前方法来源的引用入口已同步到 `PAPERS.md`；完整支持范围和候选分析亦见[原第三章扫描记录](book-08-ch03-scalable-oversight-scan-2026-10-27.md)。

## 正文、数值例与代码变更

- 增加人类互补标签的边界例子；解释 Iterated Amplification 不等于多模型投票，以及 recursive reward model 层级错误可能相关。
- 解释 Constitutional AI 是具体原则驱动训练路线，RLAIF 是更广的 AI feedback 方法家族；增加“单项研究被过度概括”的 critique/revision 示例，并扩写混合监督、judge 校准和自我置信度路由。
- 十条 toy oversight case 中，`medical_summary` 样本的 gold 为 false，AI 仍给 true，verifier/debate 缺席，自动组合标签错误为 `1/1`；人审置信度 `0.70<0.75` 未进入直接覆盖，高风险未审计，过程分 `2/4=0.50`，claim 支持 `2/4=0.50`，单位样本教学成本节省 `1-14/80=0.825`。高风险漏审仍阻断扩展。
- 十条样本汇总：人工直接覆盖 `2/10=0.20`；覆盖样本人工准确 `2/2=1`；AI feedback 错误 `3/10=0.30`；verifier 覆盖 `7/10=0.70`；人审前自动组合准确 `9/10=0.90`；过程步骤准确 `28/39≈0.718`；claim 支持 `17/24≈0.708`；高风险审计覆盖 `4/5=0.80`；成本节省 `1-132/503≈0.738`；严重度加权错误 `5/33≈0.152`。这些全是合成标签，不是模型测量或生产发生率。
- 修正空分母：加入 `safe_ratio`，空集输出 `None`；阈值判断前显式区分未测量与通过/失败。将 toy `oversight_accuracy` 说明为人审前自动组合准确率，平票沿用 AI 标签仅为可复现教学规则，不作为生产建议。

## 篇幅、状态与验收

- `length_status`：整篇作为一个核心讲解单元；排除标题、表格、围栏内容、引用清单、练习和状态标记后，机械统计 **9,037 个汉字**，高于 1,800 字门槛。机械计数不替代人工复读解释链。
- `formula_status`：11 个 `math` 围栏静态检查变量与分母；手算一个高风险样本和全样本主要指标，与运行结果一致。未作目标平台公式渲染。
- `code_status`：Python 3.12.9、标准库下运行正文固定样例，原有 assertions 通过，输出与文中汇总吻合；另用临时内存样例同时触发空 direct-covered、process steps、claims、高风险集、人工成本和 severity 分母，确认其指标为 `None`、返回 N/A 动作，并进入 `hold_for_missing_measurement`。没有新增仓库测试文件。
- `visual_map`：章首图使用标准 `mermaid` 围栏、引号节点和 `-.->|标签|`；源码静态复读通过。GitHub 页面尚无包含本地改动的可查看版本，图和数学渲染门禁仍开放。
- `link_status`：唯一相对链接指向 `progress_v2.md`，目标存在；全章外链 HTTP 状态未全扫。
- `validation`：`git diff --check` 通过；24 对 Markdown 围栏（11 math、1 Python 和其余示意/输出围栏）配对；解释性汉字机械统计 9,037；固定运行及空分母边界输入通过。
- `open_risks`：目标平台 Mermaid/MathJax 渲染、外链 HTTP 全扫及整册连续复读仍未做；合成投票与成本不能外推到真实 judge、专家工作流、用户损失或生产部署。因此正文和 demo 已实质修订且本地验证，但章节完整门禁仍开放。
