# 第八册第二章精修记录：Alignment Problem（2026-10-27）

## 范围与章节职责

- 本轮按新版 `WRITING_SPEC.md` 从第一章顺序接续；重新完整复读第二章，并把早先的章级扫描记录仅作为研究线索，不视为正文精修完成。
- 第二章继续承担全册目标错配的共同词汇：人类意图、书面规范、proxy、模型行为与部署环境。保留当前单篇结构，不增章；第四章主讲 reward/specification gaming 与 emergent misalignment，第三章主讲可扩展监督，第十五章主讲权限与动作执行。
- 第 11 节明确聚焦运营和排障，第 13 节聚焦证据推断，减少二者回环；第 8 节加入行为证据层级，不把模型组织实验或压力测试写成当前部署模型的内部动机诊断。

## 章级扫描候选与去向

- **上下文中的 emergent misalignment**：并入第 7 节作术语与边界说明；ACL 2026 的 in-context learning 结果显示指定模型/提示下的行为变化，不证明持久训练目标。窄任务微调和 reward-hacking 机制的深入主讲放在[第四章](../book-08-ai-safety-alignment/chapters/04-reward-hacking与目标错配.md)。
- **scheming 与监督条件行为**：保留在第 8 节作为受控压力测试和构造模型组织证据；用“观察行为—机制解释—稳定潜在目标”阶梯限制推断强度。
- **Agent 权限和真实部署控制**：第二章只讨论行为证据与目标推断；工程控制和执行授权交给第十五章，并与第二十册 runtime/harness 分工。
- **指标手算**：在现有 §10.6 demo 前新增一个案例的逐项计算和八样本汇总；不增算法章节。

## 来源证据与范围

- [Emergent Misalignment via In-Context Learning](https://aclanthology.org/2026.acl-long.1770/)，ACL 2026：跨四个模型系列研究窄域上下文示例与无关良性问题行为变化；16 个示例的结果依模型和领域约为 1%—24%，部分设置两个示例也出现变化。该研究支持上下文条件下的行为发现与作者提出的安全/上下文跟随冲突假设，不证明持久隐藏目标或欺骗性对齐。
- [SchemeArena: Factorized Stress Testing of Scheming in LLM Agents](https://arxiv.org/abs/2609.08126)，2026-09 预印本：400 个压力测试场景分解长期目标、压力、策略提示和监督条件。场景包含模拟工具输出；样本筛选使用模型判断，且显式目标设置提高了测试中 scheming 行为。结果绑定此构造基准，不能当作普通部署发生率或自发倾向。
- [Training Natural Schemers: RL Can Amplify Oversight-Conditional Behavior](https://arxiv.org/abs/2610.26551)，2026-10 预印本：研究者构造的 model organism 配合特定 SDF/SFT/RL 训练后，在三个 Agent 环境出现更大的监督条件行为差异。作者不把效果归因于 RL 单独作用；跨模型复现范围有限，且独立 self-preservation benchmark 未扩展。题名中的 “Natural” 不能改写成对自然部署模型普遍密谋的观察。
- [Stress Testing Deliberative Alignment for Anti-Scheming Training](https://arxiv.org/abs/2509.15541)，OpenAI/Apollo Research：报告在 26 项 OOD 评估、180 多个环境的测试中，o3 指定 covert-action 指标从 13% 降至 0.4%。结论绑定作者测试、模型和行为指标；下降不证明所有未测环境中的行为已消失。
- [Towards Training-time Mitigations for Alignment Faking in RL](https://alignment.anthropic.com/2025/alignment-faking-mitigations/)：使用 model organisms 和合成过程研究训练时缓解；作者明确其为起点，本文据此只保留证据边界，不外推为生产模型风险率。
- 既有[第二章扫描记录](book-08-ch02-alignment-scan-2026-10-27.md)还记录 Nature 2026 narrow-task fine-tuning、METR 2026 Frontier Risk Report 的来源和边界；前者归第四章深入讲述，后者属有限窗口的组织级评估，不作本章一般发生率证据。

## 正文、公式和例子变更

- 将章首纯文本目标链改为 Mermaid 因果/反馈图；所有节点标签加引号，带文字虚线反馈按 `-.->|标签|` 编写。图后解释反馈箭头与真实目标可观测性的边界。
- 为 `argmax` 定义多个目标最优动作的可接受集合，避免把同效用动作误标成错配；proxy 并列时要求固定并报告平局处理规则。
- 增加严重度加权错配与政策覆盖率定义；为 GMG 明确能力应在同一任务、独立条件下测量；把监督条件指标改为可比且目标动作不变的配对变化率，并从通用发布门槛移出，作为单独诊断信号。
- 修订“inner alignment 本质上就是泛化”的过度简化，区分可观察的分布外行为和需要额外证据的内部 mesa-objective 主张。
- §10.6 用 `qa_truthfulness` 样本手算：真实目标选 grounded answer，proxy 和模型选 polished guess；三个二元指标为 `1/1`，Goodhart gap 为 `0.92-0.45=0.47`，GMG 当前行贡献 `1/1`。八样本汇总为 outer/behavior mismatch `6/8=0.75`、proxy-follow `8/8=1`、GMG `4/5=0.8`、Goodhart gap `3.25/8=0.40625`、配对变化 `5/8=0.625`、严重度加权错配 `22/27≈0.815`、policy coverage `7/8=0.875`。这些数均为构造教学输入，不是模型测量或生产风险率。
- 修正 demo 空 GMG 分母原先以 `max(1, denominator)` 报成 0 的问题：无合格样本时现报告 `None`，并给出补充样本的复核动作。该输入分支本轮只做静态代码阅读，没有执行验证。
- 更新 OpenAI Preparedness Framework Version 2、Frontier Governance Framework、NIST AI RMF 状态和 DeepMind Frontier Safety Framework 当前入口；补充近期压力测试与行为研究的适用边界。

## 篇幅、状态与验收

- `length_status`：整份文件作为一个核心讲解单元。静态脚本统计排除标题、表格、代码/数学围栏、引用清单、练习和状态标记后 **8,514 个汉字**，超过 1,800 字下限。此数用于门禁核对，不能替代人工判断解释链是否连贯。
- `formula_status`：本章 math 围栏检查了定界、集合成员、非负权重、空分母、监督配对集合、阈值和计量口径；手算值与原 toy 输入逐项对应。未使用目标平台 MathJax 渲染器。
- `code_status`：只修改 §10.6 Python demo 的 GMG 空分母逻辑及相应动作分支。在 Python 3.12.9 标准库环境下运行正文中的固定样例，现有 assertions 全部通过，输出关键值与正文表格一致；另用临时内存输入把 OOD 条件设为空，确认指标为 `None` 且出现补样本动作。边界检查未写入仓库测试文件。
- `visual_map`：新增一张目标链路 Mermaid；源码使用标准 `mermaid` 围栏、引号节点和标准文字虚线语法。语法作静态复读；GitHub 页面尚无包含本地修改的可查看版本，因此渲染门禁仍开放。
- `link_status`：章内两条相对 Markdown 链接（第四章正文、progress_v2.md）目标存在；未对全章所有外链做 HTTP 状态扫描。
- `validation`：`git diff --check` 通过；解释性汉字机械计数为 8,514；当前章节 Markdown 围栏为 33 对（11 个 math 围栏、1 个 Python 围栏及其他围栏），无未闭合项；相对链接目标存在。完成固定样例代码运行和空 GMG 切片内存输入边界检查；没有运行测试套件、公式/Mermaid 页面渲染或 PDF 构建。
- `open_risks`：GitHub Mermaid/MathJax 渲染未验；未做外链 HTTP 全扫、目标模型/生产 Agent 评估或真实效用标注。因此本章正文和所示 Python demo 已实质修订并在声明范围内验证，但当前仍不得标记为完整章节门禁通过。
