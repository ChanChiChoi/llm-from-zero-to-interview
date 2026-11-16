# 第十七册第七章精修审计

审阅日期：2026-11-16

状态：首轮正文精修完成；章节代码未运行，目标平台渲染和跨册遗留链接仍待最终门禁。

## 章节契约与读者路径

- **读者前置：** 已了解前六章中的 Agent、工具接口、反馈循环、任务拆分、Memory 与证据检索；无需预先熟悉特定 Code Agent 产品或框架。
- **本章对书册目标的贡献：** 让读者能把仓库任务变成可验证状态变化，保护工作区与用户已有修改，使用测试和其他工程反馈修复问题，并交付可审 diff、验证证据和未解决风险。
- **中心问题：** Code Agent 怎样理解仓库、产生最小必要修改、基于真实执行反馈迭代，并证明任务已完成？
- **读者路径：** 区分代码补全与任务执行 → 解释仓库定位、上下文、编辑、命令、测试和调试机制 → 用轨迹 demo 与指标检查结果 → 沿 issue-to-patch 流程保护工作区和权限 → 阅读版本化接口与长程 kernel 案例作为选读。

## 章级扫描与候选去向

- **检索日期与范围：** 2026-11-16；复核仓库/终端 Agent 评测、当前模型/API 合同、长时程代码优化案例。来源类型包括 benchmark 官方 release/公告、供应商官方模型和开发者文档、模型发布方实验说明。没有将 benchmark 结果外推成生产质量或裸模型能力。
- **Terminal-Bench：** v4.0.0 release 明确记录任务更新；官方将其描述为带版本的持续 benchmark。Terminal-Bench Challenges 描述长时程、高 token 消耗的单任务。前者与 Challenges 均纳入 §7.16 的评测口径提醒，要求固定 release、harness、环境和 verifier；没有添加模型排行。证据见[Code Agent 评测来源记录](../../research/code-agent-benchmarks-2026-11.md)。
- **模型与 API 合同：** 当前 OpenAI 文档把 Sol 列为 `gpt-6.1-sol`；GPT-6.1 Sol 工具调用使用 Responses API，GPT-6 Luna 在 Chat Completions 中仅于 `reasoning_effort=none` 支持 function calling。Opus 5.5 的 thinking/tool-choice/compaction 和 Grok 4.7 的 encrypted reasoning 是不同的状态合同；Gemini Interactions 以 steps 与 `previous_interaction_id` 表示服务端会话。各项仍由宿主授权与执行。当前 OpenAI 快照已追加至[GPT-6 API 合同来源记录](../../research/openai-gpt-6-astra-api-contract-2026.md)；Anthropic、Google、xAI 的来源继续见对应官方链接及现有研究记录。
- **Qwen3.7-Max kernel 案例：** 保留为 2026-05 的发布日期案例，而非当前模型推荐。只陈述发布方给出的 M890 任务、初始化条件、工具轨迹、baseline 和统计量；另行区分 KernelBench L3 的 H100 指标。来源、脚注口径与限制见[Qwen3.7-Max 来源记录](../../research/model-update-2026-09/qwen3.7-max-source-notes.md)。
- **跨范围内容：** Gemini 视频 processing step 的细节已由第二十册第 23 章专讲；GUI/mobile action 属第八章；Qwen-Omni 音频架构和 Qwen-VL 视觉架构分别交由第二十一册第 91、92 章。删除第七章内的重复展开并保留目标章节链接。
- **旧模型快照：** GPT-5.3 Codex 已标为 deprecated；GPT-6 Sol 当前指向 `gpt-6.1-sol`。Claude Sonnet 5 与 5.5 是不同模型版本，未把两者合并。其他 GPT-5.6、Opus 4.6/4.7、Gemini 3.5 Flash-Lite 等片段提供的协议原则已归入 §7.24，不再逐模型复制旧榜单、GUI 或视频细节。

## Coverage map

| 核心知识 | 对读者结果的作用 | 正文位置与处置 |
|---|---|---|
| Agent 任务合同与 Code Agent 边界 | 区分生成片段与仓库状态变化 | §§7.1—7.3 保留并按原教学路径组织 |
| 仓库理解、定位、上下文和最小 patch | 减少误定位、过宽修改与上下文遗漏 | §§7.4—7.8 保留 |
| 命令、测试、debug 与多文件工作 | 从真实 observation 更新假设并验证修改 | §§7.9—7.14 保留 |
| Memory、指标和 toy audit | 维护状态并分层度量任务、执行和风险 | §§7.15—7.18 保留；§7.16 增补 benchmark release 与长程任务口径；§7.17 代码未改 |
| issue-to-patch、工作区、沙箱和练习 | 形成保护用户状态并可交付的闭环 | §§7.19—7.23 保留 |
| provider API 状态合同 | 把通用 trace 原则映射到不同 API surface | 旧版模型小节收整为 §7.24 四种接口案例 |
| 长时程性能优化 | 用 profile/编译/正确性反馈解释复杂迭代 | 旧 Qwen3.7-Max 段整理为 §7.25 一条分阶段轨迹 |
| benchmark、论文与产品文档 | 给读者后续资料入口并说明证据边界 | 收束于 §7.26 |

未发现需要拆分主章节或删除核心知识点的理由。模型接口和 kernel 优化是主线后的选读案例，不改变前 7.1—7.23 的学习结果。

## 结构与正文改动

- 将原 §7.24 references 移至章节末尾，全部章节末尾标题编号整理为连续的 7.24—7.26；消除 §7.30、§7.31、§7.32、复用 §7.24 和未编号标题。
- 在 §7.16 说明评测需固定 benchmark release 与 harness；区分持续更新的 Terminal-Bench v4.0.0 和 Terminal-Bench Challenges 的长时程单任务。
- §7.24 将多组重复的 tool/permission/verifier 责任链压缩成 OpenAI、Anthropic、Google、xAI 的状态合同对照。刷新 GPT-6.1 Sol 命名并按当前接口说明工具调用边界；不把旧版本榜单当作当前模型排行。
- §7.25 保留 Qwen 发布方 M890 kernel 优化案例，将初始性能、逐阶段改动、最终 10.0x 及与 KernelBench L3 的不可比条件串成可复核轨迹。
- 移除或交叉链接 Computer Use、GUI/mobile、视频处理和音频模型细节，避免与第八章、第二十册第 23 章和第二十一册第 91—92 章重复。
- 更新本章资料导言，说明末尾两节是选读案例；简介与目录中的章节职责和主线不变，无需同步修改。
- 第二十册第 23 章原先链接到本章已移除的 `Agentic Video` 小节；已将其改为本章文件入口，并保留该册的媒体读取责任链。本轮发现该章另外仍引用缺失的 Gemini source note；这是第二十册章节审计时需修复的既有问题。

## 代码、公式、例子与静态检查

- **代码：** §7.17 的 Python demo 未改、未运行；仍是合成轨迹审计器。章节代码按用户要求不执行、不测试。
- **例子：** demo 的合成轨迹与已有静态预期表未改；新增 Qwen 案例为外部发布方报告的逐阶段数值，不当成本地复现或通用加速保证。
- **公式：** 静态计数为 14 个 `math` 围栏；本次仅调整章节后半结构与评测文字，未改核心公式。`\operatorname`、`\(` 均未出现。
- **编号与链接：** 26 个二级章节标题从 7.1 到 7.26 连续且无重复；围栏配对；第七章所有本地 Markdown 链接目标存在；未发现旧 7.30+ 编号。第二十册指向已删除小节的旧 anchor 已改。
- **格式检查：** `git diff --check` 通过。未运行代码、测试、benchmark、模型或 PDF 构建；GitHub 页面 MathJax/Markdown 实际呈现与旧外链全量可达性未检查。

## 同步状态与剩余风险

- 本轮改动文件：第七章正文、第二十册第 23 章的一条回链、第十七册架构审计、本章进度记录、全套进度交接、两份新增 research 记录及 GPT-6 research 更新。
- 外部链接在正文中指向官方 benchmark release、研究论文和供应商文档；本次浏览复核了新增的 Terminal-Bench、OpenAI、Anthropic、Google、xAI 与 Qwen 关键来源，但没有批量探测所有既有链接。
- 第二十册第 23 章及 `BOOK_SERIES.md` 仍引用不存在的 `research/model-update-2026-09/gemini-3.8-flash-source-notes.md`；待第二十册精修时补齐来源记录并回收相关链接。
- 第七章首轮精修完成，但因按用户要求没有运行 demo、旧章节代码动态验证与 GitHub 页面公式渲染，不能作为全册或全章最终门禁完成。

下一入口：第十七册第八章《Browser 与 Computer Use Agent》，重点核对 DOM/accessibility tree/截图、执行通道选择、权限确认、不可逆动作、prompt injection 与 UI 评估是否构成完整可复核闭环。
