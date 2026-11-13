# 第十四章审计：Thinking Levels

审阅日期：2026-11-13
整册连续复读：2026-11-13

## 范围与章级扫描

- **范围：** 完整精修《第十四章 Thinking Levels》，围绕离散档位的产品合同、跨 provider 适配、质量成本评估、资源容量和模型 API 案例。对应正文位于[第十四章](../../book-16-reasoning-models/chapters/14-thinking-levels.md)。
- **章级扫描：** 2026-11-13 定向阅读 Google Gemini Thinking、3.5 Flash-Lite model page、Thought signatures 和 Interactions API；xAI Reasoning、Grok 4.7 和 Multi Agent 文档；Qwen3-Omni 技术报告与官方 GitHub。研究问题包括：相同档位词在各 model ID/API surface 下具体控制什么；输出硬上限、usage 与思考状态如何呈现；Grok 4.20 的 effort 是否控制普通推理深度；Qwen3-Omni 的变体、帧率和首包延迟能否当作 thinking level。来源范围、日期和证据边界记录于[推理时计算与评估研究记录](../../research/test-time-compute-and-evaluation-2026.md)第 14 节。
- **候选主题去向：**
  - Gemini 3.8 Flash 与 3.5 Flash-Lite 的默认值和支持档位：纳入 §14.8.1，严格绑定 model ID；Interactions 的 max-output cutoff、usage 与 signature 回放同节说明，不外推到其他 API surface。
  - Grok 4.7 reasoning effort 与 encrypted state：纳入 §14.8.2，continuity state 回链第十五章。
  - Grok 4.20 multi-agent 的 4/16 agent mapping：作为“effort 不总是推理深度”的对照留在 §14.8.2；协作/汇总的完整机制交第十六章。文档标示 beta，记录为可能变化。
  - Qwen3-Omni 的 Instruct/Thinking/Captioner 变体、Thinker/Talker、12.5 Hz codec 与理论首包延迟：作为多模态计量反例纳入 §14.8.3；未扩写成模型内部推理档位，架构主讲留给已有多模态章节和模型专题。
  - 跨轮状态、在线预算路由和多 Agent 协作不在本章重复完整主讲，分别链接第十五、十八、十六章。

## 架构与正文决定

- 原 §14.1—14.25 多次重复“档位不是统一能力刻度”“能力矩阵”“质量成本曲线”“版本回归”和“用户语义”；重新组织成 §14.1—14.7 的单一教学链：接口控制面 → 资源向量 → 能力矩阵/有损适配 → 固定任务族评估 → 活跃资源估算 → 独立权限 → 版本回归。
- 把代码示例放到 capability matrix 之后；将原先提前出现的小结/练习移至章末。Google、xAI、Qwen 案例按 provider 分组并改为连续编号，删除原 14.28、14.26、14.27、无编号 Grok 4.7、14.29 的乱序。
- 第十四章仍承担 provider/API 离散控制接口比较；第十三章提供请求资源和成本账本，第十八章提供在线策略，第十五章提供 thinking state 续接，第十六章提供多 Agent 并行协作。简介和目录中已有的第十四章摘要、阅读路径与本次修订后职责一致，无需修改。

## 篇幅与数值例子

- 全章为一个连续核心讲解单元。按规范口径去掉标题、表格、公式/代码块和围栏后，解释性正文实计 **3,454 个汉字**，高于核心单元 1,800 字下限。
- §14.2 用合成 product config 计算两个后端各自的 medium：后端 A 的目标推理 token 2,048、工具 1 次、verifier 1 次、deadline 4 秒；后端 B 分别为 1,024、0、0、2 秒。此例只说明资源向量不可按档位名横向等同，不是供应商数据。
- §14.4 的合成成功率/成本例：质量增量 0.81−0.70=0.11；成本增量 5.9−1.4=4.5；λ=0.02 时边际效用 0.11−0.02×4.5=0.02，λ=0.03 时为 −0.025。任务表全部标记为教学值。
- §14.5 的合成容量例：N=100、q=0.25、low/high 单请求分别 1/3 GiB，拆算得到 75 GiB + 75 GiB = 150 GiB。连续复读将 N 约束为正整数，并单列 N=0 时总占用为 0、无需定义 q；文中明确该式是均值估算，不能推导尾延迟。
- Qwen 示例按报告参数算出 12.5 Hz 对应 80 ms codec 帧；理论首包延迟绑定冷启动、并发和实现配置，不作生产 p99。

## 代码、公式与链接

- §14.3.1 Python 教学适配器：本轮精修后从 Markdown 提取唯一 python fenced block，使用 Python 3.12.9 标准库 compile/exec；无第三方依赖。实际输出与正文逐字一致：
  - native adapted unsupported
  - 6144 total-output-cap
  - read_only read_only
- 边界覆盖 8 项非法输入：level 为 list、backend 为 dict、tool permission 为 list、approximation 为 list 或整数、未知 level/backend/permission；均抛出 ValueError。另验证 native、output-only adapted、plain unsupported 三条路径；output-only 只设置 max_output_tokens 总上限，不冒充 reasoning budget，工具权限保持独立。
- 连续复读只修改公式前的定义域说明；可执行语句、依赖、环境、输入输出合同和预期结果均未变化，沿用 2026-11-13 Python 3.12.9 验证，不重新运行。
- 4 个 display-math 块均有变量解释和可复算示例：资源向量映射、质量/成本边际效用、边际效用具体代值、活跃容量均值。连续复读复核了公式的变量域与零请求边界；无 operatorname，代码/文本/数学围栏标记共 12 个且配对完整。
- 13 个 Markdown 链接中 3 个为本地章节链接，分别指向第十三、十五、十八章且目标存在；官方外链已通过官方页面阅读核对。git diff --check 通过。
- 视觉机会复核：capability matrix 与合成评估表能表达主要差异；公式后附具体算例。无需 Mermaid 流程图，本章不新增图形；MathJax 目标页面实际渲染尚未执行。

## 配套同步、遗留风险与门禁

- 第十六册简介、目录已具备第十四章摘要和专题阅读次序，内容仍准确；无需更改。题库、练习、术语表、论文入口和知识图谱已覆盖 thinking level、Grok agent-count、Gemini 输出硬上限和 Qwen Thinker–Talker；未复制粘贴重复材料。
- provider 文档、默认值、API surface 和 beta 状态可能变化；后续更新时按准确 model ID 和接口复核。未执行真实 API 调用或本地 Qwen 延迟实验。
- 章节正文门禁通过：章级扫描、主题去向、核心篇幅、公式例子、代码、来源、链接和配套审查均已完成。GitHub MathJax 实际渲染留在目标页面验收阶段。
- 第十六册第 1—14 章单篇精修已完成，整册连续复读现覆盖第 1—14 章及第 17 章；下一入口为第十五章《Preserve Thinking 与推理状态》。第 15—16、18—20 章及整册最终门禁待处理。PDF 与 plan.md 未更新。
