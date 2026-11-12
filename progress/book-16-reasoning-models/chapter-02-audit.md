# 第二章《Chain-of-Thought》精修记录

审阅日期：2026-11-12  
状态：章节正文已精修；章级扫描、代码、公式、链接、视觉和简介摘要已同步。按用户要求不更新 PDF。

## 范围与章级扫描

- 通读本章 §§0.1–21、合同案例、路由与安全部分、练习、资料索引和独立 Python 示例；对照第一章总览，以及第三至第五章的采样、验证器和过程监督交接。
- 定向核对范围包括 CoT 与 zero-shot prompting、scratchpad、轨迹忠实性、训练数据和错误传播、benchmark 改写/污染、API thinking 返回合同、工具轨迹、安全评估与推理成本。来源为原始论文、ACL Anthology 同行评审论文和 provider 官方文档；结论按其模型、任务、数据、接口和访问日期限定。

| 候选主题 | 来源与正文处置 |
|---|---|
| Few-shot/zero-shot CoT 与 scratchpad | 保留为本章核心机制；在历史说明附近引用原始研究，限定到论文测试的模型与任务。程序化计算链接到 Program of Thoughts，外部程序执行不外推为普遍优胜路线。 |
| 轨迹忠实性、答案合理化和干预评估 | 保留答案正确率、轨迹一致率、数字替换、步骤删除和工具检查；引用 Faithful CoT、Unfaithful Explanations 与 FACT-E。解释行为证据不宣称读取完整内部状态；更广的 reasoning 安全评估交第十、十一章。 |
| 改写 benchmark 与词面去重盲点 | 在示例污染小节加入改写/翻译近重复的边界，引用原始研究；语义相似仅作筛查信号，不等同污染证明，通用污染评估交第十章。 |
| thinking / reasoning API 的可见内容和跨轮状态 | 只写当前 OpenAI 与 Anthropic 官方文档支持的接口合同，并记录 2026-11-12 访问日期；不推断内部 raw trace。预算口径、thinking level 和保留协议分别链接到第十三至十五章。 |
| CoT 监督、数据质量和错误教师轨迹 | 保留样本结构、有效步骤、程序/人工抽查、答案与步骤损失差别及蒸馏边界；步骤级监督主讲交第五章，候选验证细节交第四章。 |
| 工具调用与可审计任务轨迹 | 保留状态—动作—观察案例和合同付款计算，明确来源、执行结果、权限分别检查；完整 Agent、Runtime 与工具协议交第十七、二十、二十二册。 |
| 搜索、self-consistency、verifier、process supervision 与 Tree-of-Thought | 本章只作为 CoT 相邻机制与交接，不重复主讲；分别链接第三至第六章。预算路由保留在任务选择层，test-time scaling 和动态预算策略交第七、十三、十八章。 |
| 轨迹暴露与安全检查 | 引用 Hidden Thoughts Are Not Secret 与 TRACE 的限定性结果；不外推闭源服务风险率或生产检测器效果，治理交第八册。 |

章级核查复用的来源证据已记在[test-time compute 与评估研究记录](../../research/test-time-compute-and-evaluation-2026.md)和[泛化与 benchmark contamination 研究记录](../../research/generalization-and-benchmark-contamination-evidence-2026-11-09.md)。本轮没有把审计状态或写作修改过程写入 `research/`。

## 正文修改

- 在章节入口补入第一、三、四、五章链接，并在结尾将第三章入口做成链接，明确 CoT 与候选采样、验证及过程监督的分工。
- 在 few-shot/zero-shot、scratchpad、Program of Thoughts、改写样本污染、轨迹干预、API 返回内容、CoT 忠实性、轨迹安全等具体论断附近补充可追溯的原始论文或官方文档链接。
- 新增 CoT、scratchpad、外部解释的对照表，区分目标、受众、优先核对对象和不能推出的结论。
- 对忠实性准确率与一致率公式定义有效样本数及轨迹解析函数；为干预式和质量—预算公式说明变量，保留分母和可比条件边界。
- 修正 §16 示例输入校验：对 route 类型先做字符串检查，避免列表等不可哈希输入触发意外 `TypeError`；拒绝非字符串 accuracy key。将 `cost_per_*` 更名为 `tokens_per_*`，并把简单任务指标明确为“直答成功条件下的 CoT token 开销比例”。
- 更新示例预期输出和相邻解读；将五条旧式 URL 索引改为可读标题链接，修正 arXiv:2211.12588 的题名为 Program of Thoughts Prompting，替换过时的 OpenAI/Anthropic 文档入口，加入 FACT-E、Hidden Thoughts Are Not Secret、TRACE 与 benchmark contamination 来源，并删除作者联网流程日志。
- 更新第十六册简介中的第二章摘要；目录和章节文件结构无需调整。

## 验收字段

- **length_status：** 本章作为一个连贯核心单元，剔除代码、公式围栏、表格、标题、练习/维护索引和空白后，解释性中文实计 9,087 汉字，高于 1,800 字下限。核心小节互为论证环节，不作为独立辅助教程重复计数。
- **example_status：** 水箱例子从进水 3 L/min、漏水 1 L/min 得到净流入 2 L/min，再以 20 L ÷ 2 L/min 得到 10 分钟；合同例子将 100,000 元依次乘以 0.95 和 1.06，得到 100,700 元，并绑定三处文档证据和独立计算器检查；§16 的六个合成样本比较直答、CoT 与分流结果。合成样本不代表真实模型或生产测量。
- **evidence：** 论文和 API 文档直接链接到论断附近；重点使用 arXiv:2201.11903v6、2205.11916v4、2301.13379v3、2305.04388v2、2211.12588v4、2311.04850v2、ACL Anthology 2026 三篇轨迹/安全研究，以及 OpenAI/Anthropic 访问日为 2026-11-12 的官方文档。各来源的模型、任务和接口范围在正文明确收窄。
- **formula_status：** 复核 14 个 display-math 公式与变量解释，覆盖条件概率分解、自回归生成、语言模型/掩码损失、答案准确率和轨迹一致率、反事实干预、parser 精确率/召回率、工具状态转移、路由效用、CoT 回归、质量—预算差分和重复比例。已说明有效样本数、解析定义、统一比较条件、非负权重、非空损失/评估分母及非零 token 分母；公式和花括号配对。
- **code_status：** §16 独立标准库示例于 2026-11-12 在 Python 3.12.9 执行。直接从 Markdown 提取后运行，stdout 与正文预期输出逐行完全一致。边界检查通过：列表 route 在 validator/router 中按合同拒绝；bool token、空步骤、非 bool 步骤、非 bool 可见性字段和非字符串准确率 key 被拒绝。示例只验证六条合成记录的路由和指标，不调用模型、API 或工具服务。
- **link_status：** 全章 42 个 Markdown 链接已扫描；本地相对链接全部存在。更新后的论文、ACL Anthology 和官方文档入口均为直达链接。未进行所有外链的生产环境可用性 crawl。
- **visual_status：** 新增三类中间对象对照表；现有水箱计算、干预公式、合同轨迹和路由输出可读清主要机制与数据流，不需额外 Mermaid。正文无 Mermaid 渲染依赖。
- **sync_status：** 已更新 `简介.md` 中第二章短摘要和第三章下一步交叉链接；目录章目和编号保持一致。未更新 PDF、`plan.md`、题库或其他全局索引。
- **open_risks：** 论文结果限于来源所测模型、任务和协议；CoT 的内部因果忠实性无法仅由文本一致性证明。Provider thinking/API 合同会随模型、接口与版本变化，后续第十三至十五章仍须按当时文档重新核验。代码仅为合成教学账本。

## 下一步

进入第三章《Self-Consistency 与采样》，检查从单条 CoT 到多候选生成、答案归一化与选择器的主线，以及候选覆盖率、投票、成本和 verifier 的职责边界。
