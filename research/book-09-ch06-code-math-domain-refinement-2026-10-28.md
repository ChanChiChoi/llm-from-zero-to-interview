# 第九册第六章研究与精修记录：Code、Math 与 Domain Data

日期：2026-10-28

## 范围与结构决定

按当前 `WRITING_SPEC.md` 从头复读整章，检查本章职责、公式、手算、代码、图表、近期证据及配套索引。保留单篇结构：共同主线是专项数据的授权、类型化质量验证、风险隔离、采样与评估；代码、数学、领域数据分别展开专属字段和失败模式。第五章已负责通用配比和样本选择方法，本章保留类型约束，并新增回链，不重复讲通用 mixture 优化。

## 定向研究与证据边界

按代码数据的授权/可验证性、数学数据的题目覆盖/解答验证、专业数据的记忆与责任边界检索并核对原始来源：

- NeurIPS 2025 [rStar-Coder](https://proceedings.neurips.cc/paper_files/paper/2025/hash/54e847e1dffc87a8063844b149148557-Abstract-Conference.html) 报告 418K 代码问题、约 580K 长解答轨迹，以及生成测试和互相验证流程。正文将其作为关联问题、解答、测试和验证回执的案例；训练/评估结果只绑定论文所用 Qwen 模型和代码 benchmark。
- NeurIPS 2025 [The Common Pile v0.1](https://proceedings.neurips.cc/paper_files/paper/2025/hash/52acc050138d6f40dad6f12f91a4ce22-Abstract-Datasets_and_Benchmarks_Track.html) 报告 30 个来源、约 8 TB 公有领域或开放许可语料，其中包括代码，并提供 7B 训练案例。它说明许可来源可构成研究语料，不能代替逐项目、逐版本授权判断。
- NeurIPS 2025 [ResearchCodeBench](https://proceedings.neurips.cc/paper_files/paper/2025/hash/cd0d0a873cc3e601c76f46dccc3d4c5f-Abstract-Datasets_and_Benchmarks_Track.html) 将 2024—2025 年机器学习研究方法转成实现评测任务。正文用它说明代码评测可覆盖新方法实现，但 benchmark 本身不是训练集质量或污染清洁证明。
- Findings EMNLP 2025 [FLAMES](https://aclanthology.org/2025.findings-emnlp.1346/) 比较十种数学合成策略，区分题目覆盖和解答可靠性；结果绑定论文生成预算、模型和任务，不能据此把未经核验的答案标成正确监督。
- EMNLP 2025 Industry Track [More Data or Better Data?](https://aclanthology.org/2025.emnlp-industry.43/) 比较统一流水线中的开源数学数据和合成方案。正文仅据此说明结构化数据或较强 teacher 在论文设置下可能胜过单纯扩量，不外推成通用排序。
- EMNLP 2026 [MathDebugger](https://aclanthology.org/2026.emnlp-main.608/) 分别检查答案是否错误、错误类型及求解与验证能力差距；论文在其数据上评估 14 个模型和 3 个过程奖励模型。正文将它用于说明 verifier 需要按错误类别检查，不能证明本地 verifier 已校准。
- 2025 年医学记忆研究预印本 [Memorization in Large Language Models in Medicine](https://arxiv.org/abs/2509.08604) 覆盖医学继续预训练、公开 benchmark 微调和真实临床记录微调设置，并区分有益知识回忆、无信息量复述和可能泄露的内容。它支持分类监测方案，不提供跨语料、模型和训练阶段的通用记忆率。
- QuaDMix 作为样本级质量/多样性选择候选留在第五章的一般配比方法范围；本章只链接第五章，不新增重复方法段。

## 正文修改

- 修正章首能力迁移和数据收益的过度概括，明确收益依任务和评测而定。
- 将章首文本箭头替换为共享授权/发布门禁、类型专属验证、独立切片评估反馈的 Mermaid 图；解释质量分不能覆盖授权、隐私和污染门禁。
- 补充三类质量分的取值、权重、阈值和不可横向比较条件，并解释缺失测试证据与测试失败的区别。
- 为代码、数学、领域样本分别补齐同一 demo 的分数代入；再手算 secret、PII 硬门禁、分类型 token 保留率和保留后 mixture。
- 增加上述近期代码、数学和医学记忆研究及明确的条件边界；把“能力拓扑”表述收束为比喻与待验证迁移假设。
- 增加继续预训练、SFT、RAG、工具+独立验证器的需求分流表和医学示例；§21 回链第五章通用 mixture 讲解。
- 固定 Python demo 增加输入校验，覆盖空列表、重复 ID、NaN 质量分、非法类型和缺少代码测试；保持核心 12 条教学样本及打印指标不变，并同步预期输出中的 `no_retained_tokens` 状态。
- 在资料清单中加入本轮研究来源及其证据限制。

## 数值复核

- `code_api_doc`：`0.35×1+0.35×(8/8)+0.20×0.85=0.87`。
- `math_word_verified`：`0.45×1+0.40×0.92+0.15×0.55=0.9005≈0.901`。
- `domain_med_guideline`：`0.42×0.95+0.25×0.90+0.20×1-0.35×0.08=0.796`。
- 高分不能覆盖硬门禁：`code_secret_config` 得 `0.80` 仍因 secret 拒绝；`domain_private_case` 得 `0.636` 仍因 PII 拒绝。
- 十二条合成样本原始 token 为 `12,140`，保留 `5,570`，总保留率 `0.459`；代码/数学/领域 token 保留率为 `0.269/0.568/0.621`，保留后 mixture 为 `0.251/0.264/0.485`。这些是教学样例输出，不是建议配比。

## 代码、篇幅与图示验收

- 唯一 Python demo 从 Markdown 抽取到 `/tmp/book09ch06_fixed_demo.py`，运行命令为 `python3 /tmp/book09ch06_fixed_demo.py`；环境为 Python 3.12.9 与标准库。12 条合成样本的输出与正文预期块逐行一致，断言通过；新增空输入、重复 ID、NaN、非法类型和无测试路径均按预期拒绝。未新增测试文件或运行测试套件。
- 本章作为一篇核心讲解单元，按非标题、非围栏、非表格、非引用清单、非状态行、非行内代码/公式的汉字口径统计为 **9,001 个汉字**，高于 1,800 字下限。
- 章首 Mermaid 使用标准围栏、引号节点标签和 `-.->|标签|` 文字虚线；源码语法与围栏配对静态复核通过。GitHub 页面实际渲染未核验。
- 第五章相对链接目标存在；本轮新增研究页面已核对，但未对全章既有外链做 HTTP 全量扫描。

## 配套同步与开放项

`PAPERS.md` 增加七项代码、数学及医学记忆来源；`EXERCISES.md` 修订专项数据审计、demo、verifier 和领域治理题；`INTERVIEW_BANK.md` 更新专项数据面试回答；`GLOSSARY_EN_ZH.md` 增加题目覆盖、求解—验证差距和答案错误类型。简介和目录职责未变。

仍待 GitHub Mermaid/MathJax 页面渲染、整章外链扫描、第九册主阅读路径连续复读，以及真实授权语料、模型、工具环境和专家标签上的项目验证。论文与 toy demo 不证明目标系统收益或专业可靠性，因此本章完整正文门禁保持开放。
