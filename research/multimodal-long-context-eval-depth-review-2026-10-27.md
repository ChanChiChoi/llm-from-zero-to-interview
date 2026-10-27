# 第七册第七章：多模态与长上下文评估研究记录

- `date`：2026-10-27。
- `scope`：依照当前 `WRITING_SPEC.md` 复核 `book-07-evaluation-experiments/chapters/07-多模态与长上下文评估.md` §1—§20；本轮新增 benchmark 方法对照和 ANLS 协议审计，未删除原章节覆盖的任务、公式、流程、练习或结论。
- `research_question`：一个评估方法如何把某类失败从总分中分离出来？它的样本如何构造、实际支持什么判断、受什么假设限制？针对该限制，下一步应加哪种反事实、任务切片、证据标签或真实样本？
- `basis`：原章节已有细分指标与 toy demo，但基准来源主要列举名称，读者无法据此判断 VQA v2、POPE、MMMU、DocVQA、ChartQA、视频/音频基准及多种 long-context evaluation 各自解决的问题。新增 §15 作为方法谱系与选择指南；在 OCR / 文档小节新增 ANLS 机制和评分协议敏感性。

## 原有知识点承接图

| 原章范围 | 原有知识点 | 本轮处置 | 新位置/说明 |
| --- | --- | --- | --- |
| §1—§2 | task contract、结果/过程/迁移、多模态能力与证据向量 | 保留；新 benchmark 选择以“要区分哪一种失败”为入口 | §15 开头承接 |
| §3 | VQA soft accuracy、语言先验、图像反事实、计数/空间/否定、grounding | 全部保留；补 VQA v2 互补图像对和 POPE 对象存在探测的研究设计 | §3、§15.1 |
| §4 | CER/WER、字段准确率、表格结构、低质量输入、拒答 | 全部保留；新增 DocVQA 的 ANLS 公式、阈值跳变、格式敏感例子和单页/集合检索任务区别 | §4.5、§15.1 |
| §5 | chart 任务层级、数值误差、趋势和错误归因 | 保留；补 ChartQA 人写/合成任务构造、5% 数值容差和容差边界 | §15.1 |
| §6—§8 | 视频 tIoU、采样协议、语音 WER/DER/MOS/RTF、grounding 和反事实 | 全部保留；补 AudioBench、Video-MME、MLVU 的任务覆盖与评测边界 | §15.1 |
| §9 | 上下文长度、NIAH、位置网格、检索局限 | 全部保留；把 NIAH、Lost in the Middle、RULER、NoLiMa 按新增难点串成研究路线 | §15.2 |
| §10—§11 | 多证据 QA、摘要、冲突、状态/代码库、多页 PDF、长视频、交叉模态任务 | 全部保留；增加 LongBench 家族和 MMLongBench-Doc 的构造与证据边界 | §15.1—§15.3 |
| §12—§13 | 客观/人工/LLM judge、错误树和 evaluator 错误 | 全部保留；将协议/解析器变化纳入具体的 ANLS 和各 benchmark 对照 | §4.5、§15、§16.3 |
| §14 | 标准库 toy demo、多种指标、门禁和诊断动作 | 保留；给延迟失败门禁补动作，把输出标为静态核对预期值，并声明阈值不是 benchmark 或生产门槛 | §14 |
| §16—§20 | 项目闭环、常见误区、练习、证据边界、结语 | 全部保留；现有空缺的 §15 改为 benchmark 研究与组合指南，编号连续 | §15—§20 |

本轮没有移除旧稿中的任务维度。新增 benchmark 不是把它们都要求为每个项目必测，而是作为对应问题的研究入口和选型依据；真实项目仍需按目标用户分布选择主路径并建立 holdout。

## 原始来源与研究发现

| 来源 | 实际阅读内容与它回答的问题 | 收益证据 | 限制及对正文的影响 |
| --- | --- | --- | --- |
| [VQA v2 / Making the V in VQA Matter](https://arxiv.org/abs/1612.00837) | 论文通过同一问题对应的互补图片（图片相似但答案不同）来削弱语言先验。 | 测试模型是否随视觉证据变化而改变答案，比只比较独立题目更能针对题面先验。 | 答案变化不等于能指出正确位置；正文要求继续用区域 grounding、遮挡/替换反事实补证。 |
| [POPE](https://arxiv.org/abs/2305.10355) | 阅读论文 §3—§5 与限制：自由描述解析对象名受提示和解析器影响；POPE 用 Yes/No object probes，并以 random / popular / adversarial 构造不存在对象。 | 报告 accuracy、precision、recall、F1 和“Yes”比例；作者在其模型/COCO设置中观察到负例从 random 到 popular、adversarial 时更难，并报告模板扰动下 POPE 相比 CHAIR 更稳定。 | 只测 object existence；对象词表、负例抽样和标注颗粒度影响难度。正文要求按采样模式报告，不把它扩展成所有属性、数量、空间关系的幻觉率。 |
| [MMMU](https://arxiv.org/abs/2311.16502) | 阅读摘要和任务构造：原论文从考试、测验、教材收集 11.5K 多模态题，覆盖六大学科、30 subjects / 183 subfields 与多种图像类型。 | 让评估包含学科知识和需推理的图表/地图/结构等证据，补足生活化 VQA。 | 论文准确率只表示该题库/版本/提示/解析下表现；学科平均不能代表专业工作流程或风险操作，正文要求报告学科和任务切片。 |
| [DocVQA 原论文](https://arxiv.org/abs/2007.00398) 与 [官方 2020 challenge](https://www.docvqa.org/challenges/2020) | 原论文介绍 12,000+ 文档图像、50,000 问题；官方挑战区分单页 image QA 和文档集合检索，前者 ANLS，后者 MAP。官方页还规定 Task 1 的大小写/空格处理。 | ANLS 对小 OCR 拼写偏差比 exact match 柔和；collection retrieval 与 page QA 不同，必须分开讲。 | ANLS 仍是字符串分数，不能同时代表读懂、字段结构、证据区域、格式约束。正文增加公式、可手算反例，并明确两种任务的不同分母和指标。 |
| [Most of a Document-VQA Score Is Answer Formatting](https://aclanthology.org/2026.docinsights-1.6/) | 读 ACL Anthology 的 2026 DocInsights 论文记录和摘要，检查其用的 ANLS、提示尾缀、post-processing 和控制实验。 | 作者报告在其 Qwen2.5-VL-3B 条件下，仅去掉 terse-answer suffix 就使一个 DocVQA ANLS 结果从 94.2 变成 7.5；还报告不同 suffix 在特定模型/切片中造成较大变化。 | 这是新发表的单篇审计，报告范围由作者模型、prompt、harness、样本与格式决定，不证明所有已有 DocVQA 排名无效。正文把它作为协议可变性证据，保留上述条件，要求固定并报告 suffix/解析器/格式。 |
| [ChartQA](https://arxiv.org/abs/2203.10244) | 阅读 arXiv HTML 的数据构造和 §5.1 metric：真实来源图表、9,608 名人写 QA、23,111 个从人工图表摘要生成的问题；非数值 exact match，数值答案按 5% relaxed accuracy。 | 将视觉属性指代、数据抽取、比较/差值/聚合连接到一个 benchmark；论文说明有人工 QA 和生成 QA 两条路线。 | 合成 QA 的人工验证只是部分抽样；5% relative tolerance 的零值/小值语义及单位不由 headline score 表达。正文要求拆人写/合成，另做接近零、单位、轴尺度和步骤切片。 |
| [AudioBench](https://arxiv.org/abs/2406.16020) | 阅读论文摘要及 NAACL 2025 论文入口：原始版本 8 类任务、26 个数据集、7 个新增数据集，覆盖 speech understanding、audio scene understanding 和 paralinguistic voice understanding。 | 明确指出 ASR 一项无法代表 AudioLLM 的声音场景、指令跟随和副语言理解；各任务应分开看。 | 异质数据集/指标组合成的总分不可直接解释成统一能力；AudioBench 不等于 TTS 质量、流式 latency 或双向语音对话验收。正文保留原 WER/CER/DER/MOS/RTF 专项指标并补任务层边界。 |
| [Video-MME](https://arxiv.org/abs/2405.21075) 与 [MLVU](https://arxiv.org/abs/2406.04264) | 阅读两篇原论文摘要：Video-MME 的 900 段/254 小时/2,700 QA、11 秒至 1 小时和字幕/音频设置；MLVU 的多任务长视频、类型多样性和长时长目标。 | 对照时长、视频类型及音频/字幕能显式看多模态条件的影响。 | QA 准确率会混合帧采样、字幕、音频、时长和问题模板，且不覆盖 temporal localization。正文建议固定采样/模态条件，并和事件顺序/tIoU 配套。 |
| [MMLongBench-Doc](https://arxiv.org/abs/2407.01523) | 阅读原论文摘要：130 份 PDF，平均 49.4 页、20,971 text tokens、1,062 专家题；33.2% 跨页，22.8% unanswerable。 | 同一任务可检查跨页、文字/图表/布局证据整合及不可回答拒答；作者还对比视觉模型与 lossy OCR 文本输入。 | 数据来源和样本规模有限，论文结果不说明企业合同审阅或动态文件流程。正文列出页码/区域、版本冲突和目标语料为后续补证。 |
| [Lost in the Middle](https://arxiv.org/abs/2307.03172) | 论文分析 multi-document QA 和 key-value retrieval，并移动相关信息位置。 | 作者发现不少被测模型在相关材料靠首尾较好、中间较差，建立了位置敏感评估协议。 | 这不是所有长上下文任务统一的“U 形定律”；不同任务、模型、prompt 要重新测。正文以多长度位置网格承接。 |
| [RULER](https://arxiv.org/abs/2404.06654) | 论文从 vanilla NIAH 扩展多种数量/类型 needle、multi-hop tracing、aggregation，原论文评估 13 项任务。 | 展示 NIAH 单针高分与长度/复杂任务下降可以同时发生，说明“找得到一个显眼串”不是充分证据。 | 合成数据可控但不覆盖真实文本分布；正文建议与自然多任务来源组合。 |
| [NoLiMa](https://arxiv.org/abs/2502.05167) | 阅读 2025 ICML 论文摘要：needle 与问题减少 lexical overlap，以测 latent associations，而非字面定位。 | 作者报告 13 个至少标称 128K 的模型中，11 个在 32K 时跌到各自短上下文基线的一半以下。 | 数字只绑定原论文基准、模型与提示；语义关联合成题仍非长摘要或真实跨文档推理。正文只当一种探针，不写通用容量定律。 |
| [LongBench](https://arxiv.org/abs/2308.14508) 与 [LongBench v2](https://arxiv.org/abs/2412.15204) | v1 原论文为中英双语 21 个数据集/6 类任务；v2 为 503 道高难多选题、8K–2M words、6 类真实长上下文任务，采集与审核包含近百位专业背景贡献者。 | v1 覆盖长上下文应用面，v2 把重心推向需深理解/推理的高难题；v2 给出 human 与直接作答、长推理模型的论文内对照。 | v1 多任务/多指标不宜只看平均；v2 的 15 分钟人类基线和模型数值属于作者时间/模型/评分条件，不代表线上用户分布。正文保留任务级分数和原有参数。 |
| [LongBench Pro](https://arxiv.org/abs/2601.02872) | 阅读 2026-01 arXiv 版本摘要：1,500 条中英文自然样本、11 primary/25 secondary tasks、8K–256K token；LLM 起草后由专家校验。 | 研究者试图让长上下文长度/语言/依赖类型/难度可以细分，并以模型辅助构造扩展高难题覆盖。 | 当前引文只据论文作者提交版本，需独立验证审核稳定性、跨语言难度和对真实任务迁移；正文标记为预印本，不把作者发现写成社区定论。 |

## 方法谱系与对照结论

1. **语言先验与幻觉**：VQA v2 通过互补图像对把“同一句问题”变成视觉依赖测试；POPE 又把自由描述收窄为 object yes/no probe，使样本和错误率更易自动化，但也缩窄了所能主张的幻觉类型。后续补证从图像反事实向定位、属性、计数和开放式生成扩展。
2. **文档指标协议**：ANLS 初衷是降低 OCR 轻微字符错漏对问答分数的硬惩罚。它通过归一化 edit distance 和 `τ=0.5` 实现软评分；不过 `gold=Paris, output=Answer: Paris` 时距离 `8/13≈0.615`，最后仍被阈值置为 0。2026 DocInsights 案例显示 prompt 输出格式可成为 hidden benchmark contract。改进应并列 ANLS/EM、抽取到的 span/field、结构/证据和格式指标，并固定评测提示/后处理。
3. **图表数值分数**：ChartQA 的 5% relaxed numeric accuracy容忍自动图表数值抽取小误差，但并非适合所有零值/单位/相对变化业务。后续应让容差源于决策风险和刻度精度，支持零附近绝对容差，同时保留 exact answer、单位、运算与 axis slices。
4. **长上下文路线**：NIAH 是容易自动测的检索 smoke test；Lost in the Middle 控制位置；RULER 增复杂度；NoLiMa 降词面重叠；LongBench 系列扩多任务和真实材料；MMLongBench-Doc 把文本长上下文推进为跨页视觉证据。每层增加的变量应尽量独立，否则模型变差后很难归因。论文分数不跨数据集排名。
5. **视频/语音**：Video-MME、MLVU、AudioBench 建立模态和任务广度，但不能替代章节既有 WER/DER/MOS/tIoU/RTF 等以任务指标为单位的细化评估；多任务 benchmark 要输出 per-task/per-modality breakdown。

## 修订与验证记录

- `chapter_changes`：在旧章节 §4 追加 ANLS 公式、`Paris` 手算、2026 DocVQA harness 审计及协议要素；在原本跳号处新增 §15，分别比较多模态 benchmark、长上下文 benchmark 路线和真实项目评估组合。§1—§14、§16—§20 原有主体均保留。
- `demo_review`：标准库诊断 demo 未运行（遵循本轮不执行测试/脚本要求）。静态确认每个 `signals` 的失败分支都有对应 `actions`，包括原来遗漏的 `latency_within_budget`；参考输出仍因其输入 `p95_latency_ms=5100 <= 6000` 不新增 latency action。文中明确参考输出是静态核算，不是运行结果；合成阈值不代表官方 benchmark 标准或生产门禁。
- `sync_status`：更新 `PAPERS.md` 的长上下文论文线、多模态评测来源与第七册正文导航；`EXERCISES.md` 增加 6I benchmark portfolio 设计；`INTERVIEW_BANK.md` 增加 49E 方法对比题；第七册目录本已包含本章，无导航改动。
- `validation`：手工复核 ANLS `8/13≈0.615`；检查章节编号连续为 §1—§20、code fence 成对、来源条目和本地链接目标存在；运行 `git diff --check`（若记录时通过）；未运行 Python demo、测试套件、benchmark、公式渲染器或 PDF 构建，也未做全章所有外链 HTTP 扫描。

## 未验证项与证据边界

- 所有数字都是各来源作者在其任务、数据、模型、提示、harness 与版本下的报告，不代表本书目标模型测量；特别是 LongBench v2 的 human/model 对照、NoLiMa 的 32K 结果、POPE 各采样模式和 DocInsights ANLS suffix 对照不可跨设置比较。
- 2026 DocInsights 是单篇近期审计；结论适合触发 protocol audit，不足以推断所有 benchmark 结果均受同等影响。
- 没有目标 VLM/AudioLLM、用户图像/文档/视频/语音分布、模型响应、标注员或私有 holdout；不能声称业务多模态质量、跨页证据率、音视频鲁棒性、长上下文有效长度或用户风险已验收。
- demo 代码和数值为 toy 级教学输入，未执行；它不连接真实模型、tokenizer、processor、解码器、运行时、延迟监控或 benchmark scorer。公式外部渲染、PDF 构建和全量链接扫描没有完成。
- `LongBench Pro` 在本轮仅核对 2026-01 arXiv 作者版本；后续引用应重新确认论文状态、数据版本和官方评测实现。
