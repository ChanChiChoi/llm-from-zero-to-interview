# 第十五册第五章精修审计：多模态 Instruction Tuning

审阅日期：2026-11-11

状态：本章精修、章级扫描和正文门禁通过；2026-11-12 整册复读时补齐 ratio/策略字段类型校验并复验代码；继续第六章整册复读。按用户要求不更新 PDF。

目标章节：[第五章《多模态 Instruction Tuning》](../../book-15-multimodal-generative-models/chapters/05-多模态instruction-tuning.md)

## scope 与章节结构

- 全文复读 §§0–14，保留整篇为一个连续核心讲解单元，未拆分或增删本册章节。
- 主问题是如何把媒体对象、对话、模板、证据、标签和策略要求整理成一致且可审计的监督样本，并核实这些样本是否支持预期行为。
- 教学链从样本 schema 和媒体引用开始，经 chat template、assistant-only loss mask、任务与质量、数据混合、batch/collator 和评估，最后落到合同页面问答案例与可运行的数据契约审计。
- 承接第四章的视觉 memory、placeholder/image-object 映射和有效 label 契约；生产级数据治理交第九册，规模化训练交第五册，通用评估方法交第七册，多模态任务评估继续第十一章。本章职责未改变，简介与目录中的第五章摘要仍准确。

## chapter_scan：范围、证据与去向

- **日期与范围：** 2026-11-11。定向检查多模态指令数据格式、视觉证据与 claim 支持、chat template/Processor、assistant-only loss、任务混合、拒答及多模态安全数据。资料类型包括原始论文/基准、固定版本框架文档、官方安全风险资料及仓库已有研究记录；未穷尽所有 VLM、数据集、服务 API 或训练器。
- **研究问题：** 媒体引用和消息模板如何对齐；训练目标应覆盖哪些 token；任务样本如何区分并混合；如何检查答案证据、拒答和过度拒答；论文、文档与教学 demo 各自能支持什么结论。

| 候选主题 | 来源与证据范围 | 去向与决定 |
| --- | --- | --- |
| 视觉指令微调的数据与训练路线 | LLaVA v2、InstructBLIP v2、LLaVA-1.5 v2、MiniGPT-4 v2；论文入口及版本历史、摘要和正文引用范围。 | 用作不同公开研究路线的实例，不把某个数据规模、阶段或冻结配置写成通用训练配方；完整规模训练策略交第五册。 |
| OCR、图表问答和多模态拒答任务 | TextVQA v2、ChartQA v1、MM-SafetyBench v5 的固定论文版本；来源支持各自任务和评估动机。 | 保留为任务类型与评测入口；不将单一 benchmark 表现等同于通用 OCR、图表理解或安全保证。 |
| 模板、Processor 与 assistant-only loss | Transformers v5.21.0 多模态 chat template、TRL v1.16.0 SFT Trainer、PyTorch 2.15 `CrossEntropyLoss` 官方文档。 | 说明消息编译、Processor 职责、模板 mask、`ignore_index` 和 reduction；文档版本约束限定于所引接口，不外推至其他训练器。 |
| 数据集说明与多模态 prompt injection 风险 | *Datasheets for Datasets* arXiv v8；OWASP LLM01:2025。 | 将数据集说明用于记录来源和用途，将注入作为策略/来源审查入口；不把框架写成法规，也不把建议写成已验证的系统控制。 |
| 样本质量、混合和效果判定 | 论文与官方文档提供任务背景；比例、分母、合同案例和指标数值在正文中明确作为教学输入。 | 核心用例留在本章；数据治理流程和生产评估协议分别链接第九册与第七册，不增加章节。 |

**扫描结论：** 候选问题均可放入现有单篇教学链，未发现需要改变第十五册职责或十四章骨架的主题。论文事实按特定来源、版本、任务和评估设置表达；教学数值不作为模型或生产实测。

## 修改与教学链

- 明确多模态样本是媒体对象、消息、模板、标注、证据和策略组成的文档；补齐 `media_id`、schema、manifest、annotation、claim、policy、bbox、quote 等术语与字段含义。
- 解释模板如同编译器：把角色、控制标记、媒体占位符和视觉对象引用序列化为模型输入；补充多图/多轮生命周期及 Processor 与 collator 的职责边界。
- 统一 assistant-only labels、shifted next-token targets、padding/packed/ragged batch 的有效 token 分母，区分 attention mask、loss mask 与业务质量规则。
- 按 caption、VQA、OCR、chart、grounding、多轮、多图、拒答等任务解释监督目标和数据缺口；补入数据质量、标注一致性、来源文档、采样比例、token 分布及梯度方向的推演。
- 将评估扩展到 claim-level evidence support、refusal recall、over-refusal、exact match、IoU、反事实支持和单位成功任务成本；每个指标说明判定单位与分母。
- 用合同页面问答贯穿媒体引用、OCR/grounding、答案证据和拒答边界；最小 demo 只审计合成样本契约，明确不验证图片理解或训练后的行为。

## length_status

- 全篇一个核心讲解单元，解释性中文实计 **13,281 字**，高于 1,800 字下限。
- 统计纳入讲解、练习和回顾文字；排除标题、表格、代码/JSON/数学围栏、行内代码与公式、链接目标、状态行和参考资料。篇幅通过不替代因果解释、数例和证据边界的人工审阅。

## example_status

- 通过。正文包含多组可复算输入，均作为教学输入而非实测：
  - 两个样本的 NLL 归一化对照：两个 token 的概率分别为 `0.5、0.5` 和一个 token 概率 `0.25`，推得 token mean `0.924` 与 sample mean `1.040`，说明不同归约方式改变目标权重。
  - 多媒体长度 `2×576+320+8=1,480`；分辨率 `2048×1024` 中 bbox `(512,256,1536,768)` 归一化为 `(0.25,0.25,0.75,0.75)`。
  - 图表从 40 增至 60 单位，变化率为 50%；四个 claim 中三个有支持，支持率为 `3/4=0.75`；一致性例子的 `p_o=0.9, p_e=0.5` 得到 `κ=0.8`。
  - OCR 与拒答任务混合 `0.25×80=20`、`0.75×8=6`，按 token 算的占比分别为 `0.769/0.231`；梯度示例 `0.75×2+0.25×(-1)=1.25`。
  - 精确匹配容差例（100 对 102、阈值 3）、bbox IoU（交集 8、并集 24、IoU `0.333`）、反事实率 `3/4=0.75` 和单位支持成功成本 `6/3=2`，均说明对应分母、假设或容差。

## code_status

- §11 唯一独立运行的 Python 标准库数据契约 demo，状态为“已验证（2026-11-12；Python 3.12.9；仅 Python 标准库）”。整册复读发现 `safe_ratio` 与 `safe_max` 可接受浮点/布尔计数，策略标记类型也未校验；现要求非负整数计数、布尔策略字段和字符串任务名，正常样本结果不变。运行于 CPU，不依赖外部库。
- 验证命令：`./.tmp/llm-book-cpu-check/bin/python -c 'from pathlib import Path; p=Path("book-15-multimodal-generative-models/chapters/05-多模态instruction-tuning.md"); s=p.read_text(encoding="utf-8"); code=s.split("~~~python\n",1)[1].split("\n~~~",1)[0]; exec(compile(code,str(p),"exec"))'`
- Python 3.12.9 从 Markdown 提取代码块执行并捕获 stdout，与正文预期输出逐字一致：8 条样本通过、`bad_ocr_001` 因 `placeholder_mismatch`、`unsupported_claims`、`missing_refusal`、`safety_risk` 被拒；8 类任务覆盖率 `1.0`；标签和 assistant token 均为 117；最大总 token 数 1,204；10 条 claim 中 9 条受支持，support rate `0.9`；refusal recall `0.5`；over-refusal rate `0.0`。8 个契约断言和 9 个边界断言全部为 True。
- 检查拒答、过度拒答及支持率的分母后，新增确认：ratio/max 拒绝浮点/布尔计数，非字符串任务和非布尔策略标记产生审计错误；空分母、非字符串 prompt、负 token 数和空 assistant answer 仍按预期处理。
- 本例使用九条合成记录，不解码图像、不调用模型、不训练；不能证明 OCR/grounding 正确性、真实数据质量、模型拒答行为或生产成本。

## formula_status、link_status 与 visual_status

- **formula_status：** 19 个 math 围栏，花括号均平衡；行内 `$` 标记 260 个，即 130 对。未发现 `\\operatorname`、`\text{}` 中带下划线变量或 `<t` 形式下标。源码结构已检查，未做 GitHub/浏览器 MathJax 实际渲染。
- **link_status：** 共 10 个 Markdown 链接，均为外链；无本地目标。本轮没有对全部外链进行 HTTP crawl，固定版本论文/官方文档入口与来源范围记录在本章和研究笔记中。
- **visual_status：** 一张三列表格表达资料类型和证据边界；公式、样本 schema/消息映射、任务覆盖与 token 账本承担数量及数据流解释。没有 Mermaid；本章现有视觉表达适合其以样本契约为中心的主线，无需装饰图。
- **源码结构：** 41 个 fenced blocks 配对（13 JSON、19 math、1 Python、8 text）；一个 Markdown 表格的列数一致；代码实际输出与预期输出逐字匹配。

## sync_status 与 open_risks

- **sync_status：** `PAPERS.md` 增补/固定 LLaVA v2、InstructBLIP v2、MiniGPT-4 v2、TextVQA v2、ChartQA v1 和 MM-SafetyBench v5；`EXERCISES.md` 第十五册 SFT demo 题补入 claim 支持、拒答召回、过度拒答及被拒样本分母；`INTERVIEW_BANK.md` 多模态 SFT 数据审计题同步相同指标口径；`GLOSSARY_EN_ZH.md` 补清 Evidence Support Rate、Refusal Recall 与 Refusal Accuracy 的分母说明。固定版本论文与框架接口的知识、证据范围和未验证边界补入既有 `research/instruction-tuning-data-and-templates-2026.md`，没有把修改流水写入研究记录。简介和目录的第五章职责摘要已复核，无需调整。
- **open_risks：** 未对真实图片、扫描文档、真实模型或训练后 checkpoint 做 OCR/grounding、幻觉、拒答/过度拒答或人类偏好验证；未验证目标 Processor、collator 和训练器组合；未测 GPU、吞吐、延迟或生产成本；未做全部外链 HTTP 检查或目标平台公式渲染。具体论文效果、模板行为和框架接口仍绑定其列明版本与任务条件。
- **handoff：** 第十五册第一至第五章已按主路径完成整册复读。下一步继续[第六章《Diffusion 基础》](../../book-15-multimodal-generative-models/chapters/06-diffusion基础.md)整册复读，检查加噪、训练目标、反向过程和采样器的机制链与可复算推导；十四章骨架保持不变。
