# 第三册册级架构审视与外部扫描（2026-10-28）

## 书册契约

- **定位与类型**：面向大模型算法岗位的项目训练教材，以可运行的教学实现和可复核实验为中心；不是代码片段词典，也不把单次 demo 说成生产验收。
- **读者与前置**：读者已具备基础 Python、机器学习概念和第一册的模型主线；PyTorch 基础可以边读第一章边补。需要跨越 API 调用，能解释数据、目标、模型状态、测量和失败归因。
- **能力承诺**：读者能从小型自回归模型开始，完成组件实现、训练与评估、监督微调和偏好优化、推理测试、RAG/Agent 应用；能做多模态小型实验；能把一个项目打包成含数据/配置/证据/失败样例的作品，并在面试中守住结论边界。
- **纳入范围**：教学规模的核心实现、使用主流框架的最小适配、项目端到端数据流、离线评估与 debug、推理 trace、合成数值案例、复现清单与面试迁移。
- **不承担范围**：大规模预训练、生产级训练/Serving 平台设计、完整 Agent Harness/工具协议/Infra、安全研究深挖、多模态生成模型谱系和前沿模型综述；分别交接第五、六、八、十五至十七、二十至二十四册。第三册给实践入口和可观察 demo，不替代这些书的主讲。
- **书系交接**：第一、二册给概念与机制；第十四册补 PyTorch 工程；第三册把它们变成项目；第五、六、七、九、十五、十七、二十、二十四册承担专题深挖；第十二册承接求职材料和求职流程。

## 简介与目录审视

原简介承诺“理论转化为代码、可复现实验、项目和面试表达”，但目录只是把 11 个主题并列罗列，且没有标出必须主线、选读分支、阶段产物与交接。第 9 章重新讲 miniGPT、SFT、DPO、Serving 和 RAG 的机制，和第 3—7 章职责重叠；更适合承担作品集证据与交付，而不是重复教程。原多模态文件把图文理解/检索、Diffusion、语音和系统评估放在一个 Markdown 文件内，问题、输入输出和工程流水线彼此不同，单一章节主线不成立。

修订后的全书主线为：

1. **实现与观察基础**（第 1—3 章）：PyTorch 训练对象 → Transformer 张量/组件 → 小 GPT 完整闭环。
2. **把模型改成可用系统**（第 4—8 章）：SFT/LoRA → 偏好优化 → 推理 → RAG/Agent → 评估与 debug。
3. **多模态选读实践支线**（第 9—12 章）：图文理解与检索 → Diffusion 数值实验 → 语音识别问答 → 多模态评估与系统交付。沿用原第 63—72 讲作为历史讲次标识；新文件按单一项目问题拆分。
4. **综合交付与表达**（第 13—14 章）：作品集项目证据包 → 技术表达、系统设计和论文评议。

第 13 章保留六类项目，但把职责改为项目-specific 的数据/接口/指标/失败/证据选择；已在第三至八章主讲的机制不再重复展开。它以一份统一 project manifest 汇总可复现条件、模型和数据版本、结果、失败样例、个人贡献与边界。第 14 章训练读者从约束和证据组织技术表达，不重讲 Transformer 教程。

## 册级外部扫描

检索日期：2026-10-28。问题是：项目实践书的现有评估闭环是否覆盖 Agent 多步行为的定位和回归；Serving 练习是否能代表多轮 Agent 请求；PyTorch/Hugging Face 文档是否改变了已有接口或复现边界；多模态实践部分是否应承载图像/视频生成谱系。

| 候选发现 | 证据与判断 | 去向 |
| --- | --- | --- |
| Agent 评估要从单次最终回答扩展到 trace、工具选择、handoff、guardrail 与节点级诊断，再沉淀为固定数据集做回归 | OpenAI 当前 Agent Evals 文档区分 trace 调试和数据集/Eval Run 重复测量；Agentic CLEAR 研究系统、trace、node 多层诊断；Microsoft Research 的 AgentRx 用受 guard 条件的约束定位首个不可恢复失败。方法方向适合实践教材，但三者的产品能力或论文结果不能互相当作独立复现 | 第 8 章新增 8.7“Agent 轨迹评估与首个不可恢复失败定位”；把原 8.7 线上 A/B 顺延为 8.8。Agent 安全控制面仍由第 7 章讲，失败统计和归因由第 8 章讲 |
| 单轮 Serving 压测无法代表长轨迹工具 Agent 的请求分布 | 2026 年 9 月的 AgentPerfBench 论文用真实 Agent trace 的每轮输入/输出长度和轮数分布构造性能画像，并分析多轮上下文增长与饱和测试。它是新 preprint/benchmark，不把论文自报数字写成硬件通用事实 | 第 6.7 节加入多轮 Agent workload manifest 和逐轮 TTFT/TPOT/工具等待/上下文长度记录；只教本地 trace 驱动的 profile 设计，不宣称复现论文成绩 |
| 微调 API 的损失掩码和默认行为需要固定版本、数据契约和 tokenizer/chat template | Hugging Face TRL 当前 SFT 文档明确 completion-only 与 assistant-only 依数据类型、chat template 和 generation span 支持而变化；文档为 `main`，API 会迭代 | 第 4 章保留概念/接口分层，示例版本和模板契约须显式核对；书内不把当前 `main` 参数默认值写成长期不变事实 |
| 随机种子或 deterministic flag 单独不能保证可复现 | PyTorch 当前文档明确 `torch.use_deterministic_algorithms` 不一定足够，需同时说明软件、硬件和算子限制 | 第 3、8、13 章共同使用运行 manifest；第 13 章统一整理复现产物，不新增一册 |
| 多模态生成已包含许多互异任务和快速变化的 pipeline | Diffusers 官方文档已分别列出 image-to-image、inpainting、video generation 等 recipe；第三册旧章只有一个 DDPM 数值实验。根据既定书系边界，完整文生图/图生图/视频生成专题属于后续 ideas.txt 全书扩展轮次或第十五册，不在本轮临时加进第三册 | 保留并拆分当前已有的图文理解、单一 diffusion 数值 lab、语音识别问答、评估交付；不在本轮扩写成图像/视频生成百科 |

### 主要来源

- [OpenAI Agent Evals 文档](https://developers.openai.com/api/docs/guides/agent-evals)：trace 用于找 workflow failure/regression；明确从代表性 trace 过渡到固定 dataset 与 eval run。
- [Agentic CLEAR, arXiv:2605.22608](https://arxiv.org/abs/2605.22608)：2026 年研究预印本，探索 system、trace、node 三层 Agent 诊断；作者报告的 benchmark 结果只属于其研究设置。
- [Microsoft Research AgentRx 介绍](https://www.microsoft.com/en-us/research/blog/systematic-debugging-for-ai-agents-introducing-the-agentrx-framework/)：首个不可恢复失败、guarded constraints、可审计 violation log 和 115 条人工标注失败轨迹。
- [AgentPerfBench, arXiv:2609.34683](https://arxiv.org/abs/2609.34683)：2026-09-28 预印本，研究真实 Agent 多轮请求分布与饱和测量；本书只吸收 workload representation 思路，不迁移作者的设备结果。
- [Hugging Face TRL SFTTrainer 文档](https://huggingface.co/docs/trl/main/en/sft_trainer)：assistant/completion loss 依 conversational/prompt-completion 数据和 chat template；`main` 是滚动文档。
- [PyTorch deterministic algorithms 文档](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html)：确定性设置的含义及单独启用仍不足以保证应用级可复现。
- [Diffusers Image-to-image](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/img2img) 与 [Video generation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/text-img2vid)：官方实践界面显示生成任务已有多个独立 pipeline，支持本轮把旧的多模态文件按当前项目拆分，也支持将更宽泛生成方向留待书系后续扩展。

## 章节地图与结构决定

| 新章节 | 中心问题与产物 | 前置/交接 |
| --- | --- | --- |
| 1 PyTorch 基础实战 | 数学对象怎样变成张量、梯度、参数更新和训练曲线 | 数学基础；产出小型训练循环 |
| 2 Transformer 组件实战 | 张量怎样穿过 embedding、attention、mask 和 decoder block | 第 1 章；产出组件 shape/数值检查 |
| 3 从零训练小 GPT | 怎样从文本建立 next-token 数据、训练模型并解释结果 | 第 1—2 章；产出 checkpoint、曲线和生成例子 |
| 4 Hugging Face 微调实战 | 怎样把基座模型和对话数据变成受控 SFT 更新 | 第 3 章；产出 SFT/adapters 与行为对照 |
| 5 偏好优化实战 | 怎样从 preference pair 到 RM/DPO，并验证行为变化 | 第 4 章；产出偏好数据和训练比较 |
| 6 推理优化实战 | 模型服务的延迟、吞吐、显存、质量怎样共同测量 | 第 3—5 章；产出固定负载 trace 和报告 |
| 7 RAG 与 Agent 项目 | 如何把外部证据与受控工具组合成可追溯助手 | 第 6 章；产出政策助手与执行 trace |
| 8 评估与 Debug | 如何定义可靠测量、定位失败并安全验证改动 | 第 1—7 章；产出回归集、归因和发布判断 |
| 9 图文理解与检索 | 怎样构造 CLIP/VLM/多模态 SFT 与图文 RAG 的可观察链路 | 第 2、4、7、8 章；产出视觉问答/检索证据 |
| 10 Diffusion 数值实验 | 如何跟算加噪、一次反向更新和采样预算 | 第 1、8 章；深挖交接第十五册 |
| 11 语音识别与语音问答 | ASR 错误如何进入文本问答并影响任务成功 | 第 7、8 章；产出可复核转写与问答轨迹 |
| 12 多模态评估与交付 | 如何核对来源模态、转换、中间证据、权限、质量和成本 | 第 8—11 章；产出审计/发布报告 |
| 13 作品集实战 | 如何将前文项目选题、复现、评估并变成诚实可核验证据 | 第 3—12 章；交接第十二册 |
| 14 技术表达与系统设计 | 如何从约束、机制和证据组织面试回答 | 全册；交接第十二册 |

内容去向：旧第 11 章第 63—67 讲 → 新第 9 章；第 68 讲 → 第 10 章；第 69—70 讲 → 第 11 章；第 71—72 讲 → 第 12 章。旧第 9、10 章整体改号为第 13、14 章，并修复所有活动目录与交叉链接。历史进度记录保留其当时的章号，不回写。

## 逐章审视和完成状态

册级外部扫描、章节地图、简介/目录重写、旧多模态文件拆分和活动导航同步已完成。原第 1—8、13—14 章已有 2026-10-19 至 22 日逐章精修记录；本轮为其 226 个 Python 围栏补了与既有记录对应的紧邻状态，区分实际验证、静态审阅和缺少模型/CUDA 的情况，没有重跑演示或改变可执行语句。

本轮对原第 63—72 讲逐讲补充了数值推演、数据分母/边界和解释性文字；按 WRITING_SPEC.md 第 3.3 节的口径统计，十讲正文均超过 1,800 个汉字。第 6 章与第 8 章的多轮 Agent 负载和 trace 评估增量、以及第 13—14 章的交付/表达职责也已纳入当前结构。PAPERS、EXERCISES、PROJECTS、BOOK_SERIES、README 和 ROADMAP 已按内容和章节去向同步。

2026-10-28 的本地收尾检查已完成：第三册 16 个 Markdown 文件（简介、目录和 14 章）的相对链接、标题重复、围栏配对和 Python 语法静态解析通过；226 个 Python 围栏都有紧邻状态，恢复后的代码正文逐块与当前提交中的对应版本一致。第三册自身没有本地断链；其余活动文档中发现的无关研究链接不属于本册导航，本轮没有改动。

原第 63—72 讲的汉字数依次为 1,937、1,842、1,879、1,990、1,916、1,961、1,916、1,809、2,004、2,104，按 WRITING_SPEC.md 第 3.3 节排除公式、表格、代码、标题与状态行后均达到 1,800 下限。全册当前有 9 个 Mermaid 块，已静态检查代码围栏和标准标签写法；真实 GitHub 页面渲染须等修改进入远端后确认。PDF dry-run 因环境缺少 pandoc 未能生成构建命令。

真实模型权重、GPU、OCR/ASR、线上容量与生产安全结论不由教学代码证明。第三册正文及本地结构检查已经完成；剩余平台验收是 GitHub 上的 Mermaid rich display 与可用构建环境中的 PDF 生成，不把本地源文件状态冒充为这些外部结果。
