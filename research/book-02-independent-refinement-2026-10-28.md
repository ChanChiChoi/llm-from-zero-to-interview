# 第二册整册独立精修与审计（2026-10-28）

## 审计范围

本轮先完整阅读仓库写作规范，再以当前第二册正文为审阅基线，独立复核简介、目录、13 个正文单元及 124 个讲次。既有进度记录不作为完成依据；PDF 不在本轮范围内。无须改变的单元保留原文，不为制造改动而改写。

复核重点包括章节交接、概念与公式边界、示例输入输出、论据来源、指标定义、编号与配套索引。对新增或近期研究主张，优先核对论文、官方实现或发布方原始材料，并将论文作者报告与普遍结论区分。

## 单元审阅与修订

| 单元 | 范围 | 审阅结论与本轮处理 |
| --- | --- | --- |
| 1. 深度学习与概率基础 | 1—10 | 概率目标、损失、泛化和诊断脉络连贯；无需实质改写。 |
| 2. Transformer 架构 | 11—25 | 补入离散扩散语言模型的训练与推理机制、成本比较及其与 backbone 分类的区别；为综述和 LLaDA2.0 的作者结果注明证据边界。修正 19、20 节内的小节编号。 |
| 3. Tokenization、数据与预训练 | 26—40 | 核对 tokenization、数据质量、去重、预训练目标和规模化论点；无须改动正文。 |
| 4. 大规模训练系统 | 41—50 | 核对并行策略、通信、训练恢复和吞吐口径；无须改动正文。 |
| 5. 后训练与对齐 | 51—65 | 复核 SFT、偏好优化、RLHF、安全和行为评估的定义及适用边界；无须改动正文。 |
| 6. 推理、部署与服务 | 66—75 | 明确 Prefix Cache 的模型、适配器、分词器、模板、位置/掩码选项和租户边界等兼容条件；补充检索片段 K/V 组合复用与选择性传输研究，并限制到论文配置。统一第 75.8 节 BF16 KV 占用与十进制 MB 单位，更新并复跑对应代码示例。 |
| 7. 长上下文、RAG 与 Agent | 76—86 | 修正第 84.9 节权限演示：补齐 user principal 的 public_docs allowlist、正例和输出，使声明角色与资源白名单一致；复跑代码示例。 |
| 8. Reasoning 与 Test-Time Compute | 87—96 | 增补顺序轨迹、完整候选聚合、未完成前缀搜索三种计算位置的分析框架；区分精确重放和分布可复现，并给出对应统计与重采样边界。 |
| 9. 论文精读与开放研究 | 97—104 | 复核论文主张、实验范围和研究设计方法；正文已把作者结果与可推广结论区分，无须改动。 |
| 10. 多模态理解与对齐 | 105—113 | 复核图像、视频、OCR、连接器、对齐和 grounding 的任务边界；无须改动正文。 |
| 11. 图像与视频生成 | 114—119 | 复核自回归图像生成、扩散和视频成本推导；给对象持久性、事件顺序指标补上非空样本域条件，并把权重定义为任务预设的非负权重。Sora 2 相关说法改为发布方主张，补充官方历史来源和停止提供日期，不暗示当前可用或独立验证。 |
| 12. 语音识别、生成与交互 | 120—122 | 把 Whisper 的语言识别写成语言 token 解码能力及 lang_id 入口，明确它是功能类别而非独立的 detect 控制 token；补官方音频预处理和解码实现来源。 |
| 13. 多模态评估与开放研究 | 123—124 | 补充 RVG-Bench 的任务规模和作者报告结果，并明确仅适用于其 benchmark、模型和协议；同步论文索引第 31 条。 |

## 一手来源与证据边界

- Whisper：原始论文 [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)用于核对训练范式和论文实验范围；官方 [audio.py](https://github.com/openai/whisper/blob/main/whisper/audio.py) 与 [decoding.py](https://github.com/openai/whisper/blob/main/whisper/decoding.py)用于核对 16 kHz 音频、30 秒窗口、Mel 特征、转写/翻译控制与语言 token 解码。论文基准、开源 checkpoint 和具体业务表现分别陈述。
- 离散扩散： [LLaDA2.0](https://arxiv.org/abs/2512.15745)用于核对作者报告的 100B block-diffusion 模型及训练阶段；[How To Build A Modern Diffusion Language Model](https://arxiv.org/abs/2610.33391)作为 2026 年综述。综述分类与单篇论文结果均不构成普遍性能结论。
- 推理与服务： [Test-Time Scaling in Reasoning LLMs](https://arxiv.org/abs/2608.04001)用于核对计算结构分类及复现建议；[Tessera](https://arxiv.org/abs/2609.32999)讨论带位置/模型条件的可组合 K/V 复用；[SmartGen](https://arxiv.org/abs/2607.28150)报告选择性 K/V 传输实验。速度和准确率数字都限制在作者公布的模型、硬件、网络与任务配置。
- 多模态评估： [XModBench（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/21071811269c70ac11a3e7869bac747e-Abstract-Conference.html)、[LVOmniBench](https://arxiv.org/abs/2603.19217)、[WBench](https://arxiv.org/abs/2605.25874) 与 [RVG-Bench](https://arxiv.org/abs/2610.32268)分别检查跨模态一致性、长音视频理解、交互式视频世界模型和推理驱动的时间定位；不同数据和指标不合并为统一排行榜。
- Sora： [OpenAI 原始视频生成技术报告](https://openai.com/index/video-generation-models-as-world-simulators/)和 [Sora 2 发布说明](https://openai.com/index/sora-2/)是发布方的一手材料。正文将其技术描述标为发布方主张和历史资料；该产品停止提供的信息不作为模型技术性能证据。
- 推理与论文精读中核对的近期来源包括 [KisMATH](https://aclanthology.org/2026.tacl-1.59/)、[PaCoRe](https://aclanthology.org/2026.acl-long.1253/)、[ParaThinker](https://aclanthology.org/2026.findings-emnlp.1469/)、[Timely Machine](https://aclanthology.org/2026.acl-long.211/)、[ThinkRetrieve](https://aclanthology.org/2026.findings-emnlp.1703/)、[RAG over Thinking Traces](https://aclanthology.org/2026.emnlp-main.1964/)、[FOL-Traces](https://aclanthology.org/2026.findings-eacl.115/)、[Thinking Optimal](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3e22bea3b170f4c2aebb9c48d98ae64d-Abstract-Conference.html)、[DPO 数据研究](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3f37b8fbd43303106dd141a602838ad5-Abstract-Conference.html)、[FlashAttention-4](https://arxiv.org/abs/2603.05451)、[MoE 调度研究](https://arxiv.org/abs/2602.21626)和 [LoFT](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7428310c0f97f1c6bb2ef1be99c1ec2a-Abstract-Conference.html)。引用中的性能数字保留其任务、实现及实验条件，不外推为通用保证。

## 校验结果

- 目录覆盖 1—124，章节范围与正文文件一致；旧多模态章节文件仅保留到四个新单元的兼容链接。
- 对本册 16 个 Markdown 文件作只读静态检查：124 个讲次编号连续、无重号；排除代码、表格、标题、行内代码和公式后的讲次说明正文最少 1,826 个汉字，满足规范的 1,800 字门槛。
- 代码围栏闭合；相对 Markdown 链接目标存在；行内数学环境 begin/end 配对。
- 2026-10-28 使用 Python 3.12.9 标准库复跑第 75.8 节路由示例和第 84.9 节权限示例，并据实际输出刷新正文状态和输出。
- `git diff --check -- PAPERS.md book-02-advanced-100` 通过。未运行测试套件；未修改或构建 PDF。

