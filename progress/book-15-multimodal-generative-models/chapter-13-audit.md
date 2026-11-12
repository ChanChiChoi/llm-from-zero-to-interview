# 第十五册第十三章精修审计：Encoder-Free 统一多模态

审计日期：2026-11-12

## scope

- 正文：[第十三章《Encoder-Free 统一多模态》](../../book-15-multimodal-generative-models/chapters/13-encoder-free-unified-multimodal.md)，覆盖 §§13.0–13.18。
- 本章职责是沿第十章架构地图，深入模态前端、token/latent 序列、位置、mask、训练与处理器协议；第十一章承担通用多模态评估和安全，第十四章承担请求级媒体预算。
- 保留十四章骨架和第十三章选读定位；未更新 PDF 或 plan.md。

## chapter_scan：候选与决定

- 完成章节全文审读和定向扫描，范围包括 encoder-free / early-fusion 术语、连续与离散表示、序列布局、位置和可见性、loss 归一、统一模型代表路线、processor manifest、反事实评估、媒体注入边界及演示代码。
- 2026-11-12 核对原始论文/技术报告：Chameleon arXiv v2、Unified-IO 2 v1、Emu3 v1、Show-o v7、Transfusion v1、ImageBind v2、Qwen3.5-Omni 技术报告 v2；并复用现有安全标准研究记录中的 OWASP LLM01:2026 边界。没有引入论文成绩或跨设置排名。

| 候选问题 | 正文决定 | 证据与边界 |
|---|---|---|
| 第十章已介绍 unified/native/end-to-end 等术语，存在重复风险 | 本章开头明确只深入前端到共享序列的数据路径，不重讲完整架构分类或请求预算。 | 第十章保留 taxonomy；第十三章处理序列/前端；第十四章接请求账本。 |
| 表示统一公式用 E_m 表示所有模态映射，容易把 encoder-free 写成“没有任何 encoder” | 改用中性 g_m，说明它可以是 patchifier、tokenizer、codec、专用 encoder 或其组合；离散量化公式同样使用 g_m。 | 映射符号本身不指定前端深度或模块边界；独立 encoder 的取消范围必须由来源明确说明。 |
| unified、early-fusion、token-based、共享嵌入被混成同一架构声明 | 表格比较七条路线：混合早期融合、统一任务 encoder-decoder、离散 next-token、自回归与离散扩散、文本 next-token 与图像 diffusion、共同嵌入、具名的专用前端。 | ImageBind v2 明确保留各模态 encoder 和 projection head；Qwen3.5-Omni v2 报告列出 AuT、SigLIP2、Thinker/Talker。共享表示或 early fusion 不推出 encoder-free。 |
| 公式使用 operatorname，且关键表示只给符号 | 把 Interleave、Transformer 改用 mathrm；补充 mask softmax、压缩失真、loss 归一和 token 预算的可复算例。 | 所有公式变量、假设和边界在正文解释；玩具输入不表述为论文或产品实测。 |
| 章节讲了反事实和 mask，但示例把声明当成已执行检查 | 加入 paired counterfactual 设计和 task-specific mask 解释；演示代码只检查 schema / manifest，并将不完整声明置为 False 或拒绝非法输入。 | 代码不会构造线上 attention mask、运行模型或执行反事实实验；不能把 manifest_complete 当作模型质量门禁。 |
| 13.16 示例对位置、processor 和反事实使用硬编码 True | 改成由 config 输入 position_schema、mask_schema、counterfactual_plan 和 processor_revision；加入数据类型及边界验证。 | 检查结果只支持教学 manifest 的一致性，不证明实际 processor、tokenizer 或 kernel 行为。 |
| 13.14 安全内容可能重复第十一章 | 保留仅与 encoder-free 数据路径相关的信任标签传递、媒体内容与 system/tool 角色分离、执行器权限检查，并链接 OWASP 风险说明。 | 第十一章仍承担完整安全测量和运营响应；指南不构成单项缓解措施有效性的实验。 |

## length_status

按连续中文解释性正文统计，排除标题、代码、表格、公式、引用清单及行内代码。

| 单元 | 类型 | 汉字数 | 下限 | 结果 |
|---|---|---:|---:|---|
| §§13.0–13.6：前端、统一层次、序列、位置、mask 与表示路线 | 核心 | 2,499 | 1,800 | 通过 |
| §§13.7–13.12：压缩、训练目标、干扰、取舍、研究路线与部署协议 | 核心 | 1,951 | 1,800 | 通过 |
| §§13.13–13.15：评估、反事实、安全与故障回放 | 核心 | 2,039 | 1,800 | 通过 |
| §13.16：token / manifest 审计 demo 说明 | 辅助 | 1,113 | 1,000 | 通过 |

## example_status 与 formula_status

- §13.5：三个 attention logits 为 [0, ln 3, 2]，屏蔽未来 key 后权重为 [1/4, 3/4, 0]；说明 mask 在 softmax 前改变可见关系。
- §13.7：合成的 10 个字段例子中，64 个 token 时错误率 1/10=0.10，16 个 token 时 3/10=0.30；序列减少 75%，任务错误上升。
- §13.8：文本两个 token 的均值 loss 为 0.4，音频六个 token 的均值为 0.8；按有效 token 合并为 0.7，等权模态均值为 0.6，分母不同会改变模态权重。
- §13.13：50 份合成合同得到 OCR 覆盖 46/50=0.92、区域定位条件准确率 43/46≈0.935、数值计算条件准确率 40/43≈0.930、引用支持率 38/40=0.95、全链路 38/50=0.76；所有样本上的数值正确率另为 40/50=0.80。另有错误 OCR 跟随率 15/20=0.75。均明确为教学数据。
- §13.16：图片 576、音频 480、视频 784、文本 96、特殊 token 12，总数 1,948；单序列单 head 单 layer 的稠密 score-cell 代理为 1,948²=3,794,704。更改图像 patch 14→28 后图像 token 为 144、总数 1,516、代理为 2,298,256。此代理不等于 FLOPs、显存、延迟或服务成本。
- 13 个 display-math 围栏花括号均衡；无 operatorname 和行内美元公式；Interleave 与 Transformer 用 mathrm。未声称完成浏览器 MathJax 渲染。

## code_status

- §13.16 唯一独立 Python 标准库 demo 在 Python 3.12.9 执行。实际 stdout 与正文参考输出逐行一致。
- 覆盖并通过 7 类非法输入拒绝：bool 伪装 token 整数、NaN、无穷时长、非字符串 media_trust、重复反事实项、畸形媒体映射、浮点乘法溢出。
- 覆盖 6 个不完整配置状态：位置 schema 错误、反事实计划缺项、超 context、空 processor revision、错误 trust 标签、mask 规则为 False；各自都令对应检查和 manifest_complete 返回 False。
- 超大整数时长可安全得到超预算状态而不因 float 转换溢出；patch 敏感性计算结果与正文一致。
- 代码仅核对教学 manifest、整数预算和声明，不读取真实媒体、不调用模型、processor 或 GPU，不测质量、延迟、显存或安全效果。

## visual_status 与 link_status

- 新增一个 Mermaid 数据路径图，节点标签已加引号，使用标准 flowchart LR 语法；未在 GitHub 页面做实际渲染检查。
- 16 个 Markdown 链接中本地目标存在；论文/技术报告和 OWASP 来源已按结论贴近链接。本轮未做全量 HTTP crawl。
- 10 组反引号围栏和 14 组波浪围栏配对；两张三列表格的列结构一致。

## sync_status 与 open_risks

- 更新简介中的第十三章摘要；目录核对后保持不变。
- PAPERS.md 补充六条统一多模态路线入口；GLOSSARY_EN_ZH.md 增加 Encoder-Free Multimodal Route；INTERVIEW_BANK.md 第 49 题细化统一序列、共同嵌入和不同生成目标；KNOWLEDGE_GRAPH.md 增加 encoder-free 声明核验路径。
- research/multimodal-systems-and-evaluation-2026.md 新增 ImageBind v2 的分模态 encoder、projection head 与共同 embedding 证据及边界；Qwen3.5-Omni 与其他路线复用既有来源记录。审计和纳入决定仅在 progress/。
- 仍未做 GitHub Mermaid/MathJax 实际页面渲染和全部外链 HTTP crawl；本地结构、链接目标、公式和代码已检查。
- 保留的证据边界：所有数值都是构造示例；实验反事实计划不等于运行结果；公开报告只支持指定版本与披露范围。整册连续复读已读完 §§13.0–13.18，并对照第十章架构地图、第十一章评估/安全和第十四章请求预算职责；未发现新增结构问题。将正文中的 OWASP 引用从 2025 版更新为 2026 正式条目，以保持当前版本和第十一章一致；本章代码未变，沿用已有可追溯验证记录。

## 下一步

- 第十三章门禁通过。下一步进入第十四章《Native Multimodal Token Budget》，把分辨率、patch、帧采样、audio token rate、上下文、cache 和质量放入同一请求级预算，同时避免重讲第十章架构 taxonomy 和本章序列机制。
- 第十五册第十四章结束后沿主线连续复读并做整册门禁；全套书精修 goal 继续 active。按用户要求不更新 PDF 或 plan.md。
