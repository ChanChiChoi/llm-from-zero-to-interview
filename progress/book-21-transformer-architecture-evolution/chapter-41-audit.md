# 第四十一章《卷积序列模型在 LLM 中的复兴可能性》精修审计

## 范围与诊断

按全套书精修的主题重写要求保留章名、文件路径和目录位置。完整复读旧稿后发现其 32 个 H2 反复展开感受野、变量绑定/精确引用、流式缓存、snapshot/rollback、p99 与一般硬件验收；同一边界被多次重述，部分内容越入第六、二十四册 serving 与 kernel 职责、第四十二章架构选型，以及本章之前的 SSM/Hyena 专题。旧稿还把通用 im2col/Winograd、频域状态、量化等内容写成卷积 LLM 的普遍工程结论，缺少本章相称的证据；“多层感受野计算器”不承载 LLM 卷积机制，且 $R_{\mathrm{cite}}$ 被放在 math 围栏中。

先重新设计六节章内大纲并同步目录，再依此重写：三种卷积角色、attention 前局部深度卷积、长卷积、StripedHyena 混合主干、训练与生成的结构成本、实验范围与证据边界。保留因果卷积、局部/长程范围、门控与 attention 差别、全序列与增量计算等读者必要知识；删除重复的 serving runbook、未经来源支持的普遍硬件清单和玩具代码。

## 研究与内容

研究问题、实际检索词、来源版本及逐项证据边界见[卷积序列模型研究记录](../../research/convolution-sequence-models-in-llms-2026-11-23.md)。以 arXiv 2026 技术报告核对 Qwen3 上 17 个 Conv1D 位置、P5 选择、残差 depthwise $k=3$ 与 FineWeb/七项 benchmark 设置；报告显示平均分均有提升，但一些单项下降。正文保留摘要“新增参数低于 $0.01\%$”与详细参数表之间的未解释差异，不把精确比例当作确定事实。

以 ICML 2023 长卷积论文及方法作者博客核对核平滑正则和 H3+LongConvs 的 Pile perplexity；该变体仍保留两层 attention。以 Together AI 官方发布文章核对 StripedHyena-7B 的 attention/gated-convolution 混合、ZeroScrolls 作者报告和其 FIR/IIR 表示。课程讲义与 ShortConv 教学页仅用于组织局部卷积解释，不作为 Qwen3 或 StripedHyena 的架构证据。结论限制为局部补充、混合序列层和混合主干，不宣称纯卷积主干已普遍替代 Transformer。

## 配套同步

新增六节有效大纲，并在第二十一册目录加入 41.1—41.6。PAPERS.md 增补 2023 长卷积论文、2026 Qwen3 卷积技术报告和 StripedHyena 发布材料；KNOWLEDGE_GRAPH.md 补充卷积从局部到长程再到混合的路径；GLOSSARY_EN_ZH.md 增加 depthwise/long convolution 术语。新增练习 31AT 与面试题 31AU。

## 验收与遗留

- 正文、大纲、目录均为六个小节且标题顺序一致；第 39、40、42 章职责分别保持在 Hyena 算子、早期混合路线和横向架构比较。
- 公式围栏使用 `math` 而非代码围栏；逐式复核了因果索引、残差项、卷积核长度、FFT 补零下界和计算复杂度适用形状。Qwen3、Pile 与 ZeroScrolls 数字均带各自模型、预算/任务和来源边界。
- GitHub MathJax 页面逐式渲染未验收，公式门禁仍开；未发布本地修改。外链最终逐条复核、目标读者试读与第二十一册连续复读留待全册门禁。
- `git diff --check` 及本地导航/题号检查在本轮完成后记录。未运行代码、测试、模型、API、kernel、benchmark 或 PDF 构建；不更新 PDF 或 `plan.md`。

第四十一章完成首轮修订。下一章为第四十二章《替代架构横向对比：复杂度、表达力、硬件和生态》。
