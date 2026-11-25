# 第 81 章精修审计

日期：2026-11-25

## 诊断与决定

旧稿（1,076 行）已完整阅读。它把 CED、CSA2、FP4 KV、SWA replay、MoE/Engram、mHC、DSpark 与视觉输入等模型架构，同 API/DSML 协议、deepseek-recipe、Harness/MCP/webhook、长篇 vLLM/SGLang release 清单和多套 serving benchmark 混在 21 个正文小节里；正文目录只列章节名称，读者目录与实际正文不同步。书册契约要求本册解释结构、信息流、计算/存储取舍和证据边界，完整接口、Agent runtime 与推理引擎应回链相应书册。

按主题思想不变重写：保留 CED、CSA2/HSI、Bounded Replay、FP4 global KV、MoE/Engram、Single-Pass mHC、DSpark、视觉路径与报告证据，按模型身份、机制、资源和证据重新组织。API、Harness、协议和引擎实施细节不再充当本章架构内容。第 81 章大纲明确 14 个 H2、32 个 H3；正文、读者目录和设计目录按该结构同步。

## 版本与证据边界

正文以官方 1017 checkpoint 为当前公开模型对象，结构机制与报告数字仍绑定 2026-09 报告及对应固定 revision。两代固定配置仅支持本章结构字段相符，不证明权重、训练数据、后训练和 serving runtime 完全相同。官方 1017 benchmark 只作为指定任务/harness 下的发布方整体结果，不归因到单个架构组件。backbone、Engram、激活参数、global KV 字节和上下文长度分开表述，未对 HF 自动参数统计与模型卡口径自行求和。

详细技术发现、固定链接、教学来源和未决证据边界见[研究记录](../../research/deepseek-v4.1-flash-architecture-evidence-2026-11-25.md)。有效章节设计见[完整大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/81-DeepSeek-V4.1-Flash-CED-CSA2与缓存重算.md)。

## 同步与验收

正文逐层解释 CED/CSA2 状态、候选池筛选、SWA replay 和结构账本。正文、完整大纲、读者目录和设计目录已同步；相关练习、面试题、PAPERS、术语表、知识图谱、BOOK_SERIES 与 ROADMAP 已按新职责调整。架构证据保留在研究记录，章节取舍和进度保留在本审计。

静态核对结果：大纲、正文和两份目录的 14 个 H2、32 个 H3 标题、层级与顺序完全一致；两组 math 围栏成对，行内美元定界平衡，无 `\operatorname` 或意外双反斜杠；Mermaid 源码围栏闭合，语法已静态复核。第 81 章、两份目录、研究记录、进度记录及新整理的 PAPERS 段落本地链接均存在；`git diff --check` 通过。整体扫描 BOOK_SERIES/ROADMAP 时还发现与本章无关的既有研究/进度链接缺失，未在本章任务中改动这些引用，留待相应索引审阅时处理。

本章修改了数学公式并新增 Mermaid 图；GitHub MathJax 与 Mermaid 页面尚未实际查看，故公式与图表渲染验收未完成。未运行代码、测试、模型、benchmark 或 PDF 构建；未更新 PDF 或 `plan.md`。读者试读、外链终审和第二十一册连续复读属于整册门禁，不由本章单独宣称完成。

## 交接

第 81 章完成首轮正文修订；第 82—95 章共 14 章待精修，下一入口为第 82 章《K2 Horizon 3.7B 与 MoVA 36B/A4B：dense、value routing、长上下文和推测解码》。全套书精修 goal 继续 active。
