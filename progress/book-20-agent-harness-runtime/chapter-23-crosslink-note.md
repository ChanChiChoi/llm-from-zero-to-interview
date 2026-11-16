# 第二十册第 23 章交叉链接维护记录

维护日期：2026-11-16

状态：只记录一次因第十七册章节结构调整而做的回链修复；第二十册第 23 章尚未进行完整章级审阅。

第十七册第七章收整末尾模型案例时移除了旧的 `Agentic Video` 小节锚点。为避免第二十册第 23 章继续指向失效 anchor，已将该处回链改为第十七册第七章文件入口，并明确视频 processing step 的证据、媒体范围与宿主责任由第二十册本章主讲。

后续第 23 章完整精修时还需处理既有引用问题：本章正文和 `BOOK_SERIES.md` 指向 `research/model-update-2026-09/gemini-3.8-flash-source-notes.md`，但该文件当前不存在。届时需按官方 API reference、模型页、Thinking/Video 文档与固定 SDK revision 补建来源记录，复核字段差异和版本/哈希后再关闭该项。
