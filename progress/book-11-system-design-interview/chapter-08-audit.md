# 第十一册第八章审计：设计实时语音助手

- **审阅日期：** 2026-11-05
- **状态：** 全章扫描、正文精修和本地静态门禁完成；GitHub Mermaid/MathJax 渲染、外链全量巡检及本册连续复读留待最终门禁。
- **范围：** book-11-system-design-interview/chapters/08-设计实时语音助手.md 全章，保留 8.0—8.21 编号和原有章节职责。

## 章级扫描与候选去向

- **扫描问题：** 双向媒体如何保留采集、处理和播放时钟；VAD 与端点如何平衡误截断和等待；partial/final 如何约束预取及副作用；流式 ASR/TTS 怎样处理延迟、稳定前缀、取消和播放偏移；打断、断线、未知写结果如何收敛；语音任务应如何同时评估识别、时序、交付、安全和成本。
- **资料范围与日期：** 2026-11-05。核阅 Emformer v4、FastSpeech 2 v8、VITS PMLR 论文、Moshi v1、InteractionBench v1；复核 W3C WebRTC Recommendation、RFC 6455、OWASP LLM01:2025 与 NIST AI RMF 官方页面。来源事实和版本边界见本章附近引用及[语音识别与生成来源记录](../../research/speech-recognition-generation-and-realtime-2026.md)、[实时语音评测来源记录](../../research/realtime-voice-evaluation-2026.md)。扫描聚焦实时系统设计与评估，不代表穷尽语音识别、合成或生产服务研究。

| 候选主题 | 处理位置 | 去向与理由 |
| --- | --- | --- |
| 流式 Transformer 的历史压缩、左上下文缓存与延迟/WER 取舍 | §8.5.4 | 纳入 Emformer 作为架构实例；LibriSpeech 数据和作者测量仅支持论文设置，未外推到普通话电话或业务字段。 |
| 语音转写的数字字段错误与通用 WER 的局限 | §8.5.5 | 增加固定分词、1/4 替换错误的逐步计算，并解释订单号精确匹配不能由总体 WER 代替。 |
| 流式 TTS 架构差异 | §8.8.3 | 精确说明 FastSpeech 2 的真实目标与韵律条件、VITS 的变分/流/对抗/时长路线及单说话人 MOS 边界；不新增模型专题。 |
| 全双工重叠流的代表架构 | §8.9.1 | 以 Moshi 并行用户/助手音频流说明研究路线，160 ms 理论、约 200 ms 实践数字限定于论文设置。 |
| 面向实时语音系统的多维评估 | §8.14.5、§8.19 | 纳入 InteractionBench v1 的同一 harness、多维指标和合成英语/协议/阈值限制；留在本章，不改变十四章骨架。 |
| 语音转写及检索内容中的提示注入 | §8.10.1 | 引用 OWASP LLM01:2025，明确外部内容不提升为系统命令、权限在代码侧限制；不将指南视为系统安全证明。 |

## 主要修订

1. 将开篇串行文字改为 Mermaid 系统数据流图，标出麦克风、时间戳帧、网关、回声/VAD、ASR、编排、工具、LLM/TTS、播放及插话控制路径；说明停止播放不能撤销外部副作用，并补充与第七、九、十五册及本册第六章的入口。
2. 补全 ASR/LLM/TTS、VAD、WER、MOS 等首次术语说明，合并重复的 W3C 描述，给 WebRTC 与 WebSocket 分别添加 W3C 规范和 RFC 6455 近源链接。
3. 为首声延迟加入 70+420+110+180+160+80=1,020 ms（约 1.02 s）的串行假设；为 600 路活跃音频加入 17.28 Mbit/s（约 2.16 MB/s）的带宽代入；为 300 ms 语音内部停顿对照 200/500 ms 端点阈值。
4. 增加 Emformer 的记忆库、左上下文键值缓存和分块机制及其 LibriSpeech 延迟/WER 结果；为 WER 增加 1/4=25% 的订单号例，说明字段精确性需单列评估。
5. 将轮次文本序列改为 Mermaid 状态图，覆盖 listening、user_speaking、endpoint_pending、thinking、tool_waiting、assistant_speaking、interrupted、transfer_human 与 failed；分清 Orchestrator、Playback Controller 和 Tool Gateway 的状态所有权。
6. 细化 FastSpeech 2、VITS、Moshi 的架构说明和来源边界；强调论文延迟不是用户端到端 SLA。
7. 延展首声/完整响应、LLM 请求率、录音容量、单位成功会话成本、打断成功率和链式任务成功率的可复算例：完整响应 2.42 s；300 会话下约 20.8 次 LLM 请求/秒；0.72 MB/通、21.6 GB/日、648 GB/30 日；1.20 元/成功会话；88/100=88%；条件成功率约 77.1%。假设值均标为教学例，注明计时、分母和遗漏成本。
8. 将配送订单 JSON 明确标为合成数据并更新示例日期；资料表加入 InteractionBench，资料入口改为可读 Markdown 链接并删除自动请求超时等工作流残留。

## 验收记录

- **length_status：** 按连续论证计三个核心教学单元，静态汉字计数分别为 §§8.0—8.4 **2,519**、§§8.5—8.9 **2,530**、§§8.10—8.18 **3,839**，均高于核心单元 1,800 字下限。计数只含解释性汉字，排除标题、表格、围栏、行内代码/公式、链接跨度和空白；§8.19 是证据矩阵，§8.20 是练习，§8.21 是收束。
- **example_status：** 已加入首声 1.02 s、上行 17.28 Mbit/s、端点 200/500 ms、WER 25%、完整响应 2.42 s、LLM 20.8 请求/秒、录音 0.72 MB/通/21.6 GB/日/648 GB/30 日、单位成功成本 1.20 元、打断成功率 88% 和链式任务成功率 77.1% 等逐步计算。论文/benchmark 结果均绑定各自来源条件。
- **code_status：** 13 个围栏块：9 math、2 Mermaid、2 JSON；共 26 个围栏标记且配对。没有 Python、Shell 或其他可执行章节代码；JSON 为事件/业务数据示例。未运行代码、测试、模型或 benchmark。
- **formula_status：** 9 个 math 块逐块检查花括号平衡；未发现 operatorname。公式变量、单位、分母及无样本时的未定义情形均在正文说明。
- **visual_status：** 两个 Mermaid 图分别表达端到端组件/数据流与轮次状态。源码经人工静态检查；GitHub 页面实际渲染留待最终门禁。
- **link_status：** 23 个 Markdown 链接，5 个本地目标全部存在；18 个外链未逐一做全量可达性巡检。新增论文与规范来源已直接引用。
- **sync_status：** PAPERS.md 语音模型与流式对话条目补入 Emformer 和 InteractionBench；Emformer 证据补入既有语音来源记录，InteractionBench 新建独立 research 来源记录。简介、目录的本章摘要与主题承诺仍一致，无需结构同步。未修改 plan.md，未构建或更新 PDF。
- **静态检查：** H2 8.0—8.21 连续，共 22 个；H3 共 76 个；围栏和本地链接检查通过；git diff --check 通过。

## 遗留项与交接

- 两个 Mermaid 图和 MathJax 公式仍需在 GitHub 读者页面确认实际渲染；18 个外链未做全量巡检，保留至第十一册最终门禁。
- InteractionBench 的协议/提示差异、合成英语用户音频、客户端网络时延、规范性阈值和 Gemini-family judge 重合均已在正文及 research 记录；不得将其结果解释为通用排名或生产 SLA。
- 下一步按目录处理第九章《设计视频生成服务》，检查异步长作业、资源排队、媒体资产和交付/回收状态；全套书精修 goal 保持 active，PDF 不更新。
