# 实时语音系统交互评测来源

检索与阅读日期：2026-11-05。

本记录只保存联网发现的评测方法、论文结果及其证据边界，不记录章节审阅或修改过程。

## InteractionBench

- **来源：** [InteractionBench: Measuring Timing, Task Accuracy, Whisper and Volume, and Grounding in Real-Time Voice Systems](https://arxiv.org/abs/2611.02430v1)，arXiv:2611.02430v1，2026-11-02。
- **阅读范围：** 摘要、§3 的 13 个场景与评分规则、§4 harness、§5 系统和结果、§6 讨论、§7 limitations。
- **来源支持：** 论文用同一评测 harness 驱动六个 full-duplex 系统和一个 turn-based cascade；涵盖 conversational timing、spoken-task accuracy、whisper/volume、长对话 groundedness 四类指标。每系统设计 221 个条目、13 个场景，论文报告 1,519 次 graded runs，并说明部分系统存在缺失运行。测试用户音频为同一批预先合成的英语语音，按实时方式送入各系统；评分包含规则化的 floor-control 与 scripted-answer 检查，耳语按音频信号评分，语义相关判断使用 Gemini 2.5 Flash judge。
- **测量口径：** 所有系统从客户端可听端计时，因此 hosted service 的 timing 包含网络与缓冲；论文只能对 GPT-Live 额外用服务端时钟估计网络差异。示例规则包括用户末词结束前 50 ms 至结束后 2.5 s 的 turn-taking 窗口，以及其他场景的明示阈值。作者同时提供替代阈值重评分，说明门槛会改变部分结果。
- **证据等级：** source-backed（公开预印本中的基准构造、harness、论文评分与作者讨论）。
- **边界：** 结果不构成一般产品排名。系统使用的提示与部分协议不同；网络条件只对一个 hosted system 有服务器侧校正；语音为少数预设音色合成的英语，缺少噪声、混响和真实信道效应；每场景样本量为 10–30。论文不测自然度/听者偏好、安全、工具使用或长期任务完成，且 Gemini-family judge 与 Gemini Live 存在模型家族重合。基准结果只适合作为评测维度和协议设计参考。
- **未解决问题：** 需要在真实用户语音、多语言、噪声/电话信道、相同系统提示、网络可分离计时和人工偏好评价下复测；工具副作用和端到端业务成功需要单独设计受控评估。
