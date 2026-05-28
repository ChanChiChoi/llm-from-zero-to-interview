# 第十六章：Harness 实战坑与面试题

## 高频坑

1. 工具权限过大。
2. trace 不完整导致无法 debug。
3. prompt template 变化导致 eval 不可比。
4. 文件编辑覆盖用户改动。
5. 终端命令有副作用。
6. 工具输出过长挤掉关键上下文。
7. agent 循环调用导致成本失控。
8. 缺少 replay 导致线上事故无法复现。

## 高频面试题

1. Harness 是什么？
2. Coding agent 为什么需要 runtime？
3. 如何设计 tool registry？
4. 如何设计权限系统？
5. 如何设计 evaluation harness？
6. Claude Code、OpenCode、Codex 的核心差异可能在哪里？
7. MCP 如何接入 harness？
8. 如何防止 agent 执行危险命令？
9. 如何记录 trace 并支持 replay？
10. 如何评估 coding agent？

## 回答原则

回答 harness 问题时，要从模型、工具、上下文、状态、安全、可观测性、评估和用户体验几个维度展开。
