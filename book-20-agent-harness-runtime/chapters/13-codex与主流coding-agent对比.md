# 第十三章：Codex 与主流 Coding Agent 对比

## 本章定位

本章比较 Codex、Claude Code、OpenCode、Cursor、Aider、SWE-agent、OpenHands 等 coding agent / coding assistant 的系统设计差异。

## 对比维度

1. 产品形态：CLI、IDE、Web、API。
2. 上下文来源：文件、repo、terminal、issue、test。
3. 编辑方式：patch、whole-file、AST、IDE edit。
4. 工具执行：内置工具、MCP、shell、browser。
5. 权限模型：自动、确认、沙箱。
6. 评估方式：SWE-bench、单测、人工评估。
7. 适用场景：修 bug、写功能、解释代码、重构、测试。

## 核心问题

不同 coding agent 的差异，往往不只来自模型能力，还来自 harness 设计。
