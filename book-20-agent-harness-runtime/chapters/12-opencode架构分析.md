# 第十二章：OpenCode 架构分析

## 本章定位

本章用于分析 OpenCode 这类开源 coding agent 的架构思想，重点关注它如何组织工具、权限、上下文、编辑和执行流程。

## 分析重点

1. Agent 配置。
2. Tool system。
3. 文件读写。
4. Shell execution。
5. Patch editing。
6. Session state。
7. Permission rules。
8. MCP 或外部工具扩展。
9. 多模型适配。

## 对比 Claude Code

可以从以下角度对比：

1. 开放性。
2. 工具抽象。
3. 权限模型。
4. 可扩展性。
5. 上下文管理。
6. trace 能力。
7. 用户体验。
