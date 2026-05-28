# 第十四章：MCP、A2A 与 Harness 集成

## MCP

MCP，即 Model Context Protocol，目标是为模型应用连接外部工具、资源和上下文提供标准协议。

Harness 可以通过 MCP 接入：

1. 文件系统。
2. 数据库。
3. 浏览器。
4. 企业系统。
5. 代码工具。

## A2A

A2A，即 Agent-to-Agent，关注 agent 之间如何通信、协作、委派任务和交换结果。

它涉及：

1. agent identity。
2. message format。
3. capability discovery。
4. task delegation。
5. trust and permission。
6. audit。

## Harness 如何集成

Harness 需要把 MCP 和 A2A 纳入统一安全和执行模型：

1. 工具注册。
2. 权限检查。
3. 输入输出过滤。
4. trace 记录。
5. 错误恢复。
6. replay。

## 真实坑

1. MCP server 权限过大。
2. 外部工具返回 prompt injection 内容。
3. A2A 中责任边界不清。
4. 多 agent 协作时 trace 难以统一。
