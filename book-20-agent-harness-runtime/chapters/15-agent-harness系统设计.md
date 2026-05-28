# 第十五章：Agent Harness 系统设计

## 面试题

设计一个 coding agent harness。

## 需求澄清

1. 支持读取代码库。
2. 支持搜索。
3. 支持编辑文件。
4. 支持运行测试。
5. 支持权限控制。
6. 支持 trace 和 replay。
7. 支持 evaluation。

## 高层架构

1. UI / CLI。
2. Session Manager。
3. Context Builder。
4. Model Adapter。
5. Tool Registry。
6. Execution Engine。
7. Permission System。
8. Sandbox。
9. Trace Store。
10. Eval Runner。

## 关键 trade-off

1. 自动化程度 vs 安全。
2. 上下文完整性 vs 成本。
3. 工具能力强弱 vs 风险。
4. 速度 vs 可观测性。
5. 通用性 vs 产品体验。
