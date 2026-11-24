# 第13章精修记录（重置流程）

日期：2026-11-24

章节：`book-05-llm-training/chapters/13-训练成本与资源规划.md`

## 执行范围

- 按本轮重置流程对照第13章章节契约，完整审阅正文；不读取既有 `progress/` 记录。
- 对 `6ND`、MFU/HFU、MegaScale 扩展与故障数据、compute-optimal/inference-aware scaling、公开能耗估算和 MLPerf Training 测量目标查阅原始论文或官方页面。

## 正文修订

- 区分执行 FLOPs 中的实际序列长度与有效 token 数：dense kernel 若仍计算 padding，padding 会增加执行量；varlen 和 packed block-diagonal kernel 应按实际长度/样本块计数。
- 明确 MFU 的吞吐 token 单位必须和每 token FLOPs 口径一致。assistant-only label 吞吐不能直接乘以基于输入序列 token 定义的 `6N`。
- 为成本审计 demo 明确 `train_tokens` 的口径：按模型实际处理的非 padding 序列 token 计。

## 研究记录

原始论文、官方基准页面、证据等级与适用边界记录在 [`research/training-resource-planning-evidence-2026-11-24.md`](../../research/training-resource-planning-evidence-2026-11-24.md)。

## 验收

- 正文 15 个主章节和 35 个编号小节与第13章章节契约对应。
- 检查 18 个 Markdown 链接，没有未解析的本地目标。
- 检查 fenced code/math block：反引号围栏 20 行、波浪号围栏 64 行，均成对闭合。
- `git diff --check -- 'book-05-llm-training/chapters/13-训练成本与资源规划.md'` 通过。
- 新增研究和进度记录的尾随空白检查通过。
- 未运行章节 demo、测试或 PDF 构建。

## 后续

按目录顺序进入第14章；继续使用本轮重置后的规范与研究流程，不从旧 `progress/` 记录恢复导航。
