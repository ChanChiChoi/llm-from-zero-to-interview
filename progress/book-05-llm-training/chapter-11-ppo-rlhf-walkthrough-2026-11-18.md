# 第五册第十一章：PPO-RLHF 贯穿算例补写

日期：2026-11-18

## 本次处理

- 仅修改 `book-05-llm-training/chapters/11-后训练与对齐训练.md` 的 §3「PPO-RLHF」讲解路径。
- 将原先的六步术语清单改成一条假设训练线：从“云舟-32B”已有预训练 checkpoint 出发，用 SFT 建立指令行为、用人工偏好训练 reward model、由当前 policy 采样、结合 reward 与 KL 形成 token 奖励、由 critic 估计 advantage，再由 PPO 更新并开始下一轮。
- 将后文 reward shaping、GAE 和 clipped ratio 算例接回同一条训练线；模型名、反馈和数值均标为教学假设。对三个生成位置注明实际 token 数由 tokenizer 决定。

## 范围与同步

- 本次用例只解释章节已有的 PPO-RLHF 机制，没有增加方法、时效事实或生产配方；因此未新增 research 记录。
- 未改动代码、章节结构、目录、链接或配套索引；不据此标记本章整体精修完成。

## 检查与遗留

- 手工复核 GAE 与 clipped ratio 的数值：末 token advantage 为 `1.07`，前缀 advantage `A_1≈1.1049`；ratio `0.25/0.20=1.25`，`min(1.25×1.07, 1.2×1.07)=1.284`。
- `git diff --check` 通过；静态检查确认本章 50 个 Markdown 链接的本地目标均存在、132 行围栏标记成对；目录和简介仍指向本章。
- 未运行测试或 PDF 构建。
- 偏好模型分数仍只是对偏好标签的代理，不构成数学正确性证明；本例已在正文注明这一边界。
