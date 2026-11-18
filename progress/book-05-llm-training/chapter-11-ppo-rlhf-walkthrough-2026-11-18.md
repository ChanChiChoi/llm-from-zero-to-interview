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

## 本轮继续修订：补上训练脚本顺序

- 根据读者反馈，将 §3 开头改为 PyTorch 风格的端到端伪代码：从 Base/SFT 数据与 checkpoint 开始，分别展示 policy、冻结 reference、偏好对训练的 reward model、以 GAE return 为目标训练的 critic，再展示 rollout、token log probability、KL shaping、GAE、PPO minibatch 更新和评估循环。
- 同一算术题贯穿 SFT 示范、chosen/rejected 偏好对和 policy rollout。伪代码之后按其顺序拆开 reward、GAE/value target 与 clipped policy loss；明确 actor 和 critic 使用不同 loss 与 optimizer。
- 更正 token reward 的解释：三个位置的 KL 成本合计 `0.06`，末位置承担 `0.03`，所以加入 `1.50` 的回答分后该位置 reward 为 `1.47`。按教学用 `0.5 × token mean squared error`，critic loss 初值约 `0.5895`。
- 验收：`git diff --check` 通过；手工静态检查确认 Markdown fence 数为偶数、50 个本地 Markdown 链接均可解析且目标存在。约定伪代码不作为可运行代码执行，未运行测试或 PDF 构建。仓库临时目录中的 `check_ch11_review.py` 不存在，因此本地链接使用一次性只读检查复核。
