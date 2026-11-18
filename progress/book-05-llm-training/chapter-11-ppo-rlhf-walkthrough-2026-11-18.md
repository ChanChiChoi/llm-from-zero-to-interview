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

## 本轮继续修订：压缩 PPO 伪代码

- 根据读者反馈，删去伪代码里的 dataloader、loss 展开、反向传播、优化器和梯度裁剪等实现步骤；SFT、reward model 训练、reward shaping、GAE 和 PPO 更新改用简短动作函数表示。
- 保留读者需要追踪的主线：从云舟-32B-SFT 复制 policy 与冻结 reference，用 chosen/rejected 偏好对训练并冻结 reward model，从 SFT 初始化 critic，再按“采样—打分与 KL—GAE—更新 policy/critic—评估”循环。
- 伪代码仍与后文同一条算术回答衔接；详细 reward、GAE、value target 和 clipped ratio 手算留在后文，不在伪代码重复推导。
- 伪代码为说明流程的非执行片段。完成 `git diff --check`、Markdown 围栏和链接的静态检查；本轮未改公式。未运行测试或 PDF 构建。

## 本轮补充 score/value head 结构

- 在不展开训练器细节的前提下补齐两个 head：`score_head = Linear(H, 1)`，将完整回答末尾的隐藏向量映射为一个回答分数；`value_head = Linear(H, 1)`，将每个生成位置的隐藏向量映射为该前缀的 `V(s_t)`。
- 伪代码其余主流程与后文算例保持不变。检查结果：`git diff --check` 通过，134 行 Markdown 围栏成对，50 个本地链接目标存在；未运行测试或 PDF 构建。

## 本轮重写 PPO 的 reward 与 GAE 说明

- 根据读者反馈，删去 §3 中为演算临时设定的 RM 分数、token log-ratio、value、returns、loss 和 PPO ratio 数值链；改用同一条回答说明“整条回答分数落在末 token、KL 项逐 token 计算、GAE 把终局信号传回前缀、returns 训练 critic”的关系。
- 补上对应的符号表达和终止/截断 mask 说明；clipped surrogate 的数值例子沿用前文 §1.3，避免在 §3 重复另设一组数字。将 `β` 的额外数值例子改为其对偏好分数与 reference 偏移的定性取舍。
- 此次改写覆盖此前记录中的 §3 教学手算；旧数值只代表较早版本的编辑记录，不再对应当前正文。
- 本轮未改变章节结论或研究事实。检查结果：`git diff --check` 通过，138 行 Markdown 围栏成对，50 个本地链接目标存在；公式围栏已静态检查。未运行测试或 PDF 构建。

## 本轮按伪代码调用顺序重写 PPO 说明

- 根据读者反馈，删去伪代码后脱离流程的通用目标、重复 KL 解释和分散的工程监控段，改为沿循环逐项解释 `sample_and_record`、`shape_rewards`、`critic_model.value`/`GAE`、policy 更新、critic 更新和评估。
- 补清各模型来源与训练方式：policy/reference 从 SFT 复制且只有 policy 更新；reward model 接 score head 后用 chosen/rejected 偏好对训练并冻结；critic 接 value head 后用 GAE returns 作为目标训练。用同一条算术回答连接 rollout、token reward、advantage 和后续更新。
- 保留必要公式和边界说明，包括逐 token reward shaping、GAE 递推、截断与终止的区别，以及 reference KL 与 PPO 新旧策略 ratio 各自约束什么；移除本节没有解释作用的任意数值演算。
- 本轮仍是 §3 的局部教学改写，不改变章节结构、研究结论或配套索引。目标正文和进度文件的 `git diff --check` 通过；本章 132 行代码围栏标记成对，50 个 Markdown 链接中的 5 个本地目标均存在；公式围栏已静态检查。未运行测试或 PDF 构建。

## 本轮补充 GAE 小例子

- 根据读者反馈，沿用 PPO 的算术回答，在 GAE 处加入三步手算：从末尾 reward 和三个前缀的 critic 预测出发，逐步计算 TD residual、向前递推 advantage，并得到 critic 的 returns target。
- 明确三步是便于讲解的生成位置简化，不代表 tokenizer 切分；暂时忽略逐 token KL，只展示终局分数怎样传回前面，避免把示例数值误当作真实 reward 标尺或训练配方。
- 目标正文和进度文件的 `git diff --check` 通过；本章 132 行代码围栏标记成对，50 个 Markdown 链接中的 5 个本地目标均存在；新增公式与推算已静态复核。未运行测试或 PDF 构建。

## 本轮修复 PPO 公式的 GitHub 渲染

- 排查 §3 的公式源码：两处块级 `math` 公式原先缩进在有序列表步骤中，改为顶层公式块；将六个有序列表项改成带步骤号的普通段落，避免公式继续嵌套在列表容器里。
- 检查 §3 行内 `$...$`：定界符成对、均在同一物理行内，公式内容两侧没有空格，因此没有改写这些行内公式。
- GitHub 文档支持 `$...$` 行内公式和 `math` 围栏块；本次保留仓库使用的 `math` 围栏格式，并确保公式块从顶格开始。
- 静态复核：全章代码围栏成对，其中 35 个 `math` 公式块均无缩进；行内公式扫描未发现缺失定界符、跨行或定界符内侧空格。`git diff --check` 通过；未运行测试或 PDF 构建。
