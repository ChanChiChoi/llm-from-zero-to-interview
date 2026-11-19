# 第五册第十一章：正文教学路径修订

日期：2026-11-19

## 范围与阅读路径

- 按用户要求，仅处理 `book-05-llm-training/chapters/11-后训练与对齐训练.md` 的正文编写，不重新审查书册定位或章节定位/组织。
- 沿正文现有承诺复核后训练反馈、偏好优化、在线 RL/RLVR、方法比较、训练风险、工程案例、评估与诊断。保留原有章节覆盖、外部链接和第 17 节已验证的分析示例代码；不改训练代码、目录或配套资料。
- 讲解主线继续使用算术题 `$3/4+1/4=?$`：示范回答解释 SFT，chosen/rejected 解释 RM、DPO 和 SimPO，同题多条采样解释 GRPO，PPO-RLHF 保留独立的六步贯穿算例。长回答的 ratio 粒度、采样、裁剪和长度归一化方法则用对应的 token/response 小例子说明。

## 正文改动

- 开篇先区分示范、离线偏好、在线采样与可验证反馈，修正容易误读为固定先后关系的路线图，明确 PPO-RLHF 与 DPO 是不同路线，不是必须顺序经过的阶段。
- SFT 小节改用同一算术题说明示范 token、mask 和两位置交叉熵，并保留 EOS、工具字段与空 mask 的必要边界。
- 为 DPO、SimPO、GRPO、GSPO、DAPO、DCPO 增加有具体内容的背景、目的、问题或意义说明；补齐/统一适用场景与优缺点表达。例子标明假设值，不把 reward 分数说成正确性证明。
- 压缩 SimPO 中重复解释的 margin、平均 log probability 和长度影响，保留 token sum/mean 对目标的区别、调参含义、reference-free 漂移风险与 mask 边界。
- 根据 arXiv 页面元数据更新 GAPO 状态：该论文已被 EMNLP 2026 主会录用，不再与仍为早期预印本的 DHPO 一并称作“早期预印本候选”。正文及研究记录仍将其结果限定为作者报告的模型、任务和协议；来源证据补入 `research/rlhf-preference-and-policy-optimization-2026.md`。
- PPO-RLHF 六步示例及其中的公式、六个教学 token 片段、reward、GAE、returns 和 ratio 数值均保留；不改已验证的第 17 节代码。

## 联网核验与证据边界

- 本轮依模块 03 联网核对正文的原始论文和官方披露；直接复核 PPO/InstructGPT、DPO、SimPO、DeepSeekMath/GRPO、DAPO、DrGRPO、GSPO、DCPO、DHPO、GAPO 的论文页面，以及 gpt-oss、Grok 4.7 的发布页。GLM-5.3 内容沿用 `research/llm-training-and-post-training-systems-2026.md` 已记录的官方来源和证据边界。
- 重点复核论文原文与版本元数据：DAPO 的组合式 recipe 和 AIME24 设定；GSPO 的 sequence likelihood ratio 与 sequence-level clipping；DCPO 的动态 clipping、平滑 advantage 与论文自定义 RUR；DHPO 的 token/sequence ratio 混合；GAPO 的 advantage-adaptive clipping 与 EMNLP 2026 录用状态。
- 论文和发布方报告仅支持各自公开的定义、实验和披露；章节保留“作者在其模型、任务、数据、评估协议下报告”的边界，不据此拼成跨论文总排名或推断未公开训练配方。

## 检查与状态

- 手工复核本轮新增的 SFT 平均损失、DPO margin/logit、GRPO 组内均值/标准差/advantage 示例。
- 完成后检查 Markdown 公式、围栏、链接和 `git diff --check`；未运行测试或 PDF 构建。
- 本记录只表示本轮正文教学修订，不代表重新审定书册或章节结构；若正文发现会改变主题范围或章节职责的新证据，应另按模块 01/02a 处理。
