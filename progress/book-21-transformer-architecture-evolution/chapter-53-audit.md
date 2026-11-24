# 第五十三章精修审计

## 章节职责与旧稿诊断

第五十三章位于第二十一册的模型演进时间线，回答后训练怎样改变生成式预训练模型的使用行为。它承接第五十一章的 GPT 历史和第五十二章的预训练预算，说明 InstructGPT/ChatGPT 的历史位置；不取代第五册第十一章的 SFT/RM/PPO/DPO 算法讲解，也不与第五十八章的 reasoning RL/RLVR 混写。内部 Transformer block、服务系统和对话产品实现分别由相关章节承担。

旧稿有 27 个二级标题，反复展开 SFT、奖励模型、PPO、DPO、reward hacking、能力评估、工具安全、训练流程和玩具代码；多个小节重复同一“偏好是代理目标”的结论，并把模型协议、执行器权限和产品验收揉在一起。按 02a 对照本册定位、目录、上下游章节和第五册训练章节后，将职责收束为五节：预训练与指令任务的差距、InstructGPT 的训练信号、偏好结果及限制、ChatGPT 发布期对话入口、架构/行为/产品边界。

## 研究依据与编辑决定

- InstructGPT 原论文支持 GPT-3 基座上的 SFT、偏好比较/RM、PPO 流程，以及 API prompt 分布上的人类成对偏好实验。正文把 $85\pm3\%$ 与 $71\pm4\%$ 绑定各自的 GPT-3 基线和同一偏好比较条件，不将胜率外推成跨任务能力排名。
- 论文对标注群体代表性、事实错误、偏见评估和部分公开 NLP 回归的说明，支撑将人类偏好与事实、安全和普遍价值判断分开。
- OpenAI 的 2022 年 ChatGPT 发布页支持初始 GPT-3.5 微调、标注者扮演用户/助手、偏好比较和多轮对话描述；正文明确这些是发布期资料，不描述当前产品全貌或架构细节。
- Ziegler 等（2019）和 Stiennon 等（2020）的原始研究用于校准 RLHF 的历史起点，避免把偏好学习写成始于 InstructGPT。来源和逐项支持边界见[研究记录](../../research/instructgpt-chatgpt-rlhf-evidence-2026-11-24.md)。

## 正文和同步

- 正文按五节重写，使用阶段表解释 SFT、偏好比较/RM 和 PPO 各自接收的信号及其证明边界；保留限定条件下的核心实验结果，区分模型结构、后训练行为和对话产品。
- 删除重复的算法推导、DPO/RLAIF/Constitutional AI 总览、玩具代码、生产评估清单和工具执行器安全流程；将算法细节链接到第五册第十一章，将 reasoning RL/RLVR 链接到第五十八章。
- 新建完整五节[章节大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/53-InstructGPT-ChatGPT-RLHF.md)，同步[第二十一册目录](../../book-21-transformer-architecture-evolution/目录.md)、[架构时间线](../../book-21-transformer-architecture-evolution/主流大模型架构创新时间线.md)、EXERCISES.md 练习 31BK、INTERVIEW_BANK.md 面试题 31BL、PAPERS.md 第 50 项和 KNOWLEDGE_GRAPH.md。术语表已有 RLHF 条目，本轮不新增。

## 验收与遗留事项

- 正文、大纲与目录的五个小节标题和顺序一致；章节、目录、大纲、配套索引、研究记录和进度交接中涉及本章的本地目标逐项检查，无缺失目标。练习 31BK、面试题 31BL 和论文索引第 50 项各唯一。
- 正文无代码围栏、可执行代码或 Mermaid；包含模型规模与偏好比例行内数学表达式，不含块公式。变量、比例、单位和偏好率的数学定界符已检查；**GitHub MathJax 实际页面渲染尚未查看，行内数学渲染门禁仍待验收**。
- git diff --check 通过。未运行测试、代码、模型、API、benchmark 或 PDF 构建；未更新 PDF 或 plan.md。
- 下一章为第五十四章《LLaMA 系列：RoPE、RMSNorm、SwiGLU 与开源 LLM 基线》。目标读者试读、外链最终逐条复核、第二十一册连续复读及本章行内数学页面渲染仍待整册/全局门禁；全套书精修 goal 继续 active。
