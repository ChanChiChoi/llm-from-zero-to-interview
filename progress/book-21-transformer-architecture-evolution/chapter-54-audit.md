# 第五十四章精修审计

## 章节职责与旧稿诊断

第五十四章负责说明 LLaMA 家族经典稠密基线的形成、组件所处的信息路径，以及后续版本对这条基线的延续和改变。它承接第五十一至五十三章的时间线；第十七章负责逐版本读取公开配置，第五十七章负责开源训练数据与工程成熟度。归一化、RoPE、GQA、MoE、长上下文和多模态的完整机制由本册专题及第十五册承担。

旧稿有 26 个二级标题，重复解释 RoPE、RMSNorm、SwiGLU、开源权重与模型评测，并包含通用训练数据清单、参数/显存公式、部署成本、兼容性核对、toy PyTorch block、p99/SLO 选型和多个重复 manifest。它同时与第十七章版本配置比较、第十八章位置外推和第五十七章训练工程交叠。依据 02a 收束为五节：原始稠密基线、组件的信息路径、家族版本变化、Llama 4 对经典配方的边界、可追溯基线的判断方法。

## 研究依据与编辑决定

- 原始 LLaMA 论文支持模型规模、公开数据集及 Pre-Norm/RMSNorm、SwiGLU、RoPE 组合，并直接援引 GPT-3、PaLM、GPT-Neo 等先前工作。正文据此称其组合已有设计，不称这些组件由 LLaMA 首创。
- Llama 2 论文支持 $2\mathrm{k}$ 到 $4\mathrm{k}$ 的上下文扩展、较大模型使用 GQA 和 $70\mathrm{B}$ 发布模型；这些变化与数据和训练变化并行，正文不把总结果归因于单一部件。
- Llama 3 技术报告、Meta Llama 3.1 发布说明分别支持稠密 Transformer、GQA 与 $128\mathrm{k}$ 上下文等版本信息；正文明确区分 Llama 3 与 Llama 3.1 的模型范围。
- Meta 当前模型目录与 Llama 4 模型卡支持 Llama 3.2/3.3 分支范围，以及 Scout/Maverick 的 MoE、early fusion、专家数、总/激活参数和 context 字段。context 数字按官方披露处理，不外推为任务级可靠性或成本保证。
- 来源类型、阅读位置和支持边界见[研究记录](../../research/llama-family-architecture-evidence-2026-11-24.md)。

## 正文和同步

- 正文重写为五节，以组件路径表和版本演进表说明结构变化，补入 Llama 4 对经典稠密“LLaMA-like”基线的边界。
- 删除重复机制推导、通用数据工程教程、参数/服务成本模型、玩具代码、评测/发布清单和重复 manifest；深讲链接回第 4、6、7、18、21、22 章、第十五册和第五十七章。
- 新建完整五节[章节大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/54-LLaMA现代开源LLM基线.md)，同步[第二十一册目录](../../book-21-transformer-architecture-evolution/目录.md)、[架构时间线](../../book-21-transformer-architecture-evolution/主流大模型架构创新时间线.md)、EXERCISES.md 练习 31BL、INTERVIEW_BANK.md 面试题 31BM、PAPERS.md 第 51 项和 KNOWLEDGE_GRAPH.md。术语表已有 RMSNorm、RoPE、SwiGLU、GQA、MoE 等条目，本轮不新增。

## 验收与遗留事项

- 正文、大纲、目录的五个 H2 标题与顺序一致；正文引用的 9 个本地章节目标存在，目录、练习和论文索引的本地目标也已核对。练习 31BL、面试题 31BM、论文索引第 51 项各唯一。
- 正文无代码围栏、可执行代码或 Mermaid；行内模型规模、上下文与参数数量的数学定界符成对，不含块公式。**GitHub MathJax 实际页面渲染尚未查看，行内公式渲染门禁仍待验收**。
- git diff --check 通过。未运行测试、代码、模型、API、benchmark 或 PDF 构建；未更新 PDF 或 plan.md。
- 下一章为第五十五章《Mistral/Mixtral：GQA、Sliding Window Attention 与稀疏 MoE》。目标读者试读、外链终审、第二十一册连续复读和本章行内公式页面渲染仍待全册/全局门禁；全套书精修 goal 继续 active。
