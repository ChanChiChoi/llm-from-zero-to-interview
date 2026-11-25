# 第二十一册第七十五章精修审计

## 任务与旧稿诊断

本章按“主题思想不变重写”执行，保留 Attention Residuals 沿网络深度读取历史表示的中心主题，重新组织解释路线。旧稿已区分深度选择与 token self-attention，也有一个简化的 Full 计算代码，但用任意分数说明 softmax，没有沿论文的同 key=value 来源向量走完 RMSNorm 打分、深度权重和 raw value 聚合；零初始化的均匀平均、Block 首次/后续读入候选集、块提交和最终输出读入也没有构成完整轨迹。旧稿对残差子层、块数、块宽的关系和 Full/Block 的系统成本讲得概括；K3 段落把 “12-layer size” 简写为每块 12 层，未与固定 config revision 的字段并列核对。

新稿按标准残差、Full 定义、具体 token 走查、Block 状态轨迹、训练与推理成本、论文证据、K3 个案和后续变体组织。标准残差递推明确以 $h_1=v_0$（token embedding）初始化；Full 例子让两个 token 在相同伪查询、同一 $k_i=v_i$ 定义下得到不同深度权重；Block 例子逐个列出 attention/FFN 读入的 embedding、完成块和 partial sum，并交代最终输出聚合。

## 结构与分工

正文和[现行大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/75-Attention-Residuals深度方向聚合.md)均为 8 个 H2、29 个 H3。reader TOC 和设计 TOC 已列出相同标题层级；设计目录链接本章大纲。第 61 章负责 KDA 的序列递归状态，第 76 章负责 Kimi Linear 的 KDA/全局 MLA 混合模型案例，第 78 章负责 mHC 的多路残差混合几何，第 88 章负责 Kimi K3 全模型案例。本章说明 AttnRes 与这些路线的状态轴差异，不将 K3 整体能力归因于 AttnRes。

## 研究依据与证据边界

核心机制、公式、消融和 Kimi Linear 实验依据 Kimi Team [Attention Residuals 原论文](https://arxiv.org/pdf/2603.15031)。Full 复杂度、Block 两阶段 online softmax、pipeline cache 及 <4%/<2% 开销均按论文设置表述。Kimi K3 [技术报告 v2](https://arxiv.org/html/2607.24653v2)与[固定 Hugging Face config revision](https://huggingface.co/moonshotai/Kimi-K3/blob/2496450e92e425c886db095102a52a6682ca3970/config.json)只支持其模型实例的 8 个块、“12-layer size”、partial final block、独立 embedding 来源和 `attn_res_block_size=12`；K3 全模型 benchmark 不作为 AttnRes 单因素对照。AMD MI355X/TokenSpeed 文章只用于说明特定 kernel 设计与资源压力。Low-Rank、RD 与 Multi-Stream AttnRes 的作者结果分别绑定各自预印本模型、训练预算和评测协议，不跨论文排名。

来源角色、版本、阅读位置和未验证边界见[AttnRes 研究记录](../../research/attention-residuals-evidence-2026-11-24.md)。研究记录仅存机制与来源证据；本文件记录修改决定、执行状态和剩余门禁。

## 正文、配套与验收

- 已新增本章完整大纲并重写正文；同步 reader/design TOC、`EXERCISES.md` 前沿架构第 1 题、`INTERVIEW_BANK.md` 第 31CI 与 Frontier Architecture Q1、`GLOSSARY_EN_ZH.md` 的 AttnRes/伪查询/Block AttnRes 和 K3 条目、`PAPERS.md` 的 Multi-Stream/Low-Rank/RD/K3 索引，以及 `KNOWLEDGE_GRAPH.md`。
- 静态检查确认正文、大纲与两份目录的 8 个 H2、29 个 H3 标题名称、层级和顺序匹配；11 个 math 围栏成对、围栏外行内 `$` 定界平衡，未发现 `\operatorname` 或意外双反斜杠；Full/Block 公式和手算结果已逐式核对。正文及两份目录中的本地链接目标存在，31CI 唯一；`git diff --check` 通过。
- GitHub MathJax 页面未实际查看，公式视觉渲染验收未完成；目标读者试读、外链终审和第二十一册连续复读仍留待全册门禁。
- 未运行代码、测试、模型、API、kernel、benchmark 或 PDF 构建；未更新 PDF 或 `plan.md`。

下一章为第七十六章《Kimi Delta Attention：从可更新记忆到线性注意力》。第七十七章与第七十四章重复讲解 CSA/HCA 机制的旧稿问题留待按序处理。全套书精修 goal 继续 active。
