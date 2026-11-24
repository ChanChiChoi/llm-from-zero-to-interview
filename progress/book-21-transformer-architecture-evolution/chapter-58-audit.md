# 第五十八章精修审计

## 章节职责与旧稿诊断

本章定位为阅读 DeepSeek-V2、V3 与 R1 固定版本公开资料的架构和训练路线案例。读者需要分清 V2/V3 的架构关系、V3 的路由负载均衡与 MTP 目标，以及建立在 V3-Base 上的 R1 后训练和以 Qwen/Llama 为基座的蒸馏模型。MLA、MoE、RLVR/GRPO 的通用机制由前序章节和第五册、第十六册主讲；本章只保留模型版本、结构归属和报告证据边界。

旧稿有 26 个二级标题，重复铺陈 MLA、MoE、强化学习、serving、通用评估和成本清单；V2、V3、R1 的架构与训练阶段被混写成一套路线，局部示例和一般性结论也没有稳定绑定到报告版本。按现行章节职责重组为六节、十六个小节，保留足以解释版本关系和机制差异的细节，并删去与前序主讲章节重叠的泛化教程。

## 研究依据与编辑决定

- [DeepSeek-V2 技术报告 v5](../../research/deepseek-v2-v3-r1-architecture-training-2026-11-24.md)支持 MLA 的共享低秩 KV latent 与解耦 RoPE key、DeepSeekMoE 的共享/路由专家配置，以及报告模型的规模与评估字段。相关 cache、benchmark 和成本结论保留在作者报告条件内。
- [DeepSeek-V3 技术报告 v2](../../research/deepseek-v2-v3-r1-architecture-training-2026-11-24.md)支持 V2 架构延续、routing bias 参与 top-k 选择而原 affinity 仍用于输出 gate、sequence-wise 辅助平衡、报告设置中的 token dropping 声明及顺序式 MTP。MTP 的训练目标与可选 speculative reuse 分开讲，不把报告消融或提案写成通用性能保证。
- [DeepSeek-R1 论文 v2 与官方仓库](../../research/deepseek-v2-v3-r1-architecture-training-2026-11-24.md)支持 R1-Zero/R1 与 V3-Base 的关系、完整 R1 的阶段路线及 Qwen/Llama 蒸馏模型的学生基座。论文评测与输出问题均归因于其公开模型、任务和评测条件，不推断通用能力或未披露的生产实现。
- DeepSpeed Course 的 MLA 讲解、Hugging Face R1 course、Manning 章节介绍和 Stanford CS336 notes 用于检查 cache、负载均衡及后训练阶段的教学组织；技术事实以固定论文版本为准。研究范围、阅读位置和未解决边界集中记录在对应 `research/` 文件中。

## 正文和同步

- 正文与现行章节大纲采用六节结构：版本边界、V2 架构组合、V3 路由变化、MTP、R1 后训练、R1 蒸馏与证据矩阵。标题保持《第二十一册目录》一致。
- 同步架构时间线中的 V2/V3/R1 版本记录与相关章节段落；新增练习 31BP、面试题 31BQ、`PAPERS.md` 第 55 项和知识图谱谱系；复用术语表已有 MTP 条目并明确其训练目标与可选推理用途。
- 第二十一册架构审计入口已推进到第 58 章；本轮状态为第 1—58 章完成首轮精修，第 59—95 章待处理。

## 验收与遗留事项

- 正文、大纲和目录中的 22 个二级/三级标题名称、层级和顺序匹配；目录、正文、大纲、时间线和研究记录共检查 175 个本地链接实例。正文的 9 个链接、目录的 141 个链接和研究/大纲入口均可解析。
- 整份架构时间线中仍有 3 个未改动的旧失效链接，分别指向 Qwen3-Omni、Qwen3-VL 与 K2 Horizon 的来源笔记；第五十六章审计已将其交由对应专题处理。本章变更的时间线入口可解析。
- 正文不含数学公式、代码围栏或 Mermaid 图，因此没有触发公式/代码/图形渲染门禁。外链终审、目标读者试读和第二十一册连续复读仍属于全册门禁。
- `git diff --check` 通过。未运行测试、代码、模型、API、benchmark 或 PDF 构建；未更新 PDF 或 `plan.md`。
