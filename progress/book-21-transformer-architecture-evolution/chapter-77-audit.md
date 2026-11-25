# 第七十七章精修审计

日期：2026-11-25。范围：第 77 章《DeepSeek-V4：百万上下文配置与推理实现》的章节结构复核、正文主题思想不变重写及本地导航/配套同步；不处理 PDF。

## 旧稿诊断与写作决定

旧稿重复讲解 CSA/HCA 的通用压缩与 query 读取机制，与第 74 章完整主讲范围重叠；标量玩具公式和简化代码没有把 V4 的逐层配置、缓存状态或公开推理路径连成可核对的解释。旧稿后段混入 Pro API、reasoning effort、Agent harness 与 DataCurve 结果，混淆模型架构、后续 checkpoint、服务接口和外部评测证据。正文没有对应的完整 02a 章节大纲，且不同版本和测试条件未形成明确边界。

本轮保留第 77 章文件路径和主题职责，以 DeepSeek-V4 技术报告 v1 的 preview Pro/Flash 为模型层配置、Table 7 和资源估算对象；使用固定 Hugging Face revision 核对 Pro 配置与参考代码；Pro-0813、V4.1-Flash 和 SGLang 指定 workload 案例分别标明版本或测量条件。第 74 章继续主讲 CSA/HCA 机制，第 78 章主讲 mHC，第 21、58 章分别承接 MoE 与 V2/V3/R1 机制，第 6、24 册承接通用 serving/cache/推理引擎内容。

## 本轮改动

- 新建[第 77 章完整章节大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/77-DeepSeek-V4百万上下文配置与推理实现.md)，按来源版本、逐层配置、百万上下文评测、资源基线、异构 cache、固定实现/框架证据和证据分层组织为 7 个二级标题、20 个三级标题。
- 重写[第 77 章正文](../../book-21-transformer-architecture-evolution/chapters/77-csa-hca从压缩kv到百万上下文.md)，删除重复的通用机制讲解、非参考实现的玩具代码以及与本章架构责任不符的 API/harness/DataCurve 展开；加入模型配置表、候选条目估算、报告指标和缓存更新/恢复边界，并分开报告估算、固定源码与框架性能数字。
- 同步第二十一册读者目录和设计目录。两份目录链接同一章正文与详细大纲，标题层级和章节正文保持一致。
- 在相关 V4 Pro、SGLang 练习段落加入本章/第 74 章入口；在知识图谱补充第 74、77、78 章各自的职责链。`PAPERS.md` 第 71 项已说明第 77 章负责具体配置、效率数值和推理实现，故保留现有条目。
- 新增[研究记录](../../research/deepseek-v4-million-context-implementation-2026-11-25.md)，记录报告、模型卡/config revision、固定源码、课程/博客以及 SGLang release/PR 的主张范围和未复现边界。

## 静态验收与遗留门禁

- 正文、大纲、读者目录和设计目录中 7 个二级标题、20 个三级标题的名称、层级和顺序一致。
- 正文共 25 个 Markdown 链接目标，均存在；第二十一册读者目录、设计目录和 `EXERCISES.md` 的本地链接扫描均无失效目标。
- 正文与本轮新增练习中的行内数学定界符均按行成对；正文没有块级 `math` 公式，TeX 命令检查也未发现 `operatorname`。模型层计数、压缩候选规模、Table 7 值、两种资源 baseline 和 cache 对齐关系已与研究来源逐项对照。
- `git diff --check` 通过。
- GitHub MathJax 公式页面未实际渲染，故正文与练习的公式视觉门禁仍待验收；外链最终逐条抽查、目标读者试读及第二十一册按读者路线连续复读属于全册门禁。未运行程序、测试、模型、kernel 或 benchmark，未构建或更新 PDF，未修改 `plan.md`。

下一章为第 78 章《mHC：用双随机矩阵约束残差流》。第二十一册第 78—95 章仍待首轮精修，全套书精修 goal 继续 active。
