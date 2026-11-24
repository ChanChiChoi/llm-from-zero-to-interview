# 第四十四章精修审计：Attention + Convolution + SSM 的模块分工

## 范围与诊断

完整复读旧稿及其章节标题。旧稿有 29 个二级标题：开头用“卷积局部、SSM 长程、Attention 全局”固定模块职责，后续再重复信息访问、查询分配和消融框架；同时混入模型缓存协议、状态版本化、rollback、Serving 资源画像、p99、生产回归和外部检索路径。部分工程与性能说法没有绑定到对应模型或运行设置，玩具 Python 代码也不承担本章读者结果。第四十一章已负责卷积结构与实例，第四十二章负责跨架构比较，第四十三章详讲 Attention–SSM 融合，第四十五章负责外部检索。本章收束为三类信息路径的机制比较、组合位置和条件化判断。

## 结构与正文决定

保留章节标题、编号和正文文件路径；经 02a 设计后重组为六节：

1. 以历史表示和读取方式取代固定的“局部—长程—全局”职责口诀。
2. 区分短因果卷积与长卷积，并以 Qwen3 和 H3+LongConvs 说明放置位置。
3. 说明 SSM 的递归状态、固定 LTI 与 Mamba 输入选择性之间的差别。
4. 说明 Attention 按 query 读取 mask 可见历史的方式及其边界。
5. 比较卷积修正、长卷积替换和 Attention–SSM 融合等不同组合位置。
6. 将任务信息需求转成可检验假设，用受控消融与任务切片评价路径贡献。

正文以一张信息路径对照表和一张任务假设表呈现比较维度；删除旧稿的泛化融合公式、无依据的 80/15/5 数字示例、玩具 Python ablation、cache/state envelope、rollback、p99 调度和生产验收清单。外部检索只作为边界说明并交接至第四十五章。

## 研究与证据

新发现及来源阅读位置记于[Attention、卷积与 SSM 模块路径研究记录](../../research/attention-convolution-ssm-module-roles-2026-11-23.md)。核对 Vaswani 等人的 Attention 定义、Bai 等人的 causal TCN/dilation、Mamba 原论文的输入依赖选择，以及 Qwen3 Conv1D 位置消融和 ICML 2023 LongConv/H3 架构。

Qwen3 报告的 P5 结果限于该论文的 Qwen3 block 位置消融；作者提出的“为 Q/K/V 加入局部上下文”没有被单独隔离证明。LongConv 语言建模沿 H3 设置保留两层 Attention、以长卷积替换 SSM 核；结果不是纯卷积主干胜过 Transformer 的证明。未运行本地代码、模型、API 或 benchmark。

## 同步与静态验收

- 已建立[第四十四章有效大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/44-Attention-卷积-SSM模块分工.md)，并将六个标题同步至第二十一册[目录](../../book-21-transformer-architecture-evolution/目录.md)。
- 更新 PAPERS.md 与 KNOWLEDGE_GRAPH.md；新增练习 31AZ 和面试题 31BA。已有术语足以表达本章内容，无需重复新增术语表条目。
- 正文、大纲与目录六个章节标题及顺序一致；本地链接目标检查无缺失；题号 31AZ、31BA 各出现一次；`git diff --check` 通过。
- 正文仅有一个行内变量 `$t$`，表示因果序列中的当前位置；源码定界符成对，未引入块公式或代码。GitHub MathJax 实际页面渲染仍待确认，公式门禁未关闭。
- 未运行代码、测试、模型、API、kernel、benchmark 或 PDF 构建。目标读者试读、外链逐项终审与第二十一册连续复读留待全册门禁。

本章首轮精修完成，不代表第二十一册或全套书精修完成。下一入口为第四十五章《Retrieval-Augmented Architecture：把外部检索变成架构组件》。全套书精修 goal 继续 active；不更新 PDF 或 `plan.md`。
