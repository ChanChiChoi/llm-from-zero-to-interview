# 第四十三章精修审计：Attention + SSM 混合架构为什么现实

## 范围与诊断

完整复读旧稿及第三十六、四十、四十二、四十四、四十五章的关联内容。旧稿有 27 个 H2，混合了早期 Mega/GSS/H3 机制复述、混合层玩具代码、训练目标与路由检查、KV/state 缓存服务协议、回退和生产验收清单；多个部分重复同一结论，且容易把一般工程建议写成已验证的架构规律。第 40 章已负责早期混合算子，第 42 章已完成跨架构比较，第 44 章负责 Attention/卷积/SSM 职责分工，第 45 章负责外部检索。本章应收束为 Attention–SSM 融合粒度、代表性配置与证据边界。

## 结构与正文决定

保留章名、文件路径和编号，经 02a 研究后重组为六节：

1. Attention 显式 key/value 历史查询与 SSM 递归状态摘要的互补路径。
2. 层间串接：Jamba 和 Mamba-2-Hybrid 的配置与额外设计变量。
3. 层内并行 head：Hymba 的融合方式及共同设计。
4. 比例与位置作为有条件的消融变量。
5. 规模、任务与测量口径对结果归因的限制。
6. 混合结构的适用判断，以及与第 44、45 章的交接。

删去未由对应论文支持的通用路由训练/塌缩推演、混合代码、状态同步或回滚步骤、生产级验收和面试清单式重复段落。正文含行内数量、比例和单位公式，无块公式或可执行代码；保留一个示意层间与层内融合位置的 Mermaid 图。

## 研究与证据

新知识和来源阅读位置记于[Attention–SSM 混合架构研究记录](../../research/hybrid-attention-ssm-architectures-2026-11-23.md)。依据包括 Jamba ICLR 2025 正式论文、Waleffe 等人的 arXiv v1、Hymba ICLR 2025 正式论文、Bae 等人的 arXiv v3、Lee 等人的 arXiv v1，以及 Mamba-2 作者团队系统博客。

正文限定 Jamba 的 1:3/1:7 结果于 1.3B、250B-token 无 MoE 消融；限定 Mamba-2-Hybrid 的 +2.65 分于约 8B、3.5T-token 和 12 项任务；将最高约 8 倍长上下文生成加速标为 batch 32 设置下的作者预测；保留 16K/32K 长上下文任务的相对表现变化。Hymba 的整体结果不归因于 hybrid-head 单项；Bae 的比例/位置经验限于其约 350M/1B、60B-token 实验；Lee 的观察标为特定设置下的预印本结果。未运行本地模型或 benchmark。

## 同步与静态验收

- 已建立[第 43 章有效大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/43-Attention与SSM混合架构.md)，并将六个标题同步到第二十一册[目录](../../book-21-transformer-architecture-evolution/目录.md)。
- 已将融合证据链加入 KNOWLEDGE_GRAPH.md，在 PAPERS.md 新增混合架构来源入口，新增练习 31AX 和面试题 31AY；未重复添加已有 Hybrid Architecture 术语。
- 正文、大纲和目录六个标题已逐项核对且顺序一致；本地链接目标检查 7 项，缺失 0；练习 31AX 和面试题 31AY 各出现一次。新 Mermaid 图的 GitHub 页面渲染尚待确认。尚未进行目标读者试读、外链最终逐条审查或第二十一册连续复读。
- 行内公式的美元定界符成对，数量、单位和比例的 TeX 源码已静态核对；GitHub MathJax 实际渲染尚待确认。正文无块公式和可执行代码；git diff --check 通过。

本章首轮精修完成，不代表第二十一册或全套书精修完成。下一入口为第 44 章。全套书精修 goal 继续 active；不更新 PDF 或 plan.md。
