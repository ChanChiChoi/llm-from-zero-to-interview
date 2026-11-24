# 第六十三章精修审计：Gated MLA

日期：2026-11-24

## 旧稿诊断

旧稿有 34 个二级标题，经历多轮追加后重复讨论 latent 维度、信息瓶颈、读放大、引用回退、量化、版本化和 speculative 状态事务。它把通用 cache/serving 工程清单放进架构机制章，与第 6、24、70 章及其他册职责重叠。更重要的是，旧稿没有基于 Kimi K3 原始资料定义 Gated MLA，把门控写成显式 KV 与 latent KV 间可按 token、head 或 layer 选择的路由/融合，并将 gate 解释为证据选择器；Kimi K3 v2 报告定义的是在 MLA 全局 attention 输出后施加输入依赖的逐通道 full-rank gate。

## 编辑决定

- 将 Gated MLA 限定为 MLA 的 token 级 KV latent、全局 attention 读取和 output gate 三个连续步骤；避免把它说成与显式 KV 动态混合的两路模块。
- 用 DeepSeek-V2 的共享 KV latent、上投影和 decode-time projection absorption 说明 MLA 基础；分别说明 decoupled RoPE 的位置 cache 分量与 Kimi K3 NoPE 的固定版本设置。
- 按 Kimi K3 v2 Eq. 7 写出 gate 位于 attention 聚合之后、输出投影之前的公式，区分 channel modulation、缓存压缩、attention probability 和 KDA 的递归状态门。
- 保留逐 token ideal cache payload 公式与标明为教学配置的数字账本，明确它不计真实页元数据、临时工作区、cache dtype、KDA 状态或运行时布局；不从理论 payload 推断端到端收益。
- 将通用 latent cache 的保真、质量切片和读取开销回链第 70 章；NoPE 性质交第 64 章；KDA/MLA 组合交第 65 章；Kimi K3 全模型案例交第 88 章。删除泛化的 cache manifest、snapshot/restore、tenant isolation、fallback、RAG 与 speculative rollback 方案。

## 证据与同步

依据 DeepSeek-V2 v5 §2.1.2–2.1.3、Kimi K3 v2 §2.1–2.1.2/Eq. 7 和 Moonshot 官方固定 HF config revision 核对 MLA latent、projection absorption、decoupled RoPE、NoPE、output gate 与版本参数。证据版本、定位和边界见[研究记录](../../research/kimi-gated-mla-evidence-2026-11-24.md)。

新增完整章节结构大纲，同步正文、第二十一册 reader TOC 与设计 TOC、练习 31BV、面试题 31BW、`PAPERS.md` 第 60 项和 `KNOWLEDGE_GRAPH.md`。术语表已有 Gated MLA 条目，没有重复添加。

## 静态验收与遗留门禁

- 正文、大纲及两份目录的 7 个二级标题和 8 个三级标题名称、层级、顺序匹配。
- 手工核对输出 gate 作用位置、MLA cache 的 latent/位置分支维度、NoPE 的版本边界及 Kimi K3 跨 MLA 层的 per-token latent rank 估算；练习/面试题编号和论文索引编号唯一。
- 新增本地链接目标存在，`git diff --check` 通过。未运行测试、构建、模型、API、benchmark 或 PDF 构建，未更新 PDF/`plan.md`。
- 公式源码与代数含义已静态检查；GitHub MathJax 页面实际渲染、目标读者试读、外链终审及第二十一册连续复读仍待完成。
