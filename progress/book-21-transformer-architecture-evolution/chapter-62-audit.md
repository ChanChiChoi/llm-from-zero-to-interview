# 第六十二章精修审计：Gated DeltaNet

日期：2026-11-24

## 旧稿诊断

旧稿共有 30 个二级标题，经过追加后把 DeltaNet、Gated DeltaNet、KDA、chunkwise kernel、serving 状态生命周期和通用部署检查混为一章。状态矩阵的左右乘方向、衰减和残差写入的先后顺序缺少统一约定；若干内容把 output gate、遗忘门和 delta 写入门放在同一层比较，读者难以判断每个变量改变的是哪条信息路径。缓存隔离、租户边界、speculative rollback 等系统清单没有为本章架构主线提供相应读者结果，并与第六、二十四册及本册其他章节职责重叠。

## 编辑决定

- 固定正文记号为 key-by-value 状态 $A_t\in\mathbb{R}^{d_k\times d_v}$，由原论文 $S_t$ 转置得到；先定义预测和 Delta rule，再从衰减后的状态推导 Gated Delta Rule，避免公式左右方向变化。
- 以“全局状态衰减、沿当前 key 定向修正、query 读取”组织机制解释。分开说明 $\alpha_t$、$\beta_t$ 与模块末端 output gate 的职责，并用二维轨迹展示先衰减、再预测、求残差和写入的顺序。
- 将 chunkwise WY/UT 限定为递归转移的块内计算视图：说明块入口状态、块内因果计算与边界传递，不把并行计算误说成 full attention，也不从渐近复杂度推出通用吞吐优势。
- 只用有出处的 GDN 和 GDN-2 匹配实验作条件化例证；按公式区分 GDN、KDA 与 GDN-2 的 decay/erase/write 参数轴，避免把后续特例当成通用实现定义。
- 删除与读者结果无关的通用 serving 生命周期、cache manifest、租户隔离和 speculative rollback 清单；保留精确回读、状态干扰和有限容量作为架构边界，并将混合路线细节交给第 65 章。

## 证据与同步

依据 Gated Delta Networks v3 与 Gated DeltaNet-2 v2 原论文核对递归状态、残差更新、chunkwise WY/UT、参数约化关系及作者报告的实验设置。Erik Steiger 的 chunkwise 推导和 Brandon Behring 的 delta-rule 笔记用于核对教学解释；研究记录列出版本、阅读位置、支持范围和未验证边界：[`gated-delta-net-evidence-2026-11-24.md`](../../research/gated-delta-net-evidence-2026-11-24.md)。

新增完整章节结构大纲，同步正文、第二十一册 reader TOC 与设计 TOC、练习 31BU、面试题 31BV、`PAPERS.md` 第 59 项和 `KNOWLEDGE_GRAPH.md`。术语表已有 Gated DeltaNet、Delta Rule 与 GDN-2 条目，无需重复添加。

## 静态验收与遗留门禁

- 正文、大纲及两份目录的 7 个二级标题和 9 个三级标题名称、层级、顺序匹配。
- 手工核对矩阵方向、预测残差梯度、衰减后更新顺序、示例数值与 GDN-2 的 erase/write 维度；练习/面试题编号唯一，新增本地链接目标存在。
- `git diff --check` 通过；未运行测试、构建、模型、API 或 benchmark，未更新 PDF 或 `plan.md`。
- 正文公式源码与代数含义已静态检查；GitHub MathJax 页面实际渲染、目标读者试读、外链终审和本册连续复读仍待完成。
