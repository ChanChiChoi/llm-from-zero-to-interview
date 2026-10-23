# 第四册第十四份正文《重要论文索引》整章验收研究记录（2026-10-23）

对象：`book-04-llm-encyclopedia/chapters/14-重要论文索引.md`（14.1—14.45，831 行）。

## 结构与格式

- 全章围栏成对闭合（30 math、1 python、text 若干）；math 围栏花括号平衡。
- 修正 13 处 `\operatorname{...}` 为 `\mathrm{...}`（GainPerCost、Attention、softmax、KVBytes、TopK、Bytes、Select/Expand、CE、ASR、Success）；修正后无 `\operatorname`、无 `$$`。

## 来源核验（本章为论文索引章，重点核对 14.45）

- 全章 51 个 Markdown 引用经指定代理逐条核验：49 个唯一 arXiv ID/官方入口题名与链接文字一致；GPT-1 PDF 与 GSM8K 仓库 HTTP 200。
- 发现并修正 1 处题名错配：`2211.00593` 实为 "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small"，正文误标为 "A Mathematical Framework for Transformer Circuits"。修正为两条并列条目：2211.00593 用实际题名；"A Mathematical Framework..."（Elhage et al., 2021）未上 arXiv，改指 transformer-circuits.pub 官方入口（HTTP 200）。
- 此前章节已核验的重复 ID（2001.08361、2203.15556、1910.02054、2005.11401、ReAct/Toolformer/CoT/Self-Consistency/Verifiers/ToT/SWE-bench/ROME/MEMIT/Constitutional AI 等）沿用前轮证据。
- 14.6/14.7 引用 Kaplan 2020 与 Chinchilla 2022 的 ID 与题名核对一致。

## 代码与数值核验

- 1 个 Python 块（证据账本：gain、gain_per_cost、零成本拒绝）独立执行通过。
- 数值复算：LoRA 训练参数 r(din+dout)=65536 对 full 4096²=16777216，与正文一致；ViT 224/16 patch 数=196 与公式一致；GainPerCost 仅在 ΔC>0 定义的限定有正文与代码双重声明。
- 公式边界：Kaplan 分离项注明"只是教学近似、非唯一拟合形式"；GRPO 相对优势在 s=0 时定义为 0 并拒绝加常数；ASR 分母限定"有效攻击样本总数"；对比损失/扩散损失/ALiBi/Mamba 等公式均带变量说明与范围声明。

## 配套同步

- 本章本身即论文索引，条目与 `PAPERS.md` 各区互补，无需新增；前轮已为 ch10—ch13 补齐对应论文直链。
- 第四册目录第 14 章映射正确；正文除论文链接外无相对链接。

## 未验证项

- 题名与 abs 页面核对不等于全文与实验复核；transformer-circuits.pub 条目为官方入口而非 arXiv，正文 14.45 已声明"链接用于定位来源，不表示结果可无条件外推"。
- PDF 构建受环境缺少 Pandoc 限制，未执行。
