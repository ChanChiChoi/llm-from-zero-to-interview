# 第四册第十五份正文《方法比较与工程取舍》整章验收研究记录（2026-10-23）

对象：`book-04-llm-encyclopedia/chapters/15-面试高频对比表.md`（15.1—15.43，795 行）。

## 结构与格式

- 全章围栏成对闭合（34 math、1 python、text/表格若干）；math 围栏花括号平衡。
- 修正 20 处 `\operatorname{...}` 为 `\mathrm{...}`（LN、RMSNorm、Norm、ReLU、SwiGLU/Swish、FNR_AI、KVBytes、ContextCost、Accuracy、pass@k、UnsafePassRate、FalseRefusalRate、softmax、TopK 等）；修正后无 `\operatorname`、无 `$$`。
- 两张 Markdown 表（LoRA/Adapter/Prefix、KV cache/Activation checkpoint）表头与分隔行格式正确；15.42 的比较表模板列对齐。

## 来源核验

- 15.43 的 20 个 arXiv 链接经指定代理全部 HTTP 200；其中 AdamW 1711.05101、LayerNorm 1607.06450、RMSNorm 1910.07467 三篇为新题名核对，均与链接文字一致；其余 17 篇沿用第 14 章及此前章节的题名证据。

## 代码与数值核验

- 1 个 Python 块（Adam vs AdamW 最小更新、非法学习率拒绝）独立执行通过；实现省略 bias correction 与正文声明一致。
- 数值复算：AdamW 解耦衰减因子 (1-ηλ)=0.99 与公式一致；pass@k 无偏估计例（n=20,c=4,k=5）=0.718，公式与定义域（n>0、0≤k≤n）核对通过；KV cache 2BNLH_kv·d_h·b 与 15.24 的 KVBytes 公式量纲一致。
- 公式边界：GainPerCost 类比较要求 ΔQ 与 ΔC/ΔT 同报；FNR_AI 分母 TP+FN；FalseRefusalRate 分母 TN+FP，与 15.38 的两侧错误语义一致；robustness Δ 定义要求声明哪些字段应不变。

## 配套同步

- 15.43 入口与 `PAPERS.md` 各区重叠良好，无需新增；`GLOSSARY_EN_ZH.md` 已有 LN/RMSNorm/GQA/LoRA/pass@k 等条目；`INTERVIEW_BANK.md` 已覆盖 Adam/AdamW、BF16、MoE、DPO/RLHF 等高频比较。
- 第四册目录第 15 章映射正确。

## 未验证项

- 未运行真实优化器、量化、蒸馏、检索或解码实验；比较关系为机制与账本层面的教学陈述，正文已声明结论依赖 workload 与测量口径。
- PDF 构建受环境缺少 Pandoc 限制，未执行。
