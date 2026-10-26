# 第四册第十一份正文《长上下文、RAG 与 Agent》整章验收研究记录（2026-10-23）

对象：`book-04-llm-encyclopedia/chapters/11-长上下文rag与agent.md`（11.1—11.25，961 行）。

## 结构与格式

- 全章 64 个 `~~~` 围栏成对闭合（23 math、5 python、其余 text）；math 围栏花括号平衡；无 `\operatorname`、无 `$$`、无尖括号下标。
- 工具调用 Precision/Recall、SupportRate、Pass@k、RRF、退避、成本/延迟分解等公式均带分母非空、定义域和"不适用"语义限定。

## 来源核验

- 12 个唯一外链经指定代理核验，抽查 10 个（含全部 6 篇 arXiv、MCP/A2A 规范、Anthropic、HELM）均 HTTP 200。
- 6 篇 arXiv 题名经 abs 页面核对，全部与引用主题一致：DPR 2004.04906、RAG 2005.11401、ReAct 2210.03629、Toolformer 2302.04761、Lost in the Middle 2307.03172、SWE-bench 2310.06770。
- MCP 规范固定 2025-06-18 版本，A2A 固定 latest 入口并在正文要求实现绑定版本。

## 代码与数值核验

- 5 个 Python 块逐块独立执行，5/5 通过（needle 位置实验、RRF 融合、工具参数校验、退避序列、路径包含检查）。
- 数值复算：RRF 例（sparse [A,B,C]、dense [B,D,A]、k=60）排名 B > A > D > C，与正文"B 领先"一致；退避序列 [1,2,4,4,4] 与 budget 拒绝一致。
- 边界覆盖：RRF 拒绝重复 ID 与非法常数、needle 拒绝 NaN/空串/bool、工具参数拒绝缺失/未知字段/非法 JSON、退避超预算与非有限延迟拒绝、路径穿越拒绝。

## 配套同步

- `PAPERS.md` Agent 区补入 ReAct、Toolformer 直链与 SWE-bench 入口；DPR/RAG/Lost in the Middle 已有条目不重复。
- `GLOSSARY_EN_ZH.md` 已有 RRF、Recall@k、MRR、提示注入、幂等等条目（35 处命中）；`EXERCISES.md` 已覆盖 RAG 评估、工具 schema、Agent 安全等主题。
- 第四册目录第 11 章映射正确；正文无相对链接。

## 未验证项

- 未运行真实检索器、embedding 模型、MCP/A2A Server、浏览器/代码执行沙箱或生产 Agent；未复现 DPR/RAG/ReAct/SWE-bench 实验。
- `u(p)` 位置敏感性和 `P(success)` 层级分解为教学模型，正文已声明不是通用标准定义或独立概率。
- PDF 构建受环境缺少 Pandoc 限制，未执行。

## 本轮重新精修检查点（2026-10-26）

- 重新读取 11.1—11.25 后发现退避公式的上限口径问题：原式 `min(t_max, base)+jitter` 可能超过 `t_max`；正文已改为 `min(t_max, base+jitter)`，并将 `t_max` 明确为包含抖动的最终上限。教学代码仍只验证无抖动基线 `[1,2,4,4,4]`。
- 从正文直接提取 5 个 Python 块并逐块执行，5/5 通过；23 个 math 围栏花括号平衡，外链和相对链接结构检查通过，`git diff --check` 通过。
- 本检查点不代表整章验收；全章来源版本、逐节口径、真实检索/工具/Agent 环境和 PDF 仍未验证。
