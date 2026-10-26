# 第四册第十二份正文《Reasoning 与评估》整章验收研究记录（2026-10-23）

对象：`book-04-llm-encyclopedia/chapters/12-reasoning与评估.md`（12.1—12.17，700 行）。

## 结构与格式

- 全章围栏成对闭合（35 math、2 python、text 若干）；math 围栏花括号平衡。
- 修正 13 处 `\operatorname{...}` 为 `\mathrm{...}`（arg\,max、ECE、Acc、Conf、BS、TopK、UCT、pass@k、CI），符合 WRITING_SPEC 禁用项要求；修正后无 `\operatorname`、无 `$$`。

## 来源核验

- 11 个唯一外链（9 篇 arXiv、HELM、lm-evaluation-harness）经指定代理核验均 HTTP 200。
- 9 篇 arXiv 题名经 abs 页面核对，全部与引用主题一致：CoT 2201.11903、Self-Consistency 2203.11171、Training Verifiers 2110.14168、Let's Verify Step by Step 2305.20050、Tree of Thoughts 2305.10601、HumanEval 2107.03374、Calibration 1706.04599、MT-Bench 2306.05685、SWE-bench 2310.06770。

## 代码与数值核验

- 2 个 Python 块逐块独立执行通过（多数投票、配对 lift + bootstrap CI）；配对 lift=0.2 且区间包含点估计、非配对输入与非法 rounds 拒绝断言通过。
- 数值复算：多数投票 ("42",2) 一致；pass@k 无偏估计例（n=10,c=5,k=2）=0.778，公式结构正确。

## 配套同步

- `PAPERS.md` Reasoning 区已完整覆盖 CoT、Zero-shot、忠实性、Self-Consistency、Verifier、PRM800K、ToT/LATS、TTC Scaling、CodeRL、Self-Debug 等入口，无需新增。
- `GLOSSARY_EN_ZH.md` 已有 ECE、CoT 系列、PRM 等条目；`EXERCISES.md`/`INTERVIEW_BANK.md` 已覆盖污染、judge 偏差、pass@k 等主题。
- 第四册目录第 12 章映射正确。

## 未验证项

- 未运行真实模型、采样器、verifier、MCTS/ToT 搜索或完整评估 harness；bootstrap 实现为小样本教学版本，正文已声明分位点与种子处理可变。
- 论文题名核对不等于全文与实验复核；PDF 构建受 Pandoc 缺失限制。

## 本轮重新精修检查点（2026-10-26）

- 重新读取 12.1—12.17 后补强 paired bootstrap 示例的输入边界：`_validate_pairs` 现在拒绝字符串和不可迭代对象，统一转换成序列后再检查非空、等长及 0/1 标签，并增加 `None` 输入拒绝断言。
- 从正文直接提取 2 个 Python 块并逐块执行，2/2 通过；35 个 math 围栏花括号平衡，78 个 fenced 标记成对，11 个外链唯一且无相对链接，目录映射和 `git diff --check` 通过。
- 本检查点不代表整章验收；真实模型、verifier、搜索、代码沙箱、benchmark 和全部来源版本仍未验证。

## 本轮最终精修收口（2026-10-26）

- 过程监督公式补充 `a_ij` 为二值自动覆盖标记、`c_ij` 为有限非负人工成本，并说明零步骤样本不进入该分母。
- 重新执行正文 2 个 Python 块，2/2 通过；35 个 math 围栏、78 个 fenced 标记、11 个唯一外链和目录映射检查通过，`git diff --check` 通过。
- 本章声明范围内精修验收完成；真实模型/搜索/benchmark 及动态来源版本仍属于未验证项。
