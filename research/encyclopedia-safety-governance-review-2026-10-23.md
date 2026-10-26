# 第四册第十三份正文《安全与治理》整章验收研究记录（2026-10-23）

对象：`book-04-llm-encyclopedia/chapters/13-安全与治理.md`（13.1—13.27，1,181 行）。

## 结构与格式

- 全章围栏成对闭合（40 math、3 python、text 若干）；math 围栏花括号与 begin/end 环境平衡。
- 修正 25 处 `\operatorname{...}` 为 `\mathrm{...}`（Risk、corr、FNR、ASR、UnsafePassRate、FalseRefusalRate、RedactionRecall、MinimizationRate、SWR、Route、Tradeoff、ReLU、Allow/Auth/Scope/Policy、TrustedArtifact 及其合取项、Gap 等）；修正后无 `\operatorname`、无 `$$`。

## 来源核验

- 18 个唯一外链（11 篇 arXiv、MITRE ATLAS、NIST SSDF/AI RMF、OWASP x2、SLSA、C2PA）经指定代理核验均 HTTP 200。
- 11 篇 arXiv 题名经 abs 页面核对全部与引用主题一致：Concrete Problems 1606.06565、DP-SGD 1607.00133、Membership Inference 1610.05820、Model Cards 1810.03993、Risks from Learned Optimization 1906.01820、StereoSet 2004.09456、RealToxicityPrompts 2009.11462、Extracting Training Data 2012.07805、ROME 2202.05262、MEMIT 2210.07229、Constitutional AI 2212.08073。
- 章首两条书内相对链接（第八册目录、第二十三册安全治理章）目标存在。

## 代码与数值核验

- 3 个 Python 块逐块独立执行通过（Hazard 风险排序与非法等级拒绝、taint 标签传播与外发阻断、activation patching 线性干预）。
- Wilson 区间、DP-SGD 裁剪/噪声、z 分数水印检测、CTX 公式的定义域（n>0、权重非负、分母大于零）均有正文限定并核对。

## 配套同步

- `PAPERS.md` Safety 与 Interpretability 区新增对齐/隐私/编辑/披露/偏见毒性共 11 个论文直链与证据边界说明。
- 术语表已有 Prompt Injection、Jailbreak、DP、Unlearning、Guardrail 等主题；练习与题库已覆盖安全评估主题。
- 第四册目录第 13 章映射正确。

## 未验证项

- 未运行真实模型攻击、红队、membership inference、DP-SGD 训练、SAE/ROME/MEMIT 编辑或生产 guardrail；教学代码只验证排序、标签传播和线性干预逻辑。
- 论文题名核对不等于全文与实验复核；水印 z 分数为教学简化，正文已声明阈值与攻击者知识影响；PDF 仍受 Pandoc 缺失限制。

## 本轮最终精修收口（2026-10-26）

- 数据最小化率补充实际字段总量必须大于零，并要求加权字段量绑定同一字段清单和权重版本。
- 重新执行正文 3 个 Python 块，3/3 通过；40 个 math 围栏、90 个 fenced 标记、18 个唯一外链、2 个书内相对链接和目录映射检查通过，`git diff --check` 通过。
- 本章声明范围内精修验收完成；真实红队、DP-SGD、模型编辑和生产 guardrail 仍属于未验证项。
