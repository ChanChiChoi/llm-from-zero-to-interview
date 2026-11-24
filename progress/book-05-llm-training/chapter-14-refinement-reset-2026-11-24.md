# 第十四章精修记录（重启流程，2026-11-24）

## 范围与原则

- 按用户本轮要求，从本册既定章节顺序精修；不以已有 `progress/` 记录作为导航或判断依据。
- 本章完整对照契约 `写作规范/全套书不同册的章节大纲结构/book-05-llm-training/14-训练系统综合设计与故障诊断.md`；本章中心保持为“以目标、时间线和证据闭环综合设计与诊断训练系统”。
- 查证本章涉及的数据 mixture、loss spike、per-domain loss 和 contamination 代表研究。证据事实及适用边界见 `research/integrated-training-design-diagnostics-evidence-2026-11-24.md`。

## 本次修改

1. 补入契约要求的“本章定位与中心”标题，使正文的 H2/H3 结构完整对应契约。
2. 收紧 Adam loss spike 的研究适用范围：明确 Bai 等论文的 187M LLaMA/100B SlimPajama 实验验证了降低 β₂ 的作用；增大 ε 的直接对照来自另一组函数逼近实验，不再混写为同一语言模型设置的证据。
3. 改进第 15 节审计账本：把 DP=16、TP=4、PP=2 明确列出，要求其乘积与 128 张 GPU 相等；解释只有 DP 维度参与 global sample batch。将示例的 `valid_token_fraction` 移入数据账本并加范围校验，不再硬编码在批次计算语句中。
4. 更新代码样例的静态审阅日期。代码没有执行；手算预期输出增加 `parallel_world_size=128`。

## 研究依据与边界

- DoReMi 的论文报告了 280M reference/proxy 到 8B 模型的实验、6.5 个百分点平均准确率增益、75k 对 200k steps，以及约 8% 主模型 FLOPs 的权重搜索成本；这些数字限于论文的 The Pile/GLaM 设置，不作为通用预期。
- PALOMA 的 546 个英文与代码域及其污染、训练顺序、分层抽样、词表和评测格式控制说明了为何应分域报告 LM fit；这些指标不能替代下游能力评测。
- PaCoST 的配对 P(True) 与显著性检验被描述为 benchmark 级潜在行为信号，不等价于训练数据来源证明或分数膨胀估计。
- 2026 年有限精度论文的 28M、16B-token 结果只支持在该研究设置中把 LM-head 量化误差作为可检验机制；不据此推断一般大模型 spike 的发生率或指定默认修复。

## 验收

- 正文 H2/H3 契约标题：契约 83 项，正文 83 项，缺失 0、额外 0。
- Markdown 链接：50 个；本地目标缺失 0。
- 围栏：`~~~` 50 个、反引号围栏 6 个，全部成对。
- `git diff --check -- 'book-05-llm-training/chapters/14-训练系统综合设计与故障诊断.md'` 通过；研究记录无尾随空白。
- 只做静态检查。未运行正文代码、训练 demo、benchmark、测试套件或 PDF 构建。

## 交接

第十四章正文、研究依据和本次精修记录分别写在 `book-05-llm-training/chapters/14-训练系统综合设计与故障诊断.md`、`research/integrated-training-design-diagnostics-evidence-2026-11-24.md` 和本记录中。继续本次任务时以本章正文和规范为准，不回读旧进度记录；本书的整体完成状态应由本轮实际精修范围核对后再汇总。
