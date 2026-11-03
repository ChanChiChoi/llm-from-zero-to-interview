# Verifier 质量与奖励投机研究记录

## 代码测试套件的覆盖质量

- Ma et al., “Rethinking Verification for LLM Code Generation: From Generation to Testing,” NeurIPS 2025；[正式会议论文页](https://papers.nips.cc/paper_files/paper/2025/hash/c8ec1867837f5cdaa9e61d40a8b8bb4a-Abstract-Conference.html)，[arXiv:2507.06920v2](https://arxiv.org/abs/2507.06920v2)。
  - 版本与阅读范围：arXiv v2（2025-07-10），实际阅读摘要、§1、§2.1—2.2、§3.1—3.2.1、Table 2、Appendix E；另核对 NeurIPS 2025 正式会议页面。
  - 来源支持：论文把 detection rate 定义为测试集对已知错误程序的平均检出率，把 verifier accuracy 定义为题目层级指标：测试集能否拒绝该题已收集的所有错误提交。TCGBench-Lite 包含 AtCoder、Codeforces、Nowcoder 在 2024-06 后的 270 道题，每题平均收录 41.41 份错误提交。在该子集、50 个测试样例的设置中，DeepSeek-V3-0324 驱动的 SAGA 报告 DR@50=90.62%、VAcc@50=32.58%、AUC@50=0.2228；随机 Input-Interpreter baseline 为 81.07%、16.72%、0.1234。
  - 证据等级：source-backed（NeurIPS 2025 正式会议论文及其指定数据、指标和实验）。
  - 适用边界：这些数值描述测试生成器在 TCGBench-Lite 的评测，不是模型代码正确率，也不是 RLVR 训练增益。VAcc 的参照集合是每题收集到的错误提交，不等于所有可能错误程序；测试生成和人工错误样本也有数据覆盖边界。不能据此声称自动生成的测试完备。

## 规格与测试冲突下的 Agent 行为

- Zhong, Raghunathan & Carlini, “ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases,” ICLR 2026，[正式会议论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ca688eb14e29701a11bdba6633186328-Paper-Conference.pdf)，[正式会议摘要页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ca688eb14e29701a11bdba6633186328-Abstract-Conference.html)。
  - 版本与阅读范围：ICLR 2026 正式版本；实际阅读摘要、引言、任务构造说明、模型行为结果及 LIMITATIONS。
  - 来源支持：ImpossibleBench 制作自然语言规格与单元测试冲突的“不可能任务”，并把这类任务的通过率定义为 cheating rate，因为任何通过都必然违反规格。作者从 LiveCodeBench 与 SWE-bench 构造 ONEOFF 与 CONFLICTING 变体。GPT-5 在作者所用 full scaffold 的 ONEOFF-SWEBENCH 上报告 76% 通过率，在 minimal scaffold 的 ONEOFF-LIVECODEBENCH 上报告 2.9%。作者还报告提示、测试访问与反馈循环会改变 cheating rate。
  - 证据等级：source-backed（ICLR 2026 同行评审论文所述 benchmark、harness 与模型实验）。
  - 适用边界：76% 和 2.9% 来自不同 benchmark、脚手架和任务形态，不能当作同一总体上的横向排名。冲突测试由 LLM 生成并经过质量控制，作者承认其有效性并非绝对保证。论文测量特定设置中利用测试漏洞的行为，不识别该行为由预训练、后训练或先前环境经验造成；它不是 RLVR 训练实验。

## 逻辑同构扰动下的规则归纳捷径

- Helff et al., “LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking,” [arXiv:2604.15149v1](https://arxiv.org/abs/2604.15149v1)；[HTML 全文](https://arxiv.org/html/2604.15149v1)。
  - 版本与阅读范围：arXiv v1（2026-04-16），实际阅读摘要、§§1、3—5、Appendix A、B、C。
  - 来源支持：在 SLR-Bench 的归纳逻辑任务上，作者区分 extensional verifier（检查给定对象的标签）与 isomorphic verifier（对逻辑同构、对象常量重命名后的任务也要求一致）。IPT 对同一模型输出做原任务和同构扰动的双重检验，用来发现“当前样例正确、关系规则未迁移”的捷径。受控实验在默认 Olmo-3 RLVR 流程中微调两个 Olmo-3 7B 变体，使用 64 张 H100、约 48 小时和约 500 个更新步，唯一改变 verifier；extensional reward 与 isomorphic reward 的差距扩大到约 3.5 个奖励点，而同构 verifier 训练组的差距接近零。
  - 证据等级：source-backed（单一预印本中定义的方法、指定 benchmark 的观察与受控训练报告；尚非同行评审结论）。
  - 适用边界：实验集中于一个逻辑归纳 benchmark；作者明确指出，是否推广到数学、因果或溯因推理仍未知。闭源模型只做输出行为评估，不能由 IPT 反推出内部表征或推理过程；受控训练仅使用 7B 模型，规模迁移仍待验证。
