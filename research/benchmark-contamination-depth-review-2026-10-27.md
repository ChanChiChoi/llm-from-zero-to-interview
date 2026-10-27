# 第七册第五章：污染检测与可信评估资料驱动精修记录

## 范围与保留的知识点

本轮依据当前 [`WRITING_SPEC.md`](../WRITING_SPEC.md)，复读并精修
[`05-污染检测与可信评估.md`](../book-07-evaluation-experiments/chapters/05-污染检测与可信评估.md)
§1—§14。保留原有数据流风险、数据血缘、污染类别、exact/near duplicate、
MinHash/LSH、SimHash、embedding、代码和数学近重复、人工阈值校准、时间切分、
canary、行为异常、扰动测试、私有 holdout、多源评估、任务类型风险、报告字段、
诊断 demo 和证据边界；本轮补方法机制和证据分层，没有删除原有核心主题。

## 研究问题

1. 数据源出现相同文本，何时能证明目标模型实际接触了它？
2. 如何区分数据重合、模型题面记忆和公开分数不能泛化？
3. 黑盒、训练语料可访问、仅有模型输出等条件下，哪些检测路线可用？
4. 代表性方法的机制、成立假设、证据强度和失败条件分别是什么？
5. 动态/新鲜 benchmark 能降低哪些风险，又如何维护跨版本可比性？

## 主要研究来源与证据范围

| 来源 | 解决的问题与机制 | 可支持的结论 | 边界和限制 |
| --- | --- | --- | --- |
| [Oren et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/46e624c244cff669223d488defd4e835-Abstract-Conference.html) | 对比 benchmark 公开顺序与随机排列的序列 log-likelihood；利用无污染时的可交换性，按 shard 进行统计检验 | 在可交换性前提下，对某些被训练文本顺序记忆留下的痕迹提供黑盒假阳性控制；无需权重或训练语料 | 需要序列 log-prob API；检测目标窄，重排/改写/部分信息暴露可能没有顺序信号；未检出不是无污染证明。论文审计的若干公开模型结果只绑定方法功效和对象 |
| [ConStat, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a7f89793b9e6f8c6568dbbb6ff727b9b-Abstract-Conference.html) | 将污染定义为不泛化的异常 benchmark 优势；以参考模型分数分布校正 benchmark 难度，用 bootstrap 估计效应和显著性 | 可量化目标模型在原集分数相对重述/合成/同任务参考集的异常优势 | 衡量性能泛化差，不证明训练语料包含某题；参考模型、难度映射和合成集构造会影响结果。benchmark-specific 差异需要谨慎解释 |
| [PaCoST, arXiv 2024](https://arxiv.org/abs/2406.18326) | 让被测模型分别回答原题与语义改写题，以自评 `P(Yes)` 得到配对置信度，再对差值做单侧 paired t-test | 对公开原题相对改写题出现置信优势时给出一项统计信号；作者用 BERTScore 和人工抽样检查改写质量 | 改写必须语义/难度近似，P(Yes) 需要校准且提示稳定；统计差异不定位数据来源，也不等于分数提升或因果证明；预印本结论按作者测试设置解释 |
| [Sander et al., Detecting Benchmark Contamination Through Watermarking](https://arxiv.org/abs/2502.17259)；[Meta 研究页](https://ai.meta.com/research/publications/detecting-benchmark-contamination-through-watermarking/) | 在 benchmark 发布前用水印模型改写题目，再检测模型训练后的水印残留 | 受控实验在 1B 模型、10B tokens、ARC-Easy/ARC-Challenge/MMLU 上展示可检测性；ARC-Easy 的约 5% 性能增益条件下报告 `p=10^-3` | 前瞻性设计，不能追溯旧 benchmark；结果来自受控设置，水印稳健性、改写效用和现代大模型真实管线还需验证；属于 ICLR 2025 Workshop 工作 |
| [AntiLeakBench, ACL 2025](https://aclanthology.org/2025.acl-long.901/) | 从 Wikidata/Wikipedia 历史版本找 cutoff 后更新的事实，自动构造带来源文档的 QA 样本 | 提供动态新鲜知识集构造方案，指出仅收集新发布题可能遇到已存在知识、人工维护成本高 | 主要面向 QA；作者指出自动识别新知识、样本质量与数据源错误仍有限制。cutoff 是模型/数据事实的条件，不是对所有暴露通道的绝对保证 |
| [Fu et al., Findings NAACL 2025](https://aclanthology.org/2025.findings-naacl.291/) | 系统综述 50 篇检测论文，整理 8 类方法假设，对其中 3 类 membership-inference 假设做案例测试 | 论文使用的预训练数据样本中，部分成员推断假设下的方法接近随机猜测；提醒先验证方法假设 | 不可外推为所有 membership inference 都无效，也不证明模型未记忆；适合作为检测方法假设的审计来源 |
| [LiveBench v2](https://arxiv.org/abs/2406.19314v2) | 用近期题源、客观答案与周期更新降低静态 benchmark 饱和 | 展示动态 benchmark 如何把“题目新鲜度”纳入评估生命周期 | 公开题随时间可能进入后续训练；每个快照只对相应版本和日期成立，动态更新会带来跨版本可比性问题 |
| [Detecting Pretraining Data from Large Language Models](https://arxiv.org/abs/2310.16789) 与 [Rephrased Samples Improve Benchmark Robustness](https://arxiv.org/abs/2305.13104) | 分别提供概率特征/记忆推断和原题改写稳健性研究入口 | 补充传统 near-duplicate 之外的行为证据路线 | 检测分数与公开模型实验均依赖具体任务、分布、API和阈值，不能直接当训练来源审计 |

## 方法之间如何互补

这些方法测量的不是同一个“污染量”：

| 证据路线 | 主要命题 | 优势 | 失败条件与下一步 |
| --- | --- | --- | --- |
| 实际语料血缘与 exact/near overlap | 某来源快照是否与评估样本重合；实际模型训练链路能否追到该快照 | 可定位来源和处理动作；白盒条件下信息流证据直接 | 公开网页命中不证明目标模型消费；精确/近似匹配受字段、规范化和阈值影响。补训练 run manifest、抽样人工复核及来源时点 |
| Oren 顺序似然检验 | 模型是否保留某 benchmark 的 canonical 排列痕迹 | 无需训练数据，具有明确统计检验目标 | 需要 exchangeability 和 token/sequence log-prob；与内容知识记忆无关时无信号。补语料扫描与新题测量 |
| ConStat 性能差距 | 原 benchmark 的分数是否超过同能力/难度参考集所预测的水平 | 不必知道具体训练样本，可报告效应量与显著性 | 不证明直接数据重合；参考集不匹配可制造差距。补目标分布样本、难度切片、参考模型审计和多重检验校正 |
| PaCoST 原题/改写题配对置信度 | 模型对公开表述是否比语义近似版本更有把握 | 配对控制部分难度差，直接检查措辞敏感性 | paraphrase 偏差与自评置信失准会混入信号；需要人工验证改写、多个提示/改写器和效应量 |
| Benchmark 水印 | 后续训练链路是否残留发布前嵌入的水印 | 发布前可留下可追踪标记 | 只对新版本前瞻有效；水印可能被清洗，必须验证 benchmark utility 与检测功效 |
| cutoff 后新鲜知识集 | 模型是否能使用截止时间之后的知识与证据 | 可持续补入新的事实样本；适合动态 QA | 对稳定能力覆盖不足，cutoff/来源不等于绝对不可见；维护版本、来源正确性和可比协议 |

面试或审计时先问“我想证明哪一句话”。若要证明信息经过某条训练链路，应看
训练 manifest 和来源证据；若只能访问闭源模型，则通过不同方法观察统计迹象和
泛化差距，但结论必须降级为“风险信号”。若目标是持续测当前能力，则采用固定
历史回归集加时间更新 holdout，并把分数、版本和样本生命周期分开管理。

## 关键定义与正文落点

- §4.3 把 exact overlap 的证据分成“与真实训练快照重合”和“只在公开来源找到相同文本”；前者支持信息流证据，后者只是潜在暴露来源，均不能单独证明记忆造成分数提升。
- §8.6 加入 Oren、ConStat、PaCoST、benchmark watermark、AntiLeakBench 和 Fu 等人的综述/评估。ConStat 的核心量写为 `δ = S_D(M) − F_D^{-1}(F_Dref(S_Dref(M)))`，并解释参考模型分位数校正与 bootstrap。
- 研究对照把 contamination 拆为信息流、模型行为、成绩泛化三个命题；不把不同检测输出合成污染率。
- §9.5 补充报告时分别记录重合文本、行为异常和未泛化成绩；没有可靠暴露标签与定义分母，不报告单一污染率。
- §11.3 标注 60/20/10/10 仅为报告示例，不是通用样本配比建议。
- §13 更新论文入口并明确来源所支持的证据和各自限制。

## 代码、链接与未完成验证

- §12 demo 只做静态复读和逻辑核对，没有运行 Python。exact digest 调整为 NFC + casefold + 空白折叠且保留标点；shingle 使用 Unicode 字母/数字字符，避免把完整宽字符范围写死。toy 语料行加入与 eval join 一致的句间空格。
- 修正了门禁的行动覆盖：manifest 字段缺失会产生 `complete_eval_manifest`，final holdout 未隔离会产生 `restrict_final_holdout_access`。当前合成 manifest 完整、访问策略为 hidden-holdout，所以预期输出不增加这两项动作。
- 输出标题改为“按代码逻辑核对的预期输出（本轮未执行 demo）”，不再称为实际运行结果。未执行 demo、单元测试或其他测试套件。
- 检查章节结构、关键公式和源码对应的已知命中；修改本轮新增引用均指向 ICLR/NeurIPS/ACL Anthology/arXiv 或论文作者单位发布页。动态 LiveBench 页面和 cutoff/API事实需依具体版本复核。
- 仍未取得目标模型训练 manifest、真实语料、闭源模型 log-prob、专家评估数据、目标 API cutoff 或 holdout 访问日志；不对目标模型/benchmark 给出污染率、无污染证明或分数因果归因。
