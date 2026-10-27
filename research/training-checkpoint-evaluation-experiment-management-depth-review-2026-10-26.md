# 第五册第十章：Checkpoint、评估与实验管理深度精修研究记录

日期：2026-10-26  
正文：[`book-05-llm-training/chapters/10-checkpoint评估与实验管理.md`](../book-05-llm-training/chapters/10-checkpoint评估与实验管理.md)

## 研究问题与旧知识点覆盖图

本轮按 `progress_v2.md` 从第五册第九章之后进入第十章，没有重做更早的章节。原章的状态保存、频率/存储、恢复探针、latest 与 best 区别、validation/PPL、per-domain、benchmark、regression、实验记录、命名、复现、公平对照、checkpoint 选择流程、标准库 demo 和常见误判均予保留。修改的重点是把“评测结果可信度”和“多目标选择”讲到可操作层，并修复 demo 的不一致量纲与边界问题。

| 原有主题 | 重要性判断 | 本轮去向与处理 |
| --- | --- | --- |
| checkpoint 恢复字段、格式/分片、保存频率、存储占用 | 核心：训练状态正确与训练容错的基础 | §1—4 保留字段集合、容量和保存频率公式；§4 补 DCP 2.14 的 load-time resharding、rank metadata、异步保存的 staging 内存和恢复版本边界。 |
| 最后一个 checkpoint 不一定最好 | 核心：训练步数与使用目标不是同一问题 | §5、§14 保留多目标选择、硬门槛、Pareto 前沿与场景权重。 |
| validation loss/PPL、domain loss | 核心：训练目标信号和领域退化信号不能混为能力总分 | §6—7 原样保留机制和分母说明；不以 PPL 替代下游 benchmark。 |
| benchmark 覆盖和可信度 | 核心：分数高度依赖 protocol，且有抽样波动与污染风险 | §8 新增评测协议对象、HELM/lm-eval 分工、配对检验、题目与 seed 不确定性、多重比较及污染检测路线。 |
| regression、安全、人工评测 | 核心：平均分提升可掩盖严重的旧能力退化 | §9 保留 lost-case 分母与 weighted regression；与 §8 的样本级评测协议相连。 |
| run manifest、命名、数据/代码/环境版本、公平比较、复现等级 | 核心：结果必须能归因、恢复和复核 | §10—13 保留；§8 明确 prompt/template/scorer/harness 与 task version 也是实验条件。 |
| toy checkpoint 选择代码、误判场景、总结 | 核心：把概念落到真实字段和可检查输出 | §15 保留并修正打分尺度、配置键比较、候选为空、step/token 单调性与 benchmark 集合一致性；§16 的恢复、last-vs-best、loss、seed 与公平性诊断保留。 |
| 生产级对象存储、跨机房灾备、完整 MLOps/权限审计 | 支持性/低优先级：本章边界内不完整展开 | 原范围声明继续保留；checkpoint 完成标记只解释到读取端可判断完整状态，不扩展成对象存储一致性协议教程。 |

## 主要研究与方法差异

### 评测覆盖与统一执行是两个不同层次

- Liang et al., [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110), 2022。阅读论文摘要、scenario/metric taxonomy 与 evaluation protocol。论文用 16 个 core scenarios 和 7 类 metrics 组织共同评测，并发布 prompts/completions；报告在所选 core scenarios 上把先前平均覆盖率从 17.9% 提高至 96.0%。正文只把它用于说明共同覆盖、多指标及可复核原始输出的价值；其场景集合受覆盖/可行性取舍限制，不能解释成完整代表所有真实使用场景。
- Biderman et al., [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782), arXiv v3（页面标注 2026-05-31）。阅读评测协议丢失、Best Practices、lm-eval 设计、bootstrap 和 limitation 部分。论文把 lm-eval 定位为非规定 benchmark 的 orchestration library：YAML/Python task 将 data、prompt/rendering、request 类型、postprocess 和 metric 放到共同接口。它解决执行和沟通细节分散的问题，不决定某个 benchmark 是否有代表性。论文还区分固定模型下的题目 bootstrap 波动与模型/训练 seed、prompt-selection 波动；正文将其拆开说明。
- EleutherAI [`lm-evaluation-harness` task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md) 与 [CLI interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)：核对 `num_fewshot`、output type、generation kwargs、task version、samples 与 `--log_samples` 等工程字段。GitHub `main` 是动态来源，具体运行结论需固定仓库 commit。

正文据此把一次评测表达为 `(checkpoint, tokenizer, template, task/split, prompt/few-shot, generation, scorer, harness/version)`，要求保存逐样本输入、输出、解析和评分。HELM 与 lm-eval 不是互相替代的排行榜：一个侧重“评什么和如何保持共同覆盖”，另一个侧重“怎样可复核地执行”。

### 配对统计的证据和不确定性边界

- Dror et al., [The Hitchhiker's Guide to Testing Statistical Significance in NLP](https://aclanthology.org/P18-1128/), ACL 2018；以及 [Recommended Statistical Significance Tests for NLP Tasks](https://arxiv.org/abs/1809.01448)。阅读其按任务类型选择检验的指南。正文用二元正确性差异说明 `b=old-only correct`、`c=new-only correct`，差值为 `(c-b)/n`；McNemar 查看 discordant pairs，而不是把同题上的两个准确率当作独立样本。连续逐题分数适合按题成对 bootstrap/置换，检验方式仍须按 metric 和样本依赖选择。
- lm-eval 论文将题目分布 bootstrap 与模型多次运行、few-shot 示例变化等随机性区分。正文据此明确：重采样固定 checkpoint 的测试题估计的是题目样本波动；训练 seed 重跑、few-shot 选择或随机解码才覆盖另外的变异来源。统计显著不等于业务有意义；正文建议预先声明主要指标、最小有意义差异和多重比较处理。

文中手算关系：若 `n=1000`、`b=10`、`c=30`，新模型净多对 20 题，准确率差 `20/1000=0.02`。这个差值本身不能证明统计稳定；还要查看不一致样本总数、区间/检验，并决定 2 个百分点是否超过实际决策门槛。

### benchmark 污染不是一个单一可观察变量

- Oren et al., [Proving Test Set Contamination in Black-Box Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/46e624c244cff669223d488defd4e835-Abstract-Conference.html), ICLR 2024。论文利用 exchangeability 直觉比较 canonical benchmark 排序与 shuffled 排序的序列似然，报告在其条件下对少量出现/小样本污染具备检测能力，并称有 exact false-positive guarantees。此方法需要能查询序列似然并接受相应题目分布假设；它测得的是行为信号，未报警不证明训练数据不存在任何形式泄漏。
- [ConStat: Performance-Based Contamination Detection in Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a7f89793b9e6f8c6568dbbb6ff727b9b-Abstract-Conference.html), NeurIPS 2024。作者把污染操作化为主要 benchmark 上异常高但无法迁移到重述、同分布合成样本或同任务参考集的表现；它比字面字符串 overlap 扩大到行为泛化，但结论依赖 reference tasks/models，不等于对训练语料中出现原始样本的直接证明。
- 方法关系：开放语料可以扫描 exact/near overlap、来源日期和 split；闭源模型可用似然/行为信号、新写或隐藏留出题和逐样本输出复核。不同方法检查污染层级不同，正文不把“无重叠”或“未触发检测”写成 clean certificate。

### DCP 与训练恢复接口

- [PyTorch Distributed Checkpoint 2.14](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html)：官方文档说明多 rank 并行读写、load-time resharding，以及按 canonical FQN 生成能够跨 trainer 数量/并行策略加载的 state dict API；也提醒各 rank key 应一致，保存的 state dict 不保证跨 PyTorch 版本向后兼容。应用发布指针、object store transaction 与数据游标恢复仍要自行验证。
- [PyTorch asynchronous DCP recipe 2.14](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)：异步保存由 future 管理并需限制并发请求；staging 会增加 CPU/pinned memory 压力。正文只引出状态拷贝、future 完成和应用发布边界，没有宣称在本地执行过 DCP/GPU。
- [Hugging Face Trainer v5.19.0](https://huggingface.co/docs/transformers/v5.19.0/main_classes/trainer)：访问核对 best checkpoint 指标、save/eval strategy、resume state 行为；本轮把链接固定到实际可打开页面。文档说明当前有更新版本，按目标 lockfile 再核。

## 正文修改和公式/代码验收

- §8 的旧 accuracy standard error 保留，并维持 `n>0`、`p∈[0,1]`、近似 iid 限定；新增 paired comparison、discordant counts、seed/prompt variation 和 contamination methods。没有把一个通用 SE 公式扩展成所有 judge/generative metrics 的置信区间。
- §14 既有加权选择公式保留其“归一化量纲、先设硬门槛”的限定。§15 toy policy 固定 `val_loss_cap=4.0`，把 `val_loss`、benchmark、safety 和 regression rate 映射到 `[0,1]`，权重为 `0.40/0.30/0.15/0.15`；稳定性、状态完整、安全分数、domain regression 和最多 5/100 回归失败作为硬门槛。权重、cap 和阈值是展示决策偏好的 toy 参数，不是研究结论。
- 主路径运行结果：validation/PPL 为 `10.07 / 8.85 / 8.67`；domain regression 只在 `ckpt_3000` 的 code；normalized scores 分别 `0.565 / 0.595 / 0.577`；硬门槛后选择 `ckpt_2000`；保留 `ckpt_2000` 与 `ckpt_3000` 共 `360 GiB`；保存开销 `0.0075`；最大回滚量 `52.5B tokens`；恢复探针四项连续性均通过；manifest 完整；学习率单变量对照公平，坏对照识别出 `data_version` 与 `tokens_b`。
- 运行代码没有增加项目依赖。以 Python 执行了章节 demo，文中期望输出与实际输出逐字一致；另检查 missing/extra config key、忽略变量、空 manifest、合法零 seed、NaN metric、PPL overflow、失败数超过样本分母、domain key mismatch、无合格候选、空 checkpoint 列表、benchmark key mismatch、权重不为 1 和 duplicate step 共 14 条边界；均按预期返回或拒绝。示例 checkpoint profile 明确假设启用了 FP16 scaler，BF16/无 scaler 项目须改写必需字段策略。

## 配套同步与验证

- `PAPERS.md`：添加 HELM、lm-eval reproducibility、NLP 显著性指南、Oren 和 ConStat 论文入口；将第五册第十章加入评估方法正文入口。
- `GLOSSARY_EN_ZH.md`：补 Evaluation Contract、HELM、lm-eval、Contamination Signal；paired/bootstrap/McNemar 既有词条已存在，不重复造词。
- `INTERVIEW_BANK.md`：增加 HELM 对 lm-eval、污染检测路线、paired comparison/seed variation、多目标选择四题，含各方案证据和限制。
- `EXERCISES.md` 和正文 §18：增加评测协议、成对统计、题目/seed/prompt 波动、污染审计和归一化 checkpoint 选择练习。
- 标题编号调整为 1—19 连续；公式围栏、Python/text fence、表格和本册目录映射复核；新增外链核心来源已逐页打开核验；`git diff --check` 通过。

## 仍未验证

没有运行真实训练、checkpoint restore、DCP、多 GPU、对象存储、评测 benchmark、污染检测或目标业务 scorer；没有验证 checkpoint 是否能跨目标 PyTorch/Transformers revisions 恢复，也没有估计真实 seed variance 或部署权重。HELM、lm-eval、Oren、ConStat 和统计指南的研究结论仍绑定论文的任务、模型、假设和评测接口。GitHub `main`、MLflow `latest` 与 Hugging Face 未固定文档页需要按使用版本复查。标准库 toy demo 只能验证本章数据和口径。

## 验收状态

第五册第十章完成本轮资料驱动精修与代码/公式/配套审查。旧知识点仍可在上述 coverage map 中追踪；后续按 `progress_v2.md` 进入第五册第十一章，不回头重做本章之前的进度。
