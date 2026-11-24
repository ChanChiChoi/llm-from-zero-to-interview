# Checkpoint、评估与实验管理证据（2026-11-24）

本文记录本轮检索到的第 10 章新证据。来源限于框架官方文档、原始论文和评测方法论文；版本化 API 结论只适用于所标版本。这里不记录章节审阅进度或编辑历史。

## PyTorch checkpoint 与恢复边界

### Distributed Checkpoint 2.15

- 来源：[PyTorch 2.15 Distributed Checkpoint 文档](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html)，官方 API 文档，版本 2.15，访问日期 2026-11-24。
- DCP 为多 rank 并行保存和加载 checkpoint，并支持加载时重新切分，使保存和加载可采用不同集群拓扑。它生成多文件工件；对具体模型、优化器和并行策略，仍需核对实际 state-dict 形式。
- 文档明确不保证保存的 state dict 在不同 PyTorch 版本间向后兼容。实验应记录框架版本，并用目标恢复环境做加载演练；“有 reshard 能力”不等于“跨版本兼容”。
- PyTorch 2.15 DCP API 文档说明，`async_save` 默认先把 state dict de-stage 到 CPU staging storage，再在独立线程执行保存；该 API 被标注为实验性、可能变化。异步保存教程补充 GPU 到 CPU 的 staging 拷贝本身仍可能阻塞主线程。使用 `AsyncStager` 可把 state-dict 创建和 D2H 拷贝移至后台线程。该模式的返回对象暴露本地 staging 完成和 checkpoint 写入完成；训练更新原状态前应等待 staging 完成，发布完成版本前应等待写入完成。该返回对象、两个 future 和执行方式不应泛化到所有版本或所有异步保存调用方式。
- 官方教程指出异步 staging 会带来额外 CPU 线程、主机/pinned memory 与并发开销，并在示例中等待前一请求完成以限制队列；具体并发度仍须按内存与 I/O 实测。[PyTorch 2.15 异步保存教程](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)
- `FileSystemWriter` 文档所述文件创建原子性是单文件层面的保证；多文件、多 rank 的 checkpoint 发布仍需应用维护 manifest、完成标记或后端事务语义。不能把单个文件的原子创建推成整个训练状态集合的事务原子性。

### 普通 checkpoint 与随机性

- 来源：[PyTorch 2.15 可复现性说明](https://docs.pytorch.org/docs/2.15/notes/randomness.html)，官方文档，访问日期 2026-11-24。文档说明即使随机种子相同，不同 PyTorch 发布版、提交、平台，甚至 CPU/GPU 执行也不保证完全复现。
- Hugging Face Trainer 的 `resume_from_checkpoint` 用于从 checkpoint 恢复模型、优化器和调度器等训练状态，但可恢复字段取决于保存选项和具体版本；model-only 保存不等于完整续训状态。[Transformers Trainer 文档](https://huggingface.co/docs/transformers/main/main_classes/trainer)，官方动态文档，访问日期 2026-11-24。引用时应将具体语义绑定到项目锁定的 Transformers 版本。
- [PyTorch 2.15 `torch.load` 文档](https://docs.pytorch.org/docs/2.15/generated/torch.load.html)，官方 API 文档，访问日期 2026-11-24：`weights_only=True` 将允许反序列化的范围限制为张量、基本类型、字典及安全列表中的类型；`map_location="cpu"` 可把存储先载入 CPU。官方仍警告不要加载不可信来源的序列化文件；受限 unpickler 不构成对恶意文件的绝对安全保证。

## Validation loss、perplexity 与评测协议

- 来源：[Hugging Face fixed-length model perplexity 说明](https://huggingface.co/docs/transformers/perplexity)，官方说明页，访问日期 2026-11-24。对上下文长度有限的 causal LM，滑动窗口和 stride 会改变每个目标 token 获得的上下文；评估应只计新目标位置并保持 mask、stride、有效 token 分母一致。较小 stride 通常给更多上下文、常产生较低 PPL，但这是测量协议变化，不能拿来单独证明模型更好。
- 来源：Liang et al., [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)，原始研究（HELM，2022 预印本；正式版本发表于 2023）。HELM 把评测组织为场景与指标；其 core suite 覆盖 16 个场景和 7 类指标，论文报告共同覆盖率从既有结果汇总的 17.9% 提高到 96.0%，并公开 prompts/completions。数字严格限定于论文的模型集合和 core scenario 口径，不表示所有真实用例都被覆盖，也不支持压成单一总排名。
- 来源：Gao et al., [Lessons from the Trenches on Reproducible Evaluation of Language Models v3](https://arxiv.org/abs/2405.14782v3)，原始方法论文，v3 发布于 2026-05-31。论文强调任务定义、prompt、模型适配和评测脚本的小差异都会影响分数，并说明 lm-eval 默认 bootstrap 区间估计从同一任务分布重新抽题时的波动；这与同一设置下重复运行模型造成的行为波动不同，prompt 与评分器选择还会带来其他波动来源，应分别报告。
- 评测协议至少要让模型/tokenizer、样本和切分、prompt/few-shot、解码、解析和评分器、harness 版本可追溯；工具提供可重复执行机制，但不替用户证明 benchmark 的代表性。

## 成对比较与污染检测

- 来源：Dror et al., [The Hitchhiker’s Guide to Testing Statistical Significance in NLP](https://aclanthology.org/P18-1128/)，ACL 2018 方法论文。检验应依任务和 metric 选择；同题评估要保留配对信息，显著性也不等于实际重要性。指南的主要对象是 NLP 实验设置中单项比较的检验选择，跨多个数据集/大量指标的多重比较需要额外策略。
- 来源：Oren et al., [Proving Test Set Contamination in Black-Box Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/46e624c244cff669223d488defd4e835-Abstract-Conference.html)，ICLR 2024 原始论文。方法比较 canonical 顺序与打乱顺序的 benchmark 序列似然；“精确假阳性保证”依赖论文的 exchangeability 假设和可取得似然的接口。论文在其特定实验中报告可检测到 1.4B 模型、约 1,000 题及少量重复出现的 benchmark；不能外推为所有 benchmark 或所有黑盒 API 都可检测。
- 来源：Dekoninck et al., [ConStat](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a7f89793b9e6f8c6568dbbb6ff727b9b-Abstract-Conference.html)，NeurIPS 2024 原始论文。其污染定义是原始 benchmark 上异常升高且不能泛化到改写、合成同分布题或同任务参考 benchmark 的表现；方法用参考模型和参考 benchmark 的关系估计异常。它测的是非泛化表现风险，不是训练语料中具体样本包含关系的直接证明。

## 实验追踪工具来源

- 来源：[MLflow Tracking 文档](https://mlflow.org/docs/latest/ml/tracking)，MLflow 动态官方文档，访问日期 2026-11-24。文档展示 run、参数、指标、artifact 与实验检索的工具组织方式；这是产品能力说明，不规定所有项目都必须采用 MLflow，也不自动保证数据、模型或 artifact 长期可取。

## 本章推导的量及解释边界

- 若 $C_{save}$ 表示一次 checkpoint 的保存时间，$f$ 是保存间隔 step 数，$T_{step}$ 是平均 step 时间，那么 $C_{save}/(fT_{step})$ 是 checkpoint 时间相对计算时间的开销比。若要表示它占“计算加保存”总墙钟时间的比例，分母应计入保存时间；因此正文中应避免把当前公式称为总时间占比。
- 保存次数、最大回退 token 数、容量去重和评估分数归一化属于在显式假设下的账本或政策定义，不是论文提供的通用训练常数。示例数字必须保留“构造输入”边界。
- Checkpoint“可读取”与“可恢复”是不同判断。字段存在只能完成文件级审计；是否续训连续仍要通过 optimizer/scheduler/RNG/data cursor 与一次短恢复探针验证。
- 恢复探针的 loss 比较应使用相同 batch、mask、归约和随机状态下的未中断参考轨迹；直接比较中断前后相邻日志会混入 batch 难度差异。学习率应与恢复步对应的 scheduler 预期值比较。此对照方法是基于状态恢复目标的实验设计建议，不是 PyTorch 提供的通用 API 保证；非确定性环境需要预先说明容差。
