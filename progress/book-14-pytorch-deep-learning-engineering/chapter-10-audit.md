# 第十章精修审计：工程问题与判断

审计日期：2026-11-11

## 范围与章级扫描（scope / chapter_scan）

- 完成《工程问题与判断》§§10.0–10.16 全文审阅与精修。章节主问题仍是“怎样把模糊工程现象转成成功条件、可证伪假设、最小实验和可复现结论”，没有改变第十四册骨架。
- 2026-11-11 按正文问题复核 PyTorch 2.15 官方 Tensor Views、Autograd/inference mode、Module、DataLoader/DistributedSampler、AMP、DDP、FSDP2、serialization、Profiler、CUDA 和 SDPA 文档；原子替换细节核对 Python 3.12 `os.replace` 文档。相关 API 事实和来源已在既有[PyTorch API 研究记录](../../research/pytorch-training-api-boundaries-2026.md)及[分布式训练研究记录](../../research/distributed-training-parallelism-2026.md)中记录；本章没有发现需要另立一条研究事实的独立主题。
- 扫描问题：tensor layout 改变的 view/copy 边界；`eval`、`no_grad`、`inference_mode` 是否互相替代；padding/shift 后真实监督分母；梯度累积、AMP 和 DDP 的 update/reduction 口径；训练 checkpoint 是否保存完整状态；资源和计时数字如何标定假设；Transformer cache 的可见性、位置和 tokenizer 协议能否纳入回归。

| 候选主题 | 去向与理由 |
| --- | --- |
| PyTorch API 版本与运行环境混称 | 正文 PyTorch 入口由 `docs/stable` 改为固定 2.15 文档；代码旁仍分别记录本地 Python 3.12.9 / PyTorch 2.12.0+cpu 运行状态。文档版本不能代替本地 runtime 证据。 |
| `inference_mode` 与 `no_grad` 区别 | 纳入 §10.3.3：前者关闭额外 view/version 跟踪，但其 tensor 有更严格的后续 autograd 使用边界；二者均不自动调用 `eval()`。 |
| DDP 有效 token 全局均值 | 在 §10.8.2 加入由默认 rank 梯度平均推导的 `W S_r/M` 缩放及数例，明确限定标准 process group、默认 reducer 与完整同步步；`join()`、自定义 hook、累积尾窗等仍须单独核对。来源和推导边界见既有 distributed research note。 |
| checkpoint 写入原子性 | 在 §10.6.2 限定同一文件系统和支持原子 rename 的存储路径，说明对象存储/多文件状态需要自己的提交协议；不把文件名替换等同于掉电持久性。 |
| CUDA kernel、NCCL 网络调优、生产级分片恢复和 serving 性能 | 保留为目标系统实测或后续册范围，不把 CPU 教学 demo 外推为 GPU、集群或业务质量结论。 |

## 篇幅与解释链（length_status）

计数方式：统计连续解释文字中的汉字；剔除标题、代码、数学围栏、表格、引用链接、行内代码和代码状态。三条教学链均属核心单元，门槛为 1,800 汉字。

| 教学链 | 范围 | 实计汉字 | 类型 / 门槛 | 状态 |
| --- | --- | ---: | --- | --- |
| 从问题契约到训练状态与恢复 | §§10.1–10.6 | 2,482 | 核心 / 1,800 | 达标；沿 shape/dtype、数据与有效 label、梯度更新、checkpoint 状态和恢复回归逐步推进。 |
| 资源账本、分布式统计与故障定位 | §§10.7–10.9 | 1,829 | 核心 / 1,800 | 达标；包含参数/梯度/optimizer state 估算、attention score 内存、AMP 顺序、DDP token 缩放与 loss/NaN/性能诊断。 |
| Transformer 审查、综合事故与工程证据 | §§10.10–10.13 | 1,989 | 核心 / 1,800 | 达标；从 cache/位置/tokenizer 协议进入三个事故、toy audit 解释和分层证据记录。 |

## 数值例子（example_status）

- §§10.2–10.3：`B=2,T=5,d=16,V=32,H_head=4` 得 `D_h=4`；Q/K/V 是 `[2,4,5,4]`、score 是 `[2,4,5,5]`、logits 是 `[2,5,32]`。view 示例运行得到 `y_shape=(4,3)`、stride `(1,4)`，连续化后值为 `[0,4,8,1,5,9,2,6,10,3,7,11]`。
- §§10.4–10.5：样本长度 `[4,2,5]` 右 padding 到长度 5，shift 后有效 token 是 `3+1+4=8`。有效 token 不等时，micro-batch mean 等权与 token 总和/总数是不同目标；梯度累积尾窗不能继续除以固定窗口长度。
- §10.7：假设 `100M` 参数，BF16 参数/梯度各 2 bytes，AdamW 两份 FP32 state 各 4 bytes，三者为 `200+200+800=1,200 MB≈1.12 GiB`，未包括 activation/workspace/master weights。假设一个朴素层物化 `B=2,H=32,T=2048` FP32 score matrix，为 `268,435,456` 个数即 `1 GiB`；`T=1024` 时约 `256 MiB`。明确这不是融合 attention 的固定显存结果。
- §10.8：rank `(loss sum, valid tokens)` 分别为 `(4,4)` 与 `(12,16)`，rank mean 是 `(1+0.75)/2=0.875`，global token mean 是 `16/20=0.8`。若按 default DDP rank 平均，local mean 的梯度权重因子分别是 `W n_r/M=0.4`、`1.6`；各 rank 缩放后的本地 mean 为 `0.4` 与 `1.2`，归约平均得到 `0.8`。
- §10.10：sample `[7,8,0]` 的末位 `0` 是真实 EOS，且 PAD/EOS 共用 ID。pad 到长度 5 后按真实长度生成 labels `[7,8,0,-100,-100]`，shift 后 `[8,0,-100,-100]`，有效目标数为 2；只按 token id 屏蔽会丢 EOS。
- §10.11：合成的二类 logits `[5,0]` 下，未 shift 标签命中当前输入时 CE 约 `0.0067`，正确 next token 为第二类时 CE 约 `5.0067`，说明实现错位能造成异常低 loss。恢复例子用合成 step `12,400`、学习率 `3e-5` 与 scheduler 重启后的 `1e-4` 区分模型权重 round trip 和完整训练状态。
- §10.12：CPU demo 输出有效 token `8`、有限梯度范数 `4.5739`、参数更新、checkpoint logits round trip、rank mean `0.875`、global token mean `0.8`；不同 reduction 是通过的结构检查，同时作为待选择的统计 signal。

## 证据、公式、图表和链接（evidence / formula_status / link_status）

- API 来源固定为 PyTorch 2.15 官方页面；Python 文件替换引用 Python 3.12 标准库文档。数字包括自构造教学值和明确假设值，不作为生产训练、显存或性能测量。DDP token scaling 是由默认梯度平均和全局 token mean 目标推导出的结果，条件已写在正文。
- 15 个 display-math 围栏均成对、花括号平衡；逐项检查变量定义与假设。三条教学链各有手算或资源预算例子，代码只用于复核，不替代推导。
- 全文 29 对波浪围栏：15 个数学块、2 个 Python 块和 12 个 text 轨迹/公式示意；均配对。两个 Python 块 AST 均可解析，紧邻状态均准确。没有 Mermaid；章节保留面试/诊断流程文本图、张量和内存公式、对照表及 rank trace 字段，已足以呈现当前的步骤和数量关系；未声称平台图形渲染测试。
- 29 个 Markdown 链接的相对目标都解析到现存文件；所有 PyTorch API 链接固定 `/docs/2.15/`，Python 原子替换链接固定 Python 3.12；没有 `docs/stable`、`\operatorname` 或维护路径。未宣称完整外链 crawl。

## 代码状态（code_status）

| 稳定位置 / 代码块 | 状态与验证记录 |
| --- | --- |
| §10.3.1 Python 代码块 01 | **已验证**：2026-11-11，Python 3.12.9 / PyTorch 2.12.0+cpu，CPU；提取正文片段后以 `.tmp/llm-book-cpu-check/bin/python /tmp/llm_book_ch10_view_demo.py` 运行，shape、stride 和 contiguous 后的值与正文相符。 |
| §10.12 Python 代码块 02 | **已验证**：2026-11-11，Python 3.12.9 / PyTorch 2.12.0+cpu，CPU；提取正文后以 `.tmp/llm-book-cpu-check/bin/python /tmp/llm_book_ch10_demo.py` 运行。8 项检查全过：变长 padding 和有效 token、有限 loss/grad、optimizer 参数变化、checkpoint logits round trip、layout shape、rank/global-token 统计差异。 |

边界：两块都只在 CPU PyTorch 2.12 环境运行，没有 PyTorch 2.15 runtime、CUDA、NCCL、多 rank、DataLoader 多 worker、目标 workload 或生产 serving 验证。主 demo 只覆盖 CPU PyTorch RNG，代码将 rank mean 与 global token mean 的差异作为 signal，而非错误 gate。

## 配套同步与遗留项（sync_status / open_risks）

- 本章未改变第十四册对第十章的简介/目录承诺，无需另改摘要；目录链接路径已存在。
- 在 `EXERCISES.md` 增加第十四册第十章六道综合题，覆盖低 loss 假设、完整恢复、DDP 归约、梯度模式、性能测量和 gate 注入；在 `INTERVIEW_BANK.md` 增加 Q79“训练 loss 低而验证不提升的最小实验”。术语均已在既有词条中解释，没有新增容易产生歧义的专有名词，因此未增加 Glossary 条目。
- 外部 PyTorch 来源事实已由现有 research 文件承载；章节修改、证据状态与审阅决定只在此 `progress/` 记录。
- 遗留：目标 PyTorch 2.15、CUDA/多 rank、真实 workload 和性能收益未本地验证；FSDP 与 collective hang 只作接口/事件日志级工程讨论，不代表集群演练。未更新 PDF 或 `plan.md`。

## 工作交接（work_handoff）

第十四册第一至第十章均已完成逐章扫描、正文精修与审计，十章骨架未变。2026-11-11 已按简介/目录主路径完成书册门禁复核：承诺、摘要、章节交接、重复/缺口和本地链接均通过；MathJax 实际渲染、全量外链 crawl、PyTorch 2.15 runtime、CUDA 与真实多 rank/workload 继续作为明确边界保留。下一步进入第十五册架构审视。
