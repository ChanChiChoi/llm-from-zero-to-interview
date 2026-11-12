# 第四章审计：Dataset 与 DataLoader

审计日期：2026-11-09

状态：章节精修完成；十章骨架不变。

## scope

- 完整复读第四章 §§4.1–4.20，沿三条问题链审阅：单样本到有效监督 batch、key/sampler 到 rank 与 worker 分片、DataLoader 等待到吞吐和内存诊断。
- 章级核验范围：map-style / IterableDataset、`__getitem__` / `__getitems__`、自动 batching 与 collate、padding/shift/loss mask、packing、shuffle/sampler/batch_sampler、length bucket、worker seed 与内存、pinning/prefetch/`in_order`、`DistributedSampler` 的补样/丢尾。
- 保留单章定位，没有新增章节或改变第十四册边界；流式数据治理、目标数据源吞吐、多机恢复和生产数据管线继续交给数据工程/训练系统相关章节。

## chapter_scan

- **日期与范围：** 2026-11-09；以 PyTorch 2.15 `torch.utils.data` 文档和同版本数据加载优化教程为准，定向核验公开 API 行为。另在 PyTorch 2.12.0+cpu 做小型 CPU 代码核对；不把版本文档等同于本地 2.15、CUDA 或生产验证。
- **主要问题：** Dataset 协议是否准确区分可选长度和批量取样；变长 batch 的 attention mask、label ignore mask 与 causal shift 是否一致；不同 `drop_last` 属于哪一阶段；IterableDataset 多 worker/rank 是否会重复、漏样或误报长度；worker 随机性、内存和 DataLoader 性能参数的适用边界是什么。

| 候选缺口 | 证据与决定 |
| --- | --- |
| map-style Dataset 的 key/长度协议与 `__getitems__` | 修订为 `__getitem__` 必需，有限索引集常实现 `__len__`，key 不限整数；补 PyTorch 2.15 可选 `__getitems__` 批量取样协议及其只减少特定固定开销、不保证加速的边界。 |
| IterableDataset shuffle、sampler 和 worker 行为 | 明确 iterable 顺序由 `__iter__` 决定，不能使用 map-style sampler/batch_sampler/`shuffle=True`；worker 必须显式分片，分布式需把 rank 与 worker 组合为全局 shard。 |
| padding mask 与训练目标 | 增加手算样本 `[11,12,13,14]`、`[21,22]`：padding 后 `[B,T]=[2,4]`，有效输入 token 6/8，shift 后有效目标数 4；明确 attention mask 与 loss ignore mask 不可互换。 |
| `DistributedSampler` 和 DataLoader 的 `drop_last` | 用职责表分开 sampler rank 切分前的补索引/截尾与每 rank 组 batch 后的尾批处理；以 `N=10,R=3` 手算各 rank 索引、重复和丢弃。 |
| worker seed、内存、pinning 与 `in_order` | 补 worker Python/NumPy 独立播种、`generator` 的作用范围、访问大型 Python 对象造成的多进程内存增长、自定义 batch `pin_memory()`、预取估算以及 `in_order=False` 的可复现/分布风险。 |
| IterableDataset 长度和尾批 | 记录多 worker 下 `drop_last` 对每个 worker 副本分别生效；若 Dataset 提供长度，`len(dataloader)` 对分片流仍只是估算，验收应统计实际 shard/record id。 |

来源事实、版本和未验证边界见[PyTorch DataLoader 2.15 研究记录](../../research/pytorch-training-api-boundaries-2026.md#dataloader-与-dataset-pytorch-215)；本文件记录的是章节取舍与验收。

## length_status

仅计中文解释性正文，排除代码、公式、表格、标题、代码状态、URL 和行内代码标识。三条教学链均超过核心单元 1,800 字门槛；§4.11 同时连接采样/worker 随机性和性能诊断，故在相应链中作为共享桥段，以下链长不可相加。

| 教学链 | 覆盖范围 | 类型 | 实计汉字数 | 下限 | 状态 |
| --- | --- | --- | ---: | ---: | --- |
| 单样本到有效监督 batch | §§4.1–4.8 | 核心 | 3,039 | 1,800 | 通过 |
| key、采样顺序到 rank/worker 数据覆盖 | §§4.9–4.12 | 核心 | 2,633 | 1,800 | 通过 |
| worker 到加载测量和故障审计 | §4.11、§§4.13–4.17 | 核心 | 2,093 | 1,800 | 通过 |

综合 demo、练习和资源清单不单独承担完整教学，不另作为独立辅助单元计数。

## example_status

- **padding/causal shift 手算：** §4.1.1。两条长度 4 和 2 的序列 pad 到 4；真实 token 6/总槽位 8，利用率 0.75、padding 0.25；labels 右移后有效目标为 3+1=4。手算解释 `attention_mask` 与 label ignore mask 的不同职责。
- **length bucket：** §4.10。长度 `[4,2,6,3,5,3]` 按原顺序组成两个 batch 共需 33 个槽、真实 token 23 个，waste `10/33≈0.303`；分桶后 27 个槽、真实 token仍为 23，waste `4/27≈0.148`。
- **DistributedSampler：** §4.12。`N=10,R=3`、无 shuffle 时，默认补到 12 个位置并产生两个跨 rank 重复；sampler `drop_last=True` 时分成三个各 3 个样本的 rank 并丢掉索引 9。另说明 DataLoader 每 rank 的 batch 尾丢弃是后续独立决策。
- **prefetch 容量估算：** §4.11。4 worker × factor 2 约排 8 个 batch；每 batch 假设 64 MiB 时约为 512 MiB，明确不含 Dataset 副本、临时对象、共享内存和主进程 batch，非峰值保证。

## code_status

全章 26 个 Python 围栏都有紧邻状态：6 个已在 CPU 环境执行，18 个静态审阅未运行，2 个标为伪代码。环境为 Python 3.12.9、PyTorch 2.12.0+cpu；worker 随机数例另用 NumPy 2.4.6。官方 API 资料版本为 PyTorch 2.15，与本地运行版本分开记录。

| 位置/代码块 | 状态、命令输入与核验边界 | 实际结果 |
| --- | --- | --- |
| §4.6.2 `pad_labels` | 已验证；从章节提取函数并在 `import torch` 环境调用 `[[1],[2,3]]` 和空列表 | 前者得到 `[[1,-100],[2,3]]`；空列表抛出预期 `ValueError`。 |
| §4.11 DataLoader 参数配置 | 已验证；章节配置使用 4 workers、batch 8、`persistent_workers=True`、factor 2，对 `TensorDataset(torch.arange(16))` 取一批 | batch shape `(8,)`；运行环境没有 accelerator，收到 pin memory 未启用的提示，因此未验证锁页或 CUDA 传输。该验证在 sandbox 外完成，因为 sandbox 限制 PyTorch worker 的本地多进程通信。 |
| §4.11 `seed_worker` | 已验证；2 workers、`generator.manual_seed(7)`，含 Python `random` 和 NumPy 随机值的 8 条记录；同 seed 重跑 | 两次记录完全一致、worker id 为 0/1，逐样本随机对不重复；仅说明该 CPU 环境的 worker seed 示例。 |
| §4.12 `DistributedSampler` 参数片段 | 已验证；`dataset=range(10)`、`world_size=3`、`rank=1` | rank 1 产生 4 个索引并组成 batch；这是单进程模拟 rank，不是多进程通信验证。 |
| §4.13 完整文本数据管线 | 已验证；章节内 3 条文本、batch size 2、动态 padding | 两批 tensor shape 分别为 `(2,8)`、`(1,8)`；mask 与长度一致。 |
| §4.15 综合审计 demo | 已验证；固定 toy sequences、DataLoader generator seed 7、sampler seed 13、2 rank 模拟 | 关键断言通过；首批长度 `[2,6,3]`，valid ratio `0.611`，shift 后目标数 8，padding waste `0.303→0.148`；整除数据集 rank 无重叠，不可整除时补齐交集为 `[0]`，sampler 丢尾结果为 `[[0,2],[1,3]]`。 |

其余 18 个片段依赖相邻定义、只示范返回结构或属于 API 配置入口，标为静态审阅；§4.6.3 的通用 collate 和 §4.8 packing 仍未做单独运行边界测试。未验证 PyTorch 2.15 runtime、CUDA pinning、目标数据源、真实多 rank/DistributedDataParallel、IterableDataset 端到端流分片及 DataLoader 实际吞吐。

## formula_status

- 11 个 `~~~math` 公式围栏成对，花括号平衡；章节无行内美元公式。
- 覆盖样本 key 映射、batch shape、padding/有效 token 比例、causal shift、rank 集合、loss token 归一化、DistributedSampler 补齐/截尾和有效 tokens/s。关键关系均有具体手算输入和中间量。
- 最终结构核对中，60 个三反引号标记配对、26 个 Python 围栏均有相邻状态，11 个数学围栏的花括号平衡；本轮最后两处文字补充未改变代码或公式。
- 公式完成源码层级围栏与花括号检查；未宣称 GitHub/MathJax 页面实际渲染。

## link_status

- 章节 16 个 Markdown 链接全部指向 PyTorch 官方 2.15 文档或对应教程；本章无本地相对链接。
- 新增 API 结论旁有对应官方链接。已阅读 DataLoader 文档中 Dataset 类型、自动 batching、worker、随机性、pinning、`in_order` 和 DistributedSampler 段落，以及 PyTorch 2.15 数据加载优化教程。
- 第四章审计、架构审计、简介和目录中的本地 Markdown 链接目标均通过静态存在性检查；未执行全项目外链 crawl。
- 没有对全部外链执行 HTTP crawler；已打开官方 2.15 `torch.utils.data` 页面核对关键 API 和 anchor，教程页面也直接阅读相关 `__getitems__`、`in_order` 和测量段落。

## visual_status

- 增加 map-style 与 iterable-style 对照表，区分取样、顺序/分批、多 worker 和典型数据源。
- 增加 sampler 与 DataLoader 两种 `drop_last` 的处理阶段表，并以 `N=10,R=3` 的索引轨迹给出具体结果。pad/shift 和 bucketing 使用同一组数字逐步解释，不额外加入装饰图。

## sync_status

- [第十四册简介](../../book-14-pytorch-deep-learning-engineering/简介.md)与[目录](../../book-14-pytorch-deep-learning-engineering/目录.md)已更新第四章问题、能力和训练循环交接。
- `INTERVIEW_BANK.md` 更新 Q25–Q32 的 Dataset 边界、mask/shift、sampler、distributed tail、worker 和审计 demo 回答；`EXERCISES.md` 更新第 117–124 题；`GLOSSARY_EN_ZH.md` 添加 `__getitems__` 与 `in_order`。
- PyTorch 2.15 API 新事实与来源边界写入现有 research note；审计、取舍、代码验证和遗留范围只写入本 `progress/` 文件。
- 未更新 `PAPERS.md`、项目索引、`plan.md` 或 PDF。

## open_risks

1. 本地执行环境是 PyTorch 2.12.0+cpu；未在 PyTorch 2.15 runtime 或 CUDA 下验证。配置示例在无 accelerator 环境无法验证 pinning 实际效果。
2. `DistributedSampler` 为单进程 rank 对象模拟；没有启动真实多 rank DDP 或核对目标集群的 epoch 步调。
3. IterableDataset 的全局 rank/worker shard 规则需要绑定实际数据源和 shard 元数据；本章仅证明整数范围示例的分片逻辑，不验证远程流故障恢复。
4. `in_order=False`、prefetch 调参、worker 内存、length bucket 和 pinning 的实际性能收益都依赖目标环境，正文只给机制、容量估算与测量方法。
5. 公式源码检查完成，但未做页面 MathJax 实际渲染；官方外链没有全量 HTTP crawl。

## handoff

- 第四章正文、PyTorch 2.15 章级扫描、三条教学链、手算例、6 个代码块的 CPU 验证，以及简介/目录/题库/术语同步已完成；未改变十章骨架。
- 下一步进入[第五章《训练循环工程》](../../book-14-pytorch-deep-learning-engineering/chapters/05-训练循环工程.md)，检查 loss reduction、有效 token、梯度累积、optimizer/scheduler 步进、日志和 checkpoint 恢复如何接成一致的训练状态链。
- 全套书精修 goal 继续 active；不更新 PDF，不修改 `plan.md`。
