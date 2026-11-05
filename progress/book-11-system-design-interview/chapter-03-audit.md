# 第十一册第三章审计：设计训练平台

审阅日期：2026-11-05  
状态：章级扫描、静态精修和静态门禁完成；GitHub Mermaid 渲染与历史外链全量巡检待第十一册最终门禁。  
范围：`book-11-system-design-interview/chapters/03-设计训练平台.md` 全章，保留 3.0—3.21 编号结构。

## 章级扫描与候选去向

- **扫描问题：** 任务状态是否把运行、重复 checkpoint、周期评估和模型发布误画为单一路径；状态由哪个组件持有；弹性训练在 worker 故障后究竟是进程续跑还是整个 worker group 重启；不同框架的 checkpoint 能力能否跨 topology/版本外推；资源调度是否能从 GPU 数量推到 topology 与时间；章节是否有可复算的训练量、GPU 小时和成本账本；本章与第五册训练机制、第二十三册平台工程及第四章推理平台是否分工明确。
- **资料范围与日期：** 2026-11-05，定向核验 PyTorch Distributed、FSDP、Distributed Checkpoint 与 torchrun 2.15 官方文档，DeepSpeed 0.19.10 Model Checkpointing、MLflow Tracking 和 NCCL 官方文档；并沿用本章 Megatron-LM、ZeRO 的原论文入口。事实版本、来源类型和不外推边界记于[训练平台 checkpoint 与弹性恢复证据记录](../../research/training-platform-checkpoint-elastic-evidence-2026-11-05.md)。
- **候选主题去向：** 不新增章节。训练任务、运行尝试、checkpoint、评估运行和模型发布拆成独立状态对象；PyTorch DCP 的 load-time reshard 及 torchrun 的 worker-group restart 按明确版本作为实现示例；容量和成本用同一组 SFT 假设输入复算；训练理论交第五册，平台工程交第二十三册，在线 serving 交下一章。

| 候选问题 | 本章处理 | 纳入或排除理由 |
| --- | --- | --- |
| 单一枚举混合了训练作业、checkpoint、评估和发布 | 新增对象/状态所有者表与 Mermaid 生命周期图；失败、取消、抢占和模型 blocked 分开 | 会直接改变状态字段、恢复控制器和发布门禁，属于本章主问题 |
| checkpoint 跨 world size 或并行策略恢复 | 补充 PyTorch DCP 2.15 与 DeepSpeed 当前文档的版本范围和边界；要求登记恢复矩阵并测试目标拓扑 | 官方框架功能不能外推到所有格式、版本或等价训练轨迹 |
| worker 故障和成员变化 | 明确 torchrun 重启 worker group、rank/world size 可变化；应用仍需恢复 checkpoint | 防止把 launcher 容错误读成 optimizer、数据游标和 RNG 自动恢复 |
| 资源规划缺少手算 | 增加 70B 模型状态内存量级、SFT 有效 token、训练吞吐、墙钟时间、GPUH 与费用例 | GPU 数、step 数与成本须从单位和假设逐步推导，不能只列目标值 |
| 与相邻书册内容重叠 | 链接第五册分布式训练与 checkpoint、第二十三册训练平台工程；本章在线模型加载交第四章 | 保留系统设计面试所需机制与取舍，不复制算法或 Infra 实现主讲 |

## 主要修订

1. 开篇接入第一章统一答题路线，并在平台边界中链接第五册、第二十三册和下一章；说明本章交付的是经过门禁的模型版本及元数据。
2. 重构 §3.2：按训练任务、attempt、checkpoint、评估运行和模型发布拆分状态与 owner；抢占后可重排新 attempt；checkpoint 保存不再被当作训练阶段；增加带版本条件和允许边集合的状态更新公式及版本 12→13 的冲突例。
3. 在数据集版本处增加 1,000 个文件增至 1,001 个文件的教学例，说明新文件产生新 dataset version，旧任务继续引用固定版本。
4. 澄清 FSDP/ZeRO 按策略分片的对象可能不同；把混合精度 Adam 的教学假设算成约 18 字节/参数，并手算 70B 参数约 1.26 TB、64 路理想分片约 19.7 GB/GPU，明确尚未计激活、缓冲区和拓扑代价。
5. 修订训练 checkpoint 的恢复状态说明：在 optimizer update 边界保存时记录必要状态；仅在允许中途保存时才必须额外保存未完成梯度和 accumulation cursor。
6. 对 checkpoint 恢复增加当前官方实现范围：PyTorch DCP 2.15 公开说明 load-time reshard，同时不承诺跨 PyTorch 版本的 state_dict 向后兼容；DeepSpeed 文档把 Universal Checkpoint 标为开发中。相邻的弹性段落明确新 worker group 与新 rank/world size，不承诺优化轨迹等价。
7. 用 §3.16 的 500-step/2.4 秒/90 秒假设复算 checkpoint 暂停、单次故障平均丢失的工作量，并对比 250-step 间隔；统一 GPUH、资源类型价格、存储时长和 I/O 成本口径。
8. 为 SFT 主案例补齐 1,024 样本×2,048 有效 token、41.94B token、约 873,813 token/s、14 小时 50 分钟、474.7 GPU 小时和假设单价下约 949.4 美元的完整账本；每项输入均标为教学假设。

## 验收记录

- **length_status：** 按连续论证的四个核心单元计数，排除标题、表格、代码/图示围栏、公式、练习、资料和总结；仅计解释性汉字。§§3.0—3.3（任务契约、生命周期和数据版本）**2,083**；§§3.4—3.7（manifest、资源调度和分布式执行）**2,204**；§§3.8—3.10（launcher、checkpoint 和恢复）**1,999**；§§3.11—3.18（实验证据、质量/发布治理和完整案例）**3,510**。均为核心单元，超过 1,800 汉字下限；分组对应同一训练平台案例中的连续因果链。
- **example_status：** 数据版本：1,000 文件加 1 个新文件生成 1,001 文件的新版本。训练状态：任务版本 12 的 queued→running 更新为版本 13，过期的版本 12 写入失败。内存：70×10^9×18 B=1.26×10^12 B=1.26 TB；除以 64 得约 19.7 GB/GPU（十进制，未含激活和通信缓冲区）。Checkpoint：500×2.4=1,200 秒/间隔，90/1,200=7.5% 暂停开销；40 次共 1 小时。间隔改为 250 step 后，平均故障丢失量从 10 分钟降至 5 分钟，保存暂停从 1 小时升至 2 小时。SFT：每次更新 2,097,152 有效 token，20,000 步约 41.94B token；训练/保存/评估时间合计 53,400 秒，资源 474.7 GPU 小时；假设 2 美元/GPU 小时得到约 949.4 美元。所有数值均是明确标注的教学输入，不代表生产测量或当前市场报价。
- **evidence：** PyTorch 2.15 官方文档支持本章关于 DCP 加载时 reshard、state_dict 向后兼容边界及 torchrun worker-group/rank 行为的具体说法；DeepSpeed 当前文档支持其 Universal Checkpoint 的状态标记；MLflow/NCCL 文档只支持其公开功能定义。完整来源、版本和外推边界见上述 research 记录。
- **code_status：** 无可执行章节代码。17 个围栏块中有 13 个 math、1 个 Mermaid、2 个 text 示意和 1 个 YAML 概念 manifest；未运行代码、测试、模型、benchmark 或 PDF。Mermaid 源码人工静态检查通过，GitHub 页面渲染待最终门禁。
- **formula_status：** 13 个 math 围栏配对；花括号逐块平衡；无 `\operatorname`、未转义行内美元符号或 TODO/TBD/`@@`/“小白”。手算步骤及单位已记录在 example_status；GPUH 与成本不重复计费的边界已补充。
- **link_status：** 22 个 Markdown 链接，其中 6 个本地目标全部存在，16 个外链。本轮在线核验 PyTorch Distributed/FSDP/DCP/torchrun 2.15、DeepSpeed 0.19.10、MLflow Tracking 和 NCCL 官方页面；历史外链未全量巡检。
- **visual_status：** 新增 1 个 Mermaid 状态与状态所有权图，使用带引号节点和管道边标签；源码语法人工检查通过，GitHub 页面渲染留待书册最终门禁。
- **sync_status：** 简介与目录的第三章职责、训练 job graph、资源与产物账本、团队隔离摘要仍准确，无需修改；新增第一、四章及第五、二十三册正文链接。第十一册架构审计和全局交接同步更新。`PAPERS.md` 无新论文条目，本章已有 Megatron-LM/ZeRO 来源入口；未更新 PDF。

## 遗留与交接

- Mermaid/MathJax 目标平台渲染与历史外链全量巡检待第十一册最终门禁；本章没有 executable demo，因此无代码运行验证项。
- topology 恢复矩阵、SFT 的 batch/吞吐/保存/评估耗时与单价都是示例输入；实际系统必须按框架版本、模型、集群、训练数据和计费策略替换并验证。
- 下一步按目录进入第四章《设计推理平台》，重点检查训练模型交接、在线请求执行和资源组织与第十三章路由/缓存的边界。全套书修书 goal 保持 active，PDF 不更新。
