# 第五册第 7 章精修记录（2026-11-24）

## 本轮范围与依据

本记录从写作规范模块 02a 起按本轮任务重新审阅，并按第五册第 7 章契约检查正文。本轮不读取或沿用既有 `progress/` 记录；章节顺序按正文从第 7 章继续，不跳章。精修以保留分布式训练主线、纠正版本敏感事实、减少重复并强化证据边界为目标。

## 本轮新审阅结论

- 章节的 DP、ZeRO/FSDP、TP、PP、SP/CP、EP、检查点及故障诊断主线与契约相符；公式与教学账本按所给假设基本自洽。
- DDP 正文和延伸阅读仍引用 PyTorch 2.14，而本轮按 PyTorch 2.15 文档核验；需统一版本引用及文档默认值说明。
- DP 表格“增加吞吐和 global batch”容易被读成吞吐与 batch 必然同时增长；应说明在固定 global batch 与固定每卡 batch 的不同扩展条件。
- 7.11 有部分通用激活重计算介绍与第八册范围相邻；保留其对 FSDP、TP、PP 显存和通信的影响，把泛化讲解压缩到服务本章目标的程度。
- 资源账本代码明确标记为静态审阅，不执行；将审阅日期更新为本轮日期。
- Megatron Core `latest` 文档的 CP 与层级并行说明需标注本轮核验日期，避免把持续更新文档当作固定版本接口。

## 本轮核对的主来源

- PyTorch 2.15 DDP：`https://docs.pytorch.org/docs/2.15/generated/torch.nn.parallel.DistributedDataParallel.html`
- PyTorch 2.15 FSDP2：`https://docs.pytorch.org/docs/2.15/distributed.fsdp.fully_shard.html`
- PyTorch 2.15 Distributed Checkpoint：`https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html`
- PyTorch 2.15 发布说明：`https://pytorch.org/blog/pytorch-2-15-release-blog/`
- Megatron Core 并行策略与 Context Parallelism 官方文档：
  - `https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html`
  - `https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html`
- NCCL 官方文档：`https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html`
- 本章采用的激活重计算、PaLM、DeepSeek-V3 等论文主张按各论文设置限定，详细事实与边界另记于 research 证据文件。

## 修改与验收

- DP 对照表改成明确表达：扩大数据并行处理能力；固定每卡 batch 时可扩大 global batch，避免暗示吞吐与 global batch 必然同时增长。
- 将 DDP 全局 loss 缩放、不等长输入和 bucket 说明，以及推荐阅读链接统一到 PyTorch 2.15 文档。
- 将激活检查点开头改为状态分片与激活显存的分工，并保留分段重算示例、定量研究结果及其配置边界。
- 更新资源账本的静态审阅日期和 Megatron `latest` 文档核对日期为 2026-11-24；toy 代码只读审阅，未运行。
- 新增对应来源证据文件：`research/distributed-training-system-and-api-evidence-2026-11-24.md`。

静态验收：`git diff --check -- book-05-llm-training/chapters/07-分布式训练.md` 通过；Markdown 代码围栏配对；本地链接均可解析；正文不存在遗留的 PyTorch 2.14 DDP 引用。未执行代码、测试或 PDF 构建。

## 后续顺序

本章完成后，从第 8 章开始依次精修，不参考既有 `progress/` 断点跳转。
