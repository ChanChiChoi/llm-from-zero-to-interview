# 第十一册第四章审计：设计推理平台

审阅日期：2026-11-05  
状态：章级扫描、静态精修和源码静态门禁完成；GitHub Mermaid/MathJax 渲染及历史外链全量巡检留待第十一册最终门禁。  
范围：`book-11-system-design-interview/chapters/04-设计推理平台.md` 全章，保留 4.0—4.25 编号。

## 章级扫描与候选去向

- **扫描问题：** 从 QPS 推出输入、输出 token 速率后，容量估算是否保留负载组合、SLO 和 N−1 条件；prefill 与 decode 的资源差异如何引出 continuous batching、chunked prefill 和解耦；KV Cache 公式是否区分完整逻辑状态、并行 rank 本地容量与服务池副本；论文系统收益是否限定实验边界；本章怎样与第六册部署、第十三章路由/缓存、第二十三册平台工程及第二十四册 runtime 实现分工。
- **资料范围与日期：** 2026-11-05，定向阅读 Orca、PagedAttention、FlashAttention、Sarathi-Serve、DistServe、Mooncake、Speculative Decoding 原论文和 2026 年投机解码、prefill 调度预印本；核阅 vLLM APC/量化、Kubernetes Device Plugins/HPA 官方文档。具体新知识、版本、来源支持范围与限制见[推理服务证据记录](../../research/llm-inference-serving-evidence-2026-11-05.md)。研究用于核对机制和适用边界，不是对所有推理框架或生产集群的穷尽调查。
- **扫描候选去向：** 不新增章节。调度算法和 P/D 解耦留在既有 §4.9、§4.11；请求级质量/成本路由及缓存键、失效和策略闭环交第十三章；推理优化基础与部署实施交第六册；runtime 和 allocator 实现交第二十四册；集群平台交第二十三册。

| 候选主题 | 处理位置 | 去向理由与证据边界 |
| --- | --- | --- |
| 固定 chunk size 与 decode deadline 的冲突 | §4.9.3 增补 Sarathi-Serve 与 SLOWeave | 前者用目标 TBT/SLO 和 token budget 构造 stall-free batching；后者按活跃 decode 的下一 token deadline 选 chunk。SLOWeave 的逐轮结论依赖 decode-only 可行及准确、单调的成本预测，不能作为持续过载下的端到端保证。 |
| 解耦后 prefill 池排队、decode 池有余量 | §4.11.5 增补 Kairos | 说明按预测 TTFT/TBT 将 prefill 转派到 decode 节点的研究方向，同时限定 vLLM 0.18.1、A100、DeepSeek-V2-Lite 和特定 trace；保留预测误差、陈旧状态、prefix-cache 局部性及 dispatcher 瓶颈。 |
| FlashAttention 的 IO-aware 机制 | §4.8.1 就近引用原论文 | 支持 tiling、HBM/SRAM 数据移动和 exact attention 机制，不据此推导线上队列、KV 分配或平台 SLA。 |
| Acceptance-aware draft model training | 不扩写到本章；仅留在研究资料的观察范围 | 这是 draft 模型训练目标方向，不直接构成推理平台容量或调度合同；投机解码收益仍按本章 acceptance length 和目标配置实测判断。 |

## 主要修订

1. 开篇界定推理平台在请求工作量、在线资源与恢复上的职责，并链接第六册、第二十四册和第十三章主讲位置。
2. 连通工作负载、prefill/decode 阶段、调度预算、KV 状态、并行 topology、扩缩容、缓存、安全、观测和故障处理；澜沧案例先确定请求类型与 SLO，再选择资源池并计算副本数。
3. 为控制平面/数据平面增加 Mermaid 请求链路图，区分同步请求路径和发布/遥测反馈；组件责任同时保留在契约表中。
4. 将 continuous batching 示例标为不可执行的 text 伪代码；增加 2,048-token prefill、3 个 decode 序列和 12,000 个 KV 槽位的逐轮预算案例。
5. 明确 KV 总量公式描述完整模型逻辑状态，再按 TP/PP rank 的实际 KV heads 与层数计算本地上限；示例得 128 KiB/token 全模型、32 KiB/token/rank、10 GiB/rank 理想 327,680 槽位，并说明 instance 取最紧 rank。
6. 量化部分增加 BF16/INT8 成对任务切片假设；投机解码公式和 acceptance length 案例区分理论加速与真实执行成本。
7. 路由效用公式先经过硬性质量/SLO/区域门槛，再将质量、延迟、成本和安全统一归一化到高分更优的 0—1 区间，并给出 BF16/INT8 的加权代入。
8. 流式输出示例区分已生成、已发送与客户端确认序号，并按取消状态释放 KV；容量规划增加 §4.21.4 和单副本故障后的 N−1 余量。
9. 共享/专用实例决策使用相同月份小时、GPU 小时价格、GPU 数和 SLO 对照；缓存示例区分复用输入前缀状态与复用最终响应。
10. 更新来源与证据边界，引用 vLLM、Kubernetes、FlashAttention、Sarathi-Serve、SLOWeave 和 Kairos 的实际资料，并把新研究事实写入单独 research 记录。

## 验收记录

- **length_status：** 按连续论证链划分六个核心单元；计数只含解释性汉字，排除标题、表格、代码/图示围栏、公式、资料、练习和小结。§§4.0—4.3 **1,965**；§§4.4—4.8 **2,250**；§§4.9—4.11 **3,430**；§§4.12—4.14 **1,952**；§§4.15—4.20 **2,554**；§§4.21—4.22 **1,924**。均高于核心单元 1,800 字下限；分组是同一推理平台案例中的连续解释链。
- **example_status：** 工作负载：Chat/RAG/Agent 合计 3.012M input token/s、506k output token/s，并分到 short/long pool。调度：KV 总额 12,000、当前占用 10,000、空闲 2,000；一个 2,500 槽位请求退出后空闲 4,500，足以接收预留 3,000 的新请求，并可同时处理 2,048 prefill token 和两个 decode token。KV：`2×32×8×128×2=131,072 B=128 KiB/token`；TP=4、每 rank 2 个 KV heads 时为 32 KiB/token/rank；10 GiB 可用容量约 327,680 token slots，按每请求 16,384 槽位得理想 20 个请求。量化：100 个任务对照得 BF16 74%、INT8 68%，差 6 个百分点；该小样本为教学输入，不作显著性结论。投机解码：`k=4`、平均接受长度 `a=2.5`，代入成本/输出 token 的比较式，说明 draft/验证开销可抵消较低的轮数。路由：明确 0—1 归一化和权重后复算 BF16 `0.624`、INT8 `0.521`。流式：服务生成 150、客户端确认至 120 后断开，区分生成、发送、确认及取消释放状态。容量：u=0.65、h=0.10，short 10 个、long 14 个 8-GPU 副本；总计 192 GPU，并逐池检查一个副本故障后剩余容量。共享实例案例统一 730 小时、假设 3 美元/GPU·小时，对照 8 与 16 GPU 的月费用和 P95。
- **evidence：** 论文支持作者提出的机制和对应实验配置，不支持无条件生产收益；官方文档支持公开接口或版本功能，不代替模型/GPU/runtime 的目标基准。SLOWeave 安全陈述条件、Kairos 限制、FlashAttention 的 IO 机制，以及 vLLM APC/量化和 Kubernetes 基础资源行为均在正文就近引用；来源版本与边界见上述 research 记录。
- **code_status：** 39 个围栏块：34 math、1 Mermaid、4 text；无可执行 Python。§4.9.2 text 块为带明确说明的不可运行调度伪代码，静态阅读未运行；Mermaid 源码人工静态核对节点、子图和带标签箭头，GitHub 页面渲染待最终门禁。未运行章节代码、测试、模型或 benchmark。
- **formula_status：** 34 个 math 围栏均配对，花括号平衡；130 个美元标记成对，无 `\operatorname`、TODO/TBD/`@@`/“小白”。关键容量、rank KV、投机成本和加权效用公式均配数值输入与逐步结果；所用容量和质量值明确为教学假设。
- **link_status：** 49 个 Markdown 链接；本地目标均存在。外部链接包括新核验的 arXiv 原文、vLLM 官方动态文档和 Kubernetes 官方文档；动态页面需按未来目标部署版本复核，历史外链全量巡检待本册最终门禁。
- **visual_status：** §4.5.2 使用 1 个 Mermaid 展示控制平面配置发布、数据平面请求快路径和遥测反馈；源代码通过静态语法核对，GitHub 页面渲染待本册最终门禁。
- **sync_status：** `简介.md` 第四章摘要和 `目录.md` 职责/交接仍准确，无需改变章节结构；`PAPERS.md` 推理调度阅读路线已补充 SLOWeave 与 Kairos；第十一册架构审计与全局交接已同步。第十三章继续主讲请求级路由与缓存策略；未更新 PDF。

## 遗留风险与交接

- 澜沧容量、SLO、质量差异、成本和故障均是教学假设；实际部署应按相同请求混合、目标 SLO 和拓扑重做基准。N−1 仅做了单副本算术检查，不替代共享故障域、排队和 KV 水位压测。
- SLOWeave 与 Kairos 均为预印本，相关收益和保证不可跨配置外推；尤其需要重新校准 cost predictor、节点状态和 prefix-cache 局部性。
- Mermaid/MathJax 页面渲染及历史外链全量巡检留待第十一册最终门禁。
- 下一步按目录处理第五章《设计 RAG 系统》，重点检查数据摄取、索引、检索、证据链与第二、九、三册的交接；全套书修书 goal 保持 active，PDF 不更新。
