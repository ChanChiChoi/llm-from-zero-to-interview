# 第六册第 8 章《并行推理与多 GPU 服务》章级审计

审计日期：2026-11-02。状态：完成章级定向扫描、正文精修、配套同步与静态门禁。保留原有第 1—16 节顺序，新增 §1.3 容量算例；代码只做静态审阅，未运行。本章没有运行测试、模型、benchmark 或 PDF，也没有更新 PDF。

## Scope 与章节职责

- **中心问题**：怎样按权重/请求状态容量、层内关键路径、工作负载和故障目标，选择 TP、PP、EP、多副本或 Prefill/Decode 资源切分，并保证 KV 所有权、迁移与恢复语义可操作。
- **本章职责**：承接第 3 章的 KV 容量与生命周期、第 5 章的 runtime 能力、第 6 章的调度、第 7 章的压缩状态；讲清多 GPU 拓扑、并行维度组合、expert 负载、路由、跨节点通信、状态迁移和故障边界。runtime 源码实现交第二十四册，集群拓扑/网络平台治理交第二十三册。
- **结构决定**：保留第 1—16 节及原有顺序，新增 §1.3“用一组容量数字区分分片和复制”。没有重排、拆分或删除现有章节。简介与目录第 8 章摘要已同步加入 P/D 资源切分，和正文职责保持一致。

## 章级扫描与候选去向

- **范围与资料类型**：2026-11-02 定向扫描。围绕 TP/PP/EP 和组合并行、专家热点与冗余、请求级副本路由、跨节点 collective/all-to-all、KV 分片/迁移/故障恢复、P/D 解耦及 KV handoff 检索并阅读原始论文和官方文档。研究记录中的论文版本、阅读位置、来源支持事实和证据边界见 `research/inference-serving-metrics-and-disaggregation-2026.md`、`research/kv-cache-capacity-paging-and-compression-2026.md`；本章记录编辑去向，不复制研究笔记。
- **TP、PP、EP 与混合并行**：作为本章核心，分别解释张量/层/expert 的切分对象和通信路径；以矩阵、stage 时间、MoE 路由和 TP×DP×EP 例子建立可复算账本。vLLM 动态文档仅用于说明其访问日公开 backend/组合配置，不外推为其他 runtime 的共同实现。
- **多副本与请求路由**：保留为水平容量和服务隔离主线。路由需估 token/队列/KV/locality 成本，并先执行容量、健康和租户硬门槛；不把专家重映射与副本级请求调度混为同一种均衡。
- **跨节点通信、KV 所有权与故障恢复**：保留为部署闭环，连接逻辑 payload、实际通信测量、状态兼容、owner fencing、客户端流游标、重算/迁移及外部副作用幂等。平台级 fabric 设计和框架源码细节分别交第二十三、二十四册。
- **P/D 解耦与选择性 KV 传输**：纳入 §12.1 作为新的资源切分轴，不新增章节。计算 profile 与完整 KV transfer 账本共同决定是否解耦；按共置/chunked prefill、DistServe、Mooncake、TaiChi、SmartGen 对照各自问题和指标。SpectrumKV 作为传输表示与质量边界的连接入口，量化主体仍指向第 7 章。
- **扫描边界**：论文和访问日动态文档的结果不代表目标 workload、GPU 或 runtime 的实测；未据此增加新的书册主题或章级主线。引用中的当前支持条件需在目标 release 固定后复核。

## 篇幅、例子与代码状态

解释性汉字静态计数口径：排除标题、表格、代码围栏、公式、行内代码、Markdown 链接文本、URL 和代码状态标记，仅计中文汉字。核心单元下限 1,800，独立辅助单元下限 1,000。

| 教学单元 | 汉字数 | 类型 | 下限 | 状态与合并理由 |
| --- | ---: | --- | ---: | --- |
| 开场与 §1 容量/服务约束 | 1,950 | 核心连续链 | 1,800 | 通过；开场定义四种切法并提出问题，§1 沿容量、吞吐和可用性推到分片/复制的数值边界。 |
| §§2—4 TP、PP 与拓扑选择 | 2,277 | 核心连续比较链 | 1,800 | 通过；从张量切分机制、流水线 stage 到同一预算下的选择条件连续推进。 |
| §5 Expert Parallel for MoE | 1,870 | 核心 | 1,800 | 通过；包括 token assignment、expert/rank 偏斜、dispatch payload、capacity factor、EPLB 和组合并行。 |
| §§6—7 多副本与负载均衡 | 2,008 | 核心连续链 | 1,800 | 通过；先回答副本容量/故障域，再回答请求成本、路由分数和路由器/副本内调度器边界。 |
| §§8—11 通信、KV 状态与故障恢复 | 2,968 | 核心连续链 | 1,800 | 通过；从通信关键路径推进到状态布局、迁移成本、故障重试与唯一 owner 交接。 |
| §12 并行推理选型与 P/D | 2,090 | 核心 | 1,800 | 通过；含 P/D 计算容量、KV 字节率、TTFT、SLO goodput 和方案证据边界。 |
| §13 Python toy 账本说明 | 1,157 | 辅助代码演示 | 1,000 | 通过；讲解容量、逻辑通信 payload 和简化路由；代码不替代手算，也不作为性能预测器。 |
| §14 八卡综合部署案例 | 1,897 | 核心综合案例 | 1,800 | 通过；统一比较容量、混合长度、通信、队列和故障余量。 |
| §15 参考资料、§16 小结 | 500、293 | 参考/收束 | 不适用 | 不承担独立教学单元，不单独计下限。 |

**数例与状态推演**：70B FP16/BF16 权重 130.39 GiB；标准全注意力 80 层、8 KV heads、head dimension 128、BF16 时，4,096-token 单请求 KV 为 1.25 GiB、24 条为 30 GiB；TP=4/8 与 4K/8K/32K 的每卡容量账标明均匀切分假设。TP 的 4×8 与 8×16 小矩阵分别算列切分和行切分，并核对 512 个乘加位置/每 rank 128 个；24×8,192 BF16 activation 为 0.375 MiB，80 层、每层两次逻辑 collective 输入为 60 MiB/rank/decode iteration，未冒充 fabric 收发字节。PP 教学时间 11+0.5+13=24.5 ms，区分单请求延迟与有重叠时接近 13 ms 的稳定输出间隔。

MoE 例子使用 32 token、top-2、8 expert，合计 64 assignments；偏斜向量合计仍为 64、max/mean=2。hidden=4,096 的 BF16 assignment payload 为 8 KiB，dispatch 逻辑量 512 KiB；capacity factor 1.5 得到每 expert 名义 12 个槽，并明确超出部分不必然丢弃。4 个 EP rank 汇总为 28/18/11/7，max/mean=1.75。路由例手算 `request_cost=856`，候选分数约 3,617.85 与 1,209.5，选 replica 1；新增硬过滤会排除已达 active sequence 上限的副本，预期输出没有变化。

状态迁移例子手算 1.25 GiB / 20 GiB/s = 62.5 ms，加 15 ms 固定开销为 77.5 ms；512-token 状态约 0.15625 GiB、传输约 7.8 ms、合计约 22.8 ms，并与重算时间比较。P/D 例子手算 40 req/s 的 P/D token demand、15% 余量、44 req/s 时的离散扩容，以及完整 KV 传输 50 GiB/s 对 25 Gbit/s 链路约 2.91 GiB/s 理论上限；明确完整传输假设未扣 prefix-cache 复用。TTFT 例 180+50+77.5=307.5 ms，超过 300 ms 目标。八卡案例另算 19 条 4K 与 5 条 32K 的混合 KV 73.75 GiB、TP4 每卡简化账本约 59.04 GiB，并推演一个四卡副本组故障后名义活动容量由 48 降至 24。

- **代码状态**：§13 仅使用 Python 标准库，代码附近标注“静态审阅，未运行（2026-11-02）”，输出标注为“手算预期输出（未执行）”。本次增加 active-sequence 容量筛选并手工核对无候选时返回 `None`；正常例子仍选 replica 1。未执行代码、测试或 benchmark；KV watermark、租户配额、健康状态和 prefix locality 明确列为代码未模拟的硬门槛。
- **公式状态**：静态检查共 30 对围栏（26 个 math、1 个 Python 和其他 text/示意块）；围栏闭合。数学花括号配对；禁用 `\operatorname`、`(` / `)`、`[` / `]`、尖括号下标、`	ext` 内代码下划线和无反斜线 `qquad` 检查无命中。关键公式都给出变量/假设及相应手算值。
- **链接状态**：29 个 Markdown 链接；2 个相对本地链接目标存在，27 个外链未逐条在线可达性复查。关键论文与动态官方文档按研究记录阅读并限定证据范围；部署前仍需固定目标版本。
- **视觉表达**：用 TP/PP 方案表、MoE 和副本容量表、状态布局表、请求恢复表、owner 交接表及 P/D 研究对照表承载布局/取舍/状态关系，正文解释了读表条件；本章没有额外加入 Mermaid 图。
- **静态门禁**：章节 workflow residue 搜索无命中；`git diff --check` 通过。未执行章节代码、测试、模型、benchmark 或 PDF。

## 配套同步

- `PAPERS.md` 已有 DistServe、Mooncake、TaiChi、SmartGen、SpectrumKV、DeepEP/EPLB 与 P/D 文档入口；本次仅把 SGLang P/D 条目更新为官方动态文档 URL，并记录 2026-11-02 访问日。
- `EXERCISES.md` 已有 35E23—35E31、35E44 等资源画像、P/D、路由、跨节点网络和分布式 KV 练习；`INTERVIEW_BANK.md` 已有 40ZZZZL37、40ZZZZL57、40ZZZZL58 等 P/D 动机、router 与 TP/PP/KV 题目；没有新增重复题目。
- `GLOSSARY_EN_ZH.md` 已覆盖 Tensor/Pipeline/Expert Parallel、DeepEP、EPLB、P/D transfer 与相关术语；`KNOWLEDGE_GRAPH.md` 已有 serving parallelism、P/D resource profile、KV transfer 和 cross-node network 子链。本轮未重复添加词条或节点。
- 第六册 `简介.md` 与 `目录.md` 的第 8 章摘要已加入 P/D 资源切分；没有章节拆合或顺序变化。

## 遗留边界与交接

- 代码没有在 Python 环境运行；没有目标 GPU/runtime 的实际显存、collective、TTFT/TPOT、吞吐、质量或故障演练数据。静态账本不能作为部署配置或性能保证。
- 外链没有逐条检查可达性；vLLM/SGLang 主分支/current 文档会变化，固定目标 release、backend、硬件和拓扑后要重新确认公开支持条件。
- P/D 的 50 GiB/s 示例假设每请求完整生成并传输 prompt KV；prefix 命中、分层 cache、压缩、流式 transfer 和 overlap 会改变实际字节与关键路径，需要从目标 trace 测量。
- 八卡案例对权重、KV 和 reserve 做均匀简化分摊；没有完整描述模型未切分层、workspace、allocator rounding、rank 不均衡、共享故障域或多租户在线状态。
- **下一步入口**：按第六册顺序处理第 9 章《RAG 与 Agent 部署》。第 9—18 章、全册连续复读与最终门禁仍未完成；全套书修书 goal 保持 active。
