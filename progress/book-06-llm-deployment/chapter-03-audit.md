# 第六册第 3 章《KV Cache 与内存管理》章级审计

审计日期：2026-11-02。状态：完成章级定向扫描、正文精修、配套同步和静态门禁；未执行章节代码、测试、模型、benchmark 或 PDF。修改决定和状态留在 `progress/`；研究事实与来源证据见 `research/kv-cache-capacity-paging-and-compression-2026.md`。

## Scope 与章节职责

- 通读并精修 `book-06-llm-deployment/chapters/03-kv-cache与内存管理.md`，保留 3.1—3.17 章节编号和文件入口。
- 中心问题：模型保存哪些历史状态、怎样从模型结构和请求长度估算容量、动态分配如何处理碎片与共享、低精度和淘汰带来哪些质量/性能边界、怎样由 trace 区分活动 cache 与可复用工作集并定位 OOM。
- 与前后章的职责：承接第 2 章 prefill/decode、指标和内存测量；为第 6 章调度、第 7 章量化、第 8 章并行、第 12 章成本和第 16 章 KV 量化提供账本。框架源码实现下交第二十四册；RAG/Agent 请求工作流不在本章重讲。

## 章级扫描与候选去向

- **扫描日期与范围**：2026-11-02。围绕单位 token 状态和混合 attention layout、池预算/分页、KV 量化、prefix hash 与多租户隔离、活动状态 eviction、可复用前缀工作集和分布式 KV 传输开展定向核验；资料类型包括原始论文、固定版本 runtime 文档、发布方技术资料和动态官方页面。该扫描用于本章取舍，不是对 KV 系统研究的穷尽检索。
- **vAttention**：纳入 §3.8.6，作为 PagedAttention 的另一种显存地址管理路线；固定到 ASPLOS ’25/arXiv v3，并把 1.99×、1.18×、1.23×各自绑定论文模型、A100 数量、vLLM v0.2.7 与不同 kernel baseline。明确 1.99×同时改变 attention kernel，不能单独归因于 VMM。
- **KVSET 工作集**：纳入 §3.11.2，因为 trace 驱动的 reusable-prefix 容量与 active request payload 是两类不同账本。明确 coverage 是相对无限 cache 理论 hit rate 的请求覆盖口径，不是普通 request hit rate；24K 内部 trace 的容量不外推为通用建议。
- **UltraQuant 4-bit KV**：纳入 §3.9 作为低位宽 cache 的条件化系统案例；绑定 v3、MiniMax M2.5/Qwen3-235B、2×MI355X、TP=2 和并发 6—48 的 TTFT-qualified goodput 报告。明确 4-bit、指定 AMD kernel 和容量压力共同影响结果，不能推广为其他硬件或低负载场景的普遍加速。
- **Tessera 检索式复用**：在 §3.10.2 说明普通 prefix hash 与非前缀 context-unit reuse 的边界。它是 arXiv v1 原型，所述范围限于 RoPE 全注意力、逐 token KV 与论文 RAG/Agent trace；hybrid attention、MLA、在线授权撤销没有在该评估覆盖。更完整的 RAG/Agent 工作流路由到第 9 章。
- **SmartGen 选择性 KV 传输**：扫描后不展开在本章；其核心是 P/D 分离系统中的跨节点状态传输、TTST/TBT 与网络重叠，不是本章的 block allocator 或容量公式，留给第 8 章并行推理与状态传递审视。
- **H₂O、StreamingLLM 与租户侧信道**：纳入 §3.11.3 对照有损历史保留机制，并在 §3.10.2 引用 NDSS 2025 研究限定 cache-sharing 时序侧信道的 threat model。TensorRT-LLM 采用访问日期标记；leaf-only eviction 的页面 caveat 未绑定 manager/release，不外推到所有模式。vLLM v0.32.0 的 Paged Attention 页面明确是历史论文说明，不用作当前代码实现证据。
- 没有新增、拆分或重排章节；新主题均纳入已有容量、量化、复用或淘汰主线。

## 正文修改与解释链

- 将 KV 数值载荷与显存池拆开计账，区分池内/池外 metadata、尾槽碎片、free-but-allocated block 和 headroom，注明避免重复扣账的方法。
- 用相同三条请求长度比较 64/128/256-token block 的 block 数、尾槽和 MiB，并补上 block table 地址换算、共享尾块 copy-on-write 与按请求长度 admission 的关系。
- 量化部分从理想字节比例、scale 粒度到反量化误差，再走到 K/V 对 attention 的不同影响；加入整组/分组 scale 的数值对照、V 加权输出误差、池占用示例和分桶评测流程。
- 将 prefix hash、cache salt、full-block 复用、active/reusable 生命周期和多租户授权连成请求状态轨迹；区分普通前缀命中、Tessera 式检索复用和 KVSET trace 工作集。
- 长上下文部分区分滑窗、摘要、检索刷新、offload/recompute、有损 eviction 和量化；H₂O/StreamingLLM 的模型范围及 sink-token 示例限定为论文设置。
- OOM 诊断增加完整 12 GiB 教学预算，并展示 3,000 active token 分成多个请求后须逐条 block-rounding；容量代码演示的布尔判断明确只适用于脚本输入。
- 修正 GQA 的边界表述：必须使用已经按 GQA 训练/转换并通过质量回归的模型/权重，不能把它写成部署期开关。

## 篇幅与数例门禁

计数为解释性汉字静态计数：排除标题、代码/数学/文本围栏、表格、链接文本和行内代码；按连续教学链计数，辅助内容必须明确链接完整主讲。

| 连续单元 | 类型 | 汉字数 | 门槛 | 判断 |
| --- | --- | ---: | ---: | --- |
| 章首导读 + §§3.1—3.4：状态动机、形状、生命周期与计算代价 | 核心 | 2,104 | 1,800 | 通过；包含 8K cache 手算 |
| §§3.5—3.7：MHA/GQA、混合状态和显存/并行预算 | 核心 | 1,896 | 1,800 | 通过；含 global/windowed layer 与 TP 逐步估算 |
| §3.8：PagedAttention、槽位、地址换算、共享和 COW | 核心 | 1,935 | 1,800 | 通过；包含 3-request block size、MiB 与物理 slot 手算 |
| §3.9：量化、scale、K/V 误差和系统评估 | 核心 | 1,971 | 1,800 | 通过；包含 INT8、分组 scale、attention 输出与占用手算 |
| §§3.10—3.11：prefix 生命周期、长上下文策略和 reusable working set | 核心 | 2,403 | 1,800 | 通过；包含 prefix 命中轨迹、H₂O/StreamingLLM toy trace 与 1M 字节换算 |
| §§3.12—3.13：OOM 诊断与内存审计 demo | 辅助案例链 | 1,140 | 1,000 | 通过；含池预算、逐请求取整和静态推导输出 |
| §3.14：常见误解速查卡 | 非独立单元 | 518 | — | 通过；各项链接到相应主讲段，不以速查卡替代教学 |
| §3.15：长上下文多租户 cache 设计 | 辅助综合案例 | 1,135 | 1,000 | 通过；含隔离 salt 的 128/384 MiB 代价案例 |
| §§3.16—3.17：资料边界与小结 | 参考/收束 | — | — | 不计为独立知识讲解单元 |

## 例子、公式、代码和链接状态

- **example_status**：§3.2.3 以 L=32、H_kv=8、D_h=128、BF16、8,192 token 手算 1 GiB；§§3.5—3.7手算 GQA、滑窗混合层和 rank 容量；§3.8 比较 5,460 active token 在不同 block size 下的槽位/尾部；§3.9从向量量化与两值 attention 手算误差；§§3.10—3.11跟踪 full-block hit、淘汰和百万 token 容量；§3.12算池预算；§3.15计算租户 salt 的重复模板代价。教学数字已标注假设，不冒充线上测量。
- **formula_status**：35 个 display-math 围栏的花括号配对；禁用 `\operatorname`、尖括号数学下标和逐行不配对 `$` 扫描均为 0。新增 pool/payload 和 mixed-layer/rank 公式的变量与扣账范围已在正文解释。
- **code_status**：§3.13 唯一 Python demo 标为“静态审阅，未运行（2026-11-02）”。INT8 算法按原始 1 byte/value 计算，避免将舍入后的 BF16 显示值再除以 2；预期输出为手算推导。代码块和异常路径没有实际执行，本轮也没有运行代码、测试、benchmark 或模型。
- **link_status**：43 个 Markdown 链接，35 个外链、8 个章节内锚点；静态检查无失效锚点或重复 heading slug。外链未逐个在线可达性复查；固定论文版本与 runtime 文档边界记录在对应引用和 research 笔记。
- **fence_status**：80 个围栏标记配成 40 对：35 math、4 text、1 Python。

## 配套同步与遗留风险

- `简介.md` 与 `目录.md` 的第 3 章摘要已说明 active-cache 估算、reusable-prefix 工作集、分页、量化、淘汰和长上下文并发。
- `GLOSSARY_EN_ZH.md` 新增 KV Cache Working Set；`KNOWLEDGE_GRAPH.md` 在 KV Capacity Admission → Paged Block Management → Prefix Cache Reuse 后加入该节点；`PAPERS.md` 绑定 PagedAttention v1、vAttention v3、KVSET v1、UltraQuant v3 和 Tessera v1。
- `EXERCISES.md` 已有 80F 覆盖普通 prefix 与 Tessera 式跨请求 KV 复用、工作集容量和联合验收，本轮未重复新增练习。`INTERVIEW_BANK.md` 新增 31R，考查 active payload 与 prefix working set 的口径差异。
- 研究来源、阅读位置与证据范围在 `research/kv-cache-capacity-paging-and-compression-2026.md`；研究文件只保存事实和证据，不记录本审计进度。
- 未全量访问 33 个外链；TensorRT-LLM 页面为动态资料且 caveat 未绑定 manager/release。真实 runtime 的 cache allocator、显存池驻留方式、量化质量和性能仍须按目标模型、release、硬件和 workload 单独核验。§3.13 是静态教学代码，不构成本地复现。
- 下一步：处理第六册第 4 章《解码策略与生成控制》。
