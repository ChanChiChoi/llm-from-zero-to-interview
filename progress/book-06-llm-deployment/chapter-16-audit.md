# 第六册第 16 章《FP8 KV Cache》章级审计

审计日期：2026-11-02。状态：完成定向扫描、正文精修、配套同步和静态门禁；保留原有 §§1—13，在 §5 增加 §5.5 生命周期案例。Python 示例只做静态审阅、未运行；没有运行测试、模型、benchmark 或 PDF。

## Scope 与章节职责

- **中心问题**：FP8 如何改变在线 KV 状态的字节数、数值误差和 decode 路径；scale、page、attention backend 和请求生命周期必须满足什么条件，容量收益才可安全进入服务。
- **章节边界**：第 3 章主讲活动 KV 容量、分页、复用和淘汰；第 7 章主讲通用量化对象、校准与质量框架；本章聚焦 FP8 KV 的表示、scale、K/V 误差传播、scale/page 生命周期、容量/性能评测及发布回滚。第 15 章主讲 FP4 权重，第 17 章主讲 native INT4 artifact，第 5 章主讲 runtime 兼容性。导读已新增这些章节的本地导航链接；简介和目录原有第 16 章摘要与本章职责一致，无须修改。
- **教学链**：由 active KV 字节账推进到 FP8 编码与 scale，再解释 K/V 的不同误差路径；随后沿 page 生命周期和 allocator 协作处理状态兼容，接着从请求路由、长上下文评估和 runtime 报告推导容量/性能边界，最后落实到 manifest、灰度、回滚与合成质量门禁。

## 定向扫描与候选去向

- **扫描日期与范围**：2026-11-02；围绕 FP8 E4M3/E5M2 编码、scale 粒度/校准、K/V 分布差异、长上下文 attention 精度、per-head 配置、分页 cache 生命周期、decode break-even 和回滚条件开展定向核验。阅读 OCP FP8 规范、KIVI/KVQuant 论文、vLLM v0.32.0 量化 KV 文档、2026-04-22 vLLM 工程验证报告，以及 PyTorch/ROCm 和 TensorRT-LLM/Transformer Engine 文档。该扫描服务于本章，不声称穷尽所有 KV 压缩或 runtime 实现。
- **OCP E4M3/E5M2 与 FNUZ**：纳入 §§3.1—3.3，区分编码范围/特殊值与 scale 选择和维护；OCP 格式不被写成 KV runtime 契约。Per-tensor、per-head、per-block 的容量/误差/元数据取舍均保留实现条件。
- **KIVI 与 KVQuant**：保留于 §4 的方法对照，说明 K/V 的量化轴、outlier、校准和流式写入约束；二者的低 bit 论文实验不被外推成 FP8 serving 默认值或性能保证。
- **FP8 cache 生命周期与 allocator**：纳入 §§5—6。第 3 章所有权仍负责物理页、池和工作集；本章解释 payload 与 scale 绑定、prefix sharing、copy-on-write、swap/restore、speculative rollback 和 page schema 兼容。新增 §5.5 合成轨迹，明确数值解释和状态生命周期的分工。
- **vLLM 2026 FP8 KV/attention 报告**：纳入 §8.4，并在 `research/kv-cache-capacity-paging-and-compression-2026.md` 新增来源记录。正文说明 FlashAttention 3 路径同时在 FP8 域执行 QK/score-value attention，不能把实验解释成纯 KV 存储消融；把 128K NIAH 的 91%/13%/89% 绑定 H100、Llama-3.1-8B 和特定累加路径；明确报告的**质量评测**使用未校准 per-tensor `scale=1.0`，不外推为 calibrated scale 结论。补入报告 ITL 拟合的手算 break-even，并区分 ITL slope、TTFT 和并发吞吐。
- **vLLM v0.32.0 与其它当前接口**：per-head scale、未校准默认 scale 和敏感层跳过均绑定 v0.32.0 官方文档；LLM Compressor 仍为动态页面，正文标明访问日期并提示部署时锁定 release/backend/hardware。TensorRT-LLM 文档用于说明不同 cache dtype 配置入口，不外推到其他 runtime。
- **低于 8 bit 的 cache 方案**：UltraQuant、TurboQuant 等留在第 3/7 章和研究索引；本章不增加低 bit 方法综述，以免冲淡 FP8 在线状态和数值路径的中心问题。
- 没有新增、删除或重排顶层章节；所有纳入/排除的来源事实和证据边界只写入 `research/kv-cache-capacity-paging-and-compression-2026.md`，正文修改决定和审阅状态只记在本文件。

## 正文修改与篇幅门禁

- 补足每 token 公式的逐步换算、GQA/MHA 的 KV head 差异和长短请求混合的容量影响；增加 9 条 2K 请求加 1 条 128K 请求的手算，说明以短请求代表总体会低估显存。
- 增加 per-head scale 的合成例子，逐步比较共享 scale 下 E4M3 最小 subnormal 附近的相对误差与独立 head scale；在 K/V 标量例子后补 softmax 中间量及一阶误差分解。
- 把 cache scale revision 当作 payload 解释条件；增加 prefix 共享、未闭合 scale group、swap/restore、引用计数和 speculative rejection 的状态轨迹，以及 metadata 类别和迁移前后校验方法。没有重讲第 3 章 allocator 实现。
- 在容量例中明确 32 GiB BF16 → 17.28 GiB FP8(含 8% 合成 overhead) 代表 46% payload 账本减少、约 1.85× 理论请求容量；40 GiB 池、32K token 请求的理想上界为 5 条 BF16 与 9 条 FP8。性能报告的显示系数给出约 7,000 token 手算 break-even，解释与报告未舍入拟合值 7,010 的差异。
- 扩写 manifest identity、灰度分阶段证据、兼容性转换/排空/重算策略；§10 demo 标明输入/输出、运行方法、标准库依赖、静态未运行状态和手算预期。示例代码本身未修改；`scale_overhead=0.08` 与质量分继续显式标为合成值。
- 用本章正文内部的 §2、§4、§5/6、§8、§9 指针让 §11 常见误区作为入口索引；§12 为来源边界，§13 为回顾收束，不单独承担完整教学。

汉字计数口径：只计连贯中文解释文字，排除标题、表格、代码围栏、公式、行内代码、Markdown 链接文字/URL、状态标记和空白；按连续论证计数。

| 教学单元 | 汉字数 | 类型 | 下限 | 组织理由 |
| --- | ---: | --- | ---: | --- |
| 导读与 §§1—2：KV 状态动机、容量公式和负载分布 | 1,914 | 核心 | 1,800 | 从动态状态成本推到模型结构、请求长度和池预算；包含每 token 与偏斜请求分布手算。 |
| §§3—4：FP8/scale 与 K/V 误差传播 | 1,938 | 核心 | 1,800 | 从码本和粒度推到 K 改权重、V 改内容，再到量化路线和 K/V 消融。 |
| §§5—6：状态生命周期与 allocator 协作 | 1,907 | 核心 | 1,800 | 以 scale/page 不变量贯穿 append、share、swap、restore、speculative rollback 和诊断。 |
| §§7—8：容量策略与质量/性能评估 | 1,911 | 核心 | 1,800 | 合并请求策略、成本和报告评估，分别解释容量比、ITL break-even 和并发吞吐口径。 |
| §§9—10：manifest、灰度回滚与合成 go/no-go | 1,892 | 核心 | 1,800 | 连成 artifact identity → 分阶段发布 → 回滚/排空 → 容量/质量门禁；代码只演示这一链中的决策环节。 |
| §§11—13：常见误区入口、证据索引与小结 | — | 非独立内容 | 不适用 | 指回 §§2—9 的机制，不替代主讲；来源和小结不承担新主题教学。 |

## 例子、公式、代码与链接状态

- **手算核对**：每 token BF16 KV 为 256 KiB；9×2,048+1×131,072 请求共 149,504 token、约 36.5 GiB 逻辑 BF16 payload；2-head scale 构造例中 E4M3 重构误差约 0.000953；`[1,0]` softmax 从 `[0.731,0.269]` 变为 `[0.668,0.332]`，V 的 `+0.1` 误差带来约 `+0.0731` 输出差；scale 改变后旧最大码 448 从表示 4 变成表示 8；40 GiB pool 请求数上界 5 vs 9；vLLM 报告拟合的手算 break-even 约 7,000 token，报告值约 7,010。
- **§10 demo 静态复核**：输入例手算 BF16 32 GiB、FP8 含 overhead 17.28 GiB、节省率 0.46；chat `0.975≥0.970` 通过，audit `0.890<0.950` 失败，唯一动作是保留 audit 的 BF16 回退，决策字符串为 `selective_enablement`。代码检查边界/类型与输出逻辑，未运行；预期输出为手算结果。
- **公式/围栏**：34 个 `~~~` fence marker 配成 17 对；math 花括号净差 0；正文围栏外未发现美元分隔符。禁用公式写法扫描无命中。
- **链接**：40 个 Markdown 链接：35 个外链、5 个本地目标；5 个本地文件全部存在。OCP、vLLM v0.32.0 页面与 2026 vLLM 工程报告已定向在线核验；其它外链未逐条全量检查。Mermaid 不适用：生命周期已有状态顺序、metadata 列表与症状诊断链表达，本章没有新增需靠图才能理解的分支。
- `git diff --check` 通过；没有构建或更新 PDF。

## 配套同步与遗留风险

- 更新 `research/kv-cache-capacity-paging-and-compression-2026.md` 的 vLLM v0.32 文档事实边界，并增加 2026-04-22 FP8 KV/attention 报告及 ITL/吞吐/质量口径。
- 更新 `EXERCISES.md` 第六册第十六章：修复原研究入口链接，并加入 40 GiB pool 容量上界与 break-even 手算。更新 `INTERVIEW_BANK.md` 对 v0.32.0 的固定引用，加入约 7K break-even 和共享 page scale 变更问题；更新 `PAPERS.md` 的版本链接和能力边界；`KNOWLEDGE_GRAPH.md` 增加 FP8 KV 数值—状态—发布链。`GLOSSARY_EN_ZH.md` 已有 FP8 KV、OFP8、FNUZ 和 two-level accumulation 定义，无需重复新增词条。
- 现有简介/目录的第 16 章摘要和阅读路径已经说明 KV 状态、分页、容量、质量与回滚，未重复改写。
- **遗留风险**：vLLM/LLM Compressor/TensorRT-LLM 文档和支持矩阵会变化；部署前需固定 target release、backend、GPU、scale/calibration 与请求分布。2026 报告是团队工程验证，不是跨 runtime 独立 benchmark；没有目标 GPU 上的质量、延迟、吞吐、显存峰值或迁移实验。35 个外链没有全量逐条复核，教学 8% overhead 与任务分数不代表生产数据。
- 本章已完成静态精修；下一步进入第六册第 17 章《Native INT4 与原生量化部署》。第六册全册连续复读/最终门禁和全套书其它章节仍未完成，修书 goal 保持 active。

## 整册连续复读补充：2026-11-02

第 16 章沿学习路径完成连续复读，并对照本记录复核容量、E4M3 scale、K/V 误差、page/scale 生命周期、FP8 serving 报告的 break-even 与吞吐口径，以及合成 demo 的门禁输出；数值一致。与第 3、7、15 章的职责边界清楚，无需修改正文或导航。代码仍未运行；第六册连续复读下一步进入第 17 章，最终门禁尚未完成。
