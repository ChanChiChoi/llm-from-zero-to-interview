# 第六册第 17 章《Native INT4 与原生量化部署》章级审计

审计日期：2026-11-02。状态：完成定向扫描、正文精修、配套同步和静态门禁；保留既有 §§17.1—17.18 结构。§17.14 Python 教学账本只做静态审阅、未运行；未运行测试、模型、benchmark 或 PDF。

## Scope 与章节职责

- **中心问题**：一个低位权重 artifact 怎样在量化方法、group/scale/pack 布局、kernel、runtime、硬件和模型协议之间保持同一数值语义，并证明容量、任务质量、延迟与回滚收益。
- **章节边界**：第 7 章主讲通用量化对象、校准与质量框架；第 15 章主讲 FP4/MXFP4/NVFP4 数值格式与 scale；第 16 章主讲 FP8 KV 状态；本章聚焦 INT4 权重 artifact 及其 runtime 执行契约。`Native INT4` 在正文中明确定义为工程语境，不冒充统一格式标准；weight artifact schema 与 KV cache/page schema 分开版本化。
- **教学链**：从术语和权重显存账进入 INT4 数学；区分 RTN/GPTQ/AWQ/QAT 的权重生成与 MARLIN 的 kernel 执行；继而检查 manifest、group/scale/pack 和并行切分，再以校准、任务切片、容量/延迟、回退、cache 生命周期和灰度演练完成部署闭环。

## 定向扫描与候选去向

- **日期与范围**：2026-11-02；围绕 PTQ/QAT 的分工、INT4 artifact 的分组与序列化契约、W4A16 执行方式、当前公开 runtime/backend 支持、任务质量分母、weight/KV schema 兼容与发布回滚开展定向核验。阅读 GPTQ、AWQ、LLM-QAT、MARLIN 原始论文/实验表和 TensorRT-LLM 1.0.0、vLLM v0.32.0、torchao 0.17、PyTorch FakeQuantize 2.15 官方资料。该扫描服务于本章 INT4 部署问题，不声称穷尽所有量化算法或 runtime。
- **RTN、GPTQ、AWQ**：作为权重 PTQ 生成路线纳入 §17.4；用相应论文中绑定模型、位宽、数据集的 PPL 对照说明误差补偿价值，明确不能跨设置排名，也不能由算法名称推出后端兼容。
- **LLM-QAT**：作为训练适配路线纳入 §17.4；纠正 W4A8KV4 七项常识任务表中的 RTN/LLM-QAT/全精度平均分为 41.2/63.0/66.2，并贴近引用标明其论文设置，不外推到任意 serving kernel。
- **MARLIN 与 W4A16**：作为执行路径纳入 §§17.4、17.6；把 A10 大矩阵层级结果与 vLLM 端到端结果分开，避免把 kernel 倍率归因于量化质量。TensorRT-LLM 1.0.0 文档只支持该版本公开的在线性层按需反量化描述。
- **vLLM、torchao 与其他官方接口**：在 §17.4.5 和资料边界中列出固定版本配置入口，说明支持矩阵、group size 和 backend 是候选面，不代表特定 artifact/shape/batch 命中快速 kernel。新增事实与证据范围写入 `research/weight-quantization-formats-and-ptq-2026.md`。
- **artifact 兼容与 KV 生命周期**：权重 schema、pack ABI、TP group 边界进入 §§17.5—17.7；独立 KV schema、model semantics、prefix namespace 与版本回滚进入 §§17.12—17.13。未把“tensor shape 相同”或“文件能加载”当作跨版本 cache 可复用证据。
- **高风险质量切片和 mixed precision 路由**：纳入 §§17.8—17.10、17.14—17.15；强调同一个二元任务事件才可聚合，并用高风险切片独立门槛防止总体比例掩盖失败。候选方法综述、FP4 变体和通用 KV 压缩不在本章另开主线，分别由第 7、15、16 章承接。
- 未发现需要改变第六册边界或增设顶层章节的候选；保留本章职责及现有章节顺序。论文性能和质量数据仍绑定原模型、任务、硬件和实现；官方文档结论仍绑定所列 release。

## 正文修改与篇幅门禁

- 补入从第 7、5、15、16 章进入的本地导航；收紧 Native INT4 的工程术语边界及 PTQ/QAT、packed artifact、kernel、激活、KV 和累加 dtype 的区别。
- 校正 AWQ 原论文主方法为 activation-aware 等价缩放，并区分其敏感权重动机实验；校正 LLM-QAT 表格数值；把 GPTQ/AWQ/MARLIN 的结果条件和层级差异贴近来源说明。
- 明确 `weight_artifact_schema_version` 与 `kv_cache_schema_version` 的职责不同；增加 group size/FP16 scale 存储账和二进制补码 nibble 次序例子，说明相同 shape/文件长度不能证明 scale/pack 解释兼容。
- 补充 INT4 payload、group scales、分片边界、公共显存、可用率、TP 取整、N+1 和双版本路由的手算；容量下界不写成生产卡数。加入质量同分母切片、低样本门槛舍入、kernel 覆盖率与任务成功率分账，以及 cache 版本隔离和会话连续性案例。
- §17.14 demo 保留为小型账本而非模型/硬件 benchmark；说明输入、输出、Python 标准库依赖和运行命令，标为静态审阅未运行，预期输出明确标为手算。质量门槛、scale metadata、kernel ratio 均声明为合成输入。
- 解释性汉字静态计数：排除标题、表格、代码、公式、行内代码、链接标签/URL、状态标记与空白；以连续教学链为单元，质量另由人工核对。

| 教学单元 | 汉字数 | 类型 | 下限 | 组织理由 |
| --- | ---: | --- | ---: | --- |
| 导读与 §§17.1—17.3 | 2,121 | 核心 | 1,800 | 从术语和权重/KV 成本进入对称/非对称量化、group 与 pack 数值推演。 |
| §17.4 | 1,897 | 核心 | 1,800 | 连起 PTQ/QAT 机制、代表研究结果和算法/执行层边界。 |
| §§17.5—17.7 | 1,873 | 核心 | 1,800 | 以联合 manifest、group/scale/pack ABI 和并行切分串起加载、数值与执行兼容。 |
| §§17.8—17.10 | 1,846 | 核心 | 1,800 | 从校准分布推到任务分母、质量/回退归因、prefill/decode 与容量/卡数边界。 |
| §§17.11—17.13 | 1,924 | 核心 | 1,800 | 连接低精度组合、运行时回退、schema 版本、灰度与整套回滚。 |
| §17.14 | 1,066 | 辅助 | 1,000 | 用一个有界标准库账本示范容量、切片质量和动作分支的手算及限制。 |
| §17.15 | 1,865 | 核心 | 1,800 | 用企业代码助手综合容量、任务分类、双版本成本、cache、回滚和验收。 |
| §§17.16—17.18 | — | 非独立内容 | 不适用 | 误区部分作为定位入口，资料和小结不承担独立主讲。 |

## 算例、公式、代码、链接与视觉状态

- **手算核对**：400 GiB BF16 权重的 INT4 payload 100 GiB，加 15 GiB scale/metadata/未量化层后为 115 GiB；公共项 50 GiB，两个 worker 总账 450/165 GiB；80 GiB 卡按 85% 为 68 GiB 可用，容量下界为 7/3，TP4 取整为 8/4，未含故障余量。复核权重相关空间减少率 71.25%；group 例 256 个权重 G128/G64 加 FP16 scale 为 132/136 B；pack 码值 `[-7,-4,2,5]` 为补码 nibble `[9,C,2,5]`，低 nibble 先存得到 `C9 52`。分片边界 index 4 保持 G4 组完整，index 2 拆组。
- **质量和路由算例**：1,000 条同口径请求中 `800×0.975+200×0.965=973`，整体 0.973 仍不能抵消代码/工具切片 0.965 低于 0.970；另给出统一验收事件下 900/100 分布掩盖高风险切片，以及 2,000 请求中 1 个成功差异对 0.970 门槛的影响。
- **生命周期与容量算例**：拆分 weight artifact schema 和 KV schema；展示 artifact 改变需独立 prefix namespace、固定 prefix 回放和回滚证据。双 worker 池示例为 8+4=12 张卡基础拓扑；路由误判示例 `200×5%=10`，再按假设 30% 失败率得到期望 3 个失败。
- **§17.14 demo 手算**：总显存 450/165 GiB、权重相关空间减少率 0.7125、容量下界 7/3；chat 通过、code_json 失败、合成 kernel 覆盖率不触发回退诊断、卡数拓扑检查触发，输出 `selective_int4_rollout`。代码原文未执行，未检查目标 GPU 或真实模型。
- **公式/围栏**：全章 76 个 `~~~` fence marker 配成 38 对，其中 28 个 math block 花括号平衡；围栏外未发现未配对美元符号。`\operatorname`、尖括号下标和 HTML 公式扫描无命中。
- **链接**：32 个 Markdown 链接，4 个本地目标均存在、28 个外链未逐条全量复核。对 GPTQ、AWQ、LLM-QAT、MARLIN 论文及 TensorRT-LLM 1.0.0、vLLM v0.32.0、torchao 0.17、PyTorch 2.15 官方页面做了定向核验。没有宣称所有外链可达。
- **代码状态**：1 个独立 Python 教学代码块显示“静态审阅，未运行（2026-11-02）”；周边状态/输入/输出/运行方法/手算预期/边界一致。本轮没有运行代码、测试或 benchmark。
- **视觉表达**：使用请求数据流、流程文本、对照表、manifest 字段和状态轨迹表达因果/依赖；本章未增加 Mermaid，现有关系可由表格和逐步案例核对。
- `git diff --check` 通过；没有生成或更新 PDF。

## 配套同步与遗留风险

- `research/weight-quantization-formats-and-ptq-2026.md` 记录指定版本官方接口、GPTQ/AWQ/LLM-QAT/MARLIN 论文条件和证据边界，只放来源事实，不放审阅日志。
- `EXERCISES.md` 补入 group-size/scale 存储、pack/shard、方法与 kernel 证据分层、同口径高风险切片和 schema 灰度练习；`INTERVIEW_BANK.md` 更新 GPTQ/AWQ 固定版本链接并增加 weight/KV schema 与切片分母问题；`KNOWLEDGE_GRAPH.md` 区分 weight artifact schema 与 KV cache schema。
- `PAPERS.md` 已有 GPTQ、AWQ、LLM-QAT、MARLIN 来源条目；`GLOSSARY_EN_ZH.md` 已有 Native INT4 和 MARLIN 定义，本轮不重复增加同义条目。简介与目录已覆盖本章职责和入口，无结构变化。
- **遗留风险**：28 个外链未全量逐一在线检查；runtime 文档与支持矩阵会变化，部署需重新锁定硬件、release、backend、model shape 和 artifact。没有目标 GPU kernel 覆盖、模型质量、端到端延迟、吞吐或 cache 迁移的本地实测；全部教学数字仍按正文逐项标注为论文条件或合成假设。章节代码未运行。
- 本章门禁完成。第六册下一步处理第 18 章《GPT-6 Astra 长上下文与工具预算》；第六册连续复读/最终门禁及全套书其余章节仍未完成，修书 goal 保持 active。

## 整册连续复读补充：2026-11-02

第 17 章沿学习路径完成连续复读，并对照本记录复核容量与卡数账本、group/scale/pack 手算、PTQ/QAT 与 kernel 证据边界、同分母任务质量、fallback 覆盖、双版本发布和 demo 输出；未发现需改正文或调整第 5、7、15、16 章边界的问题。代码仍未运行；第六册连续复读下一步进入第 18 章，最终门禁尚未完成。
