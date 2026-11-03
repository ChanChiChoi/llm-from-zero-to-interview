# 第六册第 15 章《FP4、MXFP4 与 NVFP4》章级审计

审计日期：2026-11-02。状态：定向扫描、正文精修、配套同步与静态文档门禁完成；保留原有 1—14 节结构。章内 Python 示例按用户要求仅静态审阅、未运行；未运行测试、模型、benchmark 或 PDF。

## Scope 与章节职责

- **中心问题**：同为四位 payload 的 FP4、MXFP4、NVFP4，为什么会有不同的误差、存储和执行行为；如何从码本与 scale 追踪到 artifact、kernel、runtime、质量和发布门禁。
- **章节边界**：承接第 7 章的通用量化对象、校准、误差与质量评估框架，深入 FP4 码本、MXFP4/NVFP4 scale、weight/activation 数据路径及部署兼容。FP8 KV 由第 16 章主讲，native INT4 artifact 由第 17 章主讲；训练 recipe 只解释低精度稳定性的来源，不代替第五册训练主线。正文新增阅读边界段，链接指向存在的章节文件。
- **教学链**：开篇与 §§1—2 解释浮点 payload、scale 和格式身份；§§3—8 从 block scale 误差推进到量化对象、kernel、校准、评估和可复算存储案例；§9 用静态 Python 示例对照码本、scale、MSE 与 metadata；§§10—11 连接 artifact manifest、回退、运行 profile 和发布矩阵；§§12—14 提供误判入口、来源边界和收束。
- **简介/目录**：核对现有第 15 章摘要与“FP4 格式、scale、artifact/kernel 部署差异”职责一致；正文没有增删、拆并章节，不需修改简介和目录。

## 定向扫描与候选去向

扫描日期为 2026-11-02，范围覆盖 2025—2026 的 FP4/MXFP4/NVFP4 scale、近饱和误差补偿、训练及 PTQ recipe、支持矩阵和部署 API。阅读 OCP 规范、Transformer Engine 2.20.2 API/动态文档、相关原始论文和官方支持矩阵；不声称枚举全部量化方法或验证目标硬件性能。

- **OCP MX v1.0 与 NVIDIA NVFP4**：正文区分 E2M1 标量码本、OCP 的 E8M0/32-value block scale、NVFP4 的 E4M3/16-value block 与 FP32 global scale，以及 TE 权重的 2D scaling。OCP 转换舍入/饱和约定、TensorRT Q/DQ scale API 和 Model Optimizer 支持矩阵各自绑定相应规范或访问日期，不把接口层次合并为同一 artifact 约定。
- **MR-GPTQ 与统一量化评估**：保留为格式专属旋转、误差补偿和 scale search 案例。正文限定模型、任务、RTN/GPTQ 变体、kernel 与硬件；不将“rotation/GPTQ 有效”外推为所有 FP4 recipe 的保证。
- **Four Over Six（4/6）**：纳入 §3.4，解释每个 1×16 NVFP4 block 分别映射到 4 与 6、反量化后按误差择优，并手算 `[10,20,30,40]` 的候选差异。将论文的 MSE 选择与 Transformer Engine 2.20.2 API 默认的 MAE、默认关闭状态和 256 E4M3 global-scale bound 分开；当前文档的 RL/PTQ 支持目标及带 RHT 的预训练未实现边界均按该版本表述。论文结果不扩展为目标 runtime 的部署保证。
- **OAS/MBS**：保留为 MXFP4/E8M0 scale 精度及 outlier 元数据的研究案例，解释 OAS 的归一化映射、MBS 的 1×128 macro block 与 1×16 局部 scale；作者质量和 6.2% prefill GEMM 开销绑定 direct-cast、指定模型/任务、CUTLASS 4.3.0、Blackwell SM100 custom kernel。
- **NVFP4 scale learning 与低精度训练 recipe**：保留作区分 PTQ calibration、训练时 scale learning、stochastic rounding、RHT 和选择性高精度层的支持背景；具体训练与 Model Optimizer 单模型结果不重写为 serving 通用结论。
- **跨章节重复控制**：不重复第 7 章通用 PTQ/质量框架，不在本章展开 FP8 KV 生命周期或 native INT4 packed artifact；4/6、OAS/MBS 和 MR-GPTQ 因直接改变本章格式的 scale/误差行为而留在专题分支。新来源事实和访问版本只写入 `research/weight-quantization-formats-and-ptq-2026.md`。

## 篇幅与教学例

按规范第 3.3 节计连贯中文解释性汉字，排除标题、表格、代码围栏、公式、行内代码、链接文本/URL、状态标记和空白；连续论证合并计数。核心下限 1,800 字，辅助下限 1,000 字。

| 教学单元 | 汉字数 | 类型 | 下限 | 组织理由 |
| --- | ---: | --- | ---: | --- |
| 开篇与 §§1—2 格式身份和 payload/scale 边界 | 2,402 | 核心连续链 | 1,800 | 从 INT4/FP4 的动机推进到格式与 recipe 的字段，建立后续 artifact 读法。 |
| §§3—8 scale 误差、量化对象、kernel/校准/评估与算例 | 3,826 | 核心连续链 | 1,800 | 沿共享 scale 的误差来源走到格式专属修正、数据对象风险、性能和可复算案例。 |
| §9 block 量化静态代码解释 | 1,066 | 辅助代码讲解 | 1,000 | 逐步解释 block 切片、`amax/6`、E2M1 最近码、MSE 和 metadata 账本；代码没有替代手算。 |
| §§10—11 artifact、回退与部署验证 | 1,906 | 核心连续链 | 1,800 | 从 manifest 字段和版本耦合推进到运行时 dispatch、回退代价、成对质量/性能验证及结论范围。 |
| §§12—14 误区索引、来源边界与小结 | 1,286 | 不单独计数 | 不适用 | 主要回指前文、定位问题和收束，不另承担独立教学；问题机制已在 §§1—11 完整讲解。 |

主要数值核对：

- **共享 scale 与 outlier**：E2M1 最大幅值 6，`s=20/6≈3.33`；`0.9/3.33≈0.27` 选到最近码 `0.5`，重构约 `1.67`、绝对误差约 `0.77`。若 `0.9` 单独成块，`s=0.9/6=0.15`，归一化为 6，可精确重构。
- **FP4 存储**：`N=4096²=16,777,216`；FP16 为 32 MiB，E2M1 payload 为 8 MiB。MXFP4 增加 `N/32=0.5 MiB` E8M0 scale，总计 8.5 MiB、理想压缩约 3.76×；NVFP4 一维增加 `N/16+4=1,048,580 bytes`，总计约 9 MiB、理想压缩约 3.56×。未计 layout、padding、转置、未量化层、KV 或 workspace。
- **Four Over Six**：论文算例中 map-to-6 的反量化向量 `[9.75,19.5,26,39]` 相对 `[10,20,30,40]` 的平方误差和为 17.3125，MSE 为 `17.3125/4=4.328125≈4.33`；map-to-4 的 scale=10，码值 `[1,2,3,4]` 可精确重构，MSE=0。该手算解释候选择优机制，不代表任务精度。
- **OAS/MBS**：归一化最大值 `3.3→6.6` 后映射码 `3→6`，示例相对误差 `0.3/3.3≈9.1%`；MBS 的存储估算为 `4+8/16+8/128=4.5625 bit/value`，未计 layout/padding。论文测试和 overhead 条件已在正文与研究记录限定。
- **静态 Python demo**：block size 8 的 scale 为 `20/6≈3.3333`，手算 MSE 约 `0.25998`、账本 7 bytes；block size 4 的 scales 为 `0.15` 与 `3.3333`，MSE `0.1077625`、四舍五入到五位为 `0.10776`、账本 10 bytes。逐项按码表核对，代码输出保持“手算预期，未执行”。

## 代码、公式、链接与静态门禁

- §9 示例在本轮正文中将 scale 改为 `amax/6`，并手算更新输出；标注“静态审阅，未运行（2026-11-02）”，输出标注为手算预期。用户明确要求不运行章节代码，因此没有执行示例、测试、模型或 benchmark。
- 静态 Markdown 门禁：24 个围栏 marker 配为 12 对，其中 8 个 math block；math 花括号净差为 0；围栏外未发现不成对美元分隔符。未命中 `\operatorname`、尖括号下标、`\(`/`\)`、`\[`/`\]`、正文维护路径、作者工作流状态或过时“实际运行输出”措辞。
- 34 个 Markdown 链接中 2 个本地目标均存在；32 个外链未逐条在线可达性检查。OCP、TE 2.20.2、Model Optimizer、MR-GPTQ、OAS/MBS、Four Over Six 等关键来源已定向核验；动态文档在研究记录中保留访问日期/版本。
- Mermaid 不适用：本章的格式账本、质量/性能决策和验证层次已用有读法说明的表格及连续文字表达，没有必要新增流程图。
- **配套同步**：`PAPERS.md` 增加 4/6 论文与当前 API 入口；`GLOSSARY_EN_ZH.md` 增加 Four Over Six；`EXERCISES.md` 将 4/6 纳入多路线机制与数值比较；`INTERVIEW_BANK.md` 增加误差度量、scale bound 和版本边界问题；`KNOWLEDGE_GRAPH.md` 补入 NVFP4 双候选 scale 子链。简介/目录摘要覆盖既有职责，未改。
- `git diff --check` 通过。没有构建或更新 PDF。

## 遗留风险与交接

- Transformer Engine、Model Optimizer、TensorRT-LLM 和 runtime 支持矩阵会变化；4/6 的默认值、适用对象和 artifact/kernel 路径须按目标软件 release 与 GPU 重新核对。论文与官方 API 的误差模式不同，本章未进行本地 runtime 复现。
- 32 个章节外链未逐条在线检查；没有目标硬件的 kernel、端到端 latency/throughput、显存峰值或模型质量数据。Demo 仅用于教学 scale 粒度机制，不能视为 OCP/NVFP4 兼容实现。
- 第 15 章完成本轮章级扫描、精修和静态门禁；第六册架构状态更新为第 1—15 章完成、第 16—18 章待处理。下一步按序进入第 16 章《FP8 KV Cache》，继续对照第 3、7 章边界；全册连续复读和最终门禁仍待处理，全套书修书 goal 保持 active。

## 整册连续复读补充：2026-11-02

第 15 章沿学习路径完成连续复读，并对照本记录复核 scale/outlier、存储账本、4/6、OAS/MBS 与代码示例的手算预期；未发现需要改正文或调整第 7、14、16、17 章边界的问题。代码仍未运行。第六册连续复读下一步进入第 16 章；全册最终门禁和全套书修书工作仍未完成。
