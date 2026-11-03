# 第六册第 5 章《推理引擎》章级审计

审计日期：2026-11-02。状态：完成本章定向扫描、正文精修、配套同步和静态门禁。保留 5.1—5.20 编号，不新增、拆分、合并或迁移章节。未运行章节代码、测试、模型、benchmark 或 PDF；外部链接未逐一在线复查。研究事实、来源阅读范围和证据边界写在 `research/inference-serving-metrics-and-disaggregation-2026.md`，本文件只记录书籍审阅决定、修改状态和交接。

## Scope 与章节职责

- 中心问题：给定模型 artifact、请求分布、服务协议、硬件、质量门槛、SLO 与维护责任，如何筛选推理 runtime，并用目标环境中的可复核证据决定是否上线。
- 本章负责 runtime 能力边界、公开执行路线、模型/协议适配、服务指标比较和迁移验收；连续批处理、公平性、抢占与 P/D 资源分离的完整调度设计交第 6 章；平台级路由、GPU 集群与观测治理交第 23 册；框架源码级执行路径交第 24 册。
- 简介与目录现有第 5 章摘要仍准确描述 workload、协议、硬件和运行要求下的 runtime 选型；本轮没有改变章节结构或其相邻职责，无需修改摘要。

## 章级扫描与候选去向

- **日期与范围**：2026-11-02。围绕 runtime 版本/维护状态、模型与协议兼容层、调度和 KV 能力、在线/离线负载、指标口径、压测方法、迁移与回滚开展定向检索。资料包括原始论文、固定版本官方文档和 release notes、当前官方动态文档、服务端 README 与实现说明。扫描范围是本章选型问题，不声称穷尽所有 runtime 或论文。
- **Transformers**：v5.22.1 官方指南把 `transformers serve` 定位为轻量本地/自托管入口，适用于评估、实验和中等负载，列出 continuous batching，并建议大规模生产评估 vLLM/SGLang；指南没有称其为 experimental。正文与资料索引固定版本，删除旧版“实验性”表述；同步修正术语表和面试题。
- **vLLM**：以 v0.32.0 文档固定 V1 API server、engine core、GPU worker、step scheduler、chunked prefill 和 benchmark CLI 的描述。token budget、KV block、协议正确性和目标设备容量分别解释，不把论文倍率写成当前服务承诺。
- **SGLang**：论文机制固定至 arXiv v2；session-aware Radix Cache 与 eviction 语义按 2026-11-02 查阅的官方动态文档描述。保留程序前端、分支/共享前缀和结构化生成路线；明确 session ID 只管理 KV 引用，不替应用保存历史、授权或硬性 pin cache。
- **TensorRT-LLM**：依据 2026-10-26 更新、commit `b96ba43` 的迁移页，改写旧 TensorRT engine build/conversion 默认流程；区分当前 PyTorch `PyExecutor`、scheduler、model engine、decoder 与目标 kernel 优化。专用 NVIDIA 路线作为配置受限的选项，不扩大成通用性能结论。
- **LMDeploy**：保留 TurboMind 与 PyTorch 两条 engine 路线，按 model/backend/device/dtype/量化坐标筛选；官方支持矩阵中的缺口只排除对应配置格，不以另一个 backend 的可用性推断兼容。
- **llama.cpp**：纳入当前 server README 的 parallel slots、continuous batching 与 LoRA 合批约束；按 GGUF、设备、上下文、KV、功耗和实际协议验收本地/边缘部署，不外推为数据中心吞吐优势。
- **TGI**：依据官方文档记录自 2025-12-11 起的 maintenance mode 与迁移提示。区分存量系统风险管理和新部署候选门槛，补入固定回放、无副作用影子流量、指标映射、灰度及回滚条件；维护标签不等同于所有存量服务立即失效。
- **方法谱系**：用 Orca、PagedAttention、Sarathi-Serve、DistServe、SGLang 和 vAttention 说明请求调度、KV 管理、prefill 分块、资源池组织、程序复用和虚拟内存各自针对的瓶颈。保留各论文的实验条件与不可比边界；更深的调度、P/D 路由和通信代价交第 6/8 章，没有新增章节。
- 相关来源及其版本、证据范围和未验证项已记录在上述 research 文件；本进度记录不重复保存事实资料。

## 篇幅、例子与代码状态

解释性汉字按正文连贯中文统计，排除标题、表格、围栏代码、公式、链接/引用清单和状态标记。按连续教学链合并必要的导入、比较、协议与选型小节；门槛为核心单元 1,800、辅助单元 1,000 汉字。

- 核心单元：§5.6 vLLM 1,922；§5.7 SGLang 1,873；§§5.12—5.14 “研究路线/能力矩阵→协议适配→指标比较”连续选型链 3,834；§§5.16—5.18 “筛选流程→常见误区→迁移案例”连续决策链 2,108。均达到核心门槛。
- 辅助单元：§§5.1—5.4 引擎与 workload 导入链 1,066；§5.5 Transformers 1,136；§5.8 TensorRT-LLM 1,080；§5.9 LMDeploy 1,143；§5.10 llama.cpp 1,117；§5.11 TGI 1,151；§5.15 选型审计 demo 解说 1,153。均达到辅助门槛。
- §5.19 是来源清单，§5.20 是收束，不作为独立知识讲解单元。计数工具为只读静态文本统计，质量判断仍依据人工检查，不以总篇幅替代解释完整度。
- 可复算例子覆盖：§5.4 混合请求率与输入/输出 token/s；§5.5 reference replay 95/100 与假设门槛；§5.6 token budget 分配及 TTFT/TPOT 门槛取舍；§5.7 三分支公共前缀 3,000 与 1,400 输入 token 的差异（不宣称 wall-clock 加速）；§5.8 8-GPU 池输出工作量换算，基线约 22.22、候选约 18.52、差约 3.70 GPU-hours/日；§5.9 backend/device/cache 组合格筛选；§5.10 64 KiB/token 的多 slot KV 预算；§5.11 994/1,000=99.4% 低于 99.5% 门槛；§5.12 队列轨迹中 D 等待 3 个额外 decode step；§5.13 102/105=97.1% 与 98%/取消全通过门槛；§5.14 QPS、输出 TPS、goodput、TPOT 和事件时间戳拆分；§5.15 硬门槛加权排序的手工计算；§5.18 单副本吞吐下的初始副本数估算。合成数字都标明是假设，不能作 runtime 实测。
- §5.15 唯一可独立运行的 Python demo 标为“静态审阅，未运行（2026-11-02）”。预期输出为纸面推演：`online_api` 的合格项为 vLLM/SGLang/LMDeploy，首位 vLLM；`agent_program` 首位 SGLang；`edge_local` 仅 llama.cpp；新服务中的 TGI 因 `new_service_status` 未过门槛。手算分数约为 vLLM 1.026、SGLang 0.967、LMDeploy 0.960。没有运行该代码。

## 静态门禁与配套同步

- 共 13 对 fenced block，其中 8 个 math fence；55 对行内美元公式。围栏配对完整，数学花括号/圆括号净差为 0；禁用的 `\operatorname`、尖括号下标和 `\text` 代码下划线扫描均为 0。代码没有执行。
- 52 个 Markdown 链接：51 个外链、1 个本地章节链接；本地目标存在，没有本地断链。外链没有逐一在线可达性复查；文中关键事实有对应来源入口，动态页面须按目标 release 再核验。
- `git diff --check` 通过。正文未残留作者指令、内部维护路径或会话进度；代码块的规定状态标记“静态审阅，未运行”紧邻代码块。
- 同步 `PAPERS.md` 的 runtime source entry，将旧 Transformers v5.6.0 与浮动 vLLM 文档链接改为 Transformers v5.22.1、vLLM v0.32.0 等带版本或访问日期的入口；同步 `GLOSSARY_EN_ZH.md` 与 `INTERVIEW_BANK.md` 中的旧“experimental”说法；在 `EXERCISES.md` 增加五层兼容矩阵和硬门槛/敏感性分析练习；在 `KNOWLEDGE_GRAPH.md` 增加 runtime capability matrix 与 model-protocol fit 节点。
- 没有修改简介/目录摘要，因为职责与结构未变化。没有构建或更新 PDF。

## 遗留边界与交接

- SGLang、TensorRT-LLM `latest`、LMDeploy stable、llama.cpp `master` 和 TGI 官方页面会变化；上线或升级前应锁定 package/release/commit、model revision、driver、设备和 workload，并复测兼容性与 SLO。
- 本章没有目标硬件的实际吞吐、质量或容量测量；§5.15 代码未运行，所有玩具数据不得表述为实测或排名。外链可达性尚未逐一验证。
- 第 5 章完成。下一步按书内顺序审阅第 6 章《批处理与调度》，重点覆盖 queue、continuous batching、token/KV budget、公平性、抢占、chunked prefill 和 P/D 解耦；与本章的 runtime 选型和第 24 册源码职责保持区分。
