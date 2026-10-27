# 全书精修进度 v2

本文件记录常态化全书精修的逐章覆盖、修订、验证和遗留项。新模型及新知识点更新另记在 `progress_v3.md`，事实证据放在 `research/`。

## 当前轮次：2026-10-07 启动

- `scope`：按书系顺序覆盖 24 册及 `book-llm-engineer` 补充篇；每册先审书系分工、定位、简介、目录、章节群结构和图表机会，必要时重排、增删、拆合或迁移，再逐篇研究、精修和同步。
- `basis`：当前版 `WRITING_SPEC.md`、`plan.md` 中常态化全书精修要求及仓库 `AGENTS.md`。
- `acceptance`：每册形成 `book_contract`、`chapter_map`、`architecture_decisions`、`visual_map` 和 `coverage_map`；书册入口与正文结构一致、各篇能连成该书类型所需的学习/检索/实践路径。逐章再审查问题、动机、连续机制解释、公式/状态、数值例子/代码/图表、深度研究、边界、失败模式、评测、证据、链接和配套同步；未完成项明确保留。
- `status`：进行中。第一册五份正文、第二册全部十份正文、第三册全部十一份正文、第四册第一至第十九份正文、第五册第一至第十八份正文、第六册第一至第十八章及第七册第一至第十五章已有本轮研究/精修记录。第五册第三至第七章的 toy Python 代码均只作静态审阅，没有运行，代码运行验收仍是开放项；第五册第八至第十八章和第六册第一至第十七章的标准库教学代码已按各自记录运行；第六册第十八章的价格估算示例已执行并与正文输出 `6.0` 一致。第七册第一至第十四章 demo 均未在本轮执行；第一至第四章依各自记录手算，第五章只核对关键路径，第六章按代码路径静态复读并手算预期摘要，第七章静态复核代码与预期输出并手算 ANLS 示例，第八章静态复核代码并手算预期输出，第九章静态复核代码并手算比例检验与区间，第十章静态复核代码并手算配对检验、精确 p 值及样本量；第十一章复核方法结构、静态审阅样例代码并手算关键输出，均未执行 Python；第十二章静态复核平台血缘和状态模型、手算 demo 输出，未执行 Python；第十三章补充 benchmark/方法对照、手算 M/G/1 队列例子并静态复读 demo，未执行 Python；第十四章补充代表性评测方法比较、检查 Claim Ledger demo 门禁并手算预期输出，未执行 Python；第十五章补充工作流 Agent 评测方法对照，静态复读 demo 并手算预期输出，未执行 Python。此前第四册第十份及之后的旧完成记录不作为本轮覆盖证据，第十至十九份已按当前轮次分别复核并重新登记。
- `goal_resume`：用户于 2026-10-27 明确要求从第八册及后续书册恢复精修。本轮覆盖第八册至第二十四册，并按书系原范围处理 `book-llm-engineer/` 补充篇；进度持续记入本文件。
- `architecture_status`：2026-10-27 新增书册架构门禁。既有逐章精修记录不自动代表完成整册架构审视；第一册架构待办仍保留。当前按用户最新范围开展第八册架构审视：已完成书册定位初稿、入口调整和外部扫描候选判断；第 2—16 章逐篇通读、章级扫描、篇幅和关系图仍待完成，不能标记整册架构通过。
- `next`：完成第八册第 2—16 章结构映射和章级扫描，确认书册骨架与内容去向；第一章已完成首轮精修，若整体结构审视引发调整再回看；随后依次处理第八册其余章节及第九至二十四册和补充篇。第一册架构审视作为单独待办保留，不覆盖用户指定的当前推进顺序。
- `chapter_refinement_next`：第八册第一章 `book-08-ai-safety-alignment/chapters/01-safety与alignment总览.md` 已完成首轮研究驱动精修：补读图说明、风险影响分类、运行时控制边界和安全评估 demo 手算，并于 2026-10-27 在 Python 3.12.9 执行 demo。正文、相邻书册导航、资料索引、题库、练习和术语表已同步；第八册整体架构仍待第 2—16 章审视，下一步继续结构扫描，之后沿书系顺序推进。
- `open_risks`：全书量大，按章节推进并逐批记录；未实际核验的代码、公式或配套同步不得标为完成。代码验证以可执行且输出能说明机制为准；优先使用 `.tmp/` 下被 Git 忽略的 CPU-only PyTorch 环境，不安装 CUDA 库，也不为性能或 100% 复现扩大依赖。第五册第三至第七章代码尚未运行，不能标记为运行验证完成；第五册第八至第十八章的 demo 运行不代表 CUDA/GPU、真实训练栈、目标模型或生产 checkpoint 验证。第五册第十章还未验证真实 benchmark 与污染审计；第十一章未运行真实 RL 训练、模型评估或外部公式渲染，官方厂商披露也不能替代算法细节。第五册第十三、十四章的公开论文结论与教学示例均不能外推成目标集群测量、真实成本或生产训练的证据；第十五章研究数字仅绑定公开论文所测模型、语料、指标和任务，未在目标领域/机构做适配训练或临床、金融工作流验收。第十六章未运行 RLVR 训练、独立 benchmark 或真实工具沙箱，论文指标仅绑定公开任务、模型与评测定义。第六册第五至第十章未在目标模型、GPU、集群或线上流量上验收；论文指标、vLLM/SGLang/DeepEP 当前文档和 TensorRT-LLM 文档分别绑定论文 workload、实现或版本，不能推断目标环境兼容、质量或性能。第六册第九章的 RAG/Agent demo 只验证 toy 文档、字符串和预设延迟，不代表目标语料召回、真实模型 grounding、AgentDojo/red-team 或生产工具安全验收；第十章 demo 只验证 token/上下文/KV/路由账本和输入合同，不连接真实视觉/语音/视频模型、processor、runtime 或目标流量；第十八章只核对 OpenAI 官方公开文档并执行 token 费用教学示例，未调用 Astra API、未测真实账号价格/速率/延迟，也未在目标 Agent executor、任务集或线上流量上验收；论文剪枝、语音延迟和 benchmark 数字均绑定作者配置。之后章节的代码/公式/配套同步也应如实记录。已有章节中的 CUDA、GPU 性能、多节点、真实模型/语料和生产 serving 边界仍按各章记录。PDF 构建 dry-run 因环境缺少 Pandoc 未能执行。
- `chapter6_open_risks`：第六册第一章只运行标准库 toy trace；第三章只验证理论 KV 账本；第四至第十章的 demo 只验证相应教学算法/选型逻辑，其中第九章不连接真实语料、embedding/ANN、模型、Agent runtime 或工具事务；第十章不连接真实 VLM/OCR/ASR/TTS、processor、视频解码器、GPU 或服务 runtime。Orca、FastServe、Sarathi-Serve、DistServe、PPD 与其它论文结果均不能跨模型、GPU、SLO、baseline 或外推到目标部署；Self-RAG、CRAG、GraphRAG、ARES、AgentDojo、retrieval-poisoning、EVS、VidCom2、Video-MME-v2 和 vLLM-Omni 结果均绑定其模型、任务、benchmark、威胁模型或硬件/负载条件；vLLM/SGLang/TensorRT-LLM/TGI/Riva 和厂商 API 等动态文档需按目标 release 重核。第十二章 demo 只验证合成价格、成功率、路由/cache 逻辑和线性延迟账本；不代表目标 API 账单、真实任务成功率、安全证据、GPU 利用率或线上 P95。全册目前没有目标模型、GPU、集群、线上流量或真实 serving benchmark 证据，不声称本地生产容量、成本、质量或安全结果。
- `chapter7_open_risks`：第七册第一至第五章 demo 均未在本轮执行；前四章按各自记录静态审阅和手算，第五章只核对关键路径。第二章的目标流量比例、层内得分、阈值和相似度输入均为教学假设；第三章的样本量、胜负平、偏差、置信区间和各研究结果均绑定其抽样单位、数据快照、任务与版本；第四章的 judge 结果、人工校准标签、反事实样本与所有论文指标均绑定作者的任务、模型、数据集和评测协议；第五章的 toy corpus、阈值、cutoff、canary 和扰动分数均为教学输入，论文证据绑定相应方法假设、模型、语料、benchmark 版本和实验配置。没有真实用户分布、授权训练语料、专家标签、目标模型 log-prob 或候选模型响应，不能声称目标 benchmark 有代表性、无污染或可靠区分模型，也不能推断目标 judge 已与人工对齐。LiveBench 项目主页、OpenAI Evals 和各厂商/框架文档会变，后续引用需按日期重查。论文数据与榜单扰动只绑定其题目、模型、harness 和版本。
- `chapter7_chapter6_open_risks`：第七册第六章的教学输入、准确率阈值和过程标签是合成例子；demo 仅静态复读并手算预期摘要，本轮没有执行 Python。其数学 parser 只覆盖简单数字/分数/百分比；隐藏测试门禁按示例中声明的行为切片差集演示，不是目标项目测试质量证明。GSM-Symbolic、Let's Verify Step by Step、ProcessBench、EvalPlus、LiveCodeBench、SWE-bench Verified 及其 2026 官方审计中的数字均绑定论文/发布方自己的题目、模型、日期、标注和 harness。没有目标模型响应、私有 holdout、仓库镜像、训练 cutoff/语料、专家标签或 target runner，不能推断目标系统数学/代码/过程能力、benchmark 污染或真实仓库修复成功率；OpenAI 关于 SWE-bench 的审计是发布方自评，不应写成独立共识。
- `chapter7_chapter7_open_risks`：第七册第七章教学数据、demo 门槛和例子全部是 toy；demo 只做静态复核，未执行 Python，也未连接真实 VLM/AudioLLM、processor、视频采样器、评测 harness 或生产流量。VQA v2、POPE、MMMU、DocVQA、ChartQA、AudioBench、Video-MME、MLVU、MMLongBench-Doc、Lost in the Middle、RULER、NoLiMa、LongBench v1/v2 与 LongBench Pro 数值和结论绑定各自论文数据、版本、模型、提示、预处理、解析器和任务；LongBench Pro 本轮只核对 2026-01 arXiv 作者版本。2026 DocInsights 的 ANLS 格式敏感性是单篇论文及特定 harness/模型/题目条件，不能推断所有 DocVQA 榜单均失效。没有目标图像/文档/图表/音视频/长文本任务集、用户分布、专家标注和私有 holdout，不能声称业务多模态质量、有效上下文长度、跨页证据率、音视频鲁棒性或用户风险已验收；没有外部公式渲染、PDF 构建或全章外链状态扫描。
- `chapter7_chapter8_open_risks`：第七册第八章的输入、门槛和代码输出为 toy，Python demo 未运行；没有目标模型、policy、真实用户分布、授权隐私样本、专家标签、工具状态环境或事故 trace，因此不代表目标产品安全性、稳健性或公平性。论文结果绑定其任务、模型、数据、评估器与攻击预算；SafeAudit 数字绑定其枚举器和环境，间接注入预印本是作者分析，厂商治理框架不能当作第三方认证。未做全章外链 HTTP 扫描、公式渲染或 PDF 验收。
- `chapter7_chapter9_open_risks`：第七册第九章 demo 使用 toy 分组和合成结果，未运行 Python，也未接入目标随机化平台、用户流量、业务日志或模型/API。现场研究的结果绑定各自企业、医疗网络、用户/任务、处理方式和观察窗口；QJE 客服研究主体是分阶段部署证据，不能称为完整生产随机 A/B。两比例 Wald 区间只是大样本教学近似；护栏使用点估计，不是非劣效检验；没有处理正式多重检验、聚类标准误、稀有安全事件或序贯停止。CUPED 方差收益仅适用于合适的实验前协变量，不能修复曝光或结果偏差。未做外链 HTTP 扫描、公式渲染或 PDF 验收。
- `chapter7_chapter10_open_risks`：第七册第十章 demo 和样本/评分均为 toy，未执行 Python、测试套件或真实 benchmark。配对差值正态样本量是规划近似；区间和检验仍依赖独立/聚类单位、目标抽样总体和适当校正。NLP 统计方法研究不自动代表当前产品流量；CUPED 方差效果绑定协变量与指标；没有目标模型响应、业务样本、人工 judge 校准、私有 holdout 或生产实验，不能据此声称目标系统提升。未做全章外链扫描、外部公式渲染或 PDF 验收。
- `chapter7_chapter11_open_risks`：第七册第十一章 toy 数据、根因干预示例和覆盖阈值均为教学输入；demo 未执行 Python，输出只按代码静态手算。50 条失败 trace 中 31 条在 oracle evidence 下修正不是线上 lift 或总体率；CheckList、ML Test Score、HELM、EvalPlus 和 Trace Grading 的方法收益分别绑定原论文任务或厂商产品实现。没有目标 RAG/Agent trace、专家标签、judge 校准、线上分布或独立 holdout，不能声称目标系统根因已确认、回归集覆盖充分或发布门禁有效；未做全章外链扫描、公式渲染或 PDF 验收。
- `chapter7_chapter12_open_risks`：第七册第十二章 demo 使用合成计划、分数、pairwise 计数和风险指标，未执行 Python；关键算术经手算。没有目标模型、真实数据、trace、调度器、judge gold、平台存储、用户流量或发布流程，不代表目标模型质量、评估器校准、运行可复现性、平台可靠性、成本或发布安全。HELM 的场景覆盖结果绑定论文方案；OpenAI、MLflow 与 OpenTelemetry 资料各自支持公开工作流、记录接口或字段规范，不构成目标系统验收。OpenAI Evals 的计划退役时间与 GenAI 语义约定均为动态信息，使用前应复核；本轮未做全章外链扫描、公式渲染或 PDF 构建。
- `chapter7_chapter13_open_risks`：第七册第十三章输入、长度桶、阈值和服务成本均为教学数据；demo 只按首末长度桶的差值作诊断，未执行 Python，不估计中间趋势、统计不确定性或目标系统表现。PI、YaRN、LongRoPE、RULER、NoLiMa、LongBench 系列和 MRCR 的结果绑定各自模型、样本、长度单位、提示、工具条件、评分和版本；LongBench Pro 为 2026-01 预印本，DeepMind MRCR v2 的 GitHub 文档滚动更新。M/G/1 队列算例只解释 Poisson 到达、单服务台、FCFS 下服务时间二阶矩对平均等待的影响，不模拟 GPU continuous batching。没有目标模型、tokenizer/runtime trace、真实长文语料、专家标注、GPU、集群或线上流量，不能声称目标系统有效使用 1M context、满足生产 SLO 或具备相应容量；未做外链 HTTP 全量扫描、公式渲染或 PDF 构建。
- `chapter7_chapter14_open_risks`：第七册第十四章的研究记录覆盖模型卡/数据说明、统一评测、行为测试、LLM judge 和评测报告方法；论文结论绑定各自语料与协议，Evaluation Cards 为 2026 年 arXiv 预印本。Grok 4.7、Claude Opus 5.5、Fable 5.1 案例分别来自厂商发布、第三方 provider/config 页面和特定 leaderboard/harness 快照；不能据此推出未披露的架构/训练配方、目标产品质量或生产安全。DeepSWE 的行缺失只表示 2026-09-22 页面快照无精确配置，不是零分。Claim Ledger demo 只作静态审阅和手算，未运行 Python；未做全章外链扫描、公式渲染或 PDF 构建，滚动页面须按引用日期重核。
- `chapter7_chapter15_open_risks`：第七册第十五章 demo 使用 toy 任务，未执行 Python，也未连接真实模型、Agent harness、仓库/网页/桌面/工具环境或目标业务。SWE-bench、SWE-Gym、Terminal-Bench、WebArena、OSWorld、τ-bench、ToolSandbox、AgentDojo、GAIA、ScienceAgentBench、PaperBench、METR、Gaia2 等结果绑定各自任务、版本、预算和 grader；BrowserGym 是实验框架，不是统一 benchmark。Executable-contract audit 是固定版本预印本方法，潜在可触达缺陷不等于真实 Agent 已触发或排名改变。未做全章外链 HTTP 扫描、公式渲染或 PDF 构建。
- `chapter11_open_risks`：第六册第十一章标准库 demo 使用 8 条合成记录，仅含 1 条攻击样本；零 ASR 的单侧 95% 上界为 0.95，输出正确保持安全证据门禁未通过。该界限假设独立同分布，不能外推到相关或自适应攻击。没有目标服务 trace、真实业务标签、judge 校准、目标 SLO、授权 red-team、GPU 或隐私流程演练；不声称生产质量、成本、延迟或安全率。OpenTelemetry GenAI 语义约定仍在演进，需按部署版本复核。
- `chapter13_open_risks`：第六册第十三章 demo 的服务时间、吞吐、GPU 预算和清单均为合成输入；`model_path_estimate_ms` 不含完整排队/检索/工具/网络尾延迟，代码将真实同步 P95、任务质量和安全证据门禁保持为未评估/缺失。没有目标模型、硬件、请求 trace 或生产发布证据，不能据此做部署 sizing 或宣称 SLO/质量/安全通过。OpenTelemetry GenAI metrics 约定处于 Development，动态官方文档需按实际版本复核。
- `chapter14_open_risks`：第六册第十四章容量参数、token 价格与事故时间线全部是合成/教学输入；8 个容量副本和 9 个含单故障余量副本不能外推到目标模型、GPU、trace 或集群。事故案例不是任何企业的真实记录，也不证明 prompt/cache 变化是生产事故根因；GPTQ/AWQ/SmoothQuant/KIVI、PagedAttention/vAttention 的研究结果分别受模型、任务、kernel、硬件、版本和测量边界约束。token-only 费用不含缓存/基础设施/重试等完整账目；demo 的安全字段只检查主题是否写入设计清单，没有验证权限或防护效果。未做真实压测、目标服务对照、外部公式渲染或 PDF 构建。
- `chapter15_open_risks`：第六册第十五章 demo 使用 E2M1 可表示值，但 scale 是未量化 Python 浮点数、storage bytes 中 metadata 是通用假设；不实现 E8M0/E4M3 scale rounding、NVFP4 FP32 global/二维 weight scaling、packing、Tensor Core 或 inference runtime。MXFP4/NVFP4 的 storage arithmetic 按给定一维 profile 推导，不能代替真实 artifact size 或 peak memory。MR-GPTQ、NVFP4 预训练和 Model Optimizer scale-learning 数字均绑定作者模型、任务、recipe、版本与 hardware；Model Optimizer support matrix 和 TensorRT/Transformer Engine 文档要随目标 release 复核。未做 GPU、真实 checkpoint、engine、量化质量/吞吐、外部公式渲染或 PDF 验收。
- `chapter16_open_risks`：第六册第十六章的容量和 scale 计算是教学推演；Python demo 的 8% metadata/对齐开销及任务分数为合成输入，不是运行时测量。未在目标模型、GPU、vLLM/TensorRT engine 或线上 trace 验证；vLLM/TensorRT/ROCm/PyTorch 等滚动文档应按目标 release 和硬件复核。vLLM 2026 报告、KIVI/KVQuant 数字绑定作者模型、任务、kernel 与配置，不能外推成目标 workload 结果。
- `chapter17_open_risks`：第六册第十七章的内存、成本、码值和教学矩阵均按正文假设推导；demo 的显存和任务分数是合成输入，不代表真实 checkpoint、量化工具、GPU 或 serving runtime。GPTQ/AWQ/QAT/MARLIN 指标绑定各自论文的模型、校准数据、任务、GPU、kernel 与 batch；vLLM latest、PyTorch 和其它滚动文档须按目标 release 复核。没有目标模型、GPU、实际权重布局、线上请求 trace 或生产 SLO 验收，不声称 INT4 在目标环境兼容、提升速度/容量或降低单位成功成本。
- `chapter18_open_risks`：第六册第十八章的费用表是 2026-10-27 官方公开价格快照，未绑定目标账户合同、区域附加费、工具费用或未来变价；Python 估算只覆盖 token 费用。Astra mode/effort、异步工具、steering、compaction 与误对齐监控按当前官方文档整理，未调用 API 验证真实事件时序、兼容矩阵、监控准确率或数据保留。job budget 的调用数、并发、deadline 和美元上限是教学示例，不是 OpenAI 限额或生产推荐。没有目标模型/工具、业务任务集、执行器 trace、真实请求费用或线上 SLO 验收，不作延迟、质量、成本或安全成效结论。
- `chapter3_open_risks`：第六册第三章代码只验证理论内存账本及输入合同，没有运行真实 attention、GPU allocator、CUDA kernel、PagedAttention/vAttention runtime、量化模型或目标 serving workload；论文性能和质量只绑定各自论文的模型、任务、硬件、kernel 和 cache budget。KIVI 2-bit 机制不能推断为 vLLM/TensorRT-LLM runtime 已支持；vLLM/TensorRT-LLM 动态文档需按实际 release 复核；cache salt 不替代鉴权、加密、日志策略或数据删除验收。未做外部数学渲染或 PDF 构建。

维护者要求本轮从头开始，不沿用此前半成品记录作为覆盖证据。旧的 `progress_v2.md` 记录已清空重建；`progress_v3.md` 保持原样。维护者进一步明确：代码用于展示和解释知识，只要代码经检查可执行且输出有助于说明问题即可；可使用 PyTorch CPU，不安装 CUDA 库，也不要求读者逐一进行完整复现。联网资料使用指定代理 `http://10.24.27.134:7890`。

## 本轮执行记录

### WRITING_SPEC 整体重构：建立端到端精修体系（2026-10-27）

- `scope`：整体重组 `WRITING_SPEC.md`，解决规范长期按新问题追加、相同要求分散在写作原则、研究流程、工作流和验收章节中的结构问题。
- `architecture`：按“宗旨与书册价值 → 逐册精修流程 → 问题诊断与处置 → 正文教学标准 → 资料研究与证据 → 图表/公式/代码 → 同步维护 → 完成门禁 → 附录”的主线重排。明确逐册审视顺序为定位、简介、目录、逐个 Markdown 文件的整体提纲、章节关系和内容去向、结构决策、必要研究、正文精修、整册复读。
- `changes`：合并重复的架构、深度、研究、视觉和验收要求；将内容类型结构与记录字段集中到附录；写明架构审视不预设全册重写，保留能兑现书册定位的内容，仅对有明确问题的结构作调整；增加小节拼凑、篇幅冗余、读者深浅不匹配、知识点迁移和图示缺失的处置规则。保留可直接引用高度契合的原始图片并在图旁注明出处的写作要求。
- `validation`：检查主标题层级、章节顺序、Markdown 围栏、文末空行和 `git diff --check`；未修改正文书稿、未运行代码或测试。
- `open_risks`：新规范已定义完整精修流程，但第一册正文的章节结构与依赖关系仍待按该流程落实；规范重排本身不代表任何书册已经通过架构门禁。

### 第八册架构审视与第一章首轮精修（2026-10-27）

- `scope`：用户于 2026-10-27 明确恢复从第八册及以后书册的精修。按当前 `WRITING_SPEC.md` 开始第八册架构审视与第一章首轮复读；本条记录架构审视的初步判断，不表示全册门禁通过。
- `book_contract`：目标读者是准备 alignment/safety 岗位、参与后训练/Agent/RAG/工具系统安全设计或治理评估的工程师。前置为 Python、深度学习、Transformer、预训练/SFT/偏好优化与工具调用基础；不要求已有 Safety 专业背景。读者应能区分目标与代理目标、分析风险和证据、设计行为评估与运行时控制。书册类型为问题递进的专题教材：第 1—12 章为主线，第 13—16 章为安全工程专题。本册负责行为风险、模型策略、评估、权限和治理，不替代第五册后训练、第七册评测方法、第九册数据治理、第十七册 Agent loop、第二十册 harness/runtime 和第二十二册工具协议细节。
- `intro_toc`：简介新增必要前置知识、书册类型/范围和边界，并链接相邻书册作为回跳入口；目标读者与能力承诺继续保留。目录将“Frontier Safety 专题”改为“安全工程专题”，避免把自动对抗评估、分类器、访问控制和安全降级都误归为前沿模型风险。
- `discovery_scan`：检索日期 2026-10-27，主要时间窗为 2025-10 至 2026-10；扫描 Agent 自主行动带来的身份/授权风险、轨迹级安全评估、发布证据链和组织治理。完整来源边界及候选去向见[第八册架构扫描研究记录](research/book-08-ai-safety-alignment-architecture-scan-2026-10-27.md)。读过 International AI Safety Report 2026、OWASP GenAI LLM Top 10 2026 与 Agent Control Standard、METR Feb–Mar 2026 Frontier Risk Report、AgentCanary 预印本、frontier safety case 研究及 NIST AI RMF / GenAI Profile。结论限于各来源自身报告、标准提案、模型/工具/窗口与公开信息范围。
- `chapter_map`（基于 16 篇标题结构和第一章全文复读的初稿，后 15 篇仍须逐篇核验）：

| 章 | 中心问题与本册作用 | 前后依赖/交接 | 主要视觉机会 |
| --- | --- | --- | --- |
| 1 | 安全与对齐如何从行为目标扩展到系统后果；全册地图 | 引出 2—11，向 12 综合；已补生命周期图 | 现有生命周期流程图与安全评估轨迹 |
| 2 | 真实目标、训练 proxy、模型学得目标和部署行为如何错位 | 承接 1，深入目标错配并交接 4 | 目标层次因果图、训练/测试分布对照 |
| 3 | 人类监督如何扩展到高复杂度任务 | 承接目标与标注限制，联系 4/6 | 方法谱系表、监督循环图 |
| 4 | 优化 proxy 如何被利用并偏离目标 | 承接 2/3，向 6 的评估与治理交接 | reward overoptimization 轨迹、偏差对照 |
| 5 | Jailbreak 和 prompt injection 如何跨越用户/外部内容边界 | 承接 1/2，风险评估交 6/13，权限边界交 15 | 信任边界和一次工具调用数据流 |
| 6 | 如何设计 red team、危险能力和发布评估 | 承接 5，评估证据进入 11/13 | 评估流程、能力 elicitation 与发布门禁表 |
| 7 | 如何从模型内部机制形成可检验解释 | 承接 2，交接 8/9 | 激活/patch/ablation 的因果轨迹 |
| 8 | 如何干预内部表示并评估收益与副作用 | 承接 7，交接 9 | intervention 强度—目标收益—副作用对照 |
| 9 | 如何编辑模型知识或降低特定记忆/能力 | 承接 7/8，和 10 区分删除与隐私风险 | edit、retain、locality 指标表 |
| 10 | 记忆、推断、差分隐私与水印各自能证明什么 | 交接第九册数据治理与第 11 章审计 | 隐私攻击与控制生命周期图 |
| 11 | 如何把规范、model/system card、证据和组织责任连到发布决定 | 汇合 6/10，概览 15/16，并承接事故响应 | claim—evidence—scope 论证图 |
| 12 | 面试中如何将本册概念迁移为系统判断 | 汇总 1—11；不是新概念的主讲位置 | 综合案例决策树与答案证据链 |
| 13 | 自动化对抗评估如何形成可复查证据 | 深化 5/6 的完整轨迹和发布闭环 | 攻击生成—执行—评分—修复闭环 |
| 14 | 多模态安全分类如何映射到策略与动作 | 与 5/6/15 相交，模型细节适时交第十五册 | 多模态输入—分数—策略—执行流程 |
| 15 | 每次 Agent 动作如何按主体、资源、风险和证据授权 | 承接 5/11/14，执行循环/协议交 17/20/22 | PDP/PEP 请求时序、状态/授权矩阵 |
| 16 | 风险或系统故障时如何限制服务并保持状态安全 | 承接 6/11/15，部署工程交第六/二十册 | fallback 状态机、重试/补偿状态表 |

- `chapter_scan`：第一章已完成全文复读和主题扫描；新增候选为影响分类与技术故障标签的双轴映射（采纳至 §6.12）以及行为规范与 runtime enforcement 的分工（采纳至 §9，交叉链接第十五章）。Agent 轨迹级评估并入第六/十三章候选，safety case 并入第十一章候选，ACS 并入第十五章候选；这些章节仍待各自定向检索与结构判断。
- `architecture_decisions`：目前保留 16 章顺序，不新开独立章节。理由是 2026 新材料中的 Agent 身份/权限、轨迹评估、safety case 和影响分类均能落在既有主讲章节，且与第十七、二十、二十二册有清楚交接。需要逐章核实的重复交界为：1/3 的 RLAIF 与 Constitutional AI，5/6/13 的攻击/红队/评估，11/15/16 的治理决定/运行时访问/fallback，9/10 的模型遗忘/隐私。未确认前不合并正文。
- `coverage_map`：原目录知识路径全部保留；新候选分别映射到第一章风险分类、第六/十三章轨迹评估、第十一章发布论证、第十五章运行时权限。现阶段没有删除或迁移重要旧内容；原有访谈综合章保留为第十二章收束，四个工程专题保留为可跳读路径。
- `chapter1_changes`：增加 Mermaid 生命周期图读法；在 §6.12 区分 misuse/malfunction/systemic risk 与 prompt injection、隐私、工具滥用等技术故障位置；在 §9 区分模型行为规范与 runtime authorization，并链接第十五章；来源表新增 International AI Safety Report 2026、OWASP 2026 LLM Top 10/ACS 和 METR Frontier Risk Report。没有改动 demo 可执行语句。
- `length_status`：第一章作为单篇核心讲解计；排除标题、表格、围栏内容、引用列表和代码状态后，脚本统计解释性行中约 9,028 个汉字，高于核心单元 1,800 字下限。此计数供门禁核对，仍以全文人工复读为准。
- `formula_status` / `example_status`：复核 HHH 多目标、漏拒/误拒、攻击成功率、工具越权率和严重度加权风险的分母语义；在 §10.5 将同一 10 条 toy case 的手算结果与 demo 对齐：unsafe compliance `3/5=0.60`、refusal accuracy `2/4=0.50`、over-refusal `1/5=0.20`、attack success `2/3≈0.667`、unauthorized tool `1/1=1.0`、safe completion quality `3.42/4=0.855`、severity-weighted risk `17/36≈0.472`。阈值均是教学输入，失败样本按 ID 去重；未外推为生产风险率。
- `code_status`：§10.5 原代码块未改动，可执行行为在 Python 3.12.9、标准库环境运行；用 `python3 -c` 从 Markdown 提取并执行该独立代码块。输入为固定 10 条 toy 样本，全部断言通过，输出与正文预期一致。覆盖了正常完整输入和主要决策路径；未测试空数据、真实模型、真实策略或工具服务。正文状态标注为“已验证”，对应本条记录。
- `visual_map`：第一章新增的从任务/模型能力到发布、运行时控制、事故复盘反馈的 Mermaid 图已在正文解释读法。第 2—16 章的现有图示和机会尚待逐篇检查；初步机会已列入 `chapter_map`。
- `evidence`：第一章新增事实仅用于风险分类、运行时标准定位和评估证据限制；论文/报告结论绑定来源版本及范围。安全 case 与 AgentCanary 暂列候选，未写成效果已证实的行业标准。
- `sync_status`：更新 `PAPERS.md` 的第八册推荐资料和本研究记录入口；扩展 `EXERCISES.md` 第 46 题的双轴分类要求；在 `INTERVIEW_BANK.md` 安全与治理部分新增第 27 题；`GLOSSARY_EN_ZH.md` 新增 Misuse Risk、Malfunction Risk、Systemic Risk。第八册简介和目录完成入口同步。
- `validation`：`git diff --check` 通过；核对简介、目录、第一章新增相对链接均存在；第一章 23 对代码/公式/图示围栏配对；计数 9,028 个解释性汉字；执行 §10.5 demo 并核对 assertions 和输出。未运行测试套件、PDF 构建或外部公式渲染。
- `open_risks`：第八册全书架构尚未通过；第 2—16 章逐篇扫描、篇幅、核心数值案例、代码标记和配套同步待办。OWASP ACS 是社区标准方案；METR 是有限窗口试点；AgentCanary 为预印本；safety case 置信度方法尚未成为通用认证。第一册架构任务仍独立开放。
- `status`：第八册定位和发现扫描已开始，第一章完成首轮结构化精修；当前 goal 继续，后续先完成第八册剩余架构审视，再逐章推进。

### 第七册第十五章：Specialized Frontier Eval Cluster 资料驱动精修验收（2026-10-27）

- `scope`：依据当前 WRITING_SPEC.md 复读 `book-07-evaluation-experiments/chapters/15-specialized-frontier-eval-cluster.md` §15.1—§15.27；承接任务契约、代码/终端、网页/桌面、事务性工具、科学/医学/安全、长周期、多模态、harness、轨迹指标、污染、统计、发布决策、复现包、demo 和练习。没有为扩充深度删除原有主线，补充内容分别落在对应知识点下。
- `changes`：对照 SWE-bench、SWE-Gym、Terminal-Bench 2.0；WebArena、OSWorld、BrowserGym、OSWorld 2.0；τ-bench、ToolSandbox、AgentDojo；GAIA、ScienceAgentBench、PaperBench；METR time horizon 与 Gaia2，解释各自解决的问题、机制、证据价值、限制和下一步补证。增加 executable-contract audit 预印本的证据边界，纠正 BrowserGym 误链；修正宏平均空切片处理和配对差值符号；补充 trace 脱敏、访问控制、保留期限与证据可追溯性。比较映射见[研究记录](research/specialized-frontier-eval-cluster-depth-review-2026-10-27.md)。
- `evidence`：核对 SWE-bench、SWE-Gym、Terminal-Bench 2.0、WebArena、OSWorld/OSWorld 2.0、BrowserGym、τ-bench、ToolSandbox、AgentDojo、GAIA、ScienceAgentBench、PaperBench、METR 官方 time horizon 页面、Gaia2 和 2026 executable-contract audit 预印本。论文数字只描述各作者的任务、模型、环境和协议；预印本 audit 的潜在可触达缺陷不能推成 Agent 已触发或公开排名失效。
- `formula_status`：切片宏平均改为只对 n_k > 0 的切片求均值，空集合报告 not_applicable；配对差值定义 A=新版、B=旧版，b=旧对新错、c=旧错新对，(c-b)/n 因而为新版减旧版。
- `code_status`：§15.23 demo 未运行。静态复读条件分支与汇总逻辑，并手算 hidden-test denominator=4、hidden pass=3、hidden pass rate=0.75、scope violations=1、dangerous failures=1、trace coverage=0.8、平均成本=2.54，决策为 hold_for_high_severity_review。
- `sync_status`：新增本章研究记录；PAPERS.md 加入 Frontier 工作流 Agent 评测方法与章节入口；EXERCISES.md 新增 6L 方法对照和合同审计设计题；INTERVIEW_BANK.md 新增 49H 代表方案比较题；GLOSSARY_EN_ZH.md 补 pass^k、BrowserGym、ToolSandbox milestone/minefield、METR time horizon 与 executable-contract audit。第七册目录已包含本章，无需修改导航。
- `validation`：复核 §15.1—§15.27 连续、章节围栏成对、公式定义、demo 关键输出手算、错误来源编号已移除、本地研究/目录入口存在；git diff --check 通过。未运行 Python、测试套件、公式渲染、全章外链 HTTP 检查或 PDF 构建。
- `open_risks`：见 chapter7_chapter15_open_risks。研究数字和工具合同结论均绑定公开任务/固定 revision；无目标模型、目标工作流、真实用户分布、环境或专家标签，不能声称业务 Agent 质量/安全通过。
- `status`：第七册第十五章完成资料驱动精修、静态复核和配套同步。依用户明确要求，本轮 goal 在本章后暂停；若恢复，章节精修下一篇为第八册第一章 `book-08-ai-safety-alignment/chapters/01-safety与alignment总览.md`。全局 next 仍保留第一册架构门禁。

### 第七册第十四章：Frontier Model 证据等级资料驱动精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 复核 `book-07-evaluation-experiments/chapters/14-frontier模型证据等级.md` §14.1—§14.23；保留并细化证据层级、最小 claim、来源类型、scope、benchmark 可比性、厂商自报、未确认名称、Claim Ledger、冲突处理、复现、未知项、demo 和发布审计案例。原有高频知识点继续承接，证据方法与案例映射见[研究记录](research/frontier-model-evidence-depth-review-2026-10-27.md)。
- `changes`：新增 §14.4.6 代表性研究对照，逐项说明 Model Cards、Datasheets、32K model card audit、HELM、CheckList、G-Eval、Evaluation Cards 各自解决的问题、设计机制、证据价值、限制及补证；把它们解释为互补环节而非单一证据排行榜。删除 §14.17 重复的未知名称模板，改为从产品名/API ID/artifact/config/技术报告逐级升级 claim。修复 demo 的 `unknowns` 列表结构和 architecture source 门禁，保证按每个架构 claim 检查直接来源。将案例与小结整理为连续结构，补 Grok 4.7、Opus 5.5、Fable 5.1 的官方/第三方/特定 harness 来源分账；明确 DataCurve 快照缺行不等于零分，并拆开 OSWorld 2.0 与 2.1 的相同 81.8% 分数。
- `evidence`：阅读 [Model Cards](https://arxiv.org/abs/1810.03993)、[Datasheets](https://arxiv.org/abs/1803.09010)、[32K Model Cards audit](https://arxiv.org/abs/2402.05160)、[HELM](https://arxiv.org/abs/2211.09110)、[CheckList](https://aclanthology.org/2020.acl-main.442/)、[G-Eval](https://aclanthology.org/2023.emnlp-main.153/)、[Evaluation Cards](https://arxiv.org/abs/2606.09809)；前沿模型事实沿用带来源、配置与边界记录的 [Grok 4.7](research/model-update-2026-09/grok-4.7-source-notes.md)、[Claude Opus 5.5](research/model-update-2026-09/claude-opus-5.5-source-notes.md)、[Claude Fable 5.1](research/model-update-2026-09/claude-fable-5.1-source-notes.md)。方法比较、数字和迁移限制见本章研究记录。
- `formula_status`：本章无新增待推导公式；检查方法论文的百分比、相关系数与模型案例分数均带原始 benchmark、数据快照、harness 或 trial 条件，不跨版本合并或跨方法排名。
- `code_status`：§14.19 Python demo 未运行。静态核对架构 claim 列表、逐条 direct-source 条件、未知项字段类型、信号与 action；按示例输入手算输出为 `all_scopes_recorded=False`、`task_claim_has_independent_evidence=True`、`architecture_claims_have_direct_sources=False`、`unknowns_field_recorded=True`，decision 为 `publish_scoped_task_claim_and_hold_architecture`。这只说明教学门禁逻辑，不是模型事实判定器。
- `sync_status`：新增本章研究记录；`PAPERS.md` 加第十四章入口及方法源对照；`EXERCISES.md` 新增 6K 代表性研究与发布 claim 审计题；`INTERVIEW_BANK.md` 新增 49G 方法解决缺口比较题；`GLOSSARY_EN_ZH.md` 新增 Claim Ledger 与 Evaluation Card。第七册目录已有该章；下一篇按用户要求移至第十五章。
- `validation`：检查 §14.1—§14.23 章节编号、示例代码围栏配对、预期输出与代码条件、正文和研究记录本地引用、配套索引落点；运行 `git diff --check`。未运行 Python、测试套件、外部公式渲染、全章 URL HTTP 扫描或 PDF 构建。
- `open_risks`：公开论文、厂商与第三方页面各有来源边界，无法替代目标模型、目标 harness、私有 holdout 或生产流量验收；benchmark 行缺失不能解读为失败或零分。未来页面随时间变化，引用时需重核；本轮没有独立复现厂商评测。
- `status`：第七册第十四章完成研究驱动精修、静态 demo 复核和配套同步；全书 goal 继续，章节精修下一篇为第七册第十五章 `15-specialized-frontier-eval-cluster.md`。`progress_v2.md` 独立全局 `next` 仍保留第一册架构门禁。

### 第七册第十三章：1M Context 有效能力资料驱动精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 复核 `book-07-evaluation-experiments/chapters/13-1m-context有效能力评测.md` §13.1—§13.21。保留原有五种“支持”口径、输入管线、位置机制、长上下文扩展/训练、KV 与资源、针测到多证据、条件化 `L_eff`、质量/证据/引用/拒答指标、分层实验、RAG、生产路由、失败诊断、证据等级、demo、版本比较、Qwen2.5-1M 阅读示例和练习；旧稿知识点承接与证据映射见[研究记录](research/long-context-effective-capability-depth-review-2026-10-27.md)。
- `changes`：在 §13.6 增 PI、YaRN、LongRoPE、Longformer/BigBird、Ring Attention、FlashAttention/PagedAttention 的分层比较，区分位置/训练、token 连接图、设备并行和执行/状态管理。§13.9 新增 benchmark 研究对照，说明 NIAH、Lost in the Middle、RULER、NoLiMa、LongBench v1/v2、LongBench Pro、MRCR v2 分别解决的问题、论文结果、长度单位及边界；VQA/DocVQA/ChartQA/AgentBench 保留为相邻模态/Agent 入口并明确非文本 1M 直接证据。修正 §13.14.1 的 `pρ` 启发式为带假设的 M/G/1 Pollaczek–Khinchine 公式和数值推导。修复 §13.17 demo 输入完整性失败无 action、`answer` 指标未参与诊断和决策的问题；把 quality、SLO、成本 action 分开，并优先修输入链路。
- `evidence`：阅读 [PI](https://arxiv.org/abs/2306.15595)、[YaRN](https://arxiv.org/abs/2309.00071)、[LongRoPE](https://arxiv.org/abs/2402.13753)、[Ring Attention](https://arxiv.org/abs/2310.01889)、[Longformer](https://arxiv.org/abs/2004.05150)、[BigBird](https://arxiv.org/abs/2007.14062)、[Lost in the Middle](https://arxiv.org/abs/2307.03172)、[LongBench v1](https://arxiv.org/abs/2308.14508)、[RULER](https://arxiv.org/abs/2404.06654)、[LongBench v2](https://arxiv.org/abs/2412.15204)、[NoLiMa](https://arxiv.org/abs/2502.05167)、[LongBench Pro](https://arxiv.org/abs/2601.02872)、Google DeepMind [MRCR v2 repo](https://github.com/google-deepmind/eval_hub/tree/master/eval_hub/mrcr_v2)、OpenAI [MRCR/Graphwalks 发布说明](https://openai.com/index/gpt-4-1/)及 Stanford [M/G/1 讲义](https://web.stanford.edu/class/ee384x/EE384X/handouts/H10.pdf)。方法、实验数字、证据范围和迁移限制见[研究记录](research/long-context-effective-capability-depth-review-2026-10-27.md)。LongBench Pro 只按 2026-01 作者预印本记录；MRCR v2 repo 是滚动页面。
- `formula_status`：手算 M/G/1 教学例：`λ=2/s`，服务时间以 98%×0.1s、2%×10s 混合，`E[S]=0.298s`、`u=0.596`、`E[S²]=2.0098s²`，平均排队时间约 `4.9748s`；全短请求对照约 `0.0125s`。复核原有 1M KV 示例 `131.072×10^9 bytes≈122.1 GiB` 及 8×80GiB、15% headroom 的 KV 预算 `54 GiB`。公式为单服务台理论，不外推成 GPU serving 延迟。
- `code_status`：§13.17 Python demo 未运行。静态复核所有质量/输入/运行/成本失败信号与 action 对应；按原有 toy 输入手算 `ingest_unstable=False`、evidence/answer/citation degradation 均为 true、SLO/cost miss 均为 true，首要决策 `route_long_tasks_to_async`，并保留质量调查动作。阈值只比较首末桶，未计算趋势、CI 或显著性。
- `link_status`：新增论文/官方来源和研究笔记的本地入口均可定位；第七册目录已包含本章，不需改导航。未做全章外链 HTTP 状态扫描；MRCR repo 和其它滚动页面需按 revision/日期复核。
- `sync_status`：`PAPERS.md` 补第十三章入口以及位置扩展、稀疏注意力/并行/缓存和评测 benchmark 分工；`EXERCISES.md` 新增 6J 长上下文评测证据链设计题；`INTERVIEW_BANK.md` 新增 49F 代表 benchmark 方案比较题；`GLOSSARY_EN_ZH.md` 补 NoLiMa、LongBench、LongBench Pro、MRCR v2 和有效上下文长度定义。
- `validation`：核对 §13.1—§13.21 连续、Math/Python/text fenced blocks 配对、新增公式命令、源文与 research note 链接、demo 条件路径及预期输出；`git diff --check` 通过。未运行 Python、测试套件、公式渲染、外链 HTTP 全扫或 PDF 构建。
- `open_risks`：教学数据、首末桶差阈值、服务预算和队列输入不代表目标系统；没有目标模型、tokenizer/runtime trace、真实长文语料、专家标注、GPU、集群或生产流量。benchmark 数字绑定各自任务、长度单位、模型、提示、工具、harness 和版本；LongBench Pro 为预印本，DeepMind MRCR v2 仓库滚动更新。M/G/1 模型不覆盖 GPU batching、抢占、chunked prefill、P/D 分离或多服务节点；未渲染公式、全量检查外链或构建 PDF。
- `status`：第七册第十三章完成研究驱动精修、来源对照、公式手算、demo 静态复核和索引同步；全书 goal 继续，章节精修下一篇为第七册第十四章 `14-frontier模型证据等级.md`。`progress_v2.md` 独立全局 `next` 仍保留第一册架构门禁。

### 第七册第十二章：评估体系综合设计与工程落地深度精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 复读第七册第十二章 §1—§16，保留并承接前文的评估目标、benchmark/golden/regression、human/pairwise、LLM judge、污染、Reasoning/Code/Math/Agent/RAG/多模态/长上下文/safety、在线实验、统计、错误分析、回归和平台设计；本章承担整合工作流，不替代第 1—11 章专题展开。章节文件、目录映射有效。
- `changes`：为 Eval Platform 增加端到端对象流图与 run/sample/attempt/score/artifact 血缘；扩展 manifest 至 data/prompt/index/evaluator/metric/runtime 版本和内容摘要，并区分可重评分与可重跑；将执行、评分和 maturity 拆成状态维度，解释远端已接受但客户端超时、重试与非幂等工具副作用；对照 HELM、OpenAI eval-driven 工作流、MLflow Tracking 和 OpenTelemetry GenAI 规范各自解决的问题、公开证据和边界。修正 §12 demo 发布门禁漏洞：所有失败 signal 现在生成动作并阻止进入发布评审，成功状态仍需责任人复核。原有知识点没有删除。
- `evidence`：[HELM 原论文](https://arxiv.org/abs/2211.09110)、OpenAI [Eval-Driven System Design](https://developers.openai.com/cookbook/examples/partners/eval_driven_system_design/receipt_inspection)、[Evaluation Best Practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)、[Working with evals](https://developers.openai.com/api/docs/guides/evals)、[Trace grading](https://developers.openai.com/api/docs/guides/trace-grading)、[MLflow Tracking](https://mlflow.org/docs/latest/tracking/)、[MLflow GenAI datasets](https://mlflow.org/docs/latest/genai/datasets) 与 [OpenTelemetry GenAI conventions](https://github.com/open-telemetry/semantic-conventions-genai)。各来源支持的判断和局限见[评估体系与平台深度研究记录](research/evaluation-platform-design-depth-review-2026-10-27.md)。OpenAI 页面截至 2026-10-27 提示托管 Evals 计划于 2026-10-31 转只读、2026-11-30 关闭；这是会变化的产品状态，非架构依赖。
- `formula_status`：无新增公式；手算 coverage `23/27=0.852`、weighted score `4.626/5.7=0.812` 和 pairwise win rate `(7+1)/12=0.667`，与示例输出一致。手算数据仅核对当前 toy 输入。
- `code_status`：静态复读 §12 demo 的信号构造、缺失项、失败动作和决策分支；给定输入 `failed_signals=['cost']`，缺失 judge calibration 和 multimodal/long-context 组件，最终应 hold。未运行 Python、demo、测试套件或真实模型调用。
- `link_status`：确认第七册目录中的本章路径、研究笔记链接及 PAPERS 入口存在；OpenAI、MLflow、OpenTelemetry 文档与 HELM 原论文页面已通过联网查阅。未做整章全外链 HTTP 扫描；动态产品日期和规范版本需再次采用前重核。
- `sync_status`：`PAPERS.md` 增加第十二章入口、代表方案和研究笔记；`EXERCISES.md` 的 35DB 平台审计补充超时/attempt 状态追问；`INTERVIEW_BANK.md` 的 40ZZZY 评估平台题增加独立状态、重试与副作用追问；术语表已有 Eval Platform Audit/Gate 相关词条，本轮未重复。第七册目录已包含本章，无需改动。
- `validation`：核对 §1—§16 主标题及 §11.1—§11.9 顺序；80 个 `~~~` 围栏行数为偶数；检查代码示例与手算预期输出、内部路径和新索引入口；`git diff --check` 通过。未执行 demo、自动测试、外部公式渲染、全章链接扫描或 PDF 构建。
- `open_risks`：demo 输入、阈值和输出均为教学数据；没有目标模型、评估集、trace、评估平台、权限、对象存储、线上实验或发布流程，不能证明目标评估器有效、平台可复现或发布门禁安全。未检查空样本/非法分数/并发调度等运行边界。HELM 数字绑定论文场景；厂商文档和规范说明各自公开工作流/字段，不等于目标系统实测。
- `status`：第七册第十二章完成研究驱动精修、demo 静态复核与配套同步；章节精修序列下一章为第七册第十三章 `13-1m-context有效能力评测.md`。`progress_v2.md` 的独立全局 `next` 继续保留第一册架构门禁。

### 第一册架构初审：明确主线定位与学习路径（2026-10-27）

- `scope`：审阅 `BOOK_SERIES.md` 中的书系角色、第一册 `简介.md` 与 `目录.md`，并对五个正文文件按讲次、部分路线、综合段和文件体量做结构普查；第二部分 Transformer 正文已在本轮前完整复读。当前记录是书册架构初审，不是五篇逐行精修完成声明。
- `book_contract`：第一册定位为大模型算法岗的共同主线入口，以文本语言模型为主对象，带读者从输入与表示走到 Transformer 预测、训练更新、后训练、生成和验证；建立跨专题共享的心智模型和阅读入口，不替代进阶研究、完整工程项目或各专题深挖。
- `chapter_map`：

  | 文件 | 当前讲次与角色 | 架构观察 |
  | --- | --- | --- |
  | `01-基础与语言模型.md` | 1–5：预测目标、损失、梯度和优化器 | 主线起点清楚；需要把每讲的新结果明确交给下一讲，并控制反复解释同一损失/训练过程。 |
  | `02-transformer核心.md` | 6–15：文本离散化、向量表示、Attention、Block、位置机制和 miniGPT | 顺序大体可形成从输入到输出的计算路径；单文件 4,769 行、十讲采用相似的展开模板，导致机制间关系容易被段落和重复解释淹没。应建立贯穿的 shape/数据流与小型输入例子，让每讲是同一模型计算过程的下一步。 |
  | `03-训练对齐推理.md` | 16–30：预训练/规模/分布式、指令微调与偏好优化、生成/部署/评估、安全、研究和综合实验 | 单文件 6,040 行，承担多个不同阶段；16–18、19–23、24–28、29–30 各自形成子链，但文件内层级未能充分表现边界和交接。需将“基础闭环必须掌握”与“专题扩展入口”区分，并评估按生命周期拆分导航。 |
  | `04-多模态基础.md` | 31–36：图文、视觉语言、扩散、语音和视频 | 角色应是主线的模态扩展地图，不应让六个不同技术栈看起来像在第一册被等深掌握；深挖入口指向第十五册及相关专题。 |
  | `05-综合复习与技术表达.md` | 37–38：开放问题、迁移和技术表达 | 具备整册收束的潜力；需要明确回收前四部分的同一条主线，并让读者产出一份可复核的综合判断。 |

- `architecture_decisions`：当前五部分“基础 → Transformer → 训练/对齐/推理 → 多模态扩展 → 综合迁移”的顶层顺序有可保留的因果骨架，本轮没有仅为调整而重排讲次。重新定义“共同主线入口”的书册价值，明确多模态是扩展地图；将第 30 讲定位为文本模型阶段综合实验，第 37–38 讲定位为整册收束。第二、三部分文件内的论述主线、冗余边界和必要拆分仍需落实到正文。
- `changes`：重写第一册 `简介.md`，说明本册独有价值、能力目标、与专题书的分工和五部分交接；重写 `目录.md`，新增从训练目标到模型回答和验证的 Mermaid 主线图，以及逐部分的核心问题、交接产物和后续入口；同步修订 `BOOK_SERIES.md` 第一册定位。在五个正文文件的开头加入对应部分的流程图，分别显示 loss/参数更新、Transformer 前向与训练/生成分支、预训练—后训练—服务—评估链路、多模态表示路径和开放问题分析闭环。
- `visual_map`：整册目录图展示训练分支（目标 token → loss → 梯度更新）、生成分支（概率 → 解码 → token 回接）及多模态输入抽象；各部分图按自身目标补充。后续还需检查章节内部是否需要更具体的 Attention 权重/张量变化图、训练账本、推理状态图，以及是否有高度契合的来源图片可直接复用并标注原图和链接。
- `validation`：核对五份正文的讲次边界与 `目录.md` 文件映射；第一至第三部分分别覆盖 1–5、6–15、16–30，第四和第五部分覆盖 31–36、37–38；确认新目录保留 38 讲全部主题并补充跨册入口。核对新增图示的主线与对应章节文本口径；尚未完成正文间依赖的逐条核验、章节重组和全册连续复读；本次 `git diff --check` 通过。
- `open_risks`：本次完成的是书册定位、导航和各部分入口图示，正文目前仍有 15,032 行；第二、三部分的局部推理链、重复讲解和每讲深度分配尚未按新定位重构。目录和入口图表达的是目标教学主线，需要在正文中真正落实各章节的承接，不能把入口改写和部分图示当作全册教学重建已完成。
- `status`：第一册定位、简介、目录和部分级流程图已按新契约调整；书册架构门禁仍进行中。下一步逐篇核对依赖关系、章节过渡与内容边界，定出正文主线重构方案，再实施相应结构调整。

### 第一册第二部分骨架复审：Transformer 主线与内容边界（2026-10-27）

- `scope`：依据重组后的 `WRITING_SPEC.md`，将第一册定位、简介、目录、五部分路线与 `02-transformer核心.md` 的整体提纲交叉审视；以本轮前对该正文的全文复读为基础，补做章节边界、重复结构和视觉机会盘点。本条是结构诊断与方案，不代表正文已完成重构。
- `book_contract_check`：`BOOK_SERIES.md`、简介和目录对第一册的承诺一致：它是算法岗共同主线入口，解释文本模型从输入、表示、预测到训练/生成/验证的核心路径；第二册和专题书负责研究深挖，第三册负责代码项目。五部分顺序“基础 → Transformer → 训练/对齐/推理 → 多模态扩展 → 综合迁移”可以保留，不需要为显得有改动而重排整册。
- `chapter_map`：`02-transformer核心.md` 当前 4,796 行，覆盖第 6–15 讲：6 Tokenization；7 Embedding 与位置编码；8 Attention 直觉；9 Attention 公式及多种变体；10 Multi-Head 与 KV 共享；11 Causal Mask 与生成；12 Transformer Block；13 LayerNorm/RMSNorm 与残差；14 RoPE 与长上下文；15 miniGPT。结构上有四组边界交叠：第 7/14 讲都讲位置与 RoPE；第 9/10 讲都展开 MQA/GQA 等变体；第 12/13 讲都解释 norm、residual 和 Pre-LN；第 8/9/11 讲分散了从直觉、公式到因果约束的同一注意力计算链。第 9 讲还把 sparse/linear attention、FlashAttention、MQA/GQA 等扩展方案放入共同主线。
- `findings`：十讲中均重复出现失败模式、易混淆项、练习和回顾等固定尾部模块，八讲另设适用边界；这种模板让单讲看似完整，却反复重启解释，削弱跨讲累积感。当前文件约 4,796 行，只有一张 Mermaid 总流程图、约 32 行表格内容；有总图和大量局部信息，但缺少贯穿各讲的同一个小输入/shape 账本。生成循环与 miniGPT 训练又会和第一部分的预测/更新以及第三部分的解码/Serving 形成边界重叠，需要在书内明确各自只讲什么。
- `architecture_proposal`：按一条贯穿路径组织内容：文本与 tokenizer 输出 → embedding/位置形成初始表示 → Q/K/V、打分、缩放、softmax、Value 聚合与因果 mask → 多头与 KV 共享 → MLP、norm、residual 组成完整 block → logits → miniGPT 端到端实现。把第 7/14 讲的 RoPE 内容合并到已经定义 Q/K 的位置；把第 8/9/11 讲改成连续的“直觉—计算—因果约束”解释链；将 MQA/GQA 收敛到一个注意力头/KV 布局对照；合并第 12/13 讲为完整 block 的结构与稳定性解释。第 15 讲保留为综合实现，但复用同一组输入和 shape，不重新从头引入一套平行概念。
- `scope_decisions`：第一册正文保留标准 decoder-only Transformer 的必要主线。FlashAttention、稀疏/线性 attention、长上下文扩展/评估、深层 KV Cache 与 Serving 细节只保留定位、关键边界和后续入口，分别映射到第二、五、六、七、二十一、二十四册；不能因迁移而删除读者入口，也不能在第一册重复讲成专题。最终章节数和文件拆分方式待完整内容去向映射后确定，不把现有 38 讲或文件数当成必须保留的约束。
- `visual_map`：以一组固定 token 和 shape 贯穿全章，补一张从 token IDs 到 logits 的结构/数据流图；用一个小的 attention score 矩阵同时显示因果 mask 与 softmax 后的权重；用结构图比较 MHA/MQA/GQA 的 Q/K/V 头布局和 KV Cache 影响；用一张完整 Pre-LN block 图连接 attention、MLP、norm、residual；miniGPT 部分展示训练分支与增量生成分支。相关外部论文或官方图像可直接引用并在图旁标注来源。
- `open_risks`：实施重排前仍需逐节映射现有例子、公式、代码、练习、图表和引用，区分重复解释与必须保留的不同问题；若移出长上下文或 attention 变体的事实性内容，需检查目标专题章节是否已有可链接落点。当前没有改动 `02-transformer核心.md` 正文。
- `status`：书册契约、简介、目录与第二部分的主问题已对齐；第二部分确认存在可以通过合并、收束和重排解决的结构问题。后续应先完成逐节 coverage_map 和专题映射，再按门禁实施章节拆合与正文收束。

### 第七册第十一章：Error Analysis 与回归测试深度精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 复读 `book-07-evaluation-experiments/chapters/11-error-analysis与回归测试.md` §1—§17；保留总体分数、误差向量、证据记录、taxonomy、标注校准、聚类/主动挖掘、根因、regression suite、样本级版本 diff、修复/自动化、demo、边界、benchmark 扩展和练习。
- `changes`：增加错误分析闭环图；在 §6.7 用 RAG 年份约束错答展开竞争根因假设、原始 trace 复现、oracle evidence 干预、中间阶段检查和未见变体验证；在 §7.9 对照 CheckList、ML Test Score、HELM、EvalPlus 和 OpenAI Trace Grading 各自解决的问题、机制与适用限制。强化 regression coverage 只是标签覆盖的边界。toy demo 将 `root_counts` 改称 `root_hypothesis_counts`，明确静态映射不是已验证根因；添加关键输出的手算核对，并说明覆盖阈值 `0.80` 仅是教学策略。
- `evidence`：核对 ACL [CheckList](https://aclanthology.org/2020.acl-main.442/)、Google Research [The ML Test Score](https://research.google.com/pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf)、[HELM](https://arxiv.org/abs/2211.09110)、NeurIPS [EvalPlus/HumanEval+](https://papers.neurips.cc/paper_files/paper/2023/hash/43e9d647ccd3e4b7b5baab53f0368686-Abstract-Conference.html) 和 OpenAI [Trace Grading](https://developers.openai.com/api/docs/guides/trace-grading)。来源各自方法、结果、局限和正文映射见[研究记录](research/error-analysis-regression-testing-depth-review-2026-10-27.md)。
- `formula_status`：手算 toy 示例四象限为 stable correct 1、fixed 2、regressed 4、stable wrong 3；回归集标签覆盖 `4/6≈0.667`，缺 `reasoning` 与 `tool`；候选假设映射计数、priority 排序和 hold 决策与示例字段一致。没有新增统计推断公式。
- `code_status`：第 8.5 节的通用对齐函数和第 11 节 toy audit 均未运行 Python；静态复读关键循环、分类、切片覆盖、动作门禁和 `root_hypothesis_counts` 映射。只手算输出，未将其写成执行结果。
- `sync_status`：`PAPERS.md` 增第十一章入口和代表性方法分工；`EXERCISES.md` 增 87F—87G；`INTERVIEW_BANK.md` 增 5J—5K；`GLOSSARY_EN_ZH.md` 增 Error Analysis、Root Cause Hypothesis、MFT、INV、DIR。第七册目录已有本章入口，无需改目录。
- `validation`：核对 §1—§17 顺序、研究链接落点、示例手算输出和配套索引；章节 Markdown 围栏成对，`git diff --check` 通过。未执行 Python、测试套件、全章外链 HTTP 扫描、外部公式渲染或 PDF 构建。
- `open_risks`：见 `chapter7_chapter11_open_risks`。目标 RAG/Agent trace、专家标签、人工 judge 校准、线上分布和独立 holdout 均未提供；来源结果绑定各自论文/产品配置。
- `status`：第七册第十一章完成本轮研究驱动精修、静态复核与配套同步；章节顺序精修下一章为第七册第十二章 `12-评估体系综合设计与工程落地.md`。`progress_v2.md` 的全局 `next` 仍保留第一册架构门禁。

### 第七册第十章：统计显著性与实验设计资料驱动精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 复读 `book-07-evaluation-experiments/chapters/10-统计显著性与实验设计.md` §1—§17；原有 estimand、分析单位、配对比较、McNemar、描述统计、显著性、置信区间、bootstrap、功效/MDE、多重比较、CUPED、消融、复现、demo、证据链和练习均保留并深化。
- `changes`：新增统计检验选择表，比较 paired t、McNemar、paired bootstrap、近似随机化检验和 cluster bootstrap；给出小 discordant count 的精确 McNemar 手算；加入配对差值的样本量公式；补充 BH 依赖条件、BY 和 Holm；加入 NLP 测试域/目标域有效性证据。修正 CUPED 和序贯推断错链、arXiv 论文标题误配；将 demo 独立两组样本数与配对样本计数不匹配的问题改成配对样本量函数。
- `evidence`：ACL [检验选择指南](https://aclanthology.org/P18-1128/) 和[显著性跨域研究](https://aclanthology.org/D12-1091/)、ExP [CUPED 原文页](https://exp-platform.com/cuped/)、[Always Valid Inference](https://arxiv.org/abs/1512.04922)、eBay [SRM/randomization validation](https://arxiv.org/abs/2208.07766)、Holm 及 Benjamini–Yekutieli 原论文入口。逐项确认 `1512.04922` 是序贯推断、`2102.12352` 是凸分析论文，不是 CUPED 或序贯检验；对照表和研究适用边界见[统计显著性研究记录](research/statistical-significance-experimental-design-depth-review-2026-10-27.md)。
- `formula_status`：手算 `b=1,c=8` 的连续性校正 McNemar `χ²=4.0` 及双侧精确二项 `p≈0.0391`；手算 `sigma_d=0.5`、MDE `0.03`、双侧 alpha `0.05`、power `0.80` 需要约 2,178 个独立配对。复核 toy accuracy 为旧 `0.45`、新 `0.80`、差 `0.35`，多重比较的 Bonferroni/BH 发现数与假设一致。
- `code_status`：§12 demo 未运行。静态复核配对 bootstrap 的索引保留、样本量函数的输入输出一致性、Bonferroni/BH 和 CUPED 分支；无 Python 执行结果，未声称其随机 bootstrap 区间经运行验证。
- `sync_status`：`PAPERS.md` 增加统计检验来源与研究记录入口；`EXERCISES.md` 增加 87D—87E；`INTERVIEW_BANK.md` 增加 5H—5I；`GLOSSARY_EN_ZH.md` 增加 estimand、ATE、配对/聚类重采样、随机化检验、Holm 与 BY。第七册目录已收录本章，无需改目录。
- `validation`：检查 §1—§17 顺序、方法表、配对公式和 demo 变量命名、来源链接目标与配套索引；Markdown 围栏配对；`git diff --check` 通过。未运行 Python、测试套件、外部公式渲染、全章外链 HTTP 扫描或 PDF。
- `open_risks`：见 `chapter7_chapter10_open_risks`。toy 数据和检验没有目标总体、真实模型、人类评审或线上随机化证据；每种方法仍须按目标抽样单位、依赖结构和决策损失重选。
- `status`：第七册第十章完成研究驱动精修、静态复核和配套同步；章节精修序列下一章为第七册第十一章 `11-error-analysis与回归测试.md`。`progress_v2.md` 的独立 `architecture_status/next` 仍记录第一册架构门禁。

### 第七册第九章：在线实验与 A/B Test 资料驱动精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 复读 `book-07-evaluation-experiments/chapters/09-在线实验与ab-test.md` §1—§16；保留并深化端到端在线价值、随机单位、分桶、曝光、SRM、主指标、护栏、用户反馈、灰度、shadow、回滚、离线/在线差异、统计工程陷阱、demo、项目闭环和练习。
- `changes`：新增客服、代码助手、临床决策支持三类现场研究比较，明确各自设计、结局和外推边界；扩展 assignment/delivery/adoption schema，定义 ITT 并解释用户采纳的 post-treatment selection；深化 SRM 分段根因诊断与网络干扰设计；补充比例指标分母、结果成熟窗口、重复观测/聚类、连续监测和实验前协变量。详细的旧知识承接与方法比较见[在线实验研究记录](research/online-experimentation-llm-depth-review-2026-10-27.md)。
- `evidence`：核对 [CUPED](https://exp-platform.com/cuped/)、[Always Valid Inference](https://arxiv.org/abs/1512.04922)、Microsoft [SRM taxonomy](https://www.microsoft.com/en-us/research/publication/diagnosing-sample-ratio-mismatch-in-online-controlled-experiments-a-taxonomy-and-rules-of-thumb-for-practitioners/) 与 [A Dirty Dozen](https://www.microsoft.com/en-us/research/publication/a-dirty-dozen-twelve-common-metric-interpretation-pitfalls-in-online-controlled-experiments/)、[Detecting Network Effects](https://www.kdd.org/kdd2017/papers/view/detecting-network-effects-randomizing-over-randomized-experiments)、[Generative AI at Work](https://doi.org/10.1093/qje/qjae044)、三家企业的代码助手随机试验（[Management Science](https://doi.org/10.1287/mnsc.2025.00535)）及[肯尼亚基层医疗集群随机试验](https://doi.org/10.1038/s41591-026-04503-6)。数值和结论均限于各自研究设计与任务，未合并为通用 LLM uplift。
- `formula_status`：将两比例零假设检验的 pooled SE 与 Wald CI 的 unpooled SE 分开；手算 toy 例中差值 `0.05`、检验 SE `0.0223`、`z≈2.245`、区间约 `[0.006, 0.094]`。说明大样本近似不适用于所有稀有事件、小样本和强聚类情形。
- `code_status`：§11 Python demo 未运行。静态复核分母为空返回 `None`、检验与区间 SE 命名和预期输出、缺失证据 action；§11.1 明确护栏点估计不构成非劣效结论，主指标门禁未做多重检验校正。
- `sync_status`：`PAPERS.md` 新增来源与现场研究索引和 research note 入口；`EXERCISES.md` 新增 87A—87C；`INTERVIEW_BANK.md` 新增 5E—5G；`GLOSSARY_EN_ZH.md` 增补 ITT、uptake、SRM、CUPED 和 interference。第七册目录已有第九章，无需修改。
- `validation`：复核 §1—§16 标题顺序、章节文件与本地研究/索引链接、Markdown 围栏配对、公式和 toy 输出手算；`git diff --check` 通过。未运行 demo、测试套件、公式渲染、外链 HTTP 扫描或 PDF。
- `open_risks`：见 `chapter7_chapter9_open_risks`；无目标系统实验数据或外部实测，所有研究结果和 toy 数值均限于明确来源/示例条件。
- `status`：第七册第九章完成研究驱动精修、静态复核和配套同步；全书 goal 继续，下一章为第七册第十章 `10-统计显著性与实验设计.md`。

### 第七册第八章：Safety 与 Robustness Eval 资料驱动精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 复读 `book-07-evaluation-experiments/chapters/08-safety与robustness-eval.md` §1—§21；保留并深化 policy、漏拒/误拒、harmful output、jailbreak、prompt injection、privacy、fairness、dangerous capability、robustness、red teaming、统计、judge、线上闭环、诊断 demo、项目流程和练习。
- `changes`：解释 0/n 与 0/0 的证据区别；比较 HarmBench、StrongREJECT、JailbreakBench；加入 InjecAgent、AgentDojo、SafeAudit 对照和预印本观点边界；区分训练数据抽取、成员/属性推断及应用权限泄露；深化 BBQ/CBBQ、危险能力的人类 uplift、CheckList 行为测试、judge 校准、风险报告和严重度加权指标的限制。修正 demo 的空分母、safe completion 和 action consistency 口径；新增零事件置信上界的手算示例。
- `evidence`：系统阅读与来源边界、旧知识承接、方案比较和复核结果见 [Safety 与 Robustness Eval 研究记录](research/safety-robustness-evaluation-depth-review-2026-10-27.md)。论文数字均绑定原始模型、数据、评估器、威胁模型和测试预算；NIST/OWASP 为风险流程入口，厂商框架按发布方治理材料处理。
- `formula_status`：手工核对零事件单侧上界 `1-α^(1/n)`，在 `α=0.05` 时 `n=10/100/300` 分别约为 `25.9%/2.95%/0.99%`；手工核对 StrongREJECT 分数构成、MIA advantage 的分母含义和 demo 指标。零事件边界依赖独立同分布尝试，不能直接用于相关/自适应攻击；未使用外部公式渲染器。
- `code_status`：§16 Python demo 未运行。静态复核空分母返回 `None`、空均值处理、缺证据触发补采 action、safe completion 排除泄露和越权工具动作、扰动一致性按正文 action 定义；手工核对预期指标和 action 列表。全部输入、阈值和输出均为教学 toy 数据。
- `link_status`：第七册目录已包含本章；正文、研究记录和 `PAPERS.md` 的本地入口可定位。未扫描外链 HTTP 状态，也未构建 PDF。
- `sync_status`：新增本章研究记录；`PAPERS.md` 补充代表研究、官方框架和预印本入口；`EXERCISES.md` 增加 benchmark、Agent 状态、隐私分类、零事件边界和行为测试练习；`INTERVIEW_BANK.md` 增加方法比较、指标边界、uplift、统计和 judge 设计问题；`GLOSSARY_EN_ZH.md` 补充缺失评估术语并修订严重度加权风险边界。
- `validation`：复核一级章节 §1—§21 连续（另含 §16.1 说明段）；Markdown 围栏 60 个、配对；研究记录和配套入口可定位；`git diff --check` 通过。未运行 Python、测试套件、外部 benchmark、公式渲染或 PDF。
- `open_risks`：教学数据、门槛和判分器不代表目标产品。没有目标模型响应、实际 policy、用户流量、授权隐私样本、专家标签、工具状态环境或生产事故 trace；研究结果限于来源配置，SafeAudit 数字限于其枚举器/环境，间接注入预印本为作者分析，厂商治理框架不是第三方认证。未做全章外链可用性与 PDF 验收，见 `chapter7_chapter8_open_risks`。
- `status`：第七册第八章完成资料驱动精修、静态复核和配套同步；全书精修继续，下一章为第七册第九章 `09-在线实验与ab-test.md`。

### 第七册第七章：多模态与长上下文评估资料驱动精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 复核 `book-07-evaluation-experiments/chapters/07-多模态与长上下文评估.md` §1—§20；原有 VQA、OCR/字段、图表、视频、语音、grounding、NIAH/长文/长代码、多模态长任务、judge、错误树、demo、项目闭环、误区与练习均保留。知识点承接和源文献对照见[研究记录](research/multimodal-long-context-eval-depth-review-2026-10-27.md)。
- `changes`：补 DocVQA ANLS 的编辑距离、阈值 `0.5`、多参考答案聚合和 `Paris`/`Answer: Paris` 手算；引用 DocVQA 2020 单页 QA 与文档集合检索 MAP 的任务区别，并加入 2026 DocInsights 对 ANLS prompt suffix / 输出格式敏感性的限域审计。新增 §15 benchmark 对照：VQA v2 互补图像对、POPE 三类负例的对象幻觉 probe、MMMU 专业多模态题、DocVQA、ChartQA 人写/生成 QA 与 5% 容差、AudioBench 三类音频任务、Video-MME/MLVU 时长和任务覆盖、MMLongBench-Doc 跨页/不可回答设计；长上下文路线比较 NIAH、Lost in the Middle、RULER、NoLiMa、LongBench v1/v2 与 LongBench Pro。加一段受控 64K 例子和目标任务评估组合表；§16—§20 保持编号和内容。修复 demo 缺少 latency 失败动作，输出标记为静态核算，不是运行结果。
- `evidence`：阅读 [VQA v2](https://arxiv.org/abs/1612.00837)、[POPE](https://arxiv.org/abs/2305.10355)、[MMMU](https://arxiv.org/abs/2311.16502)、[DocVQA](https://arxiv.org/abs/2007.00398) 与[官方挑战口径](https://www.docvqa.org/challenges/2020)、[DocInsights 2026 ANLS 审计](https://aclanthology.org/2026.docinsights-1.6/)、[ChartQA](https://arxiv.org/abs/2203.10244)、[AudioBench](https://arxiv.org/abs/2406.16020)、[Video-MME](https://arxiv.org/abs/2405.21075)、[MLVU](https://arxiv.org/abs/2406.04264)、[MMLongBench-Doc](https://arxiv.org/abs/2407.01523)、[Lost in the Middle](https://arxiv.org/abs/2307.03172)、[RULER](https://arxiv.org/abs/2404.06654)、[NoLiMa](https://arxiv.org/abs/2502.05167)、[LongBench](https://arxiv.org/abs/2308.14508)、[LongBench v2](https://arxiv.org/abs/2412.15204) 和 [LongBench Pro](https://arxiv.org/abs/2601.02872)。文中数字均保留来源版本、样本、prompt、harness 与任务边界，不跨 benchmark 排分。
- `formula_status`：核 ANLS 归一化编辑距离和 `max_j` 多参考答案聚合；`Paris` vs. `Answer: Paris` 的最少删除数为 8，输出长度 13，`8/13≈0.615>0.5`，因此按该阈值得 0。ChartQA 5% numeric tolerance 依据原论文，明确不能自动处理接近零、单位或业务风险。未使用公式渲染器。
- `code_status`：§14 标准库 demo 未运行。静态确认所有 9 个 `signals` 失败时都有对应 action，包括原缺少的 `latency_within_budget`；当前 `P95=5100ms` 小于 toy 门槛 `6000ms`，因此预期 action 列表不变。门槛为教学逻辑，不是官方或生产阈值。
- `sync_status`：新增研究笔记；`PAPERS.md` 加入 benchmark 方法线和第七章正文入口；`EXERCISES.md` 新增 6I benchmark portfolio 题；`INTERVIEW_BANK.md` 新增 49E benchmark 方案比较题。第七册目录已包含此章，无需导航修改。
- `validation`：确认一级章节 §1—§20 连续、研究笔记及配套文件引用存在、ANLS/DocVQA 的任务区分和 demo action 对应关系；静态检查代码围栏和 Markdown 补丁格式。未运行 demo、测试套件、benchmark、外部公式渲染、PDF 构建或全章外链 HTTP 扫描。
- `open_risks`：toy demo 没有目标多模态/长文本语料、模型、标注和 holdout，不能作为目标业务质量或有效上下文长度证据。公开来源结果均绑定来源数据/模型/harness；DocInsights 是单篇新审计，LongBench Pro 只核对作者提交版本。见 `chapter7_chapter7_open_risks`。
- `status`：第七册第七章完成研究驱动精修和配套同步；全书 goal 继续，下一章为第七册第八章 `08-safety与robustness-eval.md`。

### 第七册第六章：Reasoning、Code、Math Eval 资料驱动精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 复读 `book-07-evaluation-experiments/chapters/06-reasoning-code-math-eval.md` §1—§17；保留答案与过程评估、等价答案、数学证明、代码测试和沙箱、`pass@k`、self-consistency、verifier、统计报告、错误定位、demo、闭环和练习。来源与方法对照见[研究记录](research/reasoning-code-math-eval-depth-review-2026-10-27.md)。
- `changes`：补 MATH 与 GSM8K 的任务边界，并比较 GSM-Symbolic 数值变体、GSM-NoOp 无关条件扰动；区分 outcome RM、逐步 PRM 和 ProcessBench 第一错定位，加入 PRM800K 与 ProcessBench 的实验机制、数据/标注代价及泛化限制；对照 HumanEval/EvalPlus 的函数契约和差分测试、LiveCodeBench 的时间新题、SWE-bench Verified 的仓库 patch 与 `FAIL_TO_PASS`/`PASS_TO_PASS` 回归判定。加入 2026 年 OpenAI 官方停止报告 Verified 的后续审计，同时明确这是发布方自审而非独立共识。增加固定回归题、滚动新题、私有 holdout 的分工，未删除已有知识点。
- `evidence`：[MATH](https://arxiv.org/abs/2103.03874)、[GSM-Symbolic](https://arxiv.org/abs/2410.05229)、[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)、[PRM800K](https://github.com/openai/prm800k)、[ProcessBench ACL 2025](https://aclanthology.org/2025.acl-long.50/)、[EvalPlus NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/43e9d647ccd3e4b7b5baab53f0368686-Abstract-Conference.html)、[LiveCodeBench](https://arxiv.org/abs/2403.07974)、[SWE-bench Verified 初始设计](https://openai.com/index/introducing-swe-bench-verified/) 和 [2026 后续审计](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)。方法的输入、结果、限制和正文映射见研究笔记，不把不同任务的数字横向排名。
- `formula_status`：复核 MATH final-answer exact match 和 ProcessBench 第一错索引定义；保留原有 `pass@k` 公式并复核 demo 中 `n=10` 候选的 `pass@1`/`pass@5` 以及答案熵、投票差、均值的手算预期。未使用外部公式渲染器。
- `code_status`：§12 demo 未运行。将隐藏测试“少于 8 条”这一任意计数门槛改为 required/covered behavior slices 差集；self-consistency 从质量门禁改为 greedy 差值诊断；为 accuracy、解析率、单次代码可靠性、隐藏覆盖、过程标签和污染风险每个失败门禁补后续动作；污染标记改为隔离审查而非直接定罪删除。说明简单数字 parser 的适用范围，并将输出标为按代码路径手工核对的预期摘要。本轮未运行 Python、测试套件或生成代码执行环境。
- `sync_status`：新建本章研究记录；`PAPERS.md` 增加第六章正文入口及六类代表来源；`EXERCISES.md` 增加 6H 方案比较；`INTERVIEW_BANK.md` 增加 49D 面试比较题。第七册目录已包含本章文件，无需调整目录。
- `validation`：检查 §1—§17 编号及新增小节顺序、来源 URL、研究记录链接、demo 输入/门禁/预期输出和配套入口；`git diff --check` 通过。没有执行 demo、测试套件、外部公式渲染或 PDF 构建。
- `open_risks`：教学数值和门槛不是模型基准；没有目标模型输出、私有 holdout、仓库镜像、训练语料/cutoff、专家标注或 target runner。论文与 OpenAI 官方 audit 的数字均绑定来源设置，不能据此声称目标系统数学/代码/过程能力或 benchmark 无污染。`extract_number` 不是通用 parser；Demo 只验证静态逻辑，未运行。
- `status`：第七册第六章完成资料驱动精修和配套同步；全书 goal 继续，下一章为第七册第七章 `07-多模态与长上下文评估.md`。

### 第七册第五章：污染检测与可信评估资料驱动精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 复读 `book-07-evaluation-experiments/chapters/05-污染检测与可信评估.md` §1—§14。保留数据流风险、血缘、污染类型、exact/near duplicate、MinHash/LSH、SimHash、embedding、代码/数学近重复、阈值人工校准、时间切分、canary、行为证据、扰动、私有 holdout、多源评估、任务差异、报告、诊断 demo 与证据边界；内容映射和来源限制见[研究记录](research/benchmark-contamination-depth-review-2026-10-27.md)。
- `changes`：区分“训练链路实际含有相同记录”“公开来源发现重合文本”“模型行为异常”和“benchmark 成绩不迁移”，不再把 exact match 或统计信号直接写成分数污染证明。补 Oren canonical/shuffled 序列似然的可交换性、log-prob 接口及窄证据范围；补 ConStat 的参考模型分位数难度校正公式、性能泛化定义和 bootstrap 限制；补 PaCoST 原题/改写题 `P(Yes)` 配对置信、单侧 paired t-test 与改写假设；比较发布前水印和 AntiLeakBench cutoff 后更新事实集；以 Fu 等人 50 篇综述和 MIA 案例研究说明检测器假设需要审计。报告不把异质证据合为单一污染率；案例样本占比注明仅为说明。
- `evidence`：[Oren 等人 ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/46e624c244cff669223d488defd4e835-Abstract-Conference.html)、[ConStat NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a7f89793b9e6f8c6568dbbb6ff727b9b-Abstract-Conference.html)、[PaCoST](https://arxiv.org/abs/2406.18326)、[Detecting Benchmark Contamination Through Watermarking](https://arxiv.org/abs/2502.17259)、[AntiLeakBench ACL 2025](https://aclanthology.org/2025.acl-long.901/)、[Fu 等人 Findings NAACL 2025](https://aclanthology.org/2025.findings-naacl.291/)、[LiveBench v2](https://arxiv.org/abs/2406.19314v2)。来源机制、论文结果范围和限制已记录到研究笔记；作者方法结论不外推为目标模型事实。
- `formula_status`：补 ConStat 参考分位数映射 `q=F_Dref(S_Dref(M))`、预期原集分数 `F_D^-1(q)` 和非泛化效应 `δ=S_D(M)-F_D^-1(q)`；PaCoST 用配对置信差 `d_i=c_i-c_i'` 检验均值是否显著大于零。公式符号和正文含义逐项核对，未调用外部公式渲染器。
- `code_status`：§12 demo 静态复读，未执行 Python 或测试套件。exact digest 改为 NFC、casefold 和空白折叠并保留标点；near shingle 单独抽取 Unicode 字母/数字；toy train 文本与 eval join 对齐。manifest 缺失和 holdout 未隔离现在都产生动作。输出改为源码逻辑下的预期结果，不称实际运行输出。
- `sync_status`：新建本章研究记录；`PAPERS.md` 增加本章入口及 Fu、AntiLeakBench、水印论文；`EXERCISES.md` 增加方法比较题；`INTERVIEW_BANK.md` 澄清可信报告不能随意给污染率，并增加五条路线的选择题。第七册目录已有本章文件入口，无需改导航。
- `validation`：章节 §1—§14 连续，50 个 `~~~` 分隔符成对；核对新增公式符号、示例输入/门禁、研究记录链接及 PAPERS/EXERCISES/INTERVIEW_BANK 同步；全工作区 `git diff --check` 通过。没有执行 demo、单元测试或 PDF 构建。
- `open_risks`：没有目标模型训练 manifest、可授权的实际训练语料、闭源模型序列 log-prob、目标 cutoff/后训练与检索数据、专家标签或 holdout 访问日志；不能给出目标模型的污染率、无污染证明或成绩因果归因。Oren/ConStat/PaCoST/水印/AntiLeakBench 的数字和结论都绑定论文设置。
- `status`：第七册第五章完成资料驱动精修和配套同步；全书 goal 继续，下一章为第七册第六章 `06-reasoning-code-math-eval.md`。

### 第七册第四章：LLM-as-a-Judge 资料驱动精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 复读 `book-07-evaluation-experiments/chapters/04-llm-as-a-judge.md` §1—§19。保留原有测量对象、任务契约、judge 形式、prompt/rubric、结构化输出、偏见和攻击面、人工 gold 校准、交换测试、统计稳定性、多 judge 路由、RAG 案例、诊断 demo、生产 pipeline、Human Eval 衔接、失败模式和练习；研究来源与知识点映射见[研究记录](research/llm-as-a-judge-depth-review-2026-10-27.md)。
- `changes`：比较 G-Eval、Prometheus 2、SAJA 各自解决的缺口、机制、基准结果与限制；加入 G-Eval 评分 token 概率加权式、Prometheus 2 训练数据规模和“分别训练后合并”流程；对照 Chiang 与 Lee 对自动 CoT/数字输出/评分说明的研究；用 rubric-only baseline 与答案/标准反事实补充 judge 敏感性审计；区分类别一致、模型排序、绝对分数校准和自动放行所需指标。按 position-bias 论文 v9 修正为 15 个 judge、22 类任务、约 40 个候选模型、超过 150,000 次评估，并把 prompt 组成长度影响写为较弱而非无影响；按 Thakur 等人的论文将高一致下的分数差写为 10–20 分而非“10 个百分点”。修正 demo 对有效率、零分母和失败信号的处理，保证每个失败门禁都有对应 action。原有章节知识点未因补新研究而删除。
- `evidence`：[G-Eval](https://aclanthology.org/2023.emnlp-main.153/)、[Chiang 与 Lee](https://aclanthology.org/2023.findings-emnlp.599/)、[Prometheus 2](https://aclanthology.org/2024.emnlp-main.248/)、[Shi 等人位置偏差研究](https://arxiv.org/abs/2406.07791)、[Thakur 等人](https://aclanthology.org/2025.gem-1.33/)、[LLM-as-a-Judge 综述](https://arxiv.org/abs/2411.15594)、[SAJA](https://aclanthology.org/2026.acl-industry.45/) 与 [rubric artifacts 研究](https://aclanthology.org/2026.emnlp-main.2612/)。工程接口参考 LangSmith evaluator 文档和 OpenAI Graders API；它们只证明公开接口，不证明目标评估器有效。具体证据范围及论文间不可直接横比的原因见研究记录。
- `formula_status`：按 G-Eval `score=Σ p(s_k)×s_k` 手算评分 token 3/4 概率质量 0.6/0.4 得 3.4；区分 SummEval 离散版 Spearman ρ=0.514 / Kendall τ=0.418 与概率加权版 ρ=0.502 / τ=0.446，并保留论文对离散版 τ 并列分数影响的警告；核对多维校准头 `ŷ=fθ(x)` 的样本、损失和独立 holdout 解释。手工复核 §12 七样本：judge/人工胜率分别为 0.857/0.643，差 0.214；人类标签一致率 0.714；交换一致率 1.0；位置 A/B 各 7 胜；更长回答胜率 1.0；高风险 RAG 分歧仅 `rag_001`。这些是合成输入的手算值，不是本地运行结果。
- `code_status`：静态复读 §12 标准库 demo，检查 schema 字段、orig/swap canonical winner 映射、空分母约定、分层列表、偏差信号、阈值动作和输出字段；按代码路径手算预期输出。未运行 Python，不声称代码执行验收通过；章节已把输出标题改为“按代码逻辑手工核对的预期输出”。
- `sync_status`：新增本章研究记录；`PAPERS.md` 加入 G-Eval、Chiang/Lee、Prometheus 2、位置偏差、Thakur、综述、SAJA 和 rubric artifacts 论文，并将本章列入正文入口；`EXERCISES.md` 新增 6G 路线比较与反事实审计题；`INTERVIEW_BANK.md` 新增 48O 方案选择与验证追问。没有新增章节文件，无目录结构变更。
- `validation`：核对 §1—§19 标题顺序、公式和论文数字的范围限定、研究记录与配套入口；手算 demo 关键汇总值；静态检查 Markdown 内链、围栏数量和平衡情况；`git diff --check` 通过。未运行 demo、自动测试、外部公式渲染或 PDF 构建。
- `open_risks`：没有目标业务请求、目标语言专家 gold、真实 judge 响应或生产流量；不能据论文结果声称目标任务精度、校准、长度/位置偏差或安全漏报。反事实样本需领域人员审核为最小对照。SAJA 与其他论文数字绑定各自模型、标签、基准和评测配置。§12 demo 仍未执行，代码运行验收保持开放。
- `status`：第七册第四章完成资料驱动精修、核心知识点承接、机制/数字与证据边界补充、demo 静态复读及配套同步；全书 goal 继续，下一章为第七册第五章 `05-污染检测与可信评估.md`。

### 第七册第三章：Human Eval 与 Pairwise Eval 资料驱动精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 复读 `book-07-evaluation-experiments/chapters/03-human-eval与pairwise-eval.md` §1—§18。保留任务契约、绝对评分、pairwise、tie、Elo/Bradley–Terry/Arena、标注协议、一致性、偏差、样本量、生产记录、企业案例、教学 demo、人评与 judge 配合、失败模式和练习；旧稿覆盖与来源映射见[研究记录](research/human-eval-pairwise-depth-review-2026-10-27.md)。
- `changes`：深化开放任务构念、rubric、样本与系统边界；把 OAA/ORA/SPA 及 Likert 聚合效用假设展开为方案对比；补充 tie 折半后的三值方差推导、用户/prompt 聚类区间和按配对结构估样本量；比较 Elo、BT、Arena、MT-Bench、AMIP 与低质量票压力测试分别解决的问题及证据边界；扩展仲裁/分歧原因、类别失衡下一致率解释、信度与效度区别，以及人评报告的最小可复现字段。章节保留此前全部核心问题和读者入口，新增研究未取代原有主线。
- `evidence`：阅读人评方法综述、OAA/ORA/SPA 原论文、Arena 论文、MT-Bench/Arena 人评对照、位置偏差研究、计算语言学标注一致性综述、标注分歧解决方案和 ACL 2026 人评报告审计；另核对 Arena 低质量票预印本及 ICLR 2026 最终版偏好删点研究。SPA 的 5/5 对 2/5 结果绑定 GPT-3 故事生成；Arena 规模与语言比例绑定 2024-01 快照；4.4K/6.8K 是论文指定留出数据重放；10% 低质量票是预印本模拟；2/29 票是固定数据上的最坏情况删点阈值；ACL 2026 审计只描述作者筛选的 *CL 论文与报告准则。
- `formula_status`：手算 (W=420,L=350,T=230) 时折半偏好率 0.535、∑z²=477.5、样本方差约 0.1915、iid SE 约 0.0138 和 95% 正态近似区间 [0.508, 0.562]；核对约 784 个独立 prompt 的初步功效预算。另手算 demo 的 12 条胜负平、六题 prompt 均值和样本 SE、位置/长度信号及旧新绝对均分；说明聚类和设计权重会改变区间。
- `code_status`：静态检查 demo 的样本映射、每 prompt 平均、胜负平、方差、位置/长度偏差、切片与风险分歧；未运行 Python，不声称代码运行验收通过。toy 阈值和数据均为教学输入，没有目标产品标签或用户实验。
- `sync_status`：新增本章 research note；更新 `EXERCISES.md` 6F 的 tie/聚类/票质量练习、`INTERVIEW_BANK.md` 48N 的报告规范追问和 `PAPERS.md` 两项新来源；术语表已有 OAA、ORA、SPA、Elo、Bradley–Terry、一致性系数、prompt-cluster bootstrap 与 AMIP 词条。
- `validation`：核对 §1—§18、子标题与引用；逐行复核胜负平与代码示例汇总数值；检查 demo 的 prompt-level estimand 与区间说明；静态核对新增内部链接目标和研究来源版本；`git diff --check` 通过。未运行 demo、自动测试、外部公式渲染或 PDF 构建。
- `open_risks`：没有目标产品流量、用户群、专家 gold、候选响应或在线实验；toy iid 区间不覆盖用户/会话聚类。Arena、SPA、MT-Bench、低质量票模拟、AMIP 和 ACL 2026 报告审计均不能跨其样本、任务、版本或研究假设外推；章节 demo 运行验收仍开放。
- `status`：第七册第三章完成资料驱动精修、旧知识点承接、公式/数值静态复核和配套同步；全书 goal 继续，下一章为第七册第四章 `04-llm-as-a-judge.md`。

### 第七册第二章：Benchmark 设计资料驱动精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 复读 `book-07-evaluation-experiments/chapters/02-benchmark设计.md` §1—§16。保留构念、任务契约、schema 与血缘、代表性/均衡/风险采样、难度/覆盖/区分度、标注与判定、任务指标、协议、污染控制、manifest、版本、统计、企业 RAG 案例、发布证据等级和诊断 demo；旧稿覆盖与研究证据见[研究记录](research/benchmark-design-depth-review-2026-10-27.md)。
- `changes`：把模型总体分数跨度改名为 `R_score`，与题目级 discrimination 分开；补 CTT item-rest / IRT 依赖的模型集、样本和拟合边界。区分目标分布 p 与 benchmark 样本分布 q，解释 D_mix 的互斥穷尽切片条件，并加入过采样后的加权估计算例：q 加权 0.803、目标 p 加权 0.861、D_mix=0.30；说明权重无法修正切片内精选和覆盖缺失。扩展 GPQA 的分阶段专家/非专家验证与 546 Extended / 448 main / 198 Diamond 选择效应；展开 IFEval 的 541 prompts、四种 strict/loose 粒度及 loose 的假阳性代价；区分 LiveBench v2 论文更新设计与项目主页所列快照/维护策略；说明选项顺序与答案选择扰动研究在其设置中的最多 8 位名次变化和本地复核方式。补 Cohen kappa 的偶然一致解释、任务边界及空分母；增加零分母、无答案样本、pass@k 候选不足、micro/macro 解释和加权 bootstrap 的边界。诊断 demo 明确不扫描语料，只消费预先计算的风险标记；缺字段不会让 schema 诊断提前崩溃，空数据集也会显式触发失败门禁，并为全部失败信号提供 action。
- `evidence`：阅读 GPQA、IFEval、LiveBench v2、When Benchmarks are Targets、IRT 预印本和 HELM / lm-eval 方法资料；核验 ONS 抽样/加权说明、Cohen kappa 原始方法、计算语言学标注一致性综述、scikit-learn `zero_division` 文档及 OpenAI Evals 迁移状态。GPQA 不同子集数字、LiveBench 论文与主页维护口径及论文所报名次变化范围都绑定来源和版本；细节见研究笔记。
- `formula_status`：手算目标比例 p=(0.70,0.15,0.10,0.05) 与样本比例 q=(0.40,0.25,0.20,0.15) 的 D_mix=0.30；切片分数 (0.92,0.78,0.60,0.80) 的 q 平均为 0.803、p 加权为 0.861。原 demo 的任务占比为 qa=2/7、其他五类各 1/7，D_mix≈0.079；唯一失败信号是 leakage_clear。复核 accuracy、Recall@k、EvidenceSupport、ASR、over-refusal、pass@k、micro/macro 的分母语义；Cohen kappa 在 p_e=1 时未定义。未用外部公式渲染器。
- `code_status`：静态审阅诊断 demo 的 schema 缺字段处理、空集合路径、计数、指标映射、manifest 字段、各 signal/action 一一对应和预期输出；未运行 Python，因此不声称代码运行验收通过。固定字段 `train_similarity=0.91` 只模拟上游风险标记，阈值 0.80、mix_distance 0.25 和所有样本均为 toy 输入；没有读取训练语料或证明污染。
- `sync_status`：新增本章 research note；`EXERCISES.md` 增 6E benchmark 设计对照练习；`INTERVIEW_BANK.md` 增 48L GPQA/IFEval/LiveBench 与协议扰动追问；`PAPERS.md` 补 benchmark 研究、GPQA 子集边界与 LiveBench v2 入口。第七册目录已映射第二章，章节引用下一章和第十章作为深入阅读。
- `validation`：核对 §1—§16、§5.5.1—§5.5.4 与 §7.8 标题、66 行 `~~~` 围栏为 33 对；手算加权估计和 demo 汇总值；检查本章相对 Markdown 链接目标和新增索引入口；`git diff --check` 通过。未运行 demo、自动测试、外部公式渲染或 PDF 构建。
- `open_risks`：没有目标流量、训练语料、独立候选模型评测、专家标注、线上试验或生产判定器；样本权重示例不能替代真实抽样设计。GPQA、IFEval、LiveBench 和 leaderboard sensitivity 结论分别绑定作者子集、版本、模型与协议。项目主页“最新快照”会变化；本轮仅记录 2026-10-27 核验到的页面状态。demo 代码运行验证保持开放。
- `status`：第七册第二章完成资料驱动精修、旧知识点承接、静态公式/示例核对和索引同步；全书 goal 继续，下一章为第七册第三章 `03-human-eval与pairwise-eval.md`。

### 第七册第一章：评估总览资料驱动精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 复读 `book-07-evaluation-experiments/chapters/01-评估总览.md` §1—§14。保留评估多维证据、任务契约、六类 eval、benchmark 地图、指标公式、judge、人评、统计不确定性、线上验证、客服 RAG 案例和可运行诊断 demo；逐条研究来源与旧稿知识点去向见[研究记录](research/evaluation-overview-depth-review-2026-10-27.md)。
- `changes`：补充 HELM、lm-eval、MT-Bench/Chatbot Arena、ConStat 与持续评测工程循环分别解决的评估缺口、机制、收益和边界；澄清 offline/online 是运行环境维度、human/automatic 是评分来源维度、safety/regression 是评估目的维度，可在同一实验中组合。加入 pairwise 样本顺序交换诊断，并明确 judge 稳定不代表判分正确。核对 OpenAI 官方托管 Evals 弃用时间，区分开源 repository 与 dashboard/API，记录当前只读/关闭时间和迁移指南。
- `evidence`：阅读 HELM 原论文、lm-eval 维护者复现经验总结与官方实现、MT-Bench/Chatbot Arena、位置偏差系统研究、ConStat；核验 OpenAI evaluation best practices、API deprecations 与 Promptfoo 迁移文档。资料证明的对象和不能外推的范围在[研究记录](research/evaluation-overview-depth-review-2026-10-27.md)逐项登记。
- `formula_status`：保留原有 accuracy、micro/macro、pairwise、pass@k、RAG、ASR/over-refusal、ECE、单位成功成本、bootstrap 与两比例近似公式；手算 demo 加权旧/新分数 `8.99/11.2≈0.8027`、`9.383/11.2≈0.8378`，case-level 比较为 5 胜、3 平，教学胜率 `0.8125`，线上 `z≈1.772`。示例量表、权重、三条 unsafe 样本和 `z≥1.64` 均为演示输入，不构成发布门禁。
- `code_status`：静态复读多层诊断 demo；把旧字段 `pairwise` 改名为 `case_score_comparison`，显式假设各行已归一到同一 0–1 方向量表，并让质量、slice、回归、安全、线上、延迟、成本失败都生成动作/failed gate。没有执行代码，因此只记录手算值和源码检查，不声称运行验证完成。
- `sync_status`：新增本章研究记录；`EXERCISES.md` 增加对 HELM、harness、人评/pairwise、ConStat 与线上实验的评估链设计题；`INTERVIEW_BANK.md` 扩展 HELM/lm-eval 对照；`PAPERS.md` 增位置偏差论文入口。第七册目录原已包含第一章，无需导航改动。
- `validation`：复核 §1—§14 标题、60 行 tilde fence 组成 30 对；新增研究记录、Sol 历史 toy 与第七册后续章节文件目标存在；手算 demo 分数与正文预期输出对应；`git diff --check` 通过。未运行 demo、自动测试或 PDF 构建。
- `open_risks`：没有目标客服数据、judge gold set、线上随机实验、benchmark 环境或实际 OpenAI Evals 数据迁移演练；章节公式与案例不代表业务测量。Evals 平台时间表只绑定 2026-10-27 官方状态，后续实施前需重查。
- `status`：第七册第一章完成资料驱动精修和配套同步；全书 goal 继续，下一章为第七册第二章 `02-benchmark设计.md`。

### 第六册第十八章：GPT-6 Astra 长上下文与工具预算资料驱动精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 复读 `book-06-llm-deployment/chapters/18-gpt-6-astra长上下文与工具预算.md` §1—§11，并对照原稿逐项保留长上下文、reasoning effort、模态/工具、费用示例、runtime、状态 replay、动态 effort、tool search、async、steering、skills、安全监控、compaction、Codex/Sol/Luna/6.1 Sol 及 GPT-OSS 对照和原有练习。逐节证据与去向见[研究记录](research/gpt6-astra-long-context-tool-budget-depth-review-2026-10-27.md)。
- `changes`：把 Astra 的 922K 从“官方 maximum input”修正为 `1,050,000−128,000` 的条件算术余量；补 mode 与 effort 正交、reasoning token 占输出预算和 incomplete 诊断；按当前价页补 Standard/Batch/Flex/Fast/Ultrafast、272K whole-request threshold、30m/12h cache writes 和区域费用边界；将“工具预算”扩展为跨 response 的 job 级调用/并发/timeout/重试/验证/deadline/cost ledger。精化 configuration update 的缓存与兼容边界、async/steering 状态、误对齐监控覆盖矩阵、两类 compaction replay 规则；修正 GPT-5.3 Codex Deprecated、endpoint 和 model page 未单列 max input 口径。旧稿其他模型对照、工具权限、skills、compaction、安全、练习等主题均保留并扩深。
- `evidence`：2026-10-27 重新核对 OpenAI 官方 GPT-6 Astra/model family、Reasoning、pricing、Fast/Ultrafast、prompt caching、conversation state、compaction、tool search、async tool calling、steering、misalignment monitoring、skills 博客，以及 Codex、Sol、Luna、6.1 Sol 模型页。官网支持范围及不支持的内部/生产结论记录于[研究笔记](research/gpt6-astra-long-context-tool-budget-depth-review-2026-10-27.md)。
- `formula_status`：Astra 理论余量 `1,050,000−128,000=922,000`，不等于公开 max input；费用代码将 300K input 拆为 140K uncached、100K cached read、60K 30m write，另计 20K output：`2.80+0.20+1.50+1.50=$6.00`。阈值、cache write TTL 和模式倍率均按 2026-10-27 官方价页快照；未包括工具、地域、账户及应用基础设施费用。
- `code_status`：从正文提取零依赖估算示例并执行，stdout 为 `6.0`，与 fenced output 一致；代码只演示公开价格算术，不调用 OpenAI API。没有执行真实 agent 状态机、API 行为或生产请求。
- `sync_status`：修订 `EXERCISES.md` 中 Astra 长度、价格、mode/effort 与 job-budget 题；修订并扩展 `INTERVIEW_BANK.md` 相关问题，新增全局工具预算题；更新 `PAPERS.md` 的官方接口资料入口；更新 `GLOSSARY_EN_ZH.md` 中 Astra 输入预算、steering、工具预算、缓存 TTL、Ultrafast 等定义。Book 6 目录已包含第十八章；下一章按书系顺序为 Book 7 Chapter 1。
- `validation`：核对 §1—§11 顺序、14 条 tilde fence 行组成 7 组围栏、正文输出与执行结果一致；章节、研究笔记、配套代码和 Book 7 下一章链接目标存在；费用手算与 922K 推导复核；`git diff --check` 通过。
- `open_risks`：没有目标 API probe、账户合同价、生产任务/工具 trace、真实延迟或质量评测；mode/tool compatibility、事件时序和安全监控效果仍以 2026-10-27 官方文档为证据。相关限制见 `chapter18_open_risks`。
- `status`：第六册第十八章完成资料驱动精修、旧知识点承接、价格示例运行核对、配套同步与研究记录；全书 goal 继续，下一章为第七册第一章 `book-07-evaluation-experiments/chapters/01-评估总览.md`。

### 第六册第十七章：Native INT4 与原生量化部署资料驱动精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 完整复读 `book-06-llm-deployment/chapters/17-native-int4与原生量化部署.md` §17.1—§17.18。保留原有量化数学、显存账本、PTQ/QAT、artifact/manifest、kernel 与并行布局、校准、任务质量、容量、KV/speculative/multimodal/adapter 组合、启动回退、灰度回滚、demo 和企业取舍；逐节覆盖与来源判断见[研究记录](research/native-int4-deployment-depth-review-2026-10-27.md)。
- `changes`：区分有符号 INT4 的二进制补码范围 `[-8,7]` 与常见对称量化码本 `[-7,7]`，新增手算和反量化；展开 GPTQ 的局部层重建目标、校准激活近似 Hessian、逆 Hessian 误差补偿、分块/稳定化和论文实验边界；展开 AWQ 的等价通道缩放、`s_X^α` 搜索与 clipping，并澄清全低比特主路径不要求保留 1% FP16 权重；补 fake quant、LLM-QAT 案例，并比较 RTN/GPTQ/AWQ/LLM-QAT 与 MARLIN 的算法、训练、artifact、kernel 层次。修正单位成功成本的原始请求分母和内存并发公式中固定权重/每请求增量的关系；其余原知识点保留。
- `evidence`：阅读 GPTQ 与 AWQ 原论文、TACL 量化综述、LLM-QAT Findings of ACL 2024、MARLIN、PyTorch FakeQuantize 2.14 和 vLLM 当前量化/linear backend 文档。GPTQ OPT-175B PPL 8.34/8.37/10.54、AWQ OPT-6.7B INT3-g128 PPL 10.86/11.39/23.54、LLM-QAT LLaMA-7B W4A8KV4 41.2/60.7、MARLIN A10 层级 batch 16–32/128 的速度报告，均分别绑定论文的模型、数据、任务、GPU、kernel 和配置；不跨表排名。具体证据与边界见[研究记录](research/native-int4-deployment-depth-review-2026-10-27.md)。
- `formula_status`：INT4 示例 scale `1.1/7≈0.15714`，码值 `[-6,-1,2,7]`，反量化约 `[-0.9429,-0.1571,0.3143,1.1]`；GPTQ 二坐标例 `0.50−0.20×0.8/1=0.34`；单位成功成本 `120/90=1.33`。并发式扣除常驻权重和固定占用后，才按每请求 KV/workspace 增量估算；它是线性下界，真实容量需 runtime profile。
- `code_status`：运行 §17.14 标准库 Python demo，输出与正文 text 围栏逐字一致：BF16/INT4 总账本 450/165 GiB、权重空间缩减 0.713、卡数下界 7/3、质量失败切片为 `code_json`，建议选择性灰度。所有数值/质量/容量字段均为合成教学输入；代码不实现 INT4 pack、GPU kernel 或目标模型评测。
- `sync_status`：`EXERCISES.md` 阶段 2026-08 新增第 21—25 题；`INTERVIEW_BANK.md` 新增第六册第十七章五题；`PAPERS.md` 新增 LLM-QAT、MARLIN、PyTorch FakeQuantize 入口；`GLOSSARY_EN_ZH.md` 修订 Native INT4 通用定义并补 MARLIN。第六册目录已含第十七章，无导航变更。
- `validation`：核对 §17.1—§17.18 连续、38 组 Markdown 围栏配对；demo stdout 精确匹配；本章本地 Markdown 链接目标存在；手算 INT4 反量化、GPTQ 更新和单位成功成本复核通过；`git diff --check` 通过。
- `open_risks`：没有目标 checkpoint、量化工具链、GPU、runtime/kernel、实际请求 trace 或生产 SLO 验收。demo 的显存与质量是合成值，论文结果不能外推到目标业务；vLLM latest 等滚动文档须按目标 release 重核。未做外部公式渲染或 PDF 构建。
- `status`：第六册第十七章完成资料驱动深度精修、原知识点承接、公式/demo 核验与配套同步；全书 goal 继续，下一章为第六册第十八章 `18-gpt-6-astra长上下文与工具预算.md`。

### 第六册第十六章：FP8 KV Cache 资料驱动深度精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 复读 `book-06-llm-deployment/chapters/16-fp8-kv-cache.md` §1—§13。保留 KV 动态账本、MHA/GQA/MQA 容量公式、scale 设计、K/V 不对称、cache 生命周期、prefix sharing、抢占/swap、分页 allocator、任务路由、单位成功成本、长上下文评测、manifest、灰度/回滚、demo、误区与总结；来源和原稿知识点映射见[研究记录](research/fp8-kv-cache-depth-review-2026-10-27.md)。
- `changes`：补 OCP E4M3/E5M2 的指数/尾数、范围、normal/subnormal 和特殊值，并区分 OCP 与 FNUZ 编码；解释 OCP 不定义 scale 选择/维护；加入 E4M3 scale 数值算例、动态 scale 的双读成本和 append 时 scale 变更必须重编码的状态不变量。加入 K/V 两位置 softmax 推演，并比较 KIVI、KVQuant 与 vLLM per-head FP8 各自解决的问题及代价；用 vLLM 公开 serving 案例区分量化误差、attention 累加误差、prefill/decode 和单请求/并发性能指标。明确示例质量分数与 metadata 开销为合成值，补 scale/variant/attention/accumulation 字段及输入校验。
- `evidence`：核对 [OCP OFP8 v1.0](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)、[FP8 原论文](https://arxiv.org/abs/2209.05433)、[ROCm FP8 类型表](https://rocm.docs.amd.com/en/docs-7.2.3/reference/precision-support.html)、[PyTorch dtype 表](https://docs.pytorch.org/docs/main/tensor_attributes.html)、KIVI、KVQuant、2026 KV serving survey、vLLM/LLM Compressor 当前 KV 文档、[vLLM 2026 FP8 serving report](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html)、TensorRT-LLM FP8 guide 与 Transformer Engine 2.20.2 Current Scaling。来源支持范围和限制逐项写入研究记录；动态框架文档按部署 release 重核。
- `formula_status`：`M_KV≈2LBTH_kvD_hb`，在 `L=64,B=1,T=131072,H_kv=8,D_h=128` 下 BF16 为 32 GiB、FP8 payload 为 16 GiB；加 8% 教学 metadata 假设为 17.28 GiB。1M token 同配置为 BF16/FP8 256/128 GiB。E4M3 `s=500/448≈1.1161` 时 `x=1` 约重构为 0.9766；logits `[1,0]→[0.7,0]` 的 softmax 第一权重约 `0.731→0.668`。这些是理论/教学数，不是运行时观测。
- `code_status`：运行 §10 标准库 demo，stdout 与正文 fenced `text` 逐字匹配；验证 18 类非法维度、bytes/value、overhead 和任务分数输入均按预期抛出 `ValueError`。没有运行 FP8 cast、attention kernel、GPU 或实际 cache manager。
- `sync_status`：`EXERCISES.md` 新增五项算例/方案对比/发布验收练习；`INTERVIEW_BANK.md` 新增五题；`PAPERS.md` 补 OCP/FP8 原论文、serving report 和官方 runtime 文档；`GLOSSARY_EN_ZH.md` 更新 E4M3/E5M2、FP8 KV/OFP8，并增加 FNUZ 与 two-level accumulation。
- `validation`：§1—§13 标题连续；代码围栏成对；章节与研究记录本地链接存在；代码输出和边界输入验证通过；`git diff --check` 通过。未做 GPU/真实模型评测、外部公式渲染或 PDF 构建。
- `open_risks`：未在目标模型、GPU、vLLM/TensorRT engine 或生产 trace 验证。8% metadata 开销与任务质量门限均为合成数；serving 报告与 KIVI/KVQuant 结果绑定作者配置，不能外推成目标 workload。MLA/混合注意力、state-space cache 和模型专用 page layout 需另建公式，滚动文档需按目标 release 复核。
- `status`：第六册第十六章完成资料驱动深度精修、代表方案对比、公式与 demo 验收及配套同步；全书 goal 继续，下一章为第六册第十七章 `17-native-int4与原生量化部署.md`。

### 第六册第十五章：FP4、MXFP4 与 NVFP4 资料驱动深度精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 复读 `book-06-llm-deployment/chapters/15-fp4-mxfp4-nvfp4.md` §1—§14。保留 FP16/BF16/INT4 背景、payload 与 scale 账本、五类 dtype、权重/activation/KV/accumulation/logits 差异、kernel/engine/task 四层证据、PTQ/QAT/校准、任务与性能切片、outlier、manifest、fallback、验证矩阵、常见误区和总结；逐节来源及知识点映射见[研究记录](research/fp4-mxfp4-nvfp4-deployment-depth-review-2026-10-27.md)。
- `changes`：区别均匀 INT4 格点与非均匀 E2M1；依据 OCP MX v1.0 补出 E2M1 码本、MXFP4 的 32-value E8M0 scale 与 `4.25 bit/value`；依据 Transformer Engine NVFP4 文档说明 16-value E4M3 + per-tensor FP32 global scale 的一维表示、二维权重 scaling 变体和 `4.5+32/N bit/value` 限定。增加 TensorRT 显式 Q/DQ scale 参数与紧凑 NVFP4 recipe 的接口层次辨析。用数值演算一个 outlier 使 E2M1 `0.9` 映射到 0 的误差，补 4096² 权重的 MXFP4/NVFP4 scale 与压缩账本。增加 MR-GPTQ 的 block rotation、GPTQ 误差回填、format-specific scale search 与发布方 scale-learning 的训练机制/限制；保留所有原有 deployment 主线。
- `evidence`：读取 [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)、[Transformer Engine 2.20.2 NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)、[TensorRT 10.x quantized types](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html) 与 2026-10-27 快照的 [Model Optimizer support matrix](https://nvidia.github.io/Model-Optimizer/guides/0_support_matrix.html)。比较 [Comprehensive Evaluation](https://arxiv.org/abs/2507.17417) 对 scale/granularity/rotation 的统一研究、[MR-GPTQ](https://arxiv.org/abs/2509.23202) 的格式专属 PTQ、[NVFP4 pretraining report](https://arxiv.org/abs/2509.25149) 的训练 recipe，以及 [Scale Learning](https://nvidia.github.io/Model-Optimizer/announcements/20260916-scale-learning.html) 的冻结/联合训练 scales。数字均保留各自作者/发布方 workload，未合并排名。
- `formula_status`：E2M1 非负幅值 `0,0.5,1,1.5,2,3,4,6`；MXFP4 `(32×4+8)/32=4.25 bit/value`；NVFP4 1D `4+8/16+32/N=4.5+32/N bit/value`。`N=16,777,216` 时 FP16 32 MiB、E2M1 payload 8 MiB；MXFP4 scale 0.5 MiB，总 8.5 MiB、约 3.76×；NVFP4 1D scale `N/16+4` bytes、总约 9 MiB、约 3.56×。这些不是含对齐/二维 layout/workspace 的真实 artifact 或 HBM 数字。
- `code_status`：运行 §9 纯 Python demo，输出逐字匹配正文；另确认非正 block size、空 input、NaN、MSE 长度不匹配和负元素数都抛出 `ValueError`。示例用真实 E2M1 离散码值，但 scale 和 metadata 账本为教学假设，不模拟 MXFP4/NVFP4 scale、packing 或 GPU。
- `sync_status`：`EXERCISES.md` 新增四项格式账本、量化方法对照和目标任务评估练习；`INTERVIEW_BANK.md` 新增五题，覆盖有效 bit、E8M0/E4M3、MR-GPTQ、scale learning 与 PTQ/预训练证据边界；`PAPERS.md` 加 MR-GPTQ、NVFP4 预训练、scale-learning 与 TensorRT Q/DQ 来源；`GLOSSARY_EN_ZH.md` 更新 FP4/MXFP4/NVFP4 并增 E2M1、E8M0、MR-GPTQ、NVFP4 Scale Learning。
- `link_status`：章节和研究记录中的本地 Markdown 目标均存在；OCP、Transformer Engine、TensorRT、Model Optimizer 和论文入口直链到对应规范/文档/论文。第六册目录已存在第十五章及后续章节入口。
- `validation`：§1—§14 标题连续；代码 fence 配对；章节/研究 note 的本地链接校验通过；§9 stdout 与 fenced `text` 精确匹配，5 个非法输入边界均按预期拒绝；`git diff --check` 通过。未运行 GPU/框架/模型、外部公式渲染或 PDF 构建。
- `open_risks`：没有目标模型/GPU、量化工具、推理 engine 或 production workload 验证。MR-GPTQ、NVFP4 预训练和 Model Optimizer scale-learning 结果均绑定作者/发布方设置；MR-GPTQ 摘要对主实验的“within 1–2%”概括与附录 I PlatinumBench 的最佳 NVFP4/MXFP4 recovery 差距（96.6%/92.3%，平均超过 4 个百分点）属于不同评测切片，不能合并成通用差距。TensorRT Q/DQ scale 参数与 Transformer Engine 紧凑 profile 的转换需对目标导出器/engine 查源码或实测；二维权重 scale 的物理 storage 需从真实 checkpoint 计算。
- `status`：第六册第十五章完成来源驱动深度精修、原知识点承接、格式/有效位数推演、代码与边界验证及配套同步；全书 goal 继续，下一章为第六册第十六章 `16-fp8-kv-cache.md`。

### 第六册第十四章：部署方案推导与故障复盘资料驱动深度精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 从头至尾复读 `book-06-llm-deployment/chapters/14-部署方案推导与故障复盘.md` §1—§16。原有 70B 显存与并行选择、TTFT、continuous batching、KV Cache、PagedAttention、INT8/INT4、服务生命周期、RAG、prompt injection/工具权限、成本、跨组件排障、设计 demo、误判和总结均保留；逐节映射见[研究记录](research/deployment-design-incident-review-depth-review-2026-10-27.md)。
- `changes`：区分模型 runtime 活跃序列与端到端 job 并发，明确每个模型池按路由调用数与 runtime 槽位时间计算；修复容量示例结果，从原 6/7 口径改为 `ceil(1.2×max(0.84375,4.8,6))=8` 个容量副本、若单副本故障后仍需保留相同 20% 余量则总池为 9。加入 PagedAttention/vAttention 机制比较及 GPTQ/AWQ/SmoothQuant/KIVI 的量化对象、方案边界和待验证改进路径。将原通用复盘清单改成明确标注的合成 prompt 前缀/cache/prefill/queue 时间线，含支持/反证、系统促成因素、无责分析和 owner/验证关闭条件。修正 demo 的吞吐函数加入 active sequence 容量；将成本标成假设费率下的 token-only 账单，安全 signal 改为只表示主题已声明。
- `evidence`：基于 GPTQ、AWQ、SmoothQuant、KIVI、PagedAttention 与 vAttention 原论文和前序量化/KV 专题的细读记录比较机制、作者证据和限制，不跨论文 benchmark 横向排名；核对 [Google SRE Postmortem Culture](https://sre.google/sre-book/postmortem-culture/)、[Google SRE Workbook Incident Response](https://sre.google/workbook/incident-response/) 与 [NIST SP 800-61 Rev. 3](https://csrc.nist.gov/pubs/sp/800/61/r3/final)。复盘模板来源支持事故记录、系统级促成因素和行动项流程，不证明合成事故是真实案例；NIST 来源用于区分网络安全事件响应与性能回滚。
- `formula_status`：输入需 `60×1500×0.75=67,500 token/s`，输出 `60×200=12,000 token/s`，平均 runtime 活跃序列 `60×1.2=72`。单副本瓶颈比例为 0.84375/4.8/6，`γ=1.2` 后为 8；失一副本且仍保留这份余量为 9。理论 70B FP16/INT4 权重为 130.39/32.6 GiB，KV toy 示例 20 GiB；假设 token 价格下每小时 token cost 为 1404.0/1161.0，缓存节省比例 17.3%。未做外部公式渲染。
- `code_status`：提取并运行 §13 唯一标准库 Python 示例，退出码 0、stderr 为空；stdout 与更新后的 fenced `text` 示例逐字相同。输出显示 72 active sequences、8 base replicas、9 N+1 replicas；移除一个副本后，prefill/decode/active-sequence 三个 survivor 检查都覆盖 `γ=1.2` 需求。Token-only cost estimate 为 $1,404/$1,161，清单缺 `throughput.kv_budget`。toy demo 只验证算术和声明型清单，不模拟目标模型、GPU、cache、账单、安全控制、跨区故障、冷启动或容量非线性。
- `link_status`：本章目录已列第十四章；research note 和本章外链指向论文、Google SRE、NIST 等来源；新增练习、题库、论文索引和术语均检查本地目标。未做整章旧链接自动 HTTP 扫描。
- `sync_status`：`PAPERS.md` 增生产复盘与 NIST 来源；`GLOSSARY_EN_ZH.md` 明确 per-model-pool active sequence 公式，并增 runtime occupancy、N+1、prefix token reuse 和 blameless postmortem；`INTERVIEW_BANK.md` 增三道容量推导、量化诊断和证据型复盘题；`EXERCISES.md` 增五项复算/对比/事故/行动项/demo 练习。
- `validation`：复核 §1—§16 标题连续、公式手算、fenced block 配对、本地 Markdown 链接，并再次运行 demo 对照 stdout；`git diff --check` 通过。未运行 GPU/生产 workload、PDF 构建或外部公式渲染。
- `open_risks`：没有目标模型/GPU、请求 trace、cache 指标、线上质量标签、真实服务账单或生产事故证据。副本 sizing、收费和故障情境均为教学假设；安全清单信号不是安全验收；论文与动态框架文档不能证明目标环境的速度、兼容性或质量。
- `status`：第六册第十四章完成来源驱动精修、知识点承接、合成故障分析、demo 输出对照与配套同步；全书 goal 继续，下一章为第六册第十五章 `15-fp4-mxfp4-nvfp4.md`。

### 第六册第十三章：生产系统设计资料驱动深度精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 从首行至末行复读 `book-06-llm-deployment/chapters/13-生产系统设计.md` §1—§16。原有生产目标、同步/流式/异步、SLO、容量与 KV、请求生命周期、路由/缓存、安全、重试/取消/幂等、观测、发布、过载、扩展、多租户、案例、demo、误区和总结均保留；逐节覆盖及资料边界见 [研究记录](research/production-system-design-depth-review-2026-10-27.md)。
- `changes`：把 SLI/SLO 分母、测量边界、错误预算和安全/质量门禁分开；区分端到端任务并发与模型 runtime active sequence，也区分 raw tokens/s 与 SLO goodput；补全抖动退避、总 deadline、重试放大、commit-unknown、幂等键/Transactional Outbox/Saga 的承诺边界；补 control/canary 归因、Prometheus label cardinality、OpenTelemetry GenAI Development 状态和 Kubernetes startup/readiness/liveness 差异。修正 demo 固定 `0.04ms/token` 与模型池 `prefill_tps` 不一致，按同池 prefill 假设重算 §11.3 与 §12 active sequence 和 latency 示例；将 trace 样例字段改为 `cache_saved_token_ratio`。
- `evidence`：复核 Google SRE SLI/SLO 与 canary、AWS 幂等 API 和 timeout/retry/jitter、Microsoft Transactional Outbox、Azure microservice patterns、Kubernetes probes、Prometheus instrumentation 与 OpenTelemetry GenAI metrics；并读 [Revisiting SLO and Goodput Metrics in LLM Serving](https://arxiv.org/abs/2410.14257) 对照 request/token-level goodput 与 smooth-goodput。既有来源包含 vLLM serving/metrics、Kubernetes HPA、OpenTelemetry primer、NIST AI RMF/GenAI Profile、OWASP 和 serving 系统论文。官方文档用于确认操作语义，研究结果按论文 workload/硬件/测量边界解释，不把二者互换。
- `formula_status`：复核 95%/10k 请求预算为 500、5%×20% canary 聚合贡献为 1%、`TTFT+(O−1)TPOT` 的 token 计数边界、token bucket admitted-only debit、两层各三次总尝试最多九次。容量案例的 input/output token/s 为 small 25,200/6,300 和 large 39,600/6,120；job 并发 70/74.4。按同池 prefill 吞吐、首 token 固定开销与 TPOT 得到 active sequence 35.3/54.9，model-path 估算 chat/RAG/Agent 为 1,105/2,299/9,755ms；small/large 分别需 2/5 副本，large 失去一个副本后 decode 为 7,200 token/s。所有数值只绑定教学输入。
- `code_status`：提取 §12 唯一标准库 Python demo，编译执行成功；对照实际 stdout 与正文 `~~~text` 输出。缺少实际同步 P95、任务质量和安全证据时，门禁输出 `pause_and_investigate`；清单项只代表方案声明，不代表运行时验证。
- `link_status`：新增 SRE、AWS、Microsoft、Azure、Kubernetes、Prometheus、OpenTelemetry 官方资料入口；逐个映射机制和适用边界，未对全章既有所有外链执行自动 HTTP 扫描。第六册目录已有第十三章入口，无需改导航。
- `sync_status`：`PAPERS.md` 加入生产可靠性和服务治理资料；`GLOSSARY_EN_ZH.md` 增/细化 SLI、SLO、错误预算、幂等键、Outbox、commit-unknown 和 active job/sequence 区分；`INTERVIEW_BANK.md` 新增容量、SLO、事务恢复、canary 面试题；`EXERCISES.md` 新增 6 项容量/SLO/事务/canary/demo/probe 练习。
- `validation`：核对章节 §1—§16 层级、关键公式、§11/§12 容量数字、demo 编译执行且 stdout 与正文样例逐字一致、研究笔记本地链接和第六册目录导航；全工作区 `git diff --check` 通过。未运行生产压测、外部公式渲染、目标 API 或 PDF 构建。
- `open_risks`：无目标模型/GPU/混合 trace、线上 P95、真实 RAG/工具延迟、质量标签、安全样本或发布演练；capacity 和延迟为教学假设，不可当生产 sizing。OpenTelemetry GenAI conventions 和 vLLM/HPA 等动态资料需按实际部署版本复核。
- `status`：第六册第十三章完成来源驱动精修、原知识点承接、demo 口径修正与配套同步；全书 goal 继续，下一章为第六册第十四章 `14-部署方案推导与故障复盘.md`。

### 第六册第十二章：成本优化资料驱动深度精修验收（2026-10-27）

- `scope`：依据当前 `WRITING_SPEC.md` 从首行至末行复读 `book-06-llm-deployment/chapters/12-成本优化.md` §1—§16。原有成本账本、GPU、路由、prompt 压缩、缓存、speculative decoding、量化/蒸馏、batch/queue/autoscaling、限流、RAG/Agent/多模态、实验、案例、demo、误区和总结均保留；逐节覆盖见 [研究记录](research/cost-optimization-depth-review-2026-10-27.md)。
- `changes`：拆开资源向量与货币核算，避免输入 token、媒体量、GPU 时间直接求和及 API/GPU 重复入账；区分墙钟吞吐与忙碌吞吐，修正利用率重复乘法；补 API/自托管成本交叉点；用 FrugalGPT cascade 与 RouteLLM preference router 说明不同调用路径和 domain shift；比较 LLMLingua、LongLLMLingua、LLMLingua-2；修正 response-cache 期望成本；speculative sampling 期望长度纳入 target token/bonus token；比较 GPTQ/AWQ/SmoothQuant/KIVI，补 KIVI 的 K/V 不同量化轴、分组与 residual 全精度窗口；补 Minitron 的作者成本模型边界、Sarathi-Serve/DistServe 调度差异、M/M/1 排队例子、HPA 控制环和 Agent 历史二次增长。
- `evidence`：核对 [FrugalGPT](https://arxiv.org/abs/2305.05176)、[RouteLLM](https://arxiv.org/abs/2406.18665)、[LLMLingua](https://arxiv.org/abs/2310.05736)、[LongLLMLingua](https://arxiv.org/abs/2310.06839)、[LLMLingua-2](https://arxiv.org/abs/2403.12968)、[Speculative Decoding](https://arxiv.org/abs/2211.17192)、[GPTQ](https://arxiv.org/abs/2210.17323)、[AWQ](https://arxiv.org/abs/2306.00978)、[SmoothQuant](https://arxiv.org/abs/2211.10438)、[KIVI](https://arxiv.org/abs/2402.02750)、[Minitron](https://arxiv.org/abs/2407.14679)、[Sarathi-Serve](https://arxiv.org/abs/2403.02310)、[DistServe](https://arxiv.org/abs/2401.09670)、[vLLM APC](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)、[vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)、[OpenAI Prompt Caching](https://developers.openai.com/api/docs/guides/prompt-caching)、[Kubernetes HPA](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/) 和 [LLM inference survey](https://arxiv.org/abs/2506.21901)。论文数值绑定其模型、任务、价格、GPU、实现和评估口径；动态官方文档需按目标版本复核。
- `formula_status`：复核 cost/success 的 5% 变化、GPU 单位成本约 $11.11/百万输出 token、API/自托管 30M 请求交叉点、压缩净省 $0.0045、response cache 纯经济阈值 3%、speculative E[L]=3.3616、M/M/1 的 0.4s/1.9s、HPA 建议 8 副本和 Agent 历史 12,000 vs 固定摘要 6,000 输入 token。推导只绑定正文列出的合成参数。
- `code_status`：提取 Markdown 内唯一标准库 Python demo，编译执行成功；实际 stdout 与紧邻 text 输出逐字一致。演示成本从 0.05238 降至 0.02292、候选 P95 5,088ms 超过 5,000ms、没有安全证据，输出 require_safety_evidence 并暂停。6 条等权合成样本、价格、success probability 与线性 latency 均不是生产测量。
- `link_status`：章内参考列表已补齐 routing、compression、speculative、quantization、distillation、serving schedule、cache、HPA 官方/原始资料；第六册目录已有第十二章入口。检查新增源链接与证据表映射；未对全章外链做自动 HTTP 扫描，也未运行外部公式渲染。
- `sync_status`：`PAPERS.md` 增 FrugalGPT、RouteLLM、LLMLingua 三路线和 Minitron；`GLOSSARY_EN_ZH.md` 补 Token Cost、Cost per Token、Cost per Successful Answer、Unit Economics、Marginal Cost、Break-even Request Volume 和 request/token Cache Hit Rate；`INTERVIEW_BANK.md` 深化 56、58—60 题的资源计量、代表方案比较、成功成本和交叉点；`EXERCISES.md` 新增第六册第十二章 9 项推导/设计题。无目录变动。
- `validation`：核对章节 §1—§16 连续、原章主题覆盖、公式假设与数值、demo 编译执行和 stdout 对照；`git diff --check` 通过。未做外部公式渲染、生产 workload 实验或 PDF 构建。
- `open_risks`：没有目标 API 合同、请求分布、GPU 成本/利用率、缓存 trace、压缩器运行、量化 kernel、真实任务标签或安全评测；不声称目标生产成本、质量、容量或安全改进。论文结果与 vLLM/OpenAI/Kubernetes 当前文档均按各自版本和条件使用。
- `status`：第六册第十二章完成资料驱动精修、原知识点承接、研究记录、demo 输出对照和配套同步；全书 goal 继续，下一章为第六册第十三章 `13-生产系统设计.md`。

### 第六册第十一章：监控、评估与安全资料驱动精修验收（2026-10-26）

- `scope`：从首行至末行复读 `book-06-llm-deployment/chapters/11-监控评估与安全.md` §1—§15。原有监控/指标、GPU 与 KV 健康、成本、质量评估、幻觉与证据、有害输出、提示注入/Agent、隐私、灰度、事故响应、质量闭环、demo 和小结均保留；覆盖图见 [`research/monitoring-evaluation-safety-depth-review-2026-10-26.md`](research/monitoring-evaluation-safety-depth-review-2026-10-26.md)。
- `changes`：明确并行 span 不可直接相加为 E2E；区分服务 output tokens/s 与逐请求/pooled decode rate；用 HELM、MT-Bench/Chatbot Arena、RAGAS、ARES/PPI 比较场景覆盖、开放回答偏好和 RAG 组件评估；解释 PPI 的预测均值加人工残差校正；补 HarmBench 的行为级攻击/拒答评估与 AgentDojo 的状态化工具 utility/security 评估机制及限制。修正安全 demo 的攻击/安全分组、FNR/FPR/ASR 分母，增加零成功置信上界；补 TTFT 可观测数、质量覆盖率、无效空集语义。
- `evidence`：核对 [HELM](https://arxiv.org/abs/2211.09110)、[MT-Bench/Chatbot Arena](https://arxiv.org/abs/2306.05685)、[RAGAS](https://arxiv.org/abs/2309.15217)、[ARES](https://aclanthology.org/2024.naacl-long.20/)、[Prediction-Powered Inference](https://arxiv.org/abs/2301.09633)、[HarmBench](https://arxiv.org/abs/2402.04249)、[AgentDojo](https://proceedings.neurips.cc/paper_files/paper/2024/file/97091a5177d8dc64b1da8bf3e1f6fb54-Paper-Datasets_and_Benchmarks_Track.pdf) 与 [OpenTelemetry GenAI spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)；[Trustworthy Online Controlled Experiments](https://ai.stanford.edu/~ronnyk/puzzlingOutcomesInControlledExperiments.pdf)。论文结论保留各自模型、场景、数据、攻击预算、判分器和版本边界。
- `formula_status`：零事件单侧上界 `1−α^(1/n)` 在 n=1、α=0.05 时为 0.95；FNR 用真实攻击/危险样本为分母、FPR 用真实安全样本为分母、ASR 用攻击尝试为分母，零分母返回不可计算。E2E 只按非重叠串行路径加和；paired 比较按样本差分，bootstrap 单位与随机化单元一致。
- `code_status`：从 Markdown 提取唯一 Python block，编译执行成功，stdout 与文中输出逐字一致。结果含 8 条合成记录、6 条已送达 TTFT、7 条 TPOT、1 条攻击与 7 条安全样本、1/8 全量 block rate、质量覆盖率 0.75/评分均值 0.88、零 ASR 上界 0.95 和 `hold_for_more_safety_evidence`。空分位数/非法 q/NaN/Inf/布尔值、q=0/1、空均值、零分母、零/单 token、n=0/n=1、负/非有限/布尔阶段时长、均值 NaN/Inf/布尔值、负样本量、非法 alpha、无质量/攻击/安全样本均按约定处理。只验证 toy 数据与计量，不代表模型、GPU、Exporter、目标业务或线上 SLO。
- `link_status`：新增 HELM、MT-Bench、RAGAS、ARES/PPI、HarmBench、AgentDojo、OpenTelemetry 与 Trustworthy Online Controlled Experiments 页面已打开核对；第六册目录已有第十一章。全章旧链接未做全量 HTTP 状态扫描，未做外部公式渲染和 PDF 构建。
- `sync_status`：`PAPERS.md` 补充 ARES/PPI、AgentDojo 与 HarmBench 方法边界；`GLOSSARY_EN_ZH.md` 新增 FNR/FPR 分母定义；`EXERCISES.md` 新增第六册第十一章 4 项度量/安全/评估练习。`INTERVIEW_BANK.md` 已有 judge 校准、发布门禁、RAGAS/ARES、AgentDojo 对应题目，本轮不重复添加。
- `open_risks`：没有目标 trace、任务/人工标签、目标 judge 校准、部署 SLO、真实工具 Agent 与安全演练。benchmark 和论文数字不等于生产质量/安全；零事件区间依赖 i.i.d. 假设，不能应对适应性攻击与覆盖缺口。GenAI semantic conventions 和 vLLM metrics 动态页面需随目标版本复核。
- `status`：第六册第十一章完成当前标准下的研究、精修、代码输出/边界核验和配套同步；全书 goal 继续，下一章为第六册第十二章。

### 第六册第十章：多模态部署资料驱动深度精修验收（2026-10-26）

- `scope`：按当前 `WRITING_SPEC.md` 全文复读 `book-06-llm-deployment/chapters/10-多模态部署.md` §1—§15。原有文件/媒体边界、VLM、OCR、ASR、TTS、视频、安全、容量、评估、发票与实时语音案例、标准库 demo、排障和联合发布案例均保留；逐节覆盖与知识点去向见 [`research/multimodal-deployment-depth-review-2026-10-26.md`](research/multimodal-deployment-depth-review-2026-10-26.md)。
- `changes`：新增 Qwen2-VL 动态分辨率和视觉 token cap 手算；比较 LayoutLMv3/OCR pipeline、Donut OCR-free、OCR+VLM 升级路由各自省去的环节和证据代价；用 Riva chunk/padding 参数推导 4 秒窗口与频率/并发权衡；比较级联 ASR→LLM→TTS、原生 Thinker–Talker/ARIA 与专用 TTS；区分 encoder 前抽帧和 encoder 后 EVS/VidCom2 pruning，并说明 vLLM 兼容/graph 限制；补媒体 URL SSRF 与 image-based prompt injection 研究；补控制面 orchestrator、stage pool、connector 和 async-chunk handoff；评估加入 Video-MME-v2 的跨题一致性。修复 demo 图像 cap 在极端长宽比下超限、`NaN/Inf` 通过输入检查、context overflow 仍走同步路由和示例漏输出 diagnostics 的问题。
- `evidence`：研究笔记对照 Qwen2-VL、Donut、LayoutLMv3、NVIDIA Riva 配置/overview/performance、Qwen3-Omni、Qwen3.5-Omni、EVS、VidCom2、vLLM 当前多模态文档、Video-MME-v2、Image-based Prompt Injection 和 vLLM-Omni 报告。EVS 的“最多 4× TTFT”、VidCom2 的 `25% token/99.6% LLaVA-OneVision 原表现/70.8% generation latency reduction`、图像注入最高 64%、vLLM-Omni 在 Qwen3-Omni Fixed-Len-2500/900、C32、2×H100 下 async-chunk/no-async TTFT `507.8/3420.1ms`、E2EL `72.63/136.62s`、audio RTF `0.630/0.927` 均绑定各自论文和负载；TTFT 指 Thinker 首文本 token，不是音频首包。Riva 窗口参数绑定其 Conformer 配置；详见研究笔记。
- `formula_status`：复核 `57×75=4275` raw image patches；教学 resize 得到 `33×44=1452≤1536`。Riva `0.16+2×1.92=4.00s` 和 `0.8+2×1.6=4.00s` 两种窗口，更新间隔相差 5 倍；25×/5×输入覆盖仅为每窗重算假设，不当作 GPU FLOPs。45 秒、0.5 FPS 得 23 帧，12 帧 cap 后 3072 token，额外 75% 均匀剪枝估算为 768，明确不是 pruning 算法真实输出。demo context `620+2372+128=3120/4096`，剩 976；KV 理论账本 390 MiB；ASR RTF `4.2/12=0.35`。
- `code_status`：从 Markdown 提取 §11 标准库 Python block，编译/正常运行通过，实际 stdout 与文中输出逐字一致，包括 `diagnostics`。边界检查通过：极端长宽比及 cap=1、非正/非整数/布尔/NaN 图像尺寸、超过 demo safety limit、KV 非有限/负输入、路由 NaN/Inf/负数/布尔输入；将 context limit 改为 1000 的变体进入 `trim_or_reject`。只验证 toy 账本和输入合同，不代表 processor、模型、视频解码、GPU 或线上 serving。
- `link_status`：新增论文、官方 serving 文档的关键机制和数字已逐项打开核对；章内 bibliography 扩至 27 项。第六册目录已有第十章导航，无需更改。既有外链未做全量 HTTP 状态扫描。
- `sync_status`：`PAPERS.md` 加多模态预算/剪枝及 serving/安全/评估研究；`GLOSSARY_EN_ZH.md` 加动态分辨率、M-RoPE、EVS/VidCom2、Video-MME-v2、async chunk、RTF/RTFx；`INTERVIEW_BANK.md` 加视频剪枝、ASR chunk/padding、多阶段 runtime 三题；`EXERCISES.md` 加五项部署机制练习。
- `validation`：确认一级章节 §1—§15 连续、原章节所有主体知识保留；检查新增方法比较均写明问题、机制、证据、代价和适用边界；demo 输出一致，`git diff --check` 通过。未运行真实图像/音频/视频模型、目标 GPU、服务压测、外部公式渲染或 PDF 构建。
- `open_risks`：没有目标媒体语料、目标模型/processor、ASR/TTS/video runtime、GPU、请求 trace 或安全 red-team 环境；不能推断真实质量、实时性、成本、吞吐或攻击成功率。EVS/VidCom2/Donut/Riva/Qwen/vLLM-Omni/Video-MME-v2 结果绑定模型、数据、版本、硬件、commit 和指标。当前 vLLM/Riva/Hugging Face `latest` 文档需按目标 release 复核。
- `status`：第六册第十一章完成资料驱动精修、原知识点覆盖、指标分母/统计边界审计、示例运行和配套同步；全书 goal 继续，下一章为第六册第十二章。

### 第六册第九章：RAG 与 Agent 部署资料驱动深度精修验收（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 全文复读 `book-06-llm-deployment/chapters/09-rag与agent部署.md` §1—§17。保留原有生产 RAG 链路、解析与切分、embedding、向量库、混合检索与 reranker、上下文组装、RAG 评估、Agent 循环、tool calling、状态管理、权限/审计、可观测性、风险、Python demo、企业发布案例、来源和小结；逐节覆盖见 [`research/rag-agent-deployment-depth-review-2026-10-26.md`](research/rag-agent-deployment-depth-review-2026-10-26.md)。
- `changes`：新增固定 top-k、Self-RAG、CRAG、GraphRAG、Agentic RAG 的故障目标/机制/作者证据/代价限制对比；补 RAGAS 与 ARES 的自动评估机制、人工校准方法及论文内排序结果范围；补检索投毒论文与 AgentDojo 的正常效用/攻击安全双重评估；比较逐次直接工具调用、确定性 workflow 与 OpenAI Programmatic Tool Calling 的控制流选择。加入 Recall@5 与 context recall 的手算例。修正 demo 将 citation ID 存在误称为证据支持、把正则词项误称为 tokenizer token 的问题，改为准确解释引用 ID 检查、word units、工具校验覆盖面和预设延迟边界。
- `evidence`：核对 RAG 原论文、Self-RAG、CRAG、GraphRAG、RAGAS、ARES、ReAct、AgentDojo、USENIX Security 2026《Overcoming the Retrieval Barrier》及 OpenAI Programmatic Tool Calling 官方指南。论文及仓库证据、解决问题、方案代价和范围记录在研究笔记。ARES 相对 RAGAS 的 Kendall's tau 差值仅绑定原 benchmark/标注/judge 设置；USENIX 的接近 100% retrieval 和特定 poisoned-email 多 Agent 外泄结果只绑定论文威胁模型；GraphRAG 结果只绑定其全局 sensemaking 任务和语料规模。
- `formula_status`：复算 gold `{A,B,C}`、top-5 `{A,X,B,Y,Z}` 得 `Recall@5=2/3`、`Hit@5=1`；若上下文只保留 A，则 context recall=`1/3`。demo 使用 regex `word_units`，输出 26，不等同模型 tokenizer token 数。未使用外部公式渲染器。
- `code_status`：运行 §14 标准库 Python demo，退出码 0、stderr 为空、stdout 与 fenced `text` 示例逐字一致；核验租户过滤、已知/未知/缺失引用 ID、未知工具和缺少 required 参数的边界。代码只做 toy 字符串、权限字段和预设延迟检查，不代表真实 embedding/ANN、claim grounding、JSON Schema、Agent runtime 或工具事务。
- `link_status`：本章无相对 Markdown 文件链接；重点核对新增代表论文、USENIX 页面、Microsoft GraphRAG 仓库和 OpenAI 官方工具编排文档。章节现有全部外链未做自动 HTTP 状态扫描；第六册目录已有第九章条目，无需改导航。
- `sync_status`：`PAPERS.md` 增 Self-RAG、CRAG、GraphRAG、RAGAS/ARES、检索投毒、PTC 入口；`GLOSSARY_EN_ZH.md` 增相关方法、安全评测术语；`INTERVIEW_BANK.md` 加入代表方法比较、自动 judge 校准和检索投毒问题；`EXERCISES.md` 加入方法选择、评估校准和红队设计练习。
- `validation`：确认 §1—§17 连续、40 个 fenced-code/math/text 标记闭合、14 张表格列数一致；demo 输出逐字匹配，`git diff --check` 通过。未做外部数学渲染、生产 RAG/Agent 实验、目标流量压测、真实 red team 或 PDF 构建。
- `open_risks`：没有目标企业语料、query trace、模型、retrieval service、Agent runtime 或业务工具，未验证真实召回、claim-to-span 支持、注入成功率、权限安全、延迟或成本。论文结果、当前 GraphRAG 仓库状态和厂商 API 能力分别受任务、版本和配置限制，投入生产前需按目标环境复测。
- `status`：第六册第九章完成资料驱动精修、原知识点覆盖、代表方案比较、配套同步及声明范围内代码/链接/格式验收；全书 goal 继续，下一章为第六册第十章。

### 第五册第一章：训练总览与 Scaling Laws 深度精修验收（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 全文复读 `book-05-llm-training/chapters/01-训练总览.md` 的 1.0—1.20。原有训练证据链、目标定义、历史、全链路阶段、自回归目标、loss/mask、token/batch、6ND、Dense/MoE、资源估算、Tokenizer/数据/架构联动、优化器、模型阶段、SFT/DPO/RL、监控、checkpoint、失败排查、交付账本、Python 代码和七项练习都保留；旧知识点去向见 `research/training-overview-scaling-laws-depth-review-2026-10-26.md` 的覆盖图。
- `changes`：重写并扩展 §1.7.2，先定义 scaling law 的经验范围、拟合 loss 和“最优”的目标，再引出 `C≈6ND` 约束。加入 Chinchilla `L(N,D)` 参数形式及变量解释；比较 Kaplan 2020 的 `C^0.73/C^0.27`、Chinchilla 三种估计方式和同预算模型实验、数据受限重复训练、计入 serving 的 inference-aware scaling，以及 Llama 3 报告中的旗舰/小模型预算取舍。加入 `40B×2T` 与 `80B×1T` 的等一阶 FLOPs 手算，分开 `seen/unique tokens`，补 scaling study 的变量、拟合和外推校验步骤。新增第八项固定预算模型选择练习，并扩充来源清单和总结。
- `evidence`：研究笔记记录 Kaplan、Chinchilla、Muennighoff、Sardana、Llama 3、`(Mis)Fitting` 综述的具体章节、拟合方法、数据范围、实际结果和限制；交叉核对 Google DeepMind 发布博客与 Chinchilla 论文，记录博客写 1.3T、论文写 1.4T，正文使用论文表格数值。查看 `(Mis)Fitting` 作者 GitHub 仓库 README 的训练代码、拟合分析、数据和 checkpoint 状态；仓库仅作为代码入口，未作本地复现声明。
- `formula_status`：核对 Chinchilla 参数化 loss 的 E/A/B/α/β 含义和固定 FLOPs 约束；手算 `6×40×10^9×2×10^12 = 4.8×10^23`，`6×80×10^9×1×10^12 = 4.8×10^23`。确认 toy 例子只支持一阶 `6ND` 数量关系，不推出质量、墙钟时间或货币成本相同。现有其它公式没有改写；未使用外部公式渲染器。
- `code_status`：§1.17 Python 训练规划账本未修改；本轮未运行该代码、训练脚本、成本仿真或其他测试，故不把本轮记为代码运行验证。
- `link_status`：新增原论文、官方 Meta 报告、DeepMind 博客和作者复现实验仓库入口；章节相对 Markdown 链接和本册章节目录映射有效。
- `sync_status`：`PAPERS.md` 增加数据受限、推理成本、`(Mis)Fitting` 与 Llama 3 文献；`GLOSSARY_EN_ZH.md` 补 Data-Constrained Scaling、Inference-Optimal Scaling；`INTERVIEW_BANK.md` 改写 31L，加入代表研究目标、证据及限制。本章新增第八项练习，`EXERCISES.md` 不重复复制。
- `validation`：检查标题连续、fenced code/math 围栏配对、公式反斜线与定义、内部链接目标；本轮新增外链均可访问且其支持范围已记录；`git diff --check` 通过。未执行章节 Python、训练实验、真实 Serving、成本测算或 PDF 构建。
- `open_risks`：缩放指数和同预算性能属于论文设置下的作者结果；论文从小模型外推到大型模型、数据重复、请求成本等结论有各自覆盖范围。没有在目标数据、现代架构、目标集群和实际服务账单上拟合或复现。本轮没有执行既有 Python 账本；后续若按写作规范要求逐章执行代码验证，需补该项记录。
- `status`：第五册第一份正文完成本轮资料驱动的 scaling law 精修与静态验收；全书 goal 继续，下一份为第五册第二份正文。

### 第五册第二章：数据工程深度精修验收（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 复读 `book-05-llm-training/chapters/02-数据工程.md` 的 2.0—2.23。保留来源契约与 lineage、解析、质量信号、PII / secret、安全、exact / near dedup、benchmark contamination、mixture、合成数据、SFT / preference / process data、代码 / 数学 / 领域、多语言 / 多模态、packing / shard、删除、评估、toy 审计器、故障诊断和练习；旧知识点去向见 `research/training-data-engineering-depth-review-2026-10-26.md` 的 coverage map。
- `changes`：质量筛选补 classifier score 的参考集定义和 C4、Dolma、FineWeb、DCLM、Nemotron-CC 构建路线对照；PII 用 Dolma 阈值做单一语料案例并限定结果范围。near-dedup 加 MinHash 估计式、FineWeb 14×8 LSH 参数下的召回概率演算，以及 BFF 与 MinHash + Suffix Array 在重复定义、短 n-gram 误删和公开消融上的比较。污染部分加入 DCLM benchmark 识别及移除前后分数，说明规则误报不能推出“无污染”。mixture 部分演算 `100:10:1` 温度采样的曝光和重复倍数，比较 DoReMi 的 excess-loss Group DRO 与 RegMix 的 proxy 回归，并用 DCLM mixing 结果解释数据互补依赖基础语料。合成数据部分区分递归替换原始数据与保留并累积真实数据的前提。
- `evidence`：来源、原文机制、表格结果、版本和适用限制记录在上述 research note。重点核对数据选择综述、C4 / T5、Dolma、FineWeb、DCLM、Nemotron-CC、DoReMi、RegMix、dedup、污染综述和 model-collapse 论文；查阅 Hugging Face DataTrove、DCLM 与 Dolma 工具仓库作为工程入口。所有论文数字绑定其模型、token 预算、benchmark 和评测版本；DCLM README 的聚合修正与论文旧版分数明确分账。
- `formula_status`：核对 MinHash 估计式及 `1-(1-s^8)^{14}` 的四个数值；核对 `100:10:1` 在 `alpha=0.5` 时归一化为约 `70.6:22.3:7.1`，并由 200-token-unit 预算推得曝光与重复倍数；增加 DoReMi excess-loss 的 Group DRO 目标式并解释内外层含义。未使用外部公式渲染器。
- `code_status`：原有 toy Python 审计器未修改、未运行；未下载语料、执行去重或训练 proxy / LM。代码检查不等于数据管线复现。
- `link_status`：章节阅读清单加入综述、原始论文、预印本及代码库入口；本册目录已有第二章映射。论文与仓库结果按其版本隔离记录。
- `sync_status`：`PAPERS.md` 加入 data selection、Dolma、DCLM、DoReMi、RegMix、Nemotron-CC 和 2026 预印本入口；`GLOSSARY_EN_ZH.md` 加入 classifier quality filtering、BFF、mixture / diversity 与新采样术语；`INTERVIEW_BANK.md` 增加语料构建、过滤覆盖和 mixture 方案比较。章节已有练习保留，未重复复制到 `EXERCISES.md`。
- `validation`：复核章节小节连续、公式变量与手算、相对链接和阅读入口；`git diff --check` 通过。未运行 Python 示例、论文实验、外部公式渲染或 PDF 构建。
- `open_risks`：不同研究的 tokenizer、模型规模、数据分布和训练协议不同，不据此横向排名；过滤器选择的是目标数据的子分布，去污染规则有误报，自动 mixture 依赖 proxy 和任务指标，合成数据结论依赖真实数据是否保留。目标数据、训练栈、硬件和真实评测上的收益尚未复现。
- `status`：第五册第二份正文完成本轮资料驱动精修与静态验收；全书 goal 继续，下一份为第五册第三份正文。

### 第五册第三章：Tokenizer 与数据格式深度精修（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 从头复读 `book-05-llm-training/chapters/03-tokenizer与数据格式.md` 的 3.0—3.21。保留并逐项映射原有 BPE、SentencePiece/Unigram、byte handling、词表、特殊符号、chat template、EOS/PAD、assistant loss、packing、SFT/preference/RL/tool 格式、多模态 token、扩词表、toy auditor、协议故障、评估和八项练习；覆盖图见 `research/tokenizer-data-format-depth-review-2026-10-26.md`。
- `changes`：修正 BPE 手算的第二轮 pair 计数并显式声明 tie-break；增加 normalizer 可逆性和语言分类反例；分别解释 byte-level BPE 与 byte fallback；量化词表从 32K 到 128K 的 embedding 与 `L×V` 输出账本；对照压缩、算法目标/搜索、多语 fairness 和 tokenizer intrinsic metrics 研究；补 MUTANT 多语 recipe 的训练与 H100/Triton 测量条件并指出 tokens/s 的分母变化。数据协议部分深化 Harmony 特定控制事件、PAD=EOS 长度 mask、assistant label shift/reduction，以及独立 SFT packing 的 attention 和 boundary label。
- `evidence`：研究笔记记录 BPE、SentencePiece、Subword Regularization、Unpacking Tokenization、Tokenization Is More Than Compression、Petrov 等人公平性、Tokenizer Choice、Objective vs. Search、No Single Feature、TokLens、MUTANT、FOCUS、Hindi vocabulary initialization preprint 的机制、实验数字、限制及其对正文的支持范围；核对 tiktoken / GPT-2 实现、SentencePiece options、Hugging Face Transformers/TRL、PyTorch、NeMo 25.02 和 OpenAI gpt-oss Harmony 文档。不同模型、语料、预算和指标没有横向裸分排名。
- `formula_status`：手算并复核 BPE pair counts、`32,000×4,096=131.072M`、`128,000×4,096=524.288M`、`32M→76.8M=2.4×` 输出位置×词表代理量、toy token/char/token/byte 统计、`1,000×1,000` patch 与 merge/frame 账本，以及 10-token/100-token 两种 loss reduction (`1.91` vs `1.5`)。BPB 明确要求相同规范化文本 bytes 和 BOS/EOS 边界。公式仅静态核对，未用外部 renderer。
- `code_status`：修改并静态审阅 §3.16 toy auditor 的固定 byte fallback、roundtrip、角色输入类型、assistant-only labels、按原始长度生成 padding mask、packing/EOS 与新 token 注册；没有运行代码或测试，不声明执行结果已验证。
- `link_status`：本章目录入口 `book-05-llm-training/目录.md` 已映射；新增论文和官方文档链接逐项研究并打开，链接版本边界在研究笔记标注。无新增跨章正文迁移。
- `sync_status`：`PAPERS.md` 增加 2024—2026 tokenizer 选择/搜索过程/多语评估研究；`GLOSSARY_EN_ZH.md` 增加 STRR、tokenizer intrinsic metric 与 search procedure；`INTERVIEW_BANK.md` 深化压缩率研究对照，并新增 tokens/s 与 tokenizer 变化的口径题。本章原有八项练习保留并补 fixed-FLOPs/BPB 设计，未复制到 `EXERCISES.md`。
- `validation`：核对 3.0—3.21 标题连续、原有主题保留、代码与数学 fence 数量成对、BPE/参数/压缩率/label 算例静态一致、章节目录入口和新引用目标；`git diff --check` 通过。没有运行 toy Python、训练实验、性能基准、外部公式渲染或 PDF 构建。
- `open_risks`：toy Python 尚未实际运行，语法/行为结论因此仅限人工静态审读；完整 tokenizer 统计、模型重训、多语 benchmark、MUTANT/Harmony/NeMo 目标栈复测均未在本地进行。Hugging Face `main` 与 GitHub `master` 文档会变，应绑定目标 revision；作者研究数字保留各自语料、模型、硬件、benchmark 与版本边界。
- `status`：第五册第三份正文的资料驱动内容精修和静态检查完成，代码运行门禁保持开放；全书 goal 继续，按用户要求不回头重做之前章节，下一份为第五册第四份正文。

### 第五册第四章：模型架构与初始化深度精修（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 从 4.0—4.18 全文复读 `book-05-llm-training/chapters/04-模型架构与初始化.md`。保留原有 decoder-only/encoder-decoder、参数与 `6PD`、RoPE、Norm、SwiGLU、MHA/MQA/GQA、长上下文、MoE、初始化、训练稳定性、推理成本、账本代码、故障排查、六项练习和小结；知识点迁移/缩写映射见 `research/training-architecture-initialization-depth-review-2026-10-26.md`。
- `changes`：增加 GQA attention projection 与 KV cache 的参数手算，区分二者比例；补 RoPE 频率日程和频率数值例，比较 PI、YaRN、LongRoPE 的目标、频率策略、论文预算与边界。用 LayerNorm/RMSNorm 与 Pre/Post-LN 公式解释归一化和梯度路径，比较 DeepNorm 深度稳定与 μP 宽度超参迁移；为 SwiGLU 添加等参数中间维度演算及其 seq2seq 证据边界。GQA 补 mean-pool 转换/5% uptraining 结果和 Llama 2 attention ablation、TP 拓扑取舍。MoE 部分明确 routed/shared/router 账目，区分负载均衡项与 router z-loss，比较 Sparsely-Gated MoE、GShard、Switch、ST-MoE、Expert Choice、DeepSeek-V3 的解决目标与限制。架构 toy ledger 改名为 dense reference，MoE FFN 改成独立单层子账本；RMSNorm 数值实现改为缩放计算，RoPE helper 限定成固定频率二维对。
- `evidence`：研究笔记核读 RoFormer、PI、YaRN、LongRoPE、RMSNorm、Pre-LN/Post-LN、DeepNorm、μP、SwiGLU、GQA、Llama 2、Sparsely-Gated MoE、GShard、Switch、ST-MoE、Expert Choice 与 DeepSeek-V3 一手论文/技术报告。GQA 研究笔记记录 T5-XXL 5% uptraining、GQA-8 47.1/0.28s、MHA 47.2/1.51s、MQA 46.6/0.24s，并限定为论文多任务平均分和 TPUv4 setup；MoE、长度扩展与初始化结果均绑定各自论文模型/任务/训练设置。逐项来源及正文 claim map 见上述研究笔记。
- `formula_status`：手算 `d=4096,H_q=32,d_h=128,H_kv=8` 时 MHA attention projection 67,108,864、GQA 41,943,040，分别占比 100%/62.5%，而 KV cache 为 100%/25%；手算普通 `d_ff=4d` FFN 134,217,728 与 SwiGLU `d_ff=10,923` 的 134,221,824。检查 `d_rot=8,base=10,000` 频率为 `1,0.1,0.01,0.001`，RMSNorm 缩放算法与直接定义等价，MoE active/total 公式及 router z-loss 口径。以上为手算/静态检查；未运行 Python 或数学渲染器。
- `code_status`：修订了 §4.14 dense reference 的总参数与 `6PD` 命名，MoE 单层 expert FFN 子账本不再混进 dense 总数；注释 tied embedding、两组 norm scale、输出头/final norm/router/shared expert 未计入。RMSNorm helper 使用缩放实现，RoPE helper 改名并明确非完整 RoPE。代码未运行；不能标为 syntax/runtime 验证完成。
- `link_status`：本章目录映射存在；正文和阅读清单加入 PI、YaRN、LongRoPE、DeepNorm、μP、GQA、Llama 2、GShard、ST-MoE、Expert Choice 和 DeepSeek-V3 来源入口；来源边界记录在 research note。尚未做所有外链的 HTTP 状态扫描。
- `sync_status`：`PAPERS.md` 增加长 RoPE、DeepNorm、μTransfer、Expert Choice 与 DeepSeek-V3 论文入口；`GLOSSARY_EN_ZH.md` 深化 MQA/GQA、MoE、Expert Choice、router z-loss、RMSNorm、DeepNorm、μP、YaRN、LongRoPE 定义；`INTERVIEW_BANK.md` 深化 31D—31G 并增加 MoE 负载/稳定性和 active-parameter 追问题。章节已有六项练习保留，未重复复制至 `EXERCISES.md`。
- `validation`：静态检查 4.0—4.18 标题和主节连续；38 组 `~~~` fenced block 开闭配对（36 math、1 text、1 Python）；本册目录路径与章节文件一致；手算和关键公式口径已复核；`git diff --check` 通过。未运行 §4.14 Python、训练/Serving 实验或 PDF 构建。
- `open_risks`：PI/YaRN/LongRoPE、Norm/DeepNorm/μP、GQA 与 MoE 各来源实验设置不同，不得用窗口、单个平均分、加速倍率或 active parameters 横向排名；目标模型/硬件/任务仍需复测。第三章和本章 toy Python 均未运行；全书其他章节的 GPU、训练稳定性和 Serving 实测边界仍按各自记录。
- `status`：第五册第四份正文完成资料驱动精修与静态审读；代码运行验收保持开放。按用户要求从第五册第四章之后继续，下一份为第五册第五章，不回头重复精修前序章节。

### 第五册第五章：预训练目标与训练循环深度精修（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 全文复读 `book-05-llm-training/chapters/05-预训练目标与训练循环.md` 的 5.0—5.17。保留原有自回归概率、teacher forcing、causal mask、label shift、cross entropy/PPL、三类 mask、packing、EOS 边界、optimizer/clipping、effective batch、梯度累积、混合精度、validation、日志、checkpoint/resume、checkpoint 选择、toy 训练循环、故障诊断和练习；旧主题去向见 `research/training-objectives-and-loop-depth-review-2026-10-26.md` 的 coverage map。
- `changes`：新增预训练目标方法对照，分别解释 Causal LM、BERT-style MLM、T5 span corruption、UL2 R/S/X Mixture-of-Denoisers 和 FIM 的可见上下文、监督 token、适用场景与限制；加入 T5 sentinel 例子、UL2 论文跨任务聚合结果与规模/数据边界、FIM PSM 训练轨迹和 FIM rate 消融。明确 FIM 是 token 重排后的 causal loss，补 marker token、context-level packing 和 PPL/pass-rate 差异。变长 micro-batch 增加 `sum(n_a g_a)/sum(n_a)` 梯度公式、2+4 token 手算和 DDP `P/N` 简化情况；validation 目标数按模型内部/外部 shift 约定统计。toy 代码把 accumulation 拆为 2+4 有效 token，比较全局 token 加权与错误的等 batch 加权，并保留原数据/边界/SGD 教学目标。
- `evidence`：研究笔记记录实际核读 T5 原论文 §3.3/表 3、4、7；UL2 §3.1/§4.2—4.5 与表 1、3、4；FIM §3、§4.2—4.5 的 FIM rate、context-level 变换和 infilling 评价；Hugging Face Accelerate v1.15.0 变长 accumulation 指南、Transformers label-shift/loss scaling 文档、Accelerate autoregressive accumulation GitHub 示例以及 PyTorch DDP reducer 说明。UL2 +43.6%/+76.1% 明确为作者在约 167M/335M 模型、C4、约 32B tokens 和异质任务上的归一化总分；FIM 结果限定为论文语料/模型/采样评测。来源及证据边界见该 research note。
- `formula_status`：确认 token mean 梯度 `g_window=Σ n_a g_a / Σ n_a`；2+4 token 权重是 `2/6` 与 `4/6`，非各 `1/2`。简单 DDP 公式以 rank mean-gradient、全局 token count `N` 和无 framework extra scaling 为前提；rank 本地 loss sum 乘 `P/N` 后由 DDP 平均得到 global token mean。手算 toy 输入产生 6 个有效 pair，切成 2+4；按独立前序 token 行可静态验证错误等权梯度不会等于 full-batch gradient。未运行脚本或外部公式渲染。
- `code_status`：toy Python 通过逐行静态检查；修改为按 micro-batch valid-token 数加权，并新增 `equal_batch_weighting_differs` 差异诊断。代码、assert 和输出未执行，不能记为 runtime/syntax 通过；代码验证项仍开放。
- `link_status`：本册目录中的第五章入口存在；新增 T5/UL2/FIM 原论文、Google Research T5 / SeqIO、官方 Hugging Face token accumulation 文档及 GitHub example、PyTorch DDP 文档入口。FIM HumanEval infilling 仓库注明已归档；GitHub `main` 代码与无版本 docs 需依目标 revision 复核。
- `sync_status`：`PAPERS.md` 增补 T5、UL2、FIM 论文及实现入口；`GLOSSARY_EN_ZH.md` 增加 MLM、span corruption、MoD、FIM/FIM rate、token-weighted accumulation 和 valid prediction target count；`INTERVIEW_BANK.md` 增加目标函数对照题与变长 accumulation/DDP denominator 题。章节新增第八项选型练习，未重复复制到 `EXERCISES.md`。
- `validation`：复核 5.0—5.17 标题连续、公式和 code fence 成对、内部 label shift/validation denominator 说明、2+4 token 梯度权重、章节目录映射和新增链接目标；`git diff --check` 通过。未执行 Python、训练代码、分布式实验、外部公式渲染或 PDF 构建。
- `open_risks`：T5/UL2/FIM 证据不能跨论文设置直接排名；DDP/Trainer/Accelerate/DeepSpeed/FSDP 的缩放仍需按实际版本与 loss API 核验。本章 toy 只展现有限 token-to-logits 局部梯度，不验证真实 Transformer、AMP、分布式通信或 checkpoint 恢复。代码运行验收未完成。
- `status`：第五册第五份正文完成资料驱动的内容精修和静态审读；代码运行验收保持开放。按用户要求不回头重做前序章节，下一份为第五册第六章。

### 第五册第六章：优化器与学习率策略深度精修验收（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 复读 `book-05-llm-training/chapters/06-优化器与学习率策略.md` 的 6.0—6.24。旧章中 SGD/Momentum、Adam/AdamW、状态显存与精度、Adafactor、Lion、Sophia、Muon、GaLore、参数分组、warmup/cosine/linear、WSD、weight decay、clipping、batch scaling、checkpoint、loss spike 排查、toy 示例和练习均保留；旧主题去向及本轮判断见 `research/training-optimizers-and-schedules-depth-review-2026-10-26.md`。
- `changes`：Adam 与 AdamW 加首步数值反例，拆开耦合 L2 与 decoupled shrinkage。Sophia 更正为 `max(gamma*h, epsilon)` 分母和 `[-1,1]` 逐坐标 clip，比较 Hutchinson Hessian 对角与 GNB Gauss-Newton estimator，补曲率刷新/EMA、负估计行为、调参方法和原论文 GPT 实验范围。Muon 补 momentum polar 方向、shape-dependent update RMS、参数组与 Newton–Schulz；并对照 Shampoo、SOAP。更正 Moonlight MoE 参数描述为约 2.24B activated / 15.29B total、5.7T tokens；optimizer 加速倍率按各自规模、数据、调参、评价口径说明，并将比较论文更新为 ICLR 2026 正式发表信息。WSD 加主干/衰减分支关系、MiniCPM 与 River Valley/WSD-S 的具体实验和边界。有效 batch 改按 loss 中真实 prediction targets 计数，补 internal/external shift、padding/mask、变长 rank 和手算例子。增加按瓶颈选型表与公平对照要求；toy 代码改名并明确不实现 Sophia-Hessian 或 Muon polar 算法。
- `evidence`：研究笔记记录实际阅读的原论文方法、实验章节和来源局限：Sophia 算法 3、Hutchinson/GNB 与 GPT 实验；Adafactor factorization/recipe、Lion discovery/tuning/limitations、GaLore projection/state、Muon method/scaling/distributed implementation；Shampoo、SOAP 方法；WSD 的 MiniCPM 36M 实验和 River Valley/WSD-S；Deconstructing Optimizers、ICLR 2026 Fantastic Pretraining Optimizers、NeurIPS 2025 hyperparameter transfer、NVIDIA 2026 大 batch SOAP/Muon 报告。Moonshot 52% FLOPs、不同矩阵 optimizer 的 1.1×/1.4×、Sophia 约 2×、WSD-S cooldown 观察均保留为各论文作者报告，不作跨设置排名。具体 section 和证据边界见研究笔记。
- `formula_status`：手算 AdamW 的 7.82 与 coupled-L2 Adam 的约 7.90 首步例子；静态复核 Sophia estimator refresh、正分母下界与逐坐标 clip，负/零估计不被直接除；核对 Muon polar/shape-scale 表述非统一实现公式；按 causal shift 计算 `L-1` prediction targets，并将变长 batch 分母改为逐 rank/accumulation 汇总。未用外部公式渲染器。
- `code_status`：§6.19 仅做静态逐行审读并调整变量名和输出标签；没有运行 Python、optimizer benchmark 或训练代码。代码运行验收仍开放，toy 示例不能视为 Sophia/Muon 实现或论文复现。
- `link_status`：本章新增 Sophia、Moonshot Muon、ICLR 2026 optimizer 比较、NeurIPS 2025 transfer、Shampoo/SOAP、WSD/WSD-S/NVIDIA 原始论文入口；本册目录已有第六章路径映射。Hugging Face `main` 文档为动态链接，实施时需按项目版本核对；PyTorch stable 文档本轮没有作为新算法结论的唯一依据。
- `sync_status`：`PAPERS.md` 增加 Shampoo/SOAP 和 ICLR 2026/NeurIPS 2025/2026 系统比较入口；`GLOSSARY_EN_ZH.md` 已补齐 Adafactor、Lion、Sophia、Muon、GaLore、Shampoo、SOAP、WSD、WSD-S 等术语；`INTERVIEW_BANK.md` 在“训练”新增六题，覆盖 Sophia estimator、矩阵优化器分工、跨论文加速不可比、WSD checkpoint、状态内存方法和公平比较。章节原有练习保留并新增 Sophia/WSD/Muon 练习，未重复复制到 `EXERCISES.md`。
- `validation`：静态检查 6.0—6.24 标题连续、代码/math/text fenced blocks 配对、章节目录与相关研究/索引路径存在；复读 Sophia、Muon、WSD 与 target-batch 核心公式以及新增论文链接；`git diff --check` 通过。未执行 Python、训练实验、外部公式渲染或 PDF 构建。
- `open_risks`：未在目标训练栈、模型、数据和硬件上复现任何 optimizer 或 WSD 论文；矩阵 optimizer 的通信、kernel、低精度与层级 update RMS 需要特定实现下 profiling。优化器状态节省不一定降低总体峰值显存。Toy 代码没有 runtime 通过记录；外部框架文档/API 仍须按锁定版本核对。
- `status`：第五册第六份正文完成本轮资料驱动精修与静态审读；代码运行验收保持开放。按用户要求不回头重做前序章节，下一份为第五册第七章。

### 第五册第七章：分布式训练深度精修（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 从头复读 `book-05-llm-training/chapters/07-分布式训练.md` 的 7.0—7.21。保留 rank/process group、DDP、collectives、ZeRO/FSDP、TP/PP/SP/CP/EP、activation checkpointing、overlap、checkpoint/recovery、故障模型、指标、选型、toy 账本、练习和小结；旧主题去向及解释级别见 `research/training-distributed-parallelism-depth-review-2026-10-26.md` 的 `coverage_map`。
- `changes`：将 batch 分母明确为对齐后的有效 prediction targets，并与 input tokens 区分；补 3-rank `loss sum/target count` 数值演算和 PyTorch DDP 默认 rank averaging 下 `P/N` 缩放条件；补 DDP uneven-input `join()` 的 initial/effective world-size 语义。ZeRO 加 7B/DP=8/AdamW 的状态 GiB 账本；区分 FSDP1 wrapper/FlatParameter 与 FSDP2 `fully_shard`/DTensor。流水线比较 GPipe flush、non-interleaved 1F1B、interleaved 1F1B，解释 activation、bubble 和每样本通信的共同变化；SP/CP 增加张量通信轨迹并区分 Ring Attention 算法；MoE 容量加入 4096 tokens/top-2/16 experts 手算及 Switch top-1/GShard top-2 对照。重算部分区分 full 与 selective recomputation，并限定论文节省数值。checkpoint 增 DCP 2.14 load-time reshard/async staging 与应用层 manifest/原子提交边界；可观测性补 PaLM 的 MFU/HFU 口径。
- `evidence`：研究笔记记录 ZeRO、Megatron-LM、GPipe、GShard、Switch Transformer、Sequence Parallel/Activation Recomputation、Ring Attention、PaLM 原论文对应章节/表格与限制；核对 PyTorch DDP/FSDP2/DCP 2.14、NCCL 文档和 Megatron Core CP 当前指南。论文数字绑定作者模型、硬件、序列长度、parallel degrees、FLOPs 和评估配置，不作跨论文裸排名；在线 Megatron Core 指南标为动态文档。
- `formula_status`：手算 `380/700≈0.5429` 与错误 rank-mean `≈0.5333`；7B BF16+AdamW 状态估算 DDP/ZeRO1/2/3 分别约 78.2/32.6/21.2/9.8 GiB；MoE 每专家 capacity 640、偏斜示例 overflow 约 1818；ring 4 GiB/DP8 每 rank 发送 7 GiB、200 GB/s 理想下界约 0.0376 s；toy target batch 262080，input batch 262144。Pipeline `p=8,m=16` 区分 bubble/ideal `43.8%` 与 idle/total `30.4%` 两个分母；代码 PP=2/m=8 返回 idle/total `1/9`。activation toy 约 24→10 GiB、训练 FLOPs 代理 `6N+2N`；论文的 selective recomputation 70%/65% memory 与 2.7%/1.6% FLOPs overhead、SP+selective 约 5×组合节省均只记在对应作者配置下。未用外部公式渲染器。
- `code_status`：§7.17 修正带宽字段为十进制 GB/s，并将监督 target 数与输入位置数分开；激活账本改成注明边界的 24 层分段 toy。Python 只作静态审读，未执行脚本、分布式 collective 或测试，不记 runtime verified。
- `link_status`：本章把 PyTorch DDP/FSDP2/DCP 入口固定到 2.14；新增 ZeRO、GShard、PaLM、Ring Attention 和 Megatron Core Context Parallelism 来源。Megatron Core `latest` 标记动态；章节目录已含第七章入口。
- `sync_status`：`PAPERS.md` 增加 GShard/PaLM 与分布式并行研究导读和固定文档版本；`GLOSSARY_EN_ZH.md` 解释 FSDP1/FSDP2 与 MFU/HFU，并沿用既有 Valid Prediction Target Count；`INTERVIEW_BANK.md` 增加 DDP 变长归一化、FSDP 版本、pipeline schedule、MFU/HFU、DCP resharding 问题；本章练习由八题扩展为十一题，未重复复制到 `EXERCISES.md`。第五册目录原链接正确。
- `validation`：检查 7.0—7.21 标题及子标题连续，代码/math/text fenced blocks 配对；复核 loss scale、ZeRO 字节账本、capacity、ring 带宽、pipeline bubble、activation toy 算术和新链接目标；运行 `git diff --check`。未运行 Python、GPU、多节点、checkpoint restore、公式 renderer 或 PDF 构建。
- `open_risks`：目标训练栈下的 DDP gradient hook/loss scaling、FSDP 峰值、CP/EP 通信、MFU/HFU、NCCL 吞吐和 DCP 跨拓扑恢复均未实测；代码 runtime 仍未验证。论文数字只在作者配置内成立；DCP manifest/commit 是应用设计建议，不是库的原子事务保证。
- `status`：第五册第七份正文完成本轮资料驱动精修与静态检查；代码运行验收保持开放。按用户要求不回头重做前序章节，下一份为第五册第八章。

### 第五册第八章：显存优化与数值精度深度精修验收（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 从头复读 `book-05-llm-training/chapters/08-显存优化与数值精度.md` 的 8.0—8.19。保留参数/梯度/optimizer/activation 账本、瞬时峰值、7B 估算、长上下文、FlashAttention、FP16/BF16/FP8、AMP/loss scaling、activation checkpointing、gradient accumulation、ZeRO/FSDP/offload、8-bit optimizer、LoRA/QLoRA、OOM、NaN/Inf、toy 审计账本和原七项练习；知识点去向和缩减/扩写级别见 `research/training-memory-precision-depth-review-2026-10-26.md` 的 `coverage_map`。
- `changes`：区分逻辑状态量、live tensor 与 allocator reserve；显存峰值改按同时存活对象在时间上取最大，补 PyTorch allocator 可见范围、`empty_cache()` 边界及 Adam 首次 step 懒创建 m/v 的版本化源码路径。用 4K/32K context 手算 hidden/score tensor 量级。展开 FlashAttention 的 online softmax `(m,l,o)` 更新和手算，并界定其 IO/中间存储改进不改变精确 full attention 的平方级 pair 计算。比较 FP16/BF16 表示边界、E4M3/E5M2、delayed/block scaling。修正 FP16 GradScaler 代码顺序和 accumulation 生命周期；把 checkpoint 理想估算改为 `(K+L/K)A`，对照 Chen 与 Korthikanti 方案和限定条件。补变长 token loss 与 DDP rank-average 分母；解释 8-bit Adam blockwise update、LoRA 参数量、QLoRA NF4/Double Quant/paged optimizer 与 input-gradient/activation 的不同预算。扩展 12 项练习。
- `evidence`：研究笔记记录 Mixed Precision Training、BF16、FlashAttention、Chen 子线性存储、Korthikanti selective recompute、8-bit Optimizers、LoRA、QLoRA 的问题、机制、作者实验和适用边界；并核对 PyTorch 2.14 AMP/checkpoint/CUDA memory/Adam v2.14.0 source、NVIDIA Transformer Engine FP8 Primer 2.20.2。相关作者数字均绑定论文模型、数据、硬件、配置和评估范围，不作跨论文裸排名。
- `formula_status`：复核 7B 的 16-byte/parameter 与无 master-copy 12-byte 对照、GiB 换算；BF16 hidden tensor 32 MiB/层与 T=32K 的 256 MiB/层；单层 FP16 score proxy 4K/32K 为 1/64 GiB；online softmax 手算输出约 18.45；FP16/BF16 spacing 与 min-normal；32 层 checkpoint segment toy；2/6 target 的 loss 从错误等权 0.75 改为正确 token-mean 0.825；7B 8-bit Adam 理想 payload 节省约 39.11 GiB；rank-16 LoRA 为 241,664 参数/约 0.536%；QLoRA NF4+Double Quant metadata 账本。未使用外部公式渲染器。
- `code_status`：§8.15 标准库 Python 账本从 Markdown 提取并执行；各状态、activation、attention、checkpoint、LoRA 与 diagnostics 输出和正文一致。另验证 9 类边界拒绝：NaN、bool、bool integer、缺少字段、负 byte count、零 micro-batch、负 optimizer state、checkpoint segments 超层数、LoRA rank 超矩阵维度。此项仅为 local-demo，不是训练系统或 GPU profile。
- `link_status`：正文加入论文、固定版本 PyTorch 2.14 文档、PyTorch Adam v2.14.0 源码和 NVIDIA Transformer Engine 2.20.2 资料；章节相对链接、阅读清单目标与本册目录入口有效。
- `sync_status`：`PAPERS.md` 增加训练内存/重计算/低比特 optimizer/FP8/QLoRA 研究入口；`GLOSSARY_EN_ZH.md` 补 E4M3/E5M2、Dynamic Loss Scaling、subnormal、master weights、amax/delayed scaling、MXFP8、NF4、Double Quantization、blockwise optimizer state 和 paged optimizer；`INTERVIEW_BANK.md` 增加峰值生命周期、online softmax、变长 accumulation、8-bit Adam、QLoRA、selective recompute 与 FP8 scale 策略追问。本册目录原有第八章入口正确，练习保留在正文。
- `validation`：主标题 8.0—8.19 连续；49 个 fenced blocks 成对，含 1 个 Python 块与 33 个 math blocks；公式花括号平衡，12 项练习存在，章节相对 Markdown 链接全部解析。运行标准库账本及 9 项边界拒绝检查；运行 `git diff --check`。未执行外部公式渲染、GPU profiling 或 PDF 构建。
- `open_risks`：没有目标 GPU、模型 checkpoint、数据或集群的实机实验；账本未模拟 allocator fragmentation、真实通信工作区、FSDP/ZeRO 生命周期或真实 checkpoint 保存峰值；FP8/MXFP8/NVFP4、8-bit optimizer、QLoRA 和重计算结果仍依赖目标硬件、kernel、recipe 与质量评测。第三至第七章代码运行验收继续开放。
- `status`：第五册第八份正文完成本轮资料驱动的精修、代码运行和静态验收；全书 goal 继续，下一份为第五册第九份正文。

### 第五册第九章：训练稳定性与 debug 深度精修验收（2026-10-26）

- `scope`：按当前 `WRITING_SPEC.md` 复读 `book-05-llm-training/chapters/09-训练稳定性与debug.md` 全章。保留原有基线、时间轴、loss spike、NaN/Inf、梯度过大/过小、数据与 mask、学习率/optimizer、混合精度、DDP、checkpoint 连续性、toy 诊断、复盘及七项练习；没有因扩展研究删除旧知识点。
- `changes`：新增 Adam time-correlated gradient estimates、adaptive-preconditioner decoupling、SPAM、GradientStabilizer 和 global norm clipping 的机制/证据/边界比较；展开方向曲率的 `2/eta` 一维手算，加入 Pre-LN、DeepNorm、Peri-LN 对照和层间 RMS 诊断。明确 SPAM 回顾性 GSS 与在线 `g_i^2/V_i` proxy 不同；修正 loss spike 与数据异常候选不直接判根因。为 PyTorch 2.14 autograd/AMP/DDP/FSDP/DCP 行为固定版本链接，并区分分布式 API 与应用层 checkpoint 原子发布。修正 `T_attn=0` 的监督比例、零参数范数相对更新率及零梯度方向曲率等边界表达。
- `evidence`：研究记录见 `research/training-stability-debug-depth-review-2026-10-26.md`。读取 Molybog Adam instability、Bai adaptive preconditioner、SPAM、GradientStabilizer、Xiong Pre-LN、DeepNet、Peri-LN 原始来源，并核对 PyTorch 2.14 文档。保存具体训练规模、指标、机制和作者局限；SPAM 主表与附录对 60M token 数量有差异，正文不引用该冲突数据。真实模型/集群未复现。
- `formula_status`：directional curvature 的分母要求有限且非零梯度；`eta=0.001,h=2400` 的一维 quadratic loss ratio 为 `1.96`，标注为阈值直觉而非 LLM spike 预测。更新量和监督比例对零分母采用显式边界，不用 epsilon 掩盖未定义值。18 个 fenced math 块花括号和环境配对检查通过。
- `code_status`：§9.12 唯一标准库 toy 代码从 Markdown 提取运行通过；输出 step 4 loss ratio `2.66`、step 6 nonfinite、grad norm `7.5`、两个候选 batch、rank mismatch、缺失 checkpoint 字段和 rollback checkpoint `None`。另拒绝 bool/非法 step、空历史、负 grad norm、NaN ratio、单 rank loss 等 6 类无效输入。该示例不模拟 autograd、HVP、GPU、DDP/FSDP 或生产 checkpoint。
- `link_status`：PyTorch 2.14 autograd、clip_grad_norm、AMP、DDP、FSDP、DCP 及各原始研究来源已加入正文；相对链接指向本册第四章架构说明并解析成功。
- `sync_status`：`PAPERS.md` 增 Adam spike 和 normalization 研究入口；`GLOSSARY_EN_ZH.md` 增 GSS、Adam 二阶矩失配、方向曲率、GradientStabilizer、Peri-LN；`INTERVIEW_BANK.md` 增 34—36 三题研究比较；`EXERCISES.md` 增 174—176 三项方案对照练习；本章练习由 7 项扩至 10 项。
- `validation`：17 个一级章节号 `9.0—9.16` 连续；62 个 fenced marker 成对，18 个 math block 花括号/环境检查通过；10 项练习、1 个内部链接均验证。§9.12 代码正常和异常输入检查通过；`git diff --check` 通过。未运行 GPU、真实模型、分布式故障注入、HVP 规模评估或 PDF 构建。
- `open_risks`：论文结论受模型规模、数据、训练配置与指标约束，尚无目标系统对照实验；Molybog/Bai 机制不能合并成统一根因；较低 beta2、SPAM reset、clipping 或 gradient magnitude stabilization 仍需目标质量/吞吐验证。Pre-LN/DeepNorm/Peri-LN 是架构和初始化路线，不能由当前 toy 例或中间训练状态验证。
- `status`：第五册第九份正文完成本轮资料驱动精修与验收；不回头重做前序章节，下一份为第五册第十份正文。

### 第五册第十章：Checkpoint、评估与实验管理深度精修验收（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 全文复读 `book-05-llm-training/chapters/10-checkpoint评估与实验管理.md`。保留 checkpoint 状态、保存频率/存储、恢复探针、latest/best、validation/PPL、per-domain、benchmark、regression、实验记录、命名、复现、公平比较、选择流程、demo、误判路径及已有来源；旧主题去向见 `research/training-checkpoint-evaluation-experiment-management-depth-review-2026-10-26.md` 的 coverage map。
- `changes`：§4 补 PyTorch DCP 2.14 的多 rank state dict / load-time resharding、成功 metadata、异步 future、staging CPU/pinned-memory 成本与版本兼容边界。§8 定义完整 evaluation contract，对照 HELM 的场景/指标覆盖与 lm-eval 的统一执行框架；补同题配对检验、McNemar/paired bootstrap、题目波动与训练 seed/prompt 波动区分、污染检测访问条件与负面证据边界。§15 修复 val loss 单位与 benchmark/safety 分数混加、回归惩罚缺少分母、配置缺失键漏检、benchmark 键集合可变、step/token 序列异常和无候选 `max()` 崩溃；加入固定 0–1 归一化、硬门槛、分母检查和无候选 `None`。新增五项章节练习。
- `evidence`：研究笔记记录 HELM、lm-eval reproducibility、Dror 显著性检验指南、Oren 黑盒顺序似然污染检测、ConStat 表现非泛化污染定义，以及 PyTorch DCP/async recipe 2.14、Transformers Trainer v5.19.0。论文证据包括方法设定和适用边界；HELM 16 core scenarios/7 metrics 与其报告的 17.9%→96.0% 覆盖仅绑定论文集合；Oren/ConStat 不被概括为绝对 clean/contaminated 证明。详情见研究笔记。
- `formula_status`：保留既有 loss/PPL、保存成本与恢复偏差定义；新增配对准确率差 `(c-b)/n` 和 evaluation contract 状态元组。手算 `n=1000,b=10,c=30` 得 2pp，但不据此宣称统计稳定。demo 将 `val_loss_cap=4.0` 映射至 0–1，benchmark、安全、regression rate 同尺度；固定权重 `0.40/0.30/0.15/0.15`，先按完整状态、稳定性、安全、regression 和 domain 门槛筛选。复核 PPL `10.07/8.85/8.67`、分数 `0.565/0.595/0.577`、保存开销 `0.0075` 与回滚上限 `52.5B tokens`。
- `code_status`：§15 标准库代码运行正常，文中期望输出与执行结果逐字一致，选择 `ckpt_2000`；另验证配置缺失/忽略字段差异、空 manifest、合法零 seed、NaN、PPL overflow、回归分母超界、domain key 不匹配、无合格 checkpoint、空 checkpoint 列表、benchmark key 不匹配、权重和错误、重复 step 共 14 种边界。未运行训练、benchmark、真实 DCP、GPU、分布式环境或对象存储。
- `link_status`：新增 HELM、lm-eval、Dror、Oren、ConStat、PyTorch 2.14 DCP/async 和 HF Trainer v5.19.0 来源入口；本册目录第十章路径已存在。PyTorch 2.14 与核心论文页面已打开核验；GitHub `main`、MLflow `latest` 与部分 HF 通用指南仍需按目标版本复查。
- `sync_status`：`PAPERS.md` 增加五类评测/污染论文并更新正文入口；`GLOSSARY_EN_ZH.md` 补 Evaluation Contract、HELM、lm-eval 和 Contamination Signal，配对统计术语已有不重复；`INTERVIEW_BANK.md` 增 HELM/lm-eval、污染方法、成对比较、量纲归一化四题；`EXERCISES.md` 和章节新增评测协议、配对不确定性、污染审计与 checkpoint 选择练习。
- `validation`：本章一级标题 1—19 连续；Python 输出和 8 类边界路径通过；新增公式定义域、DCP 文档版本边界、练习与索引位置已复核；`git diff --check` 通过。未渲染 PDF/MathJax，未运行模型或真实评测。
- `open_risks`：未运行真实训练恢复、DCP/GPU、多 rank、对象存储、benchmark、污染检测或 scorer；没有估计目标模型 seed/prompt 方差，也没有目标服务多目标权重。论文结论仍绑定论文任务、模型和假设；toy 归一化权重不代表通用政策。
- `status`：第五册第十份正文完成本轮资料驱动精修与声明范围内的代码/结构验收；全书 goal 继续，下一份为第五册第十一章。

### 第五册第十一章：后训练与对齐训练深度精修验收（2026-10-26）

- `scope`：按当前 `WRITING_SPEC.md` 全文复读 `book-05-llm-training/chapters/11-后训练与对齐训练.md`。保留并补深后训练阶段、反馈来源/算法/训练配方/推理时计算的层次、PPO-RLHF、DPO、SimPO、GRPO、R1/R1-Zero、训练风险、数据要求、pipeline、诊断、总结和练习；对偏离本章主线的安全说明归回安全小节，其他冷门信息仍留有简短定位。覆盖图和来源 claim map 见 `research/training-post-training-alignment-depth-review-2026-10-26.md`。
- `changes`：PPO 拆清 current policy、rollout policy、reference policy、reward model 和 critic 的作用；DPO 给出 KL 约束解、partition function 抵消、pairwise loss 与概率 margin 手算，并标明 reference/β 的边界。新增 IPO、KTO、ORPO、SimPO 的目标和适用输入比较。GRPO 展开逐 token ratio、response advantage 广播、mask/reduction 和小组 reward 例子；补 DAPO 的 Clip-Higher、动态采样、token loss、overlong 处理，DrGRPO 的长度/组方差项，GSPO 的序列级几何平均 ratio，以及 DCPO 的 dynamic clipping、SAS、OTM、RUR。保留并扩写 DeepSeek-R1/R1-Zero 阶段和奖励设计；把偏好、RLVR 与在线 rollout 的关系落实为选型路径。用同一张表比较 RLVR、GRPO、DAPO、DrGRPO、GSPO、DCPO 各自瞄准的问题、改动、代表性证据与限制。模型案例改为有官方来源的 DeepSeek-R1、GLM-5.3、gpt-oss、Grok 4.7，并避免从产品披露臆测具体 optimizer/reward。修正 demo：方法建议根据任务和反馈/rollout 条件生成候选，不再由 preference category 推断算法；将所谓 GRPO loss 改称序列级 surrogate proxy，paired score gain 不再冒称偏好 win rate。修正算法表多列分隔符。
- `evidence`：研究笔记逐项记录 DPO、IPO、KTO、ORPO、SimPO、DeepSeekMath/GRPO、DAPO、DrGRPO、GSPO、DCPO、DeepSeek-R1、奖励模型过优化和 post-training scaling survey 的一手来源、主要机制、实验条件和外推边界。代表性原始资料包括 [DAPO](https://arxiv.org/html/2503.14476v2)、[DrGRPO](https://arxiv.org/html/2503.20783v2)、[GSPO](https://arxiv.org/html/2507.18071v2)、[DCPO](https://arxiv.org/html/2509.02333v2)、[DeepSeek-R1](https://arxiv.org/html/2501.12948) 与 [post-training scaling survey](https://aclanthology.org/2025.acl-long.140/)。官方案例核对 [GLM-5.3](https://z.ai/blog/glm-5.3)、[gpt-oss](https://openai.com/index/introducing-gpt-oss/) 和 [Grok 4.7](https://x.ai/news/grok-4-7)；这些是发布方披露，不等于统一设置的独立算法对照。GLM 博客页面无法由本轮 HTML 抽取器完整展开，相关数字在正文中明确保留为作者/发布方报告。
- `formula_status`：DPO 推导核对 KL 约束目标、归一化项消除及 pairwise loss；GRPO ratio/advantage/mask 的张量粒度明确；DrGRPO 以 response 长度 2/100、固定预算 100 展示长度系数差别；DCPO 动态边界按旧 token 概率 `q(x)` 展开，并记录 SAS/OTM/RUR 的定义差别。静态核对 20 个编号章节连续、35 个 math fence 花括号平衡。数值例均标为手算/教学例或绑定原论文配置；未运行外部数学渲染器。
- `code_status`：§17 单个标准库 Python demo 从 Markdown 提取后执行成功，退出码为 0，文中期望输出与实际标准输出完全一致。输出包含 RM loss `0.3749`、DPO loss `0.6786`、paired score gain rate `0.75`、safety recall `1.0`、over-refusal `0.3333` 和候选 `RLVR_candidates: GRPO/DAPO/DrGRPO/GSPO/DCPO`。示例仅计算数据/损失代理/诊断，不执行参数更新，不代表实际 GRPO、GSPO 或 DCPO 训练；未运行真实模型、训练栈或 benchmark。
- `link_status`：章节内部四个相对 Markdown 目标均存在；增加对应论文和官方技术来源链接。外部来源未做全量 HTTP 状态扫描，GLM 博客抽取可读性限制记录于研究笔记。
- `sync_status`：`PAPERS.md` 补入偏好优化、RLVR、post-training survey、reward overoptimization 和官方披露入口；`GLOSSARY_EN_ZH.md` 补 IPO/KTO/ORPO、GRPO 家族及 SAS/OTM/RUR；`INTERVIEW_BANK.md` 增加五类 RLVR 更新方法的机制、证据和公平对照题；`EXERCISES.md` 增加分母、全同 reward、RUR 与方案对照练习。第五册目录已存在本章入口，无需改动。
- `validation`：20 个编号一级标题连续；134 个 fenced marker 成对，35 个 math fence 花括号平衡，10 张表列数一致，4 个相对链接目标存在；demo 实际输出与文中输出一致；`git diff --check` 通过。未渲染 PDF/MathJax，也未运行真实训练、benchmark、GPU 或分布式栈。
- `open_risks`：DAPO 的 progressive ablation 是累积式配方，不是独立随机消融；DrGRPO 的 unbiased 结论受推导和采样假设约束；RUR 不等于正确率或有益梯度；各方法结果受模型、数据、模板、reward、采样、loss reduction 和评估协议约束，不能跨论文直接排名。厂商公开资料不披露完整训练 recipe，GLM 博客抽取受限。没有目标任务/模型上的统一复现。
- `status`：第五册第十一份正文完成本轮来源驱动精修与声明范围内的代码/静态验收；全书 goal 继续，下一份为第五册第十二章。

### 第五册第十二章：多模态训练深度精修验收（2026-10-26）

- `scope`：按当前 `WRITING_SPEC.md` 复读 `book-05-llm-training/chapters/12-多模态训练.md`。保留多模态任务图谱、CLIP、VLM、instruction tuning 与 label mask、caption/VQA、OCR/文档、ASR/语音问答、视频理解/生成、diffusion、数据组织、对齐、算力与 demo、安全/评估、策略、诊断和旧练习；旧知识点覆盖与转移原因记于 `research/training-multimodal-depth-review-2026-10-26.md`。
- `changes`：新增 CLIP→SigLIP→SigLIP 2 的目标/配方对照，说明伪负例不因 sigmoid 自动消除；以 BLIP-2、LLaVA-1.5、MM1 比较冻结 Q-Former、指令 connector 与数据/分辨率消融；以 OCR pipeline、LayoutLMv3、Donut、Pix2Struct 比较文字/布局/图像监督和 OCR-free 页面生成，加入 Pix2Struct screenshot parsing 消融；补 CTC、RNN-T、Whisper 的对齐格点和流式前提；加入固定帧/分桶/Frame Pack 成本对照、参数化相关的 Min-SNR 权重，以及数据配比、loss 系数和 JEST joint-batch 选择的区分。新增 6 项配方与机制练习，补齐原 7.5/7.6 编号顺序；原有安全、评估和工程主题均保留。
- `evidence`：研究记录逐篇整理 CLIP、SigLIP/官方 Big Vision loss 实现、SigLIP 2、BLIP-2、LLaVA-1.5、MM1、JEST、LayoutLMv3、Donut、Pix2Struct、CTC、RNN-T、Whisper、Min-SNR、CogVideoX 与 ACL 2025 MLLM 综述。定量结论均限定在作者配置：MM1 `45/45/10` 是其 1.2B ablation mixture；Pix2Struct DocVQA 完整 parsing 与移除 parsing 的 ANLS 67.8/12.2 属论文 base 配置；Min-SNR 约 3.4x 收敛迭代与 FID 2.06 属 ImageNet 256 / ViT-B 设置；CogVideoX 的 frame pack 与 timestep 观察属该论文 recipe。资料来源、机制和外推限制见研究笔记及章节链接。
- `formula_status`：核对 SigLIP pair label/scale/bias logistic loss 与多正例目标；手算 CLIP `e³/(e³+e²)≈0.731`、loss≈0.313 和 SigLIP 错负例 `softplus(2)≈2.127`。CTC `book` 字符级最短路径为 5 帧，RNN-T `160×20≈3,200` 只是格点量级；patch16 视频 token 数 1,568/12,544，对应 full-attention score cell 比 64；复算 Min-SNR γ=5、SNR=100/0.1 时 `x0/epsilon/v` 权重，并把 SNR=0 的 epsilon 权重按连续极限设为 1；说明任务混合系数不等同梯度贡献。
- `code_status`：§12 标准库 demo 提取执行，实际 stdout 与文中期望输出逐字一致。另对零视频帧、零向量归一化、空审计批次、越界 CE target 和重复样本 ID 做针对性异常输入检查，5 项均按预期拒绝。未运行真实 OCR/ASR、VLM/diffusion 模型训练、GPU、benchmark 或生产媒体管线。
- `sync_status`：`PAPERS.md` 增多模态训练配方、文档/语音/视频/diffusion 机制和 MLLM 综述入口；`GLOSSARY_EN_ZH.md` 增 SigLIP、BLIP-2、MM1、JEST、Donut、Pix2Struct、RNN-T lattice、Min-SNR 与 Frame Pack；`INTERVIEW_BANK.md` 增 57—62 六道研究设计/方法比较题；`EXERCISES.md` 新增“多模态训练：目标函数与配方对照”六题。第五册目录已映射本章，无需改动。
- `validation`：18 个编号一级标题连续；108 个 fenced marker 成对，41 个 math fence 花括号平衡，9 张 Markdown 表列数一致，6 个语音小节编号连续；demo 与 expected output 一致，5 类异常输入均拒绝；章节和配套资料共检查 104 个本地 Markdown 链接目标，均存在；`git diff --check` 通过。
- `open_risks`：没有在目标多模态语料或硬件上训练/复现任何论文方法，也没有比较统一模型、分辨率、token/FLOPs 和 benchmark；论文成绩受原始数据、参数、采样、prompt、预处理和评估脚本约束。签名式 pair loss 不会修复标签噪声；Frame Pack 不改变总 token 的全注意力平方项；合成文档和网页截图来源需要目标业务域验证；原有代码只是多项局部审计，不是完整多模态训练系统。未渲染 PDF/MathJax。
- `status`：第五册第十二份正文完成本轮资料驱动精修与声明范围内的教学代码/静态验收；全书 goal 继续，下一份为第五册第十三章。

### 第五册第十三章：训练成本与资源规划深度精修验收（2026-10-26）

- `scope`：按当前 `WRITING_SPEC.md` 复读 `book-05-llm-training/chapters/13-训练成本与资源规划.md` 全章。保留总成本、`6ND` 与长序列/MoE 边界、吞吐与 GPU 数反推、MFU/HFU、数据/Chinchilla、状态显存、checkpoint 带宽、故障重跑、资源规划 demo、误判和诊断练习；旧稿覆盖与新证据映射见 `research/training-cost-resource-planning-depth-review-2026-10-26.md`。
- `changes`：增加设施电量/运行碳的 `IT energy → PUE → grid intensity` 核算和 Llama 2/OLMo 统计范围；补 PaLM MFU/HFU 与 MegaScale 同论文强扩展及长任务恢复的证据层对照；比较 Kaplan、Chinchilla 与 inference-aware scaling 的不同优化目标，明确训练计算最优不自动等于寿命期请求成本最优；补六项资源/缩放/碳/恢复练习。
- `evidence`：研究记录实际阅读 Kaplan、Chinchilla、Beyond Chinchilla、PaLM、MegaScale、Llama 2、OLMo 原论文对应方法、实验或限制。代表数值分别绑定论文配置：Chinchilla 70B/1.4T 与 Gopher 同预算案例；PaLM 540B/6144 TPU v4 的 MFU/HFU；MegaScale 175B 强扩展及专有生产任务故障恢复；Llama 2/OLMo 的碳排是不同功率、PUE、电网和范围假设下的估算，不能跨报告排序。
- `formula_status`：复核 `E_IT=ΣP_it_i`、设施电量约 `PUE×E_IT`、排放约 `CI×E_facility` 的量纲；手算 0.4kW×100,000 card-hours=40,000kWh IT，PUE 1.2 得48,000kWh设施用电，碳强度0.5kg/kWh得24tCO₂e。`6ND`、吞吐与失败损失仍按正文条件解释，不将近似当作实测账单。
- `code_status`：标准库 demo 执行成功，12 行输出与正文一致；5 类非法输入（零参数、MFU 超界、overhead=1、负 checkpoint 数和 NaN 故障率）均被拒绝。只验证局部教学账本，不代表集群 benchmark、真实 GPU-hours、故障分布或电表读数。
- `link_status`：章节 17 个 Markdown 链接中无失效本地目标；第五册目录已有第十三章映射。未声称本轮对所有既有外链做了全量 HTTP 扫描。
- `sync_status`：`PAPERS.md` 更新 inference-aware scaling 与 MegaScale/训练能耗入口；`GLOSSARY_EN_ZH.md` 补训练能源、PUE、电网碳强度和运行排放；`INTERVIEW_BANK.md` 加计算最优与生命周期成本、GPU-hours/能耗核算题；`EXERCISES.md` 增资源规划专项练习。
- `validation`：第 1—15 节编号连续；82 个 fenced marker 配对，31 个 math block 花括号平衡，9 张表列数一致；章节内本地链接目标完整；demo 输出一致；`git diff --check` 通过。未运行云价格查询、实际训练、GPU 多节点、生产 checkpoint 或 PDF/MathJax。
- `open_risks`：论文报告、预算假设、硬件和碳核算范围各异；未在目标集群、云账单、功率计、GPU 或真实恢复演练上验证。代码阈值是教学输入，不能视为行业标准。
- `status`：第五册第十三章完成来源驱动精修和声明范围内的公式/代码/静态验收；该检查点的下一章为第十四章。

### 第五册第十四章：训练系统综合设计与故障诊断深度精修验收（2026-10-26）

- `scope`：按当前 `WRITING_SPEC.md` 全文复读 `book-05-llm-training/chapters/14-训练系统综合设计与故障诊断.md`。保留训练账本、7B 方案、FLOPs/token batch、loss spike、能力回归、benchmark contamination、batch/LR、并行、显存、SFT、VLM、成本、阶段、DPO/PPO、完整账本代码与练习；本章作为前文综合设计入口，不替代数据、优化器、并行或后训练章节的主讲。
- `changes`：加入 DoReMi reference/proxy + Group DRO → 大模型 resampling 的机制、The Pile 280M→8B 结果及 proxy 尺度限制；对照 Molybog 2023 时间相关 Adam 更新与 Bai 2025 二阶矩失配两类 spike 假说，并补 `beta2=0.999,k=1000` 下 `v`/平方根分母变化的教学推演和首次异常反事实排查步骤。增加 PALOMA 多域评测及其 PPL 范围；把 exact/近似匹配、PaCoST 配对置信信号和新鲜/隐藏评测分开，区分 candidate/confirmed overlap rate。补 PyTorch 2.14 DCP 跨拓扑 reshard 与训练语义恢复边界。综合审计代码将目标、数据、模型、优化、系统、评估、风险、证据作为独立字段。
- `evidence`：研究记录阅读 DoReMi 方法与规模消融、Molybog 的大模型训练轨迹/讨论、Bai 的预条件 Hessian 与 187M/100B-token LLaMA 实验、PALOMA 评测指南、PaCoST 配对检验/限制、PyTorch 2.14 DCP API 与教程。所有数值和结论均绑定原文设置；不据论文机制诊断真实生产 run。
- `formula_status`：手算在零梯度近似下 `0.999^1000≈0.368`，平方根项约 `0.607`，倒数缩放约 `1.65×`；明确这只解释预条件项，不预测包含一阶矩、bias correction 和梯度方向的完整 Adam 轨迹。污染扫描候选率与人工确认率采用分开的分母定义；`N_eval>0`。
- `code_status`：综合账本 Python demo 运行成功，6 行输出与正文完全一致；验证缺失 evaluation/risk、非法污染状态和未知评测隔离状态的门禁。只证明 toy 结构/资源账本，不验证训练或 contamination 检测算法。
- `link_status`：章节 22 个 Markdown 链接无失效本地目标；新增 DoReMi、Adam instability、Adaptive Preconditioners、PALOMA、PaCoST 和 PyTorch 2.14 DCP 文献/官方文档入口；本册目录已有第十四章映射。
- `sync_status`：`PAPERS.md` 增 PALOMA 与 PaCoST；`GLOSSARY_EN_ZH.md` 增 Candidate Overlap Rate、PaCoST、PALOMA；`INTERVIEW_BANK.md` 扩充 loss spike、污染审计和完整训练设计回答；`EXERCISES.md` 增系统审计、mixture、spike 反事实和污染报告练习。
- `validation`：章节 0—19 标题连续；54 个 fenced marker 配对，24 个 math block 花括号平衡，22 个链接的本地目标完整；demo 标准输出逐字匹配，三类字段/状态边界被拒绝或触发不完整审计；`git diff --check` 通过。未运行真实训练、GPU/FSDP/DCP、多轮模型评估、PaCoST 或外部公式渲染。
- `open_risks`：DoReMi、Adam spike、PALOMA 和 PaCoST 均未本地复现；没对目标 benchmark、目标模型、私有训练语料或黑盒服务做污染/行为检测；PyTorch DCP 文档不代表本地多卡测试。研究结果不能替项目日志和目标任务实测。
- `status`：第五册第十四章完成资料驱动精修和声明范围内代码/公式/链接验收；全书 goal 继续，下一份为第五册第十五章。

### 第五册第十五章：领域专家 SFT 与 RL 深度精修验收（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 全文复读 `book-05-llm-training/chapters/15-领域专家sft与rl.md` 的 15.0—15.30。保留原有领域路线、SFT mask、专家样本层次、标注链、金融例子、工具安全、评估/失败分类、奖励与策略边界、verifier 组合、分布覆盖、三层正确性、专家分歧、遗忘与版本更新、成本、发布、Python 审计 demo 和练习；原有知识点去向见 `research/training-domain-expert-sft-rl-depth-review-2026-10-26.md`。
- `changes`：在 15.0 加基于研究的路线选择和方案对照，区分 DAPT/TAPT、FinGPT/RLSP 市场弱标签、TIMER 时间戳 SFT、DeepSeekMath 结果可验证 RL；用 FinBen 与 MedAlign 对照 benchmark 文本分析和实际行业工作流评估。明确 DPO 偏好、市场代理标签与 RLVR verifier 的监督语义不同；补 TIMER 数据构造、同底模对照、纵向时间线偏差，以及临床/金融评测集的实体、机构、时间与版本隔离要求。表中将论文观察和本书建议的后续验证分开。所有旧主题仍保留，核心是加深适配信号与评测选择依据。
- `evidence`：research note 逐项记录 DAPT/TAPT 原始实验、LLM continual learning survey、FinGPT 两篇论文、NeurIPS 2024 FinBen、AAAI 2024 MedAlign、npj Digital Medicine 2025 TIMER 和 DeepSeekMath。FinBen 按最终发表版记 42 datasets/24 tasks；TIMER Llama 3.1 8B 在 MedAlign correctness 30.69%→34.32%、TIMER-Eval 45.02%→48.51%，受控分布消融的 +6.5 是相对 win advantage；DeepSeekMath 的 MATH 46.8%→51.7%、GSM8K 82.9%→88.2% 绑定其 7B/语料/判分设置。市场价格标签受混杂，不描述为新闻因果效应或收益保证。
- `formula_status`：核对 13 个 math fence 中花括号、方括号和圆括号配对；手算 TIMER 两项 accuracy 差分别为 +3.63/+3.49 个百分点，区分与分布消融 +6.5 win advantage；DeepSeekMath 两项 RL 前后差分别为 MATH +4.9、GSM8K +5.3 个百分点。公式均保留适用条件和分母/未知状态定义，未使用外部公式渲染器。
- `code_status`：§15.28 标准库 demo 提取执行成功；正常输出毛利率 `0.36`、单位成功成本 `7.5`、发布通过；缺 revision、零收入、越权写入、零成功任务、负复核成本均触发预期错误，未完成 rollback drill 保持发布阻断。只验证局部审计逻辑，不代表财务准则、真实授权、模型正确性或生产发布。
- `link_status`：章节和研究笔记相对 Markdown 目标均存在；关键论文链接指向 arXiv、AAAI/NeurIPS 论文页或 Nature 发表页。第五册目录已有第十五章映射。PAPERS/EXERCISES/INTERVIEW_BANK 的第十五章新入口存在。
- `sync_status`：`PAPERS.md` 增领域适配和行业评测来源；`GLOSSARY_EN_ZH.md` 增 DAPT、TAPT、CPT、RLSP、FinBen、MedAlign、TIMER；`INTERVIEW_BANK.md` 增领域训练路线和市场弱反馈 vs RLVR 比较题；`EXERCISES.md` 增领域方案选择、研究对照、时间评估、市场标签和评测计划五项。
- `validation`：一级章节标题 15.0—15.30 连续；17 对 Markdown fence 成对，13 个 math fence 符号平衡；章节与研究记录本地链接通过；demo 六类正常/异常路径符合正文说明；`git diff --check` 通过。
- `open_risks`：未在目标机构/领域数据上重训或复现 TIMER、FinGPT、DeepSeekMath；没有执行真实医疗/金融评测或权限系统。论文数值仅是对应实验范围的观察；FinGPT 市场标签有混杂，TIMER 生成指令仅有限专家审阅且未完成亚群公平性/校准评估，FinBen/MedAlign 的任务和数据来源也有限。
- `status`：第五册第十五章完成本轮来源驱动精修和声明范围内代码/公式/链接验收；全书 goal 继续，下一章为第五册第十六章。

### 第五册第十六章：RLVR 与可验证奖励深度精修验收（2026-10-26）

- `scope`：按当前 `WRITING_SPEC.md` 逐段复读 `book-05-llm-training/chapters/16-rlvr与可验证奖励.md` 的 16.0—16.22。任务契约、reward mapping、unknown、outcome/process reward、verifier 类型、soundness/completeness、覆盖率、代码/SQL、沙箱、GRPO、DeepSeekMath/DeepSeek-R1/DAPO、reward hacking、污染、独立评测、成本、稳定性、故障路径、RLVR/RLHF 对比、安全门禁、结构化答案 demo 均保留；旧知识点去向和研究判断见 [`research/training-rlvr-verifier-depth-review-2026-10-26.md`](research/training-rlvr-verifier-depth-review-2026-10-26.md)。
- `changes`：增加 PRM800K 的结果监督/过程监督实证，并限定固定生成器候选排序边界；增加 extensional correctness 与逻辑结构泛化区别，用 IPT 展开 verifier 只查样例时的 shortcut；将 SAGA/TCGBench 的人工正负代码解分析和验证器质量度量、ImpossibleBench 的规格/测试冲突任务用于补强代码 verifier；增加 GRPO clipping bias 与 spurious reward 归因对照，提醒 reward 上升可能包含算法和底模先验；在评测矩阵增加同构/变形测试、mutation testing、独立 artifact 重算和随机 reward 对照。修复 demo 中极大整数转 float 时的潜在 `OverflowError`，补 500 位数值输入和实际输出表。
- `evidence`：资料笔记记录 PRM 在 MATH best-of-1860 的 78.2% vs ORM 72.4% vs majority 69.6%，明确是固定候选排序而非在线 RL；SAGA 在 270 题 TCGBench-Lite 的 DR@50 90.62%、VAcc@50 32.58%、AUC@50 0.2228，对比随机 Input-Interpreter 81.07%、16.72%、0.1234，按 collected wrong submissions 解释 VAcc 分母；IPT 的 Olmo-3 7B/verifier 对照在约 500 步后约 3.5 reward-point gap、研究范围限 SLR-Bench；ICML 2026 Spurious Rewards 报告 Qwen2.5-Math-7B 的 MATH-500 随机/ground-truth reward 增量 21.4/29.1 个百分点并分析 GRPO clipping bias，外推限数学/底模/配方；ImpossibleBench 的 conflict mutation 仅作为 agent exploit evaluation；动态 TRL docs 仅用于 implementation/version awareness。各来源的限制、机制与建议分别记录于 research note。
- `formula_status`：核对 PRM 78.2/72.4/69.6 为 best-of-1860 指标；SAGA DR@50/VAcc@50 的定义及 270 题、约 41.41 incorrect submissions/题范围；Spurious Rewards 的 21.4/29.1 均为 Qwen2.5-Math-7B、MATH-500 的绝对准确率增量，差为 7.7 个百分点；IPT 3.5 是论文受控训练图中的 reward gap，不是 accuracy。对新增计算逐项复核；未运行外部公式渲染。
- `code_status`：提取并运行 §16.15 标准库 demo。成功路径输出 `reward=1.0, reason=success`；错答为 `wrong_answer`；越权动作为 `hard_condition_failed:permission`；截断 JSON、NaN、重复 key、字节预算、500 位整数分别得到 `invalid_json`、非有限值拒绝、`duplicate JSON key`、`budget_exceeded`、`schema_error`；三种非法调用配置均抛出调用方 `ValueError`。仅验证 toy parser/verifier 的声明范围，不代表 GRPO、真实模型、服务接口或生产沙箱。
- `link_status`：核对新增原论文/论文页、NeurIPS 2025 SAGA、ICLR 2026 ImpossibleBench、PMLR ICML 2026 Spurious Rewards、IPT arXiv 和动态 TRL 文档；新增研究记录与本章相对链接目标存在。第五册第十六章文件已在目录中，无新增章节文件。
- `sync_status`：`PAPERS.md` 增 SAGA、ImpossibleBench、IPT、Spurious Rewards 与 GRPOTrainer 入口；`GLOSSARY_EN_ZH.md` 增 Extensional Correctness、IPT、Metamorphic Testing、Spurious Reward、Reward Signal Attribution；`EXERCISES.md` 增 43F verifier/red-team 与 dummy reward 对照练习；`INTERVIEW_BANK.md` 增 20G verifier 分数/能力归因题。第五册目录路径不变。
- `validation`：复核 16.0—16.22 编号与小节顺序；检查新增加的 markdown/math/python fences 配对、SAGA/IPT/dummy reward 数值条件、内部相对链接；运行 §16.15 后输出与新增判定表逐项一致；`git diff --check` 通过。未执行 GRPO/TRL 训练、代码模型 benchmark、外部公式渲染、PDF 构建。
- `open_risks`：PRM800K 是 reward-model best-of-N 任务，不能替代策略训练比较；SAGA VAcc 只按已收集错误提交定义；ImpossibleBench 与 IPT 是不同类型 shortcut 评测，不能互换；spurious-reward 现象的模型和训练配方依赖很强；本轮未在目标 verifier、数据、底模或工具工作流上做真实实验。动态 TRL 文档需绑定所用 revision 才能审计当前 API。
- `status`：第五册第十六章完成来源驱动的内容精修、旧知识点覆盖映射和声明范围内 demo/链接验收；全书 goal 继续，下一章为第五册第十七章。

### 第五册第十七章：On-Policy Distillation 深度精修验收（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 复读 `book-05-llm-training/chapters/17-on-policy-distillation.md` 的 §17.0—17.24；离线/在线分布、状态契约、token/sequence/artifact 目标、teacher reliability、rollout、成本/隐私、估计器、位置偏差、稳定方法、verifier、多 rollout、安全、独立评估、阶段组合、失败诊断、方法对照和练习均保留。旧知识点映射和研究证据见 [`research/training-on-policy-distillation-depth-review-2026-10-26.md`](research/training-on-policy-distillation-depth-review-2026-10-26.md)。
- `changes`：纠正 sampled `log p_T-log p_S` 与直接监督 loss 的混淆；补充固定前缀期望、full-vocabulary local KL、score-function surrogate、action-independent baseline 和 sequence-level return-to-go 区别，并加两动作数值手算。研究表新增 Qwen3 技术报告对照，MOPD 行加入 AIME'24、LiveCodeBench-v6 的 mean@8、peer-context 消融和 teacher compute 代价；修订 REOPOLD 的代码状态措辞以避免把“尚未复现”写成“没有代码”。新增 Tinker 的 mid-training 后行为恢复案例及 TRL DistillationTrainer/GKDTrainer 实现/版本语义。补方法阅读地图、来源分层和 5 项练习。
- `evidence`：核对 DAgger 原论文、MiniLLM/GKD、Qwen3 Technical Report §4.5、§4.7/Table 21、OPD v5 survey、formula-driven survey、MOPD 原文、Tinker 工程文章、TRL 动态官方文档及 Microsoft LMOps MiniLLM GitHub。Qwen3 表中同一 off-policy-distilled Qwen3-8B 起点下，OPD/RL AIME'24 为 74.4/67.6、AIME'25 65.5/55.5、LiveCodeBench v5 60.3/52.9，GPU-hours 1,800/17,920；MOPD Qwen3-4B AIME'24 mean@8 为 28.54，base 16.89、SDPO 7.29；LCBv6 mean@8 57.01，base 28.80、SDPO 48.84。各结果受模型、数据、teacher、生成预算和评测设置限制。
- `formula_status`：手算 `p_S=(0.5,0.5)`、`p_T=(0.8,0.2)` 的 `r_A≈0.470`、`r_B≈−0.916`、`E[r]≈−0.223=−KL`、baseline 后 advantage `(+0.693,−0.693)`；复核 Qwen3 OPD-RL pp 差和 `17,920/1,800≈9.96×`；复核 MOPD mean@8 差。说明这些手算只验证概率/表格关系，不证明任务正确或跨论文排名。
- `code_status`：提取运行 §17.16 标准库 Python teacher-call/cache demo；正常触发与异常输入输出符合正文，cache identity 会随 action/risk/failure state 改变。只验证 toy 代码，不代表 TRL、真实 teacher API、OPD 训练或生产缓存。
- `link_status`：研究原文/官方报告、Tinker、TRL 和 Microsoft GitHub 来源已打开核验；章节与研究笔记相对链接有效。第五册目录已有第十七章条目，无需另改。
- `sync_status`：`PAPERS.md` 增 OPD 专题索引；`GLOSSARY_EN_ZH.md` 补 student rollout、policy lag、sampled log-ratio、temporal credit、GKD、MiniLLM 和同名 MOPD 区分；`INTERVIEW_BANK.md` 增 20H 方法与证据比较题；`EXERCISES.md` 增同分布手算、方法对照、Qwen3 复现设计和版本化 trainer 审计。
- `validation`：核对 §17.0—17.24 编号、Markdown/math/code fences、表格列、公式和新增数值、章节和研究笔记本地链接；运行 §17.16 demo；`git diff --check` 通过。未运行真实 TRL/GPU/LLM 训练、独立 benchmark、外部公式渲染或 PDF 构建。
- `open_risks`：Qwen3 GPU-hours 是单一报告的总量，训练硬件/外部复现不足以推导跨机构成本比；Tinker QA eval 为实验室自有数据；TRL 当前 `main` 动态且未本地安装；MOPD 没有成功 peer 的 prompt 无法产生成功对照；本轮没有在目标领域、模型或工具环境训练 OPD。
- `status`：第五册第十七章完成来源驱动精修、知识点覆盖映射、配套同步和声明范围内代码/公式/链接验收；全书 goal 继续，下一章为第五册第十八章。

### 第五册第十八章：DeepSeek V4 领域专家与统一蒸馏深度精修验收（2026-10-26）

- `scope`：按当前 `WRITING_SPEC.md` 从头复读 `book-05-llm-training/chapters/18-deepseek-v4领域专家与统一蒸馏.md` 全文。保留原稿的领域 SFT/GRPO、权限边界、学生 rollout、reverse-KL、教师选择、审计循环、成本/失败模式、面试和练习；逐项旧知识去向见 [`research/training-deepseek-v4-unified-distillation-depth-review-2026-10-26.md`](research/training-deepseek-v4-unified-distillation-depth-review-2026-10-26.md) 的 `coverage_map`。
- `changes`：区分完整领域 teacher checkpoint、最终 MoE 内 routed expert 与统一学生；澄清“OPD 替换 mixed RL”不表示 V4 完全不用 RL；新增 GRM 的作者方案、校准风险和披露缺口。把 weighted reverse-KL、学生状态分布、sampled log-ratio 的 stop-gradient policy surrogate、full-vocabulary logits 选择动机连成机制；加三 token KL/标准差数值例及可执行 Python。推导多教师 reverse-KL 的归一化加权几何平均，说明教师冲突和路由权重影响。新增 V4 §5.2.2 教师权重分片、末层 hidden-state 缓存、输出头按需重建、teacher-index batching、异步读写和 TileLang exact KL；区别框架支持上限与实际 teacher 数量。补 Model Soups、离线 response distillation、GKD、MiniLLM、V4 full-vocabulary multi-teacher OPD 的机制/证据/限制表，并给出明确标成研究假设的改进实验表。
- `evidence`：读取 [DeepSeek V4 技术报告](https://arxiv.org/html/2606.19348) §5.1.1、§5.1.2、§5.2.2 与[官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)；核对 [GKD](https://arxiv.org/abs/2306.13649)、[MiniLLM](https://arxiv.org/abs/2306.08543) 原论文入口以及 [Model Soups](https://proceedings.mlr.press/v162/wortsman22a.html) PMLR 页面/论文。报告支持领域 SFT+GRPO、10+ teacher、reverse KL、full-vocabulary OPD 和教师调度机制；数据、路由权重、完整 reward/loss、对照消融和系统容量指标均未公开，不能从最终模型成绩单独归因 OPD 增益。
- `formula_status`：核对 11 个 display-math 围栏和花括号平衡；计算 `p_S=(0.60,0.30,0.10)`、`p_T=(0.70,0.20,0.10)` 的 reverse KL `≈0.029149` nat 与单 token log-ratio 标准差 `≈0.25046` nat；推导两教师等权最优分布 `(0.6,0.4)`；复算 BF16 logits `8,192×100,000×2≈1.526 GiB/teacher` 和 `logits = W_out × hidden_state` 维度。数值均为教学算例，不代表论文实测。
- `code_status`：逐块运行两个标准库 Python demo，输出与正文一致；边界检查覆盖空组/空分布、NaN/Inf、bool、非正 epsilon、长度不符、负概率、归一化失败、教师 support 缺失和零方差，共 11 个预期 ValueError 用例及 2 个合法边界用例通过。只验证 toy 代码，不代表真实 GRPO、V4 权重、GPU 或 OPD 训练。
- `link_status`：打开核验 DeepSeek 报告相关章节、官方 Hugging Face 模型卡、GKD、MiniLLM 与 PMLR Model Soups；第五册目录已存在第十八章条目，无需改导航；新增研究笔记、练习和题库相对链接目标存在。
- `sync_status`：`PAPERS.md` 增 DeepSeek V4 full-vocabulary OPD 和 Model Soups 比较入口；`GLOSSARY_EN_ZH.md` 增 Domain Specialist Model、GRM、Full-Vocabulary Logit Distillation、Teacher Scheduling；`INTERVIEW_BANK.md` 增 mixed-RL、MoE/领域专家区分、全词表方差、教师调度和方案比较题；`EXERCISES.md` 增 5 项多教师 OPD 练习。
- `validation`：全文复读；检查代码/数学 Markdown fences 成对、11 个数学围栏括号平衡、5 张表格列数一致、本地链接目标存在、反向 KL 与显存数值、公式符号定义；运行两个代码块与异常输入检查；`git diff --check` 通过。未运行真实 DeepSeek V4/GRM/OPD、GPU benchmark、外部公式渲染或 PDF 构建。
- `open_risks`：没有 V4 权重、目标集群或真实 GRM/OPD 训练复现；报告未公开教师路由、GRM schema/calibration、full-vocabulary 与 sampled OPD 受控消融、峰值显存和可迁移端到端吞吐。Model Soups/GKD/MiniLLM 结果来自不同模型/任务，本文只比较机制与证据边界。
- `status`：第五册第十八章完成来源驱动精修、原知识点覆盖、方法比较、配套同步和声明范围内代码/公式/链接验收；全书 goal 继续，下一章为第六册第一章。

### 第六册第八章：并行推理与多 GPU 服务资料驱动精修验收（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 全文复读 `book-06-llm-deployment/chapters/08-并行推理与多gpu服务.md` §1—§16。保留 TP/PP/EP、数据并行副本、请求路由、通信拓扑、KV 布局与迁移、故障处理、选型流程、8 卡方案、教学 demo 和原有资料；原稿去向、研究问题与证据详见 [`research/inference-parallel-multigpu-depth-review-2026-10-26.md`](research/inference-parallel-multigpu-depth-review-2026-10-26.md)。
- `coverage_map`：研究记录逐段映射原稿。§1—§4 容量与 TP/PP 保留并补清计算/通信账本；§5 EP 保留 dispatch/combine 与 expert imbalance，增加 DeepEP、EPLB 和冗余专家显存约束；§6—§8 副本、调度、通信与拓扑保留；§9—§11 保留 KV/prefix locality、迁移与恢复语义，改写重复总结为带 owner epoch 的交接设计建议；§12 保留选型流程并增加 P/D 方案比较；§13—§16 保留 demo、8 卡案例、来源和小结。
- `changes`：比较共置 + chunked prefill、DistServe、Mooncake、TaiChi、SmartGen 的问题、机制、作者证据、成本和适用边界；以 SpectrumKV 补充 KV 传输精度—质量权衡，指出请求吞吐、goodput、TTST 和 PPL 不可混排。明确 P/D 解耦与 TP/PP/EP 的资源切分轴不同，并将 KV payload、可隐藏传输比例、SLO 差异和目标端 KV 预留纳入判断。给 §11 加入 freeze、传输校验、epoch 提交/fence 与回退步骤，且明确它是设计建议，不声称框架已实现。修正 TP 教学账本范围和 PP2 payload 输出；明确 PP payload 表示一次 decode iteration 跨全部 stage 边界的逻辑总量。
- `evidence`：研究笔记对照 DistServe、Mooncake、TaiChi、SmartGen、SpectrumKV 的目标、机制、作者测量和局限，并核对 vLLM P/D 与 EPLB 文档、SGLang P/D 文档、DeepEP 仓库、NCCL 与并行配置文档。所有吞吐/SLO/TTST/PPL 数字均绑定各自模型、硬件、trace 和指标定义，不作跨论文排名；动态主分支文档按核验时点记录。
- `formula_status`：复核 70B FP16 权重约 130.39 GiB；TP4 两副本每 rank toy 总账为 32.6+7.5+8=48.1 GiB，TP8/TP4×PP2 每 rank 约 28.05 GiB。TP payload 按 `2×80×8192×2×24 / 2^20 = 60 MiB` 计算，是每 rank 参与 collective 的逻辑输入量，不是物理网络收发量。PP2 边界为 `8192×2×24 / 2^20 = 0.375 MiB`。均不代表目标拓扑、运行时或服务性能测量。
- `code_status`：提取并运行 §13 标准库 Python demo，退出码 0、stderr 为空，stdout 与 fenced `text` 示例逐字一致；核对 PP2 边界 payload 和 TP=1/PP=1 零通信路径。教学代码未连接 GPU/NCCL。
- `link_status`：研究笔记收录论文及官方文档入口；正文和研究笔记的本地 Markdown 链接目标均存在。第六册目录已有第八章入口，无需改目录。
- `sync_status`：`PAPERS.md` 补 P/D 解耦研究、SmartGen/SpectrumKV 和 DeepEP/EPLB；`GLOSSARY_EN_ZH.md` 补 DeepEP、EPLB、DistServe、Mooncake、TaiChi、SmartGen、SpectrumKV、TTST 及 owner epoch/fencing；`INTERVIEW_BANK.md` 扩展 PD 分离比较题；`EXERCISES.md` 增 P/D 方案同条件对照题。章节目录已存在，无需改动。
- `validation`：检查 §1—§16 连续、28 对代码/数学/文本围栏闭合、11 张表格列数一致、内部 Markdown 链接目标存在；demo 输出、核心数值账本和 PP/TP 边界通过；`git diff --check` 通过。未做外部公式渲染、PDF 构建、目标 GPU/网络 benchmark 或故障注入。
- `open_risks`：没有在目标模型、GPU、拓扑、请求 trace 或 runtime 上复现 TP/PP/EP/P-D/Mooncake/SmartGen。论文指标不可混比，当前 vLLM/SGLang/DeepEP 主分支文档会变化；EPLB 冗余专家显存依赖模型与配置。owner epoch handoff 需结合实际控制面、流式 API 和外部副作用实现。demo 中 60 MiB 是假设下的逻辑 collective 输入量，不是实际链路流量。
- `status`：第六册第八章完成资料驱动精修、原知识点覆盖、代表方案比较、配套同步及声明范围内代码/公式/链接验收；全书 goal 继续，下一章为第六册第九章。

### 第六册第七章：量化与模型压缩资料驱动精修验收（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 全文复读 `book-06-llm-deployment/chapters/07-量化与模型压缩.md` §1—§19。原稿格式、基本量化公式、PTQ/QAT、GPTQ、AWQ、SmoothQuant、KV Cache、蒸馏、剪枝、评估、误区、Python demo、长上下文决策案例和小结全部保留；知识点覆盖图与来源证据见 [`research/quantization-model-compression-depth-review-2026-10-26.md`](research/quantization-model-compression-depth-review-2026-10-26.md)。
- `changes`：补 GPTQ/AWQ 与 LLM.int8()/ZeroQuant/SmoothQuant 的误差机制、论文测量结果和执行代价对照；展开 KIVI 的 K/V 分组轴、residual 生命周期及数值误差；加入 TurboQuant 论文失真目标与 vLLM 真实 runtime 评测差异、MXFP4/NVFP4 scale 合约、active/cold-page cache 边界；保留蒸馏和剪枝，并增加 SparseGPT 研究证据。新增 §14.1 把校准偏移、动态 scale、residual 存储、低 bit 解码开销、稀疏 kernel 缺失和学生容量限制转换为可证伪的下一轮实验假设，明确它们不是论文已验证的通用改进。修复 demo 的非对称常量激活映射和输入验证；修正长上下文显存账本假设，避免把 7B W4A16 + BF16 KV 的预算写成“必然不够”。
- `evidence`：研究记录比较 GPTQ、AWQ、LLM.int8()、ZeroQuant、SmoothQuant、KIVI、TurboQuant、SparseGPT、压缩综述和统一量化评测，逐一保留解决目标、机制、作者证据、代价和适用边界。核对 [TurboQuant 原论文](https://arxiv.org/abs/2504.19874) 与 [vLLM 实测](https://vllm.ai/blog/2026-05-11-turboquant) 的层次：Qwen3 MRCR 最长 256k AUC 中 BF16 为 45.8、3bit-nc 为 31.2；vLLM 实测绑定 0.20.2 commit、模型、H100 数量与 workload。[TensorRT-LLM KV Cache Compression 文档](https://nvidia.github.io/TensorRT-LLM/features/kv-cache-compression.html) 当前列出的 cold-page 条件为 PyTorch backend、KVCM V2、SM100/SM103/SM107；NVFP4 cold-page 另需 native C++ KVCM V2 和非零 Host/Disk cache。其页面标注 2026-10-19 更新、源码 commit `0372799`。
- `formula_status`：复核 `M_KV≈2BLTH_kvd_hb_q` 和 group-scale 开销公式；长上下文案例按每 token 131,072 bytes、16 个近 P95 请求、8,704 cached tokens 手算得 17 GiB KV；7B FP16 约13.04 GiB，INT4 raw+128-group FP16 scale 约3.36 GiB，W4A16+BF16 KV 总预算约20.36 GiB，尚未含 activation/workspace/runtime 等全部峰值。数值只验证账本和容量边界，不代表目标框架峰值；未用外部公式渲染器。
- `code_status`：提取并运行 §15 标准库 demo，退出码 0、stderr 为空，七行 stdout 与正文 fenced `text` 示例完全一致；另验证正/负常量、全零、全正/全负取值范围的量化误差界，以及空输入、NaN、bool、非法 bit/group 等 11 个边界用例。只证明 toy 函数的输入合同和数值行为，不代表模型量化或生产性能。
- `link_status`：新增论文、综述、Google Research、vLLM、OCP、NVIDIA 和框架官方文档入口；正文与研究笔记本地 Markdown 链接目标存在。TensorRT-LLM 动态支持条件按本次联网核验日期记录，部署时仍需固定 release。
- `sync_status`：`PAPERS.md` 增 GPTQ/AWQ/outlier 对照、KIVI/TurboQuant 和 vLLM serving 评测；`GLOSSARY_EN_ZH.md` 补 KIVI、TurboQuant、QJL、active/cold-page KV 和 FP4 格式边界；`INTERVIEW_BANK.md` 增 GPTQ/AWQ/SmoothQuant/TurboQuant 方案比较题；`EXERCISES.md` 增 KIVI 生命周期推演和多路线同条件 benchmark 设计题。第六册目录已含第七章，不需改目录。
- `validation`：检查 §1—§19 连续、章节/研究笔记围栏均闭合、两文件本地链接目标存在；demo 标准输出逐字一致，11 个数值/输入边界通过；复核新增表格字段、关键公式和 cold-page 支持条件；`git diff --check` 通过。未跑 GPU、真实模型量化、论文 benchmark、目标 serving、外部公式渲染或 PDF 构建。
- `open_risks`：没有在 GPU、真实模型或目标服务上复现 GPTQ/AWQ/SmoothQuant/KIVI/TurboQuant/SparseGPT；论文指标、vLLM 工程实现结果和 NVIDIA 当前文档属于不同证据层次，均不能外推为目标模型质量、kernel 速度、容量或兼容性。尚无本项目目标 GPU/模型/线上 workload；TensorRT-LLM 与 vLLM 支持随版本变化。
- `status`：第六册第七章完成资料驱动精修、原知识点覆盖、代表性方法/改进假设对照、配套同步和声明范围内代码/公式/链接验收；全书 goal 继续，下一章为第六册第八章。

### 第六册第六章：批处理与调度资料驱动精修验收（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 全文复读 `book-06-llm-deployment/chapters/06-批处理与调度.md` §1—§17；保留 batching 类型、请求生命周期、Little 定律、队列、prefill/decode、token/KV 预算、priority、多租户、指标、调度 demo、抢占/取消/重试及故障复盘。逐节覆盖、方法证据与边界见 [`research/batching-scheduling-depth-review-2026-10-26.md`](research/batching-scheduling-depth-review-2026-10-26.md)。
- `coverage_map`：研究记录映射原 §1—§17。原有 static/dynamic/continuous batching、chunked prefill、P/D 取舍、队头阻塞、公平性、租户隔离、指标和生命周期均保留；新增 FastServe、Sarathi-Serve、DistServe 和 PPD 的不同调度层次，没有把它们混为一类或删去原问题。
- `changes`：用 M/M/1 教学例解释到达率接近服务率时延迟曲线变陡，并明确不是 LLM 容量模型。比较 Orca iteration 重排、FastServe token 级抢占、Sarathi chunked prefill、DistServe P/D 池分离，以及 PPD 对多轮 append-prefill 的动态路由；使用 NSDI 2026 FastServe 正式版 6.1×，显式标出早期 arXiv 版 31.4×/17.9×不可混用。补 vLLM/SGLang priority 数值方向、队列准入、token/sequence budget、抢占与 KV retraction 的语义差异；更正 `\eta` 公式符号；扩充 Gini 权重反例。Demo 改为预留最大输出 KV 逻辑 token 上界，并拒绝不适合单轮 prefill 的过长 prompt。
- `evidence`：研究笔记记录 Orca、FastServe NSDI 2026、Sarathi-Serve、DistServe、PPD 的实际问题、机制、论文指标和限制；核对 vLLM scheduler/chunked-prefill、SGLang server arguments、TensorRT-LLM 当前 PyExecutor、Hugging Face TGI/Transformers 文档。所有倍率绑定作者模型、GPU、流量、SLO 和 baseline；FastServe 数字按 arXiv/正式版分开。
- `formula_status`：复核 Little 定律和 M/M/1 示例：`μ=10, λ=5` 时 `W=0.2s, L=1`；`λ=9` 时 `W=1s, L=9`。复核两租户 Gini：`G(80,20)=0.3`，`G(50,50)=0`，并说明按 4:1 entitlement 判断需看归一化服务份额。检查其它资源账本和 priority 公式的变量边界；未做外部公式渲染。
- `code_status`：提取并运行 §13 标准库 continuous-batching demo，stdout 与 fenced `text` 完全一致；把 prefill budget 降到 1,000 后，长 prompt validation 正确拒绝，避免 waiting 请求无期限滞留。KV budget 按逻辑 token reservation 近似，不是物理 block allocator。
- `link_status`：核对 Orca、FastServe 正式/早期版本、Sarathi-Serve、DistServe、PPD、vLLM scheduler/metrics/engine arguments、SGLang scheduler arguments、TGI 和 TensorRT-LLM 当前 PyTorch architecture。旧 TensorRT Build Workflow 保留为可访问的 legacy 文档，正文标出不代表当前 backend 路径。
- `sync_status`：`PAPERS.md` 补 FastServe/PPD 与当前 vLLM/SGLang scheduler 参数入口；`GLOSSARY_EN_ZH.md` 补 Skip-Join MLFQ、Token-Level Preemption、P/D Disaggregation、Append-Prefill、M/M/1 Queue；`INTERVIEW_BANK.md` 增代表调度方案比较题；`EXERCISES.md` 增方法证伪 benchmark 与 Little/M/M/1 手算题。第六册目录已有第六章入口。
- `validation`：检查 §1—§17 顺序、58 个 code/math/text fence marker 成对、26 个 math blocks 括号计数平衡、6 张表格列数一致、链接和目录；运行 demo 及超长 prefill 边界；`git diff --check` 通过。未运行真实 runtime/GPU/模型、论文 benchmark、外部公式渲染或 PDF 构建。
- `open_risks`：没有目标模型、GPU、集群、线上流量或调度 fairness 实测；没有测 token-level preemption state migration、P/D state transfer、PPD append-prefill 或队列稳定容量。框架优先级方向与队列参数属于动态版本契约，部署时仍需固定 release 并回放真实租户/API 请求。
- `status`：第六册第六章完成资料驱动精修、原知识点覆盖、代表研究对照、配套同步和声明范围内代码/公式/链接验收；全书 goal 继续，下一章为第六册第七章。

### 第六册第五章：推理引擎资料驱动精修验收（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 全文复读 `book-06-llm-deployment/chapters/05-推理引擎.md` §5.1—§5.20；保留引擎边界、请求生命周期、框架介绍、选型矩阵、模型协议、指标/压测、教学 demo、迁移案例、常见误解及原有资料入口。逐节知识点去向、研究证据与限制见 [`research/inference-engine-depth-review-2026-10-26.md`](research/inference-engine-depth-review-2026-10-26.md)。
- `coverage_map`：研究笔记映射原稿 §5.1—§5.20。Transformers、vLLM、SGLang、TensorRT-LLM、LMDeploy、llama.cpp、TGI 均保留并按当前公开路径补充；协议兼容、框架选型、压测和迁移知识均保留。没有因增加论文比较而删除已有的主要问题。
- `changes`：把“有哪些引擎”改为按执行路径、调度机制、模型/backend/设备组合、版本状态、适用 workload 与验证动作逐项说明。代表研究比较 Orca iteration-level scheduling/selective batching、PagedAttention/vLLM、Sarathi-Serve chunked prefill/stall-free schedule、SGLang RadixAttention/压缩 FSM、vAttention CUDA VMM 各自解决的瓶颈，并把作者数据与模型、GPU、baseline、SLO/指标绑定。增补 Transformers `serve`、vLLM V1、TensorRT-LLM 当前 PyTorch-only backend、LMDeploy TurboMind/PyTorch 双路径、llama.cpp continuous batching、TGI maintenance mode 等版本事实。SGLang session-aware cache 补充 session ID 仅登记缓存引用、不自动拼接上下文、引用不构成 hard pin 的边界。benchmark 补开环/闭环、request rate/burstiness/concurrency、cache 冷热、重复 prefix 和 goodput/TPOT/ITL 分母。
- `evidence`：研究笔记逐项记录五篇代表工作的问题、方法、论文测量及可迁移性限制，并核对 Transformers、vLLM、SGLang、TensorRT-LLM、LMDeploy、llama.cpp、TGI 官方文档。Orca 36.9×、vLLM 2–4×、Sarathi-Serve 2.6×/3.7×、SGLang 最高 6.4×、vAttention 最高 1.23×均只作原论文设置下的研究证据，不横向排名或外推。
- `formula_status`：检查 §5.14 的 SLO goodput 定义，分子明确要求完成、质量通过且指定 latency `L_i` 满足门槛，分母为测量 window 的 wall time；要求把 `L_i` 具体标成 TTFT、TPOT、E2E 或业务 deadline。章节其它延迟和吞吐公式未改写，未做外部公式渲染。
- `code_status`：提取运行 §5.15 零依赖 Python 选型 demo，实际 stdout 与 fenced `text` 完全一致；额外将 throughput 门槛设为不可达，确认空候选边界正常。所有数字均是教学输入，不是框架 benchmark；未运行真实 serving engine、GPU 或模型。
- `link_status`：核对原论文、Transformers v5.6.0、vLLM benchmark/V1、SGLang session-aware cache 与 eviction、TensorRT-LLM migration/PyExecutor、LMDeploy support matrix、llama.cpp server、TGI maintenance notice 等链接支持范围。本章无新增本地相对链接；官方动态文档的架构与状态按 2026-10-26 记录，部署时需要固定目标 release。
- `sync_status`：`PAPERS.md` 增补 SGLang session-aware Radix Cache 与各框架当前服务文档入口；`GLOSSARY_EN_ZH.md` 增 Session-Aware Radix Cache；`INTERVIEW_BANK.md` 新增框架版本变更和 serving benchmark 设计题；`EXERCISES.md` 增代表研究比较及开环/闭环压测设计题。第六册目录已有第五章入口。
- `validation`：检查 §5.1—§5.20 标题顺序、fenced block 配对、公式定界符、表格列数、目录映射及正文/来源链接；运行 demo 与 no-candidate 边界；`git diff --check` 通过。未运行真实 runtime、GPU/模型、论文 benchmark、外部公式渲染或 PDF 构建。
- `open_risks`：没有目标 GPU、模型 revision、集群、生产流量或线上 trace；没有测引擎容量、SLO、成本、质量或跨租户 cache 隔离。论文倍率不代表当前 release 性能；框架官方文档确认功能路径不代表目标模型/backend/硬件兼容。TGI 维护状态、TensorRT-LLM backend 变更和当前 serving API 均需部署时复核。
- `status`：第六册第五章完成来源驱动精修、原知识点覆盖、代表研究对照、配套同步和声明范围内代码/链接验收；全书 goal 继续，下一章为第六册第六章。

### 第六册第四章：解码策略与生成控制资料驱动精修验收（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 全文复读 `book-06-llm-deployment/chapters/04-解码策略与生成控制.md` 的 §4.1—§4.19。保留 logits、greedy、beam、sampling、temperature、top-k/top-p/min-p、惩罚项、停止语义、结构化输出、reasoning effort、speculative decoding、复现、调参、教学 demo 和工具任务；原稿问题与本轮取舍见[研究记录](research/decoding-generation-control-depth-review-2026-10-26.md)。
- `coverage_map`：研究记录逐节映射旧稿 §4.1—§4.19。核心机制和边界补深；eta/epsilon 截断、API 版本语义等支持性内容保留简要入口；没有因增补专题而删去原有主要知识点。
- `changes`：比较 top-p、locally typical、min-p 的候选规则，并给出同分布手算；并列呈现 min-p 原论文主张和 2025 年批判性复核，不把争议写成定论。结构化输出加入 grammar state 到 token mask 的状态轨迹，比较 XGrammar 的 mask 预处理/缓存与 tokenizer-CFG sound alignment 的解决问题和不同计时口径。Speculative decoding 增独立 draft、prompt lookup、Medusa、EAGLE、checkpoint MTP 路线表；用 p/q 数值演示接受与残差校正，说明精确条件下的分布保持与同 seed 重放不是一回事。补入高 batch vLLM 评测的 workload、verification 成本及迁移限制；修正期望提交长度中联合接受前缀概率的定义和吞吐分子。
- `evidence`：研究记录逐源说明 nucleus/typical/min-p、XGrammar、Flexible and Efficient Grammar-Constrained Decoding、speculative sampling、Medusa、EAGLE、官方接口和 vLLM 评测分别解决什么、实验设置、证据范围及不能外推的边界。min-p 争议保持双方归因；论文性能数值均绑定作者模型、任务、硬件、batch、版本或评价协议。
- `formula_status`：复算 toy 分布熵 `H≈1.280` nat、typical 距离排序、top-p/min-p/typical 候选集合；手算 p=(0.50,0.30,0.20)、q=(0.20,0.50,0.30) 的拒绝残差并验证最终边际仍为 p。确认 `E[N_committed]=1+Σ_j P(前 j 个草稿 token 均接受)`，条件接受率写成前缀连乘；将零残差规定为不应进入拒绝分支的 invariant failure。未做外部公式渲染。
- `code_status`：提取运行 §4.15 标准库 Python demo，确定性输出逐项与 fenced text 一致；该 demo 是 logits 控制教学示例，不实现真实模型、grammar runtime 或 speculative decoding。
- `link_status`：打开核对本章新增论文和官方文档；`Truncation Sampling as Language Model Desmoothing` 确认提供 eta-sampling 的熵相关阈值背景。第六册目录已有第四章及第五章入口。动态 HF/vLLM/OpenAI/Gemini 文档只代表 2026-10-26 所见内容，实施时仍须固定实际 release/API。
- `sync_status`：`PAPERS.md` 增加采样、结构化生成与推测解码的代表研究入口；`GLOSSARY_EN_ZH.md` 补 locally typical、min-p、grammar-constrained decoding、acceptance length 与 residual distribution；`INTERVIEW_BANK.md` 加入截断策略研究证据与实验边界对照题；`EXERCISES.md` 加入候选集手算及 speculative 接受/残差分布推导。
- `validation`：检查 §4.1—§4.19 标题顺序、fenced blocks 成对、数学符号与定义、公式和数值例、Markdown 表格列数、本册目录映射和本章新增来源；demo 输出核对通过；`git diff --check` 通过。未运行真实模型、GPU、目标 serving runtime、生产 trace、人评复现、PDF 构建或外部公式渲染。
- `open_risks`：论文收益与 runtime 接口不能外推到目标模型、采样配置、grammar、硬件、batch 和服务版本；仍需在目标 workload 对照无 speculative/无 grammar baseline 测质量、格式成功率、TTFT/TPOT、吞吐、显存和失败恢复。本章的纸上轨迹验证教学算法，不验证目标 runtime 的随机数、浮点和 cache 状态实现。
- `status`：第六册第四章完成资料驱动精修、知识点覆盖、配套同步和声明范围内验收；全书 goal 继续，下一章为第六册第五章。

### 第六册第三章：KV Cache 与内存管理资料驱动精修验收（2026-10-26）

- scope：按当前 WRITING_SPEC.md 复读 book-06-llm-deployment/chapters/03-kv-cache与内存管理.md §3.1—§3.17。保留 KV 生命周期、MHA/MQA/GQA/MLA、显存账本、分页、量化、前缀复用、长上下文策略、OOM 排查、多租户、安全、教学 demo 和练习；原知识点去向见 [研究笔记](research/kv-cache-memory-management-depth-review-2026-10-26.md)。
- coverage_map：研究笔记逐节映射原稿内容。核心问题加深了逻辑 token 到物理 block 的寻址、共享尾块写时复制、PagedAttention/vAttention、KIVI 的 K/V 非对称量化、prefix-cache 状态生命周期与跨租户侧信道、H₂O/StreamingLLM 的保留差异。其它已涵盖的公式、策略、排障和容量问题均保留；没有因增加专题删除原有知识点。
- changes：加入分页地址换算和两个请求分叉的 COW 状态轨迹；对照 block table 与 CUDA VMM 两条内存路线。解释 KIVI 的 K per-channel、V per-token、group/residual 机制及论文容量实验；加入 vLLM hash/refcount/free-queue 与 TensorRT-LLM radix-tree/LRU 的实现对照；演算 prefix hit 到释放淘汰的过程；比较 H₂O heavy hitters 和 StreamingLLM sink+recent 的保留目标；修正 demo 的输入校验顺序与非法请求处理。
- evidence：研究笔记逐项记录 PagedAttention、vAttention、KIVI、H₂O、StreamingLLM 原论文，KV 管理综述、NDSS 2025 cache-sharing 论文、vLLM/TensorRT-LLM/Hugging Face 官方资料的支持点和适用边界。论文吞吐、容量和质量数字均限定在作者模型、任务、硬件、kernel、cache budget 与基线，不作跨研究排名。
- formula_status：核对 block 数及尾浪费，手算 token i=9、block size 4、物理 block 11 对应 slot 45；复核 KIVI 教学轨迹、论文分组参数、显存比例和 demo 的容量算术。检查本章公式无禁用 operatorname 写法；未做外部数学渲染。
- code_status：从 Markdown 提取并执行 §3.13 标准库 Python demo。正常输出与正文相符；零 block size、预留小于活跃 token、bool token 数、非 mapping 请求、重复 ID 和空 batch 等 6 类边界均按预期拒绝。仅验证教学账本和输入契约，不代表真实 attention、CUDA allocator、runtime 或线上负载。
- link_status：打开核对本章新增论文、综述、官方文档及安全研究链接；版本化 vLLM 文档固定在 v0.25.0，latest/current 页面仅代表 2026-10-26 的公开文档，实际部署须按 release 复核。第六册目录已有章节入口。
- sync_status：PAPERS.md 增加 vAttention、H₂O/StreamingLLM、KV 综述及 cache-sharing 研究；GLOSSARY_EN_ZH.md 增 Cache Salt、Copy-on-Write KV Block、KV Cache Eviction、Attention Sink、Heavy-Hitter Token、KIVI；INTERVIEW_BANK.md 和 EXERCISES.md 增加代表方案比较与状态推演。
- validation：检查 17 个主节顺序、代码/数学围栏配对、公式和表格算术、相对链接目标；章节 demo 运行通过；全局 git diff --check 通过。未运行真实 GPU/runtime、论文 benchmark、外部公式渲染或 PDF 构建。
- open_risks：无目标 GPU、模型、Serving workload 或生产 trace 实测；论文数字不外推到其它环境；不能据论文推定当前 runtime 支持 KIVI 2-bit；cache salt 不能代替鉴权、加密、日志策略或删除验收。
- status：第六册第三章完成本轮资料驱动精修与范围内验证，全书 goal 继续；下一章为第六册第四章。

### 第六册第二章：推理基础资料驱动精修验收（2026-10-26）

- `scope`：从头复读 `book-06-llm-deployment/chapters/02-推理基础.md` §2.1—§2.19 并保留原有知识主线；精修未删节原有条件概率、prefill/decode、KV、指标、batch、排队、cache、workload、误区、trace 和容量案例。coverage map 见 [`research/inference-basics-depth-review-2026-10-26.md`](research/inference-basics-depth-review-2026-10-26.md)。
- `changes`：补全 attention 公式的单层/全模型边界；新增 `2BP` 权重 FLOPs、KV 读取和算术强度推导及 7B 手算表，解释 batch 摊薄权重、长上下文放大 KV 流量。比较 FlashAttention、Orca、Sarathi-Serve、Splitwise 解决的不同瓶颈、代价和验证条件。修正 client TTFT 的首内容事件时间戳、TPOT/ITL 聚合权重、E2E 的统一 clock 与 event 数、percentile 的 singleton/endpoint 定义。Demo 的阶段强度改标为单层 attention proxy、同时到达的 QPS 改为 cohort rate、cache 总和改称全重叠 KV envelope；补 bool/non-finite percentile 边界。压测步骤增加 load model、warm-up、cache state、seed、window、failures 和 profiling overhead。
- `evidence`：研究记录了 LLM inference survey taxonomy、FlashAttention IO method、Roofline/Nsight、Orca、Sarathi-Serve、Splitwise，以及 vLLM/TensorRT-LLM 当前 benchmark 文档的具体支持和版本边界。论文结果绑定作者 workload、模型、硬件和 SLO，不跨论文排名。
- `formula_status`：复核 5 个 decode roofline case（AI 1.007/56.020/98.889/1.214/14.155 FLOPs/B，理想下界 8.771/10.092/15.076/9.421/51.700 ms）；prefill attention FLOPs 占比为 7.12%/23.48%/55.10%。指标例的 request-weighted TPOT 为 15 ms，合并 102 个 ITL samples 的 event-weighted mean 为 19.80 ms。所有 roofline 数值均是理想教学估算。
- `code_status`：提取并执行 §2.14 的零依赖 Python demo；8 条 request 输出与正文示例一致，finite cohort rate 为 3.22 req/s、241.47 output tokens/s，平均 TTFT 104.6 ms、P95 latency 2147.6 ms、P95 TPOT 16.4 ms；全重叠 token envelope 为 10,200 tokens / 1,275 MiB。内置异常输入覆盖零 input、重复 ID、范围外/bool/非有限 percentile 和非有限 sample，并检查 singleton `p=1`。仅证明 toy demo 的输出一致性，不证明真实 serving 性能。
- `link_status`：打开核对 inference survey、FlashAttention、Splitwise、Sarathi-Serve、Orca、vLLM benchmark CLI、TensorRT-LLM benchmark guide 和 Nsight Compute；动态 `main` 文档需按实际 runtime 固定 revision。正文有 26 个外部链接，未发现失效的本地 Markdown 目标（本章没有本地相对链接）。
- `sync_status`：`PAPERS.md` 补入 Splitwise 与 Roofline；`GLOSSARY_EN_ZH.md` 加 Arithmetic Intensity、Roofline Ridge Point、Open-loop Load、Closed-loop Load。`INTERVIEW_BANK.md` 和 `EXERCISES.md` 不新增重复题：已有推理系统、KV/bandwidth、压测和 accelerator audit 覆盖；本章已提供可复算示例与运行 demo。第六册目录本来已有第二章入口。
- `validation`：检查 §2.1—§2.19 连续、§2.8.1—§2.8.4 连续、94 个 tilde fence marker 配对、41 个 math blocks 的括号平衡、本地 Markdown 链接目标、公式/表格算术；运行本地 demo；全局 `git diff --check` 通过。未运行 GPU kernel、目标模型、线上 trace、论文 benchmark、外部公式渲染或 PDF 构建。
- `open_risks`：7B dense BF16 表假设单卡权重、相同 `T_ctx`、指定参数形状与理论峰值；真实 kernel 的因果 FLOPs 常数、MoE active weights、cache hierarchy、L2 reuse、HBM counters、分布式通信、调度与 runtime overhead 会改变瓶颈和时间。未对任何生产集群形成容量/成本/延迟结论。
- `status`：第六册第二章完成资料驱动精修、原知识点覆盖、方法比较、配套索引和声明范围内公式/代码/链接审查；全书 goal 继续进行，下一份正文为第六册第三章。

### 第六册第一章：部署总览资料驱动精修验收（2026-10-26）

- `scope`：承接当前进度直接精修 `book-06-llm-deployment/chapters/01-部署总览.md` 的 §1.0—§1.19；此前章节不重做。原有服务对象、离线/在线、请求时间、容量、部署拓扑、引擎、模板、KV、请求路径、观测、质量安全、发布回滚、toy demo、RAG 案例、来源和总结均保留；逐项覆盖图见 [`research/inference-deployment-overview-depth-review-2026-10-26.md`](research/inference-deployment-overview-depth-review-2026-10-26.md)。
- `changes`：从研究中补齐 ITL 与 request-level TPOT 的事件/请求分母；加入 token bundling 数值例和长停顿反例。容量段增加 request SLO goodput、attainment、到达率/并发负载差异及论文指标偏差；区分持续 SLO 容量和峰值 tokens/s。代表方案表比较 Orca iteration scheduling、Sarathi chunked prefill、DistServe P/D disaggregation 的问题、机制和限制；补观测信号诊断路径、Prometheus quantile 聚合规则、bucket 成本、TP/PP per-rank KV 账本以及 prefix cache salt 的隔离/删除边界。toy trace 改为输出有限成功请求 cohort rate 和活跃 KV token envelope，并标清不能代表稳态 QPS、实配 KV 或生产容量。
- `evidence`：研究笔记记录 inference systems survey 分类、Orca OSDI、PagedAttention、Sarathi-Serve、DistServe、SLO/smooth-goodput 论文与当前 vLLM metrics、SGLang bench_serving、OpenMetrics、TensorRT-LLM disaggregation、vLLM prefix caching、TGI 官方文档的具体支持点和版本边界。论文提升均绑定作者模型、硬件、负载、基线和 SLO，不作跨研究排名。
- `formula_status`：复核 `60/4=15 ms` request TPOT 与 `[30,30] ms` ITL；`(19×20+500)/20=44 ms` 仍含 500 ms 停顿；goodput `60/10=6 req/s` 与 `90/10=9 req/s`；容量示例结果分别为 5、4、压力包络 7 副本；toy trace 为 3.03 s、1.32 req/s、87.46 output tokens/s、230 ms avg TTFT、2,830 ms P95 latency、20 ms P95 eligible TPOT、3,885 peak live tokens 和 485.62 MiB 上界。
- `code_status`：提取并执行 §1.14 标准库 Python demo，输出与正文一致；编译通过。非法输入测试覆盖负队列时间、bool token 数、成功请求零输出和重复 trace ID。仅验证 toy trace 统计，不代表 GPU/runtime 或服务容量。
- `link_status`：本地 Markdown 链接 7 个目标均存在；新加的论文、官方指标/benchmark、标准、框架和厂商方案引用已打开核对具体支持范围。动态 `main`/`latest` 文档按记录日期使用，后续须按部署版本复核。
- `sync_status`：`PAPERS.md` 新增 Orca/Sarathi/DistServe、inference survey 和 SLO/goodput 研究；`GLOSSARY_EN_ZH.md` 增 ITL、Request-level TPOT、SLO Goodput、Smooth Goodput；`INTERVIEW_BANK.md` 增 raw tokens/s 与 goodput 面试问答；`EXERCISES.md` 增 goodput 练习并把相邻部署练习编号整理为 35E45—35E49。第六册目录已有本章入口。
- `validation`：检查 §1.0—§1.19 连续、48 个 tilde fence marker 配对、本地链接目标、公式算术和 table/source statement；`git diff --check` 通过。运行 §1.14 code；未运行真实 GPU、模型、线上请求、论文 benchmark、PDF 或外部数学渲染。
- `open_risks`：没有目标集群、生产流量或真实模型数据；章节容量算式是 workload/SLO 规划示例。论文性能数字不可跨条件排名；vLLM/SGLang/TensorRT-LLM/TGI 动态文档需依实际版本复查。章节的 P/D、KV、scheduler、缓存和生产 rollout 细节由第六册后续章节继续展开。
- `status`：第六册第一章完成来源驱动精修、旧问题覆盖、方法比较、配套同步和声明范围内代码/公式/链接验收；全书 goal 继续，下一章为第六册第二章。

### 第四册第十九章：前沿架构与模型更新深度精修验收（2026-10-26）

- `scope`：从头复读 `book-04-llm-encyclopedia/chapters/19-frontier-architecture-updates.md` 全文。原有 AttnRes、KDA、CSA/HCA、mHC 和全部厂商模型/产品快照条目均保留原位；旧知识点去向与支持性条目的专题入口见 `research/encyclopedia-frontier-architecture-depth-review-2026-10-26.md` 的 coverage map。
- `changes`：更正章名以同时覆盖架构方法与模型接口快照；新增按“深度聚合、递归序列状态、显式 KV 压缩/筛选、残差流混合、value 路由、n-gram 查表”划分的问题地图。补 DenseFormer→DCA/AttnRes 的前序关系及实验边界，避免把 AttnRes 写成首个深度可学习聚合；补 KDA 的 128K 长上下文分数和非全面胜出范围；补 V4-Pro/Flash 相对 V3.2 的 1M FLOPs/KV 报告值并限定为全系统联合结果；补 HC→mHC 机制关系和 `n=4`、6.7% wall-time 的论文实验条件。加入 16 个 token→4 个压缩条目→top-2 的教学推演，拆分压缩信息损失与索引漏选，并加入按故障选评测的表格。
- `evidence`：研究笔记记录综述 [`Efficient Transformers`](https://arxiv.org/abs/2009.06732)、循环模型综述 [`On the Resurgence of Recurrent Models`](https://arxiv.org/abs/2402.08132)、DenseFormer、DeepCrossAttention、Attention Residuals、Kimi Linear、Hyper-Connections、mHC 和 DeepSeek V4 原始资料的已读内容、主张与适用边界。论文公开数据分别绑定 DCA 语言建模设置、Kimi Linear 128K benchmark、V4 1M FLOPs/KV baseline 和 mHC expansion rate `n=4`；不作跨论文排名。
- `formula_status`：未新增新公式；静态核对现有 AttnRes 深度聚合、KDA 递推、CSA/HCA 定义和双随机矩阵说明与第二十一册第 75—78 章的主讲位置一致。手算例子明确为教学参数，不冒充 DeepSeek 配置。
- `code_status`：本章未增删代码；原有协议/流程代码块未执行。第二十一册第 75—78 章中的教学代码本轮未改动、未运行；本轮新增推演以表格和数字说明机制。
- `link_status`：本章新增 DenseFormer、DCA、HC、mHC、Kimi Linear 和 DeepSeek V4 来源链接及第 75—78 章入口；研究笔记内相对 Markdown 目标全部存在。第四册目录第十九章路径映射有效。
- `sync_status`：`PAPERS.md` 增加高效注意力/循环综述和深度聚合/多路残差前序论文入口；`GLOSSARY_EN_ZH.md` 增加 DenseFormer、DCA、HC；`INTERVIEW_BANK.md` 增加深度聚合与多路残差机制对比题。`EXERCISES.md` 已有对应实现与消融练习，本轮不重复复制。
- `validation`：`git diff --check` 通过；相对 Markdown 链接目标存在；本章 2 个 fenced code marker 成对，研究笔记无未闭合代码围栏。复核原有厂商条目仍全部在位。未运行模型、Python 教学代码、GPU profiling、外部公式渲染或 PDF 构建。
- `open_risks`：各方法未在统一参数/FLOPs、训练 token、数据、硬件和 Serving 协议下做独立复现。模型接口、价格、effort 与榜单数字仍是各自来源日期的快照，本轮未把所有产品页作为实时 API 手册逐页更新；遇到版本变化时需沿该条研究笔记重新核验。CSA/HCA、KDA、DCA 和 mHC 的性能数字均是作者报告，不等于本地结果。
- `status`：第四册第十九份正文完成本轮来源驱动的架构问题对照和静态验收；第四册正文目录到此结束。全书 goal 继续进行，下一份为第五册第一份正文。

### 第四册第十八章：语音大模型深度精修验收（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 复读 `book-04-llm-encyclopedia/chapters/18-语音大模型.md` 的 18.0—18.23，保留原有波形/Mel、ASR/WER/CER、Whisper、speech token/neural codec、TTS/韵律、声纹/克隆、VAD/endpoint、barge-in、多模态语音 Agent、安全、成本账本代码及六项练习。核心问题和旧主题去向见 `research/encyclopedia-speech-llm-depth-review-2026-10-26.md` 的 `coverage_map`。
- `changes`：区分 wav2vec 2.0 latent 对比学习、HuBERT 聚类伪标签与 Whisper 互联网弱监督多任务训练，说明标注量、语种覆盖和标签噪声边界；新增 CTC blank/路径求和公式、RNN-T 时间×输出前缀格点、attention seq2seq 流式缓冲的机制比较，引用同 encoder/多语种 CTC/RNN-T 实验但限定其协议。TTS 加 Tacotron 2、FastSpeech 2、VITS、VALL-E/VALL-E 2 的问题、生成机制、特定实验收益和失败边界，不跨论文排名 MOS。speech token 扩 semantic/acoustic/混合目标取舍，说明 codec code 总数与串行模型步数不同；端到端语音加 Moshi、Qwen3-Omni、Qwen3.5-Omni/ARIA 的路线比较，区分 full-duplex、流式输出与双轨同步。评估补 MOS、TTFT/TTFC/首包/首音和 ASR-as-TTS-proxy 的测量边界；新增方案选型练习并修正旧 ASR 公式的 `y_{<u}` 下标写法。
- `evidence`：研究笔记覆盖综述和原始论文/技术报告，包括 ACL 2025 SpeechLM survey、离散语音 token review、wav2vec 2.0、HuBERT、Whisper、CTC/RNN-T 对照、Tacotron 2、FastSpeech 2、VITS、VALL-E/2、Moshi、Qwen3-Omni/Qwen3.5-Omni。数字均绑定对应数据集、样本、硬件、听评和作者报告设置；ARIA 细节交叉链接至第二十一册第 93 章和 `research/model-update-2026-09/qwen3.5-omni-source-notes.md`。
- `formula_status`：静态核对 CTC 路径和 blank 折叠、`y_{1:u-1}` 自回归上下文、codec `DqK` 码数与 AR 步数区分、既有 WER/CER/RTF/任务成功率定义；检查 fenced math 计数与代码围栏配对。未使用外部公式渲染器。
- `code_status`：原有 18.20 Python ledger 未改写，本轮未运行 Python 示例或测试；此前 2026-10-23 旧轮次代码运行记录不作为本轮执行证据。
- `link_status`：章节新增原论文、ACL Anthology、PMLR 来源链接；新增第二十一册第 93 章相对入口，第四册目录映射维持有效。
- `sync_status`：`PAPERS.md` 补 SpeechLM/离散 token 综述、CTC/RNN-T 对照、VALL-E 2 和 Moshi；`INTERVIEW_BANK.md` 深化语音架构、Whisper 和 codec LM 问答并新增 ASR 目标选择题；`GLOSSARY_EN_ZH.md` 补 CTC、RNN-T、语义/声学 token、TTFA/TTFC/RTF、ARIA。练习新增于本章，无需复制到 `EXERCISES.md`。
- `validation`：核对 18.0—18.23 标题连续、fenced code/math 42 个 marker 成对、14 个 math block、目录映射和相对链接；`git diff --check` 通过。未运行章节 Python、ASR/TTS 模型、PDF 构建或公式渲染。
- `open_risks`：未用目标语言、噪声、设备或并发复现论文；未实测真实流式首音/端点、MOS、speaker verification、voice cloning 授权/水印、语音 Agent 工具安全。不同论文首包、MOS、WER、硬件和数据分布不可直接排名；Qwen3.5 数字为作者内部 serving 报告，不是本机或用户端 SLO。
- `status`：第四册第十八份正文完成本轮来源驱动的深度精修与静态验收；全书 goal 继续进行，下一份为第四册第十九份正文。

### 第四册第十七章：图像生成与视频生成深度精修验收（2026-10-26）

- `scope`：按当前 `WRITING_SPEC.md` 复读并精修 `book-04-llm-encyclopedia/chapters/17-图像生成与视频生成.md` 的 17.0—17.23；原有扩散公式/参数化、U-Net/DiT/VAE、DALL·E、自回归路线、CFG、图像编辑、视频时空 token/一致性、Sora/world model、FID/FVD、安全、账本代码和六项练习都保留。新增知识点与旧主题解释层级见 `research/encyclopedia-image-video-generation-depth-review-2026-10-26.md` 的 `coverage_map`。
- `changes`：在图像路线处比较 DALL·E 1、DALL·E 2、Imagen、Latent Diffusion 和 DiT 各自改变的瓶颈及证据范围；推导 `v` prediction 与 `x0/epsilon` 的逆变换；将 DDIM、DPM-Solver 与 Progressive Distillation 区分为无需重训改采样路径和额外训练蒸馏。图像编辑新增 SDEdit/InstructPix2Pix/ControlNet 的输入信号对照；视频生成新增 Make-A-Video、Imagen Video、CogVideoX、Wan 2.1 的数据、时空压缩、推理成本和失效边界比较。图像评估补 DrawBench/GenEval/T2I-CompBench，FID 补特征分布与类别偏置案例，视频评估补 VBench/VBench-2.0 和 Beyond FVD 的时序盲区；FID/FVD 矩阵根式改为对称半正定形式。Sora 小节更新到 2024 技术报告、2025 Sora 2 System Card 和官方页面注明的 2026-04-26 产品状态。将 toy 代码中的 `cost_per_accepted` 改名为 `latency_per_accepted_ms`，避免把毫秒代理称作货币成本，并加入手算结果及被省略的 pipeline 开销说明；新增一项代表路线选型练习。
- `evidence`：研究笔记 `research/encyclopedia-image-video-generation-depth-review-2026-10-26.md` 记录问题图谱、方法比较、主张/来源、限制与 `coverage_map`。核对 DALL·E 2、Imagen、DiT、SDEdit、InstructPix2Pix、DDIM、DPM-Solver、Progressive Distillation、Make-A-Video、Imagen Video、CogVideoX、Wan 2.1、GenEval、T2I-CompBench、VBench/VBench-2.0、FID/FVD 指标研究的原始论文摘要及支持段落，并查 OpenAI Sora 官方技术报告/System Card/当前页面、Hugging Face Diffusers 官方 scheduler 文档。数字仅绑定各论文 benchmark、参数化、样本与硬件配置，不作跨论文排名。
- `formula_status`：静态核对 `v=aε-sx0` 变换逆式、CFG 公式与 `w` 约定、DiT patch 数和 attention pair 数、FID/FVD 对称协方差根式、编辑 mask 分母、成本概率分母及其他原有定义域。示例 `\bar{α}=0.36` 得 `a=0.6,s=0.8`；32×32 latent 的 patch 从 8 改 4 后为 16→64 token、pair 数 256→4096。
- `code_status`：保留 17.20 Python 账本；重命名指标并静态核对 512/8 latent cells 为 4,096、30×40ms/0.60 为 2,000ms/accepted、12×40ms/0.20 为 2,400ms/accepted。按当前执行约束，本轮未运行 Python 示例或测试；代码记录不等于实机 latency 或货币成本。
- `link_status`：本章增补来源链接，修订阅读清单并加入 Diffusers Scheduler 官方文档；Sora 相关页面为动态状态，文中注明核验日期与产品限制。第 17 章目录入口原已有效，无文件拆分或新相对导航。
- `sync_status`：`PAPERS.md` 多模态区增补 Imagen、Progressive Distillation、SDEdit/InstructPix2Pix、Make-A-Video、Imagen Video、CogVideoX、Wan、FID/FVD 局限及组合/视频 benchmark；`INTERVIEW_BANK.md` 改写图像路线题、采样器题、生成审计题与视频路线题，加入各方案的问题、机制、成本和不可比边界。原六项章节练习保留并新增第七项方案选择练习。
- `validation`：检查 17.0—17.23 标题连续、文中公式和代码围栏、链接文字与主要目标；执行 `git diff --check`。未执行可执行 Python 示例、PDF 构建或公式渲染。
- `open_risks`：未在本地训练或运行图像/视频模型，未实测真实 CFG/scheduler、VAE、长视频 latency、显存、人工偏好或安全分类；各论文结果协议不一致，正文明确不横向排名。Sora/Diffusers 动态页面后续需按目标日期/版本复核。17.20 是教学账本代理，不是生产容量或成本模型。
- `status`：第四册第十七份正文按当前 `WRITING_SPEC.md` 完成本轮精修与静态验收；全书 goal 继续，下一份为第四册第十八份正文。

### 第四册第十六章：多模态学习深度精修验收（2026-10-26）

- `scope`：从头到尾复读 `book-04-llm-encyclopedia/chapters/16-多模态学习.md` 的 16.0—16.31；修订前 1,372 行，修订后 1,424 行。保留输入表示、质量分层、CLIP/SigLIP、ViT、VLM connector、多模态 SFT、DDPM、CFG/ControlNet、latent diffusion、视频/FVD、waveform/log-mel/codec、ASR/TTS/VAD、WER/CER/MOS、any-to-any 架构、评估、安全、事故账本、toy 代码和原有练习；旧主题去向见 `research/encyclopedia-multimodal-depth-review-2026-10-26.md` 的 `coverage_map`。
- `changes`：将多模态任务成功概率改成对明确事件 `U,E,C,A` 的精确链式条件概率，并说明成功定义的边界；解释 processor 展开后的实际 token/latent 数与媒体文件大小、时长及 API 计费口径不同。补 CLIP 的 4 亿图文训练设定、SigLIP 的 batch scaling 条件、ViT/Swin 的视觉骨干差异；将 Flamingo、BLIP-2、早期 LLaVA 的瓶颈、机制、训练数据/参数数据和证据范围放入对照表。新增 DDPM/DDIM/DPM-Solver 采样路径比较、Diffusers checkpoint/scheduler 兼容约束、CFG scale 的公式口径；补 SoundStream/EnCodec codec 目标与码率范围、ITU MOS 报告要求、VQA 官方标注者协议、图像注入预印本范围和 C2PA 真实性边界。新增 Chameleon/NExT-GPT/AnyGPT 的统一位置、代表数据规模和失效限制对照，保留 16.0—16.31 全部既有主题并新增路线选型练习；在简介中链接第 17、18 章专项内容。
- `evidence`：研究记录 `research/encyclopedia-multimodal-depth-review-2026-10-26.md` 保存来源与主张映射及旧主题 coverage map。核读 CLIP、ViT、Swin、SigLIP、Flamingo、BLIP-2、LLaVA、Chameleon、NExT-GPT、AnyGPT、DDPM、DDIM、DPM-Solver、CFG、ControlNet、Latent Diffusion、DALL·E、Video Diffusion Models、FVD、SoundStream、EnCodec、Whisper 等论文的方法/实验/限制内容，以及 VQA 官方评分、ITU-T P.800.2、C2PA 2.4 和 Hugging Face Transformers/Diffusers 文档与处理源码。论文数字均绑定各自设置；本轮未运行目标模型或生产实验。
- `formula_status`：核对 16.2 条件概率、16.3—16.5 输入/证据分母、16.4 token budget、16.6 CLIP 双向损失、16.7 sigmoid loss、16.14 assistant loss mask、16.15 DDPM 反向均值、16.16 CFG、16.18 FVD、16.19 codec rate、16.21 WER/CER、16.23 VQA/chart/IoU/hallucination、16.25 incident rate、16.26 OCR/ASR/temporal 分母及 16.28 单位成功成本。将 FVD 矩阵平方根写成对称半正定形式；codec token 数注明固定帧率/K 个活跃码本的假设。
- `code_status`：16.27 标准库 toy 示例未改动；静态检查输入约束、None/0 区别和断言与正文一致，未执行代码或测试。
- `link_status`：新增论文/标准/官方文档直链指向对应来源；Transformers/Diffusers 的 `main` 文档标为动态页面，需按目标版本复核。第 16 章链接到第 17、18 章，两个本地目标均存在；原有外部链接没有全部在本轮重新打开。
- `sync_status`：`PAPERS.md` 多模态区增补 Swin、Chameleon、NExT-GPT、AnyGPT、VQA/ITU/C2PA 和官方 Transformers/Diffusers 文档入口；`INTERVIEW_BANK.md` 的统一多模态路线问题加入三种代表研究各自解决的问题及数据/证据限制；已有 `EXERCISES.md` 包含多模态 token、研究 roadmap、统一系统审计与面试复盘，章节另加一题选型练习。第四册目录 16—18 章映射有效。
- `validation`：`git diff --check` 通过；检查 16.0—16.31 标题连续，数学/文本/代码围栏闭合，新增相对链接目标存在，论文与正文阅读入口同步。未构建 PDF、未调用公式渲染器。
- `open_risks`：未独立复现论文，也未评估真实 VLM、扩散模型、ASR/TTS、人工 MOS、跨模态红队或线上 workflow；Chameleon/NExT-GPT/AnyGPT 等数据与 benchmark 不同，正文明确不作横向排名。2026 图像注入材料为预印本。章节保留单文件，细分生成和语音内容可分别继续读第 17、18 章。
- `status`：第四册第十六份正文按当前 `WRITING_SPEC.md` 精修验收完成；全书 goal 继续进行，下一份为第四册第十七份正文。

### 第四册第十五份正文：面试高频对比表深度精修验收（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 从头复读并精修 `book-04-llm-encyclopedia/chapters/15-面试高频对比表.md` 的 15.1—15.43；保留原有 43 组比较关系和 Adam/AdamW 教学代码，旧知识点去向见 `research/encyclopedia-comparison-tables-depth-review-2026-10-26.md` 的 `coverage_map`。
- `changes`：把比较条件具体化为同一任务、样本切片、质量指标、成本和尾延迟；限定 AdamW、GQA、ALiBi、RLAIF、LoRA、FlashAttention、PagedAttention 和长上下文/RAG 对照论文的证据范围。补足 Pre/Post-Norm 梯度路径、DPO 的 canonical offline 边界、top-p/temperature/beam 数值推演、LoRA 矩阵维度、ZeRO 1/2/3 与 FSDP 机制、KV decode 成本随长度增长、Retriever→Reranker 条件成功率和拒答率分母。新增企业客服知识库的 RAG/SFT/DPO 选型演练及错误定位路径；公式成本项区分初始上下文长度与生成期增长的 KV 长度。
- `evidence`：研究范围、代表性方案各自解决的问题、适用假设、实验边界与 coverage map 见上述 research note。主要来源包括原始优化/归一化/对齐/检索/推理/评测论文、NVIDIA CUDA 与 PyTorch FSDP 文档，以及 Anthropic 工程文章；后者只支撑其团队采用的 workflow/agent 定义，不作为行业标准。在线复核 arXiv 元数据显示 2501.01880 于 2024-12-27 提交，正文将其标为预印本；发现 GPT 原先引用的 1801.06146 是 ULMFiT，已改为 OpenAI 官方 GPT 原始预训练报告。
- `formula_status`：静态复核本章成本和检索链式概率变量；Retriever 召回概率与条件 reranker 成功率仅在单一 gold evidence 的定义下相乘，生成成本模型注明是账本分解而非通用线性延迟公式；教学数值均与其公式变量对应。未使用公式渲染器。
- `code_status`：保留 Adam/AdamW 单步更新 toy code，逐项核对初值及首步结果（Adam `0.6838`、AdamW `0.6738`）；未运行代码测试，也未把 toy 输出当成优化器质量结论。
- `link_status`：核对并修正 GPT 原始报告引用；补齐 ZeRO/FSDP、RLAIF、RAG/长上下文等支撑链接。两项 RAG/长上下文论文按各自 benchmark、模型、检索器和过滤条件限定结论，不主张跨任务排名。
- `sync_status`：`PAPERS.md` Transformer 核心机制入口补入 GPT 原始预训练报告，并保留长上下文/RAG 对照论文条目；第四册目录第十五章映射有效，下一章文件和目录入口均存在。
- `validation`：核对 15.1—15.43 标题连续、章内引用映射及公式源码；`git diff --check` 通过。未执行 PDF 构建、外部公式渲染或可执行代码测试。
- `open_risks`：本章含多主题比较，本轮重点阅读了与关键具体 claim 直接相关的原始材料，但不构成每一项技术的系统综述或独立复现；动态框架文档、论文修订和目标硬件性能仍需按当前版本/负载复核。toy 更新、基准结果和案例均不替代真实训练与线上评估。
- `status`：第四册第十五份正文本轮精修验收完成；全书 goal 继续进行，下一份为第四册第十六份正文。

### 第四册第十四份正文：重要论文索引深度精修验收（2026-10-26）

- `scope`：依据当前 `WRITING_SPEC.md` 从头复读并精修 `book-04-llm-encyclopedia/chapters/14-重要论文索引.md` 的 14.1—14.46；在旧条目上补证据与机制，新增按问题组织的方案对照；相关旧知识点去向见 `research/encyclopedia-important-papers-depth-review-2026-10-26.md` 的 `coverage_map`。
- `changes`：BERT 补回原始 NSP，并加入与 RoBERTa 的配方级比较；补齐 GQA/KV cache 的层数、batch、token、head 与字节数维度；将错误断行的 GRPO advantage 改为完整分段公式；区分 RAG-Sequence 与 RAG-Token，并解释 REALM、RETRO 的检索位置；拆分 R1-Zero/R1 多阶段配方，补充 DAPO 与 Dr.GRPO 各自处理的问题及证据限制。新增架构/系统、位置机制、检索范式、偏好对齐、推理 RL、测试时计算和 Agent benchmark 对照表。原谱系、论文阅读记录、阅读路径和来源目录均保留并顺延编号。
- `evidence`：研究记录逐条说明实际阅读的综述、原论文方法/实验章节、当前官方工具调用文档及 MCP 2026-07-28 规范；本章新增的研究范围和证据限制见上方 research note。在线阅读包括 BERT/RoBERTa、RAG/REALM/RETRO、DeepSeek-R1 v2、DeepSeekMath、DAPO、Dr.GRPO 等来源；未把所有引用当成本轮全文复读。
- `formula_status`：章节 32 个 math 环境均闭合且花括号计数平衡；修正 GRPO cases 换行；KV 容量按全层和 batch 逻辑形状定义；RAG 公式明确为 RAG-Sequence 的 top-k 简化。
- `code_status`：现有证据账本 Python 示例未改动；本轮没有新增或修改可执行代码，未运行代码测试。
- `link_status`：新增关键论断的原论文/官方资料链接；新增 RoBERTa、DAPO、Dr.GRPO、RAG 与 reasoning RL 综述、function calling 和 MCP 入口。新加的在线来源通过原始论文 HTML 或官方规范/文档核读；10 月 23 日旧记录中的全部既有链接未在本轮逐一重开。
- `sync_status`：`PAPERS.md` 增加 RoBERTa、RETRO/RAG 综述、Dr.GRPO 和版本化官方协议入口；第四册目录继续正确映射第十四、十五章。本章未拆分文件，现有阅读导航和来源入口保留。
- `validation`：`git diff --check` 通过；章节 fenced marker 共 64 个且成对，math 环境 32 个，未发现 `\\operatorname` 或旧式 `$$`；检查 14.42—14.46 顺延编号、目录映射和相对链接。
- `open_risks`：这是论文索引综合章，本轮对重点方案阅读了原始材料，但没有对全部 45 类论文做全文系统综述或独立复现；未重开旧来源清单的每个 URL；未构建 PDF。既有 Python 例子未执行，其他论文数字均限定为论文作者设置。
- `status`：第四册第十四份正文本轮精修验收完成；全书 goal 继续进行，下一份为第四册第十五份正文。

### 全书精修：增加代表性方案对照要求（2026-10-26）

- `scope`：根据维护者对 Memory 章节“代表性研究各自解决了什么”的认可，将跨方案分析纳入长期 `WRITING_SPEC.md`。
- `changes`：要求重要主题存在多种代表性研究/工程方案时，不止列论文和结论；需比较问题与约束、机制和假设、证据支持的收益、代价与失效边界、适用场景及后续改进。明确论文算法、框架接口和生产架构层次不同；改进方向需区分已发表方案与基于证据的推理。面试表达应从约束推导比较与选型，而非背泛化优缺点。
- `evidence`：依据维护者对 Memory 样章“代表性研究各自解决了什么、有什么优缺点和限制、如何改进”的反馈修订；技术方法细节仍须按各章节的研究资料核实。
- `validation`：复读相关规范章节和质量门禁，执行 `git diff --check`。
- `status`：长期精修规范已补充多方案研究与面试迁移要求。

### 全书精修：明确旧知识点的分层保留规则（2026-10-26）

- `scope`：根据维护者对 Memory 样章的反馈，微调 `WRITING_SPEC.md` 中的精修继承、知识点判定和章节拆分规则。
- `changes`：明确精修不要求把旧稿所有内容同等扩写，也不能因重组而让相关知识点无声消失。应结合书系目标和行业资料筛出主流、高频、基础、关键、热点、难点及必要前置问题并讲透；支持性内容简明说明；冷门或低优先级内容至少保留准确定位、并入相关小节/对比表或提供延伸入口，给未来升为重点留下文字根。要求记录旧知识点到新位置及解释层级的 `coverage_map`；核心内容放不进单文件时拆分文件并补齐导航。
- `evidence`：根据维护者对 Memory 记忆分类、记忆冲突等概念被样章删去的反馈，以及其对“重点讲透、冷门留根”的明确要求修订；本次没有新增技术结论。
- `open_risks`：Memory 样章尚未按新增 `coverage_map` 对照旧稿复核并补齐分类、冲突等主题的去向；本条规范更新不代表该章修订已完成。
- `status`：精修规范已更新；后续章节重组须执行分层筛选和留根记录。

### 第十七册第五章：按新版规范复审 Memory 样稿（2026-10-26）

- `scope`：依据新版 `WRITING_SPEC.md`，对照旧稿知识点与当前样稿，并复读 `book-17-agent-tool-use/chapters/05-memory系统.md`；本次是 Memory 章节复审与补深，不代表第十七册或全书精修完成。
- `changes`：明确短期/会话/长期是时间跨度与持久性视角，工作、情景、语义、程序性和偏好是功能视角；补足任务 checkpoint 和外部副作用恢复窗口、记忆冲突判定、更新/过期/压缩/用户删除的区别。将旧稿知识点逐项映射到现稿，并为冲突处置准确率和不安全候选写入率增加分母明确的评测项；旧 audit demo 不再以带任意阈值的代码形式保留，其需要检验的能力已逐项映射到机制说明、轨迹和评测。
- `evidence`：复审映射及资料边界见 `research/agent-memory-depth-review-2026-10-26.md`。对照记忆综述、LangGraph 官方 memory 文档、ICLR 2026 MemoryAgentBench 和 EverMemBench 预印本全文；EverMemBench 的数字采用正文详细表格口径，并标出摘要与正文口径差异。
- `validation`：检查章节标题、表格、代码围栏和内部导航；执行 `git diff --check`。本章没有可执行的新增代码，因此未运行代码测试。
- `sync_status`：章节仍由第十七册目录现有入口链接；本次未拆分文件，无需新增导航。
- `open_risks`：EverMemBench 仍为预印本，未在本项目独立复现；删除传播内容是明确标注的教学架构建议；旧 demo 的代码练习形式被移除，但原知识点和验证目标按映射保留。
- `status`：Memory 样稿已按新版规范复审并完成本轮补齐，供维护者继续评阅；不改变全书精修主线状态。

### 第十七册第五章：Memory 记忆系统深度样稿（2026-10-26）

- scope：按新增的知识点深度研究要求，对 book-17-agent-tool-use/chapters/05-memory系统.md 做样章重构；本次供维护者评阅，不代表第十七册或全书精修覆盖完成。
- changes：拆分记忆的载体、功能和生命周期三个分类轴；用 Python 项目版本贯穿记录 schema、来源/范围/有效时间、写入、权限先行的检索、上下文组装、冲突更新和删除传播；把评测从指标清单改成含分母和 failure mode 的评测维度；分别说明 Generative Agents、MemGPT、Reflexion、LongMemEval、Mem0 的机制与外推边界；对照 LangGraph、OpenAI Sessions 和工程团队文章解释框架接口与工程视角。移除了原先约 470 行的 toy demo，改为可追溯的端到端案例。
- evidence：研究笔记见 research/agent-memory-depth-review-2026-10-26.md；读取并交叉比较 6 项论文/综述及 LangGraph、OpenAI Agents SDK 文档和 LangChain、Databricks 工程文章。论文结果按各自 benchmark 和作者报告限定；删除传播状态作为教学设计建议，不冒充行业标准。
- validation：无可执行 Python demo；已核对章节标题、代码围栏与外链结构，执行 git diff --check。
- open_risks：本次目标是提供正文样稿供维护者评阅，尚未收到维护者反馈；产品级删除语义、敏感数据策略和动态框架文档需结合具体系统与版本进一步核验。
- status：Memory 深度样稿已完成并待维护者评阅；不改变全书精修主线完成状态。

### 全书精修：知识点深度规范与信息源目录（2026-10-26）

- `scope`：复读 `WRITING_SPEC.md` 中的知识点深度、资料检索、证据等级和质量门禁要求；本次只完善长期写作规范和资料发现入口，不代表已开始或完成 `book-17` 章节修订。
- `changes`：新增知识点深度检索与展开规则，加入对象/假设、因果机制、具体推演、失败观测、边界证据等可操作检查；用明确标注的记忆删除假设架构说明如何把泛化风险清单展开成可验证的状态轨迹。新增根目录 `信息源收录.md`，分类收录论文检索与正式出版入口、LLM 技术博客、LLM 教学与实现仓库及主流模型厂商官方文档；README 增加导航。
- `evidence`：信息源文档记录 2026-10-26 的入口核对日期；抽查 Stanford CS336、Hugging Face Course、LLMs from Scratch、nanochat、Megatron-LM、TRL、lm-evaluation-harness、vLLM、llama.cpp 等 GitHub 仓库的官方页面。GitHub 项仅作 LLM 范围内的发现入口；具体章节仍须使用支持对应 claim 的原论文、版本化文档、固定 revision 或实验，并记录证据边界。
- `open_risks`：尚未按新增规则精修任何正文；记忆删除示例是教学假设，不代表具体产品实现。全书逐章覆盖和相应资料核验仍按当前 `next` 主线执行。
- `status`：写作规范和来源导航增补完成；当前全书精修覆盖状态不变。

### 全书精修：将资料研究设为正文重写前置门槛（2026-10-26）

- `scope`：根据维护者对“知识点必须先有足够行业素材，才能写出有深度且兼顾初学者与专家的正文”的要求，强化 `WRITING_SPEC.md` 的全书精修研究流程。
- `changes`：明确重要知识点须先联网检索、阅读并交叉比较综述/教程、原始研究、官方资料与高质量工程材料；先在 `research/` 记录来源、机制、数据、分歧和边界，再重构正文。增加“不得仅凭模型已有知识扩写”“搜索摘要不是已读证据”“来源不足时记录缺口并缩小结论”等要求，并将研究带来的实质增量纳入精修记录。
- `evidence`：本次为写作规范调整；根据维护者对精修目标和素材要求的明确说明修订，未新增技术事实或章节结论。
- `open_risks`：该规则尚未用于后续章节精修验收；全书逐章研究和重写仍需按当前轮次推进。
- `status`：规范已更新；不代表任何章节已完成深度精修。

### 第四册第十份正文起进度记录重置说明（2026-10-23）

- 原第四册第十至第十九份正文及收尾条目系验证记录，未完成本轮要求的全书精修，现已从有效精修记录中移除。
- 第四册第九份及之前的记录保留；下一项从第四册第十份正文重新执行精修，逐章验收后再记录。

### 第四册第十份正文：内容精修重新启动检查点（2026-10-23）

- `scope`：重新从 `book-04-llm-encyclopedia/chapters/10-分布式训练系统.md` 开始，按 `WRITING_SPEC.md` 检查并行度定义、数值例子、代码输入边界和机制说明；本检查点不代表整章验收。
- `changes`：将 world size 关系改为不重复计入 MoE 专家并行度的基础 DP×TP×PP 口径，并说明 EP 与 DP 的定义依框架拓扑而异；修复并行维度及行切分 shape 校验接受 `bool` 的问题并加拒绝断言；step breakdown 拒绝布尔时长并补边界断言；MFU 数值例明确单卡吞吐与峰值配对、标明峰值为假设输入及 6N 估算限制，移除泛化的效率阈值结论；改正尾部梯度累积反例，使正确分母 2 得 3、误用固定分母 3 得 2，并说明不同有效 token 数时需按 token 总数归一化；成本有效利用率和 GPU 碎片率补齐分母非零、比例范围、同构资源池和任务 GPU 数定义。
- `evidence`：经指定代理读取 5 个 arXiv abs 页面并核对正式题名（1910.02054、2104.07857、2104.04473、1811.06965、2204.01075）；更正 Megatron-LM 和 ZeRO-Infinity 两处不完整题名。另对 13 个官方文档入口进行可达性检查，均返回 HTTP 200，PyTorch Distributed 有域名跳转。详细范围见 `research/encyclopedia-distributed-training-review-2026-10-23.md`；题名/可达性检查不代表通读论文或核验动态文档全部语义。
- `code_status`：本地独立抽取并执行 11 个 Python 围栏，11/11 通过；新增检查覆盖 bool 并行维度、bool tensor shape 和 bool 时长。输出含 `global_batch=64`、`ring_gib=14.0`、`incomplete_correct=3.0`、`fixed_window=2.0`、`collective_ratio=0.167`。
- `formula_status`：32 个 math 围栏闭合且花括号平衡；无 `\\operatorname`、旧式 `$$`、尖括号下标或 `\\text{...}` 下划线；`git diff --check` 通过。
- `sync_status`：第四册目录已映射本章；`PAPERS.md`、`EXERCISES.md`、`INTERVIEW_BANK.md` 和 `GLOSSARY_EN_ZH.md` 已有分布式训练内容，本检查点未作配套改动。
- `open_risks`：全章逐节深度复读、动态官方文档语义与版本核对、其他教学代码边界、全章证据和同步收口尚未完成；旧 research 笔记仅作线索，未作为本轮精修完成证据；未运行 GPU/NCCL、多节点或目标集群实验。
- `status`：第四册第十份正文精修进行中；下一步继续完成本章机制、例子、失败边界、来源和配套审计，未完成前不推进到第十一章。

### 第四册第十份正文：全章逐节精修检查点（2026-10-23）

- `scope`：按 `WRITING_SPEC.md` 深度复读 10.1—10.24，覆盖训练账本、并行策略、状态分片、重计算/精度、效率、训练故障、拓扑/网络、存储与 checkpoint、Kubernetes GPU、多租户、TrainingJob、实验血缘、可观测性、SLO、成本、安全、审计、平台架构、综合实验及延伸阅读；未宣称全章验收。
- `changes`：将 ZeRO/FSDP 内存表述限定为被切分状态的缩减，不泛化为所有状态同时缩至 `1/D`；梯度累积和更新代码拒绝溢出结果，故障分类器限制 ratio 为 `[0,1]` 并验证 flag/phase 字段。checkpoint 教学示例改为只校验必需字段及文件名映射摘要的 gate，明确摘要不覆盖文件字节，不能证明原子提交或恢复连续性；调度 toy 拒绝超容量占用/预留及非整数 GPU 计数，dominant share 的 GPU 口径同样限定整卡；SLO 补符合条件事件的分母与 good/bad 口径，修正请求失败率不能直接换算成坏状态时长。
- `evidence`：读取 PyTorch 2.14 Distributed Checkpoint 的 state-dict API 段落、Kubernetes device plugin 的 extended resource / `ListAndWatch` / `Allocate` 说明、Google SRE Workbook 的 SLI/error budget/burn-rate 段落。局部页面语义及证据限制见 `research/encyclopedia-distributed-training-review-2026-10-23.md`；不是所有版本的全量语义审计。
- `code_status`：从本章直接提取 11 个 Python 围栏并逐块编译、执行，11/11 通过；新增梯度/更新溢出、故障 ratio/flag/phase、半卡 GPU、usage/capacity、reservation/capacity 拒绝检查。
- `formula_status`：SLO 目标与坏事件比例限定同一符合条件事件分母；未使用外部公式渲染器；`git diff --check` 通过。
- `sync_status`：第四册目录第十章映射有效，PAPERS、EXERCISES、INTERVIEW_BANK、GLOSSARY 已有本主题入口，暂无必要重复新增；补充核对梯度平均只适用于同一 DP process group，并在数据切分段说明 sampler 的分片/epoch 语义应绑定框架版本；research 标题已标出旧记录是历史验证，不作本轮精修验收依据。
- `validation`：本轮再次从正文直接提取 11 个 Python 块并逐块编译、执行，11/11 通过；输出包括 `global_batch=64`、`ring_gib=14.0`、`incomplete_correct=3.0`、`fixed_window=2.0`、`manifest_digest=314b8834b309`、`dominant_share=0.25` 和 `burn_rate≈2.0`。32 个 math 围栏花括号平衡，118 个 fenced 标记成对，19 个外链唯一且无相对链接，第四册目录入口有效；`git diff --check` 通过。
- `open_risks`：旧的 HTTP 200 记录只作为历史可达性证据，本轮未能重新在线读取全部外链或完成动态官方文档的全量版本语义复核；未运行 GPU/NCCL/FSDP/Megatron、多节点、故障注入及目标集群测试，未构建 PDF。上述限制不影响本章声明范围内的文本、公式、教学代码和导航验收；未调用外部公式渲染器。
- `status`：第四册第十份正文本轮精修验收完成；下一步进入第四册第十一份正文。

### 第四册第十份正文：最终精修验收（2026-10-26）

- `scope`：完成 10.1—10.24 全章最终口径审计，覆盖训练账本、并行组、collective、状态分片、故障取证、checkpoint、调度、SLO、成本和平台治理；本轮不将历史验证记录当作完成依据。
- `changes`：补充梯度平均只在同一 DP process group 内成立，明确 TP/PP rank 不进入该分母；补充 sampler 分片/epoch 语义须绑定 PyTorch 版本；sampler 延伸阅读改用稳定的 PyTorch data 文档入口，避免依赖未在线核验的深层锚点。
- `validation`：正文直接提取的 11 个 Python 块全部编译并执行通过；32 个 math 围栏花括号平衡，全部 118 个围栏标记成对，无禁用 `\\operatorname` 或旧式 `$$`，19 个外链唯一且无相对链接；第四册目录入口有效，`git diff --check` 通过。
- `evidence`：本章已有的论文题名核对、官方语义局部抽查和历史外链可达性记录保留在 `research/encyclopedia-distributed-training-review-2026-10-23.md`；本轮网络连接失败，因此不新增在线语义核验声明。
- `sync_status`：PAPERS、EXERCISES、INTERVIEW_BANK、GLOSSARY 已有分布式训练主题入口，无需重复添加；目录映射保持有效。
- `open_risks`：未运行真实 GPU/NCCL/FSDP/Megatron、多节点训练、checkpoint 故障注入或目标集群性能测试；动态文档全量版本核验和 PDF 构建仍待环境支持。教学公式和代码的适用边界已在正文说明。
- `status`：本章声明范围内精修验收完成；主线进入第四册第十一份正文，继续全书 goal。

### 第四册第十一份正文：精修重新启动检查点（2026-10-26）

- `scope`：重新读取 11.1—11.25，覆盖长上下文预算与位置利用、上下文工程、RAG 召回/重排/证据支持、Agent 状态机、工具契约与副作用、MCP/A2A、Memory、Code/Browser/Multi-Agent、评估、安全和成本；既有 research 验收记录只作线索，不作为本轮完成证据。
- `changes`：发现重试退避公式原先先截断指数项再添加抖动，可能超过声明的 `t_max`；改为在加入抖动后统一截断，并明确 `t_max` 包含抖动、教学代码只验证无抖动基线。
- `code_status`：从正文直接提取 5 个 Python 块逐块编译、执行，5/5 通过；输出覆盖 needle 位置、RRF 排名、工具参数拒绝、退避预算和路径穿越拒绝。
- `formula_status`：23 个 math 围栏花括号平衡；未发现 `\\operatorname` 或旧式 `$$`；退避公式已与文字边界一致。
- `sync_status`：第四册目录第十一章入口有效，PAPERS、EXERCISES、INTERVIEW_BANK、GLOSSARY 已有长上下文/RAG/Agent 入口；本检查点未重复添加配套条目。
- `open_risks`：全章逐节最终口径、全部来源版本核验、外链在线复核和配套收口尚未完成；未运行真实 embedding/ANN/reranker、MCP/A2A 服务、浏览器/代码沙箱、生产 Agent 或 PDF 构建。
- `status`：第四册第十一份正文精修进行中；`next` 仍为本章，不进入第十二章。

### 第四册第十一份正文：最终精修验收（2026-10-26）

- `scope`：完成 11.1—11.25 全章复读与最终口径审计，覆盖长上下文预算/位置利用、上下文工程、RAG 检索与证据、Agent 状态、工具契约/副作用、MCP/A2A、Memory、Code/Browser/Multi-Agent、Trace、评估、安全和成本。
- `changes`：修正重试退避公式先截断再加抖动导致可能超过 `t_max` 的问题；改为加入抖动后统一截断，并明确 `t_max` 包含抖动，教学代码只验证无抖动基线。其余章节定义域、分母、状态和失败边界复核通过。
- `validation`：5 个 Python 块逐块编译、执行，5/5 通过；23 个 math 围栏花括号平衡，64 个 fenced 标记成对，无禁用 `\\operatorname` 或旧式 `$$`，12 个外链唯一且无相对链接；第四册目录第十一章入口有效，`git diff --check` 通过。
- `evidence`：既有论文题名、协议版本和官方资料局部核验记录保留在 `research/encyclopedia-long-context-rag-agent-review-2026-10-23.md`；本轮不把历史外链可达性记录扩写为新的在线语义核验。
- `sync_status`：PAPERS、EXERCISES、INTERVIEW_BANK、GLOSSARY 已覆盖本章主题，无需重复新增；目录映射有效。
- `open_risks`：未运行真实 embedding/ANN/reranker、MCP/A2A 服务、浏览器/代码沙箱、生产 Agent 或论文 benchmark；动态协议/官方文档全量版本核验和 PDF 构建仍待环境支持。教学位置利用率、成功概率分解和 toy 代码边界已在正文声明。
- `status`：本章声明范围内精修验收完成；主线进入第四册第十二份正文，继续全书 goal。

### 第四册第十二份正文：精修重新启动检查点（2026-10-26）

- `scope`：重新读取 12.1—12.17，覆盖 CoT、scratchpad、self-consistency、verifier、process supervision、搜索、test-time compute、数学/代码推理、统计比较、benchmark 污染、人工与模型评审、事实性、校准、安全和发布决策；既有 research 验收记录只作线索。
- `changes`：补强 paired bootstrap 教学代码的输入边界：先把成对输入转换为序列，拒绝字符串、不可迭代对象和长度不一致数据，并加入 `None` 输入拒绝断言，避免评估错误以未分类异常暴露。
- `code_status`：从正文直接提取 2 个 Python 块逐块编译、执行，2/2 通过；覆盖多数投票和 paired lift/bootstrap，输出区间 `(0.0, 0.6)` 与输入边界拒绝结果。
- `formula_status`：35 个 math 围栏花括号平衡；未发现 `\\operatorname` 或旧式 `$$`；CoT 回归、Pass@k、配对提升、bootstrap、校准和严重度分母均保留定义域说明。
- `sync_status`：第四册目录第十二章入口有效，PAPERS、EXERCISES、INTERVIEW_BANK、GLOSSARY 已有 reasoning/评估/verifier 主题入口；本检查点未重复添加配套条目。
- `open_risks`：全章逐节最终口径、来源版本/外链复核和配套收口尚未完成；未运行真实模型、verifier、搜索、代码沙箱或 benchmark，未构建 PDF。
- `status`：第四册第十二份正文精修进行中；`next` 仍为本章，不进入第十三章。

### 第四册第十二份正文：最终精修验收（2026-10-26）

- `scope`：完成 12.1—12.17 全章复读与最终口径审计，覆盖 CoT、self-consistency、verifier、过程监督、搜索、test-time compute、数学/代码推理、统计比较、benchmark 污染、人工/模型评审、事实性、校准、安全和发布决策。
- `changes`：paired bootstrap 代码补充序列输入、字符串/不可迭代对象、非空等长和二值标签校验；过程监督的自动覆盖率与人工成本公式补充 `a_ij` 为 0/1、`c_ij` 为有限非负值及零步骤样本分母规则。
- `validation`：2 个 Python 块逐块编译、执行，2/2 通过；35 个 math 围栏花括号平衡，78 个 fenced 标记成对，无禁用 `\\operatorname` 或旧式 `$$`，11 个外链唯一且无相对链接；第四册目录第十二章入口有效，`git diff --check` 通过。
- `evidence`：既有论文题名、benchmark 和评估工具链来源记录保留在 `research/encyclopedia-reasoning-eval-review-2026-10-23.md`；本轮未把历史可达性记录扩写为新的在线语义核验。
- `sync_status`：PAPERS、EXERCISES、INTERVIEW_BANK、GLOSSARY 已覆盖 reasoning、verifier、污染、pass@k、judge 和校准主题，无需重复新增；目录映射有效。
- `open_risks`：未运行真实模型、verifier、搜索、代码沙箱或 benchmark；动态来源全量版本核验和 PDF 构建仍待环境支持。教学 bootstrap、Pass@k、校准和严重度公式的适用边界已在正文说明。
- `status`：本章声明范围内精修验收完成；主线进入第四册第十三份正文，继续全书 goal。

### 第四册第十三份正文：最终精修验收（2026-10-26）

- `scope`：完成 13.1—13.27 全章复读与最终口径审计，覆盖威胁模型、对齐/监督、策略层级、注入与 taint、拒答、隐私/DP、数据治理、来源证明、红队、评估、事故响应、解释性、steering、model editing、公平、guardrail、供应链和发布治理。
- `changes`：补充数据最小化率要求实际字段分母大于零，并明确加权字段量须绑定同一字段清单和权重版本，避免把字段数或字符数直接当作可比隐私指标。
- `validation`：3 个 Python 块逐块编译、执行，3/3 通过；40 个 math 围栏花括号平衡，90 个 fenced 标记成对，无禁用 `\\operatorname` 或旧式 `$$`，18 个外链唯一；两个书内相对链接和第四册目录第十三章入口均存在，`git diff --check` 通过。
- `evidence`：既有论文题名、NIST/OWASP/MITRE/C2PA/SLSA 来源记录保留在 `research/encyclopedia-safety-governance-review-2026-10-23.md`；本轮未把历史外链可达性记录扩写为新的在线语义核验。
- `sync_status`：PAPERS、EXERCISES、INTERVIEW_BANK、GLOSSARY 已覆盖安全、隐私、供应链、解释性、公平和治理主题，无需重复新增；目录映射有效。
- `open_risks`：未运行真实红队、membership inference、DP-SGD、SAE/ROME/MEMIT、guardrail 或生产安全评估；动态来源版本核验和 PDF 构建仍待环境支持。教学风险排序、taint 传播和线性干预边界已在正文说明。
- `status`：本章声明范围内精修验收完成；主线进入第四册第十四份正文，继续全书 goal。

### 第二册第十份正文：整章遗留审计与验收（2026-10-23）

- `scope`：补齐 2026-10-14 至 10-19 局部检查点遗留的整章审计：第 105—124 讲全章 15,753 行的来源核验、公式格式统一、代码回归、配套同步与整章验收；未修改长期规范。
- `changes`：发现并修正第 120—124 讲遗留的 152 对旧式 `$$` 数学围栏（10-19 审计记录声称全章旧式围栏为 0，实际未覆盖这些讲次），机械转换为 fenced `math` block；本轮未改动正文语义内容。
- `evidence`：32 个唯一外链经指定代理核验，29 个 HTTP 200，3 个 openai.com 页面 403（代理限制，与第一册多模态章记录一致，不作链接失效证据）；23 个唯一 arXiv ID 题名经 abs 页面逐个核对，与引用主题全部一致（含 Whisper 正确题名）。Atom API 批量查询限流（429/000），改用 abs 页面。详见 `research/multimodal-advanced-chapter10-review-2026-10-23.md`。
- `code_status`：转换后 17 个 Python 块逐块独立执行，17/17 通过；此前各讲检查点记录的 90 余项边界断言适用于未改动代码路径。未安装 CUDA。
- `formula_status`：全章 318 个 `math` 围栏成对、花括号平衡；无 `\operatorname`、无尖括号下标；全部围栏（text/python/json）闭合；`git diff --check` 通过。
- `sync_status`：`PAPERS.md` 多模态区新增 DALL·E 1/2、Tacotron 2、FastSpeech 1/2、VITS、VALL-E、CLIPScore、MME、MMBench、MMMU、POPE、LayoutLMv3 入口；`GLOSSARY_EN_ZH.md` 新增 POPE、CLIPScore、FID、LayoutLMv3 条目；`EXERCISES.md`、`INTERVIEW_BANK.md` 已覆盖本章主题，不重复添加；第二册目录第 105—124 讲映射正确。
- `open_risks`：未运行真实多模态模型、GPU、生成质量回归或生产 serving；论文题名/链接核验不等于全文与实验复核；openai.com 三页 403 须在引用当前状态前重新核验；PDF 构建仍受 Pandoc 缺失限制。
- `status`：第二册第十份正文整章验收完成，第二册全部十份正文已有本轮验收记录；全书精修主线继续第四册第十份正文 `book-04-llm-encyclopedia/chapters/10-分布式训练系统.md`，全书 goal active。

### 第四册第九份正文：推理部署账本首个检查点（2026-10-23）

- `scope`：全文初读第 9.1—9.19，局部核验请求生命周期、token workload、scheduler、KV/PagedAttention、online softmax、虚拟页、量化、runtime、PD、扩缩容和分位数摘要。
- `changes`：所有教学代码整数输入拒绝 bool；scheduler 分离 running sequence 与 waiting prefill token 预算；percentile 明确 nearest-rank 离散口径。
- `evidence`：新增 `research/encyclopedia-inference-deployment-review-2026-10-23.md`；尚未完成来源论文/官方文档逐项读取。
- `code_status`：`python3 /tmp/check_deployment_review.py` 通过，12 个标准库 Python 块独立执行，非法输入、预算、分位数及 21 个 math 块结构/围栏检查通过。
- `open_risks`：动态 batching 公平性、paged/prefix cache 引用计数、FlashAttention/量化公式、PD/多级 KV/路由保护、API/流式协议/发布回滚、全章链接及配套仍待核验；第二册第十份正文遗留审计保留。
- `status`：第四册第九份正文局部检查点，顶部 next 保持，全书 goal active。

### 第四册第九份正文：生命周期与缓存边界检查点（2026-10-23）

- `scope`：继续核验请求状态、prefix cache 命中分母、online softmax 与量化教学代码边界。
- `changes`：生命周期账本增加显式合法状态转移，并修正示例漏过 TOKENIZED 状态；prefix 命中率明确输入 token 分母大于零；量化值拒绝 bool。
- `code_status`：`python3 /tmp/check_deployment_review.py` 通过，额外验证 RECEIVED 直接 DECODING 被拒、bool 量化值被拒；12 个代码块及 21 个 math 块结构仍通过。
- `open_risks`：动态 batching 公平性、PagedAttention/Prefix Cache 完整引用计数、FlashAttention/量化来源、PD/多级 KV/路由保护、API/流式协议、全章链接和配套仍待收口。
- `status`：局部检查点，第四册第九份正文继续进行中，全书 goal active。

### 第四册第九份正文：后半部署边界与来源检查点（2026-10-23）

- `scope`：继续核验多级 KV/跨节点、模型路由与扩缩容、保护策略、流式 API、发布治理、事故诊断及 Agent serving；同步部署练习。
- `changes`：生命周期状态、prefix cache 分母和量化输入边界继续收紧；补 FlashAttention tiling、vAttention 连续虚拟视图、GPTQ 二阶近似和 AWQ activation-aware 的来源边界说明。
- `evidence`：读取 FlashAttention v2、vAttention、GPTQ、AWQ HTML 摘要/方法段落；`python3 /tmp/check_deployment_review.py --links` 通过代理核验 18 个链接中的 17 个 HTTP200，`docs.vllm.ai` 返回 403；本地 `--links` 与代码边界检查通过。
- `sync_status`：EXERCISES.md 新增 5 项推理优化与部署练习。
- `code_status`：12 个标准库 Python 块、非法输入和 21 个 math 块结构检查通过；未运行 GPU/kernel/serving 压测。
- `open_risks`：PagedAttention 固定 HTML 404、vLLM 文档 403、完整来源/导航收口、API/协议/回滚语义及全章验收仍待完成；第二册第十份正文遗留审计保留。
- `status`：局部检查点，第四册第九份正文继续进行中，全书 goal active。

### 第四册第九份正文：整章验收（2026-10-23）

- `scope`：完成第 9.1—9.19 及回顾的全文复读、部署账本代码、公式结构、来源、导航和配套同步验收。
- `changes`：修正生命周期状态链、running 与 prefill token 预算混用、prefix 命中率空分母、量化 bool 输入和 nearest-rank 口径；补动态 batching、KV 引用、PD 取消、fallback 幂等、startup/readiness/drain、Agent 工具权限边界。
- `evidence`：读取 FlashAttention v2、vAttention、GPTQ、AWQ 摘要/方法段落及官方文档；18 个链接目标中 17 个 HTTP200，vLLM 文档 403、PagedAttention HTML 固定路径 404，abs 入口可达；详见 `research/encyclopedia-inference-deployment-review-2026-10-23.md`。可达性和摘要不代表完整论文或运行时验证。
- `code_status`：`python3 /tmp/check_deployment_review.py` 通过；12 个标准库 Python 块独立执行，状态转移、bool 输入、scheduler、KV、页表、量化、PD、replica、percentile 边界通过。
- `formula_status`：21 个 math 块花括号/环境及围栏结构通过，未使用外部公式渲染器；`git diff --check` 通过。
- `sync_status`：EXERCISES.md 新增 5 项推理优化与部署练习；下一章路径核对为 `book-04-llm-encyclopedia/chapters/10-分布式训练系统.md`。
- `open_risks`：未运行真实 GPU、kernel、serving、网络压力、跨节点迁移或量化质量回归；两个来源链接受代理/版本路径限制；第二册第十份正文遗留审计保留。
- `status`：本章声明范围内精修验收完成；主线进入第四册第十份正文 `book-04-llm-encyclopedia/chapters/10-分布式训练系统.md`，全书 goal active。

### 第四册第八份正文：采样与延迟局部检查点（2026-10-22）

- `scope`：全文初读第 8.1—8.15 节及回顾，局部核验采样数值、top-p 与首 token/延迟口径；整章进行中。
- `changes`：修复采样全局 random 状态、高温极端 logits 差值溢出、top-p 归一化溢出与零概率尾项；明确排序前缀、greedy 确定性和硬约束失败；首 token 使用 prefill logits，延迟拒绝空观测与原点不一致。预算/KV 拒绝 bool，已选 token 概率限定 (0,1]。
- `evidence`：读取 Transformers GenerationConfig 相关参数与 speculative decoding v2 采样方法段落；投机部分正文尚待修订。见 `research/encyclopedia-decoding-review-2026-10-22.md`。
- `code_status`：13 个标准库 Python 块独立编译执行通过；极端温度概率、非法分布、局部随机流、top-p 大权重/零概率和延迟边界断言通过。命令 `python3 /tmp/check_decoding_review.py`。
- `formula_status`：24 个 math 块花括号/环境及围栏结构通过，未使用外部数学渲染器；git diff --check 通过。
- `open_risks`：penalty/beam/entropy/约束/账本代码、best-of-N、KV 边界、投机解码及全章来源/链接/配套仍待收口；第二册遗留审计保留。
- `status`：第四册第八份正文局部检查点，顶部 next 保持，全书 goal active。

### 第四册第八份正文：投机解码与边界扩展检查点（2026-10-23）

- `scope`：继续第八章，核验 speculative decoding 的同 token 空间、p/q 支持集、接受—拒绝和残差分布；并补查 penalty、beam、best-of-N、KV、约束与配置账本代码边界。
- `changes`：正文补原论文方法链接、sampling-adjusted 分布和 tokenizer 对齐要求；残差单步 toy 例恢复 target `[0.5,0.3,0.2]`，q=0 提议、p=q 零残差及全部接受边界明确。修正 repetition 计数、beam 非正 logprob、best-of-N 独立性、投票最高票/平票语义、KV 适用前提和配置字段校验。
- `evidence`：读取 speculative decoding v2 第 2 节和 Transformers GenerationConfig 页面；论文入口题名已核对，2202.00666 修正为正式题名 `Locally Typical Sampling`。HF 文档随后 HTTP403，参数核对来自已下载页面，不作为线上可达性证据。
- `code_status`：13 个标准库 Python 块独立执行；残差分布、非法候选、局部随机、top-p、大权重、延迟、penalty、beam、KV、约束及配置账本边界断言通过。命令 `python3 /tmp/check_decoding_review.py`。
- `formula_status`：24 个 math 块结构通过，未使用外部公式渲染器；git diff --check 待本批完成后复跑。
- `open_risks`：HF 文档线上 403、剩余 arXiv/官方链接完整核验、配套同步和整章收口仍待完成；未运行真实模型、GPU、serving 或完整 speculative rollout。第二册遗留审计保留。
- `status`：局部检查点，第四册第八份正文继续进行中，全书 goal active。

### 第四册第八份正文：整章验收（2026-10-23）

- `scope`：完成第 8.1—8.15 及回顾的全文复读、算法边界、教学代码、来源、导航和配套同步验收。
- `changes`：采样改局部 RNG 与极端 logits 稳定实现；top-p 采用排序前缀并处理大权重；投机解码补残差恢复 target 的数值证据；延迟区分 prefill 首 token、TTFT 和 TPOT；best-of-N、投票、KV、约束与配置账本补适用边界。
- `evidence`：读取 speculative decoding v2 第 2 节、Transformers GenerationConfig 参数页；13 个链接目标中 11 个外部页面 HTTP200，两个 docs 页面受代理 403 限制，论文题名与正文资料项核对一致；详见 `research/encyclopedia-decoding-review-2026-10-22.md`。
- `code_status`：13 个标准库 Python 块独立执行；极端数值、随机状态、top-p、penalty、beam、预算、延迟、残差分布、KV、约束、best-of-N 和配置输入边界断言通过。命令 `python3 /tmp/check_decoding_review.py`。
- `formula_status`：24 个 math 块花括号/环境与围栏结构通过，未使用外部公式渲染器；`git diff --check` 通过。
- `sync_status`：EXERCISES.md 新增 5 项推理与解码练习；第四册目录/简介映射、第六册/第二十四册专题入口及下一章路径核对。
- `open_risks`：未运行真实模型、GPU、serving 性能或完整 speculative rollout；docs.vllm 与一个 Transformers 文档页面的线上可达性受代理限制；PDF 仍受 Pandoc 缺失限制。第二册第十份正文遗留审计继续保留。
- `status`：本章声明范围内精修验收完成；主线进入第四册第九份正文 `book-04-llm-encyclopedia/chapters/09-推理优化与部署.md`，全书 goal active。

### 第四册第七份正文：整章验收（2026-10-22）

- `scope`：接续四个局部检查点，复核 SFT—偏好优化—RLVR—蒸馏—发布链路，完成本章声明范围内验收；长期规范未修改。
- `changes`：诊断账本增加逐条非负整数计数、enabled 不超 expected、布尔标签及安全切片冲突校验；RLHF 公式标明最大化目标与负号约定；线上反馈限定授权/脱敏/保留。配套题库纠正 reference-free 推导部署成本更低的错误，论文索引明确 KTO 使用 reference。
- `evidence`：此前方法段落核验范围见 `research/encyclopedia-posttraining-alignment-review-2026-10-22.md`；本次 24 个唯一外部页面 HTTP200，29 个链接目标的本地路径及外部引用锚点通过；16 篇 abs 页面题名核对所引方法一致。页面可达和题名不作为全文核验。
- `code_status`：唯一标准库块独立编译执行通过，正常五项输出不变；六组非法账本输入、DPO 非法概率/beta/溢出、零分母、小 softplus 及此前方法数值例子通过。命令为 `python3 /tmp/check_posttraining_review.py`；外链检查使用 `--links`，沙箱网络失败后经授权通过代理执行。
- `formula_status`：35 个 math 块花括号/环境与围栏结构通过，重要目标语义及数值已按局部记录核对；未调用外部公式渲染器。
- `sync_status`：EXERCISES.md 的 5 项练习、INTERVIEW_BANK.md 对应题、PAPERS.md 分类已同步；本册目录/简介章节映射、本章专题入口及下一章路径核对；git diff --check 通过。
- `open_risks`：未运行完整训练、真实模型、GPU、RL/工具生产系统或安全验证；官方 main 文档动态变化，PDF 仍受 Pandoc 缺失限制。第二册第十份正文遗留审计仍未完成。
- `status`：本章声明范围内精修验收完成；主线进入第四册第八份正文，全书 goal 保持 active。

### 第四册第七份正文：IPO、SimPO 与 V3.2 来源检查点（2026-10-22）

- `scope`：接续第七章，核验 IPO/SimPO 方法段落、RLAIF 摘要及 V3.2 后训练报告与模型卡；整章仍在进行中。
- `changes`：IPO 明确 τ 为正则强度、原文动作概率与实现均值口径；补两侧梯度例子。SimPO 补 margin/β 阈值、损失及长度翻倍反例，限定长度平均与下游回归。长度偏差不再由 logprob 求和直接推出；RLAIF 区分 AI 评价、规则奖励与离线偏好训练。V3.2 补固定报告引用和四项稳定化措施的独立解释。
- `evidence`：读取 IPO v2 第 5.2 节及相关正则说明、SimPO v2 目标和限制段落、RLAIF 摘要、V3.2 v1 第 3/3.1 节与官方模型卡相应段落。详见对应 research note；未声称全文阅读或训练复现。
- `code_status`：`python3 /tmp/check_posttraining_review.py` 通过；单个标准库代码块正常输出不变，新增 IPO 平方误差和 SimPO 0.513015 数值断言通过；35 个公式块结构及 git diff --check 通过。
- `sync_status`：EXERCISES.md 新增 5 项 mask、目标、归一化、奖励与蒸馏边界练习。
- `open_risks`：诊断代码完整输入边界、全章外链/锚点与论文题名、导航和剩余配套验收仍待完成；未使用外部公式渲染器。第二册第十份正文遗留审计保留。
- `status`：局部检查点，顶部 next 保持第四册第七份正文，全书 goal 继续 active。

### 第四册第七份正文：DAPO、GSPO 与蒸馏边界检查点（2026-10-22）

- `scope`：接续第七章，核验 DAPO 动态采样、GSPO 比率/截断及蒸馏公式适用条件；整章保持进行中。
- `changes`：DAPO 明确过滤组内全对/全错并补混合奖励例子、分布变化和采样预算；GSPO 修正 GRPO 连乘暗示，补几何平均与逐 token clip 反例，限定非空有效生成 token、clip 尺度及共享优势；蒸馏区分固定历史 forward KL、教师软分布与硬修订；安全指标零分母记未定义，轨迹保留加入权限和脱敏边界。
- `evidence`：读取 DAPO v1 第 3.2 节与 GSPO v1 第 4.1 节方法段落并加入固定版本引用；未声称全文核验或训练复现。蒸馏交叉熵/KL 恒等式为局部数学检查，不代表具体训练方法复现。
- `code_status`：`python3 /tmp/check_posttraining_review.py` 通过；唯一标准库代码块独立执行，五项输出不变；三组奖励过滤、优势 [-1,-1,1,1]、GSPO 几何平均重复不变及 CE-H=KL 数值断言通过。
- `formula_status`：35 个 math 块花括号/环境及围栏结构通过，未使用外部公式渲染器；git diff --check 通过。
- `open_risks`：IPO/SimPO、RLAIF 定义、章末 V3.2 来源、诊断器完整边界、全章外链和配套同步仍待验收；第二册第十份正文遗留审计保持未完成。
- `status`：局部检查点，顶部 next 仍为第四册第七份正文；全书 goal 保持 active。

### 第四册第七份正文：偏好变体与归一化局部检查点（2026-10-22）

- `scope`：接续本章进行中任务，深入核验KTO、ORPO、Dr.GRPO及reward-human gap；整章未验收。
- `changes`：KTO从BCE示意替换为有界sigmoid价值及KL参考点目标；ORPO补长度平均概率、odds-ratio项与数值例子；Dr.GRPO补去组内std和固定L_max分母；reward-human gap先统一事件与量纲。
- `evidence`：读取KTO v3第4节、ORPO v2目标、Dr.GRPO v1长度/难度偏置相关段落，见 `research/encyclopedia-posttraining-alignment-review-2026-10-22.md`，未复现训练。
- `code_status`：唯一标准库Python块重新独立编译执行通过，正常输出不变；KTO0.268941/0.731059、ORPO0.223144数值复算通过。
- `formula_status`：35个math块花括号/环境结构通过；未使用外部公式渲染器，git diff --check通过。
- `open_risks`：IPO/SimPO、GRPO/DAPO/GSPO、RLAIF、蒸馏、V3.2证据、诊断器边界、全章外链与配套仍待收口。
- `status`：继续第四册第七份正文，顶部next保留；第二册遗留审计仍未完成。

### 第四册第七份正文：后训练与对齐局部检查点（2026-10-22）

- `scope`：全文初读7.1—7.15、小结及V3.2补充，局部核验SFT、LoRA、PPO和DPO；整章保持进行中。
- `changes`：修正SFT shift示意、LoRA秩上界、PPO clip非硬约束、DPO sum/mean目标差异和beta含义；诊断代码使用log1p并拒绝非法概率/beta和溢出margin。
- `evidence`：读取DPO v2推导、Spinning Up PPO及TRL DPO相关段落，范围见 `research/encyclopedia-posttraining-alignment-review-2026-10-22.md`；尚未完成全章来源验收。
- `code_status`：唯一标准库Python块独立编译执行通过；输出0.917/0.688/0.667/0.500/1.000，首条DPO loss0.673347167228034；bool/正log probability/非法beta/溢出margin拒绝断言与小softplus值通过。
- `formula_status`：34个math块和围栏结构通过，整章语义核验尚未结束；未调用外部数学渲染器。
- `open_risks`：偏好变体、GRPO系列、奖励指标量纲、蒸馏、章末V3.2证据、外链及配套仍待核验；未运行真实训练或GPU，PDF仍受Pandoc缺失限制。
- `status`：继续第四册第七份正文，顶部next保留；第二册遗留审计仍未完成；git diff --check通过。

### 第四册第六份正文：整章验收（2026-10-22）

- `scope`：接续局部检查点复读6.1—6.14及小结，按百科定义、机制、反例和专题入口验收，未修改长期规范。
- `changes`：明确温度采样、空池和effective epoch口径；区分替换/累积合成数据的collapse证据；scaling改有来源的N/D拟合与6ND约束，训练compute-optimal不混称生命周期成本；收紧消融预算、主动学习、扫描命中/风险率和版本删除边界。
- `evidence`：读取collapse与accumulation摘要，复核Chinchilla公式，接续TRL与去重资料；8篇正文arXiv题名/版本核对，13个唯一外链HTTP200。详见 `research/encyclopedia-pretraining-data-review-2026-10-22.md`。
- `code_status`：5个标准库Python块独立编译执行通过；质量分数/风险flag、去重计数、packing维度/token与shift目标、合成字段和误标边界通过；pack目标数3，EOS监督保留。
- `formula_status`：13个math块与围栏结构通过；温度权重[0.2,0.8]与[1/3,2/3]、6ND等数值复算通过，未使用外部数学渲染器。
- `sync_status`：EXERCISES.md新增5项数据口径/边界验收，本章、专题、本册目录、练习及下一章目标存在，git diff --check通过。
- `open_risks`：未运行真实语料平台、近重复系统、TRL、训练、GPU、多节点恢复、模型遗忘或生产安全；来源可达/题名不当作全文与合规核验；PDF仍受Pandoc缺失限制。
- `status`：本章声明范围内精修验收完成；继续第四册第七份正文，第二册遗留审计保持未完成。

### 第四册第六份正文：预训练与数据局部检查点（2026-10-22）

- `scope`：全文初读6.1—6.14与小结，局部核验token计数、去重、packing与合成过滤；整章保持进行中。
- `changes`：分开输入/监督token与shift后分母；限定URI hash和碰撞；去重区分共有hash与命中文档数；独立样本attention与跨边界loss分别处理，补packing数值例子；修复单字符答案“2”被过滤却宣称通过的问题，明确verified不是事实证明。
- `evidence`：读取TRL packing配置相关说明、去重论文2107.06499摘要；范围见 `research/encyclopedia-pretraining-data-review-2026-10-22.md`。未安装TRL或运行去重平台。
- `code_status`：5个标准库Python块独立编译执行通过；重复评估文档计数、单字符/空白答案及verified误标反例通过。packed labels为[-100,2,0,-100,0]，shift目标3，EOS监督保留。
- `formula_status`：13个math块和围栏结构通过；完整语义核验尚未结束，scaling公式明确待修正；未运行外部渲染器。
- `open_risks`：mixture、collapse、scaling、删除恢复、消融、全章来源和配套仍待验收；PDF仍受Pandoc缺失限制。
- `status`：继续第四册第六份正文；顶部next不变，第二册遗留审计保留；git diff --check通过。

### 第四册第五份正文：整章验收（2026-10-22）

- `scope`：接续局部检查点复读5.1—5.15与小结，按百科类型验收，长期规范未修改。
- `changes`：修复极端logits/温度的差值溢出；限定greedy确定性、ICL冲突分母、记忆/泛化与涌现证据；成本分类避免重试/人工重复计数，补单位成功成本；可靠性按任务合取而非边际相乘，日志保留权限与复现边界。
- `evidence`：接续Chinchilla/CE/random资料，读取Memorization和Emergence摘要；11篇论文题名/版本核对，14个唯一外链HTTP200。详见 `research/encyclopedia-llm-foundations-review-2026-10-22.md`，可达性不冒充全文核验。
- `code_status`：两个标准库Python块独立编译执行，NLL/温度/概率/全局随机流边界通过；±1e308与温度1e308的概率约0.880797/0.119203，相等超大logits和极小温度为0.5/0.5。
- `formula_status`：18个math块和围栏结构通过；6ND、PPL反例、20/80=0.25元、联合率0.5与乘积0.25复算通过，未使用外部公式渲染器。
- `sync_status`：EXERCISES.md新增5项目标/分母验收；目录、本章、练习与下一章入口有效；git diff --check通过。
- `open_risks`：未运行真实模型、训练、生产解码/安全/运营；摘要与元数据范围已注明，PDF仍受Pandoc缺失限制。
- `status`：本章声明范围内精修验收完成；继续第四册第六份正文，第二册第十份正文遗留审计保留。

### 第四册第五份正文：大语言模型基础局部检查点（2026-10-22）

- `scope`：全文初读5.1—5.15与小结；局部核验概率目标、NLL、PPL、采样与scaling，整章保持进行中。
- `changes`：明确hidden目标索引及shift后有效分母；NLL拒绝未归一化行、变动词表宽度和非法目标；区分PPL与预测熵、temperature=0接口开关；采样改局部Random且不选零概率；scaling改为有来源的N/D拟合和6ND计算约束。
- `evidence`：读取Chinchilla v1相关公式、PyTorch2.12 CE与Python random说明，见 `research/encyclopedia-llm-foundations-review-2026-10-22.md`；尚未完成全章外链/论文核验。
- `code_status`：两个标准库Python块独立编译执行通过，NLL0.458145、采样概率[0.771,0.1848,0.0443]；非法分布、bool目标、全忽略目标、零概率与全局随机状态断言通过。
- `formula_status`：18个math块和围栏结构通过；6ND=1.2×10^20FLOPs复算通过；未运行外部数学渲染器，整章语义验收尚未结束。
- `open_risks`：极端温度、记忆/泛化、涌现、ICL/成本分母、全章来源和配套仍待核验；未运行模型或训练，PDF仍受Pandoc缺失限制。
- `status`：继续第四册第五份正文，顶部next保留；第二册遗留审计仍未完成。git diff --check通过。

### 第四册第四份正文：Transformer 架构（2026-10-22）

- `scope`：全文复读4.1—4.13与小结，按百科机制、定义、反例和专题入口验收；长期规范未修改。
- `changes`：修正next-token方向、共享embedding更新、RoPE相位、dropout行和、布尔mask与缓存query绝对位置；补CPU未来/padding隔离与缓存对齐；补线性attention分母、FFN向量约定、具体gated delta纠错；限定KV复用、TTFT和batch缓存口径。
- `evidence`：读取SDPA2.12、RoFormer、Gated Delta、Performers相关段落及Kimi Linear摘要；13篇arXiv题名/版本核对。发现Kimi Linear原链接为软件安全论文，修正2510.26620为2510.26692，最终15个外链入口HTTP200。范围见 `research/encyclopedia-transformer-architecture-review-2026-10-22.md`。
- `code_status`：5个Python块独立编译执行通过；CPU PyTorch2.12.0+cpu下未来/padding隔离、切片缓存对齐、GQA重复映射通过；非方形causal反例输出3/1，维度与head整除边界通过。
- `formula_status`：24个math块结构与围栏通过；attention7.01253185、缓存4GiB、RoPE相对相位、delta2.5/2.75复算通过，未使用外部数学渲染器。
- `sync_status`：EXERCISES.md新增5项信息路径/状态验收；相对入口、专题、本册目录、练习与下一章目标存在；git diff --check通过。
- `open_risks`：局部attention示例不证明完整模型缓存正确；未运行权重、训练、GPU、FlashAttention性能、分页/多租户缓存或生产serving；元数据/HTTP200不等于全部论文全文验证；PDF仍受Pandoc缺失限制。
- `status`：本章声明范围内精修验收完成；继续第四册第五份正文，第二册遗留审计保持未完成。

### 第四册第三份正文：整章验收（2026-10-22）

- `scope`：接续局部检查点复读 3.1—3.12 与小结，按百科类型验收，长期规范未修改。
- `changes`：补 WordPiece 贪心失败与 Unigram 路径反例、Unicode 码点/字素簇和文本/任意字节边界；共享 EOS/PAD 按位置屏蔽；说明特殊 token 非自动安全隔离；修正上下文预算重复计数、causal shift 分母和不同 tokenizer PPL 比较，补 BPB 数值与条件。
- `evidence`：读取 UAX #29、Transformers PPL、BERT WordPiece 源码类、Subword Regularization v1 第3节相关段落，加上此前模板和切分资料。12 个唯一外链均 HTTP 200。详细范围及代理重试见 `research/encyclopedia-tokenization-review-2026-10-22.md`。
- `code_status`：五个标准库 Python 块独立编译执行通过；长度、id、mask、label、padding 和重叠 merge 边界通过；提取 BERT WordPiece 类验证 abc 贪心 UNK，未运行完整 BERT 或真实 tokenizer。
- `formula_status`：16 个 math 块与围栏结构通过；Unigram 边缘路径0.49、窗口预算900、PPL7.389056/2.718282与共同BPB0.721348复算通过；未调用外部数学渲染器。
- `sync_status`：EXERCISES.md 新增5项覆盖/计数验收，GLOSSARY_EN_ZH.md 同步 fertility 分母；本章、专题、练习、目录与下一章入口存在；git diff --check 通过。
- `open_risks`：未实测目标 tokenizer、模型训练、生产服务、安全或吞吐；动态文档/源码非固定commit，外链可达不等于所有论文全文核验；PDF仍受Pandoc缺失限制。
- `status`：本章声明范围内精修验收完成；继续第四册第四份正文，第二册遗留审计保留。

### 第四册第三份正文：NLP 与 Tokenization 局部检查点（2026-10-22）

- `scope`：全文初读 3.1—3.12，局部核验 BPE、模板及审计代码，整章保持进行中。
- `changes`：修正 BPE 编码/解码混淆、普通字符 BPE 不保证字节覆盖、重叠 pair 的非重叠替换；分开模板边界与训练 label mask，补重复 BOS/EOS 风险；审计代码拒绝非法负标签、非整数 mask 和被监督 padding，明确 valid_labels 尚未 shift。
- `evidence`：读取 Transformers tokenizer summary 和 chat template 对应段落，SentencePiece 模型协议说明；动态来源与未验证范围见 `research/encyclopedia-tokenization-review-2026-10-22.md`。
- `code_status`：三个标准库 Python 块独立编译执行通过；五组审计拒绝断言和 aaa 重叠 merge 反例通过，git diff --check 通过。
- `open_risks`：Unicode、WordPiece/Unigram、上下文预算、PPL、全章公式/外链及配套同步尚待验收；未运行真实 tokenizer 或模型，不能将本检查点视为整章完成。
- `status`：继续第四册第三份正文；顶部下一步保留不变，第二册遗留审计仍未完成。

### 第四册第二份正文：`book-04-llm-encyclopedia/chapters/02-深度学习组件.md`（2026-10-22）

- `scope`：全文复读 2.1—2.17 与小结，按百科类型验收，遵循当前 WRITING_SPEC.md，未修改规范。
- `changes`：修复 AMP 裁剪参数与调用，补 CPU 可执行例子；修正归一化统计、仿射输出、线性组合方向、warmup 定义域、reserved 重复计数、布尔 mask、checkpoint 浅引用和 shape；PAD/EOS 计数与 causal shift 分开；限定变长累积、数据并行组、no_sync 和 KV 复用条件。
- `evidence`：读取 PyTorch 2.12 AMP、BN、LN、DDP、Module、MHA 相关段落；18 个正文唯一外链均 HTTP 200，8 篇 arXiv 题名/版本核对。范围与限制见 `research/encyclopedia-deep-learning-components-review-2026-10-22.md`。
- `code_status`：8 个 Python 块独立编译执行通过；CPU PyTorch 2.12.0+cpu 下 AdamW 公式对照、scaler 梯度裁剪、mask 相反语义、BN running variance、LN affine、错误 shape 拒绝、checkpoint 快照与 batch 边界通过。有效 label/shift 目标/padding/pad ID 为 5/3/1/3。
- `formula_status`：34 个 math 块及围栏结构通过；复核 AdamW 首步 0.9890000004999999、梯度范数 2 与参数 0.9；变长 micro-batch 均值 2 与 token 加权 2.5 反例通过。未运行外部公式渲染器。
- `link_status`：本章相对链接与专题目标存在；PyTorch 引用固定为 2.12 文档入口。
- `sync_status`：EXERCISES.md 新增 5 项组件验收，INTERVIEW_BANK.md 同步 no_sync 与归一化要求；git diff --check 通过。
- `open_risks`：未运行 GPU/autocast、真实训练、多进程 DDP、性能或生产恢复；来源可达及元数据不等于全文验证。PDF 仍受 Pandoc 缺失限制。
- `status`：本章声明范围内精修验收完成；继续第四册第三份正文，第二册第十份正文遗留审计仍未完成。

### 第四册第一份正文：`book-04-llm-encyclopedia/chapters/01-数学与机器学习.md`（2026-10-22）

- `scope`：全文复读，按百科的定义/关系/误区/入口类型验收；核对本册简介和目录，不替代第十三册专题精修。
- `changes`：修正特征向量、SVD误差/能量、条件概率、KL、PPL与熵、似然比和Bayes更新；Dropout缩放、global batch、测试集无偏条件；precision/F1/AUC边界；PPO不保证硬约束；标准误/区间/bootstrap/McNemar/BH前提；交叉熵大数消减；LoRA参数比和成功成本分母。
- `evidence`：读取Deep Learning概率、PyTorch2.12 Dropout、NIST均值区间、Spinning Up PPO、scikit-learn指标和SciPy FDR相关材料；11个正文外链均HTTP200。来源读取范围、TLS/候选路径失败与限制见 `research/encyclopedia-math-ml-review-2026-10-22.md`。
- `code_status`：唯一标准库Python块独立编译和执行通过；空/非法target、NaN/Inf/bool、单类别与±1e20等边界通过。loss为0.939831/0.229358，相同超大logits为log(3)=1.098612。
- `formula_status`：6个math块和总围栏结构通过；贝叶斯后验0.875/0.7/0.6747、熵/PPL反例、LoRA参数比128、token平均6/11与成功成本0.25元复算通过。未使用外部数学渲染器。
- `link_status`：本章相对链接和第十三册/第五册专题目标均存在；PyTorch引用改用当前内容域，Dropout固定2.12。
- `sync_status`：EXERCISES.md新增5项反例练习，INTERVIEW_BANK.md第62题同步PPO边界；术语表已有条目不重复；git diff --check通过。
- `open_risks`：索引目标正文不因链接存在而视为已精修；未运行真实训练、GPU、统计工具库或生产评估。动态资料有版本边界，PDF仍受Pandoc缺失限制。
- `status`：本章声明范围内复读、修订、来源/代码/公式/导航与配套验收完成；继续第四册第二份正文，第二册遗留审计仍未完成。

### 第三册第十一份正文：整章验收（2026-10-22）

- `scope`：第 63—72 讲全章复读完成，接续首个局部检查点核验后六讲，遵循当前 WRITING_SPEC.md，未修改规范。
- `changes`：证据融合落实租户/ACL、关联扩展复查、空候选和图片版本；DDPM增加反向一步，说明索引、干净边界和toy非纯噪声；Whisper特征标为抽象并补实现差异；说明时间戳契约、CER/WER规范化及加权口径；规则评估改称rule_pass，加入关键词/否定误判边界；容量到达率与硬件吞吐、总成本与任务分母分开；补练习及跨册入口。
- `code_status`：最终12个Python块编译，10个标准库示例独立执行通过，2个外部接口未运行。前后两部分边界检查通过，覆盖权限、空集、关联越权、扩散shape/数值/时间步/零方差、CER>1、空参考、时间戳、规则漏检误放及零基线。未安装CUDA或下载模型。
- `formula_status`：56个math块及总围栏结构检查通过，无operatorname/尖括号历史下标；复算扩散反向均值[1.976,-0.3]、后验方差0.034063、CER0.0909/WER0.1667、输入/输出需求1536/256 token/s和单位成功成本0.25元。
- `evidence`：21个唯一外链最终HTTP200，10篇arXiv题名/版本核对；读取DDPM采样算法、Whisper音频源码和Azure security filter相关段落，加上此前CLIP/模板/模型卡来源。完整范围及未全文复核限制见 `research/multimodal-practical-review-2026-10-22.md`。
- `link_status`：目录、章节相对入口、第四册导航与练习/项目目标存在；未使用外部公式渲染器。
- `sync_status`：EXERCISES.md新增6项机制/边界验收，PROJECTS.md项目11/12补权限、关键字段及分母要求；论文索引已有主要来源，不复制摘要。git diff --check通过。
- `open_risks`：未运行真实VLM/Whisper/CLIP、训练/OCR、GPU、线上容量或安全隔离；规则盲点只被演示，没有冒充语义验证。动态来源未固定为依赖版本；PDF受Pandoc缺失限制。
- `status`：本章及第三册声明范围内精修验收完成，继续第四册第一份正文；第二册第十份正文仍保留未完成审计。

### 第三册第十一份正文：第 63—66 讲局部检查点（2026-10-22）

- `scope`：第 11 章全文初读；深入修订 CLIP 检索/损失、VLM 接口、数据构造四讲，后六讲未验收。长期规范未修改。
- `changes`：补实际打乱配对反例、多相关目标 Recall 分母、N=1 loss 退化与温度条件；跨卡标签按 logits 布局区分；VLM 真实模型入口、tokenize=False 和新增 token 截取；assistant 位置 mask；拒绝冲突标注和重复 ID；明确路径切分不等于去泄漏；补导航和公式宏/历史下标格式。
- `evidence`：读取 CLIP v1 HTML 相关机制、Transformers 多模态模板文档以及 Qwen2.5-VL-3B 模型卡相关接口示例；不是逐篇全文或本地权重验证。详见 `research/multimodal-practical-review-2026-10-22.md`。
- `code_status`：全章 12 个 Python 块编译通过，其中 10 个标准库块独立执行通过，2 个外部模型接口未运行；第 63—66 讲额外边界检查通过，后六讲尚无完整边界验收。正常 loss=0.000095，打乱=10.008890；检索打乱 Recall@1=0；数据4条、同图不跨 split。
- `formula_status`：全章机械统一 mathrm 和历史序列下标；第 63—66 讲核对对比损失、矩阵 shape、视觉预算与位置 mask。后六讲数学、数值和结构仍待整章复查，未运行外部渲染器。
- `link_status`：新增及既有本章相对链接目标存在；三项读取资料成功，整章外链尚未逐一检查。
- `sync_status`：研究记录和顶部下一步同步；题库/练习/项目等配套留待整章验收时按需补齐。`git diff --check` 通过。
- `open_risks`：后续重点是证据融合 ACL、DDPM 反向计算、Whisper 特征与时间戳、语音关键实体和规则评分误判、容量指标口径；真实模型未运行，PDF 仍受 Pandoc 缺失限制。
- `status`：局部修订及声明范围内验证完成，整章保持进行中；第二册第十份正文遗留审计仍未完成。

### 第三册第十份正文：`book-03-practical-handbook/chapters/10-面试实战训练.md`（2026-10-22）

- `scope`：全文复读 10.1—10.7，覆盖 attention、GPT、项目证据、候选决策、系统设计、论文表达和回答纪律，遵循当前 `WRITING_SPEC.md`。
- `changes`：补 cache mask 绝对位置与全 mask 边界、相同 mask 下的熵对照、shift 与 KV 复用条件；证据引用率不冒充语义支持；无合格方案返回 None；Little's Law 均值口径、ceil 精度、串行预算与分位数边界；增加在线 softmax 分子递推和独立代码；限定 Mamba 递推与 LoRA 卡片检查范围，字段完整与证据核验分开；补章节导航和练习。
- `code_status`：七个标准库 Python 块独立编译和执行通过，输入边界及不变量检查通过；CPU PyTorch 2.12.0 SDPA 对照一致，并验证 cache 显式 mask 与 upper-left causal 对齐差异。未新增依赖或安装 CUDA。
- `formula_status`：61 个 math 围栏结构检查通过；修复 qquad 转义和 operatorname；分块加权输出 2.897438、router 分数 0.5572、容量 KV 1265.62 MiB 等与示例一致。
- `evidence`：13 个唯一来源入口均 HTTP 200，六篇 arXiv 题名/版本核对；FlashAttention、Mamba、SRE 相关内容读取范围与 SDPA 跳转页限制见 `research/technical-expression-system-design-review-2026-10-22.md`，未将可达性写成全文核验。
- `sync_status`：章节前后入口及目录目标存在；练习、题库和项目文件已有相关主题，本轮以本章新增边界练习承接，不复制摘要；研究记录与顶部进度同步。
- `open_risks`：合成审计不能证明语义支持、真实模型收益、GPU 性能、生产 SLO 或安全非劣效；未运行外部公式渲染器，PDF 仍受 Pandoc 缺失限制。
- `status`：本章全文复读、修订和声明范围内的代码、公式、来源与导航检查完成；继续第三册第十一份正文，第二册第十份正文遗留审计保持未完成。

### 第三册第九份正文：`book-03-practical-handbook/chapters/09-简历项目实战.md`（2026-10-22）

- `scope`：全文复读 9.1—9.7，覆盖 miniGPT 序列审计、SFT/LoRA、DPO、服务 trace、RAG、数学评估与实验纪律；遵循当前 `WRITING_SPEC.md`，不修改长期规范。
- `changes`：补项目代码与已有训练章节的职责边界、交付目录及面试表达；修正 softmax 索引和公式宏；概率分布完整校验；声明 SFT 只监督最后一轮且 labels 供内部 shift；DPO 校验 log probability 与重复 ID；压测窗口纳入失败结束时间、全失败吞吐返回 0；RAG 空 ACL 默认拒绝、空候选安全返回、RRF 拒绝重复；数学评分保留单位、不任取过程数字、不从采样分母删除解析失败；限制算术解释器数值范围并区分候选归因与确定根因。
- `code_status`：Python 3.12.9 下七个标准库 Python 围栏均语法编译、独立运行通过；正常输出与正文一致。边界验证涵盖非法 token/概率、SFT 截断/分隔符、DPO 非法 beta/正 log probability/NaN/重复 ID、晚失败/全失败/等值容量、零向量/空 ACL/重复 RRF/无证据、错误单位/解析失败/非法表达式/pass@k/重复问题。真实训练、GPU、模型服务与外部工具未执行。
- `formula_status`：83 个 math 围栏结构平衡，围栏成对；operatorname 已统一为 mathrm，无尖括号下标、text 下划线或旧式 $$。复算 NLL 1.609438/0.223144、LoRA 参数 448、DPO loss 0.664830、QPS 1.3363 与晚失败 QPS 0.3、KV 384 MiB、合成投票和 pass@3。
- `evidence`：19 个正文唯一来源入口最终均 HTTP 200；9 个 arXiv 题名经 Atom 核准；读取 DPO 和 HumanEval/pass@k 原文相关定义，以及 TRL SFT/DPO、chat template、vLLM APC 官方相关内容。详见 `research/project-practicals-review-2026-10-22.md`；入口可达不冒充逐篇全文核验。
- `link_status`：第三册目录、本章前后入口以及相关项目/练习链接目标存在；vLLM APC 改为当前目录路径。
- `sync_status`：`PROJECTS.md` 同步产物证据范围、失败 trace 窗口、ACL 和数学采样验收；`EXERCISES.md` 增加项目边界练习。论文索引、题库和术语已有主题，无需复制摘要；`git diff --check` 通过，正文与研究笔记异常内容扫描无无关命中。
- `open_risks`：本章仅验证合成数据的审计与评估层，不证明真实训练收益、模型能力、语义引用、生产权限隔离、GPU 性能或线上可用性；main/latest 文档未固定为依赖版本。未运行外部数学渲染器，PDF 构建仍受 Pandoc 缺失限制。
- `status`：本章全文复读、机制与边界修订、公式/代码/数值检查、来源核验及按需配套同步完成；继续第三册第十份正文。第二册第十份正文的遗留审计保持未完成。

### 第三册第八份正文：`book-03-practical-handbook/chapters/08-评估与debug实战.md`（2026-10-22）

- `scope`：从头复读 8.1—8.7 全文，覆盖 benchmark、污染与泄漏、claim 事实性、训练调试、微调退化、人工评测和线上 A/B；依据当前 `WRITING_SPEC.md`，未修改长期规范。
- `changes`：分开定义回答支持率与必需事实完整性；claim 检查器使用独立政策证据；修正 FActScore 和 Cohen's Kappa 错引、回放比例方向及样本量分位数；使用稳定 logits 二元交叉熵；澄清裁剪指标、KL 方向、人工标注统计单元、SRM 判定和稀疏安全事件边界；A/B 不再按安全点估计自动扩量；补前置/后续入口、数值输出和练习。
- `code_status`：Python 3.12.9 下七个 Python 围栏全部语法编译、独立运行通过，无第三方依赖；相关边界断言通过，覆盖空集、重复 ID/标注、非法计数、零分母、NaN/Inf、否定证据误判、全平局和分桶稳定性。训练最终预测 `[0,0,0,1]`；claim r1/r2 为 1.0/0.0；A/B 解决率差约 2.39 个百分点，安全复核未通过，决策保持观察。
- `formula_status`：46 个 math 围栏结构平衡，围栏成对，无 `$$`、`operatorname`、尖括号下标或 `text` 下划线；复核支持率/完整性、Wilson、梯度与更新、KL、回放、人工偏好、二比例、SRM 和每组样本量例子。
- `evidence`：最终 20 个唯一外链中 16 个 HTTP 200，4 个 DOI 落地页 HTTP 403；四项 DOI 由 Crossref API 核准书目，不能当作出版商全文验证。10 个 arXiv 题名核准；另读取 FActScore 定义、NIST 比例区间与 Microsoft SRM 相关内容。详见 `research/evaluation-debug-practical-review-2026-10-22.md`。
- `sync_status`：第三册目录及新增相邻章节链接目标有效；`PAPERS.md` 增加评审偏差、一致性和实验方法入口，`EXERCISES.md` 增加证据分母、重复标注和安全放行练习；术语和题库已有相关主题，不复制摘要。顶部状态摘要同步第三册已有记录，保留第二册未收口缺口。
- `open_risks`：字符串 toy 不代表语义事实核验；未运行真实模型、污染检测、完整论文实验、线上 A/B 或安全非劣效验收。四项 DOI 全文访问受限；未运行外部数学渲染器。PDF dry-run 因缺少 Pandoc 失败。
- `status`：本章全文复读、机制修订、代码/边界/数值检查、来源核验与按需配套同步完成，保留上述警告；全书精修继续进入第三册第九份正文。

### 第三册第七份正文：`book-03-practical-handbook/chapters/07-rag与agent项目实战.md`（2026-10-21）

- `scope`：全文约 2,659 行，从头复读 7.1—7.8，覆盖文档清洗与切分、embedding/FAISS、reranker、引用与证据、分层评估、Tool Calling、ReAct、Agent 安全和企业政策助手贯通案例。
- `evidence`：正文去重后 34 个唯一 URL 最终均 HTTP 200；17 个唯一 arXiv ID 的元数据题名与引用主题一致；FActScore 错链由 `2305.14250` 修正为 `2305.14251`，AttributionBench 改为 ACL Anthology `2024.findings-acl.886`。完整来源与证据范围见 `research/rag-agent-practical-review-2026-10-21.md`。
- `code_status`：28 个 Python 围栏全部语法编译通过；25 个无外部模型依赖块在 `.tmp/llm-book-cpu-check`（Python 3.12.9、PyTorch 2.12.0+cpu）独立运行通过；3 个需 numpy、faiss-cpu、sentence-transformers 和模型文件的块只做语法验证，未下载模型或安装额外依赖。9 组边界断言全部通过，覆盖非法 chunk/top-k、空 gold、预算、计算器、工具异常、call id、零步 Agent、高风险动作和脱敏。
- `formula_status`：51 个 `math` 围栏结构平衡；无 `$$`、旧式数学围栏、`\\operatorname`、不安全尖括号下标或带下划线的 `\\text{...}`；选择性准确率、RAG 指标和工具状态约束已复核。
- `sync_status`：第三册目录链接有效；`PAPERS.md` 已同步 FActScore、AttributionBench、RAGAS、AgentDojo 和 ToolSandbox，并标注其数据、协议和评测环境边界；题库、练习和术语表已有对应主题，无需复制摘要。
- `open_risks`：本地 toy 和 mock 验证不代表真实 embedding/FAISS/向量数据库、LLM 生成、工具服务、浏览器/代码沙箱、付费 API、非幂等外部写操作或生产安全性能；未运行外部数学渲染器和 PDF 构建，后者受环境缺少 Pandoc 限制。
- `status`：本章全文复读、局部修订、来源核验、公式/代码/边界验证、目录和配套同步完成；全书精修继续进入第三册第八份正文。

### 在制检查点：第三册第八份正文（2026-10-22）

- `scope`：开始从头审阅 `book-03-practical-handbook/chapters/08-评估与debug实战.md`；覆盖 8.1—8.7 的 benchmark、污染与泄漏、事实性、训练调试、微调退化、人工评测和线上 A/B。
- `next`：逐节检查结构、公式定义域、Python 围栏的独立执行与边界、来源入口和第三册目录映射；完成后再登记研究记录和验收结果。
- `status`：进行中，不代表本章完成。

#### 第三册第六份正文：`book-03-practical-handbook/chapters/06-推理优化实战.md`

- `scope`：全文约 1,907 行，从头复读，覆盖 6.1—6.7：KV Cache 形状/显存/RoPE offset/prefix cache 边界、Prefill/Decode 指标与 CUDA 计时、vLLM PagedAttention/连续批处理/参数与协议边界、量化三本账、Speculative Decoding、长上下文压测和实验报告模板；第三册目录映射正确。
- `changes`：5 处 `\operatorname{...}` 统一为 `\mathrm{...}`；NVIDIA Transformer Engine FP8 primer 链接由失效的 `user-guide/examples/` 路径更新为当前 `examples/fp8_primer.html`；vLLM engine arguments 入口由 `serving/engine_args.html` 更新为当前 `configuration/engine_args/`。
- `code_status`：11 个 Python 围栏语法编译全部通过；8 个无依赖块在 `.tmp/llm-book-cpu-check`（Python 3.12.9、PyTorch 2.12.0+cpu）逐块独立执行通过，输出与正文一致（KV 容器 prefill/decode 区间、RoPE 旋转对、延迟账本 p95=63.3ms、block allocator 引用计数、分组量化 MSE 随 group size 下降、greedy 验证接受长度、位置检索行、压测客户端契约）。CUDA 计时块前置守卫单独验证（无 CUDA 时抛 RuntimeError）；QLoRA 与 assisted generation 块因缺少 `transformers`/CUDA 未执行，正文已声明依赖。未安装 CUDA。
- `formula_status`：60 个 `math` 围栏花括号/方括号平衡；`\operatorname` 清零，无 `$$`、旧式围栏、`\text{...}` 下划线或尖括号下标。正文数值例子独立复算一致：KV Cache 1 GiB / MHA 4 GiB、block 2 MiB 与 1 GiB 池约 512 block、7B 权重 13.0/6.5/3.3 GiB。
- `link_status`：36 个唯一外链经指定代理最终均 HTTP 200；arXiv Atom API 核对 16 个 arXiv 引用题名全部与引用主题一致；第三册目录链接有效。
- `sync_status`：`PAPERS.md` 已含本章全部论文入口（PagedAttention、Speculative Decoding、Medusa/EAGLE、GPTQ/AWQ/SmoothQuant/KIVI、Lost in the Middle/LongBench/RULER），题库/练习/术语表已覆盖 KV Cache、推测解码与长上下文主题，无需重复；新增研究记录 `research/inference-optimization-practical-review-2026-10-20.md`；`git diff --check` 通过。
- `open_risks`：本地 CPU toy 验证不代表 CUDA/GPU kernel、FlashAttention 数值、vLLM/SGLang/TensorRT-LLM 调度、量化质量回归、真实接受率或长上下文实测；p95 分位数算法与压测矩阵是测量定义，数字须来自目标 revision/硬件/负载的成对实验。PDF dry-run 因缺少 Pandoc 未运行。
- `status`：本章全文复读、修订、代码输出核验、数值复算、公式结构检查、36 个来源入口与 16 个 arXiv 题名核验及配套同步完成。全书精修继续进入第三册第七份正文。

### 在制检查点：第三册第六份正文（2026-10-20）

- `scope`：开始从头审阅 `book-03-practical-handbook/chapters/06-推理优化实战.md`；章节约 1,900 行、7 个主体部分，目录映射正确。首轮读完引言、6.1 KV Cache 和 6.2.1—6.2.4 指标/异步计时。
- `structure_status`：识别 KV cache 形状/显存、Prefill/Decode、vLLM PagedAttention、量化、Speculative Decoding、长上下文压测和实验报告等部分；代码、公式、来源及配套审计尚未完成。
- `preliminary_review`：KV 字节账本、MHA/GQA/MQA 的 KV-head 关系、RoPE decode offset 和异步 CUDA Event 计时逻辑初读一致；尚未进行独立数值/代码核验。
- `next`：继续从 6.2.5 开始审阅，重点核验吞吐/分位数定义、异步流同步契约、连续批处理与 vLLM 参数，再按顺序覆盖余下章节。
- `status`：进行中，不代表章节完成。

#### 第三册第五份正文：`book-03-practical-handbook/chapters/05-偏好优化实战.md`

- `scope`：从头复读 5.1—5.6，覆盖偏好数据、Reward Model、DPO 手写目标、TRL、LoRA/QLoRA 和评估；第三册目录映射正确。
- `changes`：5.3.4 改为 CPU toy backbone 和内存 tokenized pair，完成训练/评估/checkpoint 闭环；checkpoint 写入 `.tmp/`。5.4.6 标明 reference 代码所需的 `ref_model` 前置变量；5.5.2 按实时 TRL 文档说明 `processing_class`/`eval_strategy`、平均 token logprob 与 response logprob 求和的尺度差异，并限定固定 max_length 文本示例不应直接用于可能切断图像 token 的 VLM 样本。
- `code_status`：44 个 Python 围栏语法检查通过；20 个无外部上下文片段独立运行通过；组合 CPU torch、Reward Model 训练/评估/checkpoint 和 DPO sequence logprob 合约测试通过。环境为 `.tmp/llm-book-cpu-check`、Python 3.12.9、PyTorch 2.12.0+cpu；无模型下载或 CUDA 安装。
- `formula_status`：46 个 `math` 围栏花括号/方括号结构检查通过；`git diff --check` 通过。
- `evidence_status`：指定代理下 16 个章节来源入口均 HTTP 200；PyTorch 链接重定向到 `docs.pytorch.org` 后成功。实时读取 TRL DPO 文档并核对当前接口、偏好数据格式、logged metrics 和 VLM 截断提示；arXiv API 核对 DPO、InstructGPT、PPO、摘要偏好和偏好理论共 5 项元数据/摘要，均与引文主题一致。证据边界见研究记录。
- `sync_status`：`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`INTERVIEW_BANK.md` 已覆盖本章主要主题，无需复制；第三册目录映射正确。
- `open_risks`：未执行真实模型/TRL/PEFT 端到端训练或 GPU/QLoRA；未逐篇阅读全文、附录和完整实验 harness。教学验证限于本地 CPU toy 行为。
- `status`：本章全文复读、来源与链接核验、公式结构、CPU 代码及配套索引审计完成；全书精修继续进入第三册第六份正文。

### 在制检查点：第三册第一份正文首轮代码审计（2026-10-19）

- `scope`：开始第三册 `book-03-practical-handbook/chapters/01-pytorch基础实战.md`；完成文件结构盘点和 23 个 Python 块独立执行，尚未完成逐段全文复读、公式核对、来源审计和配套同步。
- `code_status`：使用仓库 `.tmp/llm-book-cpu-check` 的 CPU-only PyTorch 2.12.0 环境执行，23/23 块通过；系统 Python 缺少 PyTorch，不作为章节失败证据。未安装 CUDA 或额外依赖。
- `evidence_status`：章节已有 PyTorch Autograd、CrossEntropyLoss、logsumexp、优化器和 scheduler 官方文档链接；本批通过指定代理核验 Autograd 和 CrossEntropyLoss 链接返回 HTTP 301（重定向到 `docs.pytorch.org`），Hugging Face optimizer/scheduler 页面返回 HTTP 200，重定向本身未改变链接可访问性。
- `changes`：正文补充本章示例按 CPU 设计、不要求 CUDA/GPU，并说明随机初始化、版本和设备变化时应优先核对形状、梯度关系和趋势。未改变任何 CUDA 或 GPU 依赖。
- `changes`：发现 checkpoint 示例直接写仓库根目录会留下生成物，已改为写入 Git 忽略的 `.tmp/pytorch-chapter-checkpoint.pt`，并补充真实项目应配置持久化路径的说明。
- `open_risks`：第 1.1—1.6 节仍待逐段复读；代码输出与正文预期、公式/数值例子、API 版本边界、其余来源和目录/索引同步尚未完成。

### 回归检查点：第三册第一份正文（2026-10-19）

- `code_status`：补充运行前提和 checkpoint 路径后，从当前正文重新提取并独立执行 23 个 Python 块，全部通过；环境为 `.tmp/llm-book-cpu-check`、PyTorch 2.12.0+cpu，未安装 CUDA；执行后清理临时 checkpoint。
- `sync_status`：`git diff --check` 通过；本章仍处于在制状态。

### 结构与公式检查点：第三册第一份正文（2026-10-19）

- `formula_status`：6 个小节（1.1—1.6）顺序与文件结构一致；44 个 `math` 围栏花括号和方括号计数平衡，未发现旧式 `\\[`/`\\]` 或 `$$` 围栏。
- `link_status`：正文识别到 8 个外部资料入口；PyTorch 官方链接本批核验到重定向或成功响应，Hugging Face scheduler 页面返回 HTTP 200。其余论文/页面内容和固定版本范围仍待逐链接核对。
- `open_risks`：尚未完成本章每个代码块与紧邻预期输出逐项比对、API 版本差异复核及简介/目录/索引同步；不能标记本章完成。
- `changes`：1.2.3 的训练循环改为最后一次参数更新后重新计算 loss，使 loss、预测与准确率对应同一参数状态；按 PyTorch `CrossEntropyLoss` 对类别索引目标的定义，补充 class weight 存在时 `reduction="mean"` 分母为有效目标权重和。官方文档链接统一为 `docs.pytorch.org` 当前内容域。
- `evidence_status`：PyTorch CrossEntropyLoss 与 LambdaLR 官方文档通过指定代理返回 HTTP 200；Autograd 与 CrossEntropyLoss 旧域返回 301 指向 `docs.pytorch.org`，故正文入口更新到最终内容域。
- `code_status`：更新后的 MLP 示例实测输出为预测 `[0, 0, 1, 1, 2, 2]`、最终 loss `0.0869`、准确率 `1.0`，三项来自同一最终参数状态；全章 23 个代码块重新执行通过。
- `changes`：checkpoint 示例新增 `Path(".tmp").mkdir(exist_ok=True)`，确保新环境中目标目录不存在时仍可执行；此前误放入其他代码块的导入已纠正。
- `code_status`：修正后重新执行 23 个 Python 块通过，并清理 `.tmp/pytorch-chapter-checkpoint.pt`；`git diff --check` 通过。

### 检查点：第三册第二份正文首轮验收（2026-10-20）

- `scope`：开始复读 `book-03-practical-handbook/chapters/02-transformer组件实战.md`；文件 1,979 行，覆盖 embedding/位置表示、scaled dot-product attention、多头 attention、mask、Transformer block、RoPE、RMSNorm 和 SwiGLU。
- `changes`：为 2.5.5 和 2.7.6 的 decoder block 加入缺少 `keep_mask` 时的显式错误，避免调用方静默运行双向 attention；正文说明因果 keep mask 的调用契约，并补充 PyTorch LayerNorm、Dropout 官方文档入口。
- `evidence`：正文 10 个唯一入口经指定代理检查最终均 HTTP 200；来源类型与支持边界见 `research/transformer-components-practical-review-2026-10-20.md`。
- `code_status`：当前正文 22 个 Python 块在 `.tmp/llm-book-cpu-check`、PyTorch 2.12.0 CPU 环境独立执行通过；SDPA 对照和形状/不变量输出符合正文。两个 decoder block 分别通过漏传 mask 的负向检查。未安装 CUDA。
- `formula_status`：68 个 `math` 围栏花括号和方括号计数平衡；复核 attention 缩放/归一化、MHA shape/复杂度、causal 与 padding mask 组合、Pre-LN 残差、RoPE 相对位置、RMSNorm 和 SwiGLU 参数量。未运行外部数学渲染器。
- `link_status`：10 个唯一外链最终均 HTTP 200；第三册目录指向的章节文件存在。
- `sync_status`：简介与目录映射无需修改；新增研究记录 `research/transformer-components-practical-review-2026-10-20.md`。
- `open_risks`：本地 CPU toy 验证不代表 GPU kernel/FlashAttention 性能、模型训练质量或长上下文外推；不同 RoPE 布局和 runtime padding/position 协议须按目标模型核验。PDF dry-run 因缺少 Pandoc 失败，未安装依赖。
- `status`：本章全文复读、局部修正、代码输出和边界核验、公式结构检查、来源入口与导航核对完成；全书精修继续进入第三册第三份正文。

### 检查点：第三册第三份正文首轮验收（2026-10-20）

- `scope`：开始复读 `book-03-practical-handbook/chapters/03-从零训练小gpt.md`；文件约 1,798 行，覆盖 token 数据契约、最小 GPT、解码、checkpoint、曲线诊断、BPE 迁移和公平比较。
- `changes`：日志示例统一写到 `.tmp/`；BPE FLOPs/参数账本拆成不同单位的指标并标注差值方向；checkpoint step 示例补齐运行变量、`.tmp/` 保存路径和可观察的 step/update 输出。
- `evidence`：12 个 PyTorch、Hugging Face 与论文入口经指定代理检查最终均 HTTP 200；来源范围见 `research/tiny-gpt-practical-review-2026-10-20.md`。
- `code_status`：17 个 Python 块中 15 个在 PyTorch 2.12.0 CPU 环境按章节依赖顺序执行通过；checkpoint 循环输出保存 step `[0, 1, 2]`、optimizer updates `2`，BPE 演示的 roundtrip 与 shift 断言通过。两段 Hugging Face tokenizer 示例因缺少 `transformers` 和 tokenizer 缓存未执行，正文有依赖说明。未安装 CUDA。
- `formula_status`：41 个 `math` 围栏花括号和方括号计数平衡；复核 next-token 分解/shift、attention 因果条件、token loss/PPL、top-k/top-p 集合、checkpoint step、BPE 序列/词表指标和 token budget 口径。
- `link_status`：12 个唯一来源入口最终均 HTTP 200；第三册目录映射正确。
- `sync_status`：`PAPERS.md` 已有 BPE、采样和训练论文入口，本轮无需重复增加；新增研究记录 `research/tiny-gpt-practical-review-2026-10-20.md`。
- `open_risks`：未运行依赖 Hugging Face tokenizer artifact 的两段代码、外部数学渲染器或目标数据上的真实训练；教学语料、手动 merge、toy 曲线和 CPU 输出不代表 tokenizer 优劣、模型能力或 GPU/生产效果。PDF dry-run 因缺少 Pandoc 失败，未安装依赖。
- `status`：本章全文复读、修订、可运行代码输出/边界、来源入口、公式结构与目录同步完成；全书精修继续进入第三册第四份正文。

#### 第三册第四份正文：`book-03-practical-handbook/chapters/04-huggingface微调实战.md`

- `scope`：全文约 2,500 行，从头复读，覆盖 checkpoint/生成、SFT 数据与 mask、全参数训练、LoRA、QLoRA 和行为评估；第三册目录映射正确。
- `changes`：更新 PyTorch CrossEntropyLoss 文档入口；说明 `sshleifer/tiny-gpt2` 只验证接口流程，不支持任务质量或微调收益结论。
- `evidence`：10 个 Transformers、PEFT、bitsandbytes、TRL、PyTorch 与论文入口经指定代理检查均 HTTP 200；来源和支持范围见 `research/huggingface-finetuning-practical-review-2026-10-20.md`。入口可达不等于逐篇全文复核。
- `code_status`：48 个 Python 围栏通过语法编译；6 个纯 Python/mock 教学实验和 CPU collator 契约检查通过。当前环境没有 Transformers、PEFT/bitsandbytes；未下载模型或执行真实外部训练器代码，未安装 CUDA。
- `formula_status`：40 个 `math` 围栏花括号/方括号结构检查通过；未运行外部数学渲染器。
- `sync_status`：`PAPERS.md` 增补 QLoRA 原论文入口；新增研究记录。简介和目录无需修改；`git diff --check` 通过。
- `open_risks`：验证限于局部教学逻辑和 CPU 数据契约，不证明真实模型效果、GPU 性能、量化收益或论文复现；PDF dry-run 未运行。
- `status`：全文复读、局部修订、声明范围内的代码和公式结构验证、来源与导航核对完成。全书精修继续进入第三册第五份正文。

#### 第三册第一份正文：`book-03-practical-handbook/chapters/01-pytorch基础实战.md`

- `scope`：覆盖第 1.1—1.6 节，从首行至延伸阅读完整复读；第三册目录映射正确，简介与目录中的书册定位一致。
- `changes`：修复 1.5.5 的不完整引导句；把梯度累积示例的输出解释明确为 5 个 micro-batch 按 4+1 分组、共更新两次；将 AdamW 原论文补为可点击直链，并同步到 `PAPERS.md`。
- `evidence`：通过指定代理检查 10 个来源入口（PyTorch Autograd、CrossEntropyLoss、logsumexp、SGD、Adam、AdamW、scheduler guide，Hugging Face optimizer schedules，Adam 与 AdamW 原论文），最终目标均 HTTP 200。来源支持范围和限制见 `research/pytorch-basics-practical-review-2026-10-20.md`；HTTP 状态不代表逐篇全文和源码审计。
- `code_status`：当前正文 23 个 Python 块在 `.tmp/llm-book-cpu-check`、PyTorch 2.12.0 CPU 环境逐块独立执行通过；各段实测输出与正文示例对照一致。MLP 最终 loss 与预测/准确率来自同一组更新后参数；梯度累积输出 2；checkpoint 写入 `.tmp/` 并已清理。未安装 CUDA。
- `formula_status`：44 个 `math` 围栏，花括号和方括号计数平衡；复核 MSE 梯度、交叉熵稳定形式与类别梯度、类别权重 mean 分母、有效 token loss、Adam 矩与偏差修正、AdamW 简化更新、梯度累积归一化、warmup 和 scheduler 计数语义。未运行外部数学渲染器。
- `link_status`：正文 10 个唯一来源入口最终均 HTTP 200；AdamW 原论文入口已补全。第三册目录相对链接有效。
- `sync_status`：`PAPERS.md` 训练方向增加 Adam/AdamW 原论文入口；简介和目录不需要改动。新增研究记录 `research/pytorch-basics-practical-review-2026-10-20.md`。
- `open_risks`：验证限于本地 CPU toy 代码，不代表 GPU/CUDA 数值、真实模型训练、生产性能或论文复现；scheduler 输出绑定 PyTorch 2.12.0 调用顺序。PDF dry-run 因缺少 Pandoc 未运行成功，未安装依赖。
- `status`：本章全文复读、修订、代码输出核验、来源入口检查、公式结构检查及配套同步完成；结果限于记录的来源和本地环境。全书精修继续进入第三册第二份正文。

### 在制检查点：第二册第十份正文第 124 讲（2026-10-19）

- `scope`：完成第 124 讲首轮全文复读和局部代码修订；本讲不是整章完成证据。
- `changes`：研究方案评分器增加 proposal 字段完整性、分数范围和 baseline/ablation 布尔值检查，避免评分器把缺失字段、布尔值或越界分数静默纳入排序。正文明确软分只用于排序，必要条件和高风险约束不能被平均分抵消。
- `code_status`：第 124 讲 1 个 Python 示例正常输出；另通过 2 项评分输入边界检查。代码只演示研究方案筛选，不自动判断真实研究价值。
- `evidence_status`：本讲列出 CLIP、Flamingo、BLIP-2、MME、MMBench、MMMU、POPE、LayoutLMv3、FID、FVD、CLIPScore、Sora、NIST、OWASP、GPT-4o System Card、Model Spec、Preparedness Framework 和 OpenAI Evals 入口；统一 HTTP/内容核验尚未完成。
- `open_risks`：第 108—124 讲的来源、公式/代码全章审计、配套索引和研究笔记仍待统一完成；第二册第十份正文仍保持进行中。

### 回归检查点：第二册第十份正文第 105—124 讲示例（2026-10-19）

- `code_status`：从当前正文重新提取并独立执行全部 20 个 Python 块，全部通过；本回归只证明示例在当前文本状态下可运行，不替代逐讲来源、公式和边界审计。
- `sync_status`：`git diff --check` 通过；第十章整章来源核验、研究笔记和配套索引仍待统一处理。

### 来源与格式审计尝试：第二册第十份正文（2026-10-19）

- `link_status`：通过指定代理单独请求 `https://arxiv.org/abs/2306.13394` 返回 HTTP 200；批量子进程请求本次均返回 HTTP 000，无法据此判断其余链接失效，待逐链接重试。正文当前共识别 32 个外部 URL。
- `formula_status`：已将第 117 讲历史段的旧式 `\\[`/`\\]` 数学围栏机械转换为 `math` 围栏；当前全章旧式围栏命中为 0，`git diff --check` 通过。转换后全章 Python 20 块再次独立执行通过；仍待实际 Markdown 渲染检查。
- `open_risks`：来源内容核对、剩余链接、旧式公式围栏转换、目录/索引同步和研究笔记仍未完成；不得将第十章标为完成。

### 在制检查点：第二册第十份正文第 123 讲（2026-10-19）

- `scope`：完成第 123 讲首轮全文复读和局部代码修订；来源核验与整章验收仍待完成。
- `changes`：多模态评估示例不再用 `max(1, n)` 掩盖空分母；补充空样本、无 claim、无 grounding 样本、非法 box、空工具集、阈值和混淆计数检查。全零混淆矩阵按协议返回未定义 precision/recall/F1。正文继续区分 benchmark、claim 支持、人工评估和线上风险指标。
- `code_status`：第 123 讲示例输出与正文一致；另通过 5 项输入边界检查。示例仅为 6 条人工样本，不代表真实模型或上线安全结论。
- `evidence_status`：MME、MMBench、MMMU、POPE、FID、FVD、CLIPScore、NIST AI RMF 和 OWASP 入口已在正文列出；本批未取得新增代理响应，来源范围仍待统一核验。
- `open_risks`：第 124 讲尚待逐段审阅，第 108—123 讲来源、研究笔记和整章配套同步仍未完成。

### 在制检查点：第二册第十份正文第 122 讲（2026-10-16）

- `scope`：完成第 122 讲首轮全文复读和局部代码修订；来源核验与整章验收仍待完成。
- `changes`：端到端语音示例补充 speech token、codec、RTF、准确率、上下文、打断检测和综合分目标的定义域检查；空样本不再被 `max(1, n)` 伪装成有效指标。正文明确 TTFA 时间原点、播放缓冲、barge-in 状态一致性和端到端不等于内部无模块。
- `code_status`：第 122 讲示例输出与正文一致；另通过 7 项输入边界检查。未运行真实 speech-to-speech 模型或实时音频链路。
- `evidence_status`：AudioLM、VALL-E、AudioPaLM、SpeechGPT、Moshi 和 GPT-4o System Card 等资料在正文中作为路线入口；本批未取得新增代理响应，公开资料与推断范围已分开表述。
- `open_risks`：第 123—124 讲尚待逐段审阅，第 108—122 讲来源与整章配套同步仍未完成。

### 在制检查点：第二册第十份正文第 121 讲（2026-10-16）

- `scope`：完成第 121 讲首轮全文复读和局部代码修订；来源核验与整章验收仍待完成。
- `changes`：TTS/voice cloning 示例补充 duration 对齐与正速度、误差向量形状、声纹余弦输入、WER 空参考、codec 参数、MOS 均值、EER genuine/impostor 样本及安全请求字段检查，避免空分母或静默截断。正文继续将 voice cloning 授权、水印和滥用控制与音色相似度分开。
- `code_status`：第 121 讲示例输出与正文一致；另通过 7 项输入边界检查。未合成真实音频、未运行声码器或声纹模型。
- `evidence_status`：Tacotron 2、FastSpeech、FastSpeech 2、VITS、VALL-E 等原论文入口已在正文列出；本批未取得新增代理响应，论文实验不外推为产品能力。
- `open_risks`：第 122—124 讲尚待逐段审阅，第 108—121 讲来源与整章配套同步仍未完成。

### 在制检查点：第二册第十份正文第 120 讲（2026-10-16）

- `scope`：完成第 120 讲首轮全文复读和局部代码修订；来源核验与整章验收仍待完成。
- `changes`：补上示例遗漏的 `math` 导入；WER/CER 空参考文本改为显式拒绝，避免与正文声明冲突；VAD 输入和时间区间要求有效定义域，区间 IoU 不再对零长度区间静默返回 0。
- `code_status`：第 120 讲示例现可独立执行，输出与正文一致；另通过 4 项输入边界检查，覆盖空 WER/CER 参考、空 VAD 输入和零长度时间区间。
- `evidence_status`：Whisper 原始论文链接及训练数据范围说明保留；本批未取得新增代理响应，具体实现行为仍以对应版本代码/配置为准。
- `open_risks`：第 121—124 讲尚待逐段审阅，第 108—120 讲来源与整章配套同步仍未完成。

### 在制检查点：第二册第十份正文第 119 讲（2026-10-16）

- `scope`：完成第 119 讲首轮全文复读和局部代码修订；来源核验与整章验收仍待完成。
- `changes`：世界模型审计示例拒绝空均值、距离向量不等长、无效时间步/间隔、空事件约束和空控制转移，避免 `max(1, n)` 将未定义指标伪装成分数；对象运动阈值也要求为正有限数。正文继续区分 open-loop 视频生成与 closed-loop 交互世界模型，并限定 Sora 未公开细节。
- `code_status`：第 119 讲示例输出与正文一致；另通过 5 项输入边界检查，覆盖空均值、距离形状、物理轨迹长度、空事件约束和空控制列表。
- `evidence_status`：正文引用 Sora 世界模拟器公开页面，但本批未重新取得代理响应；未公开架构、训练数据和安全策略未作确定断言。
- `open_risks`：第 120—124 讲尚待逐段审阅，第 108—119 讲来源与整章配套同步仍未完成。

### 在制检查点：第二册第十份正文第 118 讲（2026-10-16）

- `scope`：完成第 118 讲首轮全文复读和局部代码修订；来源核验与整章验收仍待完成。
- `changes`：视频示例不再用 `max(1, n)` 掩盖空输入，不再用 `zip` 静默截断不等长向量；补充 latent 尺寸整除、attention patch、时序帧数和特征向量宽度检查。正文区分概念 shape 与 PyTorch 常见布局，并限定 toy FVD 不能与真实 FVD 数值比较。
- `code_status`：第 118 讲示例输出与正文一致；另通过 5 项输入边界检查，覆盖空均值、不等长 MSE、latent 尺寸不整除、patch 不整除和单帧时序评估。
- `evidence_status`：Video Diffusion Models 论文入口已在正文给出，本批未取得新增代理响应；FVD 和视频路线结论仍绑定各自评估协议。
- `open_risks`：第 119—124 讲尚待逐段审阅，第 108—118 讲来源及全章配套同步仍未完成。

### 在制检查点：第二册第十份正文第 117 讲（2026-10-16）

- `scope`：完成第 117 讲首轮全文复读和局部代码修订；来源核验与整章验收仍待完成。
- `changes`：VQ/自回归示例补充 codebook、特征距离、NLL 概率和下采样整除检查，避免 `zip` 截断、`max(p, epsilon)` 掩盖非法概率或尺寸不整除时静默取整。正文保留 DALL·E 1/2 的历史路线范围限定。
- `code_status`：第 117 讲示例输出与正文一致；另通过 3 项输入边界检查，覆盖非法概率、距离向量长度不一致和下采样不整除。未运行真实 tokenizer 或 Transformer。
- `evidence_status`：正文已含 DALL·E 1/2 原论文入口，但本批未通过代理重新核验；论文结论仍限于各自公开方法与实验。
- `open_risks`：第 118—124 讲尚待逐段审阅，第 108—117 讲来源与整章配套同步仍未完成。

### 在制检查点：第二册第十份正文第 116 讲（2026-10-16）

- `scope`：完成第 116 讲首轮全文复读和局部代码修订；来源核验与整章验收仍待完成。
- `changes`：补充 cross-attention key/value 宽度和 inpainting mask 与 latent 向量的形状检查，避免示例因广播或截断而掩盖接口错误。正文明确 latent 元素比不是端到端加速比，latent scaling、prediction type、scheduler 和 checkpoint 必须作为版本契约记录。
- `code_status`：第 116 讲示例输出与正文一致；另通过 2 项形状边界检查，覆盖 attention value/key 宽度和 mask/latent 长度不一致。未运行真实 VAE、U-Net、ControlNet 或 GPU。
- `evidence_status`：本批未取得新增代理响应；LDM、Stable Diffusion、LoRA、DreamBooth、Textual Inversion、ControlNet 和编辑方法来源仍待核验。
- `open_risks`：第 117—124 讲尚待逐段审阅，第 108—116 讲来源和全章配套同步仍未完成。

### 在制检查点：第二册第十份正文第 115 讲（2026-10-16）

- `scope`：完成第 115 讲首轮全文复读和局部代码修订；来源核验与整章验收仍待完成。
- `changes`：扩散教学代码对 CFG scale、延迟输入补充有限数和布尔值检查，避免非有限配置静默进入成本与采样计算。公式中已明确 schedule、prediction type、sampler 和 latent scaling 必须作为接口契约共同记录，并保留 DDIM 公式的简化范围说明。
- `code_status`：第 115 讲 1 个 Python 示例正常输出与正文一致；另通过 2 项非有限 guidance 输入检查。示例只验证一维向量和成本代理，不代表真实 U-Net、DiT、GPU 或 sampler 结果。
- `evidence_status`：本批未取得新增论文网络响应；DDPM、DDIM、CFG、DiT 等引用仍待指定代理恢复后核验。
- `open_risks`：第 115 讲来源、训练/推理配置的版本边界和第 116—124 讲内容仍待审阅。

### 在制检查点：第二册第十份正文第 114 讲（2026-10-16）

- `scope`：完成第 114 讲路线级首轮复读；未完成来源访问与整章验收。
- `changes`：本批未需正文修改；核对 VAE、GAN、Flow、自回归、Diffusion、Latent Diffusion、CFG、ControlNet 和生成评估的公式口径及定义域，确认生成代码的教学范围声明未越界。
- `code_status`：第 114 讲 1 个 Python 示例独立执行成功，输出与正文一致；覆盖 VAE 重建/KL、GAN 二元损失、Diffusion 噪声 MSE、视觉 token 数和 latent 元素压缩比。未运行真实生成模型或 CUDA。
- `evidence_status`：本批尝试通过指定代理获取生成模型论文 Atom 资料，代理连接失败（curl 7，HTTP 000）；不得据此判定论文失效，来源核验待网络恢复后进行。
- `open_risks`：第 114 讲引用的 VAE、GAN、RealNVP、VQ-VAE、DALL-E、DDPM、LDM、ControlNet、FID、CLIPScore 来源尚待核验；第 115—124 讲尚待逐段审阅。

### 在制检查点：第二册第十份正文第 113 讲（2026-10-16）

- `scope`：完成第 113 讲首轮全文复读和局部修订，尚未完成来源核验和全章验收。
- `changes`：点包含判断与半开区间坐标协议统一，右边界和下边界不计为框内；pointing 主指标不再排除预测缺失/解析失败样本，另报有效输出条件指标时须报告有效比例；短语匹配明确一对一计数。补充空间关系阈值等号语义、非 mapping 样本的输入错误提示，并收紧单样本 IoU 达标的归因措辞。
- `code_status`：第 113 讲示例独立执行通过，原预期输出不变；9 项检查覆盖点的四边界/框外、同框和无交集 IoU、空间关系阈值等号及非法样本类型。`git diff --check` 通过。
- `open_risks`：Grounding DINO、Kosmos-2、Shikra 和 RefCOCO 系列来源及相关机制仍待核验；本批仅验证手工几何和评估代码，未运行真实模型。第 114—124 讲尚待逐段审阅。

### 在制检查点：第二册第十份正文第 108—112 讲（2026-10-16）

- `scope`：延续第 108—110 讲局部复读，审阅第 111—112 讲并修订教学代码；尚未完成这些讲次的来源核验和整章验收。
- `changes`：补充 Q-Former 训练目标、冻结 LLM 时的梯度路径及 Flamingo resampler/mask 解释，机制细节仍待原文核验。第 111 讲将 DPO logistic 输出改称偏好概率，使用稳定 softplus，并明确 log-prob 聚合口径。第 112 讲修正缺失键与 `None` 混淆、无 TP 时 F1 错误返回未定义、图表年份集合不一致触发 KeyError 的问题；单列零参考值绝对误差，补充 OCR 规范化、空参考协议及单头单层 attention 统计口径。
- `code_status`：本检查点从当前正文提取并独立执行全章 17 个 Python 块，全部通过；第 112 讲另通过 10 项边界断言，覆盖空字段、缺失/显式空值、零分 F1、零参考图表值、年份集合不一致、空 OCR 参考及 CER 大于 1。仅验证教学示例，不代表剩余讲次已经审阅。
- `evidence_status`：此前经指定代理取得 BLIP、BLIP-2、Flamingo、LLaVA 的 arXiv Atom 响应（HTTP 200，`.tmp/ch10-bridge-arxiv.xml`），尚未解析核对其内容或用论文原文验证新增机制；后续页面访问连接失败不能视作论文失效。
- `open_risks`：第 113—124 讲尚待逐段审阅；第 108—112 讲来源与全文验收、配套索引和全章研究记录待完成。整章及全书均保持进行中。

### 在制检查点：第二册第十份正文第 108—110 讲局部修订（2026-10-14）

- `scope`：继续审阅第 108—110 讲的 connector、架构比较和多模态 SFT；本批为局部修订，不作为这些讲次全文验收或全章完成证据。
- `changes`：修正 Flamingo 成本公式及教学代码：视觉 latent 不拼入文本 self-attention 序列，文本通过层内 cross-attention 读取 latent，并单列 resampler 的视觉/latent key-value 成本。明确只统计所列层的 dense score cell，完整网络仍需计入各自层数与 Q-Former self-attention。第 110 讲 mask 改为 assistant 目标位置集合，补充整数 labels、next-token shift、接口内部 shift 和视觉 embedding 展开后的 mask 对齐说明。
- `code_status`：修订后的第 108—110 讲三个标准库 Python 示例均在本地 Python 下执行成功，输出与正文一致；未安装依赖、未使用 CUDA。本批运行验证正常路径，边界复核尚未完成。
- `formula_status`：修正 Flamingo 的文本长度、融合层 query/key 长度和 resampler 成本项；修正 SFT mask 的位置语义。本批未完成全章公式审计。
- `evidence` / `link_status`：沿用已有论文入口作为待复核来源；本批未新增来源核验成功证据。上一轮代理连接在沙箱内失败，提权请求后的结果因中断尚未确认；不得据此宣称来源不可达或已核验。
- `sync_status`：本批修改正文和本进度检查点；配套索引待来源与全章审阅后统一检查。`git diff --check` 通过。
- `open_risks`：第 108—110 讲全文复读验收、来源访问、边界检查仍待完成；第 111—124 讲尚未纳入本批覆盖。全书目标保持进行中。

### 在制检查点：第二册第十份正文第 105—107 讲（2026-10-08）

- `scope`：完成第 105—107 讲的首轮全文复读与局部修订；全章第 108—124 讲、剩余来源复核、总体验收和研究笔记仍待完成。
- `changes`：第 105 讲明确语音与一般音频的子集关系、attention cell 统计口径和 indicator 函数；第 106 讲的 CLIP toy batch 改用显式图文 pair ID、补齐向量/温度/模板边界，并把 prompt ensemble 效果限定为需在目标集上验证；第 107 讲补 ViT/ConvNeXt/SigLIP 原论文直达引用，明确 score-cell 口径、卷积参数假设和 ConvNeXt 参数统计边界，并使用稳定的负 log-sigmoid 计算 SigLIP 教学 loss。
- `code_status`：第 105—107 讲共 3 个纯标准库 Python 块，输出均与正文一致；34 项局部边界和不变量检查通过。环境为 `.tmp/llm-book-cpu-check` Python 3.12.9；未使用 PyTorch 或 CUDA。
- `evidence_status`：arXiv Atom API 已核对 CLIP、ViT、ConvNeXt 和 SigLIP 的标题/摘要。代理对 GitHub CLIP、ConvNeXt 和 SigLIP 页面返回 200；arXiv 的 CLIP、ViT 页在本次 HTTP 检查中遇到 TLS EOF，需全章链接复查后记最终状态。
- `sync_status`：`PAPERS.md` 的多模态入口补充 ConvNeXt 原论文；后续索引同步待第十份正文全章复读后统一检查。
- `status`：仅为在制检查点，不作为第二册第十份正文或全书精修完成记录。

### 2026-10-07：启动与规范确认

确认本轮依据是当前 `WRITING_SPEC.md`。其中对常态化精修的要求不止格式清理，还包括正确性、深度、广度、可读性、可验证性和配套同步；每份正文须按实际声明范围进行全文复读，代码需检查语法、正常路径和相关边界，公式需核对定义域、渲染和数值例子，链接应检查目标及引用范围。执行记录写入本文件，研究事实继续按需写入 `research/`。

#### 第一册第一份正文：`book-01-core-30/chapters/01-基础与语言模型.md`

- `scope`：修订前全文 2,378 行，第 1—5 讲及综合实验；本轮从头复读，旧轮次记录不计入覆盖。修订后文件 2,477 行。
- `changes`：区分 next-token 训练信号与能力涌现的证据边界；澄清预训练、SFT/instruction tuning、RLHF/DPO 与运行时 RAG/工具调用的关系；修正梯度下降演示中 loss 与参数状态错配；补足 label shift、数值例子、代码契约及非法/非有限输入检查；将 54 个数学围栏改为 GitHub 兼容的 `math` 围栏。配套 `PAPERS.md` 增加 RAG 原始论文入口。
- `evidence`：正文引用 GPT-3、Adam、AdamW、Chinchilla、InstructGPT、DPO、RAG、Toolformer、DeepSeek-R1 原文及 PyTorch Autograd 文档；Hugging Face 文档站本次返回 503，因此改用固定 Transformers 仓库 revision 的官方文档源。证据仅支持文中限定范围，不外推为所有模型或生产系统的通用结论。
- `code_status`：10 个 Python 块全部通过语法编译和运行，包括 6 个标准库、3 个 NumPy 与 1 个 PyTorch 块；使用 Python 3.13.13、NumPy 2.4.6、PyTorch 2.12.0+cpu。原有边界检查通过。未声称 CUDA 或生产环境验证。
- `formula_status`：54 个公式围栏统一为 GitHub 兼容的 `math` 围栏；公式及代码围栏成对，花括号检查通过，并核对主要数值例子。未使用外部 Markdown 渲染器。
- `link_status`：11 个正文外部来源链接本轮均返回 HTTP 200；Hugging Face 文档站此前不可达，已改用固定 revision 的 Transformers 仓库文档，该链接返回 200。
- `sync_status`：新增 `PAPERS.md` 的 RAG 原始论文入口；简介、目录及其他配套索引暂未发现需要因本轮修改而变更的条目。
- `status`：完成；结果为 `source-backed`、`verified` 和 `local-demo`。验证限于已记录的本地 Python/NumPy/PyTorch CPU 环境。

#### 第一册第二份正文：`book-01-core-30/chapters/02-transformer核心.md`

- `scope`：修订前全文 4,541 行，第 6—15 讲；本轮从首行复读到末尾，当前文件 4,769 行。
- `changes`：澄清 BPE、WordPiece、SentencePiece 的边界和特殊 token/角色名的差异；修正 attention 点积、排列等变性及计算/存储复杂度表述；统一自定义 attention mask 与 PyTorch `MultiheadAttention` 的布尔语义；补入 attention 手算、MHA 参数量、MHA/GQA/MQA KV Cache 估算、LayerNorm/RMSNorm 和 RoPE 数值例子；区分 Pre-Norm 位置与 RMSNorm 算子类型；限定 RoPE 外推结论并更正相对位移符号；补充 SwiGLU、ALiBi 机制入口；为代码补充运行契约、shape 与非法输入处理。100 个公式围栏统一为 `math`。
- `evidence`：原始 Transformer、BERT、BPE、SentencePiece、RoFormer、ALiBi、RMSNorm、Pre-LN/Post-LN、FlashAttention、GQA、Lost in the Middle 论文及 PyTorch `MultiheadAttention` 官方文档。12 个外部链接本轮均返回 HTTP 200。WordPiece 的精确评分资料页面本轮不可达，因此正文不指定统一评分公式，仅说明实现间的构造与编码规则不同。详细证据范围见 `research/transformer-core-review-2026-10-07.md`。
- `code_status`：22 个 Python 块全部通过语法编译；21 个独立运行，训练 loop 按正文标注与 MiniGPT 定义在同一进程运行。Python 3.13.13、NumPy 2.4.6、PyTorch 2.12.0+cpu；15 个代表性非法输入边界检查通过。未测试 CUDA、生产 serving 或分布式训练。
- `formula_status`：100 个 `math` 围栏成对，花括号检查通过；检查 `\operatorname`、`\text{..._}` 和尖括号下标均无命中。手算 attention、KV Cache、归一化、RoPE 结果与代码输出一致。
- `link_status`：12 个外部链接均返回 HTTP 200；第二册和百科两条相对链接目标存在；同文第 14 讲锚点已检查。
- `sync_status`：`PAPERS.md` 增补本章基础机制论文入口；新增 `research/transformer-core-review-2026-10-07.md` 记录来源与证据边界。百科和第二册已有更深的 SwiGLU、Norm、RoPE、attention 专题，本章提供概念入口和链接，不复制整段专题。
- `status`：完成；结果为 `source-backed`、`verified` 和 `local-demo`。没有将论文实验或 CPU toy 结果外推为目标模型或生产系统结论。

#### 第一册第三份正文：`book-01-core-30/chapters/03-训练对齐推理.md`

- `scope`：修订前全文 5,997 行，第 16—30 讲；本轮从首行复读至末尾，修订后 6,040 行。目录中的第 16—30 讲范围与文件导航已核对。
- `changes`：修正预训练 packing 的窗口重叠、尾块处理和 EOS 不自动隔离跨文档 attention 的边界；checkpoint 选择改为预设门槛加分领域指标；修正 AdamW 状态估算的 GiB 单位、fp32 master weights 与 all-gather 峰值假设；补足 assistant-only SFT 的 next-token shift、无有效目标处理及长回答在 token 平均中的权重边界；限定 DPO 流程复杂度结论；修正 KV Cache、attention matrix 的单位和多头数值；澄清 FlashAttention 的精确数学目标与浮点舍入；补全量化裁剪公式、手算例和 QAT/INT8/INT4 的适用边界；修复综合评估表，限定 CoT、扰动测试、test-time compute、verifier 和安全评估结论；为采样 demo 补齐依赖、运行契约、数值输出及温度/top-k/top-p 边界检查。
- `evidence`：核对 GPT-3、Chinchilla、ZeRO、Megatron-LM、InstructGPT、FLAN、LoRA、QLoRA、PPO、DPO、摘要反馈、FlashAttention、PagedAttention、GPTQ、AWQ、TruthfulQA、HELM 和 Snell 等论文；使用 PyTorch CrossEntropyLoss/Distributed/FSDP、Transformers chat template/generation 固定 revision、TRL DPO Trainer 固定 revision、vLLM 设计文档及 OWASP LLM01:2025。章节中 19 个原有外部来源初查返回 HTTP 200；Transformers 与 TRL 两个 Hugging Face 动态文档页经代理返回 503，正文改用固定 Git revision 来源。完整支持范围和来源限制见 `research/training-alignment-inference-review-2026-10-07.md`；`PAPERS.md` 已有 Snell 论文入口。
- `code_status`：最终版本 13 个 Python 块逐块独立运行全部通过；Python 3.13.13、NumPy 2.4.6、PyTorch 2.12.0+cpu。采样代码额外通过 38 项边界断言；此前对训练 FLOPs、显存、SFT mask、KV Cache、量化和评估示例记录的 20 项边界检查仍适用于未改动的测试路径。未声称 CUDA、多卡训练、目标模型或生产 serving 验证。
- `formula_status`：37 个 math 围栏、其余代码/文本围栏均成对；检查公式花括号、行内美元符号和禁用写法无命中。3 个 Markdown 表格列数一致；packing、SFT shift、KV Cache、attention 矩阵、量化误差等数值结果已按定义或本地代码核对。未用外部 Markdown 渲染器。
- `link_status`：正文 29 处引用对应 25 个唯一外部 URL；19 个原有链接初查为 HTTP 200，两个临时不可达的 Hugging Face 动态文档链接已改为固定 revision。章节无相对 Markdown 链接；目录中的章节文件和讲次锚点已核对。动态 PyTorch stable 和 vLLM latest 文档需在依赖升级时复核。
- `sync_status`：新增 `research/training-alignment-inference-review-2026-10-07.md`，保存来源、证据等级、访问边界、本地验证与未验证项；`PAPERS.md` 已含 Snell 条目，无需重复添加；简介、目录及其他索引无需因本轮内容修订另行改写。
- `open_risks`：未在 GPU 上测试 FlashAttention/量化 kernel、数值误差、显存峰值或吞吐；未运行真实模型预训练、SFT、DPO/PPO、安全评测、多卡训练或真实请求 serving。代码与数值示例仅为机制演示和容量估算。
- `status`：完成；结果为 `source-backed`、`verified` 和 `local-demo`，范围仅限本记录所列来源与本地 CPU 环境。

#### 第一册第四份正文：`book-01-core-30/chapters/04-多模态基础.md`

- `scope`：修订前 1,076 行，第 31—36 讲；本轮从首行复读至末尾，当前 1,161 行。
- `changes`：统一图像 patch/tubelet 的补齐假设与估算代码；加入 2026 原生多模态公开实例，并把 GPT-4o/Sora 限定为历史证据案例；补足 CLIP 手算、prompt ensemble 归一化和 SigLIP 的 `±1` 标签、缩放分数与 bias 定义；为 VLM、扩散、语音和视频的关键结论补就近来源；增强 patch 数、音频 hop、CER、扩散标量和视频 token demo 的运行契约及边界处理。
- `evidence`：核对论文、生成指标、OWASP 与 C2PA 来源；完整来源、适用范围和访问状态见 `research/multimodal-foundations-review-2026-10-07.md`。25 个唯一外部 URL 中 23 个返回 HTTP 200；两条 OpenAI 历史页面经代理返回 Cloudflare 403，因此正文仅保留其历史公开声明，不推断当前产品状态或未披露架构。书系内 Qwen3.5 多模态与语音专题目标均存在。
- `code_status`：7 个 Python 块逐块独立运行通过；Python 3.13.13、NumPy 2.4.6、PyTorch 2.12.0+cpu。patch 数、CLIP loss、VLM shape、扩散标量、音频 hop、CER 和视频 token 输出与正文一致；前次记录的 36 项边界/反事实检查仍适用于本轮未改动的代码路径。示例不构成真实多模态模型、CUDA 或生产服务验证。
- `formula_status`：39 个 `math` 围栏成对且花括号平衡；再次核对 patch/tubelet 补齐约定、CLIP 数值、prompt ensemble 二次归一化、SigLIP bias、CFG 两种等价写法和示例参数、扩散定义域、音频帧边界及 CER 分母。表格列数一致；未使用外部 Markdown 数学渲染器。
- `link_status`：章节内相对链接目标存在；25 个外部 URL 的状态和证据用途按研究记录留档。OpenAI 两页当前 403 的访问限制已在正文与研究记录注明，引用仅支持历史公开材料。
- `sync_status`：`PAPERS.md` 补充多模态论文与评估入口；新增 `research/multimodal-foundations-review-2026-10-07.md`。简介、目录中的第 31—36 讲文件和次序与正文一致，无需额外改写。
- `open_risks`：未运行真实视觉/音频 encoder、VLM、扩散生成器、ASR 或视频生成模型；未验证 CUDA kernel、显存、实时延迟和生产吞吐；未复现 CLIP/SigLIP、FID/FVD、Whisper、codec 或扩散论文实验。OpenAI 页面本轮不可直接访问，若后续要描述其当前产品状态须重新核验。
- `status`：完成；结论限于本记录中的公开来源、本地 toy 示例和 CPU 环境。


#### 第一册第五份正文：`book-01-core-30/chapters/05-综合复习与技术表达.md`

- `scope`：修订前 540 行，第 37—38 讲与学习迁移计划；本轮从首行复读至末尾，当前 585 行。
- `changes`：将非标准的 VOI 公式改为明确标注的教学优先级评分；把数学诊断与 SFT 示例表标为构造数据，并注明准确率、百分点、请求/claim 分母和 p95 毫秒单位；补全标准误、加权支持率、Pareto 支配、复习衰减、时间预算、面试评分和迁移差距中的变量范围与解释；为优先级排序和复习调度 demo 增加输入检查及预期输出。
- `evidence`：复核 Roediger & Karpicke (2006) 与 Cepeda et al. (2006) 的论文元数据和可访问摘要；正文只保留其 prose-passage / verbal-recall 实验范围内的发现，不推出所有技术学习者的统一复习间隔。Crossref 与 OpenAlex 的访问和摘要边界见 `research/integrative-learning-transfer-review-2026-10-07.md`；`PAPERS.md` 新增学习科学入口。
- `code_status`：2 个 Python 块逐块运行通过，使用系统 Python 3.12.9。预期实验排序和调度状态输出匹配；另有 12 项边界检查覆盖非法/非有限收益评分、零成本、Topic 初值、负数/NaN/布尔 day 和非布尔结果。仅验证教学规则，不代表真实学习效果。
- `formula_status`：11 个 `math` 围栏成对、花括号平衡；复核信息增益/启发式评分、二项标准误、claim 支持率、Pareto 方向、回忆衰减、时间预算、加权学习与面试评分、迁移差距的变量和定义域。Markdown 表格列数一致；未使用外部数学渲染器。
- `link_status`：章节相对研究记录链接和书内目录目标存在。Roediger DOI 路径返回 HTTP 200 但只呈现人机验证页；Cepeda DOI 页面返回 HTTP 403，Crossref API 可访问且 OpenAlex 索引摘要可访问。原始出版商 DOI 保留为读者入口，访问限制已说明。
- `sync_status`：`PAPERS.md` 增加两篇学习科学论文入口；新增 `research/integrative-learning-transfer-review-2026-10-07.md`；第一册 `目录.md` 的第 37—38 讲与本文件一致，无需修改目录或简介。
- `open_risks`：学习评分权重、指数衰减模型和调度器均为教学启发式，未经技术课程或目标岗位用户实验校准；Cepeda 论文全文本轮不可直接访问，相关表述限定在索引摘要支持的 verbal-recall 结论。
- `status`：完成；来源范围受限、代码本地验证通过，未将教学分数解释为真实产品或心理测量结论。

#### 第二册第一份正文：`book-02-advanced-100/chapters/01-深度学习与概率基础进阶.md`

- `scope`：修订前全文 1,905 行，覆盖第 1—10 讲；本轮从头复读到末尾，修订后 2,026 行。第二册目录中的文件链接和讲次范围已核对。
- `changes`：把 Scheduled Sampling 的提出与目标不一致批评并列说明；限定 label smoothing、bias–variance、double descent、scaling law、Bayes/MAP、不确定性、校准和 FFN key–value memory 的假设及证据范围；补充符号定义和边界；修复 KL 极小概率的比值溢出风险、梯度裁剪符号文本中的控制字符、二维优化的非有限更新处理，以及大范数裁剪和 sigmoid 输入边界。
- `evidence`：核对 24 个 arXiv 元数据条目及额外 Huszár、Dinh、Lewis 来源，并检查 Glorot/Bengio 的 PMLR 页面。证据范围、链接访问边界和各论文可支持内容见 `research/deep-learning-probability-review-2026-10-07.md`。论文实验均按原任务范围引用；代理对部分 arXiv HEAD 请求间歇超时，不作为链接失效证据。
- `code_status`：10 个纯标准库 Python 块在系统 Python 3.12.9 下逐块编译、独立运行通过，正常输出与正文一致；另通过 24 项输入、数值稳定性和非法边界断言。没有验证真实模型训练、GPU、论文复现或生产性能。
- `formula_status`：87 个 `math` 围栏及代码围栏成对，公式花括号和 `begin/end` 环境平衡；复核 temperature、label smoothing、偏差—方差分解、Scaling Law、MAP 先验和 loss、ECE、梯度裁剪定义域及主要数值输出。未使用外部 Markdown 数学渲染器。
- `link_status`：章节 25 个行内引用均在参考资料列表中，参考区共 30 个唯一外部链接；24 个 arXiv ID 命中元数据 feed，3 个额外 arXiv 来源单独核对，PMLR、Shannon 和教材入口可访问。arXiv HEAD 受代理间歇超时影响，目标存在性以 API 元数据/页面内容核验。
- `sync_status`：`PAPERS.md` 增补本章研究入口；同步百科的 Scaling Laws 教学公式和计算预算边界，并校正其偏差—方差假设及 label smoothing 的泛化表述。术语表已有相关术语，`EXERCISES.md` 已覆盖偏差—方差、MAP、Beta 更新、熵、ECE、不确定性和 Scaling Law，本轮未重复添加题目。研究记录新增 `research/deep-learning-probability-review-2026-10-07.md`。
- `open_risks`：示例只验证 Python 3.12.9 本地 toy 行为，不代表目标模型、GPU kernel、实际训练稳定性或真实校准效果；经验缩放关系不用于预测未验证任务能力；未使用外部公式渲染器。`python3 scripts/build_pdfs.py --dry-run` 因环境缺少 pandoc 未能运行，本轮未构建 PDF。
- `status`：完成；结论限于所列来源、本地代码与同步条目，结果为 `source-backed`、`verified` 和 `local-demo`。全书精修目标仍在进行中。

#### 第二册第二份正文：`book-02-advanced-100/chapters/02-transformer架构进阶.md`

- `scope`：修订前 1,610 行，覆盖第 11—25 讲；本轮从头复读至末尾，修订后 1,766 行。文末架构统一评估改为章节总结，不占第 26 讲；第二册目录和下一章从第 26 讲开始的编号已核对。
- `changes`：修正 32K→1M attention 交互倍率和十进制/二进制单位说明；补全 Encoder–Decoder aligned 公式换行；补充 RoPE 常见频率公式并区分频率基数；限定 FlashAttention、Pre-LN、ScaleNorm、MoE z-loss/capacity、PagedAttention、长上下文扩展和 SSM 结论的支持范围；修正 SwiGLU 参数量 demo 对三矩阵 MLP 的漏算；为 15 段代码补足运行契约、形状与非法/非有限输入处理及数值输出。
- `evidence`：原有 25 个 arXiv 来源及新增 8 个来源的题名元数据由 arXiv Atom API 核验；新增 LayerNorm、ScaleNorm、GeLU、FFN key-value 分析、ST-MoE、PagedAttention、Position Interpolation 和 YaRN。重点摘要覆盖 FlashAttention、Performer、Pre-LN/DeepNet、稀疏注意力、S4/Mamba/SSD 及 PI/YaRN；一批 arXiv 摘要请求返回 503，来源元数据仍已在其他 feed 查询中命中。GPT 论文 PDF 返回 HTTP 200。详细支持范围见 `research/transformer-architecture-advanced-review-2026-10-07.md`。
- `code_status`：15 个标准库 Python 块在系统 Python 3.12.9 下逐块独立运行通过，正常输出与正文一致；另通过 18 项非法输入/溢出检查和 2 项数值不变量检查，覆盖 NaN/Inf、形状、bool-as-int、head 整除、Norm 溢出、capacity、mask、状态递推和配置维度。未验证 GPU kernel、真实模型、论文复现或生产 serving。
- `formula_status`：53 个 `math` 围栏，代码/文本围栏配对；花括号、环境、禁用宏和尖括号下标检查通过。重新核对 attention 复杂度、显存、SwiGLU 参数量、RoPE 旋转恒等式、MoE 均衡及 z-loss、capacity 和 SSM 状态公式。主要数值结果与代码一致；未使用外部 Markdown 数学渲染器。
- `link_status`：33 个唯一 arXiv ID 均由元数据 feed 命中；GPT PDF 直链返回 HTTP 200。章节引用补到支持的主张附近；目录所指章节文件及 11—25/26—40 讲次连续且不重复。
- `sync_status`：`PAPERS.md` 增加 LayerNorm、ScaleNorm、GeLU、长上下文位置扩展、PagedAttention 和 MoE 稳定性论文入口。第二册目录无需改写；`GLOSSARY_EN_ZH.md`、`EXERCISES.md`、`KNOWLEDGE_GRAPH.md` 和百科已有对应主题，不重复复制内容。研究笔记新增 `research/transformer-architecture-advanced-review-2026-10-07.md`。
- `open_risks`：toy 代码和显存账本不代表 CUDA、目标 checkpoint、真实长上下文质量、MoE 通信尾延迟或线上吞吐；未构建 PDF 或使用外部公式渲染器。arXiv 一批摘要查询暂时 503，但来源 ID/题名已核验，正文结论按论文任务限定。
- `status`：完成；结果为 `source-backed`、`verified` 和 `local-demo`。全书精修目标仍在进行中。

#### 第二册第三份正文：`book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md`

- `scope`：修订前 1,632 行，覆盖第 26—40 讲；当前 1,936 行。第二册目录中的讲次范围仍为第 26—40 讲，下一章从第 41 讲开始。
- `changes`：把数据质量分数改为仅对通过准入的样本排序，将许可、来源追溯、隐私/安全和评估污染设为独立门槛；修复“许可未知仍 `keep=True`”；补齐 BPE 频数、tokenizer 样本量、shingle 空集合、温度采样、合成数据、Scaling demo、loss dtype/device/ignore index、梯度裁剪、checkpoint 往返、多语言采样和 FIM 区间边界；将 BF16 参考修正为 Kalamkar 等人的真实 arXiv 论文；为核心机制补就近来源。
- `evidence`：arXiv Atom API 核实 `1905.12322` 的真实题名为 *A Study of BFLOAT16 for Deep Learning Training*，作者首位为 Dhiraj Kalamkar；原有 Wang and Kanwar 题名与该 ID 不匹配，已更正。31 个唯一外部 URL 均有成功状态检查（AMP 旧 URL 曾超时，替换为 PyTorch 官方 `docs.pytorch.org` 页面后 HTTP 200）；31 个中其余链接均 HTTP 200。arXiv 元数据批量查询本轮间歇连接失败，链接存在性不等同于对所有摘要和实验全文的完整复核；研究记录见 `research/tokenization-data-pretraining-review-2026-10-07.md`。
- `code_status`：14 个 Python 块均通过语法编译；10 个纯标准库块在 Python 3.12.9 下独立运行，43 项边界断言通过。其余 4 个 PyTorch/NumPy 块在 Python 3.12.9、PyTorch 2.12.0+cpu、NumPy 2.4.6 下逐块运行通过；输出包括 `loss=2.2648, ppl=9.6288`、`loss=0.0374, preclip_grad_norm=0.2892`、FP16 对 70,000/1,000,000 溢出而 BF16/FP32 仍有限，以及 checkpoint 参数与游标恢复成功。CPU-only 环境位于仓库 `.tmp/llm-book-cpu-check/`，`.tmp/` 已加入 `.gitignore`；未安装 CUDA 库。
- `formula_status`：当前 29 个 `math` 围栏；公式花括号、环境与围栏配对通过，5 个 Markdown 表格列数一致，未发现禁用宏、下划线 `\text{}` 或尖括号下标写法。复核 fertility、BPE 计数、词表成本、Jaccard 空集合、温度权重、exposure、Scaling 曲线、token loss/PPL、梯度裁剪、浮点缩放、checkpoint 轨迹和 pass@k 的定义域与示例。
- `link_status`：正文当前 31 个唯一外链均返回 HTTP 200；ACL anthology、Nature、OpenAI PDF、PyTorch 文档与 arXiv 入口均有访问检查。无相对 Markdown 链接；第 26—40 讲文件和顺序与目录一致。
- `sync_status`：`PAPERS.md` 新增 tokenizer、数据集、去重、Scaling、BF16 来源入口；`GLOSSARY_EN_ZH.md` 补充 Unigram tokenizer 和 token fertility；`EXERCISES.md` 的数据源估值题增加许可/隐私/污染硬门槛要求。第二册目录和简介无需修改。
- `open_risks`：未使用 CUDA、多卡训练、真实模型/数据或生产服务；未运行外部 Markdown/MathJax 渲染器及 PDF 构建。CPU 代码输出仅验证这些局部教学示例，不代表目标硬件性能或生产训练。
- `status`：本章编辑、来源核验、公式检查和全部教学代码核验完成；结论为 `source-backed`、`verified` 和 `local-demo`，适用范围限于所列来源与本地 CPU 环境。全书精修继续进入第二册第四份正文。

#### 第二册第四份正文：`book-02-advanced-100/chapters/04-大规模训练系统进阶.md`

- `scope`：修订前 1,184 行，覆盖第 41—50 讲；本轮从头复读到末尾，当前 1,280 行。第二册目录链接和第 41—50 讲范围核对一致。
- `changes`：补足 DDP 在变长 token 下的全局 loss/gradient 缩放推导和数值例子；把 GPU/NCCL 专属的 DDP 骨架改成双进程 CPU/Gloo 可运行 demo；修复 TP 公式；定义 blockwise softmax 统计量；让 activation checkpoint 示例比较输出及梯度；完善显存、MFU、吞吐和事故分类 demo 的契约/边界；限定 checkpoint 原子提交在文件系统与对象存储中的不同语义。
- `evidence`：来源包含 PyTorch DDP、FSDP、checkpoint、Profiler、Distributed Checkpoint 和 NCCL 官方文档，以及 ZeRO、Megatron-LM、GPipe、Sequence Parallel、FlashAttention、Ring Attention 等论文入口。21 个唯一外链本轮均返回 HTTP 200；状态码确认不等于逐篇论文全文复核。详细范围见 `research/distributed-training-systems-review-2026-10-07.md`。
- `code_status`：10 个 Python 块逐块通过语法和正常路径执行，Python 3.12.9、PyTorch 2.12.0+cpu；环境保存在 Git 忽略的 `.tmp/llm-book-cpu-check/`，`torch.cuda.is_available()` 为 False，未安装 CUDA 库。DDP 双进程 Gloo 输出 `world_size=2 local_batch=2 global_batch=4 weight=1.5000`；其余示例包括 7B/8 ranks 状态估算、PP 利用率 0.727、online attention full/online 相等、checkpoint 输出及输入/参数梯度均相等、重叠模拟 205ms→135ms、MFU 估算 0.2206、事故分类五个信号。另通过 7 项输入/边界断言。
- `formula_status`：43 个 `math` 围栏；公式花括号、`begin/end` 环境与总围栏配对通过。重点核对 DDP rank/token 平均、ZeRO 状态分片、TP 矩阵维度、PP bubble、online softmax 合并、token throughput、MFU 和相对监控量。两张 Markdown 表格列数一致；未使用外部公式渲染器。
- `link_status`：21 个唯一外链本轮均返回 HTTP 200；PyTorch 文档统一到当前 `docs.pytorch.org` 入口。章节没有相对 Markdown 链接，目录讲次映射一致。
- `sync_status`：`PAPERS.md` 增加 ZeRO、Megatron-LM、GPipe、Sequence Parallel、Ring Attention 等直接论文入口；`EXERCISES.md` 增加变长 token 下 DDP 梯度缩放推导题；术语表已有本章主要并行及 MFU 术语，无需重复。
- `open_risks`：未测试 CUDA/NCCL、GPU throughput/MFU、真实 FSDP/ZeRO 分片、TP/PP/CP 集群、多节点、真实模型 checkpoint 或生产故障恢复；CPU 输出只支持局部教学行为。`python3 scripts/build_pdfs.py --dry-run` 因环境缺少 Pandoc 未能执行，本章未构建 PDF。
- `status`：完成；结论限于所列公开来源、数学推导与本地 CPU 教学示例，结果为 `source-backed`、`verified` 和 `local-demo`。全书精修继续进入第二册第五份正文。

#### 第二册第五份正文：`book-02-advanced-100/chapters/05-后训练与对齐进阶.md`

- `scope`：修订前 2,564 行，覆盖第 51—65 讲；本轮从头复读到末尾，当前 2,566 行。第二册目录仍将本章映射到第 51—65 讲，下一章从第 66 讲开始。
- `changes`：PPO 教学环节改为用 rollout 时冻结的 `old_logp` 计算 sampled-KL reward，并明确 `new_logp` 只参与 PPO ratio；Truthfulness 评估分别定义事实正确 `z_i`、幻觉 `h_i` 与 unsupported claim `u_i`，不再把所有事实错误直接计为幻觉；TRL SFT/DPO 来源固定到 commit `956d742c1b8df9491c8382614e66415fb15bff9c`，删去把 PPO Trainer 描述成该版本当前接口的错误引用；为 MMLU/MMLU-Pro、GSM8K/MATH、HumanEval/MBPP、IFEval 和 LongBench 增加原始论文入口及任务维度说明；OWASP 引用更新为最终项目页。
- `evidence`：arXiv Atom API 返回并核对了 8 个 benchmark ID 的题名和版本，均与正文用途对应；28 个唯一外部链接本轮均返回 HTTP 200。固定 TRL revision 的完整 Git tree（`truncated=false`）包含 `sft_trainer.md`、`dpo_trainer.md`，没有任何 PPO 路径；两个 raw 文档和固定 revision 文档树均返回 200，原动态文档代理返回 503，预期 PPO 文档路径返回 404。证据只支持固定快照的公开文档/源码树，不推断其他 TRL 版本行为。详细快照见 `research/post-training-alignment-review-2026-10-07.md`。
- `code_status`：17 个 Python 块在 Python 3.12.9 下逐块独立运行通过；环境为 PyTorch 2.12.0+cpu、NumPy 2.4.6，`torch.cuda.is_available()` 为 False，未安装 CUDA 库。另在 5 个示例上通过 8 项无效输入检查，覆盖 softmax 空/非有限 logits、assistant mask 空/错位/越界、PPO 空 rollout、calibration 越界置信度和 checkpoint 零基线长度。输出与正文中的 mask、SFT 选择、偏好偏差、PPO clipping、校准、风险指标和 checkpoint 选择结论一致；只验证局部教学行为。
- `formula_status`：76 个 `math` 围栏；Markdown 数学、代码与文本围栏配对，公式花括号及 `begin/end` 环境检查通过。重新核对 PPO rollout/reference KL 与 ratio 的变量职责、Truthfulness 三种标签定义和第 65 节 checkpoint 硬门槛/软目标关系；未使用外部公式渲染器。
- `link_status`：28 个唯一正文外链均返回 HTTP 200；8 个 benchmark arXiv 元数据项已逐一核对题名和版本。章节无相对 Markdown 链接，OWASP 最终目标、TRL 固定快照和第二册目录映射已检查。
- `sync_status`：`PAPERS.md` 增加“后训练评估与公开基准”入口；`EXERCISES.md` 已包含模型发布 benchmark 审计表和评估门禁题，`INTERVIEW_BANK.md` 已按能力切片解释主流 benchmark，术语表已有 MMLU-Pro 与 Model Release Benchmark Catalog，因此不重复添加。第二册目录无需修改；新增 `research/post-training-alignment-review-2026-10-07.md`。
- `open_risks`：代码只演示局部机制，不代表实际模型训练、目标 checkpoint、TRL 训练器运行或生产验收；论文链接和元数据核验不等于逐篇全文复核或 benchmark 复现。未使用外部 Markdown/MathJax 渲染器，也未构建 PDF；Pandoc 缺失导致此前全书 PDF dry-run 无法执行。
- `status`：本章精修、来源/链接检查、公式结构检查及声明范围内的代码验证完成；结论为 `source-backed`、`verified` 和 `local-demo`。全书目标仍在进行中，下一章为第二册第七份正文。

#### 第二册第六份正文：`book-02-advanced-100/chapters/06-推理部署与服务进阶.md`

- `scope`：覆盖第 66—75 节；修订后 1,754 行。第二册目录映射第 66—75 讲正确，第七份正文从第 76 讲开始。
- `changes`：定义 TTFT、TPOT、端到端时延及不同吞吐口径；零输出请求的 TTFT/TPOT 记为 N/A，单请求端到端输出速率命名为 `request_output_rate_tps`，避免与并发服务吞吐混淆。KV block allocator 在修改引用计数前先校验整组输入；调度例加入动态到达；量化零值组的 `scale=1.0` 哨兵行为有说明；回归审计器拒绝空样本、非布尔安全字段及非有限 P95；端云路由落实 `local_only` 硬约束，超出本地上下文时拒绝转发。Prefix Cache 身份边界限定为影响 K/V 计算的输入和配置；补充 Python 运行版本及服务协议边界。
- `evidence`：核对 13 篇论文的 arXiv ID/题名，包括 FlashAttention/2、GQA、KIVI、PagedAttention、SGLang、Sarathi-Serve、Speculative Decoding、Medusa、EAGLE、GPTQ、AWQ 和 SmoothQuant；链接状态检查记录为 31 个唯一外链中 30 个 HTTP 200，OpenAI latency guide 受代理限制返回 403。TGI 固定文档快照 `b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed` 可访问，仓库已归档，因此正文仅作历史 API 参考；bitsandbytes 使用官方仓库入口。证据范围和边界见 `research/inference-deployment-service-review-2026-10-07.md`。
- `code_status`：10 个标准库 Python 块在 Python 3.12.9 下逐块执行通过；`.tmp/verify_chapter6.py` 另做 10 组边界测试，全部通过。验证环境在 Git 忽略的 `.tmp/llm-book-cpu-check/`；本章示例本身只依赖 Python 标准库，不安装 CUDA。输出核对了 TTFT/TPOT 账本、KV cache 估算、block 引用计数、动态调度、推测接受率、量化误差、服务状态审计、容量成本、回归诊断和端云路由。
- `formula_status`：40 个 `math` 围栏；Markdown 围栏配对，数学块花括号平衡。重新核对 Prefill/Decode 复杂度、时延/吞吐分母、KV Cache 字节、推测解码分布校正、量化误差、容量/成本和端侧资源账本。未运行外部数学渲染器。
- `link_status`：31 个唯一外链中 30 个 HTTP 200；OpenAI latency guide 返回代理 403 并保留为工程入口，访问限制已记录。无相对 Markdown 链接；目录范围正确。
- `sync_status`：`PAPERS.md` 增补 FlashAttention、Speculative Decoding、Medusa/EAGLE、GPTQ/AWQ/SmoothQuant/KIVI、Orca/Sarathi-Serve/SGLang 等直接论文入口；题库、面试库和术语表已有相关 KV Cache、调度、时延、推测解码与量化主题，不重复复制章节。
- `open_risks`：未在 CUDA/GPU、目标 runtime 或生产服务验证性能、量化质量、峰值内存、并发 goodput 或端侧能耗；论文元数据和链接状态不等于逐篇论文全文复核；未运行外部 Markdown/MathJax 渲染器或构建 PDF。
- `status`：本章从头复读、修订、链接/公式结构检查和声明范围内的代码验证完成；结论限于所列来源与本地教学代码。全书精修继续进入第二册第七份正文。

#### 第二册第七份正文：`book-02-advanced-100/chapters/07-长上下文rag与agent.md`

- `scope`：覆盖第 76—86 节；修订后 2,596 行。目录映射正确，下一份正文从第 87 讲开始。
- `changes`：修正多 gold 检索场景下 Recall@k 定义，并与 Hit@k 区分；补全长上下文混合损失权重、质量评分和检索/精排指标的定义域；限定 PQ payload 和因果 attention 的复杂度估算；修正期望故障损失中的条件恢复成本；Function Calling 示例增加参数类型检查并把策略 `allow` 与真实执行区分；Memory 示例明确工程分类、敏感度边界和 tombstone 输出；退款贯通案例补上地区、退款余额及政策例外的证据依赖。
- `evidence`：40 个唯一来源链接在第一次检查均返回 HTTP 200；添加 Stanford IR book 指标定义链接后总计 41 个，第二轮检查 40 个 HTTP 200，HNSW arXiv 页 `1603.09320` 瞬时超时，但此前检查为 200 且单独摘要页抓取成功。34 个 arXiv 页面题名元数据与引用方法/论文相符；Atom 批量 API 返回 503，改为逐页核对。RAGAS、Stanford IR book、OWASP、NIST、JSON Schema 链接返回 200。摘要/标题核对不等于全文和实验复核。研究详情见 `research/long-context-rag-agent-review-2026-10-08.md`。
- `code_status`：10 个标准库 Python 块在 Python 3.12.9 下逐块执行成功；`.tmp/verify_chapter7.py` 的 10 组边界检查通过，覆盖长度/标签、ACL、零向量、预算与多样性、引用缺失、工具 schema、依赖环、策略边界和 memory 租户/敏感度/删除。环境为 Python 3.12.9、PyTorch 2.12.0+cpu、CUDA unavailable；本章代码不依赖 PyTorch，也未安装 CUDA 库。
- `formula_status`：62 个 `math` 围栏、136 组成对 Markdown 围栏；花括号和 `begin/end` 环境平衡，表格列数一致。复核 RoPE/长序列成本、长上下文评估、Recall/Hit、Precision、rerank、claim/citation 支持、动作授权、故障损失和 Memory 评分等定义。未用外部公式渲染器。
- `link_status`：41 个唯一外链；40 个返回 HTTP 200，1 个 HNSW 页面复查超时，但已记录先前 200 和本次摘要页成功抓取。无相对 Markdown 链接，第二册目录链接与第 76—86 节范围一致。
- `sync_status`：`PAPERS.md` 增补长上下文、检索/RAG 评估、工具/API 基准、规划、安全与 Memory 研究入口；`GLOSSARY_EN_ZH.md` 统一 Recall@k/Hit@k 定义；`EXERCISES.md` 的向量检索评估题补充多 gold 与空 gold 样本；`INTERVIEW_BANK.md` 的检索评估回答补充分母与 Hit 区别。简介和目录无需改写。
- `open_risks`：未运行真实长上下文模型、企业文档解析、embedding、ANN、reranker、RAG/Agent benchmark、真实工具或持久化 Memory；不证明生产系统事实性、权限隔离、删除传播、安全或硬件成本。未运行外部 MathJax/PDF 构建；HNSW 链接状态存在瞬时超时。
- `status`：本章全文复读、修订、来源/链接检查、公式结构检查、索引同步和声明范围内的代码验证完成；结论仅限所列来源和本地教学代码。全书目标继续进行，下一章为第二册第八份正文。

#### 第二册第八份正文：`book-02-advanced-100/chapters/08-reasoning与test-time-compute.md`

- `scope`：覆盖第 87—96 讲；从首行到末行完成全文复读。第二册目录仍映射本文件到第 87—96 讲，下一章为第 97—104 讲。
- `changes`：修正 CoT 联合生成分解和干预敏感度分母；为 PRM 首错阈值和 pass@k 补定义域；Self-Consistency 加权代码与指数温度权重公式对齐；CoT 过程审计器只接受整数加/减/乘，将解析不到的等式标为不可检查，并要求显式 final 字段；答案归一化使用完整字符串和精确分数，避免把“not 4”误判为 4；限定 `A_final ≤ P_hit` 的候选池前提；test-time compute 路由实际使用 `verifiable`，对高风险且不可验证请求转人工复核；污染率注明训练/评估集合非空；修复 CodeRL 的 arXiv 错链（`2302.00003` 实为外部记忆论文，正确来源为 `2207.01780`）。
- `evidence`：23 个 arXiv 引用的 ID、题名和摘要已通过 Atom API 核对；25 个唯一外链（23 个 arXiv 页面和两个 OpenAI GitHub 仓库）经用户指定代理 `10.24.27.134:7890` 检查均 HTTP 200。元数据/摘要核对不等于全文或所有实验复核；研究边界见 `research/reasoning-test-time-compute-review-2026-10-08.md`。
- `code_status`：10 个标准库 Python 块在 `.tmp/llm-book-cpu-check` 的 Python 3.12.9 环境下逐块编译、独立运行，输出与正文逐字一致；另通过 19 项边界检查，覆盖 final 字段、未支持/混合算式、负面答案文本、分数/小数/百分数归一化、非法温度、高风险未验证路由、非文本答案和 pass@k 数值边界。环境为 PyTorch 2.12.0+cpu、NumPy 2.4.6，章节代码未导入 PyTorch、未安装 CUDA。
- `formula_status`：89 个 `math` 围栏；花括号、`begin/end` 环境、总 Markdown 围栏配对通过。核对了 CoT 概率分解、Self-Consistency 温度、PRM 阈值、候选命中上界适用条件、污染率分母和 pass@k 组合数定义域；未使用外部 LaTeX 渲染器。
- `link_status`：25 个唯一外链均 HTTP 200；相对链接目标、围栏和第二册目录讲次映射通过。
- `sync_status`：`PAPERS.md` 新增 Reasoning 与 Test-Time Compute 入口；练习、面试库和术语表已有 CoT、自一致性、PRM、搜索、pass@k 与预算路由内容，本轮无需重复添加。
- `open_risks`：只核对了元数据和摘要，未逐篇复核全文、附录、全部实验表或 scoring harness；toy verifier、路由、成本和延迟数字不代表真实模型或生产收益；未运行目标模型/benchmark，也未使用外部公式渲染器。
- `status`：本章全文复读、修订、23 个 arXiv 元数据/摘要核验、25 个外链检查、公式结构、索引同步和声明范围内的代码验证完成；结论为 `source-backed`、`verified` 和 `local-demo`，适用范围限于已读资料与本地 CPU 示例。全书精修继续进入第二册第九份正文。

#### 第二册第九份正文：`book-02-advanced-100/chapters/09-论文精读与开放研究题.md`

- `scope`：覆盖第 97—104 讲，从首行复读至末尾；第二册目录和简介的章节映射正确，下一份正文从第 105 讲开始。
- `changes`：明确 attention demo 将同一组手工向量复用于 Q/K/V；将 Scaling 示例的推理量改名为固定 dense 架构与生成长度假设下的 `relative_dense_decode_compute` 代理，并说明未包含上下文、KV Cache、批处理和硬件成本；DPO loss 改用稳定的负 log-sigmoid；LoRA demo 增加 adapter 与 base weight 的维度/rank 校验；FlashAttention 输出标为 score-cell 数量比而非总峰值显存比；开放研究评分要求先归一化异单位指标，成功成本采用统一口径；按最新写作规范统一 `\\mathrm` 与因果下标记法。
- `evidence`：arXiv Atom API 通过用户指定代理核对 11 篇论文的 ID、版本、题名和摘要主题；15 个唯一外链中 14 个最终 HTTP 200，早期 GPT 官方研究页面返回 HTTP 403。arXiv HTML 页面后续访问出现代理连接拒绝/TLS EOF；来源元数据和摘要不等于逐篇全文、实验表或 harness 复核。完整来源范围见 `research/paper-reading-open-research-review-2026-10-08.md`。
- `code_status`：8 个纯标准库 Python 块在 `.tmp/llm-book-cpu-check` 的 Python 3.12.9 环境下逐块编译、独立运行，输出与正文一致；31 项边界与不变量断言通过，覆盖空输入、非有限值、非法维度、DPO 极端 logit、attention 分块等价、MoE 路由和研究计划硬门槛。未导入 PyTorch 或 CUDA。
- `formula_status`：77 个 `math` 块、总计 104 个成对 fenced block；花括号、环境和表格列数检查通过。复核 attention mask/复杂度、RLHF sampled log-ratio、Scaling 分配推导、DPO margin、LoRA 参数量、online softmax 递推、MoE capacity/auxiliary loss 和研究评分定义域；未用外部 MathJax 渲染器。
- `link_status`：15 个唯一外链中 14 个 HTTP 200，1 个官方 GPT 页面返回 403并在正文注明。无相对 Markdown 链接；arXiv 页面元数据题名与引用主题一致，第二册导航映射正确。
- `sync_status`：`PAPERS.md` 为 LoRA、DPO 增加原论文直达链接。`EXERCISES.md`、`INTERVIEW_BANK.md` 和 `GLOSSARY_EN_ZH.md` 已有 LoRA/DPO、Scaling Laws、FlashAttention 和 MoE 的练习、问答和术语，本轮不复制既有条目；简介和目录无需修改。
- `open_risks`：没有逐篇读完 11 篇论文全文或附录；OpenAI 早期 GPT 页面受代理 403 限制；未使用外部 Markdown/MathJax 渲染器。PDF dry-run 因缺少 Pandoc 未完成，本轮未安装依赖或构建 PDF；toy 输出不代表论文复现、GPU 性能或生产结论。
- `status`：本章全文复读、修订、论文元数据/摘要与链接核验、代码输出和边界、公式结构以及论文索引同步完成；结论为 `source-backed`、`verified` 和 `local-demo`，适用范围以研究记录为准。全书精修继续进入第二册第十份正文。

#### 第一册第二份正文：Transformer 核心（2026-10-27 重构）

- scope：完整重写第 6—15 讲正文，按第一册“输入—表示—计算—学习—生成—验证”主线聚焦 decoder-only 文本模型；保留讲次编号，便于兼容题库和其他导航。此前章节审视记录见本文件前文。
- architecture_decisions：将十讲组织为四段递进：6—7 输入表示；8—10 上下文计算；11—13 因果训练与深层结构；14—15 位置机制与端到端实现。避免逐讲重复起承转合，把 attention 的直觉/公式/mask 合为连贯链，把多头和 KV 共享放在唯一主讲位置，把 Block 组合与 Norm/残差稳定性明确拆成相邻两步，并在 Q/K 定义后集中讲 RoPE。
- changes：建立全章统一的 B、T、D、H、H_kv、d_h、V shape 账本和贯穿输入例子；新增全章数据流、注意力头、Pre-LN Block 和生成时序四张 Mermaid 图；用对比表展示 tokenizer 粒度、位置机制、Attention shape/causal visibility、MHA/GQA/MQA、KV Cache 载荷、长上下文方案和调试不变量。核心公式包括 scaled dot-product attention、causal mask、因果语言模型交叉熵、Pre-LN/Post-LN、LayerNorm/RMSNorm 和 RoPE。第 15 讲以一个完整字符级 miniGPT 脚本闭合输入、训练和生成路径。
- coverage_map：旧稿 tokenization/BPE/WordPiece/SentencePiece/byte 内容归入第 6 讲；embedding/位置概览归第 7 讲；attention 动机归第 8 讲、推导/复杂度/FlashAttention 边界归第 9 讲；MHA/MQA/GQA/KV 估算归第 10 讲；标签右移、causal mask、teacher forcing 和生成归第 11 讲；Block 归第 12 讲，norm/residual/Pre-LN/Post-LN 归第 13 讲；RoPE 和长上下文评估边界归第 14 讲；训练与生成集成归第 15 讲。机械重复的失败模式、易混淆、边界和复盘模板已合并到必要解释及全章收束，专题细节通过第二、五、六、二十四册入口承接。
- evidence：引用原始 Transformer、BPE、BERT/WordPiece、SentencePiece、RoFormer、FlashAttention、GQA、RMSNorm、Pre-LN/Post-LN 和 Lost in the Middle 论文；来源支持范围和外推边界记录在 research/transformer-core-rewrite-2026-10-27.md，并沿用 research/transformer-core-review-2026-10-07.md 的来源核验。正文没有使用外部图片；图示均为本章 Mermaid 自绘。
- code_status：miniGPT 代码按仓库锁定的 Python 3.13+ 与 PyTorch 2.12.0 撰写。本轮没有运行脚本或新增测试；只做静态阅读和 shape/控制流核对，因此不标记为 verified 或 local-demo。
- formula_status：手算 attention 例、KV Cache 字节估算、LayerNorm/RMSNorm 数值和二维 RoPE 相对位移算例；检查行内公式定界符及 math 围栏结构，未使用外部 MathJax/PDF 渲染器。
- link_status：本轮静态确认正文所有相对 Markdown 链接目标存在；外链按原始论文页面引用，本轮未重新检查外网 HTTP 状态。
- sync_status：book-01-core-30/目录.md 已按四阶段路线更新。简介与第一册定位对本章的承诺一致；题库、练习、论文索引和术语表保留各自职责，不重复复制章节摘要。进阶专题入口指向第二册架构/长上下文、第五册模型架构、第六册推理/KV Cache 和第二十四册 KV Cache 实现。
- open_risks：代码未运行；未做公式外部渲染、PDF 构建或改写后第一册连续复读；本轮未核对外链连通性。上述事项不改变本次章节结构重写完成状态，但不能据此声称全册审计或代码验证完成。
- status：完成第二部分正文重构、目录同步、旧内容去向映射、来源范围记录和声明范围内的静态检查。

### WRITING_SPEC 增补代码验证状态复用规则（2026-10-27）

- scope：更新 WRITING_SPEC.md 的代码验证、正文完成门禁和精修记录字段。
- changes：要求重要可运行代码块在旁标明已验证、静态审阅或待验证状态；执行验证后记录日期、环境、依赖、命令、输入和结果。后续精修先比对代码及验证记录；未变化的已验证代码沿用原记录，不重复运行。可执行语句变化（纯排版除外）时复验该块；依赖、接口、环境或预期结果变化时只复验受影响范围，不重跑整章。旧代码无标记时先查已有记录，不因新一轮精修而批量执行；静态审阅不冒称已验证。
- sync_status：规范正文代码流程、完成门禁与附录记录要求已同步；第一册重构章节的 miniGPT 紧邻代码块标为“静态审阅，未运行”，与其既有 code_status 一致。其他章节的代码状态按后续精修时的记录逐步补标；本次不执行代码验证。
- status：规则修订完成；未运行代码或新增测试。

### WRITING_SPEC 明确核心数值例子的教学与验收要求（2026-10-27）

- scope：更新全书架构审视流程、正文教学标准、公式/代码规范、完成门禁和精修记录字段。
- changes：要求每篇知识讲解或方法说明围绕核心问题提供至少一个可跟着复算的数值例子：明确输入、变量/单位、逐步计算、中间量、结果和机制解释；核心公式必须配套数值代入。适合时，图表与代码复用同一组输入，代码补充并核对手算而不取代它。教学构造值须明确标成示例，实测数字需有来源与测量条件。纯定性主题须记录算术例子不适用的原因，并提供带具体输入、状态变化和可观察结果的逐步案例。
- sync_status：例子缺口纳入逐篇结构审视和章节关系图；正文门禁与书册门禁要求可复算例子或等价案例，附录增加 example_status 记录项。
- status：规范修订完成；未修改具体章节或执行代码。

### 第一册 Transformer 核心第二轮教学深化（2026-10-27）

- scope：复读并扩充 `book-01-core-30/chapters/02-transformer核心.md` 第 6—15 讲；保留原四阶段顺序、讲次编号和端到端主线，处理全章过度摘要化的问题。
- changes：各讲增加机制解释、中间步骤和有数字的推演：BPE/Tokenizer 单元计数；Embedding 查表、位置相加和 sinusoidal 数值；Attention 权重混合、T²/因果位置计数、scaled scores/softmax/Value 逐步计算；MHA/GQA shape 与 KV bytes 展开；标签右移和交叉熵；Block 残差与 MLP shape；残差数值及 LayerNorm/RMSNorm 中间计算；RoPE 两组位置对；miniGPT 数据长度、batch/logits/loss shape 和生成长度。明确标注构造值为教学假设，实测结论保留来源与适用条件。
- evidence：新增教学算例及计算边界补入 `research/transformer-core-rewrite-2026-10-27.md`；沿用已有论文来源与支持范围，未扩大实证结论。
- code_status：完整 miniGPT 源码未改，旁注仍为“静态审阅，未运行”；本轮没有执行脚本或测试。
- sync_status：讲次顺序和目录映射未变，无需改目录；更新本章教学内容及对应研究记录。
- formula_status：手工复核新增的向量相加、softmax、交叉熵、shape、计数和字节换算；未用外部公式渲染器。
- status：完成全章教学深度补充；这不代表完成第一册连续复读或 miniGPT 运行验证。

### WRITING_SPEC 增加逐讲篇幅与讲解展开门禁（2026-10-27）

- scope：修订 `WRITING_SPEC.md` 的全册审视流程、篇幅规则、正文教学标准、完成门禁及维护记录字段。
- changes：将独立核心讲解单元的解释性正文下限定为 1,800 个汉字，独立辅助讲解单元下限定为 1,000 个汉字；多讲次 Markdown 按每讲分别统计。明确汉字统计范围、非讲解类内容的适用边界，以及低于门槛不能标记完成、不得用代码/表格/图示凑字数的规则。增加“娓娓道来”的可检查要求：从具体问题和必要前置进入，解释中间步骤与因果，完成例子后解释其含义和边界，再承接后续；摘要和结论清单不得替代解释链。维护记录增加逐讲 `length_status`。
- sync_status：章节结构审视、正文门禁和阻断项均已同步篇幅要求；本次只更新规范和进度，不修改正文、不运行代码或新增测试。既有章节不因旧规范下的完成记录自动满足新篇幅门槛，后续复审须逐讲核算并补足未达项。
- status：规范修订完成。

### WRITING_SPEC 将定位驱动的外部知识扫描纳入书册架构审视（2026-10-27）

- scope：调整 `WRITING_SPEC.md` 的目录审查流程、外部研究要求、书册完成门禁和架构审视记录字段。
- changes：要求书册级架构审视在目录定稿前，根据本册定位和目标能力联网扫描领域近期综述、原始论文、官方文档、技术报告与可信工程实践，识别现有章节之外可能值得新增的主题。对候选议题按证据质量、成熟度、持续价值、本册适配、前置依赖和跨册重复度判断，明确新开章节、并入现章、列为选读、交给其他书册或暂不纳入。候选扫描范围、时间窗口和决定理由须留记录；明确检索不声称穷尽，搜索热度和摘要不能单独作为新增依据。
- sync_status：已同步到固定审视顺序及流程图、诊断表、资料研究要求、书册门禁和 `discovery_scan` 记录字段；本次只更新规范与进度，未对某一册执行网络调研或正文修改。
- status：规范修订完成。

### WRITING_SPEC 增加逐篇正文的章级主题扫描（2026-10-27）

- scope：把外部知识发现流程延伸到单篇重要正文的结构审视和实质性精修。
- changes：要求围绕每篇正文的中心问题、主题边界和领域变化节奏做定向检索，发现可能值得新增的讲次或小节；候选可新增、并入、链接至其他章节、列为选读或暂不纳入，且需记录证据和去向。明确册级扫描负责发现可能影响全书主线的新领域，章级扫描负责发现单篇内部缺失的机制、演进、评估和工程环节；同一轮对同一章的扫描可以复用，主题范围或关键结论变化时再补查。
- sync_status：已同步到固定审视顺序、流程图、诊断表、资料研究要求、书册及正文门禁，以及 `chapter_scan` 维护字段；本次只改规范和进度，没有对具体正文执行联网调研或增改章节。
- status：规范修订完成。
