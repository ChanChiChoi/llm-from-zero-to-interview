# 全书精修进度 v2

本文件记录常态化全书精修的逐章覆盖、修订、验证和遗留项。新模型及新知识点更新另记在 `progress_v3.md`，事实证据放在 `research/`。

## 当前轮次：2026-10-07 启动

- `scope`：从第一册第一份正文开始，按书系顺序覆盖 24 册及 `book-llm-engineer` 补充篇；每份正文逐行复读，并检查必要的简介、目录、索引和配套资料。
- `basis`：当前版 `WRITING_SPEC.md`、`plan.md` 中常态化全书精修要求及仓库 `AGENTS.md`。
- `acceptance`：按内容类型审查问题和动机、连续机制解释、公式/状态、数值例子/代码、工程边界、失败模式、评测、证据范围、链接、导航和按需配套同步；全文复读并验证声明范围内的代码和公式；未完成项明确保留。
- `status`：进行中。第一册五份正文、第二册全部十份正文、第三册全部十一份正文以及第四册第一至十八份正文（全部 18 份）已有本轮验收记录。第四册全部正文已完成整章精修验收。
- `next`：第四册全部 19 个章节文件核验完毕（18 份正文整章验收 + 第 19 份 frontier 更新条目格式/链接核验），整册目录与导航一致。主线下一步：进入第五册第一份正文，按 `plan.md` 4.2/4.4 继续全书精修。
- `open_risks`：全书量大，按章节推进并逐批记录；未实际核验的代码、公式或配套同步不得标为完成。代码验证以可执行且输出能说明机制为准；优先使用 `.tmp/` 下被 Git 忽略的 CPU-only PyTorch 环境，不安装 CUDA 库，也不为性能或 100% 复现扩大依赖。已有章节中的 CUDA、GPU 性能、多节点、真实模型/语料和生产 serving 边界仍按各章记录。PDF 构建 dry-run 因环境缺少 Pandoc 未能执行。

维护者要求本轮从头开始，不沿用此前半成品记录作为覆盖证据。旧的 `progress_v2.md` 记录已清空重建；`progress_v3.md` 保持原样。维护者进一步明确：代码用于展示和解释知识，只要代码经检查可执行且输出有助于说明问题即可；可使用 PyTorch CPU，不安装 CUDA 库，也不要求读者逐一进行完整复现。联网资料使用指定代理 `http://10.24.27.134:7890`。

## 本轮执行记录

### 第四册收尾：第 19 份正文与整册导航核对（2026-10-23）

- `scope`：检查第四册第 19 份正文 `chapters/19-frontier-architecture-updates.md`（frontier 模型更新条目，progress_v3 轨道）的格式与链接完整性；核对整册目录与 19 个章节文件的映射。
- `changes`：第 19 份正文无需修改（无 `$$`、无 `\operatorname`、backtick 围栏闭合）；`progress_v2.md` 修正 1 处笔误（voice clong→cloning）。
- `evidence`：23 个相对链接（book-06/16/17/21 各专题章与 `research/model-update-2026-09/*` 笔记及审计脚本）全部存在；第四册目录第 19 条映射正确。
- `code_status`/`formula_status`/`sync_status`：该文件无代码块与 math 围栏，条目边界与"尚未确认"声明完整，属 progress_v3 轨道维护内容，本轮只做格式与链接核验，不重复登记模型事实。
- `open_risks`：frontier 条目的模型事实核验归属 progress_v3 轨道与 `research/model-update-2026-09/`，其快照随时间失效属预期。
- `status`：第四册 19 个章节文件全部核验完毕，整册目录与导航一致。第四册精修收官；主线下一步按 `plan.md` 进入下一册（第五册第一份正文）。

### 第四册第十八份正文：`book-04-llm-encyclopedia/chapters/18-语音大模型.md`（2026-10-23）

- `scope`：全文复读 18.0—18.23，覆盖 waveform/Mel/speech encoder、ASR 与 WER/CER/关键字段、Whisper 路线、speech-to-text 链路、speech token 四类表示、neural codec 三角关系、TTS 质量维度与 RTF、prosody、speaker embedding 与 voice cloning 授权、speech-to-speech、streaming ASR、VAD/endpointing、turn-taking 与 barge-in 状态机、端到端语音模型、Voice Agent 权限与失败链、系统评估、安全与来源、工程账本；按百科类型验收，未修改长期规范。
- `changes`：修正 3 处 `\operatorname{...}` 为 `\mathrm{...}`；其余无需语义修改。
- `evidence`：9 个唯一外链，6 篇新 arXiv 题名经 abs 页面核对全部一致，其余沿用前轮证据。详见 `research/encyclopedia-speech-llm-review-2026-10-23.md`。
- `code_status`：1 个 Python 块（语音账本）独立执行通过；first_audio 630ms、字段准确率 0.5、4,800 codec token 复算一致。
- `formula_status`：13 个 math 围栏花括号平衡、无禁用宏；WER/CER 分母与可大于 1 的说明、RTF 定义域、T_response 并行语义、P_task 条件分母、C_success 定义域核对通过；`git diff --check` 通过。
- `sync_status`：`PAPERS.md` 语音区已有主要入口；术语表、练习、题库已覆盖语音主题；第四册目录映射正确。
- `open_risks`：未运行真实语音系统；P_task/T_response 为教学分解（正文已声明）；论文题名核对不等于全文复核；PDF 仍受 Pandoc 缺失限制。
- `status`：本章声明范围内精修验收完成。至此第四册 18 份正文全部完成整章精修验收；主线进入第四册收尾核对（目录/导航/剩余文件），随后按计划进入下一册。

### 第四册第十七份正文：`book-04-llm-encyclopedia/chapters/17-图像生成与视频生成.md`（2026-10-23）

- `scope`：全文复读 17.0—17.23，覆盖三条生成路线、扩散前向/反向/参数化、scheduler 与采样预算、U-Net/DiT、VAE 与 latent 压缩、CFG 与负条件、DALL·E 自回归路线、文生图评估、图像编辑保真、视频扩散、时空 patch、时序一致性、text-to-video、world model 与 Sora 边界、FID/FVD、安全与来源、工程账本；按百科类型验收，未修改长期规范。
- `changes`：修正 6 处 `\operatorname{...}` 为 `\mathrm{...}`；修正 2 处链接文字错配：`2402.17177` 实为第三方 Sora 综述（OpenAI 技术报告在其官网未上 arXiv），`1706.08500` 实为 TTUR 论文（提出 FID），均已改为实际题名并注明出处。
- `evidence`：12 个唯一外链，5 篇新 arXiv 题名经 abs 页面核对（含上述 2 处修正），其余沿用前轮证据。详见 `research/encyclopedia-image-video-generation-review-2026-10-23.md`。
- `code_status`：1 个 Python 块（生成账本）独立执行通过；latent cells 4096、零采纳率拒绝断言一致。
- `formula_status`：22 个 math 围栏花括号平衡、无禁用宏；SNR 定义域、D_preserve 分母、T≥2 代理指标、FID/FVD 协议披露、整除性假设核对通过；`git diff --check` 通过。
- `sync_status`：`PAPERS.md` 已覆盖主要生成论文入口，延伸阅读直链保留在正文 17.21；第四册目录映射正确。
- `open_risks`：未运行真实扩散训练或采样；像素级代理指标有正文声明局限；OpenAI world simulators 官网报告未重新核验可达性（openai.com 代理历史 403）；PDF 仍受 Pandoc 缺失限制。
- `status`：本章声明范围内精修验收完成；主线进入第四册第十八份正文 `book-04-llm-encyclopedia/chapters/18-语音大模型.md`，全书 goal active。

### 第四册第十六份正文：`book-04-llm-encyclopedia/chapters/16-多模态学习.md`（2026-10-23）

- `scope`：全文复读 16.0—16.31，覆盖模态表示与采样、modality gap、上下文预算、输入保真与证据召回、CLIP/SigLIP 对齐、视觉编码器与形状审计、VLM 三条路线、连接器（Linear/MLP/Q-Former/Resampler/Cross-Attention）、placeholder 契约、token 压缩、多模态指令微调与模板审计、扩散与条件控制、Latent Diffusion、视频生成与一致性、音频表示与 ASR/TTS、WER/CER/MOS、统一多模态三层含义、评估六层、多模态安全、事故复盘、端到端账本与产品取舍；按百科类型验收，未修改长期规范。
- `changes`：修正 8 处 `\operatorname{...}` 为 `\mathrm{...}`；其余无需语义修改。
- `evidence`：14 个唯一外链，9 篇新 arXiv 题名经 abs 页面核对全部一致，其余沿用前轮证据。详见 `research/encyclopedia-multimodal-learning-review-2026-10-23.md`。
- `code_status`：1 个 Python 块独立执行通过（token 预算 660、支持率 1.0、拒答漏判 0.5、None/0 语义区分断言）；json 样本可解析。
- `formula_status`：64 个 math 围栏花括号平衡、无禁用宏；patch 数 576/2304（4 倍）、196、960,000 等复算一致；F_input/R_evidence/S_support/FVD/C_success 等分母与定义域限定核对通过；`git diff --check` 通过。
- `sync_status`：`PAPERS.md` 多模态区已有本章主要入口；术语表、练习、题库已覆盖对应主题；第四册目录映射正确。
- `open_risks`：未运行真实多模态训练或推理；flicker 代理、成功概率链等为教学模型（正文已声明）；论文题名核对不等于全文复核；PDF 仍受 Pandoc 缺失限制。
- `status`：本章声明范围内精修验收完成；主线进入第四册第十七份正文 `book-04-llm-encyclopedia/chapters/17-图像生成与视频生成.md`，全书 goal active。

### 第四册第十五份正文：`book-04-llm-encyclopedia/chapters/15-面试高频对比表.md`（2026-10-23）

- `scope`：全文复读 15.1—15.43，覆盖比较方法论、Adam/AdamW/SGD、Norm 系列、BERT/GPT/Encoder-Decoder、MHA/MQA/GQA、位置编码三式、激活函数、SFT/RLHF/DPO/RLAIF、采样与解码、LoRA/Adapter/Prefix、量化与蒸馏、PTQ/QAT、FP16/BF16、Flash/Sparse/PagedAttention、KV cache 与重算、三种并行、RAG 与微调、检索与重排、长上下文、Agent 与 Workflow、评估指标与安全边界、Dense 与 MoE；按对比表类型验收，未修改长期规范。
- `changes`：修正 20 处 `\operatorname{...}` 为 `\mathrm{...}`；其余无需语义修改。
- `evidence`：15.43 的 20 个 arXiv 链接经指定代理全部 HTTP 200；AdamW/LayerNorm/RMSNorm 三篇新题名核对一致，其余沿用前轮证据。详见 `research/encyclopedia-comparison-tables-review-2026-10-23.md`。
- `code_status`：1 个 Python 块（Adam/AdamW 最小更新）独立执行通过；pass@k 例（n=20,c=4,k=5）=0.718 与 AdamW 衰减因子 0.99 复算一致。
- `formula_status`：34 个 math 围栏花括号平衡、无禁用宏；FNR/FalseRefusalRate/UnsafePassRate 分母、pass@k 定义域、KVBytes 量纲核对通过；`git diff --check` 通过。
- `sync_status`：15.43 入口与 `PAPERS.md` 重叠良好；术语表与题库已覆盖高频比较主题；第四册目录映射正确。
- `open_risks`：未运行真实优化器、量化、检索或解码实验；比较结论依赖 workload（正文已声明）；PDF 仍受 Pandoc 缺失限制。
- `status`：本章声明范围内精修验收完成；主线进入第四册第十六份正文 `book-04-llm-encyclopedia/chapters/16-多模态学习.md`，全书 goal active。

### 第四册第十四份正文：`book-04-llm-encyclopedia/chapters/14-重要论文索引.md`（2026-10-23）

- `scope`：全文复读 14.1—14.45 及小结，覆盖论文阅读方法论、架构/位置/系统/后训练/推理/Agent/多模态/安全各论文条目、谱系继承与修正、阅读记录模板与按问题组织的阅读路径；按索引章类型重点核对 14.45 全部链接与题名，未修改长期规范。
- `changes`：修正 13 处 `\operatorname{...}` 为 `\mathrm{...}`；修正 1 处题名错配（`2211.00593` 实为 Indirect Object Identification 电路论文，非 "A Mathematical Framework for Transformer Circuits"），后者未上 arXiv，改指 transformer-circuits.pub 官方入口并保留两条并列条目。
- `evidence`：51 个 Markdown 引用逐条核验，49 个唯一入口题名与链接一致，GPT-1 PDF 与 GSM8K 仓库 HTTP 200，transformer-circuits.pub HTTP 200；重复 ID 沿用前轮证据。详见 `research/encyclopedia-paper-index-review-2026-10-23.md`。
- `code_status`：1 个 Python 块（证据账本）独立执行通过；LoRA 参数量与 ViT patch 数复算与正文一致。
- `formula_status`：30 个 math 围栏花括号平衡、无禁用宏；GainPerCost 定义域、GRPO 零标准差分支、ASR 分母等限定核对通过；`git diff --check` 通过。
- `sync_status`：本章即论文索引，条目与 `PAPERS.md` 互补无需新增；第四册目录映射正确。
- `open_risks`：题名核对不等于全文与实验复核；Kaplan 分离项为教学近似（正文已声明）；PDF 仍受 Pandoc 缺失限制。
- `status`：本章声明范围内精修验收完成；主线进入第四册第十五份正文 `book-04-llm-encyclopedia/chapters/15-面试高频对比表.md`，全书 goal active。

### 第四册第十三份正文：`book-04-llm-encyclopedia/chapters/13-安全与治理.md`（2026-10-23）

- `scope`：全文复读 13.1—13.27，覆盖威胁模型、对齐错配、scalable oversight、3H 冲突、指令层级、jailbreak/注入、taint 数据流、拒答曲线、PII 与记忆提取、DP-SGD、数据治理、水印与 C2PA、红队与危险能力、分层安全评估、事故响应、机制可解释性、steering、model editing、unlearning、工具/Agent 安全、模型卡与治理发布、偏见公平毒性、审核 guardrail、供应链和安全工程账；按百科类型验收，未修改长期规范。
- `changes`：修正 25 处 `\operatorname{...}` 为 `\mathrm{...}`；其余无需语义修改。
- `evidence`：18 个唯一外链（11 篇 arXiv、MITRE ATLAS、NIST、OWASP x2、SLSA、C2PA）经指定代理均 HTTP 200；11 篇 arXiv 题名经 abs 页面核对全部一致；章首两条书内相对链接目标存在。详见 `research/encyclopedia-safety-governance-review-2026-10-23.md`。
- `code_status`：3 个 Python 块逐块独立执行通过（风险排序、taint 传播、activation patching 线性干预）。
- `formula_status`：40 个 math 围栏花括号与环境平衡、无禁用宏；Wilson 区间、DP-SGD、水印 z 分数、FNR/ASR/拒答率等分母与定义域限定核对通过；`git diff --check` 通过。
- `sync_status`：`PAPERS.md` Safety 区新增 11 个论文直链；术语表/练习/题库已有对应主题；第四册目录映射正确。
- `open_risks`：未运行真实攻击、红队、DP 训练或模型编辑实验；水印检测为教学简化；论文题名核对不等于全文复核；PDF 仍受 Pandoc 缺失限制。
- `status`：本章声明范围内精修验收完成；主线进入第四册第十四份正文 `book-04-llm-encyclopedia/chapters/14-重要论文索引.md`，全书 goal active。

### 第四册第十二份正文：`book-04-llm-encyclopedia/chapters/12-reasoning与评估.md`（2026-10-23）

- `scope`：全文复读 12.1—12.17，覆盖推理对象与候选、CoT/Scratchpad、Self-Consistency、Verifier 与校准、PRM/ORM 过程监督、搜索推理与 TTC、数学与代码推理、统计比较、Benchmark 污染、人评与 LLM judge、Factuality/Robustness/Calibration、推理安全与发布决策；按百科类型验收，未修改长期规范。
- `changes`：修正 13 处 `\operatorname{...}` 为 `\mathrm{...}`（arg max、ECE、Acc、Conf、BS、TopK、UCT、pass@k、CI）；其余无需语义修改。
- `evidence`：11 个唯一外链（9 篇 arXiv、HELM、lm-evaluation-harness）经指定代理均 HTTP 200；9 篇 arXiv 题名经 abs 页面核对全部一致。详见 `research/encyclopedia-reasoning-eval-review-2026-10-23.md`。
- `code_status`：2 个 Python 块独立执行通过；多数投票、配对 lift=0.2 与 bootstrap 区间、pass@k 公式复算一致。
- `formula_status`：35 个 math 围栏花括号平衡、无禁用宏；回归率/成对提升/bootstrap/pass@k/严重度加权的分母与定义域限定核对通过；`git diff --check` 通过。
- `sync_status`：`PAPERS.md` Reasoning 区已覆盖本章全部论文入口；术语表与练习已有对应条目；第四册目录映射正确。
- `open_risks`：未运行真实模型、verifier、搜索或评估 harness；bootstrap 为教学实现；论文题名核对不等于全文复核；PDF 仍受 Pandoc 缺失限制。
- `status`：本章声明范围内精修验收完成；主线进入第四册第十三份正文 `book-04-llm-encyclopedia/chapters/13-安全与治理.md`，全书 goal active。

### 第四册第十一份正文：`book-04-llm-encyclopedia/chapters/11-长上下文rag与agent.md`（2026-10-23）

- `scope`：全文复读 11.1—11.25，覆盖长上下文预算与位置敏感性、上下文工程、RAG 架构、检索与融合、证据状态与 Agentic RAG、引用与版本、Agent 抽象与规划、工具契约与执行、MCP、A2A、Memory、Code/Browser/Multi-Agent、组合边界、trace/replay、评估、提示注入安全、成本延迟和端到端设计；按百科类型验收，未修改长期规范。
- `changes`：正文无需语义修改；格式与来源核验通过。MCP 规范固定 2025-06-18 版本、A2A 固定 latest 并要求实现绑定版本的写法保留。
- `evidence`：12 个唯一外链中 10 个经指定代理核验 HTTP 200（含全部 6 篇 arXiv、MCP/A2A 规范、Anthropic、HELM）；6 篇 arXiv 题名经 abs 页面核对全部与引用主题一致。详见 `research/encyclopedia-long-context-rag-agent-review-2026-10-23.md`。
- `code_status`：5 个 Python 块逐块独立执行，5/5 通过；RRF 数值复算排名 B > A > D > C 与正文一致；退避 [1,2,4,4,4] 与 budget 拒绝一致。
- `formula_status`：23 个 math 围栏花括号平衡；工具调用 Precision/Recall、SupportRate、Pass@k、成本/延迟分解的分母非空与定义域限定核对通过；`git diff --check` 通过。
- `sync_status`：`PAPERS.md` Agent 区补入 ReAct、Toolformer 直链与 SWE-bench 入口；术语表和练习已有对应主题，不重复添加；第四册目录映射正确。
- `open_risks`：未运行真实检索器、embedding、MCP/A2A Server、沙箱或生产 Agent；`u(p)` 与 `P(success)` 为教学模型，正文已声明；PDF 仍受 Pandoc 缺失限制。
- `status`：本章声明范围内精修验收完成；主线进入第四册第十二份正文 `book-04-llm-encyclopedia/chapters/12-reasoning与评估.md`，全书 goal active。

### 第四册第十份正文：`book-04-llm-encyclopedia/chapters/10-分布式训练系统.md`（2026-10-23）

- `scope`：全文复读 10.1—10.24 及小结，覆盖训练账本、数据/张量/流水并行、ZeRO/FSDP、重计算与混合精度、效率与 MFU、故障定位、拓扑、网络、存储与 checkpoint、容器与 K8s、多租户调度、平台契约、实验追踪、可观测性、SLO、成本、安全供应链、审计变更和综合实验；按百科类型验收，未修改长期规范。
- `changes`：修正 10.23 的 GPipe 论文题名（实际为 "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism"）；修正 Data Cards 错链（`2203.15556` 实为 Chinchilla 论文，改为 `2204.01075`）；`\operatorname{Optimizer}` 改为 `\mathrm{Optimizer}`；围栏风格与第四册已验收章节一致，保持 `~~~` 不变。
- `evidence`：18 个唯一外链经指定代理全部 HTTP 200；5 篇 arXiv 题名经 abs 页面核对，2 处错误已修正，3 处一致。详见 `research/encyclopedia-distributed-training-review-2026-10-23.md`。
- `code_status`：11 个标准库 Python 块逐块独立执行，11/11 通过；未安装 CUDA 或额外依赖。
- `formula_status`：32 个 math 围栏花括号平衡、无禁用宏；复算 ring 14 GiB、bubble 0.2、ZeRO 30 GiB、dominant share 0.25、burn rate 2.0、global batch 64 及 TP 切分 1376/512，均与正文一致；`git diff --check` 通过。
- `sync_status`：`PAPERS.md` 补入 Megatron-LM 集群训练与 ZeRO-Infinity 入口；`EXERCISES.md`、`INTERVIEW_BANK.md`、`GLOSSARY_EN_ZH.md` 已覆盖本章主题，不重复添加；第四册目录映射正确。
- `open_risks`：未在真实 GPU/NCCL/多节点环境验证通信、MFU、吞吐或故障恢复；未运行 ZeRO/FSDP/Megatron 实际训练；教学公式为数量级近似；PDF 仍受 Pandoc 缺失限制。
- `status`：本章声明范围内精修验收完成；主线进入第四册第十一份正文 `book-04-llm-encyclopedia/chapters/11-长上下文rag与agent.md`，全书 goal active。

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
