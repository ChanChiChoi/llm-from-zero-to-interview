# 全书精修进度 v2

本文件记录常态化全书精修的逐章覆盖、修订、验证和遗留项。新模型及新知识点更新另记在 `progress_v3.md`，事实证据放在 `research/`。

## 当前轮次：2026-10-07 启动

- `scope`：从第一册第一份正文开始，按书系顺序覆盖 24 册及 `book-llm-engineer` 补充篇；每份正文逐行复读，并检查必要的简介、目录、索引和配套资料。
- `basis`：当前版 `WRITING_SPEC.md`、`plan.md` 中常态化全书精修要求及仓库 `AGENTS.md`。
- `acceptance`：按内容类型审查问题和动机、连续机制解释、公式/状态、数值例子/代码、工程边界、失败模式、评测、证据范围、链接、导航和按需配套同步；全文复读并验证声明范围内的代码和公式；未完成项明确保留。
- `status`：进行中。第一册五份正文和第二册第一至九份正文已完成；第二册第十份正文正在精修，目前完成第 105—107 讲的首轮复读、修订和局部代码验证，整份正文未完成。
- `next`：继续核对第二册第十份正文 `book-02-advanced-100/chapters/10-多模态大模型进阶.md` 第 105—107 讲的来源访问状态，再从头审阅第 108 讲及后续内容；全章来源和剩余工作待完成后统一记录到 `research/`。
- `open_risks`：全书量大，按章节推进并逐批记录；未实际核验的代码、公式或配套同步不得标为完成。代码验证以可执行且输出能说明机制为准；优先使用 `.tmp/` 下被 Git 忽略的 CPU-only PyTorch 环境，不安装 CUDA 库，也不为性能或 100% 复现扩大依赖。已有章节中的 CUDA、GPU 性能、多节点、真实模型/语料和生产 serving 边界仍按各章记录。PDF 构建 dry-run 因环境缺少 Pandoc 未能执行。

维护者要求本轮从头开始，不沿用此前半成品记录作为覆盖证据。旧的 `progress_v2.md` 记录已清空重建；`progress_v3.md` 保持原样。维护者进一步明确：代码用于展示和解释知识，只要代码经检查可执行且输出有助于说明问题即可；可使用 PyTorch CPU，不安装 CUDA 库，也不要求读者逐一进行完整复现。联网资料使用指定代理 `http://10.24.27.134:7890`。

## 本轮执行记录

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
