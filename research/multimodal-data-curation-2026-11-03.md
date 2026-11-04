# 多模态数据筛选、对齐与覆盖证据（2026-11-03）

本记录整理多模态数据筛选、长音视频时间对齐和人工质量抽样方面的来源事实。不同数据集、模型、任务及实验配置不可直接拼成通用质量排名。

## 图文筛选

### MLM-Filter

- **来源与版本：** Wang 等，《Finetuned Multimodal Language Models Are High-Quality Image-Text Data Filters》，arXiv v1，2024-03-05。阅读范围为方法、过滤器分数、DataComp 实验与作者分析：[论文 HTML](https://arxiv.org/html/2403.02677v1)。
- **来源支持的事实：** 过滤器以微调多模态语言模型评估四个维度：图文匹配（image-text matching, ITM）、物体细节兑现（object detail fulfillment, ODF）、caption 文本质量（caption text quality, CTQ）和语义理解（semantic understanding, SU）。在 DataComp Large 的 CLIP 预训练设置中，ODF/GPT-4 过滤器筛选 30% 数据时，38 项任务平均得分为 54.2；CLIPScore 对照为 52.9，差 1.3 分。
- **证据等级与边界：** source-backed，属于 arXiv 预印本中的特定数据池、过滤比例、过滤器及 CLIP 训练和评估结果。各分数代表模型过滤信号，不证明样本事实正确、具备使用授权或已完成隐私审查；不同下游任务对分数的响应并不相同。

## 音频转录与时间戳

### CHILDES-Aligned / BEACON

- **来源与版本：** Zheng 等，《CHILDES-Aligned: A Curated Children’s Speech Dataset via Multi-Model Timestamp Ensembling》，arXiv v2，2026-09-24。阅读范围为方法、语料整理、消融、时间戳质量、下游结果和限制：[论文 HTML](https://arxiv.org/html/2607.03670v2)。
- **来源支持的事实：** BEACON 使用四种具有不同架构和时间戳机制的 ASR 系统生成逐词时间戳；分别把词流对齐到可信人工转录，经局部候选搜索与单调动态规划选择候选区间，再以多模型共识决定话语边界。对模型不可靠的边界允许标记缺失/弃权。论文报告约 413 小时的儿童语音修正时间戳语料和约 283 小时的 ASR 训练子集；微调结果在四个域外儿童语音 benchmark 上最高报告平均相对 WER 降低 19.5%。
- **证据等级与边界：** source-backed，数据是英文 CHILDES 儿童语音，收益绑定论文模型、切分、训练集和四个评估集。作者指出弃权以降低错位为代价减少可用量，保留语音会偏向较易识别样本；参数依据有限的人工检查设定，跨语料可能需要重新调节。

### Lipi-Ghor

- **来源与版本：** Hasan 等，《Lipi-Ghor: A Large-Scale, Diarized, Multi-Domain Bengali Speech Corpus and a Scalable Pipeline for Building It》，SALMA 2026 正式论文，发表于 2026-10-28，ACL Anthology 190–201 页。阅读范围包括语料构建、统计、错误分析、伦理说明和附录：[论文 PDF](https://aclanthology.org/2026.salma-1.13.pdf)、[Anthology 记录](https://aclanthology.org/2026.salma-1.13/)。
- **来源支持的事实：** 语料由 998 个 YouTube 视频、596 个频道构成，共 885 小时，覆盖 17 个内容域。记录采用 Speaker、Speech、Transcription、Timestamp（SSTT）结构，结合 Precision-2 自动说话人分离与孟加拉语字幕。另有 17 名志愿标注者投入 30 小时制作人工修订并分域抽样的 gold evaluation split。原始音频不再分发；作者贡献的 video-ID 列表、分离结果、代码和 gold annotations 按 CC-BY-4.0 发布，字幕派生转录仅供非商业研究使用。
- **证据等级与边界：** source-backed，语料是平台来源的孟加拉语弱监督数据。作者说明区域方言只有 4 个视频、1.29 小时；speaker 标签由 Precision-2 产生且未经人工验证，不能视为人工标注的分离基准。字幕目标有噪声，人工 gold 集不等于全语料人工核验；YouTube API 政策和部分人工采集过程影响精确复现。数据发布许可不能替代其他项目的来源与用途审查。

## 视频时序标注与抽样评估

### Kairos

- **来源与版本：** Ming 等，《Kairos: A Dataset for Fine-Grained Video-Language Modeling over Space, Time, and Dynamics》，arXiv v1，2026-09-08。阅读范围为数据集和标注流程、benchmark 设计与统计、泄漏检查、局限及影响说明：[论文 HTML](https://arxiv.org/html/2609.08755v1)。
- **来源支持的事实：** 数据包含 19,004 个视频，平均 17 分钟，合计 5,420 小时，覆盖 202 类场景，重点处理 10–30 分钟视频。每秒采样的描述由 Qwen3-VL-8B-Instruct 驱动；从 820 个视频构建的 benchmark 含 2,870 道带时间依据的问题，作者称这些问题经过 sanity checks 和人工核验。在固定文本-only 检查中，Gemini 3.1 Pro 对 benchmark 问题的准确率比随机基线高 9.6 个百分点；这是十个被测 benchmark 中最低的值，但不为零。作者还指出两个能力单元分别只有 5 和 14 道题。
- **证据等级与边界：** source-backed，标注主要为自动生成。作者明确列出 Qwen3-VL-8B-Instruct 的感知偏差可能系统性进入描述，以及仅依靠文本的跨镜头实体匹配局限。泄漏数值依指定 solver、提示、抽样和随机基线协议；稀疏能力单元的单项准确率方差较大。平台来源可公开访问不自动决定任何新用途的许可。

### HAVEN

- **来源与版本：** Shi 与 Zhang，《HAVEN: Hierarchically Aligned Multimodal Benchmark for Unified Video Understanding》，arXiv v1，2026-05-19。阅读范围包括层级对齐表示、构建流程、标注质量评估及人工评审附录：[论文 HTML](https://arxiv.org/html/2605.19223v1)。
- **来源支持的事实：** benchmark 在 frame、shot 和 video 三层连接视频与文本，并覆盖摘要、时间推理、多模态 grounding 和显著性排序。作者通过人工评估 15 个抽样视频的各层标注；7 位专家参与，每个视频由两位标注者按 1–5 Likert 量表评分，论文报告平均分与标准差。
- **证据等级与边界：** source-backed。15 个视频的人工审查是样本级检查，不是全量语料质量保证；benchmark 分数仍绑定发布数据、标注过程和评测协议。
