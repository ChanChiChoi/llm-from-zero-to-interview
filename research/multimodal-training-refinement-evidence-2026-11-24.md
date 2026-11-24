# 多模态训练精修证据（2026-11-24）

本记录只保留本次第十二章修订中核对的新事实、来源和适用边界；不记录逐章进度。

## 视觉语言训练与阶段化配方

- **Qwen3-VL Technical Report v2**（[arXiv:2511.21631](https://arxiv.org/abs/2511.21631)，2025-11-27）：表 1 报告 S0 merger-only 为 67B/8,192，S1 全模型约 1T/8,192，S2 全模型约 1T/32,768，S3 全模型 100B/262,144。正文说明 S0 冻结 vision encoder 和 LLM；S1 加入文本数据和少量视频；S2 增加长视频与 agent 指令数据；S3 加重长视频、长文档。报告称从 per-sample loss 改为 square-root-normalized per-token loss，但未提供可迁移的完整权重定义，不能据此自行补写通用公式。
- **Qwen3.5-Omni Technical Report v2**（[arXiv:2604.15804](https://arxiv.org/abs/2604.15804)，2026-04-21）：S1 固定 LLM，先训练视觉/音频 adapter，再训练 encoder；S2 序列长度 32,768，分项报告文本 0.92T、音频 1.99T、图像 0.95T、视频 0.14T、音视频 0.29T；S3 将最大序列长度提升到 262,144 并增加长音频、长视频数据。分项合计为 4.29T，报告概括为“约 4T”，未进一步解释舍入或统计口径。Thinker 的 OPD 用文本条件下生成的回答作为配对音频条件查询的蒸馏目标；报告没有把它定义为通用的同前缀全词表 logits 对齐算法。Talker 阶段另含 DPO、规则奖励与 GSPO。
- **LLaVA-1.5**（[arXiv:2310.03744 v2](https://arxiv.org/abs/2310.03744)，2024-05-15）：使用 CLIP ViT-L/336px、MLP connector 和学术任务 VQA 数据；作者报告公开数据 1.2M、13B checkpoint 在 11 个 benchmark 上表现，以及单个 8-A100 节点约一天的完整训练时间。该成本仅对应其模型、数据和训练设置。
- **MM1**（[arXiv:2403.09611](https://arxiv.org/abs/2403.09611)，2024）：1.2B 消融设置中的数据混合为 captioned image 45%、interleaved image-text 45%、text-only 10%，C-Abstractor 输出 144 个视觉 token。该比例和消融结论不能视作其他规模或任务的默认配方。
- **BLIP-2**（[arXiv:2301.12597](https://arxiv.org/abs/2301.12597)，2023）：作者报告在其 zero-shot VQAv2 设置超过 Flamingo-80B，且可训练参数少 54 倍。比较只限论文所述评测与参数口径。

## 文档与生成训练证据

- **Donut**（[arXiv:2111.15664 v5](https://arxiv.org/abs/2111.15664)，2022-10-06）：预训练使用带 CLOVA OCR 伪文字标签的 11M 张英文 IIT-CDIP 扫描文档；SynthDoG 分别基于英文、中文、日文、韩文 Wikipedia 生成每种语言 0.5M 样本。因此推理时 OCR-free 不等于预训练不依赖 OCR 伪标签。
- **Pix2Struct**（[arXiv:2210.03347 v2](https://arxiv.org/abs/2210.03347)，2023-06-15）：base 模型消融总计 100K 步，其中 30K BooksCorpus reading warmup、70K screenshot parsing；DocVQA 验证 ANLS 从完整 recipe 的 67.8 降到移除 screenshot parsing 后的 12.2。完整主模型另训练 270K screenshot-parsing steps，不能把 100K 消融配置误作完整训练预算。
- **CogVideoX**（[arXiv:2408.06072 v3](https://arxiv.org/abs/2408.06072)，2025-03-26）：提出 Multi-Resolution Frame Pack，将不同视频时长与分辨率的样本组织到 batch，并按 data-parallel rank 把 diffusion timestep 区间分段采样。作者报告 40K steps 时验证 timestep 100、300、500、700、900 的 loss 均略低。该结果是论文设置内的消融，不能推出所有视频扩散模型都需 rank 分区采样。
- **Min-SNR-γ**（[arXiv:2303.09556 v3](https://arxiv.org/abs/2303.09556)，2024-03-11）：ImageNet 256×256 的 ViT-B 对照实验中，Min-SNR-5 达到 FID 10 的速度约为对照的 3.4 倍。FID 2.06 属于另一项 ViT-XL、$x_0$ 预测、约 7M iteration 设置；这两个结果不能拼成同一模型或训练预算的承诺。

## 证据等级与未验证边界

以上均为原始论文或作者技术报告中的作者配置/实验结果，属于一手证据；它们证明的是各自论文所述训练条件和评测协议下的发现，不证明在其他数据、硬件、分辨率、模型规模或任务上的普遍收益。对 Qwen3.5-Omni 预算分项与总量之间的差异、以及 Qwen3-VL square-root loss 的完整权重计算，公开报告未提供足以独立重建的全部细节。
