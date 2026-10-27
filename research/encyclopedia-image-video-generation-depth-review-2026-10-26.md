# 第四册第十七章《图像生成与视频生成》当前轮次深度复审（2026-10-26）

## 复审范围与问题图谱

- 对象：`book-04-llm-encyclopedia/chapters/17-图像生成与视频生成.md`，原有 17.0—17.23；复审前 724 行。本记录是 `progress_v2.md` 当前轮次覆盖证据。2026-10-23 的旧研究记录保留为历史材料，不代替当前轮复审。
- 依据：`WRITING_SPEC.md` 知识深度、方案对照、证据分层、继承覆盖图和代码/公式检查门槛。
- 核心问题图谱：图像路线把成本放在何处；噪声参数化和 scheduler 如何匹配；少步采样是改路径还是训练蒸馏；编辑指令、输入保真和空间控制分别由什么机制承担；视频模型如何处理配对字幕、时空计算和长程一致；FID/FVD 分数在哪些情况下失真；如何从公开厂商材料区分技术披露、产品能力和未知实现。
- 本轮结果：没有删除原有 17.0—17.23 主题；增加有来源支撑的代表方法对照、评测边界与 2026 Sora 状态。章节仍保持单文件，未新增子章节文件。

## 研究带来的正文增量

### 图像生成路线与表示选择

| 方法 | 实际解决的问题与机制 | 代价、证据范围和后续追问 |
| --- | --- | --- |
| [DALL·E 1](https://arxiv.org/abs/2102.12092) | dVAE 把图像压成离散 token，自回归 Transformer 按文本和前序图像 token 预测。 | next-token 接口统一，但序列串行且图像 tokenizer 的重建能力形成上限。 |
| [DALL·E 2 / unCLIP](https://arxiv.org/abs/2204.06125) | prior 由 caption 预测 CLIP image embedding，diffusion decoder 将 embedding 合成为图像；论文比较了 AR 与 diffusion prior。 | 论文报告其设置下 diffusion prior 更高效且样本质量更高；CLIP embedding 保留语义/风格但不含全部像素细节。不是与 DALL·E 1 的同条件对比。 |
| [Imagen](https://arxiv.org/abs/2205.11487) | 以 T5 文本编码器提供语言条件，以级联扩散生成和放大图像；作者消融研究文本编码器规模的收益。 | COCO FID 7.27 与 DrawBench 人评绑定其模型、数据和评测；级联增加组件和推理阶段。 |
| [Latent Diffusion](https://arxiv.org/abs/2112.10752) | 感知压缩 autoencoder 先把图像映射到低维 latent，在 latent 网格执行条件扩散。 | 降低高分辨率去噪成本；VAE 信息损失会限制文字、细线和高频纹理。 |
| [DiT](https://arxiv.org/abs/2212.09748) | 把 latent 切成 patch，使用 Transformer 并比较 in-context、cross-attention、adaLN/adaLN-Zero 条件注入。 | 论文报告 class-conditional ImageNet 256×256 的 2.27 FID，并分析 FLOPs/patch size scaling；该分数不是 text-to-image 对齐结果。patch 变小会增加 token-pair 预算。 |

### 采样与条件控制

- [DDIM](https://arxiv.org/abs/2010.02502) 在保持训练目标的前提下选择稀疏且可确定的采样轨迹；[DPM-Solver](https://arxiv.org/abs/2206.00927) 用 probability-flow ODE 的专用高阶积分器降低模型评估次数。二者主要改变推理路径，不为每种 sampler 重新训练权重。
- [Progressive Distillation](https://arxiv.org/abs/2202.00512) 将教师两次确定性更新蒸馏为学生一次更新并逐次减半步数。论文报告 CIFAR-10 等任务中从最高 8192 步压到 4 步，CIFAR-10 FID 为 3.0；不能外推成现代文生图或视频模型的质量保证。
- [Diffusers Scheduler 官方文档](https://huggingface.co/docs/diffusers/main/en/using-diffusers/schedulers)具体展示 timestep spacing、sigma 和由现有 scheduler config 初始化另一 scheduler 的参数。该动态文档说明接口配置，不证明所有 scheduler/checkpoint 组合兼容。
- CFG 正文补充 condition dropout 训练方式，按给定公式解释 `w=0/1/>1`，用二维数值例子说明外推，并提醒不同实现对 guidance scale 的定义可能偏移一位。
- `v` prediction 增加线性变换和逆变换推导，给出 `\bar{\alpha}=0.36` 的转换例子，说明 scheduler 将错误参数化当成噪声时为何 shape 正常但生成语义错误。

### 图像编辑方法

| 方法 | 机制和解决的问题 | 限制 |
| --- | --- | --- |
| [SDEdit](https://arxiv.org/abs/2108.01073) | 输入先加噪再由 diffusion prior 去噪，不需每类任务专门训练；噪声强度调节写实度与输入忠实度。 | 强度高时布局/身份容易漂移，强度低时不真实痕迹保留。 |
| [InstructPix2Pix](https://arxiv.org/abs/2211.09800) | GPT-3 生成指令、Stable Diffusion/Prompt-to-Prompt 合成图像对，训练模型直接按原图+自然语言指令编辑。 | 合成数据域差异可能导致漏改或非目标区域漂移；快速前向不等于编辑正确。 |
| [ControlNet](https://arxiv.org/abs/2302.05543) | 额外空间分支把边缘、姿态、深度等条件注入预训练扩散 backbone。 | 结构对齐和条件图准备增加成本；满足姿态不自动满足语义或安全约束。 |

正文以“编辑请求的权威信号是什么”组织选型：风格/属性文字优先试指令路线，草图写实可调 SDEdit，布局/姿态需结构条件或 mask，并分别测目标区域完成度与非目标区域保真。

### 视频生成结构演进

| 方法 | 论文尝试解决的瓶颈 | 代价/边界 |
| --- | --- | --- |
| [Video Diffusion Models](https://arxiv.org/abs/2204.03458) | 将扩散建模扩展到视频时空表示，处理动作和帧间一致性。 | 序列和激活随帧数/分辨率增加，图像质量不代表时间连续。 |
| [Make-A-Video](https://arxiv.org/abs/2209.14792) | 复用文生图模型学习外观/语言，并从未配字幕视频学习运动，减少 text-video 配对依赖。 | 迁移图像 prior 不会自动产生动作理解；高分辨率链路还包含时空模块、插帧和超分。 |
| [Imagen Video](https://arxiv.org/abs/2210.02303) | base video diffusion 后串接空间和时间超分，结合 `v` 参数化与 CFG progressive distillation。 | 多阶段 inference 造成级联成本和误差传播；蒸馏结果需按目标 checkpoint 回归。 |
| [CogVideoX](https://arxiv.org/abs/2408.06072) | 3D causal VAE 压缩时空维度；expert Transformer 深度融合文本与视频 latent；渐进训练、multi-resolution frame pack 和稠密 caption 支持更长视频。 | latent 重建、caption 质量和数据筛选仍是上游瓶颈；论文自动指标与人评绑定其评测集。 |
| [Wan 2.1](https://arxiv.org/abs/2503.20314) | 3D causal Wan-VAE 约压缩至 `[1+T/4,H/8,W/8]` 并增 latent 通道，DiT 处理 patchified latent；feature cache 用于分块 VAE 推理。 | 报告的 1.3B/14B、任务、速度和显存绑定论文/仓库配置，不能当成任意机器承诺。需要同时审计压缩误差、时间边界和注意力 token 数。 |

这些论文的 benchmark、帧率、分辨率、模型预算均不同；正文只沿瓶颈迁移讲演进，不把数字拼成统一排行榜。

### 图像与视频评估

- FID/FVD 改为对称半正定协方差平方根写法，解释其高斯特征分布近似和分布级语义。
- FID 增加 ImageNet 类别特征可能奖励无关类别分布匹配的例子（[The Role of ImageNet Classes in FID](https://arxiv.org/abs/2203.06026)）；[Rethinking FID](https://arxiv.org/abs/2401.09603)质疑 Inception 特征对现代文生图的覆盖、有限样本稳定性和与人评的一致性，并提出 CLIP 特征 CMMD。正文没有把 CMMD 写成适用于所有业务的替代标准。
- 图像 prompt benchmark 对照 [DrawBench](https://arxiv.org/abs/2205.11487)、[GenEval](https://arxiv.org/abs/2310.11513) 与 [T2I-CompBench](https://arxiv.org/abs/2307.06350)，分别覆盖人工挑战 prompt、人/对象组合属性、绑定关系/空间关系；都受样本设计与自动评估器能力限制。
- [VBench](https://arxiv.org/abs/2311.17982) 拆成 16 个视频维度；[VBench-2.0](https://arxiv.org/abs/2503.21755)评估人类保真、可控性、创造性、物理和常识等 5 类/18 个细项，结合 VLM/LLM 与专用模型并报告人评对齐。正文说明自动 judge 并非真值，需抽样人工复核。
- [Beyond FVD](https://arxiv.org/abs/2410.05203)指出 I3D 特征空间非高斯、对某些时间扭曲不敏感且估计需要大量样本，并提出 JEDi。正文把它作为对 FVD 的方法学挑战，不宣称 JEDi 已普遍取代 FVD。

### Sora 的公开材料与时效

- [OpenAI 2024 技术报告](https://openai.com/index/video-generation-models-as-world-simulators/)公开 latent 压缩、spacetime patches 和 diffusion Transformer 的表示路径，同时明确不含完整模型/实现细节；报告列出玻璃破碎、对象状态与长片段连贯等局限。
- [Sora 2 System Card](https://openai.com/index/sora-2-system-card/)于 2025 年描述物理表现、同步音频、可控性和安全缓解，是发布方系统资料而非完整架构论文。
- 当前 OpenAI [Sora 页面](https://openai.com/index/sora/)标注 Sora 产品自 2026-04-26 起不再提供。正文把它作为研究和治理案例，不当作可调用产品状态。
- Sora 2 的公开系统资料给出输入 prompt/图片审核、生成后视频帧/音频转录扫描和阻断等层次；此为特定版本发布方说明，不作其它系统默认安全保证。

## 原有知识点去向（coverage_map）

| 原章节 | 原有主题 | 当前处理与新位置 |
| --- | --- | --- |
| 17.0—17.1 | 生成任务与扩散/自回归/统一路线 | 全部保留；17.1 增加 DALL·E 1、DALL·E 2、Imagen、LDM、DiT 问题迁移对照。 |
| 17.2—17.4 | 扩散定义、forward process、SNR、epsilon/x0/v、反向均值 | 全部保留；17.4 增加 v 预测矩阵逆变换和数值诊断。 |
| 17.5 | scheduler、采样预算、可复现字段 | 全部保留；增加 DDIM/DPM-Solver/Progressive Distillation 的免重训/重训对照与 Diffusers 文档边界。 |
| 17.6—17.8 | U-Net、DiT、VAE、latent diffusion、CFG | 全部保留；DiT 补 adaLN-Zero 与 patch token-pair 成本；CFG 补训练、外推、scale 约定和调用开销。 |
| 17.9—17.10 | DALL·E、自回归生成、文生图对齐 | 全部保留；明确 DALL·E 2 prior/decoder 与 ImageNet FID 证据不可外推。增加 DrawBench/GenEval/T2I-CompBench。 |
| 17.11 | img2img、inpainting、ControlNet、编辑保真 | 全部保留；增加 SDEdit/InstructPix2Pix/ControlNet 的输入信号和失效对照。 |
| 17.12—17.15 | video diffusion、时空 token、temporal consistency、motion prior、T2V | 全部保留；增加 Make-A-Video/Imagen Video/CogVideoX/Wan 的方案演进和证据边界。 |
| 17.16 | world model、Sora 公开证据 | 全部保留；更新 Sora 1 技术报告限制、Sora 2 System Card 证据层、2026-04-26 产品状态。 |
| 17.17—17.18 | FID/FVD、人工/任务型评估 | 全部保留；修正矩阵平方根，补指标假设、FID/FVD 已知盲区与 GenEval/T2I/VBench。 |
| 17.19 | 安全、版权和 provenance | 全部保留；增加 Sora 2 系统卡分层安全实例并绑定具体配置。 |
| 17.20 | 质量/成本账本、Python toy | 保留；把 `cost_per_accepted` 改为 `latency_per_accepted_ms`，修正代码将毫秒命名成货币成本的问题，明确遗漏的 pipeline 开销。代码未运行。 |
| 17.21—17.23 | 证据等级、延伸阅读、六项练习、小结 | 全部保留；增补论文与官方来源，增加代表路线选择练习；原六项练习均在。 |

## 公式、代码和链接状态

- `v` 参数化公式按 `v=aε-sx0` 约定求逆；示例 `\bar{\alpha}=0.36` 得 `a=0.6,s=0.8`。已提醒 checkpoint `prediction_type` 和 scheduler 必须匹配。
- FID/FVD 均写作 `Tr(Σr+Σg-2(Σr^(1/2)ΣgΣr^(1/2))^(1/2))`；矩阵平方根对称半正定，且使用基于特征高斯分布近似的语义明确。
- DiT patch 例：32×32 latent 从 patch=8 的 16 个 token 改成 patch=4 的 64 个 token，密集 attention token-pair 从 256 增到 4096（16 倍）。
- 工程 toy 静态核对：512/8 得 4,096 latent cells；30×40ms/0.60=2,000ms/accepted，12×40ms/0.20=2,400ms/accepted。代码已显式声明这是 latency proxy 而不是 monetary cost。未运行 Python block 或任何测试。
- 本章新增 URL 均直连论文、OpenAI 官方报告/system card、Diffusers 官方文档；保留原有链接。本轮未逐一重新打开旧章节的全部外链。
- 本轮未构建 PDF；不涉及章节拆分或新增导航目标。已同步 PAPERS 与 INTERVIEW_BANK。

## 尚未验证或外推风险

- 未训练/运行任何图像或视频模型，未实测采样延迟、VAE 重建、帧率、显存、CF guidance 或模型排名。
- 论文表中的 FID、COCO/DrawBench 人评、Distillation 的 CIFAR-10 数字、Wan 的模型预算都只代表原文设定，不是当前产品比较。
- `v` 参数定义、scheduler 配置和 guidance scale 命名在实现间可能不同，落到具体 checkpoint 时需读取其配置/代码版本。
- VLM/LLM benchmark judge、FID/FVD 特征和 prompt suite 都会带来自己的测量偏差；高风险或业务结论应加人工与任务型指标。
- Sora 页面和 Diffusers `main` 文档会随时间更新；产品可用状态在正文中注明访问日期与官方页面口径，动态接口应按目标版本复查。
