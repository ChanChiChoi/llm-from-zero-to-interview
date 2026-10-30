# 视频生成、交互式世界建模与 Sora 生命周期来源

核验日期：2026-10-29。

本记录只保存联网核验得到的技术事实及证据范围，不记录章节审阅、正文改动或工作进度。

## 视频生成方法演进

- **来源一：** Ho et al., [Video Diffusion Models](https://arxiv.org/html/2204.03458)，arXiv:2204.03458v2，2022-06-22。
- **阅读范围：** 摘要、§3 中 3D U-Net 空间/时间分解（HTML §3，模型结构段）。
- **支持事实：** 论文以 3D U-Net 处理定长帧块，把空间卷积写成只跨 H/W 的 1×3×3 卷积，空间注意力后接沿帧轴的时间注意力，并联合图像与视频建模；长视频使用条件自回归延展。
- **证据边界：** 以上是论文模型结构和其实验设定，不说明所有视频扩散系统都采用同一分解，也不直接比较后续系统的质量或成本。

- **来源二：** Singer et al., [Make-A-Video: Text-to-Video Generation without Text-Video Data](https://arxiv.org/html/2209.14792)，arXiv:2209.14792v1，2022-09-29。
- **阅读范围：** 摘要、§3.2–3.4（时空层、插帧网络与训练设计）。
- **支持事实：** 论文使用配对图文学习文本与视觉语义、无文本配对视频学习运动；将预训练文生图模型扩展为空间/时间卷积及注意力模块，并加入视频解码、插帧和空间/时间超分辨率。
- **证据边界：** 训练数据来源、结构和收益限于论文中的 Make-A-Video 实例；作者对质量和效率的比较受其模型、数据和评估条件约束。

- **来源三：** Ho et al., [Imagen Video: High Definition Video Generation with Diffusion Models](https://arxiv.org/html/2210.02303)，arXiv:2210.02303v1，2022-10-05。
- **阅读范围：** 摘要、§2.2–2.7、实验中的模型规模与采样结果。
- **支持事实：** 论文描述由 7 个子模型构成的视频扩散级联；其配置输出 1280×768、24 fps、128 帧（约 5.3 秒）。系统以 base video model 和交替空间/时间超分辨率模型构成。
- **证据边界：** 规模、分辨率、帧率和时长是作者特定模型设置，不代表同类架构的一般能力；级联的跨窗口一致性批评需归属提出该批评的具体来源。

- **来源四：** Bar-Tal et al., [Lumiere: A Space-Time Diffusion Model for Video Generation](https://arxiv.org/html/2401.12945)，arXiv:2401.12945v2，2024-02-05。
- **阅读范围：** 摘要、§1、§3 架构说明及限制讨论。
- **支持事实：** 论文以 Space-Time U-Net 在多个时空尺度上处理视频完整时长，再使用重叠时间窗口和 MultiDiffusion 做空间超分辨率。作者将先生成稀疏关键帧、再做局部时间超分辨率的级联路线的长程一致性困难归因于运动混叠、有限时间窗口及级联训练域差异；论文也明确空间超分辨率受显存约束，并将多镜头场景转场列为未解决范围。
- **证据边界：** 对级联缺陷的描述属于论文作者的分析和对比协议，不证明所有级联都会失败，也不证明全片生成在所有分辨率、时长或硬件上更优。

- **来源五：** Yang et al., [CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/html/2408.06072)，arXiv:2408.06072v3，2025-03-26（原始 v1 于 2024-08-12 发布）。
- **阅读范围：** 摘要、§2.1–2.2、§3.1–3.4。
- **支持事实：** 论文用 3D 因果 VAE 同时压缩空间和时间维度，将视觉 latent patch 与文本 embedding 拼接后送入 expert Transformer；提出 3D 全注意力和渐进式训练，并报告特定 CogVideoX 配置可生成 10 秒、16 fps、768×1360 视频。
- **证据边界：** 架构、公开能力、时长和分辨率限于该论文版本与模型设置；作者的自动指标和人工评估不能用来与不同数据、分辨率、采样配置的模型做无条件排名。

## Sora：历史研究说明与服务生命周期

- **来源一：** OpenAI，[Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/)，2024-02-15 官方研究说明。
- **阅读范围：** 研究说明摘要与“Turning visual data into patches”开头。
- **支持事实：** 页面将当时 Sora 描述为文本条件的图像/视频扩散模型，联合处理不同持续时间、分辨率和纵横比，并说明其表示使用时空 patch。报告聚焦统一视觉表示和能力/限制的定性评估；页面明确表示不包括模型和实现细节。
- **证据边界：** 该页面支持 2024 年公开的研究范围和作者陈述，不披露 tokenizer、完整架构、数据配方或训练规模，不能据此推断这些内部细节，也不代表后续服务状态。
- **来源二：** OpenAI Help Center，[What to know about the Sora discontinuation](https://help.openai.com/en/articles/20001152-what-to-know-about-the-sora-discontinuation)，访问日 2026-10-29。
- **支持事实：** 官方说明 Sora 网页与应用于 2026-04-26 停止，Sora API 于 2026-09-24 停止。
- **证据边界：** 这是官方服务生命周期信息，不是模型能力或研究结果；它与 2024 年技术报告所述研究机制属于不同时间和证据类型。

## WBench

- **来源：** Ying et al., [WBench: A Comprehensive Multi-turn Benchmark for Interactive Video World Model Evaluation](https://arxiv.org/html/2605.25874)，arXiv:2605.25874v1，2026-05-25。
- **阅读范围：** 摘要、§3 数据集构造与统计、§4 评估套件、§5 实验与人类偏好对齐。
- **支持事实：** 论文按视频质量、场景遵循、交互遵循、一致性和物理符合度五维组织评估；报告 289 个测试案例、1,058 轮交互，覆盖导航、主体动作、事件编辑、视角切换四类交互。其评估包括 22 个自动子指标，结合专用视觉模型与多模态模型；作者报告这些指标经过人工判断校验，并在 20 个模型上进行了实验。
- **证据边界：** 以上数据、自动评分器、人类校验和模型结果属于论文的 benchmark、接口和协议。指标与案例不能穷尽交互式世界建模，也不能单独证明现实环境中的因果或物理理解。

## World-in-Context

- **来源：** Chen et al., [World-in-Context: Interactive Video World Modeling with In-Context Conditioning](https://arxiv.org/html/2610.33169)，arXiv:2610.33169v1，2026-10-23。
- **阅读范围：** 摘要、§4 控制表示与训练、§5 实验、§6 适用性。
- **支持事实：** 预印本把视角模式、角色动作和摄像机运动编码为共享上下文内的显式控制 token，并提出 Interaction-Transport RoPE（IT-RoPE）用累计控制变化表示跨片段时间关系。作者报告其 5B 模型在 WBench navigation score 为 81.5，与 22B EchoWM 的 81.6 接近；并报告 perspective-switching success 为 93.8%，对比其所测 5B baselines 的 18.8%。
- **证据边界：** 这是 arXiv v1 预印本，数字为作者在其模型、WBench 版本、评估任务和协议下报告的结果，不能视为跨模型通用排名或独立复现。结构设计示例不证明视频生成模型普遍具备世界建模能力。
