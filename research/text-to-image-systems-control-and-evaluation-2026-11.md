# 文生图系统、控制与评估：来源证据

记录日期：2026-11-11

本记录仅保存来源支持的技术事实、版本、阅读位置和证据边界；书籍审阅结论、正文修改历史和后续工作交接写入 `progress/`。

## 系统路线与条件接口

### Latent Diffusion

- **来源：** Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, arXiv v2，<https://arxiv.org/abs/2112.10752v2>。
- **阅读位置：** 论文 §§1–3，重点为 latent-space diffusion 与 cross-attention 条件注入的机制描述。
- **来源支持：** 论文把扩散过程置于预训练 autoencoder 的感知压缩表示中，以降低高分辨率像素空间建模成本；论文将 cross-attention 作为接入文本等条件的通用机制之一。
- **边界：** 压缩倍率、latent scaling、VAE 重建质量和 denoiser 结构依具体 checkpoint 与实现；逻辑 attention score 数量不能直接推出 FLOPs、峰值显存或端到端延迟。

### SDXL 双文本编码器

- **来源：** Podell et al., *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*, arXiv v1，<https://arxiv.org/abs/2307.01952v1>；Hugging Face Diffusers v0.35.1 SDXL pipeline 文档，<https://huggingface.co/docs/diffusers/v0.35.1/en/api/pipelines/stable_diffusion/stable_diffusion_xl>。
- **阅读位置：** SDXL 论文 §2.1/Table 1；Diffusers 文档 `StableDiffusionXLPipeline` 参数说明、`encode_prompt` 与 pooled embedding 参数。
- **来源支持：** 论文列出 CLIP ViT-L 与 OpenCLIP ViT-bigG 两种文本编码器，并显示 pooled text embedding 来自第二编码器。固定版本 Diffusers API 将二者分别表示为 `text_encoder`、`text_encoder_2`，并提供两个 tokenizer 和 pooled prompt embedding 接口。
- **边界：** 这是 SDXL 论文与指定 Diffusers 文档版本的模型/API 配置，不是所有 Stable Diffusion checkpoint 的通用结构；API 参数不保证其他版本或其他 pipeline 相同。

### Stable Diffusion 3 / MM-DiT

- **来源：** Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*, arXiv v1，<https://arxiv.org/abs/2403.03206v1>。
- **阅读位置：** §4 `Text-to-Image Architecture`，尤其文本和图像序列连接及 MM-DiT 权重设计段落；§5 为论文实验。
- **来源支持：** 该论文在 latent 表示上使用 rectified-flow Transformer；文本和图像 token 形成联合序列进行注意力计算，同时为不同模态使用独立参数，使表示保留各自空间并读取另一模态信息。
- **边界：** 这是 Stable Diffusion 3 论文路线与实验设置的描述，不能推广到所有 Stable Diffusion pipeline、产品名称或未披露系统。

### DALL·E 2 prior-decoder

- **来源：** Ramesh et al., *Hierarchical Text-Conditional Image Generation with CLIP Latents*, arXiv v1，<https://arxiv.org/abs/2204.06125v1>。
- **阅读位置：** §2 `Method`、§2.1 `Decoder` 及 decoder 文字条件消融对应的小节。
- **来源支持：** 论文把 prior 写为 `P(z_i|y)`，从 caption `y` 预测 CLIP image embedding `z_i`；decoder 写为 `P(x|z_i,y)`，以 `z_i` 和可选 caption 条件生成图像。论文保留 GLIDE 的文本条件路径，但报告该路径对其所研究的某些能力帮助有限。
- **边界：** 公式描述论文中的 unCLIP 系统；不能据此推断后续 DALL·E 产品架构。prior 与 decoder 的具体选择和论文实验范围绑定。

### DALL·E 3 公开产品集成边界

- **来源：** OpenAI, *DALL·E 3 System Card*, 2023-10-03，<https://openai.com/index/dall-e-3-system-card/>；官方 PDF <https://cdn.openai.com/papers/DALL_E_3_System_Card.pdf>。
- **阅读位置：** 系统卡首页日期、摘要与 §1 Introduction。
- **来源支持：** 系统卡描述 DALL·E 3 作为 ChatGPT/GPT-4 产品中的图像生成组件；GPT-4 与用户用自然语言交互，并合成发送给 DALL·E 3 的 prompt，较含糊的用户请求可以被扩展为更详细描述。系统卡也说明发布准备、安全评估与缓解工作。
- **边界：** 该材料支持公开产品编排和安全流程的描述，不是 text encoder、latent、denoiser 或完整生产架构披露。

## 编辑、控制与个性化

### Image-to-image 与 inpainting 接口

- **来源：** Hugging Face Diffusers v0.35.1 img2img 文档，<https://huggingface.co/docs/diffusers/v0.35.1/api/pipelines/stable_diffusion/img2img>；inpainting 指南，<https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/inpaint>。
- **阅读位置：** img2img API `strength` 参数说明；inpainting mask 指南中的白/黑区域约定。
- **来源支持：** 该 img2img 接口将 strength 限定在 `[0,1]`，strength 越大加入的噪声越多；strength 为 1 时运行完整指定步数并基本忽略源图。指定 inpainting 文档使用白色 mask 区域作为重绘区域、黑色区域作为保留区域。
- **边界：** strength 到具体 scheduler timestep 的映射由 pipeline 与 scheduler 决定；教学中的列表索引映射不能当作通用实现契约。其他 mask 接口可能采用相反的 0/1 约定。

### ControlNet 零卷积初始化

- **来源：** Zhang et al., *Adding Conditional Control to Text-to-Image Diffusion Models*, arXiv v1，<https://arxiv.org/abs/2302.05543v1>。
- **阅读位置：** §3.1 `ControlNet`，公式 (1)–(4)。
- **来源支持：** 原论文冻结原始网络参数，复制可训练分支，并在分支间使用权重和偏置均零初始化的 `1×1` zero convolution。训练开始时零卷积控制输出为零，因此初始输出与无 ControlNet 主干输出一致；后续训练学习控制注入。
- **边界：** 这是论文结构和初始化推导；控制效果还依赖条件预处理、空间对齐、训练配置和具体 checkpoint。玩具残差数例不是论文或模型输出。

### DreamBooth 与 LoRA

- **来源：** Ruiz et al., *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation*, arXiv v1，<https://arxiv.org/abs/2208.12242v1>；Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv v2，<https://arxiv.org/abs/2106.09685v2>；Diffusers v0.35.1 training overview 与 LoRA loader，<https://huggingface.co/docs/diffusers/v0.35.1/en/training/overview>、<https://huggingface.co/docs/diffusers/v0.35.1/api/loaders/lora>。
- **阅读位置：** DreamBooth 摘要及 §§4.1–4.2；LoRA §4.1；Diffusers 固定版本 DreamBooth/LoRA 脚本与加载接口说明。
- **来源支持：** DreamBooth 是 subject-driven 个性化微调方法，使用少量主体图像、唯一标识词和 class-specific prior-preservation loss；LoRA 冻结原权重、训练低秩分解矩阵来表示权重更新。两者是方法目标与参数更新方式的不同层次，Diffusers 固定版本列出 DreamBooth-LoRA 训练路线。
- **边界：** LoRA 原论文的实验对象是语言模型，扩散模型中的模块位置和训练效果应以对应框架实现/目标 checkpoint 为准。低秩可训练参数量比例不等于总训练 FLOPs、激活内存或 wall-clock 时间比例。

## 文生图评估与来源凭证

### 指标和 benchmark

- **FID：** Heusel et al., arXiv v6，<https://arxiv.org/abs/1706.08500v6>。FID 比较指定 Inception 特征表示中真实与生成样本的均值和协方差；结论依赖 feature extractor、预处理、样本集与估计口径，不能单独证明 prompt 遵循或语义质量。
- **CLIPScore：** Hessel et al., arXiv v3，<https://arxiv.org/abs/2104.08718v3>。原论文定义和评估对象是 image captioning；将 prompt-image 相似度迁移为文生图指标时应视为代理信号，不能说原论文完整验证了文生图任务。
- **GenEval：** Ghosh et al., arXiv v1，<https://arxiv.org/abs/2310.11513v1>。§3 使用 object-focused 检测框架评估对象共现、位置、计数、颜色等结构属性；结论受 detector、阈值、prompt 集和评估规则影响。
- **T2I-CompBench：** Huang et al., 会议论文对应 arXiv v2，<https://arxiv.org/abs/2307.06350v2>。§§3–4 覆盖 attribute binding、object relationships 和 complex compositions，并按类别使用不同评估设计；论文指出单一 metric 不能覆盖全部类别。后续 T2I-CompBench++ 是不同的增强版本，不应与原 benchmark 的会议版混为一谈。

## 未解决的证据边界

- 论文架构、框架公开 API、产品集成与实际部署系统是不同证据类型，不能相互代替。
- 不同 checkpoint 的 VAE scaling、text encoder、CFG 基线、scheduler 与 mask 约定可能不同；需要对目标版本核对。
- FID、CLIPScore、GenEval 与 T2I-CompBench 只覆盖各自定义的图像/文本任务切片；业务成功仍需单独定义 prompt claims、评估分母、人工审查和安全门禁。
- 所有小数值和 shape 预算均为教学输入下的算术推演，不是模型质量、GPU 时间、显存或生产性能测量。
