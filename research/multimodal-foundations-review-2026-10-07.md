# 第一册第四份正文：多模态基础资料核验

核验日期：2026-10-07。范围：`book-01-core-30/chapters/04-多模态基础.md` 第 31—36 讲。本记录更新 2026-08-14 的正文来源清单，说明可支持范围、链接访问情况和本地验证边界；不是目标模型 benchmark 或生产验收。

## 来源与证据等级

| 主题 | 一手来源 | 本轮支持的范围 | 不能据此推出 |
| --- | --- | --- | --- |
| 视觉 patch 与对齐 | [ViT](https://arxiv.org/abs/2010.11929)、[CLIP](https://arxiv.org/abs/2103.00020)、[SigLIP](https://arxiv.org/abs/2303.15343) | patch 表示、双编码器、图文对比学习与 sigmoid pair loss | 不能从检索/分类结果推出 OCR、计数或通用视觉问答能力 |
| 视觉语言融合 | [Flamingo](https://arxiv.org/abs/2204.14198)、[BLIP-2](https://arxiv.org/abs/2301.12597)、[LLaVA](https://arxiv.org/abs/2304.08485) | 交叉注意力、Q-Former、projector/视觉 token 接入和视觉指令微调的论文实例 | 论文结构不是所有商业 VLM 的内部实现；合成指令数据不自动正确 |
| 扩散建模 | [DDPM](https://arxiv.org/abs/2006.11239)、[Latent Diffusion](https://arxiv.org/abs/2112.10752)、[DiT](https://arxiv.org/abs/2212.09748)、[SDXL](https://arxiv.org/abs/2307.01952) | 前向加噪、噪声预测、latent 压缩、cross-attention 条件和 DiT/SDXL 公开组件 | 论文实验不代表任意闭源生成器或目标硬件的质量/延迟 |
| 扩散采样与评估 | [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598)、[DDIM](https://arxiv.org/abs/2010.02502)、[DPM-Solver](https://arxiv.org/abs/2206.00927)、[噪声 schedule 分析](https://arxiv.org/abs/2305.08891)、[FID](https://arxiv.org/abs/1706.08500)、[FVD](https://arxiv.org/abs/1812.01717) | CFG 代数、少步采样方法、schedule 边界，以及 FID/FVD 的指标研究入口 | 少步数不保证质量不变；FID/FVD 不能单独判定 prompt 遵循、任务正确性或安全性 |
| 语音与音频表示 | [Whisper](https://arxiv.org/abs/2212.04356)、[SoundStream](https://arxiv.org/abs/2107.03312)、[EnCodec](https://arxiv.org/abs/2210.13438) | Whisper 论文公开的弱监督 ASR 设定，以及神经音频 codec 的压缩表示路线 | 不保证所有语言、口音、噪声、实时服务或 TTS 质量 |
| 视频建模 | [Video Diffusion Models](https://arxiv.org/abs/2204.03458) | 视频扩散、时空建模和视频延展的研究路线 | 不证明生成模型具备可交互、可验证的完整 world model |
| 安全与来源信息 | [OWASP LLM01:2025](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)、[C2PA 2.2 规范](https://spec.c2pa.org/specifications/specifications/2.2/index.html) | prompt injection 风险分类、模型外权限控制建议和内容来源凭证规范 | 风险清单不是应用安全验收；来源凭证/水印不证明内容真实 |
| 2026 多模态实例 | [Qwen3.5-397B-A17B 研究记录](model-update-2026-09/qwen3.5-397b-a17b-source-notes.md)、[第二十一册第 83 章](../book-21-transformer-architecture-evolution/chapters/83-qwen3.8-qsa-gated-residual-n-gram-muon.md)、[第 93 章](../book-21-transformer-architecture-evolution/chapters/93-qwen3.5-omni-aria-aut-timestamp与流式语音.md) | 2026 年公开模型卡/论文所述的 early-fusion、原生视觉和流式语音例子；使用书系已有的 revision 绑定记录 | 发布方声明不等于独立复现，也不能把 Qwen 专属字段迁移到其他模型 |

GPT-4o 与 Sora 段落只把 OpenAI 2024 年发布页作为历史案例，说明产品公开声明与未披露架构的证据边界；正文没有把它们当作当前模型目录。对应的[官方 GPT-4o 页面](https://openai.com/index/hello-gpt-4o/)和[Sora 研究说明](https://openai.com/research/video-generation-models-as-world-simulators)在 2026-10-07 经当前代理访问均返回 Cloudflare HTTP 403。两页此前的核验日期记为 2026-08-14；本轮不据此增加后续产品状态或新版本架构结论。[Sora 二手综述](https://arxiv.org/abs/2402.17177)当前返回 HTTP 200，只用于公开讨论脉络，不替代官方证据。

## 链接访问记录

章节共有 25 个唯一外部 URL：21 个 arXiv 页面、OWASP、C2PA 和两个 OpenAI 页面。复核时前 23 个均返回 HTTP 200；OpenAI 两页因 Cloudflare challenge 返回 403。DPM-Solver 页面首次 HEAD 请求超时，重试后返回 200。arXiv 的 SigLIP abstract 页面返回 200，但 `arxiv.org/html/2303.15343` 返回 404；正文引用 abstract 入口，不依赖 HTML 版本。C2PA 规范使用版本化的 2.2 页面。

高时效的 2026 模型实例不在本章重写细节，直接链接至 2026-09 的模型研究记录和第二十一册专题。两处固定资料支持的是发布方披露的配置/论文范围，不能证明模型的完整服务端实现或生产表现。

## 本地代码与公式验证

- 环境：Python 3.13.13、PyTorch 2.12.0+cpu；本章示例不依赖新增项目包。代码运行使用 `/tmp/llm-book-deps` 中的临时环境。
- 7 个 Python 代码块逐块独立运行全部通过；输出与正文中的 patch 计数、CLIP loss、VLM shape、CFG、音频 hop 数、CER 和视频 token 数示例一致。
- 36 项边界/反事实检查通过，覆盖非正或非整数 patch 尺寸、非整倍帧数、非法音频时长/采样率/frame shift、零分母 CER、非法扩散标量、VLM shape 不匹配和视频 token 参数。
- CLIP embedding、VLM projector 和扩散代码是固定随机种子的局部机制 demo；patch/audio/video 代码是公式估算。均不构成论文复现、模型质量评估或 serving benchmark。
- 公式中的边界约定已检查：图像 patch/tubelet 的不足部分按 padding 估算；`β_t∈(0,1)`；WER/CER 参考序列长度非零；prompt ensemble 在向量平均后重新归一化；SigLIP 标签取 `±1` 且 bias 加在缩放分数上。

## 未验证项

- 未运行真实视觉/音频 encoder、VLM、扩散生成器、语音识别或视频生成模型。
- 未使用 CUDA 测试 patch/token 预算与显存、扩散采样质量、实时音频延迟或视频吞吐。
- 未重现 CLIP/SigLIP、FID/FVD、Whisper、SoundStream/EnCodec 或扩散论文实验；本地代码仅验证所声明的教学关系。
- OpenAI 两个历史产品页在当前代理中的 403 限制保留为警告；若后续章节要描述 Sora/GPT-4o 当前服务状态，必须另行核验当前官方资料。
