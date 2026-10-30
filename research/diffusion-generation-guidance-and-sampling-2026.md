# 扩散生成、指导与采样的来源证据

- 检索与阅读日期：2026-10-29 至 2026-10-30。

## DDPM 前向与反向过程

- 来源：Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239v2)，arXiv v2。阅读 arXiv HTML 的前向扩散、逆向过程和噪声参数化推导（§§2.1–2.2；HTML 中对应公式约第 62–99 行）。
- 来源支持：前向过程是按时间步定义的 Gaussian Markov chain；累计信号保留率 `ᾱ_t=∏_{s≤t}α_s` 给出从 `x_0` 直接构造 `x_t` 的闭式分布。论文讨论以网络预测噪声来参数化反向 Gaussian 转移均值，并单独处理反向方差。
- 边界：噪声预测和简化 MSE 是论文采用的重要参数化/训练目标，不是所有扩散模型都必须使用的定义；反向均值也不是包含方差与随机项的完整采样更新。
- 证据等级：方法公式 source-backed；不代表其他预测参数化、方差设置或采样器都采用同一实现。

## Classifier-Free Guidance 与参数记法

- 来源：Ho and Salimans, [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598v1)，arXiv v1。阅读 HTML §3.2 与 Algorithm 1–2（约第 98–140 行）。
- 来源支持：同一个网络可通过随机丢弃条件训练出条件与空条件预测；论文写法为 `(1+w)ε_cond−wε_uncond`，因此 `w=0` 返回条件预测。§3.2 以 Bayes 关系解释 conditional 与 unconditional score 的差值对应隐式分类概率梯度，并指出神经 score 估计一般不保证构成精确的分类器梯度。论文摘要将指导强度描述为生成质量和多样性之间的取舍。
- 实现参数边界：Hugging Face Diffusers 的 [v0.35.1 Stable Diffusion pipeline](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py) 源码说明 `guidance_scale=1` 对应不启用 CFG；有 guidance 时把 negative prompt 与正向 prompt 的表示拼成同一 batch，源码注释说明这样可避免两次前向调用。因而在此 pipeline 中，negative prompt 表示占据 CFG 公式中的无条件分支位置；严格说它不是 null-condition prediction。该实现减少了调用次数，但 batch 仍需计算两组预测。此版本的 `guidance_rescale` 源码说明其用途包括改善图像质量和缓解过曝，这只是该 pipeline 的参数说明，不能外推为其他模型/版本的保证。结合论文 `w=0` 的基线可知，不能只凭 API 名称把数值当作原论文的 `w`；应查看固定版本的组合公式、输入分支和无 guidance 基线。该示例只支持该固定 pipeline 的接口描述，不能外推到所有模型库或版本。
- 证据等级：论文算法 source-backed；接口映射是特定软件版本的实现证据，不是统一 API 标准。

## U-Net、DiT 与 latent diffusion

- 来源：Peebles and Xie, [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748v2)，arXiv v2；阅读摘要。Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752v2)，arXiv v2；阅读摘要和 latent-space 方法部分（§3）。
- 来源支持：DiT 将 Transformer 用作扩散模型主干，并在论文所测设置中报告计算量与生成质量间的关系；latent diffusion 将去噪移入感知压缩表示，以降低高分辨率图像生成的计算需求。
- 边界：DiT 的质量/计算关系绑定论文模型和训练评估条件；latent 元素数下降不等于端到端 FLOPs 或延迟同比下降，VAE 重建损失和系统其余模块仍需计入。正文中的具体 shape 比较是标注假设的教学算例，不是上述论文普遍给出的模型配置。
- 证据等级：方法与作者报告 source-backed；未独立复现实验。

## DiT scaling-law 摘要报告

- 来源：[Scaling Laws for Diffusion Transformers](https://proceedings.iclr.cc/paper_files/paper/2026/hash/880d8999c07a8efc9bbbeb0c38f50765-Abstract-Conference.html)，ICLR 2026 官方论文摘要页。此次仅阅读摘要。
- 来源支持：摘要报告直接实验覆盖约 `10^17` 至 `6×10^18` FLOPs，并给出 1B 参数模型在 `1.5×10^21` FLOPs 的损失预测。
- 边界：该预测 FLOPs 是摘要所报直接实验上限的 250 倍，必须作为外推而不是实测。此次未阅读全文方法，也未复现拟合或预测；不据此推出目标任务、分辨率或部署条件下的 scaling 收益。
- 证据等级：source-backed 的摘要级报告；拟合质量与外推可靠性未独立审查。

## 少步采样与终端信噪比

- 来源：Song et al., [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502v4)，arXiv v4；阅读摘要。Lu et al., [DPM-Solver](https://arxiv.org/abs/2206.00927v3)，arXiv v3；阅读摘要。
- 来源支持：DDIM 以非马尔可夫扩散路径支持更少步的采样，并保留 DDPM 的训练程序；DPM-Solver 将采样写成扩散 ODE 的数值求解，论文报告不需为求解器重新训练模型。
- 边界：论文中的步数、质量和加速结果依赖其模型、数据、评估和硬件设置；少量函数评估不自动等于更低端到端延迟或相同生成质量。Progressive Distillation 属于通过训练学生模型减少步数的不同路线，不能和无需重新训练的 sampler 混为一类。
- 证据等级：方法类别 source-backed；具体性能未独立复现。

- 来源：Lin et al., [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891v4)，arXiv v4。阅读摘要、§2 和 §3.1。
- 来源支持：论文在所分析的常见 schedule 中报告非零 terminal SNR，并指出训练末端保留信号而推理从纯 Gaussian noise 开始会造成边界不匹配；作者提出零终端 SNR、`v` prediction、调整采样起点及 guidance rescale 等方案。
- 边界：分析和改进绑定作者测试的 schedule、checkpoint 与模型设置，包括其 Stable Diffusion 分析；不表示所有扩散管线都有同一缺陷，也不表示同时采用所有修正就必然改善其他 checkpoint。
- 证据等级：论文设置内的分析和作者实验 source-backed；未在其他 checkpoint 上独立验证。

## 未独立验证项

- 未运行 diffusion checkpoint，未复现上述论文训练、采样加速或图像质量 benchmark。
- ICLR 2026 scaling-law 仅查阅官方摘要；全文实验细节、拟合稳定性与预测误差尚未独立审查。
- Diffusers 证据固定在 v0.35.1 的经典 Stable Diffusion pipeline；其他 pipeline、版本或框架应重新检查对应源码与参数定义。

## 图像生成评估指标

- 来源：Heusel et al., [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf)，NeurIPS 2017。核对摘要和 FID 定义段。
- 来源支持：FID 将真实图像和生成图像经 Inception coding layer 得到的特征分布，以高斯近似的均值与协方差通过 Fréchet 距离比较；论文在其 GAN、数据集和扰动设置中评估该指标。
- 边界：论文在特定 GAN 实验中提出并检验 FID，不证明它能定位 prompt 属性、局部关系、安全或文字错误，也不支持脱离数据、特征提取器与样本设置比较不同评测结果。
- 证据等级：定义及原论文实验 source-backed；本地未复现。

- 来源：Salimans et al., [Improved Techniques for Training GANs](https://proceedings.neurips.cc/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf)，NeurIPS 2016。核对 Inception Score 定义段。
- 来源支持：Inception Score 通过预训练分类器的条件类别分布 \(p(y\mid x)\) 与生成样本上的边缘分布 \(p(y)\) 计算指数化 KL 期望；设计直觉是单样本类别置信度和生成集类别多样性。原论文指出多样性项需要足够大的样本数，并在其研究中使用 50k samples。
- 边界：分类器、类别空间和样本分布限制指标含义；它没有直接把生成集与目标业务的真实图像分布比较。样本量建议绑定原文设置，不作为所有生成评估的固定要求。
- 证据等级：方法定义和原论文设置 source-backed；对业务适配性未独立验证。

- 来源：Hessel et al., [CLIPScore: A Reference-free Evaluation Metric for Image Captioning](https://aclanthology.org/2021.emnlp-main.595/)，EMNLP 2021。核对 ACL Anthology 官方摘要。
- 来源支持：作者将 CLIPScore 作为无参考图像描述评估，报告了多个 captioning corpus 上与人工判断的相关结果，并称其关注图文兼容性；摘要指出其对需要丰富上下文的新闻 caption 相比 reference-based 指标较弱。
- 边界：本次只检查官方摘要，未复核全文公式、所有数据集或数值；captioning 结果不能直接等同于 text-to-image prompt adherence、图像质量或安全保证。
- 证据等级：摘要级 source-backed；不外推至文生图的一般质量结论。

## CFG 与少步采样的训练边界

- 来源：Salimans and Ho, [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)，arXiv。核对官方摘要。
- 来源支持：方法把训练过的确定性 diffusion sampler 逐轮蒸馏到步数减半的学生模型；摘要报告了 CIFAR-10、ImageNet、LSUN 上的作者实验结果。
- 边界：本次只核对摘要，不把作者报告的 4-step 结果外推到其他 checkpoint、分辨率或文本任务。
- 证据等级：方法概述及摘要级作者报告 source-backed；未复现。

- 来源：Meng et al., [On Distillation of Guided Diffusion Models](https://arxiv.org/abs/2210.03142)，arXiv。核对官方摘要。
- 来源支持：该工作先训练单一模型拟合 classifier-free guidance 的条件/无条件合成预测，再逐步蒸馏为较少步数的模型；摘要称其在所测数据与配置上获得较少采样步数。
- 边界：仅核对摘要；模型、数据集、采样设置和质量结论不外推到任意 latent diffusion pipeline。
- 证据等级：摘要级 source-backed；未复现。

- 来源：Tran et al., [Data-free Multistep Diffusion Distillation](https://arxiv.org/abs/2610.19305)，arXiv，2026-10-13。仅核对官方摘要。
- 来源支持：摘要提出按学生分布分配适配网络的 progressive multistep data-free distillation，并报告特定图像设置中的实验数字。
- 边界：仅有摘要级核对；方法细节、数据来源、指标口径和结果尚未审查，本轮不将其作为已建立的通用路线或性能证据。
- 证据等级：预印本摘要级 candidate；不用于正文定论。
