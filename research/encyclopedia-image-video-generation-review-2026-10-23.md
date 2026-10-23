# 第四册第十七份正文《图像生成与视频生成》整章验收研究记录（2026-10-23）

对象：`book-04-llm-encyclopedia/chapters/17-图像生成与视频生成.md`（17.0—17.23，724 行）。

## 结构与格式

- 全章围栏成对闭合（22 math、1 python、text/表格若干）；math 围栏花括号平衡。
- 修正 6 处 `\operatorname{...}` 为 `\mathrm{...}`（SNR、FID、FVD、cos 等）；修正后无 `\operatorname`、无 `$$`。

## 来源核验

- 12 个唯一外链：5 篇新 arXiv 题名经 abs 页面核对——DDIM 2010.02502、DPM-Solver 2206.00927、DiT 2212.09748、1706.08500 实为提出 FID 的 TTUR 论文、2402.17177 实为第三方 Sora 综述（"Sora: A Review on Background, Technology, Limitations..."）。
- 发现并修正 2 处链接文字与实际论文不符：
  1. `2402.17177` 原标为 "Video generation models as world simulators"，但 OpenAI 技术报告发布于其官网、未上 arXiv，该 ID 实为第三方综述；已改为综述题名并注明 OpenAI 报告出处。
  2. `1706.08500` 原标为 "Fréchet Inception Distance"，实际题名为 "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"（TTUR，提出 FID 指标）；已改为实际题名并注明其提出 FID。
- DDPM/LDM/ControlNet/DALL-E/VDM/FVD 视频/NIST 沿用第 16 章（前轮）证据。

## 代码与数值核验

- 1 个 Python 块（生成账本：latent cells 4096、少步低采纳率成本反而上升、零采纳率拒绝）独立执行通过。
- 数值复算：512/8→64²=4096 与 16×32×32=16,384 一致；C_success 分母 P>0 的不可定义语义与代码的 ValueError 拒绝一致。
- 公式边界：SNR 仅在 0<ᾱ<1 有定义且 t=0 为无穷；D_preserve 要求 mask 外面积大于零；C_id/F_flicker 要求 T≥2 且标记不可比片段；FID/FVD 要求样本量、协议披露；整除性假设在 DiT patch 与 VAE 压缩处均有正文声明。

## 配套同步

- `PAPERS.md` 多模态/生成区已有 DDPM、LDM、ControlNet、DALL-E、VDM 入口；本章新出现的 DDIM、DPM-Solver、DiT、FID(TTUR)、Sora 综述作为 17.21 延伸阅读，`PAPERS.md` 扩散与评估条目已部分覆盖（DDPM、LDM、ControlNet、FID 在 glossary/前轮条目），DiT 与 DDIM/DPM-Solver 直链未在 PAPERS.md 单列，属于延伸阅读层不必重复登记。
- 第四册目录第 17 章映射正确。

## 未验证项

- 未运行真实扩散训练、采样、VAE 重建或视频生成；D_preserve、C_id、F_flicker 为像素/特征级教学代理（正文已声明局限）。
- OpenAI "Video generation models as world simulators" 报告为官网资料，未经代理重新核验当前可达性（openai.com 历史代理 403）。
- PDF 构建受 Pandoc 缺失限制，未执行。
