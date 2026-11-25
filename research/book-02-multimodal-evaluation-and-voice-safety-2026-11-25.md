# 第二册多模态评估与语音克隆证据（2026-11-25）

本记录只保存可复核的外部知识、来源和适用边界，不记录章节修改历史。

## 视觉 Transformer 的序列与注意力形状

- **来源：** Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)，2021 年修订版。
- **证据等级：** 原始论文。
- **可支持的判断：** ViT 将图像切成 patch 序列后送入 Transformer。若一层有 $h$ 个 attention heads、每个 head 对 $n$ 个位置计算完整 attention score，那么每个 head 的 score 矩阵有 $n^2$ 个元素，合计 $hn^2$；batch 和层数另行相乘。
- **边界：** $hn^2$ 是 score-cell 数量的账目，不等于完整 FLOPs、内存流量或运行时成本；实际开销还取决于实现、精度、融合 kernel 和硬件。

## 录音克隆的同意流程

- **来源：** OpenAI API 官方文档，[Custom voices](https://developers.openai.com/api/docs/guides/custom-voices)，检索日期 2026-11-25。
- **证据等级：** 官方产品接口文档。
- **可支持的判断：** 当前文档描述的录音式自定义声音流程，分别要求提供说话人同意录音和用于复现的样本录音。授权记录可以作为语音克隆系统中的独立输入和校验对象，而不是只依赖音色相似度分数。
- **边界：** 这是单一服务的接口流程，不构成全行业通用协议，也不替代具体产品的政策审查、授权范围核验或法律意见。语音克隆是否允许、公众人物是否需额外限制，需按适用规则和产品政策处理。

## Qwen3-Omni 的公开架构与首包延迟

- **来源：** Qwen 团队, *Qwen3-Omni Technical Report*, [arXiv:2509.17765](https://arxiv.org/abs/2509.17765)，2025-09-22。
- **证据等级：** 模型发布团队技术报告。
- **可支持的判断：** 报告描述 Thinker-Talker MoE 架构；冷启动设定下给出 234 ms 的 theoretical end-to-end first-packet latency。报告同时列出文本交互、语音理解和语音生成支持的语言数。
- **边界：** 234 ms 是论文设定中的理论首包指标，不是所有硬件、并发、网络和播放缓冲下的线上 p95。模型团队报告的结果仍需按目标部署条件复测。

## LVOmniBench 的长音视频评测范围

- **来源：** Tao et al., *LVOmniBench: Pioneering Long Audio-Video Understanding Evaluation for Omnimodal LLMs*, [arXiv:2603.19217](https://arxiv.org/abs/2603.19217)，v1 于 2026-03-19 提交；[论文 HTML](https://arxiv.org/html/2603.19217v1)。
- **证据等级：** 研究者提出的预印本基准与论文实验。
- **可支持的判断：** 数据包含 275 段 10—90 分钟视频和 1,014 组问答。论文报告 Qwen3-Omni-30B 准确率为 35.8%、Gemini-3.0-Pro 为 65.8%；论文设置中本地实验使用 H100 80GB 或 L40S 48GB，Gemini 3.0 系列按 1 FPS 输入并启用 deep thinking。结果说明该基准的长时音视频任务仍具挑战。
- **边界：** 这些数字绑定 v1 数据、模型名称/版本、最大输入帧配置和评测协议；它们不是模型当前总体能力或线上产品可靠性的排名。论文结果属于作者报告，需留意后续版本是否修订。

## RVG-Bench 的推理式音视频时间定位

- **来源：** Feng et al., *RVG-Bench: A Benchmark for Reasoning-Driven Audio-Visual Temporal Grounding*, [arXiv:2610.32268](https://arxiv.org/abs/2610.32268)，v1 于 2026-10-23 提交；[论文 HTML](https://arxiv.org/html/2610.32268v1)。
- **证据等级：** 研究者提出的预印本基准与论文实验。
- **可支持的判断：** v1 包含 1,255 段视频、1,890 组 query-interval 配对和八类推理问题；论文标注中 11.38% 查询仅需音频、43.28% 需音视频共同推理，故超过一半查询需要音频证据。论文报告 Qwen3.8-Omni-Flash 的 mIoU 为 49.6，最强本地评测基线为 30.1。作者说明基准目前以短到中等长度视频为主，没有覆盖整部电影或小时级未剪辑视频。
- **边界：** mIoU 是特定时间定位任务、模型/API 输入和论文评分协议下的结果，不是总体视频理解能力分数；该预印本的模型版本名和 API 行为会随时间变化，引用时应保留版本与检索日期。
