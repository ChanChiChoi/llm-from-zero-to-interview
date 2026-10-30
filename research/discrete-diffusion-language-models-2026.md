# 离散扩散语言模型的架构与训练路线（2026）

检索日期：2026-10-29

## LLaDA2.0：自回归初始化与 block diffusion

- 来源：[LLaDA2.0: Scaling Up Diffusion Language Models to 100B](https://arxiv.org/html/2512.15745v2)，arXiv v2（2025-12-24）。
- 阅读位置：摘要；§4 的 Warmup–Stable–Decay（WSD）三阶段转换；§5 后训练概述；§6.1 评测设置。
- 支持事实：作者报告从预训练自回归模型转换到最多 100B 总参数的离散扩散模型。WSD 依次增大 block size、进行 full-sequence masked diffusion 稳定训练，再回到较小 block size；后训练包含 SFT 与 DPO。摘要列出 16B 的 LLaDA2.0-mini 与 100B 的 LLaDA2.0-flash 两个 MoE 变体。
- 证据等级：原始预印本方法、作者模型说明和评测协议。
- 边界：模型规模、训练方案和评测结论属于该模型系列与作者配置，尚非独立复现。并行去噪的单步数或并行宽度不等于端到端延迟优势；需同时检查 block size、迭代数、质量、硬件、批量及服务实现。

## 现代离散扩散语言模型综述

- 来源：[How To Build A Modern Diffusion Language Model](https://arxiv.org/html/2610.33391)，arXiv v1（2026-10-24）。
- 阅读位置：摘要；§3.5 的离散扩散架构讨论；§5.1 的限制与开放问题。
- 支持事实：综述将离散扩散描述为训练/采样方法，而非由目标函数唯一规定的 backbone。其例子指出，常见 masked diffusion 模型可使用 encoder-only Transformer；部分较新的系统采用 encoder–decoder，让较轻的 decoder 在迭代去噪时读取 encoder 表示。综述还总结了速度与质量、不同噪声过程和后训练之间的开放取舍。
- 证据等级：综述预印本；架构实例与经验总结依赖其引用的原始论文和公开材料。
- 边界：该综述不是新受控实验。encoder-only 或 encoder–decoder 的描述不能外推到所有扩散模型；速度、质量与架构细节需回到相应原始论文、官方资料和固定实现核验。
