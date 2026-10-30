# Flow Matching 与视频生成：来源证据

检索与阅读日期：2026-10-29。本记录只保存联网核验的技术事实、版本与证据边界；书籍审阅和纳入决定记录在 `progress/`。

## Flow Matching 与 Normalizing Flow 的关系

- Lipman et al., “Flow Matching for Generative Modeling,” [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)，ICLR 2023。阅读 arXiv HTML 摘要、§2 的 continuous normalizing flow 定义、§3 的 Flow Matching/Conditional Flow Matching 目标、§4 的条件概率路径及结论。
  - 来源支持：论文将 Flow Matching 定义为训练连续正规化流的一种 simulation-free 向量场回归目标；通过构造每样本条件路径和向量场，可以获得可训练的条件目标。论文讨论的 Gaussian 路径族包含扩散路径，也允许非扩散路径；其中的 optimal-transport displacement interpolation 对应直线路径。
  - 证据等级：source-backed（原始方法论文及其指定图像实验）。
  - 边界：Flow Matching 是依赖所选概率路径的训练框架，不是某一固定采样器，也不等于传统离散可逆 Normalizing Flow 的换名。论文中的训练稳定性、似然和样本质量比较只绑定其数据、模型和实验协议；不能据此断言任意 Flow Matching 一定更快或更好。

- Liu et al., “Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow,” [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)。阅读论文摘要和版本信息。
  - 来源支持：作者提出用 ODE 学习从一个经验分布到另一个经验分布的传输，并让轨迹尽可能接近连接样本对的直线路径；摘要报告递归 rectification 可产生更直的轨迹，并在其图像任务中以较粗的离散步长进行推理。
  - 证据等级：source-backed（原始预印本摘要级方法描述和作者报告）。
  - 边界：Rectified Flow 是一类具体路径/训练策略，不与所有 Flow Matching 同义；一步或少步的质量结论受该论文模型、训练和任务设置约束，本项目不把这些数字用作通用性能保证。

## Pyramid Flow：Flow Matching 与时空金字塔

- Jin et al., “Pyramidal Flow Matching for Efficient Video Generative Modeling,” ICLR 2025，[会议论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/3ab228c4703c4459b1a600ebadc5732c-Paper-Conference.pdf)，[arXiv 页面](https://arxiv.org/abs/2410.05954)。阅读官方论文摘要、§3.1—3.3、结论和限制部分。
  - 来源支持：论文提出把视频生成轨迹组织为空间分辨率金字塔阶段，以统一 Flow Matching 目标在单个 DiT 中联合训练，且只有最后阶段运行于全分辨率；自回归视频扩展使用时间金字塔压缩全分辨率历史条件。作者报告的公开视频时长、分辨率、帧率和训练成本是其特定设置。
  - 证据等级：source-backed（ICLR 2025 正式论文的方法、作者实验和自述限制）。
  - 边界：空间/时间金字塔及效率结果绑定该架构、公开数据和论文评测；论文 limitation 提到只支持自回归生成，且 temporal pyramid 在长时间范围可能出现细微主体不一致。它是 Flow Matching、DiT 与自回归视频组合的实例，不能代表所有视频生成系统或跨模型胜负。
