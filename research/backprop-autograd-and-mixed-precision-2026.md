# Backpropagation, autograd, and mixed-precision evidence

检索与阅读日期：2026-10-28

## Rumelhart、Hinton 与 Williams（1986）

- 来源：[Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0)，Nature 323, 533–536，1986-10-09。
- 实际阅读位置：Nature 页面题名、作者和出版信息，以及摘要（页面第 13–36 行）；全文受订阅限制，本记录不把摘要当作全文证据。
- 来源支持的事实：论文摘要描述一种用于类神经元网络的 back-propagation 学习过程，通过反复调整连接权重来降低网络实际输出与目标输出的差异；摘要还指出隐藏单元可能表征任务领域的重要特征，并将这种能力与更早的 perceptron-convergence 方法作比较。
- 证据等级：原始论文摘要，可支持该文公开描述的方法与作者主张。
- 适用边界：摘要不能单独证明该方法的历史首创权、完整推导、所有网络适用性或现代大模型训练表现。因此仅用作代表性技术谱系来源，不据此作优先权断言。
- 未解决项：本次未取得论文全文；若正文需要比较其完整算法假设或历史优先权，需要另查早期原始论文和完整出版物。

## PyTorch autograd mechanics（v2.14）

- 来源：[Autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html)，PyTorch v2.14 官方文档，页面标记最后更新于 2026-08-03。
- 实际阅读位置：`How autograd encodes the history`（页面第 406–416 行），以及关于前向保存中间张量的说明（第 417–422 行）。
- 来源支持的事实：文档将 autograd 定义为反向自动微分系统；前向执行时记录运算形成有向无环图，反向遍历由链式法则计算梯度；每次迭代都会重新建立图。
- 证据等级：固定版本官方文档，支持 PyTorch v2.14 公开接口与实现概念。
- 适用边界：文档说明 PyTorch 行为，不代表所有自动微分系统的接口或内部实现；不单独证明某模型训练的计算成本、吞吐或显存。
- 未解决项：本次未运行 PyTorch 代码，文档阅读不是本地执行验证。

## PyTorch Automatic Mixed Precision（v2.14）

- 来源：[Automatic Mixed Precision package - torch.amp](https://docs.pytorch.org/docs/2.14/amp.html)，PyTorch v2.14 官方文档。
- 实际阅读位置：`Gradient Scaling`（页面第 3967–3980 行），含 FP16 梯度下溢、损失缩放、更新前反缩放以及 BF16 预训练模型与 FP16 数值范围不兼容的提示。
- 来源支持的事实：FP16 梯度可能低于可表示范围而下溢为零；对 loss 进行缩放可放大反向梯度以减少这种下溢，优化器更新前应先反缩放。文档还警告，部分 BF16 预训练模型无法工作在 FP16 最大数值 65504 的范围内，可能造成溢出；GradScaler 的比例可能降到 1 以下。
- 证据等级：固定版本官方文档，支持 PyTorch v2.14 对混合精度和梯度缩放的公开说明。
- 适用边界：这些是框架文档给出的数值风险与配置说明，不表示所有 BF16 模型、硬件或算子都具有相同行为；损失缩放不保证解决前向溢出或所有数值稳定性问题。
- 未解决项：特定训练配置的 dtype 可行性仍需基于目标模型、算子、硬件和实际数值范围评估；本次未执行本地实验。
