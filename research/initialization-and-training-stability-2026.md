# 初始化与训练稳定性：来源证据

记录日期：2026-10-29

本记录保存训练稳定性研究中的来源事实、版本、阅读范围和适用边界；书内纳入决定、修改结果和逐章状态写入进度记录。

## AdaGC：逐张量自适应梯度裁剪

- Wang et al., “AdaGC: Enhancing LLM Pretraining Stability via Adaptive Gradient Clipping,” ICML 2026, PMLR 306:127545–127572，[PMLR 论文页](https://proceedings.mlr.press/v306/wang26dg.html)，arXiv:2502.11034 v2（2026-02-22），[HTML 正文](https://arxiv.org/html/2502.11034v2)。
  - 实际阅读范围：PMLR 官方摘要；arXiv v2 HTML 的方法 §4.2、资源/通信讨论 §4.3、评测指标、训练结果中的表 7，以及相关 warmup 和 EMA 消融段落。
  - 来源支持：AdaGC 对每个参数张量 $i$ 使用阈值 $\lambda_{rel}\gamma_{t-1,i}$；$\gamma$ 是张量历史裁剪梯度范数的 EMA，超阈值时按比例缩放。作者默认训练前 $T_{start}=100$ 步采用 GlobalGC warmup，再应用 AdaGC。论文称每个张量只需一个 FP32 EMA 标量，计算 L2 范数的复杂度与 GlobalGC 相同；其 DP/TP/PP 设置下，AdaGC 的局部范数归约范围与 GlobalGC 的全局范数归约不同。
  - 作者将 spike score 定义为时间序列中偏离前 1,000 点滚动均值至少 10 个标准差的值所占百分比，主要用于训练 loss。表 7 报告 Llama-2 7B（9k steps）、Mixtral 8×1B（36k）和 ERNIE 10B-A1.4B（21k）的 AdaGC 得分均为 0；相应 GlobalGC 得分为 0.0333%、0.0144% 和 0.0100%。ICML 摘要按其口径另报告三种模型的平均下游准确率分别提高 1.32%、1.27% 和 2.48%。
  - 证据等级：source-backed（同行评审论文的算法描述、作者训练配置、指标定义和实验报告）；不是本项目的独立复现。
  - 适用边界：结果绑定论文中的模型、token/step 预算、训练并行布局、优化器、benchmark 与 spike score 定义。指标为零不等于模型没有其他失稳现象，也不能证明所有 loss spike 都由异常梯度引起；通信收益需结合目标集群和归约拓扑核验。
