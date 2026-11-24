# 训练成本与资源规划：公式口径和公开证据

更新时间：2026-11-24

适用主题：训练 FLOPs、MFU/HFU、规模扩展、生命周期成本、能源与碳排估算。

## 公式口径

### 恢复 checkpoint 的梯度状态

- **证据等级：A，官方框架文档/实现。** PyTorch Distributed Checkpoint 2.15.0 教程中的训练状态示例保存 model 与 optimizer state；当前 NVIDIA Megatron-LM checkpoint 配置将 optimizer、RNG 与 wgrad/dgrad 保存分别配置，梯度有独立的可选保存间隔。
- **本章采用的计算口径：** optimizer step 边界的恢复点按模型权重、optimizer 状态、可能存在的 FP32 master weights 与附加状态估算，默认不加当前梯度；若 checkpoint 位于 gradient-accumulation 中途且实现需要无损续训，则把梯度按实际精度另计。例中 7B 模型每参数 14 bytes 对应 BF16 权重 + 两个 FP32 Adam moments + FP32 master copy；中途梯度若为 BF16，再加 2 bytes/parameter。
- **边界：** 框架、优化器、精度、checkpoint 时点和是否允许在 step 边界恢复都会改变实际状态；Megatron-LM 的 `main` 分支是移动来源，本次仅记录 2026-11-24 观察到的配置，不代表所有版本默认设置。
- **来源：** [PyTorch Distributed Checkpoint 教程 2.15.0](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)；[NVIDIA Megatron-LM checkpoint 配置（main，2026-11-24 访问）](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/config/training_config.py)。

### Dense Transformer 的 `6ND` 近似

- **证据等级：A，原始论文。** Chinchilla 将训练 FLOPs 近似为 $6ND$，其中 $N$ 是参数量、$D$ 是训练 token 数；其附录按前向/反向操作计数，并与更细的架构 FLOPs 计算作比较，差异在所列模型中约为几个百分点到 10%。
- **边界：** 这不是精确硬件时间公式。长序列 attention、重计算、padding 执行、MoE 路由、通信和输入管线需要分别计数或实测。
- **口径推论：** MFU 的模型 FLOPs 与吞吐中的 token 单位必须一致。若吞吐只统计 assistant label token，不能直接乘按实际输入序列 token 计算的 $6N$；应统一使用前向处理 token，或由完整 step FLOPs 除以同一计数口径。Padding 可不算作有效数据 token，但若 kernel 实际计算了 padding，它仍属于执行 FLOPs。
- **来源：** [Chinchilla 原论文，§3 与附录 F](https://arxiv.org/html/2203.15556)；[PaLM 原论文，附录 B](https://arxiv.org/html/2204.02311)。

### MFU、HFU 与长程训练吞吐

- **证据等级：A，原始论文。** PaLM 将 MFU 定义为观测 token 吞吐相对理论峰值吞吐的比例，分子只计目标前向/反向工作，不计重计算；PaLM 540B 在 6,144 个 TPU v4 芯片上报告 MFU 46.2%、HFU 57.8%。
- **证据等级：A，原始论文。** MegaScale 在 175B、序列长度 2,048、batch size 6,144 的强扩展对照中，GPU 数从 3,072 增至 12,288 时，MegaScale 的 MFU 从 59.1% 降至 55.2%，Megatron-LM 从 48.7% 降至 41.2%；12,288 卡一行的 1.34× 是 MFU 比值，而表中吞吐 1,984.0k/1,466.8k 约为 1.35×。其专有生产训练记录还报告超过 100 次重启、90% 以上异常自动检测/恢复、从最新 checkpoint 追赶进度低于 15 分钟及有效训练时间率高于 90%。
- **边界：** MFU/HFU 数字依赖模型计算口径；MegaScale 的性能和可靠性来自其集群、实现和生产任务，不能当作其它集群的扩展效率或故障率。
- **来源：** [PaLM 原论文，§4 与附录 B](https://arxiv.org/html/2204.02311)；[MegaScale 原论文，§6.1–6.2](https://arxiv.org/html/2402.15627)。

### 计算最优与推理成本

- **证据等级：A，原始论文。** Chinchilla 论文基于 400 多个训练运行，提出固定计算预算下模型参数量与训练 token 应大致同比例扩大；其 70B/1.4T-token 模型以与 Gopher 大致相同的计算预算验证这一点。论文还转述 Kaplan 对 10× 计算预算的分配建议：参数约扩大 5.5×、token 约扩大 1.8×。
- **证据等级：A，原始论文。** Inference-aware scaling 把预期推理请求纳入模型/数据规模选择，报告约 10 亿请求情景下更小、训练更久的模型可能更便宜，并以 47 个模型及最高 10,000 token/parameter 的设置验证拟合方法。
- **边界：** 两项结果分别优化论文定义的预训练计算/损失或给定质量与推理负载下的成本；数据质量、请求量、输入输出长度、硬件、质量门槛和价格变化会改变结果。
- **来源：** [Chinchilla 原论文](https://arxiv.org/html/2203.15556)；[Inference-aware scaling 原论文 v3](https://arxiv.org/html/2401.00448v3)。

### 公开能耗与碳排数字的边界

- **证据等级：A，模型技术报告。** Llama 2 报告累计约 330 万 GPU-hours，按 GPU TDP 和碳效率估算训练排放 539 tCO₂e；作者明确说明实际 GPU 功率随利用率而变化，并指出其计算没有纳入互联、非 GPU 服务器、数据中心冷却以及硬件生产相关排放。
- **证据等级：A，原始论文。** OLMo 报告按 GPU 能耗、PUE 和训练地点碳强度估算排放，并把调试、超参数调优和停机排除在估算之外；相同的能耗会因供电地点/时间的碳强度假设而产生不同排放结果。
- **边界：** 这些是作者基于设备和数据中心参数的模型估算，不是可跨集群比较的统一电表记录，也不是完整生命周期碳盘查。
- **来源：** [Llama 2 原论文，§2.2.1](https://arxiv.org/html/2307.09288)；[OLMo 原论文，附录 B](https://arxiv.org/html/2402.00838)。

### MLPerf Training 的测量目标

- **证据等级：A，基准组织官方页面。** MLPerf Training 说明其套件测量系统将模型训练到指定质量指标所需的速度；benchmarks 由数据集和质量目标定义，已发布结果可能更改或失效，规则是正式口径。
- **边界：** 公开 benchmark 适合核对测量规则和任务/质量门槛；不同版本、模型、精度和系统的分数不可直接当作当前目标作业的吞吐。
- **来源：** [MLCommons MLPerf Training](https://mlcommons.org/benchmarks/training/)。
