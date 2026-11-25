# Step 3.5 Flash：S3F1、MTP-3 与 11B 激活参数研究记录

检索与复核日期：2026-11-25。范围：第 80 章正文精修所需的层布局、参数统计、注意力消融、MTP 训练目标、speculative sampling 机制、当前公开 serving 配置和 Agent benchmark harness。技术结论优先核对论文、固定 revision 配置、模型卡和框架官方 recipe；博客及课程 notes 用于检查面向初学者的讲解路径，不用来替代模型事实来源。

## 检索记录

本轮使用网页搜索和直接打开来源，检索词包括：

- “Step 3.5 Flash technical report S3F1 MTP head query-head ablation”
- “Step 3.5 Flash config layer_types num_nextn_predict_layers revision”
- “Step 3.5 Flash vLLM recipe MTP num speculative tokens”
- “multi-token prediction auxiliary training vs speculative decoding explanation”
- “speculative decoding accepted prefix rejection sampling lecture notes”

检索范围涵盖 arXiv 技术报告、StepFun Hugging Face 模型卡、固定 Hugging Face config revision、vLLM 官方 recipe、speculative sampling 原始论文，以及研究者 MTP 图解和大学课程推理讲义。动态页面按本记录日期读取；vLLM recipe 的 main 分支没有固定到 commit，不代表其他版本的 serving 能力。

## 来源与证据边界

### StepFun 技术报告 v2

[Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters](https://arxiv.org/html/2602.10604v2)，StepFun，arXiv v2。重点阅读第 2.1、2.2、2.3 节及附录表 1、2、6，A.1、A.2、A.4。

- 支持：报告口径的 196B backbone/11B 每 token 激活参数；45 层中 3 dense + 42 MoE；每个 MoE 层 288 routed experts、top-8；起始 Full 后跟 11 组 3-SWA/1-Full；SWA 窗口 512、8 KV heads、Full/SWA query heads 64/96；MTP 头的未来偏移、每头模块和训练顺序；注意力消融、head-wise gate 定义及模拟证据。
- 表 6 把 backbone 统计列为 196B/11B，把含 MTP-3 统计列为 198B/13B，并明确 activated parameters 排除 embedding/output matrices。正文说每个 MTP head 增加约 0.81B；表内总量以 B 为单位取整。
- 表 1 的 30B-A3B 消融固定 W=512、关闭 MTP；训练预算约 1.4T tokens 后接 SFT。S3F1 的 normalized attention-side FLOPs 是 prefill/decode 分开归一化的结果；LongCtx 是该 proxy 协议下的指标。它不能证明 196B 发布模型的 wall-clock 加速。
- 表 2 的 sink 与 head-wise gate 比较属于 100B-A10B、约 250B token 的 pre-training-only 评估；附录 A.2 latency 是理想 workload 模拟。二者均不等于任意 runtime 上的实测效果。
- 第 2.1 节关于 SWA 相对线性注意力的文字是作者设计动机及当时证据判断，不是对线性注意力的普遍否定。

### 固定 Hugging Face 配置 revision

[Step-3.5-Flash config.json at revision 8847e9698c01fa9b33ed6dc5e0b0c24ad9883686](https://huggingface.co/stepfun-ai/Step-3.5-Flash/blob/8847e9698c01fa9b33ed6dc5e0b0c24ad9883686/config.json)，2026-11-25 读取。

- 支持：num_hidden_layers=45、max_seq_len=262144、vocab_size=128896、moe_layers_enum 为 3–44、288 routed experts、top-k=8、sliding_window=512、num_nextn_predict_layers=3 和 layer_types 的实际起始/重复次序。
- 配置字段描述 checkpoint 结构；它不能单独证明 256K 上限下的任务质量、某个 serving 后端启用了全部 MTP 头，或实际延迟。

### StepFun Hugging Face 模型卡

[Step 3.5 Flash model card](https://huggingface.co/stepfun-ai/Step-3.5-Flash)，动态页面，2026-11-25 读取第 2、4.1、4.3、6.1 节及部署示例。

- 支持：模型卡写出的 196.81B（196B backbone + 0.81B Head）、约 11B active、45 层、256K context、MoE 专家字段；卡片还报告典型 100–300 tok/s、单流编码任务峰值约 350 tok/s。吞吐是发布方陈述，未在该处提供足够条件形成跨硬件比较。
- 参数字段与技术报告附录的 MTP-3 合计值未完全对齐，公开资料没有将具体统计范围逐项对账。记录差异，不推测模型卡中的 Head 对应所有三个 MTP 模块。
- MTP 段的“4 tokens simultaneously”是宽泛产品表述；位置语义以技术报告给出的标准 LM head + 三个额外偏移为准，候选不等于已接受输出。
- 部署说明写 vLLM 尚未提供完整 MTP3 支持，示例将 num_nextn_predict_layers 和 speculative token 数设为 1。该动态页面只说明读取时公开的建议，不代表每个分支/commit 的完整支持状态。
- Context Manager 注释描述有效上下文超过阈值后 reset context 并 restart agent loop；BrowseComp 普通与带 Context Manager 的成绩标签不同。它们是模型卡报告的评测协议，不是架构部件的单因素实验。

### vLLM 官方 recipe

[vLLM recipes: Step-3.5-Flash.md](https://github.com/vllm-project/recipes/blob/main/StepFun/Step-3.5-Flash.md)，main 分支，2026-11-25 读取。

- MTP 示例设置 num_nextn_predict_layers=1、num_speculative_tokens=1；可支持“该公开 recipe 展示单层、单 speculative token 配置”，不能推出其他 recipe 或 runtime 版本都一样。
- recipe 的 bench serve 使用 random dataset、随机 2048 input/1024 output、最大并发 10，并展示样例输出；它不是同 checkpoint 开关 MTP 前后的配对实验，不能归因 MTP 增益。
- recipe 为可变 main 分支，复查部署行为时应固定 commit、模型 revision、量化、硬件与后端版本。

### 推测采样原始论文

Chen 等，[Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)，2023。

- 支持通用 draft—target 并行评分、从左向右接受草稿前缀、在首个拒绝位置进行 residual resampling，以及全部接受时最多返回 K+1 个 token 的标准路径。
- 论文的 2–2.5× 结果对应其 Chinchilla、硬件和任务设置；本章不将该结果移用于 Step 3.5。
- 实际实现可能采用不同停止、缓存和验证细节；Step 3.5 的部署语义仍需以具体 runtime 代码和配置为准。

### 教学材料

- Sebastian Raschka，[Multi-Token Prediction (MTP)](https://sebastianraschka.com/llm-architecture-gallery/mtp/)，2026-11-25 读取。用于区分 MTP 的训练辅助目标与 serving 时把它用作草稿器的两种角色，并提醒“checkpoint 有 MTP”不意味着常规推理栈自动启用；不用于证明 Step 3.5 的具体 head 定义。
- University of Edinburgh 2026-10，[Model Inference Systems](https://auth.opencourse.inf.ed.ac.uk/sites/default/files/2026-10/model_inference_systems.pdf)，第 30–32 页。图解 draft、单次 target 验证、接受前缀、残差替代 token 和 bonus token；用于设计具体状态轨迹，不代替 speculative sampling 原始论文。
- 上述教学材料与 StepFun 事实分开引用，不将其他 MTP 模型的模块连接方式外推到 Step 3.5。

## 采用结论与未解项

1. 45 层层型采用“第 0 层 Full + 11 组 S3F1”的描述；配置可数得 33 SWA、12 Full。SWA pair 统计明确包含当前位置，作为教学可见性交互账本，不当作 FLOPs 或服务延迟。
2. 参数统计并列保留技术报告 backbone 196B/11B、含 MTP-3 198B/13B、模型卡 196.81B/约 11B；公开口径差异未解决。
3. MTP 训练偏移写成标准 LM head 预测 x(t+1)，MTP-h 预测 x(t+1+h)。推测解码采用通用目标分布校正说明，明确输出数由接受/拒绝、终止与 runtime 决定。
4. SWA 消融、gate 消融、理想 workload 模拟、vLLM 随机 serving 示例和带 Context Manager 的 Agent benchmark 分属不同证据类型，不能合并成一条端到端加速结论。
5. 未下载 checkpoint，未在本地加载模型、运行 vLLM/SGLang、测接受率、测 latency 或吞吐；所有运行时效益仍需固定版本的本地/线上实验验证。
