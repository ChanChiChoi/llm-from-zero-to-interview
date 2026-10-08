# 第一册第三份正文：训练、对齐与推理资料核验

核验日期：2026-10-07。用途：精修 `book-01-core-30/chapters/03-训练对齐推理.md` 第 16—30 讲。本记录用于追溯事实来源，不是目标模型配置、论文复现或生产系统评测。

## 资料与可支持范围

| 来源 | 等级 | 本轮用于支持的范围 | 边界 |
| --- | --- | --- | --- |
| [GPT-3](https://arxiv.org/abs/2005.14165) | `paper` | decoder-only 语言模型与 next-token 预训练的代表性技术报告 | 不代表所有模型的数据配方、训练规模或能力表现 |
| [Training Compute-Optimal Large Language Models（Chinchilla）](https://arxiv.org/abs/2203.15556) | `paper` | 参数量、训练 token 与计算预算应联合规划 | 实验结论绑定论文模型族、数据和计算设定 |
| [ZeRO](https://arxiv.org/abs/1910.02054) | `paper` | 优化器状态、梯度和参数分片的动机与方法 | 本地容量演算不覆盖框架临时 buffer、碎片和通信峰值 |
| [Megatron-LM](https://arxiv.org/abs/1909.08053) | `paper` | 张量并行、流水线并行的大规模训练方法入口 | 不据此推断任意集群的扩展效率 |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | `paper` | 示范数据、偏好数据、Reward Model 与 PPO 的代表性流程 | 论文结果只支持其模型、标注和训练配置 |
| [Scaling Instruction-Finetuned Language Models（FLAN）](https://arxiv.org/abs/2210.11416) | `paper` | 指令微调规模、任务多样性与泛化的论文实验 | 不保证增加任意指令数据都会提高目标任务效果 |
| [LoRA](https://arxiv.org/abs/2106.09685) | `paper` | 低秩适配矩阵及其参数效率动机 | 参数量公式不等于端到端显存或质量收益 |
| [QLoRA](https://arxiv.org/abs/2305.14314) | `paper` | 量化基座上的低秩微调方法 | 质量、显存和速度取决于校准、实现和硬件 |
| [Proximal Policy Optimization（PPO）](https://arxiv.org/abs/1707.06347) | `paper` | PPO 的裁剪策略目标和优化动机 | 算法论文不验证本章 toy RLHF 流程或具体超参数 |
| [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) | `paper` | 摘要任务中的偏好反馈与 Reward Model 研究实例 | 不将摘要实验推广成所有任务的人类偏好规律 |
| [Direct Preference Optimization（DPO）](https://arxiv.org/abs/2305.18290) | `paper` | DPO 目标、KL 约束推导及论文假设 | 理论关系依赖参考策略、偏好模型和数据分布等假设 |
| [PagedAttention](https://arxiv.org/abs/2309.06180) | `paper` | 分页 KV Cache 的系统设计与论文评测 | 不保证其他引擎、请求分布或硬件有相同吞吐收益 |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | `paper` | IO-aware 分块 attention 和减少 HBM 往返的机制 | 实际速度取决于 GPU、shape、dtype 和 kernel；浮点顺序可能造成舍入差异 |
| [GPTQ](https://arxiv.org/abs/2210.17323) | `paper` | 逐层后训练量化的方法入口 | 论文质量/速度结果绑定校准集、模型和实现设置 |
| [AWQ](https://arxiv.org/abs/2306.00978) | `paper` | activation-aware 权重量化的思路 | 不据此宣称 AWQ 对所有模型或硬件最优 |
| [TruthfulQA](https://arxiv.org/abs/2109.07958) | `paper` | 事实性评估研究入口及其特定基准设计 | 单一基准不覆盖全部事实性、安全或实际用户风险 |
| [HELM](https://arxiv.org/abs/2211.09110) | `paper` | 多维语言模型评估框架入口 | 不将其分数视为跨配置、跨 harness 的通用排序 |
| [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) | `paper` | 特定模型和可验证任务中的搜索及自适应推理预算实验 | 不证明增加推理预算对所有模型、任务或成本目标均有效 |
| [PyTorch `CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) | `official` | `ignore_index` 的公开接口语义 | 本地示例验证 CPU 教学路径，不覆盖所有训练框架封装 |
| [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html) 与 [FSDP](https://pytorch.org/docs/stable/fsdp.html) | `official` | 分布式通信和 FSDP 公开接口入口 | 稳定版页面随版本更新；没有在本轮运行多卡训练 |
| [Transformers chat template 文档，固定 revision](https://github.com/huggingface/transformers/blob/e598fbad926d80bc356c0c4d96532030e3dfdb62/docs/source/en/chat_templating.md) | `source` | 指定 revision 的消息序列化与 generation prompt 说明 | 只支持该文档 revision，不代表所有 tokenizer 的模板 |
| [Transformers generation 文档，固定 revision](https://github.com/huggingface/transformers/blob/e598fbad926d80bc356c0c4d96532030e3dfdb62/docs/source/en/main_classes/text_generation.md) | `source` | 指定 revision 的生成参数和公开接口说明 | 实际边界行为仍要以安装版本和运行结果为准 |
| [TRL DPO Trainer 文档，固定 revision](https://github.com/huggingface/trl/blob/adff4933bca014c6110436c8ba6bdbe88b28283c/docs/source/dpo_trainer.md) | `source` | 指定 revision 的偏好数据字段与 trainer 接口 | 不证明训练质量、稳定性或所有版本的兼容性 |
| [vLLM PagedAttention 设计文档](https://docs.vllm.ai/en/latest/design/paged_attention.html) | `official` | vLLM 文档所述的分页注意力设计入口 | `latest` 页面会更新；本章未运行 vLLM serving benchmark |
| [OWASP LLM01:2025 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) | `official` | 直接/间接 prompt injection 风险与权限最小化等缓解方向 | 风险清单不是对某个应用安全性的验收 |

## 链接与访问记录

章节原有的 19 个外部来源链接在本轮初查返回 HTTP 200。Transformers 与 TRL 的两个 Hugging Face 动态文档页经当前代理返回 503，因此正文改为上述 GitHub 固定 revision 页面；这些 revision 用于稳定引用范围，不把文档描述写成目标模型的实测行为。新增的 OWASP 页面和 test-time compute 论文已读取并用于限定相应结论。PyTorch 稳定版和 vLLM `latest` 文档是动态页面，维护时应按实际依赖版本重新核对。

`PAPERS.md` 已有 Snell 等人的 test-time compute 论文入口，本轮无需重复添加。正文保留读者所需引用，完整证据范围和访问限制记录在本文件。

## 本地教学计算与代码验证

- 临时环境：Python 3.13.13、NumPy 2.4.6、PyTorch 2.12.0+cpu；依赖安装在 `/tmp/llm-book-deps`，未改动项目依赖。
- 最终版本中的 13 个 Python 代码块逐块独立运行，均通过；其中采样 demo 输出温度升高后分布变平，top-k/top-p 示例各保留三个候选。
- 采样 demo 另通过 38 项边界断言，覆盖零/负/非有限/不可表示温度、空或错误 shape logits、非浮点 logits、非法 `k`/`p`、负/非有限/全零/求和溢出概率，以及输出归一化和候选数量。没有覆盖 CUDA 或其他 PyTorch 版本。
- 本轮此前记录的 20 项代码边界检查覆盖训练 FLOPs、显存、SFT mask、KV Cache、量化和评估示例。最终全体代码块重新运行通过；边界结果只对记录的输入和 CPU 环境成立。
- packing、AdamW 状态、SFT next-token shift、KV Cache 与 attention matrix、FlashAttention IO 账本、量化误差和评估分母均是由公式/代码定义直接核算的教学例子，不构成模型训练或 serving 复现。

## 未验证项

- 未在 CUDA/GPU 上测试 FlashAttention kernel、量化 kernel、显存峰值、吞吐或不同浮点实现的误差。
- 未运行真实模型的预训练、SFT、DPO/PPO、Reward Model 或安全评测；示例只验证机制、shape、索引、容量公式和边界处理。
- 未执行多卡 DDP/FSDP/ZeRO 训练，也未用真实请求分布验证 KV Cache 并发、延迟或 vLLM 调度表现。
- 论文结论、官方接口和本地教学 demo 的证据等级保持分开；不能将上述结果外推为生产质量或目标模型结论。
