# SFT loss、参数高效微调与能力遗忘：来源证据

检索日期：2026-10-28。

本记录只保存联网核验得到的知识点及来源边界，不记录书籍审阅过程、正文修改或进度状态。

## 目标 mask 与交叉熵分母

- PyTorch, `torch.nn.CrossEntropyLoss`, v2.14，[官方 API 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)。
  - 实际阅读范围：类别索引目标的 loss 公式、`ignore_index`、`reduction='mean'` 定义和张量 shape。
  - 来源支持：输入可为未归一化 logits；类别索引等于 `ignore_index` 时该项不参与 loss/输入梯度；类别索引目标采用 `mean` reduction 时对未忽略目标求平均（如设置类别权重，分母按有效权重和）。
  - 证据等级：source-backed（PyTorch v2.14 公开 API 语义）。
  - 适用边界：文档定义损失接口，不决定语言模型的 causal shift 在模型内部还是调用方完成；自定义的样本加权、分布式归约和每设备平均可能改变聚合口径。正文公式是未加权、按有效目标 token 求平均的教学目标。

## LoRA 的低秩增量

- Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” arXiv:2106.09685v2 (2021)，[论文 HTML](https://arxiv.org/html/2106.09685)。
  - 实际阅读范围：摘要、§2 问题设定、§4.1 低秩参数化、§4.2 Transformer 中的应用、§5 实验与 §7 rank 分析。
  - 来源支持：LoRA 冻结预训练权重，对所选权重增量使用低秩可训练矩阵；作者在论文指定的 RoBERTa、DeBERTa、GPT-2、GPT-3 任务与配置中报告了与全参微调相当或更好的指标，并讨论参数和显存节省。
  - 证据等级：source-backed（原始方法和指定模型/数据/评测实验）。
  - 适用边界：论文结果不构成跨任务质量保证；rank、目标矩阵、初始化、优化器和数据会影响结果。公式中的低秩形式说明参数化约束，不单独决定全模型显存、吞吐或输出质量。

## QLoRA 的量化基座适配

- Dettmers et al., “QLoRA: Efficient Finetuning of Quantized LLMs,” arXiv:2305.14314v1 (2023)，[论文 HTML](https://arxiv.org/html/2305.14314)；后发表于 NeurIPS 2023。
  - 实际阅读范围：摘要、§2–3 的量化与 adapter 背景、NF4/double quantization/paged optimizer 方法，以及 §4 和 Limitations/Discussion。
  - 来源支持：论文方法对冻结的 4-bit 量化预训练模型进行前向与反向传播，并更新 LoRA adapter；NF4、double quantization 和 paged optimizers 是其具体方法组成。作者在所测模型、数据和硬件上报告了显存与任务结果。
  - 证据等级：source-backed（方法定义和论文实验）。
  - 适用边界：论文报告可在单 48GB GPU 上训练 65B 模型，但其与 16-bit 全参微调的受控性能比较只覆盖较小规模实验；作者明确表示没有证明 33B/65B 也能匹配全参微调。低比特存储不意味着所有激活、adapter optimizer 状态或临时工作区同样量化，也不保证任意任务无性能差异。

## LoFT 的低秩优化器状态（2026）

- Tastan et al., “LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning,” ICLR 2026，[官方会议摘要](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7428310c0f97f1c6bb2ef1be99c1ec2a-Abstract-Conference.html)。
  - 实际阅读范围：ICLR 官方论文页摘要；未在本记录中复核全文算法推导与实验表格。
  - 来源支持：摘要称 LoFT 在低秩更新之外，也将 Adam 一阶、二阶矩投影到低秩子空间，以对齐全参更新的优化器动态；作者报告在其训练设置下缩小了 adapter 与全参微调的效果差距。
  - 证据等级：source-backed（ICLR 2026 同行评审论文的官方摘要）。
  - 适用边界：摘要所述表现绑定论文模型、任务、优化实现和预算；不证明所有标准 LoRA 都表现不佳，也不支持把 LoFT 的优化器行为推广到任意 adapter/训练框架。

## 微调过程中的遗忘

- Kalajdzievski, “Scaling Laws for Forgetting When Fine-Tuning Large Language Models,” arXiv:2401.05605v1 (2024)，[论文 HTML](https://arxiv.org/html/2401.05605)。
  - 实际阅读范围：摘要、§1 问题与贡献、§3 实验设定、§4 观察和 §5 结论。
  - 来源支持：论文以 Llama 2 7B Chat、指定微调数据、单 epoch 及若干 LoRA 配置研究新任务效果和先前分布变化；报告 LoRA 微调也可能出现遗忘，并提出其设置下的量化关系。
  - 证据等级：source-backed（原研究的模型、数据、训练和评测实验）。
  - 适用边界：这是单一模型家族及有限数据/配置的研究，不支持所有 SFT 都会发生同样遗忘，也不能把其 scaling relation 外推为通用预测器。正文只用它说明参数高效方法不能自动保证旧能力不变。

## 固定版本接口与模型快照

- Hugging Face Transformers v5.17.0, [bitsandbytes quantization guide](https://huggingface.co/docs/transformers/v5.17.0/quantization/bitsandbytes), `Finetuning` 段。
  - 实际阅读范围：训练设备放置说明及 QLoRA/NF4 使用段落。
  - 来源支持：文档指出 8-bit 或 4-bit 量化训练仅支持训练额外参数；该版本明确写出 `device_map="auto"` 只应用于推理。该页面说明用于加载量化模型的 `dtype="auto"` 以模型配置或 checkpoint dtype 为线索。
  - 证据等级：source-backed（固定版本官方文档）。
  - 适用边界：文档中的建议不替代具体训练框架的单卡、多卡、FSDP 或 DeepSpeed 设备策略；其他 Transformers 版本应查对应文档。
- Hugging Face Transformers v5.17.0, [tokenizer/chat template API](https://huggingface.co/docs/transformers/v5.17.0/main_classes/tokenizer), `apply_chat_template` 参数。
  - 实际阅读范围：`return_assistant_tokens_mask` 定义和 generation marker 条件。
  - 来源支持：assistant mask 输出只适用于模板支持 `{% generation %}` 的情况；掩码以 1 标记 assistant 生成 token、以 0 标记 system/user token。
  - 证据等级：source-backed（固定版本官方 API 文档）。
  - 适用边界：具体模型模板可能不提供 generation 区域；调用方仍须检查实际返回字段和监督覆盖。
- Hugging Face TRL v1.14.1, [SFTTrainer 文档](https://huggingface.co/docs/trl/v1.14.1/sft_trainer)，assistant/completion-only loss 与 pre-tokenized labels 段落。
  - 来源支持：`assistant_only_loss` 对 conversational dataset 使用，并要求可识别 assistant generation 区间；prompt-completion dataset 的 `completion_only_loss` 监督 completion；预 tokenized 数据含 `labels` 时训练器按提供的 labels 使用。
  - 证据等级：source-backed（固定版本官方文档）。
  - 适用边界：数据字段、模板和 trainer/preprocessing 版本决定具体 mask；不同数据格式不能仅凭参数名类比。
- Hugging Face PEFT v0.18.0, [LoRA API](https://huggingface.co/docs/peft/v0.18.0/package_reference/lora)。
  - 来源支持：`fan_in_fan_out=True` 适配 GPT-2 `Conv1D` 的 `(fan_in, fan_out)` 权重布局；常规 scaling 是 `lora_alpha/r`，`use_rslora=True` 时为 `lora_alpha/sqrt(r)`；默认 `init_lora_weights=True` 将 LoRA B 置零，因此未训练 adapter 为 no-op。
  - 证据等级：source-backed（固定版本官方 API 文档）。
  - 适用边界：其他 PEFT 版本、模型层封装和自定义初始化应以对应实现为准。
- Hugging Face Model Hub snapshots: [`sshleifer/tiny-gpt2` at `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be](https://huggingface.co/sshleifer/tiny-gpt2/tree/5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be); [`Qwen/Qwen2.5-0.5B` at `b937cc073b55e980e6f21d5f8b7a77818efcf0eb`](https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/b937cc073b55e980e6f21d5f8b7a77818efcf0eb).
  - 实际阅读范围：两个固定 commit 的文件树、模型配置及模型卡 Transformers 示例。
  - 来源支持：tiny-gpt2 的 5f91 snapshot 包含 `pytorch_model.bin`、配置和 tokenizer 文件；另一个早期 [commit `725ee81525af8ec6431c83b8522a72bd5281472d`](https://huggingface.co/sshleifer/tiny-gpt2/tree/725ee81525af8ec6431c83b8522a72bd5281472d) 的文件树只列出配置/tokenizer 文件，不适合作为加载权重的固定 revision。Qwen snapshot 包含 Qwen2 配置、Safetensors 权重和 tokenizer 文件；该快照的官方示例使用 Auto 类，未要求 `trust_remote_code=True`。
  - 证据等级：source-backed（Hub 固定 revision 文件树与配置）。
  - 适用边界：revision 固定解决仓库内容漂移，不固定 Transformers、PEFT、bitsandbytes、CUDA 或硬件运行环境；这些仍要分别记录。

## 生命周期

loss mask 接口、LoRA/QLoRA 方法和指定实验结果：active。具体训练目标、有效 token 归一化、量化内核、参数层选择、部署合并和能力回归必须按实际框架、模型、任务与评测单独核对。
