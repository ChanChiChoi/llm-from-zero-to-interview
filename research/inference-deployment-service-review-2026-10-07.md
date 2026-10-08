# 推理部署与服务进阶精修研究记录（2026-10-07）

## 范围与结论

- `chapter`：`book-02-advanced-100/chapters/06-推理部署与服务进阶.md`。
- `scope`：第 66—75 节；当前章节 1,754 行，目录映射一致。
- `evidence_level`：`paper`、`official`、`local-demo`。
- `status`：章节复读、修订和声明范围内的验证完成；未将教学示例解释为目标 GPU、runtime 或生产服务实测。

## 主要修订

- 将 TTFT、TPOT、ITL、端到端时延、请求输出速率、服务吞吐和 goodput 的计时范围/分母分开；输出为零或一个 token 时不伪造 TPOT。
- KV block allocator 对共享请求的 block 集合先完成类型、有效性和引用计数检查，再统一修改状态，避免部分更新；调度器示例加入请求动态到达。
- 明确推测解码候选/目标概率需归一化，量化实现对非有限输入拒绝并解释全零张量的 scale 哨兵；容量和回归审计示例补足有限值、空样本与布尔类型边界。
- 服务章节用事件状态审计区分排队、完成和错误；端云路由把 `local_only` 作为硬约束，超出端侧上下文时返回错误而不是上传云端。
- Prefix Cache 的身份键绑定会影响 K/V 计算的 token、模型和配置；只影响后续采样的 temperature/top-p 通常不要求新建 K/V 缓存身份。
- TGI 引用固定到归档仓库中的文档快照并标明历史接口；bitsandbytes 改用官方仓库入口；容量指标和端侧检查明确教学估算与目标设备实测的区别。

## 论文元数据与链接

经 arXiv Atom 元数据核对的 13 个 ID 与题名如下：

| ID | 题名 |
| --- | --- |
| [2205.14135](https://arxiv.org/abs/2205.14135) | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness |
| [2307.08691](https://arxiv.org/abs/2307.08691) | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning |
| [2305.13245](https://arxiv.org/abs/2305.13245) | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints |
| [2402.02750](https://arxiv.org/abs/2402.02750) | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache |
| [2309.06180](https://arxiv.org/abs/2309.06180) | Efficient Memory Management for Large Language Model Serving with PagedAttention |
| [2312.07104](https://arxiv.org/abs/2312.07104) | SGLang: Efficient Execution of Structured Language Model Programs |
| [2403.02310](https://arxiv.org/abs/2403.02310) | Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve |
| [2211.17192](https://arxiv.org/abs/2211.17192) | Fast Inference from Transformers via Speculative Decoding |
| [2401.10774](https://arxiv.org/abs/2401.10774) | Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads |
| [2401.15077](https://arxiv.org/abs/2401.15077) | EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty |
| [2210.17323](https://arxiv.org/abs/2210.17323) | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers |
| [2306.00978](https://arxiv.org/abs/2306.00978) | AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration |
| [2211.10438](https://arxiv.org/abs/2211.10438) | SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models |

论文 ID/题名核对用于确认读者入口与论文对象相符，不等于逐篇复核完整实验或验证其速度、质量数字可迁移到其他硬件/负载。

章节含 31 个唯一外链。链接检查记录为 30 个 HTTP 200；OpenAI latency optimization guide 在当前代理路径返回 403，引用仅作为工程主题入口，不把不可访问解释为内容结论。TGI 文档使用 commit `b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed` 的文档快照，快照可访问，仓库 metadata 标记为 archived，因此不作为当前维护建议。bitsandbytes 指向官方仓库；使用时应按具体 release 核对硬件能力。

## 本地代码与公式检查

- 10 个 Python 代码块逐块运行成功，使用 Python 3.12.9；章节示例只依赖标准库。
- `.tmp/verify_chapter6.py` 在 Git 忽略的 `.tmp/llm-book-cpu-check/` 环境运行；10 组边界测试通过，覆盖非法/非有限时间账本输入、零输出指标、分数 token 数、KV block 分享原子性、非法到达时间、概率分布归一化、量化零/NaN、事件审计、容量 NaN、回归样本和本地专属路由等。
- 环境使用 PyTorch 2.12.0+cpu、NumPy 2.4.6，`torch.cuda.is_available()` 为 False；本章代码并不依赖 PyTorch。没有安装 CUDA 库，也没有把环境放进系统 `/tmp`。
- 40 个 `math` 围栏；代码、文本、公式围栏成对，公式花括号计数平衡。重点核算 TTFT/TPOT 定义、KV Cache 字节数、调度预算、推测解码接受率、量化误差、吞吐和单位成功成本。没有使用外部 Markdown 数学渲染器。
- 10 个 Python 块输出均与正文列出的示例输出对应；验证说明这些教学函数在给定输入下的行为，不构成 GPU kernel、runtime、目标模型质量、生产吞吐或论文结果复现。

## 配套同步与未验证项

- `PAPERS.md` 增补 FlashAttention、Speculative Decoding、Medusa/EAGLE、GPTQ/AWQ/SmoothQuant/KIVI、Orca、Sarathi-Serve 和 SGLang 入口。已有题库、面试库和术语表已覆盖主要概念，因此未重复复制章节段落。
- 第二册目录中第 66—75 讲映射到本章，下一章第 76—86 讲链接存在；无相对 Markdown 链接。
- 未使用目标 GPU/CUDA、实际 vLLM/SGLang/TensorRT-LLM/TGI runtime 或真实请求轨迹；未实测吞吐、goodput、功耗、显存峰值、端侧发热、量化任务质量及生产安全性。论文元数据核验不等于全文复核或 benchmark 复现。
