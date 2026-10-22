# 第三册第六章：推理优化实战 审阅记录

## 范围与来源

- 正文：`book-03-practical-handbook/chapters/06-推理优化实战.md`，覆盖 6.1—6.7（KV Cache、Prefill/Decode、vLLM/PagedAttention、量化、Speculative Decoding、长上下文压测、实验报告）。
- 配套：第三册目录第 6 条指向本章文件，链接有效；`PAPERS.md` 已有 PagedAttention、Speculative Decoding、Medusa/EAGLE、GPTQ/AWQ/SmoothQuant/KIVI、Lost in the Middle/LongBench/RULER 入口，题库、练习、术语表已覆盖 KV Cache、推测解码与长上下文主题，本轮未重复复制。
- 环境：仓库 `.tmp/llm-book-cpu-check`，Python 3.12.9、PyTorch 2.12.0+cpu；未安装 CUDA、`transformers`、bitsandbytes。

通过指定代理 `http://10.24.27.134:7890` 检查正文 36 个唯一外部入口，最终均 HTTP 200。两个入口在本轮更新过：NVIDIA Transformer Engine FP8 primer 旧 `user-guide/examples/` 路径返回 404，改为当前 `examples/fp8_primer.html`；vLLM engine arguments 由 `serving/engine_args.html` 改为当前 `configuration/engine_args/`。arXiv Atom API 核对 16 个 arXiv 引用的题名，均与正文引用主题一致（Attention Is All You Need、GQA、QLoRA、GPTQ、AWQ、SmoothQuant、KIVI、PagedAttention、Speculative Decoding/Speculative Sampling、Medusa、EAGLE/EAGLE-2、Lost in the Middle、LongBench、RULER）。

论文入口支持机制与原论文实验；PyTorch/Hugging Face/vLLM/SGLang/TensorRT-LLM/NVIDIA 文档支持公开接口与参数；HTTP 状态和题名核对不等于逐篇全文复核，也不代表目标模型在目标硬件上的性能结论。

## 修订与本地验证

- 5 处 `\operatorname{...}`（Attn、softmax、shape、clip、round）改为规范要求的 `\mathrm{...}`；修改后全章 `\operatorname` 命中为 0。
- 60 个 `math` 围栏花括号与方括号计数平衡；未发现 `$$`、旧式 `\[`/`\]`、`\text{...}` 内下划线或尖括号下标写法。
- 11 个 Python 围栏全部通过语法编译；8 个无依赖块在 CPU 环境逐块独立执行通过，输出与正文一致：KV cache 容器 `prefill_range=(0,5)`、`decode_range=(5,6)`、`used_tokens=6`；RoPE 旋转 `position_0=(1.0,0.0)`、`position_5=(1.0,0.0005)`；延迟账本 `ttft_ms=[23.0,48.0,16.0,66.0]`、`ttft_p95_ms=63.3`；block allocator `free_blocks=4`；分组量化 group_size 8/4/2 的 `mse=0.0108375/0.00568827/0.00091071`；greedy 验证四组接受长度 0/3/1/0；位置检索 demo 三种长度的位置行 `[1,1,0,1,1]`；压测客户端契约块无输出运行通过。
- CUDA 计时块的前置守卫单独验证：`torch.cuda.is_available()` 为 False 时抛出 `RuntimeError("CUDA is required for this timing example")`，不触及未定义的 `model`。QLoRA 与 assisted generation 两块因缺少 `transformers`/CUDA 未执行，正文已声明依赖前提。
- 正文数值例子独立复算一致：KV Cache 单请求 1 GiB 与 MHA 4 GiB、KV block 2 MiB 与 1 GiB 池约 512 block/8192 token、7B 权重 FP16 约 13.0 GiB / INT8 约 6.5 GiB / INT4 约 3.3 GiB。
- 注入内容检查（公益/通知群/广告/推广/中转站/加群/QQ群/VX/微信/base64/无关链接）无命中；`git diff --check` 通过。

## 适用边界

验证说明的是本地 CPU 教学示例、容量账本公式和接口入口，不代表 CUDA/GPU kernel、FlashAttention 数值、vLLM/SGLang/TensorRT-LLM 真实调度行为、量化质量回归、推测解码真实接受率或长上下文实测。p95 分位数算法、压测矩阵和单位成功任务成本公式是测量定义，具体数字必须来自目标模型 revision、runtime 版本、硬件和代表性负载的成对实验。未运行外部数学渲染器；全局 PDF dry-run 因环境缺少 Pandoc 失败，未安装依赖。
