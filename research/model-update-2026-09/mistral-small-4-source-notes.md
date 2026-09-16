# Mistral Small 4：官方模型卡核验

核验日期：2026-09-09。主要来源为 [Mistral-Small-4-119B-2603 模型卡](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) 及其仓库 README。模型卡是 Mistral AI 发布的第一方资料，但 Hugging Face 的 `lastModified` 字段不能直接当作发布日。

## 已确认

- 模型名称为 **Mistral Small 4 119B A6B**，采用稀疏 MoE：128 个专家、每 token 激活 4 个；总参数约 119B，每 token 激活约 6.5B。
- 上下文长度为 256K；输入支持文本和图像，输出为文本。
- 同一模型提供 instruct 与 reasoning 两种工作模式，通过每请求 `reasoning_effort` 切换：`none` 表示不启用推理，`high` 表示启用推理。
- 模型卡称它统一 Instruct、Reasoning（此前称 Magistral）与 Devstral 三个模型家族的能力。
- README 披露了训练好的 EAGLE speculative-decoding head 和 NVFP4 量化检查点，可分别用于推测解码和 4-bit 浮点推理。
- 模型卡采用 Apache 2.0 许可证，允许商业和非商业使用；部署示例覆盖 vLLM、Transformers、SGLang 与 llama.cpp。
- 官方对比称，相比 Mistral Small 3，在其测试环境中端到端完成时间降低 40%，吞吐提高 3 倍；这是发布方基准，不能外推到所有硬件。

## 可以扩写的技术点

1. 单模型模式切换如何把推理预算变成 API 参数，以及评测时为何要固定该参数。
2. MoE 的 total/active parameters、专家路由和显存账本。
3. EAGLE 草稿头与 NVFP4 量化如何影响吞吐、显存和质量。
4. 多模态输入、函数调用和 agent 场景中的运行时边界。

## 尚待核验

模型卡没有给出完整训练数据、优化器、RL 算法、层数的完整配置、专家负载均衡损失或一份独立技术报告。模型名中的 `2603` 也不能单独证明发布日期。发布方 benchmark 的硬件、批大小和实现细节需在复现实验中重新固定。
