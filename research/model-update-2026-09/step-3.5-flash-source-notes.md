# Step 3.5 Flash：官方模型卡核验

核验日期：2026-09-09。来源为 [StepFun 官方 Hugging Face 模型卡](https://huggingface.co/stepfun-ai/Step-3.5-Flash)，模型卡链接到技术报告 [arXiv:2602.10604](https://arxiv.org/abs/2602.10604)、PaCoRe 论文和模型系统协同设计论文。

## 已确认

- Step 3.5 Flash 是 Apache 2.0 开源模型，稀疏 MoE 总参数约 196.81B，每 token 激活约 11B。
- 模型卡给出的骨干规格为 45 层、4096 hidden size、128,896 词表、256K context；每层有 288 个 routed experts 和 1 个 shared expert，路由选择 top-8 experts。
- 推理加速使用 3-way Multi-Token Prediction（MTP-3）头；模型卡称典型吞吐 100–300 tok/s，单流编码任务峰值约 350 tok/s。vLLM 当前页面同时注明完整 MTP3 支持仍在集成中，因此“模型具备 MTP 头”和“任意后端都能达到宣传吞吐”必须分开。
- 长上下文采用 3:1 Sliding Window Attention 与 full-attention 的混合比例，目标是在 256K 窗口下降低注意力成本。
- 模型卡给出 SWE-bench Verified 74.4%、Terminal-Bench 2.0 51.0% 等结果，并说明带 Context Manager 的 BrowseComp 会在上下文超过阈值时重启 agent loop；这些数字依赖其评测协议。
- 仓库提供 Transformers、vLLM、SGLang 和 OpenRouter/StepFun API 示例。

## 资料边界

模型卡是可追溯的一手实现资料，但 196B/11B、MTP-3、SWA 比例和 benchmark 都应绑定具体 revision、硬件和后端。不能从吞吐数字推导训练 FLOPs，也不能把 context manager 的 agent 策略写成模型内部记忆机制。完整训练数据和 RL 细节仍需通读技术报告。
