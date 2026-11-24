# GPT、LLaMA 与 Mistral 架构比较：按公开版本核验

研究日期：2026-11-20。问题范围：为第二十一册第十七章确定可以逐项复核的 GPT、LLaMA、Mistral 架构样本；区分共享的 decoder-only 主干、具体版本的结构选择，以及模型卡中的位置/上下文字段。本文只记录来源事实与证据边界；章节职责、内容取舍和修改见 `progress/book-21-transformer-architecture-evolution/chapter-17-audit.md`。

## 检索与筛选

- 使用英文与中文检索词，包括 `GPT-3 architecture`, `GPT-4 technical report architecture details`, `gpt-oss model architecture`, `Llama 3 architecture GQA RoPE`, `Mistral 7B sliding window attention`, `GQA grouped-query attention`, `LLM architecture comparison explained`、`transformer architecture lecture notes`。先检索论文、官方模型卡与发布资料，并行查找研究者技术文章和课程讲义，再打开支持具体主张的原文位置。
- 研究对象限定为公开、可命名的模型或版本，不把 GPT、LLaMA、Mistral 品牌名当成稳定模板。核心样本为 GPT-3 175B（历史公开基线）、GPT-4 技术报告（披露边界）、gpt-oss-20b（OpenAI 公开权重样本）、Llama 3/3.1 8B（Meta 发布版本）和 Mistral 7B v0.1。
- 博客和课程用于发现解释顺序与可视化方法，不用于替代论文/官方文件核验架构数字。性能比较不属于本章问题。

## 已核验事实与支持边界

### GPT：同一名称下需要区分公开对象

- Brown 等人的 GPT-3 论文 §2.1 将实验模型描述为沿用 GPT-2 模型与架构（包括改良初始化、预归一化和可逆 tokenization），例外是 Transformer 层交替使用 dense 与 locally banded sparse attention。论文表 2.1 中 GPT-3 175B 为 96 层、模型宽度 12,288、96 个注意力头、每头 128 维；其上下文窗口为 2,048 tokens。论文支持这些具体设计，不支持将其推广为所有 GPT 版本的固定配置。[GPT-3 论文](https://arxiv.org/abs/2005.14165)
- OpenAI 归档的 GPT-2 参考实现把 token embedding 与可学习的位置表 `wpe` 相加，使用 LayerNorm、GELU MLP 与多头因果 attention。将这些字段用于 GPT-3 是对“GPT-3 沿用 GPT-2、只报告特定 attention pattern 例外”的有来源依据的推断；正文需标明这是 GPT-2/3 公开基线，不是闭源新 GPT 的规格。[OpenAI GPT-2 参考实现](https://github.com/openai/gpt-2/blob/master/src/model.py)
- OpenAI 的 GPT-4 技术报告 §2 明确表示，报告不再披露架构（包括模型规模）、硬件、训练计算量、数据集构建或训练方法等细节。因此，报告只能支持“这份技术报告未披露 block 级规格”，不能据此推测 GPT-4 的层、attention 类型、MoE 或内部配置。[GPT-4 Technical Report](https://cdn.openai.com/papers/gpt-4.pdf)
- OpenAI 的 gpt-oss 模型卡（2025-08-05）说明 gpt-oss 是建立在 GPT-2/GPT-3 架构基础上的自回归 MoE Transformer。gpt-oss-20b 为 24 层、20.9B 总参数、3.61B 每 token 激活参数；模型卡描述 RMSNorm/pre-LN、SwiGLU MoE、每 token 选择 top-4 experts，以及带宽 128 的交替局部窗口/全局 dense attention。每层有 64 个 query heads、8 个 KV heads，使用 RoPE，并通过 YaRN 扩展 dense attention 的上下文到 131,072 tokens。OpenAI 产品文章简述其 128k context 与 GQA group size 8。[gpt-oss 模型卡](https://deploymentsafety.openai.com/gpt-oss/architecture)；[OpenAI 发布说明](https://openai.com/index/introducing-gpt-oss/)
- gpt-oss 与 GPT-3 的报告表明，公开 GPT 示例本身就有版本差异：GPT-3 的披露重点是 dense/local attention 交替；gpt-oss 同样报告局部/全局 attention，但还采用 GQA 与 MoE。GPT-4 的 block 细节则未由报告披露。该对照足以驳斥“GPT 是一个固定 block 配方”，不支持关于未公开 GPT 服务模型的推断。

### LLaMA：明确 checkpoint 与扩展版本

- 原始 LLaMA 论文 §2.2 明确报告 pre-normalization/RMSNorm、SwiGLU 和 RoPE。它描述的是 2023 年原始 LLaMA 配方。[LLaMA 论文](https://arxiv.org/abs/2302.13971)
- Meta 的 Llama 3 技术报告 §3.2、表 3 说明 8B 模型使用 32 层、模型维度 4,096、32 个 attention heads、8 个 KV heads、FFN 维度 14,336、SwiGLU、128k 词表和 RoPE（θ=500,000）。正文说明采用 GQA；这是 Llama 3 公布模型的证据，不自动概括后续 Llama 版本。[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- Meta 的 Llama 3.1 官方模型卡列出 8B/70B/405B 版本均采用 GQA、context length 为 128k，发布时间为 2024-07-23。该卡用于核验 Llama 3.1 的发布范围和上下文声明；Llama 3 与 3.1 的数字不能混为同一版本。[Llama 3.1 Model Card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md)
- Meta 卡片不单独给出所有 block 字段；本章仅将 Llama 3 论文表格与 Llama 3.1 官方卡各自对应到它们实际描述的版本，不把一版的字段无标记迁移到另一版。

### Mistral：区分窗口、传播跨度与上下文字段

- Mistral 7B 论文 §2 表 1 列出 dim=4,096、32 层、32 个 query heads、8 个 KV heads、window size=4,096 和 context_len=8,192；论文说明模型使用 GQA 与 sliding-window attention。它说明局部窗口外的信息可经堆叠层逐层传递，报告约 131K 的理论 attention span。该“理论 span”是逐层信息可达范围，不等同于声明的 8,192-token context_len，也不证明模型在 131K 输入上具备可靠能力。[Mistral 7B](https://arxiv.org/abs/2310.06825)
- 为固定发布配置另查了 Mistral 官方 Hugging Face 仓库中 `Mistral-7B-v0.1/config.json` 的 revision `93bc63fc98d90b891e2ab513e1ebb7480424cec3`：配置包括 `num_attention_heads=32`、`num_key_value_heads=8`、`sliding_window=4096`，并将 `max_position_embeddings` 写为 32,768。该配置字段与论文表中的 `context_len=8192` 名称和数值不同；应并列标注字段来源，不能直接替换或假设字段语义相同。[固定 revision 的官方配置](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/93bc63fc98d90b891e2ab513e1ebb7480424cec3/config.json)
- 为补充核对该固定配置中的 block 组件，读取 Hugging Face Transformers v4.34.0 的 `modeling_mistral.py`：该版本实现 `MistralRMSNorm`、在 Q/K 上应用 RoPE，并以 `gate_proj`、`up_proj`、SiLU 门控和 `down_proj` 实现 MLP。结合固定配置的 `rms_norm_eps`、`rope_theta` 与 `hidden_act` 字段，可将这一特定配置解释为 pre-norm RMSNorm、RoPE 和 SiLU-gated FFN；这条实现证据只支持所查 Transformers 版本与该配置的对应解释，不代表 Mistral 家族所有版本或官方性能结果。[Transformers v4.34.0 implementation](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mistral/modeling_mistral.py)
- Mistral 论文将一次 16K 输入长度、W=4,096、修改过的 FlashAttention/xFormers 实现下的 2× speed improvement 作为特定实验报告；32K 下的 8× cache-memory reduction 也绑定论文配置。它们不是跨硬件、内核、版本的架构性能保证，本章不将这些数字用作模型家族排名。

### GQA 的可讲解定义与证据边界

- Ainslie 等人将 GQA 定义为 query heads 分组、每组共享一个 key/value head；KV-head 数介于 MQA 的 1 个和 MHA 的每 query head 一个之间。论文的 T5 checkpoint uptraining 实验显示 GQA 在该设置下接近 MHA 的质量、速度接近 MQA。可用这一研究说明结构意图与实验条件；不要把“相近质量/速度”写成所有模型、负载或硬件上的保证。[GQA 原论文](https://arxiv.org/abs/2305.13245)
- 若正文只需解释配置字段的读法，可用一两句说明 GQA 的共享关系，并回链第四章和第十一章；不在本章重复 KV-cache 公式、显存演算或注意力机制完整推导。

## 教学组织材料

- Sebastian Raschka 的《The Big LLM Architecture Comparison》（2025-07-19 发布，2026-04-02 更新）明确分离架构变化与 benchmark/training recipe，逐模型解释组件差异，并提醒数据、训练技巧和超参数的差异妨碍性能归因。它适合作为本章“先定版本—对齐组件—解释差别—限定结论”的教学组织样本，不作配置事实来源。[文章](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
- Raschka 的 GPT/Llama/Qwen/Gemma 简表明确把 GPT-2/3、Llama 3 等标成代表性设计，并说明家族名不足以精确比较、闭源 GPT 缺少完整 block 规格。其维度（归一化、位置、attention、FFN、dense/MoE）可作为比较坐标的候选，正文需按本章 GPT/LLaMA/Mistral 具体公开版本重新核验。[高层架构比较](https://sebastianraschka.com/faq/docs/gpt-llama-qwen-gemma-comparison.html)
- 更新中的 LLM Architecture Gallery 用可筛选的逐模型事实卡和 side-by-side 视图展示 GPT-2、Llama、gpt-oss、Mistral 等结构，适合借鉴“明确模型名、日期、规模和关键组件”的呈现方式。它是个人维护的解释/导航资源，卡片不替代论文、官方模型卡或固定 revision。[LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- Princeton COS 484 (Spring 2024) lectures 13、15、16 先介绍 Transformer 信息流/attention，再进入 context 与 GPT/decoder-only，提供了从共用主干到生成模型的课堂顺序。课件中的简化断言不直接移入书稿；章节 16 已负责三类信息流，本章仅借鉴“先唤回共享架构、再比较 block 选择”的衔接。[COS 484 lecture 13](https://nlp.cs.princeton.edu/cos484-sp24/lectures/lec13.pdf)、[lecture 15](https://nlp.cs.princeton.edu/cos484-sp24/lectures/lec15.pdf)、[lecture 16](https://nlp.cs.princeton.edu/cos484-sp24/lectures/lec16.pdf)

## 内容边界与待核项

- 架构选择改变的是组件/可见性/状态与计算路径；单纯比较不同模型的 benchmark 或推理速度无法把因果归到架构。需分开记录 model id、训练/指令版本、参数规模、tokenizer、数据与训练 recipe、上下文扩展、软件/硬件及测量条件。
- 模型卡字段（如 `context_len`、`max_position_embeddings`、`sliding_window`、dense layer 的长上下文）有不同语义。读者应追踪字段所属文件、revision 与定义；不能仅用一个“最大上下文”数字代替窗口范围、理论感受野、训练长度和可用长度。
- 本章采用的样本分别绑定 GPT-3 论文、GPT-4 报告、gpt-oss-20b 官方卡、Llama 3/3.1 官方材料、Mistral 7B v0.1 论文/配置。不会推断 GPT-4/后续闭源 GPT 内部结构，也不会从一个 checkpoint 外推所有 LLaMA 或 Mistral 版本。
- 研究未尝试对所有 2026 模型做目录式盘点；近期模型谱系及单模型前沿架构留给本册第 51 章后续谱系专题与第 79—95 章逐章审阅。
