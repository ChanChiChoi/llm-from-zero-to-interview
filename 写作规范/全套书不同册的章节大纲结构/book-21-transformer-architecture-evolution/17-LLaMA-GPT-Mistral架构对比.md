# 第十七章：LLaMA、GPT、Mistral 风格架构对比

## 章节契约

- **定位：** 承接第十六章的 decoder-only 通用生成接口，比较公开模型在 decoder block 中做出的具体结构选择，训练读者按模型 ID、版本和来源读配置。
- **前置：** 读者已了解 Transformer decoder、自回归注意力、位置编码、LayerNorm/RMSNorm、FFN、MHA/GQA 和 KV cache 的基本概念。必要内容只作短唤回；完整机制分别回链第 4、6、7、11、18、23 章。
- **离章结果：** 读者能对指定 checkpoint 比较位置表示、attention head/KV head、可见范围、dense/MoE FFN 与上下文字段；能区分公开事实、同一来源的作者主张、尚未披露和自己的推断；不会把家族标签当成固定 block，也不会从异模型 benchmark 推出结构优劣。
- **边界：** 第 16 章讲 decoder-only 为什么成为通用生成接口；本章讲版本化的模型 block 配置；第 18 章深入 RoPE/长上下文扩展，第 21—22 章负责 MoE 路由和 Dense/MoE 成本，第 23 章讨论小模型与端侧效率，第 11 章负责 KV cache 计算，第 51 章负责 GPT/LLaMA/Mistral 谱系及训练历史，第 79—95 章逐一审阅近期模型与架构。Serving/runtime、内核与硬件性能不在本章展开。

## 研究主题与章节去向

主题研究、固定版本、来源阅读位置和证据边界见[GPT、LLaMA 与 Mistral 架构比较研究记录](../../../research/gpt-llama-mistral-architecture-comparison-2026-11-20.md)。

| 外部主题 | 支持范围 | 本章处理 |
| --- | --- | --- |
| 家族标签与具体架构版本 | GPT-3、GPT-4 报告、gpt-oss 官方卡公开细节不相同；LLaMA/Mistral 版本亦变 | 17.1 先固定样本与证据种类；闭源 GPT 未披露处标为未知，不推测 |
| Decoder block 的可比坐标 | 原始论文、技术报告和模型卡公开位置表示、归一化、attention、FFN/专家等不同字段 | 17.2 统一共享主干和组件坐标；用表格对齐同名异义配置字段 |
| 位置与 FFN 选择 | LLaMA 论文报告 RoPE、SwiGLU、RMSNorm；gpt-oss 卡报告 RoPE/RMSNorm/SwiGLU MoE | 17.3 只解释会改变 block 的差别，RoPE/RMSNorm/SwiGLU 深讲回链对应章节 |
| MHA 与 GQA | GQA 原论文定义每组 query 共享 KV head，并在指定实验中比较质量/速度 | 17.4 用头数展示配置差别；不重复 KV 计算推导，不把论文性能结论推广为普遍保证 |
| dense、局部窗口与信息传播 | GPT-3、gpt-oss、Mistral 7B 论文给出各自的 dense/local pattern；Mistral 特别区分窗口、理论信息跨度和 `context_len` | 17.5 解释逐层信息可达与声明上下文不是同一个量，进一步的位置扩展回链第 18 章 |
| 性能归因与教学比较 | 研究者架构比较文章明确分组件呈现并指出训练/数据/hyperparameter 混杂 | 17.6 使用固定版本对照收束，不据跨模型 benchmark 或 serving 速度评判架构优越性 |

## 正文编号大纲

### 17.1 先问“哪一个版本”，再问“什么架构”

从读者常见的“GPT 用什么架构”问题开始：GPT-3 论文公开了一个具体基线，GPT-4 技术报告没有披露 block 细节，而 gpt-oss-20b 又有自己的公开 MoE/GQA 配置。说明“家族名”不能替代 checkpoint、版本和来源；本章选 GPT-3 作为历史参照，gpt-oss-20b、Llama 3/3.1 8B 和 Mistral 7B v0.1 作为可查实例。将产品能力、指令微调格式和内部模型结构分开。

### 17.2 用同一张 decoder block 图确定比较坐标

唤回第十六章的 causal decoder 主干，指出这些版本共享“逐层残差流、attention 子层、FFN 或专家子层”的高层轮廓，但模块的归一化放置、位置注入、Q/KV head、attention mask/window、FFN/experts 可以不同。给出供后文使用的坐标表：位置表示、norm、attention heads 与 visibility、FFN/专家、上下文相关字段。表中每项解释“字段记录什么”；不将检查坐标机械变成正文配额。

### 17.3 位置、归一化与 FFN：同一主干里的不同部件

以具体版本说明差异：GPT-2/3 公开基线用可学习绝对位置、LayerNorm 和 GELU；原始 LLaMA 报告 RMSNorm、SwiGLU 和 RoPE；Llama 3 论文记录其自身的 RoPE、SwiGLU 配置；gpt-oss 卡也公开 RMSNorm、RoPE 与 SwiGLU 专家块。Mistral 7B v0.1 固定配置记录 RMSNorm 和 RoPE 字段。说明同一 decoder 主干可替换位置表示、归一化与 FFN；不把某一代设计推广为家族定律，也不复推这些组件的数学，分别链接第 6、7、18 章。无法由所选来源确认的 GPT-4 字段保持未知。

### 17.4 MHA 与 GQA：看清 Q 头和 KV 头是否一一对应

用简短的头组关系示意 MHA、MQA、GQA：多个 query head 在 GQA 中按组共享 key/value heads；让读者读懂配置中的 `num_attention_heads` 和 `num_key_value_heads`。对齐 Llama 3 8B、Mistral 7B v0.1 与 gpt-oss-20b 已公开的具体头数，解释共享 KV 的结构动机与 cache 关联；把 GQA 完整机制和 KV cache 成本留给第 4、11 章。速度/质量只引用原研究的受控实验条件，不承诺普遍收益。

### 17.5 Dense 与滑动窗口：局部窗口不等于总上下文

沿同一 token 的信息路径比较 full causal attention、GPT-3/gpt-oss 报告的局部与全局层交替、Mistral 7B v0.1 的滑动窗口。用一个少层数的执行轨迹说明信息可经过多层传递到窗口外的位置；随后区分单层可见窗口、逐层理论感受范围、模型卡/论文的 context length、长上下文训练与模型实际可靠性。Mistral 的 4,096 窗口、论文 `context_len=8,192`、固定配置 `max_position_embeddings=32,768` 并列呈现并解释来源字段，避免把它们压成同一个“上下文长度”。完整位置扩展、RoPE scaling 与长上下文评测回链第 18 章。

### 17.6 用版本化对照表得出有限结论

将 GPT-3 175B、GPT-4 报告、gpt-oss-20b、Llama 3/3.1 8B、Mistral 7B v0.1 放入一个带来源和字段定义的对照表。表中展示架构事实、报告未披露处、上下文字段来源；不加入跨论文质量排名。解释参数量、训练数据/recipe、tokenizer、后训练、上下文扩展、软件/硬件和测量协议也影响结果，因此不能靠模型名、结构数目或 benchmark 直接归因。给读者“锁定版本—核对直接来源—对齐字段含义—区分事实与推断—检查其他变量”的复查顺序，并转入第 18 章或第 51 章。

## 表达形式

- 一张 decoder block 比较坐标表；一张版本固定且每行附来源的实例比较表。
- 可用紧凑 attention 可见范围示意或状态轨迹说明窗口外信息如何经过多层传递；若文字已清楚，不为了配额添加图。没有必要新增公式或 Python 代码。
- 交叉引用第 4、6、7、11、18、23、51 章和本册后续具体模型章；链接只指向本章当前解释需要的机制，不代替必要说明。
