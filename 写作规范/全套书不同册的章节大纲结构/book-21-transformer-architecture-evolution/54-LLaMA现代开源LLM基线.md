# 第五十四章大纲：LLaMA 系列：RoPE、RMSNorm、SwiGLU 与开源 LLM 基线

## 本章定位

本章讲 LLaMA 家族中经典稠密基线的形成、关键组件所在位置及后续版本如何扩展或改变这套基线。核心历史链为：原始 LLaMA 将 Pre-Norm/RMSNorm、RoPE 与 SwiGLU 组合到可公开研究的 decoder-only 模型；Llama 2 引入更长上下文和较大模型的 GQA；Llama 3/3.1 延续稠密主干并扩展版本与上下文；Llama 4 转向 MoE 和原生多模态，显示“LLaMA-like”不是跨版本固定模板。

本章不完整重推位置编码、归一化、GQA、MoE、长上下文或多模态机制；分别回链第 4、6、7、18、21、22 章和第十五册。第十七章负责逐版本读取具体 checkpoint 的架构字段，本章仅呈现家族层面的演进；第五十七章负责 Qwen/LLaMA 3/Gemma 的训练数据与工程成熟化。

## 54.1 原始 LLaMA 建立了怎样的稠密基线

以 2023 年原始论文为证据，说明 LLaMA 是一组 $7\mathrm{B}$—$65\mathrm{B}$ 的基础模型，使用公开可获得数据并提供给研究社区。区分“新组件发明”和“已有组件组合成可复用公开基线”；也区分公开数据、可获得权重、完整训练复现和使用许可。

## 54.2 三个常见组件改变的是不同路径

用表格简述 Pre-Norm/RMSNorm、RoPE、SwiGLU 分别处于 decoder block 的哪个位置、改变什么路径、有哪些结论边界。指出原论文援引了 GPT-3、PaLM、GPT-Neo 等先前工作，避免把组件归为 LLaMA 首创。详细机制链接第六、七、十八章；FFN 细节链接相关 FFN 专题。

## 54.3 LLaMA 系列沿用基线，也逐版调整

用版本表比较原始 LLaMA、Llama 2、Llama 3/3.1 与 Llama 3.2/3.3：只保留模型范围和结构演进。原始 LLaMA 是稠密 decoder 基线；Llama 2 将上下文扩大到 $4\mathrm{k}$ 并在较大模型中采用 GQA；Llama 3/3.1 保持稠密主干、采用 GQA，Llama 3.1 发布目录列出 $8\mathrm{B}$、$70\mathrm{B}$、$405\mathrm{B}$ 和 $128\mathrm{k}$ context；后续 3.2/3.3 扩展了小型文本与视觉版本。数据配方和训练工程回链第五十七章；逐模型精确字段回链第十七章。

## 54.4 Llama 4 已超出经典稠密配方

引用 Meta 当前模型目录和 Llama 4 官方模型卡，说明 Scout/Maverick 采用 MoE 与 early fusion 的原生多模态设计。保留官方发布字段：Scout $16$ 专家、$109\mathrm{B}$ 总参数/$17\mathrm{B}$ 激活参数、$10\mathrm{M}$ context length；Maverick $128$ 专家、$400\mathrm{B}$ 总参数/$17\mathrm{B}$ 激活参数、$1\mathrm{M}$ context length。明确这是模型卡披露，context 字段不证明长上下文任务可靠性。MoE 成本回链第二十一、二十二章，多模态概念回链第十五册。

## 54.5 把 LLaMA 当作可追溯的基线

以简短清单收束版本识别、基础/指令权重区分、dense/MoE、公开来源、上下文字段定义及许可核对。强调可下载权重不等于训练语料、完整 recipe、服务实现和无条件使用权全部开放。将结论限定为“基线可提供比较起点”，不把任何版本或组件宣称为普遍最优。
