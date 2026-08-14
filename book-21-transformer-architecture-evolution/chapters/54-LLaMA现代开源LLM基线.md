# 第 54 章 LLaMA 系列：现代开源 LLM 的架构基线

## 54.1 LLaMA 的历史位置

LLaMA 的影响不只在于发布了若干参数规模的模型，更在于它把一套相对简洁、高效、可复现的 decoder-only 配方带入开源生态。研究者和工程师可以在公开权重、公开论文和成熟工具上继续训练、微调、量化和部署。

LLaMA-like 并不是严格标准，而是一个经验标签，通常指 decoder-only、RoPE、RMSNorm、SwiGLU、Pre-Norm 和高质量数据等组合。不同版本的词表、层数、上下文、训练数据、指令配方和许可都可能不同，不能只凭“LLaMA-like”推断全部细节。

## 54.2 现代 decoder block 的组合

一个教学化的 Pre-Norm block 可以写成：

~~~math
\tilde h^{(l)}
=h^{(l-1)}
+\mathrm{Attention}
\left(\mathrm{Norm}(h^{(l-1)})\right)
~~~

~~~math
h^{(l)}
=\tilde h^{(l)}
+\mathrm{FFN}
\left(\mathrm{Norm}(\tilde h^{(l)})\right)
~~~

RMSNorm 可以写成：

~~~math
\mathrm{RMSNorm}(x)
=\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}
\odot g
~~~

它不减去均值，计算更简单。Pre-Norm 把归一化放在子层输入附近，通常有利于深层训练的梯度路径；具体稳定性还取决于初始化、学习率、残差缩放和训练配方。

## 54.3 SwiGLU 为什么成为常见 FFN

普通 FFN 是两个线性投影和非线性：

~~~math
\mathrm{FFN}(x)=W_2\sigma(W_1x)
~~~

SwiGLU 引入门控支路：

~~~math
\mathrm{SwiGLU}(x)
=W_o\left(\mathrm{SiLU}(W_gx)
\odot W_ux\right)
~~~

门控让一条支路调节另一条支路的信息通过程度。它增加了投影结构和参数，需要调整中间维度，不能只把普通 FFN 的 hidden size 原样替换。

SwiGLU 的价值来自多种因素的组合：非线性形状、门控表达、参数预算和现代训练 recipe。仅凭某个模块名称不能保证模型一定更强，需在同参数、同 token 和同训练条件下消融。

## 54.4 RoPE 与相对位置信息

RoPE 将 query 和 key 的二维子空间按位置旋转：

~~~math
q'_p=R(p)q_p,\qquad
k'_r=R(r)k_r
~~~

点积中的相位关系携带位置差异。它与 learned absolute embedding 的工程路径不同，扩展到更长上下文时也需要考虑频率、训练长度、数据和 attention pattern。

RoPE 并不等于“天然支持无限上下文”。如果模型只在短序列训练，长位置的外推、lost-in-the-middle、精确引用和 cache offset 仍需专门测试。

## 54.5 数据和 tokenizer 是基线的一半

开源模型的可用性高度依赖数据工程。数据量、质量、语言覆盖、代码比例、重复清理、污染控制和文档配比都会影响最终能力。不同 tokenizer 的 token 数不能直接比较，中文、代码、数字和多语言的压缩率可能相差很大。

一个训练数据版本至少要绑定：

~~~text
dataset_revision
source_mix
dedup_rule
quality_filter
tokenizer_revision
sampling_recipe
eval_exclusion
~~~

如果这些信息缺失，用户很难复现能力或解释回归。开放权重不等于完整开放数据，也不等于许可和商用边界没有限制。

## 54.6 参数规模和推理成本

dense decoder 每层都激活全部参数。粗略的权重内存可写成：

~~~math
M_{\mathrm{weight}}\approx N b_w
~~~

其中 N 是参数量，b_w 是每个参数的字节数。推理还要加入 KV cache、激活、临时 buffer 和框架开销：

~~~math
M_{\mathrm{total}}
\approx M_{\mathrm{weight}}
+M_{\mathrm{KV}}
+M_{\mathrm{activation}}
+M_{\mathrm{workspace}}
~~~

因此一个 70B 模型能否部署，不只取决于显卡能否装下权重，还取决于 batch、上下文、并发、量化和 p99。

## 54.7 开源基线如何被复用

LLaMA-like 基线的工程价值在于模块接口稳定。研究者可以在其上替换 attention、加入 MoE、扩展上下文、做 domain continued pretraining、SFT、DPO、量化和推理优化。

但改造时要区分：

1. 权重兼容：新 block 是否能读取旧 checkpoint。
2. tokenizer 兼容：词表和 special token 是否一致。
3. position 兼容：position id、RoPE 和 cache offset 是否一致。
4. template 兼容：训练、评估、线上请求是否使用同一角色格式。
5. 评估兼容：改动后是否与原 baseline 使用同一解码和数据。

## 54.8 一个最小的 decoder block 骨架

下面的 PyTorch 代码只展示 Pre-Norm、残差和 gated FFN 的组织方式，省略真实 attention、并行和初始化细节：

~~~python
import torch
from torch import nn


class TinyBlock(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden)
        self.norm2 = nn.RMSNorm(hidden)
        self.attn = nn.Linear(hidden, hidden, bias=False)
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        gated = torch.nn.functional.silu(self.gate(self.norm2(x)))
        x = x + self.down(gated * self.up(self.norm2(x)))
        return x
~~~

这个骨架不是 LLaMA 的完整实现，不能用于推断真实模型的 head、位置、并行和权重布局。教学重点是理解现代 block 的残差、归一化和门控 FFN 边界。

## 54.9 LLaMA 2 和后续演进的差异

LLaMA 2 论文公开了预训练与 chat 模型的训练和安全细节，说明现代基线不只是一个 block 配置，也包含数据、SFT、偏好、拒答和评测。后续开源模型在上下文、GQA、数据规模、多语言、代码、工具和多模态上继续变化。

因此把“LLaMA 架构”当成一个固定版本会掩盖演进。阅读具体模型时，应查看官方 model card 和技术报告的模型配置，而不是只看社区代码中的默认参数。

## 54.10 许可、权重和可复现边界

开源、开放权重、可下载和可商用不是同义词。一个模型可能提供权重，却限制用途、再分发或品牌使用；训练数据也可能没有公开。生产采用前要查许可证、acceptable use policy、依赖模型和权重来源。

可复现性也有层次：

1. 能加载权重并生成文本。
2. 能复现官方推理结果。
3. 能复现微调和评测。
4. 能复现预训练数据和训练曲线。

多数开放权重模型只覆盖前两层或前三层，不应把它们写成完整开源训练项目。

## 54.11 评估现代开源基线

评估需要同时看能力和可部署性。能力侧包括通用知识、代码、数学、多语言、指令遵循、长上下文和安全；系统侧包括权重内存、KV 内存、TTFT、TPOT、吞吐、p99、量化误差和 batch 扩展。

比较两个模型时，至少固定：

~~~text
model_revision
tokenizer
chat_template
prompt
sampling
max_new_tokens
hardware
runtime
quantization
~~~

否则“模型 A 更快”可能只是使用了不同模板、短 prompt 或更激进的量化。

## 54.12 常见误区

第一，认为 LLaMA-like 就是完整相同架构。第二，把开放权重写成开放数据。第三，只看参数量和 benchmark，不看 tokenizer、训练 token 和推理显存。第四，改了 RoPE 或模板却仍直接复用旧评估。第五，把 chat checkpoint 当作 base checkpoint 继续训练而不检查模板和 loss mask。第六，忽略许可证。

## 54.13 开源基线的价值在可复现链路

一个可用的开源基线不只是下载权重。需要同时固定 tokenizer、config、chat template、数据版本、训练脚本、量化方法、评测 harness 和许可证。任何一项变化都可能让“同一个模型”的结果发生回归。

模型卡中的参数量和 benchmark 还要绑定 revision、prompt、shot、上下文长度、推理精度和硬件。开放权重使结构和行为更容易复核，却不意味着训练数据、清洗规则和完整系统都公开。

## 54.14 现代 block 的资源传导

RMSNorm、SwiGLU、RoPE、GQA 和 Pre-Norm 不是互相独立的清单。SwiGLU 改变 FFN 中间维度和权重计算，GQA 改变 KV 读写，RoPE 影响位置和 cache offset，Pre-Norm 与初始化影响训练稳定。替换一个模块后，参数、激活、吞吐和质量都可能变化。

应使用参数和资源账本：

~~~math
C_{\mathrm{block}}
=C_{\mathrm{attention}}(H_q,H_{\mathrm{kv}})
+C_{\mathrm{FFN}}(d_{\mathrm{ff}})
+C_{\mathrm{norm}}+C_{\mathrm{projection}}
~~~

这比说“现代架构更高效”更具体，也便于在同预算消融中定位收益。

## 54.15 LLaMA-like 不是终点

LLaMA 配方成为开源基线后，后续模型继续在 MoE、MLA、长上下文、混合 state、训练数据、后训练和推理优化上变化。基线的意义是提供可比较的起点，而不是规定未来结构。

面对新模型时，可以先问它相对该基线改变了什么：是降低 KV，增加条件容量，改善数据和训练，还是只改变 serving kernel。每个变化都要对应任务质量、资源和证据边界。

## 54.16 开源基线的价值是可复现的配置空间

LLaMA 风格基线的重要性不只在模型名，而在于提供了可公开检查的 decoder-only 配置、tokenizer、训练与评估路线。复现时要绑定权重、代码、tokenizer、chat template、数据和 revision；“同一个 7B”可能因 tokenizer、上下文、量化或后训练不同而不是同一实验对象。

## 54.17 开放权重不等于开放训练事实

公开权重可以让研究者测试推理和微调，但训练数据配比、清洗、污染、完整计算、失败实验和内部安全策略可能仍未公开。阅读资料时应把事实分成三层：

1. **可直接检查的 artifact**：权重文件、config、tokenizer、代码、许可证和公开推理接口。
2. **发布方报告的训练事实**：数据规模、训练 token、过滤规则、硬件和阶段配方；它们需要对应技术报告或模型卡，不能从权重反推。
3. **本书的行为推断**：通过 probing、消融或服务压测观察到的能力，只能绑定模型 revision、任务和 harness。

例如从 checkpoint 可以测出某个词表和 hidden size，却不能仅凭参数张量推断训练数据没有污染，也不能把社区对某个 layer schedule 的逆向猜测写成官方结构。反过来，训练报告披露了 data mixture，也不代表读者已经获得数据清洗脚本和完整失败样本。开放权重主要开放了可执行对象，不自动开放完整因果解释。

对基线做复现时，建议保存一张 manifest：`weight_revision`、`config_revision`、`tokenizer_revision`、`template_revision`、`data_revision`、`eval_revision`、`license` 和 `unknowns`。当能力回归时，先判断是权重、模板、数据还是评估变化，再讨论模型架构。这样“开放”才会转化为可复查的工程事实，而不是宣传标签。

## 54.18 基线如何进入工程决策

选开源基线时同时看质量、许可证、显存、量化、工具模板、长上下文、社区 runtime、漏洞响应和回滚。一个分数稍低但协议和 kernel 成熟的模型，可能比理论更强却无法稳定部署的模型更适合作为生产基线。

## 54.19 资料范围与基线判断

LLaMA 论文和 LLaMA 2 论文是理解现代开源 decoder 基线的主要原始资料。具体版本的层数、上下文、词表、GQA、数据和许可证应以对应模型卡为准。

LLaMA 的真正遗产，是一套足够简洁、性能强、生态广的共同基线。研究价值不在于背出一个固定配置，而在于能从 block、数据、训练、权重许可和 serving 全链路解释一个开源模型。

## 54.20 面试问题与练习

**问：为什么 LLaMA-like 模型常用 RMSNorm、RoPE 和 SwiGLU？**

它们分别对应归一化稳定性、相对位置表达和门控 FFN 表达；但效果依赖整体训练配方，不能把模块名称当成单独的性能保证。

**问：开放权重模型和开源模型有什么区别？**

开放权重只说明权重可获得，数据、训练代码、许可证、商用限制和完整可复现性可能仍未开放。

**练习：**设计一个开源模型审计表，覆盖配置、tokenizer、数据、许可证、评测、量化和服务 SLO。

资料入口：

- LLaMA: https://arxiv.org/abs/2302.13971
- Llama 2: https://arxiv.org/abs/2307.09288
- RMSNorm: https://arxiv.org/abs/1910.07467
- SwiGLU: https://arxiv.org/abs/2002.05202

## 54.21 基线消融与选型实验

选型时可以把 LLaMA-like block 作为 baseline，分别替换 tokenizer、GQA、FFN、RoPE、量化、后训练和 serving engine，但每次只改变一个主要变量。记录参数、KV bytes、训练/推理吞吐、短任务、长上下文、工具协议和单位成功成本，才能知道收益来自结构还是环境。

## 54.22 Baseline manifest 与未知事实

一个可复用的 LLaMA-like baseline 不只是权重文件，还应有 config、tokenizer、chat template、position、norm、FFN、数据版本、训练 checkpoint、许可证、量化格式、runtime 和评估脚本。manifest 中若某字段没有公开证据，应明确写 `unknown`，不能用社区猜测填空。

做结构消融时，每次只改变一个主要因素，并保留原始权重、转换脚本、golden logits 和服务协议测试。这样才能区分收益来自 GQA、RoPE、SwiGLU、数据或后训练。开放权重的可复现优势很大，但它不等于训练过程、数据清洗和安全策略完全开放。

## 54.23 基线比较的阶段性结论

LLaMA-like 模型的价值在于提供了可复核的共同基线，而不是一组永远最优的模块名称。选择开源基线时要把 block、数据、后训练、许可证、tokenizer、runtime 和服务 SLO 一起比较；开放权重也不等于训练事实全部公开。

## 54.24 Baseline 的可比性

两个都叫 LLaMA-like 的模型可能在 tokenizer、词表、RoPE scaling、GQA、训练长度、数据混合、后训练和许可证上完全不同。比较前应先冻结接口与评测条件，再列出差异；否则“基线更强”可能只是数据、模板或推理预算不同。

至少保存参数量口径、层数、hidden、FFN、Q/KV heads、context、tokenizer、template、dtype、量化、adapter 和 engine。total parameters、active parameters、显存与实际 tokens/s 不能互相替代。

## 54.25 开放权重的复现边界

开放权重让读者可以检查 config、权重、tokenizer 和部分代码，却不自动公开数据清洗、训练中断、过滤规则、RL 轨迹、人工偏好和安全策略。复现一个模型应区分结构复现、训练 recipe 复现、行为复现和生产服务复现。

如果只能拿到权重和模型卡，能验证的通常是结构加载、前向输出和特定 checkpoint 的行为；不能把社区逆向的训练细节写成事实。许可证和商用限制也属于工程能力边界，不能只看 benchmark。

## 54.26 LLaMA-like 基线的现代扩展

现代开源模型往往在 LLaMA-like decoder block 上加入 GQA/MLA、长上下文位置方案、MoE、量化感知训练、工具后训练和推理优化。分析时应问这些变化解决了什么瓶颈：KV、训练计算、专家容量、上下文外推、协议一致性还是成本。

一个变化只有在相同任务和预算下带来可复现收益，才适合写成工程结论。否则保留为结构观察，并标明它可能增加的 cache、通信、迁移和许可证成本。
