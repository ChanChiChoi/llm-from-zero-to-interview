# 英文面试表达模板

## 自我介绍

I am an algorithm engineer focusing on large language models. My preparation covers model architecture, training, alignment, inference optimization, evaluation, and multimodal systems.

## 不确定时的回答

I am not fully certain about the exact implementation detail, but I would reason about it from the objective, data, optimization, and system constraints.

## 解释 Transformer

The core idea of Transformer is self-attention. Each token builds a contextual representation by attending to other tokens, and causal masking ensures that an autoregressive model cannot look into the future.

## 解释 RLHF

RLHF usually consists of supervised fine-tuning, reward model training from human preferences, and policy optimization with a KL constraint to keep the model close to the reference model.

## 解释 Trade-off

The main trade-off is between quality, latency, throughput, memory, and cost. A better model may improve output quality, but it can increase inference latency and serving cost.

## Behavioral interview：用真实经历说明证据

STAR 可以帮助整理顺序，但它不是需要逐字背诵的脚本。先替换方括号里的内容，再检查每个结果是否有来源、定义和个人责任边界。

> I can share a real example from [project or situation]. We were trying to [goal], and the main constraint was [constraint]. My responsibility was [specific scope]; [team or partner] owned [other scope]. I initially thought [hypothesis], so I [action]. When [evidence or feedback] showed [observation], I [decision or correction]. We measured [result and metric definition] against [baseline or denominator]. The outcome was [result], with [trade-off or limitation]. Looking back, I changed [practice or process] so that [learning or follow-up].

## 讲一次分歧

> We agreed on [shared goal], but disagreed about [specific decision]. [Partner or team] was concerned about [constraint], while I was concerned about [other constraint]. I suggested comparing [options] on [shared evidence or experiment]. The result was [what the evidence showed], so [decision owner] chose [decision]. I was directly responsible for [my contribution], while [team] owned [team outcome].

不要把分歧写成“我说服了对方”，除非这确实是发生过的过程。说明对方的合理约束、谁作出决定，以及仍未解决的风险。

## 讲一次失败或收到的反馈

> I made a wrong assumption about [specific issue]. The signal was [observable evidence], and I first [response]. I then [containment or correction] and told [relevant people] about [impact or uncertainty]. The result was [measured outcome or remaining limitation]. Since then, I have [specific change], and I check it by [follow-up evidence].

讲失败时，描述当时能够获得的信息，不要用事后知识改写自己的原始判断。把责任、系统条件、止损和复发预防分开说明。

## 贡献边界与不确定性

> The team achieved [team result]. My direct contribution was [specific work], and [other owner] was responsible for [other work]. I can verify [fact or record]. I do not remember the exact [number or detail], so I would describe it as [supported range] or check [source] before giving a precise value.

只使用本人真实、可核验并获准披露的经历。章节中的构造案例和数字不可直接填入个人回答；如果没有完全对应的经历，可以说明相近任务和迁移边界，不要把课堂练习说成生产项目。
