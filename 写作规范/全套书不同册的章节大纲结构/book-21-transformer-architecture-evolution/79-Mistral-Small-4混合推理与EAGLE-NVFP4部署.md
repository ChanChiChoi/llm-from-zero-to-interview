# 第七十九章 Mistral Small 4：混合推理与 EAGLE/NVFP4 部署

本章以 Mistral Small 4 为模型级案例，连接公开的 MoE/MLA 结构、请求级 reasoning effort、EAGLE 推测解码草稿头和 NVFP4 量化权重。读者需要区分模型本体、请求配置、辅助草稿权重、量化检查点和 serving runtime；能力合并的发布主张不等于已公开训练配方，模型卡列出的部署选项也不等于它们已被证明可以组合。章节结构与性能数字均固定到可复核的模型配置、实现或报告版本，并把 API 和推理引擎的完整运维交给第六、二十四册。

## 79.1 一份模型发布包含模型、请求配置与推理制品

本节厘清 Mistral Small 4 发布中的对象层次：一个开放权重模型声称整合 Instruct、Reasoning/Magistral 与 Devstral 能力；`reasoning_effort` 是请求可设置的推理模式；EAGLE head 和 NVFP4 则是与模型部署相关的独立制品。中心是读者不能仅凭“混合”或“统一”这些产品描述推断模型内部按任务切换了哪些网络，也不能把 API 参数、权重结构和服务运行时当成同一种设计。

### 79.1.1 Instruct、Reasoning 与 Devstral 是发布方的能力合并主张

说明 Mistral 官方如何描述三条产品/模型路线进入 Small 4，并把公开事实限定为“一个模型提供这些使用能力”。具体训练数据、后训练方法、是否存在专家专门化以及能力如何映射到网络内部均未由发布材料完整披露。

### 79.1.2 `reasoning_effort` 改变请求行为，不代表另一套权重

说明官方资料公开的 `none` 和 `high` 两种请求设置及其用途描述；它们是同一模型的推理配置入口，不能与另一模型、另一套权重或跨供应商统一等级等同。公开页面没有给出该参数触发的内部路由、计算图或 token 预算机制。

### 79.1.3 模型结构、检查点、解码方法和服务层属于不同对象

界定本章的模型配置、EAGLE draft/head、NVFP4 检查点、目标模型验证和运行时之间的关系。明确同一系统可能同时包含这些对象，但每个组合都需要固定 artifact 与 runtime 证据；协议解析、调度与线上 API 行为分别回链第六册和第二十四册。

## 79.2 固定配置显示 Mistral4 文本主干与多模态封装

本节将官方模型卡、指定 revision 的配置文件和 Transformers 固定实现分层对读，重建足以支持架构分析的公开结构。重点包括外层图像/文本条件生成封装、36 层 Mistral4 文本解码器、MLA 的低秩表示和 MoE 的路由/共享专家；模型配置能证明公开实现字段，不自动证明未公开的训练过程或产品能力。

### 79.2.1 顶层 Mistral3 条件生成封装内部使用 Mistral4 文本配置

解释 `Mistral3ForConditionalGeneration`、`model_type=mistral3` 和 `text_config.model_type=mistral4` 分属多模态外层与文本主干的公开配置字段。文本/图像输入与文本输出的产品能力以模型卡为依据，不从外层类名推出未披露的图像编码器细节。

### 79.2.2 36 层 MLA 用低秩 latent 缓存和分离的 RoPE key

依据固定配置与模型实现说明 `kv_lora_rank=256`、`q_lora_rank=1024`、`qk_nope_head_dim=64`、`qk_rope_head_dim=64` 及 value/head 维度的关系。把 MLA 的压缩缓存表示与随后为注意力计算展开成逐头 K/V 的实现区分开，提供后续 cache 估算的研究抓手。

### 79.2.3 四路 routed top-k 与 shared expert 不能混为一个数字

依据 `n_routed_experts=128`、`num_experts_per_tok=4` 和 `n_shared_experts=1` 及实现路径，说明 token 选择的 routed experts 与共享 MLP 的不同职责。模型卡的“128 experts、4 active”与配置的 shared expert 字段需要分别列账，不能据标题数字直接还原完整激活参数计算。

### 79.2.4 公布的 256K 上下文不由位置配置字段单独决定

说明官方模型卡/产品页把上下文窗口标为 256K，而固定配置中的 `max_position_embeddings` 是另一字段。可用位置扩展配置不等同于经过质量验证的有效上下文或所有 serving backend 支持的窗口。

## 79.3 参数、KV 状态和模型窗口需要分开估算

本节把总参数、每 token 路由计算、MLA KV cache 与上下文长度放入不同资源账本。通过配置及固定实现推得 cache 的元素数和理论字节数，但明确这些是指定 dtype/实现下的静态估算，不能代替完整推理显存或端到端延迟测量。

### 79.3.1 6B、6.5B 与包含嵌入/输出的 8B 有不同公开口径

对照 Mistral 发布文章、官方模型卡和模型文档的激活参数表述，记录其口径并未由公开材料精确对账。说明总权重、路由专家、共享专家、嵌入/输出层及每 token 计算不可仅凭一个 headline number 相互替代。

### 79.3.2 压缩 KV cache 的指定实现每层每 token 存 320 个 BF16 数值

由 256 维 KV latent 与 64 维共享 RoPE key 推出每层每 token 的缓存元素量，并结合 36 层和 BF16 得到约 22.5 KiB/token 的理论账本。对照具有 32 个 KV heads、每头 128 维的显式 K/V 账本时，明确这是表示与实现对照，不是跨引擎实测倍率。

### 79.3.3 256K 单序列只计上述 cache 已约为 5.625 GiB

将每 token 的实现级缓存估算延伸到 262,144 token，说明长上下文依然占用多 GiB 状态。该数值未计权重、图像 token 的扩展方式、page/allocator 对齐、并发 batch、临时激活、draft 状态和 runtime workspace，因此不是 GPU 最低显存承诺。

## 79.4 `reasoning_effort` 是请求级控制面，效果要按工作负载评估

本节将两个公开模式视为可测的请求配置，分析它们可能改变输出行为与测试时资源，但不推测未披露的模型内部执行策略。中心是模式名称不构成质量保证或跨模型刻度；评测需要对齐任务、模型版本、采样和工具条件，并报告质量与生成代价。

### 79.4.1 `none` 与 `high` 是同一模型公开的两个模式值

记录 Mistral 对两值用途、输出风格的描述以及模型卡给出的推荐采样条件。将这些描述标记为发布方说明，而不是对不同数据集上质量、延迟或 token 数的独立保证。

### 79.4.2 测试时计算的具体内部机制仍属未披露范围

区分用户可观察的推理长度/输出行为与潜在内部计算。模型卡提到按需 reasoning/test-time compute，但不公开路由专家、额外前向、token 上限或具体停止算法；不得从 `high` 名称还原计算图。

### 79.4.3 固定任务后同时测答案质量与生成成本

将 `none/high` 对照绑定相同任务集、模型 revision、提示词、工具/环境、采样条件和最大输出预算；按任务报告成功率或正确率、输出 token、TTFT、TPOT、p95 和单位成功成本。解释总完成时间与单 token 解码延迟不能互相代替。

## 79.5 EAGLE 把特征级草稿与目标模型验证接成一轮

本节从目标模型逐 token 自回归造成的解码等待出发，解释推测解码的 proposer/verifier 分工以及 EAGLE 原论文的特征级草稿机制。知识范围覆盖 one-step-ahead token 如何帮助预测特征、树状候选如何并行验证、接受/拒绝校正怎样维持目标分布，以及接受长度与 draft/verify 成本的共同作用。

### 79.5.1 草稿只提出候选，输出权仍由目标模型掌握

区分 target model 的目标分布与 EAGLE head 产生的 draft 分布。介绍每轮候选、目标模型批量计算条件分布、验证器决定接受前缀及拒绝时修正 token 的职责，不把 speculative decoding 描述为用小头替换目标模型。

### 79.5.2 原始 EAGLE 在第二顶部特征上自回归并输入前移 token

依据 EAGLE 论文区分 token 级草稿、Medusa 式并行头与特征级 EAGLE。说明原始 EAGLE 逐步预测目标模型 LM head 之前的特征，再利用前移一个时间步的 token 作为离散条件；该机制说明 EAGLE 家族的核心设计，不自动确定 Mistral artifact 属于哪个代际。

### 79.5.3 Tree Attention 让目标模型一次读取候选树的条件分布

说明树上不同候选共享前缀但不能互相看到错误分支；树 attention mask 保持候选路径的因果关系，使目标模型能在一个验证前向中计算多分支概率。draft 树深度、宽度和 batch 会改变一次验证的计算量。

### 79.5.4 接受概率和残差分布解释为何拒绝后仍保持目标分布

围绕 draft 分布 $q$ 与 target 分布 $p$，说明候选接受与拒绝校正如何重新分配概率质量、两种结果如何合成后保持目标分布，并将结论限定在相同条件分布和正确校正规则下。

### 79.5.5 接受的前缀被提交，拒绝分支及其缓存被丢弃

说明 target verification 如何确定最长可接受前缀，拒绝时的替换 token、全接受后的 bonus token 和候选后缀作废怎样影响 KV 状态；具体 cache 回滚布局由 serving backend 决定，错误分支不能作为已确认上下文继续生成。

### 79.5.6 加速取决于接受长度和整轮计算，而非草稿 token 数

关联平均接受草稿数、draft latency、target verification、tree expansion、cache 工作量与 batch 中其他请求。高并发或长上下文可能改变可用算力和带宽比例；论文报告的速度比只能保留在其模型、任务、精度、硬件和 serving 设置下。

## 79.6 Mistral EAGLE artifact 的名称不足以确认代际和配对能力

本节检查 Mistral 官方模型卡链接的 EAGLE checkpoint，并将其作为该模型相关的训练草稿制品记录。其公开名称和用途可确认存在训练好的 head，但不能据此判断是 EAGLE-1/2/3、具体树结构、训练数据、推测接受率或与任意量化目标的兼容性。

### 79.6.1 官方链接指向一个附属于 Mistral Small 4 的草稿制品

记录基础模型卡与 EAGLE checkpoint 的 model id、访问时固定 revision 和官方部署说明。区分 target model 的完整参数与额外 draft head，避免把 EAGLE repo 当作独立通用聊天模型。

### 79.6.2 model card 未标注 EAGLE 代际或 Small 4 专属性能表

对照 EAGLE 原论文与 EAGLE-3 论文，指出 Mistral 卡只称 trained eagle head，未明确方法代际、树配置、温度范围或单独 benchmark。不得把论文在 LLaMA/Vicuna 等模型上的倍率移植为 Small 4 结果。

### 79.6.3 head 与目标权重或 NVFP4 checkpoint 的适配必须单独核验

说明 draft 隐状态来自特定目标模型，权重量化会改变特征和目标分布；即使校正算法对当前 verifier 分布保持精确，也不意味着输出分布仍等于未量化版本。公开资料没有确认 Mistral EAGLE head 与 NVFP4 checkpoint 的组合结果。

## 79.7 NVFP4 把低位 payload 与尺度层级一起存储

本节解释 NVFP4 一般格式、Mistral 发布的量化检查点和存储/运行效果之间的边界。正文说明 E2M1 小数值如何经 block scale 和 tensor scale 解码，并把尺度元数据、可能保留高精度的张量、量化误差及硬件 kernel 纳入部署账本；不将 NVIDIA 通用 recipe 未经证实地套在 Mistral checkpoint 的每个权重上。

### 79.7.1 E2M1 的 4-bit 编码依赖 E4M3 block scale 与 FP32 tensor scale

依据 NVIDIA 格式文档说明 NVFP4 payload、每 16 元素共享 E4M3 scale 与全 tensor FP32 scale 的层级表示，并指出 E2M1 有限动态范围。区分这一通用一维 recipe 与权重可能使用的二维 scale 设计。

### 79.7.2 Mistral 卡公开 checkpoint 来源但没有完整逐层量化清单

记录 Mistral NVFP4 model card 将其描述为由 `llm-compressor` 与 vLLM/Red Hat 合作制得的 post-training quantized 变体。说明在所读卡片中未找到完整层排除表、校准数据、精确 scale layout、激活/KV 状态策略及对照质量结果；不把格式定义当成该 checkpoint 全部实现细节。

### 79.7.3 4-bit payload 下界不等于 NVFP4 artifact 或 GPU 占用

在明确十进制 GB 和“119B 参数全部仅计 4-bit payload”的假设下，估计权重 raw payload 下界，并与增加 scales、打包/对齐、未量化层、KV 状态、路由缓冲区及 workspace 后的部署账本区分。官方内存估算和 Hub 文件体积须说明各自的测量口径。

### 79.7.4 低精度能否变快取决于硬件、kernel 与工作负载

说明 NVFP4 的内存/带宽机会和低精度矩阵乘所需硬件支持；若缺少匹配 kernel、发生转换/回退或 workload 不受限于相应瓶颈，较小 artifact 不自动产生更高 tok/s。量化质量按文本、代码、推理、图像、长上下文分别检验。

## 79.8 部署评估须将模式、draft、量化和服务条件分开

本节把模型能力、EAGLE 和 NVFP4 转化为可比较的测试组合，并明确哪些模型卡数据支持什么。中心是先固定 target/revision、模式与 runtime，再逐步加入已确认兼容的草稿头或量化 checkpoint；发布方对 Small 3 的端到端比较不能被改写为某一组件的单独贡献或通用加速保证。

### 79.8.1 先建立目标 checkpoint 基线，再加入已受支持的单项制品

列出基线、reasoning mode、NVFP4 单项、EAGLE 单项及“组合仅在 runtime/artifact 明确支持时测试”的对照逻辑。每一条记录 target 与 draft checkpoint revision、量化状态、runtime/backend、GPU、并行方案、上下文与 batch，避免只写模型产品名。

### 79.8.2 质量门同时测任务结果和推测校正行为

按任务选择正确率、代码测试、图像理解/文档问答、长上下文检索及输出格式指标；另测 draft acceptance、accept length、fallback、目标调用数、拒绝后的输出/状态正确性。区分 target 分布保持和量化相对原权重的质量变化。

### 79.8.3 速度与资源报告要区分 prefill、decode 和并发区间

报告 TTFT、TPOT、端到端 latency、p50/p95、吞吐、有效输出 token、显存、KV cache、带宽与不同 batch/上下文条件。NVFP4 可能作用于权重搬运和计算，EAGLE 改变解码候选/验证工作；两种机制在同一 workload 中的净收益由实测决定。

### 79.8.4 官方 40% latency 与 3 倍吞吐是独立设置下的模型比较

将 Mistral Small 4 对 Small 3 的官方 latency-optimized 与 throughput-optimized 发布数字限定为发布方描述的模型级比较。没有相同基线、硬件、量化/草稿配置和测量协议时，不能把 headline 数字归因于 EAGLE、NVFP4、reasoning effort 或某一 MLA 字段。
