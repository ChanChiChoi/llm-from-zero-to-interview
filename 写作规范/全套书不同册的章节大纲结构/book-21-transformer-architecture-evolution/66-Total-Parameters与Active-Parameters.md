# 第 66 章 Total Parameters 与 Active Parameters：按统计口径读懂模型规模

本章说明模型报告中的 Total Parameters 与 Active Parameters 分别统计什么，以及为什么一个 MoE 模型可以同时具有较大的全量权重集合和较短的逐 token 专家路径。中心是把“模型参数库存”“每 token 选中的专家路径”“批次触及的专家并集”和“指定精度下的权重字节”分开；对比公开数字时记录其模型版本、来源和计数规则。本章不再展开 MoE 架构谱系、训练预算或推理服务系统：专家组合回看第 47、55 章，训练 FLOPs 与预算见第五册第 13 章。

## 66.1 两个数字回答不同问题

本节界定总参数量和激活参数量的对象差异，并说明它们是带来源口径的模型统计值。关键研究抓手包括 expert bank、shared/non-expert path、每 token 路由集合，以及模型卡中 total、activated、non-embedding 等字段的并列关系。

### 66.1.1 Total Parameters：清点模型的参数库存

说明总参数量通常统计发布口径纳入的全部专家权重、共享权重及其他可学习参数；是否计入 embedding、output head、router、shared expert、tied weights 等应按来源确认。Total 描述参数集合的规模，不代表某 token 实际执行了集合中每个权重。

### 66.1.2 Active Parameters：追踪一个 token 经过的权重路径

说明激活参数量通常指模型报告按单 token 的前向路径统计的参数规模，其中包含按口径纳入的共同路径和该 token 被选中的 expert 参数。每层 MoE 都有自己的路由集合；它不是一个固定的全局专家子集，也不等于 FLOPs、显存或请求成本。

### 66.1.3 参数量是报告口径，不是唯一物理量

区分参数个数、checkpoint tensor 布局、存储字节和运行时驻留状态。报告应说明统计口径；例如 non-embedding 与 total 是不同字段，而权重是否量化或按设备分片不会自动改变数学参数个数。

## 66.2 从一层 MoE 算到整模型

本节以逐层专家集合形式化总量和单 token 激活量的关系，避免把单层 $k/E$ 比例误当作整模型比例。研究抓手包括专家大小 $P_{\ell,e}$、每层专家数 $E_\ell$、token 路由集合 $S_\ell(x)$、共享路径及其报告口径。

### 66.2.1 Total：把全部 expert 权重计入

说明每层 expert bank 对总参数的贡献是所有专家参数量之和，再加共享参数和其他纳入统计的参数；各类别须按来源口径避免重复计数。对层数、expert 宽度或专家尺寸不同的模型，不能用单一专家数乘一个假设尺寸替代逐层求和。

### 66.2.2 Active：按每层路由集合累加

说明对 token $x$，每个 MoE 层由 router 从该层专家中选择 $S_\ell(x)$；选中专家参数与报告口径纳入的共同路径构成激活计数。不同 token 及同一 token 的不同层可以选择不同专家；若专家尺寸不等，需按实际选中集合求和。

### 66.2.3 一个等大小专家模型的总量与逐 token 计数

用明确标为教学设定的相同尺寸专家模型连接总量、激活量和路径比例；列出一 token 在多层中的不同路由集合，使读者看出 top-k 是每层应用。该例子用于验算计数关系，不是具体模型报告值。

### 66.2.4 k/E 只适用于等大小的 expert bank

推导在每层专家尺寸相同、固定选中 $k$ 个且只看专家库时，选中部分占专家库 $k/E$；再指出共同路径等其他参数会改变全模型 Active/Total 比例。层间 top-k 或专家大小不同则按实际数量计算。

## 66.3 每 token 活跃参数、请求内并集与存储字节

本节区分单 token 路径、批次/请求路由集合以及权重载荷。即使每个 token 只选少数专家，一组 token 的 route union 也可能较大；它仍不能替代逐 token active count 或设备内存统计。

### 66.3.1 路由集合按层、按 token 变化

说明 router 读取 token 的隐藏表示并为当前层专家打分，所选集合在层与 token 之间可以变化。应以每层集合 $S_\ell(x)$ 表示具体路径，而不要称“每个 token 只激活同样的两个全局专家”。

### 66.3.2 一批 token 的专家并集可以更大

定义一个批次在第 $\ell$ 层触及的专家并集 $U_\ell(B)=\bigcup_{x\in B}S_\ell(x)$，并说明 route union 描述批内至少一次被选中的专家权重集合，不是每个 token 都执行的权重，也不决定这些权重驻留在哪张设备上。

### 66.3.3 参数个数乘精度字节只估原始权重载荷

给出在所有被计参数采用同一存储精度时的原始载荷估算关系 $M_{\mathrm{raw}}\approx N b$。说明低比特 scale/zero-point、padding、混合精度、共享/tied weights、分片和运行时布局会改变实际文件/显存账本；运行时 tensor 重排不等于模型参数个数改变。

## 66.4 用公开模型看数字，不混成同一口径

本节以论文和官方模型卡中的明确披露展示 total/active 的用法，并把每个数字与其来源保持绑定。案例只说明计数如何呈现；不据其路由比例反推未披露参数构成，也不把不同发布方的统计值视为已统一审计。

### 66.4.1 Mixtral：top-2 不等于整模型的 active 比例

根据 Mixtral 论文报告 47B total、13B active parameters；说明其每层含 8 个 feed-forward experts、每 token 每层选两个。指出论文的整模型报告值是该模型口径下的近似统计，不能只按 2/8 精确推出 13B，也不在本章重复其性能基准分析。

### 66.4.2 DeepSeek-V3：共享 expert 与 routed experts 都要看

根据 DeepSeek-V3 技术报告 v2 记录 671B total、37B activated/token；报告还说明前三层之外采用 MoE、MoE 层包含一个 shared expert 与 256 个 routed experts、每 token 选 8 个 routed experts。强调报告的整模型 activated value 不能简单用 8/256 推得。

### 66.4.3 Qwen3：同一模型卡并列 total、activated 与 non-embedding

根据 Qwen3-30B-A3B 官方模型卡记录 30.5B total、3.3B activated、29.9B non-embedding，以及 128 个 experts/8 个 activated experts。说明模型名中的约数、模型卡字段与 architecture config 不是可任意相减的完整账本；记录页面访问时间和模型 revision。

### 66.4.4 发布数字留在各自版本与定义下比较

归纳跨来源对比最低条件：固定 model ID/revision、报告版本、参数统计字段定义、embedding/head/router/shared 的纳入范围、每 token 或其他激活范围及四舍五入。字段缺失时保留 unknown，不用名称或专家比例补算。

## 66.5 参数数字不等于算力、内存、延迟和质量

本节说明这些统计值可提供什么第一层线索，以及哪些结论仍需单独计算或实测。重点是保持 FLOPs、权重载荷、运行状态、服务表现与模型质量的分母分离；训练预算详细计算由第五册第 13 章负责。

### 66.5.1 Active Parameters 和 FLOPs 有关联但不是同一量

在专家尺寸相同、每 token 走相似矩阵形状等条件下，选中参数数目可近似对应专家矩阵乘的计算规模；Stanford MS&E 319 讲义给出等尺寸 expert 情形下 EP 总参数、kP 单 token active 参数及相应近似 FLOPs。说明 attention、router、shared layers、token 数、矩阵形状和 FLOP 计数约定都会影响整模型计算量，因此 active count 不是 FLOPs 公式。

### 66.5.2 Total Parameters 和 GPU 显存不是同一量

区分模型总参数个数、压缩/量化后的 checkpoint 权重字节、分片后每设备常驻权重与整次运行的 GPU memory。原始权重载荷只可在指定精度下用参数量乘每参数字节数估算；scale、布局、workspace、activation、KV/state 与复制需另计。详细 cache 资源问题由相关缓存及推理章节负责。

### 66.5.3 延迟要看路由执行和 workload

说明 top-k 只界定选中专家数，端到端时延还依赖路由实现、专家张量布局、dispatch/聚合、通信、batch 与 prefill/decode 形状。Transformers 团队关于 WeightConverter 的文章可作为 checkpoint 专家张量到运行时打包布局的实现例子，但其具体处理不代表所有 runtime；不从 active 参数直接预测 tokens/s 或 p99。

### 66.5.4 总参数不能单独预测模型质量

说明参数总量和单 token active count 都不足以单独给出质量结论。可用 2026 年 arXiv v1 MoE 配置研究作为有限案例：超过 2,000 次、总参数最高 6.6B、active 参数规模约 10M–300M 的设置中，作者报告在固定 active FFN 参数等实验条件下增加 inactive expert 参数可改善其 held-out loss；结果不可推广为 frontier 规模或普遍质量定律。

## 66.6 审核并呈现模型参数数字

本节把读者的结果收束到可复核的模型报告。关键不是补猜所有未知数字，而是清楚标出来源、统计范围、每 token 激活定义和另外需要测量的资源指标。

### 66.6.1 先记录来源、模型 revision 与计入范围

建议核对论文/model card/config 的精确版本与日期，查看 total/active/non-embedding 分别如何定义，并核对共享参数、所有专家、router、embedding、output head 和 tied weights 是否计入。对动态网页保留访问日期；对模型权重/配置使用固定 revision。未知项明确标为 unknown。

### 66.6.2 用分栏报告比较，不从缺失字段反推

采用分栏记录模型 ID/revision、来源版本、total parameters、active parameters per token、计数包含范围、experts 与 top-k、non-embedding 口径、precision/quantization、估计权重字节或实测显存，以及明确绑定 workload 的 FLOPs/延迟/质量结果。面试回答先说两个参数数字回答不同问题，再给路由和计数规则，最后把内存、计算、速度与质量作为独立指标；没有证据的字段保持未知。
