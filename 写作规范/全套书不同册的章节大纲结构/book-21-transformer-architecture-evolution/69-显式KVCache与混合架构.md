# 第 69 章 显式 KV Cache 在混合架构中的角色

本章说明混合序列架构的历史表示为何按层而异：全局 attention 可能保存逐 token 的 K/V 或其 latent，局部 attention 只保留可见窗口，递归路径则更新状态。读者需要能把层表映射为显式 KV 的 payload 公式，区分层间交错、层内混合、局部/全局/稀疏读取与表示压缩各自改变什么。第十一章负责常规 KV cache 机制与基线容量，第六十五章比较历史读取路径，第六十三章和第七十章负责 MLA/latent cache，第七十一章负责递归状态生命周期；本章不复述分页 allocator、scheduler、prefix-cache、安全或生产排障流程。

## 69.1 混合模型中的 KV 是按注意力路径保留的状态

本节建立“模型层—历史表示—读取范围”的对应关系。混合架构中的 cache 不是一种统一张量；层类型、位置可见范围和存储表示共同决定某条路径保存什么历史以及它能否逐位置读取。

### 69.1.1 全局 attention 为保留位置写入 K/V

以第十一章已讲的 causal Transformer 为前置，回顾 prefill 为各历史 token 生成 K/V、decode 追加当前 token K/V 并供后续 query 读取。重点转向混合架构：只要某些层保留逐位置的 attention 路径，这些层对应的历史表示仍须可供读取；不能因为其他层使用递归状态便断言整模没有随长度增长的显式 cache。

### 69.1.2 局部 KV、latent cache 与递归 state 代表不同历史

局部 attention 仍可逐 token 保存 K/V，但只保留其窗口内可见部分；MLA 类路径仍可按 token 位置进行全局 attention，却把每位置的 key/value 信息压进 latent；递归路径则持续更新 state，不保留一份可直接索引的完整 token 序列。三种表示对应不同的长度增长、访问方式与误差来源，MLA 计算和 state 生命周期分别由相邻章节主讲。

### 69.1.3 层表需要记录表示类型和可见历史

同一个模型可同时包含 full K/V、局部窗口 K/V、逐位置 latent 与递归 state。读架构配置时应分别记录每层的 mixer、可见范围、cache 表示、表示维度及其随序列长度的增长方式；“attention layer”本身并不保证存的是普通全维 K/V，“cache”一词也不足以说明能否按原位置读回历史。

## 69.2 混合架构选择显式路径的位置与形式

本节把混合架构的组织方式映射到显式 cache 的出现位置和保存形式。层间交错、层内并行、注意力范围和跨层共享是相互独立的设计轴，不能只用一个“attention/state 比例”概括。

### 69.2.1 层间交错只让部分层增长为显式历史

Qwen3.8-Next 报告中的基础 hybrid 以三层 GDN 配一层 global attention，continued pretraining 阶段又以 QSA 替换全局层；Kimi K3 则将 KDA 与 Gated MLA 按层交错。两者说明全局读取可以只放在部分深度，但 cache 表示仍须逐路径核对：例如 Kimi 的 Gated MLA 保存逐 token latent，而不是普通 head-specific K/V。层比例是具体模型配置，不构成普遍最佳配比。

### 69.2.2 层内 hybrid heads 与跨层 KV sharing 改变不同成本

Hymba 在同层并行 attention heads 与 SSM heads，并另行研究局部/全局 attention 和相邻层 KV sharing。并行加入状态分支改变一层可用的信息路径；KV sharing 改变重复保存的表示数量；local attention 改变每层可见的 token 范围。论文的质量、cache 与吞吐结果绑定其小模型、训练和评测设置，不能将整体收益归给单一改动。

### 69.2.3 全局、窗口与稀疏路径改变的不是同一个量

全局 attention 的 query 可对所有保留位置打分；窗口 attention 限制本层直接可见的历史；稀疏 attention 先选择部分位置或块再读取。它们分别影响可见集合、计算和 cache 的读取路径，递归状态则改变历史如何表示。QSA 等索引路径不能简单等同于“少保存了 KV”，具体缓存表示需依据模型报告或实现单独确认。

## 69.3 按层计算显式 KV payload

本节将架构层表转成可检查的容量估算。公式只统计普通逐 token K/V 的逻辑 payload；窗口、latent、递归状态及 allocator 元数据分别记账，不用层数占比直接代替真实总显存或性能。

### 69.3.1 全局 K/V 层按保留 token 数逐层求和

对每层独立存储普通显式 K/V 的层集合 $\mathcal{A}_{\mathrm{KV}}$，第 $\ell$ 层每个请求保留 $T_{i,\ell}$ 个位置、$H_{\mathrm{kv},\ell}$ 个 KV head、每 head 维度 $d_{h,\ell}$、每元素 $b_\ell$ bytes 时，批内 payload 由请求数和层内维度共同决定。正文给出其逐层求和式，并说明 batch 内请求长度不同时以各自实际保留位置数相加；跨层 sharing 按唯一存储项核算，模型权重、workspace、alignment 和分页元数据不包含在内。

### 69.3.2 窗口、GQA、latent 与 state 分别改动账本字段

窗口层的 $T_{i,\ell}$ 受窗口上限约束；GQA/MQA 改变 KV head 数；跨层共享减少重复存储；MLA 将每位置缓存维度换成 latent 相关维度；递归 state 的形状和更新成本由具体状态结构决定。它们不应混成一个压缩率，标准 KV 量化与分页物理管理也分别属于精度/布局层面。

### 69.3.3 少数 global 层的比例只在等形状假设下成立

用相同 head 数、维度、dtype 与上下文长度的假设，比较 32 层中 4 层或 8 层保留普通全历史 K/V 时的 payload 占比。该算例只展示等形状条件下的层数比例；若 global 层的 KV head、窗口、latent 维度或精度不同，比例随之改变，且 payload 也不包含 state、workspace 和 allocator 开销。

## 69.4 评估保留路径的质量与成本

本节把“为何保留显式读取”转为可复核的模型问题。论文中的 copy/retrieval、召回与下游 benchmark 能提供特定设计的证据，但不能由 attention 路径存在本身推出答案质量、长上下文有效性或端到端速度。

### 69.4.1 检索与复制任务应针对目标信息粒度

比较路径时覆盖精确数字/字符串复制、远程事实检索、跨段关系、冲突信息和常规语言任务；报告证据位置、长度与评价指标。Griffin、Hymba 和 Qwen 的实验分别绑定其模型、训练预算与任务，适合说明某种路径在特定设置下的作用，不证明所有递归模型都无法保留某类信息。

### 69.4.2 消融应隔离层位置、数量与表示方式

评估少量显式层是否够用时，对齐模型规模、训练 token、数据、上下文、优化和测试流水线，并分别调整显式层的数量/位置、full 或 local 范围、KV head/latent 维度和 state 路径。若一个模型同时改了这些组件，只能报告整套设计的联合结果，不能把变化单独归因给显式 KV。

### 69.4.3 报告质量、payload 与系统成本的条件

报告每层表示类型、每 token payload、随上下文增长的曲线、检索质量及 prefill/decode 测量条件，并将理论 payload 与实际设备占用和延迟区分。第十一章主讲一般 KV 容量，第十九/二十章主讲 attention pattern 与 kernel/PagedAttention，第六册和第二十四册负责服务系统、allocator、调度与端到端性能；本章只解释架构路径和逻辑状态账本。
