# 第 63 章 Gated MLA：潜变量缓存与通道输出门

## 章节任务

本章解释 Kimi K3 报告中的 Gated MLA：先用 MLA 的低秩联合压缩保存每个历史 token 的 KV latent，再通过全局注意力读取这些位置，最后由输入依赖的逐通道输出门调制 attention result。核心区分是“压缩历史表示”与“调制读出结果”，不把模型描述成在显式 KV 与 latent KV 两条路径之间动态选路。

第 70 章负责通用 latent cache 的压缩、保真与测量；第 64 章负责 NoPE 的顺序信息边界；第 65 章负责 KDA 与全局 MLA 的层间组合；第 88 章负责 Kimi K3 全模型。这里仅引入理解 Gated MLA 所需的基础公式与模型版本事实。

## 63.1 Gated MLA 改变历史表示和输出调制

- 说明标准显式 K/V cache 随 token 数线性增长，并点出 MLA 要压缩每个 token 的 KV 表示。
- 用一段路线图分开 latent 压缩、全局注意力和通道输出门三件事。

## 63.2 MLA 将每个 token 的 key/value 联合压到 latent

### 63.2.1 共享 latent 与低秩 key/value 投影

- 定义输入、KV latent、key/value 上投影及其维度；说明每个历史 token 保存一个共享的低维 KV latent。
- 与标准显式 K/V cache 比较每 token 的理想元素量。

### 63.2.2 Decode 时吸收投影的代数视图

- 解释 key/value 上投影可分别折入 query 与 output projection，从而无需总是物化完整 K/V。
- 区分可进行的代数变换与具体 kernel 是否采用该路径。

## 63.3 位置路径按模型版本区分

### 63.3.1 DeepSeek-V2 的 decoupled RoPE 分支

- 说明 RoPE 与低秩 KV latent 的耦合问题，以及额外共享 RoPE key 的解耦处理。
- 在 cache 账本中计入该位置分量，避免把 latent 误当成所有版本的完整 cache。

### 63.3.2 Kimi K3 Gated MLA 的 NoPE 配置

- 依据 Kimi K3 v2 报告和固定 config 说明 MLA 层不对 query/key 施加显式位置变换。
- 将“位置与近期性主要由 KDA 层处理”标记为该模型报告的设计解释，并回链第 64、65、88 章。

## 63.4 Gated MLA 在 attention 输出后施加通道门

### 63.4.1 输入依赖的逐通道输出公式

- 定义未门控 MLA 输出、输入依赖 gate、sigmoid 与输出投影；说明 gate 的维度与逐通道乘法。
- 依据原报告说明 Kimi K3 使用 full-rank gate。

### 63.4.2 输出门不是 latent/explicit 路由

- 指明该 gate 调制 attention result，不选择缓存表示、不改变历史 token 的注意力权重，也不在 latent 与显式 KV 输出间做凸组合。
- 区分 output gate、MLA latent 压缩与第 61 章 KDA state/output 参数。

## 63.5 K3 配置中的 cache 账本与代价边界

### 63.5.1 位置分支如何进入 per-token payload

- 简要列出显式 KV、带 decoupled RoPE key 的 MLA 与 NoPE MLA 的元素量差别，并回链第 70 章的通用压缩账本。
- 说明计算仍要对历史位置做全局注意力；压缩 KV 不等于常数状态或自动降低所有 decode 成本。

### 63.5.2 Kimi K3 公布 rank 的理想估算

- 用固定 config 的 24 个 Gated MLA 层和 KV rank 512 计算跨 MLA 层的每 token latent 元素数。
- 说明按元素字节数换算的理想值，不代表实测运行时 cache；排除 KDA 状态、页元数据和临时 buffer。

## 63.6 Token 级 latent cache 与递归状态

- 对比 MLA 为各历史位置保留 token 级 latent、KDA 将历史汇总为递归状态的差异。
- 说明 MLA 保留全局位置访问路径，但低秩表示有容量边界；将通用信息保真和读取成本交给第 70 章。

## 63.7 Kimi K3 证据能支持到哪里

- 绑定 Kimi K3 v2 技术报告与固定 HF config，核对 gate 公式、NoPE、KV rank 和 attention layer composition。
- 仅把报告披露作为架构事实；不从整模型成绩或公式本身单独归因 output gate 的效果。
- 收束读者判断：latent 表示保存什么、attention 仍访问哪些位置、gate 在哪一步调制输出。
