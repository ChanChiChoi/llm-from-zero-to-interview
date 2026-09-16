# Kimi Delta Attention：从可更新记忆到线性注意力

> 资料核验：Kimi Team，*Kimi Linear: An Expressive, Efficient Attention Architecture*，arXiv:2510.26692，首次提交 2025-10-30。Kimi K3 的官方发布文章明确提到 KDA，但没有公开 K3 的完整层配置；本章的递推式、3:1 混合比例和实验数字来自 Kimi Linear 论文，不能直接当作 K3 的全部实现。

## 为什么要把注意力写成“记忆更新”

普通全注意力在生成第 `t` 个 token 时，会把 query 和此前所有 key 做匹配，再读取对应 value。它保留了细粒度检索能力，却需要随着上下文增长保存越来越长的 KV cache。对于百万 token、长时间运行的 Agent 或持续代码会话，这个缓存会变成显存和带宽瓶颈。

线性注意力选择另一条路线：不保存全部历史 token，而是维护一个固定大小的状态矩阵。新 token 到来时，模型先让旧记忆衰减，再把当前 key/value 写入状态。这样解码状态的形状不随序列长度增加，但也意味着模型必须学会在有限容量中保留重要信息。

Kimi Delta Attention（KDA）属于带门控的线性注意力。它继承 DeltaNet/Gated DeltaNet 的“纠正记忆”思路，同时把一个标量衰减扩展为逐通道的衰减向量，让不同记忆通道可以有不同的遗忘速度。

## 先看一个极简记忆矩阵

假设 key 有两个维度，value 也有两个维度。状态 `S` 是 `2×2` 矩阵，可以理解为一个从 key 空间到 value 空间的可更新映射。若某个 key `k=[1,0]` 对应 value `v=[3,5]`，最简单的写入会产生外积：

```math
kv^\top=\begin{bmatrix}1\\0\end{bmatrix}\begin{bmatrix}3&5\end{bmatrix}=\begin{bmatrix}3&5\\0&0\end{bmatrix}.
```

随后给定 query `q`，模型用 `S^\top q` 读出 value。它不需要遍历全部历史 token；代价是多个 token 可能写入相同状态区域，旧信息会被覆盖或衰减。

## KDA 的递推公式

论文给出单头递推：

```math
S_t=\left(I-\beta_t k_tk_t^\top\right)\mathrm{Diag}(\alpha_t)S_{t-1}+\beta_t k_tv_t^\top,
```

```math
o_t=S_t^\top q_t.
```

这里：

- `S_t` 是 `d_k × d_v` 的记忆状态。
- `q_t`、`k_t` 分别是 query 和 key。
- `v_t` 是要写入的 value。
- `alpha_t` 是长度为 `d_k` 的逐通道衰减向量，通常限制在 `[0,1]`。
- `beta_t` 是标量更新门，控制本次 delta 写入强度。
- `I-beta_t k_t k_t^T` 是对旧状态的 rank-1 修正。

可以按三个动作理解这条式子：

1. `Diag(alpha_t) S_{t-1}`：不同 key 通道以不同速度保留或忘记旧记忆。
2. `- beta_t k_t k_t^T (...)`：沿当前 key 方向消除一部分旧映射，避免新内容和旧内容无条件叠加。
3. `+ beta_t k_t v_t^T`：把当前 key 到 value 的关联写入状态。

因此 KDA 不是简单的“指数滑动平均”，也不是把全注意力矩阵近似成一个普通低秩乘积。它同时包含衰减、定向擦除和 rank-1 写入。

## 为什么逐通道门控有用

如果所有通道只有一个衰减标量，状态中的所有方向会以相同速度遗忘。但不同信息的时间尺度可能不同：一个通道负责短期语法线索，另一个通道负责跨段落的实体关系，第三个通道可能需要快速覆盖临时变量。`alpha_t` 为每个 key 通道提供独立的衰减选择。

论文将 KDA 与 DPLR（Diagonal-Plus-Low-Rank）结构联系起来。通用 DPLR 可以表达更丰富的状态转移，但计算和 chunkwise 并行更复杂。KDA 将 DPLR 中的两个向量绑定到同一个 `k_t`，减少二级 chunking 和矩阵乘法，从而更适合 GPU kernel。论文报告的效率提升来自其具体实现和测量设置，不能直接泛化到任意硬件。

## KDA 与全注意力如何分工

有限状态擅长以固定成本持续更新，却可能在精确检索很久以前的某个 token 时丢失细节。Kimi Linear 论文因此采用混合架构：实验中以三个 KDA 层搭配一个全局 MLA 层，让 KDA 负责高效的递归状态和位置/新近性建模，全局层负责定点检索。全局层使用 NoPE 是该论文的架构选择，并不是 KDA 的必然要求。

这给出一个实际的设计思路：不要把“线性注意力”和“全注意力”当成只能二选一的路线。前者控制长序列的状态成本，后者补回精确检索能力；比例、位置编码、局部卷积分支和硬件 kernel 需要通过同预算实验决定。

## 零依赖最小实现

下面的代码对应递推式，使用纯 Python，便于观察状态矩阵始终保持 `d_k × d_v`。它省略了论文的 ShortConv、Swish、L2Norm、低秩门控和 chunkwise kernel，因此只用于教学。

```python
from math import sqrt


def outer(a, b):
    return [[x * y for y in b] for x in a]


def l2_normalize(x):
    norm = sqrt(sum(v * v for v in x))
    return [v / norm for v in x]


def kda_step(state, q, k, v, alpha, beta):
    dk, dv = len(k), len(v)
    decayed = [[alpha[i] * state[i][j] for j in range(dv)]
               for i in range(dk)]

    # k^T @ decayed: a value-dimension vector
    old_value = [sum(k[i] * decayed[i][j] for i in range(dk))
                 for j in range(dv)]
    write = outer(k, v)
    state = [[decayed[i][j] - beta * k[i] * old_value[j]
              + beta * write[i][j]
              for j in range(dv)] for i in range(dk)]

    # S^T @ q
    output = [sum(state[i][j] * q[i] for i in range(dk))
              for j in range(dv)]
    return state, output


state = [[0.0, 0.0], [0.0, 0.0]]
q = l2_normalize([1.0, 0.2])
k = l2_normalize([1.0, 0.0])
v = [3.0, 5.0]
state, output = kda_step(
    state, q, k, v,
    alpha=[0.9, 0.5],  # 每个 key 通道不同的保留率
    beta=0.8,          # 本次写入强度
)
print("state shape:", (len(state), len(state[0])))
print("output:", [round(x, 3) for x in output])
```

运行后无论处理了多少 token，`state shape` 都是 `(2, 2)`。这解释了“解码状态大小与序列长度无关”的含义，但不应被误解为“模型不会遗忘”：状态容量固定，衰减和覆盖仍然存在。

## 训练与推理的两种计算形态

论文在训练和长序列预填充中使用 chunkwise 并行，将一段递推展开成矩阵计算，以利用 GPU 的矩阵乘法；在自回归解码时则维护一个状态矩阵，每到一个 token 执行一次递推。两种形态需要保持数值和因果语义一致，不能简单把训练时的并行公式当作推理 kernel。

工程实现需要特别关注：

- `alpha` 很小时，长序列乘积可能下溢，数值稳定性会影响 chunkwise 展开。
- `q/k` 的归一化和 gate 参数化会影响状态谱和梯度稳定性。
- 预填充偏向吞吐，解码偏向低 I/O 和固定状态；同一算子通常需要不同 kernel。
- 线性状态降低 KV cache，但混合全注意力层仍需要自己的 cache，不能把整模型 cache 直接视为常数。

## 和 Kimi K3 的边界

Kimi K3 官方文章称其采用 KDA、AttnRes、Stable LatentMoE，并披露了百万 token 上下文等信息；该文章没有给出 K3 完整技术报告的公式、每层 KDA/全注意力比例或全部 kernel 配置。本章引用的 Kimi Linear 论文用于解释 KDA 原理，不等于证明 K3 的每个实现参数都相同。

## 面试追问

**问：KDA 为什么是线性注意力？**  因为解码时维护固定大小状态 `S_t`，不需要随序列长度保存并扫描完整 KV 历史；每步更新和读取主要由固定维度矩阵运算组成。

**问：KDA 会不会完全替代全注意力？**  不一定。有限状态会损失精确长程检索，因此论文采用 KDA 与全局 MLA 的混合结构。

**问：`alpha` 和 `beta` 的区别？**  `alpha` 是逐 key 通道的旧状态衰减，`beta` 是当前 token 的整体更新强度；一个控制保留，一个控制本次修正和写入。

**问：KDA 与普通 RNN 有什么关系？**  两者都递归维护状态，但 KDA 的状态是 key-value 关联矩阵，更新包含 delta rule 的定向纠正，并可用 chunkwise 形式并行化；不能把它简化成标量 hidden state RNN。

## 小练习

1. 用二维状态手算一次 `alpha=[1,1]`、`beta=1` 的更新，观察 delta 修正如何改变旧映射。
2. 把 `alpha` 改为标量，比较逐通道衰减和统一衰减的状态轨迹。
3. 为代码加入长度为 `T` 的循环，验证状态内存不随 `T` 增长。
4. 设计一个混合层消融：全 KDA、3:1 KDA/MLA 和 1:1 KDA/MLA，在相同训练 token 与推理预算下比较长程检索、吞吐、显存和延迟。
