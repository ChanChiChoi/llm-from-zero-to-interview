# 第一章审计：PyTorch 张量基础

审阅日期：2026-11-09  
状态：章级扫描与正文精修完成；源码结构检查完成。目标平台的公式实际渲染未验收，GPU 环境未验证。

## scope

- **核心问题：** 如何沿语言模型的输入、张量变换、attention 与 loss 数据流追踪 shape、dtype、device 和内存布局，并判断“代码能运行”是否也代表语义正确。
- **读者收获：** 能写出 batch-first shape contract；区分类别轴与序列轴；选择广播、矩阵乘法、view/reshape；检查 padding、mask、有效 loss 目标和 layout。
- **本轮范围：** 完整复读第一章；定向查证 PyTorch 2.15 张量构造、Linear、device、广播、索引、存储视图及 cross-entropy 文档；精修正文、手算例子、代码状态、来源链接和简介摘要。
- **不改变书册边界：** 不加入 autograd、CUDA kernel、编译器或 profiler 教学；继续由后续章节分别处理。

## chapter_scan

- **日期与范围：** 2026-11-09；针对本章每个现有主问题检查官方 PyTorch 2.15 文档；以稳定 API 语义为主，不推断 GPU 性能和生产训练行为。
- **资料类型：** PyTorch 2.15 官方 API 文档与 Tensor Views、Broadcasting 开发者说明；另复用本地 PyTorch 2.12.0+cpu 的明确标注观察。

| 候选/缺口 | 来源与证据 | 去向与决定 |
| --- | --- | --- |
| `[B,T,V]` 的 `cross_entropy` 类别轴及目标 shape | 官方文档定义多维输入以 `dim=1` 为类别轴，类索引 target 省去类别轴。 | 纳入 §1.1.1、§1.2；明确直接传入 `[B,T,V]` 会把 `T` 解释为类别数，并给出 flatten 和移轴两条合法路径。 |
| 数学线性权重与 PyTorch `nn.Linear.weight` 方向 | 官方 `nn.Linear` 文档给出 `y=xA^T+b` 和 `(out_features,in_features)` 权重形状。 | 纳入 §1.1.1；将数学权重记法与参数存储方向并列说明。 |
| `torch.tensor` / `torch.as_tensor` 的复制与共享 | 官方文档说明 `torch.tensor` 复制输入；`as_tensor` 尽可能共享并在转换时复制。 | 纳入 §1.5；收窄“通常拷贝/尽量共享”表述，不声称 `as_tensor` 总是零拷贝。 |
| CPU/GPU 设备规则中的零维标量例外 | 官方 Tensor Attributes 文档记录 CPU 标量可在部分 CPU→GPU 运算中自动转移；反方向和非标量不会自动转移。 | 纳入 §1.4 和小结；工程建议仍是显式设置 device。 |
| `view`、`reshape`、`expand`、索引与广播的 storage 语义 | 官方 Tensor Views、view/reshape/expand 和 broadcasting 文档。 | 核验并保留现有解释；补充广播不复制输入、原地运算限制，以及 `expand` 后可能由其他操作物化分配。 |
| 全 mask 行和全忽略 loss 的边界 | PyTorch 2.15 API 文档说明忽略目标的 reduction 规则，但不承诺全忽略时所有设备的具体返回值；PyTorch 2.12.0+cpu 本地观察到全忽略 `mean` 返回 NaN，整行 `-inf` 的普通 softmax 返回 NaN。 | 正文限定本地观察的版本/设备，不外推到 fused kernel 或其他版本；建议显式保证有效目标数大于零和有效 query 的 mask 约束。 |
| 稀疏/nested tensor、CUDA kernel、autograd、compile/profiler | 与本章基础 tensor contract 不符，分别由后续章节或专门接口处理。 | 不新增主题，不改变十章骨架。 |

**扫描结论：** 未发现需要新增章节或改变书册边界的主题。第一章继续作为一个核心教学单元，以 token-to-loss 路径串联 API；不把连续小节机械拆成独立讲次。

## length_status

| 独立单元 | 类型 | 实计汉字数 | 下限 | 状态 |
| --- | --- | ---: | ---: | --- |
| 第一章《PyTorch 张量基础》全文 | 核心 | 5,599 | 1,800 | 通过 |

统计方法：以单篇章节文件作为一个连续核心单元；统计至章末练习之前的解释性中文字符，排除标题、代码围栏、数学公式、表格、状态行和引用链接 URL，保留正文列表解释并计入链接文字。篇幅来自连贯的 tensor 数据流讲解、shape/stride 机制、边界与手算案例，不以代码或表格充数。

## evidence

- 正文主要 API 语义固定到 PyTorch 2.15，链接在章节 §1.22 和对应结论附近，覆盖 Tensor Attributes、tensor/as_tensor、Linear、broadcasting、matmul/mm/bmm/einsum、Tensor Views/view/reshape/expand/contiguous 与 cross_entropy。
- 新核验事实及实际阅读位置、适用边界写入[PyTorch 训练组件 API 边界研究记录](../../research/pytorch-training-api-boundaries-2026.md)；此处只记录本章审阅决定和进度。
- API 文档支持接口语义，不支持吞吐或硬件表现结论；PyTorch 2.12.0+cpu 的两个边界结果仅记为 local-demo/observation，不外推到 PyTorch 2.15、CUDA 或 fused attention。

## example_status

- **核心数例：** §1.2，假设输入 `[[0,1]]`、embedding 表 `[[1,0],[0,1]]`、bias `[0,1]`、单位矩阵投影、标签 `[[0,1]]`。查表得 `[[[1,0],[0,1]]]`；广播相加得 `[[[1,1],[0,2]]]`；next-token shift 得 logits `[1,1]`、目标类 `1`；softmax 目标概率 `1/2`，交叉熵 `ln 2≈0.6931`。
- **与代码对应：** §1.2 的 Python 示例复用完全相同输入，检查 hidden 数值、shift 后 `[1,2]` logits、目标 `[1]` 和 loss 与 `ln 2` 的误差。
- **完整数据流：** §1.1.1 的 shape contract 表、attention 公式和 §1.19 demo 展示输入、embedding、attention mask、layout 与 loss flatten 的联系。
- **视觉机会：** shape contract 表呈现逐阶段 shape 与语义检查；手算例子紧随其后。无需额外 Mermaid，避免重复表示同一条线性数据流。

## code_status

| 位置/代码块 | 状态 | 验证范围 |
| --- | --- | --- |
| §1.2 手算数例代码 | 已验证 | Python 3.12.9、PyTorch 2.12.0+cpu；CPU。输出 hidden `[[[1,1],[0,2]]]`、shift logits `[[1,1]]`、target `[1]`、loss `0.6931`。 |
| §1.19 完整 tensor 审计 demo | 已验证 | 同一环境、CPU fallback。覆盖 shape、broadcast、matmul/einsum 数值一致、mask 行和、`view`/`contiguous`、loss 有限和六项 contract 检查；输出均与正文注释一致。 |
| §1.20 `audit_tensor` | 静态审阅，未运行 | 审阅 shape/dtype/device/grad/layout 元信息和 finite 检查；`.item()` 在 CUDA 路径可能同步，正文已说明。 |
| 其他短片段 | 静态审阅，未逐段运行 | 逐段核对输入输出 shape、dtype/device、API 用法和文中预期；未把静态审阅标成执行验证。 |

手算样例执行命令：

~~~sh
.tmp/llm-book-cpu-check/bin/python -c 'import re,pathlib; p=pathlib.Path("book-14-pytorch-deep-learning-engineering/chapters/01-pytorch张量基础.md"); s=p.read_text(); section=s.split("以下代码使用同一组数值，对照手算结果",1)[1]; code=re.search(r"```python\n(.*?)\n```",section,re.S).group(1); exec(compile(code,str(p),"exec"))'
~~~

完整 demo 执行命令：

~~~sh
.tmp/llm-book-cpu-check/bin/python -c 'import re,pathlib; p=pathlib.Path("book-14-pytorch-deep-learning-engineering/chapters/01-pytorch张量基础.md"); s=p.read_text(); section=s.split("## 1.19 最小可运行 PyTorch 张量审计 demo",1)[1].split("## 1.20 一个完整的张量审计过程",1)[0]; code=re.findall(r"```python\n(.*?)\n```",section,re.S)[0]; exec(compile(code,str(p),"exec"))'
~~~

## formula_status

- 已明确 `B,T,d,V,H,d_h`、token 范围、label/ignore_index 约定及 batch-first 假设；数学线性权重和 PyTorch 参数权重方向已区分。
- display-math 使用 13 组 `~~~math` 围栏并配对；花括号和公式围栏已作源码层检查。GitHub/浏览器中的 MathJax 实际渲染尚未确认。
- 无新增 Mermaid 图；表格和数例各自承担 shape 关系与逐步计算表达。

## link_status

- 章节包含 27 个 Markdown 链接，均为固定版本 PyTorch 2.15 官方文档；本章无本地文件链接。重要 API 页面已直接检查原始文档内容。
- Markdown 围栏成对；`git diff --check` 通过。没有做全量外链爬取，也没有宣称所有链接都经过逐条在线可用性检查。

## sync_status

- [第十四册简介](../../book-14-pytorch-deep-learning-engineering/简介.md)第一章摘要已按正文改为 token id→embedding→线性投影→next-token loss 的手算路径，并突出类别轴与 tensor contract。
- 目录中的第一章入口、依赖和产出仍与正文一致；第一道章末练习已补充 cross-entropy 类别轴判断。
- [第十四册架构审计](architecture-audit.md)和全套书交接已更新为第一章完成、第二章接续；新 API 来源证据在 `research/`，本审计只存修改与进度。
- 未修改题库、术语、PDF 或总计划；本章没有发现必须同步的其他配套资料。

## open_risks

1. 两个可运行例子仅在 PyTorch 2.12.0+cpu 的 CPU 环境验证；CUDA、其他版本、mixed precision 和 fused attention 路径未验证。
2. 局部短片段仍为静态审阅；`audit_tensor` 未执行。
3. 公式只完成源码结构检查，目标平台 MathJax 实际渲染门禁仍开。
4. 全目标被忽略时的 NaN 是 PyTorch 2.12.0+cpu 观察，不作为其他版本/设备保证；正文已要求检查有效目标数。

## handoff

- 第一章正文、扫描、长度、数例、代码状态和简介同步已完成；没有新增章节或改动书册定位。
- 接下来进入[第二章《Autograd 与反向传播》](../../book-14-pytorch-deep-learning-engineering/chapters/02-autograd与反向传播.md)，重新按章节任务加载规范并复核计算图、梯度边界和代码状态。
- 不更新 PDF；第十四册第二至第十章仍待逐章处理，整册门禁未完成。
