# 第二章审计：Autograd 与反向传播

审阅日期：2026-11-09  
状态：章级扫描与正文精修完成；源码结构检查和六组代码验证完成。目标平台的公式实际渲染、CUDA 与分布式运行未验收。

## scope

- **核心问题：** PyTorch 如何从前向运算构图、沿计算图回传梯度，并在 `.grad`、图生命周期、梯度累积与优化器更新之间管理不同状态。
- **读者收获：** 能手算链式法则和向量—雅可比积；区分叶子梯度与中间梯度；正确清梯度并按有效 token 归一化；选择 `detach`、`no_grad` 或 `inference_mode`；定位原地操作、hook、checkpoint、AMP 和 DDP 的 Autograd 边界。
- **本轮范围：** 完整审阅第二章；对照 PyTorch 2.15 官方文档检查动态计算图、梯度模式、in-place、`autograd.grad`、`zero_grad`、hooks、activation checkpoint、AMP 和 DDP；精修手算例、代码标记、引文链接和简介/目录摘要。
- **不改变书册边界：** 本章仍是 PyTorch eager 与梯度诊断入口；完整 checkpoint/AMP 在第六章、DDP 与 rank 数据权重在第七章展开。未增加章节。

## chapter_scan

- **日期与范围：** 2026-11-09；围绕本章现有小节检查 Autograd 记录条件、梯度归约、梯度模式、图释放、梯度高阶计算及工程集成边界。资料限 PyTorch 2.15 官方用户指南、API 与开发者说明；文档语义不推断其他版本或硬件表现。
- **资料类型：** PyTorch 官方 Autograd mechanics、`autograd.grad` / `backward`、`no_grad` / `inference_mode`、`zero_grad`、hook、checkpoint、AMP examples 和 DDP 文档；数字案例使用明确构造的教学输入。

| 候选或缺口 | 来源证据 | 去向与决定 |
| --- | --- | --- |
| 变长序列/不等有效 token 数下的梯度累积 | 官方 AMP accumulation 指南说明累积发生在 effective batch 层级；本章 token loss 需按有效 token 总数而不是 micro-batch 个数作分母。 | 纳入 §2.6：修正尾组处理，手算一组 1 token 与 3 token 的梯度；用同输入代码对照 micro-batch 等权 `-1.0` 与 token 等权 `-0.5`。跨 rank 归一化留第七章。 |
| `no_grad` 与 inference mode 的边界 | 2.15 API 文档包含 factory function / `nn.Parameter` 例外、inference tensor 不能直接进入 Autograd、view tracking / version counter 及 `eval()` 正交性。 | 纳入 §2.8；收窄性能表述，按后续是否进入 Autograd 选择上下文，不声称固定性能收益。 |
| `autograd.grad` 与 `.backward()`、DDP 的差异 | 官方 API 说明 `grad` 返回梯度且不累积到 `.grad`；DDP 文档要求标准路径写入参数 `.grad`。 | 纳入 §2.11、§2.14.4；说明两条路径不可直接互换，并链接第七章深入。 |
| Checkpoint 重入实现/API 默认变化 | 2.15 文档要求显式传 `use_reentrant` 并推荐 `False`；两种实现的记录图、重算和 API 支持有差别。 | 在 §2.14.2 增加版本边界，不将 checkpoint 参数或省显存效果写成跨配置保证；详细选择链接第六章。 |
| Hook 是否适合作全局裁剪 | Tensor hook 处理某个 Tensor 的梯度；全局范数需要跨参数统计。CUDA `.item()` 会把设备数值带回 Python。 | §2.12 删除“hook 可做梯度裁剪”的宽泛说法，收窄为逐 Tensor mask/变换；全局裁剪仍显式置于 backward 后、step 前。 |
| `masked_fill` 因果 mask 布尔语义 | 原代码使用未定义的 `causal_mask`，读者无法判断 `True` 是保留还是屏蔽。 | §2.9 改成显式 `future_mask`：上三角对角线以上为 `True`，由 `masked_fill` 替换；注明此约定不自动适用于其他 attention API。 |
| `grad_norm` 的度量口径 | 原代码累加每个参数的 L2 范数，这不是拼接所有参数后的全局 L2 范数。 | §2.10 与 §2.15 改为平方和再开方，区分全局范数；诊断代码限定单设备场景。 |
| `torch.func`、自定义 autograd、forward-mode AD、完整分布式 autograd | PyTorch 文档有相关独立 API/机制，但不属于本章 eager 反向梯度排障主线。 | 在范围中保留边界，不扩为新单元；需要高阶/函数式梯度或自定义算子时转专门文档与对应主题。 |

**扫描结论：** 不改变十章骨架。第二章继续作为一个贯穿“前向→图→反向→`.grad`→参数更新→梯度诊断”的核心单元；多设备/完整精度优化按册级职责交由第六、七章。

## length_status

| 独立单元 | 类型 | 实计汉字数 | 下限 | 状态 |
| --- | --- | ---: | ---: | --- |
| 第二章《Autograd 与反向传播》全文（至练习之前） | 核心 | 7,176 | 1,800 | 通过 |

计数方式：单篇章节按一个连续核心教学单元统计；排除标题、Python/文本代码块、数学公式、表格和代码状态行，剔除链接 URL 与行内代码，保留中文解释正文和可读链接文字。单节编号用于组织同一梯度生命周期与排障链，不作为独立讲次重复计数。

## evidence

- API 事实以 PyTorch 2.15 官方文档为准；本轮核对的来源、实际阅读章节、证据范围和版本边界记在[PyTorch 训练组件 API 边界研究记录](../../research/pytorch-training-api-boundaries-2026.md)。本审计记录章节取舍和修改，不复写研究事实。
- 关键边界包括：仅叶子张量默认累积 `.grad`；`autograd.grad` 返回梯度而不写 `.grad`；`no_grad` 的工厂函数例外；inference tensor 不可直接回到被 Autograd 记录的路径；`zero_grad(set_to_none=True)` 中 None 与零梯度语义不同；`use_reentrant` 的显式选择；hook 返回梯度及 DDP 的 `.grad` / `no_sync()` 要求。
- 官方文档支撑公开接口行为，不支持 PyTorch 2.12/2.15 之间所有边界都相同，也不证明 CUDA、分布式生产路径或固定性能收益。

## example_status

- **链式法则手算：** §2.1.1/§2.5。假设 $x=2$、$h=x^2$、$z=3h$，前向 $h=4,z=12$；局部导数为 $3$ 与 $4$，相乘得 $dz/dx=12$。
- **向量—雅可比积：** §2.5。输入 $x=[1,2,3]$，$y=x^2$，外部梯度 $v=[1,2,3]$；对角 Jacobian 为 $[2,4,6]$，元素积后得到 $[2,8,18]$。与 Python 输出一致。
- **有效 token 累积：** §2.6。教学损失 $\ell(\theta;y)=\frac12(\theta-y)^2$、$\theta=0$；micro-batch A 目标 `[2]` 梯度为 `-2`，B 目标 `[0,0,0]` 梯度为 `0`。两批均值等权得到 `-1`，四 token 总均值为 `-0.5`；代码逐项复用同一输入。
- **因果 mask：** §2.9。取 `T=4`、scores `[1,1,4,4]`，mask 屏蔽严格上三角，保留对角线与下三角；四行 softmax 均有限。
- **梯度范数：** §2.10。单设备 `Linear(2,1)`、输入 `[1,2]`，全局 L2 范数平方和再开方得 `2.4494898`，与 `clip_grad_norm_(max_norm=1e6)` 返回值一致。
- **综合数据流：** §2.15 完整 demo 覆盖标量导数、VJP、叶子/非叶子、梯度累积、`detach`、优化器更新及 `no_grad` 验证。
- **视觉表达：** §2.2 新增图生命周期/`.grad` 状态表，区分清梯度、forward、backward、step 和下一轮重建；无 Mermaid 图，不涉及平台图渲染。

## code_status

当前章共 50 个 Python 围栏块，正文逐块标状态：6 个已验证、44 个静态审阅未运行。静态状态不表示已执行；上下文片段所需的模型、批次、参数和依赖由相邻正文说明。

| 位置/代码块 | 状态 | 日期、环境与输入 | 结果/边界 |
| --- | --- | --- | --- |
| §2.5 VJP 示例 | 已验证 | 2026-11-09；Python 3.12.9、PyTorch 2.12.0+cpu；`x=[1,2,3]`、`v=[1,2,3]` | 输出 `[2,8,18]`，与手算一致。 |
| §2.6 梯度累积循环 | 已验证 | 同环境；五个等有效权重 micro-batch；单设备 `Linear` 与 MSE | 累积组大小 4 后尾组大小 1，合计执行两次 optimizer step；无残余未更新尾梯度。未验证变长 token 权重，此项由下一行数例处理。 |
| §2.6 token 权重数例代码 | 已验证 | 同环境；目标 `[2]` 与 `[0,0,0]`、标量参数初值 0 | 输出 `equal_microbatch_gradient=-1.0`、`equal_token_gradient=-0.5`。 |
| §2.9 `future_mask` 片段 | 已验证 | 同环境；`T=4`、scores shape `[1,1,4,4]` | 屏蔽严格上三角；softmax 输出全有限且每行保留当前位置。仅验证 `masked_fill` 语义，不代表所有 attention API 的布尔 mask 约定。 |
| §2.10 全局梯度范数片段 | 已验证 | 同环境；单设备 `Linear(2,1)`、输入 `[1,2]` | 范数 `2.4494898`，与 PyTorch 返回的全局范数相同；仅检查单设备普通 dense 梯度。 |
| §2.15 完整 Autograd demo | 已验证 | 同环境；`torch.manual_seed(13)`，CPU | 标量导数 `14`；VJP `[2,8,18]`；leaf 梯度 `96`、保留的中间梯度 `24`；累积 `2→5`、清后 `3`；detach 支路参数梯度 `5`；8 步 loss 从 `15.882` 降到 `0.1121`，验证 loss `0.0649` 且不需要梯度，缺失梯度列表为空；7 项断言全通过。 |

执行方式：用 `.tmp/llm-book-cpu-check/bin/python`（Python 3.12.9 / PyTorch 2.12.0+cpu）从章节 Markdown 提取目标 `python` 围栏并在相应 CPU 命名空间 `compile/exec`。完整 demo stdout 与状态记录一致。PyTorch 2.15 文档是 API 证据版本，本地运行版本是 2.12 CPU，两者不可混写。

## formula_status

- 全章 15 个 `~~~math` display 块均成对，花括号平衡；行内 `$` 分隔符 58 个、成对。
- `K` 在 micro-batch 公式上下文定义；`\mathcal{T}_k` 表示 micro-batch 中有效 token 集；新增加权数例列明参数、目标值、损失、梯度和分母，可手算复核。
- 执行源码结构检查；未用浏览器/MathJax 目标页实测公式渲染。

## link_status

- 章节含 32 个 Markdown 链接（29 个 PyTorch 官方外链、3 个相对路径链接）；3 个本地目标均存在，包括第六、七章交接入口。
- 外链均使用 `docs.pytorch.org` 的固定 2.15 文档 URL；关键 Autograd、grad、mode、zero-grad、checkpoint、AMP、DDP 和 hook 页面已直接核对。未做全章外链 HTTP 扫描，也未宣称全部外链经过在线可用性检查。
- 代码围栏配对，数学围栏/括号检查通过；`git diff --check` 通过。

## sync_status

- [第十四册简介](../../book-14-pytorch-deep-learning-engineering/简介.md)第二章摘要已同步链式法则/VJP、有效 token 归一化及 Autograd 工程边界。
- [第十四册目录](../../book-14-pytorch-deep-learning-engineering/目录.md)第二章读者问题补入累积时有效 token 权重；前置、产出和后续连接仍一致。
- 2.14 checkpoint/AMP 回链本册第六章；DDP 与 rank 归一化回链第七章。PyTorch 新 API 事实进 `research/`；修改决定、长度、代码状态和交接只记 `progress/`。
- 未修改题库、术语表、PDF 或 `plan.md`。

## open_risks

1. 验证环境为 PyTorch 2.12.0+cpu，未运行 PyTorch 2.15、CUDA、混合精度或多进程 DDP；版本化 API 说明和本地执行证据已分开。
2. 除 6 个验证块外的 44 个短片段仍是静态审阅；训练模型/验证代码取决于正文所述调用方对象，未作为可运行实验宣称。
3. `inference_mode`、checkpoint、AMP 与 DDP 的设备成本及组合行为未在目标硬件复测；代码没有声称性能保证。
4. 15 个 display-math 公式完成源码结构检查，但目标平台 MathJax 实际渲染留待本册最终门禁。
5. 未做全部外链 HTTP 巡检；链接均固定到 PyTorch 官方 2.15 文档页面。

## handoff

- 第二章扫描、手算例、正文修订、简介/目录同步和六组代码运行已完成；十章骨架与职责未改变。
- 下一步按目录进入[第三章《nn.Module 与模型组织》](../../book-14-pytorch-deep-learning-engineering/chapters/03-nn-module与模型组织.md)，检查参数/buffer/子模块、训练/评估模式和状态保存，并同步本册简介摘要。
- 全套书精修 goal 保持 active；不更新 PDF，不修改 `plan.md`。
