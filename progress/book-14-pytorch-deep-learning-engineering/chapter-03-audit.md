# 第三章审计：nn.Module 与模型组织

审计日期：2026-11-09  
状态：章节精修完成；十章骨架不变。

## scope

- 完整复读第三章 §§3.1–3.30，按对象注册树、模型状态生命周期和模型工程集成三条链审阅。
- 章级核验范围：PyTorch Module/Parameter/buffer 注册、容器、optimizer 参数组、`state_dict`、`torch.load`/`load_state_dict`、训练/评估模式、迁移、初始化、hooks、DDP 和 `torch.compile`。
- 未新增独立主题或章节。复杂 FSDP/checkpoint shard、编译器内部、生产级分布式与硬件性能仍由相应章节处理。

## chapter_scan

- **日期与范围：** 2026-11-09；以 PyTorch 2.15 官方文档为准，对本章技术主张做定向扫描；未把搜索摘要当作证据，也未将该扫描当作所有版本/设备的穷尽保证。
- **主要问题：** 哪些对象会进入注册树；保存和加载实际管理哪些状态；`strict=False`、参数迁移、DDP 包装和编译对参数对象及模块层级有什么影响；评估模式能否保证输出独立于 batch。

| 候选缺口 | 证据与决定 |
| --- | --- |
| `state_dict` 默认内容和自定义 extra state 的边界 | 纳入 §§3.1.1、3.10：标准状态为注册参数和持久 buffer；自定义 `get_extra_state` / `set_extra_state` 或 hook 可以追加非 tensor 状态，默认 state dict 是持有参数/buffer 引用的浅拷贝。补充其序列化兼容边界。 |
| `torch.load(weights_only=True)` 和配置保存 | 纳入 §3.11：固定 PyTorch 2.6 起的默认变化，建议配置使用受支持的基本类型字典；受限加载缩小 pickle 风险但不是任意文件的安全保证。 |
| `load_state_dict(assign=...)` 与 optimizer 时序 | 纳入 §3.12：说明默认 `assign=False` 的复制行为，`assign=True` 下 optimizer 创建时序的 PyTorch 2.15 文档警告。 |
| `strict=False` 是否容忍同名 shape mismatch | 纳入 §3.12/§3.23：仅放宽 missing/unexpected key 检查；同名不同 shape 在本地 PyTorch 2.12 CPU 环境验证仍抛出 size mismatch。尺寸变化需显式筛选或复制对应行。 |
| DDP 动态增删参数和 `module.` 前缀 | 纳入 §§3.20、3.25：DDP 包装前注册完整参数/buffer、各 rank 注册顺序相同；包装后不得增删。加载 DDP state dict 时改用官方 prefix helper；不做多进程或 CUDA 声明。 |
| 编译与 Module/distributed wrapper 的关系 | 保留在 §3.26：按 PyTorch 2.15 troubleshooting 文档建议选问题较少的高层函数或内部 module；性能、正确性和 DDP/FSDP 组合均未在本章本地运行，不外推成保证。 |
| `eval()` 和 BatchNorm 统计 | 纳入 §3.13：补充 `track_running_stats=False` 时评估仍使用当前 batch 统计；`eval()` 仅改变响应 `training` 标志的模块行为，不关闭 Autograd。 |

取舍与正文改动只记在本审计；来源事实与版本边界见[PyTorch 训练组件 API 边界研究记录](../../research/pytorch-training-api-boundaries-2026.md)。

## length_status

按能独立回答读者问题的连续教学链计数；仅计连贯中文解释正文，排除标题、公式、代码、表格、状态标记和链接 URL。三条核心链均高于 1,800 汉字下限。

| 教学链 | 覆盖范围 | 类型 | 实计汉字数 | 下限 | 状态 |
| --- | --- | --- | ---: | ---: | --- |
| 注册树到 optimizer | §§3.1–3.9 | 核心 | 2,736 | 1,800 | 通过 |
| 状态与运行生命周期 | §§3.10–3.16 | 核心 | 2,540 | 1,800 | 通过 |
| 结构、故障与集成 | §§3.17–3.27 | 核心 | 2,320 | 1,800 | 通过 |

导言至综合 Module 审计 demo 的解释性文字合计 8,266 汉字；综合 demo 与练习的代码、标题、表格、公式、链接和代码状态不计入字数。

## evidence

- 版本基线为 [PyTorch 2.15 `nn.Module`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html)、[`nn.Parameter`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.parameter.Parameter.html)、[`ModuleList`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.ModuleList.html)、[`ModuleDict`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.ModuleDict.html) 及对应容器文档。
- 序列化事实依据 [PyTorch 2.15 `torch.load`](https://docs.pytorch.org/docs/2.15/generated/torch.load.html)、[序列化说明](https://docs.pytorch.org/docs/2.15/notes/serialization.html) 和 [保存/加载教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)。
- DDP 和编译边界依据 [PyTorch 2.15 DDP API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.parallel.DistributedDataParallel.html) 与 [torch.compile troubleshooting](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_troubleshooting.html)。
- 官方文档支持公开 API 语义，不替代 PyTorch 2.15 runtime、目标硬件、真实 checkpoint、DDP/FSDP 拓扑或 compile 性能验证。详细新事实和来源证据已附到既有 research note。

## example_status

- **核心手算例：** §3.5。输入 `x=[3,2]`；第一层 weight `[2,-1]`、bias `1`，故 `2×3 + (-1)×2 + 1 = 5`；第二层 weight `2`、bias `-1`，加持久 buffer `offset=0`，输出 `2×5 - 1 + 0 = 9`。两层分别含 3 和 2 个参数，共 5 个 trainable scalar；buffer 另含 1 个 scalar，state dict 共 6 个 tensor 元素。手算输入与代码完全一致。
- **注册对照：** 上述代码使用 `ModuleList`。普通 list 中的模块仍可参加 Python 前向，但其参数不由 `model.parameters()`、`state_dict()` 或 `model.to()` 递归管理。
- **综合案例：** §3.27 同时检查 list/ModuleList、Parameter、持久/非持久 buffer、递归 train/eval、冻结后 optimizer 参数身份、strict-false missing keys 和 save/load logits 一致性。

## code_status

本章 84 个 Python 围栏均有相邻状态：8 个完成 CPU 执行验证，76 个静态审阅未运行。运行环境为 Python 3.12.9、PyTorch 2.12.0+cpu；官方语义资料版本为 PyTorch 2.15，两者不混写。执行方式是从 Markdown 提取目标围栏并在隔离命名空间运行；代码状态行不包含维护路径。

| 位置/代码块 | 状态与输入 | 预期/实际结果 |
| --- | --- | --- |
| §3.5 `HandCalcStack` | 已验证；`x=[[3,2]]`，两层固定参数 | stdout：`output: 9.0`、参数元素 5、state tensor 元素 6；五个 key 与两层参数及 `offset` 对应。 |
| §3.10 state dict value 检查 | 已验证；自定义 extra state 返回 dict，另含 `weight` 参数 | tensor 显示 shape `(2,)`；`_extra_state` 显示类型 `dict`，不会对非 tensor 错用 `.shape`。 |
| §3.10 独立快照 | 已验证；单权重从 2 改到 9 | clone 快照仍为 2，live 参数为 9。 |
| §3.12 DDP prefix helper | 已验证；`module.weight`/`module.bias` 映射加载普通 Linear | helper 移除 `module.`；权重与偏置加载正确，无 missing/unexpected keys。 |
| §3.22 `CustomDropout` | 已验证；`p=0,0.5,1`，shape `[8]` | p=0 输出原样；p=0.5 shape 有限；p=1 输出全零。 |
| §3.23 词表权重扩展 | 已验证；旧 embedding 4 行、目标 3 行 | 在 `no_grad` 下复制前 3 行，目标 shape 保持 `(3,2)`。 |
| §3.24 LoRA 参数快照 | 已验证；Linear weight 冻结、bias 可训练 | 只保存 `bias`，输出为 detached CPU clone。 |
| §3.27 完整审计 demo | 已验证；CPU，固定 seed 7，40 步小型合成分类任务 | 11 项断言全为 True；bad list 参数为空，ModuleList 4 个参数张量，训练 loss 从 0.8445 降至 0.0，身份集合与 optimizer 参数相符，save/load eval logits 一致。 |

另外验证 `strict=False` 只允许 missing/unexpected key：同名 Linear 权重 shape 从 `(1,3)` 加载到 `(1,2)` 时，PyTorch 2.12.0+cpu 仍抛出 size mismatch。其余 76 个围栏为依赖上下文的片段或 API 入口，均标注“静态审阅，未运行”，未将其描述为可执行实验。

## formula_status

- 8 个 `~~~math` 块围栏成对，花括号平衡；正文没有行内 `$...$` 公式。
- 线性层、embedding、MLP 参数量及可训练参数比例给出变量含义；`state_dict` 默认内容用 `S_default = P ∪ B_p` 表达，并紧邻说明自定义 extra state 是例外。
- 公式完成源码结构检查；没有宣称 MathJax/GitHub 页面实际渲染通过。

## link_status

- 章节共 31 个正文 Markdown 链接：28 个 PyTorch 官方文档链接、3 个本地章节入口；3 个本地目标均存在（第十三册第五章、第十四册第七/八章）。
- 关键 2.15 Module、Parameter、ModuleList/Dict、serialization、torch.load、DDP、BatchNorm 和 compiler troubleshooting 文档已直接阅读。没有执行全章外链 HTTP crawl，也没有宣称发布页面渲染已验收。
- 全章 184 个三反引号标记成对；`git diff --check` 通过。

## sync_status

- [第十四册简介](../../book-14-pytorch-deep-learning-engineering/简介.md)第三章摘要已更新为注册、optimizer、迁移、checkpoint、模式与分布式/编译边界。
- [第十四册目录](../../book-14-pytorch-deep-learning-engineering/目录.md)问题、产出与后续入口已同步。
- PyTorch 2.15 新核验事实写入既有 research note；章节取舍、代码状态、字数和交接留在 `progress/`。
- 未修改 `PAPERS.md`、题库、练习索引、术语表、`plan.md` 或 PDF。

## open_risks

1. 代码验证运行 PyTorch 2.12.0+cpu；未在 PyTorch 2.15 runtime、CUDA、多进程 DDP/FSDP 或真实 checkpoint 上执行。
2. 84 个代码块中 76 个依赖调用方对象或目标环境，仍为静态审阅；代码状态标记如实区分。
3. `torch.compile` 的编译正确性、重编译、包装顺序及性能未本地验证；只记录 PyTorch 2.15 文档建议。
4. 8 个 display-math 公式完成源码括号/围栏检查，但未在 GitHub/MathJax 实际渲染；外链未全量 HTTP 检查。

## handoff

- 第三章正文、章级扫描、手算例、8 个代码块执行核验和简介/目录同步已完成；没有改变十章骨架。
- 下一步按目录进入[第四章《Dataset 与 DataLoader》](../../book-14-pytorch-deep-learning-engineering/chapters/04-dataset-dataloader.md)，定向检查 Dataset/IterableDataset、collate/padding、worker、shuffle/pinning 和 sampler 的版本化事实。
- 全套书精修 goal 继续 active；不更新 PDF，不修改 `plan.md`。
