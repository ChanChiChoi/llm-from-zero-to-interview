# 第八章《强化学习数学》精修审计

审计日期：2026-11-09

## scope

- 全文复读 §§8.1–8.22；检查 MDP 状态与轨迹、终止奖励、return/value/advantage、policy gradient、GAE、PPO、KL、reward model、RLHF、DPO、奖励投机、验证器、评估、案例和代码之间的解释链。
- 保留第八章作为数学推导和工程判断入口；完整 reasoning/RLVR/GRPO 训练流程仍由第十六册承担。本章未改变册级骨架，不更新 PDF。
- 将 $T$ 统一为 action/reward 时间步数，索引范围为 $0,\ldots,T-1$；把零基 RL 步骤 $t$ 与一基 response token $y_{t+1}$ 明确对应；加入 GRPO 组内优势的最小手算入口，并把同一奖励组复用到代码例子。

## chapter_scan

- **日期与范围：** 2026-11-09；针对 GRPO 的组内 baseline、标准化优势、PPO 风格裁剪目标，以及该数学桥接和第十六册 reasoning/RLVR 主讲边界进行定向核验。
- **问题：** 第八章已有 MDP、policy gradient、GAE 和 PPO，是否缺少从组内奖励到优势信号的可复算连接；公开论文的定义能否作为一般实现规范；什么内容应交给第十六册。
- **来源：** Shao 等，DeepSeekMath arXiv:2402.03300v3，§§4.1.1–4.1.3；DeepSeek-AI，DeepSeek-R1 arXiv:2501.12948v2，§2.1、式（1）—（3）。已阅读两篇原始论文对应方法段与公式：[DeepSeekMath](https://arxiv.org/html/2402.03300v3)、[DeepSeek-R1 v2](https://arxiv.org/html/2501.12948v2)。定义和证据边界另见[策略优化研究记录](../../research/rlhf-preference-and-policy-optimization-2026.md)。

| 候选内容 | 处理与位置 | 边界/理由 |
| --- | --- | --- |
| 组内奖励构造相对优势 | 纳入 §8.8.6；增加四回答总体标准差手算，并连接已有 PPO clipped surrogate | 论文支持同题回答组的均值/标准差归一化；正文把总体标准差明确列为教学手算约定，不推断所有训练器采用相同 `ddof`、epsilon 或 reduction |
| 全套 RLVR、reasoning 训练与零奖励组诊断 | 链接第十六册第十七章 §§17.7–17.9 | 数学册只建立算法接口；reward/verifier 生命周期、诊断和蒸馏由第十六册完整主讲 |
| DAPO、GSPO、DrGRPO 等后续目标和归约变体 | 不在本章展开；保留到研究记录和对应专题章节 | 其 clipping、长度处理、采样和 loss reduction 是方法/实现特定变化；不宜压成单一“GRPO 标准公式” |

扫描不是对所有 RLHF/GRPO 实现的穷尽审查。正文只把论文能支持的版本化数学描述写成事实，并将跨实现差异保留为核验边界。

## length_status

计数按连续中文解释性正文中的 CJK 汉字字符统计，排除标题、代码/公式/Mermaid 围栏、表格、独立列表、URL/链接文字、行内代码和行内公式。连续小节按同一教学链分组；练习、资料和章末回顾不单独算作核心教学单元。

| 教学单元 | 类型 | 实计汉字 | 下限 | 状态与分组理由 |
| --- | --- | ---: | ---: | --- |
| §§8.1–8.5 | 核心 | 2,723 | 1,800 | 通过。从 SFT/RL 对象映射、MDP 与终止约定推进到 return、value、advantage 和 Bellman 关系。 |
| §§8.6–8.9 | 核心 | 2,149 | 1,800 | 通过。从 log-derivative 和 baseline 推进到 GAE、PPO clipping、GRPO 组内优势与 KL 边界。 |
| §§8.10–8.17 | 核心 | 2,274 | 1,800 | 通过。从偏好打分到 RLHF/DPO，再分析目标错配、verifier、数据分布、评估和合同助手案例。 |
| §§8.18–8.22 | 辅助 | 1,333 | 1,000 | 通过。把可运行教学代码、实验复查方法、练习和工程回扣作为上述算法链的实践与迁移入口；代码和公式本身不计字数。 |

## example_status

- **终止回报：** $[r_0,r_1,r_2]=[0,0.2,1]$、$\gamma=0.9$，逐步得到 $G_2=1$、$G_1=1.1$、$G_0=0.99$。终止奖励在索引 $T-1$，与三步轨迹和代码长度一致。
- **PPO clipping：** old/new probability 为 0.25/0.40，$A=0.8$、$\epsilon=0.2$，ratio 为 1.6；未裁剪项 1.28、裁剪项 0.96，surrogate 取 0.96。
- **GRPO：** 同题组奖励 $[0,0,1,1]$，$\mu_G=0.5$；采用总体标准差时 $\sigma_G=0.5$，优势为 $[-1,-1,1,1]$。正文再以正优势、负优势分别代入 PPO clipped surrogate，展示组内排序信号和局部更新保护的衔接。
- **其他已保留例子：** GAE 终止 mask、KL 正则、Bradley–Terry 偏好损失、DPO margin、reward-human/task gap、验证器结果、成本与合同助手投机案例。数字均为教学构造，不作论文或生产数据陈述。

## code_status

- **位置/状态：** §8.18 唯一独立 Python 代码块；正文标记为“已验证（2026-11-09；Python 3.12.9 标准库，无第三方依赖）”。
- **变化：** 新增 `group_relative_advantages`，复用正文 $[0,0,1,1]$ 组内手算；统一拒绝 bool/非有限数，检查容器形状、bootstrap 长度、终止 mask、分布归一化和概率边界；KL 改用 log 差避免概率比中间量溢出。最终路径复读又发现 PPO 有限输入乘积可能溢出，现显式检查 surrogate，并用缩放后的 `math.fsum` 计算 batch 均值，避免有限的大数在累加阶段溢出。
- **运行：** 最终版本从 Markdown 提取后，在 `.tmp/llm-book-cpu-check/bin/python`（Python 3.12.9）执行。完整代码块 stdout 与正文预期逐字一致；代码内检查项全部为 True。
- **边界检查：** 初次完整审计的 28 项覆盖 GRPO 组内均值/标准差、同分组零优势、单样本组、bool/NaN、均值/方差溢出；return/GAE 的 bool、非有限和长度/mask；PPO 的 bool、零旧概率和 ratio 溢出；KL 的概率归一化、零支持、tuple 输入；sigmoid、reward score 和 DPO 参数校验。最终路径复读后重新检查改动的 PPO 函数：bool advantage、零旧概率、ratio 溢出、surrogate 溢出均按预期拒绝；两个 $10^{308}$ 有限 surrogate 的 batch 均值仍为有限值，五项检查通过。其他函数和依赖没有变化，沿用原验证记录。
- **教学范围：** 函数接受 list/tuple；`dones=True` 表示自然终止。没有实现截断 bootstrap、padding mask、变长 batch、完整词表分布或反向传播，已在代码前明示。

## formula_status

- 将 $T$ 定义为 action/reward 步数，索引为 $t=0,\ldots,T-1$；response token $y_{t+1}$ 对应零基 action $a_t$，终止奖励置于 $T-1$。discounted return、序列概率、KL sampled log-ratio、GAE 数组长度、bootstrap value 和 `dones` 长度现采用同一约定。
- 全章 70 个 display-math 围栏的花括号均平衡；146 个 tilde 围栏标记配成 73 对；围栏外行内 `$` 符号 58 个，成对。
- 手工复算 return、PPO 正/负 advantage 的 min 方向、GRPO 总体标准差和归一化优势；未运行 MathJax 渲染器。

## visual_status

- §8.1 增加 Mermaid rollout—评分—优势—策略更新回路；明确区分继续生成和轨迹结束。另保留 MDP 对象映射表、评估矩阵和合同助手案例表。
- Mermaid 源码按节点加引号和管道箭头标签检查；本地未找到 `mmdc` 或其他 Mermaid CLI，且当前工作区修改未发布，尚未完成实际渲染核验。

## link_status

- 本章 1 个本地跨册链接指向第十六册第十七章，目标文件存在；第十三册目录 20 个本地章节链接全部存在。
- DeepSeekMath v3 与 DeepSeek-R1 v2 原始论文链接已直接打开并核对对应算法章节；未逐个探测其余所有外部链接 HTTP 状态。

## sync_status

- 已同步[第十三册简介](../../book-13-math-foundations/简介.md)与[目录](../../book-13-math-foundations/目录.md)第八章摘要及 GRPO 深入专题的第十六册交接。
- §8.20 新增 GRPO 优势练习。`PAPERS.md` 已有 DeepSeekMath、DeepSeek-R1 和 GRPO 变体条目；术语表已有 GRPO 定义，因此未重复新增索引条目。研究事实仍记录在现有 `research/rlhf-preference-and-policy-optimization-2026.md`；精修决定只记在本进度记录和架构审计。

## open_risks

1. 新增 Mermaid 图尚未实际渲染；待本地 Mermaid CLI 或可访问的目标页面可用时复核。
2. 未逐项访问全部旧外链；两篇 GRPO 一手来源已核对，其他参考资料保留原状。
3. GRPO 的标准差分母、数值稳定项、token/sequence reduction、长度处理和 KL estimator 存在实现差异；本章总体标准差仅为明确的教学 convention。
4. `PAPERS.md` 既有 28 个本地缺失目标仍待集中清理，与本章新增链接无关。第九、十章主路径复读待继续。
