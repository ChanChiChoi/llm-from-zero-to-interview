# 第八册第七章 Interpretability 与 Mechanistic Interpretability 章级扫描

- `date`：2026-10-27。
- `scope`：依据当前 `WRITING_SPEC.md` 通读 `07-interpretability与mechanistic-interpretability.md` 的 19 个二级段，检查主线、独立讲解单元、公式、数值例、代码、视觉表达和安全边界；定向阅读 2025-10 至 2026-10 的近期研究，并补看仍影响当前方法判断的 2025 基准与 Anthropic 原始案例。本记录是结构/研究扫描，不代表正文已精修。
- `chapter_role`：本章解释可解释性证据如何从相关线索推进到局部因果机制假设，重点是 feature/circuit、attention head、activation patching/ablation、SAE 和把机制信号安全地用于调试。第六章负责红队与危险能力评估；第八章负责表征干预和 steering；第九章负责模型编辑与 unlearning；第十章负责记忆、隐私和水印；第十三章负责自动化评估闭环。第七章讲机制定位与证据强度，不把单个 feature 直接等同于安全控制、完整模型理解或安全保证。

## 结构与内容去向

- `structure_read`：开篇先提出“行为正确不代表理解机制”，再按 attribution → representation/probing → causal intervention → mechanism/circuit 递进，之后展开 feature/neuron/circuit、attention head、patching、ablation/causal tracing、SAE，最后接入 safety、grokking、项目工作流、局限、未来、综合案例、资料、练习和 demo。主线明确且已有多处反例与边界说明。需关注 §4–§8 方法篇与 §8.9 指标群之间的解释密度，以及 §9/§11/§14 对安全用途与干预边界的轻度重复；建议保留 §14 作为端到端案例，压缩重述，统一回到“竞争假设—对照/干预—行为回归”的同一实验框架。
- `candidate_placement`：不增章。SAE 评估对照和 feature 解释的反例搜索并入 §8.8–§8.9；trigger 检测与 trigger 控制可能分离的安全案例并入 §9 或 §14，并连接第九章；formal circuit verification 作为 §13.4 的研究方向短述，注明目前证据以 vision 模型为主。CRISP 等参数级 unlearning 的算法细节交由第九章，本章只说明机制证据如何为编辑目标提供候选。
- `cross_chapter_boundary`：第八章专讲将表征方向用于控制/steering 的方法与副作用；第九章专讲持久参数修改、编辑和遗忘；本章中的 feature ablation、activation patching 是研究干预/因果检验，不应与部署时持续 steering 或参数更新混称。第十章的数据记忆/隐私结论也不能从某个 feature 的激活或可编辑性直接推出。

## 外部主题与证据去向

| 候选与来源 | 建议落点 | 证据范围和限制 |
| --- | --- | --- |
| [SAEBench](https://arxiv.org/abs/2503.09532) | §8.9/§15：把单一重构分数扩为多维评估，并比较 proxy 指标与解释/解耦/实际任务效用。 | 2025 ICML 研究在八类指标、200 余个 SAE 和八种架构上比较，发现 proxy 指标改善不能稳定推出实际表现改善；例如 Matryoshka SAE 在部分旧 proxy 指标下略差，但 feature disentanglement 指标更好。数据来自其模型、架构和任务，不构成所有 SAE 的统一排名。 |
| [MIB: A Mechanistic Interpretability Benchmark](https://proceedings.mlr.press/v267/mueller25a.html) | §4.7/§8.9：增加可比较的 circuit localization 与 causal-variable localization 评测；把“SAE feature 更可解释”与“比神经元更能定位因果变量”分开。 | 基准覆盖两个 track、四类任务和五个模型，使用任务特定高层因果模型和反事实；作者发现部分 circuit localization 方法更好，而在其 causal-variable 任务上 SAE features 没有优于普通 hidden dimensions。作者指出 ground-truth completeness 难以评估、任务/高层因果图可能不完备，且希望扩展到其它模态；不能据此概括全部任务或排除 SAE 在其它用途的价值。 |
| [Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?](https://arxiv.org/abs/2602.14111) | §8.6–§8.9：用冻结随机方向/激活基线说明，高重构保真、自动可解释性分数或单一 feature editing 分数不足以证明发现了真实机制；加入匹配的 random/neuron baseline。 | 2026 arXiv 预印本。在作者的合成重尾特征设置中，报告解释方差约 0.71 但只恢复约 9% ground-truth features；在 Gemma-2-2B、Llama-3-8B 指定层和两类 SAE 上，冻结组件基线在若干指标接近训练 SAE。结论受合成分布、模型层、SAE 变体、指标与基线设计约束，宜作为重要反例而非“SAE 全部无效”的共识。 |
| [Revising and Falsifying Sparse Autoencoder Feature Explanations](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5b6ce2529a79862a2d8ac57ac0e01d90-Abstract-Conference.html) | §8.8：解释不能只用高激活正例自洽；增加相似负例，检查说明能否排除边界近邻，并保留分解特征和不确定部分。 | NeurIPS 2025 主会论文，提出相似度检索近邻负例、结构化解释和迭代树式解释。研究改进的是其 feature explanation 生成/评估流程，不意味着自动文本解释已成为 feature 的语义真值。 |
| [LLM Forensics: Where Do Backdoors Hide?](https://arxiv.org/abs/2609.07746) | §9.2/§9.4/§14：用安全相关案例说明“能检测 trigger 的 feature”可能不控制触发后的行为；需要把 feature label、因果干预和端到端行为回归分开。 | Findings of EMNLP 2026；在 1B/8B 模型的无害英转法/德 trigger backdoor 上实验。论文报告 attention/MLP SAE feature 可高 F1 检测触发，却常不能靠消融抑制或靠激活诱发行为；部分 residual-stream feature 才呈现控制效应。此为合成无害 backdoor，不是恶意供应链事件或真实危险 payload 的证据。 |
| [Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b5afe13494c825089b1e3944fdaba212-Abstract-Conference.html) | §13.4：作为形式化输入域稳健、robust patching 和 circuit minimality 的研究方向，说明“对哪些扰动范围给出保证”比笼统声称“理解模型”更精确。 | ICLR 2026 论文；实验在多个 vision models 上验证，不是 LLM 或安全机制的直接证据。可作为跨领域方法进展短述，不能宣称已为前沿语言模型提供可扩展保证。 |
| Anthropic [Tracing the thoughts of a large language model](https://www.anthropic.com/research/tracing-thoughts-language-model)、[On the biology of a large language model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) 和[开源 circuit tracing 工具](https://www.anthropic.com/research/open-source-circuit-tracing) | §9/§11/§15：补充真实 LLM 上的机制研究示范、开源 attribution graph 工具和可复查的实验形态。 | Claude 3.5 Haiku 的具体短任务案例包含跨语言表示、诗歌规划、迎合错误提示、拒答/幻觉与 jailbreak 生命周期；机构页面明确承认方法只覆盖部分计算、有工具伪影，复杂 prompt 仍需大量人工分析。工具开源文章介绍 Gemma-2-2B / Llama-3.2-1B 等支持范围，不能由研究演示推断模型整体透明或安全。 |
| [CRISP: Persistent Concept Unlearning via Sparse Autoencoders](https://aclanthology.org/2026.acl-long.82/) | 不在本章展开；交由第九章讲参数级持久 unlearning，本章只交叉链接 SAE 目标定位与行为效用评估。 | ACL 2026 论文在两种 LLM 和 WMDP 任务上比较方法，仍绑定其模型、数据、目标概念和能力保留指标；不等同于生产遗忘保证或抵御所有再学习/恢复攻击。 |

## 篇幅、公式、数值例和代码

- `length_scan`：粗计每个 `##` 段的汉字，排除标题、代码围栏和表格行（仅定位风险，未人工剥离引文/清单/公式说明）：§1 约 574、§2 142、§3 183、§4 920、§5 638、§6 769、§7 520、§8 1,725、§9 615、§10 482、§11 654、§12 1,073、§13 764、§14 1,426、§15 483、§16 93、§17 103、§18 173、§19 189。一个 Markdown 文件当前承担单篇课程，不能机械将每个小节都当独立单元；但 §8 SAE 主讲接近且粗计略低于核心下限，§14 综合案例超过辅助下限但若作为核心独立讲解仍不足。重组后按 WRITING_SPEC.md §3.3 确认教学单元和解释性字数，不用全篇总数或代码/公式过门禁。
- `formula_and_example_scan`：本章有多个 `$$` 公式围栏和 `\(...\)` 行内公式，需按当前规范迁移到 `$...$` 行内及 `math` 围栏，并检查指标量纲/定义域和分母接近零的数值稳定性。§6.5 已解释 patch 恢复分的分子、分母与零分母；§14 案例仍是定性描述。§18 demo 虽含完整小样本输出，但正文没有可跟算的手算轨迹。精修可沿用同一输入手算 `S_patch=(1.4-(-1))/(2-(-1))=2.4/3=0.8`：clean logit margin 为 2.0、corrupted 为 -1.0、patched 为 1.4；再解释它只表示该干预在这个 toy 上恢复了 80% 的 clean/corrupted gap，不能推出自然运行时机制充分或模型安全。
- `code_status`：§18 预期值经静态手算与代码路径一致：四个 patch score 为 0.4、0.2、0.8、0.7；ablation effects 为 0.5、0.8、1.6；path scores 为 0.75、约 0.583；六条 toy residual/SAE 样本得到 reconstruction fidelity 约 0.993，平均激活 feature 数为 1.0，三项 top-2 purity 均为 1.0，所有 toy thresholds 为 True。未运行 Python；当前代码块没有“已验证/静态审阅，未运行/待验证”状态标记。静态边界检查发现空 residuals 会令 `mean_vector` 访问 `rows[0]` 失败，空 patch/purity 集合也会在 `max`/`min` 失败；若精修宣称支持空切片，应显式返回 N/A 并让决策传播未知值，否则限定输入契约。实际改可执行行后需重新验证。

## 图表和验收门禁

- `visual_map`：章首已有 `text` 线性证据链，其他方法段使用若干 `text` 示意；没有 Mermaid。首选将“行为现象 → 归因/读出 → 干预 → 机制假设 → holdout 与行为回归”画成一张证据路径图，避免把一个 attention/SAE 图示误读成自动证明；SAE 编码/重构用结构示意或表格，patching 的 clean/corrupted/patched 数据流适合独立小图或轨迹表。若改为 Mermaid，必须使用标准 `mermaid` 围栏；中文、空格或标点节点标签优先加引号；带文字虚线箭头统一写成 `A -.->|局部因果线索| B`。代码围栏/语法/目标 GitHub 页面渲染都要核验。
- `completion_gate`：章级扫描及来源阅读完成；未精修正文、公式、代码标记和图表。书册级仍需第八至十六章扫描、内容去向/coverage map、视觉地图、章节入口一致性检查，再逐章修订并沿阅读路径复读。
- `open_risks`：公式格式旧；核心 SAE 解释可能需更多解释性正文；安全案例缺数值流程；SAE 指标易把重构/可读性误当作机制恢复；研究结果绑定有限模型、任务与反事实；代码没有验证状态和空切片保障；Mermaid 语法/渲染与全册关系图未验收。
- `validation`：完成 19 个二级段结构复读、静态章节篇幅诊断、SAE demo 关键输出手算及 primary-source 证据阅读；未运行本章 Python、未对公式或 Mermaid 做外部渲染、未构建 PDF，也未请求全章外链。

## 参考来源

1. Karvonen et al., [SAEBench](https://arxiv.org/abs/2503.09532), arXiv v4, 2025-06-04；Mueller et al., [MIB](https://proceedings.mlr.press/v267/mueller25a.html), arXiv, 2025-04-17。
2. Ma et al., [Revising and Falsifying Sparse Autoencoder Feature Explanations](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5b6ce2529a79862a2d8ac57ac0e01d90-Abstract-Conference.html), NeurIPS 2025；Korznikov et al., [Sanity Checks for Sparse Autoencoders](https://arxiv.org/abs/2602.14111), arXiv, 2026-02-15。
3. Antoun et al., [LLM Forensics: Where Do Backdoors Hide?](https://arxiv.org/abs/2609.07746), arXiv v2, 2026-09-23, accepted Findings of EMNLP 2026；Hadad et al., [Formal Mechanistic Interpretability](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b5afe13494c825089b1e3944fdaba212-Abstract-Conference.html), ICLR 2026。
4. Anthropic, [Tracing the thoughts of a large language model](https://www.anthropic.com/research/tracing-thoughts-language-model), 2025-03-27；[On the biology of a large language model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)；[Open-sourcing circuit tracing tools](https://www.anthropic.com/research/open-source-circuit-tracing), 2025-05-29。
5. Ashuach et al., [CRISP: Persistent Concept Unlearning via Sparse Autoencoders](https://aclanthology.org/2026.acl-long.82/), ACL 2026。

- `status`：第七章全文结构和章级主题扫描完成；正文尚未修改，demo 未运行，公式迁移和 GitHub Mermaid 渲染未核验。下一步扫描第八章，并将 SAE/steering/unlearning 的主讲边界纳入第八册 coverage map。
