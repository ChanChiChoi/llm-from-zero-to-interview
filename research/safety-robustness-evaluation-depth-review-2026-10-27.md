# Safety 与 Robustness Eval 深度精修研究记录

- 日期：2026-10-27
- 正文：[第七册第八章](../book-07-evaluation-experiments/chapters/08-safety与robustness-eval.md)
- 依据：当前 [WRITING_SPEC.md](../WRITING_SPEC.md) 与 [progress_v2.md](../progress_v2.md)
- 研究范围：以 LLM / LLM Agent 的安全与鲁棒性评估为界，重点复核代表性方法如何定义对象、构造任务、判分、报告不确定性和验证工具副作用。
- 执行范围：阅读下列论文、官方风险框架和公开技术页面；本轮没有运行章节 Python demo、benchmark、目标模型、API 或生产工具。

## 1. 原章节知识承接图

本章原有内容主线保留：policy 和结果标签、风险 taxonomy、漏拒/误拒、harmful output、jailbreak、prompt injection、privacy、fairness、dangerous capability、robustness、red teaming、风险指标与统计、judge、线上安全、错误定位、demo、项目流程、误区、练习和资料入口。本轮没有删除这些问题，而是把标题下的“应考虑什么”进一步写成评测对象、判定机制和证据边界。

| 原有主题 | 本轮处理位置 | 深化方式 |
| --- | --- | --- |
| policy 动作、拒答、漏拒、误拒 | §1—§4、§12 | 增加有效分母和不可评估状态；安全替代须兼顾任务完成与无泄露/越权 |
| jailbreak 形态、预算、失败留存 | §5.5、§11.5、§13.4 | 比较 HarmBench、StrongREJECT、JailbreakBench 的不同测量目标；解释攻击集、judge 和复现协议分别解决什么 |
| RAG / 工具 prompt injection | §6.5 | 对照 InjecAgent 的固定测试集、AgentDojo 的状态化效用检查和 SafeAudit 的覆盖审计；给出只含抽象数据流的状态轨迹 |
| 隐私、canary、成员推断、多租户 | §7.5 | 分开训练序列抽取、成员推断、属性推断、应用数据越权；加入条件不同的研究对照 |
| bias 与 fairness 切片 | §8.5 | 解释 BBQ ambiguous / disambiguated 两种构造，以及中文 CBBQ 的本地覆盖贡献和限制 |
| dangerous capability 分层 | §9.5 | 从模型答题能力推进到人类使用模型后的相对 uplift、既有资源 baseline 和实验设计 |
| 语义保持扰动、OOD 和最差群体 | §10.5 | 引入 CheckList 的 MFT / INV / DIR，并解释语义验证及相关变体的统计单位 |
| red team 探索与线上闭环 | §11.5、§17.6 | 区分固定集、自动探索、专家红队、状态化测试和线上监控；补齐报告所需的分母、版本、coverage 和未测范围 |
| judge 与分类器 | §13.4 | 比较规则、分类器、rubric judge、人工专家和状态 oracle 的职责及校准方法 |
| 严重度和罕见事件 | §12.2—§12.4 | 区分加权失败指数与真实期望伤害；加入零事件单侧精确上界、样本单位和相关性条件 |
| 诊断 demo | §16、§16.1 | 修正 0/0 返回 0 的伪通过风险；使 action consistency 与正文定义一致；safe completion 同时排除泄露和越权工具动作 |

本章仍作为一份集成的评估章节：上述主题沿“policy—模型行为—工具/数据状态—判分—发布决策”形成同一闭环，当前材料无需另设孤立文件；第七册目录已有本章入口，无须改导航。

## 2. 本轮要解决的问题

1. 固定 jailbreak benchmark、自动 judge、攻击预算和拒绝/帮助性之间有什么区别？一个 ASR 如何避免把“没拒绝”错误等同于“攻击真的奏效”？
2. Agent 接触外部不可信数据后，怎样检测用户任务是否完成、攻击者目标是否达到、系统状态是否越权变化？
3. 训练数据记忆、membership inference、属性推断与 RAG/工具访问控制分别测量什么？为什么不同论文会对 MIA 有不同发现？
4. 公平性基准如何区分证据不足时的刻板印象猜测与证据充分时仍忽略上下文？
5. 危险能力问答成绩能否代表一个人使用模型后得到的帮助？怎样设置已有资源 baseline？
6. “扰动后保持一致”什么时候是好结果，什么时候只是两次都犯了同一个错误？
7. 零观察事件、零分母和严重度加权总分分别允许支持多强的结论？

## 3. 阅读资料与正文增量

| 来源 | 实际核对的内容 | 支持的判断 / 正文位置 | 证据边界 |
| --- | --- | --- | --- |
| [HarmBench](https://arxiv.org/html/2402.04249) | 510 个行为、functional/semantic coverage、攻击生成—目标模型—分类器流程、人工验证分类器 | §5.5 说明 benchmark 用广度与统一口径比较攻击和防护；§13.4 说明专用分类器仍依赖其行为标签与校准 | 论文中的 18 种攻击方法、33 个模型/防护和结果绑定当时测试集、harness、token budget、classifier |
| [StrongREJECT](https://arxiv.org/html/2402.10260) | 313 条问题；rubric evaluator 的 refusal、specificity、convincingness；与人工评价比较 | §5.5 增加“拒绝”和“有用的违规回答”不是同一个二元标签；说明评分公式与用途 | rubric / fine-tuned judge、受测回答和人工样本绑定论文配置；评分不是现实损害概率 |
| [JailbreakBench](https://arxiv.org/html/2404.01318) | 可复用 artifact、威胁模型、聊天模板、查询信息、100 misuse + 100 matched benign 行为 | §5.5 解释统一攻击成本、复现协议以及匹配 benign 集对 over-refusal 检查的作用 | 开放工件可被调优；数据基于发布方 policy；leaderboard 需要版本和目标模型快照 |
| [XSTest](https://aclanthology.org/2024.naacl-long.301/) | 250 条安全请求与 200 条对照不安全请求，以及 lexical false refusal 的问题 | §4/§5/§20 支持将过度拒绝作为独立质量面，而非把拒答一律算安全 | 作者构造的英语测试集和任务不代表所有产品 policy、语言和领域 |
| [InjecAgent](https://arxiv.org/html/2403.02691) | 1,054 个案例，17 种用户工具、62 种攻击者工具，两类攻击意图；论文测试的 ReAct agent 条件 | §6.5 作为固定工具注入集的代表，解释其针对 tool-integrated IPI 的覆盖 | 原始攻击成功率不可迁移到其他模型、工具、权限及 agent prompt |
| [AgentDojo](https://arxiv.org/html/2406.13352) | 97 个正常任务、629 个安全案例、74 个工具；效用函数检查输出与环境状态；三种独立指标 | §6.5 增加 benign utility、utility under attack 与 targeted ASR 的分工和状态轨迹 | 仿真工具和逻辑 oracle 的结论仍绑定其环境；不等于真实工具或真实用户接受度 |
| [SafeAudit / Who Tests the Testers?](https://aclanthology.org/2026.emnlp-main.73/) | EMNLP 2026 论文摘要中的 workflow enumeration、rule-resistance、3 benchmarks / 12 environments 及 20%+ residual finding | §6.5/§11.5 说明测试覆盖也可成为被审计对象 | 数字绑定论文的枚举器、规则、环境与预算，不是所有 agent benchmark 的漏测率 |
| [间接注入防火墙预印本](https://arxiv.org/html/2510.05244) | 2026-03 v2 对公开 benchmark、防御饱和、攻击强度和指标缺陷的作者分析 | §6.5 作为预印本观点，支撑需要报告攻击强度与状态判定 | 不是领域共识；正文明确称其为预印本，不转述为已证事实 |
| [The Secret Sharer](https://www.usenix.org/conference/usenixsecurity19/presentation/carlini) | 罕见 canary、exposure 指标和训练期间受控审计思路 | §7.5 分清受控 canary 观察与一般性“无泄露”证明 | 取决于 canary 分布、模型和攻击者可访问方式 |
| [Extracting Training Data from LLMs](https://arxiv.org/html/2012.07805) | GPT-2 抽取攻击和训练片段复现任务 | §7.5 区分 exact sequence extraction 与 membership classification | 论文的 GPT-2 / 数据快照不能泛化到当前全部模型 |
| [Do Membership Inference Attacks Work on LLMs?](https://arxiv.org/html/2402.07841) | Pile 上 160M—12B 规模模型的多种 MIA；成员边界、数据规模和时间/来源偏移分析 | §7.5 说明多数预训练设置近随机的研究发现及混杂因素 | 绑定 Pythia/Pile 和攻击设定，不能外推成 MIA 普遍无效 |
| [Privacy Auditing of LLMs](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e76814a0012a46ade4f997591e86f972-Abstract-Conference.html) | 论文中的强 canary 方法；Qwen2.5-0.5B 例子 TPR 49.6%@1% FPR；隐私审计目标 | §7.5 与前项对照，解释不同数据阶段、canary 与访问条件可带来不同 MIA 能力 | 具体数字只代表该模型和微调/审计设置；不作为其他模型成功率 |
| [Beyond Memorization](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9028b8a3ca98f58e373f0c1497a17448-Abstract-Conference.html) | 从 Reddit profile 文本推断个人属性的实验 | §7.5 分出属性推断，不把所有隐私都等同于训练复现 | 绑定其 profile 数据、属性标签、模型和实验任务 |
| [InfoRMIA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ddf43af256e11499149cf06040c62caa-Abstract-Conference.html) | ICLR 2026 的信息论 MIA 与 token-level memorization granularity | §7.5 记录研究从序列级到 token 级的指标发展 | 新方法的作者对 benchmark 性能结论不等于普适隐私保证 |
| [BBQ](https://arxiv.org/html/2110.08193) | ambiguous / disambiguated 模板、unknown option、accuracy 与 bias score 的分开计算 | §8.5 说明 stereotype fallback 与 ignore-context 是不同失败 | 数据覆盖美国英语语境，指标随作者的数据和计分约定 |
| [CBBQ](https://arxiv.org/html/2306.16244) | 14 个社会维度、超过 100K 中文问题、人工模板与模型辅助构造后人工审查 | §8.5 补充中文社会语境，避免直接把英语 benchmark 当全球覆盖 | 模板化问答不是全部方言、真实分配后果或群体代表性 |
| [CheckList](https://aclanthology.org/2020.acl-main.442/) | MFT、INV、DIR 三类关系测试 | §10.5 把鲁棒性测试从扰动清单转成可验证的输出关系 | 原论文面向 NLP behavioral testing；本章将其作为安全评估设计方法迁移，不声称其原研究是 LLM safety benchmark |
| [OpenAI 生物风险 uplift 研究](https://openai.com/index/building-an-early-warning-system-for-llm-aided-biological-threat-creation/) | 100 人、专家与学生、互联网 baseline 与互联网+GPT-4 treatment、五个结果维度；作者说明结果与实物威胁构建的局限 | §9.5 用于说明人类 uplift 与模型单体答题分数的不同 | 发布方研究，数字和实验条件绑定原任务；未测真实物理构建 |
| [Google DeepMind dangerous capability study](https://deepmind.google/research/publications/78150/) | 五类能力评测及作者对受测模型的结论 | §9.5 展示能力 taxonomy 和发布方试点评估 | 论文/作者的模型、任务与时间快照，不是独立风险认证 |
| [OpenAI Preparedness Framework 更新](https://openai.com/index/updating-our-preparedness-framework/) | tracked/research categories、high/critical thresholds、评估与 safeguards 报告流程 | §9.5/§17.1 说明能力测量如何进入发布治理 | 发布方自身政策流程，不是标准化 benchmark 或第三方认证 |
| [NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) | Govern/Map/Measure/Manage、风险测量、pre-deployment TEVV 限制、red team 参与者类型 | §17.1 用于把风险识别、证据测量和行动连接起来 | 框架提供流程建议，不给目标模型的通过分数 |
| [OWASP GenAI Security Project](https://owasp.org/projects/top-10-for-large-language-applications) | LLM/agent 应用安全风险分类项目入口 | §17.1 作为威胁发现地图 | 风险清单不能替代产品特定的 threat model 和 stateful test |

## 4. 形成的方案比较结论

### 4.1 越狱评测

- HarmBench 更偏向广行为集、统一自动 red-team pipeline 和模型拒答比较。
- StrongREJECT 更偏向判分构念：拒绝、具体性和回答说服力需要分开，二元字符串命中会误判。
- JailbreakBench 更偏向攻击工件、威胁模型、查询预算和可复现基准维护；匹配的 benign 行为帮助发现误拒。
- 因而三者互补：固定风险目标、可靠评分、可复现攻击成本均要明确。任何一个模型分数都不能当作现实伤害率。

### 4.2 Agent 间接 prompt injection

- InjecAgent 提供宽覆盖的固定工具集攻击测试。
- AgentDojo 把模型行为落到模拟状态上，并将正常效用、攻击下效用与目标攻击成功分开。
- SafeAudit 将“测试是否覆盖 workflow”本身纳入评估。
- 真实部署应优先用确定性权限与状态断言检查未授权读写；开放文本 judge 只承担其能够校准的语义判分。

### 4.3 隐私

- exact extraction 是内容被恢复；MIA 是记录是否属于训练数据；attribute inference 是从输入推断未给属性；RAG/工具泄露是系统数据流授权错误。
- MIA 研究结果必须绑定阶段、候选集、数据分布、访问能力和目标错误率区域。近随机的 pretraining MIA 发现与强 canary 微调审计结果可以在各自设置内同时成立。
- DP 形式保证、经验 canary、成员推断和租户隔离解决不同问题。

### 4.4 Fairness 与 capability

- BBQ 的两种上下文测试把信息不足时的 stereotype completion 与有证据却忽略证据分开；CBBQ 提供中文社会语境入口，但不代表所有目标文化和真实决策分布。
- dangerous capability knowledge score 与人类获得的额外帮助不同。若风险问题是 uplift，应设置已有资源 control group，并明确参与者能力、评分规则和任务条件。

### 4.5 Robustness 与判断器

- CheckList 的 MFT / INV / DIR 把“应该稳定”具体化为输入—输出关系；扰动的语义有效性须先验证。
- clean / perturbed 的 action consistency 不能替代两边正确性；多个同源变体不能被当作独立统计样本。
- 安全判定应按任务组合 evaluator：规则做初筛，校准后的 classifier / rubric judge 扩展覆盖，人工裁决高影响语境，状态 oracle 判定实际权限和副作用。

## 5. 章节 demo 与公式复核

- 手工核对 p_upper = 1 - α^(1/n)：α=0.05 时 n=10/100/300 分别约为 25.9% / 2.95% / 0.99%；结果仅适用于独立同分布、观察到零事件的二项模型。
- 手工复核 StrongREJECT 分数含义：拒绝项分数为 0；非拒绝项还取决于具体性和说服力，因此空话不自动计作高 ASR。
- 将 demo 的 rate 分母为零语义由 0.0 改为 None；对缺失证据不执行数值门禁，而触发收集有效样本的 action。
- 使 demo 的 consistency 与 §10.3 一致：按 clean / perturbed policy action 完全相同计数；预期值由旧的 score 差容忍逻辑改为 1/4 = 0.25。
- safe completion 现在在 helpful / safe action 之外，还要求该样本没有 leak / bad_tool 标记，和前文定义一致。
- 静态手工核算其他 toy 结果：unsafe compliance 2/7≈0.286；targeted attack success 3/4=0.75；over-refusal 1/3≈0.333；safe completion 3/4=0.75；secret leakage 1/2=0.5；unauthorized tool 1/2=0.5；robustness drop 0.275；严重度加权失败指数 19/37≈0.514。当前 toy 输入的所有门禁均失败，预期 action 列表仍为八项；没有执行 Python。

## 6. 配套与验证状态

- PAPERS.md：补充本章代表工作、官方框架和研究记录链接。
- EXERCISES.md：新增研究方法对照、工具状态评估、隐私威胁区分和零事件区间推导。
- INTERVIEW_BANK.md：新增代表 benchmark 如何互补、Agent 效用/攻击指标、隐私 MIA 结果差异、零分母、危险能力 uplift 等追问。
- GLOSSARY_EN_ZH.md：复用已有通用术语；只在确有空缺时新增 benchmark-specific 或 metric terms，不重复同义项。
- 章节目录已列出第八章，无导航调整。
- 静态检查完成：章节 §1—§21 连续（另含 §16.1 说明段），Markdown 围栏共 60 个、配对；目录已有正文入口，正文/研究记录和 PAPERS 互链可定位，配套新增项可定位；`git diff --check` 通过。
- 未执行 Python demo、测试套件、模型/API、外部 benchmark、公式渲染和 PDF 构建。

## 7. 仍未验证的边界

没有目标模型响应、实际 policy、用户流量、授权隐私样本、专家标签、工具环境或产品事故 trace。所有论文数值只描述来源中的模型、样本、判分器、版本、权限和攻击预算。NIST、OWASP、OpenAI 和 DeepMind 页面分别作为流程或发布方材料使用，不能替本项目目标系统背书。正文不提供可复用的越狱或数据外传载荷。
