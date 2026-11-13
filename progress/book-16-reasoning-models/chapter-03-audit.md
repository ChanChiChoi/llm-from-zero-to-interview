# 第三章《Self-Consistency 与采样》精修记录

审阅日期：2026-11-12  
状态：章节正文已精修；章级扫描、代码、公式、链接、视觉和简介摘要已同步。按用户要求不更新 PDF。

## 范围与章级扫描

- 通读本章 §§0.1–20、采样与归一化推导、聚合器、pass@k、成本/早停、verifier 组合、实验建议、安全、练习、资料索引和独立 Python 示例；对照第二章 CoT、第四章 verifier、第五章过程监督和第六章搜索边界。
- 复核原始 self-consistency 论文、HumanEval/pass@k 定义、nucleus sampling 原始论文、Transformers 固定版生成接口，以及 min-p 原始论文和独立证据复核。新发现的知识证据已存在于[解码采样与多路径推理研究记录](../../research/decoding-sampling-and-test-time-compute-2026.md)；本轮只在 progress 记录章节审阅和去向。

| 候选主题 | 来源与正文处置 |
|---|---|
| Self-consistency 的 sample-and-marginalize | 纳入本章核心；引用 Wang 等人的固定版本论文，并把效果限定在论文测试的模型、算术/常识任务及采样协议。候选相关性和共同错误作为主要失效边界。 |
| Temperature、top-k、top-p | 纳入逐 token 分布机制、合法参数域及记录要求；用 nucleus sampling 原论文解释开放式文本生成动机，用 Transformers v5.20.0 版本化文档说明接口字段，不把文本质量实验外推为 reasoning 准确率。 |
| Min-p、typical-p、top-h、epsilon/eta 等截断方式 | 在 §2.5 作为当前实现候选简述，不另立完整讲次；过滤阈值与分布统计不同，min-p 原始研究和独立复核对普遍优势结论不同，因此不当作默认更优方案。 |
| 答案抽取、等价类和归一化 | 纳入本章，因为 Self-Consistency 聚合的是解析后的答案；数值、单位、精度和条件必须按任务 schema 归并。教学 parser 明确限制在合成数据。 |
| 多数/最高频投票、加权投票与 verifier rerank | 纳入同一比较链；明确本文 majority vote 指最高频答案（可仅为相对多数），平票须显式处理；未校准 verifier 分数不能直接当正确概率。完整 verifier 训练和验证职责交第四章。 |
| Pass@k 与系统选择准确率 | 纳入核心区别；引用 HumanEval 的原始定义，手算 n=5、c=2、k=2 得 0.7，并用表格对比候选集合、聚合交付、rerank 和执行成功指标。 |
| 采样数量、成本、延迟与早停 | 纳入 token、单位成功资源、质量边际收益和风险；test-time scaling 总体预算关系交第七章，按反馈动态分配交第十八章。 |
| verifier 组合、过程监督和搜索 | 本章解释前置过滤/重排和依赖关系，完整验证器训练交第四章、步骤信号交第五章、节点评分与树搜索交第六章。 |
| 高风险候选与工具动作 | 保留多采样可能扩展危险候选的安全边界，不能靠多数投票执行外部副作用；治理和权限控制交第八、十七、二十及二十二册。 |

## 正文修改

- 整册连续复读（2026-11-13）发现 §3.1 多样性公式使用了评估题数 (N)，但原句未显式定义；已补明 (N) 与每题可解析候选数 (K_i)。可执行代码、输入输出合同和依赖未变化，沿用原代码验证记录。
- 为入口补入第二章链接；成本、动态早停、评估、verifier 与搜索等位置增加第四、六、七、十、十八章交接；结尾链接第四章。
- 在证据边界、self-consistency 机制、nucleus sampling 和 pass@k 公式处补充原始论文或固定版本官方文档引用。
- 修正四候选示例的解释，明确 42 是 2/4 的相对多数而非严格过半；在聚合公式旁统一定义 majority vote 的本章用法。
- 增加 pass@2 的手算步骤，并将 pass@k、majority accuracy、rerank accuracy 和 execution success 放进对照表，避免把“候选中存在正确答案”混同“系统实际交付正确答案”。
- 简述 min-p/典型性/熵过滤类参数作为候选，而不宣称其通用质量收益；来源索引绑定 Transformers v5.20.0 文档和访问日期。
- 修正资料索引：把 HumanEval 写为论文 “Evaluating Large Language Models Trained on Code”；删除将 arXiv:1904.09751 重复列为两个题名的条目；删除错误把 arXiv:1609.08144（GNMT 论文）标为文本控制生成论文的条目；修正 min-p 独立复核 v2 的正式题名；去除作者联网日志。
- 加强 §14 教学代码的 verifier 分数合同为有限的 [0,1] 置信度，避免 math.isfinite 把超大整数转换成浮点时报错；token 与 pass@k 整数仍排除 bool。把 cost_per_majority_correct 更名为 tokens_per_majority_correct，更新文中输出及解释。
- 更新第十六册简介中的第三章摘要；章节编号、目录顺序不变。

## 验收字段

- **length_status：** 本章作为一个连贯核心单元，剔除代码、公式围栏、表格、标题、练习/维护索引和空白后，解释性中文实计 7,056 汉字，高于 1,800 字下限。连续复读补足符号定义后计数增加 18 字。采样、答案处理、聚合、评估和预算小节共同组成一条机制链，不拆作重复的独立教程。
- **example_status：** 四候选示例将 42、36、40 的计数算为 2、1、1，说明最高频而非严格过半；pass@2 按 1-C(3,2)/C(5,2)=0.7 手算；distractor_math 的多数票 111（3 票）对 12（2 票），加权分数分别为 1.12 和 1.79，五候选共耗 267 token。数值与 §14 六个预制 toy 问题共享，不代表模型实测。
- **evidence：** self-consistency 论文固定 arXiv:2203.11171v4；HumanEval/pass@k 为 arXiv:2107.03374v2；nucleus sampling 为 arXiv:1904.09751v2；Transformers Generation API 固定 v5.20.0（访问日 2026-10-30）；min-p 原始研究 v8 与独立复核 v2 均贴近论断。论文成绩与 API 参数语义均限定于来源自身，不推成跨模型或生产保证。
- **formula_status：** 复核 20 个 display-math 公式和变量边界：候选集合、temperature/top-k/top-p、答案/轨迹多样性、独立模型下多数概率、数值匹配、等价类、majority/weighted vote、rerank、pass@k、成本、单位成功成本、边际收益、投票间距、自适应 K 与 Brier score。补明答案多样性式中的评估题数 (N) 与每题有效候选数 (K_i)；说明非空分布、采样数/候选数、分数域、评估分母、固定成本单位、成功样本分母和概率范围；公式花括号配对。
- **code_status：** §14 独立标准库示例于 2026-11-12 在 Python 3.12.9 执行，直接从 Markdown 提取，stdout 与预期块逐行完全一致。边界检查覆盖 bool/NaN/无穷/负值/越界/超大整数分数、bool/浮点 token、bool 与非法范围的 pass@k，以及 plurality、平票稳定性和 n=5/c=2/k=2 数值例子。只验证预先构造的五道题和聚合算术，不运行真实模型、API、代码沙箱或在线 verifier。
- **link_status：** 全章 23 个 Markdown 链接已扫描，本地相对链接均存在；来源链接已更换为论文固定版本、官方固定版文档和 ACL/arXiv 原始页面。未进行全量外链生产环境 crawl。
- **visual_status：** 新增 pass@k、聚合准确率、rerank 与执行成功的指标对照表；原有采样分布公式、候选轨迹和代码输出足以说明状态流，不需增加 Mermaid。
- **sync_status：** 已更新简介.md 的第三章摘要和章节内第 2、4、6、7、10、18 章交叉链接；目录条目及编号保持一致。未更新 PDF、plan.md、题库或其他全局索引。
- **open_risks：** min-p 质量/多样性优势存在论文间证据分歧，需绑定任务、模型、采样预算和评价协议；Transformers 字段按 v5.20.0 文档理解，其他版本和 runtime 需重新核对。多候选和 score 加权依赖 parser、verifier 与错误相关性，教学数据不能证明生产表现。

## 下一步

进入第四章《Verifier 与 Reward Model》，围绕验证对象、训练目标、排序/校准与候选选择，区分程序化结果 oracle 和 learned verifier/reward model。
