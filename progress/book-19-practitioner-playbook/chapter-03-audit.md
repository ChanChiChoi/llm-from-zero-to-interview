# 第十九册第三章《Tokenizer 与格式坑》精修审计

精修日期：2026-11-17。状态：首轮正文修改、章级扫描和配套同步完成；下一入口为第四章。

## 范围与读者路径

- **scope**：检查 tokenizer、chat template、special token、BOS/EOS/PAD、assistant-only 监督、工具/结构化输出、截断、多语言 token 预算、多模态映射、checkpoint 协议及其训练/推理故障；承接第二章的数据记录与分布诊断，交接到第四章训练异常和第六章 SFT/对齐行为诊断。
- **reader_path**：读者从一条原始多轮消息开始，依次检查模板渲染、token ID、角色边界、attention mask、loss mask、causal shift、结束边界、截断和模型专属媒体映射；再把可复现差异和版本写入事故记录。
- **中心问题与结果**：消息经分词、模板、标签、截断与多模态拼接后，是否仍符合目标模型训练和推理协议？读者能定位序列中的第一次分歧，分清模型看见什么、哪些位置成为监督目标以及生成器如何判停。
- **章节主线**：序列对象 → tokenizer 计算单位 → 控制 token 与结束语义 → padding/mask/shift → 模板与目标 → 工具和结构化输出 → 截断 → 语言预算和媒体映射 → 版本/最小复现 → 指标/事故 → toy 审计与回归。
- **视觉机会**：用六 token 对齐表直接展示输出位置与目标位置的 shift；用审计脚本打印 token IDs、attention mask 和 labels。未另加 Mermaid，关系表足以核对该例。

## 章级扫描与证据

- **扫描日期和范围**：2026-11-17。定向核对 BPE/SentencePiece 与 tokenizer 评估、Transformer v5.17 chat template/tokenizer 文档、TRL v0.29 SFT 字段、Transformers v5.17 固定 collator 与 GPT-2 实现、OpenAI 当前 Function Calling/Structured Outputs 文档、LLaVA/BLIP-2 原论文、PyTorch checkpoint 教程、RFC 3629 和 ACL 2026 tokenizer 研究。
- **候选主题与去向**：将 SentencePiece 从算法并列项改为可实现 BPE/Unigram 的工具；将 token 数限定为长度/成本诊断而非普遍性能指标；区分消息结束、轮次结束和工具交接；补上 attention mask、loss mask、causal shift 三者职责；将 assistant-only loss 的目标位置绑定训练合同；按模型规定的媒体映射检查 placeholder/features；为结构化输出补 schema 与业务规则边界。
- **固定实现案例**：Transformers v5.17 collator 的 `mlm=False` 路径按 `pad_token_id` 相等屏蔽 labels；同版 GPT-2 文档说明模型头内部 shift labels。此证据只支持对应固定版本和路径，不推广到所有框架/模型。共用 PAD/EOS 是否屏蔽真实 EOS 取决于实际 collator 与 mask 来源。
- **研究去向**：tokenizer、UTF-8、LLaVA/BLIP-2 与 ACL 2026 事实记入[Tokenizer 与聊天输入协议研究记录](../../research/tokenization-input-contract-2026.md)；训练标签、collator 行为与版本边界记入[因果 LM 训练与聊天模板研究记录](../../research/causal-lm-training-and-chat-template-contracts-2026.md)；OpenAI Structured Outputs 的动态 API 边界记入[工具调用与安全来源记录](../../research/agent-tool-calling-security-2026.md)。本审计只记录纳入决定和正文修改。
- **不纳入/不扩章理由**：新证据修正原章节已有的 token 统计、训练标签和媒体对齐边界，没有发现需要新增独立主题或改变章序。模型/tokenizer 算法推导回链第二册，训练数据字段与扩词表方法回链第五册；本章聚焦故障定位。
- **证据边界**：ACL 2026 研究结论限于论文覆盖的模型、语言和任务；HF/TRL 与 OpenAI 文档绑定各自版本/API；LLaVA/BLIP-2 只是两种架构示例；玩具 tokenizer 和一对一媒体关系不代表真实模型协议。

## 原有知识覆盖与去向

| 原稿内容 | 精修后的去向与处理理由 |
| --- | --- |
| 四层序列对象与版本化序列协议 | 保留于 3.1；将示例改成有效 JSON，并把“不一致即漂移”改为协议字段不兼容才构成漂移。 |
| tokenizer 粒度、token budget、多语言切片 | 保留于 3.2、3.9；分清 BPE/Unigram、SentencePiece 工具和 byte-level 表示；明确 token 数不是普遍质量代理，字符分母固定为 Unicode 码点并可另报 UTF-8 字节比。 |
| BOS/EOS/PAD/UNK/角色/工具/媒体 token | 保留于 3.3—3.4；删除 PAD/EOS 共 ID 会必然提前停止的说法，改为依 collator 和停止配置核验；区分消息结束、assistant 轮次结束和工具交接。 |
| Chat template 和训练/推理差异 | 保留于 3.5；所有模板串示例标明是抽象伪格式，不对应具体模型，并链接固定版本 HF 文档。 |
| assistant-only loss 和 prompt 泄漏 | 保留于 3.6；把 `T_assistant` 与 `P` 定义为任务声明的位置集合，允许角色头/控制 token 是否成为目标由合同决定；分开 attention mask、loss mask 和 causal shift，并用六 token 表手算两个有效目标。 |
| 多轮、工具调用和结构化输出 | 保留于 3.7；补工具 schema/语义/权限分层，并将 OpenAI 当前 API 的公开 schema 能力限定到该 API。 |
| 截断策略 | 保留于 3.8；记录协议要求的结束边界、证据/角色删减和 labels 变化。 |
| 中文、代码和数学 token 效率 | 保留于 3.9；不使用含义不明的字符/字节分母，不设通用压缩阈值。 |
| placeholder 与视觉特征数量检查 | 保留于 3.10；将通用计数相等替换为模型版本声明的映射符合率；只有特定协议一对一时数量相等才必要。 |
| tokenizer、模板、collator 和 checkpoint 版本 | 保留于 3.11—3.12；继续要求固定原始消息与最小复现，不把权重加载成功等同协议兼容。 |
| 模板、assistant/PAD/EOS 等格式指标 | 保留于 3.13；有效损失分母改按模型 shift 和 mask 后的实际目标集合定义；指标保留 `unknown`/`not_applicable`，不折成总分。 |
| 四类格式事故 | 保留于 3.14；媒体事故改为协议映射不一致，不把特定模型计数规则写成普遍规律。 |
| Python 格式审计程序 | 保留于 3.15 的 toy 审计示例；移除未用导入、PAD/EOS 不同 ID 的硬断言和固定 token/字符门槛；增加 attention mask、共享 ID 下按 ID/按位置屏蔽的对比、UTF-8/码点观察值；明确 toy 媒体映射为一对一假设。代码未运行，状态标为静态审阅。 |
| 事故记录与回归 | 保留于 3.16；继续记录当时实际输入、labels、mask、版本和修复前后输出。 |
| 参考来源和结论边界 | 扩展于 3.17；新增固定版本实现锚点、ACL 2026、OpenAI schema、LLaVA/BLIP-2 与 PyTorch checkpoint 资料，并说明它们各自不支持什么外推。 |

## 例子、公式、代码与同步状态

- **example_status**：3.6 六 token 教学序列包含 BOS、user、问题、assistant 头、答案、EOS。标签为 `[-100,-100,-100,-100,答案 ID,EOS ID]`；模型头内部 shift 时，位置 3 logits 预测答案、位置 4 logits 预测 EOS，故有效目标数为 2。3.15 脚本另构造模板/截断/媒体映射故障和共享 PAD/EOS ID 的 mask 对照；所有输出属于 toy 例，不是目标模型实测。
- **formula_status**：保留 token 长度、token/码点比、loss、assistant coverage、prompt leak、模板匹配、PAD leak、EOS coverage、媒体映射符合率与有效 loss denominator 定义。12 个 `math` 围栏静态成对；核对变量定义、空分母状态、causal shift 和映射口径。GitHub MathJax 页面未实际渲染核验。
- **code_status**：3.15 为标准库 toy Python 示例；本轮修改了代码、未运行。正文标注“静态审阅，未运行”。它只检查序列标签/掩码和声明的一对一媒体规则，不模拟真实模型 logits/loss 或通用协议。
- **link_status**：已核对本章两处本地专题书链接；新增来源均链接论文、RFC 或官方文档。外链未做全量可达性检查。
- **sync_status**：更新 `EXERCISES.md` 第 69 题，改为协议声明的 toy audit、shift 后目标、PAD/EOS 共 ID 的位置掩码和模型专属媒体映射；更新 `INTERVIEW_BANK.md` SFT 数据/故障题及多模态 placeholder、assistant-only loss 题。目录第三章摘要仍符合本章职责；章序与主标题未变，无需修改目录或简介。

## 遗留风险与交接

- 第 3.15 示例代码静态审阅但未执行；目标模型的真实 tokenizer、collator、loss shift 和 generation 停止配置仍需在具体系统中回放验证。
- GitHub MathJax 页面实际渲染和外部链接全量可达性未核验；动态 API 文档需在产品/API 变更时重新核查。
- 下一入口：第四章《训练不稳定排查宝典》。承接本章交出的实际 labels/有效目标位置与数据协议，检查异常 step、坏 batch、梯度和优化器状态；若训练事故章节把原始 `labels != -100` 直接当作统一有效 token 分母，应按模型的 shift/reduction 规则复核。
- 按用户要求不更新 PDF 或 `plan.md`；全套书精修 goal 继续 active。
