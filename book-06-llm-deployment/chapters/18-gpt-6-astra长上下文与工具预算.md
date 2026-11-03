# GPT-6 Astra：长上下文、推理档位与工具预算

> 资料来源：[GPT-6 Astra 模型页](https://developers.openai.com/api/docs/models/gpt-6-astra)、[GPT-6 family 指南](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra)、[Reasoning](https://developers.openai.com/api/docs/guides/reasoning)、[API 定价](https://developers.openai.com/api/docs/pricing)、[Fast mode](https://developers.openai.com/api/docs/guides/fast-mode)、[Ultrafast mode](https://developers.openai.com/api/docs/guides/ultrafast-mode)、[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)、[Conversation state](https://developers.openai.com/api/docs/guides/conversation-state)、[Compaction](https://developers.openai.com/api/docs/guides/compaction)、[Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search)、[Async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling)、[Mid-turn steering](https://developers.openai.com/api/docs/guides/steering)、[Background mode](https://developers.openai.com/api/docs/guides/background)、[Misalignment monitoring](https://developers.openai.com/api/docs/guides/safety-checks/misalignment-monitoring) 和 [Rethinking skills and prompts](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)。GPT-5.3 Codex、GPT-6 Sol、GPT-6 Luna、GPT-6.1 Sol 的对照分别链接在对应小节。核验日期：2026-11-02。价格、可用模型、参数兼容性和安全能力会变化，接入前应重查官方页面。本章只陈述公开的接口和运行时契约，不据此推断模型参数、内部架构或训练过程。

## 1. 先分清总上下文、输入预算与输出预算

GPT-6 Astra 官方模型页明确列出 <code>1,050,000</code> context window 和 <code>128,000</code> max output tokens。当前模型页没有单独列出 maximum input tokens。因此，<code>922,000 = 1,050,000 - 128,000</code> 只是“假设本轮把最大输出额度全部预留”时的算术余量，不是官方另行承诺的 maximum input，也不是每个请求都能使用的固定输入上限。([模型页](https://developers.openai.com/api/docs/models/gpt-6-astra))

可以先用一个简化账本理解单次响应：

~~~math
I + O \leq C,\quad O \leq O_{\max}
~~~

其中 <code>C</code> 是 context window，<code>I</code> 是本次请求输入的 token，<code>O</code> 是本次响应产生的 token，<code>O_max</code> 是 API 的最大输出限制。实际的 <code>I</code> 不只有用户刚输入的文字，还可能包含系统指令、工具定义、历史消息、工具回执和多模态输入；<code>O</code> 则可能包含推理 token、可见文本和结构化输出。上下文里还需容纳运行时状态。多轮 Agent 不能把每一轮的 1.05M 相加，当作一个可无限累积的“任务窗口”。

这一区分用于做容量规划，而不是证明请求会被接受。更实用的做法是为每个请求预留输出和状态空间，再给输入设一个低于理论余量的上限：

~~~text
input allowance
  = context window
  - reserved reasoning and visible output
  - tool / protocol state
  - safety margin
~~~

例如，长代码任务若把 <code>max_output_tokens</code> 设为 128K，就不能再把 <code>922K</code> 当作无条件可用的输入额度；系统指令、工具 schema、工具返回值、推理输出和安全余量仍会占位置。对于尚未量过的任务，先从较低的输入目标开始，记录实际 <code>usage</code>、截断原因和成功率，再逐步调高。

可以把一项长文档审阅任务先拆成几类输入，再看预留是否真实可行。假设系统与开发者指令占 12K token，工具定义占 18K，历史对话占 220K，待读资料占 470K，本轮用户问题占 30K，图像输入经 usage 统计占 64K，则预计输入为 <code>12K+18K+220K+470K+30K+64K=814K</code>。这里的图像 token 数是假设值，不可用文件大小或像素数直接代替；生产记录要以接口返回的 usage 为准。

如果本轮仍预留 128K 的最大输出额度，应用再为协议状态留 16K、为计数误差和任务波动留 80K，那么项目自己的输入目标是 <code>1,050K-128K-16K-80K=826K</code>。814K 输入落在目标内，尚余 12K 的请求级余量。若只把文档又加长 20K，输入就变成 834K，超过项目上限 8K；即便它仍小于简单减法得到的 922K，也应先分批、缩短历史或重做预算。16K 和 80K 是这个系统设计者选的保留量，不是 OpenAI 公布的隐藏协议开销或推荐值。

这个手算的作用是让各项预算能够追责：系统指令由应用版本决定，schema 由工具清单决定，历史由会话重放策略决定，资料由检索器决定，图像等多模态输入由媒体处理路径决定，输出由任务与模型响应共同决定。每项都应有来源和采样窗口。若只看用户文字，实际输入常被 schema、重复历史和工具回执抬高；若将整个窗口都填满输入，又会在模型需要调用工具、生成结构化结果或继续回答时挤掉输出空间。

预算还要区分“单次响应”与“整项工作”。同一任务的第 2 轮会携带新输入以及需要重放的会话状态；上一轮消耗过的 token 不会因为 API 有 <code>previous_response_id</code> 就自动变成新的空窗口，也不能把每轮的 1.05M 相加成任务总容量。应用可选择服务端会话引用或 stateless input-array replay，但两种路径都应保留一份自己的任务账本，记录本轮传入哪些状态、输出还剩多少空间，以及工具回执是否已经被模型消费。

接近预算时，系统应按优先级裁剪或分批，而不是随机删消息。先删除重复的检索片段和已核验的冗余材料，再把长文档改成带来源位置的分块读取；工具结果可保留关键字段、哈希和原始记录入口；当前目标、未完成 job、授权和已发生副作用则写进外部状态库，不应依赖一段可能被裁剪的自然语言摘要。每次裁剪都要在评测中检查漏掉关键证据、重复执行工具和恢复失败的比例。

因此，922K 只能回答“在这个简化容量减法里，最大输出预算全部占满时还剩多少算术空间”，不能回答“我可以稳定发送多少用户资料”。预检器应使用本地保守上限，在线请求后再检查真实 usage、截断/不完整状态和任务完成率；如果模型页增加单独的 maximum input 字段，应按新字段更新 manifest，而不要静默改写历史实验中的输入限制。

预检账本要用线上请求分布校准，而不是把 80K 安全余量当固定答案。假设最近 1,000 个请求中，本地 tokenizer 对实际 usage 的低估量 P95 是 14K，而最大值是 67K；若工作负载的目标是覆盖 P95 级变化，80K 留量在这批样本中包得住，但当输入变化来自新工具 schema 或增加图像帧时，旧分布就不再代表新流量。重新部署前需用新 schema 和媒体路径抽样，估计误差分布并记录请求版本。

如果把同一个 814K 例子中的 470K 资料扩展为 520K，本地估算就增加 50K；此时预留的输入上限 826K 只剩 12K，超过 38K。选择可以有三种：按章节切成多次提问、检索最相关的证据块并保留来源位置、或降低本轮输出 reserve 并用任务评测确认答案仍完整。不可只看输入数字降低就判断质量不变，应同步检查引用覆盖、漏读关键章节和 incomplete response。

容量门槛的监控最好同时保留估计值与真实 usage：按端点、模型版本、提示模板、工具清单和媒体类型分组，报告输入估算误差、输出耗尽率、上下文拒绝/截断、成本和成功率。请求 ID 与预算版本让事故能回溯到当时的定义。这样当官方模型页或 API 行为变化时，历史记录还可说明“旧预算对旧合同通过”，不会被误读成新合同也具有相同可用输入空间。

多模态请求尤其要留意估算单位。文本字节数不等于 token 数，同一张图像可能经过裁切或高分辨率处理变成不同输入负担；工具返回的 OCR/截图也会加入后续请求。账本可把文本、图像、音视频和工具回执分列，分别保存数据来源和接口 usage。若当前 API 不返回足够细的模态分项，就明确记录总输入和未知分摊，不要伪造精确到单个文件的价格。面向容量的余量需覆盖可预见的输入变化，同时为最大输出和已知工具状态保留空间。

输出 reserve 可以按任务变化，而不是所有任务固定用 128K。若一次结构化审阅预计 reasoning 为 20K、可见 JSON 与解释为 12K，应用可以给该类任务设 64K 的 max output reserve；同时保留 16K 协议预算和 80K 安全边际，本地输入目标便为 <code>1,050K-64K-16K-80K=890K</code>。相对于预留 128K 输出时的 826K，多出 64K 输入空间；但若少数请求需要 70K reasoning/output，它们可能触顶。上线前要用真实任务分布确认尾部需求，不能因为平均只用 32K 就把上限降到 32K。

## 2. 推理档位是可测的预算设置，不是质量等级

GPT-6 Astra 模型页列出 <code>reasoning.effort</code> 的 <code>low</code>、<code>medium</code>、<code>high</code>、<code>xhigh</code> 和 <code>max</code>。这些字符串是该模型的 API 选项，不能与其他厂商或其他模型的同名选项作数值等价比较。GPT-6 家族还提供 <code>reasoning.mode</code> 的 <code>standard</code> 与 <code>pro</code>：mode 选择执行模式，effort 控制该模式中的推理投入，两者彼此独立。官方把 <code>pro</code> 描述为适合需要更多模型工作、可容忍更高延迟和 token 用量的任务。([Astra 模型页](https://developers.openai.com/api/docs/models/gpt-6-astra)，[Reasoning 指南](https://developers.openai.com/api/docs/guides/reasoning))

推理 token 不会作为普通文字展示，但它们会占 context、计入输出 token 并产生输出费用。官方说明，复杂度不同的请求可能生成几百到数万 reasoning tokens；若 reasoning 加可见输出触及 <code>max_output_tokens</code> 或窗口边界，响应可能以 <code>incomplete</code> 结束，甚至还没产生可见答案。官方建议初次摸索 GPT-6 类模型时至少给 reasoning 和输出预留 25K tokens，再按本地观测调节；这是起步建议，不是模型的固定需求。([Reasoning 指南](https://developers.openai.com/api/docs/guides/reasoning))

因此，<code>max_output_tokens</code> 不是“答案最多写多少字”，而是 reasoning 与可见输出共同受限的硬上限。若任务要先思考、再调用工具、最后产出 JSON 或代码，输出预算必须覆盖整个响应过程。排查“模型没回答”时，应先检查 <code>status</code>、<code>incomplete_details.reason</code> 和 <code>usage.output_tokens_details.reasoning_tokens</code>，不能只看空的 <code>output_text</code> 就断定模型没有运行。

<code>effort</code> 是否值得升档，应由任务结果决定。至少同时看成功率、reasoning/output tokens、工具次数、TTFT、TPOT、超时率和单位成功成本。若失败是检索漏资料、工具返回错误、权限不够或验证器不准确，单纯调高 effort 可能只增加推理消耗，不会修复失败路径。

下面用一组明确标注为合成数据的 100 个编码任务，说明为什么不能只比较“成功率多了多少”。两组固定同一版模型、输入、工具集合、deadline、验证器和 Standard 费率，只改变 effort。每个任务输入 50K token，低档的输出合计（reasoning 加可见结果）为 10K，高档为 25K；低档成功 72 项，高档成功 86 项。成功定义为补丁通过测试、接口约束与权限检查，不把模型的自我报告当成验收。

在未超过长上下文阈值时，50K 输入按每百万 10 美元计为 <code>50K×$10/M=$0.50</code>。低档的 10K 输出按每百万 50 美元计为 <code>$0.50</code>，单次 token 成本约 <code>$1.00</code>，100 项总计约 <code>$100</code>。高档 25K 输出约 <code>$1.25</code>，单次共 <code>$1.75</code>，100 项共 <code>$175</code>。这里未计工具、重试、人工处理和区域附加费，因此是模型 token 账，不是任务总账。

将总花费除以成功数，低档为 <code>$100/72≈$1.39</code> 每个成功任务，高档为 <code>$175/86≈$2.03</code>。高档多成功 14 项，成功率上升 14 个百分点，但本例单位成功成本约高 46%。若业务门槛是成功率至少 85%，低档不合格，高档可能值得；若门槛是至少 70%，低档已合格，继续升档须证明质量改善足以覆盖约 46% 的成功成本增加。若样本量、难度或验证器不同，这个对照就不能用于因果判断。

实际实验还应按任务难度分层。把短问答、长代码修改和外部工具任务混成一个平均数，可能出现低难度任务占多数、掩盖高风险切片失败的情况。除总体成功率，还要报告各切片分母、失败类型、响应 incomplete 比例、推理 token 分布、工具调用数及 p50/p95 延迟。尤其当模型在高档多花 token，却仍因检索缺证或权限拒绝失败时，应修数据和系统边界，不能把更多 reasoning token 当成补救。

<code>standard</code>/<code>pro</code> 与 <code>low</code>/<code>medium</code>/<code>high</code>/<code>xhigh</code>/<code>max</code> 是两个实验维度。先固定 effort 比较 mode，再固定 mode 比较 effort，或使用有足够样本的二维矩阵；不要在同一对照里同时改模型、mode、effort、工具版本和输出上限。OpenAI 对 GPT-6 公布了相应接口能力，但不同 model ID 的支持项并不完全相同，部署 manifest 应保留精确模型、端点、参数和访问日期。mode 名称不提供跨模型的数值刻度，也不保证固定成本倍率。

Reasoning 指南给出的“初次实验至少为 reasoning 与输出预留 25K token”是探索阶段的预算起点，不是每道题都会消耗 25K 的预测值。工程团队应从目标工作负载中采样，在响应 <code>status</code> 与 <code>incomplete_details</code> 中区分输出耗尽、上下文边界和其他不完整原因，再调整 reserve。若预留过小，评测会把预算截断误判为能力失败；若固定预留过大，又可能把可用于输入证据的容量挤掉。

最后把 mode 和 effort 的作用与底层系统故障分开诊断：文档缺失时应修检索，工具 schema 错误时应修接口，沙箱拒绝时应检查权限，验证器不完整时应重写验收条件。只有当任务确实需要更深的多步推理、且同条件实验显示成功收益足以覆盖延迟和成本时，升档才是合理选择。

100 项的比例本身也有采样误差。把两组先当作相互独立的二项样本，低档 72/100 的比例标准误约为 <code>sqrt(0.72×0.28/100)=0.045</code>，高档 86/100 约为 <code>sqrt(0.86×0.14/100)=0.035</code>；两者差值 0.14 的粗略标准误约 <code>sqrt(0.045²+0.035²)=0.057</code>。这只是合成例子的近似，不是显著性检验结论；真实实验若对同一任务逐项配对，应分析成对结果，并用足够样本估计区间，避免被一次小样本波动误导。

更稳的设计会在请求提交前固定 task ID、输入快照、工具版本、mode/effort、输出上限和 verifier 版本，低/高设置的运行次序随机化以降低时间漂移。对每项任务记录两种设置是否通过，做配对差值和失败样本审查；若任务有随机性则安排重复运行，并标明重复实验是否算作新的独立样本。报告总 token 而非只报告 reasoning token，因为可见输出变化、缓存读写和工具回执都可能影响总账。

选择方式还取决于损失函数。若高风险任务失败成本很高，系统可以只把一部分疑难样本送入较高 effort，并让验证器不通过时再升级；若大部分请求都低风险，则默认低档、只对复杂度触发条件升级可能更省。两种路由策略都要在独立留出任务上检查误分类后果，并把筛选器本身的成本、调用次数及失败留在单位成功成本分母中。

升级规则要预先定义停止条件。例如 low 首轮在 JSON schema 校验失败时，可先把结构错误回传并要求一次低成本修正；如果问题是证据矛盾，再尝试 medium/high；若工具权限被拒或远端数据缺失，则不要升 effort，而应停下等待授权或补资料。每次升级都占用全局预算，且可能生成新 response，因此规则需同时限制升级次数和剩余 deadline。否则“失败就升到 max”的隐式策略会让少数坏请求耗尽整批预算。

人工判断仍需分开回答“结果更好了吗”和“多花的钱值得吗”。质量门槛可能来自准确率、测试通过、答案证据或人审评级，不应只用成本最优覆盖硬性安全要求。对通过门槛的多种配置，可比较单位成功成本、p95 任务时间和人工返工率；对未通过门槛的配置，即便平均成本最低也不能宣布上线。已上线后还要监控任务分布漂移，必要时重做 effort 分层评测。

## 3. 模态与工具支持不等于执行权限

GPT-6 Astra 官方页列出文本输入/输出、图像输入，以及 web search、file search、image generation、code interpreter、hosted shell、apply patch、skills、computer use、MCP 和 tool search 等工具。它把图像列为输入模态、文本列为输出模态；这不等于模型原生输出图像 token。模型页也说明，工具类模型可能按调用另行计费。([模型页](https://developers.openai.com/api/docs/models/gpt-6-astra))

一次工具调用至少要分清四个事件：

~~~text
模型提出调用
  -> 应用或平台校验 schema 与参数
  -> 策略层授权
  -> executor 执行并回传结果
  -> 模型消费结果，或 verifier 检查产物
~~~

模型页上出现 <code>hosted_shell</code> 或 MCP，不能据此推导它能访问任意文件、网络或终端。对应用自定义工具，执行器负责身份、路径、网络、超时、幂等和审计；对托管工具，也要根据平台公开的工具权限与计费条款单独验收。一次“模型返回了 tool call”既不代表宿主已授权，也不代表外部副作用已成功提交。

用一次内部知识库查询与生产发布提案，可以把这些责任拆开。假设用户询问“本季度的退款政策是什么”，Astra 生成名为 <code>search_policy</code> 的结构化调用，参数指定租户 <code>tenant-41</code> 和最多 20 个结果。第一步，API 根据请求中的 tools schema 校验字段形状；应用再检查当前用户是否属于该租户，以及是否具有只读查询范围。schema 合法只说明参数可解析，不证明用户有权访问这个租户。

策略层通过后，执行器才在固定的索引和只读凭据下执行查询，并设置 2 秒 timeout、结果上限和审计 ID。它返回 3 条资料及其来源位置，应用保存原始响应哈希，再把结果附在对应工具调用之后交还模型。模型可以据此组织答复；独立 verifier 则检查引用是否来自这 3 条记录、文档版本是否仍有效。若索引超时，执行器返回明确的超时结果而不伪装为空列表；若用户无权访问，策略层拒绝且不启动查询。

接着模型提出 <code>publish_policy</code>，要把一段新文字覆盖到生产站点。即使参数符合 schema，即使模型在前一轮已经成功读取文档，这一写操作仍需单独检查身份、目标环境、审批人和内容版本。策略层发现没有人工批准就拒绝，执行器不得发出 HTTP 写请求；应用记录“提议已拒绝”而不是“发布失败后已回滚”。若用户随后批准，也要重新检查待发布内容的哈希、审批有效期和目标版本，避免批准的是另一份内容。

完整轨迹可按状态区分为 proposed、validated、authorized/denied、dispatched、completed/failed、verified。每次状态变化关联 API response ID、tool <code>call_id</code>、执行操作 ID、调用者身份、输入哈希、权限策略版本、幂等键、开始/结束时间、结果哈希和 verifier 结论。重复收到同一结果时，job registry 根据幂等键去重；同一个工具名的两个不同调用则保留为不同操作。这样才能判断失败发生在模型提出、参数校验、授权、执行还是结果验证阶段。

托管工具和应用自定义工具的责任边界也不同。模型页列出的 web search、code interpreter、hosted shell 或 computer use 表示文档公开了对应 API 能力；实际可用性、工具作用域、区域限制、调用计费和数据处理仍按当前平台合同核验。自定义 function/custom tool 的执行代码由应用拥有，API 收到 tool call 后不会替应用部署沙箱或替它保证业务权限。GPT-6 的工具调用应按当前 family guide 使用 Responses API；模型页同时列有 Chat Completions endpoint，不等于工具调用也可在该 endpoint 上使用。

如果工具返回一段看似可信的文本，模型也可能把它误当作已验证事实。读取型结果应保留来源和时间，写操作应查询目标系统确认实际状态，支付/删除/部署等高影响动作还应以外部事务回执或人工审批为准。模型后续说“成功”只能作为输出，不能代替 executor receipt；工具返回“请求已接受”也不等于最终状态已经提交。

因此，设计工具时至少列出可调用角色、资源范围、读写类别、参数约束、超时、并发、重试/幂等策略、审批点、回执格式与核验来源。拒绝路径要返回可操作的错误类型，使模型能提出替代方案；但不要向它暴露能绕过策略的秘密细节。将能力、授权和执行分层后，工具菜单才不会被误读成一张无限权限清单。

把两个失败轨迹分开记会更容易定位。若参数里的 tenant ID 不在当前用户许可集合内，策略应在启动 executor 前拒绝，预期状态是 denied，副作用数为 0；若策略通过但查询超时，executor 状态为 timed_out，应用仍需确认底层查询是否在远端结束，不能把超时记为“没有读取”。对于发布写操作，如果审批有效但目标站点版本已变化，应在写入前用版本条件拒绝，返回 stale_version，而不是把旧内容覆盖到新状态上。

每一种工具回执还应有可验证的含义：accepted 可能只表示排队，completed 表示执行器自称结束，verified 才表示外部状态与预期一致。以支付工具为例，模型收到“已受理”不能回答“款项已到账”；应用需向支付系统查询交易 ID，再按业务规则检查最终状态。错误类型应区分权限拒绝、schema 错误、过载/timeout、远端失败、重复请求和 verifier 不通过，模型才能在无副作用的条件下修正请求。

一次工具动作的风险由授权和数据流共同决定。只读搜索也可能泄露跨租户信息；生成补丁可能改动正确文件之外的配置；computer use 可能点到错误账户或按钮。策略层应对输入资源、可访问目标和输出位置一起做边界检查，执行环境使用最小凭据和最小目录。需要确认外部变化时，以系统自己的版本号、事务 ID 或内容 hash 作为事实来源，避免把自然语言确认当作操作收据。

读取型工具还会把外部内容带回模型上下文，因此要把“内容可信度”与“工具来源可信”分开。查询结果可以来自受信任的数据库，但其中用户上传的文字仍可能包含恶意指令；模型应把它当作待分析资料，不能允许其覆盖系统授权规则。应用可在结果中保留来源、检索时间、权限范围与内容哈希，必要时过滤敏感字段，并让 verifier 检查答复引用是否来自授权记录。

schema 也应限定的不只是数据类型，还包括资源边界和语义约束。例如把任意路径字符串改成受限 workspace-relative path，把条数限制为 20，把金额写成明确币种的整数最小单位，把 URL 域名限制到允许列表。即使 JSON schema 合法，业务层仍要检查范围和对象归属；“通过 schema”只进入下一门，不自动进入授权门。拒绝理由可使用稳定错误码和可补救提示，避免模型不断换参数试探未知权限。

## 4. 长上下文费用要按整次请求和缓存类别拆开

官方定价页按每百万 token 报价，并区分输入、缓存读取、缓存写入时长、输出和处理档位。对 Astra，总输入超过 272K token 后，整次请求的 input/cache 费率按 2 倍、output 按 1.5 倍计费；不是只对超过 272K 的那部分加价。缓存写入分 30 分钟与 12 小时两档。下表是 2026-11-02 核对的美元价格，不包括符合条件的区域处理/FedRAMP 附加费、工具逐次费用和账户合同折扣。([API 定价页](https://developers.openai.com/api/docs/pricing)，[模型页](https://developers.openai.com/api/docs/models/gpt-6-astra))

| 处理档位 | 输入范围 | 未缓存输入 | 缓存读取 | 缓存写入 30 分钟 | 缓存写入 12 小时 | 输出 |
|---|---:|---:|---:|---:|---:|---:|
| Standard | ≤272K | $10 | $1 | $12.50 | $25 | $50 |
| Standard | >272K | $20 | $2 | $25 | $50 | $75 |
| Batch / Flex | ≤272K | $5 | $0.50 | $6.25 | $12.50 | $25 |
| Batch / Flex | >272K | $10 | $1 | $12.50 | $25 | $37.50 |
| Fast | ≤272K | $20 | $2 | $25 | $50 | $100 |
| Fast | >272K | $40 | $4 | $50 | $100 | $150 |
| Ultrafast | ≤272K | $60 | $6 | $75 | $150 | $300 |
| Ultrafast | >272K | $120 | $12 | $150 | $300 | $450 |

以上各列单位都是美元/百万 token。Batch/Flex 是 Standard 的 50%；Fast 是适用费率的 2 倍；Ultrafast 是 6 倍。下面用两个相邻输入量检验阈值：Standard、全为未缓存输入、输出固定 10K 时，272,000 输入尚未超过门槛，账单为 <code>272K×$10/M+10K×$50/M=$2.72+$0.50=$3.22</code>。输入只增加 1 token 到 272,001，整次请求切换到长上下文费率，账单为 <code>272,001×$20/M+10K×$75/M=$5.44002+$0.75=$6.19002</code>。这是一 token 变化造成约 $2.97 的差异，说明应用不能只按超出部分计价。这个算例固定了缓存类别、档位、输出量，也没有计工具费。

实际判断阈值时要把当前请求中所有输入 token 类别合并：未缓存输入、缓存读取和缓存写入都属于 prompt input；推理 token 已计入 output usage，不要再作为一份输入重复加总。缓存写入费用又取决于 TTL：30 分钟为未缓存输入价的 1.25 倍，12 小时为 2.5 倍；开启缓存不代表该轮必定命中，需同时查看读取量、写入量和实际请求前缀。

缓存 TTL 是否划算可用复用次数手算。假设一个满足当前缓存规则、会被完整复用的 60K 稳定前缀，在每个请求均处于 Standard 短输入费率时不缓存，N 次费用为 <code>N×60K×$10/M=$0.60N</code>。若先写一次 30 分钟缓存、之后 N−1 次均命中，费用约为 <code>$0.75+$0.06(N−1)</code>；两次请求合计 $0.81，低于不缓存的 $1.20。12 小时写入第一次需 $1.50，后续每次读取仍按 $1/M；两次合计 $1.56，高于不缓存，但三次合计 $1.62，低于不缓存的 $1.80。这个推演忽略缓存过期、断点位置、最小可缓存长度和前缀变化，实际是否命中仍应从 usage 验证。

地区与处理档位会继续改变账单。定价页当前说明，2026-03-05 及之后发布且具备数据驻留资格的模型，使用区域处理 endpoint 会有 10% uplift；从 2026-11-01 起，FedRAMP endpoint 也列有 10% uplift。具体模型、端点、区域与合同资格要在上线前核对，不应把这两项机械叠乘；Astra Ultrafast 当前列出的 data residency 范围是 US，另有 global processing，没有 EU residency。Astra Fast 没有 latency SLA；Ultrafast 也有单独速率限制。档位费率不能推出某次请求必然更快，工具等待、排队和客户端网络都可能主导端到端延迟。([Fast mode](https://developers.openai.com/api/docs/guides/fast-mode)，[Ultrafast mode](https://developers.openai.com/api/docs/guides/ultrafast-mode))

服务档位应按工作负载和失败成本选择。Batch/Flex 的低价可能适合对完成时间不敏感的批量任务；Fast/Ultrafast 的费率更高，应先确认资格、地域和速率上限，再在相同输入、缓存状态、工具版本和输出约束下测端到端 p50/p95。若某档只降低模型生成时间，却没有缩短工具等待或审批排队，它对整项任务的延迟改善可能很小。成本评价最后仍要除以通过同一 verifier 的成功任务数。

还要把页面上的 token rate 与任务级实际成本区分开。假设上一节的 300K 示例本轮模型 token 费为 $6，工作流另使用 3 次按调用计价的托管搜索工具，每次费用假设为 $0.02，则模型和工具合计为 <code>$6+3×$0.02=$6.06</code>；$0.02 是演算输入，不是当前官方 tool price。若应用有 1 秒 executor 工作和一次人工审批，应继续加上自己的资源计量，而不是把 $6.06 叫作 agent task 总价。

假设 100 个这样的任务只 80 个通过同一验收，平均模型和工具账单为 $6.06，则已知部分总额为 $606，单位成功成本为 <code>$606/80=$7.575</code>。如果把失败任务也当成功分母，结果变成 $6.06，会少报 $1.515/成功任务，即约 20% 的差异。更完整的账单还要计失败重试、应用 worker、verifier、人工接管与第三方网络费用，最好把每项成本归因到任务与响应 ID。

上线成本监控应使用定价快照和 usage 字段重算。价格表变化时生成新的 rate version，旧请求仍按其发生时间回放；不把新费率追溯应用到旧账，也不把缓存 token 从输入合计中重复扣除。阈值附近的请求应抽样核对 provider usage 和账单明细，确认当前端点实际采用哪类 input、写入 TTL 与服务档位。若无法在 request trace 中解释差异，应先标明待核验，而不是通过调整估计公式让账单“对上”。

长上下文还会让费用门槛与请求拆分策略发生耦合。一个 280K 请求切成两个各 140K 请求后，理论上可能避开整次请求的长输入倍率，但会新增 response 固定开销、重复 system/tool 输入、交接状态和可能的二次总结；若分块之间需要合并，合并响应又产生输入与输出费用。正确比较应对两条完整 workflow 计算所有请求、缓存类别、工具调用和成功分母，不能只比较首轮 prompt 的单价。

服务档位选择也不等于努力档位选择。Fast/Ultrafast 提高适用费率以改变处理配置；pro/effort 则属于推理配置面。要定位一次账单上升，应记录实际 service_tier、请求 effort、mode、总输入类别、输出 usage 与任务路由决策。只增加 Fast 可能加价但没有减少工具延迟；只调 effort 也可能改善结果却增加输出费用。把二者拆开实验才能说明是哪项设置带来了质量或时延变化。

为避免把输入类别混算，教学估算把每轮输入拆成三段：普通未缓存输入 <code>U</code>、缓存读取 <code>H</code> 和新写入缓存 <code>W</code>；三者互不重叠，<code>U+H+W</code> 才是本轮输入。推理 token 已在输出用量内，不要再重复记一次。模型 token 费用可写成：

~~~math
C_{\mathrm{tokens}} = U r_{\mathrm{input}} + H r_{\mathrm{cached}} + W r_{\mathrm{write}} + O r_{\mathrm{output}}
~~~

完整 Agent 任务成本还要把多轮响应、托管工具逐次费用、自建 executor 基础设施、验证器、失败重试和人工接管纳入。并行只能降低可并行工作部分的墙钟时间，不会自动消除工具调用费或模型 token 消耗。

下面的零依赖示例演示 Astra 的阈值倍率、服务档位和缓存读写口径。代码只使用 Python 标准库；将代码块保存为 <code>estimate_astra_token_cost.py</code>，执行 <code>python3 estimate_astra_token_cost.py</code>，会打印一个美元金额。它只估算模型 token 费用，不含工具调用、区域附加费和账户合同；生产账单应以 API 返回的 usage 与当前价格页对账。函数把 cached/read/write token 作为互斥输入类别，按三者总和判断是否超过 272K，并验证非负整数 token、有效 tier 和写入 TTL。布尔值不会被当作整数接受。

**代码状态：静态审阅，未运行（2026-11-02）。**

~~~python
TIERS = {
    "standard": 1.0,
    "batch": 0.5,
    "flex": 0.5,
    "fast": 2.0,
    "ultrafast": 6.0,
}
WRITE_TTL_MULTIPLIER = {"30m": 1.25, "12h": 2.5}


def estimate_astra_token_cost(
    uncached_input_tokens,
    cached_input_tokens,
    cache_write_tokens,
    output_tokens,
    *,
    write_ttl="30m",
    tier="standard",
):
    counts = (
        uncached_input_tokens,
        cached_input_tokens,
        cache_write_tokens,
        output_tokens,
    )
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("token counts must be non-negative integers")
    if write_ttl not in WRITE_TTL_MULTIPLIER:
        raise ValueError("write_ttl must be '30m' or '12h'")
    if tier not in TIERS:
        raise ValueError(f"unknown tier: {tier}")

    total_input = (
        uncached_input_tokens + cached_input_tokens + cache_write_tokens
    )
    is_long = total_input > 272_000
    input_rate = 10.0 * TIERS[tier] * (2.0 if is_long else 1.0)
    cached_rate = input_rate * 0.1
    write_rate = input_rate * WRITE_TTL_MULTIPLIER[write_ttl]
    output_rate = 50.0 * TIERS[tier] * (1.5 if is_long else 1.0)

    return (
        uncached_input_tokens * input_rate
        + cached_input_tokens * cached_rate
        + cache_write_tokens * write_rate
        + output_tokens * output_rate
    ) / 1_000_000


print(round(
    estimate_astra_token_cost(
        140_000, 100_000, 60_000, 20_000,
        write_ttl="30m",
        tier="standard",
    ),
    4,
))
~~~

~~~text
6.0
~~~

上面打印值是手算预期输出（未执行）。本例输入共 300K，落入 long-context 费率：140K 普通输入按 <code>$20/M</code> 得 $2.80，100K 缓存读取按 <code>$2/M</code> 得 $0.20，60K 30 分钟写入按 <code>$25/M</code> 得 $1.50，20K 输出按 <code>$75/M</code> 得 $1.50，合计 <code>$2.80+$0.20+$1.50+$1.50=$6.00</code>。若把写入误算成普通输入，会少算 $0.90；若把 reasoning token 加到输出 token 后又单独计费，会重复计数。真实工具费用和 executor 资源账本要在 token 估算之外追加。

手算边界还可用同一函数输入比较：总输入正好 272,000 时使用短输入费率，272,001 时使用长上下文倍率；负 token、浮点 token、未知 tier 和未知 TTL 应得到输入校验错误，而不是静默转成 0。即使这些边界在本地函数里通过静态阅读，也不证明真实 API 的发票完全采用同一实现细节；函数是按公开费率做的教学估算，应定期与官方页面和返回 usage 对账。

## 5. “工具预算”要覆盖完整任务，不只是一轮 Responses

一次 API 请求的 <code>max_output_tokens</code> 和平台工具限制，只约束当前响应。长任务可能包含多轮 <code>responses.create</code>、异步工具、自动 steering continuation、重试、compaction 与独立 verifier；若只限制单次调用，每次都在上限以内，整项任务仍可能持续消耗资源。因此，应用应维护一个跨响应的 job budget，并明确哪些上限由 API 提供、哪些必须由自己的 runtime 强制执行。

| 预算项 | 需要记录的限制 | 典型执行方 |
|---|---|---|
| 每响应上下文 | 输入 token 目标、输出预留、context margin | 请求构造器与 API |
| 每响应生成 | <code>max_output_tokens</code>、effort、mode | 请求构造器与 API |
| 整项任务模型费 | 所有 response 的输入/输出/缓存费用总额 | job coordinator |
| 工具资源 | 每类工具调用数、并发数、单次 timeout、外部 API 费用 | tool executor |
| 重试与恢复 | 重试上限、允许重试的错误、幂等键、恢复截止时间 | job coordinator 与状态库 |
| 验证 | 测试/查询/独立评审器调用次数和资源 | verifier |
| 总时限与副作用 | deadline、人工审批点、可补偿操作和不可重复操作 | policy / approval 层 |

一个跨轮账本可以表示为：

~~~math
C_{\mathrm{job}} = \sum_i C_{\mathrm{model},i} + \sum_j C_{\mathrm{hosted\ tool},j}
        + C_{\mathrm{app\ executor}} + C_{\mathrm{verification}} + C_{\mathrm{retry}} + C_{\mathrm{human\ review}}
~~~

这也是为什么应报告单位成功任务成本，而不只报一次响应价格。失败重试、验证器和人工接管可能提高单项成功成本；若统计分母只包含已调用的请求而不含失败任务，成本对比会偏低。评测须先定义“成功”：例如产物通过测试、业务约束和权限审查，而不是模型说“已完成”。

下面是一份纯教学限制示例：最多 8 次工具调用、最多 2 个并发只读调用、单项任务 10 分钟、幂等只读失败最多重试一次、外部写操作需审批、模型与托管工具总额封顶 <code>$3</code>、最后最多执行 2 组独立验证。这些数字不是 OpenAI API 限额或推荐值；示例要说明的是预算应有所有者和拒绝路径。async 可让两个独立只读调用重叠等待，但并发器仍要数调用数、占用时长和费用。若用户中途 steer 触发第二个响应，第二个响应的 token/tool 限制独立生效，job coordinator 仍要把它计入同一个 8-call、10 分钟和 <code>$3</code> 总预算。

长上下文代码 Agent 还需要一个外部权威状态记录：model ID/snapshot、mode/effort、输入与输出用量、cache read/write TTL、每个工具的 <code>call_id</code> 和 task handle、schema 版本、权限决定、执行状态、重试次数、compaction/steering lineage、最终 artifact hash、验证结果与总账单。没有这条记录，就很难判断超支来自推理、重复工具调用、过量验证还是失败恢复。

用一项合成的代码审阅任务把工作级预算落到账本。项目负责人设总上限 $3.00，job coordinator 预留模型 responses $1.40、托管工具调用 $0.35、应用 executor $0.30、最多两轮 verifier $0.40、有限重试 $0.20、人工接管准备 $0.15，合计 <code>$1.40+$0.35+$0.30+$0.40+$0.20+$0.15=$2.80</code>，另有 $0.20 不分配的余量。以上均是示例预算，不是 OpenAI 的价格承诺或推荐额度。

执行约束是最多 8 次工具调用，至多 2 个并发只读任务，每个工具最多等待 60 秒，整个 job 的 deadline 为 10 分钟；写入或部署必须停在审批门前。job coordinator 在每次启动 response 或工具前检查累计花费与剩余分类预算，异步结果回来后再入账。若预计下一项操作会使任一分类超过保留额，或总账超过 $2.80，自动流程停止并把状态写为 <code>budget_exhausted</code>；不能因为总上限尚有 $0.20 就偷偷挪用人工接管额度。若用户批准扩额，应新建带审批人的预算版本并保留之前的封顶值。

重试政策也要按操作性质分支。只读查询超时后，先查询 job registry 判断原任务是否仍在运行；若确认已失败，且 deadline 与重试额度尚足，才允许一次重试。对于创建发布单、发送付款或删除资源这类副作用，即使 HTTP 连接超时，也可能已经在远端提交；应用应先用幂等键或外部状态查询确认结果，无法确认时就停止自动重放并升级人工处理。否则“超时再试一次”会把不确定状态变成重复写入。

即使所有响应都在 <code>max_output_tokens</code> 内，每个 response 的限制也只覆盖单轮。假设一次任务先有规划响应、工具回执后的续答、steering 后的 continuation、失败恢复响应和 verifier 复核，总费用应逐次记在同一 job 上。每个响应自己的 token/tool cap 是请求局部保护；job 上限由应用的协调器执行，不能假定 API 会把 $3 总账自动传递给下一轮。

最后用成功任务作为成本分母。若同一批 20 个 job 共消耗 $40，其中只有 16 个通过同一 verifier，成本是 <code>$40/16=$2.50</code> 每个成功任务；除以 20 个发起任务所得的 $2.00 会把失败成本藏掉。报告还应同时列出未成功的 4 个任务、失败原因、人工接管和外部副作用，便于判断预算应投向更好的检索、工具可靠性还是模型推理。

并发限制会改变耗时，却不自动改变调用次数和账单。假设两项互不依赖的只读查询各需 60 秒，串行执行墙钟时间约 120 秒；允许两个并发 worker 后，理想情况下总等待约 60 秒，但它们仍占用 120 worker-seconds，并产生相同的两次远端调用费。若第三步必须等两个回执回来才能开始，则整条链路为查询阶段加后续模型响应和验证阶段，不能把单次工具时长当整项任务 P95。

执行器应把并发限额、每工具 timeout、队列等待和任务 deadline 分开。上例 60 秒 timeout 若从任务排队时就计时，排队 15 秒后只剩 45 秒执行；若设计为 executor 从实际启动时计时，则 job coordinator 仍要设置更长的全局 deadline。记录 start_requested、worker_started、remote_completed、result_returned 等时间点，可以分辨是内部队列、外部服务、网络回传还是模型等待造成超时。

预算拒绝也要有确定的动作：如果任务已经达到 8-call 上限，模型再次提议只读工具，不能仅因请求看似无害就绕过限制；协调器应先返回调用额度已耗尽的结构化状态，再由模型决定基于现有材料结束或请求用户授权扩额。若 timeout 未确认外部 job 状态，则状态标为 unknown/pending；不能把它改成 failed 后再次启动相同副作用。所有限制值和更改者进入版本化 manifest，以便复盘某次请求为什么被拒。

预算实施可采用“预留—执行—结算”循环。请求前按估计 token 与费率预留模型额度，启动工具前锁定调用计数和工具费用上限；执行返回后用真实 usage/receipt 结算，多退少补到 job balance。若一个工具价格未知，就用保守上界或暂停而不是记零。并发任务需原子地检查同一总余额，否则两个 worker 同时看到剩余 $0.30、各自启动 $0.25 操作，会共同超支。

每个 job 还应定义剩余预算表：响应生成可用额度、只读工具额度、写操作审批额度、verifier 额度、重试额度和人工接管保留值。将金额与任务状态一起保存在可恢复存储中，worker 重启后先恢复已经 reserve 的额度和已提交 receipt，再接收新调用。取消 job 时要处理仍运行的工具并核实取消结果；把前端标为 cancelled 并不一定能停止远端操作。

例如，job 为一个 response 预留 $0.70，真实账单到达后发现用掉 $0.62，协调器只有在拿到最终 usage 后才释放未用的 $0.08；如果 response 还在流式输出或存在继续生成，就不能提前释放。并行的两个搜索工具各 reserve $0.12，返回后分别用 receipt 结算。若第一个工具超时但远端仍 pending，对应 $0.12 仍应保持冻结，不可拿同一余额启动另一项可能重复的写操作。预留机制使并发决策使用一致的剩余预算，而不是各 worker 独立猜账。

## 6. 状态回放、动态 effort 与工具搜索

长任务的上下文不只是可读的聊天文本。Responses item 还可能包含 function call / function result、加密的 reasoning item、assistant <code>phase</code>、tool-search 加载结果以及 compaction item。无状态 replay 必须依照 API 协议保留后续生成需要的 output items；只把最终可见文本拼回去，可能丢失工具链和继续任务所需的状态。<code>previous_response_id</code> 可以引用服务端响应链，但引用状态不等于历史输入免费，也不免除应用维护业务任务状态。([Conversation state](https://developers.openai.com/api/docs/guides/conversation-state)，[Compaction](https://developers.openai.com/api/docs/guides/compaction))

GPT-6 支持在对话中追加 <code>configuration_update</code> 改变后续响应的 reasoning effort。要尽量保留缓存前缀，官方建议保持顶层 <code>reasoning.effort</code> 原值，再把新的 effort 作为追加 item；若直接改顶层参数，隐藏系统指令可能变化并打断前缀复用。这个 item 是会话输入协议，不是改权重或切换 checkpoint。官方将配置更新限定为标准模式的单 Agent 会话；它不能与自动 compaction/truncation 或 standalone <code>/responses/compact</code> 混用。应把每次更新及其生效顺序写进 replay trace，而不是只记录最终的 effort 值。([Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)，[GPT-6 family guide](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra))

工具目录本身也会占上下文。工具很少且常用时，把 schema 一开始全部提供，少一次发现步骤；目录大而每个任务只用少数工具时，可 defer 部分 function schema，通过 tool search 按需加载。Hosted tool search 由平台从本次声明的 deferred tools 中找候选；client-executed tool search 则由应用决定如何检索。加载后工具会追加到上下文，较容易保留已有前缀，但新增的 discovery step 仍可能带来延迟和漏选风险。namespace 的描述会用于发现，已 defer function 的名称和描述仍可能对模型可见；搜索到工具不是执行授权，schema、来源、版本和权限仍要检查。([Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search)，[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching))

因此，动态 effort、tool search 和 compaction 是三个不同的控制点：effort 改变后续推理预算，tool search 改变可用工具 schema 的加载集合，compaction 改变继续任务的上下文表示。日志中要分别记录，否则看见缓存命中率或任务成功率变化时，无法判断是哪个控制点产生影响。

可以用一个无状态回放样例看出“只保存最终文本”丢失了什么。首轮响应的 output 依次含有一个 reasoning 状态 item、调用 ID 为 <code>call_17</code> 的只读函数调用、同 ID 的工具结果，以及标记为最终回答阶段的 assistant message。若回放时只保存最终回答，下一轮就没有可关联的函数调用与结果，也无法分辨模型是否已经消费工具回执；统计成功率时便可能把“模型写过一句答案”误当成“基于工具结果完成”。stateless input-array 路径应按文档保留完整 output items；若走服务端 response 链，则使用 <code>previous_response_id</code> 继续，并仍在应用侧保存业务 job 状态。

缓存和工具集合是状态的一部分。假设一个工作台声明 64 个函数 schema，每个平均 350 token，启动请求就会带入约 <code>64×350=22,400</code> token。若只需先给模型一个 600-token 的 namespace 描述，再延迟加载本轮需要的 3 个 schema，加载后约为 <code>600+3×350=1,650</code> token，示例节省 <code>22,400-1,650=20,750</code> token。这个数只基于假设的 schema 长度，不包含 search call、结果文本、系统指令或缓存命中差异；还应把发现延迟、错误 schema、漏选和后续补找的成本加入评测。

deferred tool search 是否更好，要按工具目录与任务分布判断。若每项任务都调用同一组少量工具，eager schema 省掉发现步骤，可能更简单；若目录大而每项任务只用少数工具，defer 可以减少初始输入。namespace 名称和描述要说明可发现的工作范围，schema 版本、可信来源、租户隔离和移除时间要能追踪。模型搜索到某个工具后，宿主仍须重新检查当前授权；工具加载结果不能作为权限变更的唯一记录。

动态 effort 也需要在回放中记录成事件序列。例：请求从 <code>reasoning.effort=low</code> 开始，读完材料发现存在互相矛盾的日期，再附加一次 <code>configuration_update</code> 提升后续响应 effort；之后该设置保持有效，直到下一条更新覆盖。若应用同时重写顶层 effort，可能破坏稳定 prompt prefix；官方兼容说明把该 item 限定为 standard 单 Agent 请求，且不能和自动 compaction/truncation 或 standalone compact 混用。设计实验时应按指南记录更新位置和生效顺序，不能只记录最后一个 effort 值。

三种机制的状态影响不同：reasoning update 不会把模型权重换成另一个 checkpoint；tool search 不会替工具执行；compaction 不保证向应用输出一份可读的业务摘要。比较它们时固定任务和提示，再分别改变一个因素，报告首轮输入 token、搜索步骤、缓存 read/write、后续响应费用、恢复成功率和任务成功率。若把三者同时启用，效果无法归因，也无法解释某次恢复为什么丢失权限或结果。

即便静态回放能复现同一事件顺序，也要保留服务器端事实的边界：模型输出的 reasoning 状态可能是加密且不可读的，tool search 返回结果和缓存是否命中也受当时工具清单、prefix 与服务端状态影响。保存 trace 是为了复核接口路径和业务决定，不是把它当作公开模型思维过程，也不能用一次 replay 证明未来 API 版本保持完全相同的语义。

一个小型配置轨迹可以把更新规则写清楚。response A 以 request-level <code>reasoning.effort=low</code> 开始；模型先归纳四段互相一致的资料，再因发现两个发布日期冲突而追加一次 update 把后续 effort 提到 high；response B 继续处理冲突，验证完毕后，另一条 update 才能把后续普通格式化工作调回 low。应用需要保留 A 的 output item 顺序、两次 update 的位置和各响应的实际 usage，才能解释 B 为什么使用 high。

如果客户端在第二次请求时直接把顶层 reasoning effort 改成 high，就不是同一种输入路径：稳定前缀可能随参数变化，cache comparison 也被混入配置变更。要评测 update 的实际收益，应把“固定顶层参数+追加 update”与“每轮修改 request-level 参数”作为不同条件，记录缓存 read/write 与任务成功；文档关于保持 cache 的说明是接口设计信息，不能保证每次都发生 cache hit。

tool schema 变更也应写进状态回放。比如第一轮 defer 加载 <code>search_invoice</code> v3，第二轮管理员撤销工具权限并发布 v4，第三轮继续旧 response 时不能盲目照搬旧 schema 或把旧授权视为仍有效。宿主应在每次执行前重新核验权限版本，必要时拒绝并把变更原因作为工具结果回传；同时记录搜索来源、schema hash、租户和回滚时间，避免缓存前缀稳定被误读为权限可以永久沿用。

验收记录的最小字段可包括：model ID/snapshot、请求顶层参数、input item 序列号与类型、configuration update 顺序、tool namespace/schema hash、deferred search 的命中结果、reasoning/output/cache usage、compaction item 位置、每个 response ID 与 tool call ID、权限版本和外部副作用。日志不应尝试还原不可读的 reasoning 文字；应以协议项、引用 ID 和外部证据建立可复查链路。

重放对照还应考虑成本口径：stateless replay 把历史 output items 再放入下一次请求，服务端 response chain 则用 previous_response_id 引用先前状态；两种方式都不能只因请求体里少了历史正文就假定上下文免费或计费相同。对照实验应比较 provider 返回的 input/cache usage、链路保留状态、恢复成功与请求失败，并分别记录数据保留/存储设置。模型 endpoint 上有关 conversation storage 的政策也需独立核验，不能从状态引用本身推断数据保留周期。

若 request history 被手工摘要或删除，不要把摘要后的表现与官方 compaction 等同。应用摘要是自己的文本转换，可能漏掉具体回执、参数或阶段标记；tool search 的 schema 装载也影响后续可用工具集合。对每个 replay 变体使用相同事实来源和业务目标，逐项检查能否正确续接到下一动作，而不是只比较回答文本相似度。

## 7. 异步工具、steering 与后台响应是不同状态机

三者容易被混称为“异步 Agent”，但它们管理的是不同对象：

| 能力 | 在处理什么等待/变化 | 调用方仍要负责什么 |
|---|---|---|
| Async tool calling | 应用执行的慢 function/custom tool；模型可继续处理独立部分 | 启动 job、注册 task handle、保存原 <code>call_id</code>、timeout/重试/幂等、回灌结果 |
| Mid-turn steering | 同一个 WebSocket 响应运行时追加用户约束 | 保留 steering id/response lineage；核对 continuation 与原响应状态 |
| Background mode | 在兼容的模型/请求上让 Responses 生成任务在后台运行 | 检查当前模型资格，轮询响应状态、deadline、取消和结果恢复 |

Background mode、async tool calling 与 steering 不是同一种“后台任务”。Background mode 把一次 Responses 的生成放到后台并允许应用轮询响应；async tool calling 则让应用启动 function/custom tool job，模型可处理与它无依赖的工作；steering 把新输入追加到正在运行的 WebSocket Responses 工作流。当前公开的 Background mode 指南虽然在概述中举出 GPT-5.2 与 GPT-5.2 Pro，但也直接以 `gpt-6-astra` 展示 `background: true` 请求和轮询 `queued`、`in_progress` 状态直到终态。这足以说明 Astra 是该指南当前明确展示的后台响应模型；具体项目仍需核对账户、endpoint 与 model ID 的可用性。存储边界也影响轮询和恢复：ZDR 项目按 `store=false` 运行，响应数据会为异步执行与轮询暂存磁盘约 10 分钟；使用 MAM（包括增强型 MAM）的项目只有显式设置 `store=true`，才会在轮询结束后继续保留后台响应；省略该参数或设为 `false` 时，响应约 10 分钟后删除。应按项目的数据留存策略安排轮询与恢复。([Background mode](https://developers.openai.com/api/docs/guides/background)，[Async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling))

<code>async: true</code> 是应用执行工具的协议。应用在完整 tool call item 到达后启动任务，并将原始 <code>call_id</code> 与 <code>task_handle</code> 登记在 job registry；<code>task_handle</code> 应在整条对话中唯一，即使之前的任务已经完成。任务结果回来后，应用用原 <code>call_id</code> 回传，并延续最新的 <code>previous_response_id</code>。工具任务“已完成”和模型“已读到结果”是两个状态；后者必须等模型消费回执。托管内建工具不因此变为应用异步任务，官方也不建议把 async tool calling 与 multi-agent parallel tool calls 混用。([Async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling))

<code>response.steer.accepted</code> 表示输入排队，不表示模型已经按新要求行动。若不需要应用先给工具结果或审批，API 会自动创建 continuation；在此之前服务端完成当前 output item 和已经在跑的 hosted tool work。若 steering 中断原响应，它以 <code>incomplete_details.reason="steered"</code> 结束；若原响应先正常结束，则仍为 completed，但后面也可以出现 continuation。自动 continuation 继承请求设置，不过 token 和 tool-call 上限按每个 response 分别计算。不要再为同一 steering 手动额外发一个 <code>response.create</code>，也不要把每响应上限误当作 job 全局预算。([Mid-turn steering](https://developers.openai.com/api/docs/guides/steering))

steering 是追加控制输入，不是撤回操作。已经发出的文本无法从用户界面收回，已经执行的工具或外部写入也不会因新约束自动回滚。若 steering 改变了目标，应在下一步重新检查权限与产物状态，并对已执行副作用采用显式补偿或人工确认。

下面用合成的“核对发票并起草摘要”任务走一遍这几类 ID。首个 Responses 产生 <code>response_id=resp-41</code> 和 async function call <code>call_id=call-17</code>，参数给出唯一 <code>task_handle=invoice-check-2026-11-a</code>。应用先把三者和租户、工具 schema 版本、deadline 写入 registry，再启动只读发票查询。handle 必须在整段会话内唯一，即使它代表的工作已经完成；同一个远端 job 不能因下一轮重试而随便分配旧 handle。

在发票查询等待期间，模型可以继续处理不依赖发票金额的工作，例如整理报表结构。应用维护任务仍在运行的状态，也更新最新 response lineage；若查询已完成，就用原来的 <code>call-17</code> 返回工具输出。模型可能已经消费响应中其他部分，但工具“已完成”直到下一轮模型读到该回执才变成“结果已消费”。若用户在 WebSocket 响应运行中要求“只比较已付款发票”，应用记录 <code>response.steer</code> 及返回的 steering ID；<code>response.steer.accepted</code> 只说明更新排入队列，模型是否需要等工具结果或审批会影响后续 continuation 路径。

异步回执以原始 <code>call_id</code> 匹配请求，而续接时应沿用最新 response ID。比如 steering continuation 已生成 <code>resp-42</code>，发票查询稍后完成，应用在继续请求中把 <code>previous_response_id</code> 设为 <code>resp-42</code>，但结果仍匹配 <code>call-17</code>。这两类标识回答不同问题：response ID 确定对话从哪一轮继续，call ID 确定哪一个工具调用的结果正在返回。把它们调换会出现无主回执、重复执行或工具结果关联到错误轮次。

若 60 秒 timeout 发生，registry 先查远端 job 状态：仍 pending 就保留等待，不新建第二个查询；明确失败时才按预算判断是否可重试，并为新调用生成全新 task handle。写操作还需要远端幂等键，因为客户端超时不证明服务器未执行。背景 Responses 另有自己的 response ID 和状态轮询，不应拿 function 的 task handle 查询；steering 也有独立的 steer ID，接受或失败都要关联到目标 response。

此处的 ID 顺序是按官方协议组合出的教学 trace，不代表服务端事件在所有网络和请求条件下都按这一顺序到达。实际客户端要接受部分事件先后差异，按 ID 和状态转换处理重复/迟到事件；只有 API 返回的 completed/failed 状态和 executor 收据能确认相应层级的结果。接口错误后先恢复账本，再决定是否重试，不能凭前端 spinner 消失推断后台工具没有运行。

如果调用已发出且 steer 改变了授权范围，待完成工具仍需按原策略处理。只读查询可按租户边界返回或丢弃其结果；有外部副作用的任务不能因为更新指令而自行反向执行。应用必须知道哪个 worker 已经收到调用、哪个操作已提交、哪个结果还等模型消费，并在恢复时向用户展示真实状态。

将状态拆成三份账会比只写一个“agent status”更稳：response 账记录 created、in_progress、completed/incomplete/failed；工具调用账记录 proposed、authorized/denied、running、result-ready、returned；业务 job 账记录 pending、success、needs-review、cancelled。三者可以不同步，例如 response 已结束但工具仍在运行，或工具已完成而模型尚未读取结果。仪表板要显示这类跨层状态，不要因 response completed 就直接把 job 标成成功。

异常恢复也要分别执行：WebSocket 断线先查目标 response/steer 状态，不能默认 steering 已送达或自动重放；后台 response 按 response ID 轮询，并根据当前支持条件决定是否可恢复；async function job 从 job registry 检查 worker 是否运行或已提交，再把结果绑定回原 call ID。一个 ID 不可在三种注册表间互换，异步工具重试必须先解决未知状态，再明确新建或恢复操作。

应用若通过状态轮询或 WebSocket 重连发现同一个 completed event 两次，应使用事件 ID/操作 ID 做幂等记录，避免重复写入 UI 或重新派发工具结果。若旧 response 已失效、目标连接不可恢复，则创建新响应时带入外部账本摘要和经过核验的事实，并显式标为 recovery continuation；它是应用恢复决策，不能声称服务端自动恢复了原状态。

用户界面也应显示对象级状态：生成中的 response 可以显示“模型仍在处理”；应用工具 job 可显示“查询中/等待回执”；steering 更新可显示“已排队/已应用/失败”。不要用同一个 spinner 代表三件事。用户点取消时，应用先区分取消的是生成、工具 job 还是后续 continuation；若底层能力不支持可靠取消，就展示“取消请求已发送，结果待确认”，并在后台检查是否仍产生费用或副作用。

实际落地时可为每个异步任务设计至少两个独立标识：conversation 内唯一的 task handle 用于模型和应用的任务注册表，API call ID 用于把 function result 关联回原调用；远端服务若提供 request ID，则另存第三个 ID 用于查询执行事实。响应返回不同字段时建立显式映射，不用字符串相似或最近时间戳猜测。到达 deadline 时，应用仍需判断是否撤销 pending job、等待结果或向用户升级，不可默认重试。

steering 和 async tools 共享一条对话链时，应用要处理“目标改变时原工具仍在跑”的竞态。比如用户从“查所有账单”改为“只查已付款”，正在运行的旧查询可能仍返回所有结果；应用必须按原租户授权过滤结果，并重新决定是否在新目标下保留或丢弃它。已获得的宽范围数据不应仅因新指令而泄露给模型；权限缩小后，可清除不再允许使用的数据并审计清除行为。

## 8. Skills、AGENTS.md 与安全监控的适用范围

OpenAI 的技能文章建议用简短描述引导模型判断何时加载 skill，并采用“最小路由文档 + 按需读取支持文档”的渐进披露结构。由此得到的工程建议是：根级 <code>AGENTS.md</code> 应优先保留适用于所有任务的规则，将细节放到按需读取的支持文件；过多无关内容会占上下文，重复或冲突规则也会增加选错路径的机会。技能和指令应作为可评估的上下文路由设计，而不是 GPT-6 Astra 的隐藏记忆机制。评测时固定模型、harness、工具目录和权限，比较工具选择、无关 token、成功率、停止原因、压缩次数与单位成功成本。([Rethinking skills and prompts](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra))

Misalignment monitoring 是额外的安全控制面，不能当作通用的工具授权或独立 verifier。当前文档的覆盖范围取决于 API 和对话状态：

| 请求路径 | 文档描述的监控行为 | 实际边界 |
|---|---|---|
| Responses + persisted reasoning、WebSocket 或 OpenAI compaction | 可识别对话续接并自动阻断后续执行 | 不撤销此前已执行的工具或副作用 |
| Responses，不使用上述状态机制 | 可监控，并通过配置 webhook 告警 | webhook 告警本身不会自动停止对话 |
| Chat Completions | 不在该监控系统覆盖范围内 | 其他安全检查仍适用 |

被拦截的请求可能返回 <code>403</code> 和 <code>misalignment_policy_violation</code>；流式请求也要处理已经开始输出后出现的错误。收到该错误应冻结自动重试，保存 request/response/tool ID 和已发生副作用，再进入审计或人工处置。监控可能误报或漏报；权限、沙箱、审批与产物验证仍要独立部署。([Misalignment monitoring](https://developers.openai.com/api/docs/guides/safety-checks/misalignment-monitoring))

技能与根级指令可以看成一个有容量成本的路由层。假设有 12 个工作流，每份支持说明平均 1,000 token。若根文档一次把 12 份说明都展开，输入里至少有 12,000 token 与当前任务未必相关；若根文件保留 1,200-token 的公共规则和精确目录入口，只加载本次需要的一份 1,000-token 说明，则该任务约读 2,200 token，理论上少装入 9,800 token。对 100 个任务，简单乘法是 <code>100×9,800=980,000</code> 个输入 token。此数字是假设所有任务都不需要其余 11 份文档的容量算术，不等于实测的质量提升，也没有计入检索路由、重复前缀和缓存影响。

评估渐进披露时，任务集要覆盖需要不同支持文件的样例、含糊请求、规则冲突和错误路由。对照组与实验组保持模型、工具、权限、成功定义和输出预算一致，记录无关输入 token、技能选择是否正确、补读次数、工具误选、compact 次数、失败类型和成本/成功任务。若根级路由器删掉了所有任务都会用到的安全约束，token 下降也可能伴随权限误用；因此应把必须始终生效的授权、隐私和副作用规则保留在公共层，把只对某类任务适用的长解释移至专题文件。

安全监控的范围则是另一条轴。假设模型准备调用 <code>delete_record</code> 删除一条数据库记录，即使当前 Responses 路径启用了 misalignment monitoring，执行器仍须验证用户身份、租户、记录 ID、删除权限、审批要求和幂等键。应用先在授权层拒绝未批准的删除；监控系统是额外信号和阻断机制，而不是执行器的替代品。若监控报告此前意图存在风险，也不能撤销已提交删除，必须查询目标数据库并记录补偿状态。

运行时应区分自动阻断与 webhook 提醒。文档描述，使用 persisted reasoning、WebSocket 或 OpenAI compaction 的 Responses 请求可被识别为对话续接并阻断后续执行；不使用这些状态机制的 Responses 请求可通过 webhook 告警，但不会自动停止对话；Chat Completions 不在此监控系统覆盖内。响应流即使已经给过部分文本，后续仍可能收到错误，因此消费者不能在拿到首个 text delta 后就把 job 标为成功。([监控指南](https://developers.openai.com/api/docs/guides/safety-checks/misalignment-monitoring))

若收到 HTTP 403、错误码 <code>misalignment_policy_violation</code>，协调器应停止自动 retry，保存原请求、response ID、相关 tool call、worker 状态及已提交的外部操作，再进入审计。先确认该策略错误来自被监控请求，不能通过换 endpoint 逃避处理；根据证据由人工决定取消、隔离或补偿。若只是 webhook 告警，应用要查询 alert 详情并依据自己的策略暂停 job；收到了告警事件不等于 API 已阻断正在运行的其他响应。

因此，instruction routing 和安全监控都要有独立评测：前者看是否为任务装入正确上下文，后者看哪些请求被覆盖、阻断或只告警。最终权限仍由应用策略和 executor 强制执行，模型输出只作为提议；agent 的完整安全记录应包含用户/租户、指令与 skill 版本、请求路径、权限判定、已执行副作用、平台错误/告警、artifact 验证与人工处理结果。

可以用一组完全合成的安全样本说明为什么要分别报误报和漏报。假设评测集中有 100 次需要阻断的高风险行为和 100 次正常行为；监控拦下 70 次高风险行为、漏过 30 次，同时把 20 次正常行为标成告警。此时危险行为召回为 70/100=70%，被标记的 90 项中确属高风险的比例为 70/90≈77.8%，正常行为误报率为 20/100=20%。这些数只是构造出来的混淆矩阵，不能代表 OpenAI 的监控准确率；它展示的是即使响应内有阻断机制，也不能由“开了监控”推出没有危险动作。

评估集应有独立标注者、明确定义的行为类别、请求路径/状态分层和副作用结果真值。逐类报告漏报样本、误报对可用性的影响、流式部分输出、平台阻断与应用策略拒绝的区别；不要仅计算 webhook 数量。若监控覆盖被配置或 API 版本影响，需记录这些状态条件，并把工具执行前后分别采样，以确认错误发生前是否已经启动或提交副作用。

技能路由也要检查指令冲突，而不只是 token 节约。可给 100 个固定任务准备短路由与长根文档两个版本，逐项核对选中的 skill、必要授权规则是否被读取、产生的工具调用是否在允许集合内、是否在规定完成条件下停止。若短版把关键权限条件放到从未加载的专题文件，节省的 token 应视为安全退化；若长版含互相矛盾的步骤，则需统计选择错误与任务失败。胜负依据来自任务集与 verifier，不来自官方博客的产品描述。

跨控制面做防护时，至少让策略拒绝、平台监控、人工审批和结果 verifier 各有独立记录。比如未获批删除即使被模型输出、监控没告警、或工具返回成功，都不能跳过权限系统的确认；若平台返回错位错误，则把相关响应和工具操作冻结，并由人工决定恢复。多层控制的目标是降低单点失效后果，而不是宣称安全风险被消除。

监控触发后，响应流消费者应把已输出片段标记为 provisional，直到整条 response 到达完成状态并通过应用的安全检查。若错误在部分文本之后到达，UI 可以撤下或隔离输出，但不能假装用户没有看见；日志应注明已展示字符范围和时间。若之前已经调用工具，逐个查询其执行状态，必要时冻结后续 worker，不要只记录最终 403。

技能路由变更也需要版本化。对根文档、各 skill 内容和读取顺序计算 hash，任务 trace 写下实际加载集合及生效规则；当某个支持文档更改权限要求时，可重放旧任务确认旧 trace 使用哪版说明。无关文件未被读取时，不把它的修改归因于本轮行为。安全规则的变更还要通过人工审阅和固定负向任务集，避免仅用 token 下降作为压缩成功标准。

## 9. Compaction 是状态协议，不是可读摘要

server-side compaction 在 Responses 请求中设置 <code>context_management</code> 与 <code>compact_threshold</code>；达到阈值后，服务端会发出 opaque compaction item 并压缩此前上下文。采用无状态 input-array replay 时，应将该 item 连同 response output 一起保留；为减少延迟，可以丢弃最近 compaction item 之前的旧 items。若使用 <code>previous_response_id</code>，则只传新消息，不应自己裁剪服务端链路。独立 <code>/responses/compact</code> 会返回供下一次 <code>/responses</code> 原样使用的 canonical context window，其中可能既有 compaction item 也有仍保留的旧 items；不要再对其手动 prune。([Compaction](https://developers.openai.com/api/docs/guides/compaction))

这类压缩结果是不透明的，文档没有公开其内部算法或可供人读的完整摘要。它不能替代应用自己的任务状态：当前目标、权限、已提交副作用、未完成 job、工具结果、artifact hash 和 verifier 结论仍应存到外部账本。验收压缩时不要只问“摘要是否通顺”，而要问压缩前后的任务能否继续：模型是否还知道下一步、是否会重复执行已提交动作、工具结果是否仍能关联、权限是否仍有效、最终产物能否验证。

长上下文、tool search 和 compaction 各自解决不同问题：

| 方案 | 主要解决的问题 | 代价或失败边界 |
|---|---|---|
| 保留较长原始上下文 | 让本轮仍可访问更多原始信息 | token、延迟和费用上升；不能保证模型正确检索或综合远距离内容 |
| Deferred tool search | 不在首轮装入全部不常用工具 schema | 增加发现步骤；候选目录或描述不准时会漏选 |
| Compaction | 把旧上下文压成较短的可续接状态 | 内容变为 opaque 协议状态；需要保留 canonical item 并验证业务状态 |
| 外部任务/产物账本 | 在模型上下文之外保存权威状态和副作用证据 | 需要应用负责一致性、权限、清理、恢复和审计 |

面试中应先根据瓶颈选择机制：prompt 太长但工具目录小，先分析检索与裁剪；工具 schema 占比大，评估 defer；多轮历史持续增长，评估 compaction；副作用和恢复需要确定性时，外部账本仍不可省。把几个能力都称作“扩大记忆”会掩盖它们不同的输入、输出和恢复契约。

可以用“自动修复依赖升级”任务检验压缩前后的状态连续性。压缩前的输入数组包含用户目标、系统约束、工具 schema、response 的完整 output items、依赖扫描回执、补丁写入回执、测试结果和待处理的升级项。server-side compaction 在请求配置的阈值被触发时发出 opaque item，并继续生成；若使用 stateless input-array，后续输入按规定附加这些输出项。若采用 standalone <code>/responses/compact</code>，返回值可能同时含 compaction item 和仍保留的旧 items，下一次 Responses 应原样使用完整返回窗口。使用 <code>previous_response_id</code> 时则延续服务端链路，不要再手工剪裁该链路。

应用外部 ledger 记录的，是业务权威事实而不是可供模型阅读的摘要。例如目标仓库和分支、依赖文件哈希、操作 ID、补丁 artifact hash、测试环境和结果、审批人、尚未执行的发布、预算使用额与 deadline。工具 registry 还保存未完成任务的原始 <code>call_id</code>、task handle 和 worker 状态。任务完成性可以表示为一组逐步状态：扫描完成、补丁已生成、单元测试通过、集成测试待跑、发布未批准。每个状态带时间和证据地址，压缩后的对话可以重新读取它们。

续接时先对账，再让模型行动：取出 ledger 中最后确认的状态，查询 executor 是否仍有 pending job，比较 artifact hash 与目标分支当前版本，重新验证权限是否仍有效，然后将必要业务字段与 canonical context 一同交给下一步。若 ledger 表示补丁已提交、但压缩后的模型不确定，就读取 commit/hash 证据而不再次提交；若测试还在运行，先等待原 job 并用原 call ID 回传；若审批已过期，重新获取授权。这样可以把状态缺失修成“查询”而不是猜测性重做。

具体例子中，外部 ledger 记录 <code>artifact_hash=sha256:ab12…</code>、单元测试 <code>18/18</code> 通过、集成测试仍 pending、发布操作尚未授权。compaction item 是不透明上下文状态，不能保证自然语言里仍可见这四项业务事实。续接时执行器按 hash 检查补丁是否仍是同一份，测试服务查询原 job，权限服务重新确认审批；只有收到测试完成与发布授权后才进入后续操作。若模型只依据压缩后的自然语言说“都做完了”，那并没有证据证明发布已执行或验证完成。

压缩质量因此不能只用摘要相似度或语句流畅度衡量。应准备压缩前的业务状态真值，验证压缩后是否能正确报告未完成事项、是否保留工具调用关联、是否重复/漏做副作用、权限是否误过期以及最终产物是否通过同一 verifier。还可比较手工 summary、server-side compaction 和完整原始历史的任务成功、token、延迟与错误恢复，但要承认 compaction 内部算法不公开，不能从一次结果推断它具体保留了哪些 hidden state。

出错时的恢复顺序也要分层：先根据 response ID 与 mode 恢复协议上下文，再从业务 ledger 读取任务事实，随后查询工具 registry 和外部系统确认副作用，最后才让模型决定下一项工作。不得把手动删除 opaque item、重排 tool call/result 或复用过期权限当成普通 prompt 清理；这些做法可能破坏 API 协议和事务状态。compaction 让长对话更易续接，却不承担数据库、审计日志或分布式事务的职责。

比较 server-side 与 standalone 流程时，保存的是不同的输入边界。server-side 模式中，服务端在达到设置的阈值时压缩并把 opaque item 放进本次响应；stateless client 可保留完整 output sequence，或依文档在最新 compaction item 后缩短旧前缀。standalone compact 则把整个返回窗口作为下一次 Responses 的 canonical context；这两种规则不能混成“找到压缩项就把更早的 item 全删掉”。如果采用 previous_response_id，客户端应使用服务端链而不在本地重建前缀。

一次恢复验收可以设置故障注入：压缩前让两个只读 job 中一个 completed、另一个 pending，补丁操作已生成但尚未审批；随后让客户端断线并重新启动。恢复程序要分别从 API、工具 registry、版本控制和审批服务读取状态，再检查是否能回答“哪个 job 的结果未消费”“哪个 artifact 尚未批准”“是否已发生写入”。如果它仅凭 compaction 后可读的上下文猜测状态，实验就没有通过。

当业务状态和对话状态冲突时，优先从对应权威系统刷新：工单系统确认当前任务状态，版本控制读取实际 commit，支付系统查交易 ID，权限服务取当前审批，工具 registry 确认异步 job。新确认的事实作为带时间戳的输入加入下一轮，并标明它覆盖了哪条旧信息。只有在这些证据一致后才继续副作用；不一致时暂停并交给人工或补偿流程处理。

压缩带来的 token 节省应与恢复成本一起评估。若压缩令每个 job 平均少输入 100K，但漏掉关键回执的概率增加，使 1,000 个任务中 3 个需要人工恢复，团队就要将恢复费用、额外延迟和潜在副作用纳入成本/成功任务。100K 是示例差值，3 次也是假设；目的是明确“更短 prompt”与“更可靠系统”是不同测量维度。

在一个假设任务中，执行器曾提交操作 op-9，远端返回 transaction ID tx-31，随后上下文压缩。若继续时模型记不清是否执行，应用先按 op-9/tx-31 查询实际状态；查询确认已提交就把收据作为新输入，不再重复写。若查询结果是未知，则不直接重试，而先检查服务端幂等支持、操作窗口和可补偿性；无法确认时转人工。这个逐步决策保护了外部状态，即使模型上下文丢失了可读的中间细节。

压缩验收应覆盖不同任务阶段，而不仅是“聊天记录很长”的单一用例：没有副作用的资料研究、等待中的异步工具、已生成但未批准的补丁、刚发生权限撤销、部分流式输出和跨版本迁移。对每一种情况，准备压缩前状态快照和 expected next action，检查任务是否重复、遗漏或越权。由于 compaction item 内部不透明，验收判断应落在 API 交接、外部状态和最终 artifact 上。

## 10. 本章相关 GPT-6 模型与 GPT-5.3 Codex：只比较已公开合同

本节只对照旧稿已涉及的 model IDs，不是 GPT-6 家族的完整目录。本次核验的官方定价页也列出 GPT-6.1 Terra 等相邻模型；它们的接口和能力不能由本节 Astra/Sol/Luna 对照推断。新部署应按实际 model ID 查看当日模型目录、模型页与参数兼容文档。([API 定价页](https://developers.openai.com/api/docs/pricing))

GPT-5.3 Codex 可用作历史对照，但当前官方页面已标为 Deprecated。它列出 400K context window、128K max output，没有单独列 maximum input；<code>400K - 128K = 272K</code> 同样只是把全部输出额度预留后得到的算术余量。页面列出 Chat Completions 与 Responses 端点及 function calling，不能再写成 “Responses-only”；迁移或新部署应先确认模型是否仍可用以及目标功能的 endpoint 兼容性。([GPT-5.3 Codex 模型页](https://developers.openai.com/api/docs/models/gpt-5.3-codex))

GPT-6 Astra、GPT-6 Sol、GPT-6 Luna 与 GPT-6.1 Sol 的当前模型页都列出约 1.05M context window 和 128K max output，但模型页未分别列出 maximum input。模型之间不能只按窗口大小或产品定位推断质量、速度、架构或成本；reasoning options、tool calling endpoint、费率和区域处理各自要按精确 model ID 核验。([Astra](https://developers.openai.com/api/docs/models/gpt-6-astra)，[GPT-6 Sol](https://developers.openai.com/api/docs/models/gpt-6-sol)，[GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)，[GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol))

当前 GPT-6 family guide 说明：Astra 不支持 <code>none</code> effort，Astra 的 reasoning tool calling 需要 Responses；Sol/Luna 在 <code>reasoning_effort="none"</code> 时才支持 Chat Completions function calling。Sol/Luna 模型页列出的 endpoints 与功能还应逐个按当前 model ID 检查，不能由端点列表推断全部参数组合都兼容。<code>reasoning.mode</code> 的 <code>standard/pro</code> 与 <code>effort</code> 是两个不同维度。GPT-6.1 Sol 另有 <code>reasoning.context="all_turns"</code> 等状态选项；不要把它们无条件迁移到 Astra。([GPT-6 family guide](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra)，[Reasoning](https://developers.openai.com/api/docs/guides/reasoning)，[GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol))

<code>gpt-6-sol</code> 的当前官方页指向较新的 <code>gpt-6.1-sol</code>；精修部署文档时应把 <code>gpt-6-sol</code> 当作较早模型页下的合同快照，而非整个 Sol 产品线的最新推荐。GPT-6 Luna 在既有模型目录中被列为高吞吐/高频使用定位；这是产品路由线索，不是参数量或 kernel 结构证据。对 Sol、Luna 的第三方评测应绑定自己的模型 ID、effort、provider、harness、任务集、工具、verifier 和时间，不迁移到 Astra 或 GPT-6.1 Sol。([GPT-6 Sol](https://developers.openai.com/api/docs/models/gpt-6-sol)，[GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol)，[GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna))

用一个纯算术例子可以看出为何历史窗口字段不应混成同一个部署承诺。假设一个任务本轮计划输入 300K、预留输出 80K，总计 380K token。Astra 的 1.05M context 减去该计划总量，算术上还剩 670K；GPT-5.3 Codex 的 400K context 减同一总量，算术上还剩 20K。两者的 80K 输出都低于各自列出的 128K max output，但这不证明 300K 是官方承诺的 maximum input，也没有计入系统、工具、媒体等真实请求构成。实际容量仍要按具体 endpoint 和请求状态观察。

这组比较只用于解释窗口余量，不是模型质量、速度或选择建议。GPT-5.3 Codex 的当前模型页标为 Deprecated，因此它是历史接口对照；不要把其旧能力描述成新部署选项。GPT-6 Sol 模型页明确指向较新的 GPT-6.1 Sol，而 Astra、Sol、Luna 即使有相同的公开 context/output 数值，也可能在 effort、Chat Completions function calling 条件、region、service tier 和价格上不同。迁移 manifest 应绑定 model ID、endpoint、mode/effort、tool schemas、region、价格快照及验收数据，不能复制另一模型的配置或评测分数。

本章与开放权重部署不能混用一张账本。gpt-oss-120b/20b 的部署成本取决于参数驻留/分片、专家路由、MXFP4 权重布局、KV cache、通信、kernel 与 workspace；API 模型则还要记录缓存档位、服务 tier、托管工具费用和账户价格。两类系统都需固定 model/revision、tokenizer/template、任务集、硬件/runtime 和 verifier 后再比较延迟、吞吐、质量与单位成功成本。

模型迁移矩阵至少应有 model ID、页面日期、context、max output、页面是否公开 maximum input、reasoning effort、mode、工具调用 endpoint、可用 service tier 和数据区域。当前对照中，Astra 列出的 effort 不含 none，GPT-6 Sol 与 Luna 的 Chat Completions function calling 条件是 none；较新的 GPT-6.1 Sol 工具调用须走 Responses，且不支持 none。GPT-5.3 Codex 已 deprecated。把这些字段拆开，可以避免把“有 Chat Completions endpoint”误写成“所有工具调用都能通过 Chat Completions”。

模型页面会随生命周期更新，因此每次迁移还要区分服务商公开的状态与本地实验结果。公开 context、endpoint 和费用可填进合同矩阵；任务成功率、p95、错误恢复率与单位成功成本必须由固定 harness 实测。发布文案中的智能、快速或高性价比属于供应商定位，不应作为对生产工作负载的独立质量证明；参数量、训练细节和内部 serving 实现仍未在这些页面披露。

表中可以把 300K 输入/80K 输出方案作为回归样例，核对每个 model ID 的窗口字段和输出约束，再按当日公开费率计算任务成本。若从旧 GPT-6 Sol 页面迁移到 GPT-6.1 Sol，即使公开窗口值近似，也要重新确认 none effort、tool endpoint、Fast/Ultrafast 区域资格与价格；旧压测样本仍绑定旧 model ID。若目标改成 Astra，工具调用和输出状态规则要按 Astra 的当前页面重新构造，不可仅以 context 更大作为兼容证据。

deprecated 标签也会改变工程决策：可以把页面数字用于解释旧账单或阅读历史配置，但新项目不应将其作为默认可部署模型。部署 manifest 可记录 observed_at、deprecated/current 状态、API endpoint 和替换评测；若服务商停止旧模型，迁移方案必须有行为回归和回退目标，而不是照搬 400K/128K 两个容量数字。

## 11. 面试追问与自查练习

**问：1.05M context window 能直接输入 1.05M token 吗？**

不能这样回答。官方单独列出最大输出 128K，但没有在 Astra 模型页列出独立 maximum input。922K 只是预留满额输出后的理论减法结果；工具定义、系统指令、历史状态和推理输出还要共享窗口。应按当前接口与响应状态测可用输入空间。

**问：把 effort 调到 <code>max</code> 会不会一定提高质量？**

不会。它只调节该模型接受的推理预算，不是跨模型的质量等级。若任务瓶颈在数据、工具、权限或 verifier，升 effort 不会自动补足这些缺口，还可能加大 token、延迟和任务费用。

**问：为何请求设置了较大的 <code>max_output_tokens</code>，结果仍然没有答案？**

reasoning token 也会消耗输出额度。若 reasoning 已触及输出上限，响应可能以 <code>incomplete</code> 结束而没有可见文本；要结合 <code>status</code>、<code>incomplete_details</code> 和 usage 判断。下一次可减少输入、给可见输出留空间，或重新设计任务/验证路径。

**问：工具调用并行是否一定降低成本？**

不会。并行可能降低等待时间，但工具请求费、执行器资源和模型处理工具结果的 token 仍然消耗。并发还可能增加峰值资源与外部写入冲突；必须分别统计墙钟延迟、调用数、费用、结果 token 和成功率。

**问：缓存读写怎样进入预算？**

把缓存读取、30 分钟写入、12 小时写入和普通未缓存 token 分开；超过 272K 输入时按整次请求切到 long-context 费率。缓存命中减少 token 价格不等于缓存无成本，cache write 和上下文变化也可能影响复用。完整费用还须加入工具、重试、验证与人工接管。

**问：服务端 compaction 是不是生成了一段普通摘要？**

不应这样假设。compaction item 是 opaque 的协议状态；standalone endpoint 返回的 canonical window 可能还包含其他 item，必须按文档原样交给下一请求。业务目标、副作用和 artifact 仍由应用自己的持久状态负责。

**问：Misalignment monitoring 能代替权限检查吗？**

不能。它只在文档说明的请求类型和对话状态下监测；一些 Responses 请求只有 webhook 告警而不会自动停止，Chat Completions 不在该系统覆盖内。即使拦截成功，也不会撤销此前已经发生的副作用。

把一次真实任务建成可比较的实验时，至少固定 model ID/snapshot、mode、effort、context reserve、服务档位、工具 schema/权限、最大调用数、deadline、重试/幂等规则、compaction 策略、verifier 和成功定义。报告 p50/p95 延迟、reasoning 与可见输出 token、cache read/write、工具数、超时/恢复率、任务成功率、成本/成功任务及失败样本。没有这些字段，“长上下文更强”或“高 effort 更聪明”都无法指导工程选择。

## 小练习

1. 给定 Astra 的 <code>1,050,000</code> context 与 <code>128,000</code> max output，推导 <code>922,000</code> 为什么只是算术余量；把工具 schema、历史输入、reasoning 和输出预留画进一个请求账本。
2. 用本节 Python 示例计算 300K 输入、不同 cached/read/write 分布和 30m/12h 缓存写入；覆盖阈值恰好为 272K 与超过 272K、非法 tier、负 token、浮点 token 等边界。
3. 对同一个任务比较 <code>standard</code> 与 <code>pro</code>、<code>low</code>/<code>high</code>/<code>max</code> 的组合；固定工具、输入、超时和验证器，报告成功率、reasoning token、可见输出、TTFT/TPOT 与单位成功成本。
4. 设计一个整项 Agent job budget，分别给出模型总额、每类工具调用数、最大并发、单工具 timeout、重试条件、verifier 预算、deadline 和副作用审批门禁。
5. 为 hosted shell、MCP、apply patch 和 computer use 画出 model、API、宿主 executor、沙箱、策略、审批和审计日志的责任边界。
6. 比较无状态完整 output item replay、只保留最终文本、<code>previous_response_id</code> 三种做法；说明各自保留什么状态、缓存与恢复如何核验。
7. 为 deferred tool search 设计 schema 版本、namespace、工具移除和权限撤销的审计流程；对比 eager 与 deferred 的上下文、发现步骤和漏选风险。
8. 模拟 async function tool：并发启动两个只读 job，再发生 timeout、重复完成回传和响应 lineage 更新；分别记录 original <code>call_id</code>、unique task handle、执行器状态和模型是否已消费结果。
9. 画出 <code>response.create -> response.created -> response.steer -> response.steer.accepted -> continuation</code> 的状态图；另覆盖原响应先 <code>completed</code>、steer 中断原响应、需要 app tool result/approval 三条路径。
10. 扩展安全故障演练：<code>misalignment_policy_violation</code> 到达后停止自动 retry，审计已启动副作用，再决定人工检查或补偿；明确 webhook 告警不等于执行阻断。
11. 对比服务器 compaction、standalone <code>/responses/compact</code>、长原始上下文和应用外部任务账本；说明每项解决什么问题、损失/边界是什么、需如何验证状态连续。
12. 把一个过长的 skill/<code>AGENTS.md</code> 拆成最小路由入口和渐进披露支持文件，固定同一任务集比较无关上下文 token、工具选择、停止原因、压缩次数与成功率。
13. 为 async tool、background response 和 WebSocket steering 分别设计状态记录；给重复 task handle、迟到结果、steering 失败、响应连接断开和权限撤销写拒绝/恢复路径。教学状态图必须注明是本地模型，不要据此声称复现了服务端事件调度。
14. 以 300K 输入、80K 输出比较 Astra 与已标记 deprecated 的 GPT-5.3 Codex 窗口算术余量；说明为什么算术余量不能写成 maximum input，再用当前 model ID 检查 endpoint、effort、region 与价格。

逐节来源、当前官方接口事实与未验证边界见[研究记录](../../research/openai-gpt-6-astra-api-contract-2026.md)。本章数值账本、job registry、工具生命周期和恢复轨迹均为合成教学案例，不调用真实 API，也不证明服务端事件时序、监控准确率或生产吞吐。
