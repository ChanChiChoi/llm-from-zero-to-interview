# 第十二章：Safety 的系统化表达与综合复习

前面十一章分别讨论了 alignment、监督、奖励错配、越狱、红队、解释性、steering、model editing、隐私和治理。真正困难的地方，是把这些知识放回同一个系统问题里：

> 一个模型在什么条件下可以帮助人完成任务？当它不确定、被攻击、接触私有数据或拥有工具时，系统如何发现风险、限制动作、保留证据并持续修复？

Safety 不是一句“让模型拒答危险问题”。过度拒答会损害正常用户，过度迎合会放大危险，模型外的 RAG、工具、日志和权限还会引入新的失败路径。高质量的安全判断需要同时说明目标、威胁、机制、测量、控制和残余风险。

本章把全册内容串成一条可以复述、实现和审计的推理链：

1. 先定义希望系统保护什么，以及什么行为算成功。
2. 再区分错误来自目标、数据、模型、应用还是组织流程。
3. 为每条风险路径定义有分母的指标和代表性样本。
4. 把训练、策略、权限、工具、人工和回滚放在相应的位置。
5. 发布时只声明证据覆盖的范围，发布后继续监控和复盘。

这套方法适用于学习、研究、工程设计和技术沟通。它的价值不是背出更多术语，而是在面对陌生模型、陌生产品或不完整资料时，仍然能够给出可验证的判断。

## 1. Safety 的对象：模型行为还是系统后果

### 1.1 三个层次

Safety 问题至少有三个层次。

第一层是输出行为：模型是否生成有害内容、泄露 PII、伪造引用、过度拒答或表现出不确定性。

第二层是交互行为：模型是否把外部文档当成指令，是否接受了越权工具参数，是否在多轮对话中逐步改变安全边界，是否把内部状态暴露给用户。

第三层是现实后果：工具是否真的修改了数据库，Agent 是否真的发送邮件，代码是否真的执行，敏感日志是否真的被保存和传播。

同一个文本输出在三个层次上的含义可能不同。一个看似无害的工具参数如果触发了高权限写操作，风险不在句子本身，而在系统授予了它什么能力。反过来，一个危险话题的高层教育回答可能没有现实伤害，不能只根据关键词判定。

### 1.2 Safety 与 Alignment

Alignment 关注系统实际优化的目标和行为是否符合人类意图、价值和约束。Safety 关注这些行为及其部署组合是否会造成可接受范围之外的伤害。两者重叠，但不是同义词。

例如，客服 Agent 可能非常“对齐”地追求提高用户满意度，却为了减少投诉而隐瞒退款条件；它在局部代理目标上表现良好，系统安全和诚实性却失败。另一个模型可能在危险请求上拒答，但对正常医疗问题全部拒绝，也不满足有用性和公平性要求。

可以把安全效用写成多个目标的组合：

$$
U_{\mathrm{safety}}=\alpha H+\beta T+\gamma C-\delta R-\eta O
$$

其中 \(H\) 是对正常任务的帮助程度，\(T\) 是事实和证据忠实度，\(C\) 是对高风险动作的可控性，\(R\) 是严重度加权风险，\(O\) 是过度拒答或不必要阻断，\(\alpha,\beta,\gamma,\delta,\eta\) 是任务相关权重。

这个式子不是要求把所有价值压成一个生产分数，而是提醒我们：只优化其中一项，其他项就可能退化。发布报告应并列展示这些信号和它们的分母。

### 1.3 HHH 不是三个独立按钮

Helpful、Honest、Harmless 经常被当成三个口号，但它们在真实场景中会冲突：

1. 用户请求危险操作时，直接帮助可能损害 harmless。
2. 用户要求一个模型不知道的事实时，helpful 和 honest 冲突。
3. 对所有敏感领域都拒答，虽然降低一部分风险，却损害 helpful。
4. 过于详细地披露内部安全规则，可能损害系统 security。

安全系统需要把冲突转成行为层选择：澄清、限制范围、引用证据、提供安全替代、只读执行、人工确认或拒绝。目标不是让一个模型在每次冲突中凭直觉做哲学决定，而是把高风险动作交给可审计的策略和权限服务。

### 1.4 小白例子：机场安检

可以把 Safety 想象成机场安检。安检不是把所有人都拒绝登机，而是：

1. 先确认旅客和行李的身份边界。
2. 把普通物品、需要检查的物品和明确禁止的物品分层。
3. 对不确定情况进行额外检查。
4. 记录关键决定。
5. 发现异常时暂停相关流程，并让有权限的人处理。

模型的输出过滤相当于检查一个结果，但工具授权、租户权限、沙箱和人工确认才相当于控制行李真正能不能被带上飞机。只在文本出口贴一个分类器，不能覆盖整个系统。

## 2. 从真实目标到代理目标

### 2.1 为什么安全目标难以直接训练

“帮助用户解决问题”“不伤害任何人”“诚实表达不确定性”都不是一个可以直接计算的单一标签。训练团队通常需要把它们转成可观察代理：

1. 人类偏好比较。
2. 安全分类标签。
3. 程序验证结果。
4. 引用是否支持答案。
5. 工具参数是否符合 schema。
6. 用户是否完成任务。

代理越容易测量，越可能遗漏真正目标。长度、流畅度、自信语气和拒答模板都可能提高代理分，却不一定提高真实质量。

### 2.2 RLHF 的价值与边界

RLHF 用人类偏好训练奖励模型，再优化策略。一个简化目标可以写成：

$$
\max_{\pi_\theta}\;
\mathbb E_{x,y\sim\pi_\theta}[r_\phi(x,y)]
-\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
$$

其中 \(\pi_\theta\) 是当前策略，\(r_\phi\) 是奖励模型，\(\pi_{\mathrm{ref}}\) 是参考策略，\(\beta\) 控制偏离参考模型的惩罚。实际系统还会受到采样、优势估计、批次、奖励缩放和训练稳定性的影响。

RLHF 能把很多人类偏好注入模型，是重要的后训练方法；它不能保证奖励模型等于真实目标。偏好数据可能包含标注员风格、长度、品牌和文化偏差，奖励模型也可能在分布外失效。

### 2.3 DPO 改变优化形式，不消除目标错配

DPO 直接用 chosen/rejected 偏好对优化策略和参考策略的相对 log probability。教学形式可以写成：

$$
\mathcal L_{\mathrm{DPO}}
=-\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right]\right)
$$

\(y_w\) 是偏好回答，\(y_l\) 是非偏好回答。DPO 减少了显式奖励模型和 PPO 循环的工程复杂度，但它仍然依赖偏好数据的构造。如果 chosen 只是更长、更自信或更符合标注模板，模型会稳定地学到这些表面信号。

### 2.4 Reward hacking 的诊断

当代理奖励 \(r\) 上升而真实质量 \(q\) 下降时，可以观察 reward-human gap：

$$
\Delta_{\mathrm{RH}}=\mathbb E[r]-\mathbb E[q]
$$

更重要的是按任务和严重度切片。常见症状包括：

1. 回答变长，但关键事实没有增加。
2. 语气更自信，但引用更不可靠。
3. 拒答率升高，安全分数上升，但正常任务无法完成。
4. 引用数量增加，却没有支持正文中的 claim。
5. 工具调用格式正确，却选择了不该调用的工具。

修复不一定是重新训练。可以先检查奖励标签、加入 hard negatives、降低优化强度、引入程序验证、提高人工抽检，并把高严重度反例放入回归集。

### 2.5 worked example：客服回答

假设奖励模型偏爱“礼貌、完整、承诺明确”的回答。一个回答说“我已经为您完成退款，款项会在今天到账”，可能获得很高的风格分；但系统实际上没有调用退款工具，订单也不满足退款条件。

这个例子把四个事实分开：

1. 语言质量可能很高。
2. 用户满意度预测可能很高。
3. 业务事实是错误的。
4. 工具状态没有发生变化。

安全评估不能只看回答是否像客服，而要看 claim 是否有证据、工具是否真的成功、失败时是否诚实说明状态，以及错误是否会造成财务后果。

## 3. Scalable Oversight：当监督跟不上能力

### 3.1 监督瓶颈

一个人可以直接判断短答案是否符合要求，但很难逐行检查长代码、复杂证明、长周期 Agent 轨迹或专业研究报告。模型能力提高后，监督成本、注意力和领域知识可能成为瓶颈。

Scalable Oversight 试图让监督随任务规模增长，同时保持人类价值和安全约束。它不是“让 AI 自己监督自己”这么简单，而是要说明哪些部分可以自动化，哪些部分必须由人工或专家校准。

### 3.2 监督方法的假设

| 方法 | 主要想法 | 关键假设 | 主要风险 |
| --- | --- | --- | --- |
| 任务分解 | 把复杂任务拆成小问题 | 子问题组合后仍保留真实目标 | 分解错误、接口遗漏 |
| AI critique | 用模型检查答案 | critique 能识别关键错误 | judge 盲点和迎合 |
| Debate | 让不同方案相互质询 | 人类能判断更好的论证 | 说服力胜过真实性 |
| Constitutional AI | 用原则生成反馈和修正 | 原则足够明确且覆盖冲突 | 原则选择和解释偏差 |
| 人机混合 | AI 扩展筛选，人工审计 | 风险切片能被正确升级 | 自动筛选漏掉高风险样本 |

### 3.3 监督指标

设真实错误集合为 \(E\)，系统发现的错误集合为 \(\hat E\)，过程监督可以报告：

$$
Recall_{\mathrm{oversight}}=\frac{|E\cap\hat E|}{|E|}
$$

AI feedback 还应与人工 gold set 比较：

$$
Err_{\mathrm{AI\ feedback}}=
\frac{\text{AI feedback 与专家判断不一致的样本数}}
{\text{被 AI feedback 覆盖的样本数}}
$$

监督的成本节省也要按“单位成功发现的严重错误”计算，而不是只报告调用次数少了多少。低成本但漏掉高严重度错误的监督器，不能简单称为可扩展。

### 3.4 什么时候升级人工

自动流程至少应在以下情况升级：

1. 输出涉及医疗、财务、法律或关键基础设施。
2. 模型的证据和引用不一致。
3. 工具动作不可逆或影响第三方。
4. 多个评估器意见分歧。
5. 样本落在训练和评估分布之外。
6. 涉及隐私主体、删除请求或安全事故。

人工升级不是流程失败。它是把有限的专家注意力用在自动系统最不可靠或后果最严重的地方。
## 4. Jailbreak 与 Prompt Injection 的边界

### 4.1 Jailbreak 是模型策略边界被绕过

Jailbreak 通常指用户通过改写、角色包装、多轮对话、编码或上下文操纵，使模型输出本来受到安全策略限制的内容。评估它时，不能只看某个字符串是否出现，要看：

1. 请求是否真的属于被禁止的行为。
2. 输出是否提供了可执行帮助。
3. 结果是否比安全替代更危险。
4. 多轮尝试和预算是多少。
5. 正常安全教育请求是否被误拒。

防护可以分为安全训练、输入和输出分类、策略路由、人工复核、速率限制和持续红队。没有一种方法能覆盖所有自然语言组合，因此必须报告样本范围和未知攻击风险。

### 4.2 Prompt Injection 是信任边界混乱

Prompt injection 的核心不是用户一定想绕过模型安全，而是系统把不可信文本误当成高优先级指令。来源可能是网页、邮件、RAG 文档、工具返回、代码仓库或图像中的文字。

最小防御原则包括：

1. 明确 system、developer、user、external content 和 tool result 的信任等级。
2. 把外部内容当证据或数据，不当作系统指令。
3. 在检索前按服务端身份做权限过滤。
4. 工具调用由服务端检查 schema、资源和动作范围。
5. 高风险动作需要确认、幂等、沙箱和回滚。
6. 对多轮和间接注入做轨迹级回归。

### 4.3 两类风险的比较

| 维度 | Jailbreak | Prompt Injection |
| --- | --- | --- |
| 主要目标 | 绕过模型行为限制 | 劫持应用的指令和动作流程 |
| 常见来源 | 当前用户和对话历史 | 文档、网页、邮件、工具返回 |
| 主要对象 | 模型安全策略 | LLM 应用的信任边界 |
| 典型后果 | 有害回答、危险细节 | 数据外泄、越权工具、错误写操作 |
| 关键控制 | 安全训练、分类、红队 | 数据/指令隔离、权限、动作确认 |

两者可以叠加。一个恶意文档先污染 Agent 的上下文，再诱导它调用高权限工具，既是注入问题，也可能暴露模型的越狱脆弱性。

### 4.4 正常任务不能被安全措施摧毁

安全评估必须加入干净样本。只报告攻击成功率会诱导系统通过全面拒答来获得好看的数字。至少要并列：

1. 攻击下的有害遵循率。
2. 正常请求的任务成功率。
3. 安全教育和防御请求的帮助质量。
4. 误拒率和澄清率。
5. 工具越权率和正常工具完成率。

## 5. Safety Evaluation：先定义任务再算指标

### 5.1 一条评估样本的契约

一次评估不只是一个 prompt。可以记录：

$$
z_i=(x_i,y_i,c_i,s_i,r_i,h_i,\ell_i)
$$

其中 \(x_i\) 是输入，\(y_i\) 是输出或轨迹，\(c_i\) 是任务契约，\(s_i\) 是风险切片，\(r_i\) 是真实结果，\(h_i\) 是 harness 和工具条件，\(\ell_i\) 是日志和证据状态。

如果样本没有任务契约，就无法判断“拒答”是正确安全行为还是不必要阻断；如果没有工具和权限条件，就无法把模型行为推广到 Agent 系统。

### 5.2 风险 taxonomy

一个通用 Safety eval 可以包括：

1. 有害内容和危险建议。
2. 隐私、记忆和敏感信息泄露。
3. Jailbreak 和多轮绕过。
4. Prompt injection 和数据外泄。
5. 工具误用、越权和不可逆动作。
6. 偏见、群体差异和不公平拒绝。
7. 事实性、引用和不确定性。
8. 长周期状态漂移和恢复失败。
9. 高风险专业能力和 capability uplift。
10. 过度拒答、正常任务失败和可用性退化。

分类不是越多越好。每一类都要能定义样本、分母、指标、责任人和修复路径。

### 5.3 有害遵循率与误拒率

在明确属于禁止请求的集合 \(H\) 上，设模型产生不可接受帮助时 \(u_i=1\)：

$$
R_{\mathrm{unsafe}}=
\frac{\sum_{i\in H}w_i u_i}
{\sum_{i\in H}w_i}
$$

在允许请求集合 \(A\) 上，设模型错误拒绝或没有完成任务时 \(o_i=1\)：

$$
R_{\mathrm{overrefusal}}=
\frac{\sum_{i\in A}v_i o_i}
{\sum_{i\in A}v_i}
$$

两者的权重和样本必须分开。用所有请求作一个拒答率分母，会把危险请求和正常请求混在一起，失去解释力。

### 5.4 安全替代质量

拒绝危险细节后仍可以提供风险解释、合规建议、预防措施或求助渠道。设安全替代满足相关性、可执行性和不泄露危险细节三个条件时 \(a_i=1\)，可以报告：

$$
Q_{\mathrm{safe\ alternative}}=
\frac{\sum_{i\in H}w_i a_i}
{\sum_{i\in H}w_i}
$$

这个指标不应奖励冗长模板。人工评分需要区分“看起来礼貌”和“真正帮助用户完成安全目标”。

### 5.5 攻击成功与工具越权

攻击成功率应该绑定攻击任务和搜索预算：

$$
ASR(B)=
\frac{\text{在预算 }B\text{ 内达到攻击目标的测试任务数}}
{\text{执行了预算 }B\text{ 的测试任务数}}
$$

工具越权率应只在无权或不应执行的动作集合 \(U\) 上计算：

$$
R_{\mathrm{unauth\ tool}}=
\frac{\text{越权工具动作数}}
{\text{无权工具动作尝试数}}
$$

不能把一次模型拒答和一次服务端阻断都算成同一种成功；轨迹需要标记阻断发生在哪一层。

### 5.6 严重度加权失败

对每个失败事件 \(j\)，用 \(s_j\) 表示严重度、\(p_j\) 表示事件是否发生，报告：

$$
R_{\mathrm{severity}}=
\frac{\sum_j s_jp_j}
{\sum_j s_j}
$$

这是比较和排序的教学指标，不是现实损失的精确估计。重大事件应单独呈现，不能被大量低严重度样本平均掉。

### 5.7 Safety Eval 与 Red Team 的关系

Safety eval 侧重预先定义的任务、协议和指标，适合做版本比较和回归；red teaming 侧重主动探索未知失败，适合发现固定测试集没有覆盖的路径。两者不是互斥的：

1. 评估集提供稳定基线。
2. 红队发现新失败。
3. 新失败经过确认、脱敏和分类后进入回归集。
4. 修复后重新运行固定集和探索性红队。

红队样本不能直接当作总体概率估计。它通常是有目标、有搜索偏差的样本，报告时必须写明抽样和预算。

## 6. Dangerous Capability 与 Capability Elicitation

### 6.1 危险能力不等于危险文本

一个模型生成了高层网络安全解释，不等于它能独立完成真实攻击；一个模型没有主动输出危险细节，也不等于在工具、代码执行、专家辅助和长周期规划下没有危险增量。

危险能力评估关注模型是否降低了高风险任务的门槛，常见维度包括：

1. 任务完成率和隐藏测试通过率。
2. 所需人工知识和操作步骤是否减少。
3. 工具、网络、代码执行和多轮预算。
4. 失败是否容易被发现和恢复。
5. 相对于非 AI baseline 是否有实质增量。

### 6.2 Capability Elicitation

Capability elicitation 是用合理 prompt、工具、scaffold、分解、重试或专家辅助，测量模型可能达到的能力上限。它有两个作用：

1. 避免只用简单 prompt 低估风险。
2. 避免把不现实的最大化 harness 当成普通用户风险。

因此报告至少要同时记录自然部署条件和增强评估条件。增强条件下发现能力，不代表所有用户都能获得；但发现能力后也不能假设生产中永远不会被工具或 Agent harness 激发。

### 6.3 Baseline 与增量

设模型系统在条件 \(c\) 下的任务成功率为 \(S_{\mathrm{AI}}(c)\)，非 AI baseline 为 \(S_{\mathrm{base}}(c)\)，可以观察：

$$
\Delta_{\mathrm{cap}}(c)=S_{\mathrm{AI}}(c)-S_{\mathrm{base}}(c)
$$

baseline 可以是搜索引擎、普通脚本、专家流程或没有模型的现有系统。选择哪个 baseline 会影响解释，必须说明它的成本、时间、知识和工具条件。

### 6.4 结果如何进入发布范围

危险能力结果不应只写成“高”或“低”。更有用的记录包括：

1. 能力在哪些任务和工具条件下出现。
2. 需要多少预算、重试和专家辅助。
3. 哪些动作被服务端阻断。
4. 评估是否覆盖现实部署的身份和权限。
5. 是否有可逆操作、人工确认和事故响应。

当能力增量明显而控制证据不足时，可以先开放只读、沙箱或可信访问，而不是直接开放生产写入。

## 7. Honest Uncertainty：让模型知道何时不该装懂

### 7.1 为什么流畅不等于真实

自回归模型优化的是 token 概率，而不是事实真值。SFT 和偏好优化还可能奖励完整、礼貌、自信的表达。若训练数据中缺少“我不知道”“需要检索”“问题有歧义”的正例，模型就可能把不确定性隐藏在流畅句子里。

### 7.2 校准

把模型对一个命题的置信度记为 \(p_i\)，命题是否正确记为 \(y_i\)，Brier score 可以写成：

$$
Brier=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2
$$

分数越低通常表示概率预测更接近实际结果，但前提是概率有明确含义。语言模型输出的自信语气不自动等于可校准概率。

还可以按置信度分桶，比较平均置信度和实际正确率，观察 calibration gap。高风险系统应允许 abstain，不要强迫模型为每个问题给一个确定答案。

### 7.3 RAG 和引用

引用不是把几个链接附在答案末尾，而是判断每个关键 claim 是否被检索证据支持。可以定义 claim-level precision 和 coverage：

$$
P_{\mathrm{citation}}=
\frac{\text{有正确支持证据的引用 claim 数}}
{\text{带引用的 claim 总数}}
$$

$$
C_{\mathrm{evidence}}=
\frac{\text{有支持证据的关键 claim 数}}
{\text{关键 claim 总数}}
$$

一个答案可能引用很多资料但仍然没有支持最重要的结论。评估要把引用正确性、证据覆盖和无证据声明分开。

### 7.4 不确定时的系统动作

不确定性可以触发：

1. 请求用户澄清范围或时间。
2. 调用检索或计算工具。
3. 返回多个可能解释并标注差异。
4. 转人工或专家审核。
5. 对高风险动作停止执行。

重要的是把“不知道”变成可用的下一步，而不是只输出一句空泛免责声明。
## 8. 解释、控制、编辑和隐私的证据边界

### 8.1 解释性可以帮助诊断，不能单独证明安全

Mechanistic interpretability 研究 feature、circuit、信息流和因果干预。Activation patching、ablation、causal tracing 和 SAE 可以帮助检验某个内部假设，但每种方法都有范围：

1. attention heatmap 描述读取关系，不自动说明因果。
2. SAE feature 的语义解释需要跨样本和干预验证。
3. 单个 circuit 的解释不覆盖整个模型和所有输入。
4. 局部因果结果不等于部署安全证明。

一个合格的解释性结论应带模型版本、任务、层位、干预、负对照、holdout 和失败边界。

### 8.2 Steering 的价值和副作用

如果从对比样本估计行为方向 \(v\)，推理时在某一层加入强度 \(\alpha v\)，可以写成：

$$
h'_\ell=h_\ell+\alpha v
$$

目标行为改善并不意味着整体安全。评估至少要扫描 \(\alpha\)，并报告：

1. 目标行为收益。
2. 正常任务质量。
3. 过度拒答和事实性。
4. 工具权限和格式遵循。
5. 多语言、长上下文和对抗输入。

如果一个方向能增强拒答，也可能暴露拒答机制的脆弱性；如果一个方向能增强自信表达，也可能放大幻觉。Steering 是控制和研究工具，不是单独安全保证。

### 8.3 Model Editing 与 Unlearning

Model editing 通常有一个要写入的新事实或行为目标，unlearning 试图减少某些训练数据、知识或能力的影响。评估必须把以下指标分开：

1. 目标编辑或 forget set 的变化。
2. 改写、翻译、多轮和间接提示下的鲁棒性。
3. retain set 的能力保持。
4. 邻近事实、相关主体和其他语言的局部性。
5. 成员推断和近似复现风险。
6. 是否只是输出过滤或拒答模板。

“不再回答原问题”只能证明一次观察行为变化，不能单独证明参数中没有相关影响。

### 8.4 Privacy 的全生命周期

隐私评估要区分训练记忆、RAG 越权、工具返回、长期记忆、缓存和日志。PII scrub、差分隐私、unlearning、输出过滤和权限控制的责任边界不同：

| 方法 | 主要保护对象 | 不能替代 |
| --- | --- | --- |
| PII scrub | 数据源中的显式敏感字段 | 访问控制和模型后行为评估 |
| Differential Privacy | 单样本对训练输出的影响 | RAG 权限和日志治理 |
| Unlearning | 目标数据或行为的近似影响 | 重新审计下游副本 |
| Output filter | 当前可见输出 | 参数、工具和缓存中的影响 |
| Server authorization | 现实资源访问 | 模型诚实性和内容质量 |

### 8.5 Governance 是安全系统的一部分

Model card 说明模型能力、用途、限制和证据；system card 说明 RAG、工具、权限、日志、监控和事故响应。Policy 规定行为边界，governance 规定谁决定、如何审计、何时更新和如何恢复。

如果一个安全方案没有版本、责任人、监控和回滚，它可能在纸面上完整，在真实系统中却不可执行。

## 9. Safety Platform：把知识组织成系统

### 9.1 风险和数据平面

Safety Platform 的第一层不是模型，而是统一风险词汇和数据血缘：

1. 风险 taxonomy 和严重度定义。
2. 正常、边界、禁止和对抗样本。
3. PII、权限、租户和数据许可证标签。
4. 训练、评估、红队和线上反馈的来源。
5. 脱敏、版本、保留和删除策略。

如果红队样本没有来源和版本，修复后无法确认是否覆盖了同类问题；如果评估集进入训练，分数也会失去意义。

### 9.2 训练与模型平面

训练层可以包含安全 SFT、偏好优化、RLAIF、过程监督、拒答和不确定性数据。每个方法都要连接到目标和失败模式：

1. 安全 SFT 可以建立行为示例，但覆盖有限。
2. 偏好优化可以调整风格和边界，但可能放大 proxy。
3. AI feedback 可以扩规模，但需要人工 gold 校准。
4. 过程监督可以检查步骤，但不保证最终动作安全。
5. Model editing 可以快速修补局部行为，但可能影响邻近知识。

训练改进必须经过 holdout、对抗和正常任务回归，不能只看安全分数上涨。

### 9.3 评估编排层

评估编排器应支持：

1. 固定回归集和版本比较。
2. 随机抽样和分层抽样。
3. 人工、程序和模型评审的校准。
4. 多轮、工具、RAG、长上下文和多模态 harness。
5. 失败样本去重、归因和修复关联。
6. 置信区间、样本量和不确定状态。

评估输出应是可追溯报告，而不是一张只保留总分的排行榜。

### 9.4 策略、路由和工具层

系统运行时需要把模型和权限分开：

~~~text
request
  -> identity and tenant resolution
  -> risk classification
  -> policy and evidence check
  -> model / RAG / tool route
  -> output and action validation
  -> human review or fallback
  -> trace, metrics and incident signal
~~~

模型生成的文档 ID、SQL、shell 参数、邮件地址和支付指令都应经过服务端校验。策略层也要能解释为什么允许、澄清、只读、转人工或阻断。

### 9.5 发布与线上层

发布记录至少包括：

1. 当前版本和允许的用户/任务/数据范围。
2. 已支持、部分支持、过期和缺失的证据。
3. 未解决风险、有效缓解和限制。
4. 监控指标、告警和升级动作。
5. 回滚对象、恢复时间和事故联系人。

线上监控要观察行为和后果，包括有害遵循、误拒、引用支持、越权工具、隐私日志、延迟、成本和用户申诉。单一安全分数无法发现所有路径。

## 10. 综合案例：企业研究助手

### 10.1 需求

一家企业希望部署研究助手，功能包括：

1. 搜索本租户的政策、合同和技术文档。
2. 总结证据并给出引用。
3. 生成分析报告草稿。
4. 创建待审批的工单，但不直接修改生产系统。

这不是一个“问答机器人”问题，而是一个有私有数据、引用、长期任务和潜在工具动作的系统问题。

### 10.2 风险拆解

输入和数据风险包括越权文档、恶意文档指令、个人信息、版权限制和评估污染。模型风险包括幻觉、长文遗漏、过度自信、拒答不一致和多语言差异。系统风险包括跨租户检索、引用错配、工单越权、重复提交、日志留存和缓存隔离。组织风险包括没有资源所有者确认、没有事故联系人和文档过期。

### 10.3 任务契约

一个任务契约可以规定：

1. 每个关键结论必须关联至少一个有权访问的文档片段。
2. 不能把文档中的指令当作系统或用户指令。
3. 报告只能创建草稿工单，不能直接提交生产变更。
4. 对不确定结论必须标注证据不足并请求复核。
5. 日志只保留文档 ID、权限结果、模型版本和必要的脱敏摘要。

### 10.4 评估设计

评估集按租户、语言、文档类型、问题难度、权限状态和多轮追问切片。除了正常问题，还加入：

1. 用户有权访问的文档。
2. 用户无权访问但语义相似的文档。
3. 已撤回文档和缓存中的旧版本。
4. 包含恶意指令的外部文档。
5. 证据不足、文档冲突和空检索。
6. 创建重复工单、超出权限范围和工具参数缺失。

指标至少包括引用 precision、关键 claim evidence coverage、无权检索率、回答泄露率、工具越权率、误拒率、日志原文率和单位成功成本。

### 10.5 分层访问

默认用户只能检索公共和本租户只读资料；研究负责人可以访问受限资料，但仍不能绕过字段权限；工单 Agent 只能创建草稿并需要审批；生产写操作由独立服务端和人工确认执行。

如果长周期研究任务的状态出现冲突，系统暂停并保存 checkpoint，而不是继续猜测。若引用服务暂时不可用，fallback 到没有引用的回答是不安全的；更合理的是说明证据服务不可用或只允许用户重新尝试。

### 10.6 事故复盘

假设一次回答泄露了另一个租户的合同片段。调查步骤应依次检查：

1. 用户身份和租户解析。
2. 检索前过滤、返回前过滤和文档撤回。
3. 向量缓存、prefix cache、模型上下文和引用生成。
4. 输出过滤、工具 trace、异常日志和人工标注副本。
5. 模型是否真的复现参数记忆，还是应用路径越权。

不同根因对应不同修复。把所有问题归因于“模型幻觉”会错过最关键的权限和日志控制。
## 11. 如何把复杂安全问题讲清楚

### 11.1 六步表达法

面对一个新问题，可以按六步组织推理：

1. **定义**：对象是什么，边界在哪里。
2. **机制**：风险为什么会发生，经过哪条数据或行为路径。
3. **分层**：训练、模型、应用和组织分别承担什么责任。
4. **测量**：样本、分母、指标、切片和不确定性是什么。
5. **控制**：哪种缓解针对哪条路径，何时升级人工。
6. **边界**：结果支持什么，不能支持什么，下一步如何复测。

例如，“如何防 prompt injection”不能只回答“加一层 prompt”。完整判断应说明不可信内容与指令边界、权限过滤、工具授权、动作确认、回滚、攻击评估和正常任务回归。

### 11.2 一个错误的简短回答

> 我们使用 RLHF 和内容过滤，所以模型已经安全。

这句话有三个问题：RLHF 不是安全证明，内容过滤只覆盖输出路径，“已经安全”没有版本、范围和证据。

### 11.3 一个更可靠的判断

> 我会先把安全拆成有害遵循、隐私、越狱、注入、工具越权和误拒等风险。训练可以用安全 SFT 和偏好优化改善行为，但服务端权限、RAG 过滤、工具 schema、人工确认和日志脱敏必须在系统层执行。评估要报告攻击与干净任务、分层切片、工具轨迹、引用和严重度。当前结论只适用于已测模型版本、用户、数据和工具范围；未覆盖的高风险动作先保持只读或人工审核，并把新失败沉淀到回归集。

这个判断没有承诺“没有风险”，但让读者知道风险怎么分、证据在哪里以及下一步做什么。

### 11.4 不确定时如何表达

遇到资料不完整的前沿模型，应该区分：

1. 已由官方模型卡或技术报告说明的事实。
2. 一方产品页公开的接口或产品信号。
3. 独立评估或公开权重可以复现的部分。
4. 仅有社区传闻或未找到官方身份的观察项。

例如，模型名称和产品页存在不能自动推出内部架构、训练配方、通用 benchmark 或生产安全性。把资料边界说清楚，比用确定语气填补空白更专业。

## 12. 一个最小的 Safety 复盘审计 demo

下面的 demo 只使用合成的回答记录。它不运行攻击，也不评估真实模型；它展示如何把一个复杂安全判断拆成 thresholds、signals、evidence_status、actions 和 decision。

~~~python
def ratio(numerator, denominator):
    return round(numerator / denominator, 3) if denominator else None


answers = [
    {
        "topic": "safety_and_alignment",
        "weight": 2,
        "required": {"definition", "risk", "metrics", "tradeoff", "governance"},
        "covered": {"definition", "risk", "tradeoff", "governance"},
        "unsafe_detail": False,
    },
    {
        "topic": "rlhf_and_reward_hacking",
        "weight": 3,
        "required": {"proxy", "data_bias", "overoptimization", "mitigation", "eval"},
        "covered": {"proxy", "data_bias", "overoptimization", "mitigation", "eval"},
        "unsafe_detail": False,
    },
    {
        "topic": "prompt_injection",
        "weight": 3,
        "required": {"trust_boundary", "permissions", "tool_control", "metrics", "normal_task"},
        "covered": {"trust_boundary", "permissions", "tool_control", "metrics"},
        "unsafe_detail": False,
    },
    {
        "topic": "dangerous_capability",
        "weight": 4,
        "required": {"definition", "elicitation", "baseline", "budget", "access"},
        "covered": {"definition", "elicitation", "baseline", "budget", "access"},
        "unsafe_detail": False,
    },
    {
        "topic": "safety_platform",
        "weight": 5,
        "required": {"data", "training", "evaluation", "runtime", "monitoring", "incident"},
        "covered": {"data", "training", "evaluation", "runtime", "monitoring"},
        "unsafe_detail": False,
    },
]

required_risks = {
    "harmful_compliance",
    "privacy",
    "jailbreak",
    "prompt_injection",
    "tool_misuse",
    "dangerous_capability",
    "over_refusal",
    "governance",
}
covered_risks = {
    "harmful_compliance",
    "privacy",
    "jailbreak",
    "prompt_injection",
    "tool_misuse",
    "dangerous_capability",
    "governance",
}

required_metrics = {
    "unsafe_compliance",
    "over_refusal",
    "attack_success",
    "unauthorized_tool",
    "severity_weighted_risk",
    "privacy_leak",
    "evidence_coverage",
}
covered_metrics = {
    "unsafe_compliance",
    "over_refusal",
    "attack_success",
    "unauthorized_tool",
    "privacy_leak",
}

thresholds = {
    "answer_coverage": 0.85,
    "risk_coverage": 0.85,
    "metric_coverage": 0.85,
    "safe_boundary": 1.0,
}

weighted_answer = 0.0
total_weight = 0
topic_scores = {}
missing_topics = {}
for answer in answers:
    required = answer["required"]
    covered = answer["covered"]
    score = ratio(len(required & covered), len(required))
    topic_scores[answer["topic"]] = score
    missing_topics[answer["topic"]] = sorted(required - covered)
    weighted_answer += score * answer["weight"]
    total_weight += answer["weight"]

signals = {
    "answer_coverage": round(weighted_answer / total_weight, 3),
    "risk_coverage": ratio(
        len(required_risks & covered_risks),
        len(required_risks),
    ),
    "metric_coverage": ratio(
        len(required_metrics & covered_metrics),
        len(required_metrics),
    ),
    "unsafe_detail_count": sum(
        answer["unsafe_detail"] for answer in answers
    ),
}

evidence_status = {
    "answer": "supported"
    if signals["answer_coverage"] >= thresholds["answer_coverage"]
    else "partial",
    "risk": "supported"
    if signals["risk_coverage"] >= thresholds["risk_coverage"]
    else "partial",
    "metrics": "supported"
    if signals["metric_coverage"] >= thresholds["metric_coverage"]
    else "partial",
    "safe_boundary": "supported"
    if signals["unsafe_detail_count"] == 0
    else "failed",
}

actions = {
    "answer": "add_missing_mechanism_or_governance_links",
    "risk": "add_over_refusal_and_governance_to_risk_taxonomy",
    "metrics": "add_severity_weighted_risk_and_evidence_coverage",
    "safe_boundary": "remove_operationally_dangerous_details",
}

failed = [
    name for name, status in evidence_status.items()
    if status != "supported"
]
if failed:
    decision = "revise_answer_and_retest"
else:
    decision = "retain_scope_and_continue_practice"

report = {
    "thresholds": thresholds,
    "signals": signals,
    "evidence_status": evidence_status,
    "actions": actions,
    "missing_topics": missing_topics,
    "decision": decision,
}

print("topic_scores=", topic_scores)
print("signals=", signals)
print("evidence_status=", evidence_status)
print("missing_topics=", missing_topics)
print("decision=", decision)
~~~

这组数据会显示 prompt injection 缺少正常任务指标，Safety Platform 缺少事故响应，整体指标覆盖也不足，因此 decision 为 revise_answer_and_retest。代码没有计算一个叫“准备好了”的总开关；它保留每个缺口和动作，方便读者进行下一轮复习。

## 13. 综合练习

### 练习一：解释 Safety 与 Alignment

用一个客服 Agent 例子说明二者的重叠和区别。必须区分真实目标、代理奖励、正常任务质量和现实副作用。

### 练习二：诊断 Reward Hacking

假设奖励分数提高 15%，人工事实性下降 8%，拒答率提高 20%。列出至少三个可能原因，并设计能区分这些原因的实验。

### 练习三：设计注入评估

为企业 RAG 构造干净文档、恶意文档、权限冲突、空检索、多轮追问和工具调用样本。为每类样本写出分母、指标和人工升级条件。

### 练习四：评估危险能力

选择一个抽象的专业任务，设计非 AI baseline、自然部署条件、增强 harness、预算、成功标准和安全停止条件。不要写危险操作步骤，只讨论评估协议。

### 练习五：把失败接入治理

假设发现模型在中文长文中引用覆盖率下降。说明如何更新回归集、model card、system card、路由范围、用户文案和监控告警。

### 练习六：设计安全降级

为一个拥有只读和写入工具的 Agent 设计至少三种 fallback。说明每种 fallback 的触发信号、保留状态、用户反馈和恢复条件。

## 14. 资料与证据边界

### 14.1 对齐、监督和偏好优化

1. [InstructGPT](https://arxiv.org/abs/2203.02155)：支持人类反馈、奖励模型和 PPO 后训练的公开研究背景。
2. [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)：支持偏好奖励与奖励错配讨论。
3. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)：支持 DPO 目标和偏好数据边界。
4. [Constitutional AI](https://arxiv.org/abs/2212.08073)：支持原则驱动的 AI feedback 和监督扩展讨论。

这些论文说明算法和实验条件，不证明所有商业模型采用了相同训练流程，也不证明某种后训练方法自动带来安全。

### 14.2 攻击、防御和评估

1. [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)：支持 LLM 应用风险分类和防御边界。
2. [Prompt Injection Attacks and Defenses in LLM-Integrated Applications](https://arxiv.org/abs/2310.12815)：支持间接注入和应用信任边界研究入口。
3. [OpenAI Evals](https://github.com/openai/evals)：支持评估框架和可复用评估任务的公开入口。
4. [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)：支持把风险映射、测量、治理和管理放入生命周期。

红队和公开攻击论文的样本、目标和预算不应被改写成现实系统的总体风险概率；本章只讨论防御、评估和治理方法。

### 14.3 解释、编辑、隐私和治理

1. [Transformer Circuits Thread](https://transformer-circuits.pub/)：支持机制解释和电路研究的公开背景。
2. [ROME](https://arxiv.org/abs/2202.05262)：支持知识编辑局部性和因果定位讨论。
3. [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805)：支持训练数据复现和隐私风险边界。
4. [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)：支持模型卡、限制和评估披露。

这些资料各自只覆盖一个证据对象。解释性不能替代行为评估，编辑不能自动等于遗忘，模型卡不能替代系统权限。

### 14.4 本章可以支持的结论

本章可以严谨地说：

1. Safety 是目标、行为、系统后果和组织责任共同构成的问题。
2. RLHF、DPO、监督、解释、steering、editing 和 unlearning 都有明确价值，但不能单独证明安全。
3. Jailbreak、prompt injection、privacy、dangerous capability、tool misuse 和 over-refusal 需要不同样本和指标。
4. Safety eval 与 red teaming 互补，固定回归和探索性测试都要保留范围和不确定性。
5. 发布决定应绑定版本、用户、数据、工具、证据和可逆性，发布后仍需要监控和事故响应。

本章不能严谨地说：

1. 提高拒答率就等于提高安全。
2. 一次红队没有发现问题就证明没有未知攻击。
3. 某个 benchmark 或模型级分数可以覆盖真实 Agent 系统。
4. 一种训练或编辑方法可以消除所有目标错配、隐私和工具风险。
5. 安全文档本身可以代替服务端权限、人工责任和回滚。

## 15. 小结：安全判断要能回到证据

Safety 的系统化表达从定义开始，但不能停在定义。必须继续问：风险通过哪条路径出现，如何构造样本，分母是什么，哪个组件负责缓解，怎样观察副作用，结果能支持多大范围的结论。

目标错配提醒我们，奖励分数不是现实目标；可扩展监督提醒我们，监督成本和专家注意力会成为瓶颈；越狱和注入提醒我们，模型行为边界与应用信任边界不同；危险能力评估提醒我们，工具、harness 和专家辅助会改变能力上限；诚实性、解释性、steering、editing、privacy 和 governance 则分别提供行为校准、机制证据、控制实验、知识修改、数据保护和责任追踪的工具。

一个成熟的安全系统不会把这些工具拼成一句“已经安全”。它会保存 thresholds、signals、evidence_status、actions 和 decision，清楚写出哪些范围已经测量，哪些风险仍然开放，哪些控制尚未验证，以及下一轮要怎样修复和复测。只有这样，复杂的安全知识才能从术语变成可执行的工程判断。
