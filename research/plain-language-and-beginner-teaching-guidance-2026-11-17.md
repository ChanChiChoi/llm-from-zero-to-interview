# 面向初学者的清楚讲解：写作指南与教学研究

检索与阅读日期：2026-11-17。通过用户指定的本地 HTTP 代理 `http://127.0.0.1:7890` 检索并阅读政府写作指南、技术写作课程、无障碍指南、大学教学资料、国家科学院报告、同行评审研究，以及 O’Reilly 技术书籍的出版社目录和公开样章。资料覆盖 plain language、technical writing、初学者教学、worked examples、类比、分段、先备知识和读者理解检查。

这份记录整理可借鉴的做法和证据边界，不声称穷尽所有写作研究。官方指南说明机构建议怎样写，不能自动证明每条建议都有独立因果实验；教学研究的效果也受读者经验、学科、任务和材料设计影响。下面的项目规范是基于这些来源形成的编辑综合。

## 一、来源与能支持的结论

### 清楚沟通、读者需要与无障碍

| 来源 | 类型和阅读范围 | 可以支持 | 适用边界 |
| --- | --- | --- | --- |
| [Digital.gov Plain Language Guide Series](https://digital.gov/guides/plain-language) 及 [Principles](https://digital.gov/guides/plain-language/principles)、[Writing](https://digital.gov/guides/plain-language/writing)、[Design](https://digital.gov/guides/plain-language/design)、[Test](https://digital.gov/guides/plain-language/test) | 美国政府数字服务指南 | 先确定读者、目的和任务；用清楚的组织、直接表达、标题和视觉层级帮助读者找到信息；尽早测试、依据反馈修改并再次检查。清楚表达不是把内容降级。 | 面向政府服务信息的实践指南；不等于每种技术教材都要采用同样的开篇方式或版式。 |
| [CDC Clear Communication Index User Guide](https://www.cdc.gov/ccindex/tool/)、[PDF guide](https://www.cdc.gov/ccindex/pdf/clear-communication-user-guide.pdf)、[How to Use the Index](https://www.cdc.gov/ccindex/tool/how-to-use.html)、[FAQ](https://www.cdc.gov/ccindex/tool/faq.html) | CDC 研究支持的公共沟通评估框架，含 20 项、7 个评估领域 | 起草前先写明主要受众、受众能力、沟通目标和主信息；检查语言、信息设计、数字与风险表达；让独立读者审阅，再按反馈修改。CDC 也提醒，仅靠词长、音节数和句长等可读性公式无法判断沟通是否有效。 | 面向公共卫生材料，但“先确定受众和目标、让独立读者指出断点”可借鉴到教材；Index 分数本身不是本书系的教学质量分数。 |
| [CDC Plain Language Materials & Resources](https://www.cdc.gov/health-literacy/php/develop-materials/plain-language.html) | CDC 的 plain language 清单与示例 | 先安排受众需要的主信息，再分成合乎逻辑的小块；标题要有信息；一句话聚焦一个主要意思；必要术语应在出现处解释。 | 清单里的句数、字数建议是公共材料的检查线索，不应改成技术正文的机械字数上限。 |
| [Office for National Statistics: User Needs](https://service-manual.ons.gov.uk/content/writing-for-users/user-needs) 与 [Plain Language](https://service-manual.ons.gov.uk/content/writing-for-users/plain-language) | 英国国家统计局内容设计规范 | 需求应说明“谁要做什么、为什么”；尽量从证据而非作者假设出发；标题应让读者找到需要的信息；技术词和缩写要解释、术语要前后一致。 | 原文主要服务网站访问者的任务；“读者要学会什么”是把用户任务框架迁移到教材后的编辑用法。 |
| [GOV.UK: Identify user needs](https://guidance.publishing.service.gov.uk/writing-to-gov-uk-standards/plan-manage-content/identify-user-needs/) 与 [Create a clear structure](https://guidance.publishing.service.gov.uk/writing-to-gov-uk-standards/writing-guidelines/clear-structure/) | 英国政府内容设计指导 | 识别读者是谁、想完成什么、现在如何完成、遇到什么困难；结构要服务于这些任务，而不是作者已有的资料分组。 | 政务网页有较强的任务导向；教材还要呈现机制、推导和知识联系，不能只保留“读者马上要做的事”。 |
| [UK Department for Education: Plain Language](https://design.education.gov.uk/content-design/plain-language) | 政府内容设计实践指南 | plain language 要按用户需要调整；有充分理由时可以使用专业术语，但第一次出现要附清楚解释。 | 具体示例服务于政府用户，不代表可以不加选择地把口语化作为所有技术正文的风格。 |
| [W3C Cognitive Accessibility: Clear and Understandable Content](https://www.w3.org/WAI/WCAG2/supplemental/objectives/o3-clear-content/) | W3C 认知无障碍补充指导，页面明确标注为 WCAG 额外建议、非合规要求 | 清楚词语、短文本块、无歧义表达、清楚图像、长文摘要、分开说明各项指令，并解释隐含信息，可以减少认知和语言障碍。 | 这是面向认知无障碍的补充建议，不是 WCAG 的强制成功标准；“避免依赖数学能力”不能被误读成技术书籍应删除数学。 |
| [W3C: Headings and Labels](https://www.w3.org/WAI/WCAG21/Understanding/headings-and-labels) | WCAG 2.1 成功标准说明 | 描述准确的标题能帮助读者定位内容、预测小节内容，并理解内容之间的关系；对短时记忆有限或阅读较慢的读者尤其有帮助。 | 标题能改善导航，但不能代替段落内的因果讲解。 |

### 技术写作与信息呈现

| 来源 | 类型和阅读范围 | 可以支持 | 适用边界 |
| --- | --- | --- | --- |
| [Google Technical Writing One: Audience](https://developers.google.com/tech-writing/one/audience)、[Words](https://developers.google.com/tech-writing/one/words)、[Paragraphs](https://developers.google.com/tech-writing/one/paragraphs)、[Short Sentences](https://developers.google.com/tech-writing/one/short-sentences) | Google 官方技术写作课程 | 识别读者角色和已有知识，防止“知识诅咒”；解释读者可能不熟悉的术语；保持术语一致；段落围绕一个主题，例子帮助解释 what、why、how；复杂句可按主要想法拆分。 | 课程面向技术文档作者。教材要讲清的不仅是任务步骤，也包括知识机制、推理和边界。 |
| [Google Technical Writing Two: course summary](https://developers.google.com/tech-writing/course-summaries/two)、[Think like your audience](https://developers.google.com/tech-writing/two/editing)、[Creating sample code](https://developers.google.com/tech-writing/two/sample-code) | Google 官方中阶技术写作课程 | 先列大纲或整理草稿；在适用时渐进披露；考虑不同复杂度的示例；教程中用例子巩固概念；必要时给反例；复杂代码机制应先用正文解释，再展示代码；对新手从简单例子走向中等、较复杂例子。 | 主要讨论技术文档和代码教程。书系可以借鉴示例递进和解释代码的顺序，不应把“提供反例”等变成每节硬性要求。 |
| [Microsoft Learn contributor style quick start](https://learn.microsoft.com/en-us/contribute/content/style-quick-start) | Microsoft Learn 技术内容指南 | 用读者熟悉的自然语言，提供解释新概念的例子；“少些正式腔”不等于“少些技术”；长流程应拆成可处理的小步骤。 | 面向产品和开发文档；实践建议不能代替书系自己的术语、推导和证据规范。 |
| [Microsoft Writing Style Guide: Use technical terms carefully](https://learn.microsoft.com/en-us/style-guide/word-choice/use-technical-terms-carefully)、[Writing step-by-step instructions](https://learn.microsoft.com/en-us/style-guide/procedures-instructions/writing-step-by-step-instructions) | Microsoft 技术写作规范 | 精确表达需要技术术语时就保留它，在上下文中定义并保持一致；多步操作使用编号步骤，避免一条步骤混入多个操作。 | 操作步骤规范主要针对用户界面和任务指令；数学推导或机制解释要按因果需要组织，不必强行写成操作手册。 |
| [Technical Writing Essentials](https://pressbooks.bccampus.ca/technicalwriting/) | 大学开放教材，涵盖受众和任务分析、读者中心写作、简洁表达、信息设计与研究 | 技术沟通是一种面向读者和问题的设计活动；受众与任务会影响信息深度、术语和组织。 | 开放教材的综合性教学资源，不是同一作者群体的实验研究。 |

### 教学设计与学习研究

| 来源 | 类型和阅读范围 | 可以支持 | 适用边界 |
| --- | --- | --- | --- |
| [CAST UDL Guidelines: Connect prior knowledge to new learning](https://udlguidelines.cast.org/representation/building-knowledge/prior-knowledge/) | 通用学习设计指导 | 通过激活先备知识、概念图、示范或模型、相关类比和显式连接，帮助读者把新内容接到已有知识；也强调关系识别、迁移和不同支架。 | UDL 是设计框架，不是逐条经独立实验确认的固定教材模板；读者差异需要针对内容和场景处理。 |
| [Carnegie Mellon Eberly Center: Assessing Prior Knowledge](https://www.cmu.edu/teaching/designteach/teach/priorknowledge.html) 与 [Lectures](https://www.cmu.edu/teaching/designteach/teach/instructionalstrategies/lectures.html) | 大学教学中心指导 | 先备知识、错误概念和技能差异会影响新信息如何被注意、解释和组织；可用短问答或其他方式摸底。例子和类比应充分展开并指出限制；摘要和提示可帮助新手看出组织关系。 | 主要服务课堂教学；教材作者常没有实时课堂反馈，应把相应做法转为试读和自检，而不是假装已测过读者。 |
| [Cornell Center for Teaching Innovation: Assessing Prior Knowledge & Addressing Learning Gaps](https://teaching.cornell.edu/teaching-resources/assessment-evaluation/assessing-prior-knowledge-addressing-learning-gaps) | 大学教学指导 | 通过短问答、判断题或图像材料发现先备知识和误解，再补上必要前置或提供补充材料；同伴解释也可用于检查表达。 | 提供教学策略和建议，不是每种检查方法都有同等证据或适合自学阅读。 |
| [National Research Council, *How People Learn*, Chapter 3](https://www.nationalacademies.org/read/9853/chapter/6) | 美国国家科学院关于学习与迁移的综合报告 | 学习者会用已有知识构建新理解；相关知识未必会自动被激活，教师或材料需要显式搭桥。先备知识也可能包含错误信念，干扰新概念。 | 综合报告讨论学校和学科教学；它支持重视先备知识与误解，不直接规定某个技术章节应写多少背景。 |
| [Gray & Holyoak, “Teaching by Analogy: From Theory to Practice” (2021)](https://reasoninglab.psych.ucla.edu/wp-content/uploads/sites/273/2021/12/Gray_Holyoak.2021.pdf) | 同行评审的类比教学综述，重点涉及 STEM | 类比从熟悉的源领域映射关系到陌生的目标领域；应明确指出共享关系、考虑读者已有知识，并说明映射不成立的地方，防止错误迁移。 | 综述讨论教育中的类比教学，不要求每个概念都配类比；类比不能替代定义、证明、计算或真实机制。 |
| van Gog, Kester & Paas, [“Effects of worked examples, example-problem, and problem-example pairs on novices’ learning” (2011)](https://doi.org/10.1016/j.cedpsych.2010.10.004) | 同行评审实验；荷兰中学生学习电路故障排查，比较完整示例、示例后练习、先练习后看示例和只练习 | 在该任务和样本中，完整 worked examples 与“先示例再练习”比单独解题有较低学习负荷和更好的学习结果；说明新手在复杂任务中可能需要先看完整解法。 | 结论限于实验任务、样本和材料。不是所有问题都必须先给答案，也不能据此否定引导式探索、讨论或熟练者直接练习。 |
| Spanjers, Van Gog & Van Merriënboer, [“Segmentation of Worked Examples” (2012)](https://cris.maastrichtuniversity.nl/files/72615148/spanjers_2012_segmentation_of_worked_examples.pdf) | 同行评审实验；52 名荷兰高中生学习概率计算示例 | 在这组概率示例中，预先分段的示例用更少学习努力达到相同学习结果；让学生自己执行分段没有带来同等效率。提示有意义的分块可替读者省去寻找边界的工作。 | 样本和任务都较窄；不能推出段落越短越好，也不能把自然语言章节等同于该实验材料。 |
| Kalyuga, Ayres, Chandler & Sweller, [“The Expertise Reversal Effect” (2003)](https://lexiconic.net/pedagogy/2003-Kalyuga_et_al.pdf) | 同行评审综述，回顾不同经验水平与教学设计效果的交互 | 对新手有效的指导，随着读者经验上升可能失去效果，甚至妨碍学习；示例、提示和背景量应考虑读者知识。 | 综述主要基于认知负荷理论与既有实验；不能仅凭“专家”标签判断每个具体读者已掌握什么。 |
| Chi et al., [“Self-Explanations: How Students Study and Use Examples in Learning to Solve Problems” (1989)](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1302_1) | 同行评审研究；分析学生学习力学 worked examples 时的口头自我解释 | 能把解题动作和文本原理联系起来、主动解释步骤的学生，表现出更好的例子独立理解；仅看完示例不保证读者抽象出规则。 | 研究分析特定任务和学生的自我解释，不证明所有读者都能靠自问自行补足糟糕正文；作者仍要把关键解释写出来。 |
| Roediger & Karpicke, [“Test-Enhanced Learning” (2006)](https://doi.org/10.1111/j.1467-9280.2006.01693.x) | 同行评审实验；大学生阅读说明性文章后进行回忆测试或重复学习，比较短期与延迟记忆 | 在该实验中，延迟几天或一周的记忆测验显示，主动回忆可改善材料的延迟保持；熟悉感或重复阅读后的信心不能单独证明记住了。 | 研究测的是对文章内容的记忆保持，不等于已理解机制、能迁移到新问题或能复走计算；项目里的复述/变式检查属于有边界的编辑自检。 |
| [AHRQ Health Literacy Universal Precautions Toolkit](https://www.ahrq.gov/health-literacy/improve/precautions/index.html) 与 [第三版工具包 PDF](https://www.ahrq.gov/sites/default/files/wysiwyg/health-literacy/3rd-edition-toolkit/health-literacy-toolkit-third-edition.pdf) | 美国医疗研究与质量局患者沟通工具包，含 teach-back 与书面材料检查方法 | 沟通后请对方用自己的话说明或演示关键信息，可暴露表达中的误解；确认理解后应调整解释方式。 | teach-back 原本用于医患沟通和健康素养场景。本项目借它提出读者试读与复述，是编辑类比和自检启发，不是本书系效果已被该工具包直接验证。 |

### O’Reilly 技术书的具体样章

以下是截至 2026-11-17 对 O’Reilly 出版社网页、目录和公开样章的抽样观察；只阅读公开可见内容，没有把样章观察外推成整本书的评价。样本包括常被读者以封面动物称呼的书，也包括 O’Reilly 采用其他教学形式的作品。

| 书目与阅读范围 | 样章呈现的讲法 | 可借鉴之处与边界 |
| --- | --- | --- |
| [*Learning Perl*, 8th Edition：序言与第 1 章](https://www.oreilly.com/library/view/learning-perl-8th/9781492094944/ch01.html)，另见[序言](https://www.oreilly.com/library/view/learning-perl-8th/9781492094944/preface01.html) | “Llama book”先用问答回应初学者会问的问题，再交代这本书是教程而非语言参考手册、能带读者做到什么、哪些内容会留到后面。序言说明学习时间和章节节奏来自作者的线下课程；目录显示各章配有练习和答案。 | 可以先消除“这本书适不适合我、读完能做什么、现在为什么学这个”的疑问，再开始讲术语。明确范围不等于删掉重要内容；技术版本和事实仍须另行核验。 |
| [*Learning Python*, 6th Edition：序言与第 1 章](https://www.oreilly.com/library/view/learning-python-6th/9781098171292/ch01.html)，另见[序言](https://www.oreilly.com/library/view/learning-python-6th/9781098171292/preface01.html) | 序言把作品定位为深入讲解语言基础的教程，并说明内容依据长期学习者反馈；第 1 章采用问答形式，从新手常问的“为什么学 Python”开始。目录还显示，章节反复使用小结、自测题、答案和分部练习。 | 用读者真实问题安排讲解顺序；在篇末留出能独立作答、核对的入口。出版社页面对读者水平的分类与序言定位并不完全相同，说明分类标签不能替代对具体章节和读者前置的判断。 |
| [*Programming Perl*, 3rd Edition：第 1 章](https://www.oreilly.com/library/view/programming-perl-3rd/0596000278/ch01.html) | 开篇给出一行可运行的 `print` 程序，随即告诉读者运行后屏幕会出现什么，并解释 `\n` 对输出的作用。 | 例子尽量短，但要接着讲输入、运行结果和关键符号的含义；代码本身不能代替解释。该书是 2000 年的中高级 Perl 书，这里只参考开篇的示例组织，不用它核对今天的语言行为。 |
| [*Head First Python*, 3rd Edition：第 1 章](https://www.oreilly.com/library/view/head-first-python/9781492051282/ch01.html) | 以一副扑克牌为持续对象，用 Jupyter 运行代码，逐步引出多种内置数据类型；章节中还用对话段落回应读者可能提出的问题。 | 熟悉对象可以帮助读者在一节内跟踪多个概念；对话或插图可以增加提示，但只有在确实解释机制时才保留，不能让热闹的形式替代对象变化和原因。 |
| [*Introducing Python*, 3rd Edition：第 1 章](https://www.oreilly.com/library/view/introducing-python-3rd/9781098174392/ch01.html) | 开篇用编织说明和食谱两个“小谜题”呈现同一段指令为何会因读者经验不同而难懂，再进入 Python。 | 例子可以先让读者感受到“熟悉前置会改变理解难度”，帮助作者检视是否把专业背景误当成常识。这个开场不是技术机制的定义或证明；类比仍要落回真实对象、规则和限制。 |

从这些具体样章中可抽取几种适合本项目的编辑做法：先回答读者此刻的问题并讲清阅读目标；用一个小而完整的程序、数据或场景把规则演出来；每一步紧跟可观察的结果和原因；在复杂度增加前保留同一组例子；用小结、自测或练习让读者检查自己是否能复走过程。问答、对话、幽默、图像和生活类比是不同作者的可选表达手段，不是统一配方。

这些作品的版次、受众、篇幅、术语密度和叙述风格并不相同；O’Reilly 的书目包含多种教学形式，并没有一套适用于所有书的统一讲法。封面动物是识别这类书的外观传统，具体讲解方法仍要从章节内容中判断。借鉴时应指向具体书目、版次、章节和可复用动作，不照搬整本书的口吻、插图密度或技术结论。旧版样章尤其只能作为讲解形式参考，时效事实须用当前官方文档或一手资料核验。

## 二、面向初学者讲解的可执行做法

下列做法是项目综合，不是任一来源逐字给出的固定配方。遇到抽象概念、推导、算法或系统机制时，按内容选用；不要求所有小节都同时出现例子、类比、图表和练习。

1. **先定义读者，不要想当然地猜读者。** 写清读者已具备的知识、当前主题需要的前置知识、常见误解，以及读完要能完成什么。目标可写成可观察的动作：解释一次状态变化、沿流程追踪一个请求、复算一个损失、比较两个方案的边界。把“我作为专家觉得显然”的内容单独检查一遍。
2. **从读者遇到的问题开始，再命名术语。** 先给现象、目标或疑问，让读者知道为什么现在要讲这个；说明输入、期望结果和目前的困难。需要的术语在它首次有用时定义，并立即解释它在本例里指谁、做什么。跨章依赖只唤回理解当前内容必需的一小段，不能假定读者会自行回忆出隐藏前提。
3. **选一个足够小、但保留关键关系的完整例子。** 明确起始状态、对象、输入、规则和目标结果。例子应小到能手工跟踪，又不能小到删掉了真实机制的关键条件。注明人为假设、玩具数值和简化；让同一批对象贯穿解释，避免换例子增加记忆负担。
4. **把过程写出来，不只报结论或步骤名。** 每一步交代“对哪个对象做了什么”“它改变成什么”“为什么规则要求这样做”。保留会影响理解的中间结果；有分支时说明触发条件；数字、单位、索引和形状要让读者能核对。不要用“然后得到”“显然可知”“剩余部分类似”跳过首次出现的关键步骤。
5. **把例子中的具体对象接到抽象规则。** 例子走完后逐项映射术语、定义、公式符号、代码变量或图中元素；说明其中哪些关系能推广，哪些只是演示设定。若讲公式，先说它回答什么，再解释符号、逐步代入例子、保留中间计算、解释结果的含义和优化方向，最后给出一般表达。口头直觉不能替代必要推导。
6. **保持技术准确，减少不必要的术语负担。** 日常词更清楚且不损失精度时用日常词；必须使用技术词时就保留，并在需要处定义。一个概念尽量稳定使用一个名字；相近术语要说明区分维度；缩写不要连发。读者友好不是删掉参数、条件、反例或局限，而是让这些内容有可跟随的次序。
7. **把“是什么、为什么、怎样运行、何时不适用”连起来。** 说明方法解决哪个具体困难、选择了什么表示或假设、关键动作如何产生结果、结果改善了什么、代价和失效边界是什么。不要用孤立的术语解释术语，也不要把“它很重要”当作动机。
8. **类比只在能搭桥时用，并写明映射和断点。** 选读者确实熟悉的场景，逐项指出故事里的角色、动作、约束分别对应技术机制中的什么；随后指出类比不能解释的地方，并回到真实对象、公式或实现。不要让类比承担证明、计算或完整定义。
9. **让段落和视觉形式显出关系。** 一段聚焦一个主要解释任务；标题说清这一节解决什么问题；长流程分成编号步骤；并列方案用同一维度的表格；流程图说明每个框和箭头代表的状态或动作。图表应突出结论和关系，不能把读者需要的前提藏在标签、图例或脚注里。短句、空白和清晰标题是帮助定位的工具，不能机械拆碎因果链。
10. **从完整示范逐渐走向读者自己能做。** 对初学者且机制步骤多的内容，可先完整示范，再给一个只留关键空缺的例子，最后让读者处理一个数值或条件略有变化的问题。只在能澄清边界时加反例或对照例子。读者已有经验时，减少重复铺垫，允许跳到推导、实现或进阶内容；不要让单一讲法拖慢所有人。
11. **检查能否复述、复走和迁移。** 请目标读者用自己的话说明问题和规则，沿例子指出输入、操作、中间结果和理由，再预测一个小变式中哪些结果会变。若读者只能复述术语或最终答案，就查找缺的是先备知识、对象对应、推理步骤、因果解释还是边界条件。试读时先观察对方在哪里停顿或猜测，不要先提示答案再问“懂了吗”。
12. **按证据改稿，别把字数指标当理解指标。** 朗读适合检查语句节奏；可读性分数适合提示长句或词汇问题；二者都不能证明技术机制讲清楚。独立读者试读、复述或复走例子更能暴露具体断点。改完后再看同一断点是否消失；若没法安排试读，应如实称为作者自检，不写成读者验证。

## 三、对本项目的归纳

本项目将“说人话”定义为：**作者不降低技术准确性，而是替目标读者显式搭出从问题到机制、从具体例子到一般规则的推理台阶；读者能够沿文本看见对象、动作、条件、变化和理由，并能将它们重新对应到术语、公式、代码或真实系统。**

可用于初稿的讲解路径是：

> 读者的问题或现象 → 本节目标和必要前提 → 小而完整的例子 → 逐步展示动作、变化和原因 → 正式术语/定义 → 公式、代码或架构对应 → 一般规则 → 适用边界 → 复述、复走或变式检查

这不是每种内容的硬模板。讲算法时仍按“问题—定义与假设—目标—推导—算法—算例—局限”完整展开；讲系统时可追踪一条请求、一个数据块或一个资源账本；读者已有足够基础时可压缩已掌握的铺垫。关键要求是不能用缩小知识范围、口语化结论或一个漂亮比喻代替必要的解释。

## 四、证据和适用边界

- Digital.gov、CDC、ONS、GOV.UK、Google、Microsoft、W3C、CAST 和大学教学中心主要提供实践指南或设计框架。它们能帮助建立编辑检查项，不能据此声称每项建议均已由实验独立证明。
- worked-example 研究支持在一些高步骤、低先备知识任务中先提供完整示例；研究对象涉及电路故障排查、概率计算等，并不直接等同于所有机器学习和大模型教材。认知水平上升后可能出现 expertise reversal，所以“示例先行”应是初学者的默认支架，不是所有读者的统一规则。
- 小例子只有在保留关键关系时才有用。玩具例子应标出假设和简化，再明确回到真实机制；过度简化会制造错误直觉。
- 类比教学综述支持显式映射和限制，但并未主张凡事都用类比。若真实的小例子已经够清楚，继续讲故事只会增加额外对象。
- 自我解释与检索练习研究分别关注特定学习任务中的解释活动和延迟记忆。它们启发我们检查读者是否能复述、重走推理、处理变式，但不支持把“能说出一句话”当作充分掌握的证明。
- 句子短、用词常见、标题清楚有助于定位和读取，但可读性分数不能测出知识覆盖、因果推理、公式正确性或跨场景迁移。技术正文仍须保留关键术语、数学、假设和失败条件。
- 医疗场景的 teach-back、政府网页的 user needs、软件文档的步骤规范都有各自场景。本项目借鉴其中可迁移的检查方法，须标为编辑综合，不把场景外推说成来源原结论。

## 后续核验

- 不同领域和不同经验水平的读者，对示例密度、术语背景和类比的最优选择可能不同；具体章节应根据读者反馈调整。
- 本书系目前没有系统的初学者试读数据。因此“复述、复走、变式检查”是规范要求的自检方法，不代表项目已经完成用户研究。
- 需要把教学研究推广到大模型技术教材时，应结合各章读者前置、任务复杂度、既有研究和实际试读，避免把特定实验结论外推成统一写作定律。
