# 第二章 Web-Scale 数据采集：从可访问页面到可治理语料

大模型的数据采集常被简化成“写一个爬虫，把网页下载下来”。这个比喻只描述了最早的一步，甚至容易把真正困难的部分遮住：下载成功不代表获得了训练许可，解析出文本不代表保留了原文结构，页面数量增加不代表有效知识增加，公开可见也不代表没有隐私、版权或评估污染风险。

更准确的对象是一个带有来源、权限、时间、结构、风险和处理历史的数据资产。它要能回答：这段文本从哪里来，按照什么规则取得，原始页面是什么版本，解析时丢失了什么，为什么被保留或删除，进入了哪些数据集和模型，以及收到删除或纠错请求后如何定位下游副本。

本章从 Web-scale 数据采集的历史动机开始，逐步讲清来源类型、访问和使用边界、Common Crawl 的数据形态、采集与解析 pipeline、质量与安全检查、去重和污染、代码与多语言数据、血缘与删除、线上观测以及一个合成审计案例。所有工程例子都使用公开、授权或合成场景；涉及具体法域的法律结论必须交由组织的法务和隐私团队确认。

## 2.1 采集的对象不是网页，而是带证据的数据对象

### 2.1.1 小白视角：建一座有借阅记录的图书馆

可以把 Web-scale 采集想象成建设一座城市图书馆。把纸箱运到仓库只是搬运，真正的图书馆还要知道每本书的来源、版本、许可、分类、保存期限和借阅记录。某本书被发现有错误时，管理员要能找到它存在哪些书架；版权方要求撤下某一版时，管理员要知道哪些复制品和索引需要处理。

网页数据也有同样的层次：

1. URL 或文件地址只是定位线索，不是内容身份。
2. HTML 响应只是原始载体，不是适合训练的文本。
3. 清洗后的文本不是“无来源的字符串”，仍然需要保留内容 hash、来源、时间和处理版本。
4. 进入训练的 token 不是唯一副本，数据还可能存在于 raw lake、解析缓存、去重索引、token shard、评估副本和日志中。
5. 一条“可以访问”的记录，不能自动推出“可以用于训练、可以商业发布或可以再分发”。

因此，采集项目的第一个交付物不应是 URL 数量，而应是一个可以追溯的 source registry 和一套明确的数据对象 schema。

### 2.1.2 专家视角：采集是受约束的分布构造

设原始记录为：

~~~math
r_i=(c_i,s_i,t_i,a_i,h_i,m_i),
~~~

其中 `c_i` 是原始内容，`s_i` 是来源标识，`t_i` 是取得时间，`a_i` 是访问和使用状态，`h_i` 是内容哈希，`m_i` 是 MIME、语言、响应头和其他元数据。后续数据集不是原始记录的简单子集，而是多个处理函数共同作用的结果：

~~~math
D_train
= F(D_raw; policy, parse, quality, privacy, dedup, contamination, version).
~~~

`policy` 决定用途和权限，`parse` 决定如何从载体提取结构，`quality` 描述内容和任务质量，`privacy` 描述个人信息或秘密风险，`dedup` 处理重复，`contamination` 处理评估污染，`version` 把整条链路绑定到一个可回放版本。这个表达式的意义是：数据工程改变了模型最终看到的经验分布，采集阶段的决策会在很久以后表现为能力、风格、幻觉、安全和评估结果。

## 2.2 为什么大模型需要 Web-scale 数据

### 2.2.1 从专项标注到通用预训练

早期 NLP 系统通常围绕一个任务收集标注样本，例如情感分类、命名实体识别、机器翻译或问答。这样的数据集可以有很高的标签质量，却很难覆盖一个通用助手需要的全部内容：多种文体、多个领域、不同语言、代码、表格、长文结构和真实用户表达。

预训练把学习目标改成从大规模文本中预测下一个 token 或重建被遮挡的内容。模型因此有机会接触语言结构、事实陈述、程序语法、文档布局和任务模式。Web 的优势是规模大、更新快、领域广、语言多；它的代价是噪声、重复、权利状态和偏见也一起被放大。

### 2.2.2 GPT-3 的规模启发不能被误读

GPT-3 研究展示了大规模自回归模型在海量文本上预训练后，通过上下文示例完成多种任务的能力。它支持一个重要判断：训练数据规模和多样性是通用能力的重要条件。

它并不支持以下更强的结论：

1. 所有网页都同样有价值。
2. 只增加抓取页面就能线性增加能力。
3. 公开网页可以不经许可、隐私和安全审查直接使用。
4. 一个公开 benchmark 的高分就能证明数据分布没有污染。

Web-scale 的真正难点从“找到更多页面”转向“找到能够解释、筛选、复现和治理的有效信号”。

### 2.2.3 The Pile、C4 与开放语料的启发

The Pile 把多种来源组织为一个公开语料，说明来源多样性、子集文档化和领域覆盖可以共同设计。C4 则展示了从 Common Crawl 快照经过过滤得到大规模语料的过程，同时也暴露了来源意外、评估样本混入和过滤规则可能产生群体差异等问题。

这些工作的共同启发不是“复制某个公开数据集”，而是把数据集当作研究对象：要记录来源组成、过滤规则、重复处理、语言分布、风险和评估边界。一个数据集名称本身不能替代这些信息。

## 2.3 不同来源承担不同责任

来源类型不能只按“质量高/质量低”排序。每类来源都提供特定能力信号，也带来特定的结构、许可、隐私和偏差问题。

### 2.3.1 网页：覆盖广，但来源和结构复杂

网页可以提供新闻、教程、讨论、产品文档、百科、博客、代码和多语言内容。它的优势是覆盖广、更新快、真实用户表达丰富；弱点是模板、广告、SEO 页面、镜像、自动生成内容和过时信息数量巨大。

网页采集至少要区分：

1. 页面本身的内容和导航、广告、推荐等 boilerplate。
2. 页面当前版本与历史版本。
3. 页面发布者、托管平台和内容作者的不同身份。
4. 页面可见性、抓取偏好和训练使用权的不同含义。
5. 正文中的事实、引用、代码、图片和外部链接。

“抓到了一个 HTML 文件”只说明传输完成，不说明正文抽取、事实可靠性或用途授权已经完成。

### 2.3.2 书籍：长结构和完整解释，但权利边界更敏感

书籍对长文结构、系统知识、叙事和专业表达很有价值。它们通常比碎片网页更连贯，也更适合学习章节层次、定义和推导。

但书籍常有明确版权和发行渠道限制。扫描书还会引入 OCR 错误、页眉页脚重复、公式丢失和图表断裂。采集时要把许可来源、版本、页码和 OCR/解析版本记录下来，不能因为文件已经在某个下载目录中就把它视为可训练资产。

### 2.3.3 论文和技术文档：信息密度高，但解析误差会改变含义

论文和 API 文档包含定义、实验条件、代码接口和引用关系，适合科学、工程和专业领域能力。PDF 解析却可能把双栏顺序打乱，把上下标丢失，把表格变成无序文本，把代码和正文粘在一起。

对于论文数据，应尽量保留标题、作者、章节、公式、图表说明、引用和版本；对技术文档，还要保留版本号、发布日期、接口路径和示例代码的边界。解析出的文本应与原始页面或 PDF 建立坐标关系，便于抽样复核。

### 2.3.4 代码：需要文件级和许可证级上下文

代码来源可能包括开源仓库、官方文档、教程、问答和测试。代码模型不仅要学习局部语法，还要学习依赖、目录结构、构建命令、测试和错误修复。

代码采集的特殊问题包括：

1. 仓库 fork 和复制造成的重复权重。
2. 不同文件的许可证和仓库级声明可能不一致。
3. 示例中的邮箱、内部 URL、凭证和测试秘密。
4. 代码能够运行不等于安全，漏洞模式可能被重复学习。
5. 函数片段脱离依赖和测试后，无法代表真实工程任务。

如果目标是 coding agent，最好保留 commit、目录、依赖、测试结果和文件关系；如果目标只是代码补全，可能选择更细的片段，但要明确能力边界。

### 2.3.5 论坛和问答：真实问题多，但答案质量分布宽

论坛和问答包含长尾问题、调试过程、用户语言和失败经验，这是正式文档不容易覆盖的信号。它们也更容易出现未经验证的答案、过时版本、攻击性内容、个人故事和平台条款限制。

处理论坛数据时，问题、回答、评论、投票和编辑时间不能全部拼成一段文本。应保存角色、时间、上下文、采纳状态和版本；“得票高”可以作为质量先验，但不是事实正确性的证明。

### 2.3.6 对话和用户日志：最贴近产品，也最需要治理

人工标注对话、客服记录、用户反馈和线上日志可以揭示真实任务、失败模式和语言分布。但它们可能包含个人信息、商业秘密、健康信息、身份关系和未授权的第三方内容。

用户日志进入训练前至少要明确用途、同意或其他适用依据、脱敏方式、保留期限、人工访问权限、删除流程和数据版本。日志中的“用户没有投诉”不能当作质量标签，“用户说得很像某人”也不能直接保留为训练文本。

### 2.3.7 多语言数据：自然比例通常不是目标比例

高资源语言在互联网上更容易获得大量文本，低资源语言可能只有少量、翻译生成或质量不均的样本。若直接按抓取量训练，模型会把互联网的资源不平等当作目标分布。

语言识别在短文本、混合语言、方言、代码和低资源语言上可能出错。翻译扩充可以增加覆盖，却可能传播翻译腔、事实错误和文化语境损失。最终配比应同时考虑用户分布、任务重要性、资源稀缺度、样本质量和模型容量。

## 2.4 访问、许可、robots 和用途不是同一件事

### 2.4.1 四个问题必须分开

采集项目经常把下面四个问题混在一起：

| 问题 | 它回答什么 | 它不能推出什么 |
| --- | --- | --- |
| 能否建立连接 | 网络和服务是否允许这次请求到达 | 不能推出训练许可 |
| robots 规则怎么写 | 站点向自动抓取程序表达的访问偏好 | 不能单独替代版权或合同审查 |
| Terms of Service 怎么写 | 平台与使用者之间的合同和使用条件 | 具体法律效力需结合适用法域解释 |
| 内容许可是什么 | 内容作者或权利人授予的使用范围 | 不能自动覆盖页面中的第三方内容或个人信息 |

RFC 9309 标准化了 Robots Exclusion Protocol 的语法和处理方式，但 robots 文件不是万能的授权文件，也不是绕过访问控制的技术指南。工程上应尊重站点声明和访问控制；对于是否可以训练、商业使用、再分发或永久保存，应另行完成许可、隐私和法务审查。

### 2.4.2 代码许可证和数据集许可证也要分层

开源代码不等于“没有条件”。仓库可能使用 MIT、Apache-2.0、GPL 或其他许可证，文件还可能包含第三方代码、生成文件和不同声明。训练使用、模型发布、生成代码分发和许可证义务之间的关系不能用一个 `license = open` 字段表示。

数据集也可能只允许研究使用、禁止再分发、要求署名或限制商业用途。source registry 应保存原始许可证文本或稳定引用、审查结论、审查人/团队、用途范围和失效时间；不确定时应把不确定性作为状态，而不是默认为允许。

### 2.4.3 个人信息和秘密是另一条风险轴

即使一段文本可以公开访问，也可能包含邮箱、电话号码、地址、健康信息、账户标识、访问令牌、内部 URL 或商业秘密。许可证审查不能替代 PII、秘密和安全扫描；把邮箱替换成占位符也不一定能解决可重识别问题。

对于高风险字段，处理动作可能是删除、泛化、脱敏、隔离、人工复核或不进入训练。动作要保留理由和版本，避免“扫描通过”被误读成“绝对没有个人信息”。

## 2.5 Common Crawl 及其数据形态

### 2.5.1 为什么需要区分原始归档和提取文本

Common Crawl 等公开 Web 归档让研究者可以使用已经完成的抓取结果，但使用归档并不等于跳过治理。数据中仍有来源、时间、重复、版权、隐私、恶意文件和评估污染问题。

Web 归档通常至少有三类互补信息：

1. WARC 保存抓取记录和响应载荷，适合追溯原始响应、响应头和时间。
2. WAT 保存面向元数据和链接等结构的提取结果，适合筛选和来源分析。
3. WET 保存从网页响应中提取的纯文本，便于批量文本处理，但可能已经丢失布局、图片、脚本关系和部分结构。

不同产品或快照的字段和处理细节要以对应版本文档为准。WET 不是 WARC 的无损替代：当正文抽取出现疑问时，需要回到原始归档或可核验的页面版本。

### 2.5.2 快照选择会改变时间分布

选择单个 crawl 快照会引入时间偏差。选择多个快照可以增加更新覆盖，却可能重复采集同一页面、放大某些站点的更新频率或把已删除内容再次带入数据集。

设某个来源在时间窗口 `T` 内被采集 `n_s(T)` 次，训练中真正使用的 token 数为 `d_s(T)`，则来源权重不仅由页面数量决定，还由更新频率、页面长度、过滤结果和采样策略共同决定：

~~~math
w_s
=\frac{d_s(T)}{\sum_j d_j(T)}.
~~~

报告来源比例时要注明是原始记录比例、过滤后文档比例、token 比例还是训练实际采样比例。否则同一个数据集可以产生几个看似矛盾的“来源占比”。

## 2.6 从来源规划到可回放数据集

### 2.6.1 先定义能力目标和排除范围

采集前要回答模型要服务什么任务：通用文本、多语言、代码、科学、企业知识、实时资料还是多模态理解。目标会影响来源优先级、时间窗口、保留结构、质量标准和评估集合。

还要明确排除范围：未授权登录内容、无法解释许可证的来源、未经处理的用户私密日志、带秘密的代码、评估 holdout、不能安全解析的文件和超出组织处理区域的数据。排除范围写清楚，后续才能解释为什么“采集总量”比“最终 token”大很多。

### 2.6.2 source registry：把来源当作一等对象

一个来源记录至少应包括：

~~~text
source_id
owner_or_publisher
source_type
access_method
license_reference
allowed_use
terms_reference
robots_snapshot
region_constraint
privacy_class
collection_window
retention_policy
deletion_contact
processor_revision
status
~~~

`robots_snapshot` 记录当时看到的规则，不能假设今天的 robots 文件能解释半年前的采集行为。`processor_revision` 用来区分同一来源经过不同解析和过滤代码后的结果。`status` 可以是待审查、授权使用、限制使用、暂停、撤回或不使用，不能只有一个布尔字段。

### 2.6.3 采集请求的工程行为

在获得适用授权并确认请求范围后，采集器仍需要像一个可靠的分布式客户端：

1. 遵守服务端明确的速率和并发限制。
2. 设置清晰的 User-Agent 和联系信息。
3. 使用超时、指数退避和有限重试。
4. 区分 4xx、5xx、连接失败、内容截断和解析失败。
5. 使用 ETag、Last-Modified 或来源提供的快照机制减少重复传输。
6. 保存请求时间、响应状态、内容长度和 hash，而不是只保存成功 URL。
7. 对压缩包、PDF、脚本和图片使用隔离解析环境，限制资源和文件大小。
8. 绝不把绕过登录、绕过访问控制或规避站点限制当成采集策略。

采集器的成功率不能只看 HTTP 200。一个返回 200 的登录页、验证码页或错误模板可能会污染整个数据集，因此还要检查内容类型、正文长度、模板指纹和解析状态。

### 2.6.4 raw layer：可重放但不过度复制敏感内容

原始层的目标是支持审计和重跑。常见字段包括原始字节的 hash、来源 ID、URL 的规范化和原始形式、响应头、状态码、取得时间、快照 ID、内容长度、压缩方式和存储位置。

原始层并不意味着无限期保存所有个人信息。对高敏感数据，可以保留受控的加密原文、内容摘要、哈希和访问记录，并将保留期限和删除流程写入数据资产管理。原始副本越多，删除和访问审计的范围越大。

## 2.7 解析：把载体变成结构，同时保留可复核性

### 2.7.1 HTML 解析不能只用正则删除标签

HTML 页面可能包含主标题、章节、代码、表格、引用、图片 alt、导航、评论、广告和脚本。简单删除所有标签会丢失标题层级和代码边界；只保留 `article` 标签又可能漏掉正文或把用户评论当成正式内容。

解析器应输出结构化 artifact，例如：

~~~text
document_id
source_id
title
sections[]
paragraphs[]
code_blocks[]
tables[]
links[]
published_at
updated_at
parser_revision
source_coordinates
parse_warnings[]
~~~

`source_coordinates` 可以是 DOM 路径、页码、字符区间或快照记录。它让人工审计能够从清洗文本回到原始载体；`parse_warnings` 则避免把“解析失败后得到的空文本”误当成页面没有内容。

### 2.7.2 PDF、扫描件和技术文档

PDF 可能是可搜索文本、双栏排版、扫描图像或混合文档。不同类型需要不同路径：

1. 可搜索 PDF 要检查阅读顺序、字符编码和公式上下标。
2. 扫描 PDF 需要 OCR，并记录 OCR 引擎、语言和置信度。
3. 双栏和表格需要版面分析，不能只按字符流拼接。
4. 公式、代码和引用要单独识别，避免把变量名变成普通词。
5. 图像中的文字要保留图像坐标和 OCR 版本，不能只存一段不可定位的文本。

解析质量可以用人工抽样、字段召回、标题顺序准确率、表格单元格准确率和公式可读性评估。对于高影响领域，解析失败样本应进入隔离队列而不是静默丢弃。

### 2.7.3 代码仓库的解析

代码解析要同时保留语言、路径、仓库、commit、依赖、许可证和测试信息。删除注释和字符串可能损伤上下文；保留全部文件又可能包含构建产物、二进制、锁文件和秘密。

一个仓库级样本可以表示为：

~~~math
\mathcal{R}=(F,G,T,E,L,V),
~~~

其中 `F` 是文件集合，`G` 是目录和依赖关系，`T` 是测试或执行结果，`E` 是运行环境，`L` 是许可证信息，`V` 是 commit 或版本。不同训练目标可以从同一仓库对象构造函数级、文件级或任务级样本，但不能在构造时忘记它们的共同来源。

### 2.7.4 Boilerplate removal 的反事实检查

去导航、页脚和 cookie banner 可以提高信息密度，但规则可能误删正文。一个可靠的检查不是只看删除比例，而是抽取删除前后的成对样本：

1. 删除的文本是否在不同页面重复出现。
2. 被保留的标题是否仍能解释段落。
3. 代码、表格和警告框是否被错误删除。
4. 不同语言页面是否受到不同影响。
5. 低资源领域是否比普通网页有更高的误删率。

过滤器的 precision 和 recall 只能相对于一个标注协议解释。若没有人工样本和负对照，“删除了 70% 模板”并不能证明剩下的 30% 都是正文。

## 2.8 质量、隐私、安全和污染检查

### 2.8.1 质量检查的层次

质量可以拆成几个不同层次，而不是用一个总分替代：

1. 载体质量：状态码、编码、MIME、正文长度和解析完整性。
2. 语言质量：语言识别、混合语言、乱码和字符分布。
3. 内容质量：重复、模板、广告、事实来源和时效。
4. 任务质量：输入输出是否清楚、答案是否可验证、难度是否匹配。
5. 结构质量：标题、代码、表格、引用和多轮关系是否保留。

网页质量分类器可以作为排序信号，不能取代来源审查和抽样。一个“写得很流畅”的自动生成页面，可能比一段不规范但有价值的调试记录更容易拿到高分，因此需要按任务和领域校准。

### 2.8.2 PII、秘密和恶意内容

扫描可以使用规则、命名实体识别、秘密检测器、分类模型和人工复核的组合。规则对邮箱、电话和常见令牌格式有用，语义模型对间接身份和上下文有帮助，但任何单一检测器都会漏报和误报。

工程动作应与风险类型对应：

| 风险 | 可能动作 | 仍需验证的事项 |
| --- | --- | --- |
| 明文邮箱或电话 | 删除、掩码、隔离 | 是否可由上下文重新识别 |
| API key 或 token | 删除并轮换相关凭证 | 是否已进入缓存、日志或版本库 |
| 医疗或财务信息 | 高敏感隔离、人工复核或不使用 | 法域、用途和保留期限 |
| 恶意代码或危险指令 | 内容隔离、沙箱分析或不使用 | 是否在训练和评估链路留下副本 |
| 外部指令注入 | 保留来源标签，避免当作系统指令 | 下游训练格式是否会放大其优先级 |

扫描通过只能说明当前检测器没有命中，不等于证明不存在隐私或安全问题。

### 2.8.3 评估污染

benchmark 题目、答案、模板和近义改写可能出现在网页、代码仓库、教程和论坛。污染检测可以使用 exact match、n-gram、MinHash、embedding 或结构化代码相似度，但相似不自动等于泄漏。

设评估集有 `M` 个样本，`e_j` 与训练集合中最相似样本的相似度为 `sim_max(e_j)`，给定阈值 `tau`，一个污染候选率可以写成：

~~~math
R_contam
= (1 / M) * sum(j = 1..M) I[sim_max(e_j) >= tau].
~~~

这个指标必须附带阈值、相似度方法、人工复核协议、评估版本和训练数据时间范围。命中后可以删除、隔离、改换评估集、降低结论等级或公开污染说明；不能只把数字从报告里删掉。

## 2.9 去重：减少冗余，也减少错误权重

### 2.9.1 URL 去重不是内容去重

不同 URL 可能返回相同内容，同一个 URL 也可能随着时间变化。URL 规范化可以处理大小写、默认端口、追踪参数和尾部斜杠，但过度规范化会把有意义的查询参数或语言版本合并掉。

因此至少要同时记录原始 URL、规范化 URL、响应时间和内容 hash。URL hash 保护日志和 catalog 中的敏感路径，内容 hash 用于判断实际载荷是否相同。

### 2.9.2 exact、near 和结构化去重

Exact dedup 对规范化后的文本计算 hash，便宜而精确；near dedup 使用 MinHash、SimHash、n-gram 或 embedding 发现改写、模板和复制；代码还需要 AST、token 序列或仓库关系等结构化信号。

设解析后 token 数为 `D_before`，完全去重后 token 数为 `D_after`，token 口径的减少率为：

~~~math
R_removed = 1 - D_after / D_before.
~~~

减少率越高不一定越好。教程中的重复定义、安全边界和代码模式有时是有意的教学信号；若 near-dedup 把这些独立样本合并，可能损失覆盖。应保留 cluster ID、代表样本选择规则、阈值和分领域统计。

### 2.9.3 去重与记忆风险的关系

重复会提高某些文本的有效采样权重，可能增加训练记忆和评估重叠；但记忆还受模型容量、训练步数、样本稀有度和后训练影响。去重是降低风险的手段，不是“模型不会记住数据”的证明。

## 2.10 配比、时间和语言覆盖

### 2.10.1 原始比例不等于训练比例

来源的原始 token 数会经过过滤、去重、截断、packing 和 sampler 改变。若来源 `i` 经过处理后有 `d_i` 个有效 token，训练配比为：

~~~math
p_i=\frac{d_i}{\sum_jd_j}.
~~~

但真正的曝光量还要考虑重复采样和不同阶段的 mixture。报告时要同时给出 raw、processed、sampled 和 consumed 四种口径。

### 2.10.2 温度采样与低资源语言

若语言或来源的有效大小为 `n_i`，可以用温度指数 `alpha` 作为一种采样抽象：

~~~math
p_i=\frac{n_i^{\alpha}}{\sum_j n_j^{\alpha}}.
~~~

`alpha = 1` 接近按数据量采样，`alpha < 1` 提高小数据源的相对权重，`alpha = 0` 接近来源级均匀。这个公式只描述采样倾向，不解决低资源数据的事实质量、翻译腔、重复暴露和文化覆盖问题。

### 2.10.3 时间切分和事实新鲜度

网页知识会变化，训练数据的时间窗口也会改变模型对事实的记忆。可以按发布日期、抓取时间和最后更新时间做切片；对更新频繁的资料，应同时保留版本，而不是用最新页面覆盖历史记录。

时间切分还能帮助污染排查：训练数据早于评估集不代表没有泄漏，但训练数据晚于评估集会增加评估被直接看到的可能性。时间是证据的一部分，不是独立性证明。

## 2.11 数据血缘、删除和回放

### 2.11.1 从页面到模型的血缘图

一条网页记录的下游路径可能是：

~~~text
source page / archive record
  -> raw object
  -> parsed artifact
  -> normalized document
  -> dedup cluster
  -> filtered dataset version
  -> token shard
  -> training manifest
  -> checkpoint / model release
  -> evaluation and logs
~~~

每条边都应有处理版本和 hash。只保存最终 shard 会导致“模型中是否使用过某来源”无法回答，也会让删除请求只能停留在 catalog 标记层面。

### 2.11.2 删除请求不是一个文件操作

收到删除、许可撤回或隐私请求后，工程流程通常要：

1. 暂停新训练和新索引继续使用相关来源。
2. 根据 source ID、content hash、dedup cluster 和处理版本定位副本。
3. 检查 raw、解析、缓存、token shard、评估集和日志。
4. 重新生成受影响的数据版本，或记录无法直接重建的范围。
5. 判断已训练模型需要重训、编辑、unlearning、风险披露还是法律团队给出其他处置。
6. 用回归评估验证删除或替代数据没有引入新的能力和安全回归。

从索引和 catalog 删除只能证明检索路径发生变化，不能直接证明模型权重、缓存和历史输出已经忘记。

### 2.11.3 可回放版本的最小字段

~~~text
dataset_id
source_snapshot
processor_revision
filter_revision
dedup_revision
contamination_revision
mixture_revision
tokenizer_revision
manifest_digest
created_at
retention_policy
deletion_status
~~~

版本对象还应保留变更摘要：增加和删除了哪些来源，哪些过滤器改变，语言和领域比例如何变化，风险命中和评估污染如何变化。文件名变化不等于版本可解释，文件名不变也不代表内容没有改变。

## 2.12 真实项目架构：从采集器到数据产品

### 2.12.1 组件边界

一个持续运行的采集系统可以分成：

1. Source registry：来源、许可证、用途、区域和联系人。
2. Policy review：权限、隐私、保留和使用范围的审查记录。
3. Collector：限速、重试、快照、响应和失败事件。
4. Raw store：原始对象、hash、时间和访问控制。
5. Parser：HTML、PDF、代码、表格和多模态解析。
6. Metadata service：语言、领域、版本、结构和 provenance。
7. Quality and risk services：质量、PII、秘密、安全和污染检查。
8. Dedup service：exact、near、代码和跨快照去重。
9. Dataset builder：切片、配比、抽样和 manifest。
10. Evaluation registry：训练/评估隔离、污染状态和 holdout 访问。
11. Version registry：不可变 digest、变更摘要和下游引用。
12. Audit and deletion service：查询、删除、申诉和证据导出。

这些组件之间传递的不只是文本，还要传递状态和原因。比如 parser 发现表格丢失时，dataset builder 不能只看到一个空字符串；删除服务需要知道这个字符串来源于哪个 raw object 和 processor revision。

### 2.12.2 观测指标

采集系统可以记录：

~~~math
R_fetch = N_successful_responses / N_attempts,
R_parse = N_usable_artifacts / N_successful_responses.
~~~

还应按来源、语言、MIME、时间和错误类型分桶观察：状态码、内容类型、正文长度、解析警告、质量分布、PII 命中、重复率、污染候选、处理延迟、存储成本和删除传播时长。平均 fetch rate 可能掩盖某个低资源来源几乎全部解析失败，因此切片分母比总体数字更重要。

### 2.12.3 失败重试和幂等

采集任务会被调度器重试、快照重放或人工重新运行。每个任务需要稳定的 `source_id`、请求或记录 ID、内容 hash 和处理版本，写入应具有幂等性。否则同一页面的重复事件可能被误当作新数据，重复下载也会增加服务压力和成本。

重试还要区分传输失败和内容失败：网络超时可以有限重试，解析器对同一损坏 PDF 连续失败则应进入隔离队列；被策略暂停的来源不应通过调度重试自动恢复。

## 2.13 工程案例：企业研究助手的公开资料和授权资料混合

### 2.13.1 目标和边界

一家企业要构建研究助手，资料来自公开技术文档、已授权的行业报告和内部知识库。助手需要给出带页码或段落位置的引用，不能把内部资料泄露给其他租户，也不能把许可已撤回的报告继续检索出来。

采集团队把三个来源分开注册：

1. 公共技术文档：记录快照、版本和页面更新时间。
2. 授权行业报告：记录合同范围、地区、到期日和禁止再分发字段。
3. 内部知识库：记录租户、部门 ACL、保留期限和删除联系人。

它们可以共享解析器和索引服务，但不能共用一个无差别的 `public = true` 字段。

### 2.13.2 一次解析事故

某批 PDF 的双栏阅读顺序被解析器颠倒，研究助手把“不得用于生产”的限制条件和上一段的“可以用于生产”拼在了一起。问题不是模型突然变笨，而是 parser artifact 丢失了版面关系。

修复步骤是：

1. 用原始 PDF 和页码坐标复核错误样本。
2. 把该 parser revision 产生的所有 artifact 标为受影响集合。
3. 对标题、表格、警告框和双栏顺序增加抽样指标。
4. 重新解析并生成新的 dataset/index version。
5. 用包含限制条件的回归问题测试引用范围和答案方向。
6. 检查旧 artifact、缓存和日志的访问与删除状态。

这个案例说明数据采集的质量问题可能在产品层表现为事实错误、引用缺失或权限事故；如果没有 source、page、processor revision 和 index version，团队只能重新猜测问题发生在哪里。

## 2.14 可运行的合成采集审计示例

下面的 demo 不联网、不读取真实网页，也不处理真实凭证。它用合成 HTML 模拟九条记录，展示 policy、解析、PII/秘密、评估污染、质量和 exact dedup 如何分别产生信号。代码没有把所有条件压成一个布尔总开关，而是返回保留样本、拒绝原因、阶段计数、风险信号和后续动作。

~~~python
from __future__ import annotations

import hashlib
import re
from collections import Counter


sources = {
    "tech_blog": {
        "license": "cc-by",
        "training_use": True,
        "robots": "allowed",
        "access": "public",
    },
    "oss_docs": {
        "license": "apache-2.0",
        "training_use": True,
        "robots": "allowed",
        "access": "public",
    },
    "science_preprint": {
        "license": "cc-by",
        "training_use": True,
        "robots": "allowed",
        "access": "public",
    },
    "zh_news": {
        "license": "authorized",
        "training_use": True,
        "robots": "allowed",
        "access": "public",
    },
    "private_forum": {
        "license": "unknown",
        "training_use": False,
        "robots": "disallowed",
        "access": "login_required",
    },
}

documents = [
    {
        "id": "blog_attention",
        "source_id": "tech_blog",
        "url": "https://example.invalid/blog/attention",
        "crawl_time": "2026-06-01T00:00:00Z",
        "mime": "text/html",
        "language": "en",
        "domain": "web_ml",
        "html": """
        <html><body><nav>Home Archive</nav><article>
        Transformer attention uses queries keys and values for language model training.
        This article explains data quality, deduplication, provenance, and evaluation contamination.
        </article><footer>Contact</footer></body></html>
        """,
    },
    {
        "id": "oss_vector_db",
        "source_id": "oss_docs",
        "url": "https://docs.example.invalid/vector-db",
        "crawl_time": "2026-06-01T00:05:00Z",
        "mime": "text/html",
        "language": "en",
        "domain": "code_docs",
        "html": """
        <main>Open source vector database documentation explains indexes, tests,
        examples, licenses, failure modes, and reproducible deployment commands.</main>
        """,
    },
    {
        "id": "blog_attention_copy",
        "source_id": "tech_blog",
        "url": "https://mirror.example.invalid/blog/attention-copy",
        "crawl_time": "2026-06-01T00:10:00Z",
        "mime": "text/html",
        "language": "en",
        "domain": "web_ml",
        "html": """
        <html><body><nav>Home Archive</nav><article>
        Transformer attention uses queries keys and values for language model training.
        This article explains data quality, deduplication, provenance, and evaluation contamination.
        </article><footer>Contact</footer></body></html>
        """,
    },
    {
        "id": "private_forum",
        "source_id": "private_forum",
        "url": "https://forum.example.invalid/private/thread/7",
        "crawl_time": "2026-06-01T00:15:00Z",
        "mime": "text/html",
        "language": "en",
        "domain": "forum",
        "html": "<article>Private member discussion with project details.</article>",
    },
    {
        "id": "spam_seo",
        "source_id": "tech_blog",
        "url": "https://example.invalid/seo/spam",
        "crawl_time": "2026-06-01T00:20:00Z",
        "mime": "text/html",
        "language": "en",
        "domain": "spam",
        "html": "<body>Buy now promo promo promo click click click</body>",
    },
    {
        "id": "pii_secret",
        "source_id": "tech_blog",
        "url": "https://example.invalid/leak",
        "crawl_time": "2026-06-01T00:25:00Z",
        "mime": "text/html",
        "language": "en",
        "domain": "security",
        "html": "<article>Contact reader@example.invalid and use DEMO_SECRET_12345678.</article>",
    },
    {
        "id": "eval_leak",
        "source_id": "tech_blog",
        "url": "https://example.invalid/benchmark/answer",
        "crawl_time": "2026-06-01T00:30:00Z",
        "mime": "text/html",
        "language": "en",
        "domain": "eval",
        "html": "<article>Benchmark answer key: GSM8K solution and hidden test answer.</article>",
    },
    {
        "id": "paper_scaling",
        "source_id": "science_preprint",
        "url": "https://papers.example.invalid/scaling",
        "crawl_time": "2026-06-01T00:35:00Z",
        "mime": "text/html",
        "language": "en",
        "domain": "science",
        "html": """
        <article>Scaling law experiments compare model size, data tokens, compute budget,
        validation loss, ablation controls, and reproducible training recipes.</article>
        """,
    },
    {
        "id": "zh_data_quality",
        "source_id": "zh_news",
        "url": "https://news.example.invalid/data-quality",
        "crawl_time": "2026-06-01T00:40:00Z",
        "mime": "text/html",
        "language": "zh",
        "domain": "zh_web",
        "html": """
        <article>高质量中文语料需要覆盖教育、科技、政策、生活服务和真实问答，
        同时记录来源、许可、时间、语言、质量分、隐私风险和去重状态。</article>
        """,
    },
]

required_meta = [
    "id",
    "source_id",
    "url",
    "crawl_time",
    "mime",
    "language",
    "domain",
    "html",
]
pii_or_secret = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    re.compile(r"DEMO_SECRET_[A-Z0-9]{8,}"),
]
contamination_terms = ["benchmark answer", "gsm8k solution", "hidden test answer"]


def strip_html(html: str) -> str:
    html = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def content_hash(text: str) -> str:
    normalized = " ".join(tokenize(text))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def quality_score(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    alpha_ratio = sum(char.isalpha() for char in text) / max(len(text), 1)
    length_score = min(len(tokens) / 24, 1.0)
    return round(
        0.45 * length_score
        + 0.35 * unique_ratio
        + 0.20 * min(alpha_ratio / 0.65, 1.0),
        3,
    )


def policy_allowed(doc: dict) -> bool:
    source = sources.get(doc.get("source_id"))
    if source is None:
        return False
    return (
        source["license"] not in {"unknown", "restricted"}
        and source["training_use"]
        and source["robots"] == "allowed"
        and source["access"] == "public"
    )


def has_pii_or_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in pii_or_secret)


def has_eval_contamination(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in contamination_terms)


def sum_tokens(items: list[dict], field: str) -> dict[str, int]:
    totals = Counter()
    for item in items:
        totals[item[field]] += item["tokens"]
    return dict(totals)


def safe_ratio(numerator: float, denominator: int):
    return round(numerator / denominator, 3) if denominator else None


def audit_collection(items: list[dict]) -> dict:
    seen_hashes = set()
    kept = []
    rejected = {}
    stage_counts = Counter(raw=len(items))

    for index, doc in enumerate(items):
        row_id = doc.get("id", f"row_{index}")
        missing = [field for field in required_meta if not doc.get(field)]
        if missing:
            rejected[row_id] = "missing_metadata"
            continue

        text = strip_html(doc["html"])
        doc_hash = content_hash(text)
        score = quality_score(text)

        if doc["source_id"] not in sources:
            rejected[row_id] = "unknown_source"
            continue

        if not policy_allowed(doc):
            rejected[row_id] = "policy_block"
            continue
        stage_counts["policy_pass"] += 1

        if has_pii_or_secret(text):
            rejected[row_id] = "pii_or_secret"
            continue
        stage_counts["pii_pass"] += 1

        if has_eval_contamination(text):
            rejected[row_id] = "eval_contamination"
            continue
        stage_counts["contamination_pass"] += 1

        if score < 0.62:
            rejected[row_id] = "low_quality"
            continue
        stage_counts["quality_pass"] += 1

        if doc_hash in seen_hashes:
            rejected[row_id] = "exact_duplicate"
            continue
        seen_hashes.add(doc_hash)
        kept.append(
            {
                **doc,
                "text": text,
                "hash": doc_hash,
                "tokens": len(tokenize(text)),
                "quality": score,
            }
        )
        stage_counts["dedup_pass"] += 1

    # Records without HTML are excluded from the raw-token denominator because
    # their payload cannot be measured; the missing-metadata rate remains visible.
    raw_tokens = sum(
        len(tokenize(strip_html(doc["html"])))
        for doc in items
        if doc.get("html")
    )
    kept_tokens = sum(doc["tokens"] for doc in kept)
    language_tokens = sum_tokens(kept, "language")
    domain_tokens = sum_tokens(kept, "domain")
    language_mix = {
        key: safe_ratio(value, kept_tokens)
        for key, value in sorted(language_tokens.items())
    }
    domain_mix = {
        key: safe_ratio(value, kept_tokens)
        for key, value in sorted(domain_tokens.items())
    }
    denominator = len(items)
    risk_rates = {
        "missing_metadata": safe_ratio(
            sum(reason == "missing_metadata" for reason in rejected.values()),
            denominator,
        ),
        "policy_block": safe_ratio(
            sum(reason == "policy_block" for reason in rejected.values()), denominator
        ),
        "pii_or_secret": safe_ratio(
            sum(reason == "pii_or_secret" for reason in rejected.values()), denominator
        ),
        "eval_contamination": safe_ratio(
            sum(reason == "eval_contamination" for reason in rejected.values()),
            denominator,
        ),
        "low_quality": safe_ratio(
            sum(reason == "low_quality" for reason in rejected.values()), denominator
        ),
        "exact_duplicate": safe_ratio(
            sum(reason == "exact_duplicate" for reason in rejected.values()), denominator
        ),
        "unknown_source": safe_ratio(
            sum(reason == "unknown_source" for reason in rejected.values()), denominator
        ),
    }
    signals = {
        "kept_ids": [doc["id"] for doc in kept],
        "rejected": dict(sorted(rejected.items())),
        "stage_counts": dict(stage_counts),
        "retention": safe_ratio(kept_tokens, raw_tokens),
        "language_mix": language_mix,
        "domain_mix": domain_mix,
        "average_quality": safe_ratio(
            sum(doc["quality"] for doc in kept), len(kept)
        ),
        "risk_rates": risk_rates,
    }
    actions = []
    if not items:
        actions.append("restore_or_collect_source_records")
    if not kept:
        actions.append("restore_nonempty_training_set")
    if risk_rates["missing_metadata"] is not None and risk_rates["missing_metadata"] > 0:
        actions.append("repair_missing_metadata")
    if risk_rates["policy_block"] is not None and risk_rates["policy_block"] > 0:
        actions.append("review_or_exclude_restricted_sources")
    if risk_rates["unknown_source"] is not None and risk_rates["unknown_source"] > 0:
        actions.append("register_or_exclude_unknown_sources")
    if risk_rates["pii_or_secret"] is not None and risk_rates["pii_or_secret"] > 0:
        actions.append("scrub_or_isolate_sensitive_records")
    if risk_rates["eval_contamination"] is not None and risk_rates["eval_contamination"] > 0:
        actions.append("remove_or_isolate_eval_overlap")
    if risk_rates["low_quality"] is not None and risk_rates["low_quality"] > 0:
        actions.append("tune_quality_filter_and_sample_review")
    if risk_rates["exact_duplicate"] is not None and risk_rates["exact_duplicate"] > 0:
        actions.append("run_near_dedup_and_keep_cluster_provenance")
    decision = "hold_for_repair" if actions else "continue_to_manifest"
    return {"signals": signals, "actions": actions, "decision": decision}


report = audit_collection(documents)
print("signals=", report["signals"])
print("actions=", report["actions"])
print("decision=", report["decision"])

assert report["signals"]["kept_ids"] == [
    "blog_attention",
    "oss_vector_db",
    "paper_scaling",
    "zh_data_quality",
]
assert report["signals"]["rejected"] == {
    "blog_attention_copy": "exact_duplicate",
    "eval_leak": "eval_contamination",
    "private_forum": "policy_block",
    "pii_secret": "pii_or_secret",
    "spam_seo": "low_quality",
}
assert report["decision"] == "hold_for_repair"

empty_report = audit_collection([])
assert empty_report["signals"]["retention"] is None
assert empty_report["signals"]["risk_rates"]["missing_metadata"] is None

unknown_source = {**documents[0], "id": "unknown_source", "source_id": "not_registered"}
unknown_report = audit_collection([unknown_source])
assert unknown_report["signals"]["rejected"] == {
    "unknown_source": "unknown_source",
}
assert "register_or_exclude_unknown_sources" in unknown_report["actions"]

missing_metadata = {
    "source_id": "tech_blog",
    "url": "https://example.invalid/missing-metadata",
    "crawl_time": "2026-06-01T00:45:00Z",
    "mime": "text/html",
    "language": "en",
    "domain": "web_ml",
}
missing_report = audit_collection([missing_metadata])
assert missing_report["signals"]["rejected"] == {
    "row_0": "missing_metadata",
}
assert missing_report["signals"]["retention"] is None
assert "repair_missing_metadata" in missing_report["actions"]

filtered_report = audit_collection([documents[3]])
assert filtered_report["signals"]["retention"] == 0.0
assert filtered_report["signals"]["risk_rates"]["policy_block"] == 1.0
assert filtered_report["decision"] == "hold_for_repair"
~~~

示例输出中的数值会由这组合成文本决定，真实项目不应照搬 `0.62` 这样的阈值。这个 demo 重要的地方有四点：策略拒绝、隐私/秘密、评估污染、低质量和重复是不同原因；缺失元数据不是“没有内容”，而是需要单独修复的数据质量问题；空集合和全量过滤时，比例分母会明确返回 `None` 或 `0.0`，避免把不可计算和确实没有保留样本混为一谈；即使最终保留样本的平均质量不错，只要仍有需要处理的风险，数据版本就不能被描述为“已经没有问题”。

## 2.15 如何评估一个采集 pipeline

### 2.15.1 采集质量评估

采集评估需要对每个阶段设置可解释指标：

| 阶段 | 指标例子 | 关键分母 |
| --- | --- | --- |
| 获取 | 成功响应率、截断率、重复请求率 | 请求总数或合法请求总数 |
| 解析 | 可用 artifact 率、字段召回率、警告率 | 成功响应或各 MIME 分组 |
| 质量 | 高质量 precision、误删率、信息密度 | 人工标注样本或领域切片 |
| 隐私安全 | 命中率、漏检率、人工复核一致性 | 带标注风险样本和负对照 |
| 去重 | exact/near 减少率、误合并率 | 文档、token 或 cluster |
| 污染 | 候选命中率、人工确认率 | 评估样本数 |
| 血缘 | 可追溯率、删除传播时长 | 数据对象和下游 artifact |

一个总体平均值不能替代语言、领域、来源、时间和文件类型切片。例如 HTML 解析率 98% 可能掩盖 PDF 解析率只有 40%，英文过滤误删率很低也可能掩盖低资源语言几乎全部被删。

### 2.15.2 采集策略的消融

如果过滤版本 B 比版本 A 的下游 benchmark 高，不能立即把收益归因于某个规则。两个版本可能同时改变了总 token、语言比例、重复率、时间窗口和污染程度。

一个基本消融应固定：

1. 模型架构、训练步数和随机种子范围。
2. 总 token 预算和 tokenizer 版本。
3. 评估集合和评分程序。
4. 来源与语言切片的统计口径。
5. 训练和评估之间的污染检查。

然后比较能力、事实性、拒答、代码、长文、语言覆盖、隐私风险和成本。采集策略的价值是任务条件下的结果，不是过滤规则本身的漂亮数字。

### 2.15.3 数据价值不是单一质量分

可以把某个来源对任务 `t` 的增量价值写成教学用指标：

~~~math
DeltaV(s, t) = Score(D + s, t) - Score(D, t).
~~~

`Score` 可以是代码执行成功率、事实支持率、语言任务准确率或安全边界指标。`Delta V` 受模型规模、训练预算、混合比例和随机种子影响，不能直接解释为来源的普适价值。来源贡献还可能存在互补：单独加入某个小数据源没有收益，与高质量通用数据一起训练才有收益。

## 2.16 常见失败模式

### 2.16.1 把 HTTP 200 当作可用内容

错误页、登录页、验证码和站点模板都可能返回 200。修复方法是记录内容类型、模板指纹、正文长度、解析警告和样本抽样，并按来源观察。

### 2.16.2 把 robots 当成完整授权

robots 是抓取规则的一部分，不等于版权、合同、隐私和训练用途的完整结论。修复方法是把 robots snapshot、ToS、许可证和用途审查分别登记。

### 2.16.3 只保存清洗后的文本

没有 raw 坐标、解析版本和 hash，就无法复核误删、删除、污染和模型血缘。修复方法是保留受控原始 artifact 或足以定位原文的证据对象。

### 2.16.4 只做 exact dedup

页面复制、模板改写、代码 fork 和跨快照重复会绕过 exact hash。修复方法是加入 near-dedup、结构化代码相似度和人工误合并检查。

### 2.16.5 用全局阈值处理所有语言和领域

字符分布、文档长度、标点和链接模式在语言与领域间不同。全局阈值会误删低资源语言、短问题和代码。修复方法是按切片校准并报告误删率。

### 2.16.6 过滤器越多越安全

每增加一个过滤器，都会增加误删、分布变化和难以解释的组合效应。过滤器应有目的、版本、正负对照和下游回归；高风险类别可以隔离并人工复核，不要把所有内容都交给一个黑盒分类器。

### 2.16.7 将线上日志直接回灌

线上日志包含最真实的失败，也包含最敏感的个人和业务信息。应先建立用途、同意/授权、脱敏、抽样、保留、删除和访问控制，再把复核后的样本进入版本化数据集。

### 2.16.8 把“数据集发布”当作终点

数据发布后仍会有许可证撤回、网页更新、删除请求、评估污染发现和模型事故。数据集需要生命周期、版本 diff、回滚、下游模型清单和变更通知。

## 2.17 资料与证据边界

### 2.17.1 规模与训练数据研究

1. [Language Models are Few-Shot Learners（GPT-3）](https://arxiv.org/abs/2005.14165)：支持大规模自回归模型、预训练语料与 few-shot 能力之间的研究背景。
2. [Training Compute-Optimal Large Language Models（Chinchilla）](https://arxiv.org/abs/2203.15556)：支持模型参数、训练 token 和计算预算共同设计的结论。
3. [Textbooks Are All You Need（phi-1）](https://arxiv.org/abs/2306.11644)：支持教材式和合成增强数据在代码模型案例中的效率启发。
4. [The Pile](https://arxiv.org/abs/2101.00027)：支持多来源、子集组织和大规模公开语料的研究案例。

这些论文支持各自实验条件下的观察，不提供任意闭源模型的完整配方，也不证明某个来源在所有模型和任务上都有相同边际价值。

### 2.17.2 Web 语料和数据处理

1. [Common Crawl Overview](https://commoncrawl.org/overview)：支持公开 Web 归档的来源和使用入口；具体快照字段要以对应版本文档为准。
2. [Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus（C4）](https://arxiv.org/abs/2104.08758)：支持 C4 来源追踪、过滤效果、评估样本混入和数据集文档化讨论。
3. [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116)：支持网页过滤、去重和开放 Web 数据构造的研究案例。
4. [The FineWeb Datasets](https://arxiv.org/abs/2406.17557)：支持大规模网页数据处理、质量分析和训练数据研究的较新案例。
5. [Dolma: an Open Corpus of Three Trillion Tokens](https://arxiv.org/abs/2402.00159)：支持多来源开放语料、构造流程记录和数据处理工具的案例。
6. [DataComp-LM](https://arxiv.org/abs/2406.11794)：支持通过受控数据选择实验比较训练数据价值的研究方向。
7. [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)：支持近重复、记忆、训练效率和 train-test overlap 的讨论。

这些论文和数据集文档说明作者如何构造、过滤和评估数据，不自动授予读者相同的版权、隐私或商业使用权。真实项目必须根据实际来源、合同和适用法域单独审查。

### 2.17.3 访问、文档与治理

1. [RFC 9309: Robots Exclusion Protocol](https://www.rfc-editor.org/rfc/rfc9309.html)：支持 robots 语法和协议行为，不能替代许可证或法务结论。
2. [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)：支持记录数据集动机、组成、收集过程、推荐用途和限制。
3. [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)：支持从生命周期、风险识别和治理角度组织数据管理。
4. [NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)：支持生成式 AI 的数据来源、隐私、评估和治理风险讨论。

这些资料提供协议、研究方法和治理框架；它们不能替代组织自己的许可证审查、隐私影响评估、秘密扫描、删除流程和安全测试。

## 2.18 思考与实践

### 题目一：设计来源登记表

为一个包含公共文档、授权报告和内部知识库的研究助手设计 source registry。说明每个字段如何影响采集、索引、训练、删除和再分发。

### 题目二：分析 WARC、WAT 和 WET

解释三种数据形态分别适合什么任务。设计一个解析错误案例，说明为什么只保留 WET 会让问题难以定位。

### 题目三：设计污染实验

构造一个包含 benchmark 原题、模板改写和独立相似问题的测试集。选择两种相似度方法，给出人工复核协议，并说明如何报告阈值和误报。

### 题目四：设计代码数据流程

为 coding agent 设计从仓库、commit、依赖、测试到样本的血缘。说明如何处理 fork、许可证、秘密、生成文件和不能运行的项目。

### 题目五：处理删除请求

假设某个授权来源撤回训练许可。画出 raw、parsed、dedup、token shard、checkpoint、RAG index、cache 和日志之间的路径，说明哪些动作可以直接执行，哪些结论需要重训或额外研究才能得出。

## 2.19 结语：采集是分布设计的第一步

Web-scale 数据采集的难点不在于把请求发出去，而在于把“可访问的载体”转化为“来源明确、结构可复核、风险可处理、版本可回放的数据对象”。网页提供广覆盖，书籍和论文提供长结构与专业知识，代码提供程序模式，论坛和对话提供真实任务，多语言数据决定服务范围；每种来源都需要自己的许可、解析、质量和隐私判断。

一个成熟系统会把 source registry、访问边界、raw artifact、解析版本、质量切片、PII/秘密处理、去重、污染、mixture、评估和删除连接起来。它不会用一个平均质量分或一个总布尔值掩盖不同风险，也不会把公开网页、robots 规则和训练授权混成同一个概念。

当数据团队能够回答一段内容从哪里来、为什么保留、进入了哪个版本、改变了哪种分布、发生事故时如何定位以及删除后哪些结论仍未知，Web-scale 采集才真正从“抓取工程”变成了模型能力和系统责任的基础设施。
