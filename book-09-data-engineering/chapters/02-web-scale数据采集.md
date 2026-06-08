# 第二章：Web-Scale 数据采集

重点：网页、书籍、论文、代码、论坛、对话、多语言数据来源和法律合规问题。

面试重点：Web-scale 数据采集不是“写爬虫把网页都扒下来”，而是一个涉及数据来源、解析、质量、版权、隐私、反滥用、可追溯和后续训练目标的系统工程。

合规边界：本章讨论合法合规的数据采集、公开数据集、数据治理和工程流程，不提供绕过访问控制、规避反爬、抓取未授权内容或违反网站条款的操作方法。

## 本章目标

学完本章，你要能回答：

1. Web-scale 数据采集为什么是大模型训练的基础？
2. 大模型常见数据来源有哪些？
3. 网页、书籍、论文、代码、论坛、对话、多语言数据各有什么特点？
4. Common Crawl、The Pile 这类公开语料给了我们什么启发？
5. Web 数据采集 pipeline 应该如何设计？
6. 如何处理版权、隐私、robots、Terms of Service、数据许可和合规？
7. 如何做数据来源记录和版本治理？
8. 面试中如何回答“如何从零构建 web-scale 训练数据集”？

## 0. 本讲资料边界与第二轮精修口径

按照 `WRITING_PLAN.md` 的要求，本讲精修前核对了 Common Crawl 的 WARC / WAT / WET 数据形态、IETF RFC 9309 的 Robots Exclusion Protocol、T5 / C4、The Pile、RefinedWeb、FineWeb、Dolma 和 DataComp-LM / DCLM 等公开资料。

本讲聚焦大模型训练数据采集的工程闭环：

```text
数据源规划 -> 合规审查 -> 公开或授权采集 -> 原始存储 -> 解析 -> 过滤 -> 去重 -> 配比 -> 版本审计
```

本讲不提供绕过登录、破解访问控制、规避反爬、批量抓取未授权内容或违反网站条款的做法。法律、版权和隐私问题在真实项目中需要法务、安全、隐私和业务团队共同确认；面试中也不应把技术可访问性说成“天然可用于训练”。

## 1. 来龙去脉：为什么 Web 成了大模型的数据源

### 1.1 早期 NLP 的数据规模很小

早期 NLP 任务常用人工标注数据集。

例如：

1. 情感分类数据集。
2. 机器翻译平行语料。
3. 问答数据集。
4. 命名实体识别数据集。
5. 文本分类 benchmark。

这些数据集质量高，但规模有限。

它们适合训练专项模型，不适合训练通用语言模型。

### 1.2 预训练改变了数据需求

预训练语言模型不再只学一个任务。

它要从海量文本中学习：

1. 语言结构。
2. 世界知识。
3. 常识。
4. 代码模式。
5. 多语言表达。
6. 文档格式。
7. 推理模式。

这需要非常大的语料。

互联网自然成为主要来源。

### 1.3 GPT-3 之后：Web-scale 变成标配

GPT-3 展示了大规模语言模型在海量语料上训练后可以获得强 few-shot 能力。

这让业界意识到：

```text
通用模型能力来自大规模、多样化、跨领域文本。
```

但互联网数据不是天然干净。

它包含：

1. 高质量文章。
2. 论坛讨论。
3. 广告。
4. SEO 垃圾。
5. 重复页面。
6. 错误事实。
7. 有害内容。
8. 个人信息。
9. 版权内容。

所以 Web-scale 采集之后，真正难的是筛选和治理。

### 1.4 The Pile 的启发

The Pile 是一个由多种高质量子集构成的大规模英文文本语料，强调多样性和高质量来源。

它的重要启发不是“照搬某个数据集”，而是：

1. 数据来源需要分类型组织。
2. 不同子集代表不同能力和领域。
3. 数据集构建过程需要文档化。
4. 数据本身也需要分析潜在风险。
5. 单一 web crawl 不足以代表全部高质量知识。

### 1.5 今天的观点

今天更成熟的观点是：

```text
Web-scale 数据采集只是起点，真正的壁垒是合法来源、质量筛选、覆盖设计、合规治理、版本复现和持续评估。
```

## 2. 小白例子：做一座城市图书馆

构建大模型数据集像建一座城市图书馆。

你不能只把所有纸都搬进来。

你需要决定：

1. 哪些书可以收。
2. 哪些书版权允许。
3. 哪些书质量高。
4. 哪些书重复。
5. 哪些书有隐私信息。
6. 哪些书适合儿童区。
7. 哪些书需要标注来源。
8. 哪些书过时。
9. 哪些书属于专业馆藏。

Web-scale 数据也是这样。

采集只是“搬书”。

数据工程是“建馆、分类、审查、维护和更新”。

## 3. 常见数据来源

### 3.1 网页数据

网页是最大的数据来源之一。

优点：

1. 规模大。
2. 领域广。
3. 更新快。
4. 多语言丰富。
5. 包含真实用户表达。

缺点：

1. 噪声大。
2. 重复多。
3. 广告和导航多。
4. SEO 内容多。
5. 版权和许可复杂。
6. PII 风险高。

### 3.2 书籍

书籍通常质量高、结构完整。

适合学习：

1. 长文结构。
2. 系统知识。
3. 叙事能力。
4. 专业表达。

风险：

1. 版权复杂。
2. 获取渠道限制。
3. 数据格式多样。
4. OCR 错误。

### 3.3 论文和技术文档

优点：

1. 专业性强。
2. 信息密度高。
3. 适合科学和技术能力。
4. 结构规范。

风险：

1. 公式、图表、引用解析困难。
2. PDF 解析噪声。
3. 许可证差异。
4. 学术文本和普通用户语言风格不同。

### 3.4 代码数据

代码数据是模型编程能力的基础。

来源包括：

1. 开源仓库。
2. 文档示例。
3. 教程。
4. 问答网站。
5. 单元测试。

关键问题：

1. 许可证。
2. 重复和 fork。
3. 生成代码质量。
4. 安全漏洞。
5. 密钥泄露。
6. 多文件上下文。

### 3.5 论坛和问答

优点：

1. 问题真实。
2. 对话自然。
3. 包含 debug 和经验。
4. 覆盖长尾问题。

风险：

1. 答案质量不稳定。
2. 可能过时。
3. 可能包含攻击性语言。
4. 隐私风险。
5. 平台条款复杂。

### 3.6 对话数据

对话数据对助手能力很重要。

来源可能包括：

1. 人工标注对话。
2. 用户日志。
3. 合成对话。
4. 客服记录。
5. 多轮任务数据。

风险：

1. 隐私敏感。
2. 用户同意问题。
3. 标注成本高。
4. 场景偏差。
5. 安全边界复杂。

### 3.7 多语言数据

多语言数据决定模型国际化能力。

难点：

1. 高质量低资源语言数据少。
2. 语言识别难。
3. 不同语言数据质量差异大。
4. 翻译数据可能带来翻译腔。
5. 文化和地区偏差。

## 4. Web-scale 采集 Pipeline

一个合规 pipeline 可以分成十步。

### 4.1 Source Planning

先定义目标。

问题：

1. 训练什么模型？
2. 目标语言是什么？
3. 目标领域是什么？
4. 是否需要代码、数学、科学、多模态？
5. 是否允许商业使用？
6. 是否需要可公开发布？

没有目标，采集会变成无差别堆数据。

### 4.2 Legal and Policy Review

采集前先审查：

1. 数据许可。
2. robots 协议。
3. Terms of Service。
4. 版权风险。
5. 隐私风险。
6. 数据使用目的。
7. 地区合规要求。

### 4.3 Collection

合法合规地获取数据。

方式包括：

1. 使用公开许可数据集。
2. 使用数据提供方 API。
3. 使用授权数据。
4. 使用组织内部有权使用的数据。
5. 使用公开 crawl dump。

本书不讨论绕过限制或规避反爬的方法。

### 4.4 Raw Storage

原始数据要保存元信息。

包括：

1. URL 或来源。
2. 抓取时间。
3. 许可信息。
4. 内容哈希。
5. MIME type。
6. 语言。
7. 数据版本。

### 4.5 Parsing

把 HTML、PDF、代码仓库、Markdown、论坛页面解析成结构化文本。

要保留：

1. 正文。
2. 标题。
3. 层级结构。
4. 代码块。
5. 表格。
6. 元数据。

### 4.6 Boilerplate Removal

去掉：

1. 导航栏。
2. 页脚。
3. Cookie banner。
4. 广告。
5. 推荐链接。
6. 社交按钮。
7. 重复模板。

### 4.7 Language and Domain Classification

识别：

1. 语言。
2. 领域。
3. 文档类型。
4. 代码语言。
5. 安全风险类别。

这些标签后续用于配比和过滤。

### 4.8 Quality Filtering

过滤：

1. 乱码。
2. 低信息密度。
3. 重复。
4. 垃圾内容。
5. 自动生成 SEO 内容。
6. 恶意或有害内容。
7. 隐私信息。

### 4.9 Deduplication

去重包括：

1. Exact dedup。
2. Near dedup。
3. Document-level dedup。
4. Paragraph-level dedup。
5. Code clone dedup。

### 4.10 Versioning and Audit

每次处理都要记录版本。

包括：

1. 原始数据版本。
2. 过滤规则版本。
3. 去重版本。
4. 质量模型版本。
5. 最终训练集版本。
6. 样本级 provenance。

### 4.11 关键公式与采集审计指标

把 web-scale 采集看成一个数据流，而不是一个爬虫脚本。

设候选数据源集合为：

```math
\mathcal{S}=\{s_1,s_2,\ldots,s_M\}
```

每个数据源可以记录成：

```math
s_k=(a_k,l_k,r_k,p_k,w_k)
```

其中 `a_k` 是访问方式，`l_k` 是 license / ToS 状态，`r_k` 是 robots 或访问策略，`p_k` 是隐私风险，`w_k` 是目标配比权重。

对第 `i` 个原始样本，采集准入门禁可以抽象为：

```math
A_i=g_{\mathrm{license}}(i)\,g_{\mathrm{tos}}(i)\,g_{\mathrm{robots}}(i)\,g_{\mathrm{privacy}}(i)\,g_{\mathrm{access}}(i)
```

这里每个 `g` 都是 0/1 检查项。任意一项为 0，样本就不应进入训练数据候选集。它不是法律结论，而是工程系统中必须显式记录的审计状态。

设原始样本集合为 `D_raw`，解析后的集合为 `D_parse`，通过质量、隐私、安全、污染和去重后的集合为 `D_keep`。按 token 数计算的保留率为：

```math
R_{\mathrm{keep}}=\frac{\sum_{x_i \in D_{\mathrm{keep}}} T_i}{\sum_{x_i \in D_{\mathrm{raw}}} T_i}
```

其中 `T_i` 是样本 `x_i` 的 token 数。这个指标帮助你判断过滤是否过松或过严。

对某个语言、领域或来源分组 `c`，最终配比为：

```math
m_c=\frac{\sum_{x_i \in D_{\mathrm{keep}},\,c_i=c} T_i}{\sum_{x_i \in D_{\mathrm{keep}}} T_i}
```

如果目标覆盖集合是 `C_target`，可以定义覆盖率：

```math
C_{\mathrm{cover}}=\frac{1}{|C_{\mathrm{target}}|}\sum_{c \in C_{\mathrm{target}}} I[T_c \ge \tau_c]
```

其中 `T_c` 是分组 `c` 的保留 token 数，`\tau_c` 是最低覆盖阈值。这个公式适合说明低资源语言、代码、数学、科学和安全数据不能只靠自然网页比例决定。

去重率可以写成：

```math
R_{\mathrm{dup}}=\frac{|D_{\mathrm{parse}}|-|D_{\mathrm{dedup}}|}{|D_{\mathrm{parse}}|}
```

PII 或密钥风险率可以写成：

```math
R_{\mathrm{risk}}=\frac{1}{|D_{\mathrm{parse}}|}\sum_i I_i
```

其中 `I_i=1` 表示样本命中隐私、密钥、评估污染或高风险安全规则。

样本级 provenance 至少应包含：

```math
p_i=(\mathrm{source}_i,\mathrm{crawl}_i,\mathrm{urlhash}_i,\mathrm{hash}_i,\mathrm{license}_i,t_i,v_i)
```

也就是来源、crawl 批次、URL hash、内容 hash、许可状态、采集时间和数据版本。没有 provenance，后续删除请求、污染排查、风险回溯和模型版本追责都会很困难。

一个最小 web 数据采集门禁可以写成：

```math
G_{\mathrm{web}}=
g_{\mathrm{source}}\,
g_{\mathrm{policy}}\,
g_{\mathrm{parse}}\,
g_{\mathrm{quality}}\,
g_{\mathrm{pii}}\,
g_{\mathrm{contam}}\,
g_{\mathrm{dedup}}\,
g_{\mathrm{version}}
```

这些检查项任意一个缺失，都说明项目还只是“拿到一些文本”，不能算完成了可审计的训练数据采集 pipeline。

## 5. 数据采集中的合规问题

### 5.1 版权

训练数据可能涉及版权。

需要关注：

1. 数据是否有明确许可。
2. 许可是否允许训练。
3. 是否允许商业使用。
4. 是否允许再分发。
5. 是否需要署名。
6. 是否支持删除请求。

不同地区法律和判例可能不同。
面试中不要给法律结论，而要说明需要法务和合规审查。

### 5.2 隐私

数据中可能包含 PII。

需要处理：

1. 邮箱。
2. 电话。
3. 地址。
4. 证件号。
5. 医疗信息。
6. 私密聊天。
7. 密钥和 token。

### 5.3 Terms of Service

网站条款可能限制抓取和训练使用。

工程团队不能只从技术可行性判断。

### 5.4 Robots 和访问控制

Robots 和访问控制体现网站对爬虫行为的限制。

合规数据采集要尊重这些边界。

### 5.5 数据可追溯

如果未来需要删除某类数据，必须知道它进入了哪个数据版本和哪个模型版本。

## 6. 采集不是越多越好

### 6.1 噪声成本

低质量数据会消耗训练 compute。

### 6.2 记忆风险

重复数据和稀有敏感数据增加 memorization 风险。

### 6.3 偏见放大

互联网分布不等于真实世界分布。

某些群体、语言和观点可能被过度或不足表示。

### 6.4 评估污染

网页中可能包含 benchmark 题目和答案。

如果进入训练集，会让评估失真。

### 6.5 安全风险

未过滤的网络数据可能包含有害内容、恶意代码、危险指导和注入文本。

## 7. 数据源元数据设计

每个样本最好带元数据。

常见字段：

1. source_id。
2. source_type。
3. license。
4. url_hash。
5. crawl_time。
6. language。
7. domain。
8. quality_score。
9. safety_score。
10. pii_flag。
11. dedup_cluster_id。
12. dataset_version。

元数据的价值：

1. 配比。
2. 过滤。
3. 回溯。
4. 删除。
5. 审计。
6. 数据 attribution。

## 8. Web 数据和专门数据的平衡

Web 数据覆盖广，但质量参差不齐。

专门数据质量高，但覆盖有限。

常见组合：

1. Web crawl 提供广覆盖。
2. 书籍和论文提供长结构和专业知识。
3. 代码数据提供编程能力。
4. 数学数据提供推理能力。
5. 对话数据提供助手风格。
6. 安全数据提供边界。
7. 合成数据补足稀缺任务。

关键不是二选一，而是配比和治理。

## 9. 多语言采集

### 9.1 低资源语言问题

低资源语言数据少，且质量更不稳定。

如果只按互联网自然比例采样，模型会偏向高资源语言。

### 9.2 语言识别

语言识别在短文本、混合语言、代码混杂和低资源语言上容易出错。

### 9.3 翻译数据

机器翻译可以扩展数据，但会带来：

1. 翻译腔。
2. 文化信息丢失。
3. 错误传播。
4. 风格单一。

### 9.4 多语言配比

需要考虑：

1. 用户分布。
2. 目标市场。
3. 语言资源稀缺度。
4. 任务重要性。
5. 模型容量。

## 10. 代码数据采集的特殊问题

### 10.1 许可证

开源不等于无限制使用。

不同 license 对训练、分发和商业使用的解释需要法务判断。

### 10.2 Fork 和重复

代码仓库大量 fork 和复制。

不去重会让热门代码过度影响模型。

### 10.3 密钥和凭证

代码中可能包含泄露密钥、token、密码和内部 URL。

必须扫描和过滤。

### 10.4 安全漏洞

训练不安全代码可能让模型生成漏洞模式。

需要结合静态分析、安全标签和高质量代码筛选。

### 10.5 Repo-level 上下文

函数片段不等于真实工程代码。

如果要训练 coding agent，需要保留目录结构、依赖、测试和 commit 信息。

## 11. 对话和用户数据采集

用户数据最敏感。

必须考虑：

1. 用户同意。
2. 用途说明。
3. 脱敏。
4. 数据保留期限。
5. 删除请求。
6. 人工审核权限。
7. 是否允许用于训练。

真实产品中，用户日志可以帮助发现问题，但不能默认无限制进入训练。

## 12. 真实项目架构

一个采集系统可以分成：

1. Source registry。
2. Legal review workflow。
3. Collector。
4. Raw data lake。
5. Parser。
6. Metadata extractor。
7. Quality filter。
8. PII and safety scanner。
9. Dedup service。
10. Dataset builder。
11. Version registry。
12. Audit dashboard。

### 12.1 Source registry

记录每个数据源的合法性、用途、联系人、许可和风险等级。

### 12.2 Raw data lake

保存原始数据和 hash，方便复现和审计。

### 12.3 Dataset builder

根据训练目标和配比规则构建最终数据集。

### 12.4 Audit dashboard

展示：

1. 数据来源占比。
2. 语言占比。
3. 领域占比。
4. PII 命中率。
5. 去重率。
6. 质量分布。
7. 许可证分布。

### 12.5 最小可运行 Web 采集审计 demo

下面这个 demo 不联网、不读取真实网页，也不需要第三方库。它用内存里的 toy HTML 模拟一个合规采集 pipeline，覆盖 source policy、HTML 解析、质量过滤、PII / 密钥扫描、评估污染扫描、exact dedup、元数据和配比统计。

它演示的不是“怎么爬网站”，而是“采集系统上线前应该检查什么”。

```python
import hashlib
import re
from collections import Counter, defaultdict


sources = {
    "tech_blog": {"license": "cc-by", "tos_train": True, "robots_allowed": True, "access": "public"},
    "oss_docs": {"license": "apache-2.0", "tos_train": True, "robots_allowed": True, "access": "public"},
    "science_preprint": {"license": "cc-by", "tos_train": True, "robots_allowed": True, "access": "public"},
    "zh_news": {"license": "authorized", "tos_train": True, "robots_allowed": True, "access": "public"},
    "private_forum": {"license": "unknown", "tos_train": False, "robots_allowed": False, "access": "login_required"},
}

documents = [
    {
        "id": "blog_attention",
        "source_id": "tech_blog",
        "url": "https://example.org/blog/attention",
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
        "url": "https://docs.example.org/vector-db",
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
        "id": "blog_attention_dup",
        "source_id": "tech_blog",
        "url": "https://mirror.example.org/blog/attention-copy",
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
        "id": "forum_private",
        "source_id": "private_forum",
        "url": "https://forum.example.org/private/thread/7",
        "crawl_time": "2026-06-01T00:15:00Z",
        "mime": "text/html",
        "language": "en",
        "domain": "forum",
        "html": "<article>Private member discussion with personal project details.</article>",
    },
    {
        "id": "spam_seo",
        "source_id": "tech_blog",
        "url": "https://example.org/seo/spam",
        "crawl_time": "2026-06-01T00:20:00Z",
        "mime": "text/html",
        "language": "en",
        "domain": "spam",
        "html": "<body>Buy now!!! promo promo promo $$$ click click click</body>",
    },
    {
        "id": "pii_secret",
        "source_id": "tech_blog",
        "url": "https://example.org/leak",
        "crawl_time": "2026-06-01T00:25:00Z",
        "mime": "text/html",
        "language": "en",
        "domain": "security",
        "html": "<article>Contact jane@example.com and use api key sk-live-abcdef for tests.</article>",
    },
    {
        "id": "eval_leak",
        "source_id": "tech_blog",
        "url": "https://example.org/benchmark/answer",
        "crawl_time": "2026-06-01T00:30:00Z",
        "mime": "text/html",
        "language": "en",
        "domain": "eval",
        "html": "<article>Benchmark answer key: GSM8K solution and hidden test answer.</article>",
    },
    {
        "id": "paper_scaling",
        "source_id": "science_preprint",
        "url": "https://papers.example.org/scaling",
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
        "url": "https://news.example.cn/data-quality",
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

REQUIRED_META = ["id", "source_id", "url", "crawl_time", "mime", "language", "domain"]
PII_OR_SECRET = [re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), re.compile(r"sk-[A-Za-z0-9-]{8,}")]
CONTAMINATION = ["benchmark answer", "gsm8k solution", "hidden test answer"]


def strip_html(html):
    html = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def content_hash(text):
    normalized = " ".join(tokenize(text))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def quality_score(text):
    toks = tokenize(text)
    if not toks:
        return 0.0
    unique_ratio = len(set(toks)) / len(toks)
    alpha_ratio = sum(ch.isalpha() for ch in text) / max(len(text), 1)
    length_score = min(len(toks) / 24, 1.0)
    return round(0.45 * length_score + 0.35 * unique_ratio + 0.20 * min(alpha_ratio / 0.65, 1.0), 3)


def policy_allowed(doc):
    source = sources[doc["source_id"]]
    return (
        source["license"] not in {"unknown", "restricted"}
        and source["tos_train"]
        and source["robots_allowed"]
        and source["access"] == "public"
    )


def has_pii_or_secret(text):
    return any(pattern.search(text) for pattern in PII_OR_SECRET)


def has_eval_contamination(text):
    lower = text.lower()
    return any(term in lower for term in CONTAMINATION)


def sum_tokens(items, field):
    sums = defaultdict(int)
    for item in items:
        sums[item[field]] += item["tokens"]
    return sums


def audit_web_collection(items):
    seen_hashes = set()
    kept = []
    rejected = {}
    stage_counts = Counter(raw=len(items))

    for doc in items:
        missing = [field for field in REQUIRED_META if not doc.get(field)]
        if missing:
            rejected[doc["id"]] = "missing_metadata"
            continue

        text = strip_html(doc["html"])
        doc_hash = content_hash(text)
        score = quality_score(text)

        if not policy_allowed(doc):
            rejected[doc["id"]] = "policy_block"
            continue
        stage_counts["policy_pass"] += 1

        if has_pii_or_secret(text):
            rejected[doc["id"]] = "pii_or_secret"
            continue
        stage_counts["pii_pass"] += 1

        if has_eval_contamination(text):
            rejected[doc["id"]] = "eval_contamination"
            continue
        stage_counts["contamination_pass"] += 1

        if score < 0.62:
            rejected[doc["id"]] = "low_quality"
            continue
        stage_counts["quality_pass"] += 1

        if doc_hash in seen_hashes:
            rejected[doc["id"]] = "exact_duplicate"
            continue
        seen_hashes.add(doc_hash)
        kept.append({**doc, "text": text, "hash": doc_hash, "tokens": len(tokenize(text)), "quality": score})
        stage_counts["dedup_pass"] += 1

    raw_tokens = sum(len(tokenize(strip_html(doc["html"]))) for doc in items)
    kept_tokens = sum(doc["tokens"] for doc in kept)
    language_mix = {lang: round(count / kept_tokens, 3) for lang, count in sorted(sum_tokens(kept, "language").items())}
    domain_mix = {dom: round(count / kept_tokens, 3) for dom, count in sorted(sum_tokens(kept, "domain").items())}
    gates = {
        "policy": rejected.get("forum_private") == "policy_block",
        "pii": rejected.get("pii_secret") == "pii_or_secret",
        "contamination": rejected.get("eval_leak") == "eval_contamination",
        "quality": rejected.get("spam_seo") == "low_quality",
        "dedup": rejected.get("blog_attention_dup") == "exact_duplicate",
        "provenance": all(doc.get("hash") and doc.get("url") and doc.get("crawl_time") for doc in kept),
    }

    return {
        "kept_ids": [doc["id"] for doc in kept],
        "rejected": dict(sorted(rejected.items())),
        "stage_counts": dict(stage_counts),
        "retention": round(kept_tokens / max(raw_tokens, 1), 3),
        "language_mix": language_mix,
        "domain_mix": domain_mix,
        "avg_quality": round(sum(doc["quality"] for doc in kept) / max(len(kept), 1), 3),
        "gates": gates,
        "gate_pass": all(gates.values()),
    }


report = audit_web_collection(documents)
print("kept_ids=", report["kept_ids"])
print("rejected=", report["rejected"])
print("stage_counts=", report["stage_counts"])
print("retention=", report["retention"])
print("language_mix=", report["language_mix"])
print("domain_mix=", report["domain_mix"])
print("avg_quality=", report["avg_quality"])
print("gates=", report["gates"])
print("gate_pass=", report["gate_pass"])

assert report["kept_ids"] == ["blog_attention", "oss_vector_db", "paper_scaling", "zh_data_quality"]
assert report["rejected"] == {
    "blog_attention_dup": "exact_duplicate",
    "eval_leak": "eval_contamination",
    "forum_private": "policy_block",
    "pii_secret": "pii_or_secret",
    "spam_seo": "low_quality",
}
assert report["stage_counts"] == {
    "raw": 9,
    "policy_pass": 8,
    "pii_pass": 7,
    "contamination_pass": 6,
    "quality_pass": 5,
    "dedup_pass": 4,
}
assert report["retention"] == 0.639
assert report["language_mix"] == {"en": 0.537, "zh": 0.463}
assert report["domain_mix"] == {"code_docs": 0.148, "science": 0.167, "web_ml": 0.222, "zh_web": 0.463}
assert report["gate_pass"] is True
```

预期输出类似：

```text
kept_ids= ['blog_attention', 'oss_vector_db', 'paper_scaling', 'zh_data_quality']
rejected= {'blog_attention_dup': 'exact_duplicate', 'eval_leak': 'eval_contamination', 'forum_private': 'policy_block', 'pii_secret': 'pii_or_secret', 'spam_seo': 'low_quality'}
stage_counts= {'raw': 9, 'policy_pass': 8, 'pii_pass': 7, 'contamination_pass': 6, 'quality_pass': 5, 'dedup_pass': 4}
retention= 0.639
language_mix= {'en': 0.537, 'zh': 0.463}
domain_mix= {'code_docs': 0.148, 'science': 0.167, 'web_ml': 0.222, 'zh_web': 0.463}
avg_quality= 0.922
gates= {'policy': True, 'pii': True, 'contamination': True, 'quality': True, 'dedup': True, 'provenance': True}
gate_pass= True
```

这个 demo 的关键不是分数阈值本身，而是工程习惯：每个样本都要有来源、许可、时间、内容 hash、拒绝原因和最终版本。真正的 web-scale pipeline 只是把这里的 toy 规则替换成更强的解析器、质量模型、PII 检测器、污染检测器、near-dedup 和数据版本系统。

## 13. 面向专家：采集策略会改变模型行为

采集策略不是中性的。

它会影响模型：

1. 语言风格。
2. 世界知识。
3. 价值观分布。
4. 安全边界。
5. 代码习惯。
6. 多语言能力。
7. 长文本能力。

例如：

1. 论坛数据多，模型更口语化。
2. 论文数据多，模型更学术化。
3. 代码数据多，模型更擅长编程。
4. 低质量网页多，模型更容易胡说。

因此，数据采集和模型行为是强耦合的。

## 14. 面试官会怎么问

### 问题 1：如何构建 web-scale 预训练数据集？

回答要点：

1. 先定义模型目标和数据需求。
2. 建立合法数据源 registry。
3. 采集公开许可、授权或合规来源。
4. 做解析、语言识别、boilerplate removal。
5. 做质量过滤、PII、安全扫描、去重。
6. 做领域分类和数据配比。
7. 做版本管理和审计。

标准回答：

```text
我会先定义模型目标，例如通用、多语言、代码还是领域模型。然后建立 source registry，记录每个数据源的许可、用途、风险和版本。采集后进入 raw data lake，保留来源和 hash。处理阶段包括解析、正文抽取、语言识别、去 boilerplate、质量过滤、PII 和密钥扫描、安全过滤、去重、领域分类。最后根据目标能力设计 mixture，构建可复现的数据版本，并记录元数据用于审计、删除和 attribution。
```

### 问题 2：网页数据有哪些风险？

回答要点：

1. 噪声。
2. 重复。
3. 广告和 SEO。
4. 版权。
5. PII。
6. 有害内容。
7. 评估污染。
8. 分布偏差。

### 问题 3：为什么要保留数据元数据？

回答要点：

1. 配比。
2. 过滤。
3. 版本复现。
4. 删除请求。
5. 合规审计。
6. 数据 attribution。
7. 问题回溯。

### 问题 4：如何处理版权和合规？

回答要点：

1. 数据源许可审查。
2. Terms of Service。
3. robots 和访问控制。
4. 商业使用限制。
5. 数据删除机制。
6. 法务和合规参与。
7. 样本级 provenance。

### 问题 5：The Pile 对数据工程有什么启发？

回答要点：

1. 多来源高质量数据比单一 web crawl 更有价值。
2. 数据子集要按来源和领域组织。
3. 数据集需要文档化和风险分析。
4. 数据多样性影响跨领域泛化。

## 15. 标准回答模板

面试中可以这样回答：

```text
Web-scale 数据采集不是简单写爬虫，而是一个从 source planning 到 governance 的完整 pipeline。首先要明确模型目标和数据需求，然后建立合规的数据源登记，包括许可、用途、robots、ToS 和风险等级。采集后保留 raw data、hash、URL、时间和 license 等元数据。处理阶段包括解析、正文抽取、语言识别、去 boilerplate、质量评分、PII 和密钥扫描、安全过滤、去重和领域分类。最后根据模型目标设计 mixture，并做 dataset versioning、provenance、审计和删除请求支持。

我会特别关注四类风险：版权合规、隐私泄露、低质量噪声和评估污染。因为这些问题最后都会变成模型幻觉、安全问题、评估失真或上线风险。
```

## 16. 常见误区

### 16.1 误区：能访问就能训练

纠正：技术可访问不等于法律、许可和政策允许。

### 16.2 误区：网页数据天然代表真实世界

纠正：互联网分布有强偏差。

### 16.3 误区：只要数据够多，质量无所谓

纠正：低质量数据会浪费 compute，并引入错误和风险。

### 16.4 误区：去重是清洗阶段的小问题

纠正：去重影响训练效率、记忆风险和评估污染。

### 16.5 误区：用户日志可以直接拿来训练

纠正：需要同意、脱敏、保留策略、删除机制和访问控制。

## 17. 小练习

### 练习 1

设计一个 web-scale 数据采集系统架构。

要求包含：source registry、legal review、collector、raw data lake、parser、quality filter、PII scanner、dedup、dataset builder、version registry。

### 练习 2

列出网页、书籍、论文、代码、论坛、对话数据各自的优缺点。

### 练习 3

为代码数据采集设计合规和安全 checklist。

### 练习 4

解释为什么 robots、ToS、license 和 provenance 都和大模型训练相关。

### 练习 5

设计一个数据源元数据 schema。

要求至少包含 10 个字段。

## 18. 本章总结

Web-scale 数据采集是大模型训练的基础，但采集本身只是起点。

大模型数据来源包括网页、书籍、论文、代码、论坛、对话、多语言和多模态数据。

不同来源有不同质量、覆盖、许可、隐私和安全风险。

一个成熟 pipeline 应覆盖 source planning、legal review、collection、raw storage、parsing、boilerplate removal、language/domain classification、quality filtering、deduplication、versioning 和 audit。

采集策略会直接影响模型行为、能力、偏见、安全和合规风险。

面试中要把 Web-scale 数据采集讲成合法、可追溯、可治理、可评估的数据系统，而不是简单爬虫工程。
