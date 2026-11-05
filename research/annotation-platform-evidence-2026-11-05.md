# 数据标注平台与标注证据

**核验日期：** 2026-11-05  
**范围：** 数据说明、人工偏好分歧与隐私框架版本状态。本文只记录来源事实、版本及适用边界，不记录书籍审阅、章节修改或任务进度。

## Data Statements for NLP

- **来源与版本：** Emily M. Bender、Batya Friedman，*Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science*，TACL 6 (2018), 587–604；[ACL Anthology 页面](https://aclanthology.org/Q18-1041/)与[论文 PDF](https://aclanthology.org/Q18-1041.pdf)。
- **实际阅读范围：** 摘要、§§1–3、§5 的长短格式方案和数据声明 schema；检索论文关于隐私与发布粒度的说明。
- **来源支持的事实：** 论文提出 data statement 作为 NLP 研发中的数据集描述实践，用于帮助读者判断结果如何泛化、软件适合怎样部署、数据可能反映哪些偏差。长格式 schema 包含语料筛选理由、语言变体、说话者背景、标注者背景、交流情境、文本特征和来源；作者也提出在已有长格式声明基础上使用精简版本。
- **证据边界：** 这是同行评审论文提出的描述框架，不是强制标准、隐私法律或字段真实性证明。作者明确指出，公开说话者和标注者信息时需要权衡隐私，必要时需取得许可并选择合适披露粒度。人口或标注员背景字段不应脱离项目目的机械收集。
- **证据等级：** 同行评审论文；支持描述维度和作者提出的实践，不证明填表会自动消除偏差或保证数据可泛化。

## 人类偏好分歧

- **来源与版本：** Michael Jq Zhang 等，*Diverging Preferences: When do Annotators Disagree and do Models Know?*，ICML 2025，PMLR 267:76193–76212；[PMLR 正式论文页](https://proceedings.mlr.press/v267/zhang25bx.html)。
- **实际阅读范围：** PMLR 论文页的摘要和书目信息；未据此摘录正文中的具体实验数值。
- **来源支持的事实：** 摘要称论文对人类偏好分歧提出覆盖十类因素、四个上位类别的分类，并指出任务欠明确、回答风格等因素在其研究中占重要位置；作者也报告标准 reward model 与 LLM judge 未能充分处理这些分歧。
- **证据边界：** 摘要层面的结论限于论文研究中的偏好数据与协议，不能外推到其他领域、标注池或分歧比例。未阅读全文，因此不记录或复述其具体分组定义、样本统计和实验效应数字。
- **证据等级：** ICML 同行评审论文；本记录仅支持将“分歧可能来自规则、风格或真实偏好差异”作为诊断候选，不支持跨项目的先验比例。

## NIST Privacy Framework 版本状态

- **来源与版本：** NIST [Privacy Framework 1.0 页面](https://www.nist.gov/privacy-framework/privacy-framework)及 [Privacy Framework 1.1 更新页](https://www.nist.gov/privacy-framework/new-projects/privacy-framework-version-11)；页面核验日期为 2026-11-05。
- **实际阅读范围：** 1.0 页面标题、版本与法律效力说明；1.1 更新页中草案、征求意见时间和发布状态。
- **来源支持的事实：** NIST 页面列出 Privacy Framework 1.0（2020 年 1 月），并说明框架不具有法律效力；截至核验日，1.1 更新页仍将最终版标为 “Coming soon”，另提供 2025 年 4 月发布、2025 年 6 月结束征求意见的 initial public draft。
- **证据边界：** 该状态会变化，描述只绑定核验日期。1.1 IPD 是草案，不应与正式定稿混称；框架本身也不替具体项目决定保留期限、授权或法律义务。
- **证据等级：** NIST 官方版本页，支持页面当时公开的发布状态，不构成法律意见。
