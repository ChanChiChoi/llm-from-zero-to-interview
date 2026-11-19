# 全书系公式源码与渲染兼容审查

启动日期：2026-11-19

## 范围

从第 1 册起检查 `book-01` 至 `book-24` 及补充篇 `book-llm-engineer` 中的 Markdown 公式问题。只检查公式定界、数学表达是否误放在行内代码、TeX/Markdown/HTML 兼容、括号/环境/矩阵语法等；只做必要的公式源码修正，并同步公式规范与本记录。按用户要求跳过书籍定位、书册/章节架构审视、章节内容精修和新知识研究。

静态扫描用于筛查候选项，命中后逐处判断；GitHub 页面实际渲染仍需在工作区版本推送后复核。全系列共有 25 个书册目录、667 个 Markdown 文件。

## 书册状态

| 顺序 | 书册目录 | Markdown 文件数 | 公式审查状态 |
| ---: | --- | ---: | --- |
| 1 | `book-01-core-30` | 11 | 进行中 |
| 2 | `book-02-advanced-100` | 16 | 待审 |
| 3 | `book-03-practical-handbook` | 34 | 待审 |
| 4 | `book-04-llm-encyclopedia` | 21 | 待审 |
| 5 | `book-05-llm-training` | 20 | 已有 2026-11-18 全册扫描与后续修正；本轮顺序复核时按新增规范确认 |
| 6 | `book-06-llm-deployment` | 20 | 待审 |
| 7 | `book-07-evaluation-experiments` | 17 | 待审 |
| 8 | `book-08-ai-safety-alignment` | 19 | 待审 |
| 9 | `book-09-data-engineering` | 14 | 待审 |
| 10 | `book-10-paper-reproduction-research` | 15 | 待审 |
| 11 | `book-11-system-design-interview` | 16 | 待审 |
| 12 | `book-12-career-interview-playbook` | 15 | 待审 |
| 13 | `book-13-math-foundations` | 12 | 待审 |
| 14 | `book-14-pytorch-deep-learning-engineering` | 12 | 待审 |
| 15 | `book-15-multimodal-generative-models` | 16 | 待审 |
| 16 | `book-16-reasoning-models` | 22 | 待审 |
| 17 | `book-17-agent-tool-use` | 17 | 待审 |
| 18 | `book-18-product-business-commercialization` | 13 | 待审 |
| 19 | `book-19-practitioner-playbook` | 17 | 待审 |
| 20 | `book-20-agent-harness-runtime` | 26 | 待审 |
| 21 | `book-21-transformer-architecture-evolution` | 98 | 待审 |
| 22 | `book-22-tool-protocol-ecosystem` | 54 | 待审 |
| 23 | `book-23-ai-infra` | 71 | 待审 |
| 24 | `book-24-llm-inference-engine` | 68 | 待审 |
| 补充篇 | `book-llm-engineer` | 23 | 待审 |

## 第 1 册

- 覆盖简介、目录和 9 个章节，共 11 个 Markdown 文件；公式扫描识别 805 组行内/块级表达式。
- 已在 7 个章节中，将 103 处紧跟全角标点的行内公式开头 `$` 改为标点后留一个普通空格；公式内部未加空格。
- 静态检查未发现未闭合美元定界符或 `math` 围栏、公式内侧空格、未配对花括号、未闭合数学环境、未匹配的 `\left`/`\right`、行内多行矩阵、`&amp;`、裸 `<字母`、连续双下标、`\operatorname`、带下划线的 `\text{}` 代码变量或未加括号的希腊字母 TeX 命令。
- 第 1 章《基础与语言模型》的公式源码检查已完成：将序列、向量、梯度更新、优化器算例、shape 和调度条件等误放在反引号中的数学内容改为行内公式；程序标识符、框架参数和配置键保留代码格式。修订后该章的公式静态扫描未发现上述语法问题。
- 第 2 章《Transformer 核心》的公式源码检查已完成：将遗漏的 shape、向量、数值算式和位置索引从裸文本或行内代码改为数学定界，移出三个行内矩阵到块级 `math`，并统一数学围栏。复核统计 783 组行内公式、25 个 `math` 块；定界、花括号、数学环境、`\left`/`\right` 和公式边界空格静态检查无标记，数学内容误放行内代码的扫描也无命中。
- 第 3 章《预训练与规模化》的公式源码检查已完成：将 token 计数、比例、FLOPs 数量级、DDP 梯度算例、张量切分矩阵、并行度和步时变量从行内代码改为数学定界；多行矩阵移入顶层 `math` 块，并统一下标大括号、文本标签和标点间距。复核统计 87 组行内公式、5 个 `math` 块；未发现美元定界、花括号、数学环境、`\left`/`\right`、下标格式或公式边界问题。行内代码扫描剩余的 `34 -> EOS`（token 目标字面示例）与 `1024**3`（Python 指数写法），均按字面代码保留。
- 第 4 章《后训练与对齐》的公式源码检查已完成：将序列、位置索引、对数函数、PPO 状态/动作变量及偏好目标中的下标从行内代码或裸文本改为数学定界；公式下标统一加大括号。复核统计 139 组行内公式、19 个 `math` 块；美元定界、花括号、环境、`left`/`ight`、下标、公式边界及兼容性扫描均无问题。剩余代码命中是 special token、token 示例 `q` 与实际索引表达式 `labels[t+1]`。
- 本地页面渲染未核验；当前静态检查不等于 GitHub 页面验收。

## 交接

继续审查第 1 册第 5 至第 9 章，再按顺序检查后续书册和补充篇；每册完成后更新状态、发现类别、改动位置和未核验的 GitHub 页面项。不得把公式审查扩展成书籍定位、架构审视或正文精修。
