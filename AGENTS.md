# Repository Guidelines

## Project Structure and File Responsibilities

这是一个以 Markdown 为主的大模型学习与面试书系仓库。

- `book-01-*` 至 `book-24-*`：主书正文；每册通常包含 `简介.md`、`目录.md`、`chapters/` 和生成的 PDF。`book-llm-engineer/` 是补充篇。
- `plan.md`：项目总计划，只写总纲、总原则、常态化计划、任务队列和完成条件。不要在这里写执行日志、逐章流水账或研究事实。
- `progress/`：所有书籍架构审视、逐章精修、正文或配套资料修改、修改历史、状态、遗留问题和交接的唯一存放位置。
- `WRITING_SPEC.md`：兼容旧导航的规范入口，指向 `写作规范/总纲.md`。
- `写作规范/总纲.md`：规范总索引和恢复入口；修改正文前先读总纲及当前任务对应的细则模块。
- `写作规范/`：按书册架构、章节精修、研究证据、技术呈现、维护与验收拆分的完整规范模块。
- `research/`：只记录联网搜索发现的新知识点及其来源、版本、证据等级和未验证边界；不得记录每册或每章的审阅进度、修改历史或交接。
- `README.md`、`BOOK_SERIES.md`、`ROADMAP.md`：项目入口、书系目录和学习路线。
- `INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`ENGLISH_INTERVIEW_TEMPLATES.md`：面试题、练习、术语和英文表达；`PAPERS.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`：论文、项目路线和知识图谱。
- `scripts/` 和 `docs/`：构建脚本及其说明；`pyproject.toml`、`uv.lock`：Python 依赖与锁定版本；`.gitignore`：生成物和本地文件排除规则。

新增内容时先判断它是计划、书籍进度/修改记录、联网搜索发现的新知识点及来源证据，还是正文，写入对应位置。所有书籍审阅与修改记录只写入 `progress/`；`research/` 仅存新知识点和来源证据；完成任务后不把执行日志或研究事实回填到 `plan.md`。

## Build and Validation

在仓库根目录运行：

```bash
python3 scripts/build_pdfs.py --dry-run       # 只检查 PDF 构建命令
python3 scripts/build_pdfs.py --book book-24-llm-inference-engine
python3 scripts/build_pdfs.py                 # 构建全部书籍（需 pandoc、XeLaTeX 和中文字体）
git diff --check                              # 检查空白和补丁格式
```

仓库没有统一的自动化测试套件。修改章节后应检查目录、Markdown 链接、代码围栏、公式和配套索引；涉及构建时至少先运行 `--dry-run`。

## Style and Naming

使用 UTF-8 Markdown，保持现有中英文混排和章节编号习惯。章节文件放在对应 `book-*/chapters/`，使用 `序号-主题.md` 命名。正文的结构、公式、代码和引用以 `写作规范/总纲.md` 及其对应模块为准；不要用临时标签替代解释。

每次对话上下文发生压缩或摘要化后，以及在新会话中恢复书籍审阅、正文或配套资料工作时，继续处理前必须从头到尾重新阅读 `写作规范/总纲.md`，再按当前任务读取对应的完整规范模块。不得用上下文摘要、记忆或此前读过的规范来替代本次重读；任务范围不清时读取全部相关模块。

## Commits and Pull Requests

提交信息沿用仓库现有的简短中文动词描述，例如 `更新 readme.md`、`细化 transformer`。每个提交聚焦一个主题。PR 应说明改动的文件和职责、内容或计划如何变化、执行过的校验命令，并在新增或移动文件时更新相关导航和链接。
