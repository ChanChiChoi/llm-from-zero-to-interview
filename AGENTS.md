# Repository Guidelines

## Project Structure and File Responsibilities

这是一个以 Markdown 为主的大模型学习与面试书系仓库。

- `book-01-*` 至 `book-24-*`：主书正文；每册通常包含 `简介.md`、`目录.md`、`chapters/` 和生成的 PDF。`book-llm-engineer/` 是补充篇。
- `plan.md`：项目总计划，只写总纲、总原则、第一次计划、两类常态化计划、任务队列和完成条件。不要在这里写执行日志、逐章流水账或研究事实。
- `progress.md`：第一次计划的记录，即从零开始建设全书、首轮编写、逐章执行和首轮 QA 的历史。
- `progress_v2.md`：常态化全书精修、逐章复读、公式/代码修订和全书审计的执行记录。
- `progress_v3.md`：以排行榜新进模型为锚点的新模型、新知识点及周边技术更新的执行记录；模型事实和来源证据放在 `research/`。
- `WRITING_SPEC.md`：全书写作宗旨、正文结构、公式/代码规范和质量门禁；修改正文前先遵循它。
- `research/`：模型、论文、框架和专题的来源、事实核验、证据等级与未验证项。
- `README.md`、`BOOK_SERIES.md`、`ROADMAP.md`：项目入口、书系目录和学习路线。
- `INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`ENGLISH_INTERVIEW_TEMPLATES.md`：面试题、练习、术语和英文表达；`PAPERS.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`：论文、项目路线和知识图谱。
- `scripts/` 和 `docs/`：构建脚本及其说明；`pyproject.toml`、`uv.lock`：Python 依赖与锁定版本；`.gitignore`：生成物和本地文件排除规则。

新增内容时先判断它是计划、执行记录、研究证据还是正文，写入对应文件；完成计划任务后把结果追加到相应进度文件，不回填到 `plan.md`。

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

使用 UTF-8 Markdown，保持现有中英文混排和章节编号习惯。章节文件放在对应 `book-*/chapters/`，使用 `序号-主题.md` 命名。正文的结构、公式、代码和引用以 `WRITING_SPEC.md` 为准；不要用临时标签替代解释。

## Commits and Pull Requests

提交信息沿用仓库现有的简短中文动词描述，例如 `更新 readme.md`、`细化 transformer`。每个提交聚焦一个主题。PR 应说明改动的文件和职责、内容或计划如何变化、执行过的校验命令，并在新增或移动文件时更新相关导航和链接。
