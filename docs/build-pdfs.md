# 批量生成每本书的 PDF

这个项目的每本书基本都放在一个 `book-*` 目录里。仓库提供了 `scripts/build_pdfs.py`，可以自动扫描这些目录，把每本书的 Markdown 按顺序合成为 PDF，并把生成结果放回对应书目录内。

## 1. 安装依赖

在 Debian / Ubuntu 系统上，先安装 `pandoc`、`xelatex`、中文 TeX 支持和中文字体：

```bash
sudo apt update
sudo apt install pandoc texlive-xetex texlive-lang-chinese fonts-noto-cjk
```

说明：

- `pandoc` 负责把 Markdown 转成 PDF。
- `xelatex` 负责实际排版生成 PDF。
- `texlive-lang-chinese` 提供中文排版支持。
- `fonts-noto-cjk` 提供脚本默认使用的中文字体。

不建议直接安装 `texlive-full`，它很大，而这个项目生成 PDF 不需要完整 TeX Live 套件。

安装后可以检查：

```bash
pandoc --version
xelatex --version
fc-match "Noto Serif CJK SC"
```

## 2. 生成全部书籍 PDF

在仓库根目录执行：

```bash
python3 scripts/build_pdfs.py
```

脚本会扫描所有 `book-*` 目录，并在每个目录里生成一个同名 PDF。例如：

```text
book-01-core-30/book-01-core-30.pdf
book-02-advanced-100/book-02-advanced-100.pdf
book-24-llm-inference-engine/book-24-llm-inference-engine.pdf
```

## 3. 只生成某一本书

如果只想生成单本书：

```bash
python3 scripts/build_pdfs.py --book book-01-core-30
```

也可以一次指定多本：

```bash
python3 scripts/build_pdfs.py \
  --book book-01-core-30 \
  --book book-24-llm-inference-engine
```

## 4. 自定义输出文件名

默认输出文件名是 `{book}.pdf`，其中 `{book}` 会替换成目录名。

例如统一输出为 `book.pdf`：

```bash
python3 scripts/build_pdfs.py --output-name book.pdf
```

生成结果会变成：

```text
book-01-core-30/book.pdf
book-02-advanced-100/book.pdf
```

## 5. 自定义字体

脚本默认使用：

- 正文字体：`Noto Serif CJK SC`
- 中文字体：`Noto Serif CJK SC`
- 等宽字体：`Noto Sans Mono CJK SC`

如果你的机器上字体不同，可以先查字体名：

```bash
fc-list :lang=zh | head
```

然后指定：

```bash
python3 scripts/build_pdfs.py \
  --main-font "Noto Serif CJK SC" \
  --cjk-font "Noto Serif CJK SC" \
  --mono-font "Noto Sans Mono CJK SC"
```

## 6. 预览将要执行的命令

如果只想看脚本会调用哪些 `pandoc` 命令，不实际生成 PDF：

```bash
python3 scripts/build_pdfs.py --dry-run
```

## 7. 脚本如何决定章节顺序

常规书目录会按下面顺序收集文件：

1. `简介.md`
2. `目录.md`
3. `chapters/*.md`

`chapters/*.md` 会按文件名自然排序，所以 `01-...md`、`02-...md`、`10-...md` 会排在正确顺序。

对于 `book-llm-engineer/` 这种没有 `chapters/` 子目录、章节直接放在书目录下的结构，脚本会收集该目录下除 `简介.md`、`目录.md` 之外的 Markdown 文件，并按文件名排序。

## 8. 常见问题

如果提示找不到 `pandoc`：

```text
Cannot find 'pandoc'
```

重新安装：

```bash
sudo apt install pandoc
```

如果提示找不到 `xelatex`：

```text
Cannot find PDF engine 'xelatex'
```

重新安装：

```bash
sudo apt install texlive-xetex
```

如果 PDF 里中文乱码或字体缺失，确认中文字体存在：

```bash
fc-match "Noto Serif CJK SC"
```

如果没有匹配结果，安装字体：

```bash
sudo apt install fonts-noto-cjk
```

如果某一本书生成失败，可以先单独生成那一本，便于查看具体报错：

```bash
python3 scripts/build_pdfs.py --book book-24-llm-inference-engine
```
