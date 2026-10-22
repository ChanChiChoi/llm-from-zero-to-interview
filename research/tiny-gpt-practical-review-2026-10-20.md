# 第三册第三章：从零训练小 GPT 审阅记录

## 范围与来源

- 正文：`book-03-practical-handbook/chapters/03-从零训练小gpt.md`，覆盖 3.1—3.7。
- 配套：第三册目录；`PAPERS.md` 已有 Transformer、BPE、top-p 和训练相关入口，本轮未重复复制。
- 环境：仓库 `.tmp/llm-book-cpu-check`，PyTorch 2.12.0 CPU build；未安装 CUDA。
- 通过指定代理 `http://10.24.27.134:7890` 检查 12 个唯一引用入口，最终均 HTTP 200：Attention Is All You Need、BPE 子词论文、The Curious Case of Neural Text Degeneration；PyTorch Dataset/DataLoader、Embedding、CrossEntropyLoss、Module mode、AdamW、save/load、multinomial 文档；Hugging Face fast tokenizers 与 Tokenizers quicktour。

论文入口用于对应方法和原论文实验；框架文档用于公开 API 行为。HTTP 状态不等于逐篇全文/附录复核，也不代表目标模型的训练与生成性能。

## 修订与本地验证

- 日志示例由系统 `/tmp` 改为仓库忽略的 `.tmp/mini_char_gpt_log.csv`，并确保目录存在。
- BPE 对比账本不再把 FLOPs 和参数数量直接相加；改为分别记录 attention FLOPs 与 embedding/head 参数数，并定义差值方向。
- checkpoint step 示例补足运行所需的评估间隔、配置、保存路径和事件记录；在前文模型/训练/保存函数上下文中执行，输出 `saved_steps=[0, 1, 2]`、`optimizer_updates=2`。
- 17 个 Python 块中，15 个在本地环境按所需上下文顺序执行通过；BPE toy 输出 `roundtrip_ok=True`、`shift_ok=True`，压缩比例为 `0.315`。两段 Hugging Face tokenizer 代码因环境未安装 `transformers` 且没有 GPT-2 tokenizer 缓存而未执行；正文已说明该依赖/模型文件前提。
- 41 个 `math` 围栏花括号与方括号计数平衡；全章 140 个围栏标记成对，第三册目录章节链接有效；`git diff --check` 通过。
- 未运行外部数学渲染器或 PDF 构建；全局 PDF dry-run 因环境缺少 Pandoc 失败，未安装依赖。

## 适用边界

验证说明的是本地 PyTorch 2.12.0 CPU toy 示例、教学采样器和 checkpoint 结构，不代表真实语料质量、模型泛化能力、BPE tokenizer 优劣、GPU 性能或生产训练可恢复性。小数据、手动 merge 规则和构造曲线不得外推成 benchmark 结论。
