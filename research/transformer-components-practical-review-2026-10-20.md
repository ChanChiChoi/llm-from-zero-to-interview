# 第三册第二章：Transformer 组件实战审阅记录

## 范围与来源

- 正文：`book-03-practical-handbook/chapters/02-transformer组件实战.md`，覆盖 2.1—2.7。
- 配套：第三册目录；本轮没有发现需要新增到其他题库或术语表的条目。
- 环境：仓库 `.tmp/llm-book-cpu-check`，PyTorch 2.12.0 CPU build；未安装 CUDA。
- 通过指定代理 `http://10.24.27.134:7890` 检查的 10 个链接最终均返回 HTTP 200：Vaswani et al. *Attention Is All You Need*；PyTorch `Embedding`、`scaled_dot_product_attention`、`MultiheadAttention`、`LayerNorm`、`Dropout`；RoFormer、RMSNorm、GLU Variants 和 LLaMA 论文。

PyTorch 文档用于确认其公开接口、张量与 mask 语义；论文用于对应方法定义和作者报告的架构背景。HTTP 状态和题名入口不等于逐篇全文、附录或目标模型训练 recipe 的完整审计。FlashAttention 相关文字仅陈述降低中间显存/读写而不消除标准 attention 二次计算项，没有援引 toy 代码推断目标硬件性能。

## 修订和验证

- 两个被称为 decoder block 的实现原先允许省略 `keep_mask`，从而退化成可见未来 token 的双向 attention。现在两者都在漏传时明确报错，正文要求调用者传入含因果约束的 keep mask；通用 RoPE attention 仍允许不传 mask。
- 全章 22 个 Python 块在 CPU-only 环境逐块独立运行通过；形状、不变量、权重共享参数量、SwiGLU 参数量和输出与正文描述一致。PyTorch SDPA 与手写 attention 在当前 CPU 环境中的比较通过。
- 两个 decoder block 分别通过正常 mask 路径和漏传 `keep_mask` 的负向检查。
- 68 个 `math` 围栏的花括号和方括号计数平衡；章节主体顺序为 2.1—2.7，第三册目录链接存在；`git diff --check` 通过。
- 未运行外部数学渲染器或 PDF 构建；`python3 scripts/build_pdfs.py --dry-run` 因环境缺少 Pandoc 失败，未安装依赖。

## 适用边界

验证只说明当前 PyTorch 2.12.0 CPU 环境中的局部教学实现和输入。它不代表 CUDA kernel、GPU 数值、FlashAttention 性能、真实模型训练质量、长上下文外推或生产 serving 结果。不同实现的 RoPE 维度布局和位置缓存、padding/position-id 协议需按目标模型核验。
