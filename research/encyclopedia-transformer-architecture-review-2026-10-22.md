# 第四册 Transformer 架构精修核验（2026-10-22）

## 范围与修订

全文复读 4.1—4.13 与小结，按百科定义、机制、反例与专题入口验收。修正 next-token 对齐方向、共享输出头的 embedding 更新、RoPE 位置符号、dropout 后权重行和、mask 布尔含义与非方形 causal 对齐；补 CPU attention 隔离/缓存示例、线性 attention 分母、FFN 向量方向、具体 gated delta 纠错与标量例子；限定 KV 复用、TTFT 和缓存内存口径。

## 来源核验与纠错

发现原 Kimi Linear 链接 `2510.26620` 实为 *Toward Automated Security Risk Detection in Large Software Using Call Graph Analysis*。错误链接虽 HTTP200，不能支持正文；参照已有 `research/model-update-2026-09/kda-source-notes.md` 定位候选后，重新读取摘要、核对元数据并修正为 [2510.26692](https://arxiv.org/abs/2510.26692)。摘要支持 KDA 为 Gated DeltaNet 的细粒度门扩展、与 MLA 混合；本次不引入其性能数字，也不将其移植为其他 Kimi 模型事实。原 HTML 候选失败是错论文，不是相关原文验证。

实际读取：

- [PyTorch 2.12 SDPA](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html)：True 允许、非方形 is_causal 左上对齐、dropout_p 需显式关闭、GQA repeat_interleave 参考结构。版本文档非源码 commit。
- [RoFormer v5 HTML](https://arxiv.org/html/2104.09864v5)：二维旋转和相对相位相关公式，非全文实验复现。
- [Gated Delta Networks v1 HTML](https://arxiv.org/html/2412.06464v1)：状态 `[d_v,d_k]`、delta rule 与 gated delta recurrence 相关公式；正文按该方向定义，不冒充 KDA 逐维门。
- [Performers v4 HTML](https://arxiv.org/html/2009.14794v4)：kernel attention 重排及 D 归一化项，支持区分分子和分母。
- Kimi Linear 当前摘要，如上所述；未阅读全文或复现训练。

通过 arXiv Atom 两次请求核对 13 篇题名和版本：Attention1706.03762v7、RoFormer2104.09864v5、ALiBi2108.12409v2、MQA1911.02150v1、GQA2305.13245v3、FlashAttention2205.14135v2、FlashAttention2 2307.08691v1、Longformer2004.05150v2、Performers2009.14794v4、Lost in the Middle2307.03172v3、RetNet2307.08621v4、Gated Delta2412.06464v3、Kimi Linear2510.26692v2。Atom 当前版本与读取 HTML 的版本分别记录，不混称。第一次 API 默认仅返回10项，第二次补足3项。

原15个唯一来源均HTTP200，但包含上述错论文；替换后的正确Kimi链接独立检查HTTP200，因此最终15个入口可达。最初沙箱脚本无法连接代理，获准重试后完成。可达和题名核对不等于其他论文全文阅读。

## 本地验证

临时脚本 `/tmp/check_architecture_review.py` 在 `.tmp/llm-book-cpu-check`、PyTorch2.12.0+cpu 执行，无新依赖或CUDA安装。

- 五个Python块各自编译和独立执行通过，四块标准库、一块PyTorch。
- 未来 K/V 改动不影响较早输出，padding key 改动不影响允许的输出；预生成 K/V 切片逐位置 attention 与全量 attention 一致，误差舍入12位为0。不是完整模型缓存验证。
- 零 Q/K、V=[1,2,6] 时单 query 显式全允许 mask 输出3，非方形 is_causal=True 输出1。
- 四query heads、两KV heads的 enable_gqa 与 repeat_interleave 结果在float64容差1e-12内一致。
- 非法mask长度、零长度mask、缓存维度、head整除关系边界通过；32层、batch4、8192token、8KV heads、128维、2字节缓存为4GiB。
- attention手算值7.01253185042974；RoPE二维旋转相对相位恒等式数值通过；标量delta连续写入2.5、2.75。
- 24个math块花括号/环境、总围栏结构通过，无operatorname等旧格式。相对链接、练习标题、下一章与本册目录目标存在，git diff --check通过。

## 配套与限制

EXERCISES.md 增加五项信息路径与状态验收；题库与术语已有对应主题，不复制摘要。没有运行完整Transformer、真实权重、训练、GPU/FlashAttention性能、分页缓存、多租户或多进程；归一化结构检查不等于外部公式渲染。PDF仍受Pandoc缺失限制。本章仅在声明范围内验收，全书及第二册遗留审计未完成。
