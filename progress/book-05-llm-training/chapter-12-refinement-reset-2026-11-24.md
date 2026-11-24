# 第十二章精修记录（2026-11-24）

本记录根据本轮实际章节审阅新增；章节先后按 `book-05-llm-training/chapters/` 中的文件顺序推进，未使用既有 progress 记录决定范围。

## 修改文件

- 正文：`book-05-llm-training/chapters/12-多模态训练.md`
- 研究证据：`research/multimodal-training-refinement-evidence-2026-11-24.md`

## 精修内容

- 将 Qwen3-VL 的平方根归一化逐 token loss 说明改为贴近技术报告的表述，并保留完整权重定义未公开的边界。
- 明确 Qwen3.5-Omni 分项 token 预算合计 4.29T、报告总量概括约 4T 的口径差异，不自行调平。
- 改写 Frame Pack 的 token 与 attention 预算解释，明确 token 增至 8 倍时普通 full attention 的 score cell 量级约增至 64 倍。
- 将 CogVideoX 的 timestep 消融限定为论文设置，并准确描述 40K steps 时列出的五个验证时间点。
- 区分 Min-SNR-5 的 ViT-B/FID 10 收敛速度实验与 ViT-XL/$x_0$/约 7M iteration 的 FID 2.06 结果。

## 验收

- 对照第十二章契约核对章节骨架，未发现缺少的主体标题。
- Markdown 静态检查：围栏配对正常、行内 `$` 分隔符为偶数、本地链接目标缺失 0、尾随空白 0。
- `git diff --check -- book-05-llm-training/chapters/12-多模态训练.md research/multimodal-training-refinement-evidence-2026-11-24.md` 通过。
- 未运行代码示例、模型训练、测试或 PDF 构建。

## 待续

本轮第五册精修继续按实际章节文件顺序处理第十三章及后续章节；各章事实修改另按需要补充一手研究证据。
