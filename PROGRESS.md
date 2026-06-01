# 学习进度追踪

## 当前阶段

阶段：第二轮全系列精修阶段

## 书籍完成度

1. 第一册：大纲完成；第 1-38 讲已扩写，正文第一版完成
2. 第二册：大纲完成；第 1-120 讲已扩写，正文第一版完成
3. 第三册：大纲完成；第 1-72 讲已扩写，正文第一版完成
4. 第四册：大纲完成；第 1-18 章已扩写，正文第一版完成
5. 第五册：大纲完成；第 1-14 章已扩写，正文第一版完成
6. 第六册：大纲完成；第 1-13 章已扩写，正文第一版完成
7. 第七册：大纲完成；第 1-12 章已扩写，正文第一版完成
8. 第八册：大纲完成；第 1-12 章已扩写，正文第一版完成
9. 第九册：大纲完成；第 1-12 章已扩写，正文第一版完成
10. 第十册：大纲完成；第 1-13 章已扩写，正文第一版完成
11. 第十一册：大纲完成；第 1-14 章已扩写，正文第一版完成
12. 第十二册：大纲完成；第 1-13 章已扩写，正文第一版完成
13. 第十三册：大纲完成；第 1-10 章已扩写，正文第一版完成
14. 第十四册：大纲完成；第 1-10 章已扩写，正文第一版完成
15. 第十五册：大纲完成；第 1-12 章已扩写，正文第一版完成
16. 第十六册：大纲完成；第 1-12 章已扩写，正文第一版完成
17. 第十七册：大纲完成；第 1-12 章已扩写，正文第一版完成
18. 第十八册：大纲完成；第 1-11 章已扩写，正文第一版完成
19. 第十九册：大纲完成；第 1-15 章已扩写，正文第一版完成
20. 第二十册：大纲完成；第 1-16 章已扩写，正文第一版完成
21. 第二十一册：大纲完成；第 1-12 章已扩写，正文第一版完成
22. 第二十二册：大纲完成；第 1-50 章已扩写，正文第一版完成
23. 第二十三册：大纲完成；第 1-60 章已扩写，正文第一版完成；另有早期 7 个合并版章节文件保留作历史材料或后续清理参考
24. 第二十四册：大纲完成；第 1-60 章已扩写，正文第一版完成
25. 大模型工程师面试补充章节：第 60-82 章已扩写，补充篇第一版完成

## 第二轮精修进度

1. 第一册 `book-01-core-30/chapters/01-基础与语言模型.md`：第 1-5 讲已完成首轮公式可读性修正与 demo 补充；新增批量 cross entropy、学习率对比、AdamW 解耦 weight decay 的 NumPy demo，并已用 `uv run python` 验证可运行。
2. 第一册 `book-01-core-30/chapters/02-transformer核心.md`：第 7-9 讲和第 11 讲 causal mask 相关内容已完成首轮公式可读性修正与 demo 补充；新增 Embedding/位置编码、scaled dot-product attention、attention 复杂度、mask-before-softmax 的最小 demo，并已用 `uv run --no-sync python` 验证可运行。
3. 第一册 `book-01-core-30/chapters/02-transformer核心.md`：第 10 讲 Multi-Head Attention 已完成首轮公式可读性修正与 demo 补充；将 `d_model`、`d_head` 等 shape 表达改为 GitHub Markdown 更稳的文本或简化数学符号，新增带 shape 断言和 causal mask 的最小 MHA demo，并已用 `uv run --no-sync python` 验证可运行。
4. 第一册 `book-01-core-30/chapters/02-transformer核心.md`：第 12 讲 Transformer Block 已完成首轮公式可读性修正与 demo 补充；将 Pre-LN/Post-LN、MLP 和残差公式改为 fenced math 或文本 shape 表达，新增带 shape 断言的 TinyPreLNBlock demo，用于验证 attention/MLP 输出必须和残差分支同形，并已用 `uv run --no-sync python` 验证可运行。
5. 第一册 `book-01-core-30/chapters/02-transformer核心.md`：第 13 讲 LayerNorm、RMSNorm 与残差连接已完成首轮公式可读性修正与 demo 补充；清理本讲 `\text{...}` 和 `d_{\text{model}}` 写法，改用 `z`、`d`、文本 shape 和 fenced math，新增手写 LayerNorm/RMSNorm 数值对比 demo，并已用 `uv run --no-sync python` 验证可运行。
6. 第一册 `book-01-core-30/chapters/02-transformer核心.md`：第 14 讲 RoPE 与长上下文已完成首轮公式可读性修正与 demo 补充；清理本讲 `\text{position}` 写法，将 RoPE 旋转公式改为 `R_i q_i`、`R_j k_j` 等更稳表达，新增二维旋转 demo 展示 attention score 随相对位置差变化，并已用 `uv run --no-sync python` 验证可运行。
7. 第一册 `book-01-core-30/chapters/02-transformer核心.md`：第 15 讲从零实现一个小 GPT 已完成首轮公式可读性修正与 demo 补充；将 input/label 右移公式改为文本 shape 表达，补充 causal LM input/label 构造 demo，并已用 `uv run --no-sync python` 验证可运行。
8. 第一册 `book-01-core-30/chapters/02-transformer核心.md`：完成全文件公式风险收尾检查；已清理剩余 `\text{...}`、`d_{\text{model}}`、`x_{<...}`、`y_{<...}` 等 GitHub Markdown 风险命中，并完成广告注入关键词和 `git diff --check` 检查。
9. 第一册 `book-01-core-30/chapters/03-训练对齐推理.md`：第 16 讲 LLM 预训练流程已完成首轮 demo 补充；新增文档 packing、EOS 插入、定长切块和 input/label 右移构造的最小 Python demo，并已用 `uv run --no-sync python` 验证可运行；广告关键词命中仅为数据清洗正常语境。
10. 第一册 `book-01-core-30/chapters/03-训练对齐推理.md`：第 17 讲数据质量与 Scaling Law 已完成首轮 demo 补充；新增固定 compute 下参数量与训练 token 数权衡的 FLOPs 估算 demo，展示 `C ≈ 6ND` 的直觉用途，并已用 `uv run --no-sync python` 验证可运行；广告关键词命中仅为数据质量正常语境。
11. 第一册 `book-01-core-30/chapters/03-训练对齐推理.md`：第 18 讲分布式训练基础已完成首轮 demo 补充；新增 DDP、ZeRO-1、ZeRO-2、ZeRO-3/FSDP 的训练状态显存粗略估算 demo，展示参数、梯度和 AdamW optimizer states 分片对单卡显存的影响，并已用 `uv run --no-sync python` 验证可运行。
12. 第一册 `book-01-core-30/chapters/03-训练对齐推理.md`：第 19 讲 Instruction Tuning 已完成首轮 demo 补充；新增不依赖外部 tokenizer 的 chat template 与 assistant-only label mask demo，展示 system/user token 设为 `-100`、assistant token 参与监督训练，并已用 `uv run --no-sync python` 验证可运行。
13. 第一册 `book-01-core-30/chapters/03-训练对齐推理.md`：第 20 讲 SFT 已完成首轮公式可读性修正与 demo 补充；将 SFT 条件概率和目标函数改为 fenced math，新增 `torch.nn.functional.cross_entropy(ignore_index=-100)` demo，验证只有 assistant label 位置参与 loss，并已用 `uv run --no-sync python` 验证可运行。
14. 第一册 `book-01-core-30/chapters/03-训练对齐推理.md`：第 21 讲 RLHF 已完成首轮公式可读性修正与 demo 补充；将 Reward Model pairwise ranking、pairwise loss 和 KL penalty 公式改为 GitHub Markdown 更稳的 fenced math 与简化下标，新增 `-log sigmoid(r_w-r_l)` 的最小 PyTorch demo，并已用 `uv run --no-sync python` 验证可运行。
15. 第一册 `book-01-core-30/chapters/03-训练对齐推理.md`：第 22 讲 DPO 已完成首轮公式可读性修正与 demo 补充；将 DPO 偏好概率、policy/reference log probability 差和 DPO loss 改为 fenced math 与 `\mathrm{ref}`/`\mathrm{DPO}` 稳定下标，新增最小 DPO loss PyTorch demo，并已用 `uv run --no-sync python` 验证可运行。
16. 第一册 `book-01-core-30/chapters/03-训练对齐推理.md`：第 23 讲 Reward Model 与 Reward Hacking 已完成首轮公式可读性修正与 demo 补充；将 pairwise ranking、pairwise loss 和 distribution shift 公式改为 fenced math 与稳定下标，新增代理 RM 分数和真实偏好背离的 reward hacking 玩具 demo，并已用 `uv run --no-sync python` 验证可运行。
17. 第一册 `book-01-core-30/chapters/03-训练对齐推理.md`：第 24 讲推理采样策略已完成首轮公式可读性修正与 demo 补充；将 greedy decoding 和 temperature softmax 公式改为 fenced math 与简化变量，新增 temperature、top-k、top-p 分布过滤 demo，并已用 `uv run --no-sync python` 验证可运行。
18. 第一册 `book-01-core-30/chapters/03-训练对齐推理.md`：第 25 讲 KV Cache 与推理加速已完成首轮公式可读性修正与 demo 补充；将 KV Cache 显存公式改为稳定符号 `M_kv`、`H_kv`、`D_h`、`b`，新增 MHA/GQA/MQA KV Cache 显存估算 demo，并已用 `uv run --no-sync python` 验证可运行。
19. 第一册 `book-01-core-30/chapters/03-训练对齐推理.md`：第 26 讲 FlashAttention 与显存优化已完成首轮公式可读性修正与 demo 补充；将 attention 公式、复杂度和 `T^2` 示例改为 fenced math 或文本表达，新增显式 attention scores/probs 矩阵显存随序列长度平方增长的估算 demo，并已用 `uv run --no-sync python` 验证可运行。
20. 第一册 `book-01-core-30/chapters/03-训练对齐推理.md`：第 27 讲量化与部署已完成首轮公式可读性修正与 demo 补充；将量化/反量化公式改为 `x_q`、`x_hat` 等稳定符号，将 13B 权重显存估算改为文本表达，新增 symmetric INT8 量化/反量化误差 demo，并已用 `uv run --no-sync python` 验证可运行；本文件常见公式风险模式已清理完毕。
21. 第一册 `book-01-core-30/chapters/03-训练对齐推理.md`：第 28 讲幻觉、评估与安全已完成首轮 demo 补充；新增按任务切片和错误类型汇总评估结果的最小 Python demo，用于说明不能只看总分，还要追踪事实幻觉、推理错误、工具参数错误、过度拒答等错误类型，并已用 `uv run --no-sync python` 验证可运行。
22. 第一册 `book-01-core-30/chapters/03-训练对齐推理.md`：第 29 讲开放研究问题已完成首轮结构化模板补充；新增研究卡片模板，将开放题拆成 research question、hypothesis、baseline、experiment、metrics、ablation 和 risks，帮助把研究方向转化为可验证实验计划；已完成公式风险、广告注入关键词和 `git diff --check` 检查。
23. 第一册 `book-01-core-30/chapters/03-训练对齐推理.md`：第 30 讲模拟面试与复盘已完成首轮结构化模板补充；新增 mock interview 六维打分表，覆盖 correctness、structure、trade-off、engineering、evaluation 和 communication，并给出弱点修复动作示例。
24. 第一册 `book-01-core-30/chapters/03-训练对齐推理.md`：完成全文件第二轮阶段性收尾检查；第 16-30 讲已完成公式可读性修正、demo/模板补充或结构化增强，常见公式风险模式无命中，广告关键词仅保留数据清洗正常语境，`git diff --check` 通过。
25. 第一册 `book-01-core-30/chapters/04-多模态基础.md`：第 31 讲多模态大模型概览已完成首轮 demo 补充；新增图像 patch token 与视频帧 token 成本估算 demo，用于说明多模态输入会显著增加上下文长度、显存和延迟，并已用 `uv run --no-sync python` 验证可运行。
26. 第一册 `book-01-core-30/chapters/04-多模态基础.md`：第 32 讲 CLIP 与图文对齐已完成首轮 demo 补充与验证；验证原有 CLIP batch 内对比 loss 代码可运行，新增 zero-shot classification 相似度选择 demo，展示 image embedding 与多个文本 prompt embedding 比较后选择最高相似度类别，并已用 `uv run --no-sync python` 验证可运行。
27. 第一册 `book-01-core-30/chapters/04-多模态基础.md`：第 33 讲 Vision-Language Model 已完成首轮 demo 补充；新增 connector 将 image features 从 vision hidden size 投影到 LLM hidden size，并与 text embeddings 拼接的 shape demo，验证 `[B,N,d_v] -> [B,N,d] -> [B,N+T,d]` 数据流，并已用 `uv run --no-sync python` 验证可运行。
28. 第一册 `book-01-core-30/chapters/04-多模态基础.md`：第 34 讲 Diffusion Model 与图像生成已完成首轮 demo 补充与验证；验证原有单步 forward noising 代码可运行，新增不同 `alpha_bar` 下加噪强度变化的一维向量 demo，展示 timestep 越靠后原始信号占比越低、噪声占比越高，并已用 `uv run --no-sync python` 验证可运行。

## 项目完成度

待开始。

## 论文阅读状态

待开始。

## 模拟面试记录

待开始。

## 当前薄弱点

待后续学习后记录。

## 待办项

1. 第二轮全系列精修已启动，详见 `SECOND_PASS_REVIEW_PLAN.md`。
2. 第一优先级：从第一册开始逐章遍历，全量修正数学式子和数学公式，保证 GitHub Markdown 阅读效果和变量解释正确。
3. 第二优先级：从第一册开始逐章记录并补充适合小白理解的 demo 级 Python 代码，优先使用标准库、`numpy`、`pandas`，必要时使用 `matplotlib` 或 `torch`。
4. 第三优先级：审计全书是否完整覆盖大模型相关领域，标记缺失、薄弱、过时和重复主题。
5. 第四优先级：对每个重点知识点补齐“是什么、为什么、机制、代码 demo、边界、工程实现、面试追问、专家视角”。
6. 第五优先级：第二轮每册、每个重点主题精修前必须联网检索官方论文、技术报告、官方文档、GitHub 仓库、框架文档和权威工程博客作为辅助资料。
7. 第二轮正文修改后，同步完善第四册百科、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md` 和 `KNOWLEDGE_GRAPH.md`。

## 下一步计划

24 本主书正文第一版已完成。大模型工程师面试补充章节第 60-82 章已扩写，补充篇第一版完成。第二轮全系列精修已启动，详见 `SECOND_PASS_REVIEW_PLAN.md`。下一步从第一册开始逐章执行公式全量修正，并同步记录和补充适合小白理解的 demo 级 Python 代码；所有重点主题精修前必须联网检索辅助资料。
