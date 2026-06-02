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
29. 第一册 `book-01-core-30/chapters/04-多模态基础.md`：第 35 讲语音大模型已完成首轮 demo 补充；新增 16 kHz 音频在 20 ms frame shift 下的帧数估算 demo，展示音频输入随时长线性增长，并用于解释实时语音助手的 streaming、chunk size、首包延迟和打断处理成本，已用 `uv run --no-sync python` 验证可运行。
30. 第一册 `book-01-core-30/chapters/04-多模态基础.md`：第 36 讲视频生成与 Sora 类模型基础已完成首轮 demo 补充；新增视频时空 patch token 成本估算 demo，展示 token 数随帧数和分辨率快速增长，并已用 `uv run --no-sync python` 验证可运行。
31. 第一册 `book-01-core-30/chapters/04-多模态基础.md`：完成全文件第二轮阶段性收尾检查；第 31-36 讲已完成 demo 补充或结构化增强，常见公式风险模式、广告注入关键词和无关链接均无命中，`git diff --check` 通过。
32. 第一册 `book-01-core-30/chapters/05-面试与复习.md`：第 37 讲 OpenAI 风格开放题已完成首轮结构化模板补充；新增 10 分制开放题回答评分表，覆盖目标澄清、问题拆解、实验设计、trade-off 和表达可信度，帮助将开放题练习转为可复盘标准；已完成公式风险、广告注入关键词和 `git diff --check` 检查。
33. 第一册 `book-01-core-30/chapters/05-面试与复习.md`：第 38 讲模拟面试与总复习已完成首轮结构化模板补充；新增第一册最终验收打卡表，覆盖基础、优化、Transformer、训练、分布式、后训练、推理、评估安全、多模态、开放题、项目表达和 mock interview，用于进入第二册前的自查。
34. 第一册 `book-01-core-30/chapters/01-基础与语言模型.md`：完成全文件公式风险补充收尾；清理残留 `\text{...}`、`x_{<t}` 和 `\theta_{\text{new}}` 等 GitHub Markdown 风险写法，改为文本表达或 `\mathrm{...}` 下标，保持原有语义不变。
35. 第一册 `book-01-core-30/chapters/05-面试与复习.md`：完成全文件第二轮阶段性收尾检查；第 37-38 讲已完成结构化模板补充，常见公式风险模式、广告注入关键词和无关链接均无命中，`git diff --check` 通过。
36. 第一册 `book-01-core-30/chapters/`：完成第一册第二轮阶段性整体验收；5 个章节文件常见公式风险模式均无命中，广告关键词仅保留 `03-训练对齐推理.md` 数据清洗正常语境，`git diff --check` 通过。
37. 第二册 `book-02-advanced-100/chapters/01-深度学习与概率基础进阶.md`：第 1 讲概率建模视角下的语言模型已启动第二轮精修；将联合概率、自回归链式分解和三变量链式法则公式改为 fenced math，新增条件概率乘积与 negative log likelihood 的最小 Python demo，并已用 `uv run --no-sync python` 验证可运行。
38. 第二册 `book-02-advanced-100/chapters/01-深度学习与概率基础进阶.md`：第 2 讲交叉熵、KL 散度与信息论统一视角已完成首轮公式和代码精修；将交叉熵、KL、MLE、NLL 和 perplexity 公式改为 fenced math，新增数值稳定 softmax 与 cross entropy demo，说明减最大 logit 避免 `exp` 溢出，并已用 `uv run --no-sync python` 验证可运行。
39. 第二册 `book-02-advanced-100/chapters/01-深度学习与概率基础进阶.md`：第 3 讲优化景观与深度网络训练动态已完成首轮代码补充；新增一维二次函数上不同 learning rate 的梯度下降轨迹 demo，展示学习率过小收敛慢、合适学习率稳定下降、过大学习率震荡发散，并已用 `uv run --no-sync python` 验证可运行。
40. 第二册 `book-02-advanced-100/chapters/01-深度学习与概率基础进阶.md`：第 4 讲泛化、过拟合与双下降现象已完成首轮资料边界增强和代码补充；按 `WRITING_PLAN.md` 联网核对 Deep Double Descent 论文，补充 double descent 也可能随 epoch、数据量和 effective model complexity 出现的谨慎表述，新增双下降玩具曲线 demo，并已用 `uv run --no-sync python` 验证可运行。
41. 第二册 `book-02-advanced-100/chapters/01-深度学习与概率基础进阶.md`：第 5 讲贝叶斯视角、先验与不确定性已完成首轮代码补充；新增 Beta-Binomial 硬币后验更新 demo，展示先验、观测数据和后验均值的关系，用于解释数据少时先验影响大、数据多时似然影响大的直觉，并已用 `uv run --no-sync python` 验证可运行。
42. 第二册 `book-02-advanced-100/chapters/01-深度学习与概率基础进阶.md`：第 6 讲损失函数设计与优化目标错配已完成首轮代码补充；新增可运行的代理指标错配 demo，展示模型 loss 更低但 exact match 和 safety pass rate 更差的情况，用于说明单一训练指标不能替代任务指标、安全指标和 bad case 分析，并已用 `uv run --no-sync python` 验证可运行。
43. 第二册 `book-02-advanced-100/chapters/01-深度学习与概率基础进阶.md`：第 7 讲归一化、初始化与训练稳定性已完成首轮代码补充；按 `WRITING_PLAN.md` 核对 PyTorch `clip_grad_norm_` 接口资料，新增可直接运行的 tiny model 梯度范数裁剪 demo，展示裁剪前后 grad norm 变化，并已用 `uv run --no-sync python` 验证可运行；运行时出现本机 CUDA driver 警告但 CPU demo 正常完成。
44. 第二册 `book-02-advanced-100/chapters/01-深度学习与概率基础进阶.md`：第 8 讲激活函数、门控结构与 SwiGLU 已完成首轮代码补充；新增可运行的 GELU FFN 与 SwiGLU MLP shape/参数量对比 demo，验证 SwiGLU 输入输出 shape 不变，并展示通过调小 hidden dimension 可让 SwiGLU 参数量接近普通 `4*d_model` GELU FFN，已用 `uv run --no-sync python` 验证可运行。
45. 第二册 `book-02-advanced-100/chapters/01-深度学习与概率基础进阶.md`：第 9 讲优化器进阶已完成首轮资料边界核对和代码补充；按 `WRITING_PLAN.md` 核对 AdamW decoupled weight decay 论文与 PyTorch AdamW 文档，补充可运行的 tiny model 参数分组 demo，验证 Linear weight 进入 decay group、bias/norm 参数进入 no_decay group，并已用 `uv run --no-sync python` 验证可运行。
46. 第二册 `book-02-advanced-100/chapters/01-深度学习与概率基础进阶.md`：第 10 讲深度学习基础高频开放题已完成首轮代码化模板补充；新增开放题回答自评脚本，用关键词维度检查回答是否覆盖 target、mechanism、evaluation 和 tradeoff，帮助把开放题训练转化为可复盘标准，并已用 `uv run --no-sync python` 验证可运行。
47. 第二册 `book-02-advanced-100/chapters/01-深度学习与概率基础进阶.md`：完成全文件第二轮阶段性收尾检查；第 1-10 讲已完成公式兼容修正、代码 demo 补充、资料边界核对或结构化增强，常见公式风险模式、广告注入关键词和无关链接均无命中，`git diff --check` 通过。
48. 第二册 `book-02-advanced-100/chapters/02-transformer架构进阶.md`：第 11 讲 Self-Attention 的矩阵视角与信息路由已完成首轮公式和代码精修；将 `d_{\text{head}}` 等 attention 公式改为 fenced math 和 `d_h` 稳定符号，新增可运行的 causal self-attention shape/mask demo，验证输出 shape 和未来 token attention 权重为 0，并已用 `uv run --no-sync python` 验证可运行。
49. 第二册 `book-02-advanced-100/chapters/02-transformer架构进阶.md`：第 12 讲 Attention 复杂度瓶颈与稀疏化思路已完成首轮公式和代码精修；按 `WRITING_PLAN.md` 核对 FlashAttention、Longformer 和 BigBird 论文摘要边界，将复杂度符号统一为 `d_h`/`d_model`，补充 KV Cache 元素量公式、FlashAttention 与稀疏/线性 attention 的区别，并新增 sliding window mask 与 attention 成本估算 demo，已用 `uv run --no-sync python` 验证可运行；常见公式风险模式、广告注入关键词和 `git diff --check` 均通过。
50. 第二册 `book-02-advanced-100/chapters/02-transformer架构进阶.md`：第 13 讲 Pre-LN、Post-LN、Sandwich-LN 与 DeepNorm 已完成首轮资料边界核对和代码补充；按 `WRITING_PLAN.md` 核对 DeepNet/DeepNorm 论文摘要，补充 DeepNorm 是 residual connection 修改与配套初始化的组合设计，新增可运行的 Pre-LN/Post-LN 多层 toy block 输入梯度范数对比 demo，并已用 `uv run --no-sync python` 验证可运行；运行时出现本机 CUDA driver 警告但 CPU demo 正常完成。
51. 第二册 `book-02-advanced-100/chapters/02-transformer架构进阶.md`：第 14 讲 RMSNorm、ScaleNorm 与归一化简化已完成首轮资料边界核对、公式表达修正和代码补充；按 `WRITING_PLAN.md` 核对 RMSNorm、ScaleNorm 论文摘要及 PyTorch RMSNorm 文档入口，补充 RMSNorm 去掉 re-centering 但保留 re-scaling 的边界、ScaleNorm 的 L2 norm + 单一 scale 表述，并新增 LayerNorm/RMSNorm/ScaleNorm 手写对比 demo，已用 `uv run --no-sync python` 验证可运行；本讲新增内容已完成常见公式风险、广告注入关键词和 `git diff --check` 检查。
52. 第二册 `book-02-advanced-100/chapters/02-transformer架构进阶.md`：第 15 讲 FFN、GeLU、SwiGLU 与门控 MLP 已完成首轮资料边界核对、公式表达修正和代码补充；按 `WRITING_PLAN.md` 核对 GeLU、GLU 和 GLU variants/SwiGLU 论文摘要，补充 GeLU 的 `z * Phi(z)` 边界、GLU/SwiGLU 是逐 token MLP 门控而非 attention 机制的说明，将参数量公式改为更稳的 ASCII 表达，并新增 GeLU FFN 与 SwiGLU MLP 的 shape/参数量对比 demo，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
53. 第二册 `book-02-advanced-100/chapters/02-transformer架构进阶.md`：第 16 讲 RoPE 的数学直觉与位置外推已完成首轮资料边界核对、公式表达修正和代码补充；按 `WRITING_PLAN.md` 核对 RoFormer/RoPE、Position Interpolation 和 YaRN 论文摘要，补充 RoPE 是用旋转矩阵编码绝对位置并在 self-attention 中体现相对位置依赖的边界，以及长上下文扩展不能只靠直接外推或改 max length 的说明；新增二维 RoPE 点积相对距离等价 demo，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
54. 第二册 `book-02-advanced-100/chapters/02-transformer架构进阶.md`：第 17 讲 ALiBi、相对位置编码与长度泛化已完成首轮资料边界核对、公式表达修正和代码补充；按 `WRITING_PLAN.md` 核对 ALiBi 原论文和 T5 relative position bias 资料边界，补充 ALiBi 不加位置 embedding、而是在 query-key attention score 上加距离成比例惩罚的说明，并区分 ALiBi 线性 bias 与 T5 bucket learned bias；新增多 head ALiBi bias demo，验证 causal mask、距离越远 bias 越负、不同 slope 产生不同惩罚强度，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
55. 第二册 `book-02-advanced-100/chapters/02-transformer架构进阶.md`：第 18 讲 MQA、GQA 与推理效率优化已完成首轮资料边界核对、公式表达修正和代码补充；按 `WRITING_PLAN.md` 核对 MQA、GQA 和 PagedAttention 论文摘要，补充 MQA 通过共享 K/V 降低增量解码内存带宽成本、GQA 是 MQA 的中间 KV head 数泛化、PagedAttention 是 KV Cache 管理而非减少 KV head 数的边界；将 KV Cache 估算公式改为 ASCII 稳定表达，并新增 MHA/GQA/MQA KV Cache MiB 估算与 query head 到 KV head 分组映射 demo，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
56. 第二册 `book-02-advanced-100/chapters/02-transformer架构进阶.md`：第 19 讲 MoE 架构基础已完成首轮资料边界核对、公式表达修正和代码补充；按 `WRITING_PLAN.md` 核对 Sparsely-Gated MoE、GShard 和 Switch Transformer 论文摘要，补充 MoE 的 conditional computation 边界、GShard 的自动分片/通信视角以及 Switch top-1 routing 仍需负载均衡和稳定化技巧的说明；将 MoE 聚合公式改为 ASCII 稳定表达，并新增可运行 TinyTop1MoE routing demo，验证输出 shape、expert id 和 token counts，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
57. 第二册 `book-02-advanced-100/chapters/02-transformer架构进阶.md`：第 20 讲 MoE 训练稳定性与负载均衡已完成首轮资料边界核对、公式表达修正和代码补充；沿用并核对 Sparsely-Gated MoE、GShard 和 Switch Transformer 关于负载均衡、capacity、top-1 routing 与 router 稳定化的边界，补充 MoE 不能只堆 expert 数量、Switch z-loss 稳定 router logits 但不替代 load balancing loss 的说明，并新增可运行 capacity/overflow demo，展示 capacity factor 对 kept/dropped tokens 和 expert token counts 的影响，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
58. 第二册 `book-02-advanced-100/chapters/02-transformer架构进阶.md`：第 21 讲 Encoder-only、Decoder-only 与 Encoder-Decoder 再比较已完成首轮资料边界核对和代码补充；按 `WRITING_PLAN.md` 核对 BERT、GPT 类 decoder-only 和 T5 text-to-text encoder-decoder 的架构边界，补充 BERT 是 deep bidirectional representations、GPT 类训练/推理/产品交互的 next-token 闭环、T5 统一 text-to-text 但仍保留 encoder-decoder 条件生成优势的说明；新增可运行 attention mask 对比 demo，展示 encoder-only 双向 self-attention、decoder-only causal self-attention、encoder-decoder 的 encoder self-attention、decoder self-attention 与 cross-attention 信息流，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
59. 第二册 `book-02-advanced-100/chapters/02-transformer架构进阶.md`：第 22 讲状态空间模型与 Transformer 替代路线已完成首轮资料边界核对、公式表达修正和代码补充；按 `WRITING_PLAN.md` 核对 S4、Mamba 和 Mamba-2/SSD 论文摘要，补充 S4 是结构化 SSM 长序列路线、Mamba 通过输入相关 selective SSM 弥补内容相关推理弱点并使用硬件感知 scan、Mamba-2/SSD 强调 SSM 与 attention 结构联系而非简单替代的边界；将本讲 `O(T²)` 风险表达改为 `O(T^2)`，并新增可运行 scalar SSM 递推 demo，展示状态对历史输入的滚动压缩和 `A` 对遗忘速度的影响，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
60. 第二册 `book-02-advanced-100/chapters/02-transformer架构进阶.md`：第 23 讲混合架构：Attention、SSM 与卷积的结合已完成首轮资料边界核对和代码补充；按 `WRITING_PLAN.md` 核对 Mamba-2/SSD、Mamba 和 Conformer 论文摘要，补充混合架构不是简单拼装，而是在 attention/SSM/卷积的归纳偏置、效率和硬件实现之间重新组合的边界；新增可运行混合层粗略成本估算 demo，比较 all-attention、hybrid 和 mostly-SSM 三种层比例在长序列下的 relative cost，并强调成本下降不等于质量必然更好，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
61. 第二册 `book-02-advanced-100/chapters/02-transformer架构进阶.md`：第 24 讲模型宽度、深度、head 数和 hidden size 的设计已完成首轮资料边界核对、公式表达修正和代码补充；按 `WRITING_PLAN.md` 核对 Chinchilla compute-optimal scaling、LLaMA 和 GLU/SwiGLU 资料边界，补充模型尺寸不能脱离训练 token/compute、相对高效模型可通过更充分训练提升质量、SwiGLU intermediate size 需要按参数量和硬件取整设计的说明；将本讲 `d_model²` 和 `≈` 风险公式改为 ASCII 稳定表达，并新增可运行 Transformer block 参数量估算 demo，比较 GeLU `4d` FFN 与取整后 SwiGLU intermediate size 的参数量，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
62. 第二册 `book-02-advanced-100/chapters/02-transformer架构进阶.md`：第 25 讲现代 LLM 架构案例拆解已完成首轮资料边界核对、公式表达修正和代码补充；按 `WRITING_PLAN.md` 核对 LLaMA、GQA 和 RoPE 资料边界，补充 LLaMA 风格是现代 decoder-only 组件组合代表、GQA/MQA 主要服务推理效率、RoPE 服务位置建模且需结合训练长度和评估理解的说明；将案例中的 `intermediate_size ≈ 11008` 改为稳定表达，并新增可运行 LLM config 解析 demo，自动判断 GQA、KV Cache 相对 MHA 比例、SwiGLU-like MLP、RMSNorm 和 RoPE，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
63. 第二册 `book-02-advanced-100/chapters/02-transformer架构进阶.md`：完成第 11-25 讲首轮阶段性收尾；本文件已覆盖 Self-Attention、attention 复杂度、norm placement、RMSNorm/ScaleNorm、FFN/SwiGLU、RoPE、ALiBi、MQA/GQA、MoE、架构范式、SSM/混合架构、尺寸设计和现代 LLM 案例拆解的资料边界核对、公式表达修正和可运行 demo 补充；新增内容均已完成广告注入关键词和 `git diff --check` 检查。
64. 第二册 `book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md`：第 26 讲 Tokenizer 进阶：BPE、Unigram 与 SentencePiece 已完成首轮资料边界核对和代码补充；按 `WRITING_PLAN.md` 核对 BPE 子词、Unigram/subword regularization 和 SentencePiece 资料边界，补充 BPE 缓解 open-vocabulary/rare words、Unigram 支持概率候选切分、SentencePiece 从 raw sentence 直接训练且不依赖空格预分词的说明；新增可运行 BPE merge 与字符/UTF-8 byte 压缩率对比 demo，展示高频 pair 合并和中文在 byte 粒度下更长的现象，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
65. 第二册 `book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md`：第 27 讲数据清洗 pipeline 设计已完成首轮资料边界核对和代码补充；按 `WRITING_PLAN.md` 核对 C4/T5、Gopher 和 RefinedWeb 资料边界，补充数据清洗不是越狠越好，而是在规模、质量、多样性、安全和可追踪之间平衡，properly filtered and deduplicated web data 也可以有竞争力的说明；新增可运行 toy data cleaning pipeline demo，覆盖 HTML 清理、简单语言识别、质量过滤、安全关键词过滤、exact dedup、lineage 字段和统计输出，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
66. 第二册 `book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md`：第 28 讲数据去重与 Benchmark Contamination 已完成首轮资料边界核对和代码补充；按 `WRITING_PLAN.md` 核对 Deduplicating Training Data Makes Language Models Better、RefinedWeb 和 rephrased contamination 资料边界，补充去重可降低记忆和 train-test overlap、filtered + deduplicated web data 的重要性，以及纯字符串匹配不足以覆盖改写/翻译污染的说明；新增可运行 exact dedup、word-set Jaccard near dedup 和 benchmark n-gram overlap contamination demo，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
67. 第二册 `book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md`：第 29 讲数据配比与 Curriculum Learning 已完成首轮资料边界核对和代码补充；按 `WRITING_PLAN.md` 核对 T5/C4、Chinchilla compute-optimal scaling，并尝试核对 Curriculum Learning 原始资料但 ACM 页面受限，补充数据 mixture 不能脱离训练目标、数据质量和 compute budget，curriculum 是可实验 schedule 而非默认优于随机混合的说明；新增可运行 mixture/curriculum sampling demo，展示不同阶段 planned tokens、exposure multiplier 和小数据集重复暴露风险，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
68. 第二册 `book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md`：第 30 讲合成数据的价值与风险已完成首轮资料边界核对和代码补充；按 `WRITING_PLAN.md` 核对 Self-Instruct、Orca 和 rephrased contamination 资料边界，补充合成指令需要过滤无效/重复/相似样本、蒸馏应关注复杂解释轨迹而非只模仿风格、合成数据也可能造成 benchmark contamination 的说明；新增可运行 synthetic data filtering demo，覆盖答案存在性、质量阈值、near duplicate、benchmark overlap 和 synthetic ratio 计算，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
69. 第二册 `book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md`：第 31 讲 Scaling Law 经典形式已完成首轮资料边界核对、公式表达修正和代码补充；按 `WRITING_PLAN.md` 核对 Kaplan scaling law、Chinchilla 和 emergent abilities 资料边界，补充经典 scaling law 主要拟合 cross-entropy loss 的幂律趋势、曲线依赖数据/架构/tokenizer/recipe 条件、compute-optimal scaling 关注预算切分、下游离散能力可能呈非线性显现的说明；将本讲 `C ≈ 6ND`、`L(N)=A/N^α+...` 等风险公式改为 ASCII 稳定表达，并新增可运行 power-law loss toy demo，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
70. 第二册 `book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md`：第 32 讲 Chinchilla Scaling Law 已完成首轮资料边界核对、公式表达修正和代码补充；按 `WRITING_PLAN.md` 核对 Chinchilla、Gopher 和 Kaplan scaling law 资料边界，补充 Chinchilla 研究的是固定 compute 下模型大小与训练 token 数的分配，指出许多早期大模型 undertrained，但不是机械主张模型越小越好的说明；将本讲 `C ≈ 6ND` 风险公式改为 `C = 6 * N * D` 稳定表达，并新增可运行固定 compute 权衡 toy demo，展示参数量增大时可训练 token 数下降且存在中间候选最优点，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
71. 第二册 `book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md`：第 33 讲 Scaling Law 的局限与新趋势已完成首轮资料边界核对和代码补充；按 `WRITING_PLAN.md` 核对 emergent abilities 和 self-consistency/test-time compute 资料边界，补充 emergent abilities 不等于所有能力神秘不可预测、平均 loss 可较平滑但离散任务指标可能因阈值或评分方式非线性显现的说明，并明确第三个检索结果偏离主题未纳入正文；新增可运行 scaling extrapolation shift demo，展示小规模 power-law 外推在大规模数据分布变化后会系统性低估 loss 风险，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
72. 第二册 `book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md`：第 34 讲训练 loss、验证 loss 与下游能力的关系已完成首轮资料边界核对、公式表达修正和代码补充；按 `WRITING_PLAN.md` 核对 Hugging Face perplexity 文档和 Kaplan scaling law，补充 PPL 是序列平均 negative log-likelihood 指数形式、适用于 causal LM 且受 tokenizer/验证集/上下文切分影响，loss/PPL 是预训练质量信号但不能替代能力、偏好和安全评估的说明；将本讲 PPL 和 LM loss 风险公式改为稳定文本表达，并新增完整可运行 loss/PPL 与目标能力错配 demo，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
73. 第二册 `book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md`：第 35 讲训练稳定性：loss spike、NaN 与梯度爆炸已完成首轮资料边界核对和代码补充；按 `WRITING_PLAN.md` 核对 PyTorch `clip_grad_norm_` 和 `torch.isfinite` 文档入口，补充 `clip_grad_norm_` 计算全局梯度范数并原地裁剪、返回裁剪前范数，AMP 页面请求失败但保留先 `unscale_` 再 clip 的标准流程说明；新增完整可运行 tiny gradient clipping demo，验证 finite loss 检查、裁剪前 grad norm、`clip_grad_norm_` 返回值和裁剪后 grad norm，已用 `uv run --no-sync python` 验证可运行；运行时出现本机 CUDA driver 警告但 CPU demo 正常完成，本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。
74. 第二册 `book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md`：第 36 讲混合精度训练：FP16、BF16 与 FP8 已完成首轮资料边界核对、公式表达修正和代码补充；按 `WRITING_PLAN.md` 核对 PyTorch AMP/type info 文档入口和 NVIDIA Transformer Engine FP8 文档，补充 FP8 依赖 scaling recipe、amax 统计、FP8 tensor 格式、高精度 accumulation 和硬件 kernel，不是单独 dtype 开关的说明；将稳定 softmax 公式改为 ASCII 文本表达，并新增可运行 `torch.finfo` dtype 范围与 FP16 overflow demo，展示 FP16 在 70000 以上溢出而 BF16 保持 finite 但精度更粗，已用 `uv run --no-sync python` 验证可运行；本讲范围内常见公式风险、广告注入关键词和 `git diff --check` 检查通过。

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
