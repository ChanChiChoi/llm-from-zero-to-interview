# 第四册推理与解码精修核验（2026-10-22）

## 范围与状态

全文初读第 8.1—8.15 节及回顾；本次完成采样数值与部分延迟口径修订，整章尚未验收。

## 来源

- [Transformers GenerationConfig](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)：读取 do_sample、temperature、top_p、min_p 及生成策略选择说明；核对 top-p 为最高概率排序的最短前缀、min-p 相对最高概率的阈值。main 为动态文档，未安装 Transformers 或验证运行时默认值。
- [Speculative Decoding v2](https://arxiv.org/html/2211.17192v2)：读取第 2 节及接受率、残差分布相关说明；p/q 是已经按 sampling 配置变换后的分布，拒绝后使用正残差归一化，全部接受时另取额外 token。该部分已读取，但正文与代码的完整修订仍待后续完成，不算验收。

## 本次修订与本地证据

采样改局部 Random，验证输入分布且不回落到零概率尾项；高温先缩放以避免极端有限 logits 差值溢出。top-p 先按最大权重缩放，避免权重求和溢出，明确排序前缀及 p=1 正概率集合；硬约束全屏蔽不能绕过回退到任意最高 logit。greedy 的确定性限定固定 logits 与 tie-break。平均已选 token 概率限制在 (0,1]；预算与 KV 维度拒绝 bool。

首 token 时间线改为使用 prefill 最后位置 logits，区分选中首 token 与下一次送入该 token 的 forward。延迟示例拒绝空输出、非单调时间和与 TTFT 不同原点的首时间戳，单 token 时 TPOT 未定义。

`python3 /tmp/check_decoding_review.py`：13 个标准库 Python 块独立编译执行成功；极端 logits [1e308,-1e308]、温度 1e308 的概率约 0.880797/0.119203，相同超大 logits 配极小温度为 0.5/0.5；非法分布拒绝、局部随机流、top-p 大权重及零概率边界、延迟无样本/原点/单调性检查通过。24 个 math 块花括号/环境与围栏结构通过；未使用外部数学渲染器。

## 第二个局部检查点：边界与来源检查

投机解码补充同 token 空间、sampling-adjusted p/q、q=0 提议、p=q 零残差、全部接受的额外 target token 和标准块验证的后续草稿丢弃边界。penalty、beam、best-of-N、投票、KV、约束及配置账本输入边界均已加入本地断言。

`python3 /tmp/check_decoding_review.py --links` 经授权代理运行：13 个 Python 块和 24 个 math 块通过，13 个链接目标中 11 个外部页面 HTTP200；两个文档页面出现代理限制：`https://docs.vllm.ai/en/latest/` 返回 403，Transformers 主类页面曾返回 403 但本地下载内容已用于参数核对，generation strategies 页面 HTTP200。13 个 arXiv/HTML 目标中的论文题名与正文资料项相符；本地相对链接存在。该结果不等于完整论文阅读、真实模型/serving 或性能复现。

## 待办与限制

继续核验 penalty/beam/entropy/约束与配置账本的输入边界、best-of-N 独立性与投票语义、KV 架构适用范围、投机解码支持集与残差示例；全章外链/题名/导航/配套尚未验收。未运行真实模型、GPU、serving 性能或完整投机算法。PDF 仍受 Pandoc 缺失限制。

本次 `--links` 运行经授权代理成功核验 7 个 arXiv abs 页面及投机解码 HTML 页面；随后访问 Hugging Face 文档时代理返回 HTTP 403，流程未将该页面记作 HTTP200。此前通过 curl 下载的本地 `/tmp/decoding-transformers-generation.html` 已用于核对 `do_sample`、temperature、top_p、min_p 和 speculative 参数说明；这不能替代线上可达性证明。
