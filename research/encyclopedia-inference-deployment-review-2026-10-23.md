# 第四册推理优化与部署精修核验（2026-10-23）

## 范围与首个检查点

全文初读第 9.1—9.19；本批核验请求生命周期、token workload、scheduler、KV/PagedAttention 账本、online softmax、虚拟页、量化、runtime 选择、PD 粗略判定、扩缩容和分位数摘要代码。整章尚未验收。

## 修订与本地证据

- 教学代码的整数输入拒绝 bool；`workload_summary`、KV、虚拟页、量化、PD、replica 和 percentile 输入契约收紧。
- scheduler 将 running sequence 数与 waiting prefill token 预算分开，避免用请求数充当 token 负载。
- percentile 明确采用 nearest-rank 离散口径：q=0 取最小值，q>0 取 ceil(q*n) 位置。
- 生命周期账本增加显式状态转移表，拒绝从 RECEIVED 直接进入 DECODING；prefix 命中率说明输入 token 分母必须大于零；量化输入拒绝 bool。

`python3 /tmp/check_deployment_review.py` 通过：12 个标准库 Python 块独立执行，非法输入、预算、分位数和 21 个 math 块结构/围栏检查通过。未运行真实 runtime、GPU、网络 serving 或压力测试。

## 待办

继续核验生命周期状态转移、动态 batching 公平性、paged/prefix cache 引用计数、FlashAttention/量化公式、PD/多级 KV/路由保护、API/流式协议/发布回滚及整章来源、链接和配套同步。

## 整章收口记录

全文第 9.1—9.19 复读完成。12 个 Python 块和 21 个公式块通过；18 个链接目标中 17 个 HTTP200，vLLM 文档因代理返回 403，PagedAttention 固定 HTML 路径曾 404，但 abs 入口 HTTP200。来源阅读限于 FlashAttention、vAttention、GPTQ、AWQ 的摘要/方法段落及官方文档页面，未运行真实 GPU、kernel、serving 或压力测试。章节结论仅覆盖教学账本和所读资料。

## 来源初查

- [FlashAttention v2 HTML](https://arxiv.org/html/2205.14135v2)：读取摘要与方法图注，确认 IO-aware、tiling、避免物化 `N×N` attention 矩阵和精确 softmax；未运行 CUDA kernel。
- [vAttention v1 HTML](https://arxiv.org/html/2405.04437v1)：核对连续虚拟 KV 视图与 PagedAttention 的路线差异；未验证 GPU VMM。
- [GPTQ v2 HTML](https://arxiv.org/html/2210.17323v2)：读取摘要，确认 approximate second-order、one-shot weight quantization 的范围。
- [AWQ v2 HTML](https://arxiv.org/html/2306.00978v2)：读取摘要，确认 activation-aware、保护 salient weights 的动机；未复现实验。
- PagedAttention 的 `2309.06180` HTML 版本本次尝试返回 404，仍保留 abs 入口并待后续固定版本核验。
