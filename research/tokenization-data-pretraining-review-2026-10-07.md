# Tokenization、数据与预训练进阶精修记录（2026-10-07）

## 范围与状态

- `chapter`：`book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md`。
- `scope`：修订前 1,632 行，第 26—40 讲；当前 1,936 行。第二册目录所列范围保持不变。
- `evidence_level`：`paper`、`official`、`local-demo`、`inference`。
- `lifecycle`：`complete`。论文支持的结论绑定论文任务与实验；toy 代码不表示生产系统和真实模型验收。
- `status`：已完成章节审阅和本地教学代码核验；CPU-only PyTorch 展示解释性输出，不安装 CUDA 库，也不要求完整复现。

## 主要修订

- 将质量组合分数限定为已获准样本的排序分数；许可与来源可追溯性、隐私/安全、评估污染作为独立准入或隔离门槛，避免高质量分抵消硬性风险。
- 清洗示例现在拒绝未知或未列入项目白名单的许可，并列出可审计拒绝原因。
- 为 BPE 词频与合并、tokenizer 分桶统计、shingle 空集合、温度采样、合成样本筛选和 Scaling 曲线补输入契约及边界处理。
- 为交叉熵/PPL 检查 logits/labels 的 dtype、shape、device、`ignore_index` 和全 mask 状态；为梯度裁剪检查有限阈值、标量 loss、梯度范数和更新后参数。
- checkpoint demo 现在实际保存状态、改写参数并调用恢复，核验参数与 step/data cursor；正文限定 `weights_only=False` 只能用于可信文件，并区分文件系统 rename 与对象存储 manifest 提交。
- 多语言采样改为 log-space 权重，拒绝布尔、负数和非有限值；FIM 代码检查半开区间，pass@k 用逐项乘积计算以避免直接构造巨大组合数。
- 更正 BF16 参考：arXiv `1905.12322` 的实际题名为 *A Study of BFLOAT16 for Deep Learning Training*，首位作者为 Dhiraj Kalamkar；原列 Wang and Kanwar 的 Cloud TPU 文章标题与该 ID 不匹配。

## 来源与证据范围

章节中的 BPE、SentencePiece、Unigram、C4、去重、FineWeb、代码数据、合成数据、Scaling Law、FP16/BF16、多语言、HumanEval、FIM 和 SWE-bench 主张已在相邻段落增加论文入口；PyTorch 交叉熵、AMP 与 checkpoint 说明引用官方文档。

`1905.12322` 的 arXiv Atom API 返回题名、作者和摘要，确认其内容是 BFLOAT16 格式及多类训练任务实验，而非 Cloud TPU 博客。该研究可支持论文所测格式和任务，不支持所有当前硬件上的速度或 kernel 行为。其余论文 URL 均做了目标状态检查；批量 arXiv 元数据查询本轮间歇返回连接错误，因此不把 HTTP 200 当成对全部论文摘要或全文的独立复核。正文只保留各论文可支持范围内的描述。

当前 31 个唯一外部 URL 的最终目标均返回 HTTP 200。原 PyTorch AMP 教程旧域名请求超时，已换为 `docs.pytorch.org` 官方教程，替代目标返回 HTTP 200。已分别确认 ACL anthology 三个入口、Nature model-collapse 页面、OpenAI GPT PDF、PyTorch CrossEntropyLoss/AMP/distributed checkpoint/general checkpoint 页面及 arXiv 页面目标存在。

## 本地教学代码与格式检查

- 14 个 Python 围栏均通过语法编译。
- 10 个纯标准库代码块在系统 Python 3.12.9 下逐块运行；BPE 频数、清洗准入、tokenizer 统计、shingle、温度采样、合成筛选、Scaling 曲线、checkpoint 候选选择、多语言计划和 FIM/pass@k 的输出与正文描述一致。
- 另外通过 43 项边界断言，覆盖布尔/零/负/NaN/Inf 参数、未知许可、空 shingle、极端温度、样本分母为零、非法 FIM span 和 pass@k 定义域。
- 另外 4 个依赖 PyTorch/NumPy 的代码块在 Python 3.12.9、PyTorch 2.12.0+cpu、NumPy 2.4.6 下逐块执行通过，固定输出分别为 `loss= 2.2648 ppl= 9.6288`、`loss= 0.0374 preclip_grad_norm= 0.2892`、FP16 对 70,000 和 1,000,000 转换为 `inf` 而 BF16/FP32 保持有限，以及 `restored: 1 {'shard': 2, 'offset': 16} parameters= True`。PyTorch 报告 `torch.cuda.is_available() == False`。环境安装在仓库 `.tmp/llm-book-cpu-check/`，`.tmp/` 已加入 `.gitignore`；没有下载或安装 CUDA 库。
- 29 个 fenced `math` 块、代码/文本/公式围栏配对；数学花括号和 `begin/end` 环境平衡。5 个 Markdown 表格列数一致。源级检查未发现禁用公式写法；未使用外部 MathJax 渲染器。
- `git diff --check` 通过。章节无相对 Markdown 链接；第二册目录文件与第 26—40 讲范围一致。

## 配套同步

- `PAPERS.md` 增加 tokenizer、语料、去重、Scaling 与 BF16 来源入口。
- `GLOSSARY_EN_ZH.md` 增加 Unigram tokenizer 与 token fertility。
- `EXERCISES.md` 的数据源估值题要求将许可、隐私和污染作为独立门槛。
- 章节目录和简介无须因本次修订改变。

## 未验证项

- 未运行 CUDA、多卡训练、真实模型/语料、生产 tokenizer、数据治理流水线或 serving；不把 CPU 教学 demo 外推为这些场景的结果。
- 未运行 Pandoc/PDF 构建或外部公式渲染。
