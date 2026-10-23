# 第三册第十一章多模态实战精修证据

- 日期：2026-10-22；状态：声明范围内整章精修验收完成；保留下面首个局部检查点及后续补充证据。
- 范围：第 63—72 讲全文复读与修订；不将教学代码验收推广为真实模型或生产系统验证。
- 证据等级：paper、official、local-demo；未运行真实 CLIP/VLM/Whisper 或模型训练，不安装 CUDA。

## 已核对来源

通过维护者指定代理读取：

1. [CLIP v1 HTML](https://arxiv.org/html/2103.00020v1)：核对双向对称交叉熵、batch 内配对与负样本、学习 log 参数化的温度缩放，以及文本编码器构造 zero-shot 分类器的相关段落；不是全文实验复核。
2. [Transformers multimodal chat templates](https://huggingface.co/docs/transformers/chat_templating_multimodal)：读取 Processor、AutoModelForImageTextToText、apply_chat_template 和生成输出包含原对话的提示；本日页面内链接标示 v5.20.2，属于动态文档快照，不能当成本地安装版本。
3. [Qwen2.5-VL-3B-Instruct 模型卡原文件](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/README.md)：读取 tokenize=False、processor 图像输入和按输入长度截取新增生成 token 的示例；main 未固定 revision，未运行权重或 qwen_vl_utils。

三项请求成功取得内容。尚未检查整章所有外部链接。选择该 checkpoint 是为替代不可运行的虚构模型 ID，不将接口检查扩展为该模型能力或本地推理验证。

## 第 63—66 讲修订

- CLIP 检索补上原文声称但未执行的向量轮换反例；固定正确标签后 Recall@1 从 1 到 0。说明同步排列不是破坏配对，单相关目标下 Recall 等于 Hit，多相关目标需另定分母。
- loss helper 拒绝非方阵 paired logits；补 N=1 零 loss 的退化情况与当前向量的三个损失值。温度与 loss 的大小关系不推广到任意输入。
- 区分 local-query/global-candidate logits 与完整 global logits 的标签；不能无条件加 rank offset，变长 batch 不能直接 W 乘 local batch。
- VLM 骨架使用官方文档展示的 AutoModelForImageTextToText 和真实模型 ID，显式 tokenize=False，仅对 decoder-only 输出去掉输入前缀；注明外部接口未运行、版本/revision 仍须固定。
- 数据覆盖率不再声称无任务样本就不可能迁移；assistant mask 用位置集合，避免把相同 token ID 的用户词和回答词混淆。
- 稳定 ID 序列化使用 JSON，避免分隔符拼接歧义；同图同问题的冲突答案/任务/证据报错，拒绝不同记录复用 ID。图片路径分组不冒充内容/近重复去泄漏。
- 全章机械统一 mathrm 和安全历史序列下标；新增前置学习与目录入口。

## 实际本地验证

使用 Python 3.12.9，从正文提取 12 个 Python 围栏：10 段标准库示例逐块编译且独立执行通过；2 段 transformers 外部接口仅编译，未导入/下载模型或运行推理。

第 63—66 讲额外边界检查通过：空/零/NaN/Inf/bool 向量，非法温度，空/不齐/非方阵 logits，稳定 logsumexp(1000,1000)，单配对 loss=0，完全重复去重，冲突答案/任务/证据，非字符串 evidence，重复 ID，图片数不足和同 seed 切分确定性。后六讲本轮只有正常示例执行证据，不能据此声称边界已验收。

可核对输出：检索 shape=(4,4)、四个正常 rank=1、打乱 Recall@1=0；CLIP loss=0.000095、打乱=10.008890、温度0.5=0.402856；VQA 规则计数切片=0，object/OCR=1；数据4条，cat=valid、invoice=train、chart=test。

本章所有相对 Markdown 目标存在，git diff --check 通过。未运行外部公式渲染器或 PDF 构建；Pandoc 缺失限制沿用。后续需继续第 67—72 讲的 ACL 与证据融合、DDPM 反向机制/数值、Whisper 特征及时间戳、关键实体与分层评估、容量口径、全章来源及配套同步。

## 第 67—72 讲与整章验收补充

以上“后续需继续”描述首个局部检查点，以下为本日继续完成的工作。

### 新增来源核验

- [DDPM v2 HTML](https://arxiv.org/html/2006.11239v2)：读取 forward 闭式关系、后验与 Algorithm 2 反向采样，包括 t=1 随机项为零；没有训练或复现论文实验。
- [Whisper audio.py raw main](https://raw.githubusercontent.com/openai/whisper/main/whisper/audio.py)：读取 log_mel_spectrogram 的功率谱、Mel、clamp、log10、动态范围和缩放；核对采样率 16000、CHUNK_LENGTH=30。只证明所读取 main 快照，不代表所有 checkpoint 或未来版本。
- [Azure security filter](https://learn.microsoft.com/en-us/azure/search/search-security-trimming-for-azure-search)：读取身份字符串过滤模式；文档明确过滤字符串本身不提供身份认证或授权。toy groups 是调用方传入的教学身份条件，真实服务必须从可信认证上下文取得，不能接受用户自报组即授权。
- arXiv Atom API 返回并核对 10 个 ID、题名与版本：CLIP v1、LLaVA v2、BLIP-2 v3、RAG v4、DDPM v2、DDIM v4、Latent Diffusion v2、DiT v2、CFG v1、Whisper v1。元数据核对不等于完整原文/实验核验。
- 21 个正文唯一外链最终全部 HTTP 200。第一次脚本在受限子进程中连接代理失败，获准网络执行后逐项成功；不能把第一次 HTTP 000 当作来源失效。FAISS、HELM、Evaluate、NIST、OpenTelemetry、模型文档等本轮主要确认入口可达，未执行其服务或评测工具。

### 修订与验证范围

1. 证据融合先过滤租户/ACL，关联扩展重新判断可见性，空 ACL 默认拒绝；无权限/空候选/零重合返回空集合，拒绝重复 ID，并为图片保留版本。输出 doc_504/img_504 分数 0.2697/0.4835。内存检查不能证明权限变更竞争、缓存隔离或实际索引安全。
2. DDPM 增加 reverse_step，区分零基代码索引与数学时间步，补干净边界 alpha_bar_0=1；五步最终 alpha_bar=0.858278，反向均值约 [1.976,-0.3]、后验方差0.034063，最后一步方差为零。已知 x0 的噪声反解不充当生成模型，五步 schedule 不是纯噪声起点。
3. Whisper 对数特征公式标为抽象；时间戳约束是检查器选择，不当作模型输出保证。CER/WER 单位、规范化盲点、空参考、插入大于1与语料加权口径分别说明。
4. 语音代码只验证转写检查、检索和合成阶段预算，没有最终问答；中文数字和其他否定形式会漏检。3.19秒不冒充实测总延迟。
5. 统一评估返回 rule_pass，明确关键词误放、禁句误拒、低CER关键错误等反例。代码规则未升级为语义评分器，真实 gold 与人工标注仍必需。
6. 容量公式是工作量到达率，拆输入1536/输出256 token/s；总成本20元/80次成功=0.25元，失败成本不能剔除；记录完整与修复通过分开。

最终重新提取12段Python：10段标准库独立执行，2段外部模型接口仅编译，全部符合声明范围。边界追加覆盖错误租户、空组/空ACL、关联私有/缺失文档、零候选/零重合、重复ID、扩散非法步/shape/非有限数、各步代数反解及方差、空参考、CER>1、时间戳重叠/未知值、检索无命中、关键实体漏检、否定句关键词误放、低CER否定遗漏和零基线。规则盲点通过断言验证存在，不表示语义问题已由规则解决。

56个math块的花括号/环境及总围栏配对检查通过；没有operatorname、尖括号下标或text下划线；相对目标、配套练习与项目入口存在。EXERCISES.md增加6项可操作验收，PROJECTS.md项目11/12补ACL与关键字段要求；论文索引已有主要论文，未重复添加摘要。git diff --check通过。

结果标记为 source-backed、verified、local-demo，范围以以上条目为限。未运行真实模型、GPU、OCR、ASR、生产服务、人工标注实验或外部公式渲染器；未构建PDF。动态文档/main来源没有固定为依赖版本。
