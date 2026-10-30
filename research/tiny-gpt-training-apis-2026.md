# 小型因果语言模型训练接口：来源证据

核验日期：2026-10-29。

本记录只保存联网核验得到的接口事实、来源版本与适用边界，不记录章节审阅或修改进度。

## PyTorch 2.12

- **交叉熵：** [CrossEntropyLoss / `F.cross_entropy`](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.cross_entropy.html)。阅读接口签名、Shape、类别索引 target、`ignore_index` 和 reduction 说明。该版本要求类别索引 target 为 `torch.long`；输入是未归一化 logits；默认 `mean` 按未忽略的 loss 项求平均。证据等级：source-backed（固定版本官方 API）。边界：若使用 class weight 或概率标签，reduction 分母和 target 契约会不同；书中公式须注明无 padding 或有效 token 分母。
- **Embedding：** [Embedding](https://docs.pytorch.org/docs/2.12/generated/torch.nn.Embedding.html)。阅读查表定义、整数索引输入形状和 `padding_idx`。证据等级：source-backed（固定版本官方 API）。边界：示例采用 LongTensor/`torch.long`；embedding 输出和模型是否支持其他输入 dtype 取决于后续模块。
- **模式与梯度：** [Module](https://docs.pytorch.org/docs/2.12/generated/torch.nn.Module.html) 的 `eval()` 说明及 [`no_grad`](https://docs.pytorch.org/docs/2.12/generated/torch.no_grad.html) 说明。`eval()` 只改变受训练/评估模式影响的模块；`no_grad` 单独关闭该代码区域的反向图记录。证据等级：source-backed（固定版本官方 API）。边界：评估逻辑还需按目标模型检查其他状态，例如自定义缓存、随机采样和分布式归约。
- **数据集：** [`torch.utils.data`](https://docs.pytorch.org/docs/2.12/data.html)。阅读 map-style `Dataset` 与 iterable-style dataset 说明：前者实现 `__getitem__` 和 `__len__`，后者实现 `__iter__`；多 worker iterable 数据需避免每个 worker 重复读取同一数据。证据等级：source-backed（固定版本官方文档）。边界：实际 shard 和 worker 状态由具体数据管线负责。
- **抽样：** [`torch.multinomial`](https://docs.pytorch.org/docs/2.12/generated/torch.multinomial.html)。接口支持向量或矩阵输入；矩阵每行对应一条权重分布，权重须非负、有限且总和非零，并可传入 `generator`。证据等级：source-backed（固定版本官方 API）。边界：具有多个前导 batch 维的 logits 需要展平为行再调用，并恢复输出形状；generator 设备须与输入设备兼容。
- **checkpoint：** [`torch.load`](https://docs.pytorch.org/docs/2.12/generated/torch.load.html) 签名默认 `weights_only=True`，并说明该模式限制反序列化对象范围；同一页明确警告不要加载不可信来源。[`torch.save`](https://docs.pytorch.org/docs/2.12/generated/torch.save.html) 说明保存对象接口。证据等级：source-backed（固定版本官方文档）。边界：`weights_only` 不是来源认证或完整性校验；保存了模型和优化器状态也不自动包含 RNG、数据迭代器、scheduler 或分布式训练状态。

## Hugging Face Transformers v5.17.0

- [Tokenizer API](https://huggingface.co/docs/transformers/v5.17.0/main_classes/tokenizer)：阅读 `from_pretrained`、词表扩容和 `resize_token_embeddings` 示例。文档指出扩容后应把 tokenizer 的完整长度传给模型 embedding resize。证据等级：source-backed（固定版本官方文档）。边界：同名 Hub 仓库的文件可能随 revision 改变；跨运行复现需固定 commit revision 并保存 tokenizer artifact。
- [Tokenization algorithms](https://huggingface.co/docs/transformers/v5.17.0/tokenizer_summary)：阅读 BPE、Unigram、WordPiece 与 SentencePiece 概述。证据等级：source-backed（固定版本官方文档）。边界：算法概述不完整定义特定模型的 pre-tokenizer、normalization、special token 或全部训练配置。

## 未验证边界

- 没有运行 PyTorch、Transformers、tokenizer 或任何本章示例；API 页面只支持所记录版本的接口语义。
- 本记录不证明代码在本地依赖、设备或输入范围内通过执行验证，也不评价训练效果。
