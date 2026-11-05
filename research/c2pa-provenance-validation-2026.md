# C2PA 内容来源与验证边界

核验日期：2026-11-05。

本记录保存官方规范的版本事实和验证边界，不记录书籍修改过程。

- **来源：** Coalition for Content Provenance and Authenticity, [Content Credentials: C2PA Technical Specification](https://spec.c2pa.org/specifications/specifications/2.5/specs/C2PA_Specification.html)，C2PA Specification 2.5，2026-10。
- **阅读范围：** §1.1–1.3（范围、断言、claim、manifest、签名及信任模型）、§5.3 版本历史、§14 验证流程和来源显示。
- **来源支持：** C2PA manifest 组织关于资产创建、编辑操作、内容绑定等 assertions，由 signer 对 claim 作数字签名；manifest 验证依据指定的信任模型检查签名、内容绑定、声明结构和关联成分。2.5 版的版本历史日期为 2026 年 10 月，并把此前的 live video 扩展命名为 live media。
- **证据边界：** 验证结果表示声明与资产之间的绑定及签名状态按规范可验证，不会判定 assertions 的事实真伪、版权许可、人物同意或输出安全。是否信任 signer 依验证器配置的信任模型和信任列表；内容缺少 manifest 或验证失败也不能自动证明内容造假。版本能力必须按具体规范版本和实现支持范围判断。
