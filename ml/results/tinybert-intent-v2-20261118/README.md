# TinyBERT 32能力三态训练 · 2026-11-18

已完成实验性单句微调，训练705条/141族、验证180条/36族、测试候选295条/59族。训练55轮，按验证集选第43轮；需要门槛0.8、禁止门槛0.75。原始候选文件和现有五标签应用未替换。

| 集合 | 需要能力F1 | 明确禁止F1 | 意图准确率 | 全状态+意图完全一致 |
| --- | ---: | ---: | ---: | ---: |
| train | 91.6% | 88.2% | 100.0% | 530/705 |
| validation | 54.0% | 36.5% | 90.6% | 51/180 |
| test | 44.1% | 54.9% | 67.5% | 44/295 |

测试候选152/295条存在额外需要能力预测，41/295条触及明确禁止能力；只有2/59个场景的五种改写全部答对。范围外0/50、无需设备能力0/40被正确识别，意图宏F1为36.7%。模型泛化不足，不应接管权限判断。训练集高分不能作为真实用户准确率。

本次是AI合成候选的内部开发评估。人工审核未完成，测试完整禁止题仍缺14项能力，早期试验也已观察过这份候选；不是新鲜盲测或已冻结验收。

输入为text经配套分词器编码的input_ids/attention_mask。每项能力输出未提及/需要/明确禁止三态，另有四类意图；不生成对象、时长、范围槽位或具体追问句。

## 已验证

- 9项标签、损失、校准、梯度、保存/加载、长度和分组检查通过。
- 最优权重保存再加载输出完全一致。
- 保存分词器与应用真实WordPiece在1500条原话上input_ids/attention_mask完全一致。
- 标准ONNX三档CPU对照分别5、60、295条，标签/状态无差异，最大绝对logit差9.5367e-6。16档只覆盖能完整放入的5条测试句，不能据此声称覆盖全部输入。
- 未编译新OMC，未执行手机NPU验收，未更改现有HAP模型。

## 完整模型包中的文件

- checkpoint/：模型权重、词表、配置和protocol.json。
- onnx/：16/32/64 token标准ONNX及PyTorch参考向量。
- training_report.json：训练、验证、内部测试详细指标。
- test_review.csv / test_predictions.jsonl：逐条标准答案与预测。
- error_analysis.json / smoke_predictions.jsonl：错误分布和额外示例，未用来修改最终权重。
- data_manifest.json / run_splits/：实际训练划分、转移ID和排除的上下文题。
- development_ledger.json / development_attempts/：早期失败试验与源码快照；最终权重从固定官方底座独立训练。
- artifact_manifest.json：文件SHA256与字节数。

在完整审核包解压根目录运行：

    python3 ml/predict_permission_intent_v2.py --checkpoint run/checkpoint "录一段视频，但不要收音。"

依赖见ml/requirements-training-cpu.txt。此模型是自定义双输出checkpoint，请使用配套加载器；不能直接替换现有五标签OMC/CPU回退文件。

本仓库目录仅保存轻量指标、协议和审阅表；大权重与ONNX位于完整模型包，不在本目录。复现入口见[训练说明](../../../docs/tinybert-intent-v2-training.md)。

## 下载

- [主包：权重、64档ONNX、代码、数据与全部报告](https://drive.google.com/file/d/1LoC0yhKytiP9JF93Vh_qSvBgmuo8COF2/view?usp=drivesdk)（86.9 MB）。
- [补充包：16/32档ONNX与参考向量](https://drive.google.com/file/d/1KnPcbL94Xepb4sZecpyjWfT5s9pvkfT1/view?usp=drivesdk)（85.2 MB）。

两个ZIP可解压到同一根目录，无需二进制拼接。SHA256和文件字节数见[downloads.json](downloads.json)。Google Drive链接沿用上传账户访问权限。
