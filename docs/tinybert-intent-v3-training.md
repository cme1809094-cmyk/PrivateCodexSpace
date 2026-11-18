# TinyBERT 对照补强训练 v3

本轮针对 v2 多报能力、禁令识别弱、无法区分无需设备能力与范围外意图的问题，补充场景对照数据并训练实验模型。输入仍是中文单句，输出仍是 32 项能力的三态与四类意图。平台权限映射留在后处理；模型不生成范围、对象等槽位或具体追问文本。

训练使用固定官方 TinyBERT 4L 中文底座及配套词表。初始权重哈希由 `ml/fetch_tinybert.py` 检查。依赖见 `ml/requirements-training-cpu.txt`。

## 数据和选择规则

新数据共 1,400 条／350 组。每组包含两个意思不同的角色，每个角色两种改写。896 条训练、224 条验证，另 280 条作为独立保留测试。全部是 AI 生成候选、作者自审及独立 AI 复核，`human_review=pending`。

保留题由单独作者准备。训练侧在模型选择前只接收 ID、统计、哈希与错误类型，不读取题目、金标或预测。作者和复核者会阅读题目进行标注检查，但不会运行模型。冻结由 `freeze_manifest.json` 记录；模型选择锁包含权重、协议、配置和词表等文件哈希。

旧 v2 训练／验证样本继续参与，保留对全部 32 项能力的覆盖。与新验证近似的旧训练族、与新训练近似的旧验证族整族隔离；记录见 `ai_review_report.json` 的 `legacy_exclusions`。原文件不改。旧测试已经被观察过，只算开发回归。

先在旧开发集的八项能力投影上，以相同底座、数据、随机种子和轮次上限对比当前分类头与标签注意力头。按事先固定的、偏重所需能力精确率的验证指标选择分类头。这只是小规模探索，不能据此认定一种结构普遍更优。

完整训练加入角色对照间隔损失，并调整能力状态及意图类别的权重。按验证集选择轮次和语义阈值，再选择低置信度时要求澄清的策略。该策略同时约束所需能力召回和范围内输入接受率；达不到召回条件时，仍优先保留接受率并公开报告风险，不能靠全部拒绝输入得到好看结果。

`experiment_targets.json` 在完整训练和保留测试前声明实验目标。最终比较包括：旧版原阈值、旧权重在当前验证集重新校准的版本、新版语义结果、新版澄清策略。280 条保留题只在最终选择锁定后读取，评测后不据此调参。

完整配置预算最多两个：先从官方底座训练最多 35 轮；如果验证集所需能力 F1 未到 0.75 或禁止 F1 未到 0.70，再从 v2 权重继续最多 25 轮，编码器学习率 2e-5、分类头 1e-4，重新初始化优化器。两者按同一个验证指标选择，保留题始终不参与选择。所有尝试都会保留配置、历史和源码快照。

## 复现入口

在工程根目录安装依赖并下载固定底座后，先检查数据：

```sh
python3 ml/prepare_intent_contrast_data.py --require-review --render
python3 -m unittest discover -s tests -p test_intent_training.py -v
```

最终实际参数、实际样本 ID 和源码哈希保存于结果目录中的 `training_config.json`、`run_splits.json`。使用全新输出目录复训：

```sh
python3 ml/train_intent_contrast.py --mode full --head pooled \
  --base /path/to/pinned-tinybert --output /path/to/new-run \
  --epochs 35 --min-epochs 12 --patience 8 --batch-size 16 --threads 4
```

训练不带 `--evaluate-reserved`，先生成 `selection_lock.json`。锁定后比较旧版与新版：

```sh
python3 ml/evaluate_intent_contrast.py --run /path/to/new-run \
  --baseline /path/to/v2/checkpoint --output /path/to/new-run/evaluation
python3 ml/export_intent_contrast.py --run /path/to/new-run
python3 ml/verify_intent_tokenizer.py --checkpoint /path/to/new-run/checkpoint \
  --data ml/data/permission_intent_v3/reserved_test.jsonl \
  --output /path/to/new-run/tokenizer_reserved_verification.json
python3 ml/analyze_intent_contrast.py --run /path/to/new-run
```

续训的 v2 初始权重可从[已有 v2 主包](https://drive.google.com/file/d/1LoC0yhKytiP9JF93Vh_qSvBgmuo8COF2/view?usp=drivesdk)取得。它没有重复装入 v3 主包；当前 v3 checkpoint 单独即可推理。

## 如何看验收结果

除所需能力 F1 和明确禁止 F1，还要看多报样本数、违反明确禁令的样本数、四类意图的分别召回，以及整组四条是否全部正确。评分覆盖全部 32 项输出，因此重点八项以外的误报也会扣分。

语义结果和澄清策略结果分别报告。后者必须同时给出范围内输入接受率、所需能力召回，以及接受子集的指标。范围内但低置信度的输入可被策略要求澄清，原始语义类别不因此被改写。

训练后导出 16／32／64 token 标准 ONNX，并校验 PyTorch 与 ONNX Runtime CPU 的数值、语义结果及澄清决策一致性。每档仅检查能完整容纳的输入，不截断句子末尾的禁令。实际应用 WordPiece 与保存分词器另外逐条对照。

这些检查不代表 OMC 编译或手机 NPU 验收。本实验的自定义双输出协议与现有五标签应用不同；权重、词表、阈值和参考向量需要成套适配，不能直接替换现有 HAP 的模型文件。最终实测指标与下载位置见 `ml/results/tinybert-intent-v3-20261118/README.md`。
