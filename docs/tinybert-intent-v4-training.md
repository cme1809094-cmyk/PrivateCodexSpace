# TinyBERT v4：澄清覆盖与能力证据训练

v3 在独立题中漏掉了矛盾请求、来源／目标／范围未定的歧义；能力头还会将不同动作的允许／禁止混淆。v4 首先补这些语义边界，再比较轻量模型配置。

输出仍是 32×3 能力三态和 4 类意图，不生成对象槽位或追问文本。平台权限映射不变。模型保留 4 层、312 隐藏维度；候选证据头只增加 10,016 个参数，沿用已有分类器，以可学习门控混合整句表示与各能力关注的词语表示。

## 数据与公平比较

新增训练增强 1,248 条，使用可审计的 32 能力原子和有限模板；新验证 224 条与新保留 320 条分别由独立作者写，再独立 AI 逐句审核。受控增强只声明原子／模板 AI 审核和程序全量验证，不冒充逐句审核。全部人审 pending。

实际训练由旧开发训练 1,601＋新增 1,248 组成，验证由旧开发验证 399＋新增 224 组成。原 280 条已经被观察过，只作为已知回归，不用于本轮训练或模型选择。新 320 条在开始训练前冻结；训练侧不读取题目或金标。

两组配置以相同 v3 权重、数据、随机种子和 6 轮预算起步：

- `pooled`：原分类头、原交叉熵及类别权重，作为新增数据的对照。
- `focal_evidence`：保留旧分类器，增加门控能力注意力，搭配 focal loss、较缓的能力类别权重和 0.03 平滑项。这是组合配置，不能把差异归因于单一组件。

验证表现更好的配置最多再续训 10 轮；在两次初训和续训的候选中选择验证分数最高者。分数为 0.40 所需能力 F1＋0.30 禁止 F1＋0.30 意图宏 F1－0.20 禁令违反率。阈值校准在可能时要求所需能力召回至少 50%，澄清策略同时报告接受率和召回，防止全部拒绝输入。

为降低过度自信，候选 focal 损失减少容易的未提及项对训练的支配，保留平滑项；不因概率较高就声称可靠。最终记录概率校准误差与高置信度错判。

训练按每批实际长度移除右侧 padding，不截断文字。部署导出仍是 16／32／64 token 静态档。已有分类头的加载与数值保持检查、注意力 mask、查询梯度、门控、保存重载等测试在训练前通过。

## 复现入口

依赖见 `ml/requirements-training-cpu.txt`。固定 v3 初始权重可从[上一轮主包](https://drive.google.com/file/d/1MCWkosDEW5DRMhmzndbtckr7cVqX_079/view?usp=drivesdk)取得。

```sh
python3 ml/prepare_intent_v4.py --require-review
python3 -m unittest discover -s tests -p test_intent_training.py -v
python3 ml/train_intent_v4.py --variant pooled --initial /path/to/v3/checkpoint --output /path/to/probe-pooled
python3 ml/train_intent_v4.py --variant focal_evidence --initial /path/to/v3/checkpoint --output /path/to/probe-evidence
```

验证选出的配置再用 `--epochs 10 --min-epochs 6 --patience 4 --learning-rate 1e-5 --head-learning-rate 1e-4` 从对应 probe/checkpoint 继续训练，`--variant` 保持不变。全部实际配置、初始文件 SHA、训练 ID、曲线和源码快照随产物保存。

锁定最终候选后才运行：

```sh
python3 ml/evaluate_intent_v4.py --run /path/to/selected \
  --baseline /path/to/v3/checkpoint --output /path/to/selected/evaluation
python3 ml/export_intent_contrast.py --run /path/to/selected \
  --data ml/data/permission_intent_v4/reserved_test.jsonl
```

旧 v3 权重分别保留原协议、以及仅在当前验证集重新校准的协议，与 v4 在同一新 320 条上比较；原 280 条及旧 295 条只作已知回归。新题结果不用于继续调参，所有未达标项均公开。

三档标准 ONNX 与 CPU 对照不代表 OMC 编译或手机 NPU 验收。32 能力协议也不能直接替换现有五标签 HAP，权重、配置、词表、阈值和参考向量必须配套。本轮结果与下载将保存于 `ml/results/tinybert-intent-v4-20261118/`。

## 验证驱动的追加实验

原定两组初训和一次续训结束后，验证集仍显示冲突类判断弱，因此在打开新 320 条前另行记录 `extension_plan.json`、`head_refit_plan.json`、`specialist_distillation_plan.json`。这些属于看到开发验证结果后的追加实验，不能表述成最初预注册的预算；冻结的数据和九项目标没有改动。

追加实验先训练公开的中文 NLI 教师，再固定编码器重拟合分类头。分类头的均值／方差只用训练集计算，最后折叠回线性权重，推理不增加归一化操作。小模型验证分数由续训的 0.6026 升到 0.6361，所需能力 F1 约 0.6429。

教师使用 `IDEA-CCNL/Erlangshen-Roberta-110M-NLI`，固定 revision `864d25be3ce5e90d9193cb49d9fbd52722cdc6b0`，来源及逐文件 SHA 见 `teacher_source.json`。它约 1.02 亿参数，仅用于云端训练；底部 8 层冻结，顶部 4 层、pooler 和新任务头训练。教师整任务验证分数 0.5992，没有达到完整多头蒸馏门槛，因此不复制其能力头和其它意图头。

教师的澄清 F1 为 0.8852，高于小模型的 0.7679，因此再试一次最多 8 轮的专项蒸馏：只迁移“需澄清／不需澄清”，用小模型原输出约束能力和其它意图，并加入训练专用特征投影。投影在部署时丢弃。最终仍以相同验证分数选优，蒸馏前小模型作为候补，不能为了采用蒸馏而强行换模型。

```sh
python3 ml/fetch_permission_teacher.py --output /path/to/teacher-base
python3 ml/train_permission_teacher.py --help
python3 ml/refit_permission_heads.py --help
python3 ml/distill_permission_intent.py --help
```

教师词表与学生词表的两个特殊字符位置不同，教师独立分词后按冻结训练 ID 对齐缓存；学生仍用原词表，另验证其与应用内 WordPiece 的一致性。缓存不是新标注，也不包含保留测试输入。

`product_examples.jsonl` 是按日常功能挑选的 25 条说明性案例，覆盖相机／麦克风、选择器／图库、剪贴板路径、联系人读写、系统生物验证、提醒、位置和歧义。它们不参与训练、验证或选模型，也不冒充真实用户反馈或独立盲测。最终给出对应的新旧预测审核表。
