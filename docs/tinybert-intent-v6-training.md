# TinyBERT v6 请求与规避清单

本轮用两张清单评价同一个 TinyBERT 编码器的输出。`requested_capabilities` 是用户当前想做的设备动作。`avoid_capabilities` 是用户当前明确禁止的动作。没有提到的能力、举例中的功能名和已撤回的要求依照上下文标注。两张清单是语义识别结果。实际的范围审查与权限决策仍走独立策略。

新增 376 条受控训练对照。58 条验证题和 80 条保留题分别撰写，训练、验证和保留集在两路都覆盖全部 32 项能力。侧重容易混报的同领域动作，并加入只请求、只禁止、引用功能名、改口和多动作的句子。新数据均为助手编写，人工审核 pending。

从锁定的 v5 双路版本启动，训练集包括 v4 原开发训练、v5 新训练和 v6 新训练，共 3,609 条。保留旧损失，并对每路里分数最高的错误能力加排序惩罚。两路独立校准阈值，选择同时考虑各自 F1 和两张清单完全匹配。冻结的 80 条保留题在锁定候选之后才评分。阈值重校准的 v5 使用原权重，作为从新数据和训练获得收益的对照。[冻结方案](../ml/data/permission_intent_v6/experiment_plan.json)

从 v5 实验包提取 `run/checkpoint/` 与 `run/selection_lock.json` 到 `inputs/v5/`，然后用更新的 NVIDIA Docker 构建配置复跑。[v5 训练起点](https://drive.google.com/file/d/1L37xP565LO9P3eE8HWQ50LGsCha8jI8n/view?usp=drivesdk) · [Docker 指南](tinybert-local-training.md)

```bash
docker build --platform linux/amd64 -f docker/tinybert-training/Dockerfile \
  -t tinybert-training:v4-cu128 .
bash docker/tinybert-training/run.sh "$PWD/inputs" "$PWD/runs" \
  python ml/train_intent_v6.py --initial /inputs/v5/checkpoint --output /runs/v6 --device cuda
bash docker/tinybert-training/run.sh --cpu "$PWD/inputs" "$PWD/runs" \
  python ml/evaluate_intent_v6.py --baseline /inputs/v5/checkpoint --run /runs/v6 --output /runs/v6-evaluation
```

使用同一推理入口查看策略前的两张清单。

```bash
bash docker/tinybert-training/run.sh --cpu "$PWD/inputs" "$PWD/runs" \
  python ml/predict_intent_v5.py --lists-only --checkpoint /runs/v6/checkpoint \
  '请通过系统面板选一张图片，应用不要直接读取图库。'
```
