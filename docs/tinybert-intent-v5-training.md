# TinyBERT v5 双路证据实验

v4 的能力分类用互斥的三态输出。v5 继续使用同一个四层 TinyBERT、同一套 32 个能力 ID 和整句意图分类头。每项能力现在有三个独立 Sigmoid 分数，分别表示当前请求、当前禁令和范围重叠。第三个分数只在已知同一能力同时出现两种证据的训练样本上监督。原来的能力矩阵形状仍为 `[1, 32, 3]`，新版协议记录了通道次序。旧三态读取器会拒绝这份协议。

词表与编码器继承 v4。同一能力需要与禁止可以同时出现。范围重叠时提示澄清，保留禁令并清空执行列表。高置信的范围分离返回 `review_scope`，因为能力列表无法表达对象、时间等执行约束。若两路都有证据且范围关系不确定，也会澄清。模型返回证据、原因码和可用于审查的能力 ID。实际标注、措辞和平台执行授权仍由上层承担。

v4 的有效执行名单和禁令名单继续用于评估。训练额外保存 `requested_evidence` 和 `prohibited_evidence`，以便在最终执行为空时仍能学习原始动作。256 条 v4 受控冲突训练句的原始动作为模板已记录信息。其它历史歧义句中未知的请求证据不计入这一路损失。

## 数据与验证

新增训练 384 条，验证 36 条，保留 64 条。新保留集有 16 个场景，每组两种冲突表达加两个非冲突对照。对照包含不同能力、同一能力不同范围和明确改口。训练和验证共 3,233／659 条。保留集正文在训练前冻结，两个训练脚本只验证预先记录的文件哈希，选定权重及阈值后才进行最终测量。新测试为助手撰写，现等真人审查。v4 的旧 320 条视为已知回归。

对照原始 v4、v4 权重继续微调但保留三态的 `data_only`，以及同数据上的 `dual_evidence`。两个训练候选使用相同的种子、固定四轮训练预算、批量与学习率。经过开发验证选轮与校准后，三个锁定模型一起评估。[冻结方案](../ml/data/permission_intent_v5/experiment_plan.json)

## 本地复跑

把 v4 主包内的 `run/checkpoint/` 和 `run/selection_lock.json` 放入 `inputs/v4/`。Docker 构建使用现有 NVIDIA 训练环境。[v4 主包](https://drive.google.com/file/d/1hfTwQ16wuDVt2fgHACx5ECSN3wyehSPr/view?usp=drivesdk) · [Docker 指南](tinybert-local-training.md)

```bash
docker build --platform linux/amd64 -f docker/tinybert-training/Dockerfile \
  -t tinybert-training:v4-cu128 .

for variant in data_only dual_evidence; do
  bash docker/tinybert-training/run.sh "$PWD/inputs" "$PWD/runs" \
    python ml/train_intent_v5.py --variant "$variant" --device cuda \
    --initial /inputs/v4/checkpoint --output "/runs/v5-$variant"
done

bash docker/tinybert-training/run.sh --cpu "$PWD/inputs" "$PWD/runs" \
  python ml/evaluate_intent_v5.py --baseline /inputs/v4/checkpoint \
  --data-only /runs/v5-data_only --dual /runs/v5-dual_evidence --output /runs/v5-evaluation
```

用已训练的 checkpoint 推理，再导出 64 token 静态 ONNX。

```bash
bash docker/tinybert-training/run.sh --cpu "$PWD/inputs" "$PWD/runs" \
  python ml/predict_intent_v5.py --checkpoint /runs/v5-dual_evidence/checkpoint \
  '禁止修改通讯录，但请更新这一位联系人的备注。'
bash docker/tinybert-training/run.sh --cpu "$PWD/inputs" "$PWD/runs" \
  python ml/export_intent_v5.py --checkpoint /runs/v5-dual_evidence/checkpoint \
  --output /runs/v5-dual_evidence/onnx --tokens 64
```
