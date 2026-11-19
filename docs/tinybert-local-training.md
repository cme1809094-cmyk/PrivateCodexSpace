# 在本地用 NVIDIA Docker 训练 TinyBERT v4

训练包大小为 43,592,568 字节，SHA-256 为 `d71afb7baa70dfa6d77b9d2687c2f91a3173c7b3b060bc42a898a3da6bdc4daa`。[下载训练包](https://drive.google.com/file/d/1rLQw_JBF4ub6o9p0FnGqgsHi5aXhghtX/view?usp=drivesdk)

下载训练包，解压后进入包含 `docker/` 和 `ml/` 的目录。包内的 `inputs/v3/` 是训练起点，包含上一轮配套的权重、配置、词表、协议和 SHA-256 锁定文件。通过 Git 使用本目录时，从上一轮主包提取 `run/checkpoint/` 和 `run/selection_lock.json` 到 `inputs/v3/` 即可。[v3 主包](https://drive.google.com/file/d/1MCWkosDEW5DRMhmzndbtckr7cVqX_079/view?usp=drivesdk)

镜像面向 Linux x86-64（也可在 WSL 2 中构建）。准备能运行 `docker run --gpus all` 的 NVIDIA Container Toolkit，以及与 CUDA 12.8 兼容的 NVIDIA 驱动。构建时会下载 CUDA 版 PyTorch 及其运行库，请预留数 GB 下载空间和约 10 GB 以上可用磁盘空间。教师实验还会下载约 409 MB 的公开权重。

```bash
docker build --platform linux/amd64 -f docker/tinybert-training/Dockerfile \
  -t tinybert-training:v4-cu128 .
bash docker/tinybert-training/run.sh "$PWD/inputs" "$PWD/runs"
```

启动检查会在 CUDA 上实际运行张量运算，打印 GPU 名称、PyTorch 和 CUDA 版本。容器内 `/inputs` 为只读输入，`/runs` 为宿主可写输出，启动脚本按当前用户身份运行容器。学生、教师、分类头重拟合和蒸馏在容器中默认使用 GPU，可用 `--device cuda:0` 等指定可见设备。训练启动日志和 `training_config.json` 的 `runtime` 记录实际设备。

## 先跑一个训练轮次

```bash
bash docker/tinybert-training/run.sh "$PWD/inputs" "$PWD/runs" \
  python ml/train_intent_v4.py --device cuda --variant focal_evidence \
  --initial /inputs/v3/checkpoint --output /runs/gpu-smoke \
  --epochs 1 --min-epochs 1
```

完成后，`runs/gpu-smoke/checkpoint/` 是可由 CPU 或 GPU 加载的配套模型，旁边有 `training_report.json`、`history.jsonl` 和 `selection_lock.json`。再次运行时指定新的输出目录。想显式用 CPU 跑，把启动脚本的第一个参数改为 `--cpu`，并把训练参数改为 `--device cpu`。

## 复跑 v4 的训练步骤

原实验在 Codex 的 Debian Linux x86-64 云端环境进行，使用 4 个 CPU 核、16 GiB 内存配额、Python 3.10.22 和 PyTorch 2.8.0 CPU 版。统一使用 2,849 条训练、623 条验证。以下 GPU 命令使用相同的训练配方和冻结数据，浮点运算和训练结果可能与原 CPU 记录有所差异。学生训练可调 `--batch-size`、`--epochs` 和学习率等。每次换参数仍以验证集选择权重和阈值。

在 Bash 中定义一个简短的启动函数，然后训练两组对照。历史上 `focal_evidence` 的验证分数更高，所以用它继续训练。复跑时先检查本地两份 `training_report.json` 的 `validation_score`，若另一组更高则使用对应的 `--variant` 和 `--initial`。

```bash
trainbox() {
  bash docker/tinybert-training/run.sh "$PWD/inputs" "$PWD/runs" "$@"
}

trainbox python ml/train_intent_v4.py --device cuda --variant pooled \
  --initial /inputs/v3/checkpoint --output /runs/probe-pooled
trainbox python ml/train_intent_v4.py --device cuda --variant focal_evidence \
  --initial /inputs/v3/checkpoint --output /runs/probe-evidence
trainbox python ml/train_intent_v4.py --device cuda --variant focal_evidence \
  --initial /runs/probe-evidence/checkpoint --output /runs/refine-evidence \
  --epochs 10 --min-epochs 6 --patience 4 \
  --learning-rate 1e-5 --head-learning-rate 1e-4
trainbox python ml/refit_permission_heads.py --device cuda --kind student \
  --initial /runs/refine-evidence/checkpoint --output /runs/refine-head-student
```

专项蒸馏是可选阶段。教师权重下载绑定固定版本和 SHA-256。重拟合只有在验证分数提升时才保存候选。若重拟合被拒绝，可用当前最好的已有学生或教师候选。专项蒸馏要求教师的澄清 F1 比所选学生至少高 0.03，未达时保留学生候选。训练后按验证分数比较各个学生候选。

```bash
trainbox python ml/fetch_permission_teacher.py --output /runs/teacher-base
trainbox python ml/train_permission_teacher.py --device cuda \
  --base /runs/teacher-base --output /runs/teacher-nli --epochs 3
trainbox python ml/refit_permission_heads.py --device cuda --kind teacher \
  --base /runs/teacher-base --initial /runs/teacher-nli --output /runs/teacher-head-refit
trainbox python ml/distill_permission_intent.py --device cuda --clarification-specialist \
  --initial /runs/refine-head-student/checkpoint --teacher /runs/teacher-head-refit \
  --output /runs/full-specialist-distilled --epochs 8
```

## 验收和导出

选定最终目录后再运行保留测试。下面以原实验的获选目录为例，使用 CPU 做评估和标准 ONNX 数值对照。已有 v4 的 320 条题适合复现本轮结果。继续优化模型时，应准备新的保留题来判断泛化。

```bash
selected=/runs/full-specialist-distilled
bash docker/tinybert-training/run.sh --cpu "$PWD/inputs" "$PWD/runs" \
  python ml/evaluate_intent_v4.py --run "$selected" \
  --baseline /inputs/v3/checkpoint --output "$selected/evaluation"
bash docker/tinybert-training/run.sh --cpu "$PWD/inputs" "$PWD/runs" \
  python ml/export_intent_contrast.py --run "$selected" \
  --data ml/data/permission_intent_v4/reserved_test.jsonl
```

评估结果在 `evaluation/comparison_report.json`，三档标准 ONNX 和参考向量在 `onnx/`。CUDA 在这里用于电脑训练。HarmonyOS 真机部署需要另行转换 OMC 并完成 NPU 数值验收。[v4 历史训练记录](tinybert-intent-v4-training.md)
