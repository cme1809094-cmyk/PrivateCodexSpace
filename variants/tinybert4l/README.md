# 中文 TinyBERT 4L · Kirin 9030 试验版本

基座为 `huawei-noah/TinyBERT_4L_zh`，固定 revision `152bf15d86715b41dd89c1e03cf5664963d9b005`。编码器4层、隐藏维度312、12个注意力头、每头26维，接五标签权限分类头，总参数11,420,885。源权重校验见 `source-manifest.json`，许可证和第三方声明保存在本目录。

## 已验证

- 使用原项目99条合成训练样本、31条验证样本和33条测试样本，seed=2026，训练36轮；仅按验证集选择第23轮及阈值。验证集完全匹配26/31，测试集25/33，四类权限过度预测样本3/33。本次未增加真实数据，也未根据测试集调参。
- PyTorch 与原始 ONNX 最大 logit 差为 `5.662441253662109e-6`。
- 三档原生 V2 整图编译均通过。每档8个 BatchMatMulV2、4个 SoftmaxV2 有原厂 tiling 与 NPU 任务记录，最终为 NPUCL=1、CPUCL=0，未生成旧版 Softmax 内核。26维注意力头没有人工补齐。
- 标准 ONNX 改图对照的权限集合分别匹配27/27、33/33、33/33，最大 logit 差为 `3.933906555175781e-6`。
- TinyBERT 的 MindSpore Lite 2.1.0 CPU 模型通过33条对照，最大 logit 差为 `2.4332202e-5`。
- 未修改的应用 WordPiece 实现在本模型33条测试指令上的 input_ids 和 attention_mask 与 BertTokenizerFast 完全一致。

| token | OMC 字节数 | 原生 V2 节点 | 编译分区 |
| --- | ---: | --- | --- |
| 16 | 24,524,711 | 8 BMM + 4 Softmax | NPUCL=1, CPUCL=0 |
| 32 | 24,544,692 | 8 BMM + 4 Softmax | NPUCL=1, CPUCL=0 |
| 64 | 24,584,700 | 8 BMM + 4 Softmax | NPUCL=1, CPUCL=0 |

这些是离线编译和主机对照结果，尚未执行手机上的 OMC。HAP 构建、调试签名、Mate 80 Pro Max 的加载、自检、时延及峰值内存仍待验证。TinyBERT OMC 比当前 MiniRBT 的约22.6 MB略大，4层也不能直接推断为更快。25/33与MiniRBT既有报告24/33的差异很小，验证集表现也不同，不能据此认定TinyBERT更优。

## 生成可供 DevEco 打开的独立工程

先在仓库根目录执行：

```sh
git lfs pull
python3 ml/materialize_variant.py variants/tinybert4l /absolute/path/to/TinyBERT4LDemo
```

目标目录必须在仓库之外且尚不存在。脚本先校验 `variant.json` 的所有文件，再生成工程。工程内的三档 NPU 模型、CPU 回退、词表、阈值和 HiAI 自检参考均来自 TinyBERT；包名为 `com.intentlab.permissiondemotinybert4l`。测速只包含三档 TinyBERT NPU 和 TinyBERT CPU，不混入 MiniRBT 的旧基线模型。

在生成目录运行 `python3 ml/verify_artifacts.py` 可核对五个模型文件。用 DevEco Studio 打开后配置签名并构建 HAP。成功加载 NPU 模型后应查看 HiAI 自检结果，再测量时延；应用会保留同模型的 CPU 回退。

## 重新训练与导出

使用独立 Python 3.10 环境，避免改变 CANN 的 Python 依赖：

```sh
python3 -m venv /absolute/path/to/tinybert-venv
. /absolute/path/to/tinybert-venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r ml/requirements-training-cpu.txt
python3 ml/fetch_tinybert.py /absolute/path/to/tinybert-source
python3 ml/train_export.py \
  --base /absolute/path/to/tinybert-source \
  --source-id huawei-noah/TinyBERT_4L_zh \
  --revision 152bf15d86715b41dd89c1e03cf5664963d9b005 \
  --epochs 36 --threads 4 \
  --output /absolute/path/to/tinybert-training \
  --raw /absolute/path/to/tinybert-project/entry/src/main/resources/rawfile
```

训练输出包含 checkpoint、permission.onnx、metrics.json、golden.json 和 test_vectors.json。重新训练后应将新 ONNX 和 test_vectors.json 配入目标工程的 `ml/permission.onnx`、`ml/test_vectors.json`；词表、阈值及配套的 hiai_reference.json 已写入 `--raw`。不能把新权重与本目录旧阈值、旧自检参考混用。

## 重新编译和对照

在初始化 Mobile CANN 9.2.0、Tools 7.0.1.0、Kirin9030 插件及配套 Ascend C 环境的 shell 中，使用含 onnx/numpy/onnxruntime 的 Python：

```sh
bash ml/native_v2/run_frontend.sh "$TINYBERT_PROJECT" "$CANN_INCLUDE" "$MOBILE_DDK" "$BUILD_ROOT" --compile
python3 ml/native_v2/check_host_permissions.py "$TINYBERT_PROJECT" "$BUILD_ROOT/v2_XXXXXX"
bash "$TINYBERT_PROJECT/ml/convert.sh" "$MINDSPORE_LITE_ROOT" "$TINYBERT_PROJECT/ml/permission.onnx"
python3 "$TINYBERT_PROJECT/ml/verify_ms.py" "$MINDSPORE_LITE_ROOT"
```

`v2_XXXXXX` 应替换为脚本打印的实际构建目录。`--compile` 会输出各档 mobile.omc、最终模型 JSON、逐节点报告和 native-verification.json。编译结束后按清单将三档 mobile.omc 安装为 rawfile/permission_16.om、permission_32.om、permission_64.om。完整日志及转储见 `evidence/`，汇总见 `native-verification.json`。
