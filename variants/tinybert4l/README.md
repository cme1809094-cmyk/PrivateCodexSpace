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

1.1.0 的仓库根工程已经同时包含 MiniRBT 和 TinyBERT，默认选择 TinyBERT，可直接切换模型和长度、执行三档 NPU 验收并复制 JSON。详见 [切换与验收](../../docs/model-switch-acceptance.md)。下面的独立工程生成方式仍可用于减少应用体积、使用独立包名；不再是测试 TinyBERT 的必需步骤。

先在仓库根目录执行：

```sh
git lfs pull
python3 ml/materialize_variant.py variants/tinybert4l /absolute/path/to/TinyBERT4LDemo
```

目标目录必须在仓库之外且尚不存在。脚本先校验 `variant.json` 的所有文件，再生成工程。工程内的三档 NPU 模型、CPU 回退、词表、阈值和 HiAI 自检参考均来自 TinyBERT；包名为 `com.intentlab.permissiondemotinybert4l`。测速只包含三档 TinyBERT NPU 和 TinyBERT CPU，不混入 MiniRBT 的旧基线模型。

在生成目录运行 `python3 ml/verify_artifacts.py` 可核对五个模型文件。用 DevEco Studio 打开后配置签名并构建 HAP。成功加载 NPU 模型后应查看 HiAI 自检结果，再测量时延；应用会保留同模型的 CPU 回退。

### NNRT 后端与 native 异常诊断（1.0.9）

1.0.4 将任意 `OH_NN_ACCELERATOR` 都作为 OMC 目标，这个条件不充分。真机日志中的 `[NNBackend] CreateCompiler failed, only support build NN model and NN model cache.` 对应通用 NNRT 后端拒绝离线 buffer；该后端并不等同于 CANN 的 `HIAI_F`。官方 [CANN 模型集成](https://developer.huawei.com/consumer/cn/doc/HarmonyOS-Guides/cannkit-integration-model/) 使用 `HIAI_F` 加载 OMC，[通用 NNBackend 源码](https://github.com/openharmony/ai_neural_network_runtime/blob/master/frameworks/native/neural_network_runtime/nnbackend.cpp) 明确拒绝 offlineModelBuffer。

1.0.5 补充动态加载并保留 `libhiai_foundation.so`，调用 `HMS_HiAI_GetVersion` 后再枚举后端，仅为 OMC 选择 CANN 后端，并在可用时调用模型兼容性检查。Foundation 不可用或 HIAI_F 未注册时，给出明确诊断并使用 CPU 回退。是否能在该手机上注册后端和执行当前 OMC，仍需更新 HAP 后确认，主机测试不代表已恢复实机 NPU 推理。

同时加固 native 边界：FFRT execute/complete 回调捕获标准和未知异常，错误消息使用固定缓冲区；创建 JS Error 失败时不再在 catch 中二次抛 C++ 异常。CMake/Hvigor 显式开启异常及展开信息，移除禁用/忽略异常的选项，并在构建后检查实际库中存在 C++ catch 处理代码。不能仅根据预处理宏判断，因为 `-fignore-exceptions` 仍可能定义 `__EXCEPTIONS`。

1.0.5 的后续日志进入 CANN 模型加载后仍报 `Unknown native exception during NNRT work`，未给出具体失败接口。1.0.6 用 Job 持有的记录保留最后调用和返回状态，并将异常类型、CANN 版本、原生桥接版本放进回退提示。字符串、整数异常保留内容；未知类型不做不安全的异常对象转换。这不等于已解决真机执行问题，仍需新日志确认兼容性检查、Build、张量分配或 RunSync 的实际失败位置。

建议从最新源码重新生成到一个新目录，再配置签名、Clean/Rebuild 并安装1.1.0。已有工程若手工更新，需同步整个 `entry/src/main/cpp/`（含 `native_diagnostics.h`、`tensor_shape.h`、`bridge_error.h`）、`entry/build-profile.json5`、`PermissionModel.ets` 和 AppScope 版本，不能遗漏新头文件或构建后检查脚本。仅替换 `.om` 或只更新 ArkTS 不够。

已确认此次上报设备为 Mate 80 Pro Max / HarmonyOS 5.0.0。驱动名中的 Kirin9020 字样不等同于已验证的 SoC 型号或模型兼容结论；当前产物仍按 kirin9030 编译。名称检查失败发生在模型加载前，更新后需要观察带实际设备名称的 `OH_NNCompilation_Build`、`OH_NNExecutor_RunSync` 和模型自检结果。

调试时在 HiLog 按 Tag `PermissionNPU` 过滤或搜索 `[PNPU]`，日志级别包含 Info/Warn/Error，依次查看 `CANN Foundation loaded, version=...`、HIAI_F 设备信息、模型兼容性、Build 与自检结果。回退提示应包含 `failed at ...; exception_type=...; CANN=...; bridge=1.1.0`。若出现 `Cannot load CANN Foundation`，应先检查系统能力和运行库加载错误，不能通过改设备字符串解决。native 抛出位置若仍需符号化，应使用与崩溃日志 Build ID 完全一致的未剥离 `libpermission_hiai.so`。

1.0.7 的原生与 ArkTS 日志统一采用 `PermissionNPU` Tag 和 `[PNPU]` 消息前缀；先看 `[RUNTIME]`、`[DEVICE]`、`[LOAD]`、`[API]`，再保留首条 `[ERROR]` 与 `[FALLBACK]`。`[SELFTEST]` 表示自检，`[CPU]` 表示 CPU 模型状态。详细张量调用为 Debug，普通推理和测速循环不输出逐次成功日志。命令行可用 `hdc hilog -T PermissionNPU`。

1.0.8 适配 token 输入的等价单例维度表示，同时保留长度 16/32/64、int32、元素数及输入名称检查。`[PNPU][TENSOR]` 和失败提示会记录实际形状；应用自身的校验错误使用显式捕获的 BridgeError 保留原始消息。后续日志已确认 MiniRBT 三档实际输入为 `[1,L,1,1]`，1.0.8 已通过形状校验并完成会话创建。这不能代替 TinyBERT 的实机 RunSync 和参考自检。

1.0.9 修复 N-API TypedArray 长度口径差异：OpenHarmony 5.0 的接口返回字节数 256，旧代码却按 64 个元素校验，导致自检预测在排队前被拒绝。新版核对视图 length/byteLength、类型、缓冲区和偏移后支持两种 API 口径，异步排队前复制数据；以 `[PNPU][INPUT]` 每会话记录一次元数据。69 个主机场景通过，实机 RunSync 和 `[SELFTEST]` 仍需验证。

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
