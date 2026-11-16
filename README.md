# MiniRBT 权限意图 · Mate 80 Pro Max

目标设备为搭载麒麟 9030 Pro 的 Mate 80 Pro Max。移动 CANN 的编译目标为 `kirin9030`。项目使用原 MiniRBT-H256 权重、WordPiece 词表和四类权限及 out_of_scope 标签的阈值。

本工程的模型二进制使用 Git LFS。完整克隆后执行 `git lfs pull`，再执行 `python3 ml/verify_artifacts.py` 校验六个模型文件。工具链配置和编译统一在 Codex 环境进行。

另提供 [中文 TinyBERT 4L 试验版本](variants/tinybert4l/README.md)。它有独立训练得到的分类头、词表、阈值、CPU 回退模型和三档原生 V2 NPU 模型；可通过 `ml/materialize_variant.py` 生成独立包名的 HarmonyOS 工程。

## 模型质量与当前准备程度

`ml/metrics.json` 记录的训练集是 99 条合成中文样本，验证集为 31 条、完全匹配 30 条，测试集为 33 条、完全匹配 24 条（约 72.7%），其中 4 条存在多预测标签。这是既有训练报告，本次没有重新训练或独立复评。标签是 camera、microphone、coarse_location、fine_location 和 out_of_scope。

CPU 改图对照的 27/27、33/33、33/33 表示改图前后的权限输出一致，不代表这些指令的预测全部符合真实标签。当前状态为可继续真机验收的 Demo，尚不能依据离线编译结果判断生产准确率或 NPU 性能。

后续优化优先级：先完成 HAP 构建、签名与 Mate 80 Pro Max 的 HiAI 自检，记录冷启动、端到端耗时、P50/P95、峰值内存与运行时回退；再扩充真实中文、否定、组合权限和歧义指令，独立评估多标签 Precision/Recall/F1、完全匹配率与过度预测。性能优化根据实测选择预热常用档位、限制会话缓存或精简生产包中的旧基线模型。当前已具备分档路由、会话/张量复用、末层 CLS query 裁剪和原生 V2 注意力，无需先把五个输出的阈值计算迁到 NPU。

## 当前原生 V2 结果（2026-11-16）

当前工作副本的 `permission_16.om`、`permission_32.om`、`permission_64.om` 已接入新编译的原生 V2 OMC；`permission_baseline.om` 仍用于旧路径对照。三档各有 12 个 `BatchMatMulV2` 和 6 个 `SoftmaxV2` 的 NPU 任务，最终编译分区均为 `NPUCL=1, CPUCL=0`，无旧版 Softmax 内核。编译成功不代表已通过 Mate 80 Pro Max 的运行时兼容性、自检和性能验收。本次没有重新构建或签名 HAP。

| token | OMC 字节数 | 编译器子图内存（字节） | CPU 权限对照 |
| --- | ---: | ---: | ---: |
| 16 | 22,636,729 | 396,288 | 27/27 |
| 32 | 22,653,914 | 790,016 | 33/33 |
| 64 | 22,686,784 | 1,577,472 | 33/33 |

CPU 对照使用改图后的标准 ONNX，最大 logit 差为 `6.556510925292969e-6`。独立 host tiling 为 24/24 通过，包含 `[512,64]`、`[8,64]` 的 Softmax。报告见 `ml/kirin9030-native-v2-verification.json`，原生 V2 适配、探针和验收脚本见 `ml/native_v2/`。模型 SHA-256 清单见 `ml/artifact-manifest.json`，逐节点编译日志与最终模型转储见 `ml/native_v2/evidence/full/`。原始 ZIP、七个分片和旧模型备份保留在源工作区，不放入 Git。

复现时先初始化 Mobile CANN 与自带 Ascend C 环境，再运行：

```sh
bash ml/native_v2/run_frontend.sh "$PWD" "$CANN_INCLUDE" "$MOBILE_DDK" /path/to/native-build --compile
python3 ml/native_v2/check_host_permissions.py "$PWD" /path/to/native-build/v2_XXXXXX
```

`--compile` 会检查最终 OMC 的逐节点 NPU 任务、原生内核和分区，生成 `native-verification.json`。脚本还沿用 `ml/mask_cast_first.py`，避免 mask 的整数 reshape 引入 CPUCL。正式编译未使用独立 tiling 探针的上下文补丁。

## 历史兼容路径的离线编译

| 模型 | NPUCL | CPUCL | 编译器子图内存 |
| --- | ---: | ---: | ---: |
| 原始 MiniRBT，64 token | 0 | 1 | — |
| 64 token NPU 对照，全查询 | 1 | 0 | 6.57 MiB |
| 16 token，末层只计算 CLS | 1 | 0 | 0.52 MiB |
| 32 token，末层只计算 CLS | 1 | 0 | 1.79 MiB |
| 64 token，末层只计算 CLS | 1 | 0 | 6.57 MiB |

原始 ONNX 在旧版 OMG 中触发超出分区上限的整图 CPU 回退。测试版把 36 个常量权重投影改为 1×1 卷积，同时调整分类头和 LayerNorm 的算子表达。QK、PV 使用等价的 Tile、Mul 和分组卷积。掩码先转浮点再 reshape，并使用 FP16 可表示的 `-10000`。这样可继续使用图里的 Softmax，并编译到 `executor_softmax_5d_w_tiling0_fp16`。分词和权限阈值计算在应用 CPU 端执行。

以下是历史环境记录；当前原生 V2 结果见上文。插件登记了原生 `BatchMatMulV2` 和 `SoftmaxV2`。OMG 重新导入显式 AIR 后仍将两者改写成普通算子，ATC 单算子探针在 InitCannKB 初始化阶段报 `E49999`。编译报告位于 `ml/kirin9030-verification.json`，编译日志节选位于 `ml/kirin9030-compiler.txt`。

主机 ONNX 回归以原模型为参考，33 条留出指令的权限集合全部一致，最大 logit 差为 6.92e-6。三档另通过 112 组随机掩码回归。真机兼容性、NPU 精度和时延尚待验证。

## 安装与测量

1. 用 DevEco Studio 打开此工程，连接 Mate 80 Pro Max，配置调试签名。包名为 `com.intentlab.permissiondemohiaperf`，版本是 1.0.4。本仓库提供源码与模型，需要重新构建并签名 HAP；不包含已更新原生 V2 模型的安装包。
2. 输入一条中文操作指令，查看后端是否显示实际 NNRT 加速器名称，再查看诊断中的模型自检。设备按 `OH_NNDevice_GetType` 返回的 `OH_NN_ACCELERATOR` 选择，不要求名为 `HIAI_F`。出错时应用会显示原因，并允许退到 MindSpore Lite CPU 继续演示。
3. 点击「测量推理时延」。界面里的「旧版 NPU 64 token」指本包内的 9030 全查询对照。各模型使用相同的分词结果，预热 5 次、测量 50 次。p50/p95 含推理接口、张量复制和调度。冷启动、分词和自检位于测速区间之外。

## 历史兼容模型的重新构建

当前原生 V2 的复现命令见前文；下面的 `convert_hiai_9030.sh` 会重新生成历史兼容模型并覆盖三档 rawfile，请仅在需要对照时使用。编译环境使用 Mobile CANN 9.2.0、CANN Kit Tools 7.0.1.0 和对应的 Kirin9030 插件。应用可用 DevEco Studio 构建，命令行使用配套的 OHPM 和 Hvigor。

```sh
python -m pip install onnx==1.19.0 onnxruntime==1.23.2
bash ml/convert_hiai_9030.sh /path/to/tools_omg/omg /path/to/build-directory
ohpm install --all
hvigorw assembleHap --mode module -p product=default -p buildMode=debug --no-daemon
```

[CANN Kit 开发准备](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-preparations) · [Mate 80 Pro Max 规格](https://consumer.huawei.com/cn/phones/mate80-pro-max/specs/)
