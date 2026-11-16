# MiniRBT 原生 V2 算子复现

目标为 Kirin 9030 插件与 Mate 80 Pro Max。模型使用已有的 MiniRBT 权重，序列长度为 16、32、64，注意力共 6 层、8 个头，末层只计算 CLS query。

本脚本现已通过 `ml/inspect_bert.py` 从原始 ONNX 读取 BERT 层数、隐藏维度、注意力头数及每头维度，再执行末层 CLS 裁剪和逐节点验收。已验证的另一种结构为中文 TinyBERT 4L（4层、12头、每头26维）；其记录见 `variants/tinybert4l/`，每档应有8个 BatchMatMulV2 和4个 SoftmaxV2。下文原始表格与历史探针仍描述 MiniRBT；新结构需要使用对应模型的测试向量、阈值、词表及 CPU 回退模型。

| 检查 | 实测结果 |
|---|---|
| OMG 导入 | 三档中间图均保留 12 个 `BatchMatMulV2` 和 6 个 `SoftmaxV2`。检查覆盖 FP16、形状与算子属性。旧 OMG 的导出 JSON 将这些描述符标为 `NCHW`，最终内核使用的布局待验证。 |
| 原厂 `BatchMatMulV2` host tiling | 18/18 通过，覆盖 QK、PV、末层 CLS 和右矩阵转置对照。`block_dim=1`，`tiling_key` 为 1 或 2。 |
| 原厂 `SoftmaxV2` host tiling | 0/6 通过。`SoftmaxV2` 收到 FP16 ND 的正确形状，计算分块时报 `CoreNum is 0`。插件配置的 `vector_core_cnt=1`，编译器实际传入的 tiling 上下文待验证。 |
| ATC 单算子 | 修正解包环境下 `libcann_kb.so` 的查找后，`Add` 和 `SoftmaxV2` 均在 `InitCannKB` 初始化失败。日志记录了 `getifaddrs` 返回 `EPERM`。`CANN_KNOWLEDGE_BANK_PROCESS_NUM=0` 也未使 `Add` 通过。 |
| OMG 最终 OMC | 三档都在 Ascend C 适配器初始化时遇到 `AF_UNIX` 创建失败，随后兼容性检查拒绝 V2。 |
| ONNX Runtime 数值对照 | 用同一改图的标准 ONNX 算子版本与原始模型比较，三档权限集合分别匹配 27/27、33/33、33/33，最大 logit 误差约 6.56e-6。 |

`evidence/` 包含验证报告与日志摘录。当前验证范围止于中间图、离线 tiling 和 CPU 对照。

## 当前 Codex 环境复核

上述表格保留此前环境的记录。当前工作区的 `getifaddrs` 和 `AF_UNIX` 检查均通过；原始探针仍复现了 `BatchMatMulV2` 18/18 通过、`SoftmaxV2` 0/6 通过。

平台资源已向 tiling 上下文提供 `SoCInfo.vector_core_cnt=1`。进一步检查发现，原厂 Softmax tiler 使用传入的 `blockDim` 作为可用核数预算，而独立工具的初始值是 0。`plugin/native_tiling_schema.cpp` 现在仅在初始值为 0 时，从实际平台资源初始化预算，再调用未修改的原厂 tiling 函数。已有的非零预算保持原值。修改后 24/24 组 host tiling 通过，6 组 Softmax 的 `block_dim=1`、`tiling_key=2`、tiling 数据均为 56 字节。

使用本包原有的 ONNX 别名适配插件，6 组相同 FP16 形状的独立 Softmax 图均生成 OMC，日志及产物确认绑定 `SoftmaxV2_d50aa7c67f654684e7eb33e1962150d8_2` NPU 内核，内核大小 7640 字节。64-token 全序列形状还以 `use_origin_format=false` 做了对照，也绑定同名内核。因此不能把原生路径的成功仅归因于 `use_origin_format=true`。

完整工程现已通过 Google Drive 的 7 个分片恢复，原始 ZIP 的 SHA-256 为 `3c95eb293e647a6804e18558c731b16e23ce1a54739b125505a4ee6a1455018d`，与交接清单一致，ZIP CRC 检查通过。

2026-11-16 的整网验证已通过：16、32、64 档均生成 `mobile.omc`；逐节点日志确认每档 12 个 `BatchMatMulV2` 和 6 个 `SoftmaxV2` 的原厂 tiling 成功且生成 NPU RuntimeConfig，内嵌原生 V2 内核，未生成旧版 `executor_softmax` 内核。最终模型转储均为 `NPUCL=1, CPUCL=0`，两路输入为 `[1,L]` INT32，输出为 `[1,5]` FP32。这里的 CPUCL 数量描述编译分区，不代表应用端分词、阈值判断和 I/O 转换不使用 CPU。

首轮整网出现了 mask 路径的 CPUCL 分区。`run_frontend.sh` 已接入原工程的 `mask_cast_first.py`，在 reshape 前对 rank-2 mask 转浮点，重新编译后去除了该分区。独立探针的 blockDim 初始化修复只作用于诊断上下文；正式编译没有加载该诊断库，也没有修改原厂编译器或内核。

同一次改图的标准 ONNX 版本在 CPU 上与原始模型比较，三档权限集合分别匹配 27/27、33/33、33/33，最大 logit 误差均为 `6.556510925292969e-6`。这不是 OMC 的真机推理结果。真机加载、精度、时延与运行时回退仍待验证。当前 ATC Add 单算子通过初始化后仍在 `RunCustomPass` 阶段失败；已通过的移动 OMC 编译使用直接 OMG 路径。后续工具链与编译统一在 Codex 环境完成；本次验证主机为 Debian 13 x86-64。

本次控制组、修复后 host tiling 报告及独立 OMC 内核绑定结果保存在 `evidence/codex-*.json`。

## 复现

沿用 MiniRBT 的 [Kirin 9030 项目](../../README.md)，脚本从其 `ml/permission.onnx` 重建 3 档注意力图。通过华为官方安装流程在 Ubuntu x86-64 安装 CANN 9.2.0 Mobile Station、CANN Kit Tools 7.0.1.0 和 Kirin9030 插件，加载安装目录的 `set_env.sh`，并为 OMG 初始化 DDK 自带的 Ascend C 工具环境。ATC 和 OMG 可以在分别初始化的 shell 中运行。使用包含 `onnx`、`numpy` 的 Python，CPU 对照另需 `onnxruntime`。

- `PROJECT` 为原项目目录，包含 `ml/permission.onnx`。
- `CANN_INCLUDE` 为包含 `register/op_def_registry.h` 的 CANN 头文件目录。
- `MOBILE_DDK` 为包含 `tools/tools_omg/master` 和 `tools/platform/kirin9030` 的目录，常见位置为 CANN 安装目录下的 `omg`。
- `ATC_EXT_BIN` 为已安装的 `atc_ext.bin`。单算子探针直接调用二进制。
- 可用 `PYTHON_BIN` 指定 Python。脚本默认使用 Linux x86-64 的 ELF 加载器，可以用 `OMG_LOADER` 指定其他对应的加载器。

```bash
bash probe_atc.sh "$ATC_EXT_BIN" ./build
bash run_frontend.sh "$PROJECT" "$CANN_INCLUDE" "$MOBILE_DDK" ./build
python3 probe_tiling.py "$CANN_INCLUDE" "$MOBILE_DDK" ./build
bash run_frontend.sh "$PROJECT" "$CANN_INCLUDE" "$MOBILE_DDK" ./build --compile
```

`probe_atc.sh` 先运行 `Add`，成功后检查 64 档的全序列和 CLS 单算子。其 `host.log` 会记录网卡枚举与 AF_UNIX 检查结果。 `probe_tiling.py` 运行 24 组，部分用例失败时仍生成 `report.json`，并返回非零退出码。诊断插件通过匹配版本的 C++ ABI 调用原厂 tiling 函数。

`run_frontend.sh` 的 ONNX 适配插件将两个别名映射到 V2 类型。它只适用于这份 MiniRBT 图中用到的 FP16 形状与属性。默认输出 `frontend.om` 和逐节点验收报告，`--compile` 还会生成 `mobile.omc`，调用 `inspect_mobile.sh` 转储最终模型，并用 `verify_native_omc.py` 验证所有 18 个 V2 节点、原生内核和 NPU/CPU 分区，输出 `native-verification.json`。CPU 对照可用 `python3 check_host_permissions.py "$PROJECT" <run_frontend.sh 打印的构建目录>` 复现。需要原工程包含 `ml/mask_cast_first.py`。

原生路径的验收需要从编译日志或图转储中逐节点确认 18 个 V2 节点均绑定 NPU 内核，并在真机核对权限输出与推理耗时。中间图和 tiling 结果只覆盖各自阶段。

[华为 CANN Kit 安装文档](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-preparations) · [ONNX 适配插件文档](https://developer.huawei.com/consumer/cn/doc/HarmonyOS-Guides/cannkit-onnx-framework/)
