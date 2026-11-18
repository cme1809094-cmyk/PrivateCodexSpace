# MiniRBT / TinyBERT 权限意图 · Mate 80 Pro Max

目标设备为搭载麒麟 9030 Pro 的 Mate 80 Pro Max / HarmonyOS 5.0.0。移动 CANN 的编译目标为 `kirin9030`。1.1.2 在同一个应用中支持 MiniRBT-H256 与中文 TinyBERT 4L 切换，每个模型均有自动／16／32／64 token 选择，默认 TinyBERT。切换时同步更换权重、词表、五标签阈值、参考输出和 CPU 回退模型。

本工程的模型二进制使用 Git LFS。完整克隆后执行 `git lfs pull`，再执行 `python3 ml/verify_artifacts.py` 校验十个模型文件、模型目录和配套元数据。工具链配置和离线编译统一在 Codex 环境进行。已在 Codex 云端安装固定版 HarmonyOS 5.0.0.71 / API 12 工具链，1.1.1 完整干净构建和包内校验通过，已生成未签名 HAP；签名与真机执行仍待验证。复现入口见 [云端构建](tools/harmony/README.md)。

直接在 DevEco Studio 打开仓库根目录即可构建双模型应用，不再需要生成独立工程才能测试 TinyBERT。MiniRBT 资源保留在 rawfile 根目录，TinyBERT 位于 `rawfile/models/tinybert4l/`，身份和配套文件校验值见 `model_catalog.json`。仍可通过 `ml/materialize_variant.py` 生成独立包名的 [TinyBERT 工程](variants/tinybert4l/README.md)。

点击「验收 TinyBERT 4L 三档 NPU」逐档检查 27/27、33/33、33/33 个配套参考输出，检查最大绝对 logit 差，并测量预热后 P50/P95；结果可复制为 JSON。验收禁止应用层 CPU 回退。操作、判定标准、已通过的主机检查和待完成的手机检查见 [切换与验收说明](docs/model-switch-acceptance.md)，切换的主机证据见 `ml/model-switch-verification.json`；1.1.1 的实际 SDK 构建证据见 `ml/harmony-build-verification.json`。

TinyBERT手机日志已进入数值自检阶段：16档26/27、32／64档30/33，当前会回退至同模型CPU。1.1.2增加逐样本logit、标签差异和手机实际API版本诊断，并接入三档LayerNorm等价缩放OMC候选；训练权重和验收门槛不变，尚待手机复测。SDK版本区别、主机精度线索与后续验收见 [TinyBERT NPU一致性调查](docs/tinybert-npu-parity.md)。

## 模型质量与当前准备程度

下一阶段的[首批能力字典与代表性指令](ml/data/permission_intent_v2/README.md)覆盖 12 个新增能力组，加上已有相机、麦克风和定位，共 32 项能力及 300 条种子指令。字典分别记录 HarmonyOS API 12 与 Android 的授权路线、版本条件和待核实限制；语料区分任务所需、明确禁止、未提及以及需要追问的情况。材料经 AI 交叉复核，人工审核仍待完成，尚未用于重新训练或扩展应用当前的五标签输出。可用 `python3 ml/validate_permission_intent_v2.py --render` 复查并生成两份人审 CSV。

[扩充版](ml/data/permission_intent_v2/expansion/README.md)再加入 1200 条，合计 1500 条。新增部分按 240 个场景族组织为 750 条训练候选、150 条验证候选和 300 条测试候选，同一场景的五种改写留在同一集合；旧 300 条只作参考。2026-11-18已完成[实验性TinyBERT三态微调](ml/results/tinybert-intent-v2-20261118/README.md)：排除上下文题并补齐验证意图类别后，使用705/180/295条，选第43轮；内部测试需要F1为44.1%、禁止F1为54.9%，全状态和意图同时正确44/295，泛化不足，未接入五标签应用。三档标准ONNX的CPU数值对照和1500条分词一致性通过；未编译新OMC或执行手机验收。人工审核与独立基准冻结仍待完成。[测试集的通俗解释](docs/permission-intent-test-design.md)和[训练复现](docs/tinybert-intent-v2-training.md)分别说明数据与训练流程。

`ml/metrics.json` 记录的训练集是 99 条合成中文样本，验证集为 31 条、完全匹配 30 条，测试集为 33 条、完全匹配 24 条（约 72.7%），其中 4 条存在多预测标签。这是既有训练报告，本次没有重新训练或独立复评。标签是 camera、microphone、coarse_location、fine_location 和 out_of_scope。

CPU 改图对照的 27/27、33/33、33/33 表示改图前后的权限输出一致，不代表这些指令的预测全部符合真实标签。当前状态为可继续真机验收的 Demo，尚不能依据离线编译结果判断生产准确率或 NPU 性能。

后续优化优先级：在已通过的 HAP 构建基础上，先完成签名与 Mate 80 Pro Max 的 HiAI 自检，记录冷启动、端到端耗时、P50/P95、峰值内存与运行时回退；再扩充真实中文、否定、组合权限和歧义指令，独立评估多标签 Precision/Recall/F1、完全匹配率与过度预测。性能优化根据实测选择预热常用档位、限制会话缓存或精简生产包中的旧基线模型。当前已具备分档路由、会话/张量复用、末层 CLS query 裁剪和原生 V2 注意力，无需先把五个输出的阈值计算迁到 NPU。

## 当前原生 V2 结果（2026-11-16）

当前工作副本的 `permission_16.om`、`permission_32.om`、`permission_64.om` 已接入新编译的原生 V2 OMC；`permission_baseline.om` 仍用于旧路径对照。三档各有 12 个 `BatchMatMulV2` 和 6 个 `SoftmaxV2` 的 NPU 任务，最终编译分区均为 `NPUCL=1, CPUCL=0`，无旧版 Softmax 内核。编译成功不代表已通过 Mate 80 Pro Max 的运行时兼容性、自检和性能验收。该段为原生 V2 离线编译记录；1.1.1 已完成未签名 HAP 构建，仍未签名或上机。

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

1. 用 DevEco Studio 打开此工程，连接 Mate 80 Pro Max，配置调试签名并 Clean/Rebuild native 和 ArkTS。包名为 `com.intentlab.permissiondemohiaperf`，版本是 1.1.2。本仓库提供源码与模型，需要重新构建并签名 HAP。
2. 输入一条中文操作指令，检查实际 CANN 后端和模型自检。OMC 需要 CANN 的 `HIAI_F` 后端；通用 NNRT 加速器存在并不代表它支持离线 OMC。应用先加载 `libhiai_foundation.so` 并查询版本，再枚举设备和检查模型兼容性。失败时应用显示原因，并退到配套的 MindSpore Lite CPU 模型。
3. 点击「测量推理时延」。界面里的「旧版 NPU 64 token」指本包内的 9030 全查询对照。各模型使用相同的分词结果，预热 5 次、测量 50 次。p50/p95 含推理接口、张量复制和调度。冷启动、分词和自检位于测速区间之外。

根目录工程包含两个模型，包名为 `com.intentlab.permissiondemohiaperf`；通过 `variants/tinybert4l` 生成的独立工程只包含 TinyBERT，包名为 `com.intentlab.permissiondemotinybert4l`。切换后识别结果明确显示模型名、实际后端和 token 档位。自动模式尝试能容纳输入的 NPU 档位；手动模式只尝试指定 NPU 档位，失败后可回退到当前模型的 CPU，不会暗中改用另一 NPU 档位。超过手动长度时直接提示，不截断。

1.0.5 的后续真机日志已出现 CANN / AI_NPUCL 的模型加载记录，但三档随后都回退为 `Unknown native exception during NNRT work`。这条消息来自桥接层的 `catch (...)`，只说明未匹配 `std::exception`，不能据此确定是 Softmax、芯片目标、权限或某一个 NNRT 接口失败。附件缺少原生 Info 日志，`hiai foundation not support extension config` 与 `hiappevent` 的目录错误也不足以单独证明根因。

1.0.6 修复诊断信息丢失：回退提示直接保留 `Load OMC` / `Predict`、最后调用的 API、返回状态、异常 ABI 类型、CANN 版本和原生桥接版本。字符串和整数异常保留原值；未知异常只读取类型信息，不强制转换异常对象。构造器展开和资源清理不会覆盖原始位置。正常推理不输出逐调用日志。当前版本更新后 Clean/Rebuild，按下文的 `PermissionNPU` 标签过滤；界面的完整回退提示应带 `failed at ...; exception_type=...; CANN=...; bridge=1.1.2`。这是定位增强，尚未确认修复真机 NPU 执行失败。

补充的启动日志显示 `init hiai foundation success` 和内部版本 `109.633.120.010`，同时有 `nnrt_device_service` 获取失败及后端注册失败。[NNRT 的 HDI 注册源码](https://github.com/openharmony/ai_neural_network_runtime/blob/master/frameworks/native/neural_network_runtime/register_hdi_device_v2_0.cpp) 在设备服务不可用时返回空后端，[注册管理器](https://github.com/openharmony/ai_neural_network_runtime/blob/master/frameworks/native/neural_network_core/backend_manager.cpp) 随后输出对应注册错误；这本身不证明独立注册的 CANN HIAI_F 也不可用。需要实际枚举结果和同一次模型请求的失败步骤来确定。内部版本日志也不能代替 `HMS_HiAI_GetVersion` 的桥接层输出。

## 输入形状适配（1.0.8）

1.0.7 的过滤日志已确认 MiniRBT 三档均枚举到 HIAI_F，运行库版本为 `109.633.120.010`，兼容性检查和 Build 均返回 0，Executor 创建成功。`GetShape` 也返回 0，接下来在应用自己的输入形状校验处抛出 `std::runtime_error`；最后记录的成功 API 并不是失败 API。当前错误发生在输入校验，尚未调用 RunSync。此前关于后端未注册或 Build 失败的阻塞在这次日志中已不成立。

旧代码只接受二维 `[1,L]`。1.0.8 支持同一个连续 token 向量的等价单例维度表示，如 `[L]`、`[1,L]`、`[1,L,1,1]`、`[1,1,1,L]`；L 仍只能为 16/32/64，rank 为 1–8，且只能有一个非 1 的维度。`[2,8]`、动态/零维度、多个非单例轴等仍被拒绝。输入 int32 类型、元素数、ids/mask 长度一致和名称检查保留，分配张量继续使用原始描述，不修改或重新编译 OMC。

新版在校验前以 `[PNPU][TENSOR]` 记录实际 rank 和 shape，失败提示也保留该上下文。后续 1.0.8 真机日志已确认三档的两个输入都是 `[1,L,1,1]`，形状和类型校验通过，输入/输出张量创建成功，并显示 LOAD ready。空描述或其他非等价形状仍会明确拒绝。该手机上 `std::runtime_error` 未被 `std::exception` 分支识别，导致原始提示丢失；具体 C++ 运行库原因尚未证实。应用自有校验改用独立 `BridgeError` 并显式捕获，保留消息，未知外部异常继续安全回退。

主机已验证三档、四种表示的实际桥接加载/复制/推理回调，共 12 个往返场景，以及非等价形状、空 shape、长度和元素数不符的拒绝。真机 RunSync 与参考自检需在完成下文 TypedArray 修复后继续验证。此次日志的模型大小对应仓库根目录 MiniRBT；TinyBERT 的运行时兼容性尚未由这些日志证明。

## TypedArray 长度口径适配（1.0.9）

1.0.8 已完成三档 MiniRBT 会话创建，随后自检首次调用预测时在 N-API 参数读取处报 `Expected an Int32Array of length 64`，尚未排入工作线程或调用 RunSync。原代码假定 `napi_get_typedarray_info` 的 length 为元素数。[OpenHarmony 5.0 实现](https://github.com/openharmony/arkui_napi/blob/OpenHarmony-5.0.0-Release/native_engine/native_api.cpp) 实际赋值为 `typedArray->ByteLength(vm)`，即 64 个 int32 对应 256 字节；[HarmonyOS 文档](https://developer.huawei.com/consumer/en/doc/harmonyos-references/napi/) 也说明原生/导出符号的长度口径差异。这解释了有效 64 元素输入被旧检查拒绝。

1.0.9 保持 ArkTS `Int32Array(64)` 接口，读取视图的 `length=64`、`byteLength=256`，并验证 int32 类型、底层缓冲区容量及字节偏移。仅在视图属性正确时兼容 API 返回元素数 64 或字节数 256；256 个元素的 Int32Array、其他类型、短缓冲区和越界偏移仍拒绝。从底层 buffer + byteOffset 复制固定 256 字节到 Job 自有数据后再排队，避免把 JS 缓冲区指针带入异步线程。正常输入元数据以 `[PNPU][INPUT]` 每会话记录一次，失败时记录具体字段，不记录 token 值。

新增回归调用真正的导出 Predict 回调，覆盖三档、两种长度口径、零/非零偏移，并在排队后清空 JS 缓冲区来验证复制和生命周期；另验证 13 个无效视图不会进入推理。合计 69 个主机场景通过。仍需安装 1.0.9，确认 `[INPUT]` 元数据、RunSync 与 `[SELFTEST]` 结果；主机 mock 不能证明实机 NPU 数值或性能。

## 日志过滤（1.0.7）

Native 和 ArkTS 共用 HiLog Tag **`PermissionNPU`**、Domain `0xFF00`；消息统一以 **`[PNPU]`** 开头。在 DevEco HiLog 中按 Tag `PermissionNPU` 过滤并显示 Info/Warn/Error，或在消息中搜索 `[PNPU]`。命令行可用：

```sh
hdc hilog -T PermissionNPU
```

| 消息标签 | 含义 |
| --- | --- |
| `[PNPU][RUNTIME]` | Foundation 加载、版本或运行库错误 |
| `[PNPU][DEVICE]` | 原生桥接版本、实际枚举设备 |
| `[PNPU][LOAD]` | 模型文件、token 档位、加载开始及完成 |
| `[PNPU][MODEL]` / `[PNPU][ACCEPT]` | 选中的模型／逐档验收结论 |
| `[PNPU][INPUT]` | N-API 输入类型、两种长度、偏移和缓冲区范围 |
| `[PNPU][TENSOR]` | 实际输入 rank 和 shape，校验失败上下文 |
| `[PNPU][API]` | 兼容性检查、Build、Executor 构造的调用及返回 |
| `[PNPU][ERROR]` | 原始失败接口、返回状态、异常类型和运行库版本 |
| `[PNPU][FALLBACK]` | ArkTS 回退原因，包含 token 档位 |
| `[PNPU][SELFTEST]` | 自检匹配数及最大 logit 差 |
| `[PNPU][CPU]` / `[PNPU][BENCH]` | CPU 模型加载状态 / 测速错误 |

定位当前问题时，保留从 `[RUNTIME]` 到第一条 `[ERROR]` 和 `[FALLBACK]` 的记录即可。张量描述、内存等详细 API 日志降为 Debug，原生 worker 的同一异常不再在 Promise 拒绝时重复输出。普通推理和测速循环不记录逐次成功日志，也不记录用户输入文本。旧版本使用的 `PermissionHiAI` 标签和 `JSAPP` console 回退输出已统一为上述标签；需重新构建 native 和 ArkTS 后生效。

## 历史兼容模型的重新构建

当前原生 V2 的复现命令见前文；下面的 `convert_hiai_9030.sh` 会重新生成历史兼容模型并覆盖三档 rawfile，请仅在需要对照时使用。编译环境使用 Mobile CANN 9.2.0、CANN Kit Tools 7.0.1.0 和对应的 Kirin9030 插件。应用可用 DevEco Studio 构建，命令行使用配套的 OHPM 和 Hvigor。

```sh
python -m pip install onnx==1.19.0 onnxruntime==1.23.2
bash ml/convert_hiai_9030.sh /path/to/tools_omg/omg /path/to/build-directory
ohpm install --all
hvigorw assembleHap --mode module -p product=default -p buildMode=debug --no-daemon
```

[CANN Kit 开发准备](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-preparations) · [Mate 80 Pro Max 规格](https://consumer.huawei.com/cn/phones/mate80-pro-max/specs/)
