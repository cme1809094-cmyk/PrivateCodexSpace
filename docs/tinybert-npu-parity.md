# TinyBERT NPU 一致性偏差与 API 版本 · 1.1.2

## 当前手机结果

下表来自用户上报的汇总日志，不是云端设备测量。尚未取得完整逐样本 NPU 输出和手机 ROM/API/CANN 版本。

| token 档位 | 权限集合与参考一致 | 最大绝对 logit 差 |
| --- | --- | ---: |
| 16 | 26/27 | 0.974641 |
| 32 | 30/33 | 1.038904 |
| 64 | 30/33 | 1.038904 |

这些数字表示与原始模型参考结果的一致性，不是对真实标签的准确率。32/64档的计数和最大偏差相同，尚不能证明失败的是同三条指令。日志说明已执行到输出数值自检；没有证据将此问题归因于 API 12、后端枚举失败或某个指定 NPU 内核。

当前回退符合门槛：参考覆盖数量完整、五标签集合全部一致、最大绝对 logit 差≤0.1必须同时成立。仅放宽0.1仍不能解决30/33的标签差异。1.1.2保持原训练权重、词表、分类阈值、参考输出、CPU回退模型和该门槛不变；三档OMC包含下面的LayerNorm等价缩放候选改写。

## API 12 和新版本

2026-11-17核查华为官方文档：

- HarmonyOS 5.0.0对应API 12，5.1.0对应API 18。
- 官方已列出26.0.0 Release，发布日期2026-08-29。
- compileSdkVersion决定编译工具与可用API；targetSdkVersion决定部分系统行为；compatibleSdkVersion决定最低安装版本，三者需满足 compatible ≤ target ≤ compile。
- 可以升级编译SDK，同时保留较低的最低兼容版本，并对新接口做运行时保护。当前云端复现环境、target和最低兼容均固定API 12，是沿用此前提供的HarmonyOS 5.0.0设备信息，并不表示API 12是最新版本。

应用编译SDK、CANN离线模型编译工具、手机内置CANN/NPU驱动是不同版本链。升级应用SDK不会自动重新编译OMC或升级手机驱动，不能据此宣布数值问题已解决。先取得手机的真实版本与逐样本差异，再决定版本升级和模型精度实验。

官方参考：[API 26.0.0 Release](https://developer.huawei.com/consumer/en/doc/harmonyos-releases/overview-2600/)、[API 18版本对应](https://developer.huawei.com/consumer/cn/doc/harmonyos-releases/overview-510/)、[编译／目标／最低兼容SDK](https://developer.huawei.com/consumer/en/doc/harmonyos-releases/app-compatibility-influence-factor/)。

## 主机数值线索

对既有64 token标准改图ONNX运行逐节点FP16往返舍入探针，关闭ORT图优化，使用配套的33条测试向量和TinyBERT阈值：

| 舍入策略 | 权限一致数 | 最大 logit 差 |
| --- | --- | ---: |
| 原始改图，未增加舍入 | 33/33 | 0.000004649 |
| 仅权重按FP16舍入 | 33/33 | 0.006878 |
| 权重和Conv输出按FP16舍入 | 33/33 | 0.007355 |
| 权重及LayerNorm内部结果按FP16舍入 | 30/33 | 1.864163 |
| 权重及其他结果按FP16舍入，保留LayerNorm内部FP32 | 33/33 | 0.012079 |
| 权重及全部浮点结果按FP16舍入 | 30/33 | 1.862334 |

该探针支持“LayerNorm内部精度敏感”的调查方向。主机失败编号为5、26、31，尚不能作为手机的失败编号。最大误差也与手机不同；初步16档全舍入对照为24/27，与手机26/27不同。因此它不是Kirin硬件、累加精度或具体内核的仿真，不能作为已经定位或修复手机问题的证据。

复现：

~~~sh
python3 ml/native_v2/probe_fp16_sensitivity.py /path/to/l64/permission_native.onnx \
  --vectors variants/tinybert4l/test_vectors.json \
  --config variants/tinybert4l/model_config.json \
  --output /tmp/tinybert-fp16-sensitivity.json
~~~

需要onnx、onnxruntime、numpy；输入应为标准ONNX，不能使用含自定义导入算子的adapted.onnx。探针不会修改模型或安装产物。此版Mobile OMG帮助虽列出mode=6，实际调用返回“不支持mode 6”；未获得厂商精度仿真结果。

## LayerNorm候选修复

读取原始改图的LayerNorm中心化张量后，在有效token上测得最高绝对值354.5598；FP16平方能表示的输入上界约为sqrt(65504)=255.9375。这个模型存在具体的FP16平方溢出风险。云端观测不是手机中间张量，实际NPU内核是否发生同样的溢出仍须通过复测确认。

令d=x-mean(x)、s=0.25，把标准化改成：

~~~text
(s*d) / sqrt(mean((s*d)^2) + s*s*epsilon)
~~~

分子与分母同步缩放、epsilon乘0.0625，保持FP32数学语义。0.25为二进制精确幂，不改gamma/beta。平方的中心化输入范围提高到约1023.75；这是类型范围推导，不代表所有未知输入都已验证。对TinyBERT的9处LayerNorm应用该改写，保留原生注意力与五个FP32输出。

三档新候选已重新编译；标准FP32改图对照仍为27/27、33/33、33/33，最大logit差3.9339066e-6。逐节点FP16舍入对照恢复到27/27、33/33、33/33，最大差分别0.012079、0.012079、0.011478。每档保持8个原生BatchMatMulV2、4个SoftmaxV2，最终NPUCL=1、CPUCL=0。

| token | 新候选OMC字节数 |
| --- | ---: |
| 16 | 24,563,743 |
| 32 | 24,583,752 |
| 64 | 24,623,760 |

新二进制已同步到variants和双模型rawfile，身份清单已更新；旧二进制仍在此前Git LFS版本中。已在云端重新构建包含新候选的1.1.2未签名HAP，并验证包内文件SHA-256。下一步必须在手机上重新验收，仍以全部标签一致且最大差≤0.1为准。

复现候选编译时，先初始化Mobile CANN及其自带Ascend C环境，然后设置：

~~~sh
export NATIVE_LAYERNORM_SCALE=0.25
bash ml/native_v2/run_frontend.sh "$TINYBERT_PROJECT" "$CANN_INCLUDE" "$MOBILE_DDK" "$BUILD_ROOT" --compile
~~~

不设置该变量时保留原流水线，MiniRBT二进制在本轮未改变。动态范围检查可用probe_layernorm_range.py；改写脚本stabilize_layernorm.py会验证平方和归一化分子共享同一个中心化输入，并拒绝未识别的结构。

## 1.1.2 获取下一步所需数据

已在API 12真实SDK下完成干净构建，包含ArkTS、AArch64原生桥接和未签名HAP打包，并通过包内模型与元数据校验。22组主机路由回归通过，包括模拟26/27、30/33、30/33时保留样本编号、错误中断时保留部分结果、系统版本读取失败不影响NPU结论。主机mock不代表手机数值通过。

拉取最新分支、Clean/Rebuild并签名安装1.1.2，选择TinyBERT后执行三档NPU验收，再复制验收JSON。JSON schemaVersion=2保留旧汇总字段并新增：

- runtimeSystem：手机productModel、osFullName、sdkApiVersion；读取失败时保留error。
- 每档labelMismatchCaseIds、driftCaseIds、worstCaseId。
- cases：稳定的参考样本编号、参考文本、有效token数、参考/NPU五个logits、标签差异、距分类边界的logit余量。
- 出错或数值超限样本的64个inputIds和attentionMask，便于与CPU的实际输入对照。
- 即使数值自检失败，也保留已验证的模型字节数状态和实际HIAI_F设备；中途异常保留失败样本及之前的成功结果。

HiLog按Tag PermissionNPU过滤，重点查看[PNPU][DIFF]：仅记录标签发生变化的参考样本，以及最多一条额外的最差数值偏差；其他逐样本数据留在JSON，普通用户输入不写入日志。

下一步先比对手机失败样本的token、logits和标签余量；若输入一致，再测试LayerNorm的更高精度或稳定表达式，并重新验证每档原生BMM/Softmax绑定、CPUCL=0和手机数值。1.1.2包含上面的候选OMC，尚未得到手机复测结果；不能把主机33/33当作真机通过。
